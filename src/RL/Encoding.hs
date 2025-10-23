{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MagicHash #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE Strict #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}

module RL.Encoding where

import Common
import Data.Foldable qualified as F
import Data.HashSet qualified as HS
import Data.Hashable (Hashable)
import Data.List qualified
import Data.Maybe (catMaybes, mapMaybe)
import Data.TypeNums (KnownInt, KnownNat, Nat, TInt (..), intVal, intVal', type (*), type (+), type (-))
import Data.Vector.Sized qualified as VS
import Debug.Trace qualified as DT
import GHC.Exts (Proxy#, proxy#)
import GreedyParser
import Internal.MultiSet qualified as MS
import Musicology.Pitch
import PVGrammar (Edge, Edges (Edges), Freeze (FreezeOp), InnerEdge, Notes (Notes), PVAnalysis, PVLeftmost, Split, Spread)
import PVGrammar.Generate (derivationPlayerPV)
import PVGrammar.Parse (protoVoiceEvaluator, pvThaw)
import RL.Common
import RL.ModelTypes
import Torch qualified as T
import Torch.Typed qualified as TT

-- Utilities
-- =========

-- Stackable class

class Stackable a where
  type Stacked a (n :: Nat)
  stack :: (KnownNat n) => VS.Vector n a -> Stacked a n

class Batchable a where
  type Batched a
  addBatchDim :: a -> Batched a

instance Batchable (TT.Tensor dev dtype shape) where
  type Batched (TT.Tensor dev dtype shape) = TT.Tensor dev dtype (1 : shape)
  addBatchDim = TT.unsqueeze @0

-- Masked Maybe
-- ------------

data QMaybe (batchShape :: [Nat]) a = QMaybe
  { qmMask :: QTensor batchShape
  , qmContent :: a
  }
  deriving (Show)

qNothing
  :: ( TT.TensorOptions batchShape QDType QDevice
     )
  => a
  -> QMaybe batchShape a
qNothing = QMaybe TT.zeros

qJust
  :: (TT.TensorOptions batchShape QDType QDevice)
  => a
  -> QMaybe batchShape a
qJust = QMaybe TT.ones

instance (Stackable a) => Stackable (QMaybe batchShape a) where
  type Stacked (QMaybe batchShape a) n = QMaybe (n ': batchShape) (Stacked a n)
  stack ms = QMaybe masks contents
   where
    masks = TT.vecStack @0 $ qmMask <$> ms
    contents = stack $ qmContent <$> ms

instance (Batchable a) => Batchable (QMaybe shape a) where
  type Batched (QMaybe shape a) = QMaybe (1 : shape) (Batched a)
  addBatchDim (QMaybe mask content) = QMaybe (TT.unsqueeze @0 mask) (addBatchDim content)

-- Masked List
-- -----------

data QBoundedList (dtype :: TT.DType) (maxLen :: Nat) (batchShape :: [Nat]) (innerShape :: [Nat])
  = QBoundedList
  { qlMask :: QTensor (batchShape TT.++ '[maxLen])
  , qlContent :: TT.Tensor QDevice dtype (batchShape TT.++ '[maxLen] TT.++ innerShape)
  }
  deriving (Show)

qBoundedList
  :: forall dtype maxLen innerShape
   . ( KnownNat maxLen
     , TT.KnownShape innerShape
     , TT.TensorOptions innerShape QDType QDevice
     , TT.TensorOptions innerShape dtype QDevice
     )
  => [TT.Tensor QDevice dtype innerShape]
  -> QBoundedList dtype maxLen '[] innerShape
qBoundedList [] = QBoundedList TT.zeros TT.zeros
qBoundedList lst = QBoundedList (TT.UnsafeMkTensor mask) (TT.UnsafeMkTensor paddedContent)
 where
  maxLen = TT.natValI @maxLen
  content = T.stack (T.Dim 0) $ take maxLen $ TT.toDynamic <$> lst
  len = min maxLen $ length lst
  padLen = maxLen - len
  innerShape = TT.shapeVal @innerShape
  -- padSpec: two numbers per dim for pre and post padding, respectively
  -- here: list dim (only post) + inner dims (no padding)
  padSpec = replicate (2 * length innerShape) 0 ++ [0, padLen]
  paddedContent = T.constantPadNd1d padSpec 0 content
  mask = T.cat (T.Dim 0) [T.ones [len] opts, T.zeros [padLen] opts]

instance Stackable (QBoundedList dtype maxLen batchShape innerShape) where
  type
    Stacked (QBoundedList dtype maxLen batchShape innerShape) n =
      QBoundedList dtype maxLen (n ': batchShape) innerShape
  stack xs = QBoundedList masks contents
   where
    masks = TT.vecStack @0 $ qlMask <$> xs
    contents = TT.vecStack @0 $ qlContent <$> xs

instance Batchable (QBoundedList dtype maxLen batchShape innerShape) where
  type
    Batched (QBoundedList dtype maxLen batchShape innerShape) =
      QBoundedList dtype maxLen (1 : batchShape) innerShape
  addBatchDim (QBoundedList mask content) =
    QBoundedList (TT.unsqueeze @0 mask) (TT.unsqueeze @0 content)

-- Tagged StartStop
-- ----------------

data QStartStop (batchShape :: [Nat]) a = QStartStop
  { qssTag :: TT.Tensor QDevice TT.Int64 batchShape
  , qssContent :: a
  }
  deriving (Show)

qInner :: (TT.TensorOptions batchShape TT.Int64 QDevice) => a -> QStartStop batchShape a
qInner = QStartStop (TT.full (1 :: Int))

qStart :: (TT.TensorOptions batchShape TT.Int64 QDevice) => a -> QStartStop batchShape a
qStart = QStartStop (TT.full (0 :: Int))

qStop :: (TT.TensorOptions batchShape TT.Int64 QDevice) => a -> QStartStop batchShape a
qStop = QStartStop (TT.full (2 :: Int))

qStartStop
  :: (TT.TensorOptions batchShape TT.Int64 QDevice)
  => (a -> b)
  -> b
  -> StartStop a
  -> QStartStop batchShape b
qStartStop f def val = case val of
  Start -> qStart def
  Stop -> qStop def
  Inner x -> qInner $ f x

instance (Stackable a) => Stackable (QStartStop batchShape a) where
  type Stacked (QStartStop batchShape a) n = QStartStop (n ': batchShape) (Stacked a n)
  stack xs = QStartStop tags contents
   where
    tags = TT.vecStack @0 $ qssTag <$> xs
    contents = stack $ qssContent <$> xs

instance (Batchable a) => Batchable (QStartStop shape a) where
  type Batched (QStartStop shape a) = QStartStop (1 : shape) (Batched a)
  addBatchDim (QStartStop tag content) =
    QStartStop (TT.unsqueeze @0 tag) (addBatchDim content)

-- Slice Encoding
-- ==============

-- types of slice encodings
-- ------------------------

newtype SliceEncodingSparse batchShape = SliceEncodingSparse
  {getSliceEncodingSparse :: QBoundedList TT.Int64 MaxPitches batchShape '[2]}
  deriving (Show)

instance Stackable (SliceEncodingSparse batchShape) where
  type Stacked (SliceEncodingSparse batchShape) n = SliceEncodingSparse (n ': batchShape)
  stack slices = SliceEncodingSparse $ stack $ getSliceEncodingSparse <$> slices

instance Batchable (SliceEncodingSparse shape) where
  type Batched (SliceEncodingSparse shape) = SliceEncodingSparse (1 ': shape)
  addBatchDim (SliceEncodingSparse slice) = SliceEncodingSparse $ addBatchDim slice

newtype SliceEncodingDense batchShape = SliceEncodingDense
  {getSliceEncodingDense :: QBoundedList QDType MaxPitches batchShape (1 : PShape)}
  deriving (Show)

instance Stackable (SliceEncodingDense batchShape) where
  type Stacked (SliceEncodingDense batchShape) n = SliceEncodingDense (n ': batchShape)
  stack slices = SliceEncodingDense $ stack $ getSliceEncodingDense <$> slices

instance Batchable (SliceEncodingDense shape) where
  type Batched (SliceEncodingDense shape) = SliceEncodingDense (1 ': shape)
  addBatchDim (SliceEncodingDense slice) = SliceEncodingDense $ addBatchDim slice

-- choose slice encoding type:
-- ---------------------------

type SliceEncoding = SliceEncodingDense

getSlice
  :: forall batchShape
   . ( TT.KnownShape batchShape
     )
  => SliceEncoding batchShape
  -> QBoundedList QDType MaxPitches batchShape (1 : PShape)
getSlice = getSliceEncodingDense -- . sliceIndex2OneHot

encodePitches
  :: [SPitch]
  -> SliceEncoding '[]
encodePitches = pitchesOneHots

sliceIndex2OneHot
  :: forall batchShape
   . ( TT.KnownShape batchShape
     )
  => SliceEncodingSparse batchShape
  -> SliceEncodingDense batchShape
sliceIndex2OneHot (SliceEncodingSparse (QBoundedList mask values)) =
  SliceEncodingDense $ QBoundedList mask values'
 where
  fifthSize = TT.natValI @FifthSize
  octaveSize = TT.natValI @OctaveSize
  shape = TT.shapeVal @batchShape
  hotF = T.toType qDType $ T.oneHot fifthSize $ T.select (-1) 0 $ TT.toDynamic values
  hotO = T.toType qDType $ T.oneHot octaveSize $ T.select (-1) 1 $ TT.toDynamic values
  outer = T.einsum "...i,...j->...ij" [hotF, hotO] [1, 0]
  values' = TT.UnsafeMkTensor $ T.unsqueeze (T.Dim (-3)) outer

-- slice variants
-- --------------

pitch2index
  :: SPitch
  -> [Int]
pitch2index p =
  [ clamp fifthSize (fifths p - fifthLow)
  , clamp octaveSize (octaves p - octaveLow)
  ]
 where
  clamp m i = max 0 $ min m i
  fifthLow = fromIntegral $ intVal' @FifthLow proxy#
  octaveLow = fromIntegral $ intVal' @OctaveLow proxy#
  fifthSize = TT.natValI @FifthSize
  octaveSize = TT.natValI @OctaveSize

pitchesMultiHot
  :: HS.HashSet SPitch
  -> QTensor PShape
pitchesMultiHot ps = TT.UnsafeMkTensor out
 where
  out =
    if HS.null ps
      then zeros
      else T.indexPut True indices values zeros
  ~indices = T.asTensor <$> Data.List.transpose (pitch2index <$> F.toList ps)
  values = T.ones [F.length ps] opts
  zeros = T.zeros dims opts
  fifthSize = TT.natValI @FifthSize
  octaveSize = TT.natValI @OctaveSize
  dims = [fifthSize, octaveSize]

pitchesOneHots
  :: [SPitch]
  -> SliceEncodingDense '[]
pitchesOneHots [] = SliceEncodingDense $ QBoundedList TT.zeros TT.zeros
pitchesOneHots ps = SliceEncodingDense $ QBoundedList mask (TT.reshape out)
 where
  pitches = take maxPitches ps
  n = length pitches
  maxPitches = TT.natValI @MaxPitches
  mkIndex i pitch = i : pitch2index pitch
  indices = T.asTensor <$> Data.List.transpose (zipWith mkIndex [0 ..] pitches)
  values = T.ones [n] opts
  zeros :: QTensor (MaxPitches ': PShape)
  zeros = TT.zeros
  out :: QTensor (MaxPitches : PShape)
  out = TT.UnsafeMkTensor $ T.indexPut True indices values $ TT.toDynamic zeros
  mask :: QTensor '[MaxPitches]
  mask = TT.UnsafeMkTensor $ T.cat (T.Dim 0) [values, T.zeros [maxPitches - n] opts]

pitchesTokens
  :: [SPitch]
  -> QBoundedList QDType MaxPitches '[] '[PSize] -- SliceEncoding '[]
pitchesTokens ps = qBoundedList (mkToken <$> ps)
 where
  -- todo: batch oneHot
  mkToken p =
    TT.UnsafeMkTensor $ toOpts $ T.cat (T.Dim 0) [T.oneHot fifthSize f, T.oneHot octaveSize o]
   where
    f = T.asTensor' (fifths p - fifthLow) $ T.withDType T.Int64 opts
    o = T.asTensor' (octaves p - octaveLow) $ T.withDType T.Int64 opts
  fifthLow = fromIntegral $ intVal' @FifthLow proxy#
  octaveLow = fromIntegral $ intVal' @OctaveLow proxy#
  fifthSize = TT.natValI @FifthSize
  octaveSize = TT.natValI @OctaveSize

pitchesIndices
  :: [SPitch]
  -> SliceEncodingSparse '[]
pitchesIndices ps = SliceEncodingSparse $ qBoundedList (mkToken <$> ps)
 where
  mkIndex = pitch2index
  mkToken p = TT.UnsafeMkTensor $ T.asTensor' (mkIndex p) $ T.withDType T.Int64 opts

encodeSlice
  :: Notes SPitch
  -> SliceEncoding '[]
-- encodeSlice = encodeSliceIndices
encodeSlice (Notes notes) = encodePitches $ MS.toList notes

emptySlice
  :: SliceEncoding '[]
emptySlice = encodePitches []

-- Transition Encoding
-- ===================

data TransitionEncoding batchShape = TransitionEncoding
  { trencPassing :: QBoundedList QDType MaxEdges batchShape (2 ': PShape)
  , trencInner :: QBoundedList QDType MaxEdges batchShape (2 ': PShape)
  , trencLeft :: SliceEncoding batchShape
  , trencRight :: SliceEncoding batchShape
  , trencRoot :: QTensor batchShape
  }
  deriving (Show)

instance Stackable (TransitionEncoding batchShape) where
  type
    Stacked (TransitionEncoding batchShape) n =
      TransitionEncoding (n ': batchShape)
  stack xs = TransitionEncoding passing inner left right root
   where
    passing = stack $ trencPassing <$> xs
    inner = stack $ trencInner <$> xs
    left = stack $ trencLeft <$> xs
    right = stack $ trencRight <$> xs
    root = TT.vecStack @0 $ trencRoot <$> xs

instance Batchable (TransitionEncoding shape) where
  type Batched (TransitionEncoding shape) = TransitionEncoding (1 : shape)
  addBatchDim (TransitionEncoding p i l r rt) =
    TransitionEncoding
      (addBatchDim p)
      (addBatchDim i)
      (addBatchDim l)
      (addBatchDim r)
      (TT.unsqueeze @0 rt)

edgesMultiHot
  :: HS.HashSet (InnerEdge SPitch)
  -> QTensor (EShape')
edgesMultiHot es = TT.UnsafeMkTensor out
 where
  out =
    if HS.null es
      then zeros
      else T.indexPut True indexTensors values zeros
  edge2index (p1, p2) =
    pitch2index p1
      ++ pitch2index p2
  indices = edge2index <$> F.toList es
  ~indexTensors = T.asTensor <$> Data.List.transpose indices
  values = T.ones [F.length es] opts
  zeros = T.zeros dims opts
  fifthSize = TT.natValI @FifthSize
  octaveSize = TT.natValI @OctaveSize
  dims = [fifthSize, octaveSize, fifthSize, octaveSize]

edgesOneHots
  :: [InnerEdge SPitch]
  -> QBoundedList QDType MaxEdges '[] (2 ': PShape)
edgesOneHots es = QBoundedList mask $ TT.cat @1 (hots1 TT.:. hots2 TT.:. TT.HNil)
 where
  SliceEncodingDense (QBoundedList mask hots1) = pitchesOneHots $ fst <$> es
  SliceEncodingDense (QBoundedList _ hots2) = pitchesOneHots $ snd <$> es

edgesTokens
  :: [InnerEdge SPitch]
  -> QBoundedList QDType MaxEdges '[] '[ESize] -- Maybe (QTensor EShape)
edgesTokens es = qBoundedList (mkToken <$> es)
 where
  -- todo: batch oneHot
  mkToken (p1, p2) =
    TT.UnsafeMkTensor $!
      toOpts $
        T.cat
          (T.Dim 0)
          [ T.oneHot fifthSize f1
          , T.oneHot octaveSize o1
          , T.oneHot fifthSize f2
          , T.oneHot octaveSize o2
          ]
   where
    toIndex i = T.asTensor' i $ T.withDType T.Int64 opts
    f1 = toIndex $ fifths p1 - fifthLow
    o1 = toIndex $ octaves p1 - octaveLow
    f2 = toIndex $ fifths p2 - fifthLow
    o2 = toIndex $ octaves p2 - octaveLow
  fifthLow = fromIntegral $ intVal' @FifthLow proxy#
  octaveLow = fromIntegral $ intVal' @OctaveLow proxy#
  fifthSize = TT.natValI @FifthSize
  octaveSize = TT.natValI @OctaveSize

encodeTransition
  :: Edges SPitch
  -> TransitionEncoding '[]
encodeTransition (Edges reg pass) =
  TransitionEncoding
    { trencPassing = edgesOneHots $ MS.toList pass
    , -- , trencPassing = edgesOneHot $ MS.toSet pass
      trencInner = edgesOneHots $ getEdges getInner
    , -- , trencInner = edgesOneHot $ HS.fromList $ getEdges getInner
      trencLeft = pitchesOneHots $ getEdges getLeft
    , trencRight = pitchesOneHots $ getEdges getRight
    , trencRoot = if HS.member (Start, Stop) reg then 1 else 0
    }
 where
  regulars = HS.toList reg
  getEdges :: (Hashable a) => (Edge SPitch -> Maybe a) -> [a]
  getEdges f = mapMaybe f regulars
  getInner (Inner a, Inner b) = Just (a, b)
  getInner _ = Nothing
  getLeft (Start, Inner b) = Just b
  getLeft _ = Nothing
  getRight (Inner a, Stop) = Just a
  getRight _ = Nothing

emptyTransition
  :: TransitionEncoding '[]
emptyTransition = encodeTransition $ Edges HS.empty MS.empty

-- Action Encoding
-- ---------------

data ActionTop batchShape = ActionTop
  { atopSl :: !(QStartStop batchShape (SliceEncoding batchShape))
  , atopT1 :: !(TransitionEncoding batchShape)
  , atopSm :: !(QMaybe batchShape (SliceEncoding batchShape))
  , atopT2 :: !(QMaybe batchShape (TransitionEncoding batchShape))
  , atopSr :: !(QStartStop batchShape (SliceEncoding batchShape))
  }
  deriving (Show)

instance Stackable (ActionTop batchShape) where
  type Stacked (ActionTop batchShape) n = ActionTop (n ': batchShape)
  stack xs = ActionTop sl t1 sm t2 sr
   where
    sl = stack $ atopSl <$> xs
    t1 = stack $ atopT1 <$> xs
    sm = stack $ atopSm <$> xs
    t2 = stack $ atopT2 <$> xs
    sr = stack $ atopSr <$> xs

instance Batchable (ActionTop shape) where
  type Batched (ActionTop shape) = ActionTop (1 : shape)
  addBatchDim (ActionTop sl t1 sm t2 sr) =
    ActionTop
      (addBatchDim sl)
      (addBatchDim t1)
      (addBatchDim sm)
      (addBatchDim t2)
      (addBatchDim sr)

data ActionEncoding batchShape = ActionEncoding
  { actionEncodingTop :: !(ActionTop batchShape) -- (Either (SingleTop batchShape) (DoubleTop batchShape))
  , actionEncodingOp :: !(TT.Tensor QDevice 'TT.Int64 batchShape) -- !(Leftmost () () ())
  }
  deriving (Show)

instance Stackable (ActionEncoding batchShape) where
  type Stacked (ActionEncoding batchShape) n = ActionEncoding (n ': batchShape)
  stack xs = ActionEncoding tops ops
   where
    tops = stack $ actionEncodingTop <$> xs
    ops = TT.vecStack @0 $ actionEncodingOp <$> xs

instance Batchable (ActionEncoding shape) where
  type Batched (ActionEncoding shape) = ActionEncoding (1 : shape)
  addBatchDim (ActionEncoding top op) = ActionEncoding (addBatchDim top) (TT.unsqueeze @0 op)

encodePVAction
  :: PVAction
  -> ActionEncoding '[]
encodePVAction (Left (ActionSingle top action)) = ActionEncoding encTop encAction
 where
  (SingleParent sl t sr) = top
  encTop =
    ActionTop
      (qStartStop (encodeSlice) (emptySlice) sl)
      (encodeTransition t)
      (qNothing $ emptySlice)
      (qNothing $ emptyTransition)
      (qStartStop encodeSlice emptySlice sr)
  encAction = case action of
    LMSingleFreeze FreezeOp -> TT.full (0 :: Int) --  LMFreezeOnly ()
    LMSingleSplit _split -> TT.full (1 :: Int) -- LMSplitOnly ()
encodePVAction (Right (ActionDouble top action)) = ActionEncoding encTop encAction
 where
  (DoubleParent sl t1 sm t2 sr) = top
  encTop =
    ActionTop
      (qStartStop encodeSlice emptySlice sl)
      (encodeTransition t1)
      (qJust $ encodeSlice sm)
      (qJust $ encodeTransition t2)
      (qStartStop encodeSlice emptySlice sr)

  encAction = case action of
    LMDoubleFreezeLeft FreezeOp -> TT.full (2 :: Int) -- LMFreezeLeft ()
    LMDoubleSpread spread -> TT.full (3 :: Int) -- LMSpread ()
    LMDoubleSplitLeft split -> TT.full (4 :: Int) -- LMSplitLeft ()
    LMDoubleSplitRight split -> TT.full (5 :: Int) -- LMSplitRight ()

-- State Encoding
-- --------------

data StateEncoding = StateEncoding
  { stateEncodingMid :: !(StartStop (SliceEncoding '[]))
  , stateEncodingFrozen :: ![(TransitionEncoding '[], StartStop (SliceEncoding '[]))]
  , stateEncodingOpen :: ![(TransitionEncoding '[], StartStop (SliceEncoding '[]))]
  }
  deriving (Show)

type PVState t =
  GreedyState
    (Edges SPitch)
    (t (Edge SPitch))
    (Notes SPitch)
    (PVLeftmost SPitch)

getFrozen
  :: forall t
   . (Foldable t)
  => Path (Maybe (t (Edge SPitch))) (Notes SPitch)
  -> [(TransitionEncoding '[], StartStop (SliceEncoding '[]))]
getFrozen frozen = encodePair <$> pathTake 3 Inner Start frozen
 where
  encodePair (tr, slc) = (encodeTransition $ pvThaw tr, encodeSlice <$> slc)

-- case frozen of
-- PathEnd tr -> (encodeTransition $ pvThaw tr, Start)
-- Path tr slc _ ->
--   (encodeTransition $ pvThaw tr, Inner $ encodeSlice slc)

getOpen
  :: Path (Edges SPitch) (Notes SPitch)
  -> [(TransitionEncoding '[], StartStop (SliceEncoding '[]))]
getOpen open = encodePair <$> pathTake 3 Inner Stop open
 where
  encodePair (tr, slc) = (encodeTransition tr, encodeSlice <$> slc)

encodePVState
  :: (Foldable t)
  => PVState t
  -> StateEncoding
encodePVState (GSFrozen frozen) = StateEncoding Stop (getFrozen frozen) []
encodePVState (GSOpen open _) = StateEncoding Start [] (getOpen open)
encodePVState (GSSemiOpen frozen mid open _) =
  StateEncoding (Inner $ encodeSlice mid) (getFrozen frozen) (getOpen open)

-- Step Encoding
-- -------------

data QEncoding batchShape = QEncoding
  { qActionEncoding :: !(ActionEncoding batchShape)
  , qStateEncoding :: !(StateEncoding)
  }
  deriving (Show)

instance Batchable (QEncoding shape) where
  type Batched (QEncoding shape) = QEncoding (1 : shape)
  addBatchDim (QEncoding ac st) = QEncoding (addBatchDim ac) st

encodeStep
  :: (Foldable t)
  => PVState t
  -> PVAction
  -> QEncoding '[]
encodeStep state action =
  QEncoding
    (encodePVAction action)
    (encodePVState state)

withBatchedEncoding
  :: (Foldable t)
  => PVState t
  -> [PVAction]
  -> (forall n. (KnownNat n) => QEncoding '[n] -> r)
  -> r
withBatchedEncoding state actions f =
  VS.withSizedList aEncs $ \aEncs' -> f $ QEncoding (stack aEncs') sEnc
 where
  aEncs = encodePVAction <$> actions
  sEnc = encodePVState state
