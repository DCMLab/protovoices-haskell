{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MagicHash #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE Strict #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}

module RL.Encoding where

import Common
import GreedyParser
import Internal.MultiSet qualified as MS
import PVGrammar (Edge, Edges (Edges), Freeze (FreezeOp), InnerEdge, Note (..), Notes (Notes), PVAnalysis, PVLeftmost, Split, Spread)
import PVGrammar.Generate (derivationPlayerPV)
import PVGrammar.Parse (protoVoiceEvaluator, pvThaw)
import RL.ModelTypes

import Control.DeepSeq
import Data.Foldable qualified as F
import Data.HashSet qualified as HS
import Data.Hashable (Hashable)
import Data.List qualified
import Data.List.NonEmpty (NonEmpty (..))
import Data.Maybe (catMaybes, mapMaybe)
import Data.Proxy (Proxy (..))
import Data.Type.Equality ((:~:) (..))
import Data.TypeNums (KnownInt, KnownNat, Nat, TInt (..), intVal, intVal', type (*), type (+), type (-), type (>=))
import Data.Vector qualified as V
import Data.Vector.Generic.Sized.Internal qualified as VSU
import Data.Vector.Sized qualified as VS
import Debug.Trace qualified as DT
import GHC.Exts (Proxy#, proxy#)
import GHC.Generics
import Musicology.Pitch
import Torch qualified as T
import Torch.Lens qualified as T
import Torch.Typed qualified as TT
import Unsafe.Coerce (unsafeCoerce)

-- Utilities
-- =========

-- -- Tensorized: get all tensors out of a data structure
-- -- ---------------------------------------------------

-- class GTensorized f where
--   gFlattenTensors :: forall a. f a -> [T.Tensor]

-- instance GTensorized U1 where
--   gFlattenTensors U1 = []

-- instance (GTensorized f, GTensorized g) => GTensorized (f :+: g) where
--   gFlattenTensors (L1 x) = gFlattenTensors x
--   gFlattenTensors (R1 x) = gFlattenTensors x

-- instance (GTensorized f, GTensorized g) => GTensorized (f :*: g) where
--   gFlattenTensors (x :*: y) = gFlattenTensors x ++ gFlattenTensors y

-- instance (Tensorized c) => GTensorized (K1 i c) where
--   gFlattenTensors (K1 x) = flattenTensors x

-- instance (GTensorized f) => GTensorized (M1 i t f) where
--   gFlattenTensors (M1 x) = gFlattenTensors x

-- class Tensorized a where
--   flattenTensors :: a -> [T.Tensor]
--   default flattenTensors :: (Generic a, GTensorized (Rep a)) => a -> [T.Tensor]
--   flattenTensors f = gFlattenTensors (from f)

-- instance Tensorized (TT.Tensor dev dtype shape) where
--   flattenTensors t = [TT.toDynamic t]

-- instance Tensorized (TT.Parameter dev dtype shape) where
--   flattenTensors t = [TT.toDynamic $ TT.toDependent t]

-- instance (Tensorized a) => Tensorized (StartStop a)

-- instance (Tensorized a, Tensorized b) => Tensorized (a, b)

-- instance (Tensorized a) => Tensorized [a]

-- instance Tensorized Double where
--   flattenTensors _ = []

-- instance Tensorized (TT.Conv2d a b c d dtype dev)

-- instance Tensorized (TT.Linear i o dtype device)

-- instance Tensorized (TT.LayerNorm shape dtype device)

-- Stackable and Batchable class
-- -----------------------------

class Stackable a where
  type Stacked a (n :: Nat)
  stack :: (KnownNat n, KnownNat (1 + n)) => VS.Vector (1 + n) a -> Stacked a (1 + n)

stackUnsafe :: (Stackable a) => [a] -> Stacked a FakeSize
stackUnsafe things = stack $ VSU.Vector $ V.fromList things

class Batchable a where
  type Batched a
  addBatchDim :: a -> Batched a

instance Batchable (TT.Tensor dev dtype shape) where
  type Batched (TT.Tensor dev dtype shape) = TT.Tensor dev dtype (1 : shape)
  addBatchDim = TT.unsqueeze @0

-- Masked Maybe
-- ------------

data QMaybe dev (batchShape :: [Nat]) a = QMaybe
  { qmMask :: QTensor dev batchShape
  , qmContent :: a
  }
  deriving (Show, Generic, NFData)

qNothing
  :: ( TT.TensorOptions batchShape QDType dev
     )
  => a
  -> QMaybe dev batchShape a
qNothing = QMaybe TT.zeros

qJust
  :: (TT.TensorOptions batchShape QDType dev)
  => a
  -> QMaybe dev batchShape a
qJust = QMaybe TT.ones

instance (Stackable a) => Stackable (QMaybe dev batchShape a) where
  type Stacked (QMaybe dev batchShape a) n = QMaybe dev (n ': batchShape) (Stacked a n)
  stack ms = QMaybe masks contents
   where
    masks = TT.vecStack @0 $ qmMask <$> ms
    contents = stack $ qmContent <$> ms

instance (Batchable a) => Batchable (QMaybe dev shape a) where
  type Batched (QMaybe dev shape a) = QMaybe dev (1 : shape) (Batched a)
  addBatchDim (QMaybe mask content) = QMaybe (TT.unsqueeze @0 mask) (addBatchDim content)

-- instance (T.HasTypes a T.Tensor) => T.HasTypes (QMaybe shape a) T.Tensor

-- Masked List
-- -----------

data QBoundedList dev (dtype :: TT.DType) (maxLen :: Nat) (batchShape :: [Nat]) (innerShape :: [Nat])
  = QBoundedList
  { qlMask :: QTensor dev (batchShape TT.++ '[maxLen])
  , qlContent :: TT.Tensor dev dtype (batchShape TT.++ '[maxLen] TT.++ innerShape)
  }
  deriving (Show, Generic, NFData)

qBoundedList
  :: forall dev dtype maxLen innerShape
   . ( KnownNat maxLen
     , TT.KnownDevice dev
     , TT.KnownShape innerShape
     , TT.TensorOptions innerShape QDType dev
     , TT.TensorOptions innerShape dtype dev
     )
  => [TT.Tensor dev dtype innerShape]
  -> QBoundedList dev dtype maxLen '[] innerShape
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
  mask = T.cat (T.Dim 0) [T.ones [len] $ opts @dev, T.zeros [padLen] $ opts @dev]

instance Stackable (QBoundedList dev dtype maxLen batchShape innerShape) where
  type
    Stacked (QBoundedList dev dtype maxLen batchShape innerShape) n =
      QBoundedList dev dtype maxLen (n ': batchShape) innerShape
  stack xs = QBoundedList masks contents
   where
    masks = TT.vecStack @0 $ qlMask <$> xs
    contents = TT.vecStack @0 $ qlContent <$> xs

instance Batchable (QBoundedList dev dtype maxLen batchShape innerShape) where
  type
    Batched (QBoundedList dev dtype maxLen batchShape innerShape) =
      QBoundedList dev dtype maxLen (1 : batchShape) innerShape
  addBatchDim (QBoundedList mask content) =
    QBoundedList (TT.unsqueeze @0 mask) (TT.unsqueeze @0 content)

-- instance T.HasTypes (QBoundedList dtype maxLen batchShape innerShape) T.Tensor

-- Tagged StartStop
-- ----------------

data QStartStop dev (batchShape :: [Nat]) a = QStartStop
  { qssTag :: TT.Tensor dev TT.Int64 batchShape
  , qssContent :: a
  }
  deriving (Show, Generic, NFData)

qInner :: (TT.TensorOptions batchShape TT.Int64 dev) => a -> QStartStop dev batchShape a
qInner = QStartStop (TT.full (1 :: Int))

qStart :: (TT.TensorOptions batchShape TT.Int64 dev) => a -> QStartStop dev batchShape a
qStart = QStartStop (TT.full (0 :: Int))

qStop :: (TT.TensorOptions batchShape TT.Int64 dev) => a -> QStartStop dev batchShape a
qStop = QStartStop (TT.full (2 :: Int))

qStartStop
  :: (TT.TensorOptions batchShape TT.Int64 dev)
  => (a -> b)
  -> b
  -> StartStop a
  -> QStartStop dev batchShape b
qStartStop f def val = case val of
  Start -> qStart def
  Stop -> qStop def
  Inner x -> qInner $ f x

instance (Stackable a) => Stackable (QStartStop dev batchShape a) where
  type Stacked (QStartStop dev batchShape a) n = QStartStop dev (n ': batchShape) (Stacked a n)
  stack xs = QStartStop tags contents
   where
    tags = TT.vecStack @0 $ qssTag <$> xs
    contents = stack $ qssContent <$> xs

instance (Batchable a) => Batchable (QStartStop dev shape a) where
  type Batched (QStartStop dev shape a) = QStartStop dev (1 : shape) (Batched a)
  addBatchDim (QStartStop tag content) =
    QStartStop (TT.unsqueeze @0 tag) (addBatchDim content)

-- instance (T.HasTypes a T.Tensor) => T.HasTypes (QStartStop shape a) T.Tensor

-- Slice Encoding
-- ==============

-- types of slice encodings
-- ------------------------

newtype SliceEncodingSparse dev batchShape = SliceEncodingSparse
  {getSliceEncodingSparse :: QBoundedList dev TT.Int64 MaxPitches batchShape '[2]}
  deriving (Show, Generic)
  deriving newtype (NFData)

instance Stackable (SliceEncodingSparse dev batchShape) where
  type Stacked (SliceEncodingSparse dev batchShape) n = SliceEncodingSparse dev (n ': batchShape)
  stack slices = SliceEncodingSparse $ stack $ getSliceEncodingSparse <$> slices

instance Batchable (SliceEncodingSparse dev shape) where
  type Batched (SliceEncodingSparse dev shape) = SliceEncodingSparse dev (1 ': shape)
  addBatchDim (SliceEncodingSparse slice) = SliceEncodingSparse $ addBatchDim slice

-- instance T.HasTypes (SliceEncodingSparse shape) T.Tensor

newtype SliceEncodingDense dev batchShape = SliceEncodingDense
  {getSliceEncodingDense :: QBoundedList dev QDType MaxPitches batchShape (1 : PShape)}
  deriving (Show, Generic)
  deriving newtype (NFData)

instance Stackable (SliceEncodingDense dev batchShape) where
  type Stacked (SliceEncodingDense dev batchShape) n = SliceEncodingDense dev (n ': batchShape)
  stack slices = SliceEncodingDense $ stack $ getSliceEncodingDense <$> slices

instance Batchable (SliceEncodingDense dev shape) where
  type Batched (SliceEncodingDense dev shape) = SliceEncodingDense dev (1 ': shape)
  addBatchDim (SliceEncodingDense slice) = SliceEncodingDense $ addBatchDim slice

-- instance T.HasTypes (SliceEncodingDense shape) T.Tensor

-- choose slice encoding type:
-- ---------------------------

type SliceEncoding = SliceEncodingDense

getSlice
  :: forall dev batchShape
   . SliceEncoding dev batchShape
  -> QBoundedList dev QDType MaxPitches batchShape (1 : PShape)
getSlice = getSliceEncodingDense -- . sliceIndex2OneHot

encodePitches
  :: (TT.KnownDevice dev)
  => [SPitch]
  -> SliceEncoding dev '[]
encodePitches = pitchesOneHots

sliceIndex2OneHot
  :: forall dev batchShape
   . ( TT.KnownShape batchShape
     )
  => SliceEncodingSparse dev batchShape
  -> SliceEncodingDense dev batchShape
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
  :: forall dev
   . (TT.KnownDevice dev)
  => HS.HashSet SPitch
  -> QTensor dev PShape
pitchesMultiHot ps = TT.UnsafeMkTensor out
 where
  out =
    if HS.null ps
      then zeros
      else T.indexPut True indices values zeros
  ~indices = T.asTensor <$> Data.List.transpose (pitch2index <$> F.toList ps)
  values = T.ones [F.length ps] $ opts @dev
  zeros = T.zeros dims $ opts @dev
  fifthSize = TT.natValI @FifthSize
  octaveSize = TT.natValI @OctaveSize
  dims = [fifthSize, octaveSize]

pitchesOneHots
  :: forall dev
   . (TT.KnownDevice dev)
  => [SPitch]
  -> SliceEncodingDense dev '[]
pitchesOneHots [] = SliceEncodingDense $ QBoundedList TT.zeros TT.zeros
pitchesOneHots ps = SliceEncodingDense $ QBoundedList mask (TT.reshape out)
 where
  pitches = take maxPitches ps
  n = length pitches
  maxPitches = TT.natValI @MaxPitches
  mkIndex i pitch = i : pitch2index pitch
  indices = T.asTensor <$> Data.List.transpose (zipWith mkIndex [0 ..] pitches)
  values = T.ones [n] $ opts @dev
  zeros :: QTensor dev (MaxPitches ': PShape)
  zeros = TT.zeros
  out :: QTensor dev (MaxPitches : PShape)
  out = TT.UnsafeMkTensor $ T.indexPut True indices values $ TT.toDynamic zeros
  mask :: QTensor dev '[MaxPitches]
  mask = TT.UnsafeMkTensor $ T.cat (T.Dim 0) [values, T.zeros [maxPitches - n] $ opts @dev]

pitchesTokens
  :: forall dev
   . (TT.KnownDevice dev)
  => [SPitch]
  -> QBoundedList dev QDType MaxPitches '[] '[PSize] -- SliceEncoding '[]
pitchesTokens ps = qBoundedList (mkToken <$> ps)
 where
  -- todo: batch oneHot
  opts' = T.withDType
  mkToken p =
    TT.UnsafeMkTensor $ T.toDType qDType $ T.cat (T.Dim 0) [T.oneHot fifthSize f, T.oneHot octaveSize o]
   where
    f = T.asTensor' (fifths p - fifthLow) $ T.withDType T.Int64 $ opts @dev
    o = T.asTensor' (octaves p - octaveLow) $ T.withDType T.Int64 $ opts @dev
  fifthLow = fromIntegral $ intVal' @FifthLow proxy#
  octaveLow = fromIntegral $ intVal' @OctaveLow proxy#
  fifthSize = TT.natValI @FifthSize
  octaveSize = TT.natValI @OctaveSize

pitchesIndices
  :: forall dev
   . (TT.KnownDevice dev)
  => [SPitch]
  -> SliceEncodingSparse dev '[]
pitchesIndices ps = SliceEncodingSparse $ qBoundedList (mkToken <$> ps)
 where
  mkIndex = pitch2index
  mkToken p = TT.UnsafeMkTensor $ T.asTensor' (mkIndex p) $ T.withDType T.Int64 $ opts @dev

encodeSlice
  :: (TT.KnownDevice dev)
  => Notes SPitch
  -> SliceEncoding dev '[]
-- encodeSlice = encodeSliceIndices
encodeSlice (Notes notes) = encodePitches $ notePitch <$> HS.toList notes

emptySlice
  :: (TT.KnownDevice dev) => SliceEncoding dev '[]
emptySlice = encodePitches []

-- Transition Encoding
-- ===================

data TransitionEncoding dev batchShape = TransitionEncoding
  { trencPassing :: QBoundedList dev QDType MaxEdges batchShape (2 ': PShape)
  , trencInner :: QBoundedList dev QDType MaxEdges batchShape (2 ': PShape)
  , trencLeft :: SliceEncoding dev batchShape
  , trencRight :: SliceEncoding dev batchShape
  , trencRoot :: QTensor dev batchShape
  }
  deriving (Show, Generic, NFData)

instance Stackable (TransitionEncoding dev batchShape) where
  type
    Stacked (TransitionEncoding dev batchShape) n =
      TransitionEncoding dev (n ': batchShape)
  stack xs = TransitionEncoding passing inner left right root
   where
    passing = stack $ trencPassing <$> xs
    inner = stack $ trencInner <$> xs
    left = stack $ trencLeft <$> xs
    right = stack $ trencRight <$> xs
    root = TT.vecStack @0 $ trencRoot <$> xs

instance Batchable (TransitionEncoding dev shape) where
  type Batched (TransitionEncoding dev shape) = TransitionEncoding dev (1 : shape)
  addBatchDim (TransitionEncoding p i l r rt) =
    TransitionEncoding
      (addBatchDim p)
      (addBatchDim i)
      (addBatchDim l)
      (addBatchDim r)
      (TT.unsqueeze @0 rt)

edgesMultiHot
  :: forall dev
   . (TT.KnownDevice dev)
  => HS.HashSet (InnerEdge SPitch)
  -> QTensor dev EShape'
edgesMultiHot es = TT.UnsafeMkTensor out
 where
  out =
    if HS.null es
      then zeros
      else T.indexPut True indexTensors values zeros
  edge2index (Note p1 _, Note p2 _) =
    pitch2index p1
      ++ pitch2index p2
  indices = edge2index <$> F.toList es
  ~indexTensors = T.asTensor <$> Data.List.transpose indices
  values = T.ones [F.length es] $ opts @dev
  zeros = T.zeros dims $ opts @dev
  fifthSize = TT.natValI @FifthSize
  octaveSize = TT.natValI @OctaveSize
  dims = [fifthSize, octaveSize, fifthSize, octaveSize]

edgesOneHots
  :: forall dev
   . (TT.KnownDevice dev)
  => [InnerEdge SPitch]
  -> QBoundedList dev QDType MaxEdges '[] (2 ': PShape)
edgesOneHots es = QBoundedList mask $ TT.cat @1 (hots1 TT.:. hots2 TT.:. TT.HNil)
 where
  SliceEncodingDense (QBoundedList mask hots1) = pitchesOneHots @dev $ (notePitch . fst) <$> es
  SliceEncodingDense (QBoundedList _ hots2) = pitchesOneHots @dev $ (notePitch . snd) <$> es

edgesTokens
  :: forall dev
   . (TT.KnownDevice dev)
  => [InnerEdge SPitch]
  -> QBoundedList dev QDType MaxEdges '[] '[ESize] -- Maybe (QTensor EShape)
edgesTokens es = qBoundedList (mkToken <$> es)
 where
  -- todo: batch oneHot
  mkToken (Note p1 _, Note p2 _) =
    TT.UnsafeMkTensor $!
      toOpts @dev $
        T.cat
          (T.Dim 0)
          [ T.oneHot fifthSize f1
          , T.oneHot octaveSize o1
          , T.oneHot fifthSize f2
          , T.oneHot octaveSize o2
          ]
   where
    toIndex i = T.asTensor' i $ T.withDType T.Int64 $ opts @dev
    f1 = toIndex $ fifths p1 - fifthLow
    o1 = toIndex $ octaves p1 - octaveLow
    f2 = toIndex $ fifths p2 - fifthLow
    o2 = toIndex $ octaves p2 - octaveLow
  fifthLow = fromIntegral $ intVal' @FifthLow proxy#
  octaveLow = fromIntegral $ intVal' @OctaveLow proxy#
  fifthSize = TT.natValI @FifthSize
  octaveSize = TT.natValI @OctaveSize

encodeTransition
  :: (TT.KnownDevice dev)
  => Edges SPitch
  -> TransitionEncoding dev '[]
encodeTransition (Edges reg pass) =
  TransitionEncoding
    { trencPassing = edgesOneHots $ MS.toList pass
    , -- , trencPassing = edgesOneHot $ MS.toSet pass
      trencInner = edgesOneHots $ getEdges getInner
    , -- , trencInner = edgesOneHot $ HS.fromList $ getEdges getInner
      trencLeft = pitchesOneHots $ notePitch <$> getEdges getLeft
    , trencRight = pitchesOneHots $ notePitch <$> getEdges getRight
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
  :: (TT.KnownDevice dev) => TransitionEncoding dev '[]
emptyTransition = encodeTransition $ Edges HS.empty MS.empty

-- Action Encoding
-- ---------------

data ActionTop dev batchShape = ActionTop
  { atopSl :: !(QStartStop dev batchShape (SliceEncoding dev batchShape))
  , atopT1 :: !(TransitionEncoding dev batchShape)
  , atopSm :: !(QMaybe dev batchShape (SliceEncoding dev batchShape))
  , atopT2 :: !(QMaybe dev batchShape (TransitionEncoding dev batchShape))
  , atopSr :: !(QStartStop dev batchShape (SliceEncoding dev batchShape))
  }
  deriving (Show, Generic, NFData)

instance Stackable (ActionTop dev batchShape) where
  type Stacked (ActionTop dev batchShape) n = ActionTop dev (n ': batchShape)
  stack xs = ActionTop sl t1 sm t2 sr
   where
    sl = stack $ atopSl <$> xs
    t1 = stack $ atopT1 <$> xs
    sm = stack $ atopSm <$> xs
    t2 = stack $ atopT2 <$> xs
    sr = stack $ atopSr <$> xs

instance Batchable (ActionTop dev shape) where
  type Batched (ActionTop dev shape) = ActionTop dev (1 : shape)
  addBatchDim (ActionTop sl t1 sm t2 sr) =
    ActionTop
      (addBatchDim sl)
      (addBatchDim t1)
      (addBatchDim sm)
      (addBatchDim t2)
      (addBatchDim sr)

data ActionEncoding dev batchShape = ActionEncoding
  { actionEncodingTop :: !(ActionTop dev batchShape) -- (Either (SingleTop batchShape) (DoubleTop batchShape))
  , actionEncodingOp :: !(TT.Tensor dev 'TT.Int64 batchShape) -- !(Leftmost () () ())
  }
  deriving (Show, Generic, NFData)

instance Stackable (ActionEncoding dev batchShape) where
  type Stacked (ActionEncoding dev batchShape) n = ActionEncoding dev (n ': batchShape)
  stack xs = ActionEncoding tops ops
   where
    tops = stack $ actionEncodingTop <$> xs
    ops = TT.vecStack @0 $ actionEncodingOp <$> xs

instance Batchable (ActionEncoding dev shape) where
  type Batched (ActionEncoding dev shape) = ActionEncoding dev (1 : shape)
  addBatchDim (ActionEncoding top op) = ActionEncoding (addBatchDim top) (TT.unsqueeze @0 op)

-- instance T.HasTypes (ActionEncoding shape) T.Tensor

encodePVAction
  :: (TT.KnownDevice dev)
  => PVAction
  -> ActionEncoding dev '[]
encodePVAction (Left (ActionSingle top action)) = ActionEncoding encTop encAction
 where
  (SingleParent sl t sr) = top
  encTop =
    ActionTop
      (qStartStop encodeSlice emptySlice sl)
      (encodeTransition t)
      (qNothing $ emptySlice)
      (qNothing $ emptyTransition)
      (qStartStop encodeSlice emptySlice sr)
  encAction = case action of
    LMSingleFreeze _freeze -> TT.full (0 :: Int) --  LMFreezeOnly ()
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
    LMDoubleFreezeLeft _freeze -> TT.full (2 :: Int) -- LMFreezeLeft ()
    LMDoubleSpread _spread -> TT.full (3 :: Int) -- LMSpread ()
    LMDoubleSplitLeft _split -> TT.full (4 :: Int) -- LMSplitLeft ()
    LMDoubleSplitRight _split -> TT.full (5 :: Int) -- LMSplitRight ()

-- State Encoding
-- --------------

data StateEncoding dev = StateEncoding
  { stateEncodingMid :: !(QStartStop dev '[] (SliceEncoding dev '[]))
  , stateEncodingFrozen :: !(QMaybe dev '[] (TransitionEncoding dev '[FakeSize], QStartStop dev '[FakeSize] (SliceEncoding dev '[FakeSize])))
  , stateEncodingOpen :: !(QMaybe dev '[] (TransitionEncoding dev '[FakeSize], QStartStop dev '[FakeSize] (SliceEncoding dev '[FakeSize])))
  }
  deriving (Show, Generic, NFData)

-- type PVState t =
--   GreedyState
--     (Edges SPitch)
--     (t (Edge SPitch))
--     (Notes SPitch)
--     (PVLeftmost SPitch)

getFrozen
  :: forall dev t
   . (Foldable t, TT.KnownDevice dev)
  => Path (Maybe (t (Edge SPitch))) (Notes SPitch)
  -> (TransitionEncoding dev '[FakeSize], QStartStop dev '[FakeSize] (SliceEncoding dev '[FakeSize]))
getFrozen frozen = (stackUnsafe trEncs, stackUnsafe slcEncs)
 where
  (trs, slcs) = unzip $ pathTake 3 Inner Start frozen
  trEncs = encodeTransition . pvThaw <$> trs
  slcEncs = qStartStop encodeSlice emptySlice <$> slcs

getOpen
  :: (TT.KnownDevice dev)
  => Path (Edges SPitch) (Notes SPitch)
  -> (TransitionEncoding dev '[FakeSize], QStartStop dev '[FakeSize] (SliceEncoding dev '[FakeSize]))
getOpen open = (stackUnsafe trEncs, stackUnsafe slcEncs)
 where
  (trs, slcs) = unzip $ pathTake 3 Inner Stop open
  trEncs = encodeTransition <$> trs
  slcEncs = qStartStop encodeSlice emptySlice <$> slcs

encodePVState
  :: (TT.KnownDevice dev)
  => PVState
  -> StateEncoding dev
encodePVState (GSFrozen frozen) =
  StateEncoding
    (qStop emptySlice)
    (qJust $ getFrozen frozen)
    (qNothing (stackUnsafe [emptyTransition], stackUnsafe [qStop emptySlice]))
encodePVState (GSOpen open _) =
  StateEncoding
    (qStart emptySlice)
    (qNothing (stackUnsafe [emptyTransition], stackUnsafe [qStart emptySlice]))
    (qJust $ getOpen open)
encodePVState (GSSemiOpen frozen mid open _) =
  StateEncoding
    (qInner $ encodeSlice mid)
    (qJust $ getFrozen frozen)
    (qJust $ getOpen open)

-- Step Encoding
-- -------------

data QEncoding dev batchShape = QEncoding
  { qActionEncoding :: !(ActionEncoding dev batchShape)
  , qStateEncoding :: !(StateEncoding dev)
  }
  deriving (Show, Generic, NFData)

instance Batchable (QEncoding dev shape) where
  type Batched (QEncoding dev shape) = QEncoding dev (1 : shape)
  addBatchDim (QEncoding ac st) = QEncoding (addBatchDim ac) st

encodeStep
  :: (TT.KnownDevice dev)
  => PVState
  -> PVAction
  -> QEncoding dev '[]
encodeStep state action =
  QEncoding
    (encodePVAction action)
    (encodePVState state)

withBatchedEncoding
  :: forall dev r
   . (TT.KnownDevice dev)
  => PVState
  -> NonEmpty PVAction
  -> (forall n. (KnownNat n) => QEncoding dev '[n] -> r)
  -> r
withBatchedEncoding state (a0 :| actions) f =
  VS.withSizedList aEncs inner
 where
  inner :: forall n. (KnownNat n) => VS.Vector n (ActionEncoding dev '[]) -> r
  inner aEncs' = f $ QEncoding (stack (VS.cons a0Enc aEncs')) sEnc
  a0Enc = encodePVAction a0
  aEncs = encodePVAction <$> actions
  sEnc = encodePVState state
