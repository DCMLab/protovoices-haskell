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

-- Masked List
-- -----------

data QBoundedList (maxLen :: Nat) (batchShape :: [Nat]) (innerShape :: [Nat])
  = QBoundedList
  { qlMask :: QTensor (batchShape TT.++ '[maxLen])
  , qlContent :: QTensor (batchShape TT.++ '[maxLen] TT.++ innerShape)
  }
  deriving (Show)

qBoundedList
  :: forall maxLen innerShape
   . ( KnownNat maxLen
     , TT.KnownShape innerShape
     , TT.TensorOptions innerShape QDType QDevice
     )
  => [QTensor innerShape]
  -> QBoundedList maxLen '[] innerShape
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

instance Stackable (QBoundedList maxLen batchShape innerShape) where
  type
    Stacked (QBoundedList maxLen batchShape innerShape) n =
      QBoundedList maxLen (n ': batchShape) innerShape
  stack xs = QBoundedList masks contents
   where
    masks = TT.vecStack @0 $ qlMask <$> xs
    contents = TT.vecStack @0 $ qlContent <$> xs

-- Tagged StartStop
-- ----------------

data QStartStop (batchShape :: [Nat]) a = QStartStop
  { qssTag :: TT.Tensor QDevice TT.Int64 batchShape
  , qssContent :: a
  }

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

-- Slice Encoding
-- ==============

type SliceEncoding batchShape spec = QBoundedList MaxPitches batchShape '[PSize spec] -- QMaybe '[] (QTensor (PShape spec))

pitch2index
  :: forall (flow :: TInt) (olow :: TInt)
   . (KnownInt flow, KnownInt olow)
  => SPitch
  -> [Int]
pitch2index p = [fifths p - fifthLow, octaves p - octaveLow]
 where
  fifthLow = fromIntegral $ intVal' @flow proxy#
  octaveLow = fromIntegral $ intVal' @olow proxy#

pitchesOneHot
  :: forall (spec :: TGeneralSpec)
   . ( KnownInt (GenFifthLow spec)
     , KnownInt (GenOctaveLow spec)
     , KnownNat (GenFifthSize spec)
     , KnownNat (GenOctaveSize spec)
     )
  => HS.HashSet SPitch
  -> QTensor (PShape' spec)
pitchesOneHot ps = TT.UnsafeMkTensor out
 where
  out =
    if HS.null ps
      then zeros
      else T.indexPut True indices values zeros
  ~indices = T.asTensor <$> Data.List.transpose (pitch2index @(GenFifthLow spec) @(GenOctaveLow spec) <$> F.toList ps)
  values = T.ones [F.length ps] opts
  zeros = T.zeros dims opts
  fifthSize = TT.natValI @(GenFifthSize spec)
  octaveSize = TT.natValI @(GenOctaveSize spec)
  dims = [fifthSize, octaveSize]

encodeSlice'
  :: forall (spec :: TGeneralSpec)
   . ( KnownInt (GenFifthLow spec)
     , KnownInt (GenOctaveLow spec)
     , KnownNat (GenFifthSize spec)
     , KnownNat (GenOctaveSize spec)
     )
  => Notes SPitch
  -> QTensor (PShape' spec)
encodeSlice' (Notes notes) =
  -- DT.trace ("ecoding slice" <> show notes) $
  pitchesOneHot @spec $ MS.toSet notes

pitchesTokens
  :: forall (spec :: TGeneralSpec)
   . ( KnownInt (GenFifthLow spec)
     , KnownInt (GenOctaveLow spec)
     , KnownNat (GenFifthSize spec)
     , KnownNat (GenOctaveSize spec)
     , KnownNat (PSize spec)
     )
  => [SPitch]
  -> SliceEncoding '[] spec
pitchesTokens ps = qBoundedList (mkToken <$> ps)
 where
  -- todo: batch oneHot
  mkToken p =
    TT.UnsafeMkTensor $ toOpts $ T.cat (T.Dim 0) [T.oneHot fifthSize f, T.oneHot octaveSize o]
   where
    f = T.asTensor' (fifths p - fifthLow) $ T.withDType T.Int64 opts
    o = T.asTensor' (octaves p - octaveLow) $ T.withDType T.Int64 opts
  fifthLow = fromIntegral $ intVal' @(GenFifthLow spec) proxy#
  octaveLow = fromIntegral $ intVal' @(GenOctaveLow spec) proxy#
  fifthSize = TT.natValI @(GenFifthSize spec)
  octaveSize = TT.natValI @(GenOctaveSize spec)

encodeSlice
  :: forall (spec :: TGeneralSpec)
   . ( KnownInt (GenFifthLow spec)
     , KnownInt (GenOctaveLow spec)
     , KnownNat (GenFifthSize spec)
     , KnownNat (GenOctaveSize spec)
     , KnownNat (PSize spec)
     )
  => Notes SPitch
  -> SliceEncoding '[] spec
encodeSlice (Notes notes) =
  pitchesTokens @spec $ MS.toList notes

emptySlice
  :: forall spec
   . ( KnownInt (GenFifthLow spec)
     , KnownInt (GenOctaveLow spec)
     , KnownNat (GenFifthSize spec)
     , KnownNat (GenOctaveSize spec)
     , KnownNat (PSize spec)
     )
  => SliceEncoding '[] spec
emptySlice = encodeSlice @spec $ Notes MS.empty

-- Transition Encoding
-- -------------------

data TransitionEncoding batchShape spec = TransitionEncoding
  { trencPassing :: QBoundedList MaxEdges batchShape '[ESize spec] -- Maybe (QTensor (EShape spec))
  , trencInner :: QBoundedList MaxEdges batchShape '[ESize spec] -- Maybe (QTensor (EShape spec))
  , trencLeft :: SliceEncoding batchShape spec
  , trencRight :: SliceEncoding batchShape spec
  , trencRoot :: QTensor batchShape
  }
  deriving (Show)

instance Stackable (TransitionEncoding batchShape spec) where
  type
    Stacked (TransitionEncoding batchShape spec) n =
      TransitionEncoding (n ': batchShape) spec
  stack xs = TransitionEncoding passing inner left right root
   where
    passing = stack $ trencPassing <$> xs
    inner = stack $ trencInner <$> xs
    left = stack $ trencLeft <$> xs
    right = stack $ trencRight <$> xs
    root = TT.vecStack @0 $ trencRoot <$> xs

edgesOneHot
  :: forall (spec :: TGeneralSpec)
   . ( KnownInt (GenFifthLow spec)
     , KnownInt (GenOctaveLow spec)
     , KnownNat (GenFifthSize spec)
     , KnownNat (GenOctaveSize spec)
     )
  => HS.HashSet (InnerEdge SPitch)
  -> QTensor (EShape spec)
edgesOneHot es = TT.UnsafeMkTensor out
 where
  out =
    if HS.null es
      then zeros
      else T.indexPut True indexTensors values zeros
  edge2index (p1, p2) =
    pitch2index @(GenFifthLow spec) @(GenOctaveLow spec) p1
      ++ pitch2index @(GenFifthLow spec) @(GenOctaveLow spec) p2
  indices = edge2index <$> F.toList es
  ~indexTensors = T.asTensor <$> Data.List.transpose indices
  values = T.ones [F.length es] opts
  zeros = T.zeros dims opts
  fifthSize = TT.natValI @(GenFifthSize spec)
  octaveSize = TT.natValI @(GenOctaveSize spec)
  dims = [fifthSize, octaveSize, fifthSize, octaveSize]

edgesTokens
  :: forall (spec :: TGeneralSpec)
   . ( KnownInt (GenFifthLow spec)
     , KnownInt (GenOctaveLow spec)
     , KnownNat (GenFifthSize spec)
     , KnownNat (GenOctaveSize spec)
     , KnownNat (PSize spec + PSize spec)
     )
  => [InnerEdge SPitch]
  -> QBoundedList MaxEdges '[] '[ESize spec] -- Maybe (QTensor (EShape spec))
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
  fifthLow = fromIntegral $ intVal' @(GenFifthLow spec) proxy#
  octaveLow = fromIntegral $ intVal' @(GenOctaveLow spec) proxy#
  fifthSize = TT.natValI @(GenFifthSize spec)
  octaveSize = TT.natValI @(GenOctaveSize spec)

encodeTransition
  :: forall spec
   . ( KnownInt (GenFifthLow spec)
     , KnownInt (GenOctaveLow spec)
     , KnownNat (GenFifthSize spec)
     , KnownNat (GenOctaveSize spec)
     , KnownNat (PSize spec)
     , KnownNat (PSize spec + PSize spec)
     )
  => Edges SPitch
  -> TransitionEncoding '[] spec
encodeTransition (Edges reg pass) =
  TransitionEncoding
    { trencPassing = edgesTokens @spec $ MS.toList pass
    , -- , trencPassing = edgesOneHot @spec $ MS.toSet pass
      trencInner = edgesTokens @spec $ getEdges getInner
    , -- , trencInner = edgesOneHot @spec $ HS.fromList $ getEdges getInner
      trencLeft = pitchesTokens @spec $ getEdges getLeft
    , trencRight = pitchesTokens @spec $ getEdges getRight
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
  :: forall spec
   . ( KnownInt (GenFifthLow spec)
     , KnownInt (GenOctaveLow spec)
     , KnownNat (GenFifthSize spec)
     , KnownNat (GenOctaveSize spec)
     , KnownNat (PSize spec)
     , KnownNat (PSize spec + PSize spec)
     )
  => TransitionEncoding '[] spec
emptyTransition = encodeTransition $ Edges HS.empty MS.empty

-- Action Encoding
-- ---------------

data ActionTop batchShape spec = ActionTop
  { atopSl :: !(QStartStop batchShape (SliceEncoding batchShape spec))
  , atopT1 :: !(TransitionEncoding batchShape spec)
  , atopSm :: !(QMaybe batchShape (SliceEncoding batchShape spec))
  , atopT2 :: !(QMaybe batchShape (TransitionEncoding batchShape spec))
  , atopSr :: !(QStartStop batchShape (SliceEncoding batchShape spec))
  }

instance Stackable (ActionTop batchShape spec) where
  type Stacked (ActionTop batchShape spec) n = ActionTop (n ': batchShape) spec
  stack xs = ActionTop sl t1 sm t2 sr
   where
    sl = stack $ atopSl <$> xs
    t1 = stack $ atopT1 <$> xs
    sm = stack $ atopSm <$> xs
    t2 = stack $ atopT2 <$> xs
    sr = stack $ atopSr <$> xs

data ActionEncoding batchShape spec = ActionEncoding
  { actionEncodingTop :: !(ActionTop batchShape spec) -- (Either (SingleTop batchShape spec) (DoubleTop batchShape spec))
  , actionEncodingOp :: !(TT.Tensor QDevice 'TT.Int64 batchShape) -- !(Leftmost () () ())
  }

instance Stackable (ActionEncoding batchShape spec) where
  type Stacked (ActionEncoding batchShape spec) n = ActionEncoding (n ': batchShape) spec
  stack xs = ActionEncoding tops ops
   where
    tops = stack $ actionEncodingTop <$> xs
    ops = TT.vecStack @0 $ actionEncodingOp <$> xs

encodePVAction
  :: forall (spec :: TGeneralSpec)
   . ( KnownInt (GenFifthLow spec)
     , KnownInt (GenOctaveLow spec)
     , KnownNat (GenFifthSize spec)
     , KnownNat (GenOctaveSize spec)
     , KnownNat (PSize spec)
     , KnownNat (PSize spec + PSize spec)
     )
  => PVAction
  -> ActionEncoding '[] spec
encodePVAction (Left (ActionSingle top action)) = ActionEncoding encTop encAction
 where
  (SingleParent sl t sr) = top
  encTop =
    ActionTop
      (qStartStop (encodeSlice @spec) (emptySlice @spec) sl)
      (encodeTransition @spec t)
      (qNothing $ emptySlice @spec)
      (qNothing $ emptyTransition @spec)
      (qStartStop (encodeSlice @spec) (emptySlice @spec) sr)
  encAction = case action of
    LMSingleFreeze FreezeOp -> TT.full (0 :: Int) --  LMFreezeOnly ()
    LMSingleSplit _split -> TT.full (1 :: Int) -- LMSplitOnly ()
encodePVAction (Right (ActionDouble top action)) = ActionEncoding encTop encAction
 where
  (DoubleParent sl t1 sm t2 sr) = top
  encTop =
    ActionTop
      (qStartStop (encodeSlice @spec) (emptySlice @spec) sl)
      (encodeTransition @spec t1)
      (qJust $ encodeSlice @spec sm)
      (qJust $ encodeTransition @spec t2)
      (qStartStop (encodeSlice @spec) (emptySlice @spec) sr)

  encAction = case action of
    LMDoubleFreezeLeft FreezeOp -> TT.full (2 :: Int) -- LMFreezeLeft ()
    LMDoubleSpread spread -> TT.full (3 :: Int) -- LMSpread ()
    LMDoubleSplitLeft split -> TT.full (4 :: Int) -- LMSplitLeft ()
    LMDoubleSplitRight split -> TT.full (5 :: Int) -- LMSplitRight ()

-- State Encoding
-- --------------

data StateEncoding spec = StateEncoding
  { stateEncodingMid :: !(StartStop (SliceEncoding '[] spec))
  , stateEncodingFrozen :: !(Maybe (TransitionEncoding '[] spec, StartStop (SliceEncoding '[] spec)))
  , stateEncodingOpen :: ![(TransitionEncoding '[] spec, StartStop (SliceEncoding '[] spec))]
  }

type PVState t =
  GreedyState
    (Edges SPitch)
    (t (Edge SPitch))
    (Notes SPitch)
    (PVLeftmost SPitch)

getFrozen
  :: forall spec t
   . ( Foldable t
     , KnownInt (GenFifthLow spec)
     , KnownInt (GenOctaveLow spec)
     , KnownNat (GenFifthSize spec)
     , KnownNat (GenOctaveSize spec)
     , KnownNat (PSize spec)
     , KnownNat (PSize spec + PSize spec)
     )
  => Path (Maybe (t (Edge SPitch))) (Notes SPitch)
  -> (TransitionEncoding '[] spec, StartStop (SliceEncoding '[] spec))
getFrozen frozen = case frozen of
  PathEnd tr -> (encodeTransition $ pvThaw tr, Start)
  Path tr slc _ ->
    (encodeTransition $ pvThaw tr, Inner $ encodeSlice @spec slc)

getOpen
  :: forall spec
   . ( KnownInt (GenFifthLow spec)
     , KnownInt (GenOctaveLow spec)
     , KnownNat (GenFifthSize spec)
     , KnownNat (GenOctaveSize spec)
     , KnownNat (PSize spec)
     , KnownNat (PSize spec + PSize spec)
     )
  => Path (Edges SPitch) (Notes SPitch)
  -> [(TransitionEncoding '[] spec, StartStop (SliceEncoding '[] spec))]
getOpen open = encodePair <$> pathTake 3 Inner Stop open
 where
  encodePair (tr, slc) = (encodeTransition tr, encodeSlice @spec <$> slc)

encodePVState
  :: forall spec t
   . ( Foldable t
     , KnownInt (GenFifthLow spec)
     , KnownInt (GenOctaveLow spec)
     , KnownNat (GenFifthSize spec)
     , KnownNat (GenOctaveSize spec)
     , KnownNat (PSize spec)
     , KnownNat (PSize spec + PSize spec)
     )
  => PVState t
  -> StateEncoding spec
encodePVState (GSFrozen frozen) = StateEncoding Stop (Just $! getFrozen frozen) []
encodePVState (GSOpen open _) = StateEncoding Start Nothing (getOpen open)
encodePVState (GSSemiOpen frozen mid open _) =
  StateEncoding (Inner $ encodeSlice @spec mid) (Just $! getFrozen frozen) (getOpen open)

-- Step Encoding
-- -------------

data QEncoding batchShape spec = QEncoding
  { qActionEncoding :: !(ActionEncoding batchShape spec)
  , qStateEncoding :: !(StateEncoding spec)
  }

encodeStep
  :: (Foldable t)
  => PVState t
  -> PVAction
  -> QEncoding '[] TGeneralSpecDefault
encodeStep state action =
  QEncoding
    (encodePVAction @TGeneralSpecDefault action)
    (encodePVState @TGeneralSpecDefault state)

withBatchedEncoding
  :: (Foldable t)
  => PVState t
  -> [PVAction]
  -> (forall n. (KnownNat n) => QEncoding '[n] TGeneralSpecDefault -> r)
  -> r
withBatchedEncoding state actions f =
  VS.withSizedList aEncs $ \aEncs' -> f $ QEncoding (stack aEncs') sEnc
 where
  aEncs = encodePVAction @TGeneralSpecDefault <$> actions
  sEnc = encodePVState @TGeneralSpecDefault state
