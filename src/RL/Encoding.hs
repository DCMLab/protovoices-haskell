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

-- Encoding
-- ========

-- Slice Encoding
-- --------------

type SliceEncoding spec = Maybe (QTensor (PShape spec))

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
     )
  => [SPitch]
  -> SliceEncoding spec
pitchesTokens [] = Nothing
pitchesTokens ps = Just $ TT.UnsafeMkTensor $ toOpts $ T.stack (T.Dim 0) (mkToken <$> ps)
 where
  -- todo: batch oneHot
  mkToken p =
    T.cat (T.Dim 0) [T.oneHot fifthSize f, T.oneHot octaveSize o]
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
     )
  => Notes SPitch
  -> SliceEncoding spec
encodeSlice (Notes notes) =
  pitchesTokens @spec $ MS.toList notes

-- Transition Encoding
-- -------------------

data TransitionEncoding spec = TransitionEncoding
  { trencPassing :: Maybe (QTensor (EShape spec))
  , trencInner :: Maybe (QTensor (EShape spec))
  , trencLeft :: SliceEncoding spec
  , trencRight :: SliceEncoding spec
  , trencRoot :: Bool
  }
  deriving (Show)

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
     )
  => [InnerEdge SPitch]
  -> Maybe (QTensor (EShape spec))
edgesTokens [] = Nothing
edgesTokens es = Just $ TT.UnsafeMkTensor $ toOpts $ T.stack (T.Dim 0) (mkToken <$> es)
 where
  -- todo: batch oneHot
  mkToken (p1, p2) =
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
     )
  => Edges SPitch
  -> TransitionEncoding spec
encodeTransition (Edges reg pass) =
  TransitionEncoding
    { trencPassing = edgesTokens @spec $ MS.toList pass
    , -- , trencPassing = edgesOneHot @spec $ MS.toSet pass
      trencInner = edgesTokens @spec $ getEdges getInner
    , -- , trencInner = edgesOneHot @spec $ HS.fromList $ getEdges getInner
      trencLeft = pitchesTokens @spec $ getEdges getLeft
    , trencRight = pitchesTokens @spec $ getEdges getRight
    , trencRoot = HS.member (Start, Stop) reg
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

-- Action Encoding
-- ---------------

type SingleTop spec =
  SingleParent (SliceEncoding spec) (TransitionEncoding spec)

type DoubleTop spec = DoubleParent (SliceEncoding spec) (TransitionEncoding spec)

data ActionEncoding spec = ActionEncoding
  { actionEncodingTop :: !(Either (SingleTop spec) (DoubleTop spec))
  , actionEncodingOp :: !(Leftmost () () ())
  }

encodePVAction
  :: forall (spec :: TGeneralSpec)
   . ( KnownInt (GenFifthLow spec)
     , KnownInt (GenOctaveLow spec)
     , KnownNat (GenFifthSize spec)
     , KnownNat (GenOctaveSize spec)
     )
  => PVAction
  -> ActionEncoding spec
encodePVAction (Left (ActionSingle top action)) = ActionEncoding encTop encAction
 where
  (SingleParent sl t sr) = top
  encTop = Left $ SingleParent (encodeSlice @spec <$> sl) (encodeTransition @spec t) (encodeSlice @spec <$> sr)
  encAction = case action of
    LMSingleFreeze FreezeOp -> LMFreezeOnly ()
    LMSingleSplit _split -> LMSplitOnly ()
encodePVAction (Right (ActionDouble top action)) = ActionEncoding encTop encAction
 where
  (DoubleParent sl t1 sm t2 sr) = top
  encTop =
    Right $
      DoubleParent
        (encodeSlice @spec <$> sl)
        (encodeTransition @spec t1)
        (encodeSlice @spec sm)
        (encodeTransition @spec t2)
        (encodeSlice @spec <$> sr)

  encAction = case action of
    LMDoubleFreezeLeft FreezeOp -> LMFreezeLeft ()
    LMDoubleSplitLeft split -> LMSplitLeft ()
    LMDoubleSplitRight split -> LMSplitRight ()
    LMDoubleSpread spread -> LMSpread ()

-- State Encoding
-- --------------

data StateEncoding spec = StateEncoding
  { stateEncodingMid :: !(StartStop (SliceEncoding spec))
  , stateEncodingFrozen :: !(Maybe (TransitionEncoding spec, StartStop (SliceEncoding spec)))
  , stateEncodingOpen :: ![(TransitionEncoding spec, StartStop (SliceEncoding spec))]
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
     )
  => Path (Maybe (t (Edge SPitch))) (Notes SPitch)
  -> (TransitionEncoding spec, StartStop (SliceEncoding spec))
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
     )
  => Path (Edges SPitch) (Notes SPitch)
  -> [(TransitionEncoding spec, StartStop (SliceEncoding spec))]
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
     )
  => PVState t
  -> StateEncoding spec
encodePVState (GSFrozen frozen) = StateEncoding Stop (Just $ getFrozen frozen) []
encodePVState (GSOpen open _) = StateEncoding Start Nothing (getOpen open)
encodePVState (GSSemiOpen frozen mid open _) =
  StateEncoding (Inner $ encodeSlice @spec mid) (Just $ getFrozen frozen) (getOpen open)

-- Step Encoding
-- -------------

data QEncoding spec = QEncoding
  { qActionEncoding :: !(ActionEncoding spec)
  , qStateEncoding :: !(StateEncoding spec)
  }

encodeStep
  :: (Foldable t)
  => PVState t
  -> PVAction
  -> QEncoding TGeneralSpecDefault
encodeStep state action =
  QEncoding
    (encodePVAction @TGeneralSpecDefault action)
    (encodePVState @TGeneralSpecDefault state)
