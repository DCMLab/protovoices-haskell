{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MagicHash #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeData #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}

module ReinforcementParser.Encoding where

import Common
import Data.Foldable qualified as F
import Data.HashSet qualified as HS
import Data.Hashable (Hashable)
import Data.Maybe (catMaybes, mapMaybe)
import Data.TypeNums (KnownInt, KnownNat, Nat, TInt (..), intVal, intVal', type (*), type (+), type (-))
import GHC.Exts (Proxy#, proxy#)
import GreedyParser
import Internal.MultiSet qualified as MS
import Musicology.Pitch
import PVGrammar (Edge, Edges (Edges), Freeze (FreezeOp), InnerEdge, Notes (Notes), PVAnalysis, PVLeftmost, Split, Spread)
import PVGrammar.Generate (derivationPlayerPV)
import PVGrammar.Parse (protoVoiceEvaluator, pvThaw)
import ReinforcementParser.ModelTypes
import Torch qualified as T
import Torch.Typed qualified as TT

-- Encoding
-- ========

-- Slice Encoding
-- --------------

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
  -> QTensor (PShape spec)
pitchesOneHot ps = TT.UnsafeMkTensor out
 where
  out =
    if HS.null ps
      then zeros
      else T.indexPut True indices values zeros
  indices = fmap T.asTensor $ T.asValue @[[Int]] $ T.transpose2D $ T.asTensor (pitch2index @(GenFifthLow spec) @(GenOctaveLow spec) <$> F.toList ps)
  values = T.ones [F.length ps] opts
  zeros = T.zeros dims opts
  fifthSize = TT.natValI @(GenFifthSize spec)
  octaveSize = TT.natValI @(GenOctaveSize spec)
  dims = [fifthSize, octaveSize]

encodeSlice
  :: forall (spec :: TGeneralSpec)
   . ( KnownInt (GenFifthLow spec)
     , KnownInt (GenOctaveLow spec)
     , KnownNat (GenFifthSize spec)
     , KnownNat (GenOctaveSize spec)
     )
  => Notes SPitch
  -> QTensor (PShape spec)
encodeSlice (Notes notes) =
  -- DT.trace ("ecoding slice" <> show notes) $
  pitchesOneHot @spec $ MS.toSet notes

-- Transition Encoding
-- -------------------

data TransitionEncoding spec = TransitionEncoding
  { trencPassing :: QTensor (EShape spec)
  , trencInner :: QTensor (EShape spec)
  , trencLeft :: QTensor (PShape spec)
  , trencRight :: QTensor (PShape spec)
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
  indexTensors = fmap T.asTensor $ T.asValue @[[Int]] $ T.transpose2D $ T.asTensor indices
  values = T.ones [F.length es] opts
  zeros = T.zeros dims opts
  fifthSize = TT.natValI @(GenFifthSize spec)
  octaveSize = TT.natValI @(GenOctaveSize spec)
  dims = [fifthSize, octaveSize, fifthSize, octaveSize]

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
    { trencPassing = edgesOneHot @spec $ MS.toSet pass
    , trencInner = edgesOneHot @spec $ getEdges getInner
    , trencLeft = pitchesOneHot @spec $ getEdges getLeft
    , trencRight = pitchesOneHot @spec $ getEdges getRight
    , trencRoot = HS.member (Start, Stop) reg
    }
 where
  regulars = HS.toList reg
  getEdges :: (Hashable a) => (Edge SPitch -> Maybe a) -> HS.HashSet a
  getEdges f = HS.fromList $ mapMaybe f regulars
  getInner (Inner a, Inner b) = Just (a, b)
  getInner _ = Nothing
  getLeft (Start, Inner b) = Just b
  getLeft _ = Nothing
  getRight (Inner a, Stop) = Just a
  getRight _ = Nothing

-- Action Encoding
-- ---------------

type PVAction = Action (Notes SPitch) (Edges SPitch) (Split SPitch) Freeze (Spread SPitch)

type SingleTop s =
  (StartStop (QTensor (PShape s)), TransitionEncoding s, StartStop (QTensor (PShape s)))
type DoubleTop s =
  ( StartStop (QTensor (PShape s))
  , TransitionEncoding s
  , QTensor (PShape s)
  , TransitionEncoding s
  , StartStop (QTensor (PShape s))
  )

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
  (sl, GreedyParser.Trans t _2nd, sr) = top
  encTop = Left (encodeSlice @spec <$> sl, encodeTransition @spec t, encodeSlice @spec <$> sr)
  encAction = case action of
    LMSingleFreeze FreezeOp -> LMFreezeOnly ()
    LMSingleSplit _split -> LMSplitOnly ()
encodePVAction (Right (ActionDouble top action)) = ActionEncoding encTop encAction
 where
  (sl, GreedyParser.Trans t1 _, sm, Trans t2 _, sr) = top
  encTop =
    Right
      ( encodeSlice @spec <$> sl
      , encodeTransition @spec t1
      , encodeSlice @spec sm
      , encodeTransition @spec t2
      , encodeSlice @spec <$> sr
      )
  encAction = case action of
    LMDoubleFreezeLeft FreezeOp -> LMFreezeLeft ()
    LMDoubleSplitLeft split -> LMSplitLeft ()
    LMDoubleSplitRight split -> LMSplitRight ()
    LMDoubleSpread spread -> LMSpread ()

-- State Encoding
-- --------------

data StateEncoding spec = StateEncoding
  { stateEncodingMid :: !(StartStop (QTensor (PShape spec)))
  , stateEncodingFrozen :: !(Maybe (TransitionEncoding spec, StartStop (QTensor (PShape spec))))
  , stateEncodingOpen :: ![(TransitionEncoding spec, StartStop (QTensor (PShape spec)))]
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
  -> (TransitionEncoding spec, StartStop (QTensor (PShape spec)))
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
  => Path (Trans (Edges SPitch)) (Notes SPitch)
  -> [(TransitionEncoding spec, StartStop (QTensor (PShape spec)))]
getOpen open = encodePair <$> pathTake 3 Inner Stop open
 where
  encodePair (Trans tr _, slc) = (encodeTransition tr, encodeSlice @spec <$> slc)

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
