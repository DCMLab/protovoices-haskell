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
{-# HLINT ignore "Use <$>" #-}
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}

module ReinforcementParser.Model where

import Common
import Control.Arrow ((>>>))
import Control.DeepSeq (force)
import Control.Exception (Exception, catch, onException)
import Control.Foldl qualified as Foldl
import Control.Monad (foldM, foldM_, forM_, replicateM, when)
import Control.Monad.Except qualified as ET
import Control.Monad.Primitive (RealWorld)
import Control.Monad.State qualified as ST
import Control.Monad.Trans (lift)
import Data.Foldable qualified as F
import Data.Function ((&))
import Data.HashSet qualified as HS
import Data.Hashable (Hashable)
import Data.Kind (Type)
import Data.List.Extra qualified as E
import Data.Maybe (catMaybes, mapMaybe)
import Data.Proxy (Proxy (Proxy))
import Data.TypeNums (KnownInt, KnownNat, Nat, TInt (..), intVal, intVal', type (*), type (+), type (-))
import Data.Vector qualified as V
import Debug.Trace qualified as DT
import Display (replayDerivation, viewGraph)
import GHC.Exts (Proxy#, proxy#)
import GHC.Generics (Generic)
import Graphics.Rendering.Chart.Backend.Cairo as Plt
import Graphics.Rendering.Chart.Easy ((.=))
import Graphics.Rendering.Chart.Easy qualified as Plt
import Graphics.Rendering.Chart.Gtk qualified as Plt
import GreedyParser (Action, ActionDouble (ActionDouble), ActionSingle (ActionSingle), GreedyState (..), Trans (Trans), getActions, initParseState, parseGreedy, parseStep, pickRandom)
import Inference.Conjugate (HyperRep, Prior (expectedProbs), evalTraceLogP, sampleProbs)
import Internal.MultiSet qualified as MS
import Internal.TorchHelpers qualified as TH
import Musicology.Pitch
import PVGrammar (Edge, Edges (Edges), Freeze (FreezeOp), InnerEdge, Notes (Notes), PVAnalysis, PVLeftmost, Split, Spread)
import PVGrammar.Generate (derivationPlayerPV)
import PVGrammar.Parse (protoVoiceEvaluator, pvThaw)
import PVGrammar.Prob.Simple (PVParams, observeDerivation, observeDerivation', sampleDerivation')
import System.Random (RandomGen, getStdRandom)
import System.Random.MWC.Distributions (categorical)
import System.Random.MWC.Probability qualified as MWC
import System.Random.Shuffle (shuffle')
import System.Random.Stateful as Rand (StatefulGen, UniformRange (uniformRM), split)
import Torch qualified as T
import Torch.Lens qualified
import Torch.Typed qualified as TT
import Torch.Typed.NamedTensor qualified as TT

-- Notes
-- -----

{-
Variant of Q-learning:
- instead of Q value (expected total reward) under optimal policy
  learn "P value": expected probability under random policy
- does this lead to a policy where p(as) ∝ reward?
  - then you learn a method of sampling from the reward distribution
  - if reward is a probability (e.g. p(deriv)), you learn to sample from that!
    - useful for unsupervised inference
- changes:
  - use proportional random policy (is this MC-tree-search?)
  - loss uses E[] instead of max over next actions.
-}

-- global settings
-- ---------------

device :: T.Device
-- device = T.Device T.CUDA 0
device = T.Device T.CPU 0

type QDevice = '(TT.CPU, 0)

type QDType = TT.Double

type QType = Double

qDType :: TT.DType
qDType = T.Double

type QTensor shape = TT.Tensor QDevice QDType shape

opts :: T.TensorOptions
opts = T.withDType qDType $ T.withDevice device T.defaultOpts

toOpts :: forall a. (Torch.Lens.HasTypes a T.Tensor) => a -> a
toOpts = T.toDevice device . T.toType qDType

activation :: QTensor shape -> QTensor shape
activation = TT.selu

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

-- Q net
-- =====

-- General Spec
-- ------------

data GeneralSpec (spec :: TGeneralSpec) = GeneralSpec

defaultGSpec :: GeneralSpec spec
defaultGSpec =
  GeneralSpec

type data TGeneralSpec = TGenSpec TInt Nat TInt Nat Nat

type family GenFifthLow (spec :: TGeneralSpec) where
  GenFifthLow (TGenSpec flow _ _ _ _) = flow

type family GenFifthSize (spec :: TGeneralSpec) where
  GenFifthSize (TGenSpec _ fsize _ _ _) = fsize

type family GenOctaveLow (spec :: TGeneralSpec) where
  GenOctaveLow (TGenSpec _ _ olow _ _) = olow

type family GenOctaveSize (spec :: TGeneralSpec) where
  GenOctaveSize (TGenSpec _ _ _ osize _) = osize

type family GenEmbSize (spec :: TGeneralSpec) where
  GenEmbSize (TGenSpec _ _ _ _ esize) = esize

type family PShape (spec :: TGeneralSpec) where
  PShape (TGenSpec _ fs _ os _) = '[fs, os]

type family EShape (spec :: TGeneralSpec) where
  EShape (TGenSpec _ fs _ os _) = '[fs, os, fs, os]

-- Learned Constant Embeddings
-- ---------------------------

data ConstEmbSpec (shape :: [Nat]) = ConstEmbSpec -- [Int]

newtype ConstEmb shape = ConstEmb (TT.Parameter QDevice QDType shape)
  deriving (Show, Generic)
  deriving anyclass (TT.Parameterized)

instance
  (TT.TensorOptions shape QDType QDevice)
  => T.Randomizable (ConstEmbSpec shape) (ConstEmb shape)
  where
  sample :: ConstEmbSpec shape -> IO (ConstEmb shape)
  sample ConstEmbSpec = ConstEmb <$> (TT.makeIndependent =<< TT.randn)

instance T.HasForward (ConstEmb size) () (QTensor size) where
  forward :: ConstEmb size -> () -> QTensor size
  forward (ConstEmb emb) () = TT.toDependent emb
  forwardStoch :: ConstEmb size -> () -> IO (QTensor size)
  forwardStoch model input = pure $ T.forward model input

-- Slice Encoder
-- -------------

data SliceSpec (hidden :: Nat) = SliceSpec

data SliceEncoder spec hidden = SliceEncoder
  { _slcL1 :: !(TT.Linear (TT.Product (PShape spec)) hidden QDType QDevice)
  , _slcL2 :: !(TT.Linear hidden (GenEmbSize spec) QDType QDevice)
  , _slcStart :: !(ConstEmb '[GenEmbSize spec])
  , _slcStop :: !(ConstEmb '[GenEmbSize spec])
  }
  deriving (Show, Generic, TT.Parameterized)

instance
  ( KnownNat hidden
  , KnownNat (GenEmbSize spec)
  , KnownNat (TT.Product (PShape spec))
  )
  => T.Randomizable (GeneralSpec spec, SliceSpec hidden) (SliceEncoder spec hidden)
  where
  sample :: (GeneralSpec spec, SliceSpec hidden) -> IO (SliceEncoder spec hidden)
  sample _ =
    SliceEncoder
      <$> T.sample TT.LinearSpec
      <*> T.sample TT.LinearSpec
      <*> T.sample ConstEmbSpec
      <*> T.sample ConstEmbSpec

instance
  (embshape ~ '[GenEmbSize spec], pshape ~ PShape spec, TT.KnownShape pshape)
  => TT.HasForward (SliceEncoder spec hidden) (StartStop (QTensor pshape)) (QTensor embshape)
  where
  forward model@(SliceEncoder _ _ start stop) input =
    case input of
      Inner slc -> T.forward model slc
      Start -> T.forward start ()
      Stop -> T.forward stop ()
  forwardStoch model input = pure $ T.forward model input

instance
  ( embshape ~ '[GenEmbSize spec]
  , pshape ~ PShape spec
  , TT.KnownShape pshape
  )
  => T.HasForward (SliceEncoder spec hidden) (QTensor pshape) (QTensor embshape)
  where
  forward (SliceEncoder l1 l2 _ _) =
    TT.flattenAll
      >>> T.forward l1
      >>> activation
      >>> TT.forward l2
      >>> activation
  forwardStoch model = pure . T.forward model

-- Transition Encoder
-- ------------------

data TransitionSpec (hidden :: Nat) = TransitionSpec

data TransitionEncoder spec hidden = TransitionEncoder
  { trL1Passing :: !(TT.Linear (TT.Product (EShape spec)) hidden QDType QDevice)
  , trL1Inner :: !(TT.Linear (TT.Product (EShape spec)) hidden QDType QDevice)
  , trL1Left :: !(TT.Linear (TT.Product (PShape spec)) hidden QDType QDevice)
  , trL1Right :: !(TT.Linear (TT.Product (PShape spec)) hidden QDType QDevice)
  , trL1Root :: !(ConstEmb '[hidden])
  , trL2 :: !(TT.Linear hidden (GenEmbSize spec) QDType QDevice)
  }
  deriving (Show, Generic, TT.Parameterized)

instance
  ( KnownNat hidden
  , KnownNat (GenEmbSize spec)
  , KnownNat (TT.Product (PShape spec))
  , KnownNat (TT.Product (EShape spec))
  )
  => T.Randomizable (GeneralSpec spec, TransitionSpec hidden) (TransitionEncoder spec hidden)
  where
  sample :: (GeneralSpec spec, TransitionSpec hidden) -> IO (TransitionEncoder spec hidden)
  sample _ = do
    trL1Passing <- T.sample TT.LinearSpec
    trL1Inner <- T.sample TT.LinearSpec
    trL1Left <- T.sample TT.LinearSpec
    trL1Right <- T.sample TT.LinearSpec
    trL1Root <- T.sample ConstEmbSpec
    trL2 <- T.sample TT.LinearSpec
    pure $ TransitionEncoder{..}

instance
  forall spec hidden embshape
   . ( embshape ~ '[GenEmbSize spec]
     , TT.KnownShape (EShape spec)
     , TT.KnownShape (PShape spec)
     )
  => T.HasForward
      (TransitionEncoder spec hidden)
      (TransitionEncoding spec)
      (QTensor embshape)
  where
  forward TransitionEncoder{..} TransitionEncoding{..} = activation $ T.forward trL2 all'
   where
    pass = activation $ T.forward trL1Passing $ TT.flattenAll trencPassing
    inner = activation $ T.forward trL1Inner $ TT.flattenAll trencInner
    left = activation $ T.forward trL1Left $ TT.flattenAll trencLeft
    right = activation $ T.forward trL1Right $ TT.flattenAll trencRight
    all = pass + inner + left + right
    all' = if trencRoot then all + activation (T.forward trL1Root ()) else all
  forwardStoch = undefined

-- ActionEncoder
-- -------------

data ActionSpec (hidden :: Nat) = ActionSpec

data ActionEncoder spec hidden = ActionEncoder
  { actTop1sl :: TT.Linear (GenEmbSize spec) hidden QDType QDevice
  , actTop1sm :: TT.Linear (GenEmbSize spec) hidden QDType QDevice
  , actTop1sr :: TT.Linear (GenEmbSize spec) hidden QDType QDevice
  , actTop1t1 :: TT.Linear (GenEmbSize spec) hidden QDType QDevice
  , actTop1t2 :: TT.Linear (GenEmbSize spec) hidden QDType QDevice
  , actTop2 :: TT.Linear hidden (GenEmbSize spec) QDType QDevice
  , actSplit :: ConstEmb '[GenEmbSize spec - 3] -- TODO: fill in with actual module
  , actSpread :: ConstEmb '[GenEmbSize spec - 3] -- TODO: fill in with actual module
  , actFreeze :: ConstEmb '[GenEmbSize spec - 3]
  }
  deriving (Show, Generic, TT.Parameterized)

instance
  ( emb ~ GenEmbSize spec
  , KnownNat hidden
  , KnownNat emb
  , KnownNat (emb - 3)
  )
  => T.Randomizable (GeneralSpec spec, ActionSpec hidden) (ActionEncoder spec hidden)
  where
  sample :: (GeneralSpec spec, ActionSpec hidden) -> IO (ActionEncoder spec hidden)
  sample (GeneralSpec, ActionSpec) = do
    actTop1sl <- T.sample TT.LinearSpec
    actTop1sm <- T.sample TT.LinearSpec
    actTop1sr <- T.sample TT.LinearSpec
    actTop1t1 <- T.sample TT.LinearSpec
    actTop1t2 <- T.sample TT.LinearSpec
    actTop2 <- T.sample TT.LinearSpec
    actSplit <- T.sample ConstEmbSpec
    actSpread <- T.sample ConstEmbSpec
    actFreeze <- T.sample ConstEmbSpec
    pure ActionEncoder{..}

instance
  forall (spec :: TGeneralSpec) slcHidden trHidden actHidden outShape emb
   . ( emb ~ GenEmbSize spec
     , outShape ~ '[emb]
     , emb ~ (emb - 3) + 3
     , TT.KnownShape (PShape spec)
     , TT.KnownShape (EShape spec)
     )
  => T.HasForward
      (ActionEncoder spec actHidden)
      (SliceEncoder spec slcHidden, TransitionEncoder spec trHidden, ActionEncoding spec)
      (QTensor outShape)
  where
  forward ActionEncoder{..} (slc, tr, ActionEncoding top op) = topEmb + opEmb
   where
    topCombined :: QTensor '[actHidden]
    topCombined = case top of
      Left (sl, t, sr) ->
        let
          embl = activation $ T.forward actTop1sl $ T.forward slc sl
          embt = activation $ T.forward actTop1t1 $ T.forward slc sl
          embr = activation $ T.forward actTop1sr $ T.forward slc sl
         in
          embl + embt + embr
      Right (sl, t1, sm, t2, sr) ->
        let
          embl = activation $ T.forward actTop1sl $ T.forward slc sl
          embm = activation $ T.forward actTop1sl $ T.forward slc sm
          embr = activation $ T.forward actTop1sl $ T.forward slc sr
          embt1 = activation $ T.forward actTop1sl $ T.forward tr t1
          embt2 = activation $ T.forward actTop1sl $ T.forward tr t2
         in
          embl + embm + embr + embt1 + embt2
    topEmb :: QTensor '[GenEmbSize spec]
    topEmb = activation $ T.forward actTop2 topCombined
    opEmb :: QTensor '[GenEmbSize spec]
    opEmb = case op of
      LMFreezeOnly _ -> TT.cat @0 (TT.selectIdx @0 opTypes 0 TT.:. T.forward actFreeze () TT.:. TT.HNil)
      LMSplitOnly _ -> TT.cat @0 (TT.selectIdx @0 opTypes 1 TT.:. T.forward actSplit () TT.:. TT.HNil)
      LMFreezeLeft _ -> TT.cat @0 (TT.selectIdx @0 opTypes 2 TT.:. T.forward actFreeze () TT.:. TT.HNil)
      LMSpread _ -> TT.cat @0 (TT.selectIdx @0 opTypes 3 TT.:. T.forward actSpread () TT.:. TT.HNil)
      LMSplitLeft _ -> TT.cat @0 (TT.selectIdx @0 opTypes 4 TT.:. T.forward actSplit () TT.:. TT.HNil)
      LMSplitRight _ -> TT.cat @0 (TT.selectIdx @0 opTypes 5 TT.:. T.forward actSplit () TT.:. TT.HNil)
    opTypes :: QTensor '[6, 3]
    opTypes =
      TT.UnsafeMkTensor $
        T.asTensor' @[[QType]]
          [ [0, 0, 0] -- freeze only
          , [0, 1, 0] -- split only
          , [1, 0, 0] -- freeze left
          , [1, 0, 1] -- spread
          , [1, 1, 0] -- freeze left
          , [1, 1, 1] -- freeze right
          ]
          opts
  forwardStoch a i = pure $ T.forward a i

-- State Encoder
-- -------------

data StateSpec (hidden :: Nat) = StateSpec

data StateEncoder spec hidden = StateEncoder
  { stL1mid :: TT.Linear (GenEmbSize spec) hidden QDType QDevice
  , stL1frozenSlc :: TT.Linear (GenEmbSize spec) hidden QDType QDevice
  , stL1frozenTr :: TT.Linear (GenEmbSize spec) hidden QDType QDevice
  , stL1openSlc1 :: TT.Linear (GenEmbSize spec) hidden QDType QDevice
  , stL1openSlc2 :: TT.Linear (GenEmbSize spec) hidden QDType QDevice
  , stL1openSlc3 :: TT.Linear (GenEmbSize spec) hidden QDType QDevice
  , stL1openTr1 :: TT.Linear (GenEmbSize spec) hidden QDType QDevice
  , stL1openTr2 :: TT.Linear (GenEmbSize spec) hidden QDType QDevice
  , stL1openTr3 :: TT.Linear (GenEmbSize spec) hidden QDType QDevice
  , stL2 :: TT.Linear hidden hidden QDType QDevice
  , stL3 :: TT.Linear hidden (GenEmbSize spec) QDType QDevice
  }
  deriving (Show, Generic, TT.Parameterized)

instance
  ( KnownNat (GenEmbSize spec)
  , KnownNat hidden
  )
  => T.Randomizable (GeneralSpec spec, StateSpec hidden) (StateEncoder spec hidden)
  where
  sample _ = do
    stL1mid <- TT.sample TT.LinearSpec
    stL1frozenSlc <- TT.sample TT.LinearSpec
    stL1frozenTr <- TT.sample TT.LinearSpec
    stL1openSlc1 <- TT.sample TT.LinearSpec
    stL1openSlc2 <- TT.sample TT.LinearSpec
    stL1openSlc3 <- TT.sample TT.LinearSpec
    stL1openTr1 <- TT.sample TT.LinearSpec
    stL1openTr2 <- TT.sample TT.LinearSpec
    stL1openTr3 <- TT.sample TT.LinearSpec
    stL2 <- TT.sample TT.LinearSpec
    stL3 <- TT.sample TT.LinearSpec
    pure StateEncoder{..}

instance
  forall (spec :: TGeneralSpec) slcHidden trHidden stHidden outShape emb
   . ( emb ~ GenEmbSize spec
     , outShape ~ '[emb]
     , TT.KnownShape (PShape spec)
     , TT.KnownShape (EShape spec)
     )
  => T.HasForward
      (StateEncoder spec stHidden)
      (SliceEncoder spec slcHidden, TransitionEncoder spec trHidden, StateEncoding spec)
      (QTensor outShape)
  where
  forward StateEncoder{..} (slc, tr, StateEncoding mid frozen open) =
    fullEmb
      & T.forward stL2
      & activation
      & T.forward stL3
      & activation
   where
    -- embed the mid slice
    midEmb = activation $ T.forward stL1mid $ T.forward slc mid
    -- embed the frozen segment (if it exists) and add to midEmb
    midAndFrozen = case frozen of
      Nothing -> midEmb
      Just (ft, fs) ->
        let ftEmb = activation $ T.forward stL1frozenTr $ T.forward tr ft
            fsEmb = activation $ T.forward stL1frozenSlc $ T.forward slc fs
         in midEmb + ftEmb + fsEmb
    -- embed an open segment using its respective layers
    embedOpen ((ot, os), (l1tr, l1slc)) = otEmb + osEmb
     where
      otEmb = activation $ T.forward l1tr $ T.forward tr ot
      osEmb = activation $ T.forward l1slc $ T.forward slc os
    -- the list of layers for the 3 open transitions and slices
    openEncoders =
      [ (stL1openSlc1, stL1openTr1)
      , (stL1openSlc2, stL1openTr2)
      , (stL1openSlc3, stL1openTr3)
      ]
    -- embed the open segments and add them to mid and frozen
    fullEmb = F.foldl' (+) midAndFrozen $ zipWith (curry embedOpen) open openEncoders
  forwardStoch a i = pure $ T.forward a i

-- Full Model
-- ----------

data SpecialSpec (hidden :: Nat) = SpecialSpec

data QSpec (spec :: TGeneralSpec) specialSpec sliceSpec transSpec actionSpec stateSpec
  = QSpec
      (GeneralSpec spec)
      (SpecialSpec specialSpec)
      (SliceSpec sliceSpec)
      (TransitionSpec transSpec)
      (ActionSpec actionSpec)
      (StateSpec stateSpec)

type family QSpecGeneral qspec where
  QSpecGeneral (QSpec g _ _ _ _ _) = g

type family QSpecSpecial qspec where
  QSpecSpecial (QSpec _ s _ _ _ _) = s

type family QSpecSlice qspec where
  QSpecSlice (QSpec _ _ s _ _ _) = s

type family QSpecTrans qspec where
  QSpecTrans (QSpec _ _ _ t _ _) = t

type family QSpecAction qspec where
  QSpecAction (QSpec _ _ _ _ a _) = a

type family QSpecState qspec where
  QSpecState (QSpec _ _ _ _ _ st) = st

type TGeneralSpecDefault = TGenSpec (Neg 3) 12 (Pos 2) 5 32

type DefaultQSpec = QSpec TGeneralSpecDefault 32 32 32 32 64

defaultSpec :: DefaultQSpec
defaultSpec =
  QSpec
    defaultGSpec
    SpecialSpec
    SliceSpec
    TransitionSpec
    ActionSpec
    StateSpec

data QModel spec = QModel
  { qModelSlc :: !(SliceEncoder (QSpecGeneral spec) (QSpecSlice spec))
  , qModelTr :: !(TransitionEncoder (QSpecGeneral spec) (QSpecTrans spec))
  , qModelAct :: !(ActionEncoder (QSpecGeneral spec) (QSpecAction spec))
  , qModelSt :: !(StateEncoder (QSpecGeneral spec) (QSpecState spec))
  , qModelFinal1 :: !(TT.Linear (GenEmbSize (QSpecGeneral spec)) (QSpecSpecial spec) QDType QDevice)
  , qModelFinal2 :: !(TT.Linear (QSpecSpecial spec) 1 QDType QDevice)
  }
  deriving (Show, Generic, TT.Parameterized)

instance
  ( spec ~ QSpec g sp sl tr ac st
  , embsize ~ GenEmbSize g
  , KnownNat sp
  , KnownNat sl
  , KnownNat tr
  , KnownNat ac
  , KnownNat st
  , KnownNat embsize
  , KnownNat (embsize - 3)
  , -- , KnownNat (embsize + embsize)
    -- , KnownNat (embsize + (embsize + embsize))
    KnownNat (TT.Product (PShape g))
  , KnownNat (TT.Product (EShape g))
  )
  => T.Randomizable (QSpec g sp sl tr ac st) (QModel spec)
  where
  sample :: spec -> IO (QModel spec)
  sample (QSpec gspec sspec slcspec trspec actspec stspec) = do
    qModelSlc <- T.sample (gspec, slcspec)
    qModelTr <- T.sample (gspec, trspec)
    qModelAct <- T.sample (gspec, actspec)
    qModelSt <- T.sample (gspec, stspec)
    qModelFinal1 <- T.sample TT.LinearSpec
    qModelFinal2 <- T.sample TT.LinearSpec
    pure QModel{..}

instance
  ( gspec ~ QSpecGeneral spec
  , ((GenEmbSize gspec - 3) + 3) ~ GenEmbSize gspec
  , TT.KnownShape (PShape gspec)
  , TT.KnownShape (EShape gspec)
  )
  => T.HasForward (QModel spec) (QEncoding gspec) (QTensor '[1])
  where
  forward :: QModel spec -> QEncoding (QSpecGeneral spec) -> QTensor '[1]
  forward (QModel slc tr act st final1 final2) (QEncoding actEnc stEnc) =
    T.forward final2 $ activation $ T.forward final1 (actEmb + stEmb)
   where
    actEmb :: QTensor '[GenEmbSize gspec]
    actEmb = T.forward act (slc, tr, actEnc)
    stEmb :: QTensor '[GenEmbSize gspec]
    stEmb = T.forward st (slc, tr, stEnc)

  forwardStoch :: QModel spec -> QEncoding (QSpecGeneral spec) -> IO (QTensor '[1])
  forwardStoch model input = pure $ T.forward model input

{- | A loss for any model with 0 gradients everywhere.
Can be used to ensure that all parameters have a gradient,
if not all parameters are used in the real loss.
-}
fakeLoss
  :: forall ps
   . (ps ~ TT.Parameters (QModel DefaultQSpec))
  => QModel DefaultQSpec
  -> QTensor '[]
fakeLoss model = tzero * total
 where
  tzero :: QTensor '[]
  tzero = TT.zeros
  params = TT.flattenParameters model
  deps :: (TT.HMap' TT.ToDependent ps ys) => TT.HList ys
  deps = TT.hmap' TT.ToDependent params
  sums = TT.hmap' TH.SumAll deps
  -- total
  total = TT.hfoldr TH.Add tzero sums

mkQModel :: DefaultQSpec -> IO (QModel DefaultQSpec)
mkQModel = T.sample

loadModel :: FilePath -> IO (QModel DefaultQSpec)
loadModel path = do
  modelPlaceholder <- mkQModel defaultSpec
  tensors :: (TT.HMap' TT.ToDependent (TT.Parameters (QModel DefaultQSpec)) ts) => TT.HList ts <-
    TT.load path
  params <- TT.hmapM' TT.MakeIndependent tensors
  pure $ TT.replaceParameters modelPlaceholder params

modelSize :: QModel DefaultQSpec -> Int
modelSize model = sum $ product <$> sizes
 where
  sizes = TT.hfoldr TH.ToList ([] :: [[Int]]) $ TT.hmap' TH.ShapeVal $ TT.flattenParameters model

runQ
  :: (s -> a -> QEncoding (QSpecGeneral DefaultQSpec))
  -> QModel DefaultQSpec
  -> s
  -> a
  -> QType
runQ !encode !model s a = T.asValue $ TT.toDynamic $ T.forward model $ encode s a

runQ'
  :: (s -> a -> QEncoding (QSpecGeneral DefaultQSpec))
  -> QModel DefaultQSpec
  -> s
  -> a
  -> QTensor '[1]
runQ' !encode !model s a = T.forward model $ encode s a
