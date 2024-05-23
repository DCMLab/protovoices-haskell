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
import GreedyParser (Action, ActionDouble (ActionDouble), ActionSingle (ActionSingle), GreedyState, Trans (Trans), getActions, initParseState, parseGreedy, parseStep, pickRandom)
import Inference.Conjugate (HyperRep, Prior (expectedProbs), evalTraceLogP, sampleProbs)
import Internal.MultiSet qualified as MS
import Internal.TorchHelpers qualified as TH
import Musicology.Pitch
import PVGrammar (Edge, Edges (Edges), Freeze (FreezeOp), InnerEdge, Notes (Notes), PVAnalysis, PVLeftmost, Split, Spread)
import PVGrammar.Generate (derivationPlayerPV)
import PVGrammar.Parse (protoVoiceEvaluator)
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
pitchesOneHot ps = TT.UnsafeMkTensor $ T.indexPut True indices values zeros
 where
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
  => GeneralSpec spec
  -> Notes SPitch
  -> QTensor (PShape spec)
encodeSlice genspec (Notes notes) =
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

edgesOneHot
  :: forall (spec :: TGeneralSpec)
   . ( KnownInt (GenFifthLow spec)
     , KnownInt (GenOctaveLow spec)
     , KnownNat (GenFifthSize spec)
     , KnownNat (GenOctaveSize spec)
     )
  => HS.HashSet (InnerEdge SPitch)
  -> QTensor (EShape spec)
edgesOneHot es = TT.UnsafeMkTensor $ T.indexPut True indices values zeros
 where
  edge2index (p1, p2) =
    pitch2index @(GenFifthLow spec) @(GenOctaveLow spec) p1
      ++ pitch2index @(GenFifthLow spec) @(GenOctaveLow spec) p2
  indices = fmap T.asTensor $ T.asValue @[[Int]] $ T.transpose2D $ T.asTensor (edge2index <$> F.toList es)
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
  { actionEncodingTop :: Either (SingleTop spec) (DoubleTop spec)
  , actionEncodingOp :: Leftmost () () ()
  }

encodePVAction
  :: forall (spec :: TGeneralSpec)
   . ( KnownInt (GenFifthLow spec)
     , KnownInt (GenOctaveLow spec)
     , KnownNat (GenFifthSize spec)
     , KnownNat (GenOctaveSize spec)
     )
  => GeneralSpec spec
  -> PVAction
  -> ActionEncoding spec
encodePVAction spec (Left (ActionSingle top action)) = ActionEncoding encTop encAction
 where
  (sl, GreedyParser.Trans t _2nd, sr) = top
  encTop = Left (encodeSlice @spec spec <$> sl, encodeTransition @spec t, encodeSlice @spec spec <$> sr)
  encAction = case action of
    LMSingleFreeze FreezeOp -> LMFreezeOnly ()
    LMSingleSplit _split -> LMSplitOnly ()
encodePVAction spec (Right (ActionDouble top action)) = ActionEncoding encTop encAction
 where
  (sl, GreedyParser.Trans t1 _, sm, Trans t2 _, sr) = top
  encTop =
    Right
      ( encodeSlice @spec spec <$> sl
      , encodeTransition @spec t1
      , encodeSlice @spec spec sm
      , encodeTransition @spec t2
      , encodeSlice @spec spec <$> sr
      )
  encAction = case action of
    LMDoubleFreezeLeft FreezeOp -> LMFreezeLeft ()
    LMDoubleSplitLeft split -> LMSplitLeft ()
    LMDoubleSplitRight split -> LMSplitRight ()
    LMDoubleSpread spread -> LMSpread ()

-- Step Encoding
-- -------------

newtype QEncoding spec = QEncoding
  { _actionEncoding :: ActionEncoding spec
  }

encodeStep
  :: GeneralSpec TGeneralSpecDefault
  -> p
  -> PVAction
  -> QEncoding TGeneralSpecDefault
encodeStep spec _ action = QEncoding (encodePVAction @TGeneralSpecDefault spec action)

-- Q net
-- =====

-- General Spec
-- ------------

data GeneralSpec (spec :: TGeneralSpec) = GeneralSpec

defaultGSpec :: GeneralSpec spec
defaultGSpec =
  GeneralSpec

type data TGeneralSpec = TGenSpec TInt Nat TInt Nat Nat

type TGeneralSpecDefault = TGenSpec (Neg 3) 12 (Pos 2) 5 16

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
      >>> TT.relu
      >>> TT.forward l2
      >>> TT.relu
  forwardStoch model = pure . T.forward model

-- Transition Encoder
-- ------------------

data TransitionSpec (hidden :: Nat) = TransitionSpec

data TransitionEncoder spec hidden = TransitionEncoder
  { -- TODO: should there be a separate trL1Pass?
    trL1Inner :: !(TT.Linear (TT.Product (EShape spec)) hidden QDType QDevice)
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
  forward TransitionEncoder{..} TransitionEncoding{..} = TT.relu $ T.forward trL2 all'
   where
    pass = TT.relu $ T.forward trL1Inner $ TT.flattenAll trencPassing
    inner = TT.relu $ T.forward trL1Inner $ TT.flattenAll trencInner
    left = TT.relu $ T.forward trL1Left $ TT.flattenAll trencLeft
    right = TT.relu $ T.forward trL1Right $ TT.flattenAll trencRight
    all = pass + inner + left + right
    all' = if trencRoot then all + TT.relu (T.forward trL1Root ()) else all
  forwardStoch = undefined

-- ActionEncoder
-- -------------

data ActionSpec (hidden :: Nat) = ActionSpec

data ActionEncoder spec hidden = ActionEncoder
  { actTop1_2 :: TT.Linear (GenEmbSize spec + GenEmbSize spec) hidden QDType QDevice
  , actTop1_3 :: TT.Linear (GenEmbSize spec + (GenEmbSize spec + GenEmbSize spec)) hidden QDType QDevice
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
  , KnownNat (emb + emb)
  , KnownNat (emb + (emb + emb))
  , KnownNat (emb - 3)
  )
  => T.Randomizable (GeneralSpec spec, ActionSpec hidden) (ActionEncoder spec hidden)
  where
  sample :: (GeneralSpec spec, ActionSpec hidden) -> IO (ActionEncoder spec hidden)
  sample (GeneralSpec, ActionSpec) =
    ActionEncoder
      <$> T.sample TT.LinearSpec
      <*> T.sample TT.LinearSpec
      <*> T.sample TT.LinearSpec
      <*> T.sample ConstEmbSpec
      <*> T.sample ConstEmbSpec
      <*> T.sample ConstEmbSpec

instance
  forall (spec :: TGeneralSpec) slcHidden actHidden outShape emb
   . ( emb ~ GenEmbSize spec
     , outShape ~ '[emb + emb]
     , emb ~ (emb - 3) + 3
     , TT.KnownShape (PShape spec)
     )
  => T.HasForward (ActionEncoder spec actHidden) (SliceEncoder spec slcHidden, ActionEncoding spec) (QTensor outShape)
  where
  forward ActionEncoder{..} (slc, ActionEncoding top op) = TT.cat @0 (topEmb TT.:. opEmb TT.:. TT.HNil)
   where
    hidden :: QTensor '[actHidden]
    hidden = case top of
      Left (sl, _t, sr) -> T.forward actTop1_2 $ TT.cat @0 (T.forward slc sl TT.:. T.forward slc sr TT.:. TT.HNil)
      Right (sl, _t1, sm, _t2, sr) ->
        let
          embl = T.forward slc sl
          embm = T.forward slc sm
          embr = T.forward slc sr
          slcEmb = TT.cat @0 (embl TT.:. embm TT.:. embr TT.:. TT.HNil)
         in
          T.forward actTop1_3 slcEmb
    topEmb :: QTensor '[GenEmbSize spec]
    topEmb = TT.relu $ T.forward actTop2 $ TT.relu hidden
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

-- Full Model
-- ----------

data SpecialSpec (hidden :: Nat) = SpecialSpec

data QSpec (spec :: TGeneralSpec) specialSpec sliceSpec actionSpec
  = QSpec (GeneralSpec spec) (SpecialSpec specialSpec) (SliceSpec sliceSpec) (ActionSpec actionSpec)

type family QSpecGeneral qspec where
  QSpecGeneral (QSpec g _ _ _) = g

type family QSpecSpecial qspec where
  QSpecSpecial (QSpec _ s _ _) = s

type family QSpecSlice qspec where
  QSpecSlice (QSpec _ _ s _) = s

type family QSpecAction qspec where
  QSpecAction (QSpec _ _ _ a) = a

type DefaultQSpec = QSpec TGeneralSpecDefault 32 32 32

defaultSpec :: DefaultQSpec
defaultSpec =
  QSpec
    defaultGSpec
    SpecialSpec
    SliceSpec
    ActionSpec

data QModel spec = QModel
  { qModelSlc :: !(SliceEncoder (QSpecGeneral spec) (QSpecSlice spec))
  , qModelAct :: !(ActionEncoder (QSpecGeneral spec) (QSpecAction spec))
  , qModelFinal1 :: !(TT.Linear (GenEmbSize (QSpecGeneral spec) + GenEmbSize (QSpecGeneral spec)) (QSpecSpecial spec) QDType QDevice)
  , qModelFinal2 :: !(TT.Linear (QSpecSpecial spec) 1 QDType QDevice)
  }
  deriving (Show, Generic, TT.Parameterized)

instance
  ( spec ~ QSpec g sp sl ac
  , embsize ~ GenEmbSize g
  , KnownNat sp
  , KnownNat sl
  , KnownNat ac
  , KnownNat embsize
  , KnownNat (embsize - 3)
  , KnownNat (embsize + embsize)
  , KnownNat (embsize + (embsize + embsize))
  , KnownNat (TT.Product (PShape g))
  )
  => T.Randomizable (QSpec g sp sl ac) (QModel spec)
  where
  sample :: spec -> IO (QModel spec)
  sample (QSpec gspec sspec slcspec actspec) =
    QModel
      <$> T.sample (gspec, slcspec)
      <*> T.sample (gspec, actspec)
      <*> T.sample TT.LinearSpec
      <*> T.sample TT.LinearSpec

instance
  ( gspec ~ QSpecGeneral spec
  , ((GenEmbSize gspec - 3) + 3) ~ GenEmbSize gspec
  , TT.KnownShape (PShape gspec)
  )
  => T.HasForward (QModel spec) (QEncoding gspec) (QTensor '[1])
  where
  forward :: QModel spec -> QEncoding (QSpecGeneral spec) -> QTensor '[1]
  forward (QModel slc act final1 final2) (QEncoding actEnc) =
    T.forward final2 $ TT.relu $ T.forward final1 actEmb
   where
    actEmb :: QTensor '[GenEmbSize gspec + GenEmbSize gspec]
    actEmb = T.forward act (slc, actEnc)

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
