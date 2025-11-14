{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeData #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}

module RL.ModelTypes where

import GreedyParser (Action, ActionDouble (ActionDouble), ActionSingle (ActionSingle), DoubleParent (DoubleParent), GreedyState, SingleParent (SingleParent), gsOps, opGoesLeft)
import Musicology.Pitch (SPitch)
import PVGrammar

import Data.List.NonEmpty qualified as NE
import Data.TypeNums (Nat, TInt (..), type (*), type (+))
import Torch qualified as T
import Torch.Lens qualified
import Torch.Typed qualified as TT

-- global settings
-- ---------------

device :: T.Device
device = T.Device T.CPU 0
type QDevice = '(TT.CPU, 0)

-- device = T.Device T.CUDA 0
-- type QDevice = '(TT.CUDA, 0)

type QDType = TT.Double

type QType = Double

inf :: QType
inf = 1 / 0

qDType :: TT.DType
qDType = T.Double

type QTensor shape = TT.Tensor QDevice QDType shape

opts :: T.TensorOptions
opts = T.withDType qDType $ T.withDevice device T.defaultOpts

toOpts :: forall a. (Torch.Lens.HasTypes a T.Tensor) => a -> a
toOpts = T.toDevice device . T.toType qDType

toQTensor' :: QType -> T.Tensor
toQTensor' a = T.asTensor' a opts

toQTensor :: QType -> QTensor '[]
toQTensor = TT.UnsafeMkTensor . toQTensor'

type FakeSize = 1337 :: Nat

type MaxPitches = 10 :: Nat
type MaxEdges = 10 :: Nat

-- States and Actions
-- ------------------

type PVAction = Action (Notes SPitch) (Edges SPitch) (Split SPitch) (Freeze SPitch) (Spread SPitch)

type PVState = GreedyState (Edges SPitch) [Edge SPitch] (Notes SPitch) (PVLeftmost SPitch)

type PVActionResult = Either PVState (Edges SPitch, [PVLeftmost SPitch])

type PVRewardFn label = PVActionResult -> Maybe (NE.NonEmpty PVAction) -> PVAction -> label -> IO QType

-- General Spec
-- ------------

-- starts to get more efficient on GPU from ~64 on
type CommonHiddenSize = 8

type FifthLow = Neg 3
type FifthPadding = 6
type OctaveLow = (Pos 2)
type OctavePadding = 2
type EmbSize = CommonHiddenSize

type FifthSize = (2 * FifthPadding) + 1
type OctaveSize = (2 * OctavePadding) + 1

type PShape = '[FifthSize, OctaveSize]
type PSize = FifthSize + OctaveSize -- or maybe *Padding?
type EmbShape = EmbSize ': PShape

type ESize = PSize + PSize
type EShape' = '[FakeSize, ESize]

-- Specific Module Specs
-- ---------------------

type QOutHidden = CommonHiddenSize -- output module hidden size
type QSliceHidden = CommonHiddenSize -- slice encoder hidden size
type QTransHidden = CommonHiddenSize -- transition encoder hidden size
type QActionHidden = CommonHiddenSize -- action encoder hidden size
type QStateHidden = CommonHiddenSize -- state encoder hidden size
