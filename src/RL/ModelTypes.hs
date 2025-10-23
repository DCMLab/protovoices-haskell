{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeData #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}

module RL.ModelTypes where

import Data.TypeNums (Nat, TInt (..), type (*), type (+))
import Torch qualified as T
import Torch.Lens qualified
import Torch.Typed qualified as TT

-- global settings
-- ---------------

device :: T.Device
-- device = T.Device T.CPU 0
-- type QDevice = '(TT.CPU, 0)

device = T.Device T.CUDA 0
type QDevice = '(TT.CUDA, 0)

type QDType = TT.Double

type QType = Double

qDType :: TT.DType
qDType = T.Double

type QTensor shape = TT.Tensor QDevice QDType shape

opts :: T.TensorOptions
opts = T.withDType qDType $ T.withDevice device T.defaultOpts

toOpts :: forall a. (Torch.Lens.HasTypes a T.Tensor) => a -> a
toOpts = T.toDevice device . T.toType qDType

type FakeSize = 1337 :: Nat

type MaxPitches = 10 :: Nat
type MaxEdges = 10 :: Nat

-- General Spec
-- ------------

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
