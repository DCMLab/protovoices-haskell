{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeData #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}

module RL.ModelTypes where

import Data.TypeNums (KnownInt, KnownNat, Nat, TInt (..), intVal, intVal', type (*), type (+), type (-))
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

type family PShape' (spec :: TGeneralSpec) where
  PShape' (TGenSpec _ fs _ os _) = '[fs, os]

type family PSize (spec :: TGeneralSpec) where
  PSize (TGenSpec _ fs _ os _) = fs + os

type family PShape (spec :: TGeneralSpec) where
  PShape spec = '[FakeSize, PSize spec]

type family EShape' (spec :: TGeneralSpec) where
  EShape' (TGenSpec _ fs _ os _) = '[fs, os, fs, os]

type family ESize (spec :: TGeneralSpec) where
  ESize spec = PSize spec + PSize spec

type family EShape (spec :: TGeneralSpec) where
  EShape spec = '[FakeSize, ESize spec]

-- full model spec

type data TQSpec = TQSpecData TGeneralSpec Nat Nat Nat Nat Nat

type family QSpecGeneral qspec where
  QSpecGeneral (TQSpecData g _ _ _ _ _) = g

type family QSpecSpecial qspec where
  QSpecSpecial (TQSpecData _ s _ _ _ _) = s

type family QSpecSlice qspec where
  QSpecSlice (TQSpecData _ _ s _ _ _) = s

type family QSpecTrans qspec where
  QSpecTrans (TQSpecData _ _ _ t _ _) = t

type family QSpecAction qspec where
  QSpecAction (TQSpecData _ _ _ _ a _) = a

type family QSpecState qspec where
  QSpecState (TQSpecData _ _ _ _ _ st) = st

type TGeneralSpecDefault = TGenSpec (Neg 3) 12 (Pos 2) 5 64

type DefaultQSpec = TQSpecData TGeneralSpecDefault 64 64 64 64 128

-- type TGeneralSpecDefault = TGenSpec (Neg 3) 12 (Pos 2) 5 4

-- type DefaultQSpec = TQSpecData TGeneralSpecDefault 2 2 2 2 4
