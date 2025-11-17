{-# LANGUAGE DataKinds #-}
{-# LANGUAGE Strict #-}

module RL.A2CHelpers where

import Control.DeepSeq (force)
import RL.Model
import RL.ModelTypes
import RL.TorchHelpers
import Torch.Typed qualified as TT

-- helpers for operating on HLists
-- ===============================

type ModelParams dev = TT.Parameters (QModel dev)
type ModelTensors dev = ToModelTensors (ModelParams dev)

newtype UpdateEligCritic = UpdateEligCritic QType

instance (TT.KnownDevice dev) => TT.Apply' UpdateEligCritic (QTensor dev shape, QTensor dev shape) (QTensor dev shape) where
  apply' (UpdateEligCritic factor) (zV, grad) = TT.mulScalar factor zV + grad

updateEligCritic :: (TT.KnownDevice dev) => QType -> QType -> TT.HList (ModelTensors dev) -> TT.HList (ModelTensors dev) -> TT.HList (ModelTensors dev)
updateEligCritic gamma lambdaV = force $ TT.hzipWith (UpdateEligCritic $ gamma * lambdaV)
{-# NOINLINE updateEligCritic #-}

data UpdateEligActor = UpdateEligActor QType QType

instance (TT.KnownDevice dev) => TT.Apply' UpdateEligActor (QTensor dev shape, QTensor dev shape) (QTensor dev shape) where
  apply' (UpdateEligActor intensity factor) (zP, grad) =
    TT.mulScalar factor zP + TT.mulScalar intensity grad

updateEligActor :: (TT.KnownDevice dev) => QType -> QType -> QType -> TT.HList (ModelTensors dev) -> TT.HList (ModelTensors dev) -> TT.HList (ModelTensors dev)
updateEligActor gamma lambdaP intensity =
  force $ TT.hzipWith (UpdateEligActor intensity $ gamma * lambdaP)
{-# NOINLINE updateEligActor #-}

mulModelTensors :: (IsValidDevice dev) => QTensor dev '[] -> TT.HList (ModelTensors dev) -> TT.HList (ModelTensors dev)
mulModelTensors factor = force $ TT.hmap' (Mul' factor)
{-# NOINLINE mulModelTensors #-}

modelZeros :: (IsValidDevice dev) => QModel dev -> TT.HList (ModelTensors dev)
modelZeros model = force $ TT.hmap' TT.ZerosLike $ TT.flattenParameters model
{-# NOINLINE modelZeros #-}

sumTensorList :: forall dev. (IsValidDevice dev) => TT.HList (ModelTensors dev) -> QTensor dev '[]
sumTensorList ts = TT.hfoldr Add (TT.zeros :: QTensor dev '[]) $ TT.hmap' SumAll ts
