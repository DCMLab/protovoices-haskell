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

newtype UpdateEligCritic = UpdateEligCritic QType

instance (TT.KnownDevice dev) => TT.Apply' UpdateEligCritic (QTensor dev shape, QTensor dev shape) (QTensor dev shape) where
  apply' (UpdateEligCritic factor) (zV, grad) = TT.mulScalar factor zV + grad

updateEligCritic :: (TT.KnownDevice dev) => QType -> QType -> TT.HList (ModelTensors dev hidden) -> TT.HList (ModelTensors dev hidden) -> TT.HList (ModelTensors dev hidden)
updateEligCritic gamma lambdaV = force $ TT.hzipWith (UpdateEligCritic $ gamma * lambdaV)
{-# NOINLINE updateEligCritic #-}

data UpdateEligActor = UpdateEligActor QType QType

instance (TT.KnownDevice dev) => TT.Apply' UpdateEligActor (QTensor dev shape, QTensor dev shape) (QTensor dev shape) where
  apply' (UpdateEligActor intensity factor) (zP, grad) =
    TT.mulScalar factor zP + TT.mulScalar intensity grad

updateEligActor :: (TT.KnownDevice dev) => QType -> QType -> QType -> TT.HList (ModelTensors dev hidden) -> TT.HList (ModelTensors dev hidden) -> TT.HList (ModelTensors dev hidden)
updateEligActor gamma lambdaP intensity =
  force $ TT.hzipWith (UpdateEligActor intensity $ gamma * lambdaP)
{-# NOINLINE updateEligActor #-}

mulModelTensors :: (IsValidDevice dev) => QTensor dev '[] -> TT.HList (ModelTensors dev hidden) -> TT.HList (ModelTensors dev hidden)
mulModelTensors factor = force $ TT.hmap' (Mul' factor)
{-# NOINLINE mulModelTensors #-}

modelZeros :: (ValidParams dev hidden) => QModel dev hidden -> TT.HList (ModelTensors dev hidden)
modelZeros model = force $ TT.hmap' TT.ZerosLike $ TT.flattenParameters model
{-# NOINLINE modelZeros #-}

sumTensorList :: forall dev hidden. (IsValidDevice dev) => TT.HList (ModelTensors dev hidden) -> QTensor dev '[]
sumTensorList ts = TT.hfoldr Add (TT.zeros :: QTensor dev '[]) $ TT.hmap' SumAll ts
