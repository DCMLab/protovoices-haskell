{-# LANGUAGE DataKinds #-}
{-# LANGUAGE Strict #-}

module RL.A2CHelpers where

import Control.DeepSeq (force)
import Internal.TorchHelpers
import RL.Model
import RL.ModelTypes
import Torch.Typed qualified as TT

-- helpers for operating on HLists
-- ===============================

type ModelParams = TT.Parameters QModel
type ModelTensors = ToModelTensors ModelParams

newtype UpdateEligCritic = UpdateEligCritic QType

instance TT.Apply' UpdateEligCritic (QTensor shape, QTensor shape) (QTensor shape) where
  apply' (UpdateEligCritic factor) (zV, grad) = TT.mulScalar factor zV + grad

updateEligCritic :: QType -> QType -> TT.HList ModelTensors -> TT.HList ModelTensors -> TT.HList ModelTensors
updateEligCritic gamma lambdaV = force $ TT.hzipWith (UpdateEligCritic $ gamma * lambdaV)
{-# NOINLINE updateEligCritic #-}

data UpdateEligActor = UpdateEligActor QType QType

instance TT.Apply' UpdateEligActor (QTensor shape, QTensor shape) (QTensor shape) where
  apply' (UpdateEligActor intensity factor) (zP, grad) =
    TT.mulScalar factor zP + TT.mulScalar intensity grad

updateEligActor :: QType -> QType -> QType -> TT.HList ModelTensors -> TT.HList ModelTensors -> TT.HList ModelTensors
updateEligActor gamma lambdaP intensity =
  force $ TT.hzipWith (UpdateEligActor intensity $ gamma * lambdaP)
{-# NOINLINE updateEligActor #-}

mulModelTensors :: QTensor '[] -> TT.HList ModelTensors -> TT.HList ModelTensors
mulModelTensors factor = force $ TT.hmap' (Mul' factor)
{-# NOINLINE mulModelTensors #-}

modelZeros :: QModel -> TT.HList ModelTensors
modelZeros model = force $ TT.hmap' TT.ZerosLike $ TT.flattenParameters model
{-# NOINLINE modelZeros #-}

sumTensorList :: TT.HList ModelTensors -> QTensor '[]
sumTensorList ts = TT.hfoldr Add (TT.zeros :: QTensor '[]) $ TT.hmap' SumAll ts
