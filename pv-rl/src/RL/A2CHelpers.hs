{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE Strict #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}

module RL.A2CHelpers where

import Control.DeepSeq (force)
import RL.Model.Interface
import RL.ModelTypes
import RL.TorchHelpers
import Torch.Typed qualified as TT

-- helpers for operating on HLists
-- ===============================

newtype UpdateEligCritic dev = UpdateEligCritic QType

instance
  (TT.KnownDevice dev)
  => TT.Apply' (UpdateEligCritic dev) (QTensor dev shape, QTensor dev shape) (QTensor dev shape)
  where
  apply' (UpdateEligCritic factor) (zV, grad) = TT.mulScalar factor zV + grad

updateEligCritic
  :: forall dev model
   . (TT.KnownDevice dev, _)
  => QType
  -> QType
  -> TT.HList (ModelTensors model dev)
  -> TT.HList (ModelTensors model dev)
  -> TT.HList (ModelTensors model dev)
updateEligCritic gamma lambdaV = force $ TT.hzipWith (UpdateEligCritic @dev $ gamma * lambdaV)
{-# NOINLINE updateEligCritic #-}

data UpdateEligActor = UpdateEligActor QType QType

instance
  (TT.KnownDevice dev)
  => TT.Apply' UpdateEligActor (QTensor dev shape, QTensor dev shape) (QTensor dev shape)
  where
  apply' (UpdateEligActor intensity factor) (zP, grad) =
    TT.mulScalar factor zP + TT.mulScalar intensity grad

updateEligActor
  :: forall dev model
   . (TT.KnownDevice dev, _)
  => QType
  -> QType
  -> QType
  -> TT.HList (ModelTensors model dev)
  -> TT.HList (ModelTensors model dev)
  -> TT.HList (ModelTensors model dev)
updateEligActor gamma lambdaP intensity =
  force $ TT.hzipWith (UpdateEligActor intensity $ gamma * lambdaP)
{-# NOINLINE updateEligActor #-}

mulModelTensors
  :: forall dev model
   . (IsValidDevice dev, _)
  => QTensor dev '[]
  -> TT.HList (ModelTensors model dev)
  -> TT.HList (ModelTensors model dev)
mulModelTensors factor = force $ TT.hmap' (Mul' factor)
{-# NOINLINE mulModelTensors #-}

modelZeros
  :: (_)
  => model dev
  -> TT.HList (ModelTensors model dev)
modelZeros model = force $ TT.hmap' TT.ZerosLike $ TT.flattenParameters model
{-# NOINLINE modelZeros #-}

sumTensorList
  :: forall dev model
   . (TT.KnownDevice dev, _)
  => TT.HList (ModelTensors model dev)
  -> QTensor dev '[]
sumTensorList ts = TT.hfoldr Add (TT.zeros :: QTensor dev '[]) $ TT.hmap' SumAll ts
