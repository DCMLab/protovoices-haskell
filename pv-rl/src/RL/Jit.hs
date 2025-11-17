{-# LANGUAGE DataKinds #-}

module RL.Jit where

import RL.Encoding
import RL.Model

import Data.TypeNums (KnownNat)
import RL.ModelTypes (IsValidDevice)
import Torch qualified as T
import Torch.Jit qualified as TJit
import Torch.Lens qualified as TL

compileBatchedPolicy :: forall dev bs. (IsValidDevice dev, KnownNat bs) => TJit.ScriptCache -> QModel dev -> QEncoding dev '[bs] -> T.Tensor
compileBatchedPolicy scriptCache model encoding =
  head $ TJit.jit scriptCache policy $ TL.flattenValues TL.types (model, encoding)
 where
  policy :: [T.Tensor] -> [T.Tensor]
  policy tensors = [runBatchedPolicy model' encoding']
   where
    (model', encoding') = TL.replaceValues TL.types (model, encoding) tensors
