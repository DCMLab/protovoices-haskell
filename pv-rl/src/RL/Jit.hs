{-# LANGUAGE DataKinds #-}
{-# OPTIONS_GHC -Wno-unused-imports #-}

module RL.Jit where

import RL.Encoding
import RL.Model
import RL.ModelTypes

import Data.TypeNums (KnownNat)
import Torch qualified as T
import Torch.Jit qualified as TJit
import Torch.Lens qualified as TL

-- compileBatchedPolicy
--   :: forall dev hidden bs
--    . ( ValidParams dev hidden
--      , KnownNat bs
--      )
--   => TJit.ScriptCache
--   -> QType
--   -> QModel dev hidden
--   -> QEncoding dev '[bs]
--   -> T.Tensor
-- compileBatchedPolicy scriptCache temp model encoding =
--   case TJit.jit scriptCache policy $ TL.flattenValues TL.types (model, encoding) of
--     [] -> error "Jit model didn't return any tensors"
--     (res : _) -> res
--  where
--   policy :: [T.Tensor] -> [T.Tensor]
--   policy tensors = [runBatchedPolicy temp model' encoding']
--    where
--     (model', encoding') = TL.replaceValues TL.types (model, encoding) tensors
