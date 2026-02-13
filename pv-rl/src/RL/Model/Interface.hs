{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ExplicitNamespaces #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}

module RL.Model.Interface where

import RL.Encoding
import RL.ModelTypes

import RL.TorchHelpers qualified as TH
import Torch qualified as T
import Torch.Typed qualified as TT

import Data.TypeNums (KnownNat)

-- general model interface
-- =======================

-- describing and manipulating models
-- ----------------------------------

type ModelParams model dev = TT.Parameters (model dev)
type ModelTensors model dev = TH.ToModelTensors (ModelParams model dev)

-- | Returns the total number of parameters in a model
modelSize :: (_) => model -> Int
modelSize model = sum $ product <$> sizes
 where
  sizes = TT.hfoldr TH.ToList ([] :: [[Int]]) $ TT.hmap' TH.ShapeVal $ TT.flattenParameters model

saveModel :: (_) => FilePath -> model -> IO ()
saveModel path model = TT.save (TT.hmap' TT.ToDependent $ TT.flattenParameters model) path

-- helpers
-- ------

-- | An existential type that contains a policy of dynamic size.
data SomePolicy dev = forall batchSize. (KnownNat batchSize) => SomePolicy (QTensor dev '[batchSize, 1])

-- | Turn a 'SomePolicy' into a dynamic tensor.
dynPolicy :: SomePolicy dev -> T.Tensor
dynPolicy (SomePolicy p) = TT.toDynamic p

{- | A loss for any model with 0 gradients everywhere.
Can be used to ensure that all parameters have a gradient,
if not all parameters are used in the real loss.
-}
fakeLoss
  :: forall model dev
   . (IsValidDevice dev, _)
  => model dev
  -> QTensor dev '[]
fakeLoss model = tzero * total
 where
  tzero :: QTensor dev '[]
  tzero = TT.zeros
  params :: TT.HList (ModelParams model dev)
  params = TT.flattenParameters model
  deps :: TT.HList (ModelTensors model dev) -- (TT.HMap' TT.ToDependent ps ys) => TT.HList ys
  deps = TT.hmap' TT.ToDependent params
  sums = TT.hmap' TH.SumAll deps
  -- total
  total = TT.hfoldr TH.Add tzero sums

-- raw forward passes
-- ------------------

-- | Forward pass (without activation) of the policy (single action)
forwardPolicy
  :: (TT.HasForward model (QEncoding dev '[]) (QTensor dev '[1]))
  => model
  -> QEncoding dev '[]
  -> QTensor dev '[1]
forwardPolicy = TT.forward

-- | Forward pass (without activation) of the policy (batched action)
forwardPolicyBatched
  :: forall model dev batchSize
   . (TT.HasForward model (QEncoding dev '[batchSize]) (QTensor dev [batchSize, 1]))
  => model
  -> QEncoding dev '[batchSize]
  -> QTensor dev '[batchSize, 1]
forwardPolicyBatched = TT.forward

-- | Forward pass (without activation) of the policy (fully batched)
forwardPolicyFullyBatched
  :: forall model dev
   . (TT.HasForward model (QEncodingBatch dev) [T.Tensor])
  => model
  -> QEncodingBatch dev
  -> [T.Tensor] -- TODO: could be changed to SomePolicy?
forwardPolicyFullyBatched = TT.forward

-- | Forward pass (without activation) of the state value
forwardValue
  :: (TT.HasForward model (StateEncoding dev) (QTensor dev '[1]))
  => model
  -> StateEncoding dev
  -> QTensor dev '[1]
forwardValue = TT.forward

-- running the model
-- -----------------

-- | Returns the Q-value of an action in a state as a 'QType'.
runQ
  :: (_)
  => model
  -> QEncoding dev '[]
  -> QType
runQ model enc = T.asValue $ TT.toDynamic $ runQ' model enc

-- | Returns the Q-value of an action in a state as a tensor.
runQ'
  :: (_)
  => model
  -> QEncoding dev '[]
  -> QTensor dev '[1]
runQ' = forwardPolicy

-- | Returns the V-value of a state.
runV
  :: (TT.HasForward model (StateEncoding dev) (QTensor dev '[1]))
  => model
  -> StateEncoding dev
  -> QTensor dev '[1]
runV = forwardValue

-- | Returns the Q-values for a batch of actions in a state.
runBatchedQ
  :: forall model dev batchSize
   . ( KnownNat batchSize
     , TT.HasForward model (QEncoding dev '[batchSize]) (QTensor dev [batchSize, 1])
     )
  => model
  -> QEncoding dev '[batchSize]
  -> SomePolicy dev
runBatchedQ actor encoding = SomePolicy $ forwardPolicyBatched actor encoding

-- | Returns the policy after temperature and softmax for a batch of actions in a state.
runBatchedPolicy
  :: forall model dev batchSize
   . (_)
  => QType
  -> model
  -> QEncoding dev '[batchSize]
  -> SomePolicy dev
runBatchedPolicy temp actor encoding = case runBatchedQ actor encoding of
  SomePolicy policy -> SomePolicy $ TT.softmax @0 $ TT.mulScalar (1 / temp) policy

-- | Returns the log-policy after temperature and softmax for a batch of actions in a state.
runBatchedLogPolicy
  :: forall model dev batchSize
   . (_)
  => QType
  -> model
  -> QEncoding dev '[batchSize]
  -> SomePolicy dev
runBatchedLogPolicy temp actor encoding = case runBatchedQ actor encoding of
  SomePolicy policy -> SomePolicy $ TT.logSoftmax @0 $ TT.mulScalar (1 / temp) policy

-- | Returns the Q-values for a full batch of actions and states.
runFullyBatchedQ
  :: forall model dev
   . (_)
  => model
  -> QEncodingBatch dev
  -> [T.Tensor]
runFullyBatchedQ = forwardPolicyFullyBatched

-- | Returns the policies after temperature and softmax for a batch of actions and states.
runFullyBatchedPolicy
  :: forall model dev
   . (_)
  => QType
  -> model
  -> QEncodingBatch dev
  -> [T.Tensor]
runFullyBatchedPolicy temp model batch = fmap activate $ runFullyBatchedQ model batch
 where
  activate = T.softmax (T.Dim 0) . T.mulScalar (1 / temp)

-- | Returns the log-policies after temperature and softmax for a batch of actions and states.
runFullyBatchedLogPolicy
  :: forall model dev
   . (_)
  => QType
  -> model
  -> QEncodingBatch dev
  -> [T.Tensor]
runFullyBatchedLogPolicy temp model batch = fmap activate $ runFullyBatchedQ model batch
 where
  activate = T.logSoftmax (T.Dim 0) . T.mulScalar (1 / temp)
