{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE UndecidableInstances #-}

module RL.Model.Common where

import RL.ModelTypes

import Torch qualified as T
import Torch.Internal.Cast (cast2)
import Torch.Internal.Managed.Type.Tensor qualified as ATen
import Torch.Typed qualified as TT

import Control.DeepSeq (NFData)
import Data.TypeNums (Nat)
import Debug.Trace qualified as DT
import GHC.Generics (Generic)
import NoThunks.Class (NoThunks)
import System.IO.Unsafe (unsafePerformIO)

-- Global Settings
-- ===============

activation :: (IsValidDevice dev) => QTensor dev shape -> QTensor dev shape
activation = TT.gelu

-- helpers
-- =======

expandAs :: T.Tensor -> T.Tensor -> T.Tensor
expandAs t1 t2 = unsafePerformIO $ cast2 ATen.tensor_expand_as_t t1 t2

traceDyn :: TT.Tensor a b c -> TT.Tensor a b c
traceDyn t = DT.traceShow (T.shape $ TT.toDynamic t) t

unsafeReshape :: [Int] -> TT.Tensor dev dtype shape -> TT.Tensor dev dtype shape'
unsafeReshape shape t = TT.UnsafeMkTensor $ T.reshape shape $ TT.toDynamic t

-- Q net
-- =====

-- Learned Constant Embeddings
-- ---------------------------

data ConstEmbSpec dev (shape :: [Nat]) = ConstEmbSpec

newtype ConstEmb dev shape = ConstEmb (TT.Parameter dev QDType shape)
  deriving (Show, Generic)
  deriving newtype (TT.Parameterized, NFData, NoThunks)

instance
  (IsValidDevice dev, TT.TensorOptions shape QDType dev)
  => T.Randomizable (ConstEmbSpec dev shape) (ConstEmb dev shape)
  where
  sample :: ConstEmbSpec dev shape -> IO (ConstEmb dev shape)
  sample ConstEmbSpec = ConstEmb <$> (TT.makeIndependent =<< TT.randn)

instance T.HasForward (ConstEmb dev size) () (QTensor dev size) where
  forward :: ConstEmb dev size -> () -> QTensor dev size
  forward (ConstEmb emb) () = TT.toDependent emb
  forwardStoch :: ConstEmb dev size -> () -> IO (QTensor dev size)
  forwardStoch model input = pure $ T.forward model input
