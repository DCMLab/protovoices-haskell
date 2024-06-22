{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

module Internal.TorchHelpers where

import Data.Kind (Type)
import Torch qualified as TD
import Torch.Functional.Internal qualified as TFI
import Torch.Typed qualified as TT

-- | Helper Type to map sumAll over a HList.
data SumAll = SumAll

instance
  (dtype' ~ TT.SumDType dtype, TT.SumDTypeIsValid dev dtype)
  => TT.Apply' SumAll (TT.Tensor dev dtype shape) (TT.Tensor dev dtype' '[])
  where
  apply' _ = TT.sumAll

-- | Helper Type to fold a HList by adding the values.
data Add = Add

instance
  ( TT.BasicArithmeticDTypeIsValid dev dtype
  , TT.CheckBroadcast
      shape1
      shape2
      ( TT.ComputeBroadcast
          (TT.ReverseImpl shape1 '[])
          (TT.ReverseImpl shape2 '[])
      )
      ~ shapeOut
  )
  => TT.Apply' Add (TT.Tensor dev dtype shape1, TT.Tensor dev dtype shape2) (TT.Tensor dev dtype shapeOut)
  where
  apply' _ (a, b) = TT.add a b

-- | Helper Type to multiply a HList with a scalar
newtype Mul num
  = Mul num

instance
  (TT.Scalar num)
  => TT.Apply' (Mul num) (TT.Tensor dev dtype shape) (TT.Tensor dev dtype shape)
  where
  apply' (Mul n) = TT.mulScalar n

newtype Mul' dev dtype
  = Mul' (TT.Tensor dev dtype '[])

instance
  (shape ~ TT.Broadcast '[] shape, TT.BasicArithmeticDTypeIsValid dev dtype)
  => TT.Apply' (Mul' dev dtype) (TT.Tensor dev dtype shape) (TT.Tensor dev dtype shape)
  where
  apply' (Mul' n) = TT.mul n

-- | Detach a typed tensor.
detach :: TT.Tensor dev dtype shape -> IO (TT.Tensor dev dtype shape)
detach = fmap TT.UnsafeMkTensor . TD.detach . TT.toDynamic

-- | Helper type for combining detach and 'TT.Apply''.
data Detach = Detach

instance TT.Apply' Detach (TT.Tensor dev dtype shape) (IO (TT.Tensor dev dtype shape)) where
  apply' _ = detach

-- | Helper Type for interpolating qnet parameters.
newtype Interpolate num = Interpolate num

instance
  ( TT.Scalar num
  , Num num
  , TT.BasicArithmeticDTypeIsValid dev dtype
  , TT.CheckBroadcast
      shape
      shape
      ( TT.ComputeBroadcast
          (TT.ReverseImpl shape '[])
          (TT.ReverseImpl shape '[])
      )
      ~ shape
  )
  => TT.Apply' (Interpolate num) (TT.Tensor dev dtype shape, TT.Tensor dev dtype shape) (TT.Tensor dev dtype shape)
  where
  apply' (Interpolate tau) (p, t) = TT.mulScalar tau p `TT.add` TT.mulScalar (1 - tau) t

-- | Helper Type for getting the number of parameters in a model
data ShapeVal = ShapeVal

instance (TT.KnownShape shape) => TT.Apply' ShapeVal (TT.Tensor dev dtype shape) [Int] where
  apply' _ t = TT.shapeVal @shape

instance (TT.KnownShape shape) => TT.Apply' ShapeVal (TT.Parameter dev dtype shape) [Int] where
  apply' _ t = TT.shapeVal @shape

-- | Helper Type for getting a list out of a HList
data ToList = ToList

instance TT.Apply' ToList (t, [t]) [t] where
  apply' _ (x, xs) = x : xs

-- -- | Helper Type for getting zeros like the parameters of a model
-- data ZerosLike = ZerosLike

-- instance TT.Apply' ZerosLike (TT.Tensor dev dtype shape) (TT.Tensor dev dtype shape) where
--   apply' _ = TT.zerosLike

type family ToModelTensors (params :: [Type]) :: [Type] where
  ToModelTensors '[] = '[]
  ToModelTensors (TT.Parameter dev dtype shape ': rst) = TT.Tensor dev dtype shape : ToModelTensors rst
