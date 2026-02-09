{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}

module RL.TorchHelpers where

import Data.Kind (Type)
import GHC.TypeLits
import System.IO.Unsafe (unsafePerformIO)
import Torch qualified as T
import Torch qualified as TD
import Torch.Internal.Cast qualified as ATen
import Torch.Internal.Managed.Native qualified as ATen.Managed
import Torch.Internal.Type qualified as ATen
import Torch.Typed qualified as TT
import Torch.Typed.Auxiliary qualified

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

-- | Run a batched operation in an unbatched context
withBatchDim
  :: forall dev1 dtype1 shape1 dev2 dtype2 shape2
   . (TT.Tensor dev1 dtype1 (1 : shape1) -> TT.Tensor dev2 dtype2 (1 : shape2))
  -> TT.Tensor dev1 dtype1 shape1
  -> TT.Tensor dev2 dtype2 shape2
withBatchDim op input = TT.squeezeDim @0 $ op batchedIn
 where
  batchedIn :: TT.Tensor dev1 dtype1 (1 : shape1)
  batchedIn = TT.unsqueeze @0 input

-- | conv2d with dropped batch size constraint
conv2dRelaxed
  :: forall
    (stride :: (Nat, Nat))
    (padding :: (Nat, Nat))
    inputChannelSize
    outputChannelSize
    kernelSize0
    kernelSize1
    inputSize0
    inputSize1
    batchSize
    outputSize0
    outputSize1
    dtype
    device
   . ( TT.All
        KnownNat
        '[ Torch.Typed.Auxiliary.Fst stride
         , Torch.Typed.Auxiliary.Snd stride
         , Torch.Typed.Auxiliary.Fst padding
         , Torch.Typed.Auxiliary.Snd padding
         ]
     , TT.ConvSideCheck inputSize0 kernelSize0 (Torch.Typed.Auxiliary.Fst stride) (Torch.Typed.Auxiliary.Fst padding) outputSize0
     , TT.ConvSideCheck inputSize1 kernelSize1 (Torch.Typed.Auxiliary.Snd stride) (Torch.Typed.Auxiliary.Snd padding) outputSize1
     )
  => TT.Tensor device dtype '[outputChannelSize, inputChannelSize, kernelSize0, kernelSize1]
  -- ^ weight
  -> TT.Tensor device dtype '[outputChannelSize]
  -- ^ bias
  -> TT.Tensor device dtype '[batchSize, inputChannelSize, inputSize0, inputSize1]
  -- ^ input
  -> TT.Tensor device dtype '[batchSize, outputChannelSize, outputSize0, outputSize1]
  -- ^ output
conv2dRelaxed weight bias input =
  unsafePerformIO $
    ATen.cast7
      ATen.Managed.conv2d_tttllll
      input
      weight
      bias
      ([TT.natValI @(Torch.Typed.Auxiliary.Fst stride), TT.natValI @(Torch.Typed.Auxiliary.Snd stride)] :: [Int])
      ([TT.natValI @(Torch.Typed.Auxiliary.Fst padding), TT.natValI @(Torch.Typed.Auxiliary.Snd padding)] :: [Int])
      ([1, 1] :: [Int])
      (1 :: Int)

conv2dForwardRelaxed
  :: forall
    (stride :: (Nat, Nat))
    (padding :: (Nat, Nat))
    inputChannelSize
    outputChannelSize
    kernelSize0
    kernelSize1
    inputSize0
    inputSize1
    batchSize
    outputSize0
    outputSize1
    dtype
    device
   . ( TT.All
        KnownNat
        '[ Torch.Typed.Auxiliary.Fst stride
         , Torch.Typed.Auxiliary.Snd stride
         , Torch.Typed.Auxiliary.Fst padding
         , Torch.Typed.Auxiliary.Snd padding
         ]
     , TT.ConvSideCheck inputSize0 kernelSize0 (Torch.Typed.Auxiliary.Fst stride) (Torch.Typed.Auxiliary.Fst padding) outputSize0
     , TT.ConvSideCheck inputSize1 kernelSize1 (Torch.Typed.Auxiliary.Snd stride) (Torch.Typed.Auxiliary.Snd padding) outputSize1
     )
  => TT.Conv2d inputChannelSize outputChannelSize kernelSize0 kernelSize1 dtype device
  -> TT.Tensor device dtype '[batchSize, inputChannelSize, inputSize0, inputSize1]
  -> TT.Tensor device dtype '[batchSize, outputChannelSize, outputSize0, outputSize1]
conv2dForwardRelaxed TT.Conv2d{..} input =
  conv2dRelaxed @stride @padding
    (TT.toDependent weight)
    (TT.toDependent bias)
    input

layerNormForwardRelaxed
  :: forall normalizedShape shape dtype device
   . ( TT.KnownShape normalizedShape
     )
  => TT.LayerNorm normalizedShape dtype device
  -> TT.Tensor device dtype shape
  -> TT.Tensor device dtype shape
layerNormForwardRelaxed TT.LayerNorm{..} =
  layerNormRelaxed @normalizedShape
    (TT.toDependent layerNormWeight)
    (TT.toDependent layerNormBias)
    layerNormEps

layerNormRelaxed
  :: forall normalizedShape shape dtype device
   . ( TT.KnownShape normalizedShape
     )
  => TT.Tensor device dtype normalizedShape
  -- ^ weight
  -> TT.Tensor device dtype normalizedShape
  -- ^ bias
  -> Double
  -- ^ eps
  -> TT.Tensor device dtype shape
  -- ^ input tensor
  -> TT.Tensor device dtype shape
  -- ^ output tensor
layerNormRelaxed weight bias eps input =
  unsafePerformIO $
    ATen.cast6
      ATen.Managed.layer_norm_tlttdb
      input
      (TT.shapeVal @normalizedShape)
      weight
      bias
      eps
      ( TT.cudnnIsAcceptable weight
          && TT.cudnnIsAcceptable bias
          && TT.cudnnIsAcceptable input
      )

normalizeBatch
  :: forall device dtype batchSize inner
   . (_)
  => TT.Tensor device dtype [batchSize, inner]
  -> TT.Tensor device dtype [batchSize, inner]
normalizeBatch input = shifted `TT.div` s
 where
  mn :: TT.Tensor device dtype '[inner]
  mn = TT.mean @0 @TT.DropDim input
  shifted = input `TT.sub` mn
  s :: TT.Tensor device dtype '[inner]
  s = TT.sqrt $ TT.sumDim @0 $ TT.powScalar (2 :: Double) shifted

-- sIs0 :: TT.Tensor device TT.Bool '[inner]
-- sIs0 = s TT.==. (TT.zeros :: TT.Tensor device dtype '[])
-- s' :: TT.Tensor device dtype '[inner]
-- s' = TT.maskedFill sIs0 (1 :: Double) s

-- normalizeBatch input = normalizeBatchGrouped input TT.ones

normalizeBatchGrouped
  :: forall device dtype batchSize inner
   . ( KnownNat batchSize
     , TT.KnownDevice device
     , TT.SumDTypeIsValid device dtype
     , TT.SumDType dtype ~ dtype
     , TT.StandardFloatingPointDTypeValidation device dtype
     , TT.KnownDType dtype
     , TT.ComparisonDTypeIsValid device dtype
     , TT.BasicArithmeticDTypeIsValid device dtype
     )
  => TT.Tensor device dtype [batchSize, inner]
  -> TT.Tensor device dtype [batchSize, batchSize]
  -> TT.Tensor device dtype [batchSize, inner]
normalizeBatchGrouped input mask = shifted `TT.div` std'
 where
  mask' :: TT.Tensor device dtype '[batchSize, batchSize, 1]
  mask' = TT.reshape mask
  maskedInput :: TT.Tensor device dtype '[batchSize, batchSize, inner]
  maskedInput = TT.mul mask' input
  sum :: TT.Tensor device dtype '[batchSize, inner]
  sum = TT.sumDim @1 maskedInput
  n :: TT.Tensor device dtype '[batchSize, 1]
  n = TT.sumDim @0 mask'
  mean :: TT.Tensor device dtype '[batchSize, inner]
  mean = TT.div sum n
  shifted :: TT.Tensor device dtype '[batchSize, inner]
  shifted = input `TT.sub` mean
  maskedShifted :: TT.Tensor device dtype '[batchSize, batchSize, inner]
  maskedShifted = TT.mul mask' shifted
  std :: TT.Tensor device dtype '[batchSize, inner]
  std = TT.sqrt $ TT.sumDim @1 $ TT.powScalar (2 :: Double) maskedShifted
  stdIs0 :: TT.Tensor device 'TT.Bool '[batchSize, inner]
  stdIs0 = std TT.==. (TT.zeros :: TT.Tensor device dtype '[])
  std' :: TT.Tensor device dtype '[batchSize, inner]
  std' = TT.maskedFill stdIs0 (1 :: Double) std

checkNaN ctx t =
  if TT.toInt (TT.any $ TT.isNaN t) == 1
    then error $ "nan values in " <> show t <> "\ncontext:" <> show ctx
    else t
