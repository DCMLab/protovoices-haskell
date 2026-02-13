{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE UndecidableInstances #-}

module RL.Model.Slice where

import RL.Encoding
import RL.Model.Common
import RL.ModelTypes

import Torch qualified as T
import Torch.Typed qualified as TT

import Control.DeepSeq (NFData)
import Data.TypeNums (KnownNat)
import GHC.Generics (Generic)
import NoThunks.Class (NoThunks)

-- Slice Encoder
-- -------------

data SliceSpec dev hidden = SliceSpec

data SliceEncoder dev hidden = SliceEncoder
  { _slcL1 :: !(TT.Conv2d 1 hidden 1 1 QDType dev) -- !(TT.Linear (PSize spec) hidden QDType QDevice)
  , _slcL2 :: !(TT.Conv2d hidden hidden FifthSize OctaveSize QDType dev) -- !(TT.Linear hidden (EmbSize spec) QDType QDevice)
  , _slcStart :: !(ConstEmb dev (hidden : PShape))
  , _slcStop :: !(ConstEmb dev (hidden : PShape))
  , _slcNorm1 :: !(TT.LayerNorm (hidden : PShape) QDType dev)
  , _slcNorm2 :: !(TT.LayerNorm (hidden : PShape) QDType dev)
  -- TODO: learn embedding for empty slice
  }
  deriving (Show, Generic, TT.Parameterized, NoThunks, NFData)

instance (IsValidDevice dev, KnownNat hidden) => T.Randomizable (SliceSpec dev hidden) (SliceEncoder dev hidden) where
  sample :: SliceSpec dev hidden -> IO (SliceEncoder dev hidden)
  sample _ =
    SliceEncoder
      <$> T.sample TT.Conv2dSpec
      <*> T.sample TT.Conv2dSpec
      <*> T.sample (ConstEmbSpec @dev)
      <*> T.sample (ConstEmbSpec @dev)
      <*> T.sample (TT.LayerNormSpec 1e-05)
      <*> T.sample (TT.LayerNormSpec 1e-05)

-- | HasFoward for slice (unbatched)
instance
  (embshape ~ hidden : PShape, IsValidDevice dev, IsValidHidden hidden)
  => T.HasForward (SliceEncoder dev hidden) (SliceEncoding dev '[]) (QTensor dev embshape)
  where
  forward (SliceEncoder l1 l2 _ _ n1 n2) slice = TT.squeezeDim @0 out2
   where
    input = TT.unsqueeze @0 $ TT.unsqueeze @0 $ getSlice slice
    out1 :: QTensor dev (1 : hidden : PShape)
    out1 = activation $ TT.layerNormForward n1 $ TT.conv2dForward @'(1, 1) @'(0, 0) l1 input
    out2 :: QTensor dev (1 : hidden : PShape)
    out2 = activation $ TT.layerNormForward n2 $ TT.conv2dForward @'(1, 1) @'(FifthPadding, OctavePadding) l2 out1
  forwardStoch model = pure . T.forward model

-- | HasFoward for slice (batched)
instance
  ( ValidParams dev hidden
  , embshape ~ '[batchSize, hidden, FifthSize, OctaveSize]
  , KnownNat batchSize
  )
  => T.HasForward (SliceEncoder dev hidden) (SliceEncoding dev '[batchSize]) (QTensor dev embshape)
  where
  forward (SliceEncoder l1 l2 _ _ n1 n2) slice = out2
   where
    input = TT.unsqueeze @1 $ getSlice slice
    out1 :: QTensor dev '[batchSize, hidden, FifthSize, OctaveSize]
    out1 = activation $ TT.layerNormForward n1 $ TT.conv2dForward @'(1, 1) @'(0, 0) l1 input
    out2 :: QTensor dev '[batchSize, hidden, FifthSize, OctaveSize]
    out2 = activation $ TT.layerNormForward n2 $ TT.conv2dForward @'(1, 1) @'(FifthPadding, OctavePadding) l2 out1
  forwardStoch model = pure . T.forward model

-- | HasForward for slice wrappend in QStartStop (unbatched).
instance
  (embshape ~ hidden : PShape, ValidParams dev hidden)
  => TT.HasForward (SliceEncoder dev hidden) (QStartStop dev '[] (SliceEncoding dev '[])) (QTensor dev embshape)
  where
  forward model@(SliceEncoder _ _ start stop _ _) (QStartStop tag input) = TT.squeezeDim @0 out
   where
    -- compute the possible outputs for start/stop/inner
    outStart :: QTensor dev (hidden : PShape)
    outStart = activation $ TT.forward start ()
    outStop :: QTensor dev (hidden : PShape)
    outStop = activation $ TT.forward stop ()
    outInner :: QTensor dev (hidden : PShape)
    outInner = T.forward model input
    -- combine the outputs into one tensor
    combined :: QTensor dev (3 : hidden : PShape)
    combined = TT.stack @0 $ outStart TT.:. outInner TT.:. outStop TT.:. TT.HNil
    -- use gather to select the right output.
    -- gather can select different elements from 'dim' for each position,
    -- so we expand the tag to the right shape, selecting the *same* 'dim'-index everywhere
    tag' :: TT.Tensor dev TT.Int64 (1 : hidden : PShape)
    tag' = TT.expand False $ TT.reshape @[1, 1, 1, 1] tag
    out :: QTensor dev (1 : hidden : PShape)
    out = TT.gatherDim @0 tag' combined
  forwardStoch model input = pure $ T.forward model input

-- | HasForward for slice wrapped in QStartStop (batched).
instance
  ( ValidParams dev hidden
  , embshape ~ (batchSize : hidden : PShape)
  , KnownNat batchSize
  )
  => TT.HasForward (SliceEncoder dev hidden) (QStartStop dev '[batchSize] (SliceEncoding dev '[batchSize])) (QTensor dev embshape)
  where
  forward model@(SliceEncoder _ _ start stop _ _) (QStartStop tag input) = TT.squeezeDim @0 out
   where
    -- compute the possible outputs for start/stop/inner
    outStart :: QTensor dev (batchSize : hidden : PShape)
    outStart = activation $ TT.UnsafeMkTensor $ expandAs (TT.toDynamic $ TT.forward start ()) $ TT.toDynamic outInner
    outStop :: QTensor dev (batchSize : hidden : PShape)
    outStop = activation $ TT.UnsafeMkTensor $ expandAs (TT.toDynamic $ TT.forward stop ()) $ TT.toDynamic outInner
    outInner :: QTensor dev (batchSize : hidden : PShape)
    outInner = T.forward model input
    -- combine the outputs into one tensor
    combined :: QTensor dev (3 : batchSize : hidden : PShape)
    combined = TT.stack @0 $ outStart TT.:. outInner TT.:. outStop TT.:. TT.HNil
    -- use gather to select the right output.
    -- gather can select different elements from 'dim' for each position,
    -- so we expand the tag to the right shape, selecting the *same* 'dim'-index everywhere
    tag' :: TT.Tensor dev 'TT.Int64 (1 : batchSize : hidden : PShape)
    tag' =
      TT.UnsafeMkTensor
        $ T.unsqueeze (T.Dim 0)
        $ expandAs
          (T.reshape [-1, 1, 1, 1] $ TT.toDynamic tag)
        $ TT.toDynamic outInner
    out = TT.gatherDim @0 tag' combined
  forwardStoch model input = pure $ T.forward model input
