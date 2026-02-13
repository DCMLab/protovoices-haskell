{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE ExplicitNamespaces #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}

module RL.Model.Transition where

import RL.Encoding
import RL.Model.Common
import RL.ModelTypes

import Torch qualified as T
import Torch.Typed qualified as TT

import Control.DeepSeq (NFData)
import Data.TypeNums (KnownNat, type (*))
import GHC.Generics (Generic)
import NoThunks.Class (NoThunks)

-- Transition Encoder
-- ------------------

data TransitionSpec dev hidden = TransitionSpec

data TransitionEncoder dev hidden = TransitionEncoder
  { trL1Passing :: !(TT.Conv2d 2 hidden FifthSize OctaveSize QDType dev)
  , trL1Inner :: !(TT.Conv2d 2 hidden FifthSize OctaveSize QDType dev)
  , trL1Left :: !(TT.Conv2d 1 hidden 1 1 QDType dev)
  , trL1Right :: !(TT.Conv2d 1 hidden 1 1 QDType dev)
  , trL1Root :: !(ConstEmb dev '[hidden])
  , trL2 :: !(TT.Conv2d hidden hidden FifthSize OctaveSize QDType dev)
  , trNorm1 :: !(TT.LayerNorm (hidden : PShape) QDType dev)
  , trNorm2 :: !(TT.LayerNorm (hidden : PShape) QDType dev)
  }
  deriving (Show, Generic, TT.Parameterized, NoThunks, NFData)

instance (IsValidDevice dev, KnownNat hidden) => T.Randomizable (TransitionSpec dev hidden) (TransitionEncoder dev hidden) where
  sample :: TransitionSpec dev hidden -> IO (TransitionEncoder dev hidden)
  sample _ = do
    trL1Passing <- T.sample TT.Conv2dSpec
    trL1Inner <- T.sample TT.Conv2dSpec
    trL1Left <- T.sample TT.Conv2dSpec
    trL1Right <- T.sample TT.Conv2dSpec
    trL1Root <- T.sample $ ConstEmbSpec @dev
    trL2 <- T.sample TT.Conv2dSpec
    trNorm1 <- T.sample $ TT.LayerNormSpec 1e-05
    trNorm2 <- T.sample $ TT.LayerNormSpec 1e-05
    pure $ TransitionEncoder{..}

-- | HasForward for transitions (unbatched)
instance
  forall dev hidden embshape
   . ( ValidParams dev hidden
     , embshape ~ (hidden : PShape)
     )
  => T.HasForward (TransitionEncoder dev hidden) (TransitionEncoding dev '[]) (QTensor dev embshape)
  where
  forward TransitionEncoder{..} TransitionEncoding{..} =
    activation $
      TT.layerNormForward trNorm2 $
        TT.squeezeDim @0 $
          TT.conv2dForward @'(1, 1) @'(FifthPadding, OctavePadding) trL2 $
            TT.unsqueeze @0 all
   where
    runConv
      :: (KnownNat nin)
      => TT.Conv2d nin hidden FifthSize OctaveSize QDType dev
      -> QBoundedList dev QDType MaxEdges '[] (nin : PShape)
      -> QTensor dev (hidden : PShape)
    runConv conv (QBoundedList mask edges) = TT.sumDim @0 $ TT.mul mask' out
     where
      out :: QTensor dev (MaxEdges : hidden : PShape)
      out = TT.conv2dForward @'(1, 1) @'(FifthPadding, OctavePadding) conv edges
      mask' :: QTensor dev '[MaxEdges, 1, 1, 1]
      mask' = TT.reshape mask
    runSlice conv slice = TT.squeezeDim @0 $ TT.conv2dForward @'(1, 1) @'(0, 0) conv input
     where
      input = TT.unsqueeze @0 $ TT.unsqueeze @0 slice
    pass :: QTensor dev (hidden : PShape)
    pass = runConv trL1Passing trencPassing
    inner :: QTensor dev (hidden : PShape)
    inner = runConv trL1Inner trencInner
    left :: QTensor dev (hidden : PShape)
    left = runSlice trL1Left $ getSlice trencLeft
    right :: QTensor dev (hidden : PShape)
    right = runSlice trL1Right $ getSlice trencRight
    root :: QTensor dev '[hidden, 1, 1]
    root = TT.reshape $ TT.mul trencRoot (activation (T.forward trL1Root ()))
    all :: QTensor dev (hidden : PShape)
    all = activation $ TT.layerNormForward trNorm1 $ (pass + inner + left + right) `TT.add` root

  forwardStoch tr input = pure $ T.forward tr input

-- | HasForward for transitions (batched)
instance
  forall dev hidden batchSize embshape
   . ( ValidParams dev hidden
     , embshape ~ (batchSize : hidden : PShape)
     , KnownNat batchSize
     )
  => T.HasForward (TransitionEncoder dev hidden) (TransitionEncoding dev '[batchSize]) (QTensor dev embshape)
  where
  forward TransitionEncoder{..} TransitionEncoding{..} =
    activation $ TT.layerNormForward trNorm2 $ TT.conv2dForward @'(1, 1) @'(FifthPadding, OctavePadding) trL2 all
   where
    runConv
      :: forall nin
       . (KnownNat nin)
      => TT.Conv2d nin hidden FifthSize OctaveSize QDType dev
      -> QBoundedList dev QDType MaxEdges '[batchSize] (nin : PShape)
      -> QTensor dev (batchSize : hidden : PShape)
    runConv conv (QBoundedList mask edges) = TT.sumDim @1 $ TT.mul mask' outReshaped
     where
      shape = TT.shapeVal @(nin : PShape)
      shape' = TT.shapeVal @(MaxEdges : hidden : PShape)
      inputShaped :: QTensor dev (batchSize * MaxEdges : nin : PShape)
      inputShaped = unsafeReshape (-1 : shape) edges
      out :: QTensor dev (batchSize * MaxEdges : hidden : PShape)
      out = TT.conv2dForward @'(1, 1) @'(FifthPadding, OctavePadding) conv inputShaped
      outReshaped :: QTensor dev (batchSize : MaxEdges : hidden : PShape)
      outReshaped = unsafeReshape (-1 : shape') out
      mask' :: QTensor dev '[batchSize, MaxEdges, 1, 1, 1]
      mask' = unsafeReshape [-1, TT.natValI @MaxEdges, 1, 1, 1] mask
    runSlice conv slice = TT.conv2dForward @'(1, 1) @'(0, 0) conv input
     where
      input = TT.unsqueeze @1 slice
    pass :: QTensor dev (batchSize : hidden : PShape)
    pass = runConv trL1Passing trencPassing
    inner :: QTensor dev (batchSize : hidden : PShape)
    inner = runConv trL1Inner trencInner
    left :: QTensor dev (batchSize : hidden : PShape)
    left = runSlice trL1Left $ getSlice trencLeft
    right :: QTensor dev (batchSize : hidden : PShape)
    right = runSlice trL1Right $ getSlice trencRight
    root :: QTensor dev '[batchSize, hidden, 1, 1]
    root = unsafeReshape [-1, TT.natValI @hidden, 1, 1] $ TT.mul (TT.unsqueeze @1 trencRoot) $ activation $ T.forward trL1Root ()
    all :: QTensor dev (batchSize : hidden : PShape)
    all = activation $ TT.layerNormForward trNorm1 $ (pass + inner + left + right) `TT.add` root

  forwardStoch tr input = pure $ T.forward tr input
