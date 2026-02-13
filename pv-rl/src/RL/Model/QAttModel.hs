{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE ExplicitNamespaces #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE UndecidableInstances #-}

module RL.Model.QAttModel where

import RL.Encoding
import RL.Model.Action
import RL.Model.Common
import RL.Model.Slice
import RL.Model.State
import RL.Model.Transition
import RL.ModelTypes

import RL.TorchHelpers qualified as TH
import Torch qualified as T
import Torch.Functional.Internal qualified as TI
import Torch.Typed qualified as TT

import Control.DeepSeq (NFData)
import Data.TypeNums (KnownNat, type (<=))
import GHC.Generics (Generic)
import NoThunks.Class (NoThunks)
import System.IO.Unsafe (unsafePerformIO)

-- Q Model with Attention
-- ----------------------

data QAttSpec hidden dev = QAttSpec

data QAttModel hidden dev = QAttModel
  { qAttModelSlc :: !(SliceEncoder dev hidden)
  , qAttModelTr :: !(TransitionEncoder dev hidden)
  , qAttModelAct :: !(ActionEncoder dev hidden)
  , qAttModelSt :: !(StateEncoder dev hidden)
  , qAttModelFinal1 :: !(TT.Conv2d hidden hidden FifthSize OctaveSize QDType dev) -- !(TT.Linear (EmbSize (QSpecGeneral DefaultQSpec)) QOutHidden QDType dev)
  , qAttModelAtt1 :: !(TT.MultiheadAttention hidden hidden hidden 1 QDType dev)
  , qAttModelNorm1 :: !(TT.LayerNorm '[hidden] QDType dev)
  , qAttModelNorm2 :: !(TT.LayerNorm '[hidden] QDType dev)
  , qAttModelFinal2 :: !(TT.Linear hidden 1 QDType dev)
  , qAttModelValue1 :: !(TT.Linear hidden hidden QDType dev)
  , qAttModelValueNorm :: !(TT.LayerNorm '[hidden] QDType dev)
  , qAttModelValue2 :: !(TT.Linear hidden 1 QDType dev)
  }
  deriving (Show, Generic, TT.Parameterized, NoThunks, NFData)

instance (ValidParams dev hidden) => T.Randomizable (QAttSpec hidden dev) (QAttModel hidden dev) where
  sample :: QAttSpec hidden dev -> IO (QAttModel hidden dev)
  sample QAttSpec = do
    qAttModelSlc <- T.sample $ SliceSpec @dev @hidden
    qAttModelTr <- T.sample $ TransitionSpec @dev @hidden
    qAttModelAct <- T.sample $ ActionSpec @dev @hidden
    qAttModelSt <- T.sample $ StateSpec @dev @hidden
    qAttModelFinal1 <- T.sample TT.Conv2dSpec
    qAttModelAtt1 <- T.sample $ TT.MultiheadAttentionSpec $ TT.DropoutSpec 0
    qAttModelNorm1 <- T.sample $ TT.LayerNormSpec 1e-05
    qAttModelNorm2 <- T.sample $ TT.LayerNormSpec 1e-05
    qAttModelFinal2 <- T.sample TT.LinearSpec
    qAttModelValue1 <- T.sample TT.LinearSpec
    qAttModelValueNorm <- T.sample $ TT.LayerNormSpec 1e-05
    qAttModelValue2 <- T.sample TT.LinearSpec
    pure QAttModel{..}

mkQAttModel :: forall dev hidden. (ValidParams dev hidden) => IO (QAttModel hidden dev)
mkQAttModel = T.sample $ QAttSpec @hidden @dev

loadQAttModel
  :: forall dev hidden
   . (ValidParams dev hidden)
  => FilePath
  -> IO (QAttModel hidden dev)
loadQAttModel path = do
  modelPlaceholder <- mkQAttModel @dev
  tensors
    :: (TT.HMap' TT.ToDependent (TT.Parameters (QAttModel hidden dev)) ts)
    => TT.HList ts <-
    TT.load path
  -- TT.load doesn't move the parameters to the correct device, so we move them manually
  let tensorsCPU = TT.toDevice @'(TT.CPU, 0) @dev tensors
  let tensorsDevice = TT.toDevice @dev @'(TT.CPU, 0) tensorsCPU
  params <- TT.hmapM' TT.MakeIndependent tensorsDevice
  pure $ TT.replaceParameters modelPlaceholder params

forwardQAttModel
  :: (ValidParams dev hidden)
  => QAttModel hidden dev
  -> QEncoding dev '[]
  -> QTensor dev '[1]
forwardQAttModel model input = TT.squeezeDim @0 $ forwardQAttModelBatched model $ addBatchDim input

forwardQAttModelBatched
  :: forall dev hidden batchSize
   . ( ValidParams dev hidden
     , 1 <= batchSize
     , KnownNat batchSize
     )
  => QAttModel hidden dev
  -> QEncoding dev '[batchSize]
  -> QTensor dev '[batchSize, 1]
forwardQAttModelBatched (QAttModel slc tr act st final1 att1 norm1 norm2 final2 _ _ _) (QEncoding actEncs stEnc) = out2
 where
  actEmb :: QTensor dev (batchSize : hidden : PShape)
  actEmb = T.forward act (slc, tr, actEncs)
  stEmb :: QTensor dev (hidden : PShape)
  stEmb = T.forward st (slc, tr, stEnc)
  inputEmb = actEmb `TT.add` stEmb
  out1 :: QTensor dev (batchSize : hidden : PShape)
  out1 = TH.conv2dForwardRelaxed @'(1, 1) @'(FifthPadding, OctavePadding) final1 inputEmb
  sum1 :: QTensor dev '[batchSize, hidden]
  sum1 = TH.layerNormForwardRelaxed norm1 $ TT.sumDim @2 $ TT.sumDim @2 out1
  attIn :: QTensor dev '[1, batchSize, hidden]
  attIn = TT.unsqueeze @0 sum1
  attOut :: QTensor dev '[1, batchSize, hidden]
  (attOut, _) = unsafePerformIO $ TT.multiheadAttention att1 False Nothing Nothing Nothing Nothing attIn attIn attIn
  out1norm :: QTensor dev '[batchSize, hidden]
  out1norm = activation $ TH.layerNormForwardRelaxed norm2 (TT.squeezeDim @0 attOut) + sum1
  -- out1norm :: QTensor dev '[batchSize, hidden]
  -- out1norm = activation $ TH.layerNormForwardRelaxed norm1 sum1
  out2 :: QTensor dev '[batchSize, 1]
  out2 = T.forward final2 out1norm

forwardQAttModelFullyBatched
  :: forall dev hidden
   . ( ValidParams dev hidden
     )
  => QAttModel hidden dev
  -> QEncodingBatch dev
  -> [T.Tensor] -- TODO: could be changed to SomePolicy?
forwardQAttModelFullyBatched (QAttModel slc tr act st final1 att1 norm1 norm2 final2 _ _ _) (QEncodingBatch @dev @batchSize actsEnc stEncs sizes) =
  getOuts 0 sizes
 where
  actEmb :: QTensor dev (batchSize : hidden : PShape)
  actEmb = T.forward act (slc, tr, actsEnc)
  stEmbs :: [QTensor dev (hidden : PShape)]
  stEmbs = fmap (\stEnc -> T.forward st (slc, tr, stEnc)) stEncs
  stEmbs' :: [T.Tensor]
  stEmbs' = zipWith (\emb size -> T.repeat [size, 1, 1, 1] $ TT.toDynamic emb) stEmbs sizes
  stEmb :: QTensor dev (batchSize : hidden : PShape)
  stEmb = TT.UnsafeMkTensor (T.cat (T.Dim 0) stEmbs')
  inputEmb :: QTensor dev (batchSize : hidden : PShape)
  inputEmb = actEmb `TT.add` stEmb
  out1 :: QTensor dev (batchSize : hidden : PShape)
  out1 = TH.conv2dForwardRelaxed @'(1, 1) @'(FifthPadding, OctavePadding) final1 inputEmb
  sum1 :: QTensor dev '[batchSize, hidden]
  sum1 = TT.sumDim @2 $ TT.sumDim @2 out1
  groupMask :: QTensor dev [batchSize, batchSize]
  groupMask = TT.UnsafeMkTensor $ TI.block_diag $ fmap mkAttMask sizes
  sum1Norm :: QTensor dev '[batchSize, hidden]
  -- sum1Norm = TH.checkNaN (sum1, sizes) $ TH.normalizeBatch sum1
  sum1Norm = TH.layerNormForwardRelaxed norm1 sum1
  attIn :: QTensor dev '[1, batchSize, hidden]
  attIn = TT.unsqueeze @0 sum1Norm
  mkAttMask size = T.ones [size, size] (opts @dev)
  attMask :: QTensor dev '[1, batchSize, batchSize]
  attMask = (-1 / (TT.unsqueeze @0 $ groupMask)) + 1
  attOut :: QTensor dev '[1, batchSize, hidden]
  (attOut, _) = unsafePerformIO $ TT.multiheadAttention att1 False (Just attMask) Nothing Nothing Nothing attIn attIn attIn
  out1norm :: QTensor dev '[batchSize, hidden]
  out1norm = activation $ TH.layerNormForwardRelaxed norm2 (TT.squeezeDim @0 attOut) + sum1Norm
  -- out1norm :: QTensor dev '[batchSize, hidden]
  -- out1norm = activation $ TH.layerNormForwardRelaxed norm1 sum1
  out2 :: QTensor dev '[batchSize, 1]
  out2 = T.forward final2 out1norm
  outAll = TT.toDynamic out2
  getOuts :: Int -> [Int] -> [T.Tensor]
  getOuts _ [] = []
  getOuts start (size : sizes) = (outAll T.! (T.Slice (start, start + size))) : getOuts (start + size) sizes

-- | HasForward for QAttModel (unbatched)
instance
  ( ValidParams dev hidden
  )
  => T.HasForward (QAttModel hidden dev) (QEncoding dev '[]) (QTensor dev '[1])
  where
  forward :: QAttModel hidden dev -> QEncoding dev '[] -> QTensor dev '[1]
  forward model encoding = forwardQAttModel model encoding

  forwardStoch :: QAttModel hidden dev -> QEncoding dev '[] -> IO (QTensor dev '[1])
  forwardStoch model input = pure $ T.forward model input

-- | HasForward for QAttModel (batched actions)
instance
  ( ValidParams dev hidden
  , KnownNat batchSize
  , 1 <= batchSize
  )
  => T.HasForward (QAttModel hidden dev) (QEncoding dev '[batchSize]) (QTensor dev '[batchSize, 1])
  where
  forward :: QAttModel hidden dev -> QEncoding dev '[batchSize] -> QTensor dev '[batchSize, 1]
  forward model encoding = forwardQAttModelBatched model encoding

  forwardStoch model input = pure $ T.forward model input

-- | HasForward for QAttModel (fully batched)
instance
  ( ValidParams dev hidden
  )
  => T.HasForward (QAttModel hidden dev) (QEncodingBatch dev) [T.Tensor]
  where
  forward :: QAttModel hidden dev -> QEncodingBatch dev -> [T.Tensor]
  forward = forwardQAttModelFullyBatched

  forwardStoch model input = pure $ T.forward model input

-- | HasForward for QAttModel (value)
instance
  ( ValidParams dev hidden
  )
  => T.HasForward (QAttModel hidden dev) (StateEncoding dev) (QTensor dev '[1])
  where
  forward :: QAttModel hidden dev -> StateEncoding dev -> QTensor dev '[1]
  forward (QAttModel slc tr _ st _ _ _ _ _ value1 norm value2) stateEncoding = out2
   where
    outSlc = TT.sumDim @1 $ TT.sumDim @1 $ T.forward st (slc, tr, stateEncoding)
    out1 = activation $ T.forward norm $ T.forward value1 outSlc
    out2 = TT.log $ TT.sigmoid $ T.forward value2 out1

  forwardStoch model input = pure $ T.forward model input
