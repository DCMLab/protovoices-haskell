{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE ExplicitNamespaces #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE UndecidableInstances #-}

module RL.Model.QModel where

import RL.Encoding
import RL.Model.Action
import RL.Model.Common
import RL.Model.Slice
import RL.Model.State
import RL.Model.Transition
import RL.ModelTypes

import RL.TorchHelpers qualified as TH
import Torch qualified as T
import Torch.Typed qualified as TT

import Control.DeepSeq (NFData)
import Data.TypeNums (KnownNat, type (<=))
import GHC.Generics (Generic)
import NoThunks.Class (NoThunks)

-- Full Q Model
-- ------------

data QSpec hidden dev = QSpec

data QModel hidden dev = QModel
  { qModelSlc :: !(SliceEncoder dev hidden)
  , qModelTr :: !(TransitionEncoder dev hidden)
  , qModelAct :: !(ActionEncoder dev hidden)
  , qModelSt :: !(StateEncoder dev hidden)
  , qModelFinal1 :: !(TT.Conv2d hidden hidden FifthSize OctaveSize QDType dev) -- !(TT.Linear (EmbSize (QSpecGeneral DefaultQSpec)) QOutHidden QDType dev)
  , qModelNorm1 :: !(TT.LayerNorm '[hidden] QDType dev)
  , qModelFinal2 :: !(TT.Linear hidden 1 QDType dev)
  , qModelValue1 :: !(TT.Linear hidden hidden QDType dev)
  , qModelValueNorm :: !(TT.LayerNorm '[hidden] QDType dev)
  , qModelValue2 :: !(TT.Linear hidden 1 QDType dev)
  }
  deriving (Show, Generic, TT.Parameterized, NoThunks, NFData)

instance (ValidParams dev hidden) => T.Randomizable (QSpec hidden dev) (QModel hidden dev) where
  sample :: QSpec hidden dev -> IO (QModel hidden dev)
  sample QSpec = do
    qModelSlc <- T.sample $ SliceSpec @dev @hidden
    qModelTr <- T.sample $ TransitionSpec @dev @hidden
    qModelAct <- T.sample $ ActionSpec @dev @hidden
    qModelSt <- T.sample $ StateSpec @dev @hidden
    qModelFinal1 <- T.sample TT.Conv2dSpec
    qModelNorm1 <- T.sample $ TT.LayerNormSpec 1e-05
    qModelFinal2 <- T.sample TT.LinearSpec
    qModelValue1 <- T.sample TT.LinearSpec
    qModelValueNorm <- T.sample $ TT.LayerNormSpec 1e-05
    qModelValue2 <- T.sample TT.LinearSpec
    pure QModel{..}

mkQModel :: forall dev hidden. (ValidParams dev hidden) => IO (QModel hidden dev)
mkQModel = T.sample $ QSpec @hidden @dev

loadQModel
  :: forall dev hidden
   . (ValidParams dev hidden)
  => FilePath
  -> IO (QModel hidden dev)
loadQModel path = do
  modelPlaceholder <- mkQModel @dev
  tensors
    :: (TT.HMap' TT.ToDependent (TT.Parameters (QModel hidden dev)) ts)
    => TT.HList ts <-
    TT.load path
  -- TT.load doesn't move the parameters to the correct device, so we move them manually
  let tensorsCPU = TT.toDevice @'(TT.CPU, 0) @dev tensors
  let tensorsDevice = TT.toDevice @dev @'(TT.CPU, 0) tensorsCPU
  params <- TT.hmapM' TT.MakeIndependent tensorsDevice
  pure $ TT.replaceParameters modelPlaceholder params

forwardQModel
  :: (ValidParams dev hidden)
  => QModel hidden dev
  -> QEncoding dev '[]
  -> QTensor dev '[1]
forwardQModel model input = TT.squeezeDim @0 $ forwardQModelBatched model $ addBatchDim input

forwardQModelBatched
  :: forall dev hidden batchSize
   . ( ValidParams dev hidden
     , 1 <= batchSize
     )
  => QModel hidden dev
  -> QEncoding dev '[batchSize]
  -> QTensor dev '[batchSize, 1]
forwardQModelBatched (QModel slc tr act st final1 norm1 final2 _ _ _) (QEncoding actEncs stEnc) = out2
 where
  actEmb :: QTensor dev (batchSize : hidden : PShape)
  actEmb = T.forward act (slc, tr, actEncs)
  stEmb :: QTensor dev (hidden : PShape)
  stEmb = T.forward st (slc, tr, stEnc)
  inputEmb = actEmb `TT.add` stEmb
  out1 :: QTensor dev (batchSize : hidden : PShape)
  out1 = TH.conv2dForwardRelaxed @'(1, 1) @'(FifthPadding, OctavePadding) final1 inputEmb
  sum1 :: QTensor dev '[batchSize, hidden]
  sum1 = TT.sumDim @2 $ TT.sumDim @2 out1
  out1norm :: QTensor dev '[batchSize, hidden]
  out1norm = activation $ TH.layerNormForwardRelaxed norm1 sum1
  out2 :: QTensor dev '[batchSize, 1]
  out2 = T.forward final2 out1norm

forwardQModelFullyBatched
  :: forall dev hidden
   . ( ValidParams dev hidden
     )
  => QModel hidden dev
  -> QEncodingBatch dev
  -> [T.Tensor] -- TODO: could be changed to SomePolicy?
forwardQModelFullyBatched (QModel slc tr act st final1 norm1 final2 _ _ _) (QEncodingBatch @dev @batchSize actsEnc stEncs sizes) =
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
  out1norm :: QTensor dev '[batchSize, hidden]
  out1norm = activation $ TH.layerNormForwardRelaxed norm1 sum1
  out2 :: QTensor dev '[batchSize, 1]
  out2 = T.forward final2 out1norm
  outAll = TT.toDynamic out2
  getOuts :: Int -> [Int] -> [T.Tensor]
  getOuts _ [] = []
  getOuts start (size : sizes) = (outAll T.! (T.Slice (start, start + size))) : getOuts (start + size) sizes

-- | HasForward for QModel (unbatched)
instance
  ( ValidParams dev hidden
  )
  => T.HasForward (QModel hidden dev) (QEncoding dev '[]) (QTensor dev '[1])
  where
  forward :: QModel hidden dev -> QEncoding dev '[] -> QTensor dev '[1]
  forward model encoding = forwardQModel model encoding

  forwardStoch :: QModel hidden dev -> QEncoding dev '[] -> IO (QTensor dev '[1])
  forwardStoch model input = pure $ T.forward model input

-- | HasForward for QModel (batched actions)
instance
  ( ValidParams dev hidden
  , KnownNat batchSize
  , 1 <= batchSize
  )
  => T.HasForward (QModel hidden dev) (QEncoding dev '[batchSize]) (QTensor dev '[batchSize, 1])
  where
  forward :: QModel hidden dev -> QEncoding dev '[batchSize] -> QTensor dev '[batchSize, 1]
  forward model encoding = forwardQModelBatched model encoding

  forwardStoch model input = pure $ T.forward model input

-- | HasForward for QModel (fully batched)
instance
  ( ValidParams dev hidden
  )
  => T.HasForward (QModel hidden dev) (QEncodingBatch dev) [T.Tensor]
  where
  forward :: QModel hidden dev -> QEncodingBatch dev -> [T.Tensor]
  forward = forwardQModelFullyBatched

  forwardStoch model input = pure $ T.forward model input

-- | HasForward for QModel (value)
instance
  ( ValidParams dev hidden
  )
  => T.HasForward (QModel hidden dev) (StateEncoding dev) (QTensor dev '[1])
  where
  forward :: QModel hidden dev -> StateEncoding dev -> QTensor dev '[1]
  forward (QModel slc tr _ st _ _ _ value1 norm value2) stateEncoding = out2
   where
    outSlc = TT.sumDim @1 $ TT.sumDim @1 $ T.forward st (slc, tr, stateEncoding)
    out1 = activation $ T.forward norm $ T.forward value1 outSlc
    out2 = TT.log $ TT.sigmoid $ T.forward value2 out1

  forwardStoch model input = pure $ T.forward model input

-- -- Specializations
-- -- ===============
--
-- {-# SPECIALIZE TT.hmap' ::
--   forall hidden dev
--    . () -- (TT.HMap TT.ToDependent (TT.Parameters (QModel hidden dev)) ys)
--   => TT.ToDependent
--   -> TT.HList (ModelParams (QModel hidden) dev)
--   -> TT.HList (ModelTensors (QModel hidden) dev)
--   #-}
