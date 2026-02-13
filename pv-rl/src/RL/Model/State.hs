{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE ExplicitNamespaces #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}

module RL.Model.State where

import RL.Encoding
import RL.Model.Common
import RL.Model.Slice
import RL.Model.Transition
import RL.ModelTypes

import Torch qualified as T
import Torch.Typed qualified as TT

import Control.DeepSeq (NFData)
import Data.TypeNums (KnownNat, type (<=))
import GHC.Generics (Generic)
import NoThunks.Class (NoThunks)

-- State Encoder
-- -------------

data StateSpec dev hidden = StateSpec

data StateEncoder dev hidden = StateEncoder
  { stL1mid :: TT.Conv2d hidden hidden FifthSize OctaveSize QDType dev
  , stL1frozenSlc :: TT.Conv2d hidden hidden FifthSize OctaveSize QDType dev
  , stL1frozenTr :: TT.Conv2d hidden hidden FifthSize OctaveSize QDType dev
  , stL1openSlc :: TT.Conv2d hidden hidden FifthSize OctaveSize QDType dev
  , stL1openTr :: TT.Conv2d hidden hidden FifthSize OctaveSize QDType dev
  , stL2 :: TT.Conv2d hidden hidden FifthSize OctaveSize QDType dev
  , stL3 :: TT.Conv2d hidden hidden FifthSize OctaveSize QDType dev
  , stNorm1 :: TT.LayerNorm (hidden : PShape) QDType dev
  , stNorm2 :: TT.LayerNorm (hidden : PShape) QDType dev
  , stNorm3 :: TT.LayerNorm (hidden : PShape) QDType dev
  }
  deriving (Show, Generic, TT.Parameterized, NoThunks, NFData)

instance (IsValidDevice dev, KnownNat hidden) => T.Randomizable (StateSpec dev hidden) (StateEncoder dev hidden) where
  sample _ = do
    stL1mid <- TT.sample TT.Conv2dSpec
    stL1frozenSlc <- TT.sample TT.Conv2dSpec
    stL1frozenTr <- TT.sample TT.Conv2dSpec
    stL1openSlc <- TT.sample TT.Conv2dSpec
    stL1openTr <- TT.sample TT.Conv2dSpec
    stL2 <- TT.sample TT.Conv2dSpec
    stL3 <- TT.sample TT.Conv2dSpec
    stNorm1 <- T.sample $ TT.LayerNormSpec 1e-05
    stNorm2 <- T.sample $ TT.LayerNormSpec 1e-05
    stNorm3 <- T.sample $ TT.LayerNormSpec 1e-05
    pure StateEncoder{..}

-- | HasForward for the parsing state (doesn't need batching)
instance
  forall dev hidden outShape
   . ( ValidParams dev hidden
     , outShape ~ (hidden : PShape)
     )
  => T.HasForward
      (StateEncoder dev hidden)
      (SliceEncoder dev hidden, TransitionEncoder dev hidden, StateEncoding dev)
      (QTensor dev outShape)
  where
  forward StateEncoder{..} (slc, tr, StateEncoding mid frozen open) = out3
   where
    -- helpers: running convolutions (batched and unbatched)
    runConv'
      :: (KnownNat nin, KnownNat nout, KnownNat batch)
      => TT.Conv2d nin nout FifthSize OctaveSize QDType dev
      -> QTensor dev (batch : nin : PShape)
      -> QTensor dev (batch : nout : PShape)
    runConv' conv input = TT.conv2dForward @'(1, 1) @'(FifthPadding, OctavePadding) conv input
    runConv
      :: (KnownNat nin, KnownNat nout)
      => TT.Conv2d nin nout FifthSize OctaveSize QDType dev
      -> QTensor dev (nin : PShape)
      -> QTensor dev (nout : PShape)
    runConv conv input = TT.squeezeDim @0 $ runConv' conv $ TT.unsqueeze @0 input

    -- embedding segments (open and frozen)
    embedSegments
      :: forall nsegs
       . (KnownNat nsegs, 1 <= nsegs)
      => TT.Conv2d hidden hidden FifthSize OctaveSize QDType dev
      -> TT.Conv2d hidden hidden FifthSize OctaveSize QDType dev
      -> QMaybe dev '[] (TransitionEncoding dev '[nsegs], QStartStop dev '[nsegs] (SliceEncoding dev '[nsegs]))
      -> QTensor dev (nsegs : hidden : PShape)
    embedSegments trEnc slcEnc (QMaybe mask (ft, fs)) =
      TT.mul (TT.reshape @[1, 1, 1, 1] mask) $ ftEmb + fsEmb
     where
      ftEmb :: QTensor dev (nsegs : hidden : PShape)
      ftEmb = runConv' trEnc $ T.forward tr ft
      fsEmb :: QTensor dev (nsegs : hidden : PShape)
      fsEmb = runConv' slcEnc $ T.forward slc fs

    -- embed frozen segments
    frozenEmb :: QTensor dev (hidden : PShape)
    frozenEmb = TT.meanDim @0 $ embedSegments stL1frozenTr stL1frozenSlc frozen
    -- embed open segments
    openEmb :: QTensor dev (hidden : PShape)
    openEmb = TT.meanDim @0 $ embedSegments stL1openTr stL1openSlc open
    -- embed the mid slice
    midEmb :: QTensor dev (hidden : PShape)
    midEmb = activation $ runConv stL1mid $ T.forward slc mid

    -- combined embeddings and compute output
    fullEmb :: QTensor dev (hidden : PShape)
    fullEmb = activation $ TT.layerNormForward stNorm1 $ midEmb + frozenEmb + openEmb
    out2 :: QTensor dev (hidden : PShape)
    out2 = activation $ TT.layerNormForward stNorm2 $ runConv stL2 fullEmb
    out3 :: QTensor dev (hidden : PShape)
    out3 = activation $ TT.layerNormForward stNorm3 $ runConv stL3 out2
  forwardStoch a i = pure $ T.forward a i
