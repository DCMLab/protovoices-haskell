{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE ExplicitNamespaces #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}

module RL.Model.Action where

import RL.Encoding
import RL.Model.Common
import RL.Model.Slice
import RL.Model.Transition
import RL.ModelTypes

import Torch qualified as T
import Torch.Typed qualified as TT

import Control.DeepSeq (NFData)
import Data.TypeNums (KnownNat, type (-), type (<=))
import GHC.Generics (Generic)
import NoThunks.Class (NoThunks)

-- ActionEncoder
-- -------------

data ActionSpec dev hidden = ActionSpec

data ActionEncoder dev hidden = ActionEncoder
  { actTop1sl :: !(TT.Conv2d hidden hidden FifthSize OctaveSize QDType dev)
  , actTop1sm :: !(TT.Conv2d hidden hidden FifthSize OctaveSize QDType dev)
  , actTop1sr :: !(TT.Conv2d hidden hidden FifthSize OctaveSize QDType dev)
  , actTop1t1 :: !(TT.Conv2d hidden hidden FifthSize OctaveSize QDType dev)
  , actTop1t2 :: !(TT.Conv2d hidden hidden FifthSize OctaveSize QDType dev)
  , actTop2 :: !(TT.Conv2d hidden hidden FifthSize OctaveSize QDType dev)
  , actSplit :: ConstEmb dev '[hidden - 3] -- TODO: fill in with actual module
  , actSpread :: ConstEmb dev '[hidden - 3] -- TODO: fill in with actual module
  , actFreeze :: ConstEmb dev '[hidden - 3]
  , actNorm1 :: TT.LayerNorm (hidden : PShape) QDType dev
  , actNorm2 :: TT.LayerNorm (hidden : PShape) QDType dev
  }
  deriving (Show, Generic, TT.Parameterized, NoThunks, NFData)

instance (IsValidDevice dev, KnownNat hidden, KnownNat (hidden - 3)) => T.Randomizable (ActionSpec dev hidden) (ActionEncoder dev hidden) where
  sample :: ActionSpec dev hidden -> IO (ActionEncoder dev hidden)
  sample ActionSpec = do
    actTop1sl <- T.sample TT.Conv2dSpec
    actTop1sm <- T.sample TT.Conv2dSpec
    actTop1sr <- T.sample TT.Conv2dSpec
    actTop1t1 <- T.sample TT.Conv2dSpec
    actTop1t2 <- T.sample TT.Conv2dSpec
    actTop2 <- T.sample TT.Conv2dSpec
    actSplit <- T.sample $ ConstEmbSpec @dev
    actSpread <- T.sample $ ConstEmbSpec @dev
    actFreeze <- T.sample $ ConstEmbSpec @dev
    actNorm1 <- T.sample $ TT.LayerNormSpec 1e-05
    actNorm2 <- T.sample $ TT.LayerNormSpec 1e-05
    pure ActionEncoder{..}

opTypes :: forall dev. (TT.KnownDevice dev) => QTensor dev '[6, 3]
opTypes =
  TT.UnsafeMkTensor
    $! T.asTensor' @[[QType]]
      [ [0, 0, 0] -- freeze only
      , [0, 1, 0] -- split only
      , [1, 0, 0] -- freeze left
      , [1, 0, 1] -- spread
      , [1, 1, 0] -- freeze left
      , [1, 1, 1] -- freeze right
      ]
    $ opts @dev

-- | HasForward for actions (batched)
instance
  forall dev hidden batchSize outShape
   . ( ValidParams dev hidden
     , outShape ~ (batchSize : hidden : PShape)
     , 1 <= batchSize
     , KnownNat batchSize
     )
  => T.HasForward
      (ActionEncoder dev hidden)
      (SliceEncoder dev hidden, TransitionEncoder dev hidden, ActionEncoding dev '[batchSize])
      (QTensor dev outShape)
  where
  forward ActionEncoder{..} (slc, tr, ActionEncoding (ActionTop sl t1 (QMaybe smMask sm) (QMaybe t2Mask t2) sr) opIndex) =
    activation $ TT.layerNormForward actNorm2 $ topEmb `TT.add` opEmbReshaped
   where
    runConv
      :: forall nin nout
       . (KnownNat nin, KnownNat nout)
      => TT.Conv2d nin nout FifthSize OctaveSize QDType dev
      -> QTensor dev (batchSize : nin : PShape)
      -> QTensor dev (batchSize : nout : PShape)
    runConv conv input =
      TT.conv2dForward @'(1, 1) @'(FifthPadding, OctavePadding) conv input
    runConvMasked
      :: (KnownNat nin, KnownNat nout)
      => QTensor dev '[batchSize]
      -> TT.Conv2d nin nout FifthSize OctaveSize QDType dev
      -> QTensor dev (batchSize : nin : PShape)
      -> QTensor dev (batchSize : nout : PShape)
    runConvMasked mask conv input =
      TT.mul (unsafeReshape [-1, 1, 1, 1] mask :: QTensor dev '[batchSize, 1, 1, 1]) $ runConv conv input
    -- top embedding
    embl :: QTensor dev (batchSize : hidden : PShape)
    embl = runConv actTop1sl $ T.forward slc sl
    embm = runConv actTop1sm $ T.forward slc sm
    embr = runConvMasked smMask actTop1sr $ T.forward slc sr
    embt1 = runConv actTop1t1 $ T.forward tr t1
    embt2 = runConvMasked t2Mask actTop1t2 $ T.forward tr t2
    topCombined :: QTensor dev (batchSize : hidden : PShape)
    topCombined =
      activation $ TT.layerNormForward actNorm1 $ embl + embm + embr + embt1 + embt2
    topEmb :: QTensor dev (batchSize : hidden : PShape)
    topEmb = runConv actTop2 topCombined
    -- operation embedding
    opFreeze = T.forward actFreeze ()
    opSplit = T.forward actSplit ()
    opSpread = T.forward actSpread ()
    opCombined = TT.stack @0 $ opFreeze TT.:. opSplit TT.:. opFreeze TT.:. opSpread TT.:. opSplit TT.:. opSplit TT.:. TT.HNil
    opEmbeddings :: QTensor dev '[6, hidden]
    opEmbeddings = TT.cat @1 $ opTypes @dev TT.:. opCombined TT.:. TT.HNil
    opIndex' :: TT.Tensor dev TT.Int64 [batchSize, hidden]
    opIndex' = TT.UnsafeMkTensor $ T.expand (TT.toDynamic $ TT.unsqueeze @1 opIndex) False [-1, TT.natValI @hidden]
    opEmb :: QTensor dev '[batchSize, hidden]
    opEmb = TT.gatherDim @0 opIndex' opEmbeddings
    opEmbReshaped :: QTensor dev '[batchSize, hidden, 1, 1]
    opEmbReshaped = TT.unsqueeze @3 $ TT.unsqueeze @2 opEmb
  forwardStoch a i = pure $ T.forward a i
