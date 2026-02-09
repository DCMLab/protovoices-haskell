{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DerivingVia #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RecordWildCards #-}
-- {-# LANGUAGE Strict #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}
{-# HLINT ignore "Use <$>" #-}
-- {-# OPTIONS_GHC -O0 #-}
-- {-# OPTIONS_GHC -v #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# OPTIONS_GHC -Wredundant-constraints #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}

module RL.Model where

import Common
import GreedyParser (DoubleParent (DoubleParent), SingleParent (SingleParent))

import RL.Encoding
import RL.ModelTypes
import RL.TorchHelpers (ToModelTensors, withBatchDim)
import RL.TorchHelpers qualified as TH

import Control.DeepSeq
import Data.Foldable qualified as F
import Data.Kind (Type)
import Data.Proxy (Proxy (Proxy))
import Data.Type.Equality (type (:~:) (Refl), type (==))
import Data.TypeNums (KnownNat, Nat, TInt (..), intVal, intVal', type (*), type (+), type (-), type (<=))
import Debug.Trace qualified as DT
import GHC.ForeignPtr qualified as Ptr
import GHC.Generics (Generic)
import GHC.TypeLits (OrderingI (..), cmpNat, sameNat)
import NoThunks.Class (NoThunks (..), OnlyCheckWhnf (..), allNoThunks)
import System.IO.Unsafe
import Torch (batchNormForwardIO)
import Torch qualified as T
import Torch.Functional.Internal qualified as TI
import Torch.Internal.Cast (cast2)
import Torch.Internal.Managed.Type.Tensor qualified as ATen
import Torch.Jit qualified as TJit
import Torch.Lens qualified as TL
import Torch.Typed qualified as TT
import Unsafe.Coerce (unsafeCoerce)

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

-- Slice Encoder
-- -------------

data SliceSpec dev hidden = SliceSpec

data SliceEncoder dev hidden = SliceEncoder
  { _slcL1 :: !(TT.Conv2d 1 hidden 1 1 QDType dev) -- !(TT.Linear (PSize spec) hidden QDType QDevice)
  , _slcL2 :: !(TT.Conv2d hidden hidden FifthSize OctaveSize QDType dev) -- !(TT.Linear hidden (EmbSize spec) QDType QDevice)
  , _slcStart :: !(ConstEmb dev (hidden : PShape))
  , _slcStop :: !(ConstEmb dev (hidden : PShape))
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

-- | HasFoward for slice (unbatched)
instance
  (embshape ~ hidden : PShape, IsValidDevice dev, KnownNat hidden)
  => T.HasForward (SliceEncoder dev hidden) (SliceEncoding dev '[]) (QTensor dev embshape)
  where
  forward (SliceEncoder l1 l2 _ _) slice = TT.squeezeDim @0 out2
   where
    input = TT.unsqueeze @0 $ TT.unsqueeze @0 $ getSlice slice
    out1 :: QTensor dev (1 : hidden : PShape)
    out1 = TT.conv2dForward @'(1, 1) @'(0, 0) l1 input
    out2 :: QTensor dev (1 : hidden : PShape)
    out2 = activation $ TT.conv2dForward @'(1, 1) @'(FifthPadding, OctavePadding) l2 out1
  forwardStoch model = pure . T.forward model

-- | HasFoward for slice (batched)
instance
  ( IsValidDevice dev
  , embshape ~ '[batchSize, hidden, FifthSize, OctaveSize]
  )
  => T.HasForward (SliceEncoder dev hidden) (SliceEncoding dev '[batchSize]) (QTensor dev embshape)
  where
  forward (SliceEncoder l1 l2 _ _) slice = out2
   where
    input = TT.unsqueeze @1 $ getSlice slice
    out1 :: QTensor dev '[batchSize, hidden, FifthSize, OctaveSize]
    out1 = activation $ TH.conv2dForwardRelaxed @'(1, 1) @'(0, 0) l1 input
    out2 :: QTensor dev '[batchSize, hidden, FifthSize, OctaveSize]
    out2 = activation $ TH.conv2dForwardRelaxed @'(1, 1) @'(FifthPadding, OctavePadding) l2 out1
  forwardStoch model = pure . T.forward model

-- | HasForward for slice wrappend in QStartStop (unbatched).
instance
  (embshape ~ hidden : PShape, IsValidDevice dev, KnownNat hidden)
  => TT.HasForward (SliceEncoder dev hidden) (QStartStop dev '[] (SliceEncoding dev '[])) (QTensor dev embshape)
  where
  forward model@(SliceEncoder _ _ start stop) (QStartStop tag input) = TT.squeezeDim @0 out
   where
    -- compute the possible outputs for start/stop/inner
    outStart :: QTensor dev (hidden : PShape)
    outStart = TT.forward start ()
    outStop :: QTensor dev (hidden : PShape)
    outStop = TT.forward stop ()
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
  ( IsValidDevice dev
  , embshape ~ (batchSize : hidden : PShape)
  )
  => TT.HasForward (SliceEncoder dev hidden) (QStartStop dev '[batchSize] (SliceEncoding dev '[batchSize])) (QTensor dev embshape)
  where
  forward model@(SliceEncoder _ _ start stop) (QStartStop tag input) = TT.squeezeDim @0 out
   where
    -- compute the possible outputs for start/stop/inner
    outStart :: QTensor dev (batchSize : hidden : PShape)
    outStart = TT.UnsafeMkTensor $ expandAs (TT.toDynamic $ TT.forward start ()) $ TT.toDynamic outInner
    outStop :: QTensor dev (batchSize : hidden : PShape)
    outStop = TT.UnsafeMkTensor $ expandAs (TT.toDynamic $ TT.forward stop ()) $ TT.toDynamic outInner
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

-- Transition Encoder
-- ------------------

data TransitionSpec dev hidden = TransitionSpec

data TransitionEncoder dev hidden = TransitionEncoder
  { trL1Passing :: !(TT.Conv2d 2 hidden FifthSize OctaveSize QDType dev) -- !(TT.Linear (ESize spec) hidden QDType QDevice)
  , trL1Inner :: !(TT.Conv2d 2 hidden FifthSize OctaveSize QDType dev) -- !(TT.Linear (ESize spec) hidden QDType QDevice)
  , trL1Left :: !(TT.Conv2d 1 hidden 1 1 QDType dev) -- !(TT.Linear (PSize spec) hidden QDType QDevice)
  , trL1Right :: !(TT.Conv2d 1 hidden 1 1 QDType dev) -- !(TT.Linear (PSize spec) hidden QDType QDevice)
  , trL1Root :: !(ConstEmb dev '[hidden])
  , trL2 :: !(TT.Conv2d hidden hidden FifthSize OctaveSize QDType dev) -- !(TT.Linear hidden (EmbSize) QDType QDevice)
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
    pure $ TransitionEncoder{..}

-- | HasForward for transitions (unbatched)
instance
  forall dev hidden embshape
   . ( IsValidDevice dev
     , KnownNat hidden
     , embshape ~ (hidden : PShape)
     )
  => T.HasForward (TransitionEncoder dev hidden) (TransitionEncoding dev '[]) (QTensor dev embshape)
  where
  forward TransitionEncoder{..} TransitionEncoding{..} =
    TT.squeezeDim @0 $
      activation $
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
      out = activation $ TT.conv2dForward @'(1, 1) @'(FifthPadding, OctavePadding) conv edges
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
    all = (pass + inner + left + right) `TT.add` root

  forwardStoch tr input = pure $ T.forward tr input

-- | HasForward for transitions (batched)
instance
  forall dev hidden batchSize embshape
   . ( ValidParams dev hidden
     , embshape ~ (batchSize : hidden : PShape)
     )
  => T.HasForward (TransitionEncoder dev hidden) (TransitionEncoding dev '[batchSize]) (QTensor dev embshape)
  where
  forward TransitionEncoder{..} TransitionEncoding{..} =
    activation $ TH.conv2dForwardRelaxed @'(1, 1) @'(FifthPadding, OctavePadding) trL2 all
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
      out = activation $ TH.conv2dForwardRelaxed @'(1, 1) @'(FifthPadding, OctavePadding) conv inputShaped
      outReshaped :: QTensor dev (batchSize : MaxEdges : hidden : PShape)
      outReshaped = unsafeReshape (-1 : shape') out
      mask' :: QTensor dev '[batchSize, MaxEdges, 1, 1, 1]
      mask' = unsafeReshape [-1, TT.natValI @MaxEdges, 1, 1, 1] mask
    runSlice conv slice = TH.conv2dForwardRelaxed @'(1, 1) @'(0, 0) conv input
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
    all = (pass + inner + left + right) `TT.add` root

  forwardStoch tr input = pure $ T.forward tr input

-- ActionEncoder
-- -------------

data ActionSpec dev hidden = ActionSpec

data ActionEncoder dev hidden = ActionEncoder
  { actTop1sl :: !(TT.Conv2d hidden hidden FifthSize OctaveSize QDType dev) -- TT.Linear (EmbSize) hidden QDType dev
  , actTop1sm :: !(TT.Conv2d hidden hidden FifthSize OctaveSize QDType dev) -- TT.Linear (EmbSize) hidden QDType dev
  , actTop1sr :: !(TT.Conv2d hidden hidden FifthSize OctaveSize QDType dev) -- TT.Linear (EmbSize) hidden QDType dev
  , actTop1t1 :: !(TT.Conv2d hidden hidden FifthSize OctaveSize QDType dev) -- TT.Linear (EmbSize) hidden QDType dev
  , actTop1t2 :: !(TT.Conv2d hidden hidden FifthSize OctaveSize QDType dev) -- TT.Linear (EmbSize) hidden QDType dev
  , actTop2 :: !(TT.Conv2d hidden hidden FifthSize OctaveSize QDType dev) -- TT.Linear hidden (EmbSize) QDType dev
  , actSplit :: ConstEmb dev '[hidden - 3] -- TODO: fill in with actual module
  , actSpread :: ConstEmb dev '[hidden - 3] -- TODO: fill in with actual module
  , actFreeze :: ConstEmb dev '[hidden - 3]
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
     )
  => T.HasForward
      (ActionEncoder dev hidden)
      (SliceEncoder dev hidden, TransitionEncoder dev hidden, ActionEncoding dev '[batchSize])
      (QTensor dev outShape)
  where
  forward ActionEncoder{..} (slc, tr, ActionEncoding (ActionTop sl t1 (QMaybe smMask sm) (QMaybe t2Mask t2) sr) opIndex) = topEmb `TT.add` opEmbReshaped
   where
    runConv
      :: TT.Conv2d nin nout FifthSize OctaveSize QDType dev
      -> QTensor dev (batchSize : nin : PShape)
      -> QTensor dev (batchSize : nout : PShape)
    runConv conv input =
      activation $ TH.conv2dForwardRelaxed @'(1, 1) @'(FifthPadding, OctavePadding) conv input
    runConvMasked
      :: QTensor dev '[batchSize]
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
    topCombined = embl + embm + embr + embt1 + embt2
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
  forward StateEncoder{..} (slc, tr, StateEncoding @nfrozen @nopen mid frozen open) = out3
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
       . (KnownNat nsegs)
      => TT.Conv2d hidden hidden FifthSize OctaveSize QDType dev
      -> TT.Conv2d hidden hidden FifthSize OctaveSize QDType dev
      -> QMaybe dev '[] (TransitionEncoding dev '[nsegs], QStartStop dev '[nsegs] (SliceEncoding dev '[nsegs]))
      -> QTensor dev (nsegs : hidden : PShape)
    embedSegments trEnc slcEnc (QMaybe mask (ft, fs)) =
      TT.mul (TT.reshape @[1, 1, 1, 1] mask) $ ftEmb + fsEmb
     where
      ftEmb :: QTensor dev (nsegs : hidden : PShape)
      ftEmb = activation $ runConv' trEnc $ T.forward tr ft
      fsEmb :: QTensor dev (nsegs : hidden : PShape)
      fsEmb = activation $ runConv' slcEnc $ T.forward slc fs

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
    fullEmb = midEmb + frozenEmb + openEmb
    out2 :: QTensor dev (hidden : PShape)
    out2 = activation $ runConv stL2 fullEmb
    out3 :: QTensor dev (hidden : PShape)
    out3 = activation $ runConv stL3 out2
  forwardStoch a i = pure $ T.forward a i

-- Full Q Model
-- ------------

data QSpec dev hidden = QSpec

data QModel dev hidden = QModel
  { qModelSlc :: !(SliceEncoder dev hidden)
  , qModelTr :: !(TransitionEncoder dev hidden)
  , qModelAct :: !(ActionEncoder dev hidden)
  , qModelSt :: !(StateEncoder dev hidden)
  , qModelFinal1 :: !(TT.Conv2d hidden hidden FifthSize OctaveSize QDType dev) -- !(TT.Linear (EmbSize (QSpecGeneral DefaultQSpec)) QOutHidden QDType dev)
  , qModelAtt1 :: !(TT.MultiheadAttention hidden hidden hidden 1 QDType dev)
  , qModelNorm1 :: !(TT.LayerNorm '[hidden] QDType dev)
  , qModelNorm2 :: !(TT.LayerNorm '[hidden] QDType dev)
  , qModelFinal2 :: !(TT.Linear hidden 1 QDType dev)
  , qModelValue1 :: !(TT.Linear hidden hidden QDType dev)
  , qModelValueNorm :: !(TT.LayerNorm '[hidden] QDType dev)
  , qModelValue2 :: !(TT.Linear hidden 1 QDType dev)
  }
  deriving (Show, Generic, TT.Parameterized, NoThunks, NFData)

type ModelParams dev hidden = TT.Parameters (QModel dev hidden)
type ModelTensors dev hidden = ToModelTensors (ModelParams dev hidden)

instance (ValidParams dev hidden) => T.Randomizable (QSpec dev hidden) (QModel dev hidden) where
  sample :: QSpec dev hidden -> IO (QModel dev hidden)
  sample QSpec = do
    qModelSlc <- T.sample $ SliceSpec @dev @hidden
    qModelTr <- T.sample $ TransitionSpec @dev @hidden
    qModelAct <- T.sample $ ActionSpec @dev @hidden
    qModelSt <- T.sample $ StateSpec @dev @hidden
    qModelFinal1 <- T.sample TT.Conv2dSpec
    qModelAtt1 <- T.sample $ TT.MultiheadAttentionSpec $ TT.DropoutSpec 0
    qModelNorm1 <- T.sample $ TT.LayerNormSpec 1e-05
    qModelNorm2 <- T.sample $ TT.LayerNormSpec 1e-05
    qModelFinal2 <- T.sample TT.LinearSpec
    qModelValue1 <- T.sample TT.LinearSpec
    qModelValueNorm <- T.sample $ TT.LayerNormSpec 1e-05
    qModelValue2 <- T.sample TT.LinearSpec
    pure QModel{..}

{- | A loss for any model with 0 gradients everywhere.
Can be used to ensure that all parameters have a gradient,
if not all parameters are used in the real loss.
-}
fakeLoss
  :: forall dev hidden ps
   . (IsValidDevice dev, ps ~ TT.Parameters (QModel dev hidden))
  => QModel dev hidden
  -> QTensor dev '[]
fakeLoss model = tzero * total
 where
  tzero :: QTensor dev '[]
  tzero = TT.zeros
  params = TT.flattenParameters model
  deps :: (TT.HMap' TT.ToDependent ps ys) => TT.HList ys
  deps = TT.hmap' TT.ToDependent params
  sums = TT.hmap' TH.SumAll deps
  -- total
  total = TT.hfoldr TH.Add tzero sums

mkQModel :: forall dev hidden. (ValidParams dev hidden) => IO (QModel dev hidden)
mkQModel = T.sample $ QSpec @dev @hidden

loadModel :: forall dev hidden. (ValidParams dev hidden) => FilePath -> IO (QModel dev hidden)
loadModel path = do
  modelPlaceholder <- mkQModel @dev
  tensors
    :: (TT.HMap' TT.ToDependent (TT.Parameters (QModel dev hidden)) ts)
    => TT.HList ts <-
    TT.load path
  -- TT.load doesn't move the parameters to the correct device, so we move them manually
  let tensorsCPU = TT.toDevice @'(TT.CPU, 0) @dev tensors
  let tensorsDevice = TT.toDevice @dev @'(TT.CPU, 0) tensorsCPU
  params <- TT.hmapM' TT.MakeIndependent tensorsDevice
  pure $ TT.replaceParameters modelPlaceholder params

saveModel :: FilePath -> QModel dev hidden -> IO ()
saveModel path model = TT.save (TT.hmap' TT.ToDependent $ TT.flattenParameters model) path

modelSize :: (IsValidHidden hidden) => QModel dev hidden -> Int
modelSize model = sum $ product <$> sizes
 where
  sizes = TT.hfoldr TH.ToList ([] :: [[Int]]) $ TT.hmap' TH.ShapeVal $ TT.flattenParameters model

-- | HasForward for model (unbatched)
instance
  ( ValidParams dev hidden
  )
  => T.HasForward (QModel dev hidden) (QEncoding dev '[]) (QTensor dev '[1])
  where
  forward :: QModel dev hidden -> QEncoding dev '[] -> QTensor dev '[1]
  forward model encoding = TT.log $ TT.sigmoid $ forwardQModel model encoding

  forwardStoch :: QModel dev hidden -> QEncoding dev '[] -> IO (QTensor dev '[1])
  forwardStoch model input = pure $ T.forward model input

-- | HasForward for model (batched)
instance
  ( ValidParams dev hidden
  , KnownNat batchSize
  , 1 <= batchSize
  )
  => T.HasForward (QModel dev hidden) (QEncoding dev '[batchSize]) (QTensor dev '[batchSize, 1])
  where
  forward :: QModel dev hidden -> QEncoding dev '[batchSize] -> QTensor dev '[batchSize, 1]
  forward model encoding =
    TT.log $ TT.sigmoid $ forwardQModelBatched model encoding

  forwardStoch model input = pure $ T.forward model input

forwardQModel
  :: (ValidParams dev hidden)
  => QModel dev hidden
  -> QEncoding dev '[]
  -> QTensor dev '[1]
forwardQModel model input = TT.squeezeDim @0 $ forwardQModelBatched model $ addBatchDim input

forwardQModelBatched
  :: forall dev hidden batchSize
   . ( ValidParams dev hidden
     , 1 <= batchSize
     , KnownNat batchSize
     )
  => QModel dev hidden
  -> QEncoding dev '[batchSize]
  -> QTensor dev '[batchSize, 1]
forwardQModelBatched (QModel slc tr act st final1 att1 norm1 norm2 final2 _ _ _) (QEncoding actEncs stEnc) = out2
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

forwardPolicy
  :: (_)
  => QModel dev hidden
  -> QEncoding dev '[]
  -> QTensor dev '[1]
forwardPolicy = forwardQModel

forwardPolicyBatched
  :: forall dev hidden batchSize
   . (_)
  => QModel dev hidden
  -> QEncoding dev '[batchSize]
  -> QTensor dev '[batchSize, 1]
forwardPolicyBatched = forwardQModelBatched

forwardValue
  :: (ValidParams dev hidden)
  => QModel dev hidden
  -> StateEncoding dev
  -> QTensor dev '[1]
forwardValue (QModel slc tr _ st _ _ _ _ _ value1 norm value2) stateEncoding = out2
 where
  outSlc = TT.sumDim @1 $ TT.sumDim @1 $ T.forward st (slc, tr, stateEncoding)
  out1 = activation $ T.forward norm $ T.forward value1 outSlc
  out2 = TT.log $ TT.sigmoid $ T.forward value2 out1

runQ
  :: (ValidParams dev hidden)
  => (s -> a -> QEncoding dev '[])
  -> QModel dev hidden
  -> s
  -> a
  -> QType
runQ !encode !model s a = T.asValue $ TT.toDynamic $ T.forward model $ encode s a

runQ'
  :: (ValidParams dev hidden)
  => (s -> a -> QEncoding dev '[])
  -> QModel dev hidden
  -> s
  -> a
  -> QTensor dev '[1]
runQ' !encode !model s a = T.forward model $ encode s a

data SomePolicy dev = forall batchSize. (KnownNat batchSize) => SomePolicy (QTensor dev '[batchSize, 1])

dynPolicy :: SomePolicy dev -> T.Tensor
dynPolicy (SomePolicy p) = TT.toDynamic p

runBatchedPolicy
  :: forall dev hidden batchSize
   . ( ValidParams dev hidden
     , KnownNat batchSize
     )
  => QType
  -> QModel dev hidden
  -> QEncoding dev '[batchSize]
  -> SomePolicy dev
runBatchedPolicy temp actor encoding = SomePolicy $ TT.softmax @0 $ TT.mulScalar (1 / temp) policy
 where
  policy :: QTensor dev '[batchSize, 1]
  policy = case cmpNat (Proxy @1) (Proxy @batchSize) of
    EQI -> forwardPolicyBatched @dev @hidden @batchSize actor encoding
    LTI -> forwardPolicyBatched @dev @hidden @batchSize actor encoding
    GTI -> error "batched policy: no actions"

runBatchedLogPolicy
  :: forall dev hidden batchSize
   . ( ValidParams dev hidden
     , KnownNat batchSize
     )
  => QType
  -> QModel dev hidden
  -> QEncoding dev '[batchSize]
  -> SomePolicy dev
runBatchedLogPolicy temp actor encoding = SomePolicy $ TT.logSoftmax @0 $ TT.mulScalar (1 / temp) policy
 where
  policy :: QTensor dev '[batchSize, 1]
  policy = case cmpNat (Proxy @1) (Proxy @batchSize) of
    EQI -> forwardPolicyBatched @dev @hidden @batchSize actor encoding
    LTI -> forwardPolicyBatched @dev @hidden @batchSize actor encoding
    GTI -> error "batched policy: no actions"

runBatchedQ
  :: forall dev hidden batchSize
   . ( ValidParams dev hidden
     , KnownNat batchSize
     )
  => QModel dev hidden
  -> QEncoding dev '[batchSize]
  -> SomePolicy dev
runBatchedQ actor encoding = SomePolicy policy
 where
  policy :: QTensor dev '[batchSize, 1]
  policy = case cmpNat (Proxy @1) (Proxy @batchSize) of
    EQI -> forwardPolicyBatched @dev @hidden @batchSize actor encoding
    LTI -> forwardPolicyBatched @dev @hidden @batchSize actor encoding
    GTI -> error "batched policy: no actions"

forwardPolicyFullyBatched
  :: forall dev hidden
   . ( ValidParams dev hidden
     )
  => QModel dev hidden
  -> QEncodingBatch dev
  -> [T.Tensor] -- TODO: could be changed to SomePolicy?
forwardPolicyFullyBatched (QModel slc tr act st final1 att1 norm1 norm2 final2 _ _ _) (QEncodingBatch @dev @batchSize actsEnc stEncs sizes) =
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

runFullyBatchedLogPolicy
  :: ( ValidParams dev hidden
     )
  => QType
  -> QModel dev hidden
  -> QEncodingBatch dev
  -> [T.Tensor]
runFullyBatchedLogPolicy temp model batch = fmap activate $ forwardPolicyFullyBatched model batch
 where
  activate = T.logSoftmax (T.Dim 0) . T.mulScalar (1 / temp)
