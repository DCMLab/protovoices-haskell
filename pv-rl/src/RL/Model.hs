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

-- {-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
-- {-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}

module RL.Model where

import Common
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
import GreedyParser (DoubleParent (DoubleParent), SingleParent (SingleParent))
import NoThunks.Class (NoThunks (..), OnlyCheckWhnf (..), allNoThunks)
import RL.Encoding
import RL.ModelTypes
import RL.TorchHelpers (withBatchDim)
import RL.TorchHelpers qualified as TH
import Torch qualified as T
import Torch.Jit qualified as TJit
import Torch.Lens qualified as TL
import Torch.Typed qualified as TT

import System.IO.Unsafe
import Torch.Internal.Cast (cast2)
import Torch.Internal.Managed.Type.Tensor qualified as ATen

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

data SliceSpec dev = SliceSpec

data SliceEncoder dev = SliceEncoder
  { _slcL1 :: !(TT.Conv2d 1 QSliceHidden 1 1 QDType dev) -- !(TT.Linear (PSize spec) hidden QDType QDevice)
  , _slcL2 :: !(TT.Conv2d QSliceHidden EmbSize FifthSize OctaveSize QDType dev) -- !(TT.Linear hidden (EmbSize spec) QDType QDevice)
  , _slcStart :: !(ConstEmb dev EmbShape)
  , _slcStop :: !(ConstEmb dev EmbShape)
  -- TODO: learn embedding for empty slice
  }
  deriving (Show, Generic, TT.Parameterized, NoThunks, NFData)

instance (IsValidDevice dev) => T.Randomizable (SliceSpec dev) (SliceEncoder dev) where
  sample :: SliceSpec dev -> IO (SliceEncoder dev)
  sample _ =
    SliceEncoder
      <$> T.sample TT.Conv2dSpec
      <*> T.sample TT.Conv2dSpec
      <*> T.sample (ConstEmbSpec @dev)
      <*> T.sample (ConstEmbSpec @dev)

-- | HasFoward for slice (unbatched)
instance
  (embshape ~ EmbShape, IsValidDevice dev)
  => T.HasForward (SliceEncoder dev) (SliceEncoding dev '[]) (QTensor dev embshape)
  where
  forward (SliceEncoder l1 l2 _ _) slice = TT.squeezeDim @0 out2
   where
    input = TT.unsqueeze @0 $ TT.unsqueeze @0 $ getSlice slice
    out1 :: QTensor dev (1 : QSliceHidden : PShape)
    out1 = TT.conv2dForward @'(1, 1) @'(0, 0) l1 input
    out2 :: QTensor dev (1 : EmbShape)
    out2 = activation $ TT.conv2dForward @'(1, 1) @'(FifthPadding, OctavePadding) l2 out1
  forwardStoch model = pure . T.forward model

-- | HasFoward for slice (batched)
instance
  ( IsValidDevice dev
  , embshape ~ '[batchSize, EmbSize, FifthSize, OctaveSize]
  )
  => T.HasForward (SliceEncoder dev) (SliceEncoding dev '[batchSize]) (QTensor dev embshape)
  where
  forward (SliceEncoder l1 l2 _ _) slice = out2
   where
    input = TT.unsqueeze @1 $ getSlice slice
    out1 :: QTensor dev '[batchSize, QSliceHidden, FifthSize, OctaveSize]
    out1 = activation $ TH.conv2dForwardRelaxed @'(1, 1) @'(0, 0) l1 input
    out2 :: QTensor dev '[batchSize, EmbSize, FifthSize, OctaveSize]
    out2 = activation $ TH.conv2dForwardRelaxed @'(1, 1) @'(FifthPadding, OctavePadding) l2 out1
  forwardStoch model = pure . T.forward model

-- | HasForward for slice wrappend in QStartStop (unbatched).
instance
  (embshape ~ EmbShape, IsValidDevice dev)
  => TT.HasForward (SliceEncoder dev) (QStartStop dev '[] (SliceEncoding dev '[])) (QTensor dev embshape)
  where
  forward model@(SliceEncoder _ _ start stop) (QStartStop tag input) = TT.squeezeDim @0 out
   where
    -- compute the possible outputs for start/stop/inner
    outStart :: QTensor dev (EmbSize : PShape)
    outStart = TT.forward start ()
    outStop :: QTensor dev (EmbSize : PShape)
    outStop = TT.forward stop ()
    outInner :: QTensor dev (EmbSize : PShape)
    outInner = T.forward model input
    -- combine the outputs into one tensor
    combined :: QTensor dev (3 : EmbSize : PShape)
    combined = TT.stack @0 $ outStart TT.:. outInner TT.:. outStop TT.:. TT.HNil
    -- use gather to select the right output.
    -- gather can select different elements from 'dim' for each position,
    -- so we expand the tag to the right shape, selecting the *same* 'dim'-index everywhere
    tag' :: TT.Tensor dev TT.Int64 (1 : EmbSize : PShape)
    tag' = TT.expand False $ TT.reshape @[1, 1, 1, 1] tag
    out :: QTensor dev (1 : EmbSize : PShape)
    out = TT.gatherDim @0 tag' combined
  forwardStoch model input = pure $ T.forward model input

-- | HasForward for slice wrapped in QStartStop (batched).
instance
  ( IsValidDevice dev
  , embshape ~ (batchSize : EmbSize : PShape)
  )
  => TT.HasForward (SliceEncoder dev) (QStartStop dev '[batchSize] (SliceEncoding dev '[batchSize])) (QTensor dev embshape)
  where
  forward model@(SliceEncoder _ _ start stop) (QStartStop tag input) = TT.squeezeDim @0 out
   where
    -- compute the possible outputs for start/stop/inner
    outStart :: QTensor dev (batchSize : EmbSize : PShape)
    outStart = TT.UnsafeMkTensor $ expandAs (TT.toDynamic $ TT.forward start ()) $ TT.toDynamic outInner
    outStop :: QTensor dev (batchSize : EmbSize : PShape)
    outStop = TT.UnsafeMkTensor $ expandAs (TT.toDynamic $ TT.forward stop ()) $ TT.toDynamic outInner
    outInner :: QTensor dev (batchSize : EmbSize : PShape)
    outInner = T.forward model input
    -- combine the outputs into one tensor
    combined :: QTensor dev (3 : batchSize : EmbSize : PShape)
    combined = TT.stack @0 $ outStart TT.:. outInner TT.:. outStop TT.:. TT.HNil
    -- use gather to select the right output.
    -- gather can select different elements from 'dim' for each position,
    -- so we expand the tag to the right shape, selecting the *same* 'dim'-index everywhere
    tag' :: TT.Tensor dev 'TT.Int64 (1 : batchSize : EmbSize : PShape)
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

data TransitionSpec dev = TransitionSpec

data TransitionEncoder dev = TransitionEncoder
  { trL1Passing :: !(TT.Conv2d 2 QTransHidden FifthSize OctaveSize QDType dev) -- !(TT.Linear (ESize spec) hidden QDType QDevice)
  , trL1Inner :: !(TT.Conv2d 2 QTransHidden FifthSize OctaveSize QDType dev) -- !(TT.Linear (ESize spec) hidden QDType QDevice)
  , trL1Left :: !(TT.Conv2d 1 QTransHidden 1 1 QDType dev) -- !(TT.Linear (PSize spec) hidden QDType QDevice)
  , trL1Right :: !(TT.Conv2d 1 QTransHidden 1 1 QDType dev) -- !(TT.Linear (PSize spec) hidden QDType QDevice)
  , trL1Root :: !(ConstEmb dev '[QTransHidden])
  , trL2 :: !(TT.Conv2d QTransHidden (EmbSize) FifthSize OctaveSize QDType dev) -- !(TT.Linear hidden (EmbSize) QDType QDevice)
  }
  deriving (Show, Generic, TT.Parameterized, NoThunks, NFData)

instance (IsValidDevice dev) => T.Randomizable (TransitionSpec dev) (TransitionEncoder dev) where
  sample :: TransitionSpec dev -> IO (TransitionEncoder dev)
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
  forall dev embshape
   . ( IsValidDevice dev
     , embshape ~ (EmbSize : PShape)
     )
  => T.HasForward (TransitionEncoder dev) (TransitionEncoding dev '[]) (QTensor dev embshape)
  where
  forward TransitionEncoder{..} TransitionEncoding{..} =
    TT.squeezeDim @0 $
      activation $
        TT.conv2dForward @'(1, 1) @'(FifthPadding, OctavePadding) trL2 $
          TT.unsqueeze @0 all
   where
    runConv
      :: (KnownNat nin)
      => TT.Conv2d nin QTransHidden FifthSize OctaveSize QDType dev
      -> QBoundedList dev QDType MaxEdges '[] (nin : PShape)
      -> QTensor dev (QTransHidden : PShape)
    runConv conv (QBoundedList mask edges) = TT.sumDim @0 $ TT.mul mask' out
     where
      out :: QTensor dev (MaxEdges : QTransHidden : PShape)
      out = activation $ TT.conv2dForward @'(1, 1) @'(FifthPadding, OctavePadding) conv edges
      mask' :: QTensor dev '[MaxEdges, 1, 1, 1]
      mask' = TT.reshape mask
    runSlice conv slice = TT.squeezeDim @0 $ TT.conv2dForward @'(1, 1) @'(0, 0) conv input
     where
      input = TT.unsqueeze @0 $ TT.unsqueeze @0 slice
    pass :: QTensor dev (QTransHidden : PShape)
    pass = runConv trL1Passing trencPassing
    inner :: QTensor dev (QTransHidden : PShape)
    inner = runConv trL1Inner trencInner
    left :: QTensor dev (QTransHidden : PShape)
    left = runSlice trL1Left $ getSlice trencLeft
    right :: QTensor dev (QTransHidden : PShape)
    right = runSlice trL1Right $ getSlice trencRight
    root :: QTensor dev '[QTransHidden, 1, 1]
    root = TT.reshape $ TT.mul trencRoot (activation (T.forward trL1Root ()))
    all :: QTensor dev (QTransHidden : PShape)
    all = (pass + inner + left + right) `TT.add` root

  forwardStoch tr input = pure $ T.forward tr input

-- | HasForward for transitions (batched)
instance
  forall dev batchSize embshape
   . ( IsValidDevice dev
     , embshape ~ (batchSize : EmbSize : PShape)
     )
  => T.HasForward (TransitionEncoder dev) (TransitionEncoding dev '[batchSize]) (QTensor dev embshape)
  where
  forward TransitionEncoder{..} TransitionEncoding{..} =
    activation $ TH.conv2dForwardRelaxed @'(1, 1) @'(FifthPadding, OctavePadding) trL2 all
   where
    runConv
      :: forall nin
       . (KnownNat nin)
      => TT.Conv2d nin QTransHidden FifthSize OctaveSize QDType dev
      -> QBoundedList dev QDType MaxEdges '[batchSize] (nin : PShape)
      -> QTensor dev (batchSize : QTransHidden : PShape)
    runConv conv (QBoundedList mask edges) = TT.sumDim @1 $ TT.mul mask' outReshaped
     where
      shape = TT.shapeVal @(nin : PShape)
      shape' = TT.shapeVal @(MaxEdges : QTransHidden : PShape)
      inputShaped :: QTensor dev (batchSize * MaxEdges : nin : PShape)
      inputShaped = unsafeReshape (-1 : shape) edges
      out :: QTensor dev (batchSize * MaxEdges : QTransHidden : PShape)
      out = activation $ TH.conv2dForwardRelaxed @'(1, 1) @'(FifthPadding, OctavePadding) conv inputShaped
      outReshaped :: QTensor dev (batchSize : MaxEdges : QTransHidden : PShape)
      outReshaped = unsafeReshape (-1 : shape') out
      mask' :: QTensor dev '[batchSize, MaxEdges, 1, 1, 1]
      mask' = unsafeReshape [-1, TT.natValI @MaxEdges, 1, 1, 1] mask
    runSlice conv slice = TH.conv2dForwardRelaxed @'(1, 1) @'(0, 0) conv input
     where
      input = TT.unsqueeze @1 slice
    pass :: QTensor dev (batchSize : QTransHidden : PShape)
    pass = runConv trL1Passing trencPassing
    inner :: QTensor dev (batchSize : QTransHidden : PShape)
    inner = runConv trL1Inner trencInner
    left :: QTensor dev (batchSize : QTransHidden : PShape)
    left = runSlice trL1Left $ getSlice trencLeft
    right :: QTensor dev (batchSize : QTransHidden : PShape)
    right = runSlice trL1Right $ getSlice trencRight
    root :: QTensor dev '[batchSize, QTransHidden, 1, 1]
    root = unsafeReshape [-1, TT.natValI @QTransHidden, 1, 1] $ TT.mul (TT.unsqueeze @1 trencRoot) $ activation $ T.forward trL1Root ()
    all :: QTensor dev (batchSize : QTransHidden : PShape)
    all = (pass + inner + left + right) `TT.add` root

  forwardStoch tr input = pure $ T.forward tr input

-- ActionEncoder
-- -------------

data ActionSpec dev = ActionSpec

data ActionEncoder dev = ActionEncoder
  { actTop1sl :: !(TT.Conv2d EmbSize QActionHidden FifthSize OctaveSize QDType dev) -- TT.Linear (EmbSize) hidden QDType dev
  , actTop1sm :: !(TT.Conv2d EmbSize QActionHidden FifthSize OctaveSize QDType dev) -- TT.Linear (EmbSize) hidden QDType dev
  , actTop1sr :: !(TT.Conv2d EmbSize QActionHidden FifthSize OctaveSize QDType dev) -- TT.Linear (EmbSize) hidden QDType dev
  , actTop1t1 :: !(TT.Conv2d EmbSize QActionHidden FifthSize OctaveSize QDType dev) -- TT.Linear (EmbSize) hidden QDType dev
  , actTop1t2 :: !(TT.Conv2d EmbSize QActionHidden FifthSize OctaveSize QDType dev) -- TT.Linear (EmbSize) hidden QDType dev
  , actTop2 :: !(TT.Conv2d QActionHidden EmbSize FifthSize OctaveSize QDType dev) -- TT.Linear hidden (EmbSize) QDType dev
  , actSplit :: ConstEmb dev '[EmbSize - 3] -- TODO: fill in with actual module
  , actSpread :: ConstEmb dev '[EmbSize - 3] -- TODO: fill in with actual module
  , actFreeze :: ConstEmb dev '[EmbSize - 3]
  }
  deriving (Show, Generic, TT.Parameterized, NoThunks, NFData)

instance (IsValidDevice dev) => T.Randomizable (ActionSpec dev) (ActionEncoder dev) where
  sample :: ActionSpec dev -> IO (ActionEncoder dev)
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
  forall dev batchSize outShape
   . ( IsValidDevice dev
     , outShape ~ (batchSize : EmbSize : PShape)
     , 1 <= batchSize
     )
  => T.HasForward
      (ActionEncoder dev)
      (SliceEncoder dev, TransitionEncoder dev, ActionEncoding dev '[batchSize])
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
    embl :: QTensor dev (batchSize : QActionHidden : PShape)
    embl = runConv actTop1sl $ T.forward slc sl
    embm = runConv actTop1sm $ T.forward slc sm
    embr = runConvMasked smMask actTop1sr $ T.forward slc sr
    embt1 = runConv actTop1t1 $ T.forward tr t1
    embt2 = runConvMasked t2Mask actTop1t2 $ T.forward tr t2
    topCombined :: QTensor dev (batchSize : QActionHidden : PShape)
    topCombined = embl + embm + embr + embt1 + embt2
    topEmb :: QTensor dev (batchSize : EmbSize : PShape)
    topEmb = runConv actTop2 topCombined
    -- operation embedding
    opFreeze = T.forward actFreeze ()
    opSplit = T.forward actSplit ()
    opSpread = T.forward actSpread ()
    opCombined = TT.stack @0 $ opFreeze TT.:. opSplit TT.:. opFreeze TT.:. opSpread TT.:. opSplit TT.:. opSplit TT.:. TT.HNil
    opEmbeddings :: QTensor dev '[6, EmbSize]
    opEmbeddings = TT.cat @1 $ opTypes @dev TT.:. opCombined TT.:. TT.HNil
    opIndex' :: TT.Tensor dev TT.Int64 [batchSize, EmbSize]
    opIndex' = TT.UnsafeMkTensor $ T.expand (TT.toDynamic $ TT.unsqueeze @1 opIndex) False [-1, TT.natValI @EmbSize]
    opEmb :: QTensor dev '[batchSize, EmbSize]
    opEmb = TT.gatherDim @0 opIndex' opEmbeddings
    opEmbReshaped :: QTensor dev '[batchSize, EmbSize, 1, 1]
    opEmbReshaped = TT.unsqueeze @3 $ TT.unsqueeze @2 opEmb
  forwardStoch a i = pure $ T.forward a i

-- State Encoder
-- -------------

data StateSpec dev = StateSpec

data StateEncoder dev = StateEncoder
  { stL1mid :: TT.Conv2d (EmbSize) QStateHidden FifthSize OctaveSize QDType dev
  , stL1frozenSlc :: TT.Conv2d (EmbSize) QStateHidden FifthSize OctaveSize QDType dev
  , stL1frozenTr :: TT.Conv2d (EmbSize) QStateHidden FifthSize OctaveSize QDType dev
  , stL1openSlc :: TT.Conv2d (EmbSize) QStateHidden FifthSize OctaveSize QDType dev
  , stL1openTr :: TT.Conv2d (EmbSize) QStateHidden FifthSize OctaveSize QDType dev
  , stL2 :: TT.Conv2d QStateHidden QStateHidden FifthSize OctaveSize QDType dev
  , stL3 :: TT.Conv2d QStateHidden (EmbSize) FifthSize OctaveSize QDType dev
  }
  deriving (Show, Generic, TT.Parameterized, NoThunks, NFData)

instance (IsValidDevice dev) => T.Randomizable (StateSpec dev) (StateEncoder dev) where
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
  forall dev outShape
   . ( IsValidDevice dev
     , outShape ~ (EmbSize : PShape)
     )
  => T.HasForward
      (StateEncoder dev)
      (SliceEncoder dev, TransitionEncoder dev, StateEncoding dev)
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
      :: TT.Conv2d EmbSize QStateHidden FifthSize OctaveSize QDType dev
      -> TT.Conv2d EmbSize QStateHidden FifthSize OctaveSize QDType dev
      -> QMaybe dev '[] (TransitionEncoding dev '[FakeSize], QStartStop dev '[FakeSize] (SliceEncoding dev '[FakeSize]))
      -> QTensor dev (FakeSize : EmbSize : PShape)
    embedSegments trEnc slcEnc (QMaybe mask (ft, fs)) =
      TT.mul (TT.reshape @[1, 1, 1, 1] mask) $ ftEmb + fsEmb
     where
      ftEmb :: QTensor dev (FakeSize : EmbSize : PShape)
      ftEmb = activation $ runConv' trEnc $ T.forward tr ft
      fsEmb :: QTensor dev (FakeSize : EmbSize : PShape)
      fsEmb = activation $ runConv' slcEnc $ T.forward slc fs

    -- embed frozen segments
    frozenEmb :: QTensor dev (EmbSize : PShape)
    frozenEmb = TT.meanDim @0 $ embedSegments stL1frozenTr stL1frozenSlc frozen
    -- embed open segments
    openEmb :: QTensor dev (EmbSize : PShape)
    openEmb = TT.meanDim @0 $ embedSegments stL1openTr stL1openSlc open
    -- embed the mid slice
    midEmb :: QTensor dev (QStateHidden : PShape)
    midEmb = activation $ runConv stL1mid $ T.forward slc mid

    -- combined embeddings and compute output
    fullEmb :: QTensor dev (EmbSize : PShape)
    fullEmb = midEmb + frozenEmb + openEmb
    out2 :: QTensor dev (QStateHidden : PShape)
    out2 = activation $ runConv stL2 fullEmb
    out3 :: QTensor dev (EmbSize : PShape)
    out3 = activation $ runConv stL3 out2
  forwardStoch a i = pure $ T.forward a i

-- Full Q Model
-- ------------

data QSpec dev = QSpec

data QModel dev = QModel
  { qModelSlc :: !(SliceEncoder dev)
  , qModelTr :: !(TransitionEncoder dev)
  , qModelAct :: !(ActionEncoder dev)
  , qModelSt :: !(StateEncoder dev)
  , qModelFinal1 :: !(TT.Conv2d EmbSize QOutHidden FifthSize OctaveSize QDType dev) -- !(TT.Linear (EmbSize (QSpecGeneral DefaultQSpec)) QOutHidden QDType dev)
  , qModelNorm1 :: !(TT.LayerNorm '[QOutHidden] QDType dev)
  , qModelFinal2 :: !(TT.Linear QOutHidden 1 QDType dev)
  , qModelValue1 :: !(TT.Linear EmbSize QOutHidden QDType dev)
  , qModelValueNorm :: !(TT.LayerNorm '[QOutHidden] QDType dev)
  , qModelValue2 :: !(TT.Linear QOutHidden 1 QDType dev)
  }
  deriving (Show, Generic, TT.Parameterized, NoThunks, NFData)

instance (IsValidDevice dev) => T.Randomizable (QSpec dev) (QModel dev) where
  sample :: QSpec dev -> IO (QModel dev)
  sample QSpec = do
    qModelSlc <- T.sample $ SliceSpec @dev
    qModelTr <- T.sample $ TransitionSpec @dev
    qModelAct <- T.sample $ ActionSpec @dev
    qModelSt <- T.sample $ StateSpec @dev
    qModelFinal1 <- T.sample TT.Conv2dSpec
    qModelNorm1 <- T.sample $ TT.LayerNormSpec 1e-05
    qModelFinal2 <- T.sample TT.LinearSpec
    qModelValue1 <- T.sample TT.LinearSpec
    qModelValueNorm <- T.sample $ TT.LayerNormSpec 1e-05
    qModelValue2 <- T.sample TT.LinearSpec
    pure QModel{..}

-- | HasForward for model (unbatched)
instance
  (IsValidDevice dev, TT.CheckIsSuffixOf '[QOutHidden] [1, QOutHidden] (QOutHidden == QOutHidden))
  => T.HasForward (QModel dev) (QEncoding dev '[]) (QTensor dev '[1])
  where
  forward :: QModel dev -> QEncoding dev '[] -> QTensor dev '[1]
  forward model encoding = TT.log $ TT.sigmoid $ forwardQModel model encoding

  forwardStoch :: QModel dev -> QEncoding dev '[] -> IO (QTensor dev '[1])
  forwardStoch model input = pure $ T.forward model input

-- | HasForward for model (batched)
instance
  ( IsValidDevice dev
  , KnownNat batchSize
  , 1 <= batchSize
  , TT.CheckIsSuffixOf '[QOutHidden] [batchSize, QOutHidden] (QOutHidden == QOutHidden)
  )
  => T.HasForward (QModel dev) (QEncoding dev '[batchSize]) (QTensor dev '[batchSize, 1])
  where
  forward :: QModel dev -> QEncoding dev '[batchSize] -> QTensor dev '[batchSize, 1]
  forward model encoding =
    TT.log $ TT.sigmoid $ forwardQModelBatched model encoding

  forwardStoch model input = pure $ T.forward model input

forwardQModel
  :: (IsValidDevice dev)
  => QModel dev
  -> QEncoding dev '[]
  -> QTensor dev '[1]
forwardQModel model input = TT.squeezeDim @0 $ forwardQModelBatched model $ addBatchDim input

forwardQModelBatched
  :: forall dev batchSize
   . ( IsValidDevice dev
     , 1 <= batchSize
     )
  => QModel dev
  -> QEncoding dev '[batchSize]
  -> QTensor dev '[batchSize, 1]
forwardQModelBatched (QModel slc tr act st final1 norm1 final2 _ _ _) (QEncoding actEncs stEnc) = out2
 where
  actEmb :: QTensor dev (batchSize : EmbSize : PShape)
  actEmb = T.forward act (slc, tr, actEncs)
  stEmb :: QTensor dev (EmbSize : PShape)
  stEmb = T.forward st (slc, tr, stEnc)
  inputEmb = actEmb `TT.add` stEmb
  out1 :: QTensor dev (batchSize : QOutHidden : PShape)
  out1 = TH.conv2dForwardRelaxed @'(1, 1) @'(FifthPadding, OctavePadding) final1 inputEmb
  sum1 :: QTensor dev '[batchSize, QOutHidden]
  sum1 = TT.sumDim @2 $ TT.sumDim @2 out1
  out1norm :: QTensor dev '[batchSize, QOutHidden]
  out1norm = activation $ T.forward norm1 sum1
  out2 :: QTensor dev '[batchSize, 1]
  out2 = T.forward final2 out1norm

forwardPolicy
  :: (_)
  => QModel dev
  -> QEncoding dev '[]
  -> QTensor dev '[1]
forwardPolicy = forwardQModel

forwardPolicyBatched
  :: forall dev batchSize
   . (_)
  => QModel dev
  -> QEncoding dev '[batchSize]
  -> QTensor dev '[batchSize, 1]
forwardPolicyBatched = forwardQModelBatched

forwardValue
  :: (IsValidDevice dev)
  => QModel dev
  -> StateEncoding dev
  -> QTensor dev '[1]
forwardValue (QModel slc tr _ st _ _ _ value1 norm value2) stateEncoding = out2
 where
  outSlc = TT.sumDim @1 $ TT.sumDim @1 $ T.forward st (slc, tr, stateEncoding)
  out1 = activation $ T.forward norm $ T.forward value1 outSlc
  out2 = TT.log $ TT.sigmoid $ T.forward value2 out1

{- | A loss for any model with 0 gradients everywhere.
Can be used to ensure that all parameters have a gradient,
if not all parameters are used in the real loss.
-}
fakeLoss
  :: forall dev ps
   . (IsValidDevice dev, ps ~ TT.Parameters (QModel dev))
  => QModel dev
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

mkQModel :: forall dev. (IsValidDevice dev) => IO (QModel dev)
mkQModel = T.sample $ QSpec @dev

loadModel :: forall dev. (IsValidDevice dev) => FilePath -> IO (QModel dev)
loadModel path = do
  modelPlaceholder <- mkQModel @dev
  tensors
    :: (TT.HMap' TT.ToDependent (TT.Parameters (QModel dev)) ts)
    => TT.HList ts <-
    TT.load path
  -- TT.load doesn't move the parameters to the correct device, so we move them manually
  let tensorsCPU = TT.toDevice @'(TT.CPU, 0) @dev tensors
  let tensorsDevice = TT.toDevice @dev @'(TT.CPU, 0) tensorsCPU
  params <- TT.hmapM' TT.MakeIndependent tensorsDevice
  pure $ TT.replaceParameters modelPlaceholder params

saveModel :: FilePath -> QModel dev -> IO ()
saveModel path model = TT.save (TT.hmap' TT.ToDependent $ TT.flattenParameters model) path

modelSize :: QModel dev -> Int
modelSize model = sum $ product <$> sizes
 where
  sizes = TT.hfoldr TH.ToList ([] :: [[Int]]) $ TT.hmap' TH.ShapeVal $ TT.flattenParameters model

runQ
  :: (IsValidDevice dev)
  => (s -> a -> QEncoding dev '[])
  -> QModel dev
  -> s
  -> a
  -> QType
runQ !encode !model s a = T.asValue $ TT.toDynamic $ T.forward model $ encode s a

runQ'
  :: (IsValidDevice dev)
  => (s -> a -> QEncoding dev '[])
  -> QModel dev
  -> s
  -> a
  -> QTensor dev '[1]
runQ' !encode !model s a = T.forward model $ encode s a

runBatchedPolicy
  :: forall dev batchSize
   . (IsValidDevice dev, KnownNat batchSize)
  => QModel dev
  -> QEncoding dev '[batchSize]
  -> T.Tensor
runBatchedPolicy actor encoding = TT.toDynamic $ TT.softmax @0 $ policy
 where
  policy :: QTensor dev '[batchSize, 1]
  policy = case cmpNat (Proxy @1) (Proxy @batchSize) of
    EQI -> forwardPolicyBatched @dev @batchSize actor encoding
    LTI -> forwardPolicyBatched @dev @batchSize actor encoding
    GTI -> error "batched policy: no actions"
