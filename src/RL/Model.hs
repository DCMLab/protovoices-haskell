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
import Internal.TorchHelpers (withBatchDim)
import Internal.TorchHelpers qualified as TH
import NoThunks.Class (NoThunks (..), OnlyCheckWhnf (..), allNoThunks)
import RL.Encoding
import RL.ModelTypes
import Torch qualified as T
import Torch.Jit qualified as TJit
import Torch.Lens qualified as TL
import Torch.Typed qualified as TT

import System.IO.Unsafe
import Torch.Internal.Cast (cast2)
import Torch.Internal.Managed.Type.Tensor qualified as ATen

-- Global Settings
-- ===============

activation :: QTensor shape -> QTensor shape
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

-- orphan instances for NFData
-- ---------------------------

-- deriving newtype instance NFData T.IndependentTensor

-- deriving instance NFData (TT.Parameter dev dtype shape)

-- deriving instance NFData (TT.Linear nin nout dtype dev)

-- orphan instances for NoThunks
-- -----------------------------

deriving via
  OnlyCheckWhnf T.Tensor
  instance
    NoThunks T.Tensor

-- instance NFData T.Tensor where
--   rnf tensor = ()

deriving instance NoThunks (TT.Tensor dev dtype shape)
deriving instance NFData (TT.Tensor dev dtype shape)

deriving newtype instance NoThunks T.IndependentTensor
deriving newtype instance NFData T.IndependentTensor

deriving instance Generic (TT.Parameter dev dtype shape)
deriving newtype instance NoThunks (TT.Parameter dev dtype shape)
deriving newtype instance NFData (TT.Parameter dev dtype shape)

deriving instance NoThunks (TT.Linear nin nout dtype dev)
deriving instance NFData (TT.Linear nin nout dtype dev)

deriving instance NoThunks (TT.Conv2d cin cout k0 k1 dtype dev)
deriving instance NFData (TT.Conv2d cin cout k0 k1 dtype dev)

deriving instance NoThunks (TT.LayerNorm shape dtype dev)
deriving instance NFData (TT.LayerNorm shape dtype dev)

instance NoThunks (TT.HList '[]) where
  showTypeOf _ = "HNil"
  wNoThunks ctxt TT.HNil = pure Nothing

instance (NoThunks x, NoThunks (TT.HList xs)) => NoThunks (TT.HList (x : (xs :: [Type]))) where
  showTypeOf _ = "HCons " <> showTypeOf (Proxy @x)
  wNoThunks ctxt (x TT.:. xs) = allNoThunks [noThunks ctxt x, noThunks ctxt xs]

instance NFData (TT.HList '[]) where
  rnf TT.HNil = ()

instance (NFData x, NFData (TT.HList xs)) => NFData (TT.HList (x : xs :: [Type])) where
  rnf (x TT.:. xs) = deepseq x $ rnf xs

deriving instance Generic (TT.Adam momenta)
deriving instance (NoThunks (TT.HList momenta)) => NoThunks (TT.Adam momenta)
deriving instance (NFData (TT.HList momenta)) => NFData (TT.Adam momenta)

deriving instance Generic TT.GD
deriving instance NoThunks TT.GD
deriving instance NFData TT.GD

-- General Model Constraints
-- -------------------------

type GSpecConstraints spec =
  ( TT.ConvSideCheck FifthSize FifthSize 1 FifthPadding FifthSize
  , TT.ConvSideCheck OctaveSize OctaveSize 1 OctavePadding OctaveSize
  , FifthSize <= FifthSize + 2 * FifthPadding -- TODO: remove
  , OctaveSize <= OctaveSize + 2 * OctavePadding -- TODO: remove
  , KnownNat FifthSize
  , KnownNat OctaveSize
  , KnownNat FifthPadding
  , KnownNat OctavePadding
  , KnownNat PSize -- TODO: remove
  )

-- Learned Constant Embeddings
-- ---------------------------

data ConstEmbSpec (shape :: [Nat]) = ConstEmbSpec

newtype ConstEmb shape = ConstEmb (TT.Parameter QDevice QDType shape)
  deriving (Show, Generic)
  deriving newtype (TT.Parameterized, NFData, NoThunks, Tensorized)

instance
  (TT.TensorOptions shape QDType QDevice)
  => T.Randomizable (ConstEmbSpec shape) (ConstEmb shape)
  where
  sample :: ConstEmbSpec shape -> IO (ConstEmb shape)
  sample ConstEmbSpec = ConstEmb <$> (TT.makeIndependent =<< TT.randn)

instance T.HasForward (ConstEmb size) () (QTensor size) where
  forward :: ConstEmb size -> () -> QTensor size
  forward (ConstEmb emb) () = TT.toDependent emb
  forwardStoch :: ConstEmb size -> () -> IO (QTensor size)
  forwardStoch model input = pure $ T.forward model input

-- Slice Encoder
-- -------------

data SliceSpec = SliceSpec

data SliceEncoder = SliceEncoder
  { _slcL1 :: !(TT.Conv2d 1 QSliceHidden FifthSize OctaveSize QDType QDevice) -- !(TT.Linear (PSize spec) hidden QDType QDevice)
  , _slcL2 :: !(TT.Conv2d QSliceHidden EmbSize FifthSize OctaveSize QDType QDevice) -- !(TT.Linear hidden (EmbSize spec) QDType QDevice)
  , _slcStart :: !(ConstEmb EmbShape)
  , _slcStop :: !(ConstEmb EmbShape)
  -- TODO: learn embedding for empty slice
  }
  deriving (Show, Generic, TT.Parameterized, NoThunks, NFData, Tensorized)

instance T.Randomizable SliceSpec SliceEncoder where
  sample :: SliceSpec -> IO SliceEncoder
  sample _ =
    SliceEncoder
      <$> T.sample TT.Conv2dSpec
      <*> T.sample TT.Conv2dSpec
      <*> T.sample ConstEmbSpec
      <*> T.sample ConstEmbSpec

-- {- | HasForward for slice wrapped in StartStop.
-- Could be removed if StateEncoding is changed to QStartStop.
-- -}
-- instance
--   ( embshape ~ EmbShape
--   )
--   => TT.HasForward SliceEncoder (StartStop (SliceEncoding '[])) (QTensor embshape)
--   where
--   forward model@(SliceEncoder _ _ start stop) input =
--     case input of
--       Inner slc -> T.forward model slc
--       Start -> T.forward start ()
--       Stop -> T.forward stop ()
--   forwardStoch model input = pure $ T.forward model input

-- | HasForward for slice wrappend in QStartStop (unbatched).
instance
  (embshape ~ EmbShape)
  => TT.HasForward SliceEncoder (QStartStop '[] (SliceEncoding '[])) (QTensor embshape)
  where
  forward model@(SliceEncoder _ _ start stop) (QStartStop tag input) = TT.squeezeDim @0 out
   where
    -- compute the possible outputs for start/stop/inner
    outStart :: QTensor (EmbSize : PShape)
    outStart = TT.forward start ()
    outStop :: QTensor (EmbSize : PShape)
    outStop = TT.forward stop ()
    outInner :: QTensor (EmbSize : PShape)
    outInner = T.forward model input
    -- combine the outputs into one tensor
    combined :: QTensor (3 : EmbSize : PShape)
    combined = TT.stack @0 $ outStart TT.:. outInner TT.:. outStop TT.:. TT.HNil
    -- use gather to select the right output.
    -- gather can select different elements from 'dim' for each position,
    -- so we expand the tag to the right shape, selecting the *same* 'dim'-index everywhere
    tag' :: TT.Tensor QDevice TT.Int64 (1 : EmbSize : PShape)
    tag' = TT.expand False $ TT.reshape @[1, 1, 1, 1] tag
    out :: QTensor (1 : EmbSize : PShape)
    out = TT.gatherDim @0 tag' combined
  forwardStoch model input = pure $ T.forward model input

-- | HasForward for slice wrapped in QStartStop (batched).
instance
  ( embshape ~ (batchSize : EmbSize : PShape)
  )
  => TT.HasForward SliceEncoder (QStartStop '[batchSize] (SliceEncoding '[batchSize])) (QTensor embshape)
  where
  forward model@(SliceEncoder _ _ start stop) (QStartStop tag input) = TT.squeezeDim @0 out
   where
    -- compute the possible outputs for start/stop/inner
    outStart :: QTensor (batchSize : EmbSize : PShape)
    outStart = TT.UnsafeMkTensor $ expandAs (TT.toDynamic $ TT.forward start ()) $ TT.toDynamic outInner
    outStop :: QTensor (batchSize : EmbSize : PShape)
    outStop = TT.UnsafeMkTensor $ expandAs (TT.toDynamic $ TT.forward stop ()) $ TT.toDynamic outInner
    outInner :: QTensor (batchSize : EmbSize : PShape)
    outInner = T.forward model input
    -- combine the outputs into one tensor
    combined :: QTensor (3 : batchSize : EmbSize : PShape)
    combined = TT.stack @0 $ outStart TT.:. outInner TT.:. outStop TT.:. TT.HNil
    -- use gather to select the right output.
    -- gather can select different elements from 'dim' for each position,
    -- so we expand the tag to the right shape, selecting the *same* 'dim'-index everywhere
    tag' :: TT.Tensor QDevice 'TT.Int64 (1 : batchSize : EmbSize : PShape)
    tag' =
      TT.UnsafeMkTensor
        $ T.unsqueeze (T.Dim 0)
        $ expandAs
          (T.reshape [-1, 1, 1, 1] $ TT.toDynamic tag)
        $ TT.toDynamic outInner
    out = TT.gatherDim @0 tag' combined
  forwardStoch model input = pure $ T.forward model input

-- | HasFoward for slice (unbatched)
instance
  (embshape ~ EmbShape)
  => T.HasForward SliceEncoder (SliceEncoding '[]) (QTensor embshape)
  where
  forward (SliceEncoder l1 l2 _ _) slice = out
   where
    (QBoundedList mask input) = getSlice slice
    out1 :: QTensor (MaxPitches : QSliceHidden : PShape)
    out1 = activation $ TT.conv2dForward @'(1, 1) @'(FifthPadding, OctavePadding) l1 input
    out2 :: QTensor (MaxPitches : EmbSize : PShape)
    out2 = activation $ TT.conv2dForward @'(1, 1) @'(FifthPadding, OctavePadding) l2 out1
    mask' :: QTensor '[MaxPitches, 1, 1, 1]
    mask' = TT.reshape mask
    outMasked :: QTensor '[MaxPitches, EmbSize, FifthSize, OctaveSize]
    outMasked = TT.mul mask' out2
    out = TT.sumDim @0 outMasked
  forwardStoch model = pure . T.forward model

-- | HasFoward for slice (batched)
instance
  ( embshape ~ '[batchSize, EmbSize, FifthSize, OctaveSize]
  )
  => T.HasForward SliceEncoder (SliceEncoding '[batchSize]) (QTensor embshape)
  where
  forward (SliceEncoder l1 l2 _ _) slice = out
   where
    (QBoundedList mask input) = getSlice slice
    inputShaped :: QTensor '[batchSize * MaxPitches, 1, FifthSize, OctaveSize]
    inputShaped = TT.UnsafeMkTensor $ T.flatten (T.Dim 0) (T.Dim 1) $ TT.toDynamic input -- TT.reshape input
    out1 :: QTensor '[batchSize * MaxPitches, QSliceHidden, FifthSize, OctaveSize]
    out1 = activation $ TH.conv2dForwardRelaxed @'(1, 1) @'(FifthPadding, OctavePadding) l1 inputShaped
    out2 :: QTensor '[batchSize * MaxPitches, EmbSize, FifthSize, OctaveSize]
    out2 = activation $ TH.conv2dForwardRelaxed @'(1, 1) @'(FifthPadding, OctavePadding) l2 out1
    outReshaped :: QTensor '[batchSize, MaxPitches, EmbSize, FifthSize, OctaveSize]
    outReshaped = unsafeReshape (-1 : TT.shapeVal @(MaxPitches : EmbSize : PShape)) out2
    mask' :: QTensor '[batchSize, MaxPitches, 1, 1, 1]
    mask' = TT.UnsafeMkTensor $ T.reshape [-1, TT.natValI @MaxPitches, 1, 1, 1] $ TT.toDynamic mask
    outMasked :: QTensor '[batchSize, MaxPitches, EmbSize, FifthSize, OctaveSize]
    outMasked = TT.mul mask' outReshaped
    out :: QTensor (batchSize : EmbSize : PShape)
    out = TT.sumDim @1 outMasked
  forwardStoch model = pure . T.forward model

-- Transition Encoder
-- ------------------

data TransitionSpec = TransitionSpec

data TransitionEncoder = TransitionEncoder
  { trL1Passing :: !(TT.Conv2d 2 QTransHidden FifthSize OctaveSize QDType QDevice) -- !(TT.Linear (ESize spec) hidden QDType QDevice)
  , trL1Inner :: !(TT.Conv2d 2 QTransHidden FifthSize OctaveSize QDType QDevice) -- !(TT.Linear (ESize spec) hidden QDType QDevice)
  , trL1Left :: !(TT.Conv2d 1 QTransHidden FifthSize OctaveSize QDType QDevice) -- !(TT.Linear (PSize spec) hidden QDType QDevice)
  , trL1Right :: !(TT.Conv2d 1 QTransHidden FifthSize OctaveSize QDType QDevice) -- !(TT.Linear (PSize spec) hidden QDType QDevice)
  , trL1Root :: !(ConstEmb '[QTransHidden])
  , trL2 :: !(TT.Conv2d QTransHidden (EmbSize) FifthSize OctaveSize QDType QDevice) -- !(TT.Linear hidden (EmbSize) QDType QDevice)
  }
  deriving (Show, Generic, TT.Parameterized, NoThunks, NFData, Tensorized)

instance T.Randomizable TransitionSpec TransitionEncoder where
  sample :: TransitionSpec -> IO TransitionEncoder
  sample _ = do
    trL1Passing <- T.sample TT.Conv2dSpec
    trL1Inner <- T.sample TT.Conv2dSpec
    trL1Left <- T.sample TT.Conv2dSpec
    trL1Right <- T.sample TT.Conv2dSpec
    trL1Root <- T.sample ConstEmbSpec
    trL2 <- T.sample TT.Conv2dSpec
    pure $ TransitionEncoder{..}

-- | HasForward for transitions (unbatched)
instance
  forall embshape
   . ( embshape ~ (EmbSize : PShape)
     )
  => T.HasForward TransitionEncoder (TransitionEncoding '[]) (QTensor embshape)
  where
  forward TransitionEncoder{..} TransitionEncoding{..} =
    TT.squeezeDim @0 $
      activation $
        TT.conv2dForward @'(1, 1) @'(FifthPadding, OctavePadding) trL2 $
          TT.unsqueeze @0 all
   where
    runConv
      :: (KnownNat nin)
      => TT.Conv2d nin QTransHidden FifthSize OctaveSize QDType QDevice
      -> QBoundedList QDType MaxEdges '[] (nin : PShape)
      -> QTensor (QTransHidden : PShape)
    runConv conv (QBoundedList mask edges) = TT.sumDim @0 $ TT.mul mask' out
     where
      out :: QTensor (MaxEdges : QTransHidden : PShape)
      out = activation $ TT.conv2dForward @'(1, 1) @'(FifthPadding, OctavePadding) conv edges
      mask' :: QTensor '[MaxEdges, 1, 1, 1]
      mask' = TT.reshape mask
    pass :: QTensor (QTransHidden : PShape)
    pass = runConv trL1Passing trencPassing
    inner :: QTensor (QTransHidden : PShape)
    inner = runConv trL1Inner trencInner
    left :: QTensor (QTransHidden : PShape)
    left = runConv trL1Left $ getSlice trencLeft
    right :: QTensor (QTransHidden : PShape)
    right = runConv trL1Right $ getSlice trencRight
    root :: QTensor '[QTransHidden, 1, 1]
    root = TT.reshape $ TT.mul trencRoot (activation (T.forward trL1Root ()))
    all :: QTensor (QTransHidden : PShape)
    all = (pass + inner + left + right) `TT.add` root

  forwardStoch tr input = pure $ T.forward tr input

-- | HasForward for transitions (batched)
instance
  forall batchSize embshape
   . ( embshape ~ (batchSize : EmbSize : PShape)
     )
  => T.HasForward TransitionEncoder (TransitionEncoding '[batchSize]) (QTensor embshape)
  where
  forward TransitionEncoder{..} TransitionEncoding{..} =
    activation $ TH.conv2dForwardRelaxed @'(1, 1) @'(FifthPadding, OctavePadding) trL2 all
   where
    runConv
      :: forall nin
       . (KnownNat nin)
      => TT.Conv2d nin QTransHidden FifthSize OctaveSize QDType QDevice
      -> QBoundedList QDType MaxEdges '[batchSize] (nin : PShape)
      -> QTensor (batchSize : QTransHidden : PShape)
    runConv conv (QBoundedList mask edges) = TT.sumDim @1 $ TT.mul mask' outReshaped
     where
      shape = TT.shapeVal @(nin : PShape)
      shape' = TT.shapeVal @(MaxEdges : QTransHidden : PShape)
      inputShaped :: QTensor (batchSize * MaxEdges : nin : PShape)
      inputShaped = unsafeReshape (-1 : shape) edges
      out :: QTensor (batchSize * MaxEdges : QTransHidden : PShape)
      out = activation $ TH.conv2dForwardRelaxed @'(1, 1) @'(FifthPadding, OctavePadding) conv inputShaped
      outReshaped :: QTensor (batchSize : MaxEdges : QTransHidden : PShape)
      outReshaped = unsafeReshape (-1 : shape') out
      mask' :: QTensor '[batchSize, MaxEdges, 1, 1, 1]
      mask' = unsafeReshape [-1, TT.natValI @MaxEdges, 1, 1, 1] mask
    pass :: QTensor (batchSize : QTransHidden : PShape)
    pass = runConv trL1Passing trencPassing
    inner :: QTensor (batchSize : QTransHidden : PShape)
    inner = runConv trL1Inner trencInner
    left :: QTensor (batchSize : QTransHidden : PShape)
    left = runConv trL1Left $ getSlice trencLeft
    right :: QTensor (batchSize : QTransHidden : PShape)
    right = runConv trL1Right $ getSlice trencRight
    root :: QTensor '[batchSize, QTransHidden, 1, 1]
    root = unsafeReshape [-1, TT.natValI @QTransHidden, 1, 1] $ TT.mul (TT.unsqueeze @1 trencRoot) $ activation $ T.forward trL1Root ()
    all :: QTensor (batchSize : QTransHidden : PShape)
    all = (pass + inner + left + right) `TT.add` root

  forwardStoch tr input = pure $ T.forward tr input

-- ActionEncoder
-- -------------

data ActionSpec = ActionSpec

data ActionEncoder = ActionEncoder
  { actTop1sl :: !(TT.Conv2d (EmbSize) QActionHidden FifthSize OctaveSize QDType QDevice) -- TT.Linear (EmbSize) hidden QDType QDevice
  , actTop1sm :: !(TT.Conv2d (EmbSize) QActionHidden FifthSize OctaveSize QDType QDevice) -- TT.Linear (EmbSize) hidden QDType QDevice
  , actTop1sr :: !(TT.Conv2d (EmbSize) QActionHidden FifthSize OctaveSize QDType QDevice) -- TT.Linear (EmbSize) hidden QDType QDevice
  , actTop1t1 :: !(TT.Conv2d (EmbSize) QActionHidden FifthSize OctaveSize QDType QDevice) -- TT.Linear (EmbSize) hidden QDType QDevice
  , actTop1t2 :: !(TT.Conv2d (EmbSize) QActionHidden FifthSize OctaveSize QDType QDevice) -- TT.Linear (EmbSize) hidden QDType QDevice
  , actTop2 :: !(TT.Conv2d QActionHidden EmbSize FifthSize OctaveSize QDType QDevice) -- TT.Linear hidden (EmbSize) QDType QDevice
  , actSplit :: ConstEmb '[EmbSize - 3] -- TODO: fill in with actual module
  , actSpread :: ConstEmb '[EmbSize - 3] -- TODO: fill in with actual module
  , actFreeze :: ConstEmb '[EmbSize - 3]
  }
  deriving (Show, Generic, TT.Parameterized, NoThunks, NFData, Tensorized)

instance T.Randomizable ActionSpec ActionEncoder where
  sample :: ActionSpec -> IO ActionEncoder
  sample ActionSpec = do
    actTop1sl <- T.sample TT.Conv2dSpec
    actTop1sm <- T.sample TT.Conv2dSpec
    actTop1sr <- T.sample TT.Conv2dSpec
    actTop1t1 <- T.sample TT.Conv2dSpec
    actTop1t2 <- T.sample TT.Conv2dSpec
    actTop2 <- T.sample TT.Conv2dSpec
    actSplit <- T.sample ConstEmbSpec
    actSpread <- T.sample ConstEmbSpec
    actFreeze <- T.sample ConstEmbSpec
    pure ActionEncoder{..}

opTypes :: QTensor '[6, 3]
opTypes =
  TT.UnsafeMkTensor $!
    T.asTensor' @[[QType]]
      [ [0, 0, 0] -- freeze only
      , [0, 1, 0] -- split only
      , [1, 0, 0] -- freeze left
      , [1, 0, 1] -- spread
      , [1, 1, 0] -- freeze left
      , [1, 1, 1] -- freeze right
      ]
      opts

-- | HasForward for actions (batched)
instance
  forall batchSize outShape
   . ( outShape ~ (batchSize : EmbSize : PShape)
     , 1 <= batchSize
     )
  => T.HasForward
      ActionEncoder
      (SliceEncoder, TransitionEncoder, ActionEncoding '[batchSize])
      (QTensor outShape)
  where
  forward ActionEncoder{..} (slc, tr, ActionEncoding (ActionTop sl t1 (QMaybe smMask sm) (QMaybe t2Mask t2) sr) opIndex) = topEmb `TT.add` opEmbReshaped
   where
    runConv
      :: TT.Conv2d nin nout FifthSize OctaveSize QDType QDevice
      -> QTensor (batchSize : nin : PShape)
      -> QTensor (batchSize : nout : PShape)
    runConv conv input =
      activation $ TH.conv2dForwardRelaxed @'(1, 1) @'(FifthPadding, OctavePadding) conv input
    runConvMasked
      :: QTensor '[batchSize]
      -> TT.Conv2d nin nout FifthSize OctaveSize QDType QDevice
      -> QTensor (batchSize : nin : PShape)
      -> QTensor (batchSize : nout : PShape)
    runConvMasked mask conv input =
      TT.mul (unsafeReshape [-1, 1, 1, 1] mask :: QTensor '[batchSize, 1, 1, 1]) $ runConv conv input
    -- top embedding
    embl :: QTensor (batchSize : QActionHidden : PShape)
    embl = runConv actTop1sl $ T.forward slc sl
    embm = runConv actTop1sm $ T.forward slc sm
    embr = runConvMasked smMask actTop1sr $ T.forward slc sr
    embt1 = runConv actTop1t1 $ T.forward tr t1
    embt2 = runConvMasked t2Mask actTop1t2 $ T.forward tr t2
    topCombined :: QTensor (batchSize : QActionHidden : PShape)
    topCombined = embl + embm + embr + embt1 + embt2
    topEmb :: QTensor (batchSize : EmbSize : PShape)
    topEmb = runConv actTop2 topCombined
    -- operation embedding
    opFreeze = T.forward actFreeze ()
    opSplit = T.forward actSplit ()
    opSpread = T.forward actSpread ()
    opCombined = TT.stack @0 $ opFreeze TT.:. opSplit TT.:. opFreeze TT.:. opSpread TT.:. opSplit TT.:. opSplit TT.:. TT.HNil
    opEmbeddings :: QTensor '[6, EmbSize]
    opEmbeddings = TT.cat @1 $ opTypes TT.:. opCombined TT.:. TT.HNil
    opIndex' :: TT.Tensor QDevice TT.Int64 [batchSize, EmbSize]
    opIndex' = TT.UnsafeMkTensor $ T.expand (TT.toDynamic $ TT.unsqueeze @1 opIndex) False [-1, TT.natValI @EmbSize]
    opEmb :: QTensor '[batchSize, EmbSize]
    opEmb = TT.gatherDim @0 opIndex' opEmbeddings
    opEmbReshaped :: QTensor '[batchSize, EmbSize, 1, 1]
    opEmbReshaped = TT.unsqueeze @3 $ TT.unsqueeze @2 opEmb
  forwardStoch a i = pure $ T.forward a i

-- State Encoder
-- -------------

data StateSpec = StateSpec

data StateEncoder = StateEncoder
  { stL1mid :: TT.Conv2d (EmbSize) QStateHidden FifthSize OctaveSize QDType QDevice
  , stL1frozenSlc :: TT.Conv2d (EmbSize) QStateHidden FifthSize OctaveSize QDType QDevice
  , stL1frozenTr :: TT.Conv2d (EmbSize) QStateHidden FifthSize OctaveSize QDType QDevice
  , stL1openSlc :: TT.Conv2d (EmbSize) QStateHidden FifthSize OctaveSize QDType QDevice
  , stL1openSlc2 :: TT.Conv2d (EmbSize) QStateHidden FifthSize OctaveSize QDType QDevice -- TODO: remove
  , stL1openSlc3 :: TT.Conv2d (EmbSize) QStateHidden FifthSize OctaveSize QDType QDevice -- TODO: remove
  , stL1openTr :: TT.Conv2d (EmbSize) QStateHidden FifthSize OctaveSize QDType QDevice
  , stL1openTr2 :: TT.Conv2d (EmbSize) QStateHidden FifthSize OctaveSize QDType QDevice -- TODO: remove
  , stL1openTr3 :: TT.Conv2d (EmbSize) QStateHidden FifthSize OctaveSize QDType QDevice -- TODO: remove
  , stL2 :: TT.Conv2d QStateHidden QStateHidden FifthSize OctaveSize QDType QDevice
  , stL3 :: TT.Conv2d QStateHidden (EmbSize) FifthSize OctaveSize QDType QDevice
  }
  deriving (Show, Generic, TT.Parameterized, NoThunks, NFData, Tensorized)

instance T.Randomizable StateSpec StateEncoder where
  sample _ = do
    stL1mid <- TT.sample TT.Conv2dSpec
    stL1frozenSlc <- TT.sample TT.Conv2dSpec
    stL1frozenTr <- TT.sample TT.Conv2dSpec
    stL1openSlc <- TT.sample TT.Conv2dSpec
    stL1openSlc2 <- TT.sample TT.Conv2dSpec
    stL1openSlc3 <- TT.sample TT.Conv2dSpec
    stL1openTr <- TT.sample TT.Conv2dSpec
    stL1openTr2 <- TT.sample TT.Conv2dSpec
    stL1openTr3 <- TT.sample TT.Conv2dSpec
    stL2 <- TT.sample TT.Conv2dSpec
    stL3 <- TT.sample TT.Conv2dSpec
    pure StateEncoder{..}

-- | HasForward for the parsing state (doesn't need batching)
instance
  forall outShape
   . (outShape ~ (EmbSize : PShape))
  => T.HasForward
      StateEncoder
      (SliceEncoder, TransitionEncoder, StateEncoding)
      (QTensor outShape)
  where
  forward StateEncoder{..} (slc, tr, StateEncoding mid frozen open) = out3
   where
    -- helpers: running convolutions (batched and unbatched)
    runConv'
      :: (KnownNat nin, KnownNat nout, KnownNat batch)
      => TT.Conv2d nin nout FifthSize OctaveSize QDType QDevice
      -> QTensor (batch : nin : PShape)
      -> QTensor (batch : nout : PShape)
    runConv' conv input = TT.conv2dForward @'(1, 1) @'(FifthPadding, OctavePadding) conv input
    runConv
      :: (KnownNat nin, KnownNat nout)
      => TT.Conv2d nin nout FifthSize OctaveSize QDType QDevice
      -> QTensor (nin : PShape)
      -> QTensor (nout : PShape)
    runConv conv input = TT.squeezeDim @0 $ runConv' conv $ TT.unsqueeze @0 input

    -- embedding segments (open and frozen)
    embedSegments
      :: TT.Conv2d EmbSize QStateHidden FifthSize OctaveSize QDType QDevice
      -> TT.Conv2d EmbSize QStateHidden FifthSize OctaveSize QDType QDevice
      -> QMaybe '[] (TransitionEncoding '[FakeSize], QStartStop '[FakeSize] (SliceEncoding '[FakeSize]))
      -> QTensor (FakeSize : EmbSize : PShape)
    embedSegments trEnc slcEnc (QMaybe mask (ft, fs)) =
      TT.mul (TT.reshape @[1, 1, 1, 1] mask) $ ftEmb + fsEmb
     where
      ftEmb :: QTensor (FakeSize : EmbSize : PShape)
      ftEmb = activation $ runConv' trEnc $ T.forward tr ft
      fsEmb :: QTensor (FakeSize : EmbSize : PShape)
      fsEmb = activation $ runConv' slcEnc $ T.forward slc fs

    -- embed frozen segments
    frozenEmb :: QTensor (EmbSize : PShape)
    frozenEmb = TT.meanDim @0 $ embedSegments stL1frozenTr stL1frozenSlc frozen
    -- embed open segments
    openEmb :: QTensor (EmbSize : PShape)
    openEmb = TT.meanDim @0 $ embedSegments stL1openTr stL1openSlc open
    -- embed the mid slice
    midEmb :: QTensor (QStateHidden : PShape)
    midEmb = activation $ runConv stL1mid $ T.forward slc mid

    -- combined embeddings and compute output
    fullEmb :: QTensor (EmbSize : PShape)
    fullEmb = midEmb + frozenEmb + openEmb
    out2 :: QTensor (QStateHidden : PShape)
    out2 = activation $ runConv stL2 fullEmb
    out3 :: QTensor (EmbSize : PShape)
    out3 = activation $ runConv stL3 out2
  forwardStoch a i = pure $ T.forward a i

-- Full Q Model
-- ------------

data QSpec = QSpec

data QModel = QModel
  { qModelSlc :: !SliceEncoder
  , qModelTr :: !TransitionEncoder
  , qModelAct :: !ActionEncoder
  , qModelSt :: !StateEncoder
  , qModelFinal1 :: !(TT.Conv2d EmbSize QOutHidden FifthSize OctaveSize QDType QDevice) -- !(TT.Linear (EmbSize (QSpecGeneral DefaultQSpec)) QOutHidden QDType QDevice)
  , qModelNorm1 :: !(TT.LayerNorm '[QOutHidden] QDType QDevice)
  , qModelFinal2 :: !(TT.Linear QOutHidden 1 QDType QDevice)
  , qModelValue1 :: !(TT.Linear EmbSize QOutHidden QDType QDevice)
  , qModelValueNorm :: !(TT.LayerNorm '[QOutHidden] QDType QDevice)
  , qModelValue2 :: !(TT.Linear QOutHidden 1 QDType QDevice)
  }
  deriving (Show, Generic, TT.Parameterized, NoThunks, NFData, Tensorized)

instance T.Randomizable QSpec QModel where
  sample :: QSpec -> IO QModel
  sample QSpec = do
    qModelSlc <- T.sample SliceSpec
    qModelTr <- T.sample TransitionSpec
    qModelAct <- T.sample ActionSpec
    qModelSt <- T.sample StateSpec
    qModelFinal1 <- T.sample TT.Conv2dSpec
    qModelNorm1 <- T.sample $ TT.LayerNormSpec 1e-05
    qModelFinal2 <- T.sample TT.LinearSpec
    qModelValue1 <- T.sample TT.LinearSpec
    qModelValueNorm <- T.sample $ TT.LayerNormSpec 1e-05
    qModelValue2 <- T.sample TT.LinearSpec
    pure QModel{..}

-- | HasForward for model (unbatched)
instance
  (TT.CheckIsSuffixOf '[QOutHidden] [1, QOutHidden] (QOutHidden == QOutHidden))
  => T.HasForward QModel (QEncoding '[]) (QTensor '[1])
  where
  forward :: QModel -> QEncoding '[] -> QTensor '[1]
  forward model encoding = TT.log $ TT.sigmoid $ forwardQModel model encoding

  forwardStoch :: QModel -> QEncoding '[] -> IO (QTensor '[1])
  forwardStoch model input = pure $ T.forward model input

-- | HasForward for model (batched)
instance
  ( KnownNat batchSize
  , 1 <= batchSize
  , TT.CheckIsSuffixOf '[QOutHidden] [batchSize, QOutHidden] (QOutHidden == QOutHidden)
  )
  => T.HasForward QModel (QEncoding '[batchSize]) (QTensor '[batchSize, 1])
  where
  forward :: QModel -> QEncoding '[batchSize] -> QTensor '[batchSize, 1]
  forward model encoding =
    TT.log $ TT.sigmoid $ forwardQModelBatched model encoding

  forwardStoch model input = pure $ T.forward model input

forwardQModel
  :: QModel
  -> QEncoding '[]
  -> QTensor '[1]
forwardQModel model input = TT.squeezeDim @0 $ forwardQModelBatched model $ addBatchDim input

forwardQModelBatched
  :: forall batchSize
   . ( 1 <= batchSize
     )
  => QModel
  -> QEncoding '[batchSize]
  -> QTensor '[batchSize, 1]
forwardQModelBatched (QModel slc tr act st final1 norm1 final2 _ _ _) (QEncoding actEncs stEnc) = out2
 where
  actEmb :: QTensor (batchSize : EmbSize : PShape)
  actEmb = T.forward act (slc, tr, actEncs)
  stEmb :: QTensor (EmbSize : PShape)
  stEmb = T.forward st (slc, tr, stEnc)
  inputEmb = actEmb `TT.add` stEmb
  out1 :: QTensor (batchSize : QOutHidden : PShape)
  out1 = TH.conv2dForwardRelaxed @'(1, 1) @'(FifthPadding, OctavePadding) final1 inputEmb
  sum1 :: QTensor '[batchSize, QOutHidden]
  sum1 = TT.sumDim @2 $ TT.sumDim @2 out1
  out1norm :: QTensor '[batchSize, QOutHidden]
  out1norm = activation $ T.forward norm1 sum1
  out2 :: QTensor '[batchSize, 1]
  out2 = T.forward final2 out1norm

forwardPolicy
  :: (_)
  => QModel
  -> QEncoding '[]
  -> QTensor '[1]
forwardPolicy = forwardQModel

forwardPolicyBatched
  :: forall batchSize
   . (_)
  => QModel
  -> QEncoding '[batchSize]
  -> QTensor '[batchSize, 1]
forwardPolicyBatched = forwardQModelBatched

forwardValue
  :: QModel
  -> StateEncoding
  -> QTensor '[1]
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
  :: forall ps
   . (ps ~ TT.Parameters QModel)
  => QModel
  -> QTensor '[]
fakeLoss model = tzero * total
 where
  tzero :: QTensor '[]
  tzero = TT.zeros
  params = TT.flattenParameters model
  deps :: (TT.HMap' TT.ToDependent ps ys) => TT.HList ys
  deps = TT.hmap' TT.ToDependent params
  sums = TT.hmap' TH.SumAll deps
  -- total
  total = TT.hfoldr TH.Add tzero sums

mkQModel :: IO QModel
mkQModel = T.sample QSpec

loadModel :: FilePath -> IO QModel
loadModel path = do
  modelPlaceholder <- mkQModel
  tensors
    :: (TT.HMap' TT.ToDependent (TT.Parameters QModel) ts)
    => TT.HList ts <-
    TT.load path
  -- TT.load doesn't move the parameters to the correct device, so we move them manually
  let tensorsCPU = TT.toDevice @'(TT.CPU, 0) @QDevice tensors
  let tensorsDevice = TT.toDevice @QDevice @'(TT.CPU, 0) tensorsCPU
  params <- TT.hmapM' TT.MakeIndependent tensorsDevice
  pure $ TT.replaceParameters modelPlaceholder params

saveModel :: FilePath -> QModel -> IO ()
saveModel path model = TT.save (TT.hmap' TT.ToDependent $ TT.flattenParameters model) path

modelSize :: QModel -> Int
modelSize model = sum $ product <$> sizes
 where
  sizes = TT.hfoldr TH.ToList ([] :: [[Int]]) $ TT.hmap' TH.ShapeVal $ TT.flattenParameters model

runQ
  :: (s -> a -> QEncoding '[])
  -> QModel
  -> s
  -> a
  -> QType
runQ !encode !model s a = T.asValue $ TT.toDynamic $ T.forward model $ encode s a

runQ'
  :: (s -> a -> QEncoding '[])
  -> QModel
  -> s
  -> a
  -> QTensor '[1]
runQ' !encode !model s a = T.forward model $ encode s a

runBatchedPolicy
  :: forall batchSize
   . (KnownNat batchSize)
  => QModel
  -> QEncoding '[batchSize]
  -> T.Tensor
runBatchedPolicy actor encoding = TT.toDynamic $ TT.softmax @0 $ policy
 where
  policy :: QTensor '[batchSize, 1]
  policy = case cmpNat (Proxy @1) (Proxy @batchSize) of
    EQI -> forwardPolicyBatched @batchSize actor encoding
    LTI -> forwardPolicyBatched @batchSize actor encoding
    GTI -> error "batched policy: no actions"
