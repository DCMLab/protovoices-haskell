{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DerivingVia #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE Strict #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}
{-# HLINT ignore "Use <$>" #-}
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}

module RL.Model where

import Common
import Control.Arrow ((>>>))
import Control.DeepSeq
import Data.Foldable qualified as F
import Data.Function ((&))
import Data.Kind (Type)
import Data.Proxy (Proxy (Proxy))
import Data.Type.Equality (type (:~:) (Refl), type (==))
import Data.TypeNums (KnownInt, KnownNat, Nat, TInt (..), intVal, intVal', type (*), type (+), type (-), type (<=))
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
import Torch.Typed qualified as TT

-- Global Settings
-- ===============

activation :: QTensor shape -> QTensor shape
activation = TT.gelu

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

instance NFData T.Tensor where
  rnf tensor = ()

deriving instance Generic (TT.Tensor dev dtype shape)
deriving instance NoThunks (TT.Tensor dev dtype shape)
deriving instance NFData (TT.Tensor dev dtype shape)

deriving newtype instance NoThunks T.IndependentTensor
deriving newtype instance NFData T.IndependentTensor

-- deriving instance Generic (TT.Parameter dev dtype shape)
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
  ( TT.ConvSideCheck (FifthSize spec) (FifthSize spec) 1 (FifthPadding spec) (FifthSize spec)
  , TT.ConvSideCheck (OctaveSize spec) (OctaveSize spec) 1 (OctavePadding spec) (OctaveSize spec)
  , FifthSize spec <= FifthSize spec + 2 * FifthPadding spec
  , OctaveSize spec <= OctaveSize spec + 2 * OctavePadding spec
  , 3 <= EmbSize spec
  , KnownNat (FifthSize spec)
  , KnownNat (OctaveSize spec)
  , KnownNat (FifthPadding spec)
  , KnownNat (OctavePadding spec)
  , KnownNat (EmbSize spec)
  , KnownNat (PSize spec) -- TODO: remove
  )

-- Learned Constant Embeddings
-- ---------------------------

data ConstEmbSpec (shape :: [Nat]) = ConstEmbSpec

newtype ConstEmb shape = ConstEmb (TT.Parameter QDevice QDType shape)
  deriving (Show, Generic)
  deriving newtype (TT.Parameterized, NFData, NoThunks)

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

data SliceSpec (hidden :: Nat) = SliceSpec

data SliceEncoder spec hidden = SliceEncoder
  { _slcL1 :: !(TT.Conv2d 1 hidden (FifthSize spec) (OctaveSize spec) QDType QDevice) -- !(TT.Linear (PSize spec) hidden QDType QDevice)
  , _slcL2 :: !(TT.Conv2d hidden (EmbSize spec) (FifthSize spec) (OctaveSize spec) QDType QDevice) -- !(TT.Linear hidden (EmbSize spec) QDType QDevice)
  , _slcStart :: !(ConstEmb (EmbShape spec))
  , _slcStop :: !(ConstEmb (EmbShape spec))
  -- TODO: learn embedding for empty slice
  }
  deriving (Show, Generic, TT.Parameterized, NoThunks, NFData)

instance
  ( KnownNat hidden
  , GSpecConstraints spec
  )
  => T.Randomizable (GeneralSpec spec, SliceSpec hidden) (SliceEncoder spec hidden)
  where
  sample :: (GeneralSpec spec, SliceSpec hidden) -> IO (SliceEncoder spec hidden)
  sample _ =
    SliceEncoder
      <$> T.sample TT.Conv2dSpec
      <*> T.sample TT.Conv2dSpec
      <*> T.sample ConstEmbSpec
      <*> T.sample ConstEmbSpec

{- | HasForward for slice wrapped in StartStop.
Could be removed if StateEncoding is changed to QStartStop.
-}
instance
  ( pshape ~ PShape spec
  , embshape ~ EmbShape spec
  , pshape ~ [FifthSize spec, OctaveSize spec]
  , GSpecConstraints spec
  , KnownNat hidden
  )
  => TT.HasForward (SliceEncoder spec hidden) (StartStop (QBoundedList MaxPitches '[] (1 : pshape))) (QTensor embshape)
  where
  forward model@(SliceEncoder _ _ start stop) input =
    case input of
      Inner slc -> T.forward model slc
      Start -> T.forward start ()
      Stop -> T.forward stop ()
  forwardStoch model input = pure $ T.forward model input

-- | HasForward for slice wrappend in QStartStop (unbatched).
instance
  ( embshape ~ EmbShape spec
  , emb ~ EmbSize spec
  , pshape ~ PShape spec
  , pshape ~ [FifthSize spec, OctaveSize spec]
  , TT.KnownShape pshape
  , KnownNat hidden
  , GSpecConstraints spec
  )
  => TT.HasForward (SliceEncoder spec hidden) (QStartStop '[] (QBoundedList MaxPitches '[] (1 : pshape))) (QTensor embshape)
  where
  forward model@(SliceEncoder _ _ start stop) (QStartStop tag input) = TT.squeezeDim @0 out
   where
    -- compute the possible outputs for start/stop/inner
    outStart :: QTensor (emb : pshape)
    outStart = TT.forward start ()
    outStop :: QTensor (emb : pshape)
    outStop = TT.forward stop ()
    outInner :: QTensor (emb : pshape)
    outInner = T.forward model input
    -- combine the outputs into one tensor
    combined :: QTensor (3 : emb : pshape)
    combined = TT.stack @0 $ outStart TT.:. outInner TT.:. outStop TT.:. TT.HNil
    -- use gather to select the right output.
    -- gather can select different elements from 'dim' for each position,
    -- so we expand the tag to the right shape, selecting the *same* 'dim'-index everywhere
    tag' :: TT.Tensor QDevice TT.Int64 (1 : emb : pshape)
    tag' = TT.expand False $ TT.reshape @[1, 1, 1, 1] tag
    out :: QTensor (1 : emb : pshape)
    out = TT.gatherDim @0 tag' combined
  forwardStoch model input = pure $ T.forward model input

-- | HasForward for slice wrappend in QStartStop (batched).
instance
  ( emb ~ EmbSize spec
  , pshape ~ PShape spec
  , embshape ~ (batchSize : emb : pshape)
  , pshape ~ [FifthSize spec, OctaveSize spec]
  , KnownNat batchSize
  , KnownNat hidden
  , TT.KnownShape pshape
  , GSpecConstraints spec
  )
  => TT.HasForward (SliceEncoder spec hidden) (QStartStop '[batchSize] (QBoundedList MaxPitches '[batchSize] (1 : pshape))) (QTensor embshape)
  where
  forward model@(SliceEncoder _ _ start stop) (QStartStop tag input) = TT.squeezeDim @0 out
   where
    -- compute the possible outputs for start/stop/inner
    outStart :: QTensor (batchSize : emb : pshape)
    outStart = TT.expand False $ TT.forward start ()
    outStop :: QTensor (batchSize : emb : pshape)
    outStop = TT.expand False $ TT.forward stop ()
    outInner :: QTensor (batchSize : emb : pshape)
    outInner = T.forward model input
    -- combine the outputs into one tensor
    combined :: QTensor (3 : batchSize : emb : pshape)
    combined = TT.stack @0 $ outStart TT.:. outInner TT.:. outStop TT.:. TT.HNil
    -- use gather to select the right output.
    -- gather can select different elements from 'dim' for each position,
    -- so we expand the tag to the right shape, selecting the *same* 'dim'-index everywhere
    tag' :: TT.Tensor QDevice 'TT.Int64 (1 : batchSize : emb : pshape)
    tag' = TT.expand False $ TT.reshape @[1, batchSize, 1, 1, 1] tag
    out = TT.gatherDim @0 tag' combined
  forwardStoch model input = pure $ T.forward model input

-- -- | HasForward for slice wrappend in QStartStop (unbatched).
-- instance
--   ( embshape ~ '[EmbSize spec]
--   , psize ~ PSize spec
--   , KnownNat (EmbSize spec)
--   )
--   => TT.HasForward (SliceEncoder spec hidden) (QStartStop '[] (QBoundedList MaxPitches '[] '[psize])) (QTensor embshape)
--   where
--   forward model@(SliceEncoder _ _ start stop) (QStartStop tag input) = TT.squeezeDim @0 out
--    where
--     -- compute the possible outputs for start/stop/inner
--     outStart = TT.forward start ()
--     outStop = TT.forward stop ()
--     outInner = T.forward model input
--     -- combine the outputs into one tensor
--     combined = TT.stack @0 $ outStart TT.:. outInner TT.:. outStop TT.:. TT.HNil
--     -- use gather to select the right output.
--     -- gather can select different elements from 'dim' for each position,
--     -- so we expand the tag to the right shape, selecting the *same* 'dim'-index everywhere
--     out = TT.gatherDim @0 (TT.expand @'[1, EmbSize spec] False tag) combined
--   forwardStoch model input = pure $ T.forward model input

-- -- | HasForward for slice wrappend in QStartStop (batched).
-- instance
--   ( emb ~ EmbSize spec
--   , psize ~ PSize spec
--   , embshape ~ '[batchSize, emb]
--   , KnownNat emb
--   , KnownNat batchSize
--   )
--   => TT.HasForward (SliceEncoder spec hidden) (QStartStop '[batchSize] (QBoundedList MaxPitches '[batchSize] '[psize])) (QTensor embshape)
--   where
--   forward model@(SliceEncoder _ _ start stop) (QStartStop tag input) = TT.squeezeDim @0 out
--    where
--     -- compute the possible outputs for start/stop/inner
--     outStart :: QTensor [batchSize, emb]
--     outStart = TT.expand @'[batchSize, emb] False $ TT.forward start ()
--     outStop :: QTensor [batchSize, emb]
--     outStop = TT.expand @'[batchSize, emb] False $ TT.forward stop ()
--     outInner :: QTensor [batchSize, emb]
--     outInner = T.forward model input
--     -- combine the outputs into one tensor
--     combined :: QTensor [3, batchSize, emb]
--     combined = TT.stack @0 $ outStart TT.:. outInner TT.:. outStop TT.:. TT.HNil
--     -- use gather to select the right output.
--     -- gather can select different elements from 'dim' for each position,
--     -- so we expand the tag to the right shape, selecting the *same* 'dim'-index everywhere
--     tag' :: TT.Tensor QDevice 'TT.Int64 [1, batchSize, emb]
--     tag' = TT.expand @'[1, batchSize, emb] False $ TT.unsqueeze @1 tag
--     out = TT.gatherDim @0 tag' combined
--   forwardStoch model input = pure $ T.forward model input

-- | HasFoward for slice (unbatched)
instance
  ( embshape ~ EmbShape spec
  , emb ~ EmbSize spec
  , pshape ~ PShape spec
  , fpad ~ FifthPadding spec
  , opad ~ OctavePadding spec
  , fifths ~ FifthSize spec
  , octs ~ OctaveSize spec
  , KnownNat hidden
  , GSpecConstraints spec
  )
  => T.HasForward (SliceEncoder spec hidden) (QBoundedList MaxPitches '[] (1 : pshape)) (QTensor embshape)
  where
  forward (SliceEncoder l1 l2 _ _) (QBoundedList mask input) = out
   where
    out1 :: QTensor (MaxPitches : hidden : pshape)
    out1 = activation $ TT.conv2dForward @'(1, 1) @'(fpad, opad) l1 input
    out2 :: QTensor (MaxPitches : emb : pshape)
    out2 = activation $ TT.conv2dForward @'(1, 1) @'(fpad, opad) l2 out1
    mask' :: QTensor '[MaxPitches, 1, 1, 1]
    mask' = TT.reshape mask
    outMasked :: QTensor '[MaxPitches, emb, fifths, octs]
    outMasked = TT.mul mask' out2
    out = TT.sumDim @0 outMasked
  forwardStoch model = pure . T.forward model

-- | HasFoward for slice (batched)
instance
  ( pshape ~ PShape spec
  , fifths ~ FifthSize spec
  , octaves ~ OctaveSize spec
  , fpad ~ FifthPadding spec
  , opad ~ OctavePadding spec
  , emb ~ EmbSize spec
  , embshape ~ '[batchSize, emb, fifths, octaves]
  , KnownNat hidden
  , KnownNat batchSize
  , GSpecConstraints spec
  )
  => T.HasForward (SliceEncoder spec hidden) (QBoundedList MaxPitches '[batchSize] (1 : pshape)) (QTensor embshape)
  where
  forward (SliceEncoder l1 l2 _ _) (QBoundedList mask input) =
    case proof of
      (Refl, Refl) -> out
       where
        inputShaped :: QTensor '[batchSize * MaxPitches, 1, fifths, octaves]
        inputShaped = TT.reshape input
        out1 :: QTensor '[batchSize * MaxPitches, hidden, fifths, octaves]
        out1 = activation $ TT.conv2dForward @'(1, 1) @'(fpad, opad) l1 inputShaped
        out2 :: QTensor '[batchSize * MaxPitches, emb, fifths, octaves]
        out2 = activation $ TT.conv2dForward @'(1, 1) @'(fpad, opad) l2 out1
        outReshaped :: QTensor '[batchSize, MaxPitches, emb, fifths, octaves]
        outReshaped = TT.reshape out2
        mask' :: QTensor '[batchSize, MaxPitches, 1, 1, 1]
        mask' = TT.reshape mask
        outMasked :: QTensor '[batchSize, MaxPitches, emb, fifths, octaves]
        outMasked = TT.mul mask' outReshaped
        out :: QTensor (batchSize : emb : pshape)
        out = TT.sumDim @1 outMasked
   where
    -- provides certain cases of associativity for Nat * Nat
    -- maybe this can be proven statically?
    proof =
      case sameNat
        (Proxy @(batchSize * (MaxPitches * (fifths * octaves))))
        (Proxy @((batchSize * MaxPitches) * (fifths * octaves))) of
        Just r1 ->
          case sameNat
            (Proxy @(batchSize * (MaxPitches * (emb * (fifths * octaves)))))
            (Proxy @((batchSize * MaxPitches) * (emb * (fifths * octaves)))) of
            Just r2 -> (r1, r2)
  forwardStoch model = pure . T.forward model

-- -- | HasFoward for slice (unbatched)
-- instance
--   ( embshape ~ '[EmbSize spec]
--   , psize ~ PSize spec
--   )
--   => T.HasForward (SliceEncoder spec hidden) (QBoundedList MaxPitches '[] '[psize]) (QTensor embshape)
--   where
--   forward (SliceEncoder l1 l2 _ _) (QBoundedList mask input) = out
--    where
--     out1 = activation $ T.forward l1 input
--     out2 = activation $ T.forward l2 out1
--     outMasked = TT.mul (TT.unsqueeze @1 mask) out2
--     out = TT.sumDim @0 outMasked
--   forwardStoch model = pure . T.forward model

-- -- | HasFoward for slice (batched)
-- instance
--   ( psize ~ PSize spec
--   , emb ~ EmbSize spec
--   , embshape ~ '[batchSize, emb]
--   )
--   => T.HasForward (SliceEncoder spec hidden) (QBoundedList MaxPitches '[batchSize] '[psize]) (QTensor embshape)
--   where
--   forward (SliceEncoder l1 l2 _ _) (QBoundedList mask input) = out
--    where
--     out1 :: QTensor '[batchSize, MaxPitches, hidden]
--     out1 = activation $ T.forward l1 input
--     out2 :: QTensor '[batchSize, MaxPitches, emb]
--     out2 = activation $ T.forward l2 out1
--     mask' :: QTensor '[batchSize, MaxPitches, 1]
--     mask' = TT.unsqueeze @2 mask
--     outMasked :: QTensor '[batchSize, MaxPitches, emb]
--     outMasked = TT.mul mask' out2
--     out :: QTensor '[batchSize, emb]
--     out = TT.sumDim @1 outMasked
--   forwardStoch model = pure . T.forward model

-- Transition Encoder
-- ------------------

data TransitionSpec (hidden :: Nat) = TransitionSpec

data TransitionEncoder spec hidden = TransitionEncoder
  { trL1Passing :: !(TT.Conv2d 2 hidden (FifthSize spec) (OctaveSize spec) QDType QDevice) -- !(TT.Linear (ESize spec) hidden QDType QDevice)
  , trL1Inner :: !(TT.Conv2d 2 hidden (FifthSize spec) (OctaveSize spec) QDType QDevice) -- !(TT.Linear (ESize spec) hidden QDType QDevice)
  , trL1Left :: !(TT.Conv2d 1 hidden (FifthSize spec) (OctaveSize spec) QDType QDevice) -- !(TT.Linear (PSize spec) hidden QDType QDevice)
  , trL1Right :: !(TT.Conv2d 1 hidden (FifthSize spec) (OctaveSize spec) QDType QDevice) -- !(TT.Linear (PSize spec) hidden QDType QDevice)
  , trL1Root :: !(ConstEmb '[hidden])
  , trL2 :: !(TT.Conv2d hidden (EmbSize spec) (FifthSize spec) (OctaveSize spec) QDType QDevice) -- !(TT.Linear hidden (EmbSize spec) QDType QDevice)
  }
  deriving (Show, Generic, TT.Parameterized, NoThunks, NFData)

instance
  ( KnownNat hidden
  , KnownNat (EmbSize spec)
  , KnownNat (PSize spec)
  , KnownNat (ESize spec)
  , KnownNat (FifthPadding spec)
  , KnownNat (OctavePadding spec)
  )
  => T.Randomizable (GeneralSpec spec, TransitionSpec hidden) (TransitionEncoder spec hidden)
  where
  sample :: (GeneralSpec spec, TransitionSpec hidden) -> IO (TransitionEncoder spec hidden)
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
  forall spec hidden embshape pshape fifths octs fpad opad
   . ( pshape ~ PShape spec
     , fifths ~ FifthSize spec
     , octs ~ OctaveSize spec
     , fpad ~ FifthPadding spec
     , opad ~ OctavePadding spec
     , embshape ~ (EmbSize spec : pshape)
     , KnownNat hidden
     , GSpecConstraints spec
     )
  => T.HasForward
      (TransitionEncoder spec hidden)
      (TransitionEncoding '[] spec)
      (QTensor embshape)
  where
  forward TransitionEncoder{..} TransitionEncoding{..} =
    TT.squeezeDim @0 $
      activation $
        TT.conv2dForward @'(1, 1) @'(fpad, opad) trL2 $
          TT.unsqueeze @0 all
   where
    runConv
      :: (KnownNat nin)
      => TT.Conv2d nin hidden (FifthSize spec) (OctaveSize spec) QDType QDevice
      -> QBoundedList MaxEdges '[] (nin : pshape)
      -> QTensor (hidden : pshape)
    runConv conv (QBoundedList mask edges) = TT.sumDim @0 $ TT.mul mask' out
     where
      out :: QTensor (MaxEdges : hidden : pshape)
      out = activation $ TT.conv2dForward @'(1, 1) @'(fpad, opad) conv edges
      mask' :: QTensor '[MaxEdges, 1, 1, 1]
      mask' = TT.reshape mask
    pass :: QTensor (hidden : pshape)
    pass = runConv trL1Passing trencPassing
    inner :: QTensor (hidden : pshape)
    inner = runConv trL1Inner trencInner
    left :: QTensor (hidden : pshape)
    left = runConv trL1Left trencLeft
    right :: QTensor (hidden : pshape)
    right = runConv trL1Right trencRight
    root :: QTensor '[hidden, 1, 1]
    root = TT.reshape $ TT.mul trencRoot (activation (T.forward trL1Root ()))
    all :: QTensor (hidden : pshape)
    all = (pass + inner + left + right) `TT.add` root

  forwardStoch tr input = pure $ T.forward tr input

-- | HasForward for transitions (batched)
instance
  forall spec hidden embshape pshape fifths octs fpad opad batchSize
   . ( pshape ~ PShape spec
     , fifths ~ FifthSize spec
     , octs ~ OctaveSize spec
     , fpad ~ FifthPadding spec
     , opad ~ OctavePadding spec
     , embshape ~ (batchSize : EmbSize spec : pshape)
     , KnownNat hidden
     , KnownNat batchSize
     , GSpecConstraints spec
     )
  => T.HasForward
      (TransitionEncoder spec hidden)
      (TransitionEncoding '[batchSize] spec)
      (QTensor embshape)
  where
  forward TransitionEncoder{..} TransitionEncoding{..} =
    activation $ TT.conv2dForward @'(1, 1) @'(fpad, opad) trL2 all
   where
    runConv
      :: forall nin
       . (KnownNat nin)
      => TT.Conv2d nin hidden (FifthSize spec) (OctaveSize spec) QDType QDevice
      -> QBoundedList MaxEdges '[batchSize] (nin : pshape)
      -> QTensor (batchSize : hidden : pshape)
    runConv conv (QBoundedList mask edges) = TT.sumDim @1 $ TT.mul mask' outReshaped
     where
      inputShaped :: QTensor (batchSize * MaxEdges : nin : pshape)
      inputShaped = TT.reshape edges
      out :: QTensor (batchSize * MaxEdges : hidden : pshape)
      out = activation $ TT.conv2dForward @'(1, 1) @'(fpad, opad) conv inputShaped
      outReshaped :: QTensor (batchSize : MaxEdges : hidden : pshape)
      outReshaped = TT.reshape out
      mask' :: QTensor '[batchSize, MaxEdges, 1, 1, 1]
      mask' = TT.reshape mask
    pass :: QTensor (batchSize : hidden : pshape)
    pass = runConv trL1Passing trencPassing
    inner :: QTensor (batchSize : hidden : pshape)
    inner = runConv trL1Inner trencInner
    left :: QTensor (batchSize : hidden : pshape)
    left = runConv trL1Left trencLeft
    right :: QTensor (batchSize : hidden : pshape)
    right = runConv trL1Right trencRight
    root :: QTensor '[batchSize, hidden, 1, 1]
    root = TT.reshape $ TT.mul (TT.unsqueeze @1 trencRoot) $ activation $ T.forward trL1Root ()
    all :: QTensor (batchSize : hidden : pshape)
    all = (pass + inner + left + right) `TT.add` root

  forwardStoch tr input = pure $ T.forward tr input

-- -- | HasForward for transitions (unbatched)
-- instance
--   forall spec hidden embshape pshape
--    . ( embshape ~ '[EmbSize spec]
--      , pshape ~ PShape spec
--      , KnownNat hidden
--      , GSpecConstraints spec
--      )
--   => T.HasForward
--       (TransitionEncoder spec hidden)
--       (TransitionEncoding '[] spec)
--       (QTensor embshape)
--   where
--   forward TransitionEncoder{..} TransitionEncoding{..} = undefined -- activation $ T.forward trL2 all
--    where
--     pass :: QTensor '[hidden]
--     pass =
--       let QBoundedList mask edgesPassing = trencPassing
--        in TT.sumDim @0 $ TT.mul (TT.unsqueeze @1 mask) $ activation $ T.forward trL1Passing edgesPassing

--   --  inner :: QTensor '[hidden]
--   --  inner =
--   --    let QBoundedList mask edgesInner = trencInner
--   --     in TT.sumDim @0 $ TT.mul (TT.unsqueeze @1 mask) $ activation $ T.forward trL1Inner edgesInner
--   --  left :: QTensor '[hidden]
--   --  left =
--   --    let QBoundedList mask notes = trencLeft
--   --     in TT.sumDim @0 $ TT.mul (TT.unsqueeze @1 mask) $ activation $ T.forward trL1Left notes
--   --  right :: QTensor '[hidden]
--   --  right =
--   --    let QBoundedList mask notes = trencRight
--   --     in TT.sumDim @0 $ TT.mul (TT.unsqueeze @1 mask) $ activation $ T.forward trL1Right notes
--   --  root = TT.mul trencRoot (activation (T.forward trL1Root ()))
--   --  all = pass + inner + left + right + root

--   forwardStoch tr input = pure $ T.forward tr input

-- -- | HasForward for transitions (batched)
-- instance
--   forall spec hidden emb embshape pshape batchSize
--    . ( emb ~ EmbSize spec
--      , pshape ~ PShape spec
--      , embshape ~ (batchSize : emb : pshape)
--      , KnownNat hidden
--      , KnownNat batchSize
--      )
--   => T.HasForward
--       (TransitionEncoder spec hidden)
--       (TransitionEncoding '[batchSize] spec)
--       (QTensor embshape)
--   where
--   forward TransitionEncoder{..} TransitionEncoding{..} = activation $ T.forward trL2 all
--    where
--     pass :: QTensor (batchSize : hidden : pshape)
--     pass =
--       let QBoundedList mask edgesPassing = trencPassing
--        in TT.sumDim @1 $ TT.mul (TT.unsqueeze @2 mask) $ activation $ T.forward trL1Passing edgesPassing
--     inner :: QTensor (batchSize : hidden : pshape)
--     inner =
--       let QBoundedList mask edgesInner = trencInner
--        in TT.sumDim @1 $ TT.mul (TT.unsqueeze @2 mask) $ activation $ T.forward trL1Inner edgesInner
--     left :: QTensor (batchSize : hidden : pshape)
--     left =
--       let QBoundedList mask notes = trencLeft
--        in TT.sumDim @1 $ TT.mul (TT.unsqueeze @2 mask) $ activation $ T.forward trL1Left notes
--     right :: QTensor (batchSize : hidden : pshape)
--     right =
--       let QBoundedList mask notes = trencRight
--        in TT.sumDim @1 $ TT.mul (TT.unsqueeze @2 mask) $ activation $ T.forward trL1Right notes
--     root :: QTensor (batchSize : hidden : pshape)
--     root =
--       TT.mul (TT.unsqueeze @1 trencRoot) $
--         TT.expand @(batchSize : hidden : pshape) False (activation (T.forward trL1Root ()))
--     all :: QTensor '[batchSize, hidden]
--     all = pass + inner + left + right + root

--   forwardStoch tr input = pure $ T.forward tr input

-- ActionEncoder
-- -------------

data ActionSpec (hidden :: Nat) = ActionSpec

data ActionEncoder spec hidden = ActionEncoder
  { actTop1sl :: !(TT.Conv2d (EmbSize spec) hidden (FifthSize spec) (OctaveSize spec) QDType QDevice) -- TT.Linear (EmbSize spec) hidden QDType QDevice
  , actTop1sm :: !(TT.Conv2d (EmbSize spec) hidden (FifthSize spec) (OctaveSize spec) QDType QDevice) -- TT.Linear (EmbSize spec) hidden QDType QDevice
  , actTop1sr :: !(TT.Conv2d (EmbSize spec) hidden (FifthSize spec) (OctaveSize spec) QDType QDevice) -- TT.Linear (EmbSize spec) hidden QDType QDevice
  , actTop1t1 :: !(TT.Conv2d (EmbSize spec) hidden (FifthSize spec) (OctaveSize spec) QDType QDevice) -- TT.Linear (EmbSize spec) hidden QDType QDevice
  , actTop1t2 :: !(TT.Conv2d (EmbSize spec) hidden (FifthSize spec) (OctaveSize spec) QDType QDevice) -- TT.Linear (EmbSize spec) hidden QDType QDevice
  , actTop2 :: !(TT.Conv2d hidden (EmbSize spec) (FifthSize spec) (OctaveSize spec) QDType QDevice) -- TT.Linear hidden (EmbSize spec) QDType QDevice
  , actSplit :: ConstEmb '[EmbSize spec - 3] -- TODO: fill in with actual module
  , actSpread :: ConstEmb '[EmbSize spec - 3] -- TODO: fill in with actual module
  , actFreeze :: ConstEmb '[EmbSize spec - 3]
  }
  deriving (Show, Generic, TT.Parameterized, NoThunks, NFData)

instance
  ( emb ~ EmbSize spec
  , KnownNat hidden
  , KnownNat (FifthPadding spec)
  , KnownNat (OctavePadding spec)
  , KnownNat emb
  , KnownNat (emb - 3)
  )
  => T.Randomizable (GeneralSpec spec, ActionSpec hidden) (ActionEncoder spec hidden)
  where
  sample :: (GeneralSpec spec, ActionSpec hidden) -> IO (ActionEncoder spec hidden)
  sample (GeneralSpec, ActionSpec) = do
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

-- -- | HasForward for actions (unbatched)
-- instance
--   forall (spec :: TGeneralSpec) slcHidden trHidden actHidden outShape emb fpad opad pshape fifths octs
--    . ( emb ~ EmbSize spec
--      , fpad ~ FifthPadding spec
--      , opad ~ OctavePadding spec
--      , fifths ~ FifthSize spec
--      , octs ~ OctaveSize spec
--      , pshape ~ PShape spec
--      , outShape ~ (emb : pshape)
--      , emb ~ (emb - 3) + 3
--      , KnownNat (PSize spec)
--      , KnownNat emb
--      , KnownNat trHidden
--      , KnownNat slcHidden
--      , KnownNat actHidden
--      , GSpecConstraints spec
--      )
--   => T.HasForward
--       (ActionEncoder spec actHidden)
--       (SliceEncoder spec slcHidden, TransitionEncoder spec trHidden, ActionEncoding '[] spec)
--       (QTensor outShape)
--   where
--   forward ActionEncoder{..} (slc, tr, ActionEncoding (ActionTop sl t1 (QMaybe smMask sm) (QMaybe t2Mask t2) sr) opIndex) = topEmb `TT.add` TT.reshape @[emb, 1, 1] opEmb
--    where
--     runConv
--       :: (KnownNat nin, KnownNat nout)
--       => TT.Conv2d nin nout fifths octs QDType QDevice
--       -> QTensor (nin : pshape)
--       -> QTensor (nout : pshape)
--     runConv conv input =
--       TT.squeezeDim @0 $
--         activation $
--           TT.conv2dForward @'(1, 1) @'(fpad, opad) conv $
--             TT.unsqueeze @0 input
--     embl :: QTensor (actHidden : pshape)
--     embl = runConv actTop1sl $ T.forward slc sl
--     embm = runConv actTop1sm $ T.forward slc sm
--     embr = TT.mul smMask $ runConv actTop1sr $ T.forward slc sr
--     embt1 = TT.mul t2Mask $ runConv actTop1t1 $ T.forward tr t1
--     embt2 = runConv actTop1t2 $ T.forward tr t2
--     topCombined :: QTensor (actHidden : pshape)
--     topCombined = embl + embm + embr + embt1 + embt2
--     topEmb :: QTensor (emb : pshape)
--     topEmb = activation $ runConv actTop2 topCombined
--     opFreeze = T.forward actFreeze ()
--     opSplit = T.forward actSplit ()
--     opSpread = T.forward actSpread ()
--     opCombined = TT.stack @0 $ opFreeze TT.:. opSplit TT.:. opFreeze TT.:. opSpread TT.:. opSplit TT.:. opSplit TT.:. TT.HNil
--     opEmbeddings :: QTensor '[6, emb]
--     opEmbeddings = TT.cat @1 $ opTypes TT.:. opCombined TT.:. TT.HNil
--     opEmb :: QTensor '[emb]
--     opEmb = TT.squeezeDim @0 $ TT.gatherDim @0 (TT.expand @'[1, emb] False opIndex) opEmbeddings
--   forwardStoch a i = pure $ T.forward a i

-- | HasForward for actions (batched)
instance
  forall (spec :: TGeneralSpec) slcHidden trHidden actHidden outShape emb fpad opad pshape fifths octs batchSize
   . ( emb ~ EmbSize spec
     , fpad ~ FifthPadding spec
     , opad ~ OctavePadding spec
     , fifths ~ FifthSize spec
     , octs ~ OctaveSize spec
     , pshape ~ PShape spec
     , outShape ~ (batchSize : emb : pshape)
     , emb ~ (emb - 3) + 3
     , KnownNat (PSize spec)
     , KnownNat emb
     , KnownNat trHidden
     , KnownNat slcHidden
     , KnownNat actHidden
     , KnownNat batchSize
     , GSpecConstraints spec
     , 1 <= batchSize
     )
  => T.HasForward
      (ActionEncoder spec actHidden)
      (SliceEncoder spec slcHidden, TransitionEncoder spec trHidden, ActionEncoding '[batchSize] spec)
      (QTensor outShape)
  where
  forward ActionEncoder{..} (slc, tr, ActionEncoding (ActionTop sl t1 (QMaybe smMask sm) (QMaybe t2Mask t2) sr) opIndex) = topEmb `TT.add` TT.reshape @[batchSize, emb, 1, 1] opEmb
   where
    runConv
      :: (KnownNat nin, KnownNat nout)
      => TT.Conv2d nin nout fifths octs QDType QDevice
      -> QTensor (batchSize : nin : pshape)
      -> QTensor (batchSize : nout : pshape)
    runConv conv input =
      activation $ TT.conv2dForward @'(1, 1) @'(fpad, opad) conv input
    runConvMasked
      :: (KnownNat nin, KnownNat nout)
      => QTensor '[batchSize]
      -> TT.Conv2d nin nout fifths octs QDType QDevice
      -> QTensor (batchSize : nin : pshape)
      -> QTensor (batchSize : nout : pshape)
    runConvMasked mask conv input =
      TT.mul (TT.reshape @[batchSize, 1, 1, 1] mask) $ runConv conv input
    -- top embedding
    embl :: QTensor (batchSize : actHidden : pshape)
    embl = runConv actTop1sl $ T.forward slc sl
    embm = runConv actTop1sm $ T.forward slc sm
    embr = runConvMasked smMask actTop1sr $ T.forward slc sr
    embt1 = runConvMasked t2Mask actTop1t1 $ T.forward tr t1
    embt2 = runConv actTop1t2 $ T.forward tr t2
    topCombined :: QTensor (batchSize : actHidden : pshape)
    topCombined = embl + embm + embr + embt1 + embt2
    topEmb :: QTensor (batchSize : emb : pshape)
    topEmb = runConv actTop2 topCombined
    -- operation embedding
    opFreeze = T.forward actFreeze ()
    opSplit = T.forward actSplit ()
    opSpread = T.forward actSpread ()
    opCombined = TT.stack @0 $ opFreeze TT.:. opSplit TT.:. opFreeze TT.:. opSpread TT.:. opSplit TT.:. opSplit TT.:. TT.HNil
    opEmbeddings :: QTensor '[6, emb]
    opEmbeddings = TT.cat @1 $ opTypes TT.:. opCombined TT.:. TT.HNil
    opIndex' :: TT.Tensor QDevice TT.Int64 [batchSize, emb]
    opIndex' = TT.expand @'[batchSize, emb] False $ TT.unsqueeze @1 opIndex
    opEmb :: QTensor '[batchSize, emb]
    opEmb = TT.gatherDim @0 opIndex' opEmbeddings
  forwardStoch a i = pure $ T.forward a i

-- -- | HasForward for actions (batched)
-- instance
--   forall (spec :: TGeneralSpec) slcHidden trHidden actHidden outShape emb batchSize
--    . ( emb ~ EmbSize spec
--      , -- , outShape ~ '[batchSize, emb]
--        emb ~ (emb - 3) + 3
--      , KnownNat (PSize spec)
--      , KnownNat emb
--      , KnownNat trHidden
--      , KnownNat batchSize
--      , GSpecConstraints spec
--      , 1 <= batchSize
--      )
--   => T.HasForward
--       (ActionEncoder spec actHidden)
--       (SliceEncoder spec slcHidden, TransitionEncoder spec trHidden, ActionEncoding '[batchSize] spec)
--       (QTensor '[batchSize, emb])
--   where
--   forward ActionEncoder{..} (slc, tr, ActionEncoding top opIndex) = topEmb + opEmb
--    where
--     topCombined :: QTensor '[batchSize, actHidden]
--     topCombined = case top of
--       (ActionTop sl t1 (QMaybe smMask sm) (QMaybe t2Mask t2) sr) ->
--         let
--           embl = undefined -- activation $ T.forward actTop1sl $ T.forward slc sl
--           embm = undefined -- activation $ T.forward actTop1sm $ T.forward slc sm
--           embr = undefined -- TT.mul (TT.unsqueeze @1 smMask) $ activation $ T.forward actTop1sr $ T.forward slc sr
--           embt1 = undefined -- TT.mul (TT.unsqueeze @1 t2Mask) $ activation $ T.forward actTop1t1 $ T.forward tr t1
--           embt2 = undefined -- activation $ T.forward actTop1t2 $ T.forward tr t2
--          in
--           embl + embm + embr + embt1 + embt2
--     topEmb :: QTensor '[batchSize, emb]
--     topEmb = activation $ T.forward actTop2 topCombined
--     opFreeze = T.forward actFreeze ()
--     opSplit = T.forward actSplit ()
--     opSpread = T.forward actSpread ()
--     opCombined = TT.stack @0 $ opFreeze TT.:. opSplit TT.:. opFreeze TT.:. opSpread TT.:. opSplit TT.:. opSplit TT.:. TT.HNil
--     opEmbeddings :: QTensor '[6, emb]
--     opEmbeddings = TT.cat @1 $ opTypes TT.:. opCombined TT.:. TT.HNil
--     opIndex' :: TT.Tensor QDevice TT.Int64 [batchSize, emb]
--     opIndex' = TT.expand @'[batchSize, emb] False $ TT.unsqueeze @1 opIndex
--     opEmb :: QTensor '[batchSize, emb]
--     opEmb = TT.gatherDim @0 opIndex' opEmbeddings
--   forwardStoch a i = pure $ T.forward a i

-- State Encoder
-- -------------

data StateSpec (hidden :: Nat) = StateSpec

data StateEncoder spec hidden = StateEncoder
  { stL1mid :: TT.Conv2d (EmbSize spec) hidden (FifthSize spec) (OctaveSize spec) QDType QDevice
  , stL1frozenSlc :: TT.Conv2d (EmbSize spec) hidden (FifthSize spec) (OctaveSize spec) QDType QDevice
  , stL1frozenTr :: TT.Conv2d (EmbSize spec) hidden (FifthSize spec) (OctaveSize spec) QDType QDevice
  , stL1openSlc1 :: TT.Conv2d (EmbSize spec) hidden (FifthSize spec) (OctaveSize spec) QDType QDevice
  , stL1openSlc2 :: TT.Conv2d (EmbSize spec) hidden (FifthSize spec) (OctaveSize spec) QDType QDevice
  , stL1openSlc3 :: TT.Conv2d (EmbSize spec) hidden (FifthSize spec) (OctaveSize spec) QDType QDevice
  , stL1openTr1 :: TT.Conv2d (EmbSize spec) hidden (FifthSize spec) (OctaveSize spec) QDType QDevice
  , stL1openTr2 :: TT.Conv2d (EmbSize spec) hidden (FifthSize spec) (OctaveSize spec) QDType QDevice
  , stL1openTr3 :: TT.Conv2d (EmbSize spec) hidden (FifthSize spec) (OctaveSize spec) QDType QDevice
  , stL2 :: TT.Conv2d hidden hidden (FifthSize spec) (OctaveSize spec) QDType QDevice
  , stL3 :: TT.Conv2d hidden (EmbSize spec) (FifthSize spec) (OctaveSize spec) QDType QDevice
  }
  deriving (Show, Generic, TT.Parameterized, NoThunks, NFData)

instance
  ( KnownNat (EmbSize spec)
  , KnownNat hidden
  , KnownNat (FifthPadding spec)
  , KnownNat (OctavePadding spec)
  )
  => T.Randomizable (GeneralSpec spec, StateSpec hidden) (StateEncoder spec hidden)
  where
  sample _ = do
    stL1mid <- TT.sample TT.Conv2dSpec
    stL1frozenSlc <- TT.sample TT.Conv2dSpec
    stL1frozenTr <- TT.sample TT.Conv2dSpec
    stL1openSlc1 <- TT.sample TT.Conv2dSpec
    stL1openSlc2 <- TT.sample TT.Conv2dSpec
    stL1openSlc3 <- TT.sample TT.Conv2dSpec
    stL1openTr1 <- TT.sample TT.Conv2dSpec
    stL1openTr2 <- TT.sample TT.Conv2dSpec
    stL1openTr3 <- TT.sample TT.Conv2dSpec
    stL2 <- TT.sample TT.Conv2dSpec
    stL3 <- TT.sample TT.Conv2dSpec
    pure StateEncoder{..}

-- | HasForward for the parsing state (doesn't need batching)
instance
  forall (spec :: TGeneralSpec) slcHidden trHidden stHidden outShape emb pshape fpad opad fifths octs
   . ( emb ~ EmbSize spec
     , fpad ~ FifthPadding spec
     , opad ~ OctavePadding spec
     , fifths ~ FifthSize spec
     , octs ~ OctaveSize spec
     , pshape ~ PShape spec
     , outShape ~ (emb : pshape)
     , KnownNat slcHidden
     , KnownNat trHidden
     , KnownNat stHidden
     , GSpecConstraints spec
     )
  => T.HasForward
      (StateEncoder spec stHidden)
      (SliceEncoder spec slcHidden, TransitionEncoder spec trHidden, StateEncoding spec)
      (QTensor outShape)
  where
  forward StateEncoder{..} (slc, tr, StateEncoding mid frozen open) = out3
   where
    runConv
      :: (KnownNat nin, KnownNat nout)
      => TT.Conv2d nin nout fifths octs QDType QDevice
      -> QTensor (nin : pshape)
      -> QTensor (nout : pshape)
    runConv conv input =
      TT.squeezeDim @0 $
        activation $
          TT.conv2dForward @'(1, 1) @'(fpad, opad) conv $
            TT.unsqueeze @0 input
    -- embed the mid slice
    midEmb :: QTensor (stHidden : pshape)
    midEmb = activation $ runConv stL1mid $ T.forward slc mid
    -- embed the frozen segment (if it exists) and add to midEmb
    midAndFrozen :: QTensor (stHidden : pshape)
    midAndFrozen = case frozen of
      Nothing -> midEmb
      Just (ft, fs) ->
        let ftEmb = activation $ runConv stL1frozenTr $ T.forward tr ft
            fsEmb = activation $ runConv stL1frozenSlc $ T.forward slc fs
         in midEmb + ftEmb + fsEmb
    -- embed an open segment using its respective layers
    embedOpen (ot, os) (l1tr, l1slc) = otEmb + osEmb
     where
      otEmb :: QTensor (stHidden : pshape)
      otEmb = activation $ runConv l1tr $ T.forward tr ot
      osEmb :: QTensor (stHidden : pshape)
      osEmb = activation $ runConv l1slc $ T.forward slc os
    -- the list of layers for the 3 open transitions and slices
    openEncoders =
      [ (stL1openSlc1, stL1openTr1)
      , (stL1openSlc2, stL1openTr2)
      , (stL1openSlc3, stL1openTr3)
      ]
    -- embed the open segments and add them to mid and frozen
    fullEmb :: QTensor (stHidden : pshape)
    fullEmb = F.foldl' (+) midAndFrozen $ zipWith embedOpen open openEncoders
    out2 :: QTensor (stHidden : pshape)
    out2 = activation $ runConv stL2 fullEmb
    out3 :: QTensor (emb : pshape)
    out3 = activation $ runConv stL3 out2
  forwardStoch a i = pure $ T.forward a i

-- Full Q Model
-- ------------

data SpecialSpec (hidden :: Nat) = SpecialSpec

data QSpec (spec :: TQSpec) = QSpec

data QModel spec = QModel
  { qModelSlc :: !(SliceEncoder (QSpecGeneral spec) (QSpecSlice spec))
  , qModelTr :: !(TransitionEncoder (QSpecGeneral spec) (QSpecTrans spec))
  , qModelAct :: !(ActionEncoder (QSpecGeneral spec) (QSpecAction spec))
  , qModelSt :: !(StateEncoder (QSpecGeneral spec) (QSpecState spec))
  , qModelFinal1 :: !(TT.Conv2d (EmbSize (QSpecGeneral spec)) (QSpecSpecial spec) (FifthSize (QSpecGeneral spec)) (OctaveSize (QSpecGeneral spec)) QDType QDevice) -- !(TT.Linear (EmbSize (QSpecGeneral spec)) (QSpecSpecial spec) QDType QDevice)
  , qModelNorm1 :: !(TT.LayerNorm '[QSpecSpecial spec] QDType QDevice)
  , qModelFinal2 :: !(TT.Linear (QSpecSpecial spec) 1 QDType QDevice)
  , qModelValue1 :: !(TT.Linear (EmbSize (QSpecGeneral spec)) (QSpecSpecial spec) QDType QDevice)
  , qModelValueNorm :: !(TT.LayerNorm '[QSpecSpecial spec] QDType QDevice)
  , qModelValue2 :: !(TT.Linear (QSpecSpecial spec) 1 QDType QDevice)
  }
  deriving (Show, Generic, TT.Parameterized, NoThunks, NFData)

instance
  ( spec ~ TQSpecData gspec sp sl tr ac st
  , embsize ~ EmbSize gspec
  , KnownNat sp
  , KnownNat sl
  , KnownNat tr
  , KnownNat ac
  , KnownNat st
  , GSpecConstraints gspec
  )
  => T.Randomizable (QSpec spec) (QModel spec)
  where
  sample :: QSpec spec -> IO (QModel spec)
  sample QSpec = do
    qModelSlc <- T.sample (GeneralSpec, SliceSpec)
    qModelTr <- T.sample (GeneralSpec, TransitionSpec)
    qModelAct <- T.sample (GeneralSpec, ActionSpec)
    qModelSt <- T.sample (GeneralSpec, StateSpec)
    qModelFinal1 <- T.sample TT.Conv2dSpec
    qModelNorm1 <- T.sample $ TT.LayerNormSpec 1e-05
    qModelFinal2 <- T.sample TT.LinearSpec
    qModelValue1 <- T.sample TT.LinearSpec
    qModelValueNorm <- T.sample $ TT.LayerNormSpec 1e-05
    qModelValue2 <- T.sample TT.LinearSpec
    pure QModel{..}

-- | HasForward for model (unbatched)
instance
  ( gspec ~ QSpecGeneral spec
  , hidden ~ QSpecSpecial spec
  , ((EmbSize gspec - 3) + 3) ~ EmbSize gspec
  , KnownNat hidden
  , KnownNat (QSpecTrans spec)
  , KnownNat (QSpecSlice spec)
  , KnownNat (QSpecAction spec)
  , KnownNat (QSpecState spec)
  , TT.CheckIsSuffixOf '[hidden] [1, hidden] (hidden == hidden)
  , GSpecConstraints gspec
  )
  => T.HasForward (QModel spec) (QEncoding '[] gspec) (QTensor '[1])
  where
  forward :: QModel spec -> QEncoding '[] (QSpecGeneral spec) -> QTensor '[1]
  forward model@(QModel slc tr act st final1 norm1 final2 _ _ _) encoding@(QEncoding actEnc stEnc) =
    TT.log $ TT.sigmoid $ forwardQModel model encoding

  forwardStoch :: QModel spec -> QEncoding '[] (QSpecGeneral spec) -> IO (QTensor '[1])
  forwardStoch model input = pure $ T.forward model input

-- | HasForward for model (batched)
instance
  ( gspec ~ QSpecGeneral spec
  , ((EmbSize gspec - 3) + 3) ~ EmbSize gspec
  , KnownNat (QSpecSpecial spec)
  , KnownNat (QSpecTrans spec)
  , KnownNat (QSpecSlice spec)
  , KnownNat (QSpecAction spec)
  , KnownNat (QSpecState spec)
  , KnownNat batchSize
  , 1 <= batchSize
  , TT.CheckIsSuffixOf '[QSpecSpecial spec] [batchSize, QSpecSpecial spec] (QSpecSpecial spec == QSpecSpecial spec)
  , GSpecConstraints gspec
  )
  => T.HasForward (QModel spec) (QEncoding '[batchSize] gspec) (QTensor '[batchSize, 1])
  where
  forward :: QModel spec -> QEncoding '[batchSize] (QSpecGeneral spec) -> QTensor '[batchSize, 1]
  forward model encoding =
    TT.log $ TT.sigmoid $ forwardQModelBatched model encoding

  forwardStoch model input = pure $ T.forward model input

forwardQModel
  :: forall spec gspec emb pshape hidden fpad opad
   . ( gspec ~ QSpecGeneral spec
     , emb ~ EmbSize gspec
     , pshape ~ PShape gspec
     , hidden ~ QSpecSpecial spec
     , fpad ~ FifthPadding gspec
     , opad ~ OctavePadding gspec
     , ((emb - 3) + 3) ~ emb
     , KnownNat (QSpecSpecial spec)
     , KnownNat (QSpecTrans spec)
     , -- , TT.CheckIsSuffixOf '[hidden] '[hidden] (hidden == hidden)
       GSpecConstraints gspec
     , KnownNat (QSpecSlice spec)
     , KnownNat (QSpecAction spec)
     , KnownNat (QSpecState spec)
     , TT.CheckIsSuffixOf '[hidden] [1, hidden] (hidden == hidden)
     )
  => QModel spec
  -> QEncoding '[] (QSpecGeneral spec)
  -> QTensor '[1]
forwardQModel model input = TT.squeezeDim @0 $ forwardQModelBatched model $ addBatchDim input

-- forwardQModel (QModel slc tr act st final1 norm1 final2 _ _ _) (QEncoding actEnc stEnc) =
--   T.forward final2 $
--     activation $
--       T.forward norm1 sum1
--  where
--   actEmb :: QTensor (emb : pshape)
--   actEmb = T.forward act (slc, tr, actEnc)
--   stEmb :: QTensor (emb : pshape)
--   stEmb = T.forward st (slc, tr, stEnc)
--   out1 :: QTensor (hidden : pshape)
--   out1 = withBatchDim (TT.conv2dForward @'(1, 1) @'(fpad, opad) final1) (actEmb + stEmb)
--   sum1 :: QTensor '[hidden]
--   sum1 = TT.sumDim @1 $ TT.sumDim @1 out1

forwardQModelBatched
  :: forall gspec spec batchSize emb hidden pshape fpad opad
   . ( gspec ~ QSpecGeneral spec
     , emb ~ EmbSize gspec
     , pshape ~ PShape gspec
     , fpad ~ FifthPadding gspec
     , opad ~ OctavePadding gspec
     , hidden ~ QSpecSpecial spec
     , ((emb - 3) + 3) ~ emb
     , KnownNat (QSpecSpecial spec)
     , KnownNat (QSpecTrans spec)
     , KnownNat batchSize
     , 1 <= batchSize
     , TT.CheckIsSuffixOf '[hidden] [batchSize, hidden] (hidden == hidden)
     , GSpecConstraints gspec
     , KnownNat (QSpecAction spec)
     , KnownNat (QSpecState spec)
     , KnownNat (QSpecSlice spec)
     )
  => QModel spec
  -> QEncoding '[batchSize] (QSpecGeneral spec)
  -> QTensor '[batchSize, 1]
forwardQModelBatched (QModel slc tr act st final1 norm1 final2 _ _ _) (QEncoding actEncs stEnc) = out2
 where
  actEmb :: QTensor (batchSize : emb : pshape)
  actEmb = T.forward act (slc, tr, actEncs)
  stEmb :: QTensor (emb : pshape)
  stEmb = T.forward st (slc, tr, stEnc)
  inputEmb = actEmb `TT.add` stEmb
  out1 :: QTensor (batchSize : hidden : pshape)
  out1 = TT.conv2dForward @'(1, 1) @'(fpad, opad) final1 inputEmb
  sum1 :: QTensor '[batchSize, hidden]
  sum1 = TT.sumDim @2 $ TT.sumDim @2 out1
  out1norm :: QTensor '[batchSize, hidden]
  out1norm = activation $ T.forward norm1 sum1
  out2 :: QTensor '[batchSize, 1]
  out2 = T.forward final2 out1norm

forwardPolicy
  :: (_)
  => QModel spec
  -> QEncoding '[] (QSpecGeneral spec)
  -> QTensor '[1]
forwardPolicy = forwardQModel

forwardPolicyBatched
  :: (_)
  => QModel spec
  -> QEncoding '[batchSize] (QSpecGeneral spec)
  -> QTensor '[batchSize, 1]
forwardPolicyBatched = forwardQModelBatched

forwardValue
  :: forall spec gspec
   . ( gspec ~ QSpecGeneral spec
     , ((EmbSize gspec - 3) + 3) ~ EmbSize gspec
     , KnownNat (QSpecSpecial spec)
     , KnownNat (QSpecTrans spec)
     , TT.CheckIsSuffixOf '[QSpecSpecial spec] '[QSpecSpecial spec] (QSpecSpecial spec == QSpecSpecial spec)
     , GSpecConstraints gspec
     , KnownNat (QSpecSlice spec)
     , KnownNat (QSpecState spec)
     )
  => QModel spec
  -> StateEncoding (QSpecGeneral spec)
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
   . (ps ~ TT.Parameters (QModel DefaultQSpec))
  => QModel DefaultQSpec
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

mkQModel :: IO (QModel DefaultQSpec)
mkQModel = T.sample QSpec

loadModel :: FilePath -> IO (QModel DefaultQSpec)
loadModel path = do
  modelPlaceholder <- mkQModel
  tensors
    :: (TT.HMap' TT.ToDependent (TT.Parameters (QModel DefaultQSpec)) ts)
    => TT.HList ts <-
    TT.load path
  -- TT.load doesn't move the parameters to the correct device, so we move them manually
  let tensorsCPU = TT.toDevice @'(TT.CPU, 0) @QDevice tensors
  let tensorsDevice = TT.toDevice @QDevice @'(TT.CPU, 0) tensorsCPU
  params <- TT.hmapM' TT.MakeIndependent tensorsDevice
  pure $ TT.replaceParameters modelPlaceholder params

modelSize :: QModel DefaultQSpec -> Int
modelSize model = sum $ product <$> sizes
 where
  sizes = TT.hfoldr TH.ToList ([] :: [[Int]]) $ TT.hmap' TH.ShapeVal $ TT.flattenParameters model

runQ
  :: (s -> a -> QEncoding '[] (QSpecGeneral DefaultQSpec))
  -> QModel DefaultQSpec
  -> s
  -> a
  -> QType
runQ !encode !model s a = T.asValue $ TT.toDynamic $ T.forward model $ encode s a

runQ'
  :: (s -> a -> QEncoding '[] (QSpecGeneral DefaultQSpec))
  -> QModel DefaultQSpec
  -> s
  -> a
  -> QTensor '[1]
runQ' !encode !model s a = T.forward model $ encode s a

runBatchedPolicy
  :: forall batchSize
   . (KnownNat batchSize)
  => QModel DefaultQSpec
  -> QEncoding '[batchSize] (QSpecGeneral DefaultQSpec)
  -> T.Tensor
runBatchedPolicy actor encoding = TT.toDynamic $ TT.softmax @0 $ policy
 where
  policy :: QTensor '[batchSize, 1]
  policy = case cmpNat (Proxy @1) (Proxy @batchSize) of
    EQI -> forwardPolicyBatched @DefaultQSpec @batchSize actor encoding
    LTI -> forwardPolicyBatched @DefaultQSpec @batchSize actor encoding
    GTI -> error "batched policy: no actions"
