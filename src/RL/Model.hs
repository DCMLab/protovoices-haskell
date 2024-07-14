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

module RL.Model where

import Common
import Control.Arrow ((>>>))
import Control.DeepSeq
import Data.Foldable qualified as F
import Data.Function ((&))
import Data.Kind (Type)
import Data.Proxy (Proxy (Proxy))
import Data.Type.Equality (type (==))
import Data.TypeNums (KnownInt, KnownNat, Nat, TInt (..), intVal, intVal', type (*), type (+), type (-), type (<=))
import Debug.Trace qualified as DT
import GHC.ForeignPtr qualified as Ptr
import GHC.Generics (Generic)
import GHC.TypeLits (OrderingI (..), cmpNat)
import GreedyParser (DoubleParent (DoubleParent), SingleParent (SingleParent))
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
  { _slcL1 :: !(TT.Linear (PSize spec) hidden QDType QDevice)
  , _slcL2 :: !(TT.Linear hidden (GenEmbSize spec) QDType QDevice)
  , _slcStart :: !(ConstEmb '[GenEmbSize spec])
  , _slcStop :: !(ConstEmb '[GenEmbSize spec])
  -- TODO: learn embedding for empty slice
  }
  deriving (Show, Generic, TT.Parameterized, NoThunks, NFData)

instance
  ( KnownNat hidden
  , KnownNat (GenEmbSize spec)
  , KnownNat (PSize spec)
  )
  => T.Randomizable (GeneralSpec spec, SliceSpec hidden) (SliceEncoder spec hidden)
  where
  sample :: (GeneralSpec spec, SliceSpec hidden) -> IO (SliceEncoder spec hidden)
  sample _ =
    SliceEncoder
      <$> T.sample TT.LinearSpec
      <*> T.sample TT.LinearSpec
      <*> T.sample ConstEmbSpec
      <*> T.sample ConstEmbSpec

{- | HasForward for slice wrapped in StartStop.
Could be removed if StateEncoding is changed to QStartStop.
-}
instance
  ( embshape ~ '[GenEmbSize spec]
  , psize ~ PSize spec
  , KnownNat (GenEmbSize spec)
  )
  => TT.HasForward (SliceEncoder spec hidden) (StartStop (QBoundedList MaxPitches '[] '[psize])) (QTensor embshape)
  where
  forward model@(SliceEncoder _ _ start stop) input =
    case input of
      Inner slc -> T.forward model slc
      Start -> T.forward start ()
      Stop -> T.forward stop ()
  forwardStoch model input = pure $ T.forward model input

-- | HasForward for slice wrappend in QStartStop (unbatched).
instance
  ( embshape ~ '[GenEmbSize spec]
  , psize ~ PSize spec
  , KnownNat (GenEmbSize spec)
  )
  => TT.HasForward (SliceEncoder spec hidden) (QStartStop '[] (QBoundedList MaxPitches '[] '[psize])) (QTensor embshape)
  where
  forward model@(SliceEncoder _ _ start stop) (QStartStop tag input) = TT.squeezeDim @0 out
   where
    -- compute the possible outputs for start/stop/inner
    outStart = TT.forward start ()
    outStop = TT.forward stop ()
    outInner = T.forward model input
    -- combine the outputs into one tensor
    combined = TT.stack @0 $ outStart TT.:. outInner TT.:. outStop TT.:. TT.HNil
    -- use gather to select the right output.
    -- gather can select different elements from 'dim' for each position,
    -- so we expand the tag to the right shape, selecting the *same* 'dim'-index everywhere
    out = TT.gatherDim @0 (TT.expand @'[1, GenEmbSize spec] False tag) combined
  forwardStoch model input = pure $ T.forward model input

-- | HasForward for slice wrappend in QStartStop (batched).
instance
  ( emb ~ GenEmbSize spec
  , psize ~ PSize spec
  , embshape ~ '[batchSize, emb]
  , KnownNat emb
  , KnownNat batchSize
  )
  => TT.HasForward (SliceEncoder spec hidden) (QStartStop '[batchSize] (QBoundedList MaxPitches '[batchSize] '[psize])) (QTensor embshape)
  where
  forward model@(SliceEncoder _ _ start stop) (QStartStop tag input) = TT.squeezeDim @0 out
   where
    -- compute the possible outputs for start/stop/inner
    outStart :: QTensor [batchSize, emb]
    outStart = TT.expand @'[batchSize, emb] False $ TT.forward start ()
    outStop :: QTensor [batchSize, emb]
    outStop = TT.expand @'[batchSize, emb] False $ TT.forward stop ()
    outInner :: QTensor [batchSize, emb]
    outInner = T.forward model input
    -- combine the outputs into one tensor
    combined :: QTensor [3, batchSize, emb]
    combined = TT.stack @0 $ outStart TT.:. outInner TT.:. outStop TT.:. TT.HNil
    -- use gather to select the right output.
    -- gather can select different elements from 'dim' for each position,
    -- so we expand the tag to the right shape, selecting the *same* 'dim'-index everywhere
    tag' :: TT.Tensor QDevice 'TT.Int64 [1, batchSize, emb]
    tag' = TT.expand @'[1, batchSize, emb] False $ TT.unsqueeze @1 tag
    out = TT.gatherDim @0 tag' combined
  forwardStoch model input = pure $ T.forward model input

-- | HasFoward for slice (unbatched)
instance
  ( embshape ~ '[GenEmbSize spec]
  , psize ~ PSize spec
  )
  => T.HasForward (SliceEncoder spec hidden) (QBoundedList MaxPitches '[] '[psize]) (QTensor embshape)
  where
  forward (SliceEncoder l1 l2 _ _) (QBoundedList mask input) = out
   where
    out1 = activation $ T.forward l1 input
    out2 = activation $ T.forward l2 out1
    outMasked = TT.mul (TT.unsqueeze @1 mask) out2
    out = TT.sumDim @0 outMasked
  forwardStoch model = pure . T.forward model

-- | HasFoward for slice (batched)
instance
  ( psize ~ PSize spec
  , emb ~ GenEmbSize spec
  , embshape ~ '[batchSize, emb]
  )
  => T.HasForward (SliceEncoder spec hidden) (QBoundedList MaxPitches '[batchSize] '[psize]) (QTensor embshape)
  where
  forward (SliceEncoder l1 l2 _ _) (QBoundedList mask input) = out
   where
    out1 :: QTensor '[batchSize, MaxPitches, hidden]
    out1 = activation $ T.forward l1 input
    out2 :: QTensor '[batchSize, MaxPitches, emb]
    out2 = activation $ T.forward l2 out1
    mask' :: QTensor '[batchSize, MaxPitches, 1]
    mask' = TT.unsqueeze @2 mask
    outMasked :: QTensor '[batchSize, MaxPitches, emb]
    outMasked = TT.mul mask' out2
    out :: QTensor '[batchSize, emb]
    out = TT.sumDim @1 outMasked
  forwardStoch model = pure . T.forward model

-- Transition Encoder
-- ------------------

data TransitionSpec (hidden :: Nat) = TransitionSpec

data TransitionEncoder spec hidden = TransitionEncoder
  { trL1Passing :: !(TT.Linear (ESize spec) hidden QDType QDevice)
  , trL1Inner :: !(TT.Linear (ESize spec) hidden QDType QDevice)
  , trL1Left :: !(TT.Linear (PSize spec) hidden QDType QDevice)
  , trL1Right :: !(TT.Linear (PSize spec) hidden QDType QDevice)
  , trL1Root :: !(ConstEmb '[hidden])
  , trL2 :: !(TT.Linear hidden (GenEmbSize spec) QDType QDevice)
  }
  deriving (Show, Generic, TT.Parameterized, NoThunks, NFData)

instance
  ( KnownNat hidden
  , KnownNat (GenEmbSize spec)
  , KnownNat (PSize spec)
  , KnownNat (ESize spec)
  )
  => T.Randomizable (GeneralSpec spec, TransitionSpec hidden) (TransitionEncoder spec hidden)
  where
  sample :: (GeneralSpec spec, TransitionSpec hidden) -> IO (TransitionEncoder spec hidden)
  sample _ = do
    trL1Passing <- T.sample TT.LinearSpec
    trL1Inner <- T.sample TT.LinearSpec
    trL1Left <- T.sample TT.LinearSpec
    trL1Right <- T.sample TT.LinearSpec
    trL1Root <- T.sample ConstEmbSpec
    trL2 <- T.sample TT.LinearSpec
    pure $ TransitionEncoder{..}

-- | HasForward for transitions (unbatched)
instance
  forall spec hidden embshape
   . ( embshape ~ '[GenEmbSize spec]
     , PShape spec ~ '[FakeSize, PSize spec]
     , EShape spec ~ '[FakeSize, ESize spec]
     , TT.KnownShape (EShape spec)
     , TT.KnownShape (PShape spec)
     , KnownNat hidden
     )
  => T.HasForward
      (TransitionEncoder spec hidden)
      (TransitionEncoding '[] spec)
      (QTensor embshape)
  where
  forward TransitionEncoder{..} TransitionEncoding{..} = activation $ T.forward trL2 all
   where
    pass :: QTensor '[hidden]
    pass =
      let QBoundedList mask edgesPassing = trencPassing
       in TT.sumDim @0 $ TT.mul (TT.unsqueeze @1 mask) $ activation $ T.forward trL1Passing edgesPassing
    inner :: QTensor '[hidden]
    inner =
      let QBoundedList mask edgesInner = trencInner
       in TT.sumDim @0 $ TT.mul (TT.unsqueeze @1 mask) $ activation $ T.forward trL1Inner edgesInner
    left :: QTensor '[hidden]
    left =
      let QBoundedList mask notes = trencLeft
       in TT.sumDim @0 $ TT.mul (TT.unsqueeze @1 mask) $ activation $ T.forward trL1Left notes
    right :: QTensor '[hidden]
    right =
      let QBoundedList mask notes = trencRight
       in TT.sumDim @0 $ TT.mul (TT.unsqueeze @1 mask) $ activation $ T.forward trL1Right notes
    root = TT.mul trencRoot (activation (T.forward trL1Root ()))
    all = pass + inner + left + right + root
  forwardStoch tr input = pure $ T.forward tr input

-- | HasForward for transitions (batched)
instance
  forall spec hidden emb embshape batchSize
   . ( emb ~ GenEmbSize spec
     , embshape ~ '[batchSize, emb]
     , PShape spec ~ '[FakeSize, PSize spec]
     , EShape spec ~ '[FakeSize, ESize spec]
     , TT.KnownShape (EShape spec)
     , TT.KnownShape (PShape spec)
     , KnownNat hidden
     , KnownNat batchSize
     )
  => T.HasForward
      (TransitionEncoder spec hidden)
      (TransitionEncoding '[batchSize] spec)
      (QTensor embshape)
  where
  forward TransitionEncoder{..} TransitionEncoding{..} = activation $ T.forward trL2 all
   where
    pass :: QTensor '[batchSize, hidden]
    pass =
      let QBoundedList mask edgesPassing = trencPassing
       in TT.sumDim @1 $ TT.mul (TT.unsqueeze @2 mask) $ activation $ T.forward trL1Passing edgesPassing
    inner :: QTensor '[batchSize, hidden]
    inner =
      let QBoundedList mask edgesInner = trencInner
       in TT.sumDim @1 $ TT.mul (TT.unsqueeze @2 mask) $ activation $ T.forward trL1Inner edgesInner
    left :: QTensor '[batchSize, hidden]
    left =
      let QBoundedList mask notes = trencLeft
       in TT.sumDim @1 $ TT.mul (TT.unsqueeze @2 mask) $ activation $ T.forward trL1Left notes
    right :: QTensor '[batchSize, hidden]
    right =
      let QBoundedList mask notes = trencRight
       in TT.sumDim @1 $ TT.mul (TT.unsqueeze @2 mask) $ activation $ T.forward trL1Right notes
    root :: QTensor '[batchSize, hidden]
    root =
      TT.mul (TT.unsqueeze @1 trencRoot) $
        TT.expand @'[batchSize, hidden] False (activation (T.forward trL1Root ()))
    all :: QTensor '[batchSize, hidden]
    all = pass + inner + left + right + root
  forwardStoch tr input = pure $ T.forward tr input

-- ActionEncoder
-- -------------

data ActionSpec (hidden :: Nat) = ActionSpec

data ActionEncoder spec hidden = ActionEncoder
  { actTop1sl :: TT.Linear (GenEmbSize spec) hidden QDType QDevice
  , actTop1sm :: TT.Linear (GenEmbSize spec) hidden QDType QDevice
  , actTop1sr :: TT.Linear (GenEmbSize spec) hidden QDType QDevice
  , actTop1t1 :: TT.Linear (GenEmbSize spec) hidden QDType QDevice
  , actTop1t2 :: TT.Linear (GenEmbSize spec) hidden QDType QDevice
  , actTop2 :: TT.Linear hidden (GenEmbSize spec) QDType QDevice
  , actSplit :: ConstEmb '[GenEmbSize spec - 3] -- TODO: fill in with actual module
  , actSpread :: ConstEmb '[GenEmbSize spec - 3] -- TODO: fill in with actual module
  , actFreeze :: ConstEmb '[GenEmbSize spec - 3]
  }
  deriving (Show, Generic, TT.Parameterized, NoThunks, NFData)

instance
  ( emb ~ GenEmbSize spec
  , KnownNat hidden
  , KnownNat emb
  , KnownNat (emb - 3)
  )
  => T.Randomizable (GeneralSpec spec, ActionSpec hidden) (ActionEncoder spec hidden)
  where
  sample :: (GeneralSpec spec, ActionSpec hidden) -> IO (ActionEncoder spec hidden)
  sample (GeneralSpec, ActionSpec) = do
    actTop1sl <- T.sample TT.LinearSpec
    actTop1sm <- T.sample TT.LinearSpec
    actTop1sr <- T.sample TT.LinearSpec
    actTop1t1 <- T.sample TT.LinearSpec
    actTop1t2 <- T.sample TT.LinearSpec
    actTop2 <- T.sample TT.LinearSpec
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

-- | HasForward for actions (unbatched)
instance
  forall (spec :: TGeneralSpec) slcHidden trHidden actHidden outShape emb
   . ( emb ~ GenEmbSize spec
     , outShape ~ '[emb]
     , emb ~ (emb - 3) + 3
     , PShape spec ~ '[FakeSize, PSize spec]
     , TT.KnownShape (PShape spec)
     , TT.KnownShape (EShape spec)
     , KnownNat (PSize spec)
     , KnownNat emb
     , KnownNat trHidden
     )
  => T.HasForward
      (ActionEncoder spec actHidden)
      (SliceEncoder spec slcHidden, TransitionEncoder spec trHidden, ActionEncoding '[] spec)
      (QTensor outShape)
  where
  forward ActionEncoder{..} (slc, tr, ActionEncoding top opIndex) = topEmb + opEmb
   where
    topCombined :: QTensor '[actHidden]
    topCombined = case top of
      (ActionTop sl t1 (QMaybe smMask sm) (QMaybe t2Mask t2) sr) ->
        let
          embl = activation $ T.forward actTop1sl $ T.forward slc sl
          embm = activation $ T.forward actTop1sm $ T.forward slc sm
          embr = TT.mul smMask $ activation $ T.forward actTop1sr $ T.forward slc sr
          embt1 = TT.mul t2Mask $ activation $ T.forward actTop1t1 $ T.forward tr t1
          embt2 = activation $ T.forward actTop1t2 $ T.forward tr t2
         in
          embl + embm + embr + embt1 + embt2
    topEmb :: QTensor '[GenEmbSize spec]
    topEmb = activation $ T.forward actTop2 topCombined
    opFreeze = T.forward actFreeze ()
    opSplit = T.forward actSplit ()
    opSpread = T.forward actSpread ()
    opCombined = TT.stack @0 $ opFreeze TT.:. opSplit TT.:. opFreeze TT.:. opSpread TT.:. opSplit TT.:. opSplit TT.:. TT.HNil
    opEmbeddings :: QTensor '[6, emb]
    opEmbeddings = TT.cat @1 $ opTypes TT.:. opCombined TT.:. TT.HNil
    opEmb :: QTensor '[emb]
    opEmb = TT.squeezeDim @0 $ TT.gatherDim @0 (TT.expand @'[1, emb] False opIndex) opEmbeddings
  forwardStoch a i = pure $ T.forward a i

-- | HasForward for actions (batched)
instance
  forall (spec :: TGeneralSpec) slcHidden trHidden actHidden outShape emb batchSize
   . ( emb ~ GenEmbSize spec
     , -- , outShape ~ '[batchSize, emb]
       emb ~ (emb - 3) + 3
     , PShape spec ~ '[FakeSize, PSize spec]
     , TT.KnownShape (PShape spec)
     , TT.KnownShape (EShape spec)
     , KnownNat (PSize spec)
     , KnownNat emb
     , KnownNat trHidden
     , KnownNat batchSize
     , 1 <= batchSize
     )
  => T.HasForward
      (ActionEncoder spec actHidden)
      (SliceEncoder spec slcHidden, TransitionEncoder spec trHidden, ActionEncoding '[batchSize] spec)
      (QTensor '[batchSize, emb])
  where
  forward ActionEncoder{..} (slc, tr, ActionEncoding top opIndex) = topEmb + opEmb
   where
    topCombined :: QTensor '[batchSize, actHidden]
    topCombined = case top of
      (ActionTop sl t1 (QMaybe smMask sm) (QMaybe t2Mask t2) sr) ->
        let
          embl = activation $ T.forward actTop1sl $ T.forward slc sl
          embm = activation $ T.forward actTop1sm $ T.forward slc sm
          embr = TT.mul (TT.unsqueeze @1 smMask) $ activation $ T.forward actTop1sr $ T.forward slc sr
          embt1 = TT.mul (TT.unsqueeze @1 t2Mask) $ activation $ T.forward actTop1t1 $ T.forward tr t1
          embt2 = activation $ T.forward actTop1t2 $ T.forward tr t2
         in
          embl + embm + embr + embt1 + embt2
    topEmb :: QTensor '[batchSize, emb]
    topEmb = activation $ T.forward actTop2 topCombined
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

-- State Encoder
-- -------------

data StateSpec (hidden :: Nat) = StateSpec

data StateEncoder spec hidden = StateEncoder
  { stL1mid :: TT.Linear (GenEmbSize spec) hidden QDType QDevice
  , stL1frozenSlc :: TT.Linear (GenEmbSize spec) hidden QDType QDevice
  , stL1frozenTr :: TT.Linear (GenEmbSize spec) hidden QDType QDevice
  , stL1openSlc1 :: TT.Linear (GenEmbSize spec) hidden QDType QDevice
  , stL1openSlc2 :: TT.Linear (GenEmbSize spec) hidden QDType QDevice
  , stL1openSlc3 :: TT.Linear (GenEmbSize spec) hidden QDType QDevice
  , stL1openTr1 :: TT.Linear (GenEmbSize spec) hidden QDType QDevice
  , stL1openTr2 :: TT.Linear (GenEmbSize spec) hidden QDType QDevice
  , stL1openTr3 :: TT.Linear (GenEmbSize spec) hidden QDType QDevice
  , stL2 :: TT.Linear hidden hidden QDType QDevice
  , stL3 :: TT.Linear hidden (GenEmbSize spec) QDType QDevice
  }
  deriving (Show, Generic, TT.Parameterized, NoThunks, NFData)

instance
  ( KnownNat (GenEmbSize spec)
  , KnownNat hidden
  )
  => T.Randomizable (GeneralSpec spec, StateSpec hidden) (StateEncoder spec hidden)
  where
  sample _ = do
    stL1mid <- TT.sample TT.LinearSpec
    stL1frozenSlc <- TT.sample TT.LinearSpec
    stL1frozenTr <- TT.sample TT.LinearSpec
    stL1openSlc1 <- TT.sample TT.LinearSpec
    stL1openSlc2 <- TT.sample TT.LinearSpec
    stL1openSlc3 <- TT.sample TT.LinearSpec
    stL1openTr1 <- TT.sample TT.LinearSpec
    stL1openTr2 <- TT.sample TT.LinearSpec
    stL1openTr3 <- TT.sample TT.LinearSpec
    stL2 <- TT.sample TT.LinearSpec
    stL3 <- TT.sample TT.LinearSpec
    pure StateEncoder{..}

-- | HasForward for the parsing state (doesn't need batching)
instance
  forall (spec :: TGeneralSpec) slcHidden trHidden stHidden outShape emb
   . ( emb ~ GenEmbSize spec
     , outShape ~ '[emb]
     , PShape spec ~ [FakeSize, PSize spec]
     , TT.KnownShape (PShape spec)
     , TT.KnownShape (EShape spec)
     , KnownNat (PSize spec)
     , KnownNat trHidden
     , KnownNat emb
     )
  => T.HasForward
      (StateEncoder spec stHidden)
      (SliceEncoder spec slcHidden, TransitionEncoder spec trHidden, StateEncoding spec)
      (QTensor outShape)
  where
  forward StateEncoder{..} (slc, tr, StateEncoding mid frozen open) =
    fullEmb
      & T.forward stL2
      & activation
      & T.forward stL3
      & activation
   where
    -- embed the mid slice
    midEmb = activation $ T.forward stL1mid $ T.forward slc mid
    -- embed the frozen segment (if it exists) and add to midEmb
    midAndFrozen = case frozen of
      Nothing -> midEmb
      Just (ft, fs) ->
        let ftEmb = activation $ T.forward stL1frozenTr $ T.forward tr ft
            fsEmb = activation $ T.forward stL1frozenSlc $ T.forward slc fs
         in midEmb + ftEmb + fsEmb
    -- embed an open segment using its respective layers
    embedOpen ((ot, os), (l1tr, l1slc)) = otEmb + osEmb
     where
      otEmb = activation $ T.forward l1tr $ T.forward tr ot
      osEmb = activation $ T.forward l1slc $ T.forward slc os
    -- the list of layers for the 3 open transitions and slices
    openEncoders =
      [ (stL1openSlc1, stL1openTr1)
      , (stL1openSlc2, stL1openTr2)
      , (stL1openSlc3, stL1openTr3)
      ]
    -- embed the open segments and add them to mid and frozen
    fullEmb = F.foldl' (+) midAndFrozen $ zipWith (curry embedOpen) open openEncoders
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
  , qModelFinal1 :: !(TT.Linear (GenEmbSize (QSpecGeneral spec)) (QSpecSpecial spec) QDType QDevice)
  , qModelNorm1 :: !(TT.LayerNorm '[QSpecSpecial spec] QDType QDevice)
  , qModelFinal2 :: !(TT.Linear (QSpecSpecial spec) 1 QDType QDevice)
  , qModelValue1 :: !(TT.Linear (GenEmbSize (QSpecGeneral spec)) (QSpecSpecial spec) QDType QDevice)
  , qModelValueNorm :: !(TT.LayerNorm '[QSpecSpecial spec] QDType QDevice)
  , qModelValue2 :: !(TT.Linear (QSpecSpecial spec) 1 QDType QDevice)
  }
  deriving (Show, Generic, TT.Parameterized, NoThunks, NFData)

instance
  ( spec ~ TQSpecData g sp sl tr ac st
  , embsize ~ GenEmbSize g
  , KnownNat (PSize g)
  , KnownNat (ESize g)
  , KnownNat sp
  , KnownNat sl
  , KnownNat tr
  , KnownNat ac
  , KnownNat st
  , KnownNat embsize
  , KnownNat (embsize - 3)
  -- , KnownNat (embsize + embsize)
  -- , KnownNat (embsize + (embsize + embsize))
  )
  => T.Randomizable (QSpec spec) (QModel spec)
  where
  sample :: QSpec spec -> IO (QModel spec)
  sample QSpec = do
    qModelSlc <- T.sample (GeneralSpec, SliceSpec)
    qModelTr <- T.sample (GeneralSpec, TransitionSpec)
    qModelAct <- T.sample (GeneralSpec, ActionSpec)
    qModelSt <- T.sample (GeneralSpec, StateSpec)
    qModelFinal1 <- T.sample TT.LinearSpec
    qModelNorm1 <- T.sample $ TT.LayerNormSpec 1e-05
    qModelFinal2 <- T.sample TT.LinearSpec
    qModelValue1 <- T.sample TT.LinearSpec
    qModelValueNorm <- T.sample $ TT.LayerNormSpec 1e-05
    qModelValue2 <- T.sample TT.LinearSpec
    pure QModel{..}

-- | HasForward for model (unbatched)
instance
  ( gspec ~ QSpecGeneral spec
  , ((GenEmbSize gspec - 3) + 3) ~ GenEmbSize gspec
  , PShape gspec ~ '[FakeSize, PSize gspec]
  , TT.KnownShape (PShape gspec)
  , TT.KnownShape (EShape gspec)
  , KnownNat (QSpecSpecial spec)
  , KnownNat (PSize gspec)
  , KnownNat (GenEmbSize gspec)
  , KnownNat (QSpecTrans spec)
  , TT.CheckIsSuffixOf '[QSpecSpecial spec] '[QSpecSpecial spec] (QSpecSpecial spec == QSpecSpecial spec)
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
  , ((GenEmbSize gspec - 3) + 3) ~ GenEmbSize gspec
  , PShape gspec ~ '[FakeSize, PSize gspec]
  , TT.KnownShape (PShape gspec)
  , TT.KnownShape (EShape gspec)
  , KnownNat (QSpecSpecial spec)
  , KnownNat (PSize gspec)
  , KnownNat (GenEmbSize gspec)
  , KnownNat (QSpecTrans spec)
  , KnownNat batchSize
  , 1 <= batchSize
  , TT.CheckIsSuffixOf '[QSpecSpecial spec] [batchSize, QSpecSpecial spec] (QSpecSpecial spec == QSpecSpecial spec)
  )
  => T.HasForward (QModel spec) (QEncoding '[batchSize] gspec) (QTensor '[batchSize, 1])
  where
  forward :: QModel spec -> QEncoding '[batchSize] (QSpecGeneral spec) -> QTensor '[batchSize, 1]
  forward model encoding =
    TT.log $ TT.sigmoid $ forwardQModelBatched model encoding

  forwardStoch model input = pure $ T.forward model input

forwardQModel
  :: forall spec gspec
   . ( gspec ~ QSpecGeneral spec
     , ((GenEmbSize gspec - 3) + 3) ~ GenEmbSize gspec
     , PShape gspec ~ '[FakeSize, PSize gspec]
     , TT.KnownShape (PShape gspec)
     , TT.KnownShape (EShape gspec)
     , KnownNat (PSize gspec)
     , KnownNat (QSpecSpecial spec)
     , KnownNat (GenEmbSize gspec)
     , KnownNat (QSpecTrans spec)
     , TT.CheckIsSuffixOf '[QSpecSpecial spec] '[QSpecSpecial spec] (QSpecSpecial spec == QSpecSpecial spec)
     )
  => QModel spec
  -> QEncoding '[] (QSpecGeneral spec)
  -> QTensor '[1]
forwardQModel (QModel slc tr act st final1 norm1 final2 _ _ _) (QEncoding actEnc stEnc) =
  T.forward final2 $
    activation $
      T.forward norm1 $
        T.forward final1 (actEmb + stEmb)
 where
  actEmb :: QTensor '[GenEmbSize gspec]
  actEmb = T.forward act (slc, tr, actEnc)
  stEmb :: QTensor '[GenEmbSize gspec]
  stEmb = T.forward st (slc, tr, stEnc)

forwardQModelBatched
  :: forall gspec spec batchSize
   . ( gspec ~ QSpecGeneral spec
     , ((GenEmbSize gspec - 3) + 3) ~ GenEmbSize gspec
     , PShape gspec ~ '[FakeSize, PSize gspec]
     , TT.KnownShape (PShape gspec)
     , TT.KnownShape (EShape gspec)
     , KnownNat (QSpecSpecial spec)
     , KnownNat (PSize gspec)
     , KnownNat (GenEmbSize gspec)
     , KnownNat (QSpecTrans spec)
     , KnownNat batchSize
     , 1 <= batchSize
     , TT.CheckIsSuffixOf '[QSpecSpecial spec] [batchSize, QSpecSpecial spec] (QSpecSpecial spec == QSpecSpecial spec)
     )
  => QModel spec
  -> QEncoding '[batchSize] (QSpecGeneral spec)
  -> QTensor '[batchSize, 1]
forwardQModelBatched (QModel slc tr act st final1 norm1 final2 _ _ _) (QEncoding actEncs stEnc) = out2
 where
  actEmb :: QTensor '[batchSize, GenEmbSize gspec]
  actEmb = T.forward act (slc, tr, actEncs)
  stEmb :: QTensor '[GenEmbSize gspec]
  stEmb = T.forward st (slc, tr, stEnc)
  inputEmb = actEmb `TT.add` stEmb
  out1 :: QTensor '[batchSize, QSpecSpecial spec]
  out1 = activation $ T.forward norm1 $ T.forward final1 inputEmb
  out2 :: QTensor '[batchSize, 1]
  out2 = T.forward final2 out1

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
     , ((GenEmbSize gspec - 3) + 3) ~ GenEmbSize gspec
     , PShape gspec ~ [FakeSize, PSize gspec]
     , TT.KnownShape (PShape gspec)
     , TT.KnownShape (EShape gspec)
     , KnownNat (PSize gspec)
     , KnownNat (GenEmbSize gspec)
     , KnownNat (QSpecSpecial spec)
     , KnownNat (QSpecTrans spec)
     , TT.CheckIsSuffixOf '[QSpecSpecial spec] '[QSpecSpecial spec] (QSpecSpecial spec == QSpecSpecial spec)
     )
  => QModel spec
  -> StateEncoding (QSpecGeneral spec)
  -> QTensor '[1]
forwardValue (QModel slc tr _ st _ _ _ value1 norm value2) stateEncoding =
  TT.log $
    TT.sigmoid $
      T.forward value2 $
        activation $
          T.forward norm $
            -- DT.traceShowId $
            T.forward value1 $
              T.forward st (slc, tr, stateEncoding)

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
  tensors :: (TT.HMap' TT.ToDependent (TT.Parameters (QModel DefaultQSpec)) ts) => TT.HList ts <-
    TT.load path
  params <- TT.hmapM' TT.MakeIndependent tensors
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
