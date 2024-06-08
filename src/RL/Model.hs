{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}
{-# HLINT ignore "Use <$>" #-}
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}

module RL.Model where

import Common
import Control.Arrow ((>>>))
import Data.Foldable qualified as F
import Data.Function ((&))
import Data.Type.Equality (type (==))
import Data.TypeNums (KnownInt, KnownNat, Nat, TInt (..), intVal, intVal', type (*), type (+), type (-))
import Debug.Trace qualified as DT
import GHC.Generics (Generic)
import Internal.TorchHelpers qualified as TH
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

-- Learned Constant Embeddings
-- ---------------------------

data ConstEmbSpec (shape :: [Nat]) = ConstEmbSpec

newtype ConstEmb shape = ConstEmb (TT.Parameter QDevice QDType shape)
  deriving (Show, Generic)
  deriving anyclass (TT.Parameterized)

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
  { _slcL1 :: !(TT.Linear (TT.Product (PShape spec)) hidden QDType QDevice)
  , _slcL2 :: !(TT.Linear hidden (GenEmbSize spec) QDType QDevice)
  , _slcStart :: !(ConstEmb '[GenEmbSize spec])
  , _slcStop :: !(ConstEmb '[GenEmbSize spec])
  }
  deriving (Show, Generic, TT.Parameterized)

instance
  ( KnownNat hidden
  , KnownNat (GenEmbSize spec)
  , KnownNat (TT.Product (PShape spec))
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

instance
  (embshape ~ '[GenEmbSize spec], pshape ~ PShape spec, TT.KnownShape pshape)
  => TT.HasForward (SliceEncoder spec hidden) (StartStop (QTensor pshape)) (QTensor embshape)
  where
  forward model@(SliceEncoder _ _ start stop) input =
    case input of
      Inner slc -> T.forward model slc
      Start -> T.forward start ()
      Stop -> T.forward stop ()
  forwardStoch model input = pure $ T.forward model input

instance
  ( embshape ~ '[GenEmbSize spec]
  , pshape ~ PShape spec
  , TT.KnownShape pshape
  )
  => T.HasForward (SliceEncoder spec hidden) (QTensor pshape) (QTensor embshape)
  where
  forward (SliceEncoder l1 l2 _ _) =
    TT.flattenAll
      >>> T.forward l1
      >>> activation
      >>> TT.forward l2
      >>> activation
  forwardStoch model = pure . T.forward model

-- Transition Encoder
-- ------------------

data TransitionSpec (hidden :: Nat) = TransitionSpec

data TransitionEncoder spec hidden = TransitionEncoder
  { trL1Passing :: !(TT.Linear (TT.Product (EShape spec)) hidden QDType QDevice)
  , trL1Inner :: !(TT.Linear (TT.Product (EShape spec)) hidden QDType QDevice)
  , trL1Left :: !(TT.Linear (TT.Product (PShape spec)) hidden QDType QDevice)
  , trL1Right :: !(TT.Linear (TT.Product (PShape spec)) hidden QDType QDevice)
  , trL1Root :: !(ConstEmb '[hidden])
  , trL2 :: !(TT.Linear hidden (GenEmbSize spec) QDType QDevice)
  }
  deriving (Show, Generic, TT.Parameterized)

instance
  ( KnownNat hidden
  , KnownNat (GenEmbSize spec)
  , KnownNat (TT.Product (PShape spec))
  , KnownNat (TT.Product (EShape spec))
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

instance
  forall spec hidden embshape
   . ( embshape ~ '[GenEmbSize spec]
     , TT.KnownShape (EShape spec)
     , TT.KnownShape (PShape spec)
     )
  => T.HasForward
      (TransitionEncoder spec hidden)
      (TransitionEncoding spec)
      (QTensor embshape)
  where
  forward TransitionEncoder{..} TransitionEncoding{..} = activation $ T.forward trL2 all'
   where
    pass = activation $ T.forward trL1Passing $ TT.flattenAll trencPassing
    inner = activation $ T.forward trL1Inner $ TT.flattenAll trencInner
    left = activation $ T.forward trL1Left $ TT.flattenAll trencLeft
    right = activation $ T.forward trL1Right $ TT.flattenAll trencRight
    all = pass + inner + left + right
    all' = if trencRoot then all + activation (T.forward trL1Root ()) else all
  forwardStoch = undefined

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
  deriving (Show, Generic, TT.Parameterized)

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

instance
  forall (spec :: TGeneralSpec) slcHidden trHidden actHidden outShape emb
   . ( emb ~ GenEmbSize spec
     , outShape ~ '[emb]
     , emb ~ (emb - 3) + 3
     , TT.KnownShape (PShape spec)
     , TT.KnownShape (EShape spec)
     )
  => T.HasForward
      (ActionEncoder spec actHidden)
      (SliceEncoder spec slcHidden, TransitionEncoder spec trHidden, ActionEncoding spec)
      (QTensor outShape)
  where
  forward ActionEncoder{..} (slc, tr, ActionEncoding top op) = topEmb + opEmb
   where
    topCombined :: QTensor '[actHidden]
    topCombined = case top of
      Left (sl, t, sr) ->
        let
          embl = activation $ T.forward actTop1sl $ T.forward slc sl
          embt = activation $ T.forward actTop1t1 $ T.forward slc sl
          embr = activation $ T.forward actTop1sr $ T.forward slc sl
         in
          embl + embt + embr
      Right (sl, t1, sm, t2, sr) ->
        let
          embl = activation $ T.forward actTop1sl $ T.forward slc sl
          embm = activation $ T.forward actTop1sl $ T.forward slc sm
          embr = activation $ T.forward actTop1sl $ T.forward slc sr
          embt1 = activation $ T.forward actTop1sl $ T.forward tr t1
          embt2 = activation $ T.forward actTop1sl $ T.forward tr t2
         in
          embl + embm + embr + embt1 + embt2
    topEmb :: QTensor '[GenEmbSize spec]
    topEmb = activation $ T.forward actTop2 topCombined
    opEmb :: QTensor '[GenEmbSize spec]
    opEmb = case op of
      LMFreezeOnly _ -> TT.cat @0 (TT.selectIdx @0 opTypes 0 TT.:. T.forward actFreeze () TT.:. TT.HNil)
      LMSplitOnly _ -> TT.cat @0 (TT.selectIdx @0 opTypes 1 TT.:. T.forward actSplit () TT.:. TT.HNil)
      LMFreezeLeft _ -> TT.cat @0 (TT.selectIdx @0 opTypes 2 TT.:. T.forward actFreeze () TT.:. TT.HNil)
      LMSpread _ -> TT.cat @0 (TT.selectIdx @0 opTypes 3 TT.:. T.forward actSpread () TT.:. TT.HNil)
      LMSplitLeft _ -> TT.cat @0 (TT.selectIdx @0 opTypes 4 TT.:. T.forward actSplit () TT.:. TT.HNil)
      LMSplitRight _ -> TT.cat @0 (TT.selectIdx @0 opTypes 5 TT.:. T.forward actSplit () TT.:. TT.HNil)
    opTypes :: QTensor '[6, 3]
    opTypes =
      TT.UnsafeMkTensor $
        T.asTensor' @[[QType]]
          [ [0, 0, 0] -- freeze only
          , [0, 1, 0] -- split only
          , [1, 0, 0] -- freeze left
          , [1, 0, 1] -- spread
          , [1, 1, 0] -- freeze left
          , [1, 1, 1] -- freeze right
          ]
          opts
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
  deriving (Show, Generic, TT.Parameterized)

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

instance
  forall (spec :: TGeneralSpec) slcHidden trHidden stHidden outShape emb
   . ( emb ~ GenEmbSize spec
     , outShape ~ '[emb]
     , TT.KnownShape (PShape spec)
     , TT.KnownShape (EShape spec)
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

-- Full Model
-- ----------

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
  }
  deriving (Show, Generic, TT.Parameterized)

instance
  ( spec ~ TQSpecData g sp sl tr ac st
  , embsize ~ GenEmbSize g
  , KnownNat sp
  , KnownNat sl
  , KnownNat tr
  , KnownNat ac
  , KnownNat st
  , KnownNat embsize
  , KnownNat (embsize - 3)
  , -- , KnownNat (embsize + embsize)
    -- , KnownNat (embsize + (embsize + embsize))
    KnownNat (TT.Product (PShape g))
  , KnownNat (TT.Product (EShape g))
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
    pure QModel{..}

instance
  ( gspec ~ QSpecGeneral spec
  , ((GenEmbSize gspec - 3) + 3) ~ GenEmbSize gspec
  , TT.KnownShape (PShape gspec)
  , TT.KnownShape (EShape gspec)
  , KnownNat (QSpecSpecial spec)
  , TT.CheckIsSuffixOf '[QSpecSpecial spec] '[QSpecSpecial spec] (QSpecSpecial spec == QSpecSpecial spec)
  )
  => T.HasForward (QModel spec) (QEncoding gspec) (QTensor '[1])
  where
  forward :: QModel spec -> QEncoding (QSpecGeneral spec) -> QTensor '[1]
  forward (QModel slc tr act st final1 norm1 final2) (QEncoding actEnc stEnc) =
    TT.log $
      TT.sigmoid $
        T.forward final2 $
          activation $
            T.forward norm1 $
              T.forward final1 (actEmb + stEmb)
   where
    actEmb :: QTensor '[GenEmbSize gspec]
    actEmb = T.forward act (slc, tr, actEnc)
    stEmb :: QTensor '[GenEmbSize gspec]
    stEmb = T.forward st (slc, tr, stEnc)

  forwardStoch :: QModel spec -> QEncoding (QSpecGeneral spec) -> IO (QTensor '[1])
  forwardStoch model input = pure $ T.forward model input

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
  :: (s -> a -> QEncoding (QSpecGeneral DefaultQSpec))
  -> QModel DefaultQSpec
  -> s
  -> a
  -> QType
runQ !encode !model s a = T.asValue $ TT.toDynamic $ T.forward model $ encode s a

runQ'
  :: (s -> a -> QEncoding (QSpecGeneral DefaultQSpec))
  -> QModel DefaultQSpec
  -> s
  -> a
  -> QTensor '[1]
runQ' !encode !model s a = T.forward model $ encode s a
