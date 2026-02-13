{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DerivingVia #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE TypeData #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}

module RL.ModelTypes where

import GreedyParser (Action, GreedyState)
import Musicology.Pitch (SPitch)
import PVGrammar

import Common (Eval)
import Control.DeepSeq
import Data.Kind (Type)
import Data.List.NonEmpty qualified as NE
import Data.Proxy (Proxy (Proxy))
import Data.Type.Equality (type (==))
import Data.TypeNums (KnownInt, KnownNat, Nat, TInt (..), intVal, type (*), type (+), type (-), type (<=))
import GHC.Generics (Generic)
import NoThunks.Class (NoThunks (..), OnlyCheckWhnf (..), allNoThunks)
import Torch qualified as T
import Torch.Lens qualified
import Torch.Typed qualified as TT

-- global settings
-- ---------------

-- device :: T.Device
-- device = T.Device T.CPU 0
-- type QDevice = '(TT.CPU, 0)

-- device = T.Device T.CUDA 0
-- type QDevice = '(TT.CUDA, 0)

type QDType = TT.Double

type IsValidDevice dev =
  ( TT.GeluDTypeIsValid dev QDType
  , TT.RandDTypeIsValid dev QDType
  , TT.MatMulDTypeIsValid dev QDType
  , TT.BasicArithmeticDTypeIsValid dev QDType
  , TT.SumDTypeIsValid dev QDType
  , TT.MeanDTypeValidation dev QDType
  , TT.StandardFloatingPointDTypeValidation dev QDType
  , TT.StandardDTypeValidation dev QDType
  , TT.ComparisonDTypeIsValid dev QDType
  , TT.KnownDevice dev
  )

type IsValidHidden (hidden :: Nat) =
  ( KnownNat hidden
  , KnownNat (hidden - 3)
  , 3 <= hidden
  , 1 <= hidden
  , -- , TT.CheckIsSuffixOf '[hidden] [1, hidden] (hidden == hidden)
    -- , TT.CheckIsSuffixOf '[hidden] '[hidden] (hidden == hidden)
    (hidden == hidden) ~ 'True
  )

type ValidParams dev hidden = (IsValidDevice dev, IsValidHidden hidden)

type QType = Double

inf :: QType
inf = 1 / 0

qDType :: TT.DType
qDType = T.Double

type QTensor device shape = TT.Tensor device QDType shape

opts :: forall dev. (TT.KnownDevice dev) => T.TensorOptions
opts = T.withDevice dev $ T.withDType qDType $ T.defaultOpts
 where
  dev = TT.deviceVal @dev

toOpts :: forall dev a. (TT.KnownDevice dev, Torch.Lens.HasTypes a T.Tensor) => a -> a
toOpts = T.toDevice device . T.toType qDType
 where
  device = TT.deviceVal @dev

toQTensor' :: forall dev a. (TT.KnownDevice dev, T.TensorLike a) => a -> T.Tensor
toQTensor' a = T.asTensor' a $ opts @dev

toQTensor :: forall dev. (TT.KnownDevice dev) => QType -> QTensor dev '[]
toQTensor = TT.UnsafeMkTensor . toQTensor' @dev

type MaxPitches = 8 :: Nat
type MaxEdges = 8 :: Nat

-- States and Actions
-- ------------------

type PVAction = Action (Notes SPitch) (Edges SPitch) (Split SPitch) (Freeze SPitch) (Spread SPitch)

type PVState = GreedyState (Edges SPitch) [Edge SPitch] (Notes SPitch) (PVLeftmost SPitch)

type PVActionResult = Either PVState (Edges SPitch, [PVLeftmost SPitch])

type PVRewardFn label = PVActionResult -> Maybe (NE.NonEmpty PVAction) -> PVAction -> label -> IO QType

type PVEval p = Eval (Edges p) [Edge p] (Notes p) [Note p] (Spread p) (PVLeftmost p)

-- General Spec
-- ------------

type FifthLow = Neg 3
type FifthPadding = 6
type OctaveLow = (Pos 2)
type OctavePadding = 2

type FifthSize = (2 * FifthPadding) + 1
type OctaveSize = (2 * OctavePadding) + 1

type PShape = '[FifthSize, OctaveSize]
type PSize = FifthSize + OctaveSize

type ESize = PSize + PSize
type EShape' batchSize = '[batchSize, ESize]

intValI :: forall n. (KnownInt n) => Int
intValI = fromInteger $ intVal @n Proxy

-- orphan instances
-- ================

deriving instance Generic (TT.Tensor dev dtype shape)

-- deriving newtype instance NFData T.IndependentTensor

-- deriving instance NFData (TT.Parameter dev dtype shape)

-- deriving instance NFData (TT.Linear nin nout dtype dev)

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

deriving instance NoThunks TT.Dropout
deriving instance NFData TT.Dropout

deriving instance NoThunks (TT.MultiheadAttention emd kemb vemb heads dtype dev)
deriving instance NFData (TT.MultiheadAttention emd kemb vemb heads dtype dev)

deriving instance NoThunks (TT.TransformerMLP emd ffndim dtype dev)
deriving instance NFData (TT.TransformerMLP emd ffndim dtype dev)

deriving instance NoThunks (TT.TransformerLayer emd kemb vemb heads ffndim dtype dev)
deriving instance NFData (TT.TransformerLayer emd kemb vemb heads ffndim dtype dev)

deriving instance NoThunks (TT.LayerNorm shape dtype dev)
deriving instance NFData (TT.LayerNorm shape dtype dev)

instance NoThunks (TT.HList '[]) where
  showTypeOf _ = "HNil"
  wNoThunks _ctxt TT.HNil = pure Nothing

instance (NoThunks x, NoThunks (TT.HList xs)) => NoThunks (TT.HList (x : (xs :: [Type]))) where
  showTypeOf _ = "HCons " <> showTypeOf (Proxy @x)
  wNoThunks ctxt (TT.HCons (x, xs)) = allNoThunks [noThunks ctxt x, noThunks ctxt xs]

instance NFData (TT.HList '[]) where
  rnf TT.HNil = ()

instance (NFData x, NFData (TT.HList xs)) => NFData (TT.HList (x : xs :: [Type])) where
  rnf (TT.HCons (x, xs)) = deepseq x $ rnf xs

deriving instance Generic (TT.Adam momenta)
deriving instance (NoThunks (TT.HList momenta)) => NoThunks (TT.Adam momenta)
deriving instance (NFData (TT.HList momenta)) => NFData (TT.Adam momenta)

deriving instance Generic TT.GD
deriving instance NoThunks TT.GD
deriving instance NFData TT.GD
