{-# LANGUAGE ExplicitNamespaces #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Common where

-- StartStop
-- =========

import           GHC.Generics                   ( Generic )
import           Control.DeepSeq                ( NFData )

import qualified Data.Semiring                 as R
import qualified Data.Set                      as S
import           Data.Foldable                  ( foldl'
                                                , foldlM
                                                )
import qualified Control.Monad.Writer.Strict   as MW
import           Debug.Trace                    ( trace )
import           Data.Semigroup                 ( stimesMonoid )
import qualified Control.Monad.Trans.State.Strict
                                               as ST
import           GHC.TypeNats                   ( Nat
                                                , KnownNat(..)
                                                , type (<=)
                                                , type (+)
                                                , type (-)
                                                , natVal
                                                )
import           Control.Monad.Identity         ( runIdentity )
import           Data.Bifunctor                 ( second )
import           Data.Typeable                  ( Proxy(Proxy) )

-- | A container type that augements the type @a@
-- with symbols for beginning (@:⋊@) and end (@:⋉@).
-- Every other value is wrapped in an @Inner@ constructor.
data StartStop a = (:⋊)
                  | Inner !a
                  | (:⋉)
  deriving (Ord, Eq, Generic)

-- some instances for StartStop

instance (NFData a) => NFData (StartStop a)

instance Show a => Show (StartStop a) where
  show (:⋊)      = "⋊"
  show (:⋉)      = "⋉"
  show (Inner a) = show a

instance Functor StartStop where
  fmap f (:⋊)      = (:⋊)
  fmap f (:⋉)      = (:⋉)
  fmap f (Inner a) = Inner $ f a

-- some helper functions for StartStop

-- | From a list of @StartStop@s returns only the elements that are not @:⋊@ or @:⋉@,
-- unwrapped to their original type.
onlyInner :: [StartStop a] -> [a]
onlyInner []              = []
onlyInner (Inner a : rst) = a : onlyInner rst
onlyInner (_       : rst) = onlyInner rst

-- | Returns the content of an 'Inner', or 'Nothing'.
getInner :: StartStop a -> Maybe a
getInner (Inner a) = Just a
getInner _         = Nothing

getInnerE :: StartStop a -> Either String a
getInnerE (Inner a) = Right a
getInnerE (:⋊)      = Left "expected inner but found ⋊"
getInnerE (:⋉)      = Left "expected inner but found ⋉"

isInner (Inner _) = True
isInner _         = False

isStart (:⋊) = True
isStart _    = False

isStop (:⋉) = True
isStop _    = False

distStartStop :: StartStop (a, b) -> (StartStop a, StartStop b)
distStartStop (:⋊)           = ((:⋊), (:⋊))
distStartStop (:⋉)           = ((:⋉), (:⋉))
distStartStop (Inner (a, b)) = (Inner a, Inner b)

-- evaluator interface
-- ===================

type IsLast = Bool

data SplitType = LeftOfTwo
               | LeftOnly
               | RightOfTwo

-- | An evaluator for verticalizations.
-- Returns the verticalization of a (middle) transition, if possible.
type VertMiddle e a v = (a, e, a) -> Maybe (a, v)

-- | An evaluator returning the possible left parent edges of a verticalization.
type VertLeft e a = (e, a) -> a -> [e]

-- | An evaluator returning the possible right parent edges of a verticalization.
type VertRight e a = (a, e) -> a -> [e]

-- | An evaluator for merges.
-- Returns possible merges of a given pair of transitions.
type Merge e a v
  = StartStop a -> e -> a -> e -> StartStop a -> SplitType -> [(e, v)]

-- | A combined evaluator for verticalizations, merges, and thaws.
-- Additionally, contains a function for mapping terminal slices to semiring values.
data Eval e e' a a' v = Eval
  { evalVertMiddle  :: !(VertMiddle e a v)
  , evalVertLeft :: !(VertLeft e a)
  , evalVertRight :: !(VertRight e a)
  , evalMerge :: !(Merge e a v)
  , evalThaw  :: !(StartStop a -> Maybe e' -> StartStop a -> IsLast -> [(e, v)])
  , evalSlice :: !(a' -> a)
  }

-- | Maps a function over all scores produced by the evaluator.
mapEvalScore :: (v -> w) -> Eval e e' a a' v -> Eval e e' a a' w
mapEvalScore f (Eval vm vl vr m t s) = Eval vm' vl vr m' t' s
 where
  vm' = fmap (fmap f) . vm
  m' sl l sm r sr typ = fmap f <$> m sl l sm r sr typ
  t' l e r isLast = fmap f <$> t l e r isLast

-- product evaluators
-- ------------------

productEval
  :: Eval e1 e' a1 a' v1
  -> Eval e2 e' a2 a' v2
  -> Eval (e1, e2) e' (a1, a2) a' (v1, v2)
productEval (Eval vertm1 vertl1 vertr1 merge1 thaw1 slice1) (Eval vertm2 vertl2 vertr2 merge2 thaw2 slice2)
  = Eval vertm vertl vertr merge thaw slice
 where
  vertm ((l1, l2), (m1, m2), (r1, r2)) = do
    (a, va) <- vertm1 (l1, m1, r1)
    (b, vb) <- vertm2 (l2, m2, r2)
    pure ((a, b), (va, vb))
  vertl ((l1, l2), (c1, c2)) (t1, t2) = do
    a <- vertl1 (l1, c1) t1
    b <- vertl2 (l2, c2) t2
    pure (a, b)
  vertr ((c1, c2), (r1, r2)) (t1, t2) = do
    a <- vertr1 (c1, r1) t1
    b <- vertr2 (c2, r2) t2
    pure (a, b)
  merge sl (tl1, tl2) (sm1, sm2) (tr1, tr2) sr typ = do
    (a, va) <- merge1 sl1 tl1 sm1 tr1 sr1 typ
    (b, vb) <- merge2 sl2 tl2 sm2 tr2 sr2 typ
    pure ((a, b), (va, vb))
   where
    (sl1, sl2) = distStartStop sl
    (sr1, sr2) = distStartStop sr
  thaw l e r isLast = do
    (a, va) <- thaw1 l1 e r1 isLast
    (b, vb) <- thaw2 l2 e r2 isLast
    pure ((a, b), (va, vb))
   where
    (l1, l2) = distStartStop l
    (r1, r2) = distStartStop r
  slice s = (slice1 s, slice2 s)

-- restricting branching
-- ---------------------

data RightBranchHori = RBBranches
                        | RBClear
  deriving (Eq, Ord, Show)

evalRightBranchHori :: Eval RightBranchHori e' () a' ()
evalRightBranchHori = Eval vertm vertl vertr merge thaw slice
 where
  vertm (_, RBBranches, _) = Nothing
  vertm (_, RBClear   , _) = Just ((), ())
  vertl _ _ = [RBClear]
  vertr _ _ = [RBBranches]
  merge _ _ _ _ _ _ = [(RBClear, ())]
  thaw _ _ _ _ = [(RBClear, ())]
  slice _ = ()

rightBranchHori
  :: Eval e2 e' a2 a' w -> Eval (RightBranchHori, e2) e' ((), a2) a' w
rightBranchHori = mapEvalScore snd . productEval evalRightBranchHori

-- restricting derivation order
-- ----------------------------

data Merged = Merged
               | NotMerged
  deriving (Eq, Ord, Show)

evalSplitBeforeHori :: (Eval Merged e' () a' ())
evalSplitBeforeHori = Eval vertm vertl vertr merge thaw slice
 where
  vertm _ = Just ((), ())
  vertl (Merged   , _) _ = []
  vertl (NotMerged, _) _ = [NotMerged]
  vertr (_, Merged   ) _ = []
  vertr (_, NotMerged) _ = [NotMerged]
  merge _ _ _ _ _ _ = [(Merged, ())]
  thaw _ _ _ _ = [(NotMerged, ())]
  slice _ = ()

splitFirst :: Eval e2 e' a2 a' w -> Eval (Merged, e2) e' ((), a2) a' w
splitFirst = mapEvalScore snd . productEval evalSplitBeforeHori

-- left-most derivation outer operations
-- =====================================

data Leftmost s f h = LMSplitLeft !s
                    | LMFreeze !f
                    | LMSplitRight !s
                    | LMHorizontalize !h
                    | LMSplitLeftOnly !s
                    | LMFreezeOnly !f
  deriving (Eq, Ord, Show)

mkLeftmostEval
  :: VertMiddle e a h
  -> VertLeft e a
  -> VertRight e a
  -> (StartStop a -> e -> a -> e -> StartStop a -> [(e, s)])
  -> (StartStop a -> Maybe e' -> StartStop a -> [(e, f)])
  -> (a' -> a)
  -> Eval e e' a a' (Leftmost s f h)
mkLeftmostEval vm vl vr m t = Eval vm' vl vr m' t'
 where
  smap f = fmap (second f)
  -- vm' :: VertMiddle e a (Leftmost s f h)
  vm' vert = smap LMHorizontalize $ vm vert
  m' sl tl sm tr sr typ = smap split res
   where
    res   = m sl tl sm tr sr
    split = case typ of
      LeftOfTwo  -> LMSplitLeft
      LeftOnly   -> LMSplitLeftOnly
      RightOfTwo -> LMSplitRight
  t' sl e sr isLast | isLast    = smap LMFreezeOnly res
                    | otherwise = smap LMFreeze res
    where res = t sl e sr

newtype PartialDeriv s f h (n :: Nat) (snd :: Bool) = PD { runPD :: [Leftmost s f h]}

buildDerivation
  :: (PartialDeriv s f h 1 False -> PartialDeriv s f h n snd)
  -> [Leftmost s f h]
buildDerivation build = reverse $ runPD $ build (PD [])

buildPartialDerivation
  :: Proxy (n :: Nat)
  -> (PartialDeriv s f h n False -> PartialDeriv s f h n' snd)
  -> [Leftmost s f h]
buildPartialDerivation _ build = reverse $ runPD $ build (PD [])

infixl 1 .>
f .> g = g . f

infixr 2 $$
f $$ a = f a

split
  :: forall n s f h
   . (KnownNat n, 1 <= n)
  => s
  -> PartialDeriv s f h n False
  -> PartialDeriv s f h (n+1) False
split s (PD d) | natVal (Proxy :: Proxy n) == 1 = PD $ LMSplitLeftOnly s : d
               | otherwise                      = PD $ LMSplitLeft s : d

freeze
  :: forall n s h f
   . (KnownNat n, 1 <= n)
  => f
  -> PartialDeriv s f h n False
  -> PartialDeriv s f h (n-1) False
freeze f (PD d) | natVal (Proxy :: Proxy n) == 1 = PD $ LMFreezeOnly f : d
                | otherwise                      = PD $ LMFreeze f : d

-- freezeOnly :: f -> PartialDeriv s f h 1 False -> PartialDeriv s f h 0 False
-- freezeOnly f (PD d) = PD $ LMFreezeOnly f : d

splitRight
  :: (2 <= n) => s -> PartialDeriv s f h n snd -> PartialDeriv s f h (n+1) True
splitRight s (PD d) = PD $ LMSplitRight s : d

hori
  :: (2 <= n) => h -> PartialDeriv s f h n snd -> PartialDeriv s f h (n+1) False
hori h (PD d) = PD $ LMHorizontalize h : d

-- useful semirings
-- ================

data Derivations a = Do !a
                   | Or (Derivations a) (Derivations a)
                   | Then !(Derivations a) !(Derivations a)
                   | NoOp
                   | Cannot
  deriving (Eq, Ord)

data DerivOp = OpNone
             | OpOr
             | OpThen
  deriving Eq

instance Show a => Show (Derivations a) where
  show = go 0 OpNone
   where
    indent n = stimesMonoid n "  "
    go n _    (Do a)   = indent n <> show a
    go n _    NoOp     = indent n <> "NoOp"
    go n _    Cannot   = indent n <> "Cannot"
    go n OpOr (Or a b) = go n OpOr a <> "\n" <> go n OpOr b
    go n _ (Or a b) =
      indent n <> "Or\n" <> go (n + 1) OpOr a <> "\n" <> go (n + 1) OpOr b
    go n OpThen (Then a b) = go n OpThen a <> "\n" <> go n OpThen b
    go n _ (Then a b) =
      indent n <> "Then\n" <> go (n + 1) OpThen a <> "\n" <> go (n + 1) OpThen b


instance R.Semiring (Derivations a) where
  zero = Cannot
  one  = NoOp
  plus Cannot a      = a
  plus a      Cannot = a
  plus a      b      = Or a b
  times Cannot _      = Cannot
  times _      Cannot = Cannot
  times NoOp   a      = a
  times a      NoOp   = a
  times a      b      = Then a b

mapDerivations :: (R.Semiring r) => (a -> r) -> Derivations a -> r
mapDerivations f (Do a)     = f a
mapDerivations f NoOp       = R.one
mapDerivations f Cannot     = R.zero
mapDerivations f (Or   a b) = mapDerivations f a R.+ mapDerivations f b
mapDerivations f (Then a b) = mapDerivations f a R.* mapDerivations f b

flattenDerivations :: Ord a => Derivations a -> S.Set [a]
flattenDerivations = mapDerivations (\a -> S.singleton [a])

flattenDerivationsRed :: Ord a => Derivations a -> [[a]]
flattenDerivationsRed (Do a) = pure [a]
flattenDerivationsRed NoOp   = pure []
flattenDerivationsRed Cannot = []
flattenDerivationsRed (Or a b) =
  flattenDerivationsRed a <> flattenDerivationsRed b
flattenDerivationsRed (Then a b) = do
  a <- flattenDerivationsRed a
  b <- flattenDerivationsRed b
  pure (a <> b)

-- utilities
-- =========

traceLevel :: Int
traceLevel = 0

traceIf :: Int -> [Char] -> Bool -> Bool
traceIf l msg value =
  if traceLevel >= l && value then trace msg value else value
