{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}

{- | Semiring scores with "holes".
 Holes are used to express "partially applied" scores that occur
 when the score of a verticalization (unspread) is distributed to the two parent edges.
 The full score of the operation is restored when the two parent edges are eventually combined again.

 This module implements partial scores as typesafe functions with phantom types
 that indicate the number of holes on each side.
 The grammatical combinators use an existential type 'Score'
 that reifies the phantom types as singletons,
 which allows different scores to be easily stored together
 and is compatible with the 'Score' types from the other Scoring* modules.
 Thus, the grammatical combinators are partial
 and fail when used with incompatible scores, indicating a parser bug.
-}
module Scoring.FunTyped
  ( -- * The Score Type
    Score (..)
  , TypedScore (..)
  , val
  , showScore

    -- ** IDs
  , LeftId (..)
  , RightId (..)
  , leftSide
  , rightSide

    -- ** Hole Types
  , LeftHoles
  , RightHoles
  , RightHole
  , BothHoles

    -- * Semiring operations

    -- | Semiring operations can be lifted to partial scores,
    -- but since it is not guaranteed that their arguments can be combined,
    -- they are partial.
  , times
  , plus

    -- * grammatical combinators

    -- | The following combinators correspond to the unsplit and unspread operations
    -- of the path-graph grammar.
  , unsplitScores
  , unspreadScoresLeft
  , unspreadScoresRight
  , addScores
  , getScoreVal
  ) where

import Control.DeepSeq
  ( NFData (rnf)
  , deepseq
  )
import Data.Hashable (Hashable)
import Data.Kind (Type)
import Data.Maybe (fromMaybe)
import Data.Semiring qualified as R
import Data.Type.Equality
import Data.Type.Nat
  ( Nat (..)
  , SNat (..)
  )
import GHC.Generics (Generic)

----------------
-- Score type --
----------------

-- | Newtype for the left ID of a partial score.
newtype LeftId i = LeftId i
  deriving (Eq, Ord, Generic, NFData, Hashable)

instance Show i => Show (LeftId i) where
  show (LeftId i) = show i

-- | Newtype for the right ID of a partial score.
newtype RightId i = RightId i
  deriving (Eq, Ord, Generic, NFData, Hashable)

instance Show i => Show (RightId i) where
  show (RightId i) = show i

match :: Eq a => RightId a -> LeftId a -> Bool
match (RightId ir) (LeftId il) = il == ir

-- | A single right hole (helper for 'RightHoles').
data RightHole (n :: Nat) s where
  RightHole :: !(RightHoles n s) -> RightHole ('S n) s
  RightEnd :: !s -> RightHole 'Z s

instance (NFData s) => NFData (RightHole n s) where
  rnf (RightHole f) = rnf f
  rnf (RightEnd s) = rnf s

-- | The type of a function representing @n@ right holes.
type RightHoles (n :: Nat) s = s -> RightHole n s

-- | The type of a function representing @n@ left holes.
type LeftHoles (n :: Nat) s = (RightHoles n s -> s)

-- | The type of a function containing @nl@ left and @nr@ right holes.
type BothHoles (nl :: Nat) (nr :: Nat) s = (RightHoles nr s -> RightHoles nl s)

{- | A partially applied score of type @s@.
 Comes in four variants,
 depending on whether the score is fully applied
 or needs to be combined on either or both sides.
 Values that need to be combined are lists that represent scores with holes.
 Each variant carries IDs of type @i@ that determine which objects fit on either of its sides.
 Only score objects with matching IDs can be combined.

 As a shorthand notation, we use @a-b@ to indicate a value
 that depends on @a@ on its left and on @b@ on its right.
 If the value does not depend on anything on either side, we use @()@,
 i.e. @()-a@ stands for @SLeft _ a@ and @()-()@ stands for @SVal _@.
-}
data TypedScore (nl :: Nat) (nr :: Nat) s i where
  -- | Carries a fully applied value
  SVal :: !s -> TypedScore 'Z 'Z s i
  -- | The right part of a combination, expects an argument to its left.
  -- Implemented as a function that takes a left counterpart
  -- and returns a score with fewer holes.
  SRight :: !(LeftId i) -> !(RightHoles nl s) -> TypedScore ('S nl) 'Z s i
  -- | The left part of a combination, expects an argument to its right.
  -- Implemented as a function that takes a right hole and applies it.
  SLeft :: !(LeftHoles nr s) -> !(RightId i) -> TypedScore 'Z ('S nr) s i
  -- | A combination of 'SLeft' and 'SRight' that expects arguments on both sides.
  -- Implemented as a function that expects both a right and a left hole to be complete.
  SBoth :: !(LeftId i) -> !(BothHoles nl nr s) -> !(RightId i) -> TypedScore ('S nl) ('S nr) s i

instance (NFData s, NFData i) => NFData (TypedScore nl nr s i) where
  rnf (SVal s) = rnf s
  rnf (SRight i _) = rnf i
  rnf (SLeft _ i) = rnf i
  rnf (SBoth il _ ir) = il `deepseq` rnf ir

{- | A paritally applied score of type @s@
 with an unknown number of holes (as used by the "ChartParser").
 Wraps a 'TypedScore' together with witnesses for the number of holes on each side.
-}
data Score s i :: Type where
  MkScore :: SNat nl -> SNat nr -> TypedScore nl nr s i -> Score s i

instance (NFData s, NFData i) => NFData (Score s i) where
  rnf (MkScore _ _ s) = rnf s

-- | Creates a simple value score of type ()-().
val :: s -> Score s i
val s = MkScore SZ SZ (SVal s)

{- | Returns the ID on the left side of an 'Score',
 or 'Nothing' for 'SVal' and 'SLeft'.

 > a-b -> a
-}
leftSide :: Score s i -> Maybe (LeftId i)
leftSide (MkScore _ _ (SVal _)) = Nothing
leftSide (MkScore _ _ (SLeft _ _)) = Nothing
leftSide (MkScore _ _ (SRight i _)) = Just i
leftSide (MkScore _ _ (SBoth i _ _)) = Just i

{- | Returns the ID on the right side of an 'Score',
 or 'Nothing' for 'SVal' and 'SRight'.

 > a-b -> b
-}
rightSide :: Score s i -> Maybe (RightId i)
rightSide (MkScore _ _ (SVal _)) = Nothing
rightSide (MkScore _ _ (SLeft _ i)) = Just i
rightSide (MkScore _ _ (SRight _ _)) = Nothing
rightSide (MkScore _ _ (SBoth _ _ i)) = Just i

instance (Show i) => Show (TypedScore nl nr s i) where
  show (SVal _) = "()-()"
  show (SLeft _ ir) = "()-" <> show ir
  show (SRight il _) = show il <> "-()"
  show (SBoth il _ ir) = show il <> "-" <> show ir

instance (Show i) => Show (Score s i) where
  show (MkScore _ _ s) = show s

showTScore :: (Show s, Show i) => TypedScore nl nr s i -> String
showTScore (SVal v) = show v
showTScore (SLeft _ ir) = "()-" <> show ir
showTScore (SRight il _) = show il <> "-()"
showTScore (SBoth il _ ir) = show il <> "-" <> show ir

{- | Returns a string representation of a 'Score'
 (more compact than it's 'Show' instance).
-}
showScore :: (Show s, Show i) => Score s i -> String
showScore (MkScore _ _ s) = showTScore s

-------------------------
-- semiring operations --
-------------------------

appendRight :: R.Semiring s => s -> RightHoles n s -> RightHoles n s
appendRight !s' fr !l = app $ fr l
 where
  app (RightEnd r) = RightEnd $ r R.* s'
  app (RightHole fr') = RightHole $ appendRight s' fr'

prependLeft :: R.Semiring s => s -> LeftHoles n s -> LeftHoles n s
prependLeft !s fl = (s R.*) . fl

prependRight :: R.Semiring s => s -> RightHoles n s -> RightHoles n s
prependRight !s fr !l = prep $ fr l
 where
  prep (RightEnd r) = RightEnd $ s R.* r
  prep (RightHole fr') = RightHole $ prependRight s fr'

{- | Combines two partially applied 'Score's
 by applying them to each other and/or multiplying the underlying semiring values.
 Shapes and IDs at the adjacent sides must match, otherwise 'Nothing' is returned.

 > a-b × b-c -> a-c
-}
times
  :: (R.Semiring s, Eq i, Show i)
  => TypedScore nl n s i
  -> TypedScore n nr s i
  -> Maybe (TypedScore nl nr s i)
-- creates value
times (SVal s1) (SVal s2) = Just $! SVal $ s1 R.* s2
times (SLeft fl il) (SRight ir fr) | il `match` ir = Just $! SVal $ fl fr
-- creates right
times (SRight i r) (SVal s) = Just $! SRight i $ appendRight s r
times (SBoth il fb ir) (SRight i fr) | ir `match` i = Just $ SRight il $ fb fr
-- creates left
times (SVal s) (SLeft fl i) = Just $! SLeft (prependLeft s fl) i
times (SLeft fl i) (SBoth il fb ir) | i `match` il = Just $! SLeft (fl . fb) ir
-- creates both
times (SRight il fr) (SLeft fl ir) =
  Just $! SBoth il ((`appendRight` fr) . fl) ir
times (SBoth il fa ia) (SBoth ib fb ir)
  | ia `match` ib =
      Just $! SBoth il (fa . fb) ir
-- otherwise
times _ _ = Nothing

rhplus :: (R.Semiring s) => RightHoles n s -> RightHoles n s -> RightHoles n s
rhplus r1 r2 !l = case (r1 l, r2 l) of
  (RightEnd e1, RightEnd e2) -> RightEnd $ e1 R.+ e2
  (RightHole f1, RightHole f2) -> RightHole $ rhplus f1 f2

{- | Adds two partially applied 'TypedScore's
 by adding their underlying (or resulting) semiring values.
 This operation is only admitted
 if the two scores are of the same shape and have matching IDs.
 Otherwise, 'Nothing' is returned.

 > a-b + a-b -> a-b
-}
plus
  :: (R.Semiring s, Eq i)
  => TypedScore nl nr s i
  -> TypedScore nl nr s i
  -> Maybe (TypedScore nl nr s i)
plus (SVal s1) (SVal s2) = Just $! SVal $ s1 R.+ s2
plus (SRight i fr1) (SRight i' fr2)
  | i == i' =
      Just $! SRight i $ rhplus fr1 fr2
plus (SLeft fl1 i) (SLeft fl2 i')
  | i == i' =
      Just $! SLeft (\r -> fl1 r R.+ fl2 r) i
plus (SBoth il bs1 ir) (SBoth il' bs2 ir')
  | il == il' && ir == ir' =
      Just $! SBoth il (\r -> rhplus (bs1 r) (bs2 r)) ir
plus _ _ = Nothing

-- helpers for constructing holes
-- ------------------------------

addHoleLeft :: s -> LeftHoles n s -> LeftHoles ('S n) s
addHoleLeft !s fl fr = case fr s of
  RightHole fr' -> fl fr'

mkLeftHole :: s -> LeftHoles 'Z s
mkLeftHole !s fr = case fr s of
  RightEnd v -> v

addHoleRight :: R.Semiring s => s -> RightHoles n s -> RightHoles ('S n) s
addHoleRight !r !rh !l = RightHole $ prependRight (l R.* r) rh

mkRightHole :: R.Semiring s => s -> RightHoles 'Z s
mkRightHole !r !l = RightEnd $ l R.* r

-- proof helpers

type family CreateHole (n :: Nat) where
  CreateHole 'Z = 'S ('S 'Z)
  CreateHole ('S n) = 'S ('S n)

class AddHole (n :: Nat) where
  addHole :: SNat n -> SNat (CreateHole n)

instance AddHole 'Z where
  addHole SZ = SS

instance AddHole ('S n) where
  addHole SS = SS

canAddHole :: SNat n -> (AddHole n => r) -> r
canAddHole SZ r = r
canAddHole SS r = r

-----------
-- rules --
-----------

{- | Extracts the value from a fully applied 'Score'.
 This function is intended to be used to extract the final score of the parser.
 If the score is not fully applied,
 throws an exception to indicate parser bugs.
-}
getScoreVal :: Score s i -> s
getScoreVal (MkScore _ _ (SVal s)) = s
getScoreVal _ = error "cannot get value from partial score"

{- | Adds two 'Score's that are alternative derivations of the same transition.
 This is expected to be called on compatible scores
 and will throw an error otherwise to indicate parser bugs.

 > a-b   a-b
 > --------- add
 >    a-b
-}
addScores :: (R.Semiring s, Eq i) => Score s i -> Score s i -> Score s i
addScores (MkScore nla nra a) (MkScore nlb nrb b) = MkScore nla nra res
 where
  res = fromMaybe (error "illegal plus") $ do
    Refl <- testEquality nla nlb
    Refl <- testEquality nra nrb
    plus a b

{- | Combines the 'Score's of two edges with a @split@ operation into the score of the parent edge.
 This is expected to be called on compatible scores
 and will throw an error otherwise to indicate parser bugs.

 > a-b   b-c
 > --------- unsplit
 >    a-c
-}
unsplitScores
  :: forall s i
   . (R.Semiring s, Eq i, Show i, Show s)
  => s
  -- ^ The score of the split operation.
  -> Score s i
  -- ^ The 'Score' of the left child edge.
  -> Score s i
  -- ^ The 'Score' of the right child edge.
  -> Score s i
  -- ^ The 'Score' of the parent edge, if it exists.
unsplitScores op (MkScore nll nrl left) (MkScore nlr nrr right) =
  MkScore nll nrr $ fromMaybe err $ do
    Refl <- testEquality nrl nlr
    times (prep left) right
 where
  err =
    error $
      "Attempting illegal unsplit: left="
        <> show left
        <> ", right="
        <> show right
  prep :: TypedScore nl nr s i -> TypedScore nl nr s i
  prep l = case l of
    SVal s -> SVal (op R.* s)
    SLeft fl i -> SLeft (prependLeft op fl) i
    SRight i rs -> SRight i (prependRight op rs)
    SBoth il bs ir -> SBoth il (prependRight op . bs) ir

{- | Creates the 'Score' of a left parent edge from a left child edge of an @unspread@.
 Will throw an error if called on invalid input to indicate parser bugs.
-}
unspreadScoresLeft
  :: forall s i
   . (Eq i, Show i, R.Semiring s, Show s)
  => i
  -- ^ The new ID that marks both parent edges
  -> Score s i
  -- ^ The 'Score' of the left child edge.
  -> Score s i
  -- ^ The 'Score' of the left parent edge, if it exists.
unspreadScoresLeft newid (MkScore SZ nr s) =
  canAddHole nr $ MkScore SZ (addHole nr) $ wrap s
 where
  newir = RightId newid
  -- wrap the left input score into a new layer with a new ID
  wrap :: TypedScore 'Z nr s i -> TypedScore 'Z (CreateHole nr) s i
  wrap (SVal v) = SLeft (addHoleLeft R.one $ mkLeftHole v) newir
  wrap (SLeft fl _) = SLeft (addHoleLeft R.one fl) newir
unspreadScoresLeft _ (MkScore _ _ s) =
  error $ "Attempting illegal left-unspread on " <> show s

{- | Creates the 'Score' of a right parent edge
 from the middle and right child edges of an @unspread@
 and a @spread@ operation.
-}
unspreadScoresRight
  :: forall i s
   . (Eq i, R.Semiring s, Show i, Show s)
  => i
  -- ^ The new ID that marks both parent edges.
  -> s
  -- ^ The score of the @spread@ operation.
  -> Score s i
  -- ^ The 'Score' of the middle child edge.
  -> Score s i
  -- ^ The 'Score' of the right child edge.
  -> Score s i
  -- ^ The 'Score' of the right parent edge, if it exists.
unspreadScoresRight newid op (MkScore nlm nrm m) (MkScore nlr nrr r) =
  canAddHole nlm $ MkScore (addHole nlm) nrr $ fromMaybe err $ do
    Refl <- testEquality nrm nlr
    mr <- times m r
    pure $ unwrap mr
 where
  err =
    error $
      "Attempting illegal right-unspread: m="
        <> show m
        <> ", r="
        <> show r
  -- generate a value on the right
  -- that consumes the left parent edge's value when supplied
  -- and combines with m on the right
  newil = LeftId newid
  unwrap :: TypedScore nl nr s i -> TypedScore (CreateHole nl) nr s i
  unwrap (SVal s) = SRight newil $ addHoleRight op $ mkRightHole s -- [op, s]
  unwrap (SRight _ rs) = SRight newil (addHoleRight op rs)
  unwrap (SLeft fl ir) =
    SBoth newil ((`appendRight` addHoleRight op (mkRightHole R.one)) . fl) ir
  unwrap (SBoth _ fb ir) = SBoth newil (addHoleRight op . fb) ir
