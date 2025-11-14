{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DerivingStrategies #-}

{- | /This module is deprecated, use "Scoring.FunTyped" instead./

 Semiring scores with "holes".
 Holes are used to express "partially applied" scores that occur
 when the score of a verticalization (unspread) is distributed to two parent edges.
 The full score of the operation is restored when the two parent edges are eventually combined again.

 This module implements partial scores using lists,
 which is slow and not very elegant.
 The grammar combinators are partial and will fail if used incorrectly,
 indicating parser bugs.
-}
module Scoring.Deprecated.Flat
  ( -- * The Score Type
    Score (..)
  , Holes
  , val
  , LeftId (..)
  , RightId (..)
  , leftSide
  , rightSide
  , showScore

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

import Control.DeepSeq (NFData)
import Data.Bifunctor (first)
import Data.Foldable (foldl')
import Data.Hashable (Hashable)
import Data.List qualified as L
import Data.Maybe (fromMaybe)
import Data.Semiring qualified as R
import GHC.Generics (Generic)

----------------
-- Score type --
----------------

-- | Newtype for the left ID of a partial score.
newtype LeftId i = LeftId i
  deriving (Eq, Ord, Generic)
  deriving anyclass (Hashable, NFData)

instance Show i => Show (LeftId i) where
  show (LeftId i) = show i

-- | Newtype for the right ID of a partial score.
newtype RightId i = RightId i
  deriving (Eq, Ord, Generic)
  deriving anyclass (NFData, Hashable)

instance Show i => Show (RightId i) where
  show (RightId i) = show i

match :: Eq a => RightId a -> LeftId a -> Bool
match (RightId ir) (LeftId il) = il == ir

-- | A type alias for the holes in a 'Score'.
type Holes s = [s]

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
data Score s i
  = -- | Carries a fully applied value
    SVal !s
  | -- | The right part of a combination, expects an argument to its left.
    -- Implemented as a list of right elements
    SRight !(LeftId i) ![Holes s]
  | -- | The left part of a combination, expects an argument to its right.
    -- Implemented as a list of right elements
    SLeft ![Holes s] !(RightId i)
  | -- | A combination of 'SLeft' and 'SRight' that expects arguments on both sides.
    -- Implemented as a list of right elements on the left
    -- and a list of left elements to the right
    SBoth !(LeftId i) ![(Holes s, Holes s)] !(RightId i)
  deriving (Generic, NFData)

-- | Creates a simple value score of type ()-().
val :: s -> Score s i
val = SVal

{- | Returns the ID on the left side of an 'Score',
 or 'Nothing' for 'SVal' and 'SLeft'.

 > a-b -> a
-}
leftSide :: Score s i -> Maybe (LeftId i)
leftSide (SVal _) = Nothing
leftSide (SLeft _ _) = Nothing
leftSide (SRight i _) = Just i
leftSide (SBoth i _ _) = Just i

{- | Returns the ID on the right side of an 'Score',
 or 'Nothing' for 'SVal' and 'SRight'.

 > a-b -> b
-}
rightSide :: Score s i -> Maybe (RightId i)
rightSide (SVal _) = Nothing
rightSide (SLeft _ i) = Just i
rightSide (SRight _ _) = Nothing
rightSide (SBoth _ _ i) = Just i

-- Show instance

showLeftHoles :: Show a => [a] -> [Char]
showLeftHoles ls = L.intercalate " _ " (show <$> ls) <> " _"

showRightHoles :: Show a => [a] -> [Char]
showRightHoles rs = "_ " <> L.intercalate " _ " (show <$> rs)

showBothHoles :: (Show a1, Show a2) => ([a1], [a2]) -> [Char]
showBothHoles (ls, rs) = showLeftHoles ls <> " | " <> showRightHoles rs

showOpts :: (a -> [Char]) -> [a] -> [Char]
showOpts shower opts = "-[" <> L.intercalate " / " (shower <$> opts) <> "]-"

instance (Show i, Show s) => Show (Score s i) where
  show (SVal s) = "()-" <> show s <> "-()"
  show (SLeft ls ir) = "()" <> showOpts showLeftHoles ls <> show ir
  show (SRight il rs) = show il <> showOpts showRightHoles rs <> "()"
  show (SBoth il bs ir) = show il <> showOpts showBothHoles bs <> show ir

-- simplified showing (only "type")

{- | Returns a string representation of a 'Score'
 (more compact than it's 'Show' instance).
-}
showScore :: (Show s, Show i) => Score s i -> String
showScore (SVal v) = show v
showScore (SLeft _ ir) = "()-" <> show ir
showScore (SRight il _) = show il <> "-()"
showScore (SBoth il _ ir) = show il <> "-" <> show ir

-------------------------
-- semiring operations --
-------------------------

zipHoles :: R.Semiring s => Holes s -> Holes s -> Maybe s
zipHoles !lefts !rights = go R.one lefts rights
 where
  go !acc [] [] = Just $! acc
  go !acc (l : ls) (r : rs) = go (acc R.* (l R.* r)) ls rs
  go _ _ _ = Nothing

combineAlts :: (a -> b -> Maybe c) -> [a] -> [b] -> Maybe [c]
combineAlts f ls rs = sequence $ f <$> ls <*> rs

appendRight :: R.Semiring s => s -> Holes s -> Holes s
appendRight !s [] = [s]
appendRight !s [!a] = [a R.* s]
appendRight !s (a : as) = a : appendRight s as

prependLeft :: R.Semiring s => s -> Holes s -> Holes s
prependLeft !s [] = [s]
prependLeft !s (a : as) = (s R.* a) : as

addHoleLeft :: s -> Holes s -> Holes s
addHoleLeft = (:)

{- | Combines two partially applied 'Score's
 by applying them to each other and/or multiplying the underlying semiring values.
 Shapes and IDs at the adjacent sides must match, otherwise 'Nothing' is returned.

 > a-b × b-c -> a-c
-}
times
  :: (R.Semiring s, Eq i, Show i) => Score s i -> Score s i -> Maybe (Score s i)
-- creates value
times (SVal !s1) (SVal !s2) = Just $! SVal $! s1 R.* s2
times (SLeft !ls !il) (SRight !ir !rs)
  | il `match` ir =
      SVal . foldl' R.plus R.zero <$> combineAlts zipHoles ls rs
-- creates right
times (SRight !i !rs) (SVal !s) = Just $! SRight i $! appendRight s <$> rs
times (SBoth !il !bs !ir) (SRight !i !rs)
  | ir `match` i =
      SRight il <$> combineAlts bplusr bs rs
 where
  bplusr (bl, br) r = flip appendRight bl <$> zipHoles br r
-- creates left
times (SVal !s) (SLeft !ls !i) = Just $! SLeft (prependLeft s <$> ls) i
times (SLeft !ls !i) (SBoth !il !bs !ir) | i `match` il =
  fmap (`SLeft` ir) $ sequence $ do
    !l <- ls
    (!bl, !br) <- bs
    let !vl = zipHoles l bl
    pure $! flip prependLeft br <$> vl
-- creates both
times (SRight !il !rs) (SLeft !ls !ir) = Just $! SBoth il bs ir
 where
  bs = do
    !l <- rs
    !r <- ls
    pure (l, r)
times (SBoth !il !as !ia) (SBoth !ib !bs !ir) | ia `match` ib = do
  -- Maybe
  cs <- sequence $ do
    -- []
    (!al, !ar) <- as
    (!bl, !br) <- bs
    pure $! do
      -- Maybe
      !vm <- zipHoles ar bl
      pure (al, prependLeft vm br)
  pure $! SBoth il cs ir
-- otherwise
times _ _ = Nothing

{- | Adds two partially applied 'Score's
 by adding their underlying (or resulting) semiring values.
 This operation is only admitted
 if the two scores are of the same shape and have matching IDs.
 Otherwise, 'Nothing' is returned.

 > a-b + a-b -> a-b
-}
plus :: (R.Semiring s, Eq i) => Score s i -> Score s i -> Maybe (Score s i)
plus (SVal !s1) (SVal !s2) = Just $! SVal (s1 R.+ s2)
plus (SRight !i !rs1) (SRight !i' !rs2)
  | i == i' =
      Just $! SRight i (rs1 <> rs2)
plus (SLeft !ls1 !i) (SLeft !ls2 !i') | i == i' = Just $! SLeft (ls1 <> ls2) i
plus (SBoth !il !bs1 !ir) (SBoth !il' !bs2 !ir')
  | il == il' && ir == ir' =
      Just $! SBoth il (bs1 <> bs2) ir
plus _ _ = Nothing

-----------
-- rules --
-----------

{- | Extracts the value from a fully applied 'Score'.
 This function is intended to be used to extract the final score of the parser.
 If the score is not fully applied,
 throws an exception to indicate parser bugs.
-}
getScoreVal :: Score s i -> s
getScoreVal (SVal s) = s
getScoreVal _ = error "cannot get value from partial score"

{- | Adds two 'Score's that are alternative derivations of the same transition.
 This is expected to be called on compatible scores
 and will throw an error otherwise to indicate parser bugs.

 > a-b   a-b
 > --------- add
 >    a-b
-}
addScores :: (R.Semiring s, Eq i) => Score s i -> Score s i -> Score s i
addScores a b = fromMaybe (error "illegal times") $ plus a b

{- | Combines the 'Score's of two edges with a @split@ operation into the score of the parent edge.
 This is expected to be called on compatible scores
 and will throw an error otherwise to indicate parser bugs.

 > a-b   b-c
 > --------- unsplit
 >    a-c
-}
unsplitScores
  :: (R.Semiring s, Eq i, Show i, Show s)
  => s
  -- ^ The score of the split operation.
  -> Score s i
  -- ^ The 'Score' of the left child edge.
  -> Score s i
  -- ^ The 'Score' of the right child edge.
  -> Score s i
  -- ^ The 'Score' of the parent edge, if it exists.
unsplitScores op left right = fromMaybe err $ times left' right
 where
  err =
    error $
      "Attempting illegal unsplit: left="
        <> show left
        <> ", right="
        <> show right
  left' = case left of
    SVal s -> SVal (op R.* s)
    SLeft ls i -> SLeft (prependLeft op <$> ls) i
    SRight i rs -> SRight i (prependLeft op <$> rs)
    SBoth il bs ir -> SBoth il (first (prependLeft op) <$> bs) ir

{- | Creates the 'Score' of a left parent edge from a left child edge of an @unspread@.
 Will throw an error if called on invalid input to indicate parser bugs.
-}
unspreadScoresLeft
  :: (Eq i, Show i, R.Semiring s, Show s)
  => i
  -- ^ The new ID that marks both parent edges
  -> Score s i
  -- ^ The 'Score' of the left child edge.
  -> Score s i
  -- ^ The 'Score' of the left parent edge, if it exists.
unspreadScoresLeft newid = wrap
 where
  newir = RightId newid
  -- wrap the left input score into a new layer with a new ID
  wrap (SVal v) = SLeft [[R.one, v]] newir
  wrap (SLeft ls _) = SLeft (addHoleLeft R.one <$> ls) newir
  wrap other = error $ "Attempting illegal left-unspread on " <> show other

{- | Creates the 'Score' of a right parent edge
 from the middle and right child edges of an @unspread@
 and a @spread@ operation.
-}
unspreadScoresRight
  :: (Eq i, R.Semiring s, Show i, Show s)
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
unspreadScoresRight newid op m r = fromMaybe err $ do
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
  unwrap (SVal s) = SRight newil [[op, s]]
  unwrap (SRight _ rs) = SRight newil (addHoleLeft op <$> rs)
  unwrap (SLeft ls ir) = SBoth newil (([op, R.one],) <$> ls) ir
  unwrap (SBoth _ bs ir) = SBoth newil (first (addHoleLeft op) <$> bs) ir
