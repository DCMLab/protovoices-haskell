{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE LambdaCase #-}

{- | A chart-based semiring parser for path grammars (e.g. the PV grammar).
Path grammars operate on "paths"
consisting of nodes (slices) and edges (transitions),
both of which can contain arbitrary content.
Paths are elaborated through two operations,
@split@ting transitions and @spread@ing slices
(plus @freeze@, which terminates generation on a transition).

The parser is polymorphic in the grammar
as well as the contents of slices (path nodes) and transitions (path edges).
The grammar to parse is definend in an "evaluator" ('Common.Eval')
which provides completions for parsing the splits, spreads and freezes.
-}
module ChartParser
  ( -- * Parsing Interface
    parse
  , parseSize
  , parseSilent
  , logSize
  , logTikz

    -- * Charts

    -- ** Basic Elements
  , Slice
  , Transition
  , transLen
  , Item
  , TItem

    -- ** Transition Chart
  , TContents
  , TChart
  , tcGetByLength

    -- ** Verticalization Chart
  , Vert
  , VChart
  , vcGetByLength

    -- * Constraint Aliases
  , Parsable
  , Normal
  , Normal'
  ) where

import Common
import Scoring.FunTyped qualified as S

import Data.HashMap.Strict qualified as HM
import Data.IntMap.Strict qualified as IM
import Data.Semiring qualified as R

import Control.Monad (foldM, foldM_, mzero)
import Control.Monad.State as ST

import Control.DeepSeq
import Control.Parallel.Strategies qualified as P
import Data.Foldable (foldl')
import Data.Hashable
  ( Hashable
  , hash
  , hashWithSalt
  )
import Data.Kind (Constraint, Type)
import Data.Maybe
  ( catMaybes
  , fromMaybe
  , mapMaybe
  , maybeToList
  )
import Data.Set qualified as Set
import GHC.Generics (Generic)

-- Basic Types
-- ===========

-- | An alias for common constraints on slices and transitions
type Normal :: Type -> Constraint
type Normal x = (Eq x, Ord x, Show x, Hashable x, NFData x)

-- | An alias for common constraints on semiring values
type Normal' :: Type -> Constraint
type Normal' x = (Eq x, Show x, NFData x, R.Semiring x)

-- | A summary constraint for transitions, slices, and semiring values
type Parsable' :: Type -> Type -> Type -> Constraint
type Parsable' tr slc v = (Normal tr, Normal slc, Normal' v)

-- | A summary constraint for transitions, slices, hori-operations, and semiring values
type Parsable :: Type -> Type -> Type -> Type -> Constraint
type Parsable tr slc h v = (Normal tr, Normal slc, Normal h, Normal' v)

-- Slices
---------

{- | A slice during chart parsing.
 Besides the slice content (e.g., notes),
 it maintains indices to the first and last surface slice covered,
 as well as an ID that is used for matching compatible parents of a spread.
-}
data Slice slc = Slice
  { sFirst :: !Int
  -- ^ index of the first surface slice covered
  , sContent :: !(StartStop slc)
  -- ^ slice content (or 'Start'/'Stop')
  , sID :: !Int
  -- ^ unique slice ID
  , sLast :: !Int
  -- ^ index of the last surface slice covered
  }
  deriving (Eq, Ord, Generic, NFData)

instance (Eq slc) => Hashable (Slice slc) where
  hashWithSalt s (Slice _ _ i _) = hashWithSalt s i

instance (Show slc) => Show (Slice slc) where
  show (Slice f c i l) =
    show f <> "-" <> show c <> "@" <> show i <> "-" <> show l

-- Transitions
--------------

{- | A transition during chart parsing.
 Has pointers to the two slices it connects,
 a content (e.g., protovoice connections),
 and a flag indicating whether it is the second (right) parent of a spread.
-}
data Transition tr slc = Transition
  { tLeftSlice :: !(Slice slc)
  , tContent :: !tr
  , tRightSlice :: !(Slice slc)
  , t2nd :: !Bool
  }
  deriving (Eq, Ord, Generic, NFData, Hashable)

instance (Show a, Show e) => Show (Transition e a) where
  show (Transition l c r s) =
    "<"
      <> show l
      <> ","
      <> show c
      <> ","
      <> show r
      <> ">"
      <> if s
        then "2"
        else ""

-- | Returns the "length" of the transition in terms of surface slices covered.
transLen :: Transition e a -> Int
transLen (Transition l _ r _) = sLast r - sFirst l + 1

-- Items
--------

{- | A parsing item.
 Combines an intermediate value (e.g. a transition) with a semiring score.
-}
data Item i v = (:=)
  { iItem :: !i
  , iScore :: !(S.Score v Int)
  }
  deriving (Generic, NFData)

instance (Show i, Show v) => Show (Item i v) where
  show (i := v) = show i <> " := " <> show v

-- | A transition item.
type TItem tr slc v = Item (Transition tr slc) v

-- Vert Items

{- | Represents the middle part of an incomplete unspread ("verticalization").
 Expresses how the middle transition and the two child slices (@vMiddle@)
 are derived from the parent slice (@vTop@) using a spread operation (@vOp@).

 'Vert' objects are stored in the 'VChart'
 to record the intermediate steps of an unspread,
 which is found by first parsing the middle transition into the parent slice
 (generating a 'Vert')
 and then combining the 'Vert' with the left and right child transitions
 to generate the left and right parent transitions, respectively.
-}
data Vert tr slc h v = Vert
  { vTop :: !(Slice slc)
  , vOp :: !h
  , vVal :: !v
  , vMiddle :: !(TItem tr slc v)
  }
  deriving (Generic, NFData)

instance (Show e, Show a, Show h, Show v) => Show (Vert e a h v) where
  show (Vert top op val m) =
    "Vert"
      <> "\n top: "
      <> show top
      <> "\n op:  "
      <> show op
      <> "\n val:  "
      <> show val
      <> "\n m:   "
      <> show m

-- slice and transition charts
-- ===========================

-- vert chart
-------------

-- ops:
-- - get all of len n
-- - get all with left child = x
-- - get all with right child = x
-- - check ID for (top,left,leftid)

{- | A verticalization chart.
 Stores 'Vert' objects at certain chart positions.
 To support efficient lookup of 'Vert' objects from different indices,
 each 'Vert' is redundantly stored in several hash maps,
 one for each index:

 - by surface length
 - by surface length (only left border of a 'Vert')
 - by left child slice ID and mid transition length
 - by right child ID

 In addition, the 'VChart' maintains IDs of new slices.
 (Every new slice is the parent of an unspread.)
-}
data VChart tr slc h v = VChart
  { vcNextId :: !Int
  -- ^ next free ID
  , vcIDs :: !(HM.HashMap (Int, Int) Int)
  -- ^ a mapping from child slice ids to the corresponding parent id
  , vcByLength :: !(IM.IntMap [Vert tr slc h v])
  -- ^ maps surface length to the 'Vert' with that length
  , vcByLengthLeft :: !(IM.IntMap (Set.Set (Slice slc, Slice slc, h)))
  -- ^ maps surface length to the "left borders" of 'Vert' objects with that length
  -- (parent slice, left child slice)
  , vcByLeftChild :: !(HM.HashMap (Int, Int) (Set.Set (Slice slc, h)))
  -- ^ maps a left child slice ID and the surface length of the middle transition
  -- to its potential parent slices
  , vcByRightChild :: !(HM.HashMap (Int, Int) [Vert tr slc h v])
  -- ^ maps a right child slice ID and the surface length of the middle transition
  -- to all 'Vert' objects it is part of.
  }
  deriving (Generic, NFData)

instance (Show tr, Show slc, Show h, Show v) => Show (VChart tr slc h v) where
  show (VChart n _ is _ _ _) = "VChart (next id: " <> show n <> ")" <> levels
   where
    levels = concatMap showLevel $ IM.toAscList is
    showLevel (l, items) = "\nlevel " <> show l <> ":" <> sitems
     where
      sitems = concatMap (("\n  " <>) . show) items

-- | Returns an empty 'VChart' with the next free ID set to @n + 1@.
vcEmpty :: Int -> VChart tr slc h v
vcEmpty n = VChart (n + 1) HM.empty IM.empty IM.empty HM.empty HM.empty

-- | Insert a new 'Vert' object into a 'VChart'.
vcInsert
  :: (Hashable slc, Ord slc, Ord h)
  => VChart tr slc h v
  -- ^ the old chart
  -> (slc, h, v, TItem tr slc v)
  -- ^ the new 'Vert' item's parent slice, operation, value, and middle child transition.
  -> VChart tr slc h v
  -- ^ the new chart
vcInsert (VChart nextid ids bylen bylenleft byleft byright) (topContent, op, val, mid@(tmid := _)) =
  let left = tLeftSlice tmid
      right = tRightSlice tmid
      idKey = (sID left, sID right)
      (nextid', ids', i) = case HM.lookup idKey ids of
        Just i' -> (nextid, ids, i')
        Nothing -> (nextid + 1, HM.insert idKey nextid ids, nextid)
      top = Slice (sFirst left) (Inner topContent) i (sLast right)
      vert = [Vert top op val mid]
      vert' = Set.singleton (top, tLeftSlice tmid, op)
      vertl = Set.singleton (top, op)
      bylen' = IM.insertWith (<>) (transLen tmid) vert bylen
      bylenleft' = IM.insertWith (<>) (transLen tmid) vert' bylenleft
      byleft' = HM.insertWith (<>) (sID left, transLen tmid) vertl byleft
      byright' = HM.insertWith (<>) (sID right, transLen tmid) vert byright
   in VChart nextid' ids' bylen' bylenleft' byleft' byright'

-- | Merge a sequence of new items into a 'VChart'
vcMerge
  :: (Foldable t, Ord slc, Hashable slc, Ord h)
  => VChart tr slc h v
  -> t (slc, h, v, TItem tr slc v)
  -> VChart tr slc h v
vcMerge = foldl' vcInsert

-- | Returns all 'Vert' objects in the 'VChart' with the same length.
vcGetByLength
  :: VChart tr slc h v
  -- ^ the chart
  -> Int
  -- ^ surface length of a middle transition
  -> [Vert tr slc h v]
  -- ^ all corresponding 'Vert' objects
vcGetByLength chart len = fromMaybe [] $ IM.lookup len $ vcByLength chart

-- | Returns the "left borders" of all 'Vert' objects in the 'VChart' with the same length.
vcGetByLengthLeft
  :: VChart tr slc h v
  -- ^ the chart
  -> Int
  -- ^ the surface length of a middle transition
  -> [(Slice slc, Slice slc, h)]
  -- ^ (parent slice, left slice, op) of all corresponding 'Vert' objects (without duplicates)
vcGetByLengthLeft chart len =
  maybe [] Set.toList $ IM.lookup len (vcByLengthLeft chart)

{- | Returns the all potential parents of a left child slice
 up to a certain middle transition length.
-}
vcGetByLeftChild
  :: (Ord slc, Hashable slc, Ord h)
  => Int
  -- ^ maximum middle transition length
  -> VChart tr slc h v
  -- ^ the chart
  -> Slice slc
  -- ^ the left child slice
  -> [(Slice slc, h)]
  -- ^ all potential parent slices
vcGetByLeftChild maxn chart left =
  Set.toList $ Set.unions $ mapMaybe getN [2 .. maxn]
 where
  getN n = HM.lookup (sID left, n) $ vcByLeftChild chart

{- | Returns all 'Vert' objects with the same right child
 up to a certain middle transition length.
-}
vcGetByRightChild
  :: (Ord slc, Hashable slc)
  => Int
  -- ^ ID of the right child
  -> VChart tr slc h v
  -> Slice slc
  -> [Vert tr slc h v]
vcGetByRightChild maxn chart right =
  concat $ mapMaybe getN [2 .. maxn]
 where
  getN n = HM.lookup (sID right, n) $ vcByRightChild chart

-- transition chart
-------------------

-- ops:
-- - get all of length n
-- - get all with left slice l
-- - get all with right slice r

{- | The contents of a transition chart (under a particular index).
 A mapping from transitions (with score ID constraints left and right)
 to (partial) semiring scores.
 This mapping usually contains all transition items that satisfy a certain criterion,
 irrespective of their position in the chart (which is encoded in the transitions themselves).

 When new transition items are added, if the transition already exists in the chart
 (as the result of a different partial parse),
 the scores of the new and existing items are "added" (this also requires the score IDs to match).
-}
type TContents tr slc v =
  HM.HashMap
    (Transition tr slc, Maybe (S.LeftId Int), Maybe (S.RightId Int))
    (S.Score v Int)

{- | A transition chart.
 Stores intermediate transition items redundantly under several indices:

 - by surface length
 - by left slice
 - by right slice
-}
data TChart tr slc v = TChart
  { tcByLength :: !(IM.IntMap (TContents tr slc v))
  , tcByLeft :: !(HM.HashMap (Slice slc) (TContents tr slc v))
  , tcByRight :: !(HM.HashMap (Slice slc) (TContents tr slc v))
  }
  deriving (Show, Generic, NFData)

-- | Returns an empty transition chart.
tcEmpty :: TChart tr slc v
tcEmpty = TChart IM.empty HM.empty HM.empty

-- TODO: there might be room for improvement here

{- | Insert a new transition item into the transition chart.
 If the item's transition already exists, the existing and new score are "added".
-}
tcInsert :: (Parsable' tr slc v) => TChart tr slc v -> TItem tr slc v -> TChart tr slc v
tcInsert (TChart len left right) (t := v) =
  let new = HM.singleton (t, S.leftSide v, S.rightSide v) v
      len' = IM.insertWith insert (transLen t) new len
      left' = HM.insertWith insert (tLeftSlice t) new left
      right' = HM.insertWith insert (tRightSlice t) new right
   in TChart len' left' right'
 where
  insert = HM.unionWithKey (\_ s1 s2 -> S.addScores s1 s2)

-- | Insert several transition items into the transition chart.
tcMerge
  :: (Foldable t, Parsable' tr slc v)
  => TChart tr slc v
  -> t (TItem tr slc v)
  -> TChart tr slc v
tcMerge = foldl' tcInsert

-- | Helper function for getting transition items from the transition chart.
tcGetAny
  :: (TChart tr slc v -> m)
  -> (TContents tr slc v -> k -> m -> TContents tr slc v)
  -> TChart tr slc v
  -> k
  -> [TItem tr slc v]
tcGetAny field getter chart key =
  fmap mkItem $ HM.toList $ getter HM.empty key $ field chart
 where
  mkItem ((t, _, _), v) = t := v

-- | Returns all transition items with the same length.
tcGetByLength :: TChart tr slc v -> Int -> [TItem tr slc v]
tcGetByLength = tcGetAny tcByLength IM.findWithDefault

-- | Returns all transition items with the same left slice.
tcGetByLeft :: (Ord slc, Hashable slc) => TChart tr slc v -> Slice slc -> [TItem tr slc v]
tcGetByLeft = tcGetAny tcByLeft HM.findWithDefault

-- | Returns all transition items with the same right slice.
tcGetByRight :: (Ord slc, Hashable slc) => TChart tr slc v -> Slice slc -> [TItem tr slc v]
tcGetByRight = tcGetAny tcByRight HM.findWithDefault

-- parsing machinery
-- =================

-- applying evaluators
----------------------
-- TODO: add checks that adjacent transitions and slices match?

-- | Unspreads the two slices of a (middle) transition, if possible.
unspreadMiddle
  :: UnspreadMiddle tr slc h v
  -- ^ the UnspreadMiddle evaluator
  -> TItem tr slc v
  -- ^ the middle transition
  -> [(slc, h, v, TItem tr slc v)]
  -- ^ the top slice, unspread operation,
  -- and middle transition
unspreadMiddle unspreadm im@((Transition l m r _) := _) = do
  il <- maybeToList $ getInner $ sContent l
  ir <- maybeToList $ getInner $ sContent r
  (top, op, val) <- unspreadm (il, m, ir)
  pure (top, op, val, im)

-- | Infers the possible left parent transitions of an unspread.
unspreadLeft
  :: (Show slc, Show tr, R.Semiring v, Show v)
  => UnspreadLeft tr slc h
  -- ^ the UnspreadLeft evaluator
  -> TItem tr slc v
  -- ^ the left child transition
  -> Slice slc
  -- ^ the Vert's top slice and ID
  -> h
  -- ^ the Vert's operation
  -> [TItem tr slc v]
  -- ^ all possible left parent transitions
unspreadLeft unspreadl (tleft@(Transition ll lt lr is2nd) := vleft) top op
  | is2nd = []
  | otherwise = fromMaybe err $ do
      ir <- getInner $ sContent lr
      itop <- getInner $ sContent top
      pure $ mkParent v' <$> unspreadl (lt, ir) itop op
 where
  err =
    error $
      "Illegal left-unspread: left="
        <> show tleft
        <> ", top="
        <> show top
  v' = S.unspreadScoresLeft (sID top) vleft
  mkParent v t = Transition ll t top False := v

-- | Infers the possible right parent transitions of an unspread.
unspreadRight
  :: (R.Semiring v, NFData slc, NFData tr, NFData v, Show tr, Show slc, Show v, Show h)
  => UnspreadRight tr slc h
  -- ^ the UnspreadRight evaluator
  -> Vert tr slc h v
  -- ^ the center 'Vert'
  -> TItem tr slc v
  -- ^ the right child transition
  -> [TItem tr slc v]
  -- ^ all possible right parent transitions
unspreadRight unspreadr vert@(Vert top op val (_ := vm)) tright@((Transition rl rt rr _) := vr) =
  fromMaybe err $ do
    ir <- getInner $ sContent rl
    pure $ force $ mkParent v' <$> unspreadr (ir, rt) ir op
 where
  err =
    error $
      "Illegal right-unspread: vert="
        <> show vert
        <> ", right="
        <> show tright
  v' = S.unspreadScoresRight (sID top) val vm vr
  mkParent v t = Transition top t rr True := v

-- | Infers the possible parent transitions of a split.
unsplit
  :: (R.Semiring v, NFData slc, NFData tr, NFData v, Show v)
  => Unsplit tr slc v
  -- ^ the Unsplit evaluator
  -> TItem tr slc v
  -- ^ the left child transition
  -> TItem tr slc v
  -- ^ the right child transition
  -> [TItem tr slc v]
  -- ^ all possible parent transitions
unsplit mg ((Transition ll lt lr l2nd) := vl) ((Transition _ !rt !rr _) := vr) =
  case getInner $ sContent lr of
    Just m ->
      force $ mkItem <$> mg (sContent ll) lt m rt (sContent rr) splitType
    Nothing -> error "trying to unsplit at a non-content slice"
 where
  splitType
    | l2nd = RightOfTwo
    | isStop (sContent rr) = SingleOfOne
    | otherwise = LeftOfTwo
  mkItem (!top, !op) = Transition ll top rr l2nd := S.unsplitScores op vl vr

-- the parsing main loop
------------------------

-- | parallelized map
pmap :: (NFData b) => (a -> b) -> [a] -> [b]
pmap f = P.withStrategy (P.parList P.rdeepseq) . map f

-- pmap = map

{- | A type alias for  pair of transition chart ('TChart')
 and verticalization chart ('VChart').
-}
type ParseState tr slc h v = (TChart tr slc v, VChart tr slc h v)

{- | Type alias for a monadic parsing operation.
 A function that takes a level and a 'ParseState'
 and produces a monadic parsing action yielding a new state.

 Used to express the main parsing loop
 as well as all substeps that transform the charts on a specific level.
-}
type ParseOp m tr slc h v = Int -> ParseState tr slc h v -> m (ParseState tr slc h v)

-- | A single level iteration of the chart parser.
parseStep
  :: (Parsable tr slc h v)
  => (TChart tr slc v -> VChart tr slc h v -> Int -> IO ())
  -- ^ a logging function that takes charts and level number.
  -> Eval tr tr' slc slc' h v
  -- ^ the grammar's evaluator
  -> ParseOp IO tr slc h v
  -- ^ the parsing operation (from level number and charts to new charts).
parseStep logCharts (Eval eMid eLeft eRight eUnsplit _ _) n charts = do
  uncurry logCharts charts n
  unspreadAllMiddles eMid n charts
    >>= unspreadAllLefts eLeft n
    >>= unspreadAllRights eRight n
    >>= unsplitAll eUnsplit n

-- | Verticalizes all edges of length @n@.
unspreadAllMiddles
  :: (Monad m, Parsable tr slc h v) => UnspreadMiddle tr slc h v -> ParseOp m tr slc h v
unspreadAllMiddles evalMid n (!tchart, !vchart) = do
  let ts = tcGetByLength tchart n
      !newVerts = concat $ pmap (unspreadMiddle evalMid) $!! ts
      vchart' = vcMerge vchart newVerts
  return (tchart, vchart')

-- | Perform all left unspreads where either @l@ or @m@ have length @n@.
unspreadAllLefts
  :: (Monad m, Parsable tr slc h v) => UnspreadLeft tr slc h -> ParseOp m tr slc h v
unspreadAllLefts evalLeft n (!tchart, !vchart) = do
  let
    -- left = n (and middle <= n)
    leftn =
      pmap unleft $!! do
        -- in list monad
        left <- tcGetByLength tchart n
        (top, op) <- vcGetByLeftChild n vchart (tRightSlice $ iItem left)
        pure (left, top, op)

    -- middle = n (and left < n)
    midn =
      pmap unleft $!! do
        -- in list monad
        (top, lslice, op) <- vcGetByLengthLeft vchart n
        left <-
          filter (\item -> transLen (iItem item) < n) $
            tcGetByRight tchart lslice
        pure (left, top, op)

    -- insert new transitions into chart
    tchart' = foldl' tcMerge (foldl' tcMerge tchart leftn) midn
  return (tchart', vchart)
 where
  unleft (left, top, op) = unspreadLeft evalLeft left top op

-- | Perform all right unspreads where either @r@ or @m@ have length @n@
unspreadAllRights
  :: (Monad m, Parsable tr slc h v) => UnspreadRight tr slc h -> ParseOp m tr slc h v
unspreadAllRights evalRight n (!tchart, !vchart) = do
  let
    -- right = n (and middle <= n)
    !rightn =
      force $ pmap (uncurry $ unspreadRight evalRight) $!! do
        -- in list monad
        right <- tcGetByLength tchart n
        vert <- vcGetByRightChild n vchart (tLeftSlice $ iItem right)
        pure (vert, right)

    -- middle = n (and left < n)
    !midn =
      force $ pmap (uncurry $ unspreadRight evalRight) $!! do
        -- in list monad
        vert <- vcGetByLength vchart n
        right <-
          filter (\i -> transLen (iItem i) < n) $
            tcGetByLeft tchart (tRightSlice $ iItem $ vMiddle vert)
        pure (vert, right)

    -- insert new transitions into chart
    !tchart' = foldl' tcMerge (foldl' tcMerge tchart rightn) midn
  return (tchart', vchart)

-- | perform all unsplits where either @l@ or @r@ have length @n@
unsplitAll
  :: forall tr slc h v m
   . (Monad m, Parsable tr slc h v)
  => Unsplit tr slc v
  -> ParseOp m tr slc h v
unsplitAll unsplitter n (!tchart, !vchart) = do
  let !byLen = force $ tcGetByLength tchart n

      -- left = n (and right <= n)
      !leftn =
        pmap (uncurry (unsplit unsplitter)) $!! do
          left <- byLen
          right <-
            filter (\r -> transLen (iItem r) <= n) $
              tcGetByLeft tchart (tRightSlice $ iItem left)
          pure (left, right)

      -- right = n (and left < n)
      !rightn =
        pmap (uncurry (unsplit unsplitter)) $!! do
          right <- byLen
          left <-
            filter (\l -> transLen (iItem l) < n) $
              tcGetByRight tchart (tLeftSlice $ iItem right)
          pure (left, right)

      -- insert new transitions into chart
      !tchart' = foldl' tcMerge (foldl' tcMerge tchart leftn) rightn
  return (tchart', vchart)

-- parsing entry point
----------------------

{- | The main entrypoint to the parser.
 Expects an evaluator for the specific grammar
 and an input path.
 Returns the combined semiring value of all full derivations.
-}
parse
  :: (Parsable tr slc h v)
  => (TChart tr slc v -> Either (VChart tr slc h v) [Slice slc] -> Int -> IO ())
  -- ^ logging function
  -> Eval tr tr' slc slc' h v
  -- ^ the grammar's evaluator
  -> Path slc' tr'
  -- ^ the input path (from first to last slice, excluding 'Start' and 'Stop')
  -> IO v
  -- ^ the semiring value at the top
parse logCharts eval path = do
  logCharts tinit (Right $ pathArounds slicePath) 1
  (tfinal, vfinal) <-
    foldM
      (flip $ parseStep (\t v i -> logCharts t (Left v) i) eval)
      (tinit, vcEmpty len)
      [2 .. len - 1]
  logCharts tfinal (Left vfinal) len
  let goals = tcGetByLength tfinal len
  return $ R.sum $ S.getScoreVal . iScore <$> goals
 where
  wrapPath (Path a e rst) = Path (Inner a) (Just e) $ wrapPath rst
  wrapPath (PathEnd a) = Path (Inner a) Nothing $ PathEnd Stop
  path' = Path Start Nothing $ wrapPath path
  len = pathLen path'
  slicePath =
    mapAroundsWithIndex
      0
      (\i notes -> Slice i (evalSlice eval <$> notes) i i)
      path'
  mkTrans l esurf r =
    mk
      <$> evalUnfreeze
        eval
        (sContent l)
        esurf
        (sContent r)
        (isStop $ sContent r)
   where
    mk (e, v) = Transition l e r False := S.val v
  trans0 = mapBetweens mkTrans slicePath
  tinit = tcMerge tcEmpty $ concat trans0

-- | A logging function that logs the sice of the charts at each level.
logSize
  :: TChart tr1 slc1 v1 -> Either (VChart tr2 slc2 h2 v2) [Slice slc2] -> Int -> IO ()
logSize tc vc n = do
  putStrLn $ "parsing level " <> show n
  putStrLn $ "transitions: " <> show (length $ tcGetByLength tc n)
  let nverts = case vc of
        Left chart -> length $ vcGetByLength chart (n - 1)
        Right lst -> length lst
  putStrLn $ "verts: " <> show nverts

-- | Parse a piece using the 'logSize' logging function.
parseSize :: (Parsable tr slc h v) => Eval tr tr' slc slc' h v -> Path slc' tr' -> IO v
parseSize = parse logSize

-- | A logging function that does nothing.
logNone :: (Applicative f) => p1 -> p2 -> p3 -> f ()
logNone _ _ _ = pure ()

-- | Parse a piece without logging.
parseSilent :: (Parsable tr slc h v) => Eval tr tr' slc slc' h v -> Path slc' tr' -> IO v
parseSilent = parse logNone

-- fancier logging
-- ---------------

-- | Generate TikZ code for a slice.
printTikzSlice :: (Show slc) => Slice slc -> IO ()
printTikzSlice (Slice f sc sid l) = do
  putStrLn $
    "    \\node[slice,align=center] (slice"
      <> show sid
      <> ") at ("
      <> show (fromIntegral (f + l) / 2.0)
      <> ",0) {"
      <> showTex sc
      <> "\\\\ "
      <> show sid
      <> "};"

-- | Generate TikZ code for a verticalization.
printTikzVert neighbors (Vert top@(Slice f c i l) _ _ middle) = do
  let index = f + l
      xpos = fromIntegral (f + l) / 2.0
      ypos = IM.findWithDefault 0 index neighbors
      neighbors' =
        IM.alter
          ( \case
              Just n -> Just (n + 1)
              Nothing -> Just 1
          )
          index
          neighbors
  putStrLn $
    "    \\node[slice,align=center] (slice"
      <> show i
      <> ") at ("
      <> show xpos
      <> ","
      <> show ypos
      <> ") {"
      <> showTex c
      <> "\\\\ ("
      <> show (sID $ tLeftSlice $ iItem middle)
      <> ") - "
      <> show i
      <> " - ("
      <> show (sID $ tRightSlice $ iItem middle)
      <> ")};"
  pure neighbors'

-- | Generate TikZ code for a transition.
printTikzTrans neighbors t@(Transition sl tc sr _) = do
  let tid = "t" <> show (hash t)
      index = sFirst sl + sLast sr
      xpos = fromIntegral index / 2.0
      ypos = IM.findWithDefault 0 index neighbors
      neighbors' =
        IM.alter
          ( \case
              Just n -> Just (n + 1)
              Nothing -> Just 1
          )
          index
          neighbors
  putStrLn $
    "  \\begin{scope}[xshift="
      <> show xpos
      <> "cm,yshift="
      <> show ypos
      <> "cm]"
  putStrLn $
    "    \\node[slice] ("
      <> tid
      <> "left) at (-0.1,0) {"
      <> show (sID sl)
      <> "};"
  putStrLn $
    "    \\node[slice] ("
      <> tid
      <> "right) at (0.1,0) {"
      <> show (sID sr)
      <> "};"
  -- printTikzSlice sl (tid <> "left")  "(-0.2,0)"
  -- printTikzSlice sr (tid <> "right") "(0.2,0)"
  putStrLn $
    "    \\draw[transition] ("
      <> tid
      <> "left) -- ("
      <> tid
      <> "right);"
  putStrLn "  \\end{scope}"
  pure neighbors'

-- | A logging function that emits the state of the chart in TikZ code at every level.
logTikz tc vc n = do
  putStrLn $ "\n% level " <> show n
  let rel =
        if n <= 2
          then ""
          else ",shift={($(0,0 |- scope" <> show (n - 1) <> ".north)+(0,1cm)$)}"
  putStrLn $ "\\begin{scope}[local bounding box=scope" <> show n <> rel <> "]"
  putStrLn "  % verticalizations:"
  case vc of
    Left chart -> foldM_ printTikzVert IM.empty $ vcGetByLength chart (n - 1)
    Right lst -> mapM_ printTikzSlice lst
  putStrLn "\n  % transitions:"
  foldM_ printTikzTrans IM.empty $ iItem <$> tcGetByLength tc n
  putStrLn "\\end{scope}"
