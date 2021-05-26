{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ConstraintKinds #-}
module PVGrammar.Parse
  ( pvDerivUnrestricted
  , pvDeriv
  , pvCount''
  , pvCount'
  , pvCount
  )
where

import           Common
import           PVGrammar

import           Musicology.Pitch               ( Notation
                                                , Diatonic
                                                , Interval(..)
                                                , pto
                                                , pc
                                                )

import qualified Data.HashSet                  as S
import qualified Data.List                     as L
import qualified Data.Map.Strict               as M
import qualified Internal.MultiSet             as MS
import           Data.Foldable                  ( toList
                                                , foldl'
                                                )
import           Data.Maybe                     ( catMaybes
                                                , maybeToList
                                                )
import           Control.Monad                  ( foldM )
import           GHC.Generics                   ( Generic )
import           Data.Hashable                  ( Hashable )
import           Control.DeepSeq                ( NFData )
import qualified Data.HashMap.Strict           as HM
import           Musicology.Core                ( HasPitch(..)
                                                , Pitched(..)
                                                , isStep
                                                , Pitch
                                                )

-- helper type: Either for terminal and non-terminal edges
-- -------------------------------------------------------

-- | A tag that distinguishes between objects related to terminal and non-terminal edges.
-- Like 'Either', but with semantic constructor names to avoid confusion.
data EdgeEither a b = T !a  -- ^ marks an terminal edge (or some related object)
                    | NT !b -- ^ marks a non-terminal edge (or some related object)
  deriving (Eq, Ord, Show, Generic, Hashable, NFData)

-- helper type: enum for possible operations
-- -----------------------------------------

partitionElaborations
  :: Foldable t => t (Elaboration a b c d) -> ([a], [b], [c], [d])
partitionElaborations = foldl' select ([], [], [], [])
 where
  select (a, b, c, d) (ET t) = (t : a, b, c, d)
  select (a, b, c, d) (EN n) = (a, n : b, c, d)
  select (a, b, c, d) (ER l) = (a, b, l : c, d)
  select (a, b, c, d) (EL r) = (a, b, c, r : d)

-- parsing Ornamentations
-- ======================

type IsNote n
  = (HasPitch n, Diatonic (ICOf (IntervalOf n)), Eq (ICOf (IntervalOf n)))

-- | Checks if `pm` is between `pl` and `pr`.
between :: (Eq i, Interval i) => Pitch i -> Pitch i -> Pitch i -> Bool
between pl pm pr =
  pl /= pm && pm /= pr && pl /= pr && dir1 == odir && dir2 == odir
 where
  odir = direction $ pl `pto` pr
  dir1 = direction $ pl `pto` pm
  dir2 = direction $ pm `pto` pr

-- | Attempts to reduce three nodes using an ornamentation operation.
-- If succesfull, returns the ornament type and the parent edge,
-- which is either a non-terminal edge for passing notes,
-- or a terminal edge for all other operations.
findOrnament
  :: (IsNote n)
  => StartStop n
  -> StartStop n
  -> StartStop n
  -> Bool
  -> Bool
  -> Maybe
       (EdgeEither (Ornament, Edge n) (Passing, InnerEdge n))
findOrnament (Inner l) (Inner m) (Inner r) True True
  | pl == pm && pm == pr = Just $ T (FullRepeat, (Inner l, Inner r))
  | pl == pm && so       = Just $ T (RightRepeatOfLeft, (Inner l, Inner r))
  | pm == pr && so       = Just $ T (LeftRepeatOfRight, (Inner l, Inner r))
 where
  pl = pc $ pitch l
  pm = pc $ pitch m
  pr = pc $ pitch r
  so = isStep $ pl `pto` pr
findOrnament (Inner l) (Inner m) (Inner r) _ _
  | pl == pr && s1               = Just $ T (FullNeighbor, (Inner l, Inner r))
  | s1 && s2 && between pl pm pr = Just $ NT (PassingMid, (l, r))
 where
  pl = pc $ pitch l
  pm = pc $ pitch m
  pr = pc $ pitch r
  s1 = isStep $ pl `pto` pm
  s2 = isStep $ pm `pto` pr
findOrnament (:⋊) (Inner _) (:⋉) _ _ = Just $ T (RootNote, ((:⋊), (:⋉)))
findOrnament _    _         _    _ _ = Nothing

-- | Attempts to reduce three notes as a passing motion
-- where one of the child edges is a non-terminal edge.
--
-- Since one of the edges is a terminal edge,
-- the corresponding outer note could be start/stop symbol, in which case the reduction fails.
-- The side with the terminal edge is thus a @StartStop Pitch i@ within a 'T',
-- while the non-terminal side is a @Pitch i@ within an 'NT'.
-- Exactly one side must be a 'T' and the other an 'NT', otherwise the reduction fails.
findPassing
  :: (IsNote n)
  => EdgeEither (StartStop n) n
  -> n
  -> EdgeEither (StartStop n) n
  -> Maybe (InnerEdge n, Passing)
findPassing (T (Inner l)) m (NT r) | isStep (pl `pto` pm) && between pl pm pr =
  Just ((l, r), PassingLeft)
 where
  pl = pc $ pitch l
  pm = pc $ pitch m
  pr = pc $ pitch r
findPassing (NT l) m (T (Inner r)) | isStep (pm `pto` pr) && between pl pm pr =
  Just ((l, r), PassingRight)
 where
  pl = pc $ pitch l
  pm = pc $ pitch m
  pr = pc $ pitch r
findPassing _ _ _ = Nothing

findRightOrnament :: (IsNote n) => n -> n -> Maybe RightOrnament
findRightOrnament l m | pl == pm             = Just SingleRightRepeat
                      | isStep (pl `pto` pm) = Just SingleRightNeighbor
                      | otherwise            = Nothing
 where
  pl = pc $ pitch l
  pm = pc $ pitch m

findLeftOrnament :: (IsNote n) => n -> n -> Maybe LeftOrnament
findLeftOrnament m r | pm == pr             = Just SingleLeftRepeat
                     | isStep (pm `pto` pr) = Just SingleLeftNeighbor
                     | otherwise            = Nothing
 where
  pm = pc $ pitch m
  pr = pc $ pitch r

-- evaluator interface
-- ===================

-- | The evaluator that represents the proto-voice grammar.
-- As scores it returns a representation of each operation.
-- These scores do not form a semiring,
-- but can be embedded into different semirings using 'evalMapScores'.
protoVoiceEvaluator
  :: (Foldable t, Foldable t2, Eq n, Ord n, IsNote n, Notation n, Hashable n)
  => Eval (Edges n) (t (Edge n)) (Notes n) (t2 n) (PVLeftMost n)
protoVoiceEvaluator =
  mkLeftmostEval pvVertMiddle pvVertLeft pvVertRight pvMerge pvThaw pvSlice

-- | Computes the verticalization of a middle transition.
-- If the verticalization is admitted, returns the corresponding operation.
pvVertMiddle
  :: (Eq n, Ord n, Hashable n) => VertMiddle (Edges n) (Notes n) (Hori n)
pvVertMiddle (Notes nl, edges, Notes nr)
  | any notARepetition (edgesT edges) = Nothing
  | otherwise                         = Just (Notes top, op)
 where
  notARepetition (p1, p2) = p1 /= p2
  top     = MS.maxUnion nl nr
  leftMS  = nl MS.\\ nr
  left    = HM.fromList $ fmap ToLeft <$> MS.toOccurList leftMS
  rightMS = nr MS.\\ nl
  right   = HM.fromList $ fmap ToRight <$> MS.toOccurList rightMS
  bothSet =
    S.intersection (MS.toSet nl) (MS.toSet nr)
      `S.difference` (MS.toSet leftMS `S.union` MS.toSet rightMS)
  both = S.foldl' (\m k -> HM.insert k ToBoth m) HM.empty bothSet
  op   = HoriOp (left <> right <> both) edges

-- | Computes all left parent transitions for a verticalization and a left child transition.
-- Here, this operation is always admitted and unique,
-- so the edges from the child transition are just passed through.
pvVertLeft :: VertLeft (Edges n) (Notes n)
pvVertLeft (el, _) _ = [el]

-- | Computes all right parent transition for a verticalization and a right child transition.
-- Here, this operation is always admitted and unique,
-- so the edges from the child transition are just passed through.
pvVertRight :: VertRight (Edges n) (Notes n)
pvVertRight (_, er) _ = [er]

-- | Computes all possible merges of two child transitions.
-- Since transitions here only represent the certain edges,
-- 'pvMerge' must also take into account unelaborated edges,
-- which are not present in the child transitions.
pvMerge
  :: (IsNote n, Notation n, Ord n, Hashable n)
  => StartStop (Notes n)
  -> Edges n
  -> Notes n
  -> Edges n
  -> StartStop (Notes n)
  -> [(Edges n, Split n)]
pvMerge notesl (Edges leftTs leftNTs) (Notes notesm) (Edges rightTs rightNTs) notesr
  = map mkTop combinations
 where
  -- preprocessing of the notes left and right of the merge
  !innerL  = T <$> innerNotes notesl
  !innerR  = T <$> innerNotes notesr

  -- find all reduction options for every pitch
  !options = noteOptions <$> MS.toOccurList notesm
  noteOptions (note, nocc)
    | nocc < MS.size mandatoryLeft || nocc < MS.size mandatoryRight
    = []
    | otherwise
    = partitionElaborations
      <$> enumerateOptions mandatoryLeft mandatoryRight nocc
   where
    -- compute the mandatory edges for the current pitch:
    mleftTs        = S.map (T . fst) $ S.filter ((== Inner note) . snd) leftTs
    mleftNTs       = MS.map (NT . fst) $ MS.filter ((== note) . snd) leftNTs
    mrightTs       = S.map (T . snd) $ S.filter ((== Inner note) . fst) rightTs
    mrightNTs      = MS.map (NT . snd) $ MS.filter ((== note) . fst) rightNTs
    mandatoryLeft  = MS.fromSet mleftTs <> mleftNTs
    mandatoryRight = MS.fromSet mrightTs <> mrightNTs

    -- the possible reductions of a (multiple) pitch are enumerated in three stages:

    -- stage 1: consume all mandatory edges on the left
    enumerateOptions ml mr n = do
      (mr', n', acc) <- MS.foldM goL (mr, n, []) ml
      (n'', acc')    <- MS.foldM goR (n', acc) mr'
      goFree freeOptions n'' acc'
    goL (_ , 0, _  ) _ = []
    goL (mr, n, acc) l = do
      (new, mr') <- pickLeft n l mr
      pure (mr', n - 1, new : acc)
    -- combine a mandatory left with a mandatory right or free right edge
    pickLeft n l mr | n > MS.size mr = mand <> opt <> single
                    | otherwise      = mand
     where
      mand = do
        r   <- MS.distinctElems mr
        red <- maybeToList $ tryReduction True True l note r
        pure (red, MS.delete r mr)
      -- TODO: remove mr options here?
      tryOpt r = tryReduction True (r `S.member` mrightTs) l note r
      opt    = fmap (, mr) $ catMaybes $ tryOpt <$> innerR
      single = fmap (, mr) $ maybeToList $ tryLeftReduction note l

    -- stage 2: consume all remaining mandatory edges on the right
    goR (0, _  ) _ = []
    goR (n, acc) r = do
      new <- pickRight r
      pure (n - 1, new : acc)
    -- combine mandatory right with free left edge
    pickRight r = opt <> single
     where
      tryOpt l = tryReduction (l `S.member` mleftTs) True l note r
      opt    = catMaybes $ tryOpt <$> innerL
      single = maybeToList $ tryRightReduction note r

    -- stage 3: explain all remaining notes through a combination of unknown edges
    goFree _            0 acc = pure acc
    goFree []           _ _   = []
    goFree [lastOpt   ] n acc = pure $ L.replicate n lastOpt <> acc
    goFree (opt : opts) n acc = do
      nopt <- [0 .. n]
      goFree opts (n - nopt) (L.replicate nopt opt <> acc)
    -- list all options for free reduction
    freeOptions  = pickFreeBoth <> pickFreeLeft <> pickFreeRight
    -- combine two free edges
    pickFreeBoth = do
      l <- innerL
      r <- innerR
      maybeToList
        $ tryReduction (l `S.member` mleftTs) (r `S.member` mrightTs) l note r
    -- reduce to left using free edge
    pickFreeLeft  = catMaybes $ tryLeftReduction note <$> innerL
    -- reduce to right using free edge
    pickFreeRight = catMaybes $ tryRightReduction note <$> innerR

  -- at all stages: try out potential reductions:

  -- two terminal edges: any ornament
  tryReduction lIsUsed rIsUsed (T notel) notem (T noter) = do
    reduction <- findOrnament notel (Inner notem) noter lIsUsed rIsUsed
    pure $ case reduction of
      (T  (orn , parent)) -> ET (parent, (notem, orn))
      (NT (pass, parent)) -> EN (parent, (notem, pass))
  -- a non-terminal edge left and a terminal edge right: passing note
  tryReduction _ _ notel@(NT _) notem noter@(T _) = do
    (parent, pass) <- findPassing notel notem noter
    pure $ EN (parent, (notem, pass))
  -- a terminal edge left and a non-terminal edge right: passing note
  tryReduction _ _ notel@(T _) notem noter@(NT _) = do
    (parent, pass) <- findPassing notel notem noter
    pure $ EN (parent, (notem, pass))
  -- all other combinations are forbidden
  tryReduction _ _ _ _ _ = Nothing

  -- single reduction to a left parent
  tryLeftReduction notem (T (Inner notel)) = do
    orn <- findRightOrnament notel notem
    pure $ ER (notel, (notem, orn))
  tryLeftReduction _ _ = Nothing

  -- single reduction to a right parent
  tryRightReduction notem (T (Inner noter)) = do
    orn <- findLeftOrnament notem noter
    pure $ EL (noter, (notem, orn))
  tryRightReduction _ _ = Nothing

  -- compute all possible combinations of reduction options
  !combinations = if any L.null options     -- check if any note has no options
    then []                                -- if yes, then no reduction is possible at all
    else foldM pickOption ([], [], [], []) options -- otherwise, compute all combinations
  -- picks all different options for a single note in the list monad
  pickOption (accT, accNT, accL, accR) opts = do
    (ts, nts, ls, rs) <- opts
    pure (ts <> accT, nts <> accNT, ls <> accL, rs <> accR)

  -- convert a combination into a derivation operation:
  -- turn the accumulated information into the format expected from the evaluator
  mkTop (ts, nts, rs, ls) = if True -- validate
    then (top, SplitOp tmap ntmap rmap lmap leftTs rightTs)
    else
      error
      $  "invalid merge:\n  notesl="
      <> show notesl
      <> "\n  notesr="
      <> show notesr
      <> "\n  notesm="
      <> show (Notes notesm)
      <> "\n  left="
      <> show (Edges leftTs leftNTs)
      <> "\n  right="
      <> show (Edges rightTs rightNTs)
      <> "\n  top="
      <> show top
   where
    -- validate =
    --   all ((`L.elem` innerNotes notesl) . fst . fst) ts
    --     && all ((`L.elem` innerNotes notesr) . snd . fst)   ts
    --     && all ((`L.elem` innerNotes notesl) . Inner . fst) rs
    --     && all ((`L.elem` innerNotes notesr) . Inner . fst) ls

    -- collect all operations
    mapify xs = M.fromListWith (<>) $ fmap (: []) <$> xs
    tmap  = mapify ts
    ntmap = mapify nts
    lmap  = mapify ls
    rmap  = mapify rs
    top   = Edges (S.fromList (fst <$> ts)) (MS.fromList (fst <$> nts))

-- | Computes all potential ways a surface transition could have been frozen.
-- In this grammar, this operation is unique and just turns ties into edges.
pvThaw
  :: (Foldable t, Ord n, Hashable n)
  => StartStop (Notes n)
  -> Maybe (t (Edge n))
  -> StartStop (Notes n)
  -> [(Edges n, Freeze)]
pvThaw _ e _ = [(Edges (S.fromList $ maybe [] toList e) MS.empty, FreezeOp)]

pvSlice :: (Foldable t, Eq n, Hashable n) => t n -> Notes n
pvSlice = Notes . MS.fromList . toList

-- evaluators in specific semirings
-- ================================

protoVoiceEvaluator'
  :: (Foldable t, Foldable t2, Eq n, Ord n, IsNote n, Notation n, Hashable n)
  => Eval (Edges n) (t (Edge n)) (Notes n) (t2 n) (PVLeftMost n)
protoVoiceEvaluator' = Eval vm vl vr filterSplit t s
 where
  (Eval vm vl vr mg t s) = protoVoiceEvaluator
  filterSplit l lt mid rt r typ = filter ok $ mg l lt mid rt r typ
  ok (_, LMSplitLeft op ) = not $ onlyRepeats op
  ok (_, LMSplitOnly op ) = not $ onlyRepeats op
  ok (_, LMSplitRight op) = not $ onlyRepeats op
  ok _                    = False
  onlyRepeats (SplitOp ts nts rs ls _ _) =
    M.null nts && (allRepetitionsLeft || allRepetitionsRight)
   where
    allSinglesRepeat = all (check (== SingleRightRepeat)) (M.toList rs)
      && all (check (== SingleLeftRepeat)) (M.toList ls)
    allRepetitionsLeft =
      all (check isRepetitionOnLeft) (M.toList ts) && allSinglesRepeat
    allRepetitionsRight =
      all (check isRepetitionOnRight) (M.toList ts) && allSinglesRepeat
  check fpred (_, os) = all (fpred . snd) os

pvDerivUnrestricted
  :: (Foldable t, Foldable t2, Eq n, Ord n, IsNote n, Notation n, Hashable n)
  => Eval
       (Edges n)
       (t (Edge n))
       (Notes n)
       (t2 n)
       (Derivations (PVLeftMost n))
pvDerivUnrestricted = mapEvalScore Do protoVoiceEvaluator'

pvDeriv
  :: (Foldable t, Foldable t2, Eq n, Ord n, IsNote n, Notation n, Hashable n)
  => Eval
       (Merged, (RightBranchHori, Edges n))
       (t (Edge n))
       ((), ((), Notes n))
       (t2 n)
       (Derivations (PVLeftMost n))
pvDeriv = splitFirst $ rightBranchHori $ mapEvalScore Do protoVoiceEvaluator'

pvCount''
  :: (Foldable t, Foldable t2, Eq n, Ord n, IsNote n, Notation n, Hashable n)
  => Eval (Edges n) (t (Edge n)) (Notes n) (t2 n) Int
pvCount'' = mapEvalScore (const 1) protoVoiceEvaluator'

pvCount'
  :: (Foldable t, Foldable t2, Eq n, Ord n, IsNote n, Notation n, Hashable n)
  => Eval (RightBranchHori, Edges n) (t (Edge n)) ((), Notes n) (t2 n) Int
pvCount' = rightBranchHori pvCount''

pvCount
  :: (Foldable t, Foldable t2, Eq n, Ord n, IsNote n, Notation n, Hashable n)
  => Eval
       (Merged, (RightBranchHori, Edges n))
       (t (Edge n))
       ((), ((), Notes n))
       (t2 n)
       Int
pvCount = splitFirst pvCount'
