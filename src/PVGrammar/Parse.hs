{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE TypeFamilies #-}

{- | This module contains code that is specific to parsing the protovoice grammar.
 It implements a number of evaluators ('Eval') that can be used with the various parsers.
-}
module PVGrammar.Parse
  ( -- * Generic Parsing

    -- | Evaluators that directly return protovoice operations.
    -- They can be embedded into a semiring using 'mapEvalScore'.
    IsNote
  , protoVoiceEvaluator
  , protoVoiceEvaluatorNoRepSplit

    -- * Parsing Derivations
  , pvDerivUnrestricted
  , pvDerivRightBranch

    -- * Counting Parses
  , pvCountUnrestricted
  , pvCountNoRepSplit
  , pvCountNoRepSplitRightBranch
  , pvCountNoRepSplitRightBranchSplitFirst

    -- * Useful Helpers
  , pvThaw
  ) where

import Common
import PVGrammar

import Musicology.Pitch
  ( Diatonic
  , Interval (..)
  , Notation
  , pc
  , pto
  )

import Control.DeepSeq (NFData)
import Control.Monad (foldM)
import Data.Foldable
  ( foldl'
  , toList
  )
import Data.HashMap.Strict qualified as HM
import Data.HashSet qualified as S
import Data.Hashable (Hashable)
import Data.Kind (Constraint, Type)
import Data.List qualified as L
import Data.Map.Strict qualified as M
import Data.Maybe
  ( catMaybes
  , mapMaybe
  , maybeToList
  )
import GHC.Generics (Generic)
import Internal.MultiSet qualified as MS
import Musicology.Core
  ( HasPitch (..)
  , Pitch
  , Pitched (..)
  , isStep
  )

-- helper type: Either for terminal and non-terminal edges
-- -------------------------------------------------------

{- | A tag that distinguishes between objects related to terminal and non-terminal edges.
 Like 'Either', but with semantic constructor names to avoid confusion.
-}
data EdgeEither a b
  = -- | marks an terminal edge (or some related object)
    Reg !a
  | -- | marks a non-terminal edge (or some related object)
    Pass !b
  deriving (Eq, Ord, Show, Generic, Hashable, NFData)

-- helper type: enum for possible operations
-- -----------------------------------------

{- | A tag that distinguishes four different types of operations:
  regular split, passing split, left ornament, and right ornament
-}
data Elaboration a b c d
  = -- | marks a terminal split
    EReg !a
  | -- | marks a non-terminal split
    EPass !b
  | -- | marks a right ornament
    ER !c
  | -- | marks a left ornament
    EL !d
  deriving (Eq, Ord, Show, Generic, Hashable, NFData)

{- | Takes a collection of 'Elaboration'
 and splits it into lists for each elaboration type.
-}
partitionElaborations
  :: (Foldable t) => t (Elaboration a b c d) -> ([a], [b], [c], [d])
partitionElaborations = foldl' select ([], [], [], [])
 where
  select (a, b, c, d) (EReg t) = (t : a, b, c, d)
  select (a, b, c, d) (EPass n) = (a, n : b, c, d)
  select (a, b, c, d) (ER l) = (a, b, l : c, d)
  select (a, b, c, d) (EL r) = (a, b, c, r : d)

-- parsing Ornamentations
-- ======================

-- | A constraint alias for note types.
type IsNote :: Type -> Constraint
type IsNote n =
  (HasPitch n, Diatonic (ICOf (IntervalOf n)), Eq (ICOf (IntervalOf n)))

-- | Checks if the middle pitch is between the left and the right pitch.
between
  :: (Eq i, Interval i)
  => Pitch i
  -- ^ left pitch
  -> Pitch i
  -- ^ middle pitch
  -> Pitch i
  -- ^ right pitch
  -> Bool
between pl pm pr =
  pl /= pm && pm /= pr && pl /= pr && dir1 == odir && dir2 == odir
 where
  odir = direction $ pl `pto` pr
  dir1 = direction $ pl `pto` pm
  dir2 = direction $ pm `pto` pr

{- | Attempts to reduce three nodes using an ornamentation operation.
 If succesfull, returns the ornament type and the parent edge,
 which is either a non-terminal edge for passing notes,
 or a terminal edge for all other operations.
-}
findOrnament
  :: (IsNote n)
  => StartStop n
  -> StartStop n
  -> StartStop n
  -> Bool
  -> Bool
  -> Maybe
      ( EdgeEither
          (DoubleOrnament, Edge n)
          (PassingOrnament, InnerEdge n)
      )
findOrnament (Inner l) (Inner m) (Inner r) True True
  | pl == pm && pm == pr = Just $ Reg (FullRepeat, (Inner l, Inner r))
  | pl == pm && so = Just $ Reg (RightRepeatOfLeft, (Inner l, Inner r))
  | pm == pr && so = Just $ Reg (LeftRepeatOfRight, (Inner l, Inner r))
 where
  pl = pc $ pitch l
  pm = pc $ pitch m
  pr = pc $ pitch r
  so = isStep $ pl `pto` pr
findOrnament (Inner l) (Inner m) (Inner r) _ _
  | pl == pr && s1 = Just $ Reg (FullNeighbor, (Inner l, Inner r))
  | s1 && s2 && between pl pm pr = Just $ Pass (PassingMid, (l, r))
 where
  pl = pc $ pitch l
  pm = pc $ pitch m
  pr = pc $ pitch r
  s1 = isStep $ pl `pto` pm
  s2 = isStep $ pm `pto` pr
findOrnament Start (Inner _) Stop _ _ = Just $ Reg (RootNote, (Start, Stop))
findOrnament _ _ _ _ _ = Nothing

{- | Attempts to reduce three notes as a passing motion
 where one of the child edges is a non-terminal edge.

 Since one of the edges is a terminal edge,
 the corresponding outer note could be start/stop symbol, in which case the reduction fails.
 The side with the terminal edge is thus a @StartStop Pitch i@ within a 'Reg',
 while the non-terminal side is a @Pitch i@ within an 'Pass'.
 Exactly one side must be a 'Reg' and the other an 'Pass', otherwise the reduction fails.
-}
findPassing
  :: (IsNote n)
  => EdgeEither (StartStop n) n
  -> n
  -> EdgeEither (StartStop n) n
  -> Maybe (InnerEdge n, PassingOrnament)
findPassing (Reg (Inner l)) m (Pass r)
  | isStep (pl `pto` pm) && between pl pm pr =
      Just ((l, r), PassingLeft)
 where
  pl = pc $ pitch l
  pm = pc $ pitch m
  pr = pc $ pitch r
findPassing (Pass l) m (Reg (Inner r))
  | isStep (pm `pto` pr) && between pl pm pr =
      Just ((l, r), PassingRight)
 where
  pl = pc $ pitch l
  pm = pc $ pitch m
  pr = pc $ pitch r
findPassing _ _ _ = Nothing

findRightOrnament :: (IsNote n) => n -> n -> Maybe RightOrnament
findRightOrnament l m
  | pl == pm = Just RightRepeat
  | isStep (pl `pto` pm) = Just RightNeighbor
  | otherwise = Nothing
 where
  pl = pc $ pitch l
  pm = pc $ pitch m

findLeftOrnament :: (IsNote n) => n -> n -> Maybe LeftOrnament
findLeftOrnament m r
  | pm == pr = Just LeftRepeat
  | isStep (pm `pto` pr) = Just LeftNeighbor
  | otherwise = Nothing
 where
  pm = pc $ pitch m
  pr = pc $ pitch r

-- evaluator interface
-- ===================

{- | The evaluator that represents the proto-voice grammar.
 As scores it returns a representation of each operation.
 These scores do not form a semiring,
 but can be embedded into different semirings using 'mapEvalScore'.
-}
protoVoiceEvaluator
  :: (Foldable t, Foldable t2, Eq n, Ord n, IsNote n, Notation n, Hashable n)
  => Eval (Edges n) (t (Edge n)) (Notes n) (t2 n) (PVLeftmost n)
protoVoiceEvaluator =
  mkLeftmostEval
    pvUnspreadMiddle
    pvUnspreadLeft
    pvUnspreadRight
    pvUnsplit
    (\_ t _ -> [(pvThaw t, FreezeOp)])
    pvSlice

{- | Computes the verticalization (unspread) of a middle transition.
 If the verticalization is admitted, returns the corresponding operation.
-}
pvUnspreadMiddle
  :: (Eq n, Ord n, Hashable n, IsNote n)
  => UnspreadMiddle (Edges n) (Notes n) (Spread n)
pvUnspreadMiddle (Notes nl, edges, Notes nr)
  | any notARepetition (edgesReg edges) = Nothing
  | otherwise = Just (Notes top, op)
 where
  notARepetition (p1, p2) = fmap (pc . pitch) p1 /= fmap (pc . pitch) p2
  top = MS.maxUnion nl nr
  leftMS = nl MS.\\ nr
  left = HM.fromList $ fmap ToLeft <$> MS.toOccurList leftMS
  rightMS = nr MS.\\ nl
  right = HM.fromList $ fmap ToRight <$> MS.toOccurList rightMS
  bothSet =
    S.intersection (MS.toSet nl) (MS.toSet nr)
      `S.difference` (MS.toSet leftMS `S.union` MS.toSet rightMS)
  both = S.foldl' (\m k -> HM.insert k ToBoth m) HM.empty bothSet
  op = SpreadOp (left <> right <> both) edges

{- | Computes all left parent transitions for a verticalization and a left child transition.
 Here, this operation is always admitted and unique,
 so the edges from the child transition are just passed through.
-}
pvUnspreadLeft :: UnspreadLeft (Edges n) (Notes n)
pvUnspreadLeft (el, _) _ = [el]

{- | Computes all right parent transition for a verticalization and a right child transition.
 Here, this operation is always admitted and unique,
 so the edges from the child transition are just passed through.
-}
pvUnspreadRight :: UnspreadRight (Edges n) (Notes n)
pvUnspreadRight (_, er) _ = [er]

{- | Computes all possible unsplits of two child transitions.
 Since transitions here only represent the certain edges,
 'pvUnsplit' must also take into account unelaborated edges,
 which are not present in the child transitions.
-}
pvUnsplit
  :: (IsNote n, Notation n, Ord n, Hashable n)
  => StartStop (Notes n)
  -> Edges n
  -> Notes n
  -> Edges n
  -> StartStop (Notes n)
  -> [(Edges n, Split n)]
pvUnsplit notesl (Edges leftRegs leftPass) (Notes notesm) (Edges rightRegs rightPass) notesr =
  map mkTop combinations
 where
  -- preprocessing of the notes left and right of the unsplit
  !innerL = Reg <$> innerNotes notesl
  !innerR = Reg <$> innerNotes notesr

  -- find all reduction options for every pitch
  !options = noteOptions <$> MS.toOccurList notesm
  noteOptions (note, nocc)
    | nocc < MS.size mandatoryLeft || nocc < MS.size mandatoryRight =
        []
    | otherwise =
        partitionElaborations
          <$> enumerateOptions mandatoryLeft mandatoryRight nocc
   where
    -- compute the mandatory edges for the current pitch:
    mleftRegs = S.map (Reg . fst) $ S.filter ((== Inner note) . snd) leftRegs
    mleftPass = MS.map (Pass . fst) $ MS.filter ((== note) . snd) leftPass
    mrightRegs = S.map (Reg . snd) $ S.filter ((== Inner note) . fst) rightRegs
    mrightPass = MS.map (Pass . snd) $ MS.filter ((== note) . fst) rightPass
    mandatoryLeft = MS.fromSet mleftRegs <> mleftPass
    mandatoryRight = MS.fromSet mrightRegs <> mrightPass

    -- the possible reductions of a (multiple) pitch are enumerated in three stages:

    -- stage 1: consume all mandatory edges on the left
    enumerateOptions ml mr n = do
      (mr', n', acc) <- MS.foldM goL (mr, n, []) ml
      (n'', acc') <- MS.foldM goR (n', acc) mr'
      goFree freeOptions n'' acc'
    goL (_, 0, _) _ = []
    goL (mr, n, acc) l = do
      (new, mr') <- pickLeft n l mr
      pure (mr', n - 1, new : acc)
    -- combine a mandatory left with a mandatory right or free right edge
    pickLeft n l mr
      | n > MS.size mr = mand <> opt <> single
      | otherwise = mand
     where
      mand = do
        r <- MS.distinctElems mr
        red <- maybeToList $ tryReduction True True l note r
        pure (red, MS.delete r mr)
      -- TODO: remove mr options here?
      tryOpt r = tryReduction True (r `S.member` mrightRegs) l note r
      opt = (,mr) <$> mapMaybe tryOpt innerR
      single = fmap (,mr) $ maybeToList $ tryLeftReduction note l

    -- stage 2: consume all remaining mandatory edges on the right
    goR (0, _) _ = []
    goR (n, acc) r = do
      new <- pickRight r
      pure (n - 1, new : acc)
    -- combine mandatory right with free left edge
    pickRight r = opt <> single
     where
      tryOpt l = tryReduction (l `S.member` mleftRegs) True l note r
      opt = mapMaybe tryOpt innerL
      single = maybeToList $ tryRightReduction note r

    -- stage 3: explain all remaining notes through a combination of unknown edges
    goFree _ 0 acc = pure acc
    goFree [] _ _ = []
    goFree [lastOpt] n acc = pure $ L.replicate n lastOpt <> acc
    goFree (opt : opts) n acc = do
      nopt <- [0 .. n]
      goFree opts (n - nopt) (L.replicate nopt opt <> acc)
    -- list all options for free reduction
    freeOptions = pickFreeBoth <> pickFreeLeft <> pickFreeRight
    -- combine two free edges
    pickFreeBoth = do
      l <- innerL
      r <- innerR
      maybeToList $
        tryReduction (l `S.member` mleftRegs) (r `S.member` mrightRegs) l note r
    -- reduce to left using free edge
    pickFreeLeft = mapMaybe (tryLeftReduction note) innerL
    -- reduce to right using free edge
    pickFreeRight = mapMaybe (tryRightReduction note) innerR

  -- at all stages: try out potential reductions:

  -- two terminal edges: any ornament
  tryReduction lIsUsed rIsUsed (Reg notel) notem (Reg noter) = do
    reduction <- findOrnament notel (Inner notem) noter lIsUsed rIsUsed
    pure $ case reduction of
      (Reg (orn, parent)) -> EReg (parent, (notem, orn))
      (Pass (pass, parent)) -> EPass (parent, (notem, pass))
  -- a non-terminal edge left and a terminal edge right: passing note
  tryReduction _ _ notel@(Pass _) notem noter@(Reg _) = do
    (parent, pass) <- findPassing notel notem noter
    pure $ EPass (parent, (notem, pass))
  -- a terminal edge left and a non-terminal edge right: passing note
  tryReduction _ _ notel@(Reg _) notem noter@(Pass _) = do
    (parent, pass) <- findPassing notel notem noter
    pure $ EPass (parent, (notem, pass))
  -- all other combinations are forbidden
  tryReduction _ _ _ _ _ = Nothing

  -- single reduction to a left parent
  tryLeftReduction notem (Reg (Inner notel)) = do
    orn <- findRightOrnament notel notem
    pure $ ER (notel, (notem, orn))
  tryLeftReduction _ _ = Nothing

  -- single reduction to a right parent
  tryRightReduction notem (Reg (Inner noter)) = do
    orn <- findLeftOrnament notem noter
    pure $ EL (noter, (notem, orn))
  tryRightReduction _ _ = Nothing

  -- compute all possible combinations of reduction options
  !combinations =
    if any L.null options -- check if any note has no options
      then [] -- if yes, then no reduction is possible at all
      else foldM pickOption ([], [], [], []) options -- otherwise, compute all combinations
      -- picks all different options for a single note in the list monad
  pickOption (accReg, accPass, accL, accR) opts = do
    (regs, pass, ls, rs) <- opts
    pure (regs <> accReg, pass <> accPass, ls <> accL, rs <> accR)

  -- convert a combination into a derivation operation:
  -- turn the accumulated information into the format expected from the evaluator
  mkTop (regs, pass, rs, ls) =
    if True -- validate
      then (top, SplitOp tmap ntmap rmap lmap leftRegs rightRegs passL passR)
      else
        error $
          "invalid unsplit:\n  notesl="
            <> show notesl
            <> "\n  notesr="
            <> show notesr
            <> "\n  notesm="
            <> show (Notes notesm)
            <> "\n  left="
            <> show (Edges leftRegs leftPass)
            <> "\n  right="
            <> show (Edges rightRegs rightPass)
            <> "\n  top="
            <> show top
   where
    -- validate =
    --   all ((`L.elem` innerNotes notesl) . fst . fst) regs
    --     && all ((`L.elem` innerNotes notesr) . snd . fst)   regs
    --     && all ((`L.elem` innerNotes notesl) . Inner . fst) rs
    --     && all ((`L.elem` innerNotes notesr) . Inner . fst) ls

    -- collect all operations
    mapify xs = M.fromListWith (<>) $ fmap (: []) <$> xs
    tmap = mapify regs
    ntmap = mapify pass
    lmap = mapify ls
    rmap = mapify rs
    top = Edges (S.fromList (fst <$> regs)) (MS.fromList (fst <$> pass))
    passL = foldr MS.delete leftPass $ mapMaybe leftPassingChild pass
    passR = foldr MS.delete rightPass $ mapMaybe rightPassingChild pass
    leftPassingChild ((l, _r), (m, orn)) =
      if orn == PassingRight then Just (l, m) else Nothing
    rightPassingChild ((_l, r), (m, orn)) =
      if orn == PassingLeft then Just (m, r) else Nothing

-- | Unfreezes a single transition, which may be 'Nothing'.
pvThaw
  :: (Foldable t, Ord n, Hashable n)
  => Maybe (t (Edge n))
  -> Edges n
pvThaw e = Edges (S.fromList $ maybe [] toList e) MS.empty

pvSlice :: (Foldable t, Eq n, Hashable n) => t n -> Notes n
pvSlice = Notes . MS.fromList . toList

-- evaluators in specific semirings
-- ================================

{- | A restricted version of the PV evaluator
 that prohibits split operations in which one of the parent slices is repeated entirely.
-}
protoVoiceEvaluatorNoRepSplit
  :: (Foldable t, Foldable t2, Eq n, Ord n, IsNote n, Notation n, Hashable n)
  => Eval (Edges n) (t (Edge n)) (Notes n) (t2 n) (PVLeftmost n)
protoVoiceEvaluatorNoRepSplit = Eval vm vl vr filterSplit t s
 where
  (Eval vm vl vr mg t s) = protoVoiceEvaluator
  filterSplit l lt mid rt r typ = filter ok $ mg l lt mid rt r typ
  ok (_, LMSplitLeft op) = not $ onlyRepeats op
  ok (_, LMSplitOnly op) = not $ onlyRepeats op
  ok (_, LMSplitRight op) = not $ onlyRepeats op
  ok _ = False
  onlyRepeats (SplitOp regs pass rs ls _ _ _ _) =
    M.null pass && (allRepetitionsLeft || allRepetitionsRight)
   where
    allSinglesRepeat =
      all (check (== RightRepeat)) (M.toList rs)
        && all (check (== LeftRepeat)) (M.toList ls)
    allRepetitionsLeft =
      all (check isRepetitionOnLeft) (M.toList regs) && allSinglesRepeat
    allRepetitionsRight =
      all (check isRepetitionOnRight) (M.toList regs) && allSinglesRepeat
  check fpred (_, os) = all (fpred . snd) os

-- | An evaluator for protovoices that produces values in the 'Derivations' semiring.
pvDerivUnrestricted
  :: (Foldable t, Foldable t2, Eq n, Ord n, IsNote n, Notation n, Hashable n)
  => Eval
      (Edges n)
      (t (Edge n))
      (Notes n)
      (t2 n)
      (Derivations (PVLeftmost n))
pvDerivUnrestricted = mapEvalScore Do protoVoiceEvaluator

{- | An evaluator for protovoices that produces values in the 'Derivations' semiring.

 - Enforces right-branching spreads (see 'rightBranchSpread').
-}
pvDerivRightBranch
  :: (Foldable t, Foldable t2, Eq n, Ord n, IsNote n, Notation n, Hashable n)
  => Eval
      (Merged, (RightBranchSpread, Edges n))
      (t (Edge n))
      ((), ((), Notes n))
      (t2 n)
      (Derivations (PVLeftmost n))
pvDerivRightBranch =
  splitFirst $ rightBranchSpread $ mapEvalScore Do protoVoiceEvaluatorNoRepSplit

-- | An evaluator for protovoices that produces values in the counting semiring.
pvCountUnrestricted
  :: (Foldable t, Foldable t2, Eq n, Ord n, IsNote n, Notation n, Hashable n)
  => Eval (Edges n) (t (Edge n)) (Notes n) (t2 n) Int
pvCountUnrestricted = mapEvalScore (const 1) protoVoiceEvaluator

{- | An evaluator for protovoices that produces values in the counting semiring.

 - Prohibits split operations in which one of the parent slices is repeated entirely (see 'protoVoiceEvaluatorNoRepSplit').
-}
pvCountNoRepSplit
  :: (Foldable t, Foldable t2, Eq n, Ord n, IsNote n, Notation n, Hashable n)
  => Eval (Edges n) (t (Edge n)) (Notes n) (t2 n) Int
pvCountNoRepSplit = mapEvalScore (const 1) protoVoiceEvaluatorNoRepSplit

{- | An evaluator for protovoices that produces values in the counting semiring.

 - Prohibits split operations in which one of the parent slices is repeated entirely (see 'protoVoiceEvaluatorNoRepSplit').
 - Enforces right-branching spreads (see 'rightBranchSpread').
-}
pvCountNoRepSplitRightBranch
  :: (Foldable t, Foldable t2, Eq n, Ord n, IsNote n, Notation n, Hashable n)
  => Eval (RightBranchSpread, Edges n) (t (Edge n)) ((), Notes n) (t2 n) Int
pvCountNoRepSplitRightBranch = rightBranchSpread pvCountNoRepSplit

{- | An evaluator for protovoices that produces values in the counting semiring.

 - Prohibits split operations in which one of the parent slices is repeated entirely (see 'protoVoiceEvaluatorNoRepSplit').
 - Enforces right-branching spreads (see 'rightBranchSpread').
 - Normalizes the order of adjacent split and spread operations to split-before-spread (see 'splitFirst').
-}
pvCountNoRepSplitRightBranchSplitFirst
  :: (Foldable t, Foldable t2, Eq n, Ord n, IsNote n, Notation n, Hashable n)
  => Eval
      (Merged, (RightBranchSpread, Edges n))
      (t (Edge n))
      ((), ((), Notes n))
      (t2 n)
      Int
pvCountNoRepSplitRightBranchSplitFirst = splitFirst pvCountNoRepSplitRightBranch
