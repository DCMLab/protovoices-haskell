{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE TypeFamilies #-}

{- | This module contains code that is specific to parsing the protovoice grammar.
 It implements a number of evaluators ('Eval') that can be used with the various parsers.
-}
module PVGrammar.Parse
  ( -- * Generic Parsing

    -- | Evaluators that directly return protovoice operations.
    -- They can be embedded into a semiring using 'mapEvalScore'.
    IsPitch
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
import Data.Traversable (for)
import GHC.Generics (Generic)
import Internal.MultiSet qualified as MS
import Musicology.Core
  ( HasPitch (..)
  , Pitch
  , Pitched (..)
  , isStep
  )

import Debug.Trace qualified as DT

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
type IsPitch :: Type -> Constraint
type IsPitch n =
  (HasPitch n, Diatonic (ICOf (IntervalOf n)), Eq (ICOf (IntervalOf n)), Eq (IntervalOf n))

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
  :: (IsPitch n)
  => StartStop (Note n)
  -> Note n
  -> StartStop (Note n)
  -> Maybe
      ( EdgeEither
          (DoubleOrnament, Edge n)
          (PassingOrnament, InnerEdge n)
      )
findOrnament (Inner l) m (Inner r)
  | pl == pm && pm == pr = Just $ Reg (FullRepeat, (Inner l, Inner r))
  | pl == pm && so = Just $ Reg (RightRepeatOfLeft, (Inner l, Inner r))
  | pm == pr && so = Just $ Reg (LeftRepeatOfRight, (Inner l, Inner r))
  | pl == pr && s1 = Just $ Reg (FullNeighbor, (Inner l, Inner r))
  | s1 && s2 && between pl pm pr = Just $ Pass (PassingMid, (l, r))
 where
  pl = pc $ pitch $ notePitch l
  pm = pc $ pitch $ notePitch m
  pr = pc $ pitch $ notePitch r
  s1 = isStep $ pl `pto` pm
  s2 = isStep $ pm `pto` pr
  so = isStep $ pl `pto` pr
findOrnament Start _ Stop = Just $ Reg (RootNote, (Start, Stop))
findOrnament _ _ _ = Nothing

{- | Attempts to reduce three notes as a passing motion
 where one of the child edges is a non-terminal edge.

 Since one of the edges is a terminal edge,
 the corresponding outer note could be start/stop symbol, in which case the reduction fails.
 The side with the terminal edge is thus a @StartStop Pitch i@ within a 'Reg',
 while the non-terminal side is a @Pitch i@ within an 'Pass'.
 Exactly one side must be a 'Reg' and the other an 'Pass', otherwise the reduction fails.
-}
findPassing
  :: (IsPitch n)
  => EdgeEither (StartStop (Note n)) (Note n)
  -> Note n
  -> EdgeEither (StartStop (Note n)) (Note n)
  -> Maybe (InnerEdge n, PassingOrnament)
findPassing (Reg (Inner l)) m (Pass r)
  | isStep (pl `pto` pm) && between pl pm pr =
      Just ((l, r), PassingLeft)
 where
  pl = pc $ pitch $ notePitch l
  pm = pc $ pitch $ notePitch m
  pr = pc $ pitch $ notePitch r
findPassing (Pass l) m (Reg (Inner r))
  | isStep (pm `pto` pr) && between pl pm pr =
      Just ((l, r), PassingRight)
 where
  pl = pc $ pitch $ notePitch l
  pm = pc $ pitch $ notePitch m
  pr = pc $ pitch $ notePitch r
findPassing _ _ _ = Nothing

findRightOrnament :: (IsPitch n) => (Note n) -> (Note n) -> Maybe RightOrnament
findRightOrnament l m
  | pl == pm = Just RightRepeat
  | isStep (pl `pto` pm) = Just RightNeighbor
  | otherwise = Nothing
 where
  pl = pc $ pitch $ notePitch l
  pm = pc $ pitch $ notePitch m

findLeftOrnament :: (IsPitch n) => (Note n) -> (Note n) -> Maybe LeftOrnament
findLeftOrnament m r
  | pm == pr = Just LeftRepeat
  | isStep (pm `pto` pr) = Just LeftNeighbor
  | otherwise = Nothing
 where
  pm = pc $ pitch $ notePitch m
  pr = pc $ pitch $ notePitch r

{- | Checks a transition to see if can can in principle be reduced.
If a note is adjacent to several regular edges, it cannot be reduced by a split or spread.
Consequently, if both sides of the transition contain notes adjacent to several edges,
neither can be reduced and the transition is irreducible
-}
edgesAreReducible :: (Hashable n) => Edges n -> Bool
edgesAreReducible (Edges reg _pass) = isFree left || isFree right
 where
  left = fst <$> S.toList reg
  right = snd <$> S.toList reg
  isFree notes = (MS.cardinality $ MS.fromList $ mapMaybe getInner notes) < 2

-- evaluator interface
-- ===================

{- | The evaluator that represents the proto-voice grammar.
 As scores it returns a representation of each operation.
 These scores do not form a semiring,
 but can be embedded into different semirings using 'mapEvalScore'.
-}
protoVoiceEvaluator
  :: (Foldable t, Foldable t2, Eq n, Ord n, IsPitch n, Notation n, Hashable n)
  => Eval (Edges n) (t (Edge n)) (Notes n) (t2 (Note n)) (Spread n) (PVLeftmost n)
protoVoiceEvaluator =
  mkLeftmostEval
    pvUnspreadMiddle
    pvUnspreadLeft
    pvUnspreadRight
    pvUnsplit
    (\_ t _ -> [(pvThaw t, FreezeOp)])
    pvSlice

{- | Computes the possible verticalizations (unspread) of a middle transition.
 The verticalization fails if the middle transition contains regular edges
 that are not repetitions.
 Otherwise, unspread matches notes connected by a repetition edge
 and then lists all possible /maximal matchings/
 between the non-paired notes in the left and right slice.
 The remaining unmatched notes are verticalized as "single children".
-}

-- implementation details: see note [Unspread]
pvUnspreadMiddle
  :: forall n
   . (Eq n, Ord n, Hashable n, IsPitch n, Notation n)
  => UnspreadMiddle (Edges n) (Notes n) (Spread n) (Spread n)
pvUnspreadMiddle (Notes notesl, edges@(Edges regular passing), Notes notesr)
  | any (not . isRepetition) regular = [] -- can't unspread non-repetition edges
  | otherwise = do
      -- List monad
      -- choose a matching for the unpaired notes
      matching <- unpairedMatchings
      -- add to the paired matching
      let pairs = foldl' (\m (l, r) -> insertPair l r m) repPairs matching
          -- find pitches that have not been matched
          matchedl = S.fromList $ fmap fst matching
          matchedr = S.fromList $ fmap fst matching
          leftoverl = S.toList $ unpairedl `S.difference` matchedl
          leftoverr = S.toList $ unpairedr `S.difference` matchedr
          -- create single parents for the leftover notes
          singlel = fmap (\l -> (mkParent1 l, SpreadLeftChild l)) leftoverl
          singler = fmap (\r -> (mkParent1 r, SpreadRightChild r)) leftoverr
          -- combine all mappings
          mappings = HM.union pairs $ HM.fromList (singlel <> singler)
          top = Notes $ HM.keysSet mappings
          op = SpreadOp mappings edges
      pure $ (top, op, op)
 where
  isRepetition (p1, p2) = fmap (pc . pitch . notePitch) p1 == fmap (pc . pitch . notePitch) p2
  mkParent2 (Note p1 i1) (Note p2 i2) = Note p1 (i1 <> "+" <> i1)
  mkParent1 (Note p i) = Note p (i <> "'")
  insertPair l r m = HM.insert (mkParent2 l r) (SpreadBothChildren l r) m

  -- pairs notes at repetition edges and collects the unpaired notes
  pairRep (paired, nl, nr) edge@(ssl, ssr)
    | Inner l <- ssl
    , Inner r <- ssr
    , pitch (notePitch l) == pitch (notePitch r) =
        ( insertPair l r paired
        , S.delete l nl
        , S.delete r nr
        )
    | otherwise = (paired, nl, nr)
  (repPairs, unpairedl, unpairedr) = foldl' pairRep (HM.empty, notesl, notesr) $ regular

  -- collects the pitches of unpaired notes left and right
  unpairedPitches = S.toList $ S.map notePitch unpairedl
  unpairedlList = S.toList unpairedl
  unpairedrList = S.toList unpairedr

  -- splits the unpaired notes by pitch,
  -- then computes all possible matchings for each group.
  -- note that only pitches from one side need to be checked
  -- since pitches on the other side that are not included in a group
  -- cannot be matched in the first place
  pairingGroups :: [[[(Note n, Note n)]]]
  pairingGroups = flip fmap unpairedPitches $ \p ->
    let ls = filter ((== p) . notePitch) unpairedlList
        rs = filter ((== p) . notePitch) unpairedrList
     in matchGroup ls rs

  -- Computes all possible matchings within a pitch group,
  -- i.e. between two sets of notes with the same pitch
  matchGroup :: [Note n] -> [Note n] -> [[(Note n, Note n)]]
  matchGroup ls rs =
    if length ls < length rs
      then allMatches ls rs
      else fmap (fmap $ \(a, b) -> (b, a)) $ allMatches rs ls
   where
    allMatches :: [Note n] -> [Note n] -> [[(Note n, Note n)]]
    allMatches [] _ = pure []
    allMatches (a : as) bs = do
      -- List monad
      b <- bs
      let bs' = filter (/= b) bs
      rest <- allMatches as bs'
      pure $ (a, b) : rest

  -- A list of possible matchings of unpaired notes.
  -- Takes all combintations of matchings per pitch group
  -- and then concatenates the groups within each combination.
  unpairedMatchings :: [[(Note n, Note n)]]
  unpairedMatchings = fmap concat $ cartProd pairingGroups

{-
Note [Unspread]
~~~~~~~~~~~~~~~

Unspread performs the following steps
1. Check if the transition contains non-repeating regular edges.
   If so, reject the unspread (return []).
2. Pair all notes that are connected by a repeating edge.
3. Compute all /maximal matchings/ for the remaining notes.
   This can be modelled as a matching problem in a bipartite graph
   where the notes from the left and right child slice are the two groups of vertices
   and pairs of notes with the same pitch are the candidate edges.
   However, because of this construction, we can make additional assumptions:
   - the graph can be decomposed into separated components (one for each pitch)
   - within each component, any note on the left can be matched with any note on the right
   Accordingly, the maximal matchings for the unpaired notes can be computed like this:
   1. Partition the graph into components, one per pitch.
      It is sufficient to do this using the pitches of one side
      since notes on the other side that are not included this way
      don't have a partner that they could be matched to anyways.
   2. Within each group, find the side with fewer pitches
      and compute all possible matchings with notes on the other side.
      This is implemented in the list monad by going through the notes on the smaller side,
      non-deterministically choosing a partner note from the other side,
      and removing that note from the pool.
      For each group, this results in a list of possiblem matchings.
   3. Find all combintations of group-level matchings across groups,
      i.e. the cartesian product of the matching lists for all groups.
      For each combination, combine the group-level matchings to one complete matching.
4. For each matching found in step 3 (together with the edge-paired notes),
   compute the corresponding verticalization.
   non-matched notes on either side are explained as "single children"
   with their own parent in the output slice.
-}

{- | Computes all left parent transitions for a verticalization and a left child transition.
 Here, this operation is always admitted and unique,
 so the edges from the child transition are just passed through.
-}
pvUnspreadLeft :: (Hashable n) => UnspreadLeft (Edges n) (Notes n) (Spread n)
pvUnspreadLeft (Edges reg pass, _) _ (SpreadOp mapping _) = maybeToList remapped
 where
  mappingList = HM.toList mapping
  inverseByLeft = catMaybes $ fmap (\(p, s) -> (,p) <$> leftSpreadChild s) mappingList
  leftMapping = HM.fromList inverseByLeft
  remapped = do
    -- Maybe
    reg' <- flip traverseSet reg $ \(l, r) -> do
      rn <- getInner r
      rn' <- HM.lookup rn leftMapping
      pure (l, Inner rn')
    pass' <- flip MS.traverse pass $ \(ln, rn) -> do
      rn' <- HM.lookup rn leftMapping
      pure (ln, rn')
    pure $ Edges reg' pass'

{- | Computes all right parent transition for a verticalization and a right child transition.
 Here, this operation is always admitted and unique,
 so the edges from the child transition are just passed through.
-}
pvUnspreadRight :: (Hashable n) => UnspreadRight (Edges n) (Notes n) (Spread n)
pvUnspreadRight (_, Edges reg pass) _ (SpreadOp mapping _) = maybeToList remapped
 where
  mappingList = HM.toList mapping
  inverseByRight = catMaybes $ fmap (\(p, s) -> (,p) <$> rightSpreadChild s) mappingList
  rightMapping = HM.fromList inverseByRight
  remapped = do
    -- Maybe
    reg' <- flip traverseSet reg $ \(l, r) -> do
      ln <- getInner l
      ln' <- HM.lookup ln rightMapping
      pure (l, Inner ln')
    pass' <- flip MS.traverse pass $ \(ln, rn) -> do
      ln' <- HM.lookup ln rightMapping
      pure (ln', rn)
    pure $ Edges reg' pass'

{- | Computes all possible unsplits of two child transitions.
 Since transitions here only represent the certain edges,
 'pvUnsplit' must also take into account unelaborated edges,
 which are not present in the child transitions.
-}
pvUnsplit
  :: forall n
   . (IsPitch n, Notation n, Ord n, Hashable n)
  => StartStop (Notes n)
  -> Edges n
  -> Notes n
  -> Edges n
  -> StartStop (Notes n)
  -> [(Edges n, Split n)]
pvUnsplit notesl (Edges leftRegs leftPass) (Notes notesm) (Edges rightRegs rightPass) notesr = do
  -- List
  -- pick one combination
  reduction <- cartProd reductions
  -- construct split from reduction
  mkTop $ partitionElaborations reduction
 where
  !innerL = innerNotes notesl
  !innerR = innerNotes notesr

  reductions
    :: [ [ Elaboration
            (Edge n, (Note n, DoubleOrnament))
            (InnerEdge n, (Note n, PassingOrnament))
            (Note n, (Note n, RightOrnament))
            (Note n, (Note n, LeftOrnament))
         ]
       ]
  reductions = findReductions <$> S.toList notesm

  -- finds all possible reductions for a single middle note
  findReductions
    :: Note n
    -> [ Elaboration
          (Edge n, (Note n, DoubleOrnament))
          (InnerEdge n, (Note n, PassingOrnament))
          (Note n, (Note n, RightOrnament))
          (Note n, (Note n, LeftOrnament))
       ]
  findReductions note
    -- more than one mandatory edge left or right -> no reduction possible
    | length leftRegParent > 1 || length rightRegParent > 1 = []
    -- case 1: two mandatory parents: must use both
    | [(lparent, _)] <- leftRegParent
    , [(_, rparent)] <- rightRegParent =
        helperDouble lparent note rparent
    -- case 2: mandatory parent left: choose right parent or none
    | [(lparent, _)] <- leftRegParent =
        let double = do
              rparent <- innerR
              helperDouble lparent note rparent
            passing = helperPassingRight lparent note rightPassParents
            single = helperSingleLeft lparent note
         in double <> passing <> single
    -- case 3: mandatory parent right
    | [(_, rparent)] <- rightRegParent =
        let double = do
              lparent <- innerL
              helperDouble lparent note rparent
            passing = helperPassingLeft leftPassParents note rparent
            single = helperSingleRight note rparent
         in double <> passing <> single
    -- case 4: no mandatory parents
    | otherwise =
        let double = do
              lparent <- innerL
              rparent <- innerR
              helperDouble lparent note rparent
            rightPassing = do
              lparent <- innerL
              helperPassingRight lparent note rightPassParents
            leftPassing = do
              rparent <- innerR
              helperPassingLeft leftPassParents note rparent
            leftSingle = do
              lparent <- innerL
              helperSingleLeft lparent note
            rightSingle = do
              rparent <- innerR
              helperSingleRight note rparent
         in double <> leftPassing <> rightPassing <> leftSingle <> rightSingle
   where
    -- mandatory edges left
    leftRegParent = filter (\(_, r) -> r == Inner note) $ S.toList leftRegs
    -- mandatory edges right
    rightRegParent = filter (\(l, _) -> l == Inner note) $ S.toList rightRegs
    -- passing edges left
    leftPassParents = filter (\(_, r) -> r == note) $ MS.toList leftPass
    -- passing edges right
    rightPassParents = filter (\(l, _) -> l == note) $ MS.toList rightPass

  -- helper functions: find specific reductions of a note to certain parents
  -- all of them return a list of reductions of the form (parent, (child, ornamentType))
  -- wrapped in the appropriate Elaboration constructor

  -- reduce with two regular parents
  helperDouble lparent note rparent =
    case findOrnament lparent note rparent of
      Nothing -> []
      Just (Reg (orn, reg)) -> [EReg (reg, (note, orn))]
      Just (Pass (orn, pass)) -> [EPass (pass, (note, orn))]

  -- reduce with a passing edge on the right and a regular parent on the left
  helperPassingRight lparent note rpassing = do
    (_, rparent) <- rpassing
    (pass, orn) <- maybeToList $ findPassing (Reg lparent) note (Pass rparent)
    pure $ EPass (pass, (note, orn))

  -- reduce with a passing edge on the left and a regular parent on the right
  helperPassingLeft lpassing note rparent = do
    (lparent, _) <- lpassing
    (pass, orn) <- maybeToList $ findPassing (Pass lparent) note (Reg rparent)
    pure $ EPass (pass, (note, orn))

  -- reduce with a single parent on the left
  helperSingleLeft lparent note = case lparent of
    Inner lp -> case findRightOrnament lp note of
      Nothing -> []
      Just orn -> [ER (lp, (note, orn))]
    _ -> []

  -- reduce with a single parent on the right
  helperSingleRight note rparent = case rparent of
    Inner rp -> case findLeftOrnament note rp of
      Nothing -> []
      Just orn -> [EL (rp, (note, orn))]
    _ -> []

  -- convert a combination into a derivation operation:
  -- turn the accumulated information into the format expected from the evaluator
  mkTop
    :: ( [(Edge n, (Note n, DoubleOrnament))]
       , [((Note n, Note n), (Note n, PassingOrnament))]
       , [(Note n, (Note n, RightOrnament))]
       , [(Note n, (Note n, LeftOrnament))]
       )
    -> [(Edges n, Split n)]
  mkTop (regs, pass, rs, ls) =
    if edgesAreReducible top
      then pure (top, SplitOp regmap passmap rmap lmap leftRegs rightRegs passL passR)
      else DT.trace "invalid top!" $ []
   where
    -- collect all operations
    mapify xs = M.fromListWith (<>) $ fmap (: []) <$> xs
    regmap = mapify regs
    passmap = mapify pass
    lmap = mapify ls
    rmap = mapify rs
    leftPassingChild ((l, _r), (m, orn)) =
      if orn == PassingRight then Just (l, m) else Nothing
    rightPassingChild ((_l, r), (m, orn)) =
      if orn == PassingLeft then Just (m, r) else Nothing
    passL = foldr MS.delete leftPass $ mapMaybe leftPassingChild pass
    passR = foldr MS.delete rightPass $ mapMaybe rightPassingChild pass
    top = Edges (S.fromList (fst <$> regs)) (MS.fromList (fst <$> pass))

-- old pvUnsplit (no IDs, multisets)
-- pvUnsplit notesl (Edges leftRegs leftPass) (Notes notesm) (Edges rightRegs rightPass) notesr =
--   map mkTop combinations
--  where
--   -- preprocessing of the notes left and right of the unsplit
--   !innerL = Reg <$> innerNotes notesl
--   !innerR = Reg <$> innerNotes notesr

--   -- find all reduction options for every pitch
--   !options = noteOptions <$> MS.toOccurList notesm
--   noteOptions (note, nocc)
--     | nocc < MS.size mandatoryLeft || nocc < MS.size mandatoryRight =
--         []
--     | otherwise =
--         partitionElaborations
--           <$> enumerateOptions mandatoryLeft mandatoryRight nocc
--    where
--     -- compute the mandatory edges for the current pitch:
--     mleftRegs = S.map (Reg . fst) $ S.filter ((== Inner note) . snd) leftRegs
--     mleftPass = MS.map (Pass . fst) $ MS.filter ((== note) . snd) leftPass
--     mrightRegs = S.map (Reg . snd) $ S.filter ((== Inner note) . fst) rightRegs
--     mrightPass = MS.map (Pass . snd) $ MS.filter ((== note) . fst) rightPass
--     mandatoryLeft = MS.fromSet mleftRegs <> mleftPass
--     mandatoryRight = MS.fromSet mrightRegs <> mrightPass

--     -- the possible reductions of a (multiple) pitch are enumerated in three stages:

--     -- stage 1: consume all mandatory edges on the left
--     enumerateOptions ml mr n = do
--       (mr', n', acc) <- MS.foldM goL (mr, n, []) ml
--       (n'', acc') <- MS.foldM goR (n', acc) mr'
--       goFree freeOptions n'' acc'
--     goL (_, 0, _) _ = []
--     goL (mr, n, acc) l = do
--       (new, mr') <- pickLeft n l mr
--       pure (mr', n - 1, new : acc)
--     -- combine a mandatory left with a mandatory right or free right edge
--     pickLeft n l mr
--       | n > MS.size mr = mand <> opt <> single
--       | otherwise = mand
--      where
--       mand = do
--         r <- MS.distinctElems mr
--         red <- maybeToList $ tryReduction True True l note r
--         pure (red, MS.delete r mr)
--       -- TODO: remove mr options here?
--       tryOpt r = tryReduction True (r `S.member` mrightRegs) l note r
--       opt = (,mr) <$> mapMaybe tryOpt innerR
--       single = fmap (,mr) $ maybeToList $ tryLeftReduction note l

--     -- stage 2: consume all remaining mandatory edges on the right
--     goR (0, _) _ = []
--     goR (n, acc) r = do
--       new <- pickRight r
--       pure (n - 1, new : acc)
--     -- combine mandatory right with free left edge
--     pickRight r = opt <> single
--      where
--       tryOpt l = tryReduction (l `S.member` mleftRegs) True l note r
--       opt = mapMaybe tryOpt innerL
--       single = maybeToList $ tryRightReduction note r

--     -- stage 3: explain all remaining notes through a combination of unknown edges
--     goFree _ 0 acc = pure acc
--     goFree [] _ _ = []
--     goFree [lastOpt] n acc = pure $ L.replicate n lastOpt <> acc
--     goFree (opt : opts) n acc = do
--       nopt <- [0 .. n]
--       goFree opts (n - nopt) (L.replicate nopt opt <> acc)
--     -- list all options for free reduction
--     freeOptions = pickFreeBoth <> pickFreeLeft <> pickFreeRight
--     -- combine two free edges
--     pickFreeBoth = do
--       l <- innerL
--       r <- innerR
--       maybeToList $
--         tryReduction (l `S.member` mleftRegs) (r `S.member` mrightRegs) l note r
--     -- reduce to left using free edge
--     pickFreeLeft = mapMaybe (tryLeftReduction note) innerL
--     -- reduce to right using free edge
--     pickFreeRight = mapMaybe (tryRightReduction note) innerR

--   -- at all stages: try out potential reductions:

--   -- two terminal edges: any ornament
--   tryReduction lIsUsed rIsUsed (Reg notel) notem (Reg noter) = do
--     reduction <- findOrnament notel (Inner notem) noter lIsUsed rIsUsed
--     pure $ case reduction of
--       (Reg (orn, parent)) -> EReg (parent, (notem, orn))
--       (Pass (pass, parent)) -> EPass (parent, (notem, pass))
--   -- a non-terminal edge left and a terminal edge right: passing note
--   tryReduction _ _ notel@(Pass _) notem noter@(Reg _) = do
--     (parent, pass) <- findPassing notel notem noter
--     pure $ EPass (parent, (notem, pass))
--   -- a terminal edge left and a non-terminal edge right: passing note
--   tryReduction _ _ notel@(Reg _) notem noter@(Pass _) = do
--     (parent, pass) <- findPassing notel notem noter
--     pure $ EPass (parent, (notem, pass))
--   -- all other combinations are forbidden
--   tryReduction _ _ _ _ _ = Nothing

--   -- single reduction to a left parent
--   tryLeftReduction notem (Reg (Inner notel)) = do
--     orn <- findRightOrnament notel notem
--     pure $ ER (notel, (notem, orn))
--   tryLeftReduction _ _ = Nothing

--   -- single reduction to a right parent
--   tryRightReduction notem (Reg (Inner noter)) = do
--     orn <- findLeftOrnament notem noter
--     pure $ EL (noter, (notem, orn))
--   tryRightReduction _ _ = Nothing

--   -- compute all possible combinations of reduction options
--   !combinations =
--     if any L.null options -- check if any note has no options
--       then [] -- if yes, then no reduction is possible at all
--       else foldM pickOption ([], [], [], []) options -- otherwise, compute all combinations
--       -- picks all different options for a single note in the list monad
--   pickOption (accReg, accPass, accL, accR) opts = do
--     (regs, pass, ls, rs) <- opts
--     pure (regs <> accReg, pass <> accPass, ls <> accL, rs <> accR)

--   -- convert a combination into a derivation operation:
--   -- turn the accumulated information into the format expected from the evaluator
--   mkTop (regs, pass, rs, ls) =
--     if True -- validate
--       then (top, SplitOp tmap ntmap rmap lmap leftRegs rightRegs passL passR)
--       else
--         error $
--           "invalid unsplit:\n  notesl="
--             <> show notesl
--             <> "\n  notesr="
--             <> show notesr
--             <> "\n  notesm="
--             <> show (Notes notesm)
--             <> "\n  left="
--             <> show (Edges leftRegs leftPass)
--             <> "\n  right="
--             <> show (Edges rightRegs rightPass)
--             <> "\n  top="
--             <> show top
--    where
--     -- validate =
--     --   all ((`L.elem` innerNotes notesl) . fst . fst) regs
--     --     && all ((`L.elem` innerNotes notesr) . snd . fst)   regs
--     --     && all ((`L.elem` innerNotes notesl) . Inner . fst) rs
--     --     && all ((`L.elem` innerNotes notesr) . Inner . fst) ls

--     -- collect all operations
--     mapify xs = M.fromListWith (<>) $ fmap (: []) <$> xs
--     tmap = mapify regs
--     ntmap = mapify pass
--     lmap = mapify ls
--     rmap = mapify rs
--     top = Edges (S.fromList (fst <$> regs)) (MS.fromList (fst <$> pass))
--     passL = foldr MS.delete leftPass $ mapMaybe leftPassingChild pass
--     passR = foldr MS.delete rightPass $ mapMaybe rightPassingChild pass
--     leftPassingChild ((l, _r), (m, orn)) =
--       if orn == PassingRight then Just (l, m) else Nothing
--     rightPassingChild ((_l, r), (m, orn)) =
--       if orn == PassingLeft then Just (m, r) else Nothing

-- | Unfreezes a single transition, which may be 'Nothing'.
pvThaw
  :: (Foldable t, Ord n, Hashable n)
  => Maybe (t (Edge n))
  -> Edges n
pvThaw e = Edges (S.fromList $ maybe [] toList e) MS.empty

pvSlice :: (Foldable t, Eq n, Hashable n) => t (Note n) -> Notes n
pvSlice = Notes . S.fromList . toList

-- evaluators in specific semirings
-- ================================

{- | A restricted version of the PV evaluator
 that prohibits split operations in which one of the parent slices is repeated entirely.
-}
protoVoiceEvaluatorNoRepSplit
  :: (Foldable t, Foldable t2, Eq n, Ord n, IsPitch n, Notation n, Hashable n)
  => Eval (Edges n) (t (Edge n)) (Notes n) (t2 (Note n)) (Spread n) (PVLeftmost n)
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
  :: (Foldable t, Foldable t2, Eq n, Ord n, IsPitch n, Notation n, Hashable n)
  => Eval
      (Edges n)
      (t (Edge n))
      (Notes n)
      (t2 (Note n))
      (Spread n)
      (Derivations (PVLeftmost n))
pvDerivUnrestricted = mapEvalScore Do protoVoiceEvaluator

{- | An evaluator for protovoices that produces values in the 'Derivations' semiring.

 - Enforces right-branching spreads (see 'rightBranchSpread').
-}
pvDerivRightBranch
  :: (Foldable t, Foldable t2, Eq n, Ord n, IsPitch n, Notation n, Hashable n)
  => Eval
      (Merged, (RightBranchSpread, Edges n))
      (t (Edge n))
      ((), ((), Notes n))
      (t2 (Note n))
      ((), ((), (Spread n)))
      (Derivations (PVLeftmost n))
pvDerivRightBranch =
  splitFirst $ rightBranchSpread $ mapEvalScore Do protoVoiceEvaluatorNoRepSplit

-- | An evaluator for protovoices that produces values in the counting semiring.
pvCountUnrestricted
  :: (Foldable t, Foldable t2, Eq n, Ord n, IsPitch n, Notation n, Hashable n)
  => Eval (Edges n) (t (Edge n)) (Notes n) (t2 (Note n)) (Spread n) Int
pvCountUnrestricted = mapEvalScore (const 1) protoVoiceEvaluator

{- | An evaluator for protovoices that produces values in the counting semiring.

 - Prohibits split operations in which one of the parent slices is repeated entirely (see 'protoVoiceEvaluatorNoRepSplit').
-}
pvCountNoRepSplit
  :: (Foldable t, Foldable t2, Eq n, Ord n, IsPitch n, Notation n, Hashable n)
  => Eval (Edges n) (t (Edge n)) (Notes n) (t2 (Note n)) (Spread n) Int
pvCountNoRepSplit = mapEvalScore (const 1) protoVoiceEvaluatorNoRepSplit

{- | An evaluator for protovoices that produces values in the counting semiring.

 - Prohibits split operations in which one of the parent slices is repeated entirely (see 'protoVoiceEvaluatorNoRepSplit').
 - Enforces right-branching spreads (see 'rightBranchSpread').
-}
pvCountNoRepSplitRightBranch
  :: (Foldable t, Foldable t2, Eq n, Ord n, IsPitch n, Notation n, Hashable n)
  => Eval (RightBranchSpread, Edges n) (t (Edge n)) ((), Notes n) (t2 (Note n)) ((), Spread n) Int
pvCountNoRepSplitRightBranch = rightBranchSpread pvCountNoRepSplit

{- | An evaluator for protovoices that produces values in the counting semiring.

 - Prohibits split operations in which one of the parent slices is repeated entirely (see 'protoVoiceEvaluatorNoRepSplit').
 - Enforces right-branching spreads (see 'rightBranchSpread').
 - Normalizes the order of adjacent split and spread operations to split-before-spread (see 'splitFirst').
-}
pvCountNoRepSplitRightBranchSplitFirst
  :: (Foldable t, Foldable t2, Eq n, Ord n, IsPitch n, Notation n, Hashable n)
  => Eval
      (Merged, (RightBranchSpread, Edges n))
      (t (Edge n))
      ((), ((), Notes n))
      (t2 (Note n))
      ((), ((), Spread n))
      Int
pvCountNoRepSplitRightBranchSplitFirst = splitFirst pvCountNoRepSplitRightBranch
