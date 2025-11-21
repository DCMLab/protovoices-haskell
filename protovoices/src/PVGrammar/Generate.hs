{-# LANGUAGE TypeFamilies #-}

{- | This module contains functions for the generative aspects of protovoice derivations:

 - manually constructing protovoice operations (see "PVGrammar") using a monadic interface
 - applying ("replaying") these operations.
-}
module PVGrammar.Generate
  ( -- * Manually Constructing Derivations

    -- | The functions in this section can be used
    -- to manually construct individual derivation operations
    -- or in conjunction with the (indexed-)monadic functions in "Common" (see 'Common.buildDerivation')
    -- to manually construct complete derivations.
    -- Each outer-structure operation ('mkSplit', 'mkSpread', 'mkFreeze') enters a writer monad
    -- in which inner-structure operations can be chained to determine the details.
    --
    -- Note that the legality of the operations is not always checked, so be careful!

    -- * Freeze
    mkFreeze

    -- ** Split
  , mkSplit
  , splitRegular
  , splitPassing
  , addFromLeft
  , addFromRight
  , addPassingLeft
  , addPassingRight

    -- ** Spread
  , mkSpread
  , SpreadDir (..)
  , spreadNote
  , addPassing
  , addOctaveRepetition

    -- * Derivation Players

    -- | These players can be used with the replay functions in the "Display" module
    -- to obtain derivation graphs for protovoice derivations.
  , derivationPlayerPV
  , derivationPlayerPVAllEdges

    -- * Applying Operations

    -- | Apply operations to parent objects and get the resulting child objects.
  , applySplit
  , applySplitAllEdges
  , applyFreeze
  , applySpread
  , freezable

    -- * Utility Functions
  , debugPVAnalysis
  , checkDerivation
  ) where

import Common
import Display
import PVGrammar

import Musicology.Pitch (Notation (..))

import Control.Monad (foldM)
import Control.Monad.Writer.Strict qualified as MW
import Data.Bifunctor (bimap)
import Data.Foldable (toList)
import Data.HashMap.Strict qualified as HM
import Data.HashSet qualified as S
import Data.Hashable (Hashable)
import Data.List qualified as L
import Data.Map.Strict qualified as M
import Data.Maybe (catMaybes)
import Data.Monoid (Endo (..))
import Data.Traversable (for)
import Internal.MultiSet qualified as MS
import Lens.Micro qualified as Lens
import Lens.Micro.Extras qualified as Lens
import Musicology.Core qualified as MC
  ( HasPitch (pitch)
  , Pitched (IntervalOf)
  )

-- building operations
-- ===================

{- | Create a freeze operation.
 Can be used together with the 'Common.freeze' action within a monadic derivation.
-}
mkFreeze :: (Hashable n) => [InnerEdge n] -> Freeze n
mkFreeze ties = FreezeOp $ S.fromList $ fmap (\(l, r) -> (Inner l, Inner r)) ties

{- | Create a split operation monadically

 > mkSplit $ do
 >   ... -- internal split actions

 Can be used together with the 'Common.split' action within a monadic derivation.
-}
mkSplit :: MW.Writer (Split n) a -> Split n
mkSplit = MW.execWriter

-- | During a split, split an existing regular edge between two notes.
splitRegular
  :: (Ord n, Hashable n)
  => StartStop (Note n)
  -- ^ left parent
  -> StartStop (Note n)
  -- ^ right parent
  -> Note n
  -- ^ the new child note
  -> DoubleOrnament
  -- ^ the ornament type of the child note
  -> Bool
  -- ^ keep the left child edge (left parent to child)?
  -> Bool
  -- ^ keep the right child edge (child to right parent)?
  -> MW.Writer (Split n) ()
splitRegular l r c o kl kr =
  MW.tell $
    SplitOp
      (M.singleton (l, r) [(c, o)])
      M.empty
      M.empty
      M.empty
      kls
      krs
      MS.empty
      MS.empty
 where
  kls = if kl then S.singleton (l, Inner c) else S.empty
  krs = if kr then S.singleton (Inner c, r) else S.empty

-- | During a split, split an existing passing edge, introducing a new passing note.
splitPassing
  :: (Ord n, Hashable n)
  => Note n
  -- ^ left parent
  -> Note n
  -- ^ right parent
  -> Note n
  -- ^ the new child note
  -> PassingOrnament
  -- ^ the ornament type of the child note
  -> Bool
  -- ^ keep the left child edge (if step)
  -> Bool
  -- ^ keep the right child edge (if step)
  -> MW.Writer (Split n) ()
splitPassing l r c o kl kr =
  MW.tell $
    SplitOp
      M.empty
      (M.singleton (l, r) [(c, o)])
      M.empty
      M.empty
      kls
      krs
      MS.empty
      MS.empty
 where
  kls =
    if o /= PassingRight && kl then S.singleton (Inner l, Inner c) else S.empty
  krs =
    if o /= PassingLeft && kr then S.singleton (Inner c, Inner r) else S.empty

-- | During a split, add a new single-sided ornament to a left parent note.
addFromLeft
  :: (Ord n, Hashable n)
  => Note n
  -- ^ parent (from the left slice)
  -> Note n
  -- ^ the new child note
  -> RightOrnament
  -- ^ the new child note's ornament type
  -> Bool
  -- ^ keep the new edge?
  -> MW.Writer (Split n) ()
addFromLeft parent child op keep =
  MW.tell $
    SplitOp
      M.empty
      M.empty
      (M.singleton parent [(child, op)])
      M.empty
      (if keep then S.singleton (Inner parent, Inner child) else S.empty)
      S.empty
      MS.empty
      MS.empty

-- | During a split, add a new single-sided ornament to a right parent note.
addFromRight
  :: (Ord n, Hashable n)
  => Note n
  -- ^ parent (from the right slice)
  -> Note n
  -- ^ the new child note
  -> LeftOrnament
  -- ^ the new child note's ornament type
  -> Bool
  -- ^ keep the new edge?
  -> MW.Writer (Split n) ()
addFromRight parent child op keep =
  MW.tell $
    SplitOp
      M.empty
      M.empty
      M.empty
      (M.singleton parent [(child, op)])
      S.empty
      (if keep then S.singleton (Inner child, Inner parent) else S.empty)
      MS.empty
      MS.empty

-- | During a split, add a new passing edge between the left parent slice and the child slice.
addPassingLeft
  :: (Ord n, Hashable n)
  => Note n
  -- ^ note from the left parent slice
  -> Note n
  -- ^ note from the child slice
  -> MW.Writer (Split n) ()
addPassingLeft l m = MW.tell $ mempty{passLeft = MS.singleton (l, m)}

-- | During a split, add a new passing edge between the child slice and the right parent slice.
addPassingRight
  :: (Ord n, Hashable n)
  => Note n
  -- ^ note from the child slice
  -> Note n
  -- ^ note from the right parent slice
  -> MW.Writer (Split n) ()
addPassingRight m r = MW.tell $ mempty{passRight = MS.singleton (m, r)}

{- | Create a spread operation monadically

 > mkSpread $ do
 >   ... -- internal spread actions

 Can be used together with the 'Common.spread' action within a monadic derivation.
-}
mkSpread :: MW.Writer (Endo (Spread n)) () -> Spread n
mkSpread actions = appEndo (MW.execWriter actions) emptySpread
 where
  emptySpread = SpreadOp HM.empty $ Edges S.empty MS.empty

-- | A helper type to express the direction in which a note is spread + the child(ren)'s new IDs.
data SpreadDir = ToLeft String | ToRight String | ToBoth String String

-- | During a spread, distribute one of the parent notes to the child slices of a spread.
spreadNote
  :: (Ord n, Hashable n)
  => Note n
  -- ^ the parent note
  -> SpreadDir
  -- ^ the distribution of the note
  -> Bool
  -- ^ introduce a repetition edge (if possible)?
  -> MW.Writer (Endo (Spread n)) ()
spreadNote note dir edge = MW.tell $ Endo h
 where
  h (SpreadOp dist (Edges mRegs mPassings)) = SpreadOp dist' (Edges mRegs' mPassings)
   where
    pitch = notePitch note
    dir' = case dir of
      ToLeft idl -> SpreadLeftChild (Note pitch idl)
      ToRight idr -> SpreadRightChild (Note pitch idr)
      ToBoth idl idr -> SpreadBothChildren (Note pitch idl) (Note pitch idr)
    dist' = HM.insert note dir' dist
    mRegs' =
      S.union mRegs $ case (edge, dir) of
        (True, ToBoth idl idr) -> S.singleton (Inner (Note pitch idl), Inner (Note pitch idr))
        _ -> S.empty

-- | During a spread, add a new passing edge between the child slices of a spread.
addPassing
  :: (Ord n, Hashable n)
  => Note n
  -- ^ the left end of the edge
  -> Note n
  -- ^ the right end of the edge
  -> MW.Writer (Endo (Spread n)) ()
addPassing l r = MW.tell $ Endo h
 where
  h (SpreadOp dist (Edges mRegs mPassings)) = SpreadOp dist (Edges mRegs mPassings')
   where
    mPassings' = MS.insert (l, r) mPassings

{- | During a spread, add a new repetition edge
 between two notes of the same pitch class but from different octaves.
-}
addOctaveRepetition
  :: (Ord n, Hashable n)
  => Note n
  -- ^ the left end of the edge
  -> Note n
  -- ^ the right end of the edge
  -> MW.Writer (Endo (Spread n)) ()
addOctaveRepetition l r = MW.tell $ Endo h
 where
  h (SpreadOp dist (Edges mRegs mPassings)) = SpreadOp dist (Edges mRegs' mPassings)
   where
    mRegs' = S.insert (Inner l, Inner r) mRegs

-- applying operations
-- ===================

-- | Tries to apply a split operation to the parent transition.
applySplit
  :: forall n
   . (Ord n, Notation n, Hashable n)
  => Split n
  -- ^ the split operation
  -> Edges n
  -- ^ the parent transition
  -> Either String (Edges n, Notes n, Edges n)
  -- ^ the resulting child transitions and slice (or an error message).
applySplit inSplit@(SplitOp splitRegs splitPassings ls rs keepl keepr passl passr) inTop@(Edges topRegs topPassings) =
  do
    notesReg <- applyRegs topRegs splitRegs
    (notesPassing, leftPassings, rightPassings) <- applyPassings topPassings splitPassings
    let notesL = collectNotes ls
        notesR = collectNotes rs
        notes = S.unions [notesReg, notesPassing, notesL, notesR]
    pure
      ( Edges keepl (MS.union leftPassings passl)
      , Notes notes
      , Edges keepr (MS.union rightPassings passr)
      )
 where
  allOps opset = do
    (parent, children) <- M.toList opset
    child <- children
    pure (parent, child)

  showEdge (p1, p2) = show p1 <> "-" <> show p2
  showEdges ts = "{" <> L.intercalate "," (showEdge <$> toList ts) <> "}"

  applyRegs top ops = do
    (top', notes) <- foldM (applyReg top) (top, S.empty) $ allOps ops
    if S.null top'
      then Right notes
      else Left $ "did not use all terminal edges, remaining: " <> showEdges top'

  applyReg topAll (top, notes) (parent, (note, _))
    | parent `S.member` topAll =
        Right (top', notes')
    | otherwise =
        Left $
          "used non-existing terminal edge\n  top="
            <> show inTop
            <> "\n  split="
            <> show inSplit
   where
    top' = S.delete parent top
    notes' = S.insert note notes

  applyPassings top ops = do
    (top', notes, lPassings, rPassings) <-
      foldM applyPassing (top, S.empty, MS.empty, MS.empty) $ allOps ops
    if MS.null top'
      then Right (notes, lPassings, rPassings)
      else
        Left $
          "did not use all non-terminal edges, remaining: "
            <> showEdges
              (MS.toList top')

  applyPassing (top, notes, lPassings, rPassings) (parent@(pl, pr), (note, pass))
    | parent `MS.member` top =
        Right (top', notes', lPassings', rPassings')
    | otherwise =
        Left $
          "used non-existing non-terminal edge\n  top="
            <> show inTop
            <> "\n  split="
            <> show inSplit
   where
    top' = MS.delete parent top
    notes' = S.insert note notes
    (newl, newr) = case pass of
      PassingMid -> (MS.empty, MS.empty)
      PassingLeft -> (MS.empty, MS.singleton (note, pr))
      PassingRight -> (MS.singleton (pl, note), MS.empty)
    lPassings' = MS.union newl lPassings
    rPassings' = MS.union newr rPassings

  singleChild (_, (note, _)) = note
  collectNotes ops = S.fromList $ singleChild <$> allOps ops

-- | Indicates whether a transition can be frozen (i.e., doesn't contain non-"tie" edges).
freezable :: (Eq (MC.IntervalOf n), MC.HasPitch n) => Edges n -> Bool
freezable (Edges ts nts) = MS.null nts && all isRep ts
 where
  isRep (a, b) = fmap (MC.pitch . notePitch) a == fmap (MC.pitch . notePitch) b

-- | Tries to apply a freeze operation to a transition.
applyFreeze
  :: (Eq (MC.IntervalOf n), MC.HasPitch n)
  => Freeze n
  -- ^ the freeze operation
  -> Edges n
  -- ^ the unfrozen edge
  -> Either String (Edges n)
  -- ^ the frozen transition
applyFreeze (FreezeOp _ties) e@(Edges ts nts)
  | not $ MS.null nts = Left "cannot freeze non-terminal edges"
  | not $ all isRep ts = Left "cannot freeze non-tie edges"
  | otherwise = Right e
 where
  isRep (a, b) = fmap (MC.pitch . notePitch) a == fmap (MC.pitch . notePitch) b

-- | Tries to apply a spread operation to the parent transitions and slice.
applySpread
  :: forall n
   . (Ord n, Notation n, Hashable n)
  => Spread n
  -- ^ the spread operation
  -> Edges n
  -- ^ the left parent transition
  -> Notes n
  -- ^ the parent slice
  -> Edges n
  -- ^ the right parent transition
  -> Either String (Edges n, Notes n, Edges n, Notes n, Edges n)
  -- ^ the child transitions and slices (or an error message)
applySpread (SpreadOp dist childm) pl (Notes notesm) pr = do
  (notesl, notesr) <-
    foldM applyDist (HM.empty, HM.empty) $
      S.toList notesm
  childl <- fixEdges Lens._2 pl notesl
  childr <- fixEdges Lens._1 pr notesr
  pure (childl, Notes (S.fromList $ HM.elems notesl), childm, Notes (S.fromList $ HM.elems notesr), childr)
 where
  -- apply spread of one parent note, collect children in accumulators
  applyDist (notesl, notesr) note = do
    d <-
      maybe (Left $ show note <> " is not distributed") Right $
        HM.lookup note dist
    case d of
      SpreadLeftChild n -> pure (HM.insert note n notesl, notesr)
      SpreadRightChild n -> pure (notesl, HM.insert note n notesr)
      SpreadBothChildren nl nr -> pure (HM.insert note nl notesl, HM.insert note nr notesr)

  -- replace notes in child edges or drop if the note was moved to the other side
  fixEdges
    :: (forall a. (Lens.Lens (a, a) (a, a) a a))
    -> Edges n
    -> HM.HashMap (Note n) (Note n)
    -> Either String (Edges n)
  fixEdges lens (Edges reg pass) notemap = do
    -- passing edges: can't be dropped, throw error if moved:
    pass' <- for (MS.toList pass) $ \edge ->
      case HM.lookup (Lens.view lens edge) notemap of
        Nothing -> Left "dropping passing edge"
        Just n' -> Right $ Lens.set lens n' edge
    -- regular edges: just drop if note was moved
    reg' <- for (S.toList reg) $ \edge ->
      case Lens.view lens edge of
        Start -> Left "invalid edge containing ⋊ encountered during spread"
        Stop -> Left "invalid edge containing ⋉ encountered during spread"
        Inner n -> Right $
          case HM.lookup n notemap of
            Nothing -> Nothing
            Just n' -> Just $ Lens.set lens (Inner n') edge
    pure $ Edges (S.fromList $ catMaybes reg') (MS.fromList pass')

{- | A variant of 'applySplit' that inserts all protovoice edges into the child transitions,
 even those that are not "kept" (used for further elaboration).
 This is useful when you want to see all relations between notes in the piece.
-}
applySplitAllEdges
  :: forall n
   . (Ord n, Notation n, Hashable n)
  => Split n
  -> Edges n
  -> Either String (Edges n, Notes n, Edges n)
applySplitAllEdges inSplit@(SplitOp splitRegs splitPassings ls rs _ _ passl passr) inTop@(Edges topRegs topPassings) =
  do
    (notesReg, leftRegsReg, rightRegsReg) <- applyRegs topRegs splitRegs
    (notesPassing, leftPassings, rightPassings, leftRegsPass, rightRegsPass) <-
      applyPassings
        topPassings
        splitPassings
    let notesL = collectNotes ls
        notesR = collectNotes rs
        notes = S.unions [notesReg, notesPassing, notesL, notesR]
        leftSingleEdges = (\(p, (c, _)) -> (Inner p, Inner c)) <$> allOps ls
        rightSingleEdges = (\(p, (c, _)) -> (Inner c, Inner p)) <$> allOps rs
        edgesl = leftRegsReg <> leftRegsPass <> S.fromList leftSingleEdges
        edgesr = rightRegsReg <> rightRegsPass <> S.fromList rightSingleEdges
    pure
      ( Edges edgesl (MS.union leftPassings passl)
      , Notes notes
      , Edges edgesr (MS.union rightPassings passr)
      )
 where
  allOps opset = do
    (parent, children) <- M.toList opset
    child <- children
    pure (parent, child)

  showEdge (p1, p2) = show p1 <> "-" <> show p2
  showEdges ts = "{" <> L.intercalate "," (showEdge <$> toList ts) <> "}"

  applyRegs top ops = do
    (notes, edgesl, edgesr) <-
      foldM (applyReg top) (S.empty, S.empty, S.empty) $
        allOps ops
    pure (notes, edgesl, edgesr)

  applyReg topAll (notes, edgesl, edgesr) (parent, (note, _))
    | parent `S.member` topAll =
        Right (notes', edgesl', edgesr')
    | otherwise =
        Left $
          "used non-existing terminal edge\n  top="
            <> show inTop
            <> "\n  split="
            <> show inSplit
   where
    notes' = S.insert note notes
    edgesl' = S.insert (fst parent, Inner note) edgesl
    edgesr' = S.insert (Inner note, snd parent) edgesr

  applyPassings top ops = do
    (top', notes, lPassings, rPassings, lRegs, rRegs) <-
      foldM applyPassing (top, S.empty, MS.empty, MS.empty, S.empty, S.empty) $
        allOps ops
    if MS.null top'
      then Right (notes, lPassings, rPassings, lRegs, rRegs)
      else
        Left $
          "did not use all non-terminal edges, remaining: "
            <> showEdges
              (MS.toList top')

  applyPassing (top, notes, lPassings, rPassings, lRegs, rRegs) (parent@(pl, pr), (note, pass))
    | parent `MS.member` top =
        Right (top', notes', lPassings', rPassings', lRegs', rRegs')
    | otherwise =
        Left $
          "used non-existing non-terminal edge\n  top="
            <> show inTop
            <> "\n  split="
            <> show inSplit
   where
    top' = MS.delete parent top
    notes' = S.insert note notes
    (newlPassing, newrPassing, newlReg, newrReg) = case pass of
      PassingMid ->
        ( MS.empty
        , MS.empty
        , S.singleton (Inner pl, Inner note)
        , S.singleton (Inner note, Inner pr)
        )
      PassingLeft ->
        ( MS.empty
        , MS.singleton (note, pr)
        , S.singleton (Inner pl, Inner note)
        , S.empty
        )
      PassingRight ->
        ( MS.singleton (pl, note)
        , MS.empty
        , S.empty
        , S.singleton (Inner note, Inner pr)
        )
    lPassings' = MS.union newlPassing lPassings
    rPassings' = MS.union newrPassing rPassings
    lRegs' = S.union newlReg lRegs
    rRegs' = S.union newrReg rRegs

  singleChild (_, (note, _)) = note
  collectNotes ops = S.fromList $ singleChild <$> allOps ops

{- | A variant of 'applyFreeze' that allows non-"tie" edges in the open transition.
 This is useful in conjunction with 'applySplitAllEdges'
 because the non-tie edges will not be dropped before freezing.
-}
applyFreezeAllEdges (FreezeOp _) e@(Edges _ts nts)
  | not $ MS.null nts = Left "cannot freeze non-terminal edges"
  | otherwise = Right e

-- debugging analyses

{- | A specialized version of 'debugAnalysis' for protovoice derivations.
 Prints the steps and intermediate configurations of a derivation.
-}
debugPVAnalysis
  :: (Notation n, Ord n, Hashable n, MC.HasPitch n, Eq (MC.IntervalOf n))
  => PVAnalysis n
  -> IO (Either String ())
debugPVAnalysis = debugAnalysis applySplit applyFreeze applySpread

-- derivation player
-- =================

{- | A derivation player for protovoices.
 The default version of the PV player drops all edges that are not used later on
 when generating child transitions.
 This behaviour matches the intermediate representation of the parsers,
 which only track edges that are necessary to explain the downstream notes.
 If you want to generate all edges (i.e., all functional relations between notes)
 use 'derivationPlayerPVAllEdges'.
-}
derivationPlayerPV
  :: (Eq n, Ord n, Notation n, Hashable n, Eq (MC.IntervalOf n), MC.HasPitch n)
  => DerivationPlayer (Split n) (Freeze n) (Spread n) (Notes n) (Edges n)
derivationPlayerPV =
  DerivationPlayer
    topTrans
    applySplit
    applyFreeze
    applySpread
 where
  topTrans = Edges (S.singleton (Start, Stop)) MS.empty

{- | A derivation player for protovoices that produces all edges
 that express a functional relation between two notes.
 For a version that only produces "necessary" edges, use 'derivationPlayerPV'.
-}
derivationPlayerPVAllEdges
  :: (Eq n, Ord n, Notation n, Hashable n, Eq (MC.IntervalOf n), MC.HasPitch n)
  => DerivationPlayer (Split n) (Freeze n) (Spread n) (Notes n) (Edges n)
derivationPlayerPVAllEdges =
  DerivationPlayer
    topTrans
    applySplitAllEdges
    applyFreezeAllEdges
    applySpread
 where
  topTrans = Edges (S.singleton (Start, Stop)) MS.empty

{- | Compares the output of a derivation
 with the original piece (as provided to the parser).
 Returns 'True' if the output matches the original
 and 'False' if the output doesn't match or the derivation is invalid.
-}
checkDerivation
  :: ( Ord n
     , Notation n
     , Hashable n
     , Eq (MC.IntervalOf n)
     , MC.HasPitch n
     , Show n
     )
  => [Leftmost (Split n) (Freeze n) (Spread n)]
  -> Path [Note n] [Edge n]
  -> Bool
checkDerivation deriv original =
  case replayDerivation derivationPlayerPV deriv of
    (Left _) -> False
    (Right g) -> do
      let path' = case dgFrozen g of
            (_ : (_, tlast, slast) : rst) -> do
              s <- getInner $ dslContent slast
              foldM foldPath (PathEnd s, tlast) rst
            _ -> Nothing
          orig' =
            bimap
              (Notes . S.fromList)
              (\e -> Edges (S.fromList e) MS.empty)
              original
      case path' of
        Nothing -> False
        Just (result, _) -> result == orig'
 where
  foldPath (pacc, tacc) (_, tnew, snew) = do
    s <- getInner $ dslContent snew
    pure (Path s tacc pacc, tnew)
