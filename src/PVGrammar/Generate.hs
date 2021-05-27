{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE DeriveAnyClass #-}
module PVGrammar.Generate
  ( mkSplit
  , splitT
  , splitNT
  , addToLeft
  , addToRight
  , mkHori
  , horiNote
  , addPassing
  , derivationPlayerPV
  , derivationPlayerOrig
  , applyHori
  , applyHoriOrig
  )
where

import           PVGrammar               hiding ( splitTs
                                                , splitNTs
                                                )
import           Common
import           Display

import           Musicology.Pitch               ( Notation(..) )

import qualified Control.Monad.Writer.Strict   as MW
import           Data.Monoid                    ( Endo(..) )
import qualified Data.Map.Strict               as M
import qualified Internal.MultiSet             as MS
import qualified Data.HashSet                  as S
import           Control.Monad                  ( foldM )
import qualified Data.List                     as L
import           Data.Foldable                  ( toList )
import           GHC.Generics                   ( Generic )
import           Control.DeepSeq                ( NFData )
import           Data.Hashable                  ( Hashable )
import qualified Data.HashMap.Strict           as HM

-- building operations
-- ===================

mkSplit :: MW.Writer (Split n) a -> Split n
mkSplit = MW.execWriter

splitT
  :: (Ord n, Hashable n)
  => StartStop n
  -> StartStop n
  -> n
  -> Ornament
  -> Bool
  -> Bool
  -> MW.Writer (Split n) ()
splitT l r c o kl kr = MW.tell
  $ SplitOp (M.singleton (l, r) [(c, o)]) M.empty M.empty M.empty kls krs
 where
  kls = if kl then S.singleton (l, Inner c) else S.empty
  krs = if kr then S.singleton (Inner c, r) else S.empty

splitNT
  :: (Ord n, Hashable n)
  => n
  -> n
  -> n
  -> Passing
  -> Bool
  -> Bool
  -> MW.Writer (Split n) ()
splitNT l r c o kl kr = MW.tell
  $ SplitOp M.empty (M.singleton (l, r) [(c, o)]) M.empty M.empty kls krs
 where
  kls =
    if o /= PassingRight && kl then S.singleton (Inner l, Inner c) else S.empty
  krs =
    if o /= PassingLeft && kr then S.singleton (Inner c, Inner r) else S.empty

addToLeft
  :: (Ord n, Hashable n)
  => n
  -> n
  -> RightOrnament
  -> Bool
  -> MW.Writer (Split n) ()
addToLeft parent child op keep = MW.tell $ SplitOp
  M.empty
  M.empty
  (M.singleton parent [(child, op)])
  M.empty
  (if keep then S.singleton (Inner parent, Inner child) else S.empty)
  S.empty

addToRight
  :: (Ord n, Hashable n)
  => n
  -> n
  -> LeftOrnament
  -> Bool
  -> MW.Writer (Split n) ()
addToRight parent child op keep = MW.tell $ SplitOp
  M.empty
  M.empty
  M.empty
  (M.singleton parent [(child, op)])
  S.empty
  (if keep then S.singleton (Inner child, Inner parent) else S.empty)


mkHori :: MW.Writer (Endo (Hori n)) () -> Hori n
mkHori actions = appEndo (MW.execWriter actions) emptyHori
  where emptyHori = HoriOp HM.empty $ Edges S.empty MS.empty

horiNote
  :: (Ord n, Hashable n)
  => n
  -> HoriDirection
  -> Bool
  -> MW.Writer (Endo (Hori n)) ()
horiNote pitch dir edge = MW.tell $ Endo h
 where
  h (HoriOp dist (Edges mTs mNTs)) = HoriOp dist' (Edges mTs' mNTs)
   where
    dist' = HM.insert pitch dir dist
    mTs'  = S.union mTs
      $ if edge then S.singleton (Inner pitch, Inner pitch) else S.empty

addPassing :: (Ord n, Hashable n) => n -> n -> MW.Writer (Endo (Hori n)) ()
addPassing l r = MW.tell $ Endo h
 where
  h (HoriOp dist (Edges mTs mNTs)) = HoriOp dist (Edges mTs mNTs')
    where mNTs' = MS.insert (l, r) mNTs

-- applying operations
-- ===================

applySplit
  :: forall n
   . (Ord n, Notation n, Hashable n)
  => Split n
  -> Edges n
  -> Either String (Edges n, Notes n, Edges n)
applySplit inSplit@(SplitOp splitTs splitNTs ls rs kl kr) inTop@(Edges topTs topNTs)
  = do
    notesT                       <- applyTs topTs splitTs
    (notesNT, leftNTs, rightNTs) <- applyNTs topNTs splitNTs
    let notesL = collectNotes ls
        notesR = collectNotes rs
        notes  = MS.unions [notesT, notesNT, notesL, notesR]
    pure (Edges kl leftNTs, Notes notes, Edges kr rightNTs)
 where

  allOps opset = do
    (parent, children) <- M.toList opset
    child              <- children
    pure (parent, child)

  showEdge (p1, p2) = showNotation p1 <> "-" <> showNotation p2
  showEdges ts = "{" <> L.intercalate "," (showEdge <$> toList ts) <> "}"

  applyTs top ops = do
    (top', notes) <- foldM (applyT top) (top, MS.empty) $ allOps ops
    if S.null top'
      then Right notes
      else
        Left $ "did not use all terminal edges, remaining: " <> showEdges top'

  applyT topAll (top, notes) (parent, (note, _))
    | parent `S.member` topAll
    = Right (top', notes')
    | otherwise
    = Left
      $  "used non-existing terminal edge\n  top="
      <> show inTop
      <> "\n  split="
      <> show inSplit
   where
    top'   = S.delete parent top
    notes' = MS.insert note notes

  applyNTs top ops = do
    (top', notes, lNTs, rNTs) <-
      foldM applyNT (top, MS.empty, MS.empty, MS.empty) $ allOps ops
    if MS.null top'
      then Right (notes, lNTs, rNTs)
      else Left $ "did not use all non-terminal edges, remaining: " <> showEdges
        (MS.toList top')

  applyNT (top, notes, lNTs, rNTs) (parent@(pl, pr), (note, pass))
    | parent `MS.member` top
    = Right (top', notes', lNTs', rNTs')
    | otherwise
    = Left
      $  "used non-existing non-terminal edge\n  top="
      <> show inTop
      <> "\n  split="
      <> show inSplit
   where
    top'         = MS.delete parent top
    notes'       = MS.insert note notes
    (newl, newr) = case pass of
      PassingMid   -> (MS.empty, MS.empty)
      PassingLeft  -> (MS.empty, MS.singleton (note, pr))
      PassingRight -> (MS.singleton (pl, note), MS.empty)
    lNTs' = MS.union newl lNTs
    rNTs' = MS.union newr rNTs

  singleChild (_, (note, _)) = note
  collectNotes ops = MS.fromList $ singleChild <$> allOps ops

applyFreeze :: Eq n => Freeze -> Edges n -> Either String (Edges n)
applyFreeze FreezeOp e@(Edges ts nts)
  | not $ MS.null nts  = Left "cannot freeze non-terminal edges"
  | not $ all isRep ts = Left "cannot freeze non-tie edges"
  | otherwise          = Right e
  where isRep (a, b) = a == b

applyHori
  :: forall n
   . (Ord n, Notation n, Hashable n)
  => Hori n
  -> Edges n
  -> Notes n
  -> Edges n
  -> Either String (Edges n, Notes n, Edges n, Notes n, Edges n)
applyHori (HoriOp dist childm) pl (Notes notesm) pr = do
  (notesl, notesr) <- foldM applyDist (MS.empty, MS.empty)
    $ MS.toOccurList notesm
  childl <- fixEdges snd pl notesl
  childr <- fixEdges fst pr notesr
  pure (childl, Notes notesl, childm, Notes notesr, childr)
 where
  applyDist (notesl, notesr) (note, n) = do
    d <-
      maybe (Left $ showNotation note <> " is not distributed") Right
        $ HM.lookup note dist
    case d of
      ToBoth -> pure (MS.insertMany note n notesl, MS.insertMany note n notesr)
      ToLeft i -> if i > n || i <= 0
        then Left "moving more notes than allowed to the right"
        else pure
          (MS.insertMany note n notesl, MS.insertMany note (n - i) notesr)
      ToRight i -> if i > n || i <= 0
        then Left "moving more notes than allowed to the left"
        else pure
          (MS.insertMany note (n - i) notesl, MS.insertMany note n notesr)
  fixEdges
    :: (forall a . (a, a) -> a)
    -> Edges n
    -> MS.MultiSet n
    -> Either String (Edges n)
  fixEdges accessor (Edges ts nts) notesms
    | not $ MS.all ((`S.member` notes) . accessor) nts = Left
      "dropping non-terminal edge in hori"
    | otherwise = pure $ Edges ts' nts
   where
    notes  = MS.toSet notesms
    notesi = S.map Inner notes
    ts'    = S.filter ((`S.member` notesi) . accessor) ts

-- derivation player
-- =================

derivationPlayerPV
  :: (Eq n, Ord n, Notation n, Hashable n)
  => DerivationPlayer (Split n) Freeze (Hori n) (Notes n) (Edges n)
derivationPlayerPV = DerivationPlayer topEdges
                                      topNotes
                                      applySplit
                                      applyFreeze
                                      applyHori
 where
  topEdges (:⋊) (:⋉) = Edges (S.singleton ((:⋊), (:⋉))) MS.empty
  topEdges _    _    = Edges S.empty MS.empty
  topNotes = Notes MS.empty


-- Applying operations that also store the origins of notes

type Orig n = Elaboration (Edge n, Ornament) (InnerEdge n, Passing) (n, RightOrnament) (n, LeftOrnament)

newtype NotesOrig n = NotesOrig [(n, Orig n)]
  deriving (Show, Eq, Ord, Generic, NFData, Hashable)

applySplitOrig
  :: forall n
   . (Ord n, Notation n, Hashable n)
  => Split n
  -> Edges n
  -> Either String ((Edges n), NotesOrig n, (Edges n))
applySplitOrig inSplit@(SplitOp splitTs splitNTs ls rs kl kr) inTop@(Edges topTs topNTs)
  = do
    notesT                       <- applyTs topTs splitTs
    (notesNT, leftNTs, rightNTs) <- applyNTs topNTs splitNTs
    let notesL = collectNotesL ls
        notesR = collectNotesR rs
        notes  = notesT ++ notesNT ++ notesL ++ notesR
    pure (Edges kl leftNTs, NotesOrig notes, Edges kr rightNTs)
 where

  allOps opset = do
    (parent, children) <- M.toList opset
    child              <- children
    pure (parent, child)

  showEdge (p1, p2) = showNotation p1 <> "-" <> showNotation p2
  showEdges ts = "{" <> L.intercalate "," (showEdge <$> toList ts) <> "}"

  applyTs top ops = do
    (top', notes) <- foldM (applyT top) (top, []) $ allOps ops
    if S.null top'
      then Right notes
      else
        Left $ "did not use all terminal edges, remaining: " <> showEdges top'

  applyT topAll (top, notes) (parent, (note, orn))
    | parent `S.member` topAll
    = Right (top', notes')
    | otherwise
    = Left
      $  "used non-existing terminal edge\n  top="
      <> show inTop
      <> "\n  split="
      <> show inSplit
   where
    top'   = S.delete parent top
    notes' = (note, ET (parent, orn)) : notes

  applyNTs top ops = do
    (top', notes, lNTs, rNTs) <-
      foldM applyNT (top, [], MS.empty, MS.empty) $ allOps ops
    if MS.null top'
      then Right (notes, lNTs, rNTs)
      else Left $ "did not use all non-terminal edges, remaining: " <> showEdges
        (MS.toList top')

  applyNT (top, notes, lNTs, rNTs) (parent@(pl, pr), (note, pass))
    | parent `MS.member` top
    = Right (top', notes', lNTs', rNTs')
    | otherwise
    = Left
      $  "used non-existing non-terminal edge\n  top="
      <> show inTop
      <> "\n  split="
      <> show inSplit
   where
    top'         = MS.delete parent top
    notes'       = (note, EN (parent, pass)) : notes
    (newl, newr) = case pass of
      PassingMid   -> (MS.empty, MS.empty)
      PassingLeft  -> (MS.empty, MS.singleton (note, pr))
      PassingRight -> (MS.singleton (pl, note), MS.empty)
    lNTs' = MS.union newl lNTs
    rNTs' = MS.union newr rNTs

  origChildR (parent, (note, orn)) = (note, EL (parent, orn))
  collectNotesR ops = map origChildR $ allOps ops
  origChildL (parent, (note, orn)) = (note, ER (parent, orn))
  collectNotesL ops = map origChildL $ allOps ops

applyFreezeOrig :: Eq n => Freeze -> Edges n -> Either String (Edges n)
applyFreezeOrig FreezeOp e@(Edges ts nts)
  | not $ MS.null nts  = Left "cannot freeze non-terminal edges"
  | not $ all isRep ts = Left "cannot freeze non-tie edges"
  | otherwise          = Right e
  where isRep (a, b) = a == b

applyHoriOrig
  :: forall n
   . (Ord n, Notation n, Hashable n)
  => Hori n
  -> Edges n
  -> NotesOrig n
  -> Edges n
  -> Either String (Edges n, NotesOrig n, Edges n, NotesOrig n, Edges n)
applyHoriOrig (HoriOp dist childm) pl (NotesOrig notes) pr = do
  let notesm = MS.fromList $ map fst notes
  (notesl, notesr) <- foldM applyDist (MS.empty, MS.empty)
    $ MS.toOccurList notesm
  let notesm = MS.fromList $ map fst notes
  childl <- fixEdges snd pl notesl
  childr <- fixEdges fst pr notesr
  pure (childl, NotesOrig $ mapNotes (MS.toList notesl) notes, childm, 
                NotesOrig $ mapNotes (MS.toList notesr) notes, childr)
 where
  applyDist (notesl, notesr) (note, n) = do
    d <-
      maybe (Left $ showNotation note <> " is not distributed") Right
        $ HM.lookup note dist
    case d of
      ToBoth -> pure (MS.insertMany note n notesl, MS.insertMany note n notesr)
      ToLeft i -> if i > n || i <= 0
        then Left "moving more notes than allowed to the right"
        else pure
          (MS.insertMany note n notesl, MS.insertMany note (n - i) notesr)
      ToRight i -> if i > n || i <= 0
        then Left "moving more notes than allowed to the left"
        else pure
          (MS.insertMany note (n - i) notesl, MS.insertMany note n notesr)
  fixEdges
    :: (forall a . (a, a) -> a)
    -> Edges n
    -> MS.MultiSet n
    -> Either String (Edges n)
  fixEdges accessor (Edges ts nts) notesms
    | not $ MS.all ((`S.member` notes) . accessor) nts = Left
      "dropping non-terminal edge in hori"
    | otherwise = pure $ Edges ts' nts
   where
    notes  = MS.toSet notesms
    notesi = S.map Inner notes
    ts'    = S.filter ((`S.member` notesi) . accessor) ts
  mapNotes :: [n] -> [(n, Orig n)] -> [(n,Orig n)]
  mapNotes ns noteOrigs = mapNotes' (L.sort ns)
                                    (L.sort noteOrigs)
  mapNotes' (n:ns) ((n',orig):noteOrigs) = if(n == n') 
			   then (n,orig):mapNotes' ns noteOrigs
                           else mapNotes' (n:ns) noteOrigs
  mapNotes' [] _ = []




-- Derivation player propagating ephemereal edges


derivationPlayerOrig
  :: (Eq n, Ord n, Notation n, Hashable n)
  => DerivationPlayer (Split n) Freeze (Hori n) (NotesOrig n) (Edges n)
derivationPlayerOrig = DerivationPlayer topEdges
                                      topNotes
                                      applySplitOrig
                                      applyFreezeOrig
                                      applyHoriOrig
 where
  topEdges (:⋊) (:⋉) = Edges (S.singleton ((:⋊), (:⋉))) MS.empty
  topEdges _    _    = Edges S.empty MS.empty
  topNotes = NotesOrig []

