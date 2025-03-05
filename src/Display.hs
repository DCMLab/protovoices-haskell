{-# LANGUAGE OverloadedStrings #-}

{- | This module contains functions for visualizing derivations using LaTeX and TikZ.
 These functions are generic over slice and transition contents and are thus somewhat limited.
 For protovoice derivations,
 an alternative to plotting a derivation would be to export an 'Analysis' to JSON
 and view it interactively in the
 [protovoice annotation tool](https://dcmlab.github.io/protovoice-annotation-tool/)
 (not implemented yet).

 Plotting happens in two steps.
 First, a the derivation is "replayed" using a (generic or grammar-specific) "player"
 to construct a 'DerivationGraph',
 which contains all graphical objects and their positions explicitly.
 The 'DerivationGraph' can then be plotted using different backends
 (currently only TikZ, but a diagrams/SVG backed would be useful too).
-}
module Display
  ( DerivationGraph (..)
  , DerivSlice (..)
  , DerivTrans

    -- * Replaying Derivations
  , replayDerivation
  , replayDerivation'
  , replayDerivationFull
  , unfoldDerivation
  , unfoldDerivation'

    -- * Derivation Players
  , DerivationPlayer (..)
  , derivationPlayerUnit
  , derivationPlayerEmpty
  , Empty

    -- * Plotting Derivation Graphs with TikZ
  , tikzDerivationGraph
  , tikzPic
  , tikzStandalone
  , writeGraph
  , writeGraphs
  , viewGraph
  , viewGraphs
  ) where

import Common

import Data.Set qualified as S

import Control.Monad (mzero)
import Control.Monad.State qualified as ST
import Control.Monad.Trans (lift)
import Data.Foldable (foldl')
import Data.List qualified as L
import Data.Map qualified as M
import Data.Text qualified as T
import Data.Text.IO qualified as T
import System.Process (callCommand)

import Data.Bifunctor (bimap)
import Data.String (IsString)
import System.FilePath qualified as FP

-- derivation graphs
-- =================

-- | A slice together with a depth and an ID
data DerivSlice slc = DerivSlice
  { dslDepth :: !Int
  , dslId :: !Int
  , dslContent :: !(StartStop slc)
  }
  deriving (Eq, Ord, Show)

-- | A transition between two 'DerivSlice's.
type DerivTrans slc tr = (DerivSlice slc, tr, DerivSlice slc)

{- | A derivation graph.
 Contains the graphical objects of a derivation plot
 as well as intermediate information that is used during a replay.
-}
data DerivationGraph slc tr = DGraph
  { dgNextId :: !Int
  -- ^ a counter for generating new IDs (used during replay)
  , dgSlices :: !(S.Set (DerivSlice slc))
  -- ^ the positioned slices of the derivation graph
  , dgTransitions :: !(S.Set (DerivTrans slc tr))
  -- ^ the positioned transitionn in the derivation graph
  , dgHoriEdges :: !(S.Set (DerivSlice slc, DerivSlice slc))
  -- ^ the "horizontalization" edges
  -- (connecting the parent slice of a spread to its children)
  , dgOpen :: ![DerivTrans slc tr]
  -- ^ the open transitions of the current surface
  , dgFrozen :: ![DerivTrans slc tr]
  -- ^ the frozen transitions of the current surface in reverse order
  , dgRoot :: ![DerivTrans slc tr]
  -- ^ the root transitions
  }
  deriving (Eq, Ord, Show)

-- | Alias for the type of a monadic action during derivation replay.
type DerivationOp slc tr = ST.StateT (DerivationGraph slc tr) (Either String)

-- | Removes and returns an open transition from the current surface.
popOpen :: DerivationOp slc tr (DerivTrans slc tr)
popOpen = do
  graph <- ST.get
  case dgOpen graph of
    [] -> ST.lift $ Left "popOpen: no transition to pop"
    t : ts -> do
      ST.put graph{dgOpen = ts}
      pure t

-- | Adds a list of new open transitions to the current surface and the derivation graph.
pushOpen :: (Ord slc, Ord tr) => [DerivTrans slc tr] -> DerivationOp slc tr ()
pushOpen newts = do
  graph <- ST.get
  let trans' = S.fromList newts <> dgTransitions graph
      surf' = newts <> dgOpen graph
  ST.put $ graph{dgTransitions = trans', dgOpen = surf'}

-- | Adds a frozen transition to the current surface and the derivation graph.
pushClosed :: (Ord slc, Ord tr) => DerivTrans slc tr -> DerivationOp slc tr ()
pushClosed newt = do
  graph <- ST.get
  let trans' = S.insert newt $ dgTransitions graph
      frozen' = newt : dgFrozen graph
  ST.put $ graph{dgTransitions = trans', dgFrozen = frozen'}

-- | Adds a new slice to the derivation graph.
addSlice :: (Ord slc) => slc -> Int -> DerivationOp slc tr (DerivSlice slc)
addSlice sliceContent depth = do
  graph <- ST.get
  let i = dgNextId graph
      newSlice = DerivSlice depth i (Inner sliceContent)
      slices' = S.insert newSlice $ dgSlices graph
  ST.put $ graph{dgNextId = i + 1, dgSlices = slices'}
  pure newSlice

-- | Adds a new horizontalization edge to the derivation graph.
addHoriEdge :: (Ord slc) => (DerivSlice slc, DerivSlice slc) -> DerivationOp slc tr ()
addHoriEdge edge = do
  graph <- ST.get
  let horis' = S.insert edge $ dgHoriEdges graph
  ST.put $ graph{dgHoriEdges = horis'}

{- | A derivation player.
 Contains functions for replaying derivations of a particular grammar,
 i.e. for deriving child elements from parent elements.
-}
data DerivationPlayer s f h slc tr = DerivationPlayer
  { dpTopTrans :: !tr
  -- ^ the grammars default starting transition for @⋊——⋉@
  , dpSplit :: !(s -> tr -> Either String (tr, slc, tr))
  -- ^ replay a split operation
  , dpFreeze :: !(f -> tr -> Either String tr)
  -- ^ replay a freeze operation
  , dpSpread :: !(h -> tr -> slc -> tr -> Either String (tr, slc, tr, slc, tr))
  -- ^ replay a spread operation
  }

-- | Replays a single derivation step and applies it to the derivation graph.
replayDerivationStep
  :: (Ord slc, Ord tr)
  => DerivationPlayer s f h slc tr
  -- ^ the derivation player
  -> Leftmost s f h
  -- ^ the operation to be applied
  -> DerivationOp slc tr ()
replayDerivationStep player = applyRule
 where
  applyRule (LMSplitLeft s) = do
    (pl, pt, pr) <- popOpen
    (cl, cm, cr) <- lift $ dpSplit player s pt
    sm <- addSlice cm $ max (dslDepth pl) (dslDepth pr) + 1
    pushOpen [(pl, cl, sm), (sm, cr, pr)]
  applyRule (LMSplitOnly s) = applyRule $ LMSplitLeft s
  applyRule (LMSplitRight s) = do
    l <- popOpen
    (pl, pt, pr) <- popOpen
    (cl, cm, cr) <- lift $ dpSplit player s pt
    sm <- addSlice cm $ max (dslDepth pl) (dslDepth pr) + 1
    pushOpen [l, (pl, cl, sm), (sm, cr, pr)]
  applyRule (LMFreezeLeft f) = do
    (pl, pt, pr) <- popOpen
    t <- lift $ dpFreeze player f pt
    pushClosed (pl, t, pr)
  applyRule (LMFreezeOnly f) = applyRule $ LMFreezeLeft f
  applyRule (LMSpread h) = do
    (lpl, lpt, pm) <- popOpen
    (_, rpt, rpr) <- popOpen
    let depth' = max (dslDepth lpl) (max (dslDepth pm) (dslDepth rpr)) + 1
    pmInner <- lift $ getInnerE (dslContent pm)
    (l, lc, m, rc, r) <- lift $ dpSpread player h lpt pmInner rpt
    ls <- addSlice lc depth'
    rs <- addSlice rc depth'
    addHoriEdge (pm, ls)
    addHoriEdge (pm, rs)
    pushOpen [(lpl, l, ls), (ls, m, rs), (rs, r, rpr)]

-- | Creates the initial state of the derivation graph.
initialGraph
  :: (Ord slc, Ord tr)
  => Path tr slc -- DerivationPlayer s f h slc tr
  -> DerivationGraph slc tr
initialGraph topPath =
  DGraph
    (pathLen topPath + 1)
    (S.fromList topSlices)
    (S.fromList top)
    S.empty
    top
    []
    top
 where
  -- collect initial slices (+ Start / Stop)
  topContents = Start : (Inner <$> pathBetweens topPath) <> [Stop]
  -- assign depth=0 and running IDs to initial slices
  topSlices = zipWith (DerivSlice 0) [0 ..] topContents
  top = zip3 topSlices (pathArounds topPath) (tail topSlices)

-- | Replay a derivation from @n@ top-level transitions.
replayDerivation'
  :: (Foldable t, Ord slc, Ord tr)
  => Path tr slc
  -- ^ the starting point of the derivation
  -> DerivationPlayer s f h slc tr
  -- ^ the derivation player
  -> t (Leftmost s f h)
  -- ^ the derivation
  -> Either String (DerivationGraph slc tr)
replayDerivation' topPath player deriv =
  ST.execStateT
    (mapM_ (replayDerivationStep player) deriv)
    (initialGraph topPath)

-- | Replay a derivation from @⋊——⋉@.
replayDerivation
  :: (Foldable t, Ord slc, Ord tr)
  => DerivationPlayer s f h slc tr
  -- ^ the derivation player
  -> t (Leftmost s f h)
  -- ^ the derivation
  -> Either String (DerivationGraph slc tr)
replayDerivation player = replayDerivation' topPath player
 where
  topPath = PathEnd $ dpTopTrans player

{- | Replay a derivation from @⋊——⋉@
 and ensure that the dervation is complete (freezing all transitions).
 Return an error message if not.
-}
replayDerivationFull
  :: (Foldable t, Ord slc, Ord tr)
  => DerivationPlayer s f h slc tr
  -- ^ the derivation player
  -> t (Leftmost s f h)
  -- ^ the derivation
  -> Either String (DerivationGraph slc tr)
replayDerivationFull player deriv = do
  graph <- replayDerivation player deriv
  if L.null $ dgOpen graph
    then Right graph
    else Left "Not all open transitions have been frozen!"

-- TODO: this should work with foldM and a Foldable input:

{- | Replays a derivation from @n@ top-level transitions
 and returns every intermediate derivation graph.
-}
unfoldDerivation'
  :: (Ord slc, Ord tr)
  => Path tr slc
  -- ^ the starting point of the derivation
  -> DerivationPlayer s f h slc tr
  -- ^ the derivation player
  -> [Leftmost s f h]
  -- ^ the derivation
  -> [Either String (DerivationGraph slc tr)]
unfoldDerivation' topPath player = go (initialGraph topPath) []
 where
  go g acc [] = Right g : acc
  go g acc (step : rest) =
    case ST.execStateT (replayDerivationStep player step) g of
      Left err -> Left err : acc
      Right g' -> go g' (Right g : acc) rest

{- | Replays a derivation from @⋊——⋉@
 and returns every intermediate derivation graph.
-}
unfoldDerivation
  :: (Ord slc, Ord tr)
  => DerivationPlayer s f h slc tr
  -- ^ the derivation player
  -> [Leftmost s f h]
  -- ^ the derivation
  -> [Either String (DerivationGraph slc tr)]
unfoldDerivation player = unfoldDerivation' topPath player
 where
  topPath = PathEnd $ dpTopTrans player

{- | A derivation player that uses @()@ for slice and transition contents.
 The actual derivation operations are ignored, so only the outer structure is produced.
-}
derivationPlayerUnit :: DerivationPlayer s f h () ()
derivationPlayerUnit = DerivationPlayer () usplit ufreeze uspread
 where
  usplit _ _ = Right ((), (), ())
  ufreeze _ _ = Right ()
  uspread _ _ _ _ = Right ((), (), (), (), ())

-- | A helper type that is like @()@ but has a 'Show' instance that returns the empty string.
data Empty = Empty
  deriving (Eq, Ord)

instance Show Empty where
  show Empty = ""

{- | A derivation player that uses 'Empty' for slice and transition content.
 The actual derivation operations are ignored, so only the outer structure is produced.
-}
derivationPlayerEmpty :: DerivationPlayer s f h Empty Empty
derivationPlayerEmpty = DerivationPlayer Empty nsplit nfreeze nspread
 where
  nsplit _ _ = Right (Empty, Empty, Empty)
  nfreeze _ _ = Right Empty
  nspread _ _ _ _ = Right (Empty, Empty, Empty, Empty, Empty)

-- plotting derivation graphs
-- ==========================

-- | Convert a derivation graph into a series of TikZ commands.
tikzDerivationGraph
  :: (Eq slc, Eq tr)
  => (slc -> T.Text)
  -- ^ a function for displaying slice contents
  -> (tr -> T.Text)
  -- ^ a function for displaying transitions contents
  -> DerivationGraph slc tr
  -- ^ the derivation graph
  -> T.Text
tikzDerivationGraph showS showT (DGraph _ slices trans horis openTrans frozenTrans _) =
  T.intercalate
    "\n"
    ( (showNode <$> tikzNodes)
        <> (showTrans <$> trans')
        <> (showHori <$> S.toList horis)
    )
 where
  showText :: (Show slc) => slc -> T.Text
  showText = T.pack . show
  -- printing nodes and edges
  showSlice Start = "$\\rtimes$"
  showSlice Stop = "$\\ltimes$"
  showSlice (Inner s) = showS s
  showNode (x, y, i, c) =
    "\\node[slice] (slice"
      <> showText i
      <> ") at ("
      <> showText x
      <> ","
      <> showText (-y)
      <> ") {"
      <> showSlice c
      <> "};"
  showTrans ((nl, e, nr), frozen) =
    "\\draw[transition,"
      <> (if frozen then "terminal" else "non-terminal")
      <> "] (slice"
      <> showText (dslId nl)
      <> ") -- (slice"
      <> showText (dslId nr)
      <> ") node[midway,below,sloped] {"
      <> showT e
      <> "};"
  showHori (p, c) =
    "\\draw[hori] (slice"
      <> showText (dslId p)
      <> ") -- (slice"
      <> showText (dslId c)
      <> ");"
  -- helpers
  leftNode (n, _, _) = n
  rightNode (_, _, n) = n
  -- computing node locations
  nodeChildren =
    M.fromListWith (++) $ bimap dslId ((: []) . dslId) <$> S.toList horis
  surface = reverse frozenTrans <> openTrans
  trans' = (\t -> (t, t `L.elem` frozenTrans)) <$> S.toList trans
  surfaceNodes = fmap dslId $ leftNode (head surface) : fmap rightNode surface
  allNodes = dslId <$> L.sortOn dslDepth (S.toList slices)
  -- compute x locations
  xloc = foldl' findX xlocInit allNodes
   where
    xlocInit = M.fromList $ zip surfaceNodes [0.0 :: Double ..]
    mean xs = sum xs / fromIntegral (length xs)
    -- findX :: M.Map Int Double -> Int -> M.Map Int Double
    findX locs i = case M.lookup i locs of
      Just _ -> locs
      Nothing ->
        let children = nodeChildren M.! i
            childxs = (\c -> findX locs c M.! c) <$> children
            x = mean childxs
         in M.insert i x locs
  tikzNodes = mkNode <$> S.toList slices
   where
    mkNode (DerivSlice depth i content) = (xloc M.! i, depth, i, content)

-- | Wraps TikZ commands in a @tikzpicture@ environment.
tikzPic :: (Semigroup a, IsString a) => a -> a
tikzPic content =
  "\\begin{tikzpicture}\n" <> content <> "\n\\end{tikzpicture}"

{- | Wraps latex code (e.g. a @tikzpicture@) in a complete standalone document.
 This environment includes default styles for slices, transitions, and hori edges.
-}
tikzStandalone
  :: (Semigroup a, IsString a)
  => Bool
  -- ^ a flag for using the the @varwidth@ option of @standalone@ (needed for multiline content)
  -> a
  -- ^ the document content
  -> a
tikzStandalone varwidth content =
  "\\documentclass"
    <> (if varwidth then "[varwidth]" else "")
    <> "{standalone}\n\
       \\\usepackage[svgnames]{xcolor}\n\
       \\\usepackage{tikz}\n\
       \\\usepackage{amssymb}\n\
       \\\begin{document}\n\
       \\\tikzstyle{slice} = [rectangle,draw,fill=WhiteSmoke,semithick,minimum size=0.4cm,inner xsep=0,inner ysep=3pt,align=center]\n\
       \\\tikzstyle{transition} = [line width=2pt,draw=lightgray]\n\
       \\\tikzstyle{non-terminal} = []\n\
       \\\tikzstyle{terminal} = [double]\n\
       \\\tikzstyle{hori} = [lightgray,dashed,line width=2pt]\n\n"
    <> content
    <> "\n\\end{document}"

-- | Write a single derivation graph to a @tex@ file.
writeGraph
  :: (Show slc, Eq slc, Eq tr, Show tr) => FilePath -> DerivationGraph slc tr -> IO ()
writeGraph fn g =
  T.writeFile fn $
    tikzStandalone False $
      tikzPic $
        tikzDerivationGraph
          showTexT
          showTexT
          g

-- | Write a single derivation graph to a @tex@ file and compile the file using @pdflatex@.
viewGraph
  :: (Eq slc, Eq tr, Show slc, Show tr) => FilePath -> DerivationGraph slc tr -> IO ()
viewGraph fn g = do
  writeGraph fn g
  callCommand $ "pdflatex -output-directory=\"" <> FP.takeDirectory fn <> "\" " <> fn

-- | Write a several derivation graphs to a @tex@ file.
writeGraphs
  :: (Show tr, Show slc, Eq slc, Eq tr) => FilePath -> [DerivationGraph slc tr] -> IO ()
writeGraphs fn gs =
  T.writeFile fn $
    tikzStandalone True $
      T.intercalate "\n\n" $
        tikzPic
          . tikzDerivationGraph showTexT showTexT
          <$> gs

-- | Write a several derivation graphs to a @tex@ file and compile the file using @pdflatex@.
viewGraphs
  :: (Show tr, Show slc, Eq slc, Eq tr) => FilePath -> [DerivationGraph slc tr] -> IO ()
viewGraphs fn gs = do
  writeGraphs fn gs
  callCommand $ "pdflatex -output-directory=\"" <> FP.takeDirectory fn <> "\" " <> fn
