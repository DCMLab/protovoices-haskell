{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
module Display where

import           Common

-- import           Diagrams.Prelude        hiding ( Leftmost
--                                                 , (^-^)
--                                                 , (^+^)
--                                                 )
-- import qualified Diagrams.TwoD.Text
-- import           Diagrams.Backend.SVG          as SVG
-- import           Data.Typeable                  ( Typeable )
-- import           Data.Colour

import qualified Data.Set                      as S

import           Control.Monad                  ( mzero )
import           Control.Monad.Trans            ( lift )
import qualified Control.Monad.State           as ST
import qualified Data.List                     as L
import qualified Data.Map                      as M
import           Data.Foldable                  ( foldl' )
import qualified Data.Text                     as T
import qualified Data.Text.IO                  as T
import           System.Process                 ( callCommand )

import           Data.Bifunctor                 ( bimap )
import           Data.String                    ( IsString )

import           GHC.Generics                   ( Generic )
import           Data.Aeson                    ( ToJSON, FromJSON )

-- type PVDiagram b = QDiagram b V2 Double Any

-- derivation graphs
-- =================

type DerivSlice a = (Int, Int, StartStop a)
type DerivTrans a e = (DerivSlice a, e, DerivSlice a)

data DerivationGraph a e = DGraph
  { dgNextId :: Int
  , dgSlices :: S.Set (DerivSlice a)
  , dgTransitions :: S.Set (DerivTrans a e)
  , dgHoriEdges :: S.Set (DerivSlice a, DerivSlice a)
  , dgSurface :: [DerivTrans a e]
  , dgFoot :: [DerivTrans a e]
  , dgRoot :: [DerivTrans a e]
  }
  deriving (Eq, Ord, Show, Generic)

instance (ToJSON a, ToJSON e) => ToJSON (DerivationGraph a e)

type DerivationOp a e = ST.StateT (DerivationGraph a e) (Either String)

popSurface :: DerivationOp a e (DerivTrans a e)
popSurface = do
  graph <- ST.get
  case dgSurface graph of
    []     -> mzero
    t : ts -> do
      ST.put graph { dgSurface = ts }
      pure t

pushOpen :: (Ord a, Ord e) => [DerivTrans a e] -> DerivationOp a e ()
pushOpen newts = do
  graph <- ST.get
  let trans' = S.fromList newts <> dgTransitions graph
      surf'  = newts <> dgSurface graph
  ST.put $ graph { dgTransitions = trans', dgSurface = surf' }

pushClosed :: (Ord a, Ord e) => DerivTrans a e -> DerivationOp a e ()
pushClosed newt = do
  graph <- ST.get
  let trans' = S.insert newt $ dgTransitions graph
      foot'  = newt : dgFoot graph
  ST.put $ graph { dgTransitions = trans', dgFoot = foot' }

addSlice :: Ord a => a -> Int -> DerivationOp a e (DerivSlice a)
addSlice slice depth = do
  graph <- ST.get
  let i        = dgNextId graph
      newSlice = (depth, i, Inner slice)
      slices'  = S.insert newSlice $ dgSlices graph
  ST.put $ graph { dgNextId = i + 1, dgSlices = slices' }
  pure newSlice

addHoriEdge :: Ord a => (DerivSlice a, DerivSlice a) -> DerivationOp a e ()
addHoriEdge edge = do
  graph <- ST.get
  let horis' = S.insert edge $ dgHoriEdges graph
  ST.put $ graph { dgHoriEdges = horis' }

data DerivationPlayer s f h a e = DerivationPlayer
  { dpTopTrans :: StartStop a -> StartStop a -> e
  , dpTopSlice :: a
  , dpSplit :: s -> e -> Either String (e, a, e)
  , dpFreeze :: f -> e -> Either String e
  , dpHorizontalize :: h -> e -> a -> e -> Either String (e, a, e, a, e)
  }

replayDerivationStep
  :: (Ord a, Ord e)
  => DerivationPlayer s f h a e
  -> Common.Leftmost s f h
  -> DerivationOp a e ()
replayDerivationStep player = applyRule
 where
  applyRule (LMSplitLeft s) = do
    (pl, pt, pr) <- popSurface
    (cl, cm, cr) <- lift $ dpSplit player s pt
    let (depthl, _, _) = pl
        (depthr, _, _) = pr
    sm <- addSlice cm $ max depthl depthr + 1
    pushOpen [(pl, cl, sm), (sm, cr, pr)]
  applyRule (LMSplitOnly  s) = applyRule $ LMSplitLeft s
  applyRule (LMSplitRight s) = do
    l            <- popSurface
    (pl, pt, pr) <- popSurface
    (cl, cm, cr) <- lift $ dpSplit player s pt
    let (depthl, _, _) = pl
        (depthr, _, _) = pr
    sm <- addSlice cm $ max depthl depthr + 1
    pushOpen [l, (pl, cl, sm), (sm, cr, pr)]
  applyRule (LMFreezeLeft f) = do
    (pl, pt, pr) <- popSurface
    t            <- lift $ dpFreeze player f pt
    pushClosed (pl, t, pr)
  applyRule (LMFreezeOnly    f) = applyRule $ LMFreezeLeft f
  applyRule (LMHorizontalize h) = do
    (lpl, lpt, pm ) <- popSurface
    (_  , rpt, rpr) <- popSurface
    let (depthl, _, _      ) = lpl
        (depthm, _, pmLabel) = pm
        (depthr, _, _      ) = rpr
        depth'               = max depthl (max depthm depthr) + 1
    pmInner           <- lift $ getInnerE pmLabel
    (l, lc, m, rc, r) <- lift $ dpHorizontalize player h lpt pmInner rpt
    ls                <- addSlice lc depth'
    rs                <- addSlice rc depth'
    addHoriEdge (pm, ls)
    addHoriEdge (pm, rs)
    pushOpen [(lpl, l, ls), (ls, m, rs), (rs, r, rpr)]

initialGraph
  :: (Ord a, Ord e) => Int -> DerivationPlayer s f h a e -> DerivationGraph a e
initialGraph n player = DGraph (1 + n)
                               (S.fromList topSlices)
                               (S.fromList top)
                               S.empty
                               top
                               []
                               top
 where
  topContents = (:⋊) : replicate (n - 1) (Inner $ dpTopSlice player) <> [(:⋉)]
  topSlices   = zipWith (\i s -> (0, i, s)) [0 ..] topContents
  gets (_, _, s) = s
  top = zipWith (\l r -> (l, dpTopTrans player (gets l) (gets r), r))
                topSlices
                (tail topSlices)

replayDerivation'
  :: (Foldable t, Ord a, Ord e)
  => Int
  -> DerivationPlayer s f h a e
  -> t (Leftmost s f h)
  -> Either String (DerivationGraph a e)
replayDerivation' n player deriv = ST.execStateT
  (mapM_ (replayDerivationStep player) deriv)
  (initialGraph n player)

replayDerivation
  :: (Foldable t, Ord a, Ord e)
  => DerivationPlayer s f h a e
  -> t (Leftmost s f h)
  -> Either String (DerivationGraph a e)
replayDerivation = replayDerivation' 1

unfoldDerivation'
  :: (Ord a, Ord e)
  => Int
  -> DerivationPlayer s f h a e
  -> [Common.Leftmost s f h]
  -> [Either String (DerivationGraph a e)]
unfoldDerivation' n player = go (initialGraph n player) []
 where
  go g acc [] = Right g : acc
  go g acc (step : rest) =
    case ST.execStateT (replayDerivationStep player step) g of
      Left  err -> Left err : acc
      Right g'  -> go g' (Right g : acc) rest

unfoldDerivation
  :: (Ord a, Ord e)
  => DerivationPlayer s f h a e
  -> [Common.Leftmost s f h]
  -> [Either String (DerivationGraph a e)]
unfoldDerivation = unfoldDerivation' 1

replayDerivationFull
  :: (Foldable t, Ord a, Ord e)
  => DerivationPlayer s f h a e
  -> t (Leftmost s f h)
  -> Either String (DerivationGraph a e)
replayDerivationFull player deriv = do
  graph <- replayDerivation player deriv
  if L.null $ dgSurface graph
    then Right graph
    else Left "Not all surface transitions have been frozen!"

derivationPlayerUnit :: DerivationPlayer s f h () ()
derivationPlayerUnit = DerivationPlayer (\_ _ -> ()) () usplit ufreeze uhori
 where
  usplit _ _ = Right ((), (), ())
  ufreeze _ _ = Right ()
  uhori _ _ _ _ = Right ((), (), (), (), ())

data Null = Null
  deriving (Eq, Ord)

instance Show Null where
  show Null = ""

derivationPlayerNull :: DerivationPlayer s f h Null Null
derivationPlayerNull = DerivationPlayer (\_ _ -> Null)
                                        Null
                                        nsplit
                                        nfreeze
                                        nhori
 where
  nsplit _ _ = Right (Null, Null, Null)
  nfreeze _ _ = Right Null
  nhori _ _ _ _ = Right (Null, Null, Null, Null, Null)

-- plotting derivation graphs
-- ==========================

tikzDerivationGraph
  :: (Eq a, Eq e)
  => (a -> T.Text)
  -> (e -> T.Text)
  -> DerivationGraph a e
  -> T.Text
tikzDerivationGraph showS showT (DGraph _ slices trans horis openTrans foot _)
  = T.intercalate
    "\n"
    (  (showNode <$> tikzNodes)
    <> (showTrans <$> trans')
    <> (showHori <$> S.toList horis)
    )
 where
  showText :: (Show a) => a -> T.Text
  showText = T.pack . show
  -- printing nodes and edges
  showSlice (:⋊)      = "$\\rtimes$"
  showSlice (:⋉)      = "$\\ltimes$"
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
      <> showText (getID nl)
      <> ") -- (slice"
      <> showText (getID nr)
      <> ") node[midway,below,sloped] {"
      <> showT e
      <> "};"
  showHori (p, c) =
    "\\draw[hori] (slice"
      <> showText (getID p)
      <> ") -- (slice"
      <> showText (getID c)
      <> ");"
  -- helpers
  getID (_, i, _) = i
  getDepth (d, _, _) = d
  leftNode (n, _, _) = n
  rightNode (_, _, n) = n
  -- computing node locations
  nodeChildren =
    M.fromListWith (++) $ bimap getID ((: []) . getID) <$> S.toList horis
  surface      = reverse foot <> openTrans
  trans'       = (\t -> (t, t `L.elem` foot)) <$> S.toList trans
  surfaceNodes = fmap getID $ leftNode (head surface) : fmap rightNode surface
  allNodes     = getID <$> L.sortOn getDepth (S.toList slices)
  -- compute x locations
  xloc         = foldl' findX xlocInit allNodes
   where
    xlocInit = M.fromList $ zip surfaceNodes [0.0 :: Double ..]
    mean xs = sum xs / fromIntegral (length xs)
    -- findX :: M.Map Int Double -> Int -> M.Map Int Double
    findX locs i = case M.lookup i locs of
      Just _ -> locs
      Nothing ->
        let children = nodeChildren M.! i
            childxs  = (\c -> findX locs c M.! c) <$> children
            x        = mean childxs
        in  M.insert i x locs
  tikzNodes = mkNode <$> S.toList slices
    where mkNode (depth, i, content) = (xloc M.! i, depth, i, content)

showTex :: Show a => a -> T.Text
showTex x = T.pack $ concatMap escapeTex $ show x
 where
  escapeTex '♭' = "$\\flat$"
  escapeTex '♯' = "$\\sharp$"
  escapeTex '{' = "\\{"
  escapeTex '}' = "\\}"
  escapeTex c   = [c]

mkTikzPic :: (Semigroup a, IsString a) => a -> a
mkTikzPic content =
  "\\begin{tikzpicture}\n" <> content <> "\n\\end{tikzpicture}"

tikzStandalone :: (Semigroup a, IsString a) => a -> a
tikzStandalone content =
  "\\documentclass[varwidth]{standalone}\n\
\\\usepackage{tikz}\n\
\\\usepackage{amssymb}\n\
\\\begin{document}\n\
\\\tikzstyle{slice} = []\n\
\\\tikzstyle{transition} = []\n\
\\\tikzstyle{non-terminal} = []\n\
\\\tikzstyle{terminal} = [double]\n\
\\\tikzstyle{hori} = [gray,dashed]\n\n"
    <> content
    <> "\n\\end{document}"

writeGraph
  :: (Show a, Eq a, Eq e, Show e) => FilePath -> DerivationGraph a e -> IO ()
writeGraph fn g =
  T.writeFile fn $ tikzStandalone $ mkTikzPic $ tikzDerivationGraph showTex
                                                                    showTex
                                                                    g

viewGraph
  :: (Eq a, Eq e, Show a, Show e) => FilePath -> DerivationGraph a e -> IO ()
viewGraph fn g = do
  writeGraph fn g
  callCommand $ "pdflatex " <> fn

writeGraphs
  :: (Show e, Show a, Eq a, Eq e) => FilePath -> [DerivationGraph a e] -> IO ()
writeGraphs fn gs =
  T.writeFile fn
    $   tikzStandalone
    $   T.intercalate "\n\n"
    $   mkTikzPic
    .   tikzDerivationGraph showTex showTex
    <$> gs

viewGraphs
  :: (Show e, Show a, Eq a, Eq e) => FilePath -> [DerivationGraph a e] -> IO ()
viewGraphs fn gs = do
  writeGraphs fn gs
  callCommand $ "pdflatex " <> fn
