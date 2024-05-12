{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE QualifiedDo #-}
{-# OPTIONS_GHC -Wno-all #-}

module Main where

import ChartParser
import Common
import Display
import GreedyParser as Greedy
import PVGrammar
import PVGrammar.Generate
import PVGrammar.Parse
import PVGrammar.Prob.Simple
  ( observeDerivation
  , sampleDerivation
  )
import ReinforcementParser qualified as RL

import Musicology.Core
import Musicology.Core.Slicing

-- import Musicology.Internal.Helpers
import Musicology.MusicXML
import Musicology.Pitch.Spelled as MT

import Data.Either (partitionEithers)
import Data.Maybe (catMaybes)
import Data.Ratio (Ratio (..))
import Lens.Micro (over)

import Control.Monad
  ( foldM
  , forM
  , forM_
  )
import Control.Monad.Except (runExceptT)
import Data.HashSet qualified as HS
import Data.List qualified as L
import Data.Semiring qualified as R
import Data.Sequence qualified as Seq
import Data.Set qualified as S
import Data.Text qualified as T
import Data.Text.IO qualified as T
import Data.Text.Lazy qualified as TL
import Data.Text.Lazy.IO qualified as TL
import Internal.MultiSet qualified as MS

import Control.DeepSeq
  ( deepseq
  , force
  )
import Control.Monad.Trans.Maybe (MaybeT (MaybeT))
import Data.Bifunctor (Bifunctor (bimap))
import Data.String (fromString)
import Inference.Conjugate
  ( runTrace
  , showTrace
  , traceTrace
  )

-- better do syntax
import Language.Haskell.DoNotation qualified as Do
import System.Random.Stateful (initStdGen, newIOGenM)

-- import           Prelude                 hiding ( Monad(..)
--                                                 , pure
--                                                 )

-- utilities
-- =========

-- reading files
-- -------------

testfile = "testdata/allemande.musicxml"

bb =
  "/home/chfin/dateien/dev/haskell/work/proto-voice-model/testdata/bluebossa.musicxml"

brahms1 =
  "/home/chfin/dateien/dev/haskell/work/proto-voice-model/testdata/brahms1.musicxml"

haydn5 = "/home/chfin/Uni/phd/data/kirlin_schenker/haydn5.xml"

invention =
  "/home/chfin/Uni/phd/data/protovoice-annotations/bach/inventions/BWV_0784.musicxml"

-- getPitchGroups :: FilePath -> IO [[OnOff SPitch (Ratio Int)]]
-- getPitchGroups file = do
--   txt <- TL.readFile file
--   return
--     $   fmap (fmap $ over onOffContent pitch)
--     $   onOffGroups
--     $   asNote
--     <$> xmlNotesHeard txt

testslices = loadSurface' testfile

-- manual inputs
-- -------------

monopath :: [a] -> Path [a] [b]
monopath = path . fmap (: [])

path :: [a] -> Path a [b]
path [] = error "cannot construct empty path"
path [a] = PathEnd a
path (a : as) = Path a [] $ path as

-- actions
-- -------

printDerivs path = do
  ds <- parseSilent pvDerivRightBranch path
  forM_ (flattenDerivations ds) $ \d -> do
    putStrLn "\nDerivation:"
    forM_ d $ \step -> do
      putStrLn $ "- " <> show step
    case replayDerivation derivationPlayerPV d of
      Left error -> putStrLn $ "Error: " <> error
      Right _ -> putStrLn "Ok."

plotDerivs fn derivs = do
  pics <- forM derivs $ \d -> case replayDerivation derivationPlayerPV d of
    Left error -> do
      putStrLn error
      print d
      return Nothing
    Right g -> return $ Just g
  viewGraphs fn $ catMaybes pics

plotDeriv fn deriv = do
  case replayDerivation derivationPlayerPV deriv of
    (Left err) -> putStrLn err
    (Right g) -> viewGraph fn g

plotSteps fn deriv = do
  let graphs = unfoldDerivation derivationPlayerPV deriv
      (errors, steps) = partitionEithers graphs
  mapM_ putStrLn errors
  viewGraphs fn $ reverse steps

checkDeriv deriv original = do
  case replayDerivation derivationPlayerPV deriv of
    (Left err) -> putStrLn err
    (Right g) -> do
      let path' = case dgFrozen g of
            (_ : (_, tlast, slast) : rst) -> do
              s <- getInner $ dslContent slast
              foldM foldPath (PathEnd s, tlast) rst
            _ -> Nothing
          orig' =
            bimap
              (Notes . MS.fromList)
              (\e -> Edges (HS.fromList e) MS.empty)
              original
      case path' of
        Nothing -> putStrLn "failed to check result path"
        Just (result, _) ->
          if result == orig'
            then putStrLn "roundtrip ok"
            else do
              putStrLn "roundtrip not ok, surfaces are not equal:"
              putStrLn "original:"
              print original
              putStrLn "recreated:"
              print result
 where
  foldPath (pacc, tacc) (_, tnew, snew) = do
    s <- getInner $ dslContent snew
    pure (Path s tacc pacc, tnew)

-- example derivations
-- ===================

derivBrahms :: [PVLeftmost (Pitch MT.SIC)]
derivBrahms = buildDerivation $ Do.do
  split $ mkSplit $ do
    splitRegular Start Stop (c' shp) RootNote False False
    splitRegular Start Stop (a' nat) RootNote False False
  spread $ mkSpread $ do
    spreadNote (a' nat) ToBoth True
    spreadNote (c' shp) (ToLeft 1) False
    addPassing (c' shp) (a' nat)
  splitRight $ mkSplit $ do
    splitPassing (c' shp) (a' nat) (b' nat) PassingMid False False
    splitRegular (Inner $ a' nat) (Inner $ a' nat) (g' shp) FullNeighbor False False
  spread $ mkSpread $ do
    spreadNote (a' nat) (ToRight 1) False
    spreadNote (c' shp) (ToLeft 1) False
    addPassing (c' shp) (a' nat)
  freeze FreezeOp
  split $ mkSplit $ do
    splitPassing (c' shp) (a' nat) (b' nat) PassingMid False False
  freeze FreezeOp
  freeze FreezeOp
  spread $ mkSpread $ do
    spreadNote (b' nat) (ToRight 1) False
    spreadNote (g' shp) (ToLeft 1) False
  split $ mkSplit $ do
    addToRight (g' shp) (a' nat) LeftNeighbor False
  freeze FreezeOp
  freeze FreezeOp
  split $ mkSplit $ do
    addToRight (b' nat) (c' shp) LeftNeighbor False
  freeze FreezeOp
  freeze FreezeOp
  freeze FreezeOp
  freeze FreezeOp
 where
  (>>) :: (Do.BindSyntax x y z) => x a -> y b -> z b
  (>>) = (Do.>>)

-- mains
-- =====

mainRL n = do
  input <- testslices 0 n
  print input
  gen <- initStdGen
  mgen <- newIOGenM gen
  Right (steps, top, deriv) <- RL.runEpisode mgen protoVoiceEvaluator (\_ _ -> 0) 1 input
  putStrLn "Finished episode"
  putStr "top: "
  print top
  putStrLn "derivation: "
  forM_ deriv (\x -> putStr "- " >> print x)
  putStrLn "RL steps: "
  forM_ steps (\x -> putStr "- " >> print x)

mainGreedy file = do
  input <- loadSurface file
  print input
  result <- runExceptT $ Greedy.parseRandom protoVoiceEvaluator input
  case result of
    Left err -> print err
    -- Right _   -> putStrLn "Ok."
    Right (Analysis deriv top) -> do
      print "done parsing."
      checkDeriv deriv input
      case replayDerivation derivationPlayerPV deriv of
        Left err -> putStrLn err
        Right g -> viewGraph "greedy.tex" g
      -- case observeDerivation deriv top of
      --   Left  err   -> print err
      --   Right trace -> do
      --     print "done observing parse."
      --     putStrLn
      --       $  "trace has "
      --       <> show (Seq.length (runTrace trace))
      --       <> " items."
      --     -- let res = traceTrace trace (sampleDerivation top)
      --     -- pure ()
      forM_ deriv print

mainCount fn = do
  input <- loadSurface fn
  print input
  count <- parseSize pvCountNoRepSplitRightBranchSplitFirst input
  putStrLn $ show count <> " derivations"

mainTest from to = do
  putStrLn $ "slices " <> show from <> " to " <> show to
  input <- testslices from to
  print input
  count <- parseSize pvCountNoRepSplitRightBranchSplitFirst input
  putStrLn $ show count <> " derivations"

mainBB = do
  input <- slicesToPath <$> slicesFromFile bb
  print input
  count <- parseSize pvCountNoRepSplitRightBranchSplitFirst input
  print count

mainBrahms = do
  input <- slicesToPath <$> slicesFromFile brahms1
  print input
  count <- parseSize pvCountNoRepSplitRightBranchSplitFirst input
  print count

mainGraph = do
  input <- slicesToPath <$> slicesFromFile brahms1
  derivs <- parseSize pvDerivRightBranch input
  let ds = S.toList $ flattenDerivations derivs
  pics <- forM ds $ \d -> case replayDerivation derivationPlayerPV d of
    Left err -> do
      putStrLn err
      print d
      return Nothing
    Right g -> return $ Just g
  print pics
  viewGraphs "brahms.tex" $ catMaybes pics

logFull tc vc n = do
  putStrLn "\n===========\n"
  putStrLn $ "level " <> show n
  putStrLn "\ntransitions:"
  mapM_ print $ tcGetByLength tc n
  putStrLn "\nverticalizations:"
  mapM_ print $ vcGetByLength vc (n - 1)

mainResult
  :: (Parsable e a v)
  => Eval e [Edge (Pitch SInterval)] a [Pitch SInterval] v
  -> Int
  -> Int
  -> IO v
mainResult evaluator from to = do
  putStrLn $ "slices " <> show from <> " to " <> show to
  input <- testslices from to
  parseSize evaluator input

parseHaydn :: (_) => _ -> IO r
parseHaydn eval = do
  slices <- slicesFromFile haydn5
  parseSize eval $ slicesToPath $ take 9 slices

mainHaydn = do
  slices <- slicesFromFile haydn5
  derivs <- parseSize pvCountNoRepSplitRightBranchSplitFirst $ slicesToPath $ take 8 slices
  print derivs
  putStrLn "done."

mainRare = do
  slices <- slicesFromFile "data/theory-article/10c_rare_int.musicxml"
  putStrLn "\\documentclass[tikz]{standalone}"
  putStrLn "\\usetikzlibrary{calc,positioning}"
  putStrLn "\\tikzstyle{slice} = []"
  putStrLn "\\tikzstyle{transition} = []"
  putStrLn "\\begin{document}"
  putStrLn "\\begin{tikzpicture}[xscale=4,yscale=1]"
  derivs <- parse logTikz pvDerivUnrestricted $ slicesToPath slices
  putStrLn "\\end{tikzpicture}"
  putStrLn "\\end{document}"
  -- pure ()
  let ds = S.toList $ flattenDerivations derivs
  pics <- forM ds $ \d -> case replayDerivation derivationPlayerPVAllEdges d of
    Left err -> do
      putStrLn err
      print d
      return Nothing
    Right g -> return $ Just g
  viewGraphs "rare.tex" $ catMaybes pics

main = mainRare
