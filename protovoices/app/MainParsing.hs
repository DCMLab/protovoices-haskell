{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# OPTIONS_GHC -Wno-all #-}

module Main where

import ChartParser
import Common
import Display
import GreedyParser qualified as Greedy
import PVGrammar
import PVGrammar.Generate
import PVGrammar.Parse
import PVGrammar.Prob.Simple
  ( PVParams (PVParams)
  , loadPVHyper
  , observeDerivation
  , observeDerivation'
  , sampleDerivation
  , sampleDerivation'
  , savePVHyper
  )

import Musicology.Core hiding (Note (..), (<.>))
import Musicology.Core.Slicing

-- import Musicology.Internal.Helpers
import Musicology.MusicXML
import Musicology.Pitch.Spelled as MT

import Data.Either (partitionEithers)
import Data.Maybe (catMaybes, listToMaybe, mapMaybe)
import Data.Ratio (Ratio (..))
import Lens.Micro (over)

import Control.Monad
  ( foldM
  , forM
  , forM_
  , replicateM
  , zipWithM_
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
import Control.Exception (SomeException, catch)
import Control.Monad.Trans.Maybe (MaybeT (MaybeT))
import Data.Bifunctor (Bifunctor (bimap))
import Data.Foldable qualified as F
import Data.String (fromString)
import GHC.Stack (currentCallStack)
import Inference.Conjugate
  ( Hyper
  , Trace
  , Uniform (uniformPrior)
  , getPosterior
  , runTrace
  , showTrace
  , traceTrace
  )
import System.FilePath
  ( (<.>)
  , (</>)
  )
import System.FilePattern qualified as FP
import System.FilePattern.Directory qualified as FP

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

dataDir :: FilePath
dataDir = "data/"

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
              (Notes . HS.fromList)
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

-- -- mains
-- -- =====

type Piece =
  (String, PVAnalysis SPitch, Trace PVParams, Path [Note SPitch] [Edge SPitch])

loadItem :: FilePath -> FilePath -> IO (Maybe Piece)
loadItem dir name = do
  ana <- loadAnalysis (dir </> name <.> "analysis.json")
  case ana of
    Left _err -> pure Nothing
    Right a ->
      if anaTop a == PathEnd topEdges
        then do
          surface <- loadSurface (dir </> name <.> "musicxml")
          case observeDerivation' (anaDerivation a) of
            Left _err -> do
              putStrLn $ "could not observe trace for " <> name <> ", skipping."
              pure Nothing
            Right trace -> pure $ Just (name, a, trace, surface)
        else do
          putStrLn $ "derivation for " <> name <> " is incomplete, skipping."
          pure Nothing

loadDir :: FilePath -> [String] -> IO [Piece]
loadDir dir exclude = do
  files <- FP.getDirectoryFiles dir ["*.analysis.json"]
  let getName file = FP.match "*.analysis.json" file >>= listToMaybe
      names =
        -- exclude duplicats
        filter (`L.notElem` exclude) $ mapMaybe getName files
  -- print names
  items <- mapM (loadItem dir) names
  pure $ catMaybes items

learn :: Hyper PVParams -> [Piece] -> IO (Hyper PVParams)
learn = foldM train
 where
  train prior (name, _, trace, _) =
    case getPosterior prior trace sampleDerivation' of
      Nothing -> do
        putStrLn $ "couldn't compute posterior for " <> name <> ", skipping."
        pure prior
      Just post -> do
        -- putStrLn $ "learned from " <> name <> "."
        pure post

learnParams = do
  let prior = uniformPrior @PVParams
  articleExamples <-
    loadDir
      (dataDir </> "theory-article")
      ["05b_cello_prelude_1-4", "09a_hinunter", "03_bwv784_pattern"]
  Just bwv939 <-
    loadItem
      (dataDir </> "bach" </> "fünf-kleine-präludien")
      "BWV_0939"
  Just bwv940 <-
    loadItem
      (dataDir </> "bach" </> "fünf-kleine-präludien")
      "BWV_0940"
  let dataset = bwv939 : bwv940 : articleExamples
  -- let dataset = take 3 articleExamples
  putStrLn "list of pieces:"
  forM_ dataset $ \(name, _ana, _trace, _surface) -> do
    putStrLn $ "  " <> name
  let pitchSets = (\(_, _, _, surface) -> HS.fromList $ fmap notePitch $ F.concat $ pathArounds surface) <$> dataset
      allPitches = HS.unions pitchSets
  putStr "fifths: "
  print $ HS.map fifths allPitches
  putStr "octaves: "
  print $ HS.map octaves allPitches
  -- compute overall posterior
  learn prior dataset

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
  :: (Parsable e a h v)
  => Eval e [Edge SPitch] a [Note SPitch] h v
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

mainPosterior = do
  posterior <- learnParams
  savePVHyper "posterior.json" posterior

main = mainPosterior
