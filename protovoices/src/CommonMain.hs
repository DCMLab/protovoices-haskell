module CommonMain where

import Common
import PVGrammar
import PVGrammar.Prob.Simple

import Inference.Conjugate (Hyper, Trace, Uniform (uniformPrior), getPosterior)
import Musicology.Pitch (SPitch, Spelled (fifths, octaves))

import Control.Monad (foldM, forM_)
import Data.Foldable qualified as F
import Data.HashSet qualified as HS
import Data.List qualified as L
import Data.Maybe (catMaybes, listToMaybe, mapMaybe)
import System.FilePath
  ( (<.>)
  , (</>)
  )
import System.FilePattern qualified as FP
import System.FilePattern.Directory qualified as FP

dataDir = "data"

type AnalysisItem =
  (String, PVAnalysis SPitch, Trace PVParams, Path [(Note SPitch)] [Edge SPitch])

loadItem :: FilePath -> FilePath -> IO (Maybe AnalysisItem)
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

loadDir :: FilePath -> [String] -> IO [AnalysisItem]
loadDir dir exclude = do
  files <- FP.getDirectoryFiles dir ["*.analysis.json"]
  let getName file = FP.match "*.analysis.json" file >>= listToMaybe
      names =
        -- exclude duplicats
        filter (`L.notElem` exclude) $ mapMaybe getName files
  -- print names
  items <- mapM (loadItem dir) names
  pure $ catMaybes items

loadArticleExamples :: IO [AnalysisItem]
loadArticleExamples = do
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
  pure dataset

learn :: Hyper PVParams -> [AnalysisItem] -> IO (Hyper PVParams)
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

learnParams :: IO (Hyper PVParams)
learnParams = do
  let prior = uniformPrior @PVParams
  dataset <- loadArticleExamples
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

mainPosterior = do
  posterior <- learnParams
  savePVHyper "posterior.json" posterior
