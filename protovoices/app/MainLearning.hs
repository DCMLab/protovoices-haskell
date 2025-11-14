{-# LANGUAGE LambdaCase #-}

module Main where

import ChartParser qualified
import Common
  ( Analysis (..)
  , Path (..)
  )
import Control.Monad
  ( foldM
  , forM
  , forM_
  , replicateM
  , zipWithM_
  )
import Control.Monad.Except (runExceptT)
import Data.Either (rights)
import Data.List (unzip4)
import Data.List qualified as L
import Data.Maybe
  ( catMaybes
  , listToMaybe
  , mapMaybe
  )
import Data.Vector qualified as V
import GHC.Float (int2Double)
import Graphics.Matplotlib qualified as Plt
import GreedyParser qualified as Greedy
import Inference.Conjugate
  ( Hyper
  , HyperRep
  , Prior (sampleProbs)
  , Probs
  , ProbsRep
  , Trace
  , evalTraceLogP
  , evalTracePredLogP
  , getPosterior
  , jeffreysPrior
  , traceTrace
  , uniformPrior
  )
import Musicology.Pitch (SPitch)
import Numeric.Log qualified as Log
import PVGrammar
  ( Edge
  , Note
  , PVAnalysis
  , loadAnalysis
  , loadSurface
  , topEdges
  )
import PVGrammar.Parse
  ( protoVoiceEvaluator
  , pvCountUnrestricted
  )
import PVGrammar.Prob.Simple
  ( PVParams (PVParams)
  , observeDerivation'
  , sampleDerivation'
  )
import Statistics.Sample qualified as Stats
import System.FilePath
  ( (<.>)
  , (</>)
  )
import System.FilePattern qualified as FP
import System.FilePattern.Directory qualified as FP
import System.Random.MWC.Probability qualified as MWC
import System.Random.Stateful
  ( StatefulGen
  , initStdGen
  , newIOGenM
  )
import Text.Pretty.Simple
  ( CheckColorTty (..)
  , OutputOptions (..)
  , defaultOutputOptionsNoColor
  , pPrintOpt
  )
import Text.Printf (printf)

prettyPrint :: (Show a) => a -> IO ()
prettyPrint =
  pPrintOpt
    NoCheckColorTty
    (defaultOutputOptionsNoColor{outputOptionsCompact = True})

dataDir :: FilePath
dataDir = "data/"

main = mainLearn

nBaselines :: Int
nBaselines = 100 -- default: 100

nSamples :: Int
nSamples = 250 -- default: 250

mainLearn :: IO ()
mainLearn = do
  -- initialize
  genPure <- initStdGen
  gen <- newIOGenM genPure
  genMWC <- MWC.create -- uses a fixed seed
  let prior = uniformPrior @PVParams
  -- load data
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
  -- compute overall posterior
  posteriorTotal <- learn prior dataset
  prettyPrint posteriorTotal
  -- cross validation
  let splits = leaveOneOut dataset
  crossPerps <- forM splits (comparePerNote gen genMWC prior)
  let (logpPriors, logpPosts, counts, basemeans, baselines) =
        L.unzip5 crossPerps
      count = sum counts
      logppnPrior = sum logpPriors / count
      logppnTrained = sum logpPosts / count
      logppnBaseline = sum basemeans / count
      -- look at each split
      logppns = zipWith (/) logpPosts counts -- fmap (\(_, logp, count, _, _) -> logp / count) crossPerps
      baselinelogppns = zipWith (\bs n -> (/ n) <$> bs) baselines counts
      testpieces = fst <$> splits
      testnames = (\(name, _, _, _) -> name) <$> testpieces
  showSplits logppns baselinelogppns counts testnames
  -- nsplits         = int2Double $ length splits
  -- meanlogppn      = sum logppns / nsplits
  putStrLn $ "prior logppn: " <> show logppnPrior
  putStrLn $ "prior perppn: " <> show (exp $ negate logppnPrior)
  putStrLn $ "overall trained logppn: " <> show logppnTrained
  putStrLn $ "overall trained perppn: " <> show (exp $ negate logppnTrained)
  putStrLn $ "baseline logppn: " <> show logppnBaseline
  putStrLn $ "baseline perppn: " <> show (exp $ negate logppnBaseline)
  putStrLn $ "prior-posterior Δlogppn:" <> show (logppnPrior - logppnTrained)
  putStrLn $
    printf
      "logppn (nats) & %.3f & %.3f & %.3f\\\\"
      logppnPrior
      logppnTrained
      logppnBaseline
  let natsbits = log 2
  putStrLn $
    printf
      "logppn (bits) & %.3f & %.3f & %.3f\\\\"
      (logppnPrior / natsbits)
      (logppnTrained / natsbits)
      (logppnBaseline / natsbits)
  putStrLn $
    printf
      "perppn & %.2f & %.2f & %.2f\\\\"
      (exp $ negate logppnPrior)
      (exp $ negate logppnTrained)
      (exp $ negate logppnBaseline)

testEstimator = do
  (long, short) <- loadExamples
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
  genMWC <- MWC.createSystemRandom
  let prior = uniformPrior @PVParams
  posteriorTotal <- learn prior dataset
  estimates <- derivationLogProb genMWC posteriorTotal long
  -- let means = meanOfLogs <$> tail (L.inits estimates)
  -- -- print means
  -- Plt.onscreen $ Plt.line [1 .. length means] means
  print $ meanOfLogs estimates

countRareIntDerivs = do
  Just (_, _, _, surface) <-
    loadItem
      (dataDir </> "theory-article")
      "10c_rare_int"
  count <- ChartParser.parseSize pvCountUnrestricted surface
  print count

-- loading data
-- ------------

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

loadExamples :: IO (Trace PVParams, Trace PVParams)
loadExamples = do
  Just (_, _, bwv940Trace, _) <-
    loadItem
      (dataDir </> "bach" </> "fünf-kleine-präludien")
      "BWV_0940"
  Just (_, _, rareTrace, _) <-
    loadItem
      (dataDir </> "theory-article")
      "10c_rare_int"
  pure (bwv940Trace, rareTrace)

-- learning
-- --------

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

-- evaluating
-- ----------

leaveOneOut :: [a] -> [(a, [a])]
leaveOneOut dataset = go dataset [] []
 where
  go [] _ splits = splits
  go (x : xs) done splits = go xs (x : done) ((x, xs <> done) : splits)

derivationLogProb
  :: MWC.GenIO -> Hyper PVParams -> Trace PVParams -> IO [Double]
derivationLogProb gen hyper trace = do
  probs <- replicateM nSamples $ MWC.sample (sampleProbs @PVParams hyper) gen
  let estimates =
        mapMaybe
          (\params -> snd <$> evalTraceLogP params trace sampleDerivation')
          probs
  pure $! estimates

derivationLogProb'
  :: MWC.GenIO -> Hyper PVParams -> Trace PVParams -> IO Double
derivationLogProb' gen hyper trace = do
  estimates <- derivationLogProb gen hyper trace
  pure $! meanOfLogs estimates

meanOfLogs :: (RealFloat b, Foldable t, Functor t) => t b -> b
meanOfLogs logs =
  Log.ln $! Log.sum (Log.Exp <$> logs) / fromIntegral (length logs)

countNotes :: Path [a] b -> Int
countNotes (PathEnd notes) = length notes
countNotes (Path notes edges rst) = length notes + countNotes rst

sampleBaselines
  :: (StatefulGen g IO)
  => g
  -> MWC.GenIO
  -> Hyper PVParams
  -> Piece
  -> IO [Double]
sampleBaselines gen genMWC posterior (name, _, _, surface) = do
  derivsTry <-
    replicateM nBaselines $
      runExceptT $
        Greedy.parseRandom'
          gen
          protoVoiceEvaluator
          surface
  let derivs = rights derivsTry
  putStrLn $ "  collected " <> show (length derivs) <> " samples for " <> name
  let baselines = flip fmap derivs $ \ana -> do
        trace <- observeDerivation' (anaDerivation ana)
        pure $ derivationLogProb' genMWC posterior trace
  baselines <- traverse sequence baselines -- maaaaagic
  fmap catMaybes $ forM baselines $ \case
    Left err -> putStrLn err >> pure Nothing
    Right val -> pure $ Just val

summarize :: [Double] -> (Double, Double)
summarize xs = (Stats.mean sample, Stats.stdDev sample)
 where
  sample = V.fromList xs

comparePerNote
  :: (StatefulGen g IO)
  => g
  -> MWC.GenIO
  -> Hyper PVParams
  -> (Piece, [Piece])
  -> IO (Double, Double, Double, Double, [Double])
comparePerNote gen genMWC prior (test@(tstName, _, tstTrace, tstSurface), train) =
  do
    putStrLn $ "testing on " <> tstName
    posterior <- learn prior train
    -- compute normalized logprobs
    let nnotes = int2Double $ countNotes tstSurface
    logpPrior <- derivationLogProb' genMWC prior tstTrace
    logpPost <- derivationLogProb' genMWC posterior tstTrace
    let logppnPrior = logpPrior / nnotes
        logppnPost = logpPost / nnotes
    baselines <- sampleBaselines gen genMWC posterior test
    let (basemean, basestd) = summarize baselines
        basemeanpn = basemean / nnotes
        basestdpn = basestd / nnotes
    putStrLn $ "  nnotes: " <> show nnotes
    putStrLn $ "  logppn prior: " <> show logppnPrior <> " nats"
    putStrLn $ "  logppn posterior: " <> show logppnPost <> " nats"
    putStrLn $
      "  logppn baseline: mean="
        <> show basemeanpn
        <> ", std="
        <> show basestdpn
    putStrLn $
      "  baselines top 5: "
        <> show
          (drop 95 $ (/ nnotes) <$> L.sort baselines)
    putStrLn $ "  perplexity prior: " <> show (exp $ negate logppnPrior)
    putStrLn $ "  perplexity posterior: " <> show (exp $ negate logppnPost)
    putStrLn $ "  Δlogppn: " <> show (logppnPost - logppnPrior) <> " nats"
    pure (logpPrior, logpPost, nnotes, basemean, baselines)

-- plotting
-- --------

a % b = a Plt.% Plt.mp Plt.# b
infixl 5 %

showSplits
  :: [Double] -> [[Double]] -> [Double] -> [String] -> IO (Either String String)
showSplits logppns blogppns counts testpieces =
  Plt.file "splits.svg" $
    Plt.readData (logppns, blogppns, counts, testpieces)
      % "import numpy as np"
      % "import pandas as pd"
      % "import seaborn as sns"
      % "from matplotlib.lines import Line2D"
      % "(logppns, blogppns, counts, pieces) = tuple(map(np.array, data))"
      % "baselines = pd.concat([pd.DataFrame({'logppn': bppns, 'piece': piece}) for bppns, piece in zip(blogppns, pieces)])"
      % "testscores = pd.DataFrame({'logppn': logppns, 'piece': pieces})"
      % "colors = sns.color_palette()"
      % "fig, ax = plot.subplots(figsize=(10,5))"
      % "sns.boxplot(ax=ax, y='piece', x='logppn', data=baselines, whis=(2,98), color=colors[0], fliersize=4)"
      % "sns.scatterplot(ax=ax, y='piece', x='logppn', data=testscores, color=colors[1])"
      % "ax.set_xlabel('log probability per note')"
      % "ax.invert_yaxis()"
      % "ax.legend(title='derivations', handles=[Line2D([0], [0], color='w', marker='d', markerfacecolor=colors[0]), Line2D([0], [0], color='w', marker='o', markerfacecolor=colors[1])], labels=['random', 'annotated'])"
      % "fig.tight_layout()"
      % "fig.savefig('splits.pdf')"
      % "fig.savefig('splits.png')"
