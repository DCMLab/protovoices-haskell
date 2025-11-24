{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE QualifiedDo #-}
{-# LANGUAGE ScopedTypeVariables #-}
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
import RL qualified

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

-- better do syntax

import Data.Foldable qualified as F
import GreedyParser qualified as RL
import Language.Haskell.DoNotation qualified as Do
import RL.A2C (runAccuracy)
import System.Random.MWC.Probability qualified as MWC
import System.Random.Stateful (initStdGen, newIOGenM)
import Torch qualified as T
import Torch.Typed qualified as TT
import Torch.Typed.Tensor ()

-- import           Prelude                 hiding ( Monad(..)
--                                                 , pure
--                                                 )

-- utilities
-- =========

dataDir :: FilePath
dataDir = "data/"

-- debugging RL

startParsing :: FilePath -> IO (RL.PVState)
startParsing file = do
  surface <- loadSurface file
  pure $ Greedy.initParseState protoVoiceEvaluator surface

rateState :: RL.QModel Device -> RL.PVState -> RL.QTensor Device '[1]
rateState model state = RL.forwardValue model $ RL.encodePVState state

listActions :: RL.QModel Device -> RL.PVState -> IO ()
listActions model state = do
  putStrLn $ "state value: " <> show (rateState model state)
  zipWithM_ showAction (getActions state) [1 ..]
 where
  showAction action i = putStrLn $ show i <> ". " <> act <> "\n => " <> state' <> "\n q = " <> show q
   where
    state' = case RL.applyAction state action of
      Left error -> error
      Right state' -> show state'
    q = RL.runQ RL.encodeStep model state action
    act = case action of
      Left (RL.ActionSingle _ singleAct) -> show singleAct
      Right (RL.ActionDouble _ doubleAct) -> show doubleAct

getActions :: RL.PVState -> [RL.PVAction]
getActions = Greedy.getActions (protoVoiceEvaluator @[] @[])

rateActions
  :: RL.QModel Device
  -> RL.PVState
  -> [RL.PVAction]
  -> [RL.QType]
rateActions model state actions = RL.runQ RL.encodeStep model state <$> actions

pickAction :: RL.PVState -> Int -> RL.PVState
pickAction state i = applyAction state $ getActions state !! (i - 1)

pickAction' :: RL.PVState -> Int -> Either String (Either RL.PVState _)
pickAction' state i = applyAction' state $ getActions state !! (i - 1)

applyAction :: RL.PVState -> RL.PVAction -> RL.PVState
applyAction state action = state'
 where
  (Right (Left state')) = RL.applyAction state action

applyAction' :: RL.PVState -> RL.PVAction -> Either String (Either RL.PVState _)
applyAction' = Greedy.applyAction

-- mains
-- =====

type Piece =
  (String, PVAnalysis SPitch, Trace PVParams, Path [(Note SPitch)] [Edge SPitch])

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

-- mainAdam = do
--   (model :: TT.Linear 2 2 RL.QDType RL.QDevice) <- TT.sample TT.LinearSpec
--   let opt = TT.mkAdam 0 0.9 0.99 (TT.flattenParameters model)
--   let inputs = replicate 100_000 (1 :: RL.QType)
--   (!model', !opt') <- foldM step (model, opt) inputs
--   print model'
--   pure ()
--  where
--   step :: (TT.Linear 2 2 RL.QDType RL.QDevice, _) -> RL.QType -> IO (TT.Linear 2 2 RL.QDType RL.QDevice, TT.Adam '[TT.Tensor RL.QDevice RL.QDType '[2, 2], TT.Tensor RL.QDevice RL.QDType '[2]])
--   step (!m, !o) !i = TT.runStep m o loss 0.1
--    where
--     loss :: RL.QTensor '[]
--     loss = TT.sumAll $ TT.forward m $ TT.UnsafeMkTensor @RL.QDevice @RL.QDType $ T.asTensor (i, i)

-- main = mainAdam

mainPosterior = do
  posterior <- learnParams
  savePVHyper "posterior.json" posterior

mainRL :: forall dev. (RL.IsValidDevice dev) => Int -> IO ()
mainRL n = do
  -- Just (_, pieceAna, _, piece) <- loadItem "data/theory-article" "10c_rare_int" -- "05b_cello_prelude_1-4" -- "05extra_cello_prelude_1-4_full"
  -- Just (_, pieceAna2, _, piece2) <- loadItem "data/theory-article" "20a_sus"
  items <- catMaybes <$> mapM (loadItem "data/theory-article") ["10c_rare_int", "20a_sus", "04a_bwv784_top", "19b_quiescenza", "20b_cadence"]
  -- Just (_, testAna, _, test) <- loadItem "data/theory-article" "20a_sus"
  gen <- initStdGen
  mgen <- newIOGenM gen
  genMWC <- MWC.create -- uses a fixed seed
  (Right posterior) <- loadPVHyper "posterior.json" -- learnParams
  -- bestReward <- RL.pvRewardExp posterior pieceAna
  -- bestReward2 <- RL.pvRewardExp posterior pieceAna2
  -- putStrLn $ "optimal reward: " <> show bestReward
  -- putStrLn $ "optimal reward 2: " <> show bestReward2
  bestRewards <- forM items $ \(_, ana, _, _) -> RL.pvRewardExp' posterior ana
  let pieces = (\(_, _, _, piece) -> (piece, pathLen piece)) <$> items
  let fReward = RL.pvRewardActionByLen posterior
      fRl = (* 0.01) <$> (RL.cosSchedule $ fromIntegral n)
      fTemp = const 1
  -- TT.save (TT.hmap' TT.ToDependent $ TT.flattenParameters model) "model.ht"
  actor0 <- RL.mkQModel :: IO (RL.QModel dev)
  critic0 <- RL.mkQModel :: IO (RL.QModel dev)
  -- actor0 <- RL.loadModel "actor.ht"
  -- critic0 <- RL.loadModel "critic.ht"
  (rewards, losses, actor, critic) <-
    RL.trainA2C protoVoiceEvaluator mgen fReward fRl fTemp (Just bestRewards) actor0 critic0 pieces n
  -- testBestReward <- RL.pvRewardExp posterior testAna
  -- testAcc <- runAccuracy protoVoiceEvaluator posterior actor test
  -- case testAcc of
  --   Left error -> putStrLn $ "Error: " <> error
  --   Right (testReward, testDeriv) -> do
  --     plotDeriv "rl/test-deriv.tex" $ anaDerivation testDeriv
  --     putStrLn "test accuracy:"
  --     putStrLn $ "  optimal: " <> show testBestReward
  --     putStrLn $ "  actual: " <> show testReward
  TT.save (TT.hmap' TT.ToDependent $ TT.flattenParameters actor) "actor.ht"
  TT.save (TT.hmap' TT.ToDependent $ TT.flattenParameters critic) "critic.ht"
  pure ()

catchAll prog = catch prog (\(e :: SomeException) -> currentCallStack >>= print >> print e)

type Device = '(TT.CPU, 0)

main = catchAll $ mainRL @Device 5000
