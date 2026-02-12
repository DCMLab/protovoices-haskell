{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PartialTypeSignatures #-}
-- {-# LANGUAGE QualifiedDo #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# OPTIONS_GHC -O0 #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}

module Main where

import Common
import CommonMain
import GreedyParser qualified as Greedy
import PVGrammar
import PVGrammar.Parse
import PVGrammar.Prob.Simple (loadPVHyper, savePVHyper)
import RL qualified
import RL.A2C qualified
import RL.DQN qualified

import Control.Exception (SomeException, catch)
import Control.Monad (forM, zipWithM_)
import Data.Maybe (catMaybes)
import GHC.Stack (currentCallStack)
import System.Random.Stateful (initStdGen, newIOGenM)
import Torch.Typed qualified as TT
import Torch.Typed.Tensor ()

-- import           Prelude                 hiding ( Monad(..)
--                                                 , pure
--                                                 )

-- utilities
-- =========

-- debugging RL

startParsing :: FilePath -> IO (RL.PVState)
startParsing file = do
  surface <- loadSurface file
  pure $ Greedy.initParseState protoVoiceEvaluator surface

rateState :: RL.QModel Hidden Device -> RL.PVState -> RL.QTensor Device '[1]
rateState model state = RL.forwardValue model $ RL.encodePVState state

listActions :: RL.QModel Hidden Device -> RL.PVState -> IO ()
listActions model state = do
  putStrLn $ "state value: " <> show (rateState model state)
  zipWithM_ showAction (getActions state) [1 ..]
 where
  showAction action i = putStrLn $ show i <> ". " <> act <> "\n => " <> state' <> "\n q = " <> show q
   where
    state' = case Greedy.applyAction state action of
      Left error -> error
      Right state' -> show state'
    q = RL.runQ model $ RL.encodeStep @Device state action
    act = case action of
      Left (Greedy.ActionSingle _ singleAct) -> show singleAct
      Right (Greedy.ActionDouble _ doubleAct) -> show doubleAct

getActions :: RL.PVState -> [RL.PVAction]
getActions = Greedy.getActions (protoVoiceEvaluator @[] @[])

rateActions
  :: RL.QModel Hidden Device
  -> RL.PVState
  -> [RL.PVAction]
  -> [RL.QType]
rateActions model state actions = RL.runQ model . RL.encodeStep @Device state <$> actions

pickAction :: RL.PVState -> Int -> RL.PVState
pickAction state i = applyAction state $ getActions state !! (i - 1)

pickAction' :: RL.PVState -> Int -> Either String (Either RL.PVState _)
pickAction' state i = applyAction' state $ getActions state !! (i - 1)

applyAction :: RL.PVState -> RL.PVAction -> RL.PVState
applyAction state action = state'
 where
  (Right (Left state')) = Greedy.applyAction state action

applyAction' :: RL.PVState -> RL.PVAction -> Either String (Either RL.PVState _)
applyAction' = Greedy.applyAction

-- mains
-- =====

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

mainQ :: forall dev hidden. (RL.ValidParams dev hidden) => Int -> IO ()
mainQ n = do
  items <- catMaybes <$> mapM (loadItem "data/theory-article") ["10c_rare_int", "20a_sus", "04a_bwv784_top", "19b_quiescenza", "20b_cadence"]
  gen <- initStdGen
  mgen <- newIOGenM gen
  (Right posterior) <- loadPVHyper "posterior.json" -- learnParams
  -- bestRewards <- forM items $ \(_, ana, _, _) -> RL.pvRewardExp' posterior ana
  let pieces = (\(_, _, _, piece) -> (piece, pathLen piece)) <$> items
  let fReward = RL.pvRewardActionByLen posterior
      fRl = (* 0.1) <$> (RL.cosSchedule $ fromIntegral n)
      fTemp = const 1
  model0 <- RL.mkQModel :: IO (RL.QModel hidden dev)
  -- model0 <- RL.loadQModel "qmodel.ht"
  (_rewards, _losses, model) <-
    RL.DQN.trainDQN protoVoiceEvaluator mgen fReward fRl fTemp model0 pieces n
  TT.save (TT.hmap' TT.ToDependent $ TT.flattenParameters model) "qmodel.ht"
  pure ()

mainRL :: forall dev hidden. (RL.ValidParams dev hidden) => Int -> IO ()
mainRL n = do
  -- Just (_, pieceAna, _, piece) <- loadItem "data/theory-article" "10c_rare_int" -- "05b_cello_prelude_1-4" -- "05extra_cello_prelude_1-4_full"
  -- Just (_, pieceAna2, _, piece2) <- loadItem "data/theory-article" "20a_sus"
  items <- catMaybes <$> mapM (loadItem "data/theory-article") ["10c_rare_int", "20a_sus", "04a_bwv784_top", "19b_quiescenza", "20b_cadence"]
  -- Just (_, testAna, _, test) <- loadItem "data/theory-article" "20a_sus"
  gen <- initStdGen
  mgen <- newIOGenM gen
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
  actor0 <- RL.mkQModel :: IO (RL.QModel hidden dev)
  critic0 <- RL.mkQModel :: IO (RL.QModel hidden dev)
  -- actor0 <- RL.loadModel "actor.ht"
  -- critic0 <- RL.loadModel "critic.ht"
  (_rewards, _losses, actor, critic) <-
    RL.A2C.trainA2C protoVoiceEvaluator mgen fReward fRl fTemp (Just bestRewards) actor0 critic0 pieces n
  -- testBestReward <- RL.pvRewardExp posterior testAna
  -- testAcc <- RL.A2C.runAccuracy protoVoiceEvaluator posterior actor test
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
type Hidden = 8

main = catchAll $ mainQ @Device @Hidden 5000
