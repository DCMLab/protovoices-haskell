{-# LANGUAGE DataKinds #-}

module Main where

import CommonMain

import PVGrammar.Prob.Simple
import RL.Imitate
import RL.Model
import RL.Plotting

import Data.Set qualified as S
import Inference.Conjugate
import System.Random.MWC (createSystemRandom)
import Torch.Typed qualified as TT

type Device = '(TT.CPU, 0)

trainImitation :: Int -> IO ()
trainImitation epochs = do
  let fLR = const 0.01 -- (* 0.01) <$> (RL.cosSchedule $ fromIntegral n)
  -- !model0 <- loadModel "rl/actor-imit-inf2.ht"
  !model0 <- mkQModel @Device
  gen <- createSystemRandom
  Right hyper <- loadPVHyper "posterior.json"
  let probs = expectedProbs @PVParams hyper
      trainData = ImitationStream @Device probs 4 20 gen
  -- trainData <- makeChordDataset @Device 10000
  testData <- makeChordDataset @Device 100
  -- putStrLn $ "train: " <> show (S.size $ TT.keys trainData)
  putStrLn $ "test:  " <> show (S.size $ TT.keys testData)
  (modelTrained, (hTrain, hTest)) <-
    trainDatastream model0 trainData testData fLR epochs 10 256
  -- plotHistories "losses-imitation" [hTrain, hTest]
  pure ()

main :: IO ()
main = trainImitation 100
