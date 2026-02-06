{-# LANGUAGE DataKinds #-}

-- {-# LANGUAGE QuasiQuotes #-}

module Main where

import CommonMain

import PVGrammar.Prob.Simple
import RL
import RL.Imitate
import RL.Model
import RL.Plotting

-- import H.Prelude qualified as H
-- import Language.R.QQ

import Control.Monad (replicateM_)
import Control.Monad.Cont (ContT (ContT, runContT))
import Data.Either (rights)
import Data.Fixed (mod')
import Data.List.NonEmpty qualified as NE
import Data.Maybe (catMaybes)
import Data.Set qualified as S
import Graphics.Matplotlib qualified as Plt
import Inference.Conjugate
import Pipes qualified as P
import Pipes.Prelude qualified as P
import RL.Encoding (ActionEncoding (actionEncodingOp), QEncoding (qActionEncoding))
import System.FilePath ((</>))
import System.Random (newStdGen)
import System.Random.MWC (createSystemRandom)
import System.Random.Shuffle (shuffle')
import Torch qualified as T
import Torch.Typed qualified as TT

-- Training
-- ========

type Device = '(TT.CPU, 0)
type Hidden = 8

main :: IO ()
main = trainImitation 500 "test"

trainImitation :: Int -> String -> IO ()
trainImitation epochs name = do
  let fLR :: QType -> QType
      fLR = (* 0.01) <$> (RL.cosSchedule 100 . (`mod'` 100)) -- const 0.01
      -- !model0 <- loadModel "rl/actor-imit.ht"
      hidden = TT.natValI @Hidden
      nBatches = 32
      batchSize = 32
      fullname = "e" <> show epochs <> "-nb" <> show nBatches <> "-bs" <> show batchSize <> "-h" <> show hidden <> "-" <> name
  !model0 <- mkQModel @Device @Hidden
  gen <- createSystemRandom
  Right hyper <- loadPVHyper "posterior.json"
  let probs = expectedProbs @PVParams hyper
      trainData = ImitationStream @Device probs 4 32 gen
  -- trainData <- makeChordDataset @Device 128
  -- putStrLn $ "train: " <> show (S.size $ TT.keys trainData)
  -- testData <- makeChordDataset @Device 100
  examples <- loadDir (dataDir </> "theory-article") []
  let getData (name, ana, _, _) = case derivationToDatapointsLenient ana of
        Left err -> Nothing
        Right ds -> Just $ ds
  genStd <- newStdGen
  let testData = concat $ catMaybes $ fmap getData examples
      testData' = take 50 $ shuffle' testData (length testData) genStd
  let testData = mkImitationDataset testData'
  putStrLn $ "test:  " <> show (S.size $ TT.keys @IO testData)
  (modelTrained, (hTrain, hTest)) <-
    trainDatastream fullname model0 trainData testData fLR epochs nBatches batchSize
  -- trainDataset name model0 trainData testData fLR epochs 32
  -- plotHistories "losses-imitation" [hTrain, hTest]
  pure ()

-- Debugging and Testing
-- =====================

testRun = do
  model <- mkQModel @Device @Hidden
  dataRandom <- makeChordDataset @Device 100
  (loss, acc) <-
    runContT (T.streamFromMap (T.datasetOpts 1) dataRandom) $
      validateEpoch model . fst
  putStrLn $ "test loss: " <> show loss
  putStrLn $ "test accuracy: " <> show acc

testModel fn = do
  model <- loadModel @Device @Hidden fn
  baseline <- mkQModel @Device @Hidden
  examples <- loadArticleExamples
  let getData (name, ana, _, _) = case derivationToDatapointsLenient ana of
        Left err -> do
          putStrLn $ "failed to convert " <> name <> ": " <> err
          pure Nothing
        Right ds -> do
          putStrLn $ name <> " ok."
          pure $ Just ds
  datapoints <- fmap (concat . catMaybes) $ traverse getData examples
  putStrLn $ show (length datapoints) <> " example states"
  let dataExamples = mkImitationDataset datapoints
  dataRandom <- makeChordDataset @Device 100
  test "model examples" model dataExamples
  test "model random" model dataRandom
  test "baseline examples" baseline dataExamples
  test "baseline random" baseline dataRandom
 where
  test name model dataset = do
    (loss, acc) <-
      runContT (T.streamFromMap (T.datasetOpts 1) dataset) $
        validateEpoch model . fst
    putStrLn $ "test loss (" <> name <> "): " <> show loss
    putStrLn $ "test accuracy (" <> name <> "): " <> show acc

a % b = a Plt.% Plt.mp Plt.# b
infixl 5 %

summarizeAnnotations = do
  -- examples <- loadArticleExamples
  examples <- loadDir (dataDir </> "theory-article") []
  -- ["05b_cello_prelude_1-4", "09a_hinunter", "03_bwv784_pattern"]
  let getData (name, ana, _, _) = case derivationToDatapointsLenient @'(TT.CPU, 0) ana of
        Left err -> do
          putStrLn $ "failed to convert " <> name <> ": " <> err
          pure Nothing
        Right ds -> do
          putStrLn $ name <> " ok."
          pure $ Just ds
  examplePoints <- fmap (concat . catMaybes) $ traverse getData examples
  putStrLn $ show (length examplePoints) <> " example states"
  sizesE <- reportSizes examplePoints
  sampledPoints <- makeChordData @'(TT.CPU, 0) 150
  putStrLn $ show (length sampledPoints) <> " sampled states"
  sizesS <- reportSizes sampledPoints
  Plt.file "rl/action-dist.svg" $
    Plt.readData (sizesE, sizesS)
      % "import numpy as np"
      % "import pandas as pd"
      % "import seaborn as sns"
      % "sns.set_theme()"
      % "(sizesE, sizesS) = tuple(map(np.array, data))"
      % "dfE = pd.DataFrame({'actions': sizesE, 'set': 'examples'})"
      % "dfS = pd.DataFrame({'actions': sizesS, 'set': 'sampled'})"
      % "df = pd.concat([dfE, dfS])"
      % "sns.displot(df, x='actions', hue='set', common_norm=False, stat='density')"

  pure ()
 where
  reportSizes datapoints = do
    let sizes :: [Double]
        sizes = fromIntegral . length . snd . dataInput <$> datapoints -- (fromIntegral . T.numel . TT.toDynamic . actionEncodingOp . qActionEncoding . dataInput) <$> datapoints
        meanSize = mean sizes
        stdSize = mean ((\l -> (l - meanSize) ** 2) <$> sizes)
    putStrLn $ "mean size: " <> show meanSize
    putStrLn $ "std size: " <> show stdSize
    pure sizes

testDataStream :: Int -> IO ()
testDataStream n = do
  gen <- createSystemRandom
  Right hyper <- loadPVHyper "posterior.json"
  let probs = expectedProbs @PVParams hyper
      stream = ImitationStream @Device probs 4 20 gen
      streamer () = ContT $ \k -> k (T.streamSamples stream (), ())
  replicateM_ n $ do
    putStrLn $ "Epoch"
    runContT (streamer ()) $
      \(dataset, ()) -> do
        let batches = T.collate 128 Just dataset
        P.foldM step (pure ()) pure $ P.enumerate batches P.>-> P.take 128
 where
  step () datapoints = do
    let inputs = dataInput <$> datapoints
    putStrLn $ show (sum $ NE.length . snd <$> inputs)
