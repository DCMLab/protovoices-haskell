{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# OPTIONS_GHC -O0 #-}

module Main where

import Common
import GreedyParser (applyAction, getActions, initParseState)
import PVGrammar
import PVGrammar.Parse (protoVoiceEvaluator)
import PVGrammar.Prob.Simple (loadPVHyper)
import RL qualified
import RL.A2C qualified as RL

-- import RL.Jit qualified as RL

import Torch qualified as T
import Torch.Internal.Unmanaged.Type.Context (hasCUDA)
import Torch.Typed qualified as TT

import Musicology.Core (SInterval, SPitch, spelledp)
import Musicology.Core qualified as Music
import Musicology.Core.Slicing qualified as Music

import Control.Monad (forM_)
import Control.Monad.Except qualified as ET
import Control.Monad.Trans (lift)
import Control.Monad.Trans.Except qualified as ET
import Data.Aeson (FromJSON (..), eitherDecodeFileStrict, withObject, (.:))
import Data.List (zipWith5)
import Data.List.NonEmpty qualified as NE
import Data.Ratio (Ratio, denominator, numerator, (%))
import GHC.Generics (Generic)
import System.ProgressBar qualified as PB
import System.Random.Stateful (initStdGen, newIOGenM)

-- loading training data
-- ---------------------

data DataChord = DataChord
  { label :: String
  , root :: Int
  , expected :: [Int]
  , corpus :: String
  , piece :: String
  , mn :: Int
  , mn_onset :: DataRatio
  , notes :: DataNotes
  }
  deriving (Generic, FromJSON)

chordLocation :: DataChord -> String
chordLocation DataChord{..} =
  corpus <> "/" <> piece <> "@" <> show mn <> "." <> show mn_onset

data DataNotes = DataNotes
  { total_onset :: [DataRatio]
  , total_offset :: [DataRatio]
  , tpc :: [Int]
  , octave :: [Int]
  }
  deriving (Generic, FromJSON)

newtype DataRatio = DataRatio {getRatio :: Ratio Int}

instance Show DataRatio where
  show (DataRatio ratio) =
    if denom == 1
      then show num
      else show num <> "/" <> show denom
   where
    num = numerator ratio
    denom = denominator ratio

instance FromJSON DataRatio where
  parseJSON = withObject "DataRatio" $ \obj -> do
    n <- obj .: "n"
    d <- obj .: "d"
    pure $ DataRatio $ n % d

convertNotes :: DataNotes -> [Music.NoteId SInterval (Ratio Int) String]
convertNotes DataNotes{..} = zipWith5 mkNote total_onset total_offset tpc octave [0 ..]
 where
  mkNote on off f o i = Music.NoteId pitch (getRatio on) (getRatio off) ("note" <> show i)
   where
    pitch = spelledp f (o - (f * 4 `div` 7))

dataToSlices :: DataNotes -> Path [Note SPitch] [Edge SPitch]
dataToSlices dataNotes =
  let
    notes = convertNotes dataNotes
    slices = Music.slicePiece Music.tiedSlicer notes
   in
    slicesToPath $ mkSlice <$> filter (not . null) slices
 where
  mkSlice notes = mkNote <$> notes
  mkNote (Music.NoteId p _ _ i, tie) = (Note p i, Music.rightTie tie)

-- running models
-- --------------

parseA2C
  :: forall dev hidden
   . (RL.ValidParams dev hidden)
  => RL.QModel hidden dev
  -> Path [Note SPitch] [Edge SPitch]
  -> IO (Either String (PVAnalysis SPitch))
parseA2C !actor !input = case take 200 $ getActions eval s0 of
  [] -> pure $ Left "cannot parse: no possible actions for first step!"
  (a : as) -> ET.runExceptT $ go s0 (a NE.:| as)
 where
  s0 = initParseState eval input
  eval = protoVoiceEvaluator
  go !state !actions = do
    let
      -- encodings = RL.encodeStep state <$> actions
      -- probs = T.softmax (T.Dim 0) $ T.cat (T.Dim 0) $ TT.toDynamic . RL.forwardPolicy actor <$> encodings
      -- showTensor t = "- " <> show (T.device $ DS.force t) <> "\n"
      -- checkEncoding enc = DT.trace (concatMap showTensor $ RL.flattenTensors enc) 0
      !probs = RL.dynPolicy $ RL.withBatchedEncoding @dev state actions (RL.runBatchedPolicy 1 actor)
      !best = T.asValue $ T.argmax (T.Dim 0) T.KeepDim probs :: Int
      -- !dummy = RL.withBatchedEncoding state actions DS.rnf
      -- best = 0
      action = actions NE.!! best
    state' <- ET.except $ applyAction state action
    let actions' = case state' of
          Left nextState -> NE.nonEmpty $ take 200 $ getActions eval nextState
          Right _ -> Nothing
    case (state', actions') of
      (Left _s, Nothing) -> do
        lift $ appendFile "incomplete.log" $ show state
        lift $ putStr "!"
        ET.throwE "cannot parse: no possible actions in non-terminal state:"
      (Left s', Just a') -> go s' a'
      (Right (top, deriv), _) -> do
        let ana = Analysis deriv (PathEnd top)
        pure ana

benchA2C
  :: forall dev hidden
   . (RL.ValidParams dev hidden)
  => RL.QModel hidden dev
  -> Path [Note SPitch] [Edge SPitch]
  -> IO (Either String (PVAnalysis SPitch))
benchA2C !actor !input = case take 200 $ getActions eval s0 of
  [] -> pure $ Left "cannot parse: no possible actions for first step!"
  (a : as) -> ET.runExceptT $ go s0 (a NE.:| as)
 where
  s0 = initParseState eval input
  eval = protoVoiceEvaluator
  go !state !actions = do
    let
      !probs = RL.dynPolicy $ RL.withBatchedEncoding @dev state actions (RL.runBatchedPolicy 1 actor)
      !_best' = T.asValue $ T.argmax (T.Dim 0) T.KeepDim probs :: Int
      best = 0
      action = actions NE.!! best
    state' <- ET.except $ applyAction state action
    let actions' = case state' of
          Left nextState -> NE.nonEmpty $ take 200 $ getActions eval nextState
          Right _ -> Nothing
    case (state', actions') of
      (Left _, Nothing) ->
        ET.throwE "cannot parse: no possible actions in non-terminal state!"
      (Left s', Just a') -> go s' a'
      (Right (top, deriv), _) -> do
        let ana = Analysis deriv (PathEnd top)
        pure ana

-- main
-- ----

mainLoading = do
  Right chords <- eitherDecodeFileStrict @[DataChord] "testdata/dcml/chords_small.json"
  putStrLn $ chordLocation $ chords !! 1
  print $ dataToSlices $ notes $ chords !! 1

mainRL :: forall dev hidden. (RL.ValidParams dev hidden) => Int -> IO ()
mainRL n = do
  _ <- hasCUDA -- delay on first call
  Right allChords <- eitherDecodeFileStrict @[DataChord] "testdata/dcml/chords_small.json"
  let chords = filter (\c -> pathLen (dataToSlices $ notes c) > 1) allChords
      mkPiece chord = (slices, (len, exptd))
       where
        slices = dataToSlices $ notes chord
        len = length $ total_onset $ notes chord
        exptd = expected chord
      pieces = mkPiece <$> chords
  -- pieces = (\piece -> (piece, pathLen piece)) <$> inputs
  gen <- initStdGen
  mgen <- newIOGenM gen
  (Right posterior) <- loadPVHyper "posterior.json" -- learnParams
  let fReward = RL.pvRewardChordAndActionByLen 10 posterior
      fRl = (* 0.01) <$> (RL.cosSchedule $ fromIntegral n)
      fTemp = const 1 -- \t -> (RL.cosSchedule 10 (mod' t 10)) * 10 + 1
      -- actor0 <- RL.mkQModel @dev hidden
      -- critic0 <- RL.mkQModel @dev hidden
  actor0 <- RL.loadQModel @dev @hidden "actor_10p_nodeadend.ht" -- "actor_checkpoint.ht"
  critic0 <- RL.loadQModel @dev @hidden "critic_10p_nodeadend.ht"
  (_rewards, _losses, _actor, _critic) <-
    RL.trainA2C protoVoiceEvaluator mgen fReward fRl fTemp Nothing actor0 critic0 pieces n
  -- saveModel "actor.ht" actor
  -- saveModel "critic.ht" critic
  pure ()

mainPlot :: forall dev hidden. (RL.ValidParams dev hidden) => IO ()
mainPlot = do
  writeFile "incomplete.log" ""
  Right allChords <- eitherDecodeFileStrict @[DataChord] "testdata/dcml/chords_small.json"
  let !chords = filter (\c -> pathLen (dataToSlices $ notes c) > 1) allChords
      !pieces = dataToSlices . notes <$> chords
  !actor <- RL.loadQModel @dev @hidden "actor_checkpoint.ht"
  putStrLn "Model loaded"
  pb <-
    PB.newProgressBar
      ( PB.defStyle
          { PB.stylePrefix = "Parsing " <> (PB.elapsedTime PB.renderDuration)
          , PB.stylePostfix = PB.exact <> " (" <> PB.percentage <> ")"
          , PB.styleWidth = PB.ConstantWidth 80
          }
      )
      10
      (PB.Progress 0 (length pieces) ())
  forM_ (zip pieces [1 :: Int ..]) $ \(piece, i) -> do
    result <- parseA2C actor piece
    case result of
      Left err -> putStrLn $ "chord " <> show i <> ": " <> err
      Right (Analysis _deriv _top) -> do
        let _fn = "/tmp/rl/deriv" <> show i
        pure ()
    -- JSON.encodeFile (fn <> ".analysis.json") ana
    -- RL.plotDeriv (fn <> ".tex") deriv
    PB.incProgress pb 1

mainBenchInference :: forall dev hidden. (RL.ValidParams dev hidden) => Maybe Int -> IO ()
mainBenchInference nPieces = do
  _hascuda <- hasCUDA
  Right allChords <- eitherDecodeFileStrict @[DataChord] "testdata/dcml/chords_small.json"
  let !chords = filter (\c -> pathLen (dataToSlices $ notes c) > 1) allChords
      !pieces =
        dataToSlices . notes <$> case nPieces of
          Just n -> take n chords
          Nothing -> chords
  !actor <- RL.mkQModel @dev @hidden
  putStrLn "Model loaded"
  pb <-
    PB.newProgressBar
      ( PB.defStyle
          { PB.stylePrefix = "Parsing " <> (PB.elapsedTime PB.renderDuration)
          , PB.stylePostfix = PB.exact <> " (" <> PB.percentage <> ")"
          , PB.styleWidth = PB.ConstantWidth 80
          }
      )
      10
      (PB.Progress 0 (length pieces) ())
  forM_ (zip pieces [1 :: Int ..]) $ \(piece, i) -> do
    result <- parseA2C actor piece
    case result of
      Left err -> putStrLn $ "chord " <> show i <> ": " <> err
      Right _ana -> pure ()
    PB.incProgress pb 1

-- type QDevice = '(TT.CUDA, 0)

type QDevice = '(TT.CPU, 0)
type QHidden = 8

main = mainRL @QDevice @QHidden 1000
