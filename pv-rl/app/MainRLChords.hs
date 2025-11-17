{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import Common
import Control.Concurrent.STM.TVar (readTVarIO)
import Control.DeepSeq qualified as DS
import Control.Monad (forM_)
import Control.Monad.Except qualified as ET
import Control.Monad.Trans (lift)
import Control.Monad.Trans.Except qualified as ET
import Data.Aeson (FromJSON (..), eitherDecodeFileStrict, withObject, (.:))
import Data.Aeson qualified as JSON
import Data.Fixed (mod')
import Data.List (zipWith5)
import Data.List.NonEmpty qualified as NE
import Data.Ratio (Ratio (..), denominator, numerator, (%))
import Data.TypeLits (KnownNat)
import Debug.Trace qualified as DT
import GHC.Generics (Generic)
import GreedyParser (applyAction, getActions, initParseState, parseGreedy)
import GreedyParser qualified as Greedy
import Musicology.Core (SInterval, SPitch, spelledp)
import Musicology.Core qualified as Music
import Musicology.Core.Slicing qualified as Music
import PVGrammar
import PVGrammar.Parse (protoVoiceEvaluator)
import PVGrammar.Prob.Simple (loadPVHyper)
import RL (plotDeriv)
import RL qualified as RL
import RL.Jit qualified as RL
import System.ProgressBar qualified as PB
import System.Random.MWC qualified as MWC
import System.Random.Stateful (initStdGen, newIOGenM)
import Torch qualified as T
import Torch.Jit qualified as Jit
import Torch.Lens qualified as TL
import Torch.Typed qualified as TT

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
  :: forall dev
   . (RL.IsValidDevice dev)
  => RL.QModel dev
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
      !probs = RL.withBatchedEncoding state actions (RL.runBatchedPolicy actor)
      !best' = T.asValue $ T.argmax (T.Dim 0) T.KeepDim probs :: Int
      -- !dummy = RL.withBatchedEncoding state actions DS.rnf
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

benchA2C
  :: forall dev
   . (RL.IsValidDevice dev)
  => RL.QModel dev
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
      !probs = RL.withBatchedEncoding state actions (RL.runBatchedPolicy actor)
      !best' = T.asValue $ T.argmax (T.Dim 0) T.KeepDim probs :: Int
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

mainRL :: forall dev. (RL.IsValidDevice dev) => Int -> IO ()
mainRL n = do
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
  genMWC <- MWC.create -- uses a fixed seed
  (Right posterior) <- loadPVHyper "posterior.json" -- learnParams
  let fReward = RL.pvRewardChordAndActionByLen 10 posterior
      fRl = (* 0.01) <$> (RL.cosSchedule $ fromIntegral n)
      fTemp = const 1 -- \t -> (RL.cosSchedule 10 (mod' t 10)) * 10 + 1
  actor0 <- RL.mkQModel @dev
  critic0 <- RL.mkQModel @dev
  -- actor0 <- RL.loadModel @dev "actor.ht"
  -- critic0 <- RL.loadModel @dev "critic.ht"
  (rewards, losses, actor, critic) <-
    RL.trainA2C protoVoiceEvaluator mgen fReward fRl fTemp Nothing actor0 critic0 pieces n
  -- TT.save (TT.hmap' TT.ToDependent $ TT.flattenParameters actor) "actor.ht"
  -- TT.save (TT.hmap' TT.ToDependent $ TT.flattenParameters critic) "critic.ht"
  pure ()

mainPlot :: forall dev. (RL.IsValidDevice dev) => IO ()
mainPlot = do
  Right allChords <- eitherDecodeFileStrict @[DataChord] "testdata/dcml/chords_small.json"
  let !chords = filter (\c -> pathLen (dataToSlices $ notes c) > 1) allChords
      !pieces = dataToSlices . notes <$> chords
  !actor <- RL.mkQModel @dev -- RL.loadModel @dev "testmodel.ht"
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
  scriptCache <- Jit.newScriptCache
  forM_ (zip pieces [1 :: Int ..]) $ \(piece, i) -> do
    result <- parseA2C actor piece
    case result of
      Left err -> putStrLn $ "chord " <> show i <> ": " <> err
      Right ana@(Analysis deriv top) -> do
        let fn = "/tmp/rl/deriv" <> show i
        pure ()
    -- JSON.encodeFile (fn <> ".analysis.json") ana
    -- RL.plotDeriv (fn <> ".tex") deriv
    PB.incProgress pb 1
  cache <- readTVarIO $ Jit.unScriptCache scriptCache
  case cache of
    Just _ -> putStrLn "cache full"
    Nothing -> putStrLn "cache empty"

mainBenchInference :: forall dev. (RL.IsValidDevice dev) => Maybe Int -> IO ()
mainBenchInference nPieces = do
  Right allChords <- eitherDecodeFileStrict @[DataChord] "testdata/dcml/chords_small.json"
  let !chords = filter (\c -> pathLen (dataToSlices $ notes c) > 1) allChords
      !pieces =
        dataToSlices . notes <$> case nPieces of
          Just n -> take n chords
          Nothing -> chords
  !actor <- RL.mkQModel @dev
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
      Right ana@(Analysis deriv top) -> pure ()
    PB.incProgress pb 1

-- type QDevice = '(TT.CUDA, 0)

type QDevice = '(TT.CPU, 0)

main = mainBenchInference @QDevice (Just 10) -- mainPlot -- mainRL 40
