{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import Common
import Control.Monad (forM_)
import Control.Monad.Except qualified as ET
import Control.Monad.Trans (lift)
import Control.Monad.Trans.Except qualified as ET
import Data.Aeson (FromJSON (..), eitherDecodeFileStrict, withObject, (.:))
import Data.Aeson qualified as JSON
import Data.List (zipWith5)
import Data.Ratio (Ratio (..), denominator, numerator, (%))
import Data.TypeLits (KnownNat)
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
import RL qualified
import System.Random.MWC qualified as MWC
import System.Random.Stateful (initStdGen, newIOGenM)
import Torch qualified as T
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
  :: RL.QModel
  -> Path [Note SPitch] [Edge SPitch]
  -> IO (Either String (PVAnalysis SPitch))
parseA2C !actor !input = ET.runExceptT $ go $ initParseState eval input
 where
  eval = protoVoiceEvaluator
  go !state = do
    let actions = take 20 $ getActions eval state
        encodings = RL.encodeStep state <$> actions
        -- probs = T.softmax (T.Dim 0) $ T.cat (T.Dim 0) $ TT.toDynamic . RL.forwardPolicy actor <$> encodings
        probs = RL.withBatchedEncoding state actions (RL.runBatchedPolicy actor)
        best = T.asValue $ T.argmax (T.Dim 0) T.KeepDim probs
        -- dummy = RL.withBatchedEncoding state actions (`seq` ())
        -- best = 0
        action = seq probs $ actions !! best
    state' <- ET.except $ applyAction state action
    case state' of
      Left s' -> go s'
      Right (top, deriv) -> do
        let ana = Analysis deriv (PathEnd top)
        pure ana

-- main
-- ----

mainLoading = do
  Right chords <- eitherDecodeFileStrict @[DataChord] "testdata/dcml/chords_small.json"
  putStrLn $ chordLocation $ chords !! 1
  print $ dataToSlices $ notes $ chords !! 1

mainRL n = do
  Right allChords <- eitherDecodeFileStrict @[DataChord] "testdata/dcml/chords_small.json"
  let chords = filter (\c -> pathLen (dataToSlices $ notes c) > 1) allChords
      pieces = (\chord -> (dataToSlices chord, length $ total_onset chord)) . notes <$> chords
  -- pieces = (\piece -> (piece, pathLen piece)) <$> inputs
  gen <- initStdGen
  mgen <- newIOGenM gen
  genMWC <- MWC.create -- uses a fixed seed
  (Right posterior) <- loadPVHyper "posterior.json" -- learnParams
  let fReward = RL.pvRewardActionByLen posterior
  actor0 <- RL.mkQModel
  critic0 <- RL.mkQModel
  -- actor0 <- RL.loadModel "actor.ht"
  -- critic0 <- RL.loadModel "critic.ht"
  (rewards, losses, actor, critic) <-
    RL.trainA2C protoVoiceEvaluator mgen fReward Nothing actor0 critic0 pieces n
  -- TT.save (TT.hmap' TT.ToDependent $ TT.flattenParameters actor) "actor.ht"
  -- TT.save (TT.hmap' TT.ToDependent $ TT.flattenParameters critic) "critic.ht"
  pure ()

mainPlot = do
  Right allChords <- eitherDecodeFileStrict @[DataChord] "testdata/dcml/chords_small.json"
  let chords = filter (\c -> pathLen (dataToSlices $ notes c) > 1) allChords
      pieces = dataToSlices . notes <$> take 10 chords
  actor <- RL.loadModel "actor_checkpoint.ht"
  forM_ (zip pieces [1 ..]) $ \(piece, i) -> do
    result <- parseA2C actor piece
    case result of
      Left err -> putStrLn $ "chord " <> show i <> ": " <> err
      Right ana@(Analysis deriv top) -> do
        print $ length deriv
        let fn = "/tmp/rl/deriv" <> show i <> ".tex"
        -- JSON.encodeFile fn ana
        RL.plotDeriv fn deriv

main = mainPlot
