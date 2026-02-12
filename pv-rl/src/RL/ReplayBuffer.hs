{-# LANGUAGE DataKinds #-}

module RL.ReplayBuffer where

import Data.List.NonEmpty qualified as NE
import GreedyParser
import RL.ModelTypes
import System.Random (getStdRandom)
import System.Random.Shuffle (shuffle')
import System.Random.Stateful as Rand (split)

-- Replay Buffer
-- -------------

data ReplayStep dev = ReplayStep
  { replayState :: !PVState
  , replayAction :: !PVAction
  , -- , replayStep :: !(QEncoding dev '[])
    replayNextState :: !(Maybe (PVState, NE.NonEmpty PVAction))
  , -- , replayNextSteps :: ![QEncoding dev '[]]
    replayReward :: !QType
  }

instance Show (ReplayStep dev) where
  show (ReplayStep s a s' r) =
    show s <> " -> " <> show s' <> " " <> show r <> "\n  " <> act
   where
    act = case a of
      Left (ActionSingle _ op) -> show op
      Right (ActionDouble _ op) -> show op

data ReplayBuffer dev
  = ReplayBuffer !Int ![ReplayStep dev]
  deriving (Show)

mkReplayBuffer :: Int -> ReplayBuffer dev
mkReplayBuffer n = ReplayBuffer n []

seedReplayBuffer :: Int -> [ReplayStep dev] -> ReplayBuffer dev
seedReplayBuffer n steps = ReplayBuffer n $ take n steps

pushStep
  :: ReplayBuffer dev
  -> ReplayStep dev
  -> ReplayBuffer dev
pushStep (ReplayBuffer n queue) trans = ReplayBuffer n $ take n $ trans : queue

sampleSteps
  :: ReplayBuffer dev
  -> Int
  -> IO [ReplayStep dev]
sampleSteps (ReplayBuffer _ queue) n = do
  -- not great, but shuffle' doesn't integrated with StatefulGen
  gen <- getStdRandom Rand.split
  pure $ take n (shuffle' queue (length queue) gen)
