{-# LANGUAGE DataKinds #-}

module RL.ReplayBuffer where

import Common
import GreedyParser
import RL.Encoding
import RL.ModelTypes
import System.Random (RandomGen, getStdRandom)
import System.Random.Shuffle (shuffle')
import System.Random.Stateful as Rand (StatefulGen, UniformRange (uniformRM), split)

-- States and Actions
-- ------------------

newtype RPState tr tr' slc s f h = RPState (GreedyState tr tr' slc (Leftmost s f h))
  deriving (Show)

newtype RPAction slc tr s f h = RPAction (Action slc tr s f h)
  deriving (Show)

-- Replay Buffer
-- -------------

data ReplayStep tr tr' slc s f h = ReplayStep
  { replayState :: !(RPState tr tr' slc s f h)
  , replayAction :: !(RPAction slc tr s f h)
  , replayStep :: !(QEncoding '[])
  , replayNextState :: !(Maybe (RPState tr tr' slc s f h))
  , replayNextSteps :: ![QEncoding '[]]
  , replayReward :: !QType
  }

instance (Show slc, Show s, Show f, Show h, Show tr) => Show (ReplayStep tr tr' slc s f h) where
  show (ReplayStep s (RPAction a) _ s' _ r) =
    show s <> " -> " <> show s' <> " " <> show r <> "\n  " <> act
   where
    act = case a of
      Left (ActionSingle _ op) -> show op
      Right (ActionDouble _ op) -> show op

data ReplayBuffer tr tr' slc s f h
  = ReplayBuffer !Int ![ReplayStep tr tr' slc s f h]
  deriving (Show)

mkReplayBuffer :: Int -> ReplayBuffer tr tr' slc s f h
mkReplayBuffer n = ReplayBuffer n []

seedReplayBuffer :: Int -> [ReplayStep tr tr' slc s f h] -> ReplayBuffer tr tr' slc s f h
seedReplayBuffer n steps = ReplayBuffer n $ take n steps

pushStep
  :: ReplayBuffer tr tr' slc s f h
  -> ReplayStep tr tr' slc s f h
  -> ReplayBuffer tr tr' slc s f h
pushStep (ReplayBuffer n queue) trans = ReplayBuffer n $ take n $ trans : queue

sampleSteps
  :: ReplayBuffer tr tr' slc s f h
  -> Int
  -> IO [ReplayStep tr tr' slc s f h]
sampleSteps (ReplayBuffer _ queue) n = do
  -- not great, but shuffle' doesn't integrated with StatefulGen
  gen <- getStdRandom Rand.split
  pure $ take n (shuffle' queue (length queue) gen)
