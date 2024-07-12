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

data ReplayStep tr tr' slc s f h r = ReplayStep
  { replayState :: !(RPState tr tr' slc s f h)
  , replayAction :: !(RPAction slc tr s f h)
  , replayStep :: !(QEncoding '[] (QSpecGeneral DefaultQSpec))
  , replayNextState :: !(Maybe (RPState tr tr' slc s f h))
  , replayNextSteps :: ![QEncoding '[] (QSpecGeneral DefaultQSpec)]
  , replayReward :: !r
  }

instance (Show slc, Show s, Show f, Show h, Show r, Show tr) => Show (ReplayStep tr tr' slc s f h r) where
  show (ReplayStep s (RPAction a) _ s' _ r) =
    show s <> " -> " <> show s' <> " " <> show r <> "\n  " <> act
   where
    act = case a of
      Left (ActionSingle _ op) -> show op
      Right (ActionDouble _ op) -> show op

data ReplayBuffer tr tr' slc s f h r
  = ReplayBuffer !Int ![ReplayStep tr tr' slc s f h r]
  deriving (Show)

mkReplayBuffer :: Int -> ReplayBuffer tr tr' slc s f h r
mkReplayBuffer n = ReplayBuffer n []

seedReplayBuffer :: Int -> [ReplayStep tr tr' slc s f h r] -> ReplayBuffer tr tr' slc s f h r
seedReplayBuffer n steps = ReplayBuffer n $ take n steps

pushStep
  :: ReplayBuffer tr tr' slc s f h r
  -> ReplayStep tr tr' slc s f h r
  -> ReplayBuffer tr tr' slc s f h r
pushStep (ReplayBuffer n queue) trans = ReplayBuffer n $ take n $ trans : queue

sampleSteps
  :: ReplayBuffer tr tr' slc s f h r
  -> Int
  -> IO [ReplayStep tr tr' slc s f h r]
sampleSteps (ReplayBuffer _ queue) n = do
  -- not great, but shuffle' doesn't integrated with StatefulGen
  gen <- getStdRandom Rand.split
  pure $ take n (shuffle' queue (length queue) gen)
