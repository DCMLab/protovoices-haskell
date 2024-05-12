{-# LANGUAGE DataKinds #-}

module ReinforcementParser where

import Common (Eval, Leftmost, Path)
import Control.Monad.Except qualified as ET
import Control.Monad.State qualified as ST
import Data.Foldable qualified as F
import Data.List.Extra qualified as E
import Data.Maybe (catMaybes)
import GreedyParser (Action, GreedyState (GSDone), initParseState, parseGreedy, parseStep, pickRandom)
import PVGrammar (Edge, Edges, Notes, PVLeftmost)
import System.Random (RandomGen)
import System.Random.Shuffle (shuffle')
import System.Random.Stateful (StatefulGen, UniformRange (uniformRM))

-- States and Actions
-- ------------------

newtype RPState tr tr' slc s f h = RPState (GreedyState tr tr' slc (Leftmost s f h))
  deriving (Show)

newtype RPAction slc tr s f h = RPAction (Action slc tr s f h)
  deriving (Show)

-- Replay Buffer
-- -------------

data ReplayStep tr tr' slc s f h r = ReplayStep
  { _state :: RPState tr tr' slc s f h
  , _action :: RPAction slc tr s f h
  , _nextState :: Maybe (RPState tr tr' slc s f h)
  , reward :: r
  }
  deriving (Show)

data ReplayBuffer tr tr' slc s f h r
  = ReplayBuffer Int [ReplayStep tr tr' slc s f h r]
  deriving (Show)

mkReplayBuffer :: Int -> ReplayBuffer tr tr' slc s f h r
mkReplayBuffer n = ReplayBuffer n []

pushStep
  :: ReplayBuffer tr tr' slc s f h r
  -> ReplayStep tr tr' slc s f h r
  -> ReplayBuffer tr tr' slc s f h r
pushStep (ReplayBuffer n queue) trans = ReplayBuffer n $ take n $ trans : queue

sampleSteps
  :: (RandomGen gen)
  => ReplayBuffer tr tr' slc s f h r
  -> Int
  -> gen
  -> [ReplayStep tr tr' slc s f h r]
sampleSteps (ReplayBuffer _ queue) n gen = take n (shuffle' queue (length queue) gen)

-- Deep Q-Learning
-- ---------------

runEpisode
  :: forall tr tr' slc slc' s f h gen randm
   . (StatefulGen gen randm)
  => gen
  -> Eval tr tr' slc slc' (Leftmost s f h)
  -> (GreedyState tr tr' slc (Leftmost s f h) -> Action slc tr s f h -> Double)
  -> Double
  -> Path slc' tr'
  -> randm
      ( Either
          String
          ( [ ( GreedyState tr tr' slc (Leftmost s f h)
              , Action slc tr s f h
              , Maybe (GreedyState tr tr' slc (Leftmost s f h))
              )
            ]
          , tr
          , [Leftmost s f h]
          )
      )
runEpisode gen eval q epsilon input =
  ST.evalStateT (ET.runExceptT $ go [] $ initParseState eval input) Nothing
 where
  go transitions state = do
    ST.put Nothing -- TODO: have parseStep return the action instead of using State
    result <- parseStep eval policy state
    action <- ST.get
    case result of
      Right (top, deriv) -> pure (addStep action Nothing transitions, top, deriv)
      Left state' -> go (addStep action (Just state') transitions) state'
   where
    addStep Nothing _state' ts = ts
    addStep (Just action) state' ts = (state, action, state') : ts

    policy :: [Action slc tr s f h] -> ET.ExceptT String (ST.StateT (Maybe (Action slc tr s f h)) randm) (Action slc tr s f h)
    policy [] = ET.throwError "no actions to select from"
    policy actions = do
      -- for now, pick random
      coin <- ET.lift $ ST.lift $ uniformRM (0, 1) gen
      action <-
        if coin > epsilon
          then pure $ E.maximumOn (q state) actions
          else do
            i <- ET.lift $ ST.lift $ uniformRM (0, length actions - 1) gen
            pure $ actions !! i
      ST.put (Just action)
      pure action
