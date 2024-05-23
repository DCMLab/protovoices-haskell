{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# HLINT ignore "Use <$>" #-}
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}

module ReinforcementParser.Learning where

import Common
import Control.DeepSeq (force)
import Control.Exception (Exception, catch, onException)
import Control.Foldl qualified as Foldl
import Control.Monad (foldM, foldM_, forM_, replicateM, when)
import Control.Monad.Except qualified as ET
import Control.Monad.Primitive (RealWorld)
import Control.Monad.State qualified as ST
import Control.Monad.Trans (lift)
import Data.Foldable qualified as F
import Data.HashSet qualified as HS
import Data.Kind (Type)
import Data.List.Extra qualified as E
import Data.Maybe (catMaybes)
import Data.Vector qualified as V
import Debug.Trace qualified as DT
import Display (replayDerivation, viewGraph)
import GHC.Float (double2Float)
import GHC.Generics (Generic)
import Graphics.Rendering.Chart.Backend.Cairo as Plt
import Graphics.Rendering.Chart.Easy ((.=))
import Graphics.Rendering.Chart.Easy qualified as Plt
import Graphics.Rendering.Chart.Gtk qualified as Plt
import GreedyParser (Action, ActionDouble (ActionDouble), ActionSingle (ActionSingle), GreedyState, Trans (Trans), getActions, initParseState, parseGreedy, parseStep, pickRandom)
import Inference.Conjugate (HyperRep, Prior (expectedProbs), evalTraceLogP, sampleProbs)
import Internal.MultiSet qualified as MS
import Internal.TorchHelpers qualified as TH
import Musicology.Pitch
import PVGrammar (Edge, Edges (Edges), Freeze (FreezeOp), Notes (Notes), PVAnalysis, PVLeftmost, Split, Spread)
import PVGrammar.Generate (derivationPlayerPV)
import PVGrammar.Parse (protoVoiceEvaluator)
import PVGrammar.Prob.Simple (PVParams, observeDerivation, observeDerivation', sampleDerivation')
import ReinforcementParser.Model
import System.Random (RandomGen, getStdRandom)
import System.Random.MWC.Distributions (categorical)
import System.Random.MWC.Probability qualified as MWC
import System.Random.Shuffle (shuffle')
import System.Random.Stateful as Rand (StatefulGen, UniformRange (uniformRM), split)
import Torch qualified as T
import Torch.HList qualified as TT
import Torch.Lens qualified
import Torch.Typed qualified as TT

-- Notes
-- -----

{-
Variant of Q-learning:
- instead of Q value (expected total reward) under optimal policy
  learn "P value": expected probability under random policy
- does this lead to a policy where p(as) ∝ reward?
  - then you learn a method of sampling from the reward distribution
  - if reward is a probability (e.g. p(deriv)), you learn to sample from that!
    - useful for unsupervised inference
- changes:
  - use proportional random policy (is this MC-tree-search?)
  - loss uses E[] instead of max over next actions.
-}

-- global settings
-- ---------------

-- discount factor
gamma :: QTensor '[]
-- gamma = toOpts $ T.asTensor @Double 0.99
gamma = 0.99

-- interpolation factor between target and policy net
tau :: QType -- T.Tensor -- QTensor '[]
-- tau = toOpts $ T.asTensor @Double 0.05
tau = 0.05

learningRate :: TT.LearningRate QDevice QDType
-- learningRate = toOpts $ T.asTensor @Double 0.01
learningRate = 0.001

-- replay buffer
bufferSize :: Int
bufferSize = 10_000

replayN :: Int
replayN = 200

-- exploration factors
epsStart :: QType
epsStart = 0.9

epsEnd :: QType
epsEnd = 0.05

epsDecay :: QType
epsDecay = 1000

eps :: Int -> QType
eps i = epsEnd + (epsStart - epsEnd) * exp (negate (fromIntegral i) / epsDecay)

-- device = T.Device T.CPU 0

-- States and Actions
-- ------------------

newtype RPState tr tr' slc s f h = RPState (GreedyState tr tr' slc (Leftmost s f h))
  deriving (Show)

newtype RPAction slc tr s f h = RPAction (Action slc tr s f h)
  deriving (Show)

-- Replay Buffer
-- -------------

data ReplayStep tr tr' slc s f h r = ReplayStep
  { _state :: !(RPState tr tr' slc s f h)
  , _action :: !(RPAction slc tr s f h)
  , _nextState :: !(Maybe (RPState tr tr' slc s f h))
  , _reward :: !r
  }
  deriving (Show)

data ReplayBuffer tr tr' slc s f h r
  = ReplayBuffer !Int ![ReplayStep tr tr' slc s f h r]
  deriving (Show)

mkReplayBuffer :: Int -> ReplayBuffer tr tr' slc s f h r
mkReplayBuffer n = ReplayBuffer n []

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

-- Reward
-- ------

inf :: QType
inf = 1 / 0

pvRewardSample
  :: MWC.Gen RealWorld
  -> PVParams HyperRep
  -> PVAnalysis SPitch
  -> IO QType
pvRewardSample gen hyper (Analysis deriv top) = do
  let trace = observeDerivation deriv top
  probs <- MWC.sample (sampleProbs @PVParams hyper) gen
  case trace of
    Left error -> do
      putStrLn $ "error giving reward: " <> error
      pure (-inf)
    Right trace -> case evalTraceLogP probs trace sampleDerivation' of
      Nothing -> do
        putStrLn "Couldn't evaluate trace while giving reward"
        pure (-inf)
      Just (_, logprob) -> pure logprob

pvRewardExp :: PVParams HyperRep -> PVAnalysis SPitch -> IO QType
pvRewardExp hyper (Analysis deriv top) =
  case trace of
    Left error -> do
      putStrLn $ "error giving reward: " <> error
      pure (-inf)
    Right trace -> case evalTraceLogP probs trace sampleDerivation' of
      Nothing -> do
        putStrLn "Couldn't evaluate trace while giving reward"
        pure (-inf)
      Just (_, logprob) -> pure logprob
 where
  probs = expectedProbs @PVParams hyper
  trace = observeDerivation deriv top

-- Deep Q-Learning
-- ---------------

data DQNState opt tr tr' slc s f h r = DQNState
  { pnet :: !(QModel DefaultQSpec)
  , tnet :: !(QModel DefaultQSpec)
  , opt :: !opt
  , buffer :: !(ReplayBuffer tr tr' slc s f h r)
  }

epsilonGreedyPolicy
  :: (StatefulGen gen m)
  => gen
  -> QType
  -> (state -> action -> QType)
  -> state
  -> [action]
  -> m action
epsilonGreedyPolicy gen epsilon q state actions = do
  coin <- uniformRM (0, 1) gen
  if coin >= epsilon
    then pure $ E.maximumOn (q state) actions
    else do
      i <- uniformRM (0, length actions - 1) gen
      pure $ actions !! i

greedyPolicy
  :: (Applicative m)
  => (state -> action -> QType)
  -> state
  -> [action]
  -> m action
greedyPolicy q state actions = do
  pure $ E.maximumOn (q state) actions

softmaxPolicy
  :: (StatefulGen gen m)
  => gen
  -> (state -> action -> QType)
  -> state
  -> [action]
  -> m action
softmaxPolicy gen q state actions = do
  let probs = T.softmax (T.Dim 0) $ T.asTensor $ q state <$> actions
  actionIndex <- categorical (V.fromList $ T.asValue $ T.toDType T.Double probs) gen
  pure $ actions !! actionIndex

runEpisode
  :: forall tr tr' slc slc' s f h gen
   . Eval tr tr' slc slc' (Leftmost s f h)
  -> ( GreedyState tr tr' slc (Leftmost s f h)
       -> [Action slc tr s f h]
       -> IO (Action slc tr s f h)
     )
  -> Path slc' tr'
  -> IO
      ( Either
          String
          ( [ ( GreedyState tr tr' slc (Leftmost s f h)
              , Action slc tr s f h
              , Maybe (GreedyState tr tr' slc (Leftmost s f h))
              )
            ]
          , Analysis s f h tr slc
          )
      )
runEpisode !eval !policyF !input =
  ST.evalStateT (ET.runExceptT $ go [] $ initParseState eval input) Nothing
 where
  go transitions state = do
    ST.put Nothing -- TODO: have parseStep return the action instead of using State
    result <- parseStep eval policy state
    action <- ST.get
    case result of
      Right (top, deriv) -> pure (addStep action Nothing transitions, Analysis deriv $ PathEnd top)
      Left state' -> go (addStep action (Just state') transitions) state'
   where
    addStep Nothing _state' ts = ts
    addStep (Just action) state' ts = (state, action, state') : ts

    policy :: [Action slc tr s f h] -> ET.ExceptT String (ST.StateT (Maybe (Action slc tr s f h)) IO) (Action slc tr s f h)
    policy [] = ET.throwError "no actions to select from"
    policy actions = do
      action <- lift $ lift $ policyF state actions
      ST.put (Just action)
      pure action

trainLoop
  :: forall tr tr' slc slc' s f h gen opt -- params (grads :: [Type])
   . -- . ( StatefulGen gen IO
  --   , params ~ TT.Parameters (QModel DefaultQSpec)
  --   , TT.HasGrad (TT.HList params) (TT.HList grads)
  --   , TT.Optimizer opt grads grads T.Double QDevice
  --   , TT.HMap' TT.ToDependent params grads
  --   , TT.HFoldrM IO TT.TensorListFold [T.ATenTensor] grads [T.ATenTensor]
  --   , TT.Apply TT.TensorListUnfold [T.ATenTensor] (TT.HUnfoldMRes IO [T.ATenTensor] grads)
  --   , TT.HUnfoldM IO TT.TensorListUnfold (TT.HUnfoldMRes IO [T.ATenTensor] grads) grads
  --   -- , Show opt
  --   )
  (_)
  => gen
  -> Eval tr tr' slc slc' (Leftmost s f h)
  -> (GreedyState tr tr' slc (Leftmost s f h) -> Action slc tr s f h -> QEncoding (QSpecGeneral DefaultQSpec))
  -> (Analysis s f h tr slc -> IO QType)
  -> Path slc' tr'
  -> DQNState opt tr tr' slc s f h QType
  -> Int
  -> IO (DQNState opt tr tr' slc s f h QType, QType, QType)
trainLoop !gen !eval !encode !reward !piece oldstate@(DQNState !pnet !tnet !opt !buffer) i = do
  -- 1. run episode, collect results
  result <- runEpisode eval (epsilonGreedyPolicy gen (eps i) $ runQ encode pnet) piece
  case result of
    -- error? skip
    Left error -> do
      print error
      pure (oldstate, 0, 0)
    Right (steps, analysis) -> do
      -- 2. compute reward and add steps to replay buffer
      r <- reward analysis
      let steps' = case steps of
            [] -> []
            last : rest -> mkStep r last : fmap (mkStep 0) rest
          buffer' = F.foldl' pushStep buffer steps'
      -- 3. optimize models
      (pnet', tnet', opt', loss) <- optimizeModels buffer'
      pure (DQNState pnet' tnet' opt' buffer', r, loss)
 where
  mkStep r (state, action, state') =
    ReplayStep (RPState state) (RPAction action) (RPState <$> state') r

  -- A single optimization step for deep q learning (DQN)
  optimizeModels buffer' = do
    -- choose batch from replay buffer
    batch <- sampleSteps buffer' replayN
    -- compute loss over batch
    let (qsNow, qsExpected) = unzip (dqnValues <$> batch)
    expectedDetached <- T.detach $ T.stack (T.Dim 0) qsExpected
    let !loss =
          T.smoothL1Loss
            T.ReduceMean
            (T.stack (T.Dim 0) qsNow)
            expectedDetached
        !lossWithFake = TT.UnsafeMkTensor $ loss + TT.toDynamic (fakeLoss pnet)
    -- print loss
    -- optimize policy net
    print lossWithFake
    -- let params = TT.flattenParameters pnet
    --     grads = TT.grad lossWithFake params
    (pnet', opt') <- TT.runStep pnet opt lossWithFake learningRate
    -- update target net
    tparams <- TT.hmapM' TH.Detach $ TT.hmap' TT.ToDependent $ TT.flattenParameters tnet
    pparams <- TT.hmapM' TH.Detach $ TT.hmap' TT.ToDependent $ TT.flattenParameters pnet'
    let tparams' = TT.hzipWith (TH.Interpolate tau) pparams tparams
    tparamsNew <- TT.hmapM' TT.MakeIndependent tparams'
    let tnet' = TT.replaceParameters tnet tparamsNew
    -- return new state
    pure (pnet', tnet', opt', T.asValue loss)

  -- The loss function of a single replay step
  dqnValues :: _ -> (T.Tensor, T.Tensor) -- (QTensor '[1], QTensor '[1])
  dqnValues (ReplayStep (RPState s) (RPAction a) s' r) = (qnow, qexpected)
   where
    qzero = TT.zeros
    qnext = case s' of
      Nothing -> qzero
      Just (RPState state') ->
        let
          nextQs :: [QTensor '[1]]
          nextQs = runQ' encode tnet state' <$> getActions eval state'
          toVal :: QTensor '[1] -> QType
          toVal = T.asValue @QType . TT.toDynamic
         in
          E.maximumOn toVal nextQs
    qnow = TT.toDynamic $ T.forward pnet (encode s a)
    qexpected = TT.toDynamic $ TT.addScalar r (gamma `TT.mul` qnext)

-- delta = qnow - qexpected

trainDQN
  :: forall gen tr tr' slc slc' s f h
   . (StatefulGen gen IO, Show s, Show f, Show h)
  => gen
  -> Eval tr tr' slc slc' (Leftmost s f h)
  -> (GreedyState tr tr' slc (Leftmost s f h) -> Action slc tr s f h -> QEncoding (QSpecGeneral DefaultQSpec))
  -> (Analysis s f h tr slc -> IO QType)
  -> [Path slc' tr']
  -> Int
  -> IO ([QType], [QType], QModel DefaultQSpec)
trainDQN gen eval encode reward pieces n = do
  model0 <- mkQModel defaultSpec
  let opt = TT.mkAdam 0 0.9 0.99 (TT.flattenParameters model0) -- T.GD
      buffer = mkReplayBuffer bufferSize
      state0 = DQNState model0 model0 opt buffer
  (DQNState modelTrained _ _ _, rewards, losses, accs) <- T.foldLoop (state0, [], [], []) n trainEpoch
  pure (reverse rewards, reverse losses, modelTrained) -- (modelTrained, rewards)
 where
  trainPiece i (state, rewards, losses) piece = do
    (state', r, loss) <- trainLoop gen eval encode reward piece state i
    pure (state', r : rewards, loss : rewards)
  trainEpoch (state, meanRewards, meanLosses, accuracies) i = do
    -- run epoch
    (state', rewards, losses) <-
      foldM (trainPiece i) (state, [], []) pieces
    let meanRewards' = Foldl.fold Foldl.mean rewards : meanRewards
        meanLosses' = Foldl.fold Foldl.mean losses : meanLosses
    -- compute greedy reward ("accuracy")
    accuracies' <-
      if (i `mod` 10) == 0
        then do
          results <- mapM (runEpisode eval $ greedyPolicy (runQ encode (pnet state'))) pieces
          case sequence results of
            Left error -> do
              putStrLn error
              pure $ (-inf) : accuracies
            Right episodes -> do
              accs <- mapM (reward . snd) episodes
              pure $ Foldl.fold Foldl.mean accs : accuracies
        else pure accuracies
    -- logging
    when ((i `mod` 10) == 0) $ do
      putStrLn $ "epoch " <> show i
      plotHistory "rewards" $ reverse meanRewards'
      plotHistory "losses" $ reverse meanLosses'
      plotHistory "accuracy" $ reverse accuracies'
    -- when ((i `mod` 100) == 0) $ do
    --   let DQNState{pnet} = state'
    --       q !s !a = T.asValue $ T.forward pnet $ encode s a
    --   results <-
    --     replicateM 100 $
    --       mapM
    --         ( runEpisode eval $ softmaxPolicy gen q
    --         -- epsilonGreedyPolicy gen (eps i) q
    --         )
    --         pieces
    -- forM_ (snd <$> episodes) $ \(Analysis deriv _) -> do
    --   putStrLn "average derivation currently:"
    --   mapM_ print deriv
    pure (state', meanRewards', meanLosses', accuracies')

-- Plotting
-- --------

mkHistoryPlot
  :: String
  -> [QType]
  -> ST.StateT
      (Plt.Layout Int QType)
      (ST.State Plt.CState)
      ()
mkHistoryPlot title values = do
  Plt.setColors $ Plt.opaque <$> [Plt.steelblue]
  Plt.layout_title .= title
  Plt.plot $ Plt.line title [points]
 where
  points = zip [1 :: Int ..] values

showHistory :: String -> [QType] -> IO ()
showHistory title values = Plt.toWindow 60 40 $ mkHistoryPlot title values

plotHistory :: String -> [QType] -> IO ()
plotHistory title values = Plt.toFile Plt.def (title <> ".svg") $ mkHistoryPlot title values

plotDeriv :: (Foldable t) => FilePath -> t (Leftmost (Split SPitch) Freeze (Spread SPitch)) -> IO ()
plotDeriv fn deriv = do
  case replayDerivation derivationPlayerPV deriv of
    (Left err) -> putStrLn err
    (Right g) -> viewGraph fn g

hi s = putStrLn $ "Found the Exception:" <> s
