{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}
{-# HLINT ignore "Use <$>" #-}
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}

module RL.DQN where

import Common
import Display (replayDerivation, viewGraph)
import GreedyParser (Action, ActionDouble (ActionDouble), ActionSingle (ActionSingle), GreedyState, getActions, initParseState, parseGreedy, parseStep, pickRandom)
import PVGrammar (Edge, Edges (Edges), Freeze (FreezeOp), Notes (Notes), PVAnalysis, PVLeftmost, Split, Spread)
import PVGrammar.Generate (derivationPlayerPV)
import PVGrammar.Parse (protoVoiceEvaluator)
import PVGrammar.Prob.Simple (PVParams, evalDoubleStep, evalSingleStep, observeDerivation, observeDerivation', observeDoubleStepParsing, observeSingleStepParsing, sampleDerivation', sampleDoubleStepParsing, sampleSingleStepParsing)
import RL.Callbacks
import RL.Encoding
import RL.Model
import RL.ModelTypes
import RL.Plotting
import RL.ReplayBuffer
import RL.TorchHelpers qualified as TH

-- import Control.DeepSeq (force)
import Control.Exception (Exception, catch, onException)
import Control.Monad (foldM, foldM_, forM_, replicateM, when)
import Control.Monad.Except qualified as ET
import Control.Monad.Primitive (RealWorld)
import Control.Monad.State qualified as ST
import Control.Monad.Trans (lift)
import Data.Foldable qualified as F
import Data.List.Extra qualified as E
import Data.Vector qualified as V
import Debug.Trace qualified as DT
import GHC.Float (double2Float)
import Inference.Conjugate (Hyper, HyperRep, Prior (expectedProbs), evalTraceLogP, printTrace, sampleProbs)
import Musicology.Pitch
import System.Random.MWC.Distributions (categorical)
import System.Random.MWC.Probability qualified as MWC
import System.Random.Stateful as Rand (StatefulGen, UniformRange (uniformRM), split)
import Torch qualified as T
import Torch.HList qualified as TT
import Torch.Lens qualified
import Torch.Typed qualified as TT

-- Notes
-- -----

{-
Idee: Variant of Q-learning:
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
gamma :: (TT.KnownDevice dev) => QTensor dev '[]
-- gamma = toOpts $ T.asTensor @Double 0.99
gamma = 0.99

-- interpolation factor between target and policy net
tau :: QType -- T.Tensor -- QTensor '[]
-- tau = toOpts $ T.asTensor @Double 0.05
tau = 0.1

learningRate :: (IsValidDevice dev) => Double -> TT.LearningRate dev QDType
-- learningRate _ = 0.1
-- learningRate progress = 0.01 + TT.mulScalar progress (-0.009)
learningRate progress = 0.1 * TT.exp (TT.mulScalar progress (TT.log 0.1))

-- replay buffer
bufferSize :: Int
bufferSize = 1_000

replayN :: Int
replayN = 200

-- exploration factors
epsStart :: QType
epsStart = 0.9

epsEnd :: QType
epsEnd = 0.2

-- epsDecay :: QType
-- epsDecay = 2

eps :: Int -> Int -> QType
eps i n = expSchedule epsStart epsEnd (fromIntegral n) (fromIntegral i)

-- device = T.Device T.CPU 0

-- Deep Q-Learning
-- ---------------

data DQNState dev opt tr tr' slc s f h r = DQNState
  { pnet :: !(QModel dev)
  , tnet :: !(QModel dev)
  , opt :: !opt
  , buffer :: !(ReplayBuffer dev tr tr' slc s f h)
  }

-- epsilonGreedyPolicy
--   :: (StatefulGen gen m)
--   => gen
--   -> QType
--   -> (embedding -> QTensor '[1])
--   -> [embedding]
--   -> m Int
-- epsilonGreedyPolicy gen epsilon q actions = do
--   coin <- uniformRM (0, 1) gen
--   if coin >= epsilon
--     then pure $ T.asValue $ T.argmax (T.Dim 0) T.RemoveDim $ T.cat (T.Dim 0) (TT.toDynamic . q <$> actions)
--     else do
--       uniformRM (0, length actions - 1) gen

greedyPolicy
  :: (Applicative m)
  => (embedding -> QTensor dev '[1])
  -> [embedding]
  -> m Int
greedyPolicy q actions = do
  pure $ T.asValue $ T.argmax (T.Dim 0) T.RemoveDim $ T.cat (T.Dim 0) (TT.toDynamic . q <$> actions)

epsilonic
  :: (StatefulGen gen m)
  => gen
  -> QType
  -> ([embedding] -> m Int)
  -> [embedding]
  -> m Int
epsilonic gen epsilon policy actions = do
  coin <- uniformRM (0, 1) gen
  if coin >= epsilon
    then policy actions
    else uniformRM (0, length actions - 1) gen

softmaxPolicy
  :: (StatefulGen gen m)
  => gen
  -> (embedding -> QTensor dev '[1])
  -> [embedding]
  -> m Int
softmaxPolicy gen q actions = do
  let probs = T.softmax (T.Dim 0) $ T.cat (T.Dim 0) $ TT.toDynamic . q <$> actions
  categorical (V.fromList $ T.asValue $ T.toDType T.Double probs) gen

runEpisode
  :: forall dev tr tr' slc slc' s f h gen state action encoding step
   . ( state ~ GreedyState tr tr' slc (Leftmost s f h)
     , action ~ Action slc tr s f h
     , encoding ~ QEncoding dev '[]
     , step ~ (state, action, encoding, Maybe (state, [encoding]), Maybe Bool)
     )
  => Eval tr tr' slc slc' h (Leftmost s f h)
  -> (state -> action -> encoding)
  -> ([encoding] -> IO Int)
  -> Path slc' tr'
  -> IO
      ( Either
          String
          ([step], Analysis s f h tr slc)
      )
runEpisode !eval !encode !policyF !input =
  ST.evalStateT (ET.runExceptT $ go [] Nothing $ initParseState eval input) (Nothing, [])
 where
  -- go :: [step] -> (state, Maybe (action, encoding), Maybe Bool) -> state -> [step]
  go transitions prev state = do
    -- run step
    ST.put (Nothing, []) -- TODO: have parseStep return the action instead of using State
    result <- parseStep eval policy state
    (actionAndEncoding, actions) <- ST.get
    -- add previous step if it exists
    let transitions' = case prev of
          Nothing -> transitions
          Just (prevState, prevAction, goLeft) ->
            addStep prevState prevAction (Just (state, actions)) goLeft transitions
    -- get previous "continueLeft" decision from previous action
    let goLeft' = case prev of
          Just (_, Just (Right (ActionDouble _ op), _), _) -> case op of
            LMDoubleFreezeLeft _ -> Just True
            LMDoubleSplitLeft _ -> Just True
            _ -> Just False
          _ -> Nothing
    -- evaluate current step
    case result of
      -- done parsing
      Right (top, deriv) ->
        pure (addStep state actionAndEncoding Nothing goLeft' transitions', Analysis deriv $ PathEnd top)
      -- continue parsing
      Left state' -> go transitions' (Just (state, actionAndEncoding, goLeft')) state'
   where
    addStep
      :: state
      -> Maybe (action, encoding)
      -> Maybe (state, [encoding])
      -> Maybe Bool
      -> [(state, action, encoding, Maybe (state, [encoding]), Maybe Bool)]
      -> [(state, action, encoding, Maybe (state, [encoding]), Maybe Bool)]
    addStep state Nothing _next _goLeft ts = ts
    addStep state (Just (action, actEnc)) next goLeft ts = (state, action, actEnc, next, goLeft) : ts

    policy :: [action] -> ET.ExceptT String (ST.StateT (Maybe (action, encoding), [encoding]) IO) action
    policy [] = ET.throwError "no actions to select from"
    policy actions = do
      let encodings = encode state <$> actions
      actionIndex <- lift $ lift $ policyF encodings
      let action = actions !! actionIndex
      ST.put (Just (actions !! actionIndex, encodings !! actionIndex), encodings)
      pure action

trainLoop
  :: forall dev tr tr' slc slc' s f h gen opt -- params (grads :: [Type])
   . -- . ( StatefulGen gen IO
  --   , params ~ TT.Parameters QModel
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
  -> Eval tr tr' slc slc' h (Leftmost s f h)
  -> (GreedyState tr tr' slc (Leftmost s f h) -> Action slc tr s f h -> QEncoding dev '[])
  -> (Analysis s f h tr slc -> IO QType)
  -> (Action slc tr s f h -> Maybe Bool -> IO QType)
  -> Path slc' tr'
  -> DQNState dev opt tr tr' slc s f h QType
  -> Int
  -> Int
  -> IO (DQNState dev opt tr tr' slc s f h QType, QType, QType)
trainLoop !gen !eval !encode !reward !rewardStep !piece oldstate@(DQNState !pnet !tnet !opt !buffer) i n = do
  -- 1. run episode, collect results
  -- let policy q = epsilonic gen (eps i n) (greedyPolicy q)
  -- let policy = softmaxPolicy gen
  let policy q = epsilonic gen (eps i n) $ softmaxPolicy gen q
  result <- runEpisode eval encode (policy $ T.forward pnet) piece
  case result of
    -- error? skip
    Left error -> do
      print error
      pure (oldstate, 0, 0)
    Right (steps, analysis) -> do
      -- 2. compute reward and add steps to replay buffer
      (steps', r) <- rewardEpisode steps analysis
      -- rall <- reward analysis
      -- putStrLn $ "total episode reward: " <> show r
      -- putStrLn $ "hypothetical reward: " <> show rall
      -- mapM_ print (anaDerivation analysis)
      -- mapM_ print steps'
      let buffer' = F.foldl' pushStep buffer steps'
      -- 3. optimize models
      (pnet', tnet', opt', loss) <- optimizeModels buffer'
      pure (DQNState pnet' tnet' opt' buffer', r, loss)
 where
  mkReplay r (!state, !action, !actEnc, !next, goLeft) =
    ReplayStep (RPState state) (RPAction action) actEnc state' steps' r
   where
    (!state', !steps') = case next of
      Nothing -> (Nothing, [])
      Just (s, acts) -> (Just $ RPState s, acts)
  rewardEpisode steps analysis = do
    r <- reward analysis
    let steps' = case steps of
          [] -> []
          last : rest -> mkReplay r last : fmap (mkReplay 0) rest
    pure (steps', r)
  -- rewardSteps steps _analysis = do
  --   steps' <- mapM mkStep steps
  --   let r = sum $ replayReward <$> steps'
  --   pure (steps', r)
  --  where
  --   mkStep step@(_, action, _, _, goLeft) = do
  --     rstep <- rewardStep action goLeft
  --     pure $ mkReplay rstep step

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
    putStr $ "loss: " <> show (T.asValue @QType $ TT.toDynamic lossWithFake)
    putStrLn $ "\tavgq: " <> show (T.asValue @QType $ T.mean $ T.stack (T.Dim 0) qsNow)
    -- let params = TT.flattenParameters pnet
    --     grads = TT.grad lossWithFake params
    let lr = learningRate $ fromIntegral i / fromIntegral n
    (pnet', opt') <- TT.runStep pnet opt lossWithFake lr
    -- update target net
    tparams <- TT.hmapM' TH.Detach $ TT.hmap' TT.ToDependent $ TT.flattenParameters tnet
    pparams <- TT.hmapM' TH.Detach $ TT.hmap' TT.ToDependent $ TT.flattenParameters pnet'
    let tparams' = TT.hzipWith (TH.Interpolate tau) pparams tparams
    tparamsNew <- TT.hmapM' TT.MakeIndependent tparams'
    let tnet' = TT.replaceParameters tnet tparamsNew
    -- return new state
    pure (pnet', tnet', opt', T.asValue loss)

  -- The loss function of a single replay step
  dqnValues :: ReplayStep dev tr tr' slc s f h -> (T.Tensor, T.Tensor) -- (QTensor '[1], QTensor '[1])
  dqnValues (ReplayStep _ _ step0Enc s' step1Encs r) = (qnow, qexpected)
   where
    qzero = TT.zeros
    qnext = case s' of
      Nothing -> qzero
      Just (RPState state') ->
        let
          nextQs :: [QTensor dev '[1]]
          nextQs = TT.forward tnet <$> step1Encs
          -- nextQs = runQ' encode tnet state' <$> getActions eval state'
          toVal :: QTensor dev '[1] -> QType
          toVal = T.asValue @QType . TT.toDynamic
         in
          E.maximumOn toVal nextQs
    qnow = TT.toDynamic $ T.forward pnet step0Enc -- (encode s a)
    qexpected = TT.toDynamic $ TT.addScalar r (gamma `TT.mul` qnext)

-- delta = qnow - qexpected

trainDQN
  :: forall dev gen tr tr' slc slc' s f h
   . ( IsValidDevice dev
     , StatefulGen gen IO
     , Show s
     , Show f
     , Show h
     , s ~ Split SPitch -- TODO: keep fully open or specialize
     , f ~ Freeze SPitch
     , h ~ Spread SPitch
     , Show slc
     , Show tr
     )
  => gen
  -> Eval tr tr' slc slc' h (Leftmost s f h)
  -> (GreedyState tr tr' slc (Leftmost s f h) -> Action slc tr s f h -> QEncoding dev '[])
  -> (Analysis s f h tr slc -> IO QType)
  -> (Action slc tr s f h -> Maybe Bool -> IO QType)
  -> [Path slc' tr']
  -> Int
  -> IO ([QType], [QType], QModel dev)
trainDQN gen eval encode reward rewardStep pieces n = do
  model0 <- mkQModel
  let opt = TT.mkAdam 0 0.9 0.99 (TT.flattenParameters model0) -- T.GD
      buffer = mkReplayBuffer bufferSize
      state0 = DQNState model0 model0 opt buffer
  (DQNState modelTrained _ _ _, rewards, losses, accs) <- T.foldLoop (state0, [], [], []) n trainEpoch
  pure (reverse rewards, reverse losses, modelTrained) -- (modelTrained, rewards)
 where
  trainPiece i (state, rewards, losses) piece = do
    (state', r, loss) <- trainLoop gen eval encode reward rewardStep piece state i n
    pure (state', r : rewards, loss : losses)
  trainEpoch (state, meanRewards, meanLosses, accuracies) i = do
    -- run epoch
    (state', rewards, losses) <-
      foldM (trainPiece i) (state, [], []) pieces
    let meanRewards' = mean rewards : meanRewards
        meanLosses' = mean losses : meanLosses
    -- compute greedy reward ("accuracy")
    accuracies' <-
      if (i `mod` 10) == 0
        then do
          results <- mapM (runEpisode eval encode $ greedyPolicy (T.forward (pnet state'))) pieces
          case sequence results of
            Left error -> do
              putStrLn error
              pure $ (-inf) : accuracies
            Right episodes -> do
              let analyses = map snd episodes
              accs <- mapM reward analyses
              when ((i `mod` 100) == 0) $ do
                putStrLn "current best analyses:"
                forM_ (zip analyses [1 ..]) $ \(Analysis deriv _, i) -> do
                  mapM_ print deriv
                  plotDeriv ("rl/deriv" <> show i <> ".tex") deriv
              pure $ mean accs : accuracies
        else pure accuracies
    -- logging
    when ((i `mod` 10) == 0) $ do
      putStrLn $ "epoch " <> show i
      let (ReplayBuffer _ bcontent) = buffer state'
      putStrLn $ "buffer size: " <> show (length bcontent)
      -- mapM_ print $ take 10 bcontent
      plotHistory "rewards" $ reverse meanRewards'
      plotHistory "losses" $ reverse meanLosses'
      plotHistory "accuracy" $ reverse accuracies'
    pure (state', meanRewards', meanLosses', accuracies')

-- Plotting
-- --------

hi s = putStrLn $ "Found the Exception:" <> s
