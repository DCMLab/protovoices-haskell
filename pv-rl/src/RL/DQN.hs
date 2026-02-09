{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}
{-# HLINT ignore "Use <$>" #-}
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}

module RL.DQN where

import Common
import Display (replayDerivation, viewGraph)
import GreedyParser (Action, ActionDouble (ActionDouble), ActionSingle (ActionSingle), GreedyState, applyAction, getActions, initParseState, parseGreedy, parseStep, pickRandom)
import PVGrammar (Edge, Edges (Edges), Freeze (FreezeOp), Note, Notes (Notes), PVAnalysis, PVLeftmost, Split, Spread)
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
import Control.Monad (foldM, foldM_, forM, forM_, replicateM, when)
import Control.Monad.Except qualified as ET
import Control.Monad.Primitive (RealWorld)
import Control.Monad.State qualified as ST
import Control.Monad.Trans (lift)
import Control.Monad.Trans.Except qualified as ET
import Data.Either.Combinators (leftToMaybe)
import Data.Foldable qualified as F
import Data.List.Extra qualified as E
import Data.List.NonEmpty qualified as NE
import Data.Text.Lazy qualified as Txt
import Data.Vector qualified as V
import Debug.Trace qualified as DT
import GHC.Float (double2Float)
import Inference.Conjugate (Hyper, HyperRep, Prior (expectedProbs), evalTraceLogP, printTrace, sampleProbs)
import Musicology.Pitch
import RL.ModelTypes (toQTensor)
import System.ProgressBar qualified as PB
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
gamma :: QType
-- gamma = toOpts $ T.asTensor @Double 0.99
gamma = 0.99

-- interpolation factor between target and policy net
tau :: QType -- T.Tensor -- QTensor '[]
-- tau = toOpts $ T.asTensor @Double 0.05
tau = 0.1

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

data DQNState dev hidden opt = DQNState
  { pnet :: !(QModel dev hidden)
  , tnet :: !(QModel dev hidden)
  , opt :: !opt
  , buffer :: !(ReplayBuffer dev)
  }

greedyPolicy
  :: (Applicative m, IsValidDevice dev)
  => SomePolicy dev
  -> m Int
greedyPolicy (SomePolicy values) = do
  let choice = TT.argmax @0 @TT.DropDim $ values
  pure $ T.asValue $ TT.toDynamic choice

epsilonic
  :: (StatefulGen gen m, TT.KnownDevice dev)
  => gen
  -> QType
  -> (SomePolicy dev -> m Int)
  -> SomePolicy dev
  -> m Int
epsilonic gen epsilon policy values@(SomePolicy pol) = do
  coin <- uniformRM (0, 1) gen
  if coin >= epsilon
    then policy values
    else uniformRM (0, (TT.shape pol !! 0) - 1) gen

softmaxPolicy
  :: (StatefulGen gen m, IsValidDevice dev)
  => gen
  -> QType
  -> SomePolicy dev
  -> m Int
softmaxPolicy gen temp (SomePolicy values) = do
  let probs = TT.softmax @0 $ (TT.mulScalar (1 / temp) values)
  categorical (V.fromList $ T.asValue $ T.toDType T.Double $ TT.toDynamic $ probs) gen

runEpisode
  :: forall dev hidden gen slc' label
   . (ValidParams dev hidden)
  => PVEval SPitch
  -> gen
  -> (SomePolicy dev -> IO Int)
  -> PVRewardFn label
  -> Path [Note SPitch] [Edge SPitch]
  -> label
  -> QModel dev hidden
  -> IO
      ( Either
          String
          ([ReplayStep dev], Maybe (PVAnalysis SPitch))
      )
runEpisode !eval !gen !fPolicy !fReward !input !label pnet =
  let
    state0 = initParseState eval input
   in
    case take 200 $ getActions eval state0 of
      [] -> pure $ Left "no actions in initial state"
      (a : as) -> ET.runExceptT $ go state0 (a NE.:| as) []
 where
  go
    :: ( GreedyState
          (Edges SPitch)
          [Edge SPitch]
          (Notes SPitch)
          (PVLeftmost SPitch)
       )
    -> (NE.NonEmpty PVAction)
    -> [ReplayStep dev]
    -> ET.ExceptT String IO ([ReplayStep dev], Maybe (PVAnalysis SPitch))
  go state actions steps = do
    let qvalues = withBatchedEncoding state actions $ runBatchedQ pnet
    actionIndex <- lift $ fPolicy qvalues
    let action = actions NE.!! actionIndex
    state' <- ET.except $ applyAction state action
    let actions' = case state' of
          Left newState -> NE.nonEmpty $ take 200 $ getActions eval newState
          Right _ -> Nothing
    reward <- lift $ fReward state' actions' action label
    case (state', actions') of
      -- both new state and actions: continue
      (Left s', Just a') -> do
        let next = Just (s', a')
            newStep = ReplayStep state action next reward
        go s' a' (newStep : steps)
      -- new state but no actions: stop
      (Left s', Nothing) ->
        pure (ReplayStep state action Nothing reward : steps, Nothing)
      -- terminal state: stop
      (Right (top, deriv), _) ->
        pure (ReplayStep state action Nothing reward : steps, Just $ Analysis deriv $ PathEnd top)

trainLoop
  :: forall dev hidden tr tr' slc slc' s f h label gen opt -- params (grads :: [Type])
   . (_)
  => PVEval SPitch
  -> gen
  -> PVRewardFn label
  -> (QType -> QType)
  -- ^ learning rate schedule
  -> (QType -> QType)
  -- ^ temperature schedule
  -> (Path [Note SPitch] [Edge SPitch], label)
  -> DQNState dev hidden opt
  -> Int
  -> Int
  -> IO (DQNState dev hidden opt, QType, QType)
trainLoop !eval !gen fReward fLr fTemp (!piece, !label) oldstate@(DQNState !pnet !tnet !opt !buffer) i n = do
  -- 1. run episode, collect results
  -- let policy = epsilonic gen (eps i n) greedyPolicy
  -- let policy = softmaxPolicy gen
  let temp = fTemp $ fromIntegral i
      policy = epsilonic gen (eps i n) $ softmaxPolicy gen temp
  result <- runEpisode eval gen policy fReward piece label pnet
  case result of
    -- error? skip
    Left error -> do
      print error
      pure (oldstate, 0, 0)
    Right (steps, analysis) -> do
      -- 2. compute reward and add steps to replay buffer
      let r = sum $ replayReward <$> steps
      -- rall <- reward analysis
      -- putStrLn $ "total episode reward: " <> show r
      -- putStrLn $ "hypothetical reward: " <> show rall
      -- mapM_ print (anaDerivation analysis)
      -- mapM_ print steps'
      let buffer' = F.foldl' pushStep buffer steps
      -- 3. optimize models
      (pnet', tnet', opt', loss) <- optimizeModels buffer'
      pure (DQNState pnet' tnet' opt' buffer', r, loss)
 where
  -- A single optimization step for deep q learning (DQN)
  optimizeModels buffer' = do
    -- choose batch from replay buffer
    batch <- sampleSteps buffer' replayN
    -- compute loss over batch
    let (qsNow, qsExpected) = unzip (dqnValues <$> batch)
    expectedDetached <- T.detach $ T.stack (T.Dim 0) $ qsExpected
    let !loss =
          T.smoothL1Loss
            T.ReduceMean
            (T.stack (T.Dim 0) $ qsNow)
            expectedDetached
        !lossWithFake = TT.UnsafeMkTensor $ loss + TT.toDynamic (fakeLoss pnet)
    -- print loss
    putStr $ "loss: " <> show (T.asValue @QType $ TT.toDynamic lossWithFake)
    putStrLn $ "\tavgq: " <> show (T.asValue @QType $ T.mean $ T.stack (T.Dim 0) qsNow)
    -- optimize policy net
    let lr = toQTensor $ fLr $ fromIntegral i
    (pnet', opt') <- TT.runStep pnet opt lossWithFake lr
    -- update target net
    tparams <- TT.hmapM' TH.Detach $ TT.hmap' TT.ToDependent $ TT.flattenParameters tnet
    pparams <- TT.hmapM' TH.Detach $ TT.hmap' TT.ToDependent $ TT.flattenParameters pnet'
    let tparams' = TT.hzipWith (TH.Interpolate tau) pparams tparams
    tparamsNew <- TT.hmapM' TT.MakeIndependent tparams'
    let tnet' = TT.replaceParameters tnet tparamsNew
    -- return new state
    pure (pnet', tnet', opt', T.asValue @QType loss)

  -- The loss function of a single replay step
  dqnValues :: ReplayStep dev -> (T.Tensor, T.Tensor) -- (QTensor dev '[1], QTensor dev '[1])
  dqnValues (ReplayStep state action next r) = (TT.toDynamic qnow, TT.toDynamic qexpected)
   where
    qnext :: QTensor dev '[1]
    qnext = case next of
      Nothing -> TT.zeros
      Just (state', actions') ->
        -- let
        --   -- TODO: could make this impredicative instead of using a fake size
        --   nextQs :: QTensor dev '[1337, 1]
        --   nextQs = TT.UnsafeMkTensor $ withBatchedEncoding state' actions' $ runBatchedQ tnet
        --  in
        case withBatchedEncoding state' actions' $ runBatchedQ tnet of
          SomePolicy nextQs -> TT.maxValues @0 @TT.DropDim nextQs
    qnow = runQ' encodeStep pnet state action
    qexpected = TT.addScalar r (TT.mulScalar gamma qnext)

trainDQN
  :: forall dev hidden gen label
   . ( ValidParams dev hidden
     , TT.KnownDevice dev
     , StatefulGen gen IO
     )
  => PVEval SPitch
  -> gen
  -> PVRewardFn label
  -> (QType -> QType)
  -- ^ learning rate schedule
  -> (QType -> QType)
  -- ^ temperature schedule
  -> QModel dev hidden
  -> [(Path [Note SPitch] [Edge SPitch], label)]
  -> Int
  -> IO ([QType], [QType], QModel dev hidden)
trainDQN eval gen fReward fRl fTemp model0 pieces n = do
  -- model0 <- mkQModel
  let opt = TT.mkAdam 0 0.9 0.99 (TT.flattenParameters model0) -- T.GD
      buffer = mkReplayBuffer bufferSize
      state0 = DQNState model0 model0 opt buffer
  (DQNState modelTrained _ _ _, rewards, losses, accs) <- T.foldLoop (state0, [], [], []) n trainEpoch
  pure (reverse rewards, reverse losses, modelTrained) -- (modelTrained, rewards)
 where
  trainPiece pb i (state, rewards, losses) !piece = do
    (!state', !r, !loss) <- trainLoop eval gen fReward fRl fTemp piece state i n
    PB.incProgress pb 1
    pure (state', r : rewards, loss : losses)

  trainEpoch (state, meanRewards, meanLosses, accuracies) i = do
    pb <-
      PB.newProgressBar
        ( PB.defStyle
            { PB.stylePrefix = "Epoch " <> (PB.msg $ Txt.show i) <> ": " <> (PB.elapsedTime PB.renderDuration)
            , PB.stylePostfix = PB.exact <> " (" <> PB.percentage <> ")"
            , PB.styleWidth = PB.ConstantWidth 80
            }
        )
        10
        (PB.Progress 0 (length pieces) ())
    -- run epoch
    (state', rewards, losses) <-
      foldM (trainPiece pb i) (state, [], []) pieces
    let meanRewards' = mean rewards : meanRewards
        meanLosses' = mean losses : meanLosses
    -- compute greedy reward ("accuracy")
    accuracies' <-
      if (i `mod` 10) == 0
        then do
          results <- forM pieces $ \(piece, label) ->
            runEpisode eval gen greedyPolicy fReward piece label (pnet state')
          -- mapM (runEpisode eval $ greedyPolicy (T.forward (pnet state'))) pieces
          case sequence results of
            Left error -> do
              putStrLn error
              pure $ (-inf) : accuracies
            Right episodes -> do
              let stepss = map fst episodes
                  analyses = map snd episodes
                  accs = (\steps -> sum $ replayReward <$> steps) <$> stepss
              when ((i `mod` 100) == 0) $ do
                putStrLn "current best analyses:"
                forM_ (zip analyses [1 ..]) $ \case
                  (Just (Analysis deriv _), i) -> do
                    mapM_ print deriv
                    plotDeriv ("rl/deriv" <> show i <> ".tex") deriv
                  (Nothing, i) -> putStrLn $ "No valid analysis for input " <> show i
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
