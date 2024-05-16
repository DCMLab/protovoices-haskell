{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RecordWildCards #-}
{-# HLINT ignore "Use <$>" #-}
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}

module ReinforcementParser where

import Common
import Control.Monad (foldM, foldM_, when)
import Control.Monad.Except qualified as ET
import Control.Monad.Primitive (RealWorld)
import Control.Monad.State qualified as ST
import Data.Foldable qualified as F
import Data.HashSet qualified as HS
import Data.List.Extra qualified as E
import Data.Maybe (catMaybes)
import Debug.Trace qualified as DT
import GHC.Generics (Generic)
import GreedyParser (Action, ActionDouble (ActionDouble), ActionSingle (ActionSingle), GreedyState, Trans (Trans), getActions, initParseState, parseGreedy, parseStep, pickRandom)
import Inference.Conjugate (HyperRep, evalTraceLogP, sampleProbs)
import Internal.MultiSet qualified as MS
import Musicology.Pitch
import PVGrammar (Edge, Edges (Edges), Freeze (FreezeOp), Notes (Notes), PVAnalysis, PVLeftmost, Split, Spread)
import PVGrammar.Parse (protoVoiceEvaluator)
import PVGrammar.Prob.Simple (PVParams, observeDerivation', sampleDerivation')
import System.Random (RandomGen, getStdRandom)
import System.Random.MWC.Probability qualified as MWC
import System.Random.Shuffle (shuffle')
import System.Random.Stateful as Rand (StatefulGen, UniformRange (uniformRM), split)
import Torch qualified as T
import Torch.Lens qualified

-- global settings
-- ---------------

device :: T.Device
-- device = T.Device T.CUDA 0
device = T.Device T.CPU 0

opts :: T.TensorOptions
opts = T.withDType T.Double $ T.withDevice device T.defaultOpts

toOpts :: forall a. (Torch.Lens.HasTypes a T.Tensor) => a -> a
toOpts = T.toDevice device . T.toType T.Double

-- discount factor
gamma :: Double
gamma = 0.99

-- interpolation factor between target and policy net
tau :: Double
tau = 0.05

learningRate :: T.LearningRate
learningRate = 0.01

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
  { _state :: RPState tr tr' slc s f h
  , _action :: RPAction slc tr s f h
  , _nextState :: Maybe (RPState tr tr' slc s f h)
  , _reward :: r
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
  :: ReplayBuffer tr tr' slc s f h r
  -> Int
  -> IO [ReplayStep tr tr' slc s f h r]
sampleSteps (ReplayBuffer _ queue) n = do
  -- not great, but shuffle' doesn't integrated with StatefulGen
  gen <- getStdRandom Rand.split
  pure $ take n (shuffle' queue (length queue) gen)

-- Encoding
-- --------

data QEncoding = QEncoding
  { _actionEncoding :: (Either SingleTop DoubleTop, Leftmost () () ())
  }

type PVAction = Action (Notes SPitch) (Edges SPitch) (Split SPitch) Freeze (Spread SPitch)

pitch2index :: GeneralSpec -> SPitch -> [Int]
pitch2index GeneralSpec{..} p = [fifths p - fifthLow, octaves p - octaveLow]

pitchesOneHot :: GeneralSpec -> HS.HashSet SPitch -> T.Tensor
pitchesOneHot spec@GeneralSpec{..} ps = T.toDense $ T.sparseCooTensor indices values dims opts
 where
  indices = T.transpose2D $ T.asTensor' (pitch2index spec <$> F.toList ps) (T.withDType T.Int64 opts)
  values = T.ones [F.length ps] opts
  dims = [fifthSize, octaveSize]

encodeSlice :: GeneralSpec -> Notes SPitch -> T.Tensor
encodeSlice spec (Notes notes) =
  -- DT.trace ("ecoding slice" <> show notes) $
  pitchesOneHot spec $ MS.toSet notes

encodeTransition :: GeneralSpec -> Edges SPitch -> T.Tensor
encodeTransition _spec (Edges _reg _pass) = T.asTensor @Double 0

type SingleTop = (StartStop T.Tensor, T.Tensor, StartStop T.Tensor)
type DoubleTop = (StartStop T.Tensor, T.Tensor, T.Tensor, T.Tensor, StartStop T.Tensor)

encodePVAction :: GeneralSpec -> PVAction -> (Either SingleTop DoubleTop, Leftmost () () ())
encodePVAction spec (Left (ActionSingle top action)) = (encTop, encAction)
 where
  (sl, GreedyParser.Trans t _2nd, sr) = top
  encTop = Left (encodeSlice spec <$> sl, encodeTransition spec t, encodeSlice spec <$> sr)
  encAction = case action of
    LMSingleFreeze FreezeOp -> LMFreezeOnly ()
    LMSingleSplit _split -> LMSplitOnly ()
encodePVAction spec (Right (ActionDouble top action)) = (encTop, encAction)
 where
  (sl, GreedyParser.Trans t1 _, sm, Trans t2 _, sr) = top
  encTop =
    Right
      ( encodeSlice spec <$> sl
      , encodeTransition spec t1
      , encodeSlice spec sm
      , encodeTransition spec t2
      , encodeSlice spec <$> sr
      )
  encAction = case action of
    LMDoubleFreezeLeft FreezeOp -> LMFreezeLeft ()
    LMDoubleSplitLeft split -> LMSplitLeft ()
    LMDoubleSplitRight split -> LMSplitRight ()
    LMDoubleSpread spread -> LMSpread ()

encodeStep :: GeneralSpec -> p -> PVAction -> QEncoding
encodeStep spec _ action = QEncoding (encodePVAction spec action)

-- Q net
-- -----

-- General Spec

data GeneralSpec = GeneralSpec
  { fifthLow :: Int
  , fifthSize :: Int
  , octaveLow :: Int
  , octaveSize :: Int
  , embSize :: Int
  }

defaultGSpec :: GeneralSpec
defaultGSpec =
  GeneralSpec
    { fifthLow = -3
    , fifthSize = 12
    , octaveLow = 2
    , octaveSize = 5
    , embSize = 64
    }

-- helper: ConstEmb

newtype ConstEmbSpec = ConstEmbSpec [Int]

newtype ConstEmb = ConstEmb T.Parameter
  deriving (Show, Generic)
  deriving anyclass (T.Parameterized)

instance T.Randomizable ConstEmbSpec ConstEmb where
  sample :: ConstEmbSpec -> IO ConstEmb
  sample (ConstEmbSpec shape) = do
    init <- T.randIO shape opts
    init' <- T.makeIndependent (T.subScalar @Double 1 $ T.mulScalar @Double 2.0 init)
    pure $ ConstEmb init'

instance T.HasForward ConstEmb () T.Tensor where
  forward (ConstEmb emb) () = T.toDependent emb
  forwardStoch model input = pure $ T.forward model input

-- Slice Encoder

newtype SliceSpec = SliceSpec {slcHidden :: Int}

data SliceEncoder = SliceEncoder
  { _slcL1 :: T.Linear
  , _slcL2 :: T.Linear
  , _slcStart :: ConstEmb
  , _slcStop :: ConstEmb
  }
  deriving (Show, Generic, T.Parameterized)

instance T.Randomizable (GeneralSpec, SliceSpec) SliceEncoder where
  sample :: (GeneralSpec, SliceSpec) -> IO SliceEncoder
  sample (GeneralSpec{..}, SliceSpec hidden) =
    SliceEncoder
      <$> (toOpts <$> T.sample (T.LinearSpec (fifthSize * octaveSize) hidden))
      <*> (toOpts <$> T.sample (T.LinearSpec hidden embSize))
      <*> T.sample (ConstEmbSpec [embSize])
      <*> T.sample (ConstEmbSpec [embSize])

instance T.HasForward SliceEncoder (StartStop T.Tensor) T.Tensor where
  forward model@(SliceEncoder _ _ start stop) input =
    case input of
      Inner slc -> T.forward model slc
      Start -> T.forward start ()
      Stop -> T.forward stop ()
  forwardStoch model input = pure $ T.forward model input

instance T.HasForward SliceEncoder T.Tensor T.Tensor where
  forward (SliceEncoder l1 l2 start stop) =
    T.relu . T.linear l2 . T.relu . T.linear l1 . T.flattenAll
  forwardStoch model = pure . T.forward model

-- Full Model

data SpecialSpec = SpecialSpec
  {finalSize :: Int}

data QSpec = QSpec GeneralSpec SpecialSpec SliceSpec

defaultSpec :: QSpec
defaultSpec =
  QSpec
    defaultGSpec
    SpecialSpec{finalSize = 128}
    SliceSpec{slcHidden = 64}

data QModel = QModel
  { _slc :: SliceEncoder
  , _final1 :: T.Linear
  , _final2 :: T.Linear
  }
  deriving (Show, Generic, T.Parameterized)

instance T.Randomizable QSpec QModel where
  sample :: QSpec -> IO QModel
  sample (QSpec gspec sspec slcspec) =
    QModel
      <$> T.sample (gspec, slcspec)
      <*> (toOpts <$> T.sample (T.LinearSpec (embSize gspec) (finalSize sspec)))
      <*> (toOpts <$> T.sample (T.LinearSpec (finalSize sspec) 1))

instance T.HasForward QModel QEncoding T.Tensor where
  forward :: QModel -> QEncoding -> T.Tensor
  forward (QModel slc final1 final2) (QEncoding (top, action)) =
    T.linear final2 $ T.relu $ T.linear final1 topEmb
   where
    topEmb = case top of
      Left (sl, _t, sr) -> T.forward slc sl + T.forward slc sr
      Right (sl, _t1, sm, _t2, sr) -> T.forward slc sl + T.forward slc sm + T.forward slc sr
  forwardStoch :: QModel -> QEncoding -> IO T.Tensor
  forwardStoch model input = pure $ T.forward model input

mkQModel :: QSpec -> IO QModel
mkQModel = T.sample

-- Reward
-- ------

inf :: Double
inf = 1 / 0

pvReward
  :: MWC.Gen RealWorld
  -> PVParams HyperRep
  -> PVAnalysis SPitch
  -> IO Double
pvReward gen hyper (Analysis deriv _top) = do
  let trace = observeDerivation' deriv
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

-- Deep Q-Learning
-- ---------------

data DQNState opt tr tr' slc s f h r = DQNState
  { qnet :: QModel
  , tnet :: QModel
  , opt :: opt
  , buffer :: ReplayBuffer tr tr' slc s f h r
  }

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
          , Analysis s f h tr slc
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
      Right (top, deriv) -> pure (addStep action Nothing transitions, Analysis deriv $ PathEnd top)
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

trainLoop
  :: forall tr tr' slc slc' s f h gen opt
   . ( StatefulGen gen IO
     , T.Optimizer opt
     )
  => gen
  -> Eval tr tr' slc slc' (Leftmost s f h)
  -> (GreedyState tr tr' slc (Leftmost s f h) -> Action slc tr s f h -> QEncoding)
  -> (Analysis s f h tr slc -> IO Double)
  -> Path slc' tr'
  -> DQNState opt tr tr' slc s f h Double
  -> IO (DQNState opt tr tr' slc s f h Double, Double)
trainLoop gen eval encode reward piece oldstate@(DQNState pnet tnet opt buffer) = do
  -- 1. run episode, collect results
  result <- runEpisode gen eval q 0.8 piece
  case result of
    -- error? skip
    Left error -> do
      print error
      pure (oldstate, 0)
    Right (steps, analysis) -> do
      -- 2. compute reward and add steps to replay buffer
      r <- reward analysis
      let steps' = case steps of
            [] -> []
            last : rest -> mkStep r last : fmap (mkStep 0) rest
          buffer' = F.foldl' pushStep buffer steps'
      -- 3. optimize models
      (pnet', tnet', opt') <- optimizeModels buffer'
      pure (DQNState pnet' tnet' opt' buffer', r)
 where
  q s a = T.asValue $ T.forward pnet $ encode s a
  mkStep r (state, action, state') =
    ReplayStep (RPState state) (RPAction action) (RPState <$> state') r

  -- A single optimization step for deep q learning (DQN)
  optimizeModels buffer' = do
    -- choose batch from replay buffer
    batch <- sampleSteps buffer' 10
    -- compute loss over batch
    let (qsNow, qsExpected) = unzip (dqnValues <$> batch)
        loss =
          T.smoothL1Loss
            T.ReduceMean
            (T.stack (T.Dim 0) qsNow)
            (T.stack (T.Dim 0) qsExpected)
    -- optimize policy net
    (pnet', opt') <- T.runStep pnet opt loss learningRate
    -- update target net
    let tparams = T.toDependent <$> T.flattenParameters tnet
        pparams = T.toDependent <$> T.flattenParameters pnet'
        interpolate p t = T.mulScalar tau p' + T.mulScalar (1 - tau) t'
         where
          p' = DT.traceShow (T.sumAll p) p
          t' = DT.traceShow (T.sumAll p) t
        tparams' = zipWith interpolate pparams tparams
    tparamsNew <- mapM T.makeIndependent tparams'
    let tnet' = T.replaceParameters tnet tparamsNew
    -- return new state
    pure (pnet', tnet', opt')

  -- The loss function of a single replay step
  dqnValues (ReplayStep (RPState s) (RPAction a) s' r) = (qnow, qexpected)
   where
    qzero = T.asTensor' [0 :: Double] opts
    qnext = case s' of
      Nothing -> qzero
      Just (RPState state') ->
        let
          nextQs = T.forward tnet . encode state' <$> getActions eval state'
         in
          E.maximumOn (T.asValue @Double) nextQs
    qnow = T.forward pnet (encode s a)
    qexpected = T.addScalar r (T.mulScalar gamma qnext)

-- delta = qnow - qexpected

trainDQN
  :: (StatefulGen gen IO, Foldable t)
  => gen
  -> Eval tr tr' slc slc' (Leftmost s f h)
  -> (GreedyState tr tr' slc (Leftmost s f h) -> Action slc tr s f h -> QEncoding)
  -> (Analysis s f h tr slc -> IO Double)
  -> t (Path slc' tr')
  -> Int
  -> IO [Double]
trainDQN gen eval encode reward pieces n = do
  model0 <- mkQModel defaultSpec
  let opt = T.mkAdam 0 0.9 0.99 (T.flattenParameters model0) -- T.GD
      buffer = mkReplayBuffer 10_000
      state0 = DQNState model0 model0 opt buffer
  (DQNState modelTrained _ _ _, rewards) <- T.foldLoop (state0, []) n trainEpoch
  pure rewards -- (modelTrained, rewards)
 where
  trainPiece (state, rewards) piece = do
    (state', r) <- trainLoop gen eval encode reward piece state
    pure (state', r : rewards)
  trainEpoch (state, meanRewards) i = do
    when ((i `mod` 10) == 0) $ putStrLn $ "epoch " <> show i
    (state', rewards) <-
      foldM trainPiece (state, []) pieces
    pure (state', T.asValue @Double (T.mean $ T.asTensor rewards) : meanRewards)
