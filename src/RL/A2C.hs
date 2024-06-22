{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE Strict #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}

module RL.A2C where

import Common
import Control.DeepSeq (NFData, force)
import Control.Foldl qualified as Foldl
import Control.Monad (foldM, forM_, when)
import Control.Monad.Except qualified as ET
import Control.Monad.Trans (lift)
import Control.Monad.Trans.Except qualified as ET
import Data.Either (partitionEithers)
import Data.Foldable qualified as F
import Data.Maybe (mapMaybe)
import Data.Vector qualified as V
import Debug.Trace qualified as DT
import GHC.Generics
import GreedyParser
import Inference.Conjugate (Hyper)
import Internal.TorchHelpers
import Musicology.Pitch (SPitch)
import NoThunks.Class (NoThunks (noThunks), ThunkInfo (thunkContext))
import PVGrammar
import PVGrammar.Prob.Simple (PVParams)
import RL.A2CHelpers
import RL.Common
import RL.Encoding
import RL.Model
import RL.ModelTypes
import StrictList qualified as SL
import System.Mem (performGC)
import System.Random.MWC.Distributions (categorical)
import System.Random.Stateful (StatefulGen)
import System.Random.Stateful qualified as Rand
import Torch qualified as T
import Torch.Typed qualified as TT
import Torch.Typed.Optim.CppOptim qualified as TT
import Torch.Typed.Optim.CppOptim qualified as TTC

-- global settings
-- ===============

-- discount factor
gamma :: QType
gamma = 0.99

-- eligibility decay factor (values)
lambdaV :: QType
lambdaV = 0.3

-- eligibility decay factor (policy)
lambdaP :: QType
lambdaP = 0.3

-- learning rate
learningRate :: TT.LearningRate QDevice QDType
learningRate = 0.001

nWorkers :: Int
nWorkers = 2

-- A2C
-- ===

printTensors :: TT.HList ModelTensors -> IO ()
printTensors (_ TT.:. t TT.:. _) = print t

printParams :: TT.HList ModelParams -> IO ()
printParams (_ TT.:. t TT.:. _) = print t

data A2CState = A2CState
  { a2cActor :: !(QModel DefaultQSpec)
  , a2cCritic :: !(QModel DefaultQSpec)
  , a2cOptActor :: !TT.GD -- !(TT.CppOptimizerState TT.AdamOptions ModelParams) -- !(TT.Adam ModelTensors) --
  , a2cOptCritic :: !TT.GD -- !(TT.Adam ModelTensors)
  }
  deriving (Generic)

-- data StepResult = StepResult
--   { stepState' :: !(Either (GreedyState (Edges SPitch) [Edge SPitch] (Notes SPitch) (PVLeftmost SPitch)) (Edges SPitch, [PVLeftmost SPitch]))
--   , stepGoLeft' :: !(Maybe Bool)
--   , stepZV' :: !(TT.HList ModelTensors)
--   , stepZP' :: !(TT.HList ModelTensors)
--   , stepDeltaV :: !(TT.HList ModelTensors)
--   , stepDeltaP :: !(TT.HList ModelTensors)
--   , stepDelta :: !(QTensor '[])
--   , stepRunninReward :: !QType
--   }

-- runStep eval gen hyper actor critic intensity (state, goleft, zV, zP, runningR) = do
--   -- EitherT String IO
--   -- preparation: list actions, compute policy
--   let actions = getActions eval state
--       encodings = encodeStep state <$> actions
--       raws = TT.toDynamic . T.forward actor <$> encodings
--       policy = T.softmax (T.Dim 0) $ T.cat (T.Dim 0) raws
--       actionProbs = V.fromList $ T.asValue $ T.toDType T.Double policy
--   -- lift $ print actions
--   -- lift $ print raws
--   -- lift $ print actionProbs
--   -- choose action according to policy
--   actionIndex <- lift $ categorical actionProbs gen
--   let action = actions !! actionIndex
--       goleft' = case action of
--         Right (ActionDouble _ op) -> case op of
--           LMDoubleFreezeLeft _ -> Just True
--           LMDoubleSplitLeft _ -> Just True
--           _ -> Just False
--         _ -> Nothing
--   -- apply action
--   state' <- ET.except $ applyAction state action
--   -- compute A2C update
--   r <- case state' of
--     Left _ -> pure 0
--     Right (top, deriv) -> lift $ pvRewardExp hyper (Analysis deriv (PathEnd top))
--   -- r <- lift $ pvRewardAction hyper action goleft
--   let vS = forwardValue critic $ encodePVState state
--       vS' :: QTensor '[1]
--       vS' = case state' of
--         Left s' -> forwardValue critic $ encodePVState s'
--         Right _ -> 0
--       delta = TT.addScalar r $ TT.squeezeAll $ TT.mulScalar gamma vS' - vS
--       gradV = TT.grad (TT.squeezeAll vS + fakeLoss critic) (TT.flattenParameters critic)
--       zV' = TT.hzipWith (UpdateEligCritic delta) zV gradV
--       actionLogProb = TT.log $ TT.UnsafeMkTensor (T.squeezeAll (policy T.! actionIndex))
--       gradP = TT.grad (actionLogProb + fakeLoss actor) (TT.flattenParameters actor)
--       zP' = TT.hzipWith (UpdateEligActor intensity delta) zP gradP
--       -- gradTotal = TT.hzipWith Add zV' zP'
--       -- deltaTotal = TT.hmap' (Mul $ TT.toDouble $ learningRate * delta) gradTotal
--       deltaV = TT.hmap' (Mul' delta) zV'
--       deltaP = TT.hmap' (Mul' delta) zP'
--       runningR' = runningR + r
--   -- lift $ do
--   --   putStr "vS = " >> print vS
--   --   putStr "vS' = " >> print vS'
--   --   putStr "r = " >> print r
--   --   putStr "delta = " >> print delta
--   --   putStr "deltaTotal = " >> print (sumTensorList deltaTotal)
--   pure $ StepResult state' goleft' zV' zP' deltaV deltaP delta runningR'

-- runEpisodeWorkers eval gen hyper input (A2CState actor critic opta optc) = ET.runExceptT $ go actor critic opta optc 1 [] states0 []
--  where
--   zeros :: TT.HList ModelTensors
--   zeros = TT.hmap' TT.ZerosLike $ TT.flattenParameters actor
--   states0 = replicate nWorkers (initParseState eval input, Nothing, zeros, zeros, 0)
--   sumTensors = F.foldl' (TT.hzipWith Add) zeros
--   nextState (StepResult state' goleft' zV' zP' _ _ delta r') =
--     case state' of
--       Left s' -> Left (s', goleft', zV', zP', r')
--       Right _ -> Right r'
--   go actor critic opta optc intensity losses states rewards = do
--     results <- mapM (runStep eval gen hyper actor critic intensity) states
--     let factor :: QType
--         factor = (-1) / fromIntegral (length states)
--         deltaV = TT.hmap' (Mul factor) $ sumTensors (stepDeltaV <$> results)
--         deltaP = TT.hmap' (Mul factor) $ sumTensors (stepDeltaP <$> results)
--     -- vparams = TT.hmap' TT.ToDependent $ TT.flattenParameters critic
--     -- pparams = TT.hmap' TT.ToDependent $ TT.flattenParameters actor
--     -- lift $ do
--     --   putStr "sum deltaV = " >> printTensors deltaV
--     --   putStr "sum deltaP = " >> printTensors deltaP
--     (actor', opta') <- lift $ TT.runStep' actor opta learningRate deltaP
--     (critic', optc') <- lift $ TT.runStep' critic optc learningRate deltaV
--     -- vparams' <- lift $ TT.hmapM' TT.MakeIndependent $ TT.hzipWith Add pparams deltaV
--     -- pparams' <- lift $ TT.hmapM' TT.MakeIndependent $ TT.hzipWith Add pparams deltaP
--     -- let critic' = TT.replaceParameters critic vparams'
--     --     actor' = TT.replaceParameters actor pparams'
--     --     opta' = opta
--     --     optc' = optc
--     let (states', newRewards) = partitionEithers $ nextState <$> results
--         rewards' = newRewards ++ rewards
--         intensity' = gamma * intensity
--         losses' = mean (TT.toDouble . stepDelta <$> results) : losses
--     case states' of
--       [] -> pure (A2CState actor critic opta' optc', mean rewards', mean losses')
--       _ -> go actor critic opta' optc' intensity' losses' states' rewards'

data A2CStepState = A2CStepState
  { a2cStepZV :: !(TT.HList ModelTensors)
  , a2cStepZP :: !(TT.HList ModelTensors)
  , a2cStepIntensity :: !QType
  , a2cStepReward :: !QType
  , a2cStepGoLeft :: !(Maybe Bool)
  , a2cStepState
      :: !( GreedyState
              (Edges SPitch)
              [Edge SPitch]
              (Notes SPitch)
              (Leftmost (Split SPitch) Freeze (Spread SPitch))
          )
  }

initPieceState
  :: Eval (Edges SPitch) [Edge SPitch] (Notes SPitch) [SPitch] (PVLeftmost SPitch)
  -> Path [SPitch] [Edge SPitch]
  -> TT.HList ModelTensors
  -> A2CStepState
initPieceState eval input z0 = A2CStepState z0 z0 1 0 Nothing $ initParseState eval input

pieceStep
  :: Eval (Edges SPitch) [Edge SPitch] (Notes SPitch) [SPitch] (PVLeftmost SPitch)
  -> Rand.IOGenM Rand.StdGen
  -> Hyper PVParams
  -> Int
  -> A2CState
  -> A2CStepState
  -> ET.ExceptT String IO (A2CState, Either A2CStepState QType, QType)
pieceStep eval gen hyper i (A2CState actor critic opta optc) (A2CStepState zV zP intensity reward goleft state) = do
  -- EitherT String IO
  -- preparation: list actions, compute policy
  let actions = getActions eval state
      encodings = encodeStep state <$> actions
      policy = T.softmax (T.Dim 0) $ T.cat (T.Dim 0) $ TT.toDynamic . forwardPolicy actor <$> encodings
  -- choose action according to policy
  actionIndex <- lift $ categorical (V.fromList $ T.asValue $ T.toDType T.Double policy) gen
  let action = actions !! actionIndex
      goleft' = case action of
        Right (ActionDouble _ op) -> case op of
          LMDoubleFreezeLeft _ -> Just True
          LMDoubleSplitLeft _ -> Just True
          _ -> Just False
        _ -> Nothing
  -- apply action
  state' <- ET.except $ applyAction state action
  -- compute A2C update
  -- r <- case state' of
  --   Left _ -> 0
  --   Right (top, deriv) -> lift $ reward (Analysis deriv (PathEnd top))
  r <- lift $ pvRewardAction hyper action goleft
  let vS = forwardValue critic $ encodePVState state
      vS' = case state' of
        Left s' -> forwardValue critic $ encodePVState s'
        Right _ -> 0
      delta = TT.addScalar r $ TT.squeezeAll $ TT.mulScalar gamma vS' - vS
      gradV = TT.grad (TT.squeezeAll vS + fakeLoss critic) (TT.flattenParameters critic)
      zV' = updateEligCritic gamma lambdaV zV gradV
      actionLogProb :: QTensor '[]
      actionLogProb = TT.log $ TT.UnsafeMkTensor (T.squeezeAll (policy T.! actionIndex))
      gradP = TT.grad (actionLogProb + fakeLoss actor) (TT.flattenParameters actor)
      zP' = updateEligActor gamma lambdaP intensity zP gradP
      --     gradTotal = TT.hzipWith Add zV' zP'
      intensity' = gamma * intensity
  --     deltaTotal = TT.hmap' (Mul $ TT.toDouble $ learningRate * delta) gradTotal
  --     params = TT.hmap' TT.ToDependent $ TT.flattenParameters model
  -- params' <- lift $ TT.hmapM' TT.MakeIndependent $ TT.hzipWith Add params deltaTotal
  -- let model' = TT.replaceParameters model params'
  --     opt' = opt
  (!actor', !opta') <- lift $ TT.runStep' actor opta (negate learningRate) $ mulModelTensors delta zP'
  -- (!actor', !opta') <- lift $ do
  --   advantage <- T.detach $ TT.toDynamic delta
  --   let lossP :: QTensor '[]
  --       lossP = negate actionLogProb `TT.mul` (TT.UnsafeMkTensor advantage :: QTensor '[])
  --   TTC.runStep actor opta lossP
  (!critic', !optc') <- lift $ TT.runStep' critic optc (negate learningRate) $ mulModelTensors delta zV'
  let loss' = T.asValue $ TT.toDynamic delta
      reward' = reward + r
  lift $ when ((i `mod` 100) == 0) $ do
    print state'
    putStr "r = " >> print r
    putStr "vS = " >> print vS
    putStr "vS' = " >> print vS'
    putStr "delta = " >> print delta
  --   putStr "gradV = " >> printTensors gradV
  --   putStr "zV' = " >> printTensors zV'
  --   putStr "log π = " >> print actionLogProb
  --   putStr "gradP = " >> printTensors gradP
  --   putStr "zP' = " >> printTensors zP'
  --   putStr "gradTotal = " >> printTensors gradTotal
  --   putStr "deltaTotal = " >> printTensors deltaTotal
  --   putStr "params' = " >> printParams params'
  --   putStr "I' = " >> print intensity'
  --   print $ qModelFinal2 model'
  let pieceState' = case state' of
        Left s' -> Left $ A2CStepState zV' zP' intensity' reward' goleft' s'
        Right _ -> Right reward' -- TT.toDouble (TT.squeezeAll vS) - r
  pure (A2CState actor' critic' opta' optc', pieceState', loss')

runEpisode
  :: (_)
  => Eval (Edges SPitch) [Edge SPitch] (Notes SPitch) [SPitch] (PVLeftmost SPitch)
  -> Rand.IOGenM Rand.StdGen
  -> Hyper PVParams
  -> Path [SPitch] [Edge SPitch]
  -> A2CState
  -> Int
  -> IO (Either String (A2CState, QType, QType))
runEpisode !eval !gen !hyper !input !modelState !i =
  ET.runExceptT $ go modelState (initPieceState eval input z0) SL.Nil
 where
  z0 :: TT.HList ModelTensors
  z0 = modelZeros $ a2cActor modelState
  go modelState pieceState losses = do
    (modelState', pieceState', loss) <- pieceStep eval gen hyper i modelState pieceState
    let losses' = loss `SL.Cons` losses
    case pieceState' of
      Left ps' -> go modelState' ps' losses'
      Right reward -> pure (modelState', reward, mean losses')

runEpisodes
  :: (_)
  => Eval (Edges SPitch) [Edge SPitch] (Notes SPitch) [SPitch] (PVLeftmost SPitch)
  -> Rand.IOGenM Rand.StdGen
  -> Hyper PVParams
  -> Path [SPitch] [Edge SPitch]
  -> A2CState
  -> Int
  -> IO (Either String (A2CState, QType, QType))
runEpisodes !eval !gen !hyper !input !modelState !i =
  ET.runExceptT $ go modelState states0 SL.Nil []
 where
  z0 :: TT.HList ModelTensors
  z0 = modelZeros $ a2cActor modelState
  -- initialize workers, each with a copy of the piece
  states0 = replicate nWorkers $ initPieceState eval input z0
  -- Worker folding function:
  -- The accumulator takes the current model state, a list of live piece states,
  -- list of losses and list of rewards.
  -- The element is the state of the current piece.
  -- Performs a single step forward on the piece, updating the model state and collecting loss.
  -- If the new piece state after the step is a terminal state,
  -- the reward is added to the reward list and the piece is dropped.
  -- If the new state is not terminal, it is added to the list of live piece states.
  iterWorker (ms, pss, ls, rs) ps = do
    (ms', ps'_, loss) <- pieceStep eval gen hyper i ms ps
    let ls' = loss `SL.Cons` ls
    pure $ case ps'_ of
      Left ps' -> (ms', ps' : pss, ls', rs)
      Right result -> (ms', pss, ls', result : rs)
  -- run the episode step by step
  go modelState pieceStates losses rewards = do
    (modelState', pieceStates', losses', rewards') <-
      foldM iterWorker (modelState, [], losses, rewards) pieceStates
    case pieceStates of
      [] -> pure (modelState', mean rewards', mean losses')
      _ -> go modelState' pieceStates' losses' rewards'

runAccuracy
  :: Eval (Edges SPitch) [Edge SPitch] (Notes SPitch) slc' (Leftmost (Split SPitch) Freeze (Spread SPitch))
  -> Hyper PVParams
  -> QModel DefaultQSpec
  -> Path slc' [Edge SPitch]
  -> IO (Either String (QType, Analysis (Split SPitch) Freeze (Spread SPitch) (Edges SPitch) slc))
runAccuracy !eval !hyper !actor !input = ET.runExceptT $ go 0 $ initParseState eval input
 where
  go cost state = do
    let actions = getActions eval state
        encodings = encodeStep state <$> actions
        probs = T.softmax (T.Dim 0) $ T.cat (T.Dim 0) $ TT.toDynamic . forwardPolicy actor <$> encodings
        best = T.asValue $ T.argmax (T.Dim 0) T.KeepDim probs
        action = actions !! best
        bestprob = probs T.! best
        cost' = cost + T.log bestprob
    state' <- ET.except $ applyAction state action
    lift $ print probs
    case state' of
      Left s' -> go cost' s'
      Right (top, deriv) -> do
        lift $ putStrLn $ "accuracy cost: " <> show cost'
        let ana = Analysis deriv (PathEnd top)
        r <- lift $ pvRewardExp hyper ana
        pure (r, ana)

deriving instance (NoThunks a) => NoThunks (SL.List a)

data A2CLoopState = A2CLoopState
  { a2clState :: A2CState
  , a2clRewards :: SL.List QType
  , a2clLosses :: SL.List QType
  , a2clAccs :: SL.List QType
  }
  deriving (Generic)

trainA2C
  :: Eval (Edges SPitch) [Edge SPitch] (Notes SPitch) [SPitch] (PVLeftmost SPitch)
  -> Rand.IOGenM Rand.StdGen
  -> Hyper PVParams
  -> QModel DefaultQSpec
  -> QModel DefaultQSpec
  -> [Path [SPitch] [Edge SPitch]]
  -> Int
  -> IO ([QType], [QType], QModel DefaultQSpec, QModel DefaultQSpec)
trainA2C eval gen hyper actor0 critic0 pieces n = do
  -- print $ qModelFinal2 model0
  -- opta <- TT.initOptimizer (TT.AdamOptions 0.0001 (0.9, 0.999) 1e-8 0 False) actor0
  let
    opta = TT.GD -- TT.mkAdam 0 0.9 0.99 (TT.flattenParameters actor0)
    optc = TT.GD -- TT.mkAdam 0 0.9 0.99 (TT.flattenParameters critic0)
    state0 = A2CState actor0 critic0 opta optc
  (A2CLoopState (A2CState actorTrained criticTrained _ _) rewards losses accs) <- T.foldLoop (A2CLoopState state0 SL.Nil SL.Nil SL.Nil) n trainEpoch
  pure (SL.toListReversed rewards, SL.toListReversed losses, actorTrained, criticTrained)
 where
  -- \| train a single episode on a single piece
  trainPiece i (!state, !rewards, !losses) !piece = do
    result <- runEpisode eval gen hyper piece state i
    case result of
      Left error -> do
        putStrLn $ "Episode error: " <> error
        pure (state, rewards, losses)
      Right (state', r, loss) -> do
        putStrLn $ "loss: " <> show loss
        pure (state', r `SL.Cons` rewards, loss `SL.Cons` losses)
  -- \| train one episode on each piece
  trainEpoch fullstate@(A2CLoopState !state !meanRewards !meanLosses !accuracies) !i = do
    -- performGC
    -- thunkCheck <- noThunks ["trainA2C", "trainEpoch"] fullstate
    -- case thunkCheck of
    --   Nothing -> pure ()
    --   Just thunkInfo -> error $ "Unexpected thunk at " <> show (thunkContext thunkInfo)
    -- run epoch
    (!state', !rewards, !losses) <-
      foldM (trainPiece i) (state, SL.Nil, SL.Nil) pieces
    let meanRewards' = mean rewards `SL.Cons` meanRewards
        meanLosses' = mean losses `SL.Cons` meanLosses
    -- compute greedy reward ("accuracy")
    accuracies' <-
      if (i `mod` 10) == 0
        then do
          results <- mapM (runAccuracy eval hyper (a2cActor state)) pieces
          case sequence results of
            Left error -> do
              putStrLn error
              pure $ (-inf) `SL.Cons` accuracies
            Right episodes -> do
              let accs = fst <$> episodes
                  analyses = snd <$> episodes
              when ((i `mod` 100) == 0) $ do
                putStrLn "current best analyses:"
                forM_ (zip analyses [1 ..]) $ \(Analysis deriv _, i) -> do
                  mapM_ print deriv
                  plotDeriv ("rl/deriv" <> show i <> ".tex") deriv
              pure $ mean accs `SL.Cons` accuracies
        else pure accuracies
    -- logging
    when ((i `mod` 10) == 0) $ do
      putStrLn $ "epoch " <> show i
      -- mapM_ print $ take 10 bcontent
      plotHistory "rewards" $ SL.toListReversed meanRewards'
      plotHistory "losses" $ SL.toListReversed meanLosses'
      plotHistory "accuracy" $ SL.toListReversed accuracies'
    -- print $ qModelFinal2 (a2cModel state)
    pure $ A2CLoopState state' meanRewards' meanLosses' accuracies'
