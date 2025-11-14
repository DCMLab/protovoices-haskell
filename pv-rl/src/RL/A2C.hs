{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE Strict #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}

module RL.A2C where

import Common
import GreedyParser
import PVGrammar
import PVGrammar.Prob.Simple (PVParams)
import RL.A2CHelpers
import RL.Encoding
import RL.Model
import RL.ModelTypes
import RL.Plotting
import RL.TorchHelpers

import Control.DeepSeq (NFData, force)
import Control.Foldl qualified as Foldl
import Control.Monad (foldM, forM, forM_, when)
import Control.Monad.Except qualified as ET
import Control.Monad.Trans (lift)
import Control.Monad.Trans.Except qualified as ET
import Data.Either (partitionEithers)
import Data.Foldable qualified as F
import Data.List qualified as L
import Data.List.NonEmpty qualified as NE
import Data.Maybe (mapMaybe)
import Data.Text.Lazy qualified as Txt
import Data.Vector qualified as V
import Debug.Trace qualified as DT
import GHC.Generics
import Inference.Conjugate (Hyper)
import Musicology.Pitch (SPitch)
import NoThunks.Class (NoThunks (noThunks), ThunkInfo (thunkContext))
import StrictList qualified as SL
import System.IO (hFlush, stdout)
import System.Mem (performGC)
import System.ProgressBar qualified as PB
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
-- learningRate :: TT.LearningRate QDevice QDType
-- learningRate = 0.01

nWorkers :: Int
nWorkers = 2

-- A2C
-- ===

printTensors :: TT.HList ModelTensors -> IO ()
printTensors (_ TT.:. t TT.:. _) = print t

printParams :: TT.HList ModelParams -> IO ()
printParams (_ TT.:. t TT.:. _) = print t

data A2CState = A2CState
  { a2cActor :: !QModel
  , a2cCritic :: !QModel
  , a2cOptActor :: !TT.GD -- !(TT.CppOptimizerState TT.AdamOptions ModelParams) -- !(TT.Adam ModelTensors) --
  , a2cOptCritic :: !TT.GD -- !(TT.Adam ModelTensors)
  }
  deriving (Generic)

data A2CStepState = A2CStepState
  { a2cStepZV :: !(TT.HList ModelTensors)
  , a2cStepZP :: !(TT.HList ModelTensors)
  , a2cStepIntensity :: !QType
  , a2cStepReward :: !QType
  , a2cStepState
      :: !( GreedyState
              (Edges SPitch)
              [Edge SPitch]
              (Notes SPitch)
              (PVLeftmost SPitch)
          )
  , a2cStepActions :: !(NE.NonEmpty PVAction)
  }

initPieceState
  :: Eval (Edges SPitch) [Edge SPitch] (Notes SPitch) [Note SPitch] (Spread SPitch) (PVLeftmost SPitch)
  -> Path [Note SPitch] [Edge SPitch]
  -> TT.HList ModelTensors
  -> Either A2CStepState QType
initPieceState eval input z0 =
  let
    state = initParseState eval input
    actions = take 200 $ getActions eval state
   in
    case actions of
      [] -> Right (-inf)
      (a : as) -> Left $ A2CStepState z0 z0 1 0 state (a NE.:| as)

pieceStep
  :: Eval (Edges SPitch) [Edge SPitch] (Notes SPitch) [Note SPitch] (Spread SPitch) (PVLeftmost SPitch)
  -> Rand.IOGenM Rand.StdGen
  -> PVRewardFn label
  -> label
  -> QType
  -- ^ learning rate
  -> QType
  -- ^ temperature
  -> Int
  -- ^ iteration
  -> A2CState
  -> A2CStepState
  -> ET.ExceptT String IO (A2CState, Either A2CStepState QType, QType)
pieceStep eval gen fReward len lr temp i (A2CState actor critic opta optc) (A2CStepState zV zP intensity reward state actions) = do
  -- EitherT String IO
  -- preparation: list actions, compute policy
  -- TODO: smarter cap than taking 200 actions
  let
    -- encodings = encodeStep state <$> actions
    -- policy = T.softmax (T.Dim 0) $ T.cat (T.Dim 0) $ TT.toDynamic . forwardPolicy actor <$> encodings
    policy = T.pow (1 / temp) $ withBatchedEncoding state actions (runBatchedPolicy actor)
  -- choose action according to policy
  actionIndex <- lift $ categorical (V.fromList $ T.asValue $ T.toDType T.Double policy) gen
  let action = actions NE.!! actionIndex
  -- apply action
  state' <- ET.except $ applyAction state action
  let actions' = case state' of
        Left newState -> NE.nonEmpty $ take 200 $ getActions eval newState
        Right _ -> Nothing
  -- compute A2C update
  r <- lift $ fReward state' actions' action len
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
      intensity' = gamma * intensity
      learningRate = toQTensor (negate lr)
  (!actor', !opta') <- lift $ TT.runStep' actor opta learningRate $ mulModelTensors delta zP'
  (!critic', !optc') <- lift $ TT.runStep' critic optc learningRate $ mulModelTensors delta zV'
  let loss' = T.asValue $ TT.toDynamic delta
      reward' = reward + r
  let pieceState' = case (state', actions') of
        (Left s', Just a') -> Left $ A2CStepState zV' zP' intensity' reward' s' a'
        (Left s', Nothing) ->
          -- DT.trace ("incomplete parse:\n" <> show s') $
          Right reward'
        (Right _, _) -> Right reward' -- TT.toDouble (TT.squeezeAll vS) - r
  pure (A2CState actor' critic' opta' optc', pieceState', loss')

-- | Run an episode
runEpisode
  :: (_)
  => Eval (Edges SPitch) [Edge SPitch] (Notes SPitch) [Note SPitch] (Spread SPitch) (PVLeftmost SPitch)
  -> Rand.IOGenM Rand.StdGen
  -> PVRewardFn label
  -> (QType -> QType)
  -> (QType -> QType)
  -> Path [Note SPitch] [Edge SPitch]
  -> label
  -> A2CState
  -> Int
  -> IO (Either String (A2CState, QType, QType))
runEpisode !eval !gen !fReward !fLr !fTemp !input !label !modelState !i =
  case initPieceState eval input z0 of
    Left s0 -> ET.runExceptT $ go modelState s0 SL.Nil
    Right reward -> pure $ pure (modelState, reward, 0)
 where
  z0 :: TT.HList ModelTensors
  z0 = modelZeros $ a2cActor modelState
  lr = fLr $ fromIntegral i
  temp = fTemp $ fromIntegral i
  -- len = pathLen input
  go modelState pieceState losses = do
    (modelState', pieceState', loss) <- pieceStep eval gen fReward label lr temp i modelState pieceState
    let losses' = loss `SL.Cons` losses
    case pieceState' of
      Left ps' -> go modelState' ps' losses'
      Right reward -> pure (modelState', reward, mean losses')

runAccuracy
  :: Eval (Edges SPitch) [Edge SPitch] (Notes SPitch) slc' (Spread SPitch) (PVLeftmost SPitch)
  -> PVRewardFn label
  -> QModel
  -> (Path slc' [Edge SPitch], label)
  -> IO (Either String (QType, PVAnalysis SPitch))
runAccuracy !eval !fReward !actor (!input, !label) = case take 200 $ getActions eval s0 of
  [] -> pure $ Left "cannot parse: no possible actions for first step!"
  (a : as) -> ET.runExceptT $ go 0 0 s0 (a NE.:| as)
 where
  s0 = initParseState eval input
  go !cost !reward !state !actions = do
    let
      -- encodings = encodeStep state <$> actions
      -- probs = T.softmax (T.Dim 0) $ T.cat (T.Dim 0) $ TT.toDynamic . forwardPolicy actor <$> encodings
      probs = withBatchedEncoding state actions (runBatchedPolicy actor)
      best = T.asValue $ T.argmax (T.Dim 0) T.KeepDim probs
      action = actions NE.!! best
      bestprob = probs T.! best
      cost' = cost + T.log bestprob
    state' <- ET.except $ applyAction state action
    let actions' = case state' of
          Left newState -> NE.nonEmpty $ take 200 $ getActions eval newState
          Right _ -> Nothing
    actionReward <- lift $ fReward state' actions' action label
    let reward' = reward + actionReward
    -- lift $ print probs
    case (state', actions') of
      (Left _, Nothing) ->
        ET.throwE "cannot parse: no possible actions in non-terminal state!"
      (Left s', Just a') -> go cost' reward' s' a'
      (Right (top, deriv), _) -> do
        lift $ putStrLn $ "accuracy cost: " <> show cost'
        when (T.asValue cost == (0 :: Double)) $ do
          lift $ putStrLn $ show bestprob
        let ana = Analysis deriv (PathEnd top)
        pure (reward', ana)

deriving instance (NoThunks a) => NoThunks (SL.List a)

data A2CLoopState = A2CLoopState
  { a2clState :: A2CState
  , a2clRewards :: SL.List (SL.List QType)
  , a2clLosses :: SL.List (SL.List QType)
  , a2clAccs :: SL.List (SL.List QType)
  }
  deriving (Generic)

trainA2C
  :: Eval (Edges SPitch) [Edge SPitch] (Notes SPitch) [Note SPitch] (Spread SPitch) (PVLeftmost SPitch)
  -> Rand.IOGenM Rand.StdGen
  -> PVRewardFn label
  -> (QType -> QType)
  -- ^ learning rate schedule
  -> (QType -> QType)
  -- ^ temperature schedule
  -> Maybe [QType]
  -> QModel
  -> QModel
  -> [(Path [Note SPitch] [Edge SPitch], label)]
  -> Int
  -> IO ([[QType]], [QType], QModel, QModel)
trainA2C eval gen fReward fLr fTemp targets actor0 critic0 pieces n = do
  -- print $ qModelFinal2 model0
  -- opta <- TT.initOptimizer (TT.AdamOptions 0.0001 (0.9, 0.999) 1e-8 0 False) actor0
  let
    opta = TT.GD -- TT.mkAdam 0 0.9 0.99 (TT.flattenParameters actor0)
    optc = TT.GD -- TT.mkAdam 0 0.9 0.99 (TT.flattenParameters critic0)
    emptyStat = SL.fromListReversed (replicate (length pieces) SL.Nil)
    state0 = A2CState actor0 critic0 opta optc
  (A2CLoopState (A2CState actorTrained criticTrained _ _) rewards losses accs) <- T.foldLoop (A2CLoopState state0 emptyStat emptyStat emptyStat) n trainEpoch
  pure
    ( SL.toListReversed $ SL.toListReversed <$> rewards
    , SL.toListReversed $ mean <$> losses
    , actorTrained
    , criticTrained
    )
 where
  -- \| train a single episode on a single piece
  trainPiece pb i (!state, !rewards, !losses) ((!piece, label), !j) = do
    !result <- runEpisode eval gen fReward fLr fTemp piece label state i
    PB.incProgress pb 1
    case result of
      Left error -> do
        putStrLn $ "Episode error: " <> error
        pure (state, rewards, losses)
      Right (state', r, loss) -> do
        -- putStrLn $ "loss " <> show j <> ": " <> show loss
        pure (state', r `SL.Cons` rewards, loss `SL.Cons` losses)
  -- \| train one episode on each piece
  trainEpoch fullstate@(A2CLoopState !state !rewardsHist !lossHist !accuracies) !i = do
    -- putStrLn $ "\nepoch " <> show i
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
    -- performGC
    -- thunkCheck <- noThunks ["trainA2C", "trainEpoch"] fullstate
    -- case thunkCheck of
    --   Nothing -> pure ()
    --   Just thunkInfo -> error $ "Unexpected thunk at " <> show (thunkContext thunkInfo)
    -- run epoch
    (!state', !rewards, !losses) <-
      foldM (trainPiece pb i) (state, SL.Nil, SL.Nil) (zip pieces [1 ..])
    let rewardsHist' = zipWithStrict SL.Cons rewards rewardsHist
        lossHist' = zipWithStrict SL.Cons losses lossHist
    -- compute greedy reward ("accuracy")
    accuracies' <-
      if (i `mod` 10) == 0
        then do
          results <- mapM (runAccuracy eval fReward (a2cActor state)) pieces
          newAccs <- forM (zip results [1 ..]) $ \(result, j) ->
            case result of
              Left error -> do
                putStrLn error
                pure (-inf)
              Right (acc, Analysis deriv top) -> do
                when ((i `mod` 100) == 0) $ do
                  putStrLn $ "current best analysis (piece " <> show j <> "):"
                  mapM_ print deriv
                -- plotDeriv ("rl/deriv" <> show j <> ".tex") deriv
                pure acc
          pure $ zipWithStrict SL.Cons (SL.fromListReversed newAccs) accuracies
        else pure accuracies
    -- logging
    when ((i `mod` 1) == 0) $ do
      -- putStrLn $ "epoch " <> show i
      -- mapM_ print $ take 10 bcontent
      let rews = SL.toListReversed <$> SL.toListReversed rewardsHist'
          accs = SL.toListReversed <$> SL.toListReversed accuracies'
          losses = SL.toListReversed <$> SL.toListReversed lossHist'
          avgReward = mean <$> L.transpose rews
          avgAbsLoss = (mean . fmap abs) <$> L.transpose losses
      plotHistories "losses" losses
      case targets of
        Nothing -> do
          plotHistories "rewards" rews
          plotHistories "accuracy" accs
        Just ts -> do
          plotHistories' "rewards" ts rews
          plotHistories' "accuracy" ts accs
      plotHistory "mean_reward" avgReward
      plotHistory "mean_loss" avgAbsLoss
      -- print $ qModelFinal2 (a2cModel state)
      TT.save (TT.hmap' TT.ToDependent $ TT.flattenParameters $ a2cActor state) "actor_checkpoint.ht"
      TT.save (TT.hmap' TT.ToDependent $ TT.flattenParameters $ a2cCritic state) "critic_checkpoint.ht"
    pure $ A2CLoopState state' rewardsHist' lossHist' accuracies'
