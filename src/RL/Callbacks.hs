{-# LANGUAGE DataKinds #-}
{-# LANGUAGE Strict #-}

module RL.Callbacks where

import Common
import GreedyParser (Action, ActionDouble (ActionDouble), ActionSingle (ActionSingle), DoubleParent (DoubleParent), GreedyState, SingleParent (SingleParent), gsOps, opGoesLeft)
import PVGrammar
import PVGrammar.Prob.Simple

import RL.ModelTypes

import Inference.Conjugate (Hyper, Prior (expectedProbs), evalTraceLogP, sampleProbs)
import Musicology.Pitch (SPitch, fifths)

import Control.Monad.Primitive (RealWorld)
import Data.List.NonEmpty qualified as NE
import Data.Map.Strict qualified as M
import Data.Maybe (listToMaybe)
import System.Random.MWC.Probability qualified as MWC

-- Reward
-- ======

pvRewardSample
  :: MWC.Gen RealWorld
  -> Hyper PVParams
  -> PVRewardFn label
-- -> PVAnalysis SPitch
-- -> IO QType
pvRewardSample _ _ (Left _) (Just _) _ _ = pure 0
pvRewardSample _ _ (Left _) Nothing _ _ = pure (-inf)
pvRewardSample gen hyper (Right (top, deriv)) _ _ _ = do
  let trace = observeDerivation deriv (PathEnd top)
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

pvRewardExp :: Hyper PVParams -> PVRewardFn label -- PVAnalysis SPitch -> IO QType
pvRewardExp _ (Left _) (Just _) _ _ = pure 0
pvRewardExp _ (Left _) Nothing _ _ = pure (-inf)
pvRewardExp hyper (Right (top, deriv)) _ _ _ =
  pvRewardExp' hyper (Analysis deriv (PathEnd top))

pvRewardExp' :: Hyper PVParams -> PVAnalysis SPitch -> IO QType
pvRewardExp' hyper (Analysis deriv top) =
  case trace of
    Left err -> do
      putStrLn $ "error giving reward: " <> err
      print deriv
      error "error"
      pure (-inf)
    Right trace -> do
      case evalTraceLogP probs trace sampleDerivation' of
        Nothing -> do
          putStrLn "Couldn't evaluate trace while giving reward"
          pure (-inf)
        Just (_, logprob) -> pure $ logprob / fromIntegral (length deriv)
 where
  probs = expectedProbs @PVParams hyper
  trace = observeDerivation deriv top

pvRewardActionByLen
  :: Hyper PVParams -> PVRewardFn Int
-- -> PVAction
-- -> Maybe Bool
-- -> Int
-- -> IO QType
pvRewardActionByLen _ (Left _) Nothing _ _ = pure (-10)
pvRewardActionByLen hyper state _ action len = do
  case result of
    Left err -> do
      putStrLn $ "error giving reward: " <> err
      pure (-inf)
    Right Nothing -> do
      putStrLn "Couldn't evaluate trace while giving reward"
      pure (-inf)
    Right (Just (_, logprob)) -> pure $ logprob / fromIntegral len
 where
  probs = expectedProbs @PVParams hyper
  singleTop (SingleParent sl t sr) = (sl, t, sr)
  doubleTop (DoubleParent sl tl sm tr sr) = (sl, tl, sm, tr, sr)
  ops = case state of
    Left gs -> gsOps gs
    Right (_, deriv) -> deriv
  decision = opGoesLeft =<< listToMaybe (drop 1 ops)
  result = case action of
    Left (ActionSingle top op) -> evalSingleStep probs (singleTop top) op decision
    Right (ActionDouble top op) -> evalDoubleStep probs (doubleTop top) op decision

pvRewardChord :: PVRewardFn [Int]
pvRewardChord (Left _) Nothing _ _ = pure (-10)
pvRewardChord (Left _) (Just _) _ _ = pure 0
pvRewardChord _ _ _ [] = pure 0
pvRewardChord (Right (_, deriv)) _ _ expected = pure $
  case deriv of
    [] -> 0
    (op : _) -> case op of
      LMSplitOnly splt ->
        let rootNotes = fmap fst $ concat $ M.elems $ splitReg splt
            correctRoots = filter (`elem` expected) $ (fifths . notePitch) <$> rootNotes
         in fromIntegral (length correctRoots) / fromIntegral (length rootNotes)
      _ -> 0

addRewards :: QType -> PVRewardFn a -> PVRewardFn b -> PVRewardFn (a, b)
addRewards beta f1 f2 st acs ac (labela, labelb) = do
  r1 <- f1 st acs ac labela
  r2 <- f2 st acs ac labelb
  pure $ r1 + beta * r2

pvRewardChordAndActionByLen :: QType -> Hyper PVParams -> PVRewardFn (Int, [Int])
pvRewardChordAndActionByLen beta hyper =
  addRewards beta (pvRewardActionByLen hyper) pvRewardChord

-- Schedules
-- =========

cosSchedule :: QType -> QType -> QType
cosSchedule total i = (cos (t * pi) + 1) / 2
 where
  t = i / total

expSchedule :: QType -> QType -> QType -> QType -> QType
expSchedule start end total i = start * exp (log (end / start) * i / total)
