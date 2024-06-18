{-# LANGUAGE DataKinds #-}
{-# LANGUAGE Strict #-}

module RL.Common where

import Common
import Control.Foldl qualified as Foldl
import Control.Monad.Primitive (RealWorld)
import Control.Monad.State qualified as ST
import Display
import Graphics.Rendering.Chart.Backend.Cairo as Plt
import Graphics.Rendering.Chart.Easy ((.=))
import Graphics.Rendering.Chart.Easy qualified as Plt
import Graphics.Rendering.Chart.Gtk qualified as Plt
import GreedyParser (Action, ActionDouble (ActionDouble), ActionSingle (ActionSingle), DoubleParent (DoubleParent), SingleParent (SingleParent))
import Inference.Conjugate (Hyper, HyperRep, Prior (expectedProbs), evalTraceLogP, printTrace, sampleProbs)
import Internal.TorchHelpers
import Musicology.Pitch (SPitch)
import PVGrammar
import PVGrammar.Generate (derivationPlayerPV)
import PVGrammar.Prob.Simple
import RL.ModelTypes
import System.Random.MWC.Probability qualified as MWC
import Torch.Typed qualified as TT

-- helpers
-- -------

mean :: (Foldable t) => t QType -> QType
mean = Foldl.fold Foldl.mean

-- plotting
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
plotHistory title values = Plt.toFile Plt.def ("rl/" <> title <> ".png") $ mkHistoryPlot title values

plotDeriv :: (Foldable t) => FilePath -> t (Leftmost (Split SPitch) Freeze (Spread SPitch)) -> IO ()
plotDeriv fn deriv = do
  case replayDerivation derivationPlayerPV deriv of
    (Left err) -> putStrLn err
    (Right g) -> viewGraph fn g

-- Reward
-- ------

type PVAction = Action (Notes SPitch) (Edges SPitch) (Split SPitch) Freeze (Spread SPitch)

inf :: QType
inf = 1 / 0

pvRewardSample
  :: MWC.Gen RealWorld
  -> Hyper PVParams
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

pvRewardExp :: Hyper PVParams -> PVAnalysis SPitch -> IO QType
pvRewardExp hyper (Analysis deriv top) =
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
        Just (_, logprob) -> pure logprob
 where
  probs = expectedProbs @PVParams hyper
  trace = observeDerivation deriv top

pvRewardAction
  :: Hyper PVParams
  -> PVAction
  -> Maybe Bool
  -> IO QType
pvRewardAction hyper action decision = do
  case result of
    Left error -> do
      putStrLn $ "error giving reward: " <> error
      pure (-inf)
    Right Nothing -> do
      putStrLn "Couldn't evaluate trace while giving reward"
      pure (-inf)
    Right (Just (_, logprob)) -> pure logprob
 where
  probs = expectedProbs @PVParams hyper
  singleTop (SingleParent sl t sr) = (sl, t, sr)
  doubleTop (DoubleParent sl tl sm tr sr) = (sl, tl, sm, tr, sr)
  result = case action of
    Left (ActionSingle top op) -> evalSingleStep probs (singleTop top) op decision
    Right (ActionDouble top op) -> evalDoubleStep probs (doubleTop top) op decision
