{-# LANGUAGE DataKinds #-}
{-# LANGUAGE Strict #-}

module RL.Common where

import Common
import Control.Foldl qualified as Foldl
import Control.Monad (forM_)
import Control.Monad.Primitive (RealWorld)
import Control.Monad.State qualified as ST
import Data.Colour.Palette.ColorSet
import Data.Colour.Palette.ColorSet (d3Colors2)
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
import StrictList qualified as SL
import System.Random.MWC.Probability qualified as MWC
import Torch.Typed qualified as TT

-- helpers
-- -------

mean :: (Foldable t) => t QType -> QType
mean = Foldl.fold Foldl.mean

zipWithStrict :: (a -> b -> c) -> SL.List a -> SL.List b -> SL.List c
zipWithStrict f SL.Nil _ = SL.Nil
zipWithStrict f _ SL.Nil = SL.Nil
zipWithStrict f (SL.Cons x xs) (SL.Cons y ys) = SL.Cons (f x y) $ zipWithStrict f xs ys

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

mkHistoriesPlot
  :: String
  -> [[QType]]
  -> ST.StateT
      (Plt.Layout Int QType)
      (ST.State Plt.CState)
      ()
mkHistoriesPlot title series = do
  Plt.setColors $
    Plt.opaque <$> (d3Colors2 Dark <$> [0 .. 9]) ++ (d3Colors2 Light <$> [0 .. 9])
  Plt.layout_title .= title
  forM_ (zip series [1 ..]) $ \(values, i) -> do
    let points = zip [1 :: Int ..] values
    Plt.plot $ Plt.line (show i) [points]

mkHistoryPlot'
  :: String
  -> QType
  -> [QType]
  -> ST.StateT
      (Plt.Layout Int QType)
      (ST.State Plt.CState)
      ()
mkHistoryPlot' title target values = do
  Plt.setColors $ Plt.opaque <$> [Plt.steelblue, Plt.orange]
  Plt.layout_title .= title
  Plt.plot $ Plt.line title [points]
  Plt.plot $ Plt.line "target" [[(1, target), (length values, target)]]
 where
  points = zip [1 :: Int ..] values

mkHistoriesPlot'
  :: String
  -> [QType]
  -> [[QType]]
  -> ST.StateT
      (Plt.Layout Int QType)
      (ST.State Plt.CState)
      ()
mkHistoriesPlot' title targets series = do
  Plt.setColors $
    Plt.opaque <$> (d3Colors2 Dark <$> [0 .. 9]) ++ (d3Colors2 Light <$> [0 .. 9])
  Plt.layout_title .= title
  forM_ (zip3 targets series [1 ..]) $ \(target, values, i) -> do
    let points = zip [1 :: Int ..] values
    Plt.plot $ Plt.line (show i) [points, [(1, target), (length values, target)]]

fileOpts :: Plt.FileOptions
fileOpts = Plt.def{_fo_format = Plt.SVG}

showHistory :: String -> [QType] -> IO ()
showHistory title values = Plt.toWindow 60 40 $ mkHistoryPlot title values

plotHistory :: String -> [QType] -> IO ()
plotHistory title values = Plt.toFile fileOpts ("rl/" <> title <> ".svg") $ mkHistoryPlot title values

plotHistories :: String -> [[QType]] -> IO ()
plotHistories title values = Plt.toFile fileOpts ("rl/" <> title <> ".svg") $ mkHistoriesPlot title values

plotHistory' :: String -> QType -> [QType] -> IO ()
plotHistory' title target values = Plt.toFile fileOpts ("rl/" <> title <> ".svg") $ mkHistoryPlot' title target values

plotHistories' :: String -> [QType] -> [[QType]] -> IO ()
plotHistories' title target values = Plt.toFile fileOpts ("rl/" <> title <> ".svg") $ mkHistoriesPlot' title target values

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
        Just (_, logprob) -> pure $ logprob / fromIntegral (length deriv)
 where
  probs = expectedProbs @PVParams hyper
  trace = observeDerivation deriv top

pvRewardAction
  :: Hyper PVParams
  -> PVAction
  -> Maybe Bool
  -> Int
  -> IO QType
pvRewardAction hyper action decision len = do
  case result of
    Left error -> do
      putStrLn $ "error giving reward: " <> error
      pure (-inf)
    Right Nothing -> do
      putStrLn "Couldn't evaluate trace while giving reward"
      pure (-inf)
    Right (Just (_, logprob)) -> pure $ logprob / fromIntegral (2 * len + 1)
 where
  probs = expectedProbs @PVParams hyper
  singleTop (SingleParent sl t sr) = (sl, t, sr)
  doubleTop (DoubleParent sl tl sm tr sr) = (sl, tl, sm, tr, sr)
  result = case action of
    Left (ActionSingle top op) -> evalSingleStep probs (singleTop top) op decision
    Right (ActionDouble top op) -> evalDoubleStep probs (doubleTop top) op decision
