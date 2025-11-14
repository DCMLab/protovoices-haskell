module RL.Plotting where

import Common
import Control.Foldl qualified as Foldl
import Control.Monad (forM_)
import Control.Monad.State qualified as ST
import Data.Colour.Palette.ColorSet
import Display
import Graphics.Rendering.Chart.Backend.Cairo as Plt
import Graphics.Rendering.Chart.Easy ((.=))
import Graphics.Rendering.Chart.Easy qualified as Plt
import Graphics.Rendering.Chart.Gtk qualified as Plt
import Musicology.Pitch (SPitch)
import PVGrammar
import PVGrammar.Generate (derivationPlayerPV)
import RL.ModelTypes
import StrictList qualified as SL

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
  Plt.layout_legend .= Nothing
  forM_ (zip series [1 ..]) $ \(values, i) -> do
    let points = zip [1 :: Int ..] values
    Plt.plot $ Plt.line "" [points]

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
    Plt.plot $ do
      color <- Plt.takeColor
      histLine <- Plt.liftEC $ do
        Plt.plot_lines_values .= [points]
        Plt.plot_lines_title .= show i
        Plt.plot_lines_style . Plt.line_color .= color
      targetLine <- Plt.liftEC $ do
        Plt.plot_lines_values .= [[(1, target), (length values, target)]]
        Plt.plot_lines_style . Plt.line_color .= color
        Plt.plot_lines_style . Plt.line_dashes .= [10, 10]
      pure $ Plt.joinPlot (Plt.toPlot histLine) (Plt.toPlot targetLine)

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

plotDeriv :: (Foldable t) => FilePath -> t (Leftmost (Split SPitch) (Freeze SPitch) (Spread SPitch)) -> IO ()
plotDeriv fn deriv = do
  case replayDerivation derivationPlayerPV deriv of
    (Left err) -> putStrLn err
    (Right g) -> viewGraph fn g
