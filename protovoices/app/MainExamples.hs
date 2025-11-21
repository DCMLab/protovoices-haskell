{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE QualifiedDo #-}
{-# OPTIONS_GHC -Wno-all #-}

module Main where

import ChartParser
import Common
import Display
import PVGrammar
import PVGrammar.Generate
import PVGrammar.Parse

import Musicology.Core
import Musicology.Core.Slicing

-- import Musicology.Internal.Helpers
import Musicology.MusicXML
import Musicology.Pitch.Spelled as MT

import Data.Either (partitionEithers)
import Data.Maybe (catMaybes)
import Data.Ratio (Ratio (..))
import Lens.Micro (over)

import Control.Monad
  ( forM
  , forM_
  )
import Data.List qualified as L
import Data.Semiring qualified as R
import Data.Set qualified as S
import Data.Text qualified as T
import Data.Text.IO qualified as T
import Data.Typeable (Proxy (Proxy))

import Data.String (fromString)

-- better do syntax
import Language.Haskell.DoNotation qualified as Do

plotSteps :: FilePath -> [Leftmost s f h] -> IO ()
plotSteps fn deriv = do
  let graphs = unfoldDerivation derivationPlayerEmpty deriv
      (errors, steps) = partitionEithers graphs
  mapM_ putStrLn errors
  writeGraphs fn $ reverse steps

putGraph n deriv = case replayDerivation' n derivationPlayerEmpty deriv of
  (Left error) -> putStrLn error
  (Right g) -> T.putStrLn $ tikzPic $ tikzDerivationGraph showTexT showTexT g

plotDeriv fn deriv = do
  case replayDerivation derivationPlayerPV deriv of
    (Left err) -> putStrLn err
    (Right g) -> viewGraph fn g

example1 = buildDerivation $ Do.do
  split ()
  splitRight ()
  spread ()
  freeze ()
  split ()
  freeze ()
  freeze ()
  freeze ()
  freeze ()

example1Main :: IO ()
example1Main = do
  let Right g = replayDerivation derivationPlayerEmpty example1
  writeGraph "doc-images/monadic-deriv.tex" g

spreadSplitLeft = buildPartialDerivation @2 $ spread () Do.>> split ()
splitLeftSpread =
  buildPartialDerivation @2 $ split () Do.>> freeze () Do.>> spread ()

splitRightSpread = buildPartialDerivation @2 $ Do.do
  splitRight ()
  spread ()

exampleBoth = buildPartialDerivation @3 $ Do.do
  splitRight ()
  spread ()
  freeze ()
  freeze ()
  freeze ()
  spread ()

examplePartials = buildDerivation $ Do.do
  split ()
  spread ()
  split ()
  freeze ()
  spread ()

-- example derivations
-- ===================

derivBrahms :: [PVLeftmost (Pitch MT.SIC)]
derivBrahms = buildDerivation $ Do.do
  split $ mkSplit $ do
    splitRegular Start Stop "C#.rootC" RootNote False False
    splitRegular Start Stop "A.rootA" RootNote False False
  spread $ mkSpread $ do
    spreadNote "A.rootA" (ToBoth "spreadAl" "spreadAr") True
    spreadNote "C#.rootC" (ToLeft "spreadCl") False
    addPassing "C#.spreadCl" "A.spreadAr"
  splitRight $ mkSplit $ do
    splitPassing "C#.spreadCl" "A.spreadAr" "B.pass" PassingMid False False
    splitRegular "A.spreadAl" "A.spreadAl" "G#.nb" FullNeighbor False False
  spread $ mkSpread $ do
    spreadNote "A.spreadAl" (ToRight "A2") False
    spreadNote "C#.spreadCl" (ToLeft "C0") False
    addPassing "C#.C0" "A.A2"
  freeze $ mkFreeze []
  split $ mkSplit $ do
    splitPassing "C#.C0" "A.A2" "B.B1" PassingMid False False
  freeze $ mkFreeze []
  freeze $ mkFreeze []
  spread $ mkSpread $ do
    spreadNote "B.pass" (ToRight "B.B6") False
    spreadNote "G#.nb" (ToLeft "G#.G#4") False
  split $ mkSplit $ do
    addFromRight "G#.G#4" "A.A3" LeftNeighbor False
  freeze $ mkFreeze []
  freeze $ mkFreeze []
  split $ mkSplit $ do
    addFromRight "B.B6" "A.A5" LeftNeighbor False
  freeze $ mkFreeze []
  freeze $ mkFreeze []
  freeze $ mkFreeze []
  freeze $ mkFreeze []
 where
  (>>) :: (Do.BindSyntax x y z) => x a -> y b -> z b
  (>>) = (Do.>>)

-- Allemande
-- =========
-- This is too long to fix.
-- Better look at the corresponding annotated analysis in the data repo.

-- derivBach :: [PVLeftmost (Pitch MT.SIC)]
-- derivBach = buildDerivation $ Do.do
--   split $ mkSplit $ do
--     splitRegular Start Stop (d' nat) RootNote False False
--     splitRegular Start Stop (d' nat) RootNote False False
--     splitRegular Start Stop (f' nat) RootNote False False
--     splitRegular Start Stop (a' nat) RootNote False False
--     splitRegular Start Stop (a' nat) RootNote False False
--   spread $ mkSpread $ do
--     spreadNote (d' nat) ToBoth True
--     spreadNote (f' nat) ToBoth True
--     spreadNote (a' nat) (ToRight 1) True
--     addPassing (d' nat) (f' nat)
--   splitRight $ mkSplit $ do
--     splitPassing (d' nat) (f' nat) (e' nat) PassingMid True False
--     splitRegular (Inner $ d' nat) (Inner $ d' nat) (c' shp) FullNeighbor True True
--     splitRegular (Inner $ d' nat) (Inner $ d' nat) (d' nat) FullRepeat True True
--     splitRegular (Inner $ a' nat) (Inner $ a' nat) (a' nat) FullRepeat True True
--     splitRegular (Inner $ f' nat) (Inner $ f' nat) (g' nat) FullNeighbor True True
--   splitRight $ mkSplit $ do
--     splitRegular (Inner $ d' nat) (Inner $ d' nat) (d' nat) FullRepeat True True
--     splitRegular (Inner $ a' nat) (Inner $ a' nat) (b' flt) FullNeighbor False False
--     splitRegular
--       (Inner $ d' nat)
--       (Inner $ c' shp)
--       (c' shp)
--       LeftRepeatOfRight
--       False
--       True
--     splitRegular
--       (Inner $ d' nat)
--       (Inner $ e' nat)
--       (e' nat)
--       LeftRepeatOfRight
--       False
--       False
--     splitRegular
--       (Inner $ f' nat)
--       (Inner $ g' nat)
--       (g' nat)
--       LeftRepeatOfRight
--       False
--       True
--   spread $ mkSpread $ do
--     spreadNote (d' nat) ToBoth True
--     spreadNote (f' nat) (ToRight 1) False
--     spreadNote (a' nat) (ToRight 1) False
--   split $ mkSplit $ addFromRight (d' nat) (d' nat) LeftRepeat False
--   freeze $ mkFreeze []
--   freeze $ mkFreeze []
--   spread $ mkSpread $ do
--     spreadNote (d' nat) (ToRight 1) True
--     spreadNote (f' nat) (ToRight 1) False
--     spreadNote (a' nat) (ToLeft 1) False
--   spread $ mkSpread $ do
--     spreadNote (d' nat) ToBoth True
--     spreadNote (a' nat) ToBoth True
--   freeze $ mkFreeze []
--   split $ mkSplit $ do
--     splitRegular (Inner $ a' nat) (Inner $ a' nat) (b' flt) FullNeighbor False False
--     splitRegular (Inner $ a' nat) (Inner $ a' nat) (g' nat) FullNeighbor False False
--     splitRegular (Inner $ d' nat) (Inner $ d' nat) (d' nat) FullRepeat True True
--   spread $ mkSpread $ do
--     spreadNote (d' nat) ToBoth True
--     spreadNote (b' flt) (ToLeft 1) False
--     spreadNote (g' nat) (ToRight 1) False
--   freeze $ mkFreeze []
--   freeze $ mkFreeze []
--   freeze $ mkFreeze []
--   spread $ mkSpread $ do
--     spreadNote (d' nat) (ToRight 1) True
--     spreadNote (f' nat) (ToLeft 1) False
--     addPassing (f' nat) (d' nat)
--   freeze $ mkFreeze []
--   split $ mkSplit $ do
--     splitPassing (f' nat) (d' nat) (e' nat) PassingMid False False
--     splitRegular (Inner $ d' nat) (Inner $ d' nat) (d' nat) FullRepeat True True
--   freeze $ mkFreeze []
--   freeze $ mkFreeze []
--   spread $ mkSpread $ do
--     spreadNote (d' nat) ToBoth True
--     spreadNote (c' shp) ToBoth True
--     spreadNote (b' flt) (ToRight 1) False
--     spreadNote (e' nat) (ToRight 1) False
--     spreadNote (g' nat) (ToRight 1) False
--   freeze $ mkFreeze []
--   spread $ mkSpread $ do
--     spreadNote (d' nat) ToBoth True
--     spreadNote (c' shp) ToBoth True
--     spreadNote (b' flt) ToBoth True
--     spreadNote (e' nat) (ToRight 1) False
--     spreadNote (g' nat) (ToRight 1) False
--   freeze $ mkFreeze []
--   spread $ mkSpread $ do
--     spreadNote (d' nat) ToBoth True
--     spreadNote (c' shp) ToBoth True
--     spreadNote (b' flt) ToBoth True
--     spreadNote (e' nat) (ToLeft 1) False
--     spreadNote (g' nat) (ToRight 1) False
--   freeze $ mkFreeze []
--   freeze $ mkFreeze []
--   splitRight $ mkSplit $ do
--     splitRegular
--       (Inner $ g' nat)
--       (Inner $ f' nat)
--       (g' nat)
--       RightRepeatOfLeft
--       False
--       False
--     splitRegular (Inner $ a' nat) (Inner $ a' nat) (a' nat) FullRepeat True True
--     splitRegular (Inner $ d' nat) (Inner $ d' nat) (d' nat) FullRepeat True True
--     splitRegular
--       (Inner $ c' shp)
--       (Inner $ d' nat)
--       (d' nat)
--       LeftRepeatOfRight
--       False
--       True
--   spread $ mkSpread $ do
--     spreadNote (d' nat) ToBoth True
--     spreadNote (c' shp) ToBoth True
--     spreadNote (a' nat) ToBoth True
--     spreadNote (g' nat) ToBoth False
--     spreadNote (e' nat) (ToRight 1) False
--   freeze $ mkFreeze []
--   spread $ mkSpread $ do
--     spreadNote (d' nat) ToBoth True
--     spreadNote (c' shp) ToBoth True
--     spreadNote (a' nat) ToBoth True
--     spreadNote (g' nat) (ToLeft 1) False
--     spreadNote (e' nat) (ToRight 1) False
--     addPassing (g' nat) (e' nat)
--   freeze $ mkFreeze []
--   split $ mkSplit $ do
--     splitRegular (Inner $ d' nat) (Inner $ d' nat) (d' nat) FullRepeat True True
--     splitRegular (Inner $ a' nat) (Inner $ a' nat) (a' nat) FullRepeat True True
--     splitRegular (Inner $ c' shp) (Inner $ c' shp) (c' shp) FullRepeat True True
--     splitPassing (g' nat) (e' nat) (f' nat) PassingMid False False
--   freeze $ mkFreeze []
--   freeze $ mkFreeze []
--   freeze $ mkFreeze []
--   spread $ mkSpread $ do
--     spreadNote (d' nat) (ToLeft 1) False
--     spreadNote (f' nat) ToBoth False
--     spreadNote (a' nat) (ToLeft 2) False
--   spread $ mkSpread $ do
--     spreadNote (d' nat) (ToLeft 1) False
--     spreadNote (f' nat) (ToLeft 1) False
--     spreadNote (a' nat) (ToRight 1) True
--     addPassing (f' nat) (d' nat)
--   freeze $ mkFreeze []
--   split $ mkSplit $ do
--     splitPassing (f' nat) (d' nat) (e' nat) PassingMid False False
--     splitRegular (Inner $ a' nat) (Inner $ a' nat) (a' nat) FullRepeat True True
--     addFromRight (a' nat) (a' nat) LeftRepeat True
--   freeze $ mkFreeze []
--   freeze $ mkFreeze []
--   freeze $ mkFreeze []
--   freeze $ mkFreeze []

main = pure ()
