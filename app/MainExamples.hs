{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE RebindableSyntax #-}
{-# OPTIONS_GHC -Wno-all #-}
module Main where

import           Common
import           Display
import           PVGrammar
import           PVGrammar.Generate
import           PVGrammar.Parse
import           Parser

import           Musicology.Core
import           Musicology.Core.Slicing
--import Musicology.Internal.Helpers
import           Musicology.MusicXML
import           Musicology.Pitch.Spelled      as MT

import           Data.Either                    ( partitionEithers )
import           Data.Maybe                     ( catMaybes )
import           Data.Ratio                     ( Ratio(..) )
import           Lens.Micro                     ( over )

import           Control.Monad                  ( forM
                                                , forM_
                                                )
import qualified Data.List                     as L
import qualified Data.Semiring                 as R
import qualified Data.Set                      as S
import qualified Data.Text                     as T
import qualified Data.Text.IO                  as T
import           Data.Typeable                  ( Proxy(Proxy) )

-- better do syntax
import           Language.Haskell.DoNotation
import           Prelude                 hiding ( Monad(..)
                                                , pure
                                                )

plotSteps :: FilePath -> [Leftmost s f h] -> IO ()
plotSteps fn deriv = do
  let graphs          = unfoldDerivation derivationPlayerNull deriv
      (errors, steps) = partitionEithers graphs
  mapM_ putStrLn errors
  writeGraphs fn $ reverse steps

putGraph n deriv = case replayDerivation' n derivationPlayerNull deriv of
  (Left error) -> putStrLn error
  (Right g) -> T.putStrLn $ mkTikzPic $ tikzDerivationGraph showTex showTex g

plotDeriv fn deriv = do
  case replayDerivation derivationPlayerPV deriv of
    (Left  err) -> putStrLn err
    (Right g  ) -> viewGraph fn g

example1 = buildDerivation $ do
  split ()
  split ()
  freeze ()
  hori ()
  freeze ()
  freeze ()
  freeze ()

horiSplitLeft = buildPartialDerivation @2 $ hori () >> split ()
splitLeftHori = buildPartialDerivation @2 $ split () >> freeze () >> hori ()

splitRightHori = buildPartialDerivation @2 $ do
  splitRight ()
  hori ()

exampleBoth = buildPartialDerivation @3 $ do
  splitRight ()
  hori ()
  freeze ()
  freeze ()
  freeze ()
  hori ()

examplePartials = buildDerivation $ do
  split ()
  hori ()
  split ()
  freeze ()
  hori ()

derivBach :: [PVLeftmost (Pitch MT.SIC)]
derivBach = buildDerivation $ do
  split $ mkSplit $ do
    splitT Start Stop (d' nat) RootNote False False
    splitT Start Stop (d' nat) RootNote False False
    splitT Start Stop (f' nat) RootNote False False
    splitT Start Stop (a' nat) RootNote False False
    splitT Start Stop (a' nat) RootNote False False
  hori $ mkHori $ do
    horiNote (d' nat) ToBoth      True
    horiNote (f' nat) ToBoth      True
    horiNote (a' nat) (ToRight 1) True
    addPassing (d' nat) (f' nat)
  splitRight $ mkSplit $ do
    splitNT (d' nat) (f' nat) (e' nat) PassingMid True False
    splitT (Inner $ d' nat) (Inner $ d' nat) (c' shp) FullNeighbor True True
    splitT (Inner $ d' nat) (Inner $ d' nat) (d' nat) FullRepeat   True True
    splitT (Inner $ a' nat) (Inner $ a' nat) (a' nat) FullRepeat   True True
    splitT (Inner $ f' nat) (Inner $ f' nat) (g' nat) FullNeighbor True True
  splitRight $ mkSplit $ do
    splitT (Inner $ d' nat) (Inner $ d' nat) (d' nat) FullRepeat   True  True
    splitT (Inner $ a' nat) (Inner $ a' nat) (b' flt) FullNeighbor False False
    splitT (Inner $ d' nat)
           (Inner $ c' shp)
           (c' shp)
           LeftRepeatOfRight
           False
           True
    splitT (Inner $ d' nat)
           (Inner $ e' nat)
           (e' nat)
           LeftRepeatOfRight
           False
           False
    splitT (Inner $ f' nat)
           (Inner $ g' nat)
           (g' nat)
           LeftRepeatOfRight
           False
           True
  hori $ mkHori $ do
    horiNote (d' nat) ToBoth      True
    horiNote (f' nat) (ToRight 1) False
    horiNote (a' nat) (ToRight 1) False
  split $ mkSplit $ addToRight (d' nat) (d' nat) LeftRepeat False
  freeze FreezeOp
  freeze FreezeOp
  hori $ mkHori $ do
    horiNote (d' nat) (ToRight 1) True
    horiNote (f' nat) (ToRight 1) False
    horiNote (a' nat) (ToLeft 1)  False
  hori $ mkHori $ do
    horiNote (d' nat) ToBoth True
    horiNote (a' nat) ToBoth True
  freeze FreezeOp
  split $ mkSplit $ do
    splitT (Inner $ a' nat) (Inner $ a' nat) (b' flt) FullNeighbor False False
    splitT (Inner $ a' nat) (Inner $ a' nat) (g' nat) FullNeighbor False False
    splitT (Inner $ d' nat) (Inner $ d' nat) (d' nat) FullRepeat   True  True
  hori $ mkHori $ do
    horiNote (d' nat) ToBoth      True
    horiNote (b' flt) (ToLeft 1)  False
    horiNote (g' nat) (ToRight 1) False
  freeze FreezeOp
  freeze FreezeOp
  freeze FreezeOp
  hori $ mkHori $ do
    horiNote (d' nat) (ToRight 1) True
    horiNote (f' nat) (ToLeft 1)  False
    addPassing (f' nat) (d' nat)
  freeze FreezeOp
  split $ mkSplit $ do
    splitNT (f' nat) (d' nat) (e' nat) PassingMid False False
    splitT (Inner $ d' nat) (Inner $ d' nat) (d' nat) FullRepeat True True
  freeze FreezeOp
  freeze FreezeOp
  hori $ mkHori $ do
    horiNote (d' nat) ToBoth      True
    horiNote (c' shp) ToBoth      True
    horiNote (b' flt) (ToRight 1) False
    horiNote (e' nat) (ToRight 1) False
    horiNote (g' nat) (ToRight 1) False
  freeze FreezeOp
  hori $ mkHori $ do
    horiNote (d' nat) ToBoth      True
    horiNote (c' shp) ToBoth      True
    horiNote (b' flt) ToBoth      True
    horiNote (e' nat) (ToRight 1) False
    horiNote (g' nat) (ToRight 1) False
  freeze FreezeOp
  hori $ mkHori $ do
    horiNote (d' nat) ToBoth      True
    horiNote (c' shp) ToBoth      True
    horiNote (b' flt) ToBoth      True
    horiNote (e' nat) (ToLeft 1)  False
    horiNote (g' nat) (ToRight 1) False
  freeze FreezeOp
  freeze FreezeOp
  splitRight $ mkSplit $ do
    splitT (Inner $ g' nat)
           (Inner $ f' nat)
           (g' nat)
           RightRepeatOfLeft
           False
           False
    splitT (Inner $ a' nat) (Inner $ a' nat) (a' nat) FullRepeat True True
    splitT (Inner $ d' nat) (Inner $ d' nat) (d' nat) FullRepeat True True
    splitT (Inner $ c' shp)
           (Inner $ d' nat)
           (d' nat)
           LeftRepeatOfRight
           False
           True
  hori $ mkHori $ do
    horiNote (d' nat) ToBoth      True
    horiNote (c' shp) ToBoth      True
    horiNote (a' nat) ToBoth      True
    horiNote (g' nat) ToBoth      False
    horiNote (e' nat) (ToRight 1) False
  freeze FreezeOp
  hori $ mkHori $ do
    horiNote (d' nat) ToBoth      True
    horiNote (c' shp) ToBoth      True
    horiNote (a' nat) ToBoth      True
    horiNote (g' nat) (ToLeft 1)  False
    horiNote (e' nat) (ToRight 1) False
    addPassing (g' nat) (e' nat)
  freeze FreezeOp
  split $ mkSplit $ do
    splitT (Inner $ d' nat) (Inner $ d' nat) (d' nat) FullRepeat True True
    splitT (Inner $ a' nat) (Inner $ a' nat) (a' nat) FullRepeat True True
    splitT (Inner $ c' shp) (Inner $ c' shp) (c' shp) FullRepeat True True
    splitNT (g' nat) (e' nat) (f' nat) PassingMid False False
  freeze FreezeOp
  freeze FreezeOp
  freeze FreezeOp
  hori $ mkHori $ do
    horiNote (d' nat) (ToLeft 1) False
    horiNote (f' nat) ToBoth     False
    horiNote (a' nat) (ToLeft 2) False
  hori $ mkHori $ do
    horiNote (d' nat) (ToLeft 1)  False
    horiNote (f' nat) (ToLeft 1)  False
    horiNote (a' nat) (ToRight 1) True
    addPassing (f' nat) (d' nat)
  freeze FreezeOp
  split $ mkSplit $ do
    splitNT (f' nat) (d' nat) (e' nat) PassingMid False False
    splitT (Inner $ a' nat) (Inner $ a' nat) (a' nat) FullRepeat True True
    addToRight (a' nat) (a' nat) LeftRepeat True
  freeze FreezeOp
  freeze FreezeOp
  freeze FreezeOp
  freeze FreezeOp

main = pure ()
