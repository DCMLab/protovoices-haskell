{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RebindableSyntax #-}

module Main where

import ChartParser
import Common
import Display
import PVGrammar hiding
  ( slicesFromFile
  , slicesToPath
  )
import PVGrammar.Generate
import PVGrammar.Parse

import Musicology.Core
import Musicology.Core.Slicing
import Musicology.MusicXML
import Musicology.Pitch.Spelled as MT

import Data.Maybe (catMaybes)
import qualified Data.Text.Lazy as TL
import qualified Data.Text.Lazy.IO as TL

-- better do syntax
import Language.Haskell.DoNotation
import Prelude hiding
  ( Monad (..)
  , pure
  )

-- | The example derivation shown in Figure 4, specified manually.
deriv321sus :: [PVLeftmost (Pitch MT.SIC)]
deriv321sus = buildDerivation $ do
  split $ mkSplit $ do
    splitT Start Stop (c' nat) RootNote False False
    splitT Start Stop (e' nat) RootNote False False
  spread $ mkSpread $ do
    spreadNote (c' nat) ToBoth True
    spreadNote (e' nat) (ToLeft 1) False
    addPassing (e' nat) (c' nat)
  freeze FreezeOp
  split $ mkSplit $ do
    splitT (Inner $ c' nat) (Inner $ c' nat) (b' nat) FullNeighbor True False
    splitNT (e' nat) (c' nat) (d' nat) PassingMid True False
  split $ mkSplit $ do
    splitT
      (Inner $ e' nat)
      (Inner $ d' nat)
      (d' nat)
      LeftRepeatOfRight
      False
      True
    splitT
      (Inner $ c' nat)
      (Inner $ b' nat)
      (c' nat)
      RightRepeatOfLeft
      True
      False
  freeze FreezeOp
  freeze FreezeOp
  freeze FreezeOp
  freeze FreezeOp

{- | The musical surface from Figure 4 as a sequence of slices and transitions.
 Can be used as an input for parsing.
-}
path321sus =
  Path [e' nat, c' nat] [(Inner $ c' nat, Inner $ c' nat)] $
    Path [d' nat, c' nat] [(Inner $ d' nat, Inner $ d' nat)] $
      Path [d' nat, b' nat] [] $
        PathEnd [c' nat]

{- | The main function that produces the results used in the paper and demonstrates the parser:
 * a diagram of the (manually specified) derivation of the suspension example
   (similar to what is shown in Figure 4)
   rendered to 321sus.{tex,pdf}
 * the number of derivations of the suspension example (Figure 4)
 * the number of derivations of the beginning of the Bach example (Figure 1)
 * an abritrary derivation of the suspension examle generated by the parser
   rendered to 321sus-parsed.{tex,pdf}
-}
main :: IO ()
main = do
  plotDeriv "321sus.tex" deriv321sus

  putStrLn "counting 321sus..."
  count321sus <- parseSilent pvCount path321sus

  putStrLn "parsing 321sus"
  parses321sus <- parseSilent pvDeriv path321sus
  let Just parse321 = firstDerivation parses321sus
  plotDeriv "321sus-parsed.tex" parse321

  putStrLn "counting Bach..."
  bachSlices <- slicesFromFile "testdata/allemande.musicxml"
  bachCount <- parseSize pvCount $ slicesToPath $ take 9 bachSlices

  putStrLn "Results:"
  putStrLn $ "number of derivations (321sus): " <> show count321sus
  putStrLn $ "number of derivations (bach): " <> show bachCount
  putStrLn "derivation of 321sus:"
  mapM_ print parse321

-- helper functions
-- ----------------

plotDeriv fn deriv = do
  case replayDerivation derivationPlayerPV deriv of
    (Left err) -> putStrLn err
    (Right g) -> viewGraph fn g

slicesFromFile :: FilePath -> IO [[(SPitch, RightTied)]]
slicesFromFile file = do
  txt <- TL.readFile file
  case parseWithoutIds txt of
    Nothing -> pure []
    Just doc -> do
      let (xmlNotes, _) = parseScore doc
          notes = asNoteHeard <$> xmlNotes
          slices = slicePiece tiedSlicer notes
      pure $ mkSlice <$> filter (not . null) slices
 where
  mkSlice notes = mkNote <$> notes
  mkNote (note, tie) = (pitch note, rightTie tie)

slicesToPath
  :: (Interval i, Ord (ICOf i), Eq i)
  => [[(Pitch i, RightTied)]]
  -> Path [Pitch (ICOf i)] [Edge (Pitch (ICOf i))]
slicesToPath = go
 where
  mkSlice = fmap (pc . fst)
  mkEdges notes = catMaybes $ mkEdge <$> notes
   where
    mkEdge (p, Ends) = Nothing
    mkEdge (p, Holds) = let p' = pc p in Just (Inner p', Inner p')
  go [] = error "cannot construct path from empty list"
  go [notes] = PathEnd (mkSlice notes)
  go (notes : rest) = Path (mkSlice notes) (mkEdges notes) $ go rest
