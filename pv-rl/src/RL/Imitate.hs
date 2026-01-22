{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE TypeFamilies #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}

module RL.Imitate where

import Common
  ( Analysis
      ( Analysis
      , anaDerivation
      , anaTop
      )
  , Leftmost (..)
  , LeftmostDouble (..)
  , LeftmostSingle (..)
  , Path (..)
  , StartStop (..)
  , getInner
  )
import PVGrammar
import PVGrammar.Generate
  ( applySplit
  , applySpread
  , freezable
  )
import PVGrammar.Prob.Simple

import RL.ModelTypes

import Control.Monad (replicateM)
import Control.Monad.Primitive (PrimMonad, PrimState)
import Control.Monad.Reader (MonadReader (..), ReaderT, lift, runReaderT)
import Control.Monad.State.Strict (MonadState (get), StateT (runStateT), evalStateT, execStateT, modify)
import Data.Aeson qualified as JSON
import Data.HashSet qualified as S
import Data.Map.Strict qualified as M
import Data.Proxy (Proxy (Proxy))
import Data.TypeNums (intVal)
import Data.Typeable (Proxy (Proxy), Typeable, typeRep)
import Debug.Trace qualified as DT
import Inference.Conjugate
import Lens.Micro
import Lens.Micro.Extras (view)
import Musicology.Pitch (Interval (octave), IntervalClass (emb), SIC (SIC), SInterval (SInterval), SPitch, embed, embedP, fifth, fifth', major, minor, seventh, seventh', spc, third, third', unison, (+^), (^*))
import System.Random.MWC.Probability (Gen, Prob (sample), binomial, categorical, createSystemRandom, discrete, discreteUniform, poisson, uniform)
import Torch.Typed qualified as TT

-- Debugging
-- =========

newtype SampleLoudI m r a = SampleLoudI (ReaderT (r ProbsRep) (Prob m) a)
  deriving (Functor, Applicative, Monad)

instance (PrimMonad m) => RandomInterpreter (SampleLoudI m r) r where
  type
    SampleCtx (SampleLoudI m r) l =
      (Typeable (Support l), Typeable l, Show (Support l))
  sampleValue
    :: forall p l
     . (Conjugate p l, Typeable (Support l), Typeable l, Show (Support l))
    => String
    -> l
    -> Accessor r p
    -> SampleLoudI m r (Support l)
  sampleValue name lk getProbs = SampleLoudI $ do
    probs <- ask
    val <- lift $ distSample lk $ runProbs $ view getProbs probs
    let loc = show (typeRep (Proxy :: Proxy l)) <> " at " <> name
    DT.traceM $ "Sampled value " <> show val <> " from a " <> loc <> "."
    pure val
  sampleConst
    :: forall d
     . (Distribution d)
    => String
    -> d
    -> Params d
    -> SampleLoudI m r (Support d)
  sampleConst _ dist params = SampleLoudI $ lift $ distSample dist params
  permutationPlate = replicateMWithI

sampleLoud :: p ProbsRep -> SampleLoudI m p a -> Gen (PrimState m) -> m a
sampleLoud probs (SampleLoudI a) = sample (runReaderT a probs)

-- Modified Sampling
-- =================

newtype SampleSafeI m r a = SampleSafeI (ReaderT (Int, r ProbsRep, r ProbsRep) (StateT Int (Prob m)) a)
  deriving (Functor, Applicative, Monad)

instance (PrimMonad m) => RandomInterpreter (SampleSafeI m r) r where
  type SampleCtx (SampleSafeI m r) l = (Typeable (Support l), Typeable l, Show (Support l))
  sampleValue
    :: forall p l
     . (Conjugate p l, Typeable (Support l), Typeable l, Show (Support l))
    => String
    -> l
    -> Accessor r p
    -> SampleSafeI m r (Support l)
  sampleValue name lk getProbs = SampleSafeI $ do
    (threshold, probsA, probsB) <- ask
    i <- get
    let probs = if i < threshold then probsA else probsB
    val <- lift $ lift $ distSample lk $ runProbs $ view getProbs probs
    -- let loc = show (typeRep (Proxy :: Proxy l)) <> " at " <> name
    -- DT.traceM $ "(" <> show i <> ") Sampled value " <> show val <> " from a " <> loc <> "."
    modify (+ 1)
    pure val
  sampleConst
    :: forall d
     . (Distribution d)
    => String
    -> d
    -> Params d
    -> SampleSafeI m r (Support d)
  sampleConst _ dist params = SampleSafeI $ do
    modify (+ 1)
    lift $ lift $ distSample dist params
  permutationPlate = replicateMWithI

sampleSafe
  :: (Monad m)
  => Int
  -> p ProbsRep
  -> p ProbsRep
  -> SampleSafeI m p a
  -> Gen (PrimState m)
  -> m a
sampleSafe threshold probsA probsB (SampleSafeI a) = sample (evalStateT (runReaderT a (threshold, probsA, probsB)) 0)

-- Sampling Derivations
-- ====================

sampleExample :: (_) => _gen -> Hyper PVParams -> m (Either String [PVLeftmost SPitch])
sampleExample gen hyper = do
  let probs = expectedProbs @PVParams hyper
  sampleResult probs sampleDerivation' gen

makeTop :: [Note SPitch] -> (Path (Edges SPitch) (Notes SPitch), PVLeftmost SPitch)
makeTop notes = (top, LMSplitOnly op)
 where
  top = Path mempty (Notes $ S.fromList notes) $ PathEnd mempty
  op = mempty{splitReg = M.singleton (Start, Stop) $ mkRoot <$> notes}
  mkRoot note = (note, RootNote)

sampleChordRoots :: (_) => _gen -> m [Note SPitch]
sampleChordRoots gen = flip sample gen $ do
  -- choose chord
  chord <- discrete chords
  -- choose root (pragmatic: centered binomial distribution over fifths range)
  rootFifths <- (+ fifthLow) <$> binomial fifthSize 0.5
  let root = spc rootFifths
  -- choose number of notes
  nnotes <- (+ 1) <$> poisson 3
  -- choose notes
  replicateM nnotes $ do
    -- pitch
    interval <- discreteUniform chord
    octs <- (`subtract` 2) <$> categorical [0.1, 0.2, 0.4, 0.2, 0.1]
    let pitch = (emb <$> (root +^ interval)) +^ (octave ^* (octs + 4))
    -- ID
    i <- uniform @_ @Int
    let id = "root" <> show (abs i)
    -- note
    pure $ Note pitch id
 where
  fifthSize = TT.natValI @FifthSize
  fifthLow = fromIntegral $ intVal @FifthLow Proxy
  chords :: [(QType, [SIC])]
  chords =
    [ (0.5, [unison, major third', fifth'])
    , (0.3, [unison, minor third', fifth'])
    , (0.2, [unison, major third', fifth', minor seventh'])
    ]

sampleChord :: (_) => _gen -> Int -> Probs PVParams -> Probs PVParams -> m (Either String [PVLeftmost SPitch])
sampleChord gen maxN probs probsStop = do
  roots <- sampleChordRoots gen
  let (top, rootOp) = makeTop roots
  deriv <- sampleSafe maxN probs probsStop (sampleDerivation top) gen
  -- deriv <- sampleLoud probs (sampleDerivation top) gen
  -- (deriv, trace) <- sampleTrace probs (sampleDerivation top) gen
  -- !_ <- pure $ traceTrace trace (sampleDerivation top)
  pure $ fmap (rootOp :) deriv

writeRandomChords n minSteps = do
  gen <- createSystemRandom
  Right hyper <- loadPVHyper "../posterior.json"
  let probs = expectedProbs @PVParams hyper
  -- modified probabilities that ensure that a derivation terminates
  let !probsStop =
        probs
          & pOuter . pSingleFreeze .~ ProbsRep 1
          & pOuter . pDoubleLeft .~ ProbsRep 1
          & pOuter . pDoubleLeftFreeze .~ ProbsRep 1
          & pInner . pKeepL .~ ProbsRep 0
          & pInner . pKeepR .~ ProbsRep 0
          & pInner . pNewPassingLeft .~ ProbsRep 1
          & pInner . pNewPassingRight .~ ProbsRep 1
          & pInner . pNewPassingMid .~ ProbsRep 1
  -- samples a chord and ensures that the piece doesn't expand further after 100 decisions.
  let sampleGoodChord = do
        derivE <- sampleChord gen 200 probs probsStop
        case derivE of
          Left err -> do
            putStrLn err
            sampleGoodChord
          Right deriv ->
            if length deriv >= minSteps then pure deriv else sampleGoodChord
  replicateMWithI n $ \i -> do
    deriv <- sampleGoodChord
    print $ length deriv
    let ana :: PVAnalysis SPitch
        ana = Analysis deriv (PathEnd mempty)
    JSON.encodeFile ("/tmp/rl/chord" <> show i <> ".analysis.json") ana

-- Training on Derivations
-- =======================
