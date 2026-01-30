{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE TypeFamilies #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}

module Sample where

import Common
import PVGrammar
import PVGrammar.Prob.Simple

import Inference.Conjugate
import Musicology.Pitch (SPitch)

import Control.Monad (replicateM, zipWithM)
import Control.Monad.Primitive (PrimMonad (..), RealWorld)
import Control.Monad.Reader
import Control.Monad.State
import Data.Aeson qualified as JSON
import Data.HashSet qualified as S
import Data.Map qualified as M
import Data.Maybe (catMaybes)
import Data.Typeable (Proxy (..), Typeable, typeRep)
import Debug.Trace qualified as DT
import Lens.Micro
import Lens.Micro.Extras (view)
import Pipes qualified as P
import Pipes.Prelude qualified as P
import System.Random.MWC.Probability (Gen, Prob (..), createSystemRandom)

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

sampleExample :: (_) => _gen -> Hyper PVParams -> m (Either String (PVAnalysis SPitch))
sampleExample gen hyper = do
  let probs = expectedProbs @PVParams hyper
  sampleResult probs sampleDerivation' gen

makeStopProbs :: Probs PVParams -> Probs PVParams
makeStopProbs probs =
  probs
    & pOuter
      . pSingleFreeze
      .~ ProbsRep 1
    & pOuter
      . pDoubleLeft
      .~ ProbsRep 1
    & pOuter
      . pDoubleLeftFreeze
      .~ ProbsRep 1
    & pInner
      . pKeepL
      .~ ProbsRep 0
    & pInner
      . pKeepR
      .~ ProbsRep 0
    & pInner
      . pNewPassingLeft
      .~ ProbsRep 1
    & pInner
      . pNewPassingRight
      .~ ProbsRep 1
    & pInner
      . pNewPassingMid
      .~ ProbsRep 1

sampleUntilGood
  :: SampleSafeI IO PVParams (Either String (PVAnalysis n))
  -> Gen RealWorld
  -> Int
  -> PVParams ProbsRep
  -> Int
  -> IO (PVAnalysis n)
sampleUntilGood model gen maxN probs minSteps = goodDeriv
 where
  probsStop = makeStopProbs probs
  goodDeriv = do
    derivE <- sampleSafe maxN probs probsStop model gen
    -- sampleChord gen maxN probs probsStop
    case derivE of
      Left err -> do
        putStrLn err
        goodDeriv
      Right deriv ->
        if length (anaDerivation deriv) >= minSteps then pure deriv else goodDeriv

sampleNSteps
  :: (Monad m)
  => P.Producer a (SampleI m p) r
  -> Gen (PrimState m)
  -> Int
  -> p ProbsRep
  -> Int
  -> m [a]
sampleNSteps producer gen n probs minSteps = sampleResult probs goodDeriv gen
 where
  goodDeriv = do
    (deriv, _) <- P.toListM' $ (producer >> pure ()) P.>-> P.take n
    if length deriv >= minSteps then pure deriv else goodDeriv

roundtripTestDerivs :: Int -> IO [(String, PVAnalysis SPitch)]
roundtripTestDerivs = roundtripTestDerivs' sampleDerivation'

roundtripTestDerivs' :: _ -> Int -> IO [(String, PVAnalysis SPitch)]
roundtripTestDerivs' model n = do
  gen <- createSystemRandom
  Right hyper <- loadPVHyper "../posterior.json"
  let probs = expectedProbs @PVParams hyper
  -- \$ uniformPrior @PVParams
  derivs <- replicateM n $ sampleUntilGood model gen 200 probs 0
  let errors = catMaybes $ fmap testDeriv derivs
  putStrLn $ show (length errors) <> " errors"
  zipWithM (\(_, ana) i -> JSON.encodeFile ("/tmp/rl/error" <> show i <> ".analysis.json") ana) errors [1 ..]
  pure errors
 where
  testDeriv ana = case roundtripTest ana of
    Left err -> Just (err, ana)
    Right _ -> Nothing
