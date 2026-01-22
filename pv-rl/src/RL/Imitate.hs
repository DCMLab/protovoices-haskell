{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ImpredicativeTypes #-}
{-# LANGUAGE LambdaCase #-}
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
  , pathAppend
  )
import GreedyParser (ActionDouble (ActionDouble), ActionSingle (ActionSingle), GreedyState (..), getActions)
import PVGrammar
import PVGrammar.Generate
  ( applyFreeze
  , applySplit
  , applySpread
  , freezable
  )
import PVGrammar.Parse
import PVGrammar.Prob.Simple

import RL.Encoding (QEncoding, withBatchedEncoding)
import RL.ModelTypes

import Control.Monad (forM_, replicateM, when, zipWithM)
import Control.Monad.Primitive (PrimMonad, PrimState)
import Control.Monad.Reader (MonadReader (..), ReaderT, lift, runReaderT)
import Control.Monad.State.Strict (MonadState (get), StateT (runStateT), evalStateT, execStateT, modify)
import Data.Aeson qualified as JSON
import Data.HashMap.Strict qualified as HM
import Data.HashSet qualified as S
import Data.List (unfoldr)
import Data.List.NonEmpty qualified as NE
import Data.Map.Strict qualified as M
import Data.Proxy (Proxy (Proxy))
import Data.Text.IO (putStr)
import Data.TypeNums (KnownNat, intVal)
import Data.Typeable (Proxy (Proxy), Typeable, typeRep)
import Debug.Trace qualified as DT
import Inference.Conjugate
import Internal.MultiSet qualified as MS
import Lens.Micro
import Lens.Micro.Extras (view)
import Musicology.Core qualified as MC
import Musicology.Pitch (Interval (octave), IntervalClass (emb), SIC (SIC), SInterval (SInterval), SPitch, embed, embedP, fifth, fifth', major, minor, seventh, seventh', spc, third, third', unison, (+^), (^*))
import System.Random.MWC.Probability (Gen, Prob (sample), binomial, categorical, createSystemRandom, discrete, discreteUniform, poisson, uniform)
import Torch qualified as T
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
    let id = "root" <> show (mod i 1000)
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

makeStopProbs :: Probs PVParams -> Probs PVParams
makeStopProbs probs =
  probs
    & pOuter . pSingleFreeze .~ ProbsRep 1
    & pOuter . pDoubleLeft .~ ProbsRep 1
    & pOuter . pDoubleLeftFreeze .~ ProbsRep 1
    & pInner . pKeepL .~ ProbsRep 0
    & pInner . pKeepR .~ ProbsRep 0
    & pInner . pNewPassingLeft .~ ProbsRep 1
    & pInner . pNewPassingRight .~ ProbsRep 1
    & pInner . pNewPassingMid .~ ProbsRep 1

sampleGoodChord gen maxN probs minSteps = goodChord
 where
  probsStop = makeStopProbs probs
  goodChord = do
    derivE <- sampleChord gen maxN probs probsStop
    case derivE of
      Left err -> do
        putStrLn err
        goodChord
      Right deriv ->
        if length deriv >= minSteps then pure deriv else goodChord

writeRandomChords n minSteps = do
  gen <- createSystemRandom
  Right hyper <- loadPVHyper "../posterior.json"
  let probs = expectedProbs @PVParams hyper
  replicateMWithI n $ \i -> do
    deriv <- sampleGoodChord gen 200 probs minSteps
    print $ length deriv
    let ana :: PVAnalysis SPitch
        ana = Analysis deriv (PathEnd mempty)
    JSON.encodeFile ("/tmp/rl/chord" <> show i <> ".analysis.json") ana

getRandomDeriv minSteps = do
  gen <- createSystemRandom
  Right hyper <- loadPVHyper "../posterior.json"
  let probs = expectedProbs @PVParams hyper
  sampleGoodChord gen 200 probs minSteps

-- Training on Derivations
-- =======================

{- | auxiliary type that captures the result of applying a derivation operation
to a parse state.
-}
data OpResult tr tr' slc
  = ORFrozen tr'
  | OROpen (Path tr slc)
  | ORBoth tr' slc (Path tr slc)

{- | Turn a derivation into a sequence of parse states.
as they would have occured in a greedy parse that finds the given derivation.

The states are returned in order of the derivation,
i.e. the last parsing state is the first in the list.
The IDs of the notes are the ones used in the derivation,
not the ones that would have been produced by the greedy parser.
-}
derivationToParseStates :: PVAnalysis SPitch -> Either String [PVState]
derivationToParseStates (Analysis deriv top) = unfoldrM nextState state0
 where
  unfoldrM :: (Monad m) => (b -> m (Maybe (a, b))) -> b -> m [a]
  unfoldrM f s = do
    next <- f s
    case next of
      Nothing -> pure []
      Just (a, snext) -> (a :) <$> unfoldrM f snext

  state0 = GSOpen top deriv

  nextState :: PVState -> Either String (Maybe (PVState, PVState))
  nextState prev = fmap (fmap (\a -> (a, a))) $ nextState' prev

  -- Takes a derivation op and applies it to the current parse state.
  -- Returns a new state unless the derivation is completed.
  nextState' :: PVState -> Either String (Maybe PVState)
  nextState' = \case
    -- frozen state: done, no further ops
    GSFrozen _ -> Right Nothing
    -- open state: apply op to open segments
    GSOpen _ [] -> Right Nothing -- this is incomplete - return a Left or ignore?
    GSOpen open (op : ops') -> do
      result <- applyOp op open
      Right $ case result of
        ORFrozen frozen' -> Just $ GSFrozen (PathEnd (Just frozen')) -- might drop ops
        OROpen open' -> Just $ GSOpen open' ops'
        ORBoth frozen' mid' open' ->
          Just $ GSSemiOpen (PathEnd (Just frozen')) mid' open' ops'
    -- semi-open state: apply op to open segments
    GSSemiOpen _ _ _ [] -> Right Nothing -- this is incomplete - return a Left or ignore?
    GSSemiOpen frozen mid open (op : ops') -> do
      result <- applyOp op open
      Right $ case result of
        ORFrozen frozen' -> Just $ GSFrozen (Path (Just frozen') mid frozen) -- might drop ops
        OROpen open' -> Just $ GSSemiOpen frozen mid open' ops'
        ORBoth frozen' mid' open' ->
          Just $ GSSemiOpen (Path (Just frozen') mid frozen) mid' open' ops'

  -- Tries to apply a derivation op to the currently open segments.
  -- Returns the new updated open segments and potentially one new frozen segment
  applyOp
    :: PVLeftmost SPitch
    -> Path (Edges SPitch) (Notes SPitch)
    -> Either String (OpResult (Edges SPitch) [Edge SPitch] (Notes SPitch))
  applyOp op open = case open of
    -- a single transition
    PathEnd trans -> case op of
      LMDouble _ -> Left "Cannot apply a double operation to a single transition."
      LMFreezeOnly freezeOp -> do
        trFrozen <- applyFreeze freezeOp trans
        Right $ ORFrozen (S.toList trFrozen)
      LMSplitOnly splitOp -> do
        (trL, slc, trR) <- applySplit splitOp trans
        Right $ OROpen $ Path trL slc $ PathEnd trR

    -- two transitions
    Path transL slc (PathEnd transR) -> do
      (frozenMaybe, open') <- applyDouble op transL slc transR
      Right $ case frozenMaybe of
        Nothing -> OROpen open'
        Just (frozen', mid') -> ORBoth frozen' mid' open'

    -- more than two transitions
    Path transL slc (Path transR slc2 rest) -> do
      (frozenMaybe, open') <- applyDouble op transL slc transR
      Right $ case frozenMaybe of
        Nothing -> OROpen $ pathAppend open' slc2 rest
        Just (frozen', mid') -> ORBoth frozen' mid' $ pathAppend open' slc2 rest

  -- Tries to apply a double operation.
  -- Returns the open path and optionally a new frozen transition + new mid slice.
  applyDouble
    :: PVLeftmost SPitch
    -> Edges SPitch
    -> Notes SPitch
    -> Edges SPitch
    -> Either String (Maybe ([Edge SPitch], Notes SPitch), Path (Edges SPitch) (Notes SPitch))
  applyDouble op transL slc transR = case op of
    LMSingle _ -> Left "Cannot apply a single operation to two or more transitions."
    LMFreezeLeft freezeOp -> do
      trFrozen <- applyFreeze freezeOp transL
      Right (Just (S.toList trFrozen, slc), PathEnd transR)
    LMSplitLeft splitOp -> do
      (trL', slc', trR') <- applySplit splitOp transL
      Right (Nothing, Path trL' slc' $ Path trR' slc $ PathEnd transR)
    LMSplitRight splitOp -> do
      (trL', slc', trR') <- applySplit splitOp transR
      Right (Nothing, Path transL slc $ Path trL' slc' $ PathEnd trR')
    LMSpread spreadOp -> do
      (trL', slcL', trMid', slcR', trR') <- applySpread spreadOp transL slc transR
      Right (Nothing, Path trL' slcL' $ Path trMid' slcR' $ PathEnd trR')

{- | Compares two splits with potentially different orders of children.

Since children of the same parents are stored in lists,
their order might differ between two otherwise equal split operations.
This function converts these lists to sets to test the equivalence of the splits.
-}
eqSplit :: Split SPitch -> Split SPitch -> Bool
eqSplit a b =
  setEq splitReg
    && setEq splitPass
    && setEq fromLeft
    && setEq fromRight
    && eq keepLeft
    && eq keepRight
    && eq passLeft
    && eq passRight
 where
  eq :: (_) => (Split SPitch -> a) -> Bool
  eq acc = acc a == acc b
  setEq :: (_) => (Split SPitch -> M.Map k [v]) -> Bool
  setEq acc = setify (acc a) == setify (acc b)
  setify m = M.map S.fromList m

{- | Renames the note IDs of the parent notes
so that they correspond to the IDs generated in a parse.
-}
renameParentIDs :: Spread SPitch -> Spread SPitch
renameParentIDs (SpreadOp spreads edges) = SpreadOp spreads' edges
 where
  mkParent2 (Note p1 i1) (Note p2 i2) = Note p1 (i1 <> "+" <> i2)
  mkParent1 (Note p i) = Note p (i <> "'")
  rename (_, spread) = case spread of
    SpreadLeftChild l -> (mkParent1 l, spread)
    SpreadRightChild r -> (mkParent1 r, spread)
    SpreadBothChildren l r -> (mkParent2 l r, spread)
  spreads' = HM.fromList $ fmap rename $ HM.toList spreads

{- | Turns a derivation into a list of labelled datapoints (x,y)
that can be used for training.

States and their actions (x) are represented as a CPS closure
that takes a continuation with typed batch size, similar to 'withBatchedEncoding'.
The chosen action (y) is represented as a 1d one-hot tensor
of the size corresponding to the number of actions in that state.
-}
derivationToDatapoints
  :: forall dev
   . (TT.KnownDevice dev)
  => PVAnalysis SPitch
  -> Either String [(forall r. (forall n. (KnownNat n) => QEncoding dev '[n] -> r) -> r, T.Tensor)]
derivationToDatapoints analysis@(Analysis deriv top) = do
  states <- derivationToParseStates analysis
  zipWithM mkData deriv states
 where
  eqOp (LMSingle op) (Left (ActionSingle _ action)) = case (op, action) of
    (LMSingleFreeze fo, LMSingleFreeze fa) -> fo == fa
    (LMSingleSplit so, LMSingleSplit sa) -> eqSplit so sa
    _ -> False
  eqOp (LMDouble op) (Right (ActionDouble _ action)) = case (op, action) of
    (LMDoubleFreezeLeft fo, LMDoubleFreezeLeft fa) -> fo == fa
    (LMDoubleSplitLeft so, LMDoubleSplitLeft sa) -> eqSplit so sa
    (LMDoubleSplitRight so, LMDoubleSplitRight sa) -> eqSplit so sa
    (LMDoubleSpread ho, LMDoubleSpread ha) -> renameParentIDs ho == ha
    _ -> False
  eqOp _ _ = False

  mkData
    :: PVLeftmost SPitch
    -> PVState
    -> Either String (forall r. (forall n. (KnownNat n) => QEncoding dev '[n] -> r) -> r, T.Tensor)
  mkData op state = do
    let actions = take 200 $ getActions (protoVoiceEvaluator @[] @[]) state
    case actions of
      [] -> Left "no actions available!"
      (a : as) -> do
        let encoding :: forall r. (forall n. (KnownNat n) => QEncoding dev '[n] -> r) -> r
            encoding = withBatchedEncoding state (a NE.:| as)
            target = fmap (eqOp op) actions
            !targetTensor = toQTensor' @dev $ target
        when (not $ any id target) $ do
          DT.traceM "Couldn't match any action!\nstate:"
          DT.traceShowM state
          DT.traceM "actual step:"
          DT.traceM $ case op of
            LMSingle op' -> show op' <> "\n"
            LMDouble op' -> case op' of
              LMDoubleSpread spread -> show (renameParentIDs spread) <> "\n"
              a -> show a <> "\n"
          DT.traceM "available actions:"
          forM_ actions $ \action -> DT.traceM $ case action of
            Left (ActionSingle _ a) -> show a <> "\n"
            Right (ActionDouble _ a) -> show a <> "\n"

          error "breaking"
        Right (encoding, targetTensor)

testDerivToData minSteps = do
  deriv <- getRandomDeriv minSteps
  -- mapM_ print deriv
  print $ length deriv
  case derivationToDatapoints @'(TT.CPU, 0) (Analysis deriv $ PathEnd topEdges) of
    Left err -> putStrLn err
    Right dat -> mapM_ print $ snd <$> dat
