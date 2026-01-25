{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ImpredicativeTypes #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE TypeFamilies #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}

module RL.Imitate where

import Common
import GreedyParser (ActionDouble (ActionDouble), ActionSingle (ActionSingle), GreedyState (..), getActions)
import Internal.MultiSet qualified as MS
import PVGrammar
import PVGrammar.Generate
  ( applyFreeze
  , applySplit
  , applySpread
  , freezable
  )
import PVGrammar.Parse
import PVGrammar.Prob.Simple
import Sample

import RL.Encoding (QEncoding, withBatchedEncoding)
import RL.ModelTypes

import Inference.Conjugate
import Musicology.Pitch (Interval (octave), IntervalClass (emb), SIC (SIC), SInterval (SInterval), SPitch, embed, embedP, fifth, fifth', major, minor, seventh, seventh', spc, third, third', unison, (+^), (^*))
import Musicology.Pitch qualified as MP

import Control.Monad (forM, forM_, replicateM, unless, when, zipWithM, zipWithM_)
import Control.Monad.Primitive (PrimMonad, PrimState)
import Control.Monad.Reader (MonadReader (..), ReaderT, lift, runReaderT)
import Control.Monad.State.Strict (MonadState (get), StateT (runStateT), evalStateT, execStateT, modify)
import Data.Aeson qualified as JSON
import Data.Either (lefts)
import Data.HashMap.Strict qualified as HM
import Data.HashSet qualified as S
import Data.Kind
import Data.List qualified as L
import Data.List.NonEmpty qualified as NE
import Data.Map.Strict qualified as M
import Data.Maybe (catMaybes, fromMaybe)
import Data.Proxy (Proxy (Proxy))
import Data.Text.IO (putStr)
import Data.Text.Lazy qualified as Txt
import Data.TypeNums (KnownNat, Nat, intVal)
import Data.Typeable (Proxy (Proxy), Typeable, typeRep)
import Data.Vector qualified as V
import Debug.Trace qualified as DT
import GHC.Generics
import Lens.Micro
import Lens.Micro.Extras (view)
import Statistics.Distribution qualified as Stats
import Statistics.Distribution.Poisson qualified as Stats
import System.ProgressBar qualified as PB
import System.Random.MWC.Probability (Gen, Prob (sample), binomial, categorical, createSystemRandom, discrete, discreteUniform, poisson, uniform)
import Torch qualified as T
import Torch.Typed qualified as TT

-- Helpers
-- =======

-- Discrete Distribution
-- ---------------------

-- type Discrete :: Type -> Nat -> Type
-- data Discrete a n = Discrete
--   deriving (Eq, Ord, Show, Generic)

-- instance Distribution (Discrete a n) where
--   type Params (Discrete a n) = [(Double, a)]
--   type Support (Discrete a n) = a
--   distSample _ = discrete
--   distLogP _ ps cat = log $ fromMaybe 0 $ ps L.!? cat

data Poisson = Poisson
  deriving (Eq, Ord, Show, Generic)

instance Distribution Poisson where
  type Params Poisson = Double
  type Support Poisson = Int
  distSample _ = poisson
  distLogP _ lambda k = Stats.logProbability (Stats.poisson lambda) k

data Choose (a :: Type) = Choose
  deriving (Eq, Ord, Show, Generic)

instance (Eq a) => Distribution (Choose a) where
  type Params (Choose a) = [a]
  type Support (Choose a) = a
  distSample _ = discreteUniform
  distLogP _ lst a = if a `L.elem` lst then log $ 1 / (fromIntegral $ length lst) else 0

-- Sampling Chords
-- ===============

chords :: [(QType, [SIC])]
chords =
  [ (0.5, [unison, major third', fifth'])
  , (0.3, [unison, minor third', fifth'])
  , (0.2, [unison, major third', fifth', minor seventh'])
  ]

sampleChordRoots :: (_) => m [Note SPitch]
sampleChordRoots = do
  -- choose chord
  chordType <- sampleConst "chordType" (Categorical @3) $ V.fromList (fst <$> chords)
  -- choose root (pragmatic: centered binomial distribution over fifths range)
  rootFifths <- (+ fifthLow) <$> sampleConst "rootFifths" (Binomial fifthSize) 0.5
  let root = spc rootFifths
  -- choose number of notes
  nnotes <- (+ 1) <$> sampleConst "chordNNotes" Poisson 3
  -- choose notes
  replicateM nnotes $ do
    -- pitch
    interval <- sampleConst "chordTone" Choose (snd $ chords !! chordType)
    octs <- fmap (`subtract` 2) $ sampleConst "chordToneOct" (Categorical @5) $ V.fromList [0.1, 0.2, 0.4, 0.2, 0.1]
    let pitch = (emb <$> (root +^ interval)) +^ (octave ^* (octs + 4))
    -- ID
    id <- sampleConst "chordRootID" (MagicalID "root") ()
    -- note
    pure $ Note pitch id
 where
  fifthSize = TT.natValI @FifthSize
  fifthLow = intValI @FifthLow

sampleChordRoots1 :: (_) => m [Note SPitch]
sampleChordRoots1 = do
  chordType <- sampleConst "chordType" (Categorical @3) $ V.fromList (fst <$> chords)
  rootFifths <- (+ fifthLow) <$> sampleConst "rootFifths" (Binomial fifthSize) 0.5
  let root = spc rootFifths
      ctones = snd $ chords !! chordType
  forM ctones $ \ctone -> do
    octs <- fmap (`subtract` 2) $ sampleConst "chordToneOct" (Categorical @5) $ V.fromList [0.1, 0.2, 0.4, 0.2, 0.1]
    let pitch = (emb <$> (root +^ ctone)) +^ (octave ^* (octs + 4))
    id <- sampleConst "chordRootID" (MagicalID "root") ()
    pure $ Note pitch id
 where
  fifthSize = TT.natValI @FifthSize
  fifthLow = intValI @FifthLow

makeTop :: [Note SPitch] -> (Path (Edges SPitch) (Notes SPitch), PVLeftmost SPitch)
makeTop notes = (top, LMSplitOnly op)
 where
  top = Path mempty (Notes $ S.fromList notes) $ PathEnd mempty
  op = mempty{splitReg = M.singleton (Start, Stop) $ mkRoot <$> notes}
  mkRoot note = (note, RootNote)

sampleChord :: (_) => m (Either String (PVAnalysis SPitch))
sampleChord = do
  roots <- sampleChordRoots1
  let (top, rootOp) = makeTop roots
  derivE <- sampleDerivation top
  pure $ do
    -- Either
    Analysis deriv _ <- derivE
    Right $ Analysis (rootOp : deriv) $ PathEnd topEdges

-- Derivations to Training Data
-- ============================

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

type ImitationDataX dev = forall r. (forall n. (KnownNat n) => QEncoding dev '[n] -> r) -> r
type ImitationDataY dev = T.Tensor
type ImitationData dev = (ImitationDataX dev, ImitationDataY dev)

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
  -> Either String [ImitationData dev]
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
    -> Either String (ImitationData dev)
  mkData op state = do
    let actionsAll = getActions (protoVoiceEvaluator @[] @[]) state
        maxActions = 1000
        actions = take maxActions actionsAll
    case actions of
      [] -> Left "no actions available!"
      (a : as) -> do
        let encoding :: ImitationDataX dev
            encoding = withBatchedEncoding state (a NE.:| as)
            target = fmap (eqOp op) actions
            !targetTensor = toQTensor' @dev $ target
        when ((length actions == maxActions) && (length (take (maxActions + 1) actionsAll) > maxActions)) $ do
          -- DT.traceM "too many actions in this state:"
          -- DT.traceShowM state
          Left $ "too many actions (more than " <> show maxActions <> ")!"
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
          DT.traceM $ show (length actions) <> " actions"
          Left "could not match any action!"
        Right (encoding, targetTensor)

-- Training on Derivations
-- =======================

sampleDerivationData
  :: forall dev
   . (TT.KnownDevice dev)
  => _model
  -> _gen
  -> Int
  -> Probs PVParams
  -> Int
  -> IO [ImitationData dev]
sampleDerivationData model gen maxN probs minSteps = goodData
 where
  goodData :: IO [ImitationData dev]
  goodData = do
    ana <- sampleUntilGood model gen maxN probs minSteps
    case derivationToDatapoints @dev ana of
      Left err -> do
        goodData
      Right dat -> pure dat

makeChordData :: forall dev. (TT.KnownDevice dev) => Int -> IO [ImitationData dev]
makeChordData n = do
  gen <- createSystemRandom
  Right hyper <- loadPVHyper "../posterior.json"
  let probs = expectedProbs @PVParams hyper
  putStrLn "Generating data."
  pb <-
    PB.newProgressBar
      ( PB.defStyle
          { PB.stylePrefix = (PB.elapsedTime PB.renderDuration)
          , PB.stylePostfix = PB.exact <> " (" <> PB.percentage <> ")"
          , PB.styleWidth = PB.ConstantWidth 80
          }
      )
      10
      (PB.Progress 0 n ())
  let samplePiece :: IO [ImitationData dev]
      samplePiece = do
        d <- sampleDerivationData @dev sampleChord gen 200 probs 4
        PB.incProgress pb 1
        pure d
  derivData <- replicateM n samplePiece
  pure $ concat derivData

-- Debugging
-- =========

writeRandomChords n minSteps = do
  gen <- createSystemRandom
  Right hyper <- loadPVHyper "../posterior.json"
  let probs = expectedProbs @PVParams hyper
  replicateMWithI n $ \i -> do
    ana <- sampleUntilGood sampleChord gen 200 probs minSteps
    print $ length $ anaDerivation ana
    JSON.encodeFile ("/tmp/rl/chord" <> show i <> ".analysis.json") ana

getRandomDeriv minSteps = do
  gen <- createSystemRandom
  Right hyper <- loadPVHyper "../posterior.json"
  let probs = expectedProbs @PVParams hyper
  sampleUntilGood sampleChord gen 200 probs minSteps

roundtripTestChords :: Int -> IO [(String, PVAnalysis SPitch)]
roundtripTestChords = roundtripTestDerivs' sampleChord

testDerivToData minSteps = do
  ana <- getRandomDeriv minSteps
  -- mapM_ print deriv
  print $ length $ anaDerivation ana
  case derivationToDatapoints @'(TT.CPU, 0) ana of
    Left err -> putStrLn err
    Right dat -> mapM_ print $ snd <$> dat

testDerivToDataMany minSteps n = do
  anas <- replicateM n $ getRandomDeriv minSteps
  let toData
        :: PVAnalysis SPitch
        -> Either (String, PVAnalysis SPitch) [ImitationData '(TT.CPU, 0)]
      toData ana = case derivationToDatapoints @'(TT.CPU, 0) ana of
        Left err -> Left (err, ana)
        Right a -> Right a
      datapoints = fmap toData anas
      errorCases = lefts datapoints
      doError (err, ana) i = do
        putStrLn err
        unless (take 3 err == "too") $
          JSON.encodeFile ("/tmp/rl/error" <> show i <> ".analysis.json") ana
  zipWithM_ doError errorCases [1 ..]
  putStrLn $ show (length errorCases) <> "/" <> show n <> " errors."

testDerivToDataFile fn = do
  anaE <- loadAnalysis fn
  case anaE of
    Left err -> putStrLn err
    Right ana -> case derivationToDatapoints @'(TT.CPU, 0) ana of
      Left err -> putStrLn err
      Right dat -> mapM_ print $ snd <$> dat
