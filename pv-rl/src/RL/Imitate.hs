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

import RL.Encoding
import RL.Model
import RL.ModelTypes
import RL.Plotting

import Inference.Conjugate
import Musicology.Pitch (Interval (octave), IntervalClass (emb), SIC (SIC), SInterval (SInterval), SPitch, embed, embedP, fifth, fifth', major, minor, seventh, seventh', spc, third, third', unison, (+^), (^*))
import Musicology.Pitch qualified as MP

import Control.Monad (foldM, forM, forM_, replicateM, unless, when, zipWithM, zipWithM_)
import Control.Monad.Cont (ContT (ContT, runContT))
import Control.Monad.Primitive (PrimMonad, PrimState, RealWorld)
import Control.Monad.Reader (MonadReader (..), ReaderT, lift, runReaderT)
import Control.Monad.State.Strict (MonadState (get), StateT (runStateT), evalStateT, execStateT, modify)
import Data.Aeson qualified as JSON
import Data.Bifunctor (Bifunctor (bimap))
import Data.Either (lefts, rights)
import Data.HashMap.Strict qualified as HM
import Data.HashSet qualified as HS
import Data.Kind
import Data.List qualified as L
import Data.List.NonEmpty qualified as NE
import Data.Map.Strict qualified as M
import Data.Maybe (catMaybes, fromMaybe)
import Data.Proxy (Proxy (Proxy))
import Data.Set qualified as S
import Data.Text.Lazy qualified as Txt
import Data.TypeNums (KnownNat, Nat, intVal)
import Data.Typeable (Proxy (Proxy), Typeable, typeRep)
import Data.Vector qualified as V
import Debug.Trace qualified as DT
import GHC.Generics
import Lens.Micro
import Lens.Micro.Extras (view)
import PVGrammar (topEdges)
import PVGrammar.Prob.Simple (produceDerivation)
import Pipes qualified as P
import Pipes.Prelude qualified as P
import RL.ModelTypes (IsValidDevice)
import Sample (sampleNSteps)
import Statistics.Distribution qualified as Stats
import Statistics.Distribution.Poisson qualified as Stats
import System.ProgressBar qualified as PB
import System.Random qualified as Rand
import System.Random.MWC.Probability (Gen, Prob (sample), binomial, categorical, createSystemRandom, discrete, discreteUniform, poisson, uniform)
import Torch (nllLoss')
import Torch qualified as T
import Torch.Typed qualified as TT

-- Helpers
-- =======

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
  top = Path mempty (Notes $ HS.fromList notes) $ PathEnd mempty
  op = mempty{splitReg = M.singleton (Start, Stop) $ mkRoot <$> notes}
  mkRoot note = (note, RootNote)

sampleChord :: (_) => m (Either String (PVAnalysis SPitch))
sampleChord = do
  roots <- sampleChordRoots
  let (top, rootOp) = makeTop roots
  derivE <- sampleDerivation top
  pure $ do
    -- Either
    Analysis deriv _ <- derivE
    Right $ Analysis (rootOp : deriv) $ PathEnd topEdges

produceChord :: (_) => P.Producer (PVLeftmost SPitch) m (Either String ())
produceChord = do
  roots <- lift $ sampleChordRoots
  let (top, rootOp) = makeTop roots
  P.yield rootOp
  produceDerivation top

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
        Right $ ORFrozen (HS.toList trFrozen)
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
      Right (Just (HS.toList trFrozen, slc), PathEnd transR)
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
  setify m = M.map HS.fromList m

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

-- type ImitationDataX dev = forall r. (forall n. (KnownNat n) => QEncoding dev '[n] -> r) -> r
type ImitationDataX dev = (PVState, NE.NonEmpty PVAction) -- QEncoding dev '[FakeSize]
type ImitationDataY dev = T.Tensor
data ImitationData dev = ImitationData
  { dataInput :: !(ImitationDataX dev)
  , dataLabel :: !(ImitationDataY dev)
  }
  deriving (Show)

{- | Turns a derivation into a list of labelled datapoints (x,y)
that can be used for training.

States and their actions (x) are represented as a CPS closure
that takes a continuation with typed batch size, similar to 'withBatchedEncoding'.
The chosen action (y) is represented as a Ax1 one-hot tensor
of the size corresponding to the number of actions in that state.
-}
derivationToDatapoints
  :: forall dev
   . (TT.KnownDevice dev)
  => PVAnalysis SPitch
  -> Either String [ImitationData dev]
derivationToDatapoints analysis@(Analysis deriv top) = do
  states <- derivationToParseStates analysis
  dataMaybe <- zipWithM stateToDatapoint deriv states
  pure $ catMaybes dataMaybe

derivationToDatapointsLenient
  :: forall dev
   . (TT.KnownDevice dev)
  => PVAnalysis SPitch
  -> Either String [ImitationData dev]
derivationToDatapointsLenient analysis@(Analysis deriv top) = do
  states <- derivationToParseStates analysis
  let dataMaybe = rights $ zipWith stateToDatapoint deriv states
  pure $ catMaybes dataMaybe

stateToDatapoint
  :: forall dev
   . (TT.KnownDevice dev)
  => PVLeftmost SPitch
  -> PVState
  -> Either String (Maybe (ImitationData dev))
stateToDatapoint op !state = do
  let actionsAll = getActions (protoVoiceEvaluator @[] @[]) state
      maxActions = 100
      actions = take maxActions actionsAll
      !lenA = length actions
  if lenA == 1
    then Right Nothing -- filter out
    else
      if (lenA == maxActions) && (length (take (maxActions + 1) actionsAll) > maxActions)
        then do
          -- DT.traceM "too many actions in this state:"
          -- DT.traceShowM state
          Left $ "too many actions (more than " <> show maxActions <> ")!"
        else do
          case actions of
            [] -> Left "no actions available!"
            (!a : as) -> do
              let encoding :: ImitationDataX dev
                  !encoding = (state, (a NE.:| as)) -- encodeStepFake state (a NE.:| as) -- withBatchedEncoding state (a NE.:| as)
              !target <- case L.findIndex (eqOp op) actions of
                Nothing -> do
                  -- DT.traceM "Couldn't match any action!\nstate:"
                  -- DT.traceShowM state
                  -- DT.traceM "actual step:"
                  -- DT.traceM $ case op of
                  --   LMSingle op' -> show op' <> "\n"
                  --   LMDouble op' -> case op' of
                  --     LMDoubleSpread spread -> show (renameParentIDs spread) <> "\n"
                  --     a -> show a <> "\n"
                  -- DT.traceM "available actions:"
                  -- forM_ actions $ \action -> DT.traceM $ case action of
                  --   Left (ActionSingle _ a) -> show a <> "\n"
                  --   Right (ActionDouble _ a) -> show a <> "\n"
                  -- DT.traceM $ show (length actions) <> " actions"
                  Left "could not match any action!"
                Just ix -> Right $ T.toDevice (TT.deviceVal @dev) $ T.asTensor [ix]
              Right $! Just $! ImitationData encoding target
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

-- Making Data
-- ===========

-- sampleDerivationData
--   :: forall dev
--    . (TT.KnownDevice dev)
--   => _model
--   -> _gen
--   -> Int
--   -> Probs PVParams
--   -> Int
--   -> IO [ImitationData dev]
-- sampleDerivationData model gen maxN probs minSteps = goodData
--  where
--   goodData :: IO [ImitationData dev]
--   goodData = do
--     ana <- sampleUntilGood model gen maxN probs minSteps
--     case derivationToDatapoints @dev ana of
--       Left err -> do
--         goodData
--       Right dat -> pure dat

sampleDerivationData'
  :: forall dev r
   . (TT.KnownDevice dev)
  => P.Producer (PVLeftmost SPitch) _m r
  -> _gen
  -> Int
  -> Probs PVParams
  -> Int
  -> IO [ImitationData dev]
sampleDerivationData' producer gen maxN probs minSteps = goodData
 where
  goodData :: IO [ImitationData dev]
  goodData = do
    deriv <- sampleNSteps producer gen maxN probs minSteps
    case derivationToDatapoints @dev (Analysis deriv $ PathEnd topEdges) of
      Left err -> goodData
      Right dat -> pure dat

makeChordData :: forall dev. (TT.KnownDevice dev) => Int -> IO [ImitationData dev]
makeChordData n = do
  gen <- createSystemRandom
  Right hyper <- loadPVHyper "posterior.json"
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
        d <- sampleDerivationData' @dev produceChord gen 32 probs 4
        PB.incProgress pb 1
        pure d
  derivData <- replicateM n samplePiece
  pure $ concat derivData

newtype ImitationDataset (m :: Type -> Type) dev = ImitationDataset (V.Vector (ImitationData dev))

instance Show (ImitationDataset m dev) where
  show _ = "ImitationDataset"

instance (Applicative m, TT.KnownDevice dev) => T.Dataset m (ImitationDataset m dev) Int (ImitationData dev) where
  getItem (ImitationDataset samples) ix = pure $ samples V.! ix
  keys (ImitationDataset samples) = S.fromList [0 .. V.length samples - 1]

mkImitationDataset :: [ImitationData dev] -> ImitationDataset m dev
mkImitationDataset samples = ImitationDataset $ V.fromList samples

makeChordDataset :: forall dev. (TT.KnownDevice dev) => Int -> IO (ImitationDataset IO dev)
makeChordDataset n = do
  samples <- makeChordData n
  pure $ mkImitationDataset samples

data ImitationStream dev = ImitationStream
  { imsProbs :: (Probs PVParams)
  , imsMinLen :: Int
  , imsMaxSteps :: Int
  , imsGen :: Gen RealWorld
  }

instance (TT.KnownDevice dev) => T.Datastream IO () (ImitationStream dev) (ImitationData dev) where
  streamSamples (ImitationStream probs minLen maxSteps gen) () = do
    samples <- P.Select $ P.repeatM $ sampleDerivationData' @dev produceChord gen maxSteps probs minLen
    P.Select $ P.each samples

-- Training
-- ========

nll :: T.Tensor -> T.Tensor -> T.Tensor
nll label pred =
  -- DT.trace info $
  T.nllLoss' label pred
 where
  info = "label: " <> show label <> "\npred: " <> show pred

hit :: T.Tensor -> T.Tensor -> Double
hit label pred =
  -- DT.trace info $
  if predIx == label then 1 else 0
 where
  predIx = T.argmax (T.Dim 1) T.RemoveDim pred
  info = "label: " <> show label <> "\npred: " <> show pred

collate :: Int -> [a] -> [[a]]
collate n as = case take n as of
  [] -> []
  batch -> batch : collate n (drop n as)

trainEpoch
  :: forall dev hidden o
   . (ValidParams dev hidden, TT.Optimizer o (ModelTensors dev hidden) (ModelTensors dev hidden) QDType dev)
  => Int
  -> Int
  -> TT.LearningRate dev QDType
  -> (QModel dev hidden, o)
  -> P.ListT IO [ImitationData dev]
  -> IO ((QModel dev hidden, o), (QType, QType))
trainEpoch i nBatches lr state batches = do
  pb <- PB.newProgressBar pbStyle 10 (PB.Progress 0 nBatches ())
  (!state', (losses, accs)) <- P.foldM (step pb) begin done $ P.enumerate batches P.>-> P.take nBatches
  let !meanl = mean losses
      !meanacc = mean accs
  --     stdl = mean ((\l -> (l - meanl) ** 2) <$> losses)
  -- putStrLn $ "std loss: " <> show stdl
  pure (state', (meanl, meanacc))
 where
  step
    :: _pb
    -> ((QModel dev hidden, o), ([QType], [QType]))
    -> [ImitationData dev]
    -> IO ((QModel dev hidden, o), ([QType], [QType]))
  step pb ((!model, !optim), (!losses, !accs)) batch = do
    let inputs = dataInput <$> batch
        labels = dataLabel <$> batch
        -- predict :: ImitationDataX dev -> T.Tensor
        -- -- predict inputF = T.transpose2D $ inputF (runBatchedLogPolicy 1 model)
        -- predict (state, actions) = T.transpose2D $ runBatchedLogPolicy 1 model $ encodeStepFake state actions
        -- predictions = fmap predict inputs
        batchEncoding = encodeBatch inputs
        predictions = T.transpose2D <$> runFullyBatchedLogPolicy 1 model batchEncoding
        loss =
          T.divScalar
            (length batch)
            (sum (zipWith nll labels predictions))
        lossTyped :: TT.Loss dev QDType
        lossTyped = TT.UnsafeMkTensor loss + fakeLoss model
        !lossScalar = T.asValue loss
        !accuracy = mean $ zipWith hit labels predictions
    !state' <- TT.runStep model optim lossTyped lr
    PB.incProgress pb 1
    -- putStrLn $ "\nActions: " <> show (sum $ NE.length . snd <$> inputs)
    pure $! (state', (lossScalar : losses, accuracy : accs))
  begin = pure (state, ([], []))
  done = pure
  pbStyle =
    PB.defStyle
      { PB.stylePrefix = "Epoch " <> (PB.msg $ Txt.show i) <> ": " <> (PB.elapsedTime PB.renderDuration)
      , PB.stylePostfix = PB.exact <> " (" <> PB.percentage <> ")"
      , PB.styleWidth = PB.ConstantWidth 80
      }

-- validateEpoch
--   :: (ValidParams dev hidden)
--   => QModel dev hidden
--   -> P.ListT IO (ImitationData dev)
--   -> IO (QType, QType)
-- validateEpoch model dataset = do
--   (losses, accs) <- P.foldM step begin done $ P.enumerate dataset
--   pure $ (mean losses, mean accs)
--  where
--   begin = pure ([], [])
--   done = pure
--   step (!losses, !accs) datapoint = pure $! (loss : losses, acc : accs)
--    where
--     prediction = T.transpose2D $ runBatchedLogPolicy 1 model $ dataInput datapoint
--     label = dataLabel datapoint
--     !loss = T.asValue $ nll label prediction
--     !acc = hit label prediction

validateEpoch
  :: (ValidParams dev hidden)
  => QModel dev hidden
  -> P.ListT IO (ImitationData dev)
  -> IO (QType, QType)
validateEpoch model dataset = do
  datapoints <- P.toListM $ P.enumerate dataset
  let batchEncoding = encodeBatch $ dataInput <$> datapoints
      predictions = runFullyBatchedLogPolicy 1 model batchEncoding
      results = zipWith lossAndAcc predictions datapoints
      (losses, accs) = unzip results
  pure $ (mean losses, mean accs)
 where
  lossAndAcc pred datapoint = (loss, acc)
   where
    prediction = T.transpose2D pred
    label = dataLabel datapoint
    !loss = T.asValue $ nll label $ prediction
    !acc = hit label prediction

train
  :: (ValidParams dev hidden)
  => QModel dev hidden
  -> shuf
  -> (shuf -> ContT ((QModel dev hidden, _o), shuf, (QType, QType)) IO (P.ListT IO (ImitationData dev), shuf))
  -> ImitationDataset IO dev
  -> (Int -> TT.LearningRate dev QDType)
  -> Int
  -> Int
  -> Int
  -> IO (QModel dev hidden, (([QType], [QType]), ([QType], [QType])))
train model0 shuffler0 trainStreamer testData fLR epochs nBatches batchSize = do
  ((modelTrained, _), _, histTrain, histTest) <-
    T.foldLoop ((model0, optim0), shuffler0, ([], []), ([], [])) epochs trainLoop
  pure (modelTrained, (histTrain & both %~ reverse, histTest & both %~ reverse))
 where
  optim0 = TT.mkAdam 0 0.9 0.99 (TT.flattenParameters model0)
  trainLoop (state, shuffler, (lossesTrain, accsTrain), (lossesVal, accsVal)) epoch = do
    -- training step
    let lr = fLR $ fromIntegral epoch
    ((!model', !optim'), !shuffler', (!trainLoss, trainAcc)) <- do
      res <- runContT (trainStreamer shuffler) $
        \(dataset, !shuf') -> do
          let batches = T.collate batchSize Just dataset
          (!state', !loss) <- trainEpoch epoch nBatches lr state batches
          pure $! (state', shuf', loss)
      pure res
    saveModel "rl/actor-imit.ht" model'
    -- test metrics
    (valLoss, valAcc) <-
      runContT (T.streamFromMap (T.datasetOpts 1) testData) $
        validateEpoch model' . fst
    -- return
    putStrLn $ "trainLoss: " <> show trainLoss
    putStrLn $ "valLoss:   " <> show valLoss
    putStrLn $ "trainAcc: " <> show trainAcc
    putStrLn $ "valAcc:   " <> show valAcc
    let lossesTrain' = trainLoss : lossesTrain
        lossesTest' = valLoss : lossesVal
        accsTrain' = trainAcc : accsTrain
        accsTest' = valAcc : accsVal
    plotHistories
      "losses-imitation"
      [reverse lossesTrain', reverse lossesTest', reverse accsTrain', reverse accsTest']
    pure ((model', optim'), shuffler', (lossesTrain', accsTrain'), (lossesTest', accsTest'))

trainDataset model0 trainData testData fLr epochs batchSize = do
  shuffler0 <- pure T.Sequential --  T.Shuffle <$> Rand.initStdGen
  let nTrain = S.size $ T.keys trainData
      nBatches = negate (negate nTrain `div` batchSize)
      streamer shuffler = T.streamFromMap ((T.datasetOpts 1){T.shuffle = shuffler}) trainData
  train model0 shuffler0 streamer testData fLr epochs nBatches batchSize

trainDatastream model0 trainStream =
  train model0 () streamer
 where
  streamer () = ContT $ \k -> k (T.streamSamples trainStream (), ())

type TestDevice = '(TT.CPU, 0)

testTrain :: Int -> IO ()
testTrain epochs = do
  let fLR = const 0.1 -- (* 0.01) <$> (RL.cosSchedule $ fromIntegral n)
  !model0 <- mkQModel @TestDevice @8
  trainData <- makeChordDataset @TestDevice 1
  testData <- makeChordDataset @TestDevice 1
  (_, ((lTrain, aTrain), (lVal, aVal))) <-
    trainDataset model0 trainData testData fLR epochs 1
  plotHistories "losses-imitation" [lTrain, lVal, aTrain, aVal]

testTrainStream :: Int -> IO ()
testTrainStream epochs = do
  let fLR = const 0.1 -- (* 0.01) <$> (RL.cosSchedule $ fromIntegral n)
  !model0 <- mkQModel @TestDevice @8
  gen <- createSystemRandom
  Right hyper <- loadPVHyper "posterior.json"
  let probs = expectedProbs @PVParams hyper
      trainData = ImitationStream @TestDevice probs 4 20 gen
  testData <- makeChordDataset @TestDevice 1
  (_, ((lTrain, aTrain), (lVal, aVal))) <-
    trainDatastream model0 trainData testData fLR epochs 3 5
  plotHistories "losses-imitation" [lTrain, lVal, aTrain, aVal]

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
    Right dat -> mapM_ print $ dataLabel <$> dat

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
      Right dat -> mapM_ print $ dataLabel <$> dat
