{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}

{- | This module contains a simple (and musically rather naive)
 probabilistic model of protovoice derivations.
 This model can be used to sample a derivation,
 evaluate a derivations probability,
 or infer posterior distributions of the model parmeters from given derivations
 (i.e., "learn" the model's probabilities).

 This model is a /locally conjugate/ model:
 It samples a derivation using a sequence of random decisions with certain probabilities.
 These probabilities are generally unknown, so they are themselves modeled as random variables with prior distributions.
 The full model \(p(d, \theta)\) thus splits into
 \[p(D, \theta) = p(d \mid \theta) \cdot p(\theta),\]
 the prior over the probability variables
 \[p(\theta) = \prod_i p(\theta_i),\]
 and the likelihood of the derivation(s) given these probabilities
 \[p(D \mid \theta) = \prod_{d \in D} p(d \mid \theta) = \prod_{d \in D} \prod_i p(d_i \mid \theta, d_0, \ldots, d_{i-1}).\]
 Given all prior decisions, the likelihood of a decision \(d_i\) based on some parameter \(\theta_a\)
 \[p(d_i \mid \theta, d_{<i})\]
 is [conjugate](https://en.wikipedia.org/wiki/Conjugate_prior) with the prior of that parameter \(p(\theta_a)\),
 which means that the posterior of the parameters given one (or several) derivation(s) \(p(\theta \mid D)\)
 can be computed analytically.

 The parameters \(\theta\) and their prior distributions
 are represented by the higher-kinded type 'PVParams'.
 Different instantiations of this type (using 'Hyper' or 'Probs') results in concrete record types
 that represent prior or posterior distributions
 or concrete values (probabilities) for the parameters.
 'PVParams' also supports 'jeffreysPrior' and 'uniformPrior' as default priors,
as well as 'sampleProbs' for sampling from a prior (see "Inferenc.Conjugate").

 The likelihood \(p(d \mid \theta)\) of a derivation is represented by
 'sampleDerivation'.
 It can be executed under different "modes" (probability monads)
 for sampling, inference, or tracing (see "Inference.Conjugate").
 The decisions during the derivation are represented by a 'Trace' (here @Trace PVParams@).
 In order to learn from a given derivation,
 the corresponding trace can be obtained using 'observeDerivation'.
 A combination of getting a trace and learning from it
 is provided by 'trainSinglePiece'.
-}
module PVGrammar.Prob.Simple
  ( -- * Model Parameters

    -- | A higher-kinded type that represents the global parameters (probabilities) of the model.
    -- Use it as 'Hyper PVParams' to represent hyperparameters (priors and posteriors)
    -- or as 'Probs PVParams' to represent actual probabilites.
    -- Each record field corresponds to one parameter
    -- that influences a specific type of decision in the generation process.
    PVParams (..)
  , PVParamsOuter (..)
  , PVParamsInner (..)

    -- * Likelihood Model

    -- | 'sampleDerivation' represents a probabilistic program that samples a derivation.
    -- that can be interpreted in various modes for
    --
    -- - sampling ('sampleTrace', 'sampleResult'),
    -- - inference ('evalTraceLogP', 'getPosterior'),
    -- - tracing ('showTrace', 'traceTrace').
    --
    -- 'observeDerivation' takes and existing derivation and returns the corresponding trace.
  , sampleDerivation
  , sampleDerivation'
  , observeDerivation
  , observeDerivation'

    -- * Utilities
  , roundtrip
  , trainSinglePiece

    -- * Likelihood model for ind
  , sampleSingleStep
  , observeSingleStep
  , sampleDoubleStep
  , observeDoubleStep
  ) where

import Common
  ( Analysis
      ( anaDerivation
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

import Control.Monad
  ( guard
  , unless
  , when
  )
import Control.Monad.Trans.Class (lift)
import Control.Monad.Trans.Except
  ( except
  , runExceptT
  )
import Control.Monad.Trans.State
  ( StateT
  , execStateT
  )
import Data.Bifunctor qualified as Bi
import Data.Foldable (forM_)
import Data.HashMap.Strict qualified as HM
import Data.HashSet qualified as S
import Data.Hashable (Hashable)
import Data.List qualified as L
import Data.Map.Strict qualified as M
import Data.Maybe
  ( catMaybes
  , fromMaybe
  )
import Debug.Trace qualified as DT
import GHC.Generics (Generic)
import Inference.Conjugate

-- import qualified Inference.Conjugate           as IC
import Internal.MultiSet qualified as MS
import Lens.Micro.TH (makeLenses)
import Musicology.Pitch as MP hiding
  ( a
  , b
  , c
  , d
  , e
  , f
  , g
  )
import System.Random.MWC.Probability (categorical)

-- | Parameters for decisions about outer operations (split, spread, freeze).
data PVParamsOuter f = PVParamsOuter
  { _pSingleFreeze :: f Beta
  , _pDoubleLeft :: f Beta
  , _pDoubleLeftFreeze :: f Beta
  , _pDoubleRightSplit :: f Beta
  }
  deriving (Generic)

deriving instance (Show (f Beta)) => Show (PVParamsOuter f)

makeLenses ''PVParamsOuter

{- | Parameters for decisions about inner operations
 (elaboration and distribution within splits and spreads).
-}
data PVParamsInner f = PVParamsInner
  -- split
  { _pElaborateRegular :: f Beta
  , _pElaborateL :: f Beta
  , _pElaborateR :: f Beta
  , _pRootFifths :: f Beta
  , _pKeepL :: f Beta
  , _pKeepR :: f Beta
  , _pRepeatOverNeighbor :: f Beta
  , _pNBChromatic :: f Beta
  , _pNBAlt :: f Beta
  , _pRepeatLeftOverRight :: f Beta
  , _pRepeatAlter :: f Beta
  , _pRepeatAlterUp :: f Beta
  , _pRepeatAlterSemis :: f Beta
  , _pConnect :: f Beta
  , _pConnectChromaticLeftOverRight :: f Beta
  , _pPassUp :: f Beta
  , _pPassLeftOverRight :: f Beta
  , _pNewPassingLeft :: f Beta
  , _pNewPassingRight :: f Beta
  , -- spread
    _pNewPassingMid :: f Beta
  , _pNoteSpreadDirection :: f (Dirichlet 3)
  , _pNotesOnOtherSide :: f Beta
  , _pSpreadRepetitionEdge :: f Beta
  }
  deriving (Generic)

deriving instance
  ( Show (f Beta)
  , Show (f Beta)
  , Show (f Beta)
  , Show (f (Dirichlet 3))
  , Show (f Beta)
  )
  => Show (PVParamsInner f)

makeLenses ''PVParamsInner

-- | The combined parameters for inner and outer operations.
data PVParams f = PVParams
  { _pOuter :: PVParamsOuter f
  , _pInner :: PVParamsInner f
  }
  deriving (Generic)

deriving instance
  ( Show (f Beta)
  , Show (f Beta)
  , Show (f Beta)
  , Show (f (Dirichlet 3))
  , Show (f Beta)
  )
  => Show (PVParams f)

makeLenses ''PVParams

data MagicalOctaves = MagicalOctaves
  deriving (Eq, Ord, Show)

instance Distribution MagicalOctaves where
  type Params MagicalOctaves = ()
  type Support MagicalOctaves = Int
  distSample _ _ = (`subtract` 2) <$> categorical [0.1, 0.2, 0.4, 0.2, 0.1]
  distLogP _ _ _ = 0

type PVProbs = PVParams ProbsRep
type PVProbsInner = PVParamsInner ProbsRep

type ContextSingle n = (StartStop (Notes n), Edges n, StartStop (Notes n))
type ContextDouble n =
  (StartStop (Notes n), Edges n, Notes n, Edges n, StartStop (Notes n))

type PVObs a = StateT (Trace PVParams) (Either String) a

{- | A helper function that tests whether 'observeDerivation''
 followed by 'sampleDerivation'' restores the original derivation.
 Useful for testing the compatibility of the two functions.
-}
roundtrip :: FilePath -> IO (Either String [PVLeftmost SPitch])
roundtrip fn = do
  anaE <- loadAnalysis fn
  case anaE of
    Left err -> error err
    Right ana -> do
      let traceE = observeDerivation' $ anaDerivation ana
      case traceE of
        Left err -> error err
        Right trace -> do
          pure $ traceTrace trace sampleDerivation'

{- | Helper function: Load a single derivation
 and infer the corresponding posterior for a uniform prior.
-}
trainSinglePiece :: FilePath -> IO (Maybe (PVParams HyperRep))
trainSinglePiece fn = do
  anaE <- loadAnalysis fn
  case anaE of
    Left err -> error err
    Right ana -> do
      let traceE = observeDerivation' $ anaDerivation ana
      case traceE of
        Left err -> error err
        Right trace -> do
          let prior = uniformPrior @PVParams
          pure $ getPosterior prior trace (sampleDerivation $ anaTop ana)

-- | A shorthand for 'sampleDerivation' starting from ⋊——⋉.
sampleDerivation' :: (_) => m (Either String [PVLeftmost SPitch])
sampleDerivation' = sampleDerivation $ PathEnd topEdges

-- | A shorthand for 'observeDerivation' starting from ⋊——⋉.
observeDerivation' :: [PVLeftmost SPitch] -> Either String (Trace PVParams)
observeDerivation' deriv = observeDerivation deriv $ PathEnd topEdges

{- | A probabilistic program that samples a derivation starting from a given root path.
 Can be interpreted by the interpreter functions in "Inference.Conjugate".
-}
sampleDerivation
  :: (_)
  => Path (Edges SPitch) (Notes SPitch)
  -- ^ root path
  -> m (Either String [PVLeftmost SPitch])
  -- ^ a probabilistic program
sampleDerivation top = runExceptT $ go Start top False
 where
  go sl surface ars = case surface of
    -- 1 trans left:
    PathEnd t -> do
      step <- lift $ sampleSingleStep (sl, t, Stop)
      case step of
        LMSingleSplit splitOp -> do
          (ctl, cs, ctr) <- except $ applySplit splitOp t
          nextSteps <- go sl (Path ctl cs (PathEnd ctr)) False
          pure $ LMSplitOnly splitOp : nextSteps
        LMSingleFreeze freezeOp -> pure [LMFreezeOnly freezeOp]
    -- 2 trans left
    Path tl sm (PathEnd tr) -> goDouble sl tl sm tr Stop ars PathEnd
    -- 3 or more trans left
    Path tl sm (Path tr sr rest) ->
      goDouble sl tl sm tr (Inner sr) ars (\tr' -> Path tr' sr rest)

  -- helper for the two cases of 2+ edges (2 and 3+):
  goDouble sl tl sm tr sr ars mkrest = do
    step <- lift $ sampleDoubleStep (sl, tl, sm, tr, sr) ars
    case step of
      LMDoubleSplitLeft splitOp -> do
        (ctl, cs, ctr) <- except $ applySplit splitOp tl
        nextSteps <- go sl (Path ctl cs (Path ctr sm (mkrest tr))) False
        pure $ LMSplitLeft splitOp : nextSteps
      LMDoubleFreezeLeft freezeOp -> do
        nextSteps <- go (Inner sm) (mkrest tr) False
        pure $ LMFreezeLeft freezeOp : nextSteps
      LMDoubleSplitRight splitOp -> do
        (ctl, cs, ctr) <- except $ applySplit splitOp tr
        nextSteps <- go sl (Path tl sm (Path ctl cs (mkrest ctr))) True
        pure $ LMSplitRight splitOp : nextSteps
      LMDoubleSpread spreadOp -> do
        (ctl, csl, ctm, csr, ctr) <- except $ applySpread spreadOp tl sm tr
        nextSteps <- go sl (Path ctl csl (Path ctm csr (mkrest ctr))) False
        pure $ LMSpread spreadOp : nextSteps

{- | Walk through a derivation (starting at a given root path)
 and return the corresponding 'Trace' (if possible).
 The trace can be used together with 'sampleDerivation'
 for inference ('getPosterior') or for showing the trace ('printTrace').
-}
observeDerivation
  :: [PVLeftmost SPitch]
  -> Path (Edges SPitch) (Notes SPitch)
  -> Either String (Trace PVParams)
observeDerivation deriv top =
  execStateT
    (go Start top False deriv)
    (Trace mempty)
 where
  go
    :: StartStop (Notes SPitch)
    -> Path (Edges SPitch) (Notes SPitch)
    -> Bool
    -> [PVLeftmost SPitch]
    -> PVObs ()
  go _sl _surface _ars [] = lift $ Left "Derivation incomplete."
  go sl (PathEnd trans) _ars (op : rest) = case op of
    LMSingle single -> do
      observeSingleStep (sl, trans, Stop) single
      case single of
        LMSingleFreeze _ -> pure ()
        LMSingleSplit splitOp -> do
          (ctl, cs, ctr) <- lift $ applySplit splitOp trans
          go sl (Path ctl cs (PathEnd ctr)) False rest
    LMDouble _ -> lift $ Left "Double operation on single transition."
  go sl (Path tl sm (PathEnd tr)) ars (op : rest) =
    goDouble op rest ars (sl, tl, sm, tr, Stop) PathEnd
  go sl (Path tl sm (Path tr sr pathRest)) ars (op : derivRest) =
    goDouble op derivRest ars (sl, tl, sm, tr, Inner sr) $
      \tr' -> Path tr' sr pathRest

  goDouble op rest ars (sl, tl, sm, tr, sr) mkRest = case op of
    LMSingle _ -> lift $ Left "Single operation with several transitions left."
    LMDouble double -> do
      observeDoubleStep (sl, tl, sm, tr, sr) ars double
      case double of
        LMDoubleFreezeLeft _ -> do
          when ars $ lift $ Left "FreezeLeft after SplitRight."
          go (Inner sm) (mkRest tr) False rest
        LMDoubleSplitLeft splitOp -> do
          when ars $ lift $ Left "SplitLeft after SplitRight."
          (ctl, cs, ctr) <- lift $ applySplit splitOp tl
          go sl (Path ctl cs $ Path ctr sm $ mkRest tr) False rest
        LMDoubleSplitRight splitOp -> do
          (ctl, cs, ctr) <- lift $ applySplit splitOp tr
          go sl (Path tl sm $ Path ctl cs $ mkRest ctr) True rest
        LMDoubleSpread spreadOp -> do
          (ctl, csl, ctm, csr, ctr) <- lift $ applySpread spreadOp tl sm tr
          go sl (Path ctl csl $ Path ctm csr $ mkRest ctr) False rest

sampleSingleStep
  :: (_) => ContextSingle SPitch -> m (LeftmostSingle (Split SPitch) Freeze)
sampleSingleStep parents@(_, trans, _) =
  if freezable trans
    then do
      shouldFreeze <-
        sampleValue "shouldFreeze (single)" Bernoulli $ pOuter . pSingleFreeze
      if shouldFreeze
        then LMSingleFreeze <$> sampleFreeze parents
        else LMSingleSplit <$> sampleSplit parents
    else LMSingleSplit <$> sampleSplit parents

observeSingleStep
  :: ContextSingle SPitch -> LeftmostSingle (Split SPitch) Freeze -> PVObs ()
observeSingleStep parents@(_, trans, _) singleOp =
  if freezable trans
    then case singleOp of
      LMSingleFreeze f -> do
        observeValue
          "shouldFreeze (single)"
          Bernoulli
          (pOuter . pSingleFreeze)
          True
        observeFreeze parents f
      LMSingleSplit s -> do
        observeValue
          "shouldFreeze (single)"
          Bernoulli
          (pOuter . pSingleFreeze)
          False
        observeSplit parents s
    else case singleOp of
      LMSingleFreeze _ -> lift $ Left "Freezing a non-freezable transition."
      LMSingleSplit s -> observeSplit parents s

sampleDoubleStep
  :: (_)
  => ContextDouble SPitch
  -> Bool
  -> m (LeftmostDouble (Split SPitch) Freeze (Spread SPitch))
sampleDoubleStep parents@(sliceL, transL, sliceM, transR, sliceR) afterRightSplit =
  if afterRightSplit
    then do
      shouldSplitRight <-
        sampleValue "shouldSplitRight" Bernoulli $ pOuter . pDoubleRightSplit
      if shouldSplitRight
        then LMDoubleSplitRight <$> sampleSplit (Inner sliceM, transR, sliceR)
        else LMDoubleSpread <$> sampleSpread parents
    else do
      continueLeft <-
        sampleValue "continueLeft" Bernoulli $ pOuter . pDoubleLeft
      if continueLeft
        then
          if freezable transL
            then do
              shouldFreeze <-
                sampleValue "shouldFreeze (double)" Bernoulli $
                  pOuter
                    . pDoubleLeftFreeze
              if shouldFreeze
                then
                  LMDoubleFreezeLeft
                    <$> sampleFreeze (sliceL, transL, Inner sliceM)
                else
                  LMDoubleSplitLeft
                    <$> sampleSplit (sliceL, transL, Inner sliceM)
            else LMDoubleSplitLeft <$> sampleSplit (sliceL, transL, Inner sliceM)
        else sampleDoubleStep parents True

observeDoubleStep
  :: ContextDouble SPitch
  -> Bool
  -> LeftmostDouble (Split SPitch) Freeze (Spread SPitch)
  -> PVObs ()
observeDoubleStep parents@(sliceL, transL, sliceM, transR, sliceR) afterRightSplit doubleOp =
  case doubleOp of
    LMDoubleFreezeLeft f -> do
      observeValue "continueLeft" Bernoulli (pOuter . pDoubleLeft) True
      observeValue
        "shouldFreeze (double)"
        Bernoulli
        (pOuter . pDoubleLeftFreeze)
        True
      observeFreeze (sliceL, transL, Inner sliceM) f
    LMDoubleSplitLeft s -> do
      observeValue "continueLeft" Bernoulli (pOuter . pDoubleLeft) True
      when (freezable transL) $
        observeValue
          "shouldFreeze (double)"
          Bernoulli
          (pOuter . pDoubleLeftFreeze)
          False
      observeSplit (sliceL, transL, Inner sliceM) s
    LMDoubleSplitRight s -> do
      unless afterRightSplit $
        observeValue "continueLeft" Bernoulli (pOuter . pDoubleLeft) False
      observeValue
        "shouldSplitRight"
        Bernoulli
        (pOuter . pDoubleRightSplit)
        True
      observeSplit (Inner sliceM, transR, sliceR) s
    LMDoubleSpread h -> do
      unless afterRightSplit $
        observeValue "continueLeft" Bernoulli (pOuter . pDoubleLeft) False
      observeValue
        "shouldSplitRight"
        Bernoulli
        (pOuter . pDoubleRightSplit)
        False
      observeSpread parents h

sampleFreeze :: (RandomInterpreter m PVParams) => ContextSingle n -> m Freeze
sampleFreeze _parents = pure FreezeOp

observeFreeze :: ContextSingle SPitch -> Freeze -> PVObs ()
observeFreeze _parents FreezeOp = pure ()

-- helper for sampleSplit and observeSplit
collectElabos
  :: [(Edge SPitch, [(SPitch, o1)])]
  -> [(InnerEdge SPitch, [(SPitch, PassingOrnament)])]
  -> [(SPitch, [(SPitch, o2)])]
  -> [(SPitch, [(SPitch, o3)])]
  -> ( M.Map (StartStop SPitch, StartStop SPitch) [(SPitch, o1)]
     , M.Map (SPitch, SPitch) [(SPitch, PassingOrnament)]
     , M.Map SPitch [(SPitch, o2)]
     , M.Map SPitch [(SPitch, o3)]
     , S.HashSet (Edge SPitch)
     , S.HashSet (Edge SPitch)
     )
collectElabos childrenT childrenNT childrenL childrenR =
  let splitTs = M.fromList childrenT
      splitNTs = M.fromList childrenNT
      fromLeft = M.fromList childrenL
      fromRight = M.fromList childrenR
      keepLeftT = getEdges childrenT (\p m -> (fst p, Inner m))
      keepLeftL = getEdges childrenL (\l m -> (Inner l, Inner m))
      keepLeftNT = do
        -- List
        ((l, _), cs) <- childrenNT
        (m, orn) <- cs
        guard $ orn /= PassingRight
        pure (Inner l, Inner m)
      leftEdges = S.fromList $ keepLeftT <> keepLeftNT <> keepLeftL
      keepRightT = getEdges childrenT (\p m -> (Inner m, snd p))
      keepRightR = getEdges childrenR (\r m -> (Inner m, Inner r))
      keepRightNT = do
        -- List
        ((_, r), cs) <- childrenNT
        (m, orn) <- cs
        guard $ orn /= PassingLeft
        pure (Inner m, Inner r)
      rightEdges = S.fromList $ keepRightT <> keepRightNT <> keepRightR
   in (splitTs, splitNTs, fromLeft, fromRight, leftEdges, rightEdges)
 where
  getEdges :: [(p, [(c, o)])] -> (p -> c -> Edge SPitch) -> [Edge SPitch]
  getEdges elabos mkEdge = do
    -- List
    (p, cs) <- elabos
    (c, _) <- cs
    pure $ mkEdge p c

-- helper for sampleSplit and observeSplit
collectNotes
  :: [(Edge SPitch, [(SPitch, o1)])]
  -> [(InnerEdge SPitch, [(SPitch, PassingOrnament)])]
  -> [(SPitch, [(SPitch, o2)])]
  -> [(SPitch, [(SPitch, o3)])]
  -> [SPitch]
collectNotes childrenT childrenNT childrenL childrenR =
  let notesT = concatMap (fmap fst . snd) childrenT
      notesNT = concatMap (fmap fst . snd) childrenNT
      notesFromL = concatMap (fmap fst . snd) childrenL
      notesFromR = concatMap (fmap fst . snd) childrenR
   in L.sort $ notesT <> notesNT <> notesFromL <> notesFromR

sampleSplit :: forall m. (_) => ContextSingle SPitch -> m (Split SPitch)
sampleSplit (sliceL, _edges@(Edges ts nts), sliceR) = do
  -- DT.traceM $ "\nPerforming split (smp) on: " <> show edges
  -- ornament regular edges at least once
  childrenT <- mapM sampleT $ L.sort $ S.toList ts
  -- DT.traceM $ "childrenT (smp): " <> show childrenT
  -- ornament passing edges exactly once
  childrenNT <- mapM sampleNT $ L.sort $ MS.toOccurList nts
  -- DT.traceM $ "childrenNT (smp): " <> show childrenNT
  -- ornament left notes
  childrenL <- case getInner sliceL of
    Nothing -> pure []
    Just (Notes notes) -> mapM sampleL $ L.sort $ MS.toList notes
  -- DT.traceM $ "childrenL (smp): " <> show childrenL
  -- ornament right notes
  childrenR <- case getInner sliceR of
    Nothing -> pure []
    Just (Notes notes) -> mapM sampleR $ L.sort $ MS.toList notes
  -- DT.traceM $ "childrenR (smp): " <> show childrenR
  -- introduce new passing edges left and right
  let notes = collectNotes childrenT childrenNT childrenL childrenR
  passLeft <- case getInner sliceL of
    Nothing -> pure MS.empty
    Just (Notes notesl) ->
      samplePassing (L.sort $ MS.toList notesl) notes pNewPassingLeft
  passRight <- case getInner sliceR of
    Nothing -> pure MS.empty
    Just (Notes notesr) ->
      samplePassing notes (L.sort $ MS.toList notesr) pNewPassingRight
  let (splitReg, splitPass, fromLeft, fromRight, leftEdges, rightEdges) =
        collectElabos childrenT childrenNT childrenL childrenR
  -- decide which edges to keep
  keepLeft <- sampleKeepEdges pKeepL leftEdges
  keepRight <- sampleKeepEdges pKeepR rightEdges
  -- combine all sampling results into split operation
  let splitOp =
        SplitOp
          { splitReg
          , splitPass
          , fromLeft
          , fromRight
          , keepLeft
          , keepRight
          , passLeft
          , passRight
          }
  -- DT.traceM $ "Performing split (smp): " <> show splitOp
  pure splitOp

observeSplit :: ContextSingle SPitch -> Split SPitch -> PVObs ()
observeSplit (sliceL, _edges@(Edges ts nts), sliceR) _splitOp@(SplitOp splitTs splitNTs fromLeft fromRight keepLeft keepRight passLeft passRight) =
  do
    -- DT.traceM $ "\nPerforming split (obs): " <> show splitOp
    -- observe ornaments of regular edges
    childrenT <- mapM (observeT splitTs) $ L.sort $ S.toList ts
    -- DT.traceM $ "childrenT (obs): " <> show childrenT
    -- observe ornaments of passing edges
    childrenNT <- mapM (observeNT splitNTs) $ L.sort $ MS.toOccurList nts
    -- DT.traceM $ "childrenNT (obs): " <> show childrenNT
    -- observe ornaments of left notes
    childrenL <- case getInner sliceL of
      Nothing -> pure []
      Just (Notes notes) -> mapM (observeL fromLeft) $ L.sort $ MS.toList notes
    -- DT.traceM $ "childrenL (obs): " <> show childrenL
    -- observe ornaments of right notes
    childrenR <- case getInner sliceR of
      Nothing -> pure []
      Just (Notes notes) ->
        mapM (observeR fromRight) $ L.sort $ MS.toList notes
    -- DT.traceM $ "childrenR (obs): " <> show childrenR
    -- observe new passing edges
    let notes = collectNotes childrenT childrenNT childrenL childrenR
    case getInner sliceL of
      Nothing -> pure ()
      Just (Notes notesl) ->
        observePassing
          (L.sort $ MS.toList notesl)
          notes
          pNewPassingLeft
          passLeft
    case getInner sliceR of
      Nothing -> pure ()
      Just (Notes notesr) ->
        observePassing
          notes
          (L.sort $ MS.toList notesr)
          pNewPassingRight
          passRight
    -- observe which edges are kept
    let (_, _, _, _, leftEdges, rightEdges) =
          collectElabos childrenT childrenNT childrenL childrenR
    observeKeepEdges pKeepL leftEdges keepLeft
    observeKeepEdges pKeepR rightEdges keepRight

sampleRootNote :: (_) => m SPitch
sampleRootNote = do
  fifthsSign <- sampleConst "rootFifthsSign" Bernoulli 0.5
  fifthsN <- sampleValue "rootFifthsN" Geometric0 $ pInner . pRootFifths
  os <- sampleConst "rootOctave" MagicalOctaves ()
  let fs = if fifthsSign then fifthsN else negate (fifthsN + 1)
      p = (emb <$> spc fs) +^ (octave ^* (os + 4))
  -- DT.traceM $ "root note (sample): " <> show p
  pure p

observeRootNote :: SPitch -> PVObs ()
observeRootNote child = do
  observeConst "rootFifthsSign" Bernoulli 0.5 fifthsSign
  observeValue "rootFifthsN" Geometric0 (pInner . pRootFifths) fifthsN
  observeConst "rootOctave" MagicalOctaves () (octaves child - 4)
 where
  -- DT.traceM $ "root note (obs): " <> show child

  fs = fifths child
  fifthsSign = fs >= 0
  fifthsN = if fifthsSign then fs else negate fs - 1

sampleOctaveShift :: (_) => String -> m SInterval
sampleOctaveShift name = do
  n <- sampleConst name MagicalOctaves ()
  let os = octave ^* (n - 4)
  -- DT.traceM $ "octave shift (smp) " <> show os
  pure os

observeOctaveShift :: (_) => String -> SInterval -> PVObs ()
observeOctaveShift name interval = do
  let n = octaves (interval ^+^ major second)
  observeConst name MagicalOctaves () $ n + 4

-- DT.traceM $ "octave shift (obs) " <> show (octave @SInterval ^* n)

sampleNeighbor :: (_) => Bool -> SPitch -> m SPitch
sampleNeighbor stepUp ref = do
  chromatic <- sampleValue "nbChromatic" Bernoulli $ pInner . pNBChromatic
  os <- sampleOctaveShift "nbOctShift"
  alt <- sampleValue "nbAlt" Geometric0 $ pInner . pNBAlt
  let altInterval = emb (alt *^ chromaticSemitone @SIC)
  if chromatic
    then do
      pure $ ref +^ os +^ if stepUp then altInterval else down altInterval
    else do
      altUp <- sampleConst "nbAltUp" Bernoulli 0.5
      let step =
            if altUp == stepUp
              then major second ^+^ altInterval
              else minor second ^-^ altInterval
      pure $ ref +^ os +^ if stepUp then step else down step

observeNeighbor :: Bool -> SPitch -> SPitch -> PVObs ()
observeNeighbor goesUp ref nb = do
  let interval = ic $ ref `pto` nb
      isChromatic = diasteps interval == 0
  observeValue "nbChromatic" Bernoulli (pInner . pNBChromatic) isChromatic
  observeOctaveShift "nbOctShift" (ref `pto` nb)
  if isChromatic
    then do
      let alt = abs (alteration interval)
      observeValue "nbAlt" Geometric0 (pInner . pNBAlt) alt
    else do
      let alt = alteration (iabs interval)
          altUp = (alt >= 0) == goesUp
          altN = if alt >= 0 then alt else (-alt) - 1
      observeValue "nbAlt" Geometric0 (pInner . pNBAlt) altN
      observeConst "nbAltUp" Bernoulli 0.5 altUp

sampleDoubleChild :: (_) => SPitch -> SPitch -> m (SPitch, DoubleOrnament)
sampleDoubleChild pl pr
  | degree pl == degree pr = do
      rep <-
        sampleValue "repeatOverNeighbor" Bernoulli $ pInner . pRepeatOverNeighbor
      if rep
        then do
          os <- sampleOctaveShift "doubleChildOctave"
          pure (pl +^ os, FullRepeat)
        else do
          stepUp <- sampleConst "stepUp" Bernoulli 0.5
          (,FullNeighbor) <$> sampleNeighbor stepUp pl
  | otherwise = do
      repeatLeft <-
        sampleValue "repeatLeftOverRight" Bernoulli $
          pInner
            . pRepeatLeftOverRight
      repeatAlter <- sampleValue "repeatAlter" Bernoulli $ pInner . pRepeatAlter
      alt <-
        if repeatAlter
          then do
            alterUp <-
              sampleValue "repeatAlterUp" Bernoulli $ pInner . pRepeatAlterUp
            semis <-
              sampleValue "repeatAlterSemis" Geometric1 $ pInner . pRepeatAlterSemis
            pure $ (if alterUp then id else down) $ chromaticSemitone ^* semis
          else pure unison
      os <- sampleOctaveShift "doubleChildOctave"
      if repeatLeft
        then pure (pl +^ os +^ alt, RightRepeatOfLeft)
        else pure (pr +^ os +^ alt, LeftRepeatOfRight)

observeDoubleChild :: SPitch -> SPitch -> SPitch -> PVObs ()
observeDoubleChild pl pr child
  | degree pl == degree pr = do
      let isRep = pc child == pc pl
      observeValue
        "RepeatOverNeighbor"
        Bernoulli
        (pInner . pRepeatOverNeighbor)
        isRep
      if isRep
        then do
          observeOctaveShift "doubleChildOctave" (pl `pto` child)
        else do
          let dir = direction (pc pl `pto` pc child)
          let goesUp = dir == GT
          observeConst "stepUp" Bernoulli 0.5 goesUp
          observeNeighbor goesUp pl child
  | otherwise = do
      let repeatLeft = degree pl == degree child
          ref = if repeatLeft then pl else pr
          alt = alteration child - alteration ref
      observeValue
        "repeatLeftOverRight"
        Bernoulli
        (pInner . pRepeatLeftOverRight)
        repeatLeft
      observeValue "repeatAlter" Bernoulli (pInner . pRepeatAlter) (alt /= 0)
      when (alt /= 0) $ do
        observeValue "repeatAlterUp" Bernoulli (pInner . pRepeatAlterUp) (alt > 0)
        observeValue
          "repeatAlterSemis"
          Geometric1
          (pInner . pRepeatAlterSemis)
          (abs alt)
      observeOctaveShift "doubleChildOctave" $ ref `pto` child

sampleT :: (_) => Edge SPitch -> m (Edge SPitch, [(SPitch, DoubleOrnament)])
sampleT (l, r) = do
  -- DT.traceM $ "elaborating T (smp): " <> show (l, r)
  n <- sampleValue "elaborateRegular" Geometric1 $ pInner . pElaborateRegular
  children <- permutationPlate n $ case (l, r) of
    (Start, Stop) -> do
      child <- sampleRootNote
      pure $ Just (child, RootNote)
    (Inner pl, Inner pr) -> do
      (child, orn) <- sampleDoubleChild pl pr
      pure $ Just (child, orn)
    _ -> pure Nothing
  pure ((l, r), catMaybes children)

observeT
  :: M.Map (Edge SPitch) [(SPitch, DoubleOrnament)]
  -> Edge SPitch
  -> PVObs (Edge SPitch, [(SPitch, DoubleOrnament)])
observeT splitTs parents = do
  -- DT.traceM $ "elaborating T (obs): " <> show parents
  let children = fromMaybe [] $ M.lookup parents splitTs
  observeValue
    "elaborateRegular"
    Geometric1
    (pInner . pElaborateRegular)
    (length children)
  forM_ children $ \(child, _) -> case parents of
    (Start, Stop) -> do
      observeRootNote child
    (Inner pl, Inner pr) -> do
      observeDoubleChild pl pr child
    _ -> lift $ Left $ "Invalid parent edge " <> show parents <> "."
  pure (parents, children)

-- requires distance >= M2
sampleChromPassing :: (_) => SPitch -> SPitch -> m (SPitch, PassingOrnament)
sampleChromPassing pl pr = do
  atLeft <-
    sampleValue "connectChromaticLeftOverRight" Bernoulli $
      pInner
        . pConnectChromaticLeftOverRight
  os <- sampleOctaveShift "connectChromaticOctave"
  let dir = if direction (pc pl `pto` pc pr) == GT then id else down
      child =
        if atLeft
          then pl +^ dir chromaticSemitone
          else pr -^ dir chromaticSemitone
  pure (child +^ os, PassingMid)

observeChromPassing :: SPitch -> SPitch -> SPitch -> PVObs ()
observeChromPassing pl pr child = do
  let isLeft = degree pl == degree child
  observeValue
    "connectChromaticLeftOverRight"
    Bernoulli
    (pInner . pConnectChromaticLeftOverRight)
    isLeft
  observeOctaveShift
    "connectChromaticOctave"
    ((if isLeft then pl else pr) `pto` child)

sampleMidPassing :: (_) => SPitch -> SPitch -> m (SPitch, PassingOrnament)
sampleMidPassing pl pr = do
  child <- sampleNeighbor (direction (pc pl `pto` pc pr) == GT) pl
  pure (child, PassingMid)

observeMidPassing :: SPitch -> SPitch -> SPitch -> PVObs ()
observeMidPassing pl pr =
  observeNeighbor (direction (pc pl `pto` pc pr) == GT) pl

sampleNonMidPassing :: (_) => SPitch -> SPitch -> m (SPitch, PassingOrnament)
sampleNonMidPassing pl pr = do
  left <-
    sampleValue "passLeftOverRight" Bernoulli $ pInner . pPassLeftOverRight
  -- TODO: sampling like this overgenerates, since it allows passing motions to change direction
  -- the direction of a passing edge should be tracked explicitly!
  dirUp <- sampleValue "passUp" Bernoulli $ pInner . pPassUp
  -- let dirUp = direction (pc pl `pto` pc pr) == GT
  if left
    then do
      child <- sampleNeighbor dirUp pl
      pure (child, PassingLeft)
    else do
      child <- sampleNeighbor (not dirUp) pr
      pure (child, PassingRight)

observeNonMidPassing :: SPitch -> SPitch -> SPitch -> PassingOrnament -> PVObs ()
observeNonMidPassing pl pr child orn = do
  let left = orn == PassingLeft
      dirUp =
        if left
          then direction (pc pl `pto` pc child) == GT
          else direction (pc pr `pto` pc child) == LT
  observeValue "passLeftOverRight" Bernoulli (pInner . pPassLeftOverRight) left
  observeValue "passUp" Bernoulli (pInner . pPassUp) dirUp
  if left
    then observeNeighbor dirUp pl child
    else observeNeighbor (not dirUp) pr child

sampleNT
  :: (_) => (InnerEdge SPitch, Int) -> m (InnerEdge SPitch, [(SPitch, PassingOrnament)])
sampleNT ((pl, pr), n) = do
  -- DT.traceM $ "Elaborating edge (smp): " <> show ((pl, pr), n)
  let dist = degree $ iabs $ pc pl `pto` pc pr
  -- DT.traceM    $  "passing from "    <> showNotation pl    <> " to "    <> showNotation pr    <> ": "    <> show dist    <> " steps."
  children <- permutationPlate n $ case dist of
    1 -> sampleChromPassing pl pr
    2 -> do
      connect <- sampleValue "passingConnect" Bernoulli $ pInner . pConnect
      if connect then sampleMidPassing pl pr else sampleNonMidPassing pl pr
    _ -> sampleNonMidPassing pl pr
  pure ((pl, pr), children)

observeNT
  :: (_)
  => M.Map (InnerEdge SPitch) [(SPitch, PassingOrnament)]
  -> (InnerEdge SPitch, Int)
  -> PVObs (InnerEdge SPitch, [(SPitch, PassingOrnament)])
observeNT splitNTs ((pl, pr), _n) = do
  -- DT.traceM $ "Elaborating edge (obs): " <> show ((pl, pr), n)
  let children = fromMaybe [] $ M.lookup (pl, pr) splitNTs
  forM_ children $ \(child, orn) -> case degree $ iabs $ pc pl `pto` pc pr of
    1 -> observeChromPassing pl pr child
    2 -> case orn of
      PassingMid -> do
        observeValue "passingConnect" Bernoulli (pInner . pConnect) True
        observeMidPassing pl pr child
      _ -> do
        observeValue "passingConnect" Bernoulli (pInner . pConnect) False
        observeNonMidPassing pl pr child orn
    _ -> observeNonMidPassing pl pr child orn
  pure ((pl, pr), children)

sampleSingleOrn
  :: (_)
  => SPitch
  -> o
  -> o
  -> Accessor PVParamsInner Beta
  -> m (SPitch, [(SPitch, o)])
sampleSingleOrn parent oRepeat oNeighbor pElaborate = do
  n <- sampleValue "elaborateSingle" Geometric0 $ pInner . pElaborate
  children <- permutationPlate n $ do
    rep <-
      sampleValue "repeatOverNeighborSingle" Bernoulli $
        pInner
          . pRepeatOverNeighbor
    if rep
      then do
        os <- sampleOctaveShift "singleChildOctave"
        pure (parent +^ os, oRepeat)
      else do
        stepUp <- sampleConst "singleUp" Bernoulli 0.5
        child <- sampleNeighbor stepUp parent
        pure (child, oNeighbor)
  pure (parent, children)

observeSingleOrn
  :: M.Map SPitch [(SPitch, o)]
  -> SPitch
  -> Accessor PVParamsInner Beta
  -> PVObs (SPitch, [(SPitch, o)])
observeSingleOrn table parent pElaborate = do
  let children = fromMaybe [] $ M.lookup parent table
  observeValue
    "elaborateSingle"
    Geometric0
    (pInner . pElaborate)
    (length children)
  forM_ children $ \(child, _) -> do
    let rep = pc child == pc parent
    observeValue
      "repeatOverNeighborSingle"
      Bernoulli
      (pInner . pRepeatOverNeighbor)
      rep
    if rep
      then do
        observeOctaveShift "singleChildOctave" (parent `pto` child)
      else do
        let dir = direction (pc parent `pto` pc child)
            up = dir == GT
        observeConst "singleUp" Bernoulli 0.5 up
        observeNeighbor up parent child
  pure (parent, children)

sampleL :: (_) => SPitch -> m (SPitch, [(SPitch, RightOrnament)])
sampleL parent = sampleSingleOrn parent RightRepeat RightNeighbor pElaborateL

observeL
  :: M.Map SPitch [(SPitch, RightOrnament)]
  -> SPitch
  -> PVObs (SPitch, [(SPitch, RightOrnament)])
observeL ls parent = observeSingleOrn ls parent pElaborateL

sampleR :: (_) => SPitch -> m (SPitch, [(SPitch, LeftOrnament)])
sampleR parent = sampleSingleOrn parent LeftRepeat LeftNeighbor pElaborateR

observeR
  :: M.Map SPitch [(SPitch, LeftOrnament)]
  -> SPitch
  -> PVObs (SPitch, [(SPitch, LeftOrnament)])
observeR rs parent = observeSingleOrn rs parent pElaborateR

sampleKeepEdges
  :: (_) => Accessor PVParamsInner Beta -> S.HashSet e -> m (S.HashSet e)
sampleKeepEdges pKeep set = do
  kept <- mapM sKeep (L.sort $ S.toList set)
  pure $ S.fromList $ catMaybes kept
 where
  sKeep elt = do
    keep <- sampleValue "keep" Bernoulli (pInner . pKeep)
    pure $ if keep then Just elt else Nothing

observeKeepEdges
  :: (Eq e, Hashable e, Ord e)
  => Accessor PVParamsInner Beta
  -> S.HashSet e
  -> S.HashSet e
  -> PVObs ()
observeKeepEdges pKeep candidates kept =
  mapM_
    oKeep
    (L.sort $ S.toList candidates)
 where
  oKeep edge =
    observeValue "keep" Bernoulli (pInner . pKeep) (S.member edge kept)

sampleSpread :: (_) => ContextDouble SPitch -> m (Spread SPitch)
sampleSpread (_sliceL, _transL, Notes sliceM, _transR, _sliceR) = do
  -- distribute notes
  dists <- mapM distNote $ MS.toOccurList sliceM
  let notesLeft = catMaybes $ flip fmap dists $ \((note, n), to) -> case to of
        ToRight dl -> if n - dl > 0 then Just note else Nothing
        _ -> Just note
      notesRight = catMaybes $ flip fmap dists $ \((note, n), to) -> case to of
        ToLeft dr -> if n - dr > 0 then Just note else Nothing
        _ -> Just note
  -- generate repetition edges
  repeats <- sequence $ do
    -- List
    l <- notesLeft
    r <- notesRight
    guard $ pc l == pc r
    pure $ do
      -- m
      rep <-
        sampleValue "spreadRepeatEdge" Bernoulli $
          pInner
            . pSpreadRepetitionEdge
      pure $ if rep then Just (Inner l, Inner r) else Nothing
  let repEdges = S.fromList $ catMaybes repeats
  -- generate passing edges
  passEdges <- samplePassing notesLeft notesRight pNewPassingMid
  -- construct result
  let distMap = HM.fromList (Bi.first fst <$> dists)
      edges = Edges repEdges passEdges
  pure $ SpreadOp distMap edges
 where
  -- distribute a note to the two child slices
  distNote (note, n) = do
    dir <-
      sampleValue "noteSpreadDirection" (Categorical @3) $
        pInner
          . pNoteSpreadDirection
    to <- case dir of
      0 -> pure ToBoth
      1 -> do
        nother <-
          sampleValue "notesOnOtherSide" (Binomial $ n - 1) $
            pInner
              . pNotesOnOtherSide
        pure $ ToLeft $ n - nother
      _ -> do
        nother <-
          sampleValue "notesOnOtherSide" (Binomial $ n - 1) $
            pInner
              . pNotesOnOtherSide
        pure $ ToRight $ n - nother
    pure ((note, n), to)

observeSpread :: ContextDouble SPitch -> Spread SPitch -> PVObs ()
observeSpread (_sliceL, _transL, Notes sliceM, _transR, _sliceR) (SpreadOp obsDists (Edges repEdges passEdges)) =
  do
    -- observe note distribution
    dists <- mapM (observeNoteDist obsDists) $ MS.toOccurList sliceM
    let notesLeft = catMaybes $ flip fmap dists $ \((note, n), to) ->
          case to of
            ToRight dl -> if n - dl > 0 then Just note else Nothing
            _ -> Just note
        notesRight = catMaybes $ flip fmap dists $ \((note, n), to) ->
          case to of
            ToLeft dr -> if n - dr > 0 then Just note else Nothing
            _ -> Just note
    -- observe repetition edges
    sequence_ $ do
      -- List
      l <- notesLeft
      r <- notesRight
      guard $ pc l == pc r
      pure $
        observeValue
          "spreadRepeatEdge"
          Bernoulli
          (pInner . pSpreadRepetitionEdge)
          (S.member (Inner l, Inner r) repEdges)
    -- observe passing edges
    observePassing notesLeft notesRight pNewPassingMid passEdges
 where
  observeNoteDist distMap (parent, n) = case HM.lookup parent distMap of
    Nothing ->
      lift $ Left $ "Note " <> showNotation parent <> " is not distributed."
    Just dir -> do
      case dir of
        ToBoth -> do
          observeValue
            "noteSpreadDirection"
            (Categorical @3)
            (pInner . pNoteSpreadDirection)
            0
        ToLeft ndiff -> do
          observeValue
            "noteSpreadDirection"
            (Categorical @3)
            (pInner . pNoteSpreadDirection)
            1
          observeValue
            "notesOnOtherSide"
            (Binomial $ n - 1)
            (pInner . pNotesOnOtherSide)
            (n - ndiff)
        ToRight ndiff -> do
          observeValue
            "noteSpreadDirection"
            (Categorical @3)
            (pInner . pNoteSpreadDirection)
            2
          observeValue
            "notesOnOtherSide"
            (Binomial $ n - 1)
            (pInner . pNotesOnOtherSide)
            (n - ndiff)
      pure ((parent, n), dir)

samplePassing
  :: (_)
  => [SPitch]
  -> [SPitch]
  -> Accessor PVParamsInner Beta
  -> m (MS.MultiSet (InnerEdge SPitch))
samplePassing notesLeft notesRight pNewPassing =
  fmap (MS.fromList . concat) $ sequence $ do
    -- List
    -- DT.traceM $ "notesLeft (smp)" <> show notesLeft
    -- DT.traceM $ "notesRight (smp)" <> show notesRight
    l <- notesLeft
    r <- notesRight
    let step = iabs (pc l `pto` pc r)
    guard $ degree step >= 2 || (degree step == 1 && alteration step >= 0)
    -- DT.traceM $ "parent edge (sample)" <> show (l, r)
    pure $ do
      -- m
      n <- sampleValue "newPassing" Geometric0 $ pInner . pNewPassing
      pure $ replicate n (l, r)

observePassing
  :: [SPitch]
  -> [SPitch]
  -> Accessor PVParamsInner Beta
  -> MS.MultiSet (InnerEdge SPitch)
  -> PVObs ()
observePassing notesLeft notesRight pNewPassing edges = sequence_ $ do
  -- DT.traceM $ "edges (obs)" <> show edges
  -- DT.traceM $ "notesLeft (obs)" <> show notesLeft
  -- DT.traceM $ "notesRight (obs)" <> show notesRight
  l <- notesLeft
  r <- notesRight
  let step = iabs (pc l `pto` pc r)
  guard $ degree step >= 2 || (degree step == 1 && alteration step >= 0)
  -- DT.traceM $ "parent edge (obs)" <> show (l, r)
  pure $
    observeValue
      "newPassing"
      Geometric0
      (pInner . pNewPassing)
      (edges MS.! (l, r))
