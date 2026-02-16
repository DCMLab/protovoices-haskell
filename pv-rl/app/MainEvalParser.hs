{-# LANGUAGE DataKinds #-}
{-# OPTIONS_GHC -O0 #-}

module Main where

import Common
import CommonMain
import GreedyParser
import PVGrammar
import PVGrammar.Parse
import PVGrammar.Prob.Simple
import RL

import Inference.Conjugate
import Musicology.Pitch.Spelled

import Torch qualified as T
import Torch.Typed qualified as TT

import Control.Monad (forM, replicateM)
import Control.Monad.Trans (lift)
import Control.Monad.Trans.Except qualified as ET
import Data.Either.Extra (maybeToEither, rights)
import Data.List qualified as L
import Data.List.NonEmpty qualified as NE
import Data.Maybe (catMaybes, listToMaybe)
import Data.Vector qualified as V
import Debug.Trace qualified as DT
import Graphics.Matplotlib qualified as Plt
import System.FilePath ((</>))
import System.Random.MWC qualified as MWC
import System.Random.MWC.Distributions qualified as MWC

-- parseRL
--   :: forall dev hidden
--    . (ValidParams dev hidden)
--   => QModel dev hidden
--   -> Path [Note SPitch] [Edge SPitch]
--   -> IO (Either String (PVAnalysis SPitch))
-- parseRL !actor !input = case take 200 $ getActions eval s0 of
--   [] -> pure $ Left "cannot parse: no possible actions for first step!"
--   (a : as) -> ET.runExceptT $ go s0 (a NE.:| as)
--  where
--   s0 = initParseState eval input
--   eval = protoVoiceEvaluator
--   go !state !actions = do
--     let
--       -- encodings = RL.encodeStep state <$> actions
--       -- probs = T.softmax (T.Dim 0) $ T.cat (T.Dim 0) $ TT.toDynamic . RL.forwardPolicy actor <$> encodings
--       -- showTensor t = "- " <> show (T.device $ DS.force t) <> "\n"
--       -- checkEncoding enc = DT.trace (concatMap showTensor $ RL.flattenTensors enc) 0
--       !probs = withBatchedEncoding state actions (runBatchedPolicy 1 actor)
--       !best = T.asValue $ T.argmax (T.Dim 0) T.KeepDim probs :: Int
--       -- !dummy = RL.withBatchedEncoding state actions DS.rnf
--       -- best = 0
--       action = actions NE.!! best
--     state' <- ET.except $ applyAction state action
--     let actions' = case state' of
--           Left nextState -> NE.nonEmpty $ take 200 $ getActions eval nextState
--           Right _ -> Nothing
--     case (state', actions') of
--       (Left s, Nothing) -> do
--         lift $ appendFile "incomplete.log" $ show state
--         lift $ putStr "!"
--         ET.throwE "cannot parse: no possible actions in non-terminal state:"
--       (Left s', Just a') -> go s' a'
--       (Right (top, deriv), _) -> do
--         let ana = Analysis deriv (PathEnd top)
--         pure ana

sampleBaseline
  :: MWC.GenIO
  -> Path [(Note SPitch)] [Edge SPitch]
  -> IO (Either String (PVAnalysis SPitch))
sampleBaseline gen surface =
  ET.runExceptT $ parseGreedy protoVoiceEvaluator (pickRandom gen) surface
 where
  pickRandom _ _ [] = ET.throwE "no actions"
  pickRandom gen _state actions = do
    i <- MWC.uniformRM (0, length actions - 1) gen
    pure $ actions !! i

sampleModel
  :: forall dev hidden
   . (ValidParams dev hidden)
  => QModel hidden dev
  -> MWC.GenIO
  -> Path [(Note SPitch)] [Edge SPitch]
  -> IO (Either String (PVAnalysis SPitch))
sampleModel model gen surface =
  ET.runExceptT $ parseGreedy protoVoiceEvaluator (pickAction gen) surface
 where
  pickAction _ _ [] = ET.throwE "no actions"
  pickAction gen state actions@(a : as) = do
    let actionsNE = a NE.:| take 199 as
        policy = dynPolicy $ withBatchedEncoding @dev state actionsNE (runBatchedPolicy 1 model)
        probs = V.fromList $ T.asValue $ T.toDType T.Double policy
    i <- lift $ MWC.categorical probs gen
    pure $ actions !! i

sampleLocal
  :: Probs PVParams
  -> MWC.GenIO
  -> Path [(Note SPitch)] [Edge SPitch]
  -> IO (Either String (PVAnalysis SPitch))
sampleLocal params gen surface =
  ET.runExceptT $ parseGreedy protoVoiceEvaluator (pickAction gen) surface
 where
  singleTop (SingleParent sl t sr) = (sl, t, sr)
  doubleTop (DoubleParent sl tl sm tr sr) = (sl, tl, sm, tr, sr)

  actionProb state action = case result of
    Left _err -> 0
    Right Nothing -> 0
    Right (Just (_, logprob)) -> exp logprob
   where
    ops = gsOps state
    decision = opGoesLeft =<< listToMaybe (drop 1 ops)
    result = case action of
      Left (ActionSingle top op) -> evalSingleStep params (singleTop top) op decision
      Right (ActionDouble top op) -> evalDoubleStep params (doubleTop top) op decision

  normalize weights = fmap (/ total) weights
   where
    total' = V.sum weights
    total = if total' == 0 then 1 else total'

  pickAction _ _ [] = ET.throwE "no actions"
  pickAction gen state actions = do
    let probs = normalize $ V.fromList $ actionProb state <$> take 200 actions
    i <- lift $ MWC.categorical probs gen
    pure $ actions !! i

maxModel
  :: forall dev hidden
   . (ValidParams dev hidden)
  => QModel hidden dev
  -> Path [(Note SPitch)] [Edge SPitch]
  -> IO (Either String (PVAnalysis SPitch))
maxModel model surface =
  ET.runExceptT $ parseGreedy protoVoiceEvaluator pickAction surface
 where
  pickAction _ [] = ET.throwE "no actions"
  pickAction state actions@(a : as) = do
    let actionsNE = a NE.:| take 199 as
        policy = dynPolicy $ withBatchedEncoding @dev state actionsNE (runBatchedPolicy 1 model)
        best = T.asValue $ T.argmax (T.Dim 0) T.KeepDim policy :: Int
    pure $ actions !! best

sampleAndEval probs genAna = do
  anaE <- genAna
  pure $ do
    -- Either
    ana <- anaE
    trace <- observeDerivation ana
    (_, lprob) <-
      maybeToEither "could not evaluate probability" $
        evalTraceLogP probs trace (sampleDerivation $ anaTop ana)
    Right lprob

countNotes :: Path [a] b -> Int
countNotes (PathEnd notes) = length notes
countNotes (Path notes _edges rst) = length notes + countNotes rst

mainCompare = do
  Right hyper <- loadPVHyper "posterior.json"
  gen <- MWC.createSystemRandom
  model <- loadQModel @Device @16 "rl/actor-imit-e300-nb32-bs128-h16-lr.01-norm-noattn.ht"
  let probs = expectedProbs @PVParams hyper
  examples <- loadDir (dataDir </> "theory-article") []
  perplexities <- forM examples $ \(name, _ana, trace, surface) -> do
    putStrLn $ "example " <> name
    case evalTraceLogP probs trace sampleDerivation' of
      Nothing -> do
        putStrLn "could not evaluate trace"
        pure Nothing
      Just (_, logprob) -> do
        let nnotes = fromIntegral $ countNotes surface
            toPerp logp = negate (logp / nnotes)
            logppn = toPerp logprob
        putStrLn $ "logppn (annot):" <> show logppn

        baselines <- replicateM 100 $ sampleAndEval probs $ sampleBaseline gen surface
        let baselppns = fmap toPerp $ rights baselines
        putStrLn $ "logppn (base):" <> show (mean baselppns)

        modellks <- replicateM 100 $ sampleAndEval probs $ sampleModel model gen surface
        let modelppns = fmap toPerp $ rights modellks
        putStrLn $ "logppn (model):" <> show (mean modelppns)

        locallks <- replicateM 100 $ sampleAndEval probs $ sampleLocal probs gen surface
        let localppns = fmap toPerp $ rights locallks
        putStrLn $ "logppn (local):" <> show (mean localppns)

        best <- sampleAndEval probs $ maxModel model surface
        putStrLn $ "greedy (model):" <> show (fmap toPerp best)
        -- putStrLn $ "best (model):" <> show (minimum modelppns)
        -- putStrLn $ "got " <> show (length $ rights baselines) <> " baselines"
        -- putStrLn $ "got " <> show (length $ rights modellks) <> " model derivs"
        -- putStrLn $ "got " <> show (length $ rights locallks) <> " local derivs"
        pure $ Just (logppn, baselppns, modelppns, localppns, name)
  let (logppns, baselppns, modelppns, localppns, names) = L.unzip5 $ catMaybes perplexities
  showRelPerplexity logppns baselppns modelppns localppns names
  pure ()

a % b = a Plt.% Plt.mp Plt.# b
infixl 5 %

showPerplexity
  :: [Double] -> [[Double]] -> [[Double]] -> [String] -> IO (Either String String)
showPerplexity logppns blogppns mlogppns testpieces =
  Plt.file "rl/perplexity.svg" $
    Plt.readData (logppns, blogppns, mlogppns, testpieces)
      % "import numpy as np"
      % "import pandas as pd"
      % "import seaborn as sns"
      % "sns.set_theme()"
      -- % "from matplotlib.lines import Line2D"
      % "(logppns, blogppns, mlogppns, pieces) = tuple(data)"
      % "baselines = pd.concat([pd.DataFrame({'logppn': np.array(bppns), 'piece': piece, 'group': 'random'}) for bppns, piece in zip(blogppns, pieces)])"
      % "modelppns = pd.concat([pd.DataFrame({'logppn': np.array(mppns), 'piece': piece, 'group': 'model'}) for mppns, piece in zip(mlogppns, pieces)])"
      % "testscores = pd.DataFrame({'logppn': np.array(logppns), 'piece': np.array(pieces), 'group': 'annotated'})"
      % "df = pd.concat([testscores, baselines, modelppns])"
      % "colors = sns.color_palette()"
      % "fig, ax = plot.subplots(figsize=(9,6))"
      % "sns.stripplot(df, y='piece', x='logppn', hue='group', ax=ax)"
      % "ax.set_xlabel('log-perplexity per note')"
      % "ax.invert_yaxis()"
      % "fig.tight_layout()"

-- % "fig.savefig('rl/perplexity.pdf')"
-- % "fig.savefig('rl/perplexity.png')"

showRelPerplexity
  :: [Double] -> [[Double]] -> [[Double]] -> [[Double]] -> [String] -> IO (Either String String)
showRelPerplexity logppns base model local testpieces =
  Plt.file "rl/perplexity.svg" $
    Plt.readData (logppns, base, model, local, testpieces)
      % "import numpy as np"
      % "import pandas as pd"
      % "import seaborn as sns"
      % "sns.set_theme()"
      % "(logppns, base, model, locals, pieces) = tuple(data)"
      % "baseppns = pd.concat([pd.DataFrame({'logppn': np.array(bppns)-annot, 'piece': piece, 'group': 'random'}) for bppns, piece, annot in zip(base, pieces, logppns)])"
      % "modelppns = pd.concat([pd.DataFrame({'logppn': np.array(mppns)-annot, 'piece': piece, 'group': 'model'}) for mppns, piece, annot in zip(model, pieces, logppns)])"
      % "localppns = pd.concat([pd.DataFrame({'logppn': np.array(lppns)-annot, 'piece': piece, 'group': 'local'}) for lppns, piece, annot in zip(locals, pieces, logppns)])"
      % "df = pd.concat([baseppns, modelppns, localppns])"
      % "colors = sns.color_palette()"
      % "fig, ax = plot.subplots(figsize=(9,12))"
      % "sns.stripplot(df, y='piece', x='logppn', hue='group', dodge=True, ax=ax)"
      % "ax.set_xlabel('log-perplexity per note (nats), relative to annotation')"
      % "fig.tight_layout()"

type Device = '(TT.CPU, 0)
type Hidden = 8

main = mainCompare
