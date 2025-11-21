{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DataKinds #-}

module Main where

import Common
import GreedyParser
import PVGrammar
import RL qualified

import Criterion.Main
import Data.List.NonEmpty qualified as NE
import Torch.Typed qualified as TT

{- | Benchmark the forward pass with varying numbers of actions.
Batches the actions before benchmarking.
-}
benchForward :: (RL.IsValidDevice dev) => String -> RL.QModel dev -> RL.PVState -> RL.PVAction -> Benchmark
benchForward what model state action =
  bgroup
    ("forward " <> what)
    [ benchN 1
    , benchN 4
    , benchN 16
    , benchN 64
    , benchN 256
    -- , benchN 1024
    -- , benchN 4096
    ]
 where
  benchN n = RL.withBatchedEncoding state (action NE.:| replicate (n - 1) action) $ \enc ->
    enc `seq` bench (show n <> " actions") $ nf (RL.runBatchedPolicy model) enc

main :: IO ()
main = do
  modelCPU <- RL.mkQModel @'(TT.CPU, 0)
  modelGPU <- RL.mkQModel @'(TT.CUDA, 0)
  defaultMain
    [ benchForward "CPU" modelCPU state action
    , benchForward "GPU" modelGPU state action
    ]
 where
  -- TODO: better states and actions
  state :: RL.PVState
  state =
    GSSemiOpen
      (Path Nothing (Notes mempty) $ PathEnd Nothing)
      (Notes mempty)
      (Path mempty (Notes mempty) $ PathEnd mempty)
      []
  action :: RL.PVAction
  action = Left $ ActionSingle (SingleParent Start mempty Stop) (LMSingleFreeze $ FreezeOp mempty)
