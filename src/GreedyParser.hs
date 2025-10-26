{-# LANGUAGE PartialTypeSignatures #-}
{-# OPTIONS_GHC -Wno-incomplete-patterns #-}
{-# OPTIONS_HADDOCK ignore-exports #-}

{- | This module contains a simple greedy parser for path grammars.
 The grammar is provided by an evaluator ('Eval').
 In addition, the parser takes a policy function
 that picks a reduction option in each step.
-}
module GreedyParser where

-- TODO: add back export list once haddock's ignore-exports works again.
-- ( parseGreedy
-- , pickRandom
-- , parseRandom
-- , parseRandom'
-- ) where

import Common

import Control.Monad.Except
  ( ExceptT
  , MonadError (throwError)
  )
import Control.Monad.IO.Class
  ( MonadIO
  )
import Control.Monad.Trans.Class (lift)
import Data.Maybe
  ( catMaybes
  , mapMaybe
  , maybeToList
  )
import System.Random (initStdGen)
import System.Random.Stateful
  ( StatefulGen
  , newIOGenM
  , uniformRM
  )

-- * Parsing State

-- {- | A transition during greedy parsing.
--  Augments transition data with a flag
--  that indicates whether the transition is a transitive right (2nd) parent of a spread
--  and/or the direct parent of a left/single-split or thaw.
--  This allows the parser to enforce the following constraints:
--  - right splits are not allowed directly before a left split or freeze
--  - right splits
-- -}
-- data Trans tr = Trans
--   { gtContent :: !tr
--   -- ^ content of the transition
--   , gtSpread2nd :: !Bool
--   -- ^ flag that indicates (transitive) right parents of spreads
--   , gtStage1Parent :: !Bool
--   -- ^ flag that indicates that the edge is the parent of a freeze or left split,
--   -- preventing it from being used for a right split.
--   }
--   deriving (Show)

{- | The state of the greedy parse between steps.
 Generally, the current reduction consists of frozen transitions
 between the ⋊ and the current location
 and open transitions between the current location and ⋉.

 > ⋊==[1]==[2]==[3]——[4]——[5]——⋉
 >   └ frozen  ┘  | └   open  ┘
 >             midSlice (current position)
 >
 > frozen:   ==[2]==[1]==
 > midSlice: [3]
 > open:     ——[4]——[5]——

 This is the 'GSSemiOpen' case:
 The slice at the current pointer (@[3]@)
 is represented as an individual slice (@midSlice@).
 The frozen part is represented by a 'Path' of frozen transitions (@tr'@) and slices (@slc@).
 __in reverse direction__, i.e. from @midslice@ back to ⋊ (excluding ⋊).
 The open part is a 'Path' of open transitions (@tr@) and slices (@slc@)
 in forward direction from @midSlice@ up to ⋉.

 There are two special cases.
 All transitions can be frozen ('GSFrozen'),
 in which case state only contains the backward 'Path' of frozen transitions
 (excluding ⋊ and ⋉):

 > ⋊==[1]==[2]==[3]==⋉
 >                    └ current position
 > represented as: ==[3]==[2]==[1]==

 Or all transitions can be open ('GSOpen'),
 in which case the state is just the forward path of open transitions:

 > ⋊——[1]——[2]——[3]——⋉
 > └ current position
 > represented as: ——[1]——[2]——[3]——

 The open and semiopen case additionally have a list of operations in generative order,
 and a flag that indicates whether the previous step was a left operation,
 which would prevent continuing with a right unsplit.
-}
data GreedyState tr tr' slc op
  = GSFrozen !(Path (Maybe tr') slc)
  | GSSemiOpen
      { _gsFrozen :: !(Path (Maybe tr') slc)
      -- ^ frozen transitions and slices from current point leftward
      , _gsMidSlice :: !slc
      -- ^ the slice at the current posision between gsFrozen and gsOpen
      , _gsOpen :: !(Path tr slc)
      -- ^ non-frozen transitions and slices from current point rightward
      , _gsDeriv :: ![op]
      -- ^ derivation from current reduction to original surface
      }
  | GSOpen !(Path tr slc) ![op]

instance (Show slc, Show o) => Show (GreedyState tr tr' slc o) where
  show (GSFrozen frozen) = showFrozen frozen <> "⋉"
  show (GSOpen open _ops) = "⋊" <> showOpen open -- <> " " <> show ops
  show (GSSemiOpen frozen mid open _ops) =
    showFrozen frozen <> show mid <> showOpen open -- <> " " <> show ops

-- | Helper function for showing the frozen part of a piece.
showFrozen :: (Show slc) => Path tr' slc -> String
showFrozen path = "⋊" <> go path
 where
  go (PathEnd _) = "="
  go (Path _ a rst) = go rst <> show a <> "="

-- | Helper function for showing the open part of a piece.
showOpen :: (Show slc) => Path tr slc -> String
showOpen path = go path <> "⋉"
 where
  go (PathEnd _) = "-"
  go (Path _ a rst) = "-" <> show a <> go rst

gsOps :: GreedyState tr tr' slc op -> [op]
gsOps (GSFrozen _) = []
gsOps (GSOpen _ ops) = ops
gsOps (GSSemiOpen _ _ _ ops) = ops

-- * Parsing Actions

-- | Single parent of a parsing action
data SingleParent slc tr = SingleParent !(StartStop slc) !tr !(StartStop slc)
  deriving (Show)

{- | A parsing action (reduction step) with a single parent transition.
 Combines the parent elements with a single-transition derivation operation.
-}
data ActionSingle slc tr s f
  = ActionSingle
      (SingleParent slc tr)
      -- ^ parent transition (and adjacent slices)
      (LeftmostSingle s f)
      -- ^ single-transition operation
  deriving (Show)

-- | Single parent of a parsing action
data DoubleParent slc tr
  = DoubleParent
      !(StartStop slc)
      !tr
      !slc
      !tr
      !(StartStop slc)
  deriving (Show)

{- | A parsing action (reduction step) with two parent transitions.
 Combines the parent elements with a double-transition derivation operation.
-}
data ActionDouble slc tr s f h
  = ActionDouble
      (DoubleParent slc tr)
      -- ^ parent transitions and slice
      (LeftmostDouble s f h)
      -- ^ double-transition operation
  deriving (Show)

-- | An alias that combines 'ActionSingle' and 'ActionDouble', representing all possible reduction steps.
type Action slc tr s f h = Either (ActionSingle slc tr s f) (ActionDouble slc tr s f h)

{- | A helper function that checks whether an action:
- - is a double action and goes left ('Just True')
- - is a double action and goes right ('Just False')
- - is a single action ('Nothing', doesn't have to choose).
- (See 'opGoesLeft'.)
-}
actionGoesLeft :: Action slc tr s f h -> Maybe Bool
actionGoesLeft (Right (ActionDouble _ op)) = case op of
  LMDoubleFreezeLeft _ -> Just True
  LMDoubleSplitLeft _ -> Just True
  _ -> Just False
actionGoesLeft _ = Nothing

{- | A helper function that checks whether a derivation operation:
- - is a double op and goes left ('Just True')
- - is a double op and goes right ('Just False')
- - is a single op ('Nothing', doesn't have to choose).
- (See 'actionGoesLeft'.)
-}
opGoesLeft :: Leftmost s f h -> Maybe Bool
opGoesLeft (LMDouble lmd) = case lmd of
  LMDoubleFreezeLeft _ -> Just True
  LMDoubleSplitLeft _ -> Just True
  _ -> Just False
opGoesLeft _ = Nothing

-- * Parsing Algorithm

{- | Parse a piece in a greedy fashion.
 At each step, a policy chooses from the possible reduction actions,
 the reduction is applied, and parsing continues
 until the piece is fully reduced or no more reduction operations are available.
 Returns the full derivation from the top (@⋊——⋉@) or an error message.
-}
parseGreedy
  :: forall m tr tr' slc slc' s f h
   . (Monad m, MonadIO m, Show tr', Show slc, Show tr, Show s, Show f, Show h)
  => Eval tr tr' slc slc' h (Leftmost s f h)
  -- ^ the evaluator of the grammar to be used
  -> ([Action slc tr s f h] -> ExceptT String m (Action slc tr s f h))
  -- ^ the policy: picks a parsing action from a list of options
  -- (determines the 'Monad' @m@, e.g., for randomness).
  -> Path slc' tr'
  -- ^ the input piece
  -> ExceptT String m (Analysis s f h tr slc)
  -- ^ the full parse or an error message
parseGreedy eval pick input = do
  (top, deriv) <- parse state0
  pure $ Analysis deriv $ PathEnd top
 where
  state0 = initParseState eval input
  parse state = do
    result <- parseStep eval pick state
    case result of
      Left state' -> parse state'
      Right result -> pure result

{- | Initializes a parse state.
Takes an evaluator and a (frozen) input path.
Returns the parsing state that corresponds to the unparsed input.
-}
initParseState
  :: forall tr tr' slc slc' h v op
   . Eval tr tr' slc slc' h v
  -> Path slc' tr'
  -> GreedyState tr tr' slc op
initParseState eval input = GSFrozen $ wrapPath Nothing (reversePath input)
 where
  -- prepare the input: eval slices, wrap in Inner, add Start/Stop
  wrapPath :: Maybe tr' -> Path slc' tr' -> Path (Maybe tr') slc
  wrapPath eleft (PathEnd a) = Path eleft (evalSlice eval a) $ PathEnd Nothing
  wrapPath eleft (Path a e rst) =
    Path eleft (evalSlice eval a) $ wrapPath (Just e) rst

{- | A single greedy parse step.
 Enumerates a list of possible actions in the current state
 and applies a policy function to select an action.
 The resulting state is returned,
 wrapped in a monad transformer stack containing
 'String' exceptions and the monad of the policy.
-}
parseStep
  :: forall m tr tr' slc slc' s f h
   . (Monad m)
  => Eval tr tr' slc slc' h (Leftmost s f h)
  -- ^ the evaluator of the grammar to be used
  -> ([Action slc tr s f h] -> ExceptT String m (Action slc tr s f h))
  -- ^ the policy: picks a parsing action from a list of options
  -- (determines the 'Monad' @m@, e.g., for randomness).
  -> GreedyState tr tr' slc (Leftmost s f h)
  -- ^ the current parsing state
  -> ExceptT String m (Either (GreedyState tr tr' slc (Leftmost s f h)) (tr, [Leftmost s f h]))
  -- ^ either the next state or the result of the parse.
parseStep eval pick state = do
  -- liftIO $ putStrLn "" >> print state
  case state of
    -- case 1: everything frozen
    GSFrozen frozen -> case frozen of
      -- only one transition: unfreeze and terminate
      PathEnd trans -> do
        (thawed, op) <-
          pickSingle $
            collectThawSingle eval Start trans Stop
        finish (thawed, [LMSingle op])
      -- several transition: unfreeze last and continue
      Path t slice rst -> do
        (thawed, op) <- pickSingle $ collectThawSingle eval (Inner slice) t Stop
        continue $ GSSemiOpen rst slice (PathEnd thawed) [LMSingle op]

    -- case 2: everything open
    GSOpen open ops -> case open of
      -- only one transition: terminate
      PathEnd t -> finish (t, ops)
      -- two transitions: unsplit single and terminate
      Path tl slice (PathEnd tr) -> do
        (ttop, optop) <-
          pickSingle $
            collectUnsplitSingle eval Start tl slice tr Stop
        finish (ttop, LMSingle optop : ops)
      -- more than two transitions: pick double operation and continue
      Path tl sl (Path tm sr rst) -> do
        let doubles = collectDoubles eval Start tl sl tm sr rst (lastWasLeft ops)
        ((topl, tops, topr), op) <- pickDouble doubles
        continue $
          GSOpen
            (Path topl tops (pathSetHead rst topr))
            (LMDouble op : ops)

    -- case 3: some parts frozen, some open
    GSSemiOpen frozen mid open ops -> case open of
      -- only one open transition: thaw
      PathEnd topen -> case frozen of
        PathEnd tfrozen -> do
          ((thawed, _, _), op) <-
            pickDouble $
              collectThawLeft eval Start tfrozen mid topen Stop
          continue $ GSOpen (Path thawed mid open) (LMDouble op : ops)
        Path tfrozen sfrozen rstFrozen -> do
          ((thawed, _, _), op) <-
            pickDouble $
              collectThawLeft eval (Inner sfrozen) tfrozen mid topen Stop
          continue $
            GSSemiOpen
              rstFrozen
              sfrozen
              (Path thawed mid open)
              (LMDouble op : ops)
      -- two open transitions: thaw or unsplit single
      Path topenl sopen (PathEnd topenr) -> do
        let
          unsplits =
            Left <$> collectUnsplitSingle eval (Inner mid) topenl sopen topenr Stop
        case frozen of
          PathEnd tfrozen -> do
            let
              thaws =
                Right
                  <$> collectThawLeft eval Start tfrozen mid topenl (Inner sopen)
            action <- pick $ thaws <> unsplits
            case action of
              -- picked unsplit
              Left (ActionSingle (SingleParent _ parent _) op) ->
                continue $
                  GSSemiOpen
                    frozen
                    mid
                    (PathEnd parent)
                    (LMSingle op : ops)
              -- picked thaw
              Right (ActionDouble (DoubleParent _ thawed _ _ _) op) ->
                continue $ GSOpen (Path thawed mid open) (LMDouble op : ops)
          Path tfrozen sfrozen rstFrozen -> do
            let thaws =
                  Right
                    <$> collectThawLeft
                      eval
                      (Inner sfrozen)
                      tfrozen
                      mid
                      topenl
                      (Inner sopen)
            action <- pick $ thaws <> unsplits
            case action of
              -- picked unsplit
              Left (ActionSingle (SingleParent _ parent _) op) ->
                continue $
                  GSSemiOpen
                    frozen
                    mid
                    (PathEnd parent)
                    (LMSingle op : ops)
              -- picked thaw
              Right (ActionDouble (DoubleParent _ thawed _ _ _) op) ->
                continue $
                  GSSemiOpen
                    rstFrozen
                    sfrozen
                    (Path thawed mid open)
                    (LMDouble op : ops)
      -- more than two open transitions: thaw or any double operation
      Path topenl sopenl (Path topenm sopenr rstOpen) -> do
        let doubles =
              collectDoubles eval (Inner mid) topenl sopenl topenm sopenr rstOpen (lastWasLeft ops)
        case frozen of
          PathEnd tfrozen -> do
            let thaws =
                  collectThawLeft eval Start tfrozen mid topenl (Inner sopenl)
            action <- pickDouble $ thaws <> doubles
            case action of
              -- picked thaw
              ((thawed, _, _), op@(LMDoubleFreezeLeft _)) ->
                continue $ GSOpen (Path thawed mid open) (LMDouble op : ops)
              -- picked non-thaw
              ((topl, tops, topr), op) ->
                continue $
                  GSSemiOpen
                    frozen
                    mid
                    (Path topl tops (pathSetHead rstOpen topr))
                    (LMDouble op : ops)
          Path tfrozen sfrozen rstFrozen -> do
            let
              thaws =
                collectThawLeft
                  eval
                  (Inner sfrozen)
                  tfrozen
                  mid
                  topenl
                  (Inner sopenl)
            action <- pickDouble $ thaws <> doubles
            case action of
              -- picked thaw
              ((thawed, _, _), op@(LMDoubleFreezeLeft _)) ->
                continue $
                  GSSemiOpen
                    rstFrozen
                    sfrozen
                    (Path thawed mid open)
                    (LMDouble op : ops)
              -- picked non-thaw
              ((topl, tops, topr), op) ->
                continue $
                  GSSemiOpen
                    frozen
                    mid
                    (Path topl tops (pathSetHead rstOpen topr))
                    (LMDouble op : ops)
 where
  continue = pure . Left
  finish = pure . Right

  pickSingle
    :: [ActionSingle slc tr s f] -> ExceptT String m (tr, LeftmostSingle s f)
  pickSingle actions = do
    -- liftIO $ putStrLn $ "pickSingle " <> show actions
    action <- pick $ Left <$> actions
    case action of
      Left (ActionSingle (SingleParent _ top _) op) -> pure (top, op)
      Right _ -> throwError "pickSingle returned a double action"

  pickDouble
    :: [ActionDouble slc tr s f h]
    -> ExceptT String m ((tr, slc, tr), LeftmostDouble s f h)
  pickDouble actions = do
    -- liftIO $ putStrLn $ "pickDouble " <> show actions
    action <- pick $ Right <$> actions
    case action of
      Left _ -> throwError "pickDouble returned a single action"
      Right (ActionDouble (DoubleParent _ topl tops topr _) op) ->
        pure ((topl, tops, topr), op)

-- | Enumerates the list of possible actions in the current state
getActions
  :: forall m tr tr' slc slc' s f h
   . Eval tr tr' slc slc' h (Leftmost s f h)
  -- ^ the evaluator of the grammar to be used
  -> GreedyState tr tr' slc (Leftmost s f h)
  -- ^ the current parsing state
  -> [Action slc tr s f h]
  -- ^ the list of possible actions
getActions eval state =
  -- check which type of state we are in
  case state of
    -- case 1: everything frozen
    GSFrozen frozen -> case frozen of
      PathEnd trans -> Left <$> collectThawSingle eval Start trans Stop
      Path t slice rst -> Left <$> collectThawSingle eval (Inner slice) t Stop
    -- case 2: everything open
    GSOpen open ops -> case open of
      PathEnd _t -> []
      Path tl slice (PathEnd tr) -> Left <$> collectUnsplitSingle eval Start tl slice tr Stop
      Path tl sl (Path tm sr rst) -> Right <$> collectDoubles eval Start tl sl tm sr rst (lastWasLeft ops)
    -- case 3: some parts frozen, some open
    -- check how many transitions are open
    GSSemiOpen frozen mid open ops -> case open of
      -- only one open transition: thaw
      PathEnd topen -> Right <$> collectAllThawLeft eval frozen mid topen Stop
      -- two open transitions: thaw or unsplit single
      Path t1 s1 (PathEnd t2) ->
        let
          unsplits = collectUnsplitSingle eval (Inner mid) t1 s1 t2 Stop
          thaws = collectAllThawLeft eval frozen mid t1 (Inner s1)
         in
          (Left <$> unsplits) <> (Right <$> thaws)
      -- more than two open transitions: thaw or any double operation
      Path t1 s1 (Path t2 s2 rstOpen) -> do
        let doubles = collectDoubles eval (Inner mid) t1 s1 t2 s2 rstOpen (lastWasLeft ops)
            thaws = collectAllThawLeft eval frozen mid t1 (Inner s1)
        Right <$> (doubles <> thaws)

-- helper functions for getActions and parseStep
-- ---------------------------------------------

lastWasLeft :: [Leftmost s f h] -> Bool
lastWasLeft [] = False
lastWasLeft (op : _) = case op of
  LMSplitLeft _ -> True
  LMFreezeLeft _ -> True
  _ -> False

collectAllThawLeft
  :: Eval tr tr' slc slc' h (Leftmost s f h)
  -> Path (Maybe tr') slc
  -> slc
  -> tr
  -> StartStop slc
  -> [ActionDouble slc tr s f h]
collectAllThawLeft eval frozen sm tr sr =
  case frozen of
    PathEnd tfrozen -> collectThawLeft eval Start tfrozen sm tr sr
    Path tfrozen sl _ -> collectThawLeft eval (Inner sl) tfrozen sm tr sr

collectThawSingle
  :: Eval tr tr' slc slc' h (Leftmost s f h)
  -> StartStop slc
  -> Maybe tr'
  -> StartStop slc
  -> [ActionSingle slc tr s f]
collectThawSingle eval sl t sr =
  mapMaybe
    getAction
    (evalUnfreeze eval sl t sr True)
 where
  getAction (t', op) = case op of
    LMSingle sop -> Just $ ActionSingle (SingleParent sl t' sr) sop
    LMDouble _ -> Nothing

collectThawLeft
  :: Eval tr tr' slc slc' h (Leftmost s f h)
  -> StartStop slc
  -> Maybe tr'
  -> slc
  -> tr
  -> StartStop slc
  -> [ActionDouble slc tr s f h]
collectThawLeft eval sl tl sm tr sr =
  mapMaybe
    getAction
    (evalUnfreeze eval sl tl (Inner sm) False)
 where
  getAction (thawed, op) = case op of
    LMDouble dop ->
      Just $ ActionDouble (DoubleParent sl thawed sm tr sr) dop
    LMSingle _ -> Nothing

collectUnsplitSingle
  :: Eval tr tr' slc slc' h (Leftmost s f h)
  -> StartStop slc
  -> tr
  -> slc
  -> tr
  -> StartStop slc
  -> [ActionSingle slc tr s f]
collectUnsplitSingle eval sl tl sm tr sr =
  mapMaybe getAction $ evalUnsplit eval sl tl sm tr sr SingleOfOne
 where
  getAction (ttop, op) = case op of
    LMSingle sop -> Just $ ActionSingle (SingleParent sl ttop sr) sop
    LMDouble _ -> Nothing

collectUnsplitLeft
  :: Eval tr tr' slc slc' h (Leftmost s f h)
  -> StartStop slc
  -> tr
  -> slc
  -> tr
  -> slc
  -> tr
  -> StartStop slc
  -> [ActionDouble slc tr s f h]
collectUnsplitLeft eval sstart tl sl tm sr tr send =
  mapMaybe getAction $ evalUnsplit eval sstart tl sl tm (Inner sr) LeftOfTwo
 where
  getAction (ttop, op) = case op of
    LMSingle _ -> Nothing
    LMDouble dop ->
      Just $
        ActionDouble
          (DoubleParent sstart ttop sr tr send)
          dop

collectUnsplitRight
  :: Eval tr tr' slc slc' h (Leftmost s f h)
  -> StartStop slc
  -> tr
  -> slc
  -> tr
  -> slc
  -> tr
  -> StartStop slc
  -> Bool
  -> [ActionDouble slc tr s f h]
collectUnsplitRight eval sstart tl sl tm sr tr send afterLeft
  | afterLeft = []
  | otherwise =
      mapMaybe getAction $
        evalUnsplit eval (Inner sl) tm sr tr send RightOfTwo
 where
  getAction (ttop, op) = case op of
    LMSingle _ -> Nothing
    LMDouble dop ->
      Just $ ActionDouble (DoubleParent sstart tl sl ttop send) dop

collectUnspreads
  :: Eval tr tr' slc slc' h (Leftmost s f h)
  -> StartStop slc
  -> tr
  -> slc
  -> tr
  -> slc
  -> tr
  -> StartStop slc
  -> [ActionDouble slc tr s f h]
collectUnspreads eval sstart tl sl tm sr tr send =
  catMaybes $ do
    -- List
    (sTop, us, op) <- evalUnspreadMiddle eval (sl, tm, sr)
    lTop <- evalUnspreadLeft eval (tl, sl) sTop us
    rTop <- evalUnspreadRight eval (sr, tr) sTop us
    pure $ getAction lTop sTop rTop op
 where
  getAction lTop sTop rTop op = case op of
    LMSingle _ -> Nothing
    LMDouble dop ->
      Just $
        ActionDouble
          (DoubleParent sstart lTop sTop rTop send)
          dop

collectDoubles
  :: Eval tr tr' slc slc' h (Leftmost s f h)
  -> StartStop slc
  -> tr
  -> slc
  -> tr
  -> slc
  -> Path tr slc
  -> Bool
  -> [ActionDouble slc tr s f h]
collectDoubles eval sstart tl sl tm sr rst afterLeft = leftUnsplits <> rightUnsplits <> unspreads
 where
  (tr, send) = case rst of
    PathEnd t -> (t, Stop)
    Path t s _ -> (t, Inner s)
  leftUnsplits = collectUnsplitLeft eval sstart tl sl tm sr tr send
  rightUnsplits = collectUnsplitRight eval sstart tl sl tm sr tr send afterLeft
  unspreads = collectUnspreads eval sstart tl sl tm sr tr send

{- | A policy that picks the next action at random.
 Must be partially applied with a random generator before passing to 'parseGreedy'.
-}
pickRandom :: (StatefulGen g m) => g -> [slc] -> ExceptT String m slc
pickRandom _ [] = throwError "No candidates for pickRandom!"
pickRandom gen xs = do
  i <- lift $ uniformRM (0, length xs - 1) gen
  pure $ xs !! i

-- * Applying actions

-- | Apply an action to a parsing state.
applyAction
  :: forall m tr tr' slc slc' s f h
   . GreedyState tr tr' slc (Leftmost s f h)
  -- ^ the current parsing state
  -> Action slc tr s f h
  -- ^ the action to be applied
  -> Either String (Either (GreedyState tr tr' slc (Leftmost s f h)) (tr, [Leftmost s f h]))
  -- ^ either the next state or an error message
applyAction state action =
  case state of
    -- case 1: everything frozen
    GSFrozen frozen ->
      case action of
        Left (ActionSingle (SingleParent _ top _) op@(LMSingleFreeze _)) ->
          case frozen of
            PathEnd _ -> finish top [LMSingle op]
            Path _ slc rst -> continue $ GSSemiOpen rst slc (PathEnd top) [LMSingle op]
        _ -> Left "cannot apply this operation to frozen state"
    -- case 2: everything open
    GSOpen open ops -> case open of
      PathEnd tr -> finish tr ops
      Path tl slice (PathEnd tr) ->
        case action of
          Left (ActionSingle (SingleParent _ top _) op@(LMSingleSplit _)) ->
            finish top (LMSingle op : ops)
          _ -> Left "cannot apply this operation to 2 open transitions"
      Path tl sl (Path tm sm rst) ->
        case action of
          Right (ActionDouble _ (LMDoubleFreezeLeft _)) ->
            Left "cannot apply unfreeze in open state"
          Right (ActionDouble (DoubleParent _ topl tops topr _) op) ->
            continue $ GSOpen (Path topl tops (pathSetHead rst topr)) (LMDouble op : ops)
    -- case 3: some open some closed
    GSSemiOpen frozen mid open ops -> case action of
      -- single op: unfreeze or unsplit
      Left (ActionSingle (SingleParent _ top _) op) -> case op of
        -- unfreeze single
        LMSingleFreeze _ -> case frozen of
          PathEnd _ -> continue $ GSOpen (Path top mid open) (LMSingle op : ops)
          Path _ frs frrest ->
            continue $ GSSemiOpen frrest frs (Path top mid open) (LMSingle op : ops)
        -- unsplit single
        LMSingleSplit _ -> case open of
          PathEnd _ -> Left "cannot apply unsplit to single open transition"
          Path topenl sopen rstopen ->
            continue $ GSSemiOpen frozen mid (pathSetHead rstopen top) (LMSingle op : ops)
      -- double op:
      Right (ActionDouble (DoubleParent _ topl tops topr _) op) -> case op of
        -- unfreeze left
        LMDoubleFreezeLeft _ -> case frozen of
          PathEnd _ -> continue $ GSOpen (Path topl mid open) (LMDouble op : ops)
          Path _ mid' frozen' ->
            continue $ GSSemiOpen frozen' mid' (Path topl mid open) (LMDouble op : ops)
        -- unsplit left
        LMDoubleSplitLeft _ -> case open of
          PathEnd _ -> Left "cannot apply unsplit to single open transition"
          Path _ sopen rst ->
            continue $ GSSemiOpen frozen mid (pathSetHead rst topl) (LMDouble op : ops)
        -- unsplit right or unspread
        _ -> case open of
          Path _tl _sl (Path _tm _sr rst) ->
            continue $ GSSemiOpen frozen mid (Path topl tops (pathSetHead rst topr)) (LMDouble op : ops)
          _ -> Left "cannot apply unsplit right or unspread to less than 3 open transitions"
 where
  continue = pure . Left
  finish top ops = pure $ Right (top, ops)

-- * Entry Points

-- | Parse a piece randomly using a fresh random number generator.
parseRandom
  :: (Show tr', Show slc, Show tr, Show s, Show f, Show h)
  => Eval tr tr' slc slc' h (Leftmost s f h)
  -- ^ the grammar's evaluator
  -> Path slc' tr'
  -- ^ the input piece
  -> ExceptT String IO (Analysis s f h tr slc)
  -- ^ a random reduction of the piece (or an error message)
parseRandom eval input = do
  gen <- lift initStdGen
  mgen <- lift $ newIOGenM gen
  parseGreedy eval (pickRandom mgen) input

-- | Parse a piece randomly using an existing random number generator.
parseRandom'
  :: (Show tr', Show slc, Show tr, Show s, Show f, Show h, StatefulGen g IO)
  => g
  -- ^ a random number generator
  -> Eval tr tr' slc slc' h (Leftmost s f h)
  -- ^ the grammar's evaluator
  -> Path slc' tr'
  -- ^ the input piece
  -> ExceptT String IO (Analysis s f h tr slc)
  -- ^ a random reduction of the piece (or an error message)
parseRandom' mgen eval input = do
  parseGreedy eval (pickRandom mgen) input
