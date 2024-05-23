{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE TypeFamilies #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}

-- | Common types and functionality that are used throughout the model.
module Common
  ( -- * Paths #paths#

    -- | Paths encode sequences of alternating objects (such as nodes and edges).
    -- They are often used to encode sequences of slices and transitions.
    -- Note that dependending on context,
    -- both slice-transition-slice and transition-slice-transition orders are used.
    Path (..)
  , pathLen
  , pathHead
  , pathSetHead
  , mapArounds
  , mapAroundsWithIndex
  , mapBetweens
  , reversePath
  , pathArounds
  , pathBetweens

    -- * StartStop #startstop#

    -- | 'StartStop' is a wrapper that augments a type with special values for beginning and end.
  , StartStop (..)
  , onlyInner
  , getInner
  , getInnerE
  , isInner
  , isStart
  , isStop
  , distStartStop

    -- * Evaluators #evals#

    -- | Evaluators ('Eval') are the main parsing interface for a grammar.
    -- They bundle a number of functions that compute local "completions"
    -- (i.e., parent objects and generative operations)
    -- from child objects.
    -- Parsers use these evaluators to generically parse an input sequence
    -- since all the grammar-specific parsing code is provided by the evaluator.
    --
    -- Evaluators can be transformed and combined using 'mapEvalScore' and 'productEval' respectively.
  , SplitType (..)
  , UnspreadMiddle
  , UnspreadLeft
  , UnspreadRight
  , Unsplit
  , Eval (..)
  , IsLast
  , mapEvalScore
  , productEval

    -- * Special Restricting Evaluators #special-evals#

    -- | Some special evaluators that can be combined with grammar-specific evaluators
    -- to restrict the possibile derivations.
  , RightBranchSpread (..)
  , evalRightBranchSpread
  , rightBranchSpread
  , Merged (..)
  , evalSplitBeforeSpread
  , splitFirst

    -- * Leftmost Derivations #leftmost#

    -- | Derivations can be represented as lists of operations in leftmost-first order.
    -- In this order, each operation (split, spread, or freeze)
    -- is applied to the leftmost non-terminal transition(s).
    -- $leftmostdoc
  , Leftmost
    ( LMDouble
    , LMFreezeLeft
    , LMFreezeOnly
    , LMSingle
    , LMSplitLeft
    , LMSplitOnly
    , LMSplitRight
    , LMSpread
    )
  , LeftmostSingle (..)
  , LeftmostDouble (..)
  , Analysis (..)
  , debugAnalysis
  , mkLeftmostEval

    -- * Monadic Interface for Constructing Derivations #monadicDeriv#
    -- $monadicdoc
  , PartialDerivation (..)
  , DerivationInfo
  , IndexedWriter
  , itell
  , DerivationAction (..)
  , buildDerivation
  , buildPartialDerivation
  , split
  , freeze
  , splitRight
  , spread

    -- * Derivations Semiring #derivSemiring#

    -- | A generic semiring that represents a collection of derivations as prefix trees.
  , Derivations (..)
  , mapDerivations
  , flattenDerivations
  , flattenDerivationsRed
  , firstDerivation

    -- * Utilities #utils#
  , traceLevel
  , traceIf
  , showTex
  , showTexT
  ) where

import Control.DeepSeq (NFData)
import Control.Monad (when)
import Control.Monad.Except
  ( ExceptT
  , runExceptT
  )
import Control.Monad.Indexed qualified as MI
import Control.Monad.Trans (lift)
import Control.Monad.Trans.Except (except)
import Control.Monad.Writer.Strict qualified as MW
import Data.Aeson
  ( FromJSON (..)
  , ToJSON (..)
  , (.:)
  )
import Data.Aeson qualified as Aeson
import Data.Aeson.Types (unexpected)
import Data.Aeson.Types qualified as Aeson
import Data.Bifunctor
  ( Bifunctor
  , bimap
  , second
  )
import Data.Hashable (Hashable)
import Data.Kind (Type)
import Data.Semigroup (stimesMonoid)
import Data.Semiring qualified as R
import Data.Set qualified as S
import Data.Text qualified as T
import Data.Typeable (Proxy (Proxy))
import Debug.Trace (trace)
import GHC.Generics (Generic)
import GHC.TypeNats
  ( KnownNat
  , Nat
  , natVal
  , type (+)
  , type (-)
  , type (<=)
  )
import GHC.Unicode (toLower)
import Musicology.Pitch (Notation (..))
import Text.ParserCombinators.ReadP qualified as ReadP

-- Path: sequences of alternating objects
-- ======================================

{- | A Path is a datastructure that represents a sequence of alternating objects,
 /arounds/ and /betweens/,
 starting and ending with the same type.
 An example would be a path in a graph,
 starting and ending with a node with edges in-between.
-}
data Path around between
  = Path !around !between !(Path around between)
  | PathEnd !around
  deriving (Eq, Ord, Generic, Functor)

instance Bifunctor Path where
  bimap fa _ (PathEnd a) = PathEnd (fa a)
  bimap fa fb (Path a b rst) = Path (fa a) (fb b) $ bimap fa fb rst

instance (Show a, Show b) => Show (Path a b) where
  show (Path a b rst) = show a <> "\n+-" <> show b <> "\n" <> show rst
  show (PathEnd a) = show a

-- | Returns the number of /arounds/ in the path.
pathLen :: Path a b -> Int
pathLen (Path _ _ rest) = pathLen rest + 1
pathLen (PathEnd _) = 1

-- | Returns the first /around/ in the path.
pathHead :: Path a b -> a
pathHead (Path l _ _) = l
pathHead (PathEnd l) = l

-- | Replaces the first /around/ in the path.
pathSetHead :: Path a b -> a -> Path a b
pathSetHead (Path _ b rst) a' = Path a' b rst
pathSetHead (PathEnd _) a' = PathEnd a'

-- | Maps a function over every /around/ in the path.
mapArounds :: (a -> a') -> Path a b -> Path a' b
mapArounds f (Path a b rest) = Path (f a) b $ mapArounds f rest
mapArounds f (PathEnd a) = PathEnd (f a)

-- | Maps a function over every /around/ in the path together with its index.
mapAroundsWithIndex :: Int -> (Int -> a -> a') -> Path a b -> Path a' b
mapAroundsWithIndex i f (Path a b rest) =
  Path (f i a) b (mapAroundsWithIndex (i + 1) f rest)
mapAroundsWithIndex i f (PathEnd a) = PathEnd (f i a)

-- | Maps a function over every /between/ and its adjacent /arounds/ in the path.
mapBetweens :: (a -> b -> a -> c) -> Path a b -> [c]
mapBetweens f (Path al b rest) = f al b ar : mapBetweens f rest
 where
  ar = pathHead rest
mapBetweens _ (PathEnd _) = []

-- | Reverses the path.
reversePath :: Path a b -> Path a b
reversePath path = case path of
  PathEnd end -> PathEnd end
  Path a b rest -> go b rest (PathEnd a)
 where
  go b (PathEnd aEnd) acc = Path aEnd b acc
  go b1 (Path a b2 rest) acc = go b2 rest $ Path a b1 acc

-- | Returns the list of /arounds/  in the path.
pathArounds :: Path a b -> [a]
pathArounds (Path a _ rst) = a : pathArounds rst
pathArounds (PathEnd a) = [a]

-- | Returns the list of /betweens/ in the path).
pathBetweens :: Path a b -> [b]
pathBetweens (Path _ b rst) = b : pathBetweens rst
pathBetweens _ = []

-- StartStop
-- =========

{- | A container type that augements the type @a@
 with symbols for beginning ('Start', ⋊) and end ('Stop', ⋉).
 Every other value is wrapped in an @Inner@ constructor.
-}
data StartStop a
  = Start
  | Inner !a
  | Stop
  deriving (Ord, Eq, Generic, NFData, Hashable, Functor, Foldable, Traversable)

-- some instances for StartStop

instance (Show a) => Show (StartStop a) where
  show Start = "⋊"
  show Stop = "⋉"
  show (Inner a) = show a

instance (Notation a) => Notation (StartStop a) where
  showNotation Start = "⋊"
  showNotation Stop = "⋉"
  showNotation (Inner a) = showNotation a
  parseNotation = ReadP.pfail

instance (FromJSON a) => FromJSON (StartStop a) where
  parseJSON (Aeson.String "start") = pure Start
  parseJSON (Aeson.String "stop") = pure Stop
  parseJSON other = Inner <$> parseJSON other

-- some helper functions for StartStop

{- | From a list of @StartStop@s returns only the elements that are not @:⋊@ or @:⋉@,
 unwrapped to their original type.
-}
onlyInner :: [StartStop a] -> [a]
onlyInner [] = []
onlyInner (Inner a : rst) = a : onlyInner rst
onlyInner (_ : rst) = onlyInner rst

-- | Returns the content of an 'Inner', or 'Nothing'.
getInner :: StartStop a -> Maybe a
getInner (Inner a) = Just a
getInner _ = Nothing

-- | Returns the content of an 'Inner', or a 'Left' with an error message.
getInnerE :: StartStop a -> Either String a
getInnerE (Inner a) = Right a
getInnerE Start = Left "expected inner but found ⋊"
getInnerE Stop = Left "expected inner but found ⋉"

-- | Returns 'True' iff the argument is an 'Inner'.
isInner :: StartStop a -> Bool
isInner (Inner _) = True
isInner _ = False

-- | Returns 'True' iff the argument is 'Start'.
isStart :: StartStop a -> Bool
isStart Start = True
isStart _ = False

-- | Returns 'True' iff the argument is 'Stop'.
isStop :: StartStop a -> Bool
isStop Stop = True
isStop _ = False

-- | Turns a pair within a 'StartStop' into a pair of 'StartStop's
distStartStop :: StartStop (a, b) -> (StartStop a, StartStop b)
distStartStop Start = (Start, Start)
distStartStop Stop = (Stop, Stop)
distStartStop (Inner (a, b)) = (Inner a, Inner b)

-- evaluator interface
-- ===================

-- | A flag indicating whether an operation is performed on the last transition.
type IsLast = Bool

{- | A flag that indicates where a split has been performed,
 on the left transition, the right transition, or the only transition
-}
data SplitType
  = LeftOfTwo
  | RightOfTwo
  | SingleOfOne

{- | An evaluator for unspreads.
 Takes the two child slices and the middle transition.
 Returns the parent slice and the spread operation, if possible.
-}
type UnspreadMiddle tr slc v = (slc, tr, slc) -> Maybe (slc, v)

{- | An evaluator returning the possible left parent edges of an unspread.
 The first argument is a pair of left child transition and left child slice.
 The second argument is the parent slice.
-}
type UnspreadLeft tr slc = (tr, slc) -> slc -> [tr]

{- | An evaluator returning the possible right parent edges of an unspread.
 The first argument is a pair of right child slice and right child transition.
 The second argument is the parent slice.
-}
type UnspreadRight tr slc = (slc, tr) -> slc -> [tr]

{- | An evaluator for unsplits.
 Returns possible unsplits of a given pair of transitions.
-}
type Unsplit tr slc v =
  StartStop slc -> tr -> slc -> tr -> StartStop slc -> SplitType -> [(tr, v)]

{- | A combined evaluator for unsplits, unspreads, and unfreezes.
 Additionally, contains a function for mapping terminal slices to derivation slices.
-}
data Eval tr tr' slc slc' v = Eval
  { evalUnspreadMiddle :: !(UnspreadMiddle tr slc v)
  , evalUnspreadLeft :: !(UnspreadLeft tr slc)
  , evalUnspreadRight :: !(UnspreadRight tr slc)
  , evalUnsplit :: !(Unsplit tr slc v)
  , evalUnfreeze
      :: !(StartStop slc -> Maybe tr' -> StartStop slc -> IsLast -> [(tr, v)])
  , evalSlice :: !(slc' -> slc)
  }

-- | Maps a function over all scores produced by the evaluator.
mapEvalScore :: (v -> w) -> Eval tr tr' slc slc' v -> Eval tr tr' slc slc' w
mapEvalScore f (Eval unspreadm unspreadl unspreadr unsplit uf s) =
  Eval
    unspreadm'
    unspreadl
    unspreadr
    unsplit'
    uf'
    s
 where
  unspreadm' = fmap (fmap f) . unspreadm
  unsplit' sl tl sm tr sr typ = fmap f <$> unsplit sl tl sm tr sr typ
  uf' l e r isLast = fmap f <$> uf l e r isLast

-- product evaluators
-- ------------------

{- | Combine two evaluators into a product evaluator.
 Each evaluation function returns the product of the two component evaluators' results.
-}
productEval
  :: Eval tr1 tr' slc1 slc' v1
  -> Eval tr2 tr' slc2 slc' v2
  -> Eval (tr1, tr2) tr' (slc1, slc2) slc' (v1, v2)
productEval (Eval unspreadm1 unspreadl1 unspreadr1 merge1 thaw1 slice1) (Eval unspreadm2 unspreadl2 unspreadr2 merge2 thaw2 slice2) =
  Eval unspreadm unspreadl unspreadr merge thaw slice
 where
  unspreadm ((l1, l2), (m1, m2), (r1, r2)) = do
    (a, va) <- unspreadm1 (l1, m1, r1)
    (b, vb) <- unspreadm2 (l2, m2, r2)
    pure ((a, b), (va, vb))
  unspreadl ((l1, l2), (c1, c2)) (t1, t2) = do
    a <- unspreadl1 (l1, c1) t1
    b <- unspreadl2 (l2, c2) t2
    pure (a, b)
  unspreadr ((c1, c2), (r1, r2)) (t1, t2) = do
    a <- unspreadr1 (c1, r1) t1
    b <- unspreadr2 (c2, r2) t2
    pure (a, b)
  merge sl (tl1, tl2) (sm1, sm2) (tr1, tr2) sr typ = do
    (a, va) <- merge1 sl1 tl1 sm1 tr1 sr1 typ
    (b, vb) <- merge2 sl2 tl2 sm2 tr2 sr2 typ
    pure ((a, b), (va, vb))
   where
    (sl1, sl2) = distStartStop sl
    (sr1, sr2) = distStartStop sr
  thaw l e r isLast = do
    (a, va) <- thaw1 l1 e r1 isLast
    (b, vb) <- thaw2 l2 e r2 isLast
    pure ((a, b), (va, vb))
   where
    (l1, l2) = distStartStop l
    (r1, r2) = distStartStop r
  slice s = (slice1 s, slice2 s)

-- restricting branching
-- ---------------------

-- | A flag that is used to restrict spread operations to right branching.
data RightBranchSpread
  = RBBranches
  | RBClear
  deriving (Eq, Ord, Show, Generic, NFData, Hashable)

{- | An evaluator that doesn't parse the input but restricts spread operations to right branching.
 Legal combinations will just return a singleton @()@ while illegal combinations return nothing.
 Combine this with any evaluator as a product (using 'productEval' or 'rightBranchSpread')
 to make the evaluator right-branching.
-}
evalRightBranchSpread :: Eval RightBranchSpread tr' () slc' ()
evalRightBranchSpread = Eval unspreadm unspreadl unspreadr merge thaw slice
 where
  unspreadm (_, RBBranches, _) = Nothing
  unspreadm (_, RBClear, _) = Just ((), ())
  unspreadl _ _ = [RBClear]
  unspreadr _ _ = [RBBranches]
  merge _ _ _ _ _ _ = [(RBClear, ())]
  thaw _ _ _ _ = [(RBClear, ())]
  slice _ = ()

-- | Restrict any evaluator to right-branching spreads.
rightBranchSpread
  :: Eval tr tr' slc slc' w -> Eval (RightBranchSpread, tr) tr' ((), slc) slc' w
rightBranchSpread = mapEvalScore snd . productEval evalRightBranchSpread

-- restricting derivation order
-- ----------------------------

{- | A flag for indicating whether a transition is the result of a split or not.
 This is used for restricting the order of splits and spreads.
-}
data Merged
  = Merged
  | NotMerged
  deriving (Eq, Ord, Show, Generic, NFData, Hashable)

{- | An evaluator that doesn't parse the input but restricts the order of operations
 to always have splits before spreads on the left and right transitions at a spread.
 Legal combinations will just return a singleton @()@ while illegal combinations return nothing.
 Combine this with any evaluator as a product (using 'productEval' or 'splitFirst')
 to make the evaluator order-restricted.
-}
evalSplitBeforeSpread :: (Eval Merged tr' () slc' ())
evalSplitBeforeSpread = Eval unspreadm unspreadl unspreadr merge thaw slice
 where
  unspreadm _ = Just ((), ())
  unspreadl (Merged, _) _ = []
  unspreadl (NotMerged, _) _ = [NotMerged]
  unspreadr (_, Merged) _ = []
  unspreadr (_, NotMerged) _ = [NotMerged]
  merge _ _ _ _ _ _ = [(Merged, ())]
  thaw _ _ _ _ = [(NotMerged, ())]
  slice _ = ()

-- | Restrict any evaluator to split-before-spread order.
splitFirst :: Eval tr tr' slc slc' w -> Eval (Merged, tr) tr' ((), slc) slc' w
splitFirst = mapEvalScore snd . productEval evalSplitBeforeSpread

-- left-most derivation outer operations
-- =====================================

{- $leftmostdoc

 More specifically, if there is only one open transition left, only two actions are possible,
 freezing or splitting that transition:

 > freeze only:   split only:
 > ...=[]——⋉      ==[]——⋉
 > ...=[]==⋉         \  /
 >                     []

 These options are encoded in 'LeftmostSingle'.

 If two or more transitions are still open, four actions are possible:

 > freeze left:         split left:          split right:         spread:
 > ...=[]——[]——[]—...   ...=[]——[]——[]—...   ...=[]——[]——[]—...   ...=[]——[]——[]—...
 > ...=[]==[]——[]—...        \  /                     \  /             \  /\  /
 >                            []                       []               []——[]

 These options are encoded in 'LeftmostDouble'.
 Note that the order of operations is restricted so that after a right split only
 only another right split or a spread are allowed.
 See [below](#monadicDeriv) for a way to construct leftmost derivations in a type-safe way,
 checking operation order and open transitions at compile time.

 Both single and double operations are combined in 'Leftmost'.
 All three operation containers are parameterized over the specific operations types for
 splits (@s@), spreads (@h@ for "horizontalization"), freezes (@f@).
-}

-- | Generative operations on a single transition (split or freeze).
data LeftmostSingle s f
  = LMSingleSplit !s
  | LMSingleFreeze !f
  deriving (Eq, Ord, Show, Generic, NFData, Functor, Foldable, Traversable)

instance (ToJSON s, ToJSON f) => ToJSON (LeftmostSingle s f) where
  toJSON =
    Aeson.genericToJSON $ variantDefaults ((<> "Only") . firstToLower . drop 8)
  toEncoding =
    Aeson.genericToEncoding $
      variantDefaults ((<> "Only") . firstToLower . drop 8)

-- | Generative operations on two transitions (split left, freeze left, split right, or spread)
data LeftmostDouble s f h
  = LMDoubleSplitLeft !s
  | LMDoubleFreezeLeft !f
  | LMDoubleSplitRight !s
  | LMDoubleSpread !h
  deriving (Eq, Ord, Show, Generic, NFData)

-- | Helper function for `LeftmostDouble`'s 'ToJSON' instance.
lmDoubleToJSONName "LMDoubleSpread" = "hori"
lmDoubleToJSONName str = firstToLower $ drop 8 str

instance (ToJSON s, ToJSON f, ToJSON h) => ToJSON (LeftmostDouble s f h) where
  toJSON = Aeson.genericToJSON $ variantDefaults lmDoubleToJSONName
  toEncoding =
    Aeson.genericToEncoding $ variantDefaults lmDoubleToJSONName

-- | A combined datatype for all leftmost-derivation operations.
data Leftmost s f h
  = LMSingle !(LeftmostSingle s f)
  | LMDouble !(LeftmostDouble s f h)
  deriving (Eq, Ord, Show, Generic, NFData)

instance (FromJSON s, FromJSON f, FromJSON h) => FromJSON (Leftmost s f h) where
  parseJSON = Aeson.withObject "Leftmost" $ \obj -> do
    typ <- obj .: "type"
    val <- obj .: "value"
    case typ of
      "freezeLeft" -> LMFreezeLeft <$> parseJSON val
      "freezeOnly" -> LMFreezeOnly <$> parseJSON val
      "splitLeft" -> LMSplitLeft <$> parseJSON val
      "splitRight" -> LMSplitRight <$> parseJSON val
      "splitOnly" -> LMSplitOnly <$> parseJSON val
      "hori" -> LMSpread <$> parseJSON val -- the JSON encoding uses "hori" instead of "spread"
      other -> unexpected other

instance (ToJSON s, ToJSON f, ToJSON h) => ToJSON (Leftmost s f h) where
  toJSON (LMSingle sg) = toJSON sg
  toJSON (LMDouble db) = toJSON db
  toEncoding (LMSingle sg) = toEncoding sg
  toEncoding (LMDouble db) = toEncoding db

pattern LMSplitLeft :: s -> Leftmost s f h
pattern LMSplitLeft s = LMDouble (LMDoubleSplitLeft s)

pattern LMFreezeLeft :: f -> Leftmost s f h
pattern LMFreezeLeft f = LMDouble (LMDoubleFreezeLeft f)

pattern LMSplitRight :: s -> Leftmost s f h
pattern LMSplitRight s = LMDouble (LMDoubleSplitRight s)

pattern LMSpread :: h -> Leftmost s f h
pattern LMSpread h = LMDouble (LMDoubleSpread h)

pattern LMSplitOnly :: s -> Leftmost s f h
pattern LMSplitOnly s = LMSingle (LMSingleSplit s)

pattern LMFreezeOnly :: f -> Leftmost s f h
pattern LMFreezeOnly f = LMSingle (LMSingleFreeze f)

{-# COMPLETE LMSplitLeft, LMFreezeLeft, LMSplitRight, LMSpread, LMSplitOnly, LMFreezeOnly #-}

-- representing full analyses
-- ==========================

{- | Encodes an analysis of a piece,
 consisting of a "top" (the starting point of the derivation,
 i.e., the smallest reduction in the analysis)
 and a derivation of the piece's surface from the top.

 Use this type's 'FromJSON' instance to load an analysis exported by the protovoice annotation tool.
-}
data Analysis s f h tr slc = Analysis
  { anaDerivation :: [Leftmost s f h]
  -- ^ The derivation steps.
  , anaTop :: Path tr slc
  -- ^ The starting configuration of the derivation.
  -- Starts with the first transition, 'Start' and 'Stop' are implied.
  }
  deriving (Eq, Ord, Show, Generic)

instance (FromJSON s, FromJSON f, FromJSON h, FromJSON tr, FromJSON slc) => FromJSON (Analysis s f h tr slc) where
  parseJSON = Aeson.withObject "Analysis" $ \v -> do
    deriv <- v .: "derivation"
    start <- v .: "start" >>= parseSlice
    case start of
      Start -> pure ()
      _ -> fail "Start slice is not ⋊."
    segments <- v .: "topSegments"
    top <- parseTop segments
    pure $ Analysis{anaDerivation = deriv, anaTop = top}
   where
    parseTop :: [Aeson.Value] -> Aeson.Parser (Path tr slc)
    parseTop segs = do
      segments <- mapM parseSegment segs
      mkPath segments
     where
      mkPath :: [(e, StartStop a)] -> Aeson.Parser (Path e a)
      mkPath [(t, Stop)] = pure $ PathEnd t
      mkPath ((t, Inner s) : rest) = Path t s <$> mkPath rest
      mkPath _ = fail "Invalid top path."
    parseSlice = Aeson.withObject "Slice" $ \v -> v .: "notes"
    parseTrans = Aeson.withObject "Transition" $ \v -> v .: "edges"
    parseSegment = Aeson.withObject "Segment" $ \v -> do
      trans <- v .: "trans" >>= parseTrans
      rslice <- v .: "rslice" >>= parseSlice
      pure (trans, rslice)

-- | Prints the steps and intermediate configurations of a derivation.
debugAnalysis
  :: forall tr slc s f h
   . (Show tr, Show slc, Show s, Show h)
  => (s -> tr -> Either String (tr, slc, tr))
  -> (f -> tr -> Either String tr)
  -> (h -> tr -> slc -> tr -> Either String (tr, slc, tr, slc, tr))
  -> Analysis s f h tr slc
  -> IO (Either String ())
debugAnalysis doSplit doFreeze doSpread (Analysis deriv top) =
  runExceptT $
    go Start top False deriv
 where
  go
    :: StartStop slc
    -> Path tr slc
    -> Bool
    -> [Leftmost s f h]
    -> ExceptT String IO ()
  go _sl _surface _ars [] = except $ Left "Derivation incomplete."
  go sl surface@(PathEnd trans) _ars (op : rest) = do
    lift $ putStrLn $ "\nCurrent surface: " <> show surface
    case op of
      LMSingle single -> do
        -- debugSingleStep (sl, trans, Stop) single
        case single of
          LMSingleFreeze freezeOp -> do
            lift $ putStrLn "freezing only (terminating)"
            _ <- except $ doFreeze freezeOp trans
            pure ()
          LMSingleSplit splitOp -> do
            lift $ putStrLn $ "splitting only: " <> show splitOp
            (ctl, cs, ctr) <- except $ doSplit splitOp trans
            go sl (Path ctl cs (PathEnd ctr)) False rest
      LMDouble _ -> except $ Left "Double operation on single transition."
  go sl surface@(Path tl sm (PathEnd tr)) ars (op : rest) = do
    lift $ putStrLn $ "\nCurrent surface: " <> show surface
    goDouble op rest ars (sl, tl, sm, tr, Stop) PathEnd
  go sl surface@(Path tl sm (Path tr sr pathRest)) ars (op : derivRest) = do
    lift $ putStrLn $ "\nCurrent surface: " <> show surface
    goDouble op derivRest ars (sl, tl, sm, tr, Inner sr) $
      \tr' -> Path tr' sr pathRest

  goDouble op rest ars (sl, tl, sm, tr, _sr) mkRest = case op of
    LMSingle _ ->
      except $ Left "Single operation with several transitions left."
    LMDouble double -> do
      -- observeDoubleStep (sl, tl, sm, tr, sr) ars double
      case double of
        LMDoubleFreezeLeft freezeOp -> do
          when ars $ except $ Left "FreezeLeft after SplitRight."
          lift $ putStrLn "freezing left"
          _ <- except $ doFreeze freezeOp tl
          go (Inner sm) (mkRest tr) False rest
        LMDoubleSplitLeft splitOp -> do
          when ars $ except $ Left "SplitLeft after SplitRight."
          lift $ putStrLn $ "splitting left: " <> show splitOp
          (ctl, cs, ctr) <- except $ doSplit splitOp tl
          go sl (Path ctl cs $ Path ctr sm $ mkRest tr) False rest
        LMDoubleSplitRight splitOp -> do
          lift $ putStrLn $ "splitting right: " <> show splitOp
          (ctl, cs, ctr) <- except $ doSplit splitOp tr
          go sl (Path tl sm $ Path ctl cs $ mkRest ctr) True rest
        LMDoubleSpread spreadOp -> do
          lift $ putStrLn $ "spreading: " <> show spreadOp
          (ctl, csl, ctm, csr, ctr) <- except $ doSpread spreadOp tl sm tr
          go sl (Path ctl csl $ Path ctm csr $ mkRest ctr) False rest

-- evaluators
-- ==========

{- | Create a leftmost evaluator from position-independent evaluation functions
 that just return spread, split, and freeze operations
 by wrapping those into the appropriate 'Leftmost' constructors.
-}
mkLeftmostEval
  :: UnspreadMiddle tr slc h
  -> UnspreadLeft tr slc
  -> UnspreadRight tr slc
  -> (StartStop slc -> tr -> slc -> tr -> StartStop slc -> [(tr, s)])
  -> (StartStop slc -> Maybe tr' -> StartStop slc -> [(tr, f)])
  -> (slc' -> slc)
  -> Eval tr tr' slc slc' (Leftmost s f h)
mkLeftmostEval unspreadm unspreadl unspreadr unsplit uf =
  Eval
    unspreadm'
    unspreadl
    unspreadr
    unsplit'
    uf'
 where
  smap f = fmap (second f)
  -- vm' :: UnspreadMiddle e a (Leftmost s f h)
  unspreadm' vert = smap LMSpread $ unspreadm vert
  unsplit' sl tl sm tr sr typ = smap splitop res
   where
    res = unsplit sl tl sm tr sr
    splitop = case typ of
      LeftOfTwo -> LMSplitLeft
      SingleOfOne -> LMSplitOnly
      RightOfTwo -> LMSplitRight
  uf' sl e sr isLast
    | isLast = smap LMFreezeOnly res
    | otherwise = smap LMFreezeLeft res
   where
    res = uf sl e sr

-- manually constructing derivations
-- =================================

{- $monadicdoc

 Use these functions to manually build a derivation,
 checking leftmost-correctness in the type.
 A good way to do this is to start a derivation using `buildDerivation` or `buildPartialDerivation`
 and follow up with a @do@ block that contains a sequence of `split`,
 `freeze`, `splitRight` and `spread` actions..

 > deriv :: [Leftmost () () ()] -- using unit for each operation type
 > deriv = buildDerivation $ do -- start with 1 transition
 >   split ()      -- (2 open transitions)
 >   splitRight () -- (3 open)
 >   spread ()     -- (4 open)
 >   freeze ()     -- (3 open)
 >   split ()      -- (4 open)
 >   freeze ()     -- (3 open)
 >   freeze ()     -- (2 open)
 >   freeze ()     -- (1 open)
 >   freeze ()     -- (0 open, end of derivation)

 The above example results in the following derivation graph:

 ![derivation of the above example](doc-images/monadic-deriv.svg)

 Since 'PartialDerivation' is an indexed monad
 (it's exact type changes between actions),
 using do-notation requires you to rebind its syntax to use indexed versions of '>>=' and '>>'
 using the @QualifiedDo@ extension.
 The easiest way to do use the generic operators from "Language.Haskell.DoNotation"
 by using this module as the @do@ qualifier:

 > import qualified Language.Haskell.DoNotation as Do
 >
 > deriv = buildDerivation $ Do.do -- requires -XQualifiedDo
 >   split ()
 >   ...
-}

{- | A wrapper around leftmost derivations
 that tracks information about the derivation state in the type.
 Number of open transitions: @openTrans@.
 Whether a right split has been performed at the current point: @afterRightSplit@.
-}
newtype PartialDerivation s f h (openTrans :: Nat) (afterRightSplit :: Bool) = PD {runPD :: [Leftmost s f h]}

{- | An "indexed" version of a writer monad, i.e. one where the monad type between two steps can change.
 This can be used for tracking the number of open transitions in a derivation on the type level
 while still providing an monadic interface for constructing a derivation.
-}
newtype IndexedWriter w i j a = IW {runIW :: MW.Writer w a}

instance MI.IxFunctor (IndexedWriter w) where
  imap f (IW w) = IW $ f <$> w

instance (Monoid w) => MI.IxPointed (IndexedWriter w) where
  ireturn a = IW $ return a

instance (Monoid w) => MI.IxApplicative (IndexedWriter w) where
  iap (IW wf) (IW wa) = IW (wf <*> wa)

instance (Monoid w) => MI.IxMonad (IndexedWriter w) where
  ibind f (IW wa) = IW $ (runIW . f) =<< wa

-- | 'MW.tell' for 'IndexedWriter'.
itell :: (Monoid w) => w -> IndexedWriter w i j ()
itell = IW . MW.tell

{- | A type-level wrapper for partial derivation info.
 Encodes the number of open transitions
 and whether the last operation was a right split.
-}
type DerivationInfo :: Nat -> Bool -> Type
data DerivationInfo a b

{- | The type of a monadic derivation action that modifies the derivation state
 (number of open transitions, after right split).
-}
type DerivationAction s f h n n' afterRight afterRight' =
  IndexedWriter
    [Leftmost s f h]
    (DerivationInfo n afterRight)
    (DerivationInfo n' afterRight')
    ()

{- | Turn a monadically constructed derivation into a proper left-most derivation.
 This function assumes the derivation to start with a single transition.
-}
buildDerivation
  -- :: (PartialDeriv s f h 1 False -> PartialDeriv s f h n snd)
  :: DerivationAction s f h 1 n 'False snd -> [Leftmost s f h]
buildDerivation build = MW.execWriter $ runIW build

{- | Turn a monadically constructed partial derivation into a left-most derivation.
 This function does not restrict the number of transitions in the starting configuration.
-}
buildPartialDerivation
  :: forall n n' snd s f h
   . DerivationAction s f h n n' 'False snd
  -> [Leftmost s f h]
buildPartialDerivation build = MW.execWriter $ runIW build

-- | Turn a split operation into a monadic (left or single) split action.
split
  :: forall n s f h
   . (KnownNat n, 1 <= n)
  => s
  -> DerivationAction s f h n (n + 1) 'False 'False
split s
  | natVal (Proxy :: Proxy n) == 1 = itell [LMSplitOnly s]
  | otherwise = itell [LMSplitLeft s]

-- | Turn a freeze operation into a monadic (left or single) freeze action.
freeze
  :: forall n s h f
   . (KnownNat n, 1 <= n)
  => f
  -> DerivationAction s f h n (n - 1) 'False 'False
freeze f
  | natVal (Proxy :: Proxy n) == 1 = itell [LMFreezeOnly f]
  | otherwise = itell [LMFreezeLeft f]

-- | Turn a split operation into a monadic right-split action.
splitRight :: (2 <= n) => s -> DerivationAction s f h n (n + 1) snd 'True
splitRight s = itell [LMSplitRight s]

-- | Turn a spread operation into a monadic spread action.
spread :: (2 <= n) => h -> DerivationAction s f h n (n + 1) snd 'False
spread h = itell [LMSpread h]

-- useful semirings
-- ================

{- | The derivations semiring.
 Similar to a free semiring, encodes sequences, alternatives, and neutral values directly.
 However, semiring equivalences are not idendified by default.
-}
data Derivations a
  = -- | a single operation
    Do !a
  | -- | combines alternative derivations
    Or !(Derivations a) !(Derivations a)
  | -- | combines sequential derivations
    Then !(Derivations a) !(Derivations a)
  | -- | the neutral element to 'Then'
    NoOp
  | -- | the neutral element to 'Or'
    Cannot
  deriving (Eq, Ord, Generic)

instance (NFData a) => NFData (Derivations a)

-- | A helper tag for pretty-printing derivations.
data DerivOp
  = OpNone
  | OpOr
  | OpThen
  deriving (Eq)

instance (Show a) => Show (Derivations a) where
  show = go (0 :: Int) OpNone
   where
    indent n = stimesMonoid n "  "
    go n _ (Do a) = indent n <> show a
    go n _ NoOp = indent n <> "NoOp"
    go n _ Cannot = indent n <> "Cannot"
    go n OpOr (Or a b) = go n OpOr a <> "\n" <> go n OpOr b
    go n _ (Or a b) =
      indent n <> "Or\n" <> go (n + 1) OpOr a <> "\n" <> go (n + 1) OpOr b
    go n OpThen (Then a b) = go n OpThen a <> "\n" <> go n OpThen b
    go n _ (Then a b) =
      indent n <> "Then\n" <> go (n + 1) OpThen a <> "\n" <> go (n + 1) OpThen b

instance R.Semiring (Derivations a) where
  zero = Cannot
  one = NoOp
  plus Cannot a = a
  plus a Cannot = a
  plus a b = Or a b
  times Cannot _ = Cannot
  times _ Cannot = Cannot
  times NoOp a = a
  times a NoOp = a
  times a b = Then a b

-- | Map the 'Derivations' semiring to another semiring.
mapDerivations :: (R.Semiring r) => (a -> r) -> Derivations a -> r
mapDerivations f (Do a) = f a
mapDerivations _ NoOp = R.one
mapDerivations _ Cannot = R.zero
mapDerivations f (Or a b) = mapDerivations f a R.+ mapDerivations f b
mapDerivations f (Then a b) = mapDerivations f a R.* mapDerivations f b

-- | Flatten the prefix-tree structure of 'Derivations' into a simple set of derivations.
flattenDerivations :: (Ord a) => Derivations a -> S.Set [a]
flattenDerivations = mapDerivations (\a -> S.singleton [a])

{- | Flatten the prefix-tree structure of 'Derivations'
 into a simple list of (potentially redundant) derivations.
-}
flattenDerivationsRed :: (Ord a) => Derivations a -> [[a]]
flattenDerivationsRed (Do a) = pure [a]
flattenDerivationsRed NoOp = pure []
flattenDerivationsRed Cannot = []
flattenDerivationsRed (Or a b) =
  flattenDerivationsRed a <> flattenDerivationsRed b
flattenDerivationsRed (Then a b) = do
  as <- flattenDerivationsRed a
  bs <- flattenDerivationsRed b
  pure (as <> bs)

-- | Obtain the first derivation from a 'Derivations' tree.
firstDerivation :: (Ord a) => Derivations a -> Maybe [a]
firstDerivation Cannot = Nothing
firstDerivation NoOp = Just []
firstDerivation (Do a) = Just [a]
firstDerivation (Or a _) = firstDerivation a
firstDerivation (Then a b) = do
  da <- firstDerivation a
  db <- firstDerivation b
  pure $ da <> db

-- utilities
-- =========

-- | The global trace level. Only trace messages >= this level are shown.
traceLevel :: Int
traceLevel = 0

-- | A helper for conditionally tracing a message.
traceIf :: Int -> [Char] -> Bool -> Bool
traceIf l msg value =
  if traceLevel >= l && value then trace msg value else value

-- toVariant :: ToJSON a => T.Text -> a -> Aeson.Value
-- toVariant typ val = Aeson.object ["type" .= typ, "value" .= val]

-- toVariantEnc :: (ToJSON a) => T.Text -> a -> Aeson.Encoding
-- toVariantEnc typ val = Aeson.pairs ("type" .= typ <> "value" .= val)

-- | Lowercase the first character in a string.
firstToLower :: String -> String
firstToLower "" = ""
firstToLower (h : rest) = toLower h : rest

-- | Aeson options for parsing "variant" types (generated in PureScript)
variantDefaults :: (String -> String) -> Aeson.Options
variantDefaults rename =
  Aeson.defaultOptions
    { Aeson.constructorTagModifier = rename
    , Aeson.sumEncoding = Aeson.TaggedObject "type" "value"
    }

-- | Convert special characters to TeX commands.
showTex :: (Show a) => a -> String
showTex x = concatMap escapeTex $ show x
 where
  escapeTex '♭' = "$\\flat$"
  escapeTex '♯' = "$\\sharp$"
  escapeTex '{' = "\\{"
  escapeTex '}' = "\\}"
  escapeTex '⋉' = "$\\ltimes$"
  escapeTex '⋊' = "$\\rtimes$"
  escapeTex c = [c]

-- | Convert special characters to TeX commands (using 'T.Text')
showTexT :: (Show a) => a -> T.Text
showTexT = T.pack . showTex
