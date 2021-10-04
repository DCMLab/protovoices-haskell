{-# LANGUAGE ExplicitNamespaces #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE EmptyDataDecls #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveTraversable #-}
module Common where

import           Control.DeepSeq                ( NFData )
import           GHC.Generics                   ( Generic )

import qualified Control.Monad.Indexed         as MI
import qualified Control.Monad.Writer.Strict   as MW
import           Data.Aeson                     ( (.:)
                                                , FromJSON(..)
                                                , ToJSON(..)
                                                )
import qualified Data.Aeson                    as Aeson
import qualified Data.Aeson.Types              as Aeson
import           Data.Aeson.Types               ( unexpected )
import           Data.Bifunctor                 ( second )
import           Data.Hashable                  ( Hashable )
import           Data.Kind                      ( Type )
import           Data.Semigroup                 ( stimesMonoid )
import qualified Data.Semiring                 as R
import qualified Data.Set                      as S
import           Data.Typeable                  ( Proxy(Proxy) )
import           Debug.Trace                    ( trace )
import           GHC.TypeNats                   ( type (+)
                                                , type (-)
                                                , type (<=)
                                                , KnownNat
                                                , Nat
                                                , natVal
                                                )
import           GHC.Unicode                    ( toLower )
import           Musicology.Pitch               ( Notation(..) )
import qualified Text.ParserCombinators.ReadP  as ReadP

-- Path: alternating slices and transitions
-- ========================================

data Path a e = Path !a !e !(Path a e)
              | PathEnd !a
  deriving (Eq, Ord , Generic)

instance (Show a, Show e) => Show (Path a e) where
  show (Path a e rst) = show a <> "\n+-" <> show e <> "\n" <> show rst
  show (PathEnd a   ) = show a

pathLen :: Path a e -> Int
pathLen (Path _ _ rest) = pathLen rest + 1
pathLen (PathEnd _    ) = 1

pathHead :: Path a e -> a
pathHead (Path l _ _) = l
pathHead (PathEnd l ) = l

mapNodesWithIndex :: Int -> (Int -> a -> b) -> Path a e -> Path b e
mapNodesWithIndex i f (Path l m rest) =
  Path (f i l) m (mapNodesWithIndex (i + 1) f rest)
mapNodesWithIndex i f (PathEnd n) = PathEnd (f i n)

mapEdges :: (a -> e -> a -> b) -> Path a e -> [b]
mapEdges f (Path l m rest) = f l m r : mapEdges f rest where r = pathHead rest
mapEdges _ (PathEnd _    ) = []

-- StartStop
-- =========

-- | A container type that augements the type @a@
-- with symbols for beginning (@:⋊@) and end (@:⋉@).
-- Every other value is wrapped in an @Inner@ constructor.
data StartStop a = Start
                 | Inner !a
                 | Stop
  deriving (Ord, Eq, Generic, NFData, Hashable, Functor, Foldable, Traversable)

-- some instances for StartStop

instance Show a => Show (StartStop a) where
  show Start     = "⋊"
  show Stop      = "⋉"
  show (Inner a) = show a

instance (Notation a) => Notation (StartStop a) where
  showNotation Start     = "⋊"
  showNotation Stop      = "⋉"
  showNotation (Inner a) = showNotation a
  parseNotation = ReadP.pfail

instance FromJSON a => FromJSON (StartStop a) where
  parseJSON (Aeson.String "start") = pure Start
  parseJSON (Aeson.String "stop" ) = pure Stop
  parseJSON other                  = Inner <$> parseJSON other

-- some helper functions for StartStop

-- | From a list of @StartStop@s returns only the elements that are not @:⋊@ or @:⋉@,
-- unwrapped to their original type.
onlyInner :: [StartStop a] -> [a]
onlyInner []              = []
onlyInner (Inner a : rst) = a : onlyInner rst
onlyInner (_       : rst) = onlyInner rst

-- | Returns the content of an 'Inner', or 'Nothing'.
getInner :: StartStop a -> Maybe a
getInner (Inner a) = Just a
getInner _         = Nothing

getInnerE :: StartStop a -> Either String a
getInnerE (Inner a) = Right a
getInnerE Start     = Left "expected inner but found ⋊"
getInnerE Stop      = Left "expected inner but found ⋉"

isInner :: StartStop a -> Bool
isInner (Inner _) = True
isInner _         = False

isStart :: StartStop a -> Bool
isStart Start = True
isStart _     = False

isStop :: StartStop a -> Bool
isStop Stop = True
isStop _    = False

distStartStop :: StartStop (a, b) -> (StartStop a, StartStop b)
distStartStop Start          = (Start, Start)
distStartStop Stop           = (Stop, Stop)
distStartStop (Inner (a, b)) = (Inner a, Inner b)

-- evaluator interface
-- ===================

type IsLast = Bool

data SplitType = LeftOfTwo
               | LeftOnly
               | RightOfTwo

-- | An evaluator for verticalizations.
-- Returns the verticalization of a (middle) transition, if possible.
type VertMiddle e a v = (a, e, a) -> Maybe (a, v)

-- | An evaluator returning the possible left parent edges of a verticalization.
type VertLeft e a = (e, a) -> a -> [e]

-- | An evaluator returning the possible right parent edges of a verticalization.
type VertRight e a = (a, e) -> a -> [e]

-- | An evaluator for merges.
-- Returns possible merges of a given pair of transitions.
type Merge e a v
  = StartStop a -> e -> a -> e -> StartStop a -> SplitType -> [(e, v)]

-- | A combined evaluator for verticalizations, merges, and thaws.
-- Additionally, contains a function for mapping terminal slices to semiring values.
data Eval e e' a a' v = Eval
  { evalVertMiddle :: !(VertMiddle e a v)
  , evalVertLeft   :: !(VertLeft e a)
  , evalVertRight  :: !(VertRight e a)
  , evalMerge      :: !(Merge e a v)
  , evalThaw :: !(StartStop a -> Maybe e' -> StartStop a -> IsLast -> [(e, v)])
  , evalSlice      :: !(a' -> a)
  }

-- | Maps a function over all scores produced by the evaluator.
mapEvalScore :: (v -> w) -> Eval e e' a a' v -> Eval e e' a a' w
mapEvalScore f (Eval vm vl vr m t s) = Eval vm' vl vr m' t' s
 where
  vm' = fmap (fmap f) . vm
  m' sl l sm r sr typ = fmap f <$> m sl l sm r sr typ
  t' l e r isLast = fmap f <$> t l e r isLast

-- product evaluators
-- ------------------

productEval
  :: Eval e1 e' a1 a' v1
  -> Eval e2 e' a2 a' v2
  -> Eval (e1, e2) e' (a1, a2) a' (v1, v2)
productEval (Eval vertm1 vertl1 vertr1 merge1 thaw1 slice1) (Eval vertm2 vertl2 vertr2 merge2 thaw2 slice2)
  = Eval vertm vertl vertr merge thaw slice
 where
  vertm ((l1, l2), (m1, m2), (r1, r2)) = do
    (a, va) <- vertm1 (l1, m1, r1)
    (b, vb) <- vertm2 (l2, m2, r2)
    pure ((a, b), (va, vb))
  vertl ((l1, l2), (c1, c2)) (t1, t2) = do
    a <- vertl1 (l1, c1) t1
    b <- vertl2 (l2, c2) t2
    pure (a, b)
  vertr ((c1, c2), (r1, r2)) (t1, t2) = do
    a <- vertr1 (c1, r1) t1
    b <- vertr2 (c2, r2) t2
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

data RightBranchHori = RBBranches
                        | RBClear
  deriving (Eq, Ord, Show, Generic, NFData, Hashable)

evalRightBranchHori :: Eval RightBranchHori e' () a' ()
evalRightBranchHori = Eval vertm vertl vertr merge thaw slice
 where
  vertm (_, RBBranches, _) = Nothing
  vertm (_, RBClear   , _) = Just ((), ())
  vertl _ _ = [RBClear]
  vertr _ _ = [RBBranches]
  merge _ _ _ _ _ _ = [(RBClear, ())]
  thaw _ _ _ _ = [(RBClear, ())]
  slice _ = ()

rightBranchHori
  :: Eval e2 e' a2 a' w -> Eval (RightBranchHori, e2) e' ((), a2) a' w
rightBranchHori = mapEvalScore snd . productEval evalRightBranchHori

-- restricting derivation order
-- ----------------------------

data Merged = Merged
               | NotMerged
  deriving (Eq, Ord, Show, Generic, NFData, Hashable)

evalSplitBeforeHori :: (Eval Merged e' () a' ())
evalSplitBeforeHori = Eval vertm vertl vertr merge thaw slice
 where
  vertm _ = Just ((), ())
  vertl (Merged   , _) _ = []
  vertl (NotMerged, _) _ = [NotMerged]
  vertr (_, Merged   ) _ = []
  vertr (_, NotMerged) _ = [NotMerged]
  merge _ _ _ _ _ _ = [(Merged, ())]
  thaw _ _ _ _ = [(NotMerged, ())]
  slice _ = ()

splitFirst :: Eval e2 e' a2 a' w -> Eval (Merged, e2) e' ((), a2) a' w
splitFirst = mapEvalScore snd . productEval evalSplitBeforeHori

-- left-most derivation outer operations
-- =====================================

-- data Leftmost s f h = LMSplitLeft !s
--                     | LMFreezeLeft !f
--                     | LMSplitRight !s
--                     | LMHorizontalize !h
--                     | LMSplitOnly !s
--                     | LMFreezeOnly !f
--   deriving (Eq, Ord, Show, Generic, NFData)

data LeftmostSingle s f = LMSingleSplit !s
                        | LMSingleFreeze !f
  deriving (Eq, Ord, Show, Generic, NFData)

instance (ToJSON s, ToJSON f) => ToJSON (LeftmostSingle s f) where
  toJSON =
    Aeson.genericToJSON $ variantDefaults ((<> "Only") . firstToLower . drop 8)
  toEncoding = Aeson.genericToEncoding
    $ variantDefaults ((<> "Only") . firstToLower . drop 8)

data LeftmostDouble s f h = LMDoubleSplitLeft !s
                          | LMDoubleFreezeLeft !f
                          | LMDoubleSplitRight !s
                          | LMDoubleHori !h
  deriving (Eq, Ord, Show, Generic, NFData)

instance (ToJSON s, ToJSON f, ToJSON h) => ToJSON (LeftmostDouble s f h) where
  toJSON = Aeson.genericToJSON $ variantDefaults (firstToLower . drop 8)
  toEncoding =
    Aeson.genericToEncoding $ variantDefaults (firstToLower . drop 8)

data Leftmost s f h = LMSingle (LeftmostSingle s f)
                    | LMDouble (LeftmostDouble s f h)
  deriving (Eq, Ord, Show, Generic)

instance (FromJSON s, FromJSON f, FromJSON h) => FromJSON (Leftmost s f h) where
  parseJSON = Aeson.withObject "Leftmost" $ \obj -> do
    typ <- obj .: "type"
    val <- obj .: "value"
    case typ of
      "freezeLeft" -> LMFreezeLeft <$> parseJSON val
      "freezeOnly" -> LMFreezeOnly <$> parseJSON val
      "splitLeft"  -> LMSplitLeft <$> parseJSON val
      "splitRight" -> LMSplitRight <$> parseJSON val
      "splitOnly"  -> LMSplitOnly <$> parseJSON val
      "hori"       -> LMHorizontalize <$> parseJSON val
      other        -> unexpected other

instance (ToJSON s, ToJSON f, ToJSON h) => ToJSON (Leftmost s f h) where
  toJSON (LMSingle sg) = toJSON sg
  toJSON (LMDouble db) = toJSON db
  toEncoding (LMSingle sg) = toEncoding sg
  toEncoding (LMDouble db) = toEncoding db

instance (NFData s, NFData f, NFData h) => NFData (Leftmost s f h)

pattern LMSplitLeft :: s -> Leftmost s f h
pattern LMSplitLeft s = LMDouble (LMDoubleSplitLeft s)

pattern LMFreezeLeft :: f -> Leftmost s f h
pattern LMFreezeLeft f = LMDouble (LMDoubleFreezeLeft f)

pattern LMSplitRight :: s -> Leftmost s f h
pattern LMSplitRight s = LMDouble (LMDoubleSplitRight s)

pattern LMHorizontalize :: h -> Leftmost s f h
pattern LMHorizontalize h = LMDouble (LMDoubleHori h)

pattern LMSplitOnly :: s -> Leftmost s f h
pattern LMSplitOnly s = LMSingle (LMSingleSplit s)

pattern LMFreezeOnly :: f -> Leftmost s f h
pattern LMFreezeOnly f = LMSingle (LMSingleFreeze f)

{-# COMPLETE LMSplitLeft, LMFreezeLeft, LMSplitRight, LMHorizontalize, LMSplitOnly, LMFreezeOnly #-}

-- representing full analyses
-- --------------------------

data Analysis s f h e a = Analysis
  { anaDerivation :: [Leftmost s f h]
  , anaTop        :: Path e a
  -- TODO: better: Path e a, and check that first and last slice are start/stop?
  }
  deriving (Eq, Ord, Show, Generic)

instance (FromJSON s, FromJSON f, FromJSON h, FromJSON e, FromJSON a) => FromJSON (Analysis s f h e a) where
  parseJSON = Aeson.withObject "Analysis" $ \v -> do
    deriv <- v .: "derivation"
    start <- v .: "start" >>= parseSlice
    case start of
      Start -> pure ()
      _     -> fail "Start slice is not ⋊."
    segments <- v .: "topSegments"
    top      <- parseTop segments
    pure $ Analysis { anaDerivation = deriv, anaTop = top }
   where
    parseTop :: [Aeson.Value] -> Aeson.Parser (Path e a)
    parseTop segs = do
      segments <- mapM parseSegment segs
      mkPath segments
     where
      mkPath :: [(e, StartStop a)] -> Aeson.Parser (Path e a)
      mkPath [(t, Stop)          ] = pure $ PathEnd t
      mkPath ((t, Inner s) : rest) = Path t s <$> mkPath rest
      mkPath _                     = fail "Invalid top path."
    parseSlice   = Aeson.withObject "Slice" $ \v -> v .: "notes"
    parseTrans   = Aeson.withObject "Transition" $ \v -> v .: "edges"
    parseSegment = Aeson.withObject "Segment" $ \v -> do
      trans  <- v .: "trans" >>= parseTrans
      rslice <- v .: "rslice" >>= parseSlice
      pure (trans, rslice)

-- evaluators
-- ==========

mkLeftmostEval
  :: VertMiddle e a h
  -> VertLeft e a
  -> VertRight e a
  -> (StartStop a -> e -> a -> e -> StartStop a -> [(e, s)])
  -> (StartStop a -> Maybe e' -> StartStop a -> [(e, f)])
  -> (a' -> a)
  -> Eval e e' a a' (Leftmost s f h)
mkLeftmostEval vm vl vr m t = Eval vm' vl vr m' t'
 where
  smap f = fmap (second f)
  -- vm' :: VertMiddle e a (Leftmost s f h)
  vm' vert = smap LMHorizontalize $ vm vert
  m' sl tl sm tr sr typ = smap splitop res
   where
    res     = m sl tl sm tr sr
    splitop = case typ of
      LeftOfTwo  -> LMSplitLeft
      LeftOnly   -> LMSplitOnly
      RightOfTwo -> LMSplitRight
  t' sl e sr isLast | isLast    = smap LMFreezeOnly res
                    | otherwise = smap LMFreezeLeft res
    where res = t sl e sr

newtype PartialDeriv s f h (n :: Nat) (snd :: Bool) = PD { runPD :: [Leftmost s f h]}

newtype IndexedWriter w i j a = IW { runIW :: MW.Writer w a }

instance MI.IxFunctor (IndexedWriter w) where
  imap f (IW w) = IW $ f <$> w

instance (Monoid w) => MI.IxPointed (IndexedWriter w) where
  ireturn a = IW $ return a

instance (Monoid w) => MI.IxApplicative (IndexedWriter w) where
  iap (IW wf) (IW wa) = IW (wf <*> wa)

instance (Monoid w) => MI.IxMonad (IndexedWriter w) where
  ibind f (IW wa) = IW $ (runIW . f) =<< wa

itell :: Monoid w => w -> IndexedWriter w i j ()
itell = IW . MW.tell

type Prod :: Nat -> Bool -> Type
data Prod a b

type DerivAction s f h n n' snd snd'
  = IndexedWriter [Leftmost s f h] (Prod n snd) (Prod n' snd') ()

buildDerivation
  -- :: (PartialDeriv s f h 1 False -> PartialDeriv s f h n snd)
  :: DerivAction s f h 1 n 'False snd -> [Leftmost s f h]
buildDerivation build = MW.execWriter $ runIW build

buildPartialDerivation
  :: forall n n' snd s f h
   . DerivAction s f h n n' 'False snd
  -> [Leftmost s f h]
buildPartialDerivation build = MW.execWriter $ runIW build

split
  :: forall n s f h
   . (KnownNat n, 1 <= n)
  => s
  -> DerivAction s f h n (n+1) 'False 'False
split s | natVal (Proxy :: Proxy n) == 1 = itell [LMSplitOnly s]
        | otherwise                      = itell [LMSplitLeft s]

freeze
  :: forall n s h f
   . (KnownNat n, 1 <= n)
  => f
  -> DerivAction s f h n (n-1) 'False 'False
freeze f | natVal (Proxy :: Proxy n) == 1 = itell [LMFreezeOnly f]
         | otherwise                      = itell [LMFreezeLeft f]

splitRight :: (2 <= n) => s -> DerivAction s f h n (n+1) snd 'True
splitRight s = itell [LMSplitRight s]

hori :: (2 <= n) => h -> DerivAction s f h n (n+1) snd 'False
hori h = itell [LMHorizontalize h]

-- useful semirings
-- ================

data Derivations a = Do !a
                   | Or !(Derivations a) !(Derivations a)
                   | Then !(Derivations a) !(Derivations a)
                   | NoOp
                   | Cannot
  deriving (Eq, Ord, Generic)

instance NFData a => NFData (Derivations a)

data DerivOp = OpNone
             | OpOr
             | OpThen
  deriving Eq

instance Show a => Show (Derivations a) where
  show = go (0 :: Int) OpNone
   where
    indent n = stimesMonoid n "  "
    go n _    (Do a)   = indent n <> show a
    go n _    NoOp     = indent n <> "NoOp"
    go n _    Cannot   = indent n <> "Cannot"
    go n OpOr (Or a b) = go n OpOr a <> "\n" <> go n OpOr b
    go n _ (Or a b) =
      indent n <> "Or\n" <> go (n + 1) OpOr a <> "\n" <> go (n + 1) OpOr b
    go n OpThen (Then a b) = go n OpThen a <> "\n" <> go n OpThen b
    go n _ (Then a b) =
      indent n <> "Then\n" <> go (n + 1) OpThen a <> "\n" <> go (n + 1) OpThen b


instance R.Semiring (Derivations a) where
  zero = Cannot
  one  = NoOp
  plus Cannot a      = a
  plus a      Cannot = a
  plus a      b      = Or a b
  times Cannot _      = Cannot
  times _      Cannot = Cannot
  times NoOp   a      = a
  times a      NoOp   = a
  times a      b      = Then a b

mapDerivations :: (R.Semiring r) => (a -> r) -> Derivations a -> r
mapDerivations f (Do a)     = f a
mapDerivations _ NoOp       = R.one
mapDerivations _ Cannot     = R.zero
mapDerivations f (Or   a b) = mapDerivations f a R.+ mapDerivations f b
mapDerivations f (Then a b) = mapDerivations f a R.* mapDerivations f b

flattenDerivations :: Ord a => Derivations a -> S.Set [a]
flattenDerivations = mapDerivations (\a -> S.singleton [a])

flattenDerivationsRed :: Ord a => Derivations a -> [[a]]
flattenDerivationsRed (Do a) = pure [a]
flattenDerivationsRed NoOp   = pure []
flattenDerivationsRed Cannot = []
flattenDerivationsRed (Or a b) =
  flattenDerivationsRed a <> flattenDerivationsRed b
flattenDerivationsRed (Then a b) = do
  as <- flattenDerivationsRed a
  bs <- flattenDerivationsRed b
  pure (as <> bs)

firstDerivation :: Ord a => Derivations a -> Maybe [a]
firstDerivation Cannot     = Nothing
firstDerivation NoOp       = Just []
firstDerivation (Do a    ) = Just [a]
firstDerivation (Or   a _) = firstDerivation a
firstDerivation (Then a b) = do
  da <- firstDerivation a
  db <- firstDerivation b
  pure $ da <> db

-- utilities
-- =========

traceLevel :: Int
traceLevel = 0

traceIf :: Int -> [Char] -> Bool -> Bool
traceIf l msg value =
  if traceLevel >= l && value then trace msg value else value

-- toVariant :: ToJSON a => T.Text -> a -> Aeson.Value
-- toVariant typ val = Aeson.object ["type" .= typ, "value" .= val]

-- toVariantEnc :: (ToJSON a) => T.Text -> a -> Aeson.Encoding
-- toVariantEnc typ val = Aeson.pairs ("type" .= typ <> "value" .= val)

firstToLower :: String -> String
firstToLower ""         = ""
firstToLower (h : rest) = toLower h : rest

variantDefaults :: (String -> String) -> Aeson.Options
variantDefaults rename = Aeson.defaultOptions
  { Aeson.constructorTagModifier = rename
  , Aeson.sumEncoding            = Aeson.TaggedObject "type" "value"
  }
