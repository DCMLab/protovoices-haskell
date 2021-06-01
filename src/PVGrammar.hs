{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE StandaloneDeriving #-}
--{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveAnyClass #-}
module PVGrammar where

import           Common

import           Musicology.Pitch               ( Notation(..) )

import qualified Data.HashSet                  as S
-- import qualified Data.Set.Internal             as S
import qualified Data.List                     as L
import qualified Data.Map.Strict               as M
import qualified Internal.MultiSet             as MS
import           GHC.Generics                   ( Generic )
import           Control.DeepSeq                ( NFData )
import           Data.Hashable                  ( Hashable )
import qualified Data.HashMap.Strict           as HM
import           Data.Aeson                    ( ToJSON, toJSON )

-- element types
-- =============

-- slice type: sets of notes
----------------------------


-- | The content type of 'Slice's.
-- Contains a multiset of pitches, representing the notes in a slice.
newtype Notes n = Notes (MS.MultiSet n)
  deriving (Eq, Ord, Generic, NFData, Hashable)

instance (Notation n) => Show (Notes n) where
  show (Notes ns) =
    "{" <> L.intercalate "," (showNote <$> MS.toOccurList ns) <> "}"
   where
    showNote (p, n) = showNotation p <> mult
      where mult = if n /= 1 then "×" <> show n else ""

-- | Return the notes or start/stop symbols inside a slice.
-- This is useful to get all objects that an 'Edge' can connect to. 
innerNotes :: StartStop (Notes n) -> [StartStop n]
innerNotes (Inner (Notes n)) = Inner <$> MS.distinctElems n
innerNotes (:⋊)              = [(:⋊)]
innerNotes (:⋉)              = [(:⋉)]

-- transition type: sets of obligatory edges
--------------------------------------------

-- TODO: could this be improved to forbid start/stop symbols on the wrong side?
-- | A proto-voice edge between two nodes (i.e. notes or start/stop symbols).
type Edge n = (StartStop n, StartStop n)

-- | A proto-voice edge between two notes (excluding start/stop symbols).
type InnerEdge n = (n, n)

-- | The content type of 'Transition's.
-- Contains a multiset of normal (terminal) edges and a multiset of (non-terminal) passing edges.
-- The represented edges are those that are definitely used later on.
-- Edges that are not used are dropped before creating a child transition.
-- A transition that contains passing edges cannot be frozen.
data Edges n = Edges
               { edgesT  :: !(S.HashSet (Edge n))            -- ^ the terminal edges
               , edgesNT :: !(MS.MultiSet (InnerEdge n)) -- ^ the non-terminal edges
               }
  deriving (Eq, Ord, Generic, NFData, Hashable)

instance (ToJSON n) => ToJSON (Edges n)

instance (ToJSON a) => ToJSON (MS.MultiSet a) where
  toJSON ms = toJSON $ MS.toList ms


instance (Notation n) => Show (Edges n) where
  show (Edges ts nts) = "{" <> L.intercalate "," (tts <> tnts) <> "}"
   where
    tts  = showT <$> S.toList ts
    tnts = showNT <$> MS.toOccurList nts
    showT (p1, p2) = showNotation p1 <> "-" <> showNotation p2
    showNT ((p1, p2), n) =
      showNotation p1 <> ">" <> showNotation p2 <> "×" <> show n

-- operations
-- ==========

-- | Marks different types of ornaments in the derivation.
data Ornament = FullNeighbor
              | FullRepeat
              | LeftRepeatOfRight
              | RightRepeatOfLeft
              | RootNote
  deriving (Eq, Ord, Show, Generic, Hashable)

instance NFData Ornament
instance ToJSON Ornament

data Passing = PassingMid
             | PassingLeft
             | PassingRight
  deriving (Eq, Ord, Show, Generic, Hashable)

instance NFData Passing
instance ToJSON Passing

data LeftOrnament = SingleLeftNeighbor
                  | SingleLeftRepeat
  deriving (Eq, Ord, Show, Generic, Hashable)

instance NFData LeftOrnament
instance ToJSON LeftOrnament

data RightOrnament = SingleRightNeighbor
                   | SingleRightRepeat
  deriving (Eq, Ord, Show, Generic, Hashable)

instance NFData RightOrnament
instance ToJSON RightOrnament

isRepetitionOnLeft :: Ornament -> Bool
isRepetitionOnLeft FullRepeat        = True
isRepetitionOnLeft RightRepeatOfLeft = True
isRepetitionOnLeft _                 = False

isRepetitionOnRight :: Ornament -> Bool
isRepetitionOnRight FullRepeat        = True
isRepetitionOnRight LeftRepeatOfRight = True
isRepetitionOnRight _                 = False

-- | Encodes the decisions made in a split operation.
-- Contains a list of elaborations for every parent edge and note.
-- Each elaboration contains the child pitch, and the corresponding ornament.
-- For every produced edge, a decisions is made whether to keep it or not.
data Split n = SplitOp
  { splitTs :: !(M.Map (Edge n) [(n, Ornament)])
  , splitNTs :: !(M.Map (InnerEdge n) [(n, Passing)])
  , fromLeft :: !(M.Map n [(n, RightOrnament)])
  , fromRight :: !(M.Map n [(n, LeftOrnament)])
  , keepLeft :: !(S.HashSet (Edge n))
  , keepRight :: !(S.HashSet (Edge n))
  }
  deriving (Eq, Ord, Generic)

instance (NFData n) => NFData (Split n)

instance (Notation n) => Show (Split n) where
  show (SplitOp ts nts ls rs kl kr) =
    "ts:"
      <> showOps opTs
      <> ", nts:"
      <> showOps opNTs
      <> ", ls:"
      <> showOps opLs
      <> ", rs:"
      <> showOps opRs
      <> ", kl:"
      <> showOps keepLs
      <> ", kr:"
      <> showOps keepRs
   where
    showOps ops = "{" <> L.intercalate "," ops <> "}"
    showEdge (n1, n2) = showNotation n1 <> "-" <> showNotation n2
    showChild (p, o) = showNotation p <> ":" <> show o
    showChildren cs = "[" <> L.intercalate "," (showChild <$> cs) <> "]"

    showSplit (e, cs) = showEdge e <> "=>" <> showChildren cs
    showL (p, lchilds) = showNotation p <> "=>" <> showChildren lchilds
    showR (p, rchilds) = showChildren rchilds <> "<=" <> showNotation p

    opTs   = showSplit <$> M.toList ts
    opNTs  = showSplit <$> M.toList nts
    opLs   = showL <$> M.toList ls
    opRs   = showR <$> M.toList rs
    keepLs = showEdge <$> S.toList kl
    keepRs = showEdge <$> S.toList kr

instance (Ord n, Hashable n) => Semigroup (Split n) where
  (SplitOp ta nta la ra kla kra) <> (SplitOp tb ntb lb rb klb krb) = SplitOp
    (ta <+> tb)
    (nta <+> ntb)
    (la <+> lb)
    (ra <+> rb)
    (S.union kla klb)
    (S.union kra krb)
   where
    (<+>) :: (Ord k, Semigroup a) => M.Map k a -> M.Map k a -> M.Map k a
    (<+>) = M.unionWith (<>)

instance (Ord n, Hashable n) => Monoid (Split n) where
  mempty = SplitOp M.empty M.empty M.empty M.empty S.empty S.empty

-- | Represents a freeze operation.
-- Since this just ties all remaining edges
-- (which must all be repetitions)
-- no decisions have to be encoded.
data Freeze = FreezeOp
  deriving (Eq, Ord, Generic)

instance NFData Freeze

instance Show Freeze where
  show _ = "()"

-- | Encodes the distribution of a pitch in a horizontalization.
-- 
-- All instances of a pitch must be either moved completely to the left or the right (or both).
-- In addition, some instances may be repeated on the other side.
-- The difference is indicated by the field of the 'ToLeft' and 'ToRight' constructors.
-- For example, @ToLeft 3@ indicates that out of @n@ instances,
-- all @n@ are moved to the left and @n-3@ are replicated on the right.
data HoriDirection = ToLeft !Int  -- ^ all to the left, n fewer to the right
                   | ToRight !Int -- ^ all to the right, n fewer to the left
                   | ToBoth      -- ^ all to both
  deriving (Eq, Ord, Show, Generic)

instance NFData HoriDirection

-- | Represents a horzontalization operation.
-- Records for every pitch how it is distributed (see 'HoriDirection').
-- The resulting edges (repetitions and passing edges) are represented in a child transition.
data Hori n = HoriOp !(HM.HashMap n HoriDirection) !(Edges n)
  deriving (Eq, Ord, Generic)

instance NFData n => NFData (Hori n)

instance (Notation n) => Show (Hori n) where
  show (HoriOp dist m) = "{" <> L.intercalate "," dists <> "} => " <> show m
   where
    dists = showDist <$> HM.toList dist
    showDist (p, to) = showNotation p <> "=>" <> show to

-- | 'Leftmost' specialized to the split, freeze, and horizontalize operations of the grammar.
type PVLeftMost n = Leftmost (Split n) Freeze (Hori n)

-- | A tag that distinguishes four different types of operations:
--  terminal split, non-termal split, left ornament, and right ornament
data Elaboration a b c d = ET !a  -- ^ marks a terminal split
                           | EN !b -- ^ marks a non-terminal split
                           | ER !c  -- ^ marks a right ornament
                           | EL !d  -- ^ marks a left ornament
  deriving (Eq, Ord, Show, Generic, Hashable, NFData)


instance (ToJSON a, ToJSON b,ToJSON c,ToJSON d) => ToJSON (Elaboration a b c d)
