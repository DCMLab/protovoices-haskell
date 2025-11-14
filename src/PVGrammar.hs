{-# LANGUAGE ApplicativeDo #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

{- | This module contains common datatypes and functions specific to the protovoice grammar.
 In a protovoice derivations, slices are multisets of notes
 while transitions contain connections between these notes.

 Code that is specific to parsing can be found in "PVGrammar.Parse",
 while generative code is located in "PVGrammar.Generate".
-}
module PVGrammar
  ( -- * Inner Structure Types

    -- ** Note: a pitch with an ID.
    Note (..)

    -- ** Slices: Notes
  , Notes (..)
  , innerNotes

    -- ** Transitions: Sets of Obligatory Edges

    -- | Transitions contain two kinds of edges, regular edges and passing edges.
  , Edges (..)
  , topEdges
  , Edge
  , InnerEdge

    -- * Generative Operations

    -- ** Freeze
  , Freeze (..)

    -- ** Split
  , Split (..)
  , DoubleOrnament (..)
  , isRepetitionOnLeft
  , isRepetitionOnRight
  , PassingOrnament (..)
  , LeftOrnament (..)
  , RightOrnament (..)

    -- ** Spread
  , Spread (..)
  -- , SpreadDirection (..)
  , SpreadChildren (..)
  , leftSpreadChild
  , rightSpreadChild

    -- * Derivations
  , PVLeftmost
  , PVAnalysis
  , analysisTraversePitch
  , analysisMapPitch

    -- * Loading Files
  , loadAnalysis
  , loadAnalysis'
  , slicesFromFile
  , slicesToPath
  , loadSurface
  , loadSurface'
  ) where

import Common

import Musicology.Pitch
  ( Interval
  , Notation (..)
  , Pitch
  , SInterval
  , SPC
  , SPitch
  , pc
  )

import Control.DeepSeq (NFData)
import Control.Monad.Identity (runIdentity)
import Data.Aeson (FromJSON (..), ToJSON (..), (.:), (.=))
import Data.Aeson qualified as Aeson
import Data.Aeson.Types qualified as Aeson
import Data.Foldable (toList)
import Data.HashMap.Strict qualified as HM
import Data.HashSet qualified as S
import Data.Hashable (Hashable)
import Data.List qualified as L
import Data.Map.Strict qualified as M
import Data.Maybe (fromMaybe, mapMaybe)
import Data.Text.Lazy.IO qualified as TL
import Data.Traversable (for)
import GHC.Generics (Generic)
import Internal.MultiSet qualified as MS
import Musicology.Core qualified as Music
import Musicology.Core.Slicing qualified as Music
import Musicology.MusicXML qualified as MusicXML

-- * Inner Structure Types

-- ** Note type: pitch + ID

-- | A note with a pitch and an ID.
data Note n = Note {notePitch :: n, noteId :: String}
  deriving (Eq, Ord, Functor, Foldable, Traversable, Generic)
  deriving anyclass (NFData, Hashable)

instance (Notation n) => Show (Note n) where
  show (Note p i) = showNotation p <> "." <> i

instance (Notation n) => FromJSON (Note n) where
  parseJSON = Aeson.withObject "Note" $ \v -> do
    pitch <- v .: "pitch"
    i <- v .: "id"
    case readNotation pitch of
      Just p -> pure $ Note p i
      Nothing -> fail $ "Could not parse pitch " <> pitch

instance (Notation n) => ToJSON (Note n) where
  toJSON (Note p i) = Aeson.object ["pitch" .= showNotation p, "id" .= i]
  toEncoding (Note p i) = Aeson.pairs ("pitch" .= showNotation p <> "id" .= i)

-- ** Slice Type: Sets of Notes

-- Slices contain a set of notes.

{- | The content type of slices in the protovoice model.
  Contains a set of notes (pitch x id), representing the notes in a slice.
-}
newtype Notes n = Notes (S.HashSet (Note n))
  deriving (Eq, Ord, Generic)
  deriving anyclass (NFData, Hashable, ToJSON)

instance (Notation n) => Show (Notes n) where
  show (Notes ns) =
    "{" <> L.intercalate "," ((showNotation . notePitch) <$> S.toList ns) <> "}"

instance (Notation n, Eq n, Hashable n) => FromJSON (Notes n) where
  parseJSON = Aeson.withArray "List of Notes" $ \notes -> do
    pitches <- mapM parseJSON notes
    pure $ Notes $ S.fromList $ toList pitches

{- | Return the notes or start/stop symbols inside a slice.
 This is useful to get all objects that an 'Edge' can connect to.
-}
innerNotes :: StartStop (Notes n) -> [StartStop (Note n)]
innerNotes (Inner (Notes n)) = Inner <$> S.toList n
innerNotes Start = [Start]
innerNotes Stop = [Stop]

-- TODO: could this be improved to forbid start/stop symbols on the wrong side?

-- | A proto-voice edge between two nodes (i.e. notes or start/stop symbols).
type Edge n = (StartStop (Note n), StartStop (Note n))

-- | A proto-voice edge between two notes (excluding start/stop symbols).
type InnerEdge n = (Note n, Note n)

{- | The content type of transitions in the protovoice model.
 Contains a multiset of regular edges and a multiset of passing edges.
 The represented edges are those that are definitely used later on.
 Edges that are not used are dropped before creating a child transition.
 A transition that contains passing edges cannot be frozen.
-}
data Edges n = Edges
  { edgesReg :: !(S.HashSet (Edge n))
  -- ^ regular edges
  , edgesPass :: !(MS.MultiSet (InnerEdge n))
  -- ^ passing edges
  }
  deriving (Eq, Ord, Generic, NFData, Hashable)

instance (Hashable n, Eq n) => Semigroup (Edges n) where
  (Edges aT aPass) <> (Edges bT bPass) = Edges (aT <> bT) (aPass <> bPass)

instance (Hashable n, Eq n) => Monoid (Edges n) where
  mempty = Edges mempty MS.empty

instance (Notation n) => Show (Edges n) where
  show (Edges reg pass) = "{" <> L.intercalate "," (tReg <> tPass) <> "}"
   where
    tReg = showReg <$> S.toList reg
    tPass = showPass <$> MS.toOccurList pass
    showReg (p1, p2) = show p1 <> "-" <> show p2
    showPass ((p1, p2), n) =
      show p1 <> ">" <> show p2 <> "×" <> show n

instance (Eq n, Hashable n, Notation n) => FromJSON (Edges n) where
  parseJSON = Aeson.withObject "Edges" $ \v -> do
    regular <- v .: "regular" >>= mapM parseEdge
    passing <- v .: "passing" >>= mapM parseInnerEdge
    pure $
      Edges
        (S.fromList (regular :: [Edge n]))
        (MS.fromList (passing :: [InnerEdge n]))

instance (Notation n) => ToJSON (Edges n) where
  toJSON (Edges reg pass) =
    Aeson.object
      [ "regular" .= fmap edgeToJSON (S.toList reg)
      , "passing" .= fmap edgeToJSON (MS.toList pass)
      ]
  toEncoding (Edges reg pass) =
    Aeson.pairs $
      "regular" .= fmap edgeToJSON (S.toList reg)
        <> "passing" .= fmap edgeToJSON (MS.toList pass)

-- | The starting transition of a derivation (@⋊——⋉@).
topEdges :: (Hashable n) => Edges n
topEdges = Edges (S.singleton (Start, Stop)) MS.empty

-- * Derivation Operations

-- | Two-sided ornament types (two parents).
data DoubleOrnament
  = -- | a full neighbor note
    FullNeighbor
  | -- | a repetition of both parents (which have the same pitch)
    FullRepeat
  | -- | a repetition of the right parent
    LeftRepeatOfRight
  | -- | a repetitions of the left parent
    RightRepeatOfLeft
  | -- | a note inserted at the top of the piece (between ⋊ and ⋉)
    RootNote
  deriving (Eq, Ord, Show, Generic, ToJSON, FromJSON, NFData)

-- | Types of passing notes (two parents).
data PassingOrnament
  = -- | a connecting passing note (step to both parents)
    PassingMid
  | -- | a step from the left parent
    PassingLeft
  | -- | a step from the right parent
    PassingRight
  deriving (Eq, Ord, Show, Generic, ToJSON, FromJSON, NFData)

{- | Types of single-sided ornaments left of the parent (@child-parent@)

 > [ ] [p]
 >     /
 >   [c]
-}
data LeftOrnament
  = -- | an incomplete left neighbor
    LeftNeighbor
  | -- | an incomplete left repetition
    LeftRepeat
  deriving (Eq, Ord, Show, Generic, ToJSON, FromJSON, NFData)

{- | Types of single-sided ornaments right of the parent (@parent--child@).

 > [p] [ ]
 >   \
 >   [c]
-}
data RightOrnament
  = -- | an incomplete right neighbor
    RightNeighbor
  | -- | an incomplete right repetition
    RightRepeat
  deriving (Eq, Ord, Show, Generic, ToJSON, FromJSON, NFData)

-- | Returns 'True' if the child repeats the left parent
isRepetitionOnLeft :: DoubleOrnament -> Bool
isRepetitionOnLeft FullRepeat = True
isRepetitionOnLeft RightRepeatOfLeft = True
isRepetitionOnLeft _ = False

-- | Returns 'True' if the child repeats the right parent
isRepetitionOnRight :: DoubleOrnament -> Bool
isRepetitionOnRight FullRepeat = True
isRepetitionOnRight LeftRepeatOfRight = True
isRepetitionOnRight _ = False

{- | Encodes the decisions made in a split operation.
 Contains a list of elaborations for every parent edge and note.
 Each elaboration contains the child pitch, and the corresponding ornament.
 For every produced edge, a decisions is made whether to keep it or not.
-}
data Split n = SplitOp
  { splitReg :: !(M.Map (Edge n) [(Note n, DoubleOrnament)])
  -- ^ Maps every regular edge to a list of ornamentations.
  , splitPass :: !(M.Map (InnerEdge n) [(Note n, PassingOrnament)])
  -- ^ Maps every passing edge to a passing tone.
  -- Since every passing edge is elaborated exactly once
  -- but there can be several instances of the same edge in a transition,
  -- the "same" edge can be elaborated with several passing notes,
  -- one for each instance of the edge.
  , fromLeft :: !(M.Map (Note n) [(Note n, RightOrnament)])
  -- ^ Maps notes from the left parent slice to lists of ornamentations.
  , fromRight :: !(M.Map (Note n) [(Note n, LeftOrnament)])
  -- ^ Maps notes from the right parent slice to lists of ornamentations.
  , keepLeft :: !(S.HashSet (Edge n))
  -- ^ The set of regular edges to keep in the left child transition.
  , keepRight :: !(S.HashSet (Edge n))
  -- ^ The set of regular edges to keep in the right child transition.
  , passLeft :: !(MS.MultiSet (InnerEdge n))
  -- ^ Contains the new passing edges introduced in the left child transition
  -- (excluding those passed down from the parent transition).
  , passRight :: !(MS.MultiSet (InnerEdge n))
  -- ^ Contains the new passing edges introduced in the right child transition
  -- (excluding those passed down from the parent transition).
  }
  deriving (Eq, Ord, Generic, NFData)

instance (Notation n) => Show (Split n) where
  show (SplitOp reg pass ls rs kl kr pl pr) =
    "regular:"
      <> showOps opReg
      <> ", passing:"
      <> showOps opPass
      <> ", ls:"
      <> showOps opLs
      <> ", rs:"
      <> showOps opRs
      <> ", kl:"
      <> showOps keepLs
      <> ", kr:"
      <> showOps keepRs
      <> ", pl:"
      <> showOps passLs
      <> ", pr:"
      <> showOps passRs
   where
    showOps ops = "{" <> L.intercalate "," ops <> "}"
    showEdge (n1, n2) = show n1 <> "-" <> show n2
    showChild (p, o) = show p <> ":" <> show o
    showChildren cs = "[" <> L.intercalate "," (showChild <$> cs) <> "]"

    showSplit (e, cs) = showEdge e <> "=>" <> showChildren cs
    showL (p, lchilds) = show p <> "=>" <> showChildren lchilds
    showR (p, rchilds) = showChildren rchilds <> "<=" <> show p

    opReg = showSplit <$> M.toList reg
    opPass = showSplit <$> M.toList pass
    opLs = showL <$> M.toList ls
    opRs = showR <$> M.toList rs
    keepLs = showEdge <$> S.toList kl
    keepRs = showEdge <$> S.toList kr
    passLs = showEdge <$> MS.toList pl
    passRs = showEdge <$> MS.toList pr

instance (Ord n, Hashable n) => Semigroup (Split n) where
  (SplitOp rega passa la ra kla kra pla pra) <> (SplitOp regb passb lb rb klb krb plb prb) =
    SplitOp
      (rega <+> regb)
      (passa <+> passb)
      (la <+> lb)
      (ra <+> rb)
      (S.union kla klb)
      (S.union kra krb)
      (MS.union pla plb)
      (MS.union pra prb)
   where
    (<+>) :: (Ord k, Semigroup a) => M.Map k a -> M.Map k a -> M.Map k a
    (<+>) = M.unionWith (<>)

instance (Ord n, Hashable n) => Monoid (Split n) where
  mempty =
    SplitOp M.empty M.empty M.empty M.empty S.empty S.empty MS.empty MS.empty

instance (Notation n, Ord n, Hashable n) => FromJSON (Split n) where
  parseJSON = Aeson.withObject "Split" $ \v -> do
    regular <- v .: "regular" >>= mapM (parseElaboration parseEdge)
    passing <- v .: "passing" >>= mapM (parseElaboration parseInnerEdge)
    fromL <- v .: "fromLeft" >>= mapM (parseElaboration parseJSON)
    fromR <- v .: "fromRight" >>= mapM (parseElaboration parseJSON)
    keepL <- v .: "keepLeft" >>= mapM parseEdge
    keepR <- v .: "keepRight" >>= mapM parseEdge
    passL <- v .: "passLeft" >>= mapM parseInnerEdge
    passR <- v .: "passRight" >>= mapM parseInnerEdge
    pure $
      SplitOp
        (M.fromList regular)
        (M.fromList passing)
        (M.fromList fromL)
        (M.fromList fromR)
        (S.fromList keepL)
        (S.fromList keepR)
        (MS.fromList (passL :: [InnerEdge n]))
        (MS.fromList (passR :: [InnerEdge n]))
   where
    parseElaboration
      :: (Notation n, FromJSON o)
      => (Aeson.Value -> Aeson.Parser p)
      -> Aeson.Value
      -> Aeson.Parser (p, [(Note n, o)])
    parseElaboration parseParent = Aeson.withObject "Elaboration" $ \reg -> do
      parent <- reg .: "parent" >>= parseParent
      children <- reg .: "children" >>= mapM parseChild
      pure (parent, children)
    parseChild
      :: (Notation n, FromJSON o) => Aeson.Value -> Aeson.Parser (Note n, o)
    parseChild = Aeson.withObject "Child" $ \cld -> do
      child <- cld .: "child" >>= parseJSON
      orn <- cld .: "orn"
      pure (child, orn)

instance (Notation n) => ToJSON (Split n) where
  toJSON (SplitOp reg pass fromL fromR keepL keepR passL passR) =
    Aeson.object
      [ "regular" .= fmap (elaboToJSON edgeToJSON) (M.toList reg)
      , "passing" .= fmap (elaboToJSON edgeToJSON) (M.toList pass)
      , "fromLeft" .= fmap (elaboToJSON toJSON) (M.toList fromL)
      , "fromRight" .= fmap (elaboToJSON toJSON) (M.toList fromR)
      , "keepLeft" .= fmap edgeToJSON (S.toList keepL)
      , "keepRight" .= fmap edgeToJSON (S.toList keepR)
      , "passLeft" .= fmap edgeToJSON (MS.toList passL)
      , "passRight" .= fmap edgeToJSON (MS.toList passR)
      ]
   where
    elaboToJSON :: (ToJSON o) => (p -> Aeson.Value) -> (p, [(Note n, o)]) -> Aeson.Value
    elaboToJSON fParent (parent, children) =
      Aeson.object
        [ "parent" .= fParent parent
        , "children" .= fmap childToJSON children
        ]
    childToJSON (n, o) = Aeson.object ["child" .= n, "orn" .= o]

{- | Represents a freeze operation.
 Since this just ties all remaining edges
 (which must all be repetitions)
 no decisions have to be encoded.
-}
newtype Freeze n = FreezeOp {freezeTies :: S.HashSet (Edge n)}
  deriving (Eq, Ord, Generic)
  deriving anyclass (NFData)

instance (Notation n) => Show (Freeze n) where
  show (FreezeOp ties) = show ties

instance (Notation n, Hashable n) => FromJSON (Freeze n) where
  parseJSON = Aeson.withObject "Freeze" $ \obj -> do
    ties <- obj .: "ties" >>= mapM parseEdge
    pure $ FreezeOp (S.fromList ties)

instance (Notation n) => ToJSON (Freeze n) where
  toJSON (FreezeOp ties) = Aeson.object ["ties" .= fmap edgeToJSON (S.toList ties)] -- TODO: add empty prevTime?

-- {- | Encodes the distribution of a pitch in a spread.

--  All instances of a pitch must be either moved completely to the left or the right (or both).
--  In addition, some instances may be repeated on the other side.
--  The difference is indicated by the field of the 'ToLeft' and 'ToRight' constructors.
--  For example, @ToLeft 3@ indicates that out of @n@ instances,
--  all @n@ are moved to the left and @n-3@ are replicated on the right.
-- -}
-- data SpreadDirection
--   = -- | all to the left, n fewer to the right
--     ToLeft !Int
--   | -- | all to the right, n fewer to the left
--     ToRight !Int
--   | -- | all to both
--     ToBoth
--   deriving (Eq, Ord, Show, Generic, NFData)

-- instance Semigroup SpreadDirection where
--   ToLeft l1 <> ToLeft l2 = ToLeft (l1 + l2)
--   ToRight l1 <> ToRight l2 = ToLeft (l1 + l2)
--   ToLeft l <> ToRight r
--     | l == r = ToBoth
--     | l < r = ToRight (r - l)
--     | otherwise = ToLeft (l - r)
--   ToBoth <> other = other
--   a <> b = b <> a

-- instance Monoid SpreadDirection where
--   mempty = ToBoth

{- | Represents the children of a note that is spread out.

A note can be distributed to either or both sub-slice.
-}
data SpreadChildren n
  = -- | a single child in the left slice
    SpreadLeftChild !(Note n)
  | -- | a single child in the right slice
    SpreadRightChild !(Note n)
  | -- | two children, on in each slice
    SpreadBothChildren !(Note n) !(Note n)
  deriving (Eq, Ord, Functor, Foldable, Traversable, Generic, NFData, Hashable)

instance (Notation n) => Show (SpreadChildren n) where
  show (SpreadLeftChild n) = show n <> "┘"
  show (SpreadRightChild n) = "└" <> show n
  show (SpreadBothChildren nl nr) = show nl <> "┴" <> show nr

instance (Notation n) => FromJSON (SpreadChildren n) where
  parseJSON = Aeson.withObject "SpreadChild" $ \cld -> do
    typ <- cld .: "type"
    case typ of
      "leftChild" -> fmap SpreadLeftChild $ cld .: "value" >>= parseJSON -- pure $ ToLeft 1
      "rightChild" -> fmap SpreadRightChild $ cld .: "value" >>= parseJSON -- pure $ ToRight 1
      "bothChildren" -> cld .: "value" >>= parseBoth -- pure ToBoth
      _ -> Aeson.unexpected typ
   where
    parseBoth = Aeson.withObject "SpreadBothChildren" $ \bth -> do
      left <- bth .: "left" >>= parseJSON
      right <- bth .: "right" >>= parseJSON
      pure $ SpreadBothChildren left right

instance (Notation n) => ToJSON (SpreadChildren n) where
  toJSON (SpreadLeftChild n) = Aeson.object ["type" .= ("leftChild" :: String), "value" .= n]
  toJSON (SpreadRightChild n) = Aeson.object ["type" .= ("rightChild" :: String), "value" .= n]
  toJSON (SpreadBothChildren nl nr) =
    Aeson.object
      [ "type" .= ("bothChildren" :: String)
      , "value" .= Aeson.object ["left" .= nl, "right" .= nr]
      ]

-- | Returns the left child of a spread note, if it exists
leftSpreadChild :: SpreadChildren n -> Maybe (Note n)
leftSpreadChild = \case
  (SpreadLeftChild n) -> Just n
  (SpreadBothChildren n _) -> Just n
  _ -> Nothing

-- | Returns the right child of a spread note, if it exists
rightSpreadChild :: SpreadChildren n -> Maybe (Note n)
rightSpreadChild = \case
  (SpreadRightChild n) -> Just n
  (SpreadBothChildren _ n) -> Just n
  _ -> Nothing

{- | Represents a spread operation.
 Records for every pitch how it is distributed (see 'SpreadDirection').
 The resulting edges (repetitions and passing edges) are represented in a child transition.
-}
data Spread n = SpreadOp !(HM.HashMap (Note n) (SpreadChildren n)) !(Edges n)
  deriving (Eq, Ord, Generic, NFData, Hashable)

instance (Notation n) => Show (Spread n) where
  show (SpreadOp dist m) = "{" <> L.intercalate "," dists <> "} => " <> show m
   where
    dists = showDist <$> HM.toList dist
    showDist (n, to) = show n <> "=>" <> show to

instance (Notation n, Eq n, Hashable n) => FromJSON (Spread n) where
  parseJSON = Aeson.withObject "Spread" $ \v -> do
    dists <- v .: "children" >>= mapM parseDist
    edges <- v .: "midEdges"
    pure $ SpreadOp (HM.fromListWith const dists) edges
   where
    parseDist = Aeson.withObject "SpreadDist" $ \dst -> do
      parent <- dst .: "parent" >>= parseJSON
      child <- dst .: "child" >>= parseJSON
      pure (parent, child)

instance (Notation n) => ToJSON (Spread n) where
  toJSON (SpreadOp dists edges) =
    Aeson.object
      [ "children" .= fmap distToJSON (HM.toList dists)
      , "midEdges" .= edges
      ]
   where
    distToJSON (parent, child) = Aeson.object ["parent" .= parent, "child" .= child]

-- | 'Leftmost' specialized to the split, freeze, and spread operations of the grammar.
type PVLeftmost n = Leftmost (Split n) (Freeze n) (Spread n)

-- helpers
-- =======

-- -- | Helper: parses a note's pitch from JSON.
-- parseJSONNote :: (Notation n) => Aeson.Value -> Aeson.Parser (Note n)
-- parseJSONNote = _

-- | Helper: parses an edge from JSON.
parseEdge
  :: (Notation n) => Aeson.Value -> Aeson.Parser (StartStop (Note n), StartStop (Note n))
parseEdge = Aeson.withObject "Edge" $ \v -> do
  l <- v .: "left" >>= mapM parseJSON -- TODO: this might be broken wrt. StartStop
  r <- v .: "right" >>= mapM parseJSON
  pure (l, r)

-- | Helper: parses an inner edge from JSON
parseInnerEdge :: (Notation n) => Aeson.Value -> Aeson.Parser (Note n, Note n)
parseInnerEdge = Aeson.withObject "InnerEdge" $ \v -> do
  l <- v .: "left"
  r <- v .: "right"
  case (l, r) of
    (Inner il, Inner ir) -> do
      pl <- parseJSON il
      pr <- parseJSON ir
      pure (pl, pr)
    _ -> fail "Edge is not an inner edge"

edgeToJSON :: (ToJSON a) => (a, a) -> Aeson.Value
edgeToJSON (l, r) = Aeson.object ["left" .= l, "right" .= r]

-- edgeToEncoding :: (ToJSON a) => (a, a) -> Aeson.Encoding
-- edgeToEncoding (l, r) = Aeson.pairs $ ("left" .= l) <> ("right" .= r)

-- | An 'Analysis' specialized to PV types.
type PVAnalysis n = Analysis (Split n) (Freeze n) (Spread n) (Edges n) (Notes n)

{- | Loads an analysis from a JSON file
 (as exported by the annotation tool).
-}
loadAnalysis :: FilePath -> IO (Either String (PVAnalysis SPitch))
loadAnalysis = Aeson.eitherDecodeFileStrict

{- | Loads an analysis from a JSON file
 (as exported by the annotation tool).
 Converts all pitches to pitch classes.
-}
loadAnalysis' :: FilePath -> IO (Either String (PVAnalysis SPC))
loadAnalysis' fn = fmap (analysisMapPitch (pc @SInterval)) <$> loadAnalysis fn

{- | Loads a MusicXML file and returns a list of salami slices.
 Each note is expressed as a pitch and a flag that indicates
 whether the note continues in the next slice.
-}
slicesFromFile :: FilePath -> IO [[(Note SPitch, Music.RightTied)]]
slicesFromFile file = do
  txt <- TL.readFile file
  case MusicXML.parseWithIds True txt of
    Nothing -> pure []
    Just doc -> do
      let (xmlNotes, _) = MusicXML.parseScore doc
          notes = MusicXML.asNoteWithIdHeard <$> xmlNotes
          slices = Music.slicePiece Music.tiedSlicer notes
      pure $ mkSlice <$> filter (not . null) slices
 where
  mkSlice notes = mkNote <$> notes
  mkNote (note, tie) = (Note (Music.pitch note) (fromMaybe "" $ Music.getId note), Music.rightTie tie)

-- | Converts salami slices (as returned by 'slicesFromFile') to a path as expected by parsers.
slicesToPath
  :: forall i
   . (Interval i, Ord i, Eq i)
  => [[(Note (Pitch i), Music.RightTied)]]
  -> Path [Note (Pitch i)] [Edge (Pitch i)]
slicesToPath = go 0
 where
  -- normalizeTies (s : next : rest) = (fixTie <$> s)
  --   : normalizeTies (next : rest)
  --  where
  --   nextNotes = fst <$> next
  --   fixTie (p, t) = if p `L.elem` nextNotes then (p, t) else (p, Ends)
  -- normalizeTies [s] = [map (fmap $ const Ends) s]
  -- normalizeTies []  = []
  mkNote i (Note p id) = Note p ("slice" <> show i <> "-" <> id)
  mkEdge i (_, Music.Ends) = Nothing
  mkEdge i (p, Music.Holds) = Just (Inner $ mkNote i p, Inner $ mkNote (i + 1) p)
  go :: Int -> [[(Note (Pitch i), Music.RightTied)]] -> Path [Note (Pitch i)] [Edge (Pitch i)]
  go _ [] = error "cannot construct path from empty list"
  go i [notes] = PathEnd (mkNote i . fst <$> notes)
  go i (notes : rest) = Path (mkNote i . fst <$> notes) (mapMaybe (mkEdge i) notes) $ go (i + 1) rest

{- | Loads a MusicXML File and returns a surface path
 as input to parsers.
-}
loadSurface :: FilePath -> IO (Path [Note SPitch] [Edge SPitch])
loadSurface = fmap slicesToPath . slicesFromFile

{- | Loads a MusicXML File
 and returns a surface path of the given range of slices.
-}
loadSurface'
  :: FilePath
  -- ^ path to a MusicXML file
  -> Int
  -- ^ the first slice to include (starting at 0)
  -> Int
  -- ^ the last slice to include
  -> IO (Path [Note SPitch] [Edge SPitch])
loadSurface' fn from to =
  slicesToPath . drop from . take (to - from + 1) <$> slicesFromFile fn

-- | Apply an applicative action to all pitches in an analysis.
analysisTraversePitch
  :: (Applicative f, Eq n', Hashable n', Ord n')
  => (n -> f n')
  -> PVAnalysis n
  -> f (PVAnalysis n')
analysisTraversePitch f (Analysis deriv top) = do
  deriv' <- traverse (leftmostTraversePitch f) deriv
  top' <- pathTraversePitch f top
  pure $ Analysis deriv' top'

-- | Map a function over all pitches in an analysis.
analysisMapPitch
  :: (Eq n', Hashable n', Ord n') => (n -> n') -> PVAnalysis n -> PVAnalysis n'
analysisMapPitch f = runIdentity . analysisTraversePitch (pure . f)

pathTraversePitch
  :: (Applicative f, Eq n', Hashable n')
  => (n -> f n')
  -> Path (Edges n) (Notes n)
  -> f (Path (Edges n') (Notes n'))
pathTraversePitch f (Path e a rest) = do
  e' <- edgesTraversePitch f e
  a' <- notesTraversePitch f a
  rest' <- pathTraversePitch f rest
  pure $ Path e' a' rest'
pathTraversePitch f (PathEnd e) = PathEnd <$> edgesTraversePitch f e

traverseEdge :: (Applicative f) => (n -> f n') -> (n, n) -> f (n', n')
traverseEdge f (n1, n2) = (,) <$> f n1 <*> f n2

notesTraversePitch
  :: (Eq n, Hashable n, Applicative f) => (a -> f n) -> Notes a -> f (Notes n)
notesTraversePitch f (Notes notes) = Notes <$> traverseSet (traverse f) notes

edgesTraversePitch
  :: (Applicative f, Eq n', Hashable n')
  => (n -> f n')
  -> Edges n
  -> f (Edges n')
edgesTraversePitch f (Edges reg pass) = do
  reg' <- traverseSet (traverseEdge $ traverse $ traverse f) reg
  pass' <- MS.traverse (traverseEdge $ traverse f) pass
  pure $ Edges reg' pass'

leftmostTraversePitch
  :: (Applicative f, Eq n', Hashable n', Ord n')
  => (n -> f n')
  -> Leftmost (Split n) (Freeze n) (Spread n)
  -> f (Leftmost (Split n') (Freeze n') (Spread n'))
leftmostTraversePitch f lm = case lm of
  LMSplitLeft s -> LMSplitLeft <$> splitTraversePitch f s
  LMSplitRight s -> LMSplitRight <$> splitTraversePitch f s
  LMSplitOnly s -> LMSplitOnly <$> splitTraversePitch f s
  LMFreezeLeft fr -> LMFreezeLeft <$> freezeTraversePitch f fr
  LMFreezeOnly fr -> LMFreezeOnly <$> freezeTraversePitch f fr
  LMSpread h -> LMSpread <$> spreadTraversePitch f h

splitTraversePitch
  :: forall f n n'
   . (Applicative f, Ord n', Hashable n')
  => (n -> f n')
  -> Split n
  -> f (Split n')
splitTraversePitch f (SplitOp reg pass ls rs kl kr pl pr) = do
  reg' <- traverseElabo (traverseEdge $ traverse $ traverse f) reg
  pass' <- traverseElabo (traverseEdge $ traverse f) pass
  ls' <- traverseElabo (traverse f) ls
  rs' <- traverseElabo (traverse f) rs
  kl' <- traverseSet (traverseEdge $ traverse $ traverse f) kl
  kr' <- traverseSet (traverseEdge $ traverse $ traverse f) kr
  pl' <- MS.traverse (traverseEdge $ traverse f) pl
  pr' <- MS.traverse (traverseEdge $ traverse f) pr
  pure $ SplitOp reg' pass' ls' rs' kl' kr' pl' pr'
 where
  traverseElabo
    :: forall p p' o
     . (Ord p')
    => (p -> f p')
    -> M.Map p [(Note n, o)]
    -> f (M.Map p' [(Note n', o)])
  traverseElabo fparent mp = fmap M.fromList $ for (M.toList mp) $ \(e, cs) ->
    do
      e' <- fparent e
      cs' <- traverse (\(n, o) -> (,o) <$> traverse f n) cs
      pure (e', cs')

spreadTraversePitch
  :: (Applicative f, Eq n', Hashable n')
  => (n -> f n')
  -> Spread n
  -> f (Spread n')
spreadTraversePitch f (SpreadOp dist edges) = do
  dist' <- traverse travDist $ HM.toList dist
  edges' <- edgesTraversePitch f edges
  pure $ SpreadOp (HM.fromListWith const dist') edges'
 where
  travDist (k, v) = do
    k' <- traverse f k
    v' <- traverse f v
    pure (k', v')

freezeTraversePitch :: (Applicative f, Hashable n') => (n -> f n') -> Freeze n -> f (Freeze n')
freezeTraversePitch f (FreezeOp ties) = FreezeOp <$> traverseSet (traverseEdge $ traverse $ traverse f) ties
