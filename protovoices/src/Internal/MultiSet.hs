{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE TupleSections #-}

{- | A simple implementation of multisets as hash maps to counts.
 Supports just what is needed for protovoice slices.
-}
module Internal.MultiSet where

import Control.DeepSeq (NFData)
import Control.Monad as M
import Data.Foldable
  ( all
  , foldl'
  )
import Data.HashMap.Strict qualified as HM
import Data.HashSet (HashSet)
import Data.Hashable (Hashable)
import GHC.Generics (Generic)
import Prelude hiding (lookup)

{- | A hash multiset.
 A unordered collection of items that can occur several times.
-}
newtype MultiSet a = MS {unMS :: HM.HashMap a Int}
  deriving (Eq, Ord, Show, Generic, Hashable, NFData)

instance (Eq a, Hashable a) => Semigroup (MultiSet a) where
  (MS a) <> (MS b) = MS $ HM.unionWith (+) a b

-- instance Foldable MultiSet where
toList :: MultiSet a -> [a]
toList (MS a) = concat $ HM.mapWithKey (flip replicate) a

all :: (a -> Bool) -> MultiSet a -> Bool
all p (MS a) = Data.Foldable.all p $ HM.keysSet a

toOccurList :: MultiSet k -> [(k, Int)]
toOccurList (MS m) = HM.toList m

distinctElems :: MultiSet k -> [k]
distinctElems (MS m) = HM.keys m

union :: (Eq a, Hashable a) => MultiSet a -> MultiSet a -> MultiSet a
union (MS a) (MS b) = MS $ HM.unionWith (+) a b

unions :: (Foldable t, Eq a0, Hashable a0) => t (MultiSet a0) -> MultiSet a0
unions = foldl' union empty

maxUnion :: (Eq a, Hashable a) => MultiSet a -> MultiSet a -> MultiSet a
maxUnion (MS a) (MS b) = MS $ HM.unionWith max a b

deleteN :: Int -> Int -> Maybe Int
deleteN n m
  | m <= n = Nothing
  | otherwise = Just (m - n)

(\\) :: (Eq a, Hashable a) => MultiSet a -> MultiSet a -> MultiSet a
(MS a) \\ (MS b) = MS $ HM.differenceWith (flip deleteN) a b

toSet :: MultiSet k -> HashSet k
toSet (MS a) = HM.keysSet a

size :: MultiSet a -> Int
size (MS a) = foldl' (+) 0 a

cardinality :: MultiSet a -> Int
cardinality (MS a) = foldl' max 0 a

map :: (Eq b, Hashable b) => (a -> b) -> MultiSet a -> MultiSet b
map f (MS a) =
  MS $ HM.foldlWithKey' (\m k o -> HM.insertWith (+) (f k) o m) HM.empty a

traverse
  :: (Eq b, Hashable b, Applicative f)
  => (a -> f b)
  -> MultiSet a
  -> f (MultiSet b)
traverse f (MS m) =
  MS . HM.fromListWith (+)
    <$> Prelude.traverse
      (\(k, v) -> (,v) <$> f k)
      (HM.toList m)

filter :: (a -> Bool) -> MultiSet a -> MultiSet a
filter f (MS a) = MS $ HM.filterWithKey (\k _ -> f k) a

fromList :: (Foldable t, Eq a, Hashable a) => t a -> MultiSet a
fromList xs = MS $ foldl' (\m x -> HM.insertWith (+) x 1 m) HM.empty xs

fromSet :: (Foldable t, Eq a, Hashable a) => t a -> MultiSet a
fromSet xs = MS $ foldl' (\m x -> HM.insert x 1 m) HM.empty xs

null :: MultiSet k -> Bool
null (MS a) = HM.null a

delete :: (Eq a, Hashable a) => a -> MultiSet a -> MultiSet a
delete x = MS . HM.update (deleteN 1) x . unMS

empty :: MultiSet a
empty = MS HM.empty

foldM :: (Monad m) => (b -> a -> m b) -> b -> MultiSet a -> m b
foldM f b (MS as) = HM.foldlWithKey' doFold (pure b) as
 where
  doFold mb a i = do
    b' <- mb
    M.foldM f b' (replicate i a)

insertMany :: (Eq a, Hashable a) => a -> Int -> MultiSet a -> MultiSet a
insertMany a n (MS as)
  | n <= 0 = MS as
  | otherwise = MS $ HM.insertWith (+) a n as

insert :: (Eq a, Hashable a) => a -> MultiSet a -> MultiSet a
insert a = insertMany a 1

singleton :: (Hashable a) => a -> MultiSet a
singleton a = MS $ HM.singleton a 1

member :: (Eq k, Hashable k) => k -> MultiSet k -> Bool
member a as = lookup a as > 0

lookup :: (Eq k, Hashable k) => k -> MultiSet k -> Int
lookup a (MS as) = HM.lookupDefault 0 a as

(!) :: (Eq k, Hashable k) => MultiSet k -> k -> Int
as ! a = lookup a as
