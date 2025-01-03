{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeApplications #-}

import Data.List (foldl')
import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.Static (R)
import Lens.Micro.TH (makeLenses)
import Control.Monad.ST (runST)
import System.Random.MWC (create, uniformR)
import Data.Traversable (for)
import Data.Maybe (fromMaybe)

import Numeric.AD (AD, diff, diff')
import Numeric.AD.Mode.Forward (Forward)
import qualified Numeric.AD.Mode.Forward as F
import Numeric.AD.Mode.Reverse (Reverse)
import qualified Numeric.AD.Mode.Reverse as R
import qualified Data.Vector.Unboxed as VU
import Data.Foldable (foldl')
import Control.Monad.State.Lazy (State, runState, get, put, MonadState)

-- Hyperparameters
vocabSize :: Int
vocabSize = 10000

dModel :: Int
dModel = 512

dFF :: Int
dFF = 2048

numHeads :: Int
numHeads = 8

numLayers :: Int
numLayers = 6

batchSize :: Int
batchSize = 64

seqLen :: Int
seqLen = 200

learningRate :: Double
learningRate = 0.001

type Vector a = R dModel a
type Matrix a = Matrix R a
type Weight a = Weights a

-- Data Structures for weights

data Weights a = Weights {
  _embeddingWeights :: Matrix a,
  _wq :: Matrix a,
  _wk :: Matrix a,
  _wv :: Matrix a,
  _wo :: Matrix a,
  _ff1 :: Matrix a,
  _ff2 :: Matrix a,
  _normGamma :: Vector a,
  _normBeta :: Vector a,
  _outputWeights :: Matrix a,
  _outputBias :: Vector a
}

makeLenses ''Weights

-- Funciones Auxiliares

scale :: (Num a) => a -> Matrix a -> Matrix a
scale a m = m * a

-- Matrix Add
matrixAdd :: (Num a) => [Vector a] -> Matrix a -> Matrix a
matrixAdd vecs mat =  fromRows $ zipWith (+) vecs (rows mat)

-- Funciones Para Operar Con AD

-- Embedding Layer
embedding :: Matrix a -> Vector a -> Vector a
embedding w input = w !> toList (map round input)
  where
    toList (x:xs) = x : toList xs
    toList [] = []

-- Positional Encoding
positionalEncoding :: Int -> Int -> Matrix Double
positionalEncoding seqLen dModel = fromColumns $ map peVector [0..seqLen-1]
  where
    peVector pos = fromList $ concatMap (peValue pos) [0..dModel-1]
    peValue pos i
        | even i  = [sin(fromIntegral pos / (10000 ** (fromIntegral i / fromIntegral dModel))) ]
        | otherwise = [cos(fromIntegral pos / (10000 ** (fromIntegral (i -1) / fromIntegral dModel)))]

-- Multi-Head Attention
type QueryKeyVal a = (Matrix a, Matrix a, Matrix a)

project :: Weight a -> Vector a -> Int -> (Vector a, Vector a, Vector a)
project w x headIdx = (q, k ,v)
  where
    q = wq w !> toList (x)
    k = wk w !> toList (x)
    v = wv w !> toList (x)
    toList (x) = [x]

scaledDotProductAttention ::  Matrix a -> Matrix a -> Matrix a -> Maybe (Matrix a)
scaledDotProductAttention q k v = Just (multStd q (trans k) `scale` (1/sqrt(fromIntegral (cols q)) ) `multStd` v)

--  * Attention Weights (Softmax)
softmax :: (Floating a, Traversable f, VU.Unbox a) => f a -> f a
softmax x = expX / sum expX
  where
    expX = fmap exp x

multiHeadAttention :: Weight a -> Matrix a -> Int -> Matrix a
multiHeadAttention weights x numHeads =  fromColumns $ concatMap (runAtt)  [0..numHeads-1]
  where
    runAtt headIdx =  fromMaybe (error "Error en atencion multi cabeza") $ fmap (toList) $ scaledDotProductAttention q k v
      where
        (q, k , v) =  fromColumns $ map (project weights) (map (\v -> x !> toList v) (rows x) ) !! headIdx

-- FeedForward Network
feedForward :: Weight a -> Vector a -> Vector a
feedForward weights x =  relu (multStd (ff1 weights) (fromList x) ) `multStd`  ff2 weights
  where
    relu v =  (fromList . map (\x -> max 0 x) . toList) v

-- Add & Layer Norm
addLayerNorm :: Weight a -> Vector a -> Vector a -> Vector a
addLayerNorm weights x sublayerOutput = layerNorm weights (x + sublayerOutput)

layerNorm :: Weight a -> Vector a -> Vector a
layerNorm weights x = ( scale (recip (sqrt (var + 1e-5))) (x - avg) ) `scale` normGamma weights + normBeta weights
  where
    avg =  (sumElements x) / (fromIntegral (dim x) )
    var = sumElements (map (\x -> (x - avg)^2 ) (toList x)) / (fromIntegral (dim x) )

-- Encoder Block
encoderBlock :: Weight a -> Vector a -> Maybe (Vector a)
encoderBlock weights x = do
  attOutput <- (fmap fromList . toList) <$> multiHeadAttention weights (fromColumns [x]) numHeads
  let addNorm1 = addLayerNorm weights x attOutput
  let ffOutput = feedForward weights addNorm1
  let addNorm2 = addLayerNorm weights addNorm1 ffOutput
  return addNorm2

-- Decoder Block
decoderBlock :: Weight a -> Vector a -> Matrix a -> Maybe (Vector a)
decoderBlock weights x encoderOutput = do
  maskedAttOutput <- (fmap fromList . toList) <$> multiHeadAttention weights (fromColumns [x]) numHeads
  let addNorm1 = addLayerNorm weights x maskedAttOutput

  attOutput <- (fmap fromList . toList) <$> multiHeadAttention weights (fromColumns [x] ) numHeads --Atiende al EncoderOutput

  let addNorm2 = addLayerNorm weights addNorm1 attOutput
  let ffOutput = feedForward weights addNorm2
  let addNorm3 = addLayerNorm weights addNorm2 ffOutput
  return addNorm3

-- Output Layer (Linear + Softmax)
outputLayer :: Weight a -> Vector a -> Vector a
outputLayer weights x = (outputWeights weights) `multStd` fromList x + outputBias weights


-- Transformer Model (Ahora usando `AD` para tipos)
transformer :: Weight (AD Reverse Double) -> Matrix Double -> Matrix Double -> Matrix (AD Reverse Double)
transformer weights src tgt = runST $ do
    -- Embeddings + Positional Encoding
    let srcEmbeddings =  matrixAdd (map (\v -> embedding (embeddingWeights weights) v) (rows src)) (positionalEncoding seqLen dModel)
    let tgtEmbeddings =  matrixAdd (map (\v -> embedding (embeddingWeights weights) v) (rows tgt)) (positionalEncoding seqLen dModel)

    -- Encoder Blocks
    encoderOutput <- foldM  (\output block -> mapM (block weights) (rows output) ) srcEmbeddings (replicate numLayers encoderBlock)
    -- Decoder Blocks
    decoderOutput <- foldM  (\output block -> mapM (\x -> block weights x encoderOutput) (rows output) )  tgtEmbeddings (replicate numLayers decoderBlock)


    return $ fromRows  (map (outputLayer weights)  decoderOutput )

-- Helpers for matrix manipulation

matrixAdd :: (Num a) => [Vector a] -> Matrix a -> Matrix a
matrixAdd vecs mat =  fromRows $ zipWith (+) vecs (rows mat)

-- Random initialization of matrices
randomWeights :: IO (Weight Double)
randomWeights = do
  gen <- create
  let initMatrix rows cols = do
      m <- for [1..rows] $ \_ ->
             for [1..cols] $ \_ -> uniformR (-1,1) gen
      return $ fromLists m

  embedW <- initMatrix vocabSize dModel
  qW     <- initMatrix dModel dModel
  kW     <- initMatrix dModel dModel
  vW     <- initMatrix dModel dModel
  oW     <- initMatrix dModel dModel
  ff1W   <- initMatrix dModel dFF
  ff2W   <- initMatrix dFF dModel
  gW   <- initMatrix dModel 1
  bW   <- initMatrix dModel 1
  outW   <- initMatrix dModel vocabSize
  outB   <- initMatrix vocabSize 1

  return $ Weights embedW qW kW vW oW ff1W ff2W
                 (fromList $ map (!!0) (toLists gW) )
                 (fromList $ map (!!0) (toLists bW) )
                 outW (fromList $ map (!!0) (toLists outB))


-- Funcion de perdida (Cross Entropy)
crossEntropyLoss :: Matrix (AD Reverse Double) -> Matrix Int -> AD Reverse Double
crossEntropyLoss output target =  sumElements (zipWith  lossRow (rows output) (rows target) ) / (fromIntegral $ rows output)
  where
    lossRow outputRow targetRow =
      let probs = softmax outputRow
          targetIdx = head (toList targetRow) -- Asumiendo que el target es un vector de 1 elemento con el id de la clase
      in  negate $ log (probs VU.! targetIdx)

-- Actualización de pesos
updateWeights :: Weight Double -> Weight (AD Reverse Double) ->  Weight Double
updateWeights weights gradWeights =
  Weights
    (embeddingWeights weights - scale learningRate (fromIntegral <$> R.grad (embeddingWeights gradWeights) ) )
    (wq weights -  scale learningRate  (fromIntegral <$> R.grad (wq gradWeights) ) )
    (wk weights - scale learningRate  (fromIntegral <$> R.grad (wk gradWeights) ) )
    (wv weights - scale learningRate  (fromIntegral <$> R.grad (wv gradWeights) ) )
    (wo weights - scale learningRate  (fromIntegral <$> R.grad (wo gradWeights) ) )
    (ff1 weights - scale learningRate  (fromIntegral <$> R.grad (ff1 gradWeights) ) )
    (ff2 weights - scale learningRate  (fromIntegral <$> R.grad (ff2 gradWeights) ) )
    (normGamma weights - scale learningRate (fromIntegral <$> R.grad (normGamma gradWeights) ) )
    (normBeta weights - scale learningRate  (fromIntegral <$> R.grad (normBeta gradWeights) ) )
    (outputWeights weights - scale learningRate (fromIntegral <$> R.grad (outputWeights gradWeights) ) )
    (outputBias weights - scale learningRate (fromIntegral <$> R.grad (outputBias gradWeights) ) )

-- | Generates random numbers in a given range.
randomRs :: (Random a, Num a) => (a, a) -> StdGen -> [a]
randomRs range gen = randoms gen'
  where gen' = gen :: StdGen

-- | Generates a single random number in a given range
randomR :: (Random a, Num a) => (a, a) -> StdGen -> a
randomR range gen = fst $ random gen

-- Training Loop
train :: Int -> Matrix Double -> Matrix Int -> Weight Double -> IO (Weight Double)
train epochs src tgt weights =  foldM (\acc epoch -> do
        let (loss, gradWeights) = diff' (\w -> crossEntropyLoss (transformer w src (fromIntegral <$> tgt)) tgt ) acc
        let updatedWeights = updateWeights acc gradWeights
        putStrLn $ "Epoch: " ++ show epoch ++ ", Loss: " ++ show loss
        return updatedWeights
    ) weights [1..epochs]

main :: IO ()
main = do
  weights <- randomWeights

  -- Sample input data (random integers representing tokens)
  let src = fromLists $ replicate batchSize $ map fromIntegral $ take seqLen $ randomRs (1, vocabSize) (mkStdGen 42) :: Matrix Double
  let tgt = fromLists $ replicate batchSize $ map (fromIntegral . head)  $ chunksOf 1 $ take seqLen $ randomRs (1, vocabSize) (mkStdGen 43) :: Matrix Int
  let epochs = 10
  trainedWeights <- train epochs src tgt weights

  let output = transformer (fmap R.auto trainedWeights) src (fromIntegral <$> tgt)
  print output

chunksOf :: Int -> [a] -> [[a]]
chunksOf _ [] = []
chunksOf n xs = take n xs : chunksOf n (drop n xs)
