#ifdef __CUDACC__
#define CUDA_CONST_VAR __device__
#endif
# include "PixelQuintuplet.cuh"
#include "allocate.h"

SDL::pixelQuintuplets::pixelQuintuplets()
{
    pixelIndices = nullptr;
    T5Indices = nullptr;
    nPixelQuintuplets = nullptr;
    isDup = nullptr;
    score = nullptr;
#ifdef TRACK_EXTENSIONS
    pixelRadius = nullptr;
    quintupletRadius = nullptr;
    centerX = nullptr;
    centerY = nullptr;
    logicalLayers = nullptr;
    hitIndices = nullptr;
    lowerModuleIndices = nullptr;
#endif
}

SDL::pixelQuintuplets::~pixelQuintuplets()
{
}

void SDL::pixelQuintuplets::freeMemoryCache()
{
#ifdef Explicit_PT5
    int dev;
    cudaGetDevice(&dev);
    cms::cuda::free_device(dev,pixelIndices);
    cms::cuda::free_device(dev,T5Indices);
    cms::cuda::free_device(dev,nPixelQuintuplets);
    cms::cuda::free_device(dev,isDup);
    cms::cuda::free_device(dev,score);
    cms::cuda::free_device(dev,eta);
    cms::cuda::free_device(dev,phi);
#ifdef TRACK_EXTENSIONS
    cms::cuda::free_device(dev, hitIndices);
    cms::cuda::free_device(dev, logicalLayers);
    cms::cuda::free_device(dev, lowerModuleIndices);
    cms::cuda::free_device(dev, centerX);
    cms::cuda::free_device(dev, centerY);
    cms::cuda::free_device(dev, pixelRadius);
    cms::cuda::free_device(dev, quintupletRadius);
#endif
#else
    cms::cuda::free_managed(pixelIndices);
    cms::cuda::free_managed(T5Indices);
    cms::cuda::free_managed(nPixelQuintuplets);
    cms::cuda::free_managed(isDup);
    cms::cuda::free_managed(score);
    cms::cuda::free_managed(eta);
    cms::cuda::free_managed(phi);
#ifdef TRACK_EXTENSIONS
    cms::cuda::free_managed(hitIndices);
    cms::cuda::free_managed(logicalLayers);
    cms::cuda::free_managed(lowerModuleIndices);
    cms::cuda::free_managed(centerX);
    cms::cuda::free_managed(centerY);
    cms::cuda::free_managed(pixelRadius);
    cms::cuda::free_managed(quintupletRadius);
#endif
#endif
}
void SDL::pixelQuintuplets::freeMemory(cudaStream_t stream)
{
    //cudaFreeAsync(pixelIndices,stream);
    //cudaFreeAsync(T5Indices,stream);
    //cudaFreeAsync(nPixelQuintuplets,stream);
    //cudaFreeAsync(isDup,stream);
    //cudaFreeAsync(score,stream);
    //cudaFreeAsync(eta,stream);
    //cudaFreeAsync(phi,stream);
    cudaFree(pixelIndices);
    cudaFree(T5Indices);
    cudaFree(nPixelQuintuplets);
    cudaFree(isDup);
    cudaFree(score);
    cudaFree(eta);
    cudaFree(phi);

#ifdef TRACK_EXTENSIONS
    cudaFree(logicalLayers);
    cudaFree(hitIndices);
    cudaFree(lowerModuleIndices);
    cudaFree(pixelRadius);
    cudaFree(quintupletRadius);
    cudaFree(centerX);
    cudaFree(centerY);
#endif
#ifdef CUT_VALUE_DEBUG
    cudaFree(rzChiSquared);
    cudaFree(rPhiChiSquared);
    cudaFree(rPhiChiSquaredInwards);
#endif
cudaStreamSynchronize(stream);
}

void SDL::pixelQuintuplets::resetMemory(unsigned int maxPixelQuintuplets,cudaStream_t stream)
{
    cudaMemsetAsync(pixelIndices,0, maxPixelQuintuplets * sizeof(unsigned int),stream);
    cudaMemsetAsync(T5Indices,0, maxPixelQuintuplets * sizeof(unsigned int),stream);
    cudaMemsetAsync(nPixelQuintuplets,0, sizeof(unsigned int),stream);
    cudaMemsetAsync(isDup,0, maxPixelQuintuplets * sizeof(bool),stream);
    cudaMemsetAsync(score,0, maxPixelQuintuplets * sizeof(FPX),stream);
    cudaMemsetAsync(eta , 0, maxPixelQuintuplets * sizeof(FPX),stream);
    cudaMemsetAsync(phi , 0, maxPixelQuintuplets * sizeof(FPX),stream);
}
void SDL::createPixelQuintupletsInUnifiedMemory(struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, unsigned int maxPixelQuintuplets,cudaStream_t stream)
{
#ifdef CACHE_ALLOC
    pixelQuintupletsInGPU.pixelIndices        = (unsigned int*)cms::cuda::allocate_managed(maxPixelQuintuplets * sizeof(unsigned int),stream);
    pixelQuintupletsInGPU.T5Indices           = (unsigned int*)cms::cuda::allocate_managed(maxPixelQuintuplets * sizeof(unsigned int),stream);
    pixelQuintupletsInGPU.nPixelQuintuplets   = (unsigned int*)cms::cuda::allocate_managed(sizeof(unsigned int),stream);
    pixelQuintupletsInGPU.isDup               = (bool*)cms::cuda::allocate_managed(maxPixelQuintuplets * sizeof(bool),stream);
    pixelQuintupletsInGPU.score               = (FPX*)cms::cuda::allocate_managed(maxPixelQuintuplets * sizeof(FPX),stream);
    pixelQuintupletsInGPU.eta                 = (FPX*)cms::cuda::allocate_managed(maxPixelQuintuplets * sizeof(FPX),stream);
    pixelQuintupletsInGPU.phi                 = (FPX*)cms::cuda::allocate_managed(maxPixelQuintuplets * sizeof(FPX),stream);
#ifdef TRACK_EXTENSIONS
    pixelQuintupletsInGPU.hitIndices = (unsigned int*)cms::cuda::allocate_managed(maxPixelQuintuplets * 14 * sizeof(unsigned int), stream);
    pixelQuintupletsInGPU.logicalLayers = (unsigned int*)cms::cuda::allocate_managed(maxPixelQuintuplets * 7 * sizeof(unsigned int), stream);
    pixelQuintupletsInGPU.lowerModuleIndices = (unsigned int*)cms::cuda::allocate_managed(maxPixelQuintuplets * 7 * sizeof(unsigned int), stream);
    pixelQuintupletsInGPU.centerX          = (FPX*)cms::cuda::allocate_managed(maxPixelQuintuplets * sizeof(FPX), stream);
    pixelQuintupletsInGPU.centerY          = (FPX*)cms::cuda::allocate_managed(maxPixelQuintuplets * sizeof(FPX), stream);
    pixelQuintupletsInGPU.pixelRadius      = (FPX*)cms::cuda::allocate_managed(maxPixelQuintuplets * sizeof(FPX), stream);
    pixelQuintupletsInGPU.quintupletRadius = (FPX*)cms::cuda::allocate_managed(maxPixelQuintuplets * sizeof(FPX), stream);
#endif
#else
    cudaMallocManaged(&pixelQuintupletsInGPU.pixelIndices, maxPixelQuintuplets * sizeof(unsigned int));
    cudaMallocManaged(&pixelQuintupletsInGPU.T5Indices, maxPixelQuintuplets * sizeof(unsigned int));
    cudaMallocManaged(&pixelQuintupletsInGPU.nPixelQuintuplets, sizeof(unsigned int));
    cudaMallocManaged(&pixelQuintupletsInGPU.isDup, maxPixelQuintuplets * sizeof(bool));
    cudaMallocManaged(&pixelQuintupletsInGPU.score, maxPixelQuintuplets * sizeof(FPX));
    cudaMallocManaged(&pixelQuintupletsInGPU.eta  , maxPixelQuintuplets * sizeof(FPX));
    cudaMallocManaged(&pixelQuintupletsInGPU.phi  , maxPixelQuintuplets * sizeof(FPX));

#ifdef TRACK_EXTENSIONS
    cudaMallocManaged(&pixelQuintupletsInGPU.logicalLayers, maxPixelQuintuplets * 7 * sizeof(unsigned int));
    cudaMallocManaged(&pixelQuintupletsInGPU.lowerModuleIndices, maxPixelQuintuplets * 7 * sizeof(unsigned int));
    cudaMallocManaged(&pixelQuintupletsInGPU.hitIndices, maxPixelQuintuplets * 14 * sizeof(unsigned int));
    cudaMallocManaged(&pixelQuintupletsInGPU.pixelRadius, maxPixelQuintuplets * sizeof(FPX));
    cudaMallocManaged(&pixelQuintupletsInGPU.quintupletRadius, maxPixelQuintuplets * sizeof(FPX));
    cudaMallocManaged(&pixelQuintupletsInGPU.centerX, maxPixelQuintuplets * sizeof(FPX));
    cudaMallocManaged(&pixelQuintupletsInGPU.centerY, maxPixelQuintuplets * sizeof(FPX));
#endif

#ifdef CUT_VALUE_DEBUG
    cudaMallocManaged(&pixelQuintupletsInGPU.rzChiSquared, maxPixelQuintuplets * sizeof(unsigned int));
    cudaMallocManaged(&pixelQuintupletsInGPU.rPhiChiSquared, maxPixelQuintuplets * sizeof(unsigned int));
    cudaMallocManaged(&pixelQuintupletsInGPU.rPhiChiSquaredInwards, maxPixelQuintuplets * sizeof(unsigned int));
#endif
#endif

    cudaMemsetAsync(pixelQuintupletsInGPU.nPixelQuintuplets, 0, sizeof(unsigned int),stream);
  cudaStreamSynchronize(stream);
}

void SDL::createPixelQuintupletsInExplicitMemory(struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, unsigned int maxPixelQuintuplets,cudaStream_t stream)
{
#ifdef CACHE_ALLOC
//    cudaStream_t stream=0;
    int dev;
    cudaGetDevice(&dev);
    pixelQuintupletsInGPU.pixelIndices        = (unsigned int*)cms::cuda::allocate_device(dev,maxPixelQuintuplets * sizeof(unsigned int),stream);
    pixelQuintupletsInGPU.T5Indices           = (unsigned int*)cms::cuda::allocate_device(dev,maxPixelQuintuplets * sizeof(unsigned int),stream);
    pixelQuintupletsInGPU.nPixelQuintuplets   = (unsigned int*)cms::cuda::allocate_device(dev,sizeof(unsigned int),stream);
    pixelQuintupletsInGPU.isDup               = (bool*)cms::cuda::allocate_device(dev,maxPixelQuintuplets * sizeof(bool),stream);
    pixelQuintupletsInGPU.score               = (FPX*)cms::cuda::allocate_device(dev,maxPixelQuintuplets * sizeof(FPX),stream);
    pixelQuintupletsInGPU.eta                 = (FPX*)cms::cuda::allocate_device(dev,maxPixelQuintuplets * sizeof(FPX),stream);
    pixelQuintupletsInGPU.phi                 = (FPX*)cms::cuda::allocate_device(dev,maxPixelQuintuplets * sizeof(FPX),stream);
#ifdef TRACK_EXTENSIONS
    pixelQuintupletsInGPU.hitIndices = (unsigned int*)cms::cuda::allocate_device(dev, maxPixelQuintuplets * 14 * sizeof(unsigned int), stream);
    pixelQuintupletsInGPU.logicalLayers = (unsigned int*)cms::cuda::allocate_device(dev, maxPixelQuintuplets * 7 * sizeof(unsigned int), stream);
    pixelQuintupletsInGPU.lowerModuleIndices = (unsigned int*)cms::cuda::allocate_device(dev, maxPixelQuintuplets * 7 * sizeof(unsigned int), stream);
    pixelQuintupletsInGPU.centerX          = (FPX*)cms::cuda::allocate_device(dev, maxPixelQuintuplets * sizeof(FPX), stream);
    pixelQuintupletsInGPU.centerY          = (FPX*)cms::cuda::allocate_device(dev, maxPixelQuintuplets * sizeof(FPX), stream);
    pixelQuintupletsInGPU.pixelRadius      = (FPX*)cms::cuda::allocate_device(dev, maxPixelQuintuplets * sizeof(FPX), stream);
    pixelQuintupletsInGPU.quintupletRadius = (FPX*)cms::cuda::allocate_device(dev, maxPixelQuintuplets * sizeof(FPX), stream);
#endif
#else
    cudaMalloc(&pixelQuintupletsInGPU.pixelIndices, maxPixelQuintuplets * sizeof(unsigned int));
    cudaMalloc(&pixelQuintupletsInGPU.T5Indices, maxPixelQuintuplets * sizeof(unsigned int));
    cudaMalloc(&pixelQuintupletsInGPU.nPixelQuintuplets, sizeof(unsigned int));
    cudaMalloc(&pixelQuintupletsInGPU.isDup, maxPixelQuintuplets * sizeof(bool));
    cudaMalloc(&pixelQuintupletsInGPU.score, maxPixelQuintuplets * sizeof(FPX));
    cudaMalloc(&pixelQuintupletsInGPU.eta  , maxPixelQuintuplets * sizeof(FPX));
    cudaMalloc(&pixelQuintupletsInGPU.phi  , maxPixelQuintuplets * sizeof(FPX));

#ifdef TRACK_EXTENSIONS
    cudaMalloc(&pixelQuintupletsInGPU.logicalLayers, maxPixelQuintuplets * 7 *sizeof(unsigned int));
    cudaMalloc(&pixelQuintupletsInGPU.hitIndices, maxPixelQuintuplets * 14 * sizeof(unsigned int));
    cudaMalloc(&pixelQuintupletsInGPU.lowerModuleIndices, maxPixelQuintuplets * 7 * sizeof(unsigned int));
    cudaMalloc(&pixelQuintupletsInGPU.pixelRadius, maxPixelQuintuplets * sizeof(FPX));
    cudaMalloc(&pixelQuintupletsInGPU.quintupletRadius, maxPixelQuintuplets * sizeof(FPX));
    cudaMalloc(&pixelQuintupletsInGPU.centerX, maxPixelQuintuplets * sizeof(FPX));
    cudaMalloc(&pixelQuintupletsInGPU.centerY, maxPixelQuintuplets * sizeof(FPX));
#endif
#endif
    cudaMemsetAsync(pixelQuintupletsInGPU.nPixelQuintuplets, 0, sizeof(unsigned int),stream);
  cudaStreamSynchronize(stream);
}

__device__ void SDL::rmPixelQuintupletToMemory(struct pixelQuintuplets& pixelQuintupletsInGPU, unsigned int pixelQuintupletIndex)
{

    pixelQuintupletsInGPU.isDup[pixelQuintupletIndex] = 1;
}
#ifdef CUT_VALUE_DEBUG
__device__ void SDL::addPixelQuintupletToMemory(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct quintuplets& quintupletsInGPU, struct pixelQuintuplets& pixelQuintupletsInGPU, unsigned int pixelIndex, unsigned int T5Index, unsigned int pixelQuintupletIndex, float& rzChiSquared, float& rPhiChiSquared, float& rPhiChiSquaredInwards, float score, float eta, float phi, float& pixelRadius, float& quintupletRadius,
        float& centerX, float& centerY)
#else
__device__ void SDL::addPixelQuintupletToMemory(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct quintuplets& quintupletsInGPU, struct pixelQuintuplets& pixelQuintupletsInGPU, unsigned int pixelIndex, unsigned int T5Index, unsigned int pixelQuintupletIndex, float score, float eta, float phi, float& pixelRadius, float& quintupletRadius, float& centerX, float& centerY)
#endif
{
    pixelQuintupletsInGPU.pixelIndices[pixelQuintupletIndex] = pixelIndex;
    pixelQuintupletsInGPU.T5Indices[pixelQuintupletIndex] = T5Index;
    pixelQuintupletsInGPU.isDup[pixelQuintupletIndex] = 0;
    pixelQuintupletsInGPU.score[pixelQuintupletIndex] = __F2H(score);
    pixelQuintupletsInGPU.eta[pixelQuintupletIndex]   = __F2H(eta);
    pixelQuintupletsInGPU.phi[pixelQuintupletIndex]   = __F2H(phi);

#ifdef TRACK_EXTENSIONS
    pixelQuintupletsInGPU.pixelRadius[pixelQuintupletIndex] = __F2H(pixelRadius);
    pixelQuintupletsInGPU.quintupletRadius[pixelQuintupletIndex] = __F2H(quintupletRadius);
    pixelQuintupletsInGPU.centerX[pixelQuintupletIndex] = __F2H(centerX);
    pixelQuintupletsInGPU.centerY[pixelQuintupletIndex] = __F2H(centerY);

    pixelQuintupletsInGPU.logicalLayers[7 * pixelQuintupletIndex] = 0;
    pixelQuintupletsInGPU.logicalLayers[7 * pixelQuintupletIndex + 1] = 0;
    pixelQuintupletsInGPU.logicalLayers[7 * pixelQuintupletIndex + 2] = quintupletsInGPU.logicalLayers[T5Index * 5];
    pixelQuintupletsInGPU.logicalLayers[7 * pixelQuintupletIndex + 3] = quintupletsInGPU.logicalLayers[T5Index * 5 + 1];
    pixelQuintupletsInGPU.logicalLayers[7 * pixelQuintupletIndex + 4] = quintupletsInGPU.logicalLayers[T5Index * 5 + 2];
    pixelQuintupletsInGPU.logicalLayers[7 * pixelQuintupletIndex + 5] = quintupletsInGPU.logicalLayers[T5Index * 5 + 3];
    pixelQuintupletsInGPU.logicalLayers[7 * pixelQuintupletIndex + 6] = quintupletsInGPU.logicalLayers[T5Index * 5 + 4];

    pixelQuintupletsInGPU.lowerModuleIndices[7 * pixelQuintupletIndex] = hitsInGPU.moduleIndices[segmentsInGPU.innerMiniDoubletAnchorHitIndices[pixelIndex]];
    pixelQuintupletsInGPU.lowerModuleIndices[7 * pixelQuintupletIndex + 1] = hitsInGPU.moduleIndices[segmentsInGPU.outerMiniDoubletAnchorHitIndices[pixelIndex]];
    pixelQuintupletsInGPU.lowerModuleIndices[7 * pixelQuintupletIndex + 2] = quintupletsInGPU.lowerModuleIndices[T5Index * 5];
    pixelQuintupletsInGPU.lowerModuleIndices[7 * pixelQuintupletIndex + 3] = quintupletsInGPU.lowerModuleIndices[T5Index * 5 + 1];
    pixelQuintupletsInGPU.lowerModuleIndices[7 * pixelQuintupletIndex + 4] = quintupletsInGPU.lowerModuleIndices[T5Index * 5 + 2];
    pixelQuintupletsInGPU.lowerModuleIndices[7 * pixelQuintupletIndex + 5] = quintupletsInGPU.lowerModuleIndices[T5Index * 5 + 3];
    pixelQuintupletsInGPU.lowerModuleIndices[7 * pixelQuintupletIndex + 6] = quintupletsInGPU.lowerModuleIndices[T5Index * 5 + 4];

    unsigned int pixelInnerMD = segmentsInGPU.mdIndices[2 * pixelIndex];
    unsigned int pixelOuterMD = segmentsInGPU.mdIndices[2 * pixelIndex + 1];

    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex] = mdsInGPU.hitIndices[2 * pixelInnerMD];
    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 1] = mdsInGPU.hitIndices[2 * pixelInnerMD + 1];
    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 2] = mdsInGPU.hitIndices[2 * pixelOuterMD];
    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 3] = mdsInGPU.hitIndices[2 * pixelOuterMD + 1];

    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 4] = quintupletsInGPU.hitIndices[10 * T5Index];
    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 5] = quintupletsInGPU.hitIndices[10 * T5Index + 1];
    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 6] = quintupletsInGPU.hitIndices[10 * T5Index + 2];
    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 7] = quintupletsInGPU.hitIndices[10 * T5Index + 3];
    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 8] = quintupletsInGPU.hitIndices[10 * T5Index + 4];
    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 9] = quintupletsInGPU.hitIndices[10 * T5Index + 5];
    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 10] = quintupletsInGPU.hitIndices[10 * T5Index + 6];
    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 11] = quintupletsInGPU.hitIndices[10 * T5Index + 7];
    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 12] = quintupletsInGPU.hitIndices[10 * T5Index + 8];
    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 13] = quintupletsInGPU.hitIndices[10 * T5Index + 9];
#endif
        
#ifdef CUT_VALUE_DEBUG
    pixelQuintupletsInGPU.rzChiSquared[pixelQuintupletIndex] = rzChiSquared;
    pixelQuintupletsInGPU.rPhiChiSquared[pixelQuintupletIndex] = rPhiChiSquared;
    pixelQuintupletsInGPU.rPhiChiSquaredInwards[pixelQuintupletIndex] = rPhiChiSquaredInwards;
#endif
}

__device__ bool SDL::runPixelQuintupletDefaultAlgo(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct triplets& tripletsInGPU, struct quintuplets& quintupletsInGPU, unsigned int& pixelSegmentIndex, unsigned int& quintupletIndex, float& rzChiSquared, float& rPhiChiSquared, float& rPhiChiSquaredInwards, float& pixelRadius, float& quintupletRadius, float& centerX, float& centerY)
{
    bool pass = true;
    
    unsigned int pixelModuleIndex = segmentsInGPU.innerLowerModuleIndices[pixelSegmentIndex];

    unsigned int pixelSegmentArrayIndex = pixelSegmentIndex - (600 * pixelModuleIndex);

    unsigned int T5InnerT3Index = quintupletsInGPU.tripletIndices[2 * quintupletIndex];
    unsigned int T5OuterT3Index = quintupletsInGPU.tripletIndices[2 * quintupletIndex + 1];

    unsigned int firstSegmentIndex = tripletsInGPU.segmentIndices[2 * T5InnerT3Index];
    unsigned int secondSegmentIndex = tripletsInGPU.segmentIndices[2 * T5InnerT3Index + 1];
    unsigned int thirdSegmentIndex = tripletsInGPU.segmentIndices[2 * T5OuterT3Index];
    unsigned int fourthSegmentIndex = tripletsInGPU.segmentIndices[2 * T5OuterT3Index + 1];

    unsigned int pixelInnerMDIndex = segmentsInGPU.mdIndices[2 * pixelSegmentIndex];
    unsigned int pixelOuterMDIndex = segmentsInGPU.mdIndices[2 * pixelSegmentIndex + 1];

    unsigned int pixelAnchorHitIndex1 = mdsInGPU.anchorHitIndices[pixelInnerMDIndex];
    unsigned int pixelNonAnchorHitIndex1 = mdsInGPU.outerHitIndices[pixelInnerMDIndex];
    unsigned int pixelAnchorHitIndex2 = mdsInGPU.anchorHitIndices[pixelOuterMDIndex];
    unsigned int pixelNonAnchorHitIndex2 = mdsInGPU.outerHitIndices[pixelOuterMDIndex];

    unsigned int anchorHitIndex1 = segmentsInGPU.innerMiniDoubletAnchorHitIndices[firstSegmentIndex];
    unsigned int anchorHitIndex2 = segmentsInGPU.outerMiniDoubletAnchorHitIndices[firstSegmentIndex]; //same as second segment inner MD anchorhit index
    unsigned int anchorHitIndex3 = segmentsInGPU.outerMiniDoubletAnchorHitIndices[secondSegmentIndex]; //same as third segment inner MD anchor hit index

    unsigned int anchorHitIndex4 = segmentsInGPU.outerMiniDoubletAnchorHitIndices[thirdSegmentIndex]; //same as fourth segment inner MD anchor hit index
    unsigned int anchorHitIndex5 = segmentsInGPU.outerMiniDoubletAnchorHitIndices[fourthSegmentIndex];

    unsigned int lowerModuleIndex1 = quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex];
    unsigned int lowerModuleIndex2 = quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex + 1];
    unsigned int lowerModuleIndex3 = quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex + 2];
    unsigned int lowerModuleIndex4 = quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex + 3];
    unsigned int lowerModuleIndex5 = quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex + 4];

    unsigned int lowerModuleIndices[] = {lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4, lowerModuleIndex5};
    unsigned int anchorHits[] = {anchorHitIndex1, anchorHitIndex2, anchorHitIndex3, anchorHitIndex4, anchorHitIndex5};
    unsigned int pixelHits[] = {pixelAnchorHitIndex1, pixelAnchorHitIndex2};
    
    float pixelRadiusTemp, pixelRadiusError, tripletRadius, rPhiChiSquaredTemp, rzChiSquaredTemp, rPhiChiSquaredInwardsTemp, centerXTemp, centerYTemp;

    pass = pass & runPixelTripletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, tripletsInGPU, pixelSegmentIndex, T5InnerT3Index, pixelRadiusTemp, pixelRadiusError, tripletRadius, centerXTemp, centerYTemp, rzChiSquaredTemp, rPhiChiSquaredTemp, rPhiChiSquaredInwardsTemp, false);

    rzChiSquared = computePT5RZChiSquared(modulesInGPU, hitsInGPU, pixelAnchorHitIndex1, pixelAnchorHitIndex2, anchorHits, lowerModuleIndices);

    rPhiChiSquared = computePT5RPhiChiSquared(modulesInGPU, hitsInGPU, segmentsInGPU, pixelSegmentArrayIndex, anchorHits, lowerModuleIndices, centerX, centerY);
    float T5CenterX, T5CenterY;
    rPhiChiSquaredInwards = computePT5RPhiChiSquaredInwards(modulesInGPU, hitsInGPU, quintupletsInGPU, quintupletIndex, pixelHits, T5CenterX, T5CenterY);

    pixelRadius = segmentsInGPU.circleRadius[pixelSegmentArrayIndex];
    quintupletRadius = quintupletsInGPU.regressionRadius[quintupletIndex];

    if(segmentsInGPU.circleRadius[pixelSegmentArrayIndex] < 5.0f/(2.f * k2Rinv1GeVf))
    {
        pass = pass & passPT5RZChiSquaredCuts(modulesInGPU, lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4, lowerModuleIndex5, rzChiSquared);

        pass = pass & passPT5RPhiChiSquaredCuts(modulesInGPU, lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4, lowerModuleIndex5, rPhiChiSquared);
    }

    if(quintupletsInGPU.regressionRadius[quintupletIndex] < 5.0f/(2.f * k2Rinv1GeVf))
    {
        pass = pass & passPT5RPhiChiSquaredInwardsCuts(modulesInGPU, lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4, lowerModuleIndex5, rPhiChiSquaredInwards);
    }
   
    //trusting the T5 regression center to also be a good estimate..
    centerX = (centerX + T5CenterX)/2;
    centerY = (centerY + T5CenterY)/2;

    //other cuts will be filled here!
    return pass;
}

__device__ bool SDL::passPT5RPhiChiSquaredCuts(struct modules& modulesInGPU, unsigned int lowerModuleIndex1, unsigned int lowerModuleIndex2, unsigned int lowerModuleIndex3, unsigned int lowerModuleIndex4, unsigned int lowerModuleIndex5, float rPhiChiSquared)
{
    const int layer1 = modulesInGPU.layers[lowerModuleIndex1] + 6 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex1] == SDL::TwoS);
    const int layer2 = modulesInGPU.layers[lowerModuleIndex2] + 6 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex2] == SDL::TwoS);
    const int layer3 = modulesInGPU.layers[lowerModuleIndex3] + 6 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex3] == SDL::TwoS);
    const int layer4 = modulesInGPU.layers[lowerModuleIndex4] + 6 * (modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex4] == SDL::TwoS);
    const int layer5 = modulesInGPU.layers[lowerModuleIndex5] + 6 * (modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex5] == SDL::TwoS);

    if(layer1 == 1 and layer2 == 2 and layer3 == 3)
    {
        if(layer4 == 12 and layer5 == 13)
        {
            return rPhiChiSquared < 48.921f;
        }
        else if(layer4 == 4 and layer5 == 12)
        {
            return rPhiChiSquared < 97.948f;
        }
        else if(layer4 == 4 and layer5 == 5)
        {
            return rPhiChiSquared < 129.3f;
        }
        else if(layer4 == 7 and layer5 == 13)
        {
            return rPhiChiSquared < 56.21f;
        }
        else if(layer4 == 7 and layer5 == 8)
        {
            return rPhiChiSquared < 74.198f;
        }
    }
    else if(layer1 == 1 and layer2 == 2 and layer3 == 7)
    {
        if(layer4 == 13 and layer5 == 14)
        {
            return rPhiChiSquared < 21.265f;
        }
        else if(layer4 == 8 and layer5 == 14)
        {
            return rPhiChiSquared < 37.058f;
        }
        else if(layer4 == 8 and layer5 == 9)
        {
            return rPhiChiSquared < 42.578f;
        }
    }
    else if(layer1 == 1 and layer2 == 7 and layer3 == 8)
    {
        if(layer4 == 9 and layer5 == 10)
        {
            return rPhiChiSquared < 32.253f;
        }
        else if(layer4 == 9 and layer5 == 15)
        {
            return rPhiChiSquared < 37.058f;
        }
    }
    else if(layer1 == 2 and layer2 == 3 and layer3 == 4)
    {
        if(layer4 == 12 and layer5 == 13)
        {
            return rPhiChiSquared < 97.947f;
        }
        else if(layer4 == 5 and layer5 == 12)
        {
            return rPhiChiSquared < 129.3f;
        }
        else if(layer4 == 5 and layer5 == 6)
        {
            return rPhiChiSquared < 170.68f;
        }
    }
    else if(layer1 == 2 and layer2 == 3 and layer3 == 7)
    {
        if(layer4 == 13 and layer5 == 14)
        {   
            return rPhiChiSquared < 48.92f;
        }
        else if(layer4 == 8 and layer5 == 14)
        {
            return rPhiChiSquared < 74.2f;
        }
    }
    else if(layer1 == 2 and layer2 == 7 and layer3 == 8)
    {
        if(layer4 == 14 and layer5 == 15)
        {
            return rPhiChiSquared < 42.58f;
        }
        else if(layer4 == 9 and layer5 == 10)
        {
            return rPhiChiSquared < 37.06f;
        }
        else if(layer4 == 9 and layer5 == 15)
        {
            return rPhiChiSquared < 48.92f;
        }
    }
    else if(layer1 == 3 and layer2 == 7 and layer3 == 8 and layer4 == 14 and layer5 == 15)
    {
        return rPhiChiSquared < 85.25f;
    }
    else if(layer1 == 7 and layer2 == 8 and layer3 == 9)
    {
        if(layer4 == 10 and layer5 == 11)
        {
            return rPhiChiSquared < 42.58f;
        }
        else if(layer4 == 10 and layer5 == 16)
        {
            return rPhiChiSquared < 37.06f;
        }
        else if(layer4 == 15 and layer5 == 16)
        {
            return rPhiChiSquared < 37.06f;
        }
    }
    return true;
}



__device__ bool SDL::passPT5RPhiChiSquaredInwardsCuts(struct modules& modulesInGPU, unsigned int lowerModuleIndex1, unsigned int lowerModuleIndex2, unsigned int lowerModuleIndex3, unsigned int lowerModuleIndex4, unsigned int lowerModuleIndex5, float rPhiChiSquared)
{
    const int layer1 = modulesInGPU.layers[lowerModuleIndex1] + 6 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex1] == SDL::TwoS);
    const int layer2 = modulesInGPU.layers[lowerModuleIndex2] + 6 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex2] == SDL::TwoS);
    const int layer3 = modulesInGPU.layers[lowerModuleIndex3] + 6 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex3] == SDL::TwoS);
    const int layer4 = modulesInGPU.layers[lowerModuleIndex4] + 6 * (modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex4] == SDL::TwoS);
    const int layer5 = modulesInGPU.layers[lowerModuleIndex5] + 6 * (modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex5] == SDL::TwoS);

    if(layer1 == 1 and layer2 == 2 and layer3 == 3)
    {
        if(layer4 == 12 and layer5 == 13)
        {
            return rPhiChiSquared < 451.141f;
        }
        else if(layer4 == 4 and layer5 == 12)
        {
            return rPhiChiSquared < 786.173f;
        }
        else if(layer4 == 4 and layer5 == 5)
        {
            return rPhiChiSquared < 595.545f;
        }
        else if(layer4 == 7 and layer5 == 13)
        {
            return rPhiChiSquared < 581.339f;
        }
        else if(layer4 == 7 and layer5 == 8)
        {
            return rPhiChiSquared < 112.537f;
        }
    }
    else if(layer1 == 1 and layer2 == 2 and layer3 == 7)
    {
        if(layer4 == 13 and layer5 == 14)
        {
            return rPhiChiSquared < 225.322f;
        }
        else if(layer4 == 8 and layer5 == 14)
        {
            return rPhiChiSquared < 1192.402f;
        }
        else if(layer4 == 8 and layer5 == 9)
        {
            return rPhiChiSquared < 786.173f;
        }
    }
    else if(layer1 == 1 and layer2 == 7 and layer3 == 8)
    {
        if(layer4 == 9 and layer5 == 10)
        {
            return rPhiChiSquared < 1037.817f;
        }
        else if(layer4 == 9 and layer5 == 15)
        {
            return rPhiChiSquared < 1808.536f;
        }
    }
    else if(layer1 == 2 and layer2 == 3 and layer3 == 4)
    {
        if(layer4 == 12 and layer5 == 13)
        {
            return rPhiChiSquared < 684.253f;
        }
        else if(layer4 == 5 and layer5 == 12)
        {
            return rPhiChiSquared < 684.253f;
        }
        else if(layer4 == 5 and layer5 == 6)
        {
            return rPhiChiSquared < 684.253f;
        }
    }
    else if(layer1 == 2 and layer2 == 3 and layer3 == 7)
    {
        if(layer4 == 13 and layer5 == 14)
        {   
            return rPhiChiSquared < 451.141f;
        }
        else if(layer4 == 8 and layer5 == 14)
        {
            return rPhiChiSquared < 518.34f;
        }
    }
    else if(layer1 == 2 and layer2 == 7 and layer3 == 8)
    {
        if(layer4 == 14 and layer5 == 15)
        {
            return rPhiChiSquared < 2077.92f;
        }
        else if(layer4 == 9 and layer5 == 10)
        {
            return rPhiChiSquared < 74.20f;
        }
        else if(layer4 == 9 and layer5 == 15)
        {
            return rPhiChiSquared < 1808.536f;
        }
    }
    else if(layer1 == 3 and layer2 == 7 and layer3 == 8 and layer4 == 14 and layer5 == 15)
    {
        return rPhiChiSquared < 786.173f;
    }
    else if(layer1 == 7 and layer2 == 8 and layer3 == 9)
    {
        if(layer4 == 10 and layer5 == 11)
        {
            return rPhiChiSquared < 1574.076f;
        }
        else if(layer4 == 10 and layer5 == 16)
        {
            return rPhiChiSquared < 5492.11f;
        }
        else if(layer4 == 15 and layer5 == 16)
        {
            return rPhiChiSquared < 2743.037f;
        }
    }
    return true;
}

__device__ float SDL::computePT5RPhiChiSquared(struct modules& modulesInGPU, struct hits& hitsInGPU, struct segments& segmentsInGPU, unsigned int pixelSegmentArrayIndex, unsigned int* anchorHits, unsigned int* lowerModuleIndices, float& g, float& f)
{
    /*
       Compute circle parameters from 3 pixel hits, and then use them to compute the chi squared for the outer hits
    */

    g = segmentsInGPU.circleCenterX[pixelSegmentArrayIndex];
    f = segmentsInGPU.circleCenterY[pixelSegmentArrayIndex];
    float radius = segmentsInGPU.circleRadius[pixelSegmentArrayIndex];
    float delta1[5], delta2[5], slopes[5];
    bool isFlat[5];
    float xs[5];
    float ys[5];
    float chiSquared = 0;
    for(size_t i = 0; i < 5; i++)
    {
        xs[i] = hitsInGPU.xs[anchorHits[i]];
        ys[i] = hitsInGPU.ys[anchorHits[i]];
    }

    computeSigmasForRegression_pT5(modulesInGPU, lowerModuleIndices, delta1, delta2, slopes, isFlat);
    chiSquared = computeChiSquared(5, xs, ys, delta1, delta2, slopes, isFlat, g, f, radius);

    return chiSquared;
}

__device__ bool SDL::passPT5RZChiSquaredCuts(struct modules& modulesInGPU, unsigned int lowerModuleIndex1, unsigned int lowerModuleIndex2, unsigned int lowerModuleIndex3, unsigned int lowerModuleIndex4, unsigned int lowerModuleIndex5, float& rzChiSquared)
{
    const int layer1 = modulesInGPU.layers[lowerModuleIndex1] + 6 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex1] == SDL::TwoS);
    const int layer2 = modulesInGPU.layers[lowerModuleIndex2] + 6 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex2] == SDL::TwoS);
    const int layer3 = modulesInGPU.layers[lowerModuleIndex3] + 6 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex3] == SDL::TwoS);
    const int layer4 = modulesInGPU.layers[lowerModuleIndex4] + 6 * (modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex4] == SDL::TwoS);
    const int layer5 = modulesInGPU.layers[lowerModuleIndex5] + 6 * (modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex5] == SDL::TwoS);

    
    if(layer1 == 1 and layer2 == 2 and layer3 == 3)
    {
        if(layer4 == 12 and layer5 == 13)
        {
            return rzChiSquared < 451.141f;
        }
        else if(layer4 == 4 and layer5 == 12)
        {
            return rzChiSquared < 392.654f;
        }
        else if(layer4 == 4 and layer5 == 5)
        {
            return rzChiSquared < 225.322f;
        }
        else if(layer4 == 7 and layer5 == 13)
        {
            return rzChiSquared < 595.546f;
        }
        else if(layer4 == 7 and layer5 == 8)
        {
            return rzChiSquared < 196.111f;
        }
    }
    else if(layer1 == 1 and layer2 == 2 and layer3 == 7)
    {
        if(layer4 == 13 and layer5 == 14)
        {
            return rzChiSquared < 297.446f;
        }
        else if(layer4 == 8 and layer5 == 14)
        {   
            return rzChiSquared < 451.141f;
        }
        else if(layer4 == 8 and layer5 == 9)
        {
            return rzChiSquared < 518.339f;
        }
    }
    else if(layer1 == 1 and layer2 == 7 and layer3 == 8)
    {
        if(layer4 == 9 and layer5 == 10)
        {
            return rzChiSquared < 341.75f;
        }
        else if(layer4 == 9 and layer5 == 15)
        {
            return rzChiSquared < 341.75f;
        }
    }
    else if(layer1 == 2 and layer2 == 3 and layer3 == 4)
    {
        if(layer4 == 12 and layer5 == 13)
        {
            return rzChiSquared < 392.655f;
        }
        else if(layer4 == 5 and layer5 == 12)
        {
            return rzChiSquared < 341.75f;
        }
        else if(layer4 == 5 and layer5 == 6)
        {
            return rzChiSquared < 112.537f;
        }
    }
    else if(layer1 == 2 and layer2 == 3 and layer4 == 7)
    {
        if(layer4 == 13 and layer5 == 14)
        {
            return rzChiSquared < 595.545f;
        }
        else if(layer4 == 8 and layer5 == 14)
        {
            return rzChiSquared < 74.198f;
        }
    }
    else if(layer1 == 2 and layer2 == 7 and layer3 == 8)
    {
        if(layer4 == 14 and layer5 == 15)
        {
            return rzChiSquared < 518.339f;
        }
        else if(layer4 == 9 and layer5 == 10)
        {
            return rzChiSquared < 8.046f;
        }
        else if(layer4 == 9 and layer5 == 15)
        {
            return rzChiSquared < 451.141f;
        }
    }
    else if(layer1 == 3 and layer2 == 7 and layer3 == 8 and layer4 == 14 and layer5 == 15)
    {
        return rzChiSquared < 56.207f;
    }
    else if(layer1 == 7 and layer2 == 8 and layer3 == 9)
    {
        if(layer4 == 10 and layer5 == 11)
        {
            return rzChiSquared < 64.578f;
        }
        else if(layer4 == 10 and layer5 == 16)
        {
            return rzChiSquared < 85.250f;
        }
        else if(layer4 == 15 and layer5 == 16)
        {
            return rzChiSquared < 85.250f;
        }
    }
    return true;
}

__device__ float SDL::computePT5RPhiChiSquaredInwards(struct modules& modulesInGPU, struct hits& hitsInGPU, struct quintuplets& quintupletsInGPU, unsigned int quintupletIndex, unsigned int* pixelHits, float& g, float& f)
{
    /*Using the computed regression center and radius, compute the chi squared for the pixels*/
    g = quintupletsInGPU.regressionG[quintupletIndex];
    f = quintupletsInGPU.regressionF[quintupletIndex];
    float r = quintupletsInGPU.regressionRadius[quintupletIndex];
    float x, y;
    float chiSquared = 0;   
    for(size_t i = 0; i < 2; i++)
    {
        x = hitsInGPU.xs[pixelHits[i]];
        y = hitsInGPU.ys[pixelHits[i]];
        float residual = (x - g) * (x -g) + (y - f) * (y - f) - r * r;
        chiSquared += residual * residual;
    }
    //chiSquared /= 2;
    chiSquared *= 0.5f;
    return chiSquared;
}

__device__ float SDL::computePT5RZChiSquared(struct modules& modulesInGPU, struct hits& hitsInGPU, unsigned int& pixelAnchorHitIndex1, unsigned int& pixelAnchorHitIndex2, unsigned int* anchorHits, unsigned int* lowerModuleIndices)
{
    //use the two anchor hits of the pixel segment to compute the slope
    //then compute the pseudo chi squared of the five outer hits

    float& rtPix1 = hitsInGPU.rts[pixelAnchorHitIndex1];
    float& rtPix2 = hitsInGPU.rts[pixelAnchorHitIndex2];
    float& zPix1 = hitsInGPU.zs[pixelAnchorHitIndex1];
    float& zPix2 = hitsInGPU.zs[pixelAnchorHitIndex2];
    float slope = (zPix2 - zPix1)/(rtPix2 - rtPix1);
    float rtAnchor, zAnchor;
    float residual = 0;
    float error = 0;
    //hardcoded array indices!!!
    float RMSE = 0;
    float drdz;
    for(size_t i = 0; i < 5; i++)
    {
        unsigned int& anchorHitIndex = anchorHits[i];
        unsigned int& lowerModuleIndex = lowerModuleIndices[i];
        rtAnchor = hitsInGPU.rts[anchorHitIndex];
        zAnchor = hitsInGPU.zs[anchorHitIndex];

        const int moduleType = modulesInGPU.moduleType[lowerModuleIndex];
        const int moduleSide = modulesInGPU.sides[lowerModuleIndex];
        const int moduleLayerType = modulesInGPU.moduleLayerType[lowerModuleIndex];
        const int layer = modulesInGPU.layers[lowerModuleIndex] + 6 * (modulesInGPU.subdets[lowerModuleIndex] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex] == SDL::TwoS);
        
        residual = (layer <= 6) ?  (zAnchor - zPix1) - slope * (rtAnchor - rtPix1) : (rtAnchor - rtPix1) - (zAnchor - zPix1)/slope;
        
        //PS Modules
        if(moduleType == 0)
        {
            error = 0.15f;
        }
        else //2S modules
        {
            error = 5.0f;
        }

        //special dispensation to tilted PS modules!
        if(moduleType == 0 and layer <= 6 and moduleSide != Center)
        {
            if(moduleLayerType == Strip)
            {
                drdz = modulesInGPU.drdzs[lowerModuleIndex];
            }
            else
            {
                drdz = modulesInGPU.drdzs[modulesInGPU.partnerModuleIndex(lowerModuleIndex)];
            }

            //error *= 1.f/sqrtf(1.f + drdz * drdz);
            error /= sqrtf(1.f + drdz * drdz);
        }
        RMSE += (residual * residual)/(error * error);
    }

    RMSE = sqrtf(0.2f * RMSE);
    return RMSE;
}
__device__ void SDL::computeSigmasForRegression_pT5(SDL::modules& modulesInGPU, const unsigned int* lowerModuleIndices, float* delta1, float* delta2, float* slopes, bool* isFlat, int nPoints, bool anchorHits)
{
   /*bool anchorHits required to deal with a weird edge case wherein
     the hits ultimately used in the regression are anchor hits, but the
     lower modules need not all be Pixel Modules (in case of PS). Similarly,
     when we compute the chi squared for the non-anchor hits, the "partner module"
     need not always be a PS strip module, but all non-anchor hits sit on strip
     modules.
    */
    ModuleType moduleType;
    short moduleSubdet, moduleSide;
    ModuleLayerType moduleLayerType;
    float drdz;
    float inv1 = 0.01f/0.009f;
    float inv2 = 0.15f/0.009f;
    float inv3 = 2.4f/0.009f;
    for(size_t i=0; i<nPoints; i++)
    {
        moduleType = modulesInGPU.moduleType[lowerModuleIndices[i]];
        moduleSubdet = modulesInGPU.subdets[lowerModuleIndices[i]];
        moduleSide = modulesInGPU.sides[lowerModuleIndices[i]];
        moduleLayerType = modulesInGPU.moduleLayerType[lowerModuleIndices[i]];
        //category 1 - barrel PS flat
        if(moduleSubdet == Barrel and moduleType == PS and moduleSide == Center)
        {
            //delta1[i] = 0.01;
            //delta2[i] = 0.01;
            delta1[i] = inv1;//1.1111f;//0.01;
            delta2[i] = inv1;//1.1111f;//0.01;
            slopes[i] = -999.f;
            isFlat[i] = true;
        }

        //category 2 - barrel 2S
        else if(moduleSubdet == Barrel and moduleType == TwoS)
        {
            //delta1[i] = 0.009;
            //delta2[i] = 0.009;
            delta1[i] = 1.f;//0.009;
            delta2[i] = 1.f;//0.009;
            slopes[i] = -999.f;
            isFlat[i] = true;
        }

        //category 3 - barrel PS tilted
        else if(moduleSubdet == Barrel and moduleType == PS and moduleSide != Center)
        {

            //get drdz
            if(moduleLayerType == Strip)
            {
                drdz = modulesInGPU.drdzs[lowerModuleIndices[i]];
                slopes[i] = modulesInGPU.slopes[lowerModuleIndices[i]];
            }
            else
            {
                drdz = modulesInGPU.drdzs[modulesInGPU.partnerModuleIndex(lowerModuleIndices[i])];
                slopes[i] = modulesInGPU.slopes[modulesInGPU.partnerModuleIndex(lowerModuleIndices[i])];
            }

            //delta1[i] = 0.01;
            delta1[i] = inv1;//1.1111f;//0.01;
            isFlat[i] = false;

            if(anchorHits)
            {
                //delta2[i] = (0.15f * drdz/sqrtf(1 + drdz * drdz));
                delta2[i] = (inv2 * drdz/sqrtf(1 + drdz * drdz));
                //delta2[i] = (inv2 * drdz*rsqrt(1 + drdz * drdz));
            }
            else
            {
                //delta2[i] = (2.4f * drdz/sqrtf(1 + drdz * drdz));
                delta2[i] = (inv3 * drdz/sqrtf(1 + drdz * drdz));
                //delta2[i] = (inv3 * drdz*rsqrt(1 + drdz * drdz));
            }
        }
        //category 4 - endcap PS
        else if(moduleSubdet == Endcap and moduleType == PS)
        {
            delta1[i] = inv1;//1.1111f;//0.01;
            //delta1[i] = 0.01;
            if(moduleLayerType == Strip)
            {
                slopes[i] = modulesInGPU.slopes[lowerModuleIndices[i]];
            }
            else
            {
                slopes[i] = modulesInGPU.slopes[modulesInGPU.partnerModuleIndex(lowerModuleIndices[i])];

            }
            isFlat[i] = false;

            /*despite the type of the module layer of the lower module index,
            all anchor hits are on the pixel side and all non-anchor hits are
            on the strip side!*/
            if(anchorHits)
            {
                delta2[i] = inv2;//16.6666f;//0.15f;
                //delta2[i] = 0.15f;
            }
            else
            {
                //delta2[i] = 2.4f;
                delta2[i] = inv3;//266.666f;//2.4f;
            }
        }

        //category 5 - endcap 2S
        else if(moduleSubdet == Endcap and moduleType == TwoS)
        {
            //delta1[i] = 0.009;
            //delta2[i] = 5.f;
            delta1[i] = 1.f;//0.009;
            delta2[i] = 500.f*inv1;//555.5555f;//5.f;
            slopes[i] = modulesInGPU.slopes[lowerModuleIndices[i]];
            isFlat[i] = false;
        }
        else
        {
            printf("ERROR!!!!! I SHOULDN'T BE HERE!!!! subdet = %d, type = %d, side = %d\n", moduleSubdet, moduleType, moduleSide);
        }
    }
    //divide everyone by the smallest possible values of delta1 and delta2
//    for(size_t i = 0; i < 5; i++)
//    {
//        delta1[i] /= 0.009;
//        delta2[i] /= 0.009;
//    }
}
