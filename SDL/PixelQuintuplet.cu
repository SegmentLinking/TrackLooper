//#ifdef __CUDACC__
//#define CUDA_CONST_VAR __device__
//#endif
//# include "PixelQuintuplet.cuh"
//#include "allocate.h"
//#include "Kernels.cuh"
//
//SDL::pixelQuintuplets::pixelQuintuplets()
//{
//    pixelIndices = nullptr;
//    T5Indices = nullptr;
//    nPixelQuintuplets = nullptr;
//    totOccupancyPixelQuintuplets = nullptr;
//    isDup = nullptr;
//    score = nullptr;
//    pixelRadius = nullptr;
//    quintupletRadius = nullptr;
//    centerX = nullptr;
//    centerY = nullptr;
//    logicalLayers = nullptr;
//    hitIndices = nullptr;
//    lowerModuleIndices = nullptr;
//}
//
//SDL::pixelQuintuplets::~pixelQuintuplets()
//{
//}
//
//void SDL::pixelQuintuplets::freeMemoryCache()
//{
//#ifdef Explicit_PT5
//    int dev;
//    cudaGetDevice(&dev);
//    cms::cuda::free_device(dev,pixelIndices);
//    cms::cuda::free_device(dev,T5Indices);
//    cms::cuda::free_device(dev,nPixelQuintuplets);
//    cms::cuda::free_device(dev,totOccupancyPixelQuintuplets);
//    cms::cuda::free_device(dev,isDup);
//    cms::cuda::free_device(dev,score);
//    cms::cuda::free_device(dev,eta);
//    cms::cuda::free_device(dev,phi);
//    cms::cuda::free_device(dev, hitIndices);
//    cms::cuda::free_device(dev, logicalLayers);
//    cms::cuda::free_device(dev, lowerModuleIndices);
//    cms::cuda::free_device(dev, centerX);
//    cms::cuda::free_device(dev, centerY);
//    cms::cuda::free_device(dev, pixelRadius);
//    cms::cuda::free_device(dev, quintupletRadius);
//#else
//    cms::cuda::free_managed(pixelIndices);
//    cms::cuda::free_managed(T5Indices);
//    cms::cuda::free_managed(nPixelQuintuplets);
//    cms::cuda::free_managed(totOccupancyPixelQuintuplets);
//    cms::cuda::free_managed(isDup);
//    cms::cuda::free_managed(score);
//    cms::cuda::free_managed(eta);
//    cms::cuda::free_managed(phi);
//    cms::cuda::free_managed(hitIndices);
//    cms::cuda::free_managed(logicalLayers);
//    cms::cuda::free_managed(lowerModuleIndices);
//    cms::cuda::free_managed(centerX);
//    cms::cuda::free_managed(centerY);
//    cms::cuda::free_managed(pixelRadius);
//    cms::cuda::free_managed(quintupletRadius);
//#endif
//}
//void SDL::pixelQuintuplets::freeMemory(cudaStream_t stream)
//{
//    //cudaFreeAsync(pixelIndices,stream);
//    //cudaFreeAsync(T5Indices,stream);
//    //cudaFreeAsync(nPixelQuintuplets,stream);
//    //cudaFreeAsync(isDup,stream);
//    //cudaFreeAsync(score,stream);
//    //cudaFreeAsync(eta,stream);
//    //cudaFreeAsync(phi,stream);
//    cudaFree(pixelIndices);
//    cudaFree(T5Indices);
//    cudaFree(nPixelQuintuplets);
//    cudaFree(totOccupancyPixelQuintuplets);
//    cudaFree(isDup);
//    cudaFree(score);
//    cudaFree(eta);
//    cudaFree(phi);
//
//    cudaFree(logicalLayers);
//    cudaFree(hitIndices);
//    cudaFree(lowerModuleIndices);
//    cudaFree(pixelRadius);
//    cudaFree(quintupletRadius);
//    cudaFree(centerX);
//    cudaFree(centerY);
//#ifdef CUT_VALUE_DEBUG
//    cudaFree(rzChiSquared);
//    cudaFree(rPhiChiSquared);
//    cudaFree(rPhiChiSquaredInwards);
//#endif
//cudaStreamSynchronize(stream);
//}
//
//void SDL::pixelQuintuplets::resetMemory(unsigned int maxPixelQuintuplets,cudaStream_t stream)
//{
//    cudaMemsetAsync(pixelIndices,0, maxPixelQuintuplets * sizeof(unsigned int),stream);
//    cudaMemsetAsync(T5Indices,0, maxPixelQuintuplets * sizeof(unsigned int),stream);
//    cudaMemsetAsync(nPixelQuintuplets,0, sizeof(unsigned int),stream);
//    cudaMemsetAsync(totOccupancyPixelQuintuplets,0, sizeof(unsigned int),stream);
//    cudaMemsetAsync(isDup,0, maxPixelQuintuplets * sizeof(bool),stream);
//    cudaMemsetAsync(score,0, maxPixelQuintuplets * sizeof(FPX),stream);
//    cudaMemsetAsync(eta , 0, maxPixelQuintuplets * sizeof(FPX),stream);
//    cudaMemsetAsync(phi , 0, maxPixelQuintuplets * sizeof(FPX),stream);
//}
//void SDL::createPixelQuintupletsInUnifiedMemory(struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, unsigned int maxPixelQuintuplets,cudaStream_t stream)
//{
//#ifdef CACHE_ALLOC
//    pixelQuintupletsInGPU.pixelIndices        = (unsigned int*)cms::cuda::allocate_managed(maxPixelQuintuplets * sizeof(unsigned int),stream);
//    pixelQuintupletsInGPU.T5Indices           = (unsigned int*)cms::cuda::allocate_managed(maxPixelQuintuplets * sizeof(unsigned int),stream);
//    pixelQuintupletsInGPU.nPixelQuintuplets   = (unsigned int*)cms::cuda::allocate_managed(sizeof(unsigned int),stream);
//    pixelQuintupletsInGPU.totOccupancyPixelQuintuplets   = (unsigned int*)cms::cuda::allocate_managed(sizeof(unsigned int),stream);
//    pixelQuintupletsInGPU.isDup               = (bool*)cms::cuda::allocate_managed(maxPixelQuintuplets * sizeof(bool),stream);
//    pixelQuintupletsInGPU.score               = (FPX*)cms::cuda::allocate_managed(maxPixelQuintuplets * sizeof(FPX),stream);
//    pixelQuintupletsInGPU.eta                 = (FPX*)cms::cuda::allocate_managed(maxPixelQuintuplets * sizeof(FPX),stream);
//    pixelQuintupletsInGPU.phi                 = (FPX*)cms::cuda::allocate_managed(maxPixelQuintuplets * sizeof(FPX),stream);
//    pixelQuintupletsInGPU.hitIndices = (unsigned int*)cms::cuda::allocate_managed(maxPixelQuintuplets * 14 * sizeof(unsigned int), stream);
//    pixelQuintupletsInGPU.logicalLayers = (uint8_t*)cms::cuda::allocate_managed(maxPixelQuintuplets * 7 * sizeof(uint8_t), stream);
//    pixelQuintupletsInGPU.lowerModuleIndices = (uint16_t*)cms::cuda::allocate_managed(maxPixelQuintuplets * 7 * sizeof(uint16_t), stream);
//    pixelQuintupletsInGPU.centerX          = (FPX*)cms::cuda::allocate_managed(maxPixelQuintuplets * sizeof(FPX), stream);
//    pixelQuintupletsInGPU.centerY          = (FPX*)cms::cuda::allocate_managed(maxPixelQuintuplets * sizeof(FPX), stream);
//    pixelQuintupletsInGPU.pixelRadius      = (FPX*)cms::cuda::allocate_managed(maxPixelQuintuplets * sizeof(FPX), stream);
//    pixelQuintupletsInGPU.quintupletRadius = (FPX*)cms::cuda::allocate_managed(maxPixelQuintuplets * sizeof(FPX), stream);
//
//#else
//    cudaMallocManaged(&pixelQuintupletsInGPU.pixelIndices, maxPixelQuintuplets * sizeof(unsigned int));
//    cudaMallocManaged(&pixelQuintupletsInGPU.T5Indices, maxPixelQuintuplets * sizeof(unsigned int));
//    cudaMallocManaged(&pixelQuintupletsInGPU.nPixelQuintuplets, sizeof(unsigned int));
//    cudaMallocManaged(&pixelQuintupletsInGPU.totOccupancyPixelQuintuplets, sizeof(unsigned int));
//    cudaMallocManaged(&pixelQuintupletsInGPU.isDup, maxPixelQuintuplets * sizeof(bool));
//    cudaMallocManaged(&pixelQuintupletsInGPU.score, maxPixelQuintuplets * sizeof(FPX));
//    cudaMallocManaged(&pixelQuintupletsInGPU.eta  , maxPixelQuintuplets * sizeof(FPX));
//    cudaMallocManaged(&pixelQuintupletsInGPU.phi  , maxPixelQuintuplets * sizeof(FPX));
//
//    cudaMallocManaged(&pixelQuintupletsInGPU.logicalLayers, maxPixelQuintuplets * 7 *sizeof(uint8_t));
//    cudaMallocManaged(&pixelQuintupletsInGPU.hitIndices, maxPixelQuintuplets * 14 * sizeof(unsigned int));
//    cudaMallocManaged(&pixelQuintupletsInGPU.lowerModuleIndices, maxPixelQuintuplets * 7 * sizeof(uint16_t));
//    cudaMallocManaged(&pixelQuintupletsInGPU.pixelRadius, maxPixelQuintuplets * sizeof(FPX));
//    cudaMallocManaged(&pixelQuintupletsInGPU.quintupletRadius, maxPixelQuintuplets * sizeof(FPX));
//    cudaMallocManaged(&pixelQuintupletsInGPU.centerX, maxPixelQuintuplets * sizeof(FPX));
//    cudaMallocManaged(&pixelQuintupletsInGPU.centerY, maxPixelQuintuplets * sizeof(FPX));
//#ifdef CUT_VALUE_DEBUG
//    cudaMallocManaged(&pixelQuintupletsInGPU.rzChiSquared, maxPixelQuintuplets * sizeof(unsigned int));
//    cudaMallocManaged(&pixelQuintupletsInGPU.rPhiChiSquared, maxPixelQuintuplets * sizeof(unsigned int));
//    cudaMallocManaged(&pixelQuintupletsInGPU.rPhiChiSquaredInwards, maxPixelQuintuplets * sizeof(unsigned int));
//#endif
//#endif
//
//    cudaMemsetAsync(pixelQuintupletsInGPU.nPixelQuintuplets, 0, sizeof(unsigned int),stream);
//    cudaMemsetAsync(pixelQuintupletsInGPU.totOccupancyPixelQuintuplets, 0, sizeof(unsigned int),stream);
//  cudaStreamSynchronize(stream);
//}
//
//void SDL::createPixelQuintupletsInExplicitMemory(struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, unsigned int maxPixelQuintuplets,cudaStream_t stream)
//{
//#ifdef CACHE_ALLOC
////    cudaStream_t stream=0;
//    int dev;
//    cudaGetDevice(&dev);
//    pixelQuintupletsInGPU.pixelIndices        = (unsigned int*)cms::cuda::allocate_device(dev,maxPixelQuintuplets * sizeof(unsigned int),stream);
//    pixelQuintupletsInGPU.T5Indices           = (unsigned int*)cms::cuda::allocate_device(dev,maxPixelQuintuplets * sizeof(unsigned int),stream);
//    pixelQuintupletsInGPU.nPixelQuintuplets   = (unsigned int*)cms::cuda::allocate_device(dev,sizeof(unsigned int),stream);
//    pixelQuintupletsInGPU.totOccupancyPixelQuintuplets   = (unsigned int*)cms::cuda::allocate_device(dev,sizeof(unsigned int),stream);
//    pixelQuintupletsInGPU.isDup               = (bool*)cms::cuda::allocate_device(dev,maxPixelQuintuplets * sizeof(bool),stream);
//    pixelQuintupletsInGPU.score               = (FPX*)cms::cuda::allocate_device(dev,maxPixelQuintuplets * sizeof(FPX),stream);
//    pixelQuintupletsInGPU.eta                 = (FPX*)cms::cuda::allocate_device(dev,maxPixelQuintuplets * sizeof(FPX),stream);
//    pixelQuintupletsInGPU.phi                 = (FPX*)cms::cuda::allocate_device(dev,maxPixelQuintuplets * sizeof(FPX),stream);
//    pixelQuintupletsInGPU.hitIndices = (unsigned int*)cms::cuda::allocate_device(dev, maxPixelQuintuplets * 14 * sizeof(unsigned int), stream);
//    pixelQuintupletsInGPU.logicalLayers = (uint8_t*)cms::cuda::allocate_device(dev, maxPixelQuintuplets * 7 * sizeof(uint8_t), stream);
//    pixelQuintupletsInGPU.lowerModuleIndices = (uint16_t*)cms::cuda::allocate_device(dev, maxPixelQuintuplets * 7 * sizeof(uint16_t), stream);
//    pixelQuintupletsInGPU.centerX          = (FPX*)cms::cuda::allocate_device(dev, maxPixelQuintuplets * sizeof(FPX), stream);
//    pixelQuintupletsInGPU.centerY          = (FPX*)cms::cuda::allocate_device(dev, maxPixelQuintuplets * sizeof(FPX), stream);
//    pixelQuintupletsInGPU.pixelRadius      = (FPX*)cms::cuda::allocate_device(dev, maxPixelQuintuplets * sizeof(FPX), stream);
//    pixelQuintupletsInGPU.quintupletRadius = (FPX*)cms::cuda::allocate_device(dev, maxPixelQuintuplets * sizeof(FPX), stream);
//#else
//    cudaMalloc(&pixelQuintupletsInGPU.pixelIndices, maxPixelQuintuplets * sizeof(unsigned int));
//    cudaMalloc(&pixelQuintupletsInGPU.T5Indices, maxPixelQuintuplets * sizeof(unsigned int));
//    cudaMalloc(&pixelQuintupletsInGPU.nPixelQuintuplets, sizeof(unsigned int));
//    cudaMalloc(&pixelQuintupletsInGPU.totOccupancyPixelQuintuplets, sizeof(unsigned int));
//    cudaMalloc(&pixelQuintupletsInGPU.isDup, maxPixelQuintuplets * sizeof(bool));
//    cudaMalloc(&pixelQuintupletsInGPU.score, maxPixelQuintuplets * sizeof(FPX));
//    cudaMalloc(&pixelQuintupletsInGPU.eta  , maxPixelQuintuplets * sizeof(FPX));
//    cudaMalloc(&pixelQuintupletsInGPU.phi  , maxPixelQuintuplets * sizeof(FPX));
//
//    cudaMalloc(&pixelQuintupletsInGPU.logicalLayers, maxPixelQuintuplets * 7 *sizeof(uint8_t));
//    cudaMalloc(&pixelQuintupletsInGPU.hitIndices, maxPixelQuintuplets * 14 * sizeof(unsigned int));
//    cudaMalloc(&pixelQuintupletsInGPU.lowerModuleIndices, maxPixelQuintuplets * 7 * sizeof(uint16_t));
//    cudaMalloc(&pixelQuintupletsInGPU.pixelRadius, maxPixelQuintuplets * sizeof(FPX));
//    cudaMalloc(&pixelQuintupletsInGPU.quintupletRadius, maxPixelQuintuplets * sizeof(FPX));
//    cudaMalloc(&pixelQuintupletsInGPU.centerX, maxPixelQuintuplets * sizeof(FPX));
//    cudaMalloc(&pixelQuintupletsInGPU.centerY, maxPixelQuintuplets * sizeof(FPX));
//#endif
//    cudaMemsetAsync(pixelQuintupletsInGPU.nPixelQuintuplets, 0, sizeof(unsigned int),stream);
//    cudaMemsetAsync(pixelQuintupletsInGPU.totOccupancyPixelQuintuplets, 0, sizeof(unsigned int),stream);
//  cudaStreamSynchronize(stream);
//}
//
//__device__ void SDL::rmPixelQuintupletToMemory(struct pixelQuintuplets& pixelQuintupletsInGPU, unsigned int pixelQuintupletIndex)
//{
//
//    pixelQuintupletsInGPU.isDup[pixelQuintupletIndex] = 1;
//}
//#ifdef CUT_VALUE_DEBUG
//__device__ void SDL::addPixelQuintupletToMemory(struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct quintuplets& quintupletsInGPU, struct pixelQuintuplets& pixelQuintupletsInGPU, unsigned int pixelIndex, unsigned int T5Index, unsigned int pixelQuintupletIndex, float& rzChiSquared, float& rPhiChiSquared, float& rPhiChiSquaredInwards, float score, float eta, float phi, float& pixelRadius, float& quintupletRadius,
//        float& centerX, float& centerY)
//#else
//__device__ void SDL::addPixelQuintupletToMemory(struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct quintuplets& quintupletsInGPU, struct pixelQuintuplets& pixelQuintupletsInGPU, unsigned int pixelIndex, unsigned int T5Index, unsigned int pixelQuintupletIndex, float score, float eta, float phi, float& pixelRadius, float& quintupletRadius, float& centerX, float& centerY)
//#endif
//{
//    pixelQuintupletsInGPU.pixelIndices[pixelQuintupletIndex] = pixelIndex;
//    pixelQuintupletsInGPU.T5Indices[pixelQuintupletIndex] = T5Index;
//    pixelQuintupletsInGPU.isDup[pixelQuintupletIndex] = 0;
//    pixelQuintupletsInGPU.score[pixelQuintupletIndex] = __F2H(score);
//    pixelQuintupletsInGPU.eta[pixelQuintupletIndex]   = __F2H(eta);
//    pixelQuintupletsInGPU.phi[pixelQuintupletIndex]   = __F2H(phi);
//
//    pixelQuintupletsInGPU.pixelRadius[pixelQuintupletIndex] = __F2H(pixelRadius);
//    pixelQuintupletsInGPU.quintupletRadius[pixelQuintupletIndex] = __F2H(quintupletRadius);
//    pixelQuintupletsInGPU.centerX[pixelQuintupletIndex] = __F2H(centerX);
//    pixelQuintupletsInGPU.centerY[pixelQuintupletIndex] = __F2H(centerY);
//
//    pixelQuintupletsInGPU.logicalLayers[7 * pixelQuintupletIndex] = 0;
//    pixelQuintupletsInGPU.logicalLayers[7 * pixelQuintupletIndex + 1] = 0;
//    pixelQuintupletsInGPU.logicalLayers[7 * pixelQuintupletIndex + 2] = quintupletsInGPU.logicalLayers[T5Index * 5];
//    pixelQuintupletsInGPU.logicalLayers[7 * pixelQuintupletIndex + 3] = quintupletsInGPU.logicalLayers[T5Index * 5 + 1];
//    pixelQuintupletsInGPU.logicalLayers[7 * pixelQuintupletIndex + 4] = quintupletsInGPU.logicalLayers[T5Index * 5 + 2];
//    pixelQuintupletsInGPU.logicalLayers[7 * pixelQuintupletIndex + 5] = quintupletsInGPU.logicalLayers[T5Index * 5 + 3];
//    pixelQuintupletsInGPU.logicalLayers[7 * pixelQuintupletIndex + 6] = quintupletsInGPU.logicalLayers[T5Index * 5 + 4];
//
//    pixelQuintupletsInGPU.lowerModuleIndices[7 * pixelQuintupletIndex] = segmentsInGPU.innerLowerModuleIndices[pixelIndex];
//    pixelQuintupletsInGPU.lowerModuleIndices[7 * pixelQuintupletIndex + 1] = segmentsInGPU.outerLowerModuleIndices[pixelIndex];
//    pixelQuintupletsInGPU.lowerModuleIndices[7 * pixelQuintupletIndex + 2] = quintupletsInGPU.lowerModuleIndices[T5Index * 5];
//    pixelQuintupletsInGPU.lowerModuleIndices[7 * pixelQuintupletIndex + 3] = quintupletsInGPU.lowerModuleIndices[T5Index * 5 + 1];
//    pixelQuintupletsInGPU.lowerModuleIndices[7 * pixelQuintupletIndex + 4] = quintupletsInGPU.lowerModuleIndices[T5Index * 5 + 2];
//    pixelQuintupletsInGPU.lowerModuleIndices[7 * pixelQuintupletIndex + 5] = quintupletsInGPU.lowerModuleIndices[T5Index * 5 + 3];
//    pixelQuintupletsInGPU.lowerModuleIndices[7 * pixelQuintupletIndex + 6] = quintupletsInGPU.lowerModuleIndices[T5Index * 5 + 4];
//
//    unsigned int pixelInnerMD = segmentsInGPU.mdIndices[2 * pixelIndex];
//    unsigned int pixelOuterMD = segmentsInGPU.mdIndices[2 * pixelIndex + 1];
//
//    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex] = mdsInGPU.anchorHitIndices[pixelInnerMD];
//    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 1] = mdsInGPU.outerHitIndices[pixelInnerMD];
//    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 2] = mdsInGPU.anchorHitIndices[pixelOuterMD];
//    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 3] = mdsInGPU.outerHitIndices[pixelOuterMD];
//
//    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 4] = quintupletsInGPU.hitIndices[10 * T5Index];
//    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 5] = quintupletsInGPU.hitIndices[10 * T5Index + 1];
//    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 6] = quintupletsInGPU.hitIndices[10 * T5Index + 2];
//    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 7] = quintupletsInGPU.hitIndices[10 * T5Index + 3];
//    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 8] = quintupletsInGPU.hitIndices[10 * T5Index + 4];
//    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 9] = quintupletsInGPU.hitIndices[10 * T5Index + 5];
//    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 10] = quintupletsInGPU.hitIndices[10 * T5Index + 6];
//    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 11] = quintupletsInGPU.hitIndices[10 * T5Index + 7];
//    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 12] = quintupletsInGPU.hitIndices[10 * T5Index + 8];
//    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 13] = quintupletsInGPU.hitIndices[10 * T5Index + 9];
//        
//#ifdef CUT_VALUE_DEBUG
//    pixelQuintupletsInGPU.rzChiSquared[pixelQuintupletIndex] = rzChiSquared;
//    pixelQuintupletsInGPU.rPhiChiSquared[pixelQuintupletIndex] = rPhiChiSquared;
//    pixelQuintupletsInGPU.rPhiChiSquaredInwards[pixelQuintupletIndex] = rPhiChiSquaredInwards;
//#endif
//}
//
//__device__ bool SDL::runPixelQuintupletDefaultAlgo(struct modules& modulesInGPU, struct objectRanges& rangesInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct triplets& tripletsInGPU, struct quintuplets& quintupletsInGPU, unsigned int& pixelSegmentIndex, unsigned int& quintupletIndex, float& rzChiSquared, float& rPhiChiSquared, float& rPhiChiSquaredInwards, float& pixelRadius, float& quintupletRadius, float& centerX, float& centerY)
//{
//    bool pass = true;
//    
//    unsigned int pixelModuleIndex = segmentsInGPU.innerLowerModuleIndices[pixelSegmentIndex];
//
//    unsigned int pixelSegmentArrayIndex = pixelSegmentIndex - rangesInGPU.segmentModuleIndices[pixelModuleIndex];
//
//    unsigned int T5InnerT3Index = quintupletsInGPU.tripletIndices[2 * quintupletIndex];
//    unsigned int T5OuterT3Index = quintupletsInGPU.tripletIndices[2 * quintupletIndex + 1];
//
//    float pixelRadiusTemp, pixelRadiusError, tripletRadius, rPhiChiSquaredTemp, rzChiSquaredTemp, rPhiChiSquaredInwardsTemp, centerXTemp, centerYTemp;
//
//    pass = pass and runPixelTripletDefaultAlgo(modulesInGPU, rangesInGPU, mdsInGPU, segmentsInGPU, tripletsInGPU, pixelSegmentIndex, T5InnerT3Index, pixelRadiusTemp, pixelRadiusError, tripletRadius, centerXTemp, centerYTemp, rzChiSquaredTemp, rPhiChiSquaredTemp, rPhiChiSquaredInwardsTemp, false);
//    if(not pass) return false;
//
//    unsigned int firstSegmentIndex = tripletsInGPU.segmentIndices[2 * T5InnerT3Index];
//    unsigned int secondSegmentIndex = tripletsInGPU.segmentIndices[2 * T5InnerT3Index + 1];
//    unsigned int thirdSegmentIndex = tripletsInGPU.segmentIndices[2 * T5OuterT3Index];
//    unsigned int fourthSegmentIndex = tripletsInGPU.segmentIndices[2 * T5OuterT3Index + 1];
//
//    unsigned int pixelInnerMDIndex = segmentsInGPU.mdIndices[2 * pixelSegmentIndex];
//    unsigned int pixelOuterMDIndex = segmentsInGPU.mdIndices[2 * pixelSegmentIndex + 1];
//    unsigned int firstMDIndex = segmentsInGPU.mdIndices[2 * firstSegmentIndex];
//    unsigned int secondMDIndex = segmentsInGPU.mdIndices[2 * secondSegmentIndex];
//    unsigned int thirdMDIndex = segmentsInGPU.mdIndices[2 * secondSegmentIndex + 1];
//    unsigned int fourthMDIndex = segmentsInGPU.mdIndices[2 * thirdSegmentIndex + 1];
//    unsigned int fifthMDIndex = segmentsInGPU.mdIndices[2 * fourthSegmentIndex + 1];
//
//    uint16_t lowerModuleIndex1 = quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex];
//    uint16_t lowerModuleIndex2 = quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex + 1];
//    uint16_t lowerModuleIndex3 = quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex + 2];
//    uint16_t lowerModuleIndex4 = quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex + 3];
//    uint16_t lowerModuleIndex5 = quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex + 4];
//
//    uint16_t lowerModuleIndices[] = {lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4, lowerModuleIndex5};
//    
//    float zPix[] = {mdsInGPU.anchorZ[pixelInnerMDIndex], mdsInGPU.anchorZ[pixelOuterMDIndex]};
//    float rtPix[] = {mdsInGPU.anchorRt[pixelInnerMDIndex], mdsInGPU.anchorRt[pixelOuterMDIndex]};
//    float zs[] = {mdsInGPU.anchorZ[firstMDIndex], mdsInGPU.anchorZ[secondMDIndex], mdsInGPU.anchorZ[thirdMDIndex], mdsInGPU.anchorZ[fourthMDIndex], mdsInGPU.anchorZ[fifthMDIndex]};
//    float rts[] = {mdsInGPU.anchorRt[firstMDIndex], mdsInGPU.anchorRt[secondMDIndex], mdsInGPU.anchorRt[thirdMDIndex], mdsInGPU.anchorRt[fourthMDIndex], mdsInGPU.anchorRt[fifthMDIndex]};
//
//    rzChiSquared = computePT5RZChiSquared(modulesInGPU, lowerModuleIndices, rtPix, zPix, rts, zs);
//
//    if(pixelRadius < 5.0f/(2.f * k2Rinv1GeVf))
//    {
//        pass = pass and passPT5RZChiSquaredCuts(modulesInGPU, lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4, lowerModuleIndex5, rzChiSquared);
//        if(not pass) return pass;
//    }
//
//    //outer T5
//    float xs[] = {mdsInGPU.anchorX[firstMDIndex], mdsInGPU.anchorX[secondMDIndex], mdsInGPU.anchorX[thirdMDIndex], mdsInGPU.anchorX[fourthMDIndex], mdsInGPU.anchorX[fifthMDIndex]};
//    float ys[] = {mdsInGPU.anchorY[firstMDIndex], mdsInGPU.anchorY[secondMDIndex], mdsInGPU.anchorY[thirdMDIndex], mdsInGPU.anchorY[fourthMDIndex], mdsInGPU.anchorY[fifthMDIndex]};
//
//    //get the appropriate radii and centers
//    centerX  = segmentsInGPU.circleCenterX[pixelSegmentArrayIndex];
//    centerY = segmentsInGPU.circleCenterY[pixelSegmentArrayIndex];
//    pixelRadius = segmentsInGPU.circleRadius[pixelSegmentArrayIndex];
//
//    float T5CenterX = quintupletsInGPU.regressionG[quintupletIndex];
//    float T5CenterY = quintupletsInGPU.regressionF[quintupletIndex];
//    quintupletRadius = quintupletsInGPU.regressionRadius[quintupletIndex];
//
//    rPhiChiSquared = computePT5RPhiChiSquared(modulesInGPU, lowerModuleIndices, centerX, centerY, pixelRadius, xs, ys);
//
//    if(pixelRadius < 5.0f/(2.f * k2Rinv1GeVf))
//    {
//        pass = pass and passPT5RPhiChiSquaredCuts(modulesInGPU, lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4, lowerModuleIndex5, rPhiChiSquared);
//        if(not pass) return pass;
//    }
//
//    float xPix[] = {mdsInGPU.anchorX[pixelInnerMDIndex], mdsInGPU.anchorX[pixelOuterMDIndex]};
//    float yPix[] = {mdsInGPU.anchorY[pixelInnerMDIndex], mdsInGPU.anchorY[pixelOuterMDIndex]};
//    rPhiChiSquaredInwards = computePT5RPhiChiSquaredInwards(modulesInGPU, T5CenterX, T5CenterY, quintupletRadius, xPix, yPix);
//
//    if(quintupletsInGPU.regressionRadius[quintupletIndex] < 5.0f/(2.f * k2Rinv1GeVf))
//    {
//        pass = pass and passPT5RPhiChiSquaredInwardsCuts(modulesInGPU, lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4, lowerModuleIndex5, rPhiChiSquaredInwards); 
//        if(not pass) return pass;
//    }
//    //trusting the T5 regression center to also be a good estimate..
//    centerX = (centerX + T5CenterX)/2;
//    centerY = (centerY + T5CenterY)/2;
//
//    //other cuts will be filled here!
//    return pass;
//}
//
//__device__ bool SDL::passPT5RPhiChiSquaredCuts(struct modules& modulesInGPU, uint16_t& lowerModuleIndex1, uint16_t& lowerModuleIndex2, uint16_t& lowerModuleIndex3, uint16_t& lowerModuleIndex4, uint16_t& lowerModuleIndex5, float rPhiChiSquared)
//{
//    const int layer1 = modulesInGPU.layers[lowerModuleIndex1] + 6 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex1] == SDL::TwoS);
//    const int layer2 = modulesInGPU.layers[lowerModuleIndex2] + 6 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex2] == SDL::TwoS);
//    const int layer3 = modulesInGPU.layers[lowerModuleIndex3] + 6 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex3] == SDL::TwoS);
//    const int layer4 = modulesInGPU.layers[lowerModuleIndex4] + 6 * (modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex4] == SDL::TwoS);
//    const int layer5 = modulesInGPU.layers[lowerModuleIndex5] + 6 * (modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex5] == SDL::TwoS);
//
//    if(layer1 == 1 and layer2 == 2 and layer3 == 3)
//    {
//        if(layer4 == 12 and layer5 == 13)
//        {
//            return rPhiChiSquared < 48.921f;
//        }
//        else if(layer4 == 4 and layer5 == 12)
//        {
//            return rPhiChiSquared < 97.948f;
//        }
//        else if(layer4 == 4 and layer5 == 5)
//        {
//            return rPhiChiSquared < 129.3f;
//        }
//        else if(layer4 == 7 and layer5 == 13)
//        {
//            return rPhiChiSquared < 56.21f;
//        }
//        else if(layer4 == 7 and layer5 == 8)
//        {
//            return rPhiChiSquared < 74.198f;
//        }
//    }
//    else if(layer1 == 1 and layer2 == 2 and layer3 == 7)
//    {
//        if(layer4 == 13 and layer5 == 14)
//        {
//            return rPhiChiSquared < 21.265f;
//        }
//        else if(layer4 == 8 and layer5 == 14)
//        {
//            return rPhiChiSquared < 37.058f;
//        }
//        else if(layer4 == 8 and layer5 == 9)
//        {
//            return rPhiChiSquared < 42.578f;
//        }
//    }
//    else if(layer1 == 1 and layer2 == 7 and layer3 == 8)
//    {
//        if(layer4 == 9 and layer5 == 10)
//        {
//            return rPhiChiSquared < 32.253f;
//        }
//        else if(layer4 == 9 and layer5 == 15)
//        {
//            return rPhiChiSquared < 37.058f;
//        }
//    }
//    else if(layer1 == 2 and layer2 == 3 and layer3 == 4)
//    {
//        if(layer4 == 12 and layer5 == 13)
//        {
//            return rPhiChiSquared < 97.947f;
//        }
//        else if(layer4 == 5 and layer5 == 12)
//        {
//            return rPhiChiSquared < 129.3f;
//        }
//        else if(layer4 == 5 and layer5 == 6)
//        {
//            return rPhiChiSquared < 170.68f;
//        }
//    }
//    else if(layer1 == 2 and layer2 == 3 and layer3 == 7)
//    {
//        if(layer4 == 13 and layer5 == 14)
//        {   
//            return rPhiChiSquared < 48.92f;
//        }
//        else if(layer4 == 8 and layer5 == 14)
//        {
//            return rPhiChiSquared < 74.2f;
//        }
//    }
//    else if(layer1 == 2 and layer2 == 7 and layer3 == 8)
//    {
//        if(layer4 == 14 and layer5 == 15)
//        {
//            return rPhiChiSquared < 42.58f;
//        }
//        else if(layer4 == 9 and layer5 == 10)
//        {
//            return rPhiChiSquared < 37.06f;
//        }
//        else if(layer4 == 9 and layer5 == 15)
//        {
//            return rPhiChiSquared < 48.92f;
//        }
//    }
//    else if(layer1 == 3 and layer2 == 7 and layer3 == 8 and layer4 == 14 and layer5 == 15)
//    {
//        return rPhiChiSquared < 85.25f;
//    }
//    else if(layer1 == 7 and layer2 == 8 and layer3 == 9)
//    {
//        if(layer4 == 10 and layer5 == 11)
//        {
//            return rPhiChiSquared < 42.58f;
//        }
//        else if(layer4 == 10 and layer5 == 16)
//        {
//            return rPhiChiSquared < 37.06f;
//        }
//        else if(layer4 == 15 and layer5 == 16)
//        {
//            return rPhiChiSquared < 37.06f;
//        }
//    }
//    return true;
//}
//
//
//
//__device__ bool SDL::passPT5RPhiChiSquaredInwardsCuts(struct modules& modulesInGPU, uint16_t& lowerModuleIndex1, uint16_t& lowerModuleIndex2, uint16_t& lowerModuleIndex3, uint16_t& lowerModuleIndex4, uint16_t& lowerModuleIndex5, float rPhiChiSquared)
//{
//    const int layer1 = modulesInGPU.layers[lowerModuleIndex1] + 6 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex1] == SDL::TwoS);
//    const int layer2 = modulesInGPU.layers[lowerModuleIndex2] + 6 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex2] == SDL::TwoS);
//    const int layer3 = modulesInGPU.layers[lowerModuleIndex3] + 6 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex3] == SDL::TwoS);
//    const int layer4 = modulesInGPU.layers[lowerModuleIndex4] + 6 * (modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex4] == SDL::TwoS);
//    const int layer5 = modulesInGPU.layers[lowerModuleIndex5] + 6 * (modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex5] == SDL::TwoS);
//
//    if(layer1 == 1 and layer2 == 2 and layer3 == 3)
//    {
//        if(layer4 == 12 and layer5 == 13)
//        {
//            return rPhiChiSquared < 451.141f;
//        }
//        else if(layer4 == 4 and layer5 == 12)
//        {
//            return rPhiChiSquared < 786.173f;
//        }
//        else if(layer4 == 4 and layer5 == 5)
//        {
//            return rPhiChiSquared < 595.545f;
//        }
//        else if(layer4 == 7 and layer5 == 13)
//        {
//            return rPhiChiSquared < 581.339f;
//        }
//        else if(layer4 == 7 and layer5 == 8)
//        {
//            return rPhiChiSquared < 112.537f;
//        }
//    }
//    else if(layer1 == 1 and layer2 == 2 and layer3 == 7)
//    {
//        if(layer4 == 13 and layer5 == 14)
//        {
//            return rPhiChiSquared < 225.322f;
//        }
//        else if(layer4 == 8 and layer5 == 14)
//        {
//            return rPhiChiSquared < 1192.402f;
//        }
//        else if(layer4 == 8 and layer5 == 9)
//        {
//            return rPhiChiSquared < 786.173f;
//        }
//    }
//    else if(layer1 == 1 and layer2 == 7 and layer3 == 8)
//    {
//        if(layer4 == 9 and layer5 == 10)
//        {
//            return rPhiChiSquared < 1037.817f;
//        }
//        else if(layer4 == 9 and layer5 == 15)
//        {
//            return rPhiChiSquared < 1808.536f;
//        }
//    }
//    else if(layer1 == 2 and layer2 == 3 and layer3 == 4)
//    {
//        if(layer4 == 12 and layer5 == 13)
//        {
//            return rPhiChiSquared < 684.253f;
//        }
//        else if(layer4 == 5 and layer5 == 12)
//        {
//            return rPhiChiSquared < 684.253f;
//        }
//        else if(layer4 == 5 and layer5 == 6)
//        {
//            return rPhiChiSquared < 684.253f;
//        }
//    }
//    else if(layer1 == 2 and layer2 == 3 and layer3 == 7)
//    {
//        if(layer4 == 13 and layer5 == 14)
//        {   
//            return rPhiChiSquared < 451.141f;
//        }
//        else if(layer4 == 8 and layer5 == 14)
//        {
//            return rPhiChiSquared < 518.34f;
//        }
//    }
//    else if(layer1 == 2 and layer2 == 7 and layer3 == 8)
//    {
//        if(layer4 == 14 and layer5 == 15)
//        {
//            return rPhiChiSquared < 2077.92f;
//        }
//        else if(layer4 == 9 and layer5 == 10)
//        {
//            return rPhiChiSquared < 74.20f;
//        }
//        else if(layer4 == 9 and layer5 == 15)
//        {
//            return rPhiChiSquared < 1808.536f;
//        }
//    }
//    else if(layer1 == 3 and layer2 == 7 and layer3 == 8 and layer4 == 14 and layer5 == 15)
//    {
//        return rPhiChiSquared < 786.173f;
//    }
//    else if(layer1 == 7 and layer2 == 8 and layer3 == 9)
//    {
//        if(layer4 == 10 and layer5 == 11)
//        {
//            return rPhiChiSquared < 1574.076f;
//        }
//        else if(layer4 == 10 and layer5 == 16)
//        {
//            return rPhiChiSquared < 5492.11f;
//        }
//        else if(layer4 == 15 and layer5 == 16)
//        {
//            return rPhiChiSquared < 2743.037f;
//        }
//    }
//    return true;
//}
//
//__device__ float SDL::computePT5RPhiChiSquared(struct modules& modulesInGPU, uint16_t* lowerModuleIndices, float& g, float& f, float& radius, float* xs, float* ys)
//{
//    /*
//       Compute circle parameters from 3 pixel hits, and then use them to compute the chi squared for the outer hits
//    */
//
//    float delta1[5], delta2[5], slopes[5];
//    bool isFlat[5];
//    float chiSquared = 0;
//
//    computeSigmasForRegression_pT5(modulesInGPU, lowerModuleIndices, delta1, delta2, slopes, isFlat);
//    chiSquared = computeChiSquaredpT5(5, xs, ys, delta1, delta2, slopes, isFlat, g, f, radius);
//
//    return chiSquared;
//}
//
//__device__ bool SDL::passPT5RZChiSquaredCuts(struct modules& modulesInGPU, uint16_t& lowerModuleIndex1, uint16_t& lowerModuleIndex2, uint16_t& lowerModuleIndex3, uint16_t& lowerModuleIndex4, uint16_t& lowerModuleIndex5, float& rzChiSquared)
//{
//    const int layer1 = modulesInGPU.layers[lowerModuleIndex1] + 6 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex1] == SDL::TwoS);
//    const int layer2 = modulesInGPU.layers[lowerModuleIndex2] + 6 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex2] == SDL::TwoS);
//    const int layer3 = modulesInGPU.layers[lowerModuleIndex3] + 6 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex3] == SDL::TwoS);
//    const int layer4 = modulesInGPU.layers[lowerModuleIndex4] + 6 * (modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex4] == SDL::TwoS);
//    const int layer5 = modulesInGPU.layers[lowerModuleIndex5] + 6 * (modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex5] == SDL::TwoS);
//
//    
//    if(layer1 == 1 and layer2 == 2 and layer3 == 3)
//    {
//        if(layer4 == 12 and layer5 == 13)
//        {
//            return rzChiSquared < 451.141f;
//        }
//        else if(layer4 == 4 and layer5 == 12)
//        {
//            return rzChiSquared < 392.654f;
//        }
//        else if(layer4 == 4 and layer5 == 5)
//        {
//            return rzChiSquared < 225.322f;
//        }
//        else if(layer4 == 7 and layer5 == 13)
//        {
//            return rzChiSquared < 595.546f;
//        }
//        else if(layer4 == 7 and layer5 == 8)
//        {
//            return rzChiSquared < 196.111f;
//        }
//    }
//    else if(layer1 == 1 and layer2 == 2 and layer3 == 7)
//    {
//        if(layer4 == 13 and layer5 == 14)
//        {
//            return rzChiSquared < 297.446f;
//        }
//        else if(layer4 == 8 and layer5 == 14)
//        {   
//            return rzChiSquared < 451.141f;
//        }
//        else if(layer4 == 8 and layer5 == 9)
//        {
//            return rzChiSquared < 518.339f;
//        }
//    }
//    else if(layer1 == 1 and layer2 == 7 and layer3 == 8)
//    {
//        if(layer4 == 9 and layer5 == 10)
//        {
//            return rzChiSquared < 341.75f;
//        }
//        else if(layer4 == 9 and layer5 == 15)
//        {
//            return rzChiSquared < 341.75f;
//        }
//    }
//    else if(layer1 == 2 and layer2 == 3 and layer3 == 4)
//    {
//        if(layer4 == 12 and layer5 == 13)
//        {
//            return rzChiSquared < 392.655f;
//        }
//        else if(layer4 == 5 and layer5 == 12)
//        {
//            return rzChiSquared < 341.75f;
//        }
//        else if(layer4 == 5 and layer5 == 6)
//        {
//            return rzChiSquared < 112.537f;
//        }
//    }
//    else if(layer1 == 2 and layer2 == 3 and layer4 == 7)
//    {
//        if(layer4 == 13 and layer5 == 14)
//        {
//            return rzChiSquared < 595.545f;
//        }
//        else if(layer4 == 8 and layer5 == 14)
//        {
//            return rzChiSquared < 74.198f;
//        }
//    }
//    else if(layer1 == 2 and layer2 == 7 and layer3 == 8)
//    {
//        if(layer4 == 14 and layer5 == 15)
//        {
//            return rzChiSquared < 518.339f;
//        }
//        else if(layer4 == 9 and layer5 == 10)
//        {
//            return rzChiSquared < 8.046f;
//        }
//        else if(layer4 == 9 and layer5 == 15)
//        {
//            return rzChiSquared < 451.141f;
//        }
//    }
//    else if(layer1 == 3 and layer2 == 7 and layer3 == 8 and layer4 == 14 and layer5 == 15)
//    {
//        return rzChiSquared < 56.207f;
//    }
//    else if(layer1 == 7 and layer2 == 8 and layer3 == 9)
//    {
//        if(layer4 == 10 and layer5 == 11)
//        {
//            return rzChiSquared < 64.578f;
//        }
//        else if(layer4 == 10 and layer5 == 16)
//        {
//            return rzChiSquared < 85.250f;
//        }
//        else if(layer4 == 15 and layer5 == 16)
//        {
//            return rzChiSquared < 85.250f;
//        }
//    }
//    return true;
//}
//
//__device__ float SDL::computePT5RPhiChiSquaredInwards(struct modules& modulesInGPU, float& g, float& f, float& r, float* xPix, float* yPix)
//{
//    /*Using the computed regression center and radius, compute the chi squared for the pixels*/
//    float chiSquared = 0;   
//    for(size_t i = 0; i < 2; i++)
//    {
//        float residual = (xPix[i] - g) * (xPix[i] -g) + (yPix[i] - f) * (yPix[i] - f) - r * r;
//        chiSquared += residual * residual;
//    }
//    chiSquared *= 0.5f;
//    return chiSquared;
//}
//
//
//__device__ float SDL::computePT5RZChiSquared(struct modules& modulesInGPU, uint16_t* lowerModuleIndices, float* rtPix, float* zPix, float* rts, float* zs)
//{
//    //use the two anchor hits of the pixel segment to compute the slope
//    //then compute the pseudo chi squared of the five outer hits
//
//    float slope = (zPix[1] - zPix[0]) / (rtPix[1] - rtPix[0]);
//    float residual = 0;
//    float error = 0;
//    //hardcoded array indices!!!
//    float RMSE = 0;
//    for(size_t i = 0; i < 5; i++)
//    {
//        uint16_t& lowerModuleIndex = lowerModuleIndices[i];
//        const int moduleType = modulesInGPU.moduleType[lowerModuleIndex];
//        const int moduleSide = modulesInGPU.sides[lowerModuleIndex];
//        const int moduleSubdet = modulesInGPU.subdets[lowerModuleIndex];
// 
//        residual = (moduleSubdet == SDL::Barrel) ? (zs[i] - zPix[0]) - slope * (rts[i] - rtPix[0]) : (rts[i] - rtPix[0]) - (zs[i] - zPix[0])/slope;
//        float& drdz = modulesInGPU.drdzs[lowerModuleIndex]; 
//        //PS Modules
//        if(moduleType == 0)
//        {
//            error = 0.15f;
//        }
//        else //2S modules
//        {
//            error = 5.0f;
//        }
//
//        //special dispensation to tilted PS modules!
//        if(moduleType == 0 and moduleSubdet == SDL::Barrel and moduleSide != Center)
//        {
//            //error *= 1.f/sqrtf(1.f + drdz * drdz);
//            error /= sqrtf(1.f + drdz * drdz);
//        }
//        RMSE += (residual * residual)/(error * error);
//    }
//
//    RMSE = sqrtf(0.2f * RMSE);
//    return RMSE;
//}
//__device__ void SDL::computeSigmasForRegression_pT5(SDL::modules& modulesInGPU, const uint16_t* lowerModuleIndices, float* delta1, float* delta2, float* slopes, bool* isFlat, int nPoints, bool anchorHits)
//{
//   /*bool anchorHits required to deal with a weird edge case wherein
//     the hits ultimately used in the regression are anchor hits, but the
//     lower modules need not all be Pixel Modules (in case of PS). Similarly,
//     when we compute the chi squared for the non-anchor hits, the "partner module"
//     need not always be a PS strip module, but all non-anchor hits sit on strip
//     modules.
//    */
//    ModuleType moduleType;
//    short moduleSubdet, moduleSide;
//    float inv1 = 0.01f/0.009f;
//    float inv2 = 0.15f/0.009f;
//    float inv3 = 2.4f/0.009f;
//    for(size_t i=0; i<nPoints; i++)
//    {
//        moduleType = modulesInGPU.moduleType[lowerModuleIndices[i]];
//        moduleSubdet = modulesInGPU.subdets[lowerModuleIndices[i]];
//        moduleSide = modulesInGPU.sides[lowerModuleIndices[i]];
//        float& drdz = modulesInGPU.drdzs[lowerModuleIndices[i]];
//        slopes[i] = modulesInGPU.slopes[lowerModuleIndices[i]];
//        //category 1 - barrel PS flat
//        if(moduleSubdet == Barrel and moduleType == PS and moduleSide == Center)
//        {
//            //delta1[i] = 0.01;
//            //delta2[i] = 0.01;
//            delta1[i] = inv1;//1.1111f;//0.01;
//            delta2[i] = inv1;//1.1111f;//0.01;
//            slopes[i] = -999.f;
//            isFlat[i] = true;
//        }
//
//        //category 2 - barrel 2S
//        else if(moduleSubdet == Barrel and moduleType == TwoS)
//        {
//            //delta1[i] = 0.009;
//            //delta2[i] = 0.009;
//            delta1[i] = 1.f;//0.009;
//            delta2[i] = 1.f;//0.009;
//            slopes[i] = -999.f;
//            isFlat[i] = true;
//        }
//
//        //category 3 - barrel PS tilted
//        else if(moduleSubdet == Barrel and moduleType == PS and moduleSide != Center)
//        {
//
//            //delta1[i] = 0.01;
//            delta1[i] = inv1;//1.1111f;//0.01;
//            isFlat[i] = false;
//
//            if(anchorHits)
//            {
//                //delta2[i] = (0.15f * drdz/sqrtf(1 + drdz * drdz));
//                delta2[i] = (inv2 * drdz/sqrtf(1 + drdz * drdz));
//                //delta2[i] = (inv2 * drdz*rsqrt(1 + drdz * drdz));
//            }
//            else
//            {
//                //delta2[i] = (2.4f * drdz/sqrtf(1 + drdz * drdz));
//                delta2[i] = (inv3 * drdz/sqrtf(1 + drdz * drdz));
//                //delta2[i] = (inv3 * drdz*rsqrt(1 + drdz * drdz));
//            }
//        }
//        //category 4 - endcap PS
//        else if(moduleSubdet == Endcap and moduleType == PS)
//        {
//            delta1[i] = inv1;//1.1111f;//0.01;
//            //delta1[i] = 0.01;
//            isFlat[i] = false;
//
//            /*despite the type of the module layer of the lower module index,
//            all anchor hits are on the pixel side and all non-anchor hits are
//            on the strip side!*/
//            if(anchorHits)
//            {
//                delta2[i] = inv2;//16.6666f;//0.15f;
//                //delta2[i] = 0.15f;
//            }
//            else
//            {
//                //delta2[i] = 2.4f;
//                delta2[i] = inv3;//266.666f;//2.4f;
//            }
//        }
//
//        //category 5 - endcap 2S
//        else if(moduleSubdet == Endcap and moduleType == TwoS)
//        {
//            //delta1[i] = 0.009;
//            //delta2[i] = 5.f;
//            delta1[i] = 1.f;//0.009;
//            delta2[i] = 500.f*inv1;//555.5555f;//5.f;
//            isFlat[i] = false;
//        }
//        else
//        {
//            printf("ERROR!!!!! I SHOULDN'T BE HERE!!!! subdet = %d, type = %d, side = %d\n", moduleSubdet, moduleType, moduleSide);
//        }
//    }
//    //divide everyone by the smallest possible values of delta1 and delta2
////    for(size_t i = 0; i < 5; i++)
////    {
////        delta1[i] /= 0.009;
////        delta2[i] /= 0.009;
////    }
//}
//
//__global__ void SDL::createPixelQuintupletsInGPUFromMapv2(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::quintuplets& quintupletsInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, unsigned int* connectedPixelSize, unsigned int* connectedPixelIndex, unsigned int nPixelSegments, unsigned int* seg_pix_gpu, unsigned int* seg_pix_gpu_offset, unsigned int totalSegs, struct SDL::objectRanges& rangesInGPU)
//{
//    //unsigned int offsetIndex = blockIdx.x * blockDim.x + threadIdx.x;
//    int blockxSize = blockDim.x*gridDim.x;
//    int blockySize = blockDim.y*gridDim.y;
//    for(int offsetIndex = blockIdx.y * blockDim.y + threadIdx.y; offsetIndex< totalSegs; offsetIndex += blockySize)
//    {
//        int segmentModuleIndex = seg_pix_gpu_offset[offsetIndex];
//        int pixelSegmentArrayIndex = seg_pix_gpu[offsetIndex];
//        if(pixelSegmentArrayIndex >= nPixelSegments) continue;//return;
//        if(segmentModuleIndex >= connectedPixelSize[pixelSegmentArrayIndex]) continue;//return;
//
//        unsigned int tempIndex = connectedPixelIndex[pixelSegmentArrayIndex] + segmentModuleIndex; //gets module array index for segment
//
//    //these are actual module indices
//        uint16_t quintupletLowerModuleIndex = modulesInGPU.connectedPixels[tempIndex];
//        if(quintupletLowerModuleIndex >= *modulesInGPU.nLowerModules) continue;//return;
//
//        uint16_t pixelModuleIndex = *modulesInGPU.nLowerModules;
//        unsigned int nOuterQuintuplets = quintupletsInGPU.nQuintuplets[quintupletLowerModuleIndex];
//
//        if(nOuterQuintuplets == 0) continue;//return;
//
//        //fetch the quintuplet
//        for(unsigned int outerQuintupletArrayIndex = blockIdx.x * blockDim.x + threadIdx.x; outerQuintupletArrayIndex< nOuterQuintuplets; outerQuintupletArrayIndex +=blockxSize)
//        {
//            unsigned int pixelSegmentIndex = rangesInGPU.segmentModuleIndices[pixelModuleIndex] + pixelSegmentArrayIndex;
//
//            unsigned int quintupletIndex = rangesInGPU.quintupletModuleIndices[quintupletLowerModuleIndex] + outerQuintupletArrayIndex;
//
//            if(segmentsInGPU.isDup[pixelSegmentArrayIndex]) continue;//return;//skip duplicated pLS
//            if(quintupletsInGPU.isDup[quintupletIndex]) continue;//return; //skip duplicated T5s
//
//            float rzChiSquared, rPhiChiSquared, rPhiChiSquaredInwards, pixelRadius, quintupletRadius, centerX, centerY;
//
//            bool success = runPixelQuintupletDefaultAlgo(modulesInGPU, rangesInGPU, mdsInGPU, segmentsInGPU, tripletsInGPU, quintupletsInGPU, pixelSegmentIndex, quintupletIndex, rzChiSquared, rPhiChiSquared, rPhiChiSquaredInwards, pixelRadius, quintupletRadius, centerX, centerY);
//            if(success)
//            {
//                atomicAdd(pixelQuintupletsInGPU.totOccupancyPixelQuintuplets, 1);
//                if(*pixelQuintupletsInGPU.nPixelQuintuplets >= N_MAX_PIXEL_QUINTUPLETS)
//                {
//#ifdef Warnings
//                    printf("Pixel Quintuplet excess alert!\n");
//#endif
//                }
//                else
//                {
//                    unsigned int pixelQuintupletIndex = atomicAdd(pixelQuintupletsInGPU.nPixelQuintuplets, 1);
//                    float eta = __H2F(quintupletsInGPU.eta[quintupletIndex]);
//                    float phi = __H2F(quintupletsInGPU.phi[quintupletIndex]);
//
//#ifdef CUT_VALUE_DEBUG
//                    addPixelQuintupletToMemory(modulesInGPU, mdsInGPU, segmentsInGPU, quintupletsInGPU, pixelQuintupletsInGPU, pixelSegmentIndex, quintupletIndex, pixelQuintupletIndex,rzChiSquared, rPhiChiSquared, rPhiChiSquaredInwards, rPhiChiSquared, eta, phi, pixelRadius, quintupletRadius, centerX, centerY);
//
//#else
//                    addPixelQuintupletToMemory(modulesInGPU, mdsInGPU, segmentsInGPU, quintupletsInGPU, pixelQuintupletsInGPU, pixelSegmentIndex, quintupletIndex, pixelQuintupletIndex,rPhiChiSquaredInwards+rPhiChiSquared, eta,phi, pixelRadius, quintupletRadius, centerX, centerY);
//#endif
//                    tripletsInGPU.partOfPT5[quintupletsInGPU.tripletIndices[2 * quintupletIndex]] = true;
//                    tripletsInGPU.partOfPT5[quintupletsInGPU.tripletIndices[2 * quintupletIndex + 1]] = true;
//                    segmentsInGPU.partOfPT5[pixelSegmentArrayIndex] = true;
//                    quintupletsInGPU.partOfPT5[quintupletIndex] = true;
//                }
//            }
//        }
//    }
//}
//__device__ void SDL::runDeltaBetaIterationspT5(float& betaIn, float& betaOut, float& betaAv, float & pt_beta, float sdIn_dr, float sdOut_dr, float dr, float lIn)
//{
//    if (lIn == 0)
//    {
//        betaOut += copysign(asinf(fminf(sdOut_dr * SDL::k2Rinv1GeVf / fabsf(pt_beta), SDL::sinAlphaMax)), betaOut);
//        return;
//    }
//
//    if (betaIn * betaOut > 0.f and (fabsf(pt_beta) < 4.f * SDL::pt_betaMax or (lIn >= 11 and fabsf(pt_beta) < 8.f * SDL::pt_betaMax)))   //and the pt_beta is well-defined; less strict for endcap-endcap
//    {
//
//        const float betaInUpd  = betaIn + copysignf(asinf(fminf(sdIn_dr * SDL::k2Rinv1GeVf / fabsf(pt_beta), SDL::sinAlphaMax)), betaIn); //FIXME: need a faster version
//        const float betaOutUpd = betaOut + copysignf(asinf(fminf(sdOut_dr * SDL::k2Rinv1GeVf / fabsf(pt_beta), SDL::sinAlphaMax)), betaOut); //FIXME: need a faster version
//        betaAv = 0.5f * (betaInUpd + betaOutUpd);
//
//        //1st update
//        //pt_beta = dr * k2Rinv1GeVf / sinf(betaAv); //get a better pt estimate
//        const float pt_beta_inv = 1.f/fabsf(dr * k2Rinv1GeVf / sinf(betaAv)); //get a better pt estimate
//
//        betaIn  += copysignf(asinf(fminf(sdIn_dr * SDL::k2Rinv1GeVf *pt_beta_inv, SDL::sinAlphaMax)), betaIn); //FIXME: need a faster version
//        betaOut += copysignf(asinf(fminf(sdOut_dr * SDL::k2Rinv1GeVf *pt_beta_inv, SDL::sinAlphaMax)), betaOut); //FIXME: need a faster version
//        //update the av and pt
//        betaAv = 0.5f * (betaIn + betaOut);
//        //2nd update
//        pt_beta = dr * SDL::k2Rinv1GeVf / sinf(betaAv); //get a better pt estimate
//    }
//    else if (lIn < 11 && fabsf(betaOut) < 0.2f * fabsf(betaIn) && fabsf(pt_beta) < 12.f * SDL::pt_betaMax)   //use betaIn sign as ref
//    {
//
//        const float pt_betaIn = dr * k2Rinv1GeVf / sinf(betaIn);
//
//        const float betaInUpd  = betaIn + copysignf(asinf(fminf(sdIn_dr * SDL::k2Rinv1GeVf / fabsf(pt_betaIn), SDL::sinAlphaMax)), betaIn); //FIXME: need a faster version
//        const float betaOutUpd = betaOut + copysignf(asinf(fminf(sdOut_dr * SDL::k2Rinv1GeVf / fabsf(pt_betaIn), SDL::sinAlphaMax)), betaIn); //FIXME: need a faster version
//        betaAv = (fabsf(betaOut) > 0.2f * fabsf(betaIn)) ? (0.5f * (betaInUpd + betaOutUpd)) : betaInUpd;
//
//        //1st update
//        pt_beta = dr * SDL::k2Rinv1GeVf / sin(betaAv); //get a better pt estimate
//        betaIn  += copysignf(asinf(fminf(sdIn_dr * SDL::k2Rinv1GeVf / fabsf(pt_beta), SDL::sinAlphaMax)), betaIn); //FIXME: need a faster version
//        betaOut += copysignf(asinf(fminf(sdOut_dr * SDL::k2Rinv1GeVf / fabsf(pt_beta), SDL::sinAlphaMax)), betaIn); //FIXME: need a faster version
//        //update the av and pt
//        betaAv = 0.5f * (betaIn + betaOut);
//        //2nd update
//        pt_beta = dr * SDL::k2Rinv1GeVf / sin(betaAv); //get a better pt estimate
//
//    }
//}
//__device__ bool SDL::checkIntervalOverlappT5(const float& firstMin, const float& firstMax, const float& secondMin, const float& secondMax)
//{
//    return ((firstMin <= secondMin) & (secondMin < firstMax)) |  ((secondMin < firstMin) & (firstMin < secondMax));
//}
//
//__device__ float SDL::computeChiSquaredpT5(int nPoints, float* xs, float* ys, float* delta1, float* delta2, float* slopes, bool* isFlat, float g, float f, float radius)
//{
//    // given values of (g, f, radius) and a set of points (and its uncertainties)
//    //compute chi squared
//    float c = g*g + f*f - radius*radius;
//    float chiSquared = 0.f;
//    float absArctanSlope, angleM, xPrime, yPrime, sigma;
//    for(size_t i = 0; i < nPoints; i++)
//    {
//        absArctanSlope = ((slopes[i] != 123456789) ? fabs(atanf(slopes[i])) : 0.5f*float(M_PI)); // Since C++ can't represent infinity, SDL_INF = 123456789 was used to represent infinity in the data table
//        if(xs[i] > 0 and ys[i] > 0)
//        {
//            angleM = 0.5f*float(M_PI) - absArctanSlope;
//        }
//        else if(xs[i] < 0 and ys[i] > 0)
//        {
//            angleM = absArctanSlope + 0.5f*float(M_PI);
//        }
//        else if(xs[i] < 0 and ys[i] < 0)
//        {
//            angleM = -(absArctanSlope + 0.5f*float(M_PI));
//        }
//        else if(xs[i] > 0 and ys[i] < 0)
//        {
//            angleM = -(0.5f*float(M_PI) - absArctanSlope);
//        }
//
//        if(not isFlat[i])
//        {
//            xPrime = xs[i] * cosf(angleM) + ys[i] * sinf(angleM);
//            yPrime = ys[i] * cosf(angleM) - xs[i] * sinf(angleM);
//        }
//        else
//        {
//            xPrime = xs[i];
//            yPrime = ys[i];
//        }
//        sigma = 2 * sqrtf((xPrime * delta1[i]) * (xPrime * delta1[i]) + (yPrime * delta2[i]) * (yPrime * delta2[i]));
//        chiSquared +=  (xs[i] * xs[i] + ys[i] * ys[i] - 2 * g * xs[i] - 2 * f * ys[i] + c) * (xs[i] * xs[i] + ys[i] * ys[i] - 2 * g * xs[i] - 2 * f * ys[i] + c) / (sigma * sigma);
//    }
//    return chiSquared;
//}
//
//__device__ bool inline SDL::runpT5DefaultAlgoPPBB(struct modules& modulesInGPU, struct objectRanges& rangesInGPU, struct miniDoublets& mdsInGPU ,struct segments& segmentsInGPU, uint16_t& pixelModuleIndex, uint16_t& outerInnerLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int thirdMDIndex, unsigned int& fourthMDIndex, float& z_OutLo, float& rt_OutLo, float& dPhiPos, float& dPhi, float& betaIn,
//        float& betaOut, float& pt_beta, float& zLo, float& zHi, float& zLoPointed, float& zHiPointed, float& sdlCut, float& betaOutCut, float& deltaBetaCut) // pixel to BB and BE segments
//{
//    bool pass = true;
//
//    bool isPS_OutLo = (modulesInGPU.moduleType[outerInnerLowerModuleIndex] == SDL::PS);
//
//    float rt_InLo = mdsInGPU.anchorRt[firstMDIndex];
//    float rt_InUp = mdsInGPU.anchorRt[secondMDIndex];
//    rt_OutLo = mdsInGPU.anchorRt[thirdMDIndex];
//    float rt_OutUp = mdsInGPU.anchorRt[fourthMDIndex];
//
//    float z_InLo = mdsInGPU.anchorZ[firstMDIndex];
//    float z_InUp = mdsInGPU.anchorZ[secondMDIndex];
//    z_OutLo = mdsInGPU.anchorZ[thirdMDIndex];
//    float z_OutUp = mdsInGPU.anchorZ[fourthMDIndex];
//
//    float x_InLo = mdsInGPU.anchorX[firstMDIndex];
//    float x_InUp = mdsInGPU.anchorX[secondMDIndex];
//    float x_OutLo = mdsInGPU.anchorX[thirdMDIndex];
//    float x_OutUp = mdsInGPU.anchorX[fourthMDIndex];
//
//    float y_InLo = mdsInGPU.anchorY[firstMDIndex];
//    float y_InUp = mdsInGPU.anchorY[secondMDIndex];
//    float y_OutLo = mdsInGPU.anchorY[thirdMDIndex];
//    float y_OutUp = mdsInGPU.anchorY[fourthMDIndex];
//
//    float& rt_InOut = rt_InUp;
//    //float& z_InOut = z_InUp;
//
//    pass = pass and (fabsf(deltaPhi(x_InUp, y_InUp, z_InUp, x_OutLo, y_OutLo, z_OutLo)) <= 0.5f * float(M_PI));
//    if(not pass) return pass;
//
//    unsigned int pixelSegmentArrayIndex = innerSegmentIndex - rangesInGPU.segmentModuleIndices[pixelModuleIndex];
//    float ptIn = segmentsInGPU.ptIn[pixelSegmentArrayIndex];
//    float ptSLo = ptIn;
//    float px = segmentsInGPU.px[pixelSegmentArrayIndex];
//    float py = segmentsInGPU.py[pixelSegmentArrayIndex];
//    float pz = segmentsInGPU.pz[pixelSegmentArrayIndex];
//    float ptErr = segmentsInGPU.ptErr[pixelSegmentArrayIndex];
//    float etaErr = segmentsInGPU.etaErr[pixelSegmentArrayIndex];
//    ptSLo = fmaxf(ptCut, ptSLo - 10.0f*fmaxf(ptErr, 0.005f*ptSLo));
//    ptSLo = fminf(10.0f, ptSLo);
//
//
//    float alpha1GeV_OutLo = asinf(fminf(rt_OutLo * k2Rinv1GeVf / ptCut, sinAlphaMax));
//    //float rtRatio_OutLoInLo = rt_OutLo / rt_InLo; // Outer segment beginning rt divided by inner segment beginning rt;
//    const float rtRatio_OutLoInOut = rt_OutLo / rt_InOut; // Outer segment beginning rt divided by inner segment beginning rt;
//
//    float dzDrtScale = tanf(alpha1GeV_OutLo) / alpha1GeV_OutLo; // The track can bend in r-z plane slightly
//    const float zpitch_InLo = 0.05f;
//    const float zpitch_InOut = 0.05f;
//    float zpitch_OutLo = (isPS_OutLo ? pixelPSZpitch : strip2SZpitch);
//    float zGeom = zpitch_InLo + zpitch_OutLo;
//    zHi = z_InUp + (z_InUp + deltaZLum) * (rtRatio_OutLoInOut - 1.f) * (z_InUp < 0.f ? 1.f : dzDrtScale) + (zpitch_InOut + zpitch_OutLo);
//    zLo = z_InUp + (z_InUp - deltaZLum) * (rtRatio_OutLoInOut - 1.f) * (z_InUp > 0.f ? 1.f : dzDrtScale) - (zpitch_InOut + zpitch_OutLo); //slope-correction only on outer end
//
//    pass = pass and ((z_OutLo >= zLo) & (z_OutLo <= zHi));
//    if(not pass) return pass;
//
//    const float coshEta = sqrtf(ptIn * ptIn + pz * pz) / ptIn;
//    // const float drt_OutLo_InLo = (rt_OutLo - rt_InLo);
//    const float drt_OutLo_InUp = (rt_OutLo - rt_InUp);
//    //const float invRt_InLo = 1.f / rt_InLo;
//    //const float r3_InLo = sqrtf(z_InLo * z_InLo + rt_InLo * rt_InLo);
//    const float r3_InUp = sqrtf(z_InUp * z_InUp + rt_InUp * rt_InUp);
//
//    float drt_InSeg = rt_InOut - rt_InLo;
//    //float dz_InSeg = z_InOut - z_InLo;
//    //float dr3_InSeg = sqrtf(rt_InOut * rt_InOut + z_InOut * z_InOut) - sqrtf(rt_InLo * rt_InLo + z_InLo * z_InLo);
//
//    const float sdlThetaMulsF = 0.015f * sqrtf(0.1f + 0.2f * (rt_OutLo - rt_InUp) / 50.f) * sqrtf(r3_InUp / rt_InUp);
//    const float sdlMuls = sdlThetaMulsF * 3.f / ptCut * 4.f; // will need a better guess than x4?
//
//    float dzErr = drt_OutLo_InUp*etaErr*coshEta; //FIXME: check with the calc in the endcap
//    dzErr *= dzErr;
//    dzErr += 0.03f*0.03f; // pixel size x2. ... random for now
//    dzErr *= 9.f; //3 sigma
//    dzErr += sdlMuls*sdlMuls*drt_OutLo_InUp*drt_OutLo_InUp/3.f*coshEta*coshEta;//sloppy
//    dzErr += zGeom*zGeom;
//    dzErr = sqrtf(dzErr);
//
//    const float dzDrIn = pz / ptIn;
//    const float zWindow = dzErr / drt_InSeg * drt_OutLo_InUp + zGeom;
//    const float dzMean = dzDrIn * drt_OutLo_InUp *
//        (1.f + drt_OutLo_InUp * drt_OutLo_InUp * 4 * k2Rinv1GeVf * k2Rinv1GeVf / ptIn /
//         ptIn / 24.f); // with curved path correction
//    // Constructing upper and lower bound
//    zLoPointed = z_InUp + dzMean - zWindow;
//    zHiPointed = z_InUp + dzMean + zWindow;
//
//    pass =  pass and ((z_OutLo >= zLoPointed) & (z_OutLo <= zHiPointed));
//    if(not pass) return pass;
//
//    const float sdlPVoff = 0.1f / rt_OutLo;
//    sdlCut = alpha1GeV_OutLo + sqrtf(sdlMuls * sdlMuls + sdlPVoff * sdlPVoff);
//
//    dPhiPos = deltaPhi(x_InUp, y_InUp, z_InUp, x_OutUp, y_OutUp, z_OutUp);
//
//    //no dphipos cut
//    float midPointX = 0.5f * (x_InLo + x_OutLo);
//    float midPointY = 0.5f * (y_InLo + y_OutLo);
//    float midPointZ = 0.5f * (z_InLo + z_OutLo);
//
//    float diffX = x_OutLo - x_InLo;
//    float diffY = y_OutLo - y_InLo;
//    float diffZ = z_OutLo - z_InLo;
//
//
//    dPhi = deltaPhi(midPointX, midPointY, midPointZ, diffX, diffY, diffZ);
//
//    pass = pass and (fabsf(dPhi) <= sdlCut);
//    if(not pass) return pass;
//
//    //lots of array accesses below this...
//
//    float alpha_InLo  = __H2F(segmentsInGPU.dPhiChanges[innerSegmentIndex]);
//    float alpha_OutLo = __H2F(segmentsInGPU.dPhiChanges[outerSegmentIndex]);
//
//    bool isEC_lastLayer = modulesInGPU.subdets[outerOuterLowerModuleIndex] == SDL::Endcap and modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::TwoS;
//
//    float alpha_OutUp,alpha_OutUp_highEdge,alpha_OutUp_lowEdge;
//    alpha_OutUp = deltaPhi(x_OutUp, y_OutUp, z_OutUp, x_OutUp - x_OutLo, y_OutUp - y_OutLo, z_OutUp - z_OutLo);
//
//    alpha_OutUp_highEdge = alpha_OutUp;
//    alpha_OutUp_lowEdge = alpha_OutUp;
//
//    float tl_axis_x = x_OutUp - x_InUp;
//    float tl_axis_y = y_OutUp - y_InUp;
//    float tl_axis_z = z_OutUp - z_InUp;
//
//    float tl_axis_highEdge_x = tl_axis_x;
//    float tl_axis_highEdge_y = tl_axis_y;
//
//    float tl_axis_lowEdge_x = tl_axis_x;
//    float tl_axis_lowEdge_y = tl_axis_y;
//
//    betaIn = -deltaPhi(px, py, pz, tl_axis_x, tl_axis_y, tl_axis_z);
//    float betaInRHmin = betaIn;
//    float betaInRHmax = betaIn;
//
//    betaOut = -alpha_OutUp + deltaPhi(x_OutUp, y_OutUp, z_OutUp, tl_axis_x, tl_axis_y, tl_axis_z);
//
//    float betaOutRHmin = betaOut;
//    float betaOutRHmax = betaOut;
//
//    if(isEC_lastLayer)
//    {
//        alpha_OutUp_highEdge = deltaPhi(mdsInGPU.anchorHighEdgeX[fourthMDIndex], mdsInGPU.anchorHighEdgeY[fourthMDIndex], z_OutUp, mdsInGPU.anchorHighEdgeX[fourthMDIndex] - x_OutLo, mdsInGPU.anchorHighEdgeY[fourthMDIndex] - y_OutLo, z_OutUp - z_OutLo);
//        alpha_OutUp_lowEdge = deltaPhi(mdsInGPU.anchorLowEdgeX[fourthMDIndex], mdsInGPU.anchorLowEdgeY[fourthMDIndex], z_OutUp, mdsInGPU.anchorLowEdgeX[fourthMDIndex] - x_OutLo, mdsInGPU.anchorLowEdgeY[fourthMDIndex] - y_OutLo, z_OutUp - z_OutLo);
//
//        tl_axis_highEdge_x = mdsInGPU.anchorHighEdgeX[fourthMDIndex] - x_InUp;
//        tl_axis_highEdge_y = mdsInGPU.anchorHighEdgeY[fourthMDIndex] - y_InUp;
//        tl_axis_lowEdge_x = mdsInGPU.anchorLowEdgeX[fourthMDIndex] - x_InUp;
//        tl_axis_lowEdge_y = mdsInGPU.anchorLowEdgeY[fourthMDIndex] - y_InUp;
//
//        betaOutRHmin = -alpha_OutUp_highEdge + deltaPhi(mdsInGPU.anchorHighEdgeX[fourthMDIndex], mdsInGPU.anchorHighEdgeY[fourthMDIndex], z_OutUp, tl_axis_highEdge_x, tl_axis_highEdge_y, tl_axis_z);
//        betaOutRHmax = -alpha_OutUp_lowEdge + deltaPhi(mdsInGPU.anchorLowEdgeX[fourthMDIndex], mdsInGPU.anchorLowEdgeY[fourthMDIndex], z_OutUp, tl_axis_lowEdge_x, tl_axis_lowEdge_y, tl_axis_z);
//    }
//
//    //beta computation
//    float drt_tl_axis = sqrtf(tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);
//    //float drt_tl_lowEdge = sqrtf(tl_axis_lowEdge_x * tl_axis_lowEdge_x + tl_axis_lowEdge_y * tl_axis_lowEdge_y);
//    //float drt_tl_highEdge = sqrtf(tl_axis_highEdge_x * tl_axis_highEdge_x + tl_axis_highEdge_y * tl_axis_highEdge_y);
//
//    //innerOuterAnchor - innerInnerAnchor
//    const float rt_InSeg = sqrtf((x_InUp - x_InLo) * (x_InUp - x_InLo) + (y_InUp - y_InLo) * (y_InUp - y_InLo));
//
//    //no betaIn cut for the pixels
//    float betaAv = 0.5f * (betaIn + betaOut);
//    pt_beta = ptIn;
//
//    const float pt_betaMax = 7.0f;
//
//    int lIn = 0;
//    int lOut = isEC_lastLayer ? 11 : 5;
//    float sdOut_dr = sqrtf((x_OutUp - x_OutLo) * (x_OutUp - x_OutLo) + (y_OutUp - y_OutLo) * (y_OutUp - y_OutLo));
//    float sdOut_d = rt_OutUp - rt_OutLo;
//
//    //const float diffDr = fabsf(rt_InSeg - sdOut_dr) / fabsf(rt_InSeg + sdOut_dr);
//
//    runDeltaBetaIterationspT5(betaIn, betaOut, betaAv, pt_beta, rt_InSeg, sdOut_dr, drt_tl_axis, lIn);
//
//    const float betaInMMSF = (fabsf(betaInRHmin + betaInRHmax) > 0) ? (2.f * betaIn / fabsf(betaInRHmin + betaInRHmax)) : 0.; //mean value of min,max is the old betaIn
//    const float betaOutMMSF = (fabsf(betaOutRHmin + betaOutRHmax) > 0) ? (2.f * betaOut / fabsf(betaOutRHmin + betaOutRHmax)) : 0.;
//    betaInRHmin *= betaInMMSF;
//    betaInRHmax *= betaInMMSF;
//    betaOutRHmin *= betaOutMMSF;
//    betaOutRHmax *= betaOutMMSF;
//
//    const float dBetaMuls = sdlThetaMulsF * 4.f / fminf(fabsf(pt_beta), pt_betaMax); //need to confirm the range-out value of 7 GeV
//    const float alphaInAbsReg =  fmaxf(fabsf(alpha_InLo), asinf(fminf(rt_InUp * k2Rinv1GeVf / 3.0f, sinAlphaMax)));
//    const float alphaOutAbsReg = fmaxf(fabsf(alpha_OutLo), asinf(fminf(rt_OutLo * k2Rinv1GeVf / 3.0f, sinAlphaMax)));
//    const float dBetaInLum = lIn < 11 ? 0.0f : fabsf(alphaInAbsReg*deltaZLum / z_InUp);
//    const float dBetaOutLum = lOut < 11 ? 0.0f : fabsf(alphaOutAbsReg*deltaZLum / z_OutLo);
//    const float dBetaLum2 = (dBetaInLum + dBetaOutLum) * (dBetaInLum + dBetaOutLum);
//
//    const float sinDPhi = sinf(dPhi);
//    const float dBetaRIn2 = 0; // TODO-RH
//
//    float dBetaROut = 0;
//    if(isEC_lastLayer)
//    {
//        dBetaROut = (sqrtf(mdsInGPU.anchorHighEdgeX[fourthMDIndex] * mdsInGPU.anchorHighEdgeX[fourthMDIndex] + mdsInGPU.anchorHighEdgeY[fourthMDIndex] * mdsInGPU.anchorHighEdgeY[fourthMDIndex]) - sqrtf(mdsInGPU.anchorLowEdgeX[fourthMDIndex] * mdsInGPU.anchorLowEdgeX[fourthMDIndex] + mdsInGPU.anchorLowEdgeY[fourthMDIndex] * mdsInGPU.anchorLowEdgeY[fourthMDIndex])) * sinDPhi / drt_tl_axis;
//    }
//
//    const float dBetaROut2 =  dBetaROut * dBetaROut;
//
//    betaOutCut = asinf(fminf(drt_tl_axis*k2Rinv1GeVf / ptCut, sinAlphaMax)) //FIXME: need faster version
//        + (0.02f / sdOut_d) + sqrtf(dBetaLum2 + dBetaMuls*dBetaMuls);
//
//    //Cut #6: The real beta cut
//    pass = pass and (fabsf(betaOut) < betaOutCut);
//    if(not pass) return pass;
//
//    //const float pt_betaOut = drt_tl_axis * k2Rinv1GeVf / sin(betaOut);
//    const float dBetaRes = 0.02f / fminf(sdOut_d, drt_InSeg);
//    const float dBetaCut2 = (dBetaRes*dBetaRes * 2.0f + dBetaMuls * dBetaMuls + dBetaLum2 + dBetaRIn2 + dBetaROut2
//            + 0.25f * (fabsf(betaInRHmin - betaInRHmax) + fabsf(betaOutRHmin - betaOutRHmax)) * (fabsf(betaInRHmin - betaInRHmax) + fabsf(betaOutRHmin - betaOutRHmax)));
//    float dBeta = betaIn - betaOut;
//    deltaBetaCut = sqrtf(dBetaCut2);
//
//    pass = pass and (dBeta * dBeta <= dBetaCut2);
//
//    return pass;
//}
//
//__device__ bool inline SDL::runpT5DefaultAlgoPPEE(struct modules& modulesInGPU, struct objectRanges& rangesInGPU, struct miniDoublets& mdsInGPU ,struct segments& segmentsInGPU, uint16_t& pixelModuleIndex, uint16_t& outerInnerLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex, unsigned int& fourthMDIndex, float& z_OutLo, float& rt_OutLo, float& deltaPhiPos, float& dPhi, float& betaIn,
//        float& betaOut, float& pt_beta, float& zLo, float& rtLo, float& rtHi, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ) // pixel to EE segments
//{
//    bool pass = true;
//    bool isPS_OutLo = (modulesInGPU.moduleType[outerInnerLowerModuleIndex] == SDL::PS);
//
//
//    float z_InLo = mdsInGPU.anchorZ[firstMDIndex];
//    float z_InUp = mdsInGPU.anchorZ[secondMDIndex];
//    z_OutLo = mdsInGPU.anchorZ[thirdMDIndex];
//    float z_OutUp = mdsInGPU.anchorZ[fourthMDIndex];
//
//    pass =  pass and (z_InUp * z_OutLo > 0);
//    if(not pass) return pass;
//
//    float rt_InLo = mdsInGPU.anchorRt[firstMDIndex];
//    float rt_InUp = mdsInGPU.anchorRt[secondMDIndex];
//    rt_OutLo = mdsInGPU.anchorRt[thirdMDIndex];
//    float rt_OutUp = mdsInGPU.anchorRt[fourthMDIndex];
//
//    float x_InLo = mdsInGPU.anchorX[firstMDIndex];
//    float x_InUp = mdsInGPU.anchorX[secondMDIndex];
//    float x_OutLo = mdsInGPU.anchorX[thirdMDIndex];
//    float x_OutUp = mdsInGPU.anchorX[fourthMDIndex];
//
//    float y_InLo = mdsInGPU.anchorY[firstMDIndex];
//    float y_InUp = mdsInGPU.anchorY[secondMDIndex];
//    float y_OutLo = mdsInGPU.anchorY[thirdMDIndex];
//    float y_OutUp = mdsInGPU.anchorY[fourthMDIndex];
//
//    unsigned int pixelSegmentArrayIndex = innerSegmentIndex - rangesInGPU.segmentModuleIndices[pixelModuleIndex];
//
//    float ptIn = segmentsInGPU.ptIn[pixelSegmentArrayIndex];
//    float ptSLo = ptIn;
//    float px = segmentsInGPU.px[pixelSegmentArrayIndex];
//    float py = segmentsInGPU.py[pixelSegmentArrayIndex];
//    float pz = segmentsInGPU.pz[pixelSegmentArrayIndex];
//    float ptErr = segmentsInGPU.ptErr[pixelSegmentArrayIndex];
//    float etaErr = segmentsInGPU.etaErr[pixelSegmentArrayIndex];
//
//    ptSLo = fmaxf(ptCut, ptSLo - 10.0f*fmaxf(ptErr, 0.005f*ptSLo));
//    ptSLo = fminf(10.0f, ptSLo);
//
//    float rtOut_o_rtIn = rt_OutLo/rt_InUp;
//    const float zpitch_InLo = 0.05f;
//    float zpitch_OutLo = (isPS_OutLo ? pixelPSZpitch : strip2SZpitch);
//    float zGeom = zpitch_InLo + zpitch_OutLo;
//
//    const float sdlSlope = asinf(fminf(rt_OutLo * k2Rinv1GeVf / ptCut, sinAlphaMax));
//    const float dzDrtScale = tanf(sdlSlope) / sdlSlope;//FIXME: need approximate value
//    zLo = z_InUp + (z_InUp - deltaZLum) * (rtOut_o_rtIn - 1.f) * (z_InUp > 0.f ? 1.f : dzDrtScale) - zGeom; //slope-correction only on outer end
//
//
//    const float dLum = copysignf(deltaZLum, z_InUp);
//    bool isOutSgInnerMDPS = modulesInGPU.moduleType[outerInnerLowerModuleIndex] == SDL::PS;
//
//    const float rtGeom1 = isOutSgInnerMDPS ? pixelPSZpitch : strip2SZpitch;//FIXME: make this chosen by configuration for lay11,12 full PS
//    const float zGeom1 = copysignf(zGeom, z_InUp); //used in B-E region
//    rtLo = rt_InUp * (1.f + (z_OutLo- z_InUp - zGeom1) / (z_InUp + zGeom1 + dLum) / dzDrtScale) - rtGeom1; //slope correction only on the lower end
//
//
//    float zInForHi = z_InUp - zGeom1 - dLum;
//    if (zInForHi * z_InUp < 0)
//        zInForHi = copysignf(0.1f, z_InUp);
//    rtHi = rt_InUp * (1.f + (z_OutLo - z_InUp + zGeom1) / zInForHi) + rtGeom1;
//
//    // Cut #2: rt condition
//    pass =  pass and ((rt_OutLo >= rtLo) & (rt_OutLo <= rtHi));
//    if(not pass) return pass;
//
//    const float dzOutInAbs = fabsf(z_OutLo - z_InUp);
//    const float coshEta = hypotf(ptIn, pz) / ptIn;
//    const float multDzDr = dzOutInAbs*coshEta/(coshEta*coshEta - 1.f);
//    const float r3_InUp = sqrtf(z_InUp * z_InUp + rt_InUp * rt_InUp);
//    const float sdlThetaMulsF = 0.015f * sqrtf(0.1f + 0.2f * (rt_OutLo - rt_InUp) / 50.f) * sqrtf(r3_InUp / rt_InUp);
//    const float sdlMuls = sdlThetaMulsF * 3.f / ptCut * 4.f; // will need a better guess than x4?
//
//    float drtErr = etaErr*multDzDr;
//    drtErr *= drtErr;
//    drtErr += 0.03f*0.03f; // pixel size x2. ... random for now
//    drtErr *= 9.f; //3 sigma
//    drtErr += sdlMuls*sdlMuls*multDzDr*multDzDr/3.f*coshEta*coshEta;//sloppy: relative muls is 1/3 of total muls
//    drtErr = sqrtf(drtErr);
//    const float drtDzIn = fabsf(ptIn / pz);//all tracks are out-going in endcaps?
//
//    const float drt_OutLo_InUp = (rt_OutLo - rt_InUp); // drOutIn
//
//    const float rtWindow = drtErr + rtGeom1;
//    const float drtMean = drtDzIn * dzOutInAbs *
//        (1.f - drt_OutLo_InUp * drt_OutLo_InUp * 4 * k2Rinv1GeVf * k2Rinv1GeVf / ptIn /
//         ptIn / 24.f); // with curved path correction
//    const float rtLo_point = rt_InUp + drtMean - rtWindow;
//    const float rtHi_point = rt_InUp + drtMean + rtWindow;
//
//    // Cut #3: rt-z pointed
//    pass =  pass and ((rt_OutLo >= rtLo_point) & (rt_OutLo <= rtHi_point));
//    if(not pass) return pass;
//
//    const float alpha1GeV_OutLo = asinf(fminf(rt_OutLo * k2Rinv1GeVf / ptCut, sinAlphaMax));
//    const float sdlPVoff = 0.1f / rt_OutLo;
//    sdlCut = alpha1GeV_OutLo + sqrtf(sdlMuls * sdlMuls + sdlPVoff * sdlPVoff);
//
//    deltaPhiPos = deltaPhi(x_InUp, y_InUp, z_InUp, x_OutUp, y_OutUp, z_OutUp);
//
//    float midPointX = 0.5f * (x_InLo + x_OutLo);
//    float midPointY = 0.5f * (y_InLo + y_OutLo);
//    float midPointZ = 0.5f * (z_InLo + z_OutLo);
//
//    float diffX = x_OutLo - x_InLo;
//    float diffY = y_OutLo - y_InLo;
//    float diffZ = z_OutLo - z_InLo;
//
//    dPhi = deltaPhi(midPointX, midPointY, midPointZ, diffX, diffY, diffZ);
//
//    // Cut #5: deltaPhiChange
//    pass =  pass and (fabsf(dPhi) <= sdlCut);
//    if(not pass) return pass;
//
//    float alpha_InLo  = __H2F(segmentsInGPU.dPhiChanges[innerSegmentIndex]);
//    float alpha_OutLo = __H2F(segmentsInGPU.dPhiChanges[outerSegmentIndex]);
//
//    bool isEC_lastLayer = modulesInGPU.subdets[outerOuterLowerModuleIndex] == SDL::Endcap and modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::TwoS;
//
//    float alpha_OutUp,alpha_OutUp_highEdge,alpha_OutUp_lowEdge;
//
//    alpha_OutUp = deltaPhi(x_OutUp, y_OutUp, z_OutUp, x_OutUp - x_OutLo, y_OutUp - y_OutLo, z_OutUp - z_OutLo);
//    alpha_OutUp_highEdge = alpha_OutUp;
//    alpha_OutUp_lowEdge = alpha_OutUp;
//
//    float tl_axis_x = x_OutUp - x_InUp;
//    float tl_axis_y = y_OutUp - y_InUp;
//    float tl_axis_z = z_OutUp - z_InUp;
//
//    float tl_axis_highEdge_x = tl_axis_x;
//    float tl_axis_highEdge_y = tl_axis_y;
//
//    float tl_axis_lowEdge_x = tl_axis_x;
//    float tl_axis_lowEdge_y = tl_axis_y;
//
//    betaIn = -deltaPhi(px, py, pz, tl_axis_x, tl_axis_y, tl_axis_z);
//    float betaInRHmin = betaIn;
//    float betaInRHmax = betaIn;
//
//    betaOut = -alpha_OutUp + deltaPhi(x_OutUp, y_OutUp, z_OutUp, tl_axis_x, tl_axis_y, tl_axis_z);
//    float betaOutRHmin = betaOut;
//    float betaOutRHmax = betaOut;
//
//    if(isEC_lastLayer)
//    {
//
//        alpha_OutUp_highEdge = deltaPhi(mdsInGPU.anchorHighEdgeX[fourthMDIndex], mdsInGPU.anchorHighEdgeY[fourthMDIndex], z_OutUp, mdsInGPU.anchorHighEdgeX[fourthMDIndex] - x_OutLo, mdsInGPU.anchorHighEdgeY[fourthMDIndex] - y_OutLo, z_OutUp - z_OutLo);
//        alpha_OutUp_lowEdge = deltaPhi(mdsInGPU.anchorLowEdgeX[fourthMDIndex], mdsInGPU.anchorLowEdgeY[fourthMDIndex], z_OutUp, mdsInGPU.anchorLowEdgeX[fourthMDIndex] - x_OutLo, mdsInGPU.anchorLowEdgeY[fourthMDIndex] - y_OutLo, z_OutUp - z_OutLo);
//
//        tl_axis_highEdge_x = mdsInGPU.anchorHighEdgeX[fourthMDIndex] - x_InUp;
//        tl_axis_highEdge_y = mdsInGPU.anchorHighEdgeY[fourthMDIndex] - y_InUp;
//        tl_axis_lowEdge_x = mdsInGPU.anchorLowEdgeX[fourthMDIndex] - x_InUp;
//        tl_axis_lowEdge_y = mdsInGPU.anchorLowEdgeY[fourthMDIndex] - y_InUp;
//
//        betaOutRHmin = -alpha_OutUp_highEdge + deltaPhi(mdsInGPU.anchorHighEdgeX[fourthMDIndex], mdsInGPU.anchorHighEdgeY[fourthMDIndex], z_OutUp, tl_axis_highEdge_x, tl_axis_highEdge_y, tl_axis_z);
//        betaOutRHmax = -alpha_OutUp_lowEdge + deltaPhi(mdsInGPU.anchorLowEdgeX[fourthMDIndex], mdsInGPU.anchorLowEdgeY[fourthMDIndex], z_OutUp, tl_axis_lowEdge_x, tl_axis_lowEdge_y, tl_axis_z);
//    }
//
//    //beta computation
//    float drt_tl_axis = sqrtf(tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);
//    //float drt_tl_lowEdge = sqrtf(tl_axis_lowEdge_x * tl_axis_lowEdge_x + tl_axis_lowEdge_y * tl_axis_lowEdge_y);
//    //float drt_tl_highEdge = sqrtf(tl_axis_highEdge_x * tl_axis_highEdge_x + tl_axis_highEdge_y * tl_axis_highEdge_y);
////no betaIn cut for the pixels
//    const float rt_InSeg = sqrtf((x_InUp - x_InLo) * (x_InUp - x_InLo) + (y_InUp - y_InLo) * (y_InUp - y_InLo));
//
//    float betaAv = 0.5f * (betaIn + betaOut);
//    pt_beta = ptIn;
//
//    const float pt_betaMax = 7.0f;
//
//    int lIn = 0;
//    int lOut = isEC_lastLayer ? 11 : 5;
//    float sdOut_dr = sqrtf((x_OutUp - x_OutLo) * (x_OutUp - x_OutLo) + (y_OutUp - y_OutLo) * (y_OutUp - y_OutLo));
//    float sdOut_d = rt_OutUp - rt_OutLo;
//
//    //const float diffDr = fabsf(rt_InSeg - sdOut_dr) / fabsf(rt_InSeg + sdOut_dr);
//
//    runDeltaBetaIterationspT5(betaIn, betaOut, betaAv, pt_beta, rt_InSeg, sdOut_dr, drt_tl_axis, lIn);
//
//     const float betaInMMSF = (fabsf(betaInRHmin + betaInRHmax) > 0) ? (2.f * betaIn / fabsf(betaInRHmin + betaInRHmax)) : 0.; //mean value of min,max is the old betaIn
//    const float betaOutMMSF = (fabsf(betaOutRHmin + betaOutRHmax) > 0) ? (2.f * betaOut / fabsf(betaOutRHmin + betaOutRHmax)) : 0.;
//    betaInRHmin *= betaInMMSF;
//    betaInRHmax *= betaInMMSF;
//    betaOutRHmin *= betaOutMMSF;
//    betaOutRHmax *= betaOutMMSF;
//
//    const float dBetaMuls = sdlThetaMulsF * 4.f / fminf(fabsf(pt_beta), pt_betaMax); //need to confirm the range-out value of 7 GeV
//
//    const float alphaInAbsReg =  fmaxf(fabsf(alpha_InLo), asinf(fminf(rt_InUp * k2Rinv1GeVf / 3.0f, sinAlphaMax)));
//    const float alphaOutAbsReg = fmaxf(fabsf(alpha_OutLo), asinf(fminf(rt_OutLo * k2Rinv1GeVf / 3.0f, sinAlphaMax)));
//    const float dBetaInLum = lIn < 11 ? 0.0f : fabsf(alphaInAbsReg*deltaZLum / z_InUp);
//    const float dBetaOutLum = lOut < 11 ? 0.0f : fabsf(alphaOutAbsReg*deltaZLum / z_OutLo);
//    const float dBetaLum2 = (dBetaInLum + dBetaOutLum) * (dBetaInLum + dBetaOutLum);
//
//    const float sinDPhi = sinf(dPhi);
//    const float dBetaRIn2 = 0; // TODO-RH
//
//    float dBetaROut = 0;
//    if(isEC_lastLayer)
//    {
//        dBetaROut = (sqrtf(mdsInGPU.anchorHighEdgeX[fourthMDIndex] * mdsInGPU.anchorHighEdgeX[fourthMDIndex] + mdsInGPU.anchorHighEdgeY[fourthMDIndex] * mdsInGPU.anchorHighEdgeY[fourthMDIndex]) - sqrtf(mdsInGPU.anchorLowEdgeX[fourthMDIndex] * mdsInGPU.anchorLowEdgeX[fourthMDIndex] + mdsInGPU.anchorLowEdgeY[fourthMDIndex] * mdsInGPU.anchorLowEdgeY[fourthMDIndex])) * sinDPhi / drt_tl_axis;
//    }
//
//    const float dBetaROut2 =  dBetaROut * dBetaROut;
//
//    betaOutCut = asinf(fminf(drt_tl_axis*k2Rinv1GeVf / ptCut, sinAlphaMax)) //FIXME: need faster version
//        + (0.02f / sdOut_d) + sqrtf(dBetaLum2 + dBetaMuls*dBetaMuls);
//
//    //Cut #6: The real beta cut
//    pass =  pass and (fabsf(betaOut) < betaOutCut);
//    if(not pass) return pass;
//
//   // const float pt_betaOut = drt_tl_axis * k2Rinv1GeVf / sin(betaOut);
//    float drt_InSeg = rt_InUp - rt_InLo;
//
//    const float dBetaRes = 0.02f / fminf(sdOut_d, drt_InSeg);
//    const float dBetaCut2 = (dBetaRes*dBetaRes * 2.0f + dBetaMuls * dBetaMuls + dBetaLum2 + dBetaRIn2 + dBetaROut2
//            + 0.25f * (fabsf(betaInRHmin - betaInRHmax) + fabsf(betaOutRHmin - betaOutRHmax)) * (fabsf(betaInRHmin - betaInRHmax) + fabsf(betaOutRHmin - betaOutRHmax)));
//    float dBeta = betaIn - betaOut;
//    deltaBetaCut = sqrtf(dBetaCut2);
//
//    pass =  pass and (dBeta * dBeta <= dBetaCut2);
//    return pass;
//}
//
//__device__ bool inline SDL::runpT5DefaultAlgo(struct modules& modulesInGPU, struct objectRanges& rangesInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, uint16_t& pixelLowerModuleIndex, uint16_t& outerInnerLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& deltaPhi, float& betaIn, float& betaOut, float& pt_beta, float& zLo, float& zHi, float& rtLo, float& rtHi, float& zLoPointed, float& zHiPointed, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ)
//{
//    zLo = -999;
//    zHi = -999;
//    rtLo = -999;
//    rtHi = -999;
//    zLoPointed = -999;
//    zHiPointed = -999;
//    kZ = -999;
//    betaInCut = -999;
//
//    short outerInnerLowerModuleSubdet = modulesInGPU.subdets[outerInnerLowerModuleIndex];
//    short outerOuterLowerModuleSubdet = modulesInGPU.subdets[outerOuterLowerModuleIndex];
//
//    unsigned int firstMDIndex = segmentsInGPU.mdIndices[2 * innerSegmentIndex];
//
//    unsigned int secondMDIndex = segmentsInGPU.mdIndices[2 * innerSegmentIndex + 1];
//    unsigned int thirdMDIndex = segmentsInGPU.mdIndices[2 * outerSegmentIndex];
//    unsigned int fourthMDIndex = segmentsInGPU.mdIndices[2 * outerSegmentIndex + 1];
//
//    if(outerInnerLowerModuleSubdet == SDL::Barrel and outerOuterLowerModuleSubdet == SDL::Barrel)
//    {
//        return runpT5DefaultAlgoPPBB(modulesInGPU, rangesInGPU, mdsInGPU, segmentsInGPU, pixelLowerModuleIndex, outerInnerLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, firstMDIndex, secondMDIndex, thirdMDIndex, fourthMDIndex, zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta, zLo, zHi, zLoPointed, zHiPointed, sdlCut, betaOutCut, deltaBetaCut);
//    }
//    else if(outerInnerLowerModuleSubdet == SDL::Barrel and outerOuterLowerModuleSubdet == SDL::Endcap)
//    {
//        return runpT5DefaultAlgoPPBB(modulesInGPU, rangesInGPU, mdsInGPU, segmentsInGPU, pixelLowerModuleIndex, outerInnerLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, firstMDIndex, secondMDIndex, thirdMDIndex, fourthMDIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta, zLo, zHi, zLoPointed, zHiPointed, sdlCut, betaOutCut, deltaBetaCut);
//    }
//    else if(outerInnerLowerModuleSubdet == SDL::Endcap and outerOuterLowerModuleSubdet == SDL::Endcap)
//    {
//        return runpT5DefaultAlgoPPEE(modulesInGPU, rangesInGPU, mdsInGPU, segmentsInGPU, pixelLowerModuleIndex, outerInnerLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, firstMDIndex, secondMDIndex, thirdMDIndex, fourthMDIndex, zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta, zLo, rtLo, rtHi, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ);
//    }
//    return false;
//
//}
