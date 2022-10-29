# include "PixelTracklet.cuh"

//#ifdef CACHE_ALLOC
#include "allocate.h"
//#endif

void SDL::createPixelTrackletsInExplicitMemory(struct pixelTracklets& pixelTrackletsInGPU, unsigned int maxPixelTracklets,cudaStream_t stream)
{
#ifdef CACHE_ALLOC
//    cudaStream_t stream = 0;
    int dev;
    cudaGetDevice(&dev);

    pixelTrackletsInGPU.segmentIndices = (unsigned int*)cms::cuda::allocate_device(dev, maxPixelTracklets * sizeof(unsigned int) * 2,stream);
    pixelTrackletsInGPU.lowerModuleIndices = (unsigned int*)cms::cuda::allocate_device(dev, maxPixelTracklets * sizeof(unsigned int) * 2,stream);//split up to avoid runtime error of exceeding max byte allocation at a time
    pixelTrackletsInGPU.nPixelTracklets = (unsigned int*)cms::cuda::allocate_device(dev, sizeof(unsigned int),stream);
    pixelTrackletsInGPU.zOut = (float*)cms::cuda::allocate_device(dev, maxPixelTracklets * sizeof(float) * 4,stream);
    pixelTrackletsInGPU.betaIn = (float*)cms::cuda::allocate_device(dev, maxPixelTracklets * sizeof(float) * 3,stream);

#else
    cudaMalloc(&pixelTrackletsInGPU.segmentIndices, 2 * maxPixelTracklets * sizeof(unsigned int));
    cudaMalloc(&pixelTrackletsInGPU.lowerModuleIndices, 2 * maxPixelTracklets * sizeof(unsigned int));
    cudaMalloc(&pixelTrackletsInGPU.nPixelTracklets, sizeof(unsigned int));
    cudaMalloc(&pixelTrackletsInGPU.zOut, maxPixelTracklets *4* sizeof(float));
    cudaMalloc(&pixelTrackletsInGPU.betaIn, maxPixelTracklets *3* sizeof(float));
#endif
    pixelTrackletsInGPU.rtOut = pixelTrackletsInGPU.zOut + maxPixelTracklets;
    pixelTrackletsInGPU.deltaPhiPos = pixelTrackletsInGPU.zOut + maxPixelTracklets * 2;
    pixelTrackletsInGPU.deltaPhi = pixelTrackletsInGPU.zOut + maxPixelTracklets * 3;
    pixelTrackletsInGPU.betaOut = pixelTrackletsInGPU.betaIn + maxPixelTracklets;
    pixelTrackletsInGPU.pt_beta = pixelTrackletsInGPU.betaIn + maxPixelTracklets * 2;

    cudaMemsetAsync(pixelTrackletsInGPU.nPixelTracklets, 0, sizeof(unsigned int),stream);
}

#ifdef CUT_VALUE_DEBUG
ALPAKA_FN_ACC void SDL::addPixelTrackletToMemory(struct pixelTracklets& pixelTrackletsInGPU, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, unsigned int innerInnerLowerModuleIndex, unsigned int innerOuterLowerModuleIndex, unsigned int outerInnerLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& deltaPhi, float& betaIn, float& betaOut, float pt_beta, float& zLo, float& zHi, float& rtLo, float& rtHi, float& zLoPointed, float&
        zHiPointed, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ, unsigned int pixelTrackletIndex)
#else
ALPAKA_FN_ACC void SDL::addPixelTrackletToMemory(struct pixelTracklets& pixelTrackletsInGPU, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, unsigned int innerInnerLowerModuleIndex, unsigned int innerOuterLowerModuleIndex, unsigned int outerInnerLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& deltaPhi, float& betaIn, float& betaOut, float pt_beta, unsigned int pixelTrackletIndex)
#endif
{
    pixelTrackletsInGPU.segmentIndices[2 * pixelTrackletIndex] = innerSegmentIndex;
    pixelTrackletsInGPU.segmentIndices[2 * pixelTrackletIndex + 1] = outerSegmentIndex;
    pixelTrackletsInGPU.lowerModuleIndices[2 * pixelTrackletIndex] = outerInnerLowerModuleIndex;
    pixelTrackletsInGPU.lowerModuleIndices[2 * pixelTrackletIndex + 1] = outerOuterLowerModuleIndex;

    pixelTrackletsInGPU.zOut[pixelTrackletIndex] = zOut;
    pixelTrackletsInGPU.rtOut[pixelTrackletIndex] = rtOut;
    pixelTrackletsInGPU.deltaPhiPos[pixelTrackletIndex] = deltaPhiPos;
    pixelTrackletsInGPU.deltaPhi[pixelTrackletIndex] = deltaPhi;

    pixelTrackletsInGPU.betaIn[pixelTrackletIndex] = betaIn;
    pixelTrackletsInGPU.betaOut[pixelTrackletIndex] = betaOut;
    pixelTrackletsInGPU.pt_beta[pixelTrackletIndex] = pt_beta;

#ifdef CUT_VALUE_DEBUG
    pixelTrackletsInGPU.zLo[pixelTrackletIndex] = zLo;
    pixelTrackletsInGPU.zHi[pixelTrackletIndex] = zHi;
    pixelTrackletsInGPU.rtLo[pixelTrackletIndex] = rtLo;
    pixelTrackletsInGPU.rtHi[pixelTrackletIndex] = rtHi;
    pixelTrackletsInGPU.zLoPointed[pixelTrackletIndex] = zLoPointed;
    pixelTrackletsInGPU.zHiPointed[pixelTrackletIndex] = zHiPointed;
    pixelTrackletsInGPU.sdlCut[pixelTrackletIndex] = sdlCut;
    pixelTrackletsInGPU.betaInCut[pixelTrackletIndex] = betaInCut;
    pixelTrackletsInGPU.betaOutCut[pixelTrackletIndex] = betaOutCut;
    pixelTrackletsInGPU.deltaBetaCut[pixelTrackletIndex] = deltaBetaCut;
    pixelTrackletsInGPU.kZ[pixelTrackletIndex] = kZ;
#endif

}

void SDL::pixelTracklets::freeMemoryCache()
{
    int dev;
    cudaGetDevice(&dev);
    cms::cuda::free_device(dev,segmentIndices);
    cms::cuda::free_device(dev,lowerModuleIndices);
    cms::cuda::free_device(dev,zOut);
    cms::cuda::free_device(dev,betaIn);
    cms::cuda::free_device(dev,nPixelTracklets);
}

void SDL::pixelTracklets::freeMemory()
{
    cudaFree(segmentIndices);
    cudaFree(lowerModuleIndices);
    cudaFree(nPixelTracklets);
    cudaFree(zOut);
    cudaFree(betaIn);
#ifdef CUT_VALUE_DEBUG
    cudaFree(zLo);
    cudaFree(zHi);
    cudaFree(rtLo);
    cudaFree(rtHi);
    cudaFree(zLoPointed);
    cudaFree(zHiPointed);
    cudaFree(sdlCut);
    cudaFree(betaInCut);
    cudaFree(betaOutCut);
    cudaFree(deltaBetaCut);
    cudaFree(kZ);
#endif
}

SDL::pixelTracklets::pixelTracklets()
{
    segmentIndices = nullptr;
    lowerModuleIndices = nullptr;
    nPixelTracklets = nullptr;
    zOut = nullptr;
    rtOut = nullptr;

    deltaPhiPos = nullptr;
    deltaPhi = nullptr;
    betaIn = nullptr;
    betaOut = nullptr;
    pt_beta = nullptr;
#ifdef CUT_VALUE_DEBUG
    zLo = nullptr;
    zHi = nullptr;
    rtLo = nullptr;
    rtHi = nullptr;
    zLoPointed = nullptr;
    zHiPointed = nullptr;
    sdlCut = nullptr;
    betaInCut = nullptr;
    betaOutCut = nullptr;
    deltaBetaCut = nullptr;
    kZ = nullptr;
#endif

}

SDL::pixelTracklets::~pixelTracklets()
{

}

