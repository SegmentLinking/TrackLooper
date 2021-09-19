# include "Segment.cuh"
//#ifdef CACHE_ALLOC
#include "allocate.h"
//#endif

///FIXME:NOTICE THE NEW maxPixelSegments!

void SDL::createSegmentsInUnifiedMemory(struct segments& segmentsInGPU, unsigned int maxSegments, unsigned int nModules, unsigned int maxPixelSegments)
{
    //FIXME:Since the number of pixel segments is 10x the number of regular segments per module, we need to provide
    //extra memory to the pixel segments
    unsigned int nMemoryLocations = maxSegments * (nModules - 1) + maxPixelSegments;
#ifdef CACHE_ALLOC
    cudaStream_t stream=0; 
    segmentsInGPU.mdIndices = (unsigned int*)cms::cuda::allocate_managed(nMemoryLocations*6 *sizeof(unsigned int),stream);
    segmentsInGPU.nSegments = (unsigned int*)cms::cuda::allocate_managed(nModules *sizeof(unsigned int),stream);
    segmentsInGPU.dPhis = (float*)cms::cuda::allocate_managed((nMemoryLocations*6 + maxPixelSegments * 8) *sizeof(float),stream);
    segmentsInGPU.superbin = (int*)cms::cuda::allocate_managed((maxPixelSegments) *sizeof(int),stream);
    segmentsInGPU.pixelType = (int*)cms::cuda::allocate_managed((maxPixelSegments) *sizeof(int),stream);
    segmentsInGPU.isDup = (bool*)cms::cuda::allocate_managed((maxPixelSegments + nMemoryLocations) *sizeof(bool),stream);
    segmentsInGPU.circleCenterX = (float*)cms::cuda::allocate_managed((maxPixelSegments) * sizeof(float), stream);
    segmentsInGPU.circleCenterY = (float*)cms::cuda::allocate_managed((maxPixelSegments) * sizeof(float), stream);
    segmentsInGPU.circleRadius = (float*)cms::cuda::allocate_managed((maxPixelSegments) * sizeof(float), stream);
    segmentsInGPU.partOfPT5 = (bool*)cms::cuda::allocate_managed(maxPixelSegments * sizeof(bool), stream);
#else
    cudaMallocManaged(&segmentsInGPU.mdIndices, nMemoryLocations * 6 * sizeof(unsigned int));
    cudaMallocManaged(&segmentsInGPU.nSegments, nModules * sizeof(unsigned int));
    cudaMallocManaged(&segmentsInGPU.dPhis, (nMemoryLocations * 6 + maxPixelSegments * 8)*sizeof(float));
    cudaMallocManaged(&segmentsInGPU.superbin, (maxPixelSegments )*sizeof(int));
    cudaMallocManaged(&segmentsInGPU.pixelType, (maxPixelSegments )*sizeof(int));
    cudaMallocManaged(&segmentsInGPU.isDup, (maxPixelSegments + nMemoryLocations)*sizeof(bool));
    cudaMallocManaged(&segmentsInGPU.circleCenterX, maxPixelSegments * sizeof(float));
    cudaMallocManaged(&segmentsInGPU.circleCenterY, maxPixelSegments * sizeof(float));
    cudaMallocManaged(&segmentsInGPU.circleRadius, maxPixelSegments * sizeof(float));
    cudaMallocManaged(&segmentsInGPU.partOfPT5, maxPixelSegments * sizeof(bool));

#ifdef CUT_VALUE_DEBUG
    cudaMallocManaged(&segmentsInGPU.zIns, nMemoryLocations * 7 * sizeof(float));
    cudaMallocManaged(&segmentsInGPU.zLo, nMemoryLocations * sizeof(float));
    cudaMallocManaged(&segmentsInGPU.zHi, nMemoryLocations * sizeof(float));
    cudaMallocManaged(&segmentsInGPU.rtLo, nMemoryLocations * sizeof(float));
    cudaMallocManaged(&segmentsInGPU.rtHi, nMemoryLocations * sizeof(float));
    cudaMallocManaged(&segmentsInGPU.sdCut, nMemoryLocations * sizeof(float));
    cudaMallocManaged(&segmentsInGPU.dAlphaInnerMDSegmentThreshold, nMemoryLocations * sizeof(float));
    cudaMallocManaged(&segmentsInGPU.dAlphaOuterMDSegmentThreshold, nMemoryLocations * sizeof(float));
    cudaMallocManaged(&segmentsInGPU.dAlphaInnerMDOuterMDThreshold, nMemoryLocations * sizeof(float));

    segmentsInGPU.zOuts = segmentsInGPU.zIns + nMemoryLocations;
    segmentsInGPU.rtIns = segmentsInGPU.zIns + nMemoryLocations * 2;
    segmentsInGPU.rtOuts = segmentsInGPU.zIns + nMemoryLocations * 3;
    segmentsInGPU.dAlphaInnerMDSegments = segmentsInGPU.zIns + nMemoryLocations * 4;
    segmentsInGPU.dAlphaOuterMDSegments = segmentsInGPU.zIns + nMemoryLocations * 5;
    segmentsInGPU.dAlphaInnerMDOuterMDs = segmentsInGPU.zIns + nMemoryLocations * 6;

#endif
#endif
    segmentsInGPU.innerLowerModuleIndices = segmentsInGPU.mdIndices + nMemoryLocations * 2;
    segmentsInGPU.outerLowerModuleIndices = segmentsInGPU.mdIndices + nMemoryLocations * 3;
    segmentsInGPU.innerMiniDoubletAnchorHitIndices = segmentsInGPU.mdIndices + nMemoryLocations * 4;
    segmentsInGPU.outerMiniDoubletAnchorHitIndices = segmentsInGPU.mdIndices + nMemoryLocations * 5;

    segmentsInGPU.dPhiMins = segmentsInGPU.dPhis + nMemoryLocations;
    segmentsInGPU.dPhiMaxs = segmentsInGPU.dPhis + nMemoryLocations * 2;
    segmentsInGPU.dPhiChanges = segmentsInGPU.dPhis + nMemoryLocations * 3;
    segmentsInGPU.dPhiChangeMins = segmentsInGPU.dPhis + nMemoryLocations * 4;
    segmentsInGPU.dPhiChangeMaxs = segmentsInGPU.dPhis + nMemoryLocations * 5;

    segmentsInGPU.ptIn = segmentsInGPU.dPhis + nMemoryLocations * 6;
    segmentsInGPU.ptErr = segmentsInGPU.dPhis + nMemoryLocations * 6 + maxPixelSegments;
    segmentsInGPU.px = segmentsInGPU.dPhis + nMemoryLocations * 6 + maxPixelSegments * 2;
    segmentsInGPU.py = segmentsInGPU.dPhis + nMemoryLocations * 6 + maxPixelSegments * 3;
    segmentsInGPU.pz = segmentsInGPU.dPhis + nMemoryLocations * 6 + maxPixelSegments * 4;
    segmentsInGPU.etaErr = segmentsInGPU.dPhis + nMemoryLocations * 6 + maxPixelSegments * 5;
    segmentsInGPU.eta = segmentsInGPU.dPhis + nMemoryLocations * 6 + maxPixelSegments * 6;
    segmentsInGPU.phi = segmentsInGPU.dPhis + nMemoryLocations * 6 + maxPixelSegments * 7;
    
#pragma omp parallel for default(shared)
    for(size_t i = 0; i < nModules; i++)
    {
        segmentsInGPU.nSegments[i] = 0;
    }
    cudaMemset(segmentsInGPU.partOfPT5, false, maxPixelSegments * sizeof(bool));

}
void SDL::createSegmentsInExplicitMemory(struct segments& segmentsInGPU, unsigned int maxSegments, unsigned int nModules, unsigned int maxPixelSegments)
{
    //FIXME:Since the number of pixel segments is 10x the number of regular segments per module, we need to provide
    //extra memory to the pixel segments
    unsigned int nMemoryLocations = maxSegments * (nModules - 1) + maxPixelSegments;
#ifdef CACHE_ALLOC
    cudaStream_t stream=0; 
    int dev;
    cudaGetDevice(&dev);
    segmentsInGPU.mdIndices = (unsigned int*)cms::cuda::allocate_device(dev,nMemoryLocations*6 *sizeof(unsigned int),stream);
    segmentsInGPU.nSegments = (unsigned int*)cms::cuda::allocate_device(dev,nModules *sizeof(unsigned int),stream);
    segmentsInGPU.dPhis = (float*)cms::cuda::allocate_device(dev,(nMemoryLocations*6 + maxPixelSegments * 8) *sizeof(float),stream);
    segmentsInGPU.superbin = (int*)cms::cuda::allocate_device(dev,(maxPixelSegments) *sizeof(int),stream);
    segmentsInGPU.pixelType = (int*)cms::cuda::allocate_device(dev,(maxPixelSegments) *sizeof(int),stream);
    segmentsInGPU.isDup = (bool*)cms::cuda::allocate_device(dev,(maxPixelSegments + nMemoryLocations) *sizeof(bool),stream);
    segmentsInGPU.circleCenterX = (float*)cms::cuda::allocate_device(dev, maxPixelSegments * sizeof(float), stream);
    segmentsInGPU.circleCenterY = (float*)cms::cuda::allocate_device(dev, maxPixelSegments * sizeof(float), stream);
    segmentsInGPU.circleRadius = (float*)cms::cuda::allocate_device(dev, maxPixelSegments * sizeof(float), stream);
    segmentsInGPU.partOfPT5 = (bool*)cms::cuda::allocate_device(dev, maxPixelSegments * sizeof(bool), stream);

#else
    cudaMalloc(&segmentsInGPU.mdIndices, nMemoryLocations * 6 * sizeof(unsigned int));
    cudaMalloc(&segmentsInGPU.nSegments, nModules * sizeof(unsigned int));
    cudaMalloc(&segmentsInGPU.dPhis, (nMemoryLocations * 6 + maxPixelSegments * 8)*sizeof(float));
    cudaMalloc(&segmentsInGPU.superbin, (maxPixelSegments )*sizeof(int));
    cudaMalloc(&segmentsInGPU.pixelType, (maxPixelSegments )*sizeof(int));
    cudaMalloc(&segmentsInGPU.isDup, (maxPixelSegments + nMemoryLocations)*sizeof(bool));
    cudaMalloc(&segmentsInGPU.circleCenterX, maxPixelSegments * sizeof(float));
    cudaMalloc(&segmentsInGPU.circleCenterY, maxPixelSegments * sizeof(float));
    cudaMalloc(&segmentsInGPU.circleRadius, maxPixelSegments * sizeof(float));
    cudaMalloc(&segmentsInGPU.partOfPT5, maxPixelSegments * sizeof(bool));

#endif
    cudaMemset(segmentsInGPU.nSegments,0,nModules * sizeof(unsigned int));
    cudaMemset(segmentsInGPU.partOfPT5, false, maxPixelSegments * sizeof(bool));

    segmentsInGPU.innerLowerModuleIndices = segmentsInGPU.mdIndices + nMemoryLocations * 2;
    segmentsInGPU.outerLowerModuleIndices = segmentsInGPU.mdIndices + nMemoryLocations * 3;
    segmentsInGPU.innerMiniDoubletAnchorHitIndices = segmentsInGPU.mdIndices + nMemoryLocations * 4;
    segmentsInGPU.outerMiniDoubletAnchorHitIndices = segmentsInGPU.mdIndices + nMemoryLocations * 5;

    segmentsInGPU.dPhiMins = segmentsInGPU.dPhis + nMemoryLocations;
    segmentsInGPU.dPhiMaxs = segmentsInGPU.dPhis + nMemoryLocations * 2;
    segmentsInGPU.dPhiChanges = segmentsInGPU.dPhis + nMemoryLocations * 3;
    segmentsInGPU.dPhiChangeMins = segmentsInGPU.dPhis + nMemoryLocations * 4;
    segmentsInGPU.dPhiChangeMaxs = segmentsInGPU.dPhis + nMemoryLocations * 5;

    segmentsInGPU.ptIn = segmentsInGPU.dPhis + nMemoryLocations * 6;
    segmentsInGPU.ptErr = segmentsInGPU.dPhis + nMemoryLocations * 6 + maxPixelSegments;
    segmentsInGPU.px = segmentsInGPU.dPhis + nMemoryLocations * 6 + maxPixelSegments * 2;
    segmentsInGPU.py = segmentsInGPU.dPhis + nMemoryLocations * 6 + maxPixelSegments * 3;
    segmentsInGPU.pz = segmentsInGPU.dPhis + nMemoryLocations * 6 + maxPixelSegments * 4;
    segmentsInGPU.etaErr = segmentsInGPU.dPhis + nMemoryLocations * 6 + maxPixelSegments * 5;
    segmentsInGPU.eta = segmentsInGPU.dPhis + nMemoryLocations * 6 + maxPixelSegments * 6;
    segmentsInGPU.phi = segmentsInGPU.dPhis + nMemoryLocations * 6 + maxPixelSegments * 7;

}

SDL::segments::segments()
{
    superbin = nullptr;
    pixelType = nullptr;
    isDup = nullptr;
    circleRadius = nullptr;
    circleCenterX = nullptr;
    circleCenterY = nullptr;
    mdIndices = nullptr;
    innerLowerModuleIndices = nullptr;
    outerLowerModuleIndices = nullptr;
    innerMiniDoubletAnchorHitIndices = nullptr;
    outerMiniDoubletAnchorHitIndices = nullptr;

    nSegments = nullptr;
    dPhis = nullptr;
    dPhiMins = nullptr;
    dPhiMaxs = nullptr;
    dPhiChanges = nullptr;
    dPhiChangeMins = nullptr;
    dPhiChangeMaxs = nullptr;
    partOfPT5 = nullptr;

#ifdef CUT_VALUE_DEBUG
    zIns = nullptr;
    zOuts = nullptr;
    rtIns = nullptr;
    rtOuts = nullptr;
    dAlphaInnerMDSegments = nullptr;
    dAlphaOuterMDSegments = nullptr;
    dAlphaInnerMDOuterMDs = nullptr;

    zLo = nullptr;
    zHi = nullptr;
    rtLo = nullptr;
    rtHi = nullptr;
    sdCut = nullptr;
    dAlphaInnerMDSegmentThreshold = nullptr;
    dAlphaOuterMDSegmentThreshold = nullptr;
    dAlphaInnerMDOuterMDThreshold = nullptr;
#endif
}

SDL::segments::~segments()
{
}

void SDL::segments::freeMemoryCache()
{
#ifdef Explicit_Seg
    int dev;
    cudaGetDevice(&dev);
    cms::cuda::free_device(dev,mdIndices);
    cms::cuda::free_device(dev,dPhis);
    cms::cuda::free_device(dev,nSegments);
    cms::cuda::free_device(dev,superbin);
    cms::cuda::free_device(dev,pixelType);
    cms::cuda::free_device(dev,isDup);
    cms::cuda::free_device(dev, circleCenterX);
    cms::cuda::free_device(dev, circleCenterY);
    cms::cuda::free_device(dev, circleRadius);
    cms::cuda::free_device(dev, partOfPT5);
#else
    cms::cuda::free_managed(mdIndices);
    cms::cuda::free_managed(dPhis);
    cms::cuda::free_managed(nSegments);
    cms::cuda::free_managed(superbin);
    cms::cuda::free_managed(pixelType);
    cms::cuda::free_managed(isDup);
    cms::cuda::free_managed(circleCenterX);
    cms::cuda::free_managed(circleCenterY);
    cms::cuda::free_managed(circleRadius);
    cms::cuda::free_managed(partOfPT5);
#endif
}
void SDL::segments::freeMemory()
{
    cudaFree(mdIndices);
    cudaFree(nSegments);
    cudaFree(dPhis);
    cudaFree(superbin);
    cudaFree(pixelType);
    cudaFree(isDup);
    cudaFree(circleCenterX);
    cudaFree(circleCenterY);
    cudaFree(circleRadius);
    cudaFree(partOfPT5);
#ifdef CUT_VALUE_DEBUG
    cudaFree(zIns);
    cudaFree(zLo);
    cudaFree(zHi);
    cudaFree(rtLo);
    cudaFree(rtHi);
    cudaFree(sdCut);
    cudaFree(dAlphaInnerMDSegmentThreshold);
    cudaFree(dAlphaOuterMDSegmentThreshold);
    cudaFree(dAlphaInnerMDOuterMDThreshold);
#endif
}


#ifdef CUT_VALUE_DEBUG
__device__ void SDL::addSegmentToMemory(struct segments& segmentsInGPU, unsigned int lowerMDIndex, unsigned int upperMDIndex, unsigned int innerLowerModuleIndex, unsigned int outerLowerModuleIndex, unsigned int innerMDAnchorHitIndex, unsigned int outerMDAnchorHitIndex, float& dPhi, float& dPhiMin, float& dPhiMax, float& dPhiChange, float& dPhiChangeMin, float& dPhiChangeMax, float& zIn, float& zOut, float& rtIn, float& rtOut, float& dAlphaInnerMDSegment, float& dAlphaOuterMDSegment, float&
        dAlphaInnerMDOuterMD, float& zLo, float& zHi, float& rtLo, float& rtHi, float& sdCut, float& dAlphaInnerMDSegmentThreshold, float& dAlphaOuterMDSegmentThreshold, float& dAlphaInnerMDOuterMDThreshold, unsigned int idx)
#else
__device__ void SDL::addSegmentToMemory(struct segments& segmentsInGPU, unsigned int lowerMDIndex, unsigned int upperMDIndex, unsigned int innerLowerModuleIndex, unsigned int outerLowerModuleIndex, unsigned int innerMDAnchorHitIndex, unsigned int outerMDAnchorHitIndex, float& dPhi, float& dPhiMin, float& dPhiMax, float& dPhiChange, float& dPhiChangeMin, float& dPhiChangeMax, unsigned int idx)
#endif
{
    //idx will be computed in the kernel, which is the index into which the 
    //segment will be written
    //nSegments will be incremented in the kernel

    segmentsInGPU.mdIndices[idx * 2] = lowerMDIndex;
    segmentsInGPU.mdIndices[idx * 2 + 1] = upperMDIndex;
    segmentsInGPU.innerLowerModuleIndices[idx] = innerLowerModuleIndex;
    segmentsInGPU.outerLowerModuleIndices[idx] = outerLowerModuleIndex;
    segmentsInGPU.innerMiniDoubletAnchorHitIndices[idx] = innerMDAnchorHitIndex;
    segmentsInGPU.outerMiniDoubletAnchorHitIndices[idx] = outerMDAnchorHitIndex;

    segmentsInGPU.dPhis[idx] = dPhi;
    segmentsInGPU.dPhiMins[idx] = dPhiMin;
    segmentsInGPU.dPhiMaxs[idx] = dPhiMax;
    segmentsInGPU.dPhiChanges[idx] = dPhiChange;
    segmentsInGPU.dPhiChangeMins[idx] = dPhiChangeMin;
    segmentsInGPU.dPhiChangeMaxs[idx] = dPhiChangeMax;

#ifdef CUT_VALUE_DEBUG
    segmentsInGPU.zIns[idx] = zIn;
    segmentsInGPU.zOuts[idx] = zOut;
    segmentsInGPU.rtIns[idx] = rtIn;
    segmentsInGPU.rtOuts[idx] = rtOut;
    segmentsInGPU.dAlphaInnerMDSegments[idx] = dAlphaInnerMDSegment;
    segmentsInGPU.dAlphaOuterMDSegments[idx] = dAlphaOuterMDSegment;
    segmentsInGPU.dAlphaInnerMDOuterMDs[idx] = dAlphaInnerMDOuterMD;

    segmentsInGPU.zLo[idx] = zLo;
    segmentsInGPU.zHi[idx] = zHi;
    segmentsInGPU.rtLo[idx] = rtLo;
    segmentsInGPU.rtHi[idx] = rtHi;
    segmentsInGPU.sdCut[idx] = sdCut;
    segmentsInGPU.dAlphaInnerMDSegmentThreshold[idx] = dAlphaInnerMDSegmentThreshold;
    segmentsInGPU.dAlphaOuterMDSegmentThreshold[idx] = dAlphaOuterMDSegmentThreshold;
    segmentsInGPU.dAlphaInnerMDOuterMDThreshold[idx] = dAlphaInnerMDOuterMDThreshold;
#endif
}

__device__ void SDL::rmPixelSegmentFromMemory(struct segments& segmentsInGPU,unsigned int pixelSegmentArrayIndex){
    segmentsInGPU.isDup[pixelSegmentArrayIndex] = 1;
}
__device__ void SDL::addPixelSegmentToMemory(struct segments& segmentsInGPU, struct miniDoublets& mdsInGPU, struct hits& hitsInGPU, struct modules& modulesInGPU, unsigned int innerMDIndex, unsigned int outerMDIndex, unsigned int pixelModuleIndex, unsigned int innerAnchorHitIndex, unsigned int outerAnchorHitIndex, float dPhiChange, float ptIn, float ptErr, float px, float py, float pz, float etaErr, float eta, float phi, unsigned int idx, unsigned int pixelSegmentArrayIndex, int superbin, int
        pixelType)

{
    segmentsInGPU.mdIndices[idx * 2] = innerMDIndex;
    segmentsInGPU.mdIndices[idx * 2 + 1] = outerMDIndex;
    segmentsInGPU.innerLowerModuleIndices[idx] = pixelModuleIndex;
    segmentsInGPU.outerLowerModuleIndices[idx] = pixelModuleIndex;
    segmentsInGPU.innerMiniDoubletAnchorHitIndices[idx] = innerAnchorHitIndex;
    segmentsInGPU.outerMiniDoubletAnchorHitIndices[idx] = outerAnchorHitIndex;
    segmentsInGPU.dPhiChanges[idx] = dPhiChange;
    segmentsInGPU.ptIn[pixelSegmentArrayIndex] = ptIn;
    segmentsInGPU.ptErr[pixelSegmentArrayIndex] = ptErr;
    segmentsInGPU.px[pixelSegmentArrayIndex] = px;
    segmentsInGPU.py[pixelSegmentArrayIndex] = py;
    segmentsInGPU.pz[pixelSegmentArrayIndex] = pz;
    segmentsInGPU.etaErr[pixelSegmentArrayIndex] = etaErr;
    segmentsInGPU.eta[pixelSegmentArrayIndex] = eta;
    segmentsInGPU.phi[pixelSegmentArrayIndex] = phi;

    segmentsInGPU.superbin[pixelSegmentArrayIndex] = superbin;
    segmentsInGPU.pixelType[pixelSegmentArrayIndex] = pixelType;
    segmentsInGPU.isDup[pixelSegmentArrayIndex] = 0;

    //computing circle parameters
    /*
       The two anchor hits are r3PCA and r3LH. p3PCA pt, eta, phi is hitIndex1
    */
    float circleRadius = hitsInGPU.xs[mdsInGPU.hitIndices[2 * innerMDIndex + 1]] / (2 * k2Rinv1GeVf);
    float circlePhi = hitsInGPU.zs[mdsInGPU.hitIndices[2 * innerMDIndex + 1]];

    float candidateCenterXs[] = {hitsInGPU.xs[innerAnchorHitIndex] + circleRadius * sinf(circlePhi), hitsInGPU.xs[innerAnchorHitIndex] - circleRadius * sinf(circlePhi)};
    float candidateCenterYs[] = {hitsInGPU.ys[innerAnchorHitIndex] - circleRadius * cosf(circlePhi), hitsInGPU.ys[innerAnchorHitIndex] + circleRadius * cosf(circlePhi)};

    //check which of the circles can accommodate r3LH better (we won't get perfect agreement)
    float bestChiSquared = 123456789.f;
    float chiSquared;
    size_t bestIndex;
    for(size_t i = 0; i < 2; i++)
    {
        chiSquared = fabsf(sqrtf((hitsInGPU.xs[outerAnchorHitIndex] - candidateCenterXs[i]) * (hitsInGPU.xs[outerAnchorHitIndex] - candidateCenterXs[i]) + (hitsInGPU.ys[outerAnchorHitIndex] - candidateCenterYs[i]) * (hitsInGPU.ys[outerAnchorHitIndex] - candidateCenterYs[i])) - circleRadius);
        if(chiSquared < bestChiSquared)
        {
            bestChiSquared = chiSquared;
            bestIndex = i;
        }
    }
    segmentsInGPU.circleCenterX[pixelSegmentArrayIndex] = candidateCenterXs[bestIndex];
    segmentsInGPU.circleCenterY[pixelSegmentArrayIndex] = candidateCenterYs[bestIndex];
    segmentsInGPU.circleRadius[pixelSegmentArrayIndex] = circleRadius;
}

__device__ void SDL::dAlphaThreshold(float* dAlphaThresholdValues, struct hits& hitsInGPU, struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, unsigned int& innerMiniDoubletAnchorHitIndex, unsigned int& outerMiniDoubletAnchorHitIndex, unsigned int& innerLowerModuleIndex, unsigned int& outerLowerModuleIndex, unsigned int& innerMDIndex, unsigned int& outerMDIndex)
{
    float sdMuls = (modulesInGPU.subdets[innerLowerModuleIndex] == SDL::Barrel) ? miniMulsPtScaleBarrel[modulesInGPU.layers[innerLowerModuleIndex]-1] * 3.f/ptCut : miniMulsPtScaleEndcap[modulesInGPU.layers[innerLowerModuleIndex]-1] * 3.f/ptCut;

    // BField dAlpha

    float innerMiniDoubletAnchorHitRt = hitsInGPU.rts[innerMiniDoubletAnchorHitIndex];
    float outerMiniDoubletAnchorHitRt = hitsInGPU.rts[outerMiniDoubletAnchorHitIndex];
    float innerMiniDoubletAnchorHitZ = hitsInGPU.zs[innerMiniDoubletAnchorHitIndex];
    float outerMiniDoubletAnchorHitZ = hitsInGPU.zs[outerMiniDoubletAnchorHitIndex];

    //more accurate then outer rt - inner rt
    float segmentY = hitsInGPU.ys[outerMiniDoubletAnchorHitIndex] - hitsInGPU.ys[innerMiniDoubletAnchorHitIndex];
    float segmentX = hitsInGPU.xs[outerMiniDoubletAnchorHitIndex]- hitsInGPU.xs[innerMiniDoubletAnchorHitIndex]; 
    float segmentDr = sqrtf((segmentY * segmentY) + (segmentX * segmentX));
    

    const float dAlpha_Bfield = asinf(fminf(segmentDr * k2Rinv1GeVf/ptCut, sinAlphaMax));

    bool isInnerTilted = modulesInGPU.subdets[innerLowerModuleIndex] == SDL::Barrel and modulesInGPU.sides[innerLowerModuleIndex] != SDL::Center;
    bool isOuterTilted = modulesInGPU.subdets[outerLowerModuleIndex] == SDL::Barrel and modulesInGPU.sides[outerLowerModuleIndex] != SDL::Center;
    float drdzInner = -1.f;
    float drdzOuter = -1.f;
    if(isInnerTilted)
    {
        if(modulesInGPU.moduleLayerType[innerLowerModuleIndex] == Strip)
        {
            drdzInner = modulesInGPU.drdzs[innerLowerModuleIndex];
        }
        else
        {
            drdzInner = modulesInGPU.drdzs[modulesInGPU.partnerModuleIndex(innerLowerModuleIndex)];
        }
    }
    if(isOuterTilted)
    {
        if(modulesInGPU.moduleLayerType[outerLowerModuleIndex] == Strip)
        {
            drdzOuter = modulesInGPU.drdzs[outerLowerModuleIndex];
        }
        else
        {
            drdzOuter = modulesInGPU.drdzs[modulesInGPU.partnerModuleIndex(outerLowerModuleIndex)];
        }

    }

    float innerModuleGapSize = SDL::moduleGapSize(modulesInGPU, innerLowerModuleIndex);
    float outerModuleGapSize = SDL::moduleGapSize(modulesInGPU, outerLowerModuleIndex);
    const float innerminiTilt = isInnerTilted ? (0.5f * pixelPSZpitch * drdzInner / sqrtf(1.f + drdzInner * drdzInner) / innerModuleGapSize) : 0;

    const float outerminiTilt = isOuterTilted ? (0.5f * pixelPSZpitch * drdzOuter / sqrtf(1.f + drdzOuter * drdzOuter) / outerModuleGapSize) : 0;

    float miniDelta = innerModuleGapSize; 
 

    float sdLumForInnerMini;    
    float sdLumForOuterMini;

    if (modulesInGPU.subdets[innerLowerModuleIndex] == SDL::Barrel)
    {
        sdLumForInnerMini = innerminiTilt * dAlpha_Bfield;
    }
    else
    {
        sdLumForInnerMini = mdsInGPU.dphis[innerMDIndex] * 15.0f / mdsInGPU.dzs[innerMDIndex];
    }

    if (modulesInGPU.subdets[outerLowerModuleIndex] == SDL::Barrel)
    {
        sdLumForOuterMini = outerminiTilt * dAlpha_Bfield;
    }
    else
    {
        sdLumForOuterMini = mdsInGPU.dphis[outerMDIndex] * 15.0f / mdsInGPU.dzs[outerMDIndex];
    }


    //Unique stuff for the segment dudes alone

    float dAlpha_res_inner = 0.02f/miniDelta * (modulesInGPU.subdets[innerLowerModuleIndex] == SDL::Barrel ? 1.0f : fabsf(innerMiniDoubletAnchorHitZ/innerMiniDoubletAnchorHitRt));
    float dAlpha_res_outer = 0.02f/miniDelta * (modulesInGPU.subdets[outerLowerModuleIndex] == SDL::Barrel ? 1.0f : fabsf(outerMiniDoubletAnchorHitZ/outerMiniDoubletAnchorHitRt));

 
    float dAlpha_res = dAlpha_res_inner + dAlpha_res_outer;

    if (modulesInGPU.subdets[innerLowerModuleIndex] == SDL::Barrel and modulesInGPU.sides[innerLowerModuleIndex] == SDL::Center)
    {
        dAlphaThresholdValues[0] = dAlpha_Bfield + sqrt(dAlpha_res * dAlpha_res + sdMuls * sdMuls);       
    }
    else
    {
        dAlphaThresholdValues[0] = dAlpha_Bfield + sqrt(dAlpha_res * dAlpha_res + sdMuls * sdMuls + sdLumForInnerMini * sdLumForInnerMini);    
    }

    if(modulesInGPU.subdets[outerLowerModuleIndex] == SDL::Barrel and modulesInGPU.sides[outerLowerModuleIndex] == SDL::Center)
    {
        dAlphaThresholdValues[1] = dAlpha_Bfield + sqrt(dAlpha_res * dAlpha_res + sdMuls * sdMuls);    
    }
    else
    {
        dAlphaThresholdValues[1] = dAlpha_Bfield + sqrt(dAlpha_res * dAlpha_res + sdMuls * sdMuls + sdLumForOuterMini * sdLumForOuterMini);
    }

    //Inner to outer 
    dAlphaThresholdValues[2] = dAlpha_Bfield + sqrt(dAlpha_res * dAlpha_res + sdMuls * sdMuls);

}


__device__ bool SDL::runSegmentDefaultAlgoEndcap(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, unsigned int& innerLowerModuleIndex, unsigned int& outerLowerModuleIndex, unsigned int& innerMDIndex, unsigned int& outerMDIndex, float& zIn, float& zOut, float& rtIn, float& rtOut, float& dPhi, float& dPhiMin, float& dPhiMax, float& dPhiChange, float& dPhiChangeMin, float& dPhiChangeMax, float& dAlphaInnerMDSegment, float& dAlphaOuterMDSegment,
        float& rtLo, float& rtHi, float& sdCut, float& dAlphaInnerMDSegmentThreshold, float& dAlphaOuterMDSegmentThreshold, float& dAlphaInnerMDOuterMDThreshold, float&
        dAlphaInnerMDOuterMD, unsigned int& innerMiniDoubletAnchorHitIndex, unsigned int& outerMiniDoubletAnchorHitIndex)
{
    bool pass = true;
    
    if(mdsInGPU.pixelModuleFlag[innerMDIndex] >= 0)
    {
        if(mdsInGPU.pixelModuleFlag[innerMDIndex] == 0)
        {    
            innerMiniDoubletAnchorHitIndex = mdsInGPU.hitIndices[innerMDIndex * 2]; 
        }
        else
        {
            innerMiniDoubletAnchorHitIndex = mdsInGPU.hitIndices[innerMDIndex * 2 + 1];
        }
    }
    else
    {
        innerMiniDoubletAnchorHitIndex = mdsInGPU.hitIndices[innerMDIndex * 2];
    }

    if(mdsInGPU.pixelModuleFlag[outerMDIndex] >= 0)
    {
        if(mdsInGPU.pixelModuleFlag[outerMDIndex] == 0)
        {    
            outerMiniDoubletAnchorHitIndex = mdsInGPU.hitIndices[outerMDIndex * 2]; 
        }
        else
        {
            outerMiniDoubletAnchorHitIndex = mdsInGPU.hitIndices[outerMDIndex * 2 + 1];
        }
    }
    else
    {
        outerMiniDoubletAnchorHitIndex = mdsInGPU.hitIndices[outerMDIndex * 2];
    }

    rtIn = hitsInGPU.rts[innerMiniDoubletAnchorHitIndex];
    rtOut = hitsInGPU.rts[outerMiniDoubletAnchorHitIndex];
    zIn = hitsInGPU.zs[innerMiniDoubletAnchorHitIndex];
    zOut = hitsInGPU.zs[outerMiniDoubletAnchorHitIndex];

    bool outerLayerEndcapTwoS = (modulesInGPU.subdets[outerLowerModuleIndex] == SDL::Endcap) and (modulesInGPU.moduleType[outerLowerModuleIndex] == SDL::TwoS);

    
    float sdSlope = asinf(fminf(rtOut * k2Rinv1GeVf / ptCut, sinAlphaMax));
    float sdPVoff = 0.1/rtOut;
    float disks2SMinRadius = 60.f;

    float rtGeom =  ((rtIn < disks2SMinRadius && rtOut < disks2SMinRadius) ? (2.f * pixelPSZpitch)
            : ((rtIn < disks2SMinRadius || rtOut < disks2SMinRadius) ? (pixelPSZpitch + strip2SZpitch)
            : (2.f * strip2SZpitch)));


    //cut 0 - z compatibility
    if(zIn * zOut < 0)
    {
        pass = false;
    }

    float dz = zOut - zIn;
    float dLum = copysignf(deltaZLum, zIn);
    float drtDzScale = sdSlope/tanf(sdSlope);

    rtLo = fmaxf(rtIn * (1.f + dz / (zIn + dLum) * drtDzScale) - rtGeom,  rtIn - 0.5f * rtGeom); //rt should increase
    rtHi = rtIn * (zOut - dLum) / (zIn - dLum) + rtGeom; //dLum for luminous; rGeom for measurement size; no tanTheta_loc(pt) correction

    //completeness

    if(not(rtOut >= rtLo and rtOut <= rtHi))
    {
        pass = false;
    }

    dPhi = deltaPhi(hitsInGPU.xs[innerMiniDoubletAnchorHitIndex], hitsInGPU.ys[innerMiniDoubletAnchorHitIndex], hitsInGPU.zs[innerMiniDoubletAnchorHitIndex], hitsInGPU.xs[outerMiniDoubletAnchorHitIndex], hitsInGPU.ys[outerMiniDoubletAnchorHitIndex], hitsInGPU.zs[outerMiniDoubletAnchorHitIndex]);

    sdCut = sdSlope;
    unsigned int outerEdgeIndex;
    if(outerLayerEndcapTwoS)
    {
        outerEdgeIndex = outerMiniDoubletAnchorHitIndex;

        float dPhiPos_high = deltaPhi(hitsInGPU.xs[innerMiniDoubletAnchorHitIndex], hitsInGPU.ys[innerMiniDoubletAnchorHitIndex], hitsInGPU.zs[innerMiniDoubletAnchorHitIndex], hitsInGPU.highEdgeXs[outerEdgeIndex], hitsInGPU.highEdgeYs[outerEdgeIndex], hitsInGPU.zs[outerMiniDoubletAnchorHitIndex]);

        float dPhiPos_low = deltaPhi(hitsInGPU.xs[innerMiniDoubletAnchorHitIndex], hitsInGPU.ys[innerMiniDoubletAnchorHitIndex], hitsInGPU.zs[innerMiniDoubletAnchorHitIndex], hitsInGPU.lowEdgeXs[outerEdgeIndex], hitsInGPU.lowEdgeYs[outerEdgeIndex], hitsInGPU.zs[outerMiniDoubletAnchorHitIndex]);

        dPhiMax = fabsf(dPhiPos_high) > fabsf(dPhiPos_low) ? dPhiPos_high : dPhiPos_low;
        dPhiMin = fabsf(dPhiPos_high) > fabsf(dPhiPos_low) ? dPhiPos_low : dPhiPos_high;
    }
    else
    {
        dPhiMax = dPhi;
        dPhiMin = dPhi;
    }

    if(fabsf(dPhi) > sdCut)
    {
        pass = false;
    }

    float dzFrac = dz/zIn;
    dPhiChange = dPhi/dzFrac * (1.f + dzFrac);
    dPhiChangeMin = dPhiMin/dzFrac * (1.f + dzFrac);
    dPhiChangeMax = dPhiMax/dzFrac * (1.f + dzFrac);

    if(fabsf(dPhiChange) > sdCut)
    {
        pass = false;
    }



    float dAlphaThresholdValues[3];
    dAlphaThreshold(dAlphaThresholdValues, hitsInGPU, modulesInGPU, mdsInGPU, innerMiniDoubletAnchorHitIndex, outerMiniDoubletAnchorHitIndex, innerLowerModuleIndex, outerLowerModuleIndex, innerMDIndex, outerMDIndex);

    dAlphaInnerMDSegmentThreshold = dAlphaThresholdValues[0];
    dAlphaOuterMDSegmentThreshold = dAlphaThresholdValues[1];
    dAlphaInnerMDOuterMDThreshold = dAlphaThresholdValues[2];

    float innerMDAlpha = mdsInGPU.dphichanges[innerMDIndex];
    float outerMDAlpha = mdsInGPU.dphichanges[outerMDIndex];
    dAlphaInnerMDSegment = innerMDAlpha - dPhiChange;
    dAlphaOuterMDSegment = outerMDAlpha - dPhiChange;
    dAlphaInnerMDOuterMD = innerMDAlpha - outerMDAlpha;
   
 
    if(fabsf(dAlphaInnerMDSegment) >= dAlphaThresholdValues[0])
    {
        pass = false;
    }

    if(fabsf(dAlphaOuterMDSegment) >= dAlphaThresholdValues[1])
    {
        pass = false;
    }

    if(fabsf(dAlphaInnerMDOuterMD) >= dAlphaThresholdValues[2])
    {
        pass = false;
    }


    return pass;
}

__device__ bool SDL::runSegmentDefaultAlgoBarrel(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, unsigned int& innerLowerModuleIndex, unsigned int& outerLowerModuleIndex, unsigned int& innerMDIndex, unsigned int& outerMDIndex, float& zIn, float& zOut, float& rtIn, float& rtOut, float& dPhi, float& dPhiMin, float& dPhiMax, float& dPhiChange, float& dPhiChangeMin, float& dPhiChangeMax, float& dAlphaInnerMDSegment, float& dAlphaOuterMDSegment, float&
        dAlphaInnerMDOuterMD, float& zLo, float& zHi, float& sdCut, float& dAlphaInnerMDSegmentThreshold, float& dAlphaOuterMDSegmentThreshold, float& dAlphaInnerMDOuterMDThreshold, unsigned int& innerMiniDoubletAnchorHitIndex, unsigned int& outerMiniDoubletAnchorHitIndex)
{
    bool pass = true;
   
    float sdMuls = (modulesInGPU.subdets[innerLowerModuleIndex] == SDL::Barrel) ? miniMulsPtScaleBarrel[modulesInGPU.layers[innerLowerModuleIndex]-1] * 3.f/ptCut : miniMulsPtScaleEndcap[modulesInGPU.layers[innerLowerModuleIndex]-1] * 3.f/ptCut;


    if(mdsInGPU.pixelModuleFlag[innerMDIndex] >= 0)
    {
        if(mdsInGPU.pixelModuleFlag[innerMDIndex] == 0)
        {    
            innerMiniDoubletAnchorHitIndex = mdsInGPU.hitIndices[innerMDIndex * 2];
        }
        else
        {
            innerMiniDoubletAnchorHitIndex = mdsInGPU.hitIndices[innerMDIndex * 2 + 1]; 
        }
    }
    else
    {
        innerMiniDoubletAnchorHitIndex = mdsInGPU.hitIndices[innerMDIndex * 2];
    }

    if(mdsInGPU.pixelModuleFlag[outerMDIndex] >= 0)
    {
        if(mdsInGPU.pixelModuleFlag[outerMDIndex] == 0)
        {    
            outerMiniDoubletAnchorHitIndex = mdsInGPU.hitIndices[outerMDIndex * 2];
 		
        }
        else
        {
            outerMiniDoubletAnchorHitIndex = mdsInGPU.hitIndices[outerMDIndex * 2 + 1];
	    
        }
    }
    else
    {
        outerMiniDoubletAnchorHitIndex = mdsInGPU.hitIndices[outerMDIndex * 2];
    }


    rtIn = hitsInGPU.rts[innerMiniDoubletAnchorHitIndex];
    rtOut = hitsInGPU.rts[outerMiniDoubletAnchorHitIndex];
    zIn = hitsInGPU.zs[innerMiniDoubletAnchorHitIndex];
    zOut = hitsInGPU.zs[outerMiniDoubletAnchorHitIndex];

    float sdSlope = asinf(fminf(rtOut * k2Rinv1GeVf / ptCut, sinAlphaMax));
    float sdPVoff = 0.1f/rtOut;
    float dzDrtScale = tanf(sdSlope)/sdSlope; //FIXME: need appropriate value

    const float zGeom = modulesInGPU.layers[innerLowerModuleIndex] <= 2 ? 2.f * pixelPSZpitch : 2.f * strip2SZpitch;

    zLo = zIn + (zIn - deltaZLum) * (rtOut / rtIn - 1.f) * (zIn > 0.f ? 1.f : dzDrtScale) - zGeom; //slope-correction only on outer end
    zHi = zIn + (zIn + deltaZLum) * (rtOut / rtIn - 1.f) * (zIn < 0.f ? 1.f : dzDrtScale) + zGeom;

    //cut 1 - z compatibility
    if(not(zOut >= zLo and zOut <= zHi))
    {
        pass = false;
    }


    dPhi = deltaPhi(hitsInGPU.xs[innerMiniDoubletAnchorHitIndex], hitsInGPU.ys[innerMiniDoubletAnchorHitIndex], hitsInGPU.zs[innerMiniDoubletAnchorHitIndex], hitsInGPU.xs[outerMiniDoubletAnchorHitIndex], hitsInGPU.ys[outerMiniDoubletAnchorHitIndex], hitsInGPU.zs[outerMiniDoubletAnchorHitIndex]);
    sdCut = sdSlope + sqrtf(sdMuls * sdMuls + sdPVoff * sdPVoff);

    if(not( fabsf(dPhi) <= sdCut ))
    {
        pass = false;
    }

    dPhiChange = deltaPhiChange(hitsInGPU.xs[innerMiniDoubletAnchorHitIndex], hitsInGPU.ys[innerMiniDoubletAnchorHitIndex], hitsInGPU.zs[innerMiniDoubletAnchorHitIndex], hitsInGPU.xs[outerMiniDoubletAnchorHitIndex], hitsInGPU.ys[outerMiniDoubletAnchorHitIndex], hitsInGPU.zs[outerMiniDoubletAnchorHitIndex]);

    if(not( fabsf(dPhiChange) <= sdCut ))
    {
        pass = false;
    }
    
    float dAlphaThresholdValues[3];
    dAlphaThreshold(dAlphaThresholdValues, hitsInGPU, modulesInGPU, mdsInGPU, innerMiniDoubletAnchorHitIndex, outerMiniDoubletAnchorHitIndex, innerLowerModuleIndex, outerLowerModuleIndex, innerMDIndex, outerMDIndex);

    float innerMDAlpha = mdsInGPU.dphichanges[innerMDIndex];
    float outerMDAlpha = mdsInGPU.dphichanges[outerMDIndex];
    dAlphaInnerMDSegment = innerMDAlpha - dPhiChange;
    dAlphaOuterMDSegment = outerMDAlpha - dPhiChange;
    dAlphaInnerMDOuterMD = innerMDAlpha - outerMDAlpha;

    dAlphaInnerMDSegmentThreshold = dAlphaThresholdValues[0];
    dAlphaOuterMDSegmentThreshold = dAlphaThresholdValues[1];
    dAlphaInnerMDOuterMDThreshold = dAlphaThresholdValues[2];
    
    if(fabsf(dAlphaInnerMDSegment) >= dAlphaThresholdValues[0])
    {
        pass = false;
    }

    if(fabsf(dAlphaOuterMDSegment) >= dAlphaThresholdValues[1])
    {
        pass = false;
    }

    if(fabsf(dAlphaInnerMDOuterMD) >= dAlphaThresholdValues[2])
    {
        pass = false;
    }

    return pass;
}

__device__ bool SDL::runSegmentDefaultAlgo(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, unsigned int& innerLowerModuleIndex, unsigned int& outerLowerModuleIndex, unsigned int& innerMDIndex, unsigned int& outerMDIndex, float& zIn, float& zOut, float& rtIn, float& rtOut, float& dPhi, float& dPhiMin, float& dPhiMax, float& dPhiChange, float& dPhiChangeMin, float& dPhiChangeMax, float& dAlphaInnerMDSegment, float& dAlphaOuterMDSegment, float&
        dAlphaInnerMDOuterMD, float& zLo, float& zHi, float& rtLo, float& rtHi, float& sdCut, float& dAlphaInnerMDSegmentThreshold, float& dAlphaOuterMDSegmentThreshold, float& dAlphaInnerMDOuterMDThreshold, unsigned int& innerMiniDoubletAnchorHitIndex, unsigned int& outerMiniDoubletAnchorHitIndex)
{
    zLo = -999;
    zHi = -999;
    rtLo = -999;
    rtHi = -999;

    bool pass = true;

    if(modulesInGPU.subdets[innerLowerModuleIndex] == SDL::Barrel)
    {
        if(modulesInGPU.subdets[outerLowerModuleIndex] == SDL::Barrel)
        {
            pass = runSegmentDefaultAlgoBarrel(modulesInGPU, hitsInGPU, mdsInGPU, innerLowerModuleIndex, outerLowerModuleIndex, innerMDIndex, outerMDIndex, zIn, zOut, rtIn, rtOut, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD, zLo, zHi, sdCut, dAlphaInnerMDSegmentThreshold, dAlphaOuterMDSegmentThreshold, dAlphaInnerMDOuterMDThreshold, innerMiniDoubletAnchorHitIndex, outerMiniDoubletAnchorHitIndex);
        }
        else
        {
            pass = runSegmentDefaultAlgoEndcap(modulesInGPU, hitsInGPU, mdsInGPU, innerLowerModuleIndex, outerLowerModuleIndex, innerMDIndex, outerMDIndex, zIn, zOut, rtIn, rtOut, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD, rtLo, rtHi, sdCut, dAlphaInnerMDSegmentThreshold, dAlphaOuterMDSegmentThreshold, dAlphaInnerMDOuterMDThreshold, innerMiniDoubletAnchorHitIndex, outerMiniDoubletAnchorHitIndex);

        }
    }  

    else
    {
        if(modulesInGPU.subdets[outerLowerModuleIndex] == SDL::Endcap)
            pass = runSegmentDefaultAlgoEndcap(modulesInGPU, hitsInGPU, mdsInGPU, innerLowerModuleIndex, outerLowerModuleIndex, innerMDIndex, outerMDIndex, zIn, zOut, rtIn, rtOut, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD, rtLo, rtHi, sdCut, dAlphaInnerMDSegmentThreshold, dAlphaOuterMDSegmentThreshold, dAlphaInnerMDOuterMDThreshold, innerMiniDoubletAnchorHitIndex, outerMiniDoubletAnchorHitIndex);
        else
            pass = runSegmentDefaultAlgoBarrel(modulesInGPU, hitsInGPU, mdsInGPU, innerLowerModuleIndex, outerLowerModuleIndex, innerMDIndex, outerMDIndex, zIn, zOut, rtIn, rtOut, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD, zLo, zHi, sdCut, dAlphaInnerMDSegmentThreshold, dAlphaOuterMDSegmentThreshold, dAlphaInnerMDOuterMDThreshold, innerMiniDoubletAnchorHitIndex, outerMiniDoubletAnchorHitIndex);

    }

    return pass;
}
void SDL::printSegment(struct SDL::segments& segmentsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::hits& hitsInGPU, struct SDL::modules& modulesInGPU, unsigned int segmentIndex)
{
    unsigned int innerMDIndex = segmentsInGPU.mdIndices[segmentIndex * 2];
    unsigned int outerMDIndex = segmentsInGPU.mdIndices[segmentIndex * 2 + 1];
    std::cout<<std::endl;
    std::cout<<"sg_dPhiChange : "<<segmentsInGPU.dPhiChanges[segmentIndex] << std::endl<<std::endl;

    std::cout << "Inner Mini-Doublet" << std::endl;
    std::cout << "------------------------------" << std::endl;
    {
        IndentingOStreambuf indent(std::cout);
        printMD(mdsInGPU, hitsInGPU, modulesInGPU, innerMDIndex);
    }
    std::cout<<std::endl<<" Outer Mini-Doublet" <<std::endl;
    std::cout << "------------------------------" << std::endl;
    {
        IndentingOStreambuf indent(std::cout);
        printMD(mdsInGPU, hitsInGPU, modulesInGPU, outerMDIndex);
    }
}
