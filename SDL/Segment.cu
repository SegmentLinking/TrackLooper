# include "Segment.cuh"
//#ifdef CACHE_ALLOC
#include "allocate.h"
#include "Constants.h"
//#endif

///FIXME:NOTICE THE NEW maxPixelSegments!

void SDL::segments::resetMemory(unsigned int nMemoryLocationsx, unsigned int nLowerModules, unsigned int maxPixelSegments,cudaStream_t stream)
{
    // unsigned int nMemoryLocationsx = maxSegments * nLowerModules + maxPixelSegments;
    cudaMemsetAsync(mdIndices,0, nMemoryLocationsx * 2 * sizeof(unsigned int),stream);
    cudaMemsetAsync(innerLowerModuleIndices,0, nMemoryLocationsx * 2 * sizeof(uint16_t),stream);
    cudaMemsetAsync(nSegments, 0,(nLowerModules+1) * sizeof(unsigned int),stream);
    cudaMemsetAsync(totOccupancySegments, 0,(nLowerModules+1) * sizeof(unsigned int),stream);
    cudaMemsetAsync(dPhis, 0,(nMemoryLocationsx * 6 )*sizeof(FPX),stream);
    cudaMemsetAsync(ptIn, 0,(maxPixelSegments * 8)*sizeof(float),stream);
    cudaMemsetAsync(superbin, 0,(maxPixelSegments )*sizeof(int),stream);
    cudaMemsetAsync(pixelType, 0,(maxPixelSegments )*sizeof(int8_t),stream);
    cudaMemsetAsync(isQuad, 0,(maxPixelSegments )*sizeof(bool),stream);
    cudaMemsetAsync(isDup, 0,(maxPixelSegments )*sizeof(bool),stream);
    cudaMemsetAsync(score, 0,(maxPixelSegments )*sizeof(float),stream);
    cudaMemsetAsync(charge, 0,maxPixelSegments * sizeof(int),stream);
    cudaMemsetAsync(circleCenterX, 0,maxPixelSegments * sizeof(float),stream);
    cudaMemsetAsync(circleCenterY, 0,maxPixelSegments * sizeof(float),stream);
    cudaMemsetAsync(circleRadius, 0,maxPixelSegments * sizeof(float),stream);
    cudaMemsetAsync(partOfPT5, 0,maxPixelSegments * sizeof(bool),stream);
    cudaMemsetAsync(pLSHitsIdxs, 0,maxPixelSegments * sizeof(uint4),stream);
}


void SDL::createSegmentArrayRanges(struct modules& modulesInGPU, struct objectRanges& rangesInGPU, struct miniDoublets& mdsInGPU, uint16_t& nLowerModules, unsigned int& nTotalSegments, cudaStream_t stream, const uint16_t& maxPixelSegments)
{
    /*
        write code here that will deal with importing module parameters to CPU, and get the relevant occupancies for a given module!*/

    int *module_segmentModuleIndices;
    short* module_subdets;
    short* module_layers;
    short* module_rings;
    float* module_eta;
    uint16_t* module_nConnectedModules;
    module_segmentModuleIndices = (int*)cms::cuda::allocate_host((nLowerModules + 1) * sizeof(unsigned int), stream);
    module_subdets = (short*)cms::cuda::allocate_host(nLowerModules* sizeof(short), stream);
    module_layers = (short*)cms::cuda::allocate_host(nLowerModules* sizeof(short), stream);
    module_rings = (short*)cms::cuda::allocate_host(nLowerModules* sizeof(short), stream);
    module_eta = (float*)cms::cuda::allocate_host(nLowerModules* sizeof(float), stream);
    module_nConnectedModules = (uint16_t*)cms::cuda::allocate_host(nLowerModules * sizeof(uint16_t), stream);
    cudaMemcpyAsync(module_subdets,modulesInGPU.subdets,nLowerModules*sizeof(short),cudaMemcpyDeviceToHost,stream);
    cudaMemcpyAsync(module_layers,modulesInGPU.layers,nLowerModules * sizeof(short),cudaMemcpyDeviceToHost,stream);
    cudaMemcpyAsync(module_rings,modulesInGPU.rings,nLowerModules * sizeof(short),cudaMemcpyDeviceToHost,stream);
    cudaMemcpyAsync(module_eta,modulesInGPU.eta,nLowerModules * sizeof(float),cudaMemcpyDeviceToHost,stream);
    cudaMemcpyAsync(module_nConnectedModules,modulesInGPU.nConnectedModules,nLowerModules*sizeof(uint16_t),cudaMemcpyDeviceToHost,stream);
 
    cudaStreamSynchronize(stream);

    nTotalSegments = 0; //start!   
    for(uint16_t i = 0; i < nLowerModules; i++)
    {
        module_segmentModuleIndices[i] = nTotalSegments; //running counter - we start at the previous index!

        unsigned int occupancy;
        unsigned int category_number, eta_number;
        if (module_layers[i]<=3 && module_subdets[i]==5) category_number = 0;
        if (module_layers[i]>=4 && module_subdets[i]==5) category_number = 1;
        if (module_layers[i]<=2 && module_subdets[i]==4 && module_rings[i]>=11) category_number = 2;
        if (module_layers[i]>=3 && module_subdets[i]==4 && module_rings[i]>=8) category_number = 2;
        if (module_layers[i]<=2 && module_subdets[i]==4 && module_rings[i]<=10) category_number = 3;
        if (module_layers[i]>=3 && module_subdets[i]==4 && module_rings[i]<=7) category_number = 3;
        if (abs(module_eta[i])<0.75) eta_number=0;
        if (abs(module_eta[i])>0.75 && abs(module_eta[i])<1.5) eta_number=1;
        if (abs(module_eta[i])>1.5 && abs(module_eta[i])<2.25) eta_number=2;
        if (abs(module_eta[i])>2.25 && abs(module_eta[i])<3) eta_number=3;

        if (category_number == 0 && eta_number == 0) occupancy = 572;
        if (category_number == 0 && eta_number == 1) occupancy = 300;
        if (category_number == 0 && eta_number == 2) occupancy = 183;
        if (category_number == 0 && eta_number == 3) occupancy = 62;
        if (category_number == 1 && eta_number == 0) occupancy = 191;
        if (category_number == 1 && eta_number == 1) occupancy = 128;
        if (category_number == 2 && eta_number == 1) occupancy = 107;
        if (category_number == 2 && eta_number == 2) occupancy = 102;
        if (category_number == 3 && eta_number == 1) occupancy = 64;
        if (category_number == 3 && eta_number == 2) occupancy = 79;
        if (category_number == 3 && eta_number == 3) occupancy = 85;

        if(module_nConnectedModules[i] == 0) occupancy = 0;

        nTotalSegments += occupancy;
    }

    module_segmentModuleIndices[nLowerModules] = nTotalSegments;
    nTotalSegments += maxPixelSegments;

    cudaMemcpyAsync(rangesInGPU.segmentModuleIndices, module_segmentModuleIndices,  (nLowerModules + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    cms::cuda::free_host(module_segmentModuleIndices);
    cms::cuda::free_host(module_nConnectedModules);
    cms::cuda::free_host(module_subdets);
    cms::cuda::free_host(module_layers);
    cms::cuda::free_host(module_rings);
    cms::cuda::free_host(module_eta);
}

void SDL::createSegmentsInExplicitMemory(struct segments& segmentsInGPU, unsigned int nMemoryLocations, uint16_t nLowerModules, unsigned int maxPixelSegments, cudaStream_t stream)
{
    //FIXME:Since the number of pixel segments is 10x the number of regular segments per module, we need to provide
    //extra memory to the pixel segments
#ifdef CACHE_ALLOC
//    cudaStream_t stream=0; 
    int dev;
    cudaGetDevice(&dev);
    segmentsInGPU.mdIndices = (unsigned int*)cms::cuda::allocate_device(dev,nMemoryLocations*4 *sizeof(unsigned int),stream);
    segmentsInGPU.innerLowerModuleIndices = (uint16_t*)cms::cuda::allocate_device(dev,nMemoryLocations*2 *sizeof(uint16_t),stream);
    segmentsInGPU.nSegments = (unsigned int*)cms::cuda::allocate_device(dev, (nLowerModules + 1) *sizeof(unsigned int),stream);
    segmentsInGPU.totOccupancySegments = (unsigned int*)cms::cuda::allocate_device(dev, (nLowerModules + 1) *sizeof(unsigned int),stream);
    segmentsInGPU.dPhis = (FPX*)cms::cuda::allocate_device(dev,nMemoryLocations*6 *sizeof(FPX),stream);
    segmentsInGPU.ptIn = (float*)cms::cuda::allocate_device(dev, maxPixelSegments * 8 *sizeof(float),stream);
    segmentsInGPU.superbin = (int*)cms::cuda::allocate_device(dev,(maxPixelSegments) *sizeof(int),stream);
    segmentsInGPU.pixelType = (int8_t*)cms::cuda::allocate_device(dev,(maxPixelSegments) *sizeof(int8_t),stream);
    segmentsInGPU.isQuad = (bool*)cms::cuda::allocate_device(dev,(maxPixelSegments) *sizeof(bool),stream);
    segmentsInGPU.isDup = (bool*)cms::cuda::allocate_device(dev,(maxPixelSegments) *sizeof(bool),stream);
    segmentsInGPU.score = (float*)cms::cuda::allocate_device(dev,(maxPixelSegments) *sizeof(float),stream);
    segmentsInGPU.charge = (int*)cms::cuda::allocate_device(dev, maxPixelSegments * sizeof(int), stream);
    segmentsInGPU.circleCenterX = (float*)cms::cuda::allocate_device(dev, maxPixelSegments * sizeof(float), stream);
    segmentsInGPU.circleCenterY = (float*)cms::cuda::allocate_device(dev, maxPixelSegments * sizeof(float), stream);
    segmentsInGPU.circleRadius = (float*)cms::cuda::allocate_device(dev, maxPixelSegments * sizeof(float), stream);
    segmentsInGPU.partOfPT5 = (bool*)cms::cuda::allocate_device(dev, maxPixelSegments * sizeof(bool), stream);
    segmentsInGPU.pLSHitsIdxs = (uint4*)cms::cuda::allocate_device(dev, maxPixelSegments * sizeof(uint4), stream);
    segmentsInGPU.nMemoryLocations = (unsigned int*)cms::cuda::allocate_device(dev, sizeof(unsigned int), stream);
#else
    cudaMalloc(&segmentsInGPU.mdIndices, nMemoryLocations * 4 * sizeof(unsigned int));
    cudaMalloc(&segmentsInGPU.innerLowerModuleIndices, nMemoryLocations * 2 * sizeof(uint16_t));
    cudaMalloc(&segmentsInGPU.nSegments, (nLowerModules + 1) * sizeof(unsigned int));
    cudaMalloc(&segmentsInGPU.totOccupancySegments, (nLowerModules + 1) * sizeof(unsigned int));
    cudaMalloc(&segmentsInGPU.dPhis, nMemoryLocations * 6 *sizeof(FPX));
    cudaMalloc(&segmentsInGPU.ptIn, maxPixelSegments * 8*sizeof(float));
    cudaMalloc(&segmentsInGPU.superbin, (maxPixelSegments )*sizeof(int));
    cudaMalloc(&segmentsInGPU.pixelType, (maxPixelSegments )*sizeof(int8_t));
    cudaMalloc(&segmentsInGPU.isQuad, (maxPixelSegments )*sizeof(bool));
    cudaMalloc(&segmentsInGPU.isDup, (maxPixelSegments )*sizeof(bool));
    cudaMalloc(&segmentsInGPU.score, (maxPixelSegments )*sizeof(float));
    cudaMalloc(&segmentsInGPU.charge, maxPixelSegments * sizeof(int));
    cudaMalloc(&segmentsInGPU.circleCenterX, maxPixelSegments * sizeof(float));
    cudaMalloc(&segmentsInGPU.circleCenterY, maxPixelSegments * sizeof(float));
    cudaMalloc(&segmentsInGPU.circleRadius, maxPixelSegments * sizeof(float));
    cudaMalloc(&segmentsInGPU.partOfPT5, maxPixelSegments * sizeof(bool));
    cudaMalloc(&segmentsInGPU.pLSHitsIdxs, maxPixelSegments * sizeof(uint4));
    cudaMalloc(&segmentsInGPU.nMemoryLocations, sizeof(unsigned int));
#endif

    //segmentsInGPU.innerLowerModuleIndices = segmentsInGPU.mdIndices + nMemoryLocations * 2;
    segmentsInGPU.outerLowerModuleIndices = segmentsInGPU.innerLowerModuleIndices + nMemoryLocations;
    segmentsInGPU.innerMiniDoubletAnchorHitIndices = segmentsInGPU.mdIndices + nMemoryLocations * 2;
    segmentsInGPU.outerMiniDoubletAnchorHitIndices = segmentsInGPU.mdIndices + nMemoryLocations * 3;

    segmentsInGPU.dPhiMins = segmentsInGPU.dPhis + nMemoryLocations;
    segmentsInGPU.dPhiMaxs = segmentsInGPU.dPhis + nMemoryLocations * 2;
    segmentsInGPU.dPhiChanges = segmentsInGPU.dPhis + nMemoryLocations * 3;
    segmentsInGPU.dPhiChangeMins = segmentsInGPU.dPhis + nMemoryLocations * 4;
    segmentsInGPU.dPhiChangeMaxs = segmentsInGPU.dPhis + nMemoryLocations * 5;

    segmentsInGPU.ptErr  = segmentsInGPU.ptIn + maxPixelSegments;
    segmentsInGPU.px     = segmentsInGPU.ptIn + maxPixelSegments * 2;
    segmentsInGPU.py     = segmentsInGPU.ptIn + maxPixelSegments * 3;
    segmentsInGPU.pz     = segmentsInGPU.ptIn + maxPixelSegments * 4;
    segmentsInGPU.etaErr = segmentsInGPU.ptIn + maxPixelSegments * 5;
    segmentsInGPU.eta    = segmentsInGPU.ptIn + maxPixelSegments * 6;
    segmentsInGPU.phi    = segmentsInGPU.ptIn + maxPixelSegments * 7;

    cudaMemsetAsync(segmentsInGPU.nSegments,0, (nLowerModules + 1) * sizeof(unsigned int),stream);
    cudaMemsetAsync(segmentsInGPU.totOccupancySegments,0, (nLowerModules + 1) * sizeof(unsigned int),stream);
    cudaMemsetAsync(segmentsInGPU.partOfPT5, false, maxPixelSegments * sizeof(bool),stream);
    cudaMemsetAsync(segmentsInGPU.pLSHitsIdxs, 0, maxPixelSegments * sizeof(uint4),stream);
    cudaMemsetAsync(segmentsInGPU.nMemoryLocations, nMemoryLocations, sizeof(unsigned int), stream);
    cudaStreamSynchronize(stream);

}

SDL::segments::segments()
{
    superbin = nullptr;
    pixelType = nullptr;
    isQuad = nullptr;
    isDup = nullptr;
    score = nullptr;
    circleRadius = nullptr;
    charge = nullptr;
    circleCenterX = nullptr;
    circleCenterY = nullptr;
    mdIndices = nullptr;
    innerLowerModuleIndices = nullptr;
    outerLowerModuleIndices = nullptr;
    innerMiniDoubletAnchorHitIndices = nullptr;
    outerMiniDoubletAnchorHitIndices = nullptr;

    nSegments = nullptr;
    totOccupancySegments = nullptr;
    dPhis = nullptr;
    dPhiMins = nullptr;
    dPhiMaxs = nullptr;
    dPhiChanges = nullptr;
    dPhiChangeMins = nullptr;
    dPhiChangeMaxs = nullptr;
    partOfPT5 = nullptr;
    pLSHitsIdxs = nullptr;

}

SDL::segments::~segments()
{
}

void SDL::segments::freeMemoryCache()
{
    int dev;
    cudaGetDevice(&dev);
    cms::cuda::free_device(dev,mdIndices);
    cms::cuda::free_device(dev,innerLowerModuleIndices);
    cms::cuda::free_device(dev,dPhis);
    cms::cuda::free_device(dev,ptIn);
    cms::cuda::free_device(dev,nSegments);
    cms::cuda::free_device(dev,totOccupancySegments);
    cms::cuda::free_device(dev, charge);
    cms::cuda::free_device(dev,superbin);
    cms::cuda::free_device(dev,pixelType);
    cms::cuda::free_device(dev,isQuad);
    cms::cuda::free_device(dev,isDup);
    cms::cuda::free_device(dev,score);
    cms::cuda::free_device(dev, circleCenterX);
    cms::cuda::free_device(dev, circleCenterY);
    cms::cuda::free_device(dev, circleRadius);
    cms::cuda::free_device(dev, partOfPT5);
    cms::cuda::free_device(dev, pLSHitsIdxs);
    cms::cuda::free_device(dev, nMemoryLocations);
}
void SDL::segments::freeMemory(cudaStream_t stream)
{
    cudaFree(mdIndices);
    cudaFree(innerLowerModuleIndices);
    cudaFree(nSegments);
    cudaFree(totOccupancySegments);
    cudaFree(dPhis);
    cudaFree(ptIn);
    cudaFree(superbin);
    cudaFree(pixelType);
    cudaFree(isQuad);
    cudaFree(isDup);
    cudaFree(score);
    cudaFree(charge);
    cudaFree(circleCenterX);
    cudaFree(circleCenterY);
    cudaFree(circleRadius);
    cudaFree(partOfPT5);
    cudaFree(pLSHitsIdxs);
    cudaFree(nMemoryLocations);
}


__device__ void SDL::addSegmentToMemory(struct segments& segmentsInGPU, unsigned int lowerMDIndex, unsigned int upperMDIndex, uint16_t innerLowerModuleIndex, uint16_t outerLowerModuleIndex, unsigned int innerMDAnchorHitIndex, unsigned int outerMDAnchorHitIndex, float& dPhi, float& dPhiMin, float& dPhiMax, float& dPhiChange, float& dPhiChangeMin, float& dPhiChangeMax, unsigned int idx)
{
    //idx will be computed in the kernel, which is the index into which the 
    //segment will be written
    //nSegments will be incremented in the kernel
    //printf("seg: %u %u %u %u\n",lowerMDIndex, upperMDIndex,innerLowerModuleIndex,outerLowerModuleIndex);
    segmentsInGPU.mdIndices[idx * 2] = lowerMDIndex;
    segmentsInGPU.mdIndices[idx * 2 + 1] = upperMDIndex;
    segmentsInGPU.innerLowerModuleIndices[idx] = innerLowerModuleIndex;
    segmentsInGPU.outerLowerModuleIndices[idx] = outerLowerModuleIndex;
    segmentsInGPU.innerMiniDoubletAnchorHitIndices[idx] = innerMDAnchorHitIndex;
    segmentsInGPU.outerMiniDoubletAnchorHitIndices[idx] = outerMDAnchorHitIndex;

    segmentsInGPU.dPhis[idx]          = __F2H(dPhi);
    segmentsInGPU.dPhiMins[idx]       = __F2H(dPhiMin);
    segmentsInGPU.dPhiMaxs[idx]       = __F2H(dPhiMax);
    segmentsInGPU.dPhiChanges[idx]    = __F2H(dPhiChange);
    segmentsInGPU.dPhiChangeMins[idx] = __F2H(dPhiChangeMin);
    segmentsInGPU.dPhiChangeMaxs[idx] = __F2H(dPhiChangeMax);

}

__device__ void SDL::addPixelSegmentToMemory(struct segments& segmentsInGPU, struct miniDoublets& mdsInGPU, struct modules& modulesInGPU, unsigned int innerMDIndex, unsigned int outerMDIndex, uint16_t pixelModuleIndex, unsigned int hitIdxs[4], unsigned int innerAnchorHitIndex, unsigned int outerAnchorHitIndex, float dPhiChange, float ptIn, float ptErr, float px, float py, float pz, float etaErr, float eta, float phi, int charge, unsigned int idx, unsigned int pixelSegmentArrayIndex, int superbin,
            int8_t pixelType, short isQuad, float score)
{
    segmentsInGPU.mdIndices[idx * 2] = innerMDIndex;
    segmentsInGPU.mdIndices[idx * 2 + 1] = outerMDIndex;
    segmentsInGPU.innerLowerModuleIndices[idx] = pixelModuleIndex;
    segmentsInGPU.outerLowerModuleIndices[idx] = pixelModuleIndex;
    segmentsInGPU.innerMiniDoubletAnchorHitIndices[idx] = innerAnchorHitIndex;
    segmentsInGPU.outerMiniDoubletAnchorHitIndices[idx] = outerAnchorHitIndex;
    segmentsInGPU.dPhiChanges[idx] = __F2H(dPhiChange);
    segmentsInGPU.ptIn[pixelSegmentArrayIndex] = ptIn;
    segmentsInGPU.ptErr[pixelSegmentArrayIndex] = ptErr;
    segmentsInGPU.px[pixelSegmentArrayIndex] = px;
    segmentsInGPU.py[pixelSegmentArrayIndex] = py;
    segmentsInGPU.pz[pixelSegmentArrayIndex] = pz;
    segmentsInGPU.etaErr[pixelSegmentArrayIndex] = etaErr;
    segmentsInGPU.eta[pixelSegmentArrayIndex] = eta;
    segmentsInGPU.phi[pixelSegmentArrayIndex] = phi;
    segmentsInGPU.charge[pixelSegmentArrayIndex] = charge;

    segmentsInGPU.superbin[pixelSegmentArrayIndex] = superbin;
    segmentsInGPU.pixelType[pixelSegmentArrayIndex] = pixelType;
    segmentsInGPU.isQuad[pixelSegmentArrayIndex] = isQuad;
    segmentsInGPU.isDup[pixelSegmentArrayIndex] = false;
    segmentsInGPU.score[pixelSegmentArrayIndex] = score;

    segmentsInGPU.pLSHitsIdxs[pixelSegmentArrayIndex].x = hitIdxs[0];
    segmentsInGPU.pLSHitsIdxs[pixelSegmentArrayIndex].y = hitIdxs[1];
    segmentsInGPU.pLSHitsIdxs[pixelSegmentArrayIndex].z = hitIdxs[2];
    segmentsInGPU.pLSHitsIdxs[pixelSegmentArrayIndex].w = hitIdxs[3];

    //computing circle parameters
    /*
       The two anchor hits are r3PCA and r3LH. p3PCA pt, eta, phi is hitIndex1 x, y, z
    */
    float circleRadius = mdsInGPU.outerX[innerMDIndex] / (2 * k2Rinv1GeVf);
    float circlePhi = mdsInGPU.outerZ[innerMDIndex];
    float candidateCenterXs[] = {mdsInGPU.anchorX[innerMDIndex] + circleRadius * sinf(circlePhi), mdsInGPU.anchorX[innerMDIndex] - circleRadius * sinf(circlePhi)};
    float candidateCenterYs[] = {mdsInGPU.anchorY[innerMDIndex] - circleRadius * cosf(circlePhi), mdsInGPU.anchorY[innerMDIndex] + circleRadius * cosf(circlePhi)};

    //check which of the circles can accommodate r3LH better (we won't get perfect agreement)
    float bestChiSquared = 123456789.f;
    float chiSquared;
    size_t bestIndex;
    for(size_t i = 0; i < 2; i++)
    {
        chiSquared = fabsf(sqrtf((mdsInGPU.anchorX[outerMDIndex] - candidateCenterXs[i]) * (mdsInGPU.anchorX[outerMDIndex] - candidateCenterXs[i]) + (mdsInGPU.anchorY[outerMDIndex] - candidateCenterYs[i]) * (mdsInGPU.anchorY[outerMDIndex] - candidateCenterYs[i])) - circleRadius);
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

__device__ void SDL::dAlphaThreshold(float* dAlphaThresholdValues, struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, float& xIn, float& yIn, float& zIn, float& rtIn, float& xOut, float& yOut, float& zOut, float& rtOut, uint16_t& innerLowerModuleIndex, uint16_t& outerLowerModuleIndex, unsigned int& innerMDIndex, unsigned int& outerMDIndex)
{
    float sdMuls = (modulesInGPU.subdets[innerLowerModuleIndex] == SDL::Barrel) ? miniMulsPtScaleBarrel[modulesInGPU.layers[innerLowerModuleIndex]-1] * 3.f/ptCut : miniMulsPtScaleEndcap[modulesInGPU.layers[innerLowerModuleIndex]-1] * 3.f/ptCut;

    //more accurate then outer rt - inner rt
    float segmentDr = sqrtf((yOut - yIn) * (yOut - yIn) + (xOut - xIn) * (xOut - xIn));

    const float dAlpha_Bfield = asinf(fminf(segmentDr * k2Rinv1GeVf/ptCut, sinAlphaMax));

    bool isInnerTilted = modulesInGPU.subdets[innerLowerModuleIndex] == SDL::Barrel and modulesInGPU.sides[innerLowerModuleIndex] != SDL::Center;
    bool isOuterTilted = modulesInGPU.subdets[outerLowerModuleIndex] == SDL::Barrel and modulesInGPU.sides[outerLowerModuleIndex] != SDL::Center;

    float& drdzInner = modulesInGPU.drdzs[innerLowerModuleIndex];
    float& drdzOuter = modulesInGPU.drdzs[outerLowerModuleIndex];
    float innerModuleGapSize = SDL::moduleGapSize_seg(modulesInGPU, innerLowerModuleIndex);
    float outerModuleGapSize = SDL::moduleGapSize_seg(modulesInGPU, outerLowerModuleIndex);
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

    float dAlpha_res_inner = 0.02f/miniDelta * (modulesInGPU.subdets[innerLowerModuleIndex] == SDL::Barrel ? 1.0f : fabsf(zIn)/rtIn);
    float dAlpha_res_outer = 0.02f/miniDelta * (modulesInGPU.subdets[outerLowerModuleIndex] == SDL::Barrel ? 1.0f : fabsf(zOut)/rtOut);

 
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

__device__ bool SDL::runSegmentDefaultAlgoEndcap(struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, uint16_t& innerLowerModuleIndex, uint16_t& outerLowerModuleIndex, unsigned int& innerMDIndex, unsigned int& outerMDIndex, float& zIn, float& zOut, float& rtIn, float& rtOut, float& dPhi, float& dPhiMin, float& dPhiMax, float& dPhiChange, float& dPhiChangeMin, float& dPhiChangeMax, float& dAlphaInnerMDSegment, float& dAlphaOuterMDSegment,
        float& rtLo, float& rtHi, float& sdCut, float& dAlphaInnerMDSegmentThreshold, float& dAlphaOuterMDSegmentThreshold, float& dAlphaInnerMDOuterMDThreshold, float&
        dAlphaInnerMDOuterMD)
{
    bool pass = true;
   
    float xIn, yIn;    
    float xOut, yOut, xOutHigh, yOutHigh, xOutLow, yOutLow;

    xIn = mdsInGPU.anchorX[innerMDIndex];
    yIn = mdsInGPU.anchorY[innerMDIndex];
    zIn = mdsInGPU.anchorZ[innerMDIndex];
    rtIn = mdsInGPU.anchorRt[innerMDIndex];

    xOut = mdsInGPU.anchorX[outerMDIndex];
    yOut = mdsInGPU.anchorY[outerMDIndex];
    zOut = mdsInGPU.anchorZ[outerMDIndex];
    rtOut = mdsInGPU.anchorRt[outerMDIndex];

    xOutHigh = mdsInGPU.anchorHighEdgeX[outerMDIndex];
    yOutHigh = mdsInGPU.anchorHighEdgeY[outerMDIndex];
    xOutLow = mdsInGPU.anchorLowEdgeX[outerMDIndex];
    yOutLow = mdsInGPU.anchorLowEdgeY[outerMDIndex];
    bool outerLayerEndcapTwoS = (modulesInGPU.subdets[outerLowerModuleIndex] == SDL::Endcap) & (modulesInGPU.moduleType[outerLowerModuleIndex] == SDL::TwoS);

    
    float sdSlope = asinf(fminf(rtOut * k2Rinv1GeVf / ptCut, sinAlphaMax));
    float sdPVoff = 0.1/rtOut;
    float disks2SMinRadius = 60.f;

    float rtGeom =  ((rtIn < disks2SMinRadius && rtOut < disks2SMinRadius) ? (2.f * pixelPSZpitch)
            : ((rtIn < disks2SMinRadius || rtOut < disks2SMinRadius) ? (pixelPSZpitch + strip2SZpitch)
            : (2.f * strip2SZpitch)));


    //cut 0 - z compatibility
    pass =  pass and (zIn * zOut >= 0);
    if(not pass) return pass;

    float dz = zOut - zIn;
    float dLum = copysignf(deltaZLum, zIn);
    float drtDzScale = sdSlope/tanf(sdSlope);

    rtLo = fmaxf(rtIn * (1.f + dz / (zIn + dLum) * drtDzScale) - rtGeom,  rtIn - 0.5f * rtGeom); //rt should increase
    rtHi = rtIn * (zOut - dLum) / (zIn - dLum) + rtGeom; //dLum for luminous; rGeom for measurement size; no tanTheta_loc(pt) correction

    //completeness

    pass =  pass and ((rtOut >= rtLo) & (rtOut <= rtHi));
    if(not pass) return pass;

    dPhi = deltaPhi(xIn, yIn, zIn, xOut, yOut, zOut);

    sdCut = sdSlope;
    if(outerLayerEndcapTwoS)
    {
        float dPhiPos_high = deltaPhi(xIn, yIn, zIn, xOutHigh, yOutHigh, zOut);
        float dPhiPos_low = deltaPhi(xIn, yIn, zIn, xOutLow, yOutLow, zOut);
        
        dPhiMax = fabsf(dPhiPos_high) > fabsf(dPhiPos_low) ? dPhiPos_high : dPhiPos_low;
        dPhiMin = fabsf(dPhiPos_high) > fabsf(dPhiPos_low) ? dPhiPos_low : dPhiPos_high;
    }
    else
    {
        dPhiMax = dPhi;
        dPhiMin = dPhi;
    }
    pass =  pass and (fabsf(dPhi) <= sdCut);
    if(not pass) return pass;

    float dzFrac = dz/zIn;
    dPhiChange = dPhi/dzFrac * (1.f + dzFrac);
    dPhiChangeMin = dPhiMin/dzFrac * (1.f + dzFrac);
    dPhiChangeMax = dPhiMax/dzFrac * (1.f + dzFrac);
    
    pass =  pass and (fabsf(dPhiChange) <= sdCut);
    if(not pass) return pass;

    float dAlphaThresholdValues[3];
    dAlphaThreshold(dAlphaThresholdValues, modulesInGPU, mdsInGPU, xIn, yIn, zIn, rtIn, xOut, yOut, zOut, rtOut,innerLowerModuleIndex, outerLowerModuleIndex, innerMDIndex, outerMDIndex);

    dAlphaInnerMDSegmentThreshold = dAlphaThresholdValues[0];
    dAlphaOuterMDSegmentThreshold = dAlphaThresholdValues[1];
    dAlphaInnerMDOuterMDThreshold = dAlphaThresholdValues[2];

    float innerMDAlpha = mdsInGPU.dphichanges[innerMDIndex];
    float outerMDAlpha = mdsInGPU.dphichanges[outerMDIndex];
    dAlphaInnerMDSegment = innerMDAlpha - dPhiChange;
    dAlphaOuterMDSegment = outerMDAlpha - dPhiChange;
    dAlphaInnerMDOuterMD = innerMDAlpha - outerMDAlpha;
   
    pass =  pass and (fabsf(dAlphaInnerMDSegment) < dAlphaThresholdValues[0]);
    if(not pass) return pass;
    pass =  pass and (fabsf(dAlphaOuterMDSegment) < dAlphaThresholdValues[1]);
    if(not pass) return pass;
    pass =  pass and (fabsf(dAlphaInnerMDOuterMD) < dAlphaThresholdValues[2]);

    return pass;
}


__device__ bool SDL::runSegmentDefaultAlgoBarrel(struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, uint16_t& innerLowerModuleIndex, uint16_t& outerLowerModuleIndex, unsigned int& innerMDIndex, unsigned int& outerMDIndex, float& zIn, float& zOut, float& rtIn, float& rtOut, float& dPhi, float& dPhiMin, float& dPhiMax, float& dPhiChange, float& dPhiChangeMin, float& dPhiChangeMax, float& dAlphaInnerMDSegment, float& dAlphaOuterMDSegment, float&
        dAlphaInnerMDOuterMD, float& zLo, float& zHi, float& sdCut, float& dAlphaInnerMDSegmentThreshold, float& dAlphaOuterMDSegmentThreshold, float& dAlphaInnerMDOuterMDThreshold)
{
    bool pass = true;
   
    float sdMuls = (modulesInGPU.subdets[innerLowerModuleIndex] == SDL::Barrel) ? miniMulsPtScaleBarrel[modulesInGPU.layers[innerLowerModuleIndex]-1] * 3.f/ptCut : miniMulsPtScaleEndcap[modulesInGPU.layers[innerLowerModuleIndex]-1] * 3.f/ptCut;


    float xIn, yIn, xOut, yOut;

    xIn = mdsInGPU.anchorX[innerMDIndex];
    yIn = mdsInGPU.anchorY[innerMDIndex];
    zIn = mdsInGPU.anchorZ[innerMDIndex];
    rtIn = mdsInGPU.anchorRt[innerMDIndex];

    xOut = mdsInGPU.anchorX[outerMDIndex];
    yOut = mdsInGPU.anchorY[outerMDIndex];
    zOut = mdsInGPU.anchorZ[outerMDIndex];
    rtOut = mdsInGPU.anchorRt[outerMDIndex];

    float sdSlope = asinf(fminf(rtOut * k2Rinv1GeVf / ptCut, sinAlphaMax));
    float sdPVoff = 0.1f/rtOut;
    float dzDrtScale = tanf(sdSlope)/sdSlope; //FIXME: need appropriate value

    const float zGeom = modulesInGPU.layers[innerLowerModuleIndex] <= 2 ? 2.f * pixelPSZpitch : 2.f * strip2SZpitch;

    zLo = zIn + (zIn - deltaZLum) * (rtOut / rtIn - 1.f) * (zIn > 0.f ? 1.f : dzDrtScale) - zGeom; //slope-correction only on outer end
    zHi = zIn + (zIn + deltaZLum) * (rtOut / rtIn - 1.f) * (zIn < 0.f ? 1.f : dzDrtScale) + zGeom;

    pass =  pass and ((zOut >= zLo) & (zOut <= zHi));
    if(not pass) return pass;

    sdCut = sdSlope + sqrtf(sdMuls * sdMuls + sdPVoff * sdPVoff);

    dPhi  = deltaPhi(xIn, yIn, zIn, xOut, yOut, zOut);

    pass =  pass and (fabsf(dPhi) <= sdCut);
    if(not pass) return pass;

    dPhiChange = deltaPhiChange(xIn, yIn, zIn, xOut, yOut, zOut);

    pass =  pass and (fabsf(dPhiChange) <= sdCut);
    if(not pass) return pass;

    float dAlphaThresholdValues[3];
    dAlphaThreshold(dAlphaThresholdValues, modulesInGPU, mdsInGPU, xIn, yIn, zIn, rtIn, xOut, yOut, zOut, rtOut, innerLowerModuleIndex, outerLowerModuleIndex, innerMDIndex, outerMDIndex);

    float innerMDAlpha = mdsInGPU.dphichanges[innerMDIndex];
    float outerMDAlpha = mdsInGPU.dphichanges[outerMDIndex];
    dAlphaInnerMDSegment = innerMDAlpha - dPhiChange;
    dAlphaOuterMDSegment = outerMDAlpha - dPhiChange;
    dAlphaInnerMDOuterMD = innerMDAlpha - outerMDAlpha;

    dAlphaInnerMDSegmentThreshold = dAlphaThresholdValues[0];
    dAlphaOuterMDSegmentThreshold = dAlphaThresholdValues[1];
    dAlphaInnerMDOuterMDThreshold = dAlphaThresholdValues[2];
    
    pass =  pass and (fabsf(dAlphaInnerMDSegment) < dAlphaThresholdValues[0]);
    if(not pass) return pass;
    pass =  pass and (fabsf(dAlphaOuterMDSegment) < dAlphaThresholdValues[1]);
    if(not pass) return pass;
    pass =  pass and (fabsf(dAlphaInnerMDOuterMD) < dAlphaThresholdValues[2]);

    return pass;
}

__device__ bool SDL::runSegmentDefaultAlgo(struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, uint16_t& innerLowerModuleIndex, uint16_t& outerLowerModuleIndex, unsigned int& innerMDIndex, unsigned int& outerMDIndex, float& zIn, float& zOut, float& rtIn, float& rtOut, float& dPhi, float& dPhiMin, float& dPhiMax, float& dPhiChange, float& dPhiChangeMin, float& dPhiChangeMax, float& dAlphaInnerMDSegment, float& dAlphaOuterMDSegment, float&
        dAlphaInnerMDOuterMD, float& zLo, float& zHi, float& rtLo, float& rtHi, float& sdCut, float& dAlphaInnerMDSegmentThreshold, float& dAlphaOuterMDSegmentThreshold, float& dAlphaInnerMDOuterMDThreshold)
{
    zLo = -999.f;
    zHi = -999.f;
    rtLo = -999.f;
    rtHi = -999.f;

    //bool pass = true;

    if(modulesInGPU.subdets[innerLowerModuleIndex] == SDL::Barrel and modulesInGPU.subdets[outerLowerModuleIndex] == SDL::Barrel)
    {
        return runSegmentDefaultAlgoBarrel(modulesInGPU, mdsInGPU, innerLowerModuleIndex, outerLowerModuleIndex, innerMDIndex, outerMDIndex, zIn, zOut, rtIn, rtOut, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD, zLo, zHi, sdCut, dAlphaInnerMDSegmentThreshold, dAlphaOuterMDSegmentThreshold, dAlphaInnerMDOuterMDThreshold);
    }
    else
    {
        return runSegmentDefaultAlgoEndcap(modulesInGPU, mdsInGPU, innerLowerModuleIndex, outerLowerModuleIndex, innerMDIndex, outerMDIndex, zIn, zOut, rtIn, rtOut, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD, rtLo, rtHi, sdCut, dAlphaInnerMDSegmentThreshold, dAlphaOuterMDSegmentThreshold, dAlphaInnerMDOuterMDThreshold);
    }

}
void SDL::printSegment(struct SDL::segments& segmentsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::hits& hitsInGPU, struct SDL::modules& modulesInGPU, unsigned int segmentIndex)
{
    unsigned int innerMDIndex = segmentsInGPU.mdIndices[segmentIndex * 2];
    unsigned int outerMDIndex = segmentsInGPU.mdIndices[segmentIndex * 2 + 1];
    std::cout<<std::endl;
    std::cout<<"sg_dPhiChange : "<<__H2F(segmentsInGPU.dPhiChanges[segmentIndex]) << std::endl<<std::endl;

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
__device__ inline float SDL::isTighterTiltedModules_seg(struct modules& modulesInGPU, unsigned int moduleIndex)
{
    // The "tighter" tilted modules are the subset of tilted modules that have smaller spacing
    // This is the same as what was previously considered as"isNormalTiltedModules"
    // See Figure 9.1 of https://cds.cern.ch/record/2272264/files/CMS-TDR-014.pdf
    short subdet = modulesInGPU.subdets[moduleIndex];
    short layer = modulesInGPU.layers[moduleIndex];
    short side = modulesInGPU.sides[moduleIndex];
    short rod = modulesInGPU.rods[moduleIndex];

    if (
           (subdet == Barrel and side != Center and layer== 3)
           or (subdet == Barrel and side == NegZ and layer == 2 and rod > 5)
           or (subdet == Barrel and side == PosZ and layer == 2 and rod < 8)
           or (subdet == Barrel and side == NegZ and layer == 1 and rod > 9)
           or (subdet == Barrel and side == PosZ and layer == 1 and rod < 4)
       )
        return true;
    else
        return false;

}

__device__ inline float SDL::isTighterTiltedModules_seg(short subdet, short layer, short side, short rod)
{
    // The "tighter" tilted modules are the subset of tilted modules that have smaller spacing
    // This is the same as what was previously considered as"isNormalTiltedModules"
    // See Figure 9.1 of https://cds.cern.ch/record/2272264/files/CMS-TDR-014.pdf
    if (
           (subdet == Barrel and side != Center and layer== 3)
           or (subdet == Barrel and side == NegZ and layer == 2 and rod > 5)
           or (subdet == Barrel and side == PosZ and layer == 2 and rod < 8)
           or (subdet == Barrel and side == NegZ and layer == 1 and rod > 9)
           or (subdet == Barrel and side == PosZ and layer == 1 and rod < 4)
       )
        return true;
    else
        return false;

}

//__device__ float SDL::moduleGapSize_seg(struct modules& modulesInGPU, unsigned int moduleIndex)
__device__ float SDL::moduleGapSize_seg(short layer, short ring, short subdet, short side, short rod)
{
    float miniDeltaTilted[3] = {0.26f, 0.26f, 0.26f};
    float miniDeltaFlat[6] ={0.26f, 0.16f, 0.16f, 0.18f, 0.18f, 0.18f};
    float miniDeltaLooseTilted[3] = {0.4f,0.4f,0.4f};
    float miniDeltaEndcap[5][15];

    for (size_t i = 0; i < 5; i++)
    {
        for (size_t j = 0; j < 15; j++)
        {
            if (i == 0 || i == 1)
            {
                if (j < 10)
                {
                    miniDeltaEndcap[i][j] = 0.4f;
                }
                else
                {
                    miniDeltaEndcap[i][j] = 0.18f;
                }
            }
            else if (i == 2 || i == 3)
            {
                if (j < 8)
                {
                    miniDeltaEndcap[i][j] = 0.4f;
                }
                else
                {
                    miniDeltaEndcap[i][j]  = 0.18f;
                }
            }
            else
            {
                if (j < 9)
                {
                    miniDeltaEndcap[i][j] = 0.4f;
                }
                else
                {
                    miniDeltaEndcap[i][j] = 0.18f;
                }
            }
        }
    }


    unsigned int iL = layer-1;
    unsigned int iR = ring - 1;

    float moduleSeparation = 0;

    if (subdet == Barrel and side == Center)
    {
        moduleSeparation = miniDeltaFlat[iL];
    }
    else if (isTighterTiltedModules_seg(subdet, layer, side, rod))
    {
        moduleSeparation = miniDeltaTilted[iL];
    }
    else if (subdet == Endcap)
    {
        moduleSeparation = miniDeltaEndcap[iL][iR];
    }
    else //Loose tilted modules
    {
        moduleSeparation = miniDeltaLooseTilted[iL];
    }

    return moduleSeparation;
}


__device__ float SDL::moduleGapSize_seg(struct modules& modulesInGPU, unsigned int moduleIndex)
{
    float miniDeltaTilted[3] = {0.26f, 0.26f, 0.26f};
    float miniDeltaFlat[6] ={0.26f, 0.16f, 0.16f, 0.18f, 0.18f, 0.18f};
    float miniDeltaLooseTilted[3] = {0.4f,0.4f,0.4f};
    float miniDeltaEndcap[5][15];

    for (size_t i = 0; i < 5; i++)
    {
        for (size_t j = 0; j < 15; j++)
        {
            if (i == 0 || i == 1)
            {
                if (j < 10)
                {
                    miniDeltaEndcap[i][j] = 0.4f;
                }
                else
                {
                    miniDeltaEndcap[i][j] = 0.18f;
                }
            }
            else if (i == 2 || i == 3)
            {
                if (j < 8)
                {
                    miniDeltaEndcap[i][j] = 0.4f;
                }
                else
                {
                    miniDeltaEndcap[i][j]  = 0.18f;
                }
            }
            else
            {
                if (j < 9)
                {
                    miniDeltaEndcap[i][j] = 0.4f;
                }
                else
                {
                    miniDeltaEndcap[i][j] = 0.18f;
                }
            }
        }
    }


    unsigned int iL = modulesInGPU.layers[moduleIndex]-1;
    unsigned int iR = modulesInGPU.rings[moduleIndex] - 1;
    short subdet = modulesInGPU.subdets[moduleIndex];
    short side = modulesInGPU.sides[moduleIndex];

    float moduleSeparation = 0;

    if (subdet == Barrel and side == Center)
    {
        moduleSeparation = miniDeltaFlat[iL];
    }
    else if (isTighterTiltedModules_seg(modulesInGPU, moduleIndex))
    {
        moduleSeparation = miniDeltaTilted[iL];
    }
    else if (subdet == Endcap)
    {
        moduleSeparation = miniDeltaEndcap[iL][iR];
    }
    else //Loose tilted modules
    {
        moduleSeparation = miniDeltaLooseTilted[iL];
    }

    return moduleSeparation;
}
__global__ void SDL::createSegmentsInGPUv2(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::objectRanges& rangesInGPU)
{
    int blockxSize = gridDim.x;
    int blockySize = blockDim.y;
    int blockzSize = blockDim.x;
    for(uint16_t innerLowerModuleIndex = blockIdx.x ; innerLowerModuleIndex< (*modulesInGPU.nLowerModules); innerLowerModuleIndex += blockxSize){

    unsigned int nConnectedModules = modulesInGPU.nConnectedModules[innerLowerModuleIndex];
//printf("HERE:  %d nConnectedModules = %d\n", innerLowerModuleIndex, nConnectedModules);

    for(uint16_t outerLowerModuleArrayIdx = threadIdx.y; outerLowerModuleArrayIdx< nConnectedModules; outerLowerModuleArrayIdx += blockySize){

        uint16_t outerLowerModuleIndex = modulesInGPU.moduleMap[innerLowerModuleIndex * MAX_CONNECTED_MODULES + outerLowerModuleArrayIdx];

        unsigned int nInnerMDs = mdsInGPU.nMDs[innerLowerModuleIndex];
        unsigned int nOuterMDs = mdsInGPU.nMDs[outerLowerModuleIndex];

        int limit = nInnerMDs*nOuterMDs;
//printf("HERE:  %d %d limit %d\n", innerLowerModuleIndex, outerLowerModuleArrayIdx, limit);

        if (limit == 0) continue;
        for(int hitIndex = threadIdx.x; hitIndex< limit; hitIndex += blockzSize)
        {
            int innerMDArrayIdx = hitIndex / nOuterMDs;
            int outerMDArrayIdx = hitIndex % nOuterMDs;
            if(outerMDArrayIdx >= nOuterMDs) continue;

            unsigned int innerMDIndex = rangesInGPU.mdRanges[innerLowerModuleIndex * 2] + innerMDArrayIdx;
            unsigned int outerMDIndex = rangesInGPU.mdRanges[outerLowerModuleIndex * 2] + outerMDArrayIdx;

            float zIn, zOut, rtIn, rtOut, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD;

            unsigned int innerMiniDoubletAnchorHitIndex = mdsInGPU.anchorHitIndices[innerMDIndex];
            unsigned int outerMiniDoubletAnchorHitIndex = mdsInGPU.anchorHitIndices[outerMDIndex];
            dPhiMin = 0;
            dPhiMax = 0;
            dPhiChangeMin = 0;
            dPhiChangeMax = 0;
            float zLo, zHi, rtLo, rtHi, sdCut , dAlphaInnerMDSegmentThreshold, dAlphaOuterMDSegmentThreshold, dAlphaInnerMDOuterMDThreshold;
            bool pass = runSegmentDefaultAlgo(modulesInGPU, mdsInGPU, innerLowerModuleIndex, outerLowerModuleIndex, innerMDIndex, outerMDIndex, zIn, zOut, rtIn, rtOut, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD, zLo, zHi, rtLo, rtHi, sdCut, dAlphaInnerMDSegmentThreshold, dAlphaOuterMDSegmentThreshold, dAlphaInnerMDOuterMDThreshold);

            if(pass)
            {
                unsigned int totOccupancySegments = atomicAdd(&segmentsInGPU.totOccupancySegments[innerLowerModuleIndex],1);
                if(totOccupancySegments >= (rangesInGPU.segmentModuleIndices[innerLowerModuleIndex + 1] - rangesInGPU.segmentModuleIndices[innerLowerModuleIndex]))
                {
#ifdef Warnings
                    printf("Segment excess alert! Module index = %d\n",innerLowerModuleIndex);
#endif
                }
                else
                {
                    unsigned int segmentModuleIdx = atomicAdd(&segmentsInGPU.nSegments[innerLowerModuleIndex],1);
                    unsigned int segmentIdx = rangesInGPU.segmentModuleIndices[innerLowerModuleIndex] + segmentModuleIdx;

                    addSegmentToMemory(segmentsInGPU,innerMDIndex, outerMDIndex,innerLowerModuleIndex, outerLowerModuleIndex, innerMiniDoubletAnchorHitIndex, outerMiniDoubletAnchorHitIndex, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, segmentIdx);

                }
            }
        }
    }
    }
}
