# include "Segment.cuh"
//#ifdef CACHE_ALLOC
#include "allocate.h"
//#endif

///FIXME:NOTICE THE NEW maxPixelSegments!

void SDL::segments::resetMemory(unsigned int maxSegments, unsigned int nLowerModules, unsigned int maxPixelSegments,cudaStream_t stream)
{
    unsigned int nMemoryLocations = maxSegments * nLowerModules + maxPixelSegments;
    cudaMemsetAsync(mdIndices,0, nMemoryLocations * 2 * sizeof(unsigned int),stream);
    cudaMemsetAsync(innerLowerModuleIndices,0, nMemoryLocations * 2 * sizeof(uint16_t),stream);
    cudaMemsetAsync(nSegments, 0,(nLowerModules+1) * sizeof(unsigned int),stream);
    cudaMemsetAsync(nInwardSegments, 0, nLowerModules * sizeof(unsigned int), stream); //no pixel stuff
    cudaMemsetAsync(totOccupancySegments, 0,(nLowerModules+1) * sizeof(unsigned int),stream);
    cudaMemsetAsync(dPhis, 0,(nMemoryLocations * 6 )*sizeof(FPX),stream);
    cudaMemsetAsync(ptIn, 0,(maxPixelSegments * 8)*sizeof(float),stream);
    cudaMemsetAsync(superbin, 0,(maxPixelSegments )*sizeof(int),stream);
    cudaMemsetAsync(pixelType, 0,(maxPixelSegments )*sizeof(int8_t),stream);
    cudaMemsetAsync(isQuad, 0,(maxPixelSegments )*sizeof(bool),stream);
    cudaMemsetAsync(isDup, 0,(maxPixelSegments )*sizeof(bool),stream);
    cudaMemsetAsync(score, 0,(maxPixelSegments )*sizeof(float),stream);
    cudaMemsetAsync(circleCenterX, 0,maxPixelSegments * sizeof(float),stream);
    cudaMemsetAsync(circleCenterY, 0,maxPixelSegments * sizeof(float),stream);
    cudaMemsetAsync(circleRadius, 0,maxPixelSegments * sizeof(float),stream);
    cudaMemsetAsync(partOfPT5, 0,maxPixelSegments * sizeof(bool),stream);
}


void SDL::createSegmentArrayRanges(struct modules& modulesInGPU, struct objectRanges& rangesInGPU, struct miniDoublets& mdsInGPU, uint16_t& nLowerModules, unsigned int& nTotalSegments, cudaStream_t stream, const uint16_t& maxSegmentsPerModule, const uint16_t& maxPixelSegments)
{
    /*
        write code here that will deal with importing module parameters to CPU, and get the relevant occupancies for a given module!*/

    int *module_segmentModuleIndices;
    cudaMallocHost(&module_segmentModuleIndices, (nLowerModules + 1) * sizeof(unsigned int));
    module_segmentModuleIndices[0] = 0;
    uint16_t* module_nConnectedModules;
    cudaMallocHost(&module_nConnectedModules, nLowerModules * sizeof(uint16_t));
    cudaMemcpyAsync(module_nConnectedModules,modulesInGPU.nConnectedModules,nLowerModules*sizeof(uint16_t),cudaMemcpyDeviceToHost,stream);
    cudaStreamSynchronize(stream);

    nTotalSegments = maxSegmentsPerModule; //start!   
    for(uint16_t i = 1; i <= nLowerModules; i++)
    {
        module_segmentModuleIndices[i] = nTotalSegments; //running counter - we start at the previous index!

        unsigned int occupancy = maxSegmentsPerModule; //placeholder! this will change from module to module
        if(i == nLowerModules)
        {
            occupancy = maxPixelSegments;
        }
        else if(module_nConnectedModules[i] == 0)
        {
            occupancy = 0;
        }
        //since we allocate memory to segments even before any object is created, nMDs[i] will always be zero!!!
/*        else if(nMDs[i] == 0)
        {
            occupancy = 0;
        }*/
        nTotalSegments += occupancy;
    }
    cudaMemcpyAsync(rangesInGPU.segmentModuleIndices, module_segmentModuleIndices,  (nLowerModules + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    cudaFreeHost(module_segmentModuleIndices);
    cudaFreeHost(module_nConnectedModules);
}

void SDL::createSegmentsInUnifiedMemory(struct segments& segmentsInGPU, unsigned int nMemoryLocations, uint16_t nLowerModules, unsigned int maxPixelSegments,cudaStream_t stream)
{
    //FIXME:Since the number of pixel segments is 10x the number of regular segments per module, we need to provide
    //extra memory to the pixel segments
#ifdef CACHE_ALLOC
//    cudaStream_t stream=0; 
    segmentsInGPU.mdIndices = (unsigned int*)cms::cuda::allocate_managed(nMemoryLocations*4 *sizeof(unsigned int),stream);
    segmentsInGPU.innerLowerModuleIndices = (uint16_t*)cms::cuda::allocate_managed(nMemoryLocations*2 *sizeof(uint16_t),stream);
    segmentsInGPU.nSegments = (unsigned int*)cms::cuda::allocate_managed((nLowerModules + 1) *sizeof(unsigned int),stream);
    segmentsInGPU.nInwardSegments = (unsigned int*)cms::cuda::allocate_managed(nLowerModules * sizeof(unsigned int), stream);
    segmentsInGPU.totOccupancySegments = (unsigned int*)cms::cuda::allocate_managed((nLowerModules + 1) *sizeof(unsigned int),stream);
    segmentsInGPU.dPhis = (FPX*)cms::cuda::allocate_managed(nMemoryLocations*6  *sizeof(FPX),stream);
    segmentsInGPU.ptIn = (float*)cms::cuda::allocate_managed(maxPixelSegments * 8 *sizeof(float),stream);
    segmentsInGPU.superbin = (int*)cms::cuda::allocate_managed((maxPixelSegments) *sizeof(int),stream);
    segmentsInGPU.pixelType = (int8_t*)cms::cuda::allocate_managed((maxPixelSegments) *sizeof(int8_t),stream);
    segmentsInGPU.isQuad = (bool*)cms::cuda::allocate_managed((maxPixelSegments) *sizeof(bool),stream);
    segmentsInGPU.isDup = (bool*)cms::cuda::allocate_managed((maxPixelSegments) *sizeof(bool),stream);
    segmentsInGPU.score = (float*)cms::cuda::allocate_managed((maxPixelSegments) *sizeof(float),stream);
    segmentsInGPU.circleCenterX = (float*)cms::cuda::allocate_managed((maxPixelSegments) * sizeof(float), stream);
    segmentsInGPU.circleCenterY = (float*)cms::cuda::allocate_managed((maxPixelSegments) * sizeof(float), stream);
    segmentsInGPU.circleRadius = (float*)cms::cuda::allocate_managed((maxPixelSegments) * sizeof(float), stream);
    segmentsInGPU.partOfPT5 = (bool*)cms::cuda::allocate_managed(maxPixelSegments * sizeof(bool), stream);
    segmentsInGPU.nMemoryLocations = (unsigned int*)cms::cuda::allocate_managed(sizeof(unsigned int), stream);
#else
    cudaMallocManaged(&segmentsInGPU.mdIndices, nMemoryLocations * 4 * sizeof(unsigned int));
    cudaMallocManaged(&segmentsInGPU.innerLowerModuleIndices, nMemoryLocations * 2 * sizeof(uint16_t));
    cudaMallocManaged(&segmentsInGPU.nSegments, (nLowerModules + 1) * sizeof(unsigned int));
    cudaMallocManaged(&segmentsInGPU.nInwardSegments, nLowerModules * sizeof(unsigned int));
    cudaMallocManaged(&segmentsInGPU.totOccupancySegments, (nLowerModules + 1) * sizeof(unsigned int));
    cudaMallocManaged(&segmentsInGPU.dPhis, nMemoryLocations * 6 *sizeof(FPX));
    cudaMallocManaged(&segmentsInGPU.ptIn, maxPixelSegments * 8*sizeof(float));
    cudaMallocManaged(&segmentsInGPU.superbin, (maxPixelSegments )*sizeof(int));
    cudaMallocManaged(&segmentsInGPU.pixelType, (maxPixelSegments )*sizeof(int8_t));
    cudaMallocManaged(&segmentsInGPU.isQuad, (maxPixelSegments )*sizeof(bool));
    cudaMallocManaged(&segmentsInGPU.isDup, (maxPixelSegments )*sizeof(bool));
    cudaMallocManaged(&segmentsInGPU.score, (maxPixelSegments )*sizeof(float));
    cudaMallocManaged(&segmentsInGPU.circleCenterX, maxPixelSegments * sizeof(float));
    cudaMallocManaged(&segmentsInGPU.circleCenterY, maxPixelSegments * sizeof(float));
    cudaMallocManaged(&segmentsInGPU.circleRadius, maxPixelSegments * sizeof(float));
    cudaMallocManaged(&segmentsInGPU.partOfPT5, maxPixelSegments * sizeof(bool));
    cudaMallocManaged(&segmentsInGPU.nMemoryLocations, sizeof(unsigned int));
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
    cudaMemsetAsync(segmentsInGPU.nInwardSegments, 0, nLowerModules * sizeof(unsigned int), stream);
    cudaMemsetAsync(segmentsInGPU.totOccupancySegments,0, (nLowerModules + 1) * sizeof(unsigned int),stream);
    cudaMemsetAsync(segmentsInGPU.partOfPT5, false, maxPixelSegments * sizeof(bool),stream);
    cudaStreamSynchronize(stream);

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
    segmentsInGPU.nInwardSegments = (unsigned int*)cms::cuda::allocate_device(dev, nLowerModules * sizeof(unsigned int), stream);
    segmentsInGPU.totOccupancySegments = (unsigned int*)cms::cuda::allocate_device(dev, (nLowerModules + 1) *sizeof(unsigned int),stream);
    segmentsInGPU.dPhis = (FPX*)cms::cuda::allocate_device(dev,nMemoryLocations*6 *sizeof(FPX),stream);
    segmentsInGPU.ptIn = (float*)cms::cuda::allocate_device(dev, maxPixelSegments * 8 *sizeof(float),stream);
    segmentsInGPU.superbin = (int*)cms::cuda::allocate_device(dev,(maxPixelSegments) *sizeof(int),stream);
    segmentsInGPU.pixelType = (int8_t*)cms::cuda::allocate_device(dev,(maxPixelSegments) *sizeof(int8_t),stream);
    segmentsInGPU.isQuad = (bool*)cms::cuda::allocate_device(dev,(maxPixelSegments) *sizeof(bool),stream);
    segmentsInGPU.isDup = (bool*)cms::cuda::allocate_device(dev,(maxPixelSegments) *sizeof(bool),stream);
    segmentsInGPU.score = (float*)cms::cuda::allocate_device(dev,(maxPixelSegments) *sizeof(float),stream);
    segmentsInGPU.circleCenterX = (float*)cms::cuda::allocate_device(dev, maxPixelSegments * sizeof(float), stream);
    segmentsInGPU.circleCenterY = (float*)cms::cuda::allocate_device(dev, maxPixelSegments * sizeof(float), stream);
    segmentsInGPU.circleRadius = (float*)cms::cuda::allocate_device(dev, maxPixelSegments * sizeof(float), stream);
    segmentsInGPU.partOfPT5 = (bool*)cms::cuda::allocate_device(dev, maxPixelSegments * sizeof(bool), stream);
    segmentsInGPU.nMemoryLocations = (unsigned int*)cms::cuda::allocate_device(dev, sizeof(unsigned int), stream);
#else
    cudaMalloc(&segmentsInGPU.mdIndices, nMemoryLocations * 4 * sizeof(unsigned int));
    cudaMalloc(&segmentsInGPU.innerLowerModuleIndices, nMemoryLocations * 2 * sizeof(uint16_t));
    cudaMalloc(&segmentsInGPU.nSegments, (nLowerModules + 1) * sizeof(unsigned int));
    cudaMalloc(&segmentsInGPU.nInwardSegments, nLowerModules * sizeof(unsigned int));
    cudaMalloc(&segmentsInGPU.totOccupancySegments, (nLowerModules + 1) * sizeof(unsigned int));
    cudaMalloc(&segmentsInGPU.dPhis, nMemoryLocations * 6 *sizeof(FPX));
    cudaMalloc(&segmentsInGPU.ptIn, maxPixelSegments * 8*sizeof(float));
    cudaMalloc(&segmentsInGPU.superbin, (maxPixelSegments )*sizeof(int));
    cudaMalloc(&segmentsInGPU.pixelType, (maxPixelSegments )*sizeof(int8_t));
    cudaMalloc(&segmentsInGPU.isQuad, (maxPixelSegments )*sizeof(bool));
    cudaMalloc(&segmentsInGPU.isDup, (maxPixelSegments )*sizeof(bool));
    cudaMalloc(&segmentsInGPU.score, (maxPixelSegments )*sizeof(float));
    cudaMalloc(&segmentsInGPU.circleCenterX, maxPixelSegments * sizeof(float));
    cudaMalloc(&segmentsInGPU.circleCenterY, maxPixelSegments * sizeof(float));
    cudaMalloc(&segmentsInGPU.circleRadius, maxPixelSegments * sizeof(float));
    cudaMalloc(&segmentsInGPU.partOfPT5, maxPixelSegments * sizeof(bool));
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
    cudaMemsetAsync(segmentsInGPU.nInwardSegments, 0, nLowerModules * sizeof(unsigned int), stream);
    cudaMemsetAsync(segmentsInGPU.totOccupancySegments,0, (nLowerModules + 1) * sizeof(unsigned int),stream);
    cudaMemsetAsync(segmentsInGPU.partOfPT5, false, maxPixelSegments * sizeof(bool),stream);
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
    circleCenterX = nullptr;
    circleCenterY = nullptr;
    mdIndices = nullptr;
    innerLowerModuleIndices = nullptr;
    outerLowerModuleIndices = nullptr;
    innerMiniDoubletAnchorHitIndices = nullptr;
    outerMiniDoubletAnchorHitIndices = nullptr;

    nSegments = nullptr;
    nInwardSegments = nullptr;
    totOccupancySegments = nullptr;
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
    cms::cuda::free_device(dev,innerLowerModuleIndices);
    cms::cuda::free_device(dev,dPhis);
    cms::cuda::free_device(dev,ptIn);
    cms::cuda::free_device(dev,nSegments);
    cms::cuda::free_device(dev, nInwardSegments);
    cms::cuda::free_device(dev,nInwardSegments);
    cms::cuda::free_device(dev,totOccupancySegments);
    cms::cuda::free_device(dev,superbin);
    cms::cuda::free_device(dev,pixelType);
    cms::cuda::free_device(dev,isQuad);
    cms::cuda::free_device(dev,isDup);
    cms::cuda::free_device(dev,score);
    cms::cuda::free_device(dev, circleCenterX);
    cms::cuda::free_device(dev, circleCenterY);
    cms::cuda::free_device(dev, circleRadius);
    cms::cuda::free_device(dev, partOfPT5);
    cms::cuda::free_device(dev, nMemoryLocations);
#else
    cms::cuda::free_managed(mdIndices);
    cms::cuda::free_managed(innerLowerModuleIndices);
    cms::cuda::free_managed(dPhis);
    cms::cuda::free_managed(ptIn);
    cms::cuda::free_managed(nSegments);
    cms::cuda::free_managed(nInwardSegments);
    cms::cuda::free_managed(nInwardSegments);
    cms::cuda::free_managed(totOccupancySegments);
    cms::cuda::free_managed(superbin);
    cms::cuda::free_managed(pixelType);
    cms::cuda::free_managed(isQuad);
    cms::cuda::free_managed(isDup);
    cms::cuda::free_managed(score);
    cms::cuda::free_managed(circleCenterX);
    cms::cuda::free_managed(circleCenterY);
    cms::cuda::free_managed(circleRadius);
    cms::cuda::free_managed(partOfPT5);
    cms::cuda::free_managed(nMemoryLocations);
#endif
}
void SDL::segments::freeMemory(cudaStream_t stream)
{
    cudaFree(mdIndices);
    cudaFree(innerLowerModuleIndices);
    cudaFree(nSegments);
    cudaFree(nInwardSegments);
    cudaFree(totOccupancySegments);
    cudaFree(dPhis);
    cudaFree(ptIn);
    cudaFree(superbin);
    cudaFree(pixelType);
    cudaFree(isQuad);
    cudaFree(isDup);
    cudaFree(score);
    cudaFree(circleCenterX);
    cudaFree(circleCenterY);
    cudaFree(circleRadius);
    cudaFree(partOfPT5);
    cudaFree(nMemoryLocations);
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
__device__ void SDL::addSegmentToMemory(struct segments& segmentsInGPU, unsigned int lowerMDIndex, unsigned int upperMDIndex, uint16_t innerLowerModuleIndex, uint16_t outerLowerModuleIndex, unsigned int innerMDAnchorHitIndex, unsigned int outerMDAnchorHitIndex, float& dPhi, float& dPhiMin, float& dPhiMax, float& dPhiChange, float& dPhiChangeMin, float& dPhiChangeMax, float& zIn, float& zOut, float& rtIn, float& rtOut, float& dAlphaInnerMDSegment, float& dAlphaOuterMDSegment, float&
        dAlphaInnerMDOuterMD, float& zLo, float& zHi, float& rtLo, float& rtHi, float& sdCut, float& dAlphaInnerMDSegmentThreshold, float& dAlphaOuterMDSegmentThreshold, float& dAlphaInnerMDOuterMDThreshold, unsigned int idx)
#else
__device__ void SDL::addSegmentToMemory(struct segments& segmentsInGPU, unsigned int lowerMDIndex, unsigned int upperMDIndex, uint16_t innerLowerModuleIndex, uint16_t outerLowerModuleIndex, unsigned int innerMDAnchorHitIndex, unsigned int outerMDAnchorHitIndex, float& dPhi, float& dPhiMin, float& dPhiMax, float& dPhiChange, float& dPhiChangeMin, float& dPhiChangeMax, unsigned int idx)
#endif
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
__device__ void SDL::addPixelSegmentToMemory(struct segments& segmentsInGPU, struct miniDoublets& mdsInGPU, struct modules& modulesInGPU, unsigned int innerMDIndex, unsigned int outerMDIndex, uint16_t pixelModuleIndex, unsigned int innerAnchorHitIndex, unsigned int outerAnchorHitIndex, float dPhiChange, float ptIn, float ptErr, float px, float py, float pz, float etaErr, float eta, float phi, unsigned int idx, unsigned int pixelSegmentArrayIndex, int superbin,
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

    segmentsInGPU.superbin[pixelSegmentArrayIndex] = superbin;
    segmentsInGPU.pixelType[pixelSegmentArrayIndex] = pixelType;
    segmentsInGPU.isQuad[pixelSegmentArrayIndex] = isQuad;
    segmentsInGPU.isDup[pixelSegmentArrayIndex] = false;
    segmentsInGPU.score[pixelSegmentArrayIndex] = score;

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
    pass = pass & (zIn * zOut >= 0);
    float dz = zOut - zIn;
    float dLum = copysignf(deltaZLum, zIn);
    float drtDzScale = sdSlope/tanf(sdSlope);

    rtLo = fmaxf(rtIn * (1.f + dz / (zIn + dLum) * drtDzScale) - rtGeom,  rtIn - 0.5f * rtGeom); //rt should increase
    rtHi = rtIn * (zOut - dLum) / (zIn - dLum) + rtGeom; //dLum for luminous; rGeom for measurement size; no tanTheta_loc(pt) correction

    //completeness

    pass = pass & ((rtOut >= rtLo) & (rtOut <= rtHi));


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
    pass = pass & (fabsf(dPhi) <= sdCut);

    float dzFrac = dz/zIn;
    dPhiChange = dPhi/dzFrac * (1.f + dzFrac);
    dPhiChangeMin = dPhiMin/dzFrac * (1.f + dzFrac);
    dPhiChangeMax = dPhiMax/dzFrac * (1.f + dzFrac);
    
    pass = pass & (fabsf(dPhiChange) <= sdCut);

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
   
    pass = pass & (fabsf(dAlphaInnerMDSegment) < dAlphaThresholdValues[0]);
    pass = pass & (fabsf(dAlphaOuterMDSegment) < dAlphaThresholdValues[1]);
    pass = pass & (fabsf(dAlphaInnerMDOuterMD) < dAlphaThresholdValues[2]);

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

    pass = pass & ((zOut >= zLo) & (zOut <= zHi));

    

    sdCut = sdSlope + sqrtf(sdMuls * sdMuls + sdPVoff * sdPVoff);

    dPhi  = deltaPhi(xIn, yIn, zIn, xOut, yOut, zOut);

    pass = pass & (fabsf(dPhi) <= sdCut);

    dPhiChange = deltaPhiChange(xIn, yIn, zIn, xOut, yOut, zOut);

    pass = pass & (fabsf(dPhiChange) <= sdCut);

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
    
    pass = pass & (fabsf(dAlphaInnerMDSegment) < dAlphaThresholdValues[0]);
    pass = pass & (fabsf(dAlphaOuterMDSegment) < dAlphaThresholdValues[1]);
    pass = pass & (fabsf(dAlphaInnerMDOuterMD) < dAlphaThresholdValues[2]);

    return pass;
}

__device__ bool SDL::runSegmentDefaultAlgo(struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, uint16_t& innerLowerModuleIndex, uint16_t& outerLowerModuleIndex, unsigned int& innerMDIndex, unsigned int& outerMDIndex, float& zIn, float& zOut, float& rtIn, float& rtOut, float& dPhi, float& dPhiMin, float& dPhiMax, float& dPhiChange, float& dPhiChangeMin, float& dPhiChangeMax, float& dAlphaInnerMDSegment, float& dAlphaOuterMDSegment, float&
        dAlphaInnerMDOuterMD, float& zLo, float& zHi, float& rtLo, float& rtHi, float& sdCut, float& dAlphaInnerMDSegmentThreshold, float& dAlphaOuterMDSegmentThreshold, float& dAlphaInnerMDOuterMDThreshold)
{
    zLo = -999.f;
    zHi = -999.f;
    rtLo = -999.f;
    rtHi = -999.f;

    bool pass = true;

    if(modulesInGPU.subdets[innerLowerModuleIndex] == SDL::Barrel and modulesInGPU.subdets[outerLowerModuleIndex] == SDL::Barrel)
    {
        pass = runSegmentDefaultAlgoBarrel(modulesInGPU, mdsInGPU, innerLowerModuleIndex, outerLowerModuleIndex, innerMDIndex, outerMDIndex, zIn, zOut, rtIn, rtOut, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD, zLo, zHi, sdCut, dAlphaInnerMDSegmentThreshold, dAlphaOuterMDSegmentThreshold, dAlphaInnerMDOuterMDThreshold);
    }
    else
    {
        pass = runSegmentDefaultAlgoEndcap(modulesInGPU, mdsInGPU, innerLowerModuleIndex, outerLowerModuleIndex, innerMDIndex, outerMDIndex, zIn, zOut, rtIn, rtOut, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD, rtLo, rtHi, sdCut, dAlphaInnerMDSegmentThreshold, dAlphaOuterMDSegmentThreshold, dAlphaInnerMDOuterMDThreshold);
    }

    return pass;
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
