#ifdef __CUDACC__
#define CUDA_CONST_VAR __device__
#endif
# include "Triplet.cuh"
# include "allocate.h"
# include "Kernels.cuh"

void SDL::triplets::resetMemory(unsigned int maxTriplets, unsigned int nLowerModules,cudaStream_t stream)
{
    cudaMemsetAsync(segmentIndices,0, 5 * maxTriplets * sizeof(unsigned int),stream);
    cudaMemsetAsync(nTriplets,0, nLowerModules * sizeof(unsigned int),stream);
    cudaMemsetAsync(betaIn,0, maxTriplets * 3 * sizeof(FPX),stream);
    cudaMemsetAsync(partOfPT5,0, maxTriplets * sizeof(bool),stream);
    cudaMemsetAsync(partOfT5,0, maxTriplets * sizeof(bool));
    cudaMemsetAsync(partOfPT3, 0, maxTriplets * sizeof(bool));
    cudaMemsetAsync(totOccupancyTriplets,0, nLowerModules * sizeof(unsigned int),stream);
}

void SDL::createTripletArrayRanges(struct modules& modulesInGPU, struct objectRanges& rangesInGPU, struct segments& segmentsInGPU, uint16_t& nLowerModules, unsigned int& nTotalTriplets, cudaStream_t stream, const uint16_t& maxTripletsPerModule)
{
    int* module_tripletModuleIndices;
    cudaMallocHost(&module_tripletModuleIndices, nLowerModules * sizeof(unsigned int));
    unsigned int* nSegments;
    cudaMallocHost(&nSegments, nLowerModules * sizeof(unsigned int));
    cudaMemcpyAsync(nSegments, segmentsInGPU.nSegments, nLowerModules * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    module_tripletModuleIndices[0] = 0; 
    nTotalTriplets = maxTripletsPerModule; //start!
    for(uint16_t i = 1; i < nLowerModules; i++)
    {
        module_tripletModuleIndices[i] = nTotalTriplets;
        unsigned int occupancy = maxTripletsPerModule;
        if(nSegments[i] == 0)
        {
            occupancy = 0;
        }
        nTotalTriplets += occupancy;
    }
    cudaMemcpyAsync(rangesInGPU.tripletModuleIndices, module_tripletModuleIndices, nLowerModules * sizeof(unsigned int), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    cudaFreeHost(module_tripletModuleIndices);
    cudaFreeHost(nSegments);
}

void SDL::createInwardTripletArrayRanges(struct modules& modulesInGPU, struct objectRanges& rangesInGPU, struct segments& segmentsInGPU, uint16_t& nLowerModules, unsigned int& nTotalTriplets, cudaStream_t stream, const uint16_t& maxTripletsPerModule)
{
    int* module_tripletInwardModuleIndices;
    cudaMallocHost(&module_tripletInwardModuleIndices, nLowerModules * sizeof(unsigned int));
    unsigned int* nInwardSegments;
    cudaMallocHost(&nInwardSegments, nLowerModules * sizeof(unsigned int));
    cudaMemcpyAsync(nInwardSegments, segmentsInGPU.nInwardSegments, nLowerModules * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    module_tripletInwardModuleIndices[0] = 0; 
    nTotalTriplets = maxTripletsPerModule; //start!
    for(uint16_t i = 1; i < nLowerModules; i++)
    {
        module_tripletInwardModuleIndices[i] = nTotalTriplets;
        unsigned int occupancy = maxTripletsPerModule;
        if(nInwardSegments[i] == 0)
        {
            occupancy = 0;
        }
        nTotalTriplets += occupancy;
    }
    cudaMemcpyAsync(rangesInGPU.tripletInwardModuleIndices, module_tripletInwardModuleIndices, nLowerModules * sizeof(unsigned int), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    cudaFreeHost(module_tripletInwardModuleIndices);
    cudaFreeHost(nInwardSegments);

}


void SDL::createTripletsInUnifiedMemory(struct triplets& tripletsInGPU, unsigned int maxTriplets, uint16_t nLowerModules,cudaStream_t stream, bool inwardTriplets)
{
#ifdef CACHE_ALLOC
 //   cudaStream_t stream=0;
    tripletsInGPU.segmentIndices = (unsigned int*)cms::cuda::allocate_managed(maxTriplets * sizeof(unsigned int) *2/*5*/,stream);
    tripletsInGPU.lowerModuleIndices = (uint16_t*)cms::cuda::allocate_managed(maxTriplets * sizeof(uint16_t) *3,stream);
    tripletsInGPU.nTriplets = (unsigned int*)cms::cuda::allocate_managed(nLowerModules * sizeof(unsigned int),stream);
    tripletsInGPU.betaIn = (FPX*)cms::cuda::allocate_managed(maxTriplets * sizeof(FPX) * 3,stream);
    tripletsInGPU.partOfPT5 = (bool*)cms::cuda::allocate_managed(maxTriplets * sizeof(bool), stream);
    tripletsInGPU.partOfPT3 = (bool*)cms::cuda::allocate_managed(maxTriplets * sizeof(bool), stream);
    tripletsInGPU.partOfT5 = (bool*)cms::cuda::allocate_managed(maxTriplets * sizeof(bool), stream);
    tripletsInGPU.partOfExtension = (bool*)cms::cuda::allocate_managed(maxTriplets * sizeof(bool), stream);
    tripletsInGPU.logicalLayers = (uint8_t*)cms::cuda::allocate_managed(maxTriplets * 3 * sizeof(uint8_t), stream);
    tripletsInGPU.hitIndices = (unsigned int*)cms::cuda::allocate_managed(maxTriplets * 6 * sizeof(unsigned int), stream);
    tripletsInGPU.nMemoryLocations = (unsigned int*)cms::cuda::allocate_managed(sizeof(unsigned int), stream);
    tripletsInGPU.totOccupancyTriplets = (unsigned int*)cms::cuda::allocate_managed(nLowerModules * sizeof(unsigned int),stream);
    if(inwardTriplets)
    {
        tripletsInGPU.outwardT3Indices = (unsigned int*)cms::cuda::allocate_managed(maxTriplets * sizeof(unsigned int), stream);
    }

#else
    cudaMallocManaged(&tripletsInGPU.segmentIndices, /*5*/2 * maxTriplets * sizeof(unsigned int));
    cudaMallocManaged(&tripletsInGPU.lowerModuleIndices, 3 * maxTriplets * sizeof(uint16_t));
    cudaMallocManaged(&tripletsInGPU.nTriplets, nLowerModules * sizeof(unsigned int));
    cudaMallocManaged(&tripletsInGPU.betaIn, maxTriplets * 3 * sizeof(FPX));

    cudaMallocManaged(&tripletsInGPU.partOfPT5, maxTriplets * sizeof(bool));
    cudaMallocManaged(&tripletsInGPU.partOfPT3, maxTriplets * sizeof(bool));
    cudaMallocManaged(&tripletsInGPU.partOfT5, maxTriplets * sizeof(bool));
    cudaMallocManaged(&tripletsInGPU.partOfExtension, maxTriplets * sizeof(bool));
    cudaMallocManaged(&tripletsInGPU.logicalLayers, maxTriplets * 3 * sizeof(uint8_t));
    cudaMallocManaged(&tripletsInGPU.hitIndices, maxTriplets * 6 * sizeof(unsigned int));
    cudaMallocManaged(&tripletsInGPU.nMemoryLocations, sizeof(unsigned int));
    cudaMallocManaged(&tripletsInGPU.totOccupancyTriplets, nLowerModules * sizeof(unsigned int));

    if(inwardTriplets)
    {
        cudaMallocManaged(&tripletsInGPU.outwardT3Indices, maxTriplets * sizeof(unsigned int));
    }

#ifdef CUT_VALUE_DEBUG
    cudaMallocManaged(&tripletsInGPU.zOut, maxTriplets * 4*sizeof(unsigned int));
    cudaMallocManaged(&tripletsInGPU.zLo, maxTriplets * sizeof(float));
    cudaMallocManaged(&tripletsInGPU.zHi, maxTriplets * sizeof(float));
    cudaMallocManaged(&tripletsInGPU.zLoPointed, maxTriplets * sizeof(float));
    cudaMallocManaged(&tripletsInGPU.zHiPointed, maxTriplets * sizeof(float));
    cudaMallocManaged(&tripletsInGPU.sdlCut, maxTriplets * sizeof(float));
    cudaMallocManaged(&tripletsInGPU.betaInCut, maxTriplets * sizeof(float));
    cudaMallocManaged(&tripletsInGPU.betaOutCut, maxTriplets * sizeof(float));
    cudaMallocManaged(&tripletsInGPU.deltaBetaCut, maxTriplets * sizeof(float));
    cudaMallocManaged(&tripletsInGPU.rtLo, maxTriplets * sizeof(float));
    cudaMallocManaged(&tripletsInGPU.rtHi, maxTriplets * sizeof(float));
    cudaMallocManaged(&tripletsInGPU.kZ, maxTriplets * sizeof(float));

    tripletsInGPU.rtOut = tripletsInGPU.zOut + maxTriplets;
    tripletsInGPU.deltaPhiPos = tripletsInGPU.zOut + maxTriplets *2;
    tripletsInGPU.deltaPhi = tripletsInGPU.zOut + maxTriplets *3;
#endif
#endif
    tripletsInGPU.betaOut = tripletsInGPU.betaIn + maxTriplets ;
    tripletsInGPU.pt_beta = tripletsInGPU.betaIn + maxTriplets * 2;
    cudaMemsetAsync(tripletsInGPU.nTriplets,0,nLowerModules * sizeof(unsigned int),stream);
    cudaMemsetAsync(tripletsInGPU.partOfPT5,0,maxTriplets * sizeof(bool),stream);
    cudaMemsetAsync(tripletsInGPU.totOccupancyTriplets,0,nLowerModules * sizeof(unsigned int),stream);
    cudaMemsetAsync(tripletsInGPU.partOfPT3,0,maxTriplets * sizeof(bool),stream);
    cudaMemsetAsync(tripletsInGPU.partOfT5,0,maxTriplets * sizeof(bool),stream);
    cudaMemsetAsync(tripletsInGPU.partOfExtension,0,maxTriplets * sizeof(bool),stream);

    cudaStreamSynchronize(stream);
}
void SDL::createTripletsInExplicitMemory(struct triplets& tripletsInGPU, unsigned int maxTriplets, uint16_t nLowerModules, cudaStream_t stream, bool inwardTriplets)
{
#ifdef CACHE_ALLOC
    //cudaStream_t stream=0;
    int dev;
    cudaGetDevice(&dev);
    tripletsInGPU.segmentIndices = (unsigned int*)cms::cuda::allocate_device(dev,maxTriplets * sizeof(unsigned int) *2,stream);
    tripletsInGPU.lowerModuleIndices = (uint16_t*)cms::cuda::allocate_device(dev,maxTriplets * sizeof(uint16_t) *3,stream);
    tripletsInGPU.betaIn = (FPX*)cms::cuda::allocate_device(dev,maxTriplets * sizeof(FPX) *3,stream);
    tripletsInGPU.nTriplets = (unsigned int*)cms::cuda::allocate_device(dev,nLowerModules * sizeof(unsigned int),stream);
    tripletsInGPU.partOfPT5 = (bool*)cms::cuda::allocate_device(dev, maxTriplets * sizeof(bool), stream);
    tripletsInGPU.partOfPT3 = (bool*)cms::cuda::allocate_device(dev, maxTriplets * sizeof(bool), stream);
    tripletsInGPU.partOfT5 = (bool*)cms::cuda::allocate_device(dev, maxTriplets * sizeof(bool), stream);
    tripletsInGPU.partOfExtension = (bool*)cms::cuda::allocate_device(dev, maxTriplets * sizeof(bool), stream);

    tripletsInGPU.logicalLayers = (uint8_t*)cms::cuda::allocate_device(dev, maxTriplets * 3 * sizeof(uint8_t), stream);
    tripletsInGPU.hitIndices = (unsigned int*)cms::cuda::allocate_device(dev, maxTriplets * 6 * sizeof(unsigned int), stream);
    tripletsInGPU.nMemoryLocations = (unsigned int*)cms::cuda::allocate_device(dev, sizeof(unsigned int), stream);
    tripletsInGPU.totOccupancyTriplets = (unsigned int*)cms::cuda::allocate_device(dev,nLowerModules * sizeof(unsigned int),stream);

    if(inwardTriplets)
    {
        tripletsInGPU.outwardT3Indices = (unsigned int*)cms::cuda::allocate_device(dev, maxTriplets * sizeof(unsigned int), stream);
    }

#else
    cudaMalloc(&tripletsInGPU.segmentIndices, /*5*/2 * maxTriplets * sizeof(unsigned int));
    cudaMalloc(&tripletsInGPU.lowerModuleIndices, 3 * maxTriplets * sizeof(uint16_t));
    cudaMalloc(&tripletsInGPU.betaIn, maxTriplets * 3 * sizeof(FPX));
    cudaMalloc(&tripletsInGPU.nTriplets, nLowerModules * sizeof(unsigned int));
    cudaMalloc(&tripletsInGPU.totOccupancyTriplets, nLowerModules * sizeof(unsigned int));
    cudaMalloc(&tripletsInGPU.partOfPT5, maxTriplets * sizeof(bool));
    cudaMalloc(&tripletsInGPU.partOfPT3, maxTriplets * sizeof(bool));
    cudaMalloc(&tripletsInGPU.partOfT5, maxTriplets * sizeof(bool));
    cudaMalloc(&tripletsInGPU.partOfExtension, maxTriplets * sizeof(bool));

    cudaMalloc(&tripletsInGPU.logicalLayers, maxTriplets * 3 * sizeof(uint8_t));
    cudaMalloc(&tripletsInGPU.hitIndices, maxTriplets * 6 * sizeof(unsigned int));
    cudaMalloc(&tripletsInGPU.nMemoryLocations, sizeof(unsigned int));
    
    if(inwardTriplets)
    {
       cudaMalloc(&tripletsInGPU.outwardT3Indices, maxTriplets * sizeof(unsigned int));
    }
#endif
    cudaMemsetAsync(tripletsInGPU.nTriplets,0,nLowerModules * sizeof(unsigned int),stream);
    cudaMemsetAsync(tripletsInGPU.partOfPT5,0,maxTriplets * sizeof(bool),stream);
    cudaMemsetAsync(tripletsInGPU.partOfPT3,0,maxTriplets * sizeof(bool),stream);
    cudaMemsetAsync(tripletsInGPU.partOfT5,0,maxTriplets * sizeof(bool),stream);
    cudaMemsetAsync(tripletsInGPU.partOfExtension,0,maxTriplets * sizeof(bool),stream);
    
    cudaStreamSynchronize(stream);

    tripletsInGPU.betaOut = tripletsInGPU.betaIn + maxTriplets;
    tripletsInGPU.pt_beta = tripletsInGPU.betaIn + maxTriplets * 2;
}

#ifdef CUT_VALUE_DEBUG
__device__ void SDL::addTripletToMemory(struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct triplets& tripletsInGPU, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, uint16_t& innerInnerLowerModuleIndex, uint16_t& middleLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& deltaPhi, float& betaIn, float& betaOut, float& pt_beta, float& zLo, float& zHi, float& rtLo, float& rtHi, float& zLoPointed, float&
        zHiPointed, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ, unsigned int& tripletIndex, unsigned int outwardTripletIndex, bool inwardTriplets)
#else
__device__ void SDL::addTripletToMemory(struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct triplets& tripletsInGPU, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, uint16_t& innerInnerLowerModuleIndex, uint16_t& middleLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, float& betaIn, float& betaOut, float& pt_beta, unsigned int& tripletIndex, unsigned int outwardTripletIndex, bool inwardTriplets)
#endif
{
    tripletsInGPU.segmentIndices[tripletIndex * 2] = innerSegmentIndex;
    tripletsInGPU.segmentIndices[tripletIndex * 2 + 1] = outerSegmentIndex;
    tripletsInGPU.lowerModuleIndices[tripletIndex * 3] = innerInnerLowerModuleIndex;
    tripletsInGPU.lowerModuleIndices[tripletIndex * 3 + 1] = middleLowerModuleIndex;
    tripletsInGPU.lowerModuleIndices[tripletIndex * 3 + 2] = outerOuterLowerModuleIndex;

    tripletsInGPU.betaIn[tripletIndex]  = __F2H(betaIn);
    tripletsInGPU.betaOut[tripletIndex] = __F2H(betaOut);
    tripletsInGPU.pt_beta[tripletIndex] = __F2H(pt_beta);
   
    //track extension stuff
    tripletsInGPU.logicalLayers[tripletIndex * 3] = modulesInGPU.layers[innerInnerLowerModuleIndex] + (modulesInGPU.subdets[innerInnerLowerModuleIndex] == 4) * 6;
    tripletsInGPU.logicalLayers[tripletIndex * 3 + 1] = modulesInGPU.layers[middleLowerModuleIndex] + (modulesInGPU.subdets[middleLowerModuleIndex] == 4) * 6;
    tripletsInGPU.logicalLayers[tripletIndex * 3 + 2] = modulesInGPU.layers[outerOuterLowerModuleIndex] + (modulesInGPU.subdets[outerOuterLowerModuleIndex] == 4) * 6;
    //get the hits
    unsigned int firstMDIndex = segmentsInGPU.mdIndices[2 * innerSegmentIndex];
    unsigned int secondMDIndex = segmentsInGPU.mdIndices[2 * innerSegmentIndex + 1];
    unsigned int thirdMDIndex = segmentsInGPU.mdIndices[2 * outerSegmentIndex + 1];

    tripletsInGPU.hitIndices[tripletIndex * 6] = mdsInGPU.anchorHitIndices[firstMDIndex];
    tripletsInGPU.hitIndices[tripletIndex * 6 + 1] = mdsInGPU.outerHitIndices[firstMDIndex];
    tripletsInGPU.hitIndices[tripletIndex * 6 + 2] = mdsInGPU.anchorHitIndices[secondMDIndex];
    tripletsInGPU.hitIndices[tripletIndex * 6 + 3] = mdsInGPU.outerHitIndices[secondMDIndex];
    tripletsInGPU.hitIndices[tripletIndex * 6 + 4] = mdsInGPU.anchorHitIndices[thirdMDIndex];
    tripletsInGPU.hitIndices[tripletIndex * 6 + 5] = mdsInGPU.outerHitIndices[thirdMDIndex];

    if(inwardTriplets)
    {
        tripletsInGPU.outwardT3Indices[tripletIndex] = outwardTripletIndex;
    }
#ifdef CUT_VALUE_DEBUG
    tripletsInGPU.zOut[tripletIndex] = zOut;
    tripletsInGPU.rtOut[tripletIndex] = rtOut;
    tripletsInGPU.deltaPhiPos[tripletIndex] = deltaPhiPos;
    tripletsInGPU.deltaPhi[tripletIndex] = deltaPhi;
    tripletsInGPU.zLo[tripletIndex] = zLo;
    tripletsInGPU.zHi[tripletIndex] = zHi;
    tripletsInGPU.rtLo[tripletIndex] = rtLo;
    tripletsInGPU.rtHi[tripletIndex] = rtHi;
    tripletsInGPU.zLoPointed[tripletIndex] = zLoPointed;
    tripletsInGPU.zHiPointed[tripletIndex] = zHiPointed;
    tripletsInGPU.sdlCut[tripletIndex] = sdlCut;
    tripletsInGPU.betaInCut[tripletIndex] = betaInCut;
    tripletsInGPU.betaOutCut[tripletIndex] = betaOutCut;
    tripletsInGPU.deltaBetaCut[tripletIndex] = deltaBetaCut;
    tripletsInGPU.kZ[tripletIndex] = kZ;
#endif
}

SDL::triplets::triplets()
{
    segmentIndices = nullptr;
    lowerModuleIndices = nullptr;
    betaIn = nullptr;
    betaOut = nullptr;
    pt_beta = nullptr;
    logicalLayers = nullptr;
    hitIndices = nullptr;
    outwardT3Indices = nullptr; //important!!!!
    partOfPT5 = nullptr;
    partOfPT3 = nullptr;
    partOfT5 = nullptr;
    partOfExtension = nullptr;
#ifdef CUT_VALUE_DEBUG
    zOut = nullptr;
    rtOut = nullptr;
    deltaPhiPos = nullptr;
    deltaPhi = nullptr;
    zLo = nullptr;
    zHi = nullptr;
    rtLo = nullptr;
    rtHi = nullptr;
    zLoPointed = nullptr;
    zHiPointed = nullptr;
    kZ = nullptr;
    betaInCut = nullptr;
    betaOutCut = nullptr;
    deltaBetaCut = nullptr;
    sdlCut = nullptr;
#endif
}

SDL::triplets::~triplets()
{
}

void SDL::triplets::freeMemoryCache()
{
#ifdef Explicit_Trips
    int dev;
    cudaGetDevice(&dev);
    cms::cuda::free_device(dev,segmentIndices);
    cms::cuda::free_device(dev,lowerModuleIndices);
    cms::cuda::free_device(dev,betaIn);
    cms::cuda::free_device(dev,nTriplets);
    cms::cuda::free_device(dev,totOccupancyTriplets);
    cms::cuda::free_device(dev, partOfPT5);
    cms::cuda::free_device(dev, partOfPT3);
    cms::cuda::free_device(dev, partOfT5);
    cms::cuda::free_device(dev, partOfExtension);
    cms::cuda::free_device(dev, logicalLayers);
    cms::cuda::free_device(dev, hitIndices);
    cms::cuda::free_device(dev, nMemoryLocations);
    if(outwardT3Indices)
    {
        cms::cuda::free_device(dev, outwardT3Indices);
    }
#else
    cms::cuda::free_managed(segmentIndices);
    cms::cuda::free_managed(lowerModuleIndices);
    cms::cuda::free_managed(betaIn);
    cms::cuda::free_managed(nTriplets);
    cms::cuda::free_managed(totOccupancyTriplets);
    cms::cuda::free_managed(partOfPT5);
    cms::cuda::free_managed(partOfPT3);
    cms::cuda::free_managed(partOfT5);
    cms::cuda::free_managed(partOfExtension);
    cms::cuda::free_managed(logicalLayers);
    cms::cuda::free_managed(hitIndices);
    cms::cuda::free_managed(nMemoryLocations);
    if(outwardT3Indices)
    {
        cms::cuda::free_managed(outwardT3Indices);
    }
#endif
}
void SDL::triplets::freeMemory(cudaStream_t stream)
{
    cudaFree(segmentIndices);
    cudaFree(lowerModuleIndices);
    cudaFree(nTriplets);
    cudaFree(totOccupancyTriplets);
    cudaFree(betaIn);
    cudaFree(partOfPT5);
    cudaFree(partOfPT3);
    cudaFree(partOfT5);
    cudaFree(partOfExtension);
    cudaFree(logicalLayers);
    cudaFree(hitIndices);
    cudaFree(nMemoryLocations);
    if(outwardT3Indices)
    {
        cudaFree(outwardT3Indices);
    }
#ifdef CUT_VALUE_DEBUG
    cudaFree(zOut);
    cudaFree(zLo);
    cudaFree(zHi);
    cudaFree(rtLo);
    cudaFree(rtHi);
    cudaFree(zLoPointed);
    cudaFree(zHiPointed);
    cudaFree(kZ);
    cudaFree(betaInCut);
    cudaFree(betaOutCut);
    cudaFree(deltaBetaCut);
    cudaFree(sdlCut);
#endif
    cudaStreamSynchronize(stream);
}


__device__ bool SDL::runTripletDefaultAlgo(struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& middleLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& deltaPhi, float& betaIn, float& betaOut, float& pt_beta, float &zLo, float& zHi, float& rtLo, float& rtHi,
        float& zLoPointed, float& zHiPointed, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ)
{
    bool pass = true;
    //check
    pass = pass & (segmentsInGPU.mdIndices[2 * innerSegmentIndex+ 1] == segmentsInGPU.mdIndices[2 * outerSegmentIndex]);
   
    unsigned int firstMDIndex = segmentsInGPU.mdIndices[2 *innerSegmentIndex];
    unsigned int secondMDIndex = segmentsInGPU.mdIndices[2 * outerSegmentIndex];
    unsigned int thirdMDIndex = segmentsInGPU.mdIndices[2 * outerSegmentIndex + 1];

    pass = pass & (passRZConstraint(modulesInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, middleLowerModuleIndex, outerOuterLowerModuleIndex, firstMDIndex, secondMDIndex, thirdMDIndex));

    pass = pass & (passPointingConstraint(modulesInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, middleLowerModuleIndex, outerOuterLowerModuleIndex, firstMDIndex, secondMDIndex, thirdMDIndex, zOut, rtOut));
    //now check tracklet algo
     
    pass = pass & (runTrackletDefaultAlgo(modulesInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, middleLowerModuleIndex, middleLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, firstMDIndex, secondMDIndex, secondMDIndex, thirdMDIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, pt_beta, zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ));

    return pass;
}

__device__ bool SDL::passRZConstraint(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& middleLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex) 
{
    //get the rt and z
    const float& r1 = mdsInGPU.anchorRt[firstMDIndex];
    const float& r2 = mdsInGPU.anchorRt[secondMDIndex];
    const float& r3 = mdsInGPU.anchorRt[thirdMDIndex];

    const float& z1 = mdsInGPU.anchorZ[firstMDIndex];
    const float& z2 = mdsInGPU.anchorZ[secondMDIndex];
    const float& z3 = mdsInGPU.anchorZ[thirdMDIndex];
        
    //following Philip's layer number prescription
    const int layer1 = modulesInGPU.layers[innerInnerLowerModuleIndex] + 6 * (modulesInGPU.subdets[innerInnerLowerModuleIndex] == SDL::Endcap) + 5 * (modulesInGPU.subdets[innerInnerLowerModuleIndex] == SDL::Endcap and modulesInGPU.moduleType[innerInnerLowerModuleIndex] == SDL::TwoS);
    const int layer2 = modulesInGPU.layers[middleLowerModuleIndex] + 6 * (modulesInGPU.subdets[middleLowerModuleIndex] == SDL::Endcap) + 5 * (modulesInGPU.subdets[middleLowerModuleIndex] == SDL::Endcap and modulesInGPU.moduleType[middleLowerModuleIndex] == SDL::TwoS);
    const int layer3 = modulesInGPU.layers[outerOuterLowerModuleIndex] + 6 * (modulesInGPU.subdets[outerOuterLowerModuleIndex] == SDL::Endcap) + 5 * (modulesInGPU.subdets[outerOuterLowerModuleIndex] == SDL::Endcap and modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::TwoS);

    const float residual = z2 - ( (z3 - z1) / (r3 - r1) * (r2 - r1) + z1);

    if (layer1 == 12 and layer2 == 13 and layer3 == 14)
    {
        return false;
    }
    else if (layer1 == 1 and layer2 == 2 and layer3 == 3)
    {
        return fabsf(residual) < 0.53f;
    }
    else if (layer1 == 1 and layer2 == 2 and layer3 == 7)
    {
        return fabsf(residual) < 1;
    }
    else if (layer1 == 13 and layer2 == 14 and layer3 == 15)
    {
        return false;
    }
    else if (layer1 == 14 and layer2 == 15 and layer3 == 16)
    {
        return false;
    }
    else if (layer1 == 1 and layer2 == 7 and layer3 == 8)
    {
        return fabsf(residual) < 1;
    }
    else if (layer1 == 2 and layer2 == 3 and layer3 == 4)
    {
        return fabsf(residual) < 1.21f;
    }
    else if (layer1 == 2 and layer2 == 3 and layer3 == 7)
    {
        return fabsf(residual) < 1.f;
    }
    else if (layer1 == 2 and layer2 == 7 and layer3 == 8)
    {
        return fabsf(residual) < 1.f;
    }
    else if (layer1 == 3 and layer2 == 4 and layer3 == 5)
    {
        return fabsf(residual) < 2.7f;
    }
    else if (layer1 == 4 and layer2 == 5 and layer3 == 6)
    {
        return fabsf(residual) < 3.06f;
    }
    else if (layer1 == 7 and layer2 == 8 and layer3 == 9)
    {
        return fabsf(residual) < 1;
    }
    else if (layer1 == 8 and layer2 == 9 and layer3 == 10)
    {
        return fabsf(residual) < 1;
    }
    else if (layer1 == 9 and layer2 == 10 and layer3 == 11)
    {
        return fabsf(residual) < 1;
    }
    else
    {
        return fabsf(residual) < 5;
    }
}

__device__ bool SDL::passPointingConstraint(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& middleLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex, float& zOut, float& rtOut)
{
    short innerInnerLowerModuleSubdet = modulesInGPU.subdets[innerInnerLowerModuleIndex];
    short middleLowerModuleSubdet = modulesInGPU.subdets[middleLowerModuleIndex];
    short outerOuterLowerModuleSubdet = modulesInGPU.subdets[outerOuterLowerModuleIndex];

    bool pass = false;

    if(innerInnerLowerModuleSubdet == SDL::Barrel
            and middleLowerModuleSubdet == SDL::Barrel
            and outerOuterLowerModuleSubdet == SDL::Barrel)
    {
        pass = passPointingConstraintBBB(modulesInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, middleLowerModuleIndex, outerOuterLowerModuleIndex, firstMDIndex, secondMDIndex, thirdMDIndex, zOut, rtOut);
    }
    else if(innerInnerLowerModuleSubdet == SDL::Barrel
            and middleLowerModuleSubdet == SDL::Barrel
            and outerOuterLowerModuleSubdet == SDL::Endcap)
    {
        pass = passPointingConstraintBBE(modulesInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, middleLowerModuleIndex, outerOuterLowerModuleIndex, firstMDIndex, secondMDIndex, thirdMDIndex, zOut, rtOut);
    }
    else if(innerInnerLowerModuleSubdet == SDL::Barrel
            and middleLowerModuleSubdet == SDL::Endcap
            and outerOuterLowerModuleSubdet == SDL::Endcap)
    {
        pass = passPointingConstraintBBE(modulesInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, middleLowerModuleIndex, outerOuterLowerModuleIndex, firstMDIndex, secondMDIndex, thirdMDIndex, zOut, rtOut);

    }

    else if(innerInnerLowerModuleSubdet == SDL::Endcap
            and middleLowerModuleSubdet == SDL::Endcap
            and outerOuterLowerModuleSubdet == SDL::Endcap)
    {
        pass = passPointingConstraintEEE(modulesInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, middleLowerModuleIndex, outerOuterLowerModuleIndex, firstMDIndex, secondMDIndex, thirdMDIndex, zOut, rtOut);
    }
    
    return pass;
}

__device__ bool SDL::passPointingConstraintBBB(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& middleLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex, float& zOut, float& rtOut)
{
    bool pass = true;
    bool isPSIn = (modulesInGPU.moduleType[innerInnerLowerModuleIndex] == SDL::PS);
    bool isPSOut = (modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::PS);

    float rtIn = mdsInGPU.anchorRt[firstMDIndex];
    float rtMid = mdsInGPU.anchorRt[secondMDIndex];
    rtOut = mdsInGPU.anchorRt[thirdMDIndex];
    
    float zIn = mdsInGPU.anchorZ[firstMDIndex];
    float zMid = mdsInGPU.anchorZ[secondMDIndex];
    zOut = mdsInGPU.anchorZ[thirdMDIndex];

    float alpha1GeVOut = asinf(fminf(rtOut * k2Rinv1GeVf / ptCut, sinAlphaMax));

    float rtRatio_OutIn = rtOut / rtIn; // Outer segment beginning rt divided by inner segment beginning rt;
    float dzDrtScale = tanf(alpha1GeVOut) / alpha1GeVOut; // The track can bend in r-z plane slightly
    float zpitchIn = (isPSIn ? pixelPSZpitch : strip2SZpitch);
    float zpitchOut = (isPSOut ? pixelPSZpitch : strip2SZpitch);

    const float zHi = zIn + (zIn + deltaZLum) * (rtRatio_OutIn - 1.f) * (zIn < 0.f ? 1.f : dzDrtScale) + (zpitchIn + zpitchOut);
    const float zLo = zIn + (zIn - deltaZLum) * (rtRatio_OutIn - 1.f) * (zIn > 0.f ? 1.f : dzDrtScale) - (zpitchIn + zpitchOut); //slope-correction only on outer end

    //Cut 1 - z compatibility
    pass = pass & ((zOut >= zLo) & (zOut <= zHi));
    float drt_OutIn = (rtOut - rtIn);
    float invRtIn = 1. / rtIn;

    float r3In = sqrtf(zIn * zIn + rtIn * rtIn);
    float drt_InSeg = rtMid - rtIn;
    float dz_InSeg = zMid - zIn;
    float dr3_InSeg = sqrtf(rtMid * rtMid + zMid * zMid) - sqrtf(rtIn * rtIn + zIn + zIn);

    float coshEta = dr3_InSeg/drt_InSeg;
    float dzErr = (zpitchIn + zpitchOut) * (zpitchIn + zpitchOut) * 2.f;

    float sdlThetaMulsF = 0.015f * sqrt(0.1f + 0.2f * (rtOut - rtIn) / 50.f) * sqrt(r3In / rtIn);
    float sdlMuls = sdlThetaMulsF * 3.f / ptCut * 4.f; // will need a better guess than x4?
    dzErr += sdlMuls * sdlMuls * drt_OutIn * drt_OutIn / 3.f * coshEta * coshEta; //sloppy
    dzErr = sqrt(dzErr);

    // Constructing upper and lower bound
    const float dzMean = dz_InSeg / drt_InSeg * drt_OutIn;
    const float zWindow = dzErr / drt_InSeg * drt_OutIn + (zpitchIn + zpitchOut); //FIXME for ptCut lower than ~0.8 need to add curv path correction
    const float zLoPointed = zIn + dzMean * (zIn > 0.f ? 1.f : dzDrtScale) - zWindow;
    const float zHiPointed = zIn + dzMean * (zIn < 0.f ? 1.f : dzDrtScale) + zWindow;

    // Constructing upper and lower bound

    // Cut #2: Pointed Z (Inner segment two MD points to outer segment inner MD)
    pass = pass & ((zOut >= zLoPointed) & (zOut <= zHiPointed));

    return pass;
}

__device__ bool SDL::passPointingConstraintBBE(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& middleLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex, float& zOut, float& rtOut)
{
    bool pass = true;
    unsigned int outerInnerLowerModuleIndex = middleLowerModuleIndex;

    bool isPSIn = (modulesInGPU.moduleType[innerInnerLowerModuleIndex] == SDL::PS);
    bool isPSOut = (modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::PS);

    float rtIn = mdsInGPU.anchorRt[firstMDIndex];
    float rtMid = mdsInGPU.anchorRt[secondMDIndex];
    rtOut = mdsInGPU.anchorRt[thirdMDIndex];
    
    float zIn = mdsInGPU.anchorZ[firstMDIndex];
    float zMid = mdsInGPU.anchorZ[secondMDIndex];
    zOut = mdsInGPU.anchorZ[thirdMDIndex];

    
    float alpha1GeV_OutLo = asinf(fminf(rtOut * k2Rinv1GeVf / ptCut, sinAlphaMax));

    float rtRatio_OutIn = rtOut / rtIn; // Outer segment beginning rt divided by inner segment beginning rt;
    float dzDrtScale = tanf(alpha1GeV_OutLo) / alpha1GeV_OutLo; // The track can bend in r-z plane slightly
    float zpitchIn = (isPSIn ? pixelPSZpitch : strip2SZpitch);
    float zpitchOut = (isPSOut ? pixelPSZpitch : strip2SZpitch);
    float zGeom = zpitchIn + zpitchOut;

    float zLo = zIn + (zIn - deltaZLum) * (rtRatio_OutIn - 1.f) * (zIn > 0.f ? 1.f : dzDrtScale) - zGeom; 

    // Cut #0: Preliminary (Only here in endcap case)
    pass = pass & (zIn * zOut > 0);
    float dLum = copysignf(deltaZLum, zIn);
    bool isOutSgInnerMDPS = modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::PS;
    float rtGeom1 = isOutSgInnerMDPS ? pixelPSZpitch : strip2SZpitch;
    float zGeom1 = copysignf(zGeom,zIn);
    float rtLo = rtIn * (1.f + (zOut - zIn - zGeom1) / (zIn + zGeom1 + dLum) / dzDrtScale) - rtGeom1; //slope correction only on the lower end
    zOut = zOut;
    rtOut = rtOut;

    //Cut #1: rt condition
    float zInForHi = zIn - zGeom1 - dLum;
    if(zInForHi * zIn < 0)
    {
        zInForHi = copysignf(0.1f,zIn);
    }
    float rtHi = rtIn * (1.f + (zOut - zIn + zGeom1) / zInForHi) + rtGeom1;

    //Cut #2: rt condition
    pass = pass & ((rtOut >= rtLo) & (rtOut <= rtHi));
    float rIn = sqrtf(zIn * zIn + rtIn * rtIn);

    const float drtSDIn = rtMid - rtIn;
    const float dzSDIn = zMid - zIn;
    const float dr3SDIn = sqrtf(rtMid * rtMid + zMid * zMid) - sqrtf(rtIn * rtIn + zIn * zIn);

    const float coshEta = dr3SDIn / drtSDIn; //direction estimate
    const float dzOutInAbs = fabsf(zOut - zIn);
    const float multDzDr = dzOutInAbs * coshEta / (coshEta * coshEta - 1.f);
    const float zGeom1_another = pixelPSZpitch; //What's this?
    const float kZ = (zOut - zIn) / dzSDIn;
    float drtErr = zGeom1_another * zGeom1_another * drtSDIn * drtSDIn / dzSDIn / dzSDIn * (1.f - 2.f * kZ + 2.f * kZ * kZ); //Notes:122316
    const float sdlThetaMulsF = 0.015f * sqrtf(0.1f + 0.2 * (rtOut - rtIn) / 50.f) * sqrtf(rIn / rtIn);
    const float sdlMuls = sdlThetaMulsF * 3.f / ptCut * 4.f; //will need a better guess than x4?
    drtErr += sdlMuls * sdlMuls * multDzDr * multDzDr / 3.f * coshEta * coshEta; //sloppy: relative muls is 1/3 of total muls
    drtErr = sqrtf(drtErr);
    const float drtMean = drtSDIn * dzOutInAbs / fabsf(dzSDIn); //
    const float rtWindow = drtErr + rtGeom1;
    const float rtLo_another = rtIn + drtMean / dzDrtScale - rtWindow;
    const float rtHi_another = rtIn + drtMean + rtWindow;

    //Cut #3: rt-z pointed

    pass = pass & (kZ >=0) & (rtOut >= rtLo) & (rtOut <= rtHi);
    return pass;
}

__device__ bool SDL::passPointingConstraintEEE(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& middleLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex, float& zOut, float& rtOut)
{
    bool pass = true;
    bool isPSIn = (modulesInGPU.moduleType[innerInnerLowerModuleIndex] == SDL::PS);
    bool isPSOut = (modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::PS);

    float rtIn = mdsInGPU.anchorRt[firstMDIndex];
    float rtMid = mdsInGPU.anchorRt[secondMDIndex];
    rtOut = mdsInGPU.anchorRt[thirdMDIndex];
    
    float zIn = mdsInGPU.anchorZ[firstMDIndex];
    float zMid = mdsInGPU.anchorZ[secondMDIndex];
    zOut = mdsInGPU.anchorZ[thirdMDIndex];


    float alpha1GeV_Out = asinf(fminf(rtOut * k2Rinv1GeVf / ptCut, sinAlphaMax));

    float rtRatio_OutIn = rtOut / rtIn; // Outer segment beginning rt divided by inner segment beginning rt;
    float dzDrtScale = tanf(alpha1GeV_Out) / alpha1GeV_Out; // The track can bend in r-z plane slightly
    float zpitchIn = (isPSIn ? pixelPSZpitch : strip2SZpitch);
    float zpitchOut = (isPSOut ? pixelPSZpitch : strip2SZpitch);
    float zGeom = zpitchIn + zpitchOut;

    const float zLo = zIn + (zIn - deltaZLum) * (rtRatio_OutIn - 1.f) * (zIn > 0.f ? 1.f : dzDrtScale) - zGeom; //slope-correction only on outer end


    // Cut #0: Preliminary (Only here in endcap case)
    pass = pass & (zIn * zOut > 0);
    
    float dLum = copysignf(deltaZLum, zIn);
    bool isOutSgOuterMDPS = modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::PS;
    bool isInSgInnerMDPS = modulesInGPU.moduleType[innerInnerLowerModuleIndex] == SDL::PS;

    float rtGeom = (isInSgInnerMDPS and isOutSgOuterMDPS) ? 2.f * pixelPSZpitch : (isInSgInnerMDPS or isOutSgOuterMDPS) ? pixelPSZpitch + strip2SZpitch : 2.f * strip2SZpitch;

    float zGeom1 = copysignf(zGeom,zIn);
    float dz = zOut - zIn;
    const float rtLo = rtIn * (1.f + dz / (zIn + dLum) / dzDrtScale) - rtGeom; //slope correction only on the lower end


    //Cut #1: rt condition
    pass = pass & (rtOut >= rtLo);
    float rtHi = rtIn * (1.f + dz / (zIn - dLum)) + rtGeom;
    pass = pass & (rtOut <= rtHi);
    
    bool isInSgOuterMDPS = modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::PS;

    float drOutIn = rtOut - rtIn;
    float drtSDIn = rtMid - rtIn;
    float dzSDIn = zMid - zIn;
    float dr3SDIn = sqrtf(rtMid * rtMid + zMid * zMid) - sqrtf(rtIn * rtIn + zIn * zIn);

    float coshEta = dr3SDIn / drtSDIn; //direction estimate
    float dzOutInAbs =  fabsf(zOut - zIn);
    float multDzDr = dzOutInAbs * coshEta / (coshEta * coshEta - 1.f);

    float kZ = (zOut - zIn) / dzSDIn;
    float sdlThetaMulsF = 0.015f * sqrtf(0.1f + 0.2f * (rtOut - rtIn) / 50.f);

    float sdlMuls = sdlThetaMulsF * 3.f / ptCut * 4.f; //will need a better guess than x4?

    float drtErr = sqrtf(pixelPSZpitch * pixelPSZpitch * 2.f / (dzSDIn * dzSDIn) * (dzOutInAbs * dzOutInAbs) + sdlMuls * sdlMuls * multDzDr * multDzDr / 3.f * coshEta * coshEta);

    float drtMean = drtSDIn * dzOutInAbs/fabsf(dzSDIn);
    float rtWindow = drtErr + rtGeom;
    float rtLo_point = rtIn + drtMean / dzDrtScale - rtWindow;
    float rtHi_point = rtIn + drtMean + rtWindow;

    // Cut #3: rt-z pointed
    // https://github.com/slava77/cms-tkph2-ntuple/blob/superDoubletLinked-91X-noMock/doubletAnalysis.C#L3765

    if (isInSgInnerMDPS and isInSgOuterMDPS) // If both PS then we can point
    {
        pass = pass & ((kZ >= 0) &  (rtOut >= rtLo_point) & (rtOut <= rtHi_point));
    }

    return pass;
}
