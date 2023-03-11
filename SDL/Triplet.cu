# include "Triplet.cuh"

void SDL::triplets::resetMemory(unsigned int maxTriplets, unsigned int nLowerModules,cudaStream_t stream)
{
    cudaMemsetAsync(segmentIndices,0, 5 * maxTriplets * sizeof(unsigned int),stream);
    cudaMemsetAsync(nTriplets,0, nLowerModules * sizeof(unsigned int),stream);
    cudaMemsetAsync(totOccupancyTriplets,0, nLowerModules * sizeof(unsigned int),stream);
    cudaMemsetAsync(betaIn,0, maxTriplets * 3 * sizeof(FPX),stream);
    cudaMemsetAsync(partOfPT5,0, maxTriplets * sizeof(bool),stream);
    cudaMemsetAsync(partOfT5,0, maxTriplets * sizeof(bool), stream);
    cudaMemsetAsync(partOfPT3, 0, maxTriplets * sizeof(bool), stream);
}

void SDL::createTripletArrayRanges(struct modules& modulesInGPU, struct objectRanges& rangesInGPU, struct segments& segmentsInGPU, uint16_t& nLowerModules, unsigned int& nTotalTriplets, cudaStream_t stream)
{
    int* module_tripletModuleIndices;
    short* module_subdets;
    short* module_layers;
    short* module_rings;
    float* module_eta;
    unsigned int* nSegments;
    module_tripletModuleIndices = (int*)cms::cuda::allocate_host(nLowerModules * sizeof(unsigned int), stream);
    module_subdets = (short*)cms::cuda::allocate_host(nLowerModules* sizeof(short), stream);
    module_layers = (short*)cms::cuda::allocate_host(nLowerModules* sizeof(short), stream);
    module_rings = (short*)cms::cuda::allocate_host(nLowerModules* sizeof(short), stream);
    module_eta = (float*)cms::cuda::allocate_host(nLowerModules* sizeof(float), stream);
    nSegments = (unsigned int*)cms::cuda::allocate_host(nLowerModules * sizeof(unsigned int), stream);
    cudaMemcpyAsync(module_subdets,modulesInGPU.subdets,nLowerModules*sizeof(short),cudaMemcpyDeviceToHost,stream);
    cudaMemcpyAsync(module_layers,modulesInGPU.layers,nLowerModules * sizeof(short),cudaMemcpyDeviceToHost,stream);
    cudaMemcpyAsync(module_rings,modulesInGPU.rings,nLowerModules * sizeof(short),cudaMemcpyDeviceToHost,stream);
    cudaMemcpyAsync(module_eta,modulesInGPU.eta,nLowerModules * sizeof(float),cudaMemcpyDeviceToHost,stream);
    cudaMemcpyAsync(nSegments, segmentsInGPU.nSegments, nLowerModules * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    nTotalTriplets = 0; //start!   
    for(uint16_t i = 0; i < nLowerModules; i++)
    {
        module_tripletModuleIndices[i] = nTotalTriplets; //running counter - we start at the previous index!
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

        if (category_number == 0 && eta_number == 0) occupancy = 543;
        if (category_number == 0 && eta_number == 1) occupancy = 235;
        if (category_number == 0 && eta_number == 2) occupancy = 88;
        if (category_number == 0 && eta_number == 3) occupancy = 46;
        if (category_number == 1 && eta_number == 0) occupancy = 755;
        if (category_number == 1 && eta_number == 1) occupancy = 347;
        if (category_number == 2 && eta_number == 1) occupancy = 0;
        if (category_number == 2 && eta_number == 2) occupancy = 0;
        if (category_number == 3 && eta_number == 1) occupancy = 38;
        if (category_number == 3 && eta_number == 2) occupancy = 46;
        if (category_number == 3 && eta_number == 3) occupancy = 39;

        if(nSegments[i] == 0) occupancy = 0;
        nTotalTriplets += occupancy;
    }
    cudaMemcpyAsync(rangesInGPU.tripletModuleIndices, module_tripletModuleIndices, nLowerModules * sizeof(unsigned int), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    cms::cuda::free_host(module_tripletModuleIndices);
    cms::cuda::free_host(nSegments);
    cms::cuda::free_host(module_subdets);
    cms::cuda::free_host(module_layers);
    cms::cuda::free_host(module_rings);
    cms::cuda::free_host(module_eta);
}

void SDL::createTripletsInExplicitMemory(struct triplets& tripletsInGPU, unsigned int maxTriplets, uint16_t nLowerModules, cudaStream_t stream)
{
#ifdef CACHE_ALLOC
    //cudaStream_t stream=0;
    int dev;
    cudaGetDevice(&dev);
    tripletsInGPU.segmentIndices = (unsigned int*)cms::cuda::allocate_device(dev,maxTriplets * sizeof(unsigned int) *2,stream);
    tripletsInGPU.lowerModuleIndices = (uint16_t*)cms::cuda::allocate_device(dev,maxTriplets * sizeof(uint16_t) *3,stream);
    tripletsInGPU.betaIn = (FPX*)cms::cuda::allocate_device(dev,maxTriplets * sizeof(FPX) *3,stream);
    tripletsInGPU.nTriplets = (int*)cms::cuda::allocate_device(dev,nLowerModules * sizeof(int),stream);
    tripletsInGPU.totOccupancyTriplets = (int*)cms::cuda::allocate_device(dev,nLowerModules * sizeof(int),stream);
    tripletsInGPU.partOfPT5 = (bool*)cms::cuda::allocate_device(dev, maxTriplets * sizeof(bool), stream);
    tripletsInGPU.partOfPT3 = (bool*)cms::cuda::allocate_device(dev, maxTriplets * sizeof(bool), stream);
    tripletsInGPU.partOfT5 = (bool*)cms::cuda::allocate_device(dev, maxTriplets * sizeof(bool), stream);

    tripletsInGPU.logicalLayers = (uint8_t*)cms::cuda::allocate_device(dev, maxTriplets * 3 * sizeof(uint8_t), stream);
    tripletsInGPU.hitIndices = (unsigned int*)cms::cuda::allocate_device(dev, maxTriplets * 6 * sizeof(unsigned int), stream);
    tripletsInGPU.nMemoryLocations = (unsigned int*)cms::cuda::allocate_device(dev, sizeof(unsigned int), stream);

#ifdef CUT_VALUE_DEBUG
    tripletsInGPU.zOut = (float*)cms::cuda::allocate_device(dev, maxTriplets * 4 * sizeof(float), stream);
    tripletsInGPU.zLo = (float*)cms::cuda::allocate_device(dev, maxTriplets * sizeof(float), stream);
    tripletsInGPU.zHi = (float*)cms::cuda::allocate_device(dev, maxTriplets * sizeof(float), stream);
    tripletsInGPU.zLoPointed = (float*)cms::cuda::allocate_device(dev, maxTriplets * sizeof(float), stream);
    tripletsInGPU.zHiPointed = (float*)cms::cuda::allocate_device(dev, maxTriplets * sizeof(float), stream);
    tripletsInGPU.sdlCut = (float*)cms::cuda::allocate_device(dev, maxTriplets * sizeof(float), stream);
    tripletsInGPU.betaInCut = (float*)cms::cuda::allocate_device(dev, maxTriplets * sizeof(float), stream);
    tripletsInGPU.betaOutCut = (float*)cms::cuda::allocate_device(dev, maxTriplets * sizeof(float), stream);
    tripletsInGPU.deltaBetaCut = (float*)cms::cuda::allocate_device(dev, maxTriplets * sizeof(float), stream);
    tripletsInGPU.rtLo = (float*)cms::cuda::allocate_device(dev, maxTriplets * sizeof(float), stream);
    tripletsInGPU.rtHi = (float*)cms::cuda::allocate_device(dev, maxTriplets * sizeof(float), stream);
    tripletsInGPU.kZ = (float*)cms::cuda::allocate_device(dev, maxTriplets * sizeof(float), stream);
    tripletsInGPU.rtOut = tripletsInGPU.zOut + maxTriplets;
    tripletsInGPU.deltaPhiPos = tripletsInGPU.zOut + maxTriplets *2;
    tripletsInGPU.deltaPhi = tripletsInGPU.zOut + maxTriplets *3;
#endif


#else
    cudaMalloc(&tripletsInGPU.segmentIndices, /*5*/2 * maxTriplets * sizeof(unsigned int));
    cudaMalloc(&tripletsInGPU.lowerModuleIndices, 3 * maxTriplets * sizeof(uint16_t));
    cudaMalloc(&tripletsInGPU.betaIn, maxTriplets * 3 * sizeof(FPX));
    cudaMalloc(&tripletsInGPU.nTriplets, nLowerModules * sizeof(int));
    cudaMalloc(&tripletsInGPU.totOccupancyTriplets, nLowerModules * sizeof(int));
    cudaMalloc(&tripletsInGPU.partOfPT5, maxTriplets * sizeof(bool));
    cudaMalloc(&tripletsInGPU.partOfPT3, maxTriplets * sizeof(bool));
    cudaMalloc(&tripletsInGPU.partOfT5, maxTriplets * sizeof(bool));

    cudaMalloc(&tripletsInGPU.logicalLayers, maxTriplets * 3 * sizeof(uint8_t));
    cudaMalloc(&tripletsInGPU.hitIndices, maxTriplets * 6 * sizeof(unsigned int));
    cudaMalloc(&tripletsInGPU.nMemoryLocations, sizeof(unsigned int));

#ifdef CUT_VALUE_DEBUG
    cudaMalloc(&tripletsInGPU.zOut, maxTriplets * 4*sizeof(unsigned int));
    cudaMalloc(&tripletsInGPU.zLo, maxTriplets * sizeof(float));
    cudaMalloc(&tripletsInGPU.zHi, maxTriplets * sizeof(float));
    cudaMalloc(&tripletsInGPU.zLoPointed, maxTriplets * sizeof(float));
    cudaMalloc(&tripletsInGPU.zHiPointed, maxTriplets * sizeof(float));
    cudaMalloc(&tripletsInGPU.sdlCut, maxTriplets * sizeof(float));
    cudaMalloc(&tripletsInGPU.betaInCut, maxTriplets * sizeof(float));
    cudaMalloc(&tripletsInGPU.betaOutCut, maxTriplets * sizeof(float));
    cudaMalloc(&tripletsInGPU.deltaBetaCut, maxTriplets * sizeof(float));
    cudaMalloc(&tripletsInGPU.rtLo, maxTriplets * sizeof(float));
    cudaMalloc(&tripletsInGPU.rtHi, maxTriplets * sizeof(float));
    cudaMalloc(&tripletsInGPU.kZ, maxTriplets * sizeof(float));

    tripletsInGPU.rtOut = tripletsInGPU.zOut + maxTriplets;
    tripletsInGPU.deltaPhiPos = tripletsInGPU.zOut + maxTriplets *2;
    tripletsInGPU.deltaPhi = tripletsInGPU.zOut + maxTriplets *3;
#endif

#endif
    cudaMemsetAsync(tripletsInGPU.nTriplets,0,nLowerModules * sizeof(int),stream);
    cudaMemsetAsync(tripletsInGPU.totOccupancyTriplets,0,nLowerModules * sizeof(int),stream);
    cudaMemsetAsync(tripletsInGPU.partOfPT5,0,maxTriplets * sizeof(bool),stream);
    cudaMemsetAsync(tripletsInGPU.partOfPT3,0,maxTriplets * sizeof(bool),stream);
    cudaMemsetAsync(tripletsInGPU.partOfT5,0,maxTriplets * sizeof(bool),stream);
    
    cudaStreamSynchronize(stream);

    tripletsInGPU.betaOut = tripletsInGPU.betaIn + maxTriplets;
    tripletsInGPU.pt_beta = tripletsInGPU.betaIn + maxTriplets * 2;
}

#ifdef CUT_VALUE_DEBUG
ALPAKA_FN_ACC void SDL::addTripletToMemory(struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct triplets& tripletsInGPU, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, uint16_t& innerInnerLowerModuleIndex, uint16_t& middleLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& deltaPhi, float& betaIn, float& betaOut, float& pt_beta, float& zLo, float& zHi, float& rtLo, float& rtHi, float& zLoPointed, float&zHiPointed, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ, unsigned int& tripletIndex)
#else
ALPAKA_FN_ACC void SDL::addTripletToMemory(struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct triplets& tripletsInGPU, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, uint16_t& innerInnerLowerModuleIndex, uint16_t& middleLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, float& betaIn, float& betaOut, float& pt_beta, unsigned int& tripletIndex)
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
    cms::cuda::free_device(dev, logicalLayers);
    cms::cuda::free_device(dev, hitIndices);
    cms::cuda::free_device(dev, nMemoryLocations);
#ifdef CUT_VALUE_DEBUG
    cms::cuda::free_device(dev, zOut);
    cms::cuda::free_device(dev, zLo);
    cms::cuda::free_device(dev, zHi);
    cms::cuda::free_device(dev, zLoPointed);
    cms::cuda::free_device(dev, zHiPointed);
    cms::cuda::free_device(dev, sdlCut);
    cms::cuda::free_device(dev, betaInCut);
    cms::cuda::free_device(dev, betaOutCut);
    cms::cuda::free_device(dev, deltaBetaCut);
    cms::cuda::free_device(dev, rtLo);
    cms::cuda::free_device(dev, rtHi);
    cms::cuda::free_device(dev, kZ);
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
    cudaFree(logicalLayers);
    cudaFree(hitIndices);
    cudaFree(nMemoryLocations);
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
