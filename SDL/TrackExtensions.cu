# include "TrackExtensions.cuh"
# include "Kernels.cuh"

SDL::trackExtensions::trackExtensions()
{
    constituentTCTypes = nullptr;
    constituentTCIndices = nullptr;
    nHitOverlaps = nullptr;
    nLayerOverlaps = nullptr;
    nTrackExtensions = nullptr;
    totOccupancyTrackExtensions = nullptr;
    rPhiChiSquared = nullptr;
    rzChiSquared = nullptr;
    regressionRadius = nullptr;
#ifdef CUT_VALUE_DEBUG
    innerRadius = nullptr;
    outerRadius = nullptr;
    isDup = nullptr;
#endif
}

SDL::trackExtensions::~trackExtensions()
{
}

void SDL::trackExtensions::freeMemory(cudaStream_t stream)
{
    cudaFree(constituentTCTypes);
    cudaFree(constituentTCIndices);
    cudaFree(nLayerOverlaps);
    cudaFree(nHitOverlaps);
    cudaFree(nTrackExtensions);
    cudaFree(totOccupancyTrackExtensions);
    cudaFree(isDup);
    cudaFree(rPhiChiSquared);
    cudaFree(rzChiSquared);
    cudaFree(regressionRadius);
#ifdef CUT_VALUE_DEBUG
    cudaFree(innerRadius);
    cudaFree(outerRadius);
#endif
    cudaStreamSynchronize(stream);
}

void SDL::trackExtensions::freeMemoryCache()
{
#ifdef Explicit_Extensions
    int dev;
    cudaGetDevice(&dev);
    cms::cuda::free_device(dev, constituentTCTypes);
    cms::cuda::free_device(dev, constituentTCIndices);
    cms::cuda::free_device(dev, nLayerOverlaps);
    cms::cuda::free_device(dev, nHitOverlaps);
    cms::cuda::free_device(dev, rPhiChiSquared);
    cms::cuda::free_device(dev, rzChiSquared);
    cms::cuda::free_device(dev, isDup);
    cms::cuda::free_device(dev, nTrackExtensions);
    cms::cuda::free_device(dev, totOccupancyTrackExtensions);
    cms::cuda::free_device(dev, regressionRadius);
#ifdef CUT_VALUE_DEBUG
    cms::cuda::free_device(dev, innerRadius);
    cms::cuda::free_device(dev, outerRadius);
#endif
#else
    cms::cuda::free_managed(constituentTCTypes);
    cms::cuda::free_managed(constituentTCIndices);
    cms::cuda::free_managed(nLayerOverlaps);
    cms::cuda::free_managed(nHitOverlaps);
    cms::cuda::free_managed(rPhiChiSquared);
    cms::cuda::free_managed(rzChiSquared);
    cms::cuda::free_managed(isDup);
    cms::cuda::free_managed(nTrackExtensions);
    cms::cuda::free_managed(totOccupancyTrackExtensions);
    cms::cuda::free_managed(regressionRadius);
#endif
}

void SDL::trackExtensions::resetMemory(unsigned int maxTrackExtensions, unsigned int nTrackCandidates, cudaStream_t stream)
{
    cudaMemsetAsync(constituentTCTypes, 0, sizeof(short) * 3 * maxTrackExtensions, stream);
    cudaMemsetAsync(constituentTCIndices, 0, sizeof(unsigned int) * 3 * maxTrackExtensions, stream);
    cudaMemsetAsync(nLayerOverlaps, 0, sizeof(uint8_t) * 2 * maxTrackExtensions, stream);
    cudaMemsetAsync(nHitOverlaps, 0, sizeof(uint8_t) * 2 * maxTrackExtensions, stream);
    cudaMemsetAsync(rPhiChiSquared, 0, sizeof(FPX) * maxTrackExtensions, stream);
    cudaMemsetAsync(rzChiSquared, 0, sizeof(FPX) * maxTrackExtensions, stream);
    cudaMemsetAsync(isDup, 0, sizeof(bool) * maxTrackExtensions, stream);
    cudaMemsetAsync(regressionRadius, 0, sizeof(FPX) * maxTrackExtensions, stream);
    cudaMemsetAsync(nTrackExtensions, 0, sizeof(unsigned int) * nTrackCandidates, stream);
    cudaMemsetAsync(totOccupancyTrackExtensions, 0, sizeof(unsigned int) * nTrackCandidates, stream);
}


/*
   Track Extensions memory allocation - 10 slots for each TC (will reduce later)
   Extensions having the same anchor object will be clustered together for easy
   duplicate cleaning
*/
void SDL::createTrackExtensionsInUnifiedMemory(struct trackExtensions& trackExtensionsInGPU, unsigned int maxTrackExtensions, unsigned int nTrackCandidates, cudaStream_t stream)
{
#ifdef CACHE_ALLOC
    trackExtensionsInGPU.constituentTCTypes = (short*)cms::cuda::allocate_managed(maxTrackExtensions * 3 * sizeof(short), stream);
    trackExtensionsInGPU.constituentTCIndices = (unsigned int*)cms::cuda::allocate_managed(maxTrackExtensions * 3 * sizeof(unsigned int), stream);
    trackExtensionsInGPU.nLayerOverlaps = (uint8_t*)cms::cuda::allocate_managed(maxTrackExtensions * 2 * sizeof(uint8_t), stream);
    trackExtensionsInGPU.nHitOverlaps = (uint8_t*)cms::cuda::allocate_managed(maxTrackExtensions * 2 * sizeof(uint8_t), stream);
    trackExtensionsInGPU.rPhiChiSquared = (FPX*)cms::cuda::allocate_managed(maxTrackExtensions * sizeof(FPX), stream);
    trackExtensionsInGPU.rzChiSquared = (FPX*)cms::cuda::allocate_managed(maxTrackExtensions * sizeof(FPX), stream);
    trackExtensionsInGPU.isDup = (bool*) cms::cuda::allocate_managed(maxTrackExtensions * sizeof(bool), stream);
    trackExtensionsInGPU.regressionRadius = (FPX*)cms::cuda::allocate_managed(maxTrackExtensions * sizeof(FPX), stream);
    trackExtensionsInGPU.nTrackExtensions = (unsigned int*)cms::cuda::allocate_managed(nTrackCandidates * sizeof(unsigned int), stream);
    trackExtensionsInGPU.totOccupancyTrackExtensions = (unsigned int*)cms::cuda::allocate_managed(nTrackCandidates * sizeof(unsigned int), stream);

#else
    cudaMallocManaged(&trackExtensionsInGPU.constituentTCTypes, sizeof(short) * 3 * maxTrackExtensions);
    cudaMallocManaged(&trackExtensionsInGPU.constituentTCIndices, sizeof(unsigned int) * 3 * maxTrackExtensions);
    cudaMallocManaged(&trackExtensionsInGPU.nLayerOverlaps, sizeof(uint8_t) * 2 * maxTrackExtensions);
    cudaMallocManaged(&trackExtensionsInGPU.nHitOverlaps, sizeof(uint8_t) * 2 * maxTrackExtensions);
    cudaMallocManaged(&trackExtensionsInGPU.rPhiChiSquared, maxTrackExtensions * sizeof(FPX));
    cudaMallocManaged(&trackExtensionsInGPU.rzChiSquared, maxTrackExtensions * sizeof(FPX));
    cudaMallocManaged(&trackExtensionsInGPU.isDup, maxTrackExtensions * sizeof(bool));
    cudaMallocManaged(&trackExtensionsInGPU.regressionRadius, maxTrackExtensions * sizeof(FPX));
    cudaMallocManaged(&trackExtensionsInGPU.nTrackExtensions, nTrackCandidates * sizeof(unsigned int));
    cudaMallocManaged(&trackExtensionsInGPU.totOccupancyTrackExtensions, nTrackCandidates * sizeof(unsigned int));

#ifdef CUT_VALUE_DEBUG
    cudaMallocManaged(&trackExtensionsInGPU.innerRadius, maxTrackExtensions * sizeof(float));
    cudaMallocManaged(&trackExtensionsInGPU.outerRadius, maxTrackExtensions * sizeof(float));
#endif
#endif

    cudaMemsetAsync(trackExtensionsInGPU.nTrackExtensions, 0, nTrackCandidates * sizeof(unsigned int), stream);
    cudaMemsetAsync(trackExtensionsInGPU.totOccupancyTrackExtensions, 0, nTrackCandidates * sizeof(unsigned int), stream);
    cudaMemsetAsync(trackExtensionsInGPU.isDup, true, maxTrackExtensions * sizeof(bool), stream);
}

void SDL::createTrackExtensionsInExplicitMemory(struct trackExtensions& trackExtensionsInGPU, unsigned int maxTrackExtensions, unsigned int nTrackCandidates, cudaStream_t stream)
{
#ifdef CACHE_ALLOC
    int dev;
    cudaGetDevice(&dev);
    trackExtensionsInGPU.constituentTCTypes = (short*)cms::cuda::allocate_device(dev,maxTrackExtensions * 3 * sizeof(short), stream);
    trackExtensionsInGPU.constituentTCIndices = (unsigned int*)cms::cuda::allocate_device(dev,maxTrackExtensions * 3 * sizeof(unsigned int), stream);
    trackExtensionsInGPU.nLayerOverlaps = (uint8_t*)cms::cuda::allocate_device(dev,maxTrackExtensions * 2 * sizeof(uint8_t), stream);
    trackExtensionsInGPU.nHitOverlaps = (uint8_t*)cms::cuda::allocate_device(dev,maxTrackExtensions * 2 * sizeof(uint8_t), stream);

    trackExtensionsInGPU.rPhiChiSquared = (FPX*)cms::cuda::allocate_device(dev,maxTrackExtensions * sizeof(FPX), stream);
    trackExtensionsInGPU.rzChiSquared   = (FPX*)cms::cuda::allocate_device(dev,maxTrackExtensions * sizeof(FPX), stream);
    trackExtensionsInGPU.isDup = (bool*) cms::cuda::allocate_device(dev,maxTrackExtensions * sizeof(bool), stream);
    trackExtensionsInGPU.nTrackExtensions = (unsigned int*)cms::cuda::allocate_device(dev,nTrackCandidates * sizeof(unsigned int), stream);
    trackExtensionsInGPU.totOccupancyTrackExtensions = (unsigned int*)cms::cuda::allocate_device(dev,nTrackCandidates * sizeof(unsigned int), stream);
    trackExtensionsInGPU.regressionRadius = (FPX*)cms::cuda::allocate_device(dev, maxTrackExtensions * sizeof(FPX), stream);
#ifdef CUT_VALUE_DEBUG
    trackExtensionsInGPU.innerRadius = (float*)cms::cuda::allocate_device(dev, maxTrackExtensions * sizeof(float), stream);
    trackExtensionsInGPU.outerRadius = (float*)cms::cuda::allocate_device(dev, maxTrackExtensions * sizeof(float), stream);
#endif
#else
    cudaMalloc(&trackExtensionsInGPU.constituentTCTypes, sizeof(short) * 3 * maxTrackExtensions);
    cudaMalloc(&trackExtensionsInGPU.constituentTCIndices, sizeof(unsigned int) * 3 * maxTrackExtensions);
    cudaMalloc(&trackExtensionsInGPU.nLayerOverlaps, sizeof(uint8_t) * 2 * maxTrackExtensions);
    cudaMalloc(&trackExtensionsInGPU.nHitOverlaps, sizeof(uint8_t) * 2 * maxTrackExtensions);
    cudaMalloc(&trackExtensionsInGPU.nTrackExtensions, nTrackCandidates * sizeof(unsigned int));
    cudaMalloc(&trackExtensionsInGPU.totOccupancyTrackExtensions, nTrackCandidates * sizeof(unsigned int));
    cudaMalloc(&trackExtensionsInGPU.rPhiChiSquared, maxTrackExtensions * sizeof(FPX));
    cudaMalloc(&trackExtensionsInGPU.rzChiSquared,   maxTrackExtensions * sizeof(FPX));
    cudaMalloc(&trackExtensionsInGPU.regressionRadius, maxTrackExtensions * sizeof(FPX));
    cudaMalloc(&trackExtensionsInGPU.isDup, maxTrackExtensions * sizeof(bool));
#endif

    cudaMemsetAsync(trackExtensionsInGPU.nTrackExtensions, 0, nTrackCandidates * sizeof(unsigned int), stream);
    cudaMemsetAsync(trackExtensionsInGPU.totOccupancyTrackExtensions, 0, nTrackCandidates * sizeof(unsigned int), stream);
    cudaMemsetAsync(trackExtensionsInGPU.isDup, true, maxTrackExtensions * sizeof(bool), stream);
    cudaStreamSynchronize(stream);
}

#ifdef CUT_VALUE_DEBUG
__device__ void SDL::addTrackExtensionToMemory(struct trackExtensions& trackExtensionsInGPU, short* constituentTCType, unsigned int* constituentTCIndex, unsigned int* nLayerOverlaps, unsigned int* nHitOverlaps, float rPhiChiSquared, float rzChiSquared, float regressionRadius, float innerRadius, float outerRadius, unsigned int trackExtensionIndex)
#else
__device__ void SDL::addTrackExtensionToMemory(struct trackExtensions& trackExtensionsInGPU, short* constituentTCType, unsigned int* constituentTCIndex, unsigned int* nLayerOverlaps, unsigned int* nHitOverlaps, float rPhiChiSquared, float rzChiSquared, float regressionRadius, unsigned int trackExtensionIndex)
#endif
{ 
    for(size_t i = 0; i < 3 ; i++)
    {
        trackExtensionsInGPU.constituentTCTypes[3 * trackExtensionIndex + i] = constituentTCType[i];
        trackExtensionsInGPU.constituentTCIndices[3 * trackExtensionIndex + i] = constituentTCIndex[i];
    }
    for(size_t i = 0; i < 2; i++)
    {
        trackExtensionsInGPU.nLayerOverlaps[2 * trackExtensionIndex + i] = nLayerOverlaps[i];
        trackExtensionsInGPU.nHitOverlaps[2 * trackExtensionIndex + i] = nHitOverlaps[i];
    }
    trackExtensionsInGPU.rPhiChiSquared[trackExtensionIndex]   = __F2H(rPhiChiSquared);
    trackExtensionsInGPU.rzChiSquared[trackExtensionIndex]     = __F2H(rzChiSquared);
    trackExtensionsInGPU.regressionRadius[trackExtensionIndex] = __F2H(regressionRadius);

#ifdef CUT_VALUE_DEBUG
    trackExtensionsInGPU.innerRadius[trackExtensionIndex] = innerRadius;
    trackExtensionsInGPU.outerRadius[trackExtensionIndex] = outerRadius;
#endif
}

//SPECIAL DISPENSATION - hitsinGPU will be used here!

__device__ bool SDL::runTrackExtensionDefaultAlgo(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct triplets& tripletsInGPU, struct quintuplets& quintupletsInGPU, struct pixelTriplets& pixelTripletsInGPU, struct pixelQuintuplets& pixelQuintupletsInGPU, struct trackCandidates& trackCandidatesInGPU, unsigned int anchorObjectIndex, unsigned int outerObjectIndex, short anchorObjectType, short outerObjectType, unsigned int anchorObjectOuterT3Index, unsigned int layerOverlapTarget, short* constituentTCType, unsigned int* constituentTCIndex, unsigned
        int* nLayerOverlaps, unsigned int* nHitOverlaps, float& rPhiChiSquared, float& rzChiSquared, float& regressionRadius, float& innerRadius, float& outerRadius)
{
    /*
       Basic premise:
       1. given two objects, get the hit and module indices
       2. check for layer and hit overlap (layer overlap first checked using
       the 2-merge approach)
       3. Additional cuts - rz and rphi chi squared criteria! 
    */

    bool pass = true;
    uint8_t* anchorLayerIndices = nullptr;
    unsigned int* anchorHitIndices = nullptr;
    uint16_t* anchorLowerModuleIndices = nullptr;

    uint8_t* outerObjectLayerIndices = nullptr;
    unsigned int* outerObjectHitIndices = nullptr;
    uint16_t* outerObjectLowerModuleIndices = nullptr;

    unsigned int nAnchorLayers = (anchorObjectType == 7) ? 7 : (anchorObjectType == 3 ? 3 : 5);
    float centerX, centerY;
    if(anchorObjectType != 3) //mostly this
    { 
        anchorLayerIndices = &trackCandidatesInGPU.logicalLayers[7 * anchorObjectIndex];
        anchorHitIndices = &trackCandidatesInGPU.hitIndices[14 * anchorObjectIndex];
        anchorLowerModuleIndices = &trackCandidatesInGPU.lowerModuleIndices[7 * anchorObjectIndex];
        centerX = __H2F(trackCandidatesInGPU.centerX[anchorObjectIndex]);
        centerY = __H2F(trackCandidatesInGPU.centerY[anchorObjectIndex]);
        innerRadius = __H2F(trackCandidatesInGPU.radius[anchorObjectIndex]);
        outerRadius = -999;
        regressionRadius = -999;
    }
    else //outlier
    {
        anchorLayerIndices = &tripletsInGPU.logicalLayers[3 * anchorObjectIndex];
        anchorHitIndices = &tripletsInGPU.hitIndices[6 * anchorObjectIndex];
        anchorLowerModuleIndices = &tripletsInGPU.lowerModuleIndices[3 * anchorObjectIndex];
    }

    unsigned int layer_binary = 0;

    unsigned int nOuterLayers =(outerObjectType == 7) ? 7 : (outerObjectType == 3 ? 3 : 5); 

    if(outerObjectType == 3) //mostly this
    {
        outerObjectLayerIndices = &tripletsInGPU.logicalLayers[3 * outerObjectIndex];
        outerObjectHitIndices = &tripletsInGPU.hitIndices[6 * outerObjectIndex];
        outerObjectLowerModuleIndices = &tripletsInGPU.lowerModuleIndices[3 * outerObjectIndex];
    }
    else //outlier
    {
        outerObjectLayerIndices = &trackCandidatesInGPU.logicalLayers[7 * outerObjectIndex];
        outerObjectHitIndices = &trackCandidatesInGPU.hitIndices[14 * outerObjectIndex];
        outerObjectLowerModuleIndices = &tripletsInGPU.lowerModuleIndices[7 * outerObjectIndex];
    }
 
    unsigned int nLayerOverlap(0), nHitOverlap(0);
   
    float zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, pt_beta; //temp stuff
    float zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ;

    unsigned int innerSegmentIndex = tripletsInGPU.segmentIndices[2 * anchorObjectOuterT3Index];
    unsigned int outerSegmentIndex = tripletsInGPU.segmentIndices[2 * outerObjectIndex];

    pass =  pass and runExtensionDefaultAlgo(modulesInGPU, mdsInGPU, segmentsInGPU, tripletsInGPU.lowerModuleIndices[3 * anchorObjectOuterT3Index], tripletsInGPU.lowerModuleIndices[3 * anchorObjectOuterT3Index + 1], tripletsInGPU.lowerModuleIndices[3 * outerObjectIndex], tripletsInGPU.lowerModuleIndices[3 * outerObjectIndex + 1], innerSegmentIndex, outerSegmentIndex, 
            segmentsInGPU.mdIndices[2 * innerSegmentIndex], segmentsInGPU.mdIndices[2 * innerSegmentIndex + 1], segmentsInGPU.mdIndices[2 * outerSegmentIndex], segmentsInGPU.mdIndices[2 * outerSegmentIndex + 1], zOut, rtOut, deltaPhiPos, deltaPhi, betaIn,
            betaOut, pt_beta, zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ);

    if(not pass) return pass;

    innerSegmentIndex = tripletsInGPU.segmentIndices[2 * anchorObjectOuterT3Index];
    outerSegmentIndex = tripletsInGPU.segmentIndices[2 * outerObjectIndex + 1];

    pass =  pass and runExtensionDefaultAlgo(modulesInGPU, mdsInGPU, segmentsInGPU, tripletsInGPU.lowerModuleIndices[3 * anchorObjectOuterT3Index], tripletsInGPU.lowerModuleIndices[3 * anchorObjectOuterT3Index + 1], tripletsInGPU.lowerModuleIndices[3 * outerObjectIndex + 1], tripletsInGPU.lowerModuleIndices[3 * outerObjectIndex + 2], innerSegmentIndex, outerSegmentIndex, segmentsInGPU.mdIndices[2 * innerSegmentIndex], segmentsInGPU.mdIndices[2 *
            innerSegmentIndex + 1], segmentsInGPU.mdIndices[2 * outerSegmentIndex], segmentsInGPU.mdIndices[2 * outerSegmentIndex + 1], zOut, rtOut, deltaPhiPos, deltaPhi,
            betaIn, betaOut, pt_beta, zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ);

    if(not pass) return pass;

    //checks for frivolous cases wherein
    pass = pass and computeLayerAndHitOverlaps(modulesInGPU, anchorLayerIndices, anchorHitIndices, anchorLowerModuleIndices, outerObjectLayerIndices, outerObjectHitIndices, outerObjectLowerModuleIndices, nAnchorLayers, nOuterLayers, nLayerOverlap, nHitOverlap, layerOverlapTarget);

    if(not pass) return pass;


    unsigned int anchorObjectAnchorHitIndices[7];
    unsigned int outerObjectAnchorHitIndices[7];
    for(size_t i=0; i<nAnchorLayers;i++)
    {
        anchorObjectAnchorHitIndices[i] = anchorHitIndices[2 * i];
        layer_binary |= (1 << anchorLayerIndices[i]);
    }

    for(size_t i=0; i<nOuterLayers;i++)
    {
        outerObjectAnchorHitIndices[i] = outerObjectHitIndices[2 * i];
        layer_binary |= (1 << outerObjectLayerIndices[i]);
    }

    if(anchorObjectType != 3)
    {
        rPhiChiSquared = computeTERPhiChiSquared(modulesInGPU, hitsInGPU, centerX, centerY, innerRadius, outerObjectAnchorHitIndices, outerObjectLowerModuleIndices);
        pass = pass and passTERPhiChiSquaredCuts(nLayerOverlap, nHitOverlap, layer_binary, rPhiChiSquared);
        if(not pass) return pass;

        rzChiSquared = computeTERZChiSquared(modulesInGPU, hitsInGPU, anchorObjectAnchorHitIndices, anchorLowerModuleIndices, outerObjectAnchorHitIndices, outerObjectLowerModuleIndices, anchorObjectType);
        pass = pass and passTERZChiSquaredCuts(nLayerOverlap, nHitOverlap, layer_binary, rzChiSquared);
        if(not pass) return pass;
    }
    else
    {
        //create a unified list of hit indices and lower module indices
        unsigned int overallAnchorIndices[6];
        uint16_t overallLowerModuleIndices[6];
        int i = 0, j = 0, nPoints = 0;
        while(j < 3)
        {
            if(i < 3)
            {
                overallAnchorIndices[nPoints] = anchorObjectAnchorHitIndices[i];
                overallLowerModuleIndices[nPoints] = anchorLowerModuleIndices[i];
                if(anchorObjectAnchorHitIndices[i] == outerObjectAnchorHitIndices[j])
                {
                    j++;
                }
                i++;
            }
            else
            {
                overallAnchorIndices[nPoints] = outerObjectAnchorHitIndices[j];
                overallLowerModuleIndices[nPoints] = outerObjectLowerModuleIndices[j];
                j++;
            }
            nPoints++;
        }

        float x1 = hitsInGPU.xs[anchorObjectAnchorHitIndices[0]];
        float x2 = hitsInGPU.xs[anchorObjectAnchorHitIndices[1]];
        float x3 = hitsInGPU.xs[anchorObjectAnchorHitIndices[2]];
        float y1 = hitsInGPU.ys[anchorObjectAnchorHitIndices[0]];
        float y2 = hitsInGPU.ys[anchorObjectAnchorHitIndices[1]];
        float y3 = hitsInGPU.ys[anchorObjectAnchorHitIndices[2]];
        float g,f;
        innerRadius = computeRadiusFromThreeAnchorHitsTCE(x1, y1, x2, y2, x3, y3, g, f);

        x1 = hitsInGPU.xs[outerObjectAnchorHitIndices[0]];
        x2 = hitsInGPU.xs[outerObjectAnchorHitIndices[1]];
        x3 = hitsInGPU.xs[outerObjectAnchorHitIndices[2]];
        y1 = hitsInGPU.ys[outerObjectAnchorHitIndices[0]];
        y2 = hitsInGPU.ys[outerObjectAnchorHitIndices[1]];
        y3 = hitsInGPU.ys[outerObjectAnchorHitIndices[2]];

        outerRadius = computeRadiusFromThreeAnchorHitsTCE(x1, y1, x2, y2, x3, y3, g, f);


        rPhiChiSquared = computeT3T3RPhiChiSquared(modulesInGPU, hitsInGPU, nPoints, overallAnchorIndices, overallLowerModuleIndices, regressionRadius);
        rzChiSquared = computeT3T3RZChiSquared(modulesInGPU, hitsInGPU, nPoints, overallAnchorIndices, overallLowerModuleIndices);
        
        if(innerRadius < 2.0/(2 * k2Rinv1GeVf))
        {
            pass = pass and passRadiusMatch(nLayerOverlap, nHitOverlap, layer_binary, innerRadius, outerRadius);   
            if(not pass) return pass;
        }
        else
        {
            pass = pass and passHighPtRadiusMatch(nLayerOverlap, nHitOverlap, layer_binary, innerRadius, outerRadius);
            if(not pass) return pass;
        }
        if(innerRadius < 5.0/(2 * k2Rinv1GeVf))
        {
            pass = pass and passTERPhiChiSquaredCuts(nLayerOverlap, nHitOverlap, layer_binary, rPhiChiSquared);
            if(not pass) return pass;
            pass = pass and passTERZChiSquaredCuts(nLayerOverlap, nHitOverlap, layer_binary, rzChiSquared);
            if(not pass) return pass;
        }
    }
   

    nLayerOverlaps[0] = nLayerOverlap;
    nHitOverlaps[0] = nHitOverlap;

    constituentTCType[0] = anchorObjectType;
    constituentTCType[1] = outerObjectType;

    constituentTCIndex[0] = anchorObjectIndex;
    constituentTCIndex[1] = outerObjectIndex;

    return pass;
}

__device__ bool SDL::passHighPtRadiusMatch(unsigned int& nLayerOverlaps, unsigned int& nHitOverlaps, unsigned int& layer_binary, float& innerRadius, float& outerRadius)
{
    float innerInvRadiusPositiveErrorBound, outerInvRadiusPositiveErrorBound;
    float innerInvRadiusNegativeErrorBound, outerInvRadiusNegativeErrorBound;
    float innerRadiusInvMin, innerRadiusInvMax, outerRadiusInvMin, outerRadiusInvMax;

    if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 30)
    {
        innerInvRadiusPositiveErrorBound = 0.291;
        innerInvRadiusNegativeErrorBound = 6.291;
        outerInvRadiusPositiveErrorBound = 0.239;
        outerInvRadiusNegativeErrorBound = 1.542;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 30)
    {
        innerInvRadiusPositiveErrorBound = 0.291;
        innerInvRadiusNegativeErrorBound = 1.149;
        outerInvRadiusPositiveErrorBound = 0.239;
        outerInvRadiusNegativeErrorBound = 0.087;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 30)
    {
        innerInvRadiusPositiveErrorBound = 0.681;
        innerInvRadiusNegativeErrorBound = 7.909;
        outerInvRadiusPositiveErrorBound = 0.491;
        outerInvRadiusNegativeErrorBound = 7.909;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 30)
    {
        innerInvRadiusPositiveErrorBound = 0.751;
        innerInvRadiusNegativeErrorBound = 6.291;
        outerInvRadiusPositiveErrorBound = 0.776;
        outerInvRadiusNegativeErrorBound = 7.909;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 30)
    {
        innerInvRadiusPositiveErrorBound = 0.681;
        innerInvRadiusNegativeErrorBound = 7.909;
        outerInvRadiusPositiveErrorBound = 0.751;
        outerInvRadiusNegativeErrorBound = 5.170;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 60)
    {
        innerInvRadiusPositiveErrorBound = 0.975;
        innerInvRadiusNegativeErrorBound = 0.445;
        outerInvRadiusPositiveErrorBound = 0.659;
        outerInvRadiusNegativeErrorBound = 0.944;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 60)
    {
        innerInvRadiusPositiveErrorBound = 0.776;
        innerInvRadiusNegativeErrorBound = 1.701;
        outerInvRadiusPositiveErrorBound = 0.659;
        outerInvRadiusNegativeErrorBound = 39.270;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 60)
    {
        innerInvRadiusPositiveErrorBound = 0.776;
        innerInvRadiusNegativeErrorBound = 7.909;
        outerInvRadiusPositiveErrorBound = 0.659;
        outerInvRadiusNegativeErrorBound = 4.843;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 62)
    {
        innerInvRadiusPositiveErrorBound = 0.291;
        innerInvRadiusNegativeErrorBound = 6.291;
        outerInvRadiusPositiveErrorBound = 0.203;
        outerInvRadiusNegativeErrorBound = 0.113;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 120)
    {
        innerInvRadiusPositiveErrorBound = 0.659;
        innerInvRadiusNegativeErrorBound = 2.138;
        outerInvRadiusPositiveErrorBound = 0.751;
        outerInvRadiusNegativeErrorBound = 0.079;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 120)
    {
        innerInvRadiusPositiveErrorBound = 0.727;
        innerInvRadiusNegativeErrorBound = 0.802;
        outerInvRadiusPositiveErrorBound = 0.703;
        outerInvRadiusNegativeErrorBound = 0.109;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 120)
    {
        innerInvRadiusPositiveErrorBound = 0.914;
        innerInvRadiusNegativeErrorBound = 39.270;
        outerInvRadiusPositiveErrorBound = 0.578;
        outerInvRadiusNegativeErrorBound = 0.491;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 124)
    {
        innerInvRadiusPositiveErrorBound = 0.217;
        innerInvRadiusNegativeErrorBound = 0.217;
        outerInvRadiusPositiveErrorBound = 0.156;
        outerInvRadiusNegativeErrorBound = 0.156;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 142)
    {
        innerInvRadiusPositiveErrorBound = 0.005;
        innerInvRadiusNegativeErrorBound = 0.005;
        outerInvRadiusPositiveErrorBound = 0.366;
        outerInvRadiusNegativeErrorBound = 0.366;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 142)
    {
        innerInvRadiusPositiveErrorBound = 0.037;
        innerInvRadiusNegativeErrorBound = 0.037;
        outerInvRadiusPositiveErrorBound = 0.281;
        outerInvRadiusNegativeErrorBound = 0.281;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 142)
    {
        innerInvRadiusPositiveErrorBound = 0.217;
        innerInvRadiusNegativeErrorBound = 1.008;
        outerInvRadiusPositiveErrorBound = 0.578;
        outerInvRadiusNegativeErrorBound = 0.659;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 142)
    {
        innerInvRadiusPositiveErrorBound = 0.024;
        innerInvRadiusNegativeErrorBound = 0.037;
        outerInvRadiusPositiveErrorBound = 0.281;
        outerInvRadiusNegativeErrorBound = 0.281;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 142)
    {
        innerInvRadiusPositiveErrorBound = 0.378;
        innerInvRadiusNegativeErrorBound = 6.716;
        outerInvRadiusPositiveErrorBound = 0.681;
        outerInvRadiusNegativeErrorBound = 108.230;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 156)
    {
        innerInvRadiusPositiveErrorBound = 0.217;
        innerInvRadiusNegativeErrorBound = 0.001;
        outerInvRadiusPositiveErrorBound = 0.751;
        outerInvRadiusNegativeErrorBound = 0.255;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 156)
    {
        innerInvRadiusPositiveErrorBound = 0.524;
        innerInvRadiusNegativeErrorBound = 0.301;
        outerInvRadiusPositiveErrorBound = 0.239;
        outerInvRadiusNegativeErrorBound = 1.876;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 156)
    {
        innerInvRadiusPositiveErrorBound = 0.524;
        innerInvRadiusNegativeErrorBound = 0.751;
        outerInvRadiusPositiveErrorBound = 0.802;
        outerInvRadiusNegativeErrorBound = 9.314;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 158)
    {
        innerInvRadiusPositiveErrorBound = 0.087;
        innerInvRadiusNegativeErrorBound = 0.087;
        outerInvRadiusPositiveErrorBound = 0.751;
        outerInvRadiusNegativeErrorBound = 0.026;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 184)
    {
        innerInvRadiusPositiveErrorBound = 0.102;
        innerInvRadiusNegativeErrorBound = 0.102;
        outerInvRadiusPositiveErrorBound = 0.491;
        outerInvRadiusNegativeErrorBound = 6.500;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 184)
    {
        innerInvRadiusPositiveErrorBound = 0.035;
        innerInvRadiusNegativeErrorBound = 0.035;
        outerInvRadiusPositiveErrorBound = 0.431;
        outerInvRadiusNegativeErrorBound = 0.431;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 184)
    {
        innerInvRadiusPositiveErrorBound = 0.156;
        innerInvRadiusNegativeErrorBound = 0.255;
        outerInvRadiusPositiveErrorBound = 0.597;
        outerInvRadiusNegativeErrorBound = 39.270;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 188)
    {
        innerInvRadiusPositiveErrorBound = 0.109;
        innerInvRadiusNegativeErrorBound = 0.109;
        outerInvRadiusPositiveErrorBound = 0.491;
        outerInvRadiusNegativeErrorBound = 6.500;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 390)
    {
        innerInvRadiusPositiveErrorBound = 0.224;
        innerInvRadiusNegativeErrorBound = 2.437;
        outerInvRadiusPositiveErrorBound = 0.475;
        outerInvRadiusNegativeErrorBound = 5.893;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 390)
    {
        innerInvRadiusPositiveErrorBound = 0.403;
        innerInvRadiusNegativeErrorBound = 10.274;
        outerInvRadiusPositiveErrorBound = 0.751;
        outerInvRadiusNegativeErrorBound = 9.314;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 390)
    {
        innerInvRadiusPositiveErrorBound = 0.597;
        innerInvRadiusNegativeErrorBound = 24.844;
        outerInvRadiusPositiveErrorBound = 0.751;
        outerInvRadiusNegativeErrorBound = 108.230;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 396)
    {
        innerInvRadiusPositiveErrorBound = 0.507;
        innerInvRadiusNegativeErrorBound = 0.028;
        outerInvRadiusPositiveErrorBound = 0.597;
        outerInvRadiusNegativeErrorBound = 3.271;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 396)
    {
        innerInvRadiusPositiveErrorBound = 0.210;
        innerInvRadiusNegativeErrorBound = 0.146;
        outerInvRadiusPositiveErrorBound = 0.659;
        outerInvRadiusNegativeErrorBound = 3.166;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 396)
    {
        innerInvRadiusPositiveErrorBound = 0.659;
        innerInvRadiusNegativeErrorBound = 1.398;
        outerInvRadiusPositiveErrorBound = 0.856;
        outerInvRadiusNegativeErrorBound = 13.790;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 398)
    {
        innerInvRadiusPositiveErrorBound = 0.178;
        innerInvRadiusNegativeErrorBound = 0.021;
        outerInvRadiusPositiveErrorBound = 0.597;
        outerInvRadiusNegativeErrorBound = 3.271;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 408)
    {
        innerInvRadiusPositiveErrorBound = 0.321;
        innerInvRadiusNegativeErrorBound = 0.321;
        outerInvRadiusPositiveErrorBound = 0.727;
        outerInvRadiusNegativeErrorBound = 0.727;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 408)
    {
        innerInvRadiusPositiveErrorBound = 0.597;
        innerInvRadiusNegativeErrorBound = 0.559;
        outerInvRadiusPositiveErrorBound = 0.703;
        outerInvRadiusNegativeErrorBound = 0.975;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 408)
    {
        innerInvRadiusPositiveErrorBound = 0.239;
        innerInvRadiusNegativeErrorBound = 1.876;
        outerInvRadiusPositiveErrorBound = 0.431;
        outerInvRadiusNegativeErrorBound = 0.301;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 408)
    {
        innerInvRadiusPositiveErrorBound = 0.597;
        innerInvRadiusNegativeErrorBound = 9.314;
        outerInvRadiusPositiveErrorBound = 0.751;
        outerInvRadiusNegativeErrorBound = 6.089;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 412)
    {
        innerInvRadiusPositiveErrorBound = 0.090;
        innerInvRadiusNegativeErrorBound = 0.090;
        outerInvRadiusPositiveErrorBound = 0.146;
        outerInvRadiusNegativeErrorBound = 0.727;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 898)
    {
        innerInvRadiusPositiveErrorBound = 0.703;
        innerInvRadiusNegativeErrorBound = 0.703;
        outerInvRadiusPositiveErrorBound = 0.021;
        outerInvRadiusNegativeErrorBound = 2.602;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 898)
    {
        innerInvRadiusPositiveErrorBound = 0.828;
        innerInvRadiusNegativeErrorBound = 31.235;
        outerInvRadiusPositiveErrorBound = 0.659;
        outerInvRadiusNegativeErrorBound = 19.125;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 898)
    {
        innerInvRadiusPositiveErrorBound = 0.776;
        innerInvRadiusNegativeErrorBound = 0.776;
        outerInvRadiusPositiveErrorBound = 0.659;
        outerInvRadiusNegativeErrorBound = 0.659;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 898)
    {
        innerInvRadiusPositiveErrorBound = 0.828;
        innerInvRadiusNegativeErrorBound = 24.844;
        outerInvRadiusPositiveErrorBound = 0.776;
        outerInvRadiusNegativeErrorBound = 108.230;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 900)
    {
        innerInvRadiusPositiveErrorBound = 0.578;
        innerInvRadiusNegativeErrorBound = 0.491;
        outerInvRadiusPositiveErrorBound = 0.884;
        outerInvRadiusNegativeErrorBound = 1.701;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 900)
    {
        innerInvRadiusPositiveErrorBound = 0.578;
        innerInvRadiusNegativeErrorBound = 6.939;
        outerInvRadiusPositiveErrorBound = 0.884;
        outerInvRadiusNegativeErrorBound = 2.777;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 900)
    {
        innerInvRadiusPositiveErrorBound = 0.597;
        innerInvRadiusNegativeErrorBound = 0.255;
        outerInvRadiusPositiveErrorBound = 0.231;
        outerInvRadiusNegativeErrorBound = 2.358;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 900)
    {
        innerInvRadiusPositiveErrorBound = 0.751;
        innerInvRadiusNegativeErrorBound = 108.230;
        outerInvRadiusPositiveErrorBound = 0.884;
        outerInvRadiusNegativeErrorBound = 51.014;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 902)
    {
        innerInvRadiusPositiveErrorBound = 0.224;
        innerInvRadiusNegativeErrorBound = 3.728;
        outerInvRadiusPositiveErrorBound = 0.884;
        outerInvRadiusNegativeErrorBound = 2.777;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 904)
    {
        innerInvRadiusPositiveErrorBound = 0.828;
        innerInvRadiusNegativeErrorBound = 2.138;
        outerInvRadiusPositiveErrorBound = 0.884;
        outerInvRadiusNegativeErrorBound = 51.014;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 1920)
    {
        innerInvRadiusPositiveErrorBound = 1.701;
        innerInvRadiusNegativeErrorBound = 1.701;
        outerInvRadiusPositiveErrorBound = 0.802;
        outerInvRadiusNegativeErrorBound = 47.784;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 1920)
    {
        innerInvRadiusPositiveErrorBound = 0.491;
        innerInvRadiusNegativeErrorBound = 0.491;
        outerInvRadiusPositiveErrorBound = 2.688;
        outerInvRadiusNegativeErrorBound = 2.688;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 1920)
    {
        innerInvRadiusPositiveErrorBound = 0.776;
        innerInvRadiusNegativeErrorBound = 7.655;
        outerInvRadiusPositiveErrorBound = 0.802;
        outerInvRadiusNegativeErrorBound = 47.784;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 1920)
    {
        innerInvRadiusPositiveErrorBound = 0.659;
        innerInvRadiusNegativeErrorBound = 4.249;
        outerInvRadiusPositiveErrorBound = 0.776;
        outerInvRadiusNegativeErrorBound = 5.170;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 1920)
    {
        innerInvRadiusPositiveErrorBound = 0.776;
        innerInvRadiusNegativeErrorBound = 98.116;
        outerInvRadiusPositiveErrorBound = 0.802;
        outerInvRadiusNegativeErrorBound = 66.269;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 1922)
    {
        innerInvRadiusPositiveErrorBound = 0.776;
        innerInvRadiusNegativeErrorBound = 31.235;
        outerInvRadiusPositiveErrorBound = 0.802;
        outerInvRadiusNegativeErrorBound = 24.844;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 1924)
    {
        innerInvRadiusPositiveErrorBound = 0.578;
        innerInvRadiusNegativeErrorBound = 6.939;
        outerInvRadiusPositiveErrorBound = 0.802;
        outerInvRadiusNegativeErrorBound = 24.844;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 3840)
    {
        innerInvRadiusPositiveErrorBound = 1.267;
        innerInvRadiusNegativeErrorBound = 1.267;
        outerInvRadiusPositiveErrorBound = 0.040;
        outerInvRadiusNegativeErrorBound = 0.040;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 3840)
    {
        innerInvRadiusPositiveErrorBound = 0.460;
        innerInvRadiusNegativeErrorBound = 4.687;
        outerInvRadiusPositiveErrorBound = 0.638;
        outerInvRadiusNegativeErrorBound = 7.909;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 3840)
    {
        innerInvRadiusPositiveErrorBound = 0.703;
        innerInvRadiusNegativeErrorBound = 0.366;
        outerInvRadiusPositiveErrorBound = 0.390;
        outerInvRadiusNegativeErrorBound = 25.670;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 3840)
    {
        innerInvRadiusPositiveErrorBound = 0.617;
        innerInvRadiusNegativeErrorBound = 47.784;
        outerInvRadiusPositiveErrorBound = 0.638;
        outerInvRadiusNegativeErrorBound = 108.230;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 3968)
    {
        innerInvRadiusPositiveErrorBound = 0.491;
        innerInvRadiusNegativeErrorBound = 4.536;
        outerInvRadiusPositiveErrorBound = 0.390;
        outerInvRadiusNegativeErrorBound = 7.909;
    }
    innerRadiusInvMin = fmaxf(0.0, (1 - innerInvRadiusPositiveErrorBound) / innerRadius);
    innerRadiusInvMax = (1 + innerInvRadiusNegativeErrorBound) / innerRadius;

    outerRadiusInvMin = fmaxf(0.0, (1 - outerInvRadiusPositiveErrorBound) / outerRadius);
    outerRadiusInvMax = (1 + outerInvRadiusNegativeErrorBound) / outerRadius;

    return checkIntervalOverlap(innerRadiusInvMin, innerRadiusInvMax, outerRadiusInvMin, outerRadiusInvMax);
}

__device__ bool SDL::passRadiusMatch(unsigned int& nLayerOverlaps, unsigned int& nHitOverlaps, unsigned int& layer_binary, float& innerRadius, float& outerRadius)
{
    float innerInvRadiusPositiveErrorBound, outerInvRadiusPositiveErrorBound;
    float innerInvRadiusNegativeErrorBound, outerInvRadiusNegativeErrorBound;
    float innerRadiusInvMin, innerRadiusInvMax, outerRadiusInvMin, outerRadiusInvMax;

    if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 30)
    {
        innerInvRadiusPositiveErrorBound = 0.093;
        innerInvRadiusNegativeErrorBound = 1.876;
        outerInvRadiusPositiveErrorBound = 0.272;
        outerInvRadiusNegativeErrorBound = 0.030;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 30)
    {
        innerInvRadiusPositiveErrorBound = 0.217;
        innerInvRadiusNegativeErrorBound = 1.876;
        outerInvRadiusPositiveErrorBound = 0.063;
        outerInvRadiusNegativeErrorBound = 0.203;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 30)
    {
        innerInvRadiusPositiveErrorBound = 0.184;
        innerInvRadiusNegativeErrorBound = 2.602;
        outerInvRadiusPositiveErrorBound = 0.331;
        outerInvRadiusNegativeErrorBound = 0.431;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 30)
    {
        innerInvRadiusPositiveErrorBound = 0.354;
        innerInvRadiusNegativeErrorBound = 0.597;
        outerInvRadiusPositiveErrorBound = 0.217;
        outerInvRadiusNegativeErrorBound = 0.541;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 30)
    {
        innerInvRadiusPositiveErrorBound = 0.291;
        innerInvRadiusNegativeErrorBound = 0.802;
        outerInvRadiusPositiveErrorBound = 0.354;
        outerInvRadiusNegativeErrorBound = 0.431;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 60)
    {
        innerInvRadiusPositiveErrorBound = 0.005;
        innerInvRadiusNegativeErrorBound = 0.005;
        outerInvRadiusPositiveErrorBound = 0.013;
        outerInvRadiusNegativeErrorBound = 0.013;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 60)
    {
        innerInvRadiusPositiveErrorBound = 0.161;
        innerInvRadiusNegativeErrorBound = 11.333;
        outerInvRadiusPositiveErrorBound = 0.146;
        outerInvRadiusNegativeErrorBound = 0.445;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 60)
    {
        innerInvRadiusPositiveErrorBound = 0.029;
        innerInvRadiusNegativeErrorBound = 0.802;
        outerInvRadiusPositiveErrorBound = 0.146;
        outerInvRadiusNegativeErrorBound = 0.076;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 60)
    {
        innerInvRadiusPositiveErrorBound = 0.184;
        innerInvRadiusNegativeErrorBound = 4.390;
        outerInvRadiusPositiveErrorBound = 0.366;
        outerInvRadiusNegativeErrorBound = 3.271;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 62)
    {
        innerInvRadiusPositiveErrorBound = 0.090;
        innerInvRadiusNegativeErrorBound = 0.047;
        outerInvRadiusPositiveErrorBound = 0.124;
        outerInvRadiusNegativeErrorBound = 0.231;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 120)
    {
        innerInvRadiusPositiveErrorBound = 0.460;
        innerInvRadiusNegativeErrorBound = 5.520;
        outerInvRadiusPositiveErrorBound = 0.541;
        outerInvRadiusNegativeErrorBound = 0.106;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 120)
    {
        innerInvRadiusPositiveErrorBound = 0.102;
        innerInvRadiusNegativeErrorBound = 0.681;
        outerInvRadiusPositiveErrorBound = 0.184;
        outerInvRadiusNegativeErrorBound = 0.106;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 120)
    {
        innerInvRadiusPositiveErrorBound = 0.354;
        innerInvRadiusNegativeErrorBound = 0.776;
        outerInvRadiusPositiveErrorBound = 0.390;
        outerInvRadiusNegativeErrorBound = 0.281;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 124)
    {
        innerInvRadiusPositiveErrorBound = 0.217;
        innerInvRadiusNegativeErrorBound = 0.038;
        outerInvRadiusPositiveErrorBound = 0.247;
        outerInvRadiusNegativeErrorBound = 0.009;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 142)
    {
        innerInvRadiusPositiveErrorBound = 0.264;
        innerInvRadiusNegativeErrorBound = 0.106;
        outerInvRadiusPositiveErrorBound = 0.272;
        outerInvRadiusNegativeErrorBound = 0.311;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 142)
    {
        innerInvRadiusPositiveErrorBound = 0.061;
        innerInvRadiusNegativeErrorBound = 0.013;
        outerInvRadiusPositiveErrorBound = 0.151;
        outerInvRadiusNegativeErrorBound = 0.069;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 142)
    {
        innerInvRadiusPositiveErrorBound = 0.264;
        innerInvRadiusNegativeErrorBound = 0.106;
        outerInvRadiusPositiveErrorBound = 0.331;
        outerInvRadiusNegativeErrorBound = 0.703;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 142)
    {
        innerInvRadiusPositiveErrorBound = 0.061;
        innerInvRadiusNegativeErrorBound = 0.055;
        outerInvRadiusPositiveErrorBound = 0.151;
        outerInvRadiusNegativeErrorBound = 0.301;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 142)
    {
        innerInvRadiusPositiveErrorBound = 0.343;
        innerInvRadiusNegativeErrorBound = 0.247;
        outerInvRadiusPositiveErrorBound = 0.460;
        outerInvRadiusNegativeErrorBound = 0.975;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 156)
    {
        innerInvRadiusPositiveErrorBound = 0.944;
        innerInvRadiusNegativeErrorBound = 0.013;
        outerInvRadiusPositiveErrorBound = 0.975;
        outerInvRadiusNegativeErrorBound = 0.460;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 156)
    {
        innerInvRadiusPositiveErrorBound = 0.002;
        innerInvRadiusNegativeErrorBound = 0.042;
        outerInvRadiusPositiveErrorBound = 0.203;
        outerInvRadiusNegativeErrorBound = 0.022;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 156)
    {
        innerInvRadiusPositiveErrorBound = 0.944;
        innerInvRadiusNegativeErrorBound = 6.716;
        outerInvRadiusPositiveErrorBound = 0.975;
        outerInvRadiusNegativeErrorBound = 0.828;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 184)
    {
        innerInvRadiusPositiveErrorBound = 0.044;
        innerInvRadiusNegativeErrorBound = 0.015;
        outerInvRadiusPositiveErrorBound = 0.093;
        outerInvRadiusNegativeErrorBound = 0.093;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 184)
    {
        innerInvRadiusPositiveErrorBound = 0.015;
        innerInvRadiusNegativeErrorBound = 0.015;
        outerInvRadiusPositiveErrorBound = 0.067;
        outerInvRadiusNegativeErrorBound = 0.067;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 184)
    {
        innerInvRadiusPositiveErrorBound = 0.051;
        innerInvRadiusNegativeErrorBound = 0.019;
        outerInvRadiusPositiveErrorBound = 0.057;
        outerInvRadiusNegativeErrorBound = 0.659;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 184)
    {
        innerInvRadiusPositiveErrorBound = 0.019;
        innerInvRadiusNegativeErrorBound = 0.019;
        outerInvRadiusPositiveErrorBound = 0.321;
        outerInvRadiusNegativeErrorBound = 0.321;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 184)
    {
        innerInvRadiusPositiveErrorBound = 0.231;
        innerInvRadiusNegativeErrorBound = 1.267;
        outerInvRadiusPositiveErrorBound = 0.321;
        outerInvRadiusNegativeErrorBound = 0.884;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 188)
    {
        innerInvRadiusPositiveErrorBound = 0.002;
        innerInvRadiusNegativeErrorBound = 0.016;
        outerInvRadiusPositiveErrorBound = 0.063;
        outerInvRadiusNegativeErrorBound = 0.272;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 390)
    {
        innerInvRadiusPositiveErrorBound = 0.051;
        innerInvRadiusNegativeErrorBound = 0.036;
        outerInvRadiusPositiveErrorBound = 0.017;
        outerInvRadiusNegativeErrorBound = 0.802;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 390)
    {
        innerInvRadiusPositiveErrorBound = 0.040;
        innerInvRadiusNegativeErrorBound = 0.042;
        outerInvRadiusPositiveErrorBound = 0.047;
        outerInvRadiusNegativeErrorBound = 0.617;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 390)
    {
        innerInvRadiusPositiveErrorBound = 0.524;
        innerInvRadiusNegativeErrorBound = 2.003;
        outerInvRadiusPositiveErrorBound = 0.638;
        outerInvRadiusNegativeErrorBound = 3.728;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 390)
    {
        innerInvRadiusPositiveErrorBound = 0.727;
        innerInvRadiusNegativeErrorBound = 0.507;
        outerInvRadiusPositiveErrorBound = 0.751;
        outerInvRadiusNegativeErrorBound = 1.041;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 390)
    {
        innerInvRadiusPositiveErrorBound = 0.417;
        innerInvRadiusNegativeErrorBound = 2.283;
        outerInvRadiusPositiveErrorBound = 0.638;
        outerInvRadiusNegativeErrorBound = 1.757;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 396)
    {
        innerInvRadiusPositiveErrorBound = 0.041;
        innerInvRadiusNegativeErrorBound = 0.041;
        outerInvRadiusPositiveErrorBound = 29.258;
        outerInvRadiusNegativeErrorBound = 29.258;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 396)
    {
        innerInvRadiusPositiveErrorBound = 0.460;
        innerInvRadiusNegativeErrorBound = 0.460;
        outerInvRadiusPositiveErrorBound = 0.475;
        outerInvRadiusNegativeErrorBound = 0.203;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 396)
    {
        innerInvRadiusPositiveErrorBound = 0.291;
        innerInvRadiusNegativeErrorBound = 0.378;
        outerInvRadiusPositiveErrorBound = 0.491;
        outerInvRadiusNegativeErrorBound = 29.258;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 396)
    {
        innerInvRadiusPositiveErrorBound = 0.076;
        innerInvRadiusNegativeErrorBound = 0.445;
        outerInvRadiusPositiveErrorBound = 0.475;
        outerInvRadiusNegativeErrorBound = 0.703;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 396)
    {
        innerInvRadiusPositiveErrorBound = 0.366;
        innerInvRadiusNegativeErrorBound = 3.608;
        outerInvRadiusPositiveErrorBound = 0.507;
        outerInvRadiusNegativeErrorBound = 4.112;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 398)
    {
        innerInvRadiusPositiveErrorBound = 0.061;
        innerInvRadiusNegativeErrorBound = 0.106;
        outerInvRadiusPositiveErrorBound = 0.431;
        outerInvRadiusNegativeErrorBound = 29.258;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 408)
    {
        innerInvRadiusPositiveErrorBound = 0.021;
        innerInvRadiusNegativeErrorBound = 0.272;
        outerInvRadiusPositiveErrorBound = 0.975;
        outerInvRadiusNegativeErrorBound = 0.975;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 408)
    {
        innerInvRadiusPositiveErrorBound = 0.156;
        innerInvRadiusNegativeErrorBound = 0.944;
        outerInvRadiusPositiveErrorBound = 0.475;
        outerInvRadiusNegativeErrorBound = 0.975;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 408)
    {
        innerInvRadiusPositiveErrorBound = 0.828;
        innerInvRadiusNegativeErrorBound = 0.828;
        outerInvRadiusPositiveErrorBound = 0.378;
        outerInvRadiusNegativeErrorBound = 0.378;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 408)
    {
        innerInvRadiusPositiveErrorBound = 0.321;
        innerInvRadiusNegativeErrorBound = 0.944;
        outerInvRadiusPositiveErrorBound = 0.460;
        outerInvRadiusNegativeErrorBound = 2.688;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 412)
    {
        innerInvRadiusPositiveErrorBound = 0.210;
        innerInvRadiusNegativeErrorBound = 0.541;
        outerInvRadiusPositiveErrorBound = 0.366;
        outerInvRadiusNegativeErrorBound = 0.475;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 432)
    {
        innerInvRadiusPositiveErrorBound = 0.255;
        innerInvRadiusNegativeErrorBound = 0.255;
        outerInvRadiusPositiveErrorBound = 0.231;
        outerInvRadiusNegativeErrorBound = 0.231;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 898)
    {
        innerInvRadiusPositiveErrorBound = 0.196;
        innerInvRadiusNegativeErrorBound = 2.870;
        outerInvRadiusPositiveErrorBound = 0.113;
        outerInvRadiusNegativeErrorBound = 0.431;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 898)
    {
        innerInvRadiusPositiveErrorBound = 0.311;
        innerInvRadiusNegativeErrorBound = 0.311;
        outerInvRadiusPositiveErrorBound = 0.681;
        outerInvRadiusNegativeErrorBound = 0.681;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 898)
    {
        innerInvRadiusPositiveErrorBound = 0.638;
        innerInvRadiusNegativeErrorBound = 2.870;
        outerInvRadiusPositiveErrorBound = 0.281;
        outerInvRadiusNegativeErrorBound = 0.431;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 898)
    {
        innerInvRadiusPositiveErrorBound = 0.124;
        innerInvRadiusNegativeErrorBound = 0.311;
        outerInvRadiusPositiveErrorBound = 0.137;
        outerInvRadiusNegativeErrorBound = 0.727;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 898)
    {
        innerInvRadiusPositiveErrorBound = 0.491;
        innerInvRadiusNegativeErrorBound = 2.518;
        outerInvRadiusPositiveErrorBound = 0.460;
        outerInvRadiusNegativeErrorBound = 0.445;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 900)
    {
        innerInvRadiusPositiveErrorBound = 0.184;
        innerInvRadiusNegativeErrorBound = 0.042;
        outerInvRadiusPositiveErrorBound = 0.042;
        outerInvRadiusNegativeErrorBound = 0.003;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 900)
    {
        innerInvRadiusPositiveErrorBound = 0.038;
        innerInvRadiusNegativeErrorBound = 0.038;
        outerInvRadiusPositiveErrorBound = 0.146;
        outerInvRadiusNegativeErrorBound = 0.146;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 900)
    {
        innerInvRadiusPositiveErrorBound = 0.184;
        innerInvRadiusNegativeErrorBound = 0.475;
        outerInvRadiusPositiveErrorBound = 0.403;
        outerInvRadiusNegativeErrorBound = 7.655;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 900)
    {
        innerInvRadiusPositiveErrorBound = 1.149;
        innerInvRadiusNegativeErrorBound = 1.149;
        outerInvRadiusPositiveErrorBound = 0.378;
        outerInvRadiusNegativeErrorBound = 2.138;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 900)
    {
        innerInvRadiusPositiveErrorBound = 0.491;
        innerInvRadiusNegativeErrorBound = 24.045;
        outerInvRadiusPositiveErrorBound = 0.524;
        outerInvRadiusNegativeErrorBound = 4.536;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 902)
    {
        innerInvRadiusPositiveErrorBound = 0.301;
        innerInvRadiusNegativeErrorBound = 0.578;
        outerInvRadiusPositiveErrorBound = 0.431;
        outerInvRadiusNegativeErrorBound = 2.138;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 904)
    {
        innerInvRadiusPositiveErrorBound = 0.036;
        innerInvRadiusNegativeErrorBound = 3.608;
        outerInvRadiusPositiveErrorBound = 0.578;
        outerInvRadiusNegativeErrorBound = 0.210;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 904)
    {
        innerInvRadiusPositiveErrorBound = 0.524;
        innerInvRadiusNegativeErrorBound = 15.212;
        outerInvRadiusPositiveErrorBound = 0.703;
        outerInvRadiusNegativeErrorBound = 15.718;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 908)
    {
        innerInvRadiusPositiveErrorBound = 0.016;
        innerInvRadiusNegativeErrorBound = 0.016;
        outerInvRadiusPositiveErrorBound = 8.172;
        outerInvRadiusNegativeErrorBound = 8.172;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 1920)
    {
        innerInvRadiusPositiveErrorBound = 0.431;
        innerInvRadiusNegativeErrorBound = 0.239;
        outerInvRadiusPositiveErrorBound = 0.491;
        outerInvRadiusNegativeErrorBound = 0.445;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 1920)
    {
        innerInvRadiusPositiveErrorBound = 0.063;
        innerInvRadiusNegativeErrorBound = 0.063;
        outerInvRadiusPositiveErrorBound = 0.311;
        outerInvRadiusNegativeErrorBound = 0.311;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 1920)
    {
        innerInvRadiusPositiveErrorBound = 0.617;
        innerInvRadiusNegativeErrorBound = 0.975;
        outerInvRadiusPositiveErrorBound = 0.681;
        outerInvRadiusNegativeErrorBound = 2.283;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 1920)
    {
        innerInvRadiusPositiveErrorBound = 0.178;
        innerInvRadiusNegativeErrorBound = 0.301;
        outerInvRadiusPositiveErrorBound = 0.210;
        outerInvRadiusNegativeErrorBound = 1.398;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 1920)
    {
        innerInvRadiusPositiveErrorBound = 0.524;
        innerInvRadiusNegativeErrorBound = 4.249;
        outerInvRadiusPositiveErrorBound = 0.659;
        outerInvRadiusNegativeErrorBound = 2.437;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 1922)
    {
        innerInvRadiusPositiveErrorBound = 0.231;
        innerInvRadiusNegativeErrorBound = 0.703;
        outerInvRadiusPositiveErrorBound = 0.102;
        outerInvRadiusNegativeErrorBound = 0.802;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 1924)
    {
        innerInvRadiusPositiveErrorBound = 0.120;
        innerInvRadiusNegativeErrorBound = 3.064;
        outerInvRadiusPositiveErrorBound = 0.311;
        outerInvRadiusNegativeErrorBound = 0.944;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 3840)
    {
        innerInvRadiusPositiveErrorBound = 0.491;
        innerInvRadiusNegativeErrorBound = 0.914;
        outerInvRadiusPositiveErrorBound = 0.178;
        outerInvRadiusNegativeErrorBound = 1.492;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 3840)
    {
        innerInvRadiusPositiveErrorBound = 0.491;
        innerInvRadiusNegativeErrorBound = 0.264;
        outerInvRadiusPositiveErrorBound = 0.071;
        outerInvRadiusNegativeErrorBound = 0.255;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 3840)
    {
        innerInvRadiusPositiveErrorBound = 0.491;
        innerInvRadiusNegativeErrorBound = 108.230;
        outerInvRadiusPositiveErrorBound = 0.507;
        outerInvRadiusNegativeErrorBound = 1.816;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 3840)
    {
        innerInvRadiusPositiveErrorBound = 0.491;
        innerInvRadiusNegativeErrorBound = 0.638;
        outerInvRadiusPositiveErrorBound = 0.431;
        outerInvRadiusNegativeErrorBound = 0.776;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 3840)
    {
        innerInvRadiusPositiveErrorBound = 0.445;
        innerInvRadiusNegativeErrorBound = 2.437;
        outerInvRadiusPositiveErrorBound = 0.460;
        outerInvRadiusNegativeErrorBound = 1.542;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 3968)
    {
        innerInvRadiusPositiveErrorBound = 0.113;
        innerInvRadiusNegativeErrorBound = 0.975;
        outerInvRadiusPositiveErrorBound = 0.403;
        outerInvRadiusNegativeErrorBound = 1.076;
    }

    innerRadiusInvMin = fmaxf(0.0, (1 - innerInvRadiusPositiveErrorBound) / innerRadius);
    innerRadiusInvMax = (1 + innerInvRadiusNegativeErrorBound) / innerRadius;

    outerRadiusInvMin = fmaxf(0.0, (1 - outerInvRadiusPositiveErrorBound) / outerRadius);
    outerRadiusInvMax = (1 + outerInvRadiusNegativeErrorBound) / outerRadius;

    return checkIntervalOverlap(innerRadiusInvMin, innerRadiusInvMax, outerRadiusInvMin, outerRadiusInvMax);

}


__device__ bool SDL::passTERZChiSquaredCuts(int nLayerOverlaps, int nHitOverlaps, unsigned int layer_binary, float rzChiSquared)
{
    if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 127)
    {
        return rzChiSquared < 7.003;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 3971)
    {
        return rzChiSquared < 97.948;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 191)
    {
        return rzChiSquared < 10.621;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 1927)
    {
        return rzChiSquared < 97.948;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 415)
    {
        return rzChiSquared < 10.621;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 911)
    {
        return rzChiSquared < 97.948;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 3973)
    {
        return rzChiSquared < 85.250;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 31)
    {
        return rzChiSquared < 258.885;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 901)
    {
        return rzChiSquared < 148.559;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 61)
    {
        return rzChiSquared < 297.446;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 143)
    {
        return rzChiSquared < 392.655;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 1921)
    {
        return rzChiSquared < 129.300;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 157)
    {
        return rzChiSquared < 392.655;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 391)
    {
        return rzChiSquared < 258.885;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 3841)
    {
        return rzChiSquared < 112.537;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 899)
    {
        return rzChiSquared < 129.300;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 397)
    {
        return rzChiSquared < 518.339;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 126)
    {
        return rzChiSquared < 28.073;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 3970)
    {
        return rzChiSquared < 903.274;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 414)
    {
        return rzChiSquared < 56.207;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 445)
    {
        return rzChiSquared < 3.044;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 190)
    {
        return rzChiSquared < 48.920;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 1926)
    {
        return rzChiSquared < 786.173;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 910)
    {
        return rzChiSquared < 684.253;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 3972)
    {
        return rzChiSquared < 595.546;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 444)
    {
        return rzChiSquared < 56.207;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 1933)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 127)
    {
        return rzChiSquared < 8.046;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 3971)
    {
        return rzChiSquared < 97.948;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 191)
    {
        return rzChiSquared < 10.621;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 1927)
    {
        return rzChiSquared < 97.948;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 415)
    {
        return rzChiSquared < 4.018;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 911)
    {
        return rzChiSquared < 74.198;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 3973)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 31)
    {
        return rzChiSquared < 297.446;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 901)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 61)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 143)
    {
        return rzChiSquared < 56.207;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 1921)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 391)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 3841)
    {
        return rzChiSquared < 18.509;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 899)
    {
        return rzChiSquared < 64.579;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 397)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 126)
    {
        return rzChiSquared < 28.073;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 1926)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 910)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 3972)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 127)
    {
        return rzChiSquared < 5.305;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 3971)
    {
        return rzChiSquared < 97.948;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 191)
    {
        return rzChiSquared < 10.621;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 1927)
    {
        return rzChiSquared < 85.250;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 415)
    {
        return rzChiSquared < 7.003;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 911)
    {
        return rzChiSquared < 48.920;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 3973)
    {
        return rzChiSquared < 37.058;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 31)
    {
        return rzChiSquared < 112.537;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 901)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 61)
    {
        return rzChiSquared < 32.254;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 143)
    {
        return rzChiSquared < 225.322;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 1921)
    {
        return rzChiSquared < 24.433;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 391)
    {
        return rzChiSquared < 196.111;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 3841)
    {
        return rzChiSquared < 129.300;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 899)
    {
        return rzChiSquared < 74.198;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 397)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 126)
    {
        return rzChiSquared < 28.073;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 3970)
    {
        return rzChiSquared < 903.274;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 414)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 1926)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 3972)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 191)
    {
        return rzChiSquared < 9.244;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 1927)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 415)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 3973)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 31)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 143)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 1921)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 391)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 3841)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 899)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 397)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 3970)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 414)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 190)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 1926)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 127)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 3971)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 191)
    {
        return rzChiSquared < 9.244;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 1927)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 415)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 911)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 3973)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 31)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 61)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 1921)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 391)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 3841)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 899)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 397)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 1926)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 3972)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 2 and layer_binary == 3975)
    {
        return rzChiSquared < 64.579;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 2 and layer_binary == 63)
    {
        return rzChiSquared < 258.885;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 2 and layer_binary == 125)
    {
        return rzChiSquared < 21.266;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 2 and layer_binary == 3969)
    {
        return rzChiSquared < 129.300;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 2 and layer_binary == 1925)
    {
        return rzChiSquared < 48.920;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 2 and layer_binary == 413)
    {
        return rzChiSquared < 10.621;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 2 and layer_binary == 903)
    {
        return rzChiSquared < 148.559;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 2 and layer_binary == 399)
    {
        return rzChiSquared < 258.885;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 2 and layer_binary == 189)
    {
        return rzChiSquared < 12.203;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 2 and layer_binary == 1923)
    {
        return rzChiSquared < 148.559;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 2 and layer_binary == 909)
    {
        return rzChiSquared < 97.948;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 2 and layer_binary == 159)
    {
        return rzChiSquared < 451.141;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 2 and layer_binary == 1935)
    {
        return rzChiSquared < 24.433;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 2 and layer_binary == 447)
    {
        return rzChiSquared < 6.095;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 2 and layer_binary == 3974)
    {
        return true;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 1 and layer_binary == 3975)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 1 and layer_binary == 63)
    {
        return rzChiSquared < 14.021;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 1 and layer_binary == 3969)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 1 and layer_binary == 1925)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 1 and layer_binary == 903)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 1 and layer_binary == 399)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 1 and layer_binary == 1923)
    {
        return rzChiSquared < 14.021;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 1 and layer_binary == 909)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 1 and layer_binary == 3974)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 3975)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 63)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 125)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 3969)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 903)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 399)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 1923)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 909)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 159)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 1935)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 446)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 3974)
    {
        return false;
    }

    //TT3

    else if(layer_binary == 30 && nLayerOverlaps == 2 and nHitOverlaps == 0)
    {
        return rzChiSquared < 1431.299;
    }
    else if(layer_binary == 30 && nLayerOverlaps == 2 and nHitOverlaps == 1)
    {
        return rzChiSquared < 1431.299;
    }
    else if(layer_binary == 30 && nLayerOverlaps == 2 and nHitOverlaps == 2)
    {
        return rzChiSquared < 1431.299;
    }
    else if(layer_binary == 30 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return rzChiSquared < 1431.299;
    }
    else if(layer_binary == 30 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return rzChiSquared < 1082.725;
    }
    else if(layer_binary == 60 && nLayerOverlaps == 2 and nHitOverlaps == 0)
    {
        return rzChiSquared < 468.686;
    }
    else if(layer_binary == 60 && nLayerOverlaps == 2 and nHitOverlaps == 1)
    {
        return rzChiSquared < 619.575;
    }
    else if(layer_binary == 60 && nLayerOverlaps == 2 and nHitOverlaps == 2)
    {
        return rzChiSquared < 819.042;
    }
    else if(layer_binary == 60 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return rzChiSquared < 819.042;
    }
    else if(layer_binary == 60 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return rzChiSquared < 819.042;
    }
    else if(layer_binary == 62 && nLayerOverlaps == 1 and nHitOverlaps == 0)
    {
        return rzChiSquared < 819.042;
    }
    else if(layer_binary == 120 && nLayerOverlaps == 2 and nHitOverlaps == 0)
    {
        return rzChiSquared < 0.000;
    }
    else if(layer_binary == 120 && nLayerOverlaps == 2 and nHitOverlaps == 1)
    {
        return rzChiSquared < 0.000;
    }
    else if(layer_binary == 120 && nLayerOverlaps == 2 and nHitOverlaps == 2)
    {
        return rzChiSquared < 0.000;
    }
    else if(layer_binary == 120 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return rzChiSquared < 0.000;
    }
    else if(layer_binary == 120 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return rzChiSquared < 0.000;
    }
    else if(layer_binary == 124 && nLayerOverlaps == 1 and nHitOverlaps == 0)
    {
        return rzChiSquared < 619.575;
    }
    else if(layer_binary == 142 && nLayerOverlaps == 2 and nHitOverlaps == 0)
    {
        return rzChiSquared < 2501.235;
    }
    else if(layer_binary == 142 && nLayerOverlaps == 2 and nHitOverlaps == 1)
    {
        return rzChiSquared < 2501.235;
    }
    else if(layer_binary == 142 && nLayerOverlaps == 2 and nHitOverlaps == 2)
    {
        return rzChiSquared < 2501.235;
    }
    else if(layer_binary == 142 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return rzChiSquared < 2501.235;
    }
    else if(layer_binary == 142 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return rzChiSquared < 1892.093;
    }
    else if(layer_binary == 156 && nLayerOverlaps == 2 and nHitOverlaps == 0)
    {
        return rzChiSquared < 1431.299;
    }
    else if(layer_binary == 156 && nLayerOverlaps == 2 and nHitOverlaps == 1)
    {
        return false;
    }
    else if(layer_binary == 156 && nLayerOverlaps == 2 and nHitOverlaps == 2)
    {
        return rzChiSquared < 1431.299;
    }
    else if(layer_binary == 156 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return rzChiSquared < 1431.299;
    }
    else if(layer_binary == 156 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return rzChiSquared < 1431.299;
    }
    else if(layer_binary == 158 && nLayerOverlaps == 1 and nHitOverlaps == 0)
    {
        return rzChiSquared < 1431.299;
    }
    else if(layer_binary == 184 && nLayerOverlaps == 2 and nHitOverlaps == 0)
    {
        return false;
    }
    else if(layer_binary == 184 && nLayerOverlaps == 2 and nHitOverlaps == 1)
    {
        return rzChiSquared < 0.000;
    }
    else if(layer_binary == 184 && nLayerOverlaps == 2 and nHitOverlaps == 2)
    {
        return rzChiSquared < 0.000;
    }
    else if(layer_binary == 184 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return rzChiSquared < 0.000;
    }
    else if(layer_binary == 184 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return rzChiSquared < 0.000;
    }
    else if(layer_binary == 188 && nLayerOverlaps == 1 and nHitOverlaps == 0)
    {
        return rzChiSquared < 819.042;
    }
    else if(layer_binary == 390 && nLayerOverlaps == 2 and nHitOverlaps == 0)
    {
        return rzChiSquared < 2501.235;
    }
    else if(layer_binary == 390 && nLayerOverlaps == 2 and nHitOverlaps == 1)
    {
        return rzChiSquared < 2501.235;
    }
    else if(layer_binary == 390 && nLayerOverlaps == 2 and nHitOverlaps == 2)
    {
        return rzChiSquared < 2501.235;
    }
    else if(layer_binary == 390 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return rzChiSquared < 2501.235;
    }
    else if(layer_binary == 390 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return rzChiSquared < 2501.235;
    }
    else if(layer_binary == 396 && nLayerOverlaps == 2 and nHitOverlaps == 0)
    {
        return rzChiSquared < 1892.093;
    }
    else if(layer_binary == 396 && nLayerOverlaps == 2 and nHitOverlaps == 1)
    {
        return rzChiSquared < 1892.093;
    }
    else if(layer_binary == 396 && nLayerOverlaps == 2 and nHitOverlaps == 2)
    {
        return rzChiSquared < 1892.093;
    }
    else if(layer_binary == 396 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return rzChiSquared < 1892.093;
    }
    else if(layer_binary == 396 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return rzChiSquared < 1892.093;
    }
    else if(layer_binary == 398 && nLayerOverlaps == 1 and nHitOverlaps == 0)
    {
        return rzChiSquared < 1892.093;
    }
    else if(layer_binary == 408 && nLayerOverlaps == 2 and nHitOverlaps == 0)
    {
        return false;
    }
    else if(layer_binary == 408 && nLayerOverlaps == 2 and nHitOverlaps == 1)
    {
        return false;
    }
    else if(layer_binary == 408 && nLayerOverlaps == 2 and nHitOverlaps == 2)
    {
        return rzChiSquared < 0.000;
    }
    else if(layer_binary == 408 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return rzChiSquared < 0.000;
    }
    else if(layer_binary == 408 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return rzChiSquared < 0.000;
    }
    else if(layer_binary == 412 && nLayerOverlaps == 1 and nHitOverlaps == 0)
    {
        return rzChiSquared < 1431.299;
    }
    else if(layer_binary == 432 && nLayerOverlaps == 2 and nHitOverlaps == 0)
    {
        return false;
    }
    else if(layer_binary == 432 && nLayerOverlaps == 2 and nHitOverlaps == 1)
    {
        return false;
    }
    else if(layer_binary == 432 && nLayerOverlaps == 2 and nHitOverlaps == 2)
    {
        return false;
    }
    else if(layer_binary == 432 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return false;
    }
    else if(layer_binary == 432 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return rzChiSquared < 0.000;
    }
    else if(layer_binary == 440 && nLayerOverlaps == 1 and nHitOverlaps == 0)
    {
        return false;
    }
    else if(layer_binary == 898 && nLayerOverlaps == 2 and nHitOverlaps == 0)
    {
        return rzChiSquared < 2501.235;
    }
    else if(layer_binary == 898 && nLayerOverlaps == 2 and nHitOverlaps == 1)
    {
        return rzChiSquared < 2501.235;
    }
    else if(layer_binary == 898 && nLayerOverlaps == 2 and nHitOverlaps == 2)
    {
        return rzChiSquared < 2501.235;
    }
    else if(layer_binary == 898 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return rzChiSquared < 2501.235;
    }
    else if(layer_binary == 898 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return rzChiSquared < 2501.235;
    }
    else if(layer_binary == 900 && nLayerOverlaps == 2 and nHitOverlaps == 0)
    {
        return rzChiSquared < 1892.093;
    }
    else if(layer_binary == 900 && nLayerOverlaps == 2 and nHitOverlaps == 1)
    {
        return rzChiSquared < 1431.299;
    }
    else if(layer_binary == 900 && nLayerOverlaps == 2 and nHitOverlaps == 2)
    {
        return rzChiSquared < 1892.093;
    }
    else if(layer_binary == 900 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return rzChiSquared < 1892.093;
    }
    else if(layer_binary == 900 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return rzChiSquared < 1892.093;
    }
    else if(layer_binary == 902 && nLayerOverlaps == 1 and nHitOverlaps == 0)
    {
        return rzChiSquared < 2501.235;
    }
    else if(layer_binary == 904 && nLayerOverlaps == 2 and nHitOverlaps == 0)
    {
        return false;
    }
    else if(layer_binary == 904 && nLayerOverlaps == 2 and nHitOverlaps == 1)
    {
        return false;
    }
    else if(layer_binary == 904 && nLayerOverlaps == 2 and nHitOverlaps == 2)
    {
        return rzChiSquared < 1892.093;
    }
    else if(layer_binary == 904 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return rzChiSquared < 1892.093;
    }
    else if(layer_binary == 904 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return rzChiSquared < 1892.093;
    }
    else if(layer_binary == 908 && nLayerOverlaps == 1 and nHitOverlaps == 0)
    {
        return rzChiSquared < 1892.093;
    }
    else if(layer_binary == 1920 && nLayerOverlaps == 2 and nHitOverlaps == 0)
    {
        return rzChiSquared < 819.042;
    }
    else if(layer_binary == 1920 && nLayerOverlaps == 2 and nHitOverlaps == 1)
    {
        return rzChiSquared < 819.042;
    }
    else if(layer_binary == 1920 && nLayerOverlaps == 2 and nHitOverlaps == 2)
    {
        return rzChiSquared < 819.042;
    }
    else if(layer_binary == 1920 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return rzChiSquared < 819.042;
    }
    else if(layer_binary == 1920 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return rzChiSquared < 819.042;
    }
    else if(layer_binary == 1922 && nLayerOverlaps == 1 and nHitOverlaps == 0)
    {
        return rzChiSquared < 1892.093;
    }
    else if(layer_binary == 1924 && nLayerOverlaps == 1 and nHitOverlaps == 0)
    {
        return false;
    }
    else if(layer_binary == 3840 && nLayerOverlaps == 2 and nHitOverlaps == 0)
    {
        return rzChiSquared < 819.042;
    }
    else if(layer_binary == 3840 && nLayerOverlaps == 2 and nHitOverlaps == 1)
    {
        return rzChiSquared < 819.042;
    }
    else if(layer_binary == 3840 && nLayerOverlaps == 2 and nHitOverlaps == 2)
    {
        return rzChiSquared < 819.042;
    }
    else if(layer_binary == 3840 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return rzChiSquared < 819.042;
    }
    else if(layer_binary == 3840 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return rzChiSquared < 819.042;
    }
    else if(layer_binary == 3968 && nLayerOverlaps == 1 and nHitOverlaps == 0)
    {
        return rzChiSquared < 819.042;
    }



    return true;
}

__device__ bool SDL::passTERPhiChiSquaredCuts(int nLayerOverlaps, int nHitOverlaps, unsigned int layer_binary, float rPhiChiSquared)
{
    if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 127)
    {
        return rPhiChiSquared < 21.266;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 3971)
    {
        return rPhiChiSquared < 6.095;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 1927)
    {
        return rPhiChiSquared < 4.617;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 415)
    {
        return rPhiChiSquared < 9.244;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 191)
    {
        return rPhiChiSquared < 18.509;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 3973)
    {
        return rPhiChiSquared < 2.649;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 911)
    {
        return rPhiChiSquared < 5.305;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 31)
    {
        return rPhiChiSquared < 12634.215;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 391)
    {
        return rPhiChiSquared < 16678.281;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 3841)
    {
        return rPhiChiSquared < 16678.281;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 61)
    {
        return rPhiChiSquared < 3151.615;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 143)
    {
        return rPhiChiSquared < 88261.109;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 901)
    {
        return rPhiChiSquared < 5492.110;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 1921)
    {
        return rPhiChiSquared < 33393.261;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 397)
    {
        return rPhiChiSquared < 1037.818;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 899)
    {
        return rPhiChiSquared < 153806.756;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 3970)
    {
        return rPhiChiSquared < 37.058;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 414)
    {
        return rPhiChiSquared < 64.579;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 126)
    {
        return rPhiChiSquared < 85.250;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 445)
    {
        return true;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 1926)
    {
        return rPhiChiSquared < 0.436;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 190)
    {
        return rPhiChiSquared < 9.244;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 3972)
    {
        return rPhiChiSquared < 0.010;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 157)
    {
        return rPhiChiSquared < 297.446;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 910)
    {
        return rPhiChiSquared < 0.759;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 444)
    {
        return true;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 127)
    {
        return rPhiChiSquared < 14.021;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 3971)
    {
        return rPhiChiSquared < 2.649;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 1927)
    {
        return rPhiChiSquared < 7.003;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 415)
    {
        return rPhiChiSquared < 2.007;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 191)
    {
        return rPhiChiSquared < 3.044;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 3973)
    {
        return rPhiChiSquared < 0.041;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 911)
    {
        return rPhiChiSquared < 1.323;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 31)
    {
        return rPhiChiSquared < 203038.514;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 391)
    {
        return rPhiChiSquared < 297.446;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 3841)
    {
        return rPhiChiSquared < 196.111;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 61)
    {
        return rPhiChiSquared < 8.046;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 143)
    {
        return rPhiChiSquared < 1370.011;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 901)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 1921)
    {
        return rPhiChiSquared < 42.578;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 397)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 899)
    {
        return rPhiChiSquared < 595.546;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 414)
    {
        return rPhiChiSquared < 0.000;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 126)
    {
        return rPhiChiSquared < 8.046;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 3840)
    {
        return rPhiChiSquared < 4.617;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 1920)
    {
        return rPhiChiSquared < 0.872;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 445)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 1926)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 190)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 3972)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 157)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 910)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 444)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 1933)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 127)
    {
        return rPhiChiSquared < 10.621;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 3971)
    {
        return rPhiChiSquared < 3.044;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 1927)
    {
        return rPhiChiSquared < 4.018;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 415)
    {
        return rPhiChiSquared < 4.617;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 191)
    {
        return rPhiChiSquared < 12.203;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 3973)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 911)
    {
        return rPhiChiSquared < 0.872;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 31)
    {
        return rPhiChiSquared < 935179.568;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 391)
    {
        return rPhiChiSquared < 0.041;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 3841)
    {
        return rPhiChiSquared < 8.046;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 61)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 143)
    {
        return rPhiChiSquared < 4.018;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 901)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 1921)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 397)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 899)
    {
        return rPhiChiSquared < 3621.052;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 3970)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 414)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 126)
    {
        return rPhiChiSquared < 6.095;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 3840)
    {
        return rPhiChiSquared < 4.617;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 1920)
    {
        return rPhiChiSquared < 0.218;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 1926)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 190)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 157)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 127)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 3971)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 1927)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 415)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 3973)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 911)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 31)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 3841)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 61)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 143)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 899)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 126)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 3840)
    {
        return rPhiChiSquared < 0.003;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 1920)
    {
        return rPhiChiSquared < 0.010;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 157)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 127)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 3971)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 1927)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 415)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 191)
    {
        return rPhiChiSquared < 2.649;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 3973)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 911)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 31)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 61)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 143)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 901)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 397)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 899)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 414)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 126)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 3840)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 1920)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 445)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 1926)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 190)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 910)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 2 and layer_binary == 3975)
    {
        return rPhiChiSquared < 5.305;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 2 and layer_binary == 1923)
    {
        return rPhiChiSquared < 44082.056;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 2 and layer_binary == 399)
    {
        return rPhiChiSquared < 1574.076;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 2 and layer_binary == 63)
    {
        return rPhiChiSquared < 8329.976;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 2 and layer_binary == 125)
    {
        return rPhiChiSquared < 4780.108;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 2 and layer_binary == 1925)
    {
        return rPhiChiSquared < 170.687;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 2 and layer_binary == 159)
    {
        return rPhiChiSquared < 3621.052;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 2 and layer_binary == 3969)
    {
        return rPhiChiSquared < 268028.788;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 2 and layer_binary == 903)
    {
        return rPhiChiSquared < 12634.215;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 2 and layer_binary == 413)
    {
        return rPhiChiSquared < 684.253;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 2 and layer_binary == 189)
    {
        return rPhiChiSquared < 1370.011;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 2 and layer_binary == 909)
    {
        return rPhiChiSquared < 225.322;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 2 and layer_binary == 1935)
    {
        return rPhiChiSquared < 0.872;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 2 and layer_binary == 447)
    {
        return true;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 2 and layer_binary == 3974)
    {
        return rPhiChiSquared < 0.002;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 2 and layer_binary == 446)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 1 and layer_binary == 3975)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 1 and layer_binary == 1923)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 1 and layer_binary == 399)
    {
        return rPhiChiSquared < 6.095;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 1 and layer_binary == 63)
    {
        return rPhiChiSquared < 4.018;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 1 and layer_binary == 125)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 1 and layer_binary == 1925)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 1 and layer_binary == 159)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 1 and layer_binary == 3969)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 1 and layer_binary == 903)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 1 and layer_binary == 413)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 1 and layer_binary == 189)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 3975)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 1923)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 399)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 63)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 125)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 1925)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 159)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 3969)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 903)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 413)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 189)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 909)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 447)
    {
        return false;
    }

    //T3T3
    if(layer_binary == 900 && nLayerOverlaps == 2 and nHitOverlaps == 1)
    {
        return rPhiChiSquared < 0.002;
    }
    else if(layer_binary == 142 && nLayerOverlaps == 2 and nHitOverlaps == 0)
    {
        return rPhiChiSquared < 0.027;
    }
    else if(layer_binary == 1922 && nLayerOverlaps == 1 and nHitOverlaps == 0)
    {
        return rPhiChiSquared < 0.437;
    }
    else if(layer_binary == 1924 && nLayerOverlaps == 1 and nHitOverlaps == 0)
    {
        return rPhiChiSquared < 0.189;
    }
    else if(layer_binary == 898 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return rPhiChiSquared < 2.333;
    }
    else if(layer_binary == 3840 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return rPhiChiSquared < 2.333;
    }
    else if(layer_binary == 1920 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return rPhiChiSquared < 2.333;
    }
    else if(layer_binary == 902 && nLayerOverlaps == 1 and nHitOverlaps == 0)
    {
        return rPhiChiSquared < 1.010;
    }
    else if(layer_binary == 390 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return rPhiChiSquared < 5.389;
    }
    else if(layer_binary == 904 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return rPhiChiSquared < 0.250;
    }
    else if(layer_binary == 3968 && nLayerOverlaps == 1 and nHitOverlaps == 0)
    {
        return rPhiChiSquared < 1.010;
    }
    else if(layer_binary == 142 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return rPhiChiSquared < 1.765;
    }
    else if(layer_binary == 390 && nLayerOverlaps == 2 and nHitOverlaps == 0)
    {
        return rPhiChiSquared < 0.331;
    }
    else if(layer_binary == 390 && nLayerOverlaps == 2 and nHitOverlaps == 1)
    {
        return rPhiChiSquared < 0.035;
    }
    else if(layer_binary == 390 && nLayerOverlaps == 2 and nHitOverlaps == 2)
    {
        return rPhiChiSquared < 2.333;
    }
    else if(layer_binary == 898 && nLayerOverlaps == 2 and nHitOverlaps == 2)
    {
        return rPhiChiSquared < 3.084;
    }
    else if(layer_binary == 142 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return rPhiChiSquared < 0.189;
    }
    else if(layer_binary == 900 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return rPhiChiSquared < 2.333;
    }
    else if(layer_binary == 30 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return rPhiChiSquared < 2.333;
    }
    else if(layer_binary == 1920 && nLayerOverlaps == 2 and nHitOverlaps == 2)
    {
        return rPhiChiSquared < 3.084;
    }
    else if(layer_binary == 142 && nLayerOverlaps == 2 and nHitOverlaps == 1)
    {
        return rPhiChiSquared < 0.035;
    }
    else if(layer_binary == 142 && nLayerOverlaps == 2 and nHitOverlaps == 2)
    {
        return rPhiChiSquared < 1.765;
    }
    else if(layer_binary == 3840 && nLayerOverlaps == 2 and nHitOverlaps == 2)
    {
        return false;
    }
    else if(layer_binary == 62 && nLayerOverlaps == 1 and nHitOverlaps == 0)
    {
        return false;
    }
    else if(layer_binary == 898 && nLayerOverlaps == 2 and nHitOverlaps == 1)
    {
        return false;
    }
    else if(layer_binary == 3840 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return false;
    }
    else if(layer_binary == 1920 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return false;
    }
    else if(layer_binary == 904 && nLayerOverlaps == 2 and nHitOverlaps == 2)
    {
        return false;
    }
    else if(layer_binary == 390 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return false;
    }
    else if(layer_binary == 898 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return false;
    }
    else if(layer_binary == 398 && nLayerOverlaps == 1 and nHitOverlaps == 0)
    {
        return false;
    }
    else if(layer_binary == 396 && nLayerOverlaps == 2 and nHitOverlaps == 1)
    {
        return false;
    }
    else if(layer_binary == 900 && nLayerOverlaps == 2 and nHitOverlaps == 2)
    {
        return false;
    }
    else if(layer_binary == 900 && nLayerOverlaps == 2 and nHitOverlaps == 0)
    {
        return false;
    }
    else if(layer_binary == 396 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return false;
    }
    else if(layer_binary == 30 && nLayerOverlaps == 2 and nHitOverlaps == 2)
    {
        return false;
    }
    else if(layer_binary == 432 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return false;
    }
    else if(layer_binary == 908 && nLayerOverlaps == 1 and nHitOverlaps == 0)
    {
        return false;
    }
    else if(layer_binary == 60 && nLayerOverlaps == 2 and nHitOverlaps == 0)
    {
        return false;
    }
    else if(layer_binary == 900 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return false;
    }
    else if(layer_binary == 120 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return false;
    }
    else if(layer_binary == 30 && nLayerOverlaps == 2 and nHitOverlaps == 0)
    {
        return false;
    }
    else if(layer_binary == 898 && nLayerOverlaps == 2 and nHitOverlaps == 0)
    {
        return false;
    }
    else if(layer_binary == 184 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return false;
    }
    else if(layer_binary == 60 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return false;
    }
    else if(layer_binary == 3840 && nLayerOverlaps == 2 and nHitOverlaps == 1)
    {
        return false;
    }
    else if(layer_binary == 408 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return false;
    }
    else if(layer_binary == 1920 && nLayerOverlaps == 2 and nHitOverlaps == 0)
    {
        return false;
    }
    else if(layer_binary == 184 && nLayerOverlaps == 2 and nHitOverlaps == 0)
    {
        return false;
    }
    else if(layer_binary == 408 && nLayerOverlaps == 2 and nHitOverlaps == 0)
    {
        return false;
    }
    else if(layer_binary == 124 && nLayerOverlaps == 1 and nHitOverlaps == 0)
    {
        return false;
    }
    else if(layer_binary == 1920 && nLayerOverlaps == 2 and nHitOverlaps == 1)
    {
        return false;
    }
    else if(layer_binary == 30 && nLayerOverlaps == 2 and nHitOverlaps == 1)
    {
        return false;
    }
    else if(layer_binary == 156 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return false;
    }
    else if(layer_binary == 30 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return false;
    }
    else if(layer_binary == 120 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return false;
    }
    else if(layer_binary == 3840 && nLayerOverlaps == 2 and nHitOverlaps == 0)
    {
        return false;
    }
    else if(layer_binary == 396 && nLayerOverlaps == 2 and nHitOverlaps == 2)
    {
        return false;
    }
    else if(layer_binary == 120 && nLayerOverlaps == 2 and nHitOverlaps == 2)
    {
        return false;
    }
    else if(layer_binary == 396 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return false;
    }
    else if(layer_binary == 188 && nLayerOverlaps == 1 and nHitOverlaps == 0)
    {
        return false;
    }
    else if(layer_binary == 60 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return false;
    }
    else if(layer_binary == 408 && nLayerOverlaps == 2 and nHitOverlaps == 2)
    {
        return false;
    }
    else if(layer_binary == 60 && nLayerOverlaps == 2 and nHitOverlaps == 2)
    {
        return false;
    }
    else if(layer_binary == 412 && nLayerOverlaps == 1 and nHitOverlaps == 0)
    {
        return false;
    }
    else if(layer_binary == 184 && nLayerOverlaps == 2 and nHitOverlaps == 1)
    {
        return false;
    }
    else if(layer_binary == 408 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return false;
    }
    else if(layer_binary == 156 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return false;
    }
    else if(layer_binary == 184 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return false;
    }
    else if(layer_binary == 396 && nLayerOverlaps == 2 and nHitOverlaps == 0)
    {
        return false;
    }
    else if(layer_binary == 184 && nLayerOverlaps == 2 and nHitOverlaps == 2)
    {
        return false;
    }
    else if(layer_binary == 156 && nLayerOverlaps == 2 and nHitOverlaps == 2)
    {
        return false;
    }
    else if(layer_binary == 60 && nLayerOverlaps == 2 and nHitOverlaps == 1)
    {
        return false;
    }
    else if(layer_binary == 120 && nLayerOverlaps == 2 and nHitOverlaps == 0)
    {
        return false;
    }
    else if(layer_binary == 120 && nLayerOverlaps == 2 and nHitOverlaps == 1)
    {
        return false;
    }
    else if(layer_binary == 156 && nLayerOverlaps == 2 and nHitOverlaps == 0)
    {
        return false;
    }
    else if(layer_binary == 156 && nLayerOverlaps == 2 and nHitOverlaps == 1)
    {
        return false;
    }
    else if(layer_binary == 158 && nLayerOverlaps == 1 and nHitOverlaps == 0)
    {
        return false;
    }
    else if(layer_binary == 408 && nLayerOverlaps == 2 and nHitOverlaps == 1)
    {
        return false;
    }
    else if(layer_binary == 432 && nLayerOverlaps == 2 and nHitOverlaps == 0)
    {
        return false;
    }
    else if(layer_binary == 432 && nLayerOverlaps == 2 and nHitOverlaps == 1)
    {
        return false;
    }
    else if(layer_binary == 432 && nLayerOverlaps == 2 and nHitOverlaps == 2)
    {
        return false;
    }
    else if(layer_binary == 432 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return false;
    }
    else if(layer_binary == 440 && nLayerOverlaps == 1 and nHitOverlaps == 0)
    {
        return false;
    }
    else if(layer_binary == 904 && nLayerOverlaps == 2 and nHitOverlaps == 0)
    {
        return false;
    }
    else if(layer_binary == 904 && nLayerOverlaps == 2 and nHitOverlaps == 1)
    {
        return false;
    }
    else if(layer_binary == 904 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return false;
    }


    return true;
}


/*
   If one out of the two hits don't overlap - check if the module indices are identical (case of multiple reco hits)
   If both hits don't overlap - check the above, and check for staggered modules using the staggered module neighbours list

   This function i complicated - computes layer overlaps and checks if layer matches and hit matches are "compatible" i.e., layer overlap = 2 * hit overlap, or if that's not the case, we know why (multiple reco hits/staggered modules)
*/
__device__ bool SDL::computeLayerAndHitOverlaps(SDL::modules& modulesInGPU, uint8_t* anchorLayerIndices, unsigned int* anchorHitIndices, uint16_t* anchorLowerModuleIndices, uint8_t* outerObjectLayerIndices, unsigned int* outerObjectHitIndices, uint16_t* outerObjectLowerModuleIndices, unsigned int nAnchorLayers, unsigned int nOuterLayers, unsigned int& nLayerOverlap, unsigned int& nHitOverlap, unsigned int& layerOverlapTarget)
{
    bool pass = true;
    //merge technique!
    size_t j = 0; //outer object tracker
    unsigned int temp; //container variable
    unsigned int staggeredNeighbours[10];
    for(size_t i = 0; i < nAnchorLayers; i++)
    {
        if(anchorLayerIndices[i] == outerObjectLayerIndices[j])
        {
            //2*i and 2*i + 1 are the hits, similarly 2*j and 2*j+1
            nLayerOverlap++;
            temp = nHitOverlap; //before the hit matching shenanigans

            //FIXME:Assumption, 2*i and 2*i+1 hits are known to be from partner modules!
            if(anchorHitIndices[2 * i] == outerObjectHitIndices[2 * j])
            {
                nHitOverlap++;
            }
            else //check for same module indices
            {
                if(anchorLowerModuleIndices[i] != outerObjectLowerModuleIndices[j])
                {
                    pass = false;
                }
            }
            if(anchorHitIndices[2*i+1] == outerObjectHitIndices[2*j+1])
            {
                nHitOverlap++;
            }
            else //check for same module indices
            {
                if(anchorLowerModuleIndices[i] != outerObjectLowerModuleIndices[j])
                {
                    pass = false;
                }
            }
            
            if(nHitOverlap == temp) //check for staggered modules!
            {
                //this is a redemption case. If both modules did not match in the above case,
                //this case should redeem them!

                //find the neighbours of the anchor lower module, if any of those matches the outer lower module, we're done
                unsigned int nStaggeredModules;                
                findStaggeredNeighbours(modulesInGPU, anchorLowerModuleIndices[i], staggeredNeighbours, nStaggeredModules);
                for(size_t idx = 0; idx < nStaggeredModules; idx++)
                {
                    if(outerObjectLowerModuleIndices[j]  == staggeredNeighbours[idx])
                    {
                        //redeemed!
                        pass = true;
                    }
                }
            }

            j++;
            if(j == nOuterLayers)
            {
                break;
            }
        }
    }
    pass =  pass and (nLayerOverlap == layerOverlapTarget); //not really required, because these cases should be handled by the other conditions
    return pass;
}


/* r-z and r-phi chi squared computation*/
__device__ float SDL::computeTERZChiSquared(struct modules& modulesInGPU, struct hits& hitsInGPU, unsigned int* anchorObjectAnchorHitIndices, uint16_t* anchorLowerModuleIndices, unsigned int* outerObjectAnchorHitIndices, uint16_t* outerLowerModuleIndices, short anchorObjectType)
{
    //using the pixel hits to create the slope
    float slope = 0, intercept = 0, RMSE = 0;
    if(anchorObjectType != 4)
    {
        //use the first two anchor object anchor hits (i.e., the pixel hits)
        float& rtPix1 = hitsInGPU.rts[anchorObjectAnchorHitIndices[0]];
        float& rtPix2 = hitsInGPU.rts[anchorObjectAnchorHitIndices[1]];

        float& zPix1 = hitsInGPU.zs[anchorObjectAnchorHitIndices[0]];
        float& zPix2 = hitsInGPU.zs[anchorObjectAnchorHitIndices[1]];

        slope = (zPix2 - zPix1)/(rtPix2 - rtPix1);
        intercept = zPix1 - slope * rtPix1;
    }
    else
    {
        /*only PS modules taken into consideration*/
        float rts[5], zs[5];
        int nPoints = 0;
        for(size_t i =0; i < 5; i++)
        {
            if(modulesInGPU.moduleType[anchorLowerModuleIndices[i]] == SDL::PS)
            {
                rts[nPoints] = hitsInGPU.rts[anchorObjectAnchorHitIndices[i]];
                zs[nPoints] = hitsInGPU.zs[anchorObjectAnchorHitIndices[i]];
                nPoints++;
            }
        }
        if(nPoints <= 1)
        {
            slope = 0;
            intercept = 0;
        }
        else
        {
            fitStraightLine(nPoints, rts, zs, slope, intercept);
        }
    }
    if(slope != 0 and intercept != 0)
    {
        float rtAnchor, zAnchor, residual, error, drdz;
        for(size_t i = 0; i < 3; i++)
        {
            unsigned int& anchorHitIndex = outerObjectAnchorHitIndices[i];
            uint16_t& lowerModuleIndex = outerLowerModuleIndices[i];
            rtAnchor = hitsInGPU.rts[anchorHitIndex];
            zAnchor = hitsInGPU.zs[anchorHitIndex];

            //outerModuleAnchorHitIndices
            const int moduleType = modulesInGPU.moduleType[lowerModuleIndex];
            const int moduleSide = modulesInGPU.sides[lowerModuleIndex];
            //const int moduleLayerType = modulesInGPU.moduleLayerType[lowerModuleIndex];
            const int moduleSubdet = modulesInGPU.subdets[lowerModuleIndex]; 
            residual = (moduleSubdet == SDL::Barrel) ?  zAnchor - (slope * rtAnchor + intercept) : rtAnchor - (zAnchor/slope + intercept/slope);
        
            //PS Modules
            if(moduleType == 0)
            {
                error = 0.15;
            }
            else //2S modules
            {
                error = 5.0;
            }

            //special dispensation to tilted PS modules!
            if(moduleType == 0 and moduleSubdet == SDL::Barrel and moduleSide != Center)
            {
                drdz = modulesInGPU.drdzs[lowerModuleIndex];
                error *= 1/sqrtf(1 + drdz * drdz);
            }
            RMSE += (residual * residual)/(error * error);
        }
        RMSE = sqrtf(0.33 * RMSE);
    }
    return RMSE;
}

__device__ void SDL::fitStraightLine(int nPoints, float* xs, float* ys, float& slope, float& intercept)
{
    float sigmaX2(0), sigmaXY(0), sigmaX(0), sigmaY(0), sigma1(0);
    sigma1 = nPoints;

    for(size_t i=0; i<nPoints; i++)
    {
        sigmaX2 += (xs[i] * xs[i]);
        sigmaXY += (xs[i] * ys[i]);
        sigmaX += xs[i];
        sigmaY += ys[i];
        sigma1 ++;
    }

    float invDenominator = 1.f/(sigma1 * sigmaX2 - sigmaX * sigmaX);
    intercept = (sigmaX2 * sigmaY - sigmaX * sigmaXY) * invDenominator;
    slope = (sigmaXY - sigmaX * sigmaY) * invDenominator;
}

__device__ float SDL::computeTERPhiChiSquared(struct modules& modulesInGPU, struct hits& hitsInGPU, float& g, float& f, float& radius, unsigned int* outerObjectAnchorHits, uint16_t* outerObjectLowerModuleIndices)
{
    //Three cases
    float delta1[3], delta2[3], slopes[3], xs[3], ys[3];
    bool isFlat[3];
    computeSigmasForRegressionTCE(modulesInGPU, outerObjectLowerModuleIndices, delta1, delta2, slopes, isFlat, 3, true);

    for(size_t i = 0; i < 3; i++)
    {
        xs[i] = hitsInGPU.xs[outerObjectAnchorHits[i]];
        ys[i] = hitsInGPU.ys[outerObjectAnchorHits[i]];
    }
    float chiSquared = computeChiSquaredTCE(3, xs, ys, delta1, delta2, slopes, isFlat, g, f, radius);
    return chiSquared;
}


__device__ float SDL::computeT3T3RPhiChiSquared(struct modules& modulesInGPU, struct hits& hitsInGPU, int nPoints, unsigned int* anchorHitIndices, uint16_t* lowerModuleIndices, float& regressionRadius)
{
    float delta1[6], delta2[6], sigmas[6], slopes[6], xs[6], ys[6], g, f;
    bool isFlat[6];
    computeSigmasForRegressionTCE(modulesInGPU, lowerModuleIndices, delta1, delta2, slopes, isFlat, nPoints, true);

    for(size_t i = 0; i < nPoints; i++)
    {
        xs[i] = hitsInGPU.xs[anchorHitIndices[i]];
        ys[i] = hitsInGPU.ys[anchorHitIndices[i]];
    }
    float chiSquared;
    regressionRadius = computeRadiusUsingRegression(nPoints, xs, ys, delta1, delta2, slopes, isFlat, g, f, sigmas, chiSquared);
    return chiSquared;
}

__device__ float SDL::computeT3T3RZChiSquared(struct modules& modulesInGPU, struct hits& hitsInGPU, int nPoints, unsigned int* anchorHitIndices, uint16_t* lowerModuleIndices)
{
    float rts[6], zs[6];
    float slope = 0, intercept = 0, RMSE = 0, error, drdz, residual;

    int nPSPoints = 0;
    int n2SPoints = 0;
    for(size_t i = 0; i < nPoints; i++)
    {
        rts[i] = hitsInGPU.rts[anchorHitIndices[i]];
        zs[i] = hitsInGPU.zs[anchorHitIndices[i]];

        if(modulesInGPU.moduleType[lowerModuleIndices[i]] == SDL::PS)
        {
            nPSPoints++;
        }
        else
        {
            n2SPoints++;
        }
    }
    if(nPSPoints <= 1)
    {
        fitStraightLine(n2SPoints, &rts[nPSPoints], &zs[nPSPoints], slope, intercept);

    }
    else
    {
        fitStraightLine(nPSPoints, rts, zs, slope, intercept);
    }
    for(size_t i=0; i<nPoints; i++)
    {
        unsigned int lowerModuleIndex = lowerModuleIndices[i];

        const int moduleType = modulesInGPU.moduleType[lowerModuleIndex];
        const int moduleSide = modulesInGPU.sides[lowerModuleIndex];
        const int moduleSubdet = modulesInGPU.subdets[lowerModuleIndex];

        residual = (moduleSubdet == SDL::Barrel) ?  zs[i] - (slope * rts[i] + intercept) : rts[i] - (zs[i]/slope + intercept/slope);
        if(moduleType == 0)
        {
            error = 0.15;
        }
        else
        {
            error = 5;
        }
        //special dispensation to tilted PS modules!
        if(moduleType == 0 and moduleSubdet == SDL::Barrel and moduleSide != Center)
        {
            drdz = modulesInGPU.drdzs[lowerModuleIndex];
            error *= 1/sqrtf(1 + drdz * drdz);
        }
        RMSE += (residual * residual)/(error * error);
    }
    RMSE = sqrtf(RMSE/nPoints);
    return RMSE;
}

__global__ void SDL::createExtendedTracksInGPUv2(struct SDL::modules& modulesInGPU, struct SDL::objectRanges& rangesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::pixelTriplets& pixelTripletsInGPU, struct SDL::quintuplets& quintupletsInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU, struct SDL::trackExtensions& trackExtensionsInGPU)
{
    for(int tcIdx = blockIdx.z*blockDim.z+threadIdx.z; tcIdx < *(trackCandidatesInGPU.nTrackCandidates); tcIdx+= blockDim.z*gridDim.z){
    short tcType = trackCandidatesInGPU.trackCandidateType[tcIdx];
    uint16_t outerT3StartingModuleIndex;
    unsigned int outerT3Index;
    if(tcType == 8) continue;//return;
    for(int layerOverlap = 1+blockIdx.y*blockDim.y+threadIdx.y; layerOverlap < 3; layerOverlap+= blockDim.y*gridDim.y){
    //FIXME: Need to use staggering modules for the first outer T3 module itself!
    if(tcType == 7 or tcType == 4)
    {
        unsigned int outerT5Index = trackCandidatesInGPU.objectIndices[2 * tcIdx + 1];
        outerT3Index = quintupletsInGPU.tripletIndices[2 * outerT5Index];
        outerT3StartingModuleIndex = quintupletsInGPU.lowerModuleIndices[5 * outerT5Index + 5 - layerOverlap];
    }
    else if(tcType == 5) //pT3
    {
        unsigned int pT3Index = trackCandidatesInGPU.objectIndices[2 * tcIdx];
        outerT3Index = pixelTripletsInGPU.tripletIndices[pT3Index];
        outerT3StartingModuleIndex = tripletsInGPU.lowerModuleIndices[3 * outerT3Index + 3 - layerOverlap];
    }


    //if(t3ArrayIdx >= tripletsInGPU.nTriplets[outerT3StartingModuleIndex]) return;
    for(int t3ArrayIdx = blockIdx.x*blockDim.x+threadIdx.x; t3ArrayIdx < tripletsInGPU.nTriplets[outerT3StartingModuleIndex]; t3ArrayIdx+= blockDim.x*gridDim.x){
    unsigned int t3Idx =  rangesInGPU.tripletModuleIndices[outerT3StartingModuleIndex] + t3ArrayIdx;
    short constituentTCType[3];
    unsigned int constituentTCIndex[3];
    unsigned int nLayerOverlaps[2], nHitOverlaps[2];
    float rzChiSquared, rPhiChiSquared, regressionRadius, innerRadius, outerRadius;

    bool success = runTrackExtensionDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, tripletsInGPU, quintupletsInGPU, pixelTripletsInGPU, pixelQuintupletsInGPU, trackCandidatesInGPU, tcIdx, t3Idx, tcType, 3, outerT3Index, layerOverlap, constituentTCType, constituentTCIndex, nLayerOverlaps, nHitOverlaps, rPhiChiSquared, rzChiSquared, regressionRadius, innerRadius, outerRadius);
    if(success)
    {
        unsigned int totOccupancyTrackExtensions = atomicAdd(&trackExtensionsInGPU.totOccupancyTrackExtensions[tcIdx], 1);
        if(totOccupancyTrackExtensions >= N_MAX_TRACK_EXTENSIONS_PER_TC)
        {
#ifdef Warnings
            printf("Track extensions overflow for TC index = %d\n", tcIdx);
#endif
        }
        else
        {
            unsigned int trackExtensionArrayIndex = atomicAdd(&trackExtensionsInGPU.nTrackExtensions[tcIdx], 1);
            unsigned int trackExtensionIndex = tcIdx * N_MAX_TRACK_EXTENSIONS_PER_TC + trackExtensionArrayIndex;
#ifdef CUT_VALUE_DEBUG
            addTrackExtensionToMemory(trackExtensionsInGPU, constituentTCType, constituentTCIndex, nLayerOverlaps, nHitOverlaps, rPhiChiSquared, rzChiSquared, regressionRadius, innerRadius, outerRadius, trackExtensionIndex);
#else
            addTrackExtensionToMemory(trackExtensionsInGPU, constituentTCType, constituentTCIndex, nLayerOverlaps, nHitOverlaps, rPhiChiSquared, rzChiSquared, regressionRadius, trackExtensionIndex);
#endif
            trackCandidatesInGPU.partOfExtension[tcIdx] = true;
            tripletsInGPU.partOfExtension[t3Idx] = true;
        }
    }}}}
}

__device__ bool SDL::runExtensionDefaultAlgoBBBB(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& innerOuterLowerModuleIndex, uint16_t& outerInnerLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex,
        unsigned int& fourthMDIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& dPhi, float& betaIn, float&
        betaOut, float& pt_beta, float& zLo, float& zHi, float& zLoPointed, float& zHiPointed, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut)
{
    bool pass = true;

    bool isPS_InLo = (modulesInGPU.moduleType[innerInnerLowerModuleIndex] == SDL::PS);
    bool isPS_OutLo = (modulesInGPU.moduleType[outerInnerLowerModuleIndex] == SDL::PS);

    float rt_InLo = mdsInGPU.anchorRt[firstMDIndex];
    float rt_InOut = mdsInGPU.anchorRt[secondMDIndex];
    float rt_OutLo = mdsInGPU.anchorRt[thirdMDIndex];

    float z_InLo = mdsInGPU.anchorZ[firstMDIndex];
    float z_InOut = mdsInGPU.anchorZ[secondMDIndex];
    float z_OutLo = mdsInGPU.anchorZ[thirdMDIndex];

    float alpha1GeV_OutLo = asinf(fminf(rt_OutLo * SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax));

    float rtRatio_OutLoInLo = rt_OutLo / rt_InLo; // Outer segment beginning rt divided by inner segment beginning rt;
    float dzDrtScale = tanf(alpha1GeV_OutLo) / alpha1GeV_OutLo; // The track can bend in r-z plane slightly
    float zpitch_InLo = (isPS_InLo ? SDL::pixelPSZpitch : SDL::strip2SZpitch);
    float zpitch_OutLo = (isPS_OutLo ? SDL::pixelPSZpitch : SDL::strip2SZpitch);

    zHi = z_InLo + (z_InLo + SDL::deltaZLum) * (rtRatio_OutLoInLo - 1.f) * (z_InLo < 0.f ? 1.f : dzDrtScale) + (zpitch_InLo + zpitch_OutLo);
    zLo = z_InLo + (z_InLo - SDL::deltaZLum) * (rtRatio_OutLoInLo - 1.f) * (z_InLo > 0.f ? 1.f : dzDrtScale) - (zpitch_InLo + zpitch_OutLo);


    //Cut 1 - z compatibility
    zOut = z_OutLo;
    rtOut = rt_OutLo;
    pass = pass and ((z_OutLo >= zLo) & (z_OutLo <= zHi));
    if(not pass) return pass;

    float drt_OutLo_InLo = (rt_OutLo - rt_InLo);
    float r3_InLo = sqrtf(z_InLo * z_InLo + rt_InLo * rt_InLo);
    float drt_InSeg = rt_InOut - rt_InLo;
    float dz_InSeg = z_InOut - z_InLo;
    float dr3_InSeg = sqrtf(rt_InOut * rt_InOut + z_InOut * z_InOut) - sqrtf(rt_InLo * rt_InLo + z_InLo * z_InLo);

    float coshEta = dr3_InSeg/drt_InSeg;
    float dzErr = (zpitch_InLo + zpitch_OutLo) * (zpitch_InLo + zpitch_OutLo) * 2.f;

    float sdlThetaMulsF = 0.015f * sqrtf(0.1f + 0.2f * (rt_OutLo - rt_InLo) / 50.f) * sqrtf(r3_InLo / rt_InLo);
    float sdlMuls = sdlThetaMulsF * 3.f / SDL::ptCut * 4.f; // will need a better guess than x4?
    dzErr += sdlMuls * sdlMuls * drt_OutLo_InLo * drt_OutLo_InLo / 3.f * coshEta * coshEta; //sloppy
    dzErr = sqrtf(dzErr);

    // Constructing upper and lower bound
    const float dzMean = dz_InSeg / drt_InSeg * drt_OutLo_InLo;
    const float zWindow = dzErr / drt_InSeg * drt_OutLo_InLo + (zpitch_InLo + zpitch_OutLo); //FIXME for SDL::ptCut lower than ~0.8 need to add curv path correction
    zLoPointed = z_InLo + dzMean * (z_InLo > 0.f ? 1.f : dzDrtScale) - zWindow;
    zHiPointed = z_InLo + dzMean * (z_InLo < 0.f ? 1.f : dzDrtScale) + zWindow;

    // Cut #2: Pointed Z (Inner segment two MD points to outer segment inner MD)
    pass =  pass and ((z_OutLo >= zLoPointed) & (z_OutLo <= zHiPointed));
    if(not pass) return pass;

    float sdlPVoff = 0.1f/rt_OutLo;
    sdlCut = alpha1GeV_OutLo + sqrtf(sdlMuls * sdlMuls + sdlPVoff * sdlPVoff);

    deltaPhiPos = SDL::deltaPhi(mdsInGPU.anchorX[secondMDIndex], mdsInGPU.anchorY[secondMDIndex], mdsInGPU.anchorZ[secondMDIndex], mdsInGPU.anchorX[fourthMDIndex], mdsInGPU.anchorY[fourthMDIndex], mdsInGPU.anchorZ[fourthMDIndex]);
    // Cut #3: FIXME:deltaPhiPos can be tighter
    pass = pass and (fabsf(deltaPhiPos) <= sdlCut);
    if(not pass) return pass;

    float midPointX = 0.5f*(mdsInGPU.anchorX[firstMDIndex] + mdsInGPU.anchorX[thirdMDIndex]);
    float midPointY = 0.5f* (mdsInGPU.anchorY[firstMDIndex] + mdsInGPU.anchorY[thirdMDIndex]);
    float midPointZ = 0.5f*(mdsInGPU.anchorZ[firstMDIndex] + mdsInGPU.anchorZ[thirdMDIndex]);
    float diffX = mdsInGPU.anchorX[thirdMDIndex] - mdsInGPU.anchorX[firstMDIndex];
    float diffY = mdsInGPU.anchorY[thirdMDIndex] - mdsInGPU.anchorY[firstMDIndex];
    float diffZ = mdsInGPU.anchorZ[thirdMDIndex] - mdsInGPU.anchorZ[firstMDIndex];

    dPhi = SDL::deltaPhi(midPointX, midPointY, midPointZ, diffX, diffY, diffZ);

    // Cut #4: deltaPhiChange
    pass = pass and (fabsf(dPhi) <= sdlCut);
    //lots of array accesses below. Cut here!
    if(not pass) return pass;

    // First obtaining the raw betaIn and betaOut values without any correction and just purely based on the mini-doublet hit positions

    float alpha_InLo  = __H2F(segmentsInGPU.dPhiChanges[innerSegmentIndex]);
    float alpha_OutLo = __H2F(segmentsInGPU.dPhiChanges[outerSegmentIndex]);

    bool isEC_lastLayer = modulesInGPU.subdets[outerOuterLowerModuleIndex] == SDL::Endcap and modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::TwoS;

    float alpha_OutUp,alpha_OutUp_highEdge,alpha_OutUp_lowEdge;

    alpha_OutUp = SDL::deltaPhi(mdsInGPU.anchorX[fourthMDIndex], mdsInGPU.anchorY[fourthMDIndex], mdsInGPU.anchorZ[fourthMDIndex], mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex], mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex], mdsInGPU.anchorZ[fourthMDIndex] - mdsInGPU.anchorZ[thirdMDIndex]);

    alpha_OutUp_highEdge = alpha_OutUp;
    alpha_OutUp_lowEdge = alpha_OutUp;

    float tl_axis_x = mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[firstMDIndex];
    float tl_axis_y = mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[firstMDIndex];
    float tl_axis_z = mdsInGPU.anchorZ[fourthMDIndex] - mdsInGPU.anchorZ[firstMDIndex];
    float tl_axis_highEdge_x = tl_axis_x;
    float tl_axis_highEdge_y = tl_axis_y;
    float tl_axis_lowEdge_x = tl_axis_x;
    float tl_axis_lowEdge_y = tl_axis_y;

    betaIn = alpha_InLo - SDL::deltaPhi(mdsInGPU.anchorX[firstMDIndex], mdsInGPU.anchorY[firstMDIndex], mdsInGPU.anchorZ[firstMDIndex], tl_axis_x, tl_axis_y, tl_axis_z);

    float betaInRHmin = betaIn;
    float betaInRHmax = betaIn;
    betaOut = -alpha_OutUp + SDL::deltaPhi(mdsInGPU.anchorX[fourthMDIndex], mdsInGPU.anchorY[fourthMDIndex], mdsInGPU.anchorZ[fourthMDIndex], tl_axis_x, tl_axis_y, tl_axis_z);

    float betaOutRHmin = betaOut;
    float betaOutRHmax = betaOut;

    if(isEC_lastLayer)
    {
        alpha_OutUp_highEdge = SDL::deltaPhi(mdsInGPU.anchorHighEdgeX[fourthMDIndex], mdsInGPU.anchorHighEdgeY[fourthMDIndex], mdsInGPU.anchorZ[fourthMDIndex], mdsInGPU.anchorHighEdgeX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex], mdsInGPU.anchorHighEdgeY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex], mdsInGPU.anchorZ[fourthMDIndex] - mdsInGPU.anchorZ[thirdMDIndex]);
        alpha_OutUp_lowEdge = SDL::deltaPhi(mdsInGPU.anchorLowEdgeX[fourthMDIndex], mdsInGPU.anchorLowEdgeY[fourthMDIndex], mdsInGPU.anchorZ[fourthMDIndex], mdsInGPU.anchorLowEdgeX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex], mdsInGPU.anchorLowEdgeY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex], mdsInGPU.anchorZ[fourthMDIndex] - mdsInGPU.anchorZ[thirdMDIndex]);

        tl_axis_highEdge_x = mdsInGPU.anchorHighEdgeX[fourthMDIndex] - mdsInGPU.anchorX[firstMDIndex];
        tl_axis_highEdge_y = mdsInGPU.anchorHighEdgeY[fourthMDIndex] - mdsInGPU.anchorY[firstMDIndex];
        tl_axis_lowEdge_x = mdsInGPU.anchorLowEdgeX[fourthMDIndex] - mdsInGPU.anchorX[firstMDIndex];
        tl_axis_lowEdge_y = mdsInGPU.anchorLowEdgeY[fourthMDIndex] - mdsInGPU.anchorY[firstMDIndex];


        betaOutRHmin = -alpha_OutUp_highEdge + SDL::deltaPhi(mdsInGPU.anchorHighEdgeX[fourthMDIndex], mdsInGPU.anchorHighEdgeY[fourthMDIndex], mdsInGPU.anchorZ[fourthMDIndex], tl_axis_highEdge_x, tl_axis_highEdge_y, tl_axis_z);
        betaOutRHmax = -alpha_OutUp_lowEdge + SDL::deltaPhi(mdsInGPU.anchorLowEdgeX[fourthMDIndex], mdsInGPU.anchorLowEdgeY[fourthMDIndex], mdsInGPU.anchorZ[fourthMDIndex], tl_axis_lowEdge_x, tl_axis_lowEdge_y, tl_axis_z);
    }

    //beta computation
    float drt_tl_axis = sqrtf(tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);
    float drt_tl_lowEdge = sqrtf(tl_axis_lowEdge_x * tl_axis_lowEdge_x + tl_axis_lowEdge_y * tl_axis_lowEdge_y);
    float drt_tl_highEdge = sqrtf(tl_axis_highEdge_x * tl_axis_highEdge_x + tl_axis_highEdge_y * tl_axis_highEdge_y);

    float corrF = 1.f;
    //innerOuterAnchor - innerInnerAnchor
    const float rt_InSeg = sqrtf((mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) * (mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) + (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]) * (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]));
    betaInCut = asinf(fminf((-rt_InSeg * corrF + drt_tl_axis) * SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax)) + (0.02f / drt_InSeg);

    //Cut #5: first beta cut
    pass = pass and (fabsf(betaInRHmin) < betaInCut);
    if(not pass) return pass;

    float betaAv = 0.5f * (betaIn + betaOut);
    pt_beta = drt_tl_axis * SDL::k2Rinv1GeVf/sinf(betaAv);
    int lIn = 5;
    int lOut = isEC_lastLayer ? 11 : 5;
    float sdOut_dr = sqrtf((mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex]) * (mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex]) + (mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]) * (mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]));
    float sdOut_d = mdsInGPU.anchorRt[fourthMDIndex] - mdsInGPU.anchorRt[thirdMDIndex];

    const float diffDr = fabsf(rt_InSeg - sdOut_dr) / fabsf(rt_InSeg + sdOut_dr);

    SDL::runDeltaBetaIterationsTCE(betaIn, betaOut, betaAv, pt_beta, rt_InSeg, sdOut_dr, drt_tl_axis, lIn);

    const float betaInMMSF = (fabsf(betaInRHmin + betaInRHmax) > 0) ? (2.f * betaIn / fabsf(betaInRHmin + betaInRHmax)) : 0.f; //mean value of min,max is the old betaIn
    const float betaOutMMSF = (fabsf(betaOutRHmin + betaOutRHmax) > 0) ? (2.f * betaOut / fabsf(betaOutRHmin + betaOutRHmax)) : 0.f;
    betaInRHmin *= betaInMMSF;
    betaInRHmax *= betaInMMSF;
    betaOutRHmin *= betaOutMMSF;
    betaOutRHmax *= betaOutMMSF;

    const float dBetaMuls = sdlThetaMulsF * 4.f / fminf(fabsf(pt_beta), SDL::pt_betaMax); //need to confirm the range-out value of 7 GeV


    const float alphaInAbsReg = fmaxf(fabsf(alpha_InLo), asinf(fminf(rt_InLo * SDL::k2Rinv1GeVf / 3.0f, SDL::sinAlphaMax)));
    const float alphaOutAbsReg = fmaxf(fabs(alpha_OutLo), asinf(fminf(rt_OutLo * SDL::k2Rinv1GeVf / 3.0f, SDL::sinAlphaMax)));
    const float dBetaInLum = lIn < 11 ? 0.0f : fabsf(alphaInAbsReg*SDL::deltaZLum / z_InLo);
    const float dBetaOutLum = lOut < 11 ? 0.0f : fabsf(alphaOutAbsReg*SDL::deltaZLum / z_OutLo);
    const float dBetaLum2 = (dBetaInLum + dBetaOutLum) * (dBetaInLum + dBetaOutLum);
    const float sinDPhi = sinf(dPhi);

    const float dBetaRIn2 = 0; // TODO-RH
    // const float dBetaROut2 = 0; // TODO-RH
    float dBetaROut = 0;
    if(isEC_lastLayer)
    {
        dBetaROut = (sqrtf(mdsInGPU.anchorHighEdgeX[fourthMDIndex] * mdsInGPU.anchorHighEdgeX[fourthMDIndex] + mdsInGPU.anchorHighEdgeY[fourthMDIndex] * mdsInGPU.anchorHighEdgeY[fourthMDIndex]) - sqrtf(mdsInGPU.anchorLowEdgeX[fourthMDIndex] * mdsInGPU.anchorLowEdgeX[fourthMDIndex] + mdsInGPU.anchorLowEdgeY[fourthMDIndex] * mdsInGPU.anchorLowEdgeY[fourthMDIndex])) * sinDPhi / drt_tl_axis;
    }

    const float dBetaROut2 =  dBetaROut * dBetaROut;

    betaOutCut = asinf(fminf(drt_tl_axis*SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax)) //FIXME: need faster version
        + (0.02f / sdOut_d) + sqrtf(dBetaLum2 + dBetaMuls*dBetaMuls);

    //Cut #6: The real beta cut
    pass =  pass and ((fabsf(betaOut) < betaOutCut));
    if(not pass) return pass;

    float pt_betaIn = drt_tl_axis * SDL::k2Rinv1GeVf/sinf(betaIn);
    float pt_betaOut = drt_tl_axis * SDL::k2Rinv1GeVf / sinf(betaOut);
    float dBetaRes = 0.02f/fminf(sdOut_d,drt_InSeg);
    float dBetaCut2 = (dBetaRes*dBetaRes * 2.0f + dBetaMuls * dBetaMuls + dBetaLum2 + dBetaRIn2 + dBetaROut2
            + 0.25f * (fabsf(betaInRHmin - betaInRHmax) + fabsf(betaOutRHmin - betaOutRHmax)) * (fabsf(betaInRHmin - betaInRHmax) + fabsf(betaOutRHmin - betaOutRHmax)));

    float dBeta = betaIn - betaOut;
    deltaBetaCut = sqrtf(dBetaCut2);
    pass = pass and (dBeta * dBeta <= dBetaCut2);

    return pass;
}

__device__ bool SDL::runExtensionDefaultAlgoBBEE(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& innerOuterLowerModuleIndex, uint16_t& outerInnerLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex,
        unsigned int& fourthMDIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& dPhi, float& betaIn, float&
        betaOut, float& pt_beta, float& zLo, float& rtLo, float& rtHi, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ)
{
    bool pass = true;
    bool isPS_InLo = (modulesInGPU.moduleType[innerInnerLowerModuleIndex] == SDL::PS);
    bool isPS_OutLo = (modulesInGPU.moduleType[outerInnerLowerModuleIndex] == SDL::PS);

    float rt_InLo = mdsInGPU.anchorRt[firstMDIndex];
    float rt_InOut = mdsInGPU.anchorRt[secondMDIndex];
    float rt_OutLo = mdsInGPU.anchorRt[thirdMDIndex];

    float z_InLo = mdsInGPU.anchorZ[firstMDIndex];
    float z_InOut = mdsInGPU.anchorZ[secondMDIndex];
    float z_OutLo = mdsInGPU.anchorZ[thirdMDIndex];

    float alpha1GeV_OutLo = asinf(fminf(rt_OutLo * SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax));

    float rtRatio_OutLoInLo = rt_OutLo / rt_InLo; // Outer segment beginning rt divided by inner segment beginning rt;
    float dzDrtScale = tanf(alpha1GeV_OutLo) / alpha1GeV_OutLo; // The track can bend in r-z plane slightly
    float zpitch_InLo = (isPS_InLo ? SDL::pixelPSZpitch : SDL::strip2SZpitch);
    float zpitch_OutLo = (isPS_OutLo ? SDL::pixelPSZpitch : SDL::strip2SZpitch);
    float zGeom = zpitch_InLo + zpitch_OutLo;

    zLo = z_InLo + (z_InLo - SDL::deltaZLum) * (rtRatio_OutLoInLo - 1.f) * (z_InLo > 0.f ? 1.f : dzDrtScale) - zGeom;

    // Cut #0: Preliminary (Only here in endcap case)
    pass = pass and (z_InLo * z_OutLo > 0);
    if(not pass) return pass;

    float dLum = copysignf(SDL::deltaZLum, z_InLo);
    bool isOutSgInnerMDPS = modulesInGPU.moduleType[outerInnerLowerModuleIndex] == SDL::PS;
    float rtGeom1 = isOutSgInnerMDPS ? SDL::pixelPSZpitch : SDL::strip2SZpitch;
    float zGeom1 = copysignf(zGeom,z_InLo);
    rtLo = rt_InLo * (1.f + (z_OutLo - z_InLo - zGeom1) / (z_InLo + zGeom1 + dLum) / dzDrtScale) - rtGeom1; //slope correction only on the lower end
    zOut = z_OutLo;
    rtOut = rt_OutLo;

    //Cut #1: rt condition
    pass =  pass and (rtOut >= rtLo);
    if(not pass) return pass;

    float zInForHi = z_InLo - zGeom1 - dLum;
    if(zInForHi * z_InLo < 0)
    {
        zInForHi = copysignf(0.1f,z_InLo);
    }
    rtHi = rt_InLo * (1.f + (z_OutLo - z_InLo + zGeom1) / zInForHi) + rtGeom1;

    //Cut #2: rt condition
    pass =  pass and ((rt_OutLo >= rtLo) & (rt_OutLo <= rtHi));
    if(not pass) return pass;

    float rIn = sqrtf(z_InLo * z_InLo + rt_InLo * rt_InLo);
    const float drtSDIn = rt_InOut - rt_InLo;
    const float dzSDIn = z_InOut - z_InLo;
    const float dr3SDIn = sqrtf(rt_InOut * rt_InOut + z_InOut * z_InOut) - sqrtf(rt_InLo * rt_InLo + z_InLo * z_InLo);

    const float coshEta = dr3SDIn / drtSDIn; //direction estimate
    const float dzOutInAbs = fabsf(z_OutLo - z_InLo);
    const float multDzDr = dzOutInAbs * coshEta / (coshEta * coshEta - 1.f);
    const float zGeom1_another = SDL::pixelPSZpitch; //What's this?
    kZ = (z_OutLo - z_InLo) / dzSDIn;
    float drtErr = zGeom1_another * zGeom1_another * drtSDIn * drtSDIn / dzSDIn / dzSDIn * (1.f - 2.f * kZ + 2.f * kZ * kZ); //Notes:122316
    const float sdlThetaMulsF = 0.015f * sqrtf(0.1f + 0.2f * (rt_OutLo - rt_InLo) / 50.f) * sqrtf(rIn / rt_InLo);
    const float sdlMuls = sdlThetaMulsF * 3.f / SDL::ptCut * 4.f; //will need a better guess than x4?
    drtErr += sdlMuls * sdlMuls * multDzDr * multDzDr / 3.f * coshEta * coshEta; //sloppy: relative muls is 1/3 of total muls
    drtErr = sqrtf(drtErr);
    const float drtMean = drtSDIn * dzOutInAbs / fabsf(dzSDIn); //
    const float rtWindow = drtErr + rtGeom1;
    const float rtLo_another = rt_InLo + drtMean / dzDrtScale - rtWindow;
    const float rtHi_another = rt_InLo + drtMean + rtWindow;

    //Cut #3: rt-z pointed
    pass =  pass and ((kZ >= 0) & (rtOut >= rtLo) & (rtOut <= rtHi));
    if(not pass) return pass;

    const float sdlPVoff = 0.1f / rt_OutLo;
    sdlCut = alpha1GeV_OutLo + sqrtf(sdlMuls * sdlMuls + sdlPVoff*sdlPVoff);


    deltaPhiPos = SDL::deltaPhi(mdsInGPU.anchorX[secondMDIndex], mdsInGPU.anchorY[secondMDIndex], mdsInGPU.anchorZ[secondMDIndex], mdsInGPU.anchorX[fourthMDIndex], mdsInGPU.anchorY[fourthMDIndex], mdsInGPU.anchorZ[fourthMDIndex]);


    //Cut #4: deltaPhiPos can be tighter
    pass =  pass and (fabsf(deltaPhiPos) <= sdlCut);
    if(not pass) return pass;

    float midPointX = 0.5f*(mdsInGPU.anchorX[firstMDIndex] + mdsInGPU.anchorX[thirdMDIndex]);
    float midPointY = 0.5f* (mdsInGPU.anchorY[firstMDIndex] + mdsInGPU.anchorY[thirdMDIndex]);
    float midPointZ = 0.5f*(mdsInGPU.anchorZ[firstMDIndex] + mdsInGPU.anchorZ[thirdMDIndex]);
    float diffX = mdsInGPU.anchorX[thirdMDIndex] - mdsInGPU.anchorX[firstMDIndex];
    float diffY = mdsInGPU.anchorY[thirdMDIndex] - mdsInGPU.anchorY[firstMDIndex];
    float diffZ = mdsInGPU.anchorZ[thirdMDIndex] - mdsInGPU.anchorZ[firstMDIndex];

    dPhi = SDL::deltaPhi(midPointX, midPointY, midPointZ, diffX, diffY, diffZ);
    // Cut #5: deltaPhiChange
    pass =  pass and (fabsf(dPhi) <= sdlCut);
    if(not pass) return pass;

    float sdIn_alpha     = __H2F(segmentsInGPU.dPhiChanges[innerSegmentIndex]);
    float sdIn_alpha_min = __H2F(segmentsInGPU.dPhiChangeMins[innerSegmentIndex]);
    float sdIn_alpha_max = __H2F(segmentsInGPU.dPhiChangeMaxs[innerSegmentIndex]);
    float sdOut_alpha = sdIn_alpha; //weird

    float sdOut_alphaOut = SDL::deltaPhi(mdsInGPU.anchorX[fourthMDIndex], mdsInGPU.anchorY[fourthMDIndex], mdsInGPU.anchorZ[fourthMDIndex], mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex], mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex], mdsInGPU.anchorZ[fourthMDIndex] - mdsInGPU.anchorZ[thirdMDIndex]);

    float sdOut_alphaOut_min = SDL::phi_mpi_pi(__H2F(segmentsInGPU.dPhiChangeMins[outerSegmentIndex]) - __H2F(segmentsInGPU.dPhiMins[outerSegmentIndex]));
    float sdOut_alphaOut_max = SDL::phi_mpi_pi(__H2F(segmentsInGPU.dPhiChangeMaxs[outerSegmentIndex]) - __H2F(segmentsInGPU.dPhiMaxs[outerSegmentIndex]));

    float tl_axis_x = mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[firstMDIndex];
    float tl_axis_y = mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[firstMDIndex];
    float tl_axis_z = mdsInGPU.anchorZ[fourthMDIndex] - mdsInGPU.anchorZ[firstMDIndex];

    betaIn = sdIn_alpha - SDL::deltaPhi(mdsInGPU.anchorX[firstMDIndex], mdsInGPU.anchorY[firstMDIndex], mdsInGPU.anchorZ[firstMDIndex], tl_axis_x, tl_axis_y, tl_axis_z);

    float betaInRHmin = betaIn;
    float betaInRHmax = betaIn;
    betaOut = -sdOut_alphaOut + SDL::deltaPhi(mdsInGPU.anchorX[fourthMDIndex], mdsInGPU.anchorY[fourthMDIndex], mdsInGPU.anchorZ[fourthMDIndex], tl_axis_x, tl_axis_y, tl_axis_z);

    float betaOutRHmin = betaOut;
    float betaOutRHmax = betaOut;

    bool isEC_secondLayer = (modulesInGPU.subdets[innerOuterLowerModuleIndex] == SDL::Endcap) and (modulesInGPU.moduleType[innerOuterLowerModuleIndex] == SDL::TwoS);

    if(isEC_secondLayer)
    {
        betaInRHmin = betaIn - sdIn_alpha_min + sdIn_alpha;
        betaInRHmax = betaIn - sdIn_alpha_max + sdIn_alpha;
    }

    betaOutRHmin = betaOut - sdOut_alphaOut_min + sdOut_alphaOut;
    betaOutRHmax = betaOut - sdOut_alphaOut_max + sdOut_alphaOut;

    float swapTemp;
    if(fabsf(betaOutRHmin) > fabsf(betaOutRHmax))
    {
        swapTemp = betaOutRHmin;
        betaOutRHmin = betaOutRHmax;
        betaOutRHmax = swapTemp;
    }

    if(fabsf(betaInRHmin) > fabsf(betaInRHmax))
    {
        swapTemp = betaInRHmin;
        betaInRHmin = betaInRHmax;
        betaInRHmax = swapTemp;
    }

    float sdIn_dr = sqrtf((mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) * (mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) + (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]) * (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]));
    float sdIn_d = rt_InOut - rt_InLo;

    float dr = sqrtf(tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);
    const float corrF = 1.f;
    betaInCut = asinf(fminf((-sdIn_dr * corrF + dr) * SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax)) + (0.02f / sdIn_d);

    //Cut #6: first beta cut
    pass =  pass and (fabsf(betaInRHmin) < betaInCut);
    if(not pass) return pass;

    float betaAv = 0.5f * (betaIn + betaOut);
    pt_beta = dr * SDL::k2Rinv1GeVf / sinf(betaAv);

    float lIn = 5;
    float lOut = 11;

    float sdOut_dr = sqrtf((mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex]) * (mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex]) + (mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]) * (mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]));
    float sdOut_d = mdsInGPU.anchorRt[fourthMDIndex] - mdsInGPU.anchorRt[thirdMDIndex];

    SDL::runDeltaBetaIterationsTCE(betaIn, betaOut, betaAv, pt_beta, sdIn_dr, sdOut_dr, dr, lIn);

     const float betaInMMSF = (fabsf(betaInRHmin + betaInRHmax) > 0) ? (2.f * betaIn / fabsf(betaInRHmin + betaInRHmax)) : 0.; //mean value of min,max is the old betaIn
    const float betaOutMMSF = (fabsf(betaOutRHmin + betaOutRHmax) > 0) ? (2.f * betaOut / fabsf(betaOutRHmin + betaOutRHmax)) : 0.;
    betaInRHmin *= betaInMMSF;
    betaInRHmax *= betaInMMSF;
    betaOutRHmin *= betaOutMMSF;
    betaOutRHmax *= betaOutMMSF;

    const float dBetaMuls = sdlThetaMulsF * 4.f / fminf(fabsf(pt_beta), SDL::pt_betaMax); //need to confirm the range-out value of 7 GeV

    const float alphaInAbsReg = fmaxf(fabsf(sdIn_alpha), asinf(fminf(rt_InLo * SDL::k2Rinv1GeVf / 3.0f, SDL::sinAlphaMax)));
    const float alphaOutAbsReg = fmaxf(fabsf(sdOut_alpha), asinf(fminf(rt_OutLo * SDL::k2Rinv1GeVf / 3.0f, SDL::sinAlphaMax)));
    const float dBetaInLum = lIn < 11 ? 0.0f : fabsf(alphaInAbsReg*SDL::deltaZLum / z_InLo);
    const float dBetaOutLum = lOut < 11 ? 0.0f : fabsf(alphaOutAbsReg*SDL::deltaZLum / z_OutLo);
    const float dBetaLum2 = (dBetaInLum + dBetaOutLum) * (dBetaInLum + dBetaOutLum);
    const float sinDPhi = sinf(dPhi);

    const float dBetaRIn2 = 0; // TODO-RH
    // const float dBetaROut2 = 0; // TODO-RH
    float dBetaROut = 0;
    if(modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::TwoS)
    {
        dBetaROut = (sqrtf(mdsInGPU.anchorHighEdgeX[fourthMDIndex] * mdsInGPU.anchorHighEdgeX[fourthMDIndex] + mdsInGPU.anchorHighEdgeY[fourthMDIndex] * mdsInGPU.anchorHighEdgeY[fourthMDIndex]) - sqrtf(mdsInGPU.anchorLowEdgeX[fourthMDIndex] * mdsInGPU.anchorLowEdgeX[fourthMDIndex] + mdsInGPU.anchorLowEdgeY[fourthMDIndex] * mdsInGPU.anchorLowEdgeY[fourthMDIndex])) * sinDPhi / dr;
    }

    const float dBetaROut2 = dBetaROut * dBetaROut;
    betaOutCut = asinf(fminf(dr*SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax)) //FIXME: need faster version
        + (0.02f / sdOut_d) + sqrtf(dBetaLum2 + dBetaMuls*dBetaMuls);

    //Cut #6: The real beta cut
    pass =  pass and (fabsf(betaOut) < betaOutCut);
    if(not pass) return pass;

    float pt_betaIn = dr * SDL::k2Rinv1GeVf/sinf(betaIn);
    float pt_betaOut = dr * SDL::k2Rinv1GeVf / sinf(betaOut);
    float dBetaRes = 0.02f/fminf(sdOut_d,sdIn_d);
    float dBetaCut2 = (dBetaRes*dBetaRes * 2.0f + dBetaMuls * dBetaMuls + dBetaLum2 + dBetaRIn2 + dBetaROut2
            + 0.25f * (fabsf(betaInRHmin - betaInRHmax) + fabsf(betaOutRHmin - betaOutRHmax)) * (fabsf(betaInRHmin - betaInRHmax) + fabsf(betaOutRHmin - betaOutRHmax)));
    float dBeta = betaIn - betaOut;
    deltaBetaCut = sqrtf(dBetaCut2);
    //Cut #7: Cut on dBet
    pass =  pass and (dBeta * dBeta <= dBetaCut2);

    return pass;
}

__device__ bool SDL::runExtensionDefaultAlgoEEEE(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& innerOuterLowerModuleIndex, uint16_t& outerInnerLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex,
        unsigned int& fourthMDIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& dPhi, float& betaIn, float&
        betaOut, float& pt_beta, float& zLo, float& rtLo, float& rtHi, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ)
{
    bool pass = true;

    bool isPS_InLo = (modulesInGPU.moduleType[innerInnerLowerModuleIndex] == SDL::PS);
    bool isPS_OutLo = (modulesInGPU.moduleType[outerInnerLowerModuleIndex] == SDL::PS);

    float rt_InLo = mdsInGPU.anchorRt[firstMDIndex];
    float rt_InOut = mdsInGPU.anchorRt[secondMDIndex];
    float rt_OutLo = mdsInGPU.anchorRt[thirdMDIndex];

    float z_InLo = mdsInGPU.anchorZ[firstMDIndex];
    float z_InOut = mdsInGPU.anchorZ[secondMDIndex];
    float z_OutLo = mdsInGPU.anchorZ[thirdMDIndex];

    float alpha1GeV_OutLo = asinf(fminf(rt_OutLo * SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax));

    float rtRatio_OutLoInLo = rt_OutLo / rt_InLo; // Outer segment beginning rt divided by inner segment beginning rt;
    float dzDrtScale = tanf(alpha1GeV_OutLo) / alpha1GeV_OutLo; // The track can bend in r-z plane slightly
    float zpitch_InLo = (isPS_InLo ? SDL::pixelPSZpitch : SDL::strip2SZpitch);
    float zpitch_OutLo = (isPS_OutLo ? SDL::pixelPSZpitch : SDL::strip2SZpitch);
    float zGeom = zpitch_InLo + zpitch_OutLo;

    zLo = z_InLo + (z_InLo - SDL::deltaZLum) * (rtRatio_OutLoInLo - 1.f) * (z_InLo > 0.f ? 1.f : dzDrtScale) - zGeom; //slope-correction only on outer end

    // Cut #0: Preliminary (Only here in endcap case)
    pass =  pass and ((z_InLo * z_OutLo) > 0);
    if(not pass) return pass;

    float dLum = copysignf(SDL::deltaZLum, z_InLo);
    bool isOutSgInnerMDPS = modulesInGPU.moduleType[outerInnerLowerModuleIndex] == SDL::PS;
    bool isInSgInnerMDPS = modulesInGPU.moduleType[innerInnerLowerModuleIndex] == SDL::PS;

    float rtGeom = (isInSgInnerMDPS and isOutSgInnerMDPS) ? 2.f * SDL::pixelPSZpitch : (isInSgInnerMDPS or isOutSgInnerMDPS) ? SDL::pixelPSZpitch + SDL::strip2SZpitch : 2.f * SDL::strip2SZpitch;

    float zGeom1 = copysignf(zGeom,z_InLo);
    float dz = z_OutLo - z_InLo;
    rtLo = rt_InLo * (1.f + dz / (z_InLo + dLum) / dzDrtScale) - rtGeom; //slope correction only on the lower end

    zOut = z_OutLo;
    rtOut = rt_OutLo;

    //Cut #1: rt condition

    rtHi = rt_InLo * (1.f + dz / (z_InLo - dLum)) + rtGeom;

    pass =  pass and ((rtOut >= rtLo) & (rtOut <= rtHi));
    if(not pass) return pass;

    bool isInSgOuterMDPS = modulesInGPU.moduleType[innerOuterLowerModuleIndex] == SDL::PS;

    float drOutIn = rtOut - rt_InLo;
    const float drtSDIn = rt_InOut - rt_InLo;
    const float dzSDIn = z_InOut - z_InLo;
    const float dr3SDIn = sqrtf(rt_InOut * rt_InOut + z_InOut * z_InOut) - sqrtf(rt_InLo * rt_InLo + z_InLo * z_InLo);
    float coshEta = dr3SDIn / drtSDIn; //direction estimate
    float dzOutInAbs =  fabsf(z_OutLo - z_InLo);
    float multDzDr = dzOutInAbs * coshEta / (coshEta * coshEta - 1.f);

    kZ = (z_OutLo - z_InLo) / dzSDIn;
    float sdlThetaMulsF = 0.015f * sqrtf(0.1f + 0.2f * (rt_OutLo - rt_InLo) / 50.f);

    float sdlMuls = sdlThetaMulsF * 3.f / SDL::ptCut * 4.f; //will need a better guess than x4?

    float drtErr = sqrtf(SDL::pixelPSZpitch * SDL::pixelPSZpitch * 2.f / (dzSDIn * dzSDIn) * (dzOutInAbs * dzOutInAbs) + sdlMuls * sdlMuls * multDzDr * multDzDr / 3.f * coshEta * coshEta);

    float drtMean = drtSDIn * dzOutInAbs/fabsf(dzSDIn);
    float rtWindow = drtErr + rtGeom;
    float rtLo_point = rt_InLo + drtMean / dzDrtScale - rtWindow;
    float rtHi_point = rt_InLo + drtMean + rtWindow;

    // Cut #3: rt-z pointed
    // https://github.com/slava77/cms-tkph2-ntuple/blob/superDoubletLinked-91X-noMock/doubletAnalysis.C#L3765

    if (isInSgInnerMDPS and isInSgOuterMDPS) // If both PS then we can point
    {
        pass =  pass and (kZ >= 0 and rtOut >= rtLo_point and rtOut <= rtHi_point);
        if(not pass) return pass;
    }

    float sdlPVoff = 0.1f/rtOut;
    sdlCut = alpha1GeV_OutLo + sqrtf(sdlMuls * sdlMuls + sdlPVoff * sdlPVoff);

    deltaPhiPos = SDL::deltaPhi(mdsInGPU.anchorX[secondMDIndex], mdsInGPU.anchorY[secondMDIndex], mdsInGPU.anchorZ[secondMDIndex], mdsInGPU.anchorX[fourthMDIndex], mdsInGPU.anchorY[fourthMDIndex], mdsInGPU.anchorZ[fourthMDIndex]);

    pass =  pass and (fabsf(deltaPhiPos) <= sdlCut);
    if(not pass) return pass;

    float midPointX = 0.5f*(mdsInGPU.anchorX[firstMDIndex] + mdsInGPU.anchorX[thirdMDIndex]);
    float midPointY = 0.5f* (mdsInGPU.anchorY[firstMDIndex] + mdsInGPU.anchorY[thirdMDIndex]);
    float midPointZ = 0.5f*(mdsInGPU.anchorZ[firstMDIndex] + mdsInGPU.anchorZ[thirdMDIndex]);
    float diffX = mdsInGPU.anchorX[thirdMDIndex] - mdsInGPU.anchorX[firstMDIndex];
    float diffY = mdsInGPU.anchorY[thirdMDIndex] - mdsInGPU.anchorY[firstMDIndex];
    float diffZ = mdsInGPU.anchorZ[thirdMDIndex] - mdsInGPU.anchorZ[firstMDIndex];

    dPhi = SDL::deltaPhi(midPointX, midPointY, midPointZ, diffX, diffY, diffZ);

    // Cut #5: deltaPhiChange
    pass =  pass and ((fabsf(dPhi) <= sdlCut));
    if(not pass) return pass;

    float sdIn_alpha = __H2F(segmentsInGPU.dPhiChanges[innerSegmentIndex]);
    float sdOut_alpha = sdIn_alpha; //weird
    float sdOut_dPhiPos = SDL::deltaPhi(mdsInGPU.anchorX[thirdMDIndex], mdsInGPU.anchorY[thirdMDIndex], mdsInGPU.anchorZ[thirdMDIndex], mdsInGPU.anchorX[fourthMDIndex], mdsInGPU.anchorY[fourthMDIndex], mdsInGPU.anchorZ[fourthMDIndex]);

    float sdOut_dPhiChange = __H2F(segmentsInGPU.dPhiChanges[outerSegmentIndex]);
    float sdOut_dPhiChange_min = __H2F(segmentsInGPU.dPhiChangeMins[outerSegmentIndex]);
    float sdOut_dPhiChange_max = __H2F(segmentsInGPU.dPhiChangeMaxs[outerSegmentIndex]);

    float sdOut_alphaOutRHmin = SDL::phi_mpi_pi(sdOut_dPhiChange_min - sdOut_dPhiPos);
    float sdOut_alphaOutRHmax = SDL::phi_mpi_pi(sdOut_dPhiChange_max - sdOut_dPhiPos);
    float sdOut_alphaOut = SDL::phi_mpi_pi(sdOut_dPhiChange - sdOut_dPhiPos);

    float tl_axis_x = mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[firstMDIndex];
    float tl_axis_y = mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[firstMDIndex];
    float tl_axis_z = mdsInGPU.anchorZ[fourthMDIndex] - mdsInGPU.anchorZ[firstMDIndex];

    betaIn = sdIn_alpha - SDL::deltaPhi(mdsInGPU.anchorX[firstMDIndex], mdsInGPU.anchorY[firstMDIndex], mdsInGPU.anchorZ[firstMDIndex], tl_axis_x, tl_axis_y, tl_axis_z);

    float sdIn_alphaRHmin = __H2F(segmentsInGPU.dPhiChangeMins[innerSegmentIndex]);
    float sdIn_alphaRHmax = __H2F(segmentsInGPU.dPhiChangeMaxs[innerSegmentIndex]);
    float betaInRHmin = betaIn + sdIn_alphaRHmin - sdIn_alpha;
    float betaInRHmax = betaIn + sdIn_alphaRHmax - sdIn_alpha;

    betaOut = -sdOut_alphaOut + SDL::deltaPhi(mdsInGPU.anchorX[fourthMDIndex], mdsInGPU.anchorY[fourthMDIndex], mdsInGPU.anchorZ[fourthMDIndex], tl_axis_x, tl_axis_y, tl_axis_z);

    float betaOutRHmin = betaOut - sdOut_alphaOutRHmin + sdOut_alphaOut;
    float betaOutRHmax = betaOut - sdOut_alphaOutRHmax + sdOut_alphaOut;

    float swapTemp;
    if(fabsf(betaOutRHmin) > fabsf(betaOutRHmax))
    {
        swapTemp = betaOutRHmin;
        betaOutRHmin = betaOutRHmax;
        betaOutRHmax = swapTemp;
    }

    if(fabsf(betaInRHmin) > fabsf(betaInRHmax))
    {
        swapTemp = betaInRHmin;
        betaInRHmin = betaInRHmax;
        betaInRHmax = swapTemp;
    }
    float sdIn_dr = sqrtf((mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) * (mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) + (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]) * (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]));
    float sdIn_d = rt_InOut - rt_InLo;

    float dr = sqrtf(tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);
    const float corrF = 1.f;
    betaInCut = asinf(fminf((-sdIn_dr * corrF + dr) * SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax)) + (0.02f / sdIn_d);

    //Cut #6: first beta cut
    pass =  pass and (fabsf(betaInRHmin) < betaInCut);
    if(not pass) return pass;

    float betaAv = 0.5f * (betaIn + betaOut);
    pt_beta = dr * SDL::k2Rinv1GeVf / sinf(betaAv);


    int lIn= 11; //endcap
    int lOut = 13; //endcap

    float sdOut_dr = sqrtf((mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex]) * (mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex]) + (mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]) * (mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]));
    float sdOut_d = mdsInGPU.anchorRt[fourthMDIndex] - mdsInGPU.anchorRt[thirdMDIndex];

    float diffDr = fabsf(sdIn_dr - sdOut_dr)/fabs(sdIn_dr + sdOut_dr);

    SDL::runDeltaBetaIterationsTCE(betaIn, betaOut, betaAv, pt_beta, sdIn_dr, sdOut_dr, dr, lIn);

     const float betaInMMSF = (fabsf(betaInRHmin + betaInRHmax) > 0) ? (2.f * betaIn / fabsf(betaInRHmin + betaInRHmax)) : 0.; //mean value of min,max is the old betaIn
    const float betaOutMMSF = (fabsf(betaOutRHmin + betaOutRHmax) > 0) ? (2.f * betaOut / fabsf(betaOutRHmin + betaOutRHmax)) : 0.;
    betaInRHmin *= betaInMMSF;
    betaInRHmax *= betaInMMSF;
    betaOutRHmin *= betaOutMMSF;
    betaOutRHmax *= betaOutMMSF;

    const float dBetaMuls = sdlThetaMulsF * 4.f / fminf(fabsf(pt_beta), SDL::pt_betaMax); //need to confirm the range-out value of 7 GeV

    const float alphaInAbsReg = fmaxf(fabsf(sdIn_alpha), asinf(fminf(rt_InLo * SDL::k2Rinv1GeVf / 3.0f, SDL::sinAlphaMax)));
    const float alphaOutAbsReg = fmaxf(fabsf(sdOut_alpha), asinf(fminf(rt_OutLo * SDL::k2Rinv1GeVf / 3.0f, SDL::sinAlphaMax)));
    const float dBetaInLum = lIn < 11 ? 0.0f : fabsf(alphaInAbsReg*SDL::deltaZLum / z_InLo);
    const float dBetaOutLum = lOut < 11 ? 0.0f : fabsf(alphaOutAbsReg*SDL::deltaZLum / z_OutLo);
    const float dBetaLum2 = (dBetaInLum + dBetaOutLum) * (dBetaInLum + dBetaOutLum);
    const float sinDPhi = sinf(dPhi);

    const float dBetaRIn2 = 0; // TODO-RH
    // const float dBetaROut2 = 0; // TODO-RH
    float dBetaROut2 = 0;//TODO-RH
    betaOutCut = asinf(fminf(dr*SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax)) //FIXME: need faster version
        + (0.02f / sdOut_d) + sqrtf(dBetaLum2 + dBetaMuls*dBetaMuls);

    //Cut #6: The real beta cut
    pass =  pass and (fabsf(betaOut) < betaOutCut);
    if(not pass) return pass;

    float pt_betaIn = dr * SDL::k2Rinv1GeVf/sinf(betaIn);
    float pt_betaOut = dr * SDL::k2Rinv1GeVf / sinf(betaOut);
    float dBetaRes = 0.02f/fminf(sdOut_d,sdIn_d);
    float dBetaCut2 = (dBetaRes*dBetaRes * 2.0f + dBetaMuls * dBetaMuls + dBetaLum2 + dBetaRIn2 + dBetaROut2
            + 0.25f * (fabsf(betaInRHmin - betaInRHmax) + fabsf(betaOutRHmin - betaOutRHmax)) * (fabsf(betaInRHmin - betaInRHmax) + fabsf(betaOutRHmin - betaOutRHmax)));
    float dBeta = betaIn - betaOut;
    //Cut #7: Cut on dBeta
    deltaBetaCut = sqrtf(dBetaCut2);

    pass =  pass and (dBeta * dBeta <= dBetaCut2);

    return pass;
}
__device__ bool SDL::runExtensionDefaultAlgo(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& innerOuterLowerModuleIndex, uint16_t& outerInnerLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex, unsigned int& fourthMDIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& deltaPhi, float& betaIn, float&
        betaOut, float& pt_beta, float& zLo, float& zHi, float& rtLo, float& rtHi, float& zLoPointed, float& zHiPointed, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ)
{

    bool pass = false;

    zLo = -999;
    zHi = -999;
    rtLo = -999;
    rtHi = -999;
    zLoPointed = -999;
    zHiPointed = -999;
    kZ = -999;
    betaInCut = -999;

    short innerInnerLowerModuleSubdet = modulesInGPU.subdets[innerInnerLowerModuleIndex];
    short innerOuterLowerModuleSubdet = modulesInGPU.subdets[innerOuterLowerModuleIndex];
    short outerInnerLowerModuleSubdet = modulesInGPU.subdets[outerInnerLowerModuleIndex];
    short outerOuterLowerModuleSubdet = modulesInGPU.subdets[outerOuterLowerModuleIndex];

    if(innerInnerLowerModuleSubdet == SDL::Barrel
            and innerOuterLowerModuleSubdet == SDL::Barrel
            and outerInnerLowerModuleSubdet == SDL::Barrel
            and outerOuterLowerModuleSubdet == SDL::Barrel)
    {
        return runExtensionDefaultAlgoBBBB(modulesInGPU,mdsInGPU,segmentsInGPU,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,innerSegmentIndex,outerSegmentIndex,firstMDIndex, secondMDIndex, thirdMDIndex, fourthMDIndex, zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta, zLo, zHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut);
    }

    else if(innerInnerLowerModuleSubdet == SDL::Barrel
            and innerOuterLowerModuleSubdet == SDL::Barrel
            and outerInnerLowerModuleSubdet == SDL::Endcap
            and outerOuterLowerModuleSubdet == SDL::Endcap)
    {
       return runExtensionDefaultAlgoBBEE(modulesInGPU,mdsInGPU,segmentsInGPU,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,innerSegmentIndex,outerSegmentIndex,firstMDIndex, secondMDIndex, thirdMDIndex, fourthMDIndex, zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta, zLo, rtLo, rtHi, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ);
    }


    else if(innerInnerLowerModuleSubdet == SDL::Barrel
            and innerOuterLowerModuleSubdet == SDL::Barrel
            and outerInnerLowerModuleSubdet == SDL::Barrel
            and outerOuterLowerModuleSubdet == SDL::Endcap)
    {
        return runExtensionDefaultAlgoBBBB(modulesInGPU,mdsInGPU,segmentsInGPU,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,innerSegmentIndex,outerSegmentIndex,firstMDIndex, secondMDIndex, thirdMDIndex, fourthMDIndex, zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta,zLo, zHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut);

    }

    else if(innerInnerLowerModuleSubdet == SDL::Barrel
            and innerOuterLowerModuleSubdet == SDL::Endcap
            and outerInnerLowerModuleSubdet == SDL::Endcap
            and outerOuterLowerModuleSubdet == SDL::Endcap)
    {
        return runExtensionDefaultAlgoBBEE(modulesInGPU,mdsInGPU,segmentsInGPU,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,innerSegmentIndex,outerSegmentIndex, firstMDIndex, secondMDIndex, thirdMDIndex, fourthMDIndex, zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta, zLo, rtLo, rtHi, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ);

    }

    else if(innerInnerLowerModuleSubdet == SDL::Endcap
            and innerOuterLowerModuleSubdet == SDL::Endcap
            and outerInnerLowerModuleSubdet == SDL::Endcap
            and outerOuterLowerModuleSubdet == SDL::Endcap)
    {
        return runExtensionDefaultAlgoEEEE(modulesInGPU,mdsInGPU,segmentsInGPU,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,innerSegmentIndex,outerSegmentIndex, firstMDIndex, secondMDIndex, thirdMDIndex, fourthMDIndex, zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta, zLo, rtLo, rtHi, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ);
    }

    return pass;
}
__device__ void SDL::runDeltaBetaIterationsTCE(float& betaIn, float& betaOut, float& betaAv, float & pt_beta, float sdIn_dr, float sdOut_dr, float dr, float lIn)
{
    if (lIn == 0)
    {
        betaOut += copysign(asinf(fminf(sdOut_dr * SDL::k2Rinv1GeVf / fabsf(pt_beta), SDL::sinAlphaMax)), betaOut);
        return;
    }

    if (betaIn * betaOut > 0.f and (fabsf(pt_beta) < 4.f * SDL::pt_betaMax or (lIn >= 11 and fabsf(pt_beta) < 8.f * SDL::pt_betaMax)))   //and the pt_beta is well-defined; less strict for endcap-endcap
    {

        const float betaInUpd  = betaIn + copysignf(asinf(fminf(sdIn_dr * SDL::k2Rinv1GeVf / fabsf(pt_beta), SDL::sinAlphaMax)), betaIn); //FIXME: need a faster version
        const float betaOutUpd = betaOut + copysignf(asinf(fminf(sdOut_dr * SDL::k2Rinv1GeVf / fabsf(pt_beta), SDL::sinAlphaMax)), betaOut); //FIXME: need a faster version
        betaAv = 0.5f * (betaInUpd + betaOutUpd);

        //1st update
        //pt_beta = dr * k2Rinv1GeVf / sinf(betaAv); //get a better pt estimate
        const float pt_beta_inv = 1.f/fabsf(dr * k2Rinv1GeVf / sinf(betaAv)); //get a better pt estimate

        betaIn  += copysignf(asinf(fminf(sdIn_dr * SDL::k2Rinv1GeVf *pt_beta_inv, SDL::sinAlphaMax)), betaIn); //FIXME: need a faster version
        betaOut += copysignf(asinf(fminf(sdOut_dr * SDL::k2Rinv1GeVf *pt_beta_inv, SDL::sinAlphaMax)), betaOut); //FIXME: need a faster version
        //update the av and pt
        betaAv = 0.5f * (betaIn + betaOut);
        //2nd update
        pt_beta = dr * SDL::k2Rinv1GeVf / sinf(betaAv); //get a better pt estimate
    }
    else if (lIn < 11 && fabsf(betaOut) < 0.2f * fabsf(betaIn) && fabsf(pt_beta) < 12.f * SDL::pt_betaMax)   //use betaIn sign as ref
    {

        const float pt_betaIn = dr * k2Rinv1GeVf / sinf(betaIn);

        const float betaInUpd  = betaIn + copysignf(asinf(fminf(sdIn_dr * SDL::k2Rinv1GeVf / fabsf(pt_betaIn), SDL::sinAlphaMax)), betaIn); //FIXME: need a faster version
        const float betaOutUpd = betaOut + copysignf(asinf(fminf(sdOut_dr * SDL::k2Rinv1GeVf / fabsf(pt_betaIn), SDL::sinAlphaMax)), betaIn); //FIXME: need a faster version
        betaAv = (fabsf(betaOut) > 0.2f * fabsf(betaIn)) ? (0.5f * (betaInUpd + betaOutUpd)) : betaInUpd;

        //1st update
        pt_beta = dr * SDL::k2Rinv1GeVf / sin(betaAv); //get a better pt estimate
        betaIn  += copysignf(asinf(fminf(sdIn_dr * SDL::k2Rinv1GeVf / fabsf(pt_beta), SDL::sinAlphaMax)), betaIn); //FIXME: need a faster version
        betaOut += copysignf(asinf(fminf(sdOut_dr * SDL::k2Rinv1GeVf / fabsf(pt_beta), SDL::sinAlphaMax)), betaIn); //FIXME: need a faster version
        //update the av and pt
        betaAv = 0.5f * (betaIn + betaOut);
        //2nd update
        pt_beta = dr * SDL::k2Rinv1GeVf / sin(betaAv); //get a better pt estimate

    }
}
__device__ float SDL::computeChiSquaredTCE(int nPoints, float* xs, float* ys, float* delta1, float* delta2, float* slopes, bool* isFlat, float g, float f, float radius)
{
    // given values of (g, f, radius) and a set of points (and its uncertainties)
    //compute chi squared
    float c = g*g + f*f - radius*radius;
    float chiSquared = 0.f;
    float absArctanSlope, angleM, xPrime, yPrime, sigma;
    for(size_t i = 0; i < nPoints; i++)
    {
        absArctanSlope = ((slopes[i] != 123456789) ? fabs(atanf(slopes[i])) : 0.5f*float(M_PI)); // Since C++ can't represent infinity, SDL_INF = 123456789 was used to represent infinity in the data table
        if(xs[i] > 0 and ys[i] > 0)
        {
            angleM = 0.5f*float(M_PI) - absArctanSlope;
        }
        else if(xs[i] < 0 and ys[i] > 0)
        {
            angleM = absArctanSlope + 0.5f*float(M_PI);
        }
        else if(xs[i] < 0 and ys[i] < 0)
        {
            angleM = -(absArctanSlope + 0.5f*float(M_PI));
        }
        else if(xs[i] > 0 and ys[i] < 0)
        {
            angleM = -(0.5f*float(M_PI) - absArctanSlope);
        }

        if(not isFlat[i])
        {
            xPrime = xs[i] * cosf(angleM) + ys[i] * sinf(angleM);
            yPrime = ys[i] * cosf(angleM) - xs[i] * sinf(angleM);
        }
        else
        {
            xPrime = xs[i];
            yPrime = ys[i];
        }
        sigma = 2 * sqrtf((xPrime * delta1[i]) * (xPrime * delta1[i]) + (yPrime * delta2[i]) * (yPrime * delta2[i]));
        chiSquared +=  (xs[i] * xs[i] + ys[i] * ys[i] - 2 * g * xs[i] - 2 * f * ys[i] + c) * (xs[i] * xs[i] + ys[i] * ys[i] - 2 * g * xs[i] - 2 * f * ys[i] + c) / (sigma * sigma);
    }
    return chiSquared;
}

__device__ void SDL::computeSigmasForRegressionTCE(SDL::modules& modulesInGPU, const uint16_t* lowerModuleIndices, float* delta1, float* delta2, float* slopes, bool* isFlat, int nPoints, bool anchorHits)
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
    float inv1 = 0.01f/0.009f;
    float inv2 = 0.15f/0.009f;
    float inv3 = 2.4f/0.009f;
    for(size_t i=0; i<nPoints; i++)
    {
        moduleType = modulesInGPU.moduleType[lowerModuleIndices[i]];
        moduleSubdet = modulesInGPU.subdets[lowerModuleIndices[i]];
        moduleSide = modulesInGPU.sides[lowerModuleIndices[i]];
        float& drdz = modulesInGPU.drdzs[lowerModuleIndices[i]];
        slopes[i] = modulesInGPU.slopes[lowerModuleIndices[i]];
        //category 1 - barrel PS flat
        if(moduleSubdet == Barrel and moduleType == PS and moduleSide == Center)
        {
            delta1[i] = inv1;//1.1111f;//0.01;
            delta2[i] = inv1;//1.1111f;//0.01;
            slopes[i] = -999.f;
            isFlat[i] = true;
        }

        //category 2 - barrel 2S
        else if(moduleSubdet == Barrel and moduleType == TwoS)
        {
            delta1[i] = 1.f;//0.009;
            delta2[i] = 1.f;//0.009;
            slopes[i] = -999.f;
            isFlat[i] = true;
        }

        //category 3 - barrel PS tilted
        else if(moduleSubdet == Barrel and moduleType == PS and moduleSide != Center)
        {

            //delta1[i] = 0.01;
            delta1[i] = inv1;//1.1111f;//0.01;
            isFlat[i] = false;

            if(anchorHits)
            {
                delta2[i] = (inv2 * drdz/sqrtf(1 + drdz * drdz));
            }
            else
            {
                delta2[i] = (inv3 * drdz/sqrtf(1 + drdz * drdz));
            }
        }
        //category 4 - endcap PS
        else if(moduleSubdet == Endcap and moduleType == PS)
        {
            delta1[i] = inv1;//1.1111f;//0.01;
            isFlat[i] = false;

            /*despite the type of the module layer of the lower module index,
            all anchor hits are on the pixel side and all non-anchor hits are
            on the strip side!*/
            if(anchorHits)
            {
                delta2[i] = inv2;//16.6666f;//0.15f;
            }
            else
            {
                delta2[i] = inv3;//266.666f;//2.4f;
            }
        }

        //category 5 - endcap 2S
        else if(moduleSubdet == Endcap and moduleType == TwoS)
        {
            delta1[i] = 1.f;//0.009;
            delta2[i] = 500.f*inv1;//555.5555f;//5.f;
            isFlat[i] = false;
        }
        else
        {
            printf("ERROR!!!!! I SHOULDN'T BE HERE!!!! subdet = %d, type = %d, side = %d\n", moduleSubdet, moduleType, moduleSide);
        }
    }
}
__device__ const int nEndcapModulesInner[] = {20,24,24,28,32,32,36,40,40,44,52,60,64,72,76};
__device__ const int nEndcapModulesOuter[] = {28,28,32,36,36,40,44,52,56,64,72,76};

__device__ const int nCentralBarrelModules[] = {7,11,15,24,24,24};
__device__ const int nCentralRods[] = {18, 26, 36, 48, 60, 78};

__device__ void SDL::findStaggeredNeighbours(struct SDL::modules& modulesInGPU, unsigned int moduleIdx, unsigned int* staggeredNeighbours, unsigned int& counter)
{
    //naive and expensive method
    counter = 0;
    bool flag = false;
    for(size_t i = 0; i < *(modulesInGPU.nLowerModules); i++)
    {
        flag = false;
        unsigned int partnerModuleIdx = i;
        //start
        unsigned int layer1 = modulesInGPU.layers[moduleIdx];
        unsigned int layer2 = modulesInGPU.layers[partnerModuleIdx];
        unsigned int module1 = modulesInGPU.modules[moduleIdx];
        unsigned int module2 = modulesInGPU.modules[partnerModuleIdx];

        if(layer1 != layer2) continue;

        if(modulesInGPU.subdets[moduleIdx] == 4 and modulesInGPU.subdets[partnerModuleIdx] == 4)
        {
            unsigned int ring1 = modulesInGPU.rings[moduleIdx];
            unsigned int ring2 = modulesInGPU.rings[partnerModuleIdx];
            if(ring1 != ring2) continue;

            if((layer1 <=2) and (fabsf(module1 - module2) == 1 or fabsf(module1 % nEndcapModulesInner[ring1 - 1] - module2 % nEndcapModulesInner[ring2 - 1]) == 1))
            {
                flag = true;
            }

            else if((layer1 > 2) and (fabsf(module1 - module2) == 1 or fabsf(module1 % nEndcapModulesOuter[ring1 - 1] - module2 % nEndcapModulesOuter[ring2 - 1]) == 1))
            {
                flag = true;
            }
        }
        else if(modulesInGPU.subdets[moduleIdx] == 5 and modulesInGPU.subdets[partnerModuleIdx] == 5)
        {
            unsigned int rod1 = modulesInGPU.rods[moduleIdx];
            unsigned int rod2 = modulesInGPU.rods[partnerModuleIdx];
            unsigned int side1 = modulesInGPU.sides[moduleIdx];
            unsigned int side2 = modulesInGPU.sides[partnerModuleIdx];


            if(side1 == side2)
            {
                if((fabsf(rod1 - rod2) == 1 and module1 == module2) or (fabsf(module1 - module2) == 1 and rod1 == rod2))
                {
                    flag = true;
                }
                else if(side1 == 3 and side2 == 3 and fabsf(rod1 % nCentralRods[layer1 - 1] - rod2 % nCentralRods[layer2 - 1]) == 1 and module1 == module2)
                {
                    flag = true;
                }
                else if(side1 != 3 and  fabsf(module1 % nCentralRods[layer1 - 1] - module2 % nCentralRods[layer2 - 1]) == 1 and rod1 == rod2)
                {
                    flag = true;
                }
            }
            else
            {
                if(side1 == 1 and side2 == 3 and rod1 == 12 and module2 == 1 and module1 == rod2)
                {
                    flag = true;
                }
                else if(side1 == 3 and side2 == 1 and rod2 == 12 and module1 == 1 and module1 == rod2)
                {
                    flag = true;
                }
                else if(side1 == 2 and side2 == 3 and rod1 == 1 and module2 == nCentralBarrelModules[layer2 - 1] and module1 == rod2)
                {
                    flag = true;
                }
                else if(side1 == 3 and side2 == 2 and module1 == nCentralBarrelModules[layer1 - 1] and rod2 == 1 and rod1 == module2)
                {
                    flag = true;
                }
            }
        }
        if(flag)
        {
            staggeredNeighbours[counter] = i;//deal in lower module indices
            counter++;
        }
    }
}

__device__ float SDL::computeRadiusFromThreeAnchorHitsTCE(float x1, float y1, float x2, float y2, float x3, float y3, float& g, float& f)
{
    float radius = 0.f;

    //writing manual code for computing radius, which obviously sucks
    //TODO:Use fancy inbuilt libraries like cuBLAS or cuSOLVE for this!
    //(g,f) -> center
    //first anchor hit - (x1,y1), second anchor hit - (x2,y2), third anchor hit - (x3, y3)

    /*
    if((y1 - y3) * (x2 - x3) - (x1 - x3) * (y2 - y3) == 0)
    {
        return -1; //WTF man three collinear points!
    }
    */

    float denomInv = 1.0f/((y1 - y3) * (x2 - x3) - (x1 - x3) * (y2 - y3));

    float xy1sqr = x1 * x1 + y1 * y1;

    float xy2sqr = x2 * x2 + y2 * y2;

    float xy3sqr = x3 * x3 + y3 * y3;

    g = 0.5f * ((y3 - y2) * xy1sqr + (y1 - y3) * xy2sqr + (y2 - y1) * xy3sqr) * denomInv;

    f = 0.5f * ((x2 - x3) * xy1sqr + (x3 - x1) * xy2sqr + (x1 - x2) * xy3sqr) * denomInv;

    float c = ((x2 * y3 - x3 * y2) * xy1sqr + (x3 * y1 - x1 * y3) * xy2sqr + (x1 * y2 - x2 * y1) * xy3sqr) * denomInv;

    if(((y1 - y3) * (x2 - x3) - (x1 - x3) * (y2 - y3) == 0) || (g * g + f * f - c < 0))
    {
        printf("three collinear points or FATAL! r^2 < 0!\n");
  radius = -1.f;
    }
    else
      radius = sqrtf(g * g  + f * f - c);

    return radius;
}
