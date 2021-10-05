# include "TrackExtensions.cuh"

SDL::trackExtensions::trackExtensions()
{
    constituentTCTypes = nullptr;
    constituentTCIndices = nullptr;
    nTrackExtensions = nullptr;
    rPhiChiSquared = nullptr;
    isDup = nullptr;
}

SDL::trackExtensions::~trackExtensions()
{
}

void SDL::trackExtensions::freeMemory()
{
    cudaFree(constituentTCTypes);
    cudaFree(constituentTCIndices);
    cudaFree(nTrackExtensions);
    cudaFree(isDup);
    cudaFree(rPhiChiSquared);
}

/*
   Track Extensions memory allocation - 10 slots for each TC (will reduce later)
   Extensions having the same anchor object will be clustered together for easy
   duplicate cleaning
*/

void SDL::createTrackExtensionsInUnifiedMemory(struct trackExtensions& trackExtensionsInGPU, unsigned int maxTrackExtensions, unsigned int nTrackCandidates)
{
    cudaMallocManaged(&trackExtensionsInGPU.constituentTCTypes, sizeof(short) * 3 * maxTrackExtensions);
    cudaMallocManaged(&trackExtensionsInGPU.constituentTCIndices, sizeof(unsigned int) * 3 * maxTrackExtensions);
    cudaMallocManaged(&trackExtensionsInGPU.nLayerOverlaps, sizeof(unsigned int) * 2 * maxTrackExtensions);
    cudaMallocManaged(&trackExtensionsInGPU.nHitOverlaps, sizeof(unsigned int) * 2 * maxTrackExtensions);
    cudaMallocManaged(&trackExtensionsInGPU.nTrackExtensions, nTrackCandidates * sizeof(unsigned int));
    cudaMallocManaged(&trackExtensionsInGPU.rPhiChiSquared, maxTrackExtensions * sizeof(float));
    cudaMallocManaged(&trackExtensionsInGPU.isDup, maxTrackExtensions * sizeof(bool));

    cudaMemset(trackExtensionsInGPU.nTrackExtensions, 0, nTrackCandidates * sizeof(unsigned int));
    cudaMemset(trackExtensionsInGPU.isDup, true, maxTrackExtensions * sizeof(bool));
}

void SDL::createTrackExtensionsInExplicitMemory(struct trackExtensions& trackExtensionsInGPU, unsigned int maxTrackExtensions, unsigned int nTrackCandidates)
{
    cudaMalloc(&trackExtensionsInGPU.constituentTCTypes, sizeof(short) * 3 * maxTrackExtensions);
    cudaMalloc(&trackExtensionsInGPU.constituentTCIndices, sizeof(unsigned int) * 3 * maxTrackExtensions);
    cudaMalloc(&trackExtensionsInGPU.nLayerOverlaps, sizeof(unsigned int) * 2 * maxTrackExtensions);
    cudaMalloc(&trackExtensionsInGPU.nHitOverlaps, sizeof(unsigned int) * 2 * maxTrackExtensions);
    cudaMalloc(&trackExtensionsInGPU.nTrackExtensions, nTrackCandidates * sizeof(unsigned int));
    cudaMalloc(&trackExtensionsInGPU.rPhiChiSquared, maxTrackExtensions * sizeof(float));
    cudaMalloc(&trackExtensionsInGPU.isDup, maxTrackExtensions * sizeof(bool));

    cudaMemset(trackExtensionsInGPU.nTrackExtensions, 0, nTrackCandidates * sizeof(unsigned int));
    cudaMemset(trackExtensionsInGPU.isDup, true, maxTrackExtensions * sizeof(bool));
}

__device__ void SDL::addTrackExtensionToMemory(struct trackExtensions& trackExtensionsInGPU, short* constituentTCType, unsigned int* constituentTCIndex, unsigned int* nLayerOverlaps, unsigned int* nHitOverlaps, float rPhiChiSquared, unsigned int trackExtensionIndex)
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
    trackExtensionsInGPU.rPhiChiSquared[trackExtensionIndex] = rPhiChiSquared;
}

__device__ bool SDL::runTrackExtensionDefaultAlgo(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct triplets& tripletsInGPU, struct quintuplets& quintupletsInGPU, struct pixelTriplets& pixelTripletsInGPU, struct pixelQuintuplets& pixelQuintupletsInGPU, struct trackCandidates& trackCandidatesInGPU, unsigned int anchorObjectIndex, unsigned int outerObjectIndex, short anchorObjectType, short outerObjectType, unsigned int anchorObjectOuterT3Index, unsigned int layerOverlapTarget, short* constituentTCType, unsigned int* constituentTCIndex, unsigned
        int* nLayerOverlaps, unsigned int* nHitOverlaps, float& rPhiChiSquared)
{
    /*
       Basic premise:
       1. given two objects, get the hit and module indices
       2. check for layer and hit overlap (layer overlap first checked using
       the 2-merge approach)
       3. Additional cuts - rz and rphi chi squared criteria! (TODO) 
    */

    bool pass = true;
    unsigned int* anchorLayerIndices = nullptr;
    unsigned int* anchorHitIndices = nullptr;
    unsigned int* anchorLowerModuleIndices = nullptr;

    unsigned int* outerObjectLayerIndices = nullptr;
    unsigned int* outerObjectHitIndices = nullptr;
    unsigned int* outerObjectLowerModuleIndices = nullptr;

    unsigned int nAnchorLayers = (anchorObjectType == 7) ? 7 : (anchorObjectType == 3 ? 3 : 5);
    unsigned int anchorObjectAnchorHitIndices[7];
    float centerX, centerY, radius;

    if(anchorObjectType != 3) //mostly this
    { 
        anchorLayerIndices = &trackCandidatesInGPU.logicalLayers[7 * anchorObjectIndex];
        anchorHitIndices = &trackCandidatesInGPU.hitIndices[14 * anchorObjectIndex];
        anchorLowerModuleIndices = &trackCandidatesInGPU.lowerModuleIndices[7 * anchorObjectIndex];
        centerX = trackCandidatesInGPU.centerX[anchorObjectIndex];
        centerY = trackCandidatesInGPU.centerY[anchorObjectIndex];
        radius = trackCandidatesInGPU.radius[anchorObjectIndex];
    }
    else //outlier
    {
        anchorLayerIndices = &tripletsInGPU.logicalLayers[3 * anchorObjectIndex];
        anchorHitIndices = &tripletsInGPU.hitIndices[6 * anchorObjectIndex];
        anchorLowerModuleIndices = &tripletsInGPU.lowerModuleIndices[3 * anchorObjectIndex];

    }
    unsigned int layer_binary = 0;
    for(size_t i=0; i<nAnchorLayers;i++)
    {
        if(modulesInGPU.isAnchor[hitsInGPU.moduleIndices[anchorHitIndices[2*i]]] or modulesInGPU.detIds[hitsInGPU.moduleIndices[anchorHitIndices[2*i]]] == 1)
        {
            anchorObjectAnchorHitIndices[i] = anchorHitIndices[2*i];
        }
        else
        {
            anchorObjectAnchorHitIndices[i] = anchorHitIndices[2*i+1];
        }
        layer_binary |= (1 << anchorLayerIndices[i]);
    }

    unsigned int nOuterLayers =(outerObjectType == 7) ? 7 : (outerObjectType == 3 ? 3 : 5); 

    unsigned int outerObjectAnchorHitIndices[7];
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

    for(size_t i=0; i<nOuterLayers;i++)
    {
        if(modulesInGPU.isAnchor[hitsInGPU.moduleIndices[outerObjectHitIndices[2*i]]] or modulesInGPU.detIds[hitsInGPU.moduleIndices[outerObjectHitIndices[2*i]]] == 1)
        {
            outerObjectAnchorHitIndices[i] = outerObjectHitIndices[2*i];
        }
        else
        {
            outerObjectAnchorHitIndices[i] = outerObjectHitIndices[2*i+1];
        }

        layer_binary |= (1 << outerObjectLayerIndices[i]);
    }
    
    unsigned int nLayerOverlap(0), nHitOverlap(0);
   
    float zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, pt_beta; //temp stuff
    float zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ;

    //checks for frivolous cases wherein
    pass = pass &  computeLayerAndHitOverlaps(modulesInGPU, anchorLayerIndices, anchorHitIndices, anchorLowerModuleIndices, outerObjectLayerIndices, outerObjectHitIndices, outerObjectLowerModuleIndices, nAnchorLayers, nOuterLayers, nLayerOverlap, nHitOverlap, layerOverlapTarget);

    pass = pass & runTrackletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, tripletsInGPU.lowerModuleIndices[3 * anchorObjectOuterT3Index], tripletsInGPU.lowerModuleIndices[3 * anchorObjectOuterT3Index + 1], tripletsInGPU.lowerModuleIndices[3 * outerObjectIndex], tripletsInGPU.lowerModuleIndices[3 * outerObjectIndex + 1], tripletsInGPU.segmentIndices[2 * anchorObjectOuterT3Index], tripletsInGPU.segmentIndices[2 * outerObjectIndex], zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, pt_beta, zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ);

    pass = pass & runTrackletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, tripletsInGPU.lowerModuleIndices[3 * anchorObjectOuterT3Index], tripletsInGPU.lowerModuleIndices[3 * anchorObjectOuterT3Index + 1], tripletsInGPU.lowerModuleIndices[3 * outerObjectIndex + 1], tripletsInGPU.lowerModuleIndices[3 * outerObjectIndex + 2], tripletsInGPU.segmentIndices[2 * anchorObjectOuterT3Index], tripletsInGPU.segmentIndices[2 * outerObjectIndex + 1], zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, pt_beta, zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ);

    rPhiChiSquared = computeTERPhiChiSquared(modulesInGPU, hitsInGPU, centerX, centerY, radius, outerObjectAnchorHitIndices, outerObjectLowerModuleIndices);


    //pass = pass and passTERPhiChiSquaredCuts(nLayerOverlap, nHitOverlap, layer_binary, rPhiChiSquared);

    nLayerOverlaps[0] = nLayerOverlap;
    nHitOverlaps[0] = nHitOverlap;

    constituentTCType[0] = anchorObjectType;
    constituentTCType[1] = outerObjectType;

    constituentTCIndex[0] = anchorObjectIndex;
    constituentTCIndex[1] = outerObjectIndex;

   return pass;
}

__device__ bool SDL::passTERPhiChiSquaredCuts(int nLayerOverlaps, int nHitOverlaps, unsigned int layer_binary, float rPhiChiSquared)
{
    bool pass = true;
    float threshold;
    //giant wall of if conditions - machine generated code!!
    if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 127)
    {
        return rPhiChiSquared < 24.433;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 1927)
    {
        return rPhiChiSquared < 4.617;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 911)
    {
        return rPhiChiSquared < 5.305;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 415)
    {
        return rPhiChiSquared < 9.244;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 191)
    {
        return rPhiChiSquared < 16.109;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 3971)
    {
        return rPhiChiSquared < 7.003;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 3973)
    {
        return rPhiChiSquared < 3.044;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 126)
    {
        return rPhiChiSquared < 97.948;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 3970)
    {
        return rPhiChiSquared < 37.058;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 414)
    {
        return rPhiChiSquared < 10.621;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 445)
    {
        return rPhiChiSquared < 14.021;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 190)
    {
        return rPhiChiSquared < 9.244;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 1926)
    {
        return rPhiChiSquared < 0.759;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 3972)
    {
        return rPhiChiSquared < 9.244;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 910)
    {
        return rPhiChiSquared < 0.872;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 444)
    {
        return rPhiChiSquared < 0.501;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 1933)
    {
        return true;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 127)
    {
        return rPhiChiSquared < 18.509;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 1927)
    {
        return rPhiChiSquared < 3.498;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 911)
    {
        return rPhiChiSquared < 4.617;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 415)
    {
        return rPhiChiSquared < 7.003;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 191)
    {
        return rPhiChiSquared < 16.109;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 3971)
    {
        return rPhiChiSquared < 9.244;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 3973)
    {
        return rPhiChiSquared < 42.578;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 126)
    {
        return rPhiChiSquared < 28.073;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 3970)
    {
        return rPhiChiSquared < 16.109;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 414)
    {
        return rPhiChiSquared < 0.379;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 445)
    {
        return rPhiChiSquared < 1.747;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 190)
    {
        return rPhiChiSquared < 3.044;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 1926)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 3972)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 910)
    {
        return rPhiChiSquared < 0.003;
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
        return rPhiChiSquared < 18.509;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 1927)
    {
        return rPhiChiSquared < 4.617;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 911)
    {
        return rPhiChiSquared < 3.498;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 415)
    {
        return rPhiChiSquared < 10.621;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 191)
    {
        return rPhiChiSquared < 12.203;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 3971)
    {
        return rPhiChiSquared < 97.948;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 3973)
    {
        return rPhiChiSquared < 1.747;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 126)
    {
        return rPhiChiSquared < 21.266;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 3970)
    {
        return rPhiChiSquared < 0.009;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 414)
    {
        return rPhiChiSquared < 4.617;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 445)
    {
        return rPhiChiSquared < 0.008;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 190)
    {
        return rPhiChiSquared < 0.082;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 1926)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 3972)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 910)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 444)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 1933)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 127)
    {
        return rPhiChiSquared < 0.125;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 1927)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 911)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 415)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 191)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 3971)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 3973)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 3841)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 61)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 31)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 391)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 1921)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 901)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 143)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 157)
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
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 126)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 3970)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 414)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 445)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 190)
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
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 910)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 127)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 1927)
    {
        return rPhiChiSquared < 0.004;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 911)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 415)
    {
        return rPhiChiSquared < 2.007;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 191)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 3971)
    {
        return rPhiChiSquared < 1.747;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 3973)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 126)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 414)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 445)
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
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 3972)
    {
        return false;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 910)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 2 and layer_binary == 3975)
    {
        return rPhiChiSquared < 3.498;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 2 and layer_binary == 1935)
    {
        return rPhiChiSquared < 7.003;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 2 and layer_binary == 447)
    {
        return rPhiChiSquared < 7.003;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 2 and layer_binary == 446)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 2 and layer_binary == 3974)
    {
        return rPhiChiSquared < 0.014;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 2 and layer_binary == 1934)
    {
        return true;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 3975)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 125)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 399)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 3969)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 63)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 159)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 903)
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
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 1925)
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
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 1935)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 447)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 3974)
    {
        return false;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 1 and layer_binary == 3975)
    {
        return rPhiChiSquared < 0.436;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 1 and layer_binary == 1935)
    {
        return false;
    }
    return pass; //escape hatch
}

/*
   If one out of the two hits don't overlap - check if the module indices are identical (case of multiple reco hits)
   If both hits don't overlap - check the above, and check for staggered modules using the staggered module neighbours list

   This function i complicated - computes layer overlaps and checks if layer matches and hit matches are "compatible" i.e., layer overlap = 2 * hit overlap, or if that's not the case, we know why (multiple reco hits/staggered modules)
*/
__device__ bool SDL::computeLayerAndHitOverlaps(SDL::modules& modulesInGPU, unsigned int* anchorLayerIndices, unsigned int* anchorHitIndices, unsigned int* anchorLowerModuleIndices, unsigned int* outerObjectLayerIndices, unsigned int* outerObjectHitIndices, unsigned int* outerObjectLowerModuleIndices, unsigned int nAnchorLayers, unsigned int nOuterLayers, unsigned int& nLayerOverlap, unsigned int& nHitOverlap, unsigned int& layerOverlapTarget)
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
    pass = pass & (nLayerOverlap == layerOverlapTarget); //not really required, because these cases should be handled by the other conditions
    return pass;
}


/* r-z and r-phi chi squared computation*/
__device__ float computeTERZChiSquared(struct modules& modulesInGPU, struct hits& hitsInGPU, unsigned int* anchorHitIndices, unsigned int* anchorLowerModuleIndices, unsigned int* outerHitIndices, unsigned int* outerLowerModuleIndices)
{
    //using the pixel hits to create the slope
    
}

__device__ float SDL::computeTERPhiChiSquared(struct modules& modulesInGPU, struct hits& hitsInGPU, float& g, float& f, float& radius, unsigned int* outerObjectAnchorHits, unsigned int* outerObjectLowerModuleIndices)
{
    //Three cases
    float delta1[3], delta2[3], slopes[3], xs[3], ys[3];
    bool isFlat[3];
    computeSigmasForRegression(modulesInGPU, outerObjectLowerModuleIndices, delta1, delta2, slopes, isFlat, 3, true);

    for(size_t i = 0; i < 3; i++)
    {
        xs[i] = hitsInGPU.xs[outerObjectAnchorHits[i]];
        ys[i] = hitsInGPU.ys[outerObjectAnchorHits[i]];
    }
    float chiSquared = computeChiSquared(3, xs, ys, delta1, delta2, slopes, isFlat, g, f, radius);
    return chiSquared;
}
