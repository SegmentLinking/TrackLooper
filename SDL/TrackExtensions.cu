# include "TrackExtensions.cuh"

SDL::trackExtensions::trackExtensions()
{
    constituentTCTypes = nullptr;
    constituentTCIndices = nullptr;
    nHitOverlaps = nullptr;
    nLayerOverlaps = nullptr;
    nTrackExtensions = nullptr;
    rPhiChiSquared = nullptr;
    rzChiSquared = nullptr;
    innerRadius = nullptr;
    outerRadius = nullptr;
    regressionRadius = nullptr;
    isDup = nullptr;
}

SDL::trackExtensions::~trackExtensions()
{
}

void SDL::trackExtensions::freeMemory()
{
    cudaFree(constituentTCTypes);
    cudaFree(constituentTCIndices);
    cudaFree(nLayerOverlaps);
    cudaFree(nHitOverlaps);
    cudaFree(nTrackExtensions);
    cudaFree(isDup);
    cudaFree(rPhiChiSquared);
    cudaFree(rzChiSquared);
    cudaFree(regressionRadius);
    cudaFree(innerRadius);
    cudaFree(outerRadius);
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
    cudaMallocManaged(&trackExtensionsInGPU.rzChiSquared, maxTrackExtensions * sizeof(float));
    cudaMallocManaged(&trackExtensionsInGPU.regressionRadius, maxTrackExtensions * sizeof(float));
    cudaMallocManaged(&trackExtensionsInGPU.innerRadius, maxTrackExtensions * sizeof(float));
    cudaMallocManaged(&trackExtensionsInGPU.outerRadius, maxTrackExtensions * sizeof(float));

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
    cudaMalloc(&trackExtensionsInGPU.rzChiSquared, maxTrackExtensions * sizeof(float));
    cudaMalloc(&trackExtensionsInGPU.regressionRadius, maxTrackExtensions * sizeof(float));
    cudaMalloc(&trackExtensionsInGPU.innerRadius, maxTrackExtensions * sizeof(float));
    cudaMalloc(&trackExtensionsInGPU.outerRadius, maxTrackExtensions * sizeof(float));

    cudaMalloc(&trackExtensionsInGPU.isDup, maxTrackExtensions * sizeof(bool));

    cudaMemset(trackExtensionsInGPU.nTrackExtensions, 0, nTrackCandidates * sizeof(unsigned int));
    cudaMemset(trackExtensionsInGPU.isDup, true, maxTrackExtensions * sizeof(bool));
}

__device__ void SDL::addTrackExtensionToMemory(struct trackExtensions& trackExtensionsInGPU, short* constituentTCType, unsigned int* constituentTCIndex, unsigned int* nLayerOverlaps, unsigned int* nHitOverlaps, float rPhiChiSquared, float rzChiSquared, float regressionRadius, float innerRadius, float outerRadius, unsigned int trackExtensionIndex)
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
    trackExtensionsInGPU.rzChiSquared[trackExtensionIndex] = rzChiSquared;
    trackExtensionsInGPU.regressionRadius[trackExtensionIndex] = regressionRadius;
    trackExtensionsInGPU.innerRadius[trackExtensionIndex] = innerRadius;
    trackExtensionsInGPU.outerRadius[trackExtensionIndex] = outerRadius;

}

__device__ bool SDL::runTrackExtensionDefaultAlgo(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct triplets& tripletsInGPU, struct quintuplets& quintupletsInGPU, struct pixelTriplets& pixelTripletsInGPU, struct pixelQuintuplets& pixelQuintupletsInGPU, struct trackCandidates& trackCandidatesInGPU, unsigned int anchorObjectIndex, unsigned int outerObjectIndex, short anchorObjectType, short outerObjectType, unsigned int anchorObjectOuterT3Index, unsigned int layerOverlapTarget, short* constituentTCType, unsigned int* constituentTCIndex, unsigned
        int* nLayerOverlaps, unsigned int* nHitOverlaps, float& rPhiChiSquared, float& rzChiSquared, float& regressionRadius, float& innerRadius, float& outerRadius)
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
    float centerX, centerY;
    if(anchorObjectType != 3) //mostly this
    { 
        anchorLayerIndices = &trackCandidatesInGPU.logicalLayers[7 * anchorObjectIndex];
        anchorHitIndices = &trackCandidatesInGPU.hitIndices[14 * anchorObjectIndex];
        anchorLowerModuleIndices = &trackCandidatesInGPU.lowerModuleIndices[7 * anchorObjectIndex];
        centerX = trackCandidatesInGPU.centerX[anchorObjectIndex];
        centerY = trackCandidatesInGPU.centerY[anchorObjectIndex];
        innerRadius = trackCandidatesInGPU.radius[anchorObjectIndex];
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

    //checks for frivolous cases wherein
    pass = pass and computeLayerAndHitOverlaps(modulesInGPU, anchorLayerIndices, anchorHitIndices, anchorLowerModuleIndices, outerObjectLayerIndices, outerObjectHitIndices, outerObjectLowerModuleIndices, nAnchorLayers, nOuterLayers, nLayerOverlap, nHitOverlap, layerOverlapTarget);

    pass = pass and runTrackletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, tripletsInGPU.lowerModuleIndices[3 * anchorObjectOuterT3Index], tripletsInGPU.lowerModuleIndices[3 * anchorObjectOuterT3Index + 1], tripletsInGPU.lowerModuleIndices[3 * outerObjectIndex], tripletsInGPU.lowerModuleIndices[3 * outerObjectIndex + 1], tripletsInGPU.segmentIndices[2 * anchorObjectOuterT3Index], tripletsInGPU.segmentIndices[2 * outerObjectIndex], zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, pt_beta, zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ);

    pass = pass and runTrackletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, tripletsInGPU.lowerModuleIndices[3 * anchorObjectOuterT3Index], tripletsInGPU.lowerModuleIndices[3 * anchorObjectOuterT3Index + 1], tripletsInGPU.lowerModuleIndices[3 * outerObjectIndex + 1], tripletsInGPU.lowerModuleIndices[3 * outerObjectIndex + 2], tripletsInGPU.segmentIndices[2 * anchorObjectOuterT3Index], tripletsInGPU.segmentIndices[2 * outerObjectIndex + 1], zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, pt_beta, zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ);


    unsigned int anchorObjectAnchorHitIndices[7];
    unsigned int outerObjectAnchorHitIndices[7];
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

    if(anchorObjectType != 3)
    {
        rPhiChiSquared = computeTERPhiChiSquared(modulesInGPU, hitsInGPU, centerX, centerY, innerRadius, outerObjectAnchorHitIndices, outerObjectLowerModuleIndices);
        rzChiSquared = computeTERZChiSquared(modulesInGPU, hitsInGPU, anchorObjectAnchorHitIndices, anchorLowerModuleIndices, outerObjectAnchorHitIndices, outerObjectLowerModuleIndices, anchorObjectType);
        pass = pass and passTERPhiChiSquaredCuts(nLayerOverlap, nHitOverlap, layer_binary, rPhiChiSquared);
        pass = pass and passTERZChiSquaredCuts(nLayerOverlap, nHitOverlap, layer_binary, rzChiSquared);
    }
    else
    {
        //create a unified list of hit indices and lower module indices
        unsigned int overallAnchorIndices[6];
        unsigned int overallLowerModuleIndices[6];
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
        innerRadius = computeRadiusFromThreeAnchorHits(x1, y1, x2, y2, x3, y3, g, f);

        x1 = hitsInGPU.xs[outerObjectAnchorHitIndices[0]];
        x2 = hitsInGPU.xs[outerObjectAnchorHitIndices[1]];
        x3 = hitsInGPU.xs[outerObjectAnchorHitIndices[2]];
        y1 = hitsInGPU.ys[outerObjectAnchorHitIndices[0]];
        y2 = hitsInGPU.ys[outerObjectAnchorHitIndices[1]];
        y3 = hitsInGPU.ys[outerObjectAnchorHitIndices[2]];

        outerRadius = computeRadiusFromThreeAnchorHits(x1, y1, x2, y2, x3, y3, g, f);


        rPhiChiSquared = computeT3T3RPhiChiSquared(modulesInGPU, hitsInGPU, nPoints, overallAnchorIndices, overallLowerModuleIndices, regressionRadius);
        rzChiSquared = computeT3T3RZChiSquared(modulesInGPU, hitsInGPU, nPoints, overallAnchorIndices, overallLowerModuleIndices);
        pass = pass and (regressionRadius > 0.95/(2 * k2Rinv1GeVf));

        pass = pass and passRadiusMatch(nLayerOverlap, nHitOverlap, layer_binary, innerRadius, outerRadius);     


        pass = pass and passTERPhiChiSquaredCuts(nLayerOverlap, nHitOverlap, layer_binary, rPhiChiSquared);
        pass = pass and passTERZChiSquaredCuts(nLayerOverlap, nHitOverlap, layer_binary, rzChiSquared);
   }
    

    nLayerOverlaps[0] = nLayerOverlap;
    nHitOverlaps[0] = nHitOverlap;

    constituentTCType[0] = anchorObjectType;
    constituentTCType[1] = outerObjectType;

    constituentTCIndex[0] = anchorObjectIndex;
    constituentTCIndex[1] = outerObjectIndex;

    return pass;
}

__device__ bool SDL::passRadiusMatch(unsigned int& nLayerOverlaps, unsigned int& nHitOverlaps, unsigned int& layer_binary, float& innerRadius, float& outerRadius)
{
    float innerInvRadiusPositiveErrorBound, outerInvRadiusPositiveErrorBound;
    float innerInvRadiusNegativeErrorBound, outerInvRadiusNegativeErrorBound;
    float innerRadiusInvMin, innerRadiusInvMax, outerRadiusInvMin, outerRadiusInvMax;

    if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 30)
    {
        innerInvRadiusPositiveErrorBound = 0.161;
        innerInvRadiusNegativeErrorBound = 1.876;
        outerInvRadiusPositiveErrorBound = 0.151;
        outerInvRadiusNegativeErrorBound = 0.120;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 30)
    {
        innerInvRadiusPositiveErrorBound = 0.343;
        innerInvRadiusNegativeErrorBound = 1.876;
        outerInvRadiusPositiveErrorBound = 0.128;
        outerInvRadiusNegativeErrorBound = 0.417;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 30)
    {
        innerInvRadiusPositiveErrorBound = 0.196;
        innerInvRadiusNegativeErrorBound = 0.659;
        outerInvRadiusPositiveErrorBound = 0.142;
        outerInvRadiusNegativeErrorBound = 0.178;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 30)
    {
        innerInvRadiusPositiveErrorBound = 0.196;
        innerInvRadiusNegativeErrorBound = 0.366;
        outerInvRadiusPositiveErrorBound = 0.146;
        outerInvRadiusNegativeErrorBound = 0.507;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 30)
    {
        innerInvRadiusPositiveErrorBound = 0.146;
        innerInvRadiusNegativeErrorBound = 0.255;
        outerInvRadiusPositiveErrorBound = 0.210;
        outerInvRadiusNegativeErrorBound = 0.224;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 60)
    {
        innerInvRadiusPositiveErrorBound = 0.019;
        innerInvRadiusNegativeErrorBound = 0.914;
        outerInvRadiusPositiveErrorBound = 0.172;
        outerInvRadiusNegativeErrorBound = 0.007;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 60)
    {
        innerInvRadiusPositiveErrorBound = 0.027;
        innerInvRadiusNegativeErrorBound = 0.021;
        outerInvRadiusPositiveErrorBound = 0.030;
        outerInvRadiusNegativeErrorBound = 0.065;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 60)
    {
        innerInvRadiusPositiveErrorBound = 0.196;
        innerInvRadiusNegativeErrorBound = 1.226;
        outerInvRadiusPositiveErrorBound = 0.231;
        outerInvRadiusNegativeErrorBound = 0.247;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 60)
    {
        innerInvRadiusPositiveErrorBound = 0.113;
        innerInvRadiusNegativeErrorBound = 0.802;
        outerInvRadiusPositiveErrorBound = 0.172;
        outerInvRadiusNegativeErrorBound = 0.172;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 60)
    {
        innerInvRadiusPositiveErrorBound = 0.161;
        innerInvRadiusNegativeErrorBound = 0.390;
        outerInvRadiusPositiveErrorBound = 0.239;
        outerInvRadiusNegativeErrorBound = 0.403;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 62)
    {
        innerInvRadiusPositiveErrorBound = 0.084;
        innerInvRadiusNegativeErrorBound = 0.255;
        outerInvRadiusPositiveErrorBound = 0.084;
        outerInvRadiusNegativeErrorBound = 0.247;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 120)
    {
        innerInvRadiusPositiveErrorBound = 0.023;
        innerInvRadiusNegativeErrorBound = 0.231;
        outerInvRadiusPositiveErrorBound = 0.030;
        outerInvRadiusNegativeErrorBound = 0.026;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 120)
    {
        innerInvRadiusPositiveErrorBound = 0.524;
        innerInvRadiusNegativeErrorBound = 0.013;
        outerInvRadiusPositiveErrorBound = 0.172;
        outerInvRadiusNegativeErrorBound = 0.172;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 120)
    {
        innerInvRadiusPositiveErrorBound = 0.460;
        innerInvRadiusNegativeErrorBound = 3.380;
        outerInvRadiusPositiveErrorBound = 0.541;
        outerInvRadiusNegativeErrorBound = 0.231;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 120)
    {
        innerInvRadiusPositiveErrorBound = 0.172;
        innerInvRadiusNegativeErrorBound = 0.659;
        outerInvRadiusPositiveErrorBound = 0.217;
        outerInvRadiusNegativeErrorBound = 0.151;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 120)
    {
        innerInvRadiusPositiveErrorBound = 0.184;
        innerInvRadiusNegativeErrorBound = 0.659;
        outerInvRadiusPositiveErrorBound = 0.239;
        outerInvRadiusNegativeErrorBound = 0.366;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 124)
    {
        innerInvRadiusPositiveErrorBound = 0.057;
        innerInvRadiusNegativeErrorBound = 0.040;
        outerInvRadiusPositiveErrorBound = 0.065;
        outerInvRadiusNegativeErrorBound = 0.113;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 142)
    {
        innerInvRadiusPositiveErrorBound = 0.120;
        innerInvRadiusNegativeErrorBound = 0.106;
        outerInvRadiusPositiveErrorBound = 0.291;
        outerInvRadiusNegativeErrorBound = 0.445;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 142)
    {
        innerInvRadiusPositiveErrorBound = 0.102;
        innerInvRadiusNegativeErrorBound = 0.113;
        outerInvRadiusPositiveErrorBound = 0.167;
        outerInvRadiusNegativeErrorBound = 0.172;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 142)
    {
        innerInvRadiusPositiveErrorBound = 0.133;
        innerInvRadiusNegativeErrorBound = 0.106;
        outerInvRadiusPositiveErrorBound = 0.301;
        outerInvRadiusNegativeErrorBound = 0.659;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 142)
    {
        innerInvRadiusPositiveErrorBound = 0.321;
        innerInvRadiusNegativeErrorBound = 0.113;
        outerInvRadiusPositiveErrorBound = 0.239;
        outerInvRadiusNegativeErrorBound = 0.659;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 142)
    {
        innerInvRadiusPositiveErrorBound = 0.151;
        innerInvRadiusNegativeErrorBound = 0.124;
        outerInvRadiusPositiveErrorBound = 0.291;
        outerInvRadiusNegativeErrorBound = 0.884;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 156)
    {
        innerInvRadiusPositiveErrorBound = 0.021;
        innerInvRadiusNegativeErrorBound = 0.033;
        outerInvRadiusPositiveErrorBound = 0.074;
        outerInvRadiusNegativeErrorBound = 0.802;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 156)
    {
        innerInvRadiusPositiveErrorBound = 0.002;
        innerInvRadiusNegativeErrorBound = 0.002;
        outerInvRadiusPositiveErrorBound = 0.007;
        outerInvRadiusNegativeErrorBound = 0.007;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 156)
    {
        innerInvRadiusPositiveErrorBound = 0.167;
        innerInvRadiusNegativeErrorBound = 0.137;
        outerInvRadiusPositiveErrorBound = 0.331;
        outerInvRadiusNegativeErrorBound = 1.076;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 156)
    {
        innerInvRadiusPositiveErrorBound = 0.113;
        innerInvRadiusNegativeErrorBound = 0.044;
        outerInvRadiusPositiveErrorBound = 0.331;
        outerInvRadiusNegativeErrorBound = 0.975;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 156)
    {
        innerInvRadiusPositiveErrorBound = 0.151;
        innerInvRadiusNegativeErrorBound = 0.239;
        outerInvRadiusPositiveErrorBound = 0.417;
        outerInvRadiusNegativeErrorBound = 1.646;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 158)
    {
        innerInvRadiusPositiveErrorBound = 0.366;
        innerInvRadiusNegativeErrorBound = 0.255;
        outerInvRadiusPositiveErrorBound = 0.403;
        outerInvRadiusNegativeErrorBound = 1.187;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 184)
    {
        innerInvRadiusPositiveErrorBound = 0.044;
        innerInvRadiusNegativeErrorBound = 0.015;
        outerInvRadiusPositiveErrorBound = 0.151;
        outerInvRadiusNegativeErrorBound = 0.093;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 184)
    {
        innerInvRadiusPositiveErrorBound = 0.029;
        innerInvRadiusNegativeErrorBound = 0.029;
        outerInvRadiusPositiveErrorBound = 0.638;
        outerInvRadiusNegativeErrorBound = 0.638;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 184)
    {
        innerInvRadiusPositiveErrorBound = 0.343;
        innerInvRadiusNegativeErrorBound = 2.965;
        outerInvRadiusPositiveErrorBound = 0.597;
        outerInvRadiusNegativeErrorBound = 0.802;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 184)
    {
        innerInvRadiusPositiveErrorBound = 0.096;
        innerInvRadiusNegativeErrorBound = 0.161;
        outerInvRadiusPositiveErrorBound = 0.541;
        outerInvRadiusNegativeErrorBound = 0.884;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 184)
    {
        innerInvRadiusPositiveErrorBound = 0.247;
        innerInvRadiusNegativeErrorBound = 1.267;
        outerInvRadiusPositiveErrorBound = 0.597;
        outerInvRadiusNegativeErrorBound = 1.701;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 188)
    {
        innerInvRadiusPositiveErrorBound = 0.030;
        innerInvRadiusNegativeErrorBound = 0.025;
        outerInvRadiusPositiveErrorBound = 0.403;
        outerInvRadiusNegativeErrorBound = 0.378;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 390)
    {
        innerInvRadiusPositiveErrorBound = 0.272;
        innerInvRadiusNegativeErrorBound = 0.239;
        outerInvRadiusPositiveErrorBound = 0.321;
        outerInvRadiusNegativeErrorBound = 1.041;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 390)
    {
        innerInvRadiusPositiveErrorBound = 0.247;
        innerInvRadiusNegativeErrorBound = 0.190;
        outerInvRadiusPositiveErrorBound = 0.109;
        outerInvRadiusNegativeErrorBound = 1.076;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 390)
    {
        innerInvRadiusPositiveErrorBound = 0.331;
        innerInvRadiusNegativeErrorBound = 0.403;
        outerInvRadiusPositiveErrorBound = 0.431;
        outerInvRadiusNegativeErrorBound = 1.267;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 390)
    {
        innerInvRadiusPositiveErrorBound = 0.727;
        innerInvRadiusNegativeErrorBound = 0.217;
        outerInvRadiusPositiveErrorBound = 0.727;
        outerInvRadiusNegativeErrorBound = 1.226;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 390)
    {
        innerInvRadiusPositiveErrorBound = 0.255;
        innerInvRadiusNegativeErrorBound = 0.431;
        outerInvRadiusPositiveErrorBound = 0.378;
        outerInvRadiusNegativeErrorBound = 1.267;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 396)
    {
        innerInvRadiusPositiveErrorBound = 0.063;
        innerInvRadiusNegativeErrorBound = 0.311;
        outerInvRadiusPositiveErrorBound = 0.403;
        outerInvRadiusNegativeErrorBound = 1.938;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 396)
    {
        innerInvRadiusPositiveErrorBound = 0.137;
        innerInvRadiusNegativeErrorBound = 0.460;
        outerInvRadiusPositiveErrorBound = 0.475;
        outerInvRadiusNegativeErrorBound = 0.541;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 396)
    {
        innerInvRadiusPositiveErrorBound = 0.264;
        innerInvRadiusNegativeErrorBound = 0.460;
        outerInvRadiusPositiveErrorBound = 0.856;
        outerInvRadiusNegativeErrorBound = 9.015;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 396)
    {
        innerInvRadiusPositiveErrorBound = 0.291;
        innerInvRadiusNegativeErrorBound = 0.445;
        outerInvRadiusPositiveErrorBound = 0.491;
        outerInvRadiusNegativeErrorBound = 8.444;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 396)
    {
        innerInvRadiusPositiveErrorBound = 0.281;
        innerInvRadiusNegativeErrorBound = 3.271;
        outerInvRadiusPositiveErrorBound = 0.597;
        outerInvRadiusNegativeErrorBound = 9.624;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 398)
    {
        innerInvRadiusPositiveErrorBound = 0.087;
        innerInvRadiusNegativeErrorBound = 0.093;
        outerInvRadiusPositiveErrorBound = 0.541;
        outerInvRadiusNegativeErrorBound = 9.944;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 408)
    {
        innerInvRadiusPositiveErrorBound = 0.021;
        innerInvRadiusNegativeErrorBound = 1.646;
        outerInvRadiusPositiveErrorBound = 0.491;
        outerInvRadiusNegativeErrorBound = 0.975;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 408)
    {
        innerInvRadiusPositiveErrorBound = 0.475;
        innerInvRadiusNegativeErrorBound = 4.843;
        outerInvRadiusPositiveErrorBound = 0.541;
        outerInvRadiusNegativeErrorBound = 24.045;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 408)
    {
        innerInvRadiusPositiveErrorBound = 0.475;
        innerInvRadiusNegativeErrorBound = 4.843;
        outerInvRadiusPositiveErrorBound = 0.475;
        outerInvRadiusNegativeErrorBound = 24.045;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 408)
    {
        innerInvRadiusPositiveErrorBound = 0.524;
        innerInvRadiusNegativeErrorBound = 0.975;
        outerInvRadiusPositiveErrorBound = 0.491;
        outerInvRadiusNegativeErrorBound = 24.045;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 412)
    {
        innerInvRadiusPositiveErrorBound = 0.210;
        innerInvRadiusNegativeErrorBound = 0.541;
        outerInvRadiusPositiveErrorBound = 0.491;
        outerInvRadiusNegativeErrorBound = 3.492;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 432)
    {
        innerInvRadiusPositiveErrorBound = 0.178;
        innerInvRadiusNegativeErrorBound = 0.178;
        outerInvRadiusPositiveErrorBound = 0.460;
        outerInvRadiusNegativeErrorBound = 0.460;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 432)
    {
        innerInvRadiusPositiveErrorBound = 0.178;
        innerInvRadiusNegativeErrorBound = 0.366;
        outerInvRadiusPositiveErrorBound = 0.524;
        outerInvRadiusNegativeErrorBound = 0.524;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 898)
    {
        innerInvRadiusPositiveErrorBound = 0.541;
        innerInvRadiusNegativeErrorBound = 3.728;
        outerInvRadiusPositiveErrorBound = 0.343;
        outerInvRadiusNegativeErrorBound = 0.431;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 898)
    {
        innerInvRadiusPositiveErrorBound = 0.390;
        innerInvRadiusNegativeErrorBound = 2.437;
        outerInvRadiusPositiveErrorBound = 0.291;
        outerInvRadiusNegativeErrorBound = 0.681;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 898)
    {
        innerInvRadiusPositiveErrorBound = 0.460;
        innerInvRadiusNegativeErrorBound = 5.893;
        outerInvRadiusPositiveErrorBound = 0.247;
        outerInvRadiusNegativeErrorBound = 0.445;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 898)
    {
        innerInvRadiusPositiveErrorBound = 0.507;
        innerInvRadiusNegativeErrorBound = 2.437;
        outerInvRadiusPositiveErrorBound = 0.390;
        outerInvRadiusNegativeErrorBound = 1.226;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 898)
    {
        innerInvRadiusPositiveErrorBound = 0.417;
        innerInvRadiusNegativeErrorBound = 2.437;
        outerInvRadiusPositiveErrorBound = 0.281;
        outerInvRadiusNegativeErrorBound = 0.491;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 900)
    {
        innerInvRadiusPositiveErrorBound = 0.184;
        innerInvRadiusNegativeErrorBound = 0.093;
        outerInvRadiusPositiveErrorBound = 0.042;
        outerInvRadiusNegativeErrorBound = 2.602;
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
        innerInvRadiusPositiveErrorBound = 0.403;
        innerInvRadiusNegativeErrorBound = 0.703;
        outerInvRadiusPositiveErrorBound = 0.524;
        outerInvRadiusNegativeErrorBound = 14.249;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 900)
    {
        innerInvRadiusPositiveErrorBound = 0.099;
        innerInvRadiusNegativeErrorBound = 0.378;
        outerInvRadiusPositiveErrorBound = 0.475;
        outerInvRadiusNegativeErrorBound = 5.893;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 900)
    {
        innerInvRadiusPositiveErrorBound = 0.343;
        innerInvRadiusNegativeErrorBound = 0.975;
        outerInvRadiusPositiveErrorBound = 0.617;
        outerInvRadiusNegativeErrorBound = 5.520;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 902)
    {
        innerInvRadiusPositiveErrorBound = 0.184;
        innerInvRadiusNegativeErrorBound = 0.239;
        outerInvRadiusPositiveErrorBound = 0.460;
        outerInvRadiusNegativeErrorBound = 3.608;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 904)
    {
        innerInvRadiusPositiveErrorBound = 0.113;
        innerInvRadiusNegativeErrorBound = 0.113;
        outerInvRadiusPositiveErrorBound = 0.099;
        outerInvRadiusNegativeErrorBound = 0.099;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 904)
    {
        innerInvRadiusPositiveErrorBound = 0.524;
        innerInvRadiusNegativeErrorBound = 5.520;
        outerInvRadiusPositiveErrorBound = 0.578;
        outerInvRadiusNegativeErrorBound = 1.876;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 904)
    {
        innerInvRadiusPositiveErrorBound = 0.178;
        innerInvRadiusNegativeErrorBound = 1.938;
        outerInvRadiusPositiveErrorBound = 0.460;
        outerInvRadiusNegativeErrorBound = 0.491;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 904)
    {
        innerInvRadiusPositiveErrorBound = 0.524;
        innerInvRadiusNegativeErrorBound = 4.687;
        outerInvRadiusPositiveErrorBound = 0.578;
        outerInvRadiusNegativeErrorBound = 21.798;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 908)
    {
        innerInvRadiusPositiveErrorBound = 0.045;
        innerInvRadiusNegativeErrorBound = 0.099;
        outerInvRadiusPositiveErrorBound = 0.431;
        outerInvRadiusNegativeErrorBound = 8.172;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 1920)
    {
        innerInvRadiusPositiveErrorBound = 0.431;
        innerInvRadiusNegativeErrorBound = 1.226;
        outerInvRadiusPositiveErrorBound = 0.541;
        outerInvRadiusNegativeErrorBound = 1.267;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 1920)
    {
        innerInvRadiusPositiveErrorBound = 0.184;
        innerInvRadiusNegativeErrorBound = 0.321;
        outerInvRadiusPositiveErrorBound = 0.239;
        outerInvRadiusNegativeErrorBound = 1.267;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 1920)
    {
        innerInvRadiusPositiveErrorBound = 0.378;
        innerInvRadiusNegativeErrorBound = 0.884;
        outerInvRadiusPositiveErrorBound = 0.507;
        outerInvRadiusNegativeErrorBound = 2.283;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 1920)
    {
        innerInvRadiusPositiveErrorBound = 0.445;
        innerInvRadiusNegativeErrorBound = 0.914;
        outerInvRadiusPositiveErrorBound = 0.559;
        outerInvRadiusNegativeErrorBound = 2.777;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 1920)
    {
        innerInvRadiusPositiveErrorBound = 0.431;
        innerInvRadiusNegativeErrorBound = 0.884;
        outerInvRadiusPositiveErrorBound = 0.507;
        outerInvRadiusNegativeErrorBound = 3.380;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 1922)
    {
        innerInvRadiusPositiveErrorBound = 0.093;
        innerInvRadiusNegativeErrorBound = 0.113;
        outerInvRadiusPositiveErrorBound = 1.008;
        outerInvRadiusNegativeErrorBound = 1.008;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 0 and layer_binary == 3840)
    {
        innerInvRadiusPositiveErrorBound = 0.491;
        innerInvRadiusNegativeErrorBound = 0.914;
        outerInvRadiusPositiveErrorBound = 0.638;
        outerInvRadiusNegativeErrorBound = 1.492;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 1 and layer_binary == 3840)
    {
        innerInvRadiusPositiveErrorBound = 0.491;
        innerInvRadiusNegativeErrorBound = 0.331;
        outerInvRadiusPositiveErrorBound = 0.203;
        outerInvRadiusNegativeErrorBound = 0.975;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 2 and layer_binary == 3840)
    {
        innerInvRadiusPositiveErrorBound = 0.445;
        innerInvRadiusNegativeErrorBound = 1.267;
        outerInvRadiusPositiveErrorBound = 0.524;
        outerInvRadiusNegativeErrorBound = 3.728;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 3 and layer_binary == 3840)
    {
        innerInvRadiusPositiveErrorBound = 0.491;
        innerInvRadiusNegativeErrorBound = 0.417;
        outerInvRadiusPositiveErrorBound = 0.524;
        outerInvRadiusNegativeErrorBound = 1.938;
    }
    else if(nLayerOverlaps == 2 and nHitOverlaps == 4 and layer_binary == 3840)
    {
        innerInvRadiusPositiveErrorBound = 0.445;
        innerInvRadiusNegativeErrorBound = 0.802;
        outerInvRadiusPositiveErrorBound = 0.524;
        outerInvRadiusNegativeErrorBound = 3.380;
    }
    else if(nLayerOverlaps == 1 and nHitOverlaps == 0 and layer_binary == 3968)
    {
        innerInvRadiusPositiveErrorBound = 0.311;
        innerInvRadiusNegativeErrorBound = 0.311;
        outerInvRadiusPositiveErrorBound = 0.638;
        outerInvRadiusNegativeErrorBound = 2.358;
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

    if(layer_binary == 30 && nLayerOverlaps == 2 and nHitOverlaps == 0)
    {
        return rPhiChiSquared < 0.062;
    }
    else if(layer_binary == 30 && nLayerOverlaps == 2 and nHitOverlaps == 1)
    {
        return false;
    }
    else if(layer_binary == 30 && nLayerOverlaps == 2 and nHitOverlaps == 2)
    {
        return rPhiChiSquared < 0.047;
    }
    else if(layer_binary == 30 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return false;
    }
    else if(layer_binary == 30 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return rPhiChiSquared < 0.062;
    }
    else if(layer_binary == 60 && nLayerOverlaps == 2 and nHitOverlaps == 0)
    {
        return false;
    }
    else if(layer_binary == 60 && nLayerOverlaps == 2 and nHitOverlaps == 1)
    {
        return false;
    }
    else if(layer_binary == 60 && nLayerOverlaps == 2 and nHitOverlaps == 2)
    {
        return false;
    }
    else if(layer_binary == 60 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return false;
    }
    else if(layer_binary == 60 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return false;
    }
    else if(layer_binary == 62 && nLayerOverlaps == 1 and nHitOverlaps == 0)
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
    else if(layer_binary == 120 && nLayerOverlaps == 2 and nHitOverlaps == 2)
    {
        return false;
    }
    else if(layer_binary == 120 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return false;
    }
    else if(layer_binary == 120 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return false;
    }
    else if(layer_binary == 124 && nLayerOverlaps == 1 and nHitOverlaps == 0)
    {
        return false;
    }
    else if(layer_binary == 142 && nLayerOverlaps == 2 and nHitOverlaps == 0)
    {
        return rPhiChiSquared < 0.015;
    }
    else if(layer_binary == 142 && nLayerOverlaps == 2 and nHitOverlaps == 1)
    {
        return false;
    }
    else if(layer_binary == 142 && nLayerOverlaps == 2 and nHitOverlaps == 2)
    {
        return rPhiChiSquared < 0.035;
    }
    else if(layer_binary == 142 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return false;
    }
    else if(layer_binary == 142 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return rPhiChiSquared < 0.062;
    }
    else if(layer_binary == 156 && nLayerOverlaps == 2 and nHitOverlaps == 0)
    {
        return false;
    }
    else if(layer_binary == 156 && nLayerOverlaps == 2 and nHitOverlaps == 1)
    {
        return false;
    }
    else if(layer_binary == 156 && nLayerOverlaps == 2 and nHitOverlaps == 2)
    {
        return false;
    }
    else if(layer_binary == 156 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return false;
    }
    else if(layer_binary == 156 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return false;
    }
    else if(layer_binary == 158 && nLayerOverlaps == 1 and nHitOverlaps == 0)
    {
        return false;
    }
    else if(layer_binary == 184 && nLayerOverlaps == 2 and nHitOverlaps == 0)
    {
        return false;
    }
    else if(layer_binary == 184 && nLayerOverlaps == 2 and nHitOverlaps == 1)
    {
        return false;
    }
    else if(layer_binary == 184 && nLayerOverlaps == 2 and nHitOverlaps == 2)
    {
        return false;
    }
    else if(layer_binary == 184 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return false;
    }
    else if(layer_binary == 184 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return false;
    }
    else if(layer_binary == 188 && nLayerOverlaps == 1 and nHitOverlaps == 0)
    {
        return false;
    }
    else if(layer_binary == 390 && nLayerOverlaps == 2 and nHitOverlaps == 0)
    {
        return rPhiChiSquared < 0.035;
    }
    else if(layer_binary == 390 && nLayerOverlaps == 2 and nHitOverlaps == 1)
    {
        return false;
    }
    else if(layer_binary == 390 && nLayerOverlaps == 2 and nHitOverlaps == 2)
    {
        return rPhiChiSquared < 0.035;
    }
    else if(layer_binary == 390 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return false;
    }
    else if(layer_binary == 390 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return rPhiChiSquared < 0.047;
    }
    else if(layer_binary == 396 && nLayerOverlaps == 2 and nHitOverlaps == 0)
    {
        return false;
    }
    else if(layer_binary == 396 && nLayerOverlaps == 2 and nHitOverlaps == 1)
    {
        return false;
    }
    else if(layer_binary == 396 && nLayerOverlaps == 2 and nHitOverlaps == 2)
    {
        return false;
    }
    else if(layer_binary == 396 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return false;
    }
    else if(layer_binary == 396 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return false;
    }
    else if(layer_binary == 398 && nLayerOverlaps == 1 and nHitOverlaps == 0)
    {
        return false;
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
        return false;
    }
    else if(layer_binary == 408 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return false;
    }
    else if(layer_binary == 408 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return false;
    }
    else if(layer_binary == 412 && nLayerOverlaps == 1 and nHitOverlaps == 0)
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
    else if(layer_binary == 432 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return false;
    }
    else if(layer_binary == 440 && nLayerOverlaps == 1 and nHitOverlaps == 0)
    {
        return false;
    }
    else if(layer_binary == 898 && nLayerOverlaps == 2 and nHitOverlaps == 0)
    {
        return false;
    }
    else if(layer_binary == 898 && nLayerOverlaps == 2 and nHitOverlaps == 1)
    {
        return false;
    }
    else if(layer_binary == 898 && nLayerOverlaps == 2 and nHitOverlaps == 2)
    {
        return rPhiChiSquared < 0.012;
    }
    else if(layer_binary == 898 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return false;
    }
    else if(layer_binary == 898 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return rPhiChiSquared < 0.020;
    }
    else if(layer_binary == 900 && nLayerOverlaps == 2 and nHitOverlaps == 0)
    {
        return false;
    }
    else if(layer_binary == 900 && nLayerOverlaps == 2 and nHitOverlaps == 1)
    {
        return false;
    }
    else if(layer_binary == 900 && nLayerOverlaps == 2 and nHitOverlaps == 2)
    {
        return false;
    }
    else if(layer_binary == 900 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return false;
    }
    else if(layer_binary == 900 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return false;
    }
    else if(layer_binary == 902 && nLayerOverlaps == 1 and nHitOverlaps == 0)
    {
        return false;
    }
    else if(layer_binary == 904 && nLayerOverlaps == 2 and nHitOverlaps == 0)
    {
        return false;
    }
    else if(layer_binary == 904 && nLayerOverlaps == 2 and nHitOverlaps == 1)
    {
        return rPhiChiSquared < 0.003;
    }
    else if(layer_binary == 904 && nLayerOverlaps == 2 and nHitOverlaps == 2)
    {
        return false;
    }
    else if(layer_binary == 904 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return false;
    }
    else if(layer_binary == 904 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return false;
    }
    else if(layer_binary == 908 && nLayerOverlaps == 1 and nHitOverlaps == 0)
    {
        return rPhiChiSquared < 0.047;
    }
    else if(layer_binary == 1920 && nLayerOverlaps == 2 and nHitOverlaps == 0)
    {
        return rPhiChiSquared < 0.007;
    }
    else if(layer_binary == 1920 && nLayerOverlaps == 2 and nHitOverlaps == 1)
    {
        return false;
    }
    else if(layer_binary == 1920 && nLayerOverlaps == 2 and nHitOverlaps == 2)
    {
        return rPhiChiSquared < 0.020;
    }
    else if(layer_binary == 1920 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return false;
    }
    else if(layer_binary == 1920 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return rPhiChiSquared < 0.035;
    }
    else if(layer_binary == 1922 && nLayerOverlaps == 1 and nHitOverlaps == 0)
    {
        return false;
    }
    else if(layer_binary == 1924 && nLayerOverlaps == 1 and nHitOverlaps == 0)
    {
        return false;
    }
    else if(layer_binary == 3840 && nLayerOverlaps == 2 and nHitOverlaps == 0)
    {
        return rPhiChiSquared < 0.015;
    }
    else if(layer_binary == 3840 && nLayerOverlaps == 2 and nHitOverlaps == 1)
    {
        return false;
    }
    else if(layer_binary == 3840 && nLayerOverlaps == 2 and nHitOverlaps == 2)
    {
        return false;
    }
    else if(layer_binary == 3840 && nLayerOverlaps == 2 and nHitOverlaps == 3)
    {
        return false;
    }
    else if(layer_binary == 3840 && nLayerOverlaps == 2 and nHitOverlaps == 4)
    {
        return rPhiChiSquared < 0.062;
    }
    else if(layer_binary == 3968 && nLayerOverlaps == 1 and nHitOverlaps == 0)
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
__device__ float SDL::computeTERZChiSquared(struct modules& modulesInGPU, struct hits& hitsInGPU, unsigned int* anchorObjectAnchorHitIndices, unsigned int* anchorLowerModuleIndices, unsigned int* outerObjectAnchorHitIndices, unsigned int* outerLowerModuleIndices, short anchorObjectType)
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
            unsigned int& lowerModuleIndex = outerLowerModuleIndices[i];
            rtAnchor = hitsInGPU.rts[anchorHitIndex];
            zAnchor = hitsInGPU.zs[anchorHitIndex];

            //outerModuleAnchorHitIndices
            const int moduleType = modulesInGPU.moduleType[lowerModuleIndex];
            const int moduleSide = modulesInGPU.sides[lowerModuleIndex];
            const int moduleLayerType = modulesInGPU.moduleLayerType[lowerModuleIndex];
            const int layer = modulesInGPU.layers[lowerModuleIndex] + 6 * (modulesInGPU.subdets[lowerModuleIndex] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex] == SDL::TwoS);
        
            residual = (layer <= 6) ?  zAnchor - (slope * rtAnchor + intercept) : rtAnchor - (zAnchor/slope + intercept/slope);
        
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

    float denominator = sigma1 * sigmaX2 - sigmaX * sigmaX;
    intercept = (sigmaX2 * sigmaY - sigmaX * sigmaXY) / denominator;
    slope = (sigmaXY - sigmaX * sigmaY) / denominator;
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


__device__ float SDL::computeT3T3RPhiChiSquared(struct modules& modulesInGPU, struct hits& hitsInGPU, int nPoints, unsigned int* anchorHitIndices, unsigned int* lowerModuleIndices, float& regressionRadius)
{
    float delta1[6], delta2[6], sigmas[6], slopes[6], xs[6], ys[6], g, f;
    bool isFlat[6];
    computeSigmasForRegression(modulesInGPU, lowerModuleIndices, delta1, delta2, slopes, isFlat, nPoints, true);

    for(size_t i = 0; i < nPoints; i++)
    {
        xs[i] = hitsInGPU.xs[anchorHitIndices[i]];
        ys[i] = hitsInGPU.ys[anchorHitIndices[i]];
    }
    float chiSquared;
    regressionRadius = computeRadiusUsingRegression(nPoints, xs, ys, delta1, delta2, slopes, isFlat, g, f, sigmas, chiSquared);
    return chiSquared;
}

__device__ float SDL::computeT3T3RZChiSquared(struct modules& modulesInGPU, struct hits& hitsInGPU, int nPoints, unsigned int* anchorHitIndices, unsigned int* lowerModuleIndices)
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
        const int moduleLayerType = modulesInGPU.moduleLayerType[lowerModuleIndex];
        const int layer = modulesInGPU.layers[lowerModuleIndex] + 6 * (modulesInGPU.subdets[lowerModuleIndex] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex] == SDL::TwoS); 
        residual = (layer <= 6) ?  zs[i] - (slope * rts[i] + intercept) : rts[i] - (zs[i]/slope + intercept/slope);
        if(moduleType == 0)
        {
            error = 0.15;
        }
        else
        {
            error = 5;
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

            error *= 1/sqrtf(1 + drdz * drdz);
        }
        RMSE += (residual * residual)/(error * error);
    }
    RMSE = sqrtf(RMSE/nPoints);
    return RMSE;
}

