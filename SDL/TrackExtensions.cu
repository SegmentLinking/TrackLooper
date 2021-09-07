# include "TrackExtensions.cuh"

SDL::trackExtensions::trackExtensions()
{
    constituentTCTypes = nullptr;
    constituentTCIndices = nullptr;
    nTrackExtensions = nullptr;
}

SDL::trackExtensions::~trackExtensions()
{
}

void SDL::trackExtensions::freeMemory()
{
    cudaFree(constituentTCTypes);
    cudaFree(constituentTCIndices);
    cudaFree(nTrackExtensions);
}

void SDL::createTrackExtensionsInUnifiedMemory(struct trackExtensions& trackExtensionsInGPU, unsigned int maxTrackExtensions)
{
    cudaMallocManaged(&trackExtensionsInGPU.constituentTCTypes, sizeof(short) * 3 * maxTrackExtensions);
    cudaMallocManaged(&trackExtensionsInGPU.constituentTCIndices, sizeof(unsigned int) * 3 * maxTrackExtensions);
    cudaMallocManaged(&trackExtensionsInGPU.nTrackExtensions, sizeof(unsigned int));

    cudaMemset(trackExtensionsInGPU.nTrackExtensions, 0, sizeof(unsigned int));
}

void SDL::createTrackExtensionsInExplicitMemory(struct trackExtensions& trackExtensionsInGPU, unsigned int maxTrackExtensions)
{
    cudaMalloc(&trackExtensionsInGPU.constituentTCTypes, sizeof(short) * 3 * maxTrackExtensions);
    cudaMalloc(&trackExtensionsInGPU.constituentTCIndices, sizeof(unsigned int) * 3 * maxTrackExtensions);
    cudaMalloc(&trackExtensionsInGPU.nTrackExtensions, sizeof(unsigned int));
    cudaMemset(trackExtensionsInGPU.nTrackExtensions, 0, sizeof(unsigned int));

}

__device__ void SDL::addTrackExtensionToMemory(struct trackExtensions& trackExtensionsInGPU, short* constituentTCType, unsigned int* constituentTCIndex, unsigned int trackExtensionIndex)
{
    
    for(size_t i = 0; i < 3 ; i++)
    {
        trackExtensionsInGPU.constituentTCTypes[3 * trackExtensionIndex + i] = constituentTCType[i];
        trackExtensionsInGPU.constituentTCIndices[3 * trackExtensionIndex + i] = constituentTCIndex[i];
    }
}

__device__ bool SDL::runTrackExtensionDefaultAlgo(struct modules& modulesInGPU, struct triplets& tripletsInGPU, struct trackCandidates& trackCandidatesInGPU, unsigned int anchorObjectIndex, unsigned int outerObjectIndex, short anchorObjectType, short outerObjectType, unsigned int layerOverlapTarget, unsigned int hitOverlapTarget, short* constituentTCType, unsigned int* constituentTCIndex)
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
    unsigned int* outerObjectLayerIndices = nullptr;
    unsigned int* outerObjectHitIndices = nullptr;

    unsigned int nAnchorLayers = (anchorObjectType == 7) ? 7 : (anchorObjectType == 3 ? 3 : 5);
    if(anchorObjectType != 3) //mostly this
    {
        
        anchorLayerIndices = &trackCandidatesInGPU.logicalLayers[nAnchorLayers * anchorObjectIndex];
        anchorHitIndices = &trackCandidatesInGPU.hitIndices[2 * nAnchorLayers * anchorObjectIndex];
    }
    else //outlier
    {
        anchorLayerIndices = &tripletsInGPU.logicalLayers[3 * anchorObjectIndex];
        anchorHitIndices = &tripletsInGPU.hitIndices[6 * anchorObjectIndex];
    }

    unsigned int nOuterLayers =(outerObjectType == 7) ? 7 : (outerObjectType == 3 ? 3 : 5); 

    if(outerObjectType == 3) //mostly this
    {
        outerObjectLayerIndices = &tripletsInGPU.logicalLayers[3 * outerObjectIndex];
        outerObjectHitIndices = &tripletsInGPU.hitIndices[6 * outerObjectIndex];
    }
    else //outlier
    {
        outerObjectLayerIndices = &trackCandidatesInGPU.logicalLayers[nOuterLayers * outerObjectIndex];
        outerObjectHitIndices = &trackCandidatesInGPU.hitIndices[2 * nOuterLayers * outerObjectIndex];
    }

    unsigned int nLayerOverlap(0), nHitOverlap(0);
    computeLayerAndHitOverlaps(anchorLayerIndices, anchorHitIndices, outerObjectLayerIndices, outerObjectHitIndices, nAnchorLayers, nOuterLayers, nLayerOverlap, nHitOverlap);

    //will add chi squared cuts later!
    pass = pass & (nLayerOverlap == layerOverlapTarget) & (nHitOverlap == hitOverlapTarget);
    printf("layeroverlap = %d, hitoverlap = %d\n", nLayerOverlap, nHitOverlap);
    constituentTCType[0] = anchorObjectType;
    constituentTCType[1] = outerObjectType;

    constituentTCIndex[0] = anchorObjectIndex;
    constituentTCIndex[1] = outerObjectIndex;

    return pass;
}

__device__ void SDL::computeLayerAndHitOverlaps(unsigned int* anchorLayerIndices, unsigned int* anchorHitIndices, unsigned int* outerObjectLayerIndices, unsigned int* outerObjectHitIndices, unsigned int nAnchorLayers, unsigned int nOuterLayers, unsigned int& nLayerOverlap, unsigned int& nHitOverlap)
{
    //merge technique!
    size_t j = 0; //outer object tracker
    for(size_t i = 0; i < nAnchorLayers; i++)
    {
        if(anchorLayerIndices[i] == outerObjectLayerIndices[j])
        {
            //2*i and 2*i + 1 are the hits, similarly 2*j and 2*j+1
            nLayerOverlap++;
            if(anchorHitIndices[2 * i] == outerObjectHitIndices[2 * j])
            {
                nHitOverlap++;
            }
            if(anchorHitIndices[2*i+1] == outerObjectHitIndices[2*j+1])
            {
                nHitOverlap++;
            }
            j++;
        }
    }
}
