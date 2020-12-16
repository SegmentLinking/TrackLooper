#include "TrackCandidate.cuh"

void SDL::createTrackCandidatesInUnifiedMemory(struct trackCandidates& trackCandidatesInGPU, unsigned int maxTrackCandidates, unsigned int maxPixelTrackCandidates, unsigned int nLowerModules)
{
    unsigned int nMemoryLocations = maxTrackCandidates * nLowerModules + maxPixelTrackCandidates;
    nLowerModules += 1;
    cudaMallocManaged(&trackCandidatesInGPU.trackCandidateType, nMemoryLocations * sizeof(short));
    cudaMallocManaged(&trackCandidatesInGPU.objectIndices, 2 * nMemoryLocations * sizeof(unsigned int));
    cudaMallocManaged(&trackCandidatesInGPU.nTrackCandidates, nLowerModules * sizeof(unsigned int));
    cudaMallocManaged(&trackCandidatesInGPU.nTrackCandidatesT4T4, nLowerModules * sizeof(unsigned int));
    cudaMallocManaged(&trackCandidatesInGPU.nTrackCandidatesT4T3, nLowerModules * sizeof(unsigned int));
    cudaMallocManaged(&trackCandidatesInGPU.nTrackCandidatesT3T4, nLowerModules * sizeof(unsigned int));

#pragma omp parallel for
    for(size_t i = 0; i<nLowerModules;i++)
    {
        trackCandidatesInGPU.nTrackCandidates[i] = 0;
        trackCandidatesInGPU.nTrackCandidatesT4T4[i] = 0;
        trackCandidatesInGPU.nTrackCandidatesT4T3[i] = 0;
        trackCandidatesInGPU.nTrackCandidatesT3T4[i] = 0;
    }
}

__device__ void SDL::addTrackCandidateToMemory(struct trackCandidates& trackCandidatesInGPU, short trackCandidateType, unsigned int innerTrackletIndex, unsigned int outerTrackletIndex, unsigned int trackCandidateIndex)
{
    trackCandidatesInGPU.trackCandidateType[trackCandidateIndex] = trackCandidateType;
    trackCandidatesInGPU.objectIndices[2 * trackCandidateIndex] = innerTrackletIndex;
    trackCandidatesInGPU.objectIndices[2 * trackCandidateIndex + 1] = outerTrackletIndex;
}

__device__ bool SDL::runTrackCandidateDefaultAlgoTwoTracklets(struct tracklets& trackletsInGPU, struct triplets& tripletsInGPU, unsigned int innerTrackletIndex, unsigned int outerTrackletIndex, short& trackCandidateType)
{
    bool pass = true;
    trackCandidateType = 0;
    if(not hasCommonSegment(trackletsInGPU, tripletsInGPU, innerTrackletIndex, outerTrackletIndex, trackCandidateType))
    {
        pass = false;
    }
    return pass;
}

SDL::trackCandidates::trackCandidates()
{
    trackCandidateType = nullptr;
    objectIndices = nullptr;
    nTrackCandidates = nullptr;
}

SDL::trackCandidates::~trackCandidates()
{
}

void SDL::trackCandidates::freeMemory()
{
    cudaFree(trackCandidateType);
    cudaFree(objectIndices);
    cudaFree(nTrackCandidates);
    cudaFree(nTrackCandidatesT4T4);
    cudaFree(nTrackCandidatesT4T3);
    cudaFree(nTrackCandidatesT3T4);
}

__device__ bool SDL::runTrackCandidateDefaultAlgoTrackletToTriplet(struct tracklets& trackletsInGPU, struct triplets& tripletsInGPU, unsigned int innerTrackletIndex, unsigned int outerTripletIndex, short& trackCandidateType)
{
    bool pass = true;
    trackCandidateType = 1;
    if(not hasCommonSegment(trackletsInGPU, tripletsInGPU, innerTrackletIndex, outerTripletIndex, trackCandidateType))
    {
        pass = false;
    }

    return pass;
}

__device__ bool SDL::runTrackCandidateDefaultAlgoTripletToTracklet(struct tracklets& trackletsInGPU, struct triplets& tripletsInGPU, unsigned int innerTripletIndex, unsigned int outerTrackletIndex, short& trackCandidateType)
{
    bool pass = true;
    trackCandidateType = 2;
    if(not hasCommonSegment(trackletsInGPU, tripletsInGPU, innerTripletIndex, outerTrackletIndex, trackCandidateType))
    {
        pass = false;
    }
    return pass;
}

__device__ bool SDL::hasCommonSegment(struct tracklets& trackletsInGPU, struct triplets& tripletsInGPU, unsigned int innerObjectIndex, unsigned int outerObjectIndex, short trackCandidateType)
{
    unsigned int innerObjectOuterSegmentIndex, outerObjectInnerSegmentIndex;

    if(trackCandidateType == 0)
    {
        //2 tracklets
        innerObjectOuterSegmentIndex = trackletsInGPU.segmentIndices[2 * innerObjectIndex + 1];
        outerObjectInnerSegmentIndex = trackletsInGPU.segmentIndices[2 * outerObjectIndex];
    }
    else if(trackCandidateType == 1)
    {
        //T4T3
        innerObjectOuterSegmentIndex = trackletsInGPU.segmentIndices[2 * innerObjectIndex + 1];
        outerObjectInnerSegmentIndex = tripletsInGPU.segmentIndices[2 * outerObjectIndex];
    }
    else if(trackCandidateType == 2)
    {
        //T3T4
        innerObjectOuterSegmentIndex = tripletsInGPU.segmentIndices[2 * innerObjectIndex + 1];
        outerObjectInnerSegmentIndex = trackletsInGPU.segmentIndices[2 * outerObjectIndex];
    }

    return (innerObjectOuterSegmentIndex == outerObjectInnerSegmentIndex);
}


