#include "TrackCandidate.cuh"

#include "allocate.h"


void SDL::createEligibleModulesListForTrackCandidates(struct modules& modulesInGPU, unsigned int& nEligibleModules, unsigned int maxTrackCandidates)
{
    //an extra array in the modulesInGPU struct that will provide us with starting indices for the memory locations. If a
    //module is not supposed to have any memory, it gets a -1

    //the array will be filled in createTrackCandidatesInUnfiedMemory

    unsigned int nLowerModules = *modulesInGPU.nLowerModules;
    cudaMemset(modulesInGPU.trackCandidateModuleIndices, -1, sizeof(int) * (nLowerModules + 1));

    //start filling
    for(size_t i = 0; i <= nLowerModules; i++)
    {
        //condition for a track candidate to exist for a module
        //TCs don't exist for layers 5 and 6 barrel, and layers 2,3,4,5 endcap
        unsigned int idx = modulesInGPU.lowerModuleIndices[i];
        if((modulesInGPU.subdets[idx] == SDL::Barrel and modulesInGPU.layers[idx] < 5) or (modulesInGPU.subdets[idx] == SDL::Endcap and modulesInGPU.layers[idx] == 1) or modulesInGPU.subdets[idx] == SDL::InnerPixel)
        {
            modulesInGPU.trackCandidateModuleIndices[i] = nEligibleModules * maxTrackCandidates;
            nEligibleModules++;
        }
    }
}


void SDL::createTrackCandidatesInUnifiedMemory(struct trackCandidates& trackCandidatesInGPU, unsigned int maxTrackCandidates, unsigned int maxPixelTrackCandidates, unsigned int nLowerModules, unsigned int nEligibleModules)
{
    unsigned int nMemoryLocations = maxTrackCandidates * (nEligibleModules-1) + maxPixelTrackCandidates;
    std::cout<<"Number of eligible modules = "<<nEligibleModules<<std::endl;
#ifdef CACHE_ALLOC
    cudaStream_t stream=0;
    trackCandidatesInGPU.trackCandidateType = (short*)cms::cuda::allocate_managed(nMemoryLocations * sizeof(short),stream);
    //trackCandidatesInGPU.objectIndices = (unsigned int*)cms::cuda::allocate_managed(2 * nMemoryLocations * sizeof(unsigned int),stream);
    cudaMallocManaged(&trackCandidatesInGPU.objectIndices, 2 * nMemoryLocations * sizeof(unsigned int));
    trackCandidatesInGPU.nTrackCandidates= (unsigned int*)cms::cuda::allocate_managed( nLowerModules * sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidatesT4T4= (unsigned int*)cms::cuda::allocate_managed( nLowerModules * sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidatesT4T3= (unsigned int*)cms::cuda::allocate_managed( nLowerModules * sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidatesT3T4= (unsigned int*)cms::cuda::allocate_managed( nLowerModules * sizeof(unsigned int),stream);
#else
    cudaMallocManaged(&trackCandidatesInGPU.trackCandidateType, nMemoryLocations * sizeof(short));
    cudaMallocManaged(&trackCandidatesInGPU.objectIndices, 2 * nMemoryLocations * sizeof(unsigned int));
    cudaMallocManaged(&trackCandidatesInGPU.nTrackCandidates, nLowerModules * sizeof(unsigned int));
    cudaMallocManaged(&trackCandidatesInGPU.nTrackCandidatesT4T4, nLowerModules * sizeof(unsigned int));
    cudaMallocManaged(&trackCandidatesInGPU.nTrackCandidatesT4T3, nLowerModules * sizeof(unsigned int));
    cudaMallocManaged(&trackCandidatesInGPU.nTrackCandidatesT3T4, nLowerModules * sizeof(unsigned int));
#endif

#pragma omp parallel for
    for(size_t i = 0; i<nLowerModules;i++)
    {
        trackCandidatesInGPU.nTrackCandidates[i] = 0;
        trackCandidatesInGPU.nTrackCandidatesT4T4[i] = 0;
        trackCandidatesInGPU.nTrackCandidatesT4T3[i] = 0;
        trackCandidatesInGPU.nTrackCandidatesT3T4[i] = 0;
    }
}
void SDL::createTrackCandidatesInExplicitMemory(struct trackCandidates& trackCandidatesInGPU, unsigned int maxTrackCandidates, unsigned int maxPixelTrackCandidates, unsigned int nLowerModules ,unsigned int nEligibleModules)
{
    unsigned int nMemoryLocations = maxTrackCandidates * (nEligibleModules-1) + maxPixelTrackCandidates;
    std::cout<<"Number of eligible modules = "<<nEligibleModules<<std::endl;
#ifdef CACHE_ALLOC
    cudaStream_t stream=0;
    int dev;
    cudaGetDevice(&dev);
    //trackCandidatesInGPU.trackCandidateType = (short*)cms::cuda::allocate_device(dev,nMemoryLocations * sizeof(short),stream);
    cudaMalloc(&trackCandidatesInGPU.trackCandidateType, nMemoryLocations * sizeof(short));
    //trackCandidatesInGPU.objectIndices = (unsigned int*)cms::cuda::allocate_device(dev,2 * nMemoryLocations * sizeof(unsigned int),stream); // too big to cache
    cudaMalloc(&trackCandidatesInGPU.objectIndices, 2 * nMemoryLocations * sizeof(unsigned int));
    trackCandidatesInGPU.nTrackCandidates= (unsigned int*)cms::cuda::allocate_device(dev, nLowerModules * sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidatesT4T4= (unsigned int*)cms::cuda::allocate_device(dev, nLowerModules * sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidatesT4T3= (unsigned int*)cms::cuda::allocate_device(dev, nLowerModules * sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidatesT3T4= (unsigned int*)cms::cuda::allocate_device(dev, nLowerModules * sizeof(unsigned int),stream);

#else
    cudaMalloc(&trackCandidatesInGPU.trackCandidateType, nMemoryLocations * sizeof(short));
    cudaMalloc(&trackCandidatesInGPU.objectIndices, 2 * nMemoryLocations * sizeof(unsigned int));
    cudaMalloc(&trackCandidatesInGPU.nTrackCandidates, nLowerModules * sizeof(unsigned int));
    cudaMalloc(&trackCandidatesInGPU.nTrackCandidatesT4T4, nLowerModules * sizeof(unsigned int));
    cudaMalloc(&trackCandidatesInGPU.nTrackCandidatesT4T3, nLowerModules * sizeof(unsigned int));
    cudaMalloc(&trackCandidatesInGPU.nTrackCandidatesT3T4, nLowerModules * sizeof(unsigned int));
#endif
    cudaMemset(trackCandidatesInGPU.nTrackCandidates,0, nLowerModules * sizeof(unsigned int));
    cudaMemset(trackCandidatesInGPU.nTrackCandidatesT4T4,0, nLowerModules * sizeof(unsigned int));
    cudaMemset(trackCandidatesInGPU.nTrackCandidatesT4T3,0, nLowerModules * sizeof(unsigned int));
    cudaMemset(trackCandidatesInGPU.nTrackCandidatesT3T4,0, nLowerModules * sizeof(unsigned int));

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

void SDL::trackCandidates::freeMemoryCache()
{
#ifdef Explicit_Track
    int dev;
    cudaGetDevice(&dev);
    //cms::cuda::free_device(dev,trackCandidateType);
    cudaFree(trackCandidateType);
    //cms::cuda::free_device(dev,objectIndices);
    cms::cuda::free_device(dev,nTrackCandidates);
    cms::cuda::free_device(dev,nTrackCandidatesT4T4);
    cms::cuda::free_device(dev,nTrackCandidatesT4T3);
    cms::cuda::free_device(dev,nTrackCandidatesT3T4);
#else
    cms::cuda::free_managed(trackCandidateType);
    //cms::cuda::free_managed(objectIndices);
    cms::cuda::free_managed(nTrackCandidates);
    cms::cuda::free_managed(nTrackCandidatesT4T4);
    cms::cuda::free_managed(nTrackCandidatesT4T3);
    cms::cuda::free_managed(nTrackCandidatesT3T4);

#endif
    cudaFree(objectIndices);

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


