#include "TrackCandidate.cuh"

#include "allocate.h"


void SDL::createTrackCandidatesInUnifiedMemory(struct trackCandidates& trackCandidatesInGPU, unsigned int maxTrackCandidates)
{
#ifdef CACHE_ALLOC
    cudaStream_t stream=0;
    trackCandidatesInGPU.trackCandidateType = (short*)cms::cuda::allocate_managed(maxTrackCandidates * sizeof(short),stream);
    cudaMallocManaged(&trackCandidatesInGPU.objectIndices, 2 * maxTrackCandidates * sizeof(unsigned int));
    trackCandidatesInGPU.nTrackCandidates= (unsigned int*)cms::cuda::allocate_managed( sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidatespT3= (unsigned int*)cms::cuda::allocate_managed(sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidatesT5= (unsigned int*)cms::cuda::allocate_managed( sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidatespT5= (unsigned int*)cms::cuda::allocate_managed( sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidatespLS= (unsigned int*)cms::cuda::allocate_managed( sizeof(unsigned int),stream);

#else
    cudaMallocManaged(&trackCandidatesInGPU.trackCandidateType, maxTrackCandidates * sizeof(short));
    cudaMallocManaged(&trackCandidatesInGPU.objectIndices, 2 * maxTrackCandidates * sizeof(unsigned int));
    cudaMallocManaged(&trackCandidatesInGPU.nTrackCandidates, sizeof(unsigned int));
    cudaMallocManaged(&trackCandidatesInGPU.nTrackCandidatespT3, sizeof(unsigned int));
    cudaMallocManaged(&trackCandidatesInGPU.nTrackCandidatesT5, sizeof(unsigned int));
    cudaMallocManaged(&trackCandidatesInGPU.nTrackCandidatespT5, sizeof(unsigned int));

    cudaMallocManaged(&trackCandidatesInGPU.nTrackCandidatespLS, sizeof(unsigned int));
#endif

    cudaMemset(trackCandidatesInGPU.nTrackCandidates,0, sizeof(unsigned int));
    cudaMemset(trackCandidatesInGPU.nTrackCandidatesT5,0, sizeof(unsigned int));
    cudaMemset(trackCandidatesInGPU.nTrackCandidatespT3,0, sizeof(unsigned int));
    cudaMemset(trackCandidatesInGPU.nTrackCandidatespT5,0, sizeof(unsigned int));
    cudaMemset(trackCandidatesInGPU.nTrackCandidatespLS,0, sizeof(unsigned int));
}
void SDL::createTrackCandidatesInExplicitMemory(struct trackCandidates& trackCandidatesInGPU, unsigned int maxTrackCandidates)
{
#ifdef CACHE_ALLOC
    cudaStream_t stream=0;
    int dev;
    cudaGetDevice(&dev);
    //TODO 
    cudaMalloc(&trackCandidatesInGPU.trackCandidateType, maxTrackCandidates * sizeof(short));
    cudaMalloc(&trackCandidatesInGPU.objectIndices, 2 * maxTrackCandidates * sizeof(unsigned int));
    trackCandidatesInGPU.nTrackCandidates= (unsigned int*)cms::cuda::allocate_device(dev, sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidatespT3= (unsigned int*)cms::cuda::allocate_device(dev, sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidatesT5= (unsigned int*)cms::cuda::allocate_device(dev, sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidatespT5= (unsigned int*)cms::cuda::allocate_device(dev, sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidatespLS= (unsigned int*)cms::cuda::allocate_device(dev, sizeof(unsigned int),stream);
    trackCandidatesInGPU.partOfExtension = (bool*)cms::cuda::allocate_device(dev, maxTrackCandidates * sizeof(bool), stream);

#else
    cudaMalloc(&trackCandidatesInGPU.trackCandidateType, maxTrackCandidates * sizeof(short));
    cudaMalloc(&trackCandidatesInGPU.objectIndices, 2 * maxTrackCandidates * sizeof(unsigned int));
    cudaMalloc(&trackCandidatesInGPU.nTrackCandidates, sizeof(unsigned int));
    cudaMalloc(&trackCandidatesInGPU.nTrackCandidatespT3, sizeof(unsigned int));
    cudaMalloc(&trackCandidatesInGPU.nTrackCandidatesT5, sizeof(unsigned int));
    cudaMalloc(&trackCandidatesInGPU.nTrackCandidatespT5, sizeof(unsigned int));
    cudaMalloc(&trackCandidatesInGPU.nTrackCandidatespLS, sizeof(unsigned int));
#endif
    cudaMemset(trackCandidatesInGPU.nTrackCandidates,0, sizeof(unsigned int));
    cudaMemset(trackCandidatesInGPU.nTrackCandidatesT5,0, sizeof(unsigned int));
    cudaMemset(trackCandidatesInGPU.nTrackCandidatespT3,0, sizeof(unsigned int));
    cudaMemset(trackCandidatesInGPU.nTrackCandidatespT5,0, sizeof(unsigned int));
    cudaMemset(trackCandidatesInGPU.nTrackCandidatespLS,0, sizeof(unsigned int));
}

__device__ void SDL::addTrackCandidateToMemory(struct trackCandidates& trackCandidatesInGPU, short trackCandidateType, unsigned int innerTrackletIndex, unsigned int outerTrackletIndex, unsigned int trackCandidateIndex)
{
    trackCandidatesInGPU.trackCandidateType[trackCandidateIndex] = trackCandidateType;
    trackCandidatesInGPU.objectIndices[2 * trackCandidateIndex] = innerTrackletIndex;
    trackCandidatesInGPU.objectIndices[2 * trackCandidateIndex + 1] = outerTrackletIndex;
}

SDL::trackCandidates::trackCandidates()
{
    trackCandidateType = nullptr;
    objectIndices = nullptr;
    nTrackCandidates = nullptr;
    nTrackCandidatesT5 = nullptr;
    nTrackCandidatespT3 = nullptr;
    nTrackCandidatespT5 = nullptr;
    nTrackCandidatespLS = nullptr;
}

SDL::trackCandidates::~trackCandidates()
{
}

void SDL::trackCandidates::freeMemoryCache()
{
#ifdef Explicit_Track
    int dev;
    cudaGetDevice(&dev);
    //FIXME
    cudaFree(trackCandidateType);
    cms::cuda::free_device(dev,nTrackCandidates);
    cms::cuda::free_device(dev,nTrackCandidatespT3);
    cms::cuda::free_device(dev,nTrackCandidatesT5);
    cms::cuda::free_device(dev,nTrackCandidatespT5);


    cms::cuda::free_device(dev,nTrackCandidatespLS);
#else
    cms::cuda::free_managed(trackCandidateType);
    cms::cuda::free_managed(nTrackCandidates);
    cms::cuda::free_managed(nTrackCandidatespT3);
    cms::cuda::free_managed(nTrackCandidatesT5);
    cms::cuda::free_managed(nTrackCandidatespT5);
    cms::cuda::free_managed(nTrackCandidatespLS);
#endif
    cudaFree(objectIndices);

}
void SDL::trackCandidates::freeMemory()
{
    cudaFree(trackCandidateType);
    cudaFree(objectIndices);
    cudaFree(nTrackCandidates);
    cudaFree(nTrackCandidatespT3);
    cudaFree(nTrackCandidatesT5);
    cudaFree(nTrackCandidatespT5);
    cudaFree(nTrackCandidatespLS);
}

