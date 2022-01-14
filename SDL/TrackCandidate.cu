#include "TrackCandidate.cuh"

#include "allocate.h"


void SDL::trackCandidates::resetMemory(unsigned int maxTrackCandidates,cudaStream_t stream)
{
    cudaMemsetAsync(trackCandidateType,0, maxTrackCandidates * sizeof(short),stream);
    cudaMemsetAsync(objectIndices, 0,2 * maxTrackCandidates * sizeof(unsigned int),stream);
    cudaMemsetAsync(nTrackCandidates, 0,sizeof(unsigned int),stream);
    cudaMemsetAsync(nTrackCandidatespT3, 0,sizeof(unsigned int),stream);
    cudaMemsetAsync(nTrackCandidatesT5, 0,sizeof(unsigned int),stream);
    cudaMemsetAsync(nTrackCandidatespT5,0, sizeof(unsigned int),stream);
    cudaMemsetAsync(nTrackCandidatespLS, 0,sizeof(unsigned int),stream);

#ifdef TRACK_EXTENSIONS
    cudaMemsetAsync(logicalLayers, 0, 7 * maxTrackCandidates * sizeof(unsigned int), stream);
    cudaMemsetAsync(lowerModuleIndices, 0, 7 * maxTrackCandidates * sizeof(unsigned int), stream);
    cudaMemsetAsync(hitIndices, 0, 14 * maxTrackCandidates * sizeof(unsigned int), stream);
    cudaMemsetAsync(centerX, 0, maxTrackCandidates * sizeof(float), stream);
    cudaMemsetAsync(centerY, 0, maxTrackCandidates * sizeof(float), stream);
    cudaMemsetAsync(radius, 0, maxTrackCandidates * sizeof(float), stream);
    cudaMemsetAsync(partOfExtension, 0, maxTrackCandidates * sizeof(bool), stream);
#endif
}
void SDL::createTrackCandidatesInUnifiedMemory(struct trackCandidates& trackCandidatesInGPU, unsigned int maxTrackCandidates,cudaStream_t stream)
{
#ifdef CACHE_ALLOC
    trackCandidatesInGPU.trackCandidateType = (short*)cms::cuda::allocate_managed(maxTrackCandidates * sizeof(short),stream);
    trackCandidatesInGPU.objectIndices = (unsigned int*)cms::cuda::allocate_managed(maxTrackCandidates * 2*sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidates= (unsigned int*)cms::cuda::allocate_managed( sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidatespT3= (unsigned int*)cms::cuda::allocate_managed(sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidatesT5= (unsigned int*)cms::cuda::allocate_managed( sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidatespT5= (unsigned int*)cms::cuda::allocate_managed( sizeof(unsigned int),stream);

#ifdef TRACK_EXTENSIONS
    trackCandidatesInGPU.logicalLayers = (unsigned int*)cms::cuda::allocate_managed(7 * maxTrackCandidates * sizeof(unsigned int), stream);
    trackCandidatesInGPU.lowerModuleIndices = (unsigned int*)cms::cuda::allocate_managed(7 * maxTrackCandidates * sizeof(unsigned int), stream);
    trackCandidatesInGPU.hitIndices = (unsigned int*)cms::cuda::allocate_managed(14 * maxTrackCandidates * sizeof(unsigned int), stream);
    trackCandidatesInGPU.centerX = (float*)cms::cuda::allocate_managed(maxTrackCandidates * sizeof(float), stream);
    trackCandidatesInGPU.centerY = (float*)cms::cuda::allocate_managed(maxTrackCandidates * sizeof(float), stream);
    trackCandidatesInGPU.radius = (float*)cms::cuda::allocate_managed(maxTrackCandidates * sizeof(float), stream);
    trackCandidatesInGPU.partOfExtension = (bool*)cms::cuda::allocate_managed(maxTrackCandidates * sizeof(bool), stream);
#endif

    trackCandidatesInGPU.nTrackCandidatespLS= (unsigned int*)cms::cuda::allocate_managed( sizeof(unsigned int),stream);
#else
    cudaMallocManaged(&trackCandidatesInGPU.trackCandidateType, maxTrackCandidates * sizeof(short));
    cudaMallocManaged(&trackCandidatesInGPU.objectIndices, 2 * maxTrackCandidates * sizeof(unsigned int));
    cudaMallocManaged(&trackCandidatesInGPU.nTrackCandidates, sizeof(unsigned int));
    cudaMallocManaged(&trackCandidatesInGPU.nTrackCandidatespT3, sizeof(unsigned int));
    cudaMallocManaged(&trackCandidatesInGPU.nTrackCandidatesT5, sizeof(unsigned int));
    cudaMallocManaged(&trackCandidatesInGPU.nTrackCandidatespT5, sizeof(unsigned int));

    cudaMallocManaged(&trackCandidatesInGPU.nTrackCandidatespLS, sizeof(unsigned int));
#ifdef TRACK_EXTENSIONS
    cudaMallocManaged(&trackCandidatesInGPU.logicalLayers, maxTrackCandidates * 7 * sizeof(unsigned int));
    cudaMallocManaged(&trackCandidatesInGPU.lowerModuleIndices, maxTrackCandidates * 7 * sizeof(unsigned int));
    cudaMallocManaged(&trackCandidatesInGPU.hitIndices, maxTrackCandidates * 14 * sizeof(unsigned int));
    cudaMallocManaged(&trackCandidatesInGPU.partOfExtension, maxTrackCandidates*sizeof(bool));
    cudaMallocManaged(&trackCandidatesInGPU.centerX, maxTrackCandidates * sizeof(float));
    cudaMallocManaged(&trackCandidatesInGPU.centerY, maxTrackCandidates * sizeof(float));
    cudaMallocManaged(&trackCandidatesInGPU.radius, maxTrackCandidates * sizeof(float));
#endif
#endif
    cudaMemsetAsync(trackCandidatesInGPU.nTrackCandidates,0, sizeof(unsigned int),stream);
    cudaMemsetAsync(trackCandidatesInGPU.nTrackCandidatesT5,0, sizeof(unsigned int),stream);
    cudaMemsetAsync(trackCandidatesInGPU.nTrackCandidatespT3,0, sizeof(unsigned int),stream);
    cudaMemsetAsync(trackCandidatesInGPU.nTrackCandidatespT5,0, sizeof(unsigned int),stream);
    cudaMemsetAsync(trackCandidatesInGPU.nTrackCandidatespLS,0, sizeof(unsigned int),stream);
#ifdef TRACK_EXTENSIONS
    cudaMemsetAsync(trackCandidatesInGPU.partOfExtension, false, maxTrackCandidates * sizeof(bool));
#endif
}
void SDL::createTrackCandidatesInExplicitMemory(struct trackCandidates& trackCandidatesInGPU, unsigned int maxTrackCandidates,cudaStream_t stream)
{
#ifdef CACHE_ALLOC
    int dev;
    cudaGetDevice(&dev);
    trackCandidatesInGPU.trackCandidateType = (short*)cms::cuda::allocate_device(dev,maxTrackCandidates * sizeof(short),stream);
    trackCandidatesInGPU.objectIndices = (unsigned int*)cms::cuda::allocate_device(dev,maxTrackCandidates * 2*sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidates= (unsigned int*)cms::cuda::allocate_device(dev, sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidatespT3= (unsigned int*)cms::cuda::allocate_device(dev, sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidatesT5= (unsigned int*)cms::cuda::allocate_device(dev, sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidatespT5= (unsigned int*)cms::cuda::allocate_device(dev, sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidatespLS= (unsigned int*)cms::cuda::allocate_device(dev, sizeof(unsigned int),stream);

#ifdef TRACK_EXTENSIONS
    trackCandidatesInGPU.partOfExtension = (bool*)cms::cuda::allocate_device(dev, maxTrackCandidates * sizeof(bool), stream);
    trackCandidatesInGPU.logicalLayers = (unsigned int*)cms::cuda::allocate_device(dev, 7 * maxTrackCandidates * sizeof(unsigned int), stream);
    trackCandidatesInGPU.lowerModuleIndices = (unsigned int*)cms::cuda::allocate_device(dev, 7 * maxTrackCandidates * sizeof(unsigned int), stream);
    trackCandidatesInGPU.hitIndices = (unsigned int*)cms::cuda::allocate_device(dev, 14 * maxTrackCandidates * sizeof(unsigned int), stream);
    trackCandidatesInGPU.centerX = (float*)cms::cuda::allocate_device(dev, maxTrackCandidates * sizeof(float), stream);
    trackCandidatesInGPU.centerY = (float*)cms::cuda::allocate_device(dev, maxTrackCandidates * sizeof(float), stream);
    trackCandidatesInGPU.radius = (float*)cms::cuda::allocate_device(dev, maxTrackCandidates * sizeof(float), stream);
#endif

#else
    cudaMalloc(&trackCandidatesInGPU.trackCandidateType, maxTrackCandidates * sizeof(short));
    cudaMalloc(&trackCandidatesInGPU.objectIndices, 2 * maxTrackCandidates * sizeof(unsigned int));
    cudaMalloc(&trackCandidatesInGPU.nTrackCandidates, sizeof(unsigned int));
    cudaMalloc(&trackCandidatesInGPU.nTrackCandidatespT3, sizeof(unsigned int));
    cudaMalloc(&trackCandidatesInGPU.nTrackCandidatesT5, sizeof(unsigned int));
    cudaMalloc(&trackCandidatesInGPU.nTrackCandidatespT5, sizeof(unsigned int));
    cudaMalloc(&trackCandidatesInGPU.nTrackCandidatespLS, sizeof(unsigned int));

#ifdef TRACK_EXTENSIONS
    cudaMalloc(&trackCandidatesInGPU.partOfExtension, maxTrackCandidates * sizeof(bool));
    cudaMalloc(&trackCandidatesInGPU.logicalLayers, 7 * maxTrackCandidates * sizeof(unsigned int));
    cudaMalloc(&trackCandidatesInGPU.lowerModuleIndices, 7 * maxTrackCandidates * sizeof(unsigned int));
    cudaMalloc(&trackCandidatesInGPU.hitIndices, 14 * maxTrackCandidates * sizeof(unsigned int));
    cudaMalloc(&trackCandidatesInGPU.centerX, maxTrackCandidates * sizeof(float));
    cudaMalloc(&trackCandidatesInGPU.centerY, maxTrackCandidates * sizeof(float));
    cudaMalloc(&trackCandidatesInGPU.radius, maxTrackCandidates * sizeof(float));
#endif
#endif
    cudaMemsetAsync(trackCandidatesInGPU.nTrackCandidates,0, sizeof(unsigned int));
    cudaMemsetAsync(trackCandidatesInGPU.nTrackCandidatesT5,0, sizeof(unsigned int));
    cudaMemsetAsync(trackCandidatesInGPU.nTrackCandidatespT3,0, sizeof(unsigned int));
    cudaMemsetAsync(trackCandidatesInGPU.nTrackCandidatespT5,0, sizeof(unsigned int));
    cudaMemsetAsync(trackCandidatesInGPU.nTrackCandidatespLS,0, sizeof(unsigned int));
#ifdef TRACK_EXTENSIONS
    cudaMemsetAsync(trackCandidatesInGPU.partOfExtension, false, maxTrackCandidates * sizeof(bool));
#endif
    cudaStreamSynchronize(stream);
}

__device__ void SDL::addTrackCandidateToMemory(struct trackCandidates& trackCandidatesInGPU, short trackCandidateType, unsigned int innerTrackletIndex, unsigned int outerTrackletIndex, unsigned int trackCandidateIndex)
{
    trackCandidatesInGPU.trackCandidateType[trackCandidateIndex] = trackCandidateType;
    trackCandidatesInGPU.objectIndices[2 * trackCandidateIndex] = innerTrackletIndex;
    trackCandidatesInGPU.objectIndices[2 * trackCandidateIndex + 1] = outerTrackletIndex;
}

#ifdef TRACK_EXTENSIONS
__device__ void SDL::addTrackCandidateToMemory(struct trackCandidates& trackCandidatesInGPU, short trackCandidateType, unsigned int innerTrackletIndex, unsigned int outerTrackletIndex, unsigned int* logicalLayerIndices, unsigned int* lowerModuleIndices, unsigned int* hitIndices, float centerX, float centerY, float radius, unsigned int trackCandidateIndex)
{
    trackCandidatesInGPU.trackCandidateType[trackCandidateIndex] = trackCandidateType;
    trackCandidatesInGPU.objectIndices[2 * trackCandidateIndex] = innerTrackletIndex;
    trackCandidatesInGPU.objectIndices[2 * trackCandidateIndex + 1] = outerTrackletIndex;
    
    size_t limits = trackCandidateType == 7 ? 7 : 5;

    //send the starting pointer to the logicalLayer and hitIndices
    for(size_t i = 0; i < limits; i++)
    {
        trackCandidatesInGPU.logicalLayers[7 * trackCandidateIndex + i] = logicalLayerIndices[i];
        trackCandidatesInGPU.lowerModuleIndices[7 * trackCandidateIndex + i] = lowerModuleIndices[i];
    }
    for(size_t i = 0; i < 2 * limits; i++)
    {
        trackCandidatesInGPU.hitIndices[14 * trackCandidateIndex + i] = hitIndices[i];
    }
    trackCandidatesInGPU.centerX[trackCandidateIndex] = centerX;
    trackCandidatesInGPU.centerY[trackCandidateIndex] = centerY;
    trackCandidatesInGPU.radius[trackCandidateIndex] = radius;
}
#endif

SDL::trackCandidates::trackCandidates()
{
    trackCandidateType = nullptr;
    objectIndices = nullptr;
    nTrackCandidates = nullptr;
    nTrackCandidatesT5 = nullptr;
    nTrackCandidatespT3 = nullptr;
    nTrackCandidatespT5 = nullptr;
    nTrackCandidatespLS = nullptr;

#ifdef TRACK_EXTENSIONS
    logicalLayers = nullptr;
    hitIndices = nullptr;
    lowerModuleIndices = nullptr;
    partOfExtension = nullptr;
    centerX = nullptr;
    centerY = nullptr;
    radius = nullptr;
#endif
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
    //cudaFree(trackCandidateType);
    cms::cuda::free_device(dev,objectIndices);
    cms::cuda::free_device(dev,trackCandidateType);
    cms::cuda::free_device(dev,nTrackCandidates);
    cms::cuda::free_device(dev,nTrackCandidatespT3);
    cms::cuda::free_device(dev,nTrackCandidatesT5);
    cms::cuda::free_device(dev,nTrackCandidatespT5);
    cms::cuda::free_device(dev,nTrackCandidatespLS);

#ifdef TRACK_EXTENSIONS
    cms::cuda::free_device(dev, logicalLayers);
    cms::cuda::free_device(dev, hitIndices);
    cms::cuda::free_device(dev, lowerModuleIndices);
    cms::cuda::free_device(dev, centerX);
    cms::cuda::free_device(dev, centerY);
    cms::cuda::free_device(dev, radius);
    cms::cuda::free_device(dev, partOfExtension);
#endif
#else
    cms::cuda::free_managed(objectIndices);
    cms::cuda::free_managed(trackCandidateType);
    cms::cuda::free_managed(nTrackCandidates);
    cms::cuda::free_managed(nTrackCandidatespT3);
    cms::cuda::free_managed(nTrackCandidatesT5);
    cms::cuda::free_managed(nTrackCandidatespT5);
    cms::cuda::free_managed(nTrackCandidatespLS);

#ifdef TRACK_EXTENSIONS
    cms::cuda::free_managed(logicalLayers);
    cms::cuda::free_managed(hitIndices);
    cms::cuda::free_managed(lowerModuleIndices);
    cms::cuda::free_managed(centerX);
    cms::cuda::free_managed(centerY);
    cms::cuda::free_managed(radius);
    cms::cuda::free_managed(partOfExtension);
#endif
#endif
}
void SDL::trackCandidates::freeMemory(cudaStream_t stream)
{
    cudaFree(trackCandidateType);
    cudaFree(objectIndices);
    cudaFree(nTrackCandidates);
    cudaFree(nTrackCandidatespT3);
    cudaFree(nTrackCandidatesT5);
    cudaFree(nTrackCandidatespT5);
    cudaFree(nTrackCandidatespLS);

#ifdef TRACK_EXTENSIONS
    cudaFree(logicalLayers);
    cudaFree(hitIndices);
    cudaFree(lowerModuleIndices);
    cudaFree(partOfExtension);
    cudaFree(centerX);
    cudaFree(centerY);
    cudaFree(radius);
#endif
    
    cudaStreamSynchronize(stream);
}
