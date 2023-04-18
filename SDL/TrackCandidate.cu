#include "TrackCandidate.cuh"

void SDL::trackCandidates::resetMemory(unsigned int maxTrackCandidates,cudaStream_t stream)
{
    cudaMemsetAsync(trackCandidateType,0, maxTrackCandidates * sizeof(short),stream);
    cudaMemsetAsync(directObjectIndices, 0, maxTrackCandidates * sizeof(unsigned int),stream);
    cudaMemsetAsync(objectIndices, 0,2 * maxTrackCandidates * sizeof(unsigned int),stream);
    cudaMemsetAsync(nTrackCandidates, 0,sizeof(int),stream);
    cudaMemsetAsync(nTrackCandidatespT3, 0,sizeof(int),stream);
    cudaMemsetAsync(nTrackCandidatesT5, 0,sizeof(int),stream);
    cudaMemsetAsync(nTrackCandidatespT5,0, sizeof(int),stream);
    cudaMemsetAsync(nTrackCandidatespLS, 0,sizeof(int),stream);

    cudaMemsetAsync(logicalLayers, 0, 7 * maxTrackCandidates * sizeof(uint8_t), stream);
    cudaMemsetAsync(lowerModuleIndices, 0, 7 * maxTrackCandidates * sizeof(uint16_t), stream);
    cudaMemsetAsync(hitIndices, 0, 14 * maxTrackCandidates * sizeof(unsigned int), stream);
    cudaMemsetAsync(centerX, 0, maxTrackCandidates * sizeof(FPX), stream);
    cudaMemsetAsync(centerY, 0, maxTrackCandidates * sizeof(FPX), stream);
    cudaMemsetAsync(radius , 0, maxTrackCandidates * sizeof(FPX), stream);
}

void SDL::createTrackCandidatesInExplicitMemory(struct trackCandidates& trackCandidatesInGPU, unsigned int maxTrackCandidates,cudaStream_t stream)
{
#ifdef CACHE_ALLOC
    int dev;
    cudaGetDevice(&dev);
    trackCandidatesInGPU.trackCandidateType = (short*)cms::cuda::allocate_device(dev,maxTrackCandidates * sizeof(short),stream);
    trackCandidatesInGPU.directObjectIndices = (unsigned int*)cms::cuda::allocate_device(dev,maxTrackCandidates * sizeof(unsigned int),stream);
    trackCandidatesInGPU.objectIndices = (unsigned int*)cms::cuda::allocate_device(dev,maxTrackCandidates * 2*sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidates= (int*)cms::cuda::allocate_device(dev, sizeof(int),stream);
    trackCandidatesInGPU.nTrackCandidatespT3= (int*)cms::cuda::allocate_device(dev, sizeof(int),stream);
    trackCandidatesInGPU.nTrackCandidatesT5= (int*)cms::cuda::allocate_device(dev, sizeof(int),stream);
    trackCandidatesInGPU.nTrackCandidatespT5= (int*)cms::cuda::allocate_device(dev, sizeof(int),stream);
    trackCandidatesInGPU.nTrackCandidatespLS= (int*)cms::cuda::allocate_device(dev, sizeof(int),stream);

    trackCandidatesInGPU.logicalLayers = (uint8_t*)cms::cuda::allocate_device(dev, 7 * maxTrackCandidates * sizeof(uint8_t), stream);
    trackCandidatesInGPU.lowerModuleIndices = (uint16_t*)cms::cuda::allocate_device(dev, 7 * maxTrackCandidates * sizeof(uint16_t), stream);
    trackCandidatesInGPU.hitIndices = (unsigned int*)cms::cuda::allocate_device(dev, 14 * maxTrackCandidates * sizeof(unsigned int), stream);
    trackCandidatesInGPU.centerX = (FPX*)cms::cuda::allocate_device(dev, maxTrackCandidates * sizeof(FPX), stream);
    trackCandidatesInGPU.centerY = (FPX*)cms::cuda::allocate_device(dev, maxTrackCandidates * sizeof(FPX), stream);
    trackCandidatesInGPU.radius  = (FPX*)cms::cuda::allocate_device(dev, maxTrackCandidates * sizeof(FPX), stream);

#else
    cudaMalloc(&trackCandidatesInGPU.trackCandidateType, maxTrackCandidates * sizeof(short));
    cudaMalloc(&trackCandidatesInGPU.directObjectIndices, maxTrackCandidates * sizeof(unsigned int));
    cudaMalloc(&trackCandidatesInGPU.objectIndices, 2 * maxTrackCandidates * sizeof(unsigned int));
    cudaMalloc(&trackCandidatesInGPU.nTrackCandidates, sizeof(int));
    cudaMalloc(&trackCandidatesInGPU.nTrackCandidatespT3, sizeof(int));
    cudaMalloc(&trackCandidatesInGPU.nTrackCandidatesT5, sizeof(int));
    cudaMalloc(&trackCandidatesInGPU.nTrackCandidatespT5, sizeof(int));
    cudaMalloc(&trackCandidatesInGPU.nTrackCandidatespLS, sizeof(int));

    cudaMalloc(&trackCandidatesInGPU.logicalLayers, 7 * maxTrackCandidates * sizeof(uint8_t));
    cudaMalloc(&trackCandidatesInGPU.lowerModuleIndices, 7 * maxTrackCandidates * sizeof(uint16_t));
    cudaMalloc(&trackCandidatesInGPU.hitIndices, 14 * maxTrackCandidates * sizeof(unsigned int));
    cudaMalloc(&trackCandidatesInGPU.centerX, maxTrackCandidates * sizeof(FPX));
    cudaMalloc(&trackCandidatesInGPU.centerY, maxTrackCandidates * sizeof(FPX));
    cudaMalloc(&trackCandidatesInGPU.radius , maxTrackCandidates * sizeof(FPX));
#endif
    cudaMemsetAsync(trackCandidatesInGPU.nTrackCandidates,0, sizeof(int), stream);
    cudaMemsetAsync(trackCandidatesInGPU.nTrackCandidatesT5,0, sizeof(int), stream);
    cudaMemsetAsync(trackCandidatesInGPU.nTrackCandidatespT3,0, sizeof(int), stream);
    cudaMemsetAsync(trackCandidatesInGPU.nTrackCandidatespT5,0, sizeof(int), stream);
    cudaMemsetAsync(trackCandidatesInGPU.nTrackCandidatespLS,0, sizeof(int), stream);
    cudaMemsetAsync(trackCandidatesInGPU.logicalLayers, 0, 7 * maxTrackCandidates * sizeof(uint8_t), stream);
    cudaMemsetAsync(trackCandidatesInGPU.lowerModuleIndices, 0, 7 * maxTrackCandidates * sizeof(uint16_t), stream);
    cudaMemsetAsync(trackCandidatesInGPU.hitIndices, 0, 14 * maxTrackCandidates * sizeof(unsigned int), stream);
    cudaStreamSynchronize(stream);
}

SDL::trackCandidates::trackCandidates()
{
    trackCandidateType = nullptr;
    directObjectIndices = nullptr;
    objectIndices = nullptr;
    nTrackCandidates = nullptr;
    nTrackCandidatesT5 = nullptr;
    nTrackCandidatespT3 = nullptr;
    nTrackCandidatespT5 = nullptr;
    nTrackCandidatespLS = nullptr;

    logicalLayers = nullptr;
    hitIndices = nullptr;
    lowerModuleIndices = nullptr;
    centerX = nullptr;
    centerY = nullptr;
    radius = nullptr;
}

SDL::trackCandidates::~trackCandidates()
{
}

void SDL::trackCandidates::freeMemoryCache()
{
    int dev;
    cudaGetDevice(&dev);
    //FIXME
    //cudaFree(trackCandidateType);
    cms::cuda::free_device(dev,directObjectIndices);
    cms::cuda::free_device(dev,objectIndices);
    cms::cuda::free_device(dev,trackCandidateType);
    cms::cuda::free_device(dev,nTrackCandidates);
    cms::cuda::free_device(dev,nTrackCandidatespT3);
    cms::cuda::free_device(dev,nTrackCandidatesT5);
    cms::cuda::free_device(dev,nTrackCandidatespT5);
    cms::cuda::free_device(dev,nTrackCandidatespLS);

    cms::cuda::free_device(dev, logicalLayers);
    cms::cuda::free_device(dev, hitIndices);
    cms::cuda::free_device(dev, lowerModuleIndices);
    cms::cuda::free_device(dev, centerX);
    cms::cuda::free_device(dev, centerY);
    cms::cuda::free_device(dev, radius);
}

void SDL::trackCandidates::freeMemory(cudaStream_t stream)
{
    cudaFree(trackCandidateType);
    cudaFree(directObjectIndices);
    cudaFree(objectIndices);
    cudaFree(nTrackCandidates);
    cudaFree(nTrackCandidatespT3);
    cudaFree(nTrackCandidatesT5);
    cudaFree(nTrackCandidatespT5);
    cudaFree(nTrackCandidatespLS);

    cudaFree(logicalLayers);
    cudaFree(hitIndices);
    cudaFree(lowerModuleIndices);
    cudaFree(centerX);
    cudaFree(centerY);
    cudaFree(radius);
    
    cudaStreamSynchronize(stream);
}
