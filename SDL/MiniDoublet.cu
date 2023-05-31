#include "MiniDoublet.cuh"

//FIXME:Add memory locations for the pixel MDs here!
void SDL::createMDsInExplicitMemory(struct miniDoublets& mdsInGPU, unsigned int nMemoryLocations, uint16_t nLowerModules, unsigned int maxPixelMDs,cudaStream_t stream)
{
#ifdef CACHE_ALLOC
    int dev;
    cudaGetDevice(&dev);
    mdsInGPU.anchorHitIndices = (unsigned int*)cms::cuda::allocate_device(dev,nMemoryLocations * 2 * sizeof(unsigned int), stream);
    mdsInGPU.moduleIndices = (uint16_t*)cms::cuda::allocate_device(dev, nMemoryLocations * sizeof(uint16_t), stream);
    mdsInGPU.dphichanges = (float*)cms::cuda::allocate_device(dev,nMemoryLocations*9*sizeof(float),stream);
    mdsInGPU.nMDs = (int*)cms::cuda::allocate_device(dev, (nLowerModules + 1) *sizeof(int),stream);
    mdsInGPU.totOccupancyMDs = (int*)cms::cuda::allocate_device(dev, (nLowerModules + 1) *sizeof(int),stream);
    mdsInGPU.anchorX = (float*)cms::cuda::allocate_device(dev, nMemoryLocations * 6 * sizeof(float), stream);
    mdsInGPU.anchorHighEdgeX = (float*)cms::cuda::allocate_device(dev, nMemoryLocations * 4 * sizeof(float), stream);
    mdsInGPU.outerX = (float*)cms::cuda::allocate_device(dev, nMemoryLocations * 6 * sizeof(float), stream);
    mdsInGPU.outerHighEdgeX = (float*)cms::cuda::allocate_device(dev, nMemoryLocations * 4 * sizeof(float), stream);
    mdsInGPU.nMemoryLocations = (unsigned int*)cms::cuda::allocate_device(dev, sizeof(unsigned int), stream);
#else
    cudaMalloc(&mdsInGPU.anchorHitIndices, nMemoryLocations * 2 * sizeof(unsigned int));
    cudaMalloc(&mdsInGPU.moduleIndices, nMemoryLocations * sizeof(uint16_t));
    cudaMalloc(&mdsInGPU.dphichanges, nMemoryLocations *9* sizeof(float));
    cudaMalloc(&mdsInGPU.nMDs, (nLowerModules + 1) * sizeof(int)); 
    cudaMalloc(&mdsInGPU.totOccupancyMDs, (nLowerModules + 1) * sizeof(int)); 
    cudaMalloc(&mdsInGPU.anchorX, nMemoryLocations * 6 * sizeof(float));
    cudaMalloc(&mdsInGPU.anchorHighEdgeX, nMemoryLocations * 4 * sizeof(float));
    cudaMalloc(&mdsInGPU.outerX, nMemoryLocations * 6 * sizeof(float));
    cudaMalloc(&mdsInGPU.outerHighEdgeX, nMemoryLocations * 4 * sizeof(float));
    cudaMalloc(&mdsInGPU.nMemoryLocations, sizeof(unsigned int));
#endif
    cudaMemsetAsync(mdsInGPU.nMDs,0, (nLowerModules + 1) *sizeof(int),stream);
    cudaMemsetAsync(mdsInGPU.totOccupancyMDs,0, (nLowerModules + 1) *sizeof(int),stream);
    cudaStreamSynchronize(stream);

    mdsInGPU.outerHitIndices = mdsInGPU.anchorHitIndices + nMemoryLocations;
    mdsInGPU.dzs  = mdsInGPU.dphichanges + nMemoryLocations;
    mdsInGPU.dphis  = mdsInGPU.dphichanges + 2*nMemoryLocations;
    mdsInGPU.shiftedXs  = mdsInGPU.dphichanges + 3*nMemoryLocations;
    mdsInGPU.shiftedYs  = mdsInGPU.dphichanges + 4*nMemoryLocations;
    mdsInGPU.shiftedZs  = mdsInGPU.dphichanges + 5*nMemoryLocations;
    mdsInGPU.noShiftedDzs  = mdsInGPU.dphichanges + 6*nMemoryLocations;
    mdsInGPU.noShiftedDphis  = mdsInGPU.dphichanges + 7*nMemoryLocations;
    mdsInGPU.noShiftedDphiChanges  = mdsInGPU.dphichanges + 8*nMemoryLocations;

    mdsInGPU.anchorY = mdsInGPU.anchorX + nMemoryLocations;
    mdsInGPU.anchorZ = mdsInGPU.anchorX + 2 * nMemoryLocations;
    mdsInGPU.anchorRt = mdsInGPU.anchorX + 3 * nMemoryLocations;
    mdsInGPU.anchorPhi = mdsInGPU.anchorX + 4 * nMemoryLocations;
    mdsInGPU.anchorEta = mdsInGPU.anchorX + 5 * nMemoryLocations;

    mdsInGPU.anchorHighEdgeY = mdsInGPU.anchorHighEdgeX + nMemoryLocations;
    mdsInGPU.anchorLowEdgeX = mdsInGPU.anchorHighEdgeX + 2 * nMemoryLocations;
    mdsInGPU.anchorLowEdgeY = mdsInGPU.anchorHighEdgeX + 3 * nMemoryLocations;

    mdsInGPU.outerY = mdsInGPU.outerX + nMemoryLocations;
    mdsInGPU.outerZ = mdsInGPU.outerX + 2 * nMemoryLocations;
    mdsInGPU.outerRt = mdsInGPU.outerX + 3 * nMemoryLocations;
    mdsInGPU.outerPhi = mdsInGPU.outerX + 4 * nMemoryLocations;
    mdsInGPU.outerEta = mdsInGPU.outerX + 5 * nMemoryLocations;

    mdsInGPU.outerHighEdgeY = mdsInGPU.outerHighEdgeX + nMemoryLocations;
    mdsInGPU.outerLowEdgeX = mdsInGPU.outerHighEdgeX + 2 * nMemoryLocations;
    mdsInGPU.outerLowEdgeY = mdsInGPU.outerHighEdgeX + 3 * nMemoryLocations;
}

SDL::miniDoublets::miniDoublets()
{
    anchorHitIndices = nullptr;
    outerHitIndices = nullptr;
    moduleIndices = nullptr;
    nMDs = nullptr;
    totOccupancyMDs = nullptr;
    dphichanges = nullptr;

    dzs = nullptr;
    dphis = nullptr;

    shiftedXs = nullptr;
    shiftedYs = nullptr;
    shiftedZs = nullptr;
    noShiftedDzs = nullptr;
    noShiftedDphis = nullptr;
    noShiftedDphiChanges = nullptr;
    
    anchorX = nullptr;
    anchorY = nullptr;
    anchorZ = nullptr;
    anchorRt = nullptr;
    anchorPhi = nullptr;
    anchorEta = nullptr;
    anchorHighEdgeX = nullptr;
    anchorHighEdgeY = nullptr;
    anchorLowEdgeX = nullptr;
    anchorLowEdgeY = nullptr;
    outerX = nullptr;
    outerY = nullptr;
    outerZ = nullptr;
    outerRt = nullptr;
    outerPhi = nullptr;
    outerEta = nullptr;
    outerHighEdgeX = nullptr;
    outerHighEdgeY = nullptr;
    outerLowEdgeX = nullptr;
    outerLowEdgeY = nullptr;
}

SDL::miniDoublets::~miniDoublets()
{
}

void SDL::miniDoublets::freeMemoryCache()
{
    int dev;
    cudaGetDevice(&dev);
    cms::cuda::free_device(dev,anchorHitIndices);
    cms::cuda::free_device(dev, moduleIndices);
    cms::cuda::free_device(dev,dphichanges);
    cms::cuda::free_device(dev,nMDs);
    cms::cuda::free_device(dev,totOccupancyMDs);
    cms::cuda::free_device(dev, anchorX);
    cms::cuda::free_device(dev, anchorHighEdgeX);
    cms::cuda::free_device(dev, outerX);
    cms::cuda::free_device(dev, outerHighEdgeX);
    cms::cuda::free_device(dev, nMemoryLocations);
}

void SDL::miniDoublets::freeMemory(cudaStream_t stream)
{
    cudaFree(anchorHitIndices);
    cudaFree(moduleIndices);
    cudaFree(nMDs);
    cudaFree(totOccupancyMDs);
    cudaFree(dphichanges);
    cudaFree(anchorX);
    cudaFree(anchorHighEdgeX);
    cudaFree(outerX);
    cudaFree(outerHighEdgeX);
    cudaFree(nMemoryLocations);
}
