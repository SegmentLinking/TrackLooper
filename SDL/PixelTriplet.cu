#include "PixelTriplet.cuh"

SDL::pixelTriplets::pixelTriplets()
{
    pixelSegmentIndices = nullptr;
    tripletIndices = nullptr;
    nPixelTriplets = nullptr;
    totOccupancyPixelTriplets = nullptr;
    pixelRadius = nullptr;
    tripletRadius = nullptr;
    pt = nullptr;
    isDup = nullptr;
    partOfPT5 = nullptr;
    centerX = nullptr;
    centerY = nullptr;
    hitIndices = nullptr;
    lowerModuleIndices = nullptr;
    logicalLayers = nullptr;
    rzChiSquared = nullptr;
    rPhiChiSquared = nullptr;
    rPhiChiSquaredInwards = nullptr;
}

void SDL::pixelTriplets::freeMemoryCache()
{
    int dev;
    cudaGetDevice(&dev);
    cms::cuda::free_device(dev,pixelSegmentIndices);
    cms::cuda::free_device(dev,tripletIndices);
    cms::cuda::free_device(dev,nPixelTriplets);
    cms::cuda::free_device(dev,totOccupancyPixelTriplets);
    cms::cuda::free_device(dev,pixelRadius);
    cms::cuda::free_device(dev,tripletRadius);
    cms::cuda::free_device(dev,pt);
    cms::cuda::free_device(dev,isDup);
    cms::cuda::free_device(dev,partOfPT5);
    cms::cuda::free_device(dev, centerX);
    cms::cuda::free_device(dev, centerY);
    cms::cuda::free_device(dev, hitIndices);
    cms::cuda::free_device(dev, logicalLayers);
    cms::cuda::free_device(dev, lowerModuleIndices);
    cms::cuda::free_device(dev, rPhiChiSquared);
    cms::cuda::free_device(dev, rPhiChiSquaredInwards);
    cms::cuda::free_device(dev, rzChiSquared);
}

void SDL::pixelTriplets::freeMemory(cudaStream_t stream)
{
    cudaFree(pixelSegmentIndices);
    cudaFree(tripletIndices);
    cudaFree(nPixelTriplets);
    cudaFree(totOccupancyPixelTriplets);
    cudaFree(pixelRadius);
    cudaFree(tripletRadius);
    cudaFree(pt);
    cudaFree(isDup);
    cudaFree(partOfPT5);
    cudaFree(centerX);
    cudaFree(centerY);
    cudaFree(logicalLayers);
    cudaFree(hitIndices);
    cudaFree(lowerModuleIndices);
    cudaFree(rPhiChiSquared);
    cudaFree(rPhiChiSquaredInwards);
    cudaFree(rzChiSquared);
}

SDL::pixelTriplets::~pixelTriplets()
{
}

void SDL::createPixelTripletsInExplicitMemory(struct pixelTriplets& pixelTripletsInGPU, unsigned int maxPixelTriplets, cudaStream_t stream)
{
#ifdef CACHE_ALLOC
    int dev;
    cudaGetDevice(&dev);
    pixelTripletsInGPU.pixelSegmentIndices       =(unsigned int*)cms::cuda::allocate_device(dev,maxPixelTriplets * sizeof(unsigned int),stream);
    pixelTripletsInGPU.tripletIndices            =(unsigned int*)cms::cuda::allocate_device(dev,maxPixelTriplets * sizeof(unsigned int),stream);
    pixelTripletsInGPU.nPixelTriplets            =(int*)cms::cuda::allocate_device(dev,sizeof(int),stream);
    pixelTripletsInGPU.totOccupancyPixelTriplets =(int*)cms::cuda::allocate_device(dev,sizeof(int),stream);
    pixelTripletsInGPU.pixelRadius               =(FPX*)cms::cuda::allocate_device(dev,maxPixelTriplets * sizeof(FPX),stream);
    pixelTripletsInGPU.tripletRadius             =(FPX*)cms::cuda::allocate_device(dev,maxPixelTriplets * sizeof(FPX),stream);
    pixelTripletsInGPU.pt                        =(FPX*)cms::cuda::allocate_device(dev,maxPixelTriplets * 6*sizeof(FPX),stream);
    pixelTripletsInGPU.isDup                     =(bool*)cms::cuda::allocate_device(dev,maxPixelTriplets * sizeof(bool),stream);
    pixelTripletsInGPU.partOfPT5                 =(bool*)cms::cuda::allocate_device(dev,maxPixelTriplets * sizeof(bool),stream);
    pixelTripletsInGPU.centerX                   = (FPX*)cms::cuda::allocate_device(dev, maxPixelTriplets * sizeof(FPX), stream);
    pixelTripletsInGPU.centerY                   = (FPX*)cms::cuda::allocate_device(dev, maxPixelTriplets * sizeof(FPX), stream);
    pixelTripletsInGPU.lowerModuleIndices        = (uint16_t*)cms::cuda::allocate_device(dev, maxPixelTriplets * sizeof(uint16_t) * 5, stream);
    pixelTripletsInGPU.hitIndices                = (unsigned int*)cms::cuda::allocate_device(dev, maxPixelTriplets * sizeof(unsigned int) * 10, stream);
    pixelTripletsInGPU.logicalLayers             = (uint8_t*)cms::cuda::allocate_device(dev, maxPixelTriplets * sizeof(uint8_t) * 5, stream);

    pixelTripletsInGPU.rPhiChiSquared = (float*)cms::cuda::allocate_device(dev, maxPixelTriplets * sizeof(float), stream);
    pixelTripletsInGPU.rPhiChiSquaredInwards = (float*)cms::cuda::allocate_device(dev, maxPixelTriplets * sizeof(float), stream);
    pixelTripletsInGPU.rzChiSquared = (float*)cms::cuda::allocate_device(dev, maxPixelTriplets * sizeof(float), stream);
#else
    cudaMalloc(&pixelTripletsInGPU.pixelSegmentIndices, maxPixelTriplets * sizeof(unsigned int));
    cudaMalloc(&pixelTripletsInGPU.tripletIndices, maxPixelTriplets * sizeof(unsigned int));
    cudaMalloc(&pixelTripletsInGPU.nPixelTriplets, sizeof(int));
    cudaMalloc(&pixelTripletsInGPU.totOccupancyPixelTriplets, sizeof(int));
    cudaMalloc(&pixelTripletsInGPU.pixelRadius, maxPixelTriplets * sizeof(FPX));
    cudaMalloc(&pixelTripletsInGPU.tripletRadius, maxPixelTriplets * sizeof(FPX));
    cudaMalloc(&pixelTripletsInGPU.pt, maxPixelTriplets * 6*sizeof(FPX));
    cudaMalloc(&pixelTripletsInGPU.isDup, maxPixelTriplets * sizeof(bool));
    cudaMalloc(&pixelTripletsInGPU.partOfPT5, maxPixelTriplets * sizeof(bool));
    cudaMalloc(&pixelTripletsInGPU.centerX, maxPixelTriplets * sizeof(FPX));
    cudaMalloc(&pixelTripletsInGPU.centerY, maxPixelTriplets * sizeof(FPX));
    cudaMalloc(&pixelTripletsInGPU.logicalLayers, maxPixelTriplets * sizeof(uint8_t) * 5);
    cudaMalloc(&pixelTripletsInGPU.hitIndices, maxPixelTriplets * sizeof(unsigned int) * 10);
    cudaMalloc(&pixelTripletsInGPU.lowerModuleIndices, maxPixelTriplets * sizeof(uint16_t) * 5);
    cudaMalloc(&pixelTripletsInGPU.rPhiChiSquared, maxPixelTriplets * sizeof(float));
    cudaMalloc(&pixelTripletsInGPU.rPhiChiSquaredInwards, maxPixelTriplets * sizeof(float));
    cudaMalloc(&pixelTripletsInGPU.rzChiSquared, maxPixelTriplets * sizeof(float));
#endif
    cudaMemsetAsync(pixelTripletsInGPU.nPixelTriplets, 0, sizeof(int),stream);
    cudaMemsetAsync(pixelTripletsInGPU.totOccupancyPixelTriplets, 0, sizeof(int),stream);
    cudaMemsetAsync(pixelTripletsInGPU.partOfPT5, 0, maxPixelTriplets*sizeof(bool),stream);
    cudaStreamSynchronize(stream);

    pixelTripletsInGPU.eta = pixelTripletsInGPU.pt + maxPixelTriplets;
    pixelTripletsInGPU.phi = pixelTripletsInGPU.pt + maxPixelTriplets * 2;
    pixelTripletsInGPU.eta_pix = pixelTripletsInGPU.pt + maxPixelTriplets *3;
    pixelTripletsInGPU.phi_pix = pixelTripletsInGPU.pt + maxPixelTriplets * 4;
    pixelTripletsInGPU.score = pixelTripletsInGPU.pt + maxPixelTriplets * 5;
}

SDL::pixelQuintuplets::pixelQuintuplets()
{
    pixelIndices = nullptr;
    T5Indices = nullptr;
    nPixelQuintuplets = nullptr;
    totOccupancyPixelQuintuplets = nullptr;
    isDup = nullptr;
    score = nullptr;
    pixelRadius = nullptr;
    quintupletRadius = nullptr;
    centerX = nullptr;
    centerY = nullptr;
    logicalLayers = nullptr;
    hitIndices = nullptr;
    lowerModuleIndices = nullptr;
}

SDL::pixelQuintuplets::~pixelQuintuplets()
{
}

void SDL::pixelQuintuplets::freeMemoryCache()
{
    int dev;
    cudaGetDevice(&dev);
    cms::cuda::free_device(dev,pixelIndices);
    cms::cuda::free_device(dev,T5Indices);
    cms::cuda::free_device(dev,nPixelQuintuplets);
    cms::cuda::free_device(dev,totOccupancyPixelQuintuplets);
    cms::cuda::free_device(dev,isDup);
    cms::cuda::free_device(dev,score);
    cms::cuda::free_device(dev,eta);
    cms::cuda::free_device(dev,phi);
    cms::cuda::free_device(dev, hitIndices);
    cms::cuda::free_device(dev, logicalLayers);
    cms::cuda::free_device(dev, lowerModuleIndices);
    cms::cuda::free_device(dev, centerX);
    cms::cuda::free_device(dev, centerY);
    cms::cuda::free_device(dev, pixelRadius);
    cms::cuda::free_device(dev, quintupletRadius);
    cms::cuda::free_device(dev, rzChiSquared);
    cms::cuda::free_device(dev, rPhiChiSquared);
    cms::cuda::free_device(dev, rPhiChiSquaredInwards);
}

void SDL::pixelQuintuplets::freeMemory(cudaStream_t stream)
{
    cudaFree(pixelIndices);
    cudaFree(T5Indices);
    cudaFree(nPixelQuintuplets);
    cudaFree(totOccupancyPixelQuintuplets);
    cudaFree(isDup);
    cudaFree(score);
    cudaFree(eta);
    cudaFree(phi);

    cudaFree(logicalLayers);
    cudaFree(hitIndices);
    cudaFree(lowerModuleIndices);
    cudaFree(pixelRadius);
    cudaFree(quintupletRadius);
    cudaFree(centerX);
    cudaFree(centerY);
    cudaFree(rzChiSquared);
    cudaFree(rPhiChiSquared);
    cudaFree(rPhiChiSquaredInwards);
    cudaStreamSynchronize(stream);
}

void SDL::createPixelQuintupletsInExplicitMemory(struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, unsigned int maxPixelQuintuplets,cudaStream_t stream)
{
#ifdef CACHE_ALLOC
    int dev;
    cudaGetDevice(&dev);
    pixelQuintupletsInGPU.pixelIndices        = (unsigned int*)cms::cuda::allocate_device(dev,maxPixelQuintuplets * sizeof(unsigned int),stream);
    pixelQuintupletsInGPU.T5Indices           = (unsigned int*)cms::cuda::allocate_device(dev,maxPixelQuintuplets * sizeof(unsigned int),stream);
    pixelQuintupletsInGPU.nPixelQuintuplets   = (int*)cms::cuda::allocate_device(dev,sizeof(int),stream);
    pixelQuintupletsInGPU.totOccupancyPixelQuintuplets   = (int*)cms::cuda::allocate_device(dev,sizeof(unsigned int),stream);
    pixelQuintupletsInGPU.isDup               = (bool*)cms::cuda::allocate_device(dev,maxPixelQuintuplets * sizeof(bool),stream);
    pixelQuintupletsInGPU.score               = (FPX*)cms::cuda::allocate_device(dev,maxPixelQuintuplets * sizeof(FPX),stream);
    pixelQuintupletsInGPU.eta                 = (FPX*)cms::cuda::allocate_device(dev,maxPixelQuintuplets * sizeof(FPX),stream);
    pixelQuintupletsInGPU.phi                 = (FPX*)cms::cuda::allocate_device(dev,maxPixelQuintuplets * sizeof(FPX),stream);
    pixelQuintupletsInGPU.hitIndices = (unsigned int*)cms::cuda::allocate_device(dev, maxPixelQuintuplets * 14 * sizeof(unsigned int), stream);
    pixelQuintupletsInGPU.logicalLayers = (uint8_t*)cms::cuda::allocate_device(dev, maxPixelQuintuplets * 7 * sizeof(uint8_t), stream);
    pixelQuintupletsInGPU.lowerModuleIndices = (uint16_t*)cms::cuda::allocate_device(dev, maxPixelQuintuplets * 7 * sizeof(uint16_t), stream);
    pixelQuintupletsInGPU.centerX          = (FPX*)cms::cuda::allocate_device(dev, maxPixelQuintuplets * sizeof(FPX), stream);
    pixelQuintupletsInGPU.centerY          = (FPX*)cms::cuda::allocate_device(dev, maxPixelQuintuplets * sizeof(FPX), stream);
    pixelQuintupletsInGPU.pixelRadius      = (FPX*)cms::cuda::allocate_device(dev, maxPixelQuintuplets * sizeof(FPX), stream);
    pixelQuintupletsInGPU.quintupletRadius = (FPX*)cms::cuda::allocate_device(dev, maxPixelQuintuplets * sizeof(FPX), stream);
    pixelQuintupletsInGPU.rzChiSquared          = (float*)cms::cuda::allocate_device(dev, maxPixelQuintuplets * sizeof(float), stream);
    pixelQuintupletsInGPU.rPhiChiSquared      = (float*)cms::cuda::allocate_device(dev, maxPixelQuintuplets * sizeof(float), stream);
    pixelQuintupletsInGPU.rPhiChiSquaredInwards = (float*)cms::cuda::allocate_device(dev, maxPixelQuintuplets * sizeof(float), stream);
#else
    cudaMalloc(&pixelQuintupletsInGPU.pixelIndices, maxPixelQuintuplets * sizeof(unsigned int));
    cudaMalloc(&pixelQuintupletsInGPU.T5Indices, maxPixelQuintuplets * sizeof(unsigned int));
    cudaMalloc(&pixelQuintupletsInGPU.nPixelQuintuplets, sizeof(int));
    cudaMalloc(&pixelQuintupletsInGPU.totOccupancyPixelQuintuplets, sizeof(int));
    cudaMalloc(&pixelQuintupletsInGPU.isDup, maxPixelQuintuplets * sizeof(bool));
    cudaMalloc(&pixelQuintupletsInGPU.score, maxPixelQuintuplets * sizeof(FPX));
    cudaMalloc(&pixelQuintupletsInGPU.eta  , maxPixelQuintuplets * sizeof(FPX));
    cudaMalloc(&pixelQuintupletsInGPU.phi  , maxPixelQuintuplets * sizeof(FPX));

    cudaMalloc(&pixelQuintupletsInGPU.logicalLayers, maxPixelQuintuplets * 7 *sizeof(uint8_t));
    cudaMalloc(&pixelQuintupletsInGPU.hitIndices, maxPixelQuintuplets * 14 * sizeof(unsigned int));
    cudaMalloc(&pixelQuintupletsInGPU.lowerModuleIndices, maxPixelQuintuplets * 7 * sizeof(uint16_t));
    cudaMalloc(&pixelQuintupletsInGPU.pixelRadius, maxPixelQuintuplets * sizeof(FPX));
    cudaMalloc(&pixelQuintupletsInGPU.quintupletRadius, maxPixelQuintuplets * sizeof(FPX));
    cudaMalloc(&pixelQuintupletsInGPU.centerX, maxPixelQuintuplets * sizeof(FPX));
    cudaMalloc(&pixelQuintupletsInGPU.centerY, maxPixelQuintuplets * sizeof(FPX));
    cudaMalloc(&pixelQuintupletsInGPU.rzChiSquared, maxPixelQuintuplets * sizeof(unsigned int));
    cudaMalloc(&pixelQuintupletsInGPU.rPhiChiSquared, maxPixelQuintuplets * sizeof(unsigned int));
    cudaMalloc(&pixelQuintupletsInGPU.rPhiChiSquaredInwards, maxPixelQuintuplets * sizeof(unsigned int));
#endif
    cudaMemsetAsync(pixelQuintupletsInGPU.nPixelQuintuplets, 0, sizeof(int),stream);
    cudaMemsetAsync(pixelQuintupletsInGPU.totOccupancyPixelQuintuplets, 0, sizeof(int),stream);
    cudaStreamSynchronize(stream);
}
