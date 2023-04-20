#include "Triplet.cuh"

void SDL::triplets::resetMemory(unsigned int maxTriplets, unsigned int nLowerModules,cudaStream_t stream)
{
    cudaMemsetAsync(segmentIndices,0, 5 * maxTriplets * sizeof(unsigned int),stream);
    cudaMemsetAsync(nTriplets,0, nLowerModules * sizeof(unsigned int),stream);
    cudaMemsetAsync(totOccupancyTriplets,0, nLowerModules * sizeof(unsigned int),stream);
    cudaMemsetAsync(betaIn,0, maxTriplets * 3 * sizeof(FPX),stream);
    cudaMemsetAsync(partOfPT5,0, maxTriplets * sizeof(bool),stream);
    cudaMemsetAsync(partOfT5,0, maxTriplets * sizeof(bool), stream);
    cudaMemsetAsync(partOfPT3, 0, maxTriplets * sizeof(bool), stream);
}

__global__ void SDL::createTripletArrayRanges(struct modules& modulesInGPU, struct objectRanges& rangesInGPU, struct segments& segmentsInGPU)
{

    short module_subdets;
    short module_layers;
    short module_rings;
    float module_eta;
    __shared__ unsigned int nTotalTriplets;
    nTotalTriplets = 0; //start!   
    __syncthreads();

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int np = gridDim.x * blockDim.x;
    for(uint16_t i = gid; i < *modulesInGPU.nLowerModules; i+= np)
    {
        if(segmentsInGPU.nSegments[i] == 0){
          rangesInGPU.tripletModuleIndices[i] = nTotalTriplets;
          rangesInGPU.tripletModuleOccupancy[i] = 0;
          continue;
        }
        module_subdets = modulesInGPU.subdets[i];
        module_layers = modulesInGPU.layers[i];
        module_rings = modulesInGPU.rings[i];
        module_eta = abs(modulesInGPU.eta[i]);
        unsigned int occupancy;
        unsigned int category_number, eta_number;
        if (module_layers<=3 && module_subdets==5) category_number = 0;
        else if (module_layers>=4 && module_subdets==5) category_number = 1;
        else if (module_layers<=2 && module_subdets==4 && module_rings>=11) category_number = 2;
        else if (module_layers>=3 && module_subdets==4 && module_rings>=8) category_number = 2;
        else if (module_layers<=2 && module_subdets==4 && module_rings<=10) category_number = 3;
        else if (module_layers>=3 && module_subdets==4 && module_rings<=7) category_number = 3;
        if (module_eta<0.75) eta_number=0;
        else if (module_eta>0.75 && module_eta<1.5) eta_number=1;
        else if (module_eta>1.5 && module_eta<2.25) eta_number=2;
        else if (module_eta>2.25 && module_eta<3) eta_number=3;

        if (category_number == 0 && eta_number == 0) occupancy = 543;
        else if (category_number == 0 && eta_number == 1) occupancy = 235;
        else if (category_number == 0 && eta_number == 2) occupancy = 88;
        else if (category_number == 0 && eta_number == 3) occupancy = 46;
        else if (category_number == 1 && eta_number == 0) occupancy = 755;
        else if (category_number == 1 && eta_number == 1) occupancy = 347;
        else if (category_number == 2 && eta_number == 1) occupancy = 0;
        else if (category_number == 2 && eta_number == 2) occupancy = 0;
        else if (category_number == 3 && eta_number == 1) occupancy = 38;
        else if (category_number == 3 && eta_number == 2) occupancy = 46;
        else if (category_number == 3 && eta_number == 3) occupancy = 39;

        rangesInGPU.tripletModuleOccupancy[i] = occupancy;
        unsigned int nTotT = atomicAdd(&nTotalTriplets,occupancy);
        rangesInGPU.tripletModuleIndices[i] = nTotT;
    }
    __syncthreads();
    if(threadIdx.x==0){
      *rangesInGPU.device_nTotalTrips = nTotalTriplets;
    }
}

void SDL::createTripletsInExplicitMemory(struct triplets& tripletsInGPU, unsigned int maxTriplets, uint16_t nLowerModules, cudaStream_t stream)
{
#ifdef CACHE_ALLOC
    int dev;
    cudaGetDevice(&dev);
    tripletsInGPU.segmentIndices = (unsigned int*)cms::cuda::allocate_device(dev,maxTriplets * sizeof(unsigned int) *2,stream);
    tripletsInGPU.lowerModuleIndices = (uint16_t*)cms::cuda::allocate_device(dev,maxTriplets * sizeof(uint16_t) *3,stream);
    tripletsInGPU.betaIn = (FPX*)cms::cuda::allocate_device(dev,maxTriplets * sizeof(FPX) *3,stream);
    tripletsInGPU.nTriplets = (int*)cms::cuda::allocate_device(dev,nLowerModules * sizeof(int),stream);
    tripletsInGPU.totOccupancyTriplets = (int*)cms::cuda::allocate_device(dev,nLowerModules * sizeof(int),stream);
    tripletsInGPU.partOfPT5 = (bool*)cms::cuda::allocate_device(dev, maxTriplets * sizeof(bool), stream);
    tripletsInGPU.partOfPT3 = (bool*)cms::cuda::allocate_device(dev, maxTriplets * sizeof(bool), stream);
    tripletsInGPU.partOfT5 = (bool*)cms::cuda::allocate_device(dev, maxTriplets * sizeof(bool), stream);

    tripletsInGPU.logicalLayers = (uint8_t*)cms::cuda::allocate_device(dev, maxTriplets * 3 * sizeof(uint8_t), stream);
    tripletsInGPU.hitIndices = (unsigned int*)cms::cuda::allocate_device(dev, maxTriplets * 6 * sizeof(unsigned int), stream);
    tripletsInGPU.nMemoryLocations = (unsigned int*)cms::cuda::allocate_device(dev, sizeof(unsigned int), stream);

#ifdef CUT_VALUE_DEBUG
    tripletsInGPU.zOut = (float*)cms::cuda::allocate_device(dev, maxTriplets * 4 * sizeof(float), stream);
    tripletsInGPU.zLo = (float*)cms::cuda::allocate_device(dev, maxTriplets * sizeof(float), stream);
    tripletsInGPU.zHi = (float*)cms::cuda::allocate_device(dev, maxTriplets * sizeof(float), stream);
    tripletsInGPU.zLoPointed = (float*)cms::cuda::allocate_device(dev, maxTriplets * sizeof(float), stream);
    tripletsInGPU.zHiPointed = (float*)cms::cuda::allocate_device(dev, maxTriplets * sizeof(float), stream);
    tripletsInGPU.sdlCut = (float*)cms::cuda::allocate_device(dev, maxTriplets * sizeof(float), stream);
    tripletsInGPU.betaInCut = (float*)cms::cuda::allocate_device(dev, maxTriplets * sizeof(float), stream);
    tripletsInGPU.betaOutCut = (float*)cms::cuda::allocate_device(dev, maxTriplets * sizeof(float), stream);
    tripletsInGPU.deltaBetaCut = (float*)cms::cuda::allocate_device(dev, maxTriplets * sizeof(float), stream);
    tripletsInGPU.rtLo = (float*)cms::cuda::allocate_device(dev, maxTriplets * sizeof(float), stream);
    tripletsInGPU.rtHi = (float*)cms::cuda::allocate_device(dev, maxTriplets * sizeof(float), stream);
    tripletsInGPU.kZ = (float*)cms::cuda::allocate_device(dev, maxTriplets * sizeof(float), stream);
    tripletsInGPU.rtOut = tripletsInGPU.zOut + maxTriplets;
    tripletsInGPU.deltaPhiPos = tripletsInGPU.zOut + maxTriplets *2;
    tripletsInGPU.deltaPhi = tripletsInGPU.zOut + maxTriplets *3;
#endif

#else
    cudaMalloc(&tripletsInGPU.segmentIndices, /*5*/2 * maxTriplets * sizeof(unsigned int));
    cudaMalloc(&tripletsInGPU.lowerModuleIndices, 3 * maxTriplets * sizeof(uint16_t));
    cudaMalloc(&tripletsInGPU.betaIn, maxTriplets * 3 * sizeof(FPX));
    cudaMalloc(&tripletsInGPU.nTriplets, nLowerModules * sizeof(int));
    cudaMalloc(&tripletsInGPU.totOccupancyTriplets, nLowerModules * sizeof(int));
    cudaMalloc(&tripletsInGPU.partOfPT5, maxTriplets * sizeof(bool));
    cudaMalloc(&tripletsInGPU.partOfPT3, maxTriplets * sizeof(bool));
    cudaMalloc(&tripletsInGPU.partOfT5, maxTriplets * sizeof(bool));

    cudaMalloc(&tripletsInGPU.logicalLayers, maxTriplets * 3 * sizeof(uint8_t));
    cudaMalloc(&tripletsInGPU.hitIndices, maxTriplets * 6 * sizeof(unsigned int));
    cudaMalloc(&tripletsInGPU.nMemoryLocations, sizeof(unsigned int));

#ifdef CUT_VALUE_DEBUG
    cudaMalloc(&tripletsInGPU.zOut, maxTriplets * 4*sizeof(unsigned int));
    cudaMalloc(&tripletsInGPU.zLo, maxTriplets * sizeof(float));
    cudaMalloc(&tripletsInGPU.zHi, maxTriplets * sizeof(float));
    cudaMalloc(&tripletsInGPU.zLoPointed, maxTriplets * sizeof(float));
    cudaMalloc(&tripletsInGPU.zHiPointed, maxTriplets * sizeof(float));
    cudaMalloc(&tripletsInGPU.sdlCut, maxTriplets * sizeof(float));
    cudaMalloc(&tripletsInGPU.betaInCut, maxTriplets * sizeof(float));
    cudaMalloc(&tripletsInGPU.betaOutCut, maxTriplets * sizeof(float));
    cudaMalloc(&tripletsInGPU.deltaBetaCut, maxTriplets * sizeof(float));
    cudaMalloc(&tripletsInGPU.rtLo, maxTriplets * sizeof(float));
    cudaMalloc(&tripletsInGPU.rtHi, maxTriplets * sizeof(float));
    cudaMalloc(&tripletsInGPU.kZ, maxTriplets * sizeof(float));

    tripletsInGPU.rtOut = tripletsInGPU.zOut + maxTriplets;
    tripletsInGPU.deltaPhiPos = tripletsInGPU.zOut + maxTriplets *2;
    tripletsInGPU.deltaPhi = tripletsInGPU.zOut + maxTriplets *3;
#endif

#endif
    cudaMemsetAsync(tripletsInGPU.nTriplets,0,nLowerModules * sizeof(int),stream);
    cudaMemsetAsync(tripletsInGPU.totOccupancyTriplets,0,nLowerModules * sizeof(int),stream);
    cudaMemsetAsync(tripletsInGPU.partOfPT5,0,maxTriplets * sizeof(bool),stream);
    cudaMemsetAsync(tripletsInGPU.partOfPT3,0,maxTriplets * sizeof(bool),stream);
    cudaMemsetAsync(tripletsInGPU.partOfT5,0,maxTriplets * sizeof(bool),stream);
    
    cudaStreamSynchronize(stream);

    tripletsInGPU.betaOut = tripletsInGPU.betaIn + maxTriplets;
    tripletsInGPU.pt_beta = tripletsInGPU.betaIn + maxTriplets * 2;
}

SDL::triplets::triplets()
{
    segmentIndices = nullptr;
    lowerModuleIndices = nullptr;
    betaIn = nullptr;
    betaOut = nullptr;
    pt_beta = nullptr;
    logicalLayers = nullptr;
    hitIndices = nullptr;
#ifdef CUT_VALUE_DEBUG
    zOut = nullptr;
    rtOut = nullptr;
    deltaPhiPos = nullptr;
    deltaPhi = nullptr;
    zLo = nullptr;
    zHi = nullptr;
    rtLo = nullptr;
    rtHi = nullptr;
    zLoPointed = nullptr;
    zHiPointed = nullptr;
    kZ = nullptr;
    betaInCut = nullptr;
    betaOutCut = nullptr;
    deltaBetaCut = nullptr;
    sdlCut = nullptr;
#endif
}

SDL::triplets::~triplets()
{
}

void SDL::triplets::freeMemoryCache()
{
    int dev;
    cudaGetDevice(&dev);
    cms::cuda::free_device(dev,segmentIndices);
    cms::cuda::free_device(dev,lowerModuleIndices);
    cms::cuda::free_device(dev,betaIn);
    cms::cuda::free_device(dev,nTriplets);
    cms::cuda::free_device(dev,totOccupancyTriplets);
    cms::cuda::free_device(dev, partOfPT5);
    cms::cuda::free_device(dev, partOfPT3);
    cms::cuda::free_device(dev, partOfT5);
    cms::cuda::free_device(dev, logicalLayers);
    cms::cuda::free_device(dev, hitIndices);
    cms::cuda::free_device(dev, nMemoryLocations);
#ifdef CUT_VALUE_DEBUG
    cms::cuda::free_device(dev, zOut);
    cms::cuda::free_device(dev, zLo);
    cms::cuda::free_device(dev, zHi);
    cms::cuda::free_device(dev, zLoPointed);
    cms::cuda::free_device(dev, zHiPointed);
    cms::cuda::free_device(dev, sdlCut);
    cms::cuda::free_device(dev, betaInCut);
    cms::cuda::free_device(dev, betaOutCut);
    cms::cuda::free_device(dev, deltaBetaCut);
    cms::cuda::free_device(dev, rtLo);
    cms::cuda::free_device(dev, rtHi);
    cms::cuda::free_device(dev, kZ);
#endif
}

void SDL::triplets::freeMemory(cudaStream_t stream)
{
    cudaFree(segmentIndices);
    cudaFree(lowerModuleIndices);
    cudaFree(nTriplets);
    cudaFree(totOccupancyTriplets);
    cudaFree(betaIn);
    cudaFree(partOfPT5);
    cudaFree(partOfPT3);
    cudaFree(partOfT5);
    cudaFree(logicalLayers);
    cudaFree(hitIndices);
    cudaFree(nMemoryLocations);
#ifdef CUT_VALUE_DEBUG
    cudaFree(zOut);
    cudaFree(zLo);
    cudaFree(zHi);
    cudaFree(rtLo);
    cudaFree(rtHi);
    cudaFree(zLoPointed);
    cudaFree(zHiPointed);
    cudaFree(kZ);
    cudaFree(betaInCut);
    cudaFree(betaOutCut);
    cudaFree(deltaBetaCut);
    cudaFree(sdlCut);
#endif
    cudaStreamSynchronize(stream);
}

__global__ void SDL::addTripletRangesToEventExplicit(struct modules& modulesInGPU, struct triplets& tripletsInGPU, struct objectRanges& rangesInGPU)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int np = gridDim.x * blockDim.x;
    for(uint16_t i = gid; i < *modulesInGPU.nLowerModules; i+= np)
    {
        if(tripletsInGPU.nTriplets[i] == 0)
        {
            rangesInGPU.tripletRanges[i * 2] = -1;
            rangesInGPU.tripletRanges[i * 2 + 1] = -1;
        }
        else
        {
            rangesInGPU.tripletRanges[i * 2] = rangesInGPU.tripletModuleIndices[i];
            rangesInGPU.tripletRanges[i * 2 + 1] = rangesInGPU.tripletModuleIndices[i] +  tripletsInGPU.nTriplets[i] - 1;
        }
    }
}