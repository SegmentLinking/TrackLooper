#include "MiniDoublet.cuh"

void SDL::miniDoublets::resetMemory(unsigned int nMemoryLocationsx, unsigned int nLowerModules,cudaStream_t stream)
{
    cudaMemsetAsync(anchorHitIndices,0, nMemoryLocationsx * 3 * sizeof(unsigned int),stream);
    cudaMemsetAsync(dphichanges,0, nMemoryLocationsx * 9 * sizeof(float),stream);
    cudaMemsetAsync(nMDs,0, (nLowerModules + 1) * sizeof(int),stream);
    cudaMemsetAsync(totOccupancyMDs,0, (nLowerModules + 1) * sizeof(unsigned int),stream);
}

__global__ void SDL::createMDArrayRangesGPU(struct modules& modulesInGPU, struct objectRanges& rangesInGPU)
{
    short module_subdets;
    short module_layers;
    short module_rings;
    float module_eta;

    __shared__ unsigned int nTotalMDs; //start!   
    nTotalMDs = 0; //start!   
    __syncthreads();
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int np = gridDim.x * blockDim.x;
    for(uint16_t i = gid; i < *modulesInGPU.nLowerModules; i+= np)
    {
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
        else if (module_eta>1.5  && module_eta<2.25) eta_number=2;
        else if (module_eta>2.25 && module_eta<3) eta_number=3;

        if (category_number == 0 && eta_number == 0) occupancy = 49;
        else if (category_number == 0 && eta_number == 1) occupancy = 42;
        else if (category_number == 0 && eta_number == 2) occupancy = 37;
        else if (category_number == 0 && eta_number == 3) occupancy = 41;
        else if (category_number == 1) occupancy = 100;
        else if (category_number == 2 && eta_number == 1) occupancy = 16;
        else if (category_number == 2 && eta_number == 2) occupancy = 19;
        else if (category_number == 3 && eta_number == 1) occupancy = 14;
        else if (category_number == 3 && eta_number == 2) occupancy = 20;
        else if (category_number == 3 && eta_number == 3) occupancy = 25;

        unsigned int nTotMDs= atomicAdd(&nTotalMDs,occupancy);
        rangesInGPU.miniDoubletModuleIndices[i] = nTotMDs; 
        rangesInGPU.miniDoubletModuleOccupancy[i] = occupancy;
    }
    __syncthreads();
    if(threadIdx.x==0){
      rangesInGPU.miniDoubletModuleIndices[*modulesInGPU.nLowerModules] = nTotalMDs;
      //*nTotalMDsx=nTotalMDs;
      *rangesInGPU.device_nTotalMDs=nTotalMDs;
    }

}

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
    mdsInGPU.anchorHighEdgeX = (float*)cms::cuda::allocate_device(dev, nMemoryLocations * 6 * sizeof(float), stream);
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
    cudaMalloc(&mdsInGPU.anchorHighEdgeX, nMemoryLocations * 6 * sizeof(float));
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
    mdsInGPU.anchorHighEdgePhi = mdsInGPU.anchorHighEdgeX + 4 * nMemoryLocations;
    mdsInGPU.anchorLowEdgePhi = mdsInGPU.anchorHighEdgeX + 5 * nMemoryLocations;

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
    anchorHighEdgePhi = nullptr;
    anchorLowEdgePhi = nullptr;
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

void SDL::printMD(struct miniDoublets& mdsInGPU, struct hits& hitsInGPU, SDL::modules& modulesInGPU, unsigned int mdIndex)
{
    std::cout<<std::endl;
    std::cout << "dz " << mdsInGPU.dzs[mdIndex] << std::endl;
    std::cout << "dphi " << mdsInGPU.dphis[mdIndex] << std::endl;
    std::cout << "dphinoshift " << mdsInGPU.noShiftedDphis[mdIndex] << std::endl;
    std::cout << "dphichange " << mdsInGPU.dphichanges[mdIndex] << std::endl;
    std::cout << "dphichangenoshift " << mdsInGPU.noShiftedDphiChanges[mdIndex] << std::endl;
    std::cout << std::endl;
    std::cout << "Anchor Hit " << std::endl;
    std::cout << "------------------------------" << std::endl;
    unsigned int lowerHitIndex = mdsInGPU.anchorHitIndices[mdIndex];
    unsigned int upperHitIndex = mdsInGPU.outerHitIndices[mdIndex];
    {
        IndentingOStreambuf indent(std::cout);
        printHit(hitsInGPU, modulesInGPU, lowerHitIndex);
    }
    std::cout << "Non-anchor Hit " << std::endl;
    std::cout << "------------------------------" << std::endl;
    {
        IndentingOStreambuf indent(std::cout);
        printHit(hitsInGPU, modulesInGPU, upperHitIndex);
    }
}

__global__ void SDL::addMiniDoubletRangesToEventExplicit(struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, struct objectRanges& rangesInGPU,struct hits& hitsInGPU)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int np = gridDim.x * blockDim.x;
    for(uint16_t i = gid; i < *modulesInGPU.nLowerModules; i+= np)
    {
        if(mdsInGPU.nMDs[i] == 0 or hitsInGPU.hitRanges[i * 2] == -1)
        {
            rangesInGPU.mdRanges[i * 2] = -1;
            rangesInGPU.mdRanges[i * 2 + 1] = -1;
        }
        else
        {
            rangesInGPU.mdRanges[i * 2] = rangesInGPU.miniDoubletModuleIndices[i] ;
            rangesInGPU.mdRanges[i * 2 + 1] = rangesInGPU.miniDoubletModuleIndices[i] + mdsInGPU.nMDs[i] - 1;
        }
    }
}
