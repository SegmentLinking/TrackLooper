# include "Quintuplet.cuh"

SDL::quintuplets::quintuplets()
{
    tripletIndices = nullptr;
    lowerModuleIndices = nullptr;
    nQuintuplets = nullptr;
    totOccupancyQuintuplets = nullptr;
    innerRadius = nullptr;
    outerRadius = nullptr;
    regressionRadius = nullptr;
    isDup = nullptr;
    TightCutFlag = nullptr;
    partOfPT5 = nullptr;
    pt = nullptr;
    layer = nullptr;
    regressionG = nullptr;
    regressionF = nullptr;
    logicalLayers = nullptr;
    hitIndices = nullptr;
    bridgeRadius = nullptr;
    chiSquared = nullptr;
    rzChiSquared = nullptr;
    nonAnchorChiSquared = nullptr;
}

SDL::quintuplets::~quintuplets()
{
}

void SDL::quintuplets::freeMemoryCache()
{
    int dev;
    cudaGetDevice(&dev);
    cms::cuda::free_device(dev, tripletIndices);
    cms::cuda::free_device(dev, lowerModuleIndices);
    cms::cuda::free_device(dev, nQuintuplets);
    cms::cuda::free_device(dev, totOccupancyQuintuplets);
    cms::cuda::free_device(dev, innerRadius);
    cms::cuda::free_device(dev, outerRadius);
    cms::cuda::free_device(dev, partOfPT5);
    cms::cuda::free_device(dev, isDup);
    cms::cuda::free_device(dev, TightCutFlag);
    cms::cuda::free_device(dev, pt);
    cms::cuda::free_device(dev, layer);
    cms::cuda::free_device(dev, regressionG);
    cms::cuda::free_device(dev, regressionF);
    cms::cuda::free_device(dev, regressionRadius);
    cms::cuda::free_device(dev, logicalLayers);
    cms::cuda::free_device(dev, hitIndices);
    cms::cuda::free_device(dev, nMemoryLocations);
    cms::cuda::free_device(dev, bridgeRadius);
    cms::cuda::free_device(dev, rzChiSquared);
    cms::cuda::free_device(dev, chiSquared);
    cms::cuda::free_device(dev, nonAnchorChiSquared);
}

void SDL::quintuplets::freeMemory(cudaStream_t stream)
{
    cudaFree(tripletIndices);
    cudaFree(lowerModuleIndices);
    cudaFree(nQuintuplets);
    cudaFree(totOccupancyQuintuplets);
    cudaFree(innerRadius);
    cudaFree(outerRadius);
    cudaFree(regressionRadius);
    cudaFree(partOfPT5);
    cudaFree(isDup);
    cudaFree(TightCutFlag);
    cudaFree(pt);
    cudaFree(layer);
    cudaFree(regressionG);
    cudaFree(regressionF);
    cudaFree(logicalLayers);
    cudaFree(hitIndices);
    cudaFree(nMemoryLocations);
    cudaFree(bridgeRadius);
    cudaFree(rzChiSquared);
    cudaFree(chiSquared);
    cudaFree(nonAnchorChiSquared);
    cudaStreamSynchronize(stream);
}
//TODO:Reuse the track candidate one instead of this!
__global__ void SDL::createEligibleModulesListForQuintupletsGPU(struct modules& modulesInGPU,struct triplets& tripletsInGPU, struct objectRanges& rangesInGPU)
{
    __shared__ int nEligibleT5Modulesx;
    __shared__ unsigned int nTotalQuintupletsx;
    nTotalQuintupletsx = 0; //start!
    nEligibleT5Modulesx = 0;
    __syncthreads();

    unsigned int occupancy;
    unsigned int category_number, eta_number;
    unsigned int layers, subdets, rings;
    float eta;
    //start filling
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int np = gridDim.x * blockDim.x;
    for(uint16_t i = gid; i < *modulesInGPU.nLowerModules; i+= np)
    {
        //condition for a quintuple to exist for a module
        //TCs don't exist for layers 5 and 6 barrel, and layers 2,3,4,5 endcap   
        layers = modulesInGPU.layers[i];
        subdets = modulesInGPU.subdets[i];
        rings = modulesInGPU.rings[i];
        eta = abs(modulesInGPU.eta[i]);  
        occupancy = 0;

        if (tripletsInGPU.nTriplets[i] == 0) continue;
        if (subdets == SDL::Barrel and layers >= 3) continue;
        if (subdets == SDL::Endcap and layers > 1) continue;

        int nEligibleT5Modules = atomicAdd(&nEligibleT5Modulesx,1);
        if (layers<=3 && subdets==5) category_number = 0;
        else if (layers>=4 && subdets==5) category_number = 1;
        else if (layers<=2 && subdets==4 && rings>=11) category_number = 2;
        else if (layers>=3 && subdets==4 && rings>=8) category_number = 2;
        else if (layers<=2 && subdets==4 && rings<=10) category_number = 3;
        else if (layers>=3 && subdets==4 && rings<=7) category_number = 3;
        if (eta<0.75) eta_number=0;
        else if (eta>0.75 && eta<1.5) eta_number=1;
        else if (eta>1.5 && eta<2.25) eta_number=2;
        else if (eta>2.25 && eta<3) eta_number=3;

        if (category_number == 0 && eta_number == 0) occupancy = 336;
        else if (category_number == 0 && eta_number == 1) occupancy = 414;
        else if (category_number == 0 && eta_number == 2) occupancy = 231;
        else if (category_number == 0 && eta_number == 3) occupancy = 146;
        else if (category_number == 3 && eta_number == 1) occupancy = 0;
        else if (category_number == 3 && eta_number == 2) occupancy = 191;
        else if (category_number == 3 && eta_number == 3) occupancy = 106;

        unsigned int nTotQ = atomicAdd(&nTotalQuintupletsx,occupancy);
        rangesInGPU.quintupletModuleIndices[i] = nTotQ;
        rangesInGPU.quintupletModuleOccupancy[i] = occupancy;
        rangesInGPU.indicesOfEligibleT5Modules[nEligibleT5Modules] = i;
    }
    __syncthreads();
    if(threadIdx.x==0){
        *rangesInGPU.nEligibleT5Modules = static_cast<uint16_t>(nEligibleT5Modulesx);
        *rangesInGPU.device_nTotalQuints = nTotalQuintupletsx;
    }
}

void SDL::createQuintupletsInExplicitMemory(struct SDL::quintuplets& quintupletsInGPU, const unsigned int& nTotalQuintuplets, const uint16_t& nLowerModules, const uint16_t& nEligibleModules,cudaStream_t stream)
{
#ifdef CACHE_ALLOC
    int dev;
    cudaGetDevice(&dev);
    quintupletsInGPU.tripletIndices = (unsigned int*)cms::cuda::allocate_device(dev, 2 * nTotalQuintuplets * sizeof(unsigned int), stream);
    quintupletsInGPU.lowerModuleIndices = (uint16_t*)cms::cuda::allocate_device(dev, 5 * nTotalQuintuplets * sizeof(uint16_t), stream);
    quintupletsInGPU.nQuintuplets = (int*)cms::cuda::allocate_device(dev, nLowerModules * sizeof(int), stream);
    quintupletsInGPU.totOccupancyQuintuplets = (int*)cms::cuda::allocate_device(dev, nLowerModules * sizeof(int), stream);
    quintupletsInGPU.innerRadius = (FPX*)cms::cuda::allocate_device(dev, nTotalQuintuplets * sizeof(FPX), stream);
    quintupletsInGPU.outerRadius = (FPX*)cms::cuda::allocate_device(dev, nTotalQuintuplets * sizeof(FPX), stream);
    quintupletsInGPU.bridgeRadius = (FPX*)cms::cuda::allocate_device(dev, nTotalQuintuplets * sizeof(float), stream);

    quintupletsInGPU.pt = (FPX*)cms::cuda::allocate_device(dev, nTotalQuintuplets *4* sizeof(FPX), stream);
    quintupletsInGPU.layer = (uint8_t*)cms::cuda::allocate_device(dev, nTotalQuintuplets * sizeof(uint8_t), stream);
    quintupletsInGPU.isDup = (bool*)cms::cuda::allocate_device(dev, nTotalQuintuplets * sizeof(bool), stream);
    quintupletsInGPU.TightCutFlag = (bool*)cms::cuda::allocate_device(dev, nTotalQuintuplets * sizeof(bool), stream);
    quintupletsInGPU.partOfPT5 = (bool*)cms::cuda::allocate_device(dev, nTotalQuintuplets * sizeof(bool), stream);
    quintupletsInGPU.regressionRadius = (float*)cms::cuda::allocate_device(dev, nTotalQuintuplets * sizeof(float), stream);
    quintupletsInGPU.regressionG = (float*)cms::cuda::allocate_device(dev, nTotalQuintuplets * sizeof(float), stream);
    quintupletsInGPU.regressionF = (float*)cms::cuda::allocate_device(dev, nTotalQuintuplets * sizeof(float), stream);
    quintupletsInGPU.logicalLayers = (uint8_t*)cms::cuda::allocate_device(dev, nTotalQuintuplets * sizeof(uint8_t) * 5, stream);
    quintupletsInGPU.hitIndices = (unsigned int*)cms::cuda::allocate_device(dev, nTotalQuintuplets * sizeof(unsigned int) * 10, stream);
    quintupletsInGPU.nMemoryLocations = (unsigned int*)cms::cuda::allocate_device(dev, sizeof(unsigned int), stream);

    quintupletsInGPU.rzChiSquared = (float*)cms::cuda::allocate_device(dev, nTotalQuintuplets * sizeof(float), stream);
    quintupletsInGPU.chiSquared = (float*)cms::cuda::allocate_device(dev, nTotalQuintuplets * sizeof(float), stream);
    quintupletsInGPU.nonAnchorChiSquared = (float*)cms::cuda::allocate_device(dev, nTotalQuintuplets * sizeof(float), stream);
#else
    cudaMalloc(&quintupletsInGPU.tripletIndices, 2 * nTotalQuintuplets * sizeof(unsigned int));
    cudaMalloc(&quintupletsInGPU.lowerModuleIndices, 5 * nTotalQuintuplets * sizeof(uint16_t));
    cudaMalloc(&quintupletsInGPU.nQuintuplets, nLowerModules * sizeof(int));
    cudaMalloc(&quintupletsInGPU.totOccupancyQuintuplets, nLowerModules * sizeof(int));
    cudaMalloc(&quintupletsInGPU.innerRadius, nTotalQuintuplets * sizeof(FPX));
    cudaMalloc(&quintupletsInGPU.outerRadius, nTotalQuintuplets * sizeof(FPX));
    cudaMalloc(&quintupletsInGPU.pt, nTotalQuintuplets *4* sizeof(FPX));
    cudaMalloc(&quintupletsInGPU.isDup, nTotalQuintuplets * sizeof(bool));
    cudaMalloc(&quintupletsInGPU.TightCutFlag, nTotalQuintuplets * sizeof(bool));
    cudaMalloc(&quintupletsInGPU.partOfPT5, nTotalQuintuplets * sizeof(bool));
    cudaMalloc(&quintupletsInGPU.layer, nTotalQuintuplets * sizeof(uint8_t));
    cudaMalloc(&quintupletsInGPU.regressionRadius, nTotalQuintuplets * sizeof(float));
    cudaMalloc(&quintupletsInGPU.regressionG, nTotalQuintuplets * sizeof(float));
    cudaMalloc(&quintupletsInGPU.regressionF, nTotalQuintuplets * sizeof(float));
    cudaMalloc(&quintupletsInGPU.logicalLayers, nTotalQuintuplets * 5 * sizeof(uint8_t));
    cudaMalloc(&quintupletsInGPU.hitIndices, nTotalQuintuplets * 10 * sizeof(unsigned int));
    cudaMalloc(&quintupletsInGPU.nMemoryLocations, sizeof(unsigned int));
    cudaMalloc(&quintupletsInGPU.bridgeRadius, nTotalQuintuplets * sizeof(float));
    cudaMalloc(&quintupletsInGPU.rzChiSquared, nTotalQuintuplets * sizeof(float));
    cudaMalloc(&quintupletsInGPU.chiSquared, nTotalQuintuplets * sizeof(float));
    cudaMalloc(&quintupletsInGPU.nonAnchorChiSquared, nTotalQuintuplets * sizeof(float));
    cudaMalloc(&quintupletsInGPU.nMemoryLocations, sizeof(unsigned int));
#endif
    cudaMemsetAsync(quintupletsInGPU.nQuintuplets,0,nLowerModules * sizeof(int),stream);
    cudaMemsetAsync(quintupletsInGPU.totOccupancyQuintuplets,0,nLowerModules * sizeof(int),stream);
    cudaMemsetAsync(quintupletsInGPU.isDup,0,nTotalQuintuplets * sizeof(bool),stream);
    cudaMemsetAsync(quintupletsInGPU.TightCutFlag,0,nTotalQuintuplets * sizeof(bool),stream);
    cudaMemsetAsync(quintupletsInGPU.partOfPT5,0,nTotalQuintuplets * sizeof(bool),stream);
    cudaStreamSynchronize(stream);
    quintupletsInGPU.eta = quintupletsInGPU.pt + nTotalQuintuplets;
    quintupletsInGPU.phi = quintupletsInGPU.pt + 2*nTotalQuintuplets;
    quintupletsInGPU.score_rphisum = quintupletsInGPU.pt + 3*nTotalQuintuplets;
}

__global__ void SDL::addQuintupletRangesToEventExplicit(struct modules& modulesInGPU, struct quintuplets& quintupletsInGPU, struct objectRanges& rangesInGPU)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int np = gridDim.x * blockDim.x;
    for(uint16_t i = gid; i < *modulesInGPU.nLowerModules; i+= np)
    {
        if(quintupletsInGPU.nQuintuplets[i] == 0 or rangesInGPU.quintupletModuleIndices[i] == -1)
        {
            rangesInGPU.quintupletRanges[i * 2] = -1;
            rangesInGPU.quintupletRanges[i * 2 + 1] = -1;
        }
       else
        {
            rangesInGPU.quintupletRanges[i * 2] = rangesInGPU.quintupletModuleIndices[i];
            rangesInGPU.quintupletRanges[i * 2 + 1] = rangesInGPU.quintupletModuleIndices[i] + quintupletsInGPU.nQuintuplets[i] - 1;
        }
    }
}
