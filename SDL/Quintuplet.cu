#ifdef __CUDACC__
#define CUDA_CONST_VAR __device__
#endif
# include "Quintuplet.cuh"
#include "allocate.h"

SDL::quintuplets::quintuplets()
{
    tripletIndices = nullptr;
    lowerModuleIndices = nullptr;
    nQuintuplets = nullptr;
    innerRadius = nullptr;
    outerRadius = nullptr;
    regressionRadius = nullptr;
    isDup = nullptr;
    partOfPT5 = nullptr;
    pt = nullptr;
    layer = nullptr;
    regressionG = nullptr;
    regressionF = nullptr;

#ifdef TRACK_EXTENSIONS
    logicalLayers = nullptr;
    hitIndices = nullptr;
#endif
#ifdef CUT_VALUE_DEBUG
    innerRadiusMin = nullptr;
    innerRadiusMin2S = nullptr;
    innerRadiusMax = nullptr;
    innerRadiusMax2S = nullptr;
    bridgeRadius = nullptr;
    bridgeRadiusMin = nullptr;
    bridgeRadiusMin2S = nullptr;
    bridgeRadiusMax = nullptr;
    bridgeRadiusMax2S = nullptr;
    outerRadiusMin = nullptr;
    outerRadiusMin2S = nullptr;
    outerRadiusMax = nullptr;
    outerRadiusMax2S = nullptr;
    chiSquared = nullptr;
    nonAnchorChiSquared = nullptr;
#endif
}

SDL::quintuplets::~quintuplets()
{
}

void SDL::quintuplets::freeMemoryCache()
{
#ifdef Explicit_T5
    int dev;
    cudaGetDevice(&dev);
    cms::cuda::free_device(dev,tripletIndices);
    cms::cuda::free_device(dev, lowerModuleIndices);
    cms::cuda::free_device(dev, nQuintuplets);
    cms::cuda::free_device(dev, innerRadius);
    cms::cuda::free_device(dev, outerRadius);
    cms::cuda::free_device(dev, partOfPT5);
    cms::cuda::free_device(dev, isDup);
    cms::cuda::free_device(dev, pt);
    cms::cuda::free_device(dev, layer);
    cms::cuda::free_device(dev, regressionG);
    cms::cuda::free_device(dev, regressionF);
    cms::cuda::free_device(dev, regressionRadius);
#ifdef TRACK_EXTENSIONS
    cms::cuda::free_device(dev, logicalLayers);
    cms::cuda::free_device(dev, hitIndices);
#endif
#else
    cms::cuda::free_managed(tripletIndices);
    cms::cuda::free_managed(lowerModuleIndices);
    cms::cuda::free_managed(nQuintuplets);
    cms::cuda::free_managed(innerRadius);
    cms::cuda::free_managed(outerRadius);
    cms::cuda::free_managed(partOfPT5);
    cms::cuda::free_managed(isDup);
    cms::cuda::free_managed(pt);
    cms::cuda::free_managed(layer);
    cms::cuda::free_managed(regressionG);
    cms::cuda::free_managed(regressionF);
    cms::cuda::free_managed(regressionRadius);

#ifdef TRACK_EXTENSIONS
    cms::cuda::free_managed(logicalLayers);
    cms::cuda::free_managed(hitIndices);
#endif
#endif
}

void SDL::quintuplets::freeMemory(cudaStream_t stream)
{
    cudaFree(tripletIndices);
    cudaFree(lowerModuleIndices);
    cudaFree(nQuintuplets);
    cudaFree(innerRadius);
    cudaFree(outerRadius);
    cudaFree(regressionRadius);
    cudaFree(partOfPT5);
    cudaFree(isDup);
    cudaFree(pt);
    cudaFree(layer);
    cudaFree(regressionG);
    cudaFree(regressionF);
#ifdef TRACK_EXTENSIONS
    cudaFree(logicalLayers);
    cudaFree(hitIndices);
#endif
#ifdef CUT_VALUE_DEBUG
    cudaFree(innerRadiusMin);
    cudaFree(innerRadiusMin2S);
    cudaFree(innerRadiusMax);
    cudaFree(innerRadiusMax2S);
    cudaFree(bridgeRadius);
    cudaFree(bridgeRadiusMin);
    cudaFree(bridgeRadiusMin2S);
    cudaFree(bridgeRadiusMax);
    cudaFree(bridgeRadiusMax2S);
    cudaFree(outerRadiusMin);
    cudaFree(outerRadiusMin2S);
    cudaFree(outerRadiusMax);
    cudaFree(outerRadiusMax2S);
    cudaFree(chiSquared);
    cudaFree(nonAnchorChiSquared);
#endif
cudaStreamSynchronize(stream);
}

//TODO:Reuse the track candidate one instead of this!
void SDL::createEligibleModulesListForQuintuplets(struct modules& modulesInGPU,struct triplets& tripletsInGPU, unsigned int& nEligibleModules, unsigned int* indicesOfEligibleModules, unsigned int maxQuintuplets, unsigned int& maxTriplets,cudaStream_t stream,struct objectRanges& rangesInGPU)
{
    unsigned int nLowerModules;
    maxTriplets = 0;
    cudaMemcpyAsync(&nLowerModules,modulesInGPU.nLowerModules,sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);
    unsigned int nModules;
    cudaMemcpyAsync(&nModules,modulesInGPU.nModules,sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);
    cudaMemsetAsync(rangesInGPU.quintupletModuleIndices, -1, sizeof(int) * (nLowerModules),stream);

    short* module_subdets;
    cudaMallocHost(&module_subdets, nModules* sizeof(short));
    cudaMemcpyAsync(module_subdets,modulesInGPU.subdets,nModules*sizeof(short),cudaMemcpyDeviceToHost,stream);
    unsigned int* module_lowerModuleIndices;
    cudaMallocHost(&module_lowerModuleIndices, nLowerModules * sizeof(unsigned int));
    cudaMemcpyAsync(module_lowerModuleIndices,modulesInGPU.lowerModuleIndices, nLowerModules * sizeof(unsigned int),cudaMemcpyDeviceToHost,stream);
    short* module_layers;
    cudaMallocHost(&module_layers, nModules * sizeof(short));
    cudaMemcpyAsync(module_layers,modulesInGPU.layers,nModules * sizeof(short),cudaMemcpyDeviceToHost,stream);
    int* module_quintupletModuleIndices;
    cudaMallocHost(&module_quintupletModuleIndices, nLowerModules * sizeof(int));
    cudaMemcpyAsync(module_quintupletModuleIndices,rangesInGPU.quintupletModuleIndices,nLowerModules *sizeof(int),cudaMemcpyDeviceToHost,stream);

    unsigned int* nTriplets;
    cudaMallocHost(&nTriplets, nLowerModules * sizeof(unsigned int));
    cudaMemcpyAsync(nTriplets, tripletsInGPU.nTriplets, nLowerModules * sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
cudaStreamSynchronize(stream);

    //start filling
    for(unsigned int i = 0; i < nLowerModules; i++)
    {
        //condition for a quintuple to exist for a module
        //TCs don't exist for layers 5 and 6 barrel, and layers 2,3,4,5 endcap
        unsigned int idx = module_lowerModuleIndices[i];
        if(((module_subdets[idx] == SDL::Barrel and module_layers[idx] < 5) or (module_subdets[idx] == SDL::Endcap and module_layers[idx] == 1)) and nTriplets[i] != 0)
        {
            module_quintupletModuleIndices[i] = nEligibleModules * maxQuintuplets;
            indicesOfEligibleModules[nEligibleModules] = i;
            nEligibleModules++;
            maxTriplets = max(nTriplets[i], maxTriplets);

        }
    }
    cudaMemcpyAsync(rangesInGPU.quintupletModuleIndices,module_quintupletModuleIndices,nLowerModules*sizeof(int),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(rangesInGPU.nEligibleT5Modules,&nEligibleModules,sizeof(unsigned int),cudaMemcpyHostToDevice,stream);
cudaStreamSynchronize(stream);
    cudaFreeHost(module_subdets);
    cudaFreeHost(module_lowerModuleIndices);
    cudaFreeHost(module_layers);
    cudaFreeHost(module_quintupletModuleIndices);
    cudaFreeHost(nTriplets);
}


void SDL::createQuintupletsInUnifiedMemory(struct SDL::quintuplets& quintupletsInGPU, const unsigned int& maxQuintuplets, const unsigned int& nLowerModules, const unsigned int& nEligibleModules, cudaStream_t stream)
{
    unsigned int nMemoryLocations = maxQuintuplets * nEligibleModules;
//    std::cout<<"Number of eligible T5 modules = "<<nEligibleModules<<std::endl;

#ifdef CACHE_ALLOC
//    cudaStream_t stream = 0;
    quintupletsInGPU.tripletIndices = (unsigned int*)cms::cuda::allocate_managed(nMemoryLocations * 2 * sizeof(unsigned int), stream);
    quintupletsInGPU.lowerModuleIndices = (unsigned int*)cms::cuda::allocate_managed(nMemoryLocations * 5 * sizeof(unsigned int), stream);
    quintupletsInGPU.nQuintuplets = (unsigned int*)cms::cuda::allocate_managed(nLowerModules * sizeof(unsigned int), stream);
    quintupletsInGPU.innerRadius = (FPX*)cms::cuda::allocate_managed(nMemoryLocations * sizeof(FPX), stream);
    quintupletsInGPU.outerRadius = (FPX*)cms::cuda::allocate_managed(nMemoryLocations * sizeof(FPX), stream);
    quintupletsInGPU.pt = (FPX*)cms::cuda::allocate_managed(nMemoryLocations *4* sizeof(FPX), stream);
    quintupletsInGPU.layer = (int*)cms::cuda::allocate_managed(nMemoryLocations * sizeof(int), stream);
    quintupletsInGPU.isDup = (bool*)cms::cuda::allocate_managed(nMemoryLocations * sizeof(bool), stream);
    quintupletsInGPU.partOfPT5 = (bool*)cms::cuda::allocate_managed(nMemoryLocations * sizeof(bool), stream);
    quintupletsInGPU.regressionRadius = (float*)cms::cuda::allocate_managed(nMemoryLocations * sizeof(float), stream);
    quintupletsInGPU.regressionG = (float*)cms::cuda::allocate_managed(nMemoryLocations * sizeof(float), stream);
    quintupletsInGPU.regressionF = (float*)cms::cuda::allocate_managed(nMemoryLocations * sizeof(float), stream);
#ifdef TRACK_EXTENSIONS
    quintupletsInGPU.logicalLayers = (unsigned int*)cms::cuda::allocate_managed(nMemoryLocations * sizeof(unsigned int) * 5, stream);
    quintupletsInGPU.hitIndices = (unsigned int*)cms::cuda::allocate_managed(nMemoryLocations * sizeof(unsigned int) * 10, stream);
#endif
#else
    cudaMallocManaged(&quintupletsInGPU.tripletIndices, 2 * nMemoryLocations * sizeof(unsigned int));
    cudaMallocManaged(&quintupletsInGPU.lowerModuleIndices, 5 * nMemoryLocations * sizeof(unsigned int));

    cudaMallocManaged(&quintupletsInGPU.nQuintuplets, nLowerModules * sizeof(unsigned int));
    cudaMallocManaged(&quintupletsInGPU.innerRadius, nMemoryLocations * sizeof(FPX));
    cudaMallocManaged(&quintupletsInGPU.outerRadius, nMemoryLocations * sizeof(FPX));
    cudaMallocManaged(&quintupletsInGPU.pt, nMemoryLocations *4* sizeof(FPX));
    cudaMallocManaged(&quintupletsInGPU.layer, nMemoryLocations * sizeof(int));
    cudaMallocManaged(&quintupletsInGPU.isDup, nMemoryLocations * sizeof(bool));
    cudaMallocManaged(&quintupletsInGPU.partOfPT5, nMemoryLocations * sizeof(bool));
    cudaMallocManaged(&quintupletsInGPU.regressionRadius, nMemoryLocations * sizeof(float));
    cudaMallocManaged(&quintupletsInGPU.regressionG, nMemoryLocations * sizeof(float));
    cudaMallocManaged(&quintupletsInGPU.regressionF, nMemoryLocations * sizeof(float));
#ifdef TRACK_EXTENSIONS
    cudaMallocManaged(&quintupletsInGPU.logicalLayers, nMemoryLocations * sizeof(unsigned int) * 5);
    cudaMallocManaged(&quintupletsInGPU.hitIndices, nMemoryLocations * sizeof(unsigned int) * 10);
#endif
#ifdef CUT_VALUE_DEBUG
    cudaMallocManaged(&quintupletsInGPU.innerRadiusMin, nMemoryLocations * sizeof(float));
    cudaMallocManaged(&quintupletsInGPU.innerRadiusMax, nMemoryLocations * sizeof(float));
    cudaMallocManaged(&quintupletsInGPU.bridgeRadius, nMemoryLocations * sizeof(float));
    cudaMallocManaged(&quintupletsInGPU.bridgeRadiusMin, nMemoryLocations * sizeof(float));
    cudaMallocManaged(&quintupletsInGPU.bridgeRadiusMax, nMemoryLocations * sizeof(float));
    cudaMallocManaged(&quintupletsInGPU.outerRadiusMin, nMemoryLocations * sizeof(float));
    cudaMallocManaged(&quintupletsInGPU.outerRadiusMax, nMemoryLocations * sizeof(float));
    cudaMallocManaged(&quintupletsInGPU.innerRadiusMin2S, nMemoryLocations * sizeof(float));
    cudaMallocManaged(&quintupletsInGPU.innerRadiusMax2S, nMemoryLocations * sizeof(float));
    cudaMallocManaged(&quintupletsInGPU.bridgeRadiusMin2S, nMemoryLocations * sizeof(float));
    cudaMallocManaged(&quintupletsInGPU.bridgeRadiusMax2S, nMemoryLocations * sizeof(float));
    cudaMallocManaged(&quintupletsInGPU.outerRadiusMin2S, nMemoryLocations * sizeof(float));
    cudaMallocManaged(&quintupletsInGPU.outerRadiusMax2S, nMemoryLocations * sizeof(float));
    cudaMallocManaged(&quintupletsInGPU.chiSquared, nMemoryLocations * sizeof(float));
    cudaMallocManaged(&quintupletsInGPU.nonAnchorChiSquared, nMemoryLocations * sizeof(float));
#endif
#endif
    quintupletsInGPU.eta = quintupletsInGPU.pt + nMemoryLocations;
    quintupletsInGPU.phi = quintupletsInGPU.pt + 2*nMemoryLocations;
    //quintupletsInGPU.score_rphi = quintupletsInGPU.pt + 3*nMemoryLocations;
    //quintupletsInGPU.score_rz = quintupletsInGPU.pt + 4*nMemoryLocations;
    quintupletsInGPU.score_rphisum = quintupletsInGPU.pt + 3*nMemoryLocations;
    //quintupletsInGPU.score_rzlsq = quintupletsInGPU.pt + 6*nMemoryLocations;
//#pragma omp parallel for
//    for(size_t i = 0; i<nLowerModules;i++)
//    {
//        quintupletsInGPU.nQuintuplets[i] = 0;
//    }

    cudaMemsetAsync(quintupletsInGPU.nQuintuplets,0,nLowerModules * sizeof(unsigned int),stream);
}

void SDL::createQuintupletsInExplicitMemory(struct SDL::quintuplets& quintupletsInGPU, const unsigned int& maxQuintuplets, const unsigned int& nLowerModules, const unsigned int& nEligibleModules,cudaStream_t stream)
{
    unsigned int nMemoryLocations = nEligibleModules * maxQuintuplets;
#ifdef CACHE_ALLOC
 //   cudaStream_t stream = 0;
    int dev;
    cudaGetDevice(&dev);
    quintupletsInGPU.tripletIndices = (unsigned int*)cms::cuda::allocate_device(dev, 2 * nMemoryLocations * sizeof(unsigned int), stream);
    quintupletsInGPU.lowerModuleIndices = (unsigned int*)cms::cuda::allocate_device(dev, 5 * nMemoryLocations * sizeof(unsigned int), stream);
    quintupletsInGPU.nQuintuplets = (unsigned int*)cms::cuda::allocate_device(dev, nLowerModules * sizeof(unsigned int), stream);
    quintupletsInGPU.innerRadius = (FPX*)cms::cuda::allocate_device(dev, nMemoryLocations * sizeof(FPX), stream);
    quintupletsInGPU.outerRadius = (FPX*)cms::cuda::allocate_device(dev, nMemoryLocations * sizeof(FPX), stream);
    quintupletsInGPU.pt = (FPX*)cms::cuda::allocate_device(dev, nMemoryLocations *4* sizeof(FPX), stream);
    quintupletsInGPU.layer = (int*)cms::cuda::allocate_device(dev, nMemoryLocations * sizeof(int), stream);
    quintupletsInGPU.isDup = (bool*)cms::cuda::allocate_device(dev, nMemoryLocations * sizeof(bool), stream);
    quintupletsInGPU.partOfPT5 = (bool*)cms::cuda::allocate_device(dev, nMemoryLocations * sizeof(bool), stream);
    quintupletsInGPU.regressionRadius = (float*)cms::cuda::allocate_device(dev, nMemoryLocations * sizeof(float), stream);
    quintupletsInGPU.regressionG = (float*)cms::cuda::allocate_device(dev, nMemoryLocations * sizeof(float), stream);
    quintupletsInGPU.regressionF = (float*)cms::cuda::allocate_device(dev, nMemoryLocations * sizeof(float), stream);
#ifdef TRACK_EXTENSIONS
    quintupletsInGPU.logicalLayers = (unsigned int*)cms::cuda::allocate_device(dev, nMemoryLocations * sizeof(unsigned int) * 5, stream);
    quintupletsInGPU.hitIndices = (unsigned int*)cms::cuda::allocate_device(dev, nMemoryLocations * sizeof(unsigned int) * 10, stream);
#endif
#else
    cudaMalloc(&quintupletsInGPU.tripletIndices, 2 * nMemoryLocations * sizeof(unsigned int));
    cudaMalloc(&quintupletsInGPU.lowerModuleIndices, 5 * nMemoryLocations * sizeof(unsigned int));
    cudaMalloc(&quintupletsInGPU.nQuintuplets, nLowerModules * sizeof(unsigned int));
    cudaMalloc(&quintupletsInGPU.innerRadius, nMemoryLocations * sizeof(FPX));
    cudaMalloc(&quintupletsInGPU.outerRadius, nMemoryLocations * sizeof(FPX));
    cudaMalloc(&quintupletsInGPU.pt, nMemoryLocations *4* sizeof(FPX));
    cudaMalloc(&quintupletsInGPU.isDup, nMemoryLocations * sizeof(bool));
    cudaMalloc(&quintupletsInGPU.partOfPT5, nMemoryLocations * sizeof(bool));
    cudaMalloc(&quintupletsInGPU.layer, nMemoryLocations * sizeof(int));
    cudaMalloc(&quintupletsInGPU.regressionRadius, nMemoryLocations * sizeof(float));
    cudaMalloc(&quintupletsInGPU.regressionG, nMemoryLocations * sizeof(float));
    cudaMalloc(&quintupletsInGPU.regressionF, nMemoryLocations * sizeof(float));
#ifdef TRACK_EXTENSIONS
    cudaMalloc(&quintupletsInGPU.logicalLayers, nMemoryLocations * 5 * sizeof(unsigned int));
    cudaMalloc(&quintupletsInGPU.hitIndices, nMemoryLocations * 10 * sizeof(unsigned int));
#endif
#endif
    cudaMemsetAsync(quintupletsInGPU.nQuintuplets,0,nLowerModules * sizeof(unsigned int),stream);
    cudaStreamSynchronize(stream);
    quintupletsInGPU.eta = quintupletsInGPU.pt + nMemoryLocations;
    quintupletsInGPU.phi = quintupletsInGPU.pt + 2*nMemoryLocations;
    //quintupletsInGPU.score_rphi = quintupletsInGPU.pt + 3*nMemoryLocations;
    //quintupletsInGPU.score_rz = quintupletsInGPU.pt + 4*nMemoryLocations;
    quintupletsInGPU.score_rphisum = quintupletsInGPU.pt + 3*nMemoryLocations;
    //quintupletsInGPU.score_rzlsq = quintupletsInGPU.pt + 6*nMemoryLocations;
}


#ifdef CUT_VALUE_DEBUG
__device__ void SDL::addQuintupletToMemory(struct SDL::triplets& tripletsInGPU, struct SDL::quintuplets& quintupletsInGPU, unsigned int innerTripletIndex, unsigned int outerTripletIndex, unsigned int lowerModule1, unsigned int lowerModule2, unsigned int lowerModule3, unsigned int lowerModule4, unsigned int lowerModule5, float innerRadius, float innerRadiusMin, float innerRadiusMax, float outerRadius, float outerRadiusMin, float outerRadiusMax, float bridgeRadius, float bridgeRadiusMin, float bridgeRadiusMax,
        float innerRadiusMin2S, float innerRadiusMax2S, float bridgeRadiusMin2S, float bridgeRadiusMax2S, float outerRadiusMin2S, float outerRadiusMax2S, float regressionG, float regressionF, float regressionRadius, float chiSquared, float nonAnchorChiSquared, float pt, float eta, float phi, float scores, int layer, unsigned int quintupletIndex)
#else
__device__ void SDL::addQuintupletToMemory(struct SDL::triplets& tripletsInGPU, struct SDL::quintuplets& quintupletsInGPU, unsigned int innerTripletIndex, unsigned int outerTripletIndex, unsigned int lowerModule1, unsigned int lowerModule2, unsigned int lowerModule3, unsigned int lowerModule4, unsigned int lowerModule5, float innerRadius, float outerRadius, float regressionG, float regressionF, float regressionRadius, float pt, float eta, float phi, float scores, int layer, unsigned int quintupletIndex)
#endif

{
    quintupletsInGPU.tripletIndices[2 * quintupletIndex] = innerTripletIndex;
    quintupletsInGPU.tripletIndices[2 * quintupletIndex + 1] = outerTripletIndex;

    quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex] = lowerModule1;
    quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex + 1] = lowerModule2;
    quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex + 2] = lowerModule3;
    quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex + 3] = lowerModule4;
    quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex + 4] = lowerModule5;
    quintupletsInGPU.innerRadius[quintupletIndex] = __F2H(innerRadius);
    quintupletsInGPU.outerRadius[quintupletIndex] = __F2H(outerRadius);
    quintupletsInGPU.pt[quintupletIndex] = __F2H(pt);
    quintupletsInGPU.eta[quintupletIndex] = __F2H(eta);
    quintupletsInGPU.phi[quintupletIndex] = __F2H(phi);
    quintupletsInGPU.score_rphisum[quintupletIndex] = __F2H(scores);
    quintupletsInGPU.layer[quintupletIndex] = layer;
    quintupletsInGPU.isDup[quintupletIndex] = false;
    quintupletsInGPU.regressionRadius[quintupletIndex] = regressionRadius;
    quintupletsInGPU.regressionG[quintupletIndex] = regressionG;
    quintupletsInGPU.regressionF[quintupletIndex] = regressionF;
#ifdef TRACK_EXTENSIONS
    quintupletsInGPU.logicalLayers[5 * quintupletIndex] = tripletsInGPU.logicalLayers[3 * innerTripletIndex];
    quintupletsInGPU.logicalLayers[5 * quintupletIndex + 1] = tripletsInGPU.logicalLayers[3 * innerTripletIndex + 1];
    quintupletsInGPU.logicalLayers[5 * quintupletIndex + 2] = tripletsInGPU.logicalLayers[3 * innerTripletIndex + 2];
    quintupletsInGPU.logicalLayers[5 * quintupletIndex + 3] = tripletsInGPU.logicalLayers[3 * outerTripletIndex + 1];
    quintupletsInGPU.logicalLayers[5 * quintupletIndex + 4] = tripletsInGPU.logicalLayers[3 * outerTripletIndex + 2];

    quintupletsInGPU.hitIndices[10 * quintupletIndex] = tripletsInGPU.hitIndices[6 * innerTripletIndex];
    quintupletsInGPU.hitIndices[10 * quintupletIndex + 1] = tripletsInGPU.hitIndices[6 * innerTripletIndex + 1];
    quintupletsInGPU.hitIndices[10 * quintupletIndex + 2] = tripletsInGPU.hitIndices[6 * innerTripletIndex + 2];
    quintupletsInGPU.hitIndices[10 * quintupletIndex + 3] = tripletsInGPU.hitIndices[6 * innerTripletIndex + 3];
    quintupletsInGPU.hitIndices[10 * quintupletIndex + 4] = tripletsInGPU.hitIndices[6 * innerTripletIndex + 4];
    quintupletsInGPU.hitIndices[10 * quintupletIndex + 5] = tripletsInGPU.hitIndices[6 * innerTripletIndex + 5];
    quintupletsInGPU.hitIndices[10 * quintupletIndex + 6] = tripletsInGPU.hitIndices[6 * outerTripletIndex + 2];
    quintupletsInGPU.hitIndices[10 * quintupletIndex + 7] = tripletsInGPU.hitIndices[6 * outerTripletIndex + 3];
    quintupletsInGPU.hitIndices[10 * quintupletIndex + 8] = tripletsInGPU.hitIndices[6 * outerTripletIndex + 4];
    quintupletsInGPU.hitIndices[10 * quintupletIndex + 9] = tripletsInGPU.hitIndices[6 * outerTripletIndex + 5];
#endif
#ifdef CUT_VALUE_DEBUG
    quintupletsInGPU.innerRadiusMin[quintupletIndex] = innerRadiusMin;
    quintupletsInGPU.innerRadiusMax[quintupletIndex] = innerRadiusMax;
    quintupletsInGPU.outerRadiusMin[quintupletIndex] = outerRadiusMin;
    quintupletsInGPU.outerRadiusMax[quintupletIndex] = outerRadiusMax;
    quintupletsInGPU.bridgeRadius[quintupletIndex] = bridgeRadius;
    quintupletsInGPU.bridgeRadiusMin[quintupletIndex] = bridgeRadiusMin;
    quintupletsInGPU.bridgeRadiusMax[quintupletIndex] = bridgeRadiusMax;
    quintupletsInGPU.innerRadiusMin2S[quintupletIndex] = innerRadiusMin2S;
    quintupletsInGPU.innerRadiusMax2S[quintupletIndex] = innerRadiusMax2S;
    quintupletsInGPU.bridgeRadiusMin2S[quintupletIndex] = bridgeRadiusMin2S;
    quintupletsInGPU.bridgeRadiusMax2S[quintupletIndex] = bridgeRadiusMax2S;
    quintupletsInGPU.outerRadiusMin2S[quintupletIndex] = outerRadiusMin2S;
    quintupletsInGPU.outerRadiusMax2S[quintupletIndex] = outerRadiusMax2S;
    quintupletsInGPU.chiSquared[quintupletIndex] = chiSquared;
    quintupletsInGPU.nonAnchorChiSquared[quintupletIndex] = nonAnchorChiSquared;
#endif

}
__device__ void SDL::rmQuintupletToMemory(struct SDL::quintuplets& quintupletsInGPU,unsigned int quintupletIndex)
{
    quintupletsInGPU.isDup[quintupletIndex] = true;

}

__device__ bool SDL::runQuintupletDefaultAlgo(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, unsigned int lowerModuleIndex1, unsigned int lowerModuleIndex2, unsigned int lowerModuleIndex3, unsigned int lowerModuleIndex4, unsigned int lowerModuleIndex5, unsigned int innerTripletIndex, unsigned int outerTripletIndex, float& innerRadius, float& innerRadiusMin, float&
    innerRadiusMax, float& outerRadius, float& outerRadiusMin, float& outerRadiusMax, float& bridgeRadius, float& bridgeRadiusMin, float& bridgeRadiusMax, float& innerRadiusMin2S, float& innerRadiusMax2S, float& bridgeRadiusMin2S, float& bridgeRadiusMax2S, float& outerRadiusMin2S, float& outerRadiusMax2S, float& regressionG, float& regressionF, float& regressionRadius, float& chiSquared, float& nonAnchorChiSquared)
{
    bool pass = true;

    unsigned int firstSegmentIndex = tripletsInGPU.segmentIndices[2 * innerTripletIndex];
    unsigned int secondSegmentIndex = tripletsInGPU.segmentIndices[2 * innerTripletIndex + 1];
    unsigned int thirdSegmentIndex = tripletsInGPU.segmentIndices[2 * outerTripletIndex];
    unsigned int fourthSegmentIndex = tripletsInGPU.segmentIndices[2 * outerTripletIndex + 1];

    unsigned int innerOuterOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * secondSegmentIndex + 1]; //inner triplet outer segment outer MD index
    unsigned int outerInnerInnerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * thirdSegmentIndex]; //outer triplet inner segmnet inner MD index

    if (innerOuterOuterMiniDoubletIndex != outerInnerInnerMiniDoubletIndex) pass = false;


    //apply T4 criteria between segments 1 and 3
    float zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, pt_beta; //temp stuff
    float zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ;
    if(not runTrackletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, segmentsInGPU.innerLowerModuleIndices[firstSegmentIndex], segmentsInGPU.outerLowerModuleIndices[firstSegmentIndex], segmentsInGPU.innerLowerModuleIndices[thirdSegmentIndex], segmentsInGPU.outerLowerModuleIndices[thirdSegmentIndex], firstSegmentIndex, thirdSegmentIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, pt_beta, zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ,600))
    {
        pass = false;
    }
    if(not runTrackletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, segmentsInGPU.innerLowerModuleIndices[firstSegmentIndex], segmentsInGPU.outerLowerModuleIndices[firstSegmentIndex], segmentsInGPU.innerLowerModuleIndices[fourthSegmentIndex], segmentsInGPU.outerLowerModuleIndices[fourthSegmentIndex], firstSegmentIndex, fourthSegmentIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, pt_beta, zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ,600))
    {
        pass = false;
    }
    //radius computation from the three triplet MD anchor hits
    unsigned int innerTripletFirstSegmentAnchorHitIndex = segmentsInGPU.innerMiniDoubletAnchorHitIndices[firstSegmentIndex];
    unsigned int innerTripletSecondSegmentAnchorHitIndex = segmentsInGPU.outerMiniDoubletAnchorHitIndices[firstSegmentIndex]; //same as second segment inner MD anchorhit index
    unsigned int innerTripletThirdSegmentAnchorHitIndex = segmentsInGPU.outerMiniDoubletAnchorHitIndices[secondSegmentIndex]; //same as third segment inner MD anchor hit index

    unsigned int outerTripletSecondSegmentAnchorHitIndex = segmentsInGPU.outerMiniDoubletAnchorHitIndices[thirdSegmentIndex]; //same as fourth segment inner MD anchor hit index
    unsigned int outerTripletThirdSegmentAnchorHitIndex = segmentsInGPU.outerMiniDoubletAnchorHitIndices[fourthSegmentIndex];

    if(not passT5RZConstraint(modulesInGPU, hitsInGPU, innerTripletFirstSegmentAnchorHitIndex, innerTripletSecondSegmentAnchorHitIndex, innerTripletThirdSegmentAnchorHitIndex, outerTripletSecondSegmentAnchorHitIndex, outerTripletThirdSegmentAnchorHitIndex, lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4, lowerModuleIndex5))
    {
        pass = false;
    }

    float x1 = hitsInGPU.xs[innerTripletFirstSegmentAnchorHitIndex];
    float x2 = hitsInGPU.xs[innerTripletSecondSegmentAnchorHitIndex];
    float x3 = hitsInGPU.xs[innerTripletThirdSegmentAnchorHitIndex];
    float x4 = hitsInGPU.xs[outerTripletSecondSegmentAnchorHitIndex];
    float x5 = hitsInGPU.xs[outerTripletThirdSegmentAnchorHitIndex];

    float y1 = hitsInGPU.ys[innerTripletFirstSegmentAnchorHitIndex];
    float y2 = hitsInGPU.ys[innerTripletSecondSegmentAnchorHitIndex];
    float y3 = hitsInGPU.ys[innerTripletThirdSegmentAnchorHitIndex];
    float y4 = hitsInGPU.ys[outerTripletSecondSegmentAnchorHitIndex];
    float y5 = hitsInGPU.ys[outerTripletThirdSegmentAnchorHitIndex];


    unsigned int firstMDIndex = segmentsInGPU.mdIndices[2 * firstSegmentIndex];
    unsigned int secondMDIndex = segmentsInGPU.mdIndices[2 * secondSegmentIndex];
    unsigned int thirdMDIndex = segmentsInGPU.mdIndices[2 * secondSegmentIndex + 1];
    unsigned int fourthMDIndex = segmentsInGPU.mdIndices[2 * thirdSegmentIndex + 1];
    unsigned int fifthMDIndex = segmentsInGPU.mdIndices[2 * fourthSegmentIndex + 1];

    unsigned int nonAnchor1 = mdsInGPU.hitIndices[2 * firstMDIndex] == innerTripletFirstSegmentAnchorHitIndex ? mdsInGPU.hitIndices[2 * firstMDIndex + 1] : mdsInGPU.hitIndices[2 * firstMDIndex];
    unsigned int nonAnchor2 = mdsInGPU.hitIndices[2 * secondMDIndex] == innerTripletSecondSegmentAnchorHitIndex ? mdsInGPU.hitIndices[2 * secondMDIndex + 1] : mdsInGPU.hitIndices[2 * secondMDIndex];
    unsigned int nonAnchor3 = mdsInGPU.hitIndices[2 * thirdMDIndex] == innerTripletThirdSegmentAnchorHitIndex ? mdsInGPU.hitIndices[2 * thirdMDIndex + 1] : mdsInGPU.hitIndices[2 * thirdMDIndex];
    unsigned int nonAnchor4 = mdsInGPU.hitIndices[2 * fourthMDIndex] == outerTripletSecondSegmentAnchorHitIndex ? mdsInGPU.hitIndices[2 * fourthMDIndex + 1] : mdsInGPU.hitIndices[2 * fourthMDIndex];
    unsigned int nonAnchor5 = mdsInGPU.hitIndices[2 * fifthMDIndex] == outerTripletThirdSegmentAnchorHitIndex ? mdsInGPU.hitIndices[2 * fifthMDIndex + 1] : mdsInGPU.hitIndices[2 * fifthMDIndex];

    //non anchor is always shifted for tilted and endcap!

    float x1NonAnchor = hitsInGPU.xs[nonAnchor1];
    float x2NonAnchor = hitsInGPU.xs[nonAnchor2];
    float x3NonAnchor = hitsInGPU.xs[nonAnchor3];
    float x4NonAnchor = hitsInGPU.xs[nonAnchor4];
    float x5NonAnchor = hitsInGPU.xs[nonAnchor5];

    float y1NonAnchor = hitsInGPU.ys[nonAnchor1];
    float y2NonAnchor = hitsInGPU.ys[nonAnchor2];
    float y3NonAnchor = hitsInGPU.ys[nonAnchor3];
    float y4NonAnchor = hitsInGPU.ys[nonAnchor4];
    float y5NonAnchor = hitsInGPU.ys[nonAnchor5];

    //construct the arrays
    float x1Vec[] = {x1, x1, x1};
    float y1Vec[] = {y1, y1, y1};
    float x2Vec[] = {x2, x2, x2};
    float y2Vec[] = {y2, y2, y2};
    float x3Vec[] = {x3, x3, x3};
    float y3Vec[] = {y3, y3, y3};
    //float x4Vec[] = {x4, x4, x4};
    //float y4Vec[] = {y4, y4, y4};
    //float x5Vec[] = {x5, x5, x5};
    //float y5Vec[] = {y5, y5, y5};

    if(modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex1] == SDL::TwoS)
    {
        x1Vec[1] = hitsInGPU.lowEdgeXs[innerTripletFirstSegmentAnchorHitIndex];
        x1Vec[2] = hitsInGPU.highEdgeXs[innerTripletFirstSegmentAnchorHitIndex];

        y1Vec[1] = hitsInGPU.lowEdgeYs[innerTripletFirstSegmentAnchorHitIndex];
        y1Vec[2] = hitsInGPU.highEdgeYs[innerTripletFirstSegmentAnchorHitIndex];
    }
    if(modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex2] == SDL::TwoS)
    {
        x2Vec[1] = hitsInGPU.lowEdgeXs[innerTripletSecondSegmentAnchorHitIndex];
        x2Vec[2] = hitsInGPU.highEdgeXs[innerTripletSecondSegmentAnchorHitIndex];

        y2Vec[1] = hitsInGPU.lowEdgeYs[innerTripletSecondSegmentAnchorHitIndex];
        y2Vec[2] = hitsInGPU.highEdgeYs[innerTripletSecondSegmentAnchorHitIndex];

    }
    if(modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex3] == SDL::TwoS)
    {
        x3Vec[1] = hitsInGPU.lowEdgeXs[innerTripletThirdSegmentAnchorHitIndex];
        x3Vec[2] = hitsInGPU.highEdgeXs[innerTripletThirdSegmentAnchorHitIndex];

        y3Vec[1] = hitsInGPU.lowEdgeYs[innerTripletThirdSegmentAnchorHitIndex];
        y3Vec[2] = hitsInGPU.highEdgeYs[innerTripletThirdSegmentAnchorHitIndex];
    }
    computeErrorInRadius(x1Vec, y1Vec, x2Vec, y2Vec, x3Vec, y3Vec, innerRadiusMin2S, innerRadiusMax2S);

    for (int i=0; i<3; i++) {
      x1Vec[i] = x4;
      y1Vec[i] = y4;
    }
    if(modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex4] == SDL::TwoS)
    {
        x1Vec[1] = hitsInGPU.lowEdgeXs[outerTripletSecondSegmentAnchorHitIndex];
        x1Vec[2] = hitsInGPU.highEdgeXs[outerTripletSecondSegmentAnchorHitIndex];

        y1Vec[1] = hitsInGPU.lowEdgeYs[outerTripletSecondSegmentAnchorHitIndex];
        y1Vec[2] = hitsInGPU.highEdgeYs[outerTripletSecondSegmentAnchorHitIndex];
    }
    computeErrorInRadius(x2Vec, y2Vec, x3Vec, y3Vec, x1Vec, y1Vec, bridgeRadiusMin2S, bridgeRadiusMax2S);

    for(int i=0; i<3; i++) {
      x2Vec[i] = x5;
      y2Vec[i] = y5;
    }
    if(modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex5] == SDL::TwoS)
    {
        x2Vec[1] = hitsInGPU.lowEdgeXs[outerTripletThirdSegmentAnchorHitIndex];
        x2Vec[2] = hitsInGPU.highEdgeXs[outerTripletThirdSegmentAnchorHitIndex];

        y2Vec[1] = hitsInGPU.lowEdgeYs[outerTripletThirdSegmentAnchorHitIndex];
        y2Vec[2] = hitsInGPU.highEdgeYs[outerTripletThirdSegmentAnchorHitIndex];
    }
    computeErrorInRadius(x3Vec, y3Vec, x1Vec, y1Vec, x2Vec, y2Vec, outerRadiusMin2S, outerRadiusMax2S);

    float g, f;
    innerRadius = computeRadiusFromThreeAnchorHits(x1, y1, x2, y2, x3, y3, g, f);
    outerRadius = computeRadiusFromThreeAnchorHits(x3, y3, x4, y4, x5, y5, g, f);
    bridgeRadius = computeRadiusFromThreeAnchorHits(x2, y2, x3, y3, x4, y4, g, f);


    if(innerRadius < 0.95f/(2.f * k2Rinv1GeVf))
    {
        pass = false;
    }
    //split by category
    bool tempPass;
    if(modulesInGPU.subdets[lowerModuleIndex1] == SDL::Barrel and modulesInGPU.subdets[lowerModuleIndex2] == SDL::Barrel and modulesInGPU.subdets[lowerModuleIndex3] == SDL::Barrel and modulesInGPU.subdets[lowerModuleIndex4] == SDL::Barrel and modulesInGPU.subdets[lowerModuleIndex5] == SDL::Barrel)
    {
       tempPass = matchRadiiBBBBB(innerRadius, bridgeRadius, outerRadius, innerRadiusMin, innerRadiusMax, bridgeRadiusMin, bridgeRadiusMax, outerRadiusMin, outerRadiusMax);
    }
    else if(modulesInGPU.subdets[lowerModuleIndex1] == SDL::Barrel and modulesInGPU.subdets[lowerModuleIndex2] == SDL::Barrel and modulesInGPU.subdets[lowerModuleIndex3] == SDL::Barrel and modulesInGPU.subdets[lowerModuleIndex4] == SDL::Barrel and modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap)
    {
        tempPass = matchRadiiBBBBE(innerRadius, bridgeRadius, outerRadius, innerRadiusMin2S, innerRadiusMax2S, bridgeRadiusMin2S, bridgeRadiusMax2S, outerRadiusMin2S, outerRadiusMax2S, innerRadiusMin, innerRadiusMax, bridgeRadiusMin, bridgeRadiusMax, outerRadiusMin, outerRadiusMax);
    }
    else if(modulesInGPU.subdets[lowerModuleIndex1] == SDL::Barrel and modulesInGPU.subdets[lowerModuleIndex2] == SDL::Barrel and modulesInGPU.subdets[lowerModuleIndex3] == SDL::Barrel and modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap and modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap)
    {
        if(modulesInGPU.layers[lowerModuleIndex1] == 1)
        {
            tempPass = matchRadiiBBBEE12378(innerRadius, bridgeRadius, outerRadius,innerRadiusMin2S, innerRadiusMax2S, bridgeRadiusMin2S, bridgeRadiusMax2S, outerRadiusMin2S, outerRadiusMax2S, innerRadiusMin, innerRadiusMax, bridgeRadiusMin, bridgeRadiusMax, outerRadiusMin, outerRadiusMax);
        }
        else if(modulesInGPU.layers[lowerModuleIndex1] == 2)
        {
            tempPass = matchRadiiBBBEE23478(innerRadius, bridgeRadius, outerRadius,innerRadiusMin2S, innerRadiusMax2S, bridgeRadiusMin2S, bridgeRadiusMax2S, outerRadiusMin2S, outerRadiusMax2S, innerRadiusMin, innerRadiusMax, bridgeRadiusMin, bridgeRadiusMax, outerRadiusMin, outerRadiusMax);
        }
        else
        {
            tempPass = matchRadiiBBBEE34578(innerRadius, bridgeRadius, outerRadius,innerRadiusMin2S, innerRadiusMax2S, bridgeRadiusMin2S, bridgeRadiusMax2S, outerRadiusMin2S, outerRadiusMax2S, innerRadiusMin, innerRadiusMax, bridgeRadiusMin, bridgeRadiusMax, outerRadiusMin, outerRadiusMax);
        }
    }

    else if(modulesInGPU.subdets[lowerModuleIndex1] == SDL::Barrel and modulesInGPU.subdets[lowerModuleIndex2] == SDL::Barrel and modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap and modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap and modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap)
    {
        tempPass = matchRadiiBBEEE(innerRadius, bridgeRadius, outerRadius, innerRadiusMin2S, innerRadiusMax2S, bridgeRadiusMin2S, bridgeRadiusMax2S, outerRadiusMin2S, outerRadiusMax2S, innerRadiusMin, innerRadiusMax, bridgeRadiusMin, bridgeRadiusMax, outerRadiusMin, outerRadiusMax);
    }
    else if(modulesInGPU.subdets[lowerModuleIndex1] == SDL::Barrel and modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap and modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap and modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap and modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap)
    {
        tempPass = matchRadiiBEEEE(innerRadius, bridgeRadius, outerRadius, innerRadiusMin2S, innerRadiusMax2S, bridgeRadiusMin2S, bridgeRadiusMax2S, outerRadiusMin2S, outerRadiusMax2S, innerRadiusMin, innerRadiusMax, bridgeRadiusMin, bridgeRadiusMax, outerRadiusMin, outerRadiusMax);
    }
    else
    {
        tempPass = matchRadiiEEEEE(innerRadius, bridgeRadius, outerRadius, innerRadiusMin2S, innerRadiusMax2S, bridgeRadiusMin2S, bridgeRadiusMax2S, outerRadiusMin2S, outerRadiusMax2S,innerRadiusMin, innerRadiusMax, bridgeRadiusMin, bridgeRadiusMax, outerRadiusMin, outerRadiusMax);
    }

    pass = pass & tempPass;


    //compute regression radius right here
    float xVec[] = {x1, x2, x3, x4, x5};
    float yVec[] = {y1, y2, y3, y4, y5};
    float sigmas[5], delta1[5], delta2[5], slopes[5];
    bool isFlat[5];
    //5 categories for sigmas
    const unsigned int lowerModuleIndices[] = {lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4, lowerModuleIndex5};

    computeSigmasForRegression(modulesInGPU, lowerModuleIndices, delta1, delta2, slopes, isFlat);
    regressionRadius = computeRadiusUsingRegression(5,xVec, yVec, delta1, delta2, slopes, isFlat, regressionG, regressionF, sigmas, chiSquared);

    //extra chi squared cuts!
    if(regressionRadius < 5.0f/(2.f * k2Rinv1GeVf))
    {
        pass = pass & passChiSquaredConstraint(modulesInGPU, lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4, lowerModuleIndex5, chiSquared);
    }
    //compute the other chisquared
    float nonAnchorDelta1[5], nonAnchorDelta2[5], nonAnchorSlopes[5];
    float nonAnchorxs[] = {x1NonAnchor, x2NonAnchor, x3NonAnchor, x4NonAnchor, x5NonAnchor};
    float nonAnchorys[] = {y1NonAnchor, y2NonAnchor, y3NonAnchor, y4NonAnchor, y5NonAnchor};

    computeSigmasForRegression(modulesInGPU, lowerModuleIndices, nonAnchorDelta1, nonAnchorDelta2, nonAnchorSlopes, isFlat, 5, false);
    nonAnchorChiSquared = computeChiSquared(5, nonAnchorxs, nonAnchorys, nonAnchorDelta1, nonAnchorDelta2, nonAnchorSlopes, isFlat, regressionG, regressionF, regressionRadius);
    return pass;
}

//90% constraint
__device__ bool SDL::passChiSquaredConstraint(struct SDL::modules& modulesInGPU, unsigned int lowerModuleIndex1, unsigned int lowerModuleIndex2, unsigned int lowerModuleIndex3, unsigned int lowerModuleIndex4, unsigned int lowerModuleIndex5, float& chiSquared)
{
    //following Philip's layer number prescription
    const int layer1 = modulesInGPU.layers[lowerModuleIndex1] + 6 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex1] == SDL::TwoS);
    const int layer2 = modulesInGPU.layers[lowerModuleIndex2] + 6 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex2] == SDL::TwoS);
    const int layer3 = modulesInGPU.layers[lowerModuleIndex3] + 6 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex3] == SDL::TwoS);
    const int layer4 = modulesInGPU.layers[lowerModuleIndex4] + 6 * (modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex4] == SDL::TwoS);
    const int layer5 = modulesInGPU.layers[lowerModuleIndex5] + 6 * (modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex5] == SDL::TwoS);

    if(layer1 == 7 and layer2 == 8 and layer3 == 9)
    {
        if(layer4 == 10 and layer5 == 11)
        {
            return chiSquared < 0.01788f;
        }
        else if(layer4 == 10 and layer5 == 16)
        {
            return chiSquared < 0.04725f;
        }
        else if(layer4 == 15 and layer5 == 16)
        {
            return chiSquared < 0.04725f;
        }
    }
    else if(layer1 == 1 and layer2 == 7 and layer3 == 8)
    {
        if(layer4 == 9 and layer5 == 10)
        {       
            return chiSquared < 0.01788f;
        }   
        else if(layer4 == 9 and layer5 == 15)
        {
            return chiSquared < 0.08234f;
        }
    }
    else if(layer1 == 1 and layer2 == 2 and layer3 == 7)
    {
        if(layer4 == 8 and layer5 == 9)
        {
            return chiSquared < 0.02360f;
        }
        else if(layer4 == 8 and layer5 == 14)
        {
            return chiSquared < 0.07167f;
        }
        else if(layer4 == 13 and layer5 == 14)
        {   
            return chiSquared < 0.08234f;
        }
    }
    else if(layer1 == 1 and layer2 == 2 and layer3 == 3)
    {
        if(layer4 == 7 and layer5 == 8)
        {
            return chiSquared < 0.01026f;
        }
        else if(layer4 == 7 and layer5 == 13)
        {
            return chiSquared < 0.06238f;
        }
        else if(layer4 == 12 and layer5 == 13)
        {
            return chiSquared < 0.06238f;
        }
    }
    else if(layer1 == 1 and layer2 == 2 and layer3 == 3 and layer4 == 4)
    {
        if(layer5 == 12)
        {
            return chiSquared < 0.09461f;
        }
        else if(layer5 == 5)
        {
            return chiSquared < 0.04725f;
        }
    }
    else if(layer1 == 2 and layer2 == 7 and layer3 == 8)
    {
        if(layer4 == 9 and layer5 == 10)
        {
            return chiSquared < 0.00512f;
        }
        if(layer4 == 9 and layer5 == 15)
        {
            return chiSquared < 0.04112f;
        }
        else if(layer4 == 14 and layer5 == 15)
        {
            return chiSquared < 0.06238f;
        }
    }
    else if(layer1 == 2 and layer2 == 3 and layer3 == 7)
    {
        if(layer4 == 8 and layer5 == 14)
        {
            return chiSquared < 0.07167f;
        }
        else if(layer4 == 13 and layer5 == 14)
        {
            return chiSquared < 0.06238f;
        }
    }
    else if(layer1 == 2 and layer2 == 3 and layer3 == 4)
    {
        if(layer4 == 12 and layer5 == 13)
        {
            return chiSquared < 0.10870f;
        }
        else if(layer4 == 5 and layer5 == 12)
        {
            return chiSquared < 0.10870f;
        }
        else if(layer4 == 5 and layer5 == 6)
        {
            return chiSquared < 0.08234f;
        }
    }
    else if(layer1 == 3 and layer2 == 7 and layer3 == 8 and layer4 == 14 and layer5 == 15)
    {
        return chiSquared < 0.09461f;
    }
    else if(layer1 == 3 and layer2 == 4 and layer3 == 5 and layer4 == 12 and layer5 == 13)
    {
        return chiSquared < 0.09461f;
    }

    return true;
}

//bounds can be found at http://uaf-10.t2.ucsd.edu/~bsathian/SDL/T5_RZFix/t5_rz_thresholds.txt
__device__ bool SDL::passT5RZConstraint(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, unsigned int anchorHitIndex1, unsigned int anchorHitIndex2, unsigned int anchorHitIndex3, unsigned int anchorHitIndex4, unsigned int anchorHitIndex5, unsigned int lowerModuleIndex1, unsigned int lowerModuleIndex2, unsigned int lowerModuleIndex3, unsigned int lowerModuleIndex4, unsigned int lowerModuleIndex5) 
{
    const float& rt1 = hitsInGPU.rts[anchorHitIndex1];
    const float& rt2 = hitsInGPU.rts[anchorHitIndex2];
    const float& rt3 = hitsInGPU.rts[anchorHitIndex3];
    const float& rt4 = hitsInGPU.rts[anchorHitIndex4];
    const float& rt5 = hitsInGPU.rts[anchorHitIndex5];

    const float& z1 = hitsInGPU.zs[anchorHitIndex1];
    const float& z2 = hitsInGPU.zs[anchorHitIndex2];
    const float& z3 = hitsInGPU.zs[anchorHitIndex3];
    const float& z4 = hitsInGPU.zs[anchorHitIndex4];
    const float& z5 = hitsInGPU.zs[anchorHitIndex5];

    //following Philip's layer number prescription
    const int layer1 = modulesInGPU.layers[lowerModuleIndex1] + 6 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex1] == SDL::TwoS);
    const int layer2 = modulesInGPU.layers[lowerModuleIndex2] + 6 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex2] == SDL::TwoS);
    const int layer3 = modulesInGPU.layers[lowerModuleIndex3] + 6 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex3] == SDL::TwoS);
    const int layer4 = modulesInGPU.layers[lowerModuleIndex4] + 6 * (modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex4] == SDL::TwoS);
    const int layer5 = modulesInGPU.layers[lowerModuleIndex5] + 6 * (modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex5] == SDL::TwoS);

    //slope computed using the internal T3s
    const int moduleLayer1 = modulesInGPU.moduleType[lowerModuleIndex1];
    const int moduleLayer2 = modulesInGPU.moduleType[lowerModuleIndex2];
    const int moduleLayer3 = modulesInGPU.moduleType[lowerModuleIndex3];
    const int moduleLayer4 = modulesInGPU.moduleType[lowerModuleIndex4];
    const int moduleLayer5 = modulesInGPU.moduleType[lowerModuleIndex5];

    float slope;
    if(moduleLayer1 == 0 and moduleLayer2 == 0 and moduleLayer3 == 1) //PSPS2S
    {
        slope = (z2 -z1)/(rt2 - rt1);
    }
    else
    {
        slope = (z3 - z1)/(rt3 - rt1);
    }
    float residual4 = (layer4 <= 6)? ((z4 - z1) - slope * (rt4 - rt1)) : ((rt4 - rt1) - (z4 - z1)/slope);
    float residual5 = (layer4 <= 6) ? ((z5 - z1) - slope * (rt5 - rt1)) : ((rt5 - rt1) - (z5 - z1)/slope);

    // creating a chi squared type quantity
    // 0-> PS, 1->2S
    residual4 = (moduleLayer4 == 0) ? residual4/2.4f : residual4/5.0f;
    residual5 = (moduleLayer5 == 0) ? residual5/2.4f : residual5/5.0f;

    const float RMSE = sqrtf(0.5 * (residual4 * residual4 + residual5 * residual5));

    //categories!
    if(layer1 == 1 and layer2 == 2 and layer3 == 3)
    {
        if(layer4 == 4 and layer5 == 5)
        {
            return RMSE < 0.545f; 
        }
        else if(layer4 == 4 and layer5 == 12)
        {
            return RMSE < 1.105f;
        }
        else if(layer4 == 7 and layer5 == 13)
        {
            return RMSE < 0.775f;
        }
        else if(layer4 == 12 and layer5 == 13)
        {
            return RMSE < 0.625f;
        }
    }
    else if(layer1 == 1 and layer2 == 2 and layer3 == 7)
    {
        if(layer4 == 8 and layer5 == 14)
        {
            return RMSE < 0.835f;
        }
        else if(layer4 == 13 and layer5 == 14)
        {
            return RMSE < 0.575f;
        }
    }
    else if(layer1 == 1 and layer2 == 7 and layer3 == 8 and layer4 == 9 and layer5 == 15)
    {
        return RMSE < 0.825f;
    }
    else if(layer1 == 2 and layer2 == 3 and layer3 == 4)
    {
        if(layer4 == 5 and layer5 == 6)
        {
            return RMSE < 0.845f;
        }
        else if(layer4 == 5 and layer5 == 12)
        {
            return RMSE < 1.365f;
        }

        else if(layer4 == 12 and layer5 == 13)
        {
            return RMSE < 0.675f;
        }
    }
    else if(layer1 == 2 and layer2 == 3 and layer3 == 7 and layer4 == 13 and layer5 == 14)
    {
            return RMSE < 0.495f;
    }
    else if(layer1 == 2 and layer2 == 3 and layer3 == 12 and layer4 == 13 and layer5 == 14)
    {
        return RMSE < 0.695f; 
    }
    else if(layer1 == 2 and layer2 == 7 and layer3 == 8)
    {
        if(layer4 == 9 and layer5 == 15)
        {
            return RMSE < 0.735f;
        }
        else if(layer4 == 14 and layer5 == 15)
        {
            return RMSE < 0.525f;
        }
    }
    else if(layer1 == 2 and layer2 == 7 and layer3 == 13 and layer4 == 14 and layer5 == 15)
    {
        return RMSE < 0.665f;
    }
    else if(layer1 == 3 and layer2 == 4 and layer3 == 5 and layer4 == 12 and layer5 == 13)
    {
        return RMSE < 0.995f;
    }
    else if(layer1 == 3 and layer2 == 4 and layer3 == 12 and layer4 == 13 and layer5 == 14)
    {
        return RMSE < 0.525f;
    }
    else if(layer1 == 3 and layer2 == 7 and layer3 == 8 and layer4 == 14 and layer5 == 15)
    {
        return RMSE < 0.525f;
    }
    else if(layer1 == 3 and layer2 == 7 and layer3 == 13 and layer4 == 14 and layer5 == 15)
    {
        return RMSE < 0.745f;
    }
    else if(layer1 == 3 and layer2 == 12 and layer3 == 13 and layer4 == 14 and layer5 == 15)
    {
        return RMSE < 0.555f; 
    }
    else if(layer1 == 7 and layer2 == 8 and layer3 == 9 and layer4 == 15 and layer5 == 16)
    {
            return RMSE < 0.525f;
    }
    else if(layer1 == 7 and layer2 == 8 and layer3 == 14 and layer4 == 15 and layer5 == 16)
    {
        return RMSE < 0.885f;
    }
    else if(layer1 == 7 and layer2 == 13 and layer3 == 14 and layer4 == 15 and layer5 == 16)
    {
        return RMSE < 0.845f;
    }

    return true;
}

__device__ bool SDL::checkIntervalOverlap(const float& firstMin, const float& firstMax, const float& secondMin, const float& secondMax)
{
    return ((firstMin <= secondMin) & (secondMin < firstMax)) |  ((secondMin < firstMin) & (firstMin < secondMax));
}

/*bounds for high Pt taken from : http://uaf-10.t2.ucsd.edu/~bsathian/SDL/T5_efficiency/efficiencies/new_efficiencies/efficiencies_20210513_T5_recovering_high_Pt_efficiencies/highE_radius_matching/highE_bounds.txt */

__device__ bool SDL::matchRadiiBBBBB(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax)
{
    float innerInvRadiusErrorBound =  0.1512f;
    float bridgeInvRadiusErrorBound = 0.1781f;
    float outerInvRadiusErrorBound = 0.1840f;

    if(innerRadius > 2.0f/(2.f * k2Rinv1GeVf))
    {
        innerInvRadiusErrorBound = 0.4449f;
        bridgeInvRadiusErrorBound = 0.4033f;
        outerInvRadiusErrorBound = 0.8016f;
    }

    innerRadiusMin = innerRadius/(1.f + innerInvRadiusErrorBound);
    innerRadiusMax = innerInvRadiusErrorBound < 1 ? innerRadius/(1.f - innerInvRadiusErrorBound) : 123456789.f;

    bridgeRadiusMin = bridgeRadius/(1.f + bridgeInvRadiusErrorBound);
    bridgeRadiusMax = bridgeInvRadiusErrorBound < 1 ? bridgeRadius/(1.f - bridgeInvRadiusErrorBound) : 123456789.f;

    outerRadiusMin = outerRadius/(1.f + outerInvRadiusErrorBound);
    outerRadiusMax = outerInvRadiusErrorBound < 1 ? outerRadius/(1.f - outerInvRadiusErrorBound) : 123456789.f;

    return checkIntervalOverlap(1.0f/innerRadiusMax, 1.0f/innerRadiusMin, 1.0f/bridgeRadiusMax, 1.0f/bridgeRadiusMin);
}

__device__ bool SDL::matchRadiiBBBBE(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S, float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax)
{

    float innerInvRadiusErrorBound =  0.1781f;
    float bridgeInvRadiusErrorBound = 0.2167f;
    float outerInvRadiusErrorBound = 1.1116f;

    if(innerRadius > 2.0f/(2.f * k2Rinv1GeVf))
    {
        innerInvRadiusErrorBound = 0.4750f;
        bridgeInvRadiusErrorBound = 0.3903f;
        outerInvRadiusErrorBound = 15.2120f;
    }


    innerRadiusMin = innerRadius/(1.f + innerInvRadiusErrorBound);
    innerRadiusMax = innerInvRadiusErrorBound < 1 ? innerRadius/(1.f - innerInvRadiusErrorBound) : 123456789.f; //large number signifying infty

    bridgeRadiusMin = bridgeRadius/(1.f + bridgeInvRadiusErrorBound);
    bridgeRadiusMax = bridgeInvRadiusErrorBound < 1 ? bridgeRadius/(1.f - bridgeInvRadiusErrorBound) : 123456789.f;

    outerRadiusMin = outerRadius/(1.f + outerInvRadiusErrorBound);
    outerRadiusMax = outerInvRadiusErrorBound < 1 ? outerRadius/(1.f - outerInvRadiusErrorBound) : 123456789.f;

    return checkIntervalOverlap(1.0f/innerRadiusMax, 1.0f/innerRadiusMin, 1.0f/bridgeRadiusMax, 1.0f/bridgeRadiusMin);
}

__device__ bool SDL::matchRadiiBBBEE12378(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S, float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax)
{
    float innerInvRadiusErrorBound = 0.178f;
    float bridgeInvRadiusErrorBound = 0.507f;
    float outerInvRadiusErrorBound = 7.655f;

    innerRadiusMin = innerRadius/(1.f + innerInvRadiusErrorBound);
    innerRadiusMax = innerInvRadiusErrorBound < 1 ? innerRadius/(1.f - innerInvRadiusErrorBound) : 123456789.f;

    bridgeRadiusMin = bridgeRadius/(1.f + bridgeInvRadiusErrorBound);
    bridgeRadiusMax = bridgeInvRadiusErrorBound < 1 ? bridgeRadius/(1.f - bridgeInvRadiusErrorBound) : 123456789.f;

    outerRadiusMin = outerRadius/(1.f + outerInvRadiusErrorBound);
    outerRadiusMax = outerInvRadiusErrorBound < 1 ? outerRadius/(1.f - outerInvRadiusErrorBound) : 123456789.f;

    return checkIntervalOverlap(1.0f/innerRadiusMax, 1.0f/innerRadiusMin, 1.0f/fmaxf(bridgeRadiusMax, bridgeRadiusMax2S),1.0f/fminf(bridgeRadiusMin, bridgeRadiusMin2S));
}

__device__ bool SDL::matchRadiiBBBEE23478(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S, float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax)
{
    float innerInvRadiusErrorBound = 0.2097f;
    float bridgeInvRadiusErrorBound = 0.8557f;
    float outerInvRadiusErrorBound = 24.0450f;

    innerRadiusMin = innerRadius/(1.f + innerInvRadiusErrorBound);
    innerRadiusMax = innerInvRadiusErrorBound < 1 ? innerRadius/(1.f - innerInvRadiusErrorBound) : 123456789.f;

    bridgeRadiusMin = bridgeRadius/(1.f + bridgeInvRadiusErrorBound);
    bridgeRadiusMax = bridgeInvRadiusErrorBound < 1 ? bridgeRadius/(1.f - bridgeInvRadiusErrorBound) : 123456789.f;

    outerRadiusMin = outerRadius/(1.f + outerInvRadiusErrorBound);
    outerRadiusMax = outerInvRadiusErrorBound < 1 ? outerRadius/(1.f - outerInvRadiusErrorBound) : 123456789.f;

    return checkIntervalOverlap(1.0f/innerRadiusMax, 1.0f/innerRadiusMin, 1.0f/fmaxf(bridgeRadiusMax, bridgeRadiusMax2S), 1.0f/fminf(bridgeRadiusMin, bridgeRadiusMin2S));

}

__device__ bool SDL::matchRadiiBBBEE34578(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S, float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax)
{
    float innerInvRadiusErrorBound = 0.066f;
    float bridgeInvRadiusErrorBound = 0.617f;
    float outerInvRadiusErrorBound = 2.688f;

    innerRadiusMin = innerRadius/(1.f + innerInvRadiusErrorBound);
    innerRadiusMax = innerInvRadiusErrorBound < 1 ? innerRadius/(1.f - innerInvRadiusErrorBound) : 123456789.f;

    bridgeRadiusMin = bridgeRadius/(1.f + bridgeInvRadiusErrorBound);
    bridgeRadiusMax = bridgeInvRadiusErrorBound < 1 ? bridgeRadius/(1.f - bridgeInvRadiusErrorBound) : 123456789.f;

    outerRadiusMin = outerRadius/(1.f + outerInvRadiusErrorBound);
    outerRadiusMax = outerInvRadiusErrorBound < 1 ? outerRadius/(1.f - outerInvRadiusErrorBound) : 123456789.f;

    return checkIntervalOverlap(1.0f/innerRadiusMax, 1.0f/innerRadiusMin, 1.0f/fmaxf(bridgeRadiusMax, bridgeRadiusMax2S), 1.0f/fminf(bridgeRadiusMin, bridgeRadiusMin2S));

}

__device__ bool SDL::matchRadiiBBBEE(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S, float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax)
{

    float innerInvRadiusErrorBound =  0.1840f;
    float bridgeInvRadiusErrorBound = 0.5971f;
    float outerInvRadiusErrorBound = 11.7102f;

    if(innerRadius > 2.0f/(2.f * k2Rinv1GeVf)) //as good as no selections
    {
        innerInvRadiusErrorBound = 1.0412f;
        outerInvRadiusErrorBound = 32.2737f;
        bridgeInvRadiusErrorBound = 10.9688f;
    }

    innerRadiusMin = innerRadius/(1.f + innerInvRadiusErrorBound);
    innerRadiusMax = innerInvRadiusErrorBound < 1 ? innerRadius/(1.f - innerInvRadiusErrorBound) : 123456789.f;

    bridgeRadiusMin = bridgeRadius/(1.f + bridgeInvRadiusErrorBound);
    bridgeRadiusMax = bridgeInvRadiusErrorBound < 1 ? bridgeRadius/(1.f - bridgeInvRadiusErrorBound) : 123456789.f;

    outerRadiusMin = outerRadius/(1.f + outerInvRadiusErrorBound);
    outerRadiusMax = outerInvRadiusErrorBound < 1 ? outerRadius/(1.f - outerInvRadiusErrorBound) : 123456789.f;

    return checkIntervalOverlap(1.0f/innerRadiusMax, 1.0f/innerRadiusMin, 1.0f/fmaxf(bridgeRadiusMax, bridgeRadiusMax2S), 1.0f/fminf(bridgeRadiusMin, bridgeRadiusMin2S));

}

__device__ bool SDL::matchRadiiBBEEE(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S, float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax)
{

    float innerInvRadiusErrorBound =  0.6376f;
    float bridgeInvRadiusErrorBound = 2.1381f;
    float outerInvRadiusErrorBound = 20.4179f;

    if(innerRadius > 2.0f/(2.f * k2Rinv1GeVf)) //as good as no selections!
    {
        innerInvRadiusErrorBound = 12.9173f;
        outerInvRadiusErrorBound = 25.6702f;
        bridgeInvRadiusErrorBound = 5.1700f;
    }

    innerRadiusMin = innerRadius/(1.f + innerInvRadiusErrorBound);
    innerRadiusMax = innerInvRadiusErrorBound < 1 ? innerRadius/(1.f - innerInvRadiusErrorBound) : 123456789.f;

    bridgeRadiusMin = bridgeRadius/(1.f + bridgeInvRadiusErrorBound);
    bridgeRadiusMax = bridgeInvRadiusErrorBound < 1 ? bridgeRadius/(1.f - bridgeInvRadiusErrorBound) : 123456789.f;

    outerRadiusMin = outerRadius/(1.f + outerInvRadiusErrorBound);
    outerRadiusMax = outerInvRadiusErrorBound < 1 ? outerRadius/(1 - outerInvRadiusErrorBound) : 123456789.f;

    return checkIntervalOverlap(1.0f/innerRadiusMax, 1.0f/innerRadiusMin, 1.0f/fmaxf(bridgeRadiusMax, bridgeRadiusMax2S), 1.0f/fminf(bridgeRadiusMin, bridgeRadiusMin2S));

}

__device__ bool SDL::matchRadiiBEEEE(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S, float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax)
{

    float innerInvRadiusErrorBound =  1.9382f;
    float bridgeInvRadiusErrorBound = 3.7280f;
    float outerInvRadiusErrorBound = 5.7030f;


    if(innerRadius > 2.0f/(2.f * k2Rinv1GeVf))
    {
        innerInvRadiusErrorBound = 23.2713f;
        outerInvRadiusErrorBound = 24.0450f;
        bridgeInvRadiusErrorBound = 21.7980f;
    }

    innerRadiusMin = innerRadius/(1.f + innerInvRadiusErrorBound);
    innerRadiusMax = innerInvRadiusErrorBound < 1 ? innerRadius/(1.f - innerInvRadiusErrorBound) : 123456789.f;

    bridgeRadiusMin = bridgeRadius/(1.f + bridgeInvRadiusErrorBound);
    bridgeRadiusMax = bridgeInvRadiusErrorBound < 1 ? bridgeRadius/(1.f - bridgeInvRadiusErrorBound) : 123456789.f;

    outerRadiusMin = outerRadius/(1.f + outerInvRadiusErrorBound);
    outerRadiusMax = outerInvRadiusErrorBound < 1 ? outerRadius/(1.f - outerInvRadiusErrorBound) : 123456789.f;

    return checkIntervalOverlap(1.0f/fmaxf(innerRadiusMax, innerRadiusMax2S), 1.0f/fminf(innerRadiusMin, innerRadiusMin2S), 1.0f/fmaxf(bridgeRadiusMax, bridgeRadiusMax2S), 1.0f/fminf(bridgeRadiusMin, bridgeRadiusMin2S));

}

__device__ bool SDL::matchRadiiEEEEE(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S, float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax)
{
    float innerInvRadiusErrorBound =  1.9382f;
    float bridgeInvRadiusErrorBound = 2.2091f;
    float outerInvRadiusErrorBound = 7.4084f;

    if(innerRadius > 2.0f/(2.f * k2Rinv1GeVf))
    {
        innerInvRadiusErrorBound = 22.5226f;
        bridgeInvRadiusErrorBound = 21.0966f;
        outerInvRadiusErrorBound = 19.1252f;
    }

    innerRadiusMin = innerRadius/(1.f + innerInvRadiusErrorBound);
    innerRadiusMax = innerInvRadiusErrorBound < 1 ? innerRadius/(1.f - innerInvRadiusErrorBound) : 123456789.f;

    bridgeRadiusMin = bridgeRadius/(1.f + bridgeInvRadiusErrorBound);
    bridgeRadiusMax = bridgeInvRadiusErrorBound < 1 ? bridgeRadius/(1.f - bridgeInvRadiusErrorBound) : 123456789.f;

    outerRadiusMin = outerRadius/(1.f + outerInvRadiusErrorBound);
    outerRadiusMax = outerInvRadiusErrorBound < 1 ? outerRadius/(1.f - outerInvRadiusErrorBound) : 123456789.f;

    return checkIntervalOverlap(1.0f/fmaxf(innerRadiusMax, innerRadiusMax2S), 1.0f/fminf(innerRadiusMin, innerRadiusMin2S), 1.0f/fmaxf(bridgeRadiusMax, bridgeRadiusMax2S), 1.0f/fminf(bridgeRadiusMin, bridgeRadiusMin2S));

}

__device__ void SDL::computeErrorInRadius(float* x1Vec, float* y1Vec, float* x2Vec, float* y2Vec, float* x3Vec, float* y3Vec, float& minimumRadius, float& maximumRadius)
{
    //brute force
    float candidateRadius;
    float g, f;
    minimumRadius = 123456789.f;
    maximumRadius = 0.f;
    for(size_t i = 0; i < 3; i++)
    {
        float x1 = x1Vec[i];
	float y1 = y1Vec[i];
        for(size_t j = 0; j < 3; j++)
        {
	    float x2 = x2Vec[j];
	    float y2 = y2Vec[j];
            for(size_t k = 0; k < 3; k++)
            {
	       float x3 = x3Vec[k];
               float y3 = y3Vec[k];
               candidateRadius = computeRadiusFromThreeAnchorHits(x1, y1, x2, y2, x3, y3, g, f);
               maximumRadius = fmaxf(candidateRadius, maximumRadius);
               minimumRadius = fminf(candidateRadius, minimumRadius);
            }
        }
    }
}
__device__ float SDL::computeRadiusFromThreeAnchorHits(float x1, float y1, float x2, float y2, float x3, float y3, float& g, float& f)
{
    float radius = 0.f;

    //writing manual code for computing radius, which obviously sucks
    //TODO:Use fancy inbuilt libraries like cuBLAS or cuSOLVE for this!
    //(g,f) -> center
    //first anchor hit - (x1,y1), second anchor hit - (x2,y2), third anchor hit - (x3, y3)

    /*
    if((y1 - y3) * (x2 - x3) - (x1 - x3) * (y2 - y3) == 0)
    {
        return -1; //WTF man three collinear points!
    }
    */

    float denomInv = 1.0f/((y1 - y3) * (x2 - x3) - (x1 - x3) * (y2 - y3));

    float xy1sqr = x1 * x1 + y1 * y1;

    float xy2sqr = x2 * x2 + y2 * y2;

    float xy3sqr = x3 * x3 + y3 * y3;

    g = 0.5f * ((y3 - y2) * xy1sqr + (y1 - y3) * xy2sqr + (y2 - y1) * xy3sqr) * denomInv;

    f = 0.5f * ((x2 - x3) * xy1sqr + (x3 - x1) * xy2sqr + (x1 - x2) * xy3sqr) * denomInv;

    float c = ((x2 * y3 - x3 * y2) * xy1sqr + (x3 * y1 - x1 * y3) * xy2sqr + (x1 * y2 - x2 * y1) * xy3sqr) * denomInv;

    if(((y1 - y3) * (x2 - x3) - (x1 - x3) * (y2 - y3) == 0) || (g * g + f * f - c < 0))
    {
        printf("three collinear points or FATAL! r^2 < 0!\n");
	radius = -1.f;
    }
    else
      radius = sqrtf(g * g  + f * f - c);

    return radius;
}

__device__ bool SDL::T5HasCommonMiniDoublet(struct SDL::triplets& tripletsInGPU, struct SDL::segments& segmentsInGPU, unsigned int innerTripletIndex, unsigned int outerTripletIndex)
{
    unsigned int innerOuterSegmentIndex = tripletsInGPU.segmentIndices[2 * innerTripletIndex + 1];
    unsigned int outerInnerSegmentIndex = tripletsInGPU.segmentIndices[2 * outerTripletIndex];
    unsigned int innerOuterOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * innerOuterSegmentIndex + 1]; //inner triplet outer segment outer MD index
    unsigned int outerInnerInnerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * outerInnerSegmentIndex]; //outer triplet inner segmnet inner MD index


    return (innerOuterOuterMiniDoubletIndex == outerInnerInnerMiniDoubletIndex);
}

__device__ void SDL::computeSigmasForRegression(SDL::modules& modulesInGPU, const unsigned int* lowerModuleIndices, float* delta1, float* delta2, float* slopes, bool* isFlat, int nPoints, bool anchorHits) 
{
   /*bool anchorHits required to deal with a weird edge case wherein 
     the hits ultimately used in the regression are anchor hits, but the
     lower modules need not all be Pixel Modules (in case of PS). Similarly,
     when we compute the chi squared for the non-anchor hits, the "partner module"
     need not always be a PS strip module, but all non-anchor hits sit on strip 
     modules.
    */
    ModuleType moduleType;
    short moduleSubdet, moduleSide;
    ModuleLayerType moduleLayerType;
    float drdz;
    float inv1 = 0.01f/0.009f;
    float inv2 = 0.15f/0.009f;
    float inv3 = 2.4f/0.009f;
    for(size_t i=0; i<nPoints; i++)
    {
        moduleType = modulesInGPU.moduleType[lowerModuleIndices[i]];
        moduleSubdet = modulesInGPU.subdets[lowerModuleIndices[i]];
        moduleSide = modulesInGPU.sides[lowerModuleIndices[i]];
        moduleLayerType = modulesInGPU.moduleLayerType[lowerModuleIndices[i]];
        //category 1 - barrel PS flat
        if(moduleSubdet == Barrel and moduleType == PS and moduleSide == Center)
        {
            delta1[i] = inv1;//1.1111f;//0.01;
            delta2[i] = inv1;//1.1111f;//0.01;
            slopes[i] = -999.f;
            isFlat[i] = true;
        }

        //category 2 - barrel 2S
        else if(moduleSubdet == Barrel and moduleType == TwoS)
        {
            delta1[i] = 1.f;//0.009;
            delta2[i] = 1.f;//0.009;
            slopes[i] = -999.f;
            isFlat[i] = true;
        }

        //category 3 - barrel PS tilted
        else if(moduleSubdet == Barrel and moduleType == PS and moduleSide != Center)
        {

            //get drdz
            if(moduleLayerType == Strip)
            {
                drdz = modulesInGPU.drdzs[lowerModuleIndices[i]];
                slopes[i] = modulesInGPU.slopes[lowerModuleIndices[i]];
            }
            else
            {
                drdz = modulesInGPU.drdzs[modulesInGPU.partnerModuleIndex(lowerModuleIndices[i])];
                slopes[i] = modulesInGPU.slopes[modulesInGPU.partnerModuleIndex(lowerModuleIndices[i])];
            }

            //delta1[i] = 0.01;
            delta1[i] = inv1;//1.1111f;//0.01;
            isFlat[i] = false;

            if(anchorHits)
            {
                delta2[i] = (inv2 * drdz/sqrtf(1 + drdz * drdz));
            }
            else
            {
                delta2[i] = (inv3 * drdz/sqrtf(1 + drdz * drdz));
            }
        }
        //category 4 - endcap PS
        else if(moduleSubdet == Endcap and moduleType == PS)
        {
            delta1[i] = inv1;//1.1111f;//0.01;
            if(moduleLayerType == Strip)
            {
                slopes[i] = modulesInGPU.slopes[lowerModuleIndices[i]];
            }
            else
            {
                slopes[i] = modulesInGPU.slopes[modulesInGPU.partnerModuleIndex(lowerModuleIndices[i])];

            }
            isFlat[i] = false;

            /*despite the type of the module layer of the lower module index,
            all anchor hits are on the pixel side and all non-anchor hits are
            on the strip side!*/
            if(anchorHits)
            {
                delta2[i] = inv2;//16.6666f;//0.15f;
            }
            else
            {
                delta2[i] = inv3;//266.666f;//2.4f;
            }
        }

        //category 5 - endcap 2S
        else if(moduleSubdet == Endcap and moduleType == TwoS)
        {
            delta1[i] = 1.f;//0.009;
            delta2[i] = 500.f*inv1;//555.5555f;//5.f;
            slopes[i] = modulesInGPU.slopes[lowerModuleIndices[i]];
            isFlat[i] = false;
        }
        else
        {
            printf("ERROR!!!!! I SHOULDN'T BE HERE!!!! subdet = %d, type = %d, side = %d\n", moduleSubdet, moduleType, moduleSide);
        }
    }
}

__device__ float SDL::computeRadiusUsingRegression(int nPoints, float* xs, float* ys, float* delta1, float* delta2, float* slopes, bool* isFlat, float& g, float& f, float* sigmas, float& chiSquared)
{
    float radius = 0.f;

    //some extra variables
    //the two variables will be caled x1 and x2, and y (which is x^2 + y^2)

    float sigmaX1Squared = 0.f;
    float sigmaX2Squared = 0.f;
    float sigmaX1X2 = 0.f; 
    float sigmaX1y = 0.f; 
    float sigmaX2y = 0.f;
    float sigmaY = 0.f;
    float sigmaX1 = 0.f;
    float sigmaX2 = 0.f;
    float sigmaOne = 0.f;

    float xPrime, yPrime, absArctanSlope, angleM;
    for(size_t i = 0; i < nPoints; i++)
    {
        //computing sigmas is a very tricky affair
        //if the module is tilted or endcap, we need to use the slopes properly!

        absArctanSlope = ((slopes[i] != 123456789) ? fabs(atanf(slopes[i])) : 0.5f*float(M_PI)); // Since C++ can't represent infinity, SDL_INF = 123456789 was used to represent infinity in the data table

        if(xs[i] > 0 and ys[i] > 0)
        {
            angleM = 0.5f*float(M_PI) - absArctanSlope;
        }
        else if(xs[i] < 0 and ys[i] > 0)
        {
            angleM = absArctanSlope + 0.5f*float(M_PI);
        }
        else if(xs[i] < 0 and ys[i] < 0)
        {
            angleM = -(absArctanSlope + 0.5f*float(M_PI));
        }
        else if(xs[i] > 0 and ys[i] < 0)
        {
            angleM = -(0.5f*float(M_PI) - absArctanSlope);
        }

        if(not isFlat[i])
        {
            xPrime = xs[i] * cosf(angleM) + ys[i] * sinf(angleM);
            yPrime = ys[i] * cosf(angleM) - xs[i] * sinf(angleM);
        }
        else
        {
            xPrime = xs[i];
            yPrime = ys[i];
        }
        sigmas[i] = 2 * sqrtf((xPrime * delta1[i]) * (xPrime * delta1[i]) + (yPrime * delta2[i]) * (yPrime * delta2[i]));

        sigmaX1Squared += (xs[i] * xs[i])/(sigmas[i] * sigmas[i]);
        sigmaX2Squared += (ys[i] * ys[i])/(sigmas[i] * sigmas[i]);
        sigmaX1X2 += (xs[i] * ys[i])/(sigmas[i] * sigmas[i]);
        sigmaX1y += (xs[i] * (xs[i] * xs[i] + ys[i] * ys[i]))/(sigmas[i] * sigmas[i]);
        sigmaX2y += (ys[i] * (xs[i] * xs[i] + ys[i] * ys[i]))/(sigmas[i] * sigmas[i]);
        sigmaY += (xs[i] * xs[i] + ys[i] * ys[i])/(sigmas[i] * sigmas[i]);
        sigmaX1 += xs[i]/(sigmas[i] * sigmas[i]);
        sigmaX2 += ys[i]/(sigmas[i] * sigmas[i]);
        sigmaOne += 1.0f/(sigmas[i] * sigmas[i]);
    }
    float denominator = (sigmaX1X2 - sigmaX1 * sigmaX2) * (sigmaX1X2 - sigmaX1 * sigmaX2) - (sigmaX1Squared - sigmaX1 * sigmaX1) * (sigmaX2Squared - sigmaX2 * sigmaX2);

    float twoG = ((sigmaX2y - sigmaX2 * sigmaY) * (sigmaX1X2 - sigmaX1 * sigmaX2) - (sigmaX1y - sigmaX1 * sigmaY) * (sigmaX2Squared - sigmaX2 * sigmaX2)) / denominator;
    float twoF = ((sigmaX1y - sigmaX1 * sigmaY) * (sigmaX1X2 - sigmaX1 * sigmaX2) - (sigmaX2y - sigmaX2 * sigmaY) * (sigmaX1Squared - sigmaX1 * sigmaX1)) / denominator;

    float c = -(sigmaY - twoG * sigmaX1 - twoF * sigmaX2)/sigmaOne;
    g = 0.5f*twoG;
    f = 0.5f*twoF;
    if(g * g + f * f - c < 0)
    {
        printf("FATAL! r^2 < 0!\n");
        return -1;
    }
    
    radius = sqrtf(g * g  + f * f - c);
    //compute chi squared
    chiSquared = 0.f;
    for(size_t i = 0; i < nPoints; i++)
    {
       chiSquared +=  (xs[i] * xs[i] + ys[i] * ys[i] - twoG * xs[i] - twoF * ys[i] + c) * (xs[i] * xs[i] + ys[i] * ys[i] - twoG * xs[i] - twoF * ys[i] + c) / (sigmas[i] * sigmas[i]);
    }
    return radius;
}

__device__ float SDL::computeChiSquared(int nPoints, float* xs, float* ys, float* delta1, float* delta2, float* slopes, bool* isFlat, float g, float f, float radius)
{
    // given values of (g, f, radius) and a set of points (and its uncertainties)
    //compute chi squared
    float c = g*g + f*f - radius*radius;
    float chiSquared = 0.f;
    float absArctanSlope, angleM, xPrime, yPrime, sigma;
    for(size_t i = 0; i < nPoints; i++)
    {
        absArctanSlope = ((slopes[i] != 123456789) ? fabs(atanf(slopes[i])) : 0.5f*float(M_PI)); // Since C++ can't represent infinity, SDL_INF = 123456789 was used to represent infinity in the data table
        if(xs[i] > 0 and ys[i] > 0)
        {
            angleM = 0.5f*float(M_PI) - absArctanSlope;
        }
        else if(xs[i] < 0 and ys[i] > 0)
        {
            angleM = absArctanSlope + 0.5f*float(M_PI);
        }
        else if(xs[i] < 0 and ys[i] < 0)
        {
            angleM = -(absArctanSlope + 0.5f*float(M_PI));
        }
        else if(xs[i] > 0 and ys[i] < 0)
        {
            angleM = -(0.5f*float(M_PI) - absArctanSlope);
        }

        if(not isFlat[i])
        {
            xPrime = xs[i] * cosf(angleM) + ys[i] * sinf(angleM);
            yPrime = ys[i] * cosf(angleM) - xs[i] * sinf(angleM);
        }
        else
        {
            xPrime = xs[i];
            yPrime = ys[i];
        }
        sigma = 2 * sqrtf((xPrime * delta1[i]) * (xPrime * delta1[i]) + (yPrime * delta2[i]) * (yPrime * delta2[i]));
        chiSquared +=  (xs[i] * xs[i] + ys[i] * ys[i] - 2 * g * xs[i] - 2 * f * ys[i] + c) * (xs[i] * xs[i] + ys[i] * ys[i] - 2 * g * xs[i] - 2 * f * ys[i] + c) / (sigma * sigma);
    }
    return chiSquared; 
}
