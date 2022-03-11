# include "Module.cuh"
#include "ModuleConnectionMap.h"
#include "allocate.h"
std::map <unsigned int, unsigned int> *SDL::detIdToIndex;

void SDL::createRangesInUnifiedMemory(struct objectRanges& rangesInGPU,unsigned int nModules,cudaStream_t stream, unsigned int nLowerModules)
{
    /* modules stucture object will be created in Event.cu*/
#ifdef CACHE_ALLOC
    rangesInGPU.hitRanges =                 (int*)cms::cuda::allocate_managed(nModules * 2 * sizeof(int),stream);
    rangesInGPU.mdRanges =                  (int*)cms::cuda::allocate_managed(nModules * 2 * sizeof(int),stream);
    rangesInGPU.segmentRanges =             (int*)cms::cuda::allocate_managed(nModules * 2 * sizeof(int),stream);
    rangesInGPU.trackletRanges =            (int*)cms::cuda::allocate_managed(nModules * 2 * sizeof(int),stream);
    rangesInGPU.tripletRanges =             (int*)cms::cuda::allocate_managed(nModules * 2 * sizeof(int),stream);
    rangesInGPU.trackCandidateRanges =      (int*)cms::cuda::allocate_managed(nModules * 2 * sizeof(int),stream);
    rangesInGPU.quintupletRanges =          (int*)cms::cuda::allocate_managed(nModules * 2 * sizeof(int),stream);
    rangesInGPU.nEligibleT5Modules = (unsigned int*)cms::cuda::allocate_managed(sizeof(unsigned int),stream);
    rangesInGPU.quintupletModuleIndices = (int*)cms::cuda::allocate_managed(nLowerModules * sizeof(int),stream);
#else
    cudaMallocManaged(&rangesInGPU.hitRanges,nModules * 2 * sizeof(int));
    cudaMallocManaged(&rangesInGPU.mdRanges,nModules * 2 * sizeof(int));
    cudaMallocManaged(&rangesInGPU.segmentRanges,nModules * 2 * sizeof(int));
    cudaMallocManaged(&rangesInGPU.trackletRanges,nModules * 2 * sizeof(int));
    cudaMallocManaged(&rangesInGPU.tripletRanges,nModules * 2 * sizeof(int));
    cudaMallocManaged(&rangesInGPU.trackCandidateRanges, nModules * 2 * sizeof(int));
    cudaMallocManaged(&rangesInGPU.quintupletRanges, nModules * 2 * sizeof(int));
    cudaMallocManaged(&rangesInGPU.nEligibleT5Modules, sizeof(unsigned int));
    cudaMallocManaged(&rangesInGPU.quintupletModuleIndices, nLowerModules * sizeof(int));
#endif
}
void SDL::createRangesInExplicitMemory(struct objectRanges& rangesInGPU,unsigned int nModules,cudaStream_t stream, unsigned int nLowerModules)
{
    /* modules stucture object will be created in Event.cu*/
#ifdef CACHE_ALLOC
    int dev;
    cudaGetDevice(&dev);
    rangesInGPU.hitRanges =                  (int*)cms::cuda::allocate_device(dev,nModules * 2 * sizeof(int),stream);
    rangesInGPU.mdRanges =                   (int*)cms::cuda::allocate_device(dev,nModules * 2 * sizeof(int),stream);
    rangesInGPU.segmentRanges =              (int*)cms::cuda::allocate_device(dev,nModules * 2 * sizeof(int),stream);
    rangesInGPU.trackletRanges =             (int*)cms::cuda::allocate_device(dev,nModules * 2 * sizeof(int),stream);
    rangesInGPU.tripletRanges =              (int*)cms::cuda::allocate_device(dev,nModules * 2 * sizeof(int),stream);
    rangesInGPU.trackCandidateRanges =       (int*)cms::cuda::allocate_device(dev,nModules * 2 * sizeof(int),stream);
    rangesInGPU.quintupletRanges =       (int*)cms::cuda::allocate_device(dev,nModules * 2 * sizeof(int),stream);
    rangesInGPU.nEligibleT5Modules = (unsigned int*)cms::cuda::allocate_device(dev,sizeof(unsigned int),stream);
    rangesInGPU.quintupletModuleIndices = (int*)cms::cuda::allocate_device(dev,nLowerModules * sizeof(int),stream);
#else
    cudaMalloc(&rangesInGPU.hitRanges,nModules * 2 * sizeof(int));
    cudaMalloc(&rangesInGPU.mdRanges,nModules * 2 * sizeof(int));
    cudaMalloc(&rangesInGPU.segmentRanges,nModules * 2 * sizeof(int));
    cudaMalloc(&rangesInGPU.trackletRanges,nModules * 2 * sizeof(int));
    cudaMalloc(&rangesInGPU.tripletRanges,nModules * 2 * sizeof(int));
    cudaMalloc(&rangesInGPU.trackCandidateRanges, nModules * 2 * sizeof(int));
    cudaMalloc(&rangesInGPU.quintupletRanges, nModules * 2 * sizeof(int));
    cudaMalloc(&rangesInGPU.nEligibleT5Modules, sizeof(unsigned int));
    cudaMalloc(&rangesInGPU.quintupletModuleIndices, nLowerModules * sizeof(int));
#endif
}
void SDL::createModulesInUnifiedMemory(struct modules& modulesInGPU,unsigned int nModules,cudaStream_t stream)
{
    cudaMallocManaged(&modulesInGPU.detIds,nModules * sizeof(unsigned int));
    cudaMallocManaged(&modulesInGPU.moduleMap,nModules * 40 * sizeof(unsigned int));
    cudaMallocManaged(&modulesInGPU.nConnectedModules,nModules * sizeof(unsigned int));
    cudaMallocManaged(&modulesInGPU.drdzs,nModules * sizeof(float));
    cudaMallocManaged(&modulesInGPU.slopes,nModules * sizeof(float));
    cudaMallocManaged(&modulesInGPU.nModules,sizeof(unsigned int));
    cudaMallocManaged(&modulesInGPU.nLowerModules,sizeof(unsigned int));
    cudaMallocManaged(&modulesInGPU.layers,nModules * sizeof(short));
    cudaMallocManaged(&modulesInGPU.rings,nModules * sizeof(short));
    cudaMallocManaged(&modulesInGPU.modules,nModules * sizeof(short));
    cudaMallocManaged(&modulesInGPU.rods,nModules * sizeof(short));
    cudaMallocManaged(&modulesInGPU.subdets,nModules * sizeof(short));
    cudaMallocManaged(&modulesInGPU.sides,nModules * sizeof(short));
    cudaMallocManaged(&modulesInGPU.isInverted, nModules * sizeof(bool));
    cudaMallocManaged(&modulesInGPU.isLower, nModules * sizeof(bool));
    cudaMallocManaged(&modulesInGPU.isAnchor, nModules * sizeof(bool));

    cudaMallocManaged(&modulesInGPU.moduleType,nModules * sizeof(ModuleType));
    cudaMallocManaged(&modulesInGPU.moduleLayerType,nModules * sizeof(ModuleLayerType));

    *modulesInGPU.nModules = nModules;
}
void SDL::createModulesInExplicitMemory(struct modules& modulesInGPU,unsigned int nModules,cudaStream_t stream)
{
    /* modules stucture object will be created in Event.cu*/
    cudaMalloc(&(modulesInGPU.detIds),nModules * sizeof(unsigned int));
    cudaMalloc(&modulesInGPU.moduleMap,nModules * 40 * sizeof(unsigned int));
    cudaMalloc(&modulesInGPU.nConnectedModules,nModules * sizeof(unsigned int));
    cudaMalloc(&modulesInGPU.drdzs,nModules * sizeof(float));
    cudaMalloc(&modulesInGPU.slopes,nModules * sizeof(float));
    cudaMalloc(&modulesInGPU.nModules,sizeof(unsigned int));
    cudaMalloc(&modulesInGPU.nLowerModules,sizeof(unsigned int));
    cudaMalloc(&modulesInGPU.layers,nModules * sizeof(short));
    cudaMalloc(&modulesInGPU.rings,nModules * sizeof(short));
    cudaMalloc(&modulesInGPU.modules,nModules * sizeof(short));
    cudaMalloc(&modulesInGPU.rods,nModules * sizeof(short));
    cudaMalloc(&modulesInGPU.subdets,nModules * sizeof(short));
    cudaMalloc(&modulesInGPU.sides,nModules * sizeof(short));
    cudaMalloc(&modulesInGPU.isInverted, nModules * sizeof(bool));
    cudaMalloc(&modulesInGPU.isLower, nModules * sizeof(bool));
    cudaMalloc(&modulesInGPU.isAnchor, nModules * sizeof(bool));
    cudaMalloc(&modulesInGPU.moduleType,nModules * sizeof(ModuleType));
    cudaMalloc(&modulesInGPU.moduleLayerType,nModules * sizeof(ModuleLayerType));

    cudaMemcpyAsync(modulesInGPU.nModules,&nModules,sizeof(unsigned int),cudaMemcpyHostToDevice,stream);
    cudaStreamSynchronize(stream);
}

void SDL::objectRanges::freeMemoryCache()//struct objectRanges& rangesInGPU)
{
#ifdef Explicit_Module
  int dev;
  cudaGetDevice(&dev);
  cms::cuda::free_device(dev,hitRanges);
  cms::cuda::free_device(dev,mdRanges);
  cms::cuda::free_device(dev,segmentRanges);
  cms::cuda::free_device(dev,trackletRanges);
  cms::cuda::free_device(dev,tripletRanges);
  cms::cuda::free_device(dev,trackCandidateRanges);
  cms::cuda::free_device(dev,quintupletRanges);
  cms::cuda::free_device(dev,nEligibleT5Modules);
  cms::cuda::free_device(dev,quintupletModuleIndices);
#else

  cms::cuda::free_managed(hitRanges);
  cms::cuda::free_managed(mdRanges);
  cms::cuda::free_managed(segmentRanges);
  cms::cuda::free_managed(trackletRanges);
  cms::cuda::free_managed(tripletRanges);
  cms::cuda::free_managed(trackCandidateRanges);
  cms::cuda::free_managed(quintupletRanges);
  cms::cuda::free_managed(nEligibleT5Modules);
  cms::cuda::free_managed(quintupletModuleIndices);
#endif
}
void SDL::objectRanges::freeMemory()//struct objectRanges& rangesInGPU)
{
  cudaFree(hitRanges);
  cudaFree(mdRanges);
  cudaFree(segmentRanges);
  cudaFree(trackletRanges);
  cudaFree(tripletRanges);
  cudaFree(trackCandidateRanges);
  cudaFree(quintupletRanges);
  cudaFree(nEligibleT5Modules);
  cudaFree(quintupletModuleIndices);
}
void SDL::freeModulesCache(struct modules& modulesInGPU,struct pixelMap& pixelMapping)
{
#ifdef Explicit_Module
  int dev;
  cudaGetDevice(&dev);
  cms::cuda::free_device(dev,modulesInGPU.detIds);
  cms::cuda::free_device(dev,modulesInGPU.moduleMap);
  cms::cuda::free_device(dev,modulesInGPU.nConnectedModules);
  cms::cuda::free_device(dev,modulesInGPU.drdzs);
  cms::cuda::free_device(dev,modulesInGPU.slopes);
  cms::cuda::free_device(dev,modulesInGPU.nModules);
  cms::cuda::free_device(dev,modulesInGPU.nLowerModules);
  cms::cuda::free_device(dev,modulesInGPU.layers);
  cms::cuda::free_device(dev,modulesInGPU.rings);
  cms::cuda::free_device(dev,modulesInGPU.modules);
  cms::cuda::free_device(dev,modulesInGPU.rods);
  cms::cuda::free_device(dev,modulesInGPU.subdets);
  cms::cuda::free_device(dev,modulesInGPU.sides);
  cms::cuda::free_device(dev,modulesInGPU.isInverted);
  cms::cuda::free_device(dev,modulesInGPU.isLower);
  cms::cuda::free_device(dev,modulesInGPU.isAnchor);
  cms::cuda::free_device(dev,modulesInGPU.moduleType);
  cms::cuda::free_device(dev,modulesInGPU.moduleLayerType);
  cms::cuda::free_device(dev,modulesInGPU.lowerModuleIndices);
  cms::cuda::free_device(dev,modulesInGPU.reverseLookupLowerModuleIndices);
  cms::cuda::free_device(dev,modulesInGPU.connectedPixels);
#else
  cms::cuda::free_managed(modulesInGPU.detIds);
  cms::cuda::free_managed(modulesInGPU.moduleMap);
  cms::cuda::free_managed(modulesInGPU.nConnectedModules);
  cms::cuda::free_managed(modulesInGPU.drdzs);
  cms::cuda::free_managed(modulesInGPU.slopes);
  cms::cuda::free_managed(modulesInGPU.nModules);
  cms::cuda::free_managed(modulesInGPU.nLowerModules);
  cms::cuda::free_managed(modulesInGPU.layers);
  cms::cuda::free_managed(modulesInGPU.rings);
  cms::cuda::free_managed(modulesInGPU.modules);
  cms::cuda::free_managed(modulesInGPU.rods);
  cms::cuda::free_managed(modulesInGPU.subdets);
  cms::cuda::free_managed(modulesInGPU.sides);
  cms::cuda::free_managed(modulesInGPU.isInverted);
  cms::cuda::free_managed(modulesInGPU.isLower);
  cms::cuda::free_managed(modulesInGPU.isAnchor);
  cms::cuda::free_managed(modulesInGPU.moduleType);
  cms::cuda::free_managed(modulesInGPU.moduleLayerType);
  cms::cuda::free_managed(modulesInGPU.lowerModuleIndices);
  cms::cuda::free_managed(modulesInGPU.reverseLookupLowerModuleIndices);
  cms::cuda::free_managed(modulesInGPU.connectedPixels);
#endif
  cudaFreeHost(pixelMapping.connectedPixelsSizes);
  cudaFreeHost(pixelMapping.connectedPixelsSizesPos);
  cudaFreeHost(pixelMapping.connectedPixelsSizesNeg);
  cudaFreeHost(pixelMapping.connectedPixelsIndex);
  cudaFreeHost(pixelMapping.connectedPixelsIndexPos);
  cudaFreeHost(pixelMapping.connectedPixelsIndexNeg);
}
void SDL::freeModules(struct modules& modulesInGPU, struct pixelMap& pixelMapping,cudaStream_t stream)
{

  cudaFree(modulesInGPU.detIds);
  cudaFree(modulesInGPU.moduleMap);
  cudaFree(modulesInGPU.nConnectedModules);
  cudaFree(modulesInGPU.drdzs);
  cudaFree(modulesInGPU.slopes);
  cudaFree(modulesInGPU.nModules);
  cudaFree(modulesInGPU.nLowerModules);
  cudaFree(modulesInGPU.layers);
  cudaFree(modulesInGPU.rings);
  cudaFree(modulesInGPU.modules);
  cudaFree(modulesInGPU.rods);
  cudaFree(modulesInGPU.subdets);
  cudaFree(modulesInGPU.sides);
  cudaFree(modulesInGPU.isInverted);
  cudaFree(modulesInGPU.isLower);
  cudaFree(modulesInGPU.isAnchor);
  cudaFree(modulesInGPU.moduleType);
  cudaFree(modulesInGPU.moduleLayerType);
  cudaFree(modulesInGPU.lowerModuleIndices);
  cudaFree(modulesInGPU.reverseLookupLowerModuleIndices);
  cudaFree(modulesInGPU.connectedPixels);

  cudaFreeHost(pixelMapping.connectedPixelsSizes);
  cudaFreeHost(pixelMapping.connectedPixelsSizesPos);
  cudaFreeHost(pixelMapping.connectedPixelsSizesNeg);
  cudaFreeHost(pixelMapping.connectedPixelsIndex);
  cudaFreeHost(pixelMapping.connectedPixelsIndexPos);
  cudaFreeHost(pixelMapping.connectedPixelsIndexNeg);
}

void SDL::createLowerModuleIndexMapExplicit(struct modules& modulesInGPU, unsigned int nLowerModules, unsigned int nModules,bool* isLower, cudaStream_t stream)
{
    //FIXME:some hacks to get the pixel module in the lower modules index without incrementing nLowerModules counter!
    //Reproduce these hacks in the explicit memory for identical results (or come up with a better method)
    unsigned int* lowerModuleIndices;
    int* reverseLookupLowerModuleIndices;
    cudaMallocHost(&lowerModuleIndices,(nLowerModules + 1) * sizeof(unsigned int));
    cudaMallocHost(&reverseLookupLowerModuleIndices,nModules * sizeof(int));

    unsigned int lowerModuleCounter = 0;
    for(auto it = (*detIdToIndex).begin(); it != (*detIdToIndex).end(); it++)
    {
        unsigned int index = it->second;
        unsigned int detId = it->first;
        if(isLower[index])
        {
            lowerModuleIndices[lowerModuleCounter] = index;
            reverseLookupLowerModuleIndices[index] = lowerModuleCounter;
            lowerModuleCounter++;
        }
        else
        {
           reverseLookupLowerModuleIndices[index] = -1;
        }
    }
    //hacky stuff "beyond the index" for the pixel module. nLowerModules will *NOT* cover the pixel module!
    lowerModuleIndices[nLowerModules] = (*detIdToIndex)[1];
    reverseLookupLowerModuleIndices[(*detIdToIndex)[1]] = nLowerModules;
    cudaMalloc(&modulesInGPU.lowerModuleIndices,(nLowerModules + 1) * sizeof(unsigned int));
    cudaMalloc(&modulesInGPU.reverseLookupLowerModuleIndices,nModules * sizeof(int));

    cudaMemcpyAsync(modulesInGPU.lowerModuleIndices,lowerModuleIndices,sizeof(unsigned int)*(nLowerModules+1),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(modulesInGPU.reverseLookupLowerModuleIndices,reverseLookupLowerModuleIndices,sizeof(int)*nModules,cudaMemcpyHostToDevice,stream);
    cudaStreamSynchronize(stream);

    cudaFreeHost(lowerModuleIndices);
    cudaFreeHost(reverseLookupLowerModuleIndices);
}
void SDL::createLowerModuleIndexMap(struct modules& modulesInGPU, unsigned int nLowerModules, unsigned int nModules,cudaStream_t stream)
{
    //FIXME:some hacks to get the pixel module in the lower modules index without incrementing nLowerModules counter!
    //Reproduce these hacks in the explicit memory for identical results (or come up with a better method)
    cudaMallocManaged(&modulesInGPU.lowerModuleIndices,(nLowerModules + 1) * sizeof(unsigned int));
    cudaMallocManaged(&modulesInGPU.reverseLookupLowerModuleIndices,nModules * sizeof(int));

    unsigned int lowerModuleCounter = 0;
    for(auto it = (*detIdToIndex).begin(); it != (*detIdToIndex).end(); it++)
    {
        unsigned int index = it->second;
        unsigned int detId = it->first;
        if(modulesInGPU.isLower[index])
        {
            modulesInGPU.lowerModuleIndices[lowerModuleCounter] = index;
            modulesInGPU.reverseLookupLowerModuleIndices[index] = lowerModuleCounter;
            lowerModuleCounter++;
        }
        else
        {
            modulesInGPU.reverseLookupLowerModuleIndices[index] = -1;
        }
    }
    //hacky stuff "beyond the index" for the pixel module. nLowerModules will *NOT* cover the pixel module!
    modulesInGPU.lowerModuleIndices[nLowerModules] = (*detIdToIndex)[1];
    modulesInGPU.reverseLookupLowerModuleIndices[(*detIdToIndex)[1]] = nLowerModules;
}

void SDL::loadModulesFromFile(struct modules& modulesInGPU, unsigned int& nModules,unsigned int& lowerModuleCounter, struct pixelMap& pixelMapping,cudaStream_t stream, const char* moduleMetaDataFilePath)
{
    detIdToIndex = new std::map<unsigned int, unsigned int>;

    /*modules structure object will be created in Event.cu*/
    /* Load the whole text file into the unordered_map first*/

    std::ifstream ifile;
    ifile.open(moduleMetaDataFilePath);
    if(!ifile.is_open())
    {
        std::cout<<"ERROR! module list file not present!"<<std::endl;
    }
    std::string line;
    unsigned int counter = 0;

    while(std::getline(ifile,line))
    {
        std::stringstream ss(line);
        std::string token;
        bool flag = 0;

        while(std::getline(ss,token,','))
        {
            if(flag == 1) break;
            (*detIdToIndex)[stoi(token)] = counter;
            flag = 1;
            counter++;
        }
    }
    (*detIdToIndex)[1] = counter; //pixel module is the last module in the module list
    counter++;
    nModules = counter;
    std::cout<<"Number of modules = "<<nModules<<std::endl;
#ifdef Explicit_Module
    createModulesInExplicitMemory(modulesInGPU,nModules,stream);
    unsigned int* host_detIds;
    short* host_layers;
    short* host_rings;
    short* host_rods;
    short* host_modules;
    short* host_subdets;
    short* host_sides;
    bool* host_isInverted;
    bool* host_isLower;
    bool* host_isAnchor;
    ModuleType* host_moduleType;
    ModuleLayerType* host_moduleLayerType;
    float* host_slopes;
    float* host_drdzs;
    cudaMallocHost(&host_detIds,sizeof(unsigned int)*nModules);
    cudaMallocHost(&host_layers,sizeof(short)*nModules);
    cudaMallocHost(&host_rings,sizeof(short)*nModules);
    cudaMallocHost(&host_rods,sizeof(short)*nModules);
    cudaMallocHost(&host_modules,sizeof(short)*nModules);
    cudaMallocHost(&host_subdets,sizeof(short)*nModules);
    cudaMallocHost(&host_sides,sizeof(short)*nModules);
    cudaMallocHost(&host_isInverted,sizeof(bool)*nModules);
    cudaMallocHost(&host_isLower,sizeof(bool)*nModules);
    cudaMallocHost(&host_isAnchor, sizeof(bool) * nModules);
    cudaMallocHost(&host_moduleType,sizeof(ModuleType)*nModules);
    cudaMallocHost(&host_moduleLayerType,sizeof(ModuleLayerType)*nModules);
    cudaMallocHost(&host_slopes,sizeof(float)*nModules);
    cudaMallocHost(&host_drdzs,sizeof(float)*nModules);
    for(auto it = (*detIdToIndex).begin(); it != (*detIdToIndex).end(); it++)
    {
        unsigned int detId = it->first;
        unsigned int index = it->second;
        printf("detId: %d -> index: %d\n", detId, index);
        host_detIds[index] = detId;
        if(detId == 1)
        {
            host_layers[index] = 0;
            host_rings[index] = 0;
            host_rods[index] = 0;
            host_modules[index] = 0;
            host_subdets[index] = SDL::InnerPixel;
            host_sides[index] = 0;
            host_isInverted[index] = 0;
            host_isLower[index] = false;
            host_isAnchor[index] = false;
            host_moduleType[index] = PixelModule;
            host_moduleLayerType[index] = SDL::InnerPixelLayer;
            host_slopes[index] = 0;
            host_drdzs[index] = 0;
        }
        else
        {
            unsigned short layer,ring,rod,module,subdet,side;
            setDerivedQuantities(detId,layer,ring,rod,module,subdet,side);
            host_layers[index] = layer;
            host_rings[index] = ring;
            host_rods[index] = rod;
            host_modules[index] = module;
            host_subdets[index] = subdet;
            host_sides[index] = side;

            host_isInverted[index] = modulesInGPU.parseIsInverted(index,subdet, side,module,layer);
            host_isLower[index] = modulesInGPU.parseIsLower(index, host_isInverted[index], detId);

            host_moduleType[index] = modulesInGPU.parseModuleType(index, subdet, layer, ring);
            host_moduleLayerType[index] = modulesInGPU.parseModuleLayerType(index, host_moduleType[index],host_isInverted[index],host_isLower[index]);

            if(host_moduleType[index] == SDL::PS and host_moduleLayerType[index] == SDL::Pixel)
            {
                host_isAnchor[index] = true;
            }
            else if(host_moduleType[index] == SDL::TwoS and host_isLower[index])
            {
                host_isAnchor[index] = true;   
            }
            else
            {
                host_isAnchor[index] = false;
            }

            host_slopes[index] = (subdet == Endcap) ? endcapGeometry.getSlopeLower(detId) : tiltedGeometry.getSlope(detId);
            host_drdzs[index] = (subdet == Barrel) ? tiltedGeometry.getDrDz(detId) : 0;
        }
          //lowerModuleCounter[0] += host_isLower[index];
          lowerModuleCounter += host_isLower[index];
    }

    cudaMemcpyAsync(modulesInGPU.nLowerModules,&lowerModuleCounter,sizeof(unsigned int),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(modulesInGPU.detIds,host_detIds,nModules*sizeof(unsigned int),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(modulesInGPU.layers,host_layers,nModules*sizeof(short),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(modulesInGPU.rings,host_rings,sizeof(short)*nModules,cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(modulesInGPU.rods,host_rods,sizeof(short)*nModules,cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(modulesInGPU.modules,host_modules,sizeof(short)*nModules,cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(modulesInGPU.subdets,host_subdets,sizeof(short)*nModules,cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(modulesInGPU.sides,host_sides,sizeof(short)*nModules,cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(modulesInGPU.isInverted,host_isInverted,sizeof(bool)*nModules,cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(modulesInGPU.isLower,host_isLower,sizeof(bool)*nModules,cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(modulesInGPU.moduleType,host_moduleType,sizeof(ModuleType)*nModules,cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(modulesInGPU.moduleLayerType,host_moduleLayerType,sizeof(ModuleLayerType)*nModules,cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(modulesInGPU.slopes,host_slopes,sizeof(float)*nModules,cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(modulesInGPU.isAnchor, host_isAnchor, sizeof(bool) * nModules, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(modulesInGPU.drdzs,host_drdzs,sizeof(float)*nModules,cudaMemcpyHostToDevice,stream);
    cudaStreamSynchronize(stream);
    cudaFreeHost(host_detIds);
    cudaFreeHost(host_layers);
    cudaFreeHost(host_rings);
    cudaFreeHost(host_rods);
    cudaFreeHost(host_modules);
    cudaFreeHost(host_subdets);
    cudaFreeHost(host_sides);
    cudaFreeHost(host_isInverted);
    cudaFreeHost(host_isLower);
    cudaFreeHost(host_isAnchor);
    cudaFreeHost(host_moduleType);
    cudaFreeHost(host_moduleLayerType);
    cudaFreeHost(host_slopes);
    cudaFreeHost(host_drdzs);
    std::cout<<"number of lower modules (without fake pixel module)= "<<lowerModuleCounter<<std::endl;
    createLowerModuleIndexMapExplicit(modulesInGPU,lowerModuleCounter, nModules,host_isLower,stream);
    fillConnectedModuleArrayExplicit(modulesInGPU,nModules,stream);
    fillPixelMap(modulesInGPU,pixelMapping,stream);

#else
    createModulesInUnifiedMemory(modulesInGPU,nModules,stream);
    lowerModuleCounter = 0;
    for(auto it = (*detIdToIndex).begin(); it != (*detIdToIndex).end(); it++)
    {
        unsigned int detId = it->first;
        unsigned int index = it->second;
        modulesInGPU.detIds[index] = detId;
        if(detId == 1)
        {
            modulesInGPU.layers[index] = 0;
            modulesInGPU.rings[index] = 0;
            modulesInGPU.rods[index] = 0;
            modulesInGPU.modules[index] = 0;
            modulesInGPU.subdets[index] = SDL::InnerPixel;
            modulesInGPU.sides[index] = 0;
            modulesInGPU.isInverted[index] = 0;
            modulesInGPU.isLower[index] = false;
            modulesInGPU.isAnchor[index] = false;
            modulesInGPU.moduleType[index] = PixelModule;
            modulesInGPU.moduleLayerType[index] = SDL::InnerPixelLayer;
            modulesInGPU.slopes[index] = 0;
            modulesInGPU.drdzs[index] = 0;
        }
        else
        {
            unsigned short layer,ring,rod,module,subdet,side;
            setDerivedQuantities(detId,layer,ring,rod,module,subdet,side);
            modulesInGPU.layers[index] = layer;
            modulesInGPU.rings[index] = ring;
            modulesInGPU.rods[index] = rod;
            modulesInGPU.modules[index] = module;
            modulesInGPU.subdets[index] = subdet;
            modulesInGPU.sides[index] = side;

            modulesInGPU.isInverted[index] = modulesInGPU.parseIsInverted(index);
            modulesInGPU.isLower[index] = modulesInGPU.parseIsLower(index);

            modulesInGPU.moduleType[index] = modulesInGPU.parseModuleType(index);
            modulesInGPU.moduleLayerType[index] = modulesInGPU.parseModuleLayerType(index);

            if(modulesInGPU.moduleType[index] == SDL::PS and modulesInGPU.moduleLayerType[index] == SDL::Pixel)
            {
                modulesInGPU.isAnchor[index] = true;
            }
            else if(modulesInGPU.moduleType[index] == SDL::TwoS and modulesInGPU.isLower[index])
            {
                modulesInGPU.isAnchor[index] = true;
            }
            else
            {
                modulesInGPU.isAnchor[index] = false;
            }

            modulesInGPU.slopes[index] = (subdet == Endcap) ? endcapGeometry.getSlopeLower(detId) : tiltedGeometry.getSlope(detId);
            modulesInGPU.drdzs[index] = (subdet == Barrel) ? tiltedGeometry.getDrDz(detId) : 0;
        }
        if(modulesInGPU.isLower[index]) lowerModuleCounter++;
    }
    *modulesInGPU.nLowerModules = lowerModuleCounter;
    std::cout<<"number of lower modules (without fake pixel module)= "<<*modulesInGPU.nLowerModules<<std::endl;
    createLowerModuleIndexMap(modulesInGPU,lowerModuleCounter, nModules,stream);
    fillConnectedModuleArray(modulesInGPU,nModules);
    fillPixelMap(modulesInGPU,pixelMapping,stream);
    #endif
}

void SDL::fillPixelMap(struct modules& modulesInGPU, struct pixelMap& pixelMapping,cudaStream_t stream)
{
    int size_superbins = 45000;//SDL::moduleConnectionMap_pLStoLayer1Subdet5.size(); //changed to 45000 to reduce memory useage on GPU
    std::vector<unsigned int> connectedModuleDetIds;
    std::vector<unsigned int> connectedModuleDetIds_pos;
    std::vector<unsigned int> connectedModuleDetIds_neg;
    unsigned int* connectedPixelsIndex;
    unsigned int* connectedPixelsIndexPos;
    unsigned int* connectedPixelsIndexNeg;
    unsigned int* connectedPixelsSizes;
    unsigned int* connectedPixelsSizesPos;
    unsigned int* connectedPixelsSizesNeg;
    cudaMallocHost(&pixelMapping.connectedPixelsIndex,size_superbins * sizeof(unsigned int));
    cudaMallocHost(&pixelMapping.connectedPixelsSizes,size_superbins * sizeof(unsigned int));
    cudaMallocHost(&pixelMapping.connectedPixelsIndexPos,size_superbins * sizeof(unsigned int));
    cudaMallocHost(&pixelMapping.connectedPixelsSizesPos,size_superbins * sizeof(unsigned int));
    cudaMallocHost(&pixelMapping.connectedPixelsIndexNeg,size_superbins * sizeof(unsigned int));
    cudaMallocHost(&pixelMapping.connectedPixelsSizesNeg,size_superbins * sizeof(unsigned int));
    int totalSizes=0;
    int totalSizes_pos=0;
    int totalSizes_neg=0;
    for(int isuperbin =0; isuperbin<size_superbins; isuperbin++)
    {
      std::vector<unsigned int> connectedModuleDetIds_pLStoLayer1Subdet5 = SDL::moduleConnectionMap_pLStoLayer1Subdet5.getConnectedModuleDetIds(isuperbin+size_superbins);// index adjustment to get high values
      std::vector<unsigned int> connectedModuleDetIds_pLStoLayer2Subdet5 = SDL::moduleConnectionMap_pLStoLayer2Subdet5.getConnectedModuleDetIds(isuperbin+size_superbins);// from the high pt bins
      std::vector<unsigned int> connectedModuleDetIds_pLStoLayer3Subdet5 = SDL::moduleConnectionMap_pLStoLayer3Subdet5.getConnectedModuleDetIds(isuperbin+size_superbins);
      std::vector<unsigned int> connectedModuleDetIds_pLStoLayer1Subdet4 = SDL::moduleConnectionMap_pLStoLayer1Subdet4.getConnectedModuleDetIds(isuperbin+size_superbins);
      std::vector<unsigned int> connectedModuleDetIds_pLStoLayer2Subdet4 = SDL::moduleConnectionMap_pLStoLayer2Subdet4.getConnectedModuleDetIds(isuperbin+size_superbins);
      std::vector<unsigned int> connectedModuleDetIds_pLStoLayer3Subdet4 = SDL::moduleConnectionMap_pLStoLayer3Subdet4.getConnectedModuleDetIds(isuperbin+size_superbins);
      std::vector<unsigned int> connectedModuleDetIds_pLStoLayer4Subdet4 = SDL::moduleConnectionMap_pLStoLayer4Subdet4.getConnectedModuleDetIds(isuperbin+size_superbins);
      connectedModuleDetIds.insert(connectedModuleDetIds.end(),connectedModuleDetIds_pLStoLayer1Subdet5.begin(),connectedModuleDetIds_pLStoLayer1Subdet5.end());
      connectedModuleDetIds.insert(connectedModuleDetIds.end(),connectedModuleDetIds_pLStoLayer2Subdet5.begin(),connectedModuleDetIds_pLStoLayer2Subdet5.end());
      connectedModuleDetIds.insert(connectedModuleDetIds.end(),connectedModuleDetIds_pLStoLayer3Subdet5.begin(),connectedModuleDetIds_pLStoLayer3Subdet5.end());
      connectedModuleDetIds.insert(connectedModuleDetIds.end(),connectedModuleDetIds_pLStoLayer1Subdet4.begin(),connectedModuleDetIds_pLStoLayer1Subdet4.end());
      connectedModuleDetIds.insert(connectedModuleDetIds.end(),connectedModuleDetIds_pLStoLayer2Subdet4.begin(),connectedModuleDetIds_pLStoLayer2Subdet4.end());
      connectedModuleDetIds.insert(connectedModuleDetIds.end(),connectedModuleDetIds_pLStoLayer3Subdet4.begin(),connectedModuleDetIds_pLStoLayer3Subdet4.end());
      connectedModuleDetIds.insert(connectedModuleDetIds.end(),connectedModuleDetIds_pLStoLayer4Subdet4.begin(),connectedModuleDetIds_pLStoLayer4Subdet4.end());

      int sizes =0;
      sizes += connectedModuleDetIds_pLStoLayer1Subdet5.size();
      sizes += connectedModuleDetIds_pLStoLayer2Subdet5.size();
      sizes += connectedModuleDetIds_pLStoLayer3Subdet5.size();
      sizes += connectedModuleDetIds_pLStoLayer1Subdet4.size();
      sizes += connectedModuleDetIds_pLStoLayer2Subdet4.size();
      sizes += connectedModuleDetIds_pLStoLayer3Subdet4.size();
      sizes += connectedModuleDetIds_pLStoLayer4Subdet4.size();
      pixelMapping.connectedPixelsIndex[isuperbin] = totalSizes;
      pixelMapping.connectedPixelsSizes[isuperbin] = sizes;
      totalSizes += sizes;


      std::vector<unsigned int> connectedModuleDetIds_pLStoLayer1Subdet5_pos = SDL::moduleConnectionMap_pLStoLayer1Subdet5_pos.getConnectedModuleDetIds(isuperbin);
      std::vector<unsigned int> connectedModuleDetIds_pLStoLayer2Subdet5_pos = SDL::moduleConnectionMap_pLStoLayer2Subdet5_pos.getConnectedModuleDetIds(isuperbin);
      std::vector<unsigned int> connectedModuleDetIds_pLStoLayer3Subdet5_pos = SDL::moduleConnectionMap_pLStoLayer3Subdet5_pos.getConnectedModuleDetIds(isuperbin);
      std::vector<unsigned int> connectedModuleDetIds_pLStoLayer1Subdet4_pos = SDL::moduleConnectionMap_pLStoLayer1Subdet4_pos.getConnectedModuleDetIds(isuperbin);
      std::vector<unsigned int> connectedModuleDetIds_pLStoLayer2Subdet4_pos = SDL::moduleConnectionMap_pLStoLayer2Subdet4_pos.getConnectedModuleDetIds(isuperbin);
      std::vector<unsigned int> connectedModuleDetIds_pLStoLayer3Subdet4_pos = SDL::moduleConnectionMap_pLStoLayer3Subdet4_pos.getConnectedModuleDetIds(isuperbin);
      std::vector<unsigned int> connectedModuleDetIds_pLStoLayer4Subdet4_pos = SDL::moduleConnectionMap_pLStoLayer4Subdet4_pos.getConnectedModuleDetIds(isuperbin);
      connectedModuleDetIds_pos.insert(connectedModuleDetIds_pos.end(),connectedModuleDetIds_pLStoLayer1Subdet5_pos.begin(),connectedModuleDetIds_pLStoLayer1Subdet5_pos.end());
      connectedModuleDetIds_pos.insert(connectedModuleDetIds_pos.end(),connectedModuleDetIds_pLStoLayer2Subdet5_pos.begin(),connectedModuleDetIds_pLStoLayer2Subdet5_pos.end());
      connectedModuleDetIds_pos.insert(connectedModuleDetIds_pos.end(),connectedModuleDetIds_pLStoLayer3Subdet5_pos.begin(),connectedModuleDetIds_pLStoLayer3Subdet5_pos.end());
      connectedModuleDetIds_pos.insert(connectedModuleDetIds_pos.end(),connectedModuleDetIds_pLStoLayer1Subdet4_pos.begin(),connectedModuleDetIds_pLStoLayer1Subdet4_pos.end());
      connectedModuleDetIds_pos.insert(connectedModuleDetIds_pos.end(),connectedModuleDetIds_pLStoLayer2Subdet4_pos.begin(),connectedModuleDetIds_pLStoLayer2Subdet4_pos.end());
      connectedModuleDetIds_pos.insert(connectedModuleDetIds_pos.end(),connectedModuleDetIds_pLStoLayer3Subdet4_pos.begin(),connectedModuleDetIds_pLStoLayer3Subdet4_pos.end());
      connectedModuleDetIds_pos.insert(connectedModuleDetIds_pos.end(),connectedModuleDetIds_pLStoLayer4Subdet4_pos.begin(),connectedModuleDetIds_pLStoLayer4Subdet4_pos.end());

      int sizes_pos =0;
      sizes_pos += connectedModuleDetIds_pLStoLayer1Subdet5_pos.size();
      sizes_pos += connectedModuleDetIds_pLStoLayer2Subdet5_pos.size();
      sizes_pos += connectedModuleDetIds_pLStoLayer3Subdet5_pos.size();
      sizes_pos += connectedModuleDetIds_pLStoLayer1Subdet4_pos.size();
      sizes_pos += connectedModuleDetIds_pLStoLayer2Subdet4_pos.size();
      sizes_pos += connectedModuleDetIds_pLStoLayer3Subdet4_pos.size();
      sizes_pos += connectedModuleDetIds_pLStoLayer4Subdet4_pos.size();
      pixelMapping.connectedPixelsIndexPos[isuperbin] = totalSizes_pos;
      pixelMapping.connectedPixelsSizesPos[isuperbin] = sizes_pos;
      totalSizes_pos += sizes_pos;


      std::vector<unsigned int> connectedModuleDetIds_pLStoLayer1Subdet5_neg = SDL::moduleConnectionMap_pLStoLayer1Subdet5_neg.getConnectedModuleDetIds(isuperbin);
      std::vector<unsigned int> connectedModuleDetIds_pLStoLayer2Subdet5_neg = SDL::moduleConnectionMap_pLStoLayer2Subdet5_neg.getConnectedModuleDetIds(isuperbin);
      std::vector<unsigned int> connectedModuleDetIds_pLStoLayer3Subdet5_neg = SDL::moduleConnectionMap_pLStoLayer3Subdet5_neg.getConnectedModuleDetIds(isuperbin);
      std::vector<unsigned int> connectedModuleDetIds_pLStoLayer1Subdet4_neg = SDL::moduleConnectionMap_pLStoLayer1Subdet4_neg.getConnectedModuleDetIds(isuperbin);
      std::vector<unsigned int> connectedModuleDetIds_pLStoLayer2Subdet4_neg = SDL::moduleConnectionMap_pLStoLayer2Subdet4_neg.getConnectedModuleDetIds(isuperbin);
      std::vector<unsigned int> connectedModuleDetIds_pLStoLayer3Subdet4_neg = SDL::moduleConnectionMap_pLStoLayer3Subdet4_neg.getConnectedModuleDetIds(isuperbin);
      std::vector<unsigned int> connectedModuleDetIds_pLStoLayer4Subdet4_neg = SDL::moduleConnectionMap_pLStoLayer4Subdet4_neg.getConnectedModuleDetIds(isuperbin);
      connectedModuleDetIds_neg.insert(connectedModuleDetIds_neg.end(),connectedModuleDetIds_pLStoLayer1Subdet5_neg.begin(),connectedModuleDetIds_pLStoLayer1Subdet5_neg.end());
      connectedModuleDetIds_neg.insert(connectedModuleDetIds_neg.end(),connectedModuleDetIds_pLStoLayer2Subdet5_neg.begin(),connectedModuleDetIds_pLStoLayer2Subdet5_neg.end());
      connectedModuleDetIds_neg.insert(connectedModuleDetIds_neg.end(),connectedModuleDetIds_pLStoLayer3Subdet5_neg.begin(),connectedModuleDetIds_pLStoLayer3Subdet5_neg.end());
      connectedModuleDetIds_neg.insert(connectedModuleDetIds_neg.end(),connectedModuleDetIds_pLStoLayer1Subdet4_neg.begin(),connectedModuleDetIds_pLStoLayer1Subdet4_neg.end());
      connectedModuleDetIds_neg.insert(connectedModuleDetIds_neg.end(),connectedModuleDetIds_pLStoLayer2Subdet4_neg.begin(),connectedModuleDetIds_pLStoLayer2Subdet4_neg.end());
      connectedModuleDetIds_neg.insert(connectedModuleDetIds_neg.end(),connectedModuleDetIds_pLStoLayer3Subdet4_neg.begin(),connectedModuleDetIds_pLStoLayer3Subdet4_neg.end());
      connectedModuleDetIds_neg.insert(connectedModuleDetIds_neg.end(),connectedModuleDetIds_pLStoLayer4Subdet4_neg.begin(),connectedModuleDetIds_pLStoLayer4Subdet4_neg.end());

      int sizes_neg =0;
      sizes_neg += connectedModuleDetIds_pLStoLayer1Subdet5_neg.size();
      sizes_neg += connectedModuleDetIds_pLStoLayer2Subdet5_neg.size();
      sizes_neg += connectedModuleDetIds_pLStoLayer3Subdet5_neg.size();
      sizes_neg += connectedModuleDetIds_pLStoLayer1Subdet4_neg.size();
      sizes_neg += connectedModuleDetIds_pLStoLayer2Subdet4_neg.size();
      sizes_neg += connectedModuleDetIds_pLStoLayer3Subdet4_neg.size();
      sizes_neg += connectedModuleDetIds_pLStoLayer4Subdet4_neg.size();
      pixelMapping.connectedPixelsIndexNeg[isuperbin] = totalSizes_neg;
      pixelMapping.connectedPixelsSizesNeg[isuperbin] = sizes_neg;
      totalSizes_neg += sizes_neg;

    }

    unsigned int* connectedPixels;
    cudaMallocHost(&connectedPixels,(totalSizes+totalSizes_pos+totalSizes_neg) * sizeof(unsigned int));
//#ifdef CACHE_ALLOC
//    //cudaStream_t stream=0;
//#ifdef Explicit_Module
//    int dev;
//    cudaGetDevice(&dev);
//    modulesInGPU.connectedPixels =    (unsigned int*)cms::cuda::allocate_device(dev,(totalSizes+totalSizes_pos+totalSizes_neg) * sizeof(unsigned int),stream);
//#else
//    modulesInGPU.connectedPixels =       (unsigned int*)cms::cuda::allocate_managed((totalSizes+totalSizes_pos+totalSizes_neg) * sizeof(unsigned int),stream);
//#endif
//#else
#ifdef Explicit_Module
    cudaMalloc(&modulesInGPU.connectedPixels,(totalSizes+totalSizes_pos+totalSizes_neg)* sizeof(unsigned int));
    //cudaMallocAsync(&modulesInGPU.connectedPixels,(totalSizes+totalSizes_pos+totalSizes_neg)* sizeof(unsigned int),stream);
#else
    cudaMallocManaged(&modulesInGPU.connectedPixels,(totalSizes+totalSizes_pos+totalSizes_neg)* sizeof(unsigned int));
#endif
//#endif

    for(int icondet=0; icondet< totalSizes; icondet++){
      connectedPixels[icondet] = (*detIdToIndex)[connectedModuleDetIds[icondet]];
    }
    for(int icondet=0; icondet< totalSizes_pos; icondet++){
      connectedPixels[icondet+totalSizes] = (*detIdToIndex)[connectedModuleDetIds_pos[icondet]];
    }
    for(int icondet=0; icondet< totalSizes_neg; icondet++){
      connectedPixels[icondet+totalSizes+totalSizes_pos] = (*detIdToIndex)[connectedModuleDetIds_neg[icondet]];
    }
    cudaMemcpyAsync(modulesInGPU.connectedPixels,connectedPixels,(totalSizes+totalSizes_pos+totalSizes_neg)*sizeof(unsigned int),cudaMemcpyHostToDevice,stream);
    cudaStreamSynchronize(stream);

    cudaFreeHost(connectedPixels);
}

void SDL::fillConnectedModuleArrayExplicit(struct modules& modulesInGPU, unsigned int nModules,cudaStream_t stream)
{
    unsigned int* moduleMap;
    unsigned int* nConnectedModules;
    cudaMallocHost(&moduleMap,nModules * 40 * sizeof(unsigned int));
    cudaMallocHost(&nConnectedModules,nModules * sizeof(unsigned int));
    for(auto it = (*detIdToIndex).begin(); it != (*detIdToIndex).end(); ++it)
    {
        unsigned int detId = it->first;
        unsigned int index = it->second;
        auto& connectedModules = moduleConnectionMap.getConnectedModuleDetIds(detId);
        nConnectedModules[index] = connectedModules.size();
        for(unsigned int i = 0; i< nConnectedModules[index];i++)
        {
            moduleMap[index * 40 + i] = (*detIdToIndex)[connectedModules[i]];
        }
    }
    cudaMemcpyAsync(modulesInGPU.moduleMap,moduleMap,nModules*40*sizeof(unsigned int),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(modulesInGPU.nConnectedModules,nConnectedModules,nModules*sizeof(unsigned int),cudaMemcpyHostToDevice,stream);
    cudaStreamSynchronize(stream);
    cudaFreeHost(moduleMap);
    cudaFreeHost(nConnectedModules);
}
void SDL::fillConnectedModuleArray(struct modules& modulesInGPU, unsigned int nModules)
{
    for(auto it = (*detIdToIndex).begin(); it != (*detIdToIndex).end(); ++it)
    {
        unsigned int detId = it->first;
        unsigned int index = it->second;
        auto& connectedModules = moduleConnectionMap.getConnectedModuleDetIds(detId);
        modulesInGPU.nConnectedModules[index] = connectedModules.size();
        for(unsigned int i = 0; i< modulesInGPU.nConnectedModules[index];i++)
        {
            modulesInGPU.moduleMap[index * 40 + i] = (*detIdToIndex)[connectedModules[i]];
        }
    }
}

void SDL::setDerivedQuantities(unsigned int detId, unsigned short& layer, unsigned short& ring, unsigned short& rod, unsigned short& module, unsigned short& subdet, unsigned short& side)
{
    subdet = (detId & (7 << 25)) >> 25;
    side = (subdet == Endcap) ? (detId & (3 << 23)) >> 23 : (detId & (3 << 18)) >> 18;
    layer = (subdet == Endcap) ? (detId & (7 << 18)) >> 18 : (detId & (7 << 20)) >> 20;
    ring = (subdet == Endcap) ? (detId & (15 << 12)) >> 12 : 0;
    module = (detId & (127 << 2)) >> 2;
    rod = (subdet == Endcap) ? 0 : (detId & (127 << 10)) >> 10;
}

//auxilliary functions - will be called as needed
bool SDL::modules::parseIsInverted(unsigned int index)
{
    if (subdets[index] == Endcap)
    {
        if (sides[index] == NegZ)
        {
            return modules[index] % 2 == 1;
        }
        else if (sides[index] == PosZ)
        {
            return modules[index] % 2 == 0;
        }
        else
        {
            return 0;
        }
    }
    else if (subdets[index] == Barrel)
    {
        if (sides[index] == Center)
        {
            if (layers[index] <= 3)
            {
                return modules[index] % 2 == 1;
            }
            else if (layers[index] >= 4)
            {
                return modules[index] % 2 == 0;
            }
            else
            {
                return 0;
            }
        }
        else if (sides[index] == NegZ or sides[index] == PosZ)
        {
            if (layers[index] <= 2)
            {
                return modules[index] % 2 == 1;
            }
            else if (layers[index] == 3)
            {
                return modules[index] % 2 == 0;
            }
            else
            {
                return 0;
            }
        }
        else
        {
            return 0;
        }
    }
    else
    {
        return 0;
    }
}
bool SDL::modules::parseIsInverted(unsigned int index, short subdet, short side, short module, short layer)
{
    if (subdet == Endcap)
    {
        if (side == NegZ)
        {
            return module % 2 == 1;
        }
        else if (side == PosZ)
        {
            return module % 2 == 0;
        }
        else
        {
            return 0;
        }
    }
    else if (subdet == Barrel)
    {
        if (side == Center)
        {
            if (layer <= 3)
            {
                return module % 2 == 1;
            }
            else if (layer >= 4)
            {
                return module % 2 == 0;
            }
            else
            {
                return 0;
            }
        }
        else if (side == NegZ or side == PosZ)
        {
            if (layer <= 2)
            {
                return module % 2 == 1;
            }
            else if (layer == 3)
            {
                return module % 2 == 0;
            }
            else
            {
                return 0;
            }
        }
        else
        {
            return 0;
        }
    }
    else
    {
        return 0;
    }
}

bool SDL::modules::parseIsLower(unsigned int index, bool isInvertedx, unsigned int detId)
{
    return (isInvertedx) ? !(detId & 1) : (detId & 1);
}
bool SDL::modules::parseIsLower(unsigned int index)
{
    return (isInverted[index]) ? !(detIds[index] & 1) : (detIds[index] & 1);
}

/*
unsigned int SDL::modules::partnerModuleIndexExplicit(unsigned int index, bool isLowerx, bool isInvertedx)
{
    // We need to ensure modules with successive det Ids are right next to each other or we're dead
    if(isLowerx)
    {
        return (isInvertedx ? index - 1: index + 1);
    }
    else
    {
        return (isInvertedx ? index + 1 : index - 1);
    }
}
unsigned int SDL::modules::partnerModuleIndex(unsigned int index)
{
    // We need to ensure modules with successive det Ids are right next to each other or we're dead
    if(isLower[index])
    {
        return (isInverted[index] ? index - 1: index + 1);
    }
    else
    {
        return (isInverted[index] ? index + 1 : index - 1);
    }
}
*/
SDL::ModuleType SDL::modules::parseModuleType(unsigned int index, short subdet, short layer, short ring)
{
    if(subdet == Barrel)
    {
        if(layer <= 3)
        {
            return PS;
        }
        else
        {
            return TwoS;
        }
    }
    else
    {
        if(layer <= 2)
        {
            if(ring <= 10)
            {
                return PS;
            }
            else
            {
                return TwoS;
            }
        }
        else
        {
            if(ring <= 7)
            {
                return PS;
            }
            else
            {
                return TwoS;
            }
        }
    }
}
SDL::ModuleType SDL::modules::parseModuleType(unsigned int index)
{
    if(subdets[index] == Barrel)
    {
        if(layers[index] <= 3)
        {
            return PS;
        }
        else
        {
            return TwoS;
        }
    }
    else
    {
        if(layers[index] <= 2)
        {
            if(rings[index] <= 10)
            {
                return PS;
            }
            else
            {
                return TwoS;
            }
        }
        else
        {
            if(rings[index] <= 7)
            {
                return PS;
            }
            else
            {
                return TwoS;
            }
        }
    }
}

SDL::ModuleLayerType SDL::modules::parseModuleLayerType(unsigned int index, ModuleType moduleTypex,bool isInvertedx, bool isLowerx)
{
    if(moduleTypex == TwoS)
    {
        return Strip;
    }
    if(isInvertedx)
    {
        if(isLowerx)
        {
            return Strip;
        }
        else
        {
            return Pixel;
        }
    }
    else
   {
        if(isLowerx)
        {
            return Pixel;
        }
        else
        {
            return Strip;
        }
    }
}
SDL::ModuleLayerType SDL::modules::parseModuleLayerType(unsigned int index)
{
    if(moduleType[index] == TwoS)
    {
        return Strip;
    }
    if(isInverted[index])
    {
        if(isLower[index])
        {
            return Strip;
        }
        else
        {
            return Pixel;
        }
    }
    else
   {
        if(isLower[index])
        {
            return Pixel;
        }
        else
        {
            return Strip;
        }
    }
}

void SDL::resetObjectRanges(struct objectRanges& rangesInGPU, unsigned int nModules,cudaStream_t stream)
{
//#ifdef Explicit_Module
        cudaMemsetAsync(rangesInGPU.hitRanges, -1,nModules*2*sizeof(int),stream);
        cudaMemsetAsync(rangesInGPU.mdRanges, -1,nModules*2*sizeof(int),stream);
        cudaMemsetAsync(rangesInGPU.segmentRanges, -1,nModules*2*sizeof(int),stream);
        cudaMemsetAsync(rangesInGPU.trackletRanges, -1,nModules*2*sizeof(int),stream);
        cudaMemsetAsync(rangesInGPU.tripletRanges, -1,nModules*2*sizeof(int),stream);
        cudaMemsetAsync(rangesInGPU.trackCandidateRanges, -1,nModules*2*sizeof(int),stream);
        cudaMemsetAsync(rangesInGPU.quintupletRanges, -1, nModules*2*sizeof(int),stream);
        cudaStreamSynchronize(stream);
}


/*
   Find the modules that are "staggered neighbours" to a given module
   
   Both barrel central : same module, adjacent rod (modulo number of rods) 
   Both barrel tilted :  same rod, adjacent module (modulo number of modules, which is the same as number of rods for flat)
   One Barrel flat, One Barrel tilted : 
      Left and Center : Left module's rod = 12, center module's module = 1, and left module's module = center module's rod
      Center and Right : Right module's rod = 1 center module's module = max module number, and right module's module = center module's rod

   Endcap : Same ring -> Adjacent modules,
   TODO: Endcap different ring, and tilted->endcap
*/


__device__ const int nEndcapModulesInner[] = {20,24,24,28,32,32,36,40,40,44,52,60,64,72,76};
__device__ const int nEndcapModulesOuter[] = {28,28,32,36,36,40,44,52,56,64,72,76};

__device__ const int nCentralBarrelModules[] = {7,11,15,24,24,24};
__device__ const int nCentralRods[] = {18, 26, 36, 48, 60, 78};

__device__ void findStaggeredNeighbours(struct SDL::modules& modulesInGPU, unsigned int moduleIdx, unsigned int* staggeredNeighbours, unsigned int& counter)
{
    //naive and expensive method
    counter = 0;
    bool flag = false;
    for(size_t i = 0; i < *(modulesInGPU.nLowerModules); i++)
    {
        flag = false;
        unsigned int partnerModuleIdx = modulesInGPU.lowerModuleIndices[i];
        //start
        unsigned int layer1 = modulesInGPU.layers[moduleIdx];
        unsigned int layer2 = modulesInGPU.layers[partnerModuleIdx];
        unsigned int module1 = modulesInGPU.modules[moduleIdx];
        unsigned int module2 = modulesInGPU.modules[partnerModuleIdx];

        if(layer1 != layer2) continue;

        if(modulesInGPU.subdets[moduleIdx] == 4 and modulesInGPU.subdets[partnerModuleIdx] == 4)
        {
            unsigned int ring1 = modulesInGPU.rings[moduleIdx];
            unsigned int ring2 = modulesInGPU.rings[partnerModuleIdx];
            if(ring1 != ring2) continue;

            if((layer1 <=2) and (fabsf(module1 - module2) == 1 or fabsf(module1 % nEndcapModulesInner[ring1 - 1] - module2 % nEndcapModulesInner[ring2 - 1]) == 1))
            {
                flag = true;
            }

            else if((layer1 > 2) and (fabsf(module1 - module2) == 1 or fabsf(module1 % nEndcapModulesOuter[ring1 - 1] - module2 % nEndcapModulesOuter[ring2 - 1]) == 1))
            {
                flag = true;
            }
        }
        else if(modulesInGPU.subdets[moduleIdx] == 5 and modulesInGPU.subdets[partnerModuleIdx] == 5)
        {
            unsigned int rod1 = modulesInGPU.rods[moduleIdx];
            unsigned int rod2 = modulesInGPU.rods[partnerModuleIdx];
            unsigned int side1 = modulesInGPU.sides[moduleIdx];
            unsigned int side2 = modulesInGPU.sides[partnerModuleIdx];
            

            if(side1 == side2)             
            {
                if((fabsf(rod1 - rod2) == 1 and module1 == module2) or (fabsf(module1 - module2) == 1 and rod1 == rod2))
                {
                    flag = true;
                }
                else if(side1 == 3 and side2 == 3 and fabsf(rod1 % nCentralRods[layer1 - 1] - rod2 % nCentralRods[layer2 - 1]) == 1 and module1 == module2)
                {
                    flag = true;
                }
                else if(side1 != 3 and  fabsf(module1 % nCentralRods[layer1 - 1] - module2 % nCentralRods[layer2 - 1]) == 1 and rod1 == rod2)
                {
                    flag = true;
                }
            }
            else
            {
                if(side1 == 1 and side2 == 3 and rod1 == 12 and module2 == 1 and module1 == rod2)
                {
                    flag = true;
                }
                else if(side1 == 3 and side2 == 1 and rod2 == 12 and module1 == 1 and module1 == rod2)
                {
                    flag = true;
                }
                else if(side1 == 2 and side2 == 3 and rod1 == 1 and module2 == nCentralBarrelModules[layer2 - 1] and module1 == rod2)
                {
                    flag = true;
                }
                else if(side1 == 3 and side2 == 2 and module1 == nCentralBarrelModules[layer1 - 1] and rod2 == 1 and rod1 == module2)
                {
                    flag = true;
                }
            }
        }
        if(flag)
        {
            staggeredNeighbours[counter] = modulesInGPU.lowerModuleIndices[i]; //deal in lower module indices
            counter++;
        }
    }
}
