# include "Module.cuh"
#include "ModuleConnectionMap.h"
#include "allocate.h"
std::map <unsigned int, unsigned int> *SDL::detIdToIndex;

void SDL::createModulesInUnifiedMemory(struct modules& modulesInGPU,unsigned int nModules)
{
    /* modules stucture object will be created in Event.cu*/
#ifdef CACHE_ALLOC
    cudaStream_t stream=0; 
    modulesInGPU.detIds =            (unsigned int*)cms::cuda::allocate_managed(nModules * sizeof(unsigned int),stream);
    modulesInGPU.moduleMap =         (unsigned int*)cms::cuda::allocate_managed(nModules * 40 * sizeof(unsigned int),stream);
    modulesInGPU.nConnectedModules = (unsigned int*)cms::cuda::allocate_managed(nModules * sizeof(unsigned int),stream);
    modulesInGPU.drdzs =                    (float*)cms::cuda::allocate_managed(nModules * sizeof(float),stream);
    modulesInGPU.slopes =                   (float*)cms::cuda::allocate_managed(nModules * sizeof(float),stream);
    modulesInGPU.nModules =          (unsigned int*)cms::cuda::allocate_managed(sizeof(unsigned int),stream);
    modulesInGPU.nLowerModules =     (unsigned int*)cms::cuda::allocate_managed(sizeof(unsigned int),stream);
    modulesInGPU.layers =                   (short*)cms::cuda::allocate_managed(nModules * sizeof(short),stream);
    modulesInGPU.rings =                    (short*)cms::cuda::allocate_managed(nModules * sizeof(short),stream);
    modulesInGPU.modules =                  (short*)cms::cuda::allocate_managed(nModules * sizeof(short),stream);
    modulesInGPU.rods =                    (short*)cms::cuda::allocate_managed(nModules * sizeof(short),stream);
    modulesInGPU.subdets =                 (short*)cms::cuda::allocate_managed(nModules * sizeof(short),stream);
    modulesInGPU.sides =                   (short*)cms::cuda::allocate_managed(nModules * sizeof(short),stream);
    modulesInGPU.isInverted =               (bool*)cms::cuda::allocate_managed(nModules * sizeof(bool),stream);
    modulesInGPU.isLower =                  (bool*)cms::cuda::allocate_managed(nModules * sizeof(bool),stream);
    modulesInGPU.nEligibleModules =     (unsigned int*)cms::cuda::allocate_managed(sizeof(unsigned int),stream);

    modulesInGPU.hitRanges =                 (int*)cms::cuda::allocate_managed(nModules * 2 * sizeof(int),stream);
    modulesInGPU.mdRanges =                  (int*)cms::cuda::allocate_managed(nModules * 2 * sizeof(int),stream);
    modulesInGPU.segmentRanges =             (int*)cms::cuda::allocate_managed(nModules * 2 * sizeof(int),stream);
    modulesInGPU.trackletRanges =            (int*)cms::cuda::allocate_managed(nModules * 2 * sizeof(int),stream);
    modulesInGPU.tripletRanges =             (int*)cms::cuda::allocate_managed(nModules * 2 * sizeof(int),stream);
    modulesInGPU.trackCandidateRanges =      (int*)cms::cuda::allocate_managed(nModules * 2 * sizeof(int),stream);

    modulesInGPU.moduleType =         (ModuleType*)cms::cuda::allocate_managed(nModules * sizeof(ModuleType),stream);
    modulesInGPU.moduleLayerType=(ModuleLayerType*)cms::cuda::allocate_managed(nModules * sizeof(ModuleLayerType),stream);
#else
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
    cudaMallocManaged(&modulesInGPU.nEligibleModules,sizeof(unsigned int));

    cudaMallocManaged(&modulesInGPU.hitRanges,nModules * 2 * sizeof(int));
    cudaMallocManaged(&modulesInGPU.mdRanges,nModules * 2 * sizeof(int));
    cudaMallocManaged(&modulesInGPU.segmentRanges,nModules * 2 * sizeof(int));
    cudaMallocManaged(&modulesInGPU.trackletRanges,nModules * 2 * sizeof(int));
    cudaMallocManaged(&modulesInGPU.tripletRanges,nModules * 2 * sizeof(int));
    cudaMallocManaged(&modulesInGPU.trackCandidateRanges, nModules * 2 * sizeof(int));

    cudaMallocManaged(&modulesInGPU.moduleType,nModules * sizeof(ModuleType));
    cudaMallocManaged(&modulesInGPU.moduleLayerType,nModules * sizeof(ModuleLayerType));
#endif


    *modulesInGPU.nModules = nModules;
}
void SDL::createModulesInExplicitMemory(struct modules& modulesInGPU,unsigned int nModules)
{
    /* modules stucture object will be created in Event.cu*/
#ifdef CACHE_ALLOC
    cudaStream_t stream=0; 
    int dev;
    cudaGetDevice(&dev);
    modulesInGPU.detIds =            (unsigned int*)cms::cuda::allocate_device(dev,nModules * sizeof(unsigned int),stream);
    modulesInGPU.moduleMap =         (unsigned int*)cms::cuda::allocate_device(dev,nModules * 40 * sizeof(unsigned int),stream);
    modulesInGPU.nConnectedModules = (unsigned int*)cms::cuda::allocate_device(dev,nModules * sizeof(unsigned int),stream);
    modulesInGPU.drdzs =                    (float*)cms::cuda::allocate_device(dev,nModules * sizeof(float),stream);
    modulesInGPU.slopes =                   (float*)cms::cuda::allocate_device(dev,nModules * sizeof(float),stream);
    modulesInGPU.nModules =          (unsigned int*)cms::cuda::allocate_device(dev,sizeof(unsigned int),stream);
    modulesInGPU.nLowerModules =     (unsigned int*)cms::cuda::allocate_device(dev,sizeof(unsigned int),stream);
    modulesInGPU.layers =                   (short*)cms::cuda::allocate_device(dev,nModules * sizeof(short),stream);
    modulesInGPU.rings =                    (short*)cms::cuda::allocate_device(dev,nModules * sizeof(short),stream);
    modulesInGPU.modules =                  (short*)cms::cuda::allocate_device(dev,nModules * sizeof(short),stream);
    modulesInGPU.rods =                     (short*)cms::cuda::allocate_device(dev,nModules * sizeof(short),stream);
    modulesInGPU.subdets =                  (short*)cms::cuda::allocate_device(dev,nModules * sizeof(short),stream);
    modulesInGPU.sides =                    (short*)cms::cuda::allocate_device(dev,nModules * sizeof(short),stream);
    modulesInGPU.isInverted =                (bool*)cms::cuda::allocate_device(dev,nModules * sizeof(bool),stream);
    modulesInGPU.isLower =                   (bool*)cms::cuda::allocate_device(dev,nModules * sizeof(bool),stream);
    modulesInGPU.nEligibleModules =     (unsigned int*)cms::cuda::allocate_device(dev,sizeof(unsigned int),stream);

    modulesInGPU.hitRanges =                  (int*)cms::cuda::allocate_device(dev,nModules * 2 * sizeof(int),stream);
    modulesInGPU.mdRanges =                   (int*)cms::cuda::allocate_device(dev,nModules * 2 * sizeof(int),stream);
    modulesInGPU.segmentRanges =              (int*)cms::cuda::allocate_device(dev,nModules * 2 * sizeof(int),stream);
    modulesInGPU.trackletRanges =             (int*)cms::cuda::allocate_device(dev,nModules * 2 * sizeof(int),stream);
    modulesInGPU.tripletRanges =              (int*)cms::cuda::allocate_device(dev,nModules * 2 * sizeof(int),stream);
    modulesInGPU.trackCandidateRanges =       (int*)cms::cuda::allocate_device(dev,nModules * 2 * sizeof(int),stream);

    modulesInGPU.moduleType =          (ModuleType*)cms::cuda::allocate_device(dev,nModules * sizeof(ModuleType),stream);
    modulesInGPU.moduleLayerType= (ModuleLayerType*)cms::cuda::allocate_device(dev,nModules * sizeof(ModuleLayerType),stream);
#else
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
    cudaMalloc(&modulesInGPU.nEligibleModules,sizeof(unsigned int));

    cudaMalloc(&modulesInGPU.hitRanges,nModules * 2 * sizeof(int));
    cudaMalloc(&modulesInGPU.mdRanges,nModules * 2 * sizeof(int));
    cudaMalloc(&modulesInGPU.segmentRanges,nModules * 2 * sizeof(int));
    cudaMalloc(&modulesInGPU.trackletRanges,nModules * 2 * sizeof(int));
    cudaMalloc(&modulesInGPU.tripletRanges,nModules * 2 * sizeof(int));
    cudaMalloc(&modulesInGPU.trackCandidateRanges, nModules * 2 * sizeof(int));

    cudaMalloc(&modulesInGPU.moduleType,nModules * sizeof(ModuleType));
    cudaMalloc(&modulesInGPU.moduleLayerType,nModules * sizeof(ModuleLayerType));
#endif

    cudaMemcpy(modulesInGPU.nModules,&nModules,sizeof(unsigned int),cudaMemcpyHostToDevice);
}

void SDL::freeModulesCache(struct modules& modulesInGPU)
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
  cms::cuda::free_device(dev,modulesInGPU.hitRanges);
  cms::cuda::free_device(dev,modulesInGPU.mdRanges);
  cms::cuda::free_device(dev,modulesInGPU.segmentRanges);
  cms::cuda::free_device(dev,modulesInGPU.trackletRanges);
  cms::cuda::free_device(dev,modulesInGPU.tripletRanges);
  cms::cuda::free_device(dev,modulesInGPU.trackCandidateRanges);
  cms::cuda::free_device(dev,modulesInGPU.moduleType);
  cms::cuda::free_device(dev,modulesInGPU.moduleLayerType);
  cms::cuda::free_device(dev,modulesInGPU.lowerModuleIndices);
  cms::cuda::free_device(dev,modulesInGPU.reverseLookupLowerModuleIndices);
  cms::cuda::free_device(dev,modulesInGPU.trackCandidateModuleIndices);
  cms::cuda::free_device(dev,modulesInGPU.nEligibleModules);
  cms::cuda::free_device(dev,modulesInGPU.connectedPixels);
  cms::cuda::free_device(dev,modulesInGPU.connectedPixelsPos);
  cms::cuda::free_device(dev,modulesInGPU.connectedPixelsNeg);
  cms::cuda::free_device(dev,modulesInGPU.connectedPixelsSizes);
  cms::cuda::free_device(dev,modulesInGPU.connectedPixelsSizesPos);
  cms::cuda::free_device(dev,modulesInGPU.connectedPixelsSizesNeg);
  cms::cuda::free_device(dev,modulesInGPU.connectedPixelsIndex);
  cms::cuda::free_device(dev,modulesInGPU.connectedPixelsIndexPos);
  cms::cuda::free_device(dev,modulesInGPU.connectedPixelsIndexNeg);
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
  cms::cuda::free_managed(modulesInGPU.hitRanges);
  cms::cuda::free_managed(modulesInGPU.mdRanges);
  cms::cuda::free_managed(modulesInGPU.segmentRanges);
  cms::cuda::free_managed(modulesInGPU.trackletRanges);
  cms::cuda::free_managed(modulesInGPU.tripletRanges);
  cms::cuda::free_managed(modulesInGPU.trackCandidateRanges);
  cms::cuda::free_managed(modulesInGPU.moduleType);
  cms::cuda::free_managed(modulesInGPU.moduleLayerType);
  cms::cuda::free_managed(modulesInGPU.lowerModuleIndices);
  cms::cuda::free_managed(modulesInGPU.reverseLookupLowerModuleIndices);
  cms::cuda::free_managed(modulesInGPU.trackCandidateModuleIndices);
  cms::cuda::free_managed(modulesInGPU.nEligibleModules);
  cms::cuda::free_managed(modulesInGPU.connectedPixels);
  cms::cuda::free_managed(modulesInGPU.connectedPixelsPos);
  cms::cuda::free_managed(modulesInGPU.connectedPixelsNeg);
  cms::cuda::free_managed(modulesInGPU.connectedPixelsSizes);
  cms::cuda::free_managed(modulesInGPU.connectedPixelsSizesPos);
  cms::cuda::free_managed(modulesInGPU.connectedPixelsSizesNeg);
  cms::cuda::free_managed(modulesInGPU.connectedPixelsIndex);
  cms::cuda::free_managed(modulesInGPU.connectedPixelsIndexPos);
  cms::cuda::free_managed(modulesInGPU.connectedPixelsIndexNeg);
#endif
}
void SDL::freeModules(struct modules& modulesInGPU)
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
  cudaFree(modulesInGPU.hitRanges);
  cudaFree(modulesInGPU.mdRanges);
  cudaFree(modulesInGPU.segmentRanges);
  cudaFree(modulesInGPU.trackletRanges);
  cudaFree(modulesInGPU.tripletRanges);
  cudaFree(modulesInGPU.trackCandidateRanges);
  cudaFree(modulesInGPU.moduleType);
  cudaFree(modulesInGPU.moduleLayerType);
  cudaFree(modulesInGPU.lowerModuleIndices);
  cudaFree(modulesInGPU.reverseLookupLowerModuleIndices);
  cudaFree(modulesInGPU.trackCandidateModuleIndices);
  cudaFree(modulesInGPU.nEligibleModules);
  cudaFree(modulesInGPU.connectedPixels);
  cudaFree(modulesInGPU.connectedPixelsPos);
  cudaFree(modulesInGPU.connectedPixelsNeg);
  cudaFree(modulesInGPU.connectedPixelsSizes);
  cudaFree(modulesInGPU.connectedPixelsSizesPos);
  cudaFree(modulesInGPU.connectedPixelsSizesNeg);
  cudaFree(modulesInGPU.connectedPixelsIndex);
  cudaFree(modulesInGPU.connectedPixelsIndexPos);
  cudaFree(modulesInGPU.connectedPixelsIndexNeg);
}

void SDL::createLowerModuleIndexMapExplicit(struct modules& modulesInGPU, unsigned int nLowerModules, unsigned int nModules,bool* isLower)
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
    #ifdef CACHE_ALLOC
    cudaStream_t stream =0;
    int dev;
    cudaGetDevice(&dev);
    modulesInGPU.lowerModuleIndices = (unsigned int*)cms::cuda::allocate_device(dev,(nLowerModules + 1) * sizeof(unsigned int),stream);
    modulesInGPU.reverseLookupLowerModuleIndices = (int*)cms::cuda::allocate_device(dev,nModules * sizeof(int),stream);
    modulesInGPU.trackCandidateModuleIndices = (int*)cms::cuda::allocate_device(dev,(nLowerModules + 1) * sizeof(int),stream);
    #else
    cudaMalloc(&modulesInGPU.lowerModuleIndices,(nLowerModules + 1) * sizeof(unsigned int));
    cudaMalloc(&modulesInGPU.reverseLookupLowerModuleIndices,nModules * sizeof(int));
    cudaMalloc(&modulesInGPU.trackCandidateModuleIndices, (nLowerModules + 1) * sizeof(int));
    #endif
    cudaMemcpy(modulesInGPU.lowerModuleIndices,lowerModuleIndices,sizeof(unsigned int)*(nLowerModules+1),cudaMemcpyHostToDevice);
    cudaMemcpy(modulesInGPU.reverseLookupLowerModuleIndices,reverseLookupLowerModuleIndices,sizeof(int)*nModules,cudaMemcpyHostToDevice);
   
    cudaFreeHost(lowerModuleIndices);
    cudaFreeHost(reverseLookupLowerModuleIndices);
}
void SDL::createLowerModuleIndexMap(struct modules& modulesInGPU, unsigned int nLowerModules, unsigned int nModules)
{
    //FIXME:some hacks to get the pixel module in the lower modules index without incrementing nLowerModules counter!
    //Reproduce these hacks in the explicit memory for identical results (or come up with a better method)
    #ifdef CACHE_ALLOC
    cudaStream_t stream =0;
    modulesInGPU.lowerModuleIndices = (unsigned int*)cms::cuda::allocate_managed((nLowerModules + 1) * sizeof(unsigned int),stream);
    modulesInGPU.reverseLookupLowerModuleIndices = (int*)cms::cuda::allocate_managed(nModules * sizeof(int),stream);
    modulesInGPU.trackCandidateModuleIndices = (int*)cms::cuda::allocate_managed((nLowerModules + 1) * sizeof(int),stream);
    #else
    cudaMallocManaged(&modulesInGPU.lowerModuleIndices,(nLowerModules + 1) * sizeof(unsigned int));
    cudaMallocManaged(&modulesInGPU.reverseLookupLowerModuleIndices,nModules * sizeof(int));
    cudaMallocManaged(&modulesInGPU.trackCandidateModuleIndices, (nLowerModules + 1) * sizeof(int));
    #endif



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

void SDL::loadModulesFromFile(struct modules& modulesInGPU, unsigned int& nModules, struct pixelMap& pixelMapping, const char* moduleMetaDataFilePath)
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
    //FIXME:MANUAL INSERTION OF PIXEL MODULE!
    (*detIdToIndex)[1] = counter; //pixel module is the last module in the module list
    counter++;
    nModules = counter;
    std::cout<<"Number of modules = "<<nModules<<std::endl;
#ifdef Explicit_Module
    createModulesInExplicitMemory(modulesInGPU,nModules);
    unsigned int* lowerModuleCounter;// = 0;
    cudaMallocHost(&lowerModuleCounter,sizeof(unsigned int));
    cudaMemset(lowerModuleCounter,0,sizeof(unsigned int));
    unsigned int* host_detIds;
    short* host_layers;
    short* host_rings;
    short* host_rods;
    short* host_modules;
    short* host_subdets;
    short* host_sides;
    bool* host_isInverted;
    bool* host_isLower;
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
    cudaMallocHost(&host_moduleType,sizeof(ModuleType)*nModules);
    cudaMallocHost(&host_moduleLayerType,sizeof(ModuleLayerType)*nModules);
    cudaMallocHost(&host_slopes,sizeof(float)*nModules);
    cudaMallocHost(&host_drdzs,sizeof(float)*nModules);
    for(auto it = (*detIdToIndex).begin(); it != (*detIdToIndex).end(); it++)
    {
        unsigned int detId = it->first;
        unsigned int index = it->second;
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

            host_slopes[index] = (subdet == Endcap) ? endcapGeometry.getSlopeLower(detId) : tiltedGeometry.getSlope(detId);
            host_drdzs[index] = (subdet == Barrel) ? tiltedGeometry.getDrDz(detId) : 0;
        }
          lowerModuleCounter[0] += host_isLower[index];
    }

    cudaMemcpy(modulesInGPU.nLowerModules,lowerModuleCounter,sizeof(unsigned int),cudaMemcpyHostToDevice);
    cudaMemcpy(modulesInGPU.detIds,host_detIds,nModules*sizeof(unsigned int),cudaMemcpyHostToDevice);
    cudaMemcpy(modulesInGPU.layers,host_layers,nModules*sizeof(short),cudaMemcpyHostToDevice);
    cudaMemcpy(modulesInGPU.rings,host_rings,sizeof(short)*nModules,cudaMemcpyHostToDevice);
    cudaMemcpy(modulesInGPU.rods,host_rods,sizeof(short)*nModules,cudaMemcpyHostToDevice);
    cudaMemcpy(modulesInGPU.modules,host_modules,sizeof(short)*nModules,cudaMemcpyHostToDevice);
    cudaMemcpy(modulesInGPU.subdets,host_subdets,sizeof(short)*nModules,cudaMemcpyHostToDevice);
    cudaMemcpy(modulesInGPU.sides,host_sides,sizeof(short)*nModules,cudaMemcpyHostToDevice);
    cudaMemcpy(modulesInGPU.isInverted,host_isInverted,sizeof(bool)*nModules,cudaMemcpyHostToDevice);
    cudaMemcpy(modulesInGPU.isLower,host_isLower,sizeof(bool)*nModules,cudaMemcpyHostToDevice);
    cudaMemcpy(modulesInGPU.moduleType,host_moduleType,sizeof(ModuleType)*nModules,cudaMemcpyHostToDevice);
    cudaMemcpy(modulesInGPU.moduleLayerType,host_moduleLayerType,sizeof(ModuleLayerType)*nModules,cudaMemcpyHostToDevice);
    cudaMemcpy(modulesInGPU.slopes,host_slopes,sizeof(float)*nModules,cudaMemcpyHostToDevice);
    cudaMemcpy(modulesInGPU.drdzs,host_drdzs,sizeof(float)*nModules,cudaMemcpyHostToDevice);
    cudaFreeHost(host_detIds);
    cudaFreeHost(host_layers);
    cudaFreeHost(host_rings);
    cudaFreeHost(host_rods);
    cudaFreeHost(host_modules);
    cudaFreeHost(host_subdets);
    cudaFreeHost(host_sides);
    cudaFreeHost(host_isInverted);
    cudaFreeHost(host_isLower);
    cudaFreeHost(host_moduleType);
    cudaFreeHost(host_moduleLayerType);
    cudaFreeHost(host_slopes);
    cudaFreeHost(host_drdzs);
    cudaFreeHost(lowerModuleCounter);
    std::cout<<"number of lower modules (without fake pixel module)= "<<lowerModuleCounter[0]<<std::endl;
    createLowerModuleIndexMapExplicit(modulesInGPU,lowerModuleCounter[0], nModules,host_isLower);
    fillConnectedModuleArrayExplicit(modulesInGPU,nModules);
    fillPixelMap(modulesInGPU,pixelMapping);
    resetObjectRanges(modulesInGPU,nModules);

#else
    createModulesInUnifiedMemory(modulesInGPU,nModules);
    unsigned int lowerModuleCounter = 0;
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

            modulesInGPU.slopes[index] = (subdet == Endcap) ? endcapGeometry.getSlopeLower(detId) : tiltedGeometry.getSlope(detId);
            modulesInGPU.drdzs[index] = (subdet == Barrel) ? tiltedGeometry.getDrDz(detId) : 0;
        }
        if(modulesInGPU.isLower[index]) lowerModuleCounter++;
    }
    *modulesInGPU.nLowerModules = lowerModuleCounter;
    std::cout<<"number of lower modules (without fake pixel module)= "<<*modulesInGPU.nLowerModules<<std::endl;
    createLowerModuleIndexMap(modulesInGPU,lowerModuleCounter, nModules);
    fillConnectedModuleArray(modulesInGPU,nModules);
    fillPixelMap(modulesInGPU,pixelMapping);
    resetObjectRanges(modulesInGPU,nModules);
#endif
}

void SDL::fillPixelMap(struct modules& modulesInGPU, struct pixelMap& pixelMapping) 
{
    //unsigned int* pixelMap;
    //unsigned int* nConnectedPixelModules;
    int size_superbins = SDL::moduleConnectionMap_pLStoLayer1Subdet5.size();
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
//#ifdef CACHE_ALLOC
//    cudaStream_t stream=0; 
//#ifdef Explicit_Module
//    int dev;
//    cudaGetDevice(&dev);
//    modulesInGPU.connectedPixelsIndex =    (unsigned int*)cms::cuda::allocate_device(dev,size_superbins * sizeof(unsigned int),stream);
//    modulesInGPU.connectedPixelsIndexPos = (unsigned int*)cms::cuda::allocate_device(dev,size_superbins * sizeof(unsigned int),stream);
//    modulesInGPU.connectedPixelsIndexNeg = (unsigned int*)cms::cuda::allocate_device(dev,size_superbins * sizeof(unsigned int),stream);
//    modulesInGPU.connectedPixelsSizes =     (unsigned int*)cms::cuda::allocate_device(dev,size_superbins * sizeof(unsigned int),stream);
//    modulesInGPU.connectedPixelsSizesPos =  (unsigned int*)cms::cuda::allocate_device(dev,size_superbins * sizeof(unsigned int),stream);
//    modulesInGPU.connectedPixelsSizesNeg =  (unsigned int*)cms::cuda::allocate_device(dev,size_superbins * sizeof(unsigned int),stream);
//#else
//    modulesInGPU.connectedPixelsIndex = (unsigned int*)cms::cuda::allocate_managed(size_superbins * sizeof(unsigned int),stream);
//    modulesInGPU.connectedPixelsIndexPos = (unsigned int*)cms::cuda::allocate_managed(size_superbins * sizeof(unsigned int),stream);
//    modulesInGPU.connectedPixelsIndexNeg = (unsigned int*)cms::cuda::allocate_managed(size_superbins * sizeof(unsigned int),stream);
//    modulesInGPU.connectedPixelsSizes = (unsigned int*)cms::cuda::allocate_managed(size_superbins * sizeof(unsigned int),stream);
//    modulesInGPU.connectedPixelsSizesPos = (unsigned int*)cms::cuda::allocate_managed(size_superbins * sizeof(unsigned int),stream);
//    modulesInGPU.connectedPixelsSizesNeg = (unsigned int*)cms::cuda::allocate_managed(size_superbins * sizeof(unsigned int),stream);
//#endif
//#else
//#ifdef Explicit_Module
//    cudaMalloc(&modulesInGPU.connectedPixelsIndex,size_superbins * sizeof(unsigned int));
//    cudaMalloc(&modulesInGPU.connectedPixelsIndexPos,size_superbins * sizeof(unsigned int));
//    cudaMalloc(&modulesInGPU.connectedPixelsIndexNeg,size_superbins * sizeof(unsigned int));
//    cudaMalloc(&modulesInGPU.connectedPixelsSizes,size_superbins * sizeof(unsigned int));
//    cudaMalloc(&modulesInGPU.connectedPixelsSizesPos,size_superbins * sizeof(unsigned int));
//    cudaMalloc(&modulesInGPU.connectedPixelsSizesNeg,size_superbins * sizeof(unsigned int));
//#else
//    cudaMallocManaged(&modulesInGPU.connectedPixelsIndex,size_superbins * sizeof(unsigned int));
//    cudaMallocManaged(&modulesInGPU.connectedPixelsIndexPos,size_superbins * sizeof(unsigned int));
//    cudaMallocManaged(&modulesInGPU.connectedPixelsIndexNeg,size_superbins * sizeof(unsigned int));
//    cudaMallocManaged(&modulesInGPU.connectedPixelsSizes,size_superbins * sizeof(unsigned int));
//    cudaMallocManaged(&modulesInGPU.connectedPixelsSizesPos,size_superbins * sizeof(unsigned int));
//    cudaMallocManaged(&modulesInGPU.connectedPixelsSizesNeg,size_superbins * sizeof(unsigned int));
//#endif
//#endif
    int totalSizes=0;
    int totalSizes_pos=0;
    int totalSizes_neg=0;
    for(int isuperbin =0; isuperbin<size_superbins; isuperbin++)
    {
      std::vector<unsigned int> connectedModuleDetIds_pLStoLayer1Subdet5 = SDL::moduleConnectionMap_pLStoLayer1Subdet5.getConnectedModuleDetIds(isuperbin);
      std::vector<unsigned int> connectedModuleDetIds_pLStoLayer2Subdet5 = SDL::moduleConnectionMap_pLStoLayer2Subdet5.getConnectedModuleDetIds(isuperbin);
      std::vector<unsigned int> connectedModuleDetIds_pLStoLayer3Subdet5 = SDL::moduleConnectionMap_pLStoLayer3Subdet5.getConnectedModuleDetIds(isuperbin);
      std::vector<unsigned int> connectedModuleDetIds_pLStoLayer1Subdet4 = SDL::moduleConnectionMap_pLStoLayer1Subdet4.getConnectedModuleDetIds(isuperbin);
      std::vector<unsigned int> connectedModuleDetIds_pLStoLayer2Subdet4 = SDL::moduleConnectionMap_pLStoLayer2Subdet4.getConnectedModuleDetIds(isuperbin);
      std::vector<unsigned int> connectedModuleDetIds_pLStoLayer3Subdet4 = SDL::moduleConnectionMap_pLStoLayer3Subdet4.getConnectedModuleDetIds(isuperbin);
      std::vector<unsigned int> connectedModuleDetIds_pLStoLayer4Subdet4 = SDL::moduleConnectionMap_pLStoLayer4Subdet4.getConnectedModuleDetIds(isuperbin);
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
      totalSizes += sizes;
      pixelMapping.connectedPixelsIndex[isuperbin] = totalSizes;
      pixelMapping.connectedPixelsSizes[isuperbin] = sizes;
      //modulesInGPU.connectedPixelsIndex[isuperbin] = totalSizes;
      //modulesInGPU.connectedPixelsSizes[isuperbin] = sizes;


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
      totalSizes_pos += sizes_pos;
      //modulesInGPU.connectedPixelsIndexPos[isuperbin] = totalSizes_pos;
      //modulesInGPU.connectedPixelsSizesPos[isuperbin] = sizes_pos;
      pixelMapping.connectedPixelsIndexPos[isuperbin] = totalSizes_pos;
      pixelMapping.connectedPixelsSizesPos[isuperbin] = sizes_pos;


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
      totalSizes_neg += sizes_neg;
      //modulesInGPU.connectedPixelsIndexNeg[isuperbin] = totalSizes_neg;
      //modulesInGPU.connectedPixelsSizesNeg[isuperbin] = sizes_neg;
      pixelMapping.connectedPixelsIndexNeg[isuperbin] = totalSizes_neg;
      pixelMapping.connectedPixelsSizesNeg[isuperbin] = sizes_neg;
    }

//    cudaMemcpy(modulesInGPU.connectedPixelsSizes,connectedPixelsSizes,size_superbins*sizeof(unsigned int),cudaMemcpyHostToDevice);
//    cudaMemcpy(modulesInGPU.connectedPixelsSizesPos,connectedPixelsSizesPos,size_superbins*sizeof(unsigned int),cudaMemcpyHostToDevice);
//    cudaMemcpy(modulesInGPU.connectedPixelsSizesNeg,connectedPixelsSizesNeg,size_superbins*sizeof(unsigned int),cudaMemcpyHostToDevice);
//    cudaMemcpy(modulesInGPU.connectedPixelsIndex,connectedPixelsIndex,size_superbins*sizeof(unsigned int),cudaMemcpyHostToDevice);
//    cudaMemcpy(modulesInGPU.connectedPixelsIndexPos,connectedPixelsIndexPos,size_superbins*sizeof(unsigned int),cudaMemcpyHostToDevice);
//    cudaMemcpy(modulesInGPU.connectedPixelsIndexNeg,connectedPixelsIndexNeg,size_superbins*sizeof(unsigned int),cudaMemcpyHostToDevice);

    unsigned int* connectedPixels;
    unsigned int* connectedPixelsPos;
    unsigned int* connectedPixelsNeg;
    cudaMallocHost(&connectedPixels,totalSizes * sizeof(unsigned int));
    cudaMallocHost(&connectedPixelsPos,totalSizes_pos * sizeof(unsigned int));
    cudaMallocHost(&connectedPixelsNeg,totalSizes_neg * sizeof(unsigned int));
#ifdef CACHE_ALLOC
#ifdef Explicit_Module
    modulesInGPU.connectedPixels =    (unsigned int*)cms::cuda::allocate_device(dev,totalSizes * sizeof(unsigned int),stream);
    modulesInGPU.connectedPixelsPos =    (unsigned int*)cms::cuda::allocate_device(dev,totalSizes * sizeof(unsigned int),stream);
    modulesInGPU.connectedPixelsNeg =    (unsigned int*)cms::cuda::allocate_device(dev,totalSizes * sizeof(unsigned int),stream);
#else
    modulesInGPU.connectedPixels =       (unsigned int*)cms::cuda::allocate_managed(totalSizes * sizeof(unsigned int),stream);
    modulesInGPU.connectedPixelsPos =    (unsigned int*)cms::cuda::allocate_managed(totalSizes * sizeof(unsigned int),stream);
    modulesInGPU.connectedPixelsNeg =    (unsigned int*)cms::cuda::allocate_managed(totalSizes * sizeof(unsigned int),stream);
#endif
#else
#ifdef Explicit_Module
    cudaMalloc(&modulesInGPU.connectedPixels,totalSizes * sizeof(unsigned int));
    cudaMalloc(&modulesInGPU.connectedPixelsPos,totalSizes_pos * sizeof(unsigned int));
    cudaMalloc(&modulesInGPU.connectedPixelsNeg,totalSizes_neg * sizeof(unsigned int));
#else
    cudaMallocManaged(&modulesInGPU.connectedPixels,totalSizes * sizeof(unsigned int));
    cudaMallocManaged(&modulesInGPU.connectedPixelsPos,totalSizes_pos * sizeof(unsigned int));
    cudaMallocManaged(&modulesInGPU.connectedPixelsNeg,totalSizes_neg * sizeof(unsigned int));
#endif
#endif

    for(int icondet=0; icondet< totalSizes; icondet++){
      connectedPixels[icondet] = (*detIdToIndex)[connectedModuleDetIds[icondet]];
    }
    for(int icondet=0; icondet< totalSizes_pos; icondet++){
      connectedPixelsPos[icondet] = (*detIdToIndex)[connectedModuleDetIds_pos[icondet]];
    }
    for(int icondet=0; icondet< totalSizes_neg; icondet++){
      connectedPixelsNeg[icondet] = (*detIdToIndex)[connectedModuleDetIds_neg[icondet]];
    }
    cudaMemcpy(modulesInGPU.connectedPixels,connectedPixels,totalSizes*sizeof(unsigned int),cudaMemcpyHostToDevice);
    cudaMemcpy(modulesInGPU.connectedPixelsPos,connectedPixelsPos,totalSizes_pos*sizeof(unsigned int),cudaMemcpyHostToDevice);
    cudaMemcpy(modulesInGPU.connectedPixelsNeg,connectedPixelsNeg,totalSizes_neg*sizeof(unsigned int),cudaMemcpyHostToDevice);
    
    cudaFreeHost(connectedPixels);
    cudaFreeHost(connectedPixelsPos);
    cudaFreeHost(connectedPixelsNeg);
    //cudaFreeHost(connectedPixelsSizes);
    //cudaFreeHost(connectedPixelsSizesPos);
    //cudaFreeHost(connectedPixelsSizesNeg);
    //cudaFreeHost(connectedPixelsIndex);
    //cudaFreeHost(connectedPixelsIndexPos);
    //cudaFreeHost(connectedPixelsIndexNeg);
}

void SDL::fillConnectedModuleArrayExplicit(struct modules& modulesInGPU, unsigned int nModules)
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
    cudaMemcpy(modulesInGPU.moduleMap,moduleMap,nModules*40*sizeof(unsigned int),cudaMemcpyHostToDevice);
    cudaMemcpy(modulesInGPU.nConnectedModules,nConnectedModules,nModules*sizeof(unsigned int),cudaMemcpyHostToDevice);
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

unsigned int SDL::modules::partnerModuleIndexExplicit(unsigned int index, bool isLowerx, bool isInvertedx)
{
    /*We need to ensure modules with successive det Ids are right next to each other
    or we're dead*/
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
    /*We need to ensure modules with successive det Ids are right next to each other
    or we're dead*/
    if(isLower[index])
    {
        return (isInverted[index] ? index - 1: index + 1);
    }
    else
    {
        return (isInverted[index] ? index + 1 : index - 1);
    }
}

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

void SDL::resetObjectRanges(struct modules& modulesInGPU, unsigned int nModules)
{
#ifdef Explicit_Module
        cudaMemset(modulesInGPU.hitRanges, -1,nModules*2*sizeof(int));
        cudaMemset(modulesInGPU.mdRanges, -1,nModules*2*sizeof(int));
        cudaMemset(modulesInGPU.segmentRanges, -1,nModules*2*sizeof(int));
        cudaMemset(modulesInGPU.trackletRanges, -1,nModules*2*sizeof(int));
        cudaMemset(modulesInGPU.tripletRanges, -1,nModules*2*sizeof(int));
        cudaMemset(modulesInGPU.trackCandidateRanges, -1,nModules*2*sizeof(int));
#else

#pragma omp parallel for default(shared)
    for(size_t i = 0; i<nModules *2; i++)
    {
        modulesInGPU.hitRanges[i] = -1;
        modulesInGPU.mdRanges[i] = -1;
        modulesInGPU.segmentRanges[i] = -1;
        modulesInGPU.trackletRanges[i] = -1;
        modulesInGPU.tripletRanges[i] = -1;
        modulesInGPU.trackCandidateRanges[i] = -1;
    }
#endif
}
