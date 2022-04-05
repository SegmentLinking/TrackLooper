# include "Module.cuh"
#include "ModuleConnectionMap.h"
#include "allocate.h"
std::map <unsigned int, uint16_t> *SDL::detIdToIndex;
std::map <unsigned int, float> *SDL::module_x;
std::map <unsigned int, float> *SDL::module_y;
std::map <unsigned int, float> *SDL::module_z;

void SDL::createRangesInUnifiedMemory(struct objectRanges& rangesInGPU,unsigned int nModules,cudaStream_t stream, unsigned int nLowerModules)
{
    /* modules stucture object will be created in Event.cu*/
#ifdef CACHE_ALLOC
    rangesInGPU.hitRanges =                 (int*)cms::cuda::allocate_managed(nModules * 2 * sizeof(int),stream);
    rangesInGPU.hitRangesLower =                 (int*)cms::cuda::allocate_managed(nModules * sizeof(int),stream);
    rangesInGPU.hitRangesUpper =                 (int*)cms::cuda::allocate_managed(nModules * sizeof(int),stream);
    rangesInGPU.hitRangesnLower =                 (int8_t*)cms::cuda::allocate_managed(nModules * sizeof(int8_t),stream);
    rangesInGPU.hitRangesnUpper =                 (int8_t*)cms::cuda::allocate_managed(nModules * sizeof(int8_t),stream);
    rangesInGPU.mdRanges =                  (int*)cms::cuda::allocate_managed(nModules * 2 * sizeof(int),stream);
    rangesInGPU.segmentRanges =             (int*)cms::cuda::allocate_managed(nModules * 2 * sizeof(int),stream);
    rangesInGPU.trackletRanges =            (int*)cms::cuda::allocate_managed(nModules * 2 * sizeof(int),stream);
    rangesInGPU.tripletRanges =             (int*)cms::cuda::allocate_managed(nModules * 2 * sizeof(int),stream);
    rangesInGPU.trackCandidateRanges =      (int*)cms::cuda::allocate_managed(nModules * 2 * sizeof(int),stream);
    rangesInGPU.quintupletRanges =          (int*)cms::cuda::allocate_managed(nModules * 2 * sizeof(int),stream);
    rangesInGPU.nEligibleT5Modules =        (uint16_t*)cms::cuda::allocate_managed(sizeof(unsigned int),stream);

    rangesInGPU.quintupletModuleIndices = (int*)cms::cuda::allocate_managed(nLowerModules * sizeof(int),stream);
    rangesInGPU.miniDoubletModuleIndices = (int*)cms::cuda::allocate_managed((nLowerModules + 1) * sizeof(int), stream);
    rangesInGPU.segmentModuleIndices = (int*)cms::cuda::allocate_managed((nLowerModules + 1) * sizeof(int), stream);
    rangesInGPU.tripletModuleIndices = (int*)cms::cuda::allocate_managed(nLowerModules * sizeof(int), stream);

#else
    cudaMallocManaged(&rangesInGPU.hitRanges,nModules * 2 * sizeof(int));
    cudaMallocManaged(&rangesInGPU.hitRangesLower,nModules  * sizeof(int));
    cudaMallocManaged(&rangesInGPU.hitRangesUpper,nModules  * sizeof(int));
    cudaMallocManaged(&rangesInGPU.hitRangesnLower,nModules  * sizeof(int8_t));
    cudaMallocManaged(&rangesInGPU.hitRangesnUpper,nModules  * sizeof(int8_t));
    cudaMallocManaged(&rangesInGPU.mdRanges,nModules * 2 * sizeof(int));
    cudaMallocManaged(&rangesInGPU.segmentRanges,nModules * 2 * sizeof(int));
    cudaMallocManaged(&rangesInGPU.trackletRanges,nModules * 2 * sizeof(int));
    cudaMallocManaged(&rangesInGPU.tripletRanges,nModules * 2 * sizeof(int));
    cudaMallocManaged(&rangesInGPU.trackCandidateRanges, nModules * 2 * sizeof(int));
    cudaMallocManaged(&rangesInGPU.quintupletRanges, nModules * 2 * sizeof(int));
    cudaMallocManaged(&rangesInGPU.nEligibleT5Modules, sizeof(uint16_t));

    cudaMallocManaged(&rangesInGPU.quintupletModuleIndices, nLowerModules * sizeof(int));
    cudaMallocManaged(&rangesInGPU.miniDoubletModuleIndices, (nLowerModules + 1) * sizeof(int));
    cudaMallocManaged(&rangesInGPU.segmentModuleIndices, (nLowerModules + 1) * sizeof(int));
    cudaMallocManaged(&rangesInGPU.tripletModuleIndices, nLowerModules * sizeof(int));

#endif
}
void SDL::createRangesInExplicitMemory(struct objectRanges& rangesInGPU,unsigned int nModules,cudaStream_t stream, unsigned int nLowerModules)
{
    /* modules stucture object will be created in Event.cu*/
#ifdef CACHE_ALLOC
    int dev;
    cudaGetDevice(&dev);
    rangesInGPU.hitRanges =                  (int*)cms::cuda::allocate_device(dev,nModules * 2 * sizeof(int),stream);
    rangesInGPU.hitRangesLower =                  (int*)cms::cuda::allocate_device(dev,nModules * sizeof(int),stream);
    rangesInGPU.hitRangesUpper =                  (int*)cms::cuda::allocate_device(dev,nModules * sizeof(int),stream);
    rangesInGPU.hitRangesnLower =                  (int8_t*)cms::cuda::allocate_device(dev,nModules * sizeof(int8_t),stream);
    rangesInGPU.hitRangesnUpper =                  (int8_t*)cms::cuda::allocate_device(dev,nModules * sizeof(int8_t),stream);
    rangesInGPU.mdRanges =                   (int*)cms::cuda::allocate_device(dev,nModules * 2 * sizeof(int),stream);
    rangesInGPU.segmentRanges =              (int*)cms::cuda::allocate_device(dev,nModules * 2 * sizeof(int),stream);
    rangesInGPU.trackletRanges =             (int*)cms::cuda::allocate_device(dev,nModules * 2 * sizeof(int),stream);
    rangesInGPU.tripletRanges =              (int*)cms::cuda::allocate_device(dev,nModules * 2 * sizeof(int),stream);
    rangesInGPU.trackCandidateRanges =       (int*)cms::cuda::allocate_device(dev,nModules * 2 * sizeof(int),stream);
    rangesInGPU.quintupletRanges =       (int*)cms::cuda::allocate_device(dev,nModules * 2 * sizeof(int),stream);
    rangesInGPU.nEligibleT5Modules =    (uint16_t*)cms::cuda::allocate_device(dev,sizeof(unsigned int),stream);

    rangesInGPU.quintupletModuleIndices = (int*)cms::cuda::allocate_device(dev,nLowerModules * sizeof(int),stream);
    rangesInGPU.miniDoubletModuleIndices = (int*)cms::cuda::allocate_device(dev, (nLowerModules + 1) * sizeof(int), stream);
    rangesInGPU.segmentModuleIndices = (int*)cms::cuda::allocate_device(dev, (nLowerModules + 1) * sizeof(int), stream);
    rangesInGPU.tripletModuleIndices = (int*)cms::cuda::allocate_device(dev, nLowerModules * sizeof(int), stream);

#else
    cudaMalloc(&rangesInGPU.hitRanges,nModules * 2 * sizeof(int));
    cudaMalloc(&rangesInGPU.hitRangesLower,nModules  * sizeof(int));
    cudaMalloc(&rangesInGPU.hitRangesUpper,nModules  * sizeof(int));
    cudaMalloc(&rangesInGPU.hitRangesnLower,nModules  * sizeof(int8_t));
    cudaMalloc(&rangesInGPU.hitRangesnUpper,nModules  * sizeof(int8_t));
    cudaMalloc(&rangesInGPU.mdRanges,nModules * 2 * sizeof(int));
    cudaMalloc(&rangesInGPU.segmentRanges,nModules * 2 * sizeof(int));
    cudaMalloc(&rangesInGPU.trackletRanges,nModules * 2 * sizeof(int));
    cudaMalloc(&rangesInGPU.tripletRanges,nModules * 2 * sizeof(int));
    cudaMalloc(&rangesInGPU.trackCandidateRanges, nModules * 2 * sizeof(int));
    cudaMalloc(&rangesInGPU.quintupletRanges, nModules * 2 * sizeof(int));
    cudaMalloc(&rangesInGPU.nEligibleT5Modules, sizeof(uint16_t));
    cudaMalloc(&rangesInGPU.quintupletModuleIndices, nLowerModules * sizeof(int));

    cudaMalloc(&rangesInGPU.miniDoubletModuleIndices, (nLowerModules + 1) * sizeof(int));
    cudaMalloc(&rangesInGPU.segmentModuleIndices, (nLowerModules + 1) * sizeof(int));
    cudaMalloc(&rangesInGPU.tripletModuleIndices, nLowerModules * sizeof(int));

#endif
}
void SDL::createModulesInUnifiedMemory(struct modules& modulesInGPU,unsigned int nModules,cudaStream_t stream)
{
    cudaMallocManaged(&modulesInGPU.detIds,nModules * sizeof(unsigned int));
    cudaMallocManaged(&modulesInGPU.moduleMap,nModules * 40 * sizeof(uint16_t));
    cudaMallocManaged(&modulesInGPU.nConnectedModules,nModules * sizeof(uint16_t));
    cudaMallocManaged(&modulesInGPU.drdzs,nModules * sizeof(float));
    cudaMallocManaged(&modulesInGPU.slopes,nModules * sizeof(float));
    cudaMallocManaged(&modulesInGPU.nModules,sizeof(uint16_t));
    cudaMallocManaged(&modulesInGPU.nLowerModules,sizeof(uint16_t));
    cudaMallocManaged(&modulesInGPU.partnerModuleIndices, nModules * sizeof(uint16_t));

    cudaMallocManaged(&modulesInGPU.layers,nModules * sizeof(short));
    cudaMallocManaged(&modulesInGPU.rings,nModules * sizeof(short));
    cudaMallocManaged(&modulesInGPU.modules,nModules * sizeof(short));
    cudaMallocManaged(&modulesInGPU.rods,nModules * sizeof(short));
    cudaMallocManaged(&modulesInGPU.subdets,nModules * sizeof(short));
    cudaMallocManaged(&modulesInGPU.sides,nModules * sizeof(short));
    cudaMallocManaged(&modulesInGPU.eta,nModules * sizeof(float));
    cudaMallocManaged(&modulesInGPU.r,nModules * sizeof(float));
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
    cudaMalloc(&modulesInGPU.moduleMap,nModules * 40 * sizeof(uint16_t));
    cudaMalloc(&modulesInGPU.nConnectedModules,nModules * sizeof(uint16_t));
    cudaMalloc(&modulesInGPU.drdzs,nModules * sizeof(float));
    cudaMalloc(&modulesInGPU.slopes,nModules * sizeof(float));
    cudaMalloc(&modulesInGPU.nModules,sizeof(uint16_t));
    cudaMalloc(&modulesInGPU.nLowerModules,sizeof(uint16_t));
    cudaMalloc(&modulesInGPU.partnerModuleIndices, nModules * sizeof(uint16_t));

    cudaMalloc(&modulesInGPU.layers,nModules * sizeof(short));
    cudaMalloc(&modulesInGPU.rings,nModules * sizeof(short));
    cudaMalloc(&modulesInGPU.modules,nModules * sizeof(short));
    cudaMalloc(&modulesInGPU.rods,nModules * sizeof(short));
    cudaMalloc(&modulesInGPU.subdets,nModules * sizeof(short));
    cudaMalloc(&modulesInGPU.sides,nModules * sizeof(short));
    cudaMalloc(&modulesInGPU.eta,nModules * sizeof(float));
    cudaMalloc(&modulesInGPU.r,nModules * sizeof(float));
    cudaMalloc(&modulesInGPU.isInverted, nModules * sizeof(bool));
    cudaMalloc(&modulesInGPU.isLower, nModules * sizeof(bool));
    cudaMalloc(&modulesInGPU.isAnchor, nModules * sizeof(bool));
    cudaMalloc(&modulesInGPU.moduleType,nModules * sizeof(ModuleType));
    cudaMalloc(&modulesInGPU.moduleLayerType,nModules * sizeof(ModuleLayerType));

    cudaMemcpyAsync(modulesInGPU.nModules,&nModules,sizeof(uint16_t),cudaMemcpyHostToDevice,stream);
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
  cms::cuda::free_device(dev, hitRangesLower);
  cms::cuda::free_device(dev, hitRangesUpper);
  cms::cuda::free_device(dev, hitRangesnLower);
  cms::cuda::free_device(dev, hitRangesnUpper);
  cms::cuda::free_device(dev, miniDoubletModuleIndices);
  cms::cuda::free_device(dev, segmentModuleIndices);
  cms::cuda::free_device(dev, tripletModuleIndices);
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
  cms::cuda::free_managed(hitRangesLower);
  cms::cuda::free_managed(hitRangesUpper);
  cms::cuda::free_managed(hitRangesnLower);
  cms::cuda::free_managed(hitRangesnUpper);
  cms::cuda::free_managed(miniDoubletModuleIndices);
  cms::cuda::free_managed(segmentModuleIndices);
  cms::cuda::free_managed(tripletModuleIndices);

#endif
}
void SDL::objectRanges::freeMemory()
{
  cudaFree(hitRanges);
  cudaFree(hitRangesLower);
  cudaFree(hitRangesUpper);
  cudaFree(hitRangesnLower);
  cudaFree(hitRangesnUpper);
  cudaFree(mdRanges);
  cudaFree(segmentRanges);
  cudaFree(trackletRanges);
  cudaFree(tripletRanges);
  cudaFree(trackCandidateRanges);
  cudaFree(quintupletRanges);
  cudaFree(nEligibleT5Modules);
  cudaFree(quintupletModuleIndices);
  cudaFree(miniDoubletModuleIndices);
  cudaFree(segmentModuleIndices);
  cudaFree(tripletModuleIndices);
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
  cudaFree(modulesInGPU.connectedPixels);
  cudaFree(modulesInGPU.partnerModuleIndices);

  cudaFreeHost(pixelMapping.connectedPixelsSizes);
  cudaFreeHost(pixelMapping.connectedPixelsSizesPos);
  cudaFreeHost(pixelMapping.connectedPixelsSizesNeg);
  cudaFreeHost(pixelMapping.connectedPixelsIndex);
  cudaFreeHost(pixelMapping.connectedPixelsIndexPos);
  cudaFreeHost(pixelMapping.connectedPixelsIndexNeg);
}

void SDL::loadModulesFromFile(struct modules& modulesInGPU, uint16_t& nModules, uint16_t& nLowerModules, struct pixelMap& pixelMapping,cudaStream_t stream, const char* moduleMetaDataFilePath)
{
    detIdToIndex = new std::map<unsigned int, uint16_t>;
    module_x = new std::map<unsigned int, float>;
    module_y = new std::map<unsigned int, float>;
    module_z = new std::map<unsigned int, float>;

    /*modules structure object will be created in Event.cu*/
    /* Load the whole text file into the unordered_map first*/

    std::ifstream ifile;
    ifile.open(moduleMetaDataFilePath);
    if(!ifile.is_open())
    {
        std::cout<<"ERROR! module list file not present!"<<std::endl;
    }
    std::string line;
    uint16_t counter = 0;

    while(std::getline(ifile,line))
    {
        std::stringstream ss(line);
        std::string token;
        int count_number = 0;

        unsigned int temp_detId;
        while(std::getline(ss,token,','))
        {
            if(count_number == 0)
            {
                temp_detId = stoi(token);
                (*detIdToIndex)[temp_detId] = counter;
            }
            if(count_number == 1)
                (*module_x)[temp_detId] = std::stof(token);
            if(count_number == 2)
                (*module_y)[temp_detId] = std::stof(token);
            if(count_number == 3)
            {
                (*module_z)[temp_detId] = std::stof(token);
                counter++;
            }
            count_number++;
            if(count_number>3)
                break;
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
    float* host_eta;
    float* host_r;
    bool* host_isInverted;
    bool* host_isLower;
    bool* host_isAnchor;
    ModuleType* host_moduleType;
    ModuleLayerType* host_moduleLayerType;
    float* host_slopes;
    float* host_drdzs;
    uint16_t* host_partnerModuleIndices;

    cudaMallocHost(&host_detIds,sizeof(unsigned int)*nModules);
    cudaMallocHost(&host_layers,sizeof(short)*nModules);
    cudaMallocHost(&host_rings,sizeof(short)*nModules);
    cudaMallocHost(&host_rods,sizeof(short)*nModules);
    cudaMallocHost(&host_modules,sizeof(short)*nModules);
    cudaMallocHost(&host_subdets,sizeof(short)*nModules);
    cudaMallocHost(&host_sides,sizeof(short)*nModules);
    cudaMallocHost(&host_eta,sizeof(float)*nModules);
    cudaMallocHost(&host_r,sizeof(float)*nModules);
    cudaMallocHost(&host_isInverted,sizeof(bool)*nModules);
    cudaMallocHost(&host_isLower,sizeof(bool)*nModules);
    cudaMallocHost(&host_isAnchor, sizeof(bool) * nModules);
    cudaMallocHost(&host_moduleType,sizeof(ModuleType)*nModules);
    cudaMallocHost(&host_moduleLayerType,sizeof(ModuleLayerType)*nModules);
    cudaMallocHost(&host_slopes,sizeof(float)*nModules);
    cudaMallocHost(&host_drdzs,sizeof(float)*nModules);
    cudaMallocHost(&host_partnerModuleIndices, sizeof(uint16_t) * nModules);
    
    //reassign detIdToIndex indices here
    nLowerModules = (nModules - 1) / 2;
    uint16_t lowerModuleCounter = 0;
    uint16_t upperModuleCounter = nLowerModules + 1;
    //0 to nLowerModules - 1 => only lower modules, nLowerModules - pixel module, nLowerModules + 1 to nModules => upper modules
    for(auto it = (*detIdToIndex).begin(); it != (*detIdToIndex).end(); it++)
    {
        unsigned int detId = it->first;
        float m_x = (*module_x)[detId];
        float m_y = (*module_y)[detId];
        float m_z = (*module_z)[detId];

        float eta,r;

        uint16_t index;
        unsigned short layer,ring,rod,module,subdet,side;
        bool isInverted, isLower;
        if(detId == 1)
        {
            layer = 0;
            ring = 0;
            rod = 0;
            module = 0;
            subdet = 0;
            side = 0;
            isInverted = false;
            isLower = false;
        }
        else
        {
            setDerivedQuantities(detId,layer,ring,rod,module,subdet,side,m_x,m_y,m_z,eta,r);
            isInverted = modulesInGPU.parseIsInverted(subdet, side, module, layer);
            isLower = modulesInGPU.parseIsLower(isInverted, detId);
        }
        if(isLower)
        {
            index = lowerModuleCounter;
            lowerModuleCounter++;
        }
        else if(detId != 1)
        {
            index = upperModuleCounter;
            upperModuleCounter++;
        }
        else
        {
            index = nLowerModules; //pixel
        }
        //reassigning indices!
        (*detIdToIndex)[detId] = index;   
        host_detIds[index] = detId;
        host_layers[index] = layer;
        host_rings[index] = ring;
        host_rods[index] = rod;
        host_modules[index] = module;
        host_subdets[index] = subdet;
        host_sides[index] = side;
        host_eta[index] = eta;
        host_r[index] = r;
        host_isInverted[index] = isInverted;
        host_isLower[index] = isLower;

        //assigning other variables!
        if(detId == 1)
        {
            host_moduleType[index] = PixelModule;
            host_moduleLayerType[index] = SDL::InnerPixelLayer;
            host_slopes[index] = 0;
            host_drdzs[index] = 0;
            host_isAnchor[index] = false;
        }
        else
        {

            host_moduleType[index] = modulesInGPU.parseModuleType(subdet, layer, ring);
            host_moduleLayerType[index] = modulesInGPU.parseModuleLayerType(host_moduleType[index],host_isInverted[index],host_isLower[index]);

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
    }

    //partner module stuff, and slopes and drdz move around
    for(auto it = (*detIdToIndex).begin(); it != (*detIdToIndex).end(); it++)
    {
        auto& detId = it->first;
        auto& index = it->second;
        if(detId != 1)
        {
            host_partnerModuleIndices[index] = (*detIdToIndex)[modulesInGPU.parsePartnerModuleId(detId, host_isLower[index], host_isInverted[index])];
            //add drdz and slope importing stuff here!
            if(host_drdzs[index] == 0)
            {
                host_drdzs[index] = host_drdzs[host_partnerModuleIndices[index]];
            }
            if(host_slopes[index] == 0)
            {
                host_slopes[index] = host_slopes[host_partnerModuleIndices[index]];
            }
        }
    }

    cudaMemcpyAsync(modulesInGPU.nLowerModules,&nLowerModules,sizeof(uint16_t),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(modulesInGPU.detIds,host_detIds,nModules*sizeof(unsigned int),cudaMemcpyHostToDevice,stream);

    cudaMemcpyAsync(modulesInGPU.layers,host_layers,nModules*sizeof(short),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(modulesInGPU.rings,host_rings,sizeof(short)*nModules,cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(modulesInGPU.rods,host_rods,sizeof(short)*nModules,cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(modulesInGPU.modules,host_modules,sizeof(short)*nModules,cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(modulesInGPU.subdets,host_subdets,sizeof(short)*nModules,cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(modulesInGPU.sides,host_sides,sizeof(short)*nModules,cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(modulesInGPU.eta,host_eta,sizeof(float)*nModules,cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(modulesInGPU.r,host_r,sizeof(float)*nModules,cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(modulesInGPU.isInverted,host_isInverted,sizeof(bool)*nModules,cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(modulesInGPU.isLower,host_isLower,sizeof(bool)*nModules,cudaMemcpyHostToDevice,stream);

    cudaMemcpyAsync(modulesInGPU.moduleType,host_moduleType,sizeof(ModuleType)*nModules,cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(modulesInGPU.moduleLayerType,host_moduleLayerType,sizeof(ModuleLayerType)*nModules,cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(modulesInGPU.slopes,host_slopes,sizeof(float)*nModules,cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(modulesInGPU.isAnchor, host_isAnchor, sizeof(bool) * nModules, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(modulesInGPU.drdzs,host_drdzs,sizeof(float)*nModules,cudaMemcpyHostToDevice,stream);

    cudaMemcpyAsync(modulesInGPU.partnerModuleIndices, host_partnerModuleIndices, sizeof(uint16_t) * nModules, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    cudaFreeHost(host_detIds);
    cudaFreeHost(host_layers);
    cudaFreeHost(host_rings);
    cudaFreeHost(host_rods);
    cudaFreeHost(host_modules);
    cudaFreeHost(host_subdets);
    cudaFreeHost(host_sides);
    cudaFreeHost(host_eta);
    cudaFreeHost(host_r);
    cudaFreeHost(host_isInverted);
    cudaFreeHost(host_isLower);
    cudaFreeHost(host_isAnchor);
    cudaFreeHost(host_moduleType);
    cudaFreeHost(host_moduleLayerType);
    cudaFreeHost(host_slopes);
    cudaFreeHost(host_drdzs);
    cudaFreeHost(host_partnerModuleIndices);
    std::cout<<"number of lower modules (without fake pixel module)= "<<lowerModuleCounter<<std::endl;
    fillConnectedModuleArrayExplicit(modulesInGPU,nModules,stream);
    fillPixelMap(modulesInGPU,pixelMapping,stream);

#else
    createModulesInUnifiedMemory(modulesInGPU,nModules,stream);
    nLowerModules = (nModules - 1) / 2;
    unsigned int lowerModuleCounter = 0;
    unsigned int upperModuleCounter = nLowerModules + 1;
    for(auto it = (*detIdToIndex).begin(); it != (*detIdToIndex).end(); it++)
    {
        unsigned int detId = it->first;
        uint16_t index;
        unsigned short layer,ring,rod,module,subdet,side;
        bool isInverted, isLower;
        if(detId == 1)
        {
            layer = 0;
            ring = 0;
            rod = 0;
            module = 0;
            subdet = 0;
            side = 0;
            isInverted = false;
            isLower = false;

        }
        else
        {
            setDerivedQuantities(detId,layer,ring,rod,module,subdet,side,m_x,m_y,m_z,eta,r);
            isInverted = modulesInGPU.parseIsInverted(subdet, side, module, layer);
            isLower = modulesInGPU.parseIsLower(isInverted, detId);
        }
        if(isLower)
        {
            index = lowerModuleCounter;
            lowerModuleCounter++;
        }
        else if(detId != 1)
        {
            index = upperModuleCounter;
            upperModuleCounter++;
        }
        else
        {
            index = nLowerModules; //pixel
        }
        //reassigning indices!
        (*detIdToIndex)[detId] = index;   
        modulesInGPU.detIds[index] = detId;
        modulesInGPU.layers[index] = layer;
        modulesInGPU.rings[index] = ring;
        modulesInGPU.rods[index] = rod;
        modulesInGPU.modules[index] = module;
        modulesInGPU.subdets[index] = subdet;
        modulesInGPU.sides[index] = side;
        modulesInGPU.isInverted[index] = isInverted;
        modulesInGPU.isLower[index] = isLower;

        if(detId == 1)
        {
            modulesInGPU.moduleType[index] = PixelModule;
            modulesInGPU.moduleLayerType[index] = SDL::InnerPixelLayer;
            modulesInGPU.slopes[index] = 0;
            modulesInGPU.drdzs[index] = 0;
            modulesInGPU.isAnchor[index] = false;
        }
        else
        {

            modulesInGPU.moduleType[index] = modulesInGPU.parseModuleType(subdet, layer, ring);
            modulesInGPU.moduleLayerType[index] = modulesInGPU.parseModuleLayerType(modulesInGPU.moduleType[index],modulesInGPU.isInverted[index],modulesInGPU.isLower[index]);

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
    }


    //partner module stuff, and slopes and drdz move around
    for(auto it = (*detIdToIndex).begin(); it != (*detIdToIndex).end(); it++)
    {
        auto& detId = it->first;
        auto& index = it->second;
        if(detId != 1)
        {
            modulesInGPU.partnerModuleIndices[index] = (*detIdToIndex)[modulesInGPU.parsePartnerModuleId(detId, modulesInGPU.isLower[index], modulesInGPU.isInverted[index])];
            //add drdz and slope importing stuff here!
            if(modulesInGPU.drdzs[index] == 0)
            {
                modulesInGPU.drdzs[index] = modulesInGPU.drdzs[modulesInGPU.partnerModuleIndices[index]];
            }
            if(modulesInGPU.slopes[index] == 0)
            {
                modulesInGPU.slopes[index] = modulesInGPU.slopes[modulesInGPU.partnerModuleIndices[index]];
            }
        }
    }

    *(modulesInGPU.nLowerModules) = nLowerModules;
    std::cout<<"number of lower modules (without fake pixel module)= "<<*modulesInGPU.nLowerModules<<std::endl;
    fillConnectedModuleArray(modulesInGPU,nModules);
    fillPixelMap(modulesInGPU,pixelMapping,stream);
    #endif
}

void SDL::fillConnectedModuleArray(struct modules& modulesInGPU, unsigned int nModules)
{
    uint16_t* moduleMap;
    uint16_t* nConnectedModules;
    for(auto it = (*detIdToIndex).begin(); it != (*detIdToIndex).end(); ++it)
    {
        unsigned int detId = it->first;
        uint16_t index = it->second;
        auto& connectedModules = moduleConnectionMap.getConnectedModuleDetIds(detId);
        modulesInGPU.nConnectedModules[index] = connectedModules.size();
        for(uint16_t i = 0; i< modulesInGPU.nConnectedModules[index];i++)
        {
            modulesInGPU.moduleMap[index * 40 + i] = (*detIdToIndex)[connectedModules[i]];
        }
    }
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
#ifdef Explicit_Module
    cudaMalloc(&modulesInGPU.connectedPixels,(totalSizes+totalSizes_pos+totalSizes_neg)* sizeof(unsigned int));
#else
    cudaMallocManaged(&modulesInGPU.connectedPixels,(totalSizes+totalSizes_pos+totalSizes_neg)* sizeof(unsigned int));
#endif

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
    uint16_t* moduleMap;
    uint16_t* nConnectedModules;
    cudaMallocHost(&moduleMap,nModules * 40 * sizeof(uint16_t));
    cudaMallocHost(&nConnectedModules,nModules * sizeof(uint16_t));
    for(auto it = (*detIdToIndex).begin(); it != (*detIdToIndex).end(); ++it)
    {
        unsigned int detId = it->first;
        uint16_t index = it->second;
        auto& connectedModules = moduleConnectionMap.getConnectedModuleDetIds(detId);
        nConnectedModules[index] = connectedModules.size();
        for(uint16_t i = 0; i< nConnectedModules[index];i++)
        {
            moduleMap[index * 40 + i] = (*detIdToIndex)[connectedModules[i]];
        }
    }
    cudaMemcpyAsync(modulesInGPU.moduleMap,moduleMap,nModules*40*sizeof(uint16_t),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(modulesInGPU.nConnectedModules,nConnectedModules,nModules*sizeof(uint16_t),cudaMemcpyHostToDevice,stream);
    cudaStreamSynchronize(stream);
    cudaFreeHost(moduleMap);
    cudaFreeHost(nConnectedModules);
}

void SDL::setDerivedQuantities(unsigned int detId, unsigned short& layer, unsigned short& ring, unsigned short& rod, unsigned short& module, unsigned short& subdet, unsigned short& side, float m_x, float m_y, float m_z, float& eta, float& r)
{
    subdet = (detId & (7 << 25)) >> 25;
    side = (subdet == Endcap) ? (detId & (3 << 23)) >> 23 : (detId & (3 << 18)) >> 18;
    layer = (subdet == Endcap) ? (detId & (7 << 18)) >> 18 : (detId & (7 << 20)) >> 20;
    ring = (subdet == Endcap) ? (detId & (15 << 12)) >> 12 : 0;
    module = (detId & (127 << 2)) >> 2;
    rod = (subdet == Endcap) ? 0 : (detId & (127 << 10)) >> 10;

    r = std::sqrt(m_x * m_x + m_y * m_y + m_z * m_z);
    eta = ((m_z > 0) - ( m_z < 0)) * std::acosh(r / std::sqrt(m_x * m_x + m_y * m_y));
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
bool SDL::modules::parseIsInverted(short subdet, short side, short module, short layer)
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

bool SDL::modules::parseIsLower(bool isInvertedx, unsigned int detId)
{
    return (isInvertedx) ? !(detId & 1) : (detId & 1);
}
bool SDL::modules::parseIsLower(unsigned int index)
{
    return (isInverted[index]) ? !(detIds[index] & 1) : (detIds[index] & 1);
}


unsigned int SDL::modules::parsePartnerModuleId(unsigned int detId, bool isLowerx, bool isInvertedx)
{
    return isLowerx ? (isInvertedx ? detId - 1 : detId + 1) : (isInvertedx ? detId + 1 : detId - 1);
}

SDL::ModuleType SDL::modules::parseModuleType(short subdet, short layer, short ring)
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

SDL::ModuleLayerType SDL::modules::parseModuleLayerType(ModuleType moduleTypex,bool isInvertedx, bool isLowerx)
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
        cudaMemsetAsync(rangesInGPU.hitRangesLower, -1,nModules*sizeof(int),stream);
        cudaMemsetAsync(rangesInGPU.hitRangesUpper, -1,nModules*sizeof(int),stream);
        cudaMemsetAsync(rangesInGPU.hitRangesnLower, -1,nModules*sizeof(int8_t),stream);
        cudaMemsetAsync(rangesInGPU.hitRangesnUpper, -1,nModules*sizeof(int8_t),stream);
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
        unsigned int partnerModuleIdx = i;
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
            staggeredNeighbours[counter] = i;//deal in lower module indices
            counter++;
        }
    }
}
