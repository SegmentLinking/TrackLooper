#include "Module.cuh"
#include "ModuleConnectionMap.h"
#include "allocate.h"
#include <unordered_map>
std::map <unsigned int, uint16_t> *SDL::detIdToIndex;
std::map <unsigned int, float> *SDL::module_x;
std::map <unsigned int, float> *SDL::module_y;
std::map <unsigned int, float> *SDL::module_z;
std::map <unsigned int, unsigned int> *SDL::module_type; // 23 : Ph2PSP, 24 : Ph2PSS, 25 : Ph2SS
// https://github.com/cms-sw/cmssw/blob/5e809e8e0a625578aa265dc4b128a93830cb5429/Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h#L29

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
    rangesInGPU.quintupletModuleOccupancy = (int*)cms::cuda::allocate_device(dev,nLowerModules * sizeof(int),stream);
    rangesInGPU.miniDoubletModuleIndices = (int*)cms::cuda::allocate_device(dev, (nLowerModules + 1) * sizeof(int), stream);
    rangesInGPU.miniDoubletModuleOccupancy = (int*)cms::cuda::allocate_device(dev, (nLowerModules + 1) * sizeof(int), stream);
    rangesInGPU.segmentModuleIndices = (int*)cms::cuda::allocate_device(dev, (nLowerModules + 1) * sizeof(int), stream);
    rangesInGPU.segmentModuleOccupancy = (int*)cms::cuda::allocate_device(dev, (nLowerModules + 1) * sizeof(int), stream);
    rangesInGPU.tripletModuleIndices = (int*)cms::cuda::allocate_device(dev, nLowerModules * sizeof(int), stream);
    rangesInGPU.tripletModuleOccupancy = (int*)cms::cuda::allocate_device(dev, nLowerModules * sizeof(int), stream);

    rangesInGPU.device_nTotalMDs = (unsigned int*)cms::cuda::allocate_device(dev, sizeof(unsigned int), stream);
    rangesInGPU.device_nTotalSegs = (unsigned int*)cms::cuda::allocate_device(dev, sizeof(unsigned int), stream);
    rangesInGPU.device_nTotalTrips = (unsigned int*)cms::cuda::allocate_device(dev, sizeof(unsigned int), stream);
    rangesInGPU.device_nTotalQuints = (unsigned int*)cms::cuda::allocate_device(dev, sizeof(unsigned int), stream);

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
    cudaMalloc(&rangesInGPU.quintupletModuleOccupancy, nLowerModules * sizeof(int));

    cudaMalloc(&rangesInGPU.miniDoubletModuleIndices, (nLowerModules + 1) * sizeof(int));
    cudaMalloc(&rangesInGPU.miniDoubletModuleOccupancy, (nLowerModules + 1) * sizeof(int));
    cudaMalloc(&rangesInGPU.segmentModuleIndices, (nLowerModules + 1) * sizeof(int));
    cudaMalloc(&rangesInGPU.segmentModuleOccupancy, (nLowerModules + 1) * sizeof(int));
    cudaMalloc(&rangesInGPU.tripletModuleIndices, nLowerModules * sizeof(int));
    cudaMalloc(&rangesInGPU.tripletModuleOccupancy, nLowerModules * sizeof(int));
    
    cudaMalloc(&rangesInGPU.device_nTotalMDs, sizeof(unsigned int));
    cudaMalloc(&rangesInGPU.device_nTotalSegs, sizeof(unsigned int));
    cudaMalloc(&rangesInGPU.device_nTotalTrips, sizeof(unsigned int));
    cudaMalloc(&rangesInGPU.device_nTotalQuints, sizeof(unsigned int));

#endif
}

void SDL::createModulesInExplicitMemory(struct modules& modulesInGPU,unsigned int nModules,cudaStream_t stream)
{
    /* modules stucture object will be created in Event.cu*/
    cudaMalloc(&(modulesInGPU.detIds),nModules * sizeof(unsigned int));
    cudaMalloc(&modulesInGPU.moduleMap,nModules * 40 * sizeof(uint16_t));
    cudaMalloc(&modulesInGPU.mapIdx, nModules*sizeof(uint16_t));
    cudaMalloc(&modulesInGPU.mapdetId, nModules*sizeof(unsigned int));
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
    cudaMalloc(&modulesInGPU.sdlLayers, nModules * sizeof(int));

    cudaMemcpyAsync(modulesInGPU.nModules,&nModules,sizeof(uint16_t),cudaMemcpyHostToDevice,stream);
    cudaStreamSynchronize(stream);
}

void SDL::objectRanges::freeMemoryCache()//struct objectRanges& rangesInGPU)
{
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
  cms::cuda::free_device(dev, indicesOfEligibleT5Modules);
  cms::cuda::free_device(dev,quintupletModuleIndices);
  cms::cuda::free_device(dev,quintupletModuleOccupancy);
  cms::cuda::free_device(dev, hitRangesLower);
  cms::cuda::free_device(dev, hitRangesUpper);
  cms::cuda::free_device(dev, hitRangesnLower);
  cms::cuda::free_device(dev, hitRangesnUpper);
  cms::cuda::free_device(dev, miniDoubletModuleIndices);
  cms::cuda::free_device(dev, miniDoubletModuleOccupancy);
  cms::cuda::free_device(dev, segmentModuleIndices);
  cms::cuda::free_device(dev, segmentModuleOccupancy);
  cms::cuda::free_device(dev, tripletModuleIndices);
  cms::cuda::free_device(dev, tripletModuleOccupancy);
  cms::cuda::free_device(dev, device_nTotalMDs);
  cms::cuda::free_device(dev, device_nTotalSegs);
  cms::cuda::free_device(dev, device_nTotalTrips);
  cms::cuda::free_device(dev, device_nTotalQuints);
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
  cudaFree(indicesOfEligibleT5Modules);
  cudaFree(quintupletModuleIndices);
  cudaFree(quintupletModuleOccupancy);
  cudaFree(miniDoubletModuleIndices);
  cudaFree(miniDoubletModuleOccupancy);
  cudaFree(segmentModuleIndices);
  cudaFree(segmentModuleOccupancy);
  cudaFree(tripletModuleIndices);
  cudaFree(tripletModuleOccupancy);
  cudaFree(device_nTotalMDs);
  cudaFree(device_nTotalSegs);
  cudaFree(device_nTotalTrips);
  cudaFree(device_nTotalQuints);
}
void SDL::freeModulesCache(struct modules& modulesInGPU,struct pixelMap& pixelMapping)
{
  int dev;
  cudaGetDevice(&dev);
  cms::cuda::free_device(dev,modulesInGPU.detIds);
  cms::cuda::free_device(dev,modulesInGPU.moduleMap);
  cms::cuda::free_device(dev,modulesInGPU.mapIdx);
  cms::cuda::free_device(dev,modulesInGPU.mapdetId);
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
  cudaFreeHost(pixelMapping.connectedPixelsSizes);
  cudaFreeHost(pixelMapping.connectedPixelsSizesPos);
  cudaFreeHost(pixelMapping.connectedPixelsSizesNeg);
  cudaFreeHost(pixelMapping.connectedPixelsIndex);
  cudaFreeHost(pixelMapping.connectedPixelsIndexPos);
  cudaFreeHost(pixelMapping.connectedPixelsIndexNeg);
  //cms::cuda::free_host(pixelMapping.connectedPixelsSizes);
  //cms::cuda::free_host(pixelMapping.connectedPixelsSizesPos);
  //cms::cuda::free_host(pixelMapping.connectedPixelsSizesNeg);
  //cms::cuda::free_host(pixelMapping.connectedPixelsIndex);
  //cms::cuda::free_host(pixelMapping.connectedPixelsIndexPos);
  //cms::cuda::free_host(pixelMapping.connectedPixelsIndexNeg);
}
void SDL::freeModules(struct modules& modulesInGPU, struct pixelMap& pixelMapping)
{

  cudaFree(modulesInGPU.detIds);
  cudaFree(modulesInGPU.moduleMap);
  cudaFree(modulesInGPU.mapIdx);
  cudaFree(modulesInGPU.mapdetId);
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
  cudaFree(modulesInGPU.eta);
  cudaFree(modulesInGPU.r);
  cudaFree(modulesInGPU.isInverted);
  cudaFree(modulesInGPU.isLower);
  cudaFree(modulesInGPU.isAnchor);
  cudaFree(modulesInGPU.moduleType);
  cudaFree(modulesInGPU.moduleLayerType);
  cudaFree(modulesInGPU.connectedPixels);
  cudaFree(modulesInGPU.partnerModuleIndices);
  cudaFree(modulesInGPU.sdlLayers);

  cudaFreeHost(pixelMapping.connectedPixelsSizes);
  cudaFreeHost(pixelMapping.connectedPixelsSizesPos);
  cudaFreeHost(pixelMapping.connectedPixelsSizesNeg);
  cudaFreeHost(pixelMapping.connectedPixelsIndex);
  cudaFreeHost(pixelMapping.connectedPixelsIndexPos);
  cudaFreeHost(pixelMapping.connectedPixelsIndexNeg);
  //cms::cuda::free_host(pixelMapping.connectedPixelsSizes);
  //cms::cuda::free_host(pixelMapping.connectedPixelsSizesPos);
  //cms::cuda::free_host(pixelMapping.connectedPixelsSizesNeg);
  //cms::cuda::free_host(pixelMapping.connectedPixelsIndex);
  //cms::cuda::free_host(pixelMapping.connectedPixelsIndexPos);
  //cms::cuda::free_host(pixelMapping.connectedPixelsIndexNeg);
}

void SDL::loadModulesFromFile(struct modules& modulesInGPU, uint16_t& nModules, uint16_t& nLowerModules, struct pixelMap& pixelMapping,cudaStream_t stream, const char* moduleMetaDataFilePath)
{
    detIdToIndex = new std::map<unsigned int, uint16_t>;
    module_x = new std::map<unsigned int, float>;
    module_y = new std::map<unsigned int, float>;
    module_z = new std::map<unsigned int, float>;
    module_type = new std::map<unsigned int, unsigned int>;

    /*modules structure object will be created in Event.cu*/
    /* Load the whole text file into the map first*/

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
                (*module_z)[temp_detId] = std::stof(token);
            if(count_number == 4)
            {
                (*module_type)[temp_detId] = std::stoi(token);
                counter++;
            }
            count_number++;
            if(count_number>4)
                break;
        }

    }
    (*detIdToIndex)[1] = counter; //pixel module is the last module in the module list
    counter++;
    nModules = counter;
    //std::cout<<"Number of modules = "<<nModules<<std::endl;
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
    int* host_sdlLayers;

    host_detIds = (unsigned int*)cms::cuda::allocate_host(sizeof(unsigned int)*nModules, stream);
    host_layers = (short*)cms::cuda::allocate_host(sizeof(short)*nModules, stream);
    host_rings = (short*)cms::cuda::allocate_host(sizeof(short)*nModules, stream);
    host_rods = (short*)cms::cuda::allocate_host(sizeof(short)*nModules, stream);
    host_modules = (short*)cms::cuda::allocate_host(sizeof(short)*nModules, stream);
    host_subdets = (short*)cms::cuda::allocate_host(sizeof(short)*nModules, stream);
    host_sides = (short*)cms::cuda::allocate_host(sizeof(short)*nModules, stream);
    host_eta = (float*)cms::cuda::allocate_host(sizeof(float)*nModules, stream);
    host_r = (float*)cms::cuda::allocate_host(sizeof(float)*nModules, stream);
    host_isInverted = (bool*)cms::cuda::allocate_host(sizeof(bool)*nModules, stream);
    host_isLower = (bool*)cms::cuda::allocate_host(sizeof(bool)*nModules, stream);
    host_isAnchor = (bool*)cms::cuda::allocate_host(sizeof(bool)*nModules, stream);
    host_moduleType = (ModuleType*)cms::cuda::allocate_host(sizeof(ModuleType)*nModules, stream);
    host_moduleLayerType = (ModuleLayerType*)cms::cuda::allocate_host(sizeof(ModuleLayerType)*nModules, stream);
    host_slopes = (float*)cms::cuda::allocate_host(sizeof(float)*nModules, stream);
    host_drdzs = (float*)cms::cuda::allocate_host(sizeof(float)*nModules, stream);
    host_partnerModuleIndices = (uint16_t*)cms::cuda::allocate_host(sizeof(uint16_t) * nModules, stream);
    host_sdlLayers = (int*)cms::cuda::allocate_host(sizeof(int) * nModules, stream);
    
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
        unsigned int m_t = (*module_type)[detId];

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

            host_moduleType[index] = ( m_t == 25 ? SDL::TwoS : SDL::PS );
            host_moduleLayerType[index] = ( m_t == 23 ? SDL::Pixel : SDL::Strip );

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

        host_sdlLayers[index] = layer + 6 * (subdet == SDL::Endcap) + 5 * (subdet == SDL::Endcap and host_moduleType[index] == SDL::TwoS);
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
    cudaMemcpyAsync(modulesInGPU.sdlLayers, host_sdlLayers, sizeof(int) * nModules, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    cms::cuda::free_host(host_detIds);
    cms::cuda::free_host(host_layers);
    cms::cuda::free_host(host_rings);
    cms::cuda::free_host(host_rods);
    cms::cuda::free_host(host_modules);
    cms::cuda::free_host(host_subdets);
    cms::cuda::free_host(host_sides);
    cms::cuda::free_host(host_eta);
    cms::cuda::free_host(host_r);
    cms::cuda::free_host(host_isInverted);
    cms::cuda::free_host(host_isLower);
    cms::cuda::free_host(host_isAnchor);
    cms::cuda::free_host(host_moduleType);
    cms::cuda::free_host(host_moduleLayerType);
    cms::cuda::free_host(host_slopes);
    cms::cuda::free_host(host_drdzs);
    cms::cuda::free_host(host_partnerModuleIndices);
    cms::cuda::free_host(host_sdlLayers);
    fillConnectedModuleArrayExplicit(modulesInGPU,nModules,stream);
    fillMapArraysExplicit(modulesInGPU, nModules, stream);
    fillPixelMap(modulesInGPU,pixelMapping,stream);
}

void SDL::fillConnectedModuleArray(struct modules& modulesInGPU, unsigned int nModules)
{
    //uint16_t* moduleMap;
    //uint16_t* nConnectedModules;
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
    cudaMallocHost(&pixelMapping.connectedPixelsIndex,size_superbins * sizeof(unsigned int));
    cudaMallocHost(&pixelMapping.connectedPixelsSizes,size_superbins * sizeof(unsigned int));
    cudaMallocHost(&pixelMapping.connectedPixelsIndexPos,size_superbins * sizeof(unsigned int));
    cudaMallocHost(&pixelMapping.connectedPixelsSizesPos,size_superbins * sizeof(unsigned int));
    cudaMallocHost(&pixelMapping.connectedPixelsIndexNeg,size_superbins * sizeof(unsigned int));
    cudaMallocHost(&pixelMapping.connectedPixelsSizesNeg,size_superbins * sizeof(unsigned int));
    //pixelMapping.connectedPixelsIndex = (unsigned int*)cms::cuda::allocate_host(size_superbins * sizeof(unsigned int), stream);
    //pixelMapping.connectedPixelsSizes = (unsigned int*)cms::cuda::allocate_host(size_superbins * sizeof(unsigned int), stream);
    //pixelMapping.connectedPixelsIndexPos = (unsigned int*)cms::cuda::allocate_host(size_superbins * sizeof(unsigned int), stream);
    //pixelMapping.connectedPixelsSizesPos = (unsigned int*)cms::cuda::allocate_host(size_superbins * sizeof(unsigned int), stream);
    //pixelMapping.connectedPixelsIndexNeg = (unsigned int*)cms::cuda::allocate_host(size_superbins * sizeof(unsigned int), stream);
    //pixelMapping.connectedPixelsSizesNeg = (unsigned int*)cms::cuda::allocate_host(size_superbins * sizeof(unsigned int), stream);
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
    connectedPixels = (unsigned int*)cms::cuda::allocate_host((totalSizes+totalSizes_pos+totalSizes_neg) * sizeof(unsigned int), stream);
    cudaMalloc(&modulesInGPU.connectedPixels,(totalSizes+totalSizes_pos+totalSizes_neg)* sizeof(unsigned int));

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

    cms::cuda::free_host(connectedPixels);
}

void SDL::fillConnectedModuleArrayExplicit(struct modules& modulesInGPU, unsigned int nModules,cudaStream_t stream)
{
    uint16_t* moduleMap;
    uint16_t* nConnectedModules;
    moduleMap = (uint16_t*)cms::cuda::allocate_host(nModules * 40 * sizeof(uint16_t), stream);
    nConnectedModules = (uint16_t*)cms::cuda::allocate_host(nModules * sizeof(uint16_t), stream);
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
    cms::cuda::free_host(moduleMap);
    cms::cuda::free_host(nConnectedModules);
}

void SDL::fillMapArraysExplicit(struct modules& modulesInGPU, unsigned int nModules,cudaStream_t stream)
{
    uint16_t* mapIdx;
    unsigned int* mapdetId;
    unsigned int counter = 0;
    mapIdx = (uint16_t*)cms::cuda::allocate_host(nModules * sizeof(uint16_t), stream);
    mapdetId = (unsigned int*)cms::cuda::allocate_host(nModules * sizeof(unsigned int), stream);
    for(auto it = (*detIdToIndex).begin(); it != (*detIdToIndex).end(); ++it)
    {
        unsigned int detId = it->first;
        unsigned int index = it->second;
        mapIdx[counter] = index;
        mapdetId[counter] = detId;
        counter++;
    }
    cudaMemcpyAsync(modulesInGPU.mapIdx,mapIdx,nModules*sizeof(uint16_t),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(modulesInGPU.mapdetId,mapdetId,nModules*sizeof(unsigned int),cudaMemcpyHostToDevice,stream);
    cudaStreamSynchronize(stream);
    cms::cuda::free_host(mapIdx);
    cms::cuda::free_host(mapdetId);
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


