#include "Module.cuh"

std::map <unsigned int, uint16_t> *SDL::detIdToIndex;
std::map <unsigned int, float> *SDL::module_x;
std::map <unsigned int, float> *SDL::module_y;
std::map <unsigned int, float> *SDL::module_z;
std::map <unsigned int, unsigned int> *SDL::module_type; // 23 : Ph2PSP, 24 : Ph2PSS, 25 : Ph2SS
// https://github.com/cms-sw/cmssw/blob/5e809e8e0a625578aa265dc4b128a93830cb5429/Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h#L29

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

void SDL::freeModules(struct modules& modulesInGPU)
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

    auto detIds_buf = allocBufWrapper<unsigned int>(devHost, nModules);
    auto layers_buf = allocBufWrapper<short>(devHost, nModules);
    auto rings_buf = allocBufWrapper<short>(devHost, nModules);
    auto rods_buf = allocBufWrapper<short>(devHost, nModules);
    auto modules_buf = allocBufWrapper<short>(devHost, nModules);
    auto subdets_buf = allocBufWrapper<short>(devHost, nModules);
    auto sides_buf = allocBufWrapper<short>(devHost, nModules);
    auto eta_buf = allocBufWrapper<float>(devHost, nModules);
    auto r_buf = allocBufWrapper<float>(devHost, nModules);
    auto isInverted_buf = allocBufWrapper<bool>(devHost, nModules);
    auto isLower_buf = allocBufWrapper<bool>(devHost, nModules);
    auto isAnchor_buf = allocBufWrapper<bool>(devHost, nModules);
    auto moduleType_buf = allocBufWrapper<ModuleType>(devHost, nModules);
    auto moduleLayerType_buf = allocBufWrapper<ModuleLayerType>(devHost, nModules);
    auto slopes_buf = allocBufWrapper<float>(devHost, nModules);
    auto drdzs_buf = allocBufWrapper<float>(devHost, nModules);
    auto partnerModuleIndices_buf = allocBufWrapper<uint16_t>(devHost, nModules);
    auto host_sdlLayers_buf = allocBufWrapper<int>(devHost, nModules);

    // Getting the underlying data pointers
    unsigned int* host_detIds = alpaka::getPtrNative(detIds_buf);
    short* host_layers = alpaka::getPtrNative(layers_buf);
    short* host_rings = alpaka::getPtrNative(rings_buf);
    short* host_rods = alpaka::getPtrNative(rods_buf);
    short* host_modules = alpaka::getPtrNative(modules_buf);
    short* host_subdets = alpaka::getPtrNative(subdets_buf);
    short* host_sides = alpaka::getPtrNative(sides_buf);
    float* host_eta = alpaka::getPtrNative(eta_buf);
    float* host_r = alpaka::getPtrNative(r_buf);
    bool* host_isInverted = alpaka::getPtrNative(isInverted_buf);
    bool* host_isLower = alpaka::getPtrNative(isLower_buf);
    bool* host_isAnchor = alpaka::getPtrNative(isAnchor_buf);
    ModuleType* host_moduleType = alpaka::getPtrNative(moduleType_buf);
    ModuleLayerType* host_moduleLayerType = alpaka::getPtrNative(moduleLayerType_buf);
    float* host_slopes = alpaka::getPtrNative(slopes_buf);
    float* host_drdzs = alpaka::getPtrNative(drdzs_buf);
    uint16_t* host_partnerModuleIndices = alpaka::getPtrNative(partnerModuleIndices_buf);
    int host_sdlLayers = alpaka::getPtrNative(host_sdlLayers_buf);

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

    fillConnectedModuleArrayExplicit(modulesInGPU,nModules,stream);
    fillMapArraysExplicit(modulesInGPU, nModules, stream);
    fillPixelMap(modulesInGPU,pixelMapping,stream);
}

void SDL::fillConnectedModuleArray(struct modules& modulesInGPU, unsigned int nModules)
{
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
    std::vector<unsigned int> connectedModuleDetIds;
    std::vector<unsigned int> connectedModuleDetIds_pos;
    std::vector<unsigned int> connectedModuleDetIds_neg;

    int totalSizes = 0;
    int totalSizes_pos = 0;
    int totalSizes_neg = 0;
    for(unsigned int isuperbin = 0; isuperbin < size_superbins; isuperbin++)
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

        int sizes = 0;
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

        int sizes_pos = 0;
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

        int sizes_neg = 0;
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

    for(int icondet = 0; icondet < totalSizes; icondet++)
    {
        connectedPixels[icondet] = (*detIdToIndex)[connectedModuleDetIds[icondet]];
    }
    for(int icondet = 0; icondet < totalSizes_pos; icondet++)
    {
        connectedPixels[icondet+totalSizes] = (*detIdToIndex)[connectedModuleDetIds_pos[icondet]];
    }
    for(int icondet = 0; icondet < totalSizes_neg; icondet++)
    {
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

unsigned int SDL::modules::parsePartnerModuleId(unsigned int detId, bool isLowerx, bool isInvertedx)
{
    return isLowerx ? (isInvertedx ? detId - 1 : detId + 1) : (isInvertedx ? detId + 1 : detId - 1);
}
