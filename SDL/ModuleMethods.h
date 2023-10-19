#ifndef ModuleMethods_cuh
#define ModuleMethods_cuh

#include <map>
#include <iostream>

#include "Constants.h"
#include "Module.h"
#include "TiltedGeometry.h"
#include "EndcapGeometry.h"
#include "ModuleConnectionMap.h"

namespace SDL
{
    // TODO: Change this to remove it from global scope.
    inline std::map <unsigned int, uint16_t>* detIdToIndex;
    inline std::map <unsigned int, float> *module_x;
    inline std::map <unsigned int, float> *module_y;
    inline std::map <unsigned int, float> *module_z;
    inline std::map <unsigned int, unsigned int> *module_type; // 23 : Ph2PSP, 24 : Ph2PSS, 25 : Ph2SS
    // https://github.com/cms-sw/cmssw/blob/5e809e8e0a625578aa265dc4b128a93830cb5429/Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h#L29

    // PixelMap is never allocated on the device.
    // This is also not passed to any of the kernels, so we can combine the structs.
    struct pixelMap
    {
        Buf<alpaka::DevCpu, unsigned int> connectedPixelsIndex_buf;
        Buf<alpaka::DevCpu, unsigned int> connectedPixelsSizes_buf;
        Buf<alpaka::DevCpu, unsigned int> connectedPixelsIndexPos_buf;
        Buf<alpaka::DevCpu, unsigned int> connectedPixelsSizesPos_buf;
        Buf<alpaka::DevCpu, unsigned int> connectedPixelsIndexNeg_buf;
        Buf<alpaka::DevCpu, unsigned int> connectedPixelsSizesNeg_buf;

        unsigned int* connectedPixelsIndex;
        unsigned int* connectedPixelsSizes;
        unsigned int* connectedPixelsIndexPos;
        unsigned int* connectedPixelsSizesPos;
        unsigned int* connectedPixelsIndexNeg;
        unsigned int* connectedPixelsSizesNeg;

        int* pixelType;

        pixelMap(unsigned int sizef = size_superbins) :
            connectedPixelsIndex_buf(allocBufWrapper<unsigned int>(devHost, sizef)),
            connectedPixelsSizes_buf(allocBufWrapper<unsigned int>(devHost, sizef)),
            connectedPixelsIndexPos_buf(allocBufWrapper<unsigned int>(devHost, sizef)),
            connectedPixelsSizesPos_buf(allocBufWrapper<unsigned int>(devHost, sizef)),
            connectedPixelsIndexNeg_buf(allocBufWrapper<unsigned int>(devHost, sizef)),
            connectedPixelsSizesNeg_buf(allocBufWrapper<unsigned int>(devHost, sizef))
        {
            connectedPixelsIndex = alpaka::getPtrNative(connectedPixelsIndex_buf);
            connectedPixelsSizes = alpaka::getPtrNative(connectedPixelsSizes_buf);
            connectedPixelsIndexPos = alpaka::getPtrNative(connectedPixelsIndexPos_buf);
            connectedPixelsSizesPos = alpaka::getPtrNative(connectedPixelsSizesPos_buf);
            connectedPixelsIndexNeg = alpaka::getPtrNative(connectedPixelsIndexNeg_buf);
            connectedPixelsSizesNeg = alpaka::getPtrNative(connectedPixelsSizesNeg_buf);
        }
    };

    template<typename TQueue, typename TAcc>
    inline void fillPixelMap(struct modulesBuffer<TAcc>* modulesBuf, struct pixelMap& pixelMapping, TQueue queue)
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

        int connectedPix_size = totalSizes + totalSizes_pos + totalSizes_neg;

        // Temporary check for module initialization.
        if(pix_tot != connectedPix_size) {
            std::cerr << "\nError: pix_tot and connectedPix_size are not equal.\n";
            std::cerr << "pix_tot: " << pix_tot << ", connectedPix_size: " << connectedPix_size << "\n";
            std::cerr << "Please change pix_tot in Constants.h to make it equal to connectedPix_size.\n";
            throw std::runtime_error("Mismatched sizes");
        }

        auto connectedPixels_buf = allocBufWrapper<unsigned int>(devHost, connectedPix_size);
        unsigned int* connectedPixels = alpaka::getPtrNative(connectedPixels_buf);

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

        alpaka::memcpy(queue, modulesBuf->connectedPixels_buf, connectedPixels_buf, connectedPix_size);
        alpaka::wait(queue);
    };

    template<typename TQueue, typename TAcc>
    inline void fillConnectedModuleArrayExplicit(struct modulesBuffer<TAcc>* modulesBuf, unsigned int nMod, TQueue queue)
    {
        auto moduleMap_buf = allocBufWrapper<uint16_t>(devHost, nMod * 40);
        uint16_t* moduleMap = alpaka::getPtrNative(moduleMap_buf);

        auto nConnectedModules_buf = allocBufWrapper<uint16_t>(devHost, nMod);
        uint16_t* nConnectedModules = alpaka::getPtrNative(nConnectedModules_buf);

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

        alpaka::memcpy(queue, modulesBuf->moduleMap_buf, moduleMap_buf, nMod * 40);
        alpaka::memcpy(queue, modulesBuf->nConnectedModules_buf, nConnectedModules_buf, nMod);
        alpaka::wait(queue);
    };

    template<typename TQueue, typename TAcc>
    inline void fillMapArraysExplicit(struct modulesBuffer<TAcc>* modulesBuf, unsigned int nMod, TQueue queue)
    {
        auto mapIdx_buf = allocBufWrapper<uint16_t>(devHost, nMod);
        uint16_t* mapIdx = alpaka::getPtrNative(mapIdx_buf);

        auto mapdetId_buf = allocBufWrapper<unsigned int>(devHost, nMod);
        unsigned int* mapdetId = alpaka::getPtrNative(mapdetId_buf);

        unsigned int counter = 0;
        for(auto it = (*detIdToIndex).begin(); it != (*detIdToIndex).end(); ++it)
        {
            unsigned int detId = it->first;
            unsigned int index = it->second;
            mapIdx[counter] = index;
            mapdetId[counter] = detId;
            counter++;
        }

        alpaka::memcpy(queue, modulesBuf->mapIdx_buf, mapIdx_buf, nMod);
        alpaka::memcpy(queue, modulesBuf->mapdetId_buf, mapdetId_buf, nMod);
        alpaka::wait(queue);
    };

    inline void setDerivedQuantities(unsigned int detId, unsigned short& layer, unsigned short& ring, unsigned short& rod, unsigned short& module, unsigned short& subdet, unsigned short& side, float m_x, float m_y, float m_z, float& eta, float& r)
    {
        subdet = (detId & (7 << 25)) >> 25;
        side = (subdet == Endcap) ? (detId & (3 << 23)) >> 23 : (detId & (3 << 18)) >> 18;
        layer = (subdet == Endcap) ? (detId & (7 << 18)) >> 18 : (detId & (7 << 20)) >> 20;
        ring = (subdet == Endcap) ? (detId & (15 << 12)) >> 12 : 0;
        module = (detId & (127 << 2)) >> 2;
        rod = (subdet == Endcap) ? 0 : (detId & (127 << 10)) >> 10;

        r = std::sqrt(m_x * m_x + m_y * m_y + m_z * m_z);
        eta = ((m_z > 0) - ( m_z < 0)) * std::acosh(r / std::sqrt(m_x * m_x + m_y * m_y));
    };

    template<typename TQueue, typename TAcc>
    void loadModulesFromFile(struct modules* modulesInGPU,
                             struct modulesBuffer<TAcc>* modulesBuf,
                             uint16_t& nModules,
                             uint16_t& nLowerModules,
                             struct pixelMap& pixelMapping,
                             TQueue& queue,
                             const char* moduleMetaDataFilePath)
    {
        detIdToIndex = new std::map<unsigned int, uint16_t>;
        module_x = new std::map<unsigned int, float>;
        module_y = new std::map<unsigned int, float>;
        module_z = new std::map<unsigned int, float>;
        module_type = new std::map<unsigned int, unsigned int>;

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

        // Temporary check for module initialization.
        if(modules_size != nModules) {
            std::cerr << "\nError: modules_size and nModules are not equal.\n";
            std::cerr << "modules_size: " << modules_size << ", nModules: " << nModules << "\n";
            std::cerr << "Please change modules_size in Constants.h to make it equal to nModules.\n";
            throw std::runtime_error("Mismatched sizes");
        }

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
        auto sdlLayers_buf = allocBufWrapper<int>(devHost, nModules);

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
        int* host_sdlLayers = alpaka::getPtrNative(sdlLayers_buf);

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
                isInverted = modulesInGPU->parseIsInverted(subdet, side, module, layer);
                isLower = modulesInGPU->parseIsLower(isInverted, detId);
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

                host_slopes[index] = (subdet == Endcap) ? endcapGeometry->getSlopeLower(detId) : tiltedGeometry.getSlope(detId);
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
                host_partnerModuleIndices[index] = (*detIdToIndex)[modulesInGPU->parsePartnerModuleId(detId, host_isLower[index], host_isInverted[index])];
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

        auto src_view_nModules = alpaka::createView(devHost, &nModules, (Idx) 1u);
        alpaka::memcpy(queue, modulesBuf->nModules_buf, src_view_nModules);

        auto src_view_nLowerModules = alpaka::createView(devHost, &nLowerModules, (Idx) 1u);
        alpaka::memcpy(queue, modulesBuf->nLowerModules_buf, src_view_nLowerModules);

        alpaka::memcpy(queue, modulesBuf->moduleType_buf, moduleType_buf);
        alpaka::memcpy(queue, modulesBuf->moduleLayerType_buf, moduleLayerType_buf);

        alpaka::memcpy(queue, modulesBuf->detIds_buf, detIds_buf);
        alpaka::memcpy(queue, modulesBuf->layers_buf, layers_buf);
        alpaka::memcpy(queue, modulesBuf->rings_buf, rings_buf);
        alpaka::memcpy(queue, modulesBuf->rods_buf, rods_buf);
        alpaka::memcpy(queue, modulesBuf->modules_buf, modules_buf);
        alpaka::memcpy(queue, modulesBuf->subdets_buf, subdets_buf);
        alpaka::memcpy(queue, modulesBuf->sides_buf, sides_buf);
        alpaka::memcpy(queue, modulesBuf->eta_buf, eta_buf);
        alpaka::memcpy(queue, modulesBuf->r_buf, r_buf);
        alpaka::memcpy(queue, modulesBuf->isInverted_buf, isInverted_buf);
        alpaka::memcpy(queue, modulesBuf->isLower_buf, isLower_buf);
        alpaka::memcpy(queue, modulesBuf->isAnchor_buf, isAnchor_buf);
        alpaka::memcpy(queue, modulesBuf->slopes_buf, slopes_buf);
        alpaka::memcpy(queue, modulesBuf->drdzs_buf, drdzs_buf);
        alpaka::memcpy(queue, modulesBuf->partnerModuleIndices_buf, partnerModuleIndices_buf);
        alpaka::memcpy(queue, modulesBuf->sdlLayers_buf, sdlLayers_buf);
        alpaka::wait(queue);

        fillConnectedModuleArrayExplicit(modulesBuf, nModules, queue);
        fillMapArraysExplicit(modulesBuf, nModules, queue);
        fillPixelMap(modulesBuf, pixelMapping, queue);
    };
}
#endif
