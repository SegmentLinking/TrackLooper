#ifndef Module_cuh
#define Module_cuh

#include <map>
#include <iostream>

#include "Constants.cuh"
#include "TiltedGeometry.h"
#include "EndcapGeometry.cuh"
#include "ModuleConnectionMap.h"

namespace SDL
{
    enum SubDet
    {
        InnerPixel = 0,
        Barrel = 5,
        Endcap = 4
    };

    enum Side
    {
        NegZ = 1,
        PosZ = 2,
        Center = 3
    };

    enum ModuleType
    {
        PS,
        TwoS,
        PixelModule
    };

    enum ModuleLayerType
    {
        Pixel,
        Strip,
        InnerPixelLayer
    };

    extern std::map <unsigned int, uint16_t>* detIdToIndex;
    extern std::map <unsigned int, float> *module_x;
    extern std::map <unsigned int, float> *module_y;
    extern std::map <unsigned int, float> *module_z;
    extern std::map <unsigned int, unsigned int> *module_type;

    struct objectRanges
    {
        int* hitRanges;
        int* hitRangesLower;
        int* hitRangesUpper;
        int8_t* hitRangesnLower;
        int8_t* hitRangesnUpper;
        int* mdRanges;
        int* segmentRanges;
        int* trackletRanges;
        int* tripletRanges;
        int* trackCandidateRanges;
        // Others will be added later
        int* quintupletRanges;

        // This number is just nEligibleModules - 1, but still we want this to be independent of the TC kernel
        uint16_t *nEligibleT5Modules;
        // Will be allocated in createQuintuplets kernel!
        uint16_t* indicesOfEligibleT5Modules;
        // To store different starting points for variable occupancy stuff
        int *quintupletModuleIndices;
        int *quintupletModuleOccupancy;
        int *miniDoubletModuleIndices;
        int *miniDoubletModuleOccupancy;
        int *segmentModuleIndices;
        int *segmentModuleOccupancy;
        int *tripletModuleIndices;
        int *tripletModuleOccupancy;

        unsigned int *device_nTotalMDs;
        unsigned int *device_nTotalSegs;
        unsigned int *device_nTotalTrips;
        unsigned int *device_nTotalQuints;

        template<typename TBuff>
        void setData(TBuff& objectRangesbuf)
        {
            hitRanges = alpaka::getPtrNative(objectRangesbuf.hitRanges_buf);
            hitRangesLower = alpaka::getPtrNative(objectRangesbuf.hitRangesLower_buf);
            hitRangesUpper = alpaka::getPtrNative(objectRangesbuf.hitRangesUpper_buf);
            hitRangesnLower = alpaka::getPtrNative(objectRangesbuf.hitRangesnLower_buf);
            hitRangesnUpper = alpaka::getPtrNative(objectRangesbuf.hitRangesnUpper_buf);
            mdRanges = alpaka::getPtrNative(objectRangesbuf.mdRanges_buf);
            segmentRanges = alpaka::getPtrNative(objectRangesbuf.segmentRanges_buf);
            trackletRanges = alpaka::getPtrNative(objectRangesbuf.trackletRanges_buf);
            tripletRanges = alpaka::getPtrNative(objectRangesbuf.tripletRanges_buf);
            trackCandidateRanges = alpaka::getPtrNative(objectRangesbuf.trackCandidateRanges_buf);
            quintupletRanges = alpaka::getPtrNative(objectRangesbuf.quintupletRanges_buf);

            nEligibleT5Modules = alpaka::getPtrNative(objectRangesbuf.nEligibleT5Modules_buf);
            indicesOfEligibleT5Modules = alpaka::getPtrNative(objectRangesbuf.indicesOfEligibleT5Modules_buf);

            quintupletModuleIndices = alpaka::getPtrNative(objectRangesbuf.quintupletModuleIndices_buf);
            quintupletModuleOccupancy = alpaka::getPtrNative(objectRangesbuf.quintupletModuleOccupancy_buf);
            miniDoubletModuleIndices = alpaka::getPtrNative(objectRangesbuf.miniDoubletModuleIndices_buf);
            miniDoubletModuleOccupancy = alpaka::getPtrNative(objectRangesbuf.miniDoubletModuleOccupancy_buf);
            segmentModuleIndices = alpaka::getPtrNative(objectRangesbuf.segmentModuleIndices_buf);
            segmentModuleOccupancy = alpaka::getPtrNative(objectRangesbuf.segmentModuleOccupancy_buf);
            tripletModuleIndices = alpaka::getPtrNative(objectRangesbuf.tripletModuleIndices_buf);
            tripletModuleOccupancy = alpaka::getPtrNative(objectRangesbuf.tripletModuleOccupancy_buf);

            device_nTotalMDs = alpaka::getPtrNative(objectRangesbuf.device_nTotalMDs_buf);
            device_nTotalSegs = alpaka::getPtrNative(objectRangesbuf.device_nTotalSegs_buf);
            device_nTotalTrips = alpaka::getPtrNative(objectRangesbuf.device_nTotalTrips_buf);
            device_nTotalQuints = alpaka::getPtrNative(objectRangesbuf.device_nTotalQuints_buf);
        }
    };

    template<typename TAcc>
    struct objectRangesBuffer : objectRanges
    {
        Buf<TAcc, int> hitRanges_buf;
        Buf<TAcc, int> hitRangesLower_buf;
        Buf<TAcc, int> hitRangesUpper_buf;
        Buf<TAcc, int8_t> hitRangesnLower_buf;
        Buf<TAcc, int8_t> hitRangesnUpper_buf;
        Buf<TAcc, int> mdRanges_buf;
        Buf<TAcc, int> segmentRanges_buf;
        Buf<TAcc, int> trackletRanges_buf;
        Buf<TAcc, int> tripletRanges_buf;
        Buf<TAcc, int> trackCandidateRanges_buf;
        Buf<TAcc, int> quintupletRanges_buf;

        Buf<TAcc, uint16_t> nEligibleT5Modules_buf;
        Buf<TAcc, uint16_t> indicesOfEligibleT5Modules_buf;

        Buf<TAcc, int> quintupletModuleIndices_buf;
        Buf<TAcc, int> quintupletModuleOccupancy_buf;
        Buf<TAcc, int> miniDoubletModuleIndices_buf;
        Buf<TAcc, int> miniDoubletModuleOccupancy_buf;
        Buf<TAcc, int> segmentModuleIndices_buf;
        Buf<TAcc, int> segmentModuleOccupancy_buf;
        Buf<TAcc, int> tripletModuleIndices_buf;
        Buf<TAcc, int> tripletModuleOccupancy_buf;

        Buf<TAcc, unsigned int> device_nTotalMDs_buf;
        Buf<TAcc, unsigned int> device_nTotalSegs_buf;
        Buf<TAcc, unsigned int> device_nTotalTrips_buf;
        Buf<TAcc, unsigned int> device_nTotalQuints_buf;

        template<typename TQueue, typename TDevAcc>
        objectRangesBuffer(unsigned int nMod,
                           unsigned int nLowerMod,
                           TDevAcc const & devAccIn,
                           TQueue& queue) :
            hitRanges_buf(allocBufWrapper<int>(devAccIn, nMod*2)),
            hitRangesLower_buf(allocBufWrapper<int>(devAccIn, nMod)),
            hitRangesUpper_buf(allocBufWrapper<int>(devAccIn, nMod)),
            hitRangesnLower_buf(allocBufWrapper<int8_t>(devAccIn, nMod)),
            hitRangesnUpper_buf(allocBufWrapper<int8_t>(devAccIn, nMod)),
            mdRanges_buf(allocBufWrapper<int>(devAccIn, nMod*2)),
            segmentRanges_buf(allocBufWrapper<int>(devAccIn, nMod*2)),
            trackletRanges_buf(allocBufWrapper<int>(devAccIn, nMod*2)),
            tripletRanges_buf(allocBufWrapper<int>(devAccIn, nMod*2)),
            trackCandidateRanges_buf(allocBufWrapper<int>(devAccIn, nMod*2)),
            quintupletRanges_buf(allocBufWrapper<int>(devAccIn, nMod*2)),
            nEligibleT5Modules_buf(allocBufWrapper<uint16_t>(devAccIn, 1)),
            indicesOfEligibleT5Modules_buf(allocBufWrapper<uint16_t>(devAccIn, nLowerMod)),
            quintupletModuleIndices_buf(allocBufWrapper<int>(devAccIn, nLowerMod)),
            quintupletModuleOccupancy_buf(allocBufWrapper<int>(devAccIn, nLowerMod)),
            miniDoubletModuleIndices_buf(allocBufWrapper<int>(devAccIn, nLowerMod+1)),
            miniDoubletModuleOccupancy_buf(allocBufWrapper<int>(devAccIn, nLowerMod+1)),
            segmentModuleIndices_buf(allocBufWrapper<int>(devAccIn, nLowerMod+1)),
            segmentModuleOccupancy_buf(allocBufWrapper<int>(devAccIn, nLowerMod+1)),
            tripletModuleIndices_buf(allocBufWrapper<int>(devAccIn, nLowerMod)),
            tripletModuleOccupancy_buf(allocBufWrapper<int>(devAccIn, nLowerMod)),
            device_nTotalMDs_buf(allocBufWrapper<unsigned int>(devAccIn, 1)),
            device_nTotalSegs_buf(allocBufWrapper<unsigned int>(devAccIn, 1)),
            device_nTotalTrips_buf(allocBufWrapper<unsigned int>(devAccIn, 1)),
            device_nTotalQuints_buf(allocBufWrapper<unsigned int>(devAccIn, 1))
        {
            alpaka::memset(queue, hitRanges_buf, -1, nMod*2);
            alpaka::memset(queue, hitRangesLower_buf, -1, nMod);
            alpaka::memset(queue, hitRangesUpper_buf, -1, nMod);
            alpaka::memset(queue, hitRangesnLower_buf, -1, nMod);
            alpaka::memset(queue, hitRangesnUpper_buf, -1, nMod);
            alpaka::memset(queue, mdRanges_buf, -1, nMod*2);
            alpaka::memset(queue, segmentRanges_buf, -1, nMod*2);
            alpaka::memset(queue, trackletRanges_buf, -1, nMod*2);
            alpaka::memset(queue, tripletRanges_buf, -1, nMod*2);
            alpaka::memset(queue, trackCandidateRanges_buf, -1, nMod*2);
            alpaka::memset(queue, quintupletRanges_buf, -1, nMod*2);
            alpaka::memset(queue, quintupletModuleIndices_buf, -1, nLowerMod);
            alpaka::wait(queue);
        }
    };

    struct modules
    {
        unsigned int* detIds;
        uint16_t* moduleMap;
        unsigned int* mapdetId;
        uint16_t* mapIdx;
        uint16_t* nConnectedModules;
        float* drdzs;
        float* slopes;
        uint16_t *nModules;
        uint16_t *nLowerModules;
        uint16_t* partnerModuleIndices;

        short* layers;
        short* rings;
        short* modules;
        short* rods;
        short* subdets;
        short* sides;
        float* eta;
        float* r;
        bool* isInverted;
        bool* isLower;
        bool* isAnchor;
        ModuleType* moduleType;
        ModuleLayerType* moduleLayerType;
        int* sdlLayers;

        unsigned int* connectedPixels;

        bool parseIsInverted(short subdet, short side, short module, short layer)
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
        };

        bool parseIsLower(bool isInvertedx, unsigned int detId)
        {
            return (isInvertedx) ? !(detId & 1) : (detId & 1);
        };

        unsigned int parsePartnerModuleId(unsigned int detId, bool isLowerx, bool isInvertedx)
        {
            return isLowerx ? (isInvertedx ? detId - 1 : detId + 1) : (isInvertedx ? detId + 1 : detId - 1);
        };

        template<typename TBuff>
        void setData(TBuff& modulesbuf)
        {
            detIds = alpaka::getPtrNative(modulesbuf.detIds_buf);
            moduleMap = alpaka::getPtrNative(modulesbuf.moduleMap_buf);
            mapdetId = alpaka::getPtrNative(modulesbuf.mapdetId_buf);
            mapIdx = alpaka::getPtrNative(modulesbuf.mapIdx_buf);
            nConnectedModules = alpaka::getPtrNative(modulesbuf.nConnectedModules_buf);
            drdzs = alpaka::getPtrNative(modulesbuf.drdzs_buf);
            slopes = alpaka::getPtrNative(modulesbuf.slopes_buf);
            nModules = alpaka::getPtrNative(modulesbuf.nModules_buf);
            nLowerModules = alpaka::getPtrNative(modulesbuf.nLowerModules_buf);
            partnerModuleIndices = alpaka::getPtrNative(modulesbuf.partnerModuleIndices_buf);

            layers = alpaka::getPtrNative(modulesbuf.layers_buf);
            rings = alpaka::getPtrNative(modulesbuf.rings_buf);
            modules = alpaka::getPtrNative(modulesbuf.modules_buf);
            rods = alpaka::getPtrNative(modulesbuf.rods_buf);
            subdets = alpaka::getPtrNative(modulesbuf.subdets_buf);
            sides = alpaka::getPtrNative(modulesbuf.sides_buf);
            eta = alpaka::getPtrNative(modulesbuf.eta_buf);
            r = alpaka::getPtrNative(modulesbuf.r_buf);
            isInverted = alpaka::getPtrNative(modulesbuf.isInverted_buf);
            isLower = alpaka::getPtrNative(modulesbuf.isLower_buf);
            isAnchor = alpaka::getPtrNative(modulesbuf.isAnchor_buf);
            moduleType = alpaka::getPtrNative(modulesbuf.moduleType_buf);
            moduleLayerType = alpaka::getPtrNative(modulesbuf.moduleLayerType_buf);

            connectedPixels = alpaka::getPtrNative(modulesbuf.connectedPixels_buf);
        }
    };

    template<typename TAcc>
    struct modulesBuffer : modules
    {
        Buf<TAcc, unsigned int> detIds_buf;
        Buf<TAcc, uint16_t> moduleMap_buf;
        Buf<TAcc, unsigned int> mapdetId_buf;
        Buf<TAcc, uint16_t> mapIdx_buf;
        Buf<TAcc, uint16_t> nConnectedModules_buf;
        Buf<TAcc, float> drdzs_buf;
        Buf<TAcc, float> slopes_buf;
        Buf<TAcc, uint16_t> nModules_buf;
        Buf<TAcc, uint16_t> nLowerModules_buf;
        Buf<TAcc, uint16_t> partnerModuleIndices_buf;

        Buf<TAcc, short> layers_buf;
        Buf<TAcc, short> rings_buf;
        Buf<TAcc, short> modules_buf;
        Buf<TAcc, short> rods_buf;
        Buf<TAcc, short> subdets_buf;
        Buf<TAcc, short> sides_buf;
        Buf<TAcc, float> eta_buf;
        Buf<TAcc, float> r_buf;
        Buf<TAcc, bool> isInverted_buf;
        Buf<TAcc, bool> isLower_buf;
        Buf<TAcc, bool> isAnchor_buf;
        Buf<TAcc, ModuleType> moduleType_buf;
        Buf<TAcc, ModuleLayerType> moduleLayerType_buf;

        Buf<TAcc, unsigned int> connectedPixels_buf;

        template<typename TDevAcc>
        modulesBuffer(TDevAcc const & devAccIn,
                      unsigned int nMod = modules_size,
                      unsigned int nPixs = pix_tot) :
            detIds_buf(allocBufWrapper<unsigned int>(devAccIn, nMod)),
            moduleMap_buf(allocBufWrapper<uint16_t>(devAccIn, nMod * 40)),
            mapdetId_buf(allocBufWrapper<unsigned int>(devAccIn, nMod)),
            mapIdx_buf(allocBufWrapper<uint16_t>(devAccIn, nMod)),
            nConnectedModules_buf(allocBufWrapper<uint16_t>(devAccIn, nMod)),
            drdzs_buf(allocBufWrapper<float>(devAccIn, nMod)),
            slopes_buf(allocBufWrapper<float>(devAccIn, nMod)),
            nModules_buf(allocBufWrapper<uint16_t>(devAccIn, 1)),
            nLowerModules_buf(allocBufWrapper<uint16_t>(devAccIn, 1)),
            partnerModuleIndices_buf(allocBufWrapper<uint16_t>(devAccIn, nMod)),

            layers_buf(allocBufWrapper<short>(devAccIn, nMod)),
            rings_buf(allocBufWrapper<short>(devAccIn, nMod)),
            modules_buf(allocBufWrapper<short>(devAccIn, nMod)),
            rods_buf(allocBufWrapper<short>(devAccIn, nMod)),
            subdets_buf(allocBufWrapper<short>(devAccIn, nMod)),
            sides_buf(allocBufWrapper<short>(devAccIn, nMod)),
            eta_buf(allocBufWrapper<float>(devAccIn, nMod)),
            r_buf(allocBufWrapper<float>(devAccIn, nMod)),
            isInverted_buf(allocBufWrapper<bool>(devAccIn, nMod)),
            isLower_buf(allocBufWrapper<bool>(devAccIn, nMod)),
            isAnchor_buf(allocBufWrapper<bool>(devAccIn, nMod)),
            moduleType_buf(allocBufWrapper<ModuleType>(devAccIn, nMod)),
            moduleLayerType_buf(allocBufWrapper<ModuleLayerType>(devAccIn, nMod)),

            connectedPixels_buf(allocBufWrapper<unsigned int>(devAccIn, nPixs))
        {}
    };

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

        auto connectedPixels_buf = allocBufWrapper<unsigned int>(devHost, totalSizes + totalSizes_pos + totalSizes_neg);
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

        alpaka::memcpy(queue, modulesBuf->connectedPixels_buf, connectedPixels_buf, totalSizes + totalSizes_pos + totalSizes_neg);
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

        alpaka::memcpy(queue, modulesBuf->detIds_buf, detIds_buf, nModules);
        alpaka::memcpy(queue, modulesBuf->layers_buf, layers_buf, nModules);
        alpaka::memcpy(queue, modulesBuf->rings_buf, rings_buf, nModules);
        alpaka::memcpy(queue, modulesBuf->rods_buf, rods_buf, nModules);
        alpaka::memcpy(queue, modulesBuf->modules_buf, modules_buf, nModules);
        alpaka::memcpy(queue, modulesBuf->subdets_buf, subdets_buf, nModules);
        alpaka::memcpy(queue, modulesBuf->sides_buf, sides_buf, nModules);
        alpaka::memcpy(queue, modulesBuf->eta_buf, eta_buf, nModules);
        alpaka::memcpy(queue, modulesBuf->r_buf, r_buf, nModules);
        alpaka::memcpy(queue, modulesBuf->isInverted_buf, isInverted_buf, nModules);
        alpaka::memcpy(queue, modulesBuf->isLower_buf, isLower_buf, nModules);
        alpaka::memcpy(queue, modulesBuf->isAnchor_buf, isAnchor_buf, nModules);
        alpaka::memcpy(queue, modulesBuf->slopes_buf, slopes_buf, nModules);
        alpaka::memcpy(queue, modulesBuf->drdzs_buf, drdzs_buf, nModules);
        alpaka::memcpy(queue, modulesBuf->partnerModuleIndices_buf, partnerModuleIndices_buf, nModules);
        alpaka::wait(queue);

        fillConnectedModuleArrayExplicit(modulesBuf, nModules, queue);
        fillMapArraysExplicit(modulesBuf, nModules, queue);
        fillPixelMap(modulesBuf, pixelMapping, queue);
    };
}
#endif
