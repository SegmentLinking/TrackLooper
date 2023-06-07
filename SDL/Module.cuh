#ifndef Module_cuh
#define Module_cuh

#include <map>
#include <iostream>

#include "Constants.cuh"
#include "TiltedGeometry.h"
#include "EndcapGeometry.cuh"
#include "ModuleConnectionMap.h"
#include "allocate.h"

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
        objectRangesBuffer(unsigned int nModules,
                           unsigned int nLowerModules,
                           TDevAcc const & devAccIn,
                           TQueue& queue) :
            hitRanges_buf(allocBufWrapper<int>(devAccIn, nModules*2)),
            hitRangesLower_buf(allocBufWrapper<int>(devAccIn, nModules)),
            hitRangesUpper_buf(allocBufWrapper<int>(devAccIn, nModules)),
            hitRangesnLower_buf(allocBufWrapper<int8_t>(devAccIn, nModules)),
            hitRangesnUpper_buf(allocBufWrapper<int8_t>(devAccIn, nModules)),
            mdRanges_buf(allocBufWrapper<int>(devAccIn, nModules*2)),
            segmentRanges_buf(allocBufWrapper<int>(devAccIn, nModules*2)),
            trackletRanges_buf(allocBufWrapper<int>(devAccIn, nModules*2)),
            tripletRanges_buf(allocBufWrapper<int>(devAccIn, nModules*2)),
            trackCandidateRanges_buf(allocBufWrapper<int>(devAccIn, nModules*2)),
            quintupletRanges_buf(allocBufWrapper<int>(devAccIn, nModules*2)),
            nEligibleT5Modules_buf(allocBufWrapper<uint16_t>(devAccIn, 1)),
            indicesOfEligibleT5Modules_buf(allocBufWrapper<uint16_t>(devAccIn, nLowerModules)),
            quintupletModuleIndices_buf(allocBufWrapper<int>(devAccIn, nLowerModules)),
            quintupletModuleOccupancy_buf(allocBufWrapper<int>(devAccIn, nLowerModules)),
            miniDoubletModuleIndices_buf(allocBufWrapper<int>(devAccIn, nLowerModules+1)),
            miniDoubletModuleOccupancy_buf(allocBufWrapper<int>(devAccIn, nLowerModules+1)),
            segmentModuleIndices_buf(allocBufWrapper<int>(devAccIn, nLowerModules+1)),
            segmentModuleOccupancy_buf(allocBufWrapper<int>(devAccIn, nLowerModules+1)),
            tripletModuleIndices_buf(allocBufWrapper<int>(devAccIn, nLowerModules)),
            tripletModuleOccupancy_buf(allocBufWrapper<int>(devAccIn, nLowerModules)),
            device_nTotalMDs_buf(allocBufWrapper<unsigned int>(devAccIn, 1)),
            device_nTotalSegs_buf(allocBufWrapper<unsigned int>(devAccIn, 1)),
            device_nTotalTrips_buf(allocBufWrapper<unsigned int>(devAccIn, 1)),
            device_nTotalQuints_buf(allocBufWrapper<unsigned int>(devAccIn, 1))
        {
            alpaka::memset(queue, hitRanges_buf, -1, nModules*2);
            alpaka::memset(queue, hitRangesLower_buf, -1, nModules);
            alpaka::memset(queue, hitRangesUpper_buf, -1, nModules);
            alpaka::memset(queue, hitRangesnLower_buf, -1, nModules);
            alpaka::memset(queue, hitRangesnUpper_buf, -1, nModules);
            alpaka::memset(queue, mdRanges_buf, -1, nModules*2);
            alpaka::memset(queue, segmentRanges_buf, -1, nModules*2);
            alpaka::memset(queue, trackletRanges_buf, -1, nModules*2);
            alpaka::memset(queue, tripletRanges_buf, -1, nModules*2);
            alpaka::memset(queue, trackCandidateRanges_buf, -1, nModules*2);
            alpaka::memset(queue, quintupletRanges_buf, -1, nModules*2);
            alpaka::memset(queue, quintupletModuleIndices_buf, -1, nLowerModules);
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

        unsigned int parsePartnerModuleId(unsigned int detId, bool isLowerx, bool isInvertedx);

        bool parseIsInverted(short subdet, short side, short module, short layer);
        bool parseIsLower(bool isInvertedx,unsigned int detId);

        unsigned int* connectedPixels;
        unsigned int* connectedPixelsIndex;
        unsigned int* connectedPixelsSizes;
        unsigned int* connectedPixelsPos;
        unsigned int* connectedPixelsIndexPos;
        unsigned int* connectedPixelsSizesPos;
        unsigned int* connectedPixelsNeg;
        unsigned int* connectedPixelsIndexNeg;
        unsigned int* connectedPixelsSizesNeg;
    };

    struct pixelMap
    {
        unsigned int* connectedPixelsIndex;
        unsigned int* connectedPixelsSizes;
        unsigned int* connectedPixelsIndexPos;
        unsigned int* connectedPixelsSizesPos;
        unsigned int* connectedPixelsIndexNeg;
        unsigned int* connectedPixelsSizesNeg;

        int* superbin;
        int* pixelType;
    };

    extern std::map <unsigned int, uint16_t>* detIdToIndex;
    extern std::map <unsigned int, float> *module_x;
    extern std::map <unsigned int, float> *module_y;
    extern std::map  <unsigned int, float> *module_z;
    extern std::map  <unsigned int, unsigned int> *module_type;

    void loadModulesFromFile(struct modules& modulesInGPU, uint16_t& nModules,uint16_t& nLowerModules,struct pixelMap& pixelMapping,cudaStream_t stream, const char* moduleMetaDataFilePath="data/centroid.txt");
    void createModulesInExplicitMemory(struct modules& modulesInGPU,unsigned int nModules,cudaStream_t stream);
    void freeModules(struct modules& modulesInGPU,struct pixelMap& pixelMapping);
    void fillPixelMap(struct modules& modulesInGPU,struct pixelMap& pixelMapping,cudaStream_t stream);
    void fillConnectedModuleArrayExplicit(struct modules& modulesInGPU, unsigned int nModules,cudaStream_t stream);
    void fillMapArraysExplicit(struct modules& modulesInGPU, unsigned int nModules,cudaStream_t stream);
    void fillConnectedModuleArray(struct modules& modulesInGPU, unsigned int nModules);
    void setDerivedQuantities(unsigned int detId, unsigned short& layer, unsigned short& ring, unsigned short& rod, unsigned short& module, unsigned short& subdet, unsigned short& side, float m_x, float m_y, float m_z, float& eta, float& r);
    void createRangesInExplicitMemory(struct objectRanges& rangesInGPU,unsigned int nModules,cudaStream_t stream, unsigned int nLowerModules);
}
#endif
