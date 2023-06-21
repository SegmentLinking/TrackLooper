#ifndef Module_cuh
#define Module_cuh

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

#include <iostream>
#include <map>
#include <unordered_map>
#include "MiniDoublet.cuh"
#include "Hit.cuh"
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
        //others will be added later
        int* quintupletRanges;

        uint16_t *nEligibleT5Modules; //This number is just nEligibleModules - 1, but still we want this to be independent of the TC kernel
        uint16_t* indicesOfEligibleT5Modules;// will be allocated in createQuintuplets kernel!!!!
        //to store different starting points for variable occupancy stuff
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
    
        void freeMemoryCache();
        void freeMemory();
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
        uint16_t *nModules; //single number
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
       
        CUDA_HOSTDEV ModuleType parseModuleType(unsigned int index);
        CUDA_HOSTDEV ModuleType parseModuleType(short subdet, short layer, short ring);
        CUDA_HOSTDEV unsigned int parsePartnerModuleId(unsigned int detId, bool isLowerx, bool isInvertedx);
        CUDA_HOSTDEV ModuleLayerType parseModuleLayerType(unsigned int index);
        CUDA_HOSTDEV ModuleLayerType parseModuleLayerType(ModuleType moduleType, bool isInvertedx, bool isLowerx);

        bool parseIsInverted(unsigned int index);
        bool parseIsInverted(short subdet, short side, short module, short layer);
        bool parseIsLower(unsigned int index);
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
    struct pixelMap{
        //unsigned int* connectedPixels;
        unsigned int* connectedPixelsIndex;
        unsigned int* connectedPixelsSizes;
        //unsigned int* connectedPixelsPos;
        unsigned int* connectedPixelsIndexPos;
        unsigned int* connectedPixelsSizesPos;
        //unsigned int* connectedPixelsNeg;
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

    //functions
    void loadModulesFromFile(struct modules& modulesInGPU, uint16_t& nModules,uint16_t& nLowerModules,struct pixelMap& pixelMapping,cudaStream_t stream, const char* moduleMetaDataFilePath="data/centroid.txt");

    void createLowerModuleIndexMap(struct modules& modulesInGPU, unsigned int nLowerModules, unsigned int nModules,cudaStream_t stream);
    void createLowerModuleIndexMapExplicit(struct modules& modulesInGPU, unsigned int nLowerModules, unsigned int nModules, bool* isLower,cudaStream_t stream);
    void createModulesInExplicitMemory(struct modules& modulesInGPU,unsigned int nModules,cudaStream_t stream);
    void freeModules(struct modules& modulesInGPU,struct pixelMap& pixelMapping);
    void freeModulesCache(struct modules& modulesInGPU,struct pixelMap& pixelMapping);
    void fillPixelMap(struct modules& modulesInGPU,struct pixelMap& pixelMapping,cudaStream_t stream);
    void fillConnectedModuleArrayExplicit(struct modules& modulesInGPU, unsigned int nModules,cudaStream_t stream);
    void fillMapArraysExplicit(struct modules& modulesInGPU, unsigned int nModules,cudaStream_t stream);
    void fillConnectedModuleArray(struct modules& modulesInGPU, unsigned int nModules);
    void setDerivedQuantities(unsigned int detId, unsigned short& layer, unsigned short& ring, unsigned short& rod, unsigned short& module, unsigned short& subdet, unsigned short& side, float m_x, float m_y, float m_z, float& eta, float& r);
    void resetObjectRanges(struct objectRanges& rangesInGPU, unsigned int nModules,cudaStream_t stream);
    void createRangesInExplicitMemory(struct objectRanges& rangesInGPU,unsigned int nModules,cudaStream_t stream, unsigned int nLowerModules);
}


#endif

