#ifndef Module_cuh
#define Module_cuh

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

#include <iostream>
#include <map>
#include "MiniDoublet.cuh"
#include "Hit.cuh"
#include "TiltedGeometry.h"
#include "EndcapGeometry.h"
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


    struct modules
    {
        unsigned int* detIds;
        unsigned int* moduleMap;
        unsigned int* nConnectedModules;
        float* drdzs;
        float* slopes;
        unsigned int *nModules; //single number
        unsigned int *nLowerModules;
        unsigned int *nEligibleModules;
        unsigned int *nEligibleT5Modules; //This number is just nEligibleModules - 1, but still we want this to be independent of the TC kernel
        unsigned int *lowerModuleIndices;
        int *reverseLookupLowerModuleIndices; //module index to lower module index reverse lookup

        int *trackCandidateModuleIndices;
        int *quintupletModuleIndices;

        
        short* layers;
        short* rings;
        short* modules;
        short* rods;
        short* subdets;
        short* sides;
        bool* isInverted;
        bool* isLower;
        ModuleType* moduleType;
        ModuleLayerType* moduleLayerType;
        
//        CUDA_HOSTDEV bool isInverted(unsigned int index);
//        CUDA_HOSTDEV bool isLower(unsigned int index);
        CUDA_HOSTDEV unsigned int partnerModuleIndex(unsigned int index);
        CUDA_HOSTDEV unsigned int partnerModuleIndexExplicit(unsigned int index,bool isLowerx, bool isInvertedx);
        CUDA_HOSTDEV ModuleType parseModuleType(unsigned int index);
        CUDA_HOSTDEV ModuleType parseModuleType(unsigned int index, short subdet, short layer, short ring);
        CUDA_HOSTDEV ModuleLayerType parseModuleLayerType(unsigned int index);
        CUDA_HOSTDEV ModuleLayerType parseModuleLayerType(unsigned int index, ModuleType moduleType, bool isInvertedx, bool isLowerx);

        bool parseIsInverted(unsigned int index);
        bool parseIsInverted(unsigned int index,short subdet, short side, short module, short layer);
        bool parseIsLower(unsigned int index);
        bool parseIsLower(unsigned int index, bool isInvertedx,unsigned int detId);

        int* hitRanges;
        int* mdRanges;
        int* segmentRanges;
        int* trackletRanges;
        int* tripletRanges;
        int* trackCandidateRanges;
        //others will be added later
        int* quintupletRanges;
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
    };

    extern std::map <unsigned int,unsigned int>* detIdToIndex;


    //functions
    void loadModulesFromFile(struct modules& modulesInGPU, unsigned int& nModules,struct pixelMap& pixelMapping, const char* moduleMetaDataFilePath="data/centroid.txt");

    void createLowerModuleIndexMap(struct modules& modulesInGPU, unsigned int nLowerModules, unsigned int nModules);
    void createLowerModuleIndexMapExplicit(struct modules& modulesInGPU, unsigned int nLowerModules, unsigned int nModules, bool* isLower);
    void createModulesInUnifiedMemory(struct modules& modulesInGPU,unsigned int nModules);
    void createModulesInExplicitMemory(struct modules& modulesInGPU,unsigned int nModules);
    void freeModules(struct modules& modulesInGPU);
    void freeModulesCache(struct modules& modulesInGPU);
    void fillPixelMap(struct modules& modulesInGPU,struct pixelMap& pixelMapping);
    void fillConnectedModuleArrayExplicit(struct modules& modulesInGPU, unsigned int nModules);
    void fillConnectedModuleArray(struct modules& modulesInGPU, unsigned int nModules);
    void setDerivedQuantities(unsigned int detId, unsigned short& layer, unsigned short& ring, unsigned short& rod, unsigned short& module, unsigned short& subdet, unsigned short& side);
    void resetObjectRanges(struct modules& modulesInGPU, unsigned int nModules);
    void resetObjectRangesExplicit(struct modules& modulesInGPU, unsigned int nModules);
}
#endif

