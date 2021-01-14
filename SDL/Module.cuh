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
        unsigned int *lowerModuleIndices;
        int *reverseLookupLowerModuleIndices; //module index to lower module index reverse lookup

        
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
        CUDA_HOSTDEV ModuleType parseModuleType(unsigned int index);
        CUDA_HOSTDEV ModuleLayerType parseModuleLayerType(unsigned int index);

        bool parseIsInverted(unsigned int index);
        bool parseIsLower(unsigned int index);

        int* hitRanges;
        int* mdRanges;
        int* segmentRanges;
        int* trackletRanges;
        int* tripletRanges;
        int* trackCandidateRanges;
        //others will be added later

    };

    extern std::map <unsigned int,unsigned int>* detIdToIndex;


    //functions
    void loadModulesFromFile(struct modules& modulesInGPU, unsigned int& nModules, const char* moduleMetaDataFilePath="data/centroid.txt");

    void createLowerModuleIndexMap(struct modules& modulesInGPU, unsigned int nLowerModules, unsigned int nModules);
    void createModulesInUnifiedMemory(struct modules& modulesInGPU,unsigned int nModules);
    void freeModulesInUnifiedMemory(struct modules& modulesInGPU);
    void fillConnectedModuleArray(struct modules& modulesInGPU, unsigned int nModules);
    void setDerivedQuantities(unsigned int detId, unsigned short& layer, unsigned short& ring, unsigned short& rod, unsigned short& module, unsigned short& subdet, unsigned short& side);
    void resetObjectRanges(struct modules& modulesInGPU, unsigned int nModules);
}
#endif

