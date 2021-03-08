#ifndef Quintuplet_cuh
#define Quintuplet_cuh

#ifdef __CUDACC__
#define CUDA_HOSTDEV  __host__ __device__
#define CUDA_DEV __device__
#define CUDA_CONST_VAR __device__
#else
#define CUDA_HOSTDEV
#define CUDA_DEV
#define CUDA_CONST_VAR
#endif

#include "Constants.h"
#include "EndcapGeometry.h"
#include "TiltedGeometry.h"
#include "Segment.cuh"
#include "MiniDoublet.cuh"
#include "Module.cuh"
#include "Hit.cuh"
#include "PrintUtil.h"
#include "Triplet.cuh"
#include "Tracklet.cuh"

namespace SDL
{
    struct quintuplets
    {
        unsigned int* tripletIndices;
        unsigned int* lowerModuleIndices;

        unsigned int* nQuintuplets;
        float* innerTripletPt;
        float* outerTripletPt;

        quintuplets();
        ~quintuplets();
        void freeMemory();
    };

void createQuintupletsInUnifiedMemory(struct quintuplets& quintupletsInGPU, unsigned int maxQuintuplets, unsigned int nLowerModules);

CUDA_DEV void addQuintupletToMemory(struct quintuplets& quintupletsInGPU, unsigned int innerTripletIndex, unsigned int outerTripletIndex, unsigned int lowerModule1, unsigned int lowerModule2, unsigned int lowerModule3, unsigned int lowerModule4, unsigned int lowerModule5, float innerTripletPt, float outerTripletPt, unsigned int quintupletIndex);

CUDA_DEV bool runQuintupletDefaultAlgo(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, unsigned int lowerModuleIndex1, unsigned int lowerMoudleIndex2, unsigned int lowerModuleIndex3, unsigned int lowerModuleIndex4, unsigned int lowerModuleIndex5, unsigned int innerTripletIndex, unsigned int outerTripletIndex, float& innerTripletPt, float& outerTripletPt);

CUDA_DEV bool T5HasCommonMiniDoublet(struct triplets& tripletsInGPU, struct segments& segmentsInGPU, unsigned int innerTripletIndex, unsigned int outerTripletIndex);

}
#endif
