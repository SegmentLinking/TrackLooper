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

CUDA_DEV bool runQuintupletDefaultAlgo(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct triplets& tripletsInGPU, unsigned int lowerModuleIndex1, unsigned int lowerMoudleIndex2, unsigned int lowerModuleIndex3, unsigned int lowerModuleIndex4, unsigned int lowerModuleIndex5, unsigned int innerTripletIndex, unsigned int outerTripletIndex, float& innerTripletPt, float& outerTripletPt);

CUDA_DEV bool T5HasCommonMiniDoublet(struct triplets& tripletsInGPU, struct segments& segmentsInGPU, unsigned int innerTripletIndex, unsigned int outerTripletIndex);

CUDA_DEV float computeRadiusFromThreeAnchorHits(struct SDL::hits& hitsInGPU, unsigned int firstAnchorHit, unsigned int secondAnchorHit, unsigned int thirdAnchorHit, float& g, float& f);

CUDA_DEV float computeRadiusFromThreeAnchorHits(float x1, float y1, float x2, float y2, float x3, float y3, float& g, float& f);

CUDA_DEV float computeErrorInRadius(float* x1Vec, float* y1Vec, float* x2Vec, float* y2Vec, float* x3Vec, float* y3Vec, float& gError, float& fError);

}
#endif
