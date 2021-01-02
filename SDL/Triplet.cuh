#ifndef Triplet_h
#define Triplet_h

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
#include "Tracklet.cuh"
#include "Segment.cuh"
#include "MiniDoublet.cuh"
#include "Module.cuh"
#include "Hit.cuh"
#include "PrintUtil.h"

namespace SDL
{
    struct triplets
    {
        unsigned int* segmentIndices;
        unsigned int* lowerModuleIndices; //3 of them now
        unsigned int* nTriplets;
        
        float* zOut;
        float* rtOut;

        float* deltaPhiPos;
        float* deltaPhi;
        //delta beta = betaIn - betaOut
        float* betaIn;
        float* betaOut;

#ifdef CUT_VALUE_DEBUG
        //debug variables
        float* zLo;
        float* zHi;
        float* zLoPointed;
        float* zHiPointed;
        float* sdlCut;
        float* betaInCut;
        float* betaOutCut;
        float* deltaBetaCut;
        float* rtLo;
        float* rtHi;
        float* kZ;

#endif

        triplets();
        ~triplets();
        void freeMemory();
        void freeMemoryCache();
    };

    void createTripletsInUnifiedMemory(struct triplets& tripletsInGPU, unsigned int maxTriplets, unsigned int nLowerModules);
    void createTripletsInExplicitMemory(struct triplets& tripletsInGPU, unsigned int maxTriplets, unsigned int nLowerModules);
#ifdef CUT_VALUE_DEBUG
    CUDA_DEV void addTripletToMemory(struct triplets& tripletsInGPU, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, unsigned int innerInnerLowerModuleIndex, unsigned int middleLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& deltaPhi, float& betaIn, float& betaOut, float& zLo, float& zHi, float& rtLo, float& rtHi, float& zLoPointed, float&
        zHiPointed, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ, unsigned int tripletIndex);


    CUDA_DEV bool runTripletDefaultAlgo(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, unsigned int innerInnerLowerModuleIndex, unsigned int middleLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& deltaPhi, float& betaIn, float& betaOut, float &zLo, float& zHi, float& rtLo, float& rtHi,
        float& zLoPointed, float& zHiPointed, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ);
    
#else
    CUDA_DEV void addTripletToMemory(struct triplets& tripletsInGPU, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, unsigned int innerInnerLowerModuleIndex, unsigned int middleLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& deltaPhi, float& betaIn, float& betaOut, unsigned int tripletIndex);

    CUDA_DEV bool runTripletDefaultAlgo(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, unsigned int innerInnerLowerModuleIndex, unsigned int middleLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& deltaPhi, float& betaIn, float& betaOut);

#endif

    CUDA_DEV bool passPointingConstraint(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, unsigned int innerInnerLowerModuleIndex, unsigned int middleLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, float& zOut, float& rtOut);

    CUDA_DEV bool passPointingConstraintBBB(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, unsigned int innerInnerLowerModuleIndex, unsigned int middleLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, float& zOut, float& rtOut);

     CUDA_DEV bool passPointingConstraintBBE(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, unsigned int innerInnerLowerModuleIndex, unsigned int middleLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, float& zOut, float& rtOut);

    CUDA_DEV bool passPointingConstraintEEE(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, unsigned int innerInnerLowerModuleIndex, unsigned int middleLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, float& zOut, float& rtOut);

    CUDA_DEV bool hasCommonMiniDoublet(struct segments& segmentsInGPU, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex);

    void printTriplet(struct triplets& tripletsInGPU, struct segments& segmentsInGPU, struct miniDoublets& mdsInGPU, struct hits& hitsInGPU, struct modules& modulesInGPU, unsigned int tripletIndex);

}

#endif
