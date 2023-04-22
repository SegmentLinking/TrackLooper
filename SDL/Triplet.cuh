#ifndef Triplet_cuh
#define Triplet_cuh

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
#include "Constants.cuh"
#include "EndcapGeometry.cuh"
#include "TiltedGeometry.h"
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
        uint16_t* lowerModuleIndices; //3 of them now
        unsigned int* nTriplets;
        unsigned int* totOccupancyTriplets;

        unsigned int* nMemoryLocations;

        uint8_t* logicalLayers;
        unsigned int* hitIndices;
        
        //delta beta = betaIn - betaOut
        FPX* betaIn;
        FPX* betaOut;
        FPX* pt_beta;

        bool* partOfPT5;
        bool* partOfT5;
        bool* partOfPT3;

#ifdef CUT_VALUE_DEBUG
        //debug variables
        float* zOut;
        float* rtOut;

        float* deltaPhiPos;
        float* deltaPhi;

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
        void freeMemory(cudaStream_t stream);
        void freeMemoryCache();
        void resetMemory(unsigned int maxTriplets, unsigned int nLowerModules,cudaStream_t stream);
    };

    __global__ void createTripletArrayRanges(struct modules& modulesInGPU, struct objectRanges& rangesInGPU, struct segments& segmentsInGPU);
    __global__ void addTripletRangesToEventExplicit(struct modules& modulesInGPU, struct triplets& tripletsInGPU, struct objectRanges& rangesInGPU);

    void createTripletsInExplicitMemory(struct triplets& tripletsInGPU, unsigned int maxTriplets, uint16_t nLowerModules,cudaStream_t stream);
#ifdef CUT_VALUE_DEBUG
CUDA_DEV void addTripletToMemory(struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct triplets& tripletsInGPU, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, uint16_t& innerInnerLowerModuleIndex, uint16_t& middleLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& deltaPhi, float& betaIn, float& betaOut, float& pt_beta, float& zLo, float& zHi, float& rtLo, float& rtHi, float& zLoPointed, float&
        zHiPointed, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ, unsigned int& tripletIndex);
#else
CUDA_DEV void addTripletToMemory(struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct triplets& tripletsInGPU, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, uint16_t& innerInnerLowerModuleIndex, uint16_t& middleLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, float& betaIn, float& betaOut, float& pt_beta, unsigned int& tripletIndex);
#endif

    CUDA_DEV bool runTripletDefaultAlgo(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& middleLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& deltaPhi, float& betaIn, float& betaOut, float& pt_beta, float &zLo, float& zHi, float& rtLo, float& rtHi,
        float& zLoPointed, float& zHiPointed, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ);
    
    CUDA_DEV bool passPointingConstraint(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& middleLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex, float& zOut, float& rtOut);

    CUDA_DEV bool passPointingConstraintBBB(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& middleLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex, float& zOut, float& rtOut);

     CUDA_DEV bool passPointingConstraintBBE(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& middleLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex, float& zOut, float& rtOut);

    CUDA_DEV bool passPointingConstraintEEE(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& middleLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex, float& zOut, float& rtOut);

    CUDA_DEV bool passRZConstraint(struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& middleLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex);

    CUDA_DEV bool hasCommonMiniDoublet(struct segments& segmentsInGPU, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex);

    void printTriplet(struct triplets& tripletsInGPU, struct segments& segmentsInGPU, struct miniDoublets& mdsInGPU, struct hits& hitsInGPU, struct modules& modulesInGPU, unsigned int tripletIndex);

    CUDA_DEV void runDeltaBetaIterationsT3(float& betaIn, float& betaOut, float& betaAv, float & pt_beta, float sdIn_dr, float sdOut_dr, float dr, float lIn);

    CUDA_DEV bool runTripletDefaultAlgoBBBB(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& innerOuterLowerModuleIndex, uint16_t& outerInnerLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex, unsigned int& fourthMDIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& dPhi, float& betaIn, float& betaOut, float& pt_beta, float& zLo, float& zHi, float& zLoPointed, float& zHiPointed, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut);

    CUDA_DEV bool runTripletDefaultAlgoBBEE(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& innerOuterLowerModuleIndex, uint16_t& outerInnerLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex, unsigned int& fourthMDIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& dPhi, float& betaIn, float& betaOut, float& pt_beta, float& zLo, float& rtLo, float& rtHi, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ);

    CUDA_DEV bool runTripletDefaultAlgoEEEE(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& innerOuterLowerModuleIndex, uint16_t& outerInnerLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex, unsigned int& fourthMDIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& dPhi, float& betaIn, float& betaOut, float& pt_beta, float& zLo, float& rtLo, float& rtHi, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ);

    CUDA_DEV bool runTripletDefaultAlgo(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& innerOuterLowerModuleIndex, uint16_t& outerInnerLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex, unsigned int& fourthMDIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& deltaPhi, float& betaIn, float& betaOut, float& pt_beta, float& zLo, float& zHi, float& rtLo, float& rtHi, float& zLoPointed, float& zHiPointed, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ);
    
    __global__ void createTripletsInGPUv2(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::objectRanges& rangesInGPU, uint16_t* index_gpu, uint16_t nonZeroModules);
}

#endif
