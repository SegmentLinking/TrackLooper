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
        uint16_t* lowerModuleIndices; //3 of them now
        unsigned int* nTriplets;
        unsigned int* totOccupancyTriplets;

        unsigned int* nMemoryLocations;

        //for track extensions
        uint8_t* logicalLayers;
        unsigned int* hitIndices;
        
        //FIXME: This is used only for the inward triplets collection. This will contain the indices of the equivalent
        //outward triplets which will be stored in the T5 arrays, so that existing machinery can be used without issues
        unsigned int* outwardT3Indices;
        
        //delta beta = betaIn - betaOut
        FPX* betaIn;
        FPX* betaOut;
        FPX* pt_beta;

        bool* partOfPT5;
        bool* partOfT5;
        bool* partOfPT3;
        bool* partOfExtension;

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

    void createTripletArrayRanges(struct modules& modulesInGPU, struct objectRanges& rangesInGPU, struct segments& segmentsInGPU, uint16_t& nLowerModules, unsigned int& nTotalTriplets, cudaStream_t stream, const uint16_t& maxTripletsPerModule);

    void createInwardTripletArrayRanges(struct modules& modulesInGPU, struct objectRanges& rangesInGPU, struct segments& segmentsInGPU, uint16_t& nLowerModules, unsigned int& nTotalTriplets, cudaStream_t stream, const uint16_t& maxTripletsPerModule);

    void createTripletsInUnifiedMemory(struct triplets& tripletsInGPU, unsigned int maxTriplets, uint16_t nLowerModules,cudaStream_t stream, bool inwardTriplets=false);
    void createTripletsInExplicitMemory(struct triplets& tripletsInGPU, unsigned int maxTriplets, uint16_t nLowerModules,cudaStream_t stream, bool inwardTriplets=false);
#ifdef CUT_VALUE_DEBUG
CUDA_DEV void addTripletToMemory(struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct triplets& tripletsInGPU, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, uint16_t& innerInnerLowerModuleIndex, uint16_t& middleLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& deltaPhi, float& betaIn, float& betaOut, float& pt_beta, float& zLo, float& zHi, float& rtLo, float& rtHi, float& zLoPointed, float&
        zHiPointed, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ, unsigned int& tripletIndex, unsigned int outwardTripletIndex = 0, bool inwardTriplets = false);
#else
CUDA_DEV void addTripletToMemory(struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct triplets& tripletsInGPU, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, uint16_t& innerInnerLowerModuleIndex, uint16_t& middleLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, float& betaIn, float& betaOut, float& pt_beta, unsigned int& tripletIndex, unsigned int outwardTripletIndex = 0, bool inwardTriplets =
        false);
#endif

    CUDA_DEV bool runTripletDefaultAlgo(struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& middleLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& deltaPhi, float& betaIn, float& betaOut, float& pt_beta, float &zLo, float& zHi, float& rtLo, float& rtHi,
        float& zLoPointed, float& zHiPointed, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ);
    
    CUDA_DEV bool passPointingConstraint(struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& middleLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex, float& zOut, float& rtOut);

    CUDA_DEV bool passPointingConstraintBBB(struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& middleLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex, float& zOut, float& rtOut);

     CUDA_DEV bool passPointingConstraintBBE(struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& middleLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex, float& zOut, float& rtOut);

    CUDA_DEV bool passPointingConstraintEEE(struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& middleLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex, float& zOut, float& rtOut);

    CUDA_DEV bool passRZConstraint(struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& middleLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex);

    CUDA_DEV bool hasCommonMiniDoublet(struct segments& segmentsInGPU, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex);

    void printTriplet(struct triplets& tripletsInGPU, struct segments& segmentsInGPU, struct miniDoublets& mdsInGPU, struct hits& hitsInGPU, struct modules& modulesInGPU, unsigned int tripletIndex);

}

#endif
