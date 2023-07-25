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

#include "Constants.cuh"
#include "EndcapGeometry.cuh"
#include "TiltedGeometry.h"
#include "Segment.cuh"
#include "MiniDoublet.cuh"
#include "Module.cuh"
#include "Hit.cuh"
#include "PrintUtil.h"
#include "Triplet.cuh"
#include "NeuralNetwork.cuh"

namespace SDL
{
    struct quintuplets
    {
        unsigned int* tripletIndices;
        uint16_t* lowerModuleIndices;
        unsigned int* nQuintuplets; // ?
        unsigned int* totOccupancyQuintuplets; //?
        unsigned int* nMemoryLocations;

        FPX* innerRadius;
        FPX* bridgeRadius;
        FPX* outerRadius;
        FPX* pt;
        FPX* eta;
        FPX* phi;
        FPX* score_rphisum;
        uint8_t* layer;
        bool* isDup;
        bool* TightCutFlag;
        bool* partOfPT5;

        float* regressionRadius;
        float* regressionG;
        float* regressionF;

        uint8_t* logicalLayers;
        unsigned int* hitIndices;
        float* rzChiSquared;
        float* chiSquared;
        float* nonAnchorChiSquared;

        quintuplets();
        ~quintuplets();
        void freeMemory(cudaStream_t stream);
        void freeMemoryCache();
    };

    void createQuintupletsInExplicitMemory(struct SDL::quintuplets& quintupletsInGPU, const unsigned int& maxQuintuplets, const uint16_t& nLowerModules, const uint16_t& nEligibleModules,cudaStream_t stream);

    __global__ void createEligibleModulesListForQuintupletsGPU(struct modules& modulesInGPU, struct triplets& tripletsInGPU, struct objectRanges& rangesInGPU);
    __global__ void addQuintupletRangesToEventExplicit(struct modules& modulesInGPU, struct quintuplets& quintupletsInGPU, struct objectRanges& rangesInGPU);

//  CUDA_DEV void rmQuintupletToMemory(struct SDL::quintuplets& quintupletsInGPU, unsigned int quintupletIndex);

    CUDA_DEV void addQuintupletToMemory(struct SDL::triplets& tripletsInGPU, struct SDL::quintuplets& quintupletsInGPU, unsigned int innerTripletIndex, unsigned int outerTripletIndex, uint16_t& lowerModule1, uint16_t& lowerModule2, uint16_t& lowerModule3, uint16_t& lowerModule4, uint16_t& lowerModule5, float& innerRadius, float& bridgeRadius, float& outerRadius, float& regressionG, float& regressionF, float& regressionRadius, float& rzChiSquared, float& rPhiChiSquared, float& nonAnchorChiSquared, float pt, float eta, float phi, float scores, uint8_t layer, unsigned int quintupletIndex, bool TightCutFlag);
   
    CUDA_DEV bool runQuintupletDefaultAlgo(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, uint16_t& lowerModuleIndex1, uint16_t& lowerModuleIndex2, uint16_t& lowerModuleIndex3, uint16_t& lowerModuleIndex4, uint16_t& lowerModuleIndex5, unsigned int& innerTripletIndex, unsigned int& outerTripletIndex, float& innerRadius, float& outerRadius, float& bridgeRadius, float& regressionG, float& regressionF, float& regressionRadius, float& rzChiSquared, float& chiSquared, float& nonAnchorChiSquared, bool& TightCutFlag);


    CUDA_DEV bool passT5RZConstraint(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, unsigned int firstMDIndex, unsigned int secondMDIndex, unsigned int thirdMDIndex, unsigned int fourthMDIndex, unsigned int fifthMDIndex, uint16_t& lowerModuleIndex1, uint16_t& lowerModuleIndex2, uint16_t& lowerModuleIndex3, uint16_t& lowerModuleIndex4, uint16_t& lowerModuleIndex5, float& rzChiSquared, float inner_pt, float innerRadius, float g, float f, bool& TightCutFlag);

    CUDA_DEV bool T5HasCommonMiniDoublet(struct triplets& tripletsInGPU, struct segments& segmentsInGPU, unsigned int innerTripletIndex, unsigned int outerTripletIndex);

    CUDA_DEV float computeRadiusFromThreeAnchorHits(float x1, float y1, float x2, float y2, float x3, float y3, float& g, float& f);

    CUDA_DEV void computeErrorInRadius(float* x1Vec, float* y1Vec, float* x2Vec, float* y2Vec, float* x3Vec, float* y3Vec, float& gError, float& fError);

    CUDA_DEV bool matchRadiiBBBBB(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax);

    CUDA_DEV bool matchRadiiBBBBE(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S, float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax);

    CUDA_DEV bool matchRadiiBBBEE(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S,float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax);

    CUDA_DEV bool matchRadiiBBBEE12378(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S,float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax);

    CUDA_DEV bool matchRadiiBBBEE23478(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S,float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax);

    CUDA_DEV bool matchRadiiBBBEE34578(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S,float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax);


    CUDA_DEV bool matchRadiiBBEEE(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S,float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax);

    CUDA_DEV bool matchRadiiBEEEE(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S,float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax);

    CUDA_DEV bool matchRadiiEEEEE(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S,float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax);

    CUDA_DEV bool checkIntervalOverlap(const float& firstMin, const float& firstMax, const float& secondMin, const float& secondMax);

    CUDA_DEV float computeRadiusUsingRegression(int nPoints, float* xs, float* ys, float* delta1, float* delta2, float* slopes, bool* isFlat, float& g, float& f, float* sigmas, float& chiSquared);

    CUDA_DEV void computeSigmasForRegression(SDL::modules& modulesInGPU, const uint16_t* lowerModuleIndices, float* delta1, float* delta2, float* slopes, bool* isFlat, int nPoints = 5, bool anchorHits = true);

    CUDA_DEV bool passChiSquaredConstraint(struct modules& modulesInGPU, uint16_t& lowerModuleIndex1, uint16_t& lowerModuleIndex2, uint16_t& lowerModuleIndex3, uint16_t& lowerModuleIndex4, uint16_t& lowerModuleIndex5, float& chiSquared);

    CUDA_DEV float computeChiSquared(int nPoints, float* xs, float* ys, float* delta1, float* delta2, float* slopes, bool* isFlat, float g, float f, float radius);

    CUDA_DEV void runDeltaBetaIterationsT5(float& betaIn, float& betaOut, float& betaAv, float & pt_beta, float sdIn_dr, float sdOut_dr, float dr, float lIn);

    CUDA_DEV bool runQuintupletDefaultAlgoBBBB(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& innerOuterLowerModuleIndex, uint16_t& outerInnerLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex, unsigned int& fourthMDIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& dPhi, float& betaIn, float& betaOut, float& pt_beta, float& zLo, float& zHi, float& zLoPointed, float& zHiPointed, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut);

    CUDA_DEV bool runQuintupletDefaultAlgoBBEE(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& innerOuterLowerModuleIndex, uint16_t& outerInnerLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex, unsigned int& fourthMDIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& dPhi, float& betaIn, float& betaOut, float& pt_beta, float& zLo, float& rtLo, float& rtHi, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ);

    CUDA_DEV bool runQuintupletDefaultAlgoEEEE(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& innerOuterLowerModuleIndex, uint16_t& outerInnerLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex, unsigned int& fourthMDIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& dPhi, float& betaIn, float& betaOut, float& pt_beta, float& zLo, float& rtLo, float& rtHi, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ);

    CUDA_DEV bool runQuintupletDefaultAlgo(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& innerOuterLowerModuleIndex, uint16_t& outerInnerLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex, unsigned int& fourthMDIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& deltaPhi, float& betaIn, float& betaOut, float& pt_beta, float& zLo, float& zHi, float& rtLo, float& rtHi, float& zLoPointed, float& zHiPointed, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ);


    __global__ void createQuintupletsInGPUv2(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::quintuplets& quintupletsInGPU, struct SDL::objectRanges& rangesInGPU,uint16_t nEligibleT5Modules);

}
#endif
