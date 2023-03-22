#ifndef PixelTriplet_cuh
#define PixelTriplet_cuh

#include <alpaka/alpaka.hpp>

#include "Constants.cuh"
#include "EndcapGeometry.cuh"
#include "TiltedGeometry.h"
#include "Triplet.cuh"
#include "Segment.cuh"
#include "MiniDoublet.cuh"
#include "Module.cuh"
#include "Hit.cuh"
#include "PrintUtil.h"
#include "Quintuplet.cuh"

namespace SDL
{
    struct pixelTriplets //one pixel segment, one outer tracker triplet!
    {
        unsigned int* pixelSegmentIndices;         
        unsigned int* tripletIndices;
        unsigned int* nPixelTriplets; //size 1
        unsigned int* totOccupancyPixelTriplets; //size 1

        float* pixelRadiusError;
        float* rPhiChiSquared;
        float* rPhiChiSquaredInwards;
        float* rzChiSquared;

        FPX* pixelRadius;
        FPX* tripletRadius;
        FPX* pt;
        FPX* eta;
        FPX* phi;
        FPX* eta_pix;
        FPX* phi_pix;
        FPX* score;
        bool* isDup;
        bool* partOfPT5;

        uint8_t* logicalLayers;
        unsigned int* hitIndices;
        uint16_t* lowerModuleIndices;
        FPX* centerX;
        FPX* centerY;


        pixelTriplets();
        ~pixelTriplets();
        void freeMemory(cudaStream_t stream);
        void freeMemoryCache();
        void resetMemory(unsigned int maxPixelTriplets,cudaStream_t stream);
    };

    void createPixelTripletsInExplicitMemory(struct pixelTriplets& pixelTripletsinGPU, unsigned int maxPixelTriplets, cudaStream_t stream);

    ALPAKA_FN_ACC void addPixelTripletToMemory(struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct triplets& tripletsInGPU, struct pixelTriplets& pixelTripletsInGPU, unsigned int pixelSegmentIndex, unsigned int tripletIndex, float pixelRadius, float tripletRadius, float centerX, float centerY, float rPhiChiSquared, float rPhiChiSquaredInwards, float rzChiSquared, unsigned int pixelTripletIndex, float pt, float eta, float phi, float eta_pix, float phi_pix,float score);


    ALPAKA_FN_ACC float computeRadiusFromThreeAnchorHitspT3(float* xs, float* ys, float& g, float& f);

//    ALPAKA_FN_ACC void rmPixelTripletToMemory(struct pixelTriplets& pixelTripletsInGPU, unsigned int pixelTripletIndex);

    ALPAKA_FN_ACC bool runPixelTripletDefaultAlgo(struct modules& modulesInGPU, struct objectRanges& rangesInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct triplets& tripletsInGPU, unsigned int& pixelSegmentIndex, unsigned int tripletIndex, float& pixelRadius, float& pixelRadiusError, float& tripletRadius, float& centerX, float& centerY, float& rzChiSquared, float& rPhiChiSquared, float& rPhiChiSquaredInwards, bool runChiSquaredCuts = true);

    ALPAKA_FN_ACC bool passRadiusCriterion(struct modules& modulesInGPU, float& pixelRadius, float& pixelRadiusError, float& tripletRadius, uint16_t& lowerModuleIndex, uint16_t& middleModuleIndex, uint16_t& upperModuleIndex);

    ALPAKA_FN_ACC bool passRadiusCriterionBBB(float& pixelRadius, float& pixelRadiusError, float& tripletRadius);

    ALPAKA_FN_ACC bool passRadiusCriterionBBE(float& pixelRadius, float& pixelRadiusError, float& tripletRadius);

    ALPAKA_FN_ACC bool passRadiusCriterionBEE(float& pixelRadius, float& pixelRadiusError, float& tripletRadius);

    ALPAKA_FN_ACC bool passRadiusCriterionEEE(float& pixelRadius, float& pixelRadiusError, float& tripletRadius);


    ALPAKA_FN_ACC float computePT3RZChiSquared(struct modules& modulesInGPU, uint16_t* lowerModuleIndices, float* rtPix, float* xPix, float* yPix, float* zPix, float* rts, float* xs, float* ys, float* zs, float pixelSegmentPt, float pixelSegmentPx, float pixelSegmentPy, float pixelSegmentPz, int pixelSegmentCharge);

    ALPAKA_FN_ACC float computePT3RPhiChiSquared(struct modules& modulesInGPU, uint16_t* lowerModuleIndices, float& g, float& f, float& radius, float* xs, float* ys);

    ALPAKA_FN_ACC float computePT3RPhiChiSquaredInwards(struct modules& modulesInGPU, float& g, float& f, float& r, float* xPix, float* yPix);

    ALPAKA_FN_ACC bool passPT3RZChiSquaredCuts(struct modules& modulesInGPU, uint16_t& lowerModuleIndex1, uint16_t& lowerModuleIndex2, uint16_t& lowerModuleIndex3, float& rzChiSquared);

    ALPAKA_FN_ACC bool passPT3RPhiChiSquaredCuts(struct modules& mdoulesInGPU, uint16_t& lowerModuleIndex1, uint16_t& lowerModuleIndex2, uint16_t& lowerModuleIndex3, float& rPhiChiSquared);

    ALPAKA_FN_ACC bool passPT3RPhiChiSquaredInwardsCuts(struct modules& modulesInGPU, uint16_t& lowerModuleIndex1, uint16_t& lowerModuleIndex2, uint16_t& lowerModuleIndex3, float& rPhiChiSquared);

    __global__ void createPixelTripletsInGPUFromMapv2(struct SDL::modules& modulesInGPU, struct SDL::objectRanges& rangesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::pixelTriplets& pixelTripletsInGPU, unsigned int* connectedPixelSize, unsigned int* connectedPixelIndex, unsigned int nPixelSegments);

    ALPAKA_FN_ACC void runDeltaBetaIterationspT3(float& betaIn, float& betaOut, float& betaAv, float & pt_beta, float sdIn_dr, float sdOut_dr, float dr, float lIn); 
    ALPAKA_FN_ACC float computeChiSquaredpT3(int nPoints, float* xs, float* ys, float* delta1, float* delta2, float* slopes, bool* isFlat, float g, float f, float radius);

    ALPAKA_FN_ACC bool inline runTripletDefaultAlgoPPBB(struct modules& modulesInGPU, struct objectRanges& rangesInGPU, struct miniDoublets& mdsInGPU ,struct segments& segmentsInGPU, uint16_t& pixelModuleIndex, uint16_t& outerInnerLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int thirdMDIndex, unsigned int& fourthMDIndex, float& z_OutLo, float& rt_OutLo, float& dPhiPos, float& dPhi, float& betaIn, float& betaOut, float& pt_beta, float& zLo, float& zHi, float& zLoPointed, float& zHiPointed, float& sdlCut, float& betaOutCut, float& deltaBetaCut);

    ALPAKA_FN_ACC bool inline runTripletDefaultAlgoPPEE(struct modules& modulesInGPU, struct objectRanges& rangesInGPU, struct miniDoublets& mdsInGPU ,struct segments& segmentsInGPU, uint16_t& pixelModuleIndex, uint16_t& outerInnerLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex, unsigned int& fourthMDIndex, float& z_OutLo, float& rt_OutLo, float& deltaPhiPos, float& dPhi, float& betaIn,float& betaOut, float& pt_beta, float& zLo, float& rtLo, float& rtHi, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ);

    ALPAKA_FN_ACC bool inline runPixelTrackletDefaultAlgopT3(struct modules& modulesInGPU, struct objectRanges& rangesInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, uint16_t& pixelLowerModuleIndex, uint16_t& outerInnerLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& deltaPhi, float& betaIn, float& betaOut, float& pt_beta, float& zLo, float& zHi, float& rtLo, float& rtHi, float& zLoPointed, float& zHiPointed, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ);
    
    ALPAKA_FN_ACC bool checkIntervalOverlappT3(const float& firstMin, const float& firstMax, const float& secondMin, const float& secondMax);
}
#endif
#ifndef PixelQuintuplet_cuh
#define PixelQuintuplet_cuh

#include "Constants.cuh"
#include "EndcapGeometry.cuh"
#include "TiltedGeometry.h"
#include "Segment.cuh"
#include "MiniDoublet.cuh"
#include "Module.cuh"
#include "Hit.cuh"
#include "PrintUtil.h"
#include "Triplet.cuh"
#include "Quintuplet.cuh"
#include "PixelTriplet.cuh"

namespace SDL
{
    struct pixelQuintuplets
    {
        unsigned int* pixelIndices;
        unsigned int* T5Indices;
        unsigned int* nPixelQuintuplets;
        unsigned int* totOccupancyPixelQuintuplets;
        bool* isDup;
        FPX* score;
        FPX* eta;
        FPX* phi;
        uint8_t* logicalLayers;
        unsigned int* hitIndices;
        uint16_t* lowerModuleIndices;
        FPX* pixelRadius;
        FPX* quintupletRadius;
        FPX* centerX;
        FPX* centerY;
        float* rzChiSquared;
        float* rPhiChiSquared;
        float* rPhiChiSquaredInwards;

        pixelQuintuplets();
        ~pixelQuintuplets();
        void freeMemory(cudaStream_t stream);
        void freeMemoryCache();
        void resetMemory(unsigned int maxPixelQuintuplets,cudaStream_t stream);

    };

    void createPixelQuintupletsInExplicitMemory(struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, unsigned int maxPixelQuintuplets, cudaStream_t stream);

    ALPAKA_FN_ACC void addPixelQuintupletToMemory(struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct quintuplets& quintupletsInGPU, struct pixelQuintuplets& pixelQuintupletsInGPU, unsigned int pixelIndex, unsigned int T5Index, unsigned int pixelQuintupletIndex, float& rzChiSquared, float& rPhiChiSquared, float& rPhiChiSquaredInwards, float score, float eta, float phi, float& pixelRadius, float& quintupletRadius,
        float& centerX, float& centerY);

    ALPAKA_FN_ACC bool runPixelQuintupletDefaultAlgo(struct modules& modulesInGPU, struct objectRanges& rangesInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct triplets& tripletsInGPU, struct quintuplets& quintupletsInGPU, unsigned int& pixelSegmentIndex, unsigned int& quintupletIndex, float& rzChiSquared, float& rPhiChiSquared, float& rPhiChiSquaredInwards, float& pixelRadius, float& quintupletRadius, float& centerX, float& centerY, unsigned int pixelSegmentArrayIndex);

    ALPAKA_FN_ACC float computePT5RZChiSquared(struct modules& modulesInGPU, uint16_t* lowerModuleIndices, float* rtPix, float* zPix, float* rts, float* zs);

    ALPAKA_FN_ACC bool passPT5RZChiSquaredCuts(struct modules& modulesInGPU, uint16_t& lowerModuleIndex1, uint16_t& lowerModuleIndex2, uint16_t& lowerModuleIndex3, uint16_t& lowerModuleIndex4, uint16_t& lowerModuleIndex5, float& rzChiSquared);

    ALPAKA_FN_ACC float computePT5RPhiChiSquared(struct modules& modulesInGPU, uint16_t* lowerModuleIndices, float& g, float& f, float& radius, float* xs, float* ys);

    ALPAKA_FN_ACC float computePT5RPhiChiSquaredInwards(struct modules& modulesInGPU, float& g, float& f, float& r, float* xPix, float* yPix);

    ALPAKA_FN_ACC bool passPT5RPhiChiSquaredInwardsCuts(struct modules& modulesInGPU, uint16_t& lowerModuleIndex1, uint16_t& lowerModuleIndex2, uint16_t& lowerModuleIndex3, uint16_t& lowerModuleIndex4, uint16_t& lowerModuleIndex5, float rPhiChiSquared);

    ALPAKA_FN_ACC bool passPT5RPhiChiSquaredCuts(struct modules& modulesInGPU, uint16_t& lowerModuleIndex1, uint16_t& lowerModuleIndex2, uint16_t& lowerModuleIndex3, uint16_t& lowerModuleIndex4, uint16_t& lowerModuleIndex5, float rPhiChiSquared);
    ALPAKA_FN_ACC void computeSigmasForRegression_pT5(SDL::modules& modulesInGPU, const uint16_t* lowerModuleIndices, float* delta1, float* delta2, float* slopes, bool* isFlat, int nPoints = 5, bool anchorHits = true);
    
    __global__ void createPixelQuintupletsInGPUFromMapv2(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::quintuplets& quintupletsInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, unsigned int* connectedPixelSize, unsigned int* connectedPixelIndex, unsigned int nPixelSegments,struct SDL::objectRanges& rangesInGPU);

    ALPAKA_FN_ACC void runDeltaBetaIterationspT5(float& betaIn, float& betaOut, float& betaAv, float & pt_beta, float sdIn_dr, float sdOut_dr, float dr, float lIn);

    ALPAKA_FN_ACC float computeChiSquaredpT5(int nPoints, float* xs, float* ys, float* delta1, float* delta2, float* slopes, bool* isFlat, float g, float f, float radius);
    ALPAKA_FN_ACC bool checkIntervalOverlappT5(const float& firstMin, const float& firstMax, const float& secondMin, const float& secondMax);
 
    ALPAKA_FN_ACC bool inline runpT5DefaultAlgoPPBB(struct modules& modulesInGPU, struct objectRanges& rangesInGPU, struct miniDoublets& mdsInGPU ,struct segments& segmentsInGPU, uint16_t& pixelModuleIndex, uint16_t& outerInnerLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int thirdMDIndex, unsigned int& fourthMDIndex, float& z_OutLo, float& rt_OutLo, float& dPhiPos, float& dPhi, float& betaIn,
        float& betaOut, float& pt_beta, float& zLo, float& zHi, float& zLoPointed, float& zHiPointed, float& sdlCut, float& betaOutCut, float& deltaBetaCut);
    ALPAKA_FN_ACC bool inline runpT5DefaultAlgoPPEE(struct modules& modulesInGPU, struct objectRanges& rangesInGPU, struct miniDoublets& mdsInGPU ,struct segments& segmentsInGPU, uint16_t& pixelModuleIndex, uint16_t& outerInnerLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex, unsigned int& fourthMDIndex, float& z_OutLo, float& rt_OutLo, float& deltaPhiPos, float& dPhi, float& betaIn,
        float& betaOut, float& pt_beta, float& zLo, float& rtLo, float& rtHi, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ);
    ALPAKA_FN_ACC bool inline runpT5DefaultAlgo(struct modules& modulesInGPU, struct objectRanges& rangesInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, uint16_t& pixelLowerModuleIndex, uint16_t& outerInnerLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& deltaPhi, float& betaIn, float& betaOut, float& pt_beta, float& zLo, float& zHi, float& rtLo, float& rtHi, float& zLoPointed, float& zHiPointed, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ); 
}

#endif
