#ifndef Segment_cuh
#define Segment_cuh

#include <alpaka/alpaka.hpp>
#ifdef __CUDACC__
#else
#endif

#include "Constants.cuh"
#include "EndcapGeometry.cuh"
#include "TiltedGeometry.h"
#include "MiniDoublet.cuh"
#include "Module.cuh"
#include "Hit.cuh"
#include "PrintUtil.h"

//CUDA MATH API
#include "math.h"

namespace SDL
{
    struct segments
    {
        unsigned int* nMemoryLocations;

        unsigned int* mdIndices;
        uint16_t* innerLowerModuleIndices;
        uint16_t* outerLowerModuleIndices;
        unsigned int* innerMiniDoubletAnchorHitIndices;
        unsigned int* outerMiniDoubletAnchorHitIndices;
        
        unsigned int* nSegments; //number of segments per inner lower module
        unsigned int* totOccupancySegments; //number of segments per inner lower module
        FPX* dPhis;
        FPX* dPhiMins;
        FPX* dPhiMaxs;
        FPX* dPhiChanges;
        FPX* dPhiChangeMins;
        FPX* dPhiChangeMaxs;


        //not so optional pixel dudes
        float* ptIn;
        float* ptErr;
        float* px;
        float* py;
        float* pz;
        float* etaErr;
        float* eta;
        float* phi;
        int* charge;
        unsigned int* seedIdx;
        int* superbin;
        int8_t* pixelType;
        bool* isQuad;
        bool* isDup;
        float* score;
        float* circleCenterX;
        float* circleCenterY;
        float* circleRadius;
        bool* partOfPT5;
        uint4* pLSHitsIdxs;


        segments();
        ~segments();
	void freeMemory(cudaStream_t stream);
	void freeMemoryCache();
    void resetMemory(unsigned int nMemoryLocationsx, unsigned int nModules, unsigned int maxPixelSegments,cudaStream_t stream);
    };

    void createSegmentsInExplicitMemory(struct segments& segmentsInGPU, unsigned int maxSegments, uint16_t nLowerModules, unsigned int maxPixelSegments,cudaStream_t stream);

    __global__ void createSegmentArrayRanges(struct modules& modulesInGPU, struct objectRanges& rangesInGPU, struct miniDoublets& mdsinGPU);


    __global__ void addSegmentRangesToEventExplicit(struct modules& modulesInGPU, struct segments& segmentsInGPU, struct objectRanges& rangesInGPU);
    
    ALPAKA_FN_ACC void dAlphaThreshold(float* dAlphaThresholdValues, struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, float& xIn, float& yIn, float& zIn, float& rtIn, float& xOut, float& yOut, float& zOut, float& rtOut, uint16_t& innerLowerModuleIndex, uint16_t& outerLowerModuleIndex, unsigned int& innerMDIndex, unsigned int& outerMDIndex);
    ALPAKA_FN_ACC void addSegmentToMemory(struct segments& segmentsInGPU, unsigned int lowerMDIndex, unsigned int upperMDIndex, uint16_t innerLowerModuleIndex, uint16_t outerLowerModuleIndex, unsigned int innerMDAnchorHitIndex, unsigned int outerMDAnchorHitIndex, float& dPhi, float& dPhiMin, float& dPhiMax, float& dPhiChange, float& dPhiChangeMin, float& dPhiChangeMax, unsigned int idx);

//    ALPAKA_FN_ACC void rmPixelSegmentFromMemory(struct segments& segmentsInGPU, unsigned int pixelSegmentArrayIndex);


    ALPAKA_FN_ACC void addPixelSegmentToMemory(struct segments& segmentsInGPU, struct miniDoublets& mdsInGPU, struct modules& modulesInGPU, unsigned int innerMDIndex, unsigned int outerMDIndex, uint16_t pixelModuleIndex, unsigned int hitIdxs[4], unsigned int innerAnchorHitIndex, unsigned int outerAnchorHitIndex, float dPhiChange, float ptIn, float ptErr, float px, float py, float pz, float etaErr, float eta, float phi, int charge, unsigned int seedIdx, unsigned int idx, unsigned int pixelSegmentArrayIndex, int superbin, int8_t pixelType, short isQuad, float score);

    ALPAKA_FN_ACC bool runSegmentDefaultAlgo(struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, uint16_t& innerLowerModuleIndex, uint16_t& outerLowerModuleIndex, unsigned int& innerMDIndex, unsigned int& outerMDIndex, float& zIn, float& zOut, float& rtIn, float& rtOut, float& dPhi, float& dPhiMin, float& dPhiMax, float& dPhiChange, float& dPhiChangeMin, float& dPhiChangeMax, float& dAlphaInnerMDSegment, float& dAlphaOuterMDSegment, float& dAlphaInnerMDOuterMD, float& zLo, float& zHi, float& rtLo, float& rtHi, float& sdCut, float& dAlphaInnerMDSegmentThreshold, float& dAlphaOuterMDSegmentThreshold, float& dAlphaInnerMDOuterMDThreshold);

    ALPAKA_FN_ACC bool runSegmentDefaultAlgoBarrel(struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, uint16_t& innerLowerModuleIndex, uint16_t& outerLowerModuleIndex, unsigned int& innerMDIndex, unsigned int& outerMDIndex, float& zIn, float& zOut, float& rtIn, float& rtOut, float& dPhi, float& dPhiMin, float& dPhiMax, float& dPhiChange, float& dPhiChangeMin, float& dPhiChangeMax, float& dAlphaInnerMDSegment, float& dAlphaOuterMDSegment, float& dAlphaInnerMDOuterMD, float& zLo, float& zHi, float& sdCut, float& dAlphaInnerMDSegmentThreshold, float& dAlphaOuterMDSegmentThreshold, float& dAlphaInnerMDOuterMDThreshold);

    ALPAKA_FN_ACC bool runSegmentDefaultAlgoEndcap(struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, uint16_t& innerLowerModuleIndex, uint16_t& outerLowerModuleIndex, unsigned int& innerMDIndex, unsigned int& outerMDIndex, float& zIn, float& zOut, float& rtIn, float& rtOut, float& dPhi, float& dPhiMin, float& dPhiMax, float& dPhiChange, float& dPhiChangeMin, float& dPhiChangeMax, float& dAlphaInnerMDSegment, float& dAlphaOuterMDSegment, float& rtLo, float& rtHi, float& sdCut, float& dAlphaInnerMDSegmentThreshold, float& dAlphaOuterMDSegmentThreshold, float& dAlphaInnerMDOuterMDThreshold, float& dAlphaInnerMDOuterMD);

    void printSegment(struct segments& segmentsInGPU, struct miniDoublets& mdsInGPU, struct hits& hitsInGPU, struct modules& modulesInGPU, unsigned int segmentIndex);
    ALPAKA_FN_ACC float moduleGapSize_seg(struct modules& modulesInGPU, unsigned int moduleIndex);
    
    ALPAKA_FN_ACC extern ALPAKA_FN_INLINE float isTighterTiltedModules_seg(struct modules& modulesInGPU, unsigned int moduleIndex);
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float isTighterTiltedModules_seg(short subdet, short layer, short side, short rod);

    ALPAKA_FN_ACC float moduleGapSize_seg(struct modules& modulesInGPU, unsigned int moduleIndex);
    ALPAKA_FN_ACC float moduleGapSize_seg(short layer, short ring, short subdet, short side, short rod);

__global__ void createSegmentsInGPUv2(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::objectRanges& rangesInGPU);
}

#endif
