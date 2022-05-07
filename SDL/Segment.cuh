#ifndef Segment_cuh
#define Segment_cuh

#ifdef __CUDACC__
#define CUDA_HOSTDEV  __host__ __device__
#define CUDA_DEV __device__
#else
#define CUDA_HOSTDEV
#define CUDA_DEV
#endif

#include "Constants.cuh"
#include "EndcapGeometry.h"
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
        int* superbin;
        int8_t* pixelType;
        bool* isQuad;
        bool* isDup;
        float* score;
        float* circleCenterX;
        float* circleCenterY;
        float* circleRadius;
        bool* partOfPT5;

#ifdef CUT_VALUE_DEBUG
        float* zIns;
        float* zOuts;
        float* rtIns;
        float* rtOuts;
        float* dAlphaInnerMDSegments;
        float* dAlphaOuterMDSegments;
        float* dAlphaInnerMDOuterMDs;
        float* zLo;
        float* zHi;
        float* rtLo;
        float* rtHi;
        float* sdCut;
        float* dAlphaInnerMDSegmentThreshold;
        float* dAlphaOuterMDSegmentThreshold;
        float* dAlphaInnerMDOuterMDThreshold;
#endif

        segments();
        ~segments();
	void freeMemory(cudaStream_t stream);
	void freeMemoryCache();
  void resetMemory(unsigned int maxSegments, unsigned int nModules, unsigned int maxPixelSegments,cudaStream_t stream);
    };

    void createSegmentsInUnifiedMemory(struct segments& segmentsInGPU, unsigned int maxSegments, uint16_t nLowerModules, unsigned int maxPixelSegments ,cudaStream_t stream);
    void createSegmentsInExplicitMemory(struct segments& segmentsInGPU, unsigned int maxSegments, uint16_t nLowerModules, unsigned int maxPixelSegments,cudaStream_t stream);

    void createSegmentArrayRanges(struct modules& modulesInGPU, struct objectRanges& rangesInGPU, struct miniDoublets& mdsinGPU, uint16_t& nLowerModules, unsigned int& nSegments, cudaStream_t stream, const uint16_t& maxSegmentsPerModule, const uint16_t& maxPixelSegments);

    CUDA_DEV void dAlphaThreshold(float* dAlphaThresholdValues, short& innerSubdet, short& innerLayer, short& innerSide, short& innerRod, short& innerRing, float& drdzInner, ModuleType& innerModuleType, short* subdets, short* layers, short* sides, short* rods, short* rings, float* drdzs, struct SDL::miniDoublets& mdsInGPU, float& xIn, float& yIn, float& zIn, float& rtIn, float& xOut, float& yOut, float& zOut, float& rtOut, uint16_t& outerLowerModuleArrayIdx, unsigned int& innerMDIndex, unsigned int& outerMDIndex);

#ifdef CUT_VALUE_DEBUG
    CUDA_DEV void addSegmentToMemory(struct segments& segmentsInGPU, unsigned int lowerMDIndex, unsigned int upperMDIndex, uint16_t innerLowerModuleIndex, uint16_t outerLowerModuleIndex, unsigned int innerMDAnchorHitIndex, unsigned int outerMDAnchorHitIndex, float& dPhi, float& dPhiMin, float& dPhiMax, float& dPhiChange, float& dPhiChangeMin, float& dPhiChangeMax, float& zIn, float& zOut, float& rtIn, float& rtOut, float& dAlphaInnerMDSegment, float& dAlphaOuterMDSegment, float&
        dAlphaInnerMDOuterMD, float& zLo, float& zHi, float& rtLo, float& rtHi, float& sdCut, float& dAlphaInnerMDSegmentThreshold, float& dAlphaOuterMDSegmentThreshold, float& dAlphaInnerMDOuterMDThreshold, unsigned int idx);
#else
    CUDA_DEV void addSegmentToMemory(struct segments& segmentsInGPU, unsigned int lowerMDIndex, unsigned int upperMDIndex, uint16_t innerLowerModuleIndex, uint16_t outerLowerModuleIndex, unsigned int innerMDAnchorHitIndex, unsigned int outerMDAnchorHitIndex, float& dPhi, float& dPhiMin, float& dPhiMax, float& dPhiChange, float& dPhiChangeMin, float& dPhiChangeMax, unsigned int idx);
#endif

//    CUDA_DEV void rmPixelSegmentFromMemory(struct segments& segmentsInGPU, unsigned int pixelSegmentArrayIndex);


    CUDA_DEV void addPixelSegmentToMemory(struct segments& segmentsInGPU, struct miniDoublets& mdsInGPU, struct modules& modulesInGPU, unsigned int innerMDIndex, unsigned int outerMDIndex, uint16_t pixelModuleIndex, unsigned int innerAnchorHitIndex, unsigned int outerAnchorHitIndex, float dPhiChange, float ptIn, float ptErr, float px, float py, float pz, float etaErr, float eta, float phi, unsigned int idx, unsigned int pixelSegmentArrayIndex, int superbin,
            int8_t pixelType, short isQuad, float score);

    CUDA_DEV bool runSegmentDefaultAlgo(struct miniDoublets& mdsInGPU, short& innerSubdet, short& innerLayer, short& innerSide, short& innerRod, short& innerRing, float& innerDrdz, ModuleType& innerModuleType, short* subdets, short*  layers, short* sides, short* rods, short* rings, ModuleType* moduleTypes, float* drdzs, unsigned int& innerMDIndex, unsigned int& outerMDIndex, uint16_t outerLowerModuleArrayIdx, float& zIn, float& zOut, float& rtIn, float& rtOut, float& dPhi, float& dPhiMin, float& dPhiMax, float& dPhiChange, float& dPhiChangeMin, float& dPhiChangeMax, float& dAlphaInnerMDSegment, float& dAlphaOuterMDSegment, float&
        dAlphaInnerMDOuterMD, float& zLo, float& zHi, float& rtLo, float& rtHi, float& sdCut, float& dAlphaInnerMDSegmentThreshold, float& dAlphaOuterMDSegmentThreshold, float& dAlphaInnerMDOuterMDThreshold);

    CUDA_DEV bool runSegmentDefaultAlgoBarrel(struct miniDoublets& mdsInGPU, short& innerSubdet, short& innerLayer, short& innerSide, short& innerRod, short& innerRing, float& innerDrdz, ModuleType& innerModuleType, short* subdets, short*  layers, short* sides, short* rods, short* rings, ModuleType* moduleType, float* drdzs, unsigned int& innerMDIndex, unsigned int& outerMDIndex, uint16_t outerLowerModuleArrayIdx, float& zIn, float& zOut, float& rtIn, float& rtOut, float& dPhi, float& dPhiMin, float& dPhiMax, float& dPhiChange, float& dPhiChangeMin, float& dPhiChangeMax, float& dAlphaInnerMDSegment, float& dAlphaOuterMDSegment, float&
        dAlphaInnerMDOuterMD, float& zLo, float& zHi, float& sdCut, float& dAlphaInnerMDSegmentThreshold, float& dAlphaOuterMDSegmentThreshold, float& dAlphaInnerMDOuterMDThreshold);

    CUDA_DEV bool runSegmentDefaultAlgoEndcap(struct miniDoublets& mdsInGPU, short& innerSubdet, short& innerLayer, short& innerSide, short& innerRod, short& innerRing, float& innerDrdz, ModuleType& innerModuleType, short* subdets, short*  layers, short* sides, short* rods, short* rings, ModuleType* moduleType, float* drdzs, unsigned int& innerMDIndex, unsigned int& outerMDIndex, uint16_t outerLowerModuleArrayIdx, float& zIn, float& zOut, float& rtIn, float& rtOut, float& dPhi, float& dPhiMin, float& dPhiMax, float& dPhiChange, float& dPhiChangeMin, float& dPhiChangeMax, float& dAlphaInnerMDSegment, float& dAlphaOuterMDSegment, float&
        dAlphaInnerMDOuterMD, float& rtLo, float& rtHi, float& sdCut, float& dAlphaInnerMDSegmentThreshold, float& dAlphaOuterMDSegmentThreshold, float& dAlphaInnerMDOuterMDThreshold);


    void printSegment(struct segments& segmentsInGPU, struct miniDoublets& mdsInGPU, struct hits& hitsInGPU, struct modules& modulesInGPU, unsigned int segmentIndex);
    CUDA_DEV float moduleGapSize_seg(struct modules& modulesInGPU, unsigned int moduleIndex);
    
    CUDA_DEV extern inline float isTighterTiltedModules_seg(struct modules& modulesInGPU, unsigned int moduleIndex);
    CUDA_DEV inline float isTighterTiltedModules_seg(short subdet, short layer, short side, short rod);

    CUDA_DEV float moduleGapSize_seg(struct modules& modulesInGPU, unsigned int moduleIndex);
    CUDA_DEV float moduleGapSize_seg(short layer, short ring, short subdet, short side, short rod);

__global__ void createSegmentsInGPUv2(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::objectRanges& rangesInGPU);
}

#endif
