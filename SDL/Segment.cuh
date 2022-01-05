#ifndef Segment_cuh
#define Segment_cuh

#ifdef __CUDACC__
#define CUDA_HOSTDEV  __host__ __device__
#define CUDA_DEV __device__
#else
#define CUDA_HOSTDEV
#define CUDA_DEV
#endif

#include "Constants.h"
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
        unsigned int* mdIndices;
        unsigned int* innerLowerModuleIndices;
        unsigned int* outerLowerModuleIndices;
        unsigned int* innerMiniDoubletAnchorHitIndices;
        unsigned int* outerMiniDoubletAnchorHitIndices;
        
        unsigned int* nSegments; //number of segments per inner lower module
        //float* dPhis;
        //float* dPhiMins;
        //float* dPhiMaxs;
        //float* dPhiChanges;
        //float* dPhiChangeMins;
        //float* dPhiChangeMaxs;
        FPX_dPhi* dPhis;
        FPX_dPhi* dPhiMins;
        FPX_dPhi* dPhiMaxs;
        FPX_dPhi* dPhiChanges;
        FPX_dPhi* dPhiChangeMins;
        FPX_dPhi* dPhiChangeMaxs;


        //not so optional pixel dudes
        FPX_seg* ptIn;
        FPX_seg* ptErr;
        FPX_seg* px;
        FPX_seg* py;
        FPX_seg* pz;
        FPX_seg* etaErr;
        FPX_seg* eta;
        FPX_seg* phi;
        FPX_seg* score;
        //float* ptIn;
        //float* ptErr;
        //float* px;
        //float* py;
        //float* pz;
        //float* etaErr;
        //float* eta;
        //float* phi;
        //float* score;
        int* superbin;
        int* pixelType;
        bool* isQuad;
        bool* isDup;
        FPX_circle* circleCenterX;
        FPX_circle* circleCenterY;
        FPX_circle* circleRadius;
        //float* circleCenterX;
        //float* circleCenterY;
        //float* circleRadius;
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

    void createSegmentsInUnifiedMemory(struct segments& segmentsInGPU, unsigned int maxSegments, unsigned int nModules, unsigned int maxPixelSegments ,cudaStream_t stream);
    void createSegmentsInExplicitMemory(struct segments& segmentsInGPU, unsigned int maxSegments, unsigned int nModules, unsigned int maxPixelSegments,cudaStream_t stream);

    CUDA_DEV void dAlphaThreshold(float* dAlphaThresholdValues, struct hits& hitsInGPU, struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, unsigned int& innerMiniDoubletAnchorHitIndex, unsigned int& outerMiniDoubletAnchorHitIndex, unsigned int& innerLowerModuleIndex, unsigned int& outerLowerModuleIndex, unsigned int& innerMDIndex, unsigned int& outerMDIndex);

#ifdef CUT_VALUE_DEBUG
    CUDA_DEV void addSegmentToMemory(struct segments& segmentsInGPU, unsigned int lowerMDIndex, unsigned int upperMDIndex, unsigned int innerLowerModuleIndex, unsigned int outerLowerModuleIndex, unsigned int innerMDAnchorHitIndex, unsigned int outerMDAnchorHitIndex, float& dPhi, float& dPhiMin, float& dPhiMax, float& dPhiChange, float& dPhiChangeMin, float& dPhiChangeMax, float& zIn, float& zOut, float& rtIn, float& rtOut, float& dAlphaInnerMDSegment, float& dAlphaOuterMDSegment, float&
        dAlphaInnerMDOuterMD, float& zLo, float& zHi, float& rtLo, float& rtHi, float& sdCut, float& dAlphaInnerMDSegmentThreshold, float& dAlphaOuterMDSegmentThreshold, float& dAlphaInnerMDOuterMDThreshold, unsigned int idx);
#else
    CUDA_DEV void addSegmentToMemory(struct segments& segmentsInGPU, unsigned int lowerMDIndex, unsigned int upperMDIndex, unsigned int innerLowerModuleIndex, unsigned int outerLowerModuleIndex, unsigned int innerMDAnchorHitIndex, unsigned int outerMDAnchorHitIndex, float& dPhi, float& dPhiMin, float& dPhiMax, float& dPhiChange, float& dPhiChangeMin, float& dPhiChangeMax, unsigned int idx);
#endif

    CUDA_DEV void rmPixelSegmentFromMemory(struct segments& segmentsInGPU, unsigned int pixelSegmentArrayIndex);
    CUDA_DEV void addPixelSegmentToMemory(struct segments& segmentsInGPU, struct miniDoublets& mdsInGPU, struct hits& hitsInGPU, struct modules& modulesInGPU, unsigned int innerMDIndex, unsigned int outerMDIndex, unsigned int pixelModuleIndex, unsigned int innerAnchorHitIndex, unsigned int outerAnchorHitIndex, float dPhiChange, float ptIn, float ptErr, float px, float py, float pz, float etaErr, float eta, float phi, unsigned int idx, unsigned int pixelSegmentArrayIndex, int superbin, int pixelType, short isQuad, float score);

    CUDA_DEV bool runSegmentDefaultAlgo(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, unsigned int& innerLowerModuleIndex, unsigned int& outerLowerModuleIndex, unsigned int& innerMDIndex, unsigned int& outerMDIndex, float& zIn, float& zOut, float& rtIn, float& rtOut, float& dPhi, float& dPhiMin, float& dPhiMax, float& dPhiChange, float& dPhiChangeMin, float& dPhiChangeMax, float& dAlphaInnerMDSegment, float& dAlphaOuterMDSegment, float&
        dAlphaInnerMDOuterMD, float& zLo, float& zHi, float& rtLo, float& rtHi, float& sdCut, float& dAlphaInnerMDSegmentThreshold, float& dAlphaOuterMDSegmentThreshold, float& dAlphaInnerMDOuterMDThreshold, unsigned int& innerMiniDoubletAnchorHitIndex, unsigned int& outerMiniDoubletAnchorHitIndex);


    CUDA_DEV bool runSegmentDefaultAlgoBarrel(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, unsigned int& innerLowerModuleIndex, unsigned int& outerLowerModuleIndex, unsigned int& innerMDIndex, unsigned int& outerMDIndex, float& zIn, float& zOut, float& rtIn, float& rtOut, float& dPhi, float& dPhiMin, float& dPhiMax, float& dPhiChange, float& dPhiChangeMin, float& dPhiChangeMax, float& dAlphaInnerMDSegment, float& dAlphaOuterMDSegment, float&
        dAlphaInnerMDOuterMD, float& zLo, float& zHi, float& sdCut, float& dAlphaInnerMDSegmentThreshold, float& dAlphaOuterMDSegmentThreshold, float& dAlphaInnerMDOuterMDThreshold, unsigned int& innerMiniDoubletAnchorHitIndex, unsigned int& outerMiniDoubletAnchorHitIndex);

    CUDA_DEV bool runSegmentDefaultAlgoEndcap(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, unsigned int& innerLowerModuleIndex, unsigned int& outerLowerModuleIndex, unsigned int& innerMDIndex, unsigned int& outerMDIndex, float& zIn, float& zOut, float& rtIn, float& rtOut, float& dPhi, float& dPhiMin, float& dPhiMax, float& dPhiChange, float& dPhiChangeMin, float& dPhiChangeMax, float& dAlphaInnerMDSegment, float& dAlphaOuterMDSegment,
        float& rtLo, float& rtHi, float& sdCut, float& dAlphaInnerMDSegmentThreshold, float& dAlphaOuterMDSegmentThreshold, float& dAlphaInnerMDOuterMDThreshold, float&
        dAlphaInnerMDOuterMD, unsigned int& innerMiniDoubletAnchorHitIndex, unsigned int& outerMiniDoubletAnchorHitIndex);

    void printSegment(struct segments& segmentsInGPU, struct miniDoublets& mdsInGPU, struct hits& hitsInPGU, struct modules& modulesInGPU, unsigned int segmentIndex);
    CUDA_DEV float moduleGapSize_seg(struct modules& modulesInGPU, unsigned int moduleIndex);
    
    CUDA_DEV extern inline float isTighterTiltedModules_seg(struct modules& modulesInGPU, unsigned int moduleIndex);
    CUDA_DEV float moduleGapSize_seg(struct modules& modulesInGPU, unsigned int moduleIndex);

}

#endif
