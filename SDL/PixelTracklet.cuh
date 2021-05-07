#ifndef Pixel_Tracklet_cuh
#define Pixel_Tracklet_cuh

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
#include "Tracklet.cuh"


namespace SDL
{
    struct pixelTracklets
    {
        unsigned int* segmentIndices;
        unsigned int* lowerModuleIndices;
        unsigned int* nPixelTracklets;
        float* zOut;
        float* rtOut;

        float* deltaPhiPos;
        float* deltaPhi;
        float* betaIn;
        float* betaOut;
        float* pt_beta;

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
        
        pixelTracklets();
        ~pixelTracklets();
        void freeMemory();
        void freeMemoryCache();
    };

    void createPixelTrackletsInUnifiedMemory(struct pixelTracklets& pixelTrackletsInGPU, unsigned int maxPixelTracklets);
    void createPixelTrackletsInExplicitMemory(struct pixelTracklets& pixelTrackletsInGPU, unsigned int maxPixelTracklets);

#ifdef CUT_VALUE_DEBUG
    CUDA_DEV void addPixelTrackletToMemory(struct pixelTracklets& pixelTrackletsInGPU, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, unsigned int innerInnerLowerModuleIndex, unsigned int innerOuterLowerModuleIndex, unsigned int outerInnerLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& deltaPhi, float& betaIn, float& betaOut, float pt_beta, float& zLo, float& zHi, float& rtLo, float& rtHi, float& zLoPointed, float&
        zHiPointed, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ, unsigned int pixelTrackletIndex);

#else
    CUDA_DEV void addPixelTrackletToMemory(struct pixelTracklets& pixelTrackletsInGPU, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, unsigned int innerInnerLowerModuleIndex, unsigned int innerOuterLowerModuleIndex, unsigned int outerInnerLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& deltaPhi, float& betaIn, float& betaOut, float pt_beta, unsigned int pixelTrackletIndex);
#endif

    CUDA_DEV bool runPixelTrackletDefaultAlgo(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, unsigned int innerInnerLowerModuleIndex, unsigned int innerOuterLowerModuleIndex, unsigned int outerInnerLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& deltaPhi, float& betaIn, float&
        betaOut, float& pt_beta, float& zLo, float& zHi, float& rtLo, float& rtHi, float& zLoPointed, float& zHiPointed, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ, unsigned int N_MAX_SEGMENTS_PER_MODULE = 600);


    CUDA_DEV bool runTrackletDefaultAlgoPPBB(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU ,struct segments& segmentsInGPU, unsigned int pixelModuleIndex, unsigned int outerInnerLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, float& zOut, float& rtOut, float& dPhiPos, float& dPhi, float& betaIn, float& betaOut, float& pt_beta, unsigned int
        N_MAX_SEGMENTS_PER_MODULE, float& zLo, float& zHi, float& zLoPointed, float& zHiPointed, float& sdlCut, float& betaOutCut, float& deltaBetaCut); // pixel to BB and BE segments

    CUDA_DEV bool runTrackletDefaultAlgoPPEE(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU ,struct segments& segmentsInGPU, unsigned int pixelModuleIndex, unsigned int outerInnerLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& dPhi, float& betaIn, float& betaOut, float& pt_beta, unsigned int
        N_MAX_SEGMENTS_PER_MODULE,  float& zLo, float& rtLo, float& rtHi, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ); // pixel to EE segments

}
#endif
