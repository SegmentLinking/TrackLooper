#ifndef Tracklet_h
#define Tracklet_h

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

namespace SDL
{
    struct tracklets
    {
        unsigned int* segmentIndices;
        unsigned int* lowerModuleIndices; //4 of them now
        
        unsigned int *nTracklets; //number of tracklets per inner segment inner MD lower module
        float* zOut;
        float* rtOut;

        float* deltaPhiPos;
        float* deltaPhi;
        //delta beta = betaIn - betaOut
        float* betaIn;
        float* betaOut;

        tracklets();
        ~tracklets();
        void freeMemory();
        void freeMemoryCache();
    };

    void createTrackletsInUnifiedMemory(struct tracklets& trackletsInGPU, unsigned int maxTracklets, unsigned int maxPixelTracklets, unsigned int nLowerModules);
    void createTrackletsInExplicitMemory(struct tracklets& trackletsInGPU, unsigned int maxTracklets, unsigned int nLowerModules);

    CUDA_DEV void addTrackletToMemory(struct tracklets& trackletsInGPU, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, unsigned int innerInnerLowerModuleIndex, unsigned int innerOuterLowerModuleIndex, unsigned int outerInnerLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& deltaPhi, float& betaIn, float& betaOut, unsigned int trackletIndex);

    CUDA_DEV void runDeltaBetaIterations(float& betaIn, float& betaOut, float& betaAv, float & pt_beta, float sdIn_dr, float sdOut_dr, float dr, float lIn);

    CUDA_DEV bool runTrackletDefaultAlgoPPBB(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU ,struct segments& segmentsInGPU, unsigned int pixelModuleIndex, unsigned int outerInnerLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, float& zOut, float& rtOut, float& dPhiPos, float& dPhi, float& betaIn, float& betaOut, unsigned int
        N_MAX_SEGMENTS_PER_MODULE); // pixel to BB and BE segments

    CUDA_DEV bool runTrackletDefaultAlgoPPEE(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU ,struct segments& segmentsInGPU, unsigned int pixelModuleIndex, unsigned int outerInnerLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& dPhi, float& betaIn, float& betaOut, unsigned int
        N_MAX_SEGMENTS_PER_MODULE); // pixel to EE segments

    CUDA_DEV bool runTrackletDefaultAlgoBBBB(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, unsigned int innerInnerLowerModuleIndex, unsigned int innerOuterLowerModuleIndex, unsigned int outerInnerLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& deltaPhi, float& betaIn, float& betaOut);

    CUDA_DEV bool runTrackletDefaultAlgoBBEE(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, unsigned int innerInnerLowerModuleIndex, unsigned int innerOuterLowerModuleIndex, unsigned int outerInnerLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& deltaPhi, float& betaIn, float& betaOut);

    CUDA_DEV bool runTrackletDefaultAlgoEEEE(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, unsigned int innerInnerLowerModuleIndex, unsigned int innerOuterLowerModuleIndex, unsigned int outerInnerLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& deltaPhi, float& betaIn, float& betaOut);

    CUDA_DEV bool runTrackletDefaultAlgo(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, unsigned int innerInnerLowerModuleIndex, unsigned int innerOuterLowerModuleIndex, unsigned int outerInnerLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& deltaPhi, float& betaIn, float&
            betaOut, unsigned int N_MAX_SEGMENTS_PER_MODULE = 600);

    extern CUDA_CONST_VAR float pt_betaMax;

    void printTracklet(struct SDL::tracklets& trackletsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::hits& hitsInGPU, struct SDL::modules& modulesInGPU, unsigned int trackletIndex);


}
#endif
