#ifndef PixelQuintuplet_cuh
#define PixelQuintuplet_cuh

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
#include "Triplet.cuh"
#include "Tracklet.cuh"
#include "Quintuplet.cuh"
#include "PixelTriplet.cuh"

namespace SDL
{
    struct pixelQuintuplets
    {
        unsigned int* pixelIndices;
        unsigned int* T5Indices;
        unsigned int* nPixelQuintuplets;
        bool* isDup;
        FPX* score;
        FPX* eta;
        FPX* phi;
#ifdef CUT_VALUE_DEBUG
        float* rzChiSquared;
        float* rPhiChiSquared;
        float* rPhiChiSquaredInwards;
#endif

        pixelQuintuplets();
        ~pixelQuintuplets();
        void freeMemory(cudaStream_t stream);
        void freeMemoryCache();
        void resetMemory(unsigned int maxPixelQuintuplets,cudaStream_t stream);

    };

    void createPixelQuintupletsInUnifiedMemory(struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, unsigned int maxPixelQuintuplets,cudaStream_t stream);
    void createPixelQuintupletsInExplicitMemory(struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, unsigned int maxPixelQuintuplets, cudaStream_t stream);

    CUDA_DEV void rmPixelQuintupletToMemory(struct pixelQuintuplets& pixelQuintupletsInGPU, unsigned int pixelQuintupletIndex);
#ifdef CUT_VALUE_DEBUG
    CUDA_DEV void addPixelQuintupletToMemory(struct pixelQuintuplets& pixelQuintupletsInGPU, unsigned int pixelIndex, unsigned int T5Index, unsigned int pixelQuintupletIndex, float& rzChiSquared, float& rPhiChiSquared, float& rPhiChiSquaredInwards, float score);

#else
    CUDA_DEV void addPixelQuintupletToMemory(struct pixelQuintuplets& pixelQuintupletsInGPU, unsigned int pixelIndex, unsigned int T5Index, unsigned int pixelQuintupletIndex, float score, float eta, float phi);
#endif

    CUDA_DEV bool runPixelQuintupletDefaultAlgo(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct triplets& tripletsInGPU, struct quintuplets& quintupletsInGPU, unsigned int& pixelSegmentIndex, unsigned int& quintupletIndex, float& rzChiSquared, float& rPhiChiSquared, float& rPhiChiSquaredInwards);

    CUDA_DEV float computePT5RZChiSquared(struct modules& modulesInGPU, struct hits& hitsInGPU, unsigned int& pixelAnchorHitIndex1, unsigned int& pixelAnchorHitIndex2, unsigned int* anchorHits, unsigned int* lowerModuleIndices);

    CUDA_DEV bool passPT5RZChiSquaredCuts(struct modules& modulesInGPU, unsigned int lowerModuleIndex1, unsigned int lowerModuleIndex2, unsigned int lowerModuleIndex3, unsigned int lowerModuleIndex4, unsigned int lowerModuleIndex5, float& rzChiSquared);


    CUDA_DEV float computePT5RPhiChiSquared(struct modules& modulesInGPU, struct hits& hitsInGPU, struct segments& segmentsInGPU, unsigned int pixelSegmentArrayIndex, unsigned int* anchorHits, unsigned int* lowerModuleIndices);

    CUDA_DEV float computePT5RPhiChiSquaredInwards(struct modules& modulesInGPU, struct hits& hitsInGPU, struct quintuplets& quintupletsInGPU, unsigned int quintupletIndex, unsigned int* pixelHits);

    CUDA_DEV bool passPT5RPhiChiSquaredInwardsCuts(struct modules& modulesInGPU, unsigned int lowerModuleIndex1, unsigned int lowerModuleIndex2, unsigned int lowerModuleIndex3, unsigned int lowerModuleIndex4, unsigned int lowerModuleIndex5, float rPhiChiSquared);

    CUDA_DEV bool passPT5RPhiChiSquaredCuts(struct modules& modulesInGPU, unsigned int lowerModuleIndex1, unsigned int lowerModuleIndex2, unsigned int lowerModuleIndex3, unsigned int lowerModuleIndex4, unsigned int lowerModuleIndex5, float rPhiChiSquared);
    CUDA_DEV void computeSigmasForRegression_pT5(SDL::modules& modulesInGPU, const unsigned int* lowerModuleIndices, float* delta1, float* delta2, float* slopes, bool* isFlat, int nPoints = 5, bool anchorHits = true);


}

#endif
