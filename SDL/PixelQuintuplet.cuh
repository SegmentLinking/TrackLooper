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
        unsigned int* pT3Indices;
        unsigned int* T5Indices;
        unsigned int* nPixelQuintuplets;
#ifdef CUT_VALUE_DEBUG
        float* rzChiSquared;
        float* rPhiChiSquared;
        float* rPhiChiSquaredInwards;
#endif

        pixelQuintuplets();
        ~pixelQuintuplets();
        void freeMemory();
        void freeMemoryCache();

    };

    void createPixelQuintupletsInUnifiedMemory(struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, unsigned int maxPixelQuintuplets);
    void createPixelQuintupletsInExplicitMemory(struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, unsigned int maxPixelQuintuplets);

#ifdef CUT_VALUE_DEBUG
    CUDA_DEV void addPixelQuintupletToMemory(struct pixelQuintuplets& pixelQuintupletsInGPU, unsigned int pT3Index, unsigned int T5Index, unsigned int pixelQuintupletIndex, float& rzChiSquared, float& rPhiChiSquared, float& rPhiChiSquaredInwards);

#else
    CUDA_DEV void addPixelQuintupletToMemory(struct pixelQuintuplets& pixelQuintupletsInGPU, unsigned int pT3Index, unsigned int T5Index, unsigned int pixelQuintupletIndex);
#endif

    CUDA_DEV bool runPixelQuintupletDefaultAlgo(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct triplets& tripletsInGPU, struct quintuplets& quintupletsInGPU, struct pixelTriplets& pixelTripletsInGPU, unsigned int& pixelTripletIndex, unsigned int& quintupletIndex, float& rzChiSquared, float& rPhiChiSquared, float& rPhiChiSquaredInwards);

    CUDA_DEV float computePT5RZChiSquared(struct modules& modulesInGPU, struct hits& hitsInGPU, unsigned int& pixelAnchorHitIndex1, unsigned int& pixelAnchorHitIndex2, unsigned int* anchorHits, unsigned int* lowerModuleIndices);

    CUDA_DEV float computePT5RPhiChiSquared(struct modules& modulesInGPU, struct hits& hitsInGPU, struct segments& segmentsInGPU, unsigned int pixelSegmentArrayIndex, unsigned int* anchorHits, unsigned int* lowerModuleIndices);

    CUDA_DEV float computePT5RPhiChiSquaredInwards(struct modules& modulesInGPU, struct hits& hitsInGPU, struct quintuplets& quintupletsInGPU, unsigned int quintupletIndex, unsigned int* pixelHits);

}

#endif
