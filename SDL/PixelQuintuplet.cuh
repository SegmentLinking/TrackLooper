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

#include "Constants.cuh"
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
        unsigned int* totOccupancyPixelQuintuplets;
        bool* isDup;
        FPX* score;
        FPX* eta;
        FPX* phi;
        //for track extensions
        uint8_t* logicalLayers;
        unsigned int* hitIndices;
        uint16_t* lowerModuleIndices;
        FPX* pixelRadius;
        FPX* quintupletRadius;
        FPX* centerX;
        FPX* centerY;
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
    CUDA_DEV void addPixelQuintupletToMemory(struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct quintuplets& quintupletsInGPU, struct pixelQuintuplets& pixelQuintupletsInGPU, unsigned int pixelIndex, unsigned int T5Index, unsigned int pixelQuintupletIndex, float& rzChiSquared, float& rPhiChiSquared, float& rPhiChiSquaredInwards, float score, float eta, float phi, float& pixelRadius, float& quintupletRadius,
        float& centerX, float& centerY);
#else
    CUDA_DEV void addPixelQuintupletToMemory(struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct quintuplets& quintupletsInGPU, struct pixelQuintuplets& pixelQuintupletsInGPU, unsigned int pixelIndex, unsigned int T5Index, unsigned int pixelQuintupletIndex, float score, float eta, float phi, float& pixelRadius, float& quintupletRadius,
        float& centerX, float& centerY);
#endif

    CUDA_DEV bool runPixelQuintupletDefaultAlgo(struct modules& modulesInGPU, struct objectRanges& rangesInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct triplets& tripletsInGPU, struct quintuplets& quintupletsInGPU, unsigned int& pixelSegmentIndex, unsigned int& quintupletIndex, float& rzChiSquared, float& rPhiChiSquared, float& rPhiChiSquaredInwards, float& pixelRadius, float& quintupletRadius, float& centerX, float& centerY);

    CUDA_DEV float computePT5RZChiSquared(struct modules& modulesInGPU, uint16_t* lowerModuleIndices, float* rtPix, float* zPix, float* rts, float* zs);

    CUDA_DEV bool passPT5RZChiSquaredCuts(struct modules& modulesInGPU, uint16_t& lowerModuleIndex1, uint16_t& lowerModuleIndex2, uint16_t& lowerModuleIndex3, uint16_t& lowerModuleIndex4, uint16_t& lowerModuleIndex5, float& rzChiSquared);

    CUDA_DEV float computePT5RPhiChiSquared(struct modules& modulesInGPU, uint16_t* lowerModuleIndices, float& g, float& f, float& radius, float* xs, float* ys);

    CUDA_DEV float computePT5RPhiChiSquaredInwards(struct modules& modulesInGPU, float& g, float& f, float& r, float* xPix, float* yPix);

    CUDA_DEV bool passPT5RPhiChiSquaredInwardsCuts(struct modules& modulesInGPU, uint16_t& lowerModuleIndex1, uint16_t& lowerModuleIndex2, uint16_t& lowerModuleIndex3, uint16_t& lowerModuleIndex4, uint16_t& lowerModuleIndex5, float rPhiChiSquared);

    CUDA_DEV bool passPT5RPhiChiSquaredCuts(struct modules& modulesInGPU, uint16_t& lowerModuleIndex1, uint16_t& lowerModuleIndex2, uint16_t& lowerModuleIndex3, uint16_t& lowerModuleIndex4, uint16_t& lowerModuleIndex5, float rPhiChiSquared);
    CUDA_DEV void computeSigmasForRegression_pT5(SDL::modules& modulesInGPU, const uint16_t* lowerModuleIndices, float* delta1, float* delta2, float* slopes, bool* isFlat, int nPoints = 5, bool anchorHits = true);


}

#endif
