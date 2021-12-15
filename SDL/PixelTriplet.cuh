#ifndef PixelTriplet_cuh
#define PixelTriplet_cuh

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
#include "Tracklet.cuh"
#include "Triplet.cuh"
#include "Segment.cuh"
#include "MiniDoublet.cuh"
#include "Module.cuh"
#include "Hit.cuh"
#include "PrintUtil.h"
#include "Quintuplet.cuh"
#include "PixelTracklet.cuh"
#
namespace SDL
{
    struct pixelTriplets //one pixel segment, one outer tracker triplet!
    {
        unsigned int* pixelSegmentIndices;         
        unsigned int* tripletIndices;
        unsigned int* nPixelTriplets; //size 1

        FPX* pixelRadius;
        FPX* pixelRadiusError;
        FPX* rPhiChiSquared;
        FPX* rPhiChiSquaredInwards;
        FPX* rzChiSquared;
        FPX* tripletRadius;
        FPX* pt;
        FPX* eta;
        FPX* phi;
        FPX* eta_pix;
        FPX* phi_pix;
        FPX* score;
        bool* isDup;
        bool* partOfPT5;

        pixelTriplets();
        ~pixelTriplets();
        void freeMemory(cudaStream_t stream);
        void freeMemoryCache();
        void resetMemory(unsigned int maxPixelTriplets,cudaStream_t stream);
    };

    void createPixelTripletsInUnifiedMemory(struct pixelTriplets& pixelTripletsInGPU, unsigned int maxPixelTriplets, cudaStream_t stream);

    void createPixelTripletsInExplicitMemory(struct pixelTriplets& pixelTripletsinGPU, unsigned int maxPixelTriplets, cudaStream_t stream);

#ifdef CUT_VALUE_DEBUG
    CUDA_DEV void addPixelTripletToMemory(struct pixelTriplets& pixelTripletsInGPU, unsigned int pixelSegmentIndex, unsigned int tripletIndex, float pixelRadius, float pixelRadiusError, float tripletRadius, float rPhiChiSquared, float rPhiChiSquaredInwards, float rzChiSquared, unsigned int pixelTripletIndex, float pt, float eta, float phi, float eta_pix, float phi_pix, float score);
#else
    CUDA_DEV void addPixelTripletToMemory(struct pixelTriplets& pixelTripletsInGPU, unsigned int pixelSegmentIndex, unsigned int tripletIndex, float pixelRadius, float tripletRadius, unsigned int pixelTripletIndex,float pt, float eta, float phi, float eta_pix, float phi_pix, float score);
#endif

    CUDA_DEV float computeRadiusFromThreeAnchorHitspT3(float x1, float y1, float x2, float y2, float x3, float y3, float& g, float& f);

    CUDA_DEV void rmPixelTripletToMemory(struct pixelTriplets& pixelTripletsInGPU, unsigned int pixelTripletIndex);

    CUDA_DEV bool runPixelTripletDefaultAlgo(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct triplets& tripletsInGPU, unsigned int& pixelSegmentIndex, unsigned int tripletIndex, float& pixelRadius, float& pixelRadiusError, float& tripletRadius, float& rzChiSquared, float& rPhiChiSquared, float& rPhiChiSquaredInwards, bool runChiSquaredCuts = true);

    CUDA_DEV bool passRadiusCriterion(struct modules& modulesInGPU, float& pixelRadius, float& pixelRadiusError, float& tripletRadius, unsigned int lowerModuleIndex, unsigned int middleModuleIndex, unsigned int upperModuleIndex);

    CUDA_DEV bool passRadiusCriterionBBB(float& pixelRadius, float& pixelRadiusError, float& tripletRadius);

    CUDA_DEV bool passRadiusCriterionBBE(float& pixelRadius, float& pixelRadiusError, float& tripletRadius);

    CUDA_DEV bool passRadiusCriterionBEE(float& pixelRadius, float& pixelRadiusError, float& tripletRadius);

    CUDA_DEV bool passRadiusCriterionEEE(float& pixelRadius, float& pixelRadiusError, float& tripletRadius);

    CUDA_DEV float computePT3RZChiSquared(struct modules& modulesInGPU, struct hits& hitsInGPU, struct segments& segmentsInGPU, unsigned int& pixelAnchorHitIndex1, unsigned int& pixelAnchorHitIndex2, unsigned int* anchorHits, unsigned int* lowerModuleIndices);

    CUDA_DEV float computePT3RPhiChiSquared(struct modules& modulesInGPU, struct hits& hitsInGPU, struct segments& segmentsInGPU, unsigned int pixelSegmentArrayIndex, unsigned int* anchorHits, unsigned int* lowerModuleIndices);

    CUDA_DEV float computePT3RPhiChiSquaredInwards(struct modules& modulesInGPU, struct hits& hitsInGPU, float& r, float& g, float& f, unsigned int* pixelAnchorHits);

    CUDA_DEV bool passPT3RZChiSquaredCuts(struct modules& modulesInGPU, unsigned int lowerModuleIndex1, unsigned int lowerModuleIndex2, unsigned int lowerModuleIndex3, float& rzChiSquared);

    CUDA_DEV bool passPT3RPhiChiSquaredCuts(struct modules& mdoulesInGPU, unsigned int lowerModuleIndex1, unsigned int lowerModuleIndex2, unsigned int lowerModuleIndex3, float& rPhiChiSquared);

    CUDA_DEV bool passPT3RPhiChiSquaredInwardsCuts(struct modules& modulesInGPU, unsigned int lowerModuleIndex1, unsigned int lowerModuleIndex2, unsigned int lowerModuleIndex3, float& rPhiChiSquared);
}
#endif
