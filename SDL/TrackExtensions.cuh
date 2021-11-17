#ifndef TrackExtensions_cuh
#define TrackExtensions_cuh

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

#include "TrackCandidate.cuh"
#include "PixelQuintuplet.cuh"
#include "PixelTriplet.cuh"
#include "Triplet.cuh"
#include "Quintuplet.cuh"
#include "Module.cuh"
#include "Hit.cuh"
#include "PrintUtil.h"

namespace SDL
{
    struct trackExtensions
    {
        short* constituentTCTypes;
        unsigned int* constituentTCIndices;
        unsigned int* nLayerOverlaps;
        unsigned int* nHitOverlaps;
        unsigned int* nTrackExtensions; //overall counter!
        float* rPhiChiSquared;
        float* rzChiSquared;
        float* radius;
        bool* isDup;

        trackExtensions();
        ~trackExtensions();
        void freeMemory();
    };

    void createTrackExtensionsInUnifiedMemory(struct trackExtensions& trackExtensionsInGPU, unsigned int maxTrackExtensions, unsigned int nTrackCandidates);

    void createTrackExtensionsInExplicitMemory(struct trackExtensions& trackExtensionsInGPU, unsigned int maxTrackExtensions, unsigned int nTrackCandidates);

    CUDA_DEV void addTrackExtensionToMemory(struct trackExtensions& trackExtensionsInGPU, short* constituentTCType, unsigned int* constituentTCIndex, unsigned int* nLayerOverlaps, unsigned int* nHitOverlaps, float rPhiChiSquared, float rzChiSquared, float radius, unsigned int trackExtensionIndex);

    //FIXME:Need to extend this to > 2 objects
    CUDA_DEV bool runTrackExtensionDefaultAlgo(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct triplets& tripletsInGPU, struct quintuplets& quintupletsInGPU, struct pixelTriplets& pixelTripletsInGPU, struct pixelQuintuplets& pixelQuintupletsInGPU, struct trackCandidates& trackCandidatesInGPU, unsigned int anchorObjectIndex, unsigned int outerObjectIndex, short anchorObjectType, short outerObjectType, unsigned int anchorObjectOuterT3Index, unsigned int layerOverlapTarget, short* constituentTCType, unsigned int* constituentTCIndex, unsigned
        int* nLayerOverlaps, unsigned int* nHitOverlaps, float& rPhiChiSquared, float& rzChiSquared, float& radius);


    CUDA_DEV bool computeLayerAndHitOverlaps(SDL::modules& modulesInGPU, unsigned int* anchorLayerIndices, unsigned int* anchorHitIndices, unsigned int* anchorLowerModuleIndices, unsigned int* outerObjectLayerIndices, unsigned int* outerObjectHitIndices, unsigned int* outerObjectLowerModuleIndice, unsigned int nAnchorLayers, unsigned int nOuterLayers, unsigned int& nLayerOverlap, unsigned int& nHitOverlap, unsigned int& layerOverlapTarget);

    CUDA_DEV float computeTERPhiChiSquared(struct modules& modulesInGPU, struct hits& hitsInGPU, float& g, float& f, float& radius, unsigned int* outerObjectAnchorHits, unsigned int* outerObjectLowerModules);

    CUDA_DEV float computeT3T3RPhiChiSquared(struct modules& modulesInGPU, struct hits& hitsInGPU, int nPoints, unsigned int* anchorHitIndices, unsigned int* lowerModuleIndices, float& regressionRadius);

    CUDA_DEV bool passTERPhiChiSquaredCuts(int nLayerOverlaps, int nHitOverlaps, unsigned int layer_binary, float rPhiChiSquared);

    CUDA_DEV float computeTERZChiSquared(struct modules& modulesInGPU, struct hits& hitsInGPU, unsigned int* anchorObjectAnchorHitIndices, unsigned int* anchorLowerModuleIndices, unsigned int* outerObjectAnchorHitIndices, unsigned int* outerLowerModuleIndices, short anchorObjectType);

    CUDA_DEV float computeT3T3RZChiSquared(struct modules& modulesInGPU, struct hits& hitsInGPU, int nPoints, unsigned int* anchorHitIndices, unsigned int* lowerModuleIndices);

    CUDA_DEV void fitStraightLine(int nPoints, float* xs, float* ys, float& slope, float& intercept);
    CUDA_DEV bool passTERZChiSquaredCuts(int nLayerOverlaps, int nHitOverlaps, unsigned int layer_binary, float rzChiSquared);

}
#endif

