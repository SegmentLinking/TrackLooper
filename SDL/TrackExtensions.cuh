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

        trackExtensions();
        ~trackExtensions();
        void freeMemory();
    };

    void createTrackExtensionsInUnifiedMemory(struct trackExtensions& trackExtensionsInGPU, unsigned int maxTrackExtensions);

    void createTrackExtensionsInExplicitMemory(struct trackExtensions& trackExtensionsInGPU, unsigned int maxTrackExtensions);

    CUDA_DEV void addTrackExtensionToMemory(struct trackExtensions& trackExtensionsInGPU, short* constituentTCType, unsigned int* constituentTCIndex, unsigned int* nLayerOverlaps, unsigned int* nHitOverlaps, unsigned int trackExtensionIndex);

    //FIXME:Need to extend this to > 2 objects

    CUDA_DEV bool runTrackExtensionDefaultAlgo(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct triplets& tripletsInGPU, struct trackCandidates& trackCandidatesInGPU, unsigned int anchorObjectIndex, unsigned int outerObjectIndex, short anchorObjectType, short outerObjectType, unsigned int outerT3Index, unsigned int layerOverlapTarget, short* constituentTCType, unsigned int* constituentTCIndex, unsigned int* nLayerOverlaps, unsigned int* nHitOverlaps);

    CUDA_DEV bool computeLayerAndHitOverlaps(SDL::modules& modulesInGPU, unsigned int* anchorLayerIndices, unsigned int* anchorHitIndices, unsigned int* anchorLowerModuleIndices, unsigned int* outerObjectLayerIndices, unsigned int* outerObjectHitIndices, unsigned int* outerObjectLowerModuleIndice, unsigned int nAnchorLayers, unsigned int nOuterLayers, unsigned int& nLayerOverlap, unsigned int& nHitOverlap, unsigned int& layerOverlapTarget);

}
#endif