#ifndef TrackCandidate_cuh
#define TrackCandidate_cuh

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
#include "EndcapGeometry.cuh"
#include "TiltedGeometry.h"
#include "Triplet.cuh"
#include "Segment.cuh"
#include "MiniDoublet.cuh"
#include "PixelTriplet.cuh"
#include "Quintuplet.cuh"
#include "Module.cuh"
#include "Hit.cuh"
#include "PrintUtil.h"

namespace SDL
{
    struct trackCandidates
    {
        short* trackCandidateType; //4-T5 5-pT3 7-pT5 8-pLS
        unsigned int* directObjectIndices; // will hold direct indices to each type containers
        unsigned int* objectIndices; //will hold tracklet and  triplet indices  - check the type!!
        unsigned int* nTrackCandidates;
        unsigned int* nTrackCandidatespT3;
        unsigned int* nTrackCandidatespT5;
        unsigned int* nTrackCandidatespLS;
        unsigned int* nTrackCandidatesT5;

        uint8_t* logicalLayers;
        unsigned int* hitIndices;
        int* pixelSeedIndex;
        uint16_t* lowerModuleIndices;

        FPX* centerX;
        FPX* centerY;
        FPX* radius;

        trackCandidates();
        ~trackCandidates();
        void freeMemory(cudaStream_t stream);
        void freeMemoryCache();
        void resetMemory(unsigned int maxTrackCandidates,cudaStream_t stream);
    };

    void createTrackCandidatesInExplicitMemory(struct trackCandidates& trackCandidatesInGPU, unsigned int maxTrackCandidates,cudaStream_t stream);
    
    CUDA_DEV void addpLSTrackCandidateToMemory(struct trackCandidates& trackCandidatesInGPU, unsigned int trackletIndex, unsigned int trackCandidateIndex, uint4 hitIndices, int pixelSeedIndex);

    CUDA_DEV void addTrackCandidateToMemory(struct trackCandidates& trackCandidatesInGPU, short trackCandidateType, unsigned int innerTrackletIndex, unsigned int outerTrackletIndex, uint8_t* logicalLayerIndices, uint16_t* lowerModuleIndices, unsigned int* hitIndices, int pixelSeedIndex, float centerX, float centerY, float radius, unsigned int trackCandidateIndex, unsigned int directObjectIndex);

__global__ void crossCleanpT3(struct SDL::modules& modulesInGPU, struct SDL::objectRanges& rangesInGPU, struct SDL::pixelTriplets& pixelTripletsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU);

__global__ void crossCleanT5(struct SDL::modules& modulesInGPU, struct SDL::quintuplets& quintupletsInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU,struct SDL::pixelTriplets& pixelTripletsInGPU, struct SDL::objectRanges& rangesInGPU);

__global__ void crossCleanpLS(struct SDL::modules& modulesInGPU, struct SDL::objectRanges& rangesInGPU, struct SDL::pixelTriplets& pixelTripletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU,struct SDL::segments& segmentsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::hits& hitsInGPU, struct SDL::quintuplets& quintupletsInGPU);

__global__ void addpT3asTrackCandidatesInGPU(uint16_t nLowerModules, struct SDL::pixelTriplets& pixelTripletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU, struct SDL::segments& segmentsInGPU, struct SDL::objectRanges& rangesInGPU);

__global__ void addT5asTrackCandidateInGPU(uint16_t nLowerModules, struct SDL::quintuplets& quintupletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU, struct SDL::objectRanges& rangesInGPU);

__global__ void addpLSasTrackCandidateInGPU(uint16_t nLowerModules, struct SDL::trackCandidates& trackCandidatesInGPU, struct SDL::segments& segmentsInGPU);

__global__ void addpT5asTrackCandidateInGPU(uint16_t nLowerModules, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU, struct SDL::segments& segmentsInGPU, struct SDL::objectRanges& rangesInGPU);

  CUDA_DEV int checkPixelHits(unsigned int ix, unsigned int jx,struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::hits& hitsInGPU);
}

#endif
