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

#include "Constants.h"
#include "EndcapGeometry.h"
#include "TiltedGeometry.h"
#include "Triplet.cuh"
#include "Tracklet.cuh"
#include "PixelTracklet.cuh"
#include "Segment.cuh"
#include "MiniDoublet.cuh"
#include "Module.cuh"
#include "Hit.cuh"
#include "PrintUtil.h"

namespace SDL
{
    struct trackCandidates
    {
        short* trackCandidateType; //4-T5 5-pT3 7-pT5 8-pLS
        unsigned int* objectIndices; //will hold tracklet and  triplet indices  - check the type!!
        unsigned int* nTrackCandidates;
        unsigned int* nTrackCandidatespT3;
        unsigned int* nTrackCandidatespT5;
        unsigned int* nTrackCandidatespLS;
        unsigned int* nTrackCandidatesT5;

        uint8_t* logicalLayers;
        unsigned int* hitIndices;
        uint16_t* lowerModuleIndices;
        bool* partOfExtension;

        FPX* centerX;
        FPX* centerY;
        FPX* radius;

        trackCandidates();
        ~trackCandidates();
        void freeMemory(cudaStream_t stream);
        void freeMemoryCache();
        void resetMemory(unsigned int maxTrackCandidates,cudaStream_t stream);
    };

    void createTrackCandidatesInUnifiedMemory(struct trackCandidates& trackCandidatesInGPU, unsigned int maxTrackCandidates,cudaStream_t stream);

    void createTrackCandidatesInExplicitMemory(struct trackCandidates& trackCandidatesInGPU, unsigned int maxTrackCandidates,cudaStream_t stream);
    
    CUDA_DEV void addTrackCandidateToMemory(struct trackCandidates& trackCandidatesInGPU, short trackCandidateType, unsigned int innerTrackletIndex, unsigned int outerTrackletIndex, unsigned int trackCandidateIndex);

    CUDA_DEV void addTrackCandidateToMemory(struct trackCandidates& trackCandidatesInGPU, short trackCandidateType, unsigned int innerTrackletIndex, unsigned int outerTrackletIndex, uint8_t* logicalLayerIndices, uint16_t* lowerModuleIndices, unsigned int* hitIndices, float centerX, float centerY, float radius, unsigned int trackCandidateIndex);
}

#endif
