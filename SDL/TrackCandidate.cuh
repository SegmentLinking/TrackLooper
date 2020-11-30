#ifndef TrackCandidate_h
#define TrackCandidate_h

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
#include "Segment.cuh"
#include "MiniDoublet.cuh"
#include "Module.cuh"
#include "Hit.cuh"
#include "PrintUtil.h"

namespace SDL
{
    struct trackCandidates
    {
        short* trackCandidateType; //3 types : 0-T4T4, 1-T4T3, 2-T3T4
        unsigned int* objectIndices; //will hold tracklet and  triplet indices  - check the type!!
        unsigned int* nTrackCandidates;
        unsigned int* nTrackCandidatesT4T4;
        unsigned int* nTrackCandidatesT4T3;
        unsigned int* nTrackCandidatesT3T4;

        trackCandidates();
        ~trackCandidates();
        void freeMemory();
    };

    void createTrackCandidatesInUnifiedMemory(struct trackCandidates& trackCandidatesInGPU, unsigned int maxTrackCandidates, unsigned int maxPixelTrackCandidates, unsigned int nLowerModules);
    void createTrackCandidatesInExplicitMemory(struct trackCandidates& trackCandidatesInGPU, unsigned int maxTrackCandidates, unsigned int maxPixelTrackCandidates, unsigned int nLowerModules);
    
    CUDA_DEV void addTrackCandidateToMemory(struct trackCandidates& trackCandidatesInGPU, short trackCandidateType, unsigned int innerTrackletIndex, unsigned int outerTrackletIndex, unsigned int trackCandidateIndex);
   
    CUDA_DEV bool runTrackCandidateDefaultAlgoTwoTracklets(struct tracklets& trackletsInGPU, struct triplets& tripletsInGPU, unsigned int innerTrackletIndex, unsigned int outerTrackletIndex, short& trackCandidateType);

    CUDA_DEV bool runTrackCandidateDefaultAlgoTrackletToTriplet(struct tracklets& trackletsInGPU, struct triplets& tripletsInGPU, unsigned int innerTrackletIndex, unsigned int outerTripletIndex, short& trackCandidateType);

    CUDA_DEV bool runTrackCandidateDefaultAlgoTripletToTracklet(struct tracklets& trackletsInGPU, struct triplets& tripletsInGPU, unsigned int innerTripletIndex, unsigned int outerTrackletIndex, short& trackCandidateType);

    CUDA_DEV bool hasCommonSegment(struct tracklets& trackletsInGPU, struct triplets& tripletsInGPU, unsigned int innerObjectIndex, unsigned int outerObjectIndex, short trackCandidateType);

}

#endif
