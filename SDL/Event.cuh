#ifndef Event_h
#define Event_h

#include <vector>
#include <list>
#include <map>
#include <cassert>
#include <stdlib.h>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <omp.h>
#include <chrono>
#include "Module.cuh"
#include "Hit.cuh"
#include "MiniDoublet.cuh"
#include "Segment.cuh"
#include "Tracklet.cuh"
#include "Triplet.cuh"
#include "TrackCandidate.cuh"

#include "cuda_profiler_api.h"

namespace SDL
{
    class Event
    {
    private:
        std::array<unsigned int, 6> n_hits_by_layer_barrel_;
        std::array<unsigned int, 5> n_hits_by_layer_endcap_;
        std::array<unsigned int, 6> n_minidoublets_by_layer_barrel_;
        std::array<unsigned int, 5> n_minidoublets_by_layer_endcap_;
        std::array<unsigned int, 6> n_segments_by_layer_barrel_;
        std::array<unsigned int, 5> n_segments_by_layer_endcap_;
        std::array<unsigned int, 6> n_tracklets_by_layer_barrel_;
        std::array<unsigned int, 5> n_tracklets_by_layer_endcap_;
        std::array<unsigned int, 6> n_triplets_by_layer_barrel_;
        std::array<unsigned int, 5> n_triplets_by_layer_endcap_;
        std::array<unsigned int, 6> n_trackCandidates_by_layer_barrel_;
        std::array<unsigned int, 5> n_trackCandidates_by_layer_endcap_;


        //CUDA stuff
        struct hits* hitsInGPU;
        struct miniDoublets* mdsInGPU;
        struct segments* segmentsInGPU;
        struct tracklets* trackletsInGPU;
        struct triplets* tripletsInGPU;
        struct trackCandidates* trackCandidatesInGPU;

    public:
        Event();
        ~Event();

        void addHitToEvent(float x, float y, float z, unsigned int detId); //call the appropriate hit function, then increment the counter here
        void addPixelSegmentToEvent(std::vector<unsigned int> hitIndices, float dPhiChange, float ptIn, float ptErr, float px, float py, float pz, float etaErr);

        /*functions that map the objects to the appropriate modules*/
        void addMiniDoubletsToEvent();
        void transfertest(struct SDL::miniDoublets& mdsInGPU, struct SDL::miniDoublets& mdsInHost, unsigned int maxds, unsigned int nModules);
        void addSegmentsToEvent();
        void addTrackletsToEvent();
        void addTrackletsWithAGapToEvent();
        void addTripletsToEvent();
        void addTrackCandidatesToEvent();

        void resetObjectsInModule();

        void createMiniDoublets();
        void createSegmentsWithModuleMap();
        void createTriplets();
        void createTrackletsWithModuleMap();
        void createPixelTracklets();
        void createTrackletsWithAGapWithModuleMap();
        void createTrackCandidates();

        unsigned int getNumberOfHits();
        unsigned int getNumberOfHitsByLayer(unsigned int layer);
        unsigned int getNumberOfHitsByLayerBarrel(unsigned int layer);
        unsigned int getNumberOfHitsByLayerEndcap(unsigned int layer);

        unsigned int getNumberOfMiniDoublets();
        unsigned int getNumberOfMiniDoubletsByLayer(unsigned int layer);
        unsigned int getNumberOfMiniDoubletsByLayerBarrel(unsigned int layer);
        unsigned int getNumberOfMiniDoubletsByLayerEndcap(unsigned int layer);

        unsigned int getNumberOfSegments();
        unsigned int getNumberOfSegmentsByLayer(unsigned int layer);
        unsigned int getNumberOfSegmentsByLayerBarrel(unsigned int layer);
        unsigned int getNumberOfSegmentsByLayerEndcap(unsigned int layer);

        unsigned int getNumberOfTracklets();
        unsigned int getNumberOfTrackletsByLayer(unsigned int layer);
        unsigned int getNumberOfTrackletsByLayerBarrel(unsigned int layer);
        unsigned int getNumberOfTrackletsByLayerEndcap(unsigned int layer);

        unsigned int getNumberOfTriplets();
        unsigned int getNumberOfTripletsByLayer(unsigned int layer);
        unsigned int getNumberOfTripletsByLayerBarrel(unsigned int layer);
        unsigned int getNumberOfTripletsByLayerEndcap(unsigned int layer);

        unsigned int getNumberOfTrackCandidates();
        unsigned int getNumberOfTrackCandidatesByLayer(unsigned int layer);
        unsigned int getNumberOfTrackCandidatesByLayerBarrel(unsigned int layer);
        unsigned int getNumberOfTrackCandidatesByLayerEndcap(unsigned int layer);

        struct hits* getHits();
        struct miniDoublets* getMiniDoublets();
        struct segments* getSegments() ;
        struct tracklets* getTracklets();
        struct triplets* getTriplets();

    };

    //global stuff

    extern struct modules* modulesInGPU;
    extern struct modules* modulesInHost;
    extern unsigned int nModules;
    void initModules(); //read from file and init
    void cleanModules();
    void initModulesHost(); //read from file and init

}

__global__ void testMiniDoublets(struct SDL::miniDoublets& mdsInGPU);
__global__ void createMiniDoubletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU);

__global__ void createSegmentsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU);

 __global__ void createSegmentsFromInnerLowerModule(struct SDL::modules&modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, unsigned int innerLowerModuleIndex, unsigned int nInnerMDs);

__global__ void createTrackletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU);

__global__ void createTrackletsFromInnerInnerLowerModule(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU, unsigned int innerInnerLowerModuleIndex, unsigned int nInnerSegments, unsigned int innerInnerLowerModuleArrayIndex);

__global__ void createPixelTrackletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU);

__global__ void createPixelTrackletsFromOuterInnerLowerModule(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU, unsigned int outerInnerLowerModuleIndex, unsigned int nInnerSegments, unsigned int nOuterSegments, unsigned int pixelModuleIndex, unsigned int pixelLowerModuleArrayIndex);

__global__ void createTrackletsWithAGapInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU);


__global__ void createTrackletsWithAGapFromInnerInnerLowerModule(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU, unsigned int innerInnerLowerModuleIndex, unsigned int nInnerSegments, unsigned int innerInnerLowerModuleArrayIndex);

__global__ void createTripletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU);

__global__ void createTripletsFromInnerInnerLowerModule(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, unsigned int innerInnerLowerModuleIndex, unsigned int nInnerSegments, unsigned int innerInnerLowerModuleArrayIndex);

__global__ void createTrackCandidatesInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU);

__global__ void createTrackCandidatesFromInnerInnerInnerLowerModule(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU, unsigned int innerInnerInnerLowerModuleArrayIndex, unsigned int nInnerTracklets, unsigned int
        nInnerTriplets);

#endif
