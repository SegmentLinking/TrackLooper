#ifndef Event_cuh
#define Event_cuh

#include <vector>
#include <list>
#include <map>
#include <cassert>
#include <stdlib.h>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <memory>
#include <algorithm>
#include <cuda_runtime.h>
#include <omp.h>
#include <chrono>
#include "Module.cuh"
#include "Hit.cuh"
#include "MiniDoublet.cuh"
#include "Segment.cuh"
#include "Tracklet.cuh"
#include "PixelTracklet.cuh"
#include "Triplet.cuh"
#include "TrackCandidate.cuh"
#include "Quintuplet.cuh"
#include "PixelTriplet.cuh"
#include "PixelQuintuplet.cuh"
#include "Kernels.cuh"

#include "cuda_profiler_api.h"
#ifdef __CUDACC__
#define CUDA_G __global__
#else
#define CUDA_G
#endif
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
        std::array<unsigned int, 6> n_quintuplets_by_layer_barrel_;
        std::array<unsigned int, 5> n_quintuplets_by_layer_endcap_;


        //CUDA stuff
        struct hits* hitsInGPU;
        struct miniDoublets* mdsInGPU;
        struct segments* segmentsInGPU;
        struct tracklets* trackletsInGPU;
        struct pixelTracklets* pixelTrackletsInGPU;
        struct triplets* tripletsInGPU;
        struct quintuplets* quintupletsInGPU;
        struct trackCandidates* trackCandidatesInGPU;
        struct pixelTriplets* pixelTripletsInGPU;
        struct pixelQuintuplets* pixelQuintupletsInGPU;

        //CPU interface stuff
        hits* hitsInCPU;
        miniDoublets* mdsInCPU;
        segments* segmentsInCPU;
        tracklets* trackletsInCPU;
        pixelTracklets* pixelTrackletsInCPU;
        triplets* tripletsInCPU;
        trackCandidates* trackCandidatesInCPU;
        modules* modulesInCPU;
        modules* modulesInCPUFull;
        quintuplets* quintupletsInCPU;
        pixelTriplets* pixelTripletsInCPU;
        pixelQuintuplets* pixelQuintupletsInCPU;

        int* superbinCPU;
        int* pixelTypeCPU;
    public:
        Event();
        ~Event();

        void addHitToEvent(std::vector<float> x, std::vector<float> y, std::vector<float> z, std::vector<unsigned int> detId, std::vector<unsigned int> idxInNtuple); //call the appropriate hit function, then increment the counter here
        void addPixelSegmentToEvent(std::vector<unsigned int> hitIndices0,std::vector<unsigned int> hitIndices1,std::vector<unsigned int> hitIndices2,std::vector<unsigned int> hitIndices3, std::vector<float> dPhiChange, std::vector<float> ptIn, std::vector<float> ptErr, std::vector<float> px, std::vector<float> py, std::vector<float> pz, std::vector<float> eta, std::vector<float> etaErr, std::vector<float> phi, std::vector<int> superbin, std::vector<int> pixelType, std::vector<short> isQuad);

        /*functions that map the objects to the appropriate modules*/
        void addMiniDoubletsToEvent();
        void addSegmentsToEvent();
        void addTrackletsToEvent();
        void addTrackletsWithAGapToEvent();
        void addTripletsToEvent();
        void addTrackCandidatesToEvent();
        void addMiniDoubletsToEventExplicit();
        void addSegmentsToEventExplicit();
        void addTrackletsToEventExplicit();
        void addTrackletsWithAGapToEventExplicit();
        void addTripletsToEventExplicit();
        void addTrackCandidatesToEventExplicit();
        void addQuintupletsToEvent();
        void addQuintupletsToEventExplicit();

        void resetObjectsInModule();

        void createMiniDoublets();
        void createSegmentsWithModuleMap();
        void createTriplets();
        void createTrackletsWithModuleMap();
        void createPixelTracklets();
        void createPixelTrackletsWithMap();
        void createTrackletsWithAGapWithModuleMap();
        void createTrackCandidates();
        void createQuintuplets();
        void createPixelTriplets();
        void createPixelQuintuplets();
        void pixelLineSegmentCleaning();

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
        unsigned int getNumberOfPixelTracklets();
        unsigned int getNumberOfTrackletsByLayer(unsigned int layer);
        unsigned int getNumberOfTrackletsByLayerBarrel(unsigned int layer);
        unsigned int getNumberOfTrackletsByLayerEndcap(unsigned int layer);

        unsigned int getNumberOfTriplets();
        unsigned int getNumberOfTripletsByLayer(unsigned int layer);
        unsigned int getNumberOfTripletsByLayerBarrel(unsigned int layer);
        unsigned int getNumberOfTripletsByLayerEndcap(unsigned int layer);

        unsigned int getNumberOfTrackCandidates();
        unsigned int getNumberOfPixelTrackCandidates();
        unsigned int getNumberOfTrackCandidatesByLayer(unsigned int layer);
        unsigned int getNumberOfTrackCandidatesByLayerBarrel(unsigned int layer);
        unsigned int getNumberOfTrackCandidatesByLayerEndcap(unsigned int layer);

        unsigned int getNumberOfQuintuplets();
        unsigned int getNumberOfQuintupletsByLayer(unsigned int layer);
        unsigned int getNumberOfQuintupletsByLayerBarrel(unsigned int layer);
        unsigned int getNumberOfQuintupletsByLayerEndcap(unsigned int layer);

        unsigned int getNumberOfPixelTriplets();
        unsigned int getNumberOfPixelQuintuplets();

        hits* getHits();
        miniDoublets* getMiniDoublets();
        segments* getSegments() ;
        tracklets* getTracklets();
        pixelTracklets* getPixelTracklets();
        triplets* getTriplets();
        quintuplets* getQuintuplets();
        trackCandidates* getTrackCandidates();
        pixelTriplets* getPixelTriplets();
        modules* getModules();
        modules* getFullModules();
        pixelQuintuplets* getPixelQuintuplets();

    };

    //global stuff

    extern struct modules* modulesInGPU;
    extern struct modules* modulesInHost;
    extern unsigned int nModules;
    void initModules(const char* moduleMetaDataFilePath="data/centroid.txt"); //read from file and init
    void cleanModules();
    void initModulesHost(); //read from file and init
    extern struct pixelMap* pixelMapping;

}

#endif
