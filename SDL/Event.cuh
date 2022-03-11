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
#include "PixelTracklet.cuh"
#include "Triplet.cuh"
#include "TrackCandidate.cuh"
#include "Quintuplet.cuh"
#include "PixelTriplet.cuh"
#include "PixelQuintuplet.cuh"
#include "TrackExtensions.cuh"
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
        cudaStream_t stream;
//        unsigned int hitOffset;
        std::array<unsigned int, 6> n_hits_by_layer_barrel_;
        std::array<unsigned int, 5> n_hits_by_layer_endcap_;
        std::array<unsigned int, 6> n_minidoublets_by_layer_barrel_;
        std::array<unsigned int, 5> n_minidoublets_by_layer_endcap_;
        std::array<unsigned int, 6> n_segments_by_layer_barrel_;
        std::array<unsigned int, 5> n_segments_by_layer_endcap_;
        std::array<unsigned int, 6> n_triplets_by_layer_barrel_;
        std::array<unsigned int, 5> n_triplets_by_layer_endcap_;
        std::array<unsigned int, 6> n_trackCandidates_by_layer_barrel_;
        std::array<unsigned int, 5> n_trackCandidates_by_layer_endcap_;
        std::array<unsigned int, 6> n_quintuplets_by_layer_barrel_;
        std::array<unsigned int, 5> n_quintuplets_by_layer_endcap_;


        //CUDA stuff
        struct objectRanges* rangesInGPU;
        struct hits* hitsInGPU;
        struct miniDoublets* mdsInGPU;
        struct segments* segmentsInGPU;
        struct triplets* tripletsInGPU;
        struct quintuplets* quintupletsInGPU;
        struct trackCandidates* trackCandidatesInGPU;
        struct pixelTriplets* pixelTripletsInGPU;
        struct pixelQuintuplets* pixelQuintupletsInGPU;
        struct trackExtensions* trackExtensionsInGPU;

        //CPU interface stuff
        objectRanges* rangesInCPU;
        hits* hitsInCPU;
        miniDoublets* mdsInCPU;
        segments* segmentsInCPU;
        triplets* tripletsInCPU;
        trackCandidates* trackCandidatesInCPU;
        modules* modulesInCPU;
        modules* modulesInCPUFull;
        quintuplets* quintupletsInCPU;
        pixelTriplets* pixelTripletsInCPU;
        pixelQuintuplets* pixelQuintupletsInCPU;
        trackExtensions* trackExtensionsInCPU;

        int* superbinCPU;
        int8_t* pixelTypeCPU;
    public:
        Event(cudaStream_t estream);
        ~Event();
        void resetEvent();

        void setHits(unsigned int offset,unsigned int loopsize);
        void addHitToEvent(std::vector<float> x, std::vector<float> y, std::vector<float> z, std::vector<unsigned int> detId, std::vector<unsigned int> idxInNtuple); //call the appropriate hit function, then increment the counter here
        //void preloadHitToEvent(std::vector<float> x, std::vector<float> y, std::vector<float> z, std::vector<unsigned int> detId, std::vector<unsigned int> idxInNtuple,unsigned int offset); 
        void addPixelSegmentToEvent(std::vector<unsigned int> hitIndices0,std::vector<unsigned int> hitIndices1,std::vector<unsigned int> hitIndices2,std::vector<unsigned int> hitIndices3, std::vector<float> dPhiChange, std::vector<float> ptIn, std::vector<float> ptErr, std::vector<float> px, std::vector<float> py, std::vector<float> pz, std::vector<float> eta, std::vector<float> etaErr, std::vector<float> phi, std::vector<int> superbin, std::vector<int8_t> pixelType, std::vector<short> isQuad);

        /*functions that map the objects to the appropriate modules*/
        void addMiniDoubletsToEvent();
        void addSegmentsToEvent();
        void addTripletsToEvent();
        void addMiniDoubletsToEventExplicit();
        void addSegmentsToEventExplicit();
        void addTripletsToEventExplicit();
        void addQuintupletsToEvent();
        void addQuintupletsToEventExplicit();
        void resetObjectsInModule();

        void createMiniDoublets();
        void createSegmentsWithModuleMap();
        void createTriplets();
        void createPixelTracklets();
        void createPixelTrackletsWithMap();
        void createTrackCandidates();
        void createExtendedTracks();
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

        unsigned int getNumberOfTriplets();
        unsigned int getNumberOfTripletsByLayer(unsigned int layer);
        unsigned int getNumberOfTripletsByLayerBarrel(unsigned int layer);
        unsigned int getNumberOfTripletsByLayerEndcap(unsigned int layer);

        unsigned int getNumberOfTrackCandidates();
        unsigned int getNumberOfPixelTrackCandidates();
        unsigned int getNumberOfPT5TrackCandidates();
        unsigned int getNumberOfPT3TrackCandidates();
        unsigned int getNumberOfT5TrackCandidates();
        unsigned int getNumberOfPLSTrackCandidates();

        unsigned int getNumberOfQuintuplets();
        unsigned int getNumberOfQuintupletsByLayer(unsigned int layer);
        unsigned int getNumberOfQuintupletsByLayerBarrel(unsigned int layer);
        unsigned int getNumberOfQuintupletsByLayerEndcap(unsigned int layer);

        unsigned int getNumberOfPixelTriplets();
        unsigned int getNumberOfPixelQuintuplets();

        unsigned int getNumberOfExtendedTracks();
        unsigned int getNumberOfT3T3ExtendedTracks();

        objectRanges* getRanges();
        hits* getHits();
        miniDoublets* getMiniDoublets();
        segments* getSegments() ;
        triplets* getTriplets();
        quintuplets* getQuintuplets();
        trackCandidates* getTrackCandidates();
        trackExtensions* getTrackExtensions();
        pixelTriplets* getPixelTriplets();
        modules* getModules();
        modules* getFullModules();
        pixelQuintuplets* getPixelQuintuplets();

    };

    //global stuff

    extern struct hits* hitsInGPUAll;
    extern struct modules* modulesInGPU;
    extern struct modules* modulesInHost;
    extern uint16_t nModules;
    extern uint16_t nLowerModules;
    void initModules(const char* moduleMetaDataFilePath="data/centroid.txt"); //read from file and init
    void initHits(std::vector<unsigned int> hitOffset,
std::vector<std::vector<float>>& out_trkX,std::vector<std::vector<float>>& out_trkY,std::vector<std::vector<float>>& out_trkZ,

std::vector<std::vector<unsigned int>>&    out_hitId,
std::vector<std::vector<unsigned int>>&    out_hitIdxs,
std::vector<std::vector<unsigned int>>&    out_hitIndices_vec0,
std::vector<std::vector<unsigned int>>&    out_hitIndices_vec1,
std::vector<std::vector<unsigned int>>&    out_hitIndices_vec2,
std::vector<std::vector<unsigned int>>&    out_hitIndices_vec3,
std::vector<std::vector<float>>&    out_deltaPhi_vec,
std::vector<std::vector<float>>&    out_ptIn_vec,
std::vector<std::vector<float>>&    out_ptErr_vec,
std::vector<std::vector<float>>&    out_px_vec,
std::vector<std::vector<float>>&    out_py_vec,
std::vector<std::vector<float>>&    out_pz_vec,
std::vector<std::vector<float>>&    out_eta_vec,
std::vector<std::vector<float>>&    out_etaErr_vec,
std::vector<std::vector<float>>&    out_phi_vec,
std::vector<std::vector<int>>&    out_superbin_vec,
std::vector<std::vector<int8_t>>&    out_pixelType_vec,
std::vector<std::vector<short>>&    out_isQuad_vec

);
    void cleanModules();
    void initModulesHost(); //read from file and init
    extern struct pixelMap* pixelMapping;
    void preloadHitToEvent(std::vector<float> x, std::vector<float> y, std::vector<float> z, std::vector<unsigned int> detId, std::vector<unsigned int> idxInNtuple,unsigned int offset); 

}
#endif
