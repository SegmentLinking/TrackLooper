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
        void addPixelSegmentToEvent(std::vector<unsigned int> hitIndices0,std::vector<unsigned int> hitIndices1,std::vector<unsigned int> hitIndices2,std::vector<unsigned int> hitIndices3, std::vector<float> dPhiChange, std::vector<float> ptIn, std::vector<float> ptErr, std::vector<float> px, std::vector<float> py, std::vector<float> pz, std::vector<float> eta, std::vector<float> etaErr, std::vector<float> phi, std::vector<int> superbin, std::vector<int> pixelType);

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
__device__ float scorepT3(struct SDL::modules& modulesInGPU,struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, unsigned int innerPix,unsigned int outerTrip,float pt, float pz);
__global__ void removeDupPixelTripletsInGPUFromMap(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::pixelTriplets& pixelTripletsInGPU, struct SDL::triplets& tripletsInGPU);
__device__ int* checkHitspT3(unsigned int ix, unsigned int jx,struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::pixelTriplets& pixelTripletsInGPU);
__global__ void removeDupPixelQuintupletsInGPUFromMap(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::pixelTriplets& pixelTripletsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, struct SDL::quintuplets& quintupletsInGPU);
__global__ void markUsedObjects(struct SDL::modules& modulesInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, struct SDL::quintuplets& quintupletsInGPU);

__global__ void checkHitspLS(struct SDL::modules& modulesInGPU,struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::hits& hitsInGPU);


__device__ void scoreT5(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU,unsigned int innerTrip,unsigned int outerTrip, int layer, float* scores);
__global__ void removeDupQuintupletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::quintuplets& quintupletsInGPU, bool secondPass);
__device__ int checkHitsT5(unsigned int hit1, unsigned int hit2,struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU,struct SDL::triplets& tripletsInGPU, struct SDL::quintuplets& quintupletsInGPU);

__global__ void testMiniDoublets(struct SDL::miniDoublets& mdsInGPU);
__global__ void createMiniDoubletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU);

__global__ void createSegmentsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU);

 __global__ void createSegmentsFromInnerLowerModule(struct SDL::modules&modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, unsigned int innerLowerModuleIndex, unsigned int nInnerMDs);

__global__ void addpT2asTrackCandidateInGPU(struct SDL::modules& modulesInGPU,struct SDL::pixelTracklets& pixelTrackletsInGPU,struct SDL::trackCandidates& trackCandidatesInGPU);

__global__ void addpT3asTrackCandidateInGPU(struct SDL::modules& modulesInGPU,struct SDL::pixelTriplets& pixelTripletsInGPU,struct SDL::trackCandidates& trackCandidatesInGPU, struct SDL::segments& segmentsInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU);

__global__ void addT5asTrackCandidateInGPU(struct SDL::modules& modulesInGPU,struct SDL::quintuplets& quintupletsInGPU,struct SDL::trackCandidates& trackCandidatesInGPU);

__global__ void addpT5asTrackCandidateInGPU(struct SDL::modules& modulesInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU);
#ifndef NESTED_PARA
#ifdef NEWGRID_Tracklet
__global__ void createTrackletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU, unsigned int *index_gpu);
__global__ void createTrackletsFromTriplets(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::tracklets& trackletsInGPU, unsigned int *threadIdx_gpu, unsigned int * threadIdx_gpu_offset);
#endif
#else
__global__ void createTrackletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU);
__global__ void createTrackletsFromInnerInnerLowerModule(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU, unsigned int innerInnerLowerModuleIndex, unsigned int nInnerSegments, unsigned int innerInnerLowerModuleArrayIndex);

__global__ void createTrackletsFromTriplets(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::tracklets& trackletsInGPU/*, unsigned int *index_gpu*/);
__global__ void createTrackletsFromTripletsP2(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::tracklets& trackletsInGPU/*, unsigned int *index_gpu*/,unsigned int innerInnerLowerModuleArrayIndex, unsigned int nTriplets);
#endif

#ifndef NESTED_PARA
#ifdef NEWGRID_Pixel
__global__ void createPixelTrackletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::pixelTracklets& pixelTrackletsInGPU, unsigned int* threadIdx_gpu, unsigned int* threadIdx_gpu_offset);
__global__ void createPixelTrackletsInGPUFromMap(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::pixelTracklets& pixelTrackletsInGPU, unsigned int* connectedPixelSize, unsigned int* connectedPixelIndex, unsigned int nInnerSegs,unsigned int* seg_pix_gpu, unsigned int* seg_pix_gpu_offset, unsigned int totalSegs);
#endif
#else
__global__ void createPixelTrackletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::pixelTracklets& pixelTrackletsInGPU);


__global__ void createPixelTrackletsFromOuterInnerLowerModule(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::pixelTracklets& pixelTrackletsInGPU, unsigned int outerInnerLowerModuleIndex, unsigned int nInnerSegments, unsigned int nOuterSegments, unsigned int pixelModuleIndex, unsigned int pixelLowerModuleArrayIndex);
#endif
__global__ void createPixelTrackletsFromOuterInnerLowerModulev3(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::pixelTracklets& pixelTrackletsInGPU, unsigned int outerInnerLowerModuleIndex, unsigned int nInnerSegments, unsigned int nOuterSegments, unsigned int pixelModuleIndex, unsigned int pixelLowerModuleArrayIndex);

__global__ void createTrackletsWithAGapInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU);


__global__ void createTrackletsWithAGapFromInnerInnerLowerModule(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU, unsigned int innerInnerLowerModuleIndex, unsigned int nInnerSegments, unsigned int innerInnerLowerModuleArrayIndex);

#ifndef NESTED_PARA
#ifdef NEWGRID_Trips
__global__ void createTripletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, str\
uct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, unsigned int *index_gpu);
#endif
#else
__global__ void createTripletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU);

__global__ void createTripletsFromInnerInnerLowerModule(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, unsigned int innerInnerLowerModuleIndex, unsigned int nInnerSegments, unsigned int innerInnerLowerModuleArrayIndex);
#endif

#ifndef NESTED_PARA
#ifdef NEWGRID_Track
__global__ void createTrackCandidatesInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU, unsigned int *threadIdx_gpu, unsigned int* threadIdx_gpu_offset);

__global__ void createPixelTrackCandidatesInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::pixelTracklets& pixelTrackletsInGPU,struct SDL::tracklets& trackletsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU, unsigned int* threadIdx_gpu, unsigned int *threadIdx_gpu_offset);

#endif
#else
__global__ void createPixelTrackCandidatesInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::pixelTracklets& pixelTrackletsInGPU,struct SDL::tracklets& trackletsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU);

__global__ void createPixelTrackCandidatesFromOuterInnerInnerLowerModule(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU, struct SDL::pixelTracklets& pixelTrackletsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU, unsigned int pixelLowerModuleArrayIndex, unsigned int outerInnerInnerLowerModuleArrayIndex, unsigned int nPixelTracklets, unsigned int nOuterLayerTracklets, unsigned int nOuterLayerTriplets);

__global__ void createTrackCandidatesInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU);

__global__ void createTrackCandidatesFromInnerInnerInnerLowerModule(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU, unsigned int innerInnerInnerLowerModuleArrayIndex, unsigned int nInnerTracklets, unsigned int nInnerTriplets);
#endif

#ifdef NEWGRID_T5
__global__ void createQuintupletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::quintuplets& quintupletsInGPU, unsigned int* threadIdx_gpu, unsigned int* threadIdx_gpu_offset, int nTotalTriplets);

#else
__global__ void createQuintupletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::quintuplets& quintupletsInGPU);
#endif

#ifdef NEWGRID_pT3
__global__ void createPixelTripletsInGPUFromMap(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::pixelTriplets& pixelTripletsInGPU, unsigned int* connectedPixelSize, unsigned int* connectedPixelIndex, unsigned int nPixelSegments, unsigned int* seg_pix_gpu, unsigned int* seg_pix_gpu_offset, unsigned int totalSegs);
#else
__global__ void createPixelTripletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::pixelTriplets& pixelTripletsInGPU);

__global__ void createPixelTripletsFromOuterInnerLowerModule(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::pixelTriplets& pixelTripletsInGPU, unsigned int outerTripletInnerLowerModuleArrayIndex, unsigned int nPixelSegments, unsigned int nOuterTriplets, unsigned int pixelModuleIndex);
#endif

#ifdef NEWGRID_pT5

__global__ void createPixelQuintupletsInGPUFromMap(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::quintuplets& quintupletsInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, unsigned int* connectedPixelSize, unsigned int* connectedPixelIndex, unsigned int nPixelSegments, unsigned int* seg_pix_gpu, unsigned int* seg_pix_gpu_offset, unsigned int totalSegs);

#else
__global__ void createPixelQuintupletsFromFirstModule(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::quintuplets& quintupletsInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, unsigned int nPixelSegments, unsigned int nOuterQuintuplets, unsigned int firstLowerModuleArrayIndex, unsigned int pixelModuleIndex);

__global__ void createPixelQuintupletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::quintuplets& quintupletsInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU);
#endif

#endif
