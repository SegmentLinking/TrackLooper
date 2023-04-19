#ifndef Event_cuh
#define Event_cuh

#include "Hit.cuh"
#include "Module.cuh"
#include "Segment.cuh"
#include "Triplet.cuh"
#include "Kernels.cuh"
#include "Quintuplet.cuh"
#include "MiniDoublet.cuh"
#include "PixelTriplet.cuh"
#include "TrackCandidate.cuh"
#include "Constants.cuh"

#include "allocate.h"

namespace SDL
{
    class Event
    {
    private:
        cudaStream_t stream;
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
        int dev;
        int nTotalSegments;
        struct objectRanges* rangesInGPU;
        struct hits* hitsInGPU;
        struct miniDoublets* mdsInGPU;
        struct segments* segmentsInGPU;
        struct triplets* tripletsInGPU;
        struct quintuplets* quintupletsInGPU;
        struct trackCandidates* trackCandidatesInGPU;
        struct pixelTriplets* pixelTripletsInGPU;
        struct pixelQuintuplets* pixelQuintupletsInGPU;

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

        int* superbinCPU;
        int8_t* pixelTypeCPU;
    public:
        Event(cudaStream_t estream);
        ~Event();
        void resetEvent();

        void addHitToEvent(std::vector<float> x, std::vector<float> y, std::vector<float> z, std::vector<unsigned int> detId, std::vector<unsigned int> idxInNtuple); //call the appropriate hit function, then increment the counter here
        void addPixelSegmentToEvent(std::vector<unsigned int> hitIndices0,std::vector<unsigned int> hitIndices1,std::vector<unsigned int> hitIndices2,std::vector<unsigned int> hitIndices3, std::vector<float> dPhiChange, std::vector<float> ptIn, std::vector<float> ptErr, std::vector<float> px, std::vector<float> py, std::vector<float> pz, std::vector<float> eta, std::vector<float> etaErr, std::vector<float> phi, std::vector<int> charge, std::vector<unsigned int> seedIdx, std::vector<int> superbin, std::vector<int8_t> pixelType, std::vector<char> isQuad);

        //Functions that map the objects to the appropriate modules
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

        int getNumberOfTrackCandidates();
        int getNumberOfPixelTrackCandidates();
        int getNumberOfPT5TrackCandidates();
        int getNumberOfPT3TrackCandidates();
        int getNumberOfT5TrackCandidates();
        int getNumberOfPLSTrackCandidates();

        unsigned int getNumberOfQuintuplets();
        unsigned int getNumberOfQuintupletsByLayer(unsigned int layer);
        unsigned int getNumberOfQuintupletsByLayerBarrel(unsigned int layer);
        unsigned int getNumberOfQuintupletsByLayerEndcap(unsigned int layer);

        int getNumberOfPixelTriplets();
        int getNumberOfPixelQuintuplets();

        unsigned int getNumberOfExtendedTracks();
        unsigned int getNumberOfT3T3ExtendedTracks();

        objectRanges* getRanges();
        hits* getHits();
        miniDoublets* getMiniDoublets();
        segments* getSegments() ;
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
    extern uint16_t nModules;
    extern uint16_t nLowerModules;
    void initModules(const char* moduleMetaDataFilePath="data/centroid.txt"); //read from file and init
    void cleanModules();
    void initModulesHost(); //read from file and init
    extern struct pixelMap* pixelMapping;
}
#endif
