#ifndef Event_h
#define Event_h

#include "Hit.h"
#include "Module.h"
#include "Segment.h"
#include "Triplet.h"
#include "Kernels.h"
#include "Quintuplet.h"
#include "MiniDoublet.h"
#include "PixelTriplet.h"
#include "TrackCandidate.h"
#include "Constants.h"

namespace SDL
{
    class Event
    {
    private:
        QueueAcc queue;
        bool addObjects;

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

        //Device stuff
        unsigned int nTotalSegments;
        struct objectRanges* rangesInGPU;
        struct objectRangesBuffer<Acc>* rangesBuffers;
        struct hits* hitsInGPU;
        struct hitsBuffer<Acc>* hitsBuffers;
        struct miniDoublets* mdsInGPU;
        struct miniDoubletsBuffer<Acc>* miniDoubletsBuffers;
        struct segments* segmentsInGPU;
        struct segmentsBuffer<Acc>* segmentsBuffers;
        struct triplets* tripletsInGPU;
        struct tripletsBuffer<Acc>* tripletsBuffers;
        struct quintuplets* quintupletsInGPU;
        struct quintupletsBuffer<Acc>* quintupletsBuffers;
        struct trackCandidates* trackCandidatesInGPU;
        struct trackCandidatesBuffer<Acc>* trackCandidatesBuffers;
        struct pixelTriplets* pixelTripletsInGPU;
        struct pixelTripletsBuffer<Acc>* pixelTripletsBuffers;
        struct pixelQuintuplets* pixelQuintupletsInGPU;
        struct pixelQuintupletsBuffer<Acc>* pixelQuintupletsBuffers;

        //CPU interface stuff
        objectRangesBuffer<alpaka::DevCpu>* rangesInCPU;
        hitsBuffer<alpaka::DevCpu>* hitsInCPU;
        miniDoubletsBuffer<alpaka::DevCpu>* mdsInCPU;
        segmentsBuffer<alpaka::DevCpu>* segmentsInCPU;
        tripletsBuffer<alpaka::DevCpu>* tripletsInCPU;
        trackCandidatesBuffer<alpaka::DevCpu>* trackCandidatesInCPU;
        modulesBuffer<alpaka::DevCpu>* modulesInCPU;
        modulesBuffer<alpaka::DevCpu>* modulesInCPUFull;
        quintupletsBuffer<alpaka::DevCpu>* quintupletsInCPU;
        pixelTripletsBuffer<alpaka::DevCpu>* pixelTripletsInCPU;
        pixelQuintupletsBuffer<alpaka::DevCpu>* pixelQuintupletsInCPU;

        void init(bool verbose);

        int* superbinCPU;
        int8_t* pixelTypeCPU;
    public:
        // Standalone constructor that has each event object create its own queue.
        Event(bool verbose);
        // Constructor used for CMSSW integration. Uses an external queue.
        template <typename TQueue>
        Event(bool verbose, const TQueue& q): queue(q)
        {
            init(verbose);
        }
        void resetEvent();

        void addHitToEvent(std::vector<float> x, std::vector<float> y, std::vector<float> z, std::vector<unsigned int> detId, std::vector<unsigned int> idxInNtuple); //call the appropriate hit function, then increment the counter here
        void addPixelSegmentToEvent(std::vector<unsigned int> hitIndices0,std::vector<unsigned int> hitIndices1,std::vector<unsigned int> hitIndices2,std::vector<unsigned int> hitIndices3, std::vector<float> dPhiChange, std::vector<float> ptIn, std::vector<float> ptErr, std::vector<float> px, std::vector<float> py, std::vector<float> pz, std::vector<float> eta, std::vector<float> etaErr, std::vector<float> phi, std::vector<int> charge, std::vector<unsigned int> seedIdx, std::vector<int> superbin, std::vector<int8_t> pixelType, std::vector<char> isQuad);

        /*functions that map the objects to the appropriate modules*/
        void addMiniDoubletsToEventExplicit();
        void addSegmentsToEventExplicit();
        void addTripletsToEventExplicit();
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

        objectRangesBuffer<alpaka::DevCpu>* getRanges();
        hitsBuffer<alpaka::DevCpu>* getHits();
        hitsBuffer<alpaka::DevCpu>* getHitsInCMSSW();
        miniDoubletsBuffer<alpaka::DevCpu>* getMiniDoublets();
        segmentsBuffer<alpaka::DevCpu>* getSegments() ;
        tripletsBuffer<alpaka::DevCpu>* getTriplets();
        quintupletsBuffer<alpaka::DevCpu>* getQuintuplets();
        trackCandidatesBuffer<alpaka::DevCpu>* getTrackCandidates();
        trackCandidatesBuffer<alpaka::DevCpu>* getTrackCandidatesInCMSSW();
        pixelTripletsBuffer<alpaka::DevCpu>* getPixelTriplets();
        pixelQuintupletsBuffer<alpaka::DevCpu>* getPixelQuintuplets();
        modulesBuffer<alpaka::DevCpu>* getModules();
        modulesBuffer<alpaka::DevCpu>* getFullModules();
    };

    //global stuff
    
    static SDL::modules* modulesInGPU() { static SDL::modules* modulesInGPU_ = new SDL::modules(); return modulesInGPU_;}
    static SDL::modulesBuffer<Acc>* modulesBuffers() { static SDL::modulesBuffer<Acc>* modulesBuffers_ = new SDL::modulesBuffer<Acc>(devAcc); return modulesBuffers_;}
//    static std::shared_ptr<SDL::pixelMap> pixelMapping = std::make_shared<pixelMap>();
    static uint16_t& nModules() { static uint16_t nModules_; return nModules_;}
    static uint16_t& nLowerModules() {static uint16_t nLowerModules_; return nLowerModules_;}    

//    extern SDL::modules* modulesInGPU;
//    extern SDL::modulesBuffer<Acc>* modulesBuffers;
//    extern uint16_t nModules;
//    extern uint16_t nLowerModules;
    void initModules(const char* moduleMetaDataFilePath="data/centroid.txt"); //read from file and init
    void freeModules();
    void initModulesHost(); //read from file and init
    extern std::shared_ptr<SDL::pixelMap> pixelMapping;
}
#endif
