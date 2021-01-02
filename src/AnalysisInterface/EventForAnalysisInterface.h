#ifndef ANALYSIS_INTERFACE_EVENT_H
#define ANALYSIS_INTERFACE_EVENT_H

#include <vector>
#include <map>
#include <tuple>
#include <memory>

#include "Module.h"
#include "Hit.h"
#include "MiniDoublet.h"
#include "Segment.h"
#include "Tracklet.h"
#include "MathUtil.h"
#include "TrackCandidate.h"
#include "Layer.h"

#include "SDL/Module.cuh"
#include "SDL/Hit.cuh"
#include "SDL/MiniDoublet.cuh"
#include "SDL/Segment.cuh"
#include "SDL/Tracklet.cuh"
#include "SDL/Triplet.cuh"
#include "SDL/TrackCandidate.cuh"

namespace SDL
{
    class EventForAnalysisInterface
    {
        private:
            std::map<unsigned int, std::shared_ptr<Module>> moduleMapByIndex_;
            std::map<unsigned int, unsigned int> detIdToIndex_;
            std::map<unsigned int, std::shared_ptr<Hit>> hits_;
            std::map<unsigned int, std::shared_ptr<Hit>> hits_2s_edges_;
            std::map<unsigned int, std::shared_ptr<MiniDoublet>> miniDoublets_;
            std::map<unsigned int, std::shared_ptr<Segment>> segments_;
            std::map<unsigned int, std::shared_ptr<Tracklet>> tracklets_;
            std::map<unsigned int, std::shared_ptr<Triplet>> triplets_;
            std::map<unsigned int, std::shared_ptr<TrackCandidate>> trackCandidates_;

            std::map<int, std::shared_ptr<Layer>> barrelLayers_;

            // map of endcap layers (this holds the actual instances)
            std::map<int, std::shared_ptr<Layer>> endcapLayers_;
            Layer pixelLayer_;
            std::vector<std::shared_ptr<Layer>> layerPtrs_;

                    
            std::vector<std::shared_ptr<Module>> modulePointers;
            std::vector<std::shared_ptr<Module>> lowerModulePointers;
            std::vector<std::shared_ptr<Hit>> hitPointers;
            std::vector<std::shared_ptr<MiniDoublet>> mdPointers;
            std::vector<std::shared_ptr<Segment>> segmentPointers;
            std::vector<std::shared_ptr<Tracklet>> trackletPointers;
            std::vector<std::shared_ptr<Triplet>> tripletPointers;
            std::vector<std::shared_ptr<TrackCandidate>> trackCandidatePointers;

        public:
	        EventForAnalysisInterface(struct modules* modulesInGPU, struct hits* hitsInGPU, struct miniDoublets* mdsInGPU, struct segments* segmentsInGPU, struct tracklets* trackletsInGPU, struct triplets* tripletsInGPU, struct trackCandidates* trackCandidatesInGPU);
	        ~EventForAnalysisInterface();

            void addModulesToAnalysisInterface(struct modules& modulesInGPU, struct miniDoublets* mdsInGPU, struct segments* segmentsInGPU, struct tracklets* trackletsInGPU, struct triplets* tripletsInGPU, struct trackCandidates* trackCandidatesInGPU);
            void getModule(unsigned int detId);
            void addHitsToAnalysisInterface(struct hits& hitsInGPU);

            void createLayers();
            Layer& getLayer(int layer, SDL::Layer::SubDet subdet);
            const std::vector<std::shared_ptr<Layer>> getLayerPtrs() const;
            Layer& getPixelLayer();

            void addMDsToAnalysisInterface(struct miniDoublets& mdsInGPU);
            void addSegmentsToAnalysisInterface(struct segments& segmentsInGPU);
            void addTrackletsToAnalysisInterface(struct tracklets& trackletsInGPU);
            void addTripletsToAnalysisInterface(struct triplets& tripletsInGPU);
            void addTrackCandidatesToAnalysisInterface(struct trackCandidates& trackCandidatesInGPU);
            void printTrackCandidateLayers(std::shared_ptr<TrackCandidate> tc);

        //add the get list of functions here
        const std::vector<std::shared_ptr<Module>> getModulePtrs() const;
        const std::vector<std::shared_ptr<Module>> getLowerModulePtrs() const;
    };
}
#endif
