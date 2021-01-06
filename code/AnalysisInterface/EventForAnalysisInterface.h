#ifndef ANALYSIS_INTERFACE_EVENT_H
#define ANALYSIS_INTERFACE_EVENT_H

#include <vector>
#include <map>
#include <tuple>

#include "Module.h"
#include "Hit.h"
#include "MiniDoublet.h"
#include "Segment.h"
#include "Tracklet.h"
#include "MathUtil.h"
#include "Layer.h"

#include "SDL/Module.cuh"
#include "SDL/Hit.cuh"
#include "SDL/MiniDoublet.cuh"
#include "SDL/Segment.cuh"
#include "SDL/Tracklet.cuh"
#include "SDL/Triplet.cuh"

namespace SDL
{
    class EventForAnalysisInterface
    {
        private:
            std::map<unsigned int, Module*> moduleMapByIndex_;
            std::map<unsigned int, unsigned int> detIdToIndex_;
            std::map<unsigned int, Hit*> hits_;
            std::map<unsigned int, Hit*> hits_2s_edges_;
            std::map<unsigned int, MiniDoublet*> miniDoublets_;
            std::map<unsigned int, Segment*> segments_;
            std::map<unsigned int, Tracklet*> tracklets_;
            std::map<unsigned int, Triplet*> triplets_;

            std::map<int, Layer> barrelLayers_;

            // map of endcap layers (this holds the actual instances)
            std::map<int, Layer> endcapLayers_;
            std::vector<Layer*> layerPtrs_;

        
            std::vector<Module*> modulePointers;
            std::vector<Module*> lowerModulePointers;
            std::vector<Hit*> hitPointers;
            std::vector<MiniDoublet*> mdPointers;
            std::vector<Segment*> segmentPointers;
            std::vector<Tracklet*> trackletPointers;
            std::vector<Triplet*> tripletPointers;

        public:
	        EventForAnalysisInterface(struct modules* modulesInGPU, struct hits* hitsInGPU, struct miniDoublets* mdsInGPU, struct segments* segmentsInGPU, struct tracklets* trackletsInGPU, struct triplets* tripletsInGPU);

            void addModulesToAnalysisInterface(struct modules& modulesInGPU, struct miniDoublets* mdsInGPU, struct segments* segmentsInGPU, struct tracklets* trackletsInGPU, struct triplets* tripletsInGPU);
            void getModule(unsigned int detId);
            void addHitsToAnalysisInterface(struct hits& hitsInGPU);

            void createLayers();
            Layer& getLayer(int layer, SDL::Layer::SubDet subdet);
            const std::vector<Layer*> getLayerPtrs() const;


            void addMDsToAnalysisInterface(struct miniDoublets& mdsInGPU);
            void addSegmentsToAnalysisInterface(struct segments& segmentsInGPU);
            void addTrackletsToAnalysisInterface(struct tracklets& trackletsInGPU);
            void addTripletsToAnalysisInterface(struct triplets& tripletsInGPU);

        //add the get list of functions here
        const std::vector<Module*> getModulePtrs() const;
        const std::vector<Module*> getLowerModulePtrs() const;
    };
}
#endif
