#ifndef AnalysisInterface_Layer_h
#define AnalysisInterface_Layer_h
#include <vector>
#include <tuple>

#include "MiniDoublet.h"
#include "Segment.h"
//#include "Triplet.h"
#include "Tracklet.h"
//#include "TrackCandidate.h"

namespace SDL
{
    class MiniDoublet;
    class Segment;
    class Triplet;
    class TrackCandidate;
}

namespace SDL
{

    class Layer
    {

        public:

            enum BarrelLayers
            {
                BarrelLayer0 = 1,
                BarrelLayer1,
                BarrelLayer2,
                BarrelLayer3,
                BarrelLayer4,
                BarrelLayer5,
                nBarrelLayer
            };

            enum EndcapLayers
            {
                EndcapLayer0 = 1,
                EndcapLayer1,
                EndcapLayer2,
                EndcapLayer3,
                EndcapLayer4,
                EndcapLayer5,
                nEndcapLayer
            };

        private:

            // which layer this is
            int layer_idx_;

        public:
            enum SubDet
            {
                Barrel = 5,
                Endcap = 4
            };

        private:
            // whether it is barrel or endcap (this distinction was added as horizontal vs. vertical alignment is a pretty big geometrical difference.)
            unsigned short subdet_;

            // vector of mini-doublets
            std::vector<std::shared_ptr<MiniDoublet>> minidoublets_;

            // vector of segments (This is used for the inefficient approach of looping over all segements in each layer)
            std::vector<std::shared_ptr<Segment>> segments_;

            // vector of triplets
            std::vector<std::shared_ptr<Triplet>> triplets_;

            // vector of tracklets
            std::vector<std::shared_ptr<Tracklet>> tracklets_;

            // vector of trackcandidates
            std::vector<std::shared_ptr<TrackCandidate>> trackcandidates_;

        private:

            // Compatible layer information for Tracklet
            static const std::vector<std::pair<std::pair<int, SubDet>, std::pair<int, SubDet>>> tracklet_compatible_layer_pairs_;

            // Compatible layer information for Segments
            static const std::vector<std::pair<std::pair<int, SubDet>, std::pair<int, SubDet>>> segment_compatible_layer_pairs_;


        public:

            // constructor/destructor
            Layer();
            Layer(int layerIdx, unsigned short subdet);
            ~Layer();

            // accessor
            const unsigned short& subdet() const;
            const int& layerIdx() const;
            const std::vector<std::shared_ptr<MiniDoublet>>& getMiniDoubletPtrs() const;
            const std::vector<std::shared_ptr<Segment>>& getSegmentPtrs() const;
            const std::vector<std::shared_ptr<Triplet>>& getTripletPtrs() const;
            const std::vector<std::shared_ptr<Tracklet>>& getTrackletPtrs() const;
            const std::vector<std::shared_ptr<TrackCandidate>>& getTrackCandidatePtrs() const;
            static const std::vector<std::pair<std::pair<int, SubDet>, std::pair<int, SubDet>>>& getListOfTrackletCompatibleLayerPairs();
            static const std::vector<std::pair<std::pair<int, SubDet>, std::pair<int, SubDet>>>& getListOfSegmentCompatibleLayerPairs();

            // modifier
            void setLayerIdx(int lidx);
            void setSubDet(SubDet subdet);
            void addMiniDoublet(std::shared_ptr<MiniDoublet> md);
            void addSegment(std::shared_ptr<Segment> sg);
            void addTriplet(std::shared_ptr<Triplet> tp);
            void addTracklet(std::shared_ptr<Tracklet> tl);
            void addTrackCandidate(std::shared_ptr<TrackCandidate> tl);

            // printing
            friend std::ostream& operator<<(std::ostream& os, const Layer& layer);
            friend std::ostream& operator<<(std::ostream& os, const Layer* layer);
    };
}

#endif
