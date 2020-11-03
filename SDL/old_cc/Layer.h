#ifndef Layer_h
#define Layer_h

#include <vector>
#include <tuple>

#include "MiniDoublet.cuh"
#include "Segment.h"
#include "Triplet.h"
#include "Tracklet.h"
#include "TrackCandidate.h"

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
            std::vector<MiniDoublet*> minidoublets_;

            // vector of segments (This is used for the inefficient approach of looping over all segements in each layer)
            std::vector<Segment*> segments_;

            // vector of triplets
            std::vector<Triplet*> triplets_;

            // vector of tracklets
            std::vector<Tracklet*> tracklets_;

            // vector of trackcandidates
            std::vector<TrackCandidate*> trackcandidates_;

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
            const std::vector<MiniDoublet*>& getMiniDoubletPtrs() const;
            const std::vector<Segment*>& getSegmentPtrs() const;
            const std::vector<Triplet*>& getTripletPtrs() const;
            const std::vector<Tracklet*>& getTrackletPtrs() const;
            const std::vector<TrackCandidate*>& getTrackCandidatePtrs() const;
            static const std::vector<std::pair<std::pair<int, SubDet>, std::pair<int, SubDet>>>& getListOfTrackletCompatibleLayerPairs();
            static const std::vector<std::pair<std::pair<int, SubDet>, std::pair<int, SubDet>>>& getListOfSegmentCompatibleLayerPairs();

            // modifier
            void setLayerIdx(int lidx);
            void setSubDet(SubDet subdet);
            void addMiniDoublet(MiniDoublet* md);
            void addSegment(Segment* sg);
            void addTriplet(Triplet* tp);
            void addTracklet(Tracklet* tl);
            void addTrackCandidate(TrackCandidate* tl);

            // printing
            friend std::ostream& operator<<(std::ostream& os, const Layer& layer);
            friend std::ostream& operator<<(std::ostream& os, const Layer* layer);
    };
}

#endif
