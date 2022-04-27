#include "Layer.h"

const std::vector<std::pair<std::pair<int, SDL::CPU::Layer::SubDet>, std::pair<int, SDL::CPU::Layer::SubDet>>> SDL::CPU::Layer::tracklet_compatible_layer_pairs_ =
{
    // {{1, SDL::CPU::Layer::Barrel},{2, SDL::CPU::Layer::Barrel}},
    // {{2, SDL::CPU::Layer::Barrel},{3, SDL::CPU::Layer::Barrel}},
    // {{3, SDL::CPU::Layer::Barrel},{4, SDL::CPU::Layer::Barrel}},
    // {{4, SDL::CPU::Layer::Barrel},{5, SDL::CPU::Layer::Barrel}},
    // {{5, SDL::CPU::Layer::Barrel},{6, SDL::CPU::Layer::Barrel}},
    // {{1, SDL::CPU::Layer::Endcap},{2, SDL::CPU::Layer::Endcap}},
    // {{2, SDL::CPU::Layer::Endcap},{3, SDL::CPU::Layer::Endcap}},
    // {{3, SDL::CPU::Layer::Endcap},{4, SDL::CPU::Layer::Endcap}},
    // {{4, SDL::CPU::Layer::Endcap},{5, SDL::CPU::Layer::Endcap}},
    // {{1, SDL::CPU::Layer::Barrel},{1, SDL::CPU::Layer::Endcap}},
    // {{2, SDL::CPU::Layer::Barrel},{1, SDL::CPU::Layer::Endcap}},
    // {{3, SDL::CPU::Layer::Barrel},{1, SDL::CPU::Layer::Endcap}},
    // {{4, SDL::CPU::Layer::Barrel},{1, SDL::CPU::Layer::Endcap}},
    // {{5, SDL::CPU::Layer::Barrel},{1, SDL::CPU::Layer::Endcap}}
    {{1, SDL::CPU::Layer::Barrel},{3, SDL::CPU::Layer::Barrel}},
    // {{1, SDL::CPU::Layer::Barrel},{4, SDL::CPU::Layer::Barrel}},
    // {{1, SDL::CPU::Layer::Barrel},{5, SDL::CPU::Layer::Barrel}},
    {{2, SDL::CPU::Layer::Barrel},{4, SDL::CPU::Layer::Barrel}},
    // {{2, SDL::CPU::Layer::Barrel},{5, SDL::CPU::Layer::Barrel}},
    {{3, SDL::CPU::Layer::Barrel},{5, SDL::CPU::Layer::Barrel}},
    // {{1, SDL::CPU::Layer::Barrel},{1, SDL::CPU::Layer::Endcap}},
    // {{2, SDL::CPU::Layer::Barrel},{1, SDL::CPU::Layer::Endcap}},
    // {{3, SDL::CPU::Layer::Barrel},{1, SDL::CPU::Layer::Endcap}},
    // {{4, SDL::CPU::Layer::Barrel},{1, SDL::CPU::Layer::Endcap}},
    // {{5, SDL::CPU::Layer::Barrel},{1, SDL::CPU::Layer::Endcap}},
};

const std::vector<std::pair<std::pair<int, SDL::CPU::Layer::SubDet>, std::pair<int, SDL::CPU::Layer::SubDet>>> SDL::CPU::Layer::segment_compatible_layer_pairs_ =
{
    {{1, SDL::CPU::Layer::Barrel},{2, SDL::CPU::Layer::Barrel}},
    {{2, SDL::CPU::Layer::Barrel},{3, SDL::CPU::Layer::Barrel}},
    {{3, SDL::CPU::Layer::Barrel},{4, SDL::CPU::Layer::Barrel}},
    {{4, SDL::CPU::Layer::Barrel},{5, SDL::CPU::Layer::Barrel}},
    {{5, SDL::CPU::Layer::Barrel},{6, SDL::CPU::Layer::Barrel}},
    {{1, SDL::CPU::Layer::Barrel},{1, SDL::CPU::Layer::Endcap}},
    {{2, SDL::CPU::Layer::Barrel},{1, SDL::CPU::Layer::Endcap}},
    {{3, SDL::CPU::Layer::Barrel},{1, SDL::CPU::Layer::Endcap}},
    {{4, SDL::CPU::Layer::Barrel},{1, SDL::CPU::Layer::Endcap}},
    {{5, SDL::CPU::Layer::Barrel},{1, SDL::CPU::Layer::Endcap}},
    {{1, SDL::CPU::Layer::Endcap},{2, SDL::CPU::Layer::Endcap}},
    {{2, SDL::CPU::Layer::Endcap},{3, SDL::CPU::Layer::Endcap}},
    {{2, SDL::CPU::Layer::Endcap},{4, SDL::CPU::Layer::Endcap}},
    {{2, SDL::CPU::Layer::Endcap},{5, SDL::CPU::Layer::Endcap}},
    {{3, SDL::CPU::Layer::Endcap},{4, SDL::CPU::Layer::Endcap}},
    {{4, SDL::CPU::Layer::Endcap},{5, SDL::CPU::Layer::Endcap}},
};


SDL::CPU::Layer::Layer()
{
}

SDL::CPU::Layer::Layer(int layerIdx, unsigned short subdet) : layer_idx_(layerIdx), subdet_(subdet)
{
}

SDL::CPU::Layer::~Layer()
{
}

const unsigned short& SDL::CPU::Layer::subdet() const
{
    return subdet_;
}

const int& SDL::CPU::Layer::layerIdx() const
{
    return layer_idx_;
}

const std::vector<SDL::CPU::MiniDoublet*>& SDL::CPU::Layer::getMiniDoubletPtrs() const
{
    return minidoublets_;
}

const std::vector<SDL::CPU::Segment*>& SDL::CPU::Layer::getSegmentPtrs() const
{
    return segments_;
}

const std::vector<SDL::CPU::Triplet*>& SDL::CPU::Layer::getTripletPtrs() const
{
    return triplets_;
}

const std::vector<SDL::CPU::Tracklet*>& SDL::CPU::Layer::getTrackletPtrs() const
{
    return tracklets_;
}

const std::vector<SDL::CPU::TrackCandidate*>& SDL::CPU::Layer::getTrackCandidatePtrs() const
{
    return trackcandidates_;
}

const std::vector<std::pair<std::pair<int, SDL::CPU::Layer::SubDet>, std::pair<int, SDL::CPU::Layer::SubDet>>>& SDL::CPU::Layer::getListOfTrackletCompatibleLayerPairs()
{
    return SDL::CPU::Layer::tracklet_compatible_layer_pairs_;
}

const std::vector<std::pair<std::pair<int, SDL::CPU::Layer::SubDet>, std::pair<int, SDL::CPU::Layer::SubDet>>>& SDL::CPU::Layer::getListOfSegmentCompatibleLayerPairs()
{
    return SDL::CPU::Layer::segment_compatible_layer_pairs_;
}

void SDL::CPU::Layer::setLayerIdx(int lidx)
{
    layer_idx_ = lidx;
}

void SDL::CPU::Layer::setSubDet(SDL::CPU::Layer::SubDet subdet)
{
    subdet_ = subdet;
}

void SDL::CPU::Layer::addMiniDoublet(SDL::CPU::MiniDoublet* md)
{
    minidoublets_.push_back(md);
}

void SDL::CPU::Layer::addSegment(SDL::CPU::Segment* sg)
{
    segments_.push_back(sg);
}

void SDL::CPU::Layer::addTriplet(SDL::CPU::Triplet* tp)
{
    triplets_.push_back(tp);
}

void SDL::CPU::Layer::addTracklet(SDL::CPU::Tracklet* tl)
{
    tracklets_.push_back(tl);
}

void SDL::CPU::Layer::addTrackCandidate(SDL::CPU::TrackCandidate* tc)
{
    trackcandidates_.push_back(tc);
}

namespace SDL
{
    namespace CPU
    {
        std::ostream& operator<<(std::ostream& out, const Layer& layer)
        {
            out << "==============================" << std::endl;
            out << "Layer(layerIdx=" << layer.layerIdx();
            out << ", subdet=" << layer.subdet();
            out << ")" << std::endl;
            out << "==============================" << std::endl;
            for (auto& segmentPtr : layer.segments_)
                out << segmentPtr << std::endl;
            for (auto& trackletPtr : layer.tracklets_)
                out << trackletPtr << std::endl;
            out << "" << std::endl;

            return out;
        }

        std::ostream& operator<<(std::ostream& out, const Layer* layer)
        {
            out << *layer;
            return out;
        }

    }


}
