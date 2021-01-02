#include "Layer.h"

void SDL::Layer::addMiniDoublet(std::shared_ptr<SDL::MiniDoublet> md)
{
    minidoublets_.push_back(md);
}

void SDL::Layer::addSegment(std::shared_ptr<SDL::Segment> sg)
{
    segments_.push_back(sg);
}

void SDL::Layer::addTracklet(std::shared_ptr<SDL::Tracklet> tl)
{
    tracklets_.push_back(tl);
}

void SDL::Layer::addTriplet(std::shared_ptr<SDL::Triplet> tp)
{
    triplets_.push_back(tp);
}

void SDL::Layer::addTrackCandidate(std::shared_ptr<SDL::TrackCandidate> tl)
{
    trackcandidates_.push_back(tl);
}

const std::vector<std::pair<std::pair<int, SDL::Layer::SubDet>, std::pair<int, SDL::Layer::SubDet>>> SDL::Layer::tracklet_compatible_layer_pairs_ =
{
    // {{1, SDL::Layer::Barrel},{2, SDL::Layer::Barrel}},
    // {{2, SDL::Layer::Barrel},{3, SDL::Layer::Barrel}},
    // {{3, SDL::Layer::Barrel},{4, SDL::Layer::Barrel}},
    // {{4, SDL::Layer::Barrel},{5, SDL::Layer::Barrel}},
    // {{5, SDL::Layer::Barrel},{6, SDL::Layer::Barrel}},
    // {{1, SDL::Layer::Endcap},{2, SDL::Layer::Endcap}},
    // {{2, SDL::Layer::Endcap},{3, SDL::Layer::Endcap}},
    // {{3, SDL::Layer::Endcap},{4, SDL::Layer::Endcap}},
    // {{4, SDL::Layer::Endcap},{5, SDL::Layer::Endcap}},
    // {{1, SDL::Layer::Barrel},{1, SDL::Layer::Endcap}},
    // {{2, SDL::Layer::Barrel},{1, SDL::Layer::Endcap}},
    // {{3, SDL::Layer::Barrel},{1, SDL::Layer::Endcap}},
    // {{4, SDL::Layer::Barrel},{1, SDL::Layer::Endcap}},
    // {{5, SDL::Layer::Barrel},{1, SDL::Layer::Endcap}}
    {{1, SDL::Layer::Barrel},{3, SDL::Layer::Barrel}},
    // {{1, SDL::Layer::Barrel},{4, SDL::Layer::Barrel}},
    // {{1, SDL::Layer::Barrel},{5, SDL::Layer::Barrel}},
    {{2, SDL::Layer::Barrel},{4, SDL::Layer::Barrel}},
    // {{2, SDL::Layer::Barrel},{5, SDL::Layer::Barrel}},
    {{3, SDL::Layer::Barrel},{5, SDL::Layer::Barrel}},
    // {{1, SDL::Layer::Barrel},{1, SDL::Layer::Endcap}},
    // {{2, SDL::Layer::Barrel},{1, SDL::Layer::Endcap}},
    // {{3, SDL::Layer::Barrel},{1, SDL::Layer::Endcap}},
    // {{4, SDL::Layer::Barrel},{1, SDL::Layer::Endcap}},
    // {{5, SDL::Layer::Barrel},{1, SDL::Layer::Endcap}},
};

const std::vector<std::pair<std::pair<int, SDL::Layer::SubDet>, std::pair<int, SDL::Layer::SubDet>>> SDL::Layer::segment_compatible_layer_pairs_ =
{
    {{1, SDL::Layer::Barrel},{2, SDL::Layer::Barrel}},
    {{2, SDL::Layer::Barrel},{3, SDL::Layer::Barrel}},
    {{3, SDL::Layer::Barrel},{4, SDL::Layer::Barrel}},
    {{4, SDL::Layer::Barrel},{5, SDL::Layer::Barrel}},
    {{5, SDL::Layer::Barrel},{6, SDL::Layer::Barrel}},
    {{1, SDL::Layer::Barrel},{1, SDL::Layer::Endcap}},
    {{2, SDL::Layer::Barrel},{1, SDL::Layer::Endcap}},
    {{3, SDL::Layer::Barrel},{1, SDL::Layer::Endcap}},
    {{4, SDL::Layer::Barrel},{1, SDL::Layer::Endcap}},
    {{5, SDL::Layer::Barrel},{1, SDL::Layer::Endcap}},
    {{1, SDL::Layer::Endcap},{2, SDL::Layer::Endcap}},
    {{2, SDL::Layer::Endcap},{3, SDL::Layer::Endcap}},
    {{2, SDL::Layer::Endcap},{4, SDL::Layer::Endcap}},
    {{2, SDL::Layer::Endcap},{5, SDL::Layer::Endcap}},
    {{3, SDL::Layer::Endcap},{4, SDL::Layer::Endcap}},
    {{4, SDL::Layer::Endcap},{5, SDL::Layer::Endcap}},
};


SDL::Layer::Layer()
{
}

SDL::Layer::Layer(int layerIdx, unsigned short subdet) : layer_idx_(layerIdx), subdet_(subdet)
{
}

SDL::Layer::~Layer()
{
}

const unsigned short& SDL::Layer::subdet() const
{
    return subdet_;
}

const int& SDL::Layer::layerIdx() const
{
    return layer_idx_;
}

const std::vector<std::shared_ptr<SDL::MiniDoublet>>& SDL::Layer::getMiniDoubletPtrs() const
{
    return minidoublets_;
}

const std::vector<std::shared_ptr<SDL::Segment>>& SDL::Layer::getSegmentPtrs() const
{
    return segments_;
}

const std::vector<std::shared_ptr<SDL::Tracklet>>& SDL::Layer::getTrackletPtrs() const
{
    return tracklets_;
}

const std::vector<std::shared_ptr<SDL::Triplet>>& SDL::Layer::getTripletPtrs() const
{
    return triplets_;
}

const std::vector<std::shared_ptr<SDL::TrackCandidate>>& SDL::Layer::getTrackCandidatePtrs() const
{
    return trackcandidates_;
}

const std::vector<std::pair<std::pair<int, SDL::Layer::SubDet>, std::pair<int, SDL::Layer::SubDet>>>& SDL::Layer::getListOfTrackletCompatibleLayerPairs()
{
    return SDL::Layer::tracklet_compatible_layer_pairs_;
}

const std::vector<std::pair<std::pair<int, SDL::Layer::SubDet>, std::pair<int, SDL::Layer::SubDet>>>& SDL::Layer::getListOfSegmentCompatibleLayerPairs()
{
    return SDL::Layer::segment_compatible_layer_pairs_;
}

void SDL::Layer::setLayerIdx(int lidx)
{
    layer_idx_ = lidx;
}

void SDL::Layer::setSubDet(SDL::Layer::SubDet subdet)
{
    subdet_ = subdet;
}


