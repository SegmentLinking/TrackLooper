#include "TrackCandidate.h"

SDL::TrackCandidate::TrackCandidate(std::shared_ptr<TrackletBase> innerTrackletPtr, std::shared_ptr<TrackletBase> outerTrackletPtr, short trackCandidateType)
{
    innerTrackletPtr_ = innerTrackletPtr;
    outerTrackletPtr_ = outerTrackletPtr;
    trackCandidateType_ = trackCandidateType;
}

std::shared_ptr<SDL::TrackletBase> SDL::TrackCandidate::innerTrackletBasePtr() const
{
    return innerTrackletPtr_;
}

std::shared_ptr<SDL::TrackletBase> SDL::TrackCandidate::outerTrackletBasePtr() const
{
    return outerTrackletPtr_;
}

std::shared_ptr<SDL::Tracklet> SDL::TrackCandidate::innerTrackletPtr() const
{
    if (std::dynamic_pointer_cast<Tracklet>(innerTrackletPtr_))
    {
        return std::dynamic_pointer_cast<Tracklet>(innerTrackletPtr_);
    }
    else
    {
       std::cout << "TrackCandidate::innerTrackletPtr() ERROR - asked for innerTracklet when this TrackCandidate doesn't have one. (maybe it has triplets?)" << std::endl;
        throw std::logic_error("");
    }

}

std::shared_ptr<SDL::Tracklet> SDL::TrackCandidate::outerTrackletPtr() const
{
    if (std::dynamic_pointer_cast<Tracklet>(outerTrackletPtr_))
    {
        return std::dynamic_pointer_cast<Tracklet>(outerTrackletPtr_);
    }
    else
    {
        std::cout << "TrackCandidate::outerTrackletPtr() ERROR - asked for outerTracklet when this TrackCandidate doesn't have one. (maybe it has triplets?)" << std::endl;
        throw std::logic_error("");
    }

}

std::shared_ptr<SDL::Triplet> SDL::TrackCandidate::innerTripletPtr() const
{
    if (std::dynamic_pointer_cast<Triplet>(innerTrackletPtr_))
    {
        return std::dynamic_pointer_cast<Triplet>(innerTrackletPtr_);
    }
    else
    {
        std::cout << "TrackCandidate::innerTripletPtr() ERROR - asked for innerTriplet when this TrackCandidate doesn't have one. (maybe it has tracklets?)" << std::endl;
        throw std::logic_error("");
    }
}

std::shared_ptr<SDL::Triplet> SDL::TrackCandidate::outerTripletPtr() const
{
    if (std::dynamic_pointer_cast<Triplet>(outerTrackletPtr_))
    {
        return std::dynamic_pointer_cast<Triplet>(outerTrackletPtr_);
    }
    else
    {
        std::cout << "TrackCandidate::outerTripletPtr() ERROR - asked for outerTriplet when this TrackCandidate doesn't have one. (maybe it has tracklets?)" << std::endl;
        throw std::logic_error("");
    }
}

short SDL::TrackCandidate::trackCandidateType() const
{
    return trackCandidateType_;
}

SDL::TrackCandidate::~TrackCandidate()
{
}
