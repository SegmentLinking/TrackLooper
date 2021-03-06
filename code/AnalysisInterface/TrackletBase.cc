#include "TrackletBase.h"

SDL::TrackletBase::~TrackletBase()
{
}

SDL::TrackletBase::TrackletBase()
{
}

SDL::TrackletBase::TrackletBase(std::shared_ptr<SDL::Segment>& innerSegmentPtr, std::shared_ptr<SDL::Segment>& outerSegmentPtr)
{
    innerSegmentPtr_ = innerSegmentPtr;
    outerSegmentPtr_ = outerSegmentPtr;
}

const std::shared_ptr<SDL::Segment>& SDL::TrackletBase::innerSegmentPtr() const
{
    return innerSegmentPtr_;
}

const std::shared_ptr<SDL::Segment>& SDL::TrackletBase::outerSegmentPtr() const
{
    return outerSegmentPtr_;
}

