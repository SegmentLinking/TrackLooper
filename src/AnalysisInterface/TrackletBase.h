#ifndef ANALYSIS_INTERFACE_TRACKLET_BASE_H
#define ANALYSIS_INTERFACE_TRACKLET_BASE_H

#include <iomanip>
#include <functional>
#include <memory>

#include "Module.h"
#include "Segment.h"

namespace SDL
{
    class Module;
    class TrackletBase;
    class Segment;
}

namespace SDL
{

    class TrackletBase
    {
        //abstract class of segments - can be triplet or tracklet

        protected: //so that we can public inherit these
            std::shared_ptr<Segment> innerSegmentPtr_;
            std::shared_ptr<Segment> outerSegmentPtr_;

        public:
            TrackletBase();
            TrackletBase(std::shared_ptr<Segment> innerSegmentPtr, std::shared_ptr<Segment> outerSegmentPtr);
            virtual ~TrackletBase();

            std::shared_ptr<Segment> innerSegmentPtr() const;
            std::shared_ptr<Segment> outerSegmentPtr() const;
    };
}
#endif
