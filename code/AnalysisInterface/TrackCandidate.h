#ifndef ANALYSIS_INTERFACE_TRACKCANDIDATE_H
#define ANALYSIS_INTERFACE_TRACKCANDIDATE_H
#include <stdexcept>
#include "Module.h"
#include "TrackletBase.h"
#include "Tracklet.h"
#include "Triplet.h"

namespace SDL
{
    class Module;
}

namespace SDL
{
    class TrackCandidate
    {
        private:
            std::shared_ptr<TrackletBase> innerTrackletPtr_;
            std::shared_ptr<TrackletBase> outerTrackletPtr_;
            short trackCandidateType_;

        public:
            TrackCandidate(std::shared_ptr<TrackletBase>& innerTrackletPtr, std::shared_ptr<TrackletBase>& outerTrackletPtr, short trackCandidateType);
            ~TrackCandidate();

            const std::shared_ptr<TrackletBase>& innerTrackletBasePtr() const;
            const std::shared_ptr<TrackletBase>& outerTrackletBasePtr() const;
            const std::shared_ptr<Tracklet>& innerTrackletPtr() const;
            const std::shared_ptr<Tracklet>& outerTrackletPtr() const;
            const std::shared_ptr<Triplet>& innerTripletPtr() const;
            const std::shared_ptr<Triplet>& outerTripletPtr() const;
            short trackCandidateType() const;

    };
}
#endif

