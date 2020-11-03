#ifndef TrackletBase_h
#define TrackletBase_h

#include <iomanip>
#include <functional>

#include "Module.cuh"
#include "Algo.h"
#include "Segment.h"
#include "PrintUtil.h"

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

        // TrackletBase is abstract class of two segments
        // It can either be a Triplet or the two segments that are linked via delta beta

        protected:

            // Inner Segment (inner means one closer to the beam position, i.e. lower "layer")
            Segment* innerSegmentPtr_;

            // Outer Segment (outer means one further away from the beam position, i.e. upper "layer")
            Segment* outerSegmentPtr_;

            // Bits to flag whether this tracklet passes some algorithm
            int passAlgo_;

            // Bits to flag whether this tracklet passes which cut of default algorithm
            int passBitsDefaultAlgo_;

            std::map<std::string, float> recovars_;

        public:
            TrackletBase();
            TrackletBase(const TrackletBase&);
            TrackletBase(Segment* innerSegmentPtr, Segment* outerSegmentPtr);
            virtual ~TrackletBase();

            Segment* innerSegmentPtr() const;
            Segment* outerSegmentPtr() const;
            const int& getPassAlgo() const;
            const int& getPassBitsDefaultAlgo() const;
            const std::map<std::string, float>& getRecoVars() const;
            const float& getRecoVar(std::string) const;

            void setRecoVars(std::string, float);

            bool isIdxMatched(const TrackletBase&) const;
            bool isAnchorHitIdxMatched(const TrackletBase&) const;

            // cout printing
            friend std::ostream& operator<<(std::ostream& out, const TrackletBase& tl);
            friend std::ostream& operator<<(std::ostream& out, const TrackletBase* tl);

    };

}

#endif
