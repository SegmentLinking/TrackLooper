#ifndef TrackCandidate_h
#define TrackCandidate_h

#include <iomanip>
#include <stdexcept>

#include "Module.cuh"
#include "Algo.h"
#include "TrackletBase.h"
#include "Tracklet.h"
#include "Triplet.h"
#include "MathUtil.cuh"
#include "PrintUtil.h"

namespace SDL
{
    class Module;
}

namespace SDL
{
    class TrackCandidate
    {

        // TrackCandidate is two tracklets
        // So physically it will look like the following:
        //
        // Below, the x's are one tracklet and the y's are another tracklet
        //
        // The x's are outer tracklet
        // The y's are inner tracklet
        //
        //    --------x--------  <-
        //    ---------x-------   |
        //                        | outer tracklet
        //    ----------x------   |
        //    -----------x-----  <-
        //
        //
        //            outer     inner
        //            tracklet  tracklet
        //    -----------x-------y-----
        //    ----------x-------y------
        //
        //    ---------x-------y-------
        //    --------x-------y--------
        //
        //
        //
        //    -----------y-----  <-
        //    ----------y------   |
        //                        | inner tracklet
        //    ---------y-------   |
        //    --------y--------  <-
        //
        // Module naming is given above
        //

        private:

            // Inner Tracklet (inner means one closer to the beam position, i.e. lower "layer")
            TrackletBase* innerTrackletPtr_;

            // Outer Tracklet (outer means one further away from the beam position, i.e. upper "layer")
            TrackletBase* outerTrackletPtr_;

            // Bits to flag whether this tracklet passes some algorithm
            int passAlgo_;

        public:
            enum TrackCandidateSelection
            {
                commonSegment = 0,
                ptBetaConsistency,
                ptConsistency,
                nCut
            };

        private:
            // Bits to flag whether this tracklet passes which cut of default algorithm
            int passBitsDefaultAlgo_;

            std::map<std::string, float> recovars_;

        public:
            TrackCandidate();
            TrackCandidate(const TrackCandidate&);
            TrackCandidate(TrackletBase* innerTrackletPtr, TrackletBase* outerTrackletPtr);
            ~TrackCandidate();

            TrackletBase* innerTrackletBasePtr() const;
            TrackletBase* outerTrackletBasePtr() const;
            Tracklet* innerTrackletPtr() const;
            Tracklet* outerTrackletPtr() const;
            Triplet* innerTripletPtr() const;
            Triplet* outerTripletPtr() const;
            const int& getPassAlgo() const;
            const int& getPassBitsDefaultAlgo() const;
            const std::map<std::string, float>& getRecoVars() const;
            const float& getRecoVar(std::string) const;

            void setRecoVars(std::string, float);

            // return whether it passed the algorithm
            bool passesTrackCandidateAlgo(TCAlgo algo) const;

            // The function to run track candidate algorithm on a track candidate candidate
            void runTrackCandidateAlgo(TCAlgo algo, SDL::LogLevel logLevel=SDL::Log_Nothing);

            // The following algorithm does nothing and accept everything
            void runTrackCandidateAllCombAlgo();

            // The default algorithms
            void runTrackCandidateDefaultAlgo(SDL::LogLevel logLevel);

            bool isIdxMatched(const TrackCandidate&) const;
            bool isAnchorHitIdxMatched(const TrackCandidate&) const;

            // cout printing
            friend std::ostream& operator<<(std::ostream& out, const TrackCandidate& tc);
            friend std::ostream& operator<<(std::ostream& out, const TrackCandidate* tc);

    };

}

#endif
