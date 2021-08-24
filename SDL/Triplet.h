#ifndef Triplet_h
#define Triplet_h

#include <iomanip>
#include <functional>

#include "Module.h"
#include "Algo.h"
#include "Segment.h"
#include "Tracklet.h"
#include "MathUtil.h"
#include "PrintUtil.h"
#include "TrackletBase.h"
#include "TrackCandidate.h"

namespace SDL
{
    namespace CPU
    {
        class Module;
        class Triplet;
        class Tracklet;
        class Segment;
    }
}

namespace SDL
{
    namespace CPU
    {
        class Triplet : public TrackletBase
        {

            // Triplet is two segments
            // So physically it will look like the following:
            //
            // Below, the pair of x's are one segment and the pair of y's are another segment
            //
            // The x's are outer segment
            // The y's are inner segment
            // The (xy) are the shared mini-doublets
            //
            //    --------x--------  <-
            //    ---------x-------   |
            //                        | outer segment
            //    ---------(xy)----   |
            //    ---------(xy)----  <-
            //                        | inner segment
            //    ---------y-------   |
            //    --------y--------  <-
            //
            // Module naming is given above
            //

            public:
                enum TripletSelection
                {
                    commonSegment = 0,
                    deltaZ,
                    deltaZPointed,
                    tracklet,
                    nCut
                };

                Tracklet tlCand;

            public:
                Triplet();
                Triplet(const Triplet&);
                Triplet(Segment* innerSegmentPtr, Segment* outerSegmentPtr);
                ~Triplet();

                void addSelfPtrToSegments();

                // return whether it passed the algorithm
                bool passesTripletAlgo(TPAlgo algo) const;

                // The function to run segment algorithm on a segment candidate
                void runTripletAlgo(TPAlgo algo, SDL::CPU::LogLevel logLevel=SDL::CPU::Log_Nothing);

                // The following algorithm does nothing and accept everything
                void runTripletAllCombAlgo();

                // The default algorithms
                void runTripletDefaultAlgo(SDL::CPU::LogLevel logLevel);

                // Pointing constraints in r-z space
                bool passPointingConstraint(SDL::CPU::LogLevel logLevel);
                bool passPointingConstraintBarrelBarrelBarrel(SDL::CPU::LogLevel logLevel);
                bool passPointingConstraintBarrelBarrelEndcap(SDL::CPU::LogLevel logLevel);
                bool passPointingConstraintBarrelEndcapEndcap(SDL::CPU::LogLevel logLevel);
                bool passPointingConstraintEndcapEndcapEndcap(SDL::CPU::LogLevel logLevel);

                // ad hoc RZ constraint based on http://uaf-10.t2.ucsd.edu/~phchang/talks/PhilipChang20210519_LST_RZConstraint.pdf
                bool passAdHocRZConstraint(SDL::CPU::LogLevel logLevel);

        };
    }

}

#endif
