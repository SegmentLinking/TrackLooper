#ifndef Triplet_h
#define Triplet_h

#include <iomanip>
#include <functional>

#include "Module.cuh"
#include "Algo.h"
#include "Segment.h"
#include "Tracklet.h"
#include "MathUtil.cuh"
#include "PrintUtil.h"
#include "TrackletBase.h"

namespace SDL
{
    class Module;
    class Triplet;
    class Tracklet;
    class Segment;
}

namespace SDL
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
                tracklet,
                nCut
            };

            Tracklet tlCand;

        public:
            Triplet();
            Triplet(const Triplet&);
            Triplet(Segment* innerSegmentPtr, Segment* outerSegmentPtr);
            ~Triplet();

            // return whether it passed the algorithm
            bool passesTripletAlgo(TPAlgo algo) const;

            // The function to run segment algorithm on a segment candidate
            void runTripletAlgo(TPAlgo algo, SDL::LogLevel logLevel=SDL::Log_Nothing);

            // The following algorithm does nothing and accept everything
            void runTripletAllCombAlgo();

            // The default algorithms
            void runTripletDefaultAlgo(SDL::LogLevel logLevel);

    };

}

#endif
