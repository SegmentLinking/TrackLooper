#ifndef Tracklet_h
#define Tracklet_h

#include <iomanip>
#include <functional>

#include "Module.cuh"
#include "Algo.h"
#include "Segment.h"
#include "PrintUtil.h"
#include "GeometryUtil.cuh"
#include "TrackletBase.h"

namespace SDL
{
    class Module;
    class Tracklet;
    class TrackletBase;
    class Segment;
}

namespace SDL
{
    class Tracklet : public TrackletBase
    {

        // Tracklet is two segments
        // So physically it will look like the following:
        //
        // Below, the pair of x's are one segment and the pair of y's are another segment
        //
        // The x's are outer segment
        // The y's are inner segment
        //
        //    --------x--------  <-
        //    ---------x-------   |
        //                        | outer segment
        //    ----------x------   |
        //    -----------x-----  <-
        //
        //
        //
        //
        //    -----------y-----  <-
        //    ----------y------   |
        //                        | inner segment
        //    ---------y-------   |
        //    --------y--------  <-
        //
        // Module naming is given above
        //

        public:
            enum TrackletSelection
            {
                deltaZ = 0,
                deltaZPointed = 1,
                deltaPhiPos = 2,
                slope=3,
                dAlphaIn=4,
                dAlphaOut=5,
                dBeta=6,
                nCut=7
            };

        private:
            // Some reco'ed quantities
            float betaIn_;
            float betaInCut_;
            float betaOut_;
            float betaOutCut_;
            float deltaBeta_;
            float deltaBetaCut_;

            bool setNm1DeltaBeta_;

        public:
            Tracklet();
            Tracklet(const Tracklet&);
            Tracklet(Segment* innerSegmentPtr, Segment* outerSegmentPtr);
            ~Tracklet();

            void addSelfPtrToSegments();

            const float& getDeltaBeta() const;
            const float& getDeltaBetaCut() const;
            const float& getBetaIn() const;
            const float& getBetaInCut() const;
            const float& getBetaOut() const;
            const float& getBetaOutCut() const;

            void setDeltaBeta(float);
            void setDeltaBetaCut(float);
            void setBetaIn(float);
            void setBetaInCut(float);
            void setBetaOut(float);
            void setBetaOutCut(float);

            // Set the N-1 setting for the deltabeta
            // This is used for studying the performance of deltabeta
            void setNm1DeltaBetaCut(bool=true);
            bool getNm1DeltaBetaCut();

            // return whether it passed the algorithm
            bool passesTrackletAlgo(TLAlgo algo) const;

            // The function to run segment algorithm on a segment candidate
            void runTrackletAlgo(TLAlgo algo, SDL::LogLevel logLevel=SDL::Log_Nothing);

            // The following algorithm does nothing and accept everything
            void runTrackletAllCombAlgo();

            // The default algorithms
            void runTrackletDefaultAlgo(SDL::LogLevel logLevel);
            void runTrackletDefaultAlgoBarrelBarrelBarrelBarrel(SDL::LogLevel logLevel);
            void runTrackletDefaultAlgoBarrelBarrelBarrelBarrel_v1(SDL::LogLevel logLevel);
            void runTrackletDefaultAlgoBarrelBarrelBarrelBarrel_v2(SDL::LogLevel logLevel);
            void runTrackletDefaultAlgoBarrelBarrelEndcapEndcap(SDL::LogLevel logLevel);
            void runTrackletDefaultAlgoEndcapEndcapEndcapEndcap(SDL::LogLevel logLevel);
            void runTrackletDefaultAlgoDeltaBetaOnlyBarrelBarrelBarrelBarrel(SDL::LogLevel logLevel);
            void runTrackletDefaultAlgoBarrelBarrel(SDL::LogLevel logLevel);
            void runTrackletDefaultAlgoBarrelEndcap(SDL::LogLevel logLevel);
            void runTrackletDefaultAlgoEndcapEndcap(SDL::LogLevel logLevel);

            // Function for delta beta
            void runDeltaBetaIterations(float& betaIn, float& betaOut, float& betaAv, float& pt_beta, float sdIn_dr, float sdOut_dr, float dr, int lIn, float pt_betaMax);

            bool hasCommonSegment(const Tracklet&) const;

            // The function to actually determine whether a pair of mini-doublets is a reco-ed segment or not
            static bool isSegmentPairATracklet(const Segment& innerSegment, const Segment& outerSegment, TLAlgo algo, SDL::LogLevel logLevel=SDL::Log_Nothing);
            static bool isSegmentPairATrackletBarrel(const Segment& innerSegment, const Segment& outerSegment, TLAlgo algo, SDL::LogLevel logLevel=SDL::Log_Nothing);
            static bool isSegmentPairATrackletEndcap(const Segment& innerSegment, const Segment& outerSegment, TLAlgo algo, SDL::LogLevel logLevel=SDL::Log_Nothing);

    };

}

#endif
