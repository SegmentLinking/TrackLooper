#ifndef TrackCandidate_h
#define TrackCandidate_h

#include <iomanip>
#include <stdexcept>

#include "Module.h"
#include "Algo.h"
#include "TrackletBase.h"
#include "Tracklet.h"
#include "Triplet.h"
#include "MathUtil.h"
#include "PrintUtil.h"

namespace SDL
{
    namespace CPU
    {
        class Module;
    }
}

namespace SDL
{
    namespace CPU
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

                enum T5Selection
                {
                    tracklet13 = 0,
                    tracklet14,
                    radiusConsistency,
                    nCutT5
                };

                enum pT3Selection
                {
                    pT3tracklet13 = 0,
                    pT3tracklet14,
                    pT3radiusConsistency,
                    nCutpT3
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
                void runTrackCandidateAlgo(TCAlgo algo, SDL::CPU::LogLevel logLevel=SDL::CPU::Log_Nothing);

                // The following algorithm does nothing and accept everything
                void runTrackCandidateAllCombAlgo();

                // The default algorithms
                void runTrackCandidateDefaultAlgo(SDL::CPU::LogLevel logLevel);

                // Connecting inner tracklet to outer triplet with share segment
                void runTrackCandidateInnerTrackletToOuterTriplet(SDL::CPU::LogLevel logLevel);

                // Connecting inner triplet to outer tracklet with share segment
                void runTrackCandidateInnerTripletToOuterTracklet(SDL::CPU::LogLevel logLevel);

                // Connecting inner triplet to outer triplet with share mini-doublet
                void runTrackCandidateT5(SDL::CPU::LogLevel logLevel);

                // Connecting pixel track to outer tracker triplet
                void runTrackCandidatepT3(SDL::CPU::LogLevel logLevel);

                bool isIdxMatched(const TrackCandidate&) const;
                bool isAnchorHitIdxMatched(const TrackCandidate&) const;

                bool matchRadiiBBBBB(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax);
                bool matchRadiiBBBBE(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S, float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax);
                bool matchRadiiBBBEE12378(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S, float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax);
                bool matchRadiiBBBEE23478(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S, float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax);
                bool matchRadiiBBBEE34578(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S, float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax);
                bool matchRadiiBBBEE(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S, float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax);
                bool matchRadiiBBEEE(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S, float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax);
                bool matchRadiiBEEEE(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S, float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax); 
                bool matchRadiiEEEEE(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S, float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax);
                bool checkIntervalOverlap(const float& firstMin, const float& firstMax, const float& secondMin, const float& secondMax);

                static float computeRadiusFromThreeAnchorHits(float x1, float y1, float x2, float y2, float x3, float y3, float& g, float& f);
                void computeErrorInRadius(std::vector<float> x1Vec, std::vector<float> y1Vec, std::vector<float> x2Vec, std::vector<float> y2Vec, std::vector<float> x3Vec, std::vector<float> y3Vec, float& minimumRadius, float& maximumRadius);

                // for pT3
                /*bounds for high Pt taken from : http://uaf-10.t2.ucsd.edu/~bsathian/SDL/T5_efficiency/efficiencies/new_efficiencies/efficiencies_20210513_T5_recovering_high_Pt_efficiencies/highE_radius_matching/highE_bounds.txt */
                bool passRadiusCriterionBBB(float& pixelRadius, float& pixelRadiusError, float& tripletRadius);
                bool passRadiusCriterionBBE(float& pixelRadius, float& pixelRadiusError, float& tripletRadius);
                bool passRadiusCriterionBEE(float& pixelRadius, float& pixelRadiusError, float& tripletRadius);
                bool passRadiusCriterionEEE(float& pixelRadius, float& pixelRadiusError, float& tripletRadius);

                // cout printing
                friend std::ostream& operator<<(std::ostream& out, const TrackCandidate& tc);
                friend std::ostream& operator<<(std::ostream& out, const TrackCandidate* tc);

        };
    }

}

#endif
