#ifndef Segment_h
#define Segment_h

#include "MiniDoublet.cuh"
#include "TiltedGeometry.h"
#include "Module.cuh"
#include "Algo.h"
#include "ModuleConnectionMap.h"
#include <unordered_map>

namespace SDL
{
    class Module;
    class MiniDoublet;
}

namespace SDL
{
    class Segment
    {

        // Segment is two mini-doublets
        // So physically it will look like the following:
        //
        // Below, the pair of x's are one mini-doublet and the pair of y's are another mini-doublet
        //
        // The x's are outer mini-doublet
        // The y's are inner mini-doublet
        //
        //    --------x--------
        //    ---------x-------  <-- outer lower module
        //
        //    ---------y-------
        //    --------y--------  <-- inner lower module
        //
        // Module naming is given above
        //

        private:

            // Inner MiniDoublet (inner means one closer to the beam position, i.e. lower "layer")
            MiniDoublet* innerMiniDoubletPtr_;

            // Outer MiniDoublet (outer means one further away from the beam position, i.e. upper "layer")
            MiniDoublet* outerMiniDoubletPtr_;

            // Bits to flag whether this segment passes some algorithm
            int passAlgo_;

            // Pointers of tracklets containing this segment as inner segment
            std::vector<Tracklet*> outwardTrackletPtrs;

            // Pointers of tracklets containing this segment as outer segment
            std::vector<Tracklet*> inwardTrackletPtrs;

        public:
            enum SegmentSelection
            {
                moduleCompatible = 0,
                deltaZ = 1,
                deltaPhiPos = 2,
                slope=3,
                alphaRef=4,
                alphaOut=5,
                alphaRefOut=6,
                nCut=7
            };

        private:
            // Bits to flag whether this segment passes which cut of default algorithm
            int passBitsDefaultAlgo_;

            // Some reco'ed quantities
            float rtLo_;
            float rtHi_;
            float rtOut_; // Rt of the outer mini-doublet (anchor hit = pixel hit, if available)
            float rtIn_; // Rt of the inner mini-doublet (anchor hit = pixel hit, if available)
            float dphi_;
            float dphi_min_;
            float dphi_max_;
            float dphichange_;
            float dphichange_min_;
            float dphichange_max_;

            float zOut_;
            float zIn_;
            float zLo_; // z constraint boundary
            float zHi_; // z constraint boundary

            float dAlphaInnerMDSegment_;
            float dAlphaOuterMDSegment_;
            float dAlphaInnerMDOuterMD_;

            std::map<std::string, float> recovars_;

        public:
            Segment();
            Segment(const Segment&);
            Segment(MiniDoublet* innerMiniDoubletPtr, MiniDoublet* outerMiniDoubletPtr);
            ~Segment();

            void addSelfPtrToMiniDoublets();

            const std::vector<Tracklet*>& getListOfOutwardTrackletPtrs();
            const std::vector<Tracklet*>& getListOfInwardTrackletPtrs();

            void addOutwardTrackletPtr(Tracklet* tl);
            void addInwardTrackletPtr(Tracklet* tl);

            MiniDoublet* innerMiniDoubletPtr() const;
            MiniDoublet* outerMiniDoubletPtr() const;
            const int& getPassAlgo() const;
            const int& getPassBitsDefaultAlgo() const;
            const float& getRtOut() const;
            const float& getRtIn() const;
            const float& getRtLo() const;
            const float& getRtHi() const;
            const float& getDeltaPhi() const;
            const float& getDeltaPhiMin() const;
            const float& getDeltaPhiMax() const;
            const float& getDeltaPhiChange() const;
            const float& getDeltaPhiMinChange() const;
            const float& getDeltaPhiMaxChange() const;
            const float& getZOut() const;
            const float & getZIn() const;
            const float& getZLo() const;
            const float& getZHi() const;
            const std::map<std::string, float>& getRecoVars() const;
            const float& getRecoVar(std::string) const;

            const float& getdAlphaInnerMDSegment() const;
            const float& getdAlphaOuterMDSegment() const;
            const float& getdAlphaInnerMDOuterMD() const;

            void setRtOut(float);
            void setRtIn(float);
            void setRtLo(float);
            void setRtHi(float);
            void setDeltaPhi(float);
            void setDeltaPhiMin(float);
            void setDeltaPhiMax(float);
            void setDeltaPhiChange(float);
            void setDeltaPhiMinChange(float);
            void setDeltaPhiMaxChange(float);
            void setZIn(float);
            void setZOut(float);
            void setZLo(float);
            void setZHi(float);
            void setRecoVars(std::string, float);
            void setdAlphaInnerMDSegment(float);
            void setdAlphaOuterMDSegment(float);
            void setdAlphaInnerMDOuterMD(float);

            // return whether it passed the algorithm
            bool passesSegmentAlgo(SGAlgo algo) const;

            // The function to run segment algorithm on a segment candidate
            void runSegmentAlgo(SGAlgo algo, SDL::LogLevel logLevel=SDL::Log_Nothing);

            // The following algorithm does nothing and accept everything
            void runSegmentAllCombAlgo();

            // The default algorithms
            void runSegmentDefaultAlgo(SDL::LogLevel logLevel);
            void runSegmentDefaultAlgoBarrel(SDL::LogLevel logLevel);
            void runSegmentDefaultAlgoEndcap(SDL::LogLevel logLevel);

            bool hasCommonMiniDoublet(const Segment&) const;
            bool isIdxMatched(const Segment&) const;
            bool isAnchorHitIdxMatched(const Segment&) const;

            // The function to actually determine whether a pair of mini-doublets is a reco-ed segment or not
            static bool isMiniDoubletPairASegment(const MiniDoublet& innerMiniDoublet, const MiniDoublet& outerMiniDoublet, SGAlgo algo, SDL::LogLevel logLevel=SDL::Log_Nothing);
            static bool isMiniDoubletPairASegmentCandidateBarrel(const MiniDoublet& innerMiniDoublet, const MiniDoublet& outerMiniDoublet, SGAlgo algo, SDL::LogLevel logLevel=SDL::Log_Nothing);
            static bool isMiniDoubletPairASegmentCandidateEndcap(const MiniDoublet& innerMiniDoublet, const MiniDoublet& outerMiniDoublet, SGAlgo algo, SDL::LogLevel logLevel=SDL::Log_Nothing);
            static bool isMiniDoubletPairAngleCompatibleEndcap(const MiniDoublet& innerMiniDoublet, const MiniDoublet& outerMiniDoublet, SGAlgo algo, SDL::LogLevel logLevel=SDL::Log_Nothing);
            std::unordered_map<std::string,float> dAlphaThreshold(const SDL::MiniDoublet &innerMiniDoublet, const SDL::MiniDoublet &outerMiniDoublet);


            // cout printing
            friend std::ostream& operator<<(std::ostream& out, const Segment& md);
            friend std::ostream& operator<<(std::ostream& out, const Segment* md);

    };

}

#endif
