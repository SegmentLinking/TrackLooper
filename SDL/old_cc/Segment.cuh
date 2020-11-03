#ifndef Segment_h
#define Segment_h

#include "MiniDoublet.cuh"
//#include "TiltedGeometry.h"
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
//            std::vector<Tracklet*> outwardTrackletPtrs;

            // Pointers of tracklets containing this segment as outer segment
//            std::vector<Tracklet*> inwardTrackletPtrs;

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

            //std::map<std::string, float> recovars_;

        public:
            CUDA_HOSTDEV Segment();
            CUDA_HOSTDEV Segment(const Segment&);
            CUDA_HOSTDEV Segment(MiniDoublet* innerMiniDoubletPtr, MiniDoublet* outerMiniDoubletPtr);
            CUDA_HOSTDEV ~Segment();

            void addSelfPtrToMiniDoublets();

            const std::vector<Tracklet*>& getListOfOutwardTrackletPtrs();
            const std::vector<Tracklet*>& getListOfInwardTrackletPtrs();

            void addOutwardTrackletPtr(Tracklet* tl);
            void addInwardTrackletPtr(Tracklet* tl);

            CUDA_HOSTDEV MiniDoublet* innerMiniDoubletPtr() const;
            CUDA_HOSTDEV MiniDoublet* outerMiniDoubletPtr() const;
            CUDA_HOSTDEV const int& getPassAlgo() const;
            CUDA_HOSTDEV const int& getPassBitsDefaultAlgo() const;
            CUDA_HOSTDEV const float& getRtOut() const;
            CUDA_HOSTDEV const float& getRtIn() const;
            CUDA_HOSTDEV const float& getRtLo() const;
            CUDA_HOSTDEV const float& getRtHi() const;
            CUDA_HOSTDEV const float& getDeltaPhi() const;
            CUDA_HOSTDEV const float& getDeltaPhiMin() const;
            CUDA_HOSTDEV const float& getDeltaPhiMax() const;
            CUDA_HOSTDEV const float& getDeltaPhiChange() const;
            CUDA_HOSTDEV const float& getDeltaPhiMinChange() const;
            CUDA_HOSTDEV const float& getDeltaPhiMaxChange() const;
            CUDA_HOSTDEV const float& getZOut() const;
            CUDA_HOSTDEV const float & getZIn() const;
            CUDA_HOSTDEV const float& getZLo() const;
            CUDA_HOSTDEV const float& getZHi() const;
            CUDA_HOSTDEV const std::map<std::string, float>& getRecoVars() const;
            CUDA_HOSTDEV const float& getRecoVar(std::string) const;

            CUDA_HOSTDEV const float& getdAlphaInnerMDSegment() const;
            CUDA_HOSTDEV const float& getdAlphaOuterMDSegment() const;
            CUDA_HOSTDEV const float& getdAlphaInnerMDOuterMD() const;

            CUDA_HOSTDEV void setRtOut(float);
            CUDA_HOSTDEV void setRtIn(float);
            CUDA_HOSTDEV void setRtLo(float);
            CUDA_HOSTDEV void setRtHi(float);
            CUDA_HOSTDEV void setDeltaPhi(float);
            CUDA_HOSTDEV void setDeltaPhiMin(float);
            CUDA_HOSTDEV void setDeltaPhiMax(float);
            CUDA_HOSTDEV void setDeltaPhiChange(float);
            CUDA_HOSTDEV void setDeltaPhiMinChange(float);
            CUDA_HOSTDEV void setDeltaPhiMaxChange(float);
            CUDA_HOSTDEV void setZIn(float);
            CUDA_HOSTDEV void setZOut(float);
            CUDA_HOSTDEV void setZLo(float);
            CUDA_HOSTDEV void setZHi(float);
            CUDA_HOSTDEV void dAlphaThreshold(const SDL::MiniDoublet &innerMiniDoublet, const SDL::MiniDoublet &outerMiniDoublet,float* dAlphaValues);

             CUDA_HOSTDEV void setdAlphaInnerMDSegment(float);
             CUDA_HOSTDEV void setdAlphaOuterMDSegment(float);
             CUDA_HOSTDEV void setdAlphaInnerMDOuterMD(float);


// return whether it passed the algorithm
             CUDA_HOSTDEV bool passesSegmentAlgo(SGAlgo algo) const;
 
             // The function to run segment algorithm on a segment candidate
             CUDA_HOSTDEV void runSegmentAlgo(SGAlgo algo, SDL::LogLevel logLevel=SDL::Log_Nothing);
 
             // The following algorithm does nothing and accept everything
             CUDA_HOSTDEV void runSegmentAllCombAlgo();
 
             // The default algorithms
             CUDA_HOSTDEV void runSegmentDefaultAlgo(SDL::LogLevel logLevel);
             CUDA_HOSTDEV void runSegmentDefaultAlgoBarrel(SDL::LogLevel logLevel);
            CUDA_HOSTDEV void runSegmentDefaultAlgoEndcap(SDL::LogLevel logLevel);
 
             bool hasCommonMiniDoublet(const Segment&) const;
             bool isIdxMatched(const Segment&) const;
             bool isAnchorHitIdxMatched(const Segment&) const;
            // cout printing
            friend std::ostream& operator<<(std::ostream& out, const Segment& md);
            friend std::ostream& operator<<(std::ostream& out, const Segment* md);

    };

}

#endif
