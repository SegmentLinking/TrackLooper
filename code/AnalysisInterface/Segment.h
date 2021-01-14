#ifndef ANALYSIS_INTERFACE_SEGMENT_H
#define ANALYSIS_INTERFACE_SEGMENT_H

#include <vector>
#include <map>
#include <tuple>
#include <memory>

#include "Module.h"
#include "Hit.h"
#include "MiniDoublet.h"

namespace SDL
{
    class Module;
    class Hit;
    class MiniDoublet;

    class Segment
    {
        private:
            float zIn_;
            float zOut_;
            float rtIn_;
            float rtOut_;
            float dphi_;
            float dphiMin_;
            float dphiMax_;
            float dphichange_;
            float dphichangeMin_;
            float dphichangeMax_;
            float dAlphaInnerMDSegment_;
            float dAlphaOuterMDSegment_;
            float dAlphaInnerMDOuterMD_;

            float ptIn_;
            float ptErr_;
            float px_;
            float py_;
            float pz_;
            float etaErr_;

            float zLo_;
            float zHi_;
            float rtLo_;
            float rtHi_;
            float sdCut_;
            float dAlphaInnerMDSegmentThreshold_;
            float dAlphaOuterMDSegmentThreshold_;
            float dAlphaInnerMDOuterMDThreshold_;

            std::shared_ptr<MiniDoublet> innerMDPtr_;
            std::shared_ptr<MiniDoublet> outerMDPtr_;

        public:
            Segment(float zIn, float zOut, float rtIn, float rtOut, float dphi, float dphiMin, float dphiMax, float dphichange, float dphichangeMin, float dphichangeMax, float dAlphaInnerMDSegment, float dAlphaOuterMDSegment, float dAlphaInnerMDOuterMD, float zLo, float zHi, float rtLo, float rtHi, float sdCut, float dAlphaInnerMDSegmentThreshold,float dAlphaOuterMDSegmentThreshold, float dAlphaInnerMDOuterMDThreshold, std::shared_ptr<MiniDoublet> innerMDPtr, std::shared_ptr<MiniDoublet> outerMDPtr);
            ~Segment();
            std::shared_ptr<MiniDoublet> innerMiniDoubletPtr() const;
            std::shared_ptr<MiniDoublet> outerMiniDoubletPtr() const;
            const float& getRtOut() const;
            const float& getRtIn() const;
            const float& getDeltaPhi() const;
            const float& getDeltaPhiMin() const;
            const float& getDeltaPhiMax() const;
            const float& getDeltaPhiChange() const;
            const float& getDeltaPhiMinChange() const;
            const float& getDeltaPhiMaxChange() const;
            const float& getZOut() const;
            const float & getZIn() const;

            const float& getDAlphaInnerMDSegment() const;
            const float& getDAlphaOuterMDSegment() const;
            const float& getDAlphaInnerMDOuterMD() const;

            void setPixelVariables(const float& ptIn, const float& ptErr, const float& px, const float& py, const float&  pz, const float& etaErr);

            const float& getPtIn() const;
            const float& getPtErr() const;
            const float& getPx() const;
            const float& getPy() const;
            const float& getPz() const;
            const float& getEtaErr() const;

            const float& getZLo() const;
            const float& getZHi() const;
            const float& getRtLo() const;
            const float& getRtHi() const;
            const float& getDAlphaInnerMDSegmentThreshold() const;
            const float& getDAlphaOuterMDSegmentThreshold() const;
            const float& getDAlphaInnerMDOuterMDThreshold() const;
            const float& getSDCut() const;
    };
}
#endif

