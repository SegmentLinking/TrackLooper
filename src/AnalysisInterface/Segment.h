#ifndef ANALYSIS_INTERFACE_SEGMENT_H
#define ANALYSIS_INTERFACE_SEGMENT_H

#include <vector>
#include <map>
#include <tuple>

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
            MiniDoublet* innerMDPtr_;
            MiniDoublet* outerMDPtr_;

        public:
            Segment(float zIn, float zOut, float rtIn, float rtOut, float dphi, float dphiMin, float dphiMax, float dphichange, float dphichangeMin, float dphichangeMax, float dAlphaInnerMDSegment, float dAlphaOuterMDSegment, float dAlphaInnerMDOuterMD, MiniDoublet* innerMDPtr, MiniDoublet* outerMDPtr);
            ~Segment();
            MiniDoublet* innerMiniDoubletPtr() const;
            MiniDoublet* outerMiniDoubletPtr() const;
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
    };
}
#endif
