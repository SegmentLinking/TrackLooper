#include "Segment.h"

SDL::Segment::Segment(float zIn, float zOut, float rtIn, float rtOut, float dphi, float dphiMin, float dphiMax, float dphichange, float dphichangeMin, float dphichangeMax, float dAlphaInnerMDSegment, float dAlphaOuterMDSegment, float dAlphaInnerMDOuterMD, SDL::MiniDoublet* innerMDPtr, SDL::MiniDoublet* outerMDPtr)
{
    zOut_ = zOut;
    zIn_ = zIn;
    rtOut_ = rtOut;
    dphi_ = dphi;
    dphiMin_ = dphiMin;
    dphiMax_ = dphiMax;
    dphichange_ = dphichange;
    dphichangeMin_ = dphichangeMin;
    dphichangeMax_ = dphichangeMax;
    dAlphaInnerMDSegment_ = dAlphaInnerMDSegment;
    dAlphaOuterMDSegment_ = dAlphaOuterMDSegment;
    dAlphaInnerMDOuterMD_ = dAlphaInnerMDOuterMD;
    innerMDPtr_ = innerMDPtr;
    outerMDPtr_ = outerMDPtr;
}

SDL::MiniDoublet* SDL::Segment::innerMiniDoubletPtr() const
{
    return innerMDPtr_;
}

SDL::MiniDoublet* SDL::Segment::outerMiniDoubletPtr() const
{
    return outerMDPtr_;
}

const float& SDL::Segment::getRtOut() const
{
    return rtOut_;
}

const float& SDL::Segment::getRtIn() const
{
    return rtIn_;
}

const float& SDL::Segment::getDeltaPhi() const
{
    return dphi_;
}

const float& SDL::Segment::getDeltaPhiMin() const
{
    return dphiMin_;
}

const float& SDL::Segment::getDeltaPhiChange() const
{
    return dphichange_;
}

const float& SDL::Segment::getDeltaPhiMinChange() const
{
    return dphichangeMin_;
}

const float& SDL::Segment::getDeltaPhiMaxChange() const
{
    return dphichangeMax_;
}

const float& SDL::Segment::getZOut() const
{
    return zOut_;
}

const float& SDL::Segment::getZIn() const
{
    return zIn_;
}

