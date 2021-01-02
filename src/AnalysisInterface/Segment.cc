#include "Segment.h"

SDL::Segment::Segment(float zIn, float zOut, float rtIn, float rtOut, float dphi, float dphiMin, float dphiMax, float dphichange, float dphichangeMin, float dphichangeMax, float dAlphaInnerMDSegment, float dAlphaOuterMDSegment, float dAlphaInnerMDOuterMD, float zLo, float zHi, float rtLo, float rtHi, float sdCut, float dAlphaInnerMDSegmentThreshold, float dAlphaOuterMDSegmentThreshold, float dAlphaInnerMDOuterMDThreshold, std::shared_ptr<SDL::MiniDoublet> innerMDPtr, std::shared_ptr<SDL::MiniDoublet> outerMDPtr)
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
    zLo_ = zLo;
    zHi_ = zHi;
    rtLo_ = rtLo;
    rtHi_ = rtHi;
    sdCut_ = sdCut;
    dAlphaInnerMDSegmentThreshold_ = dAlphaInnerMDSegmentThreshold;
    dAlphaOuterMDSegmentThreshold_ = dAlphaOuterMDSegmentThreshold;
    dAlphaInnerMDOuterMDThreshold_ = dAlphaInnerMDOuterMDThreshold;
    innerMDPtr_ = innerMDPtr;
    outerMDPtr_ = outerMDPtr;
}


std::shared_ptr<SDL::MiniDoublet> SDL::Segment::innerMiniDoubletPtr() const
{
    return innerMDPtr_;
}

std::shared_ptr<SDL::MiniDoublet> SDL::Segment::outerMiniDoubletPtr() const
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

const float& SDL::Segment::getDeltaPhiMax() const
{
    return dphiMax_;
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


const float& SDL::Segment::getDAlphaInnerMDSegment() const
{
    return dAlphaInnerMDSegment_;
}

const float& SDL::Segment::getDAlphaOuterMDSegment() const
{
    return dAlphaOuterMDSegment_;
}

const float& SDL::Segment::getDAlphaInnerMDOuterMD() const
{
    return dAlphaInnerMDOuterMD_;
}

void SDL::Segment::setPixelVariables(const float& ptIn, const float& ptErr, const float& px, const float& py, const float&  pz, const float& etaErr)
{
    ptIn_ = ptIn;
    ptErr_ = ptErr;
    px_ = px;
    py_ = py;
    pz_ = pz;
    etaErr_ = etaErr;
}

const float& SDL::Segment::getPtIn() const
{
    return ptIn_;
}

const float& SDL::Segment::getPtErr() const
{
    return ptErr_;
}

const float& SDL::Segment::getPx() const
{
    return px_;
}

const float& SDL::Segment::getPy() const
{
    return py_;
}

const float& SDL::Segment::getPz() const
{
    return pz_;
}

const float& SDL::Segment::getEtaErr() const
{
    return etaErr_;
}

const float& SDL::Segment::getZLo() const
{
    return zLo_;
}

const float& SDL::Segment::getZHi() const
{
    return zHi_;
}

const float& SDL::Segment::getRtLo() const
{
    return rtLo_;
}

const float&  SDL::Segment::getRtHi() const
{
    return rtHi_;
}

const float& SDL::Segment::getDAlphaInnerMDSegmentThreshold() const
{
    return dAlphaInnerMDSegmentThreshold_;
}

const float& SDL::Segment::getDAlphaOuterMDSegmentThreshold() const
{
    return dAlphaOuterMDSegmentThreshold_;
}

const float& SDL::Segment::getDAlphaInnerMDOuterMDThreshold() const
{
    return dAlphaInnerMDOuterMDThreshold_;
}

const float& SDL::Segment::getSDCut() const
{
    return sdCut_;
}

SDL::Segment::~Segment()
{
}
