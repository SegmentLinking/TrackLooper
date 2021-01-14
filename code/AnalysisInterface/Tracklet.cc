#include "Tracklet.h"

SDL::Tracklet::Tracklet(float zOut, float rtOut, float dPhiPos, float dPhi, float betaIn, float betaOut, float zLo, float zHi, float zLoPointed, float zHiPointed, float sdlCut, float betaInCut, float betaOutCut, float deltaBetaCut, float rtLo, float rtHi, float kZ, std::shared_ptr<SDL::Segment> innerSegment, std::shared_ptr<SDL::Segment> outerSegment) :     TrackletBase(innerSegment, outerSegment)

{
    zOut_ = zOut;
    rtOut_ = rtOut;
    dPhiPos_ = dPhiPos;
    dPhi_ = dPhi;
    betaIn_ = betaIn;
    betaOut_ = betaOut;

    zLo_ = zLo;
    zHi_ = zHi;
    zLoPointed_ = zLoPointed;
    zHiPointed_ = zHiPointed;
    sdlCut_ = sdlCut;
    betaInCut_ = betaInCut;
    betaOutCut_ = betaOutCut;
    deltaBetaCut_ = deltaBetaCut;
    rtLo_ = rtLo;
    rtHi_ = rtHi;
    kZ_ = kZ;
}

SDL::Tracklet::~Tracklet()
{
}


const float& SDL::Tracklet::getZOut() const
{
    return zOut_;
}

const float& SDL::Tracklet::getRtOut() const
{
    return rtOut_;
}

const float& SDL::Tracklet::getDeltaPhiPos() const
{
    return dPhiPos_;
}

const float& SDL::Tracklet::getDeltaPhi() const
{
    return dPhi_;
}

const float SDL::Tracklet::getDeltaBeta() const
{
    return fabs(betaIn_ - betaOut_);
}


const float& SDL::Tracklet::getBetaIn() const
{
    return betaIn_;
}

const float& SDL::Tracklet::getBetaOut() const
{
    return betaOut_;
}

const float& SDL::Tracklet::getZLo() const
{
    return zLo_;
}

const float& SDL::Tracklet::getZHi() const
{
    return zHi_;
}

const float& SDL::Tracklet::getZLoPointed() const
{
    return zLoPointed_;
}

const float& SDL::Tracklet::getZHiPointed() const
{
    return zHiPointed_;
}

const float& SDL::Tracklet::getSDLCut() const
{
    return sdlCut_;
}

const float& SDL::Tracklet::getBetaInCut() const
{
    return betaInCut_;
}

const float& SDL::Tracklet::getBetaOutCut() const
{
    return betaOutCut_;
}

const float& SDL::Tracklet::getDeltaBetaCut() const
{
    return deltaBetaCut_;
}

const float& SDL::Tracklet::getRtLo() const
{
    return rtLo_;
}

const float& SDL::Tracklet::getRtHi() const
{
    return rtHi_;
}

const float& SDL::Tracklet::getKZ() const
{
    return kZ_;
}

