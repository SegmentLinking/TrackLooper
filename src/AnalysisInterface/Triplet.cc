#include "Triplet.h"

SDL::Triplet::Triplet(float zOut, float rtOut, float dPhiPos, float dPhi, float betaIn, float betaOut, float zLo, float zHi, float zLoPointed, float zHiPointed, float sdlCut, float betaInCut, float betaOutCut, float deltaBetaCut, float rtLo, float rtHi, float kZ, std::shared_ptr<SDL::Segment> innerSegment, std::shared_ptr<SDL::Segment> outerSegment) :     TrackletBase(innerSegment, outerSegment)

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

const float& SDL::Triplet::getZOut() const
{
    return zOut_;
}

const float& SDL::Triplet::getRtOut() const
{
    return rtOut_;
}

const float& SDL::Triplet::getDeltaPhiPos() const
{
    return dPhiPos_;
}

const float& SDL::Triplet::getDeltaPhi() const
{
    return dPhi_;
}


const float SDL::Triplet::getDeltaBeta() const
{
    return fabs(betaIn_ - betaOut_);
}


const float& SDL::Triplet::getBetaIn() const
{
    return betaIn_;
}

const float& SDL::Triplet::getBetaOut() const
{
    return betaOut_;
}

const float& SDL::Triplet::getZLo() const
{
    return zLo_;
}

const float& SDL::Triplet::getZHi() const
{
    return zHi_;
}

const float& SDL::Triplet::getZLoPointed() const
{
    return zLoPointed_;
}

const float& SDL::Triplet::getZHiPointed() const
{
    return zHiPointed_;
}

const float& SDL::Triplet::getSDLCut() const
{
    return sdlCut_;
}

const float& SDL::Triplet::getBetaInCut() const
{
    return betaInCut_;
}

const float& SDL::Triplet::getBetaOutCut() const
{
    return betaOutCut_;
}

const float& SDL::Triplet::getDeltaBetaCut() const
{
    return deltaBetaCut_;
}

const float& SDL::Triplet::getRtLo() const
{
    return rtLo_;
}

const float& SDL::Triplet::getRtHi() const
{
    return rtHi_;
}

const float& SDL::Triplet::getKZ() const
{
    return kZ_;
}

SDL::Triplet::~Triplet()
{
}
