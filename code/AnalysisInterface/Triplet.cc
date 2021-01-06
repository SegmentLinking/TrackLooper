#include "Triplet.h"

SDL::Triplet::Triplet(float zOut, float rtOut, float dPhiPos, float dPhi, float betaIn, float betaOut, float betaInCut, float betaOutCut, float deltaBetaCut, SDL::Segment* innerSegment, SDL::Segment* outerSegment)
{
    zOut_ = zOut;
    rtOut_ = rtOut;
    dPhiPos_ = dPhiPos;
    dPhi_ = dPhi;
    betaIn_ = betaIn;
    betaOut_ = betaOut;
    betaInCut_ = betaInCut;
    betaOutCut_ = betaOutCut;
    deltaBetaCut_ = deltaBetaCut;
    innerSegment_ = innerSegment;
    outerSegment_ = outerSegment;
}

const float& SDL::Triplet::getDeltaBeta() const
{
    return fabs(betaIn_ - betaOut_);
}

const float& SDL::Triplet::getDeltaBetaCut() const
{
    return deltaBetaCut_;
}

const float& SDL::Triplet::getBetaIn() const
{
    return betaIn_;
}

const float& SDL::Triplet::getBetaOut() const
{
    return betaOut_;
}

const float& SDL::Triplet::getBetaInCut() const
{
    return betaInCut_;
}

const float& SDL::Triplet::getBetaOutCut() const
{
    return betaOutCut_;
}

const SDL::Segment* SDL::Triplet::innerSegmentPtr() const
{
    return innerSegment_;
}

const SDL::Segment* SDL::Triplet::outerSegmentPtr() const
{
    return outerSegment_;
}

