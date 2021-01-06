#include "TrackCandidate.h"

SDL::CPU::TrackCandidate::TrackCandidate()
{
}

SDL::CPU::TrackCandidate::~TrackCandidate()
{
}

SDL::CPU::TrackCandidate::TrackCandidate(const TrackCandidate& tl) :
    innerTrackletPtr_(tl.innerTrackletBasePtr()),
    outerTrackletPtr_(tl.outerTrackletBasePtr()),
    passAlgo_(tl.getPassAlgo()),
    passBitsDefaultAlgo_(tl.getPassBitsDefaultAlgo()),
    recovars_(tl.getRecoVars())
{
}

SDL::CPU::TrackCandidate::TrackCandidate(SDL::CPU::TrackletBase* innerTrackletPtr, SDL::CPU::TrackletBase* outerTrackletPtr) :
    innerTrackletPtr_(innerTrackletPtr),
    outerTrackletPtr_(outerTrackletPtr),
    passAlgo_(0),
    passBitsDefaultAlgo_(0)
{
}

SDL::CPU::TrackletBase* SDL::CPU::TrackCandidate::innerTrackletBasePtr() const
{
    return innerTrackletPtr_;
}

SDL::CPU::TrackletBase* SDL::CPU::TrackCandidate::outerTrackletBasePtr() const
{
    return outerTrackletPtr_;
}

SDL::CPU::Tracklet* SDL::CPU::TrackCandidate::innerTrackletPtr() const
{
    if (dynamic_cast<Tracklet*>(innerTrackletPtr_))
    {
        return dynamic_cast<Tracklet*>(innerTrackletPtr_);
    }
    else
    {
        SDL::CPU::cout << "TrackCandidate::innerTrackletPtr() ERROR - asked for innerTracklet when this TrackCandidate doesn't have one. (maybe it has triplets?)" << std::endl;
        throw std::logic_error("");
    }
}

SDL::CPU::Tracklet* SDL::CPU::TrackCandidate::outerTrackletPtr() const
{
    if (dynamic_cast<Tracklet*>(outerTrackletPtr_))
    {
        return dynamic_cast<Tracklet*>(outerTrackletPtr_);
    }
    else
    {
        SDL::CPU::cout << "TrackCandidate::outerTrackletPtr() ERROR - asked for outerTracklet when this TrackCandidate doesn't have one. (maybe it has triplets?)" << std::endl;
        throw std::logic_error("");
    }
}

SDL::CPU::Triplet* SDL::CPU::TrackCandidate::innerTripletPtr() const
{
    if (dynamic_cast<Triplet*>(innerTrackletPtr_))
    {
        return dynamic_cast<Triplet*>(innerTrackletPtr_);
    }
    else
    {
        SDL::CPU::cout << "TrackCandidate::innerTripletPtr() ERROR - asked for innerTriplet when this TrackCandidate doesn't have one. (maybe it has tracklets?)" << std::endl;
        throw std::logic_error("");
    }
}

SDL::CPU::Triplet* SDL::CPU::TrackCandidate::outerTripletPtr() const
{
    if (dynamic_cast<Triplet*>(outerTrackletPtr_))
    {
        return dynamic_cast<Triplet*>(outerTrackletPtr_);
    }
    else
    {
        SDL::CPU::cout << "TrackCandidate::outerTripletPtr() ERROR - asked for outerTriplet when this TrackCandidate doesn't have one. (maybe it has tracklets?)" << std::endl;
        throw std::logic_error("");
    }
}

const int& SDL::CPU::TrackCandidate::getPassAlgo() const
{
    return passAlgo_;
}

const int& SDL::CPU::TrackCandidate::getPassBitsDefaultAlgo() const
{
    return passBitsDefaultAlgo_;
}

const std::map<std::string, float>& SDL::CPU::TrackCandidate::getRecoVars() const
{
    return recovars_;
}

const float& SDL::CPU::TrackCandidate::getRecoVar(std::string key) const
{
    return recovars_.at(key);
}

void SDL::CPU::TrackCandidate::setRecoVars(std::string key, float var)
{
    recovars_[key] = var;
}

bool SDL::CPU::TrackCandidate::passesTrackCandidateAlgo(SDL::CPU::TCAlgo algo) const
{
    // Each algorithm is an enum shift it by its value and check against the flag
    return passAlgo_ & (1 << algo);
}

void SDL::CPU::TrackCandidate::runTrackCandidateAlgo(SDL::CPU::TCAlgo algo, SDL::CPU::LogLevel logLevel)
{
    if (algo == SDL::CPU::AllComb_TCAlgo)
    {
        runTrackCandidateAllCombAlgo();
    }
    else if (algo == SDL::CPU::Default_TCAlgo)
    {
        runTrackCandidateDefaultAlgo(logLevel);
    }
    else
    {
        SDL::CPU::cout << "Warning: Unrecognized track candidate algorithm!" << algo << std::endl;
        return;
    }
}

void SDL::CPU::TrackCandidate::runTrackCandidateAllCombAlgo()
{
    passAlgo_ |= (1 << SDL::CPU::AllComb_TCAlgo);
}

void SDL::CPU::TrackCandidate::runTrackCandidateDefaultAlgo(SDL::CPU::LogLevel logLevel)
{

    passAlgo_ &= (0 << SDL::CPU::Default_TCAlgo);

    // SDL::CPU::Segment* innerOuterSegment = innerTrackletPtr()->outerSegmentPtr();
    // SDL::CPU::Segment* outerInnerSegment = outerTrackletPtr()->innerSegmentPtr();

    // std::cout <<  " innerOuterSegment: " << innerOuterSegment <<  std::endl;
    // std::cout <<  " outerInnerSegment: " << outerInnerSegment <<  std::endl;
    // std::cout <<  " innerTrackletPtr()->hasCommonSegment(*(outerTrackletPtr())): " << innerTrackletPtr()->hasCommonSegment(*(outerTrackletPtr())) <<  std::endl;

    // passAlgo_ |= (1 << SDL::CPU::Default_TCAlgo);
    // return;

    const SDL::CPU::Tracklet& innerTracklet = (*innerTrackletPtr());
    const SDL::CPU::Tracklet& outerTracklet = (*outerTrackletPtr());

    if (not (innerTracklet.hasCommonSegment(outerTracklet)))
    {
        if (logLevel >= SDL::CPU::Log_Debug3)
        {
            SDL::CPU::cout << "Failed Cut #1 in " << __FUNCTION__ << std::endl;
        }
        passAlgo_ &= (0 << SDL::CPU::Default_TCAlgo);
        // passAlgo_ |= (1 << SDL::CPU::Default_TCAlgo);
        return;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TrackCandidateSelection::commonSegment);

    // if (not (innerTrackletPtr()->getRecoVar("pt_beta") - ))
    // {
    //     if (logLevel >= SDL::CPU::Log_Debug3)
    //     {
    //         SDL::CPU::cout << "Failed Cut #2 in " << __FUNCTION__ << std::endl;
    //     }
    //     passAlgo_ &= (0 << SDL::CPU::Default_TCAlgo);
    //     // passAlgo_ |= (1 << SDL::CPU::Default_TCAlgo);
    //     return;
    // }
    // // Flag the pass bit
    // passBitsDefaultAlgo_ |= (1 << TrackCandidateSelection::ptBetaConsistency);

    // SDL::CPU::Segment* innerInnerSegment = innerTrackletPtr()->innerSegmentPtr();
    // SDL::CPU::Segment* innerOuterSegment = innerTrackletPtr()->outerSegmentPtr();
    // SDL::CPU::Segment* outerInnerSegment = outerTrackletPtr()->innerSegmentPtr();
    // SDL::CPU::Segment* outerOuterSegment = outerTrackletPtr()->outerSegmentPtr();

    // SDL::CPU::Hit& innerA = (*innerInnerSegment->innerMiniDoubletPtr()->anchorHitPtr());
    // SDL::CPU::Hit& innerB = (*innerInnerSegment->outerMiniDoubletPtr()->anchorHitPtr());
    // SDL::CPU::Hit& innerC = (*innerOuterSegment->innerMiniDoubletPtr()->anchorHitPtr());
    // SDL::CPU::Hit& outerA = (*outerInnerSegment->outerMiniDoubletPtr()->anchorHitPtr());
    // SDL::CPU::Hit& outerB = (*outerOuterSegment->innerMiniDoubletPtr()->anchorHitPtr());
    // SDL::CPU::Hit& outerC = (*outerOuterSegment->outerMiniDoubletPtr()->anchorHitPtr());

    // SDL::CPU::Hit innerPoint = SDL::CPU::MathUtil::getCenterFromThreePoints(innerA, innerB, innerC);
    // SDL::CPU::Hit outerPoint = SDL::CPU::MathUtil::getCenterFromThreePoints(outerA, outerB, outerC);

    // float innerRadius = sqrt(pow(innerA.x() - innerPoint.x(), 2) + pow(innerA.y() - innerPoint.y(), 2));
    // float outerRadius = sqrt(pow(outerA.x() - outerPoint.x(), 2) + pow(outerA.y() - outerPoint.y(), 2));

    // float dR = (innerRadius - outerRadius) / innerRadius;
    // setRecoVars("dR", dR);
    // setRecoVars("innerR", innerRadius);

    // float upperthresh =  0.6 / 15000. * innerRadius + 0.2;
    // float lowerthresh = -1.4 / 4000. * innerRadius - 0.1;

    // if (not (dR > lowerthresh and dR < upperthresh))
    // {
    //     if (logLevel >= SDL::CPU::Log_Debug3)
    //     {
    //         SDL::CPU::cout << "Failed Cut #3 in " << __FUNCTION__ << std::endl;
    //     }
    //     // passAlgo_ &= (0 << SDL::CPU::Default_TCAlgo);
    //     passAlgo_ |= (1 << SDL::CPU::Default_TCAlgo);
    //     return;
    // }
    // // Flag the pass bit
    // passBitsDefaultAlgo_ |= (1 << TrackCandidateSelection::ptConsistency);

    passAlgo_ |= (1 << SDL::CPU::Default_TCAlgo);
}

void SDL::CPU::TrackCandidate::runTrackCandidateInnerTrackletToOuterTriplet(SDL::CPU::LogLevel logLevel)
{

    passAlgo_ &= (0 << SDL::CPU::Default_TCAlgo);

    // SDL::CPU::Segment* innerOuterSegment = innerTrackletPtr()->outerSegmentPtr();
    // SDL::CPU::Segment* outerInnerSegment = outerTrackletPtr()->innerSegmentPtr();

    // std::cout <<  " innerOuterSegment: " << innerOuterSegment <<  std::endl;
    // std::cout <<  " outerInnerSegment: " << outerInnerSegment <<  std::endl;
    // std::cout <<  " innerTrackletPtr()->hasCommonSegment(*(outerTrackletPtr())): " << innerTrackletPtr()->hasCommonSegment(*(outerTrackletPtr())) <<  std::endl;

    // passAlgo_ |= (1 << SDL::CPU::Default_TCAlgo);
    // return;

    const SDL::CPU::Tracklet& innerTracklet = (*innerTrackletPtr());
    const SDL::CPU::Triplet& outerTriplet = (*outerTripletPtr());

    if (not (innerTracklet.hasCommonSegment(outerTriplet)))
    {
        if (logLevel >= SDL::CPU::Log_Debug3)
        {
            SDL::CPU::cout << "Failed Cut #1 in " << __FUNCTION__ << std::endl;
        }
        passAlgo_ &= (0 << SDL::CPU::Default_TCAlgo);
        // passAlgo_ |= (1 << SDL::CPU::Default_TCAlgo);
        return;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TrackCandidateSelection::commonSegment);

    // if (not (innerTrackletPtr()->getRecoVar("pt_beta") - ))
    // {
    //     if (logLevel >= SDL::CPU::Log_Debug3)
    //     {
    //         SDL::CPU::cout << "Failed Cut #2 in " << __FUNCTION__ << std::endl;
    //     }
    //     passAlgo_ &= (0 << SDL::CPU::Default_TCAlgo);
    //     // passAlgo_ |= (1 << SDL::CPU::Default_TCAlgo);
    //     return;
    // }
    // // Flag the pass bit
    // passBitsDefaultAlgo_ |= (1 << TrackCandidateSelection::ptBetaConsistency);

    // SDL::CPU::Segment* innerInnerSegment = innerTrackletPtr()->innerSegmentPtr();
    // SDL::CPU::Segment* innerOuterSegment = innerTrackletPtr()->outerSegmentPtr();
    // SDL::CPU::Segment* outerInnerSegment = outerTrackletPtr()->innerSegmentPtr();
    // SDL::CPU::Segment* outerOuterSegment = outerTrackletPtr()->outerSegmentPtr();

    // SDL::CPU::Hit& innerA = (*innerInnerSegment->innerMiniDoubletPtr()->anchorHitPtr());
    // SDL::CPU::Hit& innerB = (*innerInnerSegment->outerMiniDoubletPtr()->anchorHitPtr());
    // SDL::CPU::Hit& innerC = (*innerOuterSegment->innerMiniDoubletPtr()->anchorHitPtr());
    // SDL::CPU::Hit& outerA = (*outerInnerSegment->outerMiniDoubletPtr()->anchorHitPtr());
    // SDL::CPU::Hit& outerB = (*outerOuterSegment->innerMiniDoubletPtr()->anchorHitPtr());
    // SDL::CPU::Hit& outerC = (*outerOuterSegment->outerMiniDoubletPtr()->anchorHitPtr());

    // SDL::CPU::Hit innerPoint = SDL::CPU::MathUtil::getCenterFromThreePoints(innerA, innerB, innerC);
    // SDL::CPU::Hit outerPoint = SDL::CPU::MathUtil::getCenterFromThreePoints(outerA, outerB, outerC);

    // float innerRadius = sqrt(pow(innerA.x() - innerPoint.x(), 2) + pow(innerA.y() - innerPoint.y(), 2));
    // float outerRadius = sqrt(pow(outerA.x() - outerPoint.x(), 2) + pow(outerA.y() - outerPoint.y(), 2));

    // float dR = (innerRadius - outerRadius) / innerRadius;
    // setRecoVars("dR", dR);
    // setRecoVars("innerR", innerRadius);

    // float upperthresh =  0.6 / 15000. * innerRadius + 0.2;
    // float lowerthresh = -1.4 / 4000. * innerRadius - 0.1;

    // if (not (dR > lowerthresh and dR < upperthresh))
    // {
    //     if (logLevel >= SDL::CPU::Log_Debug3)
    //     {
    //         SDL::CPU::cout << "Failed Cut #3 in " << __FUNCTION__ << std::endl;
    //     }
    //     // passAlgo_ &= (0 << SDL::CPU::Default_TCAlgo);
    //     passAlgo_ |= (1 << SDL::CPU::Default_TCAlgo);
    //     return;
    // }
    // // Flag the pass bit
    // passBitsDefaultAlgo_ |= (1 << TrackCandidateSelection::ptConsistency);

    passAlgo_ |= (1 << SDL::CPU::Default_TCAlgo);
}

void SDL::CPU::TrackCandidate::runTrackCandidateInnerTripletToOuterTracklet(SDL::CPU::LogLevel logLevel)
{

    passAlgo_ &= (0 << SDL::CPU::Default_TCAlgo);

    const SDL::CPU::Triplet& innerTriplet = (*innerTripletPtr());
    const SDL::CPU::Tracklet& outerTracklet = (*outerTrackletPtr());

    if (not (innerTriplet.hasCommonSegment(outerTracklet)))
    {
        if (logLevel >= SDL::CPU::Log_Debug3)
        {
            SDL::CPU::cout << "Failed Cut #1 in " << __FUNCTION__ << std::endl;
        }
        passAlgo_ &= (0 << SDL::CPU::Default_TCAlgo);
        // passAlgo_ |= (1 << SDL::CPU::Default_TCAlgo);
        return;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TrackCandidateSelection::commonSegment);

    passAlgo_ |= (1 << SDL::CPU::Default_TCAlgo);
}

bool SDL::CPU::TrackCandidate::isIdxMatched(const TrackCandidate& tc) const
{
    if (not innerTrackletPtr()->isIdxMatched(*(tc.innerTrackletPtr())))
        return false;
    if (not outerTrackletPtr()->isIdxMatched(*(tc.outerTrackletPtr())))
        return false;
    return true;
}

bool SDL::CPU::TrackCandidate::isAnchorHitIdxMatched(const TrackCandidate& tc) const
{
    if (not innerTrackletPtr()->isAnchorHitIdxMatched(*(tc.innerTrackletPtr())))
        return false;
    if (not outerTrackletPtr()->isAnchorHitIdxMatched(*(tc.outerTrackletPtr())))
        return false;
    return true;
}

namespace SDL
{
    namespace CPU
    {
        std::ostream& operator<<(std::ostream& out, const TrackCandidate& tc)
        {
            out << "TrackCandidate" << std::endl;
            out << "------------------------------" << std::endl;
            {
                IndentingOStreambuf indent(out);
                out << "Inner Tracklet" << std::endl;
                out << "------------------------------" << std::endl;
                {
                    IndentingOStreambuf indent(out);
                    out << tc.innerTrackletPtr_ << std::endl;
                }
                out << "Outer Tracklet" << std::endl;
                out << "------------------------------" << std::endl;
                {
                    IndentingOStreambuf indent(out);
                    out << tc.outerTrackletPtr_;
                }
            }
            return out;
        }

        std::ostream& operator<<(std::ostream& out, const TrackCandidate* tc)
        {
            out << *tc;
            return out;
        }
    }
}

