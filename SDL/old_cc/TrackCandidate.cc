#include "TrackCandidate.h"

SDL::TrackCandidate::TrackCandidate()
{
}

SDL::TrackCandidate::~TrackCandidate()
{
}

SDL::TrackCandidate::TrackCandidate(const TrackCandidate& tl) :
    innerTrackletPtr_(tl.innerTrackletBasePtr()),
    outerTrackletPtr_(tl.outerTrackletBasePtr()),
    passAlgo_(tl.getPassAlgo()),
    passBitsDefaultAlgo_(tl.getPassBitsDefaultAlgo()),
    recovars_(tl.getRecoVars())
{
}

SDL::TrackCandidate::TrackCandidate(SDL::TrackletBase* innerTrackletPtr, SDL::TrackletBase* outerTrackletPtr) :
    innerTrackletPtr_(innerTrackletPtr),
    outerTrackletPtr_(outerTrackletPtr),
    passAlgo_(0),
    passBitsDefaultAlgo_(0)
{
}

SDL::TrackletBase* SDL::TrackCandidate::innerTrackletBasePtr() const
{
    return innerTrackletPtr_;
}

SDL::TrackletBase* SDL::TrackCandidate::outerTrackletBasePtr() const
{
    return outerTrackletPtr_;
}

SDL::Tracklet* SDL::TrackCandidate::innerTrackletPtr() const
{
    if (dynamic_cast<Tracklet*>(innerTrackletPtr_))
    {
        return dynamic_cast<Tracklet*>(innerTrackletPtr_);
    }
    else
    {
        SDL::cout << "TrackCandidate::innerTrackletPtr() ERROR - asked for innerTracklet when this TrackCandidate doesn't have one. (maybe it has triplets?)" << std::endl;
        throw std::logic_error("");
    }
}

SDL::Tracklet* SDL::TrackCandidate::outerTrackletPtr() const
{
    if (dynamic_cast<Tracklet*>(outerTrackletPtr_))
    {
        return dynamic_cast<Tracklet*>(outerTrackletPtr_);
    }
    else
    {
        SDL::cout << "TrackCandidate::outerTrackletPtr() ERROR - asked for outerTracklet when this TrackCandidate doesn't have one. (maybe it has triplets?)" << std::endl;
        throw std::logic_error("");
    }
}

SDL::Triplet* SDL::TrackCandidate::innerTripletPtr() const
{
    if (dynamic_cast<Triplet*>(innerTrackletPtr_))
    {
        return dynamic_cast<Triplet*>(innerTrackletPtr_);
    }
    else
    {
        SDL::cout << "TrackCandidate::innerTripletPtr() ERROR - asked for innerTriplet when this TrackCandidate doesn't have one. (maybe it has tracklets?)" << std::endl;
        throw std::logic_error("");
    }
}

SDL::Triplet* SDL::TrackCandidate::outerTripletPtr() const
{
    if (dynamic_cast<Triplet*>(outerTrackletPtr_))
    {
        return dynamic_cast<Triplet*>(outerTrackletPtr_);
    }
    else
    {
        SDL::cout << "TrackCandidate::outerTripletPtr() ERROR - asked for outerTriplet when this TrackCandidate doesn't have one. (maybe it has tracklets?)" << std::endl;
        throw std::logic_error("");
    }
}

const int& SDL::TrackCandidate::getPassAlgo() const
{
    return passAlgo_;
}

const int& SDL::TrackCandidate::getPassBitsDefaultAlgo() const
{
    return passBitsDefaultAlgo_;
}

const std::map<std::string, float>& SDL::TrackCandidate::getRecoVars() const
{
    return recovars_;
}

const float& SDL::TrackCandidate::getRecoVar(std::string key) const
{
    return recovars_.at(key);
}

void SDL::TrackCandidate::setRecoVars(std::string key, float var)
{
    recovars_[key] = var;
}

bool SDL::TrackCandidate::passesTrackCandidateAlgo(SDL::TCAlgo algo) const
{
    // Each algorithm is an enum shift it by its value and check against the flag
    return passAlgo_ & (1 << algo);
}

void SDL::TrackCandidate::runTrackCandidateAlgo(SDL::TCAlgo algo, SDL::LogLevel logLevel)
{
    if (algo == SDL::AllComb_TCAlgo)
    {
        runTrackCandidateAllCombAlgo();
    }
    else if (algo == SDL::Default_TCAlgo)
    {
        runTrackCandidateDefaultAlgo(logLevel);
    }
    else
    {
        SDL::cout << "Warning: Unrecognized track candidate algorithm!" << algo << std::endl;
        return;
    }
}

void SDL::TrackCandidate::runTrackCandidateAllCombAlgo()
{
    passAlgo_ |= (1 << SDL::AllComb_TCAlgo);
}

void SDL::TrackCandidate::runTrackCandidateDefaultAlgo(SDL::LogLevel logLevel)
{

    passAlgo_ &= (0 << SDL::Default_TCAlgo);

    // SDL::Segment* innerOuterSegment = innerTrackletPtr()->outerSegmentPtr();
    // SDL::Segment* outerInnerSegment = outerTrackletPtr()->innerSegmentPtr();

    // std::cout <<  " innerOuterSegment: " << innerOuterSegment <<  std::endl;
    // std::cout <<  " outerInnerSegment: " << outerInnerSegment <<  std::endl;
    // std::cout <<  " innerTrackletPtr()->hasCommonSegment(*(outerTrackletPtr())): " << innerTrackletPtr()->hasCommonSegment(*(outerTrackletPtr())) <<  std::endl;

    // passAlgo_ |= (1 << SDL::Default_TCAlgo);
    // return;

    const SDL::Tracklet& innerTracklet = (*innerTrackletPtr());
    const SDL::Tracklet& outerTracklet = (*outerTrackletPtr());

    if (not (innerTracklet.hasCommonSegment(outerTracklet)))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #1 in " << __FUNCTION__ << std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_TCAlgo);
        // passAlgo_ |= (1 << SDL::Default_TCAlgo);
        return;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TrackCandidateSelection::commonSegment);

    // if (not (innerTrackletPtr()->getRecoVar("pt_beta") - ))
    // {
    //     if (logLevel >= SDL::Log_Debug3)
    //     {
    //         SDL::cout << "Failed Cut #2 in " << __FUNCTION__ << std::endl;
    //     }
    //     passAlgo_ &= (0 << SDL::Default_TCAlgo);
    //     // passAlgo_ |= (1 << SDL::Default_TCAlgo);
    //     return;
    // }
    // // Flag the pass bit
    // passBitsDefaultAlgo_ |= (1 << TrackCandidateSelection::ptBetaConsistency);

    SDL::Segment* innerInnerSegment = innerTrackletPtr()->innerSegmentPtr();
    SDL::Segment* innerOuterSegment = innerTrackletPtr()->outerSegmentPtr();
    SDL::Segment* outerInnerSegment = outerTrackletPtr()->innerSegmentPtr();
    SDL::Segment* outerOuterSegment = outerTrackletPtr()->outerSegmentPtr();

    SDL::Hit& innerA = (*innerInnerSegment->innerMiniDoubletPtr()->anchorHitPtr());
    SDL::Hit& innerB = (*innerInnerSegment->outerMiniDoubletPtr()->anchorHitPtr());
    SDL::Hit& innerC = (*innerOuterSegment->innerMiniDoubletPtr()->anchorHitPtr());
    SDL::Hit& outerA = (*outerInnerSegment->outerMiniDoubletPtr()->anchorHitPtr());
    SDL::Hit& outerB = (*outerOuterSegment->innerMiniDoubletPtr()->anchorHitPtr());
    SDL::Hit& outerC = (*outerOuterSegment->outerMiniDoubletPtr()->anchorHitPtr());

    SDL::Hit innerPoint = SDL::MathUtil::getCenterFromThreePoints(innerA, innerB, innerC);
    SDL::Hit outerPoint = SDL::MathUtil::getCenterFromThreePoints(outerA, outerB, outerC);

    float innerRadius = sqrt(pow(innerA.x() - innerPoint.x(), 2) + pow(innerA.y() - innerPoint.y(), 2));
    float outerRadius = sqrt(pow(outerA.x() - outerPoint.x(), 2) + pow(outerA.y() - outerPoint.y(), 2));

    float dR = (innerRadius - outerRadius) / innerRadius;
    setRecoVars("dR", dR);
    setRecoVars("innerR", innerRadius);

    float upperthresh =  0.6 / 15000. * innerRadius + 0.2;
    float lowerthresh = -1.4 / 4000. * innerRadius - 0.1;

    if (not (dR > lowerthresh and dR < upperthresh))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #3 in " << __FUNCTION__ << std::endl;
        }
        // passAlgo_ &= (0 << SDL::Default_TCAlgo);
        passAlgo_ |= (1 << SDL::Default_TCAlgo);
        return;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TrackCandidateSelection::ptConsistency);

    passAlgo_ |= (1 << SDL::Default_TCAlgo);
}

bool SDL::TrackCandidate::isIdxMatched(const TrackCandidate& tc) const
{
    if (not innerTrackletPtr()->isIdxMatched(*(tc.innerTrackletPtr())))
        return false;
    if (not outerTrackletPtr()->isIdxMatched(*(tc.outerTrackletPtr())))
        return false;
    return true;
}

bool SDL::TrackCandidate::isAnchorHitIdxMatched(const TrackCandidate& tc) const
{
    if (not innerTrackletPtr()->isAnchorHitIdxMatched(*(tc.innerTrackletPtr())))
        return false;
    if (not outerTrackletPtr()->isAnchorHitIdxMatched(*(tc.outerTrackletPtr())))
        return false;
    return true;
}

namespace SDL
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

