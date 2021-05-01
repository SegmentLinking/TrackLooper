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

// Connecting inner triplet to outer triplet with share mini-doublet
void SDL::CPU::TrackCandidate::runTrackCandidateT5(SDL::CPU::LogLevel logLevel)
{

    setRecoVars("innerRadius", -999);
    setRecoVars("outerRadius", -999);

    passAlgo_ &= (0 << SDL::CPU::Default_TCAlgo);

    // Tracklet between Seg1 - Seg3
    SDL::CPU::Tracklet tlCand13(innerTripletPtr()->innerSegmentPtr(), outerTripletPtr()->innerSegmentPtr());

    // Run the tracklet algo
    tlCand13.runTrackletAlgo(SDL::CPU::Default_TLAlgo);

    if (not (tlCand13.passesTrackletAlgo(SDL::CPU::Default_TLAlgo)))
    {
        if (logLevel >= SDL::CPU::Log_Debug3)
        {
            SDL::CPU::cout << "Failed Cut #1 in " << __FUNCTION__ << std::endl;
        }
        passAlgo_ &= (0 << SDL::CPU::Default_TCAlgo);
        return;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << T5Selection::tracklet13);

    // Tracklet between Seg1 - Seg4
    SDL::CPU::Tracklet tlCand14(innerTripletPtr()->innerSegmentPtr(), outerTripletPtr()->outerSegmentPtr());

    // Run the tracklet algo
    tlCand14.runTrackletAlgo(SDL::CPU::Default_TLAlgo);

    if (not (tlCand14.passesTrackletAlgo(SDL::CPU::Default_TLAlgo)))
    {
        if (logLevel >= SDL::CPU::Log_Debug3)
        {
            SDL::CPU::cout << "Failed Cut #2 in " << __FUNCTION__ << std::endl;
        }
        passAlgo_ &= (0 << SDL::CPU::Default_TCAlgo);
        return;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << T5Selection::tracklet14);

    std::vector<std::vector<SDL::CPU::Hit*>> hits;

    SDL::CPU::Hit* hit1 = innerTripletPtr()->innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr();
    SDL::CPU::Hit* hit2 = innerTripletPtr()->innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr();
    SDL::CPU::Hit* hit3 = innerTripletPtr()->outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr();
    SDL::CPU::Hit* hit4 = outerTripletPtr()->innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr();
    SDL::CPU::Hit* hit5 = outerTripletPtr()->outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr();

    bool is2S1 = hit1->getModule().moduleType() == SDL::CPU::Module::TwoS and hit1->getModule().subdet() == SDL::CPU::Module::Endcap;
    bool is2S2 = hit2->getModule().moduleType() == SDL::CPU::Module::TwoS and hit2->getModule().subdet() == SDL::CPU::Module::Endcap;
    bool is2S3 = hit3->getModule().moduleType() == SDL::CPU::Module::TwoS and hit3->getModule().subdet() == SDL::CPU::Module::Endcap;
    bool is2S4 = hit4->getModule().moduleType() == SDL::CPU::Module::TwoS and hit4->getModule().subdet() == SDL::CPU::Module::Endcap;
    bool is2S5 = hit5->getModule().moduleType() == SDL::CPU::Module::TwoS and hit5->getModule().subdet() == SDL::CPU::Module::Endcap;

    std::vector<float> x1Vec = {hit1->x(), is2S1 ? hit1->getHitLowEdgePtr()->x() : hit1->x(), is2S1 ? hit1->getHitHighEdgePtr()->x() : hit1->x()};
    std::vector<float> y1Vec = {hit1->y(), is2S1 ? hit1->getHitLowEdgePtr()->y() : hit1->y(), is2S1 ? hit1->getHitHighEdgePtr()->y() : hit1->y()};
    std::vector<float> x2Vec = {hit2->x(), is2S2 ? hit2->getHitLowEdgePtr()->x() : hit2->x(), is2S2 ? hit2->getHitHighEdgePtr()->x() : hit2->x()};
    std::vector<float> y2Vec = {hit2->y(), is2S2 ? hit2->getHitLowEdgePtr()->y() : hit2->y(), is2S2 ? hit2->getHitHighEdgePtr()->y() : hit2->y()};
    std::vector<float> x3Vec = {hit3->x(), is2S3 ? hit3->getHitLowEdgePtr()->x() : hit3->x(), is2S3 ? hit3->getHitHighEdgePtr()->x() : hit3->x()};
    std::vector<float> y3Vec = {hit3->y(), is2S3 ? hit3->getHitLowEdgePtr()->y() : hit3->y(), is2S3 ? hit3->getHitHighEdgePtr()->y() : hit3->y()};
    std::vector<float> x4Vec = {hit4->x(), is2S4 ? hit4->getHitLowEdgePtr()->x() : hit4->x(), is2S4 ? hit4->getHitHighEdgePtr()->x() : hit4->x()};
    std::vector<float> y4Vec = {hit4->y(), is2S4 ? hit4->getHitLowEdgePtr()->y() : hit4->y(), is2S4 ? hit4->getHitHighEdgePtr()->y() : hit4->y()};
    std::vector<float> x5Vec = {hit5->x(), is2S5 ? hit5->getHitLowEdgePtr()->x() : hit5->x(), is2S5 ? hit5->getHitHighEdgePtr()->x() : hit5->x()};
    std::vector<float> y5Vec = {hit5->y(), is2S5 ? hit5->getHitLowEdgePtr()->y() : hit5->y(), is2S5 ? hit5->getHitHighEdgePtr()->y() : hit5->y()};

    float innerG, innerF;
    float outerG, outerF;
    float bridgeG, bridgeF;

    float innerRadius  = computeRadiusFromThreeAnchorHits(hit1->x(), hit1->y(), hit2->x(), hit2->y(), hit3->x(), hit3->y(), innerG, innerF);
    float outerRadius  = computeRadiusFromThreeAnchorHits(hit3->x(), hit3->y(), hit4->x(), hit4->y(), hit5->x(), hit5->y(), outerG, outerF);
    float bridgeRadius = computeRadiusFromThreeAnchorHits(hit2->x(), hit2->y(), hit3->x(), hit3->y(), hit4->x(), hit4->y(), bridgeG, bridgeF);

    setRecoVars("innerRadius", innerRadius);
    setRecoVars("outerRadius", outerRadius);

    float innerRadiusMin2S, innerRadiusMax2S;
    float outerRadiusMin2S, outerRadiusMax2S;
    float bridgeRadiusMin2S, bridgeRadiusMax2S;

    computeErrorInRadius(x1Vec, y1Vec, x2Vec, y2Vec, x3Vec, y3Vec, innerRadiusMin2S, innerRadiusMax2S);    
    computeErrorInRadius(x2Vec, y2Vec, x3Vec, y3Vec, x4Vec, y4Vec, bridgeRadiusMin2S, bridgeRadiusMax2S);
    computeErrorInRadius(x3Vec, y3Vec, x4Vec, y4Vec, x5Vec, y5Vec, outerRadiusMin2S, outerRadiusMax2S);

    const float kRinv1GeVf = (2.99792458e-3 * 3.8);
    const float k2Rinv1GeVf = kRinv1GeVf / 2.;

    bool pass = true;

    if (innerRadius < 0.95 / (2 * k2Rinv1GeVf))
    {
        pass = false;
    }

    // split by category
    bool isB1 = hit1->getModule().subdet() == SDL::CPU::Module::Barrel;
    bool isB2 = hit2->getModule().subdet() == SDL::CPU::Module::Barrel;
    bool isB3 = hit3->getModule().subdet() == SDL::CPU::Module::Barrel;
    bool isB4 = hit4->getModule().subdet() == SDL::CPU::Module::Barrel;
    bool isB5 = hit5->getModule().subdet() == SDL::CPU::Module::Barrel;

    int layerFirst = hit1->getModule().layer();

    float innerRadiusMin = -999.f;
    float innerRadiusMax = -999.f;
    float bridgeRadiusMin = -999.f;
    float bridgeRadiusMax = -999.f;
    float outerRadiusMin = -999.f;
    float outerRadiusMax = -999.f;

    bool tempPass;
    if (isB1 and isB2 and isB3 and isB4 and isB5)
    {
        tempPass = matchRadiiBBBBB(innerRadius, bridgeRadius, outerRadius, innerRadiusMin, innerRadiusMax, bridgeRadiusMin, bridgeRadiusMax, outerRadiusMin, outerRadiusMax);
    }
    else if (isB1 and isB2 and isB3 and isB4 and not isB5)
    {
        tempPass = matchRadiiBBBBE(innerRadius, bridgeRadius, outerRadius, innerRadiusMin2S, innerRadiusMax2S, bridgeRadiusMin2S, bridgeRadiusMax2S, outerRadiusMin2S, outerRadiusMax2S, innerRadiusMin, innerRadiusMax, bridgeRadiusMin, bridgeRadiusMax, outerRadiusMin, outerRadiusMax);
    }
    else if (isB1 and isB2 and isB3 and not isB4 and not isB5)
    {
        if (layerFirst == 1)
        {
            tempPass = matchRadiiBBBEE12378(innerRadius, bridgeRadius, outerRadius, innerRadiusMin2S, innerRadiusMax2S, bridgeRadiusMin2S, bridgeRadiusMax2S, outerRadiusMin2S, outerRadiusMax2S, innerRadiusMin, innerRadiusMax, bridgeRadiusMin, bridgeRadiusMax, outerRadiusMin, outerRadiusMax);
        }
        else if (layerFirst == 2)
        {
            tempPass = matchRadiiBBBEE23478(innerRadius, bridgeRadius, outerRadius, innerRadiusMin2S, innerRadiusMax2S, bridgeRadiusMin2S, bridgeRadiusMax2S, outerRadiusMin2S, outerRadiusMax2S, innerRadiusMin, innerRadiusMax, bridgeRadiusMin, bridgeRadiusMax, outerRadiusMin, outerRadiusMax);
        }
        else
        {
            tempPass = matchRadiiBBBEE34578(innerRadius, bridgeRadius, outerRadius, innerRadiusMin2S, innerRadiusMax2S, bridgeRadiusMin2S, bridgeRadiusMax2S, outerRadiusMin2S, outerRadiusMax2S, innerRadiusMin, innerRadiusMax, bridgeRadiusMin, bridgeRadiusMax, outerRadiusMin, outerRadiusMax);
        }
    }
    else if (isB1 and isB2 and not isB3 and not isB4 and not isB5)
    {
        tempPass = matchRadiiBBEEE(innerRadius, bridgeRadius, outerRadius, innerRadiusMin2S, innerRadiusMax2S, bridgeRadiusMin2S, bridgeRadiusMax2S, outerRadiusMin2S, outerRadiusMax2S, innerRadiusMin, innerRadiusMax, bridgeRadiusMin, bridgeRadiusMax, outerRadiusMin, outerRadiusMax);
    }
    else if (isB1 and not isB2 and not isB3 and not isB4 and not isB5)
    {
        tempPass = matchRadiiBEEEE(innerRadius, bridgeRadius, outerRadius, innerRadiusMin2S, innerRadiusMax2S, bridgeRadiusMin2S, bridgeRadiusMax2S, outerRadiusMin2S, outerRadiusMax2S, innerRadiusMin, innerRadiusMax, bridgeRadiusMin, bridgeRadiusMax, outerRadiusMin, outerRadiusMax);
    }
    else    
    {
        tempPass = matchRadiiEEEEE(innerRadius, bridgeRadius, outerRadius, innerRadiusMin2S, innerRadiusMax2S, bridgeRadiusMin2S, bridgeRadiusMax2S, outerRadiusMin2S, outerRadiusMax2S,innerRadiusMin, innerRadiusMax, bridgeRadiusMin, bridgeRadiusMax, outerRadiusMin, outerRadiusMax);
    }

    pass = pass and tempPass;

    if (not pass)
    {
        if (logLevel >= SDL::CPU::Log_Debug3)
        {
            SDL::CPU::cout << "Failed Cut #3 in " << __FUNCTION__ << std::endl;
        }
        passAlgo_ &= (0 << SDL::CPU::Default_TCAlgo);
        return;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << T5Selection::radiusConsistency);

    passAlgo_ |= (1 << SDL::CPU::Default_TCAlgo);

    return;
}

bool SDL::CPU::TrackCandidate::checkIntervalOverlap(const float& firstMin, const float& firstMax, const float& secondMin, const float& secondMax)
{
    return ((firstMin <= secondMin) & (secondMin < firstMax)) |  ((secondMin < firstMin) & (firstMin < secondMax));
}


bool SDL::CPU::TrackCandidate::matchRadiiBBBBB(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax)
{
    float innerInvRadiusErrorBound =  0.1512;
    float bridgeInvRadiusErrorBound = 0.1781;
    float outerInvRadiusErrorBound = 0.1840;

    innerRadiusMin = innerRadius/(1 + innerInvRadiusErrorBound);
    innerRadiusMax = innerInvRadiusErrorBound < 1 ? innerRadius/(1 - innerInvRadiusErrorBound) : 123456789;

    bridgeRadiusMin = bridgeRadius/(1 + bridgeInvRadiusErrorBound);
    bridgeRadiusMax = bridgeInvRadiusErrorBound < 1 ? bridgeRadius/(1 - bridgeInvRadiusErrorBound) : 123456789;

    outerRadiusMin = outerRadius/(1 + outerInvRadiusErrorBound);
    outerRadiusMax = outerInvRadiusErrorBound < 1 ? outerRadius/(1 - outerInvRadiusErrorBound) : 123456789;
    
    return checkIntervalOverlap(1.0/innerRadiusMax, 1.0/innerRadiusMin, 1.0/bridgeRadiusMax, 1.0/bridgeRadiusMin);
}

bool SDL::CPU::TrackCandidate::matchRadiiBBBBE(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S, float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax)
{

    float innerInvRadiusErrorBound =  0.1781;
    float bridgeInvRadiusErrorBound = 0.2167;
    float outerInvRadiusErrorBound = 1.1116;

    innerRadiusMin = innerRadius/(1 + innerInvRadiusErrorBound);
    innerRadiusMax = innerInvRadiusErrorBound < 1 ? innerRadius/(1 - innerInvRadiusErrorBound) : 123456789; //large number signifying infty

    bridgeRadiusMin = bridgeRadius/(1 + bridgeInvRadiusErrorBound);
    bridgeRadiusMax = bridgeInvRadiusErrorBound < 1 ? bridgeRadius/(1 - bridgeInvRadiusErrorBound) : 123456789;

    outerRadiusMin = outerRadius/(1 + outerInvRadiusErrorBound);
    outerRadiusMax = outerInvRadiusErrorBound < 1 ? outerRadius/(1 - outerInvRadiusErrorBound) : 123456789;

    return checkIntervalOverlap(1.0/innerRadiusMax, 1.0/innerRadiusMin, 1.0/bridgeRadiusMax, 1.0/bridgeRadiusMin);
}

bool SDL::CPU::TrackCandidate::matchRadiiBBBEE12378(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S, float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax)
{
    float innerInvRadiusErrorBound = 0.178;
    float bridgeInvRadiusErrorBound = 0.507;
    float outerInvRadiusErrorBound = 7.655;

    innerRadiusMin = innerRadius/(1 + innerInvRadiusErrorBound);
    innerRadiusMax = innerInvRadiusErrorBound < 1 ? innerRadius/(1 - innerInvRadiusErrorBound) : 123456789;

    bridgeRadiusMin = bridgeRadius/(1 + bridgeInvRadiusErrorBound);
    bridgeRadiusMax = bridgeInvRadiusErrorBound < 1 ? bridgeRadius/(1 - bridgeInvRadiusErrorBound) : 123456789;

    outerRadiusMin = outerRadius/(1 + outerInvRadiusErrorBound);
    outerRadiusMax = outerInvRadiusErrorBound < 1 ? outerRadius/(1 - outerInvRadiusErrorBound) : 123456789;

    return checkIntervalOverlap(1.0/innerRadiusMax, 1.0/innerRadiusMin, 1.0/fmaxf(bridgeRadiusMax, bridgeRadiusMax2S),1.0/fminf(bridgeRadiusMin, bridgeRadiusMin2S));
}

bool SDL::CPU::TrackCandidate::matchRadiiBBBEE23478(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S, float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax)
{
    float innerInvRadiusErrorBound = 0.2097;
    float bridgeInvRadiusErrorBound = 0.8557;
    float outerInvRadiusErrorBound = 24.0450;

    innerRadiusMin = innerRadius/(1 + innerInvRadiusErrorBound);
    innerRadiusMax = innerInvRadiusErrorBound < 1 ? innerRadius/(1 - innerInvRadiusErrorBound) : 123456789;

    bridgeRadiusMin = bridgeRadius/(1 + bridgeInvRadiusErrorBound);
    bridgeRadiusMax = bridgeInvRadiusErrorBound < 1 ? bridgeRadius/(1 - bridgeInvRadiusErrorBound) : 123456789;

    outerRadiusMin = outerRadius/(1 + outerInvRadiusErrorBound);
    outerRadiusMax = outerInvRadiusErrorBound < 1 ? outerRadius/(1 - outerInvRadiusErrorBound) : 123456789;

    return checkIntervalOverlap(1.0/innerRadiusMax, 1.0/innerRadiusMin, 1.0/fmaxf(bridgeRadiusMax, bridgeRadiusMax2S), 1.0/fminf(bridgeRadiusMin, bridgeRadiusMin2S));

}

bool SDL::CPU::TrackCandidate::matchRadiiBBBEE34578(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S, float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax)
{
    float innerInvRadiusErrorBound = 0.066;
    float bridgeInvRadiusErrorBound = 0.617;
    float outerInvRadiusErrorBound = 2.688;

    innerRadiusMin = innerRadius/(1 + innerInvRadiusErrorBound);
    innerRadiusMax = innerInvRadiusErrorBound < 1 ? innerRadius/(1 - innerInvRadiusErrorBound) : 123456789;

    bridgeRadiusMin = bridgeRadius/(1 + bridgeInvRadiusErrorBound);
    bridgeRadiusMax = bridgeInvRadiusErrorBound < 1 ? bridgeRadius/(1 - bridgeInvRadiusErrorBound) : 123456789;

    outerRadiusMin = outerRadius/(1 + outerInvRadiusErrorBound);
    outerRadiusMax = outerInvRadiusErrorBound < 1 ? outerRadius/(1 - outerInvRadiusErrorBound) : 123456789;

    return checkIntervalOverlap(1.0/innerRadiusMax, 1.0/innerRadiusMin, 1.0/fmaxf(bridgeRadiusMax, bridgeRadiusMax2S), 1.0/fminf(bridgeRadiusMin, bridgeRadiusMin2S));

}

bool SDL::CPU::TrackCandidate::matchRadiiBBBEE(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S, float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax)
{

    float innerInvRadiusErrorBound =  0.1840;
    float bridgeInvRadiusErrorBound = 0.5971;
    float outerInvRadiusErrorBound = 11.7102;

    innerRadiusMin = innerRadius/(1 + innerInvRadiusErrorBound);
    innerRadiusMax = innerInvRadiusErrorBound < 1 ? innerRadius/(1 - innerInvRadiusErrorBound) : 123456789;

    bridgeRadiusMin = bridgeRadius/(1 + bridgeInvRadiusErrorBound);
    bridgeRadiusMax = bridgeInvRadiusErrorBound < 1 ? bridgeRadius/(1 - bridgeInvRadiusErrorBound) : 123456789;

    outerRadiusMin = outerRadius/(1 + outerInvRadiusErrorBound);
    outerRadiusMax = outerInvRadiusErrorBound < 1 ? outerRadius/(1 - outerInvRadiusErrorBound) : 123456789;

    return checkIntervalOverlap(1.0/innerRadiusMax, 1.0/innerRadiusMin, 1.0/fmaxf(bridgeRadiusMax, bridgeRadiusMax2S), 1.0/fminf(bridgeRadiusMin, bridgeRadiusMin2S));

}

bool SDL::CPU::TrackCandidate::matchRadiiBBEEE(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S, float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax)
{

    float innerInvRadiusErrorBound =  0.6376;
    float bridgeInvRadiusErrorBound = 2.1381;
    float outerInvRadiusErrorBound = 20.4179;

    innerRadiusMin = innerRadius/(1 + innerInvRadiusErrorBound);
    innerRadiusMax = innerInvRadiusErrorBound < 1 ? innerRadius/(1 - innerInvRadiusErrorBound) : 123456789;

    bridgeRadiusMin = bridgeRadius/(1 + bridgeInvRadiusErrorBound);
    bridgeRadiusMax = bridgeInvRadiusErrorBound < 1 ? bridgeRadius/(1 - bridgeInvRadiusErrorBound) : 123456789;

    outerRadiusMin = outerRadius/(1 + outerInvRadiusErrorBound);
    outerRadiusMax = outerInvRadiusErrorBound < 1 ? outerRadius/(1 - outerInvRadiusErrorBound) : 123456789;

    return checkIntervalOverlap(1.0/innerRadiusMax, 1.0/innerRadiusMin, 1.0/fmaxf(bridgeRadiusMax, bridgeRadiusMax2S), 1.0/fminf(bridgeRadiusMin, bridgeRadiusMin2S));

}

bool SDL::CPU::TrackCandidate::matchRadiiBEEEE(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S, float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax) 
{

    float innerInvRadiusErrorBound =  1.9382;
    float bridgeInvRadiusErrorBound = 3.7280;
    float outerInvRadiusErrorBound = 5.7030;

    innerRadiusMin = innerRadius/(1 + innerInvRadiusErrorBound);
    innerRadiusMax = innerInvRadiusErrorBound < 1 ? innerRadius/(1 - innerInvRadiusErrorBound) : 123456789;

    bridgeRadiusMin = bridgeRadius/(1 + bridgeInvRadiusErrorBound);
    bridgeRadiusMax = bridgeInvRadiusErrorBound < 1 ? bridgeRadius/(1 - bridgeInvRadiusErrorBound) : 123456789;

    outerRadiusMin = outerRadius/(1 + outerInvRadiusErrorBound);
    outerRadiusMax = outerInvRadiusErrorBound < 1 ? outerRadius/(1 - outerInvRadiusErrorBound) : 123456789;

    return checkIntervalOverlap(1.0/fmaxf(innerRadiusMax, innerRadiusMax2S), 1.0/fminf(innerRadiusMin, innerRadiusMin2S), 1.0/fmaxf(bridgeRadiusMax, bridgeRadiusMax2S), 1.0/fminf(bridgeRadiusMin, bridgeRadiusMin2S));

}

bool SDL::CPU::TrackCandidate::matchRadiiEEEEE(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S, float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax)
{
    float innerInvRadiusErrorBound =  1.9382;
    float bridgeInvRadiusErrorBound = 2.2091;
    float outerInvRadiusErrorBound = 7.4084;

    innerRadiusMin = innerRadius/(1 + innerInvRadiusErrorBound);
    innerRadiusMax = innerInvRadiusErrorBound < 1 ? innerRadius/(1 - innerInvRadiusErrorBound) : 123456789;

    bridgeRadiusMin = bridgeRadius/(1 + bridgeInvRadiusErrorBound);
    bridgeRadiusMax = bridgeInvRadiusErrorBound < 1 ? bridgeRadius/(1 - bridgeInvRadiusErrorBound) : 123456789;

    outerRadiusMin = outerRadius/(1 + outerInvRadiusErrorBound);
    outerRadiusMax = outerInvRadiusErrorBound < 1 ? outerRadius/(1 - outerInvRadiusErrorBound) : 123456789;

    return checkIntervalOverlap(1.0/fmaxf(innerRadiusMax, innerRadiusMax2S), 1.0/fminf(innerRadiusMin, innerRadiusMin2S), 1.0/fmaxf(bridgeRadiusMax, bridgeRadiusMax2S), 1.0/fminf(bridgeRadiusMin, bridgeRadiusMin2S));

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

float SDL::CPU::TrackCandidate::computeRadiusFromThreeAnchorHits(float x1, float y1, float x2, float y2, float x3, float y3, float& g, float& f)
{
    float radius = 0;

    // writing manual code for computing radius, which obviously sucks
    // TODO:Use fancy inbuilt libraries like cuBLAS or cuSOLVE for this!
    //(g,f) -> center
    // first anchor hit - (x1,y1), second anchor hit - (x2,y2), third anchor hit - (x3, y3)

    if ((y1 - y3) * (x2 - x3) - (x1 - x3) * (y2 - y3) == 0)
    {
        return -1; // WTF man three collinear points!
    }

    float denom = ((y1 - y3) * (x2 - x3) - (x1 - x3) * (y2 - y3));

    g = 0.5 * ((y3 - y2) * (x1 * x1 + y1 * y1) + (y1 - y3) * (x2 * x2 + y2 * y2) + (y2 - y1) * (x3 * x3 + y3 * y3)) / denom;

    f = 0.5 * ((x2 - x3) * (x1 * x1 + y1 * y1) + (x3 - x1) * (x2 * x2 + y2 * y2) + (x1 - x2) * (x3 * x3 + y3 * y3)) / denom;

    float c = ((x2 * y3 - x3 * y2) * (x1 * x1 + y1 * y1) + (x3 * y1 - x1 * y3) * (x2 * x2 + y2 * y2) + (x1 * y2 - x2 * y1) * (x3 * x3 + y3 * y3)) / denom;

    if (g * g + f * f - c < 0)
    {
        std::cout << "FATAL! r^2 < 0!" << std::endl;
        return -1;
    }

    radius = sqrtf(g * g + f * f - c);
    return radius;
}

void SDL::CPU::TrackCandidate::computeErrorInRadius(std::vector<float> x1Vec, std::vector<float> y1Vec, std::vector<float> x2Vec, std::vector<float> y2Vec, std::vector<float> x3Vec, std::vector<float> y3Vec, float& minimumRadius, float& maximumRadius)
{
    //brute force
    float candidateRadius;
    minimumRadius = 123456789;
    maximumRadius = 0;
    float f, g; //placeholders
    for(size_t i = 0; i < 3; i++)
    {
        for(size_t j = 0; j < 3; j++)
        {
            for(size_t k = 0; k < 3; k++)
            {
               candidateRadius = computeRadiusFromThreeAnchorHits(x1Vec[i], y1Vec[i], x2Vec[j], y2Vec[j], x3Vec[k], y3Vec[k], g, f);
               maximumRadius = std::max(candidateRadius, maximumRadius);
               minimumRadius = std::min(candidateRadius, minimumRadius);
            }
        }
    }
}
