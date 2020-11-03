#include "Tracklet.h"

SDL::Tracklet::Tracklet()
{
}

SDL::Tracklet::~Tracklet()
{
}

SDL::Tracklet::Tracklet(const Tracklet& tl) :
    TrackletBase(tl),
    deltaBeta_(tl.getDeltaBeta()),
    deltaBetaCut_(tl.getDeltaBetaCut()),
    betaIn_(tl.getBetaIn()),
    betaInCut_(tl.getBetaInCut()),
    betaOut_(tl.getBetaOut()),
    betaOutCut_(tl.getBetaOutCut()),
    setNm1DeltaBeta_(false)
{
}

SDL::Tracklet::Tracklet(SDL::Segment* innerSegmentPtr, SDL::Segment* outerSegmentPtr) :
    TrackletBase(innerSegmentPtr, outerSegmentPtr),
    deltaBeta_(0),
    deltaBetaCut_(0),
    betaIn_(0),
    betaInCut_(0),
    betaOut_(0),
    betaOutCut_(0),
    setNm1DeltaBeta_(false)
{
}

void SDL::Tracklet::addSelfPtrToSegments()
{
    innerSegmentPtr_->addOutwardTrackletPtr(this);
    outerSegmentPtr_->addInwardTrackletPtr(this);
}

const float& SDL::Tracklet::getDeltaBeta() const
{
    return deltaBeta_;
}

void SDL::Tracklet::setDeltaBeta(float deltaBeta)
{
    deltaBeta_ = deltaBeta;
}

const float& SDL::Tracklet::getDeltaBetaCut() const
{
    return deltaBetaCut_;
}

void SDL::Tracklet::setDeltaBetaCut(float deltaBetaCut)
{
    deltaBetaCut_ = deltaBetaCut;
}

const float& SDL::Tracklet::getBetaIn() const
{
    return betaIn_;
}

void SDL::Tracklet::setBetaIn(float betaIn)
{
    betaIn_ = betaIn;
}

const float& SDL::Tracklet::getBetaInCut() const
{
    return betaInCut_;
}

void SDL::Tracklet::setBetaInCut(float betaInCut)
{
    betaInCut_ = betaInCut;
}

const float& SDL::Tracklet::getBetaOut() const
{
    return betaOut_;
}

void SDL::Tracklet::setBetaOut(float betaOut)
{
    betaOut_ = betaOut;
}

const float& SDL::Tracklet::getBetaOutCut() const
{
    return betaOutCut_;
}

void SDL::Tracklet::setBetaOutCut(float betaOutCut)
{
    betaOutCut_ = betaOutCut;
}

void SDL::Tracklet::setNm1DeltaBetaCut(bool setNm1)
{
    setNm1DeltaBeta_ = setNm1;
}

bool SDL::Tracklet::getNm1DeltaBetaCut()
{
    return setNm1DeltaBeta_;
}

bool SDL::Tracklet::passesTrackletAlgo(SDL::TLAlgo algo) const
{
    // Each algorithm is an enum shift it by its value and check against the flag
    return passAlgo_ & (1 << algo);
}

void SDL::Tracklet::runTrackletAlgo(SDL::TLAlgo algo, SDL::LogLevel logLevel)
{
    if (algo == SDL::AllComb_TLAlgo)
    {
        runTrackletAllCombAlgo();
    }
    else if (algo == SDL::Default_TLAlgo)
    {
        runTrackletDefaultAlgo(logLevel);
    }
    else if (algo == SDL::DefaultNm1_TLAlgo)
    {
        setNm1DeltaBetaCut(true);
        runTrackletDefaultAlgo(logLevel);
    }
    else
    {
        SDL::cout << "Warning: Unrecognized segment algorithm!" << algo << std::endl;
        return;
    }
}

void SDL::Tracklet::runTrackletAllCombAlgo()
{
    passAlgo_ |= (1 << SDL::AllComb_TLAlgo);
}

void SDL::Tracklet::runTrackletDefaultAlgo(SDL::LogLevel logLevel)
{
    // Retreived the lower module object
    const Module& innerSgInnerMDLowerHitModule = innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule();
    const Module& outerSgInnerMDLowerHitModule = outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule();
    const Module& innerSgOuterMDLowerHitModule = innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule();
    const Module& outerSgOuterMDLowerHitModule = outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule();

    setRecoVars("betaAv", -999);
    setRecoVars("betaAv_0th", -999);
    setRecoVars("betaAv_1st", -999);
    setRecoVars("betaAv_2nd", -999);
    setRecoVars("betaAv_3rd", -999);
    setRecoVars("betaAv_4th", -999);
    setRecoVars("betaIn", -999);
    setRecoVars("betaIn_cut", -999);
    setRecoVars("betaInRHmax", -999);
    setRecoVars("betaInRHmin", -999);
    setRecoVars("betaIn_0th", -999);
    setRecoVars("betaIn_1st", -999);
    setRecoVars("betaIn_1stCorr", -999);
    setRecoVars("betaIn_2nd", -999);
    setRecoVars("betaIn_2ndCorr", -999);
    setRecoVars("betaIn_3rd", -999);
    setRecoVars("betaIn_3rdCorr", -999);
    setRecoVars("betaIn_4th", -999);
    setRecoVars("betaOut", -999);
    setRecoVars("betaOutRHmax", -999);
    setRecoVars("betaOutRHmin", -999);
    setRecoVars("betaOut_0th", -999);
    setRecoVars("betaOut_1st", -999);
    setRecoVars("betaOut_1stCorr", -999);
    setRecoVars("betaOut_2nd", -999);
    setRecoVars("betaOut_2ndCorr", -999);
    setRecoVars("betaOut_3rd", -999);
    setRecoVars("betaOut_3rdCorr", -999);
    setRecoVars("betaOut_4th", -999);
    setRecoVars("betaPt_0th", -999);
    setRecoVars("betaPt_1st", -999);
    setRecoVars("betaPt_2nd", -999);
    setRecoVars("betaPt_3rd", -999);
    setRecoVars("betaPt_4th", -999);
    setRecoVars("betacormode", -999);
    setRecoVars("dBeta", -999);
    setRecoVars("dBetaCut2", -999);
    setRecoVars("dBetaLum2", -999);
    setRecoVars("dBetaMuls", -999);
    setRecoVars("dBetaRIn2", -999);
    setRecoVars("dBetaROut2", -999);
    setRecoVars("dBetaRes", -999);
    setRecoVars("dBeta_0th", -999);
    setRecoVars("dBeta_1st", -999);
    setRecoVars("dBeta_2nd", -999);
    setRecoVars("dBeta_3rd", -999);
    setRecoVars("dBeta_4th", -999);
    setRecoVars("dBeta_midPoint", -999);
    setRecoVars("deltaZLum", -999);
    setRecoVars("dr", -999);
    setRecoVars("dzDrtScale", -999);
    setRecoVars("hit1_x", -999);
    setRecoVars("hit1_y", -999);
    setRecoVars("hit2_x", -999);
    setRecoVars("hit2_y", -999);
    setRecoVars("hit3_x", -999);
    setRecoVars("hit3_y", -999);
    setRecoVars("hit4_x", -999);
    setRecoVars("hit4_y", -999);
    setRecoVars("innerSgInnerMdDetId", -999);
    setRecoVars("innerSgOuterMdDetId", -999);
    setRecoVars("k2Rinv1GeVf", -999);
    setRecoVars("kRinv1GeVf", -999);
    setRecoVars("outerSgInnerMdDetId", -999);
    setRecoVars("outerSgOuterMdDetId", -999);
    setRecoVars("pixelPSZpitch", -999);
    setRecoVars("ptCut", -999);
    setRecoVars("pt_beta", -999);
    setRecoVars("pt_betaIn", -999);
    setRecoVars("pt_betaOut", -999);
    setRecoVars("rawBetaIn", -999);
    setRecoVars("rawBetaInCorrection", -999);
    setRecoVars("rawBetaOut", -999);
    setRecoVars("rawBetaOutCorrection", -999);
    setRecoVars("rtIn", -999);
    setRecoVars("rtOut", -999);
    setRecoVars("rtOut_o_rtIn", -999);
    setRecoVars("sdIn_alpha", -999);
    setRecoVars("sdIn_d", -999);
    setRecoVars("sdOut_alphaOut", -999);
    setRecoVars("sdOut_d", -999);
    setRecoVars("sdlSlope", -999);
    setRecoVars("sinAlphaMax", -999);
    setRecoVars("strip2SZpitch", -999);
    setRecoVars("zGeom", -999);
    setRecoVars("zIn", -999);
    setRecoVars("zLo", -999);
    setRecoVars("zHi", -999);
    setRecoVars("zLoPointed", -999);
    setRecoVars("zHiPointed", -999);
    setRecoVars("zOut", -999);
    setRecoVars("kZ", -999);
    setRecoVars("rtLo_point", -999);
    setRecoVars("rtHi_point", -999);
    setRecoVars("deltaPhiPos", -999);
    setRecoVars("dPhi", -999);
    setRecoVars("sdlCut", -999);

    if (innerSgInnerMDLowerHitModule.subdet() == SDL::Module::Barrel
            and innerSgOuterMDLowerHitModule.subdet() == SDL::Module::Barrel
            and outerSgInnerMDLowerHitModule.subdet() == SDL::Module::Barrel
            and outerSgOuterMDLowerHitModule.subdet() == SDL::Module::Barrel
            )
    {
        setRecoVars("algo", 1);
        runTrackletDefaultAlgoBarrelBarrelBarrelBarrel(logLevel);
    }
    else if (innerSgInnerMDLowerHitModule.subdet() == SDL::Module::Barrel
            and innerSgOuterMDLowerHitModule.subdet() == SDL::Module::Barrel
            and outerSgInnerMDLowerHitModule.subdet() == SDL::Module::Endcap
            and outerSgOuterMDLowerHitModule.subdet() == SDL::Module::Endcap
            )
    {
        setRecoVars("algo", 2);
        runTrackletDefaultAlgoBarrelBarrelEndcapEndcap(logLevel);
    }
    else if (innerSgInnerMDLowerHitModule.subdet() == SDL::Module::Barrel
            and innerSgOuterMDLowerHitModule.subdet() == SDL::Module::Barrel
            and outerSgInnerMDLowerHitModule.subdet() == SDL::Module::Barrel
            and outerSgOuterMDLowerHitModule.subdet() == SDL::Module::Endcap
            )
    {
        setRecoVars("algo", 1);
        // runTrackletDefaultAlgoBarrelBarrelEndcapEndcap(logLevel);
        runTrackletDefaultAlgoBarrelBarrelBarrelBarrel(logLevel);
    }
    else if (innerSgInnerMDLowerHitModule.subdet() == SDL::Module::Barrel
            and innerSgOuterMDLowerHitModule.subdet() == SDL::Module::Endcap
            and outerSgInnerMDLowerHitModule.subdet() == SDL::Module::Endcap
            and outerSgOuterMDLowerHitModule.subdet() == SDL::Module::Endcap
            )
    {
        setRecoVars("algo", 2);
        runTrackletDefaultAlgoBarrelBarrelEndcapEndcap(logLevel);
    }
    else if (innerSgInnerMDLowerHitModule.subdet() == SDL::Module::Endcap
            and innerSgOuterMDLowerHitModule.subdet() == SDL::Module::Endcap
            and outerSgInnerMDLowerHitModule.subdet() == SDL::Module::Endcap
            and outerSgOuterMDLowerHitModule.subdet() == SDL::Module::Endcap
            )
    {
        setRecoVars("algo", 3);
        runTrackletDefaultAlgoEndcapEndcapEndcapEndcap(logLevel);
    }
}

void SDL::Tracklet::runTrackletDefaultAlgoBarrelBarrelBarrelBarrel(SDL::LogLevel logLevel)
{
    runTrackletDefaultAlgoBarrelBarrelBarrelBarrel_v2(logLevel);
}

void SDL::Tracklet::runTrackletDefaultAlgoBarrelBarrelBarrelBarrel_v1(SDL::LogLevel logLevel)
{

    setRecoVars("betaAv", -999);
    setRecoVars("betaAv_0th", -999);
    setRecoVars("betaAv_1st", -999);
    setRecoVars("betaAv_2nd", -999);
    setRecoVars("betaAv_3rd", -999);
    setRecoVars("betaAv_4th", -999);
    setRecoVars("betaIn", -999);
    setRecoVars("betaInRHmax", -999);
    setRecoVars("betaInRHmin", -999);
    setRecoVars("betaIn_0th", -999);
    setRecoVars("betaIn_1st", -999);
    setRecoVars("betaIn_1stCorr", -999);
    setRecoVars("betaIn_2nd", -999);
    setRecoVars("betaIn_2ndCorr", -999);
    setRecoVars("betaIn_3rd", -999);
    setRecoVars("betaIn_3rdCorr", -999);
    setRecoVars("betaIn_4th", -999);
    setRecoVars("betaOut", -999);
    setRecoVars("betaOutRHmax", -999);
    setRecoVars("betaOutRHmin", -999);
    setRecoVars("betaOut_0th", -999);
    setRecoVars("betaOut_1st", -999);
    setRecoVars("betaOut_1stCorr", -999);
    setRecoVars("betaOut_2nd", -999);
    setRecoVars("betaOut_2ndCorr", -999);
    setRecoVars("betaOut_3rd", -999);
    setRecoVars("betaOut_3rdCorr", -999);
    setRecoVars("betaOut_4th", -999);
    setRecoVars("betaPt_0th", -999);
    setRecoVars("betaPt_1st", -999);
    setRecoVars("betaPt_2nd", -999);
    setRecoVars("betaPt_3rd", -999);
    setRecoVars("betaPt_4th", -999);
    setRecoVars("betacormode", -999);
    setRecoVars("dBeta", -999);
    setRecoVars("dBetaCut2", -999);
    setRecoVars("dBetaLum2", -999);
    setRecoVars("dBetaMuls", -999);
    setRecoVars("dBetaRIn2", -999);
    setRecoVars("dBetaROut2", -999);
    setRecoVars("dBetaRes", -999);
    setRecoVars("dBeta_0th", -999);
    setRecoVars("dBeta_1st", -999);
    setRecoVars("dBeta_2nd", -999);
    setRecoVars("dBeta_3rd", -999);
    setRecoVars("dBeta_4th", -999);
    setRecoVars("dBeta_midPoint", -999);
    setRecoVars("deltaZLum", -999);
    setRecoVars("dr", -999);
    setRecoVars("dzDrtScale", -999);
    setRecoVars("hit1_x", -999);
    setRecoVars("hit1_y", -999);
    setRecoVars("hit2_x", -999);
    setRecoVars("hit2_y", -999);
    setRecoVars("hit3_x", -999);
    setRecoVars("hit3_y", -999);
    setRecoVars("hit4_x", -999);
    setRecoVars("hit4_y", -999);
    setRecoVars("innerSgInnerMdDetId", -999);
    setRecoVars("innerSgOuterMdDetId", -999);
    setRecoVars("k2Rinv1GeVf", -999);
    setRecoVars("kRinv1GeVf", -999);
    setRecoVars("outerSgInnerMdDetId", -999);
    setRecoVars("outerSgOuterMdDetId", -999);
    setRecoVars("pixelPSZpitch", -999);
    setRecoVars("ptCut", -999);
    setRecoVars("pt_beta", -999);
    setRecoVars("pt_betaIn", -999);
    setRecoVars("pt_betaOut", -999);
    setRecoVars("rawBetaIn", -999);
    setRecoVars("rawBetaInCorrection", -999);
    setRecoVars("rawBetaOut", -999);
    setRecoVars("rawBetaOutCorrection", -999);
    setRecoVars("rtIn", -999);
    setRecoVars("rtOut", -999);
    setRecoVars("rtOut_o_rtIn", -999);
    setRecoVars("sdIn_alpha", -999);
    setRecoVars("sdIn_d", -999);
    setRecoVars("sdOut_alphaOut", -999);
    setRecoVars("sdOut_d", -999);
    setRecoVars("sdlSlope", -999);
    setRecoVars("sinAlphaMax", -999);
    setRecoVars("strip2SZpitch", -999);
    setRecoVars("zGeom", -999);
    setRecoVars("zIn", -999);
    setRecoVars("zLo", -999);
    setRecoVars("zOut", -999);

    const float deltaZLum = 15.f;
    const float pixelPSZpitch = 0.15;
    const float strip2SZpitch = 5.0;
    const float zGeom =
        ((innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).moduleType() == SDL::Module::PS ? pixelPSZpitch : strip2SZpitch)
        +
        ((outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).moduleType() == SDL::Module::PS ? pixelPSZpitch : strip2SZpitch);
    const float rtIn = innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->rt();
    const float rtOut = outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->rt();
    const float zIn = innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->z();
    const float zOut = outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->z();
    const float rtOut_o_rtIn = rtOut / rtIn;
    const float kRinv1GeVf = (2.99792458e-3 * 3.8);
    const float k2Rinv1GeVf = kRinv1GeVf / 2.;
    const float ptCut = PTCUT;
    const float sinAlphaMax = 0.95;
    const float sdlSlope = std::asin(std::min(rtOut * k2Rinv1GeVf / ptCut, sinAlphaMax));
    const float dzDrtScale = tan(sdlSlope) / sdlSlope;//FIXME: need approximate value
    const float zLo = zIn + (zIn - deltaZLum) * (rtOut_o_rtIn - 1.f) * (zIn > 0.f ? 1.f : dzDrtScale) - zGeom; //slope-correction only on outer end

    // Reset passBitsDefaultAlgo_;
    passBitsDefaultAlgo_ = 0;

    // Cut #1: Z compatibility
    if (not (zOut >= zLo))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #1 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " zLo: " << zLo <<  " zOut: " << zOut <<  std::endl;
            SDL::cout <<  " zIn: " << zIn <<  std::endl;
            SDL::cout <<  " deltaZLum: " << deltaZLum <<  std::endl;
            SDL::cout <<  " rtOut_o_rtIn: " << rtOut_o_rtIn <<  std::endl;
            SDL::cout <<  " dzDrtScale: " << dzDrtScale <<  std::endl;
            SDL::cout <<  " zGeom: " << zGeom <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_TLAlgo);
        return;
    }

    const float zHi = zIn + (zIn + deltaZLum) * (rtOut_o_rtIn - 1.f) * (zIn < 0.f ? 1.f : dzDrtScale) + zGeom;

    // Cut #2: Z compatibility
    if (not (zOut >= zLo and zOut <= zHi))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #2 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " zLo: " << zLo <<  " zOut: " << zOut <<  " zHi: " << zHi <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_TLAlgo);
        return;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TrackletSelection::deltaZ);

    const float drOutIn = (rtOut - rtIn);

    const float rtInvIn = 1. / rtIn;
    // const float ptIn = sdIn.p3.Pt(); // For Pixel Seeds
    const float rIn = sqrt(zIn*zIn + rtIn*rtIn);
    const float drtSDIn = innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->rt() - innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->rt();
    const float dzSDIn = innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->z() - innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->z();
    const float dr3SDIn = innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->r3() - innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->r3();

    const float coshEta = dr3SDIn / drtSDIn; //direction estimate
    float dzErr = zGeom * zGeom * 2.f; //both sides contribute to direction uncertainty
    //FIXME (later) more realistic accounting of material effects is needed
    const float sdlThetaMulsF = 0.015f * sqrt(0.1f + 0.2 * (rtOut - rtIn) / 50.f) * sqrt(rIn / rtIn);
    const float sdlMuls = sdlThetaMulsF * 3.f / ptCut * 4.f; //will need a better guess than x4?
    dzErr += sdlMuls * sdlMuls * drOutIn * drOutIn / 3.f * coshEta * coshEta; //sloppy
    dzErr = sqrt(dzErr);
    const float dzMean = dzSDIn / drtSDIn * drOutIn;
    const float zWindow = dzErr / drtSDIn * drOutIn + zGeom; //FIXME for ptCut lower than ~0.8 need to add curv path correction
    const float zLoPointed = zIn + dzMean * (zIn > 0.f ? 1.f : dzDrtScale) - zWindow;
    const float zHiPointed = zIn + dzMean * (zIn < 0.f ? 1.f : dzDrtScale) + zWindow;

    // Cut #3: Pointed Z
    if (not (zOut >= zLoPointed and zOut <= zHiPointed))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #3 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " zLoPointed: " << zLoPointed <<  " zOut: " << zOut <<  " zHiPointed: " << zHiPointed <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_TLAlgo);
        return;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TrackletSelection::deltaZPointed);

    const float sdlPVoff = 0.1f / rtOut;
    const float sdlCut = sdlSlope + sqrt(sdlMuls * sdlMuls + sdlPVoff*sdlPVoff);
    const MiniDoublet& sdIn_mdOut = (*innerSegmentPtr()->outerMiniDoubletPtr());
    const MiniDoublet& sdOut_mdOut = (*outerSegmentPtr()->outerMiniDoubletPtr());
    const Hit& sdIn_mdOut_hit = (*innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr());
    const Hit& sdOut_mdOut_hit = (*outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr());

    // Cut #4: deltaPhiPos can be tighter
    float deltaPhiPos = sdIn_mdOut_hit.deltaPhi(sdOut_mdOut_hit);
    if (not (std::abs(deltaPhiPos) <= sdlCut) )
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #4 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " deltaPhiPos: " << deltaPhiPos <<  " sdlCut: " << sdlCut <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_TLAlgo);
        return;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TrackletSelection::deltaPhiPos);

    const Hit& sdIn_r3 = (*innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr());
    const Hit& sdOut_r3 = (*outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr());
    Hit midR3;
    midR3 += sdIn_r3;
    midR3 += sdOut_r3;
    midR3 /= 2;
    const float dPhi = midR3.deltaPhi(sdOut_r3 - sdIn_r3);

    // Cut #5: deltaPhiChange
    if (not (std::abs(dPhi) <= sdlCut))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #5 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " dPhi: " << dPhi <<  " sdlCut: " << sdlCut <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_TLAlgo);
        return;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TrackletSelection::slope);

    const Hit& sdOut_mdRef_hit = (*outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr());
    const float sdIn_alpha = innerSegmentPtr()->getDeltaPhiChange();
    const float sdOut_alpha = outerSegmentPtr()->getDeltaPhiChange();
    const float sdOut_alpha_min = outerSegmentPtr()->getDeltaPhiMinChange();
    const float sdOut_alpha_max = outerSegmentPtr()->getDeltaPhiMaxChange();
    const bool isEC_lastLayer = (outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule().subdet() == SDL::Module::Endcap) and (outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule().moduleType() == SDL::Module::TwoS);
    // const float sdOut_alphaOut = (isEC_lastLayer ? SDL::MathUtil::Phi_mpi_pi(outerSegmentPtr()->getDeltaPhiChange() - outerSegmentPtr()->getDeltaPhi()) : (sdOut_mdOut_hit.deltaPhi(sdOut_mdOut_hit  - sdOut_mdRef_hit)));
    const float sdOut_alphaOut = sdOut_mdOut_hit.deltaPhi(sdOut_mdOut_hit  - sdOut_mdRef_hit);
    const float sdOut_alphaOut_min = isEC_lastLayer ? SDL::MathUtil::Phi_mpi_pi(outerSegmentPtr()->getDeltaPhiMinChange() - outerSegmentPtr()->getDeltaPhiMin()) : sdOut_alphaOut;
    const float sdOut_alphaOut_max = isEC_lastLayer ? SDL::MathUtil::Phi_mpi_pi(outerSegmentPtr()->getDeltaPhiMaxChange() - outerSegmentPtr()->getDeltaPhiMax()) : sdOut_alphaOut;

    // // Shifting strip hits 
    // SDL::Hit sdOut_mdOut_hit_hi = SDL::GeometryUtil::stripHighEdgeHit(sdOut_mdOut_hit);
    // SDL::Hit sdOut_mdOut_hit_lo = SDL::GeometryUtil::stripLowEdgeHit(sdOut_mdOut_hit);

    const Hit& sdOut_mdOut_r3 = (*outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr());
    const Hit dr3 = sdOut_mdOut_r3 - sdIn_r3;
    float betaIn  = sdIn_alpha - sdIn_r3.deltaPhi(dr3);
    float betaOut = -sdOut_alphaOut + sdOut_mdOut_r3.deltaPhi(dr3);
    float betaInRHmin = betaIn;
    float betaInRHmax = betaIn;
    float betaOutRHmin = betaOut + sdOut_alphaOut_min - sdOut_alphaOut;
    float betaOutRHmax = betaOut + sdOut_alphaOut_max - sdOut_alphaOut;

    setRecoVars("sdIn_alpha", sdIn_alpha);
    setRecoVars("sdOut_alphaOut", sdOut_alphaOut);
    setRecoVars("rawBetaInCorrection", sdIn_r3.deltaPhi(dr3));
    setRecoVars("rawBetaOutCorrection", sdOut_mdOut_r3.deltaPhi(dr3));
    setRecoVars("rawBetaIn", betaIn);
    setRecoVars("rawBetaOut", betaOut);

    const Hit& sdIn_mdRef_hit = (*innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr());

    const float sdIn_dr = (sdIn_mdOut_hit - sdIn_mdRef_hit).rt();
    const float sdIn_d = sdIn_mdOut_hit.rt() - sdIn_mdRef_hit.rt();

    const float dr = dr3.rt();
    //beta upper cuts: 2-strip difference for direction resolution
    const float corrF = 1.f;
    bool pass_betaIn_cut = false;//pixel seeds were already selected
    const float betaIn_cut = (-sdIn_dr * corrF + dr) * k2Rinv1GeVf / ptCut + (0.02f / sdIn_d);
    // const float betaIn_cut = std::asin((-sdIn_dr * corrF + dr) * k2Rinv1GeVf / ptCut) + (0.02f / sdIn_d);
    pass_betaIn_cut = std::abs(betaInRHmin) < betaIn_cut;

    setBetaIn(betaInRHmin);
    setBetaInCut(betaIn_cut);

    // Cut #6: first beta cut
    if (not (pass_betaIn_cut))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #6 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " betaInRHmin: " << betaInRHmin <<  " betaIn_cut: " << betaIn_cut <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_TLAlgo);
        return;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TrackletSelection::dAlphaIn);

    //now the actual segment linking magic
    float betaAv = 0.5f * (betaIn + betaOut);
    //pt/k2Rinv1GeVf/2. = R
    //R*sin(betaAv) = pt/k2Rinv1GeVf/2*sin(betaAv) = dr/2 => pt = dr*k2Rinv1GeVf/sin(betaAv);
    float pt_beta = dr * k2Rinv1GeVf / sin(betaAv);

    const float pt_betaMax = 7.0f;

    int lIn = 5;
    int lOut = 5;

    int betacormode = 0;

    setRecoVars("betaIn_0th", -999);
    setRecoVars("betaOut_0th", -999);
    setRecoVars("betaAv_0th", -999);
    setRecoVars("betaPt_0th", -999);
    setRecoVars("betaIn_1stCorr", -999);
    setRecoVars("betaOut_1stCorr", -999);
    setRecoVars("dBeta_0th", -999);
    setRecoVars("betaIn_1st", -999);
    setRecoVars("betaOut_1st", -999);
    setRecoVars("betaAv_1st", -999);
    setRecoVars("betaPt_1st", -999);
    setRecoVars("betaIn_2ndCorr", -999);
    setRecoVars("betaOut_2ndCorr", -999);
    setRecoVars("dBeta_1st", -999);
    setRecoVars("betaIn_2nd", -999);
    setRecoVars("betaOut_2nd", -999);
    setRecoVars("betaAv_2nd", -999);
    setRecoVars("betaPt_2nd", -999);
    setRecoVars("betaIn_3rdCorr", -999);
    setRecoVars("betaOut_3rdCorr", -999);
    setRecoVars("dBeta_2nd", -999);
    setRecoVars("betaIn_3rd", -999);
    setRecoVars("betaOut_3rd", -999);
    setRecoVars("betaAv_3rd", -999);
    setRecoVars("betaPt_3rd", -999);
    setRecoVars("dBeta_3rd", -999);
    setRecoVars("betaIn_4th", -999);
    setRecoVars("betaOut_4th", -999);
    setRecoVars("betaAv_4th", -999);
    setRecoVars("betaPt_4th", -999);
    setRecoVars("dBeta_4th", -999);

    const float sdOut_dr = (sdOut_mdOut_hit - sdOut_mdRef_hit).rt();
    const float sdOut_d = sdOut_mdOut_hit.rt() - sdOut_mdRef_hit.rt();
    const float diffDr = std::abs(sdIn_dr - sdOut_dr) / std::abs(sdIn_dr + sdOut_dr);
    if (true //do it for all//diffDr > 0.05 //only if segment length is different significantly
            && (not isEC_lastLayer)
            && betaIn * betaOut > 0.f
            && (std::abs(pt_beta) < 4.f * pt_betaMax
                || (lIn >= 11 && std::abs(pt_beta) < 8.f * pt_betaMax)))   //and the pt_beta is well-defined; less strict for endcap-endcap
    {
        betacormode = 1;

        setRecoVars("betaIn_0th", betaIn);
        setRecoVars("betaOut_0th", betaOut);
        setRecoVars("betaAv_0th", betaAv);
        setRecoVars("betaPt_0th", pt_beta);
        setRecoVars("betaIn_1stCorr", copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaIn));
        setRecoVars("betaOut_1stCorr", copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaOut));
        setRecoVars("dBeta_0th", betaIn - betaOut);

        const float betaInUpd  = betaIn + copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaIn); //FIXME: need a faster version
        const float betaOutUpd = betaOut + copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaOut); //FIXME: need a faster version
        betaAv = 0.5f * (betaInUpd + betaOutUpd);
        pt_beta = dr * k2Rinv1GeVf / sin(betaAv); //get a better pt estimate

        setRecoVars("betaIn_1st", betaInUpd);
        setRecoVars("betaOut_1st", betaOutUpd);
        setRecoVars("betaAv_1st", betaAv);
        setRecoVars("betaPt_1st", pt_beta);
        setRecoVars("betaIn_2ndCorr", copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaIn));
        setRecoVars("betaOut_2ndCorr", copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaOut));
        setRecoVars("dBeta_1st", betaInUpd - betaOutUpd);

        betaIn  += copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaIn); //FIXME: need a faster version
        betaOut += copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaOut); //FIXME: need a faster version
        //update the av and pt
        betaAv = 0.5f * (betaIn + betaOut);
        pt_beta = dr * k2Rinv1GeVf / sin(betaAv); //get a better pt estimate

        setRecoVars("betaIn_2nd", betaIn);
        setRecoVars("betaOut_2nd", betaOut);
        setRecoVars("betaAv_2nd", betaAv);
        setRecoVars("betaPt_2nd", pt_beta);
        setRecoVars("betaIn_3rdCorr", copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaIn));
        setRecoVars("betaOut_3rdCorr", copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaOut));
        setRecoVars("dBeta_2nd", betaIn - betaOut);

        setRecoVars("betaIn_3rd", getRecoVar("rawBetaIn") + copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaIn));
        setRecoVars("betaOut_3rd", getRecoVar("rawBetaOut") + copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaOut));
        setRecoVars("betaAv_3rd", 0.5f * (getRecoVar("betaIn_3rd") + getRecoVar("betaOut_3rd")));
        setRecoVars("betaPt_3rd", dr * k2Rinv1GeVf / sin(getRecoVar("betaAv_3rd")));
        setRecoVars("dBeta_3rd", getRecoVar("betaIn_3rd") - getRecoVar("betaOut_3rd"));

        setRecoVars("betaIn_4th", getRecoVar("rawBetaIn") + copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(getRecoVar("betaPt_3rd")), sinAlphaMax)), getRecoVar("betaIn_3rd")));
        setRecoVars("betaOut_4th", getRecoVar("rawBetaOut") + copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(getRecoVar("betaPt_3rd")), sinAlphaMax)), getRecoVar("betaOut_3rd")));
        setRecoVars("betaAv_4th", 0.5f * (getRecoVar("betaIn_4th") + getRecoVar("betaOut_4th")));
        setRecoVars("betaPt_4th", dr * k2Rinv1GeVf / sin(getRecoVar("betaAv_4th")));
        setRecoVars("dBeta_4th", getRecoVar("betaIn_4th") - getRecoVar("betaOut_4th"));

    }
    else if ((lIn < 11 && std::abs(betaOut) < 0.2 * std::abs(betaIn) && std::abs(pt_beta) < 12.f * pt_betaMax) || (isEC_lastLayer))   //use betaIn sign as ref
    {
        betacormode = 2;
        const float pt_betaIn = dr * k2Rinv1GeVf / sin(betaIn);
        const float betaInUpd  = betaIn + copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(pt_betaIn), sinAlphaMax)), betaIn); //FIXME: need a faster version
        const float betaOutUpd = betaOut + copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(pt_betaIn), sinAlphaMax)), betaIn); //FIXME: need a faster version
        betaAv = (std::abs(betaOut) > 0.2f * std::abs(betaIn)) ? (0.5f * (betaInUpd + betaOutUpd)) : betaInUpd;
        pt_beta = dr * k2Rinv1GeVf / sin(betaAv); //get a better pt estimate
        betaIn  += copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaIn); //FIXME: need a faster version
        betaOut += copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaIn); //FIXME: need a faster version
        //update the av and pt
        betaAv = 0.5f * (betaIn + betaOut);
        pt_beta = dr * k2Rinv1GeVf / sin(betaAv); //get a better pt estimate

    }
    else
    {
        betacormode = 3;
    }

    //rescale the ranges proportionally
    const float betaInMMSF = (std::abs(betaInRHmin + betaInRHmax) > 0) ? (2.f * betaIn / std::abs(betaInRHmin + betaInRHmax)) : 0.; //mean value of min,max is the old betaIn
    const float betaOutMMSF = (std::abs(betaOutRHmin + betaOutRHmax) > 0) ? (2.f * betaOut / std::abs(betaOutRHmin + betaOutRHmax)) : 0.;
    betaInRHmin *= betaInMMSF;
    betaInRHmax *= betaInMMSF;
    betaOutRHmin *= betaOutMMSF;
    betaOutRHmax *= betaOutMMSF;

    const float dBetaMuls = sdlThetaMulsF * 4.f / std::min(std::abs(pt_beta), pt_betaMax); //need to confirm the range-out value of 7 GeV

    //regularize to alpha of pt_betaMax .. review may want to add resolution
    const float sdIn_rt = sdIn_mdRef_hit.rt();
    const float sdOut_rt = sdOut_mdRef_hit.rt();
    const float sdIn_z = sdIn_mdRef_hit.z();
    const float sdOut_z = sdOut_mdRef_hit.z();
    const float alphaInAbsReg = std::max(std::abs(sdIn_alpha), std::asin(std::min(sdIn_rt * k2Rinv1GeVf / 3.0f, sinAlphaMax)));
    const float alphaOutAbsReg = std::max(std::abs(sdOut_alpha), std::asin(std::min(sdOut_rt * k2Rinv1GeVf / 3.0f, sinAlphaMax)));
    const float dBetaInLum = lIn < 11 ? 0.0f : std::abs(alphaInAbsReg*deltaZLum / sdIn_z);
    const float dBetaOutLum = (lOut < 11) && (not isEC_lastLayer) ? 0.0f : std::abs(alphaOutAbsReg*deltaZLum / sdOut_z);
    const float dBetaLum2 = (dBetaInLum + dBetaOutLum) * (dBetaInLum + dBetaOutLum);

    const float sinDPhi = std::sin(dPhi);
    // const float dBetaRIn2 = std::pow((sdIn.mdRef.rtRHout - sdIn.mdRef.rtRHin) * sinDPhi / dr, 2); //TODO-RH: Ask Slava about this rtRHout? rtRHin?
    // const float dBetaROut2 = std::pow((sdOut.mdOut.rtRHout - sdOut.mdOut.rtRHin) * sinDPhi / dr, 2); //TODO-RH
    const float dBetaRIn2 = 0; // TODO-RH
    const float dBetaROut2 = 0; // TODO-RH

    const float betaOut_cut = std::asin(std::min(dr*k2Rinv1GeVf / ptCut, sinAlphaMax)) //FIXME: need faster version
        + (0.02f / sdOut_d) + sqrt(dBetaLum2 + dBetaMuls*dBetaMuls);
    // const float betaOut_cut = std::min(0.01, std::asin(std::min(dr*k2Rinv1GeVf / ptCut, sinAlphaMax)) + (0.02f / sdOut_d) + sqrt(dBetaLum2 + dBetaMuls*dBetaMuls));

    setBetaOut(betaOut);
    setBetaOutCut(betaOut_cut);

    // Cut #7: The real beta cut
    if (not (std::abs(betaOut) < betaOut_cut))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #7 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " betaOut: " << betaOut <<  " betaOut_cut: " << betaOut_cut <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_TLAlgo);
        return;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TrackletSelection::dAlphaOut);

    float pt_betaIn = dr * k2Rinv1GeVf / sin(betaIn);
    const float pt_betaOut = dr * k2Rinv1GeVf / sin(betaOut);
    const float dBetaRes = 0.02f / std::min(sdOut_d, sdIn_d);
    const float dBetaCut2 = (dBetaRes*dBetaRes * 2.0f + dBetaMuls * dBetaMuls + dBetaLum2 + dBetaRIn2 + dBetaROut2
            + 0.25 * std::pow(std::abs(betaInRHmin - betaInRHmax) + std::abs(betaOutRHmin - betaOutRHmax), 2));
    float dBeta = betaIn - betaOut;
    // const float dZeta = sdIn.zeta - sdOut.zeta;

    const float innerSgInnerMdDetId = (innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).detId();
    const float outerSgInnerMdDetId = (outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).detId();
    const float innerSgOuterMdDetId = (innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).detId();
    const float outerSgOuterMdDetId = (outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).detId();

    setRecoVars("hit1_x", innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->x());
    setRecoVars("hit1_y", innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->y());
    setRecoVars("hit2_x", innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->x());
    setRecoVars("hit2_y", innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->y());
    setRecoVars("hit3_x", outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->x());
    setRecoVars("hit3_y", outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->y());
    setRecoVars("hit4_x", outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->x());
    setRecoVars("hit4_y", outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->y());


    std::function<float()> dBeta_midPoint = [&]()
        {

            float hit1_x = innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->x();
            float hit1_y = innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->y();
            float hit2_x = innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->x();
            float hit2_y = innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->y();
            float hit3_x = outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->x();
            float hit3_y = outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->y();
            float hit4_x = outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->x();
            float hit4_y = outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->y();

            float innerSgMidX = (hit1_x + hit2_x) / 2.;
            float innerSgMidY = (hit1_y + hit2_y) / 2.;
            float outerSgMidX = (hit3_x + hit4_x) / 2.;
            float outerSgMidY = (hit3_y + hit4_y) / 2.;

            float vecA_x = hit2_x - innerSgMidX;
            float vecA_y = hit2_y - innerSgMidY;
            float vecB_x = outerSgMidX - innerSgMidX;
            float vecB_y = outerSgMidY - innerSgMidY;
            float vecC_x = hit4_x - outerSgMidX;
            float vecC_y = hit4_y - outerSgMidY;
            float vecA_mag = sqrt(vecA_x * vecA_x + vecA_y * vecA_y);
            float vecB_mag = sqrt(vecB_x * vecB_x + vecB_y * vecB_y);
            float vecC_mag = sqrt(vecC_x * vecC_x + vecC_y * vecC_y);

            float vecA_dot_vecB = vecA_x * vecB_x + vecA_y * vecB_y;
            float vecB_dot_vecC = vecB_x * vecC_x + vecB_y * vecC_y;

            float angle_AB = std::acos(vecA_dot_vecB / vecA_mag / vecB_mag);
            float angle_BC = std::acos(vecB_dot_vecC / vecB_mag / vecC_mag);

            return angle_AB - angle_BC;

        };

    setRecoVars("sinAlphaMax", sinAlphaMax);
    setRecoVars("betaIn", betaIn);
    setRecoVars("betaInRHmax", betaInRHmax);
    setRecoVars("betaInRHmin", betaInRHmin);
    setRecoVars("betaOut", betaOut);
    setRecoVars("betaOutRHmax", betaOutRHmax);
    setRecoVars("betaOutRHmin", betaOutRHmin);
    setRecoVars("dBeta", dBeta);
    setRecoVars("dBetaCut2", dBetaCut2);
    setRecoVars("dBetaLum2", dBetaLum2);
    setRecoVars("dBetaMuls", dBetaMuls);
    setRecoVars("dBetaRIn2", dBetaRIn2);
    setRecoVars("dBetaROut2", dBetaROut2);
    setRecoVars("dBetaRes", dBetaRes);
    setRecoVars("deltaZLum", deltaZLum);
    setRecoVars("dr", dr);
    setRecoVars("dzDrtScale", dzDrtScale);
    setRecoVars("innerSgInnerMdDetId", innerSgInnerMdDetId);
    setRecoVars("innerSgOuterMdDetId", innerSgOuterMdDetId);
    setRecoVars("k2Rinv1GeVf", k2Rinv1GeVf);
    setRecoVars("kRinv1GeVf", kRinv1GeVf);
    setRecoVars("outerSgInnerMdDetId", outerSgInnerMdDetId);
    setRecoVars("outerSgOuterMdDetId", outerSgOuterMdDetId);
    setRecoVars("pixelPSZpitch", pixelPSZpitch);
    setRecoVars("ptCut", ptCut);
    setRecoVars("pt_betaIn", pt_betaIn);
    setRecoVars("pt_betaOut", pt_betaOut);
    setRecoVars("rtIn", rtIn);
    setRecoVars("rtOut", rtOut);
    setRecoVars("rtOut_o_rtIn", rtOut_o_rtIn);
    setRecoVars("sdIn_d", sdIn_d);
    setRecoVars("sdOut_d", sdOut_d);
    setRecoVars("sdlSlope", sdlSlope);
    setRecoVars("strip2SZpitch", strip2SZpitch);
    setRecoVars("zGeom", zGeom);
    setRecoVars("zIn", zIn);
    setRecoVars("zLo", zLo);
    setRecoVars("zOut", zOut);
    setRecoVars("betacormode", betacormode);
    setRecoVars("pt_beta", pt_beta);
    setRecoVars("betaAv", betaAv);
    setRecoVars("dBeta_midPoint", dBeta_midPoint());

    // dBeta = getRecoVar("dBeta_4th");

    setDeltaBeta(dBeta);
    setDeltaBetaCut(std::sqrt(dBetaCut2));

    if (not (dBeta * dBeta <= dBetaCut2))
    // if (not (dBeta * dBeta <= 0.004*0.004))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #8 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " dBeta*dBeta: " << dBeta*dBeta <<  " dBetaCut2: " << dBetaCut2 <<  std::endl;
            SDL::cout <<  " dBetaRes: " << dBetaRes <<  " dBetaMuls: " << dBetaMuls <<  " dBetaLum2: " << dBetaLum2 <<  std::endl;
            SDL::cout <<  " dBetaRIn2: " << dBetaRIn2 <<  " dBetaROut2: " << dBetaROut2 <<  std::endl;
            SDL::cout <<  " betaInRHmin: " << betaInRHmin <<  " betaInRHmax: " << betaInRHmax <<  std::endl;
            SDL::cout <<  " betaOutRHmin: " << betaOutRHmin <<  " betaOutRHmax: " << betaOutRHmax <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_TLAlgo);
        // passAlgo_ |= (1 << SDL::Default_TLAlgo);
        return;
    }
    else if (logLevel >= SDL::Log_Debug3)
    {
        SDL::cout << "Passed Cut #8 in " << __FUNCTION__ << std::endl;
        SDL::cout <<  " dBeta*dBeta: " << dBeta*dBeta <<  " dBetaCut2: " << dBetaCut2 <<  std::endl;
        SDL::cout <<  " dBetaRes: " << dBetaRes <<  " dBetaMuls: " << dBetaMuls <<  " dBetaLum2: " << dBetaLum2 <<  std::endl;
        SDL::cout <<  " dBetaRIn2: " << dBetaRIn2 <<  " dBetaROut2: " << dBetaROut2 <<  std::endl;
        SDL::cout <<  " betaInRHmin: " << betaInRHmin <<  " betaInRHmax: " << betaInRHmax <<  std::endl;
        SDL::cout <<  " betaOutRHmin: " << betaOutRHmin <<  " betaOutRHmax: " << betaOutRHmax <<  std::endl;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TrackletSelection::dBeta);

    passAlgo_ |= (1 << SDL::Default_TLAlgo);
    return;
}

void SDL::Tracklet::runTrackletDefaultAlgoBarrelBarrelBarrelBarrel_v2(SDL::LogLevel logLevel)
{

    const float deltaZLum = 15.f;
    const float kRinv1GeVf = (2.99792458e-3 * 3.8);
    const float k2Rinv1GeVf = kRinv1GeVf / 2.;
    const float ptCut = PTCUT;
    const float sinAlphaMax = 0.95; // 54.43 degrees
    const float pixelPSZpitch = 0.15;
    const float strip2SZpitch = 5.0;

    // Nomenclature
    // outer segment inner mini-doublet will be referred to as "OutLo"
    // inner segment inner mini-doublet will be referred to as "InLo"
    // outer segment outer mini-doublet will be referred to as "OutUp"
    // inner segment outer mini-doublet will be referred to as "InUp"
    // Pair of MDs in oSG will be referred to as "OutSeg"
    // Pair of MDs in iSG will be referred to as "InSeg"

    const bool isPS_InLo = ((innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).moduleType() == SDL::Module::PS);
    const bool isPS_OutLo = ((outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).moduleType() == SDL::Module::PS);
    const bool isEC_OutUp = ((outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).subdet() == SDL::Module::Endcap);

    const float rt_InLo = innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->rt();
    const float rt_OutLo = outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->rt();
    const float z_InLo = innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->z();
    const float z_OutLo = outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->z();

    const float alpha1GeV_OutLo = std::asin(std::min(rt_OutLo * k2Rinv1GeVf / ptCut, sinAlphaMax));

    const float rtRatio_OutLoInLo = rt_OutLo / rt_InLo; // Outer segment beginning rt divided by inner segment beginning rt;
    const float dzDrtScale = tan(alpha1GeV_OutLo) / alpha1GeV_OutLo; // The track can bend in r-z plane slightly
    const float zpitch_InLo = (isPS_InLo ? pixelPSZpitch : strip2SZpitch);
    const float zpitch_OutLo = (isPS_OutLo ? pixelPSZpitch : strip2SZpitch);

    // Tracklet is two segments
    // So physically it will look like the following:
    //
    // Below, the pair of x's are one segment and the pair of y's are another segment
    //
    // The x's are outer segment
    // The y's are inner segment
    //
    // rt
    //  ^
    //  |    -----------x----- ----------- -----x-----------
    //  |    ------------x---- ----------- ----x------------
    //  |
    //  |    --------------x-- ----------- --x--------------
    //  |    ---------------x- ---------|- -x--|------------ <- z_OutLo, rt_OutLo
    //  |                               <------>
    //  |                              zLo     zHi
    //  |
    //  |
    //  |    ----------------- -y-------y- -----------------
    //  |    ----------------- --y-----y-- -----------------
    //  |
    //  |    ----------------- ---y---y--- -----------------
    //  |    ----------------- ----y-y---- ----------------- <- z_InLo, rt_InLo
    //  |
    //  |                           *
    //  |                     <----> <----> <-- deltaZLum
    //  |
    //  |-----------------------------------------------------------------> z
    //

    // From the picture above, let's say we want to guess the zHi for example.
    //
    // Then we write down the equation
    //
    //    (zHi - z_InLo)           (z_InLo + deltaZLum)
    // ------------------------   = ---------------------------
    // (rt_OutLo - rt_InLo)            rt_InLo
    //
    // Then, if you solve for zHi you get most of the the below equation
    // Then there are two corrections.
    // dzDrtScale to account for the forward-bending of the helix tracks
    // z_pitch terms to additonally add error

    const float zHi = z_InLo + (z_InLo + deltaZLum) * (rtRatio_OutLoInLo - 1.f) * (z_InLo < 0.f ? 1.f : dzDrtScale) + (zpitch_InLo + zpitch_OutLo);
    const float zLo = z_InLo + (z_InLo - deltaZLum) * (rtRatio_OutLoInLo - 1.f) * (z_InLo > 0.f ? 1.f : dzDrtScale) - (zpitch_InLo + zpitch_OutLo); //slope-correction only on outer end

    // Reset passBitsDefaultAlgo_;
    passBitsDefaultAlgo_ = 0;

    //==========================================================================
    //
    // Cut #1: Z compatibility
    //
    //==========================================================================
    setRecoVars("z_OutLo", z_OutLo);
    setRecoVars("zLo", zLo);
    setRecoVars("zHi", zHi);
    if (not (z_OutLo >= zLo and z_OutLo <= zHi))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #2 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " zLo: " << zLo <<  " z_OutLo: " << z_OutLo <<  " zHi: " << zHi <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_TLAlgo);
        return;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TrackletSelection::deltaZ);
    //--------------------------------------------------------------------------

    const float drt_OutLo_InLo = (rt_OutLo - rt_InLo);
    const float invRt_InLo = 1. / rt_InLo;
    const float r3_InLo = sqrt(z_InLo * z_InLo + rt_InLo * rt_InLo);
    const float drt_InSeg = innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->rt() - innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->rt();
    const float dz_InSeg  = innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->z() - innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->z();
    const float dr3_InSeg = innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->r3() - innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->r3();

    // The x's are outer segment
    // The y's are inner segment
    //
    // rt
    //  ^
    //  |    -----------x----- ----------- -----x-----------
    //  |    ------------x---- ----------- ----x------------
    //  |
    //  |    --------------x-- ----------- --x--------------
    //  |    ---------------x- ---------|- -x--|------------ <- z_OutLo, rt_OutLo
    //  |                               <------>
    //  |                       zLoPointed  ^  zHiPointed
    //  |                                   |
    //  |                                 dzMean
    //  |    ----------------- -y-------y- -----------------
    //  |    ----------------- --y-----y-- ----------------- <-|
    //  |                                                      | "InSeg"
    //  |    ----------------- ---y---y--- -----------------   | dz, drt, tl_axis is defined above
    //  |    ----------------- ----y-y---- ----------------- <-|
    //  |
    //  |
    //  |-----------------------------------------------------------------> z
    //
    //
    // We point it via the inner segment

    // direction estimate
    const float coshEta = dr3_InSeg / drt_InSeg;

    // Defining the error terms along the z-direction
    float dzErr = (zpitch_InLo + zpitch_OutLo) * (zpitch_InLo + zpitch_OutLo) * 2.f; //both sides contribute to direction uncertainty

    // Multiple scattering
    //FIXME (later) more realistic accounting of material effects is needed
    const float sdlThetaMulsF = 0.015f * sqrt(0.1f + 0.2 * (rt_OutLo - rt_InLo) / 50.f) * sqrt(r3_InLo / rt_InLo);
    const float sdlMuls = sdlThetaMulsF * 3.f / ptCut * 4.f; // will need a better guess than x4?
    dzErr += sdlMuls * sdlMuls * drt_OutLo_InLo * drt_OutLo_InLo / 3.f * coshEta * coshEta; //sloppy
    dzErr = sqrt(dzErr);

    // Constructing upper and lower bound
    const float dzMean = dz_InSeg / drt_InSeg * drt_OutLo_InLo;
    const float zWindow = dzErr / drt_InSeg * drt_OutLo_InLo + (zpitch_InLo + zpitch_OutLo); //FIXME for ptCut lower than ~0.8 need to add curv path correction
    const float zLoPointed = z_InLo + dzMean * (z_InLo > 0.f ? 1.f : dzDrtScale) - zWindow;
    const float zHiPointed = z_InLo + dzMean * (z_InLo < 0.f ? 1.f : dzDrtScale) + zWindow;

    //==========================================================================
    //
    // Cut #2: Pointed Z (Inner segment two MD points to outer segment inner MD)
    //
    //==========================================================================
    setRecoVars("zLoPointed", zLoPointed);
    setRecoVars("zHiPointed", zHiPointed);
    if (not (z_OutLo >= zLoPointed and z_OutLo <= zHiPointed))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #3 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " zLoPointed: " << zLoPointed <<  " z_OutLo: " << z_OutLo <<  " zHiPointed: " << zHiPointed <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_TLAlgo);
        return;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TrackletSelection::deltaZPointed);
    //--------------------------------------------------------------------------

    // phi angle cut
    const float sdlPVoff = 0.1f / rt_OutLo;
    const float sdlCut = alpha1GeV_OutLo + sqrt(sdlMuls * sdlMuls + sdlPVoff * sdlPVoff);
    const MiniDoublet& md_InUp = (*innerSegmentPtr()->outerMiniDoubletPtr());
    const MiniDoublet& md_OutUp = (*outerSegmentPtr()->outerMiniDoubletPtr());
    const Hit& hit_InUp = (*innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr());
    const Hit& hit_OutUp = (*outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr());
    const Hit& hit_InRef = (*innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr());
    const Hit& hit_OutRef = (*outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr());

    //==========================================================================
    //
    // Cut #3: deltaPhiPos can be tighter
    //
    //==========================================================================
    float deltaPhiPos = hit_InUp.deltaPhi(hit_OutUp);
    setRecoVars("deltaPhiPos", deltaPhiPos);
    setRecoVars("sdlCut", sdlCut);
    if (not (std::abs(deltaPhiPos) <= sdlCut) )
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #4 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " deltaPhiPos: " << deltaPhiPos <<  " sdlCut: " << sdlCut <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_TLAlgo);
        return;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TrackletSelection::deltaPhiPos);
    //--------------------------------------------------------------------------

    // The x's are outer segment
    // The y's are inner segment
    //
    // y
    //  ^
    //  |    -----------x-----
    //  |    ------------x----
    //  |
    //  |    --------------x--
    //  |    ---------------x-
    //  |
    //  |
    //  |
    //  |                 *
    //  |    --------------y--
    //  |    -------------y---
    //  |
    //  |    -----------y-----
    //  |    ----------y------
    //  |
    //  |
    //  |-----------------------------------------------------------------> x
    //
    //
    // We point it via the inner segment
    const Hit& hit_InLo = (*innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr());
    const Hit& hit_OutLo = (*outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr());
    Hit midhit_InLoOutLo;
    midhit_InLoOutLo += hit_InLo;
    midhit_InLoOutLo += hit_OutLo;
    midhit_InLoOutLo /= 2;
    const float dPhi = midhit_InLoOutLo.deltaPhi(hit_OutLo - hit_InLo);

    //==========================================================================
    //
    // Cut #5: deltaPhiChange
    //
    //==========================================================================
    setRecoVars("dPhi", dPhi);
    if (not (std::abs(dPhi) <= sdlCut))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #5 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " dPhi: " << dPhi <<  " sdlCut: " << sdlCut <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_TLAlgo);
        return;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TrackletSelection::slope);
    //--------------------------------------------------------------------------

    // First obtaining the raw betaIn and betaOut values without any correction and just purely based on the mini-doublet hit positions
    const float alpha_InLo = innerSegmentPtr()->getDeltaPhiChange(); // Angle between deltaPhi(InLo, (InUp - InLo))

    // For the last layer in the BBBB algorithm can be different depending on the last layer being in the endcap or not
    const bool isEC_lastLayer = (outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule().subdet() == SDL::Module::Endcap) and (outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule().moduleType() == SDL::Module::TwoS);

    // The hiEdge corresponds to the case where the strip hits are shifted to reach its edge
    const Hit hit_OutUp_hiEdge = isEC_lastLayer ? GeometryUtil::stripHighEdgeHit(hit_OutUp) : hit_OutUp;
    const Hit hit_OutUp_loEdge = isEC_lastLayer ? GeometryUtil::stripLowEdgeHit(hit_OutUp) : hit_OutUp;

    // Various alpha_OutUp values are calculated depending on the edge
    const float alpha_OutUp = hit_OutUp.deltaPhi(hit_OutUp  - hit_OutLo);
    const float alpha_OutUp_hiEdge = hit_OutUp_hiEdge.deltaPhi(hit_OutUp_hiEdge  - hit_OutLo);
    const float alpha_OutUp_loEdge = hit_OutUp_loEdge.deltaPhi(hit_OutUp_loEdge  - hit_OutLo);

    // The Chord length also depends on the 2S hits
    const Hit tl_axis = hit_OutUp - hit_InLo; // The main axis of the tracklet
    const Hit tl_axis_hiEdge = hit_OutUp_hiEdge - hit_InLo; // The main axis of the tracklet
    const Hit tl_axis_loEdge = hit_OutUp_loEdge - hit_InLo; // The main axis of the tracklet

    // betaIn is unaffected
    float betaIn = alpha_InLo - hit_InLo.deltaPhi(tl_axis); // Seems to generally have negative value
    float betaInRHmin = betaIn;
    float betaInRHmax = betaIn;

    // betaOut has a range
    float betaOut = -alpha_OutUp + hit_OutUp.deltaPhi(tl_axis); // Seems to generally have negative value
    float betaOutRHmin = -alpha_OutUp_hiEdge + hit_OutUp_hiEdge.deltaPhi(tl_axis_hiEdge); // Seems to generally have negative value
    float betaOutRHmax = -alpha_OutUp_loEdge + hit_OutUp_loEdge.deltaPhi(tl_axis_loEdge); // Seems to generally have negative value

    setRecoVars("betaIn_0th", betaIn);
    setRecoVars("betaInRHmin", betaInRHmin);
    setRecoVars("betaInRHmax", betaInRHmax);
    setRecoVars("betaOut_0th", betaOut);
    setRecoVars("betaOutRHmin", betaOutRHmin);
    setRecoVars("betaOutRHmax", betaOutRHmax);
    setRecoVars("rawBetaIn", betaIn);
    setRecoVars("rawBetaOut", betaOut);

    const float drt_tl_axis = tl_axis.rt();
    const float drt_tl_axis_hiEdge = tl_axis_hiEdge.rt();
    const float drt_tl_axis_loEdge = tl_axis_loEdge.rt();
    //beta upper cuts: 2-strip difference for direction resolution
    const float corrF = 1.f;
    bool pass_betaIn_cut = false;//pixel seeds were already selected
    const float rt_InSeg = (hit_InUp - hit_InRef).rt();
    const float betaIn_cut = std::asin(std::min((-rt_InSeg * corrF + drt_tl_axis) * k2Rinv1GeVf / ptCut, sinAlphaMax)) + (0.02f / drt_InSeg);
    // const float betaIn_cut = std::asin((-rt_InSeg * corrF + drt_tl_axis) * k2Rinv1GeVf / ptCut) + (0.02f / drt_InSeg);
    pass_betaIn_cut = std::abs(betaInRHmin) < betaIn_cut;

    setRecoVars("betaIn_cut", betaIn_cut);
    setBetaIn(betaInRHmin);
    setBetaInCut(betaIn_cut);

    // Cut #6: first beta cut
    if (not (pass_betaIn_cut))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #6 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " betaInRHmin: " << betaInRHmin <<  " betaIn_cut: " << betaIn_cut <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_TLAlgo);
        return;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TrackletSelection::dAlphaIn);

    //now the actual segment linking magic
    float betaAv = 0.5f * (betaIn + betaOut);
    //pt/k2Rinv1GeVf/2. = R
    //R*sin(betaAv) = pt/k2Rinv1GeVf/2*sin(betaAv) = drt_tl_axis/2 => pt = drt_tl_axis*k2Rinv1GeVf/sin(betaAv);
    float pt_beta = drt_tl_axis * k2Rinv1GeVf / sin(betaAv);

    const float pt_betaMax = 7.0f;

    int lIn = 5;
    int lOut = isEC_lastLayer ? 11 : 5;
    int betacormode = 0;
    const float sdOut_dr = (hit_OutUp - hit_OutLo).rt();
    const float sdOut_d = hit_OutUp.rt() - hit_OutLo.rt();
    const float diffDr = std::abs(rt_InSeg - sdOut_dr) / std::abs(rt_InSeg + sdOut_dr);

    runDeltaBetaIterations(betaIn, betaOut, betaAv, pt_beta, rt_InSeg, sdOut_dr, drt_tl_axis, lIn, pt_betaMax);

    //if (true //do it for all//diffDr > 0.05 //only if segment length is different significantly
    //        && betaIn * betaOut > 0.f
    //        && (std::abs(pt_beta) < 4.f * pt_betaMax
    //            || (lIn >= 11 && std::abs(pt_beta) < 8.f * pt_betaMax)))   //and the pt_beta is well-defined; less strict for endcap-endcap
    //{
    //    betacormode = 1;

    //    runDeltaBetaIterations(betaIn, betaOut, betaAv, pt_beta, rt_InSeg, sdOut_dr, drt_tl_axis);

    //}
    //else if (lIn < 11 && std::abs(betaOut) < 0.2 * std::abs(betaIn) && std::abs(pt_beta) < 12.f * pt_betaMax)   //use betaIn sign as ref
    //{
    //    betacormode = 2;
    //    setRecoVars("betaIn_0th", betaIn);
    //    setRecoVars("betaOut_0th", betaOut);
    //    setRecoVars("betaAv_0th", betaAv);
    //    setRecoVars("betaPt_0th", pt_beta);
    //    setRecoVars("dBeta_0th", betaIn - betaOut);

    //    const float pt_betaIn = drt_tl_axis * k2Rinv1GeVf / sin(betaIn);
    //    const float betaInUpd  = betaIn + copysign(std::asin(std::min(rt_InSeg * k2Rinv1GeVf / std::abs(pt_betaIn), sinAlphaMax)), betaIn); //FIXME: need a faster version
    //    const float betaOutUpd = betaOut + copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(pt_betaIn), sinAlphaMax)), betaIn); //FIXME: need a faster version
    //    betaAv = (std::abs(betaOut) > 0.2f * std::abs(betaIn)) ? (0.5f * (betaInUpd + betaOutUpd)) : betaInUpd;
    //    pt_beta = drt_tl_axis * k2Rinv1GeVf / sin(betaAv); //get a better pt estimate
    //    betaIn  += copysign(std::asin(std::min(rt_InSeg * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaIn); //FIXME: need a faster version
    //    betaOut += copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaIn); //FIXME: need a faster version
    //    //update the av and pt
    //    betaAv = 0.5f * (betaIn + betaOut);
    //    pt_beta = drt_tl_axis * k2Rinv1GeVf / sin(betaAv); //get a better pt estimate

    //    setRecoVars("betaIn_1st", betaIn);
    //    setRecoVars("betaOut_1st", betaOut);
    //    setRecoVars("betaAv_1st", betaAv);
    //    setRecoVars("betaPt_1st", pt_beta);
    //    setRecoVars("dBeta_1st", betaIn - betaOut);

    //    setRecoVars("betaIn_2nd", betaIn);
    //    setRecoVars("betaOut_2nd", betaOut);
    //    setRecoVars("betaAv_2nd", betaAv);
    //    setRecoVars("betaPt_2nd", pt_beta);
    //    setRecoVars("dBeta_2nd", betaIn - betaOut);

    //    setRecoVars("betaIn_3rd", betaIn);
    //    setRecoVars("betaOut_3rd", betaOut);
    //    setRecoVars("betaAv_3rd", betaAv);
    //    setRecoVars("betaPt_3rd", pt_beta);
    //    setRecoVars("dBeta_3rd", betaIn - betaOut);

    //    setRecoVars("betaIn_4th", betaIn);
    //    setRecoVars("betaOut_4th", betaOut);
    //    setRecoVars("betaAv_4th", betaAv);
    //    setRecoVars("betaPt_4th", pt_beta);
    //    setRecoVars("dBeta_4th", betaIn - betaOut);

    //}
    //else
    //{
    //    betacormode = 3;

    //    setRecoVars("betaIn_0th", betaIn);
    //    setRecoVars("betaOut_0th", betaOut);
    //    setRecoVars("betaAv_0th", betaAv);
    //    setRecoVars("betaPt_0th", pt_beta);
    //    setRecoVars("dBeta_0th", betaIn - betaOut);

    //    setRecoVars("betaIn_1st", betaIn);
    //    setRecoVars("betaOut_1st", betaOut);
    //    setRecoVars("betaAv_1st", betaAv);
    //    setRecoVars("betaPt_1st", pt_beta);
    //    setRecoVars("dBeta_1st", betaIn - betaOut);

    //    setRecoVars("betaIn_2nd", betaIn);
    //    setRecoVars("betaOut_2nd", betaOut);
    //    setRecoVars("betaAv_2nd", betaAv);
    //    setRecoVars("betaPt_2nd", pt_beta);
    //    setRecoVars("dBeta_2nd", betaIn - betaOut);

    //    setRecoVars("betaIn_3rd", betaIn);
    //    setRecoVars("betaOut_3rd", betaOut);
    //    setRecoVars("betaAv_3rd", betaAv);
    //    setRecoVars("betaPt_3rd", pt_beta);
    //    setRecoVars("dBeta_3rd", betaIn - betaOut);

    //    setRecoVars("betaIn_4th", betaIn);
    //    setRecoVars("betaOut_4th", betaOut);
    //    setRecoVars("betaAv_4th", betaAv);
    //    setRecoVars("betaPt_4th", pt_beta);
    //    setRecoVars("dBeta_4th", betaIn - betaOut);

    //}

    //rescale the ranges proportionally
    const float betaInMMSF = (std::abs(betaInRHmin + betaInRHmax) > 0) ? (2.f * betaIn / std::abs(betaInRHmin + betaInRHmax)) : 0.; //mean value of min,max is the old betaIn
    const float betaOutMMSF = (std::abs(betaOutRHmin + betaOutRHmax) > 0) ? (2.f * betaOut / std::abs(betaOutRHmin + betaOutRHmax)) : 0.;
    betaInRHmin *= betaInMMSF;
    betaInRHmax *= betaInMMSF;
    betaOutRHmin *= betaOutMMSF;
    betaOutRHmax *= betaOutMMSF;

    const float dBetaMuls = sdlThetaMulsF * 4.f / std::min(std::abs(pt_beta), pt_betaMax); //need to confirm the range-out value of 7 GeV

    //regularize to alpha of pt_betaMax .. review may want to add resolution
    const float sdIn_rt = hit_InRef.rt();
    const float sdOut_rt = hit_OutLo.rt();
    const float sdIn_z = hit_InRef.z();
    const float sdOut_z = hit_OutLo.z();
    const float alpha_OutLo = outerSegmentPtr()->getDeltaPhiChange(); // Angle between deltaPhi(OutLo, (OutUp - OutLo))
    const float alphaInAbsReg = std::max(std::abs(alpha_InLo), std::asin(std::min(sdIn_rt * k2Rinv1GeVf / 3.0f, sinAlphaMax)));
    const float alphaOutAbsReg = std::max(std::abs(alpha_OutLo), std::asin(std::min(sdOut_rt * k2Rinv1GeVf / 3.0f, sinAlphaMax)));
    const float dBetaInLum = lIn < 11 ? 0.0f : std::abs(alphaInAbsReg*deltaZLum / sdIn_z);
    const float dBetaOutLum = lOut < 11 ? 0.0f : std::abs(alphaOutAbsReg*deltaZLum / sdOut_z);
    const float dBetaLum2 = (dBetaInLum + dBetaOutLum) * (dBetaInLum + dBetaOutLum);

    const float sinDPhi = std::sin(dPhi);
    // const float dBetaRIn2 = std::pow((sdIn.mdRef.rtRHout - sdIn.mdRef.rtRHin) * sinDPhi / drt_tl_axis, 2); //TODO-RH: Ask Slava about this rtRHout? rtRHin?
    const float dBetaRIn2 = 0; // TODO-RH
    const float dBetaROut2 = std::pow((hit_OutUp_hiEdge.rt() - hit_OutUp_loEdge.rt()) * sinDPhi / drt_tl_axis, 2); //TODO-RH
    // const float dBetaROut2 = 0; // TODO-RH

    const float betaOut_cut = std::asin(std::min(drt_tl_axis*k2Rinv1GeVf / ptCut, sinAlphaMax)) //FIXME: need faster version
        + (0.02f / sdOut_d) + sqrt(dBetaLum2 + dBetaMuls*dBetaMuls);
    // const float betaOut_cut = std::min(0.01, std::asin(std::min(drt_tl_axis*k2Rinv1GeVf / ptCut, sinAlphaMax)) + (0.02f / sdOut_d) + sqrt(dBetaLum2 + dBetaMuls*dBetaMuls));

    setBetaOut(betaOut);
    setBetaOutCut(betaOut_cut);

    // Cut #7: The real beta cut
    if (not (std::abs(betaOut) < betaOut_cut))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #7 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " betaOut: " << betaOut <<  " betaOut_cut: " << betaOut_cut <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_TLAlgo);
        return;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TrackletSelection::dAlphaOut);

    float pt_betaIn = drt_tl_axis * k2Rinv1GeVf / sin(betaIn);
    const float pt_betaOut = drt_tl_axis * k2Rinv1GeVf / sin(betaOut);
    const float dBetaRes = 0.02f / std::min(sdOut_d, drt_InSeg);
    const float dBetaCut2 = (dBetaRes*dBetaRes * 2.0f + dBetaMuls * dBetaMuls + dBetaLum2 + dBetaRIn2 + dBetaROut2
            + 0.25 * std::pow(std::abs(betaInRHmin - betaInRHmax) + std::abs(betaOutRHmin - betaOutRHmax), 2));
    float dBeta = betaIn - betaOut;
    // const float dZeta = sdIn.zeta - sdOut.zeta;

    const float innerSgInnerMdDetId = (innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).detId();
    const float outerSgInnerMdDetId = (outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).detId();
    const float innerSgOuterMdDetId = (innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).detId();
    const float outerSgOuterMdDetId = (outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).detId();

    setRecoVars("hit1_x", innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->x());
    setRecoVars("hit1_y", innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->y());
    setRecoVars("hit2_x", innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->x());
    setRecoVars("hit2_y", innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->y());
    setRecoVars("hit3_x", outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->x());
    setRecoVars("hit3_y", outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->y());
    setRecoVars("hit4_x", outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->x());
    setRecoVars("hit4_y", outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->y());


    std::function<float()> dBeta_midPoint = [&]()
        {

            float hit1_x = innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->x();
            float hit1_y = innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->y();
            float hit2_x = innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->x();
            float hit2_y = innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->y();
            float hit3_x = outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->x();
            float hit3_y = outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->y();
            float hit4_x = outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->x();
            float hit4_y = outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->y();

            float innerSgMidX = (hit1_x + hit2_x) / 2.;
            float innerSgMidY = (hit1_y + hit2_y) / 2.;
            float outerSgMidX = (hit3_x + hit4_x) / 2.;
            float outerSgMidY = (hit3_y + hit4_y) / 2.;

            float vecA_x = hit2_x - innerSgMidX;
            float vecA_y = hit2_y - innerSgMidY;
            float vecB_x = outerSgMidX - innerSgMidX;
            float vecB_y = outerSgMidY - innerSgMidY;
            float vecC_x = hit4_x - outerSgMidX;
            float vecC_y = hit4_y - outerSgMidY;
            float vecA_mag = sqrt(vecA_x * vecA_x + vecA_y * vecA_y);
            float vecB_mag = sqrt(vecB_x * vecB_x + vecB_y * vecB_y);
            float vecC_mag = sqrt(vecC_x * vecC_x + vecC_y * vecC_y);

            float vecA_dot_vecB = vecA_x * vecB_x + vecA_y * vecB_y;
            float vecB_dot_vecC = vecB_x * vecC_x + vecB_y * vecC_y;

            float angle_AB = std::acos(vecA_dot_vecB / vecA_mag / vecB_mag);
            float angle_BC = std::acos(vecB_dot_vecC / vecB_mag / vecC_mag);

            return angle_AB - angle_BC;

        };

    setRecoVars("sinAlphaMax", sinAlphaMax);
    setRecoVars("betaIn", betaIn);
    setRecoVars("betaInRHmax", betaInRHmax);
    setRecoVars("betaInRHmin", betaInRHmin);
    setRecoVars("betaOut", betaOut);
    setRecoVars("betaOutRHmax", betaOutRHmax);
    setRecoVars("betaOutRHmin", betaOutRHmin);
    setRecoVars("dBeta", dBeta);
    setRecoVars("dBetaCut2", dBetaCut2);
    setRecoVars("dBetaLum2", dBetaLum2);
    setRecoVars("dBetaMuls", dBetaMuls);
    setRecoVars("dBetaRIn2", dBetaRIn2);
    setRecoVars("dBetaROut2", dBetaROut2);
    setRecoVars("dBetaRes", dBetaRes);
    setRecoVars("deltaZLum", deltaZLum);
    setRecoVars("drt_tl_axis", drt_tl_axis);
    setRecoVars("dzDrtScale", dzDrtScale);
    setRecoVars("innerSgInnerMdDetId", innerSgInnerMdDetId);
    setRecoVars("innerSgOuterMdDetId", innerSgOuterMdDetId);
    setRecoVars("k2Rinv1GeVf", k2Rinv1GeVf);
    setRecoVars("kRinv1GeVf", kRinv1GeVf);
    setRecoVars("outerSgInnerMdDetId", outerSgInnerMdDetId);
    setRecoVars("outerSgOuterMdDetId", outerSgOuterMdDetId);
    setRecoVars("pixelPSZpitch", pixelPSZpitch);
    setRecoVars("ptCut", ptCut);
    setRecoVars("pt_betaIn", pt_betaIn);
    setRecoVars("pt_betaOut", pt_betaOut);
    setRecoVars("rt_InLo", rt_InLo);
    setRecoVars("rt_OutLo", rt_OutLo);
    setRecoVars("rtRatio_OutLoInLo", rtRatio_OutLoInLo);
    setRecoVars("drt_InSeg", drt_InSeg);
    setRecoVars("sdOut_d", sdOut_d);
    setRecoVars("alpha1GeV_OutLo", alpha1GeV_OutLo);
    setRecoVars("strip2SZpitch", strip2SZpitch);
    setRecoVars("zpitch_InLo", zpitch_InLo);
    setRecoVars("zpitch_OutLo", zpitch_OutLo);
    setRecoVars("z_InLo", z_InLo);
    setRecoVars("zLo", zLo);
    setRecoVars("z_OutLo", z_OutLo);
    setRecoVars("betacormode", betacormode);
    setRecoVars("pt_beta", pt_beta);
    setRecoVars("betaAv", betaAv);
    setRecoVars("dBeta_midPoint", dBeta_midPoint());

    // dBeta = getRecoVar("dBeta_4th");

    setDeltaBeta(dBeta);
    setDeltaBetaCut(std::sqrt(dBetaCut2));

    if (not (dBeta * dBeta <= dBetaCut2))
    // if (not (dBeta * dBeta <= 0.004*0.004))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #8 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " dBeta*dBeta: " << dBeta*dBeta <<  " dBetaCut2: " << dBetaCut2 <<  std::endl;
            SDL::cout <<  " dBetaRes: " << dBetaRes <<  " dBetaMuls: " << dBetaMuls <<  " dBetaLum2: " << dBetaLum2 <<  std::endl;
            SDL::cout <<  " dBetaRIn2: " << dBetaRIn2 <<  " dBetaROut2: " << dBetaROut2 <<  std::endl;
            SDL::cout <<  " betaInRHmin: " << betaInRHmin <<  " betaInRHmax: " << betaInRHmax <<  std::endl;
            SDL::cout <<  " betaOutRHmin: " << betaOutRHmin <<  " betaOutRHmax: " << betaOutRHmax <<  std::endl;
        }
        if (getNm1DeltaBetaCut())
        {
            passAlgo_ |= (1 << SDL::Default_TLAlgo);
        }
        else
        {
            passAlgo_ &= (0 << SDL::Default_TLAlgo);
            return;
        }
    }
    else if (logLevel >= SDL::Log_Debug3)
    {
        SDL::cout << "Passed Cut #8 in " << __FUNCTION__ << std::endl;
        SDL::cout <<  " dBeta*dBeta: " << dBeta*dBeta <<  " dBetaCut2: " << dBetaCut2 <<  std::endl;
        SDL::cout <<  " dBetaRes: " << dBetaRes <<  " dBetaMuls: " << dBetaMuls <<  " dBetaLum2: " << dBetaLum2 <<  std::endl;
        SDL::cout <<  " dBetaRIn2: " << dBetaRIn2 <<  " dBetaROut2: " << dBetaROut2 <<  std::endl;
        SDL::cout <<  " betaInRHmin: " << betaInRHmin <<  " betaInRHmax: " << betaInRHmax <<  std::endl;
        SDL::cout <<  " betaOutRHmin: " << betaOutRHmin <<  " betaOutRHmax: " << betaOutRHmax <<  std::endl;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TrackletSelection::dBeta);

    passAlgo_ |= (1 << SDL::Default_TLAlgo);
    if (getNm1DeltaBetaCut()) passAlgo_ |= (1 << SDL::DefaultNm1_TLAlgo);
    return;
}

void SDL::Tracklet::runTrackletDefaultAlgoBarrelBarrelEndcapEndcap(SDL::LogLevel logLevel)
{

    setRecoVars("betaAv", -999);
    setRecoVars("betaAv_0th", -999);
    setRecoVars("betaAv_1st", -999);
    setRecoVars("betaAv_2nd", -999);
    setRecoVars("betaAv_3rd", -999);
    setRecoVars("betaAv_4th", -999);
    setRecoVars("betaIn", -999);
    setRecoVars("betaInRHmax", -999);
    setRecoVars("betaInRHmin", -999);
    setRecoVars("betaIn_0th", -999);
    setRecoVars("betaIn_1st", -999);
    setRecoVars("betaIn_1stCorr", -999);
    setRecoVars("betaIn_2nd", -999);
    setRecoVars("betaIn_2ndCorr", -999);
    setRecoVars("betaIn_3rd", -999);
    setRecoVars("betaIn_3rdCorr", -999);
    setRecoVars("betaIn_4th", -999);
    setRecoVars("betaOut", -999);
    setRecoVars("betaOutRHmax", -999);
    setRecoVars("betaOutRHmin", -999);
    setRecoVars("betaOut_0th", -999);
    setRecoVars("betaOut_1st", -999);
    setRecoVars("betaOut_1stCorr", -999);
    setRecoVars("betaOut_2nd", -999);
    setRecoVars("betaOut_2ndCorr", -999);
    setRecoVars("betaOut_3rd", -999);
    setRecoVars("betaOut_3rdCorr", -999);
    setRecoVars("betaOut_4th", -999);
    setRecoVars("betaPt_0th", -999);
    setRecoVars("betaPt_1st", -999);
    setRecoVars("betaPt_2nd", -999);
    setRecoVars("betaPt_3rd", -999);
    setRecoVars("betaPt_4th", -999);
    setRecoVars("betacormode", -999);
    setRecoVars("dBeta", -999);
    setRecoVars("dBetaCut2", -999);
    setRecoVars("dBetaLum2", -999);
    setRecoVars("dBetaMuls", -999);
    setRecoVars("dBetaRIn2", -999);
    setRecoVars("dBetaROut2", -999);
    setRecoVars("dBetaRes", -999);
    setRecoVars("dBeta_0th", -999);
    setRecoVars("dBeta_1st", -999);
    setRecoVars("dBeta_2nd", -999);
    setRecoVars("dBeta_3rd", -999);
    setRecoVars("dBeta_4th", -999);
    setRecoVars("dBeta_midPoint", -999);
    setRecoVars("deltaZLum", -999);
    setRecoVars("dr", -999);
    setRecoVars("dzDrtScale", -999);
    setRecoVars("hit1_x", -999);
    setRecoVars("hit1_y", -999);
    setRecoVars("hit2_x", -999);
    setRecoVars("hit2_y", -999);
    setRecoVars("hit3_x", -999);
    setRecoVars("hit3_y", -999);
    setRecoVars("hit4_x", -999);
    setRecoVars("hit4_y", -999);
    setRecoVars("innerSgInnerMdDetId", -999);
    setRecoVars("innerSgOuterMdDetId", -999);
    setRecoVars("k2Rinv1GeVf", -999);
    setRecoVars("kRinv1GeVf", -999);
    setRecoVars("outerSgInnerMdDetId", -999);
    setRecoVars("outerSgOuterMdDetId", -999);
    setRecoVars("pixelPSZpitch", -999);
    setRecoVars("ptCut", -999);
    setRecoVars("pt_beta", -999);
    setRecoVars("pt_betaIn", -999);
    setRecoVars("pt_betaOut", -999);
    setRecoVars("rawBetaIn", -999);
    setRecoVars("rawBetaInCorrection", -999);
    setRecoVars("rawBetaOut", -999);
    setRecoVars("rawBetaOutCorrection", -999);
    setRecoVars("rtIn", -999);
    setRecoVars("rtOut", -999);
    setRecoVars("rtOut_o_rtIn", -999);
    setRecoVars("sdIn_alpha", -999);
    setRecoVars("sdIn_d", -999);
    setRecoVars("sdOut_alphaOut", -999);
    setRecoVars("sdOut_d", -999);
    setRecoVars("sdlSlope", -999);
    setRecoVars("sinAlphaMax", -999);
    setRecoVars("strip2SZpitch", -999);
    setRecoVars("zGeom", -999);
    setRecoVars("zIn", -999);
    setRecoVars("zLo", -999);
    setRecoVars("zOut", -999);

    const float deltaZLum = 15.f;
    const float pixelPSZpitch = 0.15;
    const float strip2SZpitch = 5.0;
    const float zGeom =
        ((innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).moduleType() == SDL::Module::PS ? pixelPSZpitch : strip2SZpitch)
        +
        ((outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).moduleType() == SDL::Module::PS ? pixelPSZpitch : strip2SZpitch);
    const float rtIn = innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->rt();
    const float rtOut = outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->rt();
    const float zIn = innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->z();
    const float zOut = outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->z();
    const float rtOut_o_rtIn = rtOut / rtIn;
    const float kRinv1GeVf = (2.99792458e-3 * 3.8);
    const float k2Rinv1GeVf = kRinv1GeVf / 2.;
    const float ptCut = PTCUT;
    const float sinAlphaMax = 0.95;
    const float sdlSlope = std::asin(std::min(rtOut * k2Rinv1GeVf / ptCut, sinAlphaMax));
    const float dzDrtScale = tan(sdlSlope) / sdlSlope;//FIXME: need approximate value
    const float zLo = zIn + (zIn - deltaZLum) * (rtOut_o_rtIn - 1.f) * (zIn > 0.f ? 1.f : dzDrtScale) - zGeom; //slope-correction only on outer end

    // Cut #0: Preliminary
    // (Only here in endcap case)
    if (not (zIn * zOut > 0))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #0 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " zIn: " << zIn <<  std::endl;
            SDL::cout <<  " zOut: " << zOut <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_TLAlgo);
        return;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TrackletSelection::deltaZ);


    const float dLum = std::copysign(deltaZLum, zIn);
    const bool isOutSgInnerMDPS = (outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).moduleType() == SDL::Module::PS;
    const float rtGeom1 = isOutSgInnerMDPS ? pixelPSZpitch : strip2SZpitch;//FIXME: make this chosen by configuration for lay11,12 full PS
    const float zGeom1 = std::copysign(zGeom, zIn); //used in B-E region
    const float rtLo = rtIn * (1.f + (zOut - zIn - zGeom1) / (zIn + zGeom1 + dLum) / dzDrtScale) - rtGeom1; //slope correction only on the lower end

    // Cut #1: rt condition
    if (not (rtOut >= rtLo))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #1 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " rtOut: " << rtOut <<  std::endl;
            SDL::cout <<  " rtLo: " << rtLo <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_TLAlgo);
        return;
    }

    float zInForHi = zIn - zGeom1 - dLum;
    if (zInForHi * zIn < 0)
        zInForHi = std::copysign(0.1f, zIn);
    const float rtHi = rtIn * (1.f + (zOut - zIn + zGeom1) / zInForHi) + rtGeom1;

    // Cut #2: rt condition
    if (not (rtOut >= rtLo and rtOut <= rtHi))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #2 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " rtOut: " << rtOut <<  std::endl;
            SDL::cout <<  " rtLo: " << rtLo <<  std::endl;
            SDL::cout <<  " rtHi: " << rtHi <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_TLAlgo);
        return;
    }

    const float rIn = sqrt(zIn*zIn + rtIn*rtIn);
    const float drtSDIn = innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->rt() - innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->rt();
    const float dzSDIn = innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->z() - innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->z();
    const float dr3SDIn = innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->r3() - innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->r3();
    const float coshEta = dr3SDIn / drtSDIn; //direction estimate
    const float dzOutInAbs = std::abs(zOut - zIn);
    const float multDzDr = dzOutInAbs * coshEta / (coshEta * coshEta - 1.f);
    const float zGeom1_another = pixelPSZpitch; // TODO-Q Why only one?
    const float kZ = (zOut - zIn) / dzSDIn;
    float drtErr = zGeom1_another * zGeom1_another * drtSDIn * drtSDIn / dzSDIn / dzSDIn * (1.f - 2.f * kZ + 2.f * kZ * kZ); //Notes:122316
    const float sdlThetaMulsF = 0.015f * sqrt(0.1f + 0.2 * (rtOut - rtIn) / 50.f) * sqrt(rIn / rtIn);
    const float sdlMuls = sdlThetaMulsF * 3.f / ptCut * 4.f; //will need a better guess than x4?
    drtErr += sdlMuls * sdlMuls * multDzDr * multDzDr / 3.f * coshEta * coshEta; //sloppy: relative muls is 1/3 of total muls
    drtErr = sqrt(drtErr);
    const float drtMean = drtSDIn * dzOutInAbs / std::abs(dzSDIn); //
    const float rtWindow = drtErr + rtGeom1;
    const float rtLo_another = rtIn + drtMean / dzDrtScale - rtWindow;
    const float rtHi_another = rtIn + drtMean + rtWindow;

    // Cut #3: rt-z pointed
    if (not (kZ >= 0 and rtOut >= rtLo and rtOut <= rtHi))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #3 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " kZ: " << kZ <<  std::endl;
            SDL::cout <<  " rtOut: " << rtOut <<  std::endl;
            SDL::cout <<  " rtLo: " << rtLo <<  std::endl;
            SDL::cout <<  " rtHi: " << rtHi <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_TLAlgo);
        return;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TrackletSelection::deltaZPointed);

    const float sdlPVoff = 0.1f / rtOut;
    const float sdlCut = sdlSlope + sqrt(sdlMuls * sdlMuls + sdlPVoff*sdlPVoff);
    const MiniDoublet& sdIn_mdOut = (*innerSegmentPtr()->outerMiniDoubletPtr());
    const MiniDoublet& sdOut_mdOut = (*outerSegmentPtr()->outerMiniDoubletPtr());
    const Hit& sdIn_mdOut_hit = (*innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr());
    const Hit& sdOut_mdOut_hit = (*outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr());

    // Cut #4: deltaPhiPos can be tighter
    float deltaPhiPos = sdIn_mdOut_hit.deltaPhi(sdOut_mdOut_hit);
    if (not (std::abs(deltaPhiPos) <= sdlCut) )
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #4 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " deltaPhiPos: " << deltaPhiPos <<  " sdlCut: " << sdlCut <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_TLAlgo);
        return;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TrackletSelection::deltaPhiPos);

    const Hit& sdIn_r3 = (*innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr());
    const Hit& sdOut_r3 = (*outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr());
    Hit midR3;
    midR3 += sdIn_r3;
    midR3 += sdOut_r3;
    midR3 /= 2;
    const float dPhi = midR3.deltaPhi(sdOut_r3 - sdIn_r3);

    // Cut #5: deltaPhiChange
    if (not (std::abs(dPhi) <= sdlCut))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #5 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " dPhi: " << dPhi <<  " sdlCut: " << sdlCut <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_TLAlgo);
        return;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TrackletSelection::slope);

    const Hit& sdOut_mdRef_hit = (*outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr());
    const float sdIn_alpha = innerSegmentPtr()->getDeltaPhiChange();
    const float sdIn_alpha_min = innerSegmentPtr()->getDeltaPhiMinChange();
    const float sdIn_alpha_max = innerSegmentPtr()->getDeltaPhiMaxChange();
    const float sdOut_alpha = innerSegmentPtr()->getDeltaPhiChange();
    const float sdOut_alphaOut = sdOut_mdOut_hit.deltaPhi(sdOut_mdOut_hit  - sdOut_mdRef_hit);
    const float sdOut_alphaOut_min = SDL::MathUtil::Phi_mpi_pi(outerSegmentPtr()->getDeltaPhiMinChange() - outerSegmentPtr()->getDeltaPhiMin());
    const float sdOut_alphaOut_max = SDL::MathUtil::Phi_mpi_pi(outerSegmentPtr()->getDeltaPhiMaxChange() - outerSegmentPtr()->getDeltaPhiMax());
    const Hit& sdOut_mdOut_r3 = (*outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr());
    const Hit dr3 = sdOut_mdOut_r3 - sdIn_r3;
    float betaIn  = sdIn_alpha - sdIn_r3.deltaPhi(dr3);
    float betaOut = -sdOut_alphaOut + sdOut_mdOut_r3.deltaPhi(dr3);
    const bool isEC_secondLayer = (innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule().subdet() == SDL::Module::Endcap) and (innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule().subdet() == SDL::Module::TwoS);
    float betaInRHmin = isEC_secondLayer ? betaIn - sdIn_alpha_min + sdIn_alpha : betaIn;
    float betaInRHmax = isEC_secondLayer ? betaIn - sdIn_alpha_max + sdIn_alpha : betaIn;
    float betaOutRHmin = betaOut - sdOut_alphaOut_min + sdOut_alphaOut;
    float betaOutRHmax = betaOut - sdOut_alphaOut_max + sdOut_alphaOut;
    if (std::abs(betaOutRHmin) > std::abs(betaOutRHmax)) std::swap(betaOutRHmax, betaOutRHmin);
    if (std::abs(betaInRHmin) > std::abs(betaInRHmax)) std::swap(betaInRHmax, betaInRHmin);

    const Hit& sdIn_mdRef_hit = (*innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr());

    const float sdIn_dr = (sdIn_mdOut_hit - sdIn_mdRef_hit).rt();
    const float sdIn_d = sdIn_mdOut_hit.rt() - sdIn_mdRef_hit.rt();

    const float dr = dr3.rt();
    //beta upper cuts: 2-strip difference for direction resolution
    const float corrF = 1.f;
    bool pass_betaIn_cut = false;//pixel seeds were already selected
    const float betaIn_cut = std::asin(std::min((-sdIn_dr * corrF + dr) * k2Rinv1GeVf / ptCut, sinAlphaMax)) + (0.02f / sdIn_d);
    pass_betaIn_cut = std::abs(betaInRHmin) < betaIn_cut;

    // Cut #6: first beta cut
    if (not (pass_betaIn_cut))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #6 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " betaInRHmin: " << betaInRHmin <<  " betaIn_cut: " << betaIn_cut <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_TLAlgo);
        return;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TrackletSelection::dAlphaIn);

    //now the actual segment linking magic
    float betaAv = 0.5f * (betaIn + betaOut);
    //pt/k2Rinv1GeVf/2. = R
    //R*sin(betaAv) = pt/k2Rinv1GeVf/2*sin(betaAv) = dr/2 => pt = dr*k2Rinv1GeVf/sin(betaAv);
    float pt_beta = dr * k2Rinv1GeVf / sin(betaAv);

    const float pt_betaMax = 7.0f;

    int lIn = 5;
    int lOut = 11;

    const float sdOut_dr = (sdOut_mdOut_hit - sdOut_mdRef_hit).rt();
    const float sdOut_d = sdOut_mdOut_hit.rt() - sdOut_mdRef_hit.rt();
    const float diffDr = std::abs(sdIn_dr - sdOut_dr) / std::abs(sdIn_dr + sdOut_dr);

    runDeltaBetaIterations(betaIn, betaOut, betaAv, pt_beta, sdIn_dr, sdOut_dr, dr, lIn, pt_betaMax);

    //if (true //do it for all//diffDr > 0.05 //only if segment length is different significantly
    //        && betaIn * betaOut > 0.f
    //        && (std::abs(pt_beta) < 4.f * pt_betaMax
    //            || (lIn >= 11 && std::abs(pt_beta) < 8.f * pt_betaMax)))   //and the pt_beta is well-defined; less strict for endcap-endcap
    //{

    //    runDeltaBetaIterations(betaIn, betaOut, betaAv, pt_beta, sdIn_dr, sdOut_dr, dr);

    //    //setRecoVars("betaIn_0th", betaIn);
    //    //setRecoVars("betaOut_0th", betaOut);
    //    //setRecoVars("betaAv_0th", betaAv);
    //    //setRecoVars("betaPt_0th", pt_beta);
    //    //setRecoVars("betaIn_1stCorr", copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaIn));
    //    //setRecoVars("betaOut_1stCorr", copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaOut));
    //    //setRecoVars("dBeta_0th", betaIn - betaOut);

    //    //const float betaInUpd  = betaIn + copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaIn); //FIXME: need a faster version
    //    //const float betaOutUpd = betaOut + copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaOut); //FIXME: need a faster version
    //    //betaAv = 0.5f * (betaInUpd + betaOutUpd);
    //    //pt_beta = dr * k2Rinv1GeVf / sin(betaAv); //get a better pt estimate

    //    //setRecoVars("betaIn_1st", betaInUpd);
    //    //setRecoVars("betaOut_1st", betaOutUpd);
    //    //setRecoVars("betaAv_1st", betaAv);
    //    //setRecoVars("betaPt_1st", pt_beta);
    //    //setRecoVars("betaIn_2ndCorr", copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaIn));
    //    //setRecoVars("betaOut_2ndCorr", copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaOut));
    //    //setRecoVars("dBeta_1st", betaInUpd - betaOutUpd);

    //    //betaIn  += copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaIn); //FIXME: need a faster version
    //    //betaOut += copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaOut); //FIXME: need a faster version
    //    ////update the av and pt
    //    //betaAv = 0.5f * (betaIn + betaOut);
    //    //pt_beta = dr * k2Rinv1GeVf / sin(betaAv); //get a better pt estimate

    //    //setRecoVars("betaIn_2nd", betaIn);
    //    //setRecoVars("betaOut_2nd", betaOut);
    //    //setRecoVars("betaAv_2nd", betaAv);
    //    //setRecoVars("betaPt_2nd", pt_beta);
    //    //setRecoVars("betaIn_3rdCorr", copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaIn));
    //    //setRecoVars("betaOut_3rdCorr", copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaOut));
    //    //setRecoVars("dBeta_2nd", betaIn - betaOut);

    //    //setRecoVars("betaIn_3rd", getRecoVar("betaIn_0th") + copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaIn));
    //    //setRecoVars("betaOut_3rd", getRecoVar("betaOut_0th") + copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaOut));
    //    //setRecoVars("betaAv_3rd", 0.5f * (getRecoVar("betaIn_3rd") + getRecoVar("betaOut_3rd")));
    //    //setRecoVars("betaPt_3rd", dr * k2Rinv1GeVf / sin(getRecoVar("betaAv_3rd")));
    //    //setRecoVars("dBeta_3rd", getRecoVar("betaIn_3rd") - getRecoVar("betaOut_3rd"));

    //    //setRecoVars("betaIn_4th", getRecoVar("betaIn_0th") + copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(getRecoVar("betaPt_3rd")), sinAlphaMax)), getRecoVar("betaIn_3rd")));
    //    //setRecoVars("betaOut_4th", getRecoVar("betaOut_0th") + copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(getRecoVar("betaPt_3rd")), sinAlphaMax)), getRecoVar("betaOut_3rd")));
    //    //setRecoVars("betaAv_4th", 0.5f * (getRecoVar("betaIn_4th") + getRecoVar("betaOut_4th")));
    //    //setRecoVars("betaPt_4th", dr * k2Rinv1GeVf / sin(getRecoVar("betaAv_4th")));
    //    //setRecoVars("dBeta_4th", getRecoVar("betaIn_4th") - getRecoVar("betaOut_4th"));

    //}
    //else if (lIn < 11 && std::abs(betaOut) < 0.2 * std::abs(betaIn) && std::abs(pt_beta) < 12.f * pt_betaMax)   //use betaIn sign as ref
    //{
    //    const float pt_betaIn = dr * k2Rinv1GeVf / sin(betaIn);
    //    const float betaInUpd  = betaIn + copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(pt_betaIn), sinAlphaMax)), betaIn); //FIXME: need a faster version
    //    const float betaOutUpd = betaOut + copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(pt_betaIn), sinAlphaMax)), betaIn); //FIXME: need a faster version
    //    betaAv = (std::abs(betaOut) > 0.2f * std::abs(betaIn)) ? (0.5f * (betaInUpd + betaOutUpd)) : betaInUpd;
    //    pt_beta = dr * k2Rinv1GeVf / sin(betaAv); //get a better pt estimate
    //    betaIn  += copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaIn); //FIXME: need a faster version
    //    betaOut += copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaIn); //FIXME: need a faster version
    //    //update the av and pt
    //    betaAv = 0.5f * (betaIn + betaOut);
    //    pt_beta = dr * k2Rinv1GeVf / sin(betaAv); //get a better pt estimate
    //}

    //rescale the ranges proportionally
    const float betaInMMSF = (std::abs(betaInRHmin + betaInRHmax) > 0) ? 2.f * betaIn / std::abs(betaInRHmin + betaInRHmax) : 0.; //TODO-RH: the terneary operator should no be necessary once RHmin/max is propagatedmean value of min,max is the old betaIn
    const float betaOutMMSF = (std::abs(betaOutRHmin + betaOutRHmax) > 0) ? 2.f * betaOut / std::abs(betaOutRHmin + betaOutRHmax) : 0.;
    betaInRHmin *= betaInMMSF;
    betaInRHmax *= betaInMMSF;
    betaOutRHmin *= betaOutMMSF;
    betaOutRHmax *= betaOutMMSF;

    const float dBetaMuls = sdlThetaMulsF * 4.f / std::min(std::abs(pt_beta), pt_betaMax); //need to confirm the range-out value of 7 GeV

    //regularize to alpha of pt_betaMax .. review may want to add resolution
    const float sdIn_rt = sdIn_mdRef_hit.rt();
    const float sdOut_rt = sdOut_mdRef_hit.rt();
    const float sdIn_z = sdIn_mdRef_hit.z();
    const float sdOut_z = sdOut_mdRef_hit.z();
    const float alphaInAbsReg = std::max(std::abs(sdIn_alpha), std::asin(std::min(sdIn_rt * k2Rinv1GeVf / 3.0f, sinAlphaMax)));
    const float alphaOutAbsReg = std::max(std::abs(sdOut_alpha), std::asin(std::min(sdOut_rt * k2Rinv1GeVf / 3.0f, sinAlphaMax)));
    const float dBetaInLum = lIn < 11 ? 0.0f : std::abs(alphaInAbsReg*deltaZLum / sdIn_z);
    const float dBetaOutLum = lOut < 11 ? 0.0f : std::abs(alphaOutAbsReg*deltaZLum / sdOut_z);
    const float dBetaLum2 = (dBetaInLum + dBetaOutLum) * (dBetaInLum + dBetaOutLum);

    const float sinDPhi = std::sin(dPhi);
    // const float dBetaRIn2 = std::pow((sdIn.mdRef.rtRHout - sdIn.mdRef.rtRHin) * sinDPhi / dr, 2); //TODO-RH: Ask Slava about this rtRHout? rtRHin?
    // const float dBetaROut2 = std::pow((sdOut.mdOut.rtRHout - sdOut.mdOut.rtRHin) * sinDPhi / dr, 2); //TODO-RH: Ask Slava about this rtRHout? rtRHin?
    // const float dBetaRIn2 = 0; // TODO-RH

    // The hiEdge corresponds to the case where the strip hits are shifted to reach its edge
    const Hit& hit_OutUp_hiEdge = sdOut_mdOut_hit.getModule().moduleType() == SDL::Module::TwoS ? *sdOut_mdOut_hit.getHitHighEdgePtr() : sdOut_mdOut_hit;
    const Hit& hit_OutUp_loEdge = sdOut_mdOut_hit.getModule().moduleType() == SDL::Module::TwoS ? *sdOut_mdOut_hit.getHitLowEdgePtr() : sdOut_mdOut_hit;

    const float dBetaRIn2 = 0;
    const float dBetaROut2 = std::pow((hit_OutUp_hiEdge.rt() - hit_OutUp_loEdge.rt()) * sinDPhi / dr, 2);

    const float betaOut_cut = std::asin(std::min(dr*k2Rinv1GeVf / ptCut, sinAlphaMax)) //FIXME: need faster version
        + (0.02f / sdOut_d) + sqrt(dBetaLum2 + dBetaMuls*dBetaMuls);

    // Cut #7: The real beta cut
    if (not (std::abs(betaOut) < betaOut_cut))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #7 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " betaOut: " << betaOut <<  " betaOut_cut: " << betaOut_cut <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_TLAlgo);
        return;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TrackletSelection::dAlphaOut);

    float pt_betaIn = dr * k2Rinv1GeVf / sin(betaIn);
    const float pt_betaOut = dr * k2Rinv1GeVf / sin(betaOut);
    const float dBetaRes = 0.02f / std::min(sdOut_d, sdIn_d);
    const float dBetaCut2 = (dBetaRes*dBetaRes * 2.0f + dBetaMuls * dBetaMuls + dBetaLum2 + dBetaRIn2 + dBetaROut2
            + 0.25 * std::pow(std::abs(betaInRHmin - betaInRHmax) + std::abs(betaOutRHmin - betaOutRHmax), 2));
    const float dBeta = betaIn - betaOut;
    // const float dZeta = sdIn.zeta - sdOut.zeta;

    setDeltaBeta(dBeta);

    if (not (dBeta * dBeta <= dBetaCut2))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #8 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " dBeta*dBeta: " << dBeta*dBeta <<  " dBetaCut2: " << dBetaCut2 <<  std::endl;
            SDL::cout <<  " dBetaRes: " << dBetaRes <<  " dBetaMuls: " << dBetaMuls <<  " dBetaLum2: " << dBetaLum2 <<  std::endl;
            SDL::cout <<  " dBetaRIn2: " << dBetaRIn2 <<  " dBetaROut2: " << dBetaROut2 <<  std::endl;
            SDL::cout <<  " betaInRHmin: " << betaInRHmin <<  " betaInRHmax: " << betaInRHmax <<  std::endl;
            SDL::cout <<  " betaOutRHmin: " << betaOutRHmin <<  " betaOutRHmax: " << betaOutRHmax <<  std::endl;
        }
        if (getNm1DeltaBetaCut())
        {
            passAlgo_ |= (1 << SDL::Default_TLAlgo);
        }
        else
        {
            passAlgo_ &= (0 << SDL::Default_TLAlgo);
            return;
        }
    }
    else if (logLevel >= SDL::Log_Debug3)
    {
        SDL::cout << "Passed Cut #8 in " << __FUNCTION__ << std::endl;
        SDL::cout <<  " dBeta*dBeta: " << dBeta*dBeta <<  " dBetaCut2: " << dBetaCut2 <<  std::endl;
        SDL::cout <<  " dBetaRes: " << dBetaRes <<  " dBetaMuls: " << dBetaMuls <<  " dBetaLum2: " << dBetaLum2 <<  std::endl;
        SDL::cout <<  " dBetaRIn2: " << dBetaRIn2 <<  " dBetaROut2: " << dBetaROut2 <<  std::endl;
        SDL::cout <<  " betaInRHmin: " << betaInRHmin <<  " betaInRHmax: " << betaInRHmax <<  std::endl;
        SDL::cout <<  " betaOutRHmin: " << betaOutRHmin <<  " betaOutRHmax: " << betaOutRHmax <<  std::endl;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TrackletSelection::dBeta);


    passAlgo_ |= (1 << SDL::Default_TLAlgo);
    if (getNm1DeltaBetaCut()) passAlgo_ |= (1 << SDL::DefaultNm1_TLAlgo);
    return;
}

void SDL::Tracklet::runTrackletDefaultAlgoEndcapEndcapEndcapEndcap(SDL::LogLevel logLevel)
{

    setRecoVars("betaAv", -999);
    setRecoVars("betaAv_0th", -999);
    setRecoVars("betaAv_1st", -999);
    setRecoVars("betaAv_2nd", -999);
    setRecoVars("betaAv_3rd", -999);
    setRecoVars("betaAv_4th", -999);
    setRecoVars("betaIn", -999);
    setRecoVars("betaInRHmax", -999);
    setRecoVars("betaInRHmin", -999);
    setRecoVars("betaIn_0th", -999);
    setRecoVars("betaIn_1st", -999);
    setRecoVars("betaIn_1stCorr", -999);
    setRecoVars("betaIn_2nd", -999);
    setRecoVars("betaIn_2ndCorr", -999);
    setRecoVars("betaIn_3rd", -999);
    setRecoVars("betaIn_3rdCorr", -999);
    setRecoVars("betaIn_4th", -999);
    setRecoVars("betaOut", -999);
    setRecoVars("betaOutRHmax", -999);
    setRecoVars("betaOutRHmin", -999);
    setRecoVars("betaOut_0th", -999);
    setRecoVars("betaOut_1st", -999);
    setRecoVars("betaOut_1stCorr", -999);
    setRecoVars("betaOut_2nd", -999);
    setRecoVars("betaOut_2ndCorr", -999);
    setRecoVars("betaOut_3rd", -999);
    setRecoVars("betaOut_3rdCorr", -999);
    setRecoVars("betaOut_4th", -999);
    setRecoVars("betaPt_0th", -999);
    setRecoVars("betaPt_1st", -999);
    setRecoVars("betaPt_2nd", -999);
    setRecoVars("betaPt_3rd", -999);
    setRecoVars("betaPt_4th", -999);
    setRecoVars("betacormode", -999);
    setRecoVars("dBeta", -999);
    setRecoVars("dBetaCut2", -999);
    setRecoVars("dBetaLum2", -999);
    setRecoVars("dBetaMuls", -999);
    setRecoVars("dBetaRIn2", -999);
    setRecoVars("dBetaROut2", -999);
    setRecoVars("dBetaRes", -999);
    setRecoVars("dBeta_0th", -999);
    setRecoVars("dBeta_1st", -999);
    setRecoVars("dBeta_2nd", -999);
    setRecoVars("dBeta_3rd", -999);
    setRecoVars("dBeta_4th", -999);
    setRecoVars("dBeta_midPoint", -999);
    setRecoVars("deltaZLum", -999);
    setRecoVars("dr", -999);
    setRecoVars("dzDrtScale", -999);
    setRecoVars("hit1_x", -999);
    setRecoVars("hit1_y", -999);
    setRecoVars("hit2_x", -999);
    setRecoVars("hit2_y", -999);
    setRecoVars("hit3_x", -999);
    setRecoVars("hit3_y", -999);
    setRecoVars("hit4_x", -999);
    setRecoVars("hit4_y", -999);
    setRecoVars("innerSgInnerMdDetId", -999);
    setRecoVars("innerSgOuterMdDetId", -999);
    setRecoVars("k2Rinv1GeVf", -999);
    setRecoVars("kRinv1GeVf", -999);
    setRecoVars("outerSgInnerMdDetId", -999);
    setRecoVars("outerSgOuterMdDetId", -999);
    setRecoVars("pixelPSZpitch", -999);
    setRecoVars("ptCut", -999);
    setRecoVars("pt_beta", -999);
    setRecoVars("pt_betaIn", -999);
    setRecoVars("pt_betaOut", -999);
    setRecoVars("rawBetaIn", -999);
    setRecoVars("rawBetaInCorrection", -999);
    setRecoVars("rawBetaOut", -999);
    setRecoVars("rawBetaOutCorrection", -999);
    setRecoVars("rtIn", -999);
    setRecoVars("rtOut", -999);
    setRecoVars("rtOut_o_rtIn", -999);
    setRecoVars("sdIn_alpha", -999);
    setRecoVars("sdIn_d", -999);
    setRecoVars("sdOut_alphaOut", -999);
    setRecoVars("sdOut_d", -999);
    setRecoVars("sdlSlope", -999);
    setRecoVars("sinAlphaMax", -999);
    setRecoVars("strip2SZpitch", -999);
    setRecoVars("zGeom", -999);
    setRecoVars("zIn", -999);
    setRecoVars("zLo", -999);
    setRecoVars("zOut", -999);

    const float deltaZLum = 15.f;
    const float pixelPSZpitch = 0.15;
    const float strip2SZpitch = 5.0;
    const float zGeom =
        ((innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).moduleType() == SDL::Module::PS ? pixelPSZpitch : strip2SZpitch)
        +
        ((outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).moduleType() == SDL::Module::PS ? pixelPSZpitch : strip2SZpitch);
    const float rtIn = innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->rt();
    const float rtOut = outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->rt();
    const float zIn = innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->z();
    const float zOut = outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->z();
    const float rtOut_o_rtIn = rtOut / rtIn;
    const float kRinv1GeVf = (2.99792458e-3 * 3.8);
    const float k2Rinv1GeVf = kRinv1GeVf / 2.;
    const float ptCut = PTCUT;
    const float sinAlphaMax = 0.95;
    const float sdlSlope = std::asin(std::min(rtOut * k2Rinv1GeVf / ptCut, sinAlphaMax));
    const float dzDrtScale = tan(sdlSlope) / sdlSlope;//FIXME: need approximate value
    const float zLo = zIn + (zIn - deltaZLum) * (rtOut_o_rtIn - 1.f) * (zIn > 0.f ? 1.f : dzDrtScale) - zGeom; //slope-correction only on outer end

    // Cut #0: Preliminary
    // (Only here in endcap case)
    // https://github.com/slava77/cms-tkph2-ntuple/blob/superDoubletLinked-91X-noMock/doubletAnalysis.C#L3631-L3633
    if (not (zIn * zOut > 0))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #0 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " zIn: " << zIn <<  std::endl;
            SDL::cout <<  " zOut: " << zOut <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_TLAlgo);
        return;
    }


    const float dLum = std::copysign(deltaZLum, zIn);
    const bool isOutSgInnerMDPS = (outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).moduleType() == SDL::Module::PS;
    const bool isInSgInnerMDPS = (innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).moduleType() == SDL::Module::PS;
    // https://github.com/slava77/cms-tkph2-ntuple/blob/superDoubletLinked-91X-noMock/doubletAnalysis.C#L3670-L3674
    // we're in mockMode == 3
    const float rtGeom = (isInSgInnerMDPS && isOutSgInnerMDPS ? 2.f * pixelPSZpitch
                 : (isInSgInnerMDPS || isOutSgInnerMDPS ) ? (pixelPSZpitch + strip2SZpitch)
                            : 2.f * strip2SZpitch);
    const float rtGeom1 = isOutSgInnerMDPS ? pixelPSZpitch : strip2SZpitch;//FIXME: make this chosen by configuration for lay11,12 full PS
    const float zGeom1 = std::copysign(zGeom, zIn); //used in B-E region
    const float dz = zOut - zIn;
    const float rtLo = rtIn * (1.f + dz / (zIn + dLum) / dzDrtScale) - rtGeom; //slope correction only on the lower end

    // Cut #1: rt condition
    // https://github.com/slava77/cms-tkph2-ntuple/blob/superDoubletLinked-91X-noMock/doubletAnalysis.C#L3679
    if (not (rtOut >= rtLo))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #1 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " rtOut: " << rtOut <<  std::endl;
            SDL::cout <<  " rtLo: " << rtLo <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_TLAlgo);
        return;
    }

    const float rtHi = rtIn * (1.f + dz / (zIn - dLum)) + rtGeom;

    // Cut #2: rt condition
    if (not (rtOut >= rtLo and rtOut <= rtHi))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #2 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " rtOut: " << rtOut <<  std::endl;
            SDL::cout <<  " rtLo: " << rtLo <<  std::endl;
            SDL::cout <<  " rtHi: " << rtHi <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_TLAlgo);
        return;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TrackletSelection::deltaZ);

    const bool isInSgOuterMDPS = (innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).moduleType() == SDL::Module::PS;

    const float drOutIn = (rtOut - rtIn);
    const float drtSDIn = innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->rt() - innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->rt();
    const float dzSDIn = innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->z() - innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->z();
    const float dr3SDIn = innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->r3() - innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->r3();

    const float coshEta = dr3SDIn / drtSDIn; //direction estimate
    const float dzOutInAbs = std::abs(zOut - zIn);
    const float multDzDr = dzOutInAbs * coshEta / (coshEta * coshEta - 1.f);

    const float kZ = (zOut - zIn) / dzSDIn;

    const float sdlThetaMulsF = 0.015f * sqrt(0.1f + 0.2 * (rtOut - rtIn) / 50.f);
    const float sdlMuls = sdlThetaMulsF * 3.f / ptCut * 4.f; //will need a better guess than x4?

    float drtErr = pixelPSZpitch * pixelPSZpitch * 2.f / dzSDIn / dzSDIn * dzOutInAbs * dzOutInAbs; //both sides contribute to direction uncertainty
    drtErr += sdlMuls * sdlMuls * multDzDr * multDzDr / 3.f * coshEta * coshEta; //sloppy: relative muls is 1/3 of total muls
    drtErr = sqrt(drtErr);
    const float drtMean = drtSDIn * dzOutInAbs / std::abs(dzSDIn);
    const float rtWindow = drtErr + rtGeom; //
    const float rtLo_point = rtIn + drtMean / dzDrtScale - rtWindow;
    const float rtHi_point = rtIn + drtMean + rtWindow;

    // Cut #3: rt-z pointed
    // https://github.com/slava77/cms-tkph2-ntuple/blob/superDoubletLinked-91X-noMock/doubletAnalysis.C#L3765
    if (isInSgInnerMDPS and isInSgOuterMDPS) // If both PS then we can point
    {
        if (not (kZ >= 0 and rtOut >= rtLo_point and rtOut <= rtHi_point))
        {
            if (logLevel >= SDL::Log_Debug3)
            {
                SDL::cout << "Failed Cut #3 in " << __FUNCTION__ << std::endl;
                SDL::cout <<  " kZ: " << kZ <<  std::endl;
                SDL::cout <<  " rtOut: " << rtOut <<  std::endl;
                SDL::cout <<  " rtLo: " << rtLo <<  std::endl;
                SDL::cout <<  " rtHi: " << rtHi <<  std::endl;
            }
            passAlgo_ &= (0 << SDL::Default_TLAlgo);
            return;
        }
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TrackletSelection::deltaZPointed);

    const float sdlPVoff = 0.1f / rtOut;
    const float sdlCut = sdlSlope + sqrt(sdlMuls * sdlMuls + sdlPVoff*sdlPVoff);
    const MiniDoublet& sdIn_mdOut = (*innerSegmentPtr()->outerMiniDoubletPtr());
    const MiniDoublet& sdOut_mdOut = (*outerSegmentPtr()->outerMiniDoubletPtr());
    const Hit& sdIn_mdOut_hit = (*innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr());
    const Hit& sdOut_mdOut_hit = (*outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr());

    // Cut #4: deltaPhiPos can be tighter
    float deltaPhiPos = sdIn_mdOut_hit.deltaPhi(sdOut_mdOut_hit);
    if (not (std::abs(deltaPhiPos) <= sdlCut) )
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #4 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " deltaPhiPos: " << deltaPhiPos <<  " sdlCut: " << sdlCut <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_TLAlgo);
        return;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TrackletSelection::deltaPhiPos);

    const Hit& sdIn_r3 = (*innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr());
    const Hit& sdOut_r3 = (*outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr());
    Hit midR3;
    midR3 += sdIn_r3;
    midR3 += sdOut_r3;
    midR3 /= 2;
    const float dPhi = midR3.deltaPhi(sdOut_r3 - sdIn_r3);

    // Cut #5: deltaPhiChange
    if (not (std::abs(dPhi) <= sdlCut))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #5 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " dPhi: " << dPhi <<  " sdlCut: " << sdlCut <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_TLAlgo);
        return;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TrackletSelection::slope);

    const Hit& sdOut_mdRef_hit = (*outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr());
    const float sdIn_alpha = innerSegmentPtr()->getDeltaPhiChange();
    const float sdOut_alpha = innerSegmentPtr()->getDeltaPhiChange();
    float sdOut_dPhiPos = outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->deltaPhi((*outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()));
    float sdOut_dPhiChange = outerSegmentPtr()->getDeltaPhiChange();
    float sdOut_dPhiChange_min = outerSegmentPtr()->getDeltaPhiMinChange();
    float sdOut_dPhiChange_max = outerSegmentPtr()->getDeltaPhiMaxChange();
    float sdOut_alphaOutRHmin = SDL::MathUtil::Phi_mpi_pi(sdOut_dPhiChange_min - sdOut_dPhiPos);
    float sdOut_alphaOutRHmax = SDL::MathUtil::Phi_mpi_pi(sdOut_dPhiChange_max - sdOut_dPhiPos);
    float sdOut_alphaOut = SDL::MathUtil::Phi_mpi_pi(sdOut_dPhiChange - sdOut_dPhiPos); // <--- This is for the endcap
    // const float sdOut_alphaOut = sdOut_mdOut_hit.deltaPhi(sdOut_mdOut_hit  - sdOut_mdRef_hit); // <--- this is for barrel DON'T USE FOR ENDCAP
    const Hit& sdOut_mdOut_r3 = (*outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr());
    const Hit dr3 = sdOut_mdOut_r3 - sdIn_r3;
    float betaIn  = sdIn_alpha - sdIn_r3.deltaPhi(dr3);
    float betaOut = -sdOut_alphaOut + sdOut_mdOut_r3.deltaPhi(dr3);
    float sdIn_alphaRHmin = innerSegmentPtr()->getDeltaPhiMinChange(); //TODO: there is something about dPhiPosRHin and mdRef.phiRHin/out, that I didn't fully understand.
    float sdIn_alphaRHmax = innerSegmentPtr()->getDeltaPhiMaxChange(); //TODO: there is something about dPhiPosRHin and mdRef.phiRHin/out, that I didn't fully understand.
    float betaInRHmin = betaIn + sdIn_alphaRHmin - sdIn_alpha;
    float betaInRHmax = betaIn + sdIn_alphaRHmax - sdIn_alpha;
    if (std::abs(betaInRHmin) > std::abs(betaInRHmax)) std::swap(betaInRHmax, betaInRHmin);
    float betaOutRHmin = betaOut - sdOut_alphaOutRHmin + sdOut_alphaOut; // Doesn't seem to do anything...? because alphaOutRHmin == alphaOutRHmax == alphaOut..?
    float betaOutRHmax = betaOut - sdOut_alphaOutRHmax + sdOut_alphaOut; // Doesn't seem to do anything...? because alphaOutRHmin == alphaOutRHmax == alphaOut..?
    if (std::abs(betaOutRHmin) > std::abs(betaOutRHmax)) std::swap(betaOutRHmax, betaOutRHmin);

    setRecoVars("sdIn_alpha", sdIn_alpha);
    setRecoVars("sdOut_alphaOut", sdOut_alphaOut);
    setRecoVars("rawBetaInCorrection", sdIn_r3.deltaPhi(dr3));
    setRecoVars("rawBetaOutCorrection", sdOut_mdOut_r3.deltaPhi(dr3));
    setRecoVars("rawBetaIn", betaIn);
    setRecoVars("rawBetaOut", betaOut);

    const Hit& sdIn_mdRef_hit = (*innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr());

    const float sdIn_dr = (sdIn_mdOut_hit - sdIn_mdRef_hit).rt();
    const float sdIn_d = sdIn_mdOut_hit.rt() - sdIn_mdRef_hit.rt();

    const float dr = dr3.rt();
    //beta upper cuts: 2-strip difference for direction resolution
    const float corrF = 1.f;
    bool pass_betaIn_cut = false;//pixel seeds were already selected
    const float betaIn_cut = (-sdIn_dr * corrF + dr) * k2Rinv1GeVf / ptCut + (0.02f / sdIn_d);
    // const float betaIn_cut = std::asin((-sdIn_dr * corrF + dr) * k2Rinv1GeVf / ptCut) + (0.02f / sdIn_d);
    pass_betaIn_cut = std::abs(betaInRHmin) < betaIn_cut;

    setBetaIn(betaInRHmin);
    setBetaInCut(betaIn_cut);

    // Cut #6: first beta cut
    if (not (pass_betaIn_cut))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #6 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " betaInRHmin: " << betaInRHmin <<  " betaIn_cut: " << betaIn_cut <<  std::endl;
        }
        // passAlgo_ &= (0 << SDL::Default_TLAlgo);
        // return;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TrackletSelection::dAlphaIn);

    //now the actual segment linking magic
    float betaAv = 0.5f * (betaIn + betaOut);
    //pt/k2Rinv1GeVf/2. = R
    //R*sin(betaAv) = pt/k2Rinv1GeVf/2*sin(betaAv) = dr/2 => pt = dr*k2Rinv1GeVf/sin(betaAv);
    float pt_beta = dr * k2Rinv1GeVf / sin(betaAv);

    const float pt_betaMax = 7.0f;

    int lIn = 11; //endcap
    int lOut = 13; //endcap

    int betacormode = 0;

    setRecoVars("betaIn_0th", -999);
    setRecoVars("betaOut_0th", -999);
    setRecoVars("betaAv_0th", -999);
    setRecoVars("betaPt_0th", -999);
    setRecoVars("betaIn_1stCorr", -999);
    setRecoVars("betaOut_1stCorr", -999);
    setRecoVars("dBeta_0th", -999);
    setRecoVars("betaIn_1st", -999);
    setRecoVars("betaOut_1st", -999);
    setRecoVars("betaAv_1st", -999);
    setRecoVars("betaPt_1st", -999);
    setRecoVars("betaIn_2ndCorr", -999);
    setRecoVars("betaOut_2ndCorr", -999);
    setRecoVars("dBeta_1st", -999);
    setRecoVars("betaIn_2nd", -999);
    setRecoVars("betaOut_2nd", -999);
    setRecoVars("betaAv_2nd", -999);
    setRecoVars("betaPt_2nd", -999);
    setRecoVars("betaIn_3rdCorr", -999);
    setRecoVars("betaOut_3rdCorr", -999);
    setRecoVars("dBeta_2nd", -999);
    setRecoVars("betaIn_3rd", -999);
    setRecoVars("betaOut_3rd", -999);
    setRecoVars("betaAv_3rd", -999);
    setRecoVars("betaPt_3rd", -999);
    setRecoVars("dBeta_3rd", -999);
    setRecoVars("betaIn_4th", -999);
    setRecoVars("betaOut_4th", -999);
    setRecoVars("betaAv_4th", -999);
    setRecoVars("betaPt_4th", -999);
    setRecoVars("dBeta_4th", -999);

    const float sdOut_dr = (sdOut_mdOut_hit - sdOut_mdRef_hit).rt();
    const float sdOut_d = sdOut_mdOut_hit.rt() - sdOut_mdRef_hit.rt();
    const float diffDr = std::abs(sdIn_dr - sdOut_dr) / std::abs(sdIn_dr + sdOut_dr);

    runDeltaBetaIterations(betaIn, betaOut, betaAv, pt_beta, sdIn_dr, sdOut_dr, dr, lIn, pt_betaMax);

    //if (true //do it for all//diffDr > 0.05 //only if segment length is different significantly
    //        && betaIn * betaOut > 0.f
    //        && (std::abs(pt_beta) < 4.f * pt_betaMax
    //            || (lIn >= 11 && std::abs(pt_beta) < 8.f * pt_betaMax)))   //and the pt_beta is well-defined; less strict for endcap-endcap
    //{
    //    betacormode = 1;

    //    // runDeltaBetaIterations(betaIn, betaOut, betaAv, pt_beta, sdIn_dr, sdOut_dr, dr);

    //    setRecoVars("betaIn_0th", betaIn);
    //    setRecoVars("betaOut_0th", betaOut);
    //    setRecoVars("betaAv_0th", betaAv);
    //    setRecoVars("betaPt_0th", pt_beta);
    //    setRecoVars("betaIn_1stCorr", copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaIn));
    //    setRecoVars("betaOut_1stCorr", copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaOut));
    //    setRecoVars("dBeta_0th", betaIn - betaOut);

    //    const float betaInUpd  = betaIn + copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaIn); //FIXME: need a faster version
    //    const float betaOutUpd = betaOut + copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaOut); //FIXME: need a faster version
    //    betaAv = 0.5f * (betaInUpd + betaOutUpd);
    //    pt_beta = dr * k2Rinv1GeVf / sin(betaAv); //get a better pt estimate

    //    setRecoVars("betaIn_1st", betaInUpd);
    //    setRecoVars("betaOut_1st", betaOutUpd);
    //    setRecoVars("betaAv_1st", betaAv);
    //    setRecoVars("betaPt_1st", pt_beta);
    //    setRecoVars("betaIn_2ndCorr", copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaIn));
    //    setRecoVars("betaOut_2ndCorr", copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaOut));
    //    setRecoVars("dBeta_1st", betaInUpd - betaOutUpd);

    //    betaIn  += copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaIn); //FIXME: need a faster version
    //    betaOut += copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaOut); //FIXME: need a faster version
    //    //update the av and pt
    //    betaAv = 0.5f * (betaIn + betaOut);
    //    pt_beta = dr * k2Rinv1GeVf / sin(betaAv); //get a better pt estimate

    //    setRecoVars("betaIn_2nd", betaIn);
    //    setRecoVars("betaOut_2nd", betaOut);
    //    setRecoVars("betaAv_2nd", betaAv);
    //    setRecoVars("betaPt_2nd", pt_beta);
    //    setRecoVars("betaIn_3rdCorr", copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaIn));
    //    setRecoVars("betaOut_3rdCorr", copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaOut));
    //    setRecoVars("dBeta_2nd", betaIn - betaOut);

    //    setRecoVars("betaIn_3rd", getRecoVar("rawBetaIn") + copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaIn));
    //    setRecoVars("betaOut_3rd", getRecoVar("rawBetaOut") + copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaOut));
    //    setRecoVars("betaAv_3rd", 0.5f * (getRecoVar("betaIn_3rd") + getRecoVar("betaOut_3rd")));
    //    setRecoVars("betaPt_3rd", dr * k2Rinv1GeVf / sin(getRecoVar("betaAv_3rd")));
    //    setRecoVars("dBeta_3rd", getRecoVar("betaIn_3rd") - getRecoVar("betaOut_3rd"));

    //    setRecoVars("betaIn_4th", getRecoVar("rawBetaIn") + copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(getRecoVar("betaPt_3rd")), sinAlphaMax)), getRecoVar("betaIn_3rd")));
    //    setRecoVars("betaOut_4th", getRecoVar("rawBetaOut") + copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(getRecoVar("betaPt_3rd")), sinAlphaMax)), getRecoVar("betaOut_3rd")));
    //    setRecoVars("betaAv_4th", 0.5f * (getRecoVar("betaIn_4th") + getRecoVar("betaOut_4th")));
    //    setRecoVars("betaPt_4th", dr * k2Rinv1GeVf / sin(getRecoVar("betaAv_4th")));
    //    setRecoVars("dBeta_4th", getRecoVar("betaIn_4th") - getRecoVar("betaOut_4th"));

    //}
    //else if (lIn < 11 && std::abs(betaOut) < 0.2 * std::abs(betaIn) && std::abs(pt_beta) < 12.f * pt_betaMax)   //use betaIn sign as ref
    //{
    //    betacormode = 2;
    //    const float pt_betaIn = dr * k2Rinv1GeVf / sin(betaIn);
    //    const float betaInUpd  = betaIn + copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(pt_betaIn), sinAlphaMax)), betaIn); //FIXME: need a faster version
    //    const float betaOutUpd = betaOut + copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(pt_betaIn), sinAlphaMax)), betaIn); //FIXME: need a faster version
    //    betaAv = (std::abs(betaOut) > 0.2f * std::abs(betaIn)) ? (0.5f * (betaInUpd + betaOutUpd)) : betaInUpd;
    //    pt_beta = dr * k2Rinv1GeVf / sin(betaAv); //get a better pt estimate
    //    betaIn  += copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaIn); //FIXME: need a faster version
    //    betaOut += copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaIn); //FIXME: need a faster version
    //    //update the av and pt
    //    betaAv = 0.5f * (betaIn + betaOut);
    //    pt_beta = dr * k2Rinv1GeVf / sin(betaAv); //get a better pt estimate

    //}
    //else
    //{
    //    betacormode = 3;
    //}

    //rescale the ranges proportionally
    const float betaInMMSF = (std::abs(betaInRHmin + betaInRHmax) > 0) ? (2.f * betaIn / std::abs(betaInRHmin + betaInRHmax)) : 0.; //mean value of min,max is the old betaIn
    const float betaOutMMSF = (std::abs(betaOutRHmin + betaOutRHmax) > 0) ? (2.f * betaOut / std::abs(betaOutRHmin + betaOutRHmax)) : 0.;
    betaInRHmin *= betaInMMSF;
    betaInRHmax *= betaInMMSF;
    betaOutRHmin *= betaOutMMSF;
    betaOutRHmax *= betaOutMMSF;

    const float dBetaMuls = sdlThetaMulsF * 4.f / std::min(std::abs(pt_beta), pt_betaMax); //need to confirm the range-out value of 7 GeV

    //regularize to alpha of pt_betaMax .. review may want to add resolution
    const float sdIn_rt = sdIn_mdRef_hit.rt();
    const float sdOut_rt = sdOut_mdRef_hit.rt();
    const float sdIn_z = sdIn_mdRef_hit.z();
    const float sdOut_z = sdOut_mdRef_hit.z();
    const float alphaInAbsReg = std::max(std::abs(sdIn_alpha), std::asin(std::min(sdIn_rt * k2Rinv1GeVf / 3.0f, sinAlphaMax)));
    const float alphaOutAbsReg = std::max(std::abs(sdOut_alpha), std::asin(std::min(sdOut_rt * k2Rinv1GeVf / 3.0f, sinAlphaMax)));
    const float dBetaInLum = lIn < 11 ? 0.0f : std::abs(alphaInAbsReg*deltaZLum / sdIn_z);
    const float dBetaOutLum = lOut < 11 ? 0.0f : std::abs(alphaOutAbsReg*deltaZLum / sdOut_z);
    const float dBetaLum2 = (dBetaInLum + dBetaOutLum) * (dBetaInLum + dBetaOutLum);

    const float sinDPhi = std::sin(dPhi);
    // const float dBetaRIn2 = std::pow((sdIn.mdRef.rtRHout - sdIn.mdRef.rtRHin) * sinDPhi / dr, 2); //TODO-RH: Ask Slava about this rtRHout? rtRHin?
    // const float dBetaROut2 = std::pow((sdOut.mdOut.rtRHout - sdOut.mdOut.rtRHin) * sinDPhi / dr, 2); //TODO-RH
    const float dBetaRIn2 = 0; // TODO-RH
    const float dBetaROut2 = 0; // TODO-RH

    const float betaOut_cut = std::asin(std::min(dr*k2Rinv1GeVf / ptCut, sinAlphaMax)) //FIXME: need faster version
        + (0.02f / sdOut_d) + sqrt(dBetaLum2 + dBetaMuls*dBetaMuls);
    // const float betaOut_cut = std::min(0.01, std::asin(std::min(dr*k2Rinv1GeVf / ptCut, sinAlphaMax)) + (0.02f / sdOut_d) + sqrt(dBetaLum2 + dBetaMuls*dBetaMuls));

    setBetaOut(betaOut);
    setBetaOutCut(betaOut_cut);

    // Cut #7: The real beta cut
    if (not (std::abs(betaOut) < betaOut_cut))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #7 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " betaOut: " << betaOut <<  " betaOut_cut: " << betaOut_cut <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_TLAlgo);
        return;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TrackletSelection::dAlphaOut);

    float pt_betaIn = dr * k2Rinv1GeVf / sin(betaIn);
    const float pt_betaOut = dr * k2Rinv1GeVf / sin(betaOut);
    const float dBetaRes = 0.02f / std::min(sdOut_d, sdIn_d);
    const float dBetaCut2 = (dBetaRes*dBetaRes * 2.0f + dBetaMuls * dBetaMuls + dBetaLum2 + dBetaRIn2 + dBetaROut2
            + 0.25 * std::pow(std::abs(betaInRHmin - betaInRHmax) + std::abs(betaOutRHmin - betaOutRHmax), 2));
    float dBeta = betaIn - betaOut;
    // const float dZeta = sdIn.zeta - sdOut.zeta;

    const float innerSgInnerMdDetId = (innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).detId();
    const float outerSgInnerMdDetId = (outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).detId();
    const float innerSgOuterMdDetId = (innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).detId();
    const float outerSgOuterMdDetId = (outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).detId();

    setRecoVars("hit1_x", innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->x());
    setRecoVars("hit1_y", innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->y());
    setRecoVars("hit2_x", innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->x());
    setRecoVars("hit2_y", innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->y());
    setRecoVars("hit3_x", outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->x());
    setRecoVars("hit3_y", outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->y());
    setRecoVars("hit4_x", outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->x());
    setRecoVars("hit4_y", outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->y());


    std::function<float()> dBeta_midPoint = [&]()
        {

            float hit1_x = innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->x();
            float hit1_y = innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->y();
            float hit2_x = innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->x();
            float hit2_y = innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->y();
            float hit3_x = outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->x();
            float hit3_y = outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->y();
            float hit4_x = outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->x();
            float hit4_y = outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->y();

            float innerSgMidX = (hit1_x + hit2_x) / 2.;
            float innerSgMidY = (hit1_y + hit2_y) / 2.;
            float outerSgMidX = (hit3_x + hit4_x) / 2.;
            float outerSgMidY = (hit3_y + hit4_y) / 2.;

            float vecA_x = hit2_x - innerSgMidX;
            float vecA_y = hit2_y - innerSgMidY;
            float vecB_x = outerSgMidX - innerSgMidX;
            float vecB_y = outerSgMidY - innerSgMidY;
            float vecC_x = hit4_x - outerSgMidX;
            float vecC_y = hit4_y - outerSgMidY;
            float vecA_mag = sqrt(vecA_x * vecA_x + vecA_y * vecA_y);
            float vecB_mag = sqrt(vecB_x * vecB_x + vecB_y * vecB_y);
            float vecC_mag = sqrt(vecC_x * vecC_x + vecC_y * vecC_y);

            float vecA_dot_vecB = vecA_x * vecB_x + vecA_y * vecB_y;
            float vecB_dot_vecC = vecB_x * vecC_x + vecB_y * vecC_y;

            float angle_AB = std::acos(vecA_dot_vecB / vecA_mag / vecB_mag);
            float angle_BC = std::acos(vecB_dot_vecC / vecB_mag / vecC_mag);

            return angle_AB - angle_BC;

        };

    setRecoVars("sinAlphaMax", sinAlphaMax);
    setRecoVars("betaIn", betaIn);
    setRecoVars("betaInRHmax", betaInRHmax);
    setRecoVars("betaInRHmin", betaInRHmin);
    setRecoVars("betaOut", betaOut);
    setRecoVars("betaOutRHmax", betaOutRHmax);
    setRecoVars("betaOutRHmin", betaOutRHmin);
    setRecoVars("dBeta", dBeta);
    setRecoVars("dBetaCut2", dBetaCut2);
    setRecoVars("dBetaLum2", dBetaLum2);
    setRecoVars("dBetaMuls", dBetaMuls);
    setRecoVars("dBetaRIn2", dBetaRIn2);
    setRecoVars("dBetaROut2", dBetaROut2);
    setRecoVars("dBetaRes", dBetaRes);
    setRecoVars("deltaZLum", deltaZLum);
    setRecoVars("dr", dr);
    setRecoVars("dzDrtScale", dzDrtScale);
    setRecoVars("innerSgInnerMdDetId", innerSgInnerMdDetId);
    setRecoVars("innerSgOuterMdDetId", innerSgOuterMdDetId);
    setRecoVars("k2Rinv1GeVf", k2Rinv1GeVf);
    setRecoVars("kRinv1GeVf", kRinv1GeVf);
    setRecoVars("outerSgInnerMdDetId", outerSgInnerMdDetId);
    setRecoVars("outerSgOuterMdDetId", outerSgOuterMdDetId);
    setRecoVars("pixelPSZpitch", pixelPSZpitch);
    setRecoVars("ptCut", ptCut);
    setRecoVars("pt_betaIn", pt_betaIn);
    setRecoVars("pt_betaOut", pt_betaOut);
    setRecoVars("rtIn", rtIn);
    setRecoVars("rtOut", rtOut);
    setRecoVars("rtOut_o_rtIn", rtOut_o_rtIn);
    setRecoVars("sdIn_d", sdIn_d);
    setRecoVars("sdOut_d", sdOut_d);
    setRecoVars("sdlSlope", sdlSlope);
    setRecoVars("strip2SZpitch", strip2SZpitch);
    setRecoVars("zGeom", zGeom);
    setRecoVars("zIn", zIn);
    setRecoVars("zLo", zLo);
    setRecoVars("zOut", zOut);
    setRecoVars("betacormode", betacormode);
    setRecoVars("pt_beta", pt_beta);
    setRecoVars("betaAv", betaAv);
    setRecoVars("dBeta_midPoint", dBeta_midPoint());

    // dBeta = getRecoVar("dBeta_4th");

    setDeltaBeta(dBeta);
    setDeltaBetaCut(std::sqrt(dBetaCut2));

    if (not (dBeta * dBeta <= dBetaCut2))
    // if (not (dBeta * dBeta <= 0.004*0.004))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #8 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " dBeta*dBeta: " << dBeta*dBeta <<  " dBetaCut2: " << dBetaCut2 <<  std::endl;
            SDL::cout <<  " dBetaRes: " << dBetaRes <<  " dBetaMuls: " << dBetaMuls <<  " dBetaLum2: " << dBetaLum2 <<  std::endl;
            SDL::cout <<  " dBetaRIn2: " << dBetaRIn2 <<  " dBetaROut2: " << dBetaROut2 <<  std::endl;
            SDL::cout <<  " betaInRHmin: " << betaInRHmin <<  " betaInRHmax: " << betaInRHmax <<  std::endl;
            SDL::cout <<  " betaOutRHmin: " << betaOutRHmin <<  " betaOutRHmax: " << betaOutRHmax <<  std::endl;
        }
        if (getNm1DeltaBetaCut())
        {
            passAlgo_ |= (1 << SDL::Default_TLAlgo);
        }
        else
        {
            passAlgo_ &= (0 << SDL::Default_TLAlgo);
            return;
        }
    }
    else if (logLevel >= SDL::Log_Debug3)
    {
        SDL::cout << "Passed Cut #8 in " << __FUNCTION__ << std::endl;
        SDL::cout <<  " dBeta*dBeta: " << dBeta*dBeta <<  " dBetaCut2: " << dBetaCut2 <<  std::endl;
        SDL::cout <<  " dBetaRes: " << dBetaRes <<  " dBetaMuls: " << dBetaMuls <<  " dBetaLum2: " << dBetaLum2 <<  std::endl;
        SDL::cout <<  " dBetaRIn2: " << dBetaRIn2 <<  " dBetaROut2: " << dBetaROut2 <<  std::endl;
        SDL::cout <<  " betaInRHmin: " << betaInRHmin <<  " betaInRHmax: " << betaInRHmax <<  std::endl;
        SDL::cout <<  " betaOutRHmin: " << betaOutRHmin <<  " betaOutRHmax: " << betaOutRHmax <<  std::endl;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TrackletSelection::dBeta);



    // It has passed everything
    passAlgo_ |= (1 << SDL::Default_TLAlgo);
    if (getNm1DeltaBetaCut()) passAlgo_ |= (1 << SDL::DefaultNm1_TLAlgo);
    return;

}

void SDL::Tracklet::runTrackletDefaultAlgoDeltaBetaOnlyBarrelBarrelBarrelBarrel(SDL::LogLevel logLevel)
{
    const float deltaZLum = 15.f;
    const float pixelPSZpitch = 0.15;
    const float strip2SZpitch = 5.0;
    const float zGeom =
        ((innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).moduleType() == SDL::Module::PS ? pixelPSZpitch : strip2SZpitch)
        +
        ((outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).moduleType() == SDL::Module::PS ? pixelPSZpitch : strip2SZpitch);
    const float rtIn = innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->rt();
    const float rtOut = outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->rt();
    const float zIn = innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->z();
    const float zOut = outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->z();
    const float rtOut_o_rtIn = rtOut / rtIn;
    const float kRinv1GeVf = (2.99792458e-3 * 3.8);
    const float k2Rinv1GeVf = kRinv1GeVf / 2.;
    const float ptCut = PTCUT;
    const float sinAlphaMax = 0.95;
    const float sdlSlope = std::asin(std::min(rtOut * k2Rinv1GeVf / ptCut, sinAlphaMax));
    const float dzDrtScale = tan(sdlSlope) / sdlSlope;//FIXME: need approximate value
    const float zLo = zIn + (zIn - deltaZLum) * (rtOut_o_rtIn - 1.f) * (zIn > 0.f ? 1.f : dzDrtScale) - zGeom; //slope-correction only on outer end

    // Reset passBitsDefaultAlgo_;
    passBitsDefaultAlgo_ = 0;

    // Cut #1: Z compatibility
    if (not (zOut >= zLo))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #1 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " zLo: " << zLo <<  " zOut: " << zOut <<  std::endl;
            SDL::cout <<  " zIn: " << zIn <<  std::endl;
            SDL::cout <<  " deltaZLum: " << deltaZLum <<  std::endl;
            SDL::cout <<  " rtOut_o_rtIn: " << rtOut_o_rtIn <<  std::endl;
            SDL::cout <<  " dzDrtScale: " << dzDrtScale <<  std::endl;
            SDL::cout <<  " zGeom: " << zGeom <<  std::endl;
        }
        // passAlgo_ &= (0 << SDL::Default_TLAlgo);
        // return;
    }

    const float zHi = zIn + (zIn + deltaZLum) * (rtOut_o_rtIn - 1.f) * (zIn < 0.f ? 1.f : dzDrtScale) + zGeom;

    // Cut #2: Z compatibility
    if (not (zOut >= zLo and zOut <= zHi))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #2 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " zLo: " << zLo <<  " zOut: " << zOut <<  " zHi: " << zHi <<  std::endl;
        }
        // passAlgo_ &= (0 << SDL::Default_TLAlgo);
        // return;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TrackletSelection::deltaZ);

    const float drOutIn = (rtOut - rtIn);

    const float rtInvIn = 1. / rtIn;
    // const float ptIn = sdIn.p3.Pt(); // For Pixel Seeds
    const float rIn = sqrt(zIn*zIn + rtIn*rtIn);
    const float drtSDIn = innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->rt() - innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->rt();
    const float dzSDIn = innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->z() - innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->z();
    const float dr3SDIn = innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->r3() - innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->r3();

    const float coshEta = dr3SDIn / drtSDIn; //direction estimate
    float dzErr = zGeom * zGeom * 2.f; //both sides contribute to direction uncertainty
    //FIXME (later) more realistic accounting of material effects is needed
    const float sdlThetaMulsF = 0.015f * sqrt(0.1f + 0.2 * (rtOut - rtIn) / 50.f) * sqrt(rIn / rtIn);
    const float sdlMuls = sdlThetaMulsF * 3.f / ptCut * 4.f; //will need a better guess than x4?
    dzErr += sdlMuls * sdlMuls * drOutIn * drOutIn / 3.f * coshEta * coshEta; //sloppy
    dzErr = sqrt(dzErr);
    const float dzMean = dzSDIn / drtSDIn * drOutIn;
    const float zWindow = dzErr / drtSDIn * drOutIn + zGeom; //FIXME for ptCut lower than ~0.8 need to add curv path correction
    const float zLoPointed = zIn + dzMean * (zIn > 0.f ? 1.f : dzDrtScale) - zWindow;
    const float zHiPointed = zIn + dzMean * (zIn < 0.f ? 1.f : dzDrtScale) + zWindow;

    // Cut #3: Pointed Z
    if (not (zOut >= zLoPointed and zOut <= zHiPointed))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #3 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " zLoPointed: " << zLoPointed <<  " zOut: " << zOut <<  " zHiPointed: " << zHiPointed <<  std::endl;
        }
        // passAlgo_ &= (0 << SDL::Default_TLAlgo);
        // return;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TrackletSelection::deltaZPointed);

    const float sdlPVoff = 0.1f / rtOut;
    const float sdlCut = sdlSlope + sqrt(sdlMuls * sdlMuls + sdlPVoff*sdlPVoff);
    const MiniDoublet& sdIn_mdOut = (*innerSegmentPtr()->outerMiniDoubletPtr());
    const MiniDoublet& sdOut_mdOut = (*outerSegmentPtr()->outerMiniDoubletPtr());
    const Hit& sdIn_mdOut_hit = (*innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr());
    const Hit& sdOut_mdOut_hit = (*outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr());

    // Cut #4: deltaPhiPos can be tighter
    float deltaPhiPos = sdIn_mdOut_hit.deltaPhi(sdOut_mdOut_hit);
    if (not (std::abs(deltaPhiPos) <= sdlCut) )
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #4 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " deltaPhiPos: " << deltaPhiPos <<  " sdlCut: " << sdlCut <<  std::endl;
        }
        // passAlgo_ &= (0 << SDL::Default_TLAlgo);
        // return;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TrackletSelection::deltaPhiPos);

    const Hit& sdIn_r3 = (*innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr());
    const Hit& sdOut_r3 = (*outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr());
    Hit midR3;
    midR3 += sdIn_r3;
    midR3 += sdOut_r3;
    midR3 /= 2;
    const float dPhi = midR3.deltaPhi(sdOut_r3 - sdIn_r3);

    // Cut #5: deltaPhiChange
    if (not (std::abs(dPhi) <= sdlCut))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #5 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " dPhi: " << dPhi <<  " sdlCut: " << sdlCut <<  std::endl;
        }
        // passAlgo_ &= (0 << SDL::Default_TLAlgo);
        // return;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TrackletSelection::slope);

    const Hit& sdOut_mdRef_hit = (*outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr());
    const float sdIn_alpha = innerSegmentPtr()->getDeltaPhiChange();
    const float sdOut_alpha = innerSegmentPtr()->getDeltaPhiChange();
    const float sdOut_alphaOut = sdOut_mdOut_hit.deltaPhi(sdOut_mdOut_hit  - sdOut_mdRef_hit);
    const Hit& sdOut_mdOut_r3 = (*outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr());
    const Hit dr3 = sdOut_mdOut_r3 - sdIn_r3;
    float betaIn  = sdIn_alpha - sdIn_r3.deltaPhi(dr3);
    float betaOut = -sdOut_alphaOut + sdOut_mdOut_r3.deltaPhi(dr3);
    float betaInRHmin = betaIn;
    float betaInRHmax = betaIn;
    float betaOutRHmin = betaOut;
    float betaOutRHmax = betaOut;

    setRecoVars("sdIn_alpha", sdIn_alpha);
    setRecoVars("sdOut_alphaOut", sdOut_alphaOut);
    setRecoVars("rawBetaInCorrection", sdIn_r3.deltaPhi(dr3));
    setRecoVars("rawBetaOutCorrection", sdOut_mdOut_r3.deltaPhi(dr3));
    setRecoVars("rawBetaIn", betaIn);
    setRecoVars("rawBetaOut", betaOut);

    const Hit& sdIn_mdRef_hit = (*innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr());

    const float sdIn_dr = (sdIn_mdOut_hit - sdIn_mdRef_hit).rt();
    const float sdIn_d = sdIn_mdOut_hit.rt() - sdIn_mdRef_hit.rt();

    const float dr = dr3.rt();
    //beta upper cuts: 2-strip difference for direction resolution
    const float corrF = 1.f;
    bool pass_betaIn_cut = false;//pixel seeds were already selected
    const float betaIn_cut = (-sdIn_dr * corrF + dr) * k2Rinv1GeVf / ptCut + (0.02f / sdIn_d);
    // const float betaIn_cut = std::asin((-sdIn_dr * corrF + dr) * k2Rinv1GeVf / ptCut) + (0.02f / sdIn_d);
    pass_betaIn_cut = std::abs(betaInRHmin) < betaIn_cut;

    setBetaIn(betaInRHmin);
    setBetaInCut(betaIn_cut);

    // Cut #6: first beta cut
    if (not (pass_betaIn_cut))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #6 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " betaInRHmin: " << betaInRHmin <<  " betaIn_cut: " << betaIn_cut <<  std::endl;
        }
        // passAlgo_ &= (0 << SDL::Default_TLAlgo);
        // return;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TrackletSelection::dAlphaIn);

    //now the actual segment linking magic
    float betaAv = 0.5f * (betaIn + betaOut);
    //pt/k2Rinv1GeVf/2. = R
    //R*sin(betaAv) = pt/k2Rinv1GeVf/2*sin(betaAv) = dr/2 => pt = dr*k2Rinv1GeVf/sin(betaAv);
    float pt_beta = dr * k2Rinv1GeVf / sin(betaAv);

    const float pt_betaMax = 7.0f;

    int lIn = 5;
    int lOut = 5;

    int betacormode = 0;

    setRecoVars("betaIn_0th", -999);
    setRecoVars("betaOut_0th", -999);
    setRecoVars("betaAv_0th", -999);
    setRecoVars("betaPt_0th", -999);
    setRecoVars("betaIn_1stCorr", -999);
    setRecoVars("betaOut_1stCorr", -999);
    setRecoVars("dBeta_0th", -999);
    setRecoVars("betaIn_1st", -999);
    setRecoVars("betaOut_1st", -999);
    setRecoVars("betaAv_1st", -999);
    setRecoVars("betaPt_1st", -999);
    setRecoVars("betaIn_2ndCorr", -999);
    setRecoVars("betaOut_2ndCorr", -999);
    setRecoVars("dBeta_1st", -999);
    setRecoVars("betaIn_2nd", -999);
    setRecoVars("betaOut_2nd", -999);
    setRecoVars("betaAv_2nd", -999);
    setRecoVars("betaPt_2nd", -999);
    setRecoVars("betaIn_3rdCorr", -999);
    setRecoVars("betaOut_3rdCorr", -999);
    setRecoVars("dBeta_2nd", -999);
    setRecoVars("betaIn_3rd", -999);
    setRecoVars("betaOut_3rd", -999);
    setRecoVars("betaAv_3rd", -999);
    setRecoVars("betaPt_3rd", -999);
    setRecoVars("dBeta_3rd", -999);
    setRecoVars("betaIn_4th", -999);
    setRecoVars("betaOut_4th", -999);
    setRecoVars("betaAv_4th", -999);
    setRecoVars("betaPt_4th", -999);
    setRecoVars("dBeta_4th", -999);

    const float sdOut_dr = (sdOut_mdOut_hit - sdOut_mdRef_hit).rt();
    const float sdOut_d = sdOut_mdOut_hit.rt() - sdOut_mdRef_hit.rt();
    const float diffDr = std::abs(sdIn_dr - sdOut_dr) / std::abs(sdIn_dr + sdOut_dr);
    if (true //do it for all//diffDr > 0.05 //only if segment length is different significantly
            && betaIn * betaOut > 0.f
            && (std::abs(pt_beta) < 4.f * pt_betaMax
                || (lIn >= 11 && std::abs(pt_beta) < 8.f * pt_betaMax)))   //and the pt_beta is well-defined; less strict for endcap-endcap
    {
        betacormode = 1;

        // if (isEC_lastLayer) pt_beta = dr * k2Rinv1GeVf / sin(betaIn);

        setRecoVars("betaIn_0th", betaIn);
        setRecoVars("betaOut_0th", betaOut);
        setRecoVars("betaAv_0th", betaAv);
        setRecoVars("betaPt_0th", pt_beta);
        setRecoVars("betaIn_1stCorr", copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaIn));
        setRecoVars("betaOut_1stCorr", copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaOut));
        setRecoVars("dBeta_0th", betaIn - betaOut);

        const float betaInUpd  = betaIn + copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaIn); //FIXME: need a faster version
        const float betaOutUpd = betaOut + copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaOut); //FIXME: need a faster version
        betaAv = 0.5f * (betaInUpd + betaOutUpd);
        pt_beta = dr * k2Rinv1GeVf / sin(betaAv); //get a better pt estimate

        setRecoVars("betaIn_1st", betaInUpd);
        setRecoVars("betaOut_1st", betaOutUpd);
        setRecoVars("betaAv_1st", betaAv);
        setRecoVars("betaPt_1st", pt_beta);
        setRecoVars("betaIn_2ndCorr", copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaIn));
        setRecoVars("betaOut_2ndCorr", copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaOut));
        setRecoVars("dBeta_1st", betaInUpd - betaOutUpd);

        betaIn  += copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaIn); //FIXME: need a faster version
        betaOut += copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaOut); //FIXME: need a faster version
        //update the av and pt
        betaAv = 0.5f * (betaIn + betaOut);
        pt_beta = dr * k2Rinv1GeVf / sin(betaAv); //get a better pt estimate

        setRecoVars("betaIn_2nd", betaIn);
        setRecoVars("betaOut_2nd", betaOut);
        setRecoVars("betaAv_2nd", betaAv);
        setRecoVars("betaPt_2nd", pt_beta);
        setRecoVars("betaIn_3rdCorr", copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaIn));
        setRecoVars("betaOut_3rdCorr", copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaOut));
        setRecoVars("dBeta_2nd", betaIn - betaOut);

        setRecoVars("betaIn_3rd", getRecoVar("rawBetaIn") + copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaIn));
        setRecoVars("betaOut_3rd", getRecoVar("rawBetaOut") + copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaOut));
        setRecoVars("betaAv_3rd", 0.5f * (getRecoVar("betaIn_3rd") + getRecoVar("betaOut_3rd")));
        setRecoVars("betaPt_3rd", dr * k2Rinv1GeVf / sin(getRecoVar("betaAv_3rd")));
        setRecoVars("dBeta_3rd", getRecoVar("betaIn_3rd") - getRecoVar("betaOut_3rd"));

        setRecoVars("betaIn_4th", getRecoVar("rawBetaIn") + copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(getRecoVar("betaPt_3rd")), sinAlphaMax)), getRecoVar("betaIn_3rd")));
        setRecoVars("betaOut_4th", getRecoVar("rawBetaOut") + copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(getRecoVar("betaPt_3rd")), sinAlphaMax)), getRecoVar("betaOut_3rd")));
        setRecoVars("betaAv_4th", 0.5f * (getRecoVar("betaIn_4th") + getRecoVar("betaOut_4th")));
        setRecoVars("betaPt_4th", dr * k2Rinv1GeVf / sin(getRecoVar("betaAv_4th")));
        setRecoVars("dBeta_4th", getRecoVar("betaIn_4th") - getRecoVar("betaOut_4th"));

    }
    else if (lIn < 11 && std::abs(betaOut) < 0.2 * std::abs(betaIn) && std::abs(pt_beta) < 12.f * pt_betaMax)   //use betaIn sign as ref
    {
        betacormode = 2;
        const float pt_betaIn = dr * k2Rinv1GeVf / sin(betaIn);
        const float betaInUpd  = betaIn + copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(pt_betaIn), sinAlphaMax)), betaIn); //FIXME: need a faster version
        const float betaOutUpd = betaOut + copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(pt_betaIn), sinAlphaMax)), betaIn); //FIXME: need a faster version
        betaAv = (std::abs(betaOut) > 0.2f * std::abs(betaIn)) ? (0.5f * (betaInUpd + betaOutUpd)) : betaInUpd;
        pt_beta = dr * k2Rinv1GeVf / sin(betaAv); //get a better pt estimate
        betaIn  += copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaIn); //FIXME: need a faster version
        betaOut += copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaIn); //FIXME: need a faster version
        //update the av and pt
        betaAv = 0.5f * (betaIn + betaOut);
        pt_beta = dr * k2Rinv1GeVf / sin(betaAv); //get a better pt estimate

    }
    else
    {
        betacormode = 3;
    }

    //rescale the ranges proportionally
    const float betaInMMSF = (std::abs(betaInRHmin + betaInRHmax) > 0) ? (2.f * betaIn / std::abs(betaInRHmin + betaInRHmax)) : 0.; //mean value of min,max is the old betaIn
    const float betaOutMMSF = (std::abs(betaOutRHmin + betaOutRHmax) > 0) ? (2.f * betaOut / std::abs(betaOutRHmin + betaOutRHmax)) : 0.;
    betaInRHmin *= betaInMMSF;
    betaInRHmax *= betaInMMSF;
    betaOutRHmin *= betaOutMMSF;
    betaOutRHmax *= betaOutMMSF;

    const float dBetaMuls = sdlThetaMulsF * 4.f / std::min(std::abs(pt_beta), pt_betaMax); //need to confirm the range-out value of 7 GeV

    //regularize to alpha of pt_betaMax .. review may want to add resolution
    const float sdIn_rt = sdIn_mdRef_hit.rt();
    const float sdOut_rt = sdOut_mdRef_hit.rt();
    const float sdIn_z = sdIn_mdRef_hit.z();
    const float sdOut_z = sdOut_mdRef_hit.z();
    const float alphaInAbsReg = std::max(std::abs(sdIn_alpha), std::asin(std::min(sdIn_rt * k2Rinv1GeVf / 3.0f, sinAlphaMax)));
    const float alphaOutAbsReg = std::max(std::abs(sdOut_alpha), std::asin(std::min(sdOut_rt * k2Rinv1GeVf / 3.0f, sinAlphaMax)));
    const float dBetaInLum = lIn < 11 ? 0.0f : std::abs(alphaInAbsReg*deltaZLum / sdIn_z);
    const float dBetaOutLum = lOut < 11 ? 0.0f : std::abs(alphaOutAbsReg*deltaZLum / sdOut_z);
    const float dBetaLum2 = (dBetaInLum + dBetaOutLum) * (dBetaInLum + dBetaOutLum);

    const float sinDPhi = std::sin(dPhi);
    // const float dBetaRIn2 = std::pow((sdIn.mdRef.rtRHout - sdIn.mdRef.rtRHin) * sinDPhi / dr, 2); //TODO-RH: Ask Slava about this rtRHout? rtRHin?
    // const float dBetaROut2 = std::pow((sdOut.mdOut.rtRHout - sdOut.mdOut.rtRHin) * sinDPhi / dr, 2); //TODO-RH
    const float dBetaRIn2 = 0; // TODO-RH
    const float dBetaROut2 = 0; // TODO-RH

    const float betaOut_cut = std::asin(std::min(dr*k2Rinv1GeVf / ptCut, sinAlphaMax)) //FIXME: need faster version
        + (0.02f / sdOut_d) + sqrt(dBetaLum2 + dBetaMuls*dBetaMuls);
    // const float betaOut_cut = std::min(0.01, std::asin(std::min(dr*k2Rinv1GeVf / ptCut, sinAlphaMax)) + (0.02f / sdOut_d) + sqrt(dBetaLum2 + dBetaMuls*dBetaMuls));

    setBetaOut(betaOut);
    setBetaOutCut(betaOut_cut);

    // Cut #7: The real beta cut
    if (not (std::abs(betaOut) < betaOut_cut))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #7 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " betaOut: " << betaOut <<  " betaOut_cut: " << betaOut_cut <<  std::endl;
        }
        // passAlgo_ &= (0 << SDL::Default_TLAlgo);
        // return;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TrackletSelection::dAlphaOut);

    float pt_betaIn = dr * k2Rinv1GeVf / sin(betaIn);
    const float pt_betaOut = dr * k2Rinv1GeVf / sin(betaOut);
    const float dBetaRes = 0.02f / std::min(sdOut_d, sdIn_d);
    const float dBetaCut2 = (dBetaRes*dBetaRes * 2.0f + dBetaMuls * dBetaMuls + dBetaLum2 + dBetaRIn2 + dBetaROut2
            + 0.25 * std::pow(std::abs(betaInRHmin - betaInRHmax) + std::abs(betaOutRHmin - betaOutRHmax), 2));
    float dBeta = betaIn - betaOut;
    // const float dZeta = sdIn.zeta - sdOut.zeta;

    const float innerSgInnerMdDetId = (innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).detId();
    const float outerSgInnerMdDetId = (outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).detId();
    const float innerSgOuterMdDetId = (innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).detId();
    const float outerSgOuterMdDetId = (outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).detId();

    setRecoVars("hit1_x", innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->x());
    setRecoVars("hit1_y", innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->y());
    setRecoVars("hit2_x", innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->x());
    setRecoVars("hit2_y", innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->y());
    setRecoVars("hit3_x", outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->x());
    setRecoVars("hit3_y", outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->y());
    setRecoVars("hit4_x", outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->x());
    setRecoVars("hit4_y", outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->y());


    std::function<float()> dBeta_midPoint = [&]()
        {

            float hit1_x = innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->x();
            float hit1_y = innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->y();
            float hit2_x = innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->x();
            float hit2_y = innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->y();
            float hit3_x = outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->x();
            float hit3_y = outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->y();
            float hit4_x = outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->x();
            float hit4_y = outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->y();

            float innerSgMidX = (hit1_x + hit2_x) / 2.;
            float innerSgMidY = (hit1_y + hit2_y) / 2.;
            float outerSgMidX = (hit3_x + hit4_x) / 2.;
            float outerSgMidY = (hit3_y + hit4_y) / 2.;

            float vecA_x = hit2_x - innerSgMidX;
            float vecA_y = hit2_y - innerSgMidY;
            float vecB_x = outerSgMidX - innerSgMidX;
            float vecB_y = outerSgMidY - innerSgMidY;
            float vecC_x = hit4_x - outerSgMidX;
            float vecC_y = hit4_y - outerSgMidY;
            float vecA_mag = sqrt(vecA_x * vecA_x + vecA_y * vecA_y);
            float vecB_mag = sqrt(vecB_x * vecB_x + vecB_y * vecB_y);
            float vecC_mag = sqrt(vecC_x * vecC_x + vecC_y * vecC_y);

            float vecA_dot_vecB = vecA_x * vecB_x + vecA_y * vecB_y;
            float vecB_dot_vecC = vecB_x * vecC_x + vecB_y * vecC_y;

            float angle_AB = std::acos(vecA_dot_vecB / vecA_mag / vecB_mag);
            float angle_BC = std::acos(vecB_dot_vecC / vecB_mag / vecC_mag);

            return angle_AB - angle_BC;

        };

    setRecoVars("sinAlphaMax", sinAlphaMax);
    setRecoVars("betaIn", betaIn);
    setRecoVars("betaInRHmax", betaInRHmax);
    setRecoVars("betaInRHmin", betaInRHmin);
    setRecoVars("betaOut", betaOut);
    setRecoVars("betaOutRHmax", betaOutRHmax);
    setRecoVars("betaOutRHmin", betaOutRHmin);
    setRecoVars("dBeta", dBeta);
    setRecoVars("dBetaCut2", dBetaCut2);
    setRecoVars("dBetaLum2", dBetaLum2);
    setRecoVars("dBetaMuls", dBetaMuls);
    setRecoVars("dBetaRIn2", dBetaRIn2);
    setRecoVars("dBetaROut2", dBetaROut2);
    setRecoVars("dBetaRes", dBetaRes);
    setRecoVars("deltaZLum", deltaZLum);
    setRecoVars("dr", dr);
    setRecoVars("dzDrtScale", dzDrtScale);
    setRecoVars("innerSgInnerMdDetId", innerSgInnerMdDetId);
    setRecoVars("innerSgOuterMdDetId", innerSgOuterMdDetId);
    setRecoVars("k2Rinv1GeVf", k2Rinv1GeVf);
    setRecoVars("kRinv1GeVf", kRinv1GeVf);
    setRecoVars("outerSgInnerMdDetId", outerSgInnerMdDetId);
    setRecoVars("outerSgOuterMdDetId", outerSgOuterMdDetId);
    setRecoVars("pixelPSZpitch", pixelPSZpitch);
    setRecoVars("ptCut", ptCut);
    setRecoVars("pt_betaIn", pt_betaIn);
    setRecoVars("pt_betaOut", pt_betaOut);
    setRecoVars("rtIn", rtIn);
    setRecoVars("rtOut", rtOut);
    setRecoVars("rtOut_o_rtIn", rtOut_o_rtIn);
    setRecoVars("sdIn_d", sdIn_d);
    setRecoVars("sdOut_d", sdOut_d);
    setRecoVars("sdlSlope", sdlSlope);
    setRecoVars("strip2SZpitch", strip2SZpitch);
    setRecoVars("zGeom", zGeom);
    setRecoVars("zIn", zIn);
    setRecoVars("zLo", zLo);
    setRecoVars("zOut", zOut);
    setRecoVars("betacormode", betacormode);
    setRecoVars("pt_beta", pt_beta);
    setRecoVars("betaAv", betaAv);
    setRecoVars("dBeta_midPoint", dBeta_midPoint());

    // dBeta = getRecoVar("dBeta_4th");

    setDeltaBeta(dBeta);
    setDeltaBetaCut(std::sqrt(dBetaCut2));

    if (not (dBeta * dBeta <= dBetaCut2))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #8 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " dBeta*dBeta: " << dBeta*dBeta <<  " dBetaCut2: " << dBetaCut2 <<  std::endl;
            SDL::cout <<  " dBetaRes: " << dBetaRes <<  " dBetaMuls: " << dBetaMuls <<  " dBetaLum2: " << dBetaLum2 <<  std::endl;
            SDL::cout <<  " dBetaRIn2: " << dBetaRIn2 <<  " dBetaROut2: " << dBetaROut2 <<  std::endl;
            SDL::cout <<  " betaInRHmin: " << betaInRHmin <<  " betaInRHmax: " << betaInRHmax <<  std::endl;
            SDL::cout <<  " betaOutRHmin: " << betaOutRHmin <<  " betaOutRHmax: " << betaOutRHmax <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_TLAlgo);
        // // passAlgo_ |= (1 << SDL::Default_TLAlgo);
        return;
    }
    else if (logLevel >= SDL::Log_Debug3)
    {
        SDL::cout << "Passed Cut #8 in " << __FUNCTION__ << std::endl;
        SDL::cout <<  " dBeta*dBeta: " << dBeta*dBeta <<  " dBetaCut2: " << dBetaCut2 <<  std::endl;
        SDL::cout <<  " dBetaRes: " << dBetaRes <<  " dBetaMuls: " << dBetaMuls <<  " dBetaLum2: " << dBetaLum2 <<  std::endl;
        SDL::cout <<  " dBetaRIn2: " << dBetaRIn2 <<  " dBetaROut2: " << dBetaROut2 <<  std::endl;
        SDL::cout <<  " betaInRHmin: " << betaInRHmin <<  " betaInRHmax: " << betaInRHmax <<  std::endl;
        SDL::cout <<  " betaOutRHmin: " << betaOutRHmin <<  " betaOutRHmax: " << betaOutRHmax <<  std::endl;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TrackletSelection::dBeta);

    passAlgo_ |= (1 << SDL::Default_TLAlgo);
    return;
}

void SDL::Tracklet::runTrackletDefaultAlgoBarrelBarrel(SDL::LogLevel logLevel)
{
    return;
}

void SDL::Tracklet::runTrackletDefaultAlgoBarrelEndcap(SDL::LogLevel logLevel)
{
    return;
}

void SDL::Tracklet::runTrackletDefaultAlgoEndcapEndcap(SDL::LogLevel logLevel)
{
    return;
}

void SDL::Tracklet::runDeltaBetaIterations(float& betaIn, float& betaOut, float& betaAv, float& pt_beta, float sdIn_dr, float sdOut_dr, float dr, int lIn, float pt_betaMax)
{
    const float kRinv1GeVf = (2.99792458e-3 * 3.8);
    const float k2Rinv1GeVf = kRinv1GeVf / 2.;
    const float sinAlphaMax = 0.95;

    if (true //do it for all//diffDr > 0.05 //only if segment length is different significantly
            && betaIn * betaOut > 0.f
            && (std::abs(pt_beta) < 4.f * pt_betaMax
                || (lIn >= 11 && std::abs(pt_beta) < 8.f * pt_betaMax)))   //and the pt_beta is well-defined; less strict for endcap-endcap
    {
        setRecoVars("betacormode", 1);

        setRecoVars("betaIn_0th", betaIn);
        setRecoVars("betaOut_0th", betaOut);
        setRecoVars("betaAv_0th", betaAv);
        setRecoVars("betaPt_0th", pt_beta);
        setRecoVars("betaIn_1stCorr", copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaIn));
        setRecoVars("betaOut_1stCorr", copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaOut));
        setRecoVars("dBeta_0th", betaIn - betaOut);

        const float betaInUpd  = betaIn + copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaIn); //FIXME: need a faster version
        const float betaOutUpd = betaOut + copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaOut); //FIXME: need a faster version
        betaAv = 0.5f * (betaInUpd + betaOutUpd);
        pt_beta = dr * k2Rinv1GeVf / sin(betaAv); //get a better pt estimate

        setRecoVars("betaIn_1st", betaInUpd);
        setRecoVars("betaOut_1st", betaOutUpd);
        setRecoVars("betaAv_1st", betaAv);
        setRecoVars("betaPt_1st", pt_beta);
        setRecoVars("betaIn_2ndCorr", copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaIn));
        setRecoVars("betaOut_2ndCorr", copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaOut));
        setRecoVars("dBeta_1st", betaInUpd - betaOutUpd);

        betaIn  += copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaIn); //FIXME: need a faster version
        betaOut += copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaOut); //FIXME: need a faster version
        //update the av and pt
        betaAv = 0.5f * (betaIn + betaOut);
        pt_beta = dr * k2Rinv1GeVf / sin(betaAv); //get a better pt estimate

        setRecoVars("betaIn_2nd", betaIn);
        setRecoVars("betaOut_2nd", betaOut);
        setRecoVars("betaAv_2nd", betaAv);
        setRecoVars("betaPt_2nd", pt_beta);
        setRecoVars("betaIn_3rdCorr", copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaIn));
        setRecoVars("betaOut_3rdCorr", copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaOut));
        setRecoVars("dBeta_2nd", betaIn - betaOut);

        setRecoVars("betaIn_3rd", getRecoVar("betaIn_0th") + copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaIn));
        setRecoVars("betaOut_3rd", getRecoVar("betaOut_0th") + copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaOut));
        setRecoVars("betaAv_3rd", 0.5f * (getRecoVar("betaIn_3rd") + getRecoVar("betaOut_3rd")));
        setRecoVars("betaPt_3rd", dr * k2Rinv1GeVf / sin(getRecoVar("betaAv_3rd")));
        setRecoVars("dBeta_3rd", getRecoVar("betaIn_3rd") - getRecoVar("betaOut_3rd"));

        setRecoVars("betaIn_4th", getRecoVar("betaIn_0th") + copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(getRecoVar("betaPt_3rd")), sinAlphaMax)), getRecoVar("betaIn_3rd")));
        setRecoVars("betaOut_4th", getRecoVar("betaOut_0th") + copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(getRecoVar("betaPt_3rd")), sinAlphaMax)), getRecoVar("betaOut_3rd")));
        setRecoVars("betaAv_4th", 0.5f * (getRecoVar("betaIn_4th") + getRecoVar("betaOut_4th")));
        setRecoVars("betaPt_4th", dr * k2Rinv1GeVf / sin(getRecoVar("betaAv_4th")));
        setRecoVars("dBeta_4th", getRecoVar("betaIn_4th") - getRecoVar("betaOut_4th"));


    }
    else if (lIn < 11 && std::abs(betaOut) < 0.2 * std::abs(betaIn) && std::abs(pt_beta) < 12.f * pt_betaMax)   //use betaIn sign as ref
    {
        setRecoVars("betacormode", 2);
        setRecoVars("betaIn_0th", betaIn);
        setRecoVars("betaOut_0th", betaOut);
        setRecoVars("betaAv_0th", betaAv);
        setRecoVars("betaPt_0th", pt_beta);
        setRecoVars("dBeta_0th", betaIn - betaOut);

        const float pt_betaIn = dr * k2Rinv1GeVf / sin(betaIn);
        const float betaInUpd  = betaIn + copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(pt_betaIn), sinAlphaMax)), betaIn); //FIXME: need a faster version
        const float betaOutUpd = betaOut + copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(pt_betaIn), sinAlphaMax)), betaIn); //FIXME: need a faster version
        betaAv = (std::abs(betaOut) > 0.2f * std::abs(betaIn)) ? (0.5f * (betaInUpd + betaOutUpd)) : betaInUpd;
        pt_beta = dr * k2Rinv1GeVf / sin(betaAv); //get a better pt estimate
        betaIn  += copysign(std::asin(std::min(sdIn_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaIn); //FIXME: need a faster version
        betaOut += copysign(std::asin(std::min(sdOut_dr * k2Rinv1GeVf / std::abs(pt_beta), sinAlphaMax)), betaIn); //FIXME: need a faster version
        //update the av and pt
        betaAv = 0.5f * (betaIn + betaOut);
        pt_beta = dr * k2Rinv1GeVf / sin(betaAv); //get a better pt estimate

        setRecoVars("betaIn_1st", betaIn);
        setRecoVars("betaOut_1st", betaOut);
        setRecoVars("betaAv_1st", betaAv);
        setRecoVars("betaPt_1st", pt_beta);
        setRecoVars("dBeta_1st", betaIn - betaOut);

        setRecoVars("betaIn_2nd", betaIn);
        setRecoVars("betaOut_2nd", betaOut);
        setRecoVars("betaAv_2nd", betaAv);
        setRecoVars("betaPt_2nd", pt_beta);
        setRecoVars("dBeta_2nd", betaIn - betaOut);

        setRecoVars("betaIn_3rd", betaIn);
        setRecoVars("betaOut_3rd", betaOut);
        setRecoVars("betaAv_3rd", betaAv);
        setRecoVars("betaPt_3rd", pt_beta);
        setRecoVars("dBeta_3rd", betaIn - betaOut);

        setRecoVars("betaIn_4th", betaIn);
        setRecoVars("betaOut_4th", betaOut);
        setRecoVars("betaAv_4th", betaAv);
        setRecoVars("betaPt_4th", pt_beta);
        setRecoVars("dBeta_4th", betaIn - betaOut);

    }
    else
    {
        setRecoVars("betacormode", 3);

        setRecoVars("betaIn_0th", betaIn);
        setRecoVars("betaOut_0th", betaOut);
        setRecoVars("betaAv_0th", betaAv);
        setRecoVars("betaPt_0th", pt_beta);
        setRecoVars("dBeta_0th", betaIn - betaOut);

        setRecoVars("betaIn_1st", betaIn);
        setRecoVars("betaOut_1st", betaOut);
        setRecoVars("betaAv_1st", betaAv);
        setRecoVars("betaPt_1st", pt_beta);
        setRecoVars("dBeta_1st", betaIn - betaOut);

        setRecoVars("betaIn_2nd", betaIn);
        setRecoVars("betaOut_2nd", betaOut);
        setRecoVars("betaAv_2nd", betaAv);
        setRecoVars("betaPt_2nd", pt_beta);
        setRecoVars("dBeta_2nd", betaIn - betaOut);

        setRecoVars("betaIn_3rd", betaIn);
        setRecoVars("betaOut_3rd", betaOut);
        setRecoVars("betaAv_3rd", betaAv);
        setRecoVars("betaPt_3rd", pt_beta);
        setRecoVars("dBeta_3rd", betaIn - betaOut);

        setRecoVars("betaIn_4th", betaIn);
        setRecoVars("betaOut_4th", betaOut);
        setRecoVars("betaAv_4th", betaAv);
        setRecoVars("betaPt_4th", pt_beta);
        setRecoVars("dBeta_4th", betaIn - betaOut);

    }

}

bool SDL::Tracklet::hasCommonSegment(const Tracklet& outer_tl) const
{
    if (outerSegmentPtr()->isIdxMatched(*(outer_tl.innerSegmentPtr())))
        return true;
    return false;
}

[[deprecated("SDL:: isSegmentPairATracklet() is deprecated")]]
bool SDL::Tracklet::isSegmentPairATracklet(const Segment& innerSegment, const Segment& outerSegment, TLAlgo algo, SDL::LogLevel logLevel)
{
    // If the algorithm is "do all combination" (e.g. used for efficiency calculation)
    if (algo == SDL::AllComb_TLAlgo)
    {
        return true;
    }
    else if (algo == SDL::Default_TLAlgo)
    {
        return false;
    }
    else
    {
        SDL::cout << "Warning: Unrecognized segment algorithm!" << algo << std::endl;
        return false;
    }
}

bool SDL::Tracklet::isSegmentPairATrackletBarrel(const Segment& innerSegment, const Segment& outerSegment, TLAlgo algo, SDL::LogLevel logLevel)
{
    return false;
}

bool SDL::Tracklet::isSegmentPairATrackletEndcap(const Segment& innerSegment, const Segment& outerSegment, TLAlgo algo, SDL::LogLevel logLevel)
{
    return false;
}

