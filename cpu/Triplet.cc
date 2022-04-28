#include "Triplet.h"

SDL::CPU::Triplet::Triplet()
{
}

SDL::CPU::Triplet::~Triplet()
{
}

SDL::CPU::Triplet::Triplet(const Triplet& tl) :
    TrackletBase(tl),
    tlCand(tl.tlCand)
{
}

SDL::CPU::Triplet::Triplet(SDL::CPU::Segment* innerSegmentPtr, SDL::CPU::Segment* outerSegmentPtr) :
    TrackletBase(innerSegmentPtr, outerSegmentPtr)
{
}

void SDL::CPU::Triplet::addSelfPtrToSegments()
{
    innerSegmentPtr_->addOutwardTripletPtr(this);
    outerSegmentPtr_->addInwardTripletPtr(this);
}

bool SDL::CPU::Triplet::passesTripletAlgo(SDL::CPU::TPAlgo algo) const
{
    // Each algorithm is an enum shift it by its value and check against the flag
    return passAlgo_ & (1 << algo);
}

void SDL::CPU::Triplet::runTripletAlgo(SDL::CPU::TPAlgo algo, SDL::CPU::LogLevel logLevel)
{
    if (algo == SDL::CPU::AllComb_TPAlgo)
    {
        runTripletAllCombAlgo();
    }
    else if (algo == SDL::CPU::Default_TPAlgo)
    {

        setRecoVars("betaPt_2nd", -999);

        runTripletDefaultAlgo(logLevel);
    }
    else
    {
        SDL::CPU::cout << "Warning: Unrecognized segment algorithm!" << algo << std::endl;
        return;
    }
}

void SDL::CPU::Triplet::runTripletAllCombAlgo()
{
    passAlgo_ |= (1 << SDL::CPU::AllComb_TPAlgo);
}

void SDL::CPU::Triplet::runTripletDefaultAlgo(SDL::CPU::LogLevel logLevel)
{

    passAlgo_ &= (0 << SDL::CPU::Default_TPAlgo);

    if (not (innerSegmentPtr()->hasCommonMiniDoublet(*(outerSegmentPtr()))))
    {
        if (logLevel >= SDL::CPU::Log_Debug3)
        {
            SDL::CPU::cout << "Failed Cut #1 in " << __FUNCTION__ << std::endl;
        }
        passAlgo_ &= (0 << SDL::CPU::Default_TPAlgo);
        return;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TripletSelection::commonSegment);

    // If it does not pass pointing constraint
    if (not (passPointingConstraint(logLevel)))
    {
        if (logLevel >= SDL::CPU::Log_Debug3)
        {
            SDL::CPU::cout << "Failed Cut #2 in " << __FUNCTION__ << std::endl;
        }
        passAlgo_ &= (0 << SDL::CPU::Default_TPAlgo);
        return;
    }

    //====================================================
    //
    // Running Tracklet algo within triplet
    //
    // if (false)
    {
        // Check tracklet algo on triplet
        tlCand = SDL::CPU::Tracklet(innerSegmentPtr(), outerSegmentPtr());

        tlCand.runTrackletDefaultAlgo(logLevel);

        if (not (tlCand.passesTrackletAlgo(SDL::CPU::Default_TLAlgo)))
        {
            if (logLevel >= SDL::CPU::Log_Debug3)
            {
                SDL::CPU::cout << "Failed Cut #3 in " << __FUNCTION__ << std::endl;
            }
            passAlgo_ &= (0 << SDL::CPU::Default_TPAlgo);
            return;
        }
        // Flag the pass bit
        passBitsDefaultAlgo_ |= (1 << TripletSelection::tracklet);
    }
    //
    //====================================================

    //====================================================
    //
    // Compute momentum of Triplet ( NOT USED )
    //
    if (true)
    {
        SDL::CPU::Hit& HitA = (*innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr());
        SDL::CPU::Hit& HitB = (*innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr());
        SDL::CPU::Hit& HitC = (*outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr());
        float g, f; // not used
        float radius = SDL::CPU::TrackCandidate::computeRadiusFromThreeAnchorHits(HitA.x(), HitA.y(), HitB.x(), HitB.y(), HitC.x(), HitC.y(), g, f);
        setRecoVars("tripletRadius", radius);
    }

    //====================================================
    //
    // Cut on momentum of Triplet ( NOT USED )
    //
    if (false)
    {
        SDL::CPU::Hit& HitA = (*innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr());
        SDL::CPU::Hit& HitB = (*innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr());
        SDL::CPU::Hit& HitC = (*outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr());
        SDL::CPU::Hit center = SDL::CPU::MathUtil::getCenterFromThreePoints(HitA, HitB, HitC);
        float ptEst = SDL::CPU::MathUtil::ptEstimateFromRadius((HitA - center).rt());

        if (not (ptEst > 1.))
        {
            if (logLevel >= SDL::CPU::Log_Debug3)
            {
                SDL::CPU::cout << "Failed Cut #4 in " << __FUNCTION__ << std::endl;
            }
            passAlgo_ &= (0 << SDL::CPU::Default_TPAlgo);
            return;
        }
        // Flag the pass bit
        passBitsDefaultAlgo_ |= (1 << TripletSelection::tracklet);
    }
    //
    //====================================================

    passAlgo_ |= (1 << SDL::CPU::Default_TPAlgo);
}

bool SDL::CPU::Triplet::passPointingConstraint(SDL::CPU::LogLevel logLevel)
{
    // SDL::CPU::cout << innerSegmentPtr();
    // SDL::CPU::cout << outerSegmentPtr();
    // return false;
    if (not passAdHocRZConstraint(logLevel))
        return false;
    
    const SDL::CPU::Module& ModuleA = innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule();
    const SDL::CPU::Module& ModuleB = innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule();
    const SDL::CPU::Module& ModuleC = outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule();

    if (ModuleA.subdet() == SDL::CPU::Module::Barrel
    and ModuleB.subdet() == SDL::CPU::Module::Barrel
    and ModuleC.subdet() == SDL::CPU::Module::Barrel)
    {
        return passPointingConstraintBarrelBarrelBarrel(logLevel);
    }
    else if (ModuleA.subdet() == SDL::CPU::Module::Barrel
         and ModuleB.subdet() == SDL::CPU::Module::Barrel
         and ModuleC.subdet() == SDL::CPU::Module::Endcap)
    {
        return passPointingConstraintBarrelBarrelEndcap(logLevel);
    }
    else if (ModuleA.subdet() == SDL::CPU::Module::Barrel
         and ModuleB.subdet() == SDL::CPU::Module::Endcap
         and ModuleC.subdet() == SDL::CPU::Module::Endcap)
    {
        return passPointingConstraintBarrelEndcapEndcap(logLevel);
    }
    else if (ModuleA.subdet() == SDL::CPU::Module::Endcap
         and ModuleB.subdet() == SDL::CPU::Module::Endcap
         and ModuleC.subdet() == SDL::CPU::Module::Endcap)
    {
        return passPointingConstraintEndcapEndcapEndcap(logLevel);
        // return false;
    }
    else
    {
        SDL::CPU::cout << ModuleA.subdet() << std::endl;
        SDL::CPU::cout << ModuleB.subdet() << std::endl;
        SDL::CPU::cout << ModuleC.subdet() << std::endl;
        SDL::CPU::cout << "WHY AM I HERE?" << std::endl;
        return false;
    }
}

bool SDL::CPU::Triplet::passPointingConstraintBarrelBarrelBarrel(SDL::CPU::LogLevel logLevel)
{


    // Z pointing between inner most MD to outer most MD

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

    // NOTE in triplet InUp == OutLo

    const bool isPS_InLo = ((innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).moduleType() == SDL::CPU::Module::PS);
    //const bool isPS_OutLo = ((outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).moduleType() == SDL::CPU::Module::PS);
    const bool isPS_OutUp = ((outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).moduleType() == SDL::CPU::Module::PS);
    //const bool isEC_OutUp = ((outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).subdet() == SDL::CPU::Module::Endcap);

    const float rt_InLo = innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->rt();
    const float rt_OutUp = outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->rt();
    const float z_InLo = innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->z();
    const float z_OutUp = outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->z();

    const float alpha1GeV_OutUp = std::asin(std::min(rt_OutUp * k2Rinv1GeVf / ptCut, sinAlphaMax));

    const float rtRatio_OutUpInLo = rt_OutUp / rt_InLo; // Outer segment beginning rt divided by inner segment beginning rt;
    const float dzDrtScale = tan(alpha1GeV_OutUp) / alpha1GeV_OutUp; // The track can bend in r-z plane slightly
    const float zpitch_InLo = (isPS_InLo ? pixelPSZpitch : strip2SZpitch);
    const float zpitch_OutUp = (isPS_OutUp ? pixelPSZpitch : strip2SZpitch);

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
    //  |    --------------x-- ----------- --x--------------
    //  |    ---------------x- ---------|- -x--|------------ <- z_OutUp, rt_OutUp
    //  |                               <------>
    //  |                              zLo     zHi
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
    // (rt_OutUp - rt_InLo)            rt_InLo
    //
    // Then, if you solve for zHi you get most of the the below equation
    // Then there are two corrections.
    // dzDrtScale to account for the forward-bending of the helix tracks
    // z_pitch terms to additonally add error

    const float zHi = z_InLo + (z_InLo + deltaZLum) * (rtRatio_OutUpInLo - 1.f) * (z_InLo < 0.f ? 1.f : dzDrtScale) + (zpitch_InLo + zpitch_OutUp);
    const float zLo = z_InLo + (z_InLo - deltaZLum) * (rtRatio_OutUpInLo - 1.f) * (z_InLo > 0.f ? 1.f : dzDrtScale) - (zpitch_InLo + zpitch_OutUp); //slope-correction only on outer end

    // Reset passBitsDefaultAlgo_;
    passBitsDefaultAlgo_ = 0;

    //==========================================================================
    //
    // Cut #1: Z compatibility
    //
    //==========================================================================
    setRecoVars("z_OutUp", z_OutUp);
    setRecoVars("zLo", zLo);
    setRecoVars("zHi", zHi);
    if (not (z_OutUp >= zLo and z_OutUp <= zHi))
    {
        if (logLevel >= SDL::CPU::Log_Debug3)
        {
            SDL::CPU::cout << "Failed Cut #2 in " << __FUNCTION__ << std::endl;
            SDL::CPU::cout <<  " zLo: " << zLo <<  " z_OutUp: " << z_OutUp <<  " zHi: " << zHi <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::CPU::Default_TLAlgo);
        return false;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TripletSelection::deltaZ);
    //--------------------------------------------------------------------------

    const float drt_OutUp_InLo = (rt_OutUp - rt_InLo);
    //const float invRt_InLo = 1. / rt_InLo;
    const float r3_InLo = sqrt(z_InLo * z_InLo + rt_InLo * rt_InLo);
    const float drt_InSeg = innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->rt() - innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->rt();
    const float dz_InSeg  = innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->z() - innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->z();
    const float dr3_InSeg = innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->r3() - innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->r3();

    // The x's are outer segment
    // The y's are inner segment
    //
    // rt
    //  ^
    //  |    --------------x-- ----------- --x--------------
    //  |    ---------------x- ---------|- -x--|------------ <- z_OutUp, rt_OutUp
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
    float dzErr = (zpitch_InLo + zpitch_OutUp) * (zpitch_InLo + zpitch_OutUp) * 2.f; //both sides contribute to direction uncertainty

    // Multiple scattering
    //FIXME (later) more realistic accounting of material effects is needed
    const float sdlThetaMulsF = 0.015f * sqrt(0.1f + 0.2 * (rt_OutUp - rt_InLo) / 50.f) * sqrt(r3_InLo / rt_InLo);
    const float sdlMuls = sdlThetaMulsF * 3.f / ptCut * 4.f; // will need a better guess than x4?
    dzErr += sdlMuls * sdlMuls * drt_OutUp_InLo * drt_OutUp_InLo / 3.f * coshEta * coshEta; //sloppy
    dzErr = sqrt(dzErr);

    // Constructing upper and lower bound
    const float dzMean = dz_InSeg / drt_InSeg * drt_OutUp_InLo;
    const float zWindow = dzErr / drt_InSeg * drt_OutUp_InLo + (zpitch_InLo + zpitch_OutUp); //FIXME for ptCut lower than ~0.8 need to add curv path correction
    const float zLoPointed = z_InLo + dzMean * (z_InLo > 0.f ? 1.f : dzDrtScale) - zWindow;
    const float zHiPointed = z_InLo + dzMean * (z_InLo < 0.f ? 1.f : dzDrtScale) + zWindow;

    //==========================================================================
    //
    // Cut #2: Pointed Z (Inner segment two MD points to outer segment inner MD)
    //
    //==========================================================================
    setRecoVars("zLoPointed", zLoPointed);
    setRecoVars("zHiPointed", zHiPointed);
    if (not (z_OutUp >= zLoPointed and z_OutUp <= zHiPointed))
    {
        if (logLevel >= SDL::CPU::Log_Debug3)
        {
            SDL::CPU::cout << "Failed Cut #3 in " << __FUNCTION__ << std::endl;
            SDL::CPU::cout <<  " zLoPointed: " << zLoPointed <<  " z_OutUp: " << z_OutUp <<  " zHiPointed: " << zHiPointed <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::CPU::Default_TLAlgo);
        return false;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TripletSelection::deltaZPointed);
    //--------------------------------------------------------------------------

    return true;

}

bool SDL::CPU::Triplet::passPointingConstraintBarrelBarrelEndcap(SDL::CPU::LogLevel logLevel)
{
    const float deltaZLum = 15.f;
    const float pixelPSZpitch = 0.15;
    const float strip2SZpitch = 5.0;
    const float zGeom =
        ((innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).moduleType() == SDL::CPU::Module::PS ? pixelPSZpitch : strip2SZpitch)
        +
        ((outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).moduleType() == SDL::CPU::Module::PS ? pixelPSZpitch : strip2SZpitch);
    const float rtIn = innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->rt();
    const float rtOut = outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->rt();
    const float zIn = innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->z();
    const float zOut = outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->z();
    //const float rtOut_o_rtIn = rtOut / rtIn;
    const float kRinv1GeVf = (2.99792458e-3 * 3.8);
    const float k2Rinv1GeVf = kRinv1GeVf / 2.;
    const float ptCut = PTCUT;
    const float sinAlphaMax = 0.95;
    const float sdlSlope = std::asin(std::min(rtOut * k2Rinv1GeVf / ptCut, sinAlphaMax));
    const float dzDrtScale = tan(sdlSlope) / sdlSlope;//FIXME: need approximate value
    //const float zLo = zIn + (zIn - deltaZLum) * (rtOut_o_rtIn - 1.f) * (zIn > 0.f ? 1.f : dzDrtScale) - zGeom; //slope-correction only on outer end

    // Cut #0: Preliminary
    // (Only here in endcap case)
    if (not (zIn * zOut > 0))
    {
        if (logLevel >= SDL::CPU::Log_Debug3)
        {
            SDL::CPU::cout << "Failed Cut #0 in " << __FUNCTION__ << std::endl;
            SDL::CPU::cout <<  " zIn: " << zIn <<  std::endl;
            SDL::CPU::cout <<  " zOut: " << zOut <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::CPU::Default_TLAlgo);
        return false;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TripletSelection::deltaZ);


    const float dLum = std::copysign(deltaZLum, zIn);
    const bool isOutSgInnerMDPS = (outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).moduleType() == SDL::CPU::Module::PS;
    const float rtGeom1 = isOutSgInnerMDPS ? pixelPSZpitch : strip2SZpitch;//FIXME: make this chosen by configuration for lay11,12 full PS
    const float zGeom1 = std::copysign(zGeom, zIn); //used in B-E region
    const float rtLo = rtIn * (1.f + (zOut - zIn - zGeom1) / (zIn + zGeom1 + dLum) / dzDrtScale) - rtGeom1; //slope correction only on the lower end

    // Cut #1: rt condition
    if (not (rtOut >= rtLo))
    {
        if (logLevel >= SDL::CPU::Log_Debug3)
        {
            SDL::CPU::cout << "Failed Cut #1 in " << __FUNCTION__ << std::endl;
            SDL::CPU::cout <<  " rtOut: " << rtOut <<  std::endl;
            SDL::CPU::cout <<  " rtLo: " << rtLo <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::CPU::Default_TLAlgo);
        return false;
    }

    float zInForHi = zIn - zGeom1 - dLum;
    if (zInForHi * zIn < 0)
        zInForHi = std::copysign(0.1f, zIn);
    const float rtHi = rtIn * (1.f + (zOut - zIn + zGeom1) / zInForHi) + rtGeom1;

    // Cut #2: rt condition
    if (not (rtOut >= rtLo and rtOut <= rtHi))
    {
        if (logLevel >= SDL::CPU::Log_Debug3)
        {
            SDL::CPU::cout << "Failed Cut #2 in " << __FUNCTION__ << std::endl;
            SDL::CPU::cout <<  " rtOut: " << rtOut <<  std::endl;
            SDL::CPU::cout <<  " rtLo: " << rtLo <<  std::endl;
            SDL::CPU::cout <<  " rtHi: " << rtHi <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::CPU::Default_TLAlgo);
        return false;
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
    //const float drtMean = drtSDIn * dzOutInAbs / std::abs(dzSDIn); //
    //const float rtWindow = drtErr + rtGeom1;
    //const float rtLo_another = rtIn + drtMean / dzDrtScale - rtWindow;
    //const float rtHi_another = rtIn + drtMean + rtWindow;

    // Cut #3: rt-z pointed
    if (not (kZ >= 0 and rtOut >= rtLo and rtOut <= rtHi))
    {
        if (logLevel >= SDL::CPU::Log_Debug3)
        {
            SDL::CPU::cout << "Failed Cut #3 in " << __FUNCTION__ << std::endl;
            SDL::CPU::cout <<  " kZ: " << kZ <<  std::endl;
            SDL::CPU::cout <<  " rtOut: " << rtOut <<  std::endl;
            SDL::CPU::cout <<  " rtLo: " << rtLo <<  std::endl;
            SDL::CPU::cout <<  " rtHi: " << rtHi <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::CPU::Default_TLAlgo);
        return false;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TripletSelection::deltaZPointed);

    return true;
}

bool SDL::CPU::Triplet::passPointingConstraintBarrelEndcapEndcap(SDL::CPU::LogLevel logLevel)
{
    return passPointingConstraintBarrelBarrelEndcap(logLevel);
}

bool SDL::CPU::Triplet::passPointingConstraintEndcapEndcapEndcap(SDL::CPU::LogLevel logLevel)
{
    const float deltaZLum = 15.f;
    const float pixelPSZpitch = 0.15;
    const float strip2SZpitch = 5.0;
    //const float zGeom =
    //    ((innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).moduleType() == SDL::CPU::Module::PS ? pixelPSZpitch : strip2SZpitch)
    //    +
    //    ((outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).moduleType() == SDL::CPU::Module::PS ? pixelPSZpitch : strip2SZpitch);
    const float rtIn = innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->rt();
    const float rtOut = outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->rt();
    const float zIn = innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->z();
    const float zOut = outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->z();
    //const float rtOut_o_rtIn = rtOut / rtIn;
    const float kRinv1GeVf = (2.99792458e-3 * 3.8);
    const float k2Rinv1GeVf = kRinv1GeVf / 2.;
    const float ptCut = PTCUT;
    const float sinAlphaMax = 0.95;
    const float sdlSlope = std::asin(std::min(rtOut * k2Rinv1GeVf / ptCut, sinAlphaMax));
    const float dzDrtScale = tan(sdlSlope) / sdlSlope;//FIXME: need approximate value
    //const float zLo = zIn + (zIn - deltaZLum) * (rtOut_o_rtIn - 1.f) * (zIn > 0.f ? 1.f : dzDrtScale) - zGeom; //slope-correction only on outer end

    // Cut #0: Preliminary
    // (Only here in endcap case)
    // https://github.com/slava77/cms-tkph2-ntuple/blob/superDoubletLinked-91X-noMock/doubletAnalysis.C#L3631-L3633
    if (not (zIn * zOut > 0))
    {
        if (logLevel >= SDL::CPU::Log_Debug3)
        {
            SDL::CPU::cout << "Failed Cut #0 in " << __FUNCTION__ << std::endl;
            SDL::CPU::cout <<  " zIn: " << zIn <<  std::endl;
            SDL::CPU::cout <<  " zOut: " << zOut <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::CPU::Default_TLAlgo);
        return false;
    }


    const float dLum = std::copysign(deltaZLum, zIn);
    const bool isOutSgInnerMDPS = (outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).moduleType() == SDL::CPU::Module::PS;
    const bool isInSgInnerMDPS = (innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()).moduleType() == SDL::CPU::Module::PS;
    // https://github.com/slava77/cms-tkph2-ntuple/blob/superDoubletLinked-91X-noMock/doubletAnalysis.C#L3670-L3674
    // we're in mockMode == 3
    const float rtGeom = (isInSgInnerMDPS && isOutSgInnerMDPS ? 2.f * pixelPSZpitch
                 : (isInSgInnerMDPS || isOutSgInnerMDPS ) ? (pixelPSZpitch + strip2SZpitch)
                            : 2.f * strip2SZpitch);
    //const float rtGeom1 = isOutSgInnerMDPS ? pixelPSZpitch : strip2SZpitch;//FIXME: make this chosen by configuration for lay11,12 full PS
    //const float zGeom1 = std::copysign(zGeom, zIn); //used in B-E region
    const float dz = zOut - zIn;
    const float rtLo = rtIn * (1.f + dz / (zIn + dLum) / dzDrtScale) - rtGeom; //slope correction only on the lower end

    // Cut #1: rt condition
    // https://github.com/slava77/cms-tkph2-ntuple/blob/superDoubletLinked-91X-noMock/doubletAnalysis.C#L3679
    if (not (rtOut >= rtLo))
    {
        if (logLevel >= SDL::CPU::Log_Debug3)
        {
            SDL::CPU::cout << "Failed Cut #1 in " << __FUNCTION__ << std::endl;
            SDL::CPU::cout <<  " rtOut: " << rtOut <<  std::endl;
            SDL::CPU::cout <<  " rtLo: " << rtLo <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::CPU::Default_TLAlgo);
        return false;
    }

    const float rtHi = rtIn * (1.f + dz / (zIn - dLum)) + rtGeom;

    // Cut #2: rt condition
    if (not (rtOut >= rtLo and rtOut <= rtHi))
    {
        if (logLevel >= SDL::CPU::Log_Debug3)
        {
            SDL::CPU::cout << "Failed Cut #2 in " << __FUNCTION__ << std::endl;
            SDL::CPU::cout <<  " rtOut: " << rtOut <<  std::endl;
            SDL::CPU::cout <<  " rtLo: " << rtLo <<  std::endl;
            SDL::CPU::cout <<  " rtHi: " << rtHi <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::CPU::Default_TLAlgo);
        return false;
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TripletSelection::deltaZ);

    const bool isInSgOuterMDPS = (innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()).moduleType() == SDL::CPU::Module::PS;

    //const float drOutIn = (rtOut - rtIn);
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
            if (logLevel >= SDL::CPU::Log_Debug3)
            {
                SDL::CPU::cout << "Failed Cut #3 in " << __FUNCTION__ << std::endl;
                SDL::CPU::cout <<  " kZ: " << kZ <<  std::endl;
                SDL::CPU::cout <<  " rtOut: " << rtOut <<  std::endl;
                SDL::CPU::cout <<  " rtLo: " << rtLo <<  std::endl;
                SDL::CPU::cout <<  " rtHi: " << rtHi <<  std::endl;
            }
            passAlgo_ &= (0 << SDL::CPU::Default_TLAlgo);
            return false;
        }
    }
    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << TripletSelection::deltaZPointed);
    return true;
}

bool SDL::CPU::Triplet::passAdHocRZConstraint(SDL::CPU::LogLevel logLevel)
{

    // Obtain the R's and Z's
    const float& r1 = innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->rt();
    const float& z1 = innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->z();
    const float& r2 = innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->rt();
    const float& z2 = innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->z();
    const float& r3 = outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->rt();
    const float& z3 = outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->z();

    const float residual = z2 - ( (z3 - z1) / (r3 - r1) * (r2 - r1) + z1);
    setRecoVars("residual", residual);

    const SDL::CPU::Module& ModuleA = innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule();
    const SDL::CPU::Module& ModuleB = innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule();
    const SDL::CPU::Module& ModuleC = outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule();

    const int layer1 =  ModuleA.layer() + 6 * (ModuleA.subdet() == 4) + 5 * (ModuleA.subdet() == 4 and ModuleA.moduleType() == 1);
    const int layer2 =  ModuleB.layer() + 6 * (ModuleB.subdet() == 4) + 5 * (ModuleB.subdet() == 4 and ModuleB.moduleType() == 1);
    const int layer3 =  ModuleC.layer() + 6 * (ModuleC.subdet() == 4) + 5 * (ModuleC.subdet() == 4 and ModuleC.moduleType() == 1);

    if (layer1 == 12 and layer2 == 13 and layer3 == 14)
    {
        return false;
    }
    else if (layer1 == 1 and layer2 == 2 and layer3 == 3)
    {
        return fabs(residual) < 0.53;
    }
    else if (layer1 == 1 and layer2 == 2 and layer3 == 7)
    {
        return fabs(residual) < 1;
    }
    else if (layer1 == 13 and layer2 == 14 and layer3 == 15)
    {
        return false;
    }
    else if (layer1 == 14 and layer2 == 15 and layer3 == 16)
    {
        return false;
    }
    else if (layer1 == 1 and layer2 == 7 and layer3 == 8)
    {
        return fabs(residual) < 1;
    }
    else if (layer1 == 2 and layer2 == 3 and layer3 == 4)
    {
        return fabs(residual) < 1.21;
    }
    else if (layer1 == 2 and layer2 == 3 and layer3 == 7)
    {
        return fabs(residual) < 1.;
    }
    else if (layer1 == 2 and layer2 == 7 and layer3 == 8)
    {
        return fabs(residual) < 1.;
    }
    else if (layer1 == 3 and layer2 == 4 and layer3 == 5)
    {
        return fabs(residual) < 2.7;
    }
    else if (layer1 == 4 and layer2 == 5 and layer3 == 6)
    {
        return fabs(residual) < 3.06;
    }
    else if (layer1 == 7 and layer2 == 8 and layer3 == 9)
    {
        return fabs(residual) < 1;
    }
    else if (layer1 == 8 and layer2 == 9 and layer3 == 10)
    {
        return fabs(residual) < 1;
    }
    else if (layer1 == 9 and layer2 == 10 and layer3 == 11)
    {
        return fabs(residual) < 1;
    }
    else
    {
        return fabs(residual) < 5;
    }

}
