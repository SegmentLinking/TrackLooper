#include "Segment.h"

SDL::Segment::Segment()
{
}

SDL::Segment::~Segment()
{
}

SDL::Segment::Segment(const Segment& sg) :
    innerMiniDoubletPtr_(sg.innerMiniDoubletPtr()),
    outerMiniDoubletPtr_(sg.outerMiniDoubletPtr()),
    passAlgo_(sg.getPassAlgo()),
    passBitsDefaultAlgo_(sg.getPassBitsDefaultAlgo()),
    rtLo_(sg.getRtLo()),
    rtHi_(sg.getRtHi()),
    rtOut_(sg.getRtOut()),
    rtIn_(sg.getRtIn()),
    dphi_(sg.getDeltaPhi()),
    dphi_min_(sg.getDeltaPhiMin()),
    dphi_max_(sg.getDeltaPhiMax()),
    dphichange_(sg.getDeltaPhiChange()),
    dphichange_min_(sg.getDeltaPhiMinChange()),
    dphichange_max_(sg.getDeltaPhiMaxChange()),
    zOut_(sg.getZOut()),
    zIn_(sg.getZIn()),
    zLo_(sg.getZLo()),
    zHi_(sg.getZHi()),
    recovars_(sg.getRecoVars())
{
    // addSelfPtrToMiniDoublets();
}

SDL::Segment::Segment(SDL::MiniDoublet* innerMiniDoubletPtr, SDL::MiniDoublet* outerMiniDoubletPtr) :
    innerMiniDoubletPtr_(innerMiniDoubletPtr),
    outerMiniDoubletPtr_(outerMiniDoubletPtr),
    passAlgo_(0),
    passBitsDefaultAlgo_(0),
    rtLo_(0),
    rtHi_(0),
    rtOut_(0),
    rtIn_(0),
    dphi_(0),
    dphi_min_(0),
    dphi_max_(0),
    dphichange_(0),
    dphichange_min_(0),
    dphichange_max_(0),
    zOut_(0),
    zIn_(0),
    zLo_(0),
    zHi_(0)
{
    // addSelfPtrToMiniDoublets();
}

void SDL::Segment::addSelfPtrToMiniDoublets()
{
    innerMiniDoubletPtr_->addOutwardSegmentPtr(this);
    outerMiniDoubletPtr_->addInwardSegmentPtr(this);
}

const std::vector<SDL::Tracklet*>& SDL::Segment::getListOfOutwardTrackletPtrs()
{
    return outwardTrackletPtrs;
}

const std::vector<SDL::Tracklet*>& SDL::Segment::getListOfInwardTrackletPtrs()
{
    return inwardTrackletPtrs;
}

void SDL::Segment::addOutwardTrackletPtr(SDL::Tracklet* tl)
{
    outwardTrackletPtrs.push_back(tl);
}

void SDL::Segment::addInwardTrackletPtr(SDL::Tracklet* tl)
{
    inwardTrackletPtrs.push_back(tl);
}

SDL::MiniDoublet* SDL::Segment::innerMiniDoubletPtr() const
{
    return innerMiniDoubletPtr_;
}

SDL::MiniDoublet* SDL::Segment::outerMiniDoubletPtr() const
{
    return outerMiniDoubletPtr_;
}

const int& SDL::Segment::getPassAlgo() const
{
    return passAlgo_;
}

const int& SDL::Segment::getPassBitsDefaultAlgo() const
{
    return passBitsDefaultAlgo_;
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
    return dphi_min_;
}

const float& SDL::Segment::getDeltaPhiMax() const
{
    return dphi_max_;
}

const float& SDL::Segment::getDeltaPhiChange() const
{
    return dphichange_;
}

const float& SDL::Segment::getDeltaPhiMinChange() const
{
    return dphichange_min_;
}

const float& SDL::Segment::getDeltaPhiMaxChange() const
{
    return dphichange_max_;
}

const float& SDL::Segment::getZOut() const
{
    return zOut_;
}

const float& SDL::Segment::getZIn() const
{
    return zIn_;
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

const float& SDL::Segment::getRtHi() const
{
    return rtHi_;
}

const std::map<std::string, float>& SDL::Segment::getRecoVars() const
{
    return recovars_;
}

const float& SDL::Segment::getRecoVar(std::string key) const
{
    return recovars_.at(key);
}

const float& SDL::Segment::getdAlphaInnerMDSegment() const
{
    return dAlphaInnerMDSegment_;
}

const float& SDL::Segment::getdAlphaOuterMDSegment() const
{
    return dAlphaOuterMDSegment_;
}

const float& SDL::Segment::getdAlphaInnerMDOuterMD() const
{
    return dAlphaInnerMDOuterMD_;
}

void SDL::Segment::setRtOut(float rt)
{
    rtOut_ = rt;
}

void SDL::Segment::setRtIn(float rt)
{
    rtIn_ = rt;
}

void SDL::Segment::setRtLo(float rt)
{
    rtLo_ = rt;
}

void SDL::Segment::setRtHi(float rt)
{
    rtHi_ = rt;
}

void SDL::Segment::setDeltaPhi(float dphi)
{
    dphi_ = dphi;
}

void SDL::Segment::setDeltaPhiMin(float dphimin)
{
    dphi_min_ = dphimin;
}

void SDL::Segment::setDeltaPhiMax(float dphimax)
{
    dphi_max_ = dphimax;
}

void SDL::Segment::setDeltaPhiChange(float dphichange)
{
    dphichange_ = dphichange;
}

void SDL::Segment::setDeltaPhiMinChange(float dphichangemin)
{
    dphichange_min_ = dphichangemin;
}

void SDL::Segment::setDeltaPhiMaxChange(float dphichangemax)
{
    dphichange_max_ = dphichangemax;
}

void SDL::Segment::setZOut(float zOut)
{
    zOut_ = zOut;
}

void SDL::Segment::setZIn(float zIn)
{
    zIn_ = zIn;
}

void SDL::Segment::setZLo(float zLo)
{
    zLo_ = zLo;
}

void SDL::Segment::setZHi(float zHi)
{
    zHi_ = zHi;
}

void SDL::Segment::setRecoVars(std::string key, float var)
{
    recovars_[key] = var;
}

void SDL::Segment::setdAlphaInnerMDSegment(float var)
{
    dAlphaInnerMDSegment_ = var;
}

void SDL::Segment::setdAlphaOuterMDSegment(float var)
{
    dAlphaOuterMDSegment_ = var;
}

void SDL::Segment::setdAlphaInnerMDOuterMD(float var)
{
    dAlphaInnerMDOuterMD_ = var;
}


bool SDL::Segment::passesSegmentAlgo(SDL::SGAlgo algo) const
{
    // Each algorithm is an enum shift it by its value and check against the flag
    return passAlgo_ & (1 << algo);
}

void SDL::Segment::runSegmentAlgo(SDL::SGAlgo algo, SDL::LogLevel logLevel)
{
    if (algo == SDL::AllComb_SGAlgo)
    {
        runSegmentAllCombAlgo();
    }
    else if (algo == SDL::Default_SGAlgo)
    {
        runSegmentDefaultAlgo(logLevel);
    }
    else
    {
        SDL::cout << "Warning: Unrecognized segment algorithm!" << algo << std::endl;
        return;
    }
}

void SDL::Segment::runSegmentAllCombAlgo()
{

    passAlgo_ |= (1 << SDL::AllComb_SGAlgo);
    // return;

    const MiniDoublet& innerMiniDoublet = (*innerMiniDoubletPtr());
    const MiniDoublet& outerMiniDoublet = (*outerMiniDoubletPtr());

    const Hit& innerAnchorHit = (*innerMiniDoublet.anchorHitPtr());
    const Hit& outerAnchorHit = (*outerMiniDoublet.anchorHitPtr());

    float dr = outerAnchorHit.rt() - innerAnchorHit.rt();

    // // Cut #0: Module compatibility
    // if (not (dr < 35.))
    // {
    //     passAlgo_ &= (0 << SDL::AllComb_SGAlgo);
    //     return;
    // }

    passAlgo_ |= (1 << SDL::AllComb_SGAlgo);
}

void SDL::Segment::runSegmentDefaultAlgo(SDL::LogLevel logLevel)
{
    // Retreived the lower module object
    const Module& innerLowerModule = innerMiniDoubletPtr()->lowerHitPtr()->getModule();
    const Module& outerLowerModule = outerMiniDoubletPtr()->lowerHitPtr()->getModule();

    //FIXME:Change the whole thing to a check in outer module alone if this trick works!

    if (innerLowerModule.subdet() == SDL::Module::Barrel)
    {
        if (outerLowerModule.subdet() == SDL::Module::Barrel)
        {
            //Needs a name change to BarrelBarrel later
            runSegmentDefaultAlgoBarrel(logLevel);
        }
        else
            runSegmentDefaultAlgoEndcap(logLevel);
    }
    else
    {
        if (outerLowerModule.subdet() == SDL::Module::Endcap)
            runSegmentDefaultAlgoEndcap(logLevel);
        else //shouldn't really be running
            runSegmentDefaultAlgoBarrel(logLevel);
    }
}

void SDL::Segment::runSegmentDefaultAlgoBarrel(SDL::LogLevel logLevel)
{

    const MiniDoublet& innerMiniDoublet = (*innerMiniDoubletPtr());
    const MiniDoublet& outerMiniDoublet = (*outerMiniDoubletPtr());

    const Module& innerLowerModule = innerMiniDoublet.lowerHitPtr()->getModule();
    const Module& outerLowerModule = outerMiniDoublet.lowerHitPtr()->getModule();

    // Reset passBitsDefaultAlgo_;
    passBitsDefaultAlgo_ = 0;

    // Get connected outer lower module detids
    const std::vector<unsigned int>& connectedModuleDetIds = moduleConnectionMap.getConnectedModuleDetIds(innerLowerModule.detId());


    // set default values
    
    setRecoVars("sdCut", -999);
    setRecoVars("sdSlope", -999);
    setRecoVars("sdMuls", -999);
    setRecoVars("sdPVoff", -999);
    setRecoVars("deltaPhi", -999);

    setRecoVars("dAlpha_innerMD_segment",-999);
    setRecoVars("dAlpha_outerMD_segment",-999);
    setRecoVars("dAlpha_innerMD_outerMD",-999);


    // // Loop over connected outer lower modules
    // bool found = false;
    // for (auto& outerLowerModuleDetId : connectedModuleDetIds)
    // {
    //     if (outerLowerModule.detId() == outerLowerModuleDetId)
    //     {
    //         found = true;
    //         break;
    //     }
    // }

    // // bool isOuterEndcap = (outerLowerModule.subdet() == SDL::Module::Endcap);
    // bool isOuterEndcap = false;

    // // Cut #0: Module compatibility
    // if (not (found or isOuterEndcap))
    // {
    //     if (logLevel >= SDL::Log_Debug3)
    //     {
    //         SDL::cout << "Failed Cut #0 in " << __FUNCTION__ << std::endl;
    //         SDL::cout <<  " innerLowerModule.detId(): " << innerLowerModule.detId() <<  " outerLowerModule.detId(): " << outerLowerModule.detId() << std::endl;
    //     }
    //     passAlgo_ &= (0 << SDL::Default_SGAlgo);
    //     return;
    // }

    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << SegmentSelection::moduleCompatible);

    // Constants
    const float kRinv1GeVf = (2.99792458e-3 * 3.8);
    const float k2Rinv1GeVf = kRinv1GeVf / 2.;
    const float ptCut = PTCUT;
    const float sinAlphaMax = 0.95;
    const float deltaZLum = 15.f;
    std::array<float, 6> miniMulsPtScaleBarrel {0.0052, 0.0038, 0.0034, 0.0034, 0.0032, 0.0034};
    // std::array<float, 5> miniMulsPtScaleEndcap {0.006, 0.006, 0.006, 0.006, 0.006}; //inter/extra-polated from L11 and L13 both roughly 0.006 [larger R have smaller value by ~50%]
    const float sdMuls = miniMulsPtScaleBarrel[innerLowerModule.layer()-1] * 3.f / ptCut;//will need a better guess than x2?

    // Get the relevant anchor hits
    const Hit& innerMiniDoubletAnchorHit = (innerLowerModule.moduleType() == SDL::Module::PS) ? ( (innerLowerModule.moduleLayerType() == SDL::Module::Pixel) ? *innerMiniDoublet.lowerHitPtr() : *innerMiniDoublet.upperHitPtr()): *innerMiniDoublet.lowerHitPtr();
    const Hit& outerMiniDoubletAnchorHit = (outerLowerModule.moduleType() == SDL::Module::PS) ? ( (outerLowerModule.moduleLayerType() == SDL::Module::Pixel) ? *outerMiniDoublet.lowerHitPtr() : *outerMiniDoublet.upperHitPtr()): *outerMiniDoublet.lowerHitPtr();

    // MiniDoublet information
    float innerMiniDoubletAnchorHitRt = innerMiniDoubletAnchorHit.rt();
    float outerMiniDoubletAnchorHitRt = outerMiniDoubletAnchorHit.rt();
    float innerMiniDoubletAnchorHitZ = innerMiniDoubletAnchorHit.z();
    float outerMiniDoubletAnchorHitZ = outerMiniDoubletAnchorHit.z();

    // Reco value set
    setRtOut(outerMiniDoubletAnchorHitRt);
    setRtIn(innerMiniDoubletAnchorHitRt);

    const float sdSlope = std::asin(std::min(outerMiniDoubletAnchorHitRt * k2Rinv1GeVf / ptCut, sinAlphaMax));
    const float sdPVoff = 0.1f / outerMiniDoubletAnchorHitRt;
    const float dzDrtScale = std::tan(sdSlope) / sdSlope; //FIXME: need approximate value
    const float pixelPSZpitch = 0.15;
    const float strip2SZpitch = 5.0;

    const float zGeom = (innerLowerModule.layer() <= 2) ? (2.f * pixelPSZpitch) : (2.f * strip2SZpitch); //twice the macro-pixel or strip size

    float zLo = innerMiniDoubletAnchorHitZ + (innerMiniDoubletAnchorHitZ - deltaZLum) * (outerMiniDoubletAnchorHitRt / innerMiniDoubletAnchorHitRt - 1.f) * (innerMiniDoubletAnchorHitZ > 0.f ? 1.f : dzDrtScale) - zGeom; //slope-correction only on outer end
    float zHi = innerMiniDoubletAnchorHitZ + (innerMiniDoubletAnchorHitZ + deltaZLum) * (outerMiniDoubletAnchorHitRt / innerMiniDoubletAnchorHitRt - 1.f) * (innerMiniDoubletAnchorHitZ < 0.f ? 1.f : dzDrtScale) + zGeom;

    setZOut(outerMiniDoubletAnchorHitZ);
    setZIn(innerMiniDoubletAnchorHitZ);
    setZLo(zLo);
    setZHi(zHi);

    // Cut #1: Z compatibility
    if (not (outerMiniDoubletAnchorHitZ >= zLo and outerMiniDoubletAnchorHitZ <= zHi))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #1 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " zLo: " << zLo <<  " outerMiniDoubletAnchorHitZ: " << outerMiniDoubletAnchorHitZ <<  " zHi: " << zHi <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_SGAlgo);
        return;
    }

    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << SegmentSelection::deltaZ);

    const float sdCut = sdSlope + sqrt(sdMuls * sdMuls + sdPVoff * sdPVoff);
    const float deltaPhi = innerMiniDoubletAnchorHit.deltaPhi(outerMiniDoubletAnchorHit);

    setRecoVars("sdCut", sdCut);
    setRecoVars("sdSlope", sdSlope);
    setRecoVars("sdMuls", sdMuls);
    setRecoVars("sdPVoff", sdPVoff);
    setRecoVars("deltaPhi", deltaPhi);

    // Cut #2: phi differences between the two minidoublets
    if (not (std::abs(deltaPhi) <= sdCut))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #2 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " deltaPhi: " << deltaPhi <<  " sdCut: " << sdCut <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_SGAlgo);
        return;
    }

    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << SegmentSelection::deltaPhiPos);

    setDeltaPhiChange(innerMiniDoubletAnchorHit.deltaPhiChange(outerMiniDoubletAnchorHit));

    // Cut #3: phi change between the two minidoublets
    if (not (std::abs(getDeltaPhiChange()) <= sdCut))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #3 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " deltaPhiChange: " << getDeltaPhiChange() <<  " sdCut: " << sdCut <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_SGAlgo);
        return;
    }

    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << SegmentSelection::slope);

//    float segmentDr = outerMiniDoubletAnchorHit.rt() - innerMiniDoubletAnchorHit.rt();
//    float sdZ = innerMiniDoubletAnchorHit.z();
//    float sdRt = innerMiniDoubletAnchorHit.rt();


    float inner_md_alpha = innerMiniDoublet.getDeltaPhiChange();
    float outer_md_alpha = outerMiniDoublet.getDeltaPhiChange();
    float sg_alpha = getDeltaPhiChange();

    std::unordered_map<std::string,float> dAlphaCutValues = dAlphaThreshold(innerMiniDoublet,outerMiniDoublet);
    float dAlpha_compat_inner_vs_sg = dAlphaCutValues["dAlphaInnerMDSegment"];
    float dAlpha_compat_outer_vs_sg = dAlphaCutValues["dAlphaOuterMDSegment"];
    float dAlpha_compat_inner_vs_outer = dAlphaCutValues["dAlphaInnerMDOuterMD"];

    setRecoVars("dAlpha_innerMD_segment",dAlpha_compat_inner_vs_sg);
    setRecoVars("dAlpha_outerMD_segment",dAlpha_compat_outer_vs_sg);
    setRecoVars("dAlpha_innerMD_outerMD",dAlpha_compat_inner_vs_outer);


    // Cut #4: angle compatibility between mini-doublet and segment
    float dAlpha_inner_md_sg = inner_md_alpha - sg_alpha;
    setdAlphaInnerMDSegment(dAlpha_inner_md_sg);

    if (not (std::abs(dAlpha_inner_md_sg) < dAlpha_compat_inner_vs_sg))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #4 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " dAlpha_inner_md_sg: " << dAlpha_inner_md_sg <<  " dAlpha_compat_inner_vs_sg: " << dAlpha_compat_inner_vs_sg <<  std::endl;
//            SDL::cout <<  " dAlpha_Bfield: " << dAlpha_Bfield <<  " dAlpha_res: " << dAlpha_res <<  " sdMuls: " << sdMuls <<  " sdLumForInnerMini: " << sdLumForInnerMini <<  std::endl;
            SDL::cout <<  " inner_md_alpha: " << inner_md_alpha <<  " sg_alpha: " << sg_alpha <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_SGAlgo);
        return;
    }

    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << SegmentSelection::alphaRef);

    // Cut #5: angle compatibility between mini-doublet and segment
    float dAlpha_outer_md_sg = outer_md_alpha - sg_alpha;
    setdAlphaOuterMDSegment(dAlpha_outer_md_sg);

    if (not (std::abs(dAlpha_outer_md_sg) < dAlpha_compat_outer_vs_sg))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #5 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " dAlpha_outer_md_sg: " << dAlpha_outer_md_sg <<  " dAlpha_compat_outer_vs_sg: " << dAlpha_compat_outer_vs_sg <<  std::endl;
//            SDL::cout <<  " dAlpha_Bfield: " << dAlpha_Bfield <<  " dAlpha_res: " << dAlpha_res <<  " sdMuls: " << sdMuls <<  " sdLumForOuterMini: " << sdLumForOuterMini <<  std::endl;
            SDL::cout <<  " outer_md_alpha: " << outer_md_alpha <<  " sg_alpha: " << sg_alpha <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_SGAlgo);
        return;
    }

    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << SegmentSelection::alphaOut);

    // Cut #6: angle compatibility between mini-doublet mini-doublets
    float dAlpha_outer_md_inner_md = outer_md_alpha - inner_md_alpha;
    setdAlphaOuterMDSegment(dAlpha_outer_md_inner_md);

    if (not (std::abs(dAlpha_outer_md_inner_md) < dAlpha_compat_inner_vs_outer))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #6 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " dAlpha_outer_md_inner_md: " << dAlpha_outer_md_inner_md <<  " dAlpha_compat_inner_vs_outer: " << dAlpha_compat_inner_vs_outer <<  std::endl;
//            SDL::cout <<  " dAlpha_Bfield: " << dAlpha_Bfield <<  " dAlpha_res: " << dAlpha_res <<  " sdMuls: " << sdMuls <<  " sdLumForInnerMini: " << sdLumForInnerMini <<  " sdLumForOuterMini: " << sdLumForOuterMini <<  std::endl;
            SDL::cout <<  " outer_md_alpha: " << outer_md_alpha <<  " inner_md_alpha: " << inner_md_alpha <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_SGAlgo);
        return;
    }

    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << SegmentSelection::alphaRefOut);

    passAlgo_ |= (1 << SDL::Default_SGAlgo);
    return;
}

void SDL::Segment::runSegmentDefaultAlgoEndcap(SDL::LogLevel logLevel)
{
    const MiniDoublet& innerMiniDoublet = (*innerMiniDoubletPtr());
    const MiniDoublet& outerMiniDoublet = (*outerMiniDoubletPtr());

    const Module& innerLowerModule = innerMiniDoublet.lowerHitPtr()->getModule();
    const Module& outerLowerModule = outerMiniDoublet.lowerHitPtr()->getModule();

    // Reset passBitsDefaultAlgo_;
    passBitsDefaultAlgo_ = 0;

    // Get connected outer lower module detids
    const std::vector<unsigned int>& connectedModuleDetIds = moduleConnectionMap.getConnectedModuleDetIds(innerLowerModule.detId());

    setRecoVars("sdCut", -999);
    setRecoVars("sdSlope", -999);
    setRecoVars("sdMuls", -999);
    setRecoVars("sdPVoff", -999);
    setRecoVars("deltaPhi", -999);

    setRecoVars("dAlpha_innerMD_segment",-999);
    setRecoVars("dAlpha_outerMD_segment",-999);
    setRecoVars("dAlpha_innerMD_outerMD",-999);
 

    // Loop over connected outer lower modules
    bool found = false;
    for (auto& outerLowerModuleDetId : connectedModuleDetIds)
    {
        if (outerLowerModule.detId() == outerLowerModuleDetId)
        {
            found = true;
            break;
        }
    }

    bool islayer2ring1or2or3 = (innerLowerModule.ring() == 1 or innerLowerModule.ring() == 2 or innerLowerModule.ring() == 3) and innerLowerModule.layer() == 2;

    // Cut #0: Module compatibility
    if (not (found or islayer2ring1or2or3))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #0 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " innerLowerModule.detId(): " << innerLowerModule.detId() <<  " outerLowerModule.detId(): " << outerLowerModule.detId() << std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_SGAlgo);
        return;
    }

    // Flag the pass bit
    passBitsDefaultAlgo_ |= (1 << SegmentSelection::moduleCompatible);

    // Constants
    const float kRinv1GeVf = (2.99792458e-3 * 3.8);
    const float k2Rinv1GeVf = kRinv1GeVf / 2.;
    const float ptCut = PTCUT;
    const float sinAlphaMax = 0.95;
    const float deltaZLum = 15.f;
    std::array<float, 5> miniMulsPtScaleEndcap {0.006, 0.006, 0.006, 0.006, 0.006}; //inter/extra-polated from L11 and L13 both roughly 0.006 [larger R have smaller value by ~50%]

    float sdMuls;

    sdMuls = miniMulsPtScaleEndcap[innerLowerModule.layer()-1] * 3.f / ptCut;//will need a better guess than x2?
    

    // Get the relevant anchor hits
    const Hit& innerMiniDoubletAnchorHit = *innerMiniDoublet.anchorHitPtr();
    const Hit& outerMiniDoubletAnchorHit = *outerMiniDoublet.anchorHitPtr();
    const bool outerLayerEndcapTwoS = outerLowerModule.moduleType() == SDL::Module::TwoS and outerLowerModule.subdet() == SDL::Module::Endcap;
    const Hit& outerMiniDoubletAnchorHitHighEdge = outerLayerEndcapTwoS ? *(outerMiniDoublet.anchorHitPtr()->getHitHighEdgePtr()) : *outerMiniDoublet.anchorHitPtr();
    const Hit& outerMiniDoubletAnchorHitLowEdge  = outerLayerEndcapTwoS ? *(outerMiniDoublet.anchorHitPtr()->getHitLowEdgePtr()) : *outerMiniDoublet.anchorHitPtr();

    // MiniDoublet information
    float innerMiniDoubletAnchorHitRt = innerMiniDoubletAnchorHit.rt();
    float outerMiniDoubletAnchorHitRt = outerMiniDoubletAnchorHit.rt();
    float innerMiniDoubletAnchorHitZ = innerMiniDoubletAnchorHit.z();
    float outerMiniDoubletAnchorHitZ = outerMiniDoubletAnchorHit.z();

    // Reco value set
    setRtOut(outerMiniDoubletAnchorHitRt);
    setRtIn(innerMiniDoubletAnchorHitRt);

    const float sdSlope = std::asin(std::min(outerMiniDoubletAnchorHitRt * k2Rinv1GeVf / ptCut, sinAlphaMax));
     const float sdPVoff = 0.1f / outerMiniDoubletAnchorHitRt;
    // const float dzDrtScale = std::tan(sdSlope) / sdSlope; //FIXME: need approximate value
    const float pixelPSZpitch = 0.15;
    const float strip2SZpitch = 5.0;
    const float disks2SMinRadius = 60.f;

    // const float zGeom = innerLowerModule.layer() <= 2 ? 2.f * pixelPSZpitch : 2.f * strip2SZpitch; //twice the macro-pixel or strip size

    const float rtGeom = ((innerMiniDoubletAnchorHitRt < disks2SMinRadius && outerMiniDoubletAnchorHitRt < disks2SMinRadius) ? (2.f * pixelPSZpitch)
            : ((innerMiniDoubletAnchorHitRt < disks2SMinRadius || outerMiniDoubletAnchorHitRt < disks2SMinRadius) ? (pixelPSZpitch + strip2SZpitch)
            : (2.f * strip2SZpitch)));

    // float zLo = innerMiniDoubletAnchorHitZ + (innerMiniDoubletAnchorHitZ - deltaZLum) * (outerMiniDoubletAnchorHitRt / innerMiniDoubletAnchorHitRt - 1.f) * (innerMiniDoubletAnchorHitZ > 0.f ? 1.f : dzDrtScale) - zGeom; //slope-correction only on outer end
    // float zHi = innerMiniDoubletAnchorHitZ + (innerMiniDoubletAnchorHitZ + deltaZLum) * (outerMiniDoubletAnchorHitRt / innerMiniDoubletAnchorHitRt - 1.f) * (innerMiniDoubletAnchorHitZ < 0.f ? 1.f : dzDrtScale) + zGeom;

    // Cut #0: preliminary cut (if the combo is between negative and positive don't even bother...)
    if (innerMiniDoubletAnchorHitZ * outerMiniDoubletAnchorHitZ < 0)
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #0 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " innerMiniDoubletAnchorHitZ: " << innerMiniDoubletAnchorHitZ <<  " outerMiniDoubletAnchorHitZ: " << outerMiniDoubletAnchorHitZ <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_SGAlgo);
        return; //do not even accumulate stats for wrong side combinations
    }

    const float dz = outerMiniDoubletAnchorHitZ - innerMiniDoubletAnchorHitZ;

    // Cut #1: Z compatibility
    const float dLum = std::copysign(deltaZLum, innerMiniDoubletAnchorHitZ);
    const float drtDzScale = sdSlope / std::tan(sdSlope); //FIXME: need approximate value
    float rtLo = std::max(innerMiniDoubletAnchorHitRt * (1.f + dz / (innerMiniDoubletAnchorHitZ + dLum) * drtDzScale) - rtGeom, innerMiniDoubletAnchorHitRt - 0.5f * rtGeom); //rt should increase
    float rtHi = innerMiniDoubletAnchorHitRt * (outerMiniDoubletAnchorHitZ - dLum) / (innerMiniDoubletAnchorHitZ - dLum) + rtGeom; //dLum for luminous; rGeom for measurement size; no tanTheta_loc(pt) correction


    setRtLo(rtLo);
    setRtHi(rtHi);
    if (not (outerMiniDoubletAnchorHitRt >= rtLo and outerMiniDoubletAnchorHitRt <= rtHi))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #1 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " rtLo: " << rtLo <<  " outerMiniDoubletAnchorHitRt: " << outerMiniDoubletAnchorHitRt <<  " rtHi: " << rtHi <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_SGAlgo);
        return;
    }

    // Cut #2: dPhi compatibility
    // const float sdLum = deltaZLum / std::abs(innerMiniDoubletAnchorHitZ);
    // const float sdCut = sdSlope + sqrt(sdMuls * sdMuls + sdPVoff * sdPVoff + sdLum * sdLum);
    //const float sdCut = sdSlope;
    const float dPhiPos = innerMiniDoubletAnchorHit.deltaPhi(outerMiniDoubletAnchorHit);
    const float sdLum = dPhiPos * deltaZLum / dz; 
    const float sdCut = sdSlope;	
    const float dPhiPos_high = innerMiniDoubletAnchorHit.deltaPhi(outerMiniDoubletAnchorHitHighEdge);
    const float dPhiPos_low = innerMiniDoubletAnchorHit.deltaPhi(outerMiniDoubletAnchorHitLowEdge);
    const float dPhiPos_max = abs(dPhiPos_high) > abs(dPhiPos_low) ? dPhiPos_high : dPhiPos_low;
    const float dPhiPos_min = abs(dPhiPos_high) > abs(dPhiPos_low) ? dPhiPos_low : dPhiPos_high;

    setRecoVars("sdCut",sdCut);
    setRecoVars("sdSlope",sdSlope);
    setRecoVars("deltaPhi",dPhiPos);

    setDeltaPhi(dPhiPos);
    setDeltaPhiMin(dPhiPos_min);
    setDeltaPhiMax(dPhiPos_max);

    if (not (std::abs(dPhiPos) <= sdCut))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #2 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " dPhiPos: " << dPhiPos <<  " sdCut: " << sdCut <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_SGAlgo);
        return;
    }

    // Cut #3: dPhi compatibility
    // const float dPhiChange = innerMiniDoubletAnchorHit.deltaPhiChange(outerMiniDoubletAnchorHit); // NOTE When using the full r3 coordinate (this was turned off in slava's code)
    const float dzFrac = dz / innerMiniDoubletAnchorHitZ;
    // const float dPhiChange = dPhiPos / dzFrac * (1.f + dzFrac);
    setDeltaPhiChange(dPhiPos / dzFrac * (1.f + dzFrac));
    setDeltaPhiMinChange(dPhiPos_min / dzFrac * (1.f + dzFrac));
    setDeltaPhiMaxChange(dPhiPos_max / dzFrac * (1.f + dzFrac));

    if (not (std::abs(getDeltaPhiChange()) <= sdCut))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #3 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " dPhiChange: " << getDeltaPhiChange() <<  " sdCut: " << sdCut <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_SGAlgo);
        return;
    }

//    float segmentDr = outerMiniDoubletAnchorHit.rt() - innerMiniDoubletAnchorHit.rt();
//    float sdZ = innerMiniDoubletAnchorHit.z();
//    float sdRt = innerMiniDoubletAnchorHit.rt();
    // float sdZ_outer = outerMiniDoubletAnchorHit.z();
    // float sdRt_outer = outerMiniDoubletAnchorHit.rt();


    float inner_md_alpha = innerMiniDoublet.getDeltaPhiChange();
    float outer_md_alpha = outerMiniDoublet.getDeltaPhiChange();
    float sg_alpha = getDeltaPhiChange();
    
    std::unordered_map<std::string,float> dAlphaCutValues = dAlphaThreshold(innerMiniDoublet,outerMiniDoublet);

    float dAlpha_compat_inner_vs_sg = dAlphaCutValues["dAlphaInnerMDSegment"];
    float dAlpha_compat_outer_vs_sg = dAlphaCutValues["dAlphaOuterMDSegment"];
    float dAlpha_compat_inner_vs_outer = dAlphaCutValues["dAlphaInnerMDOuterMD"];

    setRecoVars("dAlpha_innerMD_segment",dAlpha_compat_inner_vs_sg);
    setRecoVars("dAlpha_outerMD_segment",dAlpha_compat_outer_vs_sg);
    setRecoVars("dAlpha_innerMD_outerMD",dAlpha_compat_inner_vs_outer);


    // Cut #4: angle compatibility between mini-doublet and segment
    float dAlpha_inner_md_sg = inner_md_alpha - sg_alpha;
    setdAlphaInnerMDSegment(dAlpha_inner_md_sg);

    if (not (std::abs(dAlpha_inner_md_sg) < dAlpha_compat_inner_vs_sg))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #4 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " dAlpha_inner_md_sg: " << dAlpha_inner_md_sg <<  " dAlpha_compat_inner_vs_sg: " << dAlpha_compat_inner_vs_sg <<  std::endl;
            SDL::cout <<  " inner_md_alpha: " << inner_md_alpha <<  " sg_alpha: " << sg_alpha <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_SGAlgo);
        return;
    }

    // Cut #5: angle compatibility between mini-doublet and segment
    float dAlpha_outer_md_sg = outer_md_alpha - sg_alpha;
    setdAlphaOuterMDSegment(dAlpha_outer_md_sg);

    if (not (std::abs(dAlpha_outer_md_sg) < dAlpha_compat_outer_vs_sg))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #5 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " dAlpha_outer_md_sg: " << dAlpha_outer_md_sg <<  " dAlpha_compat_outer_vs_sg: " << dAlpha_compat_outer_vs_sg <<  std::endl;
            SDL::cout <<  " outer_md_alpha: " << outer_md_alpha <<  " sg_alpha: " << sg_alpha <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_SGAlgo);
        return;
    }

    // Cut #6: angle compatibility between mini-doublet mini-doublets
    float dAlpha_outer_md_inner_md = outer_md_alpha - inner_md_alpha;
    setdAlphaInnerMDOuterMD(dAlpha_outer_md_inner_md);

    if (not (std::abs(dAlpha_outer_md_inner_md) < dAlpha_compat_inner_vs_outer))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #6 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " dAlpha_outer_md_inner_md: " << dAlpha_outer_md_inner_md <<  " dAlpha_compat_inner_vs_outer: " << dAlpha_compat_inner_vs_outer <<  std::endl;
//            SDL::cout <<  " dAlpha_Bfield: " << dAlpha_Bfield <<  " dAlpha_res: " << dAlpha_res <<  " sdMuls: " << sdMuls <<  " dAlpha_uncRt: " << dAlpha_uncRt <<  " sdLumForInnerMini: " << sdLumForInnerMini <<  " sdLumForOuterMini: " << sdLumForOuterMini <<  std::endl;
//            SDL::cout <<  " sdZ: " << sdZ <<  " sdRt: " << sdRt <<  " miniDelta: " << miniDelta <<  std::endl;
//            SDL::cout <<  " segmentDr: " << segmentDr <<  " k2Rinv1GeVf: " << k2Rinv1GeVf <<  " ptCut: " << ptCut <<  " sinAlphaMax: " << sinAlphaMax <<  std::endl;
            SDL::cout <<  " outer_md_alpha: " << outer_md_alpha <<  " inner_md_alpha: " << inner_md_alpha <<  std::endl;
        }
        passAlgo_ &= (0 << SDL::Default_SGAlgo);
        return;
    }

    passAlgo_ |= (1 << SDL::Default_SGAlgo);
    return;
}

bool SDL::Segment::hasCommonMiniDoublet(const Segment& outer_sg) const
{
    if (not outerMiniDoubletPtr()->isIdxMatched(*(outer_sg.innerMiniDoubletPtr())))
        return false;
    return true;
}

bool SDL::Segment::isIdxMatched(const Segment& sg) const
{
    if (not innerMiniDoubletPtr_->isIdxMatched(*(sg.innerMiniDoubletPtr())))
        return false;
    if (not outerMiniDoubletPtr_->isIdxMatched(*(sg.outerMiniDoubletPtr())))
        return false;
    return true;
}

bool SDL::Segment::isAnchorHitIdxMatched(const Segment& sg) const
{
    if (not innerMiniDoubletPtr_->isAnchorHitIdxMatched(*(sg.innerMiniDoubletPtr())))
        return false;
    if (not outerMiniDoubletPtr_->isAnchorHitIdxMatched(*(sg.outerMiniDoubletPtr())))
        return false;
    return true;
}

std::unordered_map<std::string,float> SDL::Segment::dAlphaThreshold(const SDL::MiniDoublet &innerMiniDoublet, const SDL::MiniDoublet &outerMiniDoublet)
{

    std::unordered_map<std::string,float> dAlphaValues;
    const Module& innerLowerModule = innerMiniDoublet.lowerHitPtr()->getModule();
    const Module& outerLowerModule = outerMiniDoublet.lowerHitPtr()->getModule();
    const float kRinv1GeVf = (2.99792458e-3 * 3.8);
    const float k2Rinv1GeVf = kRinv1GeVf / 2.;
   
    float ptCut = PTCUT;
    float sinAlphaMax = 0.95;

    std::array<float, 6> miniMulsPtScaleBarrel {0.0052, 0.0038, 0.0034, 0.0034, 0.0032, 0.0034};
    std::array<float, 5> miniMulsPtScaleEndcap {0.006, 0.006, 0.006, 0.006, 0.006}; //inter/extra-polated from L11 and L13 both roughly 0.006 [larger R have smaller value by ~50%]

    
    float sdMuls = (innerLowerModule.subdet() == SDL::Module::Barrel) ? miniMulsPtScaleBarrel[innerLowerModule.layer()-1] * 3.f/ptCut : miniMulsPtScaleEndcap[innerLowerModule.layer()-1] * 3.f/ptCut;

    // BField dAlpha
    const Hit& innerMiniDoubletAnchorHit = (innerLowerModule.moduleType() == SDL::Module::PS) ? ( (innerLowerModule.moduleLayerType() == SDL::Module::Pixel) ? *innerMiniDoublet.lowerHitPtr() : *innerMiniDoublet.upperHitPtr()): *innerMiniDoublet.lowerHitPtr();
    const Hit& outerMiniDoubletAnchorHit = (outerLowerModule.moduleType() == SDL::Module::PS) ? ( (outerLowerModule.moduleLayerType() == SDL::Module::Pixel) ? *outerMiniDoublet.lowerHitPtr() : *outerMiniDoublet.upperHitPtr()): *outerMiniDoublet.lowerHitPtr();

    float innerMiniDoubletAnchorHitRt = innerMiniDoubletAnchorHit.rt();
    float outerMiniDoubletAnchorHitRt = outerMiniDoubletAnchorHit.rt();
    float innerMiniDoubletAnchorHitZ = innerMiniDoubletAnchorHit.z();
    float outerMiniDoubletAnchorHitZ = outerMiniDoubletAnchorHit.z();

    //more accurate then outer rt - inner rt

//    float segmentDr = sqrt(pow(outerMiniDoubletAnchorHit.y() - innerMiniDoubletAnchorHit.y(),2) + pow(outerMiniDoubletAnchorHit.x() - innerMiniDoubletAnchorHit.x(),2));
    
    float segmentDr = (outerMiniDoubletAnchorHit - innerMiniDoubletAnchorHit).rt();

    const float dAlpha_Bfield = std::asin(std::min(segmentDr * k2Rinv1GeVf/ptCut, sinAlphaMax));
    const float pixelPSZpitch = 0.15;
//    const float innersdPVoff = 0.1f / innerMiniDoubletAnchorHitRt;
//    const float outersdPVoff = 0.1f/ innerMiniDoubletAnchorHitRt;
//    const float sdPVoff = 0.1f/segmentDr;


    const bool isInnerTilted = innerLowerModule.subdet() == SDL::Module::Barrel and innerLowerModule.side() != SDL::Module::Center;
    const bool isOuterTilted = outerLowerModule.subdet() == SDL::Module::Barrel and outerLowerModule.side() != SDL::Module::Center;
    const unsigned int innerdetid = (innerLowerModule.moduleLayerType() == SDL::Module::Pixel) ?  innerLowerModule.partnerDetId() : innerLowerModule.detId();
    const unsigned int outerdetid = (outerLowerModule.moduleLayerType() == SDL::Module::Pixel) ?  outerLowerModule.partnerDetId() : outerLowerModule.detId();
    const float drdzinner = tiltedGeometry.getDrDz(innerdetid);
    const float drdzouter = tiltedGeometry.getDrDz(outerdetid);
    const float innerminiTilt = isInnerTilted ? (0.5f * pixelPSZpitch * drdzinner / sqrt(1.f + drdzinner * drdzinner) / SDL::MiniDoublet::moduleGapSize(innerLowerModule)) : 0;
    const float outerminiTilt = isOuterTilted ? (0.5f * pixelPSZpitch * drdzouter / sqrt(1.f + drdzouter * drdzouter) / SDL::MiniDoublet::moduleGapSize(outerLowerModule)) : 0;


    float miniDelta = SDL::MiniDoublet::moduleGapSize(innerLowerModule); 
 

    float sdLumForInnerMini;    
    float sdLumForOuterMini;

    if (innerLowerModule.subdet() == SDL::Module::Barrel)
    {
        sdLumForInnerMini = innerminiTilt * dAlpha_Bfield;
    }
    else
    {
        if (innerLowerModule.moduleType() == SDL::Module::PS)
            sdLumForInnerMini = innerMiniDoublet.getDeltaPhi() * 15.0 / innerMiniDoublet.getShiftedDz();
        else
            sdLumForInnerMini = innerMiniDoublet.getDeltaPhi() * 15.0 / innerMiniDoublet.getDz();
    }

    if (outerLowerModule.subdet() == SDL::Module::Barrel)
    {
        sdLumForOuterMini = outerminiTilt * dAlpha_Bfield;
    }
    else
    {
        if (outerLowerModule.moduleType() == SDL::Module::PS)
            sdLumForOuterMini = outerMiniDoublet.getDeltaPhi() * 15.0 / outerMiniDoublet.getShiftedDz();
        else
            sdLumForOuterMini = outerMiniDoublet.getDeltaPhi() * 15.0 / outerMiniDoublet.getDz();
    }


    //Unique stuff for the segment dudes alone

    float dAlpha_res_inner = 0.02f/miniDelta * (innerLowerModule.subdet() == SDL::Module::Barrel ? 1.0f : std::abs(innerMiniDoubletAnchorHitZ/innerMiniDoubletAnchorHitRt));
    float dAlpha_res_outer = 0.02f/miniDelta * (outerLowerModule.subdet() == SDL::Module::Barrel ? 1.0f : std::abs(outerMiniDoubletAnchorHitZ/outerMiniDoubletAnchorHitRt));

    float dAlpha_res = dAlpha_res_inner + dAlpha_res_outer;

    if (innerLowerModule.subdet() == SDL::Module::Barrel and innerLowerModule.side() == SDL::Module::Center)
    {
        dAlphaValues["dAlphaInnerMDSegment"] = dAlpha_Bfield + sqrt(pow(dAlpha_res,2) + pow(sdMuls,2));
    }
    else
    {
        dAlphaValues["dAlphaInnerMDSegment"] = dAlpha_Bfield + sqrt(pow(dAlpha_res,2) + pow(sdMuls,2) + pow(sdLumForInnerMini,2));
    }

    if (outerLowerModule.subdet() == SDL::Module::Barrel and outerLowerModule.side() == SDL::Module::Center)
    {
        dAlphaValues["dAlphaOuterMDSegment"] = dAlpha_Bfield + sqrt(pow(dAlpha_res,2) + pow(sdMuls,2));
    }
    else
    {
        dAlphaValues["dAlphaOuterMDSegment"] = dAlpha_Bfield + sqrt(pow(dAlpha_res,2) + pow(sdMuls,2) + pow(sdLumForOuterMini,2));
    }

    //Inner to outer 
    dAlphaValues["dAlphaInnerMDOuterMD"] = dAlpha_Bfield + sqrt(pow(dAlpha_res,2) + pow(sdMuls,2));

    return dAlphaValues;
}

[[deprecated("SDL:: isMiniDoubletPairASegment() is deprecated")]]
bool SDL::Segment::isMiniDoubletPairASegment(const MiniDoublet& innerMiniDoublet, const MiniDoublet& outerMiniDoublet, SGAlgo algo, SDL::LogLevel logLevel)
{
    // If the algorithm is "do all combination" (e.g. used for efficiency calculation)
    if (algo == SDL::AllComb_SGAlgo)
    {
        return true;
    }
    else if (algo == SDL::Default_SGAlgo)
    {
        const Module& innerLowerModule = innerMiniDoublet.lowerHitPtr()->getModule();
        // const Module& outerLowerModule = outerMiniDoublet.lowerHitPtr()->getModule();
        // Port your favorite segment formation algorithm code here
        // Case 1: Barrel - Barrel
        // if (innerLowerModule.subdet() == SDL::Module::Barrel and outerLowerModule.subdet() == SDL::Module::Barrel)
        if (innerLowerModule.subdet() == SDL::Module::Barrel)
        {
            if (not isMiniDoubletPairASegmentCandidateBarrel(innerMiniDoublet, outerMiniDoublet, algo, logLevel))
                return false;
        }
        else // if (innerLowerModule.subdet() == SDL::Module::Endcap)
        {
            if (not isMiniDoubletPairASegmentCandidateEndcap(innerMiniDoublet, outerMiniDoublet, algo, logLevel))
                return false;
        }
        return true;
    }
    else
    {
        SDL::cout << "Warning: Unrecognized segment algorithm!" << algo << std::endl;
        return false;
    }
}

bool SDL::Segment::isMiniDoubletPairASegmentCandidateBarrel(const MiniDoublet& innerMiniDoublet, const MiniDoublet& outerMiniDoublet, SGAlgo algo, SDL::LogLevel logLevel)
{

    const Module& innerLowerModule = innerMiniDoublet.lowerHitPtr()->getModule();
    const Module& outerLowerModule = outerMiniDoublet.lowerHitPtr()->getModule();

    // Constants
    const float kRinv1GeVf = (2.99792458e-3 * 3.8);
    const float k2Rinv1GeVf = kRinv1GeVf / 2.;
    const float ptCut = PTCUT;
    const float sinAlphaMax = 0.95;
    const float deltaZLum = 15.f;
    std::array<float, 6> miniMulsPtScaleBarrel {0.0052, 0.0038, 0.0034, 0.0034, 0.0032, 0.0034};
    // std::array<float, 5> miniMulsPtScaleEndcap {0.006, 0.006, 0.006, 0.006, 0.006}; //inter/extra-polated from L11 and L13 both roughly 0.006 [larger R have smaller value by ~50%]
    const float sdMuls = miniMulsPtScaleBarrel[innerLowerModule.layer()] * 3.f / ptCut * 2.f;//will need a better guess than x2?

    // Get the relevant anchor hits
    const Hit& innerMiniDoubletAnchorHit = innerLowerModule.moduleType() == SDL::Module::PS ? ( (innerLowerModule.moduleLayerType() == SDL::Module::Pixel) ? *innerMiniDoublet.lowerHitPtr() : *innerMiniDoublet.upperHitPtr()): *innerMiniDoublet.lowerHitPtr();
    const Hit& outerMiniDoubletAnchorHit = outerLowerModule.moduleType() == SDL::Module::PS ? ( (outerLowerModule.moduleLayerType() == SDL::Module::Pixel) ? *outerMiniDoublet.lowerHitPtr() : *outerMiniDoublet.upperHitPtr()): *outerMiniDoublet.lowerHitPtr();

    // MiniDoublet information
    float innerMiniDoubletAnchorHitRt = innerMiniDoubletAnchorHit.rt();
    float outerMiniDoubletAnchorHitRt = outerMiniDoubletAnchorHit.rt();
    float innerMiniDoubletAnchorHitZ = innerMiniDoubletAnchorHit.z();
    float outerMiniDoubletAnchorHitZ = outerMiniDoubletAnchorHit.z();

    const float sdSlope = std::asin(std::min(outerMiniDoubletAnchorHitRt * k2Rinv1GeVf / ptCut, sinAlphaMax));
    const float sdPVoff = 0.1f / outerMiniDoubletAnchorHitRt;
    const float dzDrtScale = std::tan(sdSlope) / sdSlope; //FIXME: need approximate value
    const float pixelPSZpitch = 0.15;
    const float strip2SZpitch = 5.0;

    const float zGeom = (innerLowerModule.layer() <= 2) ? (2.f * pixelPSZpitch) : (2.f * strip2SZpitch); //twice the macro-pixel or strip size

    float zLo = innerMiniDoubletAnchorHitZ + (innerMiniDoubletAnchorHitZ - deltaZLum) * (outerMiniDoubletAnchorHitRt / innerMiniDoubletAnchorHitRt - 1.f) * (innerMiniDoubletAnchorHitZ > 0.f ? 1.f : dzDrtScale) - zGeom; //slope-correction only on outer end
    float zHi = innerMiniDoubletAnchorHitZ + (innerMiniDoubletAnchorHitZ + deltaZLum) * (outerMiniDoubletAnchorHitRt / innerMiniDoubletAnchorHitRt - 1.f) * (innerMiniDoubletAnchorHitZ < 0.f ? 1.f : dzDrtScale) + zGeom;

    // Cut #1: Z compatibility
    if (not (outerMiniDoubletAnchorHitZ >= zLo and outerMiniDoubletAnchorHitZ <= zHi))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #1 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " zLo: " << zLo <<  " outerMiniDoubletAnchorHitZ: " << outerMiniDoubletAnchorHitZ <<  " zHi: " << zHi <<  std::endl;
        }
        return false;
    }

    const float sdCut = sdSlope + sqrt(sdMuls * sdMuls + sdPVoff * sdPVoff);
    const float deltaPhi = innerMiniDoubletAnchorHit.deltaPhi(outerMiniDoubletAnchorHit);

    // Cut #2: phi differences between the two minidoublets
    if (not (std::abs(deltaPhi) <= sdCut))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #2 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " deltaPhi: " << deltaPhi <<  " sdCut: " << sdCut <<  std::endl;
        }
        return false;
    }

    float dPhiChange = innerMiniDoubletAnchorHit.deltaPhiChange(outerMiniDoubletAnchorHit);

    // Cut #3: phi change between the two minidoublets
    if (not (std::abs(dPhiChange) <= sdCut))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #3 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " deltaPhiChange: " << dPhiChange <<  " sdCut: " << sdCut <<  std::endl;
        }
        return false;
    }

    return true;
}

bool SDL::Segment::isMiniDoubletPairASegmentCandidateEndcap(const MiniDoublet& innerMiniDoublet, const MiniDoublet& outerMiniDoublet, SGAlgo algo, SDL::LogLevel logLevel)
{

    const Module& innerLowerModule = innerMiniDoublet.lowerHitPtr()->getModule();
    const Module& outerLowerModule = outerMiniDoublet.lowerHitPtr()->getModule();

    // Constants
    const float kRinv1GeVf = (2.99792458e-3 * 3.8);
    const float k2Rinv1GeVf = kRinv1GeVf / 2.;
    const float ptCut = PTCUT;
    const float sinAlphaMax = 0.95;
    const float deltaZLum = 15.f;
    // std::array<float, 6> miniMulsPtScaleBarrel {0.0052, 0.0038, 0.0034, 0.0034, 0.0032, 0.0034};
    // std::array<float, 5> miniMulsPtScaleEndcap {0.006, 0.006, 0.006, 0.006, 0.006}; //inter/extra-polated from L11 and L13 both roughly 0.006 [larger R have smaller value by ~50%]
    // const float sdMuls = miniMulsPtScaleEndcap[innerLowerModule.layer()] * 3.f / ptCut * 2.f;//will need a better guess than x2?

    // Get the relevant anchor hits
    const Hit& innerMiniDoubletAnchorHit = innerLowerModule.moduleType() == SDL::Module::PS ? ( innerLowerModule.moduleLayerType() == SDL::Module::Pixel ? *innerMiniDoublet.lowerHitPtr() : *innerMiniDoublet.upperHitPtr()): *innerMiniDoublet.lowerHitPtr();
    const Hit& outerMiniDoubletAnchorHit = outerLowerModule.moduleType() == SDL::Module::PS ? ( outerLowerModule.moduleLayerType() == SDL::Module::Pixel ? *outerMiniDoublet.lowerHitPtr() : *outerMiniDoublet.upperHitPtr()): *outerMiniDoublet.lowerHitPtr();

    // MiniDoublet information
    float innerMiniDoubletAnchorHitRt = innerMiniDoubletAnchorHit.rt();
    float outerMiniDoubletAnchorHitRt = outerMiniDoubletAnchorHit.rt();
    float innerMiniDoubletAnchorHitZ = innerMiniDoubletAnchorHit.z();
    float outerMiniDoubletAnchorHitZ = outerMiniDoubletAnchorHit.z();

    const float sdSlope = std::asin(std::min(outerMiniDoubletAnchorHitRt * k2Rinv1GeVf / ptCut, sinAlphaMax));
    // const float sdPVoff = 0.1f / outerMiniDoubletAnchorHitRt;
    // const float dzDrtScale = std::tan(sdSlope) / sdSlope; //FIXME: need approximate value
    const float pixelPSZpitch = 0.15;
    const float strip2SZpitch = 5.0;
    const float disks2SMinRadius = 60.f;

    // const float zGeom = innerLowerModule.layer() <= 2 ? 2.f * pixelPSZpitch : 2.f * strip2SZpitch; //twice the macro-pixel or strip size

    const float rtGeom = ((innerMiniDoubletAnchorHitRt < disks2SMinRadius && outerMiniDoubletAnchorHitRt < disks2SMinRadius) ? (2.f * pixelPSZpitch)
            : ((innerMiniDoubletAnchorHitRt < disks2SMinRadius || outerMiniDoubletAnchorHitRt < disks2SMinRadius) ? (pixelPSZpitch + strip2SZpitch)
            : (2.f * strip2SZpitch)));

    // float zLo = innerMiniDoubletAnchorHitZ + (innerMiniDoubletAnchorHitZ - deltaZLum) * (outerMiniDoubletAnchorHitRt / innerMiniDoubletAnchorHitRt - 1.f) * (innerMiniDoubletAnchorHitZ > 0.f ? 1.f : dzDrtScale) - zGeom; //slope-correction only on outer end
    // float zHi = innerMiniDoubletAnchorHitZ + (innerMiniDoubletAnchorHitZ + deltaZLum) * (outerMiniDoubletAnchorHitRt / innerMiniDoubletAnchorHitRt - 1.f) * (innerMiniDoubletAnchorHitZ < 0.f ? 1.f : dzDrtScale) + zGeom;

    // Cut #0: preliminary cut (if the combo is between negative and positive don't even bother...)
    if (innerMiniDoubletAnchorHitZ * outerMiniDoubletAnchorHitZ < 0)
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #0 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " innerMiniDoubletAnchorHitZ: " << innerMiniDoubletAnchorHitZ <<  " outerMiniDoubletAnchorHitZ: " << outerMiniDoubletAnchorHitZ <<  std::endl;
        }
        return false; //do not even accumulate stats for wrong side combinations
    }

    const float dz = outerMiniDoubletAnchorHitZ - innerMiniDoubletAnchorHitZ;

    // Cut #1: Z compatibility
    const float dLum = std::copysign(deltaZLum, innerMiniDoubletAnchorHitZ);
    const float drtDzScale = sdSlope / std::tan(sdSlope); //FIXME: need approximate value
    float rtLo = std::max(innerMiniDoubletAnchorHitRt * (1.f + dz / (innerMiniDoubletAnchorHitZ + dLum) * drtDzScale) - rtGeom, innerMiniDoubletAnchorHitRt - 0.5f * rtGeom); //rt should increase
    float rtHi = innerMiniDoubletAnchorHitRt * (outerMiniDoubletAnchorHitZ - dLum) / (innerMiniDoubletAnchorHitZ - dLum) + rtGeom; //dLum for luminous; rGeom for measurement size; no tanTheta_loc(pt) correction
    if (not (outerMiniDoubletAnchorHitRt >= rtLo and outerMiniDoubletAnchorHitRt <= rtHi))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #1 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " rtLo: " << rtLo <<  " outerMiniDoubletAnchorHitRt: " << outerMiniDoubletAnchorHitRt <<  " rtHi: " << rtHi <<  std::endl;
        }
        return false;
    }

    // Cut #2: dPhi compatibility
    // const float sdLum = deltaZLum / std::abs(innerMiniDoubletAnchorHitZ);
    // const float sdCut = sdSlope + sqrt(sdMuls * sdMuls + sdPVoff * sdPVoff + sdLum * sdLum);
    const float sdCut = sdSlope;
    const float dPhiPos = innerMiniDoubletAnchorHit.deltaPhi(outerMiniDoubletAnchorHit);


    if (not (std::abs(dPhiPos) <= sdCut))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #2 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " dPhiPos: " << dPhiPos <<  " sdCut: " << sdCut <<  std::endl;
        }
        return false;
    }

    // Cut #3: dPhi compatibility
    // const float dPhiChange = innerMiniDoubletAnchorHit.deltaPhiChange(outerMiniDoubletAnchorHit); // NOTE When using the full r3 coordinate (this was turned off in slava's code)
    const float dzFrac = dz / innerMiniDoubletAnchorHitZ;
    const float dPhiChange = dPhiPos / dzFrac * (1.f + dzFrac);
    if (not (std::abs(dPhiChange) <= sdCut))
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << "Failed Cut #3 in " << __FUNCTION__ << std::endl;
            SDL::cout <<  " dPhiChange: " << dPhiChange <<  " sdCut: " << sdCut <<  std::endl;
        }
        return false;
    }

    return true;
}


namespace SDL
{
    std::ostream& operator<<(std::ostream& out, const Segment& sg)
    {
        out << "sg_dPhiChange " << sg.getDeltaPhiChange() << std::endl;
        out << "ptestimate " << SDL::MathUtil::ptEstimateFromDeltaPhiChangeAndRt(sg.getDeltaPhiChange(), sg.getRtOut()) << std::endl;
        out << std::endl;
        out << "Inner Mini-Doublet" << std::endl;
        out << "------------------------------" << std::endl;
        {
            IndentingOStreambuf indent(out);
            out << sg.innerMiniDoubletPtr_ << std::endl;
        }
        out << "Outer Mini-Doublet" << std::endl;
        out << "------------------------------" << std::endl;
        {
            IndentingOStreambuf indent(out);
            out << sg.outerMiniDoubletPtr_;
        }
        // out << "Inner MD Module " << std::endl;
        // out << sg.innerMiniDoubletPtr_->lowerHitPtr()->getModule();
        // out << "outer MD Module " << std::endl;
        // out << sg.outerMiniDoubletPtr_->lowerHitPtr()->getModule();
        return out;
    }

    std::ostream& operator<<(std::ostream& out, const Segment* sg)
    {
        out << *sg;
        return out;
    }
}


