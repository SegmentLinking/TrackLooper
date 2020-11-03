#include "MiniDoublet.h"

#define SDL_INF 123456789

SDL::MiniDoublet::MiniDoublet()
{
}

SDL::MiniDoublet::~MiniDoublet()
{
}

SDL::MiniDoublet::MiniDoublet(const MiniDoublet& md): lowerHitPtr_(md.lowerHitPtr()), upperHitPtr_(md.upperHitPtr())
                                                      ,passAlgo_(md.getPassAlgo())
                                                      ,lowerShiftedHit_(md.getLowerShiftedHit())
                                                      ,upperShiftedHit_(md.getUpperShiftedHit())
                                                      ,dz_(md.getDz())
                                                      ,shiftedDz_(md.getShiftedDz())
                                                      ,dphi_(md.getDeltaPhi())
                                                      ,dphi_noshift_(md.getDeltaPhiNoShift())
                                                      ,dphichange_(md.getDeltaPhiChange())
                                                      ,dphichange_noshift_(md.getDeltaPhiChangeNoShift())
                                                      ,recovars_(md.getRecoVars())
{
    setAnchorHit();
}

SDL::MiniDoublet::MiniDoublet(SDL::Hit* lowerHitPtr, SDL::Hit* upperHitPtr) : lowerHitPtr_(lowerHitPtr), upperHitPtr_(upperHitPtr)
                                                      ,passAlgo_(0)
                                                      ,dz_(0)
                                                      ,shiftedDz_(0)
                                                      ,dphi_(0)
                                                      ,dphi_noshift_(0)
                                                      ,dphichange_(0)
                                                      ,dphichange_noshift_(0)
{
    setAnchorHit();
}

const std::vector<SDL::Segment*>& SDL::MiniDoublet::getListOfOutwardSegmentPtrs()
{
    return outwardSegmentPtrs;
}

const std::vector<SDL::Segment*>& SDL::MiniDoublet::getListOfInwardSegmentPtrs()
{
    return inwardSegmentPtrs;
}

void SDL::MiniDoublet::addOutwardSegmentPtr(SDL::Segment* sg)
{
    outwardSegmentPtrs.push_back(sg);
}

void SDL::MiniDoublet::addInwardSegmentPtr(SDL::Segment* sg)
{
    inwardSegmentPtrs.push_back(sg);
}

void SDL::MiniDoublet::setAnchorHit()
{
    const SDL::Module& lowerModule = lowerHitPtr()->getModule();

    // Assign anchor hit pointers based on their hit type
    if (lowerModule.moduleType() == SDL::Module::PS)
    {
        if (lowerModule.moduleLayerType() == SDL::Module::Pixel)
        {
            anchorHitPtr_ = lowerHitPtr();
        }
        else
        {
            anchorHitPtr_ = upperHitPtr();
        }
    }
    else
    {
        anchorHitPtr_ = lowerHitPtr();
    }

}

SDL::Hit* SDL::MiniDoublet::lowerHitPtr() const
{
    return lowerHitPtr_;
}

SDL::Hit* SDL::MiniDoublet::upperHitPtr() const
{
    return upperHitPtr_;
}

SDL::Hit* SDL::MiniDoublet::anchorHitPtr() const
{
    return anchorHitPtr_;
}

const int& SDL::MiniDoublet::getPassAlgo() const
{
    return passAlgo_;
}

const SDL::Hit& SDL::MiniDoublet::getLowerShiftedHit() const
{
    return lowerShiftedHit_;
}

const SDL::Hit& SDL::MiniDoublet::getUpperShiftedHit() const
{
    return upperShiftedHit_;
}

const float& SDL::MiniDoublet::getDz() const
{
    return dz_;
}

const float& SDL::MiniDoublet::getShiftedDz() const
{
    return shiftedDz_;
}

const float& SDL::MiniDoublet::getDeltaPhi() const
{
    return dphi_;
}

const float& SDL::MiniDoublet::getDeltaPhiChange() const
{
    return dphichange_;
}

const float& SDL::MiniDoublet::getDeltaPhiNoShift() const
{
    return dphi_noshift_;
}

const float& SDL::MiniDoublet::getDeltaPhiChangeNoShift() const
{
    return dphichange_noshift_;
}

const std::map<std::string, float>& SDL::MiniDoublet::getRecoVars() const
{
    return recovars_;
}

const float& SDL::MiniDoublet::getRecoVar(std::string key) const
{
    return recovars_.at(key);
}

void SDL::MiniDoublet::setLowerShiftedHit(float x, float y, float z, int idx)
{
    lowerShiftedHit_.setXYZ(x, y, z);
    lowerShiftedHit_.setIdx(idx);
}

void SDL::MiniDoublet::setUpperShiftedHit(float x, float y, float z, int idx)
{
    upperShiftedHit_.setXYZ(x, y, z);
    upperShiftedHit_.setIdx(idx);
}

void SDL::MiniDoublet::setDz(float dz)
{
    dz_ = dz;
}

void SDL::MiniDoublet::setShiftedDz(float shiftedDz)
{
    shiftedDz_ = shiftedDz;
}

void SDL::MiniDoublet::setDeltaPhi(float dphi)
{
    dphi_ = dphi;
}

void SDL::MiniDoublet::setDeltaPhiChange(float dphichange)
{
    dphichange_ = dphichange;
}

void SDL::MiniDoublet::setDeltaPhiNoShift(float dphi)
{
    dphi_noshift_ = dphi;
}

void SDL::MiniDoublet::setDeltaPhiChangeNoShift(float dphichange)
{
    dphichange_noshift_ = dphichange;
}

void SDL::MiniDoublet::setRecoVars(std::string key, float var)
{
    recovars_[key] = var;
}

bool SDL::MiniDoublet::passesMiniDoubletAlgo(SDL::MDAlgo algo) const
{
    // Each algorithm is an enum shift it by its value and check against the flag
    return passAlgo_ & (1 << algo);
}

void SDL::MiniDoublet::runMiniDoubletAlgo(SDL::MDAlgo algo, SDL::LogLevel logLevel)
{
    if (algo == SDL::AllComb_MDAlgo)
    {
        runMiniDoubletAllCombAlgo();
    }
    else if (algo == SDL::Default_MDAlgo)
    {
        runMiniDoubletDefaultAlgo(logLevel);
    }
    else
    {
        SDL::cout << "Warning: Unrecognized mini-doublet algorithm!" << algo << std::endl;
        return;
    }
}

void SDL::MiniDoublet::runMiniDoubletAllCombAlgo()
{
    passAlgo_ |= (1 << SDL::AllComb_MDAlgo);
}

void SDL::MiniDoublet::runMiniDoubletDefaultAlgo(SDL::LogLevel logLevel)
{
    // Retreived the lower module object
    const SDL::Module& lowerModule = lowerHitPtr_->getModule();

    if (lowerModule.subdet() == SDL::Module::Barrel)
    {
        runMiniDoubletDefaultAlgoBarrel(logLevel);
    }
    else
    {
        runMiniDoubletDefaultAlgoEndcap(logLevel);
    }
}

void SDL::MiniDoublet::runMiniDoubletDefaultAlgoBarrel(SDL::LogLevel logLevel)
{
    // First get the object that the pointer points to
    const SDL::Hit& lowerHit = (*lowerHitPtr_);
    const SDL::Hit& upperHit = (*upperHitPtr_);

    // Retreived the lower module object
    const SDL::Module& lowerModule = lowerHitPtr_->getModule();

    setRecoVars("miniCut", -999);

    // There are series of cuts that applies to mini-doublet in a "barrel" region

    // Cut #1: The dz difference
    // Ref to original code: https://github.com/slava77/cms-tkph2-ntuple/blob/184d2325147e6930030d3d1f780136bc2dd29ce6/doubletAnalysis.C#L3067

    setDz(lowerHit.z() - upperHit.z());
    const float& dz = getDz();

    // const float dzCut = lowerModule.moduleType() == SDL::Module::PS ? 10.f : 1.5f; // Could be tighter for PS modules

    //*
    // const float dzCut = 10.f; // Could be tighter for PS modules
    // if (not (std::abs(dz) < dzCut)) // If cut fails continue
    //*

    //*
    const float dzCut = lowerModule.moduleType() == SDL::Module::PS ? 2.f : 10.f;
    // const bool isNotInvertedCrosser = lowerModule.moduleType() == SDL::Module::PS ? true : (lowerHit.z() * dz > 0); // Not used as this saves very little on combinatorics. but could be something we can add back later
    const float sign = ((dz > 0) - (dz < 0)) * ((lowerHit.z() > 0) - (lowerHit.z() < 0));
    const float invertedcrossercut = (abs(dz) > 2) * sign;
    if (not (std::abs(dz) < dzCut and invertedcrossercut <= 0)) // Adding inverted crosser rejection
    //*

    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << lowerModule << std::endl;
            SDL::cout << "Debug: " << __FUNCTION__ << "()" << std::endl;
            SDL::cout << "upperHit: " << upperHit << std::endl;
            SDL::cout << "lowerHit: " << lowerHit << std::endl;
            SDL::cout << "dz : " << dz << std::endl;
            SDL::cout << "dzCut : " << dzCut << std::endl;
        }

        // did not pass default algo
        passAlgo_ &= (0 << SDL::Default_MDAlgo);
        return;
    }
    else
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << lowerModule << std::endl;
            SDL::cout << "Debug: " << __FUNCTION__ << "()" << std::endl;
            SDL::cout << "upperHit: " << upperHit << std::endl;
            SDL::cout << "lowerHit: " << lowerHit << std::endl;
            SDL::cout << "dz : " << dz << std::endl;
            SDL::cout << "dzCut : " << dzCut << std::endl;
        }
    }

    // Calculate the cut thresholds for the selection
    float miniCut = 0;
    if (lowerModule.moduleLayerType() == SDL::Module::Pixel)
        miniCut = MiniDoublet::dPhiThreshold(lowerHit, lowerModule);
    else
        miniCut = MiniDoublet::dPhiThreshold(upperHit, lowerModule);

    // Cut #2: dphi difference
    // Ref to original code: https://github.com/slava77/cms-tkph2-ntuple/blob/184d2325147e6930030d3d1f780136bc2dd29ce6/doubletAnalysis.C#L3085
    float xn = 0, yn = 0, zn = 0;
    if (lowerModule.side() != SDL::Module::Center) // If barrel and not center it is tilted
    {
        // Shift the hits and calculate new xn, yn position
        std::tie(xn, yn, zn) = shiftStripHits(lowerHit, upperHit, lowerModule, logLevel);

        // Lower or the upper hit needs to be modified depending on which one was actually shifted
        if (lowerModule.moduleLayerType() == SDL::Module::Pixel)
        {
            // SDL::Hit upperHitMod(upperHit);
            // upperHitMod.setXYZ(xn, yn, upperHit.z());
            // setDeltaPhi(lowerHit.deltaPhi(upperHitMod));
            setUpperShiftedHit(xn, yn, upperHit.z());
            setDeltaPhi(lowerHit.deltaPhi(getUpperShiftedHit()));
            setDeltaPhiNoShift(lowerHit.deltaPhi(upperHit));
        }
        else
        {
            // SDL::Hit lowerHitMod(lowerHit);
            // lowerHitMod.setXYZ(xn, yn, lowerHit.z());
            // setDeltaPhi(lowerHitMod.deltaPhi(upperHit));
            setLowerShiftedHit(xn, yn, lowerHit.z());
            setDeltaPhi(getLowerShiftedHit().deltaPhi(upperHit));
            setDeltaPhiNoShift(lowerHit.deltaPhi(upperHit));
        }
    }
    else
    {
        setDeltaPhi(lowerHit.deltaPhi(upperHit));
        setDeltaPhiNoShift(lowerHit.deltaPhi(upperHit));
    }

    setRecoVars("miniCut", miniCut);

    if (not (std::abs(getDeltaPhi()) < miniCut)) // If cut fails continue
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << lowerModule << std::endl;
            SDL::cout << "Debug: " << __FUNCTION__ << "()" << std::endl;
            SDL::cout << "upperHit: " << upperHit << std::endl;
            SDL::cout << "lowerHit: " << lowerHit << std::endl;
            SDL::cout << "fabsdPhi : " << getDeltaPhi() << std::endl;
            SDL::cout << "miniCut : " << miniCut << std::endl;
        }

        // did not pass default algo
        passAlgo_ &= (0 << SDL::Default_MDAlgo);
        return;
    }
    else
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << lowerModule << std::endl;
            SDL::cout << "Debug: " << __FUNCTION__ << "()" << std::endl;
            SDL::cout << "upperHit: " << upperHit << std::endl;
            SDL::cout << "lowerHit: " << lowerHit << std::endl;
            SDL::cout << "fabsdPhi : " << getDeltaPhi() << std::endl;
            SDL::cout << "miniCut : " << miniCut << std::endl;
        }
    }

    // Cut #3: The dphi change going from lower Hit to upper Hit
    // Ref to original code: https://github.com/slava77/cms-tkph2-ntuple/blob/184d2325147e6930030d3d1f780136bc2dd29ce6/doubletAnalysis.C#L3076
    if (lowerModule.side() != SDL::Module::Center)
    {
        // When it is tilted, use the new shifted positions
        if (lowerModule.moduleLayerType() == SDL::Module::Pixel)
        {
            // SDL::Hit upperHitMod(upperHit);
            // upperHitMod.setXYZ(xn, yn, upperHit.z());
            // dPhi Change should be calculated so that the upper hit has higher rt.
            // In principle, this kind of check rt_lower < rt_upper should not be necessary because the hit shifting should have taken care of this.
            // (i.e. the strip hit is shifted to be aligned in the line of sight from interaction point to pixel hit of PS module guaranteeing rt ordering)
            // But I still placed this check for safety. (TODO: After cheking explicitly if not needed remove later?)
            // setDeltaPhiChange(lowerHit.rt() < upperHitMod.rt() ? lowerHit.deltaPhiChange(upperHitMod) : upperHitMod.deltaPhiChange(lowerHit));
            setDeltaPhiChange(((lowerHit.rt() < getUpperShiftedHit().rt()) ? lowerHit.deltaPhiChange(getUpperShiftedHit()) : getUpperShiftedHit().deltaPhiChange(lowerHit)));
            setDeltaPhiChangeNoShift(((lowerHit.rt() < upperHit.rt()) ? lowerHit.deltaPhiChange(upperHit) : upperHit.deltaPhiChange(lowerHit)));
        }
        else
        {
            // SDL::Hit lowerHitMod(lowerHit);
            // lowerHitMod.setXYZ(xn, yn, lowerHit.z());
            // dPhi Change should be calculated so that the upper hit has higher rt.
            // In principle, this kind of check rt_lower < rt_upper should not be necessary because the hit shifting should have taken care of this.
            // (i.e. the strip hit is shifted to be aligned in the line of sight from interaction point to pixel hit of PS module guaranteeing rt ordering)
            // But I still placed this check for safety. (TODO: After cheking explicitly if not needed remove later?)
            // setDeltaPhiChange(lowerHitMod.rt() < upperHit.rt() ? lowerHitMod.deltaPhiChange(upperHit) : upperHit.deltaPhiChange(lowerHitMod));
            setDeltaPhiChange(((getLowerShiftedHit().rt() < upperHit.rt()) ? getLowerShiftedHit().deltaPhiChange(upperHit) : upperHit.deltaPhiChange(getLowerShiftedHit())));
            setDeltaPhiChangeNoShift(((lowerHit.rt() < upperHit.rt()) ? lowerHit.deltaPhiChange(upperHit) : upperHit.deltaPhiChange(lowerHit)));
        }
    }
    else
    {
        // When it is flat lying module, whichever is the lowerSide will always have rt lower
        setDeltaPhiChange(lowerHit.deltaPhiChange(upperHit));
        setDeltaPhiChangeNoShift(lowerHit.deltaPhiChange(upperHit));
    }

    if (not (std::abs(getDeltaPhiChange()) < miniCut)) // If cut fails continue
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << lowerModule << std::endl;
            SDL::cout << "Debug: " << __FUNCTION__ << "()" << std::endl;
            SDL::cout << "upperHit: " << upperHit << std::endl;
            SDL::cout << "lowerHit: " << lowerHit << std::endl;
            SDL::cout << "fabsdPhiChange : " << getDeltaPhiChange() << std::endl;
            SDL::cout << "miniCut : " << miniCut << std::endl;
        }

        // did not pass default algo
        passAlgo_ &= (0 << SDL::Default_MDAlgo);
        return;
    }
    else
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << lowerModule << std::endl;
            SDL::cout << "Debug: " << __FUNCTION__ << "()" << std::endl;
            SDL::cout << "upperHit: " << upperHit << std::endl;
            SDL::cout << "lowerHit: " << lowerHit << std::endl;
            SDL::cout << "fabsdPhiChange : " << getDeltaPhiChange() << std::endl;
            SDL::cout << "miniCut : " << miniCut << std::endl;
        }
    }

    // If all cut passed this pair is good, and make and add the mini-doublet
    passAlgo_ |= (1 << SDL::Default_MDAlgo);
    return;

}

void SDL::MiniDoublet::runMiniDoubletDefaultAlgoEndcap(SDL::LogLevel logLevel)
{
    // First get the object that the pointer points to
    const SDL::Hit& lowerHit = (*lowerHitPtr_);
    const SDL::Hit& upperHit = (*upperHitPtr_);

    // Retreived the lower module object
    const SDL::Module& lowerModule = lowerHitPtr_->getModule();

    setRecoVars("miniCut", -999);

    // There are series of cuts that applies to mini-doublet in a "endcap" region

    // Cut #1 : dz cut. The dz difference can't be larger than 1cm. (max separation is 4mm for modules in the endcap)
    // Ref to original code: https://github.com/slava77/cms-tkph2-ntuple/blob/184d2325147e6930030d3d1f780136bc2dd29ce6/doubletAnalysis.C#L3093
    // For PS module in case when it is tilted a different dz (after the strip hit shift) is calculated later.
    // This is because the 10.f cut is meant more for sanity check (most will pass this cut anyway) (TODO: Maybe revisit this cut later?)

    setDz(lowerHit.z() - upperHit.z());
    float dz = getDz(); // Not const since later it might change depending on the type of module

    const float dzCut = ((lowerModule.side() == SDL::Module::Endcap) ?  1.f : 10.f);
    if (not (std::abs(dz) < dzCut)) // If cut fails continue
    {
        if (logLevel >= SDL::Log_Debug2)
        {
            SDL::cout << lowerModule << std::endl;
            SDL::cout << "Debug: " << __FUNCTION__ << "()" << std::endl;
            SDL::cout << "upperHit: " << upperHit << std::endl;
            SDL::cout << "lowerHit: " << lowerHit << std::endl;
            SDL::cout << "dz : " << dz << std::endl;
            SDL::cout << "dzCut : " << dzCut << std::endl;
        }

        // did not pass default algo
        passAlgo_ &= (0 << SDL::Default_MDAlgo);
        return;
    }
    else
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << lowerModule << std::endl;
            SDL::cout << "Debug: " << __FUNCTION__ << "()" << std::endl;
            SDL::cout << "upperHit: " << upperHit << std::endl;
            SDL::cout << "lowerHit: " << lowerHit << std::endl;
            SDL::cout << "dz : " << dz << std::endl;
            SDL::cout << "dzCut : " << dzCut << std::endl;
        }
    }

    // Cut #2 : drt cut. The dz difference can't be larger than 1cm. (max separation is 4mm for modules in the endcap)
    // Ref to original code: https://github.com/slava77/cms-tkph2-ntuple/blob/184d2325147e6930030d3d1f780136bc2dd29ce6/doubletAnalysis.C#L3100
    const float drtCut = lowerModule.moduleType() == SDL::Module::PS ? 2.f : 10.f;
    float drt = std::abs(lowerHit.rt() - upperHit.rt());
    if (not (drt < drtCut)) // If cut fails continue
    {
        if (logLevel >= SDL::Log_Debug2)
        {
            SDL::cout << lowerModule << std::endl;
            SDL::cout << "Debug: " << __FUNCTION__ << "()" << std::endl;
            SDL::cout << "upperHit: " << upperHit << std::endl;
            SDL::cout << "lowerHit: " << lowerHit << std::endl;
            SDL::cout << "drt : " << drt << std::endl;
            SDL::cout << "drtCut : " << drtCut << std::endl;
        }

        // did not pass default algo
        passAlgo_ &= (0 << SDL::Default_MDAlgo);
        return;
    }
    else
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << lowerModule << std::endl;
            SDL::cout << "Debug: " << __FUNCTION__ << "()" << std::endl;
            SDL::cout << "upperHit: " << upperHit << std::endl;
            SDL::cout << "lowerHit: " << lowerHit << std::endl;
            SDL::cout << "drt : " << drt << std::endl;
            SDL::cout << "drtCut : " << drtCut << std::endl;
        }
    }

    // Calculate the cut thresholds for the selection
    
    // Cut #3: dphi difference
    // Ref to original code: https://github.com/slava77/cms-tkph2-ntuple/blob/184d2325147e6930030d3d1f780136bc2dd29ce6/doubletAnalysis.C#L3111
    // // Old comments ----
    // // Old comments Slava, 6:17 PM
    // // Old comments here for the code to work you would need to slide (line extrapolate) the lower or the upper  hit along the strip direction to the radius of the other
    // // Old comments you'll get it almost right by assuming radial strips and just add the d_rt*(cosPhi, sinPhi)
    // // Old comments ----
    // // Old comments The algorithm assumed that the radial position is ~close according to Slava.
    // // Old comments However, for PS modules, it is not the case.
    // // Old comments So we'd have to move the hits to be in same position as the other.
    // // Old comments We'll move the pixel along the radial direction (assuming the radial direction is more or less same as the strip direction)
    // ----
    // The new scheme shifts strip hits to be "aligned" along the line of sight from interaction point to the pixel hit (if it is PS modules)
    float xn = 0, yn = 0, zn = 0;
    // if (lowerModule.moduleType() == SDL::Module::PS)
    // {
    // Shift the hits and calculate new xn, yn position
    std::tie(xn, yn, zn) = shiftStripHits(lowerHit, upperHit, lowerModule, logLevel);

    if (lowerModule.moduleType() == SDL::Module::PS)
    {
        // Appropriate lower or upper hit is modified after checking which one was actually shifted
        if (lowerModule.moduleLayerType() == SDL::Module::Pixel)
        {
            // SDL::Hit upperHitMod(upperHit);
            // upperHitMod.setXYZ(xn, yn, upperHit.z());
            // setDeltaPhi(lowerHit.deltaPhi(upperHitMod));
            setUpperShiftedHit(xn, yn, upperHit.z());
            setDeltaPhi(lowerHit.deltaPhi(getUpperShiftedHit()));
            setDeltaPhiNoShift(lowerHit.deltaPhi(upperHit));
        }
        else
        {
            // SDL::Hit lowerHitMod(lowerHit);
            // lowerHitMod.setXYZ(xn, yn, lowerHit.z());
            // setDeltaPhi(lowerHitMod.deltaPhi(upperHit));
            setLowerShiftedHit(xn, yn, lowerHit.z());
            setDeltaPhi(getLowerShiftedHit().deltaPhi(upperHit));
            setDeltaPhiNoShift(lowerHit.deltaPhi(upperHit));
        }
    }
    else
    {
        // SDL::Hit upperHitMod(upperHit);
        // upperHitMod.setXYZ(xn, yn, upperHit.z());
        // setDeltaPhi(lowerHit.deltaPhi(upperHitMod));
        setUpperShiftedHit(xn, yn, upperHit.z());
        setDeltaPhi(lowerHit.deltaPhi(getUpperShiftedHit()));
        setDeltaPhiNoShift(lowerHit.deltaPhi(upperHit));
    }

    // dz needs to change if it is a PS module where the strip hits are shifted in order to properly account for the case when a tilted module falls under "endcap logic"
    // if it was an endcap it will have zero effect
    if (lowerModule.moduleType() == SDL::Module::PS)
    {
        if (lowerModule.moduleLayerType() == SDL::Module::Pixel)
        {
            setShiftedDz(lowerHit.z() - zn);
            dz = getShiftedDz();
        }
        else
        {
            setShiftedDz(upperHit.z() - zn);
            dz = getShiftedDz();
        }
    }

    float miniCut = 0;
    if (lowerModule.moduleLayerType() == SDL::Module::Pixel)
        miniCut = MiniDoublet::dPhiThreshold(lowerHit, lowerModule, getDeltaPhi(), dz);
    else
        miniCut = MiniDoublet::dPhiThreshold(upperHit, lowerModule, getDeltaPhi(), dz);

    setRecoVars("miniCut",miniCut);

    if (not (std::abs(getDeltaPhi()) < miniCut)) // If cut fails continue
    {
        if (logLevel >= SDL::Log_Debug2)
        {
            SDL::cout << lowerModule << std::endl;
            SDL::cout << "Debug: " << __FUNCTION__ << "()" << std::endl;
            SDL::cout << "upperHit: " << upperHit << std::endl;
            SDL::cout << "lowerHit: " << lowerHit << std::endl;
            SDL::cout << "fabsdPhi : " << getDeltaPhi() << std::endl;
            SDL::cout << "miniCut : " << miniCut << std::endl;
        }

        // did not pass default algo
        passAlgo_ &= (0 << SDL::Default_MDAlgo);
        return;
    }
    else
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << lowerModule << std::endl;
            SDL::cout << "Debug: " << __FUNCTION__ << "()" << std::endl;
            SDL::cout << "upperHit: " << upperHit << std::endl;
            SDL::cout << "lowerHit: " << lowerHit << std::endl;
            SDL::cout << "fabsdPhi : " << getDeltaPhi() << std::endl;
            SDL::cout << "miniCut : " << miniCut << std::endl;
        }
    }

    // Cut #4: Another cut on the dphi after some modification
    // Ref to original code: https://github.com/slava77/cms-tkph2-ntuple/blob/184d2325147e6930030d3d1f780136bc2dd29ce6/doubletAnalysis.C#L3119-L3124

    
    float dzFrac = std::abs(dz) / fabs(lowerHit.z());
    setDeltaPhiChange(getDeltaPhi() / dzFrac * (1.f + dzFrac));
    setDeltaPhiChangeNoShift(getDeltaPhiNoShift() / dzFrac * (1.f + dzFrac));
    if (not (std::abs(getDeltaPhiChange()) < miniCut)) // If cut fails continue
    {
        if (logLevel >= SDL::Log_Debug2)
        {
            SDL::cout << lowerModule << std::endl;
            SDL::cout << "Debug: " << __FUNCTION__ << "()" << std::endl;
            SDL::cout << "upperHit: " << upperHit << std::endl;
            SDL::cout << "lowerHit: " << lowerHit << std::endl;
            SDL::cout << "dz : " << dz << std::endl;
            SDL::cout << "dzFrac : " << dzFrac << std::endl;
            SDL::cout << "fabsdPhi : " << std::abs(getDeltaPhi()) << std::endl;
            SDL::cout << "fabsdPhiMod : " << getDeltaPhiChange() << std::endl;
            SDL::cout << "miniCut : " << miniCut << std::endl;
        }

        // did not pass default algo
        passAlgo_ &= (0 << SDL::Default_MDAlgo);
        return;
    }
    else
    {
        if (logLevel >= SDL::Log_Debug3)
        {
            SDL::cout << lowerModule << std::endl;
            SDL::cout << "Debug: " << __FUNCTION__ << "()" << std::endl;
            SDL::cout << "upperHit: " << upperHit << std::endl;
            SDL::cout << "lowerHit: " << lowerHit << std::endl;
            SDL::cout << "dz : " << dz << std::endl;
            SDL::cout << "dzFrac : " << dzFrac << std::endl;
            SDL::cout << "fabsdPhi : " << std::abs(getDeltaPhi()) << std::endl;
            SDL::cout << "fabsdPhiMod : " << getDeltaPhiChange() << std::endl;
            SDL::cout << "miniCut : " << miniCut << std::endl;
        }
    }

    // If all cut passed this pair is good, and make and add the mini-doublet
    passAlgo_ |= (1 << SDL::Default_MDAlgo);
    return;
}

bool SDL::MiniDoublet::isIdxMatched(const MiniDoublet& md) const
{
    if (not (lowerHitPtr()->isIdxMatched(*(md.lowerHitPtr()))))
        return false;
    if (not (upperHitPtr()->isIdxMatched(*(md.upperHitPtr()))))
        return false;
    return true;
}

bool SDL::MiniDoublet::isAnchorHitIdxMatched(const MiniDoublet& md) const
{
    if (not anchorHitPtr_->isIdxMatched(*(md.anchorHitPtr())))
        return false;
    return true;
}

namespace SDL
{
    std::ostream& operator<<(std::ostream& out, const MiniDoublet& md)
    {
        out << "dz " << md.getDz() << std::endl;
        out << "shiftedDz " << md.getShiftedDz() << std::endl;
        out << "dphi " << md.getDeltaPhi() << std::endl;
        out << "dphinoshift " << md.getDeltaPhiNoShift() << std::endl;
        out << "dphichange " << md.getDeltaPhiChange() << std::endl;
        out << "dphichangenoshift " << md.getDeltaPhiChangeNoShift() << std::endl;
        out << "ptestimate " << SDL::MathUtil::ptEstimateFromDeltaPhiChangeAndRt(md.getDeltaPhiChange(), md.lowerHitPtr_->rt()) << std::endl;
        out << "ptestimate from noshift " << SDL::MathUtil::ptEstimateFromDeltaPhiChangeAndRt(md.getDeltaPhiChangeNoShift(), md.lowerHitPtr_->rt()) << std::endl;
        out << std::endl;
        out << "Lower Hit " << std::endl;
        out << "------------------------------" << std::endl;
        {
            IndentingOStreambuf indent(out);
            out << "Lower         " << md.lowerHitPtr_ << std::endl;
            out << "Lower Shifted " << md.lowerShiftedHit_ << std::endl;
            out << md.lowerHitPtr_->getModule() << std::endl;
        }
        out << "Upper Hit " << std::endl;
        out << "------------------------------" << std::endl;
        {
            IndentingOStreambuf indent(out);
            out << "Upper         " << md.upperHitPtr_ << std::endl;
            out << "Upper Shifted " << md.upperShiftedHit_ << std::endl;
            out << md.upperHitPtr_->getModule();
        }
        return out;
    }

    std::ostream& operator<<(std::ostream& out, const MiniDoublet* md)
    {
        out << *md;
        return out;
    }
}

float SDL::MiniDoublet::dPhiThreshold(const SDL::Hit& lowerHit, const SDL::Module& module,const float dPhi, const float dz)
{
    // =================================================================
    // Various constants
    // =================================================================
    const float kRinv1GeVf = (2.99792458e-3 * 3.8);
    const float k2Rinv1GeVf = kRinv1GeVf / 2.;
    // const float ptCut = PTCUT;
    // const float sinAlphaMax = 0.95;
    float ptCut = 1;
    // std::cout <<  " module.layer(): " << module.layer() <<  std::endl;
    // if (module.layer() == 6 or module.layer() == 5)
    // {
    //     ptCut = 0.96;
    // }
    float sinAlphaMax = 0.95;
    // if (module.layer() == 6)
    // {
    //     sinAlphaMax = 2.95;
    // }
    // p2Sim.directionT-r2Sim.directionT smearing around the mean computed with ptSim,rSim
    // (1 sigma based on 95.45% = 2sigma at 2 GeV)
    std::array<float, 6> miniMulsPtScaleBarrel {0.0052, 0.0038, 0.0034, 0.0034, 0.0032, 0.0034};
    std::array<float, 5> miniMulsPtScaleEndcap {0.006, 0.006, 0.006, 0.006, 0.006}; //inter/extra-polated from L11 and L13 both roughly 0.006 [larger R have smaller value by ~50%]
    //mean of the horizontal layer position in y; treat this as R below
    std::array<float, 6> miniRminMeanBarrel {21.8, 34.6, 49.6, 67.4, 87.6, 106.8}; // TODO: Update this with newest geometry
    std::array<float, 5> miniRminMeanEndcap {131.4, 156.2, 185.6, 220.3, 261.5};// use z for endcaps // TODO: Update this with newest geometry

    // =================================================================
    // Computing some components that make up the cut threshold
    // =================================================================
    float rt = lowerHit.rt();
    unsigned int iL = module.layer() - 1;
    const float miniSlope = std::asin(std::min(rt * k2Rinv1GeVf / ptCut, sinAlphaMax));
    const float rLayNominal = ((module.subdet() == SDL::Module::Barrel) ? miniRminMeanBarrel[iL] : miniRminMeanEndcap[iL]);
    const float miniPVoff = 0.1 / rLayNominal;
    const float miniMuls = ((module.subdet() == SDL::Module::Barrel) ? miniMulsPtScaleBarrel[iL] * 3.f / ptCut : miniMulsPtScaleEndcap[iL] * 3.f / ptCut);
    const bool isTilted = module.subdet() == SDL::Module::Barrel and module.side() != SDL::Module::Center;
    const bool tiltedOT123 = true;
    const float pixelPSZpitch = 0.15;
    const unsigned int detid = ((module.moduleLayerType() == SDL::Module::Pixel) ?  module.partnerDetId() : module.detId());
    const float drdz = tiltedGeometry.getDrDz(detid);
    const float miniTilt = ((isTilted && tiltedOT123) ? 0.5f * pixelPSZpitch * drdz / sqrt(1.f + drdz * drdz) / moduleGapSize(module) : 0);

    // Compute luminous region requirement for endcap
    const float deltaZLum = 15.f;
    const float miniLum = abs(dPhi * deltaZLum/dz); // Balaji's new error
    // const float miniLum = abs(deltaZLum / lowerHit.z()); // Old error


    // =================================================================
    // Return the threshold value
    // =================================================================
    // Following condition is met if the module is central and flatly lying
    if (module.subdet() == SDL::Module::Barrel and module.side() == SDL::Module::Center)
    {
        return miniSlope + sqrt(pow(miniMuls, 2) + pow(miniPVoff, 2));
    }
    // Following condition is met if the module is central and tilted
    else if (module.subdet() == SDL::Module::Barrel and module.side() != SDL::Module::Center) //all types of tilted modules
    {
        return miniSlope + sqrt(pow(miniMuls, 2) + pow(miniPVoff, 2) + pow(miniTilt * miniSlope, 2));
    }
    // If not barrel, it is Endcap
    else
    {
        return miniSlope + sqrt(pow(miniMuls, 2) + pow(miniPVoff, 2) + pow(miniLum, 2));
    }

}

// NOTE: Deprecated
[[deprecated("SDL:: fabsdPhiPixelShift() is deprecated")]]
float SDL::MiniDoublet::fabsdPhiPixelShift(const SDL::Hit& lowerHit, const SDL::Hit& upperHit, const SDL::Module& lowerModule, SDL::LogLevel logLevel)
{

    float fabsdPhi;

    // dependent variables for this if statement
    // lowerModule
    // lowerHit
    // upperHit
    // SDL::endcapGeometry

    float xa; // "anchor" x (strip hit x)
    float ya; // "anchor" y (strip hit y)
    float xo; // old x (before the pixel hit is moved up or down)
    float yo; // old y (before the pixel hit is moved up or down)
    float xn; // new x (after the pixel hit is moved up or down)
    float yn; // new y (after the pixel hit is moved up or down)
    unsigned int detid; // Needed to access geometry information

    if (lowerModule.moduleLayerType() == SDL::Module::Pixel)
    {
        xo = lowerHit.x();
        yo = lowerHit.y();
        xa = upperHit.x();
        ya = upperHit.y();
        detid = lowerModule.partnerDetId();
    }
    else
    {
        xo = upperHit.x();
        yo = upperHit.y();
        xa = lowerHit.x();
        ya = lowerHit.y();
        detid = lowerModule.detId();
    }

    float slope = 0;
    if (lowerModule.subdet() == SDL::Module::Endcap)
    {
       slope = SDL::endcapGeometry.getSlopeLower(detid); // Only need one slope
    }
    if (lowerModule.subdet() == SDL::Module::Barrel)
    {
       slope = SDL::tiltedGeometry.getSlope(detid); // Only need one slope
    }

    xn = (slope * xa + (1.f / slope) * xo - ya + yo) / (slope + (1.f / slope)); // new xn
    yn = (xn - xa) * slope + ya; // new yn

    if (lowerModule.subdet() == SDL::Module::Barrel)
    {
        if (slope == 123456789) // Special value designated for tilted module when the slope is exactly infinity (module lying along y-axis)
        {
            xn = xa; // New x point is simply where the anchor is
            yn = yo; // No shift in y
        }
        else if (slope == 0)
        {
            xn = xo; // New x point is simply where the anchor is
            yn = ya; // No shift in y
        }
    }

    if (lowerModule.moduleLayerType() == SDL::Module::Pixel)
    {
        SDL::Hit lowerHitMod(lowerHit);
        lowerHitMod.setXYZ(xn, yn, lowerHit.z());
        fabsdPhi = std::abs(lowerHitMod.deltaPhi(upperHit));
    }
    else
    {
        SDL::Hit upperHitMod(upperHit);
        upperHitMod.setXYZ(xn, yn, upperHit.z());
        fabsdPhi = std::abs(lowerHit.deltaPhi(upperHitMod));
    }

    if (logLevel >= SDL::Log_Debug3)
    {
        // SDL::cout <<  " use_lower: " << use_lower;
        // SDL::cout <<  " yintercept: " << yintercept <<  " slope: " << slope <<  std::endl;
        SDL::cout <<  " xa: " << xa <<  " ya: " << ya <<  std::endl;
        SDL::cout <<  " xo: " << xo <<  " yo: " << yo <<  " xn: " << xn <<  " yn: " << yn <<  std::endl;
    }

    return fabsdPhi;
}

// NOTE: Deprecated
[[deprecated("SDL:: fabsdPhiStripShift() is deprecated")]]
float SDL::MiniDoublet::fabsdPhiStripShift(const SDL::Hit& lowerHit, const SDL::Hit& upperHit, const SDL::Module& lowerModule, SDL::LogLevel logLevel)
{

    float fabsdPhi;

    // dependent variables for this if statement
    // lowerModule
    // lowerHit
    // upperHit
    // SDL::endcapGeometry

    float xa; // "anchor" x (strip hit x)
    float ya; // "anchor" y (strip hit y)
    float xo; // old x (before the pixel hit is moved up or down)
    float yo; // old y (before the pixel hit is moved up or down)
    float xn; // new x (after the pixel hit is moved up or down)
    float yn; // new y (after the pixel hit is moved up or down)
    unsigned int detid; // Needed to access geometry information

    if (lowerModule.moduleLayerType() == SDL::Module::Pixel)
    {
        xo = upperHit.x();
        yo = upperHit.y();
        xa = lowerHit.x();
        ya = lowerHit.y();
        detid = lowerModule.partnerDetId();
    }
    else
    {
        xo = lowerHit.x();
        yo = lowerHit.y();
        xa = upperHit.x();
        ya = upperHit.y();
        detid = lowerModule.detId();
    }

    float slope = 0;
    if (lowerModule.subdet() == SDL::Module::Endcap)
    {
       slope = SDL::endcapGeometry.getSlopeLower(detid); // Only need one slope
    }
    if (lowerModule.subdet() == SDL::Module::Barrel)
    {
       slope = SDL::tiltedGeometry.getSlope(detid); // Only need one slope
    }

    xn = (slope * xa + (1.f / slope) * xo - ya + yo) / (slope + (1.f / slope)); // new xn
    yn = (xn - xa) * slope + ya; // new yn

    if (lowerModule.subdet() == SDL::Module::Barrel)
    {
        if (slope == 123456789) // Special value designated for tilted module when the slope is exactly infinity (module lying along y-axis)
        {
            xn = xa; // New x point is simply where the anchor is
            yn = yo; // No shift in y
        }
        else if (slope == 0)
        {
            xn = xo; // New x point is simply where the anchor is
            yn = ya; // No shift in y
        }
    }

    if (lowerModule.moduleLayerType() == SDL::Module::Pixel)
    {
        SDL::Hit upperHitMod(upperHit);
        upperHitMod.setXYZ(xn, yn, upperHit.z());
        fabsdPhi = std::abs(lowerHit.deltaPhi(upperHitMod));
    }
    else
    {
        SDL::Hit lowerHitMod(lowerHit);
        lowerHitMod.setXYZ(xn, yn, lowerHit.z());
        fabsdPhi = std::abs(lowerHitMod.deltaPhi(upperHit));
    }

    if (logLevel >= SDL::Log_Debug3)
    {
        SDL::cout <<  " slope: " << slope <<  std::endl;
        SDL::cout <<  " xa: " << xa <<  " ya: " << ya <<  std::endl;
        SDL::cout <<  " xo: " << xo <<  " yo: " << yo <<  " xn: " << xn <<  " yn: " << yn <<  std::endl;
    }

    return fabsdPhi;
}

std::tuple<float, float, float> SDL::MiniDoublet::shiftStripHits(const SDL::Hit& lowerHit, const SDL::Hit& upperHit, const SDL::Module& lowerModule, SDL::LogLevel logLevel)
{

    // This is the strip shift scheme that is explained in http://uaf-10.t2.ucsd.edu/~phchang/talks/PhilipChang20190607_SDL_Update.pdf (see backup slides)
    // The main feature of this shifting is that the strip hits are shifted to be "aligned" in the line of sight from interaction point to the the pixel hit.
    // (since pixel hit is well defined in 3-d)
    // The strip hit is shifted along the strip detector to be placed in a guessed position where we think they would have actually crossed
    // The size of the radial direction shift due to module separation gap is computed in "radial" size, while the shift is done along the actual strip orientation
    // This means that there may be very very subtle edge effects coming from whether the strip hit is center of the module or the at the edge of the module
    // But this should be relatively minor effect

    // dependent variables for this if statement
    // lowerModule
    // lowerHit
    // upperHit
    // SDL::endcapGeometry
    // SDL::tiltedGeometry

    // Some variables relevant to the function
    float xp; // pixel x (pixel hit x)
    float yp; // pixel y (pixel hit y)
    float xa; // "anchor" x (the anchor position on the strip module plane from pixel hit)
    float ya; // "anchor" y (the anchor position on the strip module plane from pixel hit)
    float xo; // old x (before the strip hit is moved up or down)
    float yo; // old y (before the strip hit is moved up or down)
    float xn; // new x (after the strip hit is moved up or down)
    float yn; // new y (after the strip hit is moved up or down)
    float abszn; // new z in absolute value
    float zn; // new z with the sign (+/-) accounted
    float angleA; // in r-z plane the theta of the pixel hit in polar coordinate is the angleA
    float angleB; // this is the angle of tilted module in r-z plane ("drdz"), for endcap this is 90 degrees
    unsigned int detid; // The detId of the strip module in the PS pair. Needed to access geometry information
    bool isEndcap; // If endcap, drdz = infinity
    const SDL::Hit* pixelHitPtr; // Pointer to the pixel hit
    const SDL::Hit* stripHitPtr; // Pointer to the strip hit
    float moduleSeparation;
    float drprime; // The radial shift size in x-y plane projection
    float drprime_x; // x-component of drprime
    float drprime_y; // y-component of drprime
    float slope; // The slope of the possible strip hits for a given module in x-y plane
    float absArctanSlope;
    float angleM; // the angle M is the angle of rotation of the module in x-y plane if the possible strip hits are along the x-axis, then angleM = 0, and if the possible strip hits are along y-axis angleM = 90 degrees
    float absdzprime; // The distance between the two points after shifting

    // Assign hit pointers based on their hit type
    if (lowerModule.moduleType() == SDL::Module::PS)
    {
        if (lowerModule.moduleLayerType() == SDL::Module::Pixel)
        {
            pixelHitPtr = &lowerHit;
            stripHitPtr = &upperHit;
            detid = lowerModule.partnerDetId(); // partnerDetId returns the partner detId (since lower Module == pixel, get partner ID to access strip one)
        }
        else
        {
            pixelHitPtr = &upperHit;
            stripHitPtr = &lowerHit;
            detid = lowerModule.detId(); // Since the lower module is not pixel, the lower module is the strip
        }
    }
    else // if (lowerModule.moduleType() == SDL::Module::TwoS) // If it is a TwoS module (if this is called likely an endcap module) then anchor the inner hit and shift the outer hit
    {
        pixelHitPtr = &lowerHit; // Even though in this case the "pixelHitPtr" is really just a strip hit, we pretend it is the anchoring pixel hit
        stripHitPtr = &upperHit;
        detid = lowerModule.detId(); // partnerDetId returns the partner detId (since lower Module == pixel, get partner ID to access strip one)
    }

    // If it is endcap some of the math gets simplified (and also computers don't like infinities)
    isEndcap = lowerModule.subdet() == SDL::Module::Endcap;

    // NOTE: TODO: Keep in mind that the sin(atan) function can be simplifed to something like x / sqrt(1 + x^2) and similar for cos
    // I am not sure how slow sin, atan, cos, functions are in c++. If x / sqrt(1 + x^2) are faster change this later to reduce arithmetic computation time

    // The pixel hit is used to compute the angleA which is the theta in polar coordinate
    // angleA = std::atan(pixelHitPtr->rt() / pixelHitPtr->z() + (pixelHitPtr->z() < 0 ? M_PI : 0)); // Shift by pi if the z is negative so that the value of the angleA stays between 0 to pi and not -pi/2 to pi/2
    angleA = fabs(std::atan(pixelHitPtr->rt() / pixelHitPtr->z())); // Shift by pi if the z is negative so that the value of the angleA stays between 0 to pi and not -pi/2 to pi/2

    // angleB = isEndcap ? M_PI / 2. : -std::atan(tiltedGeometry.getDrDz(detid) * (lowerModule.side() == SDL::Module::PosZ ? -1 : 1)); // The tilt module on the postive z-axis has negative drdz slope in r-z plane and vice versa
    angleB = ((isEndcap) ? M_PI / 2. : std::atan(tiltedGeometry.getDrDz(detid))); // The tilt module on the postive z-axis has negative drdz slope in r-z plane and vice versa

    // https://iopscience.iop.org/article/10.1088/1748-0221/12/02/C02049/pdf says the PS modules have distances of 1.6mm 2.6mm or 4mm
    // The following verifies that the first layer has 0.26 spacing (the constants are from the fitted values, in order to rotate the module perfectly so that the distance can be read from the Scan output directly)
    // tree->Scan("(ph2_y*cos(-1.39626596)-ph2_x*sin(-1.39626596))*cos(-0.820378147)-sin(-0.820378147)*ph2_z:ph2_z*cos(-0.820378147)+sin(-0.820378147)*(ph2_y*cos(-1.39626596)-ph2_x*sin(-1.39626596))","ph2_order==0&&ph2_side==2&&ph2_rod==1&&ph2_layer==1&&ph2_module==1")
    // The following verifies that the second tilted layer has 0.26 spacing (the constants are from the fitted values, in order to rotate the module perfectly so that the distance can be read from the Scan output directly)
    // tree->Scan("(ph2_y*cos(-1.44996432)-ph2_x*sin(-1.44996432))*cos(-0.697883399)-sin(-0.697883399)*ph2_z:ph2_z*cos(-0.697883399)+sin(-0.697883399)*(ph2_y*cos(-1.44996432)-ph2_x*sin(-1.44996432))","ph2_order==0&&ph2_side==2&&ph2_rod==1&&ph2_layer==2&&ph2_module==1")
    // The following verifies that the third tilted layer has 0.26 spacing (This one has a completley 90 degrees i.e. infinite slope in x-y plane, so the math simplifies a bit)
    // tree->Scan("ph2_x*cos(-0.767954394)-ph2_z*sin(-0.767954394):ph2_z*cos(-0.767954394)+ph2_x*sin(-0.767954394)","ph2_order==0&&ph2_side==2&&ph2_rod==1&&ph2_layer==3&&ph2_module==1")
    // The following verifies that the third tilted layer with the largest negative z module also has 0.26
    // tree->Scan("ph2_x*cos(1.04706386)-ph2_z*sin(1.04706386):ph2_z*cos(1.04706386)+ph2_x*sin(1.04706386)","ph2_order==0&&ph2_side==1&&ph2_rod==1&&ph2_layer==3&&ph2_module==1")
    // For endcap it is easy to check that z has 0.4 spacing for all PS modules
    // If the lowerModule is pixel, then the module separation direction is positive and vice versa.
    // (i.e. if the pixel was the upper module, then the anchor points are going to shift inward in x-y plane wrt to pixel x-y point and vice versa.)
    // (cf. this should be taking care of the "rt" size comparison that would be done when calculating fabsdPhiChange variable.)
/*    if (lowerModule.moduleType() == SDL::Module::PS) // ensure this happens only for PS modules
        moduleSeparation = (isEndcap ? 0.40 : 0.26) * ((lowerModule.moduleLayerType() == SDL::Module::Pixel) ? 1 : -1);
    else
        moduleSeparation = (isEndcap ? 0.40 : 0.26);*/


    moduleSeparation = moduleGapSize(lowerModule);

    // Sign flips if the pixel is later layer
    if (lowerModule.moduleType() == SDL::Module::PS and lowerModule.moduleLayerType() != SDL::Module::Pixel)
    {
        moduleSeparation *= -1;
    }

    drprime = (moduleSeparation / std::sin(angleA + angleB)) * std::sin(angleA);

    if (lowerModule.subdet() == SDL::Module::Endcap)
    {
        slope = SDL::endcapGeometry.getSlopeLower(detid); // Only need one slope
    }
    if (lowerModule.subdet() == SDL::Module::Barrel)
    {
        slope = SDL::tiltedGeometry.getSlope(detid);
    }

    // Compute arctan of the slope and take care of the slope = infinity case
    absArctanSlope = ((slope != SDL_INF) ? fabs(std::atan(slope)) : M_PI / 2); // Since C++ can't represent infinity, SDL_INF = 123456789 was used to represent infinity in the data table

    // The pixel hit position
    xp = pixelHitPtr->x();
    yp = pixelHitPtr->y();

    // Depending on which quadrant the pixel hit lies, we define the angleM by shifting them slightly differently
    if (xp > 0 and yp > 0)
    {
        angleM = absArctanSlope;
    }
    else if (xp > 0 and yp < 0)
    {
        angleM = M_PI - absArctanSlope;
    }
    else if (xp < 0 and yp < 0)
    {
        angleM = M_PI + absArctanSlope;
    }
    else // if (xp < 0 and yp > 0)
    {
        angleM = 2 * M_PI - absArctanSlope;
    }

    // Then since the angleM sign is taken care of properly
    drprime_x = drprime * std::sin(angleM);
    drprime_y = drprime * std::cos(angleM);

    // The new anchor position is
    xa = xp + drprime_x;
    ya = yp + drprime_y;

    // The original strip hit position
    xo = stripHitPtr->x();
    yo = stripHitPtr->y();

    // Compute the new strip hit position (if the slope vaule is in special condition take care of the exceptions)
    if (slope == SDL_INF) // Special value designated for tilted module when the slope is exactly infinity (module lying along y-axis)
    {
        xn = xa; // New x point is simply where the anchor is
        yn = yo; // No shift in y
    }
    else if (slope == 0)
    {
        xn = xo; // New x point is simply where the anchor is
        yn = ya; // No shift in y
    }
    else
    {
        xn = (slope * xa + (1.f / slope) * xo - ya + yo) / (slope + (1.f / slope)); // new xn
        yn = (xn - xa) * slope + ya; // new yn
    }

    // Computing new Z position
    absdzprime = fabs(moduleSeparation / std::sin(angleA + angleB) * std::cos(angleA)); // module separation sign is for shifting in radial direction for z-axis direction take care of the sign later

    // Depending on which one as closer to the interactin point compute the new z wrt to the pixel properly
    if (lowerModule.moduleLayerType() == SDL::Module::Pixel)
    {
        abszn = fabs(pixelHitPtr->z()) + absdzprime;
    }
    else
    {
        abszn = fabs(pixelHitPtr->z()) - absdzprime;
    }

    zn = abszn * ((pixelHitPtr->z() > 0) ? 1 : -1); // Apply the sign of the zn

    if (logLevel == SDL::Log_Debug3)
    {
        SDL::cout << upperHit << std::endl;
        SDL::cout << lowerHit << std::endl;
        SDL::cout <<  " lowerModule.moduleType()==SDL::Module::PS: " << (lowerModule.moduleType()==SDL::Module::PS) <<  std::endl;
        SDL::cout <<  " lowerModule.moduleLayerType()==SDL::Module::Pixel: " << (lowerModule.moduleLayerType()==SDL::Module::Pixel) <<  std::endl;
        SDL::cout <<  " pixelHitPtr: " << pixelHitPtr <<  std::endl;
        SDL::cout <<  " stripHitPtr: " << stripHitPtr <<  std::endl;
        SDL::cout <<  " detid: " << detid <<  std::endl;
        SDL::cout <<  " isEndcap: " << isEndcap <<  std::endl;
        SDL::cout <<  " pixelHitPtr->rt(): " << pixelHitPtr->rt() <<  std::endl;
        SDL::cout <<  " pixelHitPtr->z(): " << pixelHitPtr->z() <<  std::endl;
        SDL::cout <<  " angleA: " << angleA <<  std::endl;
        SDL::cout <<  " angleB: " << angleB <<  std::endl;
        SDL::cout <<  " moduleSeparation: " << moduleSeparation <<  std::endl;
        SDL::cout <<  " drprime: " << drprime <<  std::endl;
        SDL::cout <<  " slope: " << slope <<  std::endl;
        SDL::cout <<  " absArctanSlope: " << absArctanSlope <<  std::endl;
        SDL::cout <<  " angleM: " << angleM <<  std::endl;
        SDL::cout <<  " drprime_x: " << drprime_x <<  std::endl;
        SDL::cout <<  " drprime_y: " << drprime_y <<  std::endl;
        SDL::cout <<  " xa: " << xa <<  std::endl;
        SDL::cout <<  " ya: " << ya <<  std::endl;
        SDL::cout <<  " xo: " << xo <<  std::endl;
        SDL::cout <<  " yo: " << yo <<  std::endl;
        SDL::cout <<  " xn: " << xn <<  std::endl;
        SDL::cout <<  " yn: " << yn <<  std::endl;
        SDL::cout <<  " absdzprime: " << absdzprime <<  std::endl;
        SDL::cout <<  " zn: " << zn <<  std::endl;
    }

    return std::make_tuple(xn, yn, zn);

}

[[deprecated("SDL:: useBarrelLogic() is deprecated")]]
bool SDL::MiniDoublet::useBarrelLogic(const SDL::Module& lowerModule)
{

    // Either it is a barrel and flat module (i.e. "Center")
    // or if it is a "normal" tilted modules (ones that are not too steeply tilted)
    // then use barrel logic

    if ( (lowerModule.subdet() == SDL::Module::Barrel and lowerModule.side() == SDL::Module::Center) or isNormalTiltedModules(lowerModule))
        return true;
    else
        return false;
}

[[deprecated("SDL:: isNormalTiltedModules() is deprecated. Use isTighterTiltedModules instead")]]
bool SDL::MiniDoublet::isNormalTiltedModules(const SDL::Module& lowerModule)
{
    // The "normal" tilted modules are the subset of tilted modules that will use the tilted module logic
    // If a tiltde module is not part of the "normal" tiltded modules, they will default to endcap logic (the actual defaulting logic is implemented elsewhere)
    if (
           (lowerModule.subdet() == SDL::Module::Barrel and lowerModule.side() != SDL::Module::Center and lowerModule.layer() == 3)
           or (lowerModule.subdet() == SDL::Module::Barrel and lowerModule.side() == SDL::Module::NegZ and lowerModule.layer() == 2 and lowerModule.rod() > 5)
           or (lowerModule.subdet() == SDL::Module::Barrel and lowerModule.side() == SDL::Module::PosZ and lowerModule.layer() == 2 and lowerModule.rod() < 8)
           or (lowerModule.subdet() == SDL::Module::Barrel and lowerModule.side() == SDL::Module::NegZ and lowerModule.layer() == 1 and lowerModule.rod() > 9)
           or (lowerModule.subdet() == SDL::Module::Barrel and lowerModule.side() == SDL::Module::PosZ and lowerModule.layer() == 1 and lowerModule.rod() < 4)
       )
        return true;
    else
        return false;
}

bool SDL::MiniDoublet::isTighterTiltedModules(const SDL::Module& lowerModule)
{
    // The "tighter" tilted modules are the subset of tilted modules that have smaller spacing
    // This is the same as what was previously considered as"isNormalTiltedModules"
    // See Figure 9.1 of https://cds.cern.ch/record/2272264/files/CMS-TDR-014.pdf
    if (
           (lowerModule.subdet() == SDL::Module::Barrel and lowerModule.side() != SDL::Module::Center and lowerModule.layer() == 3)
           or (lowerModule.subdet() == SDL::Module::Barrel and lowerModule.side() == SDL::Module::NegZ and lowerModule.layer() == 2 and lowerModule.rod() > 5)
           or (lowerModule.subdet() == SDL::Module::Barrel and lowerModule.side() == SDL::Module::PosZ and lowerModule.layer() == 2 and lowerModule.rod() < 8)
           or (lowerModule.subdet() == SDL::Module::Barrel and lowerModule.side() == SDL::Module::NegZ and lowerModule.layer() == 1 and lowerModule.rod() > 9)
           or (lowerModule.subdet() == SDL::Module::Barrel and lowerModule.side() == SDL::Module::PosZ and lowerModule.layer() == 1 and lowerModule.rod() < 4)
       )
        return true;
    else
        return false;
}

// The function to determine gap
float SDL::MiniDoublet::moduleGapSize(const Module& lowerModule)
{
    std::array<float, 3> miniDeltaTilted {0.26, 0.26, 0.26};
    std::array<float,3> miniDeltaLooseTilted {0.4,0.4,0.4};
    //std::array<float, 6> miniDeltaEndcap {0.4, 0.4, 0.4, 0.18, 0.18, 0.18};
    std::array<float, 6> miniDeltaFlat {0.26, 0.16, 0.16, 0.18, 0.18, 0.18};
    std::array<std::array<float, 15>, 5> miniDeltaEndcap; //15 rings, 5 layers

    for (size_t i = 0; i < 5; i++)
    {
        for (size_t j = 0; j < 15; j++)
        {
            if (i == 0 || i == 1)
            {
                if (j < 10)
                {
                    miniDeltaEndcap[i][j] = 0.4;
                }
                else
                {
                    miniDeltaEndcap[i][j] = 0.18;
                }
            }
            else if (i == 2 || i == 3)
            {
                if (j < 8)
                {
                    miniDeltaEndcap[i][j] = 0.4;
                }
                else
                {
                    miniDeltaEndcap[i][j]  = 0.18;
                }
            }
            else
            {
                if (j < 9)
                {
                    miniDeltaEndcap[i][j] = 0.4;
                }
                else
                {
                    miniDeltaEndcap[i][j] = 0.18;
                }
            }
        }
    }

    unsigned int iL = lowerModule.layer() - 1;
    int iR = lowerModule.subdet() == SDL::Module::Endcap ? lowerModule.ring() - 1 : -1;

    float moduleSeparation = 0;

    if (lowerModule.subdet() == SDL::Module::Barrel and lowerModule.side() == SDL::Module::Center)
    {
        moduleSeparation = miniDeltaFlat[iL];
    }
    else if (isTighterTiltedModules(lowerModule))
    {
        moduleSeparation = miniDeltaTilted[iL];
    }
    else if (lowerModule.subdet() == SDL::Module::Endcap)
    {
        moduleSeparation = miniDeltaEndcap[iL][iR];
    }
    else //Loose tilted modules
    {
        moduleSeparation = miniDeltaLooseTilted[iL];
    }

    return moduleSeparation;

}

// NOTE: Deprecated
[[deprecated("SDL:: isHitPairAMiniDoublet() is deprecated. Create an instance of MiniDoublet and Use runMiniDoubletAlgo()")]]
bool SDL::MiniDoublet::isHitPairAMiniDoublet(const SDL::Hit& lowerHit, const SDL::Hit& upperHit, const SDL::Module& lowerModule, SDL::MDAlgo algo, SDL::LogLevel logLevel)
{
    // If the algorithm is "do all combination" (e.g. used for efficiency calculation)
    if (algo == SDL::AllComb_MDAlgo)
    {
        return true;
    }
    // If the algorithm is default
    else if (algo == SDL::Default_MDAlgo)
    {

        // There are several cuts applied to possible hit pairs, and if the hit pairs passes the cut, it is considered as mini-doublet.
        // The logic is split into two parts, when it's for barrel and endcap.
        // Internally, if it is in barrel, there is some subtle difference depending on whether it was tilted or not tilted.
        // The difference is encoded in the "SDL::MiniDoublet::dPhiThreshold()" function

        // If barrel, apply cuts for barrel mini-doublets
        // if (lowerModule.subdet() == SDL::Module::Barrel)
        if ( (lowerModule.subdet() == SDL::Module::Barrel and lowerModule.side() == SDL::Module::Center) or isTighterTiltedModules(lowerModule))
        {

            // Cut #1: The dz difference
            // Ref to original code: https://github.com/slava77/cms-tkph2-ntuple/blob/184d2325147e6930030d3d1f780136bc2dd29ce6/doubletAnalysis.C#L3067
            float dzCut = lowerModule.moduleType() == SDL::Module::PS ? 2.f : 10.f;

            float dz = std::abs(lowerHit.z() - upperHit.z());
            if (not (dz < dzCut)) // If cut fails continue
            {
                if (logLevel >= SDL::Log_Debug3)
                {
                    SDL::cout << lowerModule << std::endl;
                    SDL::cout << "Debug: " << __FUNCTION__ << "()" << std::endl;
                    SDL::cout << "upperHit: " << upperHit << std::endl;
                    SDL::cout << "lowerHit: " << lowerHit << std::endl;
                    SDL::cout << "dz : " << dz << std::endl;
                    SDL::cout << "dzCut : " << dzCut << std::endl;
                }
                return false;
            }
            else
            {
                if (logLevel >= SDL::Log_Debug3)
                {
                    SDL::cout << lowerModule << std::endl;
                    SDL::cout << "Debug: " << __FUNCTION__ << "()" << std::endl;
                    SDL::cout << "upperHit: " << upperHit << std::endl;
                    SDL::cout << "lowerHit: " << lowerHit << std::endl;
                    SDL::cout << "dz : " << dz << std::endl;
                    SDL::cout << "dzCut : " << dzCut << std::endl;
                }
            }

            // Calculate the cut thresholds for the selection
            float miniCut = 0;

            if (lowerModule.moduleLayerType() == SDL::Module::Pixel)
                miniCut = MiniDoublet::dPhiThreshold(lowerHit, lowerModule);
            else
                miniCut = MiniDoublet::dPhiThreshold(upperHit, lowerModule);

            // Cut #2: dphi difference
            // Ref to original code: https://github.com/slava77/cms-tkph2-ntuple/blob/184d2325147e6930030d3d1f780136bc2dd29ce6/doubletAnalysis.C#L3085
            // float fabsdPhi = std::abs(lowerHit.deltaPhi(upperHit));
            float fabsdPhi = 0;
            float xn = 0, yn = 0, zn = 0;
            if (lowerModule.side() != SDL::Module::Center) // If barrel and not center it is tilted
            {
                // Shift the hits and calculate new xn, yn position
                std::tie(xn, yn, zn) = shiftStripHits(lowerHit, upperHit, lowerModule, logLevel);

                // Lower or the upper hit needs to be modified depending on which one was actually shifted
                if (lowerModule.moduleLayerType() == SDL::Module::Pixel)
                {
                    SDL::Hit upperHitMod(upperHit);
                    upperHitMod.setXYZ(xn, yn, upperHit.z());
                    fabsdPhi = std::abs(lowerHit.deltaPhi(upperHitMod));
                }
                else
                {
                    SDL::Hit lowerHitMod(lowerHit);
                    lowerHitMod.setXYZ(xn, yn, lowerHit.z());
                    fabsdPhi = std::abs(lowerHitMod.deltaPhi(upperHit));
                }
            }
            else
            {
                fabsdPhi = std::abs(lowerHit.deltaPhi(upperHit));
            }

            if (not (fabsdPhi < miniCut)) // If cut fails continue
            {
                if (logLevel >= SDL::Log_Debug3)
                {
                    SDL::cout << lowerModule << std::endl;
                    SDL::cout << "Debug: " << __FUNCTION__ << "()" << std::endl;
                    SDL::cout << "upperHit: " << upperHit << std::endl;
                    SDL::cout << "lowerHit: " << lowerHit << std::endl;
                    SDL::cout << "fabsdPhi : " << fabsdPhi << std::endl;
                    SDL::cout << "miniCut : " << miniCut << std::endl;
                }
                return false;
            }
            else
            {
                if (logLevel >= SDL::Log_Debug3)
                {
                    SDL::cout << lowerModule << std::endl;
                    SDL::cout << "Debug: " << __FUNCTION__ << "()" << std::endl;
                    SDL::cout << "upperHit: " << upperHit << std::endl;
                    SDL::cout << "lowerHit: " << lowerHit << std::endl;
                    SDL::cout << "fabsdPhi : " << fabsdPhi << std::endl;
                    SDL::cout << "miniCut : " << miniCut << std::endl;
                }
            }

            // Cut #3: The dphi change going from lower Hit to upper Hit
            // Ref to original code: https://github.com/slava77/cms-tkph2-ntuple/blob/184d2325147e6930030d3d1f780136bc2dd29ce6/doubletAnalysis.C#L3076
            float fabsdPhiChange;
            if (lowerModule.side() != SDL::Module::Center)
            {
                // When it is tilted, use the new shifted positions
                if (lowerModule.moduleLayerType() == SDL::Module::Pixel)
                {
                    SDL::Hit upperHitMod(upperHit);
                    upperHitMod.setXYZ(xn, yn, upperHit.z());
                    // dPhi Change should be calculated so that the upper hit has higher rt.
                    // In principle, this kind of check rt_lower < rt_upper should not be necessary because the hit shifting should have taken care of this.
                    // (i.e. the strip hit is shifted to be aligned in the line of sight from interaction point to pixel hit of PS module guaranteeing rt ordering)
                    // But I still placed this check for safety. (TODO: After cheking explicitly if not needed remove later?)
                    fabsdPhiChange = ((lowerHit.rt() < upperHitMod.rt()) ? std::abs(lowerHit.deltaPhiChange(upperHitMod)) : std::abs(upperHitMod.deltaPhiChange(lowerHit)));
                }
                else
                {
                    SDL::Hit lowerHitMod(lowerHit);
                    lowerHitMod.setXYZ(xn, yn, lowerHit.z());
                    // dPhi Change should be calculated so that the upper hit has higher rt.
                    // In principle, this kind of check rt_lower < rt_upper should not be necessary because the hit shifting should have taken care of this.
                    // (i.e. the strip hit is shifted to be aligned in the line of sight from interaction point to pixel hit of PS module guaranteeing rt ordering)
                    // But I still placed this check for safety. (TODO: After cheking explicitly if not needed remove later?)
                    fabsdPhiChange = ((lowerHitMod.rt() < upperHit.rt()) ? std::abs(lowerHitMod.deltaPhiChange(upperHit)) : std::abs(upperHit.deltaPhiChange(lowerHitMod)));
                }
            }
            else
            {
                // When it is flat lying module, whichever is the lowerSide will always have rt lower
                fabsdPhiChange = std::abs(lowerHit.deltaPhiChange(upperHit));
            }

            if (not (fabsdPhiChange < miniCut)) // If cut fails continue
            {
                if (logLevel >= SDL::Log_Debug3)
                {
                    SDL::cout << lowerModule << std::endl;
                    SDL::cout << "Debug: " << __FUNCTION__ << "()" << std::endl;
                    SDL::cout << "upperHit: " << upperHit << std::endl;
                    SDL::cout << "lowerHit: " << lowerHit << std::endl;
                    SDL::cout << "fabsdPhiChange : " << fabsdPhiChange << std::endl;
                    SDL::cout << "miniCut : " << miniCut << std::endl;
                }
                return false;
            }
            else
            {
                if (logLevel >= SDL::Log_Debug3)
                {
                    SDL::cout << lowerModule << std::endl;
                    SDL::cout << "Debug: " << __FUNCTION__ << "()" << std::endl;
                    SDL::cout << "upperHit: " << upperHit << std::endl;
                    SDL::cout << "lowerHit: " << lowerHit << std::endl;
                    SDL::cout << "fabsdPhiChange : " << fabsdPhiChange << std::endl;
                    SDL::cout << "miniCut : " << miniCut << std::endl;
                }
            }

            // If all cut passed this pair is good, and make and add the mini-doublet
            return true;

        }
        // If endcap, apply cuts for endcap mini-doublets
        else // If not barrel it is endcap
        {
            // Cut #1 : dz cut. The dz difference can't be larger than 1cm. (max separation is 4mm for modules in the endcap)
            // Ref to original code: https://github.com/slava77/cms-tkph2-ntuple/blob/184d2325147e6930030d3d1f780136bc2dd29ce6/doubletAnalysis.C#L3093
            // For PS module in case when it is tilted a different dz (after the strip hit shift) is calculated later.
            // This is because the 10.f cut is meant more for sanity check (most will pass this cut anyway) (TODO: Maybe revisit this cut later?)
            const float dzCut = ((lowerModule.side() == SDL::Module::Endcap) ?  1.f : 10.f);
            float dz = std::abs(lowerHit.z() - upperHit.z());
            if (not (dz < dzCut)) // If cut fails continue
            {
                if (logLevel >= SDL::Log_Debug2)
                {
                    SDL::cout << lowerModule << std::endl;
                    SDL::cout << "Debug: " << __FUNCTION__ << "()" << std::endl;
                    SDL::cout << "upperHit: " << upperHit << std::endl;
                    SDL::cout << "lowerHit: " << lowerHit << std::endl;
                    SDL::cout << "dz : " << dz << std::endl;
                    SDL::cout << "dzCut : " << dzCut << std::endl;
                }
                return false;
            }
            else
            {
                if (logLevel >= SDL::Log_Debug3)
                {
                    SDL::cout << lowerModule << std::endl;
                    SDL::cout << "Debug: " << __FUNCTION__ << "()" << std::endl;
                    SDL::cout << "upperHit: " << upperHit << std::endl;
                    SDL::cout << "lowerHit: " << lowerHit << std::endl;
                    SDL::cout << "dz : " << dz << std::endl;
                    SDL::cout << "dzCut : " << dzCut << std::endl;
                }
            }

            // Cut #2 : drt cut. The dz difference can't be larger than 1cm. (max separation is 4mm for modules in the endcap)
            // Ref to original code: https://github.com/slava77/cms-tkph2-ntuple/blob/184d2325147e6930030d3d1f780136bc2dd29ce6/doubletAnalysis.C#L3100
            const float drtCut = 10.f; // i.e. should be smaller than the module length. Could be tighter if PS modules
            float drt = std::abs(lowerHit.rt() - upperHit.rt());
            if (not (drt < drtCut)) // If cut fails continue
            {
                if (logLevel >= SDL::Log_Debug2)
                {
                    SDL::cout << lowerModule << std::endl;
                    SDL::cout << "Debug: " << __FUNCTION__ << "()" << std::endl;
                    SDL::cout << "upperHit: " << upperHit << std::endl;
                    SDL::cout << "lowerHit: " << lowerHit << std::endl;
                    SDL::cout << "drt : " << drt << std::endl;
                    SDL::cout << "drtCut : " << drtCut << std::endl;
                }
                return false;
            }
            else
            {
                if (logLevel >= SDL::Log_Debug3)
                {
                    SDL::cout << lowerModule << std::endl;
                    SDL::cout << "Debug: " << __FUNCTION__ << "()" << std::endl;
                    SDL::cout << "upperHit: " << upperHit << std::endl;
                    SDL::cout << "lowerHit: " << lowerHit << std::endl;
                    SDL::cout << "drt : " << drt << std::endl;
                    SDL::cout << "drtCut : " << drtCut << std::endl;
                }
            }

            // Calculate the cut thresholds for the selection
            float miniCut = 0;
            if (lowerModule.moduleLayerType() == SDL::Module::Pixel)
                miniCut = MiniDoublet::dPhiThreshold(lowerHit, lowerModule);
            else
                miniCut = MiniDoublet::dPhiThreshold(upperHit, lowerModule);

            // Cut #3: dphi difference
            // Ref to original code: https://github.com/slava77/cms-tkph2-ntuple/blob/184d2325147e6930030d3d1f780136bc2dd29ce6/doubletAnalysis.C#L3111
            // // Old comments ----
            // // Old comments Slava, 6:17 PM
            // // Old comments here for the code to work you would need to slide (line extrapolate) the lower or the upper  hit along the strip direction to the radius of the other
            // // Old comments you'll get it almost right by assuming radial strips and just add the d_rt*(cosPhi, sinPhi)
            // // Old comments ----
            // // Old comments The algorithm assumed that the radial position is ~close according to Slava.
            // // Old comments However, for PS modules, it is not the case.
            // // Old comments So we'd have to move the hits to be in same position as the other.
            // // Old comments We'll move the pixel along the radial direction (assuming the radial direction is more or less same as the strip direction)
            // ----
            // The new scheme shifts strip hits to be "aligned" along the line of sight from interaction point to the pixel hit (if it is PS modules)
            float fabsdPhi = 0;
            float xn = 0, yn = 0, zn = 0;
            // if (lowerModule.moduleType() == SDL::Module::PS)
            // {
                // Shift the hits and calculate new xn, yn position
                std::tie(xn, yn, zn) = shiftStripHits(lowerHit, upperHit, lowerModule, logLevel);

            if (lowerModule.moduleType() == SDL::Module::PS)
            {
                // Appropriate lower or upper hit is modified after checking which one was actually shifted
                if (lowerModule.moduleLayerType() == SDL::Module::Pixel)
                {
                    SDL::Hit upperHitMod(upperHit);
                    upperHitMod.setXYZ(xn, yn, upperHit.z());
                    fabsdPhi = std::abs(lowerHit.deltaPhi(upperHitMod));
                }
                else
                {
                    SDL::Hit lowerHitMod(lowerHit);
                    lowerHitMod.setXYZ(xn, yn, lowerHit.z());
                    fabsdPhi = std::abs(lowerHitMod.deltaPhi(upperHit));
                }
            }
            else
            {
                SDL::Hit upperHitMod(upperHit);
                upperHitMod.setXYZ(xn, yn, upperHit.z());
                fabsdPhi = std::abs(lowerHit.deltaPhi(upperHitMod));
                // fabsdPhi = std::abs(lowerHit.deltaPhi(upperHit));
            }

            if (not (fabsdPhi < miniCut)) // If cut fails continue
            {
                if (logLevel >= SDL::Log_Debug2)
                {
                    SDL::cout << lowerModule << std::endl;
                    SDL::cout << "Debug: " << __FUNCTION__ << "()" << std::endl;
                    SDL::cout << "upperHit: " << upperHit << std::endl;
                    SDL::cout << "lowerHit: " << lowerHit << std::endl;
                    SDL::cout << "fabsdPhi : " << fabsdPhi << std::endl;
                    SDL::cout << "miniCut : " << miniCut << std::endl;
                }
                return false;
            }
            else
            {
                if (logLevel >= SDL::Log_Debug3)
                {
                    SDL::cout << lowerModule << std::endl;
                    SDL::cout << "Debug: " << __FUNCTION__ << "()" << std::endl;
                    SDL::cout << "upperHit: " << upperHit << std::endl;
                    SDL::cout << "lowerHit: " << lowerHit << std::endl;
                    SDL::cout << "fabsdPhi : " << fabsdPhi << std::endl;
                    SDL::cout << "miniCut : " << miniCut << std::endl;
                }
            }

            // Cut #4: Another cut on the dphi after some modification
            // Ref to original code: https://github.com/slava77/cms-tkph2-ntuple/blob/184d2325147e6930030d3d1f780136bc2dd29ce6/doubletAnalysis.C#L3119-L3124

            // dz needs to change if it is a PS module where the strip hits are shifted in order to properly account for the case when a tilted module falls under "endcap logic"
            // if it was an endcap it will have zero effect
            if (lowerModule.moduleType() == SDL::Module::PS)
            {
                if (lowerModule.moduleLayerType() == SDL::Module::Pixel)
                {
                    dz = fabs(lowerHit.z() - zn);
                }
                else
                {
                    dz = fabs(upperHit.z() - zn);
                }
            }

            float dzFrac = dz / fabs(lowerHit.z());
            float fabsdPhiMod = fabsdPhi / dzFrac * (1.f + dzFrac);
            if (not (fabsdPhiMod < miniCut)) // If cut fails continue
            {
                if (logLevel >= SDL::Log_Debug2)
                {
                    SDL::cout << lowerModule << std::endl;
                    SDL::cout << "Debug: " << __FUNCTION__ << "()" << std::endl;
                    SDL::cout << "upperHit: " << upperHit << std::endl;
                    SDL::cout << "lowerHit: " << lowerHit << std::endl;
                    SDL::cout << "dz : " << dz << std::endl;
                    SDL::cout << "dzFrac : " << dzFrac << std::endl;
                    SDL::cout << "fabsdPhi : " << fabsdPhi << std::endl;
                    SDL::cout << "fabsdPhiMod : " << fabsdPhiMod << std::endl;
                    SDL::cout << "miniCut : " << miniCut << std::endl;
                }
                return false;
            }
            else
            {
                if (logLevel >= SDL::Log_Debug3)
                {
                    SDL::cout << lowerModule << std::endl;
                    SDL::cout << "Debug: " << __FUNCTION__ << "()" << std::endl;
                    SDL::cout << "upperHit: " << upperHit << std::endl;
                    SDL::cout << "lowerHit: " << lowerHit << std::endl;
                    SDL::cout << "dz : " << dz << std::endl;
                    SDL::cout << "dzFrac : " << dzFrac << std::endl;
                    SDL::cout << "fabsdPhi : " << fabsdPhi << std::endl;
                    SDL::cout << "fabsdPhiMod : " << fabsdPhiMod << std::endl;
                    SDL::cout << "miniCut : " << miniCut << std::endl;
                }
            }

            // If all cut passed this pair is good, and make and add the mini-doublet
            return true;
        }
    }
    else
    {
        SDL::cout << "Warning: Unrecognized mini-doublet algorithm!" << algo << std::endl;
        return false;
    }
}
