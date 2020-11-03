#include "Module.h"

SDL::Module::Module()
{
    setDetId(0);
}

SDL::Module::Module(unsigned int detId)
{
    setDetId(detId);
}

SDL::Module::Module(const Module& module)
{
    setDetId(module.detId());
}

SDL::Module::~Module()
{
}

const unsigned short& SDL::Module::subdet() const
{
    return subdet_;
}

const unsigned short& SDL::Module::side() const
{
    return side_;
}

const unsigned short& SDL::Module::layer() const
{
    return layer_;
}

const unsigned short& SDL::Module::rod() const
{
    return rod_;
}

const unsigned short& SDL::Module::ring() const
{
    return ring_;
}

const unsigned short& SDL::Module::module() const
{
    return module_;
}

const unsigned short& SDL::Module::isLower() const
{
    return isLower_;
}

const unsigned int& SDL::Module::detId() const
{
    return detId_;
}

const unsigned int& SDL::Module::partnerDetId() const
{
    return partnerDetId_;
}

const bool& SDL::Module::isInverted() const
{
    return isInverted_;
}

const SDL::Module::ModuleType& SDL::Module::moduleType() const
{
    return moduleType_;
}

const SDL::Module::ModuleLayerType& SDL::Module::moduleLayerType() const
{
    return moduleLayerType_;
}

const std::vector<SDL::Hit*>& SDL::Module::getHitPtrs() const
{
    return hits_;
}

const std::vector<SDL::MiniDoublet*>& SDL::Module::getMiniDoubletPtrs() const
{
    return miniDoublets_;
}

const std::vector<SDL::Segment*>& SDL::Module::getSegmentPtrs() const
{
    return segments_;
}

const std::vector<SDL::Triplet*>& SDL::Module::getTripletPtrs() const
{
    return triplets_;
}

const std::vector<SDL::Tracklet*>& SDL::Module::getTrackletPtrs() const
{
    return tracklets_;
}

void SDL::Module::setDetId(unsigned int detId)
{
    detId_ = detId;
    setDerivedQuantities();
}

void SDL::Module::setDerivedQuantities()
{
    subdet_ = parseSubdet(detId_);
    side_ = parseSide(detId_);
    layer_ = parseLayer(detId_);
    rod_ = parseRod(detId_);
    ring_ = parseRing(detId_);
    module_ = parseModule(detId_);
    isLower_ = parseIsLower(detId_);
    isInverted_ = parseIsInverted(detId_);
    partnerDetId_ = parsePartnerDetId(detId_);
    moduleType_ = parseModuleType(detId_);
    moduleLayerType_ = parseModuleLayerType(detId_);
}

void SDL::Module::addHit(SDL::Hit* hit)
{
    // Set the information on the module for where this hit resides
    // So we can swim backwards to find which module this hit resides
    // for any meta-object that contains this hit
    hit->setModule(this);

    // Then add to the module
    hits_.push_back(hit);
}

void SDL::Module::addMiniDoublet(SDL::MiniDoublet* md)
{
    miniDoublets_.push_back(md);
}

void SDL::Module::addSegment(SDL::Segment* sg)
{
    segments_.push_back(sg);
}

void SDL::Module::addTriplet(SDL::Triplet* tp)
{
    triplets_.push_back(tp);
}

void SDL::Module::addTracklet(SDL::Tracklet* tp)
{
    tracklets_.push_back(tp);
}

unsigned short SDL::Module::parseSubdet(unsigned int detId)
{
    return (detId & (7 << 25)) >> 25;
}

unsigned short SDL::Module::parseSide(unsigned int detId)
{
    if (parseSubdet(detId) == SDL::Module::Endcap)
    {
        return (detId & (3 << 23)) >> 23;
    }
    else if (parseSubdet(detId) == SDL::Module::Barrel)
    {
        return (detId & (3 << 18)) >> 18;
    }
    else
    {
        return 0;
    }
}

unsigned short SDL::Module::parseLayer(unsigned int detId)
{
    if (parseSubdet(detId) == SDL::Module::Endcap)
    {
        return (detId & (7 << 18)) >> 18;
    }
    else if (parseSubdet(detId) == SDL::Module::Barrel)
    {
        return (detId & (7 << 20)) >> 20;
    }
    else
    {
        return 0;
    }
}

unsigned short SDL::Module::parseRod(unsigned int detId)
{
    if (parseSubdet(detId) == SDL::Module::Endcap)
    {
        return 0;
    }
    else if (parseSubdet(detId) == SDL::Module::Barrel)
    {
        return (detId & (127 << 10)) >> 10;
    }
    else
    {
        return 0;
    }
}

unsigned short SDL::Module::parseRing(unsigned int detId)
{
    if (parseSubdet(detId) == SDL::Module::Endcap)
    {
        return (detId & (15 << 12)) >> 12;
    }
    else if (parseSubdet(detId) == SDL::Module::Barrel)
    {
        return 0;
    }
    else
    {
        return 0;
    }

}

unsigned short SDL::Module::parseModule(unsigned int detId)
{
    return (detId & (127 << 2)) >> 2;
}

unsigned short SDL::Module::parseIsLower(unsigned int detId)
{
    return ((parseIsInverted(detId)) ? !(detId & 1) : (detId & 1));
}

bool SDL::Module::parseIsInverted(unsigned int detId)
{
    if (parseSubdet(detId) == SDL::Module::Endcap)
    {
        if (parseSide(detId) == SDL::Module::NegZ)
        {
            return parseModule(detId) % 2 == 1;
        }
        else if (parseSide(detId) == SDL::Module::PosZ)
        {
            return parseModule(detId) % 2 == 0;
        }
        else
        {
            SDL::cout << "Warning: parseIsInverted() categorization failed" << std::endl;
            return 0;
        }
    }
    else if (parseSubdet(detId) == SDL::Module::Barrel)
    {
        if (parseSide(detId) == SDL::Module::Center)
        {
            if (parseLayer(detId) <= 3)
            {
                return parseModule(detId) % 2 == 1;
            }
            else if (parseLayer(detId) >= 4)
            {
                return parseModule(detId) % 2 == 0;
            }
            else
            {
                SDL::cout << "Warning: parseIsInverted() categorization failed" << std::endl;
                return 0;
            }
        }
        else if (parseSide(detId) == SDL::Module::NegZ or parseSide(detId) == SDL::Module::PosZ)
        {
            if (parseLayer(detId) <= 2)
            {
                return parseModule(detId) % 2 == 1;
            }
            else if (parseLayer(detId) == 3)
            {
                return parseModule(detId) % 2 == 0;
            }
            else
            {
                SDL::cout << "Warning: parseIsInverted() categorization failed" << std::endl;
                return 0;
            }
        }
        else
        {
            SDL::cout << "Warning: parseIsInverted() categorization failed" << std::endl;
            return 0;
        }
    }
    else
    {
        SDL::cout << "Warning: parseIsInverted() categorization failed" << std::endl;
        return 0;
    }
}

unsigned int SDL::Module::parsePartnerDetId(unsigned int detId)
{
    if (parseIsLower(detId))
        return ((parseIsInverted(detId)) ? detId - 1 : detId + 1);
    else
        return ((parseIsInverted(detId)) ? detId + 1 : detId - 1);
}

SDL::Module::ModuleType SDL::Module::parseModuleType(unsigned int detId)
{
    if (parseSubdet(detId) == SDL::Module::Barrel)
    { 
        if (parseLayer(detId) <= 3)
            return SDL::Module::PS;
        else
            return SDL::Module::TwoS;
    }
    else
    {
        if (parseLayer(detId) <= 2)
        {
            if (parseRing(detId) <= 10)
                return SDL::Module::PS;
            else
                return SDL::Module::TwoS;
        }
        else
        {
            if (parseRing(detId) <= 7)
                return SDL::Module::PS;
            else
                return SDL::Module::TwoS;
        }
    }
}

SDL::Module::ModuleLayerType SDL::Module::parseModuleLayerType(unsigned int detId)
{
    if (parseModuleType(detId) == SDL::Module::TwoS)
        return SDL::Module::Strip;
    if (parseIsInverted(detId))
    {
        if (parseIsLower(detId))
            return SDL::Module::Strip;
        else
            return SDL::Module::Pixel;
    }
    else
    {
        if (parseIsLower(detId))
            return SDL::Module::Pixel;
        else
            return SDL::Module::Strip;
    }
}

namespace SDL
{
    std::ostream& operator<<(std::ostream& out, const Module& module)
    {
        out << "Module(detId=" << module.detId();
        out << ", subdet=" << (module.subdet_ == SDL::Module::Barrel ? "Barrel" : "Endcap");
        out << ", side=" << (module.side_ == SDL::Module::Center ? "Center" : "Side");
        out << ", layer=" << module.layer_;
        out << ", rod=" << module.rod_;
        out << ", ring=" << module.ring_;
        out << ", module=" << module.module_;
        out << ", moduleType=" << (module.moduleType_ == SDL::Module::PS ? "PS" : "2S");
        out << ", moduleLayerType=" << (module.moduleLayerType_ == SDL::Module::Pixel ? "Pixel" : "Strip");
        out << ", isLower=" << module.isLower_;
        out << ", isInverted=" << module.isInverted_;
        out << ", isNormalTitled=" << SDL::MiniDoublet::isNormalTiltedModules(module);
        out << ")" << std::endl;
        // for (auto& hitPtr : module.hits_)
        //     out << hitPtr << std::endl;
        // for (auto& mdPtr : module.miniDoublets_)
        //     out << mdPtr << std::endl;
        // for (auto& sgPtr : module.segments_)
        //     out << sgPtr << std::endl;
        out << "" << std::endl;

        return out;
    }

    std::ostream& operator<<(std::ostream& out, const Module* module)
    {
        out << *module;
        return out;
    }


}
