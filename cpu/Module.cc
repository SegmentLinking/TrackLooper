#include "Module.h"

SDL::CPU::Module::Module()
{
    setDetId(0);
}

SDL::CPU::Module::Module(unsigned int detId)
{
    setDetId(detId);
}

SDL::CPU::Module::Module(const Module& module)
{
    setDetId(module.detId());
}

SDL::CPU::Module::~Module()
{
}

const unsigned short& SDL::CPU::Module::subdet() const
{
    return subdet_;
}

const unsigned short& SDL::CPU::Module::side() const
{
    return side_;
}

const unsigned short& SDL::CPU::Module::layer() const
{
    return layer_;
}

const unsigned short& SDL::CPU::Module::rod() const
{
    return rod_;
}

const unsigned short& SDL::CPU::Module::ring() const
{
    return ring_;
}

const unsigned short& SDL::CPU::Module::module() const
{
    return module_;
}

const unsigned short& SDL::CPU::Module::isLower() const
{
    return isLower_;
}

const unsigned int& SDL::CPU::Module::detId() const
{
    return detId_;
}

const unsigned int& SDL::CPU::Module::partnerDetId() const
{
    return partnerDetId_;
}

const bool& SDL::CPU::Module::isInverted() const
{
    return isInverted_;
}

const bool SDL::CPU::Module::isPixelLayerModule() const
{
    return (detId_ == 1);
}

const SDL::CPU::Module::ModuleType& SDL::CPU::Module::moduleType() const
{
    return moduleType_;
}

const SDL::CPU::Module::ModuleLayerType& SDL::CPU::Module::moduleLayerType() const
{
    return moduleLayerType_;
}

const std::vector<SDL::CPU::Hit*>& SDL::CPU::Module::getHitPtrs() const
{
    return hits_;
}

const std::vector<SDL::CPU::MiniDoublet*>& SDL::CPU::Module::getMiniDoubletPtrs() const
{
    return miniDoublets_;
}

const std::vector<SDL::CPU::Segment*>& SDL::CPU::Module::getSegmentPtrs() const
{
    return segments_;
}

const std::vector<SDL::CPU::Triplet*>& SDL::CPU::Module::getTripletPtrs() const
{
    return triplets_;
}

const std::vector<SDL::CPU::Tracklet*>& SDL::CPU::Module::getTrackletPtrs() const
{
    return tracklets_;
}

void SDL::CPU::Module::setDetId(unsigned int detId)
{
    detId_ = detId;
    setDerivedQuantities();
}

void SDL::CPU::Module::setDerivedQuantities()
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

void SDL::CPU::Module::addHit(SDL::CPU::Hit* hit)
{
    // Set the information on the module for where this hit resides
    // So we can swim backwards to find which module this hit resides
    // for any meta-object that contains this hit
    hit->setModule(this);

    // Then add to the module
    hits_.push_back(hit);
}

void SDL::CPU::Module::addMiniDoublet(SDL::CPU::MiniDoublet* md)
{
    miniDoublets_.push_back(md);
}

void SDL::CPU::Module::addSegment(SDL::CPU::Segment* sg)
{
    segments_.push_back(sg);
}

void SDL::CPU::Module::addTriplet(SDL::CPU::Triplet* tp)
{
    triplets_.push_back(tp);
}

void SDL::CPU::Module::addTracklet(SDL::CPU::Tracklet* tp)
{
    tracklets_.push_back(tp);
}

unsigned short SDL::CPU::Module::parseSubdet(unsigned int detId)
{
    return (detId & (7 << 25)) >> 25;
}

unsigned short SDL::CPU::Module::parseSide(unsigned int detId)
{
    if (parseSubdet(detId) == SDL::CPU::Module::Endcap)
    {
        return (detId & (3 << 23)) >> 23;
    }
    else if (parseSubdet(detId) == SDL::CPU::Module::Barrel)
    {
        return (detId & (3 << 18)) >> 18;
    }
    else
    {
        return 0;
    }
}

unsigned short SDL::CPU::Module::parseLayer(unsigned int detId)
{
    if (parseSubdet(detId) == SDL::CPU::Module::Endcap)
    {
        return (detId & (7 << 18)) >> 18;
    }
    else if (parseSubdet(detId) == SDL::CPU::Module::Barrel)
    {
        return (detId & (7 << 20)) >> 20;
    }
    else
    {
        return 0;
    }
}

unsigned short SDL::CPU::Module::parseRod(unsigned int detId)
{
    if (parseSubdet(detId) == SDL::CPU::Module::Endcap)
    {
        return 0;
    }
    else if (parseSubdet(detId) == SDL::CPU::Module::Barrel)
    {
        return (detId & (127 << 10)) >> 10;
    }
    else
    {
        return 0;
    }
}

unsigned short SDL::CPU::Module::parseRing(unsigned int detId)
{
    if (parseSubdet(detId) == SDL::CPU::Module::Endcap)
    {
        return (detId & (15 << 12)) >> 12;
    }
    else if (parseSubdet(detId) == SDL::CPU::Module::Barrel)
    {
        return 0;
    }
    else
    {
        return 0;
    }

}

unsigned short SDL::CPU::Module::parseModule(unsigned int detId)
{
    return (detId & (127 << 2)) >> 2;
}

unsigned short SDL::CPU::Module::parseIsLower(unsigned int detId)
{
    return ((parseIsInverted(detId)) ? !(detId & 1) : (detId & 1));
}

bool SDL::CPU::Module::parseIsInverted(unsigned int detId)
{
    if (detId == 1 or detId == 2) // "1" or "2" detId means "pixel module" where we store all pixel hits/mini/segments into one bucket
        return 0;
    if (parseSubdet(detId) == SDL::CPU::Module::Endcap)
    {
        if (parseSide(detId) == SDL::CPU::Module::NegZ)
        {
            return parseModule(detId) % 2 == 1;
        }
        else if (parseSide(detId) == SDL::CPU::Module::PosZ)
        {
            return parseModule(detId) % 2 == 0;
        }
        else
        {
            SDL::CPU::cout << "Warning: parseIsInverted() categorization failed: detId = " << detId << std::endl;
            return 0;
        }
    }
    else if (parseSubdet(detId) == SDL::CPU::Module::Barrel)
    {
        if (parseSide(detId) == SDL::CPU::Module::Center)
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
                SDL::CPU::cout << "Warning: parseIsInverted() categorization failed: detId = " << detId << std::endl;
                return 0;
            }
        }
        else if (parseSide(detId) == SDL::CPU::Module::NegZ or parseSide(detId) == SDL::CPU::Module::PosZ)
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
                SDL::CPU::cout << "Warning: parseIsInverted() categorization failed: detId = " << detId << std::endl;
                return 0;
            }
        }
        else
        {
            SDL::CPU::cout << "Warning: parseIsInverted() categorization failed: detId = " << detId << std::endl;
            return 0;
        }
    }
    else
    {
        SDL::CPU::cout << "Warning: parseIsInverted() categorization failed: detId = " << detId << std::endl;
        return 0;
    }
}

unsigned int SDL::CPU::Module::parsePartnerDetId(unsigned int detId)
{
    if (parseIsLower(detId))
        return ((parseIsInverted(detId)) ? detId - 1 : detId + 1);
    else
        return ((parseIsInverted(detId)) ? detId + 1 : detId - 1);
}

SDL::CPU::Module::ModuleType SDL::CPU::Module::parseModuleType(unsigned int detId)
{
    if (parseSubdet(detId) == SDL::CPU::Module::Barrel)
    { 
        if (parseLayer(detId) <= 3)
            return SDL::CPU::Module::PS;
        else
            return SDL::CPU::Module::TwoS;
    }
    else
    {
        if (parseLayer(detId) <= 2)
        {
            if (parseRing(detId) <= 10)
                return SDL::CPU::Module::PS;
            else
                return SDL::CPU::Module::TwoS;
        }
        else
        {
            if (parseRing(detId) <= 7)
                return SDL::CPU::Module::PS;
            else
                return SDL::CPU::Module::TwoS;
        }
    }
}

SDL::CPU::Module::ModuleLayerType SDL::CPU::Module::parseModuleLayerType(unsigned int detId)
{
    if (parseModuleType(detId) == SDL::CPU::Module::TwoS)
        return SDL::CPU::Module::Strip;
    if (parseIsInverted(detId))
    {
        if (parseIsLower(detId))
            return SDL::CPU::Module::Strip;
        else
            return SDL::CPU::Module::Pixel;
    }
    else
    {
        if (parseIsLower(detId))
            return SDL::CPU::Module::Pixel;
        else
            return SDL::CPU::Module::Strip;
    }
}

namespace SDL
{
    namespace CPU
    {
        std::ostream& operator<<(std::ostream& out, const Module& module)
        {
            out << "Module(detId=" << module.detId();
            out << ", subdet=" << (module.subdet_ == SDL::CPU::Module::Barrel ? "Barrel" : "Endcap");
            out << ", side=" << (module.side_ == SDL::CPU::Module::Center ? "Center" : "Side");
            out << ", layer=" << module.layer_;
            out << ", rod=" << module.rod_;
            out << ", ring=" << module.ring_;
            out << ", module=" << module.module_;
            out << ", moduleType=" << (module.moduleType_ == SDL::CPU::Module::PS ? "PS" : "2S");
            out << ", moduleLayerType=" << (module.moduleLayerType_ == SDL::CPU::Module::Pixel ? "Pixel" : "Strip");
            out << ", isLower=" << module.isLower_;
            out << ", isInverted=" << module.isInverted_;
            out << ", isNormalTitled=" << SDL::CPU::MiniDoublet::isNormalTiltedModules(module);
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


}
