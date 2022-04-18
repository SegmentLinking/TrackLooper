#include "TrackletBase.h"

SDL::CPU::TrackletBase::TrackletBase()
{
}

SDL::CPU::TrackletBase::~TrackletBase()
{
}

SDL::CPU::TrackletBase::TrackletBase(const TrackletBase& tl) :
    innerSegmentPtr_(tl.innerSegmentPtr()),
    outerSegmentPtr_(tl.outerSegmentPtr()),
    passAlgo_(tl.getPassAlgo()),
    passBitsDefaultAlgo_(tl.getPassBitsDefaultAlgo()),
    recovars_(tl.getRecoVars())
{
}

SDL::CPU::TrackletBase::TrackletBase(SDL::CPU::Segment* innerSegmentPtr, SDL::CPU::Segment* outerSegmentPtr) :
    innerSegmentPtr_(innerSegmentPtr),
    outerSegmentPtr_(outerSegmentPtr),
    passAlgo_(0),
    passBitsDefaultAlgo_(0)
{
}

SDL::CPU::Segment* SDL::CPU::TrackletBase::innerSegmentPtr() const
{
    return innerSegmentPtr_;
}

SDL::CPU::Segment* SDL::CPU::TrackletBase::outerSegmentPtr() const
{
    return outerSegmentPtr_;
}

const int& SDL::CPU::TrackletBase::getPassAlgo() const
{
    return passAlgo_;
}

const int& SDL::CPU::TrackletBase::getPassBitsDefaultAlgo() const
{
    return passBitsDefaultAlgo_;
}

const std::map<std::string, float>& SDL::CPU::TrackletBase::getRecoVars() const
{
    return recovars_;
}

const float& SDL::CPU::TrackletBase::getRecoVar(std::string key) const
{
    return recovars_.at(key);
}

void SDL::CPU::TrackletBase::setRecoVars(std::string key, float var)
{
    recovars_[key] = var;
}

bool SDL::CPU::TrackletBase::isIdxMatched(const TrackletBase& md) const
{
    if (not innerSegmentPtr_->isIdxMatched(*(md.innerSegmentPtr())))
        return false;
    if (not outerSegmentPtr_->isIdxMatched(*(md.outerSegmentPtr())))
        return false;
    return true;
}

bool SDL::CPU::TrackletBase::isAnchorHitIdxMatched(const TrackletBase& md) const
{
    if (not innerSegmentPtr_->isAnchorHitIdxMatched(*(md.innerSegmentPtr())))
        return false;
    if (not outerSegmentPtr_->isAnchorHitIdxMatched(*(md.outerSegmentPtr())))
        return false;
    return true;
}

bool SDL::CPU::TrackletBase::hasCommonSegment(const TrackletBase& outer_tl) const
{
    if (outerSegmentPtr()->isIdxMatched(*(outer_tl.innerSegmentPtr())))
        return true;
    return false;
}

namespace SDL
{
    namespace CPU
    {
        std::ostream& operator<<(std::ostream& out, const TrackletBase& tl)
        {
            out << "TrackletBase" << std::endl;
            out << "------------------------------" << std::endl;
            {
                IndentingOStreambuf indent(out);
                out << "Inner Segment" << std::endl;
                out << "------------------------------" << std::endl;
                {
                    IndentingOStreambuf indent(out);
                    out << tl.innerSegmentPtr_ << std::endl;
                }
                out << "Outer Segment" << std::endl;
                out << "------------------------------" << std::endl;
                {
                    IndentingOStreambuf indent(out);
                    out << tl.outerSegmentPtr_;
                }
            }
            return out;
        }

        std::ostream& operator<<(std::ostream& out, const TrackletBase* tl)
        {
            out << *tl;
            return out;
        }
    }
}

