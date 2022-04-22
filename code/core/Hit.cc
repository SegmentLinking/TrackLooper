#include "Hit.h"

SDL::CPU::Hit::Hit(): x_(0), y_(0), z_(0)
{
    setDerivedQuantities();
}

SDL::CPU::Hit::Hit(float x, float y, float z, int idx): x_(x), y_(y), z_(z), idx_(idx)
{
    setDerivedQuantities();
}

SDL::CPU::Hit::Hit(const Hit& hit): x_(hit.x()), y_(hit.y()), z_(hit.z()), idx_(hit.idx())
{
    setDerivedQuantities();
}

SDL::CPU::Hit::~Hit()
{
}

void SDL::CPU::Hit::setX(float x)
{
    x_ = x;
    setDerivedQuantities();
}

void SDL::CPU::Hit::setY(float y)
{
    y_ = y;
    setDerivedQuantities();
}

void SDL::CPU::Hit::setZ(float z)
{
    z_ = z;
    setDerivedQuantities();
}

void SDL::CPU::Hit::setXYZ(float x, float y, float z)
{
    x_ = x;
    y_ = y;
    z_ = z;
    setDerivedQuantities();
}

void SDL::CPU::Hit::setIdx(int idx)
{
    idx_ = idx;
}

void SDL::CPU::Hit::setModule(const SDL::CPU::Module* module)
{
    modulePtr_ = module;
}

void SDL::CPU::Hit::setHitHighEdgePtr(SDL::CPU::Hit* hit)
{
    hit_high_edge_ = hit;
}

void SDL::CPU::Hit::setHitLowEdgePtr(SDL::CPU::Hit* hit)
{
    hit_low_edge_ = hit;
}

void SDL::CPU::Hit::setDerivedQuantities()
{

    // Setting r3
    r3_ = sqrt(x_ * x_ + y_ * y_ + z_ * z_);

    // Setting rt
    rt_ = sqrt(x_ * x_ + y_ * y_);

    // Setting phi
    phi_ = SDL::CPU::MathUtil::Phi_mpi_pi(M_PI + SDL::CPU::MathUtil::ATan2(-y_, -x_));

    // Setting eta
    eta_ = ((z_ > 0) - ( z_ < 0)) * std::acosh(r3_ / rt_);

}

const float& SDL::CPU::Hit::x() const
{
    return x_;
}

const float& SDL::CPU::Hit::y() const
{
    return y_;
}

const float& SDL::CPU::Hit::z() const
{
    return z_;
}

const float& SDL::CPU::Hit::r3() const
{
    return r3_;
}

const float& SDL::CPU::Hit::rt() const
{
    return rt_;
}

const float& SDL::CPU::Hit::phi() const
{
    return phi_;
}

const float& SDL::CPU::Hit::eta() const
{
    return eta_;
}

const int& SDL::CPU::Hit::idx() const
{
    return idx_;
}

const SDL::CPU::Module& SDL::CPU::Hit::getModule() const
{
    return (*modulePtr_);
}

// Set the boundary hits where the hits are shifted
const SDL::CPU::Hit* SDL::CPU::Hit::getHitHighEdgePtr() const
{
    if (not hit_high_edge_)
    {
        SDL::CPU::cout << "Error:: hit_high_edge_ does not exist but was asked" << std::endl;
    }
    return hit_high_edge_;
}

const SDL::CPU::Hit* SDL::CPU::Hit::getHitLowEdgePtr() const
{
    if (not hit_low_edge_)
    {
        SDL::CPU::cout << "Error:: hit_low_edge_ does not exist but was asked" << std::endl;
    }
    return hit_low_edge_;
}

float SDL::CPU::Hit::deltaPhi(const SDL::CPU::Hit& hit) const
{
    return SDL::CPU::MathUtil::Phi_mpi_pi(hit.phi() - phi_);
}

float SDL::CPU::Hit::deltaPhiChange(const SDL::CPU::Hit& hit) const
{
    /*
    Compute the change in phi going from point *this -> *hit
    
     \       o <-- *hit
      \     /
       \ f /
        \^/
         o <-- *this
          \
           \
            \
             x
    
    */

    return this->deltaPhi(hit - (*this));

}

bool SDL::CPU::Hit::isIdxMatched(const SDL::CPU::Hit& hit) const
{
    if (idx() == -1)
        SDL::CPU::cout << "Warning:: SDL::CPU::Hit::isIdxMatched() idx of this hit is not set. Cannot perform a match." << std::endl;
    if (hit.idx() == idx())
        return true;
    return false;
}

// operators
bool SDL::CPU::Hit::operator !=(const Hit& hit) const
{
    return ((hit.x() != x_ or hit.y() != y_ or hit.z() != z_) ? true : false);
}

bool SDL::CPU::Hit::operator ==(const Hit& hit) const
{
    return ((hit.x() == x_ and hit.y() == y_ and hit.z() == z_) ? true : false);
}

SDL::CPU::Hit SDL::CPU::Hit::operator - (const Hit& hit) const
{
    return Hit(x_-hit.x(), y_-hit.y(), z_-hit.z());
}

SDL::CPU::Hit& SDL::CPU::Hit::operator = (const Hit& hit)
{
    x_ = hit.x();
    y_ = hit.y();
    z_ = hit.z();
    setDerivedQuantities();
    return *this;
}

SDL::CPU::Hit& SDL::CPU::Hit::operator -=(const Hit& hit)
{
    x_ -= hit.x();
    y_ -= hit.y();
    z_ -= hit.z();
    setDerivedQuantities();
    return *this;
}

SDL::CPU::Hit& SDL::CPU::Hit::operator +=(const Hit& hit)
{
    x_ += hit.x();
    y_ += hit.y();
    z_ += hit.z();
    setDerivedQuantities();
    return *this;
}

SDL::CPU::Hit& SDL::CPU::Hit::operator /=(const float& denom)
{
    x_ /= denom;
    y_ /= denom;
    z_ /= denom;
    setDerivedQuantities();
    return *this;
}

namespace SDL
{
    namespace CPU
    {
        std::ostream& operator<<(std::ostream& out, const Hit& hit)
        {
            out << "Hit(x=" << hit.x() << ", y=" << hit.y() << ", z=" << hit.z() << ", r3=" << hit.r3() << ", rt=" << hit.rt() << ", phi=" << hit.phi() << ", eta=" << hit.eta() << ")";
            return out;
        }

        std::ostream& operator<<(std::ostream& out, const Hit* hit)
        {
            out << *hit;
            return out;
        }
    }
}
