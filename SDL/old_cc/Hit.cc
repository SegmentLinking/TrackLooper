#include "Hit.h"

SDL::Hit::Hit(): x_(0), y_(0), z_(0)
{
    setDerivedQuantities();
}

SDL::Hit::Hit(float x, float y, float z, int idx): x_(x), y_(y), z_(z), idx_(idx)
{
    setDerivedQuantities();
}

SDL::Hit::Hit(const Hit& hit): x_(hit.x()), y_(hit.y()), z_(hit.z()), idx_(hit.idx())
{
    setDerivedQuantities();
}

SDL::Hit::~Hit()
{
}

void SDL::Hit::setX(float x)
{
    x_ = x;
    setDerivedQuantities();
}

void SDL::Hit::setY(float y)
{
    y_ = y;
    setDerivedQuantities();
}

void SDL::Hit::setZ(float z)
{
    z_ = z;
    setDerivedQuantities();
}

void SDL::Hit::setXYZ(float x, float y, float z)
{
    x_ = x;
    y_ = y;
    z_ = z;
    setDerivedQuantities();
}

void SDL::Hit::setIdx(int idx)
{
    idx_ = idx;
}

void SDL::Hit::setModule(const SDL::Module* module)
{
    modulePtr_ = module;
}

void SDL::Hit::setHitHighEdgePtr(SDL::Hit* hit)
{
    hit_high_edge_ = hit;
}

void SDL::Hit::setHitLowEdgePtr(SDL::Hit* hit)
{
    hit_low_edge_ = hit;
}

void SDL::Hit::setDerivedQuantities()
{

    // Setting r3
    r3_ = sqrt(x_ * x_ + y_ * y_ + z_ * z_);

    // Setting rt
    rt_ = sqrt(x_ * x_ + y_ * y_);

    // Setting phi
    phi_ = SDL::MathUtil::Phi_mpi_pi(M_PI + SDL::MathUtil::ATan2(-y_, -x_));

    // Setting eta
    eta_ = ((z_ > 0) - ( z_ < 0)) * std::acosh(r3_ / rt_);

}

const float& SDL::Hit::x() const
{
    return x_;
}

const float& SDL::Hit::y() const
{
    return y_;
}

const float& SDL::Hit::z() const
{
    return z_;
}

const float& SDL::Hit::r3() const
{
    return r3_;
}

const float& SDL::Hit::rt() const
{
    return rt_;
}

const float& SDL::Hit::phi() const
{
    return phi_;
}

const float& SDL::Hit::eta() const
{
    return eta_;
}

const int& SDL::Hit::idx() const
{
    return idx_;
}

const SDL::Module& SDL::Hit::getModule() const
{
    return (*modulePtr_);
}

// Set the boundary hits where the hits are shifted
const SDL::Hit* SDL::Hit::getHitHighEdgePtr() const
{
    if (not hit_high_edge_)
    {
        SDL::cout << "Error:: hit_high_edge_ does not exist but was asked" << std::endl;
    }
    return hit_high_edge_;
}

const SDL::Hit* SDL::Hit::getHitLowEdgePtr() const
{
    if (not hit_low_edge_)
    {
        SDL::cout << "Error:: hit_low_edge_ does not exist but was asked" << std::endl;
    }
    return hit_low_edge_;
}

float SDL::Hit::deltaPhi(const SDL::Hit& hit) const
{
    return SDL::MathUtil::Phi_mpi_pi(hit.phi() - phi_);
}

float SDL::Hit::deltaPhiChange(const SDL::Hit& hit) const
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

bool SDL::Hit::isIdxMatched(const SDL::Hit& hit) const
{
    if (idx() == -1)
        SDL::cout << "Warning:: SDL::Hit::isIdxMatched() idx of this hit is not set. Cannot perform a match." << std::endl;
    if (hit.idx() == idx())
        return true;
    return false;
}

// operators
bool SDL::Hit::operator !=(const Hit& hit) const
{
    return ((hit.x() != x_ or hit.y() != y_ or hit.z() != z_) ? true : false);
}

bool SDL::Hit::operator ==(const Hit& hit) const
{
    return ((hit.x() == x_ and hit.y() == y_ and hit.z() == z_) ? true : false);
}

SDL::Hit SDL::Hit::operator - (const Hit& hit) const
{
    return Hit(x_-hit.x(), y_-hit.y(), z_-hit.z());
}

SDL::Hit& SDL::Hit::operator = (const Hit& hit)
{
    x_ = hit.x();
    y_ = hit.y();
    z_ = hit.z();
    setDerivedQuantities();
    return *this;
}

SDL::Hit& SDL::Hit::operator -=(const Hit& hit)
{
    x_ -= hit.x();
    y_ -= hit.y();
    z_ -= hit.z();
    setDerivedQuantities();
    return *this;
}

SDL::Hit& SDL::Hit::operator +=(const Hit& hit)
{
    x_ += hit.x();
    y_ += hit.y();
    z_ += hit.z();
    setDerivedQuantities();
    return *this;
}

SDL::Hit& SDL::Hit::operator /=(const float& denom)
{
    x_ /= denom;
    y_ /= denom;
    z_ /= denom;
    setDerivedQuantities();
    return *this;
}

namespace SDL
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
