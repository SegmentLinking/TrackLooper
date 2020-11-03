#ifndef Hit_h
#define Hit_h

#include <iostream>
#include <cmath>
#include <vector>

#include "MathUtil.h"
#include "PrintUtil.h"

namespace SDL
{
    class Module;
}

namespace SDL
{

    class Hit
    {
        private:
            const Module* modulePtr_;

            float x_;
            float y_;
            float z_;
            float r3_; // 3d distance from origin
            float rt_; // transverse distance
            float phi_;
            float eta_;
            int idx_; // unique index to the hit index in the ntuple

            Hit* hit_high_edge_;
            Hit* hit_low_edge_;

            void setDerivedQuantities();

        public:

            // cnstr/destr
            Hit();
            Hit(float x, float y, float z, int idx=-1);
            Hit(const Hit&);
            ~Hit();

            // modifying class content
            void setX(float x);
            void setY(float y);
            void setZ(float z);
            void setXYZ(float x, float y, float z);
            void setIdx(int idx);
            void setModule(const Module*);

            // Set the boundary hits where the hits are shifted
            void setHitHighEdgePtr(Hit* hit);
            void setHitLowEdgePtr(Hit* hit);

            // accessor functions
            const float& x() const;
            const float& y() const;
            const float& z() const;
            const float& rt() const;
            const float& r3() const;
            const float& phi() const;
            const float& eta() const;
            const int& idx() const;
            const Module& getModule() const;

            // Set the boundary hits where the hits are shifted
            const Hit* getHitHighEdgePtr() const;
            const Hit* getHitLowEdgePtr() const;

            // variable computation between two hits
            float deltaPhi(const Hit&) const;
            float deltaPhiChange(const Hit&) const;
            bool isIdxMatched(const Hit&) const;

            // operator overloading
            bool operator !=(const Hit&) const;
            bool operator ==(const Hit&) const;
            Hit  operator - (const Hit&) const;
            Hit& operator = (const Hit&);
            Hit& operator -=(const Hit&);
            Hit& operator +=(const Hit&);
            Hit& operator /=(const float&);

            // cout printing
            friend std::ostream& operator<<(std::ostream& out, const Hit& hit);
            friend std::ostream& operator<<(std::ostream& out, const Hit* hit);

    };

}

#endif
