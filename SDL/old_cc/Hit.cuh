#ifndef Hit_h
#define Hit_h

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

#include <iostream>
#include <cmath>
#include <vector>

#include "MathUtil.cuh"
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

            CUDA_HOSTDEV void setDerivedQuantities();

        public:

            // cnstr/destr
            CUDA_HOSTDEV Hit();
            CUDA_HOSTDEV Hit(float x, float y, float z, int idx=-1);
            CUDA_HOSTDEV Hit(const Hit&);
            CUDA_HOSTDEV ~Hit();

            // modifying class content
            CUDA_HOSTDEV void setX(float x);
            CUDA_HOSTDEV void setY(float y);
            CUDA_HOSTDEV void setZ(float z);
            CUDA_HOSTDEV void setXYZ(float x, float y, float z);
            CUDA_HOSTDEV void setIdx(int idx);
            CUDA_HOSTDEV void setModule(const Module*);

            // Set the boundary hits where the hits are shifted
            void setHitHighEdgePtr(Hit* hit);
            void setHitLowEdgePtr(Hit* hit);

            // accessor functions
            CUDA_HOSTDEV const float& x() const;
            CUDA_HOSTDEV const float& y() const;
            CUDA_HOSTDEV const float& z() const;
            CUDA_HOSTDEV const float& rt() const;
            CUDA_HOSTDEV const float& r3() const;
            CUDA_HOSTDEV const float& phi() const;
            CUDA_HOSTDEV const float& eta() const;
            CUDA_HOSTDEV const int& idx() const;
            CUDA_HOSTDEV const Module& getModule() const;

            // Set the boundary hits where the hits are shifted
            CUDA_HOSTDEV const Hit* getHitHighEdgePtr() const;
            CUDA_HOSTDEV const Hit* getHitLowEdgePtr() const;

            // variable computation between two hits
            CUDA_HOSTDEV float deltaPhi(const Hit&) const;
            CUDA_HOSTDEV float deltaPhi(float x, float y, float z = 0) const;
            CUDA_HOSTDEV float deltaPhiChange(const Hit&) const;
            CUDA_HOSTDEV float deltaPhiChange(float x, float y, float z = 0) const;
            bool isIdxMatched(const Hit&) const;

            // operator overloading
            bool operator !=(const Hit&) const;
            bool operator ==(const Hit&) const;
            CUDA_HOSTDEV Hit  operator - (const Hit&) const;
            CUDA_HOSTDEV Hit& operator = (const Hit&);
            CUDA_HOSTDEV Hit& operator -=(const Hit&);
            CUDA_HOSTDEV Hit& operator +=(const Hit&);
            CUDA_HOSTDEV Hit& operator /=(const float&);

            // cout printing
            friend std::ostream& operator<<(std::ostream& out, const Hit& hit);
            friend std::ostream& operator<<(std::ostream& out, const Hit* hit);

    };

}

#endif
