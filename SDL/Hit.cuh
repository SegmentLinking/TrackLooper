#ifndef Hit_cuh
#define Hit_cuh

#include <iostream>

#include "Constants.cuh"
#include "Module.cuh"
#include "allocate.h"

namespace SDL
{
    struct hits
    {
        unsigned int *nHits; //single number
        float *xs;
        float *ys;
        float *zs;

        uint16_t* moduleIndices;
        unsigned int* idxs;
        unsigned int* detid;
        
        float *rts;
        float* phis;
        float* etas;

        float *highEdgeXs;
        float *highEdgeYs;
        float *lowEdgeXs;
        float *lowEdgeYs;

        int* hitRanges;
        int* hitRangesLower;
        int* hitRangesUpper;
        int8_t* hitRangesnLower;
        int8_t* hitRangesnUpper;
        
        hits();
        void freeMemory();
        void freeMemoryCache();
        ~hits();
    };

    void printHit(struct hits& hitsInGPU, struct modules& modulesInGPU, unsigned int hitIndex);
    void createHitsInExplicitMemory(struct hits& hitsInGPU, int nModules, unsigned int maxHits,cudaStream_t stream,unsigned int evtnum);

    // Hyperbolic functions were just merged into Alpaka early 2023,
    // so we have to make use of temporary functions for now.
    template<typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float temp_acosh(TAcc const & acc, float val)
    {
        return alpaka::math::log(acc, val + alpaka::math::sqrt(acc, val * val - 1));
    };

    template<typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float temp_sinh(TAcc const & acc, float val)
    {
        return 0.5 * (alpaka::math::exp(acc, val) - alpaka::math::exp(acc, -val));
    };

    template<typename TAcc>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float eta(TAcc const & acc, float x, float y, float z)
    {
        float r3 = alpaka::math::sqrt(acc, x*x + y*y + z*z );
        float rt = alpaka::math::sqrt(acc, x*x + y*y );
        float eta = ((z > 0) - ( z < 0)) * temp_acosh(acc, r3 / rt );
        return eta;
    };

    template<typename TAcc>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float phi_mpi_pi(TAcc const & acc, float x)
    {
        if (alpaka::math::abs(acc, x) <= float(M_PI))
            return x;

        constexpr float o2pi = 1.f / (2.f * float(M_PI));
        float n = alpaka::math::round(acc, x * o2pi);
        return x - n * float(2.f * float(M_PI));
    };

    template<typename TAcc>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float phi(TAcc const & acc, float x, float y)
    {
        return phi_mpi_pi(acc, float(M_PI) + alpaka::math::atan2(acc, -y, -x));
    };

    template<typename TAcc>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float deltaPhi(TAcc const & acc, float x1, float y1, float x2, float y2)
    {
        float phi1 = phi(acc, x1,y1);
        float phi2 = phi(acc, x2,y2);
        return phi_mpi_pi(acc, (phi2 - phi1));
    };

    template<typename TAcc>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float deltaPhiChange(TAcc const & acc, float x1, float y1, float x2, float y2)
    {
        return deltaPhi(acc, x1, y1, x2-x1, y2-y1);
    };

    // Alpaka: This function is not yet implemented directly in Alpaka.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float copysignf(float a, float b)
    {
        int sign_a = (a < 0) ? -1 : 1;
        int sign_b = (b < 0) ? -1 : 1;
        return sign_a * sign_b * a;
    };

    ALPAKA_FN_ACC ALPAKA_FN_INLINE float calculate_dPhi(float phi1, float phi2)
    {
        // Calculate dPhi
        float dPhi = phi1 - phi2;

        // Normalize dPhi to be between -pi and pi
        if (dPhi > float(M_PI))
        {
            dPhi -= 2 * float(M_PI);
        }
        else if (dPhi < -float(M_PI))
        {
            dPhi += 2 * float(M_PI);
        }

        return dPhi;
    };
}
#endif