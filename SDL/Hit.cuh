#ifndef Hit_cuh
#define Hit_cuh

#include <alpaka/alpaka.hpp>

#include <iostream>
#include <cmath>
#include <vector>

#include "Module.cuh"
#include "allocate.h"
#include "PrintUtil.h"

namespace SDL
{
    struct hits
    {
        unsigned int *nHits; //single number
        //unsigned int *n2SHits;
        float *xs;
        float *ys;
        float *zs;

        uint16_t* moduleIndices;
        unsigned int* idxs;
        unsigned int* detid;
        
        float *rts;
        float* phis;
        float* etas;

        //int *edge2SMap;
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
        //void freeMemory(cudaStream_t stream);
        void freeMemoryCache();
        ~hits();

    };

    void createHitsInExplicitMemory(struct hits& hitsInGPU, int nModules, unsigned int maxHits,cudaStream_t stream,unsigned int evtnum);
    __global__ void addHitToMemoryKernel(struct hits& hitsInGPU,struct modules& modulesInGPU,const float* x,const float* y, const float* z,const uint16_t* moduleIndex,const float* phis, const int loopsize);
    void addHitToMemory(struct hits& hitsInCPU,struct modules& modulesInGPU,float x, float y, float z, unsigned int detId, unsigned int idxInNtuple,cudaStream_t stream,struct objectRanges& rangesInGPU);
    __global__ void addHitToMemoryGPU(struct hits& hitsInCPU,struct modules& modulesInGPU,float x, float y, float z, unsigned int detId, unsigned int idxInNtuple,unsigned int moduleIndex, float phis,struct objectRanges& rangesInGPU);
    
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float ATan2(float y, float x)
    {
        if (x != 0) return atan2f(y, x);
        if (y == 0) return  0;
        if (y >  0) return  float(M_PI) / 2.f;
        else        return -float(M_PI) / 2.f;
    };

    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float eta(float x, float y, float z) {
        float r3 = std::sqrt( x*x + y*y + z*z );
        float rt = std::sqrt( x*x + y*y );
        float eta = ((z > 0) - ( z < 0)) * acosh( r3 / rt );
        return eta;
    };

    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float phi_mpi_pi(float x)
    {
        if (std::isnan(x))
        {
            //printf("phi_mpi_pi() function called with NaN\n");                                                
            return x;
        }
        if (fabsf(x) <= float(M_PI))
            return x;
        constexpr float o2pi = 1.f / (2.f * float(M_PI));
        float n = std::round(x * o2pi);
        return x - n * float(2.f * float(M_PI));
    };

    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float phi(float x, float y)
    {
        return phi_mpi_pi(float(M_PI) + ATan2(-y, -x));
    };

    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float deltaPhi(float x1, float y1, float x2, float y2)
    {
        float phi1 = phi(x1,y1);
        float phi2 = phi(x2,y2);
        return phi_mpi_pi((phi2 - phi1));
    };

    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float deltaPhiChange(float x1, float y1, float x2, float y2)
    {
        return deltaPhi(x1, y1, x2-x1, y2-y1);
    };

    template<typename TAcc>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float ATan2_alpaka(TAcc const & acc, float y, float x)
    {
        if (x != 0) return alpaka::math::atan2(acc, y, x);
        if (y == 0) return  0;
        if (y >  0) return  float(M_PI) / 2.f;
        else        return -float(M_PI) / 2.f;
    };

    template<typename TAcc>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float phi_mpi_pi_alpaka(TAcc const & acc, float x)
    {
        // Alpaka: Needs to be moved over. Introduced in Alpaka 0.8.0
        if (std::isnan(x))
        {
            return x;
        }

        if (alpaka::math::abs(acc, x) <= float(M_PI))
            return x;

        constexpr float o2pi = 1.f / (2.f * float(M_PI));
        float n = alpaka::math::round(acc, x * o2pi);
        return x - n * float(2.f * float(M_PI));
    };

    template<typename TAcc>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float phi_alpaka(TAcc const & acc, float x, float y)
    {
        return phi_mpi_pi_alpaka(acc, float(M_PI) + ATan2_alpaka(acc, -y, -x));
    };

    template<typename TAcc>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float deltaPhi_alpaka(TAcc const & acc, float x1, float y1, float x2, float y2)
    {
        float phi1 = phi_alpaka(acc, x1,y1);
        float phi2 = phi_alpaka(acc, x2,y2);
        return phi_mpi_pi_alpaka(acc, (phi2 - phi1));
    };

    template<typename TAcc>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float deltaPhiChange_alpaka(TAcc const & acc, float x1, float y1, float x2, float y2)
    {
        return deltaPhi_alpaka(acc, x1, y1, x2-x1, y2-y1);
    };

    // Alpaka: This function is not yet implemented directly in Alpaka.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float copysignf_alpaka(float a, float b)
    {
        int sign_a = (a < 0) ? -1 : 1;
        int sign_b = (b < 0) ? -1 : 1;
        return sign_a * sign_b * a;
    };

    void getEdgeHits(unsigned int detId,float x, float y, float& xhigh, float& yhigh, float& xlow, float& ylow);
    ALPAKA_FN_ACC void getEdgeHitsK(float phi,float x, float y, float& xhigh, float& yhigh, float& xlow, float& ylow);

    void printHit(struct hits& hitsInGPU, struct modules& modulesInGPU, unsigned int hitIndex);
}
#endif

