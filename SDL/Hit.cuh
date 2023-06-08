#ifndef Hit_cuh
#define Hit_cuh

#include "Constants.cuh"
#include "Module.cuh"

namespace SDL
{
    struct hits
    {
        unsigned int* nHits;
        float* xs;
        float* ys;
        float* zs;
        uint16_t* moduleIndices;
        unsigned int* idxs;
        unsigned int* detid;
        float* rts;
        float* phis;
        float* etas;
        float* highEdgeXs;
        float* highEdgeYs;
        float* lowEdgeXs;
        float* lowEdgeYs;
        int* hitRanges;
        int* hitRangesLower;
        int* hitRangesUpper;
        int8_t* hitRangesnLower;
        int8_t* hitRangesnUpper;

        template<typename TBuff>
        void setData(TBuff& hitsbuf)
        {
            nHits = alpaka::getPtrNative(hitsbuf.nHits_buf);
            xs = alpaka::getPtrNative(hitsbuf.xs_buf);
            ys = alpaka::getPtrNative(hitsbuf.ys_buf);
            zs = alpaka::getPtrNative(hitsbuf.zs_buf);
            moduleIndices = alpaka::getPtrNative(hitsbuf.moduleIndices_buf);
            idxs = alpaka::getPtrNative(hitsbuf.idxs_buf);
            detid = alpaka::getPtrNative(hitsbuf.detid_buf);
            rts = alpaka::getPtrNative(hitsbuf.rts_buf);
            phis = alpaka::getPtrNative(hitsbuf.phis_buf);
            etas = alpaka::getPtrNative(hitsbuf.etas_buf);
            highEdgeXs = alpaka::getPtrNative(hitsbuf.highEdgeXs_buf);
            highEdgeYs = alpaka::getPtrNative(hitsbuf.highEdgeYs_buf);
            lowEdgeXs = alpaka::getPtrNative(hitsbuf.lowEdgeXs_buf);
            lowEdgeYs = alpaka::getPtrNative(hitsbuf.lowEdgeYs_buf);
            hitRanges = alpaka::getPtrNative(hitsbuf.hitRanges_buf);
            hitRangesLower = alpaka::getPtrNative(hitsbuf.hitRangesLower_buf);
            hitRangesUpper = alpaka::getPtrNative(hitsbuf.hitRangesUpper_buf);
            hitRangesnLower = alpaka::getPtrNative(hitsbuf.hitRangesnLower_buf);
            hitRangesnUpper = alpaka::getPtrNative(hitsbuf.hitRangesnUpper_buf);
        }
    };

    template<typename TAcc>
    struct hitsBuffer : hits
    {
        Buf<TAcc, unsigned int> nHits_buf;
        Buf<TAcc, float> xs_buf;
        Buf<TAcc, float> ys_buf;
        Buf<TAcc, float> zs_buf;
        Buf<TAcc, uint16_t> moduleIndices_buf;
        Buf<TAcc, unsigned int> idxs_buf;
        Buf<TAcc, unsigned int> detid_buf;
        Buf<TAcc, float> rts_buf;
        Buf<TAcc, float> phis_buf;
        Buf<TAcc, float> etas_buf;
        Buf<TAcc, float> highEdgeXs_buf;
        Buf<TAcc, float> highEdgeYs_buf;
        Buf<TAcc, float> lowEdgeXs_buf;
        Buf<TAcc, float> lowEdgeYs_buf;
        Buf<TAcc, int> hitRanges_buf;
        Buf<TAcc, int> hitRangesLower_buf;
        Buf<TAcc, int> hitRangesUpper_buf;
        Buf<TAcc, int8_t> hitRangesnLower_buf;
        Buf<TAcc, int8_t> hitRangesnUpper_buf;

        template<typename TQueue, typename TDevAcc>
        hitsBuffer(unsigned int nModules,
                   unsigned int nMaxHits,
                   TDevAcc const & devAccIn,
                   TQueue& queue) :
            nHits_buf(allocBufWrapper<unsigned int>(devAccIn, 1)),
            xs_buf(allocBufWrapper<float>(devAccIn, nMaxHits)),
            ys_buf(allocBufWrapper<float>(devAccIn, nMaxHits)),
            zs_buf(allocBufWrapper<float>(devAccIn, nMaxHits)),
            moduleIndices_buf(allocBufWrapper<uint16_t>(devAccIn, nMaxHits)),
            idxs_buf(allocBufWrapper<unsigned int>(devAccIn, nMaxHits)),
            detid_buf(allocBufWrapper<unsigned int>(devAccIn, nMaxHits)),
            rts_buf(allocBufWrapper<float>(devAccIn, nMaxHits)),
            phis_buf(allocBufWrapper<float>(devAccIn, nMaxHits)),
            etas_buf(allocBufWrapper<float>(devAccIn, nMaxHits)),
            highEdgeXs_buf(allocBufWrapper<float>(devAccIn, nMaxHits)),
            highEdgeYs_buf(allocBufWrapper<float>(devAccIn, nMaxHits)),
            lowEdgeXs_buf(allocBufWrapper<float>(devAccIn, nMaxHits)),
            lowEdgeYs_buf(allocBufWrapper<float>(devAccIn, nMaxHits)),
            hitRanges_buf(allocBufWrapper<int>(devAccIn, nModules*2)),
            hitRangesLower_buf(allocBufWrapper<int>(devAccIn, nModules)),
            hitRangesUpper_buf(allocBufWrapper<int>(devAccIn, nModules)),
            hitRangesnLower_buf(allocBufWrapper<int8_t>(devAccIn, nModules)),
            hitRangesnUpper_buf(allocBufWrapper<int8_t>(devAccIn, nModules))
        {
            alpaka::memset(queue, hitRanges_buf, -1, nModules*2);
            alpaka::memset(queue, hitRangesLower_buf, -1, nModules);
            alpaka::memset(queue, hitRangesUpper_buf, -1, nModules);
            alpaka::memset(queue, hitRangesnLower_buf, -1, nModules);
            alpaka::memset(queue, hitRangesnUpper_buf, -1, nModules);
            alpaka::wait(queue);
        }
    };

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

    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int binary_search(
        unsigned int *data, // Array that we are searching over
        unsigned int search_val, // Value we want to find in data array
        unsigned int ndata) // Number of elements in data array
    {
        unsigned int low = 0;
        unsigned int high = ndata - 1;

        while(low <= high)
        {
            unsigned int mid = (low + high)/2;
            unsigned int test_val = data[mid];
            if (test_val == search_val)
                return mid;
            else if (test_val > search_val)
                high = mid - 1;
            else
                low = mid + 1;
        }
        // Couldn't find search value in array.
        return -1;
    };

    struct moduleRangesKernel
    {
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(
            TAcc const & acc,
            struct SDL::modules& modulesInGPU,
            struct SDL::hits& hitsInGPU,
            int const & nLowerModules) const
        {
            using Dim = alpaka::Dim<TAcc>;
            using Idx = alpaka::Idx<TAcc>;
            using Vec = alpaka::Vec<Dim, Idx>;

            Vec const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
            Vec const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

            for(int lowerIndex = globalThreadIdx[2]; lowerIndex < nLowerModules; lowerIndex += gridThreadExtent[2])
            {
                uint16_t upperIndex = modulesInGPU.partnerModuleIndices[lowerIndex];
                if (hitsInGPU.hitRanges[lowerIndex * 2] != -1 && hitsInGPU.hitRanges[upperIndex * 2] != -1)
                {
                    hitsInGPU.hitRangesLower[lowerIndex] =  hitsInGPU.hitRanges[lowerIndex * 2]; 
                    hitsInGPU.hitRangesUpper[lowerIndex] =  hitsInGPU.hitRanges[upperIndex * 2];
                    hitsInGPU.hitRangesnLower[lowerIndex] = hitsInGPU.hitRanges[lowerIndex * 2 + 1] - hitsInGPU.hitRanges[lowerIndex * 2] + 1;
                    hitsInGPU.hitRangesnUpper[lowerIndex] = hitsInGPU.hitRanges[upperIndex * 2 + 1] - hitsInGPU.hitRanges[upperIndex * 2] + 1;
                }
            }
        }
    };

    struct hitLoopKernel
    {
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(
            TAcc const & acc,
            uint16_t Endcap, // Integer corresponding to endcap in module subdets
            uint16_t TwoS, // Integer corresponding to TwoS in moduleType
            unsigned int nModules, // Number of modules
            unsigned int nEndCapMap, // Number of elements in endcap map
            unsigned int* geoMapDetId, // DetId's from endcap map
            float* geoMapPhi, // Phi values from endcap map
            struct SDL::modules& modulesInGPU,
            struct SDL::hits& hitsInGPU,
            unsigned int const & nHits) const // Total number of hits in event
        {
            using Dim = alpaka::Dim<TAcc>;
            using Idx = alpaka::Idx<TAcc>;
            using Vec = alpaka::Vec<Dim, Idx>;

            Vec const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
            Vec const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
            for(int ihit = globalThreadIdx[2]; ihit < nHits; ihit += gridThreadExtent[2])
            {
                float ihit_x = hitsInGPU.xs[ihit];
                float ihit_y = hitsInGPU.ys[ihit];
                float ihit_z = hitsInGPU.zs[ihit];
                int iDetId = hitsInGPU.detid[ihit];

                hitsInGPU.rts[ihit] = alpaka::math::sqrt(acc, ihit_x*ihit_x + ihit_y*ihit_y);
                hitsInGPU.phis[ihit] = SDL::phi(acc, ihit_x,ihit_y);
                // Acosh has no supported implementation in Alpaka right now.
                hitsInGPU.etas[ihit] = ((ihit_z>0)-(ihit_z<0)) * SDL::temp_acosh(acc, alpaka::math::sqrt(acc, ihit_x*ihit_x+ihit_y*ihit_y+ihit_z*ihit_z)/hitsInGPU.rts[ihit]);
                int found_index = binary_search(modulesInGPU.mapdetId, iDetId, nModules);
                uint16_t lastModuleIndex = modulesInGPU.mapIdx[found_index];

                hitsInGPU.moduleIndices[ihit] = lastModuleIndex;

                if(modulesInGPU.subdets[lastModuleIndex] == Endcap && modulesInGPU.moduleType[lastModuleIndex] == TwoS)
                {
                    found_index = binary_search(geoMapDetId, iDetId, nEndCapMap);
                    float phi = 0;
                    // Unclear why these are not in map, but CPU map returns phi = 0 for all exceptions.
                    if (found_index != -1)
                        phi = geoMapPhi[found_index];
                    float cos_phi = alpaka::math::cos(acc, phi);
                    hitsInGPU.highEdgeXs[ihit] = ihit_x + 2.5f * cos_phi;
                    hitsInGPU.lowEdgeXs[ihit] = ihit_x - 2.5f * cos_phi;
                    float sin_phi = alpaka::math::sin(acc, phi);
                    hitsInGPU.highEdgeYs[ihit] = ihit_y + 2.5f * sin_phi;
                    hitsInGPU.lowEdgeYs[ihit] = ihit_y - 2.5f * sin_phi;
                }
                // Need to set initial value if index hasn't been seen before.
                int old = alpaka::atomicOp<alpaka::AtomicCas>(acc, &(hitsInGPU.hitRanges[lastModuleIndex * 2]), -1, ihit);
                // For subsequent visits, stores the min value.
                if (old != -1)
                    alpaka::atomicOp<alpaka::AtomicMin>(acc, &hitsInGPU.hitRanges[lastModuleIndex * 2], ihit);

                alpaka::atomicOp<alpaka::AtomicMax>(acc, &hitsInGPU.hitRanges[lastModuleIndex * 2 + 1], ihit);
            }
        }
    };
}
#endif