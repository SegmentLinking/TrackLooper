#ifndef Constants_cuh
#define Constants_cuh

#include <alpaka/alpaka.hpp>

#ifdef CACHE_ALLOC
#include "../code/alpaka_interface/CachedBufAlloc.h"
#endif

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <cuda_fp16.h>
#endif

// Half precision wrapper functions.
#if defined(FP16_Base)
#define __F2H __float2half
#define __H2F __half2float
typedef __half float FPX;
#else
#define __F2H
#define __H2F
typedef float FPX;
#endif

using Idx = std::size_t;
using Dim = alpaka::DimInt<3u>;
using Dim1d = alpaka::DimInt<1u>;
using Vec = alpaka::Vec<Dim,Idx>;
using Vec1d = alpaka::Vec<Dim1d,Idx>;
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  using QueueProperty = alpaka::NonBlocking;
#else
  using QueueProperty = alpaka::Blocking;
#endif
using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;

Vec const elementsPerThread(Vec::all(static_cast<Idx>(1)));

// - AccGpuCudaRt
// - AccCpuThreads
// - AccCpuSerial
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    using Acc = alpaka::AccGpuCudaRt<Dim, Idx>;
#elif ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
    using Acc = alpaka::AccCpuThreads<Dim, Idx>;
#elif ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
    using Acc = alpaka::AccCpuSerial<Dim, Idx>;
#endif

// Needed for files that are compiled by g++ to not throw an error.
// uint4 is defined only for CUDA, so we will have to revisit this soon when running on other backends.
#ifndef ALPAKA_ACC_GPU_CUDA_ENABLED
struct uint4
{
    unsigned int x;
    unsigned int y;
    unsigned int z;
    unsigned int w;
};
#endif

auto const devHost = alpaka::getDevByIdx<alpaka::DevCpu>(0u);
#if defined ALPAKA_ACC_GPU_CUDA_ENABLED || defined ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED || defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
auto const devAcc = alpaka::getDevByIdx<Acc>(0u);
using QueueAcc = alpaka::Queue<Acc, QueueProperty>;
#endif

// Buffer type for allocations where auto type can't be used.
template<typename TAcc, typename TData>
using Buf = alpaka::Buf<TAcc, TData, Dim1d, Idx>;

// Allocation wrapper function to make integration of the caching allocator easier and reduce code boilerplate.
template<typename T, typename TAcc, typename TSize, typename TQueue>
ALPAKA_FN_HOST ALPAKA_FN_INLINE Buf<TAcc, T> allocBufWrapper(TAcc const & devAccIn, TSize nElements, TQueue queue) {
#ifdef CACHE_ALLOC
    return lst::alpakatools::allocCachedBuf<T, Idx>(devAccIn, queue, Vec1d(static_cast<Idx>(nElements)));
#else
    return alpaka::allocBuf<T, Idx>(devAccIn, Vec1d(static_cast<Idx>(nElements)));
#endif
}

// Second allocation wrapper function when queue is not given. Reduces code boilerplate.
template<typename T, typename TAcc, typename TSize>
ALPAKA_FN_HOST ALPAKA_FN_INLINE Buf<TAcc, T> allocBufWrapper(TAcc const & devAccIn, TSize nElements) {
    return alpaka::allocBuf<T, Idx>(devAccIn, Vec1d(static_cast<Idx>(nElements)));
}

// Wrapper function to reduce code boilerplate for defining grid/block sizes.
ALPAKA_FN_HOST ALPAKA_FN_INLINE Vec createVec(int x, int y, int z)
{
    return Vec(static_cast<Idx>(x), static_cast<Idx>(y), static_cast<Idx>(z));
}

// Adjust grid and block sizes based on backend configuration
template <typename Vec>
ALPAKA_FN_HOST ALPAKA_FN_INLINE WorkDiv createWorkDiv(
    const Vec& blocksPerGrid, 
    const Vec& threadsPerBlock, 
    const Vec& elementsPerThread
) {
    Vec adjustedBlocks  = blocksPerGrid;
    Vec adjustedThreads = threadsPerBlock;

    // Serial execution, so all launch parameters set to 1.
#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
    adjustedBlocks  = Vec::all(static_cast<Idx>(1));
    adjustedThreads = Vec::all(static_cast<Idx>(1));
#endif

    // Threads enabled, set number of blocks to 1.
#if defined(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED)
    adjustedBlocks = Vec::all(static_cast<Idx>(1));
#endif

    return WorkDiv(adjustedBlocks, adjustedThreads, elementsPerThread);
}

const unsigned int MAX_BLOCKS = 80;
const unsigned int MAX_CONNECTED_MODULES = 40;
const unsigned int N_MAX_PIXEL_MD_PER_MODULES = 100000;
const unsigned int N_MAX_PIXEL_SEGMENTS_PER_MODULE = 50000;

const unsigned int N_MAX_PIXEL_TRIPLETS = 5000;
const unsigned int N_MAX_PIXEL_QUINTUPLETS = 15000;

const unsigned int N_MAX_PIXEL_TRACK_CANDIDATES = 30000;
const unsigned int N_MAX_NONPIXEL_TRACK_CANDIDATES = 1000;

const unsigned int N_MAX_TRACK_CANDIDATE_EXTENSIONS = 200000;
const unsigned int N_MAX_TRACK_EXTENSIONS_PER_TC = 30;
const unsigned int N_MAX_T3T3_TRACK_EXTENSIONS = 40000;

const unsigned int size_superbins = 45000;

// Temporary fix for endcap buffer allocation.
const unsigned int endcap_size = 9105;

// Temporary fix for module buffer allocation.
const unsigned int modules_size = 26401;
const unsigned int pix_tot = 1796504;

namespace SDL
{
    //defining the constant host device variables right up here
    ALPAKA_STATIC_ACC_MEM_GLOBAL const float miniMulsPtScaleBarrel[6] = {0.0052, 0.0038, 0.0034, 0.0034, 0.0032, 0.0034};
    ALPAKA_STATIC_ACC_MEM_GLOBAL const float miniMulsPtScaleEndcap[5] = {0.006, 0.006, 0.006, 0.006, 0.006}; 
    ALPAKA_STATIC_ACC_MEM_GLOBAL const float miniRminMeanBarrel[6] = {25.007152356, 37.2186993757, 52.3104270826, 68.6658656666, 85.9770373007, 108.301772384};
    ALPAKA_STATIC_ACC_MEM_GLOBAL const float miniRminMeanEndcap[5] = {130.992832231, 154.813883559, 185.352604327, 221.635123002, 265.022076742};
    ALPAKA_STATIC_ACC_MEM_GLOBAL const float k2Rinv1GeVf = (2.99792458e-3 * 3.8) / 2;
    ALPAKA_STATIC_ACC_MEM_GLOBAL const float kR1GeVf = 1./(2.99792458e-3 * 3.8);
    ALPAKA_STATIC_ACC_MEM_GLOBAL const float sinAlphaMax = 0.95;
    ALPAKA_STATIC_ACC_MEM_GLOBAL const float ptCut = 0.8;
    ALPAKA_STATIC_ACC_MEM_GLOBAL const float deltaZLum = 15.0;
    ALPAKA_STATIC_ACC_MEM_GLOBAL const float pixelPSZpitch = 0.15;
    ALPAKA_STATIC_ACC_MEM_GLOBAL const float strip2SZpitch = 5.0;
    ALPAKA_STATIC_ACC_MEM_GLOBAL const float pt_betaMax = 7.0;
    ALPAKA_STATIC_ACC_MEM_GLOBAL const float magnetic_field = 3.8112;
    // Since C++ can't represent infinity, SDL_INF = 123456789 was used to represent infinity in the data table
    ALPAKA_STATIC_ACC_MEM_GLOBAL const float SDL_INF = 123456789;
}

namespace T5DNN
{
    // Working points matching LST fake rate (43.9%) or signal acceptance (82.0%)
    ALPAKA_STATIC_ACC_MEM_GLOBAL const float LSTWP1 = 0.3418833f; // 94.0% TPR, 43.9% FPR
    ALPAKA_STATIC_ACC_MEM_GLOBAL const float LSTWP2 = 0.6177366f; // 82.0% TPR, 20.0% FPR
    // Other working points
    ALPAKA_STATIC_ACC_MEM_GLOBAL const float WP70   = 0.7776195f; // 70.0% TPR, 10.0% FPR
    ALPAKA_STATIC_ACC_MEM_GLOBAL const float WP75   = 0.7181118f; // 75.0% TPR, 13.5% FPR
    ALPAKA_STATIC_ACC_MEM_GLOBAL const float WP80   = 0.6492643f; // 80.0% TPR, 17.9% FPR
    ALPAKA_STATIC_ACC_MEM_GLOBAL const float WP85   = 0.5655319f; // 85.0% TPR, 23.8% FPR
    ALPAKA_STATIC_ACC_MEM_GLOBAL const float WP90   = 0.4592205f; // 90.0% TPR, 32.6% FPR
    ALPAKA_STATIC_ACC_MEM_GLOBAL const float WP95   = 0.3073708f; // 95.0% TPR, 47.7% FPR
    ALPAKA_STATIC_ACC_MEM_GLOBAL const float WP97p5 = 0.2001348f; // 97.5% TPR, 61.2% FPR
    ALPAKA_STATIC_ACC_MEM_GLOBAL const float WP99   = 0.1120605f; // 99.0% TPR, 75.9% FPR
    ALPAKA_STATIC_ACC_MEM_GLOBAL const float WP99p9 = 0.0218196f; // 99.9% TPR, 95.4% FPR
}
#endif
