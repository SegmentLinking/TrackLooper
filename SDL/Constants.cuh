#ifndef Constants_cuh
#define Constants_cuh

#include <alpaka/alpaka.hpp>
#include "../code/alpaka_interface/CachedBufAlloc.h"

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <cuda_fp16.h>
#endif

// Half precision wrapper functions, turned off.
#if defined(FP16_Base)
#define __F2H //__float2half
#define __H2F //__half2float
typedef /*__half*/ float FPX;
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
using QueueProperty = alpaka::NonBlocking;
using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;

Vec const elementsPerThread(Vec::all(static_cast<Idx>(1)));

// - AccGpuCudaRt
// - AccCpuThreads
// - AccCpuFibers
// - AccCpuSerial
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    using Acc = alpaka::AccGpuCudaRt<Dim, Idx>;
#elif ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
    using Acc = alpaka::AccCpuThreads<Dim, Idx>;
#elif ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED
    using Acc = alpaka::AccCpuFibers<Dim, Idx>;
#elif ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
    using Acc = alpaka::AccCpuSerial<Dim, Idx>;
#endif

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
auto const devAcc = alpaka::getDevByIdx<Acc>(0u);
using QueueAcc = alpaka::Queue<Acc, QueueProperty>;

// Buffer type for allocations where auto type can't be used.
template<typename TAcc, typename TData>
using Buf = alpaka::Buf<TAcc, TData, Dim1d, Idx>;

template<typename T, typename TAcc, typename TSize, typename TQueue>
ALPAKA_FN_HOST ALPAKA_FN_INLINE Buf<TAcc, T> allocBufWrapper(TAcc const & devAccIn, TSize nElements, TQueue queue) {
#ifdef CACHE_ALLOC
    return cms::alpakatools::allocCachedBuf<T, Idx>(devAccIn, queue, Vec1d(static_cast<Idx>(nElements)));
#else
    return alpaka::allocBuf<T, Idx>(devAccIn, Vec1d(static_cast<Idx>(nElements)));
#endif
}

template<typename T, typename TAcc, typename TSize>
ALPAKA_FN_HOST ALPAKA_FN_INLINE Buf<TAcc, T> allocBufWrapper(TAcc const & devAccIn, TSize nElements) {
    return alpaka::allocBuf<T, Idx>(devAccIn, Vec1d(static_cast<Idx>(nElements)));
}


const unsigned int MAX_BLOCKS = 80;
const unsigned int MAX_CONNECTED_MODULES = 40;
const unsigned int N_MAX_PIXEL_MD_PER_MODULES = 100000;
const unsigned int N_MAX_PIXEL_SEGMENTS_PER_MODULE = 50000;

const unsigned int N_MAX_PIXEL_TRIPLETS = 5000;
const unsigned int N_MAX_PIXEL_QUINTUPLETS = 15000;

const unsigned int N_MAX_TRACK_CANDIDATES = 1000;
const unsigned int N_MAX_PIXEL_TRACK_CANDIDATES = 4000;

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
#endif
