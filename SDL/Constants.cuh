#ifndef Constants_cuh
#define Constants_cuh

#include <cuda_fp16.h>
#include <alpaka/alpaka.hpp>

#ifdef FP16_Base //This changes pT5 and pT3 and T3 completely. T5 for non regression parameters
#define __F2H __float2half  
#define __H2F __half2float  
typedef __half FPX;
#else
#define __F2H
#define __H2F
typedef float FPX; 
#endif
#ifdef FP16_T5 // changes T5 regression values
#define __F2H_T5 __float2half  
#define __H2F_T5 __half2float  
typedef __half FPX_T5;
#else
#define __F2H_T5
#define __H2F_T5
typedef float FPX_T5; 
#endif
#ifdef FP16_dPhi // changes segment dPhi values
#define __F2H_dPhi __float2half  
#define __H2F_dPhi __half2float  
typedef __half FPX_dPhi;
#else
#define __F2H_dPhi
#define __H2F_dPhi
typedef float FPX_dPhi; 
#endif
#ifdef FP16_circle // changes segment circle values
#define __F2H_circle __float2half  
#define __H2F_circle __half2float  
typedef __half FPX_circle;
#else
#define __F2H_circle
#define __H2F_circle
typedef float FPX_circle; 
#endif
#ifdef FP16_seg // changes segment values
#define __F2H_seg __float2half  
#define __H2F_seg __half2float  
typedef __half FPX_seg;
#else
#define __F2H_seg
#define __H2F_seg
typedef float FPX_seg; 
#endif

using Idx = std::size_t;
using Dim = alpaka::DimInt<3u>;
using Dim1d = alpaka::DimInt<1u>;
using Vec = alpaka::Vec<Dim,Idx>;
using Vec1d = alpaka::Vec<Dim1d,Idx>;
using QueueProperty = alpaka::NonBlocking;
using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;

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

auto const devHost = alpaka::getDevByIdx<alpaka::DevCpu>(0u);
auto const devAcc = alpaka::getDevByIdx<Acc>(0u);
using QueueAcc = alpaka::Queue<Acc, QueueProperty>;

// Buffer type for allocations where auto type can't be used.
template<typename TAcc, typename TData>
using Buf = alpaka::Buf<TAcc, TData, Dim1d, Idx>;

template<typename T, typename TAcc, typename TSize>
Buf<TAcc, T> inline allocBufWrapper(TAcc const & devAccIn, TSize nElements) {
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
