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

// Typical Buffer types used in the code.
using float_Buf = alpaka::Buf<Acc, float, Dim1d, Idx>;
using int_Buf = alpaka::Buf<Acc, int, Dim1d, Idx>;
using uint_Buf = alpaka::Buf<Acc, unsigned int, Dim1d, Idx>;
using int8_t_Buf = alpaka::Buf<Acc, int8_t, Dim1d, Idx>;
using uint16_t_Buf = alpaka::Buf<Acc, uint16_t, Dim1d, Idx>;
using char_Buf = alpaka::Buf<Acc, char, Dim1d, Idx>;
using bool_Buf = alpaka::Buf<Acc, bool, Dim1d, Idx>;

using FPX_Buf = alpaka::Buf<Acc, FPX, Dim1d, Idx>;
using FPX_T5_Buf = alpaka::Buf<Acc, FPX_T5, Dim1d, Idx>;
using FPX_dPhi_Buf = alpaka::Buf<Acc, FPX_dPhi, Dim1d, Idx>;
using FPX_circle_Buf = alpaka::Buf<Acc, FPX_circle, Dim1d, Idx>;
using FPX_seg_Buf = alpaka::Buf<Acc, FPX_seg, Dim1d, Idx>;

template<typename T, typename TAcc, typename TSize>
alpaka::Buf<TAcc, T, Dim1d, Idx> inline allocBufWrapper(TAcc const & devAcc, TSize nElements) {
    return alpaka::allocBuf<T, Idx>(devAcc, alpaka::Vec<Dim1d, Idx>(static_cast<Idx>(nElements)));
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

namespace T5DNN
{
    // Working points matching LST fake rate (43.9%) or signal acceptance (82.0%)
    CUDA_CONST_VAR const float LSTWP1 = 0.3418833f; // 94.0% TPR, 43.9% FPR
    CUDA_CONST_VAR const float LSTWP2 = 0.6177366f; // 82.0% TPR, 20.0% FPR
    // Other working points
    CUDA_CONST_VAR const float WP70   = 0.7776195f; // 70.0% TPR, 10.0% FPR
    CUDA_CONST_VAR const float WP75   = 0.7181118f; // 75.0% TPR, 13.5% FPR
    CUDA_CONST_VAR const float WP80   = 0.6492643f; // 80.0% TPR, 17.9% FPR
    CUDA_CONST_VAR const float WP85   = 0.5655319f; // 85.0% TPR, 23.8% FPR
    CUDA_CONST_VAR const float WP90   = 0.4592205f; // 90.0% TPR, 32.6% FPR
    CUDA_CONST_VAR const float WP95   = 0.3073708f; // 95.0% TPR, 47.7% FPR
    CUDA_CONST_VAR const float WP97p5 = 0.2001348f; // 97.5% TPR, 61.2% FPR
    CUDA_CONST_VAR const float WP99   = 0.1120605f; // 99.0% TPR, 75.9% FPR
    CUDA_CONST_VAR const float WP99p9 = 0.0218196f; // 99.9% TPR, 95.4% FPR
}
#endif
