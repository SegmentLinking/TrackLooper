#ifndef Constants_cuh
#define Constants_cuh

#include <cuda_fp16.h>

#ifdef FP16_Base //This changes pT5 and pT3  and T3 completely. T5 for non regression parameters
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

#ifdef __CUDACC__
#define CUDA_CONST_VAR __device__
#else
#define CUDA_HOSTDEV
#define CUDA_CONST_VAR
#define CUDA_DEV
#endif


namespace SDL
{
    //defining the constant host device variables right up here
    extern CUDA_CONST_VAR const float miniMulsPtScaleBarrel[6];
    extern CUDA_CONST_VAR const float miniMulsPtScaleEndcap[5]; 
    extern CUDA_CONST_VAR const float miniRminMeanBarrel[6];
    extern CUDA_CONST_VAR const float miniRminMeanEndcap[5];
    extern CUDA_CONST_VAR const float k2Rinv1GeVf;
    extern CUDA_CONST_VAR const float kR1GeVf;
    extern CUDA_CONST_VAR const float sinAlphaMax;
    extern CUDA_CONST_VAR const float ptCut;
    extern CUDA_CONST_VAR const float deltaZLum;
    extern CUDA_CONST_VAR const float pixelPSZpitch;
    extern CUDA_CONST_VAR const float strip2SZpitch;
    extern CUDA_CONST_VAR const float pt_betaMax;
    extern CUDA_CONST_VAR const float magnetic_field;
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
