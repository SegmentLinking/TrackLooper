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
#define CUDA_CONST_VAR
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
}
#endif
