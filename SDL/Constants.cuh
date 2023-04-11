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
    extern __device__ const float miniMulsPtScaleBarrel[6];
    extern __device__ const float miniMulsPtScaleEndcap[5]; 
    extern __device__ const float miniRminMeanBarrel[6];
    extern __device__ const float miniRminMeanEndcap[5];
    extern __device__ const float k2Rinv1GeVf;
    extern __device__ const float kR1GeVf;
    extern __device__ const float sinAlphaMax;
    extern __device__ const float ptCut;
    extern __device__ const float deltaZLum;
    extern __device__ const float pixelPSZpitch;
    extern __device__ const float strip2SZpitch;
    extern __device__ const float pt_betaMax;
    // Since C++ can't represent infinity, SDL_INF = 123456789 was used to represent infinity in the data table
    extern __device__ const float SDL_INF;
}
#endif