#ifdef __CUDACC__
#define CUDA_CONST_VAR __device__
#endif

#include "Constants.cuh"

//defining the constant host device variables right up here
CUDA_CONST_VAR float SDL::miniMulsPtScaleBarrel[6] = {0.0052, 0.0038, 0.0034, 0.0034, 0.0032, 0.0034};
CUDA_CONST_VAR float SDL::miniMulsPtScaleEndcap[5] = {0.006, 0.006, 0.006, 0.006, 0.006}; 
#ifdef CMSSW12GEOM
CUDA_CONST_VAR float SDL::miniRminMeanBarrel[6] = {25.007152356, 37.2186993757, 52.3104270826, 68.6658656666, 85.9770373007, 108.301772384};
CUDA_CONST_VAR float SDL::miniRminMeanEndcap[5] = {130.992832231, 154.813883559, 185.352604327, 221.635123002, 265.022076742};
#else
CUDA_CONST_VAR float SDL::miniRminMeanBarrel[6] = {21.8, 34.6, 49.6, 67.4, 87.6, 106.8};
CUDA_CONST_VAR float SDL::miniRminMeanEndcap[5] = {131.4, 156.2, 185.6, 220.3, 261.5};
#endif
CUDA_CONST_VAR float SDL::k2Rinv1GeVf = (2.99792458e-3 * 3.8) / 2;
CUDA_CONST_VAR float SDL::sinAlphaMax = 0.95;
#ifdef PT0P8
CUDA_CONST_VAR float SDL::ptCut = 0.8;
#else
CUDA_CONST_VAR float SDL::ptCut = 1.0;
#endif
CUDA_CONST_VAR float SDL::deltaZLum = 15.0;
CUDA_CONST_VAR float SDL::pixelPSZpitch = 0.15;
CUDA_CONST_VAR float SDL::strip2SZpitch = 5.0;

