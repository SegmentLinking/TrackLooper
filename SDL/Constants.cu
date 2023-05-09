#ifdef __CUDACC__
#define CUDA_CONST_VAR __device__
#endif

#include "Constants.cuh"

//defining the constant host device variables right up here
CUDA_CONST_VAR const float SDL::miniMulsPtScaleBarrel[6] = {0.0052, 0.0038, 0.0034, 0.0034, 0.0032, 0.0034};
CUDA_CONST_VAR const float SDL::miniMulsPtScaleEndcap[5] = {0.006, 0.006, 0.006, 0.006, 0.006}; 
CUDA_CONST_VAR const float SDL::miniRminMeanBarrel[6] = {25.007152356, 37.2186993757, 52.3104270826, 68.6658656666, 85.9770373007, 108.301772384};
CUDA_CONST_VAR const float SDL::miniRminMeanEndcap[5] = {130.992832231, 154.813883559, 185.352604327, 221.635123002, 265.022076742};
CUDA_CONST_VAR const float SDL::k2Rinv1GeVf = (2.99792458e-3 * 3.8) / 2;
CUDA_CONST_VAR const float SDL::kR1GeVf = 1./(2.99792458e-3 * 3.8);
CUDA_CONST_VAR const float SDL::sinAlphaMax = 0.95;
CUDA_CONST_VAR const float SDL::ptCut = 0.8;
CUDA_CONST_VAR const float SDL::deltaZLum = 15.0;
CUDA_CONST_VAR const float SDL::pixelPSZpitch = 0.15;
CUDA_CONST_VAR const float SDL::strip2SZpitch = 5.0;
CUDA_CONST_VAR const float SDL::pt_betaMax = 7.0f;
CUDA_CONST_VAR const float SDL::magnetic_field = 3.8112;
