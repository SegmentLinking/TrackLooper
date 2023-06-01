#ifndef NeuralNetwork_cuh
#define NeuralNetwork_cuh

#ifdef __CUDACC__
#define CUDA_HOSTDEV  __host__ __device__
#define CUDA_DEV __device__
#define CUDA_CONST_VAR __device__
#else
#define CUDA_HOSTDEV
#define CUDA_DEV
#define CUDA_CONST_VAR
#endif

#include "Constants.cuh"
#include "NeuralNetworkWeights.cuh"
#include "EndcapGeometry.cuh"
#include "TiltedGeometry.h"
#include "Segment.cuh"
#include "MiniDoublet.cuh"
#include "Module.cuh"
#include "Hit.cuh"
#include "PrintUtil.h"
#include "Triplet.cuh"

namespace T5DNN
{
    CUDA_CONST_VAR const float WP82 = 0.6115213f;   // matches Chi2 signal efficiency
    CUDA_CONST_VAR const float WP90 = 0.4615373f;
    CUDA_CONST_VAR const float WP95 = 0.3243385f;
    CUDA_CONST_VAR const float WP97p5 = 0.2155140f;
    CUDA_CONST_VAR const float WP99 = 0.1253285f;
    CUDA_CONST_VAR const float WP99p9 = 0.0281462f;

    CUDA_DEV float runInference(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, float* xVec, float* yVec, unsigned int* mdIndices, const uint16_t* lowerModuleIndices, unsigned int& innerTripletIndex, unsigned int& outerTripletIndex, float& innerRadius, float& outerRadius, float& bridgeRadius);
}
#endif
