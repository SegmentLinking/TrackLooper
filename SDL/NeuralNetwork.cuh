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

    CUDA_DEV float runInference(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, 
                                struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, 
                                float* xVec, float* yVec, unsigned int* mdIndices, const uint16_t* lowerModuleIndices, 
                                unsigned int& innerTripletIndex, unsigned int& outerTripletIndex, 
                                float& innerRadius, float& outerRadius, float& bridgeRadius);
}
#endif
