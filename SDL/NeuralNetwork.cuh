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

    CUDA_DEV inline float runInference(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, 
                                       struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, 
                                       float* xVec, float* yVec, unsigned int* mdIndices, const uint16_t* lowerModuleIndices, 
                                       unsigned int& innerTripletIndex, unsigned int& outerTripletIndex, 
                                       float& innerRadius, float& outerRadius, float& bridgeRadius)
    {
        // Unpack x-coordinates of hits
        float x1 = xVec[0];
        float x2 = xVec[1];
        float x3 = xVec[2];
        float x4 = xVec[3];
        float x5 = xVec[4];
        // Unpack y-coordinates of hits
        float y1 = yVec[0];
        float y2 = yVec[1];
        float y3 = yVec[2];
        float y4 = yVec[3];
        float y5 = yVec[4];
        // Unpack module indices
        unsigned int mdIndex1 = mdIndices[0];
        unsigned int mdIndex2 = mdIndices[1];
        unsigned int mdIndex3 = mdIndices[2];
        unsigned int mdIndex4 = mdIndices[3];
        unsigned int mdIndex5 = mdIndices[4];
        // Unpack module indices
        uint16_t lowerModuleIndex1 = lowerModuleIndices[0];
        uint16_t lowerModuleIndex2 = lowerModuleIndices[1];
        uint16_t lowerModuleIndex3 = lowerModuleIndices[2];
        uint16_t lowerModuleIndex4 = lowerModuleIndices[3];
        uint16_t lowerModuleIndex5 = lowerModuleIndices[4];

        // Compute some convenience variables
        short layer2_adjustment = 0;
        if (modulesInGPU.layers[lowerModuleIndex1] == 1)
        {
            layer2_adjustment = 1; // get upper segment to be in second layer
        }
        unsigned int md_idx_for_t5_eta_phi = segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*innerTripletIndex + layer2_adjustment]];
        bool is_endcap1 = (modulesInGPU.subdets[lowerModuleIndex1] == 4);                // true if anchor hit 1 is in the endcap
        bool is_endcap2 = (modulesInGPU.subdets[lowerModuleIndex2] == 4);                // true if anchor hit 2 is in the endcap
        bool is_endcap3 = (modulesInGPU.subdets[lowerModuleIndex3] == 4);                // true if anchor hit 3 is in the endcap
        bool is_endcap4 = (modulesInGPU.subdets[lowerModuleIndex4] == 4);                // true if anchor hit 4 is in the endcap
        bool is_endcap5 = (modulesInGPU.subdets[lowerModuleIndex5] == 4);                // true if anchor hit 5 is in the endcap

        // Build DNN input vector (corresponding output N-tuple branch noted in parenthetical in comment)
        float x[38] = {
            log10(2*SDL::k2Rinv1GeVf*innerRadius),                                       // inner T3 pT (t3_pt)
            mdsInGPU.anchorEta[mdIndex1],                                                // inner T3 anchor hit 1 eta (t3_0_eta)
            mdsInGPU.anchorPhi[mdIndex1],                                                // inner T3 anchor hit 1 phi (t3_0_phi)
            mdsInGPU.anchorZ[mdIndex1],                                                  // inner T3 anchor hit 1 z (t3_0_z)
            sqrtf(x1*x1 + y1*y1),                                                        // inner T3 anchor hit 1 r (t3_0_r)
            float(modulesInGPU.layers[lowerModuleIndex1] + 6*is_endcap1),                // inner T3 anchor hit 1 layer (t3_0_layer)
            mdsInGPU.anchorEta[mdIndex2],                                                // inner T3 anchor hit 2 eta (t3_2_eta)
            mdsInGPU.anchorPhi[mdIndex2],                                                // inner T3 anchor hit 2 phi (t3_2_phi)
            mdsInGPU.anchorZ[mdIndex2],                                                  // inner T3 anchor hit 2 z (t3_2_z)
            sqrtf(x2*x2 + y2*y2),                                                        // inner T3 anchor hit 2 r (t3_2_r)
            float(modulesInGPU.layers[lowerModuleIndex2] + 6*is_endcap2),                // inner T3 anchor hit 2 layer (t3_2_layer)
            mdsInGPU.anchorEta[mdIndex3],                                                // inner T3 anchor hit 3 eta (t3_4_eta)
            mdsInGPU.anchorPhi[mdIndex3],                                                // inner T3 anchor hit 3 phi (t3_4_phi)
            mdsInGPU.anchorZ[mdIndex3],                                                  // inner T3 anchor hit 3 z (t3_4_z)
            sqrtf(x3*x3 + y3*y3),                                                        // inner T3 anchor hit 3 r (t3_4_r)
            float(modulesInGPU.layers[lowerModuleIndex3] + 6*is_endcap3),                // inner T3 anchor hit 3 layer (t3_4_layer)
            log10(2*SDL::k2Rinv1GeVf*outerRadius),                                       // outer T3 pT (t3_pt)
            mdsInGPU.anchorEta[mdIndex3],                                                // outer T3 anchor hit 4 eta (t3_0_eta)
            mdsInGPU.anchorPhi[mdIndex3],                                                // outer T3 anchor hit 4 phi (t3_0_phi)
            mdsInGPU.anchorZ[mdIndex3],                                                  // outer T3 anchor hit 3 eta (t3_0_z)
            sqrtf(x3*x3 + y3*y3),                                                        // outer T3 anchor hit 3 r (t3_0_r)
            float(modulesInGPU.layers[lowerModuleIndex3] + 6*is_endcap3),                // outer T3 anchor hit 3 layer (t3_0_layer)
            mdsInGPU.anchorEta[mdIndex4],                                                // outer T3 anchor hit 4 eta (t3_2_eta)
            mdsInGPU.anchorPhi[mdIndex4],                                                // outer T3 anchor hit 4 phi (t3_2_phi)
            mdsInGPU.anchorZ[mdIndex4],                                                  // outer T3 anchor hit 4 z (t3_2_z)
            sqrtf(x4*x4 + y4*y4),                                                        // outer T3 anchor hit 4 r (t3_2_r)
            float(modulesInGPU.layers[lowerModuleIndex4] + 6*is_endcap4),                // outer T3 anchor hit 4 layer (t3_2_layer)
            mdsInGPU.anchorEta[mdIndex5],                                                // outer T3 anchor hit 5 eta (t3_4_eta)
            mdsInGPU.anchorPhi[mdIndex5],                                                // outer T3 anchor hit 5 phi (t3_4_phi)
            mdsInGPU.anchorZ[mdIndex5],                                                  // outer T3 anchor hit 5 z (t3_4_z)
            sqrtf(x5*x5 + y5*y5),                                                        // outer T3 anchor hit 5 r (t3_4_r)
            float(modulesInGPU.layers[lowerModuleIndex5] + 6*is_endcap5),                // outer T3 anchor hit 5 layer (t3_4_layer)
            log10((innerRadius + outerRadius)*SDL::magnetic_field*1.602f/(2*100*5.39f)), // T5 pT (t5_pt)
            mdsInGPU.anchorEta[md_idx_for_t5_eta_phi],                                   // T5 eta (t5_eta)
            mdsInGPU.anchorPhi[md_idx_for_t5_eta_phi],                                   // T5 phi (t5_phi)
            log10(innerRadius),                                                          // T5 inner radius (t5_innerRadius)
            log10(bridgeRadius),                                                         // T5 bridge radius (t5_bridgeRadius)
            log10(outerRadius)                                                           // T5 outer radius (t5_outerRadius)
        };

        // (0): Linear(in_features=38, out_features=32, bias=True) => x = x*W_T + b
        float x_0[32];
        for (unsigned int col = 0; col < 32; ++col)
        {
            x_0[col] = 0;
            for (unsigned int inner = 0; inner < 38; ++inner)
            {
                x_0[col] += x[inner]*wgtT_0[inner][col];
            }
            x_0[col] += bias_0[col];
        }
        
        // (1): ReLU()
        float x_1[32];
        for (unsigned int col = 0; col < 32; ++col)
        {
            x_1[col] = (x_0[col] > 0.f) ? x_0[col] : 0.f;
        }
        
        // (2): Linear(in_features=32, out_features=32, bias=True) => x = x*W_T + b
        float x_2[32];
        for (unsigned int col = 0; col < 32; ++col)
        {
            x_2[col] = 0;
            for (unsigned int inner = 0; inner < 32; ++inner)
            {
                x_2[col] += x_1[inner]*wgtT_2[inner][col];
            }
            x_2[col] += bias_2[col];
        }
        
        // (3): ReLU()
        float x_3[32];
        for (unsigned int col = 0; col < 32; ++col)
        {
            x_3[col] = (x_2[col] > 0.f) ? x_2[col] : 0.f;
        }
        
        // (4): Linear(in_features=32, out_features=1, bias=True) => x = x*W_T + b
        float x_4[1];
        for (unsigned int col = 0; col < 1; ++col)
        {
            x_4[col] = 0;
            for (unsigned int inner = 0; inner < 32; ++inner)
            {
                x_4[col] += x_3[inner]*wgtT_4[inner][col];
            }
            x_4[col] += bias_4[col];
        }
        
        // (5): Sigmoid()
        float x_5[1];
        for (unsigned int col = 0; col < 1; ++col)
        {
            x_5[col] = exp(x_4[col])/(exp(x_4[col]) + 1);
        }
        
        return x_5[0];
    }
}

#endif
