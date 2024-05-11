#ifndef NeuralNetwork_cuh
#define NeuralNetwork_cuh

#ifdef LST_IS_CMSSW_PACKAGE
#include "RecoTracker/LSTCore/interface/alpaka/Constants.h"
#include "RecoTracker/LSTCore/interface/alpaka/Module.h"
#else
#include "Constants.h"
#include "Module.h"
#endif

#include "NeuralNetworkWeights.h"
#include "Segment.h"
#include "MiniDoublet.h"
#include "Hit.h"
#include "Triplet.h"

namespace T5DNN {

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float runInference(TAcc const& acc,
                                                    struct SDL::modules& modulesInGPU,
                                                    struct SDL::miniDoublets& mdsInGPU,
                                                    struct SDL::segments& segmentsInGPU,
                                                    struct SDL::triplets& tripletsInGPU,
                                                    const uint16_t* lowerModuleIndices,
                                                    const unsigned int& innerTripletIndex,
                                                    const float& innerRadius,
                                                    const float& outerRadius,
                                                    const float& bridgeRadius,
                                                    const float& rPhiChiSquared,
                                                    const float& rzChiSquared) {
    uint16_t lowerModuleIndex1 = lowerModuleIndices[0];

    // Compute some convenience variables
    short layer2_adjustment = 0;
    if (modulesInGPU.layers[lowerModuleIndex1] == 1) {
      layer2_adjustment = 1;  // get upper segment to be in second layer
    }
    unsigned int md_idx_for_t5_eta_phi =
        segmentsInGPU.mdIndices[2 * tripletsInGPU.segmentIndices[2 * innerTripletIndex + layer2_adjustment]];

    // Build DNN input vector (corresponding output N-tuple branch noted in parenthetical in comment)
    float x[6] = {alpaka::math::abs(acc, mdsInGPU.anchorEta[md_idx_for_t5_eta_phi]),  // T5 eta (t5_eta)
                  SDL::temp_log10(acc, innerRadius),   // T5 inner radius (t5_innerRadius)
                  SDL::temp_log10(acc, bridgeRadius),  // T5 bridge radius (t5_bridgeRadius)
                  SDL::temp_log10(acc, outerRadius),   // T5 outer radius (t5_outerRadius)
                  SDL::temp_log10(acc, rPhiChiSquared),
                  rzChiSquared};

    // (0): Linear(in_features=38, out_features=32, bias=True) => x = x*W_T + b
    float x_0[32];
    for (unsigned int col = 0; col < 32; ++col) {
      x_0[col] = 0;
      for (unsigned int inner = 0; inner < 6; ++inner) {
        x_0[col] += x[inner] * wgtT_0[inner][col];
      }
      x_0[col] += bias_0[col];
    }

    // (1): ReLU()
    float x_1[32];
    for (unsigned int col = 0; col < 32; ++col) {
      x_1[col] = (x_0[col] > 0.f) ? x_0[col] : 0.f;
    }

    // (2): Linear(in_features=32, out_features=32, bias=True) => x = x*W_T + b
    float x_2[32];
    for (unsigned int col = 0; col < 32; ++col) {
      x_2[col] = 0;
      for (unsigned int inner = 0; inner < 32; ++inner) {
        x_2[col] += x_1[inner] * wgtT_2[inner][col];
      }
      x_2[col] += bias_2[col];
    }

    // (3): ReLU()
    float x_3[32];
    for (unsigned int col = 0; col < 32; ++col) {
      x_3[col] = (x_2[col] > 0.f) ? x_2[col] : 0.f;
    }

    // (4): Linear(in_features=32, out_features=1, bias=True) => x = x*W_T + b
    float x_4[1];
    for (unsigned int col = 0; col < 1; ++col) {
      x_4[col] = 0;
      for (unsigned int inner = 0; inner < 32; ++inner) {
        x_4[col] += x_3[inner] * wgtT_4[inner][col];
      }
      x_4[col] += bias_4[col];
    }

    // (5): Sigmoid()
    float x_5[1];
    for (unsigned int col = 0; col < 1; ++col) {
      x_5[col] = alpaka::math::exp(acc, x_4[col]) / (alpaka::math::exp(acc, x_4[col]) + 1);
    }

    return x_5[0];
  }
}  // namespace T5DNN

#endif
