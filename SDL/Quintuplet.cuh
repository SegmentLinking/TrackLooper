#ifndef Quintuplet_cuh
#define Quintuplet_cuh

#include "Constants.cuh"
#include "EndcapGeometry.cuh"
#include "TiltedGeometry.h"
#include "Segment.cuh"
#include "MiniDoublet.cuh"
#include "Module.cuh"
#include "Hit.cuh"
#include "Triplet.cuh"

namespace SDL
{
    struct quintuplets
    {
        unsigned int* tripletIndices;
        uint16_t* lowerModuleIndices;
        int* nQuintuplets;
        int* totOccupancyQuintuplets;
        unsigned int* nMemoryLocations;

        FPX* innerRadius;
        FPX* bridgeRadius;
        FPX* outerRadius;
        FPX* pt;
        FPX* eta;
        FPX* phi;
        FPX* score_rphisum;
        uint8_t* layer;
        bool* isDup;
        bool* partOfPT5;

        float* regressionRadius;
        float* regressionG;
        float* regressionF;

        uint8_t* logicalLayers;
        unsigned int* hitIndices;
        float* rzChiSquared;
        float* chiSquared;
        float* nonAnchorChiSquared;

        quintuplets();
        ~quintuplets();
        void freeMemory(cudaStream_t stream);
        void freeMemoryCache();
    };

    void createQuintupletsInExplicitMemory(struct SDL::quintuplets& quintupletsInGPU, const unsigned int& maxQuintuplets, const uint16_t& nLowerModules, const uint16_t& nEligibleModules,cudaStream_t stream);

    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool checkIntervalOverlap(const float& firstMin, const float& firstMax, const float& secondMin, const float& secondMax)
    {
        return ((firstMin <= secondMin) & (secondMin < firstMax)) |  ((secondMin < firstMin) & (firstMin < secondMax));
    };

    ALPAKA_FN_ACC ALPAKA_FN_INLINE void addQuintupletToMemory(struct SDL::triplets& tripletsInGPU, struct SDL::quintuplets& quintupletsInGPU, unsigned int innerTripletIndex, unsigned int outerTripletIndex, uint16_t& lowerModule1, uint16_t& lowerModule2, uint16_t& lowerModule3, uint16_t& lowerModule4, uint16_t& lowerModule5, float& innerRadius, float& bridgeRadius, float& outerRadius, float& regressionG, float& regressionF, float& regressionRadius, float& rzChiSquared, float& rPhiChiSquared, float& nonAnchorChiSquared, float pt, float eta, float phi, float scores, uint8_t layer, unsigned int quintupletIndex)
    {
        quintupletsInGPU.tripletIndices[2 * quintupletIndex] = innerTripletIndex;
        quintupletsInGPU.tripletIndices[2 * quintupletIndex + 1] = outerTripletIndex;

        quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex] = lowerModule1;
        quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex + 1] = lowerModule2;
        quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex + 2] = lowerModule3;
        quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex + 3] = lowerModule4;
        quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex + 4] = lowerModule5;
        quintupletsInGPU.innerRadius[quintupletIndex] = __F2H(innerRadius);
        quintupletsInGPU.outerRadius[quintupletIndex] = __F2H(outerRadius);
        quintupletsInGPU.pt[quintupletIndex] = __F2H(pt);
        quintupletsInGPU.eta[quintupletIndex] = __F2H(eta);
        quintupletsInGPU.phi[quintupletIndex] = __F2H(phi);
        quintupletsInGPU.score_rphisum[quintupletIndex] = __F2H(scores);
        quintupletsInGPU.layer[quintupletIndex] = layer;
        quintupletsInGPU.isDup[quintupletIndex] = false;
        quintupletsInGPU.regressionRadius[quintupletIndex] = regressionRadius;
        quintupletsInGPU.regressionG[quintupletIndex] = regressionG;
        quintupletsInGPU.regressionF[quintupletIndex] = regressionF;
        quintupletsInGPU.logicalLayers[5 * quintupletIndex] = tripletsInGPU.logicalLayers[3 * innerTripletIndex];
        quintupletsInGPU.logicalLayers[5 * quintupletIndex + 1] = tripletsInGPU.logicalLayers[3 * innerTripletIndex + 1];
        quintupletsInGPU.logicalLayers[5 * quintupletIndex + 2] = tripletsInGPU.logicalLayers[3 * innerTripletIndex + 2];
        quintupletsInGPU.logicalLayers[5 * quintupletIndex + 3] = tripletsInGPU.logicalLayers[3 * outerTripletIndex + 1];
        quintupletsInGPU.logicalLayers[5 * quintupletIndex + 4] = tripletsInGPU.logicalLayers[3 * outerTripletIndex + 2];

        quintupletsInGPU.hitIndices[10 * quintupletIndex] = tripletsInGPU.hitIndices[6 * innerTripletIndex];
        quintupletsInGPU.hitIndices[10 * quintupletIndex + 1] = tripletsInGPU.hitIndices[6 * innerTripletIndex + 1];
        quintupletsInGPU.hitIndices[10 * quintupletIndex + 2] = tripletsInGPU.hitIndices[6 * innerTripletIndex + 2];
        quintupletsInGPU.hitIndices[10 * quintupletIndex + 3] = tripletsInGPU.hitIndices[6 * innerTripletIndex + 3];
        quintupletsInGPU.hitIndices[10 * quintupletIndex + 4] = tripletsInGPU.hitIndices[6 * innerTripletIndex + 4];
        quintupletsInGPU.hitIndices[10 * quintupletIndex + 5] = tripletsInGPU.hitIndices[6 * innerTripletIndex + 5];
        quintupletsInGPU.hitIndices[10 * quintupletIndex + 6] = tripletsInGPU.hitIndices[6 * outerTripletIndex + 2];
        quintupletsInGPU.hitIndices[10 * quintupletIndex + 7] = tripletsInGPU.hitIndices[6 * outerTripletIndex + 3];
        quintupletsInGPU.hitIndices[10 * quintupletIndex + 8] = tripletsInGPU.hitIndices[6 * outerTripletIndex + 4];
        quintupletsInGPU.hitIndices[10 * quintupletIndex + 9] = tripletsInGPU.hitIndices[6 * outerTripletIndex + 5];
        quintupletsInGPU.bridgeRadius[quintupletIndex] = bridgeRadius;
        quintupletsInGPU.rzChiSquared[quintupletIndex] = rzChiSquared;
        quintupletsInGPU.chiSquared[quintupletIndex] = rPhiChiSquared;
        quintupletsInGPU.nonAnchorChiSquared[quintupletIndex] = nonAnchorChiSquared;
    };

    //90% constraint
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passChiSquaredConstraint(struct SDL::modules& modulesInGPU, uint16_t& lowerModuleIndex1, uint16_t& lowerModuleIndex2, uint16_t& lowerModuleIndex3, uint16_t& lowerModuleIndex4, uint16_t& lowerModuleIndex5, float& chiSquared)
    {
        //following Philip's layer number prescription
        const int layer1 = modulesInGPU.layers[lowerModuleIndex1] + 6 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex1] == SDL::TwoS);
        const int layer2 = modulesInGPU.layers[lowerModuleIndex2] + 6 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex2] == SDL::TwoS);
        const int layer3 = modulesInGPU.layers[lowerModuleIndex3] + 6 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex3] == SDL::TwoS);
        const int layer4 = modulesInGPU.layers[lowerModuleIndex4] + 6 * (modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex4] == SDL::TwoS);
        const int layer5 = modulesInGPU.layers[lowerModuleIndex5] + 6 * (modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex5] == SDL::TwoS);

        if(layer1 == 7 and layer2 == 8 and layer3 == 9)
        {
            if(layer4 == 10 and layer5 == 11)
            {
                return chiSquared < 0.01788f;
            }
            else if(layer4 == 10 and layer5 == 16)
            {
                return chiSquared < 0.04725f;
            }
            else if(layer4 == 15 and layer5 == 16)
            {
                return chiSquared < 0.04725f;
            }
        }
        else if(layer1 == 1 and layer2 == 7 and layer3 == 8)
        {
            if(layer4 == 9 and layer5 == 10)
            {       
                return chiSquared < 0.01788f;
            }   
            else if(layer4 == 9 and layer5 == 15)
            {
                return chiSquared < 0.08234f;
            }
        }
        else if(layer1 == 1 and layer2 == 2 and layer3 == 7)
        {
            if(layer4 == 8 and layer5 == 9)
            {
                return chiSquared < 0.02360f;
            }
            else if(layer4 == 8 and layer5 == 14)
            {
                return chiSquared < 0.07167f;
            }
            else if(layer4 == 13 and layer5 == 14)
            {   
                return chiSquared < 0.08234f;
            }
        }
        else if(layer1 == 1 and layer2 == 2 and layer3 == 3)
        {
            if(layer4 == 7 and layer5 == 8)
            {
                return chiSquared < 0.01026f;
            }
            else if(layer4 == 7 and layer5 == 13)
            {
                return chiSquared < 0.06238f;
            }
            else if(layer4 == 12 and layer5 == 13)
            {
                return chiSquared < 0.06238f;
            }
        }
        else if(layer1 == 1 and layer2 == 2 and layer3 == 3 and layer4 == 4)
        {
            if(layer5 == 12)
            {
                return chiSquared < 0.09461f;
            }
            else if(layer5 == 5)
            {
                return chiSquared < 0.04725f;
            }
        }
        else if(layer1 == 2 and layer2 == 7 and layer3 == 8)
        {
            if(layer4 == 9 and layer5 == 10)
            {
                return chiSquared < 0.00512f;
            }
            if(layer4 == 9 and layer5 == 15)
            {
                return chiSquared < 0.04112f;
            }
            else if(layer4 == 14 and layer5 == 15)
            {
                return chiSquared < 0.06238f;
            }
        }
        else if(layer1 == 2 and layer2 == 3 and layer3 == 7)
        {
            if(layer4 == 8 and layer5 == 14)
            {
                return chiSquared < 0.07167f;
            }
            else if(layer4 == 13 and layer5 == 14)
            {
                return chiSquared < 0.06238f;
            }
        }
        else if(layer1 == 2 and layer2 == 3 and layer3 == 4)
        {
            if(layer4 == 12 and layer5 == 13)
            {
                return chiSquared < 0.10870f;
            }
            else if(layer4 == 5 and layer5 == 12)
            {
                return chiSquared < 0.10870f;
            }
            else if(layer4 == 5 and layer5 == 6)
            {
                return chiSquared < 0.08234f;
            }
        }
        else if(layer1 == 3 and layer2 == 7 and layer3 == 8 and layer4 == 14 and layer5 == 15)
        {
            return chiSquared < 0.09461f;
        }
        else if(layer1 == 3 and layer2 == 4 and layer3 == 5 and layer4 == 12 and layer5 == 13)
        {
            return chiSquared < 0.09461f;
        }

        return true;
    };

    //bounds can be found at http://uaf-10.t2.ucsd.edu/~bsathian/SDL/T5_RZFix/t5_rz_thresholds.txt
    template<typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passT5RZConstraint(TAcc const & acc, struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, unsigned int firstMDIndex, unsigned int secondMDIndex, unsigned int thirdMDIndex, unsigned int fourthMDIndex, unsigned int fifthMDIndex, uint16_t& lowerModuleIndex1, uint16_t& lowerModuleIndex2, uint16_t& lowerModuleIndex3, uint16_t& lowerModuleIndex4, uint16_t& lowerModuleIndex5) 
    {
        const float& rt1 = mdsInGPU.anchorRt[firstMDIndex];
        const float& rt2 = mdsInGPU.anchorRt[secondMDIndex];
        const float& rt3 = mdsInGPU.anchorRt[thirdMDIndex];
        const float& rt4 = mdsInGPU.anchorRt[fourthMDIndex];
        const float& rt5 = mdsInGPU.anchorRt[fifthMDIndex];

        const float& z1 = mdsInGPU.anchorZ[firstMDIndex];
        const float& z2 = mdsInGPU.anchorZ[secondMDIndex];
        const float& z3 = mdsInGPU.anchorZ[thirdMDIndex];
        const float& z4 = mdsInGPU.anchorZ[fourthMDIndex];
        const float& z5 = mdsInGPU.anchorZ[fifthMDIndex];

        //following Philip's layer number prescription
        const int layer1 = modulesInGPU.layers[lowerModuleIndex1] + 6 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex1] == SDL::TwoS);
        const int layer2 = modulesInGPU.layers[lowerModuleIndex2] + 6 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex2] == SDL::TwoS);
        const int layer3 = modulesInGPU.layers[lowerModuleIndex3] + 6 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex3] == SDL::TwoS);
        const int layer4 = modulesInGPU.layers[lowerModuleIndex4] + 6 * (modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex4] == SDL::TwoS);
        const int layer5 = modulesInGPU.layers[lowerModuleIndex5] + 6 * (modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex5] == SDL::TwoS);

        //slope computed using the internal T3s
        const int moduleLayer1 = modulesInGPU.moduleType[lowerModuleIndex1];
        const int moduleLayer2 = modulesInGPU.moduleType[lowerModuleIndex2];
        const int moduleLayer3 = modulesInGPU.moduleType[lowerModuleIndex3];
        const int moduleLayer4 = modulesInGPU.moduleType[lowerModuleIndex4];
        const int moduleLayer5 = modulesInGPU.moduleType[lowerModuleIndex5];

        float slope;
        if(moduleLayer1 == 0 and moduleLayer2 == 0 and moduleLayer3 == 1) //PSPS2S
        {
            slope = (z2 -z1)/(rt2 - rt1);
        }
        else
        {
            slope = (z3 - z1)/(rt3 - rt1);
        }
        float residual4 = (layer4 <= 6)? ((z4 - z1) - slope * (rt4 - rt1)) : ((rt4 - rt1) - (z4 - z1)/slope);
        float residual5 = (layer4 <= 6) ? ((z5 - z1) - slope * (rt5 - rt1)) : ((rt5 - rt1) - (z5 - z1)/slope);

        // creating a chi squared type quantity
        // 0-> PS, 1->2S
        residual4 = (moduleLayer4 == 0) ? residual4/2.4f : residual4/5.0f;
        residual5 = (moduleLayer5 == 0) ? residual5/2.4f : residual5/5.0f;

        const float RMSE = alpaka::math::sqrt(acc, 0.5 * (residual4 * residual4 + residual5 * residual5));

        //categories!
        if(layer1 == 1 and layer2 == 2 and layer3 == 3)
        {
            if(layer4 == 4 and layer5 == 5)
            {
                return RMSE < 0.545f; 
            }
            else if(layer4 == 4 and layer5 == 12)
            {
                return RMSE < 1.105f;
            }
            else if(layer4 == 7 and layer5 == 13)
            {
                return RMSE < 0.775f;
            }
            else if(layer4 == 12 and layer5 == 13)
            {
                return RMSE < 0.625f;
            }
        }
        else if(layer1 == 1 and layer2 == 2 and layer3 == 7)
        {
            if(layer4 == 8 and layer5 == 14)
            {
                return RMSE < 0.835f;
            }
            else if(layer4 == 13 and layer5 == 14)
            {
                return RMSE < 0.575f;
            }
        }
        else if(layer1 == 1 and layer2 == 7 and layer3 == 8 and layer4 == 9 and layer5 == 15)
        {
            return RMSE < 0.825f;
        }
        else if(layer1 == 2 and layer2 == 3 and layer3 == 4)
        {
            if(layer4 == 5 and layer5 == 6)
            {
                return RMSE < 0.845f;
            }
            else if(layer4 == 5 and layer5 == 12)
            {
                return RMSE < 1.365f;
            }

            else if(layer4 == 12 and layer5 == 13)
            {
                return RMSE < 0.675f;
            }
        }
        else if(layer1 == 2 and layer2 == 3 and layer3 == 7 and layer4 == 13 and layer5 == 14)
        {
            return RMSE < 0.495f;
        }
        else if(layer1 == 2 and layer2 == 3 and layer3 == 12 and layer4 == 13 and layer5 == 14)
        {
            return RMSE < 0.695f; 
        }
        else if(layer1 == 2 and layer2 == 7 and layer3 == 8)
        {
            if(layer4 == 9 and layer5 == 15)
            {
                return RMSE < 0.735f;
            }
            else if(layer4 == 14 and layer5 == 15)
            {
                return RMSE < 0.525f;
            }
        }
        else if(layer1 == 2 and layer2 == 7 and layer3 == 13 and layer4 == 14 and layer5 == 15)
        {
            return RMSE < 0.665f;
        }
        else if(layer1 == 3 and layer2 == 4 and layer3 == 5 and layer4 == 12 and layer5 == 13)
        {
            return RMSE < 0.995f;
        }
        else if(layer1 == 3 and layer2 == 4 and layer3 == 12 and layer4 == 13 and layer5 == 14)
        {
            return RMSE < 0.525f;
        }
        else if(layer1 == 3 and layer2 == 7 and layer3 == 8 and layer4 == 14 and layer5 == 15)
        {
            return RMSE < 0.525f;
        }
        else if(layer1 == 3 and layer2 == 7 and layer3 == 13 and layer4 == 14 and layer5 == 15)
        {
            return RMSE < 0.745f;
        }
        else if(layer1 == 3 and layer2 == 12 and layer3 == 13 and layer4 == 14 and layer5 == 15)
        {
            return RMSE < 0.555f; 
        }
        else if(layer1 == 7 and layer2 == 8 and layer3 == 9 and layer4 == 15 and layer5 == 16)
        {
            return RMSE < 0.525f;
        }
        else if(layer1 == 7 and layer2 == 8 and layer3 == 14 and layer4 == 15 and layer5 == 16)
        {
            return RMSE < 0.885f;
        }
        else if(layer1 == 7 and layer2 == 13 and layer3 == 14 and layer4 == 15 and layer5 == 16)
        {
            return RMSE < 0.845f;
        }

        return true;
    };

    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool T5HasCommonMiniDoublet(struct SDL::triplets& tripletsInGPU, struct SDL::segments& segmentsInGPU, unsigned int innerTripletIndex, unsigned int outerTripletIndex)
    {
        unsigned int innerOuterSegmentIndex = tripletsInGPU.segmentIndices[2 * innerTripletIndex + 1];
        unsigned int outerInnerSegmentIndex = tripletsInGPU.segmentIndices[2 * outerTripletIndex];
        unsigned int innerOuterOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * innerOuterSegmentIndex + 1]; //inner triplet outer segment outer MD index
        unsigned int outerInnerInnerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * outerInnerSegmentIndex]; //outer triplet inner segmnet inner MD index

        return (innerOuterOuterMiniDoubletIndex == outerInnerInnerMiniDoubletIndex);
    };

    template<typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float computeRadiusFromThreeAnchorHits(TAcc const & acc, float x1, float y1, float x2, float y2, float x3, float y3, float& g, float& f)
    {
        float radius = 0.f;

        //writing manual code for computing radius, which obviously sucks
        //TODO:Use fancy inbuilt libraries like cuBLAS or cuSOLVE for this!
        //(g,f) -> center
        //first anchor hit - (x1,y1), second anchor hit - (x2,y2), third anchor hit - (x3, y3)

        float denomInv = 1.0f/((y1 - y3) * (x2 - x3) - (x1 - x3) * (y2 - y3));

        float xy1sqr = x1 * x1 + y1 * y1;

        float xy2sqr = x2 * x2 + y2 * y2;

        float xy3sqr = x3 * x3 + y3 * y3;

        g = 0.5f * ((y3 - y2) * xy1sqr + (y1 - y3) * xy2sqr + (y2 - y1) * xy3sqr) * denomInv;

        f = 0.5f * ((x2 - x3) * xy1sqr + (x3 - x1) * xy2sqr + (x1 - x2) * xy3sqr) * denomInv;

        float c = ((x2 * y3 - x3 * y2) * xy1sqr + (x3 * y1 - x1 * y3) * xy2sqr + (x1 * y2 - x2 * y1) * xy3sqr) * denomInv;

        if(((y1 - y3) * (x2 - x3) - (x1 - x3) * (y2 - y3) == 0) || (g * g + f * f - c < 0))
        {
            printf("three collinear points or FATAL! r^2 < 0!\n");
            radius = -1.f;
        }
        else
            radius = alpaka::math::sqrt(acc, g * g  + f * f - c);

        return radius;
    };

    template<typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void computeErrorInRadius(TAcc const & acc, float* x1Vec, float* y1Vec, float* x2Vec, float* y2Vec, float* x3Vec, float* y3Vec, float& minimumRadius, float& maximumRadius)
    {
        //brute force
        float candidateRadius;
        float g, f;
        minimumRadius = SDL::SDL_INF;
        maximumRadius = 0.f;
        for(size_t i = 0; i < 3; i++)
        {
            float x1 = x1Vec[i];
            float y1 = y1Vec[i];
            for(size_t j = 0; j < 3; j++)
            {
                float x2 = x2Vec[j];
                float y2 = y2Vec[j];
                for(size_t k = 0; k < 3; k++)
                {
                    float x3 = x3Vec[k];
                    float y3 = y3Vec[k];
                    candidateRadius = computeRadiusFromThreeAnchorHits(acc, x1, y1, x2, y2, x3, y3, g, f);
                    maximumRadius = alpaka::math::max(acc, candidateRadius, maximumRadius);
                    minimumRadius = alpaka::math::min(acc, candidateRadius, minimumRadius);
                }
            }
        }
    };

    template<typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool matchRadiiBBBEE12378(TAcc const & acc, const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S, float& innerInvRadiusMin, float& innerInvRadiusMax, float& bridgeInvRadiusMin, float& bridgeInvRadiusMax, float& outerInvRadiusMin, float& outerInvRadiusMax)
    {
        float innerInvRadiusErrorBound = 0.178f;
        float bridgeInvRadiusErrorBound = 0.507f;
        float outerInvRadiusErrorBound = 7.655f;

        innerInvRadiusMax = (1.f + innerInvRadiusErrorBound) / innerRadius;
        innerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - innerInvRadiusErrorBound) / innerRadius);

        bridgeInvRadiusMax = (1.f + bridgeInvRadiusErrorBound) / bridgeRadius;
        bridgeInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - bridgeInvRadiusErrorBound) / bridgeRadius);

        outerInvRadiusMax = (1.f + outerInvRadiusErrorBound) / outerRadius;
        outerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - outerInvRadiusErrorBound) / outerRadius);

        return checkIntervalOverlap(innerInvRadiusMin, innerInvRadiusMax, alpaka::math::min(acc, bridgeInvRadiusMin, 1.0f/bridgeRadiusMax2S), alpaka::math::max(acc, bridgeInvRadiusMax, 1.0f/bridgeRadiusMin2S));
    };

    /*bounds for high Pt taken from : http://uaf-10.t2.ucsd.edu/~bsathian/SDL/T5_efficiency/efficiencies/new_efficiencies/efficiencies_20210513_T5_recovering_high_Pt_efficiencies/highE_radius_matching/highE_bounds.txt */
    template<typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool matchRadiiBBBBB(TAcc const & acc, const float& innerRadius, const float& bridgeRadius, const float& outerRadius, float& innerInvRadiusMin, float& innerInvRadiusMax, float& bridgeInvRadiusMin, float& bridgeInvRadiusMax, float& outerInvRadiusMin, float& outerInvRadiusMax)
    {
        float innerInvRadiusErrorBound =  0.1512f;
        float bridgeInvRadiusErrorBound = 0.1781f;
        float outerInvRadiusErrorBound = 0.1840f;

        if(innerRadius > 2.0f/(2.f * k2Rinv1GeVf))
        {
            innerInvRadiusErrorBound = 0.4449f;
            bridgeInvRadiusErrorBound = 0.4033f;
            outerInvRadiusErrorBound = 0.8016f;
        }

        innerInvRadiusMax = (1.f + innerInvRadiusErrorBound) / innerRadius;
        innerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - innerInvRadiusErrorBound) / innerRadius);

        bridgeInvRadiusMax = (1.f + bridgeInvRadiusErrorBound) / bridgeRadius;
        bridgeInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - bridgeInvRadiusErrorBound) / bridgeRadius);

        outerInvRadiusMax = (1.f + outerInvRadiusErrorBound) / outerRadius;
        outerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - outerInvRadiusErrorBound) / outerRadius);

        return checkIntervalOverlap(innerInvRadiusMin, innerInvRadiusMax, bridgeInvRadiusMin, bridgeInvRadiusMax);
    };

    template<typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool matchRadiiBBBBE(TAcc const & acc, const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S, float& innerInvRadiusMin, float& innerInvRadiusMax, float& bridgeInvRadiusMin, float& bridgeInvRadiusMax, float& outerInvRadiusMin, float& outerInvRadiusMax)
    {
        float innerInvRadiusErrorBound =  0.1781f;
        float bridgeInvRadiusErrorBound = 0.2167f;
        float outerInvRadiusErrorBound = 1.1116f;

        if(innerRadius > 2.0f/(2.f * k2Rinv1GeVf))
        {
            innerInvRadiusErrorBound = 0.4750f;
            bridgeInvRadiusErrorBound = 0.3903f;
            outerInvRadiusErrorBound = 15.2120f;
        }

        innerInvRadiusMax = (1.f + innerInvRadiusErrorBound) / innerRadius;
        innerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - innerInvRadiusErrorBound) / innerRadius);

        bridgeInvRadiusMax = (1.f + bridgeInvRadiusErrorBound) / bridgeRadius;
        bridgeInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - bridgeInvRadiusErrorBound) / bridgeRadius);

        outerInvRadiusMax = (1.f + outerInvRadiusErrorBound) / outerRadius;
        outerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - outerInvRadiusErrorBound) / outerRadius);

        return checkIntervalOverlap(innerInvRadiusMin, innerInvRadiusMax, bridgeInvRadiusMin, bridgeInvRadiusMax);
    };

    template<typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool matchRadiiBBBEE(TAcc const & acc, const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S, float& innerInvRadiusMin, float& innerInvRadiusMax, float& bridgeInvRadiusMin, float& bridgeInvRadiusMax, float& outerInvRadiusMin, float& outerInvRadiusMax)
    {
        float innerInvRadiusErrorBound =  0.1840f;
        float bridgeInvRadiusErrorBound = 0.5971f;
        float outerInvRadiusErrorBound = 11.7102f;

        if(innerRadius > 2.0f/(2.f * k2Rinv1GeVf)) //as good as no selections
        {
            innerInvRadiusErrorBound = 1.0412f;
            outerInvRadiusErrorBound = 32.2737f;
            bridgeInvRadiusErrorBound = 10.9688f;
        }

        innerInvRadiusMax = (1.f + innerInvRadiusErrorBound) / innerRadius;
        innerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - innerInvRadiusErrorBound) / innerRadius);

        bridgeInvRadiusMax = (1.f + bridgeInvRadiusErrorBound) / bridgeRadius;
        bridgeInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - bridgeInvRadiusErrorBound) / bridgeRadius);

        outerInvRadiusMax = (1.f + outerInvRadiusErrorBound) / outerRadius;
        outerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - outerInvRadiusErrorBound) / outerRadius);

        return checkIntervalOverlap(innerInvRadiusMin, innerInvRadiusMax, alpaka::math::min(acc, bridgeInvRadiusMin, 1.0f/bridgeRadiusMax2S), alpaka::math::max(acc, bridgeInvRadiusMax, 1.0f/bridgeRadiusMin2S));
    };

    template<typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool matchRadiiBBBEE23478(TAcc const & acc, const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S, float& innerInvRadiusMin, float& innerInvRadiusMax, float& bridgeInvRadiusMin, float& bridgeInvRadiusMax, float& outerInvRadiusMin, float& outerInvRadiusMax)
    {
        float innerInvRadiusErrorBound = 0.2097f;
        float bridgeInvRadiusErrorBound = 0.8557f;
        float outerInvRadiusErrorBound = 24.0450f;

        innerInvRadiusMax = (1.f + innerInvRadiusErrorBound) / innerRadius;
        innerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - innerInvRadiusErrorBound) / innerRadius);

        bridgeInvRadiusMax = (1.f + bridgeInvRadiusErrorBound) / bridgeRadius;
        bridgeInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - bridgeInvRadiusErrorBound) / bridgeRadius);

        outerInvRadiusMax = (1.f + outerInvRadiusErrorBound) / outerRadius;
        outerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - outerInvRadiusErrorBound) / outerRadius);

        return checkIntervalOverlap(innerInvRadiusMin, innerInvRadiusMax, alpaka::math::min(acc, bridgeInvRadiusMin, 1.0f/bridgeRadiusMax2S), alpaka::math::max(acc, bridgeInvRadiusMax, 1.0f/bridgeRadiusMin2S));
    };

    template<typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool matchRadiiBBBEE34578(TAcc const & acc, const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S, float& innerInvRadiusMin, float& innerInvRadiusMax, float& bridgeInvRadiusMin, float& bridgeInvRadiusMax, float& outerInvRadiusMin, float& outerInvRadiusMax)
    {
        float innerInvRadiusErrorBound = 0.066f;
        float bridgeInvRadiusErrorBound = 0.617f;
        float outerInvRadiusErrorBound = 2.688f;

        innerInvRadiusMax = (1.f + innerInvRadiusErrorBound) / innerRadius;
        innerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - innerInvRadiusErrorBound) / innerRadius);

        bridgeInvRadiusMax = (1.f + bridgeInvRadiusErrorBound) / bridgeRadius;
        bridgeInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - bridgeInvRadiusErrorBound) / bridgeRadius);

        outerInvRadiusMax = (1.f + outerInvRadiusErrorBound) / outerRadius;
        outerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - outerInvRadiusErrorBound) / outerRadius);

        return checkIntervalOverlap(innerInvRadiusMin, innerInvRadiusMax, alpaka::math::min(acc, bridgeInvRadiusMin, 1.0f/bridgeRadiusMax2S), alpaka::math::max(acc, bridgeInvRadiusMax, 1.0f/bridgeRadiusMin2S));
    };

    template<typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool matchRadiiBBEEE(TAcc const & acc, const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S, float& innerInvRadiusMin, float& innerInvRadiusMax, float& bridgeInvRadiusMin, float& bridgeInvRadiusMax, float& outerInvRadiusMin, float& outerInvRadiusMax)
    {
        float innerInvRadiusErrorBound =  0.6376f;
        float bridgeInvRadiusErrorBound = 2.1381f;
        float outerInvRadiusErrorBound = 20.4179f;

        if(innerRadius > 2.0f/(2.f * k2Rinv1GeVf)) //as good as no selections!
        {
            innerInvRadiusErrorBound = 12.9173f;
            outerInvRadiusErrorBound = 25.6702f;
            bridgeInvRadiusErrorBound = 5.1700f;
        }

        innerInvRadiusMax = (1.f + innerInvRadiusErrorBound) / innerRadius;
        innerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - innerInvRadiusErrorBound) / innerRadius);

        bridgeInvRadiusMax = (1.f + bridgeInvRadiusErrorBound) / bridgeRadius;
        bridgeInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - bridgeInvRadiusErrorBound) / bridgeRadius);

        outerInvRadiusMax = (1.f + outerInvRadiusErrorBound) / outerRadius;
        outerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - outerInvRadiusErrorBound) / outerRadius);

        return checkIntervalOverlap(innerInvRadiusMin, innerInvRadiusMax, alpaka::math::min(acc, bridgeInvRadiusMin, 1.0f/bridgeRadiusMax2S), alpaka::math::max(acc, bridgeInvRadiusMax, 1.0f/bridgeRadiusMin2S));
    };

    template<typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool matchRadiiBEEEE(TAcc const & acc, const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S, float& innerInvRadiusMin, float& innerInvRadiusMax, float& bridgeInvRadiusMin, float& bridgeInvRadiusMax, float& outerInvRadiusMin, float& outerInvRadiusMax)
    {
        float innerInvRadiusErrorBound =  1.9382f;
        float bridgeInvRadiusErrorBound = 3.7280f;
        float outerInvRadiusErrorBound = 5.7030f;

        if(innerRadius > 2.0f/(2.f * k2Rinv1GeVf))
        {
            innerInvRadiusErrorBound = 23.2713f;
            outerInvRadiusErrorBound = 24.0450f;
            bridgeInvRadiusErrorBound = 21.7980f;
        }

        innerInvRadiusMax = (1.f + innerInvRadiusErrorBound) / innerRadius;
        innerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - innerInvRadiusErrorBound) / innerRadius);

        bridgeInvRadiusMax = (1.f + bridgeInvRadiusErrorBound) / bridgeRadius;
        bridgeInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - bridgeInvRadiusErrorBound) / bridgeRadius);

        outerInvRadiusMax = (1.f + outerInvRadiusErrorBound) / outerRadius;
        outerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - outerInvRadiusErrorBound) / outerRadius);

        return checkIntervalOverlap(alpaka::math::min(acc, innerInvRadiusMin, 1.0/innerRadiusMax2S), alpaka::math::max(acc, innerInvRadiusMax, 1.0/innerRadiusMin2S), alpaka::math::min(acc, bridgeInvRadiusMin, 1.0/bridgeRadiusMax2S), alpaka::math::max(acc, bridgeInvRadiusMax, 1.0/bridgeRadiusMin2S));
    };

    template<typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool matchRadiiEEEEE(TAcc const & acc, const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S, float& innerInvRadiusMin, float& innerInvRadiusMax, float& bridgeInvRadiusMin, float& bridgeInvRadiusMax, float& outerInvRadiusMin, float& outerInvRadiusMax)
    {
        float innerInvRadiusErrorBound =  1.9382f;
        float bridgeInvRadiusErrorBound = 2.2091f;
        float outerInvRadiusErrorBound = 7.4084f;

        if(innerRadius > 2.0f/(2.f * k2Rinv1GeVf))
        {
            innerInvRadiusErrorBound = 22.5226f;
            bridgeInvRadiusErrorBound = 21.0966f;
            outerInvRadiusErrorBound = 19.1252f;
        }

        innerInvRadiusMax = (1.f + innerInvRadiusErrorBound) / innerRadius;
        innerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - innerInvRadiusErrorBound) / innerRadius);

        bridgeInvRadiusMax = (1.f + bridgeInvRadiusErrorBound) / bridgeRadius;
        bridgeInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - bridgeInvRadiusErrorBound) / bridgeRadius);

        outerInvRadiusMax = (1.f + outerInvRadiusErrorBound) / outerRadius;
        outerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - outerInvRadiusErrorBound) / outerRadius);

        return checkIntervalOverlap(alpaka::math::min(acc, innerInvRadiusMin, 1.0/innerRadiusMax2S), alpaka::math::max(acc, innerInvRadiusMax, 1.0/innerRadiusMin2S), alpaka::math::min(acc, bridgeInvRadiusMin, 1.0/bridgeRadiusMax2S), alpaka::math::max(acc, bridgeInvRadiusMax, 1.0/bridgeRadiusMin2S));
    };

    template<typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void computeSigmasForRegression(TAcc const & acc, SDL::modules& modulesInGPU, const uint16_t* lowerModuleIndices, float* delta1, float* delta2, float* slopes, bool* isFlat, int nPoints = 5, bool anchorHits = true) 
    {
        /*
        Bool anchorHits required to deal with a weird edge case wherein 
        the hits ultimately used in the regression are anchor hits, but the
        lower modules need not all be Pixel Modules (in case of PS). Similarly,
        when we compute the chi squared for the non-anchor hits, the "partner module"
        need not always be a PS strip module, but all non-anchor hits sit on strip 
        modules.
        */

        ModuleType moduleType;
        short moduleSubdet, moduleSide;
        float inv1 = 0.01f/0.009f;
        float inv2 = 0.15f/0.009f;
        float inv3 = 2.4f/0.009f;
        for(size_t i = 0; i < nPoints; i++)
        {
            moduleType = modulesInGPU.moduleType[lowerModuleIndices[i]];
            moduleSubdet = modulesInGPU.subdets[lowerModuleIndices[i]];
            moduleSide = modulesInGPU.sides[lowerModuleIndices[i]];
            float& drdz = modulesInGPU.drdzs[lowerModuleIndices[i]];
            slopes[i] = modulesInGPU.slopes[lowerModuleIndices[i]]; 
            //category 1 - barrel PS flat
            if(moduleSubdet == Barrel and moduleType == PS and moduleSide == Center)
            {
                delta1[i] = inv1;
                delta2[i] = inv1;
                slopes[i] = -999.f;
                isFlat[i] = true;
            }
            //category 2 - barrel 2S
            else if(moduleSubdet == Barrel and moduleType == TwoS)
            {
                delta1[i] = 1.f;
                delta2[i] = 1.f;
                slopes[i] = -999.f;
                isFlat[i] = true;
            }
            //category 3 - barrel PS tilted
            else if(moduleSubdet == Barrel and moduleType == PS and moduleSide != Center)
            {
                delta1[i] = inv1;
                isFlat[i] = false;

                if(anchorHits)
                {
                    delta2[i] = (inv2 * drdz/alpaka::math::sqrt(acc, 1 + drdz * drdz));
                }
                else
                {
                    delta2[i] = (inv3 * drdz/alpaka::math::sqrt(acc, 1 + drdz * drdz));
                }
            }
            //category 4 - endcap PS
            else if(moduleSubdet == Endcap and moduleType == PS)
            {
                delta1[i] = inv1;
                isFlat[i] = false;

                /*
                despite the type of the module layer of the lower module index,
                all anchor hits are on the pixel side and all non-anchor hits are
                on the strip side!
                */
                if(anchorHits)
                {
                    delta2[i] = inv2;
                }
                else
                {
                    delta2[i] = inv3;
                }
            }
            //category 5 - endcap 2S
            else if(moduleSubdet == Endcap and moduleType == TwoS)
            {
                delta1[i] = 1.f;
                delta2[i] = 500.f*inv1;
                isFlat[i] = false;
            }
            else
            {
                printf("ERROR!!!!! I SHOULDN'T BE HERE!!!! subdet = %d, type = %d, side = %d\n", moduleSubdet, moduleType, moduleSide);
            }
        }
    };

    template<typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float computeRadiusUsingRegression(TAcc const & acc, int nPoints, float* xs, float* ys, float* delta1, float* delta2, float* slopes, bool* isFlat, float& g, float& f, float* sigmas, float& chiSquared)
    {
        float radius = 0.f;

        // Some extra variables
        // the two variables will be caled x1 and x2, and y (which is x^2 + y^2)

        float sigmaX1Squared = 0.f;
        float sigmaX2Squared = 0.f;
        float sigmaX1X2 = 0.f; 
        float sigmaX1y = 0.f; 
        float sigmaX2y = 0.f;
        float sigmaY = 0.f;
        float sigmaX1 = 0.f;
        float sigmaX2 = 0.f;
        float sigmaOne = 0.f;

        float xPrime, yPrime, absArctanSlope, angleM;
        for(size_t i = 0; i < nPoints; i++)
        {
            // Computing sigmas is a very tricky affair
            // if the module is tilted or endcap, we need to use the slopes properly!

            absArctanSlope = ((slopes[i] != SDL::SDL_INF) ? alpaka::math::abs(acc, alpaka::math::atan(acc, slopes[i])) : 0.5f*float(M_PI));

            if(xs[i] > 0 and ys[i] > 0)
            {
                angleM = 0.5f*float(M_PI) - absArctanSlope;
            }
            else if(xs[i] < 0 and ys[i] > 0)
            {
                angleM = absArctanSlope + 0.5f*float(M_PI);
            }
            else if(xs[i] < 0 and ys[i] < 0)
            {
                angleM = -(absArctanSlope + 0.5f*float(M_PI));
            }
            else if(xs[i] > 0 and ys[i] < 0)
            {
                angleM = -(0.5f*float(M_PI) - absArctanSlope);
            }

            if(not isFlat[i])
            {
                xPrime = xs[i] * alpaka::math::cos(acc, angleM) + ys[i] * alpaka::math::sin(acc, angleM);
                yPrime = ys[i] * alpaka::math::cos(acc, angleM) - xs[i] * alpaka::math::sin(acc, angleM);
            }
            else
            {
                xPrime = xs[i];
                yPrime = ys[i];
            }
            sigmas[i] = 2 * alpaka::math::sqrt(acc, (xPrime * delta1[i]) * (xPrime * delta1[i]) + (yPrime * delta2[i]) * (yPrime * delta2[i]));

            sigmaX1Squared += (xs[i] * xs[i])/(sigmas[i] * sigmas[i]);
            sigmaX2Squared += (ys[i] * ys[i])/(sigmas[i] * sigmas[i]);
            sigmaX1X2 += (xs[i] * ys[i])/(sigmas[i] * sigmas[i]);
            sigmaX1y += (xs[i] * (xs[i] * xs[i] + ys[i] * ys[i]))/(sigmas[i] * sigmas[i]);
            sigmaX2y += (ys[i] * (xs[i] * xs[i] + ys[i] * ys[i]))/(sigmas[i] * sigmas[i]);
            sigmaY += (xs[i] * xs[i] + ys[i] * ys[i])/(sigmas[i] * sigmas[i]);
            sigmaX1 += xs[i]/(sigmas[i] * sigmas[i]);
            sigmaX2 += ys[i]/(sigmas[i] * sigmas[i]);
            sigmaOne += 1.0f/(sigmas[i] * sigmas[i]);
        }
        float denominator = (sigmaX1X2 - sigmaX1 * sigmaX2) * (sigmaX1X2 - sigmaX1 * sigmaX2) - (sigmaX1Squared - sigmaX1 * sigmaX1) * (sigmaX2Squared - sigmaX2 * sigmaX2);

        float twoG = ((sigmaX2y - sigmaX2 * sigmaY) * (sigmaX1X2 - sigmaX1 * sigmaX2) - (sigmaX1y - sigmaX1 * sigmaY) * (sigmaX2Squared - sigmaX2 * sigmaX2)) / denominator;
        float twoF = ((sigmaX1y - sigmaX1 * sigmaY) * (sigmaX1X2 - sigmaX1 * sigmaX2) - (sigmaX2y - sigmaX2 * sigmaY) * (sigmaX1Squared - sigmaX1 * sigmaX1)) / denominator;

        float c = -(sigmaY - twoG * sigmaX1 - twoF * sigmaX2)/sigmaOne;
        g = 0.5f * twoG;
        f = 0.5f * twoF;
        if(g * g + f * f - c < 0)
        {
            printf("FATAL! r^2 < 0!\n");
            return -1;
        }

        radius = alpaka::math::sqrt(acc, g * g + f * f - c);
        // compute chi squared
        chiSquared = 0.f;
        for(size_t i = 0; i < nPoints; i++)
        {
            chiSquared +=  (xs[i] * xs[i] + ys[i] * ys[i] - twoG * xs[i] - twoF * ys[i] + c) * (xs[i] * xs[i] + ys[i] * ys[i] - twoG * xs[i] - twoF * ys[i] + c) / (sigmas[i] * sigmas[i]);
        }
        return radius;
    };

    template<typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float computeChiSquared(TAcc const & acc, int nPoints, float* xs, float* ys, float* delta1, float* delta2, float* slopes, bool* isFlat, float g, float f, float radius)
    {
        // given values of (g, f, radius) and a set of points (and its uncertainties)
        // compute chi squared
        float c = g*g + f*f - radius*radius;
        float chiSquared = 0.f;
        float absArctanSlope, angleM, xPrime, yPrime, sigma;
        for(size_t i = 0; i < nPoints; i++)
        {
            absArctanSlope = ((slopes[i] != SDL::SDL_INF) ? alpaka::math::abs(acc, alpaka::math::atan(acc, slopes[i])) : 0.5f*float(M_PI));
            if(xs[i] > 0 and ys[i] > 0)
            {
                angleM = 0.5f*float(M_PI) - absArctanSlope;
            }
            else if(xs[i] < 0 and ys[i] > 0)
            {
                angleM = absArctanSlope + 0.5f*float(M_PI);
            }
            else if(xs[i] < 0 and ys[i] < 0)
            {
                angleM = -(absArctanSlope + 0.5f*float(M_PI));
            }
            else if(xs[i] > 0 and ys[i] < 0)
            {
                angleM = -(0.5f*float(M_PI) - absArctanSlope);
            }

            if(not isFlat[i])
            {
                xPrime = xs[i] * alpaka::math::cos(acc, angleM) + ys[i] *alpaka::math::sin(acc, angleM);
                yPrime = ys[i] * alpaka::math::cos(acc, angleM) - xs[i] *alpaka::math::sin(acc, angleM);
            }
            else
            {
                xPrime = xs[i];
                yPrime = ys[i];
            }
            sigma = 2 * alpaka::math::sqrt(acc, (xPrime * delta1[i]) * (xPrime * delta1[i]) + (yPrime * delta2[i]) * (yPrime * delta2[i]));
            chiSquared +=  (xs[i] * xs[i] + ys[i] * ys[i] - 2 * g * xs[i] - 2 * f * ys[i] + c) * (xs[i] * xs[i] + ys[i] * ys[i] - 2 * g * xs[i] - 2 * f * ys[i] + c) / (sigma * sigma);
        }
        return chiSquared; 
    };

    template<typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void runDeltaBetaIterationsT5(TAcc const & acc, float& betaIn, float& betaOut, float& betaAv, float & pt_beta, float sdIn_dr, float sdOut_dr, float dr, float lIn)
    {
        if (lIn == 0)
        {
            betaOut += SDL::copysignf(alpaka::math::asin(acc, alpaka::math::min(acc, sdOut_dr * SDL::k2Rinv1GeVf / alpaka::math::abs(acc, pt_beta), SDL::sinAlphaMax)), betaOut);
            return;
        }

        if (betaIn * betaOut > 0.f and (alpaka::math::abs(acc, pt_beta) < 4.f * SDL::pt_betaMax or (lIn >= 11 and alpaka::math::abs(acc, pt_beta) < 8.f * SDL::pt_betaMax)))   //and the pt_beta is well-defined; less strict for endcap-endcap
        {
            const float betaInUpd  = betaIn + SDL::copysignf(alpaka::math::asin(acc, alpaka::math::min(acc, sdIn_dr * SDL::k2Rinv1GeVf / alpaka::math::abs(acc, pt_beta), SDL::sinAlphaMax)), betaIn); //FIXME: need a faster version
            const float betaOutUpd = betaOut + SDL::copysignf(alpaka::math::asin(acc, alpaka::math::min(acc, sdOut_dr * SDL::k2Rinv1GeVf / alpaka::math::abs(acc, pt_beta), SDL::sinAlphaMax)), betaOut); //FIXME: need a faster version
            betaAv = 0.5f * (betaInUpd + betaOutUpd);

            //1st update
            const float pt_beta_inv = 1.f/alpaka::math::abs(acc, dr * k2Rinv1GeVf /alpaka::math::sin(acc, betaAv)); //get a better pt estimate

            betaIn  += SDL::copysignf(alpaka::math::asin(acc, alpaka::math::min(acc, sdIn_dr * SDL::k2Rinv1GeVf *pt_beta_inv, SDL::sinAlphaMax)), betaIn); //FIXME: need a faster version
            betaOut += SDL::copysignf(alpaka::math::asin(acc, alpaka::math::min(acc, sdOut_dr * SDL::k2Rinv1GeVf *pt_beta_inv, SDL::sinAlphaMax)), betaOut); //FIXME: need a faster version
            //update the av and pt
            betaAv = 0.5f * (betaIn + betaOut);
            //2nd update
            pt_beta = dr * SDL::k2Rinv1GeVf /alpaka::math::sin(acc, betaAv); //get a better pt estimate
        }
        else if (lIn < 11 && alpaka::math::abs(acc, betaOut) < 0.2f * alpaka::math::abs(acc, betaIn) && alpaka::math::abs(acc, pt_beta) < 12.f * SDL::pt_betaMax)   //use betaIn sign as ref
        {
            const float pt_betaIn = dr * k2Rinv1GeVf /alpaka::math::sin(acc, betaIn);

            const float betaInUpd  = betaIn + SDL::copysignf(alpaka::math::asin(acc, alpaka::math::min(acc, sdIn_dr * SDL::k2Rinv1GeVf / alpaka::math::abs(acc, pt_betaIn), SDL::sinAlphaMax)), betaIn); //FIXME: need a faster version
            const float betaOutUpd = betaOut + SDL::copysignf(alpaka::math::asin(acc, alpaka::math::min(acc, sdOut_dr * SDL::k2Rinv1GeVf / alpaka::math::abs(acc, pt_betaIn), SDL::sinAlphaMax)), betaIn); //FIXME: need a faster version
            betaAv = (alpaka::math::abs(acc, betaOut) > 0.2f * alpaka::math::abs(acc, betaIn)) ? (0.5f * (betaInUpd + betaOutUpd)) : betaInUpd;

            //1st update
            pt_beta = dr * SDL::k2Rinv1GeVf / alpaka::math::sin(acc, betaAv); //get a better pt estimate
            betaIn  += SDL::copysignf(alpaka::math::asin(acc, alpaka::math::min(acc, sdIn_dr * SDL::k2Rinv1GeVf / alpaka::math::abs(acc, pt_beta), SDL::sinAlphaMax)), betaIn); //FIXME: need a faster version
            betaOut += SDL::copysignf(alpaka::math::asin(acc, alpaka::math::min(acc, sdOut_dr * SDL::k2Rinv1GeVf / alpaka::math::abs(acc, pt_beta), SDL::sinAlphaMax)), betaIn); //FIXME: need a faster version
            //update the av and pt
            betaAv = 0.5f * (betaIn + betaOut);
            //2nd update
            pt_beta = dr * SDL::k2Rinv1GeVf / alpaka::math::sin(acc, betaAv); //get a better pt estimate
        }
    };

    template<typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runQuintupletDefaultAlgoBBBB(TAcc const & acc, struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& innerOuterLowerModuleIndex, uint16_t& outerInnerLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex, unsigned int& fourthMDIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& dPhi, float& betaIn, float&betaOut, float& pt_beta, float& zLo, float& zHi, float& zLoPointed, float& zHiPointed, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut)
    {
        bool pass = true;

        bool isPS_InLo = (modulesInGPU.moduleType[innerInnerLowerModuleIndex] == SDL::PS);
        bool isPS_OutLo = (modulesInGPU.moduleType[outerInnerLowerModuleIndex] == SDL::PS);

        float rt_InLo = mdsInGPU.anchorRt[firstMDIndex];
        float rt_InOut = mdsInGPU.anchorRt[secondMDIndex];
        float rt_OutLo = mdsInGPU.anchorRt[thirdMDIndex];

        float z_InLo = mdsInGPU.anchorZ[firstMDIndex];
        float z_InOut = mdsInGPU.anchorZ[secondMDIndex];
        float z_OutLo = mdsInGPU.anchorZ[thirdMDIndex];

        float alpha1GeV_OutLo = alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax));

        float rtRatio_OutLoInLo = rt_OutLo / rt_InLo; // Outer segment beginning rt divided by inner segment beginning rt;
        float dzDrtScale = alpaka::math::tan(acc, alpha1GeV_OutLo) / alpha1GeV_OutLo; // The track can bend in r-z plane slightly
        float zpitch_InLo = (isPS_InLo ? SDL::pixelPSZpitch : SDL::strip2SZpitch);
        float zpitch_OutLo = (isPS_OutLo ? SDL::pixelPSZpitch : SDL::strip2SZpitch);

        zHi = z_InLo + (z_InLo + SDL::deltaZLum) * (rtRatio_OutLoInLo - 1.f) * (z_InLo < 0.f ? 1.f : dzDrtScale) + (zpitch_InLo + zpitch_OutLo);
        zLo = z_InLo + (z_InLo - SDL::deltaZLum) * (rtRatio_OutLoInLo - 1.f) * (z_InLo > 0.f ? 1.f : dzDrtScale) - (zpitch_InLo + zpitch_OutLo);

        //Cut 1 - z compatibility
        zOut = z_OutLo;
        rtOut = rt_OutLo;
        pass = pass and ((z_OutLo >= zLo) & (z_OutLo <= zHi));
        if(not pass) return pass;

        float drt_OutLo_InLo = (rt_OutLo - rt_InLo);
        float r3_InLo = alpaka::math::sqrt(acc, z_InLo * z_InLo + rt_InLo * rt_InLo);
        float drt_InSeg = rt_InOut - rt_InLo;
        float dz_InSeg = z_InOut - z_InLo;
        float dr3_InSeg = alpaka::math::sqrt(acc, rt_InOut * rt_InOut + z_InOut * z_InOut) - alpaka::math::sqrt(acc, rt_InLo * rt_InLo + z_InLo * z_InLo);

        float coshEta = dr3_InSeg/drt_InSeg;
        float dzErr = (zpitch_InLo + zpitch_OutLo) * (zpitch_InLo + zpitch_OutLo) * 2.f;

        float sdlThetaMulsF = 0.015f * alpaka::math::sqrt(acc, 0.1f + 0.2f * (rt_OutLo - rt_InLo) / 50.f) * alpaka::math::sqrt(acc, r3_InLo / rt_InLo);
        float sdlMuls = sdlThetaMulsF * 3.f / SDL::ptCut * 4.f; // will need a better guess than x4?
        dzErr += sdlMuls * sdlMuls * drt_OutLo_InLo * drt_OutLo_InLo / 3.f * coshEta * coshEta; //sloppy
        dzErr = alpaka::math::sqrt(acc, dzErr);

        // Constructing upper and lower bound
        const float dzMean = dz_InSeg / drt_InSeg * drt_OutLo_InLo;
        const float zWindow = dzErr / drt_InSeg * drt_OutLo_InLo + (zpitch_InLo + zpitch_OutLo); //FIXME for SDL::ptCut lower than ~0.8 need to add curv path correction
        zLoPointed = z_InLo + dzMean * (z_InLo > 0.f ? 1.f : dzDrtScale) - zWindow;
        zHiPointed = z_InLo + dzMean * (z_InLo < 0.f ? 1.f : dzDrtScale) + zWindow;

        // Cut #2: Pointed Z (Inner segment two MD points to outer segment inner MD)
        pass = pass and ((z_OutLo >= zLoPointed) & (z_OutLo <= zHiPointed));
        if(not pass) return pass;

        float sdlPVoff = 0.1f/rt_OutLo;
        sdlCut = alpha1GeV_OutLo + alpaka::math::sqrt(acc, sdlMuls * sdlMuls + sdlPVoff * sdlPVoff);

        deltaPhiPos = SDL::deltaPhi(acc, mdsInGPU.anchorX[secondMDIndex], mdsInGPU.anchorY[secondMDIndex], mdsInGPU.anchorX[fourthMDIndex], mdsInGPU.anchorY[fourthMDIndex]);
        // Cut #3: FIXME:deltaPhiPos can be tighter
        pass = pass and (alpaka::math::abs(acc, deltaPhiPos) <= sdlCut);
        if(not pass) return pass;

        float midPointX = 0.5f*(mdsInGPU.anchorX[firstMDIndex] + mdsInGPU.anchorX[thirdMDIndex]);
        float midPointY = 0.5f* (mdsInGPU.anchorY[firstMDIndex] + mdsInGPU.anchorY[thirdMDIndex]);
        float midPointZ = 0.5f*(mdsInGPU.anchorZ[firstMDIndex] + mdsInGPU.anchorZ[thirdMDIndex]);
        float diffX = mdsInGPU.anchorX[thirdMDIndex] - mdsInGPU.anchorX[firstMDIndex];
        float diffY = mdsInGPU.anchorY[thirdMDIndex] - mdsInGPU.anchorY[firstMDIndex];
        float diffZ = mdsInGPU.anchorZ[thirdMDIndex] - mdsInGPU.anchorZ[firstMDIndex];

        dPhi = SDL::deltaPhi(acc, midPointX, midPointY, diffX, diffY);

        // Cut #4: deltaPhiChange
        pass = pass and (alpaka::math::abs(acc, dPhi) <= sdlCut);
        //lots of array accesses below. Cut here!
        if(not pass) return pass;

        // First obtaining the raw betaIn and betaOut values without any correction and just purely based on the mini-doublet hit positions

        float alpha_InLo  = __H2F(segmentsInGPU.dPhiChanges[innerSegmentIndex]);
        float alpha_OutLo = __H2F(segmentsInGPU.dPhiChanges[outerSegmentIndex]);

        bool isEC_lastLayer = modulesInGPU.subdets[outerOuterLowerModuleIndex] == SDL::Endcap and modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::TwoS;

        float alpha_OutUp,alpha_OutUp_highEdge,alpha_OutUp_lowEdge;

        alpha_OutUp = SDL::deltaPhi(acc, mdsInGPU.anchorX[fourthMDIndex], mdsInGPU.anchorY[fourthMDIndex], mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex], mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]);

        alpha_OutUp_highEdge = alpha_OutUp;
        alpha_OutUp_lowEdge = alpha_OutUp;

        float tl_axis_x = mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[firstMDIndex];
        float tl_axis_y = mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[firstMDIndex];
        float tl_axis_z = mdsInGPU.anchorZ[fourthMDIndex] - mdsInGPU.anchorZ[firstMDIndex];
        float tl_axis_highEdge_x = tl_axis_x;
        float tl_axis_highEdge_y = tl_axis_y;
        float tl_axis_lowEdge_x = tl_axis_x;
        float tl_axis_lowEdge_y = tl_axis_y;

        betaIn = alpha_InLo - SDL::deltaPhi(acc, mdsInGPU.anchorX[firstMDIndex], mdsInGPU.anchorY[firstMDIndex], tl_axis_x, tl_axis_y);

        float betaInRHmin = betaIn;
        float betaInRHmax = betaIn;
        betaOut = -alpha_OutUp + SDL::deltaPhi(acc, mdsInGPU.anchorX[fourthMDIndex], mdsInGPU.anchorY[fourthMDIndex], tl_axis_x, tl_axis_y);

        float betaOutRHmin = betaOut;
        float betaOutRHmax = betaOut;

        if(isEC_lastLayer)
        {
            alpha_OutUp_highEdge = SDL::deltaPhi(acc, mdsInGPU.anchorHighEdgeX[fourthMDIndex], mdsInGPU.anchorHighEdgeY[fourthMDIndex], mdsInGPU.anchorHighEdgeX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex], mdsInGPU.anchorHighEdgeY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]);
            alpha_OutUp_lowEdge = SDL::deltaPhi(acc, mdsInGPU.anchorLowEdgeX[fourthMDIndex], mdsInGPU.anchorLowEdgeY[fourthMDIndex], mdsInGPU.anchorLowEdgeX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex], mdsInGPU.anchorLowEdgeY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]);

            tl_axis_highEdge_x = mdsInGPU.anchorHighEdgeX[fourthMDIndex] - mdsInGPU.anchorX[firstMDIndex];
            tl_axis_highEdge_y = mdsInGPU.anchorHighEdgeY[fourthMDIndex] - mdsInGPU.anchorY[firstMDIndex];
            tl_axis_lowEdge_x = mdsInGPU.anchorLowEdgeX[fourthMDIndex] - mdsInGPU.anchorX[firstMDIndex];
            tl_axis_lowEdge_y = mdsInGPU.anchorLowEdgeY[fourthMDIndex] - mdsInGPU.anchorY[firstMDIndex];


            betaOutRHmin = -alpha_OutUp_highEdge + SDL::deltaPhi(acc, mdsInGPU.anchorHighEdgeX[fourthMDIndex], mdsInGPU.anchorHighEdgeY[fourthMDIndex], tl_axis_highEdge_x, tl_axis_highEdge_y);
            betaOutRHmax = -alpha_OutUp_lowEdge + SDL::deltaPhi(acc, mdsInGPU.anchorLowEdgeX[fourthMDIndex], mdsInGPU.anchorLowEdgeY[fourthMDIndex], tl_axis_lowEdge_x, tl_axis_lowEdge_y);
        }

        //beta computation
        float drt_tl_axis = alpaka::math::sqrt(acc, tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);
        float drt_tl_lowEdge = alpaka::math::sqrt(acc, tl_axis_lowEdge_x * tl_axis_lowEdge_x + tl_axis_lowEdge_y * tl_axis_lowEdge_y);
        float drt_tl_highEdge = alpaka::math::sqrt(acc, tl_axis_highEdge_x * tl_axis_highEdge_x + tl_axis_highEdge_y * tl_axis_highEdge_y);

        float corrF = 1.f;
        //innerOuterAnchor - innerInnerAnchor
        const float rt_InSeg = alpaka::math::sqrt(acc, (mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) * (mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) + (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]) * (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]));
        betaInCut = alpaka::math::asin(acc, alpaka::math::min(acc, (-rt_InSeg * corrF + drt_tl_axis) * SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax)) + (0.02f / drt_InSeg);

        //Cut #5: first beta cut
        pass = pass and (alpaka::math::abs(acc, betaInRHmin) < betaInCut);
        if(not pass) return pass;

        float betaAv = 0.5f * (betaIn + betaOut);
        pt_beta = drt_tl_axis * SDL::k2Rinv1GeVf/alpaka::math::sin(acc, betaAv);
        int lIn = 5;
        int lOut = isEC_lastLayer ? 11 : 5;
        float sdOut_dr = alpaka::math::sqrt(acc, (mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex]) * (mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex]) + (mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]) * (mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]));
        float sdOut_d = mdsInGPU.anchorRt[fourthMDIndex] - mdsInGPU.anchorRt[thirdMDIndex];

        const float diffDr = alpaka::math::abs(acc, rt_InSeg - sdOut_dr) / alpaka::math::abs(acc, rt_InSeg + sdOut_dr);

        SDL::runDeltaBetaIterationsT5(acc, betaIn, betaOut, betaAv, pt_beta, rt_InSeg, sdOut_dr, drt_tl_axis, lIn);

        const float betaInMMSF = (alpaka::math::abs(acc, betaInRHmin + betaInRHmax) > 0) ? (2.f * betaIn / alpaka::math::abs(acc, betaInRHmin + betaInRHmax)) : 0.f; //mean value of min,max is the old betaIn
        const float betaOutMMSF = (alpaka::math::abs(acc, betaOutRHmin + betaOutRHmax) > 0) ? (2.f * betaOut / alpaka::math::abs(acc, betaOutRHmin + betaOutRHmax)) : 0.f;
        betaInRHmin *= betaInMMSF;
        betaInRHmax *= betaInMMSF;
        betaOutRHmin *= betaOutMMSF;
        betaOutRHmax *= betaOutMMSF;

        const float dBetaMuls = sdlThetaMulsF * 4.f / alpaka::math::min(acc, alpaka::math::abs(acc, pt_beta), SDL::pt_betaMax); //need to confimm the range-out value of 7 GeV

        const float alphaInAbsReg = alpaka::math::max(acc, alpaka::math::abs(acc, alpha_InLo), alpaka::math::asin(acc, alpaka::math::min(acc, rt_InLo * SDL::k2Rinv1GeVf / 3.0f, SDL::sinAlphaMax)));
        const float alphaOutAbsReg = alpaka::math::max(acc, alpaka::math::abs(acc, alpha_OutLo), alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * SDL::k2Rinv1GeVf / 3.0f, SDL::sinAlphaMax)));
        const float dBetaInLum = lIn < 11 ? 0.0f : alpaka::math::abs(acc, alphaInAbsReg*SDL::deltaZLum / z_InLo);
        const float dBetaOutLum = lOut < 11 ? 0.0f : alpaka::math::abs(acc, alphaOutAbsReg*SDL::deltaZLum / z_OutLo);
        const float dBetaLum2 = (dBetaInLum + dBetaOutLum) * (dBetaInLum + dBetaOutLum);
        const float sinDPhi =alpaka::math::sin(acc, dPhi);

        const float dBetaRIn2 = 0; // TODO-RH
        float dBetaROut = 0;
        if(isEC_lastLayer)
        {
            dBetaROut = (alpaka::math::sqrt(acc, mdsInGPU.anchorHighEdgeX[fourthMDIndex] * mdsInGPU.anchorHighEdgeX[fourthMDIndex] + mdsInGPU.anchorHighEdgeY[fourthMDIndex] * mdsInGPU.anchorHighEdgeY[fourthMDIndex]) - alpaka::math::sqrt(acc, mdsInGPU.anchorLowEdgeX[fourthMDIndex] * mdsInGPU.anchorLowEdgeX[fourthMDIndex] + mdsInGPU.anchorLowEdgeY[fourthMDIndex] * mdsInGPU.anchorLowEdgeY[fourthMDIndex])) * sinDPhi / drt_tl_axis;
        }

        const float dBetaROut2 =  dBetaROut * dBetaROut;

        //FIXME: need faster version
        betaOutCut = alpaka::math::asin(acc, alpaka::math::min(acc, drt_tl_axis*SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax)) + (0.02f / sdOut_d) + alpaka::math::sqrt(acc, dBetaLum2 + dBetaMuls*dBetaMuls);

        //Cut #6: The real beta cut
        pass = pass and ((alpaka::math::abs(acc, betaOut) < betaOutCut));
        if(not pass) return pass;

        float pt_betaIn = drt_tl_axis * SDL::k2Rinv1GeVf/alpaka::math::sin(acc, betaIn);
        float pt_betaOut = drt_tl_axis * SDL::k2Rinv1GeVf /alpaka::math::sin(acc, betaOut);
        float dBetaRes = 0.02f/alpaka::math::min(acc, sdOut_d,drt_InSeg);
        float dBetaCut2 = (dBetaRes*dBetaRes * 2.0f + dBetaMuls * dBetaMuls + dBetaLum2 + dBetaRIn2 + dBetaROut2 + 0.25f * (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)) * (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)));

        float dBeta = betaIn - betaOut;
        deltaBetaCut = alpaka::math::sqrt(acc, dBetaCut2);
        pass = pass and (dBeta * dBeta <= dBetaCut2);

        return pass;
    };

    template<typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runQuintupletDefaultAlgoBBEE(TAcc const & acc, struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& innerOuterLowerModuleIndex, uint16_t& outerInnerLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex, unsigned int& fourthMDIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& dPhi, float& betaIn, float&betaOut, float& pt_beta, float& zLo, float& rtLo, float& rtHi, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ)
    {
        bool pass = true;
        bool isPS_InLo = (modulesInGPU.moduleType[innerInnerLowerModuleIndex] == SDL::PS);
        bool isPS_OutLo = (modulesInGPU.moduleType[outerInnerLowerModuleIndex] == SDL::PS);

        float rt_InLo = mdsInGPU.anchorRt[firstMDIndex];
        float rt_InOut = mdsInGPU.anchorRt[secondMDIndex];
        float rt_OutLo = mdsInGPU.anchorRt[thirdMDIndex];

        float z_InLo = mdsInGPU.anchorZ[firstMDIndex];
        float z_InOut = mdsInGPU.anchorZ[secondMDIndex];
        float z_OutLo = mdsInGPU.anchorZ[thirdMDIndex];

        float alpha1GeV_OutLo = alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax));

        float rtRatio_OutLoInLo = rt_OutLo / rt_InLo; // Outer segment beginning rt divided by inner segment beginning rt;
        float dzDrtScale = alpaka::math::tan(acc, alpha1GeV_OutLo) / alpha1GeV_OutLo; // The track can bend in r-z plane slightly
        float zpitch_InLo = (isPS_InLo ? SDL::pixelPSZpitch : SDL::strip2SZpitch);
        float zpitch_OutLo = (isPS_OutLo ? SDL::pixelPSZpitch : SDL::strip2SZpitch);
        float zGeom = zpitch_InLo + zpitch_OutLo;

        zLo = z_InLo + (z_InLo - SDL::deltaZLum) * (rtRatio_OutLoInLo - 1.f) * (z_InLo > 0.f ? 1.f : dzDrtScale) - zGeom;

        // Cut #0: Preliminary (Only here in endcap case)
        pass = pass and (z_InLo * z_OutLo > 0);
        if(not pass) return pass;

        float dLum = SDL::copysignf(SDL::deltaZLum, z_InLo);
        bool isOutSgInnerMDPS = modulesInGPU.moduleType[outerInnerLowerModuleIndex] == SDL::PS;
        float rtGeom1 = isOutSgInnerMDPS ? SDL::pixelPSZpitch : SDL::strip2SZpitch;
        float zGeom1 = SDL::copysignf(zGeom,z_InLo);
        rtLo = rt_InLo * (1.f + (z_OutLo - z_InLo - zGeom1) / (z_InLo + zGeom1 + dLum) / dzDrtScale) - rtGeom1; //slope correction only on the lower end
        zOut = z_OutLo;
        rtOut = rt_OutLo;

        //Cut #1: rt condition
        pass = pass and (rtOut >= rtLo);
        if(not pass) return pass;

        float zInForHi = z_InLo - zGeom1 - dLum;
        if(zInForHi * z_InLo < 0)
        {
            zInForHi = SDL::copysignf(0.1f,z_InLo);
        }
        rtHi = rt_InLo * (1.f + (z_OutLo - z_InLo + zGeom1) / zInForHi) + rtGeom1;

        //Cut #2: rt condition
        pass = pass and ((rt_OutLo >= rtLo) & (rt_OutLo <= rtHi));
        if(not pass) return pass;

        float rIn = alpaka::math::sqrt(acc, z_InLo * z_InLo + rt_InLo * rt_InLo);
        const float drtSDIn = rt_InOut - rt_InLo;
        const float dzSDIn = z_InOut - z_InLo;
        const float dr3SDIn = alpaka::math::sqrt(acc, rt_InOut * rt_InOut + z_InOut * z_InOut) - alpaka::math::sqrt(acc, rt_InLo * rt_InLo + z_InLo * z_InLo);

        const float coshEta = dr3SDIn / drtSDIn; //direction estimate
        const float dzOutInAbs = alpaka::math::abs(acc, z_OutLo - z_InLo);
        const float multDzDr = dzOutInAbs * coshEta / (coshEta * coshEta - 1.f);
        const float zGeom1_another = SDL::pixelPSZpitch;
        kZ = (z_OutLo - z_InLo) / dzSDIn;
        float drtErr = zGeom1_another * zGeom1_another * drtSDIn * drtSDIn / dzSDIn / dzSDIn * (1.f - 2.f * kZ + 2.f * kZ * kZ);
        const float sdlThetaMulsF = 0.015f * alpaka::math::sqrt(acc, 0.1f + 0.2f * (rt_OutLo - rt_InLo) / 50.f) * alpaka::math::sqrt(acc, rIn / rt_InLo);
        const float sdlMuls = sdlThetaMulsF * 3.f / SDL::ptCut * 4.f; //will need a better guess than x4?
        drtErr += sdlMuls * sdlMuls * multDzDr * multDzDr / 3.f * coshEta * coshEta; //sloppy: relative muls is 1/3 of total muls
        drtErr = alpaka::math::sqrt(acc, drtErr);
        const float drtMean = drtSDIn * dzOutInAbs / alpaka::math::abs(acc, dzSDIn); //
        const float rtWindow = drtErr + rtGeom1;
        const float rtLo_another = rt_InLo + drtMean / dzDrtScale - rtWindow;
        const float rtHi_another = rt_InLo + drtMean + rtWindow;

        //Cut #3: rt-z pointed
        pass = pass and ((kZ >= 0) & (rtOut >= rtLo) & (rtOut <= rtHi));
        if(not pass) return pass;

        const float sdlPVoff = 0.1f / rt_OutLo;
        sdlCut = alpha1GeV_OutLo + alpaka::math::sqrt(acc, sdlMuls * sdlMuls + sdlPVoff*sdlPVoff);

        deltaPhiPos = SDL::deltaPhi(acc, mdsInGPU.anchorX[secondMDIndex], mdsInGPU.anchorY[secondMDIndex], mdsInGPU.anchorX[fourthMDIndex], mdsInGPU.anchorY[fourthMDIndex]);

        //Cut #4: deltaPhiPos can be tighter
        pass = pass and (alpaka::math::abs(acc, deltaPhiPos) <= sdlCut);
        if(not pass) return pass;

        float midPointX = 0.5f*(mdsInGPU.anchorX[firstMDIndex] + mdsInGPU.anchorX[thirdMDIndex]);
        float midPointY = 0.5f* (mdsInGPU.anchorY[firstMDIndex] + mdsInGPU.anchorY[thirdMDIndex]);
        float midPointZ = 0.5f*(mdsInGPU.anchorZ[firstMDIndex] + mdsInGPU.anchorZ[thirdMDIndex]);
        float diffX = mdsInGPU.anchorX[thirdMDIndex] - mdsInGPU.anchorX[firstMDIndex];
        float diffY = mdsInGPU.anchorY[thirdMDIndex] - mdsInGPU.anchorY[firstMDIndex];
        float diffZ = mdsInGPU.anchorZ[thirdMDIndex] - mdsInGPU.anchorZ[firstMDIndex];

        dPhi = SDL::deltaPhi(acc, midPointX, midPointY, diffX, diffY);
        // Cut #5: deltaPhiChange
        pass = pass and (alpaka::math::abs(acc, dPhi) <= sdlCut);
        if(not pass) return pass;

        float sdIn_alpha     = __H2F(segmentsInGPU.dPhiChanges[innerSegmentIndex]);
        float sdIn_alpha_min = __H2F(segmentsInGPU.dPhiChangeMins[innerSegmentIndex]);
        float sdIn_alpha_max = __H2F(segmentsInGPU.dPhiChangeMaxs[innerSegmentIndex]);
        float sdOut_alpha = sdIn_alpha; //weird

        float sdOut_alphaOut = SDL::deltaPhi(acc, mdsInGPU.anchorX[fourthMDIndex], mdsInGPU.anchorY[fourthMDIndex], mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex], mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]);

        float sdOut_alphaOut_min = SDL::phi_mpi_pi(acc, __H2F(segmentsInGPU.dPhiChangeMins[outerSegmentIndex]) - __H2F(segmentsInGPU.dPhiMins[outerSegmentIndex]));
        float sdOut_alphaOut_max = SDL::phi_mpi_pi(acc, __H2F(segmentsInGPU.dPhiChangeMaxs[outerSegmentIndex]) - __H2F(segmentsInGPU.dPhiMaxs[outerSegmentIndex]));

        float tl_axis_x = mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[firstMDIndex];
        float tl_axis_y = mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[firstMDIndex];
        float tl_axis_z = mdsInGPU.anchorZ[fourthMDIndex] - mdsInGPU.anchorZ[firstMDIndex];

        betaIn = sdIn_alpha - SDL::deltaPhi(acc, mdsInGPU.anchorX[firstMDIndex], mdsInGPU.anchorY[firstMDIndex], tl_axis_x, tl_axis_y);

        float betaInRHmin = betaIn;
        float betaInRHmax = betaIn;
        betaOut = -sdOut_alphaOut + SDL::deltaPhi(acc, mdsInGPU.anchorX[fourthMDIndex], mdsInGPU.anchorY[fourthMDIndex], tl_axis_x, tl_axis_y);

        float betaOutRHmin = betaOut;
        float betaOutRHmax = betaOut;

        bool isEC_secondLayer = (modulesInGPU.subdets[innerOuterLowerModuleIndex] == SDL::Endcap) and (modulesInGPU.moduleType[innerOuterLowerModuleIndex] == SDL::TwoS);

        if(isEC_secondLayer)
        {
            betaInRHmin = betaIn - sdIn_alpha_min + sdIn_alpha;
            betaInRHmax = betaIn - sdIn_alpha_max + sdIn_alpha;
        }

        betaOutRHmin = betaOut - sdOut_alphaOut_min + sdOut_alphaOut;
        betaOutRHmax = betaOut - sdOut_alphaOut_max + sdOut_alphaOut;

        float swapTemp;
        if(alpaka::math::abs(acc, betaOutRHmin) > alpaka::math::abs(acc, betaOutRHmax))
        {
            swapTemp = betaOutRHmin;
            betaOutRHmin = betaOutRHmax;
            betaOutRHmax = swapTemp;
        }

        if(alpaka::math::abs(acc, betaInRHmin) > alpaka::math::abs(acc, betaInRHmax))
        {
            swapTemp = betaInRHmin;
            betaInRHmin = betaInRHmax;
            betaInRHmax = swapTemp;
        }

        float sdIn_dr = alpaka::math::sqrt(acc, (mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) * (mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) + (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]) * (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]));
        float sdIn_d = rt_InOut - rt_InLo;

        float dr = alpaka::math::sqrt(acc, tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);
        const float corrF = 1.f;
        betaInCut = alpaka::math::asin(acc, alpaka::math::min(acc, (-sdIn_dr * corrF + dr) * SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax)) + (0.02f / sdIn_d);

        //Cut #6: first beta cut
        pass = pass and (alpaka::math::abs(acc, betaInRHmin) < betaInCut);
        if(not pass) return pass;

        float betaAv = 0.5f * (betaIn + betaOut);
        pt_beta = dr * SDL::k2Rinv1GeVf /alpaka::math::sin(acc, betaAv);

        float lIn = 5;
        float lOut = 11;

        float sdOut_dr = alpaka::math::sqrt(acc, (mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex]) * (mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex]) + (mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]) * (mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]));
        float sdOut_d = mdsInGPU.anchorRt[fourthMDIndex] - mdsInGPU.anchorRt[thirdMDIndex];

        SDL::runDeltaBetaIterationsT5(acc, betaIn, betaOut, betaAv, pt_beta, sdIn_dr, sdOut_dr, dr, lIn);

        const float betaInMMSF = (alpaka::math::abs(acc, betaInRHmin + betaInRHmax) > 0) ? (2.f * betaIn / alpaka::math::abs(acc, betaInRHmin + betaInRHmax)) : 0.; //mean value of min,max is the old betaIn
        const float betaOutMMSF = (alpaka::math::abs(acc, betaOutRHmin + betaOutRHmax) > 0) ? (2.f * betaOut / alpaka::math::abs(acc, betaOutRHmin + betaOutRHmax)) : 0.;
        betaInRHmin *= betaInMMSF;
        betaInRHmax *= betaInMMSF;
        betaOutRHmin *= betaOutMMSF;
        betaOutRHmax *= betaOutMMSF;

        const float dBetaMuls = sdlThetaMulsF * 4.f / alpaka::math::min(acc, alpaka::math::abs(acc, pt_beta), SDL::pt_betaMax); //need to confirm the range-out value of 7 GeV

        const float alphaInAbsReg = alpaka::math::max(acc, alpaka::math::abs(acc, sdIn_alpha), alpaka::math::asin(acc, alpaka::math::min(acc, rt_InLo * SDL::k2Rinv1GeVf / 3.0f, SDL::sinAlphaMax)));
        const float alphaOutAbsReg = alpaka::math::max(acc, alpaka::math::abs(acc, sdOut_alpha), alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * SDL::k2Rinv1GeVf / 3.0f, SDL::sinAlphaMax)));
        const float dBetaInLum = lIn < 11 ? 0.0f : alpaka::math::abs(acc, alphaInAbsReg*SDL::deltaZLum / z_InLo);
        const float dBetaOutLum = lOut < 11 ? 0.0f : alpaka::math::abs(acc, alphaOutAbsReg*SDL::deltaZLum / z_OutLo);
        const float dBetaLum2 = (dBetaInLum + dBetaOutLum) * (dBetaInLum + dBetaOutLum);
        const float sinDPhi =alpaka::math::sin(acc, dPhi);

        const float dBetaRIn2 = 0; // TODO-RH
        float dBetaROut = 0;
        if(modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::TwoS)
        {
            dBetaROut = (alpaka::math::sqrt(acc, mdsInGPU.anchorHighEdgeX[fourthMDIndex] * mdsInGPU.anchorHighEdgeX[fourthMDIndex] + mdsInGPU.anchorHighEdgeY[fourthMDIndex] * mdsInGPU.anchorHighEdgeY[fourthMDIndex]) - alpaka::math::sqrt(acc, mdsInGPU.anchorLowEdgeX[fourthMDIndex] * mdsInGPU.anchorLowEdgeX[fourthMDIndex] + mdsInGPU.anchorLowEdgeY[fourthMDIndex] * mdsInGPU.anchorLowEdgeY[fourthMDIndex])) * sinDPhi / dr;
        }

        const float dBetaROut2 = dBetaROut * dBetaROut;
        //FIXME: need faster version
        betaOutCut = alpaka::math::asin(acc, alpaka::math::min(acc, dr*SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax)) + (0.02f / sdOut_d) + alpaka::math::sqrt(acc, dBetaLum2 + dBetaMuls*dBetaMuls);

        //Cut #6: The real beta cut
        pass = pass and (alpaka::math::abs(acc, betaOut) < betaOutCut);
        if(not pass) return pass;

        float pt_betaIn = dr * SDL::k2Rinv1GeVf/alpaka::math::sin(acc, betaIn);
        float pt_betaOut = dr * SDL::k2Rinv1GeVf /alpaka::math::sin(acc, betaOut);
        float dBetaRes = 0.02f/alpaka::math::min(acc, sdOut_d,sdIn_d);
        float dBetaCut2 = (dBetaRes*dBetaRes * 2.0f + dBetaMuls * dBetaMuls + dBetaLum2 + dBetaRIn2 + dBetaROut2 + 0.25f * (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)) * (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)));
        float dBeta = betaIn - betaOut;
        deltaBetaCut = alpaka::math::sqrt(acc, dBetaCut2);
        //Cut #7: Cut on dBet
        pass = pass and (dBeta * dBeta <= dBetaCut2);

        return pass;
    };

    template<typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runQuintupletDefaultAlgoEEEE(TAcc const & acc, struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& innerOuterLowerModuleIndex, uint16_t& outerInnerLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex, unsigned int& fourthMDIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& dPhi, float& betaIn, float&betaOut, float& pt_beta, float& zLo, float& rtLo, float& rtHi, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ)
    {
        bool pass = true;

        bool isPS_InLo = (modulesInGPU.moduleType[innerInnerLowerModuleIndex] == SDL::PS);
        bool isPS_OutLo = (modulesInGPU.moduleType[outerInnerLowerModuleIndex] == SDL::PS);

        float rt_InLo = mdsInGPU.anchorRt[firstMDIndex];
        float rt_InOut = mdsInGPU.anchorRt[secondMDIndex];
        float rt_OutLo = mdsInGPU.anchorRt[thirdMDIndex];

        float z_InLo = mdsInGPU.anchorZ[firstMDIndex];
        float z_InOut = mdsInGPU.anchorZ[secondMDIndex];
        float z_OutLo = mdsInGPU.anchorZ[thirdMDIndex];

        float alpha1GeV_OutLo = alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax));

        float rtRatio_OutLoInLo = rt_OutLo / rt_InLo; // Outer segment beginning rt divided by inner segment beginning rt;
        float dzDrtScale = alpaka::math::tan(acc, alpha1GeV_OutLo) / alpha1GeV_OutLo; // The track can bend in r-z plane slightly
        float zpitch_InLo = (isPS_InLo ? SDL::pixelPSZpitch : SDL::strip2SZpitch);
        float zpitch_OutLo = (isPS_OutLo ? SDL::pixelPSZpitch : SDL::strip2SZpitch);
        float zGeom = zpitch_InLo + zpitch_OutLo;

        zLo = z_InLo + (z_InLo - SDL::deltaZLum) * (rtRatio_OutLoInLo - 1.f) * (z_InLo > 0.f ? 1.f : dzDrtScale) - zGeom; //slope-correction only on outer end

        // Cut #0: Preliminary (Only here in endcap case)
        pass = pass and ((z_InLo * z_OutLo) > 0);
        if(not pass) return pass;

        float dLum = SDL::copysignf(SDL::deltaZLum, z_InLo);
        bool isOutSgInnerMDPS = modulesInGPU.moduleType[outerInnerLowerModuleIndex] == SDL::PS;
        bool isInSgInnerMDPS = modulesInGPU.moduleType[innerInnerLowerModuleIndex] == SDL::PS;

        float rtGeom = (isInSgInnerMDPS and isOutSgInnerMDPS) ? 2.f * SDL::pixelPSZpitch : (isInSgInnerMDPS or isOutSgInnerMDPS) ? SDL::pixelPSZpitch + SDL::strip2SZpitch : 2.f * SDL::strip2SZpitch;

        float zGeom1 = SDL::copysignf(zGeom,z_InLo);
        float dz = z_OutLo - z_InLo;
        rtLo = rt_InLo * (1.f + dz / (z_InLo + dLum) / dzDrtScale) - rtGeom; //slope correction only on the lower end

        zOut = z_OutLo;
        rtOut = rt_OutLo;

        //Cut #1: rt condition

        rtHi = rt_InLo * (1.f + dz / (z_InLo - dLum)) + rtGeom;

        pass = pass and ((rtOut >= rtLo) & (rtOut <= rtHi));
        if(not pass) return pass;

        bool isInSgOuterMDPS = modulesInGPU.moduleType[innerOuterLowerModuleIndex] == SDL::PS;

        float drOutIn = rtOut - rt_InLo;
        const float drtSDIn = rt_InOut - rt_InLo;
        const float dzSDIn = z_InOut - z_InLo;
        const float dr3SDIn = alpaka::math::sqrt(acc, rt_InOut * rt_InOut + z_InOut * z_InOut) - alpaka::math::sqrt(acc, rt_InLo * rt_InLo + z_InLo * z_InLo);
        float coshEta = dr3SDIn / drtSDIn; //direction estimate
        float dzOutInAbs =  alpaka::math::abs(acc, z_OutLo - z_InLo);
        float multDzDr = dzOutInAbs * coshEta / (coshEta * coshEta - 1.f);

        kZ = (z_OutLo - z_InLo) / dzSDIn;
        float sdlThetaMulsF = 0.015f * alpaka::math::sqrt(acc, 0.1f + 0.2f * (rt_OutLo - rt_InLo) / 50.f);

        float sdlMuls = sdlThetaMulsF * 3.f / SDL::ptCut * 4.f; //will need a better guess than x4?

        float drtErr = alpaka::math::sqrt(acc, SDL::pixelPSZpitch * SDL::pixelPSZpitch * 2.f / (dzSDIn * dzSDIn) * (dzOutInAbs * dzOutInAbs) + sdlMuls * sdlMuls * multDzDr * multDzDr / 3.f * coshEta * coshEta);

        float drtMean = drtSDIn * dzOutInAbs/alpaka::math::abs(acc, dzSDIn);
        float rtWindow = drtErr + rtGeom;
        float rtLo_point = rt_InLo + drtMean / dzDrtScale - rtWindow;
        float rtHi_point = rt_InLo + drtMean + rtWindow;

        // Cut #3: rt-z pointed
        // https://github.com/slava77/cms-tkph2-ntuple/blob/superDoubletLinked-91X-noMock/doubletAnalysis.C#L3765

        if (isInSgInnerMDPS and isInSgOuterMDPS) // If both PS then we can point
        {
            pass = pass and (kZ >= 0 and rtOut >= rtLo_point and rtOut <= rtHi_point);
            if(not pass) return pass;
        }

        float sdlPVoff = 0.1f/rtOut;
        sdlCut = alpha1GeV_OutLo + alpaka::math::sqrt(acc, sdlMuls * sdlMuls + sdlPVoff * sdlPVoff);

        deltaPhiPos = SDL::deltaPhi(acc, mdsInGPU.anchorX[secondMDIndex], mdsInGPU.anchorY[secondMDIndex], mdsInGPU.anchorX[fourthMDIndex], mdsInGPU.anchorY[fourthMDIndex]);

        pass = pass and (alpaka::math::abs(acc, deltaPhiPos) <= sdlCut);
        if(not pass) return pass;

        float midPointX = 0.5f*(mdsInGPU.anchorX[firstMDIndex] + mdsInGPU.anchorX[thirdMDIndex]);
        float midPointY = 0.5f* (mdsInGPU.anchorY[firstMDIndex] + mdsInGPU.anchorY[thirdMDIndex]);
        float midPointZ = 0.5f*(mdsInGPU.anchorZ[firstMDIndex] + mdsInGPU.anchorZ[thirdMDIndex]);
        float diffX = mdsInGPU.anchorX[thirdMDIndex] - mdsInGPU.anchorX[firstMDIndex];
        float diffY = mdsInGPU.anchorY[thirdMDIndex] - mdsInGPU.anchorY[firstMDIndex];
        float diffZ = mdsInGPU.anchorZ[thirdMDIndex] - mdsInGPU.anchorZ[firstMDIndex];

        dPhi = SDL::deltaPhi(acc, midPointX, midPointY, diffX, diffY);

        // Cut #5: deltaPhiChange
        pass = pass and ((alpaka::math::abs(acc, dPhi) <= sdlCut));
        if(not pass) return pass;

        float sdIn_alpha = __H2F(segmentsInGPU.dPhiChanges[innerSegmentIndex]);
        float sdOut_alpha = sdIn_alpha; //weird
        float sdOut_dPhiPos = SDL::deltaPhi(acc, mdsInGPU.anchorX[thirdMDIndex], mdsInGPU.anchorY[thirdMDIndex], mdsInGPU.anchorX[fourthMDIndex], mdsInGPU.anchorY[fourthMDIndex]);

        float sdOut_dPhiChange = __H2F(segmentsInGPU.dPhiChanges[outerSegmentIndex]);
        float sdOut_dPhiChange_min = __H2F(segmentsInGPU.dPhiChangeMins[outerSegmentIndex]);
        float sdOut_dPhiChange_max = __H2F(segmentsInGPU.dPhiChangeMaxs[outerSegmentIndex]);

        float sdOut_alphaOutRHmin = SDL::phi_mpi_pi(acc, sdOut_dPhiChange_min - sdOut_dPhiPos);
        float sdOut_alphaOutRHmax = SDL::phi_mpi_pi(acc, sdOut_dPhiChange_max - sdOut_dPhiPos);
        float sdOut_alphaOut = SDL::phi_mpi_pi(acc, sdOut_dPhiChange - sdOut_dPhiPos);

        float tl_axis_x = mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[firstMDIndex];
        float tl_axis_y = mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[firstMDIndex];
        float tl_axis_z = mdsInGPU.anchorZ[fourthMDIndex] - mdsInGPU.anchorZ[firstMDIndex];

        betaIn = sdIn_alpha - SDL::deltaPhi(acc, mdsInGPU.anchorX[firstMDIndex], mdsInGPU.anchorY[firstMDIndex], tl_axis_x, tl_axis_y);

        float sdIn_alphaRHmin = __H2F(segmentsInGPU.dPhiChangeMins[innerSegmentIndex]);
        float sdIn_alphaRHmax = __H2F(segmentsInGPU.dPhiChangeMaxs[innerSegmentIndex]);
        float betaInRHmin = betaIn + sdIn_alphaRHmin - sdIn_alpha;
        float betaInRHmax = betaIn + sdIn_alphaRHmax - sdIn_alpha;

        betaOut = -sdOut_alphaOut + SDL::deltaPhi(acc, mdsInGPU.anchorX[fourthMDIndex], mdsInGPU.anchorY[fourthMDIndex], tl_axis_x, tl_axis_y);

        float betaOutRHmin = betaOut - sdOut_alphaOutRHmin + sdOut_alphaOut;
        float betaOutRHmax = betaOut - sdOut_alphaOutRHmax + sdOut_alphaOut;

        float swapTemp;
        if(alpaka::math::abs(acc, betaOutRHmin) > alpaka::math::abs(acc, betaOutRHmax))
        {
            swapTemp = betaOutRHmin;
            betaOutRHmin = betaOutRHmax;
            betaOutRHmax = swapTemp;
        }

        if(alpaka::math::abs(acc, betaInRHmin) > alpaka::math::abs(acc, betaInRHmax))
        {
            swapTemp = betaInRHmin;
            betaInRHmin = betaInRHmax;
            betaInRHmax = swapTemp;
        }
        float sdIn_dr = alpaka::math::sqrt(acc, (mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) * (mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) + (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]) * (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]));
        float sdIn_d = rt_InOut - rt_InLo;

        float dr = alpaka::math::sqrt(acc, tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);
        const float corrF = 1.f;
        betaInCut = alpaka::math::asin(acc, alpaka::math::min(acc, (-sdIn_dr * corrF + dr) * SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax)) + (0.02f / sdIn_d);

        //Cut #6: first beta cut
        pass = pass and (alpaka::math::abs(acc, betaInRHmin) < betaInCut);
        if(not pass) return pass;

        float betaAv = 0.5f * (betaIn + betaOut);
        pt_beta = dr * SDL::k2Rinv1GeVf /alpaka::math::sin(acc, betaAv);

        int lIn= 11; //endcap
        int lOut = 13; //endcap

        float sdOut_dr = alpaka::math::sqrt(acc, (mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex]) * (mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex]) + (mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]) * (mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]));
        float sdOut_d = mdsInGPU.anchorRt[fourthMDIndex] - mdsInGPU.anchorRt[thirdMDIndex];

        float diffDr = alpaka::math::abs(acc, sdIn_dr - sdOut_dr)/alpaka::math::abs(acc, sdIn_dr + sdOut_dr);

        SDL::runDeltaBetaIterationsT5(acc, betaIn, betaOut, betaAv, pt_beta, sdIn_dr, sdOut_dr, dr, lIn);

        const float betaInMMSF = (alpaka::math::abs(acc, betaInRHmin + betaInRHmax) > 0) ? (2.f * betaIn / alpaka::math::abs(acc, betaInRHmin + betaInRHmax)) : 0.; //mean value of min,max is the old betaIn
        const float betaOutMMSF = (alpaka::math::abs(acc, betaOutRHmin + betaOutRHmax) > 0) ? (2.f * betaOut / alpaka::math::abs(acc, betaOutRHmin + betaOutRHmax)) : 0.;
        betaInRHmin *= betaInMMSF;
        betaInRHmax *= betaInMMSF;
        betaOutRHmin *= betaOutMMSF;
        betaOutRHmax *= betaOutMMSF;

        const float dBetaMuls = sdlThetaMulsF * 4.f / alpaka::math::min(acc, alpaka::math::abs(acc, pt_beta), SDL::pt_betaMax); //need to confirm the range-out value of 7 GeV

        const float alphaInAbsReg = alpaka::math::max(acc, alpaka::math::abs(acc, sdIn_alpha), alpaka::math::asin(acc, alpaka::math::min(acc, rt_InLo * SDL::k2Rinv1GeVf / 3.0f, SDL::sinAlphaMax)));
        const float alphaOutAbsReg = alpaka::math::max(acc, alpaka::math::abs(acc, sdOut_alpha), alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * SDL::k2Rinv1GeVf / 3.0f, SDL::sinAlphaMax)));
        const float dBetaInLum = lIn < 11 ? 0.0f : alpaka::math::abs(acc, alphaInAbsReg*SDL::deltaZLum / z_InLo);
        const float dBetaOutLum = lOut < 11 ? 0.0f : alpaka::math::abs(acc, alphaOutAbsReg*SDL::deltaZLum / z_OutLo);
        const float dBetaLum2 = (dBetaInLum + dBetaOutLum) * (dBetaInLum + dBetaOutLum);
        const float sinDPhi =alpaka::math::sin(acc, dPhi);

        const float dBetaRIn2 = 0; // TODO-RH

        float dBetaROut2 = 0;//TODO-RH
        //FIXME: need faster version
        betaOutCut = alpaka::math::asin(acc, alpaka::math::min(acc, dr*SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax)) + (0.02f / sdOut_d) + alpaka::math::sqrt(acc, dBetaLum2 + dBetaMuls*dBetaMuls);

        //Cut #6: The real beta cut
        pass = pass and (alpaka::math::abs(acc, betaOut) < betaOutCut);
        if(not pass) return pass;

        float pt_betaIn = dr * SDL::k2Rinv1GeVf/alpaka::math::sin(acc, betaIn);
        float pt_betaOut = dr * SDL::k2Rinv1GeVf /alpaka::math::sin(acc, betaOut);
        float dBetaRes = 0.02f/alpaka::math::min(acc, sdOut_d,sdIn_d);
        float dBetaCut2 = (dBetaRes*dBetaRes * 2.0f + dBetaMuls * dBetaMuls + dBetaLum2 + dBetaRIn2 + dBetaROut2 + 0.25f * (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)) * (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)));
        float dBeta = betaIn - betaOut;
        //Cut #7: Cut on dBeta
        deltaBetaCut = alpaka::math::sqrt(acc, dBetaCut2);

        pass = pass and (dBeta * dBeta <= dBetaCut2);

        return pass;
    };

    template<typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runQuintupletAlgoSelector(TAcc const & acc, struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& innerOuterLowerModuleIndex, uint16_t& outerInnerLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex, unsigned int& fourthMDIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& deltaPhi, float& betaIn, float&betaOut, float& pt_beta, float& zLo, float& zHi, float& rtLo, float& rtHi, float& zLoPointed, float& zHiPointed, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ)
    {
        bool pass = false;

        zLo = -999;
        zHi = -999;
        rtLo = -999;
        rtHi = -999;
        zLoPointed = -999;
        zHiPointed = -999;
        kZ = -999;
        betaInCut = -999;

        short innerInnerLowerModuleSubdet = modulesInGPU.subdets[innerInnerLowerModuleIndex];
        short innerOuterLowerModuleSubdet = modulesInGPU.subdets[innerOuterLowerModuleIndex];
        short outerInnerLowerModuleSubdet = modulesInGPU.subdets[outerInnerLowerModuleIndex];
        short outerOuterLowerModuleSubdet = modulesInGPU.subdets[outerOuterLowerModuleIndex];

        if(innerInnerLowerModuleSubdet == SDL::Barrel
                and innerOuterLowerModuleSubdet == SDL::Barrel
                and outerInnerLowerModuleSubdet == SDL::Barrel
                and outerOuterLowerModuleSubdet == SDL::Barrel)
        {
            return runQuintupletDefaultAlgoBBBB(acc, modulesInGPU,mdsInGPU,segmentsInGPU,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,innerSegmentIndex,outerSegmentIndex,firstMDIndex, secondMDIndex, thirdMDIndex, fourthMDIndex, zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta, zLo, zHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut);
        }
        else if(innerInnerLowerModuleSubdet == SDL::Barrel
                and innerOuterLowerModuleSubdet == SDL::Barrel
                and outerInnerLowerModuleSubdet == SDL::Endcap
                and outerOuterLowerModuleSubdet == SDL::Endcap)
        {
        return runQuintupletDefaultAlgoBBEE(acc, modulesInGPU,mdsInGPU,segmentsInGPU,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,innerSegmentIndex,outerSegmentIndex,firstMDIndex, secondMDIndex, thirdMDIndex, fourthMDIndex, zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta, zLo, rtLo, rtHi, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ);
        }
        else if(innerInnerLowerModuleSubdet == SDL::Barrel
                and innerOuterLowerModuleSubdet == SDL::Barrel
                and outerInnerLowerModuleSubdet == SDL::Barrel
                and outerOuterLowerModuleSubdet == SDL::Endcap)
        {
            return runQuintupletDefaultAlgoBBBB(acc, modulesInGPU,mdsInGPU,segmentsInGPU,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,innerSegmentIndex,outerSegmentIndex,firstMDIndex, secondMDIndex, thirdMDIndex, fourthMDIndex, zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta,zLo, zHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut);
        }
        else if(innerInnerLowerModuleSubdet == SDL::Barrel
                and innerOuterLowerModuleSubdet == SDL::Endcap
                and outerInnerLowerModuleSubdet == SDL::Endcap
                and outerOuterLowerModuleSubdet == SDL::Endcap)
        {
            return runQuintupletDefaultAlgoBBEE(acc, modulesInGPU,mdsInGPU,segmentsInGPU,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,innerSegmentIndex,outerSegmentIndex, firstMDIndex, secondMDIndex, thirdMDIndex, fourthMDIndex, zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta, zLo, rtLo, rtHi, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ);
        }
        else if(innerInnerLowerModuleSubdet == SDL::Endcap
                and innerOuterLowerModuleSubdet == SDL::Endcap
                and outerInnerLowerModuleSubdet == SDL::Endcap
                and outerOuterLowerModuleSubdet == SDL::Endcap)
        {
            return runQuintupletDefaultAlgoEEEE(acc, modulesInGPU,mdsInGPU,segmentsInGPU,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,innerSegmentIndex,outerSegmentIndex, firstMDIndex, secondMDIndex, thirdMDIndex, fourthMDIndex, zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta, zLo, rtLo, rtHi, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ);
        }

        return pass;
    };

    template<typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runQuintupletDefaultAlgo(TAcc const & acc, struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, uint16_t& lowerModuleIndex1, uint16_t& lowerModuleIndex2, uint16_t& lowerModuleIndex3, uint16_t& lowerModuleIndex4, uint16_t& lowerModuleIndex5, unsigned int& innerTripletIndex, unsigned int& outerTripletIndex, float& innerRadius, float& outerRadius, float& bridgeRadius, float& regressionG, float& regressionF, float& regressionRadius, float& rzChiSquared, float& chiSquared, float& nonAnchorChiSquared)
    {
        bool pass = true;
        unsigned int firstSegmentIndex = tripletsInGPU.segmentIndices[2 * innerTripletIndex];
        unsigned int secondSegmentIndex = tripletsInGPU.segmentIndices[2 * innerTripletIndex + 1];
        unsigned int thirdSegmentIndex = tripletsInGPU.segmentIndices[2 * outerTripletIndex];
        unsigned int fourthSegmentIndex = tripletsInGPU.segmentIndices[2 * outerTripletIndex + 1];

        unsigned int innerOuterOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * secondSegmentIndex + 1]; //inner triplet outer segment outer MD index
        unsigned int outerInnerInnerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * thirdSegmentIndex]; //outer triplet inner segmnet inner MD index

        //this cut reduces the number of candidates by a factor of 3, i.e., 2 out of 3 warps can end right here!
        if (innerOuterOuterMiniDoubletIndex != outerInnerInnerMiniDoubletIndex) return false;
        
        //apply T4 criteria between segments 1 and 3
        float zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, pt_beta; //temp stuff
        float zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ;

        unsigned int firstMDIndex = segmentsInGPU.mdIndices[2 * firstSegmentIndex];
        unsigned int secondMDIndex = segmentsInGPU.mdIndices[2 * secondSegmentIndex];
        unsigned int thirdMDIndex = segmentsInGPU.mdIndices[2 * secondSegmentIndex + 1];
        unsigned int fourthMDIndex = segmentsInGPU.mdIndices[2 * thirdSegmentIndex + 1];
        unsigned int fifthMDIndex = segmentsInGPU.mdIndices[2 * fourthSegmentIndex + 1];

        pass = pass and runQuintupletAlgoSelector(acc, modulesInGPU, mdsInGPU, segmentsInGPU, lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4, firstSegmentIndex, thirdSegmentIndex, firstMDIndex, secondMDIndex, thirdMDIndex, fourthMDIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, pt_beta, zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ);
        if(not pass) return pass;

        pass = pass and runQuintupletAlgoSelector(acc, modulesInGPU, mdsInGPU, segmentsInGPU, lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex4, lowerModuleIndex5, firstSegmentIndex, fourthSegmentIndex, firstMDIndex, secondMDIndex, fourthMDIndex, fifthMDIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, pt_beta, zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ);
        if(not pass) return pass;

        pass = pass and passT5RZConstraint(acc, modulesInGPU, mdsInGPU, firstMDIndex, secondMDIndex, thirdMDIndex, fourthMDIndex, fifthMDIndex, lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4, lowerModuleIndex5);
        if(not pass) return pass;

        float x1 = mdsInGPU.anchorX[firstMDIndex];
        float x2 = mdsInGPU.anchorX[secondMDIndex];
        float x3 = mdsInGPU.anchorX[thirdMDIndex];
        float x4 = mdsInGPU.anchorX[fourthMDIndex];
        float x5 = mdsInGPU.anchorX[fifthMDIndex];
        
        float y1 = mdsInGPU.anchorY[firstMDIndex];
        float y2 = mdsInGPU.anchorY[secondMDIndex];
        float y3 = mdsInGPU.anchorY[thirdMDIndex];
        float y4 = mdsInGPU.anchorY[fourthMDIndex];
        float y5 = mdsInGPU.anchorY[fifthMDIndex];

        //construct the arrays
        float x1Vec[] = {x1, x1, x1};
        float y1Vec[] = {y1, y1, y1};
        float x2Vec[] = {x2, x2, x2};
        float y2Vec[] = {y2, y2, y2};
        float x3Vec[] = {x3, x3, x3};
        float y3Vec[] = {y3, y3, y3};

        if(modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex1] == SDL::TwoS)
        {
            x1Vec[1] = mdsInGPU.anchorLowEdgeX[firstMDIndex];
            x1Vec[2] = mdsInGPU.anchorHighEdgeX[firstMDIndex];

            y1Vec[1] = mdsInGPU.anchorLowEdgeY[firstMDIndex];
            y1Vec[2] = mdsInGPU.anchorHighEdgeY[firstMDIndex];
        }
        if(modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex2] == SDL::TwoS)
        {
            x2Vec[1] = mdsInGPU.anchorLowEdgeX[secondMDIndex];
            x2Vec[2] = mdsInGPU.anchorHighEdgeX[secondMDIndex];

            y2Vec[1] = mdsInGPU.anchorLowEdgeY[secondMDIndex];
            y2Vec[2] = mdsInGPU.anchorHighEdgeY[secondMDIndex];
        }
        if(modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex3] == SDL::TwoS)
        {
            x3Vec[1] = mdsInGPU.anchorLowEdgeX[thirdMDIndex];
            x3Vec[2] = mdsInGPU.anchorHighEdgeX[thirdMDIndex];

            y3Vec[1] = mdsInGPU.anchorLowEdgeY[thirdMDIndex];
            y3Vec[2] = mdsInGPU.anchorHighEdgeY[thirdMDIndex];
        }

        float innerRadiusMin2S, innerRadiusMax2S;
        computeErrorInRadius(acc, x1Vec, y1Vec, x2Vec, y2Vec, x3Vec, y3Vec, innerRadiusMin2S, innerRadiusMax2S);

        for (int i=0; i<3; i++) 
        {
            x1Vec[i] = x4;
            y1Vec[i] = y4;
        }
        if(modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex4] == SDL::TwoS)
        {
            x1Vec[1] = mdsInGPU.anchorLowEdgeX[fourthMDIndex];
            x1Vec[2] = mdsInGPU.anchorHighEdgeX[fourthMDIndex];

            y1Vec[1] = mdsInGPU.anchorLowEdgeY[fourthMDIndex];
            y1Vec[2] = mdsInGPU.anchorHighEdgeY[fourthMDIndex];
        }

        float bridgeRadiusMin2S, bridgeRadiusMax2S;
        computeErrorInRadius(acc, x2Vec, y2Vec, x3Vec, y3Vec, x1Vec, y1Vec, bridgeRadiusMin2S, bridgeRadiusMax2S);

        for(int i=0; i<3; i++) 
        {
            x2Vec[i] = x5;
            y2Vec[i] = y5;
        }
        if(modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex5] == SDL::TwoS)
        {
            x2Vec[1] = mdsInGPU.anchorLowEdgeX[fifthMDIndex];
            x2Vec[2] = mdsInGPU.anchorHighEdgeX[fifthMDIndex];

            y2Vec[1] = mdsInGPU.anchorLowEdgeY[fifthMDIndex];
            y2Vec[2] = mdsInGPU.anchorHighEdgeY[fifthMDIndex];
        }

        float outerRadiusMin2S, outerRadiusMax2S;
        computeErrorInRadius(acc, x3Vec, y3Vec, x1Vec, y1Vec, x2Vec, y2Vec, outerRadiusMin2S, outerRadiusMax2S);

        float g, f;
        innerRadius = computeRadiusFromThreeAnchorHits(acc, x1, y1, x2, y2, x3, y3, g, f);
        outerRadius = computeRadiusFromThreeAnchorHits(acc, x3, y3, x4, y4, x5, y5, g, f);
        bridgeRadius = computeRadiusFromThreeAnchorHits(acc, x2, y2, x3, y3, x4, y4, g, f);

        pass = pass & (innerRadius >= 0.95f * ptCut/(2.f * k2Rinv1GeVf));

        float innerInvRadiusMin, innerInvRadiusMax, bridgeInvRadiusMin, bridgeInvRadiusMax, outerInvRadiusMin, outerInvRadiusMax;

        //split by category
        bool tempPass;
        if(modulesInGPU.subdets[lowerModuleIndex1] == SDL::Barrel and modulesInGPU.subdets[lowerModuleIndex2] == SDL::Barrel and modulesInGPU.subdets[lowerModuleIndex3] == SDL::Barrel and modulesInGPU.subdets[lowerModuleIndex4] == SDL::Barrel and modulesInGPU.subdets[lowerModuleIndex5] == SDL::Barrel)
        {
            tempPass = matchRadiiBBBBB(acc, innerRadius, bridgeRadius, outerRadius, innerInvRadiusMin, innerInvRadiusMax, bridgeInvRadiusMin, bridgeInvRadiusMax, outerInvRadiusMin, outerInvRadiusMax);
        }
        else if(modulesInGPU.subdets[lowerModuleIndex1] == SDL::Barrel and modulesInGPU.subdets[lowerModuleIndex2] == SDL::Barrel and modulesInGPU.subdets[lowerModuleIndex3] == SDL::Barrel and modulesInGPU.subdets[lowerModuleIndex4] == SDL::Barrel and modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap)
        {
            tempPass = matchRadiiBBBBE(acc, innerRadius, bridgeRadius, outerRadius, innerRadiusMin2S, innerRadiusMax2S, bridgeRadiusMin2S, bridgeRadiusMax2S, outerRadiusMin2S, outerRadiusMax2S, innerInvRadiusMin, innerInvRadiusMax, bridgeInvRadiusMin, bridgeInvRadiusMax, outerInvRadiusMin, outerInvRadiusMax);
        }
        else if(modulesInGPU.subdets[lowerModuleIndex1] == SDL::Barrel and modulesInGPU.subdets[lowerModuleIndex2] == SDL::Barrel and modulesInGPU.subdets[lowerModuleIndex3] == SDL::Barrel and modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap and modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap)
        {
            if(modulesInGPU.layers[lowerModuleIndex1] == 1)
            {
                tempPass = matchRadiiBBBEE12378(acc, innerRadius, bridgeRadius, outerRadius,innerRadiusMin2S, innerRadiusMax2S, bridgeRadiusMin2S, bridgeRadiusMax2S, outerRadiusMin2S, outerRadiusMax2S, innerInvRadiusMin, innerInvRadiusMax, bridgeInvRadiusMin, bridgeInvRadiusMax, outerInvRadiusMin, outerInvRadiusMax);
            }
            else if(modulesInGPU.layers[lowerModuleIndex1] == 2)
            {
                tempPass = matchRadiiBBBEE23478(acc, innerRadius, bridgeRadius, outerRadius,innerRadiusMin2S, innerRadiusMax2S, bridgeRadiusMin2S, bridgeRadiusMax2S, outerRadiusMin2S, outerRadiusMax2S, innerInvRadiusMin, innerInvRadiusMax, bridgeInvRadiusMin, bridgeInvRadiusMax, outerInvRadiusMin, outerInvRadiusMax);
            }
            else
            {
                tempPass = matchRadiiBBBEE34578(acc, innerRadius, bridgeRadius, outerRadius,innerRadiusMin2S, innerRadiusMax2S, bridgeRadiusMin2S, bridgeRadiusMax2S, outerRadiusMin2S, outerRadiusMax2S, innerInvRadiusMin, innerInvRadiusMax, bridgeInvRadiusMin, bridgeInvRadiusMax, outerInvRadiusMin, outerInvRadiusMax);
            }
        }

        else if(modulesInGPU.subdets[lowerModuleIndex1] == SDL::Barrel and modulesInGPU.subdets[lowerModuleIndex2] == SDL::Barrel and modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap and modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap and modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap)
        {
            tempPass = matchRadiiBBEEE(acc, innerRadius, bridgeRadius, outerRadius, innerRadiusMin2S, innerRadiusMax2S, bridgeRadiusMin2S, bridgeRadiusMax2S, outerRadiusMin2S, outerRadiusMax2S, innerInvRadiusMin, innerInvRadiusMax, bridgeInvRadiusMin, bridgeInvRadiusMax, outerInvRadiusMin, outerInvRadiusMax);
        }
        else if(modulesInGPU.subdets[lowerModuleIndex1] == SDL::Barrel and modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap and modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap and modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap and modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap)
        {
            tempPass = matchRadiiBEEEE(acc, innerRadius, bridgeRadius, outerRadius, innerRadiusMin2S, innerRadiusMax2S, bridgeRadiusMin2S, bridgeRadiusMax2S, outerRadiusMin2S, outerRadiusMax2S, innerInvRadiusMin, innerInvRadiusMax, bridgeInvRadiusMin, bridgeInvRadiusMax, outerInvRadiusMin, outerInvRadiusMax);
        }
        else
        {
            tempPass = matchRadiiEEEEE(acc, innerRadius, bridgeRadius, outerRadius, innerRadiusMin2S, innerRadiusMax2S, bridgeRadiusMin2S, bridgeRadiusMax2S, outerRadiusMin2S, outerRadiusMax2S,innerInvRadiusMin, innerInvRadiusMax, bridgeInvRadiusMin, bridgeInvRadiusMax, outerInvRadiusMin, outerInvRadiusMax);
        }

        //compute regression radius right here - this computation is expensive!!!
        pass = pass and tempPass;
        if(not pass) return pass;

        float xVec[] = {x1, x2, x3, x4, x5};
        float yVec[] = {y1, y2, y3, y4, y5};
        float sigmas[5], delta1[5], delta2[5], slopes[5];
        bool isFlat[5];
        //5 categories for sigmas
        const uint16_t lowerModuleIndices[] = {lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4, lowerModuleIndex5};

        computeSigmasForRegression(acc, modulesInGPU, lowerModuleIndices, delta1, delta2, slopes, isFlat);
        regressionRadius = computeRadiusUsingRegression(acc, 5,xVec, yVec, delta1, delta2, slopes, isFlat, regressionG, regressionF, sigmas, chiSquared);

        //extra chi squared cuts!
        if(regressionRadius < 5.0f/(2.f * k2Rinv1GeVf))
        {
            pass = pass and passChiSquaredConstraint(modulesInGPU, lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4, lowerModuleIndex5, chiSquared);
            if(not pass) return pass;
        }

        //compute the other chisquared
        //non anchor is always shifted for tilted and endcap!
        float nonAnchorDelta1[5], nonAnchorDelta2[5], nonAnchorSlopes[5];
        float nonAnchorxs[] = { mdsInGPU.outerX[firstMDIndex], mdsInGPU.outerX[secondMDIndex], mdsInGPU.outerX[thirdMDIndex], mdsInGPU.outerX[fourthMDIndex], mdsInGPU.outerX[fifthMDIndex]};
        float nonAnchorys[] = { mdsInGPU.outerY[firstMDIndex], mdsInGPU.outerY[secondMDIndex], mdsInGPU.outerY[thirdMDIndex], mdsInGPU.outerY[fourthMDIndex], mdsInGPU.outerY[fifthMDIndex]};

        computeSigmasForRegression(acc, modulesInGPU, lowerModuleIndices, nonAnchorDelta1, nonAnchorDelta2, nonAnchorSlopes, isFlat, 5, false);
        nonAnchorChiSquared = computeChiSquared(acc, 5, nonAnchorxs, nonAnchorys, nonAnchorDelta1, nonAnchorDelta2, nonAnchorSlopes, isFlat, regressionG, regressionF, regressionRadius);
        return pass;
    };

    struct createQuintupletsInGPUv2
    {
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(
                TAcc const & acc,
                struct SDL::modules& modulesInGPU,
                struct SDL::miniDoublets& mdsInGPU,
                struct SDL::segments& segmentsInGPU,
                struct SDL::triplets& tripletsInGPU,
                struct SDL::quintuplets& quintupletsInGPU,
                struct SDL::objectRanges& rangesInGPU,
                uint16_t nEligibleT5Modules) const
        {
            using Dim = alpaka::Dim<TAcc>;
            using Idx = alpaka::Idx<TAcc>;
            using Vec = alpaka::Vec<Dim, Idx>;

            Vec const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
            Vec const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

            for (int iter = globalThreadIdx[0]; iter < nEligibleT5Modules; iter += gridThreadExtent[0])
            {
                uint16_t lowerModule1 = rangesInGPU.indicesOfEligibleT5Modules[iter];
                unsigned int nInnerTriplets = tripletsInGPU.nTriplets[lowerModule1];
                for( unsigned int innerTripletArrayIndex = globalThreadIdx[1]; innerTripletArrayIndex < nInnerTriplets; innerTripletArrayIndex += gridThreadExtent[1])
                {
                    unsigned int innerTripletIndex = rangesInGPU.tripletModuleIndices[lowerModule1] + innerTripletArrayIndex;
                    uint16_t lowerModule2 = tripletsInGPU.lowerModuleIndices[3 * innerTripletIndex + 1];
                    uint16_t lowerModule3 = tripletsInGPU.lowerModuleIndices[3 * innerTripletIndex + 2];
                    unsigned int nOuterTriplets = tripletsInGPU.nTriplets[lowerModule3];
                    for (int outerTripletArrayIndex = globalThreadIdx[2]; outerTripletArrayIndex < nOuterTriplets; outerTripletArrayIndex += gridThreadExtent[2])
                    {
                        unsigned int outerTripletIndex = rangesInGPU.tripletModuleIndices[lowerModule3] + outerTripletArrayIndex;
                        uint16_t lowerModule4 = tripletsInGPU.lowerModuleIndices[3 * outerTripletIndex + 1];
                        uint16_t lowerModule5 = tripletsInGPU.lowerModuleIndices[3 * outerTripletIndex + 2];

                        float innerRadius, outerRadius, bridgeRadius, regressionG, regressionF, regressionRadius, rzChiSquared, chiSquared, nonAnchorChiSquared; //required for making distributions

                        bool success = runQuintupletDefaultAlgo(acc, modulesInGPU, mdsInGPU, segmentsInGPU, tripletsInGPU, lowerModule1, lowerModule2, lowerModule3, lowerModule4, lowerModule5, innerTripletIndex, outerTripletIndex, innerRadius, outerRadius,  bridgeRadius, regressionG, regressionF, regressionRadius, rzChiSquared, chiSquared, nonAnchorChiSquared);

                        if(success)
                        {
                            short layer2_adjustment;
                            int layer = modulesInGPU.layers[lowerModule1];
                            if(layer == 1)
                            {
                                layer2_adjustment = 1;
                            } // get upper segment to be in second layer
                            else if(layer == 2)
                            {
                                layer2_adjustment = 0;
                            } // get lower segment to be in second layer
                            else
                            {
                                return;
                            } // ignore anything else TODO: move this to start, before object is made (faster)
                            int totOccupancyQuintuplets = alpaka::atomicOp<alpaka::AtomicAdd>(acc, &quintupletsInGPU.totOccupancyQuintuplets[lowerModule1], 1);
                            if(totOccupancyQuintuplets >= (rangesInGPU.quintupletModuleIndices[lowerModule1 + 1] - rangesInGPU.quintupletModuleIndices[lowerModule1]))
                            {
#ifdef Warnings
                                printf("Quintuplet excess alert! Module index = %d\n", lowerModule1);
#endif
                            }
                            else
                            {
                                int quintupletModuleIndex = alpaka::atomicOp<alpaka::AtomicAdd>(acc, &quintupletsInGPU.nQuintuplets[lowerModule1], 1);
                                //this if statement should never get executed!
                                if(rangesInGPU.quintupletModuleIndices[lowerModule1] == -1)
                                {
                                    printf("Quintuplets : no memory for module at module index = %d\n", lowerModule1);
                                }
                                else
                                {
                                    unsigned int quintupletIndex = rangesInGPU.quintupletModuleIndices[lowerModule1] +  quintupletModuleIndex;
                                    float phi = mdsInGPU.anchorPhi[segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*innerTripletIndex+layer2_adjustment]]];
                                    float eta = mdsInGPU.anchorEta[segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*innerTripletIndex+layer2_adjustment]]];
                                    float pt = (innerRadius+outerRadius)*3.8f*1.602f/(2*100*5.39f);
                                    float scores = chiSquared + nonAnchorChiSquared;
                                    addQuintupletToMemory(tripletsInGPU, quintupletsInGPU, innerTripletIndex, outerTripletIndex, lowerModule1, lowerModule2, lowerModule3, lowerModule4, lowerModule5, innerRadius, bridgeRadius, outerRadius, regressionG, regressionF, regressionRadius, rzChiSquared, chiSquared, nonAnchorChiSquared, pt,eta,phi,scores,layer,quintupletIndex);

                                    tripletsInGPU.partOfT5[quintupletsInGPU.tripletIndices[2 * quintupletIndex]] = true;
                                    tripletsInGPU.partOfT5[quintupletsInGPU.tripletIndices[2 * quintupletIndex + 1]] = true;
                                }
                            }
                        }
                    }
                }
            }
        }
    };

    struct createEligibleModulesListForQuintupletsGPU
    {
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(
                TAcc const & acc,
                struct modules& modulesInGPU,
                struct triplets& tripletsInGPU,
                struct objectRanges& rangesInGPU) const
        {
            using Dim = alpaka::Dim<TAcc>;
            using Idx = alpaka::Idx<TAcc>;
            using Vec = alpaka::Vec<Dim, Idx>;

            Vec const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
            Vec const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

            // Initialize variables in shared memory and set to 0
            int& nEligibleT5Modulesx = alpaka::declareSharedVar<int, __COUNTER__>(acc);
            int& nTotalQuintupletsx = alpaka::declareSharedVar<int, __COUNTER__>(acc);
            nTotalQuintupletsx = 0; nEligibleT5Modulesx = 0;
            alpaka::syncBlockThreads(acc);

            unsigned int category_number, eta_number;
            for(int i = globalThreadIdx[2]; i < *modulesInGPU.nLowerModules; i += gridThreadExtent[2])
            {
                // Condition for a quintuple to exist for a module
                // TCs don't exist for layers 5 and 6 barrel, and layers 2,3,4,5 endcap   
                unsigned int layers = modulesInGPU.layers[i];
                unsigned int subdets = modulesInGPU.subdets[i];
                unsigned int rings = modulesInGPU.rings[i];
                float eta = modulesInGPU.eta[i];
                float abs_eta = alpaka::math::abs(acc, eta);
                int occupancy = 0;

                if (tripletsInGPU.nTriplets[i] == 0) continue;
                if (subdets == SDL::Barrel and layers >= 3) continue;
                if (subdets == SDL::Endcap and layers > 1) continue;

                int nEligibleT5Modules = alpaka::atomicOp<alpaka::AtomicAdd>(acc, &nEligibleT5Modulesx, 1);

                if (layers<=3 && subdets==5) category_number = 0;
                else if (layers>=4 && subdets==5) category_number = 1;
                else if (layers<=2 && subdets==4 && rings>=11) category_number = 2;
                else if (layers>=3 && subdets==4 && rings>=8) category_number = 2;
                else if (layers<=2 && subdets==4 && rings<=10) category_number = 3;
                else if (layers>=3 && subdets==4 && rings<=7) category_number = 3;

                if (abs_eta<0.75) eta_number=0;
                else if (abs_eta>0.75 && abs_eta<1.5) eta_number=1;
                else if (abs_eta>1.5 && abs_eta<2.25) eta_number=2;
                else if (abs_eta>2.25 && abs_eta<3) eta_number=3;

                if (category_number == 0 && eta_number == 0) occupancy = 336;
                else if (category_number == 0 && eta_number == 1) occupancy = 414;
                else if (category_number == 0 && eta_number == 2) occupancy = 231;
                else if (category_number == 0 && eta_number == 3) occupancy = 146;
                else if (category_number == 3 && eta_number == 1) occupancy = 0;
                else if (category_number == 3 && eta_number == 2) occupancy = 191;
                else if (category_number == 3 && eta_number == 3) occupancy = 106;

                int nTotQ = alpaka::atomicOp<alpaka::AtomicAdd>(acc, &nTotalQuintupletsx, occupancy);
                rangesInGPU.quintupletModuleIndices[i] = nTotQ;
                rangesInGPU.indicesOfEligibleT5Modules[nEligibleT5Modules] = i;
            }

            // Wait for all threads to finish before reporting final values
            alpaka::syncBlockThreads(acc);
            if(globalThreadIdx[2] == 0)
            {
                *rangesInGPU.nEligibleT5Modules = static_cast<uint16_t>(nEligibleT5Modulesx);
                *rangesInGPU.device_nTotalQuints = static_cast<unsigned int>(nTotalQuintupletsx);
            }
        }
    };

    struct addQuintupletRangesToEventExplicit
    {
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(
                TAcc const & acc,
                struct modules& modulesInGPU,
                struct quintuplets& quintupletsInGPU,
                struct objectRanges& rangesInGPU) const
        {
            using Dim = alpaka::Dim<TAcc>;
            using Idx = alpaka::Idx<TAcc>;
            using Vec = alpaka::Vec<Dim, Idx>;

            Vec const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
            Vec const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

            for(uint16_t i = globalThreadIdx[2]; i < *modulesInGPU.nLowerModules; i += gridThreadExtent[2])
            {
                if(quintupletsInGPU.nQuintuplets[i] == 0 or rangesInGPU.quintupletModuleIndices[i] == -1)
                {
                    rangesInGPU.quintupletRanges[i * 2] = -1;
                    rangesInGPU.quintupletRanges[i * 2 + 1] = -1;
                }
                else
                {
                    rangesInGPU.quintupletRanges[i * 2] = rangesInGPU.quintupletModuleIndices[i];
                    rangesInGPU.quintupletRanges[i * 2 + 1] = rangesInGPU.quintupletModuleIndices[i] + quintupletsInGPU.nQuintuplets[i] - 1;
                }
            }
        }
    };
}
#endif
