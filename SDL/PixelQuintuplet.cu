#ifdef __CUDACC__
#define CUDA_CONST_VAR __device__
#endif
# include "PixelQuintuplet.cuh"
#include "allocate.h"

SDL::pixelQuintuplets::pixelQuintuplets()
{
    pixelIndices = nullptr;
    T5Indices = nullptr;
    nPixelQuintuplets = nullptr;
    isDup = nullptr;
    score = nullptr;
    logicalLayers = nullptr;
    hitIndices = nullptr;
    lowerModuleIndices = nullptr;
}

SDL::pixelQuintuplets::~pixelQuintuplets()
{
}

void SDL::pixelQuintuplets::freeMemory()
{
    cudaFree(pixelIndices);
    cudaFree(T5Indices);
    cudaFree(nPixelQuintuplets);
    cudaFree(isDup);
    cudaFree(score);
    cudaFree(logicalLayers);
    cudaFree(hitIndices);
    cudaFree(lowerModuleIndices);
    cudaFree(eta);
    cudaFree(phi);
#ifdef CUT_VALUE_DEBUG
    cudaFree(rzChiSquared);
    cudaFree(rPhiChiSquared);
    cudaFree(rPhiChiSquaredInwards);
#endif
}

void SDL::createPixelQuintupletsInUnifiedMemory(struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, unsigned int maxPixelQuintuplets)
{
    cudaMallocManaged(&pixelQuintupletsInGPU.pixelIndices, maxPixelQuintuplets * sizeof(unsigned int));
    cudaMallocManaged(&pixelQuintupletsInGPU.T5Indices, maxPixelQuintuplets * sizeof(unsigned int));
    cudaMallocManaged(&pixelQuintupletsInGPU.nPixelQuintuplets, sizeof(unsigned int));
    cudaMallocManaged(&pixelQuintupletsInGPU.isDup, maxPixelQuintuplets * sizeof(bool));
    cudaMallocManaged(&pixelQuintupletsInGPU.score, maxPixelQuintuplets * sizeof(float));
    cudaMallocManaged(&pixelQuintupletsInGPU.logicalLayers, maxPixelQuintuplets * 7 * sizeof(unsigned int));
    cudaMallocManaged(&pixelQuintupletsInGPU.lowerModuleIndices, maxPixelQuintuplets * 7 * sizeof(unsigned int));
    cudaMallocManaged(&pixelQuintupletsInGPU.hitIndices, maxPixelQuintuplets * 14 * sizeof(unsigned int));
    cudaMallocManaged(&pixelQuintupletsInGPU.eta, maxPixelQuintuplets * sizeof(float));
    cudaMallocManaged(&pixelQuintupletsInGPU.phi, maxPixelQuintuplets * sizeof(float));
#ifdef CUT_VALUE_DEBUG
    cudaMallocManaged(&pixelQuintupletsInGPU.rzChiSquared, maxPixelQuintuplets * sizeof(unsigned int));
    cudaMallocManaged(&pixelQuintupletsInGPU.rPhiChiSquared, maxPixelQuintuplets * sizeof(unsigned int));
    cudaMallocManaged(&pixelQuintupletsInGPU.rPhiChiSquaredInwards, maxPixelQuintuplets * sizeof(unsigned int));
#endif

    cudaMemset(pixelQuintupletsInGPU.nPixelQuintuplets, 0, sizeof(unsigned int));
}

void SDL::createPixelQuintupletsInExplicitMemory(struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, unsigned int maxPixelQuintuplets)
{
    cudaMalloc(&pixelQuintupletsInGPU.pixelIndices, maxPixelQuintuplets * sizeof(unsigned int));
    cudaMalloc(&pixelQuintupletsInGPU.T5Indices, maxPixelQuintuplets * sizeof(unsigned int));
    cudaMalloc(&pixelQuintupletsInGPU.nPixelQuintuplets, sizeof(unsigned int));
    cudaMalloc(&pixelQuintupletsInGPU.isDup, maxPixelQuintuplets * sizeof(bool));
    cudaMalloc(&pixelQuintupletsInGPU.score, maxPixelQuintuplets * sizeof(float));
    cudaMalloc(&pixelQuintupletsInGPU.eta, maxPixelQuintuplets * sizeof(float));
    cudaMalloc(&pixelQuintupletsInGPU.phi, maxPixelQuintuplets * sizeof(float));

    cudaMalloc(&pixelQuintupletsInGPU.logicalLayers, maxPixelQuintuplets * 7 *sizeof(unsigned int));
    cudaMalloc(&pixelQuintupletsInGPU.hitIndices, maxPixelQuintuplets * 14 * sizeof(unsigned int));
    cudaMalloc(&pixelQuintupletsInGPU.lowerModuleIndices, maxPixelQuintuplets * 7 * sizeof(unsigned int));

    cudaMemset(pixelQuintupletsInGPU.nPixelQuintuplets, 0, sizeof(unsigned int));
}

__device__ void SDL::rmPixelQuintupletToMemory(struct pixelQuintuplets& pixelQuintupletsInGPU, unsigned int pixelQuintupletIndex)
{

    pixelQuintupletsInGPU.isDup[pixelQuintupletIndex] = 1;
}
#ifdef CUT_VALUE_DEBUG
__device__ void SDL::addPixelQuintupletToMemory(struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct quintuplets& quintupletsInGPU, struct pixelQuintuplets& pixelQuintupletsInGPU, unsigned int pixelIndex, unsigned int T5Index, unsigned int pixelQuintupletIndex, float& rzChiSquared, float& rPhiChiSquared, float& rPhiChiSquaredInwards, float score, float eta, float phi)
#else
__device__ void SDL::addPixelQuintupletToMemory(struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct quintuplets& quintupletsInGPU, struct pixelQuintuplets& pixelQuintupletsInGPU, unsigned int pixelIndex, unsigned int T5Index, unsigned int pixelQuintupletIndex, float score, float eta, float phi)
#endif
{
    pixelQuintupletsInGPU.pixelIndices[pixelQuintupletIndex] = pixelIndex;
    pixelQuintupletsInGPU.T5Indices[pixelQuintupletIndex] = T5Index;
    pixelQuintupletsInGPU.isDup[pixelQuintupletIndex] = 0;
    pixelQuintupletsInGPU.score[pixelQuintupletIndex] = score;

    pixelQuintupletsInGPU.logicalLayers[7 * pixelQuintupletIndex] = 0;
    pixelQuintupletsInGPU.logicalLayers[7 * pixelQuintupletIndex + 1] = 0;
    pixelQuintupletsInGPU.logicalLayers[7 * pixelQuintupletIndex + 2] = quintupletsInGPU.logicalLayers[T5Index * 5];
    pixelQuintupletsInGPU.logicalLayers[7 * pixelQuintupletIndex + 3] = quintupletsInGPU.logicalLayers[T5Index * 5 + 1];
    pixelQuintupletsInGPU.logicalLayers[7 * pixelQuintupletIndex + 4] = quintupletsInGPU.logicalLayers[T5Index * 5 + 2];
    pixelQuintupletsInGPU.logicalLayers[7 * pixelQuintupletIndex + 5] = quintupletsInGPU.logicalLayers[T5Index * 5 + 3];
    pixelQuintupletsInGPU.logicalLayers[7 * pixelQuintupletIndex + 6] = quintupletsInGPU.logicalLayers[T5Index * 5 + 4];

    pixelQuintupletsInGPU.lowerModuleIndices[7 * pixelQuintupletIndex] = segmentsInGPU.innerMiniDoubletAnchorHitIndices[pixelIndex];
    pixelQuintupletsInGPU.lowerModuleIndices[7 * pixelQuintupletIndex + 1] = segmentsInGPU.outerMiniDoubletAnchorHitIndices[pixelIndex];
    pixelQuintupletsInGPU.lowerModuleIndices[7 * pixelQuintupletIndex + 2] = quintupletsInGPU.lowerModuleIndices[T5Index * 5];
    pixelQuintupletsInGPU.lowerModuleIndices[7 * pixelQuintupletIndex + 3] = quintupletsInGPU.lowerModuleIndices[T5Index * 5 + 1];
    pixelQuintupletsInGPU.lowerModuleIndices[7 * pixelQuintupletIndex + 4] = quintupletsInGPU.lowerModuleIndices[T5Index * 5 + 2];
    pixelQuintupletsInGPU.lowerModuleIndices[7 * pixelQuintupletIndex + 5] = quintupletsInGPU.lowerModuleIndices[T5Index * 5 + 3];
    pixelQuintupletsInGPU.lowerModuleIndices[7 * pixelQuintupletIndex + 6] = quintupletsInGPU.lowerModuleIndices[T5Index * 5 + 4];


    unsigned int pixelInnerMD = segmentsInGPU.mdIndices[2 * pixelIndex];
    unsigned int pixelOuterMD = segmentsInGPU.mdIndices[2 * pixelIndex + 1];

    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex] = mdsInGPU.hitIndices[2 * pixelInnerMD];
    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 1] = mdsInGPU.hitIndices[2 * pixelInnerMD + 1];
    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 2] = mdsInGPU.hitIndices[2 * pixelOuterMD];
    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 3] = mdsInGPU.hitIndices[2 * pixelOuterMD + 1];

    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 4] = quintupletsInGPU.hitIndices[10 * T5Index];
    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 5] = quintupletsInGPU.hitIndices[10 * T5Index + 1];
    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 6] = quintupletsInGPU.hitIndices[10 * T5Index + 2];
    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 7] = quintupletsInGPU.hitIndices[10 * T5Index + 3];
    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 8] = quintupletsInGPU.hitIndices[10 * T5Index + 4];
    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 9] = quintupletsInGPU.hitIndices[10 * T5Index + 5];
    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 10] = quintupletsInGPU.hitIndices[10 * T5Index + 6];
    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 11] = quintupletsInGPU.hitIndices[10 * T5Index + 7];
    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 12] = quintupletsInGPU.hitIndices[10 * T5Index + 8];
    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 13] = quintupletsInGPU.hitIndices[10 * T5Index + 9];

    pixelQuintupletsInGPU.eta[pixelQuintupletIndex] = eta;
    pixelQuintupletsInGPU.phi[pixelQuintupletIndex] = phi;
    
#ifdef CUT_VALUE_DEBUG
    pixelQuintupletsInGPU.rzChiSquared[pixelQuintupletIndex] = rzChiSquared;
    pixelQuintupletsInGPU.rPhiChiSquared[pixelQuintupletIndex] = rPhiChiSquared;
    pixelQuintupletsInGPU.rPhiChiSquaredInwards[pixelQuintupletIndex] = rPhiChiSquaredInwards;
#endif
}

__device__ bool SDL::runPixelQuintupletDefaultAlgo(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct triplets& tripletsInGPU, struct quintuplets& quintupletsInGPU, unsigned int& pixelSegmentIndex, unsigned int& quintupletIndex, float& rzChiSquared, float& rPhiChiSquared, float& rPhiChiSquaredInwards)
{
    bool pass = true;
    
    unsigned int pixelModuleIndex = segmentsInGPU.innerLowerModuleIndices[pixelSegmentIndex];

    unsigned int pixelSegmentArrayIndex = pixelSegmentIndex - (600 * pixelModuleIndex);

    unsigned int T5InnerT3Index = quintupletsInGPU.tripletIndices[2 * quintupletIndex];
    unsigned int T5OuterT3Index = quintupletsInGPU.tripletIndices[2 * quintupletIndex + 1];

    unsigned int firstSegmentIndex = tripletsInGPU.segmentIndices[2 * T5InnerT3Index];
    unsigned int secondSegmentIndex = tripletsInGPU.segmentIndices[2 * T5InnerT3Index + 1];
    unsigned int thirdSegmentIndex = tripletsInGPU.segmentIndices[2 * T5OuterT3Index];
    unsigned int fourthSegmentIndex = tripletsInGPU.segmentIndices[2 * T5OuterT3Index + 1];

    unsigned int pixelInnerMDIndex = segmentsInGPU.mdIndices[2 * pixelSegmentIndex];
    unsigned int pixelOuterMDIndex = segmentsInGPU.mdIndices[2 * pixelSegmentIndex + 1];

    unsigned int pixelAnchorHitIndex1 = mdsInGPU.hitIndices[2 * pixelInnerMDIndex];
    unsigned int pixelNonAnchorHitIndex1 = mdsInGPU.hitIndices[2 * pixelInnerMDIndex + 1];
    unsigned int pixelAnchorHitIndex2 = mdsInGPU.hitIndices[2 * pixelOuterMDIndex];
    unsigned int pixelNonAnchorHitIndex2 = mdsInGPU.hitIndices[2 * pixelOuterMDIndex + 1];

    unsigned int anchorHitIndex1 = segmentsInGPU.innerMiniDoubletAnchorHitIndices[firstSegmentIndex];
    unsigned int anchorHitIndex2 = segmentsInGPU.outerMiniDoubletAnchorHitIndices[firstSegmentIndex]; //same as second segment inner MD anchorhit index
    unsigned int anchorHitIndex3 = segmentsInGPU.outerMiniDoubletAnchorHitIndices[secondSegmentIndex]; //same as third segment inner MD anchor hit index

    unsigned int anchorHitIndex4 = segmentsInGPU.outerMiniDoubletAnchorHitIndices[thirdSegmentIndex]; //same as fourth segment inner MD anchor hit index
    unsigned int anchorHitIndex5 = segmentsInGPU.outerMiniDoubletAnchorHitIndices[fourthSegmentIndex];

    unsigned int lowerModuleIndex1 = quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex];
    unsigned int lowerModuleIndex2 = quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex + 1];
    unsigned int lowerModuleIndex3 = quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex + 2];
    unsigned int lowerModuleIndex4 = quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex + 3];
    unsigned int lowerModuleIndex5 = quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex + 4];

    unsigned int lowerModuleIndices[] = {lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4, lowerModuleIndex5};
    unsigned int anchorHits[] = {anchorHitIndex1, anchorHitIndex2, anchorHitIndex3, anchorHitIndex4, anchorHitIndex5};
    unsigned int pixelHits[] = {pixelAnchorHitIndex1, pixelAnchorHitIndex2};
    
    float pixelRadius, pixelRadiusError, tripletRadius, rPhiChiSquaredTemp, rzChiSquaredTemp, rPhiChiSquaredInwardsTemp;

    pass = pass & runPixelTripletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, tripletsInGPU, pixelSegmentIndex, T5InnerT3Index, pixelRadius, pixelRadiusError, tripletRadius, rzChiSquaredTemp, rPhiChiSquaredTemp, rPhiChiSquaredInwardsTemp, false);

    rzChiSquared = computePT5RZChiSquared(modulesInGPU, hitsInGPU, pixelAnchorHitIndex1, pixelAnchorHitIndex2, anchorHits, lowerModuleIndices);

    rPhiChiSquared = computePT5RPhiChiSquared(modulesInGPU, hitsInGPU, segmentsInGPU, pixelSegmentArrayIndex, anchorHits, lowerModuleIndices);

    rPhiChiSquaredInwards = computePT5RPhiChiSquaredInwards(modulesInGPU, hitsInGPU, quintupletsInGPU, quintupletIndex, pixelHits);

    if(segmentsInGPU.circleRadius[pixelSegmentArrayIndex] < 5.0/(2 * k2Rinv1GeVf))
    {
        pass = pass & passPT5RZChiSquaredCuts(modulesInGPU, lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4, lowerModuleIndex5, rzChiSquared);

        pass = pass & passPT5RPhiChiSquaredCuts(modulesInGPU, lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4, lowerModuleIndex5, rPhiChiSquared);
    }
    
    if(quintupletsInGPU.regressionRadius[quintupletIndex] < 5.0/(2 * k2Rinv1GeVf))
    {
        pass = pass & passPT5RPhiChiSquaredInwardsCuts(modulesInGPU, lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4, lowerModuleIndex5, rPhiChiSquaredInwards);
    }

    //other cuts will be filled here!
    return pass;
}

__device__ bool SDL::passPT5RPhiChiSquaredCuts(struct modules& modulesInGPU, unsigned int lowerModuleIndex1, unsigned int lowerModuleIndex2, unsigned int lowerModuleIndex3, unsigned int lowerModuleIndex4, unsigned int lowerModuleIndex5, float rPhiChiSquared)
{
    const int layer1 = modulesInGPU.layers[lowerModuleIndex1] + 6 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex1] == SDL::TwoS);
    const int layer2 = modulesInGPU.layers[lowerModuleIndex2] + 6 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex2] == SDL::TwoS);
    const int layer3 = modulesInGPU.layers[lowerModuleIndex3] + 6 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex3] == SDL::TwoS);
    const int layer4 = modulesInGPU.layers[lowerModuleIndex4] + 6 * (modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex4] == SDL::TwoS);
    const int layer5 = modulesInGPU.layers[lowerModuleIndex5] + 6 * (modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex5] == SDL::TwoS);

    if(layer1 == 1 and layer2 == 2 and layer3 == 3)
    {
        if(layer4 == 12 and layer5 == 13)
        {
            return rPhiChiSquared < 48.921;
        }
        else if(layer4 == 4 and layer5 == 12)
        {
            return rPhiChiSquared < 97.948;
        }
        else if(layer4 == 4 and layer5 == 5)
        {
            return rPhiChiSquared < 129.3;
        }
        else if(layer4 == 7 and layer5 == 13)
        {
            return rPhiChiSquared < 56.21;
        }
        else if(layer4 == 7 and layer5 == 8)
        {
            return rPhiChiSquared < 74.198;
        }
    }
    else if(layer1 == 1 and layer2 == 2 and layer3 == 7)
    {
        if(layer4 == 13 and layer5 == 14)
        {
            return rPhiChiSquared < 21.265;
        }
        else if(layer4 == 8 and layer5 == 14)
        {
            return rPhiChiSquared < 37.058;
        }
        else if(layer4 == 8 and layer5 == 9)
        {
            return rPhiChiSquared < 42.578;
        }
    }
    else if(layer1 == 1 and layer2 == 7 and layer3 == 8)
    {
        if(layer4 == 9 and layer5 == 10)
        {
            return rPhiChiSquared < 32.253;
        }
        else if(layer4 == 9 and layer5 == 15)
        {
            return rPhiChiSquared < 37.058;
        }
    }
    else if(layer1 == 2 and layer2 == 3 and layer3 == 4)
    {
        if(layer4 == 12 and layer5 == 13)
        {
            return rPhiChiSquared < 97.947;
        }
        else if(layer4 == 5 and layer5 == 12)
        {
            return rPhiChiSquared < 129.3;
        }
        else if(layer4 == 5 and layer5 == 6)
        {
            return rPhiChiSquared < 170.68;
        }
    }
    else if(layer1 == 2 and layer2 == 3 and layer3 == 7)
    {
        if(layer4 == 13 and layer5 == 14)
        {   
            return rPhiChiSquared < 48.92;
        }
        else if(layer4 == 8 and layer5 == 14)
        {
            return rPhiChiSquared < 74.2;
        }
    }
    else if(layer1 == 2 and layer2 == 7 and layer3 == 8)
    {
        if(layer4 == 14 and layer5 == 15)
        {
            return rPhiChiSquared < 42.58;
        }
        else if(layer4 == 9 and layer5 == 10)
        {
            return rPhiChiSquared < 37.06;
        }
        else if(layer4 == 9 and layer5 == 15)
        {
            return rPhiChiSquared < 48.92;
        }
    }
    else if(layer1 == 3 and layer2 == 7 and layer3 == 8 and layer4 == 14 and layer5 == 15)
    {
        return rPhiChiSquared < 85.25;
    }
    else if(layer1 == 7 and layer2 == 8 and layer3 == 9)
    {
        if(layer4 == 10 and layer5 == 11)
        {
            return rPhiChiSquared < 42.58;
        }
        else if(layer4 == 10 and layer5 == 16)
        {
            return rPhiChiSquared < 37.06;
        }
        else if(layer4 == 15 and layer5 == 16)
        {
            return rPhiChiSquared < 37.06;
        }
    }
    return true;
}



__device__ bool SDL::passPT5RPhiChiSquaredInwardsCuts(struct modules& modulesInGPU, unsigned int lowerModuleIndex1, unsigned int lowerModuleIndex2, unsigned int lowerModuleIndex3, unsigned int lowerModuleIndex4, unsigned int lowerModuleIndex5, float rPhiChiSquared)
{
    const int layer1 = modulesInGPU.layers[lowerModuleIndex1] + 6 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex1] == SDL::TwoS);
    const int layer2 = modulesInGPU.layers[lowerModuleIndex2] + 6 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex2] == SDL::TwoS);
    const int layer3 = modulesInGPU.layers[lowerModuleIndex3] + 6 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex3] == SDL::TwoS);
    const int layer4 = modulesInGPU.layers[lowerModuleIndex4] + 6 * (modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex4] == SDL::TwoS);
    const int layer5 = modulesInGPU.layers[lowerModuleIndex5] + 6 * (modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex5] == SDL::TwoS);

    if(layer1 == 1 and layer2 == 2 and layer3 == 3)
    {
        if(layer4 == 12 and layer5 == 13)
        {
            return rPhiChiSquared < 451.141;
        }
        else if(layer4 == 4 and layer5 == 12)
        {
            return rPhiChiSquared < 786.173;
        }
        else if(layer4 == 4 and layer5 == 5)
        {
            return rPhiChiSquared < 595.545;
        }
        else if(layer4 == 7 and layer5 == 13)
        {
            return rPhiChiSquared < 581.339;
        }
        else if(layer4 == 7 and layer5 == 8)
        {
            return rPhiChiSquared < 112.537;
        }
    }
    else if(layer1 == 1 and layer2 == 2 and layer3 == 7)
    {
        if(layer4 == 13 and layer5 == 14)
        {
            return rPhiChiSquared < 225.322;
        }
        else if(layer4 == 8 and layer5 == 14)
        {
            return rPhiChiSquared < 1192.402;
        }
        else if(layer4 == 8 and layer5 == 9)
        {
            return rPhiChiSquared < 786.173;
        }
    }
    else if(layer1 == 1 and layer2 == 7 and layer3 == 8)
    {
        if(layer4 == 9 and layer5 == 10)
        {
            return rPhiChiSquared < 1037.817;
        }
        else if(layer4 == 9 and layer5 == 15)
        {
            return rPhiChiSquared < 1808.536;
        }
    }
    else if(layer1 == 2 and layer2 == 3 and layer3 == 4)
    {
        if(layer4 == 12 and layer5 == 13)
        {
            return rPhiChiSquared < 684.253;
        }
        else if(layer4 == 5 and layer5 == 12)
        {
            return rPhiChiSquared < 684.253;
        }
        else if(layer4 == 5 and layer5 == 6)
        {
            return rPhiChiSquared < 684.253;
        }
    }
    else if(layer1 == 2 and layer2 == 3 and layer3 == 7)
    {
        if(layer4 == 13 and layer5 == 14)
        {   
            return rPhiChiSquared < 451.141;
        }
        else if(layer4 == 8 and layer5 == 14)
        {
            return rPhiChiSquared < 518.34;
        }
    }
    else if(layer1 == 2 and layer2 == 7 and layer3 == 8)
    {
        if(layer4 == 14 and layer5 == 15)
        {
            return rPhiChiSquared < 2077.92;
        }
        else if(layer4 == 9 and layer5 == 10)
        {
            return rPhiChiSquared < 74.20;
        }
        else if(layer4 == 9 and layer5 == 15)
        {
            return rPhiChiSquared < 1808.536;
        }
    }
    else if(layer1 == 3 and layer2 == 7 and layer3 == 8 and layer4 == 14 and layer5 == 15)
    {
        return rPhiChiSquared < 786.173;
    }
    else if(layer1 == 7 and layer2 == 8 and layer3 == 9)
    {
        if(layer4 == 10 and layer5 == 11)
        {
            return rPhiChiSquared < 1574.076;
        }
        else if(layer4 == 10 and layer5 == 16)
        {
            return rPhiChiSquared < 5492.11;
        }
        else if(layer4 == 15 and layer5 == 16)
        {
            return rPhiChiSquared < 2743.037;
        }
    }
    return true;
}

__device__ float SDL::computePT5RPhiChiSquared(struct modules& modulesInGPU, struct hits& hitsInGPU, struct segments& segmentsInGPU, unsigned int pixelSegmentArrayIndex, unsigned int* anchorHits, unsigned int* lowerModuleIndices)
{
    /*
       Compute circle parameters from 3 pixel hits, and then use them to compute the chi squared for the outer hits
    */

    float g = segmentsInGPU.circleCenterX[pixelSegmentArrayIndex];
    float f = segmentsInGPU.circleCenterY[pixelSegmentArrayIndex];
    float radius = segmentsInGPU.circleRadius[pixelSegmentArrayIndex];
    float delta1[5], delta2[5], slopes[5];
    bool isFlat[5];
    float xs[5];
    float ys[5];
    float chiSquared = 0;
    for(size_t i = 0; i < 5; i++)
    {
        xs[i] = hitsInGPU.xs[anchorHits[i]];
        ys[i] = hitsInGPU.ys[anchorHits[i]];
    }

    computeSigmasForRegression(modulesInGPU, lowerModuleIndices, delta1, delta2, slopes, isFlat);
    chiSquared = computeChiSquared(5, xs, ys, delta1, delta2, slopes, isFlat, g, f, radius);

    return chiSquared;
}

__device__ bool SDL::passPT5RZChiSquaredCuts(struct modules& modulesInGPU, unsigned int lowerModuleIndex1, unsigned int lowerModuleIndex2, unsigned int lowerModuleIndex3, unsigned int lowerModuleIndex4, unsigned int lowerModuleIndex5, float& rzChiSquared)
{
    const int layer1 = modulesInGPU.layers[lowerModuleIndex1] + 6 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex1] == SDL::TwoS);
    const int layer2 = modulesInGPU.layers[lowerModuleIndex2] + 6 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex2] == SDL::TwoS);
    const int layer3 = modulesInGPU.layers[lowerModuleIndex3] + 6 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex3] == SDL::TwoS);
    const int layer4 = modulesInGPU.layers[lowerModuleIndex4] + 6 * (modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex4] == SDL::TwoS);
    const int layer5 = modulesInGPU.layers[lowerModuleIndex5] + 6 * (modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex5] == SDL::TwoS);

    
    if(layer1 == 1 and layer2 == 2 and layer3 == 3)
    {
        if(layer4 == 12 and layer5 == 13)
        {
            return rzChiSquared < 451.141;
        }
        else if(layer4 == 4 and layer5 == 12)
        {
            return rzChiSquared < 392.654;
        }
        else if(layer4 == 4 and layer5 == 5)
        {
            return rzChiSquared < 225.322;
        }
        else if(layer4 == 7 and layer5 == 13)
        {
            return rzChiSquared < 595.546;
        }
        else if(layer4 == 7 and layer5 == 8)
        {
            return rzChiSquared < 196.111;
        }
    }
    else if(layer1 == 1 and layer2 == 2 and layer3 == 7)
    {
        if(layer4 == 13 and layer5 == 14)
        {
            return rzChiSquared < 297.446;
        }
        else if(layer4 == 8 and layer5 == 14)
        {   
            return rzChiSquared < 451.141;
        }
        else if(layer4 == 8 and layer5 == 9)
        {
            return rzChiSquared < 518.339;
        }
    }
    else if(layer1 == 1 and layer2 == 7 and layer3 == 8)
    {
        if(layer4 == 9 and layer5 == 10)
        {
            return rzChiSquared < 341.75;
        }
        else if(layer4 == 9 and layer5 == 15)
        {
            return rzChiSquared < 341.75;
        }
    }
    else if(layer1 == 2 and layer2 == 3 and layer3 == 4)
    {
        if(layer4 == 12 and layer5 == 13)
        {
            return rzChiSquared < 392.655;
        }
        else if(layer4 == 5 and layer5 == 12)
        {
            return rzChiSquared < 341.75;
        }
        else if(layer4 == 5 and layer5 == 6)
        {
            return rzChiSquared < 112.537;
        }
    }
    else if(layer1 == 2 and layer2 == 3 and layer4 == 7)
    {
        if(layer4 == 13 and layer5 == 14)
        {
            return rzChiSquared < 595.545;
        }
        else if(layer4 == 8 and layer5 == 14)
        {
            return rzChiSquared < 74.198;
        }
    }
    else if(layer1 == 2 and layer2 == 7 and layer3 == 8)
    {
        if(layer4 == 14 and layer5 == 15)
        {
            return rzChiSquared < 518.339;
        }
        else if(layer4 == 9 and layer5 == 10)
        {
            return rzChiSquared < 8.046;
        }
        else if(layer4 == 9 and layer5 == 15)
        {
            return rzChiSquared < 451.141;
        }
    }
    else if(layer1 == 3 and layer2 == 7 and layer3 == 8 and layer4 == 14 and layer5 == 15)
    {
        return rzChiSquared < 56.207;
    }
    else if(layer1 == 7 and layer2 == 8 and layer3 == 9)
    {
        if(layer4 == 10 and layer5 == 11)
        {
            return rzChiSquared < 64.578;
        }
        else if(layer4 == 10 and layer5 == 16)
        {
            return rzChiSquared < 85.250;
        }
        else if(layer4 == 15 and layer5 == 16)
        {
            return rzChiSquared < 85.250;
        }
    }
    return true;
}

__device__ float SDL::computePT5RPhiChiSquaredInwards(struct modules& modulesInGPU, struct hits& hitsInGPU, struct quintuplets& quintupletsInGPU, unsigned int quintupletIndex, unsigned int* pixelHits)
{
    /*Using the computed regression center and radius, compute the chi squared for the pixels*/
    float g = quintupletsInGPU.regressionG[quintupletIndex];
    float f = quintupletsInGPU.regressionF[quintupletIndex];
    float r = quintupletsInGPU.regressionRadius[quintupletIndex];
    float x, y;
    float chiSquared = 0;   
    for(size_t i = 0; i < 2; i++)
    {
        x = hitsInGPU.xs[pixelHits[i]];
        y = hitsInGPU.ys[pixelHits[i]];
        float residual = (x - g) * (x -g) + (y - f) * (y - f) - r * r;
        chiSquared += residual * residual;
    }
    chiSquared /= 2;
    return chiSquared;
}

__device__ float SDL::computePT5RZChiSquared(struct modules& modulesInGPU, struct hits& hitsInGPU, unsigned int& pixelAnchorHitIndex1, unsigned int& pixelAnchorHitIndex2, unsigned int* anchorHits, unsigned int* lowerModuleIndices)
{
    //use the two anchor hits of the pixel segment to compute the slope
    //then compute the pseudo chi squared of the five outer hits

    float& rtPix1 = hitsInGPU.rts[pixelAnchorHitIndex1];
    float& rtPix2 = hitsInGPU.rts[pixelAnchorHitIndex2];
    float& zPix1 = hitsInGPU.zs[pixelAnchorHitIndex1];
    float& zPix2 = hitsInGPU.zs[pixelAnchorHitIndex2];
    float slope = (zPix2 - zPix1)/(rtPix2 - rtPix1);
    float rtAnchor, zAnchor;
    float residual = 0;
    float error = 0;
    //hardcoded array indices!!!
    float RMSE = 0;
    float drdz;
    for(size_t i = 0; i < 5; i++)
    {
        unsigned int& anchorHitIndex = anchorHits[i];
        unsigned int& lowerModuleIndex = lowerModuleIndices[i];
        rtAnchor = hitsInGPU.rts[anchorHitIndex];
        zAnchor = hitsInGPU.zs[anchorHitIndex];

        const int moduleType = modulesInGPU.moduleType[lowerModuleIndex];
        const int moduleSide = modulesInGPU.sides[lowerModuleIndex];
        const int moduleLayerType = modulesInGPU.moduleLayerType[lowerModuleIndex];
        const int layer = modulesInGPU.layers[lowerModuleIndex] + 6 * (modulesInGPU.subdets[lowerModuleIndex] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex] == SDL::TwoS);
        
        residual = (layer <= 6) ?  (zAnchor - zPix1) - slope * (rtAnchor - rtPix1) : (rtAnchor - rtPix1) - (zAnchor - zPix1)/slope;
        
        //PS Modules
        if(moduleType == 0)
        {
            error = 0.15;
        }
        else //2S modules
        {
            error = 5.0;
        }

        //special dispensation to tilted PS modules!
        if(moduleType == 0 and layer <= 6 and moduleSide != Center)
        {
            if(moduleLayerType == Strip)
            {
                drdz = modulesInGPU.drdzs[lowerModuleIndex];
            }
            else
            {
                drdz = modulesInGPU.drdzs[modulesInGPU.partnerModuleIndex(lowerModuleIndex)];
            }

            error *= 1/sqrtf(1 + drdz * drdz);
        }
        RMSE += (residual * residual)/(error * error);
    }

    RMSE = sqrtf(0.2 * RMSE);
    return RMSE;
}

