#ifdef __CUDACC__
#define CUDA_CONST_VAR __device__
#endif
# include "PixelQuintuplet.cuh"
#include "allocate.h"

SDL::pixelQuintuplets::pixelQuintuplets()
{
    pT3Indices = nullptr;
    T5Indices = nullptr;
    nPixelQuintuplets = nullptr;
}

SDL::pixelQuintuplets::~pixelQuintuplets()
{
}

void SDL::pixelQuintuplets::freeMemory()
{
    cudaFree(pT3Indices);
    cudaFree(T5Indices);
    cudaFree(nPixelQuintuplets);
}

void SDL::createPixelQuintupletsInUnifiedMemory(struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, unsigned int maxPixelQuintuplets)
{
    cudaMallocManaged(&pixelQuintupletsInGPU.pT3Indices, maxPixelQuintuplets * sizeof(unsigned int));
    cudaMallocManaged(&pixelQuintupletsInGPU.T5Indices, maxPixelQuintuplets * sizeof(unsigned int));
    cudaMallocManaged(&pixelQuintupletsInGPU.nPixelQuintuplets, sizeof(unsigned int));
#ifdef CUT_VALUE_DEBUG
    cudaMallocManaged(&pixelQuintupletsInGPU.rzChiSquared, maxPixelQuintuplets * sizeof(unsigned int));
#endif

    cudaMemset(pixelQuintupletsInGPU.nPixelQuintuplets, 0, sizeof(unsigned int));
}

void SDL::createPixelQuintupletsInExplicitMemory(struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, unsigned int maxPixelQuintuplets)
{
    cudaMalloc(&pixelQuintupletsInGPU.pT3Indices, maxPixelQuintuplets * sizeof(unsigned int));
    cudaMalloc(&pixelQuintupletsInGPU.T5Indices, maxPixelQuintuplets * sizeof(unsigned int));
    cudaMalloc(&pixelQuintupletsInGPU.nPixelQuintuplets, sizeof(unsigned int));

    cudaMemset(pixelQuintupletsInGPU.nPixelQuintuplets, 0, sizeof(unsigned int));
}

#ifdef CUT_VALUE_DEBUG
__device__ void SDL::addPixelQuintupletToMemory(struct pixelQuintuplets& pixelQuintupletsInGPU, unsigned int pT3Index, unsigned int T5Index, unsigned int pixelQuintupletIndex, float& rzChiSquared)
#else
__device__ void SDL::addPixelQuintupletToMemory(struct pixelQuintuplets& pixelQuintupletsInGPU, unsigned int pT3Index, unsigned int T5Index, unsigned int pixelQuintupletIndex)
#endif
{
    pixelQuintupletsInGPU.pT3Indices[pixelQuintupletIndex] = pT3Index;
    pixelQuintupletsInGPU.T5Indices[pixelQuintupletIndex] = T5Index;
#ifdef CUT_VALUE_DEBUG
    pixelQuintupletsInGPU.rzChiSquared[pixelQuintupletIndex] = rzChiSquared;
#endif
}

__device__ bool SDL::runPixelQuintupletDefaultAlgo(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct triplets& tripletsInGPU, struct quintuplets& quintupletsInGPU, struct pixelTriplets& pixelTripletsInGPU, unsigned int& pixelTripletIndex, unsigned int& quintupletIndex, float& rzChiSquared)
{
    bool pass = true;
    
    unsigned int pT3OuterT3Index = pixelTripletsInGPU.tripletIndices[pixelTripletIndex];
    unsigned int pT3InnerSegmentIndex = pixelTripletsInGPU.pixelSegmentIndices[pixelTripletIndex];

    unsigned int T5InnerT3Index = quintupletsInGPU.tripletIndices[2 * quintupletIndex];
    unsigned int T5OuterT3Index = quintupletsInGPU.tripletIndices[2 * quintupletIndex + 1];

    unsigned int firstSegmentIndex = tripletsInGPU.segmentIndices[2 * T5InnerT3Index];
    unsigned int secondSegmentIndex = tripletsInGPU.segmentIndices[2 * T5InnerT3Index + 1];
    unsigned int thirdSegmentIndex = tripletsInGPU.segmentIndices[2 * T5OuterT3Index];
    unsigned int fourthSegmentIndex = tripletsInGPU.segmentIndices[2 * T5OuterT3Index + 1];

    unsigned int pixelAnchorHitIndex1 = segmentsInGPU.innerMiniDoubletAnchorHitIndices[pT3InnerSegmentIndex];
    unsigned int pixelAnchorHitIndex2 = segmentsInGPU.outerMiniDoubletAnchorHitIndices[pT3InnerSegmentIndex];

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


    //cut 1 -> common T3
    pass = pass & (pT3OuterT3Index == T5InnerT3Index);

    rzChiSquared = computePT5ChiSquared(modulesInGPU, hitsInGPU, pixelAnchorHitIndex1, pixelAnchorHitIndex2, anchorHitIndex1, anchorHitIndex2, anchorHitIndex3, anchorHitIndex4, anchorHitIndex5, lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4, lowerModuleIndex5);
    //other cuts will be filled here!
    return pass;
}


__device__ float SDL::computePT5ChiSquared(struct modules& modulesInGPU, struct hits& hitsInGPU, unsigned int& pixelAnchorHitIndex1, unsigned int& pixelAnchorHitIndex2, unsigned int& anchorHitIndex1, unsigned int& anchorHitIndex2, unsigned int& anchorHitIndex3, unsigned int anchorHitIndex4, unsigned int& anchorHitIndex5, unsigned int& lowerModuleIndex1, unsigned int& lowerModuleIndex2, unsigned int& lowerModuleIndex3, unsigned int& lowerModuleIndex4, unsigned int& lowerModuleIndex5) 
{
    //use the two anchor hits of the pixel segment to compute the slope
    //then compute the pseudo chi squared of the five outer hits

    float& rtPix1 = hitsInGPU.rts[pixelAnchorHitIndex1];
    float& rtPix2 = hitsInGPU.rts[pixelAnchorHitIndex2];
    float& zPix1 = hitsInGPU.zs[pixelAnchorHitIndex1];
    float& zPix2 = hitsInGPU.zs[pixelAnchorHitIndex2];

    float& rtAnchor1 = hitsInGPU.rts[anchorHitIndex1];
    float& rtAnchor2 = hitsInGPU.rts[anchorHitIndex2];
    float& rtAnchor3 = hitsInGPU.rts[anchorHitIndex3];
    float& rtAnchor4 = hitsInGPU.rts[anchorHitIndex4];
    float& rtAnchor5 = hitsInGPU.rts[anchorHitIndex5];

    float& zAnchor1 = hitsInGPU.zs[anchorHitIndex1];
    float& zAnchor2 = hitsInGPU.zs[anchorHitIndex2];
    float& zAnchor3 = hitsInGPU.zs[anchorHitIndex3];
    float& zAnchor4 = hitsInGPU.zs[anchorHitIndex4];
    float& zAnchor5 = hitsInGPU.zs[anchorHitIndex5];

    const int moduleLayer1 = modulesInGPU.moduleType[lowerModuleIndex1];
    const int moduleLayer2 = modulesInGPU.moduleType[lowerModuleIndex2];
    const int moduleLayer3 = modulesInGPU.moduleType[lowerModuleIndex3];
    const int moduleLayer4 = modulesInGPU.moduleType[lowerModuleIndex4];
    const int moduleLayer5 = modulesInGPU.moduleType[lowerModuleIndex5];

    const int layer1 = modulesInGPU.layers[lowerModuleIndex1] + 6 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex1] == SDL::TwoS);
    const int layer2 = modulesInGPU.layers[lowerModuleIndex2] + 6 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex2] == SDL::TwoS);
    const int layer3 = modulesInGPU.layers[lowerModuleIndex3] + 6 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex3] == SDL::TwoS);
    const int layer4 = modulesInGPU.layers[lowerModuleIndex4] + 6 * (modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex4] == SDL::TwoS);
    const int layer5 = modulesInGPU.layers[lowerModuleIndex5] + 6 * (modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex5] == SDL::TwoS);


    float slope = (zPix2 - zPix1)/(rtPix2 - rtPix1);

    float residual1 = (layer1 <= 6) ?  (zAnchor1 - zPix1) - slope * (rtAnchor1 - rtPix1) : (rtAnchor1 - rtPix1) - (zAnchor1 - zPix1)/slope;
    float residual2 = (layer1 <= 6) ?  (zAnchor2 - zPix1) - slope * (rtAnchor2 - rtPix1) : (rtAnchor2 - rtPix1) - (zAnchor2 - zPix1)/slope;
    float residual3 = (layer1 <= 6) ?  (zAnchor3 - zPix1) - slope * (rtAnchor3 - rtPix1) : (rtAnchor3 - rtPix1) - (zAnchor3 - zPix1)/slope;
    float residual4 = (layer1 <= 6) ?  (zAnchor4 - zPix1) - slope * (rtAnchor4 - rtPix1) : (rtAnchor4 - rtPix1) - (zAnchor4 - zPix1)/slope;
    float residual5 = (layer1 <= 6) ?  (zAnchor5 - zPix1) - slope * (rtAnchor5 - rtPix1) : (rtAnchor5 - rtPix1) - (zAnchor5 - zPix1)/slope;

    //divide by uncertainties

    residual1 = (moduleLayer1 == 0) ? residual1/0.15 : residual1/5.0;
    residual2 = (moduleLayer2 == 0) ? residual2/0.15 : residual2/5.0;
    residual3 = (moduleLayer3 == 0) ? residual3/0.15 : residual3/5.0;
    residual4 = (moduleLayer4 == 0) ? residual4/0.15 : residual4/5.0;
    residual5 = (moduleLayer5 == 0) ? residual5/0.15 : residual5/5.0;

    const float RMSE = sqrtf(0.2 * (residual1 * residual1 + residual2 * residual2 + residual3 * residual3 + residual4 * residual4 + residual5 * residual5));

    return RMSE;
}


