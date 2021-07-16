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
    cudaMallocManaged(&pixelQuintupletsInGPU.rPhiChiSquared, maxPixelQuintuplets * sizeof(unsigned int));
    cudaMallocManaged(&pixelQuintupletsInGPU.rPhiChiSquaredInwards, maxPixelQuintuplets * sizeof(unsigned int));
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
__device__ void SDL::addPixelQuintupletToMemory(struct pixelQuintuplets& pixelQuintupletsInGPU, unsigned int pT3Index, unsigned int T5Index, unsigned int pixelQuintupletIndex, float& rzChiSquared, float& rPhiChiSquared, float& rPhiChiSquaredInwards)
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

__device__ bool SDL::runPixelQuintupletDefaultAlgo(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct triplets& tripletsInGPU, struct quintuplets& quintupletsInGPU, struct pixelTriplets& pixelTripletsInGPU, unsigned int& pixelTripletIndex, unsigned int& quintupletIndex, float& rzChiSquared, float& rPhiChiSquared, float& rPhiChiSquaredInwards)
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

    unsigned int pixelInnerMDIndex = segmentsInGPU.mdIndices[2 * pT3InnerSegmentIndex];
    unsigned int pixelOuterMDIndex = segmentsInGPU.mdIndices[2 * pT3InnerSegmentIndex + 1];

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
    unsigned int pixelHits[] = {pixelAnchorHitIndex1, pixelNonAnchorHitIndex1, pixelAnchorHitIndex2, pixelNonAnchorHitIndex2};

    //cut 1 -> common T3
    pass = pass & (pT3OuterT3Index == T5InnerT3Index);

    rzChiSquared = computePT5RZChiSquared(modulesInGPU, hitsInGPU, pixelAnchorHitIndex1, pixelAnchorHitIndex2, anchorHits, lowerModuleIndices);

    rPhiChiSquared = computePT5RPhiChiSquared(modulesInGPU, hitsInGPU, pixelHits, anchorHits, lowerModuleIndices);

    rPhiChiSquaredInwards = computePT5RPhiChiSquaredInwards(modulesInGPU, hitsInGPU, quintupletsInGPU, quintupletIndex, pixelHits);

    //other cuts will be filled here!
    return pass;
}


__device__ float SDL::computePT5RPhiChiSquared(struct modules& modulesInGPU, struct hits& hitsInGPU, unsigned int* pixelHits, unsigned int* anchorHits, unsigned int* lowerModuleIndices)
{
    /*
       Compute circle parameters from 3 pixel hits, and then use them to compute the chi squared for the outer hits
    */
    float g, f;
    unsigned int pixelAnchorHitIndex1 = pixelHits[0];
    unsigned int pixelNonAnchorHitIndex1 = pixelHits[1];
    unsigned int pixelAnchorHitIndex2 = pixelHits[2];

    float radius = computeRadiusFromThreeAnchorHits(hitsInGPU.xs[pixelAnchorHitIndex1], hitsInGPU.ys[pixelAnchorHitIndex1], hitsInGPU.xs[pixelNonAnchorHitIndex1], hitsInGPU.ys[pixelNonAnchorHitIndex1], hitsInGPU.xs[pixelAnchorHitIndex2], hitsInGPU.ys[pixelAnchorHitIndex2], g, f);

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

__device__ float SDL::computePT5RPhiChiSquaredInwards(struct modules& modulesInGPU, struct hits& hitsInGPU, struct quintuplets& quintupletsInGPU, unsigned int quintupletIndex, unsigned int* pixelHits)
{
    /*Using the computed regression center and radius, compute the chi squared for the pixels*/
    float g = quintupletsInGPU.regressionG[quintupletIndex];
    float f = quintupletsInGPU.regressionF[quintupletIndex];
    float r = quintupletsInGPU.regressionRadius[quintupletIndex];
    float x, y;
    float chiSquared = 0;   
    int nPoints = (pixelHits[3] == pixelHits[2]) ? 3 : 4;
    for(size_t i = 0; i < nPoints; i++)
    {
        x = hitsInGPU.xs[pixelHits[i]];
        y = hitsInGPU.ys[pixelHits[i]];
        float residual = (x - g) * (x -g) + (y - f) * (y - f) - r * r;
        chiSquared += residual * residual;
    }
    chiSquared /= nPoints;
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

