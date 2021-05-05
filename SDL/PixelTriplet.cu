# include "PixelTriplet.cuh"
#include "allocate.h"

SDL::pixelTriplets::pixelTriplets()
{
    pixelSegmentIndices = nullptr;
    tripletIndices = nullptr;
    nPixelTriplets = nullptr;
}

void SDL::pixelTriplets::freeMemory()
{
    cudaFree(pixelSegmentIndices);
    cudaFree(tripletIndices);
    cudaFree(nPixelTriplets);
}

SDL::pixelTriplets::~pixelTriplets()
{
}

void SDL::createPixelTripletsInUnifiedMemory(struct pixelTriplets& pixelTripletsInGPU, unsigned int maxPixelTriplets)
{
    cudaMallocManaged(&pixelTripletsInGPU.pixelSegmentIndices, maxPixelTriplets * sizeof(float));
    cudaMallocManaged(&pixelTripletsInGPU.tripletIndices, maxPixelTriplets * sizeof(float));
    cudaMallocManaged(&pixelTripletsInGPU.nPixelTriplets, sizeof(unsigned int));
    cudaMallocManaged(&pixelTripletsInGPU.pixelRadius, maxPixelTriplets * sizeof(float));
    cudaMallocManaged(&pixelTripletsInGPU.pixelRadiusError, maxPixelTriplets * sizeof(float));
    cudaMallocManaged(&pixelTripletsInGPU.tripletRadius, maxPixelTriplets * sizeof(float));

    cudaMemset(pixelTripletsInGPU.nPixelTriplets, 0, sizeof(unsigned int));
}

void SDL::createPixelTripletsInExplicitMemory(struct pixelTriplets& pixelTripletsInGPU, unsigned int maxPixelTriplets)
{
    cudaMalloc(&pixelTripletsInGPU.pixelSegmentIndices, maxPixelTriplets * sizeof(float));
    cudaMalloc(&pixelTripletsInGPU.tripletIndices, maxPixelTriplets * sizeof(float));
    cudaMalloc(&pixelTripletsInGPU.nPixelTriplets, sizeof(unsigned int));
    cudaMalloc(&pixelTripletsInGPU.pixelRadius, maxPixelTriplets * sizeof(float));
    cudaMalloc(&pixelTripletsInGPU.pixelRadiusError, maxPixelTriplets * sizeof(float));
    cudaMalloc(&pixelTripletsInGPU.tripletRadius, maxPixelTriplets * sizeof(float));

    cudaMemset(pixelTripletsInGPU.nPixelTriplets, 0, sizeof(unsigned int));

}

__device__ void SDL::addPixelTripletToMemory(struct pixelTriplets& pixelTripletsInGPU, unsigned int pixelSegmentIndex, unsigned int tripletIndex, float pixelRadius, float pixelRadiusError, float tripletRadius, unsigned int pixelTripletIndex)
{
    pixelTripletsInGPU.pixelSegmentIndices[pixelTripletIndex] = pixelSegmentIndex;
    pixelTripletsInGPU.tripletIndices[pixelTripletIndex] = tripletIndex;
    pixelTripletsInGPU.pixelRadius[pixelTripletIndex] = pixelRadius;
    pixelTripletsInGPU.pixelRadiusError[pixelTripletIndex] = pixelRadiusError;
    pixelTripletsInGPU.tripletRadius[pixelTripletIndex] = tripletRadius;

}

__device__ bool SDL::runPixelTripletDefaultAlgo(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct triplets& tripletsInGPU, unsigned int& pixelSegmentIndex, unsigned int tripletIndex, float& pixelRadius, float& pixelRadiusError, float& tripletRadius)
{
    bool pass = true;

    //run pT4 compatibility between the pixel segment and inner segment, and between the pixel and outer segment of the triplet


    //placeholder
    float zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut;
    unsigned int pixelModuleIndex = segmentsInGPU.innerLowerModuleIndices[pixelSegmentIndex];

    unsigned int lowerModuleIndex = tripletsInGPU.lowerModuleIndices[3 * tripletIndex];
    unsigned int middleModuleIndex = tripletsInGPU.lowerModuleIndices[3 * tripletIndex + 1];
    unsigned int upperModuleIndex = tripletsInGPU.lowerModuleIndices[3 * tripletIndex + 2];


    // pixel segment vs inner segment of the triplet
    pass = pass & runTrackletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, pixelModuleIndex, pixelModuleIndex, lowerModuleIndex, middleModuleIndex, pixelSegmentIndex, tripletsInGPU.segmentIndices[2 * tripletIndex], zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut);

    //pixel segment vs outer segment of triplet
    pass = pass & runTrackletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, pixelModuleIndex, pixelModuleIndex, middleModuleIndex, upperModuleIndex, pixelSegmentIndex, tripletsInGPU.segmentIndices[2 * tripletIndex + 1], zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut);


    //pt matching between the pixel ptin and the triplet circle pt
    float pixelSegmentPt = segmentsInGPU.ptIn[pixelSegmentIndex];
    float pixelSegmentPtError = segmentsInGPU.ptErr[pixelSegmentIndex];

    pixelRadius = pixelSegmentPt/(2 * k2Rinv1GeVf);
    pixelRadiusError = pixelSegmentPtError/(2 * k2Rinv1GeVf);
    
    unsigned int tripletInnerSegmentIndex = tripletsInGPU.segmentIndices[2 * tripletIndex];
    unsigned int tripletOuterSegmentIndex = tripletsInGPU.segmentIndices[2 * tripletIndex + 1];

    unsigned int innerMDAnchorHitIndex = segmentsInGPU.innerMiniDoubletAnchorHitIndices[2 * tripletInnerSegmentIndex];
    unsigned int middleMDAnchorHitIndex = segmentsInGPU.outerMiniDoubletAnchorHitIndices[2 * tripletInnerSegmentIndex + 1];
    unsigned int outerMDAnchorHitIndex = segmentsInGPU.outerMiniDoubletAnchorHitIndices[2 * tripletOuterSegmentIndex + 1];

    float x1 = hitsInGPU.xs[innerMDAnchorHitIndex];
    float x2 = hitsInGPU.xs[middleMDAnchorHitIndex];
    float x3 = hitsInGPU.xs[outerMDAnchorHitIndex];

    float y1 = hitsInGPU.ys[innerMDAnchorHitIndex];
    float y2 = hitsInGPU.ys[middleMDAnchorHitIndex];
    float y3 = hitsInGPU.ys[outerMDAnchorHitIndex];
    
    float g, f;
    tripletRadius = computeRadiusFromThreeAnchorHits(x1, y1, x2, y2, x3, y3, g, f);
    
    pass = pass & passRadiusCriterion(modulesInGPU, pixelRadius, pixelRadiusError, tripletRadius, lowerModuleIndex, middleModuleIndex, upperModuleIndex);

    return pass;
}


__device__ bool SDL::passRadiusCriterion(struct modules& modulesInGPU, float& pixelRadius, float& pixelRadiusError, float& tripletRadius, unsigned int lowerModuleIndex, unsigned int middleModuleIndex, unsigned int upperModuleIndex)
{
    if(modulesInGPU.subdets[lowerModuleIndex] == SDL::Endcap)
    {
        return passRadiusCriterionEEE(pixelRadius, pixelRadiusError, tripletRadius);
    }
    else if(modulesInGPU.subdets[middleModuleIndex] == SDL::Endcap)
    {
        return passRadiusCriterionBEE(pixelRadius, pixelRadiusError, tripletRadius);
    }
    else if(modulesInGPU.subdets[upperModuleIndex] == SDL::Endcap)
    {
        return passRadiusCriterionBBE(pixelRadius, pixelRadiusError, tripletRadius);
    }
    else
    {
        return passRadiusCriterionBBB(pixelRadius, pixelRadiusError, tripletRadius);
    }

    //return ((modulesInGPU.subdets[lowerModuleIndex] == SDL::Endcap) & (passRadiusCriterionEEE(pixelRadius, pixelRadiusError, tripletRadius))) | ((modulesInGPU.subdets[middleModuleIndex] == SDL::Endcap) & (passRadiusCriterionBEE(pixelRadius, pixelRadiusError, tripletRadius))) | ((modulesInGPU.subdets[upperModuleIndex] == SDL::Endcap) & (passRadiusCriterionBBE(pixelRadius, pixelRadiusError, tripletRadius))) |  (passRadiusCriterionBBB(pixelRadius, pixelRadiusError, tripletRadius));

}
__device__ bool SDL::passRadiusCriterionBBB(float& pixelRadius, float& pixelRadiusError, float& tripletRadius)
{
    float tripletInvRadiusErrorBound = 0;
    float tripletRadiusMin = tripletRadius/(1 + tripletInvRadiusErrorBound);
    float tripletRadiusMax = tripletRadius/(1 - tripletInvRadiusErrorBound);
    float pixelRadiusMin = pixelRadius - pixelRadiusError;
    float pixelRadiusMax = pixelRadius + pixelRadiusError;
    
    return true; //checkIntervalOverlap(1.0/tripletRadiusMax, 1.0/tripletRadiusMin, 1.0/pixelRadusMax, 1.0/pixelRadiusMin);
}

__device__ bool SDL::passRadiusCriterionBBE(float& pixelRadius, float& pixelRadiusError, float& tripletRadius)
{
    float tripletInvRadiusErrorBound = 0;
    float tripletRadiusMin = tripletRadius/(1 + tripletInvRadiusErrorBound);
    float tripletRadiusMax = tripletRadius/(1 - tripletInvRadiusErrorBound);
    float pixelRadiusMin = pixelRadius - pixelRadiusError;
    float pixelRadiusMax = pixelRadius + pixelRadiusError;

    return true; //checkIntervalOverlap(1.0/tripletRadiusMax, 1.0/tripletRadiusMin, 1.0/pixelRadusMax, 1.0/pixelRadiusMin);

}

__device__ bool SDL::passRadiusCriterionBEE(float& pixelRadius, float& pixelRadiusError, float& tripletRadius)
{
    float tripletInvRadiusErrorBound = 0;
    float tripletRadiusMin = tripletRadius/(1 + tripletInvRadiusErrorBound);
    float tripletRadiusMax = tripletRadius/(1 - tripletInvRadiusErrorBound);
    float pixelRadiusMin = pixelRadius - pixelRadiusError;
    float pixelRadiusMax = pixelRadius + pixelRadiusError;

    return true; //checkIntervalOverlap(1.0/tripletRadiusMax, 1.0/tripletRadiusMin, 1.0/pixelRadusMax, 1.0/pixelRadiusMin);

}


__device__ bool SDL::passRadiusCriterionEEE(float& pixelRadius, float& pixelRadiusError, float& tripletRadius)
{
    float tripletInvRadiusErrorBound = 0;
    float tripletRadiusMax = tripletRadius/(1 - tripletInvRadiusErrorBound);
    float pixelRadiusMin = pixelRadius - pixelRadiusError;
    float pixelRadiusMax = pixelRadius + pixelRadiusError;

    return true; //checkIntervalOverlap(1.0/tripletRadiusMax, 1.0/tripletRadiusMin, 1.0/pixelRadusMax, 1.0/pixelRadiusMin);

}
