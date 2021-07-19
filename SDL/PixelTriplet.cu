# include "PixelTriplet.cuh"
#include "allocate.h"

SDL::pixelTriplets::pixelTriplets()
{
    pixelSegmentIndices = nullptr;
    tripletIndices = nullptr;
    nPixelTriplets = nullptr;
    pixelRadius = nullptr;
    tripletRadius = nullptr;
#ifdef CUT_VALUE_DEBUG
    pixelRadiusError = nullptr;
#endif
}

void SDL::pixelTriplets::freeMemory()
{
    cudaFree(pixelSegmentIndices);
    cudaFree(tripletIndices);
    cudaFree(nPixelTriplets);
    cudaFree(pixelRadius);
    cudaFree(tripletRadius);
#ifdef CUT_VALUE_DEBUG
    cudaFree(pixelRadiusError);
#endif
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
    cudaMallocManaged(&pixelTripletsInGPU.tripletRadius, maxPixelTriplets * sizeof(float));
#ifdef CUT_VALUE_DEBUG
    cudaMallocManaged(&pixelTripletsInGPU.pixelRadiusError, maxPixelTriplets * sizeof(float));
#endif
    cudaMemset(pixelTripletsInGPU.nPixelTriplets, 0, sizeof(unsigned int));
}

void SDL::createPixelTripletsInExplicitMemory(struct pixelTriplets& pixelTripletsInGPU, unsigned int maxPixelTriplets)
{
    cudaMalloc(&pixelTripletsInGPU.pixelSegmentIndices, maxPixelTriplets * sizeof(float));
    cudaMalloc(&pixelTripletsInGPU.tripletIndices, maxPixelTriplets * sizeof(float));
    cudaMalloc(&pixelTripletsInGPU.nPixelTriplets, sizeof(unsigned int));
    cudaMalloc(&pixelTripletsInGPU.pixelRadius, maxPixelTriplets * sizeof(float));
    cudaMalloc(&pixelTripletsInGPU.tripletRadius, maxPixelTriplets * sizeof(float));

    cudaMemset(pixelTripletsInGPU.nPixelTriplets, 0, sizeof(unsigned int));

}

#ifdef CUT_VALUE_DEBUG
__device__ void SDL::addPixelTripletToMemory(struct pixelTriplets& pixelTripletsInGPU, unsigned int pixelSegmentIndex, unsigned int tripletIndex, float pixelRadius, float pixelRadiusError, float tripletRadius, unsigned int pixelTripletIndex)
#else
__device__ void SDL::addPixelTripletToMemory(struct pixelTriplets& pixelTripletsInGPU, unsigned int pixelSegmentIndex, unsigned int tripletIndex, float pixelRadius, float tripletRadius, unsigned int pixelTripletIndex)
#endif
{
    pixelTripletsInGPU.pixelSegmentIndices[pixelTripletIndex] = pixelSegmentIndex;
    pixelTripletsInGPU.tripletIndices[pixelTripletIndex] = tripletIndex;
    pixelTripletsInGPU.pixelRadius[pixelTripletIndex] = pixelRadius;
    pixelTripletsInGPU.tripletRadius[pixelTripletIndex] = tripletRadius;

#ifdef CUT_VALUE_DEBUG
    pixelTripletsInGPU.pixelRadiusError[pixelTripletIndex] = pixelRadiusError;
#endif
}

__device__ bool SDL::runPixelTripletDefaultAlgo(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct triplets& tripletsInGPU, unsigned int& pixelSegmentIndex, unsigned int tripletIndex, float& pixelRadius, float& pixelRadiusError, float& tripletRadius)
{
    bool pass = true;

    //run pT4 compatibility between the pixel segment and inner segment, and between the pixel and outer segment of the triplet


    //placeholder
    float zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, pt_beta; //temp stuff
    float zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ;

    unsigned int pixelModuleIndex = segmentsInGPU.innerLowerModuleIndices[pixelSegmentIndex];

    unsigned int lowerModuleIndex = tripletsInGPU.lowerModuleIndices[3 * tripletIndex];
    unsigned int middleModuleIndex = tripletsInGPU.lowerModuleIndices[3 * tripletIndex + 1];
    unsigned int upperModuleIndex = tripletsInGPU.lowerModuleIndices[3 * tripletIndex + 2];


    // pixel segment vs inner segment of the triplet
    pass = pass & runPixelTrackletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, pixelModuleIndex, pixelModuleIndex, lowerModuleIndex, middleModuleIndex, pixelSegmentIndex, tripletsInGPU.segmentIndices[2 * tripletIndex], zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, pt_beta, zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ);

    //pixel segment vs outer segment of triplet
    pass = pass & runPixelTrackletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, pixelModuleIndex, pixelModuleIndex, middleModuleIndex, upperModuleIndex, pixelSegmentIndex, tripletsInGPU.segmentIndices[2 * tripletIndex + 1], zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, pt_beta, zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ);

    //pt matching between the pixel ptin and the triplet circle pt
    unsigned int pixelSegmentArrayIndex = pixelSegmentIndex - (pixelModuleIndex * 600);
    float pixelSegmentPt = segmentsInGPU.ptIn[pixelSegmentArrayIndex];
    float pixelSegmentPtError = segmentsInGPU.ptErr[pixelSegmentArrayIndex];

    pixelRadius = pixelSegmentPt/(2 * k2Rinv1GeVf);
    pixelRadiusError = pixelSegmentPtError/(2 * k2Rinv1GeVf);
    unsigned int tripletInnerSegmentIndex = tripletsInGPU.segmentIndices[2 * tripletIndex];
    unsigned int tripletOuterSegmentIndex = tripletsInGPU.segmentIndices[2 * tripletIndex + 1];

    unsigned int innerMDAnchorHitIndex = segmentsInGPU.innerMiniDoubletAnchorHitIndices[tripletInnerSegmentIndex];
    unsigned int middleMDAnchorHitIndex = segmentsInGPU.outerMiniDoubletAnchorHitIndices[tripletInnerSegmentIndex];
    unsigned int outerMDAnchorHitIndex = segmentsInGPU.outerMiniDoubletAnchorHitIndices[tripletOuterSegmentIndex];

    float x1 = hitsInGPU.xs[innerMDAnchorHitIndex];
    float x2 = hitsInGPU.xs[middleMDAnchorHitIndex];
    float x3 = hitsInGPU.xs[outerMDAnchorHitIndex];

    float y1 = hitsInGPU.ys[innerMDAnchorHitIndex];
    float y2 = hitsInGPU.ys[middleMDAnchorHitIndex];
    float y3 = hitsInGPU.ys[outerMDAnchorHitIndex];
    float g,f;
    
    tripletRadius = computeRadiusFromThreeAnchorHits(x1, y1, x2, y2, x3, y3,g,f);
    
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

/*bounds for high Pt taken from : http://uaf-10.t2.ucsd.edu/~bsathian/SDL/T5_efficiency/efficiencies/new_efficiencies/efficiencies_20210513_T5_recovering_high_Pt_efficiencies/highE_radius_matching/highE_bounds.txt */
__device__ bool SDL::passRadiusCriterionBBB(float& pixelRadius, float& pixelRadiusError, float& tripletRadius)
{
    float tripletInvRadiusErrorBound = 0.15624;
    float pixelInvRadiusErrorBound = 0.17235;

    if(pixelRadius > 2.0/(2 * k2Rinv1GeVf))
    {
        pixelInvRadiusErrorBound = 0.6375;
        tripletInvRadiusErrorBound = 0.6588;
    }

    float tripletRadiusInvMax = (1 + tripletInvRadiusErrorBound)/tripletRadius;
    float tripletRadiusInvMin = fmaxf((1 - tripletInvRadiusErrorBound)/tripletRadius, 0);

    float pixelRadiusInvMax = fmaxf((1 + pixelInvRadiusErrorBound)/pixelRadius, 1.f/(pixelRadius - pixelRadiusError));
    float pixelRadiusInvMin = fminf((1 - pixelInvRadiusErrorBound)/pixelRadius, 1.f/(pixelRadius + pixelRadiusError));

    return checkIntervalOverlap(tripletRadiusInvMin, tripletRadiusInvMax, pixelRadiusInvMin, pixelRadiusInvMax);
}

__device__ bool SDL::passRadiusCriterionBBE(float& pixelRadius, float& pixelRadiusError, float& tripletRadius)
{
    float tripletInvRadiusErrorBound = 0.45972;
    float pixelInvRadiusErrorBound = 0.19644;

    if(pixelRadius > 2.0/(2 * k2Rinv1GeVf))
    {
        pixelInvRadiusErrorBound = 0.6805;
        tripletInvRadiusErrorBound = 0.8557;
    }

    float tripletRadiusInvMax = (1 + tripletInvRadiusErrorBound)/tripletRadius;
    float tripletRadiusInvMin = fmaxf((1 - tripletInvRadiusErrorBound)/tripletRadius, 0);

    float pixelRadiusInvMax = fmaxf((1 + pixelInvRadiusErrorBound)/pixelRadius, 1.f/(pixelRadius - pixelRadiusError));
    float pixelRadiusInvMin = fminf((1 - pixelInvRadiusErrorBound)/pixelRadius, 1.f/(pixelRadius + pixelRadiusError));

    return checkIntervalOverlap(tripletRadiusInvMin, tripletRadiusInvMax, pixelRadiusInvMin, pixelRadiusInvMax);

}

__device__ bool SDL::passRadiusCriterionBEE(float& pixelRadius, float& pixelRadiusError, float& tripletRadius)
{
    float tripletInvRadiusErrorBound = 1.59294;
    float pixelInvRadiusErrorBound = 0.255181;

    if(pixelRadius > 2.0/(2 * k2Rinv1GeVf)) //as good as not having selections
    {
        pixelInvRadiusErrorBound = 2.2091;
        tripletInvRadiusErrorBound = 2.3548;
    }

    float tripletRadiusInvMax = (1 + tripletInvRadiusErrorBound)/tripletRadius;
    float tripletRadiusInvMin = fmaxf((1 - tripletInvRadiusErrorBound)/tripletRadius, 0);

    float pixelRadiusInvMax = fmaxf((1 + pixelInvRadiusErrorBound)/pixelRadius, 1.f/(pixelRadius - pixelRadiusError));
    float pixelRadiusInvMin = fminf((1 - pixelInvRadiusErrorBound)/pixelRadius, 1.f/(pixelRadius + pixelRadiusError));
    pixelRadiusInvMin = fmaxf(pixelRadiusInvMin, 0);

    return checkIntervalOverlap(tripletRadiusInvMin, tripletRadiusInvMax, pixelRadiusInvMin, pixelRadiusInvMax);

}

__device__ bool SDL::passRadiusCriterionEEE(float& pixelRadius, float& pixelRadiusError, float& tripletRadius)
{
    float tripletInvRadiusErrorBound = 1.7006;
    float pixelInvRadiusErrorBound = 0.26367;

    if(pixelRadius > 2.0/(2 * k2Rinv1GeVf)) //as good as not having selections
    {
        pixelInvRadiusErrorBound = 2.286;
        tripletInvRadiusErrorBound = 2.436;
    }

    float tripletRadiusInvMax = (1 + tripletInvRadiusErrorBound)/tripletRadius;
    float tripletRadiusInvMin = fmaxf((1 - tripletInvRadiusErrorBound)/tripletRadius, 0);

    float pixelRadiusInvMax = fmaxf((1 + pixelInvRadiusErrorBound)/pixelRadius, 1.f/(pixelRadius - pixelRadiusError));
    float pixelRadiusInvMin = fminf((1 - pixelInvRadiusErrorBound)/pixelRadius, 1.f/(pixelRadius + pixelRadiusError));
    pixelRadiusInvMin = fmaxf(0, pixelRadiusInvMin);

    return checkIntervalOverlap(tripletRadiusInvMin, tripletRadiusInvMax, pixelRadiusInvMin, pixelRadiusInvMax);

}
