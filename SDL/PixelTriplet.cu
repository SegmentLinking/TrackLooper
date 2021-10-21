# include "PixelTriplet.cuh"
# include "PixelTracklet.cuh"
#include "allocate.h"

SDL::pixelTriplets::pixelTriplets()
{
    pixelSegmentIndices = nullptr;
    tripletIndices = nullptr;
    nPixelTriplets = nullptr;
    pixelRadius = nullptr;
    tripletRadius = nullptr;
    pt = nullptr;
    isDup = nullptr;
    partOfPT5 = nullptr;
#ifdef CUT_VALUE_DEBUG
    pixelRadiusError = nullptr;
    rzChiSquared = nullptr;
    rPhiChiSquared = nullptr;
    rPhiChiSquaredInwards = nullptr;
#endif
}

void SDL::pixelTriplets::freeMemory()
{
    cudaFree(pixelSegmentIndices);
    cudaFree(tripletIndices);
    cudaFree(nPixelTriplets);
    cudaFree(pixelRadius);
    cudaFree(tripletRadius);
    cudaFree(pt);
    cudaFree(isDup);
    cudaFree(partOfPT5);
#ifdef CUT_VALUE_DEBUG
    cudaFree(pixelRadiusError);
    cudaFree(rPhiChiSquared);
    cudaFree(rPhiChiSquaredInwards);
    cudaFree(rzChiSquared);
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
    cudaMallocManaged(&pixelTripletsInGPU.pt, maxPixelTriplets * 6*sizeof(float));
    cudaMallocManaged(&pixelTripletsInGPU.isDup, maxPixelTriplets * sizeof(bool));
    cudaMallocManaged(&pixelTripletsInGPU.partOfPT5, maxPixelTriplets * sizeof(bool));
#ifdef CUT_VALUE_DEBUG
    cudaMallocManaged(&pixelTripletsInGPU.pixelRadiusError, maxPixelTriplets * sizeof(float));
    cudaMallocManaged(&pixelTripletsInGPU.rPhiChiSquared, maxPixelTriplets * sizeof(float));
    cudaMallocManaged(&pixelTripletsInGPU.rPhiChiSquaredInwards, maxPixelTriplets * sizeof(float));
    cudaMallocManaged(&pixelTripletsInGPU.rzChiSquared, maxPixelTriplets * sizeof(float));
#endif
    pixelTripletsInGPU.eta = pixelTripletsInGPU.pt + maxPixelTriplets;
    pixelTripletsInGPU.phi = pixelTripletsInGPU.pt + maxPixelTriplets * 2;
    pixelTripletsInGPU.eta_pix = pixelTripletsInGPU.pt + maxPixelTriplets *3;
    pixelTripletsInGPU.phi_pix = pixelTripletsInGPU.pt + maxPixelTriplets * 4;
    pixelTripletsInGPU.score = pixelTripletsInGPU.pt + maxPixelTriplets * 5;
    cudaMemset(pixelTripletsInGPU.nPixelTriplets, 0, sizeof(unsigned int));
}

void SDL::createPixelTripletsInExplicitMemory(struct pixelTriplets& pixelTripletsInGPU, unsigned int maxPixelTriplets)
{
    cudaMalloc(&pixelTripletsInGPU.pixelSegmentIndices, maxPixelTriplets * sizeof(float));
    cudaMalloc(&pixelTripletsInGPU.tripletIndices, maxPixelTriplets * sizeof(float));
    cudaMalloc(&pixelTripletsInGPU.nPixelTriplets, sizeof(unsigned int));
    cudaMalloc(&pixelTripletsInGPU.pixelRadius, maxPixelTriplets * sizeof(float));
    cudaMalloc(&pixelTripletsInGPU.tripletRadius, maxPixelTriplets * sizeof(float));
    cudaMalloc(&pixelTripletsInGPU.pt, maxPixelTriplets * 6*sizeof(float));
    cudaMalloc(&pixelTripletsInGPU.isDup, maxPixelTriplets * sizeof(bool));
    cudaMalloc(&pixelTripletsInGPU.partOfPT5, maxPixelTriplets * sizeof(bool));

    pixelTripletsInGPU.eta = pixelTripletsInGPU.pt + maxPixelTriplets;
    pixelTripletsInGPU.phi = pixelTripletsInGPU.pt + maxPixelTriplets * 2;
    pixelTripletsInGPU.eta_pix = pixelTripletsInGPU.pt + maxPixelTriplets *3;
    pixelTripletsInGPU.phi_pix = pixelTripletsInGPU.pt + maxPixelTriplets * 4;
    pixelTripletsInGPU.score = pixelTripletsInGPU.pt + maxPixelTriplets * 5;
    cudaMemset(pixelTripletsInGPU.nPixelTriplets, 0, sizeof(unsigned int));

}

#ifdef CUT_VALUE_DEBUG
__device__ void SDL::addPixelTripletToMemory(struct pixelTriplets& pixelTripletsInGPU, unsigned int pixelSegmentIndex, unsigned int tripletIndex, float pixelRadius, float pixelRadiusError, float tripletRadius, float rPhiChiSquared, float rPhiChiSquaredInwards, float rzChiSquared, unsigned int pixelTripletIndex, float pt, float eta, float phi, float eta_pix, float phi_pix, float score)
#else
__device__ void SDL::addPixelTripletToMemory(struct pixelTriplets& pixelTripletsInGPU, unsigned int pixelSegmentIndex, unsigned int tripletIndex, float pixelRadius, float tripletRadius, unsigned int pixelTripletIndex, float pt, float eta, float phi, float eta_pix, float phi_pix,float score)
#endif
{
    pixelTripletsInGPU.pixelSegmentIndices[pixelTripletIndex] = pixelSegmentIndex;
    pixelTripletsInGPU.tripletIndices[pixelTripletIndex] = tripletIndex;
    pixelTripletsInGPU.pixelRadius[pixelTripletIndex] = pixelRadius;
    pixelTripletsInGPU.tripletRadius[pixelTripletIndex] = tripletRadius;
    pixelTripletsInGPU.pt[pixelTripletIndex] = pt;
    pixelTripletsInGPU.eta[pixelTripletIndex] = eta;
    pixelTripletsInGPU.phi[pixelTripletIndex] = phi;
    pixelTripletsInGPU.eta_pix[pixelTripletIndex] = eta_pix;
    pixelTripletsInGPU.phi_pix[pixelTripletIndex] = phi_pix;
    pixelTripletsInGPU.isDup[pixelTripletIndex] = 0;
    pixelTripletsInGPU.score[pixelTripletIndex] = score;

#ifdef CUT_VALUE_DEBUG
    pixelTripletsInGPU.pixelRadiusError[pixelTripletIndex] = pixelRadiusError;
    pixelTripletsInGPU.rPhiChiSquared[pixelTripletIndex] = rPhiChiSquared;
    pixelTripletsInGPU.rPhiChiSquaredInwards[pixelTripletIndex] = rPhiChiSquaredInwards;
    pixelTripletsInGPU.rzChiSquared[pixelTripletIndex] = rzChiSquared;
#endif
}
__device__ void SDL::rmPixelTripletToMemory(struct pixelTriplets& pixelTripletsInGPU,unsigned int pixelTripletIndex)
{
    pixelTripletsInGPU.isDup[pixelTripletIndex] = 1;
}

__device__ float SDL::computeRadiusFromThreeAnchorHitspT3(float x1, float y1, float x2, float y2, float x3, float y3, float& g, float& f)
{
    float radius = 0;

    //writing manual code for computing radius, which obviously sucks
    //TODO:Use fancy inbuilt libraries like cuBLAS or cuSOLVE for this!
    //(g,f) -> center
    //first anchor hit - (x1,y1), second anchor hit - (x2,y2), third anchor hit - (x3, y3)

    /*
    if((y1 - y3) * (x2 - x3) - (x1 - x3) * (y2 - y3) == 0)
    {
        return -1; //WTF man three collinear points!
    }
    */

    float denomInv = 1.0/((y1 - y3) * (x2 - x3) - (x1 - x3) * (y2 - y3));

    float xy1sqr = x1 * x1 + y1 * y1;

    float xy2sqr = x2 * x2 + y2 * y2;

    float xy3sqr = x3 * x3 + y3 * y3;

    g = 0.5 * ((y3 - y2) * xy1sqr + (y1 - y3) * xy2sqr + (y2 - y1) * xy3sqr) * denomInv;

    f = 0.5 * ((x2 - x3) * xy1sqr + (x3 - x1) * xy2sqr + (x1 - x2) * xy3sqr) * denomInv;

    float c = ((x2 * y3 - x3 * y2) * xy1sqr + (x3 * y1 - x1 * y3) * xy2sqr + (x1 * y2 - x2 * y1) * xy3sqr) * denomInv;

    if(((y1 - y3) * (x2 - x3) - (x1 - x3) * (y2 - y3) == 0) || (g * g + f * f - c < 0))
    {
        printf("three collinear points or FATAL! r^2 < 0!\n");
  radius = -1;
    }
    else
      radius = sqrtf(g * g  + f * f - c);

    return radius;
}

__device__ bool SDL::runPixelTripletDefaultAlgo(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct triplets& tripletsInGPU, unsigned int& pixelSegmentIndex, unsigned int tripletIndex, float& pixelRadius, float& pixelRadiusError, float& tripletRadius, float& rzChiSquared, float& rPhiChiSquared, float& rPhiChiSquaredInwards, bool runChiSquaredCuts)
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
    pass = pass & runPixelTrackletDefaultAlgo_pT3(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, pixelModuleIndex, pixelModuleIndex, lowerModuleIndex, middleModuleIndex, pixelSegmentIndex, tripletsInGPU.segmentIndices[2 * tripletIndex], zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, pt_beta, zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ);

    //pixel segment vs outer segment of triplet
    pass = pass & runPixelTrackletDefaultAlgo_pT3(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, pixelModuleIndex, pixelModuleIndex, middleModuleIndex, upperModuleIndex, pixelSegmentIndex, tripletsInGPU.segmentIndices[2 * tripletIndex + 1], zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, pt_beta, zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ);

    //pt matching between the pixel ptin and the triplet circle pt
    unsigned int pixelSegmentArrayIndex = pixelSegmentIndex - (pixelModuleIndex * 600);
    float pixelSegmentPt = segmentsInGPU.ptIn[pixelSegmentArrayIndex];
    float pixelSegmentPtError = segmentsInGPU.ptErr[pixelSegmentArrayIndex];

    unsigned int pixelInnerMDIndex = segmentsInGPU.mdIndices[2 * pixelSegmentIndex];
    unsigned int pixelOuterMDIndex = segmentsInGPU.mdIndices[2 * pixelSegmentIndex + 1];

    unsigned int pixelAnchorHitIndex1 = mdsInGPU.hitIndices[2 * pixelInnerMDIndex];
    unsigned int pixelNonAnchorHitIndex1 = mdsInGPU.hitIndices[2 * pixelInnerMDIndex + 1];
    unsigned int pixelAnchorHitIndex2 = mdsInGPU.hitIndices[2 * pixelOuterMDIndex];
    unsigned int pixelNonAnchorHitIndex2 = mdsInGPU.hitIndices[2 * pixelOuterMDIndex + 1];

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
    
    tripletRadius = computeRadiusFromThreeAnchorHitspT3(x1, y1, x2, y2, x3, y3,g,f);
    
    pass = pass & passRadiusCriterion(modulesInGPU, pixelRadius, pixelRadiusError, tripletRadius, lowerModuleIndex, middleModuleIndex, upperModuleIndex);

    unsigned int anchorHits[] = {innerMDAnchorHitIndex, middleMDAnchorHitIndex, outerMDAnchorHitIndex};
    unsigned int pixelAnchorHits[] = {pixelAnchorHitIndex1, pixelAnchorHitIndex2};
    unsigned int lowerModuleIndices[] = {lowerModuleIndex, middleModuleIndex, upperModuleIndex};

    rzChiSquared = computePT3RZChiSquared(modulesInGPU, hitsInGPU, segmentsInGPU, pixelAnchorHitIndex1, pixelAnchorHitIndex2, anchorHits, lowerModuleIndices);

    rPhiChiSquared = computePT3RPhiChiSquared(modulesInGPU, hitsInGPU, segmentsInGPU, pixelSegmentArrayIndex, anchorHits, lowerModuleIndices);

    rPhiChiSquaredInwards = computePT3RPhiChiSquaredInwards(modulesInGPU, hitsInGPU, tripletRadius, g, f, pixelAnchorHits);

    if(runChiSquaredCuts and pixelSegmentPt < 5.0)
    {
        pass = pass & passPT3RZChiSquaredCuts(modulesInGPU, lowerModuleIndex, middleModuleIndex, upperModuleIndex, rzChiSquared);
        pass = pass & passPT3RPhiChiSquaredCuts(modulesInGPU, lowerModuleIndex, middleModuleIndex, upperModuleIndex, rPhiChiSquared);

        pass = pass & passPT3RPhiChiSquaredInwardsCuts(modulesInGPU, lowerModuleIndex, middleModuleIndex, upperModuleIndex, rPhiChiSquaredInwards);
    }


    return pass;

}

__device__ bool SDL::passPT3RPhiChiSquaredInwardsCuts(struct modules& modulesInGPU, unsigned int lowerModuleIndex1, unsigned int lowerModuleIndex2, unsigned int lowerModuleIndex3, float& chiSquared)
{
    const int layer1 = modulesInGPU.layers[lowerModuleIndex1] + 6 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex1] == SDL::TwoS);
    const int layer2 = modulesInGPU.layers[lowerModuleIndex2] + 6 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex2] == SDL::TwoS);
    const int layer3 = modulesInGPU.layers[lowerModuleIndex3] + 6 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex3] == SDL::TwoS);
    
    if(layer1 == 7 and layer2 == 8 and layer3 == 9)
    {
        return chiSquared < 22016.8055;
    }
    else if(layer1 == 7 and layer2 == 8 and layer3 == 14)
    {
        return chiSquared < 935179.56807;
    }
    else if(layer1 == 8 and layer2 == 9 and layer3 == 10)
    {
        return chiSquared < 29064.12959;
    }
    else if(layer1 == 8 and layer2 == 9 and layer3 == 15)
    {
        return chiSquared < 935179.5681;
    }
    else if(layer1 == 1 and layer2 == 2 and layer3 == 3)
    {
        return chiSquared < 1370.0113195101474;
    }
    else if(layer1 == 1 and layer2 == 2 and layer3 == 7)
    {
        return chiSquared < 5492.110048314815;
    }
    else if(layer1 == 2 and layer2 == 3 and layer3 == 4)
    {
        return chiSquared < 4160.410806470067;
    }
    else if(layer1 == 1 and layer2 == 7 and layer3 == 8)
    {
        return chiSquared < 29064.129591225726;
    }
    else if(layer1 == 2 and layer2 == 3 and layer3 == 7)
    {
        return chiSquared < 12634.215376250893;
    }
    else if(layer1 == 2 and layer2 == 3 and layer3 == 12)
    {
        return chiSquared < 353821.69361145404;
    }
    else if(layer1 == 2 and layer2 == 7 and layer3 == 8)
    {
        return chiSquared < 33393.26076341235;
    }
    else if(layer1 == 2 and layer2 == 7 and layer3 == 13)
    {
        return chiSquared < 935179.5680742573;
    }

    return true;
}

__device__ float SDL::computePT3RPhiChiSquaredInwards(struct modules& modulesInGPU, struct hits& hitsInGPU, float& r, float& g, float& f, unsigned int* pixelAnchorHits)
{
    float x,y;
    float chiSquared = 0;
    for(size_t i = 0; i < 2; i++)
    {
        x = hitsInGPU.xs[pixelAnchorHits[i]];
        y = hitsInGPU.ys[pixelAnchorHits[i]];
        float residual = (x - g) * (x -g) + (y - f) * (y - f) - r * r;
        chiSquared += residual * residual;
    }
    chiSquared /= 2;
    return chiSquared;
}

__device__ bool SDL::passPT3RZChiSquaredCuts(struct modules& modulesInGPU, unsigned int lowerModuleIndex1, unsigned int lowerModuleIndex2, unsigned int lowerModuleIndex3, float& rzChiSquared)
{
    const int layer1 = modulesInGPU.layers[lowerModuleIndex1] + 6 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex1] == SDL::TwoS);
    const int layer2 = modulesInGPU.layers[lowerModuleIndex2] + 6 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex2] == SDL::TwoS);
    const int layer3 = modulesInGPU.layers[lowerModuleIndex3] + 6 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex3] == SDL::TwoS);

    if(layer1 == 8 and layer2 == 9 and layer3 == 10)
    {
        return rzChiSquared < 85.2499;
    }
    else if(layer1 == 8 and layer2 == 9 and layer3 == 15)
    {
        return rzChiSquared < 85.2499;
    }

    else if(layer1 == 7 and layer2 == 8 and layer3 == 9)
    {
        return rzChiSquared < 74.19805;
    }
    else if(layer1 == 7 and layer2 == 8 and layer3 == 14)
    {
        return rzChiSquared < 97.9479;
    }

    else if(layer1 == 1 and layer2 == 2 and layer3 == 3)
    {
        return rzChiSquared < 451.1407;;
    }
    else if(layer1 == 1 and layer2 == 2 and layer3 == 7)
    {
        return rzChiSquared < 595.546;
    }

    else if(layer1 == 1 and layer2 == 7 and layer3 == 8)
    {
        return rzChiSquared < 518.339;
    }

    else if(layer1 == 2 and layer2 == 3 and layer3 == 7)
    {
        return rzChiSquared < 684.253;
    }
    else if(layer1 == 2 and layer2 == 3 and layer3 == 12)
    {
        return rzChiSquared < 684.253;
    }
    else if(layer1 == 2 and layer2 == 3 and layer3 == 4)
    {
        return rzChiSquared  < 392.654;
    }

    else if(layer1 == 2 and layer2 == 7 and layer3 == 8)
    {
        return rzChiSquared < 518.339;
    }
    else if(layer1 == 2 and layer2 == 7 and layer3 == 13)
    {
        return rzChiSquared < 518.339;
    }

    //default - category not found!
    return true;
}

__device__ float SDL::computePT3RZChiSquared(struct modules& modulesInGPU, struct hits& hitsInGPU, struct segments& segmentsInGPU, unsigned int& pixelAnchorHitIndex1, unsigned int& pixelAnchorHitIndex2, unsigned int* anchorHits, unsigned int* lowerModuleIndices)
{
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
    for(size_t i = 0; i < 3; i++)
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

    RMSE = sqrtf(0.2 * RMSE); //the constant doesn't really matter....
    return RMSE;
}

//TODO: merge this one and the pT5 function later into a single function
__device__ float SDL::computePT3RPhiChiSquared(struct modules& modulesInGPU, struct hits& hitsInGPU, struct segments& segmentsInGPU, unsigned int pixelSegmentArrayIndex, unsigned int* anchorHits, unsigned int* lowerModuleIndices)
{
    float g = segmentsInGPU.circleCenterX[pixelSegmentArrayIndex];
    float f = segmentsInGPU.circleCenterY[pixelSegmentArrayIndex];
    float radius = segmentsInGPU.circleRadius[pixelSegmentArrayIndex];
    float delta1[3], delta2[3], slopes[3];
    bool isFlat[3];
    float xs[3];
    float ys[3];
    float chiSquared = 0;
    ModuleType moduleType;
    short moduleSubdet, moduleSide;
    ModuleLayerType moduleLayerType;
    float drdz;
    float inv1 = 0.01/0.009;
    float inv2 = 0.15/0.009;
    float inv3 = 2.4/0.009;
    for(size_t i = 0; i < 3; i++)
    {
        xs[i] = hitsInGPU.xs[anchorHits[i]];
        ys[i] = hitsInGPU.ys[anchorHits[i]];
    //}

    ////computeSigmasForRegression(modulesInGPU, lowerModuleIndices, delta1, delta2, slopes, isFlat, 3);
    //for(size_t i=0; i<3; i++)
    //{
        moduleType = modulesInGPU.moduleType[lowerModuleIndices[i]];
        moduleSubdet = modulesInGPU.subdets[lowerModuleIndices[i]];
        moduleSide = modulesInGPU.sides[lowerModuleIndices[i]];
        moduleLayerType = modulesInGPU.moduleLayerType[lowerModuleIndices[i]];
        //category 1 - barrel PS flat
        if(moduleSubdet == Barrel and moduleType == PS and moduleSide == Center)
        {
            delta1[i] = inv1;//1.1111f;//0.01;
            delta2[i] = inv1;//1.1111f;//0.01;
            slopes[i] = -999;
            isFlat[i] = true;
        }

        //category 2 - barrel 2S
        else if(moduleSubdet == Barrel and moduleType == TwoS)
        {
            delta1[i] = 1;//0.009;
            delta2[i] = 1;//0.009;
            slopes[i] = -999;
            isFlat[i] = true;
        }

        //category 3 - barrel PS tilted
        else if(moduleSubdet == Barrel and moduleType == PS and moduleSide != Center)
        {

            //get drdz
            if(moduleLayerType == Strip)
            {
                drdz = modulesInGPU.drdzs[lowerModuleIndices[i]];
                slopes[i] = modulesInGPU.slopes[lowerModuleIndices[i]];
            }
            else
            {
                drdz = modulesInGPU.drdzs[modulesInGPU.partnerModuleIndex(lowerModuleIndices[i])];
                slopes[i] = modulesInGPU.slopes[modulesInGPU.partnerModuleIndex(lowerModuleIndices[i])];
            }

            delta1[i] = inv1;//1.1111f;//0.01;
            isFlat[i] = false;

            if(anchorHits)
            {
                //delta2[i] = (0.15f * drdz/sqrtf(1 + drdz * drdz));
                delta2[i] = (inv2 * drdz/sqrtf(1 + drdz * drdz));
            }
            else
            {
                //delta2[i] = (2.4f * drdz/sqrtf(1 + drdz * drdz));
                delta2[i] = (inv3 * drdz/sqrtf(1 + drdz * drdz));
            }
        }

        //category 4 - endcap PS
        else if(moduleSubdet == Endcap and moduleType == PS)
        {
            delta1[i] = inv1;//1.1111f;//0.01;
            if(moduleLayerType == Strip)
            {
                slopes[i] = modulesInGPU.slopes[lowerModuleIndices[i]];
            }
            else
            {
                slopes[i] = modulesInGPU.slopes[modulesInGPU.partnerModuleIndex(lowerModuleIndices[i])];

            }
            isFlat[i] = false;

            /*despite the type of the module layer of the lower module index,
            all anchor hits are on the pixel side and all non-anchor hits are
            on the strip side!*/
            if(anchorHits)
            {
                delta2[i] = inv2;//16.6666f;//0.15f;
            }
            else
            {
                delta2[i] = inv3;//266.666f;//2.4f;
            }
        }

        //category 5 - endcap 2S
        else if(moduleSubdet == Endcap and moduleType == TwoS)
        {
            delta1[i] = 1;//0.009;
            delta2[i] = 500*inv1;//555.5555f;//5.f;
            slopes[i] = modulesInGPU.slopes[lowerModuleIndices[i]];
            isFlat[i] = false;
        }
        else
        {
            printf("ERROR!!!!! I SHOULDN'T BE HERE!!!! subdet = %d, type = %d, side = %d\n", moduleSubdet, moduleType, moduleSide);
        }
    }
    // this for loop is kept to keep the physics results the same but I think this is a bug in the original code. This was kept at 5 and not nPoints
    for(size_t i = 3; i < 5; i++)
    {
        delta1[i] /= 0.009;
        delta2[i] /= 0.009;
    }
    chiSquared = computeChiSquared(3, xs, ys, delta1, delta2, slopes, isFlat, g, f, radius);
    
    return chiSquared;
}


//90pc threshold
__device__ bool SDL::passPT3RPhiChiSquaredCuts(struct modules& modulesInGPU, unsigned int lowerModuleIndex1, unsigned int lowerModuleIndex2, unsigned int lowerModuleIndex3, float& chiSquared)
{

    const int layer1 = modulesInGPU.layers[lowerModuleIndex1] + 6 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex1] == SDL::TwoS);
    const int layer2 = modulesInGPU.layers[lowerModuleIndex2] + 6 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex2] == SDL::TwoS);
    const int layer3 = modulesInGPU.layers[lowerModuleIndex3] + 6 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex3] == SDL::TwoS);

    if(layer1 == 8 and layer2 == 9 and layer3 == 10)
    {
        return chiSquared < 7.003;
    }
    else if(layer1 == 8 and layer2 == 9 and layer3 == 15)
    {
        return chiSquared < 0.5;
    }

    else if(layer1 == 7 and layer2 == 8 and layer3 == 9)
    {
        return chiSquared < 8.046;
    }
    else if(layer1 == 7 and layer2 == 8 and layer3 == 14)
    {
        return chiSquared < 0.575;
    }

    else if(layer1 == 1 and layer2 == 2 and layer3 == 7)
    {
        return chiSquared < 5.304;
    }
    else if(layer1 == 1 and layer2 == 2 and layer3 == 3)
    {
        return chiSquared < 10.6211;
    }
    else if(layer1 == 1 and layer2 == 7 and layer3 == 8)
    {
        return chiSquared < 4.617;
    }

    else if(layer1 == 2 and layer2 == 7 and layer3 == 8)
    {
        return chiSquared < 8.046;
    }
    else if(layer1 == 2 and layer2 == 7 and layer3 == 13)
    {
        return chiSquared < 0.435;
    }
    else if(layer1 == 2 and layer2 == 3 and layer3 == 7)
    {
        return chiSquared < 9.244;
    }
    else if(layer1 == 2 and layer2 == 3 and layer3 == 12)
    {
        return chiSquared < 0.287;
    }
    else if(layer1 == 2 and layer2 == 3 and layer3 == 4)
    {
        return chiSquared < 18.509;
    }

    return true;
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

__device__ bool SDL::runPixelTrackletDefaultAlgo_pT3(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, unsigned int innerInnerLowerModuleIndex, unsigned int innerOuterLowerModuleIndex, unsigned int outerInnerLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& deltaPhi, float& betaIn, float&
        betaOut, float& pt_beta, float& zLo, float& zHi, float& rtLo, float& rtHi, float& zLoPointed, float& zHiPointed, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ, unsigned int N_MAX_SEGMENTS_PER_MODULE)
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

    short outerInnerLowerModuleSubdet = modulesInGPU.subdets[outerInnerLowerModuleIndex];
    short outerOuterLowerModuleSubdet = modulesInGPU.subdets[outerOuterLowerModuleIndex];

    if(outerInnerLowerModuleSubdet == SDL::Barrel and outerOuterLowerModuleSubdet == SDL::Barrel)
    {
        pass = runTrackletDefaultAlgoPPBB_pT3(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, outerInnerLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta,N_MAX_SEGMENTS_PER_MODULE, zLo, zHi, zLoPointed, zHiPointed, sdlCut, betaOutCut, deltaBetaCut);
    }
    else if(outerInnerLowerModuleSubdet == SDL::Barrel and outerOuterLowerModuleSubdet == SDL::Endcap)
    {
        pass = runTrackletDefaultAlgoPPBB_pT3(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, outerInnerLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta,N_MAX_SEGMENTS_PER_MODULE, zLo, zHi, zLoPointed, zHiPointed, sdlCut, betaOutCut, deltaBetaCut);
    }
    else if(outerInnerLowerModuleSubdet == SDL::Endcap and outerOuterLowerModuleSubdet == SDL::Endcap)
    {
            pass = runTrackletDefaultAlgoPPEE_pT3(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, outerInnerLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta,N_MAX_SEGMENTS_PER_MODULE, zLo, rtLo, rtHi, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ);
    }

    return pass;
}

__device__ bool SDL::runTrackletDefaultAlgoPPBB_pT3(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU ,struct segments& segmentsInGPU, unsigned int pixelModuleIndex, unsigned int outerInnerLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, float& zOut, float& rtOut, float& dPhiPos, float& dPhi, float& betaIn, float& betaOut, float& pt_beta, unsigned int
        N_MAX_SEGMENTS_PER_MODULE, float& zLo, float& zHi, float& zLoPointed, float& zHiPointed, float& sdlCut, float& betaOutCut, float& deltaBetaCut) // pixel to BB and BE segments
{
    bool pass = true;

    bool isPS_OutLo = (modulesInGPU.moduleType[outerInnerLowerModuleIndex] == SDL::PS);
    unsigned int innerInnerAnchorHitIndex = segmentsInGPU.innerMiniDoubletAnchorHitIndices[innerSegmentIndex];

    unsigned int outerInnerAnchorHitIndex = segmentsInGPU.innerMiniDoubletAnchorHitIndices[outerSegmentIndex];

    unsigned int innerOuterAnchorHitIndex = segmentsInGPU.outerMiniDoubletAnchorHitIndices[innerSegmentIndex];
    unsigned int outerOuterAnchorHitIndex= segmentsInGPU.outerMiniDoubletAnchorHitIndices[outerSegmentIndex];
    if(fabsf(deltaPhi(hitsInGPU.xs[innerOuterAnchorHitIndex], hitsInGPU.ys[innerOuterAnchorHitIndex], hitsInGPU.zs[innerOuterAnchorHitIndex], hitsInGPU.xs[outerInnerAnchorHitIndex], hitsInGPU.ys[outerInnerAnchorHitIndex], hitsInGPU.zs[outerInnerAnchorHitIndex])) > M_PI/2.)
    {
        pass = false;
    }

    unsigned int pixelSegmentArrayIndex = innerSegmentIndex - (pixelModuleIndex * N_MAX_SEGMENTS_PER_MODULE);
    float ptIn = segmentsInGPU.ptIn[pixelSegmentArrayIndex];
    float ptSLo = ptIn;
    float px = segmentsInGPU.px[pixelSegmentArrayIndex];
    float py = segmentsInGPU.py[pixelSegmentArrayIndex];
    float pz = segmentsInGPU.pz[pixelSegmentArrayIndex];
    float ptErr = segmentsInGPU.ptErr[pixelSegmentArrayIndex];
    float etaErr = segmentsInGPU.etaErr[pixelSegmentArrayIndex];
    ptSLo = fmaxf(PTCUT, ptSLo - 10.0f*fmaxf(ptErr, 0.005f*ptSLo));
    ptSLo = fminf(10.0f, ptSLo);

    float rt_InLo = hitsInGPU.rts[innerInnerAnchorHitIndex];
    float rt_InUp = hitsInGPU.rts[innerOuterAnchorHitIndex];
    float rt_OutLo = hitsInGPU.rts[outerInnerAnchorHitIndex];
    float z_InLo = hitsInGPU.zs[innerInnerAnchorHitIndex];
    float z_InUp = hitsInGPU.zs[innerOuterAnchorHitIndex];
    float z_OutLo = hitsInGPU.zs[outerInnerAnchorHitIndex];

    float rt_InOut = hitsInGPU.rts[innerOuterAnchorHitIndex];
    float z_InOut = hitsInGPU.zs[innerOuterAnchorHitIndex];

    float alpha1GeV_OutLo = asinf(fminf(rt_OutLo * k2Rinv1GeVf / ptCut, sinAlphaMax));
    float rtRatio_OutLoInLo = rt_OutLo / rt_InLo; // Outer segment beginning rt divided by inner segment beginning rt;
    const float rtRatio_OutLoInOut = rt_OutLo / rt_InOut; // Outer segment beginning rt divided by inner segment beginning rt;

    float dzDrtScale = tanf(alpha1GeV_OutLo) / alpha1GeV_OutLo; // The track can bend in r-z plane slightly
    const float zpitch_InLo = 0.05f;
    const float zpitch_InOut = 0.05f;
    float zpitch_OutLo = (isPS_OutLo ? pixelPSZpitch : strip2SZpitch);
    float zGeom = zpitch_InLo + zpitch_OutLo;
    zHi = z_InOut + (z_InOut + deltaZLum) * (rtRatio_OutLoInOut - 1.f) * (z_InOut < 0.f ? 1.f : dzDrtScale) + (zpitch_InOut + zpitch_OutLo);
    zLo = z_InOut + (z_InOut - deltaZLum) * (rtRatio_OutLoInOut - 1.f) * (z_InOut > 0.f ? 1.f : dzDrtScale) - (zpitch_InOut + zpitch_OutLo); //slope-correction only on outer end

    if (not (z_OutLo >= zLo and z_OutLo <= zHi))
    {
        pass = false;
    }
    const float coshEta = sqrtf(ptIn * ptIn + pz * pz) / ptIn;
    // const float drt_OutLo_InLo = (rt_OutLo - rt_InLo);
    const float drt_OutLo_InUp = (rt_OutLo - rt_InUp);
    const float invRt_InLo = 1. / rt_InLo;
    const float r3_InLo = sqrtf(z_InLo * z_InLo + rt_InLo * rt_InLo);
    const float r3_InUp = sqrtf(z_InUp * z_InUp + rt_InUp * rt_InUp);
    float drt_InSeg = hitsInGPU.rts[innerOuterAnchorHitIndex] - hitsInGPU.rts[innerInnerAnchorHitIndex];
    float dz_InSeg = hitsInGPU.zs[innerOuterAnchorHitIndex] - hitsInGPU.zs[innerInnerAnchorHitIndex];

    float dr3_InSeg = sqrtf(hitsInGPU.rts[innerOuterAnchorHitIndex] * hitsInGPU.rts[innerOuterAnchorHitIndex] + hitsInGPU.zs[innerOuterAnchorHitIndex] * hitsInGPU.zs[innerOuterAnchorHitIndex]) - sqrtf(hitsInGPU.rts[innerInnerAnchorHitIndex] * hitsInGPU.rts[innerInnerAnchorHitIndex] + hitsInGPU.zs[innerInnerAnchorHitIndex] * hitsInGPU.zs[innerInnerAnchorHitIndex]);
    const float sdlThetaMulsF = 0.015f * sqrt(0.1f + 0.2 * (rt_OutLo - rt_InUp) / 50.f) * sqrt(r3_InUp / rt_InUp);
    const float sdlMuls = sdlThetaMulsF * 3.f / ptCut * 4.f; // will need a better guess than x4?

    float dzErr = drt_OutLo_InUp*etaErr*coshEta; //FIXME: check with the calc in the endcap
    dzErr *= dzErr;
    dzErr += 0.03f*0.03f; // pixel size x2. ... random for now
    dzErr *= 9.f; //3 sigma
    dzErr += sdlMuls*sdlMuls*drt_OutLo_InUp*drt_OutLo_InUp/3.f*coshEta*coshEta;//sloppy
    dzErr += zGeom*zGeom;
    dzErr = sqrtf(dzErr);

    const float dzDrIn = pz / ptIn;
    const float zWindow = dzErr / drt_InSeg * drt_OutLo_InUp + zGeom;
    const float dzMean = dzDrIn * drt_OutLo_InUp *
        (1.f + drt_OutLo_InUp * drt_OutLo_InUp * 4 * k2Rinv1GeVf * k2Rinv1GeVf / ptIn /
         ptIn / 24.f); // with curved path correction
    // Constructing upper and lower bound
    zLoPointed = z_InUp + dzMean - zWindow;
    zHiPointed = z_InUp + dzMean + zWindow;

    if (not (z_OutLo >= zLoPointed and z_OutLo <= zHiPointed))
    {
        pass = false;
    }
    const float sdlPVoff = 0.1f / rt_OutLo;
    sdlCut = alpha1GeV_OutLo + sqrtf(sdlMuls * sdlMuls + sdlPVoff * sdlPVoff);

    dPhiPos = deltaPhi(hitsInGPU.xs[innerOuterAnchorHitIndex], hitsInGPU.ys[innerOuterAnchorHitIndex], hitsInGPU.zs[innerOuterAnchorHitIndex], hitsInGPU.xs[outerOuterAnchorHitIndex], hitsInGPU.ys[outerOuterAnchorHitIndex], hitsInGPU.zs[outerOuterAnchorHitIndex]);
    //no dphipos cut
    float midPointX = (hitsInGPU.xs[innerInnerAnchorHitIndex] + hitsInGPU.xs[outerInnerAnchorHitIndex])/2;
    float midPointY = (hitsInGPU.ys[innerInnerAnchorHitIndex] + hitsInGPU.ys[outerInnerAnchorHitIndex])/2;
    float midPointZ = (hitsInGPU.zs[innerInnerAnchorHitIndex] + hitsInGPU.zs[outerInnerAnchorHitIndex])/2;

    float diffX = hitsInGPU.xs[outerInnerAnchorHitIndex] - hitsInGPU.xs[innerInnerAnchorHitIndex];
    float diffY = hitsInGPU.ys[outerInnerAnchorHitIndex] - hitsInGPU.ys[innerInnerAnchorHitIndex] ;
    float diffZ = hitsInGPU.zs[outerInnerAnchorHitIndex] - hitsInGPU.zs[innerInnerAnchorHitIndex];

    dPhi = deltaPhi(midPointX, midPointY, midPointZ, diffX, diffY, diffZ);
    if (not (fabsf(dPhi) <= sdlCut))
    {
        pass = false;
    }

    float alpha_InLo = segmentsInGPU.dPhiChanges[innerSegmentIndex];
    float alpha_OutLo = segmentsInGPU.dPhiChanges[outerSegmentIndex];

    bool isEC_lastLayer = modulesInGPU.subdets[outerOuterLowerModuleIndex] == SDL::Endcap and modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::TwoS;

    //unsigned int outerOuterEdgeIndex = hitsInGPU.edge2SMap[outerOuterAnchorHitIndex]; //POTENTIAL NUCLEAR GANDHI
    unsigned int outerOuterEdgeIndex = outerOuterAnchorHitIndex;

    float alpha_OutUp,alpha_OutUp_highEdge,alpha_OutUp_lowEdge;

    alpha_OutUp = deltaPhi(hitsInGPU.xs[outerOuterAnchorHitIndex],hitsInGPU.ys[outerOuterAnchorHitIndex],hitsInGPU.zs[outerOuterAnchorHitIndex],hitsInGPU.xs[outerOuterAnchorHitIndex] - hitsInGPU.xs[outerInnerAnchorHitIndex], hitsInGPU.ys[outerOuterAnchorHitIndex] - hitsInGPU.ys[outerInnerAnchorHitIndex], hitsInGPU.zs[outerOuterAnchorHitIndex] - hitsInGPU.zs[outerInnerAnchorHitIndex]);

    alpha_OutUp_highEdge = alpha_OutUp;
    alpha_OutUp_lowEdge = alpha_OutUp;

    float tl_axis_x = hitsInGPU.xs[outerOuterAnchorHitIndex] - hitsInGPU.xs[innerOuterAnchorHitIndex];
    float tl_axis_y = hitsInGPU.ys[outerOuterAnchorHitIndex] - hitsInGPU.ys[innerOuterAnchorHitIndex];
    float tl_axis_z = hitsInGPU.zs[outerOuterAnchorHitIndex] - hitsInGPU.zs[innerOuterAnchorHitIndex];

    float tl_axis_highEdge_x = tl_axis_x;
    float tl_axis_highEdge_y = tl_axis_y;
    float tl_axis_highEdge_z = tl_axis_z;

    float tl_axis_lowEdge_x = tl_axis_x;
    float tl_axis_lowEdge_y = tl_axis_y;
    float tl_axis_lowEdge_z = tl_axis_z;

    betaIn = -deltaPhi(px, py, pz, tl_axis_x, tl_axis_y, tl_axis_z);
    float betaInRHmin = betaIn;
    float betaInRHmax = betaIn;
    betaOut = -alpha_OutUp + deltaPhi(hitsInGPU.xs[outerOuterAnchorHitIndex], hitsInGPU.ys[outerOuterAnchorHitIndex], hitsInGPU.zs[outerOuterAnchorHitIndex], tl_axis_x, tl_axis_y, tl_axis_z);

    float betaOutRHmin = betaOut;
    float betaOutRHmax = betaOut;

    if(isEC_lastLayer)
    {
        alpha_OutUp_highEdge = deltaPhi(hitsInGPU.highEdgeXs[outerOuterEdgeIndex],hitsInGPU.highEdgeYs[outerOuterEdgeIndex],hitsInGPU.zs[outerOuterAnchorHitIndex],hitsInGPU.highEdgeXs[outerOuterEdgeIndex] - hitsInGPU.xs[outerInnerAnchorHitIndex], hitsInGPU.highEdgeYs[outerOuterEdgeIndex] - hitsInGPU.ys[outerInnerAnchorHitIndex], hitsInGPU.zs[outerOuterAnchorHitIndex] - hitsInGPU.zs[outerInnerAnchorHitIndex]);

        alpha_OutUp_lowEdge = deltaPhi(hitsInGPU.lowEdgeXs[outerOuterEdgeIndex],hitsInGPU.lowEdgeYs[outerOuterEdgeIndex],hitsInGPU.zs[outerOuterAnchorHitIndex],hitsInGPU.lowEdgeXs[outerOuterEdgeIndex] - hitsInGPU.xs[outerInnerAnchorHitIndex], hitsInGPU.lowEdgeYs[outerOuterEdgeIndex] - hitsInGPU.ys[outerInnerAnchorHitIndex], hitsInGPU.zs[outerOuterAnchorHitIndex] - hitsInGPU.zs[outerInnerAnchorHitIndex]);

        tl_axis_highEdge_x = hitsInGPU.highEdgeXs[outerOuterEdgeIndex] - hitsInGPU.xs[innerOuterAnchorHitIndex];
        tl_axis_highEdge_y = hitsInGPU.highEdgeYs[outerOuterEdgeIndex] - hitsInGPU.ys[innerOuterAnchorHitIndex];
        tl_axis_highEdge_z = hitsInGPU.zs[outerOuterAnchorHitIndex] - hitsInGPU.zs[innerOuterAnchorHitIndex];

        tl_axis_lowEdge_x = hitsInGPU.lowEdgeXs[outerOuterEdgeIndex] - hitsInGPU.xs[innerOuterAnchorHitIndex];
        tl_axis_lowEdge_y = hitsInGPU.lowEdgeYs[outerOuterEdgeIndex] - hitsInGPU.ys[innerOuterAnchorHitIndex];
        tl_axis_lowEdge_z = hitsInGPU.zs[outerOuterAnchorHitIndex] - hitsInGPU.zs[innerOuterAnchorHitIndex];

        betaOutRHmin = -alpha_OutUp_highEdge + deltaPhi(hitsInGPU.highEdgeXs[outerOuterEdgeIndex], hitsInGPU.highEdgeYs[outerOuterEdgeIndex], hitsInGPU.zs[outerOuterAnchorHitIndex], tl_axis_highEdge_x, tl_axis_highEdge_y, tl_axis_highEdge_z);

        betaOutRHmax = -alpha_OutUp_lowEdge + deltaPhi(hitsInGPU.lowEdgeXs[outerOuterEdgeIndex], hitsInGPU.lowEdgeYs[outerOuterEdgeIndex], hitsInGPU.zs[outerOuterAnchorHitIndex], tl_axis_lowEdge_x, tl_axis_lowEdge_y, tl_axis_lowEdge_z);
    }

    //beta computation
    float drt_tl_axis = sqrtf(tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);
    float drt_tl_lowEdge = sqrtf(tl_axis_lowEdge_x * tl_axis_lowEdge_x + tl_axis_lowEdge_y * tl_axis_lowEdge_y);
    float drt_tl_highEdge = sqrtf(tl_axis_highEdge_x * tl_axis_highEdge_x + tl_axis_highEdge_y * tl_axis_highEdge_y);

    //innerOuterAnchor - innerInnerAnchor
    const float rt_InSeg = sqrtf((hitsInGPU.xs[innerOuterAnchorHitIndex] - hitsInGPU.xs[innerInnerAnchorHitIndex]) * (hitsInGPU.xs[innerOuterAnchorHitIndex] - hitsInGPU.xs[innerInnerAnchorHitIndex]) + (hitsInGPU.ys[innerOuterAnchorHitIndex] - hitsInGPU.ys[innerInnerAnchorHitIndex]) * (hitsInGPU.ys[innerOuterAnchorHitIndex] - hitsInGPU.ys[innerInnerAnchorHitIndex]));

    //no betaIn cut for the pixels
    float betaAv = 0.5f * (betaIn + betaOut);
    pt_beta = ptIn;

    const float pt_betaMax = 7.0f;

    int lIn = 0;
    int lOut = isEC_lastLayer ? 11 : 5;
    float sdOut_dr = sqrtf((hitsInGPU.xs[outerOuterAnchorHitIndex] - hitsInGPU.xs[outerInnerAnchorHitIndex]) * (hitsInGPU.xs[outerOuterAnchorHitIndex] - hitsInGPU.xs[outerInnerAnchorHitIndex]) + (hitsInGPU.ys[outerOuterAnchorHitIndex] - hitsInGPU.ys[outerInnerAnchorHitIndex]) * (hitsInGPU.ys[outerOuterAnchorHitIndex] - hitsInGPU.ys[outerInnerAnchorHitIndex]));
    float sdOut_d = hitsInGPU.rts[outerOuterAnchorHitIndex] - hitsInGPU.rts[outerInnerAnchorHitIndex];

    const float diffDr = fabsf(rt_InSeg - sdOut_dr) / fabsf(rt_InSeg + sdOut_dr);

    runDeltaBetaIterations(betaIn, betaOut, betaAv, pt_beta, rt_InSeg, sdOut_dr, drt_tl_axis, lIn);

     const float betaInMMSF = (fabsf(betaInRHmin + betaInRHmax) > 0) ? (2.f * betaIn / fabsf(betaInRHmin + betaInRHmax)) : 0.; //mean value of min,max is the old betaIn
    const float betaOutMMSF = (fabsf(betaOutRHmin + betaOutRHmax) > 0) ? (2.f * betaOut / fabsf(betaOutRHmin + betaOutRHmax)) : 0.;
    betaInRHmin *= betaInMMSF;
    betaInRHmax *= betaInMMSF;
    betaOutRHmin *= betaOutMMSF;
    betaOutRHmax *= betaOutMMSF;

    const float dBetaMuls = sdlThetaMulsF * 4.f / fminf(fabsf(pt_beta), pt_betaMax); //need to confirm the range-out value of 7 GeV

    float sdIn_rt = hitsInGPU.rts[innerOuterAnchorHitIndex];
    float sdOut_rt = hitsInGPU.rts[outerInnerAnchorHitIndex];
    float sdIn_z = hitsInGPU.zs[innerOuterAnchorHitIndex];
    float sdOut_z = hitsInGPU.zs[outerInnerAnchorHitIndex];
    const float alphaInAbsReg =  fmaxf(fabsf(alpha_InLo), asinf(fminf(sdIn_rt * k2Rinv1GeVf / 3.0f, sinAlphaMax)));
    const float alphaOutAbsReg = fmaxf(fabsf(alpha_OutLo), asinf(fminf(sdOut_rt * k2Rinv1GeVf / 3.0f, sinAlphaMax)));
    const float dBetaInLum = lIn < 11 ? 0.0f : fabsf(alphaInAbsReg*deltaZLum / sdIn_z);
    const float dBetaOutLum = lOut < 11 ? 0.0f : fabsf(alphaOutAbsReg*deltaZLum / sdOut_z);
    const float dBetaLum2 = (dBetaInLum + dBetaOutLum) * (dBetaInLum + dBetaOutLum);

    const float sinDPhi = sinf(dPhi);
    const float dBetaRIn2 = 0; // TODO-RH

    float dBetaROut = 0;
    if(isEC_lastLayer)
    {
        dBetaROut = (sqrtf(hitsInGPU.highEdgeXs[outerOuterEdgeIndex] * hitsInGPU.highEdgeXs[outerOuterEdgeIndex] + hitsInGPU.highEdgeYs[outerOuterEdgeIndex] * hitsInGPU.highEdgeYs[outerOuterEdgeIndex]) - sqrtf(hitsInGPU.lowEdgeXs[outerOuterEdgeIndex] * hitsInGPU.lowEdgeXs[outerOuterEdgeIndex] + hitsInGPU.lowEdgeYs[outerOuterEdgeIndex] * hitsInGPU.lowEdgeYs[outerOuterEdgeIndex])) * sinDPhi / drt_tl_axis;
    }

    const float dBetaROut2 =  dBetaROut * dBetaROut;

    betaOutCut = asinf(fminf(drt_tl_axis*k2Rinv1GeVf / ptCut, sinAlphaMax)) //FIXME: need faster version
        + (0.02f / sdOut_d) + sqrtf(dBetaLum2 + dBetaMuls*dBetaMuls);

    //Cut #6: The real beta cut
    if (not (fabsf(betaOut) < betaOutCut))
    {
        pass = false;
    }

    const float pt_betaOut = drt_tl_axis * k2Rinv1GeVf / sin(betaOut);
    const float dBetaRes = 0.02f / fminf(sdOut_d, drt_InSeg);
    const float dBetaCut2 = (dBetaRes*dBetaRes * 2.0f + dBetaMuls * dBetaMuls + dBetaLum2 + dBetaRIn2 + dBetaROut2
            + 0.25 * (fabsf(betaInRHmin - betaInRHmax) + fabsf(betaOutRHmin - betaOutRHmax)) * (fabsf(betaInRHmin - betaInRHmax) + fabsf(betaOutRHmin - betaOutRHmax)));
    float dBeta = betaIn - betaOut;
    deltaBetaCut = sqrtf(dBetaCut2);
    if (not (dBeta * dBeta <= dBetaCut2))
    {
        //printf("dBeta2 = %f, dBetaCut2 = %f\n",dBeta * dBeta, dBetaCut2);
        pass = false;
    }

    return pass;
}

__device__ bool SDL::runTrackletDefaultAlgoPPEE_pT3(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU ,struct segments& segmentsInGPU, unsigned int pixelModuleIndex, unsigned int outerInnerLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& dPhi, float& betaIn, float& betaOut, float& pt_beta, unsigned int
        N_MAX_SEGMENTS_PER_MODULE,  float& zLo, float& rtLo, float& rtHi, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ) // pixel to EE segments
{
    bool pass = true;
    bool isPS_OutLo = (modulesInGPU.moduleType[outerInnerLowerModuleIndex] == SDL::PS);
    unsigned int innerInnerAnchorHitIndex = segmentsInGPU.innerMiniDoubletAnchorHitIndices[innerSegmentIndex];

    unsigned int outerInnerAnchorHitIndex = segmentsInGPU.innerMiniDoubletAnchorHitIndices[outerSegmentIndex];

    unsigned int innerOuterAnchorHitIndex = segmentsInGPU.outerMiniDoubletAnchorHitIndices[innerSegmentIndex];
    unsigned int outerOuterAnchorHitIndex= segmentsInGPU.outerMiniDoubletAnchorHitIndices[outerSegmentIndex];
    unsigned int pixelSegmentArrayIndex = innerSegmentIndex - (pixelModuleIndex * N_MAX_SEGMENTS_PER_MODULE);
    float ptIn = segmentsInGPU.ptIn[pixelSegmentArrayIndex];
    float ptSLo = ptIn;
    float px = segmentsInGPU.px[pixelSegmentArrayIndex];
    float py = segmentsInGPU.py[pixelSegmentArrayIndex];
    float pz = segmentsInGPU.pz[pixelSegmentArrayIndex];
    float ptErr = segmentsInGPU.ptErr[pixelSegmentArrayIndex];
    float etaErr = segmentsInGPU.etaErr[pixelSegmentArrayIndex];

    ptSLo = fmaxf(PTCUT, ptSLo - 10.0f*fmaxf(ptErr, 0.005f*ptSLo));
    ptSLo = fminf(10.0f, ptSLo);
    float rtIn = hitsInGPU.rts[innerOuterAnchorHitIndex];
    rtOut = hitsInGPU.rts[outerInnerAnchorHitIndex];
    float zIn = hitsInGPU.zs[innerOuterAnchorHitIndex];
    zOut = hitsInGPU.zs[outerInnerAnchorHitIndex];
    float rtOut_o_rtIn = rtOut/rtIn;
    const float zpitch_InLo = 0.05f;
    float zpitch_OutLo = (isPS_OutLo ? pixelPSZpitch : strip2SZpitch);
    float zGeom = zpitch_InLo + zpitch_OutLo;

    const float sdlSlope = asinf(fminf(rtOut * k2Rinv1GeVf / ptCut, sinAlphaMax));
    const float dzDrtScale = tanf(sdlSlope) / sdlSlope;//FIXME: need approximate value
    zLo = zIn + (zIn - deltaZLum) * (rtOut_o_rtIn - 1.f) * (zIn > 0.f ? 1.f : dzDrtScale) - zGeom; //slope-correction only on outer end

    if (not (zIn * zOut > 0))
    {
        pass = false;
    }

    const float dLum = copysignf(deltaZLum, zIn);
    bool isOutSgInnerMDPS = modulesInGPU.moduleType[outerInnerLowerModuleIndex] == SDL::PS;

    const float rtGeom1 = isOutSgInnerMDPS ? pixelPSZpitch : strip2SZpitch;//FIXME: make this chosen by configuration for lay11,12 full PS
    const float zGeom1 = copysignf(zGeom, zIn); //used in B-E region
    rtLo = rtIn * (1.f + (zOut - zIn - zGeom1) / (zIn + zGeom1 + dLum) / dzDrtScale) - rtGeom1; //slope correction only on the lower end

    if (not (rtOut >= rtLo))
    {
        pass = false;
    }

    float zInForHi = zIn - zGeom1 - dLum;
    if (zInForHi * zIn < 0)
        zInForHi = copysignf(0.1f, zIn);
    rtHi = rtIn * (1.f + (zOut - zIn + zGeom1) / zInForHi) + rtGeom1;

    // Cut #2: rt condition
    if (not (rtOut >= rtLo and rtOut <= rtHi))
    {
        pass = false;
    }

    const float rt_InUp = hitsInGPU.rts[innerOuterAnchorHitIndex];
    const float rt_OutLo = hitsInGPU.rts[outerInnerAnchorHitIndex];
    const float z_InUp = hitsInGPU.zs[innerOuterAnchorHitIndex];
    const float dzOutInAbs = fabsf(zOut - zIn);
    const float coshEta = hypotf(ptIn, pz) / ptIn;
    const float multDzDr = dzOutInAbs*coshEta/(coshEta*coshEta - 1.f);
    const float r3_InUp = sqrtf(z_InUp * z_InUp + rt_InUp * rt_InUp);
    const float sdlThetaMulsF = 0.015f * sqrtf(0.1f + 0.2 * (rt_OutLo - rtIn) / 50.f) * sqrtf(r3_InUp / rtIn);
    const float sdlMuls = sdlThetaMulsF * 3.f / ptCut * 4.f; // will need a better guess than x4?

    float drtErr = etaErr*multDzDr;
    drtErr *= drtErr;
    drtErr += 0.03f*0.03f; // pixel size x2. ... random for now
    drtErr *= 9.f; //3 sigma
    drtErr += sdlMuls*sdlMuls*multDzDr*multDzDr/3.f*coshEta*coshEta;//sloppy: relative muls is 1/3 of total muls
    drtErr = sqrtf(drtErr);
    const float drtDzIn = fabsf(ptIn / pz);//all tracks are out-going in endcaps?

    const float drt_OutLo_InUp = (rt_OutLo - rt_InUp); // drOutIn

    const float rtWindow = drtErr + rtGeom1;
    const float drtMean = drtDzIn * dzOutInAbs *
        (1.f - drt_OutLo_InUp * drt_OutLo_InUp * 4 * k2Rinv1GeVf * k2Rinv1GeVf / ptIn /
         ptIn / 24.f); // with curved path correction
    const float rtLo_point = rtIn + drtMean - rtWindow;
    const float rtHi_point = rtIn + drtMean + rtWindow;

    // Cut #3: rt-z pointed
    if (not (rtOut >= rtLo_point and rtOut <= rtHi_point))
    {
        pass = false;
    }

    const float alpha1GeV_OutLo = asinf(fminf(rt_OutLo * k2Rinv1GeVf / ptCut, sinAlphaMax));
    const float sdlPVoff = 0.1f / rt_OutLo;
    sdlCut = alpha1GeV_OutLo + sqrtf(sdlMuls * sdlMuls + sdlPVoff * sdlPVoff);

    deltaPhiPos = deltaPhi(hitsInGPU.xs[innerOuterAnchorHitIndex], hitsInGPU.ys[innerOuterAnchorHitIndex], hitsInGPU.zs[innerOuterAnchorHitIndex], hitsInGPU.xs[outerOuterAnchorHitIndex], hitsInGPU.ys[outerOuterAnchorHitIndex], hitsInGPU.zs[outerOuterAnchorHitIndex]);
    //no deltaphipos cut
    float midPointX = (hitsInGPU.xs[innerInnerAnchorHitIndex] + hitsInGPU.xs[outerInnerAnchorHitIndex])/2;
    float midPointY = (hitsInGPU.ys[innerInnerAnchorHitIndex] + hitsInGPU.ys[outerInnerAnchorHitIndex])/2;
    float midPointZ = (hitsInGPU.zs[innerInnerAnchorHitIndex] + hitsInGPU.zs[outerInnerAnchorHitIndex])/2;

    float diffX = (-hitsInGPU.xs[innerInnerAnchorHitIndex] + hitsInGPU.xs[outerInnerAnchorHitIndex]);
    float diffY = (-hitsInGPU.ys[innerInnerAnchorHitIndex] + hitsInGPU.ys[outerInnerAnchorHitIndex]);
    float diffZ = (-hitsInGPU.zs[innerInnerAnchorHitIndex] + hitsInGPU.zs[outerInnerAnchorHitIndex]);

    dPhi = deltaPhi(midPointX, midPointY, midPointZ, diffX, diffY, diffZ);

    // Cut #5: deltaPhiChange
    if (not (fabsf(dPhi) <= sdlCut))
    {
        pass = false;
    }
    float alpha_InLo = segmentsInGPU.dPhiChanges[innerSegmentIndex];
    float alpha_OutLo = segmentsInGPU.dPhiChanges[outerSegmentIndex];

    bool isEC_lastLayer = modulesInGPU.subdets[outerOuterLowerModuleIndex] == SDL::Endcap and modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::TwoS;

    unsigned int outerOuterEdgeIndex = outerOuterAnchorHitIndex;

    float alpha_OutUp,alpha_OutUp_highEdge,alpha_OutUp_lowEdge;

    alpha_OutUp = deltaPhi(hitsInGPU.xs[outerOuterAnchorHitIndex],hitsInGPU.ys[outerOuterAnchorHitIndex],hitsInGPU.zs[outerOuterAnchorHitIndex],hitsInGPU.xs[outerOuterAnchorHitIndex] - hitsInGPU.xs[outerInnerAnchorHitIndex], hitsInGPU.ys[outerOuterAnchorHitIndex] - hitsInGPU.ys[outerInnerAnchorHitIndex], hitsInGPU.zs[outerOuterAnchorHitIndex] - hitsInGPU.zs[outerInnerAnchorHitIndex]);

    alpha_OutUp_highEdge = alpha_OutUp;
    alpha_OutUp_lowEdge = alpha_OutUp;

    float tl_axis_x = hitsInGPU.xs[outerOuterAnchorHitIndex] - hitsInGPU.xs[innerOuterAnchorHitIndex];
    float tl_axis_y = hitsInGPU.ys[outerOuterAnchorHitIndex] - hitsInGPU.ys[innerOuterAnchorHitIndex];
    float tl_axis_z = hitsInGPU.zs[outerOuterAnchorHitIndex] - hitsInGPU.zs[innerOuterAnchorHitIndex];

    float tl_axis_highEdge_x = tl_axis_x;
    float tl_axis_highEdge_y = tl_axis_y;
    float tl_axis_highEdge_z = tl_axis_z;

    float tl_axis_lowEdge_x = tl_axis_x;
    float tl_axis_lowEdge_y = tl_axis_y;
    float tl_axis_lowEdge_z = tl_axis_z;

    betaIn = -deltaPhi(px, py, pz, tl_axis_x, tl_axis_y, tl_axis_z);
    float betaInRHmin = betaIn;
    float betaInRHmax = betaIn;

    betaOut = -alpha_OutUp + deltaPhi(hitsInGPU.xs[outerOuterAnchorHitIndex], hitsInGPU.ys[outerOuterAnchorHitIndex], hitsInGPU.zs[outerOuterAnchorHitIndex], tl_axis_x, tl_axis_y, tl_axis_z);

    float betaOutRHmin = betaOut;
    float betaOutRHmax = betaOut;

    if(isEC_lastLayer)
    {
        alpha_OutUp_highEdge = deltaPhi(hitsInGPU.highEdgeXs[outerOuterEdgeIndex],hitsInGPU.highEdgeYs[outerOuterEdgeIndex],hitsInGPU.zs[outerOuterAnchorHitIndex],hitsInGPU.highEdgeXs[outerOuterEdgeIndex] - hitsInGPU.xs[outerInnerAnchorHitIndex], hitsInGPU.highEdgeYs[outerOuterEdgeIndex] - hitsInGPU.ys[outerInnerAnchorHitIndex], hitsInGPU.zs[outerOuterAnchorHitIndex] - hitsInGPU.zs[outerInnerAnchorHitIndex]);

        alpha_OutUp_lowEdge = deltaPhi(hitsInGPU.lowEdgeXs[outerOuterEdgeIndex],hitsInGPU.lowEdgeYs[outerOuterEdgeIndex],hitsInGPU.zs[outerOuterAnchorHitIndex],hitsInGPU.lowEdgeXs[outerOuterEdgeIndex] - hitsInGPU.xs[outerInnerAnchorHitIndex], hitsInGPU.lowEdgeYs[outerOuterEdgeIndex] - hitsInGPU.ys[outerInnerAnchorHitIndex], hitsInGPU.zs[outerOuterAnchorHitIndex] - hitsInGPU.zs[outerInnerAnchorHitIndex]);

        tl_axis_highEdge_x = hitsInGPU.highEdgeXs[outerOuterEdgeIndex] - hitsInGPU.xs[innerOuterAnchorHitIndex];
        tl_axis_highEdge_y = hitsInGPU.highEdgeYs[outerOuterEdgeIndex] - hitsInGPU.ys[innerOuterAnchorHitIndex];
        tl_axis_highEdge_z = hitsInGPU.zs[outerOuterAnchorHitIndex] - hitsInGPU.zs[innerOuterAnchorHitIndex];

        tl_axis_lowEdge_x = hitsInGPU.lowEdgeXs[outerOuterEdgeIndex] - hitsInGPU.xs[innerOuterAnchorHitIndex];
        tl_axis_lowEdge_y = hitsInGPU.lowEdgeYs[outerOuterEdgeIndex] - hitsInGPU.ys[innerOuterAnchorHitIndex];
        tl_axis_lowEdge_z = hitsInGPU.zs[outerOuterAnchorHitIndex] - hitsInGPU.zs[innerOuterAnchorHitIndex];

        betaOutRHmin = -alpha_OutUp_highEdge + deltaPhi(hitsInGPU.highEdgeXs[outerOuterEdgeIndex], hitsInGPU.highEdgeYs[outerOuterEdgeIndex], hitsInGPU.zs[outerOuterAnchorHitIndex], tl_axis_highEdge_x, tl_axis_highEdge_y, tl_axis_highEdge_z);

        betaOutRHmax = -alpha_OutUp_lowEdge + deltaPhi(hitsInGPU.lowEdgeXs[outerOuterEdgeIndex], hitsInGPU.lowEdgeYs[outerOuterEdgeIndex], hitsInGPU.zs[outerOuterAnchorHitIndex], tl_axis_lowEdge_x, tl_axis_lowEdge_y, tl_axis_lowEdge_z);
    }

    //beta computation
    float drt_tl_axis = sqrtf(tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);
    float drt_tl_lowEdge = sqrtf(tl_axis_lowEdge_x * tl_axis_lowEdge_x + tl_axis_lowEdge_y * tl_axis_lowEdge_y);
    float drt_tl_highEdge = sqrtf(tl_axis_highEdge_x * tl_axis_highEdge_x + tl_axis_highEdge_y * tl_axis_highEdge_y);
//no betaIn cut for the pixels
    const float rt_InSeg = sqrtf((hitsInGPU.xs[innerOuterAnchorHitIndex] - hitsInGPU.xs[innerInnerAnchorHitIndex]) * (hitsInGPU.xs[innerOuterAnchorHitIndex] - hitsInGPU.xs[innerInnerAnchorHitIndex]) + (hitsInGPU.ys[innerOuterAnchorHitIndex] - hitsInGPU.ys[innerInnerAnchorHitIndex]) * (hitsInGPU.ys[innerOuterAnchorHitIndex] - hitsInGPU.ys[innerInnerAnchorHitIndex]));

    float betaAv = 0.5f * (betaIn + betaOut);
    pt_beta = ptIn;

    const float pt_betaMax = 7.0f;

    int lIn = 0;
    int lOut = isEC_lastLayer ? 11 : 5;
    float sdOut_dr = sqrtf((hitsInGPU.xs[outerOuterAnchorHitIndex] - hitsInGPU.xs[outerInnerAnchorHitIndex]) * (hitsInGPU.xs[outerOuterAnchorHitIndex] - hitsInGPU.xs[outerInnerAnchorHitIndex]) + (hitsInGPU.ys[outerOuterAnchorHitIndex] - hitsInGPU.ys[outerInnerAnchorHitIndex]) * (hitsInGPU.ys[outerOuterAnchorHitIndex] - hitsInGPU.ys[outerInnerAnchorHitIndex]));
    float sdOut_d = hitsInGPU.rts[outerOuterAnchorHitIndex] - hitsInGPU.rts[outerInnerAnchorHitIndex];

    const float diffDr = fabsf(rt_InSeg - sdOut_dr) / fabsf(rt_InSeg + sdOut_dr);

    runDeltaBetaIterations(betaIn, betaOut, betaAv, pt_beta, rt_InSeg, sdOut_dr, drt_tl_axis, lIn);

     const float betaInMMSF = (fabsf(betaInRHmin + betaInRHmax) > 0) ? (2.f * betaIn / fabsf(betaInRHmin + betaInRHmax)) : 0.; //mean value of min,max is the old betaIn
    const float betaOutMMSF = (fabsf(betaOutRHmin + betaOutRHmax) > 0) ? (2.f * betaOut / fabsf(betaOutRHmin + betaOutRHmax)) : 0.;
    betaInRHmin *= betaInMMSF;
    betaInRHmax *= betaInMMSF;
    betaOutRHmin *= betaOutMMSF;
    betaOutRHmax *= betaOutMMSF;

    const float dBetaMuls = sdlThetaMulsF * 4.f / fminf(fabsf(pt_beta), pt_betaMax); //need to confirm the range-out value of 7 GeV

    float sdIn_rt = hitsInGPU.rts[innerOuterAnchorHitIndex];
    float sdOut_rt = hitsInGPU.rts[outerInnerAnchorHitIndex];
    float sdIn_z = hitsInGPU.zs[innerOuterAnchorHitIndex];
    float sdOut_z = hitsInGPU.zs[outerInnerAnchorHitIndex];
    const float alphaInAbsReg =  fmaxf(fabsf(alpha_InLo), asinf(fminf(sdIn_rt * k2Rinv1GeVf / 3.0f, sinAlphaMax)));
    const float alphaOutAbsReg = fmaxf(fabsf(alpha_OutLo), asinf(fminf(sdOut_rt * k2Rinv1GeVf / 3.0f, sinAlphaMax)));
    const float dBetaInLum = lIn < 11 ? 0.0f : fabsf(alphaInAbsReg*deltaZLum / sdIn_z);
    const float dBetaOutLum = lOut < 11 ? 0.0f : fabsf(alphaOutAbsReg*deltaZLum / sdOut_z);
    const float dBetaLum2 = (dBetaInLum + dBetaOutLum) * (dBetaInLum + dBetaOutLum);

    const float sinDPhi = sinf(dPhi);
    const float dBetaRIn2 = 0; // TODO-RH

    float dBetaROut = 0;
    if(isEC_lastLayer)
    {
        dBetaROut = (sqrtf(hitsInGPU.highEdgeXs[outerOuterEdgeIndex] * hitsInGPU.highEdgeXs[outerOuterEdgeIndex] + hitsInGPU.highEdgeYs[outerOuterEdgeIndex] * hitsInGPU.highEdgeYs[outerOuterEdgeIndex]) - sqrtf(hitsInGPU.lowEdgeXs[outerOuterEdgeIndex] * hitsInGPU.lowEdgeXs[outerOuterEdgeIndex] + hitsInGPU.lowEdgeYs[outerOuterEdgeIndex] * hitsInGPU.lowEdgeYs[outerOuterEdgeIndex])) * sinDPhi / drt_tl_axis;
    }

    const float dBetaROut2 =  dBetaROut * dBetaROut;

    betaOutCut = asinf(fminf(drt_tl_axis*k2Rinv1GeVf / ptCut, sinAlphaMax)) //FIXME: need faster version
        + (0.02f / sdOut_d) + sqrtf(dBetaLum2 + dBetaMuls*dBetaMuls);

    //Cut #6: The real beta cut
    if (not (fabsf(betaOut) < betaOutCut))
    {
        pass = false;
    }

    const float pt_betaOut = drt_tl_axis * k2Rinv1GeVf / sin(betaOut);
    float drt_InSeg = hitsInGPU.rts[innerOuterAnchorHitIndex] - hitsInGPU.rts[innerInnerAnchorHitIndex];

    const float dBetaRes = 0.02f / fminf(sdOut_d, drt_InSeg);
    const float dBetaCut2 = (dBetaRes*dBetaRes * 2.0f + dBetaMuls * dBetaMuls + dBetaLum2 + dBetaRIn2 + dBetaROut2
            + 0.25 * (fabsf(betaInRHmin - betaInRHmax) + fabsf(betaOutRHmin - betaOutRHmax)) * (fabsf(betaInRHmin - betaInRHmax) + fabsf(betaOutRHmin - betaOutRHmax)));
    float dBeta = betaIn - betaOut;
    deltaBetaCut = sqrtf(dBetaCut2);
    if (not (dBeta * dBeta <= dBetaCut2))
    {
        pass = false;
    }

    return pass;
}
