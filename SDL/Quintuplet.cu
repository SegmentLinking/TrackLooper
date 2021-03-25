#ifdef __CUDACC__
#define CUDA_CONST_VAR __device__
#endif
# include "Quintuplet.cuh"
//#ifdef CACHE_ALLOC
#include "allocate.h"
//#endif

SDL::quintuplets::quintuplets()
{
    tripletIndices = nullptr;
    lowerModuleIndices = nullptr;
    nQuintuplets = nullptr;
    innerTripletPt = nullptr;
    outerTripletPt = nullptr;

}

void SDL::quintuplets::freeMemory()
{
    cudaFree(tripletIndices);
    cudaFree(lowerModuleIndices);
    cudaFree(nQuintuplets);
    cudaFree(innerTripletPt);
    cudaFree(outerTripletPt);
}
void SDL::createQuintupletsInUnifiedMemory(struct SDL::quintuplets& quintupletsInGPU, unsigned int maxQuintuplets, unsigned int nLowerModules)
{
    cudaMallocManaged(&quintupletsInGPU.tripletIndices, 2 * maxQuintuplets * nLowerModules * sizeof(unsigned int));
    cudaMallocManaged(&quintupletsInGPU.lowerModuleIndices, 5 * maxQuintuplets * nLowerModules * sizeof(unsigned int)); 
    cudaMallocManaged(&quintupletsInGPU.innerTripletPt, 2 * maxQuintuplets * nLowerModules * sizeof(float));
    cudaMallocManaged(&quintupletsInGPU.innerRadiusFromRegression, 2 * maxQuintuplets * nLowerModules * sizeof(float));

    cudaMallocManaged(&quintupletsInGPU.nQuintuplets, nLowerModules * sizeof(unsigned int));

    quintupletsInGPU.outerTripletPt = quintupletsInGPU.innerTripletPt + nLowerModules * maxQuintuplets;
    quintupletsInGPU.outerRadiusFromRegression = quintupletsInGPU.innerRadiusFromRegression + nLowerModules * maxQuintuplets;

#pragma omp parallel for
    for(size_t i = 0; i<nLowerModules;i++)
    {
        quintupletsInGPU.nQuintuplets[i] = 0;
    }

}

__device__ void SDL::addQuintupletToMemory(struct SDL::quintuplets& quintupletsInGPU, unsigned int innerTripletIndex, unsigned int outerTripletIndex, unsigned int lowerModule1, unsigned int lowerModule2, unsigned int lowerModule3, unsigned int lowerModule4, unsigned int lowerModule5, float innerTripletPt, float outerTripletPt, float innerRadiusFromRegression, float outerRadiusFromRegression, unsigned int quintupletIndex)
{
    quintupletsInGPU.tripletIndices[2 * quintupletIndex] = innerTripletIndex;
    quintupletsInGPU.tripletIndices[2 * quintupletIndex + 1] = outerTripletIndex;

    quintupletsInGPU.innerTripletPt[quintupletIndex] = innerTripletPt;
    quintupletsInGPU.outerTripletPt[quintupletIndex] = outerTripletPt;

    quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex] = lowerModule1;
    quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex + 1] = lowerModule2;
    quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex + 2] = lowerModule3;
    quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex + 3] = lowerModule4;
    quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex + 4] = lowerModule5;

    quintupletsInGPU.innerRadiusFromRegression[quintupletIndex] = innerRadiusFromRegression;
    quintupletsInGPU.outerRadiusFromRegression[quintupletIndex] = outerRadiusFromRegression;

}

__device__ bool SDL::runQuintupletDefaultAlgo(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, unsigned int lowerModuleIndex1, unsigned int lowerModuleIndex2, unsigned int lowerModuleIndex3, unsigned int lowerModuleIndex4, unsigned int lowerModuleIndex5, unsigned int innerTripletIndex, unsigned int outerTripletIndex, float& innerRadius, float& outerRadius,
        float& innerRadiusFromRegression, float& outerRadiusFromRegression)
{
    bool pass = true;

    if(not T5HasCommonMiniDoublet(tripletsInGPU, segmentsInGPU, innerTripletIndex, outerTripletIndex))
    {
        pass = false;
    }

    unsigned int firstSegmentIndex = tripletsInGPU.segmentIndices[2 * innerTripletIndex];
    unsigned int secondSegmentIndex = tripletsInGPU.segmentIndices[2 * innerTripletIndex + 1];
    unsigned int thirdSegmentIndex = tripletsInGPU.segmentIndices[2 * outerTripletIndex];
    unsigned int fourthSegmentIndex = tripletsInGPU.segmentIndices[2 * outerTripletIndex + 1];
    //apply T4 criteria between segments 1 and 3
    float zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut; //temp stuff
    if(not runTrackletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, segmentsInGPU.innerLowerModuleIndices[firstSegmentIndex], segmentsInGPU.outerLowerModuleIndices[firstSegmentIndex], segmentsInGPU.innerLowerModuleIndices[thirdSegmentIndex], segmentsInGPU.outerLowerModuleIndices[thirdSegmentIndex], firstSegmentIndex, thirdSegmentIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut))
    {
        pass = false;
    }
    if(not runTrackletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, segmentsInGPU.innerLowerModuleIndices[firstSegmentIndex], segmentsInGPU.outerLowerModuleIndices[firstSegmentIndex], segmentsInGPU.innerLowerModuleIndices[fourthSegmentIndex], segmentsInGPU.outerLowerModuleIndices[fourthSegmentIndex], firstSegmentIndex, fourthSegmentIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut))
    {
        pass = false;
    }


    unsigned int firstMDIndex = segmentsInGPU.mdIndices[2 * firstSegmentIndex];
    unsigned int secondMDIndex = segmentsInGPU.mdIndices[2 * secondSegmentIndex];
    unsigned int thirdMDIndex = segmentsInGPU.mdIndices[2 * secondSegmentIndex + 1];
    unsigned int fourthMDIndex = segmentsInGPU.mdIndices[2 * thirdSegmentIndex + 1];
    unsigned int fifthMDIndex = segmentsInGPU.mdIndices[2 * fourthSegmentIndex + 1];

    unsigned int firstMDLowerHitIndex = mdsInGPU.hitIndices[2 * firstMDIndex];
    unsigned int firstMDUpperHitIndex = mdsInGPU.hitIndices[2 * firstMDIndex + 1];
    unsigned int secondMDLowerHitIndex = mdsInGPU.hitIndices[2 * secondMDIndex];
    unsigned int secondMDUpperHitIndex = mdsInGPU.hitIndices[2 * secondMDIndex + 1];
    unsigned int thirdMDLowerHitIndex = mdsInGPU.hitIndices[2 * thirdMDIndex];
    unsigned int thirdMDUpperHitIndex = mdsInGPU.hitIndices[2 * thirdMDIndex + 1];
    unsigned int fourthMDLowerHitIndex = mdsInGPU.hitIndices[2 * fourthMDIndex];
    unsigned int fourthMDUpperHitIndex = mdsInGPU.hitIndices[2 * fourthMDIndex + 1];
    unsigned int fifthMDLowerHitIndex = mdsInGPU.hitIndices[2 * fifthMDIndex];
    unsigned int fifthMDUpperHitIndex = mdsInGPU.hitIndices[2 * fifthMDIndex + 1];

    //radius computation from the three triplet MD anchor hits
    unsigned int innerTripletFirstSegmentAnchorHitIndex = segmentsInGPU.innerMiniDoubletAnchorHitIndices[firstSegmentIndex];
    unsigned int innerTripletSecondSegmentAnchorHitIndex = segmentsInGPU.outerMiniDoubletAnchorHitIndices[firstSegmentIndex]; //same as second segment inner MD anchorhit index
    unsigned int innerTripletThirdSegmentAnchorHitIndex = segmentsInGPU.outerMiniDoubletAnchorHitIndices[secondSegmentIndex]; //same as third segment inner MD anchor hit index

    unsigned int outerTripletSecondSegmentAnchorHitIndex = segmentsInGPU.outerMiniDoubletAnchorHitIndices[thirdSegmentIndex]; //same as fourth segment inner MD anchor hit index
    unsigned int outerTripletThirdSegmentAnchorHitIndex = segmentsInGPU.outerMiniDoubletAnchorHitIndices[fourthSegmentIndex];

    float x1 = hitsInGPU.xs[innerTripletFirstSegmentAnchorHitIndex];
    float x2 = hitsInGPU.xs[innerTripletSecondSegmentAnchorHitIndex];
    float x3 = hitsInGPU.xs[innerTripletThirdSegmentAnchorHitIndex];
    float x4 = hitsInGPU.xs[outerTripletSecondSegmentAnchorHitIndex];
    float x5 = hitsInGPU.xs[outerTripletThirdSegmentAnchorHitIndex];

    float y1 = hitsInGPU.ys[innerTripletFirstSegmentAnchorHitIndex];
    float y2 = hitsInGPU.ys[innerTripletSecondSegmentAnchorHitIndex];
    float y3 = hitsInGPU.ys[innerTripletThirdSegmentAnchorHitIndex];
    float y4 = hitsInGPU.ys[outerTripletSecondSegmentAnchorHitIndex];
    float y5 = hitsInGPU.ys[outerTripletThirdSegmentAnchorHitIndex];

    //for the inner triplet
    float x1Lower = hitsInGPU.xs[firstMDLowerHitIndex];
    float x1Upper = modulesInGPU.subdets[lowerModuleIndex2] == SDL::Barrel and modulesInGPU.sides[lowerModuleIndex1] == SDL::Center ? hitsInGPU.xs[firstMDUpperHitIndex] : mdsInGPU.shiftedXs[firstMDIndex];

    float x2Lower = hitsInGPU.xs[secondMDLowerHitIndex];
    float x2Upper = modulesInGPU.subdets[lowerModuleIndex2] == SDL::Barrel and modulesInGPU.sides[lowerModuleIndex2] == SDL::Center ? hitsInGPU.xs[secondMDUpperHitIndex] : mdsInGPU.shiftedXs[secondMDIndex];

    float x3Lower = hitsInGPU.xs[thirdMDLowerHitIndex];
    float x3Upper = modulesInGPU.subdets[lowerModuleIndex3] == SDL::Barrel and modulesInGPU.sides[lowerModuleIndex3] == SDL::Center ? hitsInGPU.xs[thirdMDUpperHitIndex] : mdsInGPU.shiftedXs[thirdMDIndex];

    float y1Lower = hitsInGPU.ys[firstMDLowerHitIndex];
    float y1Upper = modulesInGPU.subdets[lowerModuleIndex2] == SDL::Barrel and modulesInGPU.sides[lowerModuleIndex1] == SDL::Center ? hitsInGPU.ys[firstMDUpperHitIndex] : mdsInGPU.shiftedYs[firstMDIndex];

    float y2Lower = hitsInGPU.ys[secondMDLowerHitIndex];
    float y2Upper = modulesInGPU.subdets[lowerModuleIndex2] == SDL::Barrel and modulesInGPU.sides[lowerModuleIndex2] == SDL::Center ? hitsInGPU.ys[secondMDUpperHitIndex] : mdsInGPU.shiftedYs[secondMDIndex];

    float y3Lower = hitsInGPU.ys[thirdMDLowerHitIndex];
    float y3Upper = modulesInGPU.subdets[lowerModuleIndex3] == SDL::Barrel and modulesInGPU.sides[lowerModuleIndex3] == SDL::Center ? hitsInGPU.ys[thirdMDUpperHitIndex] : mdsInGPU.shiftedYs[thirdMDIndex];

    float x4Lower = hitsInGPU.xs[fourthMDLowerHitIndex];
    float x4Upper = modulesInGPU.subdets[lowerModuleIndex4] == SDL::Barrel and modulesInGPU.sides[lowerModuleIndex4] == SDL::Center ? hitsInGPU.xs[fourthMDUpperHitIndex] : mdsInGPU.shiftedXs[fourthMDIndex];

    float x5Lower = hitsInGPU.xs[fifthMDLowerHitIndex];
    float x5Upper = modulesInGPU.subdets[lowerModuleIndex5] == SDL::Barrel and modulesInGPU.sides[lowerModuleIndex5] == SDL::Center ? hitsInGPU.xs[fifthMDUpperHitIndex] : mdsInGPU.shiftedXs[fifthMDIndex];

    float y4Lower = hitsInGPU.ys[fourthMDLowerHitIndex];
    float y4Upper = modulesInGPU.subdets[lowerModuleIndex4] == SDL::Barrel and modulesInGPU.sides[lowerModuleIndex4] == SDL::Center ? hitsInGPU.ys[fourthMDUpperHitIndex] : mdsInGPU.shiftedYs[fourthMDIndex];

    float y5Lower = hitsInGPU.ys[fifthMDLowerHitIndex];
    float y5Upper = modulesInGPU.subdets[lowerModuleIndex5] == SDL::Barrel and modulesInGPU.sides[lowerModuleIndex5] == SDL::Center ? hitsInGPU.ys[fifthMDUpperHitIndex] : mdsInGPU.shiftedYs[fifthMDIndex];


    float innerXVec[] = {x1Lower,x1Upper,x2Lower,x2Upper,x3Lower,x3Upper};
    float innerYVec[] = {y1Lower,y1Upper,y2Lower,y2Upper,y3Lower,y3Upper};

    float outerXVec[] = {x3Lower,x3Upper,x4Lower,x4Upper,x5Lower,x5Upper};
    float outerYVec[] = {y3Lower,y3Upper,y4Lower,y4Upper,y5Lower,y5Upper};


/*    //construct the arrays
    float x1Vec[] = {x1, x1, x1};
    float y1Vec[] = {y1, y1, y1};
    float x2Vec[] = {x2, x2, x2};
    float y2Vec[] = {y2, y2, y2};
    float x3Vec[] = {x3, x3, x3};
    float y3Vec[] = {y3, y3, y3};
    float x4Vec[] = {x4, x4, x4};
    float y4Vec[] = {y4, y4, y4};
    float x5Vec[] = {x5, x5, x5};
    float y5Vec[] = {y5, y5, y5};

    if(modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex1] == SDL::TwoS)
    {
        x1Vec[1] = hitsInGPU.lowEdgeXs[innerTripletFirstSegmentAnchorHitIndex];
        x1Vec[2] = hitsInGPU.highEdgeXs[innerTripletFirstSegmentAnchorHitIndex];

        y1Vec[1] = hitsInGPU.lowEdgeYs[innerTripletFirstSegmentAnchorHitIndex];
        y1Vec[2] = hitsInGPU.highEdgeYs[innerTripletFirstSegmentAnchorHitIndex];
    }

    if(modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex2] == SDL::TwoS)
    {
        x2Vec[1] = hitsInGPU.lowEdgeXs[innerTripletSecondSegmentAnchorHitIndex];
        x2Vec[2] = hitsInGPU.highEdgeXs[innerTripletSecondSegmentAnchorHitIndex];

        y2Vec[1] = hitsInGPU.lowEdgeYs[innerTripletSecondSegmentAnchorHitIndex];
        y2Vec[2] = hitsInGPU.highEdgeYs[innerTripletSecondSegmentAnchorHitIndex];

    }

    if(modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex3] == SDL::TwoS)
    {
        x3Vec[1] = hitsInGPU.lowEdgeXs[innerTripletThirdSegmentAnchorHitIndex];
        x3Vec[2] = hitsInGPU.highEdgeXs[innerTripletThirdSegmentAnchorHitIndex];

        y3Vec[1] = hitsInGPU.lowEdgeYs[innerTripletThirdSegmentAnchorHitIndex];
        y3Vec[2] = hitsInGPU.highEdgeYs[innerTripletThirdSegmentAnchorHitIndex];
    }

    if(modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex4] == SDL::TwoS)
    {
        x4Vec[1] = hitsInGPU.lowEdgeXs[outerTripletSecondSegmentAnchorHitIndex];
        x4Vec[2] = hitsInGPU.highEdgeXs[outerTripletSecondSegmentAnchorHitIndex];

        y4Vec[1] = hitsInGPU.lowEdgeYs[outerTripletSecondSegmentAnchorHitIndex];
        y4Vec[2] = hitsInGPU.highEdgeYs[outerTripletSecondSegmentAnchorHitIndex];
    }

    if(modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex4] == SDL::TwoS)
    {
        x5Vec[1] = hitsInGPU.lowEdgeXs[outerTripletThirdSegmentAnchorHitIndex];
        x5Vec[2] = hitsInGPU.highEdgeXs[outerTripletThirdSegmentAnchorHitIndex];

        y5Vec[1] = hitsInGPU.lowEdgeYs[outerTripletThirdSegmentAnchorHitIndex];
        y5Vec[2] = hitsInGPU.highEdgeYs[outerTripletThirdSegmentAnchorHitIndex];
    }*/


    float innerG, innerF; //centers of inner circle
    float innerRadiusMin, innerRadiusMax;
    float outerG, outerF; //centers of outer circle
    float outerRadiusMin, outerRadiusMax;

    innerRadius = computeRadiusFromThreeAnchorHits(x1, y1, x2, y2, x3, y3, innerG, innerF);
    outerRadius = computeRadiusFromThreeAnchorHits(x3, y3, x4, y4, x5, y5, outerG, outerF);

    float innerGFromRegression, innerFFromRegression, outerGFromRegression, outerFFromRegression;
    innerRadiusFromRegression = computeRadiusUsingRegression(3, innerXVec, innerYVec, innerGFromRegression, innerFFromRegression);
    outerRadiusFromRegression = computeRadiusUsingRegression(3, outerXVec, outerYVec, outerGFromRegression, outerFFromRegression);
    
    

//    computeErrorInRadius(x1Vec, y1Vec, x2Vec, y2Vec, x3Vec, y3Vec, innerRadiusMin, innerRadiusMax);
//    computeErrorInRadius(x3Vec, y3Vec, x4Vec, y4Vec, x5Vec, y5Vec, outerRadiusMin, outerRadiusMax);

//    printf("innerRadius = %f, innerRadiusMin = %f, innerRadiusMax = %f, outerRadius = %f, outerRadiusMin = %f, outerRadiusMax = %f\n",innerRadius, innerRadiusMin, innerRadiusMax, outerRadius, outerRadiusMin, outerRadiusMax);

    if(innerRadius < 0)
    {
        pass = false;
    }

    if(outerRadius < 0)
    {
        pass = false;
    }

    //cross product check   

    return pass;
}

__device__ void SDL::computeErrorInRadius(float* x1Vec, float* y1Vec, float* x2Vec, float* y2Vec, float* x3Vec, float* y3Vec, float& minimumRadius, float& maximumRadius)
{
    //brute force
    float candidateRadius;
    minimumRadius = 0;
    maximumRadius = 123456789;
    float f,g; //placeholders
    for(size_t i = 0; i < 3; i++)
    {
        for(size_t j = 0; j < 3; j++)
        {
            for(size_t k = 0; k < 3; k++)
            {
               candidateRadius = computeRadiusFromThreeAnchorHits(x1Vec[i], x2Vec[j], x3Vec[k], y1Vec[i], y2Vec[j], y3Vec[k],g,f); 
               if(candidateRadius >= maximumRadius)
               {
                   maximumRadius = candidateRadius;
               }

               if(candidateRadius <= minimumRadius)
               {
                   minimumRadius = candidateRadius;
               }
    
            }
        }
    }
}


/*
__device__ float SDL::computeErrorInRadius(float* x1Vec, float* y1Vec, float* x2Vec, float* y2Vec, float* x3Vec, float* y3Vec, float& gError, float& fError)
{
    // Numerical differentiation baby! Scientific computing course put into good use
    float h = x1Vec[0]/100;
    float gUp, gDown;
    float fUp, fDown;

    float dgBydx1, dfBydx1;
    float dRBydx1 = (computeRadiusFromThreeAnchorHits(x1Vec[0] + h, y1Vec[0], x2Vec[0], y2Vec[0], x3Vec[0], y3Vec[0], gUp, fUp) - computeRadiusFromThreeAnchorHits(x1Vec[0] - h, y1Vec[0], x2Vec[0], y2Vec[0], x3Vec[0], y3Vec[0], gDown, fDown))/(2 * h);
    dgBydx1 = (gUp - gDown)/(2 * h);
    dfBydx1 = (fUp - fDown)/(2 * h);

    h = x2Vec[0]/100;
    float dgBydx2, dfBydx2;
    float dRBydx2 = (computeRadiusFromThreeAnchorHits(x1Vec[0], y1Vec[0], x2Vec[0] + h, y2Vec[0], x3Vec[0], y3Vec[0], gUp, fUp) - computeRadiusFromThreeAnchorHits(x1Vec[0], y1Vec[0], x2Vec[0] - h, y2Vec[0], x3Vec[0], y3Vec[0], gDown, fDown))/(2 * h);
    dgBydx2 = (gUp - gDown)/(2 * h);
    dfBydx2 = (fUp - fDown)/(2 * h);

    h = x3Vec[0]/100;
    float dgBydx3, dfBydx3;
    float dRBydx3 = (computeRadiusFromThreeAnchorHits(x1Vec[0], y1Vec[0], x2Vec[0], y2Vec[0], x3Vec[0] + h, y3Vec[0], gUp, fUp) - computeRadiusFromThreeAnchorHits(x1Vec[0], y1Vec[0], x2Vec[0], y2Vec[0], x3Vec[0] - h, y3Vec[0], gDown, fDown))/(2 * h);
    dgBydx3 = (gUp - gDown)/(2 * h);
    dfBydx3 = (fUp - fDown)/(2 * h);

    h = y1Vec[0]/100;
    float dgBydy1, dfBydy1;
    float dRBydy1 = (computeRadiusFromThreeAnchorHits(x1Vec[0], y1Vec[0] + h, x2Vec[0], y2Vec[0], x3Vec[0], y3Vec[0], gUp, fUp) - computeRadiusFromThreeAnchorHits(x1Vec[0], y1Vec[0] - h, x2Vec[0], y2Vec[0], x3Vec[0], y3Vec[0], gDown, fDown))/(2 * h);
    dgBydy1 = (gUp - gDown)/(2 * h);
    dfBydy1 = (fUp - fDown)/(2 * h);

    h = y2Vec[0]/100;
    float dgBydy2, dfBydy2;
    float dRBydy2 = (computeRadiusFromThreeAnchorHits(x1Vec[0], y1Vec[0], x2Vec[0], y2Vec[0] + h, x3Vec[0], y3Vec[0], gUp, fUp) - computeRadiusFromThreeAnchorHits(x1Vec[0], y1Vec[0], x2Vec[0], y2Vec[0] - h, x3Vec[0], y3Vec[0], gDown, fDown))/(2 * h);
    dgBydy2 = (gUp - gDown)/(2 * h);
    dfBydy2 = (fUp - fDown)/(2 * h);

    h = y3Vec[0]/100;
    float dgBydy3, dfBydy3;
    float dRBydy3 = (computeRadiusFromThreeAnchorHits(x1Vec[0], y1Vec[0], x2Vec[0], y2Vec[0], x3Vec[0], y3Vec[0] + h, gUp, fUp) - computeRadiusFromThreeAnchorHits(x1Vec[0], y1Vec[0], x2Vec[0], y2Vec[0], x3Vec[0], y3Vec[0] - h, gDown, fDown))/(2 * h);
    dgBydy3 = (gUp - gDown)/(2 * h);
    dfBydy3 = (fUp - fDown)/(2 * h);


    float radiusError2 = (dRBydx1 * dRBydx1) * (x1Vec[2] - x1Vec[1]) * (x1Vec[2] - x1Vec[1]) + (dRBydx2 * dRBydx2) * (x2Vec[2] - x2Vec[1]) * (x2Vec[2] - x2Vec[1]) + (dRBydx3 * dRBydx3) * (x3Vec[2] - x3Vec[1]) * (x3Vec[2] - x3Vec[1]) + (dRBydy1 * dRBydy1) * (y1Vec[2] - y1Vec[1]) * (y1Vec[2] - y1Vec[1]) + (dRBydy2 * dRBydy2) * (y2Vec[2] - y2Vec[1]) * (y2Vec[2] - y2Vec[1]) + (dRBydy3 * dRBydy3) * (y3Vec[2] - y3Vec[1]) * (y3Vec[2] - y3Vec[1]);

    radiusError2/= 4;

    float gError2 = (dgBydx1 * dgBydx1) * (x1Vec[2] - x1Vec[1]) * (x1Vec[2] - x1Vec[1]) + (dgBydx2 * dgBydx2) * (x2Vec[2] - x2Vec[1]) * (x2Vec[2] - x2Vec[1]) + (dgBydx3 * dgBydx3) * (x3Vec[2] - x3Vec[1]) * (x3Vec[2] - x3Vec[1]) + (dgBydy1 * dgBydy1) * (y1Vec[2] - y1Vec[1]) * (y1Vec[2] - y1Vec[1]) + (dgBydy2 * dgBydy2) * (y2Vec[2] - y2Vec[1]) * (y2Vec[2] - y2Vec[1]) + (dgBydy3 * dgBydy3) * (y3Vec[2] - y3Vec[1]) * (y3Vec[2] - y3Vec[1]);

    gError2/= 4;

    float fError2 = (dfBydx1 * dfBydx1) * (x1Vec[2] - x1Vec[1]) * (x1Vec[2] - x1Vec[1]) + (dfBydx2 * dfBydx2) * (x2Vec[2] - x2Vec[1]) * (x2Vec[2] - x2Vec[1]) + (dfBydx3 * dfBydx3) * (x3Vec[2] - x3Vec[1]) * (x3Vec[2] - x3Vec[1]) + (dfBydy1 * dfBydy1) * (y1Vec[2] - y1Vec[1]) * (y1Vec[2] - y1Vec[1]) + (dfBydy2 * dfBydy2) * (y2Vec[2] - y2Vec[1]) * (y2Vec[2] - y2Vec[1]) + (dfBydy3 * dfBydy3) * (y3Vec[2] - y3Vec[1]) * (y3Vec[2] - y3Vec[1]);

    fError2/= 4;

    gError = sqrtf(gError2);
    fError = sqrtf(fError2);

    return sqrtf(radiusError2);
}*/


__device__ float SDL::computeRadiusUsingRegression(int nPoints, float* xs, float* ys, float& g, float& f)
{
    float radius = 0;

    //3 variable linear regression
    //http://faculty.cas.usf.edu/mbrannick/regression/Part3/Reg2.html

    //some extra variables
    //the two variables will be caled x1 and x2, and y (which is x^2 + y^2)
    float sigmaX1Squared = 0.f;
    float sigmaX2Squared = 0.f;
    float sigmaX1X2 = 0.f; 
    float sigmaX1y = 0.f; 
    float sigmaX2y = 0.f;
    float ybar = 0.f;
    float x1bar = 0.f;
    float x2bar = 0.f;

    for(size_t i = 0; i < nPoints; i++)
    {
        sigmaX1Squared += xs[i] * xs[i];
        sigmaX2Squared += ys[i] * ys[i];
        sigmaX1X2 += xs[i] * ys[i];
        sigmaX1y = xs[i] * (xs[i] * xs[i] + ys[i] * ys[i]);
        sigmaX2y = ys[i] * (xs[i] * xs[i] + ys[i] * ys[i]);
        ybar += (xs[i] * xs[i] + ys[i] * ys[i]);
        x1bar += (xs[i]);
        x2bar += (ys[i]);
    }
    ybar /= nPoints;
    x1bar /= nPoints;
    x2bar /= nPoints;
    float twoG = (sigmaX2Squared * sigmaX1y - sigmaX1X2 * sigmaX2y)/(sigmaX1Squared * sigmaX2Squared - sigmaX1X2 * sigmaX1X2);
    float twoF = (sigmaX1Squared * sigmaX2y - sigmaX1X2 * sigmaX1y)/(sigmaX1Squared * sigmaX2Squared - sigmaX1X2 * sigmaX1X2);
    float c = -(ybar - twoG * x1bar * twoF * x2bar);
    g = twoG/2;
    f = twoF/2;
    if(g * g + f * f - c < 0)
    {
        printf("FATAL! r^2 < 0!\n");
        return -1;
    }
    
    radius = sqrtf(g * g  + f * f - c);
    return radius;

}
__device__ float SDL::computeRadiusFromThreeAnchorHits(float x1, float y1, float x2, float y2, float x3, float y3, float& g, float& f)
{
    float radius = 0;

    //writing manual code for computing radius, which obviously sucks
    //TODO:Use fancy inbuilt libraries like cuBLAS or cuSOLVE for this!
    //(g,f) -> center
    //first anchor hit - (x1,y1), second anchor hit - (x2,y2), third anchor hit - (x3, y3)


    if((y1 - y3) * (x2 - x3) - (x1 - x3) * (y2 - y3) == 0)
    {
        return -1; //WTF man three collinear points!
    }

    float denom = ((y1 - y3) * (x2 - x3) - (x1 - x3) * (y2 - y3));

    g = 0.5 * ((y3 - y2) * (x1 * x1 + y1 * y1) + (y1 - y3) * (x2 * x2 + y2 * y2) + (y2 - y1) * (x3 * x3 + y3 * y3))/denom;

    f = 0.5 * ((x2 - x3) * (x1 * x1 + y1 * y1) + (x3 - x1) * (x2 * x2 + y2 * y2) + (x1 - x2) * (x3 * x3 + y3 * y3))/denom;

    float c = ((x2 * y3 - x3 * y2) * (x1 * x1 + y1 * y1) + (x3 * y1 - x1 * y3) * (x2 * x2 + y2 * y2) + (x1 * y2 - x2 * y1) * (x3 * x3 + y3 * y3))/denom;

    if(g * g + f * f - c < 0)
    {
        printf("FATAL! r^2 < 0!\n");
        return -1;
    }
    
    radius = sqrtf(g * g  + f * f - c);
    return radius;
}

__device__ bool SDL::T5HasCommonMiniDoublet(struct SDL::triplets& tripletsInGPU, struct SDL::segments& segmentsInGPU, unsigned int innerTripletIndex, unsigned int outerTripletIndex)
{
    unsigned int innerOuterSegmentIndex = tripletsInGPU.segmentIndices[2 * innerTripletIndex + 1];
    unsigned int outerInnerSegmentIndex = tripletsInGPU.segmentIndices[2 * outerTripletIndex];
    unsigned int innerOuterOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * innerOuterSegmentIndex + 1]; //inner triplet outer segment outer MD index
    unsigned int outerInnerInnerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * outerInnerSegmentIndex]; //outer triplet inner segmnet inner MD index
   

    return (innerOuterOuterMiniDoubletIndex == outerInnerInnerMiniDoubletIndex);
}


