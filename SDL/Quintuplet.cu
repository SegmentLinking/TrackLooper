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

    cudaMallocManaged(&quintupletsInGPU.nQuintuplets, nLowerModules * sizeof(unsigned int));

    quintupletsInGPU.outerTripletPt = quintupletsInGPU.innerTripletPt + nLowerModules * maxQuintuplets;

#pragma omp parallel for
    for(size_t i = 0; i<nLowerModules;i++)
    {
        quintupletsInGPU.nQuintuplets[i] = 0;
    }

}

__device__ void SDL::addQuintupletToMemory(struct SDL::quintuplets& quintupletsInGPU, unsigned int innerTripletIndex, unsigned int outerTripletIndex, unsigned int lowerModule1, unsigned int lowerModule2, unsigned int lowerModule3, unsigned int lowerModule4, unsigned int lowerModule5, float innerTripletPt, float outerTripletPt, unsigned int quintupletIndex)
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

}

__device__ bool SDL::runQuintupletDefaultAlgo(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, unsigned int lowerModuleIndex1, unsigned int lowerModuleIndex2, unsigned int lowerModuleIndex3, unsigned int lowerModuleIndex4, unsigned int lowerModuleIndex5, unsigned int innerTripletIndex, unsigned int outerTripletIndex, float& innerRadius, float& outerRadius)
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

    //radius computation from the three triplet MD anchor hits
    unsigned int innerTripletFirstSegmentAnchorHitIndex = segmentsInGPU.innerMiniDoubletAnchorHitIndices[firstSegmentIndex];
    unsigned int innerTripletSecondSegmentAnchorHitIndex = segmentsInGPU.outerMiniDoubletAnchorHitIndices[firstSegmentIndex]; //same as second segment inner MD anchorhit index
    unsigned int innerTripletThirdSegmentAnchorHitIndex = segmentsInGPU.outerMiniDoubletAnchorHitIndices[secondSegmentIndex]; //same as third segment inner MD anchor hit index

    unsigned int outerTripletFirstSegmentAnchorHitIndex = segmentsInGPU.innerMiniDoubletAnchorHitIndices[thirdSegmentIndex];
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

    float innerG, innerF; //centers of inner circle
    float outerG, outerF; //centers of outer circle

    innerRadius = computeRadiusFromThreeAnchorHits(x1, y1, x2, y2, x3, y3, innerG, innerF);
    outerRadius = computeRadiusFromThreeAnchorHits(x3, y3, x4, y4, x5, y5, outerG, outerF);
    if(innerRadius < 0)
    {
        pass = false;
    }

    if(outerRadius < 0)
    {
        pass = false;
    }

    //cross product check   
    float omega1 = (innerG - x1) * (y3 - y1) - (innerF - y1) * (x3 - x1);
    float omega2 = (outerG - x3) * (y5 - y3) - (outerF - y3) * (x5 - x3);
    if(omega1 * omega2 < 0)
    {
        pass = false;
    } 
    return pass;
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


