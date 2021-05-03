#ifdef __CUDACC__
#define CUDA_CONST_VAR __device__
#endif
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
}

void SDL::createPixelTripletsInExplicitMemory(struct pixelTriplets& pixelTripletsInGPU, unsigned int maxPixelTriplets)
{
    cudaMalloc(&pixelTripletsInGPU.pixelSegmentIndices, maxPixelTriplets * sizeof(float));
    cudaMalloc(&pixelTripletsInGPU.tripletIndices, maxPixelTriplets * sizeof(float));
    cudaMalloc(&pixelTripletsInGPU.nPixelTriplets, sizeof(unsigned int));
}

__device__ void SDL::addPixelTripletToMemory(struct pixelTriplets& pixelTripletsInGPU, unsigned int pixelSegmentIndex, unsigned int tripletIndex, unsigned int pixelTripletIndex)
{
    pixelTripletsInGPU.pixelSegmentIndices[pixelTripletIndex] = pixelSegmentIndex;
    pixelTripletsInGPU.tripletIndices[pixelTripletIndex] = tripletIndex;
}

__device__ bool SDL::runPixelTripletDefaultAlgo(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct triplets& tripletsInGPU, unsigned int& pixelSegmentIndex, unsigned int& tripletIndex)
{
    bool pass = true;

    //run pT4 compatibility between the pixel segment and inner segment, and between the pixel and outer segment of the triplet


    //placeholder
    float zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut;
    unsigned int pixelModuleIndex = segmentsInGPU.innerLowerModuleIndices[pixelSegmentIndex];


    // pixel segment vs inner segment of the triplet
    pass = runTrackletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, pixelModuleIndex, pixelModuleIndex, tripletsInGPU.lowerModuleIndices[3 * tripletIndex], tripletsInGPU.lowerModuleIndices[3 * tripletIndex + 1], pixelSegmentIndex, tripletsInGPU.segmentIndices[2 * tripletIndex], zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut);

    //pixel segment vs outer segment of triplet
    pass = runTrackletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, pixelModuleIndex, pixelModuleIndex, tripletsInGPU.lowerModuleIndices[3 * tripletIndex + 1], tripletsInGPU.lowerModuleIndices[3 * tripletIndex + 2], pixelSegmentIndex, tripletsInGPU.segmentIndices[2 * tripletIndex + 1], zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut);


    //pt matching between the pixel ptin and the triplet circle pt


    return pass;
}
