# include "PixelTriplet.cuh"
# include "PixelTracklet.cuh"
#include "allocate.h"
#include "Kernels.cuh"

SDL::pixelTriplets::pixelTriplets()
{
    pixelSegmentIndices = nullptr;
    tripletIndices = nullptr;
    nPixelTriplets = nullptr;
    totOccupancyPixelTriplets = nullptr;
    pixelRadius = nullptr;
    tripletRadius = nullptr;
    pt = nullptr;
    isDup = nullptr;
    partOfPT5 = nullptr;
    centerX = nullptr;
    centerY = nullptr;
    hitIndices = nullptr;
    lowerModuleIndices = nullptr;
    logicalLayers = nullptr;
#ifdef CUT_VALUE_DEBUG
    pixelRadiusError = nullptr;
    rzChiSquared = nullptr;
    rPhiChiSquared = nullptr;
    rPhiChiSquaredInwards = nullptr;
#endif
}

void SDL::pixelTriplets::freeMemoryCache()
{
#ifdef Explicit_PT3
    int dev;
    cudaGetDevice(&dev);
    cms::cuda::free_device(dev,pixelSegmentIndices);
    cms::cuda::free_device(dev,tripletIndices);
    cms::cuda::free_device(dev,nPixelTriplets);
    cms::cuda::free_device(dev,totOccupancyPixelTriplets);
    cms::cuda::free_device(dev,pixelRadius);
    cms::cuda::free_device(dev,tripletRadius);
    cms::cuda::free_device(dev,pt);
    cms::cuda::free_device(dev,isDup);
    cms::cuda::free_device(dev,partOfPT5);
    cms::cuda::free_device(dev, centerX);
    cms::cuda::free_device(dev, centerY);
    cms::cuda::free_device(dev, hitIndices);
    cms::cuda::free_device(dev, logicalLayers);
    cms::cuda::free_device(dev, lowerModuleIndices);
#else
    cms::cuda::free_managed(pixelSegmentIndices);
    cms::cuda::free_managed(tripletIndices);
    cms::cuda::free_managed(nPixelTriplets);
    cms::cuda::free_managed(totOccupancyPixelTriplets);
    cms::cuda::free_managed(pixelRadius);
    cms::cuda::free_managed(tripletRadius);
    cms::cuda::free_managed(pt);
    cms::cuda::free_managed(isDup);
    cms::cuda::free_managed(partOfPT5);
    cms::cuda::free_managed(centerX);
    cms::cuda::free_managed(centerY);
    cms::cuda::free_managed(hitIndices);
    cms::cuda::free_managed(logicalLayers);
    cms::cuda::free_managed(lowerModuleIndices);

#endif
}
void SDL::pixelTriplets::freeMemory(cudaStream_t stream)
{
    cudaFree(pixelSegmentIndices);
    cudaFree(tripletIndices);
    cudaFree(nPixelTriplets);
    cudaFree(totOccupancyPixelTriplets);
    cudaFree(pixelRadius);
    cudaFree(tripletRadius);
    cudaFree(pt);
    cudaFree(isDup);
    cudaFree(partOfPT5);
    cudaFree(centerX);
    cudaFree(centerY);
    cudaFree(logicalLayers);
    cudaFree(hitIndices);
    cudaFree(lowerModuleIndices);
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

void SDL::pixelTriplets::resetMemory(unsigned int maxPixelTriplets,cudaStream_t stream)
{
    cudaMemsetAsync(pixelSegmentIndices,0, maxPixelTriplets * sizeof(unsigned int),stream);
    cudaMemsetAsync(tripletIndices, 0,maxPixelTriplets * sizeof(unsigned int),stream);
    cudaMemsetAsync(nPixelTriplets, 0,sizeof(unsigned int),stream);
    cudaMemsetAsync(totOccupancyPixelTriplets, 0,sizeof(unsigned int),stream);
    cudaMemsetAsync(pixelRadius, 0,maxPixelTriplets * sizeof(FPX),stream);
    cudaMemsetAsync(tripletRadius, 0,maxPixelTriplets * sizeof(FPX),stream);
    cudaMemsetAsync(pt, 0,maxPixelTriplets * 6*sizeof(FPX),stream);
    cudaMemsetAsync(isDup, 0,maxPixelTriplets * sizeof(bool),stream);
    cudaMemsetAsync(partOfPT5, 0,maxPixelTriplets * sizeof(bool),stream);
}
void SDL::createPixelTripletsInUnifiedMemory(struct pixelTriplets& pixelTripletsInGPU, unsigned int maxPixelTriplets,cudaStream_t stream)
{
#ifdef CACHE_ALLOC
//    cudaStream_t stream=0;
    pixelTripletsInGPU.pixelSegmentIndices       =(unsigned int*)cms::cuda::allocate_managed(maxPixelTriplets * sizeof(unsigned int),stream);
    pixelTripletsInGPU.tripletIndices            =(unsigned int*)cms::cuda::allocate_managed(maxPixelTriplets * sizeof(unsigned int),stream);
    pixelTripletsInGPU.nPixelTriplets            =(unsigned int*)cms::cuda::allocate_managed(sizeof(unsigned int),stream);
    pixelTripletsInGPU.totOccupancyPixelTriplets =(unsigned int*)cms::cuda::allocate_managed(sizeof(unsigned int),stream);
    pixelTripletsInGPU.pixelRadius               =(FPX*)cms::cuda::allocate_managed(maxPixelTriplets * sizeof(FPX),stream);
    pixelTripletsInGPU.tripletRadius             =(FPX*)cms::cuda::allocate_managed(maxPixelTriplets * sizeof(FPX),stream);
    pixelTripletsInGPU.pt                        =(FPX*)cms::cuda::allocate_managed(maxPixelTriplets * 6*sizeof(FPX),stream);
    pixelTripletsInGPU.isDup                     =(bool*)cms::cuda::allocate_managed(maxPixelTriplets * sizeof(bool),stream);
    pixelTripletsInGPU.partOfPT5                 =(bool*)cms::cuda::allocate_managed(maxPixelTriplets * sizeof(bool),stream);

    pixelTripletsInGPU.centerX = (FPX*)cms::cuda::allocate_managed(maxPixelTriplets * sizeof(FPX),stream);
    pixelTripletsInGPU.centerY = (FPX*)cms::cuda::allocate_managed(maxPixelTriplets * sizeof(FPX),stream);
    pixelTripletsInGPU.lowerModuleIndices = (uint16_t*)cms::cuda::allocate_managed(maxPixelTriplets * sizeof(uint16_t) * 5, stream);
    pixelTripletsInGPU.hitIndices = (unsigned int*)cms::cuda::allocate_managed(maxPixelTriplets * sizeof(unsigned int) * 10, stream);
    pixelTripletsInGPU.logicalLayers = (uint8_t*)cms::cuda::allocate_managed(maxPixelTriplets * sizeof(uint8_t) * 5, stream);

#else
    cudaMallocManaged(&pixelTripletsInGPU.pixelSegmentIndices, maxPixelTriplets * sizeof(unsigned int));
    cudaMallocManaged(&pixelTripletsInGPU.tripletIndices, maxPixelTriplets * sizeof(unsigned int));
    cudaMallocManaged(&pixelTripletsInGPU.nPixelTriplets, sizeof(unsigned int));
    cudaMallocManaged(&pixelTripletsInGPU.totOccupancyPixelTriplets, sizeof(unsigned int));
    cudaMallocManaged(&pixelTripletsInGPU.pixelRadius, maxPixelTriplets * sizeof(FPX));
    cudaMallocManaged(&pixelTripletsInGPU.tripletRadius, maxPixelTriplets * sizeof(FPX));
    cudaMallocManaged(&pixelTripletsInGPU.pt, maxPixelTriplets * 6*sizeof(FPX));
    cudaMallocManaged(&pixelTripletsInGPU.isDup, maxPixelTriplets * sizeof(bool));
    cudaMallocManaged(&pixelTripletsInGPU.partOfPT5, maxPixelTriplets * sizeof(bool));

    cudaMallocManaged(&pixelTripletsInGPU.centerX, maxPixelTriplets * sizeof(FPX));
    cudaMallocManaged(&pixelTripletsInGPU.centerY, maxPixelTriplets * sizeof(FPX));
    cudaMallocManaged(&pixelTripletsInGPU.logicalLayers, maxPixelTriplets * sizeof(uint8_t) * 5);
    cudaMallocManaged(&pixelTripletsInGPU.hitIndices, maxPixelTriplets * sizeof(unsigned int) * 10);
    cudaMallocManaged(&pixelTripletsInGPU.lowerModuleIndices, maxPixelTriplets * sizeof(uint16_t) * 5);
#ifdef CUT_VALUE_DEBUG
    cudaMallocManaged(&pixelTripletsInGPU.pixelRadiusError, maxPixelTriplets * sizeof(float));
    cudaMallocManaged(&pixelTripletsInGPU.rPhiChiSquared, maxPixelTriplets * sizeof(float));
    cudaMallocManaged(&pixelTripletsInGPU.rPhiChiSquaredInwards, maxPixelTriplets * sizeof(float));
    cudaMallocManaged(&pixelTripletsInGPU.rzChiSquared, maxPixelTriplets * sizeof(float));
#endif
#endif
    pixelTripletsInGPU.eta = pixelTripletsInGPU.pt + maxPixelTriplets;
    pixelTripletsInGPU.phi = pixelTripletsInGPU.pt + maxPixelTriplets * 2;
    pixelTripletsInGPU.eta_pix = pixelTripletsInGPU.pt + maxPixelTriplets *3;
    pixelTripletsInGPU.phi_pix = pixelTripletsInGPU.pt + maxPixelTriplets * 4;
    pixelTripletsInGPU.score = pixelTripletsInGPU.pt + maxPixelTriplets * 5;
    cudaMemsetAsync(pixelTripletsInGPU.nPixelTriplets, 0, sizeof(unsigned int),stream);
    cudaMemsetAsync(pixelTripletsInGPU.totOccupancyPixelTriplets, 0, sizeof(unsigned int),stream);
    cudaMemsetAsync(pixelTripletsInGPU.partOfPT5, 0, maxPixelTriplets*sizeof(bool),stream);
    cudaStreamSynchronize(stream);
}

void SDL::createPixelTripletsInExplicitMemory(struct pixelTriplets& pixelTripletsInGPU, unsigned int maxPixelTriplets, cudaStream_t stream)
{
#ifdef CACHE_ALLOC
//    cudaStream_t stream=0;
    int dev;
    cudaGetDevice(&dev);
    pixelTripletsInGPU.pixelSegmentIndices       =(unsigned int*)cms::cuda::allocate_device(dev,maxPixelTriplets * sizeof(unsigned int),stream);
    pixelTripletsInGPU.tripletIndices            =(unsigned int*)cms::cuda::allocate_device(dev,maxPixelTriplets * sizeof(unsigned int),stream);
    pixelTripletsInGPU.nPixelTriplets            =(unsigned int*)cms::cuda::allocate_device(dev,sizeof(unsigned int),stream);
    pixelTripletsInGPU.totOccupancyPixelTriplets =(unsigned int*)cms::cuda::allocate_device(dev,sizeof(unsigned int),stream);
    pixelTripletsInGPU.pixelRadius               =(FPX*)cms::cuda::allocate_device(dev,maxPixelTriplets * sizeof(FPX),stream);
    pixelTripletsInGPU.tripletRadius             =(FPX*)cms::cuda::allocate_device(dev,maxPixelTriplets * sizeof(FPX),stream);
    pixelTripletsInGPU.pt                        =(FPX*)cms::cuda::allocate_device(dev,maxPixelTriplets * 6*sizeof(FPX),stream);
    pixelTripletsInGPU.isDup                     =(bool*)cms::cuda::allocate_device(dev,maxPixelTriplets * sizeof(bool),stream);
    pixelTripletsInGPU.partOfPT5                 =(bool*)cms::cuda::allocate_device(dev,maxPixelTriplets * sizeof(bool),stream);
    pixelTripletsInGPU.centerX                   = (FPX*)cms::cuda::allocate_device(dev, maxPixelTriplets * sizeof(FPX), stream);
    pixelTripletsInGPU.centerY                   = (FPX*)cms::cuda::allocate_device(dev, maxPixelTriplets * sizeof(FPX), stream);
    pixelTripletsInGPU.lowerModuleIndices        = (uint16_t*)cms::cuda::allocate_device(dev, maxPixelTriplets * sizeof(uint16_t) * 5, stream);
    pixelTripletsInGPU.hitIndices                = (unsigned int*)cms::cuda::allocate_device(dev, maxPixelTriplets * sizeof(unsigned int) * 10, stream);
    pixelTripletsInGPU.logicalLayers             = (uint8_t*)cms::cuda::allocate_device(dev, maxPixelTriplets * sizeof(uint8_t) * 5, stream);
#else
    cudaMalloc(&pixelTripletsInGPU.pixelSegmentIndices, maxPixelTriplets * sizeof(unsigned int));
    cudaMalloc(&pixelTripletsInGPU.tripletIndices, maxPixelTriplets * sizeof(unsigned int));
    cudaMalloc(&pixelTripletsInGPU.nPixelTriplets, sizeof(unsigned int));
    cudaMalloc(&pixelTripletsInGPU.totOccupancyPixelTriplets, sizeof(unsigned int));
    cudaMalloc(&pixelTripletsInGPU.pixelRadius, maxPixelTriplets * sizeof(FPX));
    cudaMalloc(&pixelTripletsInGPU.tripletRadius, maxPixelTriplets * sizeof(FPX));
    cudaMalloc(&pixelTripletsInGPU.pt, maxPixelTriplets * 6*sizeof(FPX));
    cudaMalloc(&pixelTripletsInGPU.isDup, maxPixelTriplets * sizeof(bool));
    cudaMalloc(&pixelTripletsInGPU.partOfPT5, maxPixelTriplets * sizeof(bool));
    cudaMalloc(&pixelTripletsInGPU.centerX, maxPixelTriplets * sizeof(FPX));
    cudaMalloc(&pixelTripletsInGPU.centerY, maxPixelTriplets * sizeof(FPX));
    cudaMalloc(&pixelTripletsInGPU.logicalLayers, maxPixelTriplets * sizeof(uint8_t) * 5);
    cudaMalloc(&pixelTripletsInGPU.hitIndices, maxPixelTriplets * sizeof(unsigned int) * 10);
    cudaMalloc(&pixelTripletsInGPU.lowerModuleIndices, maxPixelTriplets * sizeof(uint16_t) * 5);
#endif
    cudaMemsetAsync(pixelTripletsInGPU.nPixelTriplets, 0, sizeof(unsigned int),stream);
    cudaMemsetAsync(pixelTripletsInGPU.totOccupancyPixelTriplets, 0, sizeof(unsigned int),stream);
    cudaMemsetAsync(pixelTripletsInGPU.partOfPT5, 0, maxPixelTriplets*sizeof(bool),stream);
    cudaStreamSynchronize(stream);

    pixelTripletsInGPU.eta = pixelTripletsInGPU.pt + maxPixelTriplets;
    pixelTripletsInGPU.phi = pixelTripletsInGPU.pt + maxPixelTriplets * 2;
    pixelTripletsInGPU.eta_pix = pixelTripletsInGPU.pt + maxPixelTriplets *3;
    pixelTripletsInGPU.phi_pix = pixelTripletsInGPU.pt + maxPixelTriplets * 4;
    pixelTripletsInGPU.score = pixelTripletsInGPU.pt + maxPixelTriplets * 5;

}

#ifdef CUT_VALUE_DEBUG
__device__ void SDL::addPixelTripletToMemory(struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct triplets& tripletsInGPU, struct pixelTriplets& pixelTripletsInGPU, unsigned int pixelSegmentIndex, unsigned int tripletIndex, float pixelRadius, float pixelRadiusError, float tripletRadius, float centerX, float centerY, float rPhiChiSquared, float rPhiChiSquaredInwards, float rzChiSquared, unsigned int pixelTripletIndex, float pt, float eta, float phi, float eta_pix, float phi_pix, float score)
#else
__device__ void SDL::addPixelTripletToMemory(struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct triplets& tripletsInGPU, struct pixelTriplets& pixelTripletsInGPU, unsigned int pixelSegmentIndex, unsigned int tripletIndex, float pixelRadius, float tripletRadius, float centerX, float centerY, unsigned int pixelTripletIndex, float pt, float eta, float phi, float eta_pix, float phi_pix,float score)
#endif
{
    pixelTripletsInGPU.pixelSegmentIndices[pixelTripletIndex] = pixelSegmentIndex;
    pixelTripletsInGPU.tripletIndices[pixelTripletIndex] = tripletIndex;
    pixelTripletsInGPU.pixelRadius[pixelTripletIndex] = __F2H(pixelRadius);
    pixelTripletsInGPU.tripletRadius[pixelTripletIndex] = __F2H(tripletRadius);
    pixelTripletsInGPU.pt[pixelTripletIndex] = __F2H(pt);
    pixelTripletsInGPU.eta[pixelTripletIndex] = __F2H(eta);
    pixelTripletsInGPU.phi[pixelTripletIndex] = __F2H(phi);
    pixelTripletsInGPU.eta_pix[pixelTripletIndex] = __F2H(eta_pix);
    pixelTripletsInGPU.phi_pix[pixelTripletIndex] = __F2H(phi_pix);
    pixelTripletsInGPU.isDup[pixelTripletIndex] = 0;
    pixelTripletsInGPU.score[pixelTripletIndex] = __F2H(score);

    pixelTripletsInGPU.centerX[pixelTripletIndex] = __F2H(centerX);
    pixelTripletsInGPU.centerY[pixelTripletIndex] = __F2H(centerY);
    pixelTripletsInGPU.logicalLayers[5 * pixelTripletIndex] = 0;
    pixelTripletsInGPU.logicalLayers[5 * pixelTripletIndex + 1] = 0;
    pixelTripletsInGPU.logicalLayers[5 * pixelTripletIndex + 2] = tripletsInGPU.logicalLayers[tripletIndex * 3];
    pixelTripletsInGPU.logicalLayers[5 * pixelTripletIndex + 3] = tripletsInGPU.logicalLayers[tripletIndex * 3 + 1];
    pixelTripletsInGPU.logicalLayers[5 * pixelTripletIndex + 4] = tripletsInGPU.logicalLayers[tripletIndex * 3 + 2];

    pixelTripletsInGPU.lowerModuleIndices[5 * pixelTripletIndex] = segmentsInGPU.innerLowerModuleIndices[pixelSegmentIndex];
    pixelTripletsInGPU.lowerModuleIndices[5 * pixelTripletIndex + 1] = segmentsInGPU.outerLowerModuleIndices[pixelSegmentIndex];
    pixelTripletsInGPU.lowerModuleIndices[5 * pixelTripletIndex + 2] = tripletsInGPU.lowerModuleIndices[3 * tripletIndex];
     pixelTripletsInGPU.lowerModuleIndices[5 * pixelTripletIndex + 3] = tripletsInGPU.lowerModuleIndices[3 * tripletIndex + 1];
      pixelTripletsInGPU.lowerModuleIndices[5 * pixelTripletIndex + 4] = tripletsInGPU.lowerModuleIndices[3 * tripletIndex + 2];
 
    unsigned int pixelInnerMD = segmentsInGPU.mdIndices[2 * pixelSegmentIndex];
    unsigned int pixelOuterMD = segmentsInGPU.mdIndices[2 * pixelSegmentIndex + 1];

    pixelTripletsInGPU.hitIndices[10 * pixelTripletIndex] = mdsInGPU.anchorHitIndices[pixelInnerMD];
    pixelTripletsInGPU.hitIndices[10 * pixelTripletIndex + 1] = mdsInGPU.outerHitIndices[pixelInnerMD];
    pixelTripletsInGPU.hitIndices[10 * pixelTripletIndex + 2] = mdsInGPU.anchorHitIndices[pixelOuterMD];
    pixelTripletsInGPU.hitIndices[10 * pixelTripletIndex + 3] = mdsInGPU.outerHitIndices[pixelOuterMD];

    pixelTripletsInGPU.hitIndices[10 * pixelTripletIndex + 4] = tripletsInGPU.hitIndices[6 * tripletIndex];
    pixelTripletsInGPU.hitIndices[10 * pixelTripletIndex + 5] = tripletsInGPU.hitIndices[6 * tripletIndex + 1];
    pixelTripletsInGPU.hitIndices[10 * pixelTripletIndex + 6] = tripletsInGPU.hitIndices[6 * tripletIndex + 2];
    pixelTripletsInGPU.hitIndices[10 * pixelTripletIndex + 7] = tripletsInGPU.hitIndices[6 * tripletIndex + 3];
    pixelTripletsInGPU.hitIndices[10 * pixelTripletIndex + 8] = tripletsInGPU.hitIndices[6 * tripletIndex + 4];
    pixelTripletsInGPU.hitIndices[10 * pixelTripletIndex + 9] = tripletsInGPU.hitIndices[6 * tripletIndex + 5];
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

__device__ float SDL::computeRadiusFromThreeAnchorHitspT3(float* xs, float* ys, float& g, float& f)
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

    float denomInv = 1.f/((ys[0] - ys[2]) * (xs[1] - xs[2]) - (xs[0] - xs[2]) * (ys[1] - ys[2]));

    float xy1sqr = xs[0] * xs[0] + ys[0] * ys[0];

    float xy2sqr = xs[1] * xs[1] + ys[1] * ys[1];

    float xy3sqr = xs[2] * xs[2] + ys[2] * ys[2];

    g = 0.5f * ((ys[2] - ys[1]) * xy1sqr + (ys[0] - ys[2]) * xy2sqr + (ys[1] - ys[0]) * xy3sqr) * denomInv;

    f = 0.5f * ((xs[1] - xs[2]) * xy1sqr + (xs[2] - xs[0]) * xy2sqr + (xs[0] - xs[1]) * xy3sqr) * denomInv;

    float c = ((xs[1] * ys[2] - xs[2] * ys[1]) * xy1sqr + (xs[2] * ys[0] - xs[0] * ys[2]) * xy2sqr + (xs[0] * ys[1] - xs[1] * ys[0]) * xy3sqr) * denomInv;

    if(((ys[0] - ys[2]) * (xs[1] - xs[2]) - (xs[0] - xs[2]) * (ys[1] - ys[2]) == 0) || (g * g + f * f - c < 0))
    {
        printf("three collinear points or FATAL! r^2 < 0!\n");
  radius = -1;
    }
    else
      radius = sqrtf(g * g  + f * f - c);

    return radius;
}


__device__ bool SDL::runPixelTripletDefaultAlgo(struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct triplets& tripletsInGPU, unsigned int& pixelSegmentIndex, unsigned int tripletIndex, float& pixelRadius, float& pixelRadiusError, float& tripletRadius, float& centerX, float& centerY, float& rzChiSquared, float& rPhiChiSquared, float& rPhiChiSquaredInwards, bool runChiSquaredCuts)
{
    bool pass = true;

    //run pT4 compatibility between the pixel segment and inner segment, and between the pixel and outer segment of the triplet


    //placeholder
    float zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, pt_beta; //temp stuff
    float zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ;

    uint16_t pixelModuleIndex = segmentsInGPU.innerLowerModuleIndices[pixelSegmentIndex];

    uint16_t lowerModuleIndex = tripletsInGPU.lowerModuleIndices[3 * tripletIndex];
    uint16_t middleModuleIndex = tripletsInGPU.lowerModuleIndices[3 * tripletIndex + 1];
    uint16_t upperModuleIndex = tripletsInGPU.lowerModuleIndices[3 * tripletIndex + 2];


    // pixel segment vs inner segment of the triplet
    pass = pass & runPixelTrackletDefaultAlgo(modulesInGPU, mdsInGPU, segmentsInGPU, pixelModuleIndex, lowerModuleIndex, middleModuleIndex, pixelSegmentIndex, tripletsInGPU.segmentIndices[2 * tripletIndex], zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, pt_beta, zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ, N_MAX_SEGMENTS_PER_MODULE);

    //pixel segment vs outer segment of triplet
    pass = pass & runPixelTrackletDefaultAlgo(modulesInGPU, mdsInGPU, segmentsInGPU, pixelModuleIndex, middleModuleIndex, upperModuleIndex, pixelSegmentIndex, tripletsInGPU.segmentIndices[2 * tripletIndex + 1], zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, pt_beta, zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ, N_MAX_SEGMENTS_PER_MODULE);

    //pt matching between the pixel ptin and the triplet circle pt
    unsigned int pixelSegmentArrayIndex = pixelSegmentIndex - (pixelModuleIndex * N_MAX_SEGMENTS_PER_MODULE);
    float pixelSegmentPt = segmentsInGPU.ptIn[pixelSegmentArrayIndex];
    float pixelSegmentPtError = segmentsInGPU.ptErr[pixelSegmentArrayIndex];

    unsigned int pixelInnerMDIndex = segmentsInGPU.mdIndices[2 * pixelSegmentIndex];
    unsigned int pixelOuterMDIndex = segmentsInGPU.mdIndices[2 * pixelSegmentIndex + 1];

    unsigned int pixelAnchorHitIndex1 = mdsInGPU.anchorHitIndices[pixelInnerMDIndex];
    unsigned int pixelNonAnchorHitIndex1 = mdsInGPU.outerHitIndices[pixelInnerMDIndex];
    unsigned int pixelAnchorHitIndex2 = mdsInGPU.anchorHitIndices[pixelOuterMDIndex];
    unsigned int pixelNonAnchorHitIndex2 = mdsInGPU.outerHitIndices[pixelOuterMDIndex];

    pixelRadius = pixelSegmentPt/(2.f * k2Rinv1GeVf);
    pixelRadiusError = pixelSegmentPtError/(2.f * k2Rinv1GeVf);
    unsigned int tripletInnerSegmentIndex = tripletsInGPU.segmentIndices[2 * tripletIndex];
    unsigned int tripletOuterSegmentIndex = tripletsInGPU.segmentIndices[2 * tripletIndex + 1];

    unsigned int firstMDIndex = segmentsInGPU.mdIndices[2 * tripletInnerSegmentIndex];
    unsigned int secondMDIndex = segmentsInGPU.mdIndices[2 * tripletInnerSegmentIndex + 1];
    unsigned int thirdMDIndex = segmentsInGPU.mdIndices[2 * tripletOuterSegmentIndex + 1];

    float x1 = mdsInGPU.anchorX[firstMDIndex];
    float x2 = mdsInGPU.anchorX[secondMDIndex];
    float x3 = mdsInGPU.anchorX[thirdMDIndex];

    float y1 = mdsInGPU.anchorY[firstMDIndex];
    float y2 = mdsInGPU.anchorY[secondMDIndex];
    float y3 = mdsInGPU.anchorY[thirdMDIndex];

    float z1 = mdsInGPU.anchorZ[firstMDIndex];
    float z2 = mdsInGPU.anchorZ[secondMDIndex];
    float z3 = mdsInGPU.anchorZ[thirdMDIndex];

    float rt1 = mdsInGPU.anchorRt[firstMDIndex];
    float rt2 = mdsInGPU.anchorRt[secondMDIndex];
    float rt3 = mdsInGPU.anchorRt[thirdMDIndex];

    float g,f;
    float xs[] = {x1, x2, x3};
    float ys[] = {y1, y2, y3};
    float zs[] = {z1, z2, z3};
    float rts[] = {rt1, rt2, rt3};

    float xPix[] = {mdsInGPU.anchorX[pixelInnerMDIndex], mdsInGPU.anchorX[pixelOuterMDIndex]};
    float yPix[] = {mdsInGPU.anchorY[pixelInnerMDIndex], mdsInGPU.anchorY[pixelOuterMDIndex]};
    float zPix[] = {mdsInGPU.anchorZ[pixelInnerMDIndex], mdsInGPU.anchorZ[pixelOuterMDIndex]};
    float rtPix[] = {mdsInGPU.anchorRt[pixelInnerMDIndex], mdsInGPU.anchorRt[pixelOuterMDIndex]};

    float pixelG = segmentsInGPU.circleCenterX[pixelSegmentArrayIndex];
    float pixelF = segmentsInGPU.circleCenterY[pixelSegmentArrayIndex];
    float pixelRadiusPCA = segmentsInGPU.circleRadius[pixelSegmentArrayIndex];

    tripletRadius = computeRadiusFromThreeAnchorHitspT3(xs, ys, g,f);
    
    pass = pass & passRadiusCriterion(modulesInGPU, pixelRadius, pixelRadiusError, tripletRadius, lowerModuleIndex, middleModuleIndex, upperModuleIndex);

    unsigned int pixelAnchorHits[] = {pixelAnchorHitIndex1, pixelAnchorHitIndex2};
    uint16_t lowerModuleIndices[] = {lowerModuleIndex, middleModuleIndex, upperModuleIndex};


    rzChiSquared = computePT3RZChiSquared(modulesInGPU, lowerModuleIndices, rtPix, zPix, rts, zs);

    rPhiChiSquared = computePT3RPhiChiSquared(modulesInGPU, lowerModuleIndices, pixelG, pixelF, pixelRadiusPCA, xs, ys);

    rPhiChiSquaredInwards = computePT3RPhiChiSquaredInwards(modulesInGPU, g, f, tripletRadius, xPix, yPix);

    if(runChiSquaredCuts and pixelSegmentPt < 5.0f)
    {
        pass = pass & passPT3RZChiSquaredCuts(modulesInGPU, lowerModuleIndex, middleModuleIndex, upperModuleIndex, rzChiSquared);
        pass = pass & passPT3RPhiChiSquaredCuts(modulesInGPU, lowerModuleIndex, middleModuleIndex, upperModuleIndex, rPhiChiSquared);

        pass = pass & passPT3RPhiChiSquaredInwardsCuts(modulesInGPU, lowerModuleIndex, middleModuleIndex, upperModuleIndex, rPhiChiSquaredInwards);
    }


    return pass;

}

__device__ bool SDL::passPT3RPhiChiSquaredInwardsCuts(struct modules& modulesInGPU, uint16_t& lowerModuleIndex1, uint16_t& lowerModuleIndex2, uint16_t& lowerModuleIndex3, float& chiSquared)
{
    const int layer1 = modulesInGPU.layers[lowerModuleIndex1] + 6 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex1] == SDL::TwoS);
    const int layer2 = modulesInGPU.layers[lowerModuleIndex2] + 6 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex2] == SDL::TwoS);
    const int layer3 = modulesInGPU.layers[lowerModuleIndex3] + 6 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex3] == SDL::TwoS);
    
    if(layer1 == 7 and layer2 == 8 and layer3 == 9)
    {
        return chiSquared < 22016.8055f;
    }
    else if(layer1 == 7 and layer2 == 8 and layer3 == 14)
    {
        return chiSquared < 935179.56807f;
    }
    else if(layer1 == 8 and layer2 == 9 and layer3 == 10)
    {
        return chiSquared < 29064.12959f;
    }
    else if(layer1 == 8 and layer2 == 9 and layer3 == 15)
    {
        return chiSquared < 935179.5681f;
    }
    else if(layer1 == 1 and layer2 == 2 and layer3 == 3)
    {
        return chiSquared < 1370.0113195101474f;
    }
    else if(layer1 == 1 and layer2 == 2 and layer3 == 7)
    {
        return chiSquared < 5492.110048314815f;
    }
    else if(layer1 == 2 and layer2 == 3 and layer3 == 4)
    {
        return chiSquared < 4160.410806470067f;
    }
    else if(layer1 == 1 and layer2 == 7 and layer3 == 8)
    {
        return chiSquared < 29064.129591225726f;
    }
    else if(layer1 == 2 and layer2 == 3 and layer3 == 7)
    {
        return chiSquared < 12634.215376250893f;
    }
    else if(layer1 == 2 and layer2 == 3 and layer3 == 12)
    {
        return chiSquared < 353821.69361145404f;
    }
    else if(layer1 == 2 and layer2 == 7 and layer3 == 8)
    {
        return chiSquared < 33393.26076341235f;
    }
    else if(layer1 == 2 and layer2 == 7 and layer3 == 13)
    {
        return chiSquared < 935179.5680742573f;
    }

    return true;
}

__device__ float SDL::computePT3RPhiChiSquaredInwards(struct modules& modulesInGPU, float& g, float& f, float& r, float* xPix, float* yPix)
{
    float chiSquared = 0;
    for(size_t i = 0; i < 2; i++)
    {
        float residual = (xPix[i] - g) * (xPix[i] -g) + (yPix[i] - f) * (yPix[i] - f) - r * r;
        chiSquared += residual * residual;
    }
    //chiSquared /= 2;
    chiSquared *= 0.5f;
    return chiSquared;
}

__device__ bool SDL::passPT3RZChiSquaredCuts(struct modules& modulesInGPU, uint16_t& lowerModuleIndex1, uint16_t& lowerModuleIndex2, uint16_t& lowerModuleIndex3, float& rzChiSquared)
{
    const int layer1 = modulesInGPU.layers[lowerModuleIndex1] + 6 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex1] == SDL::TwoS);
    const int layer2 = modulesInGPU.layers[lowerModuleIndex2] + 6 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex2] == SDL::TwoS);
    const int layer3 = modulesInGPU.layers[lowerModuleIndex3] + 6 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex3] == SDL::TwoS);

    if(layer1 == 8 and layer2 == 9 and layer3 == 10)
    {
        return rzChiSquared < 85.2499f;
    }
    else if(layer1 == 8 and layer2 == 9 and layer3 == 15)
    {
        return rzChiSquared < 85.2499f;
    }

    else if(layer1 == 7 and layer2 == 8 and layer3 == 9)
    {
        return rzChiSquared < 74.19805f;
    }
    else if(layer1 == 7 and layer2 == 8 and layer3 == 14)
    {
        return rzChiSquared < 97.9479f;
    }

    else if(layer1 == 1 and layer2 == 2 and layer3 == 3)
    {
        return rzChiSquared < 451.1407f;
    }
    else if(layer1 == 1 and layer2 == 2 and layer3 == 7)
    {
        return rzChiSquared < 595.546f;
    }

    else if(layer1 == 1 and layer2 == 7 and layer3 == 8)
    {
        return rzChiSquared < 518.339f;
    }

    else if(layer1 == 2 and layer2 == 3 and layer3 == 7)
    {
        return rzChiSquared < 684.253f;
    }
    else if(layer1 == 2 and layer2 == 3 and layer3 == 12)
    {
        return rzChiSquared < 684.253f;
    }
    else if(layer1 == 2 and layer2 == 3 and layer3 == 4)
    {
        return rzChiSquared  < 392.654f;
    }

    else if(layer1 == 2 and layer2 == 7 and layer3 == 8)
    {
        return rzChiSquared < 518.339f;
    }
    else if(layer1 == 2 and layer2 == 7 and layer3 == 13)
    {
        return rzChiSquared < 518.339f;
    }

    //default - category not found!
    return true;
}

__device__ float SDL::computePT3RZChiSquared(struct modules& modulesInGPU, uint16_t* lowerModuleIndices, float* rtPix, float* zPix, float* rts, float* zs)
{ 
    float slope = (zPix[1] - zPix[0])/(rtPix[1] - rtPix[0]);
    float residual = 0;
    float error = 0;
    //hardcoded array indices!!!
    float RMSE = 0;
    for(size_t i = 0; i < 3; i++)
    {
        uint16_t& lowerModuleIndex = lowerModuleIndices[i];
        const int moduleType = modulesInGPU.moduleType[lowerModuleIndex];
        const int moduleSide = modulesInGPU.sides[lowerModuleIndex];
        const int moduleLayerType = modulesInGPU.moduleLayerType[lowerModuleIndex];
        const int layer = modulesInGPU.layers[lowerModuleIndex] + 6 * (modulesInGPU.subdets[lowerModuleIndex] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex] == SDL::TwoS);
        
        residual = (layer <= 6) ?  (zs[i] - zPix[0]) - slope * (rts[i] - rtPix[0]) : (rts[i] - rtPix[0]) - (zs[i] - zPix[0])/slope;
        
        //PS Modules
        if(moduleType == 0)
        {
            error = 0.15f;
        }
        else //2S modules
        {
            error = 5.0f;
        }


        //special dispensation to tilted PS modules!
        if(moduleType == 0 and layer <= 6 and moduleSide != Center)
        {
            float& drdz = modulesInGPU.drdzs[lowerModuleIndex];
            error /= sqrtf(1 + drdz * drdz);
        }
        RMSE += (residual * residual)/(error * error);
    }

    RMSE = sqrtf(0.2f * RMSE); //the constant doesn't really matter....
    return RMSE;
}

//TODO: merge this one and the pT5 function later into a single function
__device__ float SDL::computePT3RPhiChiSquared(struct modules& modulesInGPU, uint16_t* lowerModuleIndices, float& g, float& f, float& radius, float* xs, float* ys)
{
    float delta1[3], delta2[3], slopes[3];
    bool isFlat[3];
    float chiSquared = 0;
    ModuleType moduleType;
    short moduleSubdet, moduleSide;
    ModuleLayerType moduleLayerType;
    float drdz;
    float inv1 = 0.01f/0.009f;
    float inv2 = 0.15f/0.009f;
    float inv3 = 2.4f/0.009f;
    for(size_t i = 0; i < 3; i++)
    {
        moduleType = modulesInGPU.moduleType[lowerModuleIndices[i]];
        moduleSubdet = modulesInGPU.subdets[lowerModuleIndices[i]];
        moduleSide = modulesInGPU.sides[lowerModuleIndices[i]];
        moduleLayerType = modulesInGPU.moduleLayerType[lowerModuleIndices[i]];
        float& drdz = modulesInGPU.drdzs[lowerModuleIndices[i]];
        slopes[i] = modulesInGPU.slopes[lowerModuleIndices[i]];
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

            delta1[i] = inv1;//1.1111f;//0.01;
            isFlat[i] = false;
            delta2[i] = (inv2 * drdz/sqrtf(1 + drdz * drdz));
        }

        //category 4 - endcap PS
        else if(moduleSubdet == Endcap and moduleType == PS)
        {
            delta1[i] = inv1;//1.1111f;//0.01;
            isFlat[i] = false;

            /*despite the type of the module layer of the lower module index,
            all anchor hits are on the pixel side and all non-anchor hits are
            on the strip side!*/        
            delta2[i] = inv2;//16.6666f;//0.15f;
        }

        //category 5 - endcap 2S
        else if(moduleSubdet == Endcap and moduleType == TwoS)
        {
            delta1[i] = 1;//0.009;
            delta2[i] = 500*inv1;//555.5555f;//5.f;
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
        delta1[i] /= 0.009f;
        delta2[i] /= 0.009f;
    }
    chiSquared = computeChiSquared(3, xs, ys, delta1, delta2, slopes, isFlat, g, f, radius);
    
    return chiSquared;
}


//90pc threshold
__device__ bool SDL::passPT3RPhiChiSquaredCuts(struct modules& modulesInGPU, uint16_t& lowerModuleIndex1, uint16_t& lowerModuleIndex2, uint16_t& lowerModuleIndex3, float& chiSquared)
{

    const int layer1 = modulesInGPU.layers[lowerModuleIndex1] + 6 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex1] == SDL::TwoS);
    const int layer2 = modulesInGPU.layers[lowerModuleIndex2] + 6 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex2] == SDL::TwoS);
    const int layer3 = modulesInGPU.layers[lowerModuleIndex3] + 6 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex3] == SDL::TwoS);

    if(layer1 == 8 and layer2 == 9 and layer3 == 10)
    {
        return chiSquared < 7.003f;
    }
    else if(layer1 == 8 and layer2 == 9 and layer3 == 15)
    {
        return chiSquared < 0.5f;
    }

    else if(layer1 == 7 and layer2 == 8 and layer3 == 9)
    {
        return chiSquared < 8.046f;
    }
    else if(layer1 == 7 and layer2 == 8 and layer3 == 14)
    {
        return chiSquared < 0.575f;
    }

    else if(layer1 == 1 and layer2 == 2 and layer3 == 7)
    {
        return chiSquared < 5.304f;
    }
    else if(layer1 == 1 and layer2 == 2 and layer3 == 3)
    {
        return chiSquared < 10.6211f;
    }
    else if(layer1 == 1 and layer2 == 7 and layer3 == 8)
    {
        return chiSquared < 4.617f;
    }

    else if(layer1 == 2 and layer2 == 7 and layer3 == 8)
    {
        return chiSquared < 8.046f;
    }
    else if(layer1 == 2 and layer2 == 7 and layer3 == 13)
    {
        return chiSquared < 0.435f;
    }
    else if(layer1 == 2 and layer2 == 3 and layer3 == 7)
    {
        return chiSquared < 9.244f;
    }
    else if(layer1 == 2 and layer2 == 3 and layer3 == 12)
    {
        return chiSquared < 0.287f;
    }
    else if(layer1 == 2 and layer2 == 3 and layer3 == 4)
    {
        return chiSquared < 18.509f;
    }

    return true;
}

__device__ bool SDL::passRadiusCriterion(struct modules& modulesInGPU, float& pixelRadius, float& pixelRadiusError, float& tripletRadius, uint16_t& lowerModuleIndex, uint16_t& middleModuleIndex, uint16_t& upperModuleIndex)
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
    float tripletInvRadiusErrorBound = 0.15624f;
    float pixelInvRadiusErrorBound = 0.17235f;

    if(pixelRadius > 2.0f/(2.f * k2Rinv1GeVf))
    {
        pixelInvRadiusErrorBound = 0.6375f;
        tripletInvRadiusErrorBound = 0.6588f;
    }

    float tripletRadiusInvMax = (1 + tripletInvRadiusErrorBound)/tripletRadius;
    float tripletRadiusInvMin = fmaxf((1 - tripletInvRadiusErrorBound)/tripletRadius, 0);

    float pixelRadiusInvMax = fmaxf((1 + pixelInvRadiusErrorBound)/pixelRadius, 1.f/(pixelRadius - pixelRadiusError));
    float pixelRadiusInvMin = fminf((1 - pixelInvRadiusErrorBound)/pixelRadius, 1.f/(pixelRadius + pixelRadiusError));

    return checkIntervalOverlap(tripletRadiusInvMin, tripletRadiusInvMax, pixelRadiusInvMin, pixelRadiusInvMax);
}

__device__ bool SDL::passRadiusCriterionBBE(float& pixelRadius, float& pixelRadiusError, float& tripletRadius)
{
    float tripletInvRadiusErrorBound = 0.45972f;
    float pixelInvRadiusErrorBound = 0.19644f;

    if(pixelRadius > 2.0f/(2 * k2Rinv1GeVf))
    {
        pixelInvRadiusErrorBound = 0.6805f;
        tripletInvRadiusErrorBound = 0.8557f;
    }

    float tripletRadiusInvMax = (1 + tripletInvRadiusErrorBound)/tripletRadius;
    float tripletRadiusInvMin = fmaxf((1 - tripletInvRadiusErrorBound)/tripletRadius, 0);

    float pixelRadiusInvMax = fmaxf((1 + pixelInvRadiusErrorBound)/pixelRadius, 1.f/(pixelRadius - pixelRadiusError));
    float pixelRadiusInvMin = fminf((1 - pixelInvRadiusErrorBound)/pixelRadius, 1.f/(pixelRadius + pixelRadiusError));

    return checkIntervalOverlap(tripletRadiusInvMin, tripletRadiusInvMax, pixelRadiusInvMin, pixelRadiusInvMax);

}

__device__ bool SDL::passRadiusCriterionBEE(float& pixelRadius, float& pixelRadiusError, float& tripletRadius)
{
    float tripletInvRadiusErrorBound = 1.59294f;
    float pixelInvRadiusErrorBound = 0.255181f;

    if(pixelRadius > 2.0f/(2 * k2Rinv1GeVf)) //as good as not having selections
    {
        pixelInvRadiusErrorBound = 2.2091f;
        tripletInvRadiusErrorBound = 2.3548f;
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
    float tripletInvRadiusErrorBound = 1.7006f;
    float pixelInvRadiusErrorBound = 0.26367f;

    if(pixelRadius > 2.0f/(2 * k2Rinv1GeVf)) //as good as not having selections
    {
        pixelInvRadiusErrorBound = 2.286f;
        tripletInvRadiusErrorBound = 2.436f;
    }

    float tripletRadiusInvMax = (1 + tripletInvRadiusErrorBound)/tripletRadius;
    float tripletRadiusInvMin = fmaxf((1 - tripletInvRadiusErrorBound)/tripletRadius, 0);

    float pixelRadiusInvMax = fmaxf((1 + pixelInvRadiusErrorBound)/pixelRadius, 1.f/(pixelRadius - pixelRadiusError));
    float pixelRadiusInvMin = fminf((1 - pixelInvRadiusErrorBound)/pixelRadius, 1.f/(pixelRadius + pixelRadiusError));
    pixelRadiusInvMin = fmaxf(0, pixelRadiusInvMin);

    return checkIntervalOverlap(tripletRadiusInvMin, tripletRadiusInvMax, pixelRadiusInvMin, pixelRadiusInvMax);

}

