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
#ifdef TRACK_EXTENSIONS
    centerX = nullptr;
    centerY = nullptr;
    hitIndices = nullptr;
    lowerModuleIndices = nullptr;
    logicalLayers = nullptr;
#endif
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
    cms::cuda::free_device(dev,pixelRadius);
    cms::cuda::free_device(dev,tripletRadius);
    cms::cuda::free_device(dev,pt);
    cms::cuda::free_device(dev,isDup);
    cms::cuda::free_device(dev,partOfPT5);
#ifdef TRACK_EXTENSIONS
    cms::cuda::free_device(dev, centerX);
    cms::cuda::free_device(dev, centerY);
    cms::cuda::free_device(dev, hitIndices);
    cms::cuda::free_device(dev, logicalLayers);
    cms::cuda::free_device(dev, lowerModuleIndices);
#endif
#else
    cms::cuda::free_managed(pixelSegmentIndices);
    cms::cuda::free_managed(tripletIndices);
    cms::cuda::free_managed(nPixelTriplets);
    cms::cuda::free_managed(pixelRadius);
    cms::cuda::free_managed(tripletRadius);
    cms::cuda::free_managed(pt);
    cms::cuda::free_managed(isDup);
    cms::cuda::free_managed(partOfPT5);
#ifdef TRACK_EXTENSIONS
    cms::cuda::free_managed(centerX);
    cms::cuda::free_managed(centerY);
    cms::cuda::free_managed(hitIndices);
    cms::cuda::free_managed(logicalLayers);
    cms::cuda::free_managed(lowerModuleIndices);
#endif

#endif
}
void SDL::pixelTriplets::freeMemory(cudaStream_t stream)
{
    //cudaFreeAsync(pixelSegmentIndices,stream);
    //cudaFreeAsync(tripletIndices,stream);
    //cudaFreeAsync(nPixelTriplets,stream);
    //cudaFreeAsync(pixelRadius,stream);
    //cudaFreeAsync(tripletRadius,stream);
    //cudaFreeAsync(pt,stream);
    //cudaFreeAsync(isDup,stream);
    //cudaFreeAsync(partOfPT5,stream);
    cudaFree(pixelSegmentIndices);
    cudaFree(tripletIndices);
    cudaFree(nPixelTriplets);
    cudaFree(pixelRadius);
    cudaFree(tripletRadius);
    cudaFree(pt);
    cudaFree(isDup);
    cudaFree(partOfPT5);
#ifdef TRACK_EXTENSIONS
    cudaFree(centerX);
    cudaFree(centerY);
    cudaFree(logicalLayers);
    cudaFree(hitIndices);
    cudaFree(lowerModuleIndices);
#endif
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
    cudaMemsetAsync(pixelRadius, 0,maxPixelTriplets * sizeof(float),stream);
    cudaMemsetAsync(tripletRadius, 0,maxPixelTriplets * sizeof(float),stream);
    cudaMemsetAsync(pt, 0,maxPixelTriplets * 6*sizeof(float),stream);
    cudaMemsetAsync(isDup, 0,maxPixelTriplets * sizeof(bool),stream);
    cudaMemsetAsync(partOfPT5, 0,maxPixelTriplets * sizeof(bool),stream);
}
void SDL::createPixelTripletsInUnifiedMemory(struct pixelTriplets& pixelTripletsInGPU, unsigned int maxPixelTriplets,cudaStream_t stream)
{
#ifdef CACHE_ALLOC
//    cudaStream_t stream=0;
    pixelTripletsInGPU.pixelSegmentIndices =(unsigned int*)cms::cuda::allocate_managed(maxPixelTriplets * sizeof(unsigned int),stream);
    pixelTripletsInGPU.tripletIndices      =(unsigned int*)cms::cuda::allocate_managed(maxPixelTriplets * sizeof(unsigned int),stream);
    pixelTripletsInGPU.nPixelTriplets      =(unsigned int*)cms::cuda::allocate_managed(sizeof(unsigned int),stream);
    pixelTripletsInGPU.pixelRadius         =(float*)cms::cuda::allocate_managed(maxPixelTriplets * sizeof(float),stream);
    pixelTripletsInGPU.tripletRadius       =(float*)cms::cuda::allocate_managed(maxPixelTriplets * sizeof(float),stream);
    pixelTripletsInGPU.pt                  =(float*)cms::cuda::allocate_managed(maxPixelTriplets * 6*sizeof(float),stream);
    pixelTripletsInGPU.isDup               =(bool*)cms::cuda::allocate_managed(maxPixelTriplets * sizeof(bool),stream);
    pixelTripletsInGPU.partOfPT5           =(bool*)cms::cuda::allocate_managed(maxPixelTriplets * sizeof(bool),stream);

#ifdef TRACK_EXTENSIONS
    pixelTripletsInGPU.centerX = (float*)cms::cuda::allocate_managed(maxPixelTriplets * sizeof(float),stream);
    pixelTripletsInGPU.centerY = (float*)cms::cuda::allocate_managed(maxPixelTriplets * sizeof(float),stream);
    pixelTripletsInGPU.lowerModuleIndices = (unsigned int*)cms::cuda::allocate_managed(maxPixelTriplets * sizeof(unsigned int) * 5, stream);
    pixelTripletsInGPU.hitIndices = (unsigned int*)cms::cuda::allocate_managed(maxPixelTriplets * sizeof(unsigned int) * 10, stream);
    pixelTripletsInGPU.logicalLayers = (unsigned int*)cms::cuda::allocate_managed(maxPixelTriplets * sizeof(unsigned int) * 5, stream);
#endif

#else
    cudaMallocManaged(&pixelTripletsInGPU.pixelSegmentIndices, maxPixelTriplets * sizeof(unsigned int));
    cudaMallocManaged(&pixelTripletsInGPU.tripletIndices, maxPixelTriplets * sizeof(unsigned int));
    cudaMallocManaged(&pixelTripletsInGPU.nPixelTriplets, sizeof(unsigned int));
    cudaMallocManaged(&pixelTripletsInGPU.pixelRadius, maxPixelTriplets * sizeof(float));
    cudaMallocManaged(&pixelTripletsInGPU.tripletRadius, maxPixelTriplets * sizeof(float));
        cudaMallocManaged(&pixelTripletsInGPU.pt, maxPixelTriplets * 6*sizeof(float));
    cudaMallocManaged(&pixelTripletsInGPU.isDup, maxPixelTriplets * sizeof(bool));
    cudaMallocManaged(&pixelTripletsInGPU.partOfPT5, maxPixelTriplets * sizeof(bool));

#ifdef TRACK_EXTENSIONS
    cudaMallocManaged(&pixelTripletsInGPU.centerX, maxPixelTriplets * sizeof(float));
    cudaMallocManaged(&pixelTripletsInGPU.centerY, maxPixelTriplets * sizeof(float));
    cudaMallocManaged(&pixelTripletsInGPU.logicalLayers, maxPixelTriplets * sizeof(unsigned int) * 5);
    cudaMallocManaged(&pixelTripletsInGPU.hitIndices, maxPixelTriplets * sizeof(unsigned int) * 10);
    cudaMallocManaged(&pixelTripletsInGPU.lowerModuleIndices, maxPixelTriplets * sizeof(unsigned int) * 5);
#endif
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
}

void SDL::createPixelTripletsInExplicitMemory(struct pixelTriplets& pixelTripletsInGPU, unsigned int maxPixelTriplets, cudaStream_t stream)
{
#ifdef CACHE_ALLOC
//    cudaStream_t stream=0;
    int dev;
    cudaGetDevice(&dev);
    pixelTripletsInGPU.pixelSegmentIndices =(unsigned int*)cms::cuda::allocate_device(dev,maxPixelTriplets * sizeof(unsigned int),stream);
    pixelTripletsInGPU.tripletIndices      =(unsigned int*)cms::cuda::allocate_device(dev,maxPixelTriplets * sizeof(unsigned int),stream);
    pixelTripletsInGPU.nPixelTriplets      =(unsigned int*)cms::cuda::allocate_device(dev,sizeof(unsigned int),stream);
    pixelTripletsInGPU.pixelRadius         =(float*)cms::cuda::allocate_device(dev,maxPixelTriplets * sizeof(float),stream);
    pixelTripletsInGPU.tripletRadius       =(float*)cms::cuda::allocate_device(dev,maxPixelTriplets * sizeof(float),stream);
    pixelTripletsInGPU.pt                  =(float*)cms::cuda::allocate_device(dev,maxPixelTriplets * 6*sizeof(float),stream);
    pixelTripletsInGPU.isDup               =(bool*)cms::cuda::allocate_device(dev,maxPixelTriplets * sizeof(bool),stream);
    pixelTripletsInGPU.partOfPT5           =(bool*)cms::cuda::allocate_device(dev,maxPixelTriplets * sizeof(bool),stream);
#ifdef TRACK_EXTENSIONS
    pixelTripletsInGPU.centerX = (float*)cms::cuda::allocate_device(dev, maxPixelTriplets * sizeof(float), stream);
    pixelTripletsInGPU.centerY = (float*)cms::cuda::allocate_device(dev, maxPixelTriplets * sizeof(float), stream);
    pixelTripletsInGPU.lowerModuleIndices = (unsigned int*)cms::cuda::allocate_device(dev, maxPixelTriplets * sizeof(unsigned int) * 5, stream);
    pixelTripletsInGPU.hitIndices = (unsigned int*)cms::cuda::allocate_device(dev, maxPixelTriplets * sizeof(unsigned int) * 10, stream);
    pixelTripletsInGPU.logicalLayers = (unsigned int*)cms::cuda::allocate_device(dev, maxPixelTriplets * sizeof(unsigned int) * 5, stream);
#endif
#else
    cudaMalloc(&pixelTripletsInGPU.pixelSegmentIndices, maxPixelTriplets * sizeof(unsigned int));
    cudaMalloc(&pixelTripletsInGPU.tripletIndices, maxPixelTriplets * sizeof(unsigned int));
    cudaMalloc(&pixelTripletsInGPU.nPixelTriplets, sizeof(unsigned int));
    cudaMalloc(&pixelTripletsInGPU.pixelRadius, maxPixelTriplets * sizeof(float));
    cudaMalloc(&pixelTripletsInGPU.tripletRadius, maxPixelTriplets * sizeof(float));
    cudaMalloc(&pixelTripletsInGPU.pt, maxPixelTriplets * 6*sizeof(float));
    cudaMalloc(&pixelTripletsInGPU.isDup, maxPixelTriplets * sizeof(bool));
    cudaMalloc(&pixelTripletsInGPU.partOfPT5, maxPixelTriplets * sizeof(bool));
#ifdef TRACK_EXTENSIONS
    cudaMalloc(&pixelTripletsInGPU.centerX, maxPixelTriplets * sizeof(float));
    cudaMalloc(&pixelTripletsInGPU.centerY, maxPixelTriplets * sizeof(float));
    cudaMalloc(&pixelTripletsInGPU.logicalLayers, maxPixelTriplets * sizeof(unsigned int) * 5);
    cudaMalloc(&pixelTripletsInGPU.hitIndices, maxPixelTriplets * sizeof(unsigned int) * 10);
    cudaMalloc(&pixelTripletsInGPU.lowerModuleIndices, maxPixelTriplets * sizeof(unsigned int) * 5);
#endif
#endif
    cudaMemsetAsync(pixelTripletsInGPU.nPixelTriplets, 0, sizeof(unsigned int),stream);
    cudaStreamSynchronize(stream);

    pixelTripletsInGPU.eta = pixelTripletsInGPU.pt + maxPixelTriplets;
    pixelTripletsInGPU.phi = pixelTripletsInGPU.pt + maxPixelTriplets * 2;
    pixelTripletsInGPU.eta_pix = pixelTripletsInGPU.pt + maxPixelTriplets *3;
    pixelTripletsInGPU.phi_pix = pixelTripletsInGPU.pt + maxPixelTriplets * 4;
    pixelTripletsInGPU.score = pixelTripletsInGPU.pt + maxPixelTriplets * 5;

}

#ifdef CUT_VALUE_DEBUG
__device__ void SDL::addPixelTripletToMemory(struct modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct triplets& tripletsInGPU, struct pixelTriplets& pixelTripletsInGPU, unsigned int pixelSegmentIndex, unsigned int tripletIndex, float pixelRadius, float pixelRadiusError, float tripletRadius, float centerX, float centerY, float rPhiChiSquared, float rPhiChiSquaredInwards, float rzChiSquared, unsigned int pixelTripletIndex, float pt, float eta, float phi, float eta_pix, float phi_pix, float score)
#else
__device__ void SDL::addPixelTripletToMemory(struct modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct triplets& tripletsInGPU, struct pixelTriplets& pixelTripletsInGPU, unsigned int pixelSegmentIndex, unsigned int tripletIndex, float pixelRadius, float tripletRadius, float centerX, float centerY, unsigned int pixelTripletIndex, float pt, float eta, float phi, float eta_pix, float phi_pix,float score)
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

#ifdef TRACK_EXTENSIONS
    pixelTripletsInGPU.centerX[pixelTripletIndex] = centerX;
    pixelTripletsInGPU.centerY[pixelTripletIndex] = centerY;
    pixelTripletsInGPU.logicalLayers[5 * pixelTripletIndex] = 0;
    pixelTripletsInGPU.logicalLayers[5 * pixelTripletIndex + 1] = 0;
    pixelTripletsInGPU.logicalLayers[5 * pixelTripletIndex + 2] = tripletsInGPU.logicalLayers[tripletIndex * 3];
    pixelTripletsInGPU.logicalLayers[5 * pixelTripletIndex + 3] = tripletsInGPU.logicalLayers[tripletIndex * 3 + 1];
    pixelTripletsInGPU.logicalLayers[5 * pixelTripletIndex + 4] = tripletsInGPU.logicalLayers[tripletIndex * 3 + 2];

    pixelTripletsInGPU.lowerModuleIndices[5 * pixelTripletIndex] = hitsInGPU.moduleIndices[segmentsInGPU.innerMiniDoubletAnchorHitIndices[pixelSegmentIndex]];
    pixelTripletsInGPU.lowerModuleIndices[5 * pixelTripletIndex + 1] = hitsInGPU.moduleIndices[segmentsInGPU.outerMiniDoubletAnchorHitIndices[pixelSegmentIndex]];
    pixelTripletsInGPU.lowerModuleIndices[5 * pixelTripletIndex + 2] = tripletsInGPU.lowerModuleIndices[3 * tripletIndex];
     pixelTripletsInGPU.lowerModuleIndices[5 * pixelTripletIndex + 3] = tripletsInGPU.lowerModuleIndices[3 * tripletIndex + 1];
      pixelTripletsInGPU.lowerModuleIndices[5 * pixelTripletIndex + 4] = tripletsInGPU.lowerModuleIndices[3 * tripletIndex + 2];
 
    unsigned int pixelInnerMD = segmentsInGPU.mdIndices[2 * pixelSegmentIndex];
    unsigned int pixelOuterMD = segmentsInGPU.mdIndices[2 * pixelSegmentIndex + 1];

    pixelTripletsInGPU.hitIndices[10 * pixelTripletIndex] = mdsInGPU.hitIndices[2 * pixelInnerMD];
    pixelTripletsInGPU.hitIndices[10 * pixelTripletIndex + 1] = mdsInGPU.hitIndices[2 * pixelInnerMD + 1];
    pixelTripletsInGPU.hitIndices[10 * pixelTripletIndex + 2] = mdsInGPU.hitIndices[2 * pixelOuterMD];
    pixelTripletsInGPU.hitIndices[10 * pixelTripletIndex + 3] = mdsInGPU.hitIndices[2 * pixelOuterMD + 1];

    pixelTripletsInGPU.hitIndices[10 * pixelTripletIndex + 4] = tripletsInGPU.hitIndices[6 * tripletIndex];
    pixelTripletsInGPU.hitIndices[10 * pixelTripletIndex + 5] = tripletsInGPU.hitIndices[6 * tripletIndex + 1];
    pixelTripletsInGPU.hitIndices[10 * pixelTripletIndex + 6] = tripletsInGPU.hitIndices[6 * tripletIndex + 2];
    pixelTripletsInGPU.hitIndices[10 * pixelTripletIndex + 7] = tripletsInGPU.hitIndices[6 * tripletIndex + 3];
    pixelTripletsInGPU.hitIndices[10 * pixelTripletIndex + 8] = tripletsInGPU.hitIndices[6 * tripletIndex + 4];
    pixelTripletsInGPU.hitIndices[10 * pixelTripletIndex + 9] = tripletsInGPU.hitIndices[6 * tripletIndex + 5];
#endif
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

    float denomInv = 1.f/((y1 - y3) * (x2 - x3) - (x1 - x3) * (y2 - y3));

    float xy1sqr = x1 * x1 + y1 * y1;

    float xy2sqr = x2 * x2 + y2 * y2;

    float xy3sqr = x3 * x3 + y3 * y3;

    g = 0.5f * ((y3 - y2) * xy1sqr + (y1 - y3) * xy2sqr + (y2 - y1) * xy3sqr) * denomInv;

    f = 0.5f * ((x2 - x3) * xy1sqr + (x3 - x1) * xy2sqr + (x1 - x2) * xy3sqr) * denomInv;

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


__device__ bool SDL::runPixelTripletDefaultAlgo(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct triplets& tripletsInGPU, unsigned int& pixelSegmentIndex, unsigned int tripletIndex, float& pixelRadius, float& pixelRadiusError, float& tripletRadius, float& centerX, float& centerY, float& rzChiSquared, float& rPhiChiSquared, float& rPhiChiSquaredInwards, bool runChiSquaredCuts)
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
    pass = pass & runPixelTrackletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, pixelModuleIndex, pixelModuleIndex, lowerModuleIndex, middleModuleIndex, pixelSegmentIndex, tripletsInGPU.segmentIndices[2 * tripletIndex], zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, pt_beta, zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ, 600/*N_MAX_SEGMENTS_PER_MODULE*/);

    //pixel segment vs outer segment of triplet
    pass = pass & runPixelTrackletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, pixelModuleIndex, pixelModuleIndex, middleModuleIndex, upperModuleIndex, pixelSegmentIndex, tripletsInGPU.segmentIndices[2 * tripletIndex + 1], zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, pt_beta, zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ, 600/*N_MAX_SEGMENTS_PER_MODULE*/);

    //pt matching between the pixel ptin and the triplet circle pt
    unsigned int pixelSegmentArrayIndex = pixelSegmentIndex - (pixelModuleIndex * 600);
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
    
    rPhiChiSquared = computePT3RPhiChiSquared(modulesInGPU, hitsInGPU, segmentsInGPU, pixelSegmentArrayIndex, anchorHits, lowerModuleIndices, centerX, centerY);

    rPhiChiSquaredInwards = computePT3RPhiChiSquaredInwards(modulesInGPU, hitsInGPU, tripletRadius, g, f, pixelAnchorHits);

    if(runChiSquaredCuts and pixelSegmentPt < 5.0f)
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
    //chiSquared /= 2;
    chiSquared *= 0.5f;
    return chiSquared;
}

__device__ bool SDL::passPT3RZChiSquaredCuts(struct modules& modulesInGPU, unsigned int lowerModuleIndex1, unsigned int lowerModuleIndex2, unsigned int lowerModuleIndex3, float& rzChiSquared)
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
            error = 0.15f;
        }
        else //2S modules
        {
            error = 5.0f;
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

            error /= sqrtf(1 + drdz * drdz);
        }
        RMSE += (residual * residual)/(error * error);
    }

    RMSE = sqrtf(0.2f * RMSE); //the constant doesn't really matter....
    return RMSE;
}

//TODO: merge this one and the pT5 function later into a single function
__device__ float SDL::computePT3RPhiChiSquared(struct modules& modulesInGPU, struct hits& hitsInGPU, struct segments& segmentsInGPU, unsigned int pixelSegmentArrayIndex, unsigned int* anchorHits, unsigned int* lowerModuleIndices, float& g, float& f)
{
    g = segmentsInGPU.circleCenterX[pixelSegmentArrayIndex];
    f = segmentsInGPU.circleCenterY[pixelSegmentArrayIndex];
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
    float inv1 = 0.01f/0.009f;
    float inv2 = 0.15f/0.009f;
    float inv3 = 2.4f/0.009f;
    for(size_t i = 0; i < 3; i++)
    {
        xs[i] = hitsInGPU.xs[anchorHits[i]];
        ys[i] = hitsInGPU.ys[anchorHits[i]];
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
        delta1[i] /= 0.009f;
        delta2[i] /= 0.009f;
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

