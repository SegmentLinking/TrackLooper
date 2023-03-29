#ifdef __CUDACC__
#define CUDA_CONST_VAR __device__
#endif
# include "Quintuplet.cuh"
#include "allocate.h"
#include "Kernels.cuh"

SDL::quintuplets::quintuplets()
{
    tripletIndices = nullptr;
    lowerModuleIndices = nullptr;
    nQuintuplets = nullptr;
    totOccupancyQuintuplets = nullptr;
    innerRadius = nullptr;
    outerRadius = nullptr;
    regressionRadius = nullptr;
    isDup = nullptr;
    TightCutFlag = nullptr;
    partOfPT5 = nullptr;
    pt = nullptr;
    layer = nullptr;
    regressionG = nullptr;
    regressionF = nullptr;
    logicalLayers = nullptr;
    hitIndices = nullptr;
    bridgeRadius = nullptr;
    chiSquared = nullptr;
    rzChiSquared = nullptr;
    nonAnchorChiSquared = nullptr;
}

SDL::quintuplets::~quintuplets()
{
}

void SDL::quintuplets::freeMemoryCache()
{
    int dev;
    cudaGetDevice(&dev);
    cms::cuda::free_device(dev,tripletIndices);
    cms::cuda::free_device(dev, lowerModuleIndices);
    cms::cuda::free_device(dev, nQuintuplets);
    cms::cuda::free_device(dev, totOccupancyQuintuplets);
    cms::cuda::free_device(dev, innerRadius);
    cms::cuda::free_device(dev, outerRadius);
    cms::cuda::free_device(dev, partOfPT5);
    cms::cuda::free_device(dev, isDup);
    cms::cuda::free_device(dev, TightCutFlag);
    cms::cuda::free_device(dev, pt);
    cms::cuda::free_device(dev, layer);
    cms::cuda::free_device(dev, regressionG);
    cms::cuda::free_device(dev, regressionF);
    cms::cuda::free_device(dev, regressionRadius);
    cms::cuda::free_device(dev, logicalLayers);
    cms::cuda::free_device(dev, hitIndices);
    cms::cuda::free_device(dev, nMemoryLocations);
    cms::cuda::free_device(dev, bridgeRadius);
    cms::cuda::free_device(dev, rzChiSquared);
    cms::cuda::free_device(dev, chiSquared);
    cms::cuda::free_device(dev, nonAnchorChiSquared);
}

void SDL::quintuplets::freeMemory(cudaStream_t stream)
{
    cudaFree(tripletIndices);
    cudaFree(lowerModuleIndices);
    cudaFree(nQuintuplets);
    cudaFree(totOccupancyQuintuplets);
    cudaFree(innerRadius);
    cudaFree(outerRadius);
    cudaFree(regressionRadius);
    cudaFree(partOfPT5);
    cudaFree(isDup);
    cudaFree(TightCutFlag);
    cudaFree(pt);
    cudaFree(layer);
    cudaFree(regressionG);
    cudaFree(regressionF);
    cudaFree(logicalLayers);
    cudaFree(hitIndices);
    cudaFree(nMemoryLocations);
    cudaFree(bridgeRadius);
    cudaFree(rzChiSquared);
    cudaFree(chiSquared);
    cudaFree(nonAnchorChiSquared);
    cudaStreamSynchronize(stream);
}
//TODO:Reuse the track candidate one instead of this!
__global__ void SDL::createEligibleModulesListForQuintupletsGPU(struct modules& modulesInGPU,struct triplets& tripletsInGPU, unsigned int* device_nTotalQuintuplets, cudaStream_t stream,struct objectRanges& rangesInGPU)
{
    __shared__ int nEligibleT5Modulesx;
    __shared__ unsigned int nTotalQuintupletsx;
    nTotalQuintupletsx = 0; //start!
    nEligibleT5Modulesx = 0;
    __syncthreads();

    unsigned int occupancy;
    unsigned int category_number, eta_number;
    unsigned int layers, subdets, rings;
    float eta;
    //start filling
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int np = gridDim.x * blockDim.x;
    for(uint16_t i = gid; i < *modulesInGPU.nLowerModules; i+= np)
    {
        //condition for a quintuple to exist for a module
        //TCs don't exist for layers 5 and 6 barrel, and layers 2,3,4,5 endcap   
        layers = modulesInGPU.layers[i];
        subdets = modulesInGPU.subdets[i];
        rings = modulesInGPU.rings[i];
        eta = modulesInGPU.eta[i];  
        occupancy = 0;

        if (tripletsInGPU.nTriplets[i] == 0) continue;
        if (subdets == SDL::Barrel and layers >= 3) continue;
        if (subdets == SDL::Endcap and layers > 1) continue;

        int nEligibleT5Modules = atomicAdd(&nEligibleT5Modulesx,1);
        if (nEligibleT5Modules < 0) printf("%u\n",nEligibleT5Modules);
        if (layers<=3 && subdets==5) category_number = 0;
        if (layers>=4 && subdets==5) category_number = 1;
        if (layers<=2 && subdets==4 && rings>=11) category_number = 2;
        if (layers>=3 && subdets==4 && rings>=8) category_number = 2;
        if (layers<=2 && subdets==4 && rings<=10) category_number = 3;
        if (layers>=3 && subdets==4 && rings<=7) category_number = 3;
        if (abs(eta)<0.75) eta_number=0;
        if (abs(eta)>0.75 && abs(eta)<1.5) eta_number=1;
        if (abs(eta)>1.5 && abs(eta)<2.25) eta_number=2;
        if (abs(eta)>2.25 && abs(eta)<3) eta_number=3;

        if (category_number == 0 && eta_number == 0) occupancy = 336;
        if (category_number == 0 && eta_number == 1) occupancy = 414;
        if (category_number == 0 && eta_number == 2) occupancy = 231;
        if (category_number == 0 && eta_number == 3) occupancy = 146;
        if (category_number == 3 && eta_number == 1) occupancy = 0;
        if (category_number == 3 && eta_number == 2) occupancy = 191;
        if (category_number == 3 && eta_number == 3) occupancy = 106;

        unsigned int nTotQ = atomicAdd(&nTotalQuintupletsx,occupancy);
        rangesInGPU.quintupletModuleIndices[i] = nTotQ;
        rangesInGPU.indicesOfEligibleT5Modules[nEligibleT5Modules] = i;
    }
    __syncthreads();
    if(threadIdx.x==0){
        *rangesInGPU.nEligibleT5Modules = static_cast<uint16_t>(nEligibleT5Modulesx);
        *device_nTotalQuintuplets = nTotalQuintupletsx;
    }
}

void SDL::createQuintupletsInExplicitMemory(struct SDL::quintuplets& quintupletsInGPU, const unsigned int& nTotalQuintuplets, const uint16_t& nLowerModules, const uint16_t& nEligibleModules,cudaStream_t stream)
{
    //unsigned int nMemoryLocations = nEligibleModules * maxQuintuplets;
#ifdef CACHE_ALLOC
 //   cudaStream_t stream = 0;
    int dev;
    cudaGetDevice(&dev);
    quintupletsInGPU.tripletIndices = (unsigned int*)cms::cuda::allocate_device(dev, 2 * nTotalQuintuplets * sizeof(unsigned int), stream);
    quintupletsInGPU.lowerModuleIndices = (uint16_t*)cms::cuda::allocate_device(dev, 5 * nTotalQuintuplets * sizeof(uint16_t), stream);
    quintupletsInGPU.nQuintuplets = (unsigned int*)cms::cuda::allocate_device(dev, nLowerModules * sizeof(unsigned int), stream);
    quintupletsInGPU.totOccupancyQuintuplets = (unsigned int*)cms::cuda::allocate_device(dev, nLowerModules * sizeof(unsigned int), stream);
    quintupletsInGPU.innerRadius = (FPX*)cms::cuda::allocate_device(dev, nTotalQuintuplets * sizeof(FPX), stream);
    quintupletsInGPU.outerRadius = (FPX*)cms::cuda::allocate_device(dev, nTotalQuintuplets * sizeof(FPX), stream);
    quintupletsInGPU.bridgeRadius = (FPX*)cms::cuda::allocate_device(dev, nTotalQuintuplets * sizeof(float), stream);

    quintupletsInGPU.pt = (FPX*)cms::cuda::allocate_device(dev, nTotalQuintuplets *4* sizeof(FPX), stream);
    quintupletsInGPU.layer = (uint8_t*)cms::cuda::allocate_device(dev, nTotalQuintuplets * sizeof(uint8_t), stream);
    quintupletsInGPU.isDup = (bool*)cms::cuda::allocate_device(dev, nTotalQuintuplets * sizeof(bool), stream);
    quintupletsInGPU.TightCutFlag = (bool*)cms::cuda::allocate_device(dev, nTotalQuintuplets * sizeof(bool), stream);
    quintupletsInGPU.partOfPT5 = (bool*)cms::cuda::allocate_device(dev, nTotalQuintuplets * sizeof(bool), stream);
    quintupletsInGPU.regressionRadius = (float*)cms::cuda::allocate_device(dev, nTotalQuintuplets * sizeof(float), stream);
    quintupletsInGPU.regressionG = (float*)cms::cuda::allocate_device(dev, nTotalQuintuplets * sizeof(float), stream);
    quintupletsInGPU.regressionF = (float*)cms::cuda::allocate_device(dev, nTotalQuintuplets * sizeof(float), stream);
    quintupletsInGPU.logicalLayers = (uint8_t*)cms::cuda::allocate_device(dev, nTotalQuintuplets * sizeof(uint8_t) * 5, stream);
    quintupletsInGPU.hitIndices = (unsigned int*)cms::cuda::allocate_device(dev, nTotalQuintuplets * sizeof(unsigned int) * 10, stream);
    quintupletsInGPU.nMemoryLocations = (unsigned int*)cms::cuda::allocate_device(dev, sizeof(unsigned int), stream);

    quintupletsInGPU.rzChiSquared = (float*)cms::cuda::allocate_device(dev, nTotalQuintuplets * sizeof(float), stream);
    quintupletsInGPU.chiSquared = (float*)cms::cuda::allocate_device(dev, nTotalQuintuplets * sizeof(float), stream);
    quintupletsInGPU.nonAnchorChiSquared = (float*)cms::cuda::allocate_device(dev, nTotalQuintuplets * sizeof(float), stream);
#else
    cudaMalloc(&quintupletsInGPU.tripletIndices, 2 * nTotalQuintuplets * sizeof(unsigned int));
    cudaMalloc(&quintupletsInGPU.lowerModuleIndices, 5 * nTotalQuintuplets * sizeof(uint16_t));
    cudaMalloc(&quintupletsInGPU.nQuintuplets, nLowerModules * sizeof(unsigned int));
    cudaMalloc(&quintupletsInGPU.totOccupancyQuintuplets, nLowerModules * sizeof(unsigned int));
    cudaMalloc(&quintupletsInGPU.innerRadius, nTotalQuintuplets * sizeof(FPX));
    cudaMalloc(&quintupletsInGPU.outerRadius, nTotalQuintuplets * sizeof(FPX));
    cudaMalloc(&quintupletsInGPU.pt, nTotalQuintuplets *4* sizeof(FPX));
    cudaMalloc(&quintupletsInGPU.isDup, nTotalQuintuplets * sizeof(bool));
    cudaMalloc(&quintupletsInGPU.TightCutFlag, nTotalQuintuplets * sizeof(bool));
    cudaMalloc(&quintupletsInGPU.partOfPT5, nTotalQuintuplets * sizeof(bool));
    cudaMalloc(&quintupletsInGPU.layer, nTotalQuintuplets * sizeof(uint8_t));
    cudaMalloc(&quintupletsInGPU.regressionRadius, nTotalQuintuplets * sizeof(float));
    cudaMalloc(&quintupletsInGPU.regressionG, nTotalQuintuplets * sizeof(float));
    cudaMalloc(&quintupletsInGPU.regressionF, nTotalQuintuplets * sizeof(float));
    cudaMalloc(&quintupletsInGPU.logicalLayers, nTotalQuintuplets * 5 * sizeof(uint8_t));
    cudaMalloc(&quintupletsInGPU.hitIndices, nTotalQuintuplets * 10 * sizeof(unsigned int));
    cudaMalloc(&quintupletsInGPU.nMemoryLocations, sizeof(unsigned int));
    cudaMalloc(&quintupletsInGPU.bridgeRadius, nTotalQuintuplets * sizeof(float));
    cudaMalloc(&quintupletsInGPU.rzChiSquared, nTotalQuintuplets * sizeof(float));
    cudaMalloc(&quintupletsInGPU.chiSquared, nTotalQuintuplets * sizeof(float));
    cudaMalloc(&quintupletsInGPU.nonAnchorChiSquared, nTotalQuintuplets * sizeof(float));
    cudaMalloc(&quintupletsInGPU.rzChiSquared, nTotalQuintuplets * sizeof(float));
    cudaMalloc(&quintupletsInGPU.residual_missing, nTotalQuintuplets * sizeof(float));
    cudaMalloc(&quintupletsInGPU.residual4, nTotalQuintuplets * sizeof(float));
    cudaMalloc(&quintupletsInGPU.residual5, nTotalQuintuplets * sizeof(float));
    cudaMalloc(&quintupletsInGPU.nMemoryLocations, sizeof(unsigned int));
#endif
    cudaMemsetAsync(quintupletsInGPU.nQuintuplets,0,nLowerModules * sizeof(unsigned int),stream);
    cudaMemsetAsync(quintupletsInGPU.totOccupancyQuintuplets,0,nLowerModules * sizeof(unsigned int),stream);
    cudaMemsetAsync(quintupletsInGPU.isDup,0,nTotalQuintuplets * sizeof(bool),stream);
    cudaMemsetAsync(quintupletsInGPU.TightCutFlag,0,nTotalQuintuplets * sizeof(bool),stream);
    cudaMemsetAsync(quintupletsInGPU.partOfPT5,0,nTotalQuintuplets * sizeof(bool),stream);
    cudaStreamSynchronize(stream);
    quintupletsInGPU.eta = quintupletsInGPU.pt + nTotalQuintuplets;
    quintupletsInGPU.phi = quintupletsInGPU.pt + 2*nTotalQuintuplets;
    quintupletsInGPU.score_rphisum = quintupletsInGPU.pt + 3*nTotalQuintuplets;
}


__device__ void SDL::addQuintupletToMemory(struct SDL::triplets& tripletsInGPU, struct SDL::quintuplets& quintupletsInGPU, unsigned int innerTripletIndex, unsigned int outerTripletIndex, uint16_t& lowerModule1, uint16_t& lowerModule2, uint16_t& lowerModule3, uint16_t& lowerModule4, uint16_t& lowerModule5, float& innerRadius, float& bridgeRadius, float& outerRadius, float& regressionG, float& regressionF, float& regressionRadius, float& rzChiSquared, float& rPhiChiSquared, float&
        nonAnchorChiSquared, float pt, float eta, float phi, float scores, uint8_t layer, unsigned int quintupletIndex, bool TightCutFlag)

{
    quintupletsInGPU.tripletIndices[2 * quintupletIndex] = innerTripletIndex;
    quintupletsInGPU.tripletIndices[2 * quintupletIndex + 1] = outerTripletIndex;

    quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex] = lowerModule1;
    quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex + 1] = lowerModule2;
    quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex + 2] = lowerModule3;
    quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex + 3] = lowerModule4;
    quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex + 4] = lowerModule5;
    quintupletsInGPU.innerRadius[quintupletIndex] = __F2H(innerRadius);
    quintupletsInGPU.outerRadius[quintupletIndex] = __F2H(outerRadius);
    quintupletsInGPU.pt[quintupletIndex] = __F2H(pt);
    quintupletsInGPU.eta[quintupletIndex] = __F2H(eta);
    quintupletsInGPU.phi[quintupletIndex] = __F2H(phi);
    quintupletsInGPU.score_rphisum[quintupletIndex] = __F2H(scores);
    quintupletsInGPU.layer[quintupletIndex] = layer;
    quintupletsInGPU.isDup[quintupletIndex] = false;
    quintupletsInGPU.TightCutFlag[quintupletIndex] = TightCutFlag;
    quintupletsInGPU.regressionRadius[quintupletIndex] = regressionRadius;
    quintupletsInGPU.regressionG[quintupletIndex] = regressionG;
    quintupletsInGPU.regressionF[quintupletIndex] = regressionF;
    quintupletsInGPU.logicalLayers[5 * quintupletIndex] = tripletsInGPU.logicalLayers[3 * innerTripletIndex];
    quintupletsInGPU.logicalLayers[5 * quintupletIndex + 1] = tripletsInGPU.logicalLayers[3 * innerTripletIndex + 1];
    quintupletsInGPU.logicalLayers[5 * quintupletIndex + 2] = tripletsInGPU.logicalLayers[3 * innerTripletIndex + 2];
    quintupletsInGPU.logicalLayers[5 * quintupletIndex + 3] = tripletsInGPU.logicalLayers[3 * outerTripletIndex + 1];
    quintupletsInGPU.logicalLayers[5 * quintupletIndex + 4] = tripletsInGPU.logicalLayers[3 * outerTripletIndex + 2];

    quintupletsInGPU.hitIndices[10 * quintupletIndex] = tripletsInGPU.hitIndices[6 * innerTripletIndex];
    quintupletsInGPU.hitIndices[10 * quintupletIndex + 1] = tripletsInGPU.hitIndices[6 * innerTripletIndex + 1];
    quintupletsInGPU.hitIndices[10 * quintupletIndex + 2] = tripletsInGPU.hitIndices[6 * innerTripletIndex + 2];
    quintupletsInGPU.hitIndices[10 * quintupletIndex + 3] = tripletsInGPU.hitIndices[6 * innerTripletIndex + 3];
    quintupletsInGPU.hitIndices[10 * quintupletIndex + 4] = tripletsInGPU.hitIndices[6 * innerTripletIndex + 4];
    quintupletsInGPU.hitIndices[10 * quintupletIndex + 5] = tripletsInGPU.hitIndices[6 * innerTripletIndex + 5];
    quintupletsInGPU.hitIndices[10 * quintupletIndex + 6] = tripletsInGPU.hitIndices[6 * outerTripletIndex + 2];
    quintupletsInGPU.hitIndices[10 * quintupletIndex + 7] = tripletsInGPU.hitIndices[6 * outerTripletIndex + 3];
    quintupletsInGPU.hitIndices[10 * quintupletIndex + 8] = tripletsInGPU.hitIndices[6 * outerTripletIndex + 4];
    quintupletsInGPU.hitIndices[10 * quintupletIndex + 9] = tripletsInGPU.hitIndices[6 * outerTripletIndex + 5];
    quintupletsInGPU.bridgeRadius[quintupletIndex] = bridgeRadius;
    quintupletsInGPU.rzChiSquared[quintupletIndex] = rzChiSquared;
    quintupletsInGPU.chiSquared[quintupletIndex] = rPhiChiSquared;
    quintupletsInGPU.nonAnchorChiSquared[quintupletIndex] = nonAnchorChiSquared;

}

__device__ bool SDL::runQuintupletDefaultAlgo(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, uint16_t& lowerModuleIndex1, uint16_t& lowerModuleIndex2, uint16_t& lowerModuleIndex3, uint16_t& lowerModuleIndex4, uint16_t& lowerModuleIndex5, unsigned int& innerTripletIndex, unsigned int& outerTripletIndex, float& innerRadius, float& outerRadius, float& bridgeRadius, float& regressionG, float& regressionF, float& regressionRadius, float& rzChiSquared, float& chiSquared, float& nonAnchorChiSquared, bool& TightCutFlag)
{
    bool pass = true;
    unsigned int firstSegmentIndex = tripletsInGPU.segmentIndices[2 * innerTripletIndex];
    unsigned int secondSegmentIndex = tripletsInGPU.segmentIndices[2 * innerTripletIndex + 1];
    unsigned int thirdSegmentIndex = tripletsInGPU.segmentIndices[2 * outerTripletIndex];
    unsigned int fourthSegmentIndex = tripletsInGPU.segmentIndices[2 * outerTripletIndex + 1];

    unsigned int innerOuterOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * secondSegmentIndex + 1]; //inner triplet outer segment outer MD index
    unsigned int outerInnerInnerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * thirdSegmentIndex]; //outer triplet inner segmnet inner MD index

    //this cut reduces the number of candidates by a factor of 3, i.e., 2 out of 3 warps can end right here!
    if (innerOuterOuterMiniDoubletIndex != outerInnerInnerMiniDoubletIndex) return false;
    
    //apply T4 criteria between segments 1 and 3
    float zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, pt_beta; //temp stuff
    float zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ;

    unsigned int firstMDIndex = segmentsInGPU.mdIndices[2 * firstSegmentIndex];
    unsigned int secondMDIndex = segmentsInGPU.mdIndices[2 * secondSegmentIndex];
    unsigned int thirdMDIndex = segmentsInGPU.mdIndices[2 * secondSegmentIndex + 1];
    unsigned int fourthMDIndex = segmentsInGPU.mdIndices[2 * thirdSegmentIndex + 1];
    unsigned int fifthMDIndex = segmentsInGPU.mdIndices[2 * fourthSegmentIndex + 1];

    pass = pass and runQuintupletDefaultAlgo(modulesInGPU, mdsInGPU, segmentsInGPU, lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4, firstSegmentIndex, thirdSegmentIndex, firstMDIndex, secondMDIndex, thirdMDIndex, fourthMDIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, pt_beta, zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ);
    if(not pass) return pass;

    pass = pass and runQuintupletDefaultAlgo(modulesInGPU, mdsInGPU, segmentsInGPU, lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex4, lowerModuleIndex5, firstSegmentIndex, fourthSegmentIndex, firstMDIndex, secondMDIndex, fourthMDIndex, fifthMDIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, pt_beta, zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ);
    if(not pass) return pass;

    pass = pass and runQuintupletDefaultAlgo(modulesInGPU, mdsInGPU, segmentsInGPU, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex3, lowerModuleIndex4, secondSegmentIndex, thirdSegmentIndex, secondMDIndex, thirdMDIndex, thirdMDIndex, fourthMDIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, pt_beta, zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ);
    if(not pass) return pass;
    
    pass = pass and runQuintupletDefaultAlgo(modulesInGPU, mdsInGPU, segmentsInGPU, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4, lowerModuleIndex5, secondSegmentIndex, fourthSegmentIndex, secondMDIndex, thirdMDIndex, fourthMDIndex, fifthMDIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, pt_beta, zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ);
    if(not pass) return pass;

    float x1 = mdsInGPU.anchorX[firstMDIndex];
    float x2 = mdsInGPU.anchorX[secondMDIndex];
    float x3 = mdsInGPU.anchorX[thirdMDIndex];
    float x4 = mdsInGPU.anchorX[fourthMDIndex];
    float x5 = mdsInGPU.anchorX[fifthMDIndex];
    
    float y1 = mdsInGPU.anchorY[firstMDIndex];
    float y2 = mdsInGPU.anchorY[secondMDIndex];
    float y3 = mdsInGPU.anchorY[thirdMDIndex];
    float y4 = mdsInGPU.anchorY[fourthMDIndex];
    float y5 = mdsInGPU.anchorY[fifthMDIndex];

    //construct the arrays
    float x1Vec[] = {x1, x1, x1};
    float y1Vec[] = {y1, y1, y1};
    float x2Vec[] = {x2, x2, x2};
    float y2Vec[] = {y2, y2, y2};
    float x3Vec[] = {x3, x3, x3};
    float y3Vec[] = {y3, y3, y3};

    if(modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex1] == SDL::TwoS)
    {
        x1Vec[1] = mdsInGPU.anchorLowEdgeX[firstMDIndex];
        x1Vec[2] = mdsInGPU.anchorHighEdgeX[firstMDIndex];

        y1Vec[1] = mdsInGPU.anchorLowEdgeY[firstMDIndex];
        y1Vec[2] = mdsInGPU.anchorHighEdgeY[firstMDIndex];
    }
    if(modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex2] == SDL::TwoS)
    {
        x2Vec[1] = mdsInGPU.anchorLowEdgeX[secondMDIndex];
        x2Vec[2] = mdsInGPU.anchorHighEdgeX[secondMDIndex];

        y2Vec[1] = mdsInGPU.anchorLowEdgeY[secondMDIndex];
        y2Vec[2] = mdsInGPU.anchorHighEdgeY[secondMDIndex];
    }
    if(modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex3] == SDL::TwoS)
    {
        x3Vec[1] = mdsInGPU.anchorLowEdgeX[thirdMDIndex];
        x3Vec[2] = mdsInGPU.anchorHighEdgeX[thirdMDIndex];

        y3Vec[1] = mdsInGPU.anchorLowEdgeY[thirdMDIndex];
        y3Vec[2] = mdsInGPU.anchorHighEdgeY[thirdMDIndex];
    }

    float innerRadiusMin2S, innerRadiusMax2S;
    computeErrorInRadius(x1Vec, y1Vec, x2Vec, y2Vec, x3Vec, y3Vec, innerRadiusMin2S, innerRadiusMax2S);

    for (int i=0; i<3; i++) 
    {
      x1Vec[i] = x4;
      y1Vec[i] = y4;
    }
    if(modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex4] == SDL::TwoS)
    {
        x1Vec[1] = mdsInGPU.anchorLowEdgeX[fourthMDIndex];
        x1Vec[2] = mdsInGPU.anchorHighEdgeX[fourthMDIndex];

        y1Vec[1] = mdsInGPU.anchorLowEdgeY[fourthMDIndex];
        y1Vec[2] = mdsInGPU.anchorHighEdgeY[fourthMDIndex];
    }

    float bridgeRadiusMin2S, bridgeRadiusMax2S;
    computeErrorInRadius(x2Vec, y2Vec, x3Vec, y3Vec, x1Vec, y1Vec, bridgeRadiusMin2S, bridgeRadiusMax2S);

    for(int i=0; i<3; i++) 
    {
      x2Vec[i] = x5;
      y2Vec[i] = y5;
    }
    if(modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex5] == SDL::TwoS)
    {
        x2Vec[1] = mdsInGPU.anchorLowEdgeX[fifthMDIndex];
        x2Vec[2] = mdsInGPU.anchorHighEdgeX[fifthMDIndex];

        y2Vec[1] = mdsInGPU.anchorLowEdgeY[fifthMDIndex];
        y2Vec[2] = mdsInGPU.anchorHighEdgeY[fifthMDIndex];
    }

    float outerRadiusMin2S, outerRadiusMax2S;
    computeErrorInRadius(x3Vec, y3Vec, x1Vec, y1Vec, x2Vec, y2Vec, outerRadiusMin2S, outerRadiusMax2S);

    float g, f;
    outerRadius = computeRadiusFromThreeAnchorHits(x3, y3, x4, y4, x5, y5, g, f);
    bridgeRadius = computeRadiusFromThreeAnchorHits(x2, y2, x3, y3, x4, y4, g, f);
    innerRadius = computeRadiusFromThreeAnchorHits(x1, y1, x2, y2, x3, y3, g, f);

    float inner_pt = 2 * k2Rinv1GeVf * innerRadius;
//    pass = pass and passT5RZConstraint(modulesInGPU, mdsInGPU, firstMDIndex, secondMDIndex, thirdMDIndex, fourthMDIndex, fifthMDIndex, lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4, lowerModuleIndex5, rzChiSquared, residual_missing, residual4, residual5, inner_pt, innerRadius, g, f, TightCutFlag);
    passT5RZConstraint(modulesInGPU, mdsInGPU, firstMDIndex, secondMDIndex, thirdMDIndex, fourthMDIndex, fifthMDIndex, lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4, lowerModuleIndex5, rzChiSquared, residual_missing, residual4, residual5, inner_pt, innerRadius, g, f, TightCutFlag);

    if(not pass) return pass;

    pass = pass & (innerRadius >= 0.95f * ptCut/(2.f * k2Rinv1GeVf));

    float innerInvRadiusMin, innerInvRadiusMax, bridgeInvRadiusMin, bridgeInvRadiusMax, outerInvRadiusMin, outerInvRadiusMax;

    //split by category
    bool tempPass;
    if(modulesInGPU.subdets[lowerModuleIndex1] == SDL::Barrel and modulesInGPU.subdets[lowerModuleIndex2] == SDL::Barrel and modulesInGPU.subdets[lowerModuleIndex3] == SDL::Barrel and modulesInGPU.subdets[lowerModuleIndex4] == SDL::Barrel and modulesInGPU.subdets[lowerModuleIndex5] == SDL::Barrel)
    {
       tempPass = matchRadiiBBBBB(innerRadius, bridgeRadius, outerRadius, innerInvRadiusMin, innerInvRadiusMax, bridgeInvRadiusMin, bridgeInvRadiusMax, outerInvRadiusMin, outerInvRadiusMax);
    }
    else if(modulesInGPU.subdets[lowerModuleIndex1] == SDL::Barrel and modulesInGPU.subdets[lowerModuleIndex2] == SDL::Barrel and modulesInGPU.subdets[lowerModuleIndex3] == SDL::Barrel and modulesInGPU.subdets[lowerModuleIndex4] == SDL::Barrel and modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap)
    {
        tempPass = matchRadiiBBBBE(innerRadius, bridgeRadius, outerRadius, innerRadiusMin2S, innerRadiusMax2S, bridgeRadiusMin2S, bridgeRadiusMax2S, outerRadiusMin2S, outerRadiusMax2S, innerInvRadiusMin, innerInvRadiusMax, bridgeInvRadiusMin, bridgeInvRadiusMax, outerInvRadiusMin, outerInvRadiusMax);
    }
    else if(modulesInGPU.subdets[lowerModuleIndex1] == SDL::Barrel and modulesInGPU.subdets[lowerModuleIndex2] == SDL::Barrel and modulesInGPU.subdets[lowerModuleIndex3] == SDL::Barrel and modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap and modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap)
    {
        if(modulesInGPU.layers[lowerModuleIndex1] == 1)
        {
            tempPass = matchRadiiBBBEE12378(innerRadius, bridgeRadius, outerRadius,innerRadiusMin2S, innerRadiusMax2S, bridgeRadiusMin2S, bridgeRadiusMax2S, outerRadiusMin2S, outerRadiusMax2S, innerInvRadiusMin, innerInvRadiusMax, bridgeInvRadiusMin, bridgeInvRadiusMax, outerInvRadiusMin, outerInvRadiusMax);
        }
        else if(modulesInGPU.layers[lowerModuleIndex1] == 2)
        {
            tempPass = matchRadiiBBBEE23478(innerRadius, bridgeRadius, outerRadius,innerRadiusMin2S, innerRadiusMax2S, bridgeRadiusMin2S, bridgeRadiusMax2S, outerRadiusMin2S, outerRadiusMax2S, innerInvRadiusMin, innerInvRadiusMax, bridgeInvRadiusMin, bridgeInvRadiusMax, outerInvRadiusMin, outerInvRadiusMax);
        }
        else
        {
            tempPass = matchRadiiBBBEE34578(innerRadius, bridgeRadius, outerRadius,innerRadiusMin2S, innerRadiusMax2S, bridgeRadiusMin2S, bridgeRadiusMax2S, outerRadiusMin2S, outerRadiusMax2S, innerInvRadiusMin, innerInvRadiusMax, bridgeInvRadiusMin, bridgeInvRadiusMax, outerInvRadiusMin, outerInvRadiusMax);
        }
    }

    else if(modulesInGPU.subdets[lowerModuleIndex1] == SDL::Barrel and modulesInGPU.subdets[lowerModuleIndex2] == SDL::Barrel and modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap and modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap and modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap)
    {
        tempPass = matchRadiiBBEEE(innerRadius, bridgeRadius, outerRadius, innerRadiusMin2S, innerRadiusMax2S, bridgeRadiusMin2S, bridgeRadiusMax2S, outerRadiusMin2S, outerRadiusMax2S, innerInvRadiusMin, innerInvRadiusMax, bridgeInvRadiusMin, bridgeInvRadiusMax, outerInvRadiusMin, outerInvRadiusMax);
    }
    else if(modulesInGPU.subdets[lowerModuleIndex1] == SDL::Barrel and modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap and modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap and modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap and modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap)
    {
        tempPass = matchRadiiBEEEE(innerRadius, bridgeRadius, outerRadius, innerRadiusMin2S, innerRadiusMax2S, bridgeRadiusMin2S, bridgeRadiusMax2S, outerRadiusMin2S, outerRadiusMax2S, innerInvRadiusMin, innerInvRadiusMax, bridgeInvRadiusMin, bridgeInvRadiusMax, outerInvRadiusMin, outerInvRadiusMax);
    }
    else
    {
        tempPass = matchRadiiEEEEE(innerRadius, bridgeRadius, outerRadius, innerRadiusMin2S, innerRadiusMax2S, bridgeRadiusMin2S, bridgeRadiusMax2S, outerRadiusMin2S, outerRadiusMax2S,innerInvRadiusMin, innerInvRadiusMax, bridgeInvRadiusMin, bridgeInvRadiusMax, outerInvRadiusMin, outerInvRadiusMax);
    }

    //compute regression radius right here - this computation is expensive!!!
//    pass = pass and tempPass;
    if(not pass) return pass;

    float xVec[] = {x1, x2, x3, x4, x5};
    float yVec[] = {y1, y2, y3, y4, y5};
    float sigmas[5], delta1[5], delta2[5], slopes[5];
    bool isFlat[5];
    //5 categories for sigmas
    const uint16_t lowerModuleIndices[] = {lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4, lowerModuleIndex5};

    computeSigmasForRegression(modulesInGPU, lowerModuleIndices, delta1, delta2, slopes, isFlat);
    regressionRadius = computeRadiusUsingRegression(5,xVec, yVec, delta1, delta2, slopes, isFlat, regressionG, regressionF, sigmas, chiSquared);

    //extra chi squared cuts!
    if(regressionRadius < 5.0f/(2.f * k2Rinv1GeVf))
    {
//        pass = pass and passChiSquaredConstraint(modulesInGPU, lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4, lowerModuleIndex5, chiSquared);
        if(not pass) return pass;
    }

    //compute the other chisquared
    //non anchor is always shifted for tilted and endcap!
    float nonAnchorDelta1[5], nonAnchorDelta2[5], nonAnchorSlopes[5];
    float nonAnchorxs[] = { mdsInGPU.outerX[firstMDIndex], mdsInGPU.outerX[secondMDIndex], mdsInGPU.outerX[thirdMDIndex], mdsInGPU.outerX[fourthMDIndex], mdsInGPU.outerX[fifthMDIndex]};
    float nonAnchorys[] = { mdsInGPU.outerY[firstMDIndex], mdsInGPU.outerY[secondMDIndex], mdsInGPU.outerY[thirdMDIndex], mdsInGPU.outerY[fourthMDIndex], mdsInGPU.outerY[fifthMDIndex]};

    computeSigmasForRegression(modulesInGPU, lowerModuleIndices, nonAnchorDelta1, nonAnchorDelta2, nonAnchorSlopes, isFlat, 5, false);
    nonAnchorChiSquared = computeChiSquared(5, nonAnchorxs, nonAnchorys, nonAnchorDelta1, nonAnchorDelta2, nonAnchorSlopes, isFlat, regressionG, regressionF, regressionRadius);
    return pass;
}

//90% constraint
__device__ bool SDL::passChiSquaredConstraint(struct SDL::modules& modulesInGPU, uint16_t& lowerModuleIndex1, uint16_t& lowerModuleIndex2, uint16_t& lowerModuleIndex3, uint16_t& lowerModuleIndex4, uint16_t& lowerModuleIndex5, float& chiSquared)
{
    //following Philip's layer number prescription
    const int layer1 = modulesInGPU.layers[lowerModuleIndex1] + 6 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex1] == SDL::TwoS);
    const int layer2 = modulesInGPU.layers[lowerModuleIndex2] + 6 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex2] == SDL::TwoS);
    const int layer3 = modulesInGPU.layers[lowerModuleIndex3] + 6 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex3] == SDL::TwoS);
    const int layer4 = modulesInGPU.layers[lowerModuleIndex4] + 6 * (modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex4] == SDL::TwoS);
    const int layer5 = modulesInGPU.layers[lowerModuleIndex5] + 6 * (modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex5] == SDL::TwoS);

    if(layer1 == 7 and layer2 == 8 and layer3 == 9)
    {
        if(layer4 == 10 and layer5 == 11)
        {
            return chiSquared < 0.01788f;
        }
        else if(layer4 == 10 and layer5 == 16)
        {
            return chiSquared < 0.04725f;
        }
        else if(layer4 == 15 and layer5 == 16)
        {
            return chiSquared < 0.04725f;
        }
    }
    else if(layer1 == 1 and layer2 == 7 and layer3 == 8)
    {
        if(layer4 == 9 and layer5 == 10)
        {       
            return chiSquared < 0.01788f;
        }   
        else if(layer4 == 9 and layer5 == 15)
        {
            return chiSquared < 0.08234f;
        }
    }
    else if(layer1 == 1 and layer2 == 2 and layer3 == 7)
    {
        if(layer4 == 8 and layer5 == 9)
        {
            return chiSquared < 0.02360f;
        }
        else if(layer4 == 8 and layer5 == 14)
        {
            return chiSquared < 0.07167f;
        }
        else if(layer4 == 13 and layer5 == 14)
        {   
            return chiSquared < 0.08234f;
        }
    }
    else if(layer1 == 1 and layer2 == 2 and layer3 == 3)
    {
        if(layer4 == 7 and layer5 == 8)
        {
            return chiSquared < 0.01026f;
        }
        else if(layer4 == 7 and layer5 == 13)
        {
            return chiSquared < 0.06238f;
        }
        else if(layer4 == 12 and layer5 == 13)
        {
            return chiSquared < 0.06238f;
        }
    }
    else if(layer1 == 1 and layer2 == 2 and layer3 == 3 and layer4 == 4)
    {
        if(layer5 == 5)
        {
            return chiSquared < 0.04725f;
        }
        else if(layer5 == 12)
        {
            return chiSquared < 0.09461f;
        }
    }
    else if(layer1 == 2 and layer2 == 7 and layer3 == 8)
    {
        if(layer4 == 9 and layer5 == 10)
        {
            return chiSquared < 0.00512f;
        }
        if(layer4 == 9 and layer5 == 15)
        {
            return chiSquared < 0.04112f;
        }
        else if(layer4 == 14 and layer5 == 15)
        {
            return chiSquared < 0.06238f;
        }
    }
    else if(layer1 == 2 and layer2 == 3 and layer3 == 7)
    {
        if(layer4 == 8 and layer5 == 14)
        {
            return chiSquared < 0.07167f;
        }
        else if(layer4 == 13 and layer5 == 14)
        {
            return chiSquared < 0.06238f;
        }
    }
    else if(layer1 == 2 and layer2 == 3 and layer3 == 4)
    {
        if(layer4 == 5 and layer5 == 6)
        {
            return chiSquared < 0.08234f;
        }
        else if(layer4 == 5 and layer5 == 12)
        {
            return chiSquared < 0.10870f;
        }
        else if(layer4 == 12 and layer5 == 13)
        {
            return chiSquared < 0.10870f;
        }
    }
    else if(layer1 == 3 and layer2 == 7 and layer3 == 8 and layer4 == 14 and layer5 == 15)
    {
        return chiSquared < 0.09461f;
    }
    else if(layer1 == 3 and layer2 == 4 and layer3 == 5 and layer4 == 12 and layer5 == 13)
    {
        return chiSquared < 0.09461f;
    }

    return true;
}

//bounds can be found at http://uaf-10.t2.ucsd.edu/~bsathian/SDL/T5_RZFix/t5_rz_thresholds.txt
__device__ bool SDL::passT5RZConstraint(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, unsigned int firstMDIndex, unsigned int secondMDIndex, unsigned int thirdMDIndex, unsigned int fourthMDIndex, unsigned int fifthMDIndex, uint16_t& lowerModuleIndex1, uint16_t& lowerModuleIndex2, uint16_t& lowerModuleIndex3, uint16_t& lowerModuleIndex4, uint16_t& lowerModuleIndex5, float& rzChiSquared, float& residual_missing, float& residual4, float& residual5, float inner_pt, float innerRadius, float g, float f, bool& TightCutFlag) 
{
    //(g,f) is the center of the circle fitted by the innermost 3 points on x,y coordinates
    const float& rt1 = mdsInGPU.anchorRt[firstMDIndex]/100; //in the unit of m instead of cm
    const float& rt2 = mdsInGPU.anchorRt[secondMDIndex]/100;
    const float& rt3 = mdsInGPU.anchorRt[thirdMDIndex]/100;
    const float& rt4 = mdsInGPU.anchorRt[fourthMDIndex]/100;
    const float& rt5 = mdsInGPU.anchorRt[fifthMDIndex]/100;

    const float& z1 = mdsInGPU.anchorZ[firstMDIndex]/100;
    const float& z2 = mdsInGPU.anchorZ[secondMDIndex]/100;
    const float& z3 = mdsInGPU.anchorZ[thirdMDIndex]/100;
    const float& z4 = mdsInGPU.anchorZ[fourthMDIndex]/100;
    const float& z5 = mdsInGPU.anchorZ[fifthMDIndex]/100;

    //following Philip's layer number prescription
    const int layer1 = modulesInGPU.layers[lowerModuleIndex1] + 6 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex1] == SDL::TwoS);
    const int layer2 = modulesInGPU.layers[lowerModuleIndex2] + 6 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex2] == SDL::TwoS);
    const int layer3 = modulesInGPU.layers[lowerModuleIndex3] + 6 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex3] == SDL::TwoS);
    const int layer4 = modulesInGPU.layers[lowerModuleIndex4] + 6 * (modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex4] == SDL::TwoS);
    const int layer5 = modulesInGPU.layers[lowerModuleIndex5] + 6 * (modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap) + 5 * (modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap and modulesInGPU.moduleType[lowerModuleIndex5] == SDL::TwoS);

    //slope computed using the internal T3s
    const int moduleType1 = modulesInGPU.moduleType[lowerModuleIndex1]; //0 is ps, 1 is 2s
    const int moduleType2 = modulesInGPU.moduleType[lowerModuleIndex2];
    const int moduleType3 = modulesInGPU.moduleType[lowerModuleIndex3];
    const int moduleType4 = modulesInGPU.moduleType[lowerModuleIndex4];
    const int moduleType5 = modulesInGPU.moduleType[lowerModuleIndex5];

    const float& x1 = mdsInGPU.anchorX[firstMDIndex]/100;
    const float& x2 = mdsInGPU.anchorX[secondMDIndex]/100;
    const float& x3 = mdsInGPU.anchorX[thirdMDIndex]/100;
    const float& x4 = mdsInGPU.anchorX[fourthMDIndex]/100;
    const float& x5 = mdsInGPU.anchorX[fifthMDIndex]/100;
    const float& y1 = mdsInGPU.anchorY[firstMDIndex]/100;
    const float& y2 = mdsInGPU.anchorY[secondMDIndex]/100;
    const float& y3 = mdsInGPU.anchorY[thirdMDIndex]/100;
    const float& y4 = mdsInGPU.anchorY[fourthMDIndex]/100;
    const float& y5 = mdsInGPU.anchorY[fifthMDIndex]/100;

    float residual = 0;
    float error = 0;
    float x_center=g/100, y_center=f/100; 
    float x_init=mdsInGPU.anchorX[thirdMDIndex]/100;
    float y_init=mdsInGPU.anchorY[thirdMDIndex]/100;
    float z_init=mdsInGPU.anchorZ[thirdMDIndex]/100;
    float rt_init=mdsInGPU.anchorRt[thirdMDIndex]/100; //use the second MD as initial point

    if (moduleType3==1)  // 1: if MD3 is in 2s layer
    {
        x_init=mdsInGPU.anchorX[secondMDIndex]/100;
        y_init=mdsInGPU.anchorY[secondMDIndex]/100;
        z_init=mdsInGPU.anchorZ[secondMDIndex]/100;
        rt_init=mdsInGPU.anchorRt[secondMDIndex]/100;
    }

    //start from a circle of inner T3.
    // to determine the charge
    int charge=0;
    float slope3c=(y3-y_center)/(x3-x_center);
    float slope1c=(y1-y_center)/(x1-x_center);
    if((y3-y_center)>0 && (y1-y_center)>0) 
    {
        if (slope3c>slope1c) charge=-1; 
        else if (slope3c<slope1c) charge=1;
        if (slope1c>0 && slope3c<0) charge=-1;
        if (slope1c<0 && slope3c>0) charge=1;
    }
    else if((y3-y_center)<0 && (y1-y_center)<0) 
    {
        if (slope3c>slope1c) charge=-1; 
        else if (slope3c<slope1c) charge=1;
        if (slope1c<0 && slope3c>0) charge=1;
        if (slope1c>0 && slope3c<0) charge=-1;
    }
    else if ((y3-y_center)<0 && (y1-y_center)>0)
    {
        if ((x3-x_center)>0 && (x1-x_center)>0) charge = 1;
        else if ((x3-x_center)<0 && (x1-x_center)<0) charge = -1;
    }
    else if ((y3-y_center)>0 && (y1-y_center)<0)
    {
        if ((x3-x_center)>0 && (x1-x_center)>0) charge = -1;
        else if ((x3-x_center)<0 && (x1-x_center)<0) charge = 1;
    }

    float pseudo_phi = atan((y_init-y_center)/(x_init-x_center)); //actually represent pi/2-phi, wrt helix axis z
    float Pt=inner_pt, Px=Pt*abs(sin(pseudo_phi)), Py=Pt*abs(cos(pseudo_phi));

    if (x_init>x_center && y_init>y_center) //1st quad
    {
        if (charge==1) Py=-Py;
        if (charge==-1) Px=-Px;
    }
    if (x_init<x_center && y_init>y_center) //2nd quad
    {
        if (charge==-1) {Px=-Px; Py=-Py;}
    }
    if (x_init<x_center && y_init<y_center) //3rd quad
    {
        if (charge==1) Px=-Px;
        if (charge==-1) Py=-Py;
    }        
    if (x_init>x_center && y_init<y_center) //4th quad
    {
        if (charge==1) {Px=-Px; Py=-Py;}
    }

    //to get Pz, we use pt/pz=ds/dz, ds is the arclength between MD1 and MD3.
    float AO=sqrt((x1-x_center)*(x1-x_center)+(y1-y_center)*(y1-y_center));
    float BO=sqrt((x_init-x_center)*(x_init-x_center)+(y_init-y_center)*(y_init-y_center));
    float AB=sqrt((x1-x_init)*(x1-x_init)+(y1-y_init)*(y1-y_init)); 
    float dPhi = acos((AO*AO+BO*BO-AB*AB)/(2*AO*BO));
    float ds=innerRadius/100*dPhi;

//    float ds = sqrt((y_init-y1)*(y_init-y1)+(x_init-x1)*(x_init-x1)); //large ds->smallerPz->smaller residual
    float Pz=(z_init-z1)/ds*Pt;
    float p = sqrt(Px*Px+Py*Py+Pz*Pz);

    float B = 3.8112;
    float a = -0.299792*B*charge;

    float zsi, rtsi;
    int layeri, moduleTypei;
    float expectrt4=0,expectrt5=0,expectz4=0, expectz5=0;
    rzChiSquared=0;
    for(size_t i = 2; i < 6; i++)
    {
        if (i==2){
            zsi = z2;
            rtsi = rt2;
            layeri=layer2;
            moduleTypei=moduleType2;
        }
        else if (i==3) {
            zsi = z3;
            rtsi = rt3;
            layeri=layer3;
            moduleTypei=moduleType3;
        }
        else if (i==4){
            zsi = z4;
            rtsi = rt4;
            layeri=layer4;
            moduleTypei=moduleType4;
        }
        else if (i==5){
            zsi = z5;
            rtsi = rt5;
            layeri=layer5;
            moduleTypei=moduleType5;
        }

        if (moduleType3==0) { //0: ps
            if (i==3) continue;
        }
        else{
            if (i==2) continue;
        }

        // calculation is copied from PixelTriplet.cu SDL::computePT3RZChiSquared
        float diffr=0, diffz=0;

        float rou = a/p;
        // for barrel
        float s = (zsi-z_init)*p/Pz;
        float x = x_init + Px/a*sin(rou*s)-Py/a*(1-cos(rou*s));
        float y = y_init + Py/a*sin(rou*s)+Px/a*(1-cos(rou*s));
        diffr = (rtsi-sqrt(x*x+y*y))*100;
        if (i==4) expectrt4=sqrt(x*x+y*y);
        if (i==5) expectrt5=sqrt(x*x+y*y);

        // for endcap
        float paraA = rt_init*rt_init + 2*(Px*Px+Py*Py)/(a*a) + 2*(y_init*Px-x_init*Py)/a - rtsi*rtsi;
        float paraB = 2*(x_init*Px+y_init*Py)/a;
        float paraC = 2*(y_init*Px-x_init*Py)/a+2*(Px*Px+Py*Py)/(a*a);
        float A=paraB*paraB+paraC*paraC;
        float B=2*paraA*paraB;
        float C=paraA*paraA-paraC*paraC;
        float sol1 = (-B+sqrt(B*B-4*A*C))/(2*A);
        float sol2 = (-B-sqrt(B*B-4*A*C))/(2*A);
        float solz1 = asin(sol1)/rou*Pz/p+z_init;
        float solz2 = asin(sol2)/rou*Pz/p+z_init;
        float diffz1 = (solz1-zsi)*100;
        float diffz2 = (solz2-zsi)*100;
        diffz = (fabs(diffz1)<fabs(diffz2)) ? diffz1 : diffz2;
        if (i==4 && fabs(diffz1)<fabs(diffz2)) expectz4 = solz1;
        if (i==4 && fabs(diffz1)>fabs(diffz2)) expectz4 = solz2;
        if (i==5 && fabs(diffz1)<fabs(diffz2)) expectz5 = solz1;
        if (i==5 && fabs(diffz1)>fabs(diffz2)) expectz5 = solz2;

        residual = (layeri>6) ? diffr : diffz ;

        //PS Modules
        if(moduleTypei == 0)
        {
            error = 0.15f;
        }
        else //2S modules
        {
            error = 5.0f;
        }
        if (i==4) residual4=residual/error;
        if (i==5) residual5=residual/error;

        //check the tilted module, side: PosZ, NegZ, Center(for not tilted)
        float drdz;
        short side, subdets;
        if (i==2){
            drdz=abs(modulesInGPU.drdzs[lowerModuleIndex2]);
            side=modulesInGPU.sides[lowerModuleIndex2];
            subdets=modulesInGPU.subdets[lowerModuleIndex2];
        }
        if (i==3){
            drdz=abs(modulesInGPU.drdzs[lowerModuleIndex3]);
            side=modulesInGPU.sides[lowerModuleIndex3];
            subdets=modulesInGPU.subdets[lowerModuleIndex3];
        }
        if (i==2 || i==3){
            residual = (layeri <= 6 && ((side == SDL::Center) or (drdz < 1))) ? diffz : diffr;
//            residual_missing=residual;
            float projection_missing=1;
        if (drdz<1)
            projection_missing = ((subdets == SDL::Endcap) or (side == SDL::Center)) ? 1.f : 1/sqrt(1+drdz*drdz); // cos(atan(drdz)), if dr/dz<1
        if (drdz>1)
            projection_missing = ((subdets == SDL::Endcap) or (side == SDL::Center)) ? 1.f : drdz/sqrt(1+drdz*drdz);//sin(atan(drdz)), if dr/dz>1
            error=error*projection_missing;
            residual_missing=residual/error;
        }
        rzChiSquared += 12*(residual * residual)/(error * error);
    }
//    rzChiSquared = 12*(residual4 * residual4 + residual5 * residual5 + residual_missing * residual_missing);

//    if (isnan(rzChiSquared)) printf("rzChi2: %f, residual2: %f, inner_pt:%f, pseudo_phi: %f, charge: %i, Px:%f, Py:%f, x1:%f, x2:%f, x3:%f, x4:%f, x5:%f, y1:%f, y2:%f, y3:%f, y4:%f, y5:%f, z1:%f, z2:%f, z3:%f, z4:%f, z5:%f, x_center:%f, y_center:%f, slope1c:%f, slope3c:%f\n", rzChiSquared, residual_missing, inner_pt, pseudo_phi, charge, Px, Py, x1, x2, x3, x4, x5, y1, y2, y3, y4, y5, z1, z2, z3, z4, z5, x_center, y_center, slope1c, slope3c);

//    if(layer1 == 1 and layer2 == 2 and layer3 == 3 and layer4 == 12 and layer5 == 13){
//        printf("rt1:%f, rt2:%f, rt3:%f, rt4:%f, rt5:%f\n", rt1, rt2, rt3, rt4, rt5);
//        printf("x1:%f, x2:%f, x3:%f, x4:%f, x5:%f\n", x1, x2, x3, x4, x5);
//        printf("y1:%f, y2:%f, y3:%f, y4:%f, y5:%f\n", y1, y2, y3, y4, y5);
//        printf("z1:%f, z2:%f, z3:%f, z4:%f, z5:%f\n", z1, z2, z3, z4, z5);
//        printf("rt4_ex:%f, rt5_ex:%f\n", expectrt4, expectrt5);
//        printf("z4_ex:%f, z5_ex:%f\n", expectz4, expectz5);
//        printf("residual_missing:%f\n", residual_missing);
//        printf("Pt:%f, Px:%f, Py:%f, Pz:%f, charge: %i, residual_missing: %f, residual4: %f, residual5:%f, moduleType3:%i\n", Pt, Px, Py, Pz, charge, residual_missing, residual4, residual5, moduleType3);
//        printf("rzChi2: %f, residual2: %f, inner_pt:%f, pseudo_phi: %f, charge: %i, Px:%f, Py:%f, x1:%f, x2:%f, x3:%f, x4:%f, x5:%f, y1:%f, y2:%f, y3:%f, y4:%f, y5:%f, z1:%f, z2:%f, z3:%f, z4:%f, z5:%f, x_center:%f, y_center:%f, slope1c:%f, slope3c:%f\n", rzChiSquared, residual_missing, inner_pt, pseudo_phi, charge, Px, Py, x1, x2, x3, x4, x5, y1, y2, y3, y4, y5, z1, z2, z3, z4, z5, x_center, y_center, slope1c, slope3c);
//        printf("residual_missing:%f\n", residual_missing);
//    }

    // when building T5, apply 99% chi2 cuts as default, and add to pT5 collection. But when adding T5 to TC collections, appy 95% cut to reduce the fake rate
    TightCutFlag = false;
    //categories!
    if(layer1 == 1 and layer2 == 2 and layer3 == 3)
    {
        if(layer4 == 4 and layer5 == 5) //11
        {
            if (rzChiSquared < 15.595f) TightCutFlag = 1;
            return rzChiSquared < 28.902f; 
        }
        else if(layer4 == 4 and layer5 == 12) //12
        {
            if (rzChiSquared < 14.614f) TightCutFlag = 1;
            return rzChiSquared < 23.037f;
        }
        else if(layer4 == 7 and layer5 == 8) //8
        {   
            if (rzChiSquared < 27.824f) TightCutFlag = 1;
            return rzChiSquared < 44.247f;
        }
        else if(layer4 == 7 and layer5 == 13) //9
        {
            if (rzChiSquared < 18.085f) TightCutFlag = 1;
            return rzChiSquared < 33.023f;
        }
        else if(layer4 == 12 and layer5 == 13) //10
        {
            if (rzChiSquared < 13.267f) TightCutFlag = 1;
            return rzChiSquared < 21.186f;
        }
    }
    else if(layer1 == 1 and layer2 == 2 and layer3 == 7)
    {
        if(layer4 == 8 and layer5 == 9) //5
        {
            if (rzChiSquared < 60.195f) TightCutFlag = 1;
            return rzChiSquared < 117.118f;
        }
        if(layer4 == 8 and layer5 == 14) //6
        {
            if (rzChiSquared < 19.490f) TightCutFlag = 1;
            return rzChiSquared < 55.322f;
        }
        else if(layer4 == 13 and layer5 == 14) //7
        {
            if (rzChiSquared < 10.157f) TightCutFlag = 1;
            return rzChiSquared < 14.217f;
        }
    }
    else if(layer1 == 1 and layer2 == 7 and layer3 == 8 and layer4 == 9) 
    {
        if (layer5 == 10) //3
        {
            if (rzChiSquared < 63.697f) TightCutFlag = 1;
            return rzChiSquared < 109.584f;
        }
        if (layer5 == 15) //4
        {
            if (rzChiSquared < 18.346f) TightCutFlag = 1;
            return rzChiSquared < 34.941f;
        }
    }
    else if(layer1 == 2 and layer2 == 3 and layer3 == 4)
    {
        if(layer4 == 5 and layer5 == 6) //18
        {
            if (rzChiSquared < 6.053f) TightCutFlag = 1;
            return rzChiSquared < 8.629f;
        }
        else if(layer4 == 5 and layer5 == 12) //19
        {
            if (rzChiSquared < 5.693f) TightCutFlag = 1;
            return rzChiSquared < 7.929f;
        }

        else if(layer4 == 12 and layer5 == 13) //20
        {
            if (rzChiSquared < 5.44f) TightCutFlag = 1;
            return rzChiSquared < 7.627f;
        }
    }
    else if(layer1 == 2 and layer2 == 3 and layer3 == 7) 
    {
        if(layer4 == 8 and layer5 == 14) //16
        {
            if (rzChiSquared < 23.730f) TightCutFlag = 1;
            return rzChiSquared < 23.748f;
        }
        if(layer4 == 13 and layer5 == 14) //17
        {
            if (rzChiSquared < 10.55f) TightCutFlag = 1;
            return rzChiSquared < 17.817f;
        }
    }

    else if(layer1 == 2 and layer2 == 7 and layer3 == 8)
    {
        if(layer4 == 9 and layer5 == 15) //14
        {
            if (rzChiSquared < 24.558f) TightCutFlag = 1;
            return rzChiSquared < 40.918f;
        }
        else if(layer4 == 14 and layer5 == 15) //15
        {
            if (rzChiSquared < 8.752f) TightCutFlag = 1;
            return rzChiSquared < 13.678f;
        }
    }

    else if(layer1 == 7 and layer2 == 8 and layer3 == 9 and layer4 == 15 and layer5 == 16) //2
    {
        if (rzChiSquared < 7.994f) TightCutFlag = 1;
        return rzChiSquared < 11.622f;
    }

    else if(layer1 == 7 and layer2 == 8 and layer3 == 9 and layer4 == 10 and layer5 == 11) //0
    {
        if (rzChiSquared < 56.313f) TightCutFlag = 1;
        return rzChiSquared < 93.893f;
    }

    else if(layer1 == 7 and layer2 == 8 and layer3 == 9 and layer4 == 10 and layer5 == 16) //1
    {
        if (rzChiSquared < 22.041f) TightCutFlag = 1;
        return rzChiSquared < 37.087f;
    }
    return true;
}

__device__ bool SDL::checkIntervalOverlap(const float& firstMin, const float& firstMax, const float& secondMin, const float& secondMax)
{
    return ((firstMin <= secondMin) & (secondMin < firstMax)) |  ((secondMin < firstMin) & (firstMin < secondMax));
}

/*bounds for high Pt taken from : http://uaf-10.t2.ucsd.edu/~bsathian/SDL/T5_efficiency/efficiencies/new_efficiencies/efficiencies_20210513_T5_recovering_high_Pt_efficiencies/highE_radius_matching/highE_bounds.txt */

__device__ bool SDL::matchRadiiBBBBB(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, float& innerInvRadiusMin, float& innerInvRadiusMax, float& bridgeInvRadiusMin, float& bridgeInvRadiusMax, float& outerInvRadiusMin, float& outerInvRadiusMax)
{
    float innerInvRadiusErrorBound =  0.1512f;
    float bridgeInvRadiusErrorBound = 0.1781f;
    float outerInvRadiusErrorBound = 0.1840f;

    if(innerRadius > 2.0f/(2.f * k2Rinv1GeVf))
    {
        innerInvRadiusErrorBound = 0.4449f;
        bridgeInvRadiusErrorBound = 0.4033f;
        outerInvRadiusErrorBound = 0.8016f;
    }

    innerInvRadiusMax = (1.f + innerInvRadiusErrorBound) / innerRadius;
    innerInvRadiusMin = fmaxf(0.f, (1.f - innerInvRadiusErrorBound) / innerRadius);

    bridgeInvRadiusMax = (1.f + bridgeInvRadiusErrorBound) / bridgeRadius;
    bridgeInvRadiusMin = fmaxf(0.f, (1.f - bridgeInvRadiusErrorBound) / bridgeRadius);

    outerInvRadiusMax = (1.f + outerInvRadiusErrorBound) / outerRadius;
    outerInvRadiusMin = fmaxf(0.f, (1.f - outerInvRadiusErrorBound) / outerRadius);

    return checkIntervalOverlap(innerInvRadiusMin, innerInvRadiusMax, bridgeInvRadiusMin, bridgeInvRadiusMax);
}

__device__ bool SDL::matchRadiiBBBBE(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S, float& innerInvRadiusMin, float& innerInvRadiusMax, float& bridgeInvRadiusMin, float& bridgeInvRadiusMax, float& outerInvRadiusMin, float& outerInvRadiusMax)
{

    float innerInvRadiusErrorBound =  0.1781f;
    float bridgeInvRadiusErrorBound = 0.2167f;
    float outerInvRadiusErrorBound = 1.1116f;

    if(innerRadius > 2.0f/(2.f * k2Rinv1GeVf))
    {
        innerInvRadiusErrorBound = 0.4750f;
        bridgeInvRadiusErrorBound = 0.3903f;
        outerInvRadiusErrorBound = 15.2120f;
    }

    innerInvRadiusMax = (1.f + innerInvRadiusErrorBound) / innerRadius;
    innerInvRadiusMin = fmaxf(0.f, (1.f - innerInvRadiusErrorBound) / innerRadius);

    bridgeInvRadiusMax = (1.f + bridgeInvRadiusErrorBound) / bridgeRadius;
    bridgeInvRadiusMin = fmaxf(0.f, (1.f - bridgeInvRadiusErrorBound) / bridgeRadius);

    outerInvRadiusMax = (1.f + outerInvRadiusErrorBound) / outerRadius;
    outerInvRadiusMin = fmaxf(0.f, (1.f - outerInvRadiusErrorBound) / outerRadius);

    return checkIntervalOverlap(innerInvRadiusMin, innerInvRadiusMax, bridgeInvRadiusMin, bridgeInvRadiusMax);
}

__device__ bool SDL::matchRadiiBBBEE12378(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S, float& innerInvRadiusMin, float& innerInvRadiusMax, float& bridgeInvRadiusMin, float& bridgeInvRadiusMax, float& outerInvRadiusMin, float& outerInvRadiusMax)
{
    float innerInvRadiusErrorBound = 0.178f;
    float bridgeInvRadiusErrorBound = 0.507f;
    float outerInvRadiusErrorBound = 7.655f;

    innerInvRadiusMax = (1.f + innerInvRadiusErrorBound) / innerRadius;
    innerInvRadiusMin = fmaxf(0.f, (1.f - innerInvRadiusErrorBound) / innerRadius);

    bridgeInvRadiusMax = (1.f + bridgeInvRadiusErrorBound) / bridgeRadius;
    bridgeInvRadiusMin = fmaxf(0.f, (1.f - bridgeInvRadiusErrorBound) / bridgeRadius);

    outerInvRadiusMax = (1.f + outerInvRadiusErrorBound) / outerRadius;
    outerInvRadiusMin = fmaxf(0.f, (1.f - outerInvRadiusErrorBound) / outerRadius);

    return checkIntervalOverlap(innerInvRadiusMin, innerInvRadiusMax, fminf(bridgeInvRadiusMin, 1.0f/bridgeRadiusMax2S), fmaxf(bridgeInvRadiusMax, 1.0f/bridgeRadiusMin2S));
}

__device__ bool SDL::matchRadiiBBBEE23478(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S, float& innerInvRadiusMin, float& innerInvRadiusMax, float& bridgeInvRadiusMin, float& bridgeInvRadiusMax, float& outerInvRadiusMin, float& outerInvRadiusMax)
{
    float innerInvRadiusErrorBound = 0.2097f;
    float bridgeInvRadiusErrorBound = 0.8557f;
    float outerInvRadiusErrorBound = 24.0450f;

    innerInvRadiusMax = (1.f + innerInvRadiusErrorBound) / innerRadius;
    innerInvRadiusMin = fmaxf(0.f, (1.f - innerInvRadiusErrorBound) / innerRadius);

    bridgeInvRadiusMax = (1.f + bridgeInvRadiusErrorBound) / bridgeRadius;
    bridgeInvRadiusMin = fmaxf(0.f, (1.f - bridgeInvRadiusErrorBound) / bridgeRadius);

    outerInvRadiusMax = (1.f + outerInvRadiusErrorBound) / outerRadius;
    outerInvRadiusMin = fmaxf(0.f, (1.f - outerInvRadiusErrorBound) / outerRadius);

    return checkIntervalOverlap(innerInvRadiusMin, innerInvRadiusMax, fminf(bridgeInvRadiusMin, 1.0f/bridgeRadiusMax2S), fmaxf(bridgeInvRadiusMax, 1.0f/bridgeRadiusMin2S));

}

__device__ bool SDL::matchRadiiBBBEE34578(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S, float& innerInvRadiusMin, float& innerInvRadiusMax, float& bridgeInvRadiusMin, float& bridgeInvRadiusMax, float& outerInvRadiusMin, float& outerInvRadiusMax)
{
    float innerInvRadiusErrorBound = 0.066f;
    float bridgeInvRadiusErrorBound = 0.617f;
    float outerInvRadiusErrorBound = 2.688f;

    innerInvRadiusMax = (1.f + innerInvRadiusErrorBound) / innerRadius;
    innerInvRadiusMin = fmaxf(0.f, (1.f - innerInvRadiusErrorBound) / innerRadius);

    bridgeInvRadiusMax = (1.f + bridgeInvRadiusErrorBound) / bridgeRadius;
    bridgeInvRadiusMin = fmaxf(0.f, (1.f - bridgeInvRadiusErrorBound) / bridgeRadius);

    outerInvRadiusMax = (1.f + outerInvRadiusErrorBound) / outerRadius;
    outerInvRadiusMin = fmaxf(0.f, (1.f - outerInvRadiusErrorBound) / outerRadius);

    return checkIntervalOverlap(innerInvRadiusMin, innerInvRadiusMax, fminf(bridgeInvRadiusMin, 1.0f/bridgeRadiusMax2S), fmaxf(bridgeInvRadiusMax, 1.0f/bridgeRadiusMin2S));

}

__device__ bool SDL::matchRadiiBBBEE(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S, float& innerInvRadiusMin, float& innerInvRadiusMax, float& bridgeInvRadiusMin, float& bridgeInvRadiusMax, float& outerInvRadiusMin, float& outerInvRadiusMax)
{

    float innerInvRadiusErrorBound =  0.1840f;
    float bridgeInvRadiusErrorBound = 0.5971f;
    float outerInvRadiusErrorBound = 11.7102f;

    if(innerRadius > 2.0f/(2.f * k2Rinv1GeVf)) //as good as no selections
    {
        innerInvRadiusErrorBound = 1.0412f;
        outerInvRadiusErrorBound = 32.2737f;
        bridgeInvRadiusErrorBound = 10.9688f;
    }

    innerInvRadiusMax = (1.f + innerInvRadiusErrorBound) / innerRadius;
    innerInvRadiusMin = fmaxf(0.f, (1.f - innerInvRadiusErrorBound) / innerRadius);

    bridgeInvRadiusMax = (1.f + bridgeInvRadiusErrorBound) / bridgeRadius;
    bridgeInvRadiusMin = fmaxf(0.f, (1.f - bridgeInvRadiusErrorBound) / bridgeRadius);

    outerInvRadiusMax = (1.f + outerInvRadiusErrorBound) / outerRadius;
    outerInvRadiusMin = fmaxf(0.f, (1.f - outerInvRadiusErrorBound) / outerRadius);

    return checkIntervalOverlap(innerInvRadiusMin, innerInvRadiusMax, fminf(bridgeInvRadiusMin, 1.0f/bridgeRadiusMax2S), fmaxf(bridgeInvRadiusMax, 1.0f/bridgeRadiusMin2S));

}

__device__ bool SDL::matchRadiiBBEEE(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S, float& innerInvRadiusMin, float& innerInvRadiusMax, float& bridgeInvRadiusMin, float& bridgeInvRadiusMax, float& outerInvRadiusMin, float& outerInvRadiusMax)
{
    float innerInvRadiusErrorBound =  0.6376f;
    float bridgeInvRadiusErrorBound = 2.1381f;
    float outerInvRadiusErrorBound = 20.4179f;

    if(innerRadius > 2.0f/(2.f * k2Rinv1GeVf)) //as good as no selections!
    {
        innerInvRadiusErrorBound = 12.9173f;
        outerInvRadiusErrorBound = 25.6702f;
        bridgeInvRadiusErrorBound = 5.1700f;
    }

    innerInvRadiusMax = (1.f + innerInvRadiusErrorBound) / innerRadius;
    innerInvRadiusMin = fmaxf(0.f, (1.f - innerInvRadiusErrorBound) / innerRadius);

    bridgeInvRadiusMax = (1.f + bridgeInvRadiusErrorBound) / bridgeRadius;
    bridgeInvRadiusMin = fmaxf(0.f, (1.f - bridgeInvRadiusErrorBound) / bridgeRadius);

    outerInvRadiusMax = (1.f + outerInvRadiusErrorBound) / outerRadius;
    outerInvRadiusMin = fmaxf(0.f, (1.f - outerInvRadiusErrorBound) / outerRadius);

    return checkIntervalOverlap(innerInvRadiusMin, innerInvRadiusMax, fminf(bridgeInvRadiusMin, 1.0f/bridgeRadiusMax2S), fmaxf(bridgeInvRadiusMax, 1.0f/bridgeRadiusMin2S));

}

__device__ bool SDL::matchRadiiBEEEE(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S, float& innerInvRadiusMin, float& innerInvRadiusMax, float& bridgeInvRadiusMin, float& bridgeInvRadiusMax, float& outerInvRadiusMin, float& outerInvRadiusMax)
{

    float innerInvRadiusErrorBound =  1.9382f;
    float bridgeInvRadiusErrorBound = 3.7280f;
    float outerInvRadiusErrorBound = 5.7030f;


    if(innerRadius > 2.0f/(2.f * k2Rinv1GeVf))
    {
        innerInvRadiusErrorBound = 23.2713f;
        outerInvRadiusErrorBound = 24.0450f;
        bridgeInvRadiusErrorBound = 21.7980f;
    }

    innerInvRadiusMax = (1.f + innerInvRadiusErrorBound) / innerRadius;
    innerInvRadiusMin = fmaxf(0.f, (1.f - innerInvRadiusErrorBound) / innerRadius);

    bridgeInvRadiusMax = (1.f + bridgeInvRadiusErrorBound) / bridgeRadius;
    bridgeInvRadiusMin = fmaxf(0.f, (1.f - bridgeInvRadiusErrorBound) / bridgeRadius);

    outerInvRadiusMax = (1.f + outerInvRadiusErrorBound) / outerRadius;
    outerInvRadiusMin = fmaxf(0.f, (1.f - outerInvRadiusErrorBound) / outerRadius);

    return checkIntervalOverlap(fminf(innerInvRadiusMin, 1.0/innerRadiusMax2S), fmaxf(innerInvRadiusMax, 1.0/innerRadiusMin2S), fminf(bridgeInvRadiusMin, 1.0/bridgeRadiusMax2S), fmaxf(bridgeInvRadiusMax, 1.0/bridgeRadiusMin2S));
}

__device__ bool SDL::matchRadiiEEEEE(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S, float& innerInvRadiusMin, float& innerInvRadiusMax, float& bridgeInvRadiusMin, float& bridgeInvRadiusMax, float& outerInvRadiusMin, float& outerInvRadiusMax)
{
    float innerInvRadiusErrorBound =  1.9382f;
    float bridgeInvRadiusErrorBound = 2.2091f;
    float outerInvRadiusErrorBound = 7.4084f;

    if(innerRadius > 2.0f/(2.f * k2Rinv1GeVf))
    {
        innerInvRadiusErrorBound = 22.5226f;
        bridgeInvRadiusErrorBound = 21.0966f;
        outerInvRadiusErrorBound = 19.1252f;
    }

    innerInvRadiusMax = (1.f + innerInvRadiusErrorBound) / innerRadius;
    innerInvRadiusMin = fmaxf(0.f, (1.f - innerInvRadiusErrorBound) / innerRadius);

    bridgeInvRadiusMax = (1.f + bridgeInvRadiusErrorBound) / bridgeRadius;
    bridgeInvRadiusMin = fmaxf(0.f, (1.f - bridgeInvRadiusErrorBound) / bridgeRadius);

    outerInvRadiusMax = (1.f + outerInvRadiusErrorBound) / outerRadius;
    outerInvRadiusMin = fmaxf(0.f, (1.f - outerInvRadiusErrorBound) / outerRadius);

    return checkIntervalOverlap(fminf(innerInvRadiusMin, 1.0/innerRadiusMax2S), fmaxf(innerInvRadiusMax, 1.0/innerRadiusMin2S), fminf(bridgeInvRadiusMin, 1.0/bridgeRadiusMax2S), fmaxf(bridgeInvRadiusMax, 1.0/bridgeRadiusMin2S));
}

__device__ void SDL::computeErrorInRadius(float* x1Vec, float* y1Vec, float* x2Vec, float* y2Vec, float* x3Vec, float* y3Vec, float& minimumRadius, float& maximumRadius)
{
    //brute force
    float candidateRadius;
    float g, f;
    minimumRadius = 123456789.f;
    maximumRadius = 0.f;
    for(size_t i = 0; i < 3; i++)
    {
        float x1 = x1Vec[i];
	float y1 = y1Vec[i];
        for(size_t j = 0; j < 3; j++)
        {
	    float x2 = x2Vec[j];
	    float y2 = y2Vec[j];
            for(size_t k = 0; k < 3; k++)
            {
	       float x3 = x3Vec[k];
               float y3 = y3Vec[k];
               candidateRadius = computeRadiusFromThreeAnchorHits(x1, y1, x2, y2, x3, y3, g, f);
               maximumRadius = fmaxf(candidateRadius, maximumRadius);
               minimumRadius = fminf(candidateRadius, minimumRadius);
            }
        }
    }
}
__device__ float SDL::computeRadiusFromThreeAnchorHits(float x1, float y1, float x2, float y2, float x3, float y3, float& g, float& f)
{
    float radius = 0.f;

    //writing manual code for computing radius, which obviously sucks
    //TODO:Use fancy inbuilt libraries like cuBLAS or cuSOLVE for this!
    //(g,f) -> center
    //first anchor hit - (x1,y1), second anchor hit - (x2,y2), third anchor hit - (x3, y3)

    /*
    if((y1 - y3) * (x2 - x3) - (x1 - x3) * (y2 - y3) == 0)
    {
        return -1; //WTF man, three collinear points!
    }
    */

    float denomInv = 1.0f/((y1 - y3) * (x2 - x3) - (x1 - x3) * (y2 - y3));

    float xy1sqr = x1 * x1 + y1 * y1;

    float xy2sqr = x2 * x2 + y2 * y2;

    float xy3sqr = x3 * x3 + y3 * y3;

    g = 0.5f * ((y3 - y2) * xy1sqr + (y1 - y3) * xy2sqr + (y2 - y1) * xy3sqr) * denomInv;

    f = 0.5f * ((x2 - x3) * xy1sqr + (x3 - x1) * xy2sqr + (x1 - x2) * xy3sqr) * denomInv;

    float c = ((x2 * y3 - x3 * y2) * xy1sqr + (x3 * y1 - x1 * y3) * xy2sqr + (x1 * y2 - x2 * y1) * xy3sqr) * denomInv;

    if(((y1 - y3) * (x2 - x3) - (x1 - x3) * (y2 - y3) == 0) || (g * g + f * f - c < 0))
    {
        printf("three collinear points or FATAL! r^2 < 0!\n");
	radius = -1.f;
    }
    else
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

__device__ void SDL::computeSigmasForRegression(SDL::modules& modulesInGPU, const uint16_t* lowerModuleIndices, float* delta1, float* delta2, float* slopes, bool* isFlat, int nPoints, bool anchorHits) 
{
   /*bool anchorHits required to deal with a weird edge case wherein 
     the hits ultimately used in the regression are anchor hits, but the
     lower modules need not all be Pixel Modules (in case of PS). Similarly,
     when we compute the chi squared for the non-anchor hits, the "partner module"
     need not always be a PS strip module, but all non-anchor hits sit on strip 
     modules.
    */
    ModuleType moduleType;
    short moduleSubdet, moduleSide;
    float inv1 = 0.01f/0.009f;
    float inv2 = 0.15f/0.009f;
    float inv3 = 2.4f/0.009f;
    for(size_t i=0; i<nPoints; i++)
    {
        moduleType = modulesInGPU.moduleType[lowerModuleIndices[i]];
        moduleSubdet = modulesInGPU.subdets[lowerModuleIndices[i]];
        moduleSide = modulesInGPU.sides[lowerModuleIndices[i]];
        float& drdz = modulesInGPU.drdzs[lowerModuleIndices[i]];
        slopes[i] = modulesInGPU.slopes[lowerModuleIndices[i]]; 
        //category 1 - barrel PS flat
        if(moduleSubdet == Barrel and moduleType == PS and moduleSide == Center)
        {
            delta1[i] = inv1;//1.1111f;//0.01;
            delta2[i] = inv1;//1.1111f;//0.01;
            slopes[i] = -999.f;
            isFlat[i] = true;
        }

        //category 2 - barrel 2S
        else if(moduleSubdet == Barrel and moduleType == TwoS)
        {
            delta1[i] = 1.f;//0.009;
            delta2[i] = 1.f;//0.009;
            slopes[i] = -999.f;
            isFlat[i] = true;
        }

        //category 3 - barrel PS tilted
        else if(moduleSubdet == Barrel and moduleType == PS and moduleSide != Center)
        {

            //delta1[i] = 0.01;
            delta1[i] = inv1;//1.1111f;//0.01;
            isFlat[i] = false;

            if(anchorHits)
            {
                delta2[i] = (inv2 * drdz/sqrtf(1 + drdz * drdz));
            }
            else
            {
                delta2[i] = (inv3 * drdz/sqrtf(1 + drdz * drdz));
            }
        }
        //category 4 - endcap PS
        else if(moduleSubdet == Endcap and moduleType == PS)
        {
            delta1[i] = inv1;//1.1111f;//0.01;
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
            delta1[i] = 1.f;//0.009;
            delta2[i] = 500.f*inv1;//555.5555f;//5.f;
            isFlat[i] = false;
        }
        else
        {
            printf("ERROR!!!!! I SHOULDN'T BE HERE!!!! subdet = %d, type = %d, side = %d\n", moduleSubdet, moduleType, moduleSide);
        }
    }
}

__device__ float SDL::computeRadiusUsingRegression(int nPoints, float* xs, float* ys, float* delta1, float* delta2, float* slopes, bool* isFlat, float& g, float& f, float* sigmas, float& chiSquared)
{
    float radius = 0.f;

    //some extra variables
    //the two variables will be caled x1 and x2, and y (which is x^2 + y^2)

    float sigmaX1Squared = 0.f;
    float sigmaX2Squared = 0.f;
    float sigmaX1X2 = 0.f; 
    float sigmaX1y = 0.f; 
    float sigmaX2y = 0.f;
    float sigmaY = 0.f;
    float sigmaX1 = 0.f;
    float sigmaX2 = 0.f;
    float sigmaOne = 0.f;

    float xPrime, yPrime, absArctanSlope, angleM;
    for(size_t i = 0; i < nPoints; i++)
    {
        //computing sigmas is a very tricky affair
        //if the module is tilted or endcap, we need to use the slopes properly!

        absArctanSlope = ((slopes[i] != 123456789) ? fabs(atanf(slopes[i])) : 0.5f*float(M_PI)); // Since C++ can't represent infinity, SDL_INF = 123456789 was used to represent infinity in the data table

        if(xs[i] > 0 and ys[i] > 0)
        {
            angleM = 0.5f*float(M_PI) - absArctanSlope;
        }
        else if(xs[i] < 0 and ys[i] > 0)
        {
            angleM = absArctanSlope + 0.5f*float(M_PI);
        }
        else if(xs[i] < 0 and ys[i] < 0)
        {
            angleM = -(absArctanSlope + 0.5f*float(M_PI));
        }
        else if(xs[i] > 0 and ys[i] < 0)
        {
            angleM = -(0.5f*float(M_PI) - absArctanSlope);
        }

        if(not isFlat[i])
        {
            xPrime = xs[i] * cosf(angleM) + ys[i] * sinf(angleM);
            yPrime = ys[i] * cosf(angleM) - xs[i] * sinf(angleM);
        }
        else
        {
            xPrime = xs[i];
            yPrime = ys[i];
        }
        sigmas[i] = 2 * sqrtf((xPrime * delta1[i]) * (xPrime * delta1[i]) + (yPrime * delta2[i]) * (yPrime * delta2[i]));

        sigmaX1Squared += (xs[i] * xs[i])/(sigmas[i] * sigmas[i]);
        sigmaX2Squared += (ys[i] * ys[i])/(sigmas[i] * sigmas[i]);
        sigmaX1X2 += (xs[i] * ys[i])/(sigmas[i] * sigmas[i]);
        sigmaX1y += (xs[i] * (xs[i] * xs[i] + ys[i] * ys[i]))/(sigmas[i] * sigmas[i]);
        sigmaX2y += (ys[i] * (xs[i] * xs[i] + ys[i] * ys[i]))/(sigmas[i] * sigmas[i]);
        sigmaY += (xs[i] * xs[i] + ys[i] * ys[i])/(sigmas[i] * sigmas[i]);
        sigmaX1 += xs[i]/(sigmas[i] * sigmas[i]);
        sigmaX2 += ys[i]/(sigmas[i] * sigmas[i]);
        sigmaOne += 1.0f/(sigmas[i] * sigmas[i]);
    }
    float denominator = (sigmaX1X2 - sigmaX1 * sigmaX2) * (sigmaX1X2 - sigmaX1 * sigmaX2) - (sigmaX1Squared - sigmaX1 * sigmaX1) * (sigmaX2Squared - sigmaX2 * sigmaX2);

    float twoG = ((sigmaX2y - sigmaX2 * sigmaY) * (sigmaX1X2 - sigmaX1 * sigmaX2) - (sigmaX1y - sigmaX1 * sigmaY) * (sigmaX2Squared - sigmaX2 * sigmaX2)) / denominator;
    float twoF = ((sigmaX1y - sigmaX1 * sigmaY) * (sigmaX1X2 - sigmaX1 * sigmaX2) - (sigmaX2y - sigmaX2 * sigmaY) * (sigmaX1Squared - sigmaX1 * sigmaX1)) / denominator;

    float c = -(sigmaY - twoG * sigmaX1 - twoF * sigmaX2)/sigmaOne;
    g = 0.5f*twoG;
    f = 0.5f*twoF;
    if(g * g + f * f - c < 0)
    {
        printf("FATAL! r^2 < 0!\n");
        return -1;
    }
    
    radius = sqrtf(g * g  + f * f - c);
    //compute chi squared
    chiSquared = 0.f;
    for(size_t i = 0; i < nPoints; i++)
    {
       chiSquared +=  (xs[i] * xs[i] + ys[i] * ys[i] - twoG * xs[i] - twoF * ys[i] + c) * (xs[i] * xs[i] + ys[i] * ys[i] - twoG * xs[i] - twoF * ys[i] + c) / (sigmas[i] * sigmas[i]);
    }
    return radius;
}

__device__ float SDL::computeChiSquared(int nPoints, float* xs, float* ys, float* delta1, float* delta2, float* slopes, bool* isFlat, float g, float f, float radius)
{
    // given values of (g, f, radius) and a set of points (and its uncertainties)
    //compute chi squared
    float c = g*g + f*f - radius*radius;
    float chiSquared = 0.f;
    float absArctanSlope, angleM, xPrime, yPrime, sigma;
    for(size_t i = 0; i < nPoints; i++)
    {
        absArctanSlope = ((slopes[i] != 123456789) ? fabs(atanf(slopes[i])) : 0.5f*float(M_PI)); // Since C++ can't represent infinity, SDL_INF = 123456789 was used to represent infinity in the data table
        if(xs[i] > 0 and ys[i] > 0)
        {
            angleM = 0.5f*float(M_PI) - absArctanSlope;
        }
        else if(xs[i] < 0 and ys[i] > 0)
        {
            angleM = absArctanSlope + 0.5f*float(M_PI);
        }
        else if(xs[i] < 0 and ys[i] < 0)
        {
            angleM = -(absArctanSlope + 0.5f*float(M_PI));
        }
        else if(xs[i] > 0 and ys[i] < 0)
        {
            angleM = -(0.5f*float(M_PI) - absArctanSlope);
        }

        if(not isFlat[i])
        {
            xPrime = xs[i] * cosf(angleM) + ys[i] * sinf(angleM);
            yPrime = ys[i] * cosf(angleM) - xs[i] * sinf(angleM);
        }
        else
        {
            xPrime = xs[i];
            yPrime = ys[i];
        }
        sigma = 2 * sqrtf((xPrime * delta1[i]) * (xPrime * delta1[i]) + (yPrime * delta2[i]) * (yPrime * delta2[i]));
        chiSquared +=  (xs[i] * xs[i] + ys[i] * ys[i] - 2 * g * xs[i] - 2 * f * ys[i] + c) * (xs[i] * xs[i] + ys[i] * ys[i] - 2 * g * xs[i] - 2 * f * ys[i] + c) / (sigma * sigma);
    }
    return chiSquared; 
}

__global__ void SDL::createQuintupletsInGPUv2(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::quintuplets& quintupletsInGPU, struct SDL::objectRanges& rangesInGPU, uint16_t nEligibleT5Modules)
{
    int gidy = blockIdx.y * blockDim.y + threadIdx.y;
    int npy = gridDim.y * blockDim.y;
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    int npx = gridDim.x * blockDim.x;
    int gidz = blockIdx.z * blockDim.z + threadIdx.z;
    int npz = gridDim.z * blockDim.z;

    for (int iter=gidz; iter < nEligibleT5Modules; iter+=npz){
      uint16_t lowerModule1 = rangesInGPU.indicesOfEligibleT5Modules[iter];


      unsigned int nInnerTriplets = tripletsInGPU.nTriplets[lowerModule1];
      for( unsigned int innerTripletArrayIndex =gidy; innerTripletArrayIndex < nInnerTriplets; innerTripletArrayIndex+=npy){

      unsigned int innerTripletIndex = rangesInGPU.tripletModuleIndices[lowerModule1] + innerTripletArrayIndex;
      uint16_t lowerModule2 = tripletsInGPU.lowerModuleIndices[3 * innerTripletIndex + 1];
      uint16_t lowerModule3 = tripletsInGPU.lowerModuleIndices[3 * innerTripletIndex + 2];
      unsigned int nOuterTriplets = tripletsInGPU.nTriplets[lowerModule3];
        for (int outerTripletArrayIndex=gidx; outerTripletArrayIndex < nOuterTriplets; outerTripletArrayIndex+=npx)
        {
            unsigned int outerTripletIndex = rangesInGPU.tripletModuleIndices[lowerModule3] + outerTripletArrayIndex;
            uint16_t lowerModule4 = tripletsInGPU.lowerModuleIndices[3 * outerTripletIndex + 1];
            uint16_t lowerModule5 = tripletsInGPU.lowerModuleIndices[3 * outerTripletIndex + 2];

            float innerRadius, outerRadius, bridgeRadius, regressionG, regressionF, regressionRadius, rzChiSquared, chiSquared, nonAnchorChiSquared; //required for making distributions
            bool TightCutFlag;
            bool success = runQuintupletDefaultAlgo(modulesInGPU, mdsInGPU, segmentsInGPU, tripletsInGPU, lowerModule1, lowerModule2, lowerModule3, lowerModule4, lowerModule5, innerTripletIndex, outerTripletIndex, innerRadius, outerRadius,  bridgeRadius, regressionG, regressionF, regressionRadius, rzChiSquared, chiSquared, nonAnchorChiSquared, TightCutFlag);

            if(success)
            {
                short layer2_adjustment;
                int layer = modulesInGPU.layers[lowerModule1];
                if(layer == 1)
                {
                    layer2_adjustment = 1;
                } //get upper segment to be in second layer
                else if(layer == 2)
                {
                    layer2_adjustment = 0;
                } // get lower segment to be in second layer
                else
                {
                    return;
                } // ignore anything else TODO: move this to start, before object is made (faster)
                unsigned int totOccupancyQuintuplets = atomicAdd(&quintupletsInGPU.totOccupancyQuintuplets[lowerModule1], 1);
                if(totOccupancyQuintuplets >= (rangesInGPU.quintupletModuleIndices[lowerModule1 + 1] - rangesInGPU.quintupletModuleIndices[lowerModule1]))
                {
#ifdef Warnings
                    printf("Quintuplet excess alert! Module index = %d\n", lowerModule1);
#endif
                }
                else
                {
                    unsigned int quintupletModuleIndex = atomicAdd(&quintupletsInGPU.nQuintuplets[lowerModule1], 1);
                    //this if statement should never get executed!
                    if(rangesInGPU.quintupletModuleIndices[lowerModule1] == -1)
                    {
                        printf("Quintuplets : no memory for module at module index = %d\n", lowerModule1);
                    }
                    else
                    {
                        unsigned int quintupletIndex = rangesInGPU.quintupletModuleIndices[lowerModule1] +  quintupletModuleIndex;
                        float phi = mdsInGPU.anchorPhi[segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*innerTripletIndex+layer2_adjustment]]];
                        float eta = mdsInGPU.anchorEta[segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*innerTripletIndex+layer2_adjustment]]];
                        float pt = (innerRadius+outerRadius)*3.8f*1.602f/(2*100*5.39f);
                        float scores = chiSquared + nonAnchorChiSquared;
                        addQuintupletToMemory(tripletsInGPU, quintupletsInGPU, innerTripletIndex, outerTripletIndex, lowerModule1, lowerModule2, lowerModule3, lowerModule4, lowerModule5, innerRadius, bridgeRadius, outerRadius, regressionG, regressionF, regressionRadius, rzChiSquared, chiSquared, nonAnchorChiSquared, pt,eta,phi,scores,layer,quintupletIndex, TightCutFlag);

                        tripletsInGPU.partOfT5[quintupletsInGPU.tripletIndices[2 * quintupletIndex]] = true;
                        tripletsInGPU.partOfT5[quintupletsInGPU.tripletIndices[2 * quintupletIndex + 1]] = true;
                    }
                }
            }
        }
      }
    }
}

__device__ bool SDL::runQuintupletDefaultAlgoBBBB(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& innerOuterLowerModuleIndex, uint16_t& outerInnerLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex,
        unsigned int& fourthMDIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& dPhi, float& betaIn, float&
        betaOut, float& pt_beta, float& zLo, float& zHi, float& zLoPointed, float& zHiPointed, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut)
{
    bool pass = true;

    bool isPS_InLo = (modulesInGPU.moduleType[innerInnerLowerModuleIndex] == SDL::PS);
    bool isPS_OutLo = (modulesInGPU.moduleType[outerInnerLowerModuleIndex] == SDL::PS);

    float rt_InLo = mdsInGPU.anchorRt[firstMDIndex];
    float rt_InOut = mdsInGPU.anchorRt[secondMDIndex];
    float rt_OutLo = mdsInGPU.anchorRt[thirdMDIndex];

    float z_InLo = mdsInGPU.anchorZ[firstMDIndex];
    float z_InOut = mdsInGPU.anchorZ[secondMDIndex];
    float z_OutLo = mdsInGPU.anchorZ[thirdMDIndex];

    float alpha1GeV_OutLo = asinf(fminf(rt_OutLo * SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax));

    float rtRatio_OutLoInLo = rt_OutLo / rt_InLo; // Outer segment beginning rt divided by inner segment beginning rt;
    float dzDrtScale = tanf(alpha1GeV_OutLo) / alpha1GeV_OutLo; // The track can bend in r-z plane slightly
    float zpitch_InLo = (isPS_InLo ? SDL::pixelPSZpitch : SDL::strip2SZpitch);
    float zpitch_OutLo = (isPS_OutLo ? SDL::pixelPSZpitch : SDL::strip2SZpitch);

    zHi = z_InLo + (z_InLo + SDL::deltaZLum) * (rtRatio_OutLoInLo - 1.f) * (z_InLo < 0.f ? 1.f : dzDrtScale) + (zpitch_InLo + zpitch_OutLo);
    zLo = z_InLo + (z_InLo - SDL::deltaZLum) * (rtRatio_OutLoInLo - 1.f) * (z_InLo > 0.f ? 1.f : dzDrtScale) - (zpitch_InLo + zpitch_OutLo);


    //Cut 1 - z compatibility
    zOut = z_OutLo;
    rtOut = rt_OutLo;
    pass = pass and ((z_OutLo >= zLo) & (z_OutLo <= zHi));
    if(not pass) return pass;

    float drt_OutLo_InLo = (rt_OutLo - rt_InLo);
    float r3_InLo = sqrtf(z_InLo * z_InLo + rt_InLo * rt_InLo);
    float drt_InSeg = rt_InOut - rt_InLo;
    float dz_InSeg = z_InOut - z_InLo;
    float dr3_InSeg = sqrtf(rt_InOut * rt_InOut + z_InOut * z_InOut) - sqrtf(rt_InLo * rt_InLo + z_InLo * z_InLo);

    float coshEta = dr3_InSeg/drt_InSeg;
    float dzErr = (zpitch_InLo + zpitch_OutLo) * (zpitch_InLo + zpitch_OutLo) * 2.f;

    float sdlThetaMulsF = 0.015f * sqrtf(0.1f + 0.2f * (rt_OutLo - rt_InLo) / 50.f) * sqrtf(r3_InLo / rt_InLo);
    float sdlMuls = sdlThetaMulsF * 3.f / SDL::ptCut * 4.f; // will need a better guess than x4?
    dzErr += sdlMuls * sdlMuls * drt_OutLo_InLo * drt_OutLo_InLo / 3.f * coshEta * coshEta; //sloppy
    dzErr = sqrtf(dzErr);

    // Constructing upper and lower bound
    const float dzMean = dz_InSeg / drt_InSeg * drt_OutLo_InLo;
    const float zWindow = dzErr / drt_InSeg * drt_OutLo_InLo + (zpitch_InLo + zpitch_OutLo); //FIXME for SDL::ptCut lower than ~0.8 need to add curv path correction
    zLoPointed = z_InLo + dzMean * (z_InLo > 0.f ? 1.f : dzDrtScale) - zWindow;
    zHiPointed = z_InLo + dzMean * (z_InLo < 0.f ? 1.f : dzDrtScale) + zWindow;

    // Cut #2: Pointed Z (Inner segment two MD points to outer segment inner MD)
    pass =  pass and ((z_OutLo >= zLoPointed) & (z_OutLo <= zHiPointed));
    if(not pass) return pass;

    float sdlPVoff = 0.1f/rt_OutLo;
    sdlCut = alpha1GeV_OutLo + sqrtf(sdlMuls * sdlMuls + sdlPVoff * sdlPVoff);

    deltaPhiPos = SDL::deltaPhi(mdsInGPU.anchorX[secondMDIndex], mdsInGPU.anchorY[secondMDIndex], mdsInGPU.anchorX[fourthMDIndex], mdsInGPU.anchorY[fourthMDIndex]);
    // Cut #3: FIXME:deltaPhiPos can be tighter
    pass = pass and (fabsf(deltaPhiPos) <= sdlCut);
    if(not pass) return pass;

    float midPointX = 0.5f*(mdsInGPU.anchorX[firstMDIndex] + mdsInGPU.anchorX[thirdMDIndex]);
    float midPointY = 0.5f* (mdsInGPU.anchorY[firstMDIndex] + mdsInGPU.anchorY[thirdMDIndex]);
    float midPointZ = 0.5f*(mdsInGPU.anchorZ[firstMDIndex] + mdsInGPU.anchorZ[thirdMDIndex]);
    float diffX = mdsInGPU.anchorX[thirdMDIndex] - mdsInGPU.anchorX[firstMDIndex];
    float diffY = mdsInGPU.anchorY[thirdMDIndex] - mdsInGPU.anchorY[firstMDIndex];
    float diffZ = mdsInGPU.anchorZ[thirdMDIndex] - mdsInGPU.anchorZ[firstMDIndex];

    dPhi = SDL::deltaPhi(midPointX, midPointY, diffX, diffY);

    // Cut #4: deltaPhiChange
    pass = pass and (fabsf(dPhi) <= sdlCut);
    //lots of array accesses below. Cut here!
    if(not pass) return pass;

    // First obtaining the raw betaIn and betaOut values without any correction and just purely based on the mini-doublet hit positions

    float alpha_InLo  = __H2F(segmentsInGPU.dPhiChanges[innerSegmentIndex]);
    float alpha_OutLo = __H2F(segmentsInGPU.dPhiChanges[outerSegmentIndex]);

    bool isEC_lastLayer = modulesInGPU.subdets[outerOuterLowerModuleIndex] == SDL::Endcap and modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::TwoS;

    float alpha_OutUp,alpha_OutUp_highEdge,alpha_OutUp_lowEdge;

    alpha_OutUp = SDL::deltaPhi(mdsInGPU.anchorX[fourthMDIndex], mdsInGPU.anchorY[fourthMDIndex], mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex], mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]);

    alpha_OutUp_highEdge = alpha_OutUp;
    alpha_OutUp_lowEdge = alpha_OutUp;

    float tl_axis_x = mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[firstMDIndex];
    float tl_axis_y = mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[firstMDIndex];
    float tl_axis_z = mdsInGPU.anchorZ[fourthMDIndex] - mdsInGPU.anchorZ[firstMDIndex];
    float tl_axis_highEdge_x = tl_axis_x;
    float tl_axis_highEdge_y = tl_axis_y;
    float tl_axis_lowEdge_x = tl_axis_x;
    float tl_axis_lowEdge_y = tl_axis_y;

    betaIn = alpha_InLo - SDL::deltaPhi(mdsInGPU.anchorX[firstMDIndex], mdsInGPU.anchorY[firstMDIndex], tl_axis_x, tl_axis_y);

    float betaInRHmin = betaIn;
    float betaInRHmax = betaIn;
    betaOut = -alpha_OutUp + SDL::deltaPhi(mdsInGPU.anchorX[fourthMDIndex], mdsInGPU.anchorY[fourthMDIndex], tl_axis_x, tl_axis_y);

    float betaOutRHmin = betaOut;
    float betaOutRHmax = betaOut;

    if(isEC_lastLayer)
    {
        alpha_OutUp_highEdge = SDL::deltaPhi(mdsInGPU.anchorHighEdgeX[fourthMDIndex], mdsInGPU.anchorHighEdgeY[fourthMDIndex], mdsInGPU.anchorHighEdgeX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex], mdsInGPU.anchorHighEdgeY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]);
        alpha_OutUp_lowEdge = SDL::deltaPhi(mdsInGPU.anchorLowEdgeX[fourthMDIndex], mdsInGPU.anchorLowEdgeY[fourthMDIndex], mdsInGPU.anchorLowEdgeX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex], mdsInGPU.anchorLowEdgeY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]);

        tl_axis_highEdge_x = mdsInGPU.anchorHighEdgeX[fourthMDIndex] - mdsInGPU.anchorX[firstMDIndex];
        tl_axis_highEdge_y = mdsInGPU.anchorHighEdgeY[fourthMDIndex] - mdsInGPU.anchorY[firstMDIndex];
        tl_axis_lowEdge_x = mdsInGPU.anchorLowEdgeX[fourthMDIndex] - mdsInGPU.anchorX[firstMDIndex];
        tl_axis_lowEdge_y = mdsInGPU.anchorLowEdgeY[fourthMDIndex] - mdsInGPU.anchorY[firstMDIndex];


        betaOutRHmin = -alpha_OutUp_highEdge + SDL::deltaPhi(mdsInGPU.anchorHighEdgeX[fourthMDIndex], mdsInGPU.anchorHighEdgeY[fourthMDIndex], tl_axis_highEdge_x, tl_axis_highEdge_y);
        betaOutRHmax = -alpha_OutUp_lowEdge + SDL::deltaPhi(mdsInGPU.anchorLowEdgeX[fourthMDIndex], mdsInGPU.anchorLowEdgeY[fourthMDIndex], tl_axis_lowEdge_x, tl_axis_lowEdge_y);
    }

    //beta computation
    float drt_tl_axis = sqrtf(tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);
    float drt_tl_lowEdge = sqrtf(tl_axis_lowEdge_x * tl_axis_lowEdge_x + tl_axis_lowEdge_y * tl_axis_lowEdge_y);
    float drt_tl_highEdge = sqrtf(tl_axis_highEdge_x * tl_axis_highEdge_x + tl_axis_highEdge_y * tl_axis_highEdge_y);

    float corrF = 1.f;
    //innerOuterAnchor - innerInnerAnchor
    const float rt_InSeg = sqrtf((mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) * (mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) + (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]) * (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]));
    betaInCut = asinf(fminf((-rt_InSeg * corrF + drt_tl_axis) * SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax)) + (0.02f / drt_InSeg);

    //Cut #5: first beta cut
    pass = pass and (fabsf(betaInRHmin) < betaInCut);
    if(not pass) return pass;

    float betaAv = 0.5f * (betaIn + betaOut);
    pt_beta = drt_tl_axis * SDL::k2Rinv1GeVf/sinf(betaAv);
    int lIn = 5;
    int lOut = isEC_lastLayer ? 11 : 5;
    float sdOut_dr = sqrtf((mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex]) * (mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex]) + (mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]) * (mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]));
    float sdOut_d = mdsInGPU.anchorRt[fourthMDIndex] - mdsInGPU.anchorRt[thirdMDIndex];

    const float diffDr = fabsf(rt_InSeg - sdOut_dr) / fabsf(rt_InSeg + sdOut_dr);

    SDL::runDeltaBetaIterationsT5(betaIn, betaOut, betaAv, pt_beta, rt_InSeg, sdOut_dr, drt_tl_axis, lIn);

    const float betaInMMSF = (fabsf(betaInRHmin + betaInRHmax) > 0) ? (2.f * betaIn / fabsf(betaInRHmin + betaInRHmax)) : 0.f; //mean value of min,max is the old betaIn
    const float betaOutMMSF = (fabsf(betaOutRHmin + betaOutRHmax) > 0) ? (2.f * betaOut / fabsf(betaOutRHmin + betaOutRHmax)) : 0.f;
    betaInRHmin *= betaInMMSF;
    betaInRHmax *= betaInMMSF;
    betaOutRHmin *= betaOutMMSF;
    betaOutRHmax *= betaOutMMSF;

    const float dBetaMuls = sdlThetaMulsF * 4.f / fminf(fabsf(pt_beta), SDL::pt_betaMax); //need to confimm the range-out value of 7 GeV


    const float alphaInAbsReg = fmaxf(fabsf(alpha_InLo), asinf(fminf(rt_InLo * SDL::k2Rinv1GeVf / 3.0f, SDL::sinAlphaMax)));
    const float alphaOutAbsReg = fmaxf(fabs(alpha_OutLo), asinf(fminf(rt_OutLo * SDL::k2Rinv1GeVf / 3.0f, SDL::sinAlphaMax)));
    const float dBetaInLum = lIn < 11 ? 0.0f : fabsf(alphaInAbsReg*SDL::deltaZLum / z_InLo);
    const float dBetaOutLum = lOut < 11 ? 0.0f : fabsf(alphaOutAbsReg*SDL::deltaZLum / z_OutLo);
    const float dBetaLum2 = (dBetaInLum + dBetaOutLum) * (dBetaInLum + dBetaOutLum);
    const float sinDPhi = sinf(dPhi);

    const float dBetaRIn2 = 0; // TODO-RH
    // const float dBetaROut2 = 0; // TODO-RH
    float dBetaROut = 0;
    if(isEC_lastLayer)
    {
        dBetaROut = (sqrtf(mdsInGPU.anchorHighEdgeX[fourthMDIndex] * mdsInGPU.anchorHighEdgeX[fourthMDIndex] + mdsInGPU.anchorHighEdgeY[fourthMDIndex] * mdsInGPU.anchorHighEdgeY[fourthMDIndex]) - sqrtf(mdsInGPU.anchorLowEdgeX[fourthMDIndex] * mdsInGPU.anchorLowEdgeX[fourthMDIndex] + mdsInGPU.anchorLowEdgeY[fourthMDIndex] * mdsInGPU.anchorLowEdgeY[fourthMDIndex])) * sinDPhi / drt_tl_axis;
    }

    const float dBetaROut2 =  dBetaROut * dBetaROut;

    betaOutCut = asinf(fminf(drt_tl_axis*SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax)) //FIXME: need faster version
        + (0.02f / sdOut_d) + sqrtf(dBetaLum2 + dBetaMuls*dBetaMuls);

    //Cut #6: The real beta cut
    pass =  pass and ((fabsf(betaOut) < betaOutCut));
    if(not pass) return pass;

    float pt_betaIn = drt_tl_axis * SDL::k2Rinv1GeVf/sinf(betaIn);
    float pt_betaOut = drt_tl_axis * SDL::k2Rinv1GeVf / sinf(betaOut);
    float dBetaRes = 0.02f/fminf(sdOut_d,drt_InSeg);
    float dBetaCut2 = (dBetaRes*dBetaRes * 2.0f + dBetaMuls * dBetaMuls + dBetaLum2 + dBetaRIn2 + dBetaROut2
            + 0.25f * (fabsf(betaInRHmin - betaInRHmax) + fabsf(betaOutRHmin - betaOutRHmax)) * (fabsf(betaInRHmin - betaInRHmax) + fabsf(betaOutRHmin - betaOutRHmax)));

    float dBeta = betaIn - betaOut;
    deltaBetaCut = sqrtf(dBetaCut2);
    pass = pass and (dBeta * dBeta <= dBetaCut2);

    return pass;
}

__device__ bool SDL::runQuintupletDefaultAlgoBBEE(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& innerOuterLowerModuleIndex, uint16_t& outerInnerLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex,
        unsigned int& fourthMDIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& dPhi, float& betaIn, float&
        betaOut, float& pt_beta, float& zLo, float& rtLo, float& rtHi, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ)
{
    bool pass = true;
    bool isPS_InLo = (modulesInGPU.moduleType[innerInnerLowerModuleIndex] == SDL::PS);
    bool isPS_OutLo = (modulesInGPU.moduleType[outerInnerLowerModuleIndex] == SDL::PS);

    float rt_InLo = mdsInGPU.anchorRt[firstMDIndex];
    float rt_InOut = mdsInGPU.anchorRt[secondMDIndex];
    float rt_OutLo = mdsInGPU.anchorRt[thirdMDIndex];

    float z_InLo = mdsInGPU.anchorZ[firstMDIndex];
    float z_InOut = mdsInGPU.anchorZ[secondMDIndex];
    float z_OutLo = mdsInGPU.anchorZ[thirdMDIndex];

    float alpha1GeV_OutLo = asinf(fminf(rt_OutLo * SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax));

    float rtRatio_OutLoInLo = rt_OutLo / rt_InLo; // Outer segment beginning rt divided by inner segment beginning rt;
    float dzDrtScale = tanf(alpha1GeV_OutLo) / alpha1GeV_OutLo; // The track can bend in r-z plane slightly
    float zpitch_InLo = (isPS_InLo ? SDL::pixelPSZpitch : SDL::strip2SZpitch);
    float zpitch_OutLo = (isPS_OutLo ? SDL::pixelPSZpitch : SDL::strip2SZpitch);
    float zGeom = zpitch_InLo + zpitch_OutLo;

    zLo = z_InLo + (z_InLo - SDL::deltaZLum) * (rtRatio_OutLoInLo - 1.f) * (z_InLo > 0.f ? 1.f : dzDrtScale) - zGeom;

    // Cut #0: Preliminary (Only here in endcap case)
    pass = pass and (z_InLo * z_OutLo > 0);
    if(not pass) return pass;

    float dLum = copysignf(SDL::deltaZLum, z_InLo);
    bool isOutSgInnerMDPS = modulesInGPU.moduleType[outerInnerLowerModuleIndex] == SDL::PS;
    float rtGeom1 = isOutSgInnerMDPS ? SDL::pixelPSZpitch : SDL::strip2SZpitch;
    float zGeom1 = copysignf(zGeom,z_InLo);
    rtLo = rt_InLo * (1.f + (z_OutLo - z_InLo - zGeom1) / (z_InLo + zGeom1 + dLum) / dzDrtScale) - rtGeom1; //slope correction only on the lower end
    zOut = z_OutLo;
    rtOut = rt_OutLo;

    //Cut #1: rt condition
    pass =  pass and (rtOut >= rtLo);
    if(not pass) return pass;

    float zInForHi = z_InLo - zGeom1 - dLum;
    if(zInForHi * z_InLo < 0)
    {
        zInForHi = copysignf(0.1f,z_InLo);
    }
    rtHi = rt_InLo * (1.f + (z_OutLo - z_InLo + zGeom1) / zInForHi) + rtGeom1;

    //Cut #2: rt condition
    pass =  pass and ((rt_OutLo >= rtLo) & (rt_OutLo <= rtHi));
    if(not pass) return pass;

    float rIn = sqrtf(z_InLo * z_InLo + rt_InLo * rt_InLo);
    const float drtSDIn = rt_InOut - rt_InLo;
    const float dzSDIn = z_InOut - z_InLo;
    const float dr3SDIn = sqrtf(rt_InOut * rt_InOut + z_InOut * z_InOut) - sqrtf(rt_InLo * rt_InLo + z_InLo * z_InLo);

    const float coshEta = dr3SDIn / drtSDIn; //direction estimate
    const float dzOutInAbs = fabsf(z_OutLo - z_InLo);
    const float multDzDr = dzOutInAbs * coshEta / (coshEta * coshEta - 1.f);
    const float zGeom1_another = SDL::pixelPSZpitch; //What's this?
    kZ = (z_OutLo - z_InLo) / dzSDIn;
    float drtErr = zGeom1_another * zGeom1_another * drtSDIn * drtSDIn / dzSDIn / dzSDIn * (1.f - 2.f * kZ + 2.f * kZ * kZ); //Notes:122316
    const float sdlThetaMulsF = 0.015f * sqrtf(0.1f + 0.2f * (rt_OutLo - rt_InLo) / 50.f) * sqrtf(rIn / rt_InLo);
    const float sdlMuls = sdlThetaMulsF * 3.f / SDL::ptCut * 4.f; //will need a better guess than x4?
    drtErr += sdlMuls * sdlMuls * multDzDr * multDzDr / 3.f * coshEta * coshEta; //sloppy: relative muls is 1/3 of total muls
    drtErr = sqrtf(drtErr);
    const float drtMean = drtSDIn * dzOutInAbs / fabsf(dzSDIn); //
    const float rtWindow = drtErr + rtGeom1;
    const float rtLo_another = rt_InLo + drtMean / dzDrtScale - rtWindow;
    const float rtHi_another = rt_InLo + drtMean + rtWindow;

    //Cut #3: rt-z pointed
    pass =  pass and ((kZ >= 0) & (rtOut >= rtLo) & (rtOut <= rtHi));
    if(not pass) return pass;

    const float sdlPVoff = 0.1f / rt_OutLo;
    sdlCut = alpha1GeV_OutLo + sqrtf(sdlMuls * sdlMuls + sdlPVoff*sdlPVoff);


    deltaPhiPos = SDL::deltaPhi(mdsInGPU.anchorX[secondMDIndex], mdsInGPU.anchorY[secondMDIndex], mdsInGPU.anchorX[fourthMDIndex], mdsInGPU.anchorY[fourthMDIndex]);


    //Cut #4: deltaPhiPos can be tighter
    pass =  pass and (fabsf(deltaPhiPos) <= sdlCut);
    if(not pass) return pass;

    float midPointX = 0.5f*(mdsInGPU.anchorX[firstMDIndex] + mdsInGPU.anchorX[thirdMDIndex]);
    float midPointY = 0.5f* (mdsInGPU.anchorY[firstMDIndex] + mdsInGPU.anchorY[thirdMDIndex]);
    float midPointZ = 0.5f*(mdsInGPU.anchorZ[firstMDIndex] + mdsInGPU.anchorZ[thirdMDIndex]);
    float diffX = mdsInGPU.anchorX[thirdMDIndex] - mdsInGPU.anchorX[firstMDIndex];
    float diffY = mdsInGPU.anchorY[thirdMDIndex] - mdsInGPU.anchorY[firstMDIndex];
    float diffZ = mdsInGPU.anchorZ[thirdMDIndex] - mdsInGPU.anchorZ[firstMDIndex];

    dPhi = SDL::deltaPhi(midPointX, midPointY, diffX, diffY);
    // Cut #5: deltaPhiChange
    pass =  pass and (fabsf(dPhi) <= sdlCut);
    if(not pass) return pass;

    float sdIn_alpha     = __H2F(segmentsInGPU.dPhiChanges[innerSegmentIndex]);
    float sdIn_alpha_min = __H2F(segmentsInGPU.dPhiChangeMins[innerSegmentIndex]);
    float sdIn_alpha_max = __H2F(segmentsInGPU.dPhiChangeMaxs[innerSegmentIndex]);
    float sdOut_alpha = sdIn_alpha; //weird

    float sdOut_alphaOut = SDL::deltaPhi(mdsInGPU.anchorX[fourthMDIndex], mdsInGPU.anchorY[fourthMDIndex], mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex], mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]);

    float sdOut_alphaOut_min = SDL::phi_mpi_pi(__H2F(segmentsInGPU.dPhiChangeMins[outerSegmentIndex]) - __H2F(segmentsInGPU.dPhiMins[outerSegmentIndex]));
    float sdOut_alphaOut_max = SDL::phi_mpi_pi(__H2F(segmentsInGPU.dPhiChangeMaxs[outerSegmentIndex]) - __H2F(segmentsInGPU.dPhiMaxs[outerSegmentIndex]));

    float tl_axis_x = mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[firstMDIndex];
    float tl_axis_y = mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[firstMDIndex];
    float tl_axis_z = mdsInGPU.anchorZ[fourthMDIndex] - mdsInGPU.anchorZ[firstMDIndex];

    betaIn = sdIn_alpha - SDL::deltaPhi(mdsInGPU.anchorX[firstMDIndex], mdsInGPU.anchorY[firstMDIndex], tl_axis_x, tl_axis_y);

    float betaInRHmin = betaIn;
    float betaInRHmax = betaIn;
    betaOut = -sdOut_alphaOut + SDL::deltaPhi(mdsInGPU.anchorX[fourthMDIndex], mdsInGPU.anchorY[fourthMDIndex], tl_axis_x, tl_axis_y);

    float betaOutRHmin = betaOut;
    float betaOutRHmax = betaOut;

    bool isEC_secondLayer = (modulesInGPU.subdets[innerOuterLowerModuleIndex] == SDL::Endcap) and (modulesInGPU.moduleType[innerOuterLowerModuleIndex] == SDL::TwoS);

    if(isEC_secondLayer)
    {
        betaInRHmin = betaIn - sdIn_alpha_min + sdIn_alpha;
        betaInRHmax = betaIn - sdIn_alpha_max + sdIn_alpha;
    }

    betaOutRHmin = betaOut - sdOut_alphaOut_min + sdOut_alphaOut;
    betaOutRHmax = betaOut - sdOut_alphaOut_max + sdOut_alphaOut;

    float swapTemp;
    if(fabsf(betaOutRHmin) > fabsf(betaOutRHmax))
    {
        swapTemp = betaOutRHmin;
        betaOutRHmin = betaOutRHmax;
        betaOutRHmax = swapTemp;
    }

    if(fabsf(betaInRHmin) > fabsf(betaInRHmax))
    {
        swapTemp = betaInRHmin;
        betaInRHmin = betaInRHmax;
        betaInRHmax = swapTemp;
    }

    float sdIn_dr = sqrtf((mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) * (mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) + (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]) * (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]));
    float sdIn_d = rt_InOut - rt_InLo;

    float dr = sqrtf(tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);
    const float corrF = 1.f;
    betaInCut = asinf(fminf((-sdIn_dr * corrF + dr) * SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax)) + (0.02f / sdIn_d);

    //Cut #6: first beta cut
    pass =  pass and (fabsf(betaInRHmin) < betaInCut);
    if(not pass) return pass;

    float betaAv = 0.5f * (betaIn + betaOut);
    pt_beta = dr * SDL::k2Rinv1GeVf / sinf(betaAv);

    float lIn = 5;
    float lOut = 11;

    float sdOut_dr = sqrtf((mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex]) * (mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex]) + (mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]) * (mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]));
    float sdOut_d = mdsInGPU.anchorRt[fourthMDIndex] - mdsInGPU.anchorRt[thirdMDIndex];

    SDL::runDeltaBetaIterationsT5(betaIn, betaOut, betaAv, pt_beta, sdIn_dr, sdOut_dr, dr, lIn);

     const float betaInMMSF = (fabsf(betaInRHmin + betaInRHmax) > 0) ? (2.f * betaIn / fabsf(betaInRHmin + betaInRHmax)) : 0.; //mean value of min,max is the old betaIn
    const float betaOutMMSF = (fabsf(betaOutRHmin + betaOutRHmax) > 0) ? (2.f * betaOut / fabsf(betaOutRHmin + betaOutRHmax)) : 0.;
    betaInRHmin *= betaInMMSF;
    betaInRHmax *= betaInMMSF;
    betaOutRHmin *= betaOutMMSF;
    betaOutRHmax *= betaOutMMSF;

    const float dBetaMuls = sdlThetaMulsF * 4.f / fminf(fabsf(pt_beta), SDL::pt_betaMax); //need to confirm the range-out value of 7 GeV

    const float alphaInAbsReg = fmaxf(fabsf(sdIn_alpha), asinf(fminf(rt_InLo * SDL::k2Rinv1GeVf / 3.0f, SDL::sinAlphaMax)));
    const float alphaOutAbsReg = fmaxf(fabsf(sdOut_alpha), asinf(fminf(rt_OutLo * SDL::k2Rinv1GeVf / 3.0f, SDL::sinAlphaMax)));
    const float dBetaInLum = lIn < 11 ? 0.0f : fabsf(alphaInAbsReg*SDL::deltaZLum / z_InLo);
    const float dBetaOutLum = lOut < 11 ? 0.0f : fabsf(alphaOutAbsReg*SDL::deltaZLum / z_OutLo);
    const float dBetaLum2 = (dBetaInLum + dBetaOutLum) * (dBetaInLum + dBetaOutLum);
    const float sinDPhi = sinf(dPhi);

    const float dBetaRIn2 = 0; // TODO-RH
    // const float dBetaROut2 = 0; // TODO-RH
    float dBetaROut = 0;
    if(modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::TwoS)
    {
        dBetaROut = (sqrtf(mdsInGPU.anchorHighEdgeX[fourthMDIndex] * mdsInGPU.anchorHighEdgeX[fourthMDIndex] + mdsInGPU.anchorHighEdgeY[fourthMDIndex] * mdsInGPU.anchorHighEdgeY[fourthMDIndex]) - sqrtf(mdsInGPU.anchorLowEdgeX[fourthMDIndex] * mdsInGPU.anchorLowEdgeX[fourthMDIndex] + mdsInGPU.anchorLowEdgeY[fourthMDIndex] * mdsInGPU.anchorLowEdgeY[fourthMDIndex])) * sinDPhi / dr;
    }

    const float dBetaROut2 = dBetaROut * dBetaROut;
    betaOutCut = asinf(fminf(dr*SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax)) //FIXME: need faster version
        + (0.02f / sdOut_d) + sqrtf(dBetaLum2 + dBetaMuls*dBetaMuls);

    //Cut #6: The real beta cut
    pass =  pass and (fabsf(betaOut) < betaOutCut);
    if(not pass) return pass;

    float pt_betaIn = dr * SDL::k2Rinv1GeVf/sinf(betaIn);
    float pt_betaOut = dr * SDL::k2Rinv1GeVf / sinf(betaOut);
    float dBetaRes = 0.02f/fminf(sdOut_d,sdIn_d);
    float dBetaCut2 = (dBetaRes*dBetaRes * 2.0f + dBetaMuls * dBetaMuls + dBetaLum2 + dBetaRIn2 + dBetaROut2
            + 0.25f * (fabsf(betaInRHmin - betaInRHmax) + fabsf(betaOutRHmin - betaOutRHmax)) * (fabsf(betaInRHmin - betaInRHmax) + fabsf(betaOutRHmin - betaOutRHmax)));
    float dBeta = betaIn - betaOut;
    deltaBetaCut = sqrtf(dBetaCut2);
    //Cut #7: Cut on dBet
    pass =  pass and (dBeta * dBeta <= dBetaCut2);

    return pass;
}

__device__ bool SDL::runQuintupletDefaultAlgoEEEE(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& innerOuterLowerModuleIndex, uint16_t& outerInnerLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex,
        unsigned int& fourthMDIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& dPhi, float& betaIn, float&
        betaOut, float& pt_beta, float& zLo, float& rtLo, float& rtHi, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ)
{
    bool pass = true;

    bool isPS_InLo = (modulesInGPU.moduleType[innerInnerLowerModuleIndex] == SDL::PS);
    bool isPS_OutLo = (modulesInGPU.moduleType[outerInnerLowerModuleIndex] == SDL::PS);

    float rt_InLo = mdsInGPU.anchorRt[firstMDIndex];
    float rt_InOut = mdsInGPU.anchorRt[secondMDIndex];
    float rt_OutLo = mdsInGPU.anchorRt[thirdMDIndex];

    float z_InLo = mdsInGPU.anchorZ[firstMDIndex];
    float z_InOut = mdsInGPU.anchorZ[secondMDIndex];
    float z_OutLo = mdsInGPU.anchorZ[thirdMDIndex];

    float alpha1GeV_OutLo = asinf(fminf(rt_OutLo * SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax));

    float rtRatio_OutLoInLo = rt_OutLo / rt_InLo; // Outer segment beginning rt divided by inner segment beginning rt;
    float dzDrtScale = tanf(alpha1GeV_OutLo) / alpha1GeV_OutLo; // The track can bend in r-z plane slightly
    float zpitch_InLo = (isPS_InLo ? SDL::pixelPSZpitch : SDL::strip2SZpitch);
    float zpitch_OutLo = (isPS_OutLo ? SDL::pixelPSZpitch : SDL::strip2SZpitch);
    float zGeom = zpitch_InLo + zpitch_OutLo;

    zLo = z_InLo + (z_InLo - SDL::deltaZLum) * (rtRatio_OutLoInLo - 1.f) * (z_InLo > 0.f ? 1.f : dzDrtScale) - zGeom; //slope-correction only on outer end

    // Cut #0: Preliminary (Only here in endcap case)
    pass =  pass and ((z_InLo * z_OutLo) > 0);
    if(not pass) return pass;

    float dLum = copysignf(SDL::deltaZLum, z_InLo);
    bool isOutSgInnerMDPS = modulesInGPU.moduleType[outerInnerLowerModuleIndex] == SDL::PS;
    bool isInSgInnerMDPS = modulesInGPU.moduleType[innerInnerLowerModuleIndex] == SDL::PS;

    float rtGeom = (isInSgInnerMDPS and isOutSgInnerMDPS) ? 2.f * SDL::pixelPSZpitch : (isInSgInnerMDPS or isOutSgInnerMDPS) ? SDL::pixelPSZpitch + SDL::strip2SZpitch : 2.f * SDL::strip2SZpitch;

    float zGeom1 = copysignf(zGeom,z_InLo);
    float dz = z_OutLo - z_InLo;
    rtLo = rt_InLo * (1.f + dz / (z_InLo + dLum) / dzDrtScale) - rtGeom; //slope correction only on the lower end

    zOut = z_OutLo;
    rtOut = rt_OutLo;

    //Cut #1: rt condition

    rtHi = rt_InLo * (1.f + dz / (z_InLo - dLum)) + rtGeom;

    pass =  pass and ((rtOut >= rtLo) & (rtOut <= rtHi));
    if(not pass) return pass;

    bool isInSgOuterMDPS = modulesInGPU.moduleType[innerOuterLowerModuleIndex] == SDL::PS;

    float drOutIn = rtOut - rt_InLo;
    const float drtSDIn = rt_InOut - rt_InLo;
    const float dzSDIn = z_InOut - z_InLo;
    const float dr3SDIn = sqrtf(rt_InOut * rt_InOut + z_InOut * z_InOut) - sqrtf(rt_InLo * rt_InLo + z_InLo * z_InLo);
    float coshEta = dr3SDIn / drtSDIn; //direction estimate
    float dzOutInAbs =  fabsf(z_OutLo - z_InLo);
    float multDzDr = dzOutInAbs * coshEta / (coshEta * coshEta - 1.f);

    kZ = (z_OutLo - z_InLo) / dzSDIn;
    float sdlThetaMulsF = 0.015f * sqrtf(0.1f + 0.2f * (rt_OutLo - rt_InLo) / 50.f);

    float sdlMuls = sdlThetaMulsF * 3.f / SDL::ptCut * 4.f; //will need a better guess than x4?

    float drtErr = sqrtf(SDL::pixelPSZpitch * SDL::pixelPSZpitch * 2.f / (dzSDIn * dzSDIn) * (dzOutInAbs * dzOutInAbs) + sdlMuls * sdlMuls * multDzDr * multDzDr / 3.f * coshEta * coshEta);

    float drtMean = drtSDIn * dzOutInAbs/fabsf(dzSDIn);
    float rtWindow = drtErr + rtGeom;
    float rtLo_point = rt_InLo + drtMean / dzDrtScale - rtWindow;
    float rtHi_point = rt_InLo + drtMean + rtWindow;

    // Cut #3: rt-z pointed
    // https://github.com/slava77/cms-tkph2-ntuple/blob/superDoubletLinked-91X-noMock/doubletAnalysis.C#L3765

    if (isInSgInnerMDPS and isInSgOuterMDPS) // If both PS then we can point
    {
        pass =  pass and (kZ >= 0 and rtOut >= rtLo_point and rtOut <= rtHi_point);
        if(not pass) return pass;
    }

    float sdlPVoff = 0.1f/rtOut;
    sdlCut = alpha1GeV_OutLo + sqrtf(sdlMuls * sdlMuls + sdlPVoff * sdlPVoff);

    deltaPhiPos = SDL::deltaPhi(mdsInGPU.anchorX[secondMDIndex], mdsInGPU.anchorY[secondMDIndex], mdsInGPU.anchorX[fourthMDIndex], mdsInGPU.anchorY[fourthMDIndex]);

    pass =  pass and (fabsf(deltaPhiPos) <= sdlCut);
    if(not pass) return pass;

    float midPointX = 0.5f*(mdsInGPU.anchorX[firstMDIndex] + mdsInGPU.anchorX[thirdMDIndex]);
    float midPointY = 0.5f* (mdsInGPU.anchorY[firstMDIndex] + mdsInGPU.anchorY[thirdMDIndex]);
    float midPointZ = 0.5f*(mdsInGPU.anchorZ[firstMDIndex] + mdsInGPU.anchorZ[thirdMDIndex]);
    float diffX = mdsInGPU.anchorX[thirdMDIndex] - mdsInGPU.anchorX[firstMDIndex];
    float diffY = mdsInGPU.anchorY[thirdMDIndex] - mdsInGPU.anchorY[firstMDIndex];
    float diffZ = mdsInGPU.anchorZ[thirdMDIndex] - mdsInGPU.anchorZ[firstMDIndex];

    dPhi = SDL::deltaPhi(midPointX, midPointY, diffX, diffY);

    // Cut #5: deltaPhiChange
    pass =  pass and ((fabsf(dPhi) <= sdlCut));
    if(not pass) return pass;

    float sdIn_alpha = __H2F(segmentsInGPU.dPhiChanges[innerSegmentIndex]);
    float sdOut_alpha = sdIn_alpha; //weird
    float sdOut_dPhiPos = SDL::deltaPhi(mdsInGPU.anchorX[thirdMDIndex], mdsInGPU.anchorY[thirdMDIndex], mdsInGPU.anchorX[fourthMDIndex], mdsInGPU.anchorY[fourthMDIndex]);

    float sdOut_dPhiChange = __H2F(segmentsInGPU.dPhiChanges[outerSegmentIndex]);
    float sdOut_dPhiChange_min = __H2F(segmentsInGPU.dPhiChangeMins[outerSegmentIndex]);
    float sdOut_dPhiChange_max = __H2F(segmentsInGPU.dPhiChangeMaxs[outerSegmentIndex]);

    float sdOut_alphaOutRHmin = SDL::phi_mpi_pi(sdOut_dPhiChange_min - sdOut_dPhiPos);
    float sdOut_alphaOutRHmax = SDL::phi_mpi_pi(sdOut_dPhiChange_max - sdOut_dPhiPos);
    float sdOut_alphaOut = SDL::phi_mpi_pi(sdOut_dPhiChange - sdOut_dPhiPos);

    float tl_axis_x = mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[firstMDIndex];
    float tl_axis_y = mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[firstMDIndex];
    float tl_axis_z = mdsInGPU.anchorZ[fourthMDIndex] - mdsInGPU.anchorZ[firstMDIndex];

    betaIn = sdIn_alpha - SDL::deltaPhi(mdsInGPU.anchorX[firstMDIndex], mdsInGPU.anchorY[firstMDIndex], tl_axis_x, tl_axis_y);

    float sdIn_alphaRHmin = __H2F(segmentsInGPU.dPhiChangeMins[innerSegmentIndex]);
    float sdIn_alphaRHmax = __H2F(segmentsInGPU.dPhiChangeMaxs[innerSegmentIndex]);
    float betaInRHmin = betaIn + sdIn_alphaRHmin - sdIn_alpha;
    float betaInRHmax = betaIn + sdIn_alphaRHmax - sdIn_alpha;

    betaOut = -sdOut_alphaOut + SDL::deltaPhi(mdsInGPU.anchorX[fourthMDIndex], mdsInGPU.anchorY[fourthMDIndex], tl_axis_x, tl_axis_y);

    float betaOutRHmin = betaOut - sdOut_alphaOutRHmin + sdOut_alphaOut;
    float betaOutRHmax = betaOut - sdOut_alphaOutRHmax + sdOut_alphaOut;

    float swapTemp;
    if(fabsf(betaOutRHmin) > fabsf(betaOutRHmax))
    {
        swapTemp = betaOutRHmin;
        betaOutRHmin = betaOutRHmax;
        betaOutRHmax = swapTemp;
    }

    if(fabsf(betaInRHmin) > fabsf(betaInRHmax))
    {
        swapTemp = betaInRHmin;
        betaInRHmin = betaInRHmax;
        betaInRHmax = swapTemp;
    }
    float sdIn_dr = sqrtf((mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) * (mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) + (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]) * (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]));
    float sdIn_d = rt_InOut - rt_InLo;

    float dr = sqrtf(tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);
    const float corrF = 1.f;
    betaInCut = asinf(fminf((-sdIn_dr * corrF + dr) * SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax)) + (0.02f / sdIn_d);

    //Cut #6: first beta cut
    pass =  pass and (fabsf(betaInRHmin) < betaInCut);
    if(not pass) return pass;

    float betaAv = 0.5f * (betaIn + betaOut);
    pt_beta = dr * SDL::k2Rinv1GeVf / sinf(betaAv);


    int lIn= 11; //endcap
    int lOut = 13; //endcap

    float sdOut_dr = sqrtf((mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex]) * (mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex]) + (mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]) * (mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]));
    float sdOut_d = mdsInGPU.anchorRt[fourthMDIndex] - mdsInGPU.anchorRt[thirdMDIndex];

    float diffDr = fabsf(sdIn_dr - sdOut_dr)/fabs(sdIn_dr + sdOut_dr);

    SDL::runDeltaBetaIterationsT5(betaIn, betaOut, betaAv, pt_beta, sdIn_dr, sdOut_dr, dr, lIn);

     const float betaInMMSF = (fabsf(betaInRHmin + betaInRHmax) > 0) ? (2.f * betaIn / fabsf(betaInRHmin + betaInRHmax)) : 0.; //mean value of min,max is the old betaIn
    const float betaOutMMSF = (fabsf(betaOutRHmin + betaOutRHmax) > 0) ? (2.f * betaOut / fabsf(betaOutRHmin + betaOutRHmax)) : 0.;
    betaInRHmin *= betaInMMSF;
    betaInRHmax *= betaInMMSF;
    betaOutRHmin *= betaOutMMSF;
    betaOutRHmax *= betaOutMMSF;

    const float dBetaMuls = sdlThetaMulsF * 4.f / fminf(fabsf(pt_beta), SDL::pt_betaMax); //need to confirm the range-out value of 7 GeV

    const float alphaInAbsReg = fmaxf(fabsf(sdIn_alpha), asinf(fminf(rt_InLo * SDL::k2Rinv1GeVf / 3.0f, SDL::sinAlphaMax)));
    const float alphaOutAbsReg = fmaxf(fabsf(sdOut_alpha), asinf(fminf(rt_OutLo * SDL::k2Rinv1GeVf / 3.0f, SDL::sinAlphaMax)));
    const float dBetaInLum = lIn < 11 ? 0.0f : fabsf(alphaInAbsReg*SDL::deltaZLum / z_InLo);
    const float dBetaOutLum = lOut < 11 ? 0.0f : fabsf(alphaOutAbsReg*SDL::deltaZLum / z_OutLo);
    const float dBetaLum2 = (dBetaInLum + dBetaOutLum) * (dBetaInLum + dBetaOutLum);
    const float sinDPhi = sinf(dPhi);

    const float dBetaRIn2 = 0; // TODO-RH
    // const float dBetaROut2 = 0; // TODO-RH
    float dBetaROut2 = 0;//TODO-RH
    betaOutCut = asinf(fminf(dr*SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax)) //FIXME: need faster version
        + (0.02f / sdOut_d) + sqrtf(dBetaLum2 + dBetaMuls*dBetaMuls);

    //Cut #6: The real beta cut
    pass =  pass and (fabsf(betaOut) < betaOutCut);
    if(not pass) return pass;

    float pt_betaIn = dr * SDL::k2Rinv1GeVf/sinf(betaIn);
    float pt_betaOut = dr * SDL::k2Rinv1GeVf / sinf(betaOut);
    float dBetaRes = 0.02f/fminf(sdOut_d,sdIn_d);
    float dBetaCut2 = (dBetaRes*dBetaRes * 2.0f + dBetaMuls * dBetaMuls + dBetaLum2 + dBetaRIn2 + dBetaROut2
            + 0.25f * (fabsf(betaInRHmin - betaInRHmax) + fabsf(betaOutRHmin - betaOutRHmax)) * (fabsf(betaInRHmin - betaInRHmax) + fabsf(betaOutRHmin - betaOutRHmax)));
    float dBeta = betaIn - betaOut;
    //Cut #7: Cut on dBeta
    deltaBetaCut = sqrtf(dBetaCut2);

    pass =  pass and (dBeta * dBeta <= dBetaCut2);

    return pass;
}
__device__ bool SDL::runQuintupletDefaultAlgo(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& innerOuterLowerModuleIndex, uint16_t& outerInnerLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex, unsigned int& fourthMDIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& deltaPhi, float& betaIn, float&
        betaOut, float& pt_beta, float& zLo, float& zHi, float& rtLo, float& rtHi, float& zLoPointed, float& zHiPointed, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ)
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

    short innerInnerLowerModuleSubdet = modulesInGPU.subdets[innerInnerLowerModuleIndex];
    short innerOuterLowerModuleSubdet = modulesInGPU.subdets[innerOuterLowerModuleIndex];
    short outerInnerLowerModuleSubdet = modulesInGPU.subdets[outerInnerLowerModuleIndex];
    short outerOuterLowerModuleSubdet = modulesInGPU.subdets[outerOuterLowerModuleIndex];

    if(innerInnerLowerModuleSubdet == SDL::Barrel
            and innerOuterLowerModuleSubdet == SDL::Barrel
            and outerInnerLowerModuleSubdet == SDL::Barrel
            and outerOuterLowerModuleSubdet == SDL::Barrel)
    {
        return runQuintupletDefaultAlgoBBBB(modulesInGPU,mdsInGPU,segmentsInGPU,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,innerSegmentIndex,outerSegmentIndex,firstMDIndex, secondMDIndex, thirdMDIndex, fourthMDIndex, zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta, zLo, zHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut);
    }

    else if(innerInnerLowerModuleSubdet == SDL::Barrel
            and innerOuterLowerModuleSubdet == SDL::Barrel
            and outerInnerLowerModuleSubdet == SDL::Endcap
            and outerOuterLowerModuleSubdet == SDL::Endcap)
    {
       return runQuintupletDefaultAlgoBBEE(modulesInGPU,mdsInGPU,segmentsInGPU,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,innerSegmentIndex,outerSegmentIndex,firstMDIndex, secondMDIndex, thirdMDIndex, fourthMDIndex, zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta, zLo, rtLo, rtHi, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ);
    }


    else if(innerInnerLowerModuleSubdet == SDL::Barrel
            and innerOuterLowerModuleSubdet == SDL::Barrel
            and outerInnerLowerModuleSubdet == SDL::Barrel
            and outerOuterLowerModuleSubdet == SDL::Endcap)
    {
        return runQuintupletDefaultAlgoBBBB(modulesInGPU,mdsInGPU,segmentsInGPU,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,innerSegmentIndex,outerSegmentIndex,firstMDIndex, secondMDIndex, thirdMDIndex, fourthMDIndex, zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta,zLo, zHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut);

    }

    else if(innerInnerLowerModuleSubdet == SDL::Barrel
            and innerOuterLowerModuleSubdet == SDL::Endcap
            and outerInnerLowerModuleSubdet == SDL::Endcap
            and outerOuterLowerModuleSubdet == SDL::Endcap)
    {
        return runQuintupletDefaultAlgoBBEE(modulesInGPU,mdsInGPU,segmentsInGPU,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,innerSegmentIndex,outerSegmentIndex, firstMDIndex, secondMDIndex, thirdMDIndex, fourthMDIndex, zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta, zLo, rtLo, rtHi, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ);

    }

    else if(innerInnerLowerModuleSubdet == SDL::Endcap
            and innerOuterLowerModuleSubdet == SDL::Endcap
            and outerInnerLowerModuleSubdet == SDL::Endcap
            and outerOuterLowerModuleSubdet == SDL::Endcap)
    {
        return runQuintupletDefaultAlgoEEEE(modulesInGPU,mdsInGPU,segmentsInGPU,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,innerSegmentIndex,outerSegmentIndex, firstMDIndex, secondMDIndex, thirdMDIndex, fourthMDIndex, zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta, zLo, rtLo, rtHi, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ);
    }

    return pass;
}
__device__ void SDL::runDeltaBetaIterationsT5(float& betaIn, float& betaOut, float& betaAv, float & pt_beta, float sdIn_dr, float sdOut_dr, float dr, float lIn)
{
    if (lIn == 0)
    {
        betaOut += copysign(asinf(fminf(sdOut_dr * SDL::k2Rinv1GeVf / fabsf(pt_beta), SDL::sinAlphaMax)), betaOut);
        return;
    }

    if (betaIn * betaOut > 0.f and (fabsf(pt_beta) < 4.f * SDL::pt_betaMax or (lIn >= 11 and fabsf(pt_beta) < 8.f * SDL::pt_betaMax)))   //and the pt_beta is well-defined; less strict for endcap-endcap
    {

        const float betaInUpd  = betaIn + copysignf(asinf(fminf(sdIn_dr * SDL::k2Rinv1GeVf / fabsf(pt_beta), SDL::sinAlphaMax)), betaIn); //FIXME: need a faster version
        const float betaOutUpd = betaOut + copysignf(asinf(fminf(sdOut_dr * SDL::k2Rinv1GeVf / fabsf(pt_beta), SDL::sinAlphaMax)), betaOut); //FIXME: need a faster version
        betaAv = 0.5f * (betaInUpd + betaOutUpd);

        //1st update
        //pt_beta = dr * k2Rinv1GeVf / sinf(betaAv); //get a better pt estimate
        const float pt_beta_inv = 1.f/fabsf(dr * k2Rinv1GeVf / sinf(betaAv)); //get a better pt estimate

        betaIn  += copysignf(asinf(fminf(sdIn_dr * SDL::k2Rinv1GeVf *pt_beta_inv, SDL::sinAlphaMax)), betaIn); //FIXME: need a faster version
        betaOut += copysignf(asinf(fminf(sdOut_dr * SDL::k2Rinv1GeVf *pt_beta_inv, SDL::sinAlphaMax)), betaOut); //FIXME: need a faster version
        //update the av and pt
        betaAv = 0.5f * (betaIn + betaOut);
        //2nd update
        pt_beta = dr * SDL::k2Rinv1GeVf / sinf(betaAv); //get a better pt estimate
    }
    else if (lIn < 11 && fabsf(betaOut) < 0.2f * fabsf(betaIn) && fabsf(pt_beta) < 12.f * SDL::pt_betaMax)   //use betaIn sign as ref
    {

        const float pt_betaIn = dr * k2Rinv1GeVf / sinf(betaIn);

        const float betaInUpd  = betaIn + copysignf(asinf(fminf(sdIn_dr * SDL::k2Rinv1GeVf / fabsf(pt_betaIn), SDL::sinAlphaMax)), betaIn); //FIXME: need a faster version
        const float betaOutUpd = betaOut + copysignf(asinf(fminf(sdOut_dr * SDL::k2Rinv1GeVf / fabsf(pt_betaIn), SDL::sinAlphaMax)), betaIn); //FIXME: need a faster version
        betaAv = (fabsf(betaOut) > 0.2f * fabsf(betaIn)) ? (0.5f * (betaInUpd + betaOutUpd)) : betaInUpd;

        //1st update
        pt_beta = dr * SDL::k2Rinv1GeVf / sin(betaAv); //get a better pt estimate
        betaIn  += copysignf(asinf(fminf(sdIn_dr * SDL::k2Rinv1GeVf / fabsf(pt_beta), SDL::sinAlphaMax)), betaIn); //FIXME: need a faster version
        betaOut += copysignf(asinf(fminf(sdOut_dr * SDL::k2Rinv1GeVf / fabsf(pt_beta), SDL::sinAlphaMax)), betaIn); //FIXME: need a faster version
        //update the av and pt
        betaAv = 0.5f * (betaIn + betaOut);
        //2nd update
        pt_beta = dr * SDL::k2Rinv1GeVf / sin(betaAv); //get a better pt estimate

    }
}
