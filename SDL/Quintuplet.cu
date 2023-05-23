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
__global__ void SDL::createEligibleModulesListForQuintupletsGPU(struct modules& modulesInGPU,struct triplets& tripletsInGPU, struct objectRanges& rangesInGPU)
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
        eta = abs(modulesInGPU.eta[i]);  
        occupancy = 0;

        if (tripletsInGPU.nTriplets[i] == 0) continue;
        if (subdets == SDL::Barrel and layers >= 3) continue;
        if (subdets == SDL::Endcap and layers > 1) continue;

        int nEligibleT5Modules = atomicAdd(&nEligibleT5Modulesx,1);
        if (layers<=3 && subdets==5) category_number = 0;
        else if (layers>=4 && subdets==5) category_number = 1;
        else if (layers<=2 && subdets==4 && rings>=11) category_number = 2;
        else if (layers>=3 && subdets==4 && rings>=8) category_number = 2;
        else if (layers<=2 && subdets==4 && rings<=10) category_number = 3;
        else if (layers>=3 && subdets==4 && rings<=7) category_number = 3;
        if (eta<0.75) eta_number=0;
        else if (eta>0.75 && eta<1.5) eta_number=1;
        else if (eta>1.5 && eta<2.25) eta_number=2;
        else if (eta>2.25 && eta<3) eta_number=3;

        if (category_number == 0 && eta_number == 0) occupancy = 336;
        else if (category_number == 0 && eta_number == 1) occupancy = 414;
        else if (category_number == 0 && eta_number == 2) occupancy = 231;
        else if (category_number == 0 && eta_number == 3) occupancy = 146;
        else if (category_number == 3 && eta_number == 1) occupancy = 0;
        else if (category_number == 3 && eta_number == 2) occupancy = 191;
        else if (category_number == 3 && eta_number == 3) occupancy = 106;

        unsigned int nTotQ = atomicAdd(&nTotalQuintupletsx,occupancy);
        rangesInGPU.quintupletModuleIndices[i] = nTotQ;
        rangesInGPU.quintupletModuleOccupancy[i] = occupancy;
        rangesInGPU.indicesOfEligibleT5Modules[nEligibleT5Modules] = i;
    }
    __syncthreads();
    if(threadIdx.x==0){
        *rangesInGPU.nEligibleT5Modules = static_cast<uint16_t>(nEligibleT5Modulesx);
        *rangesInGPU.device_nTotalQuints = nTotalQuintupletsx;
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
    pass = pass and passT5RZConstraint(modulesInGPU, mdsInGPU, firstMDIndex, secondMDIndex, thirdMDIndex, fourthMDIndex, fifthMDIndex, lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4, lowerModuleIndex5, rzChiSquared, inner_pt, innerRadius, g, f, TightCutFlag);

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
    pass = pass and tempPass;
    if(not pass) return pass;

    short layer2_adjustment = 0;
    if (modulesInGPU.layers[lowerModuleIndex1] == 1)
    {
        layer2_adjustment = 1; // get upper segment to be in second layer
    }
    bool is_endcap1 = (modulesInGPU.subdets[lowerModuleIndex1] == 4);
    bool is_endcap2 = (modulesInGPU.subdets[lowerModuleIndex2] == 4);
    bool is_endcap3 = (modulesInGPU.subdets[lowerModuleIndex3] == 4);
    bool is_endcap4 = (modulesInGPU.subdets[lowerModuleIndex4] == 4);
    bool is_endcap5 = (modulesInGPU.subdets[lowerModuleIndex5] == 4);
    float x[38] = { 
        log10(2.99792458e-3f*3.8f*innerRadius),        // inner t3_pt
        mdsInGPU.anchorEta[firstMDIndex],       // inner (hit 1) t3_0_eta
        mdsInGPU.anchorPhi[firstMDIndex],       // inner (hit 1) t3_0_phi
        mdsInGPU.anchorZ[firstMDIndex],         // inner (hit 1) t3_0_z
        sqrtf(x1*x1 + y1*y1),                   // inner (hit 1) t3_0_r
        float(modulesInGPU.layers[lowerModuleIndex1] + 6*is_endcap1), // inner (hit 1) t3_0_layer
        mdsInGPU.anchorEta[secondMDIndex],      // inner (hit 2) t3_2_eta
        mdsInGPU.anchorPhi[secondMDIndex],      // inner (hit 2) t3_2_phi
        mdsInGPU.anchorZ[secondMDIndex],        // inner (hit 2) t3_2_z
        sqrtf(x2*x2 + y2*y2),                   // inner (hit 2) t3_2_r
        float(modulesInGPU.layers[lowerModuleIndex2] + 6*is_endcap2), // inner (hit 2) t3_2_layer
        mdsInGPU.anchorEta[thirdMDIndex],       // inner (hit 3) t3_4_eta
        mdsInGPU.anchorPhi[thirdMDIndex],       // inner (hit 3) t3_4_phi
        mdsInGPU.anchorZ[thirdMDIndex],         // inner (hit 3) t3_4_z
        sqrtf(x3*x3 + y3*y3),                   // inner (hit 3) t3_4_r
        float(modulesInGPU.layers[lowerModuleIndex3] + 6*is_endcap3), // inner (hit 3) t3_4_layer
        log10(2.99792458e-3f*3.8f*outerRadius),        // outer t3_pt
        mdsInGPU.anchorEta[thirdMDIndex],       // outer (hit 4) t3_0_eta
        mdsInGPU.anchorPhi[thirdMDIndex],       // outer (hit 4) t3_0_phi
        mdsInGPU.anchorZ[thirdMDIndex],         // outer (hit 3) t3_0_z
        sqrtf(x3*x3 + y3*y3),                   // outer (hit 3) t3_0_r
        float(modulesInGPU.layers[lowerModuleIndex3] + 6*is_endcap3), // outer (hit 3) t3_0_layer
        mdsInGPU.anchorEta[fourthMDIndex],      // outer (hit 4) t3_2_eta
        mdsInGPU.anchorPhi[fourthMDIndex],      // outer (hit 4) t3_2_phi
        mdsInGPU.anchorZ[fourthMDIndex],        // outer (hit 4) t3_2_z
        sqrtf(x4*x4 + y4*y4),                   // outer (hit 4) t3_2_r
        float(modulesInGPU.layers[lowerModuleIndex4] + 6*is_endcap4), // outer (hit 4) t3_2_layer
        mdsInGPU.anchorEta[fifthMDIndex],       // outer (hit 5) t3_4_eta
        mdsInGPU.anchorPhi[fifthMDIndex],       // outer (hit 5) t3_4_phi
        mdsInGPU.anchorZ[fifthMDIndex],         // outer (hit 5) t3_4_z
        sqrtf(x5*x5 + y5*y5),                   // outer (hit 5) t3_4_r
        float(modulesInGPU.layers[lowerModuleIndex5] + 6*is_endcap5), // outer (hit 5) t3_4_layer
        log10((innerRadius+outerRadius)*3.8f*1.602f/(2*100*5.39f)), // t5_pt
        /* log10(2.99792458e-3f*3.8f*innerRadius), // t5_pt // BAD TEST */
        mdsInGPU.anchorEta[segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*innerTripletIndex+layer2_adjustment]]], // t5_eta
        mdsInGPU.anchorPhi[segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*innerTripletIndex+layer2_adjustment]]], // t5_phi
        log10(innerRadius),                            // t5_innerRadius
        log10(bridgeRadius),                           // t5_bridgeRadius
        log10(outerRadius)                             // t5_outerRadius"
    };

    // (0): Linear(in_features=38, out_features=32, bias=True) => x = x*W_T + b
    float bias_0[32] = {
        -0.453155100345611572265625000000,-3.414395570755004882812500000000, 0.461477220058441162109375000000,-0.154097318649291992187500000000,-0.470577239990234375000000000000, 2.228851079940795898437500000000,-0.163815096020698547363281250000,-2.296737670898437500000000000000,-0.098085217177867889404296875000, 2.589857339859008789062500000000, 2.105398178100585937500000000000,-0.525056004524230957031250000000, 1.790417790412902832031250000000,-2.162935256958007812500000000000, 0.012643844820559024810791015625,-1.595005750656127929687500000000, 0.222240477800369262695312500000,-3.097779273986816406250000000000,-0.748212277889251708984375000000, 1.027869462966918945312500000000,-0.504925668239593505859375000000, 0.079979300498962402343750000000,-0.680445253849029541015625000000, 1.331984400749206542968750000000, 2.027832508087158203125000000000,-0.063643321394920349121093750000,-0.304876327514648437500000000000,-0.020376743748784065246582031250, 2.847964286804199218750000000000,-1.884062528610229492187500000000, 2.168398141860961914062500000000,-2.673130989074707031250000000000
    };
    float wgtT_0[38][32] = {
        {  2.681395769119262695312500000000,-7.599917888641357421875000000000, 9.670552253723144531250000000000,-0.001992834033444523811340332031,-3.446841478347778320312500000000, 0.696401119232177734375000000000,-11.330449104309082031250000000000, 2.816005945205688476562500000000, 2.155102729797363281250000000000,-1.653487324714660644531250000000,-5.977705955505371093750000000000, 6.764187335968017578125000000000,-9.071163177490234375000000000000,-10.391711235046386718750000000000, 0.133476421236991882324218750000, 0.147252678871154785156250000000,-0.551621437072753906250000000000,-5.006772041320800781250000000000, 2.548298358917236328125000000000,-5.157997131347656250000000000000,-2.017001390457153320312500000000, 0.003392933635041117668151855469, 6.059939861297607421875000000000, 4.419455528259277343750000000000, 0.018351776525378227233886718750, 0.718529343605041503906250000000,17.632480621337890625000000000000, 2.147202491760253906250000000000,11.433926582336425781250000000000,-7.508767604827880859375000000000,10.208942413330078125000000000000, 1.028760910034179687500000000000 },
        { -0.524985134601593017578125000000,-0.823949456214904785156250000000,-0.705380678176879882812500000000,-0.060875885188579559326171875000, 0.593978524208068847656250000000, 0.034564595669507980346679687500,-1.687419176101684570312500000000,-5.644461154937744140625000000000,-0.242702737450599670410156250000,-1.846251010894775390625000000000,-1.071743965148925781250000000000, 5.430179595947265625000000000000, 0.983736336231231689453125000000, 1.929156541824340820312500000000,-0.067140892148017883300781250000, 1.181910157203674316406250000000,-1.198660731315612792968750000000,-0.403692513704299926757812500000,-0.931079268455505371093750000000,-3.782736063003540039062500000000,-1.083001017570495605468750000000,-0.016148230060935020446777343750, 1.717375874519348144531250000000,-1.823321461677551269531250000000, 2.168518543243408203125000000000,-0.235626965761184692382812500000, 0.064798735082149505615234375000,-0.291899144649505615234375000000, 1.270458936691284179687500000000, 2.221642732620239257812500000000,-0.648860991001129150390625000000,-2.041070222854614257812500000000 },
        {  2.428376674652099609375000000000,-0.055889230221509933471679687500, 0.605948746204376220703125000000, 0.088167086243629455566406250000, 0.253009736537933349609375000000, 0.079364776611328125000000000000,-0.037426453083753585815429687500, 0.555371403694152832031250000000, 1.478274822235107421875000000000, 0.405567944049835205078125000000, 0.292997509241104125976562500000,-0.583463370800018310546875000000, 0.101382099092006683349609375000, 0.360504746437072753906250000000,-0.104836054146289825439453125000,-0.562038660049438476562500000000,-0.380469232797622680664062500000,-0.176579430699348449707031250000, 0.693411409854888916015625000000,-0.877987742424011230468750000000,-6.362188339233398437500000000000, 0.162803128361701965332031250000,-0.503772139549255371093750000000,-0.376113206148147583007812500000, 0.049731504172086715698242187500,-2.318715810775756835937500000000, 0.005437037907540798187255859375, 0.910109341144561767578125000000, 0.499772101640701293945312500000,-0.367418885231018066406250000000,-0.428888618946075439453125000000, 0.623300611972808837890625000000 },
        { -0.136273548007011413574218750000,-0.035910826176404953002929687500,-0.023605758324265480041503906250, 0.041109450161457061767578125000, 0.075222909450531005859375000000, 0.045006752014160156250000000000,-0.011901576071977615356445312500,-0.978783249855041503906250000000,-0.081580743193626403808593750000,-0.772800743579864501953125000000, 0.327634543180465698242187500000, 1.381370544433593750000000000000, 0.034158825874328613281250000000, 0.085632212460041046142578125000,-0.106018148362636566162109375000, 0.111640326678752899169921875000,-0.362906932830810546875000000000,-0.007589066401124000549316406250,-0.248398855328559875488281250000,-1.133792281150817871093750000000, 0.063375085592269897460937500000,-0.007261080667376518249511718750, 0.393382728099822998046875000000,-0.342946767807006835937500000000,-0.056580573320388793945312500000, 0.009554558433592319488525390625,-0.000287339964415878057479858398, 0.083642348647117614746093750000, 0.164315268397331237792968750000, 0.336143314838409423828125000000,-0.075742892920970916748046875000,-0.179691836237907409667968750000 },
        {  0.342943012714385986328125000000, 0.248661294579505920410156250000, 0.315434962511062622070312500000,-0.007656908594071865081787109375, 0.038446065038442611694335937500,-0.100552536547183990478515625000,-0.000568154617212712764739990234,-0.946047484874725341796875000000, 0.553227722644805908203125000000, 0.813442349433898925781250000000,-0.585401237010955810546875000000, 0.344896465539932250976562500000,-0.068989150226116180419921875000,-0.175750449299812316894531250000,-0.083109997212886810302734375000, 1.022021651268005371093750000000,-0.607460975646972656250000000000,-0.483868956565856933593750000000, 0.744120121002197265625000000000, 1.644792437553405761718750000000,-0.360641986131668090820312500000, 0.021764716133475303649902343750,-0.951109111309051513671875000000,-0.659456253051757812500000000000,-0.551983475685119628906250000000, 0.153182744979858398437500000000,-0.005221265368163585662841796875,-0.199917539954185485839843750000,-0.062132254242897033691406250000, 1.002737998962402343750000000000,-0.169652491807937622070312500000,-0.502195477485656738281250000000 },
        { -0.529641389846801757812500000000, 0.559859812259674072265625000000,-1.498350381851196289062500000000,-0.067103281617164611816406250000,-0.280834972858428955078125000000,-0.662061095237731933593750000000,-0.748476624488830566406250000000,-0.184441581368446350097656250000, 0.457913756370544433593750000000,-1.843983411788940429687500000000,-0.393509626388549804687500000000, 1.080344915390014648437500000000, 0.338304847478866577148437500000,-0.256901085376739501953125000000, 0.116933539509773254394531250000,-0.353013753890991210937500000000,-0.014546658843755722045898437500, 0.203612253069877624511718750000, 0.203908145427703857421875000000,-1.177151918411254882812500000000, 0.046190921217203140258789062500, 0.112115144729614257812500000000,-1.460018634796142578125000000000,-3.063722133636474609375000000000,-2.418026208877563476562500000000, 0.314070373773574829101562500000, 0.056336998939514160156250000000,-1.262973904609680175781250000000, 0.301075458526611328125000000000,-0.375323414802551269531250000000,-3.724911928176879882812500000000, 0.788864374160766601562500000000 },
        { -0.231254860758781433105468750000,-0.602941215038299560546875000000,-0.064524009823799133300781250000,-0.127337485551834106445312500000, 0.432118058204650878906250000000,-0.186580061912536621093750000000,-0.707097470760345458984375000000,-0.790629386901855468750000000000, 0.068828307092189788818359375000, 0.871053338050842285156250000000,-1.535928010940551757812500000000, 0.611867964267730712890625000000, 0.591017067432403564453125000000, 0.601626574993133544921875000000, 0.124191008508205413818359375000, 0.875569522380828857421875000000,-0.191971987485885620117187500000, 0.029285309836268424987792968750,-1.051985740661621093750000000000, 0.267334282398223876953125000000, 0.529267847537994384765625000000,-0.035881459712982177734375000000,-0.234495744109153747558593750000, 0.163393601775169372558593750000, 1.725398778915405273437500000000, 0.047789610922336578369140625000, 0.062527090311050415039062500000, 0.216674238443374633789062500000, 0.746908962726593017578125000000, 1.206458568572998046875000000000, 1.134528040885925292968750000000,-0.328735858201980590820312500000 },
        {  2.104823827743530273437500000000,-0.165199279785156250000000000000,-0.224229991436004638671875000000,-0.176828786730766296386718750000, 0.202808871865272521972656250000,-0.297297805547714233398437500000,-1.939160704612731933593750000000, 1.119998693466186523437500000000, 2.449795007705688476562500000000,-0.346344232559204101562500000000,-0.116521216928958892822265625000,-0.854153394699096679687500000000, 0.202458426356315612792968750000, 0.010512894950807094573974609375,-0.056162714958190917968750000000, 1.082383751869201660156250000000, 1.028264045715332031250000000000,-0.151054635643959045410156250000, 0.524155497550964355468750000000,-0.210107475519180297851562500000,-1.815126061439514160156250000000, 0.126511782407760620117187500000,-0.585544347763061523437500000000, 0.869816422462463378906250000000,-0.354224711656570434570312500000,-2.197012662887573242187500000000,-0.028015539050102233886718750000, 0.775934636592864990234375000000, 0.167296618223190307617187500000, 0.726468741893768310546875000000, 0.119873203337192535400390625000,-0.859227716922760009765625000000 },
        {  0.148623168468475341796875000000, 0.054969754070043563842773437500,-0.016196282580494880676269531250, 0.058222603052854537963867187500,-0.060340385884046554565429687500, 0.081522755324840545654296875000, 0.010995480231940746307373046875, 0.023508522659540176391601562500, 0.113623961806297302246093750000, 0.172199845314025878906250000000,-0.334862768650054931640625000000,-0.484579294919967651367187500000, 0.005375042557716369628906250000, 0.003075006883591413497924804688,-0.015041348524391651153564453125, 0.165629908442497253417968750000, 0.460200518369674682617187500000,-0.095209315419197082519531250000,-0.265740305185317993164062500000,-0.019723346456885337829589843750, 0.401116460561752319335937500000, 0.155702561140060424804687500000,-0.213185608386993408203125000000, 0.361281007528305053710937500000,-0.217600151896476745605468750000, 0.296559095382690429687500000000,-0.007693792227655649185180664062, 0.207337334752082824707031250000,-0.035066802054643630981445312500,-0.317755430936813354492187500000,-0.106163069605827331542968750000, 0.765429973602294921875000000000 },
        {  0.095673680305480957031250000000, 0.024865347892045974731445312500, 0.673318803310394287109375000000,-0.015504667535424232482910156250,-0.141617029905319213867187500000,-0.019630333408713340759277343750, 0.048571899533271789550781250000,-0.073992997407913208007812500000, 0.040667321532964706420898437500,-0.177151247859001159667968750000,-0.220022484660148620605468750000,-0.153635993599891662597656250000,-0.100910000503063201904296875000,-0.040304876863956451416015625000,-0.086127102375030517578125000000,-0.227517798542976379394531250000, 0.285083621740341186523437500000,-0.429354280233383178710937500000, 0.589987933635711669921875000000,-0.144052118062973022460937500000,-0.169282928109169006347656250000,-0.082840152084827423095703125000, 0.523216128349304199218750000000,-0.203670844435691833496093750000,-0.062806390225887298583984375000,-0.102601811289787292480468750000, 0.009201795794069766998291015625, 0.450033366680145263671875000000,-0.126564577221870422363281250000, 0.223554939031600952148437500000, 0.334533929824829101562500000000,-0.600908279418945312500000000000 },
        {  0.062765456736087799072265625000, 0.488313466310501098632812500000, 0.877064406871795654296875000000, 0.113091208040714263916015625000,-0.108147174119949340820312500000, 0.063301309943199157714843750000,-0.641807913780212402343750000000, 0.101771391928195953369140625000,-0.673703432083129882812500000000, 0.271248042583465576171875000000,-0.162217766046524047851562500000,-0.022933615371584892272949218750, 0.031916566193103790283203125000,-0.100393228232860565185546875000,-0.037933815270662307739257812500,-0.449602723121643066406250000000,-1.557855725288391113281250000000,-0.615495204925537109375000000000,-0.936458170413970947265625000000, 0.722485363483428955078125000000,-0.939954519271850585937500000000,-0.152611345052719116210937500000,-0.157423526048660278320312500000, 0.745969712734222412109375000000,-1.416319727897644042968750000000, 1.204031109809875488281250000000, 0.043809067457914352416992187500,-2.247609615325927734375000000000,-0.510436832904815673828125000000,-1.141731381416320800781250000000, 0.027229400351643562316894531250,-3.103597879409790039062500000000 },
        { -0.064130336046218872070312500000,-0.593604266643524169921875000000, 0.329273998737335205078125000000,-0.078833319246768951416015625000, 0.269491195678710937500000000000,-0.377352952957153320312500000000,-0.679528355598449707031250000000, 0.875195324420928955078125000000, 0.196003511548042297363281250000, 1.376969933509826660156250000000,-1.383753657341003417968750000000,-0.468429654836654663085937500000, 0.083914116024971008300781250000, 0.321190953254699707031250000000,-0.145415350794792175292968750000, 0.979343593120574951171875000000, 0.096557818353176116943359375000, 0.393212616443634033203125000000,-0.687955260276794433593750000000, 1.308652281761169433593750000000, 0.670221209526062011718750000000, 0.094020560383796691894531250000,-0.841321945190429687500000000000,-0.047969307750463485717773437500, 1.734134554862976074218750000000, 0.184733375906944274902343750000,-0.156755447387695312500000000000, 0.044511202722787857055664062500, 0.649618327617645263671875000000, 1.182070016860961914062500000000, 1.305595278739929199218750000000,-0.508566677570343017578125000000 },
        {  0.701800823211669921875000000000, 0.045078206807374954223632812500,-0.159631043672561645507812500000,-0.169421955943107604980468750000, 0.715994358062744140625000000000,-0.013050395064055919647216796875, 0.631362259387969970703125000000,-0.225342184305191040039062500000, 1.529913425445556640625000000000,-0.002558978740125894546508789062,-0.024538433179259300231933593750,-0.129255548119544982910156250000,-0.011085875332355499267578125000,-0.059477403759956359863281250000, 0.075156927108764648437500000000,-0.140214055776596069335937500000,-0.505609452724456787109375000000,-0.085507236421108245849609375000,-0.206223100423812866210937500000, 0.099693328142166137695312500000, 1.237704396247863769531250000000,-0.070938147604465484619140625000,-0.134285271167755126953125000000,-0.453247398138046264648437500000, 0.003889448940753936767578125000,-3.303350448608398437500000000000,-0.085179120302200317382812500000,-0.125175178050994873046875000000, 0.047051545232534408569335937500,-0.274922072887420654296875000000, 0.364637136459350585937500000000,-0.350260406732559204101562500000 },
        {  0.331875056028366088867187500000, 0.069977365434169769287109375000,-0.065289631485939025878906250000,-0.108880102634429931640625000000,-0.104320190846920013427734375000, 0.008694163523614406585693359375, 0.111780919134616851806640625000, 0.203506380319595336914062500000, 0.114401064813137054443359375000, 0.137982100248336791992187500000,-0.177617728710174560546875000000,-0.294225335121154785156250000000, 0.013799699954688549041748046875,-0.012937829829752445220947265625, 0.076549381017684936523437500000, 0.007528936956077814102172851562, 0.161265879869461059570312500000,-0.105450265109539031982421875000,-0.127381697297096252441406250000, 0.192449867725372314453125000000, 0.498172491788864135742187500000,-0.132289081811904907226562500000,-0.492380797863006591796875000000, 0.377453178167343139648437500000, 0.231900691986083984375000000000, 0.023043131455779075622558593750, 0.075486473739147186279296875000,-0.063966199755668640136718750000,-0.013045103289186954498291015625,-0.039310052990913391113281250000, 0.417924642562866210937500000000, 0.502855300903320312500000000000 },
        {  0.229935362935066223144531250000,-0.212384849786758422851562500000, 0.308913916349411010742187500000,-0.128575935959815979003906250000,-0.344552665948867797851562500000,-0.040466558188199996948242187500,-0.194234400987625122070312500000, 0.311862885951995849609375000000,-0.260208696126937866210937500000,-0.207310289144515991210937500000, 0.125031828880310058593750000000, 0.023847851902246475219726562500,-0.060182727873325347900390625000, 0.112200327217578887939453125000,-0.117272786796092987060546875000,-0.506568789482116699218750000000, 0.603083670139312744140625000000,-0.100993752479553222656250000000, 0.571203291416168212890625000000,-0.363246500492095947265625000000, 0.390363097190856933593750000000,-0.162089779973030090332031250000, 0.541227519512176513671875000000,-0.165127560496330261230468750000, 0.184602186083793640136718750000,-0.491730093955993652343750000000,-0.054235659539699554443359375000, 0.381522327661514282226562500000, 0.173893675208091735839843750000,-0.300541281700134277343750000000, 0.051931522786617279052734375000, 0.014499680139124393463134765625 },
        {  0.728410661220550537109375000000,-0.084344185888767242431640625000,-0.233651280403137207031250000000,-0.052232533693313598632812500000,-0.265878587961196899414062500000, 0.061499141156673431396484375000,-0.951330423355102539062500000000,-0.697260916233062744140625000000, 0.285715043544769287109375000000,-0.410996496677398681640625000000,-0.854818701744079589843750000000,-1.403917551040649414062500000000,-0.034403480589389801025390625000,-0.209432289004325866699218750000,-0.008406938984990119934082031250, 0.304991364479064941406250000000,-0.708435833454132080078125000000, 0.196053922176361083984375000000,-0.289813607931137084960937500000,-0.906013667583465576171875000000, 0.238152965903282165527343750000, 0.134902209043502807617187500000,-0.234306767582893371582031250000, 0.322634667158126831054687500000,-0.935358047485351562500000000000, 0.157942399382591247558593750000, 0.164006993174552917480468750000,-0.093607746064662933349609375000, 0.006198659539222717285156250000,-2.063370227813720703125000000000,-0.463953882455825805664062500000,-0.122594557702541351318359375000 },
        {  2.560253858566284179687500000000, 3.583789825439453125000000000000, 2.662947654724121093750000000000,-0.155168861150741577148437500000, 1.691972374916076660156250000000,-2.231052398681640625000000000000,10.942481040954589843750000000000,-1.573819398880004882812500000000, 3.035347700119018554687500000000,-1.597251772880554199218750000000, 4.939544200897216796875000000000,-5.514997482299804687500000000000, 7.407028675079345703125000000000, 7.058206081390380859375000000000,-0.043456312268972396850585937500,-1.969845294952392578125000000000, 3.462084770202636718750000000000, 0.159087538719177246093750000000, 4.218122482299804687500000000000,-3.373720645904541015625000000000, 0.886464416980743408203125000000, 0.141215518116950988769531250000,-2.698859453201293945312500000000,-8.839869499206542968750000000000, 2.324672698974609375000000000000, 0.747064352035522460937500000000,-21.831840515136718750000000000000, 1.367454648017883300781250000000,-10.383153915405273437500000000000, 7.895112514495849609375000000000,-5.794469356536865234375000000000,-1.259284019470214843750000000000 },
        {  0.097051613032817840576171875000,-0.545346200466156005859375000000, 0.398123145103454589843750000000, 0.044652719050645828247070312500, 0.369166195392608642578125000000,-0.279953539371490478515625000000,-0.480253338813781738281250000000, 0.788928389549255371093750000000, 0.203059732913970947265625000000, 1.501827716827392578125000000000,-1.612030029296875000000000000000,-0.159930497407913208007812500000,-0.010579479858279228210449218750, 0.220564797520637512207031250000, 0.086818270385265350341796875000, 0.859122335910797119140625000000, 0.222208678722381591796875000000, 0.390240728855133056640625000000,-0.945640206336975097656250000000, 1.435796856880187988281250000000, 0.485675096511840820312500000000,-0.119415983557701110839843750000,-0.839518129825592041015625000000, 0.173121839761734008789062500000, 1.714885592460632324218750000000, 0.164848893880844116210937500000, 0.011612919159233570098876953125,-0.084974460303783416748046875000, 0.653332471847534179687500000000, 1.323395609855651855468750000000, 1.499850988388061523437500000000,-0.559670865535736083984375000000 },
        {  0.571823358535766601562500000000,-0.019433513283729553222656250000,-0.244986623525619506835937500000, 0.103959344327449798583984375000, 0.799324810504913330078125000000,-0.010179465636610984802246093750, 0.761551499366760253906250000000,-0.131382510066032409667968750000, 1.549323201179504394531250000000,-0.034508824348449707031250000000,-0.046313866972923278808593750000,-0.318450868129730224609375000000,-0.101109817624092102050781250000, 0.011624247767031192779541015625, 0.114663705229759216308593750000,-0.201420575380325317382812500000,-0.482427805662155151367187500000,-0.091737858951091766357421875000,-0.394173920154571533203125000000, 0.112227246165275573730468750000, 0.970535397529602050781250000000,-0.062891244888305664062500000000,-0.123602762818336486816406250000,-0.525807023048400878906250000000, 0.016155727207660675048828125000,-3.193232059478759765625000000000, 0.095808073878288269042968750000,-0.198437631130218505859375000000,-0.067583188414573669433593750000,-0.071442931890487670898437500000, 0.464652776718139648437500000000,-0.088569760322570800781250000000 },
        {  0.175562933087348937988281250000,-0.124195985496044158935546875000,-0.110135577619075775146484375000,-0.015305858105421066284179687500,-0.139956086874008178710937500000,-0.037097431719303131103515625000, 0.226938709616661071777343750000, 0.166226118803024291992187500000,-0.048766333609819412231445312500, 0.127816870808601379394531250000,-0.124405309557914733886718750000,-0.298750400543212890625000000000,-0.065063923597335815429687500000, 0.012892391532659530639648437500, 0.022033169865608215332031250000, 0.028511764481663703918457031250, 0.040466636419296264648437500000, 0.031412895768880844116210937500,-0.133911579847335815429687500000, 0.287476420402526855468750000000, 0.321584820747375488281250000000, 0.047255504876375198364257812500,-0.321841716766357421875000000000, 0.237388014793395996093750000000, 0.107724912464618682861328125000, 0.130338013172149658203125000000,-0.063520029187202453613281250000, 0.112270653247833251953125000000,-0.101979412138462066650390625000,-0.119287893176078796386718750000, 0.413636922836303710937500000000, 0.506543219089508056640625000000 },
        {  0.232762292027473449707031250000, 0.049606259912252426147460937500, 0.369498461484909057617187500000,-0.179308161139488220214843750000,-0.215157315135002136230468750000,-0.110851064324378967285156250000,-0.218158036470413208007812500000, 0.573607802391052246093750000000,-0.431137174367904663085937500000,-0.015987955033779144287109375000, 0.012779578566551208496093750000,-0.171807557344436645507812500000, 0.088214114308357238769531250000, 0.097710601985454559326171875000,-0.133426219224929809570312500000,-0.434398472309112548828125000000, 0.438165634870529174804687500000,-0.264113634824752807617187500000, 0.420685440301895141601562500000,-0.351321995258331298828125000000, 0.190314054489135742187500000000, 0.001282726065255701541900634766, 0.369004309177398681640625000000, 0.088860444724559783935546875000, 0.388227552175521850585937500000,-0.486671686172485351562500000000, 0.048729468137025833129882812500, 0.291621625423431396484375000000, 0.083145886659622192382812500000,-0.402861863374710083007812500000, 0.149641528725624084472656250000, 0.231344491243362426757812500000 },
        {  0.722625911235809326171875000000,-0.019313314929604530334472656250,-0.211568400263786315917968750000, 0.036486808210611343383789062500,-0.398582458496093750000000000000, 0.116802819073200225830078125000,-1.154521107673645019531250000000,-0.590227782726287841796875000000, 0.252344459295272827148437500000,-0.346864342689514160156250000000,-0.678294479846954345703125000000,-1.593878388404846191406250000000,-0.061062168329954147338867187500,-0.055966157466173171997070312500,-0.128128424286842346191406250000, 0.308055192232131958007812500000,-0.547070980072021484375000000000, 0.371600151062011718750000000000,-0.550005912780761718750000000000,-0.625245094299316406250000000000, 0.321538388729095458984375000000,-0.028636038303375244140625000000,-0.503177762031555175781250000000, 0.327037394046783447265625000000,-0.989046692848205566406250000000,-0.126254826784133911132812500000, 0.078017465770244598388671875000,-0.089542612433433532714843750000,-0.042310956865549087524414062500,-1.899654507637023925781250000000,-0.231878265738487243652343750000,-0.091606028378009796142578125000 },
        {  0.083934642374515533447265625000,-0.621707856655120849609375000000, 0.136384546756744384765625000000,-0.055096365511417388916015625000, 0.518171787261962890625000000000,-0.440564543008804321289062500000,-0.961045742034912109375000000000, 0.583967149257659912109375000000, 0.158686757087707519531250000000, 1.232593059539794921875000000000,-0.423261135816574096679687500000, 0.026423551142215728759765625000, 0.298696637153625488281250000000, 0.127668112516403198242187500000, 0.146882027387619018554687500000, 1.084036111831665039062500000000,-1.049437046051025390625000000000, 0.385521322488784790039062500000, 0.158378109335899353027343750000, 0.797107577323913574218750000000,-0.251428663730621337890625000000,-0.019937217235565185546875000000, 0.488125622272491455078125000000,-0.633256435394287109375000000000, 1.248124122619628906250000000000, 0.348201543092727661132812500000, 0.108102664351463317871093750000,-0.501707732677459716796875000000, 0.508260309696197509765625000000, 1.660969972610473632812500000000, 0.593935310840606689453125000000,-1.489265680313110351562500000000 },
        {  0.523537456989288330078125000000,-0.012671006843447685241699218750,-0.756707191467285156250000000000, 0.022541524842381477355957031250, 1.223816633224487304687500000000, 0.098516076803207397460937500000, 0.135689541697502136230468750000,-0.603900551795959472656250000000, 2.732395648956298828125000000000,-0.180136978626251220703125000000,-0.058773122727870941162109375000, 0.377364337444305419921875000000,-0.158604174852371215820312500000,-0.277447104454040527343750000000,-0.035171609371900558471679687500, 0.485159814357757568359375000000, 1.509233832359313964843750000000, 0.334059983491897583007812500000, 0.432152509689331054687500000000, 1.354267120361328125000000000000, 0.439237505197525024414062500000,-0.031074659898877143859863281250, 0.789955317974090576171875000000, 0.822415113449096679687500000000,-0.003722778754308819770812988281,-2.582183361053466796875000000000, 0.005577699281275272369384765625,-0.815501272678375244140625000000,-0.536845386028289794921875000000,-0.501874744892120361328125000000, 0.626843690872192382812500000000, 1.254838466644287109375000000000 },
        {  0.019574068486690521240234375000, 0.093382671475410461425781250000,-0.040202967822551727294921875000,-0.068567335605621337890625000000, 0.196897700428962707519531250000,-0.145303711295127868652343750000, 0.167457342147827148437500000000,-0.009816204197704792022705078125, 0.214539021253585815429687500000,-0.090170361101627349853515625000,-0.091977603733539581298828125000, 0.014517173171043395996093750000, 0.029669621959328651428222656250,-0.154459491372108459472656250000,-0.013502078130841255187988281250,-0.188866108655929565429687500000,-0.568850576877593994140625000000, 0.061402127146720886230468750000, 0.350244581699371337890625000000, 0.321362882852554321289062500000,-0.751947581768035888671875000000,-0.111295595765113830566406250000, 0.489118933677673339843750000000, 0.181520447134971618652343750000,-0.076170854270458221435546875000, 0.043594952672719955444335937500,-0.007644311990588903427124023438,-0.283990830183029174804687500000,-0.047331992536783218383789062500,-0.171153917908668518066406250000,-0.297220647335052490234375000000,-0.374402225017547607421875000000 },
        { -0.380128026008605957031250000000, 0.131258621811866760253906250000, 0.008614375256001949310302734375,-0.141585439443588256835937500000, 0.082255989313125610351562500000, 0.148281589150428771972656250000,-0.339317142963409423828125000000, 0.075970716774463653564453125000,-0.274273216724395751953125000000,-0.180833682417869567871093750000, 0.185964629054069519042968750000,-0.039124157279729843139648437500,-0.006181631702929735183715820312,-0.015228725969791412353515625000,-0.072899974882602691650390625000,-0.187394589185714721679687500000,-0.737279772758483886718750000000, 0.173461675643920898437500000000,-0.640574693679809570312500000000, 0.059875339269638061523437500000,-0.013128707185387611389160156250,-0.014354419894516468048095703125,-0.608681678771972656250000000000,-0.111488118767738342285156250000, 0.317772477865219116210937500000,-0.878546476364135742187500000000,-0.016493067145347595214843750000, 0.214950531721115112304687500000, 0.075516499578952789306640625000,-0.173986315727233886718750000000, 0.015861079096794128417968750000, 0.271164834499359130859375000000 },
        { -0.384006410837173461914062500000, 0.009162863716483116149902343750,-0.615911960601806640625000000000,-0.151109784841537475585937500000, 0.121625602245330810546875000000, 0.075461000204086303710937500000,-0.728917121887207031250000000000,-0.991489112377166748046875000000, 0.211264461278915405273437500000,-1.330252885818481445312500000000,-0.624064743518829345703125000000,-0.874477267265319824218750000000, 0.322812855243682861328125000000, 0.049898970872163772583007812500,-0.090785466134548187255859375000, 0.124542348086833953857421875000,-0.822253704071044921875000000000, 0.451101422309875488281250000000, 0.533658623695373535156250000000, 0.319087356328964233398437500000, 0.830891072750091552734375000000, 0.000674917770083993673324584961,-0.038458690047264099121093750000,-0.933933496475219726562500000000,-0.254446595907211303710937500000, 0.034912660717964172363281250000, 0.037369485944509506225585937500,-0.251882493495941162109375000000, 0.177388533949851989746093750000,-0.079312816262245178222656250000, 0.340542942285537719726562500000,-0.933198750019073486328125000000 },
        { -0.584230542182922363281250000000,-0.838078379631042480468750000000,-0.061515036970376968383789062500, 0.027180569246411323547363281250, 0.665422737598419189453125000000,-0.270218253135681152343750000000,-1.504381656646728515625000000000, 0.613037586212158203125000000000, 0.693742573261260986328125000000, 0.870394647121429443359375000000,-0.765844702720642089843750000000, 0.487751930952072143554687500000, 0.015539275482296943664550781250, 0.210426837205886840820312500000, 0.080762520432472229003906250000, 1.776023626327514648437500000000,-0.637655138969421386718750000000, 0.474847495555877685546875000000,-0.061571974307298660278320312500, 0.373049885034561157226562500000, 0.125899180769920349121093750000, 0.155141845345497131347656250000,-0.027319114655256271362304687500,-2.177373886108398437500000000000, 1.556130409240722656250000000000, 0.424499273300170898437500000000, 0.034514218568801879882812500000,-0.966718077659606933593750000000, 0.548672914505004882812500000000, 1.634848594665527343750000000000, 0.402413904666900634765625000000,-1.695147156715393066406250000000 },
        {  3.601962089538574218750000000000, 0.031723286956548690795898437500,-0.486349642276763916015625000000,-0.067972607910633087158203125000, 1.456925392150878906250000000000,-0.002324576256796717643737792969,-0.067015670239925384521484375000, 0.018733527511358261108398437500,-0.684249401092529296875000000000, 0.095511846244335174560546875000, 0.067323334515094757080078125000, 0.499029219150543212890625000000, 0.055325899273157119750976562500,-0.082365520298480987548828125000,-0.004257752094417810440063476562,-0.163551405072212219238281250000, 0.123053044080734252929687500000, 0.444403171539306640625000000000,-0.632802546024322509765625000000,-0.180914953351020812988281250000, 4.487323760986328125000000000000,-0.022270089015364646911621093750, 0.384218186140060424804687500000,-0.223117962479591369628906250000, 0.097903206944465637207031250000,-1.971325755119323730468750000000, 0.000986684113740921020507812500,-0.716064929962158203125000000000,-0.080332756042480468750000000000, 0.500032782554626464843750000000, 0.492225468158721923828125000000, 0.011614332906901836395263671875 },
        { -0.399567335844039916992187500000, 0.007188517134636640548706054688, 0.202259466052055358886718750000, 0.125657707452774047851562500000,-0.035918358713388442993164062500, 0.095855966210365295410156250000,-0.236331045627593994140625000000, 0.002901439554989337921142578125, 0.235848113894462585449218750000, 0.085358753800392150878906250000, 0.311344742774963378906250000000, 0.098473608493804931640625000000,-0.023778814822435379028320312500, 0.056264724582433700561523437500, 0.027794919908046722412109375000,-0.194835931062698364257812500000,-0.049929667264223098754882812500, 0.049723494797945022583007812500, 0.274781197309494018554687500000,-0.050580717623233795166015625000,-0.213106602430343627929687500000, 0.072870478034019470214843750000, 0.179663911461830139160156250000,-0.570904433727264404296875000000,-0.081411346793174743652343750000, 0.295396924018859863281250000000, 0.002247131429612636566162109375,-0.295445591211318969726562500000, 0.050929769873619079589843750000,-0.023571878671646118164062500000,-0.340447783470153808593750000000,-0.528374671936035156250000000000 },
        { -0.178125873208045959472656250000,-0.020718796178698539733886718750,-0.629121184349060058593750000000,-0.038743197917938232421875000000,-0.071373492479324340820312500000, 0.079126402735710144042968750000, 0.550931751728057861328125000000,-0.011426444165408611297607421875,-1.160261869430541992187500000000, 0.263882309198379516601562500000, 0.147467985749244689941406250000, 0.312378138303756713867187500000, 0.106889814138412475585937500000,-0.006497310008853673934936523438,-0.132688254117965698242187500000, 0.496566027402877807617187500000,-0.469917595386505126953125000000, 0.596754729747772216796875000000,-0.914137661457061767578125000000,-0.112893886864185333251953125000,-0.285028100013732910156250000000, 0.094056911766529083251953125000,-0.085830092430114746093750000000, 0.600173413753509521484375000000,-0.227956891059875488281250000000,-1.011287808418273925781250000000, 0.011980824172496795654296875000,-1.135842442512512207031250000000,-0.020158275961875915527343750000,-0.009285877458751201629638671875,-0.344594150781631469726562500000, 0.324739724397659301757812500000 },
        {  0.204394191503524780273437500000,-0.062450423836708068847656250000,-2.351300716400146484375000000000, 0.001355741638690233230590820312, 0.164912372827529907226562500000, 0.348535567522048950195312500000, 2.319313049316406250000000000000,-0.846234321594238281250000000000, 0.177357122302055358886718750000,-0.439615547657012939453125000000, 0.665181994438171386718750000000,-0.491338759660720825195312500000, 0.473608851432800292968750000000, 0.285443365573883056640625000000, 0.132020413875579833984375000000, 0.014700873754918575286865234375,-0.142403706908226013183593750000, 0.164458155632019042968750000000, 0.784836292266845703125000000000, 0.395098149776458740234375000000,-1.056737661361694335937500000000,-0.172318831086158752441406250000,-0.058115579187870025634765625000,-1.187423348426818847656250000000,-0.096243351697921752929687500000, 0.043761100620031356811523437500,-0.337699323892593383789062500000,-0.062225610017776489257812500000, 0.253457486629486083984375000000, 2.511919736862182617187500000000,-0.586129486560821533203125000000,-0.727006912231445312500000000000 },
        { -2.551548480987548828125000000000, 4.583169937133789062500000000000,-2.190436124801635742187500000000, 0.105199918150901794433593750000, 3.900011301040649414062500000000, 1.656419754028320312500000000000, 6.810500621795654296875000000000, 2.384639978408813476562500000000,-2.359154224395751953125000000000, 3.195259332656860351562500000000,-2.918020009994506835937500000000, 0.788009166717529296875000000000, 3.630579471588134765625000000000,-2.449368238449096679687500000000, 0.105536855757236480712890625000, 2.125288963317871093750000000000, 0.790766477584838867187500000000, 3.743981838226318359375000000000, 0.031394183635711669921875000000,-4.668146610260009765625000000000,-0.202959880232810974121093750000,-0.020673528313636779785156250000,-2.899333715438842773437500000000,-3.727350473403930664062500000000,-1.505283355712890625000000000000,-1.999773502349853515625000000000, 3.567448616027832031250000000000,-1.878177285194396972656250000000,-2.719932556152343750000000000000, 6.341441154479980468750000000000,-1.824370384216308593750000000000,-0.857744574546813964843750000000 },
        { -0.543507456779479980468750000000,-0.727597177028656005859375000000,-0.141112387180328369140625000000, 0.130820900201797485351562500000, 0.365750998258590698242187500000,-0.103584215044975280761718750000,-0.964259743690490722656250000000,-2.175064802169799804687500000000, 0.048502337187528610229492187500,-0.237006023526191711425781250000,-1.311480164527893066406250000000, 2.687830924987792968750000000000, 0.595047414302825927734375000000, 1.067662954330444335937500000000, 0.161213994026184082031250000000, 1.114247798919677734375000000000,-0.134806454181671142578125000000,-0.172012910246849060058593750000,-0.998761773109436035156250000000,-1.506299376487731933593750000000, 0.362417459487915039062500000000, 0.165459349751472473144531250000, 0.170954599976539611816406250000,-0.047278236597776412963867187500, 2.101269960403442382812500000000, 0.028231432661414146423339843750,-0.069698639214038848876953125000, 0.079026393592357635498046875000, 0.713514506816864013671875000000, 1.462539792060852050781250000000, 1.018770217895507812500000000000,-0.433387547731399536132812500000 },
        { -0.634411275386810302734375000000, 0.179165408015251159667968750000, 0.583461821079254150390625000000, 0.042082581669092178344726562500, 0.529222905635833740234375000000, 0.141866043210029602050781250000, 0.344348490238189697265625000000,-0.333926528692245483398437500000, 2.304046630859375000000000000000,-0.172842606902122497558593750000,-0.028251593932509422302246093750,-0.121457777917385101318359375000,-0.070724897086620330810546875000,-0.028690652921795845031738281250, 0.125783324241638183593750000000,-0.252991676330566406250000000000,-1.089144945144653320312500000000, 0.033496279269456863403320312500,-0.199801936745643615722656250000,-0.302523612976074218750000000000,-2.473547697067260742187500000000,-0.045142836868762969970703125000, 0.219988122582435607910156250000,-0.649464905261993408203125000000, 0.112803213298320770263671875000,-2.327204942703247070312500000000,-0.000668313819915056228637695312,-0.499866217374801635742187500000, 0.039233725517988204956054687500,-0.245626091957092285156250000000,-0.645513057708740234375000000000,-0.551347792148590087890625000000 },
        {  0.183810576796531677246093750000,-4.256506919860839843750000000000, 1.518771290779113769531250000000, 0.118741497397422790527343750000,-0.995078325271606445312500000000, 2.323845148086547851562500000000,-1.525617361068725585937500000000,-1.817108154296875000000000000000, 0.392612308263778686523437500000, 2.471554040908813476562500000000, 1.105098724365234375000000000000, 0.181045055389404296875000000000, 0.717661917209625244140625000000,-3.149072408676147460937500000000, 0.069827295839786529541015625000,-1.199739694595336914062500000000, 0.056081101298332214355468750000,-3.169653654098510742187500000000,-0.354798167943954467773437500000, 0.306009322404861450195312500000,-0.693456053733825683593750000000, 0.031488411128520965576171875000,-0.004706704057753086090087890625, 1.561665415763854980468750000000, 1.783095359802246093750000000000, 0.120748490095138549804687500000, 1.365911960601806640625000000000, 0.074387550354003906250000000000, 3.733332157135009765625000000000,-2.607214927673339843750000000000, 3.232627630233764648437500000000,-2.238861799240112304687500000000 },
        { -0.568708717823028564453125000000, 6.196700572967529296875000000000, 0.492914319038391113281250000000, 0.110564619302749633789062500000,-0.560010731220245361328125000000,-7.532878398895263671875000000000,-5.682581424713134765625000000000,-0.351758062839508056640625000000,-0.576390326023101806640625000000,-4.027351856231689453125000000000,-5.800712108612060546875000000000, 0.759718000888824462890625000000,-5.982936859130859375000000000000, 7.206288337707519531250000000000, 0.019034398719668388366699218750, 3.779457569122314453125000000000, 2.836686134338378906250000000000, 3.210830926895141601562500000000, 1.633451104164123535156250000000,-0.111079394817352294921875000000,-1.920373439788818359375000000000,-0.031798820942640304565429687500, 3.279193401336669921875000000000, 0.370096236467361450195312500000,-5.029094696044921875000000000000,-1.183466792106628417968750000000, 1.396517753601074218750000000000,-3.317970275878906250000000000000,-5.599634170532226562500000000000,-0.047379031777381896972656250000, 2.314316272735595703125000000000, 0.194091260433197021484375000000 },
        {  0.069253779947757720947265625000,-2.538430929183959960937500000000, 0.862707734107971191406250000000, 0.141743868589401245117187500000,-0.500090837478637695312500000000, 1.926409602165222167968750000000, 1.796001553535461425781250000000,-2.455615043640136718750000000000, 0.566797912120819091796875000000, 2.280332088470458984375000000000, 2.667954683303833007812500000000,-0.984177231788635253906250000000, 2.172750473022460937500000000000,-1.174688100814819335937500000000,-0.136859640479087829589843750000,-1.677646160125732421875000000000, 0.948042631149291992187500000000,-2.522024869918823242187500000000,-0.015574772842228412628173828125, 0.494085222482681274414062500000,-0.344312340021133422851562500000,-0.032696429640054702758789062500,-1.112177491188049316406250000000, 0.348784506320953369140625000000, 2.111689567565917968750000000000, 0.127242609858512878417968750000,-2.242399692535400390625000000000, 0.021647864952683448791503906250, 1.506944894790649414062500000000,-0.357796341180801391601562500000, 1.239420771598815917968750000000,-2.562223196029663085937500000000 },
    };
    float x_0[32];
    for (unsigned int col = 0; col < 32; ++col)
    {
        x_0[col] = 0;
        for (unsigned int inner = 0; inner < 38; ++inner)
        {
            x_0[col] += x[inner]*wgtT_0[inner][col];
        }
        x_0[col] += bias_0[col];
    }
    
    // (1): ReLU()
    float x_1[32];
    for (unsigned int col = 0; col < 32; ++col)
    {
        x_1[col] = (x_0[col] > 0.f) ? x_0[col] : 0.f;
    }
    
    // (2): Linear(in_features=32, out_features=32, bias=True) => x = x*W_T + b
    float bias_2[32] = {
         2.888562679290771484375000000000, 0.267166793346405029296875000000, 1.382908463478088378906250000000,-0.062899008393287658691406250000, 1.849197268486022949218750000000,-0.824259281158447265625000000000,-1.622051119804382324218750000000,-1.102929353713989257812500000000, 2.172684192657470703125000000000,-0.277113229036331176757812500000,-1.227879524230957031250000000000,-0.138141676783561706542968750000, 0.486872792243957519531250000000, 0.084002412855625152587890625000,-1.378330349922180175781250000000, 1.196636080741882324218750000000,-2.570324659347534179687500000000,-0.004051351919770240783691406250,-0.164678454399108886718750000000,-0.852835416793823242187500000000, 0.578797221183776855468750000000, 0.889193117618560791015625000000, 0.994170486927032470703125000000,-1.263559579849243164062500000000, 1.201083660125732421875000000000, 0.000894412340130656957626342773, 0.923921644687652587890625000000,-2.387999296188354492187500000000,-0.232334241271018981933593750000,-0.190567180514335632324218750000, 0.160316720604896545410156250000,-0.067880019545555114746093750000
    };
    float wgtT_2[32][32] = {
        { -0.071120575070381164550781250000, 0.062359873205423355102539062500, 0.132348269224166870117187500000,-0.084511600434780120849609375000, 0.029920289292931556701660156250, 0.016484286636114120483398437500, 0.048202134668827056884765625000,-3.244096517562866210937500000000, 0.001453298144042491912841796875,-0.095663353800773620605468750000,-0.068388663232326507568359375000,-0.007160167209804058074951171875, 0.017425049096345901489257812500,-5.941580772399902343750000000000, 0.191204562783241271972656250000, 0.004417046904563903808593750000, 0.022850321605801582336425781250,-0.213550627231597900390625000000,-0.155045047402381896972656250000, 0.679571866989135742187500000000, 0.003074747510254383087158203125,-0.002636981895193457603454589844, 0.055360324680805206298828125000, 0.018809702247381210327148437500,-0.160643383860588073730468750000,-0.525865137577056884765625000000,-0.211784303188323974609375000000, 0.008425508625805377960205078125,-0.074051328003406524658203125000,-0.306913822889328002929687500000,-0.365003407001495361328125000000, 0.191832706332206726074218750000 },
        { -1.051394820213317871093750000000,-0.656383693218231201171875000000,-0.417045891284942626953125000000, 0.562364876270294189453125000000,-0.897525846958160400390625000000, 0.075560882687568664550781250000, 0.152142122387886047363281250000, 0.007568782195448875427246093750,-1.892649292945861816406250000000,-0.206331297755241394042968750000, 0.283698201179504394531250000000,-0.258018016815185546875000000000,-1.672228336334228515625000000000, 0.688639938831329345703125000000,-0.235520273447036743164062500000,-0.859979867935180664062500000000, 0.485472977161407470703125000000,-0.228921309113502502441406250000, 0.057419870048761367797851562500,-0.489881277084350585937500000000,-1.077819466590881347656250000000,-0.940223217010498046875000000000, 0.775674819946289062500000000000, 1.336513280868530273437500000000, 0.031826570630073547363281250000,-0.992415189743041992187500000000,-0.851955056190490722656250000000, 0.953200280666351318359375000000, 0.006971847265958786010742187500,-0.314171671867370605468750000000,-1.505744218826293945312500000000,-1.154810309410095214843750000000 },
        { -0.230276629328727722167968750000, 0.081689707934856414794921875000, 0.022035162895917892456054687500, 0.118333734571933746337890625000, 0.022210810333490371704101562500,-0.171456977725028991699218750000, 0.341031581163406372070312500000,-0.743607521057128906250000000000,-0.527449727058410644531250000000,-0.217998787760734558105468750000,-0.319921553134918212890625000000,-0.289716631174087524414062500000, 0.174969196319580078125000000000,-0.808178722858428955078125000000,-0.254212766885757446289062500000, 0.038463577628135681152343750000,-0.309750020503997802734375000000,-0.019342159852385520935058593750,-0.204956829547882080078125000000,-0.188734814524650573730468750000,-0.006536909379065036773681640625, 0.040462363511323928833007812500,-0.909430623054504394531250000000, 0.011317624710500240325927734375, 0.068691357970237731933593750000,-0.244722336530685424804687500000, 0.030558645725250244140625000000,-0.021915454417467117309570312500,-0.259967476129531860351562500000,-0.081885524094104766845703125000,-0.577186822891235351562500000000, 0.105400398373603820800781250000 },
        { -0.021369516849517822265625000000, 0.059694204479455947875976562500,-0.064130894839763641357421875000,-0.014644864946603775024414062500, 0.041658621281385421752929687500,-0.037893053144216537475585937500, 0.123486027121543884277343750000, 0.162296667695045471191406250000, 0.079409062862396240234375000000,-0.063993334770202636718750000000,-0.103066317737102508544921875000, 0.057907767593860626220703125000, 0.105027452111244201660156250000,-0.013686556369066238403320312500, 0.014997794292867183685302734375, 0.087681345641613006591796875000, 0.069355353713035583496093750000, 0.161241725087165832519531250000,-0.059591576457023620605468750000,-0.100823372602462768554687500000,-0.057905759662389755249023437500, 0.091513790190219879150390625000, 0.132143571972846984863281250000,-0.148453459143638610839843750000,-0.092031620442867279052734375000,-0.002453185850754380226135253906,-0.104530028998851776123046875000, 0.092426016926765441894531250000, 0.027752414345741271972656250000,-0.028727592900395393371582031250,-0.127112686634063720703125000000, 0.116424344480037689208984375000 },
        {  0.071306683123111724853515625000, 0.019805565476417541503906250000,-0.302369594573974609375000000000,-0.002590813906863331794738769531,-0.008588540367782115936279296875,-1.115755319595336914062500000000, 0.023646155372262001037597656250,-0.070484437048435211181640625000,-0.018925672397017478942871093750,-0.099738165736198425292968750000, 0.337984532117843627929687500000,-0.122938968241214752197265625000,-0.061616457998752593994140625000,-0.896803379058837890625000000000, 0.040144495666027069091796875000,-0.114447578787803649902343750000,-0.053207725286483764648437500000, 0.060458008199930191040039062500, 0.060945380479097366333007812500,-0.161347240209579467773437500000, 0.010352512821555137634277343750,-0.165387436747550964355468750000, 0.020518884062767028808593750000, 0.075897812843322753906250000000,-0.151459306478500366210937500000, 0.015144100412726402282714843750, 0.204346925020217895507812500000, 0.034960661083459854125976562500,-0.136127769947052001953125000000,-0.125592216849327087402343750000, 0.063164830207824707031250000000, 0.357099145650863647460937500000 },
        { -1.337185025215148925781250000000,-0.832624614238739013671875000000, 0.966508448123931884765625000000,-0.365258306264877319335937500000, 0.341298431158065795898437500000,-1.932106852531433105468750000000,-0.567921698093414306640625000000, 0.145078837871551513671875000000,-0.122733056545257568359375000000,-0.009353929199278354644775390625, 1.808676123619079589843750000000,-0.052241690456867218017578125000, 1.699568986892700195312500000000, 0.253265202045440673828125000000, 0.475282460451126098632812500000, 0.706696152687072753906250000000,-0.185912311077117919921875000000,-0.187656864523887634277343750000,-0.167374357581138610839843750000,-0.813727557659149169921875000000, 1.280487179756164550781250000000, 0.987391769886016845703125000000,-1.006438851356506347656250000000,-1.483082652091979980468750000000,-0.684135019779205322265625000000,-0.707360804080963134765625000000, 1.246325373649597167968750000000,-1.279424548149108886718750000000,-0.358722329139709472656250000000,-0.299524456262588500976562500000,-1.739943385124206542968750000000,-0.242062464356422424316406250000 },
        { -0.418448626995086669921875000000,-0.026381887495517730712890625000,-0.773800671100616455078125000000,-0.299853324890136718750000000000, 0.387514591217041015625000000000, 0.597108006477355957031250000000,-0.310700416564941406250000000000,-0.609689950942993164062500000000, 0.270283639430999755859375000000,-0.121824949979782104492187500000,-0.039061240851879119873046875000,-0.071982629597187042236328125000, 0.078521236777305603027343750000,-0.518377125263214111328125000000, 0.031093040481209754943847656250,-2.036622762680053710937500000000,-0.262389302253723144531250000000, 0.058140460401773452758789062500,-0.259558528661727905273437500000, 0.005404288414865732192993164062, 0.138423666357994079589843750000,-1.721386790275573730468750000000,-0.267818570137023925781250000000,-0.331123113632202148437500000000, 0.342087477445602416992187500000,-0.157233759760856628417968750000,-1.349559903144836425781250000000,-0.091551095247268676757812500000,-0.227197840809822082519531250000,-0.080605074763298034667968750000, 0.224638313055038452148437500000,-2.265774965286254882812500000000 },
        {  0.202409565448760986328125000000,-0.019181933254003524780273437500,-0.272893011569976806640625000000, 0.634316802024841308593750000000, 0.074288100004196166992187500000, 0.020024923607707023620605468750,-0.115734852850437164306640625000, 0.018229341134428977966308593750,-0.234184384346008300781250000000,-0.140246376395225524902343750000, 0.927865147590637207031250000000,-0.056405644863843917846679687500,-0.089408487081527709960937500000,-0.357298642396926879882812500000,-1.226718068122863769531250000000, 0.043735709041357040405273437500, 0.388548940420150756835937500000,-0.133565366268157958984375000000,-0.134485483169555664062500000000,-3.762015104293823242187500000000,-0.044976517558097839355468750000,-0.020104357972741127014160156250, 0.075880169868469238281250000000,-0.051175199449062347412109375000,-0.795653998851776123046875000000,-0.313867330551147460937500000000, 0.010201684199273586273193359375,-0.016194852069020271301269531250,-0.084088735282421112060546875000,-0.227016374468803405761718750000,-0.184900745749473571777343750000,-2.533533811569213867187500000000 },
        {  0.083748839795589447021484375000,-0.208764001727104187011718750000,-3.728862285614013671875000000000,-0.272203028202056884765625000000, 0.065200164914131164550781250000, 0.047734867781400680541992187500,-0.496782153844833374023437500000,-2.562489509582519531250000000000,-0.992119252681732177734375000000,-0.087960027158260345458984375000, 0.152538940310478210449218750000, 0.028006417676806449890136718750, 0.001384484698064625263214111328,-0.268773317337036132812500000000,-5.003842353820800781250000000000,-0.076723217964172363281250000000,-0.252567261457443237304687500000,-0.028254466131329536437988281250,-0.021791486069560050964355468750,-0.214315608143806457519531250000, 0.005923103075474500656127929688, 0.079624220728874206542968750000,-0.048701200634241104125976562500,-0.038393396884202957153320312500,-0.026099447160959243774414062500, 0.277008742094039916992187500000, 0.347478985786437988281250000000,-0.031306140124797821044921875000,-0.105056338012218475341796875000, 0.073848903179168701171875000000,-0.129514962434768676757812500000,-0.127916455268859863281250000000 },
        {  0.144373044371604919433593750000,-0.173252433538436889648437500000, 0.148515582084655761718750000000, 1.299582004547119140625000000000, 0.048278406262397766113281250000, 0.157126024365425109863281250000,-0.508303165435791015625000000000,-1.886466383934020996093750000000,-0.027365908026695251464843750000,-0.059039443731307983398437500000, 0.060821123421192169189453125000,-0.242222800850868225097656250000, 0.138826236128807067871093750000,-0.478767812252044677734375000000,-0.962294399738311767578125000000, 0.059802263975143432617187500000,-0.221587657928466796875000000000,-0.130064904689788818359375000000, 0.066816940903663635253906250000,-0.800035119056701660156250000000, 0.124366797506809234619140625000, 0.128205150365829467773437500000,-0.771765708923339843750000000000,-0.274378448724746704101562500000,-2.407181024551391601562500000000, 0.109227485954761505126953125000,-0.221334323287010192871093750000,-0.094034776091575622558593750000,-0.049391802400350570678710937500,-0.322424829006195068359375000000, 0.342571556568145751953125000000, 0.032133661210536956787109375000 },
        {  0.084428071975708007812500000000,-0.640640735626220703125000000000,-0.601336658000946044921875000000, 0.758046805858612060546875000000, 0.213934525847434997558593750000,-0.506646811962127685546875000000, 0.195650443434715270996093750000,-1.510432243347167968750000000000,-0.314888328313827514648437500000, 0.052860222756862640380859375000,-2.005716562271118164062500000000,-0.086912982165813446044921875000, 0.043860811740159988403320312500, 0.692608594894409179687500000000,-0.665688693523406982421875000000,-0.145870611071586608886718750000, 0.852565288543701171875000000000, 0.121117860078811645507812500000, 0.102977558970451354980468750000,-1.699795603752136230468750000000, 0.175255656242370605468750000000, 0.022156789898872375488281250000, 0.627815485000610351562500000000,-0.072307728230953216552734375000,-0.839935362339019775390625000000,-0.215303525328636169433593750000,-0.103618323802947998046875000000,-0.030817691236734390258789062500,-0.057676691561937332153320312500,-0.090430736541748046875000000000, 0.157887548208236694335937500000, 1.749998807907104492187500000000 },
        { -0.305240452289581298828125000000,-0.172841489315032958984375000000, 0.138342484831809997558593750000,-0.858470439910888671875000000000, 0.072010271251201629638671875000,-0.012789812870323657989501953125,-0.000998659874312579631805419922,-0.109256081283092498779296875000,-0.281479030847549438476562500000, 0.045546457171440124511718750000,-0.382808595895767211914062500000,-0.076413147151470184326171875000, 0.043338168412446975708007812500, 0.626182854175567626953125000000, 0.608674287796020507812500000000, 0.032147292047739028930664062500, 0.230787262320518493652343750000,-0.159701302647590637207031250000,-0.029346587136387825012207031250,-0.694820761680603027343750000000, 0.027199149131774902343750000000,-0.023607501760125160217285156250,-1.246838331222534179687500000000,-0.038061298429965972900390625000, 0.186248660087585449218750000000,-0.411340087652206420898437500000,-0.039545726031064987182617187500,-0.028834354132413864135742187500, 0.017080703750252723693847656250,-0.341389328241348266601562500000, 0.192588672041893005371093750000, 0.574559688568115234375000000000 },
        { -0.710581541061401367187500000000, 0.194812998175621032714843750000,-0.294654458761215209960937500000,-0.468654066324234008789062500000,-0.106773637235164642333984375000,-1.439402222633361816406250000000, 0.365138888359069824218750000000, 0.767158627510070800781250000000,-0.730295956134796142578125000000,-0.101748183369636535644531250000, 0.430259376764297485351562500000,-0.252805739641189575195312500000, 1.473271846771240234375000000000, 0.089993782341480255126953125000, 0.217125266790390014648437500000,-0.225565120577812194824218750000, 0.635240972042083740234375000000,-0.167823940515518188476562500000,-0.271053314208984375000000000000, 1.145309448242187500000000000000, 0.853779911994934082031250000000, 1.177464127540588378906250000000,-2.888700962066650390625000000000,-1.438332200050354003906250000000,-1.035715937614440917968750000000,-0.611633598804473876953125000000,-1.362166166305541992187500000000,-1.130209803581237792968750000000,-0.105721220374107360839843750000,-0.309003651142120361328125000000,-0.628821134567260742187500000000, 0.357929617166519165039062500000 },
        {  0.467098653316497802734375000000, 1.766353487968444824218750000000,-1.059915661811828613281250000000,-0.801093459129333496093750000000,-1.408406138420104980468750000000, 0.334000259637832641601562500000, 0.527608275413513183593750000000, 0.051183413714170455932617187500,-1.681275844573974609375000000000,-0.037849832326173782348632812500, 0.065471567213535308837890625000,-0.035596940666437149047851562500,-2.890483140945434570312500000000,-0.806988954544067382812500000000, 0.836942851543426513671875000000,-1.339535832405090332031250000000,-1.856989979743957519531250000000,-0.184574365615844726562500000000,-0.043939374387264251708984375000,-0.861922144889831542968750000000,-2.028548955917358398437500000000,-1.337153315544128417968750000000,-1.416052460670471191406250000000, 2.282882690429687500000000000000, 0.362501114606857299804687500000, 1.381518363952636718750000000000,-0.824560463428497314453125000000, 1.970332384109497070312500000000, 0.059193730354309082031250000000,-0.049756813794374465942382812500,-0.548179626464843750000000000000,-0.011489526368677616119384765625 },
        { -0.078037522733211517333984375000,-0.028657095506787300109863281250, 0.100722730159759521484375000000, 0.001222911872901022434234619141,-0.053128484636545181274414062500, 0.084071792662143707275390625000, 0.101389408111572265625000000000, 0.131242439150810241699218750000,-0.067377172410488128662109375000, 0.160318344831466674804687500000, 0.007438476197421550750732421875,-0.071832112967967987060546875000,-0.154987320303916931152343750000, 0.161668926477432250976562500000, 0.040588740259408950805664062500,-0.155858784914016723632812500000, 0.074074521660804748535156250000, 0.169689252972602844238281250000,-0.006402569822967052459716796875,-0.165641978383064270019531250000,-0.118667379021644592285156250000,-0.126266747713088989257812500000,-0.078475706279277801513671875000,-0.128015384078025817871093750000, 0.090997636318206787109375000000, 0.085304647684097290039062500000,-0.107581116259098052978515625000, 0.131061479449272155761718750000, 0.061019364744424819946289062500, 0.064722262322902679443359375000, 0.136055901646614074707031250000, 0.044007442891597747802734375000 },
        { -0.151056706905364990234375000000,-1.450938940048217773437500000000, 0.049945313483476638793945312500,-1.022451281547546386718750000000, 0.215266853570938110351562500000,-0.610827207565307617187500000000, 0.271979898214340209960937500000,-1.127298712730407714843750000000,-0.627869904041290283203125000000,-0.065180614590644836425781250000,-0.995882451534271240234375000000,-0.268236368894577026367187500000,-0.168136030435562133789062500000,-0.020020300522446632385253906250, 0.673489451408386230468750000000, 0.010068529285490512847900390625,-0.096724674105644226074218750000,-0.123627156019210815429687500000, 0.001736609730869531631469726562,-1.689206242561340332031250000000,-0.049308568239212036132812500000,-0.084201723337173461914062500000,-0.148681640625000000000000000000, 0.099154487252235412597656250000,-0.928144097328186035156250000000,-2.900451421737670898437500000000,-0.506168067455291748046875000000, 0.082400374114513397216796875000,-0.036291103810071945190429687500, 0.111163884401321411132812500000, 0.893424570560455322265625000000,-1.527757167816162109375000000000 },
        { -1.132552742958068847656250000000,-0.012722413986921310424804687500,-2.452543735504150390625000000000,-0.653854250907897949218750000000, 0.161627739667892456054687500000, 0.728667736053466796875000000000,-0.857590258121490478515625000000, 1.069858908653259277343750000000,-0.023565920069813728332519531250,-0.100982643663883209228515625000,-0.198205426335334777832031250000,-0.024678930640220642089843750000,-0.147747218608856201171875000000,-0.840080320835113525390625000000,-0.546199977397918701171875000000,-1.368611931800842285156250000000,-0.565722286701202392578125000000, 0.080171376466751098632812500000,-0.067579686641693115234375000000,-0.479414433240890502929687500000,-0.044573348015546798706054687500,-0.177539572119712829589843750000, 0.200732320547103881835937500000,-0.120374873280525207519531250000,-0.849622130393981933593750000000, 1.733731865882873535156250000000,-0.466532915830612182617187500000, 0.126525789499282836914062500000,-0.132970169186592102050781250000,-0.174420982599258422851562500000,-0.068783074617385864257812500000,-0.774228453636169433593750000000 },
        { -0.178825721144676208496093750000, 0.154222890734672546386718750000, 0.305234134197235107421875000000, 0.286206483840942382812500000000,-0.485444456338882446289062500000, 0.235285609960556030273437500000, 0.436115175485610961914062500000,-0.171333342790603637695312500000, 0.714031755924224853515625000000,-0.062489613890647888183593750000, 0.133970409631729125976562500000,-0.067541882395744323730468750000,-0.433525264263153076171875000000, 0.068613588809967041015625000000,-1.107611179351806640625000000000,-0.119289226830005645751953125000, 1.241979479789733886718750000000,-0.272593259811401367187500000000, 0.120098754763603210449218750000, 0.659944534301757812500000000000,-0.349367171525955200195312500000,-0.319872319698333740234375000000,-2.000404119491577148437500000000, 0.520646929740905761718750000000,-6.656533241271972656250000000000, 0.403766989707946777343750000000,-0.170020714402198791503906250000, 0.429693371057510375976562500000, 0.034081883728504180908203125000,-0.037630591541528701782226562500, 0.957284748554229736328125000000, 1.033483386039733886718750000000 },
        {  0.594415187835693359375000000000,-0.478180646896362304687500000000,-0.523452281951904296875000000000, 0.530491054058074951171875000000, 0.141855433583259582519531250000,-0.116123080253601074218750000000,-0.010985846631228923797607421875, 0.450881779193878173828125000000,-1.450796008110046386718750000000,-0.018079012632369995117187500000,-0.882190167903900146484375000000, 0.072960086166858673095703125000,-0.206335812807083129882812500000,-1.633012175559997558593750000000,-3.506790637969970703125000000000,-3.530215263366699218750000000000,-3.547236204147338867187500000000,-0.024550694972276687622070312500,-0.118420720100402832031250000000,-2.183165073394775390625000000000,-0.127237796783447265625000000000,-2.580565690994262695312500000000, 0.698038637638092041015625000000, 0.032827682793140411376953125000, 0.051913209259510040283203125000, 0.251610398292541503906250000000, 0.116788692772388458251953125000, 0.049628831446170806884765625000,-0.180421322584152221679687500000, 0.028171084821224212646484375000,-0.515805363655090332031250000000,-2.535957574844360351562500000000 },
        { -0.169293835759162902832031250000,-1.811271905899047851562500000000,-0.228395491838455200195312500000, 0.636265039443969726562500000000, 0.211028665304183959960937500000, 0.115382738411426544189453125000,-0.336878329515457153320312500000,-1.103210687637329101562500000000,-1.214002490043640136718750000000,-0.120469532907009124755859375000, 0.069362998008728027343750000000, 0.062390986829996109008789062500,-0.019091585651040077209472656250,-0.337663918733596801757812500000,-1.661660432815551757812500000000,-0.054211705923080444335937500000, 0.115728408098220825195312500000,-0.201006934046745300292968750000,-0.136703550815582275390625000000,-2.017943620681762695312500000000, 0.061194203794002532958984375000, 0.027220621705055236816406250000, 0.162841305136680603027343750000, 0.014860041439533233642578125000,-0.205904945731163024902343750000,-0.166994839906692504882812500000, 0.268665939569473266601562500000,-0.019018739461898803710937500000,-0.011579474434256553649902343750,-0.150322318077087402343750000000,-0.405804753303527832031250000000,-3.392155408859252929687500000000 },
        {  0.177775889635086059570312500000, 0.216089129447937011718750000000, 0.078371696174144744873046875000, 0.069256603717803955078125000000, 0.280220657587051391601562500000,-0.900224804878234863281250000000, 0.500112414360046386718750000000,-1.735932469367980957031250000000,-1.272935509681701660156250000000, 0.044242829084396362304687500000,-0.082860112190246582031250000000, 0.083719052374362945556640625000, 0.002015513600781559944152832031,-2.725433588027954101562500000000,-0.100714020431041717529296875000, 0.019553687423467636108398437500,-0.384059965610504150390625000000, 0.033108022063970565795898437500, 0.059635374695062637329101562500, 0.360199838876724243164062500000,-0.009931683540344238281250000000,-0.020158378407359123229980468750,-0.038613274693489074707031250000,-0.000508444907609373331069946289, 0.728088080883026123046875000000, 0.092694610357284545898437500000, 0.015014766715466976165771484375,-0.004449838772416114807128906250,-0.097877129912376403808593750000, 0.006255468819290399551391601562,-1.725210905075073242187500000000, 0.181648582220077514648437500000 },
        { -0.039874158799648284912109375000, 0.107526376843452453613281250000, 0.172502428293228149414062500000,-0.075519166886806488037109375000,-0.036004792898893356323242187500, 0.132557332515716552734375000000, 0.090310290455818176269531250000,-0.088226333260536193847656250000, 0.120769172906875610351562500000, 0.003272170200943946838378906250, 0.004848874639719724655151367188,-0.125724121928215026855468750000, 0.145099043846130371093750000000,-0.071355849504470825195312500000, 0.111681506037712097167968750000, 0.110768854618072509765625000000,-0.144725188612937927246093750000, 0.158183842897415161132812500000,-0.016012394800782203674316406250,-0.042558651417493820190429687500, 0.141121715307235717773437500000, 0.086506001651287078857421875000,-0.064345970749855041503906250000,-0.043126184493303298950195312500,-0.145280435681343078613281250000,-0.019510084763169288635253906250, 0.123457230627536773681640625000, 0.052088744938373565673828125000, 0.111757598817348480224609375000,-0.075179114937782287597656250000, 0.151153877377510070800781250000, 0.122486062347888946533203125000 },
        {  0.695976734161376953125000000000,-0.317278563976287841796875000000, 0.331417262554168701171875000000,-0.986951351165771484375000000000, 0.084474265575408935546875000000, 0.304309129714965820312500000000,-1.434206366539001464843750000000, 0.637035965919494628906250000000,-1.618519544601440429687500000000,-0.160941377282142639160156250000,-0.688468515872955322265625000000,-0.223835244774818420410156250000,-0.510840058326721191406250000000,-3.036052703857421875000000000000, 0.910349369049072265625000000000, 0.054420441389083862304687500000,-0.506902992725372314453125000000,-0.142879888415336608886718750000,-0.338686674833297729492187500000,-0.875441491603851318359375000000,-0.281055957078933715820312500000, 0.007178509607911109924316406250,-1.474557518959045410156250000000,-0.121257424354553222656250000000, 0.293571919202804565429687500000,-0.242662176489830017089843750000,-0.220312759280204772949218750000, 0.043488960713148117065429687500,-0.055358823388814926147460937500,-0.161693215370178222656250000000,-0.762636780738830566406250000000,-3.770585536956787109375000000000 },
        {  0.476185262203216552734375000000, 0.392599135637283325195312500000, 0.033433534204959869384765625000,-0.521217167377471923828125000000,-0.045548394322395324707031250000, 0.504440307617187500000000000000,-0.244016230106353759765625000000,-0.171628683805465698242187500000, 0.590472579002380371093750000000,-0.077463746070861816406250000000, 0.184558838605880737304687500000,-0.134539812803268432617187500000, 0.053909145295619964599609375000,-0.358471125364303588867187500000,-0.068117439746856689453125000000, 0.072993576526641845703125000000,-0.424127131700515747070312500000, 0.044960334897041320800781250000,-0.181747555732727050781250000000, 0.333264797925949096679687500000, 0.024350142106413841247558593750, 0.061650257557630538940429687500, 0.908295094966888427734375000000,-0.095594599843025207519531250000,-0.188146308064460754394531250000,-0.489797323942184448242187500000, 0.023873735219240188598632812500,-0.019204987213015556335449218750,-0.014959230087697505950927734375,-0.228796020150184631347656250000,-1.078598737716674804687500000000,-0.954875171184539794921875000000 },
        { -3.028497219085693359375000000000,-0.642598152160644531250000000000, 0.219079151749610900878906250000,-0.459704160690307617187500000000, 0.174095258116722106933593750000,-0.817770421504974365234375000000, 0.126112222671508789062500000000, 0.947320520877838134765625000000,-0.239163935184478759765625000000, 0.055686909705400466918945312500, 0.019820518791675567626953125000,-0.001657493179664015769958496094, 0.207039445638656616210937500000, 0.019732816144824028015136718750, 0.357248097658157348632812500000,-0.028886135667562484741210937500,-3.478625774383544921875000000000,-0.158291533589363098144531250000,-0.186091870069503784179687500000,-1.209303379058837890625000000000, 0.233352422714233398437500000000, 0.087858185172080993652343750000,-0.220423802733421325683593750000,-0.136665925383567810058593750000,-1.496710181236267089843750000000,-0.401525914669036865234375000000, 0.085380457341670989990234375000,-0.163711145520210266113281250000,-0.145691171288490295410156250000,-0.041025254875421524047851562500,-1.858483076095581054687500000000,-1.735223412513732910156250000000 },
        {  0.084750168025493621826171875000, 0.086857713758945465087890625000,-2.459112405776977539062500000000, 0.008161000907421112060546875000, 0.035473801195621490478515625000, 0.022330470383167266845703125000,-0.517946779727935791015625000000,-2.121411800384521484375000000000,-0.055537689477205276489257812500, 0.025232860818505287170410156250, 0.133138969540596008300781250000, 0.043062478303909301757812500000, 0.025673761963844299316406250000, 0.044913053512573242187500000000,-2.930068254470825195312500000000, 0.040467742830514907836914062500,-0.091413781046867370605468750000, 0.027497747913002967834472656250,-0.090349979698657989501953125000, 0.128343224525451660156250000000, 0.023333834484219551086425781250, 0.116272889077663421630859375000,-0.110148087143898010253906250000,-0.048467431217432022094726562500,-0.069135226309299468994140625000,-0.253600627183914184570312500000, 0.267043471336364746093750000000,-0.058844346553087234497070312500, 0.035760100930929183959960937500,-0.029219312593340873718261718750,-0.324243128299713134765625000000,-0.127523019909858703613281250000 },
        { -10.073122024536132812500000000000,-12.393413543701171875000000000000,-2.164335012435913085937500000000, 0.583223581314086914062500000000,-2.249647617340087890625000000000,-0.120718598365783691406250000000, 2.116126775741577148437500000000, 0.207931414246559143066406250000,-0.514477133750915527343750000000,-0.217103019356727600097656250000, 0.479193538427352905273437500000,-0.145922914147377014160156250000, 0.254245549440383911132812500000,-0.073266118764877319335937500000,-0.239194944500923156738281250000,-5.320525169372558593750000000000, 0.050210326910018920898437500000,-0.085796855390071868896484375000, 0.073313742876052856445312500000, 0.979514837265014648437500000000,-0.250322550535202026367187500000,-2.541529178619384765625000000000,-5.166074752807617187500000000000, 0.568188667297363281250000000000,-1.927600502967834472656250000000, 0.333596497774124145507812500000,-1.086709380149841308593750000000, 0.136248290538787841796875000000, 0.176636114716529846191406250000,-0.079153031110763549804687500000, 0.298301458358764648437500000000, 0.372890800237655639648437500000 },
        { -2.223058223724365234375000000000,-0.117848500609397888183593750000,-1.555899620056152343750000000000,-1.241247773170471191406250000000, 0.045535147190093994140625000000,-4.618567943572998046875000000000,-0.420991420745849609375000000000,-2.676482677459716796875000000000, 0.684394776821136474609375000000, 0.020861821249127388000488281250,-1.489337801933288574218750000000, 0.098931677639484405517578125000, 0.017668770626187324523925781250,-1.661979317665100097656250000000,-0.093715883791446685791015625000,-2.710683584213256835937500000000, 0.169319942593574523925781250000,-0.040472690016031265258789062500,-0.060948099941015243530273437500,-0.238444879651069641113281250000,-0.011300343088805675506591796875,-2.973534822463989257812500000000, 0.518678307533264160156250000000, 0.090001560747623443603515625000,-0.896446287631988525390625000000,-0.104735739529132843017578125000,-0.053047072142362594604492187500,-0.014150190167129039764404296875,-0.037164732813835144042968750000, 0.029560007154941558837890625000,-0.452440142631530761718750000000,-0.192669346928596496582031250000 },
        {  0.192967399954795837402343750000,-0.592749178409576416015625000000, 0.423451215028762817382812500000,-0.634861230850219726562500000000, 1.124665260314941406250000000000, 0.681646823883056640625000000000,-0.342978417873382568359375000000, 0.151289224624633789062500000000, 0.610286891460418701171875000000,-0.065491139888763427734375000000,-0.079691626131534576416015625000,-0.243716135621070861816406250000, 0.628918528556823730468750000000, 0.425532758235931396484375000000, 0.369493901729583740234375000000, 0.811174273490905761718750000000,-0.314345300197601318359375000000,-0.193913713097572326660156250000,-0.153973549604415893554687500000,-0.363246977329254150390625000000, 0.680837869644165039062500000000, 0.302294939756393432617187500000,-0.257190138101577758789062500000,-0.839451014995574951171875000000, 0.790217101573944091796875000000, 0.497497409582138061523437500000, 0.834240555763244628906250000000,-0.518073081970214843750000000000,-0.295184999704360961914062500000,-0.021227849647402763366699218750, 0.038781058043241500854492187500,-0.417516380548477172851562500000 },
        { -0.893447041511535644531250000000,-0.801997661590576171875000000000, 0.370884507894515991210937500000,-0.386787295341491699218750000000,-0.090137951076030731201171875000,-0.822282671928405761718750000000,-0.074484258890151977539062500000, 1.334171414375305175781250000000, 0.157610282301902770996093750000, 0.000932417751755565404891967773, 0.640585482120513916015625000000,-0.037311688065528869628906250000,-0.038166251033544540405273437500, 0.933908700942993164062500000000, 0.346406966447830200195312500000,-2.038897752761840820312500000000,-0.015084574930369853973388671875, 0.070646546781063079833984375000,-0.108096562325954437255859375000, 0.974581718444824218750000000000,-0.013048208318650722503662109375,-1.666192173957824707031250000000, 0.009174235165119171142578125000,-0.093020826578140258789062500000, 1.024042725563049316406250000000, 0.582423865795135498046875000000,-0.115450970828533172607421875000,-0.081447347998619079589843750000,-0.145580023527145385742187500000,-0.092219769954681396484375000000, 0.210387647151947021484375000000,-2.520595312118530273437500000000 },
        {  0.525881469249725341796875000000, 0.234279677271842956542968750000,-0.407334357500076293945312500000, 0.257667988538742065429687500000, 0.106994703412055969238281250000, 0.194161847233772277832031250000,-0.102323055267333984375000000000,-0.424683123826980590820312500000, 0.252086132764816284179687500000,-0.051654528826475143432617187500, 0.148745253682136535644531250000,-0.006570613943040370941162109375,-0.049186740070581436157226562500, 0.285835832357406616210937500000,-0.257814407348632812500000000000,-0.037227168679237365722656250000,-0.033188555389642715454101562500,-0.129991650581359863281250000000,-0.193189144134521484375000000000, 0.075580403208732604980468750000,-0.077141590416431427001953125000, 0.010728851892054080963134765625,-0.022840859368443489074707031250,-0.119181096553802490234375000000,-3.186436891555786132812500000000,-0.557428479194641113281250000000, 0.711717724800109863281250000000,-0.032226204872131347656250000000, 0.060451064258813858032226562500,-0.430786311626434326171875000000, 0.362882733345031738281250000000,-0.699437499046325683593750000000 },
        {  0.004697763361036777496337890625,-0.139735549688339233398437500000,-0.834130108356475830078125000000,-0.338979661464691162109375000000,-0.149056702852249145507812500000, 0.087358199059963226318359375000,-0.222171157598495483398437500000, 0.397909760475158691406250000000, 0.034162957221269607543945312500,-0.150696828961372375488281250000, 0.492802917957305908203125000000,-0.049454193562269210815429687500,-0.092745497822761535644531250000,-0.534125149250030517578125000000,-0.271833837032318115234375000000, 0.037988625466823577880859375000, 0.223469376564025878906250000000,-0.250253975391387939453125000000,-0.173756092786788940429687500000,-0.392649829387664794921875000000,-0.017050389200448989868164062500,-0.022885102778673171997070312500,-0.830608010292053222656250000000, 0.057208299636840820312500000000,-0.514994442462921142578125000000, 0.741444766521453857421875000000, 0.079113312065601348876953125000, 0.033937420696020126342773437500,-0.271726876497268676757812500000,-0.305514395236968994140625000000,-3.671972036361694335937500000000, 0.094887703657150268554687500000 },
    };
    float x_2[32];
    for (unsigned int col = 0; col < 32; ++col)
    {
        x_2[col] = 0;
        for (unsigned int inner = 0; inner < 32; ++inner)
        {
            x_2[col] += x_1[inner]*wgtT_2[inner][col];
        }
        x_2[col] += bias_2[col];
    }
    
    // (3): ReLU()
    float x_3[32];
    for (unsigned int col = 0; col < 32; ++col)
    {
        x_3[col] = (x_2[col] > 0.f) ? x_2[col] : 0.f;
    }
    
    // (4): Linear(in_features=32, out_features=1, bias=True) => x = x*W_T + b
    float bias_4[1] = {
         1.010855674743652343750000000000
    };
    float wgtT_4[32][1] = {
        {  0.094855472445487976074218750000 },
        {  0.095064386725425720214843750000 },
        {  0.096369311213493347167968750000 },
        { -0.136295437812805175781250000000 },
        {  0.098151743412017822265625000000 },
        { -0.130086675286293029785156250000 },
        { -0.046204429119825363159179687500 },
        { -0.109909340739250183105468750000 },
        {  0.096451960504055023193359375000 },
        {  0.030295511707663536071777343750 },
        { -0.093773864209651947021484375000 },
        {  0.013709991239011287689208984375 },
        {  0.300436466932296752929687500000 },
        { -0.125020742416381835937500000000 },
        { -0.145260199904441833496093750000 },
        {  0.817734897136688232421875000000 },
        { -0.062007453292608261108398437500 },
        { -0.038836907595396041870117187500 },
        { -0.057548396289348602294921875000 },
        { -0.342613101005554199218750000000 },
        { -0.601488173007965087890625000000 },
        { -1.367728114128112792968750000000 },
        {  0.153722271323204040527343750000 },
        {  0.384998023509979248046875000000 },
        {  0.040596414357423782348632812500 },
        {  0.060411714017391204833984375000 },
        {  0.088375754654407501220703125000 },
        { -0.499434441328048706054687500000 },
        {  0.026612307876348495483398437500 },
        {  0.002126076258718967437744140625 },
        { -0.096430055797100067138671875000 },
        { -0.228787660598754882812500000000 },
    };
    float x_4[1];
    for (unsigned int col = 0; col < 1; ++col)
    {
        x_4[col] = 0;
        for (unsigned int inner = 0; inner < 32; ++inner)
        {
            x_4[col] += x_3[inner]*wgtT_4[inner][col];
        }
        x_4[col] += bias_4[col];
    }
    
    // (5): Sigmoid()
    float x_5[1];
    for (unsigned int col = 0; col < 1; ++col)
    {
        x_5[col] = exp(x_4[col])/(exp(x_4[col]) + 1);
    }

    /* pass = pass and (x_5[0] > 0.414823383092880193512286268742); // 92% sig eff 44% bkg eff */
    pass = pass and (x_5[0] > 0.611521303653717041015625000000); // 82% sig eff 24% bkg eff
    if (not pass) return pass;

    float xVec[] = {x1, x2, x3, x4, x5};
    float yVec[] = {y1, y2, y3, y4, y5};
    float sigmas[5], delta1[5], delta2[5], slopes[5];
    bool isFlat[5];
    //5 categories for sigmas
    const uint16_t lowerModuleIndices[] = {lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4, lowerModuleIndex5};

    computeSigmasForRegression(modulesInGPU, lowerModuleIndices, delta1, delta2, slopes, isFlat);
    regressionRadius = computeRadiusUsingRegression(5,xVec, yVec, delta1, delta2, slopes, isFlat, regressionG, regressionF, sigmas, chiSquared);

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
__device__ bool SDL::passT5RZConstraint(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, unsigned int firstMDIndex, unsigned int secondMDIndex, unsigned int thirdMDIndex, unsigned int fourthMDIndex, unsigned int fifthMDIndex, uint16_t& lowerModuleIndex1, uint16_t& lowerModuleIndex2, uint16_t& lowerModuleIndex3, uint16_t& lowerModuleIndex4, uint16_t& lowerModuleIndex5, float& rzChiSquared, float inner_pt, float innerRadius, float g, float f, bool& TightCutFlag) 
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

    // start from a circle of inner T3.
    // to determine the charge
    int charge=0;
    float slope3c=(y3-y_center)/(x3-x_center);
    float slope1c=(y1-y_center)/(x1-x_center);
    // these 4 "if"s basically separate the x-y plane into 4 quarters. It determines geometrically how a circle and line slope goes and their positions, and we can get the charges correspondingly.
    if((y3-y_center)>0 && (y1-y_center)>0) 
    {
        if (slope1c>0 && slope3c<0) charge=-1; // on x axis of a quarter, 3 hits go anti-clockwise
        else if (slope1c<0 && slope3c>0) charge=1; // on x axis of a quarter, 3 hits go clockwise
        else if (slope3c>slope1c) charge=-1; 
        else if (slope3c<slope1c) charge=1;
    }
    else if((y3-y_center)<0 && (y1-y_center)<0) 
    {
        if (slope1c<0 && slope3c>0) charge=1;
        else if (slope1c>0 && slope3c<0) charge=-1;
        else if (slope3c>slope1c) charge=-1; 
        else if (slope3c<slope1c) charge=1;
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

    // Above line only gives you the correct value of Px and Py, but signs of Px and Py calculated below. 
    // We look at if the circle is clockwise or anti-clock wise, to make it simpler, we separate the x-y plane into 4 quarters.
    if (x_init>x_center && y_init>y_center) //1st quad
    {
        if (charge==1) Py=-Py;
        if (charge==-1) Px=-Px;
    }
    if (x_init<x_center && y_init>y_center) //2nd quad
    {
        if (charge==-1) {
            Px=-Px; 
            Py=-Py;
        }
    }
    if (x_init<x_center && y_init<y_center) //3rd quad
    {
        if (charge==1) Px=-Px;
        if (charge==-1) Py=-Py;
    }        
    if (x_init>x_center && y_init<y_center) //4th quad
    {
        if (charge==1) {
            Px=-Px; 
            Py=-Py;
        }
    }

    // But if the initial T5 curve goes across quarters(i.e. cross axis to separate the quarters), need special redeclaration of Px,Py signs on these to avoid errors
    if (moduleType3==0){ // 0 is ps
        if (x4<x3 && x3<x2) Px=-abs(Px);
        if (x4>x3 && x3>x2) Px=abs(Px);
        if (y4<y3 && y3<y2) Py=-abs(Py);
        if (y4>y3 && y3>y2) Py=abs(Py);
    }
    else if(moduleType3==1) // 1 is 2s
    {
        if (x3<x2 && x2<x1) Px=-abs(Px);
        if (x3>x2 && x2>x1) Px=abs(Px);
        if (y3<y2 && y2<y1) Py=-abs(Py);
        if (y3>y2 && y2>y1) Py=abs(Py);        
    }

    //to get Pz, we use pt/pz=ds/dz, ds is the arclength between MD1 and MD3.
    float AO=sqrt((x1-x_center)*(x1-x_center)+(y1-y_center)*(y1-y_center));
    float BO=sqrt((x_init-x_center)*(x_init-x_center)+(y_init-y_center)*(y_init-y_center));
    float AB=sqrt((x1-x_init)*(x1-x_init)+(y1-y_init)*(y1-y_init)); 
    float dPhi = acos((AO*AO+BO*BO-AB*AB)/(2*AO*BO));
    float ds=innerRadius/100*dPhi;

    float Pz=(z_init-z1)/ds*Pt;
    float p = sqrt(Px*Px+Py*Py+Pz*Pz);

    float B = SDL::magnetic_field;
    float a = -0.299792*B*charge;

    float zsi, rtsi;
    int layeri, moduleTypei;
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
        // for endcap
        float s = (zsi-z_init)*p/Pz;
        float x = x_init + Px/a*sin(rou*s)-Py/a*(1-cos(rou*s));
        float y = y_init + Py/a*sin(rou*s)+Px/a*(1-cos(rou*s));
        diffr = (rtsi-sqrt(x*x+y*y))*100;

        // for barrel
        if (layeri<=6)
        {
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
            if (isnan(diffz1)) diffz = diffz2;
            else if (isnan(diffz2)) diffz = diffz1;
            else {diffz = (fabs(diffz1)<fabs(diffz2)) ? diffz1 : diffz2;}
        }
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
            float projection_missing=1;
        if (drdz<1)
            projection_missing = ((subdets == SDL::Endcap) or (side == SDL::Center)) ? 1.f : 1/sqrt(1+drdz*drdz); // cos(atan(drdz)), if dr/dz<1
        if (drdz>1)
            projection_missing = ((subdets == SDL::Endcap) or (side == SDL::Center)) ? 1.f : drdz/sqrt(1+drdz*drdz);//sin(atan(drdz)), if dr/dz>1
            error=error*projection_missing;
        }
        rzChiSquared += 12*(residual * residual)/(error * error);
    }
    // for set rzchi2 cut
    // if the 5 points are linear, helix calculation gives nan
    if (inner_pt>100 || isnan(rzChiSquared)){
        float slope;
        if(moduleType1 == 0 and moduleType2 == 0 and moduleType3 == 1) //PSPS2S
        {
            slope = (z2 -z1)/(rt2 - rt1);
        }
        else
        {
            slope = (z3 - z1)/(rt3 - rt1);
        }
        float residual4_linear = (layer4 <= 6)? ((z4 - z1) - slope * (rt4 - rt1)) : ((rt4 - rt1) - (z4 - z1)/slope);
        float residual5_linear = (layer4 <= 6) ? ((z5 - z1) - slope * (rt5 - rt1)) : ((rt5 - rt1) - (z5 - z1)/slope);

        // creating a chi squared type quantity
        // 0-> PS, 1->2S
        residual4_linear = (moduleType4 == 0) ? residual4_linear/0.15f : residual4_linear/5.0f;
        residual5_linear = (moduleType5 == 0) ? residual5_linear/0.15f : residual5_linear/5.0f;
        residual4_linear = residual4_linear*100;
        residual5_linear = residual5_linear*100;

        rzChiSquared = 12 * (residual4_linear * residual4_linear + residual5_linear * residual5_linear);
        return rzChiSquared < 4.677f;
    }

    // when building T5, apply 99% chi2 cuts as default, and add to pT5 collection. But when adding T5 to TC collections, appy 95% cut to reduce the fake rate
    TightCutFlag = false;
    //categories!
    // The category numbers are related to module regions and layers, decoding of the region numbers can be found here in slide 2 table. https://github.com/SegmentLinking/TrackLooper/files/11420927/part.2.pdf
    // The commented numbers after each case is the region code, and can look it up from the table to see which category it belongs to. For example, //0 means T5 built with Endcap 1,2,3,4,5 ps modules

    if(layer1 == 7 and layer2 == 8 and layer3 == 9 and layer4 == 10 and layer5 == 11) //0
    {
        if (rzChiSquared < 94.470f) TightCutFlag = 1;
        return true;
    }

    else if(layer1 == 7 and layer2 == 8 and layer3 == 9 and layer4 == 10 and layer5 == 16) //1
    {
        if (rzChiSquared < 22.099f) TightCutFlag = 1;
        return rzChiSquared < 37.956f;
    }
    
    else if(layer1 == 7 and layer2 == 8 and layer3 == 9 and layer4 == 15 and layer5 == 16) //2
    {
        if (rzChiSquared < 7.992f) TightCutFlag = 1;
        return rzChiSquared < 11.622f;
    }

    else if(layer1 == 1 and layer2 == 7 and layer3 == 8 and layer4 == 9) 
    {
        if (layer5 == 10) //3
        {
            if (rzChiSquared < 111.390f) TightCutFlag = 1;
            return true;
        }
        if (layer5 == 15) //4
        {
            if (rzChiSquared < 18.351f) TightCutFlag = 1;
            return rzChiSquared < 37.941f;
        }
    }

    else if(layer1 == 1 and layer2 == 2 and layer3 == 7)
    {
        if(layer4 == 8 and layer5 == 9) //5
        {
            if (rzChiSquared < 116.148f) TightCutFlag = 1;
            return true;
        }
        if(layer4 == 8 and layer5 == 14) //6
        {
            if (rzChiSquared < 19.352f) TightCutFlag = 1;
            return rzChiSquared < 52.561f;
        }
        else if(layer4 == 13 and layer5 == 14) //7
        {
            if (rzChiSquared < 10.392f) TightCutFlag = 1;
            return rzChiSquared < 13.76f;
        }
    }
    else if(layer1 == 1 and layer2 == 2 and layer3 == 3)
    {
        if(layer4 == 7 and layer5 == 8) //8
        {   
            if (rzChiSquared < 27.824f) TightCutFlag = 1;
            return rzChiSquared < 44.247f;
        }
        else if(layer4 == 7 and layer5 == 13) //9
        {
            if (rzChiSquared < 18.145f) TightCutFlag = 1;
            return rzChiSquared < 33.752f;
        }
        else if(layer4 == 12 and layer5 == 13) //10
        {
            if (rzChiSquared < 13.308f) TightCutFlag = 1;
            return rzChiSquared < 21.213f;
        }
        else if(layer4 == 4 and layer5 == 5) //11
        {
            if (rzChiSquared < 15.627f) TightCutFlag = 1;
            return rzChiSquared < 29.035f; 
        }
        else if(layer4 == 4 and layer5 == 12) //12
        {
            if (rzChiSquared < 14.64f) TightCutFlag = 1;
            return rzChiSquared < 23.037f;
        }
    }

    else if(layer1 == 2 and layer2 == 7 and layer3 == 8)
    {
        if(layer4 == 9 and layer5 == 15) //14
        {
            if (rzChiSquared < 24.662f) TightCutFlag = 1;
            return rzChiSquared < 41.036f;
        }
        else if(layer4 == 14 and layer5 == 15) //15
        {
            if (rzChiSquared < 8.866f) TightCutFlag = 1;
            return rzChiSquared < 14.092f;
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
            if (rzChiSquared < 10.772f) TightCutFlag = 1;
            return rzChiSquared < 17.945f;
        }
    }
    else if(layer1 == 2 and layer2 == 3 and layer3 == 4)
    {
        if(layer4 == 5 and layer5 == 6) //18
        {
            if (rzChiSquared < 6.065f) TightCutFlag = 1;
            return rzChiSquared < 8.803f;
        }
        else if(layer4 == 5 and layer5 == 12) //19
        {
            if (rzChiSquared < 5.693f) TightCutFlag = 1;
            return rzChiSquared < 7.930f;
        }

        else if(layer4 == 12 and layer5 == 13) //20
        {
            if (rzChiSquared < 5.473f) TightCutFlag = 1;
            return rzChiSquared < 7.626f;
        }
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

    float denomInv = 1.0f/((y1 - y3) * (x2 - x3) - (x1 - x3) * (y2 - y3));

    float xy1sqr = x1 * x1 + y1 * y1;

    float xy2sqr = x2 * x2 + y2 * y2;

    float xy3sqr = x3 * x3 + y3 * y3;

    g = 0.5f * ((y3 - y2) * xy1sqr + (y1 - y3) * xy2sqr + (y2 - y1) * xy3sqr) * denomInv;

    f = 0.5f * ((x2 - x3) * xy1sqr + (x3 - x1) * xy2sqr + (x1 - x2) * xy3sqr) * denomInv;

    float c = ((x2 * y3 - x3 * y2) * xy1sqr + (x3 * y1 - x1 * y3) * xy2sqr + (x1 * y2 - x2 * y1) * xy3sqr) * denomInv;

    if(((y1 - y3) * (x2 - x3) - (x1 - x3) * (y2 - y3) == 0) || (g * g + f * f - c < 0))
    {
#ifdef Warnings
        printf("three collinear points or FATAL! r^2 < 0!\n");
#endif
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
#ifdef Warnings
            printf("ERROR!!!!! I SHOULDN'T BE HERE!!!! subdet = %d, type = %d, side = %d\n", moduleSubdet, moduleType, moduleSide);
#endif
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
#ifdef Warnings
        printf("FATAL! r^2 < 0!\n");
#endif
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
                if(totOccupancyQuintuplets >= (rangesInGPU.quintupletModuleOccupancy[lowerModule1]))
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
#ifdef Warnings
                        printf("Quintuplets : no memory for module at module index = %d\n", lowerModule1);
#endif
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
__global__ void SDL::addQuintupletRangesToEventExplicit(struct modules& modulesInGPU, struct quintuplets& quintupletsInGPU, struct objectRanges& rangesInGPU)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int np = gridDim.x * blockDim.x;
    for(uint16_t i = gid; i < *modulesInGPU.nLowerModules; i+= np)
    {
        if(quintupletsInGPU.nQuintuplets[i] == 0 or rangesInGPU.quintupletModuleIndices[i] == -1)
        {
            rangesInGPU.quintupletRanges[i * 2] = -1;
            rangesInGPU.quintupletRanges[i * 2 + 1] = -1;
        }
       else
        {
            rangesInGPU.quintupletRanges[i * 2] = rangesInGPU.quintupletModuleIndices[i];
            rangesInGPU.quintupletRanges[i * 2 + 1] = rangesInGPU.quintupletModuleIndices[i] + quintupletsInGPU.nQuintuplets[i] - 1;
        }
    }
}
