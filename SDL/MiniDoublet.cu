#ifdef __CUDACC__
#define CUDA_CONST_VAR __device__
#endif

# include "MiniDoublet.cuh"
#define SDL_INF 123456789

//#ifdef CACHE_ALLOC
#include "allocate.h"
//#endif

//defining the constant host device variables right up here
CUDA_CONST_VAR float SDL::miniMulsPtScaleBarrel[6] = {0.0052, 0.0038, 0.0034, 0.0034, 0.0032, 0.0034};
CUDA_CONST_VAR float SDL::miniMulsPtScaleEndcap[5] = {0.006, 0.006, 0.006, 0.006, 0.006}; 
CUDA_CONST_VAR float SDL::miniRminMeanBarrel[6] = {21.8, 34.6, 49.6, 67.4, 87.6, 106.8};
CUDA_CONST_VAR float SDL::miniRminMeanEndcap[5] = {131.4, 156.2, 185.6, 220.3, 261.5};
//CUDA_CONST_VAR float SDL::miniDeltaTilted[3] = {0.26, 0.26, 0.26};
//CUDA_CONST_VAR float SDL::miniDeltaFlat[6] ={0.26, 0.16, 0.16, 0.18, 0.18, 0.18};
//CUDA_CONST_VAR float SDL::miniDeltaLooseTilted[3] = {0.4,0.4,0.4};
//CUDA_CONST_VAR float SDL::miniDeltaEndcap[5][15];
CUDA_CONST_VAR float SDL::k2Rinv1GeVf = (2.99792458e-3 * 3.8) / 2;
CUDA_CONST_VAR float SDL::sinAlphaMax = 0.95;
CUDA_CONST_VAR float SDL::ptCut = 1.0;
CUDA_CONST_VAR float SDL::deltaZLum = 15.0;
CUDA_CONST_VAR float SDL::pixelPSZpitch = 0.15;
CUDA_CONST_VAR float SDL::strip2SZpitch = 5.0;

//FIXME:new memory locations for the pixel MDs
void SDL::createMDsInUnifiedMemory(struct miniDoublets& mdsInGPU, unsigned int maxMDsPerModule, unsigned int nModules, unsigned int maxPixelMDs)
{
    unsigned int nMemoryLocations = maxMDsPerModule * (nModules - 1) + maxPixelMDs;
#ifdef CACHE_ALLOC
    cudaStream_t stream=0;
    mdsInGPU.hitIndices = (unsigned int*)cms::cuda::allocate_managed(nMemoryLocations * 3 * sizeof(unsigned int), stream);
    mdsInGPU.pixelModuleFlag = (short*)cms::cuda::allocate_managed(nMemoryLocations*sizeof(short),stream);
    mdsInGPU.nMDs = (unsigned int*)cms::cuda::allocate_managed(nModules*sizeof(unsigned int),stream); //should this be nMemoryLocations or nModules as before?
    mdsInGPU.dphichanges = (float*)cms::cuda::allocate_managed(nMemoryLocations*9*sizeof(float),stream);
#else
    cudaMallocManaged(&mdsInGPU.hitIndices, nMemoryLocations * 3 * sizeof(unsigned int));
    cudaMallocManaged(&mdsInGPU.pixelModuleFlag, nMemoryLocations * sizeof(short));
    cudaMallocManaged(&mdsInGPU.dphichanges, nMemoryLocations * 9 * sizeof(float));
    cudaMallocManaged(&mdsInGPU.nMDs, nModules * sizeof(unsigned int));
#endif
    mdsInGPU.moduleIndices = mdsInGPU.hitIndices + nMemoryLocations * 2 ;
    mdsInGPU.dzs  = mdsInGPU.dphichanges + nMemoryLocations;
    mdsInGPU.dphis  = mdsInGPU.dphichanges + 2*nMemoryLocations;
    mdsInGPU.shiftedXs  = mdsInGPU.dphichanges + 3*nMemoryLocations;
    mdsInGPU.shiftedYs  = mdsInGPU.dphichanges + 4*nMemoryLocations;
    mdsInGPU.shiftedZs  = mdsInGPU.dphichanges + 5*nMemoryLocations;
    mdsInGPU.noShiftedDzs  = mdsInGPU.dphichanges + 6*nMemoryLocations;
    mdsInGPU.noShiftedDphis  = mdsInGPU.dphichanges + 7*nMemoryLocations;
    mdsInGPU.noShiftedDphiChanges  = mdsInGPU.dphichanges + 8*nMemoryLocations;
#pragma omp parallel for default(shared)
    for(size_t i = 0; i< nModules; i++)
    {
        mdsInGPU.nMDs[i] = 0;
    }
}


//FIXME:Add memory locations for the pixel MDs here!
void SDL::createMDsInExplicitMemory(struct miniDoublets& mdsInGPU, unsigned int maxMDsPerModule, unsigned int nModules, unsigned int maxPixelMDs)
{

    unsigned int nMemoryLocations = maxMDsPerModule * (nModules - 1) + maxPixelMDs;
#ifdef CACHE_ALLOC
    cudaStream_t stream=0;
    int dev;
    cudaGetDevice(&dev);
    mdsInGPU.hitIndices = (unsigned int*)cms::cuda::allocate_device(dev,nMemoryLocations * 3 * sizeof(unsigned int), stream);
    mdsInGPU.pixelModuleFlag = (short*)cms::cuda::allocate_device(dev,nMemoryLocations*sizeof(short),stream);
    mdsInGPU.dphichanges = (float*)cms::cuda::allocate_device(dev,nMemoryLocations*9*sizeof(float),stream);
    mdsInGPU.nMDs = (unsigned int*)cms::cuda::allocate_device(dev,nModules*sizeof(unsigned int),stream);

#else
    cudaMalloc(&mdsInGPU.hitIndices, nMemoryLocations * 3 * sizeof(unsigned int));
    cudaMalloc(&mdsInGPU.pixelModuleFlag, nMemoryLocations * sizeof(short));
    cudaMalloc(&mdsInGPU.dphichanges, nMemoryLocations *9* sizeof(float));
    cudaMalloc(&mdsInGPU.nMDs, nModules * sizeof(unsigned int)); 
#endif
    cudaMemset(mdsInGPU.nMDs,0,nModules *sizeof(unsigned int));
    mdsInGPU.moduleIndices = mdsInGPU.hitIndices + nMemoryLocations * 2 ;
    mdsInGPU.dzs  = mdsInGPU.dphichanges + nMemoryLocations;
    mdsInGPU.dphis  = mdsInGPU.dphichanges + 2*nMemoryLocations;
    mdsInGPU.shiftedXs  = mdsInGPU.dphichanges + 3*nMemoryLocations;
    mdsInGPU.shiftedYs  = mdsInGPU.dphichanges + 4*nMemoryLocations;
    mdsInGPU.shiftedZs  = mdsInGPU.dphichanges + 5*nMemoryLocations;
    mdsInGPU.noShiftedDzs  = mdsInGPU.dphichanges + 6*nMemoryLocations;
    mdsInGPU.noShiftedDphis  = mdsInGPU.dphichanges + 7*nMemoryLocations;
    mdsInGPU.noShiftedDphiChanges  = mdsInGPU.dphichanges + 8*nMemoryLocations;

}

/*__host__*/ __device__ void SDL::addMDToMemory(struct miniDoublets& mdsInGPU, struct hits& hitsInGPU, struct modules& modulesInGPU, unsigned int lowerHitIdx, unsigned int upperHitIdx, unsigned int lowerModuleIdx, float dz, float dPhi, float dPhiChange, float shiftedX, float shiftedY, float shiftedZ, float noShiftedDz, float noShiftedDphi, float noShiftedDPhiChange, unsigned int idx)
{
    //the index into which this MD needs to be written will be computed in the kernel
    //nMDs variable will be incremented in the kernel, no need to worry about that here
    
    mdsInGPU.hitIndices[idx * 2] = lowerHitIdx;
    mdsInGPU.hitIndices[idx * 2 + 1] = upperHitIdx;
    mdsInGPU.moduleIndices[idx] = lowerModuleIdx;
    if(modulesInGPU.moduleType[lowerModuleIdx] == PS)
    {
        if(modulesInGPU.moduleLayerType[lowerModuleIdx] == Pixel)
        {
            mdsInGPU.pixelModuleFlag[idx] = 0;
        }
        else
        {
            mdsInGPU.pixelModuleFlag[idx] = 1;
        }
    }
    else
    {
        mdsInGPU.pixelModuleFlag[idx] = -1;
    }

    mdsInGPU.dphichanges[idx] = dPhiChange;

    mdsInGPU.dphis[idx] = dPhi;
    mdsInGPU.dzs[idx] = dz;
    mdsInGPU.shiftedXs[idx] = shiftedX;
    mdsInGPU.shiftedYs[idx] = shiftedY;
    mdsInGPU.shiftedZs[idx] = shiftedZ;

    mdsInGPU.noShiftedDzs[idx] = noShiftedDz;
    mdsInGPU.noShiftedDphis[idx] = noShiftedDphi;
    mdsInGPU.noShiftedDphiChanges[idx] = noShiftedDPhiChange;
}

__device__ bool SDL::runMiniDoubletDefaultAlgoBarrel(struct modules& modulesInGPU, struct hits& hitsInGPU, unsigned int lowerModuleIndex, unsigned int lowerHitIndex, unsigned int upperHitIndex, float& dz, float& dPhi, float& dPhiChange, float& shiftedX, float& shiftedY, float& shiftedZ, float& noshiftedDz, float& noShiftedDphi, float& noShiftedDphiChange)
{
    float xLower = hitsInGPU.xs[lowerHitIndex];
    float yLower = hitsInGPU.ys[lowerHitIndex];
    float zLower = hitsInGPU.zs[lowerHitIndex];

    float xUpper = hitsInGPU.xs[upperHitIndex];
    float yUpper = hitsInGPU.ys[upperHitIndex];
    float zUpper = hitsInGPU.zs[upperHitIndex];

    bool pass = true; 
    dz = zLower - zUpper;     
    const float dzCut = modulesInGPU.moduleType[lowerModuleIndex] == PS ? 2.f : 10.f;
    const float sign = ((dz > 0) - (dz < 0)) * ((hitsInGPU.zs[lowerHitIndex] > 0) - (hitsInGPU.zs[lowerHitIndex] < 0));
    const float invertedcrossercut = (fabsf(dz) > 2) * sign;


    //cut convention - when a particular cut fails, the pass variable goes to false
    //but all cuts will be checked even if a previous cut has failed, this is
    //to prevent thread divergence

    if (not (fabsf(dz) < dzCut and invertedcrossercut <= 0)) // Adding inverted crosser rejection
    {
        pass = false;
    }

    float miniCut = 0;

//    float miniCutLower = dPhiThreshold(hitsInGPU, modulesInGPU, lowerHitIndex, lowerModuleIndex);
//    float miniCutUpper = dPhiThreshold(hitsInGPU, modulesInGPU, upperHitIndex, lowerModuleIndex);

    if (modulesInGPU.moduleLayerType[lowerModuleIndex] == Pixel)
    {
        miniCut = dPhiThreshold(hitsInGPU, modulesInGPU, lowerHitIndex, lowerModuleIndex); 
    }
    else
    {
        miniCut = dPhiThreshold(hitsInGPU, modulesInGPU, upperHitIndex, lowerModuleIndex);
 
    }

    // Cut #2: dphi difference
    // Ref to original code: https://github.com/slava77/cms-tkph2-ntuple/blob/184d2325147e6930030d3d1f780136bc2dd29ce6/doubletAnalysis.C#L3085
    float xn = 0, yn = 0;// , zn = 0;
    float shiftedRt;
    if (modulesInGPU.sides[lowerModuleIndex] != Center) // If barrel and not center it is tilted
    {
        // Shift the hits and calculate new xn, yn position
        float shiftedCoords[3];
        shiftStripHits(modulesInGPU, hitsInGPU, lowerModuleIndex, lowerHitIndex, upperHitIndex, shiftedCoords);
        xn = shiftedCoords[0];
        yn = shiftedCoords[1];
//        zn = shiftedCoords[2];

        // Lower or the upper hit needs to be modified depending on which one was actually shifted
        if (modulesInGPU.moduleLayerType[lowerModuleIndex] == Pixel)
        {
            shiftedX = xn;
            shiftedY = yn;
            shiftedZ = zUpper;
            shiftedRt = sqrt(xn * xn + yn * yn);

            dPhi = deltaPhi(xLower,yLower,zLower,shiftedX, shiftedY, shiftedZ); //function from Hit.cu
            noShiftedDphi = deltaPhi(xLower, yLower, zLower, xUpper, yUpper, zUpper);
        }
        else
        {
            shiftedX = xn;
            shiftedY = yn;
            shiftedZ = zLower;
            shiftedRt = sqrt(xn * xn + yn * yn);
            dPhi = deltaPhi(shiftedX, shiftedY, shiftedZ, xUpper, yUpper, zUpper);
            noShiftedDphi = deltaPhi(xLower,yLower,zLower,xUpper,yUpper,zUpper);

        }
    }
    else
    {
        dPhi = deltaPhi(xLower, yLower, zLower, xUpper, yUpper, zUpper);
        noShiftedDphi = dPhi;
    }


    if (not (fabsf(dPhi) < miniCut)) // If cut fails continue
    {
        pass = false;
    }

    // Cut #3: The dphi change going from lower Hit to upper Hit
    // Ref to original code: https://github.com/slava77/cms-tkph2-ntuple/blob/184d2325147e6930030d3d1f780136bc2dd29ce6/doubletAnalysis.C#L3076
    if (modulesInGPU.sides[lowerModuleIndex]!= Center)
    {
        // When it is tilted, use the new shifted positions
        if (modulesInGPU.moduleLayerType[lowerModuleIndex] == Pixel)
        {
            // dPhi Change should be calculated so that the upper hit has higher rt.
            // In principle, this kind of check rt_lower < rt_upper should not be necessary because the hit shifting should have taken care of this.
            // (i.e. the strip hit is shifted to be aligned in the line of sight from interaction point to pixel hit of PS module guaranteeing rt ordering)
            // But I still placed this check for safety. (TODO: After cheking explicitly if not needed remove later?)
            // setDeltaPhiChange(lowerHit.rt() < upperHitMod.rt() ? lowerHit.deltaPhiChange(upperHitMod) : upperHitMod.deltaPhiChange(lowerHit));


            dPhiChange = (hitsInGPU.rts[lowerHitIndex] < shiftedRt) ? deltaPhiChange(xLower, yLower, zLower, shiftedX, shiftedY, shiftedZ) : deltaPhiChange(shiftedX, shiftedY, shiftedZ, xLower, yLower, zLower); 
            noShiftedDphiChange = hitsInGPU.rts[lowerHitIndex] < hitsInGPU.rts[upperHitIndex] ? deltaPhiChange(xLower,yLower, zLower, xUpper, yUpper, zUpper) : deltaPhiChange(xUpper, yUpper, zUpper, xLower, yLower, zLower);
        }
        else
        {
            // dPhi Change should be calculated so that the upper hit has higher rt.
            // In principle, this kind of check rt_lower < rt_upper should not be necessary because the hit shifting should have taken care of this.
            // (i.e. the strip hit is shifted to be aligned in the line of sight from interaction point to pixel hit of PS module guaranteeing rt ordering)
            // But I still placed this check for safety. (TODO: After cheking explicitly if not needed remove later?)

            dPhiChange = (shiftedRt < hitsInGPU.rts[upperHitIndex]) ? deltaPhiChange(shiftedX, shiftedY, shiftedZ, xUpper, yUpper, zUpper) : deltaPhiChange(xUpper, yUpper, zUpper, shiftedX, shiftedY, shiftedZ);
            noShiftedDphiChange = hitsInGPU.rts[lowerHitIndex] < hitsInGPU.rts[upperHitIndex] ? deltaPhiChange(xLower,yLower, zLower, xUpper, yUpper, zUpper) : deltaPhiChange(xUpper, yUpper, zUpper, xLower, yLower, zLower);
        }
    }
    else
    {
        // When it is flat lying module, whichever is the lowerSide will always have rt lower
        dPhiChange = deltaPhiChange(xLower, yLower, zLower, xUpper, yUpper, zUpper);
        noShiftedDphiChange = dPhiChange;
    }

    if (not (fabsf(dPhiChange) < miniCut)) // If cut fails continue
    {
        pass = false;
    }

    return pass;
}

__device__ bool SDL::runMiniDoubletDefaultAlgoEndcap(struct modules& modulesInGPU, struct hits& hitsInGPU, unsigned int lowerModuleIndex, unsigned int lowerHitIndex, unsigned int upperHitIndex, float& dz, float& dPhi, float& dPhiChange, float& shiftedX, float& shiftedY, float& shiftedZ, float& noshiftedDz, float& noShiftedDphi, float& noShiftedDphichange)
{
    float xLower = hitsInGPU.xs[lowerHitIndex];
    float yLower = hitsInGPU.ys[lowerHitIndex];
    float zLower = hitsInGPU.zs[lowerHitIndex];

    float xUpper = hitsInGPU.xs[upperHitIndex];
    float yUpper = hitsInGPU.ys[upperHitIndex];
    float zUpper = hitsInGPU.zs[upperHitIndex];

    bool pass = true; 

    // There are series of cuts that applies to mini-doublet in a "endcap" region

    // Cut #1 : dz cut. The dz difference can't be larger than 1cm. (max separation is 4mm for modules in the endcap)
    // Ref to original code: https://github.com/slava77/cms-tkph2-ntuple/blob/184d2325147e6930030d3d1f780136bc2dd29ce6/doubletAnalysis.C#L3093
    // For PS module in case when it is tilted a different dz (after the strip hit shift) is calculated later.
    // This is because the 10.f cut is meant more for sanity check (most will pass this cut anyway) (TODO: Maybe revisit this cut later?)

    dz = zLower - zUpper; // Not const since later it might change depending on the type of module

    const float dzCut = ((modulesInGPU.sides[lowerModuleIndex] == Endcap) ?  1.f : 10.f);
    if (not (fabsf(dz) < dzCut)) // If cut fails continue
    {
        pass = false;
    }

    // Cut #2 : drt cut. The dz difference can't be larger than 1cm. (max separation is 4mm for modules in the endcap)
    // Ref to original code: https://github.com/slava77/cms-tkph2-ntuple/blob/184d2325147e6930030d3d1f780136bc2dd29ce6/doubletAnalysis.C#L3100
    const float drtCut = modulesInGPU.moduleType[lowerModuleIndex] == PS ? 2.f : 10.f;
    float drt = hitsInGPU.rts[lowerHitIndex] - hitsInGPU.rts[upperHitIndex];
    if (not (fabs(drt) < drtCut)) // If cut fails continue
    {
        pass = false;
    }

    // The new scheme shifts strip hits to be "aligned" along the line of sight from interaction point to the pixel hit (if it is PS modules)
    float xn = 0, yn = 0, zn = 0;

    float shiftedCoords[3];
    shiftStripHits(modulesInGPU, hitsInGPU, lowerModuleIndex, lowerHitIndex, upperHitIndex, shiftedCoords);

    xn = shiftedCoords[0];
    yn = shiftedCoords[1];
    zn = shiftedCoords[2];

    if (modulesInGPU.moduleType[lowerModuleIndex] == PS)
    {
        // Appropriate lower or upper hit is modified after checking which one was actually shifted
        if (modulesInGPU.moduleLayerType[lowerModuleIndex] == Pixel)
        {
            // SDL::Hit upperHitMod(upperHit);
            // upperHitMod.setXYZ(xn, yn, upperHit.z());
            // setDeltaPhi(lowerHit.deltaPhi(upperHitMod));
            shiftedX = xn;
            shiftedY = yn;
            shiftedZ = zUpper;
            dPhi = deltaPhi(xLower, yLower, zLower, shiftedX, shiftedY, shiftedZ);
            noShiftedDphi = deltaPhi(xLower, yLower, zLower, xUpper, yUpper, zUpper);
        }
        else
        {
            // SDL::Hit lowerHitMod(lowerHit);
            // lowerHitMod.setXYZ(xn, yn, lowerHit.z());
            // setDeltaPhi(lowerHitMod.deltaPhi(upperHit));
            shiftedX = xn;
            shiftedY = yn;
            shiftedZ = zLower;
            dPhi = deltaPhi(shiftedX, shiftedY, shiftedZ, xUpper, yUpper, zUpper);
            noShiftedDphi = deltaPhi(xLower, yLower, zLower, xUpper, yUpper, zUpper);
        }
    }
    else
    {
        shiftedX = xn;
        shiftedY = yn;
        shiftedZ = zUpper;
        dPhi = deltaPhi(xLower, yLower, zLower, xn, yn, zUpper);
        noShiftedDphi = deltaPhi(xLower, yLower, zLower, xUpper, yUpper, zUpper);
    }

    // dz needs to change if it is a PS module where the strip hits are shifted in order to properly account for the case when a tilted module falls under "endcap logic"
    // if it was an endcap it will have zero effect
    if (modulesInGPU.moduleType[lowerModuleIndex] == PS)
    {
        if (modulesInGPU.moduleLayerType[lowerModuleIndex] == Pixel)
        {
            dz = zLower - zn;
        }
        else
        {
            dz = zUpper - zn;
        }
    }

    float miniCut = 0;
    if(modulesInGPU.moduleLayerType[lowerModuleIndex] == Pixel)
    {
        miniCut = dPhiThreshold(hitsInGPU, modulesInGPU, lowerHitIndex, lowerModuleIndex,dPhi, dz);
    }
    else
    {
        miniCut = dPhiThreshold(hitsInGPU, modulesInGPU, upperHitIndex, lowerModuleIndex, dPhi, dz);
    }

    if (not (fabsf(dPhi) < miniCut)) // If cut fails continue
    {
        pass = false;
    }

    // Cut #4: Another cut on the dphi after some modification
    // Ref to original code: https://github.com/slava77/cms-tkph2-ntuple/blob/184d2325147e6930030d3d1f780136bc2dd29ce6/doubletAnalysis.C#L3119-L3124

    
    float dzFrac = fabsf(dz) / fabsf(zLower);
    dPhiChange = dPhi / dzFrac * (1.f + dzFrac);
    noShiftedDphichange = noShiftedDphi / dzFrac * (1.f + dzFrac);
    if (not (fabsf(dPhiChange) < miniCut)) // If cut fails continue
    {
        pass = false;
    }

    return pass;
}

__device__ bool SDL::runMiniDoubletDefaultAlgo(struct modules& modulesInGPU, struct hits& hitsInGPU, unsigned int lowerModuleIndex, unsigned int lowerHitIndex, unsigned int upperHitIndex, float& dz, float& dPhi, float& dPhiChange, float& shiftedX, float& shiftedY, float& shiftedZ, float& noShiftedDz, float& noShiftedDphi, float& noShiftedDphiChange)
{
   bool pass;
   if(modulesInGPU.subdets[lowerModuleIndex] == Barrel)
   {
        pass = runMiniDoubletDefaultAlgoBarrel(modulesInGPU, hitsInGPU, lowerModuleIndex, lowerHitIndex, upperHitIndex, dz, dPhi, dPhiChange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange);
   } 
   else
   {
       pass = runMiniDoubletDefaultAlgoEndcap(modulesInGPU, hitsInGPU, lowerModuleIndex, lowerHitIndex, upperHitIndex, dz, dPhi, dPhiChange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange);

   }
   return pass;
}

__device__ float SDL::dPhiThreshold(struct hits& hitsInGPU, struct modules& modulesInGPU, unsigned int hitIndex, unsigned int moduleIndex, float dPhi, float dz)
{
    // =================================================================
    // Various constants
    // =================================================================
    // const float ptCut = PTCUT;
    // const float sinAlphaMax = 0.95;
    //mean of the horizontal layer position in y; treat this as R below
/*    __device__ __constant__ float miniRminMeanBarrel[] = {21.8, 34.6, 49.6, 67.4, 87.6, 106.8}; // TODO: Update this with newest geometry
    __device__ __constant__ float miniRminMeanEndcap[] = {131.4, 156.2, 185.6, 220.3, 261.5};// use z for endcaps // TODO: Update this with newest geometry*/

    // =================================================================
    // Computing some components that make up the cut threshold
    // =================================================================

    float rt = hitsInGPU.rts[hitIndex];
    unsigned int iL = modulesInGPU.layers[moduleIndex] - 1;
    const float miniSlope = asinf(min(rt * k2Rinv1GeVf / ptCut, sinAlphaMax));
    const float rLayNominal = ((modulesInGPU.subdets[moduleIndex]== Barrel) ? miniRminMeanBarrel[iL] : miniRminMeanEndcap[iL]);
    const float miniPVoff = 0.1 / rLayNominal;
    const float miniMuls = ((modulesInGPU.subdets[moduleIndex] == Barrel) ? miniMulsPtScaleBarrel[iL] * 3.f / ptCut : miniMulsPtScaleEndcap[iL] * 3.f / ptCut);
    const bool isTilted = modulesInGPU.subdets[moduleIndex] == Barrel and modulesInGPU.sides[moduleIndex] != Center;
    //the lower module is sent in irrespective of its layer type. We need to fetch the drdz properly

    float drdz;
    if(isTilted)
    {
        if(modulesInGPU.moduleType[moduleIndex] == PS and modulesInGPU.moduleLayerType[moduleIndex] == Strip)
        {
            drdz = modulesInGPU.drdzs[moduleIndex];
        }
        else
        {
            drdz = modulesInGPU.drdzs[modulesInGPU.partnerModuleIndex(moduleIndex)];
        }  
    }
    else
    {
        drdz = 0;
    }
    const float miniTilt = ((isTilted) ? 0.5f * pixelPSZpitch * drdz / sqrt(1.f + drdz * drdz) / moduleGapSize(modulesInGPU,moduleIndex) : 0);

    // Compute luminous region requirement for endcap
    const float miniLum = fabsf(dPhi * deltaZLum/dz); // Balaji's new error
    // const float miniLum = abs(deltaZLum / lowerHit.z()); // Old error


    // =================================================================
    // Return the threshold value
    // =================================================================
    // Following condition is met if the module is central and flatly lying
    if (modulesInGPU.subdets[moduleIndex] == Barrel and modulesInGPU.sides[moduleIndex] == Center)
    {
        return miniSlope + sqrt(miniMuls * miniMuls + miniPVoff * miniPVoff);
    }
    // Following condition is met if the module is central and tilted
    else if (modulesInGPU.subdets[moduleIndex] == Barrel and modulesInGPU.sides[moduleIndex] != Center) //all types of tilted modules
    {
        return miniSlope + sqrt(miniMuls * miniMuls + miniPVoff * miniPVoff + miniTilt * miniTilt * miniSlope * miniSlope);
    }
    // If not barrel, it is Endcap
    else
    {
        return miniSlope + sqrt(miniMuls * miniMuls + miniPVoff * miniPVoff + miniLum * miniLum);
    }

}

__device__ inline float SDL::isTighterTiltedModules(struct modules& modulesInGPU, unsigned int moduleIndex)
{
    // The "tighter" tilted modules are the subset of tilted modules that have smaller spacing
    // This is the same as what was previously considered as"isNormalTiltedModules"
    // See Figure 9.1 of https://cds.cern.ch/record/2272264/files/CMS-TDR-014.pdf
    short subdet = modulesInGPU.subdets[moduleIndex];
    short layer = modulesInGPU.layers[moduleIndex];
    short side = modulesInGPU.sides[moduleIndex];
    short rod = modulesInGPU.rods[moduleIndex];

    if (
           (subdet == Barrel and side != Center and layer== 3)
           or (subdet == Barrel and side == NegZ and layer == 2 and rod > 5)
           or (subdet == Barrel and side == PosZ and layer == 2 and rod < 8)
           or (subdet == Barrel and side == NegZ and layer == 1 and rod > 9)
           or (subdet == Barrel and side == PosZ and layer == 1 and rod < 4)
       )
        return true;
    else
        return false;

}



__device__ float SDL::moduleGapSize(struct modules& modulesInGPU, unsigned int moduleIndex)
{
    float miniDeltaTilted[3] = {0.26, 0.26, 0.26};
    float miniDeltaFlat[6] ={0.26, 0.16, 0.16, 0.18, 0.18, 0.18};
    float miniDeltaLooseTilted[3] = {0.4,0.4,0.4};
    float miniDeltaEndcap[5][15];

    for (size_t i = 0; i < 5; i++)
    {
        for (size_t j = 0; j < 15; j++)
        {
            if (i == 0 || i == 1)
            {
                if (j < 10)
                {
                    miniDeltaEndcap[i][j] = 0.4;
                }
                else
                {
                    miniDeltaEndcap[i][j] = 0.18;
                }
            }
            else if (i == 2 || i == 3)
            {
                if (j < 8)
                {
                    miniDeltaEndcap[i][j] = 0.4;
                }
                else
                {
                    miniDeltaEndcap[i][j]  = 0.18;
                }
            }
            else
            {
                if (j < 9)
                {
                    miniDeltaEndcap[i][j] = 0.4;
                }
                else
                {
                    miniDeltaEndcap[i][j] = 0.18;
                }
            }
        }
    }


    unsigned int iL = modulesInGPU.layers[moduleIndex]-1;
    unsigned int iR = modulesInGPU.rings[moduleIndex] - 1;
    short subdet = modulesInGPU.subdets[moduleIndex];
    short side = modulesInGPU.sides[moduleIndex];

    float moduleSeparation = 0;

    if (subdet == Barrel and side == Center)
    {
        moduleSeparation = miniDeltaFlat[iL];
    }
    else if (isTighterTiltedModules(modulesInGPU, moduleIndex))
    {
        moduleSeparation = miniDeltaTilted[iL];
    }
    else if (subdet == Endcap)
    {
        moduleSeparation = miniDeltaEndcap[iL][iR];
    }
    else //Loose tilted modules
    {
        moduleSeparation = miniDeltaLooseTilted[iL];
    }

    return moduleSeparation;
}

__device__ void SDL::shiftStripHits(struct modules& modulesInGPU, struct hits& hitsInGPU, unsigned int lowerModuleIndex, unsigned int lowerHitIndex, unsigned int upperHitIndex, float* shiftedCoords)
{

    // This is the strip shift scheme that is explained in http://uaf-10.t2.ucsd.edu/~phchang/talks/PhilipChang20190607_SDL_Update.pdf (see backup slides)
    // The main feature of this shifting is that the strip hits are shifted to be "aligned" in the line of sight from interaction point to the the pixel hit.
    // (since pixel hit is well defined in 3-d)
    // The strip hit is shifted along the strip detector to be placed in a guessed position where we think they would have actually crossed
    // The size of the radial direction shift due to module separation gap is computed in "radial" size, while the shift is done along the actual strip orientation
    // This means that there may be very very subtle edge effects coming from whether the strip hit is center of the module or the at the edge of the module
    // But this should be relatively minor effect

    // dependent variables for this if statement
    // lowerModule
    // lowerHit
    // upperHit
    // SDL::endcapGeometry
    // SDL::tiltedGeometry

    // Some variables relevant to the function
    float xp; // pixel x (pixel hit x)
    float yp; // pixel y (pixel hit y)
    float xa; // "anchor" x (the anchor position on the strip module plane from pixel hit)
    float ya; // "anchor" y (the anchor position on the strip module plane from pixel hit)
    float xo; // old x (before the strip hit is moved up or down)
    float yo; // old y (before the strip hit is moved up or down)
    float xn; // new x (after the strip hit is moved up or down)
    float yn; // new y (after the strip hit is moved up or down)
    float abszn; // new z in absolute value
    float zn; // new z with the sign (+/-) accounted
    float angleA; // in r-z plane the theta of the pixel hit in polar coordinate is the angleA
    float angleB; // this is the angle of tilted module in r-z plane ("drdz"), for endcap this is 90 degrees
    bool isEndcap; // If endcap, drdz = infinity
    unsigned int pixelHitIndex; // Pointer to the pixel hit
    unsigned int stripHitIndex; // Pointer to the strip hit
    float moduleSeparation;
    float drprime; // The radial shift size in x-y plane projection
    float drprime_x; // x-component of drprime
    float drprime_y; // y-component of drprime
    float slope; // The slope of the possible strip hits for a given module in x-y plane
    float absArctanSlope;
    float angleM; // the angle M is the angle of rotation of the module in x-y plane if the possible strip hits are along the x-axis, then angleM = 0, and if the possible strip hits are along y-axis angleM = 90 degrees
    float absdzprime; // The distance between the two points after shifting
    float drdz_;
    unsigned int upperModuleIndex = modulesInGPU.partnerModuleIndex(lowerModuleIndex);
    // Assign hit pointers based on their hit type
    if (modulesInGPU.moduleType[lowerModuleIndex] == PS)
    {
        if (modulesInGPU.moduleLayerType[lowerModuleIndex]== Pixel)
        {
            pixelHitIndex = lowerHitIndex;
            stripHitIndex = upperHitIndex;
        }
        else
        {
            pixelHitIndex = upperHitIndex;
            stripHitIndex = lowerHitIndex;
        }
    }
    else // if (lowerModule.moduleType() == SDL::Module::TwoS) // If it is a TwoS module (if this is called likely an endcap module) then anchor the inner hit and shift the outer hit
    {
        pixelHitIndex = lowerHitIndex; // Even though in this case the "pixelHitPtr" is really just a strip hit, we pretend it is the anchoring pixel hit
        stripHitIndex = upperHitIndex;
    }

    // If it is endcap some of the math gets simplified (and also computers don't like infinities)
    isEndcap = modulesInGPU.subdets[lowerModuleIndex]== Endcap;

    // NOTE: TODO: Keep in mind that the sin(atan) function can be simplifed to something like x / sqrt(1 + x^2) and similar for cos
    // I am not sure how slow sin, atan, cos, functions are in c++. If x / sqrt(1 + x^2) are faster change this later to reduce arithmetic computation time

    // The pixel hit is used to compute the angleA which is the theta in polar coordinate
    // angleA = atanf(pixelHitPtr->rt() / pixelHitPtr->z() + (pixelHitPtr->z() < 0 ? M_PI : 0)); // Shift by pi if the z is negative so that the value of the angleA stays between 0 to pi and not -pi/2 to pi/2

    angleA = fabsf(atanf(hitsInGPU.rts[pixelHitIndex] / hitsInGPU.zs[pixelHitIndex]));
    // angleB = isEndcap ? M_PI / 2. : -atanf(tiltedGeometry.getDrDz(detid) * (lowerModule.side() == SDL::Module::PosZ ? -1 : 1)); // The tilt module on the postive z-axis has negative drdz slope in r-z plane and vice versa
    if(modulesInGPU.moduleType[lowerModuleIndex] == PS and modulesInGPU.moduleLayerType[upperModuleIndex] == Strip)
    {
        drdz_ = modulesInGPU.drdzs[upperModuleIndex];
        slope = modulesInGPU.slopes[upperModuleIndex];
    }
    else
    {
        drdz_ = modulesInGPU.drdzs[lowerModuleIndex];
        slope = modulesInGPU.slopes[lowerModuleIndex];
    }
    angleB = ((isEndcap) ? M_PI / 2. : atan(drdz_)); // The tilt module on the postive z-axis has negative drdz slope in r-z plane and vice versa


    moduleSeparation = moduleGapSize(modulesInGPU, lowerModuleIndex);

    // Sign flips if the pixel is later layer
    if (modulesInGPU.moduleType[lowerModuleIndex] == PS and modulesInGPU.moduleLayerType[lowerModuleIndex] != Pixel)
    {
        moduleSeparation *= -1;
    }

    drprime = (moduleSeparation / std::sin(angleA + angleB)) * std::sin(angleA);
    
    // Compute arctan of the slope and take care of the slope = infinity case
    absArctanSlope = ((slope != SDL_INF) ? fabs(atanf(slope)) : M_PI / 2); // Since C++ can't represent infinity, SDL_INF = 123456789 was used to represent infinity in the data table

    // The pixel hit position
    xp = hitsInGPU.xs[pixelHitIndex];
    yp = hitsInGPU.ys[pixelHitIndex];

    // Depending on which quadrant the pixel hit lies, we define the angleM by shifting them slightly differently
    if (xp > 0 and yp > 0)
    {
        angleM = absArctanSlope;
    }
    else if (xp > 0 and yp < 0)
    {
        angleM = M_PI - absArctanSlope;
    }
    else if (xp < 0 and yp < 0)
    {
        angleM = M_PI + absArctanSlope;
    }
    else // if (xp < 0 and yp > 0)
    {
        angleM = 2 * M_PI - absArctanSlope;
    }

    // Then since the angleM sign is taken care of properly
    drprime_x = drprime * std::sin(angleM);
    drprime_y = drprime * std::cos(angleM);

    // The new anchor position is
    xa = xp + drprime_x;
    ya = yp + drprime_y;

    // The original strip hit position
    xo = hitsInGPU.xs[stripHitIndex];
    yo = hitsInGPU.ys[stripHitIndex];

    // Compute the new strip hit position (if the slope vaule is in special condition take care of the exceptions)
    if (slope == SDL_INF) // Special value designated for tilted module when the slope is exactly infinity (module lying along y-axis)
    {
        xn = xa; // New x point is simply where the anchor is
        yn = yo; // No shift in y
    }
    else if (slope == 0)
    {
        xn = xo; // New x point is simply where the anchor is
        yn = ya; // No shift in y
    }
    else
    {
        xn = (slope * xa + (1.f / slope) * xo - ya + yo) / (slope + (1.f / slope)); // new xn
        yn = (xn - xa) * slope + ya; // new yn
    }

    // Computing new Z position
    absdzprime = fabsf(moduleSeparation / std::sin(angleA + angleB) * std::cos(angleA)); // module separation sign is for shifting in radial direction for z-axis direction take care of the sign later

    // Depending on which one as closer to the interactin point compute the new z wrt to the pixel properly
    if (modulesInGPU.moduleLayerType[lowerModuleIndex] == Pixel)
    {
        abszn = fabsf(hitsInGPU.zs[pixelHitIndex]) + absdzprime;
    }
    else
    {
        abszn = fabsf(hitsInGPU.zs[pixelHitIndex]) - absdzprime;
    }

    zn = abszn * ((hitsInGPU.zs[pixelHitIndex] > 0) ? 1 : -1); // Apply the sign of the zn


    shiftedCoords[0] = xn;
    shiftedCoords[1] = yn;
    shiftedCoords[2] = zn;
}

SDL::miniDoublets::miniDoublets()
{
    hitIndices = nullptr;
    moduleIndices = nullptr;
    pixelModuleFlag = nullptr;
    nMDs = nullptr;
    dphichanges = nullptr;

    dzs = nullptr;
    dphis = nullptr;

    shiftedXs = nullptr;
    shiftedYs = nullptr;
    shiftedZs = nullptr;
    noShiftedDzs = nullptr;
    noShiftedDphis = nullptr;
    noShiftedDphiChanges = nullptr;

}

SDL::miniDoublets::~miniDoublets()
{
}

void SDL::miniDoublets::freeMemoryCache()
{
#ifdef Explicit_MD
    int dev;
    cudaGetDevice(&dev);
    cms::cuda::free_device(dev,hitIndices);
    cms::cuda::free_device(dev,pixelModuleFlag);
    cms::cuda::free_device(dev,dphichanges);
//  #ifdef Full_Explicit
    cms::cuda::free_device(dev,nMDs);
//  #else
//    cms::cuda::free_managed(nMDs);
//  #endif
#else
    cms::cuda::free_managed(hitIndices);
    cms::cuda::free_managed(pixelModuleFlag);
    cms::cuda::free_managed(dphichanges);
    cms::cuda::free_managed(nMDs);
#endif
}


void SDL::miniDoublets::freeMemory()
{
    cudaFree(hitIndices);
    cudaFree(pixelModuleFlag);
    cudaFree(nMDs);
    cudaFree(dphichanges);
}

void SDL::printMD(struct miniDoublets& mdsInGPU, struct hits& hitsInGPU, SDL::modules& modulesInGPU, unsigned int mdIndex)
{
    std::cout<<std::endl;
    std::cout << "dz " << mdsInGPU.dzs[mdIndex] << std::endl;
    std::cout << "dphi " << mdsInGPU.dphis[mdIndex] << std::endl;
    std::cout << "dphinoshift " << mdsInGPU.noShiftedDphis[mdIndex] << std::endl;
    std::cout << "dphichange " << mdsInGPU.dphichanges[mdIndex] << std::endl;
    std::cout << "dphichangenoshift " << mdsInGPU.noShiftedDphiChanges[mdIndex] << std::endl;
    std::cout << std::endl;
    std::cout << "Lower Hit " << std::endl;
    std::cout << "------------------------------" << std::endl;
    unsigned int lowerHitIndex = mdsInGPU.hitIndices[mdIndex * 2];
    unsigned int upperHitIndex = mdsInGPU.hitIndices[mdIndex * 2  + 1];
    {
        IndentingOStreambuf indent(std::cout);
        printHit(hitsInGPU, modulesInGPU, lowerHitIndex);
    }
    std::cout << "Upper Hit " << std::endl;
    std::cout << "------------------------------" << std::endl;
    {
        IndentingOStreambuf indent(std::cout);
        printHit(hitsInGPU, modulesInGPU, upperHitIndex);
    }
}
