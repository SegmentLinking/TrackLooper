#ifdef __CUDACC__
#define CUDA_CONST_VAR __device__
#endif
# include "Triplet.cuh"
//#ifdef CACHE_ALLOC
#include "allocate.h"
//#endif

void SDL::createTripletsInUnifiedMemory(struct triplets& tripletsInGPU, unsigned int maxTriplets, unsigned int nLowerModules)
{
#ifdef CACHE_ALLOC
    cudaStream_t stream=0;
    tripletsInGPU.segmentIndices = (unsigned int*)cms::cuda::allocate_managed(maxTriplets * nLowerModules * sizeof(unsigned int) *5,stream);
    tripletsInGPU.nTriplets = (unsigned int*)cms::cuda::allocate_managed(nLowerModules * sizeof(unsigned int),stream);
    tripletsInGPU.zOut = (float*)cms::cuda::allocate_managed(maxTriplets * nLowerModules * sizeof(float) *6,stream);
#else
    cudaMallocManaged(&tripletsInGPU.segmentIndices, 5 * maxTriplets * nLowerModules * sizeof(unsigned int));
    cudaMallocManaged(&tripletsInGPU.nTriplets, nLowerModules * sizeof(unsigned int));
    cudaMallocManaged(&tripletsInGPU.zOut, maxTriplets * nLowerModules * 6*sizeof(unsigned int));
#endif
    tripletsInGPU.lowerModuleIndices = tripletsInGPU.segmentIndices + nLowerModules * maxTriplets *2;

    tripletsInGPU.rtOut = tripletsInGPU.zOut + nLowerModules * maxTriplets;
    tripletsInGPU.deltaPhiPos = tripletsInGPU.zOut + nLowerModules * maxTriplets *2;
    tripletsInGPU.deltaPhi = tripletsInGPU.zOut + nLowerModules * maxTriplets *3;
    tripletsInGPU.betaIn = tripletsInGPU.zOut + nLowerModules * maxTriplets *4;
    tripletsInGPU.betaOut = tripletsInGPU.zOut + nLowerModules * maxTriplets *5;

#pragma omp parallel for
    for(size_t i = 0; i<nLowerModules;i++)
    {
        tripletsInGPU.nTriplets[i] = 0;
    }
}
void SDL::createTripletsInExplicitMemory(struct triplets& tripletsInGPU, unsigned int maxTriplets, unsigned int nLowerModules)
{
#ifdef CACHE_ALLOC
    cudaStream_t stream=0;
    int dev;
    cudaGetDevice(&dev);
    tripletsInGPU.segmentIndices = (unsigned int*)cms::cuda::allocate_device(dev,maxTriplets * nLowerModules * sizeof(unsigned int) *5,stream);
    tripletsInGPU.zOut = (float*)cms::cuda::allocate_device(dev,maxTriplets * nLowerModules * sizeof(float) *6,stream);
  #ifdef Full_Explicit
    tripletsInGPU.nTriplets = (unsigned int*)cms::cuda::allocate_device(dev,nLowerModules * sizeof(unsigned int),stream);
    cudaMemset(tripletsInGPU.nTriplets,0,nLowerModules * sizeof(unsigned int));
  #else
    tripletsInGPU.nTriplets = (unsigned int*)cms::cuda::allocate_managed(nLowerModules * sizeof(unsigned int),stream);
  #endif

#else
    cudaMalloc(&tripletsInGPU.segmentIndices, 5 * maxTriplets * nLowerModules * sizeof(unsigned int));
    cudaMalloc(&tripletsInGPU.zOut, maxTriplets * nLowerModules * 6* sizeof(unsigned int));
  #ifdef Full_Explicit
    cudaMalloc(&tripletsInGPU.nTriplets, nLowerModules * sizeof(unsigned int));
    cudaMemset(tripletsInGPU.nTriplets,0,nLowerModules * sizeof(unsigned int));
  #else
    cudaMallocManaged(&tripletsInGPU.nTriplets, nLowerModules * sizeof(unsigned int));
  #endif
#endif
    tripletsInGPU.lowerModuleIndices = tripletsInGPU.segmentIndices + nLowerModules * maxTriplets *2;

    tripletsInGPU.rtOut = tripletsInGPU.zOut + nLowerModules * maxTriplets;
    tripletsInGPU.deltaPhiPos = tripletsInGPU.zOut + nLowerModules * maxTriplets *2;
    tripletsInGPU.deltaPhi = tripletsInGPU.zOut + nLowerModules * maxTriplets *3;
    tripletsInGPU.betaIn = tripletsInGPU.zOut + nLowerModules * maxTriplets *4;
    tripletsInGPU.betaOut = tripletsInGPU.zOut + nLowerModules * maxTriplets *5;

}

__device__ void SDL::addTripletToMemory(struct triplets& tripletsInGPU, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, unsigned int innerInnerLowerModuleIndex, unsigned int middleLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& deltaPhi, float& betaIn, float& betaOut, unsigned int tripletIndex)
{
   tripletsInGPU.segmentIndices[tripletIndex * 2] = innerSegmentIndex;
   tripletsInGPU.segmentIndices[tripletIndex * 2 + 1] = outerSegmentIndex;
   tripletsInGPU.lowerModuleIndices[tripletIndex * 3] = innerInnerLowerModuleIndex;
   tripletsInGPU.lowerModuleIndices[tripletIndex * 3 + 1] = middleLowerModuleIndex;
   tripletsInGPU.lowerModuleIndices[tripletIndex * 3 + 2] = outerOuterLowerModuleIndex;

   tripletsInGPU.zOut[tripletIndex] = zOut;
   tripletsInGPU.rtOut[tripletIndex] = rtOut;
   tripletsInGPU.deltaPhiPos[tripletIndex] = deltaPhiPos;
   tripletsInGPU.deltaPhi[tripletIndex] = deltaPhi;

   tripletsInGPU.betaIn[tripletIndex] = betaIn;
   tripletsInGPU.betaOut[tripletIndex] = betaOut;
}

SDL::triplets::triplets()
{
    segmentIndices = nullptr;
    lowerModuleIndices = nullptr;
    zOut = nullptr;
    rtOut = nullptr;

    deltaPhiPos = nullptr;
    deltaPhi = nullptr;
    betaIn = nullptr;
    betaOut = nullptr;
}

SDL::triplets::~triplets()
{
}

void SDL::triplets::freeMemoryCache()
{
#ifdef Explicit_Trips
    int dev;
    cudaGetDevice(&dev);
    cms::cuda::free_device(dev,segmentIndices);
    cms::cuda::free_device(dev,zOut);
  #ifdef Full_Explicit
    cms::cuda::free_device(dev,nTriplets);
  #else
    cms::cuda::free_managed(nTriplets);
  #endif
#else
    cms::cuda::free_managed(segmentIndices);
    cms::cuda::free_managed(zOut);
    cms::cuda::free_managed(nTriplets);
#endif
}
void SDL::triplets::freeMemory()
{
    cudaFree(segmentIndices);
    cudaFree(nTriplets);
    cudaFree(zOut);
}

__device__ bool SDL::runTripletDefaultAlgo(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, unsigned int innerInnerLowerModuleIndex, unsigned int middleLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& deltaPhi, float& betaIn, float& betaOut)
{
    bool pass = true;
    //check
    if(not(hasCommonMiniDoublet(segmentsInGPU, innerSegmentIndex, outerSegmentIndex)))
    {
        pass = false;
    }
    if(not(passPointingConstraint(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, middleLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, zOut, rtOut))) //fill arguments
    {
        pass = false;
    }
    //now check tracklet algo
    
    if(not(runTrackletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, middleLowerModuleIndex, middleLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut)))
    {
        pass = false;
    }

    return pass;
}

__device__ bool SDL::passPointingConstraint(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, unsigned int innerInnerLowerModuleIndex, unsigned int middleLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, float& zOut, float& rtOut)
{
    short innerInnerLowerModuleSubdet = modulesInGPU.subdets[innerInnerLowerModuleIndex];
    short middleLowerModuleSubdet = modulesInGPU.subdets[middleLowerModuleIndex];
    short outerOuterLowerModuleSubdet = modulesInGPU.subdets[outerOuterLowerModuleIndex];

    bool pass = false;

    if(innerInnerLowerModuleSubdet == SDL::Barrel
            and middleLowerModuleSubdet == SDL::Barrel
            and outerOuterLowerModuleSubdet == SDL::Barrel)
    {
        pass = passPointingConstraintBBB(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, middleLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, zOut, rtOut);
    }
    else if(innerInnerLowerModuleSubdet == SDL::Barrel
            and middleLowerModuleSubdet == SDL::Barrel
            and outerOuterLowerModuleSubdet == SDL::Endcap)
    {
        pass = passPointingConstraintBBE(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, middleLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, zOut, rtOut);
    }
    else if(innerInnerLowerModuleSubdet == SDL::Barrel
            and middleLowerModuleSubdet == SDL::Endcap
            and outerOuterLowerModuleSubdet == SDL::Endcap)
    {
        pass = passPointingConstraintBBE(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, middleLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, zOut, rtOut);

    }

    else if(innerInnerLowerModuleSubdet == SDL::Endcap
            and middleLowerModuleSubdet == SDL::Endcap
            and outerOuterLowerModuleSubdet == SDL::Endcap)
    {
        pass = passPointingConstraintEEE(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, middleLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, zOut, rtOut);
    }
    
    return pass;
}

__device__ bool SDL::passPointingConstraintBBB(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, unsigned int innerInnerLowerModuleIndex, unsigned int middleLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, float& zOut, float& rtOut)
{
    bool pass = true;
    bool isPS_InLo = (modulesInGPU.moduleType[innerInnerLowerModuleIndex] == SDL::PS);
    bool isPS_OutUp = (modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::PS);

    unsigned int innerInnerAnchorHitIndex = segmentsInGPU.innerMiniDoubletAnchorHitIndices[innerSegmentIndex];
    unsigned int innerOuterAnchorHitIndex = segmentsInGPU.outerMiniDoubletAnchorHitIndices[innerSegmentIndex];
    unsigned int outerOuterAnchorHitIndex= segmentsInGPU.outerMiniDoubletAnchorHitIndices[outerSegmentIndex];


    float rt_InLo = hitsInGPU.rts[innerInnerAnchorHitIndex];
    float rt_OutUp = hitsInGPU.rts[outerOuterAnchorHitIndex];
    float z_InLo = hitsInGPU.zs[innerInnerAnchorHitIndex];
    float z_OutUp = hitsInGPU.zs[outerOuterAnchorHitIndex];
    
    float alpha1GeV_OutUp = asinf(fminf(rt_OutUp * k2Rinv1GeVf / ptCut, sinAlphaMax));

    float rtRatio_OutUpInLo = rt_OutUp / rt_InLo; // Outer segment beginning rt divided by inner segment beginning rt;
    float dzDrtScale = tanf(alpha1GeV_OutUp) / alpha1GeV_OutUp; // The track can bend in r-z plane slightly
    float zpitch_InLo = (isPS_InLo ? pixelPSZpitch : strip2SZpitch);
    float zpitch_OutUp = (isPS_OutUp ? pixelPSZpitch : strip2SZpitch);

    const float zHi = z_InLo + (z_InLo + deltaZLum) * (rtRatio_OutUpInLo - 1.f) * (z_InLo < 0.f ? 1.f : dzDrtScale) + (zpitch_InLo + zpitch_OutUp);
    const float zLo = z_InLo + (z_InLo - deltaZLum) * (rtRatio_OutUpInLo - 1.f) * (z_InLo > 0.f ? 1.f : dzDrtScale) - (zpitch_InLo + zpitch_OutUp); //slope-correction only on outer end


    //Cut 1 - z compatibility
    zOut = z_OutUp;
    rtOut = rt_OutUp;
    if (not (z_OutUp >= zLo and z_OutUp <= zHi))
    {
        pass = false;
    }

    float drt_OutUp_InLo = (rt_OutUp - rt_InLo);
    float invRt_InLo = 1. / rt_InLo;

    float r3_InLo = sqrtf(z_InLo * z_InLo + rt_InLo * rt_InLo);
    float drt_InSeg = hitsInGPU.rts[innerOuterAnchorHitIndex] - hitsInGPU.rts[innerInnerAnchorHitIndex];
    float dz_InSeg = hitsInGPU.zs[innerOuterAnchorHitIndex] - hitsInGPU.zs[innerInnerAnchorHitIndex];

    float dr3_InSeg = sqrtf(hitsInGPU.rts[innerOuterAnchorHitIndex] * hitsInGPU.rts[innerOuterAnchorHitIndex] + hitsInGPU.zs[innerOuterAnchorHitIndex] * hitsInGPU.zs[innerOuterAnchorHitIndex]) - sqrtf(hitsInGPU.rts[innerInnerAnchorHitIndex] * hitsInGPU.rts[innerInnerAnchorHitIndex] + hitsInGPU.zs[innerInnerAnchorHitIndex] * hitsInGPU.zs[innerInnerAnchorHitIndex]);

    float coshEta = dr3_InSeg/drt_InSeg;
    float dzErr = (zpitch_InLo + zpitch_OutUp) * (zpitch_InLo + zpitch_OutUp) * 2.f;

    float sdlThetaMulsF = 0.015f * sqrt(0.1f + 0.2 * (rt_OutUp - rt_InLo) / 50.f) * sqrt(r3_InLo / rt_InLo);
    float sdlMuls = sdlThetaMulsF * 3.f / ptCut * 4.f; // will need a better guess than x4?
    dzErr += sdlMuls * sdlMuls * drt_OutUp_InLo * drt_OutUp_InLo / 3.f * coshEta * coshEta; //sloppy
    dzErr = sqrt(dzErr);

    // Constructing upper and lower bound
    const float dzMean = dz_InSeg / drt_InSeg * drt_OutUp_InLo;
    const float zWindow = dzErr / drt_InSeg * drt_OutUp_InLo + (zpitch_InLo + zpitch_OutUp); //FIXME for ptCut lower than ~0.8 need to add curv path correction
    const float zLoPointed = z_InLo + dzMean * (z_InLo > 0.f ? 1.f : dzDrtScale) - zWindow;
    const float zHiPointed = z_InLo + dzMean * (z_InLo < 0.f ? 1.f : dzDrtScale) + zWindow;

    // Constructing upper and lower bound

    // Cut #2: Pointed Z (Inner segment two MD points to outer segment inner MD)
    if (not (z_OutUp >= zLoPointed and z_OutUp <= zHiPointed))
    {
        pass = false;
    }

    return pass;
}

__device__ bool SDL::passPointingConstraintBBE(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, unsigned int innerInnerLowerModuleIndex, unsigned int middleLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, float& zOut, float& rtOut)
{
    bool pass = true;
    unsigned int outerInnerLowerModuleIndex = middleLowerModuleIndex;

    bool isPS_InLo = (modulesInGPU.moduleType[innerInnerLowerModuleIndex] == SDL::PS);
    bool isPS_OutUp = (modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::PS);

    unsigned int innerInnerAnchorHitIndex = segmentsInGPU.innerMiniDoubletAnchorHitIndices[innerSegmentIndex];


    unsigned int innerOuterAnchorHitIndex = segmentsInGPU.outerMiniDoubletAnchorHitIndices[innerSegmentIndex];
    unsigned int outerOuterAnchorHitIndex= segmentsInGPU.outerMiniDoubletAnchorHitIndices[outerSegmentIndex];

    float rt_InLo = hitsInGPU.rts[innerInnerAnchorHitIndex];
    float rt_OutUp = hitsInGPU.rts[outerOuterAnchorHitIndex];
    float z_InLo = hitsInGPU.zs[innerInnerAnchorHitIndex];
    float z_OutUp = hitsInGPU.zs[outerOuterAnchorHitIndex];
    
    float alpha1GeV_OutLo = asinf(fminf(rt_OutUp * k2Rinv1GeVf / ptCut, sinAlphaMax));

    float rtRatio_OutUpInLo = rt_OutUp / rt_InLo; // Outer segment beginning rt divided by inner segment beginning rt;
    float dzDrtScale = tanf(alpha1GeV_OutLo) / alpha1GeV_OutLo; // The track can bend in r-z plane slightly
    float zpitch_InLo = (isPS_InLo ? pixelPSZpitch : strip2SZpitch);
    float zpitch_OutUp = (isPS_OutUp ? pixelPSZpitch : strip2SZpitch);
    float zGeom = zpitch_InLo + zpitch_OutUp;

    float zLo = z_InLo + (z_InLo - deltaZLum) * (rtRatio_OutUpInLo - 1.f) * (z_InLo > 0.f ? 1.f : dzDrtScale) - zGeom; 

    // Cut #0: Preliminary (Only here in endcap case)
    if(not(z_InLo * z_OutUp > 0))
    {
        pass = false;
    }
    float dLum = copysignf(deltaZLum, z_InLo);
    bool isOutSgInnerMDPS = modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::PS;
    float rtGeom1 = isOutSgInnerMDPS ? pixelPSZpitch : strip2SZpitch;
    float zGeom1 = copysignf(zGeom,z_InLo);
    float rtLo = rt_InLo * (1.f + (z_OutUp - z_InLo - zGeom1) / (z_InLo + zGeom1 + dLum) / dzDrtScale) - rtGeom1; //slope correction only on the lower end
    zOut = z_OutUp;
    rtOut = rt_OutUp;

    //Cut #1: rt condition
    if (not (rtOut >= rtLo))
    {
        pass = false;
    }

    float zInForHi = z_InLo - zGeom1 - dLum;
    if(zInForHi * z_InLo < 0)
    {
        zInForHi = copysignf(0.1f,z_InLo);
    }
    float rtHi = rt_InLo * (1.f + (z_OutUp - z_InLo + zGeom1) / zInForHi) + rtGeom1;

    //Cut #2: rt condition
    if (not (rt_OutUp >= rtLo and rt_OutUp <= rtHi))
    {
        pass = false;
    }

    float rIn = sqrtf(z_InLo * z_InLo + rt_InLo * rt_InLo);
    const float drtSDIn = hitsInGPU.rts[innerOuterAnchorHitIndex] - hitsInGPU.rts[innerInnerAnchorHitIndex];
    const float dzSDIn = hitsInGPU.zs[innerOuterAnchorHitIndex] - hitsInGPU.zs[innerInnerAnchorHitIndex];

    const float dr3SDIn = sqrtf(hitsInGPU.rts[innerOuterAnchorHitIndex] * hitsInGPU.rts[innerOuterAnchorHitIndex] +  hitsInGPU.zs[innerOuterAnchorHitIndex] * hitsInGPU.zs[innerOuterAnchorHitIndex]) - sqrtf(hitsInGPU.rts[innerInnerAnchorHitIndex] * hitsInGPU.rts[innerInnerAnchorHitIndex] +  hitsInGPU.zs[innerInnerAnchorHitIndex] *hitsInGPU.zs[innerInnerAnchorHitIndex]);

    const float coshEta = dr3SDIn / drtSDIn; //direction estimate
    const float dzOutInAbs = fabsf(z_OutUp - z_InLo);
    const float multDzDr = dzOutInAbs * coshEta / (coshEta * coshEta - 1.f);
    const float zGeom1_another = pixelPSZpitch; //What's this?
    const float kZ = (z_OutUp - z_InLo) / dzSDIn;
    float drtErr = zGeom1_another * zGeom1_another * drtSDIn * drtSDIn / dzSDIn / dzSDIn * (1.f - 2.f * kZ + 2.f * kZ * kZ); //Notes:122316
    const float sdlThetaMulsF = 0.015f * sqrtf(0.1f + 0.2 * (rt_OutUp - rt_InLo) / 50.f) * sqrtf(rIn / rt_InLo);
    const float sdlMuls = sdlThetaMulsF * 3.f / ptCut * 4.f; //will need a better guess than x4?
    drtErr += sdlMuls * sdlMuls * multDzDr * multDzDr / 3.f * coshEta * coshEta; //sloppy: relative muls is 1/3 of total muls
    drtErr = sqrtf(drtErr);
    const float drtMean = drtSDIn * dzOutInAbs / fabsf(dzSDIn); //
    const float rtWindow = drtErr + rtGeom1;
    const float rtLo_another = rt_InLo + drtMean / dzDrtScale - rtWindow;
    const float rtHi_another = rt_InLo + drtMean + rtWindow;

    //Cut #3: rt-z pointed
    if (not (kZ >= 0 and rtOut >= rtLo and rtOut <= rtHi))
    {
        pass = false;
    }

    return pass;
}


__device__ bool SDL::passPointingConstraintEEE(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, unsigned int innerInnerLowerModuleIndex, unsigned int middleLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, float& zOut, float& rtOut)
{
    bool pass = true;
    bool isPS_InLo = (modulesInGPU.moduleType[innerInnerLowerModuleIndex] == SDL::PS);
    bool isPS_OutUp = (modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::PS);

    unsigned int innerInnerAnchorHitIndex = segmentsInGPU.innerMiniDoubletAnchorHitIndices[innerSegmentIndex];

    unsigned int innerOuterAnchorHitIndex = segmentsInGPU.outerMiniDoubletAnchorHitIndices[innerSegmentIndex];
    unsigned int outerOuterAnchorHitIndex= segmentsInGPU.outerMiniDoubletAnchorHitIndices[outerSegmentIndex];

    float rt_InLo = hitsInGPU.rts[innerInnerAnchorHitIndex];
    float rt_OutUp = hitsInGPU.rts[outerOuterAnchorHitIndex];
    float z_InLo = hitsInGPU.zs[innerInnerAnchorHitIndex];
    float z_OutUp = hitsInGPU.zs[outerOuterAnchorHitIndex];

    float alpha1GeV_OutUp = asinf(fminf(rt_OutUp * k2Rinv1GeVf / ptCut, sinAlphaMax));

    float rtRatio_OutUpInLo = rt_OutUp / rt_InLo; // Outer segment beginning rt divided by inner segment beginning rt;
    float dzDrtScale = tanf(alpha1GeV_OutUp) / alpha1GeV_OutUp; // The track can bend in r-z plane slightly
    float zpitch_InLo = (isPS_InLo ? pixelPSZpitch : strip2SZpitch);
    float zpitch_OutUp = (isPS_OutUp ? pixelPSZpitch : strip2SZpitch);
    float zGeom = zpitch_InLo + zpitch_OutUp;

    const float zLo = z_InLo + (z_InLo - deltaZLum) * (rtRatio_OutUpInLo - 1.f) * (z_InLo > 0.f ? 1.f : dzDrtScale) - zGeom; //slope-correction only on outer end


    // Cut #0: Preliminary (Only here in endcap case)
    if(not(z_InLo * z_OutUp > 0))
    {
        pass = false;
    }
    
    float dLum = copysignf(deltaZLum, z_InLo);
    bool isOutSgOuterMDPS = modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::PS;
    bool isInSgInnerMDPS = modulesInGPU.moduleType[innerInnerLowerModuleIndex] == SDL::PS;

    float rtGeom = (isInSgInnerMDPS and isOutSgOuterMDPS) ? 2.f * pixelPSZpitch : (isInSgInnerMDPS or isOutSgOuterMDPS) ? pixelPSZpitch + strip2SZpitch : 2.f * strip2SZpitch;

    float zGeom1 = copysignf(zGeom,z_InLo);
    float dz = z_OutUp - z_InLo;
    const float rtLo = rt_InLo * (1.f + dz / (z_InLo + dLum) / dzDrtScale) - rtGeom; //slope correction only on the lower end

    zOut = z_OutUp;
    rtOut = rt_OutUp;

    //Cut #1: rt condition
    if (not (rtOut >= rtLo))
    {
        pass = false;
    }

    float rtHi = rt_InLo * (1.f + dz / (z_InLo - dLum)) + rtGeom;

    if (not (rtOut >= rtLo and rtOut <= rtHi))
    {
        pass = false;
    }
    
    unsigned int innerOuterLowerModuleIndex = middleLowerModuleIndex;
    bool isInSgOuterMDPS = modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::PS;

    float drOutIn = rtOut - rt_InLo;
    float drtSDIn = hitsInGPU.rts[innerOuterAnchorHitIndex] - hitsInGPU.rts[innerInnerAnchorHitIndex];

    float dzSDIn = hitsInGPU.zs[innerOuterAnchorHitIndex] - hitsInGPU.zs[innerInnerAnchorHitIndex];

    float dr3SDIn = sqrtf(hitsInGPU.zs[innerOuterAnchorHitIndex] * hitsInGPU.zs[innerOuterAnchorHitIndex] + hitsInGPU.rts[innerOuterAnchorHitIndex] * hitsInGPU.rts[innerOuterAnchorHitIndex]) - sqrtf(hitsInGPU.zs[innerInnerAnchorHitIndex] * hitsInGPU.zs[innerInnerAnchorHitIndex] + hitsInGPU.rts[innerInnerAnchorHitIndex] * hitsInGPU.rts[innerInnerAnchorHitIndex]); 

    float coshEta = dr3SDIn / drtSDIn; //direction estimate
    float dzOutInAbs =  fabsf(z_OutUp - z_InLo);
    float multDzDr = dzOutInAbs * coshEta / (coshEta * coshEta - 1.f);

    float kZ = (z_OutUp - z_InLo) / dzSDIn;
    float sdlThetaMulsF = 0.015f * sqrtf(0.1f + 0.2 * (rt_OutUp - rt_InLo) / 50.f);

    float sdlMuls = sdlThetaMulsF * 3.f / ptCut * 4.f; //will need a better guess than x4?

    float drtErr = sqrtf(pixelPSZpitch * pixelPSZpitch * 2.f / (dzSDIn * dzSDIn) * (dzOutInAbs * dzOutInAbs) + sdlMuls * sdlMuls * multDzDr * multDzDr / 3.f * coshEta * coshEta);

    float drtMean = drtSDIn * dzOutInAbs/fabsf(dzSDIn);
    float rtWindow = drtErr + rtGeom;
    float rtLo_point = rt_InLo + drtMean / dzDrtScale - rtWindow;
    float rtHi_point = rt_InLo + drtMean + rtWindow;

    // Cut #3: rt-z pointed
    // https://github.com/slava77/cms-tkph2-ntuple/blob/superDoubletLinked-91X-noMock/doubletAnalysis.C#L3765

    if (isInSgInnerMDPS and isInSgOuterMDPS) // If both PS then we can point
    {
        if (not (kZ >= 0 and rtOut >= rtLo_point and rtOut <= rtHi_point))
        {
            pass = false;
        }
    }

    return pass;
}

__device__ bool SDL::hasCommonMiniDoublet(struct segments& segmentsInGPU, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex)
{

    if(segmentsInGPU.mdIndices[innerSegmentIndex * 2 + 1] == segmentsInGPU.mdIndices[outerSegmentIndex * 2])
    {
        return true;
    }
    else return false;
}

