#ifdef __CUDACC__
#define CUDA_CONST_VAR __device__
#endif
#include "Tracklet.cuh"

//#ifdef CACHE_ALLOC
#include "allocate.h"
//#endif

CUDA_CONST_VAR float SDL::pt_betaMax = 7.0;


void SDL::createTrackletsInUnifiedMemory(struct tracklets& trackletsInGPU, unsigned int maxTracklets, unsigned int maxPixelTracklets, unsigned int nLowerModules)
{

    unsigned int nMemoryLocations = maxTracklets * nLowerModules + maxPixelTracklets;
#ifdef CACHE_ALLOC
    cudaStream_t stream =0;
    trackletsInGPU.segmentIndices = (unsigned int*)cms::cuda::allocate_managed(nMemoryLocations * sizeof(unsigned int) * 2,stream);
    trackletsInGPU.lowerModuleIndices = (unsigned int*)cms::cuda::allocate_managed(nMemoryLocations * sizeof(unsigned int) * 4,stream);//split up to avoid runtime error of exceeding max byte allocation at a time
    trackletsInGPU.nTracklets = (unsigned int*)cms::cuda::allocate_managed(nLowerModules * sizeof(unsigned int),stream);
    trackletsInGPU.zOut = (float*)cms::cuda::allocate_managed(nMemoryLocations * sizeof(float) * 4,stream);
    trackletsInGPU.betaIn = (float*)cms::cuda::allocate_managed(nMemoryLocations * sizeof(float) * 2,stream);
#else
    cudaMallocManaged(&trackletsInGPU.segmentIndices, 2 * nMemoryLocations * sizeof(unsigned int));
    cudaMallocManaged(&trackletsInGPU.lowerModuleIndices, 4 * nMemoryLocations * sizeof(unsigned int));
    cudaMallocManaged(&trackletsInGPU.nTracklets,nLowerModules * sizeof(unsigned int));
    cudaMallocManaged(&trackletsInGPU.zOut, nMemoryLocations *4* sizeof(float));
    cudaMallocManaged(&trackletsInGPU.betaIn, nMemoryLocations *2* sizeof(float));
#endif
    trackletsInGPU.rtOut = trackletsInGPU.zOut + maxTracklets * nLowerModules;
    trackletsInGPU.deltaPhiPos = trackletsInGPU.zOut + nMemoryLocations * 2;
    trackletsInGPU.deltaPhi = trackletsInGPU.zOut + nMemoryLocations * 3;
    trackletsInGPU.betaOut = trackletsInGPU.betaIn + nMemoryLocations;
#pragma omp parallel for
    for(size_t i = 0; i<nLowerModules;i++)
    {
        trackletsInGPU.nTracklets[i] = 0;
    }

}
void SDL::createTrackletsInExplicitMemory(struct tracklets& trackletsInGPU, unsigned int maxTracklets, unsigned int nLowerModules)
{
#ifdef CACHE_ALLOC
    cudaStream_t stream =0;
    int dev;
    cudaGetDevice(&dev);
    trackletsInGPU.segmentIndices = (unsigned int*)cms::cuda::allocate_device(dev,maxTracklets * nLowerModules * sizeof(unsigned int) * 2,stream);
    trackletsInGPU.lowerModuleIndices = (unsigned int*)cms::cuda::allocate_device(dev,maxTracklets * nLowerModules * sizeof(unsigned int) * 4,stream);//split up to avoid runtime error of exceeding max byte allocation at a time
    trackletsInGPU.zOut = (float*)cms::cuda::allocate_device(dev,maxTracklets * nLowerModules * sizeof(float) * 4,stream);
    trackletsInGPU.betaIn = (float*)cms::cuda::allocate_device(dev,maxTracklets * nLowerModules * sizeof(float) * 2,stream);
//  #ifdef Full_Explicit
    trackletsInGPU.nTracklets = (unsigned int*)cms::cuda::allocate_device(dev,nLowerModules * sizeof(unsigned int),stream);
    cudaMemset(trackletsInGPU.nTracklets,0,nLowerModules * sizeof(unsigned int));
//  #else
//    trackletsInGPU.nTracklets = (unsigned int*)cms::cuda::allocate_managed(nLowerModules * sizeof(unsigned int),stream);
//  #endif

#else
    cudaMalloc(&trackletsInGPU.segmentIndices, 2 * maxTracklets * nLowerModules * sizeof(unsigned int));
    cudaMalloc(&trackletsInGPU.lowerModuleIndices, 4 * maxTracklets * nLowerModules * sizeof(unsigned int));
    cudaMalloc(&trackletsInGPU.zOut, maxTracklets * nLowerModules * 4* sizeof(float));
    cudaMalloc(&trackletsInGPU.betaIn, maxTracklets * nLowerModules * 2*sizeof(float));
//  #ifdef Full_Explicit
    cudaMalloc(&trackletsInGPU.nTracklets,nLowerModules * sizeof(unsigned int));
    cudaMemset(trackletsInGPU.nTracklets,0,nLowerModules*sizeof(unsigned int));
//  #else
//    cudaMallocManaged(&trackletsInGPU.nTracklets,nLowerModules * sizeof(unsigned int));
//  #endif
#endif
    trackletsInGPU.rtOut = trackletsInGPU.zOut + maxTracklets * nLowerModules;
    trackletsInGPU.deltaPhiPos = trackletsInGPU.zOut + maxTracklets * nLowerModules * 2;
    trackletsInGPU.deltaPhi = trackletsInGPU.zOut + maxTracklets * nLowerModules * 3;
 
    trackletsInGPU.betaOut = trackletsInGPU.betaIn + maxTracklets * nLowerModules;
}

__device__ void SDL::addTrackletToMemory(struct tracklets& trackletsInGPU, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, unsigned int innerInnerLowerModuleIndex, unsigned int innerOuterLowerModuleIndex, unsigned int outerInnerLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& deltaPhi, float& betaIn, float& betaOut, unsigned int trackletIndex)
{
    trackletsInGPU.segmentIndices[trackletIndex * 2] = innerSegmentIndex;
    trackletsInGPU.segmentIndices[trackletIndex * 2 + 1] = outerSegmentIndex;
    trackletsInGPU.lowerModuleIndices[trackletIndex * 4] = innerInnerLowerModuleIndex;
    trackletsInGPU.lowerModuleIndices[trackletIndex * 4 + 1] = innerOuterLowerModuleIndex;
    trackletsInGPU.lowerModuleIndices[trackletIndex * 4 + 2] = outerInnerLowerModuleIndex;
    trackletsInGPU.lowerModuleIndices[trackletIndex * 4 + 3] = outerOuterLowerModuleIndex;

    trackletsInGPU.zOut[trackletIndex] = zOut;
    trackletsInGPU.rtOut[trackletIndex] = rtOut;
    trackletsInGPU.deltaPhiPos[trackletIndex] = deltaPhiPos;
    trackletsInGPU.deltaPhi[trackletIndex] = deltaPhi;

    trackletsInGPU.betaIn[trackletIndex] = betaIn;
    trackletsInGPU.betaOut[trackletIndex] = betaOut;
}


SDL::tracklets::tracklets()
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

SDL::tracklets::~tracklets()
{
}

void SDL::tracklets::freeMemoryCache()
{
#ifdef Explicit_Tracklet
    int dev;
    cudaGetDevice(&dev);
    cms::cuda::free_device(dev,segmentIndices);
    cms::cuda::free_device(dev,lowerModuleIndices);
    cms::cuda::free_device(dev,zOut);
    cms::cuda::free_device(dev,betaIn);
//  #ifdef Full_Explicit
    cms::cuda::free_device(dev,nTracklets);
//  #else
//    cms::cuda::free_managed(nTracklets);
//  #endif
#else
    cms::cuda::free_managed(segmentIndices);
    cms::cuda::free_managed(lowerModuleIndices);
    cms::cuda::free_managed(zOut);
    cms::cuda::free_managed(betaIn);
    cms::cuda::free_managed(nTracklets);
#endif
}
void SDL::tracklets::freeMemory()
{
    cudaFree(segmentIndices);
    cudaFree(lowerModuleIndices);
    cudaFree(nTracklets);
    cudaFree(zOut);
    cudaFree(betaIn);
}

__device__ bool SDL::runTrackletDefaultAlgo(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, unsigned int innerInnerLowerModuleIndex, unsigned int innerOuterLowerModuleIndex, unsigned int outerInnerLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& deltaPhi, float& betaIn, float&
        betaOut, unsigned int N_MAX_SEGMENTS_PER_MODULE)
{

    bool pass = false;

    short innerInnerLowerModuleSubdet = modulesInGPU.subdets[innerInnerLowerModuleIndex];
    short innerOuterLowerModuleSubdet = modulesInGPU.subdets[innerOuterLowerModuleIndex];
    short outerInnerLowerModuleSubdet = modulesInGPU.subdets[outerInnerLowerModuleIndex];
    short outerOuterLowerModuleSubdet = modulesInGPU.subdets[outerOuterLowerModuleIndex];

    if(innerInnerLowerModuleSubdet == SDL::InnerPixel) 
    {
        if(outerInnerLowerModuleSubdet == SDL::Barrel and outerOuterLowerModuleSubdet == SDL::Barrel)
        {
            pass = runTrackletDefaultAlgoPPBB(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, outerInnerLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,N_MAX_SEGMENTS_PER_MODULE);     
        }
        else if(outerInnerLowerModuleSubdet == SDL::Barrel and outerOuterLowerModuleSubdet == SDL::Endcap)
        {
            pass = runTrackletDefaultAlgoPPBB(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, outerInnerLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,N_MAX_SEGMENTS_PER_MODULE);     
        }
        else if(outerInnerLowerModuleSubdet == SDL::Endcap and outerOuterLowerModuleSubdet == SDL::Endcap)
        {
	    pass = runTrackletDefaultAlgoPPEE(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, outerInnerLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,N_MAX_SEGMENTS_PER_MODULE);
        }
    }

    else if(innerInnerLowerModuleSubdet == SDL::Barrel
            and innerOuterLowerModuleSubdet == SDL::Barrel
            and outerInnerLowerModuleSubdet == SDL::Barrel
            and outerOuterLowerModuleSubdet == SDL::Barrel)
    {
        pass = runTrackletDefaultAlgoBBBB(modulesInGPU,hitsInGPU,mdsInGPU,segmentsInGPU,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,innerSegmentIndex,outerSegmentIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut);
    }

    else if(innerInnerLowerModuleSubdet == SDL::Barrel
            and innerOuterLowerModuleSubdet == SDL::Barrel
            and outerInnerLowerModuleSubdet == SDL::Endcap
            and outerOuterLowerModuleSubdet == SDL::Endcap)
    {
        pass = runTrackletDefaultAlgoBBEE(modulesInGPU,hitsInGPU,mdsInGPU,segmentsInGPU,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,innerSegmentIndex,outerSegmentIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut);
    }

    else if(innerInnerLowerModuleSubdet == SDL::Barrel
            and innerOuterLowerModuleSubdet == SDL::Barrel
            and outerInnerLowerModuleSubdet == SDL::Barrel
            and outerOuterLowerModuleSubdet == SDL::Endcap)
    {
        pass = runTrackletDefaultAlgoBBBB(modulesInGPU,hitsInGPU,mdsInGPU,segmentsInGPU,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,innerSegmentIndex,outerSegmentIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut);

    }

    else if(innerInnerLowerModuleSubdet == SDL::Barrel
            and innerOuterLowerModuleSubdet == SDL::Endcap
            and outerInnerLowerModuleSubdet == SDL::Endcap
            and outerOuterLowerModuleSubdet == SDL::Endcap)
    {
        pass = runTrackletDefaultAlgoBBEE(modulesInGPU,hitsInGPU,mdsInGPU,segmentsInGPU,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,innerSegmentIndex,outerSegmentIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut);

    }

    else if(innerInnerLowerModuleSubdet == SDL::Endcap
            and innerOuterLowerModuleSubdet == SDL::Endcap
            and outerInnerLowerModuleSubdet == SDL::Endcap
            and outerOuterLowerModuleSubdet == SDL::Endcap)
    {
        pass = runTrackletDefaultAlgoEEEE(modulesInGPU,hitsInGPU,mdsInGPU,segmentsInGPU,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,innerSegmentIndex,outerSegmentIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut);
    }
    
    return pass;
}

__device__ bool SDL::runTrackletDefaultAlgoBBBB(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, unsigned int innerInnerLowerModuleIndex, unsigned int innerOuterLowerModuleIndex, unsigned int outerInnerLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& dPhi, float& betaIn, float& betaOut)
{
    bool pass = true;

    bool isPS_InLo = (modulesInGPU.moduleType[innerInnerLowerModuleIndex] == SDL::PS);
    bool isPS_OutLo = (modulesInGPU.moduleType[outerInnerLowerModuleIndex] == SDL::PS);
    unsigned int innerInnerAnchorHitIndex = segmentsInGPU.innerMiniDoubletAnchorHitIndices[innerSegmentIndex];

    unsigned int outerInnerAnchorHitIndex = segmentsInGPU.innerMiniDoubletAnchorHitIndices[outerSegmentIndex];

    unsigned int innerOuterAnchorHitIndex = segmentsInGPU.outerMiniDoubletAnchorHitIndices[innerSegmentIndex];
    unsigned int outerOuterAnchorHitIndex= segmentsInGPU.outerMiniDoubletAnchorHitIndices[outerSegmentIndex];


    float rt_InLo = hitsInGPU.rts[innerInnerAnchorHitIndex];
    float rt_OutLo = hitsInGPU.rts[outerInnerAnchorHitIndex];
    float z_InLo = hitsInGPU.zs[innerInnerAnchorHitIndex];
    float z_OutLo = hitsInGPU.zs[outerInnerAnchorHitIndex];
    
    float alpha1GeV_OutLo = asinf(fminf(rt_OutLo * k2Rinv1GeVf / ptCut, sinAlphaMax));

    float rtRatio_OutLoInLo = rt_OutLo / rt_InLo; // Outer segment beginning rt divided by inner segment beginning rt;
    float dzDrtScale = tanf(alpha1GeV_OutLo) / alpha1GeV_OutLo; // The track can bend in r-z plane slightly
    float zpitch_InLo = (isPS_InLo ? pixelPSZpitch : strip2SZpitch);
    float zpitch_OutLo = (isPS_OutLo ? pixelPSZpitch : strip2SZpitch);

    float zHi = z_InLo + (z_InLo + deltaZLum) * (rtRatio_OutLoInLo - 1.f) * (z_InLo < 0.f ? 1.f : dzDrtScale) + (zpitch_InLo + zpitch_OutLo);
    float zLo = z_InLo + (z_InLo - deltaZLum) * (rtRatio_OutLoInLo - 1.f) * (z_InLo > 0.f ? 1.f : dzDrtScale) - (zpitch_InLo + zpitch_OutLo); 


    //Cut 1 - z compatibility
    zOut = z_OutLo;
    rtOut = rt_OutLo;
    if (not (z_OutLo >= zLo and z_OutLo <= zHi))
    {
        pass = false;
    }

    float drt_OutLo_InLo = (rt_OutLo - rt_InLo);
    float r3_InLo = sqrtf(z_InLo * z_InLo + rt_InLo * rt_InLo);
//    float drt_InSeg = innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->rt() - innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->rt();
    float drt_InSeg = hitsInGPU.rts[innerOuterAnchorHitIndex] - hitsInGPU.rts[innerInnerAnchorHitIndex];
    float dz_InSeg = hitsInGPU.zs[innerOuterAnchorHitIndex] - hitsInGPU.zs[innerInnerAnchorHitIndex];

    float dr3_InSeg = sqrtf(hitsInGPU.rts[innerOuterAnchorHitIndex] * hitsInGPU.rts[innerOuterAnchorHitIndex] + hitsInGPU.zs[innerOuterAnchorHitIndex] * hitsInGPU.zs[innerOuterAnchorHitIndex]) - sqrtf(hitsInGPU.rts[innerInnerAnchorHitIndex] * hitsInGPU.rts[innerInnerAnchorHitIndex] + hitsInGPU.zs[innerInnerAnchorHitIndex] * hitsInGPU.zs[innerInnerAnchorHitIndex]);

    float coshEta = dr3_InSeg/drt_InSeg;
    float dzErr = (zpitch_InLo + zpitch_OutLo) * (zpitch_InLo + zpitch_OutLo) * 2.f;

    float sdlThetaMulsF = 0.015f * sqrtf(0.1f + 0.2 * (rt_OutLo - rt_InLo) / 50.f) * sqrtf(r3_InLo / rt_InLo);
    float sdlMuls = sdlThetaMulsF * 3.f / ptCut * 4.f; // will need a better guess than x4?
    dzErr += sdlMuls * sdlMuls * drt_OutLo_InLo * drt_OutLo_InLo / 3.f * coshEta * coshEta; //sloppy
    dzErr = sqrtf(dzErr);

    // Constructing upper and lower bound
    const float dzMean = dz_InSeg / drt_InSeg * drt_OutLo_InLo;
    const float zWindow = dzErr / drt_InSeg * drt_OutLo_InLo + (zpitch_InLo + zpitch_OutLo); //FIXME for ptCut lower than ~0.8 need to add curv path correction
    const float zLoPointed = z_InLo + dzMean * (z_InLo > 0.f ? 1.f : dzDrtScale) - zWindow;
    const float zHiPointed = z_InLo + dzMean * (z_InLo < 0.f ? 1.f : dzDrtScale) + zWindow;

    // Cut #2: Pointed Z (Inner segment two MD points to outer segment inner MD)
    if (not (z_OutLo >= zLoPointed and z_OutLo <= zHiPointed))
    {
        pass = false;
    }

    float sdlPVoff = 0.1f/rt_OutLo;
    float sdlCut = alpha1GeV_OutLo + sqrtf(sdlMuls * sdlMuls + sdlPVoff * sdlPVoff);
    
    
    deltaPhiPos = deltaPhi(hitsInGPU.xs[innerOuterAnchorHitIndex], hitsInGPU.ys[innerOuterAnchorHitIndex], hitsInGPU.zs[innerOuterAnchorHitIndex], hitsInGPU.xs[outerOuterAnchorHitIndex], hitsInGPU.ys[outerOuterAnchorHitIndex], hitsInGPU.zs[outerOuterAnchorHitIndex]);

    // Cut #3: FIXME:deltaPhiPos can be tighter
    if (not (fabsf(deltaPhiPos) <= sdlCut) )
    {
        pass = false;
    }

    float midPointX = (hitsInGPU.xs[innerInnerAnchorHitIndex] + hitsInGPU.xs[outerInnerAnchorHitIndex])/2;
    float midPointY = (hitsInGPU.ys[innerInnerAnchorHitIndex] + hitsInGPU.ys[outerInnerAnchorHitIndex])/2;
    float midPointZ = (hitsInGPU.zs[innerInnerAnchorHitIndex] + hitsInGPU.zs[outerInnerAnchorHitIndex])/2;

    float diffX = hitsInGPU.xs[outerInnerAnchorHitIndex] - hitsInGPU.xs[innerInnerAnchorHitIndex];
    float diffY = hitsInGPU.ys[outerInnerAnchorHitIndex] - hitsInGPU.ys[innerInnerAnchorHitIndex] ;
    float diffZ = hitsInGPU.zs[outerInnerAnchorHitIndex] - hitsInGPU.zs[innerInnerAnchorHitIndex]; 

    dPhi = deltaPhi(midPointX, midPointY, midPointZ, diffX, diffY, diffZ);

    // Cut #4: deltaPhiChange
    if (not (fabsf(dPhi) <= sdlCut))
    {
        pass = false;
    }

    // First obtaining the raw betaIn and betaOut values without any correction and just purely based on the mini-doublet hit positions

    float alpha_InLo = segmentsInGPU.dPhiChanges[innerSegmentIndex];
    float alpha_OutLo = segmentsInGPU.dPhiChanges[outerSegmentIndex];

    bool isEC_lastLayer = modulesInGPU.subdets[outerOuterLowerModuleIndex] == SDL::Endcap and modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::TwoS;

//    float alpha_outUp = ; 
    unsigned int outerOuterEdgeIndex = hitsInGPU.edge2SMap[outerOuterAnchorHitIndex];
    float alpha_OutUp,alpha_OutUp_highEdge,alpha_OutUp_lowEdge;
    
    alpha_OutUp = deltaPhi(hitsInGPU.xs[outerOuterAnchorHitIndex],hitsInGPU.ys[outerOuterAnchorHitIndex],hitsInGPU.zs[outerOuterAnchorHitIndex],hitsInGPU.xs[outerOuterAnchorHitIndex] - hitsInGPU.xs[outerInnerAnchorHitIndex], hitsInGPU.ys[outerOuterAnchorHitIndex] - hitsInGPU.ys[outerInnerAnchorHitIndex], hitsInGPU.zs[outerOuterAnchorHitIndex] - hitsInGPU.zs[outerInnerAnchorHitIndex]);

    alpha_OutUp_highEdge = alpha_OutUp;
    alpha_OutUp_lowEdge = alpha_OutUp;

    float tl_axis_x = hitsInGPU.xs[outerOuterAnchorHitIndex] - hitsInGPU.xs[innerInnerAnchorHitIndex];
    float tl_axis_y = hitsInGPU.ys[outerOuterAnchorHitIndex] - hitsInGPU.ys[innerInnerAnchorHitIndex];
    float tl_axis_z = hitsInGPU.zs[outerOuterAnchorHitIndex] - hitsInGPU.zs[innerInnerAnchorHitIndex];

    float tl_axis_highEdge_x = tl_axis_x;
    float tl_axis_highEdge_y = tl_axis_y;
    float tl_axis_highEdge_z = tl_axis_z;

    float tl_axis_lowEdge_x = tl_axis_x;
    float tl_axis_lowEdge_y = tl_axis_y;
    float tl_axis_lowEdge_z = tl_axis_z;

    betaIn = alpha_InLo - deltaPhi(hitsInGPU.xs[innerInnerAnchorHitIndex], hitsInGPU.ys[innerInnerAnchorHitIndex], hitsInGPU.zs[innerInnerAnchorHitIndex], tl_axis_x, tl_axis_y, tl_axis_z);

    float betaInRHmin = betaIn;
    float betaInRHmax = betaIn;

    betaOut = -alpha_OutUp + deltaPhi(hitsInGPU.xs[outerOuterAnchorHitIndex], hitsInGPU.ys[outerOuterAnchorHitIndex], hitsInGPU.zs[outerOuterAnchorHitIndex], tl_axis_x, tl_axis_y, tl_axis_z);

    float betaOutRHmin = betaOut;
    float betaOutRHmax = betaOut;

    if(isEC_lastLayer)
    {
        alpha_OutUp_highEdge = deltaPhi(hitsInGPU.highEdgeXs[outerOuterEdgeIndex],hitsInGPU.highEdgeYs[outerOuterEdgeIndex],hitsInGPU.zs[outerOuterAnchorHitIndex],hitsInGPU.highEdgeXs[outerOuterEdgeIndex] - hitsInGPU.xs[outerInnerAnchorHitIndex], hitsInGPU.highEdgeYs[outerOuterEdgeIndex] - hitsInGPU.ys[outerInnerAnchorHitIndex], hitsInGPU.zs[outerOuterAnchorHitIndex] - hitsInGPU.zs[outerInnerAnchorHitIndex]);

        alpha_OutUp_lowEdge = deltaPhi(hitsInGPU.lowEdgeXs[outerOuterEdgeIndex],hitsInGPU.lowEdgeYs[outerOuterEdgeIndex],hitsInGPU.zs[outerOuterAnchorHitIndex],hitsInGPU.lowEdgeXs[outerOuterEdgeIndex] - hitsInGPU.xs[outerInnerAnchorHitIndex], hitsInGPU.lowEdgeYs[outerOuterEdgeIndex] - hitsInGPU.ys[outerInnerAnchorHitIndex], hitsInGPU.zs[outerOuterAnchorHitIndex] - hitsInGPU.zs[outerInnerAnchorHitIndex]);

        tl_axis_highEdge_x = hitsInGPU.highEdgeXs[outerOuterEdgeIndex] - hitsInGPU.xs[innerInnerAnchorHitIndex];
        tl_axis_highEdge_y = hitsInGPU.highEdgeYs[outerOuterEdgeIndex] - hitsInGPU.ys[innerInnerAnchorHitIndex];
        tl_axis_highEdge_z = hitsInGPU.zs[outerOuterAnchorHitIndex] - hitsInGPU.zs[innerInnerAnchorHitIndex];

        tl_axis_lowEdge_x = hitsInGPU.lowEdgeXs[outerOuterEdgeIndex] - hitsInGPU.xs[innerInnerAnchorHitIndex];
        tl_axis_lowEdge_y = hitsInGPU.lowEdgeYs[outerOuterEdgeIndex] - hitsInGPU.ys[innerInnerAnchorHitIndex];
        tl_axis_lowEdge_z = hitsInGPU.zs[outerOuterAnchorHitIndex] - hitsInGPU.zs[innerInnerAnchorHitIndex];

        betaOutRHmin = -alpha_OutUp_highEdge + deltaPhi(hitsInGPU.highEdgeXs[outerOuterEdgeIndex], hitsInGPU.highEdgeYs[outerOuterEdgeIndex], hitsInGPU.zs[outerOuterAnchorHitIndex], tl_axis_highEdge_x, tl_axis_highEdge_y, tl_axis_highEdge_z);

        betaOutRHmax = -alpha_OutUp_lowEdge + deltaPhi(hitsInGPU.lowEdgeXs[outerOuterEdgeIndex], hitsInGPU.lowEdgeYs[outerOuterEdgeIndex], hitsInGPU.zs[outerOuterAnchorHitIndex], tl_axis_lowEdge_x, tl_axis_lowEdge_y, tl_axis_lowEdge_z);

    }

    //beta computation
    float drt_tl_axis = sqrtf(tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);
    float drt_tl_lowEdge = sqrtf(tl_axis_lowEdge_x * tl_axis_lowEdge_x + tl_axis_lowEdge_y * tl_axis_lowEdge_y);
    float drt_tl_highEdge = sqrtf(tl_axis_highEdge_x * tl_axis_highEdge_x + tl_axis_highEdge_y * tl_axis_highEdge_y);

    float corrF = 1.f;
    bool pass_betaIn_cut = false;
    //innerOuterAnchor - innerInnerAnchor
    const float rt_InSeg = sqrtf((hitsInGPU.xs[innerOuterAnchorHitIndex] - hitsInGPU.xs[innerInnerAnchorHitIndex]) * (hitsInGPU.xs[innerOuterAnchorHitIndex] - hitsInGPU.xs[innerInnerAnchorHitIndex]) + (hitsInGPU.ys[innerOuterAnchorHitIndex] - hitsInGPU.ys[innerInnerAnchorHitIndex]) * (hitsInGPU.ys[innerOuterAnchorHitIndex] - hitsInGPU.ys[innerInnerAnchorHitIndex]));
    float betaInCut = asinf(fminf((-rt_InSeg * corrF + drt_tl_axis) * k2Rinv1GeVf / ptCut, sinAlphaMax)) + (0.02f / drt_InSeg);
    // const float betaIn_cut = std::asin((-rt_InSeg * corrF + drt_tl_axis) * k2Rinv1GeVf / ptCut) + (0.02f / drt_InSeg);
    pass_betaIn_cut = fabsf(betaInRHmin) < betaInCut;

    //Cut #5: first beta cut
    if(not(pass_betaIn_cut))
    {
        pass = false;
    }

    float betaAv = 0.5f * (betaIn + betaOut);
    float pt_beta = drt_tl_axis * k2Rinv1GeVf/sinf(betaAv);

    int lIn = 5;
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

    float sdIn_rt = hitsInGPU.rts[innerInnerAnchorHitIndex];
    float sdOut_rt = hitsInGPU.rts[outerInnerAnchorHitIndex];
    float sdIn_z = hitsInGPU.zs[innerInnerAnchorHitIndex];
    float sdOut_z = hitsInGPU.zs[outerInnerAnchorHitIndex];

    const float alphaInAbsReg = fmaxf(fabsf(alpha_InLo), asinf(fminf(sdIn_rt * k2Rinv1GeVf / 3.0f, sinAlphaMax)));
    const float alphaOutAbsReg = fmaxf(fabs(alpha_OutLo), asinf(fminf(sdOut_rt * k2Rinv1GeVf / 3.0f, sinAlphaMax)));
    const float dBetaInLum = lIn < 11 ? 0.0f : fabsf(alphaInAbsReg*deltaZLum / sdIn_z);
    const float dBetaOutLum = lOut < 11 ? 0.0f : fabsf(alphaOutAbsReg*deltaZLum / sdOut_z);
    const float dBetaLum2 = (dBetaInLum + dBetaOutLum) * (dBetaInLum + dBetaOutLum);
    const float sinDPhi = sinf(dPhi);

    const float dBetaRIn2 = 0; // TODO-RH
    // const float dBetaROut2 = 0; // TODO-RH
    float dBetaROut = 0;
    if(isEC_lastLayer)
    {
        dBetaROut = (sqrtf(hitsInGPU.highEdgeXs[outerOuterEdgeIndex] * hitsInGPU.highEdgeXs[outerOuterEdgeIndex] + hitsInGPU.highEdgeYs[outerOuterEdgeIndex] * hitsInGPU.highEdgeYs[outerOuterEdgeIndex]) - sqrtf(hitsInGPU.lowEdgeXs[outerOuterEdgeIndex] * hitsInGPU.lowEdgeXs[outerOuterEdgeIndex] + hitsInGPU.lowEdgeYs[outerOuterEdgeIndex] * hitsInGPU.lowEdgeYs[outerOuterEdgeIndex])) * sinDPhi / drt_tl_axis;
    }

    const float dBetaROut2 =  dBetaROut * dBetaROut;

    float betaOutCut = asinf(fminf(drt_tl_axis*k2Rinv1GeVf / ptCut, sinAlphaMax)) //FIXME: need faster version
        + (0.02f / sdOut_d) + sqrtf(dBetaLum2 + dBetaMuls*dBetaMuls);

    //Cut #6: The real beta cut
    if (not (fabsf(betaOut) < betaOutCut))
    {
        pass = false;
    }
    
    float pt_betaIn = drt_tl_axis * k2Rinv1GeVf/sinf(betaIn);
    float pt_betaOut = drt_tl_axis * k2Rinv1GeVf / sinf(betaOut);
    float dBetaRes = 0.02f/fminf(sdOut_d,drt_InSeg);
    float dBetaCut2 = (dBetaRes*dBetaRes * 2.0f + dBetaMuls * dBetaMuls + dBetaLum2 + dBetaRIn2 + dBetaROut2
            + 0.25 * (fabsf(betaInRHmin - betaInRHmax) + fabsf(betaOutRHmin - betaOutRHmax)) * (fabsf(betaInRHmin - betaInRHmax) + fabsf(betaOutRHmin - betaOutRHmax)));

    float dBeta = betaIn - betaOut;
    
    //Cut #7: Cut on dBeta
    if (not (dBeta * dBeta <= dBetaCut2))
    {
        pass = false;
    }

    return pass;
}

__device__ bool SDL::runTrackletDefaultAlgoBBEE(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, unsigned int innerInnerLowerModuleIndex, unsigned int innerOuterLowerModuleIndex, unsigned int outerInnerLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& dPhi, float& betaIn, float& betaOut)
{
    bool pass = true;
    bool isPS_InLo = (modulesInGPU.moduleType[innerInnerLowerModuleIndex] == SDL::PS);
    bool isPS_OutLo = (modulesInGPU.moduleType[outerInnerLowerModuleIndex] == SDL::PS);
    unsigned int innerInnerAnchorHitIndex = segmentsInGPU.innerMiniDoubletAnchorHitIndices[innerSegmentIndex];

    unsigned int outerInnerAnchorHitIndex = segmentsInGPU.innerMiniDoubletAnchorHitIndices[outerSegmentIndex];

    unsigned int innerOuterAnchorHitIndex = segmentsInGPU.outerMiniDoubletAnchorHitIndices[innerSegmentIndex];
    unsigned int outerOuterAnchorHitIndex= segmentsInGPU.outerMiniDoubletAnchorHitIndices[outerSegmentIndex];


    float rt_InLo = hitsInGPU.rts[innerInnerAnchorHitIndex];
    float rt_OutLo = hitsInGPU.rts[outerInnerAnchorHitIndex];
    float z_InLo = hitsInGPU.zs[innerInnerAnchorHitIndex];
    float z_OutLo = hitsInGPU.zs[outerInnerAnchorHitIndex];
    
    float alpha1GeV_OutLo = asinf(fminf(rt_OutLo * k2Rinv1GeVf / ptCut, sinAlphaMax));

    float rtRatio_OutLoInLo = rt_OutLo / rt_InLo; // Outer segment beginning rt divided by inner segment beginning rt;
    float dzDrtScale = tanf(alpha1GeV_OutLo) / alpha1GeV_OutLo; // The track can bend in r-z plane slightly
    float zpitch_InLo = (isPS_InLo ? pixelPSZpitch : strip2SZpitch);
    float zpitch_OutLo = (isPS_OutLo ? pixelPSZpitch : strip2SZpitch);
    float zGeom = zpitch_InLo + zpitch_OutLo;

    float zLo = z_InLo + (z_InLo - deltaZLum) * (rtRatio_OutLoInLo - 1.f) * (z_InLo > 0.f ? 1.f : dzDrtScale) - zGeom; 

    // Cut #0: Preliminary (Only here in endcap case)
    if(not(z_InLo * z_OutLo > 0))
    {
        pass = false;
    }
    float dLum = copysignf(deltaZLum, z_InLo);
    bool isOutSgInnerMDPS = modulesInGPU.moduleType[outerInnerLowerModuleIndex] == SDL::PS;
    float rtGeom1 = isOutSgInnerMDPS ? pixelPSZpitch : strip2SZpitch;
    float zGeom1 = copysignf(zGeom,z_InLo);
    float rtLo = rt_InLo * (1.f + (z_OutLo - z_InLo - zGeom1) / (z_InLo + zGeom1 + dLum) / dzDrtScale) - rtGeom1; //slope correction only on the lower end
    zOut = z_OutLo;
    rtOut = rt_OutLo;

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
    float rtHi = rt_InLo * (1.f + (z_OutLo - z_InLo + zGeom1) / zInForHi) + rtGeom1;

    //Cut #2: rt condition
    if (not (rt_OutLo >= rtLo and rt_OutLo <= rtHi))
    {
        pass = false;
    }

    float rIn = sqrtf(z_InLo * z_InLo + rt_InLo * rt_InLo);
    const float drtSDIn = hitsInGPU.rts[innerOuterAnchorHitIndex] - hitsInGPU.rts[innerInnerAnchorHitIndex];
    const float dzSDIn = hitsInGPU.zs[innerOuterAnchorHitIndex] - hitsInGPU.zs[innerInnerAnchorHitIndex];

    const float dr3SDIn = sqrtf(hitsInGPU.rts[innerOuterAnchorHitIndex] * hitsInGPU.rts[innerOuterAnchorHitIndex] +  hitsInGPU.zs[innerOuterAnchorHitIndex] * hitsInGPU.zs[innerOuterAnchorHitIndex]) - sqrtf(hitsInGPU.rts[innerInnerAnchorHitIndex] * hitsInGPU.rts[innerInnerAnchorHitIndex] +  hitsInGPU.zs[innerInnerAnchorHitIndex] *hitsInGPU.zs[innerInnerAnchorHitIndex]);

    const float coshEta = dr3SDIn / drtSDIn; //direction estimate
    const float dzOutInAbs = fabsf(z_OutLo - z_InLo);
    const float multDzDr = dzOutInAbs * coshEta / (coshEta * coshEta - 1.f);
    const float zGeom1_another = pixelPSZpitch; //What's this?
    const float kZ = (z_OutLo - z_InLo) / dzSDIn;
    float drtErr = zGeom1_another * zGeom1_another * drtSDIn * drtSDIn / dzSDIn / dzSDIn * (1.f - 2.f * kZ + 2.f * kZ * kZ); //Notes:122316
    const float sdlThetaMulsF = 0.015f * sqrtf(0.1f + 0.2 * (rt_OutLo - rt_InLo) / 50.f) * sqrtf(rIn / rt_InLo);
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
    const float sdlPVoff = 0.1f / rt_OutLo;
    const float sdlCut = alpha1GeV_OutLo + sqrtf(sdlMuls * sdlMuls + sdlPVoff*sdlPVoff);
    
    deltaPhiPos = deltaPhi(hitsInGPU.xs[innerOuterAnchorHitIndex], hitsInGPU.ys[innerOuterAnchorHitIndex], hitsInGPU.zs[innerOuterAnchorHitIndex], hitsInGPU.xs[outerOuterAnchorHitIndex], hitsInGPU.ys[outerOuterAnchorHitIndex], hitsInGPU.zs[outerOuterAnchorHitIndex]);

    //Cut #4: deltaPhiPos can be tighter
    if (not (fabsf(deltaPhiPos) <= sdlCut) )
    {
        pass = false;
    }

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
    
    float sdIn_alpha = segmentsInGPU.dPhiChanges[innerSegmentIndex];
    float sdIn_alpha_min = segmentsInGPU.dPhiChangeMins[innerSegmentIndex];
    float sdIn_alpha_max = segmentsInGPU.dPhiChangeMaxs[innerSegmentIndex];
    float sdOut_alpha = sdIn_alpha; //weird

    float sdOut_alphaOut = deltaPhi(hitsInGPU.xs[outerOuterAnchorHitIndex], hitsInGPU.ys[outerOuterAnchorHitIndex], hitsInGPU.zs[outerOuterAnchorHitIndex], hitsInGPU.xs[outerOuterAnchorHitIndex] - hitsInGPU.xs[outerInnerAnchorHitIndex], hitsInGPU.ys[outerOuterAnchorHitIndex] - hitsInGPU.ys[outerInnerAnchorHitIndex], hitsInGPU.zs[outerOuterAnchorHitIndex] - hitsInGPU.zs[outerInnerAnchorHitIndex]);/*outerOuterAnchor, outerOuterAnchor - outerInnerAnchor*/

    float sdOut_alphaOut_min = phi_mpi_pi(segmentsInGPU.dPhiChangeMins[outerSegmentIndex] - segmentsInGPU.dPhiMins[outerSegmentIndex]);
    float sdOut_alphaOut_max = phi_mpi_pi(segmentsInGPU.dPhiChangeMaxs[outerSegmentIndex] - segmentsInGPU.dPhiMaxs[outerSegmentIndex]);

    float tl_axis_x = hitsInGPU.xs[outerOuterAnchorHitIndex] - hitsInGPU.xs[innerInnerAnchorHitIndex];
    float tl_axis_y = hitsInGPU.ys[outerOuterAnchorHitIndex] - hitsInGPU.ys[innerInnerAnchorHitIndex];
    float tl_axis_z = hitsInGPU.zs[outerOuterAnchorHitIndex] - hitsInGPU.zs[innerInnerAnchorHitIndex];



    betaIn = sdIn_alpha - deltaPhi(hitsInGPU.xs[innerInnerAnchorHitIndex], hitsInGPU.ys[innerInnerAnchorHitIndex], hitsInGPU.zs[innerInnerAnchorHitIndex], tl_axis_x, tl_axis_y, tl_axis_z);

    float betaInRHmin = betaIn;
    float betaInRHmax = betaIn;

    betaOut = -sdOut_alphaOut + deltaPhi(hitsInGPU.xs[outerOuterAnchorHitIndex], hitsInGPU.ys[outerOuterAnchorHitIndex], hitsInGPU.zs[outerOuterAnchorHitIndex], tl_axis_x, tl_axis_y, tl_axis_z);

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
    
    float sdIn_dr = sqrtf((hitsInGPU.xs[innerOuterAnchorHitIndex] - hitsInGPU.xs[innerInnerAnchorHitIndex]) * (hitsInGPU.xs[innerOuterAnchorHitIndex] - hitsInGPU.xs[innerInnerAnchorHitIndex]) + (hitsInGPU.ys[innerOuterAnchorHitIndex] - hitsInGPU.ys[innerInnerAnchorHitIndex]) * (hitsInGPU.ys[innerOuterAnchorHitIndex] - hitsInGPU.ys[innerInnerAnchorHitIndex])); 
    float sdIn_d = hitsInGPU.rts[innerOuterAnchorHitIndex] - hitsInGPU.rts[innerInnerAnchorHitIndex];

    float dr = sqrtf(tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);
    const float corrF = 1.f;
    bool pass_betaIn_cut = false;
    float betaInCut = asinf(fminf((-sdIn_dr * corrF + dr) * k2Rinv1GeVf / ptCut, sinAlphaMax)) + (0.02f / sdIn_d);
    pass_betaIn_cut = fabsf(betaInRHmin) < betaInCut;

    //Cut #6: first beta cut
    if(not(pass_betaIn_cut))
    {
        pass = false;
    }

    float betaAv = 0.5f * (betaIn + betaOut);
    float pt_beta = dr * k2Rinv1GeVf / sinf(betaAv);

    float lIn = 5;
    float lOut = 11;

    float sdOut_dr = sqrtf((hitsInGPU.xs[outerOuterAnchorHitIndex] - hitsInGPU.xs[outerInnerAnchorHitIndex]) * (hitsInGPU.xs[outerOuterAnchorHitIndex] - hitsInGPU.xs[outerInnerAnchorHitIndex]) + (hitsInGPU.ys[outerOuterAnchorHitIndex] - hitsInGPU.ys[outerInnerAnchorHitIndex]) * (hitsInGPU.ys[outerOuterAnchorHitIndex] - hitsInGPU.ys[outerInnerAnchorHitIndex]));
    float sdOut_d = hitsInGPU.rts[outerOuterAnchorHitIndex] - hitsInGPU.rts[outerInnerAnchorHitIndex];
    
    runDeltaBetaIterations(betaIn, betaOut, betaAv, pt_beta, sdIn_dr, sdOut_dr, dr, lIn);

     const float betaInMMSF = (fabsf(betaInRHmin + betaInRHmax) > 0) ? (2.f * betaIn / fabsf(betaInRHmin + betaInRHmax)) : 0.; //mean value of min,max is the old betaIn
    const float betaOutMMSF = (fabsf(betaOutRHmin + betaOutRHmax) > 0) ? (2.f * betaOut / fabsf(betaOutRHmin + betaOutRHmax)) : 0.;
    betaInRHmin *= betaInMMSF;
    betaInRHmax *= betaInMMSF;
    betaOutRHmin *= betaOutMMSF;
    betaOutRHmax *= betaOutMMSF;

    const float dBetaMuls = sdlThetaMulsF * 4.f / fminf(fabsf(pt_beta), pt_betaMax); //need to confirm the range-out value of 7 GeV

    float sdIn_rt = hitsInGPU.rts[innerInnerAnchorHitIndex];
    float sdOut_rt = hitsInGPU.rts[outerInnerAnchorHitIndex];
    float sdIn_z = hitsInGPU.zs[innerInnerAnchorHitIndex];
    float sdOut_z = hitsInGPU.zs[outerInnerAnchorHitIndex];

    const float alphaInAbsReg = fmaxf(fabsf(sdIn_alpha), asinf(fminf(sdIn_rt * k2Rinv1GeVf / 3.0f, sinAlphaMax)));
    const float alphaOutAbsReg = fmaxf(fabsf(sdOut_alpha), asinf(fminf(sdOut_rt * k2Rinv1GeVf / 3.0f, sinAlphaMax)));
    const float dBetaInLum = lIn < 11 ? 0.0f : fabsf(alphaInAbsReg*deltaZLum / sdIn_z);
    const float dBetaOutLum = lOut < 11 ? 0.0f : fabsf(alphaOutAbsReg*deltaZLum / sdOut_z);
    const float dBetaLum2 = (dBetaInLum + dBetaOutLum) * (dBetaInLum + dBetaOutLum);
    const float sinDPhi = sinf(dPhi);

    const float dBetaRIn2 = 0; // TODO-RH
    // const float dBetaROut2 = 0; // TODO-RH
    float dBetaROut = 0;//sqrtf((hitsInGPU.xs[outerOuterAnchorHitIndex] - hitsInGPU.xs[outerInnerAnchorHitIndex]) * (hitsInGPU.xs[outerOuterAnchorHitIndex] - hitsInGPU.xs[outerInnerAnchorHitIndex]) + (hitsInGPU.ys[outerOuterAnchorHitIndex] - hitsInGPU.ys[outerInnerAnchorHitIndex]) * (hitsInGPU.ys[outerOuterAnchorHitIndex] - hitsInGPU.ys[outerInnerAnchorHitIndex])) * sinDPhi/drt_tl_axis;

    if(modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::TwoS)
    {
        unsigned int outerOuterEdgeIndex = hitsInGPU.edge2SMap[outerOuterAnchorHitIndex];
                //FIXME:might need to change to outer edge rt - inner edge rt
        dBetaROut = (sqrtf(hitsInGPU.highEdgeXs[outerOuterEdgeIndex] * hitsInGPU.highEdgeXs[outerOuterEdgeIndex] + hitsInGPU.highEdgeYs[outerOuterEdgeIndex] * hitsInGPU.highEdgeYs[outerOuterEdgeIndex]) - sqrtf(hitsInGPU.lowEdgeXs[outerOuterEdgeIndex] * hitsInGPU.lowEdgeXs[outerOuterEdgeIndex] + hitsInGPU.lowEdgeYs[outerOuterEdgeIndex] * hitsInGPU.lowEdgeYs[outerOuterEdgeIndex])) * sinDPhi / dr;

    }

    const float dBetaROut2 = dBetaROut * dBetaROut;
    float betaOutCut = asinf(fminf(dr*k2Rinv1GeVf / ptCut, sinAlphaMax)) //FIXME: need faster version
        + (0.02f / sdOut_d) + sqrtf(dBetaLum2 + dBetaMuls*dBetaMuls);

    //Cut #6: The real beta cut
    if (not (fabsf(betaOut) < betaOutCut))
    {
        pass = false;
    }

    float pt_betaIn = dr * k2Rinv1GeVf/sinf(betaIn);
    float pt_betaOut = dr * k2Rinv1GeVf / sinf(betaOut);
    float dBetaRes = 0.02f/fminf(sdOut_d,sdIn_d);
    float dBetaCut2 = (dBetaRes*dBetaRes * 2.0f + dBetaMuls * dBetaMuls + dBetaLum2 + dBetaRIn2 + dBetaROut2
            + 0.25 * (fabsf(betaInRHmin - betaInRHmax) + fabsf(betaOutRHmin - betaOutRHmax)) * (fabsf(betaInRHmin - betaInRHmax) + fabsf(betaOutRHmin - betaOutRHmax)));
    float dBeta = betaIn - betaOut;
    //Cut #7: Cut on dBeta
    if (not (dBeta * dBeta <= dBetaCut2))
    {
        pass = false;
    }

    return pass;
}

__device__ bool SDL::runTrackletDefaultAlgoEEEE(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, unsigned int innerInnerLowerModuleIndex, unsigned int innerOuterLowerModuleIndex, unsigned int outerInnerLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& dPhi, float& betaIn, float& betaOut)
{
    bool pass = true;
    
    bool isPS_InLo = (modulesInGPU.moduleType[innerInnerLowerModuleIndex] == SDL::PS);
    bool isPS_OutLo = (modulesInGPU.moduleType[outerInnerLowerModuleIndex] == SDL::PS);
    unsigned int innerInnerAnchorHitIndex = segmentsInGPU.innerMiniDoubletAnchorHitIndices[innerSegmentIndex];

    unsigned int outerInnerAnchorHitIndex = segmentsInGPU.innerMiniDoubletAnchorHitIndices[outerSegmentIndex];

    unsigned int innerOuterAnchorHitIndex = segmentsInGPU.outerMiniDoubletAnchorHitIndices[innerSegmentIndex];
    unsigned int outerOuterAnchorHitIndex= segmentsInGPU.outerMiniDoubletAnchorHitIndices[outerSegmentIndex];


    float rt_InLo = hitsInGPU.rts[innerInnerAnchorHitIndex];
    float rt_OutLo = hitsInGPU.rts[outerInnerAnchorHitIndex];
    float z_InLo = hitsInGPU.zs[innerInnerAnchorHitIndex];
    float z_OutLo = hitsInGPU.zs[outerInnerAnchorHitIndex];
    
    float alpha1GeV_OutLo = asinf(fminf(rt_OutLo * k2Rinv1GeVf / ptCut, sinAlphaMax));

    float rtRatio_OutLoInLo = rt_OutLo / rt_InLo; // Outer segment beginning rt divided by inner segment beginning rt;
    float dzDrtScale = tanf(alpha1GeV_OutLo) / alpha1GeV_OutLo; // The track can bend in r-z plane slightly
    float zpitch_InLo = (isPS_InLo ? pixelPSZpitch : strip2SZpitch);
    float zpitch_OutLo = (isPS_OutLo ? pixelPSZpitch : strip2SZpitch);
    float zGeom = zpitch_InLo + zpitch_OutLo;

    const float zLo = z_InLo + (z_InLo - deltaZLum) * (rtRatio_OutLoInLo - 1.f) * (z_InLo > 0.f ? 1.f : dzDrtScale) - zGeom; //slope-correction only on outer end

    // Cut #0: Preliminary (Only here in endcap case)
    if(not(z_InLo * z_OutLo > 0))
    {
        pass = false;
    }
    
    float dLum = copysignf(deltaZLum, z_InLo);
    bool isOutSgInnerMDPS = modulesInGPU.moduleType[outerInnerLowerModuleIndex] == SDL::PS;
    bool isInSgInnerMDPS = modulesInGPU.moduleType[innerInnerLowerModuleIndex] == SDL::PS;

    float rtGeom = (isInSgInnerMDPS and isOutSgInnerMDPS) ? 2.f * pixelPSZpitch : (isInSgInnerMDPS or isOutSgInnerMDPS) ? pixelPSZpitch + strip2SZpitch : 2.f * strip2SZpitch;

    float zGeom1 = copysignf(zGeom,z_InLo);
    float dz = z_OutLo - z_InLo;
    const float rtLo = rt_InLo * (1.f + dz / (z_InLo + dLum) / dzDrtScale) - rtGeom; //slope correction only on the lower end

    zOut = z_OutLo;
    rtOut = rt_OutLo;

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

    bool isInSgOuterMDPS = modulesInGPU.moduleType[innerOuterLowerModuleIndex] == SDL::PS;

    float drOutIn = rtOut - rt_InLo;
    float drtSDIn = hitsInGPU.rts[innerOuterAnchorHitIndex] - hitsInGPU.rts[innerInnerAnchorHitIndex];

    float dzSDIn = hitsInGPU.zs[innerOuterAnchorHitIndex] - hitsInGPU.zs[innerInnerAnchorHitIndex];

    float dr3SDIn = sqrtf(hitsInGPU.zs[innerOuterAnchorHitIndex] * hitsInGPU.zs[innerOuterAnchorHitIndex] + hitsInGPU.rts[innerOuterAnchorHitIndex] * hitsInGPU.rts[innerOuterAnchorHitIndex]) - sqrtf(hitsInGPU.zs[innerInnerAnchorHitIndex] * hitsInGPU.zs[innerInnerAnchorHitIndex] + hitsInGPU.rts[innerInnerAnchorHitIndex] * hitsInGPU.rts[innerInnerAnchorHitIndex]); 

    float coshEta = dr3SDIn / drtSDIn; //direction estimate
    float dzOutInAbs =  fabsf(z_OutLo - z_InLo);
    float multDzDr = dzOutInAbs * coshEta / (coshEta * coshEta - 1.f);

    float kZ = (z_OutLo - z_InLo) / dzSDIn;
    float sdlThetaMulsF = 0.015f * sqrtf(0.1f + 0.2 * (rt_OutLo - rt_InLo) / 50.f);

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

    float sdlPVoff = 0.1f/rtOut;
    float sdlCut = alpha1GeV_OutLo + sqrtf(sdlMuls * sdlMuls + sdlPVoff * sdlPVoff);

    //sdIn_mdOut_hit = innerOuterAnchorHitIndex
    //sdOut_mdOut_hit = outerOuterAnchorHitIndex

    deltaPhiPos = deltaPhi(hitsInGPU.xs[innerOuterAnchorHitIndex], hitsInGPU.ys[innerOuterAnchorHitIndex], hitsInGPU.zs[innerOuterAnchorHitIndex], hitsInGPU.xs[outerOuterAnchorHitIndex], hitsInGPU.ys[outerOuterAnchorHitIndex], hitsInGPU.zs[outerOuterAnchorHitIndex]);

    if(not(fabsf(deltaPhiPos) <= sdlCut))
    {
        pass = false;
    }
    
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

    float sdIn_alpha = segmentsInGPU.dPhiChanges[innerSegmentIndex];
    float sdOut_alpha = sdIn_alpha; //weird
    float sdOut_dPhiPos = deltaPhi(hitsInGPU.xs[outerInnerAnchorHitIndex], hitsInGPU.ys[outerInnerAnchorHitIndex], hitsInGPU.zs[outerInnerAnchorHitIndex], hitsInGPU.xs[outerOuterAnchorHitIndex], hitsInGPU.ys[outerOuterAnchorHitIndex], hitsInGPU.zs[outerOuterAnchorHitIndex]);
    float sdOut_dPhiChange = segmentsInGPU.dPhiChanges[outerSegmentIndex];
    float sdOut_dPhiChange_min = segmentsInGPU.dPhiChangeMins[outerSegmentIndex];
    float sdOut_dPhiChange_max = segmentsInGPU.dPhiChangeMaxs[outerSegmentIndex];

    float sdOut_alphaOutRHmin = phi_mpi_pi(sdOut_dPhiChange_min - sdOut_dPhiPos);
    float sdOut_alphaOutRHmax = phi_mpi_pi(sdOut_dPhiChange_max - sdOut_dPhiPos);
    float sdOut_alphaOut = phi_mpi_pi(sdOut_dPhiChange - sdOut_dPhiPos);

    float tl_axis_x = hitsInGPU.xs[outerOuterAnchorHitIndex] - hitsInGPU.xs[innerInnerAnchorHitIndex];
    float tl_axis_y = hitsInGPU.ys[outerOuterAnchorHitIndex] - hitsInGPU.ys[innerInnerAnchorHitIndex];
    float tl_axis_z = hitsInGPU.zs[outerOuterAnchorHitIndex] - hitsInGPU.zs[innerInnerAnchorHitIndex];

    betaIn = sdIn_alpha - deltaPhi(hitsInGPU.xs[innerInnerAnchorHitIndex], hitsInGPU.ys[innerInnerAnchorHitIndex], hitsInGPU.zs[innerInnerAnchorHitIndex], tl_axis_x, tl_axis_y, tl_axis_z);
    float sdIn_alphaRHmin = segmentsInGPU.dPhiChangeMins[innerSegmentIndex];
    float sdIn_alphaRHmax = segmentsInGPU.dPhiChangeMaxs[innerSegmentIndex];

    float betaInRHmin = betaIn + sdIn_alphaRHmin - sdIn_alpha;
    float betaInRHmax = betaIn + sdIn_alphaRHmax - sdIn_alpha;

    betaOut = -sdOut_alphaOut + deltaPhi(hitsInGPU.xs[outerOuterAnchorHitIndex], hitsInGPU.ys[outerOuterAnchorHitIndex], hitsInGPU.zs[outerOuterAnchorHitIndex], tl_axis_x, tl_axis_y, tl_axis_z);

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

    float sdIn_dr = sqrtf((hitsInGPU.xs[innerOuterAnchorHitIndex] - hitsInGPU.xs[innerInnerAnchorHitIndex]) * (hitsInGPU.xs[innerOuterAnchorHitIndex] - hitsInGPU.xs[innerInnerAnchorHitIndex]) + (hitsInGPU.ys[innerOuterAnchorHitIndex] - hitsInGPU.ys[innerInnerAnchorHitIndex]) * (hitsInGPU.ys[innerOuterAnchorHitIndex] - hitsInGPU.ys[innerInnerAnchorHitIndex])); 
    float sdIn_d = hitsInGPU.rts[innerOuterAnchorHitIndex] - hitsInGPU.rts[innerInnerAnchorHitIndex];

    float dr = sqrtf(tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);
    const float corrF = 1.f;
    bool pass_betaIn_cut = false;
    float betaInCut = asinf(fminf((-sdIn_dr * corrF + dr) * k2Rinv1GeVf / ptCut, sinAlphaMax)) + (0.02f / sdIn_d);
    pass_betaIn_cut = fabsf(betaInRHmin) < betaInCut;

    //Cut #6: first beta cut
    if(not(pass_betaIn_cut))
    {
        pass = false;
    }

    float betaAv = 0.5f * (betaIn + betaOut);
    float pt_beta = dr * k2Rinv1GeVf / sinf(betaAv);


    int lIn= 11; //endcap
    int lOut = 13; //endcap

    float sdOut_dr = sqrtf((hitsInGPU.xs[outerOuterAnchorHitIndex] - hitsInGPU.xs[outerInnerAnchorHitIndex]) * (hitsInGPU.xs[outerOuterAnchorHitIndex] - hitsInGPU.xs[outerInnerAnchorHitIndex]) + (hitsInGPU.ys[outerOuterAnchorHitIndex] - hitsInGPU.ys[outerInnerAnchorHitIndex]) * (hitsInGPU.ys[outerOuterAnchorHitIndex] - hitsInGPU.ys[outerInnerAnchorHitIndex]));
    float sdOut_d = hitsInGPU.rts[outerOuterAnchorHitIndex] - hitsInGPU.rts[outerInnerAnchorHitIndex];

    float diffDr = fabsf(sdIn_dr - sdOut_dr)/fabs(sdIn_dr + sdOut_dr);
    
    runDeltaBetaIterations(betaIn, betaOut, betaAv, pt_beta, sdIn_dr, sdOut_dr, dr, lIn);

     const float betaInMMSF = (fabsf(betaInRHmin + betaInRHmax) > 0) ? (2.f * betaIn / fabsf(betaInRHmin + betaInRHmax)) : 0.; //mean value of min,max is the old betaIn
    const float betaOutMMSF = (fabsf(betaOutRHmin + betaOutRHmax) > 0) ? (2.f * betaOut / fabsf(betaOutRHmin + betaOutRHmax)) : 0.;
    betaInRHmin *= betaInMMSF;
    betaInRHmax *= betaInMMSF;
    betaOutRHmin *= betaOutMMSF;
    betaOutRHmax *= betaOutMMSF;

    const float dBetaMuls = sdlThetaMulsF * 4.f / fminf(fabsf(pt_beta), pt_betaMax); //need to confirm the range-out value of 7 GeV

    float sdIn_rt = hitsInGPU.rts[innerInnerAnchorHitIndex];
    float sdOut_rt = hitsInGPU.rts[outerInnerAnchorHitIndex];
    float sdIn_z = hitsInGPU.zs[innerInnerAnchorHitIndex];
    float sdOut_z = hitsInGPU.zs[outerInnerAnchorHitIndex];

    const float alphaInAbsReg = fmaxf(fabsf(sdIn_alpha), asinf(fminf(sdIn_rt * k2Rinv1GeVf / 3.0f, sinAlphaMax)));
    const float alphaOutAbsReg = fmaxf(fabsf(sdOut_alpha), asinf(fminf(sdOut_rt * k2Rinv1GeVf / 3.0f, sinAlphaMax)));
    const float dBetaInLum = lIn < 11 ? 0.0f : fabsf(alphaInAbsReg*deltaZLum / sdIn_z);
    const float dBetaOutLum = lOut < 11 ? 0.0f : fabsf(alphaOutAbsReg*deltaZLum / sdOut_z);
    const float dBetaLum2 = (dBetaInLum + dBetaOutLum) * (dBetaInLum + dBetaOutLum);
    const float sinDPhi = sinf(dPhi);

    const float dBetaRIn2 = 0; // TODO-RH
    // const float dBetaROut2 = 0; // TODO-RH
    float dBetaROut2 = 0;//TODO-RH
    float betaOutCut = asinf(fminf(dr*k2Rinv1GeVf / ptCut, sinAlphaMax)) //FIXME: need faster version
        + (0.02f / sdOut_d) + sqrtf(dBetaLum2 + dBetaMuls*dBetaMuls);

    //Cut #6: The real beta cut
    if (not (fabsf(betaOut) < betaOutCut))
    {
        pass = false;
    }

    float pt_betaIn = dr * k2Rinv1GeVf/sinf(betaIn);
    float pt_betaOut = dr * k2Rinv1GeVf / sinf(betaOut);
    float dBetaRes = 0.02f/fminf(sdOut_d,sdIn_d);
    float dBetaCut2 = (dBetaRes*dBetaRes * 2.0f + dBetaMuls * dBetaMuls + dBetaLum2 + dBetaRIn2 + dBetaROut2
            + 0.25 * (fabsf(betaInRHmin - betaInRHmax) + fabsf(betaOutRHmin - betaOutRHmax)) * (fabsf(betaInRHmin - betaInRHmax) + fabsf(betaOutRHmin - betaOutRHmax)));
    float dBeta = betaIn - betaOut;
    //Cut #7: Cut on dBeta
    if (not (dBeta * dBeta <= dBetaCut2))
    {
        pass = false;
    }


    return pass;
}


__device__ bool SDL::runTrackletDefaultAlgoPPBB(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU ,struct segments& segmentsInGPU, unsigned int pixelModuleIndex, unsigned int outerInnerLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, float& zOut, float& rtOut, float& dPhiPos, float& dPhi, float& betaIn, float& betaOut, unsigned int
        N_MAX_SEGMENTS_PER_MODULE) // pixel to BB and BE segments
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
    float zHi = z_InOut + (z_InOut + deltaZLum) * (rtRatio_OutLoInOut - 1.f) * (z_InOut < 0.f ? 1.f : dzDrtScale) + (zpitch_InOut + zpitch_OutLo);
    float zLo = z_InOut + (z_InOut - deltaZLum) * (rtRatio_OutLoInOut - 1.f) * (z_InOut > 0.f ? 1.f : dzDrtScale) - (zpitch_InOut + zpitch_OutLo); //slope-correction only on outer end

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
    float zLoPointed = z_InUp + dzMean - zWindow;
    float zHiPointed = z_InUp + dzMean + zWindow;

    if (not (z_OutLo >= zLoPointed and z_OutLo <= zHiPointed))
    {
        pass = false;
    }
    const float sdlPVoff = 0.1f / rt_OutLo;
    float sdlCut = alpha1GeV_OutLo + sqrtf(sdlMuls * sdlMuls + sdlPVoff * sdlPVoff);

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

    unsigned int outerOuterEdgeIndex = hitsInGPU.edge2SMap[outerOuterAnchorHitIndex]; //POTENTIAL NUCLEAR GANDHI
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
    float pt_beta = ptIn;

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

    float betaOutCut = asinf(fminf(drt_tl_axis*k2Rinv1GeVf / ptCut, sinAlphaMax)) //FIXME: need faster version
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
    float deltaBetaCut = sqrtf(dBetaCut2);
    if (not (dBeta * dBeta <= dBetaCut2))
    {
        //printf("dBeta2 = %f, dBetaCut2 = %f\n",dBeta * dBeta, dBetaCut2);
        pass = false;
    }

    return pass;
}

__device__ bool SDL::runTrackletDefaultAlgoPPEE(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU ,struct segments& segmentsInGPU, unsigned int pixelModuleIndex, unsigned int outerInnerLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& dPhi, float& betaIn, float& betaOut, unsigned int
        N_MAX_SEGMENTS_PER_MODULE) // pixel to EE segments
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
    float zLo = zIn + (zIn - deltaZLum) * (rtOut_o_rtIn - 1.f) * (zIn > 0.f ? 1.f : dzDrtScale) - zGeom; //slope-correction only on outer end

    if (not (zIn * zOut > 0))
    {
        pass = false;
    }

    const float dLum = copysignf(deltaZLum, zIn);
    bool isOutSgInnerMDPS = modulesInGPU.moduleType[outerInnerLowerModuleIndex] == SDL::PS;

    const float rtGeom1 = isOutSgInnerMDPS ? pixelPSZpitch : strip2SZpitch;//FIXME: make this chosen by configuration for lay11,12 full PS
    const float zGeom1 = copysignf(zGeom, zIn); //used in B-E region
    float rtLo = rtIn * (1.f + (zOut - zIn - zGeom1) / (zIn + zGeom1 + dLum) / dzDrtScale) - rtGeom1; //slope correction only on the lower end

    if (not (rtOut >= rtLo))
    {
        pass = false;
    }

    float zInForHi = zIn - zGeom1 - dLum;
    if (zInForHi * zIn < 0)
        zInForHi = copysignf(0.1f, zIn);
    float rtHi = rtIn * (1.f + (zOut - zIn + zGeom1) / zInForHi) + rtGeom1;

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
    float sdlCut = alpha1GeV_OutLo + sqrtf(sdlMuls * sdlMuls + sdlPVoff * sdlPVoff);

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

    unsigned int outerOuterEdgeIndex = hitsInGPU.edge2SMap[outerOuterAnchorHitIndex]; //POTENTIAL NUCLEAR GANDHI
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
    float pt_beta = ptIn;

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

    float betaOutCut = asinf(fminf(drt_tl_axis*k2Rinv1GeVf / ptCut, sinAlphaMax)) //FIXME: need faster version
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
    float deltaBetaCut = sqrtf(dBetaCut2);
    if (not (dBeta * dBeta <= dBetaCut2))
    {
        pass = false;
    }

    return pass;
}

__device__ void SDL::runDeltaBetaIterations(float& betaIn, float& betaOut, float& betaAv, float & pt_beta, float sdIn_dr, float sdOut_dr, float dr, float lIn)
{
    if (lIn == 0)
    {
        betaOut += copysign(asinf(fminf(sdOut_dr * k2Rinv1GeVf / fabsf(pt_beta), sinAlphaMax)), betaOut);
        return;
    }

    if (true //do it for all//diffDr > 0.05 //only if segment length is different significantly
            and betaIn * betaOut > 0.f
            and (fabsf(pt_beta) < 4.f * pt_betaMax
                or (lIn >= 11 && fabsf(pt_beta) < 8.f * pt_betaMax)))   //and the pt_beta is well-defined; less strict for endcap-endcap
    {

        const float betaInUpd  = betaIn + copysignf(asinf(fminf(sdIn_dr * k2Rinv1GeVf / fabsf(pt_beta), sinAlphaMax)), betaIn); //FIXME: need a faster version
        const float betaOutUpd = betaOut + copysignf(asinf(fminf(sdOut_dr * k2Rinv1GeVf / fabsf(pt_beta), sinAlphaMax)), betaOut); //FIXME: need a faster version
        betaAv = 0.5f * (betaInUpd + betaOutUpd);

        //1st update
        pt_beta = dr * k2Rinv1GeVf / sin(betaAv); //get a better pt estimate

        betaIn  += copysignf(asinf(fminf(sdIn_dr * k2Rinv1GeVf / fabsf(pt_beta), sinAlphaMax)), betaIn); //FIXME: need a faster version
        betaOut += copysignf(asinf(fminf(sdOut_dr * k2Rinv1GeVf / fabsf(pt_beta), sinAlphaMax)), betaOut); //FIXME: need a faster version
        //update the av and pt
        betaAv = 0.5f * (betaIn + betaOut);
        //2nd update
        pt_beta = dr * k2Rinv1GeVf / sin(betaAv); //get a better pt estimate

        //might need these for future iterations

/*
        setRecoVars("betaIn_2nd", betaIn);
        setRecoVars("betaOut_2nd", betaOut);
        setRecoVars("betaAv_2nd", betaAv);
        setRecoVars("betaPt_2nd", pt_beta);
        setRecoVars("betaIn_3rdCorr", copysignf(asinf(fminf(sdIn_dr * k2Rinv1GeVf / fabsf(pt_beta), sinAlphaMax)), betaIn));
        setRecoVars("betaOut_3rdCorr", copysignf(asinf(fminf(sdOut_dr * k2Rinv1GeVf / fabsf(pt_beta), sinAlphaMax)), betaOut));
        setRecoVars("dBeta_2nd", betaIn - betaOut);

        setRecoVars("betaIn_3rd", getRecoVar("betaIn_0th") + copysignf(asinf(fminf(sdIn_dr * k2Rinv1GeVf / fabsf(pt_beta), sinAlphaMax)), betaIn));
        setRecoVars("betaOut_3rd", getRecoVar("betaOut_0th") + copysignf(asinf(fminf(sdOut_dr * k2Rinv1GeVf / fabsf(pt_beta), sinAlphaMax)), betaOut));
        setRecoVars("betaAv_3rd", 0.5f * (getRecoVar("betaIn_3rd") + getRecoVar("betaOut_3rd")));
        setRecoVars("betaPt_3rd", dr * k2Rinv1GeVf / sin(getRecoVar("betaAv_3rd")));
        setRecoVars("dBeta_3rd", getRecoVar("betaIn_3rd") - getRecoVar("betaOut_3rd"));

        setRecoVars("betaIn_4th", getRecoVar("betaIn_0th") + copysignf(asinf(fminf(sdIn_dr * k2Rinv1GeVf / fabsf(getRecoVar("betaPt_3rd")), sinAlphaMax)), getRecoVar("betaIn_3rd")));
        setRecoVars("betaOut_4th", getRecoVar("betaOut_0th") + copysignf(asinf(fminf(sdOut_dr * k2Rinv1GeVf / fabsf(getRecoVar("betaPt_3rd")), sinAlphaMax)), getRecoVar("betaOut_3rd")));
        setRecoVars("betaAv_4th", 0.5f * (getRecoVar("betaIn_4th") + getRecoVar("betaOut_4th")));
        setRecoVars("betaPt_4th", dr * k2Rinv1GeVf / sin(getRecoVar("betaAv_4th")));
        setRecoVars("dBeta_4th", getRecoVar("betaIn_4th") - getRecoVar("betaOut_4th"));*/


    }
    else if (lIn < 11 && fabsf(betaOut) < 0.2 * fabsf(betaIn) && fabsf(pt_beta) < 12.f * pt_betaMax)   //use betaIn sign as ref
    {
   
        const float pt_betaIn = dr * k2Rinv1GeVf / sin(betaIn);
        const float betaInUpd  = betaIn + copysignf(asinf(fminf(sdIn_dr * k2Rinv1GeVf / fabs(pt_betaIn), sinAlphaMax)), betaIn); //FIXME: need a faster version
        const float betaOutUpd = betaOut + copysignf(asinf(fminf(sdOut_dr * k2Rinv1GeVf / fabs(pt_betaIn), sinAlphaMax)), betaIn); //FIXME: need a faster version
        betaAv = (fabsf(betaOut) > 0.2f * fabsf(betaIn)) ? (0.5f * (betaInUpd + betaOutUpd)) : betaInUpd;

        //1st update
        pt_beta = dr * k2Rinv1GeVf / sin(betaAv); //get a better pt estimate
        betaIn  += copysignf(asinf(fminf(sdIn_dr * k2Rinv1GeVf / fabsf(pt_beta), sinAlphaMax)), betaIn); //FIXME: need a faster version
        betaOut += copysignf(asinf(fminf(sdOut_dr * k2Rinv1GeVf / fabsf(pt_beta), sinAlphaMax)), betaIn); //FIXME: need a faster version
        //update the av and pt
        betaAv = 0.5f * (betaIn + betaOut);
        //2nd update
        pt_beta = dr * k2Rinv1GeVf / sin(betaAv); //get a better pt estimate

    }
}

void SDL::printTracklet(struct SDL::tracklets& trackletsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::hits& hitsInGPU, struct SDL::modules& modulesInGPU, unsigned int trackletIndex)
{
    unsigned int innerSegmentIndex  = trackletsInGPU.segmentIndices[trackletIndex * 2];
    unsigned int outerSegmentIndex = trackletsInGPU.segmentIndices[trackletIndex * 2 + 1];

    std::cout<<std::endl;
    std::cout<<"tl_betaIn : "<<trackletsInGPU.betaIn[trackletIndex] << std::endl;
    std::cout<<"tl_betaOut : "<<trackletsInGPU.betaOut[trackletIndex] << std::endl;

    std::cout<<"Inner Segment"<<std::endl;
    std::cout << "------------------------------" << std::endl;
    {
        IndentingOStreambuf indent(std::cout);
        printSegment(segmentsInGPU,mdsInGPU, hitsInGPU, modulesInGPU, innerSegmentIndex);
    }

    std::cout<<"Outer Segment"<<std::endl;
    std::cout << "------------------------------" << std::endl;
    {
        IndentingOStreambuf indent(std::cout);
        printSegment(segmentsInGPU,mdsInGPU, hitsInGPU, modulesInGPU, outerSegmentIndex);
    }
}
