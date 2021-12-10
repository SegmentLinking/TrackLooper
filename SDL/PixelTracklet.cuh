#ifndef Pixel_Tracklet_cuh
#define Pixel_Tracklet_cuh

#ifdef __CUDACC__
#define CUDA_HOSTDEV  __host__ __device__
#define CUDA_DEV __device__
#define CUDA_CONST_VAR __device__
#else
#define CUDA_HOSTDEV
#define CUDA_DEV
#define CUDA_CONST_VAR
#endif

#include "Constants.h"
#include "EndcapGeometry.h"
#include "TiltedGeometry.h"
#include "Segment.cuh"
#include "MiniDoublet.cuh"
#include "Module.cuh"
#include "Hit.cuh"
#include "PrintUtil.h"
#include "Tracklet.cuh"


namespace SDL
{
    struct pixelTracklets
    {
        unsigned int* segmentIndices;
        unsigned int* lowerModuleIndices;
        unsigned int* nPixelTracklets;
        float* zOut;
        float* rtOut;

        float* deltaPhiPos;
        float* deltaPhi;
        float* betaIn;
        float* betaOut;
        float* pt_beta;

#ifdef CUT_VALUE_DEBUG
        //debug variables
        float* zLo;
        float* zHi;
        float* zLoPointed;
        float* zHiPointed;
        float* sdlCut;
        float* betaInCut;
        float* betaOutCut;
        float* deltaBetaCut;
        float* rtLo;
        float* rtHi;
        float* kZ;
#endif
        
        pixelTracklets();
        ~pixelTracklets();
        void freeMemory();
        void freeMemoryCache();
    };

    void createPixelTrackletsInUnifiedMemory(struct pixelTracklets& pixelTrackletsInGPU, unsigned int maxPixelTracklets,cudaStream_t stream);
    void createPixelTrackletsInExplicitMemory(struct pixelTracklets& pixelTrackletsInGPU, unsigned int maxPixelTracklets,cudaStream_t stream);

#ifdef CUT_VALUE_DEBUG
    CUDA_DEV void addPixelTrackletToMemory(struct pixelTracklets& pixelTrackletsInGPU, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, unsigned int innerInnerLowerModuleIndex, unsigned int innerOuterLowerModuleIndex, unsigned int outerInnerLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& deltaPhi, float& betaIn, float& betaOut, float pt_beta, float& zLo, float& zHi, float& rtLo, float& rtHi, float& zLoPointed, float&
        zHiPointed, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ, unsigned int pixelTrackletIndex);

#else
    CUDA_DEV void addPixelTrackletToMemory(struct pixelTracklets& pixelTrackletsInGPU, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, unsigned int innerInnerLowerModuleIndex, unsigned int innerOuterLowerModuleIndex, unsigned int outerInnerLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& deltaPhi, float& betaIn, float& betaOut, float pt_beta, unsigned int pixelTrackletIndex);
#endif

//    CUDA_DEV bool runPixelTrackletDefaultAlgo(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, unsigned int innerInnerLowerModuleIndex, unsigned int innerOuterLowerModuleIndex, unsigned int outerInnerLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& deltaPhi, float& betaIn, float&
//        betaOut, float& pt_beta, float& zLo, float& zHi, float& rtLo, float& rtHi, float& zLoPointed, float& zHiPointed, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ, unsigned int N_MAX_SEGMENTS_PER_MODULE = 600);
//
//
//    CUDA_DEV bool runTrackletDefaultAlgoPPBB(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU ,struct segments& segmentsInGPU, unsigned int pixelModuleIndex, unsigned int outerInnerLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, float& zOut, float& rtOut, float& dPhiPos, float& dPhi, float& betaIn, float& betaOut, float& pt_beta, unsigned int
//        N_MAX_SEGMENTS_PER_MODULE, float& zLo, float& zHi, float& zLoPointed, float& zHiPointed, float& sdlCut, float& betaOutCut, float& deltaBetaCut); // pixel to BB and BE segments
//
//    CUDA_DEV bool runTrackletDefaultAlgoPPEE(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU ,struct segments& segmentsInGPU, unsigned int pixelModuleIndex, unsigned int outerInnerLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& dPhi, float& betaIn, float& betaOut, float& pt_beta, unsigned int
//        N_MAX_SEGMENTS_PER_MODULE,  float& zLo, float& rtLo, float& rtHi, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ); // pixel to EE segments



CUDA_DEV bool inline runTrackletDefaultAlgoPPBB(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU ,struct segments& segmentsInGPU, unsigned int pixelModuleIndex, unsigned int outerInnerLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, float& zOut, float& rtOut, float& dPhiPos, float& dPhi, float& betaIn, float& betaOut, float& pt_beta, unsigned int
        N_MAX_SEGMENTS_PER_MODULE, float& zLo, float& zHi, float& zLoPointed, float& zHiPointed, float& sdlCut, float& betaOutCut, float& deltaBetaCut) // pixel to BB and BE segments
{
    bool pass = true;

    bool isPS_OutLo = (modulesInGPU.moduleType[outerInnerLowerModuleIndex] == SDL::PS);
    unsigned int innerInnerAnchorHitIndex = segmentsInGPU.innerMiniDoubletAnchorHitIndices[innerSegmentIndex];

    unsigned int outerInnerAnchorHitIndex = segmentsInGPU.innerMiniDoubletAnchorHitIndices[outerSegmentIndex];

    unsigned int innerOuterAnchorHitIndex = segmentsInGPU.outerMiniDoubletAnchorHitIndices[innerSegmentIndex];
    unsigned int outerOuterAnchorHitIndex= segmentsInGPU.outerMiniDoubletAnchorHitIndices[outerSegmentIndex];
    if(fabsf(deltaPhi(hitsInGPU.xs[innerOuterAnchorHitIndex], hitsInGPU.ys[innerOuterAnchorHitIndex], hitsInGPU.zs[innerOuterAnchorHitIndex], hitsInGPU.xs[outerInnerAnchorHitIndex], hitsInGPU.ys[outerInnerAnchorHitIndex], hitsInGPU.zs[outerInnerAnchorHitIndex])) > 0.5f*float(M_PI))
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
    zHi = z_InOut + (z_InOut + deltaZLum) * (rtRatio_OutLoInOut - 1.f) * (z_InOut < 0.f ? 1.f : dzDrtScale) + (zpitch_InOut + zpitch_OutLo);
    zLo = z_InOut + (z_InOut - deltaZLum) * (rtRatio_OutLoInOut - 1.f) * (z_InOut > 0.f ? 1.f : dzDrtScale) - (zpitch_InOut + zpitch_OutLo); //slope-correction only on outer end

    if (not (z_OutLo >= zLo and z_OutLo <= zHi))
    {
        pass = false;
    }
    const float coshEta = sqrtf(ptIn * ptIn + pz * pz) / ptIn;
    // const float drt_OutLo_InLo = (rt_OutLo - rt_InLo);
    const float drt_OutLo_InUp = (rt_OutLo - rt_InUp);
    const float invRt_InLo = 1.f / rt_InLo;
    const float r3_InLo = sqrtf(z_InLo * z_InLo + rt_InLo * rt_InLo);
    const float r3_InUp = sqrtf(z_InUp * z_InUp + rt_InUp * rt_InUp);
    float drt_InSeg = hitsInGPU.rts[innerOuterAnchorHitIndex] - hitsInGPU.rts[innerInnerAnchorHitIndex];
    float dz_InSeg = hitsInGPU.zs[innerOuterAnchorHitIndex] - hitsInGPU.zs[innerInnerAnchorHitIndex];

    float dr3_InSeg = sqrtf(hitsInGPU.rts[innerOuterAnchorHitIndex] * hitsInGPU.rts[innerOuterAnchorHitIndex] + hitsInGPU.zs[innerOuterAnchorHitIndex] * hitsInGPU.zs[innerOuterAnchorHitIndex]) - sqrtf(hitsInGPU.rts[innerInnerAnchorHitIndex] * hitsInGPU.rts[innerInnerAnchorHitIndex] + hitsInGPU.zs[innerInnerAnchorHitIndex] * hitsInGPU.zs[innerInnerAnchorHitIndex]);
    const float sdlThetaMulsF = 0.015f * sqrt(0.1f + 0.2f * (rt_OutLo - rt_InUp) / 50.f) * sqrt(r3_InUp / rt_InUp);
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
    zLoPointed = z_InUp + dzMean - zWindow;
    zHiPointed = z_InUp + dzMean + zWindow;

    if (not (z_OutLo >= zLoPointed and z_OutLo <= zHiPointed))
    {
        pass = false;
    }
    const float sdlPVoff = 0.1f / rt_OutLo;
    sdlCut = alpha1GeV_OutLo + sqrtf(sdlMuls * sdlMuls + sdlPVoff * sdlPVoff);

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

    //unsigned int outerOuterEdgeIndex = hitsInGPU.edge2SMap[outerOuterAnchorHitIndex]; //POTENTIAL NUCLEAR GANDHI
    unsigned int outerOuterEdgeIndex = outerOuterAnchorHitIndex;

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
    pt_beta = ptIn;

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

    betaOutCut = asinf(fminf(drt_tl_axis*k2Rinv1GeVf / ptCut, sinAlphaMax)) //FIXME: need faster version
        + (0.02f / sdOut_d) + sqrtf(dBetaLum2 + dBetaMuls*dBetaMuls);

    //Cut #6: The real beta cut
    if (not (fabsf(betaOut) < betaOutCut))
    {
        pass = false;
    }

    const float pt_betaOut = drt_tl_axis * k2Rinv1GeVf / sin(betaOut);
    const float dBetaRes = 0.02f / fminf(sdOut_d, drt_InSeg);
    const float dBetaCut2 = (dBetaRes*dBetaRes * 2.0f + dBetaMuls * dBetaMuls + dBetaLum2 + dBetaRIn2 + dBetaROut2
            + 0.25f * (fabsf(betaInRHmin - betaInRHmax) + fabsf(betaOutRHmin - betaOutRHmax)) * (fabsf(betaInRHmin - betaInRHmax) + fabsf(betaOutRHmin - betaOutRHmax)));
    float dBeta = betaIn - betaOut;
    deltaBetaCut = sqrtf(dBetaCut2);
    if (not (dBeta * dBeta <= dBetaCut2))
    {
        //printf("dBeta2 = %f, dBetaCut2 = %f\n",dBeta * dBeta, dBetaCut2);
        pass = false;
    }

    return pass;
}

CUDA_DEV bool inline runTrackletDefaultAlgoPPEE(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU ,struct segments& segmentsInGPU, unsigned int pixelModuleIndex, unsigned int outerInnerLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& dPhi, float& betaIn, float& betaOut, float& pt_beta, unsigned int
        N_MAX_SEGMENTS_PER_MODULE,  float& zLo, float& rtLo, float& rtHi, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ) // pixel to EE segments
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
    zLo = zIn + (zIn - deltaZLum) * (rtOut_o_rtIn - 1.f) * (zIn > 0.f ? 1.f : dzDrtScale) - zGeom; //slope-correction only on outer end

    if (not (zIn * zOut > 0))
    {
        pass = false;
    }

    const float dLum = copysignf(deltaZLum, zIn);
    bool isOutSgInnerMDPS = modulesInGPU.moduleType[outerInnerLowerModuleIndex] == SDL::PS;

    const float rtGeom1 = isOutSgInnerMDPS ? pixelPSZpitch : strip2SZpitch;//FIXME: make this chosen by configuration for lay11,12 full PS
    const float zGeom1 = copysignf(zGeom, zIn); //used in B-E region
    rtLo = rtIn * (1.f + (zOut - zIn - zGeom1) / (zIn + zGeom1 + dLum) / dzDrtScale) - rtGeom1; //slope correction only on the lower end

    if (not (rtOut >= rtLo))
    {
        pass = false;
    }

    float zInForHi = zIn - zGeom1 - dLum;
    if (zInForHi * zIn < 0)
        zInForHi = copysignf(0.1f, zIn);
    rtHi = rtIn * (1.f + (zOut - zIn + zGeom1) / zInForHi) + rtGeom1;

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
    const float sdlThetaMulsF = 0.015f * sqrtf(0.1f + 0.2f * (rt_OutLo - rtIn) / 50.f) * sqrtf(r3_InUp / rtIn);
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
    sdlCut = alpha1GeV_OutLo + sqrtf(sdlMuls * sdlMuls + sdlPVoff * sdlPVoff);

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

    unsigned int outerOuterEdgeIndex = outerOuterAnchorHitIndex;

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
    pt_beta = ptIn;

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

    betaOutCut = asinf(fminf(drt_tl_axis*k2Rinv1GeVf / ptCut, sinAlphaMax)) //FIXME: need faster version
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
            + 0.25f * (fabsf(betaInRHmin - betaInRHmax) + fabsf(betaOutRHmin - betaOutRHmax)) * (fabsf(betaInRHmin - betaInRHmax) + fabsf(betaOutRHmin - betaOutRHmax)));
    float dBeta = betaIn - betaOut;
    deltaBetaCut = sqrtf(dBetaCut2);
    if (not (dBeta * dBeta <= dBetaCut2))
    {
        pass = false;
    }

    return pass;
}
        CUDA_DEV bool inline runPixelTrackletDefaultAlgo(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, unsigned int innerInnerLowerModuleIndex, unsigned int innerOuterLowerModuleIndex, unsigned int outerInnerLowerModuleIndex, unsigned int outerOuterLowerModuleIndex, unsigned int innerSegmentIndex, unsigned int outerSegmentIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& deltaPhi, float& betaIn, float& betaOut, float& pt_beta, float& zLo, float& zHi, float& rtLo, float& rtHi, float& zLoPointed, float& zHiPointed, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ, unsigned int N_MAX_SEGMENTS_PER_MODULE)
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

    short outerInnerLowerModuleSubdet = modulesInGPU.subdets[outerInnerLowerModuleIndex];
    short outerOuterLowerModuleSubdet = modulesInGPU.subdets[outerOuterLowerModuleIndex];

    if(outerInnerLowerModuleSubdet == SDL::Barrel and outerOuterLowerModuleSubdet == SDL::Barrel)
    {
        pass = runTrackletDefaultAlgoPPBB(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, outerInnerLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta,N_MAX_SEGMENTS_PER_MODULE, zLo, zHi, zLoPointed, zHiPointed, sdlCut, betaOutCut, deltaBetaCut);
    }
    else if(outerInnerLowerModuleSubdet == SDL::Barrel and outerOuterLowerModuleSubdet == SDL::Endcap)
    {
        pass = runTrackletDefaultAlgoPPBB(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, outerInnerLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta,N_MAX_SEGMENTS_PER_MODULE, zLo, zHi, zLoPointed, zHiPointed, sdlCut, betaOutCut, deltaBetaCut);
    }
    else if(outerInnerLowerModuleSubdet == SDL::Endcap and outerOuterLowerModuleSubdet == SDL::Endcap)
    {
            pass = runTrackletDefaultAlgoPPEE(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, outerInnerLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta,N_MAX_SEGMENTS_PER_MODULE, zLo, rtLo, rtHi, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ);
    }

    return pass;
}
}
#endif
