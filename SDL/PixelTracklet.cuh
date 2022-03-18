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

CUDA_DEV bool inline runTrackletDefaultAlgoPPBB(struct modules& modulesInGPU, struct objectRanges& rangesInGPU, struct miniDoublets& mdsInGPU ,struct segments& segmentsInGPU, uint16_t& pixelModuleIndex, uint16_t& outerInnerLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int thirdMDIndex, unsigned int& fourthMDIndex, float& z_OutLo, float& rt_OutLo, float& dPhiPos, float& dPhi, float& betaIn,
        float& betaOut, float& pt_beta, float& zLo, float& zHi, float& zLoPointed, float& zHiPointed, float& sdlCut, float& betaOutCut, float& deltaBetaCut) // pixel to BB and BE segments
{
    bool pass = true;

    bool isPS_OutLo = (modulesInGPU.moduleType[outerInnerLowerModuleIndex] == SDL::PS);

    float rt_InLo = mdsInGPU.anchorRt[firstMDIndex];
    float rt_InUp = mdsInGPU.anchorRt[secondMDIndex];
    rt_OutLo = mdsInGPU.anchorRt[thirdMDIndex];
    float rt_OutUp = mdsInGPU.anchorRt[fourthMDIndex];

    float z_InLo = mdsInGPU.anchorZ[firstMDIndex];
    float z_InUp = mdsInGPU.anchorZ[secondMDIndex];
    z_OutLo = mdsInGPU.anchorZ[thirdMDIndex];
    float z_OutUp = mdsInGPU.anchorZ[fourthMDIndex];

    float x_InLo = mdsInGPU.anchorX[firstMDIndex];
    float x_InUp = mdsInGPU.anchorX[secondMDIndex];
    float x_OutLo = mdsInGPU.anchorX[thirdMDIndex];
    float x_OutUp = mdsInGPU.anchorX[fourthMDIndex];

    float y_InLo = mdsInGPU.anchorY[firstMDIndex];
    float y_InUp = mdsInGPU.anchorY[secondMDIndex];
    float y_OutLo = mdsInGPU.anchorY[thirdMDIndex];
    float y_OutUp = mdsInGPU.anchorY[fourthMDIndex];

    float& rt_InOut = rt_InUp;
    float& z_InOut = z_InUp;

    pass = pass & (fabsf(deltaPhi(x_InUp, y_InUp, z_InUp, x_OutLo, y_OutLo, z_OutLo)) <= 0.5f * float(M_PI));;

    unsigned int pixelSegmentArrayIndex = innerSegmentIndex - rangesInGPU.segmentModuleIndices[pixelModuleIndex]; 
    float ptIn = segmentsInGPU.ptIn[pixelSegmentArrayIndex];
    float ptSLo = ptIn;
    float px = segmentsInGPU.px[pixelSegmentArrayIndex];
    float py = segmentsInGPU.py[pixelSegmentArrayIndex];
    float pz = segmentsInGPU.pz[pixelSegmentArrayIndex];
    float ptErr = segmentsInGPU.ptErr[pixelSegmentArrayIndex];
    float etaErr = segmentsInGPU.etaErr[pixelSegmentArrayIndex];
    ptSLo = fmaxf(PTCUT, ptSLo - 10.0f*fmaxf(ptErr, 0.005f*ptSLo));
    ptSLo = fminf(10.0f, ptSLo);


    float alpha1GeV_OutLo = asinf(fminf(rt_OutLo * k2Rinv1GeVf / ptCut, sinAlphaMax));
    float rtRatio_OutLoInLo = rt_OutLo / rt_InLo; // Outer segment beginning rt divided by inner segment beginning rt;
    const float rtRatio_OutLoInOut = rt_OutLo / rt_InOut; // Outer segment beginning rt divided by inner segment beginning rt;

    float dzDrtScale = tanf(alpha1GeV_OutLo) / alpha1GeV_OutLo; // The track can bend in r-z plane slightly
    const float zpitch_InLo = 0.05f;
    const float zpitch_InOut = 0.05f;
    float zpitch_OutLo = (isPS_OutLo ? pixelPSZpitch : strip2SZpitch);
    float zGeom = zpitch_InLo + zpitch_OutLo;
    zHi = z_InUp + (z_InUp + deltaZLum) * (rtRatio_OutLoInOut - 1.f) * (z_InUp < 0.f ? 1.f : dzDrtScale) + (zpitch_InOut + zpitch_OutLo);
    zLo = z_InUp + (z_InUp - deltaZLum) * (rtRatio_OutLoInOut - 1.f) * (z_InUp > 0.f ? 1.f : dzDrtScale) - (zpitch_InOut + zpitch_OutLo); //slope-correction only on outer end

    pass = pass & ((z_OutLo >= zLo) & (z_OutLo <= zHi));
    const float coshEta = sqrtf(ptIn * ptIn + pz * pz) / ptIn;
    // const float drt_OutLo_InLo = (rt_OutLo - rt_InLo);
    const float drt_OutLo_InUp = (rt_OutLo - rt_InUp);
    const float invRt_InLo = 1.f / rt_InLo;
    const float r3_InLo = sqrtf(z_InLo * z_InLo + rt_InLo * rt_InLo);
    const float r3_InUp = sqrtf(z_InUp * z_InUp + rt_InUp * rt_InUp);
    
    float drt_InSeg = rt_InOut - rt_InLo;
    float dz_InSeg = z_InOut - z_InLo;
    float dr3_InSeg = sqrtf(rt_InOut * rt_InOut + z_InOut * z_InOut) - sqrtf(rt_InLo * rt_InLo + z_InLo * z_InLo);

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

    pass = pass & ((z_OutLo >= zLoPointed) & (z_OutLo <= zHiPointed));
    const float sdlPVoff = 0.1f / rt_OutLo;
    sdlCut = alpha1GeV_OutLo + sqrtf(sdlMuls * sdlMuls + sdlPVoff * sdlPVoff);

    dPhiPos = deltaPhi(x_InUp, y_InUp, z_InUp, x_OutUp, y_OutUp, z_OutUp);

    //no dphipos cut
    float midPointX = 0.5f * (x_InLo + x_OutLo);
    float midPointY = 0.5f * (y_InLo + y_OutLo);
    float midPointZ = 0.5f * (z_InLo + z_OutLo);

    float diffX = x_OutLo - x_InLo;
    float diffY = y_OutLo - y_InLo;
    float diffZ = z_OutLo - z_InLo;


    dPhi = deltaPhi(midPointX, midPointY, midPointZ, diffX, diffY, diffZ);

    pass = pass & (fabsf(dPhi) <= sdlCut);

    float alpha_InLo  = __H2F(segmentsInGPU.dPhiChanges[innerSegmentIndex]);
    float alpha_OutLo = __H2F(segmentsInGPU.dPhiChanges[outerSegmentIndex]);

    bool isEC_lastLayer = modulesInGPU.subdets[outerOuterLowerModuleIndex] == SDL::Endcap and modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::TwoS;

    float alpha_OutUp,alpha_OutUp_highEdge,alpha_OutUp_lowEdge;
    alpha_OutUp = deltaPhi(x_OutUp, y_OutUp, z_OutUp, x_OutUp - x_OutLo, y_OutUp - y_OutLo, z_OutUp - z_OutLo);

    alpha_OutUp_highEdge = alpha_OutUp;
    alpha_OutUp_lowEdge = alpha_OutUp;

    float tl_axis_x = x_OutUp - x_InUp;
    float tl_axis_y = y_OutUp - y_InUp;
    float tl_axis_z = z_OutUp - z_InUp;
 
    float tl_axis_highEdge_x = tl_axis_x;
    float tl_axis_highEdge_y = tl_axis_y;

    float tl_axis_lowEdge_x = tl_axis_x;
    float tl_axis_lowEdge_y = tl_axis_y;

    betaIn = -deltaPhi(px, py, pz, tl_axis_x, tl_axis_y, tl_axis_z);
    float betaInRHmin = betaIn;
    float betaInRHmax = betaIn;

    betaOut = -alpha_OutUp + deltaPhi(x_OutUp, y_OutUp, z_OutUp, tl_axis_x, tl_axis_y, tl_axis_z);
 
    float betaOutRHmin = betaOut;
    float betaOutRHmax = betaOut;

    if(isEC_lastLayer)
    {
        alpha_OutUp_highEdge = deltaPhi(mdsInGPU.anchorHighEdgeX[fourthMDIndex], mdsInGPU.anchorHighEdgeY[fourthMDIndex], z_OutUp, mdsInGPU.anchorHighEdgeX[fourthMDIndex] - x_OutLo, mdsInGPU.anchorHighEdgeY[fourthMDIndex] - y_OutLo, z_OutUp - z_OutLo);
        alpha_OutUp_lowEdge = deltaPhi(mdsInGPU.anchorLowEdgeX[fourthMDIndex], mdsInGPU.anchorLowEdgeY[fourthMDIndex], z_OutUp, mdsInGPU.anchorLowEdgeX[fourthMDIndex] - x_OutLo, mdsInGPU.anchorLowEdgeY[fourthMDIndex] - y_OutLo, z_OutUp - z_OutLo);

        tl_axis_highEdge_x = mdsInGPU.anchorHighEdgeX[fourthMDIndex] - x_InUp;
        tl_axis_highEdge_y = mdsInGPU.anchorHighEdgeY[fourthMDIndex] - y_InUp;
        tl_axis_lowEdge_x = mdsInGPU.anchorLowEdgeX[fourthMDIndex] - x_InUp;
        tl_axis_lowEdge_y = mdsInGPU.anchorLowEdgeY[fourthMDIndex] - y_InUp;

        betaOutRHmin = -alpha_OutUp_highEdge + deltaPhi(mdsInGPU.anchorHighEdgeX[fourthMDIndex], mdsInGPU.anchorHighEdgeY[fourthMDIndex], z_OutUp, tl_axis_highEdge_x, tl_axis_highEdge_y, tl_axis_z);
        betaOutRHmax = -alpha_OutUp_lowEdge + deltaPhi(mdsInGPU.anchorLowEdgeX[fourthMDIndex], mdsInGPU.anchorLowEdgeY[fourthMDIndex], z_OutUp, tl_axis_lowEdge_x, tl_axis_lowEdge_y, tl_axis_z);
    }

    //beta computation
    float drt_tl_axis = sqrtf(tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);
    float drt_tl_lowEdge = sqrtf(tl_axis_lowEdge_x * tl_axis_lowEdge_x + tl_axis_lowEdge_y * tl_axis_lowEdge_y);
    float drt_tl_highEdge = sqrtf(tl_axis_highEdge_x * tl_axis_highEdge_x + tl_axis_highEdge_y * tl_axis_highEdge_y);

    //innerOuterAnchor - innerInnerAnchor
    const float rt_InSeg = sqrtf((x_InUp - x_InLo) * (x_InUp - x_InLo) + (y_InUp - y_InLo) * (y_InUp - y_InLo));

    //no betaIn cut for the pixels
    float betaAv = 0.5f * (betaIn + betaOut);
    pt_beta = ptIn;

    const float pt_betaMax = 7.0f;

    int lIn = 0;
    int lOut = isEC_lastLayer ? 11 : 5;
    float sdOut_dr = sqrtf((x_OutUp - x_OutLo) * (x_OutUp - x_OutLo) + (y_OutUp - y_OutLo) * (y_OutUp - y_OutLo));
    float sdOut_d = rt_OutUp - rt_OutLo;

    const float diffDr = fabsf(rt_InSeg - sdOut_dr) / fabsf(rt_InSeg + sdOut_dr);

    runDeltaBetaIterations(betaIn, betaOut, betaAv, pt_beta, rt_InSeg, sdOut_dr, drt_tl_axis, lIn);

     const float betaInMMSF = (fabsf(betaInRHmin + betaInRHmax) > 0) ? (2.f * betaIn / fabsf(betaInRHmin + betaInRHmax)) : 0.; //mean value of min,max is the old betaIn
    const float betaOutMMSF = (fabsf(betaOutRHmin + betaOutRHmax) > 0) ? (2.f * betaOut / fabsf(betaOutRHmin + betaOutRHmax)) : 0.;
    betaInRHmin *= betaInMMSF;
    betaInRHmax *= betaInMMSF;
    betaOutRHmin *= betaOutMMSF;
    betaOutRHmax *= betaOutMMSF;

    const float dBetaMuls = sdlThetaMulsF * 4.f / fminf(fabsf(pt_beta), pt_betaMax); //need to confirm the range-out value of 7 GeV
    const float alphaInAbsReg =  fmaxf(fabsf(alpha_InLo), asinf(fminf(rt_InUp * k2Rinv1GeVf / 3.0f, sinAlphaMax)));
    const float alphaOutAbsReg = fmaxf(fabsf(alpha_OutLo), asinf(fminf(rt_OutLo * k2Rinv1GeVf / 3.0f, sinAlphaMax)));
    const float dBetaInLum = lIn < 11 ? 0.0f : fabsf(alphaInAbsReg*deltaZLum / z_InUp);
    const float dBetaOutLum = lOut < 11 ? 0.0f : fabsf(alphaOutAbsReg*deltaZLum / z_OutLo);
    const float dBetaLum2 = (dBetaInLum + dBetaOutLum) * (dBetaInLum + dBetaOutLum);

    const float sinDPhi = sinf(dPhi);
    const float dBetaRIn2 = 0; // TODO-RH

    float dBetaROut = 0;
    if(isEC_lastLayer)
    {
        dBetaROut = (sqrtf(mdsInGPU.anchorHighEdgeX[fourthMDIndex] * mdsInGPU.anchorHighEdgeX[fourthMDIndex] + mdsInGPU.anchorHighEdgeY[fourthMDIndex] * mdsInGPU.anchorHighEdgeY[fourthMDIndex]) - sqrtf(mdsInGPU.anchorLowEdgeX[fourthMDIndex] * mdsInGPU.anchorLowEdgeX[fourthMDIndex] + mdsInGPU.anchorLowEdgeY[fourthMDIndex] * mdsInGPU.anchorLowEdgeY[fourthMDIndex])) * sinDPhi / drt_tl_axis;
    }

    const float dBetaROut2 =  dBetaROut * dBetaROut;

    betaOutCut = asinf(fminf(drt_tl_axis*k2Rinv1GeVf / ptCut, sinAlphaMax)) //FIXME: need faster version
        + (0.02f / sdOut_d) + sqrtf(dBetaLum2 + dBetaMuls*dBetaMuls);

    //Cut #6: The real beta cut
    pass = pass & (fabsf(betaOut) < betaOutCut);

    const float pt_betaOut = drt_tl_axis * k2Rinv1GeVf / sin(betaOut);
    const float dBetaRes = 0.02f / fminf(sdOut_d, drt_InSeg);
    const float dBetaCut2 = (dBetaRes*dBetaRes * 2.0f + dBetaMuls * dBetaMuls + dBetaLum2 + dBetaRIn2 + dBetaROut2
            + 0.25f * (fabsf(betaInRHmin - betaInRHmax) + fabsf(betaOutRHmin - betaOutRHmax)) * (fabsf(betaInRHmin - betaInRHmax) + fabsf(betaOutRHmin - betaOutRHmax)));
    float dBeta = betaIn - betaOut;
    deltaBetaCut = sqrtf(dBetaCut2);

    pass = pass & (dBeta * dBeta <= dBetaCut2);

    return pass;
}

CUDA_DEV bool inline runTrackletDefaultAlgoPPEE(struct modules& modulesInGPU, struct objectRanges& rangesInGPU, struct miniDoublets& mdsInGPU ,struct segments& segmentsInGPU, uint16_t& pixelModuleIndex, uint16_t& outerInnerLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex, unsigned int& fourthMDIndex, float& z_OutLo, float& rt_OutLo, float& deltaPhiPos, float& dPhi, float& betaIn,
        float& betaOut, float& pt_beta, float& zLo, float& rtLo, float& rtHi, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ) // pixel to EE segments
{
    bool pass = true;
    bool isPS_OutLo = (modulesInGPU.moduleType[outerInnerLowerModuleIndex] == SDL::PS);

    float rt_InLo = mdsInGPU.anchorRt[firstMDIndex];
    float rt_InUp = mdsInGPU.anchorRt[secondMDIndex];
    rt_OutLo = mdsInGPU.anchorRt[thirdMDIndex];
    float rt_OutUp = mdsInGPU.anchorRt[fourthMDIndex];

    float z_InLo = mdsInGPU.anchorZ[firstMDIndex];
    float z_InUp = mdsInGPU.anchorZ[secondMDIndex];
    z_OutLo = mdsInGPU.anchorZ[thirdMDIndex];
    float z_OutUp = mdsInGPU.anchorZ[fourthMDIndex];

    float x_InLo = mdsInGPU.anchorX[firstMDIndex];
    float x_InUp = mdsInGPU.anchorX[secondMDIndex];
    float x_OutLo = mdsInGPU.anchorX[thirdMDIndex];
    float x_OutUp = mdsInGPU.anchorX[fourthMDIndex];

    float y_InLo = mdsInGPU.anchorY[firstMDIndex];
    float y_InUp = mdsInGPU.anchorY[secondMDIndex];
    float y_OutLo = mdsInGPU.anchorY[thirdMDIndex];
    float y_OutUp = mdsInGPU.anchorY[fourthMDIndex];

    unsigned int pixelSegmentArrayIndex = innerSegmentIndex - rangesInGPU.segmentModuleIndices[pixelModuleIndex];

    float ptIn = segmentsInGPU.ptIn[pixelSegmentArrayIndex];
    float ptSLo = ptIn;
    float px = segmentsInGPU.px[pixelSegmentArrayIndex];
    float py = segmentsInGPU.py[pixelSegmentArrayIndex];
    float pz = segmentsInGPU.pz[pixelSegmentArrayIndex];
    float ptErr = segmentsInGPU.ptErr[pixelSegmentArrayIndex];
    float etaErr = segmentsInGPU.etaErr[pixelSegmentArrayIndex];

    ptSLo = fmaxf(PTCUT, ptSLo - 10.0f*fmaxf(ptErr, 0.005f*ptSLo));
    ptSLo = fminf(10.0f, ptSLo);

    float rtOut_o_rtIn = rt_OutLo/rt_InUp;
    const float zpitch_InLo = 0.05f;
    float zpitch_OutLo = (isPS_OutLo ? pixelPSZpitch : strip2SZpitch);
    float zGeom = zpitch_InLo + zpitch_OutLo;

    const float sdlSlope = asinf(fminf(rt_OutLo * k2Rinv1GeVf / ptCut, sinAlphaMax));
    const float dzDrtScale = tanf(sdlSlope) / sdlSlope;//FIXME: need approximate value
    zLo = z_InUp + (z_InUp - deltaZLum) * (rtOut_o_rtIn - 1.f) * (z_InUp > 0.f ? 1.f : dzDrtScale) - zGeom; //slope-correction only on outer end

    pass = pass & (z_InUp * z_OutLo > 0);

    const float dLum = copysignf(deltaZLum, z_InUp);
    bool isOutSgInnerMDPS = modulesInGPU.moduleType[outerInnerLowerModuleIndex] == SDL::PS;

    const float rtGeom1 = isOutSgInnerMDPS ? pixelPSZpitch : strip2SZpitch;//FIXME: make this chosen by configuration for lay11,12 full PS
    const float zGeom1 = copysignf(zGeom, z_InUp); //used in B-E region
    rtLo = rt_InUp * (1.f + (z_OutLo- z_InUp - zGeom1) / (z_InUp + zGeom1 + dLum) / dzDrtScale) - rtGeom1; //slope correction only on the lower end

    pass = pass & (rt_OutLo >= rtLo);


    float zInForHi = z_InUp - zGeom1 - dLum;
    if (zInForHi * z_InUp < 0)
        zInForHi = copysignf(0.1f, z_InUp);
    rtHi = rt_InUp * (1.f + (z_OutLo - z_InUp + zGeom1) / zInForHi) + rtGeom1;

    // Cut #2: rt condition
    pass = pass & ((rt_OutLo >= rtLo) & (rt_OutLo <= rtHi));

    const float dzOutInAbs = fabsf(z_OutLo - z_InUp);
    const float coshEta = hypotf(ptIn, pz) / ptIn;
    const float multDzDr = dzOutInAbs*coshEta/(coshEta*coshEta - 1.f);
    const float r3_InUp = sqrtf(z_InUp * z_InUp + rt_InUp * rt_InUp);
    const float sdlThetaMulsF = 0.015f * sqrtf(0.1f + 0.2f * (rt_OutLo - rt_InUp) / 50.f) * sqrtf(r3_InUp / rt_InUp);
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
    const float rtLo_point = rt_InUp + drtMean - rtWindow;
    const float rtHi_point = rt_InUp + drtMean + rtWindow;

    // Cut #3: rt-z pointed
    pass = pass & ((rt_OutLo >= rtLo_point) & (rt_OutLo <= rtHi_point));
    const float alpha1GeV_OutLo = asinf(fminf(rt_OutLo * k2Rinv1GeVf / ptCut, sinAlphaMax));
    const float sdlPVoff = 0.1f / rt_OutLo;
    sdlCut = alpha1GeV_OutLo + sqrtf(sdlMuls * sdlMuls + sdlPVoff * sdlPVoff);

    deltaPhiPos = deltaPhi(x_InUp, y_InUp, z_InUp, x_OutUp, y_OutUp, z_OutUp);

    //no deltaphipos cut

    float midPointX = 0.5f * (x_InLo + x_OutLo);
    float midPointY = 0.5f * (y_InLo + y_OutLo);
    float midPointZ = 0.5f * (z_InLo + z_OutLo);

    float diffX = x_OutLo - x_InLo;
    float diffY = y_OutLo - y_InLo;
    float diffZ = z_OutLo - z_InLo;

    dPhi = deltaPhi(midPointX, midPointY, midPointZ, diffX, diffY, diffZ);

    // Cut #5: deltaPhiChange
    pass = pass & (fabsf(dPhi) <= sdlCut);

    float alpha_InLo  = __H2F(segmentsInGPU.dPhiChanges[innerSegmentIndex]);
    float alpha_OutLo = __H2F(segmentsInGPU.dPhiChanges[outerSegmentIndex]);

    bool isEC_lastLayer = modulesInGPU.subdets[outerOuterLowerModuleIndex] == SDL::Endcap and modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::TwoS;

    float alpha_OutUp,alpha_OutUp_highEdge,alpha_OutUp_lowEdge;

    alpha_OutUp = deltaPhi(x_OutUp, y_OutUp, z_OutUp, x_OutUp - x_OutLo, y_OutUp - y_OutLo, z_OutUp - z_OutLo);
    alpha_OutUp_highEdge = alpha_OutUp;
    alpha_OutUp_lowEdge = alpha_OutUp;

    float tl_axis_x = x_OutUp - x_InUp;
    float tl_axis_y = y_OutUp - y_InUp;
    float tl_axis_z = z_OutUp - z_InUp;

    float tl_axis_highEdge_x = tl_axis_x;
    float tl_axis_highEdge_y = tl_axis_y;

    float tl_axis_lowEdge_x = tl_axis_x;
    float tl_axis_lowEdge_y = tl_axis_y;

    betaIn = -deltaPhi(px, py, pz, tl_axis_x, tl_axis_y, tl_axis_z);
    float betaInRHmin = betaIn;
    float betaInRHmax = betaIn;

    betaOut = -alpha_OutUp + deltaPhi(x_OutUp, y_OutUp, z_OutUp, tl_axis_x, tl_axis_y, tl_axis_z);
    float betaOutRHmin = betaOut;
    float betaOutRHmax = betaOut;

    if(isEC_lastLayer)
    {

        alpha_OutUp_highEdge = deltaPhi(mdsInGPU.anchorHighEdgeX[fourthMDIndex], mdsInGPU.anchorHighEdgeY[fourthMDIndex], z_OutUp, mdsInGPU.anchorHighEdgeX[fourthMDIndex] - x_OutLo, mdsInGPU.anchorHighEdgeY[fourthMDIndex] - y_OutLo, z_OutUp - z_OutLo);
        alpha_OutUp_lowEdge = deltaPhi(mdsInGPU.anchorLowEdgeX[fourthMDIndex], mdsInGPU.anchorLowEdgeY[fourthMDIndex], z_OutUp, mdsInGPU.anchorLowEdgeX[fourthMDIndex] - x_OutLo, mdsInGPU.anchorLowEdgeY[fourthMDIndex] - y_OutLo, z_OutUp - z_OutLo);

        tl_axis_highEdge_x = mdsInGPU.anchorHighEdgeX[fourthMDIndex] - x_InUp;
        tl_axis_highEdge_y = mdsInGPU.anchorHighEdgeY[fourthMDIndex] - y_InUp;
        tl_axis_lowEdge_x = mdsInGPU.anchorLowEdgeX[fourthMDIndex] - x_InUp;
        tl_axis_lowEdge_y = mdsInGPU.anchorLowEdgeY[fourthMDIndex] - y_InUp;

        betaOutRHmin = -alpha_OutUp_highEdge + deltaPhi(mdsInGPU.anchorHighEdgeX[fourthMDIndex], mdsInGPU.anchorHighEdgeY[fourthMDIndex], z_OutUp, tl_axis_highEdge_x, tl_axis_highEdge_y, tl_axis_z);
        betaOutRHmax = -alpha_OutUp_lowEdge + deltaPhi(mdsInGPU.anchorLowEdgeX[fourthMDIndex], mdsInGPU.anchorLowEdgeY[fourthMDIndex], z_OutUp, tl_axis_lowEdge_x, tl_axis_lowEdge_y, tl_axis_z);
    }

    //beta computation
    float drt_tl_axis = sqrtf(tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);
    float drt_tl_lowEdge = sqrtf(tl_axis_lowEdge_x * tl_axis_lowEdge_x + tl_axis_lowEdge_y * tl_axis_lowEdge_y);
    float drt_tl_highEdge = sqrtf(tl_axis_highEdge_x * tl_axis_highEdge_x + tl_axis_highEdge_y * tl_axis_highEdge_y);
//no betaIn cut for the pixels
    const float rt_InSeg = sqrtf((x_InUp - x_InLo) * (x_InUp - x_InLo) + (y_InUp - y_InLo) * (y_InUp - y_InLo));

    float betaAv = 0.5f * (betaIn + betaOut);
    pt_beta = ptIn;

    const float pt_betaMax = 7.0f;

    int lIn = 0;
    int lOut = isEC_lastLayer ? 11 : 5;
    float sdOut_dr = sqrtf((x_OutUp - x_OutLo) * (x_OutUp - x_OutLo) + (y_OutUp - y_OutLo) * (y_OutUp - y_OutLo));
    float sdOut_d = rt_OutUp - rt_OutLo;

    const float diffDr = fabsf(rt_InSeg - sdOut_dr) / fabsf(rt_InSeg + sdOut_dr);

    runDeltaBetaIterations(betaIn, betaOut, betaAv, pt_beta, rt_InSeg, sdOut_dr, drt_tl_axis, lIn);

     const float betaInMMSF = (fabsf(betaInRHmin + betaInRHmax) > 0) ? (2.f * betaIn / fabsf(betaInRHmin + betaInRHmax)) : 0.; //mean value of min,max is the old betaIn
    const float betaOutMMSF = (fabsf(betaOutRHmin + betaOutRHmax) > 0) ? (2.f * betaOut / fabsf(betaOutRHmin + betaOutRHmax)) : 0.;
    betaInRHmin *= betaInMMSF;
    betaInRHmax *= betaInMMSF;
    betaOutRHmin *= betaOutMMSF;
    betaOutRHmax *= betaOutMMSF;

    const float dBetaMuls = sdlThetaMulsF * 4.f / fminf(fabsf(pt_beta), pt_betaMax); //need to confirm the range-out value of 7 GeV

    const float alphaInAbsReg =  fmaxf(fabsf(alpha_InLo), asinf(fminf(rt_InUp * k2Rinv1GeVf / 3.0f, sinAlphaMax)));
    const float alphaOutAbsReg = fmaxf(fabsf(alpha_OutLo), asinf(fminf(rt_OutLo * k2Rinv1GeVf / 3.0f, sinAlphaMax)));
    const float dBetaInLum = lIn < 11 ? 0.0f : fabsf(alphaInAbsReg*deltaZLum / z_InUp);
    const float dBetaOutLum = lOut < 11 ? 0.0f : fabsf(alphaOutAbsReg*deltaZLum / z_OutLo);
    const float dBetaLum2 = (dBetaInLum + dBetaOutLum) * (dBetaInLum + dBetaOutLum);

    const float sinDPhi = sinf(dPhi);
    const float dBetaRIn2 = 0; // TODO-RH

    float dBetaROut = 0;
    if(isEC_lastLayer)
    {
        dBetaROut = (sqrtf(mdsInGPU.anchorHighEdgeX[fourthMDIndex] * mdsInGPU.anchorHighEdgeX[fourthMDIndex] + mdsInGPU.anchorHighEdgeY[fourthMDIndex] * mdsInGPU.anchorHighEdgeY[fourthMDIndex]) - sqrtf(mdsInGPU.anchorLowEdgeX[fourthMDIndex] * mdsInGPU.anchorLowEdgeX[fourthMDIndex] + mdsInGPU.anchorLowEdgeY[fourthMDIndex] * mdsInGPU.anchorLowEdgeY[fourthMDIndex])) * sinDPhi / drt_tl_axis;
    }

    const float dBetaROut2 =  dBetaROut * dBetaROut;

    betaOutCut = asinf(fminf(drt_tl_axis*k2Rinv1GeVf / ptCut, sinAlphaMax)) //FIXME: need faster version
        + (0.02f / sdOut_d) + sqrtf(dBetaLum2 + dBetaMuls*dBetaMuls);

    //Cut #6: The real beta cut
    pass = pass & (fabsf(betaOut) < betaOutCut);

    const float pt_betaOut = drt_tl_axis * k2Rinv1GeVf / sin(betaOut);
    float drt_InSeg = rt_InUp - rt_InLo;

    const float dBetaRes = 0.02f / fminf(sdOut_d, drt_InSeg);
    const float dBetaCut2 = (dBetaRes*dBetaRes * 2.0f + dBetaMuls * dBetaMuls + dBetaLum2 + dBetaRIn2 + dBetaROut2
            + 0.25f * (fabsf(betaInRHmin - betaInRHmax) + fabsf(betaOutRHmin - betaOutRHmax)) * (fabsf(betaInRHmin - betaInRHmax) + fabsf(betaOutRHmin - betaOutRHmax)));
    float dBeta = betaIn - betaOut;
    deltaBetaCut = sqrtf(dBetaCut2);

    pass = pass & (dBeta * dBeta <= dBetaCut2);
    return pass;
}

CUDA_DEV bool inline runPixelTrackletDefaultAlgo(struct modules& modulesInGPU, struct objectRanges& rangesInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, uint16_t& pixelLowerModuleIndex, uint16_t& outerInnerLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& deltaPhi, float& betaIn, float& betaOut, float& pt_beta, float& zLo, float& zHi, float& rtLo, float& rtHi, float& zLoPointed, float& zHiPointed, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ)
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

    unsigned int firstMDIndex = segmentsInGPU.mdIndices[2 * innerSegmentIndex];

    unsigned int secondMDIndex = segmentsInGPU.mdIndices[2 * innerSegmentIndex + 1];
    unsigned int thirdMDIndex = segmentsInGPU.mdIndices[2 * outerSegmentIndex];
    unsigned int fourthMDIndex = segmentsInGPU.mdIndices[2 * outerSegmentIndex + 1];

    if(outerInnerLowerModuleSubdet == SDL::Barrel and outerOuterLowerModuleSubdet == SDL::Barrel)
    {
        pass = runTrackletDefaultAlgoPPBB(modulesInGPU, rangesInGPU, mdsInGPU, segmentsInGPU, pixelLowerModuleIndex, outerInnerLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, firstMDIndex, secondMDIndex, thirdMDIndex, fourthMDIndex, zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta, zLo, zHi, zLoPointed, zHiPointed, sdlCut, betaOutCut, deltaBetaCut);
    }
    else if(outerInnerLowerModuleSubdet == SDL::Barrel and outerOuterLowerModuleSubdet == SDL::Endcap)
    {
        pass = runTrackletDefaultAlgoPPBB(modulesInGPU, rangesInGPU, mdsInGPU, segmentsInGPU, pixelLowerModuleIndex, outerInnerLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, firstMDIndex, secondMDIndex, thirdMDIndex, fourthMDIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta, zLo, zHi, zLoPointed, zHiPointed, sdlCut, betaOutCut, deltaBetaCut);
    }
    else if(outerInnerLowerModuleSubdet == SDL::Endcap and outerOuterLowerModuleSubdet == SDL::Endcap)
    {
            pass = runTrackletDefaultAlgoPPEE(modulesInGPU, rangesInGPU, mdsInGPU, segmentsInGPU, pixelLowerModuleIndex, outerInnerLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, firstMDIndex, secondMDIndex, thirdMDIndex, fourthMDIndex, zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta, zLo, rtLo, rtHi, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ);
    }

    return pass;
}
}
#endif
