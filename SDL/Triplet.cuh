#ifndef Triplet_cuh
#define Triplet_cuh

#include "Constants.cuh"
#include "Segment.cuh"
#include "MiniDoublet.cuh"
#include "Module.cuh"
#include "Hit.cuh"

namespace SDL
{
    struct triplets
    {
        unsigned int* segmentIndices;
        uint16_t* lowerModuleIndices; //3 of them now
        int* nTriplets;
        int* totOccupancyTriplets;

        unsigned int* nMemoryLocations;

        uint8_t* logicalLayers;
        unsigned int* hitIndices;
        
        //delta beta = betaIn - betaOut
        FPX* betaIn;
        FPX* betaOut;
        FPX* pt_beta;

        bool* partOfPT5;
        bool* partOfT5;
        bool* partOfPT3;

#ifdef CUT_VALUE_DEBUG
        //debug variables
        float* zOut;
        float* rtOut;

        float* deltaPhiPos;
        float* deltaPhi;

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
        triplets();
        ~triplets();
        void freeMemory(cudaStream_t stream);
        void freeMemoryCache();
        void resetMemory(unsigned int maxTriplets, unsigned int nLowerModules,cudaStream_t stream);
    };

    void createTripletsInExplicitMemory(struct triplets& tripletsInGPU, unsigned int maxTriplets, uint16_t nLowerModules,cudaStream_t stream);

#ifdef CUT_VALUE_DEBUG
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void addTripletToMemory(struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct triplets& tripletsInGPU, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, uint16_t& innerInnerLowerModuleIndex, uint16_t& middleLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& deltaPhi, float& betaIn, float& betaOut, float& pt_beta, float& zLo, float& zHi, float& rtLo, float& rtHi, float& zLoPointed, float&zHiPointed, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ, unsigned int& tripletIndex)
#else
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void addTripletToMemory(struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct triplets& tripletsInGPU, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, uint16_t& innerInnerLowerModuleIndex, uint16_t& middleLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, float& betaIn, float& betaOut, float& pt_beta, unsigned int& tripletIndex)
#endif
    {
        tripletsInGPU.segmentIndices[tripletIndex * 2] = innerSegmentIndex;
        tripletsInGPU.segmentIndices[tripletIndex * 2 + 1] = outerSegmentIndex;
        tripletsInGPU.lowerModuleIndices[tripletIndex * 3] = innerInnerLowerModuleIndex;
        tripletsInGPU.lowerModuleIndices[tripletIndex * 3 + 1] = middleLowerModuleIndex;
        tripletsInGPU.lowerModuleIndices[tripletIndex * 3 + 2] = outerOuterLowerModuleIndex;

        tripletsInGPU.betaIn[tripletIndex]  = __F2H(betaIn);
        tripletsInGPU.betaOut[tripletIndex] = __F2H(betaOut);
        tripletsInGPU.pt_beta[tripletIndex] = __F2H(pt_beta);
    
        tripletsInGPU.logicalLayers[tripletIndex * 3] = modulesInGPU.layers[innerInnerLowerModuleIndex] + (modulesInGPU.subdets[innerInnerLowerModuleIndex] == 4) * 6;
        tripletsInGPU.logicalLayers[tripletIndex * 3 + 1] = modulesInGPU.layers[middleLowerModuleIndex] + (modulesInGPU.subdets[middleLowerModuleIndex] == 4) * 6;
        tripletsInGPU.logicalLayers[tripletIndex * 3 + 2] = modulesInGPU.layers[outerOuterLowerModuleIndex] + (modulesInGPU.subdets[outerOuterLowerModuleIndex] == 4) * 6;
        //get the hits
        unsigned int firstMDIndex = segmentsInGPU.mdIndices[2 * innerSegmentIndex];
        unsigned int secondMDIndex = segmentsInGPU.mdIndices[2 * innerSegmentIndex + 1];
        unsigned int thirdMDIndex = segmentsInGPU.mdIndices[2 * outerSegmentIndex + 1];

        tripletsInGPU.hitIndices[tripletIndex * 6] = mdsInGPU.anchorHitIndices[firstMDIndex];
        tripletsInGPU.hitIndices[tripletIndex * 6 + 1] = mdsInGPU.outerHitIndices[firstMDIndex];
        tripletsInGPU.hitIndices[tripletIndex * 6 + 2] = mdsInGPU.anchorHitIndices[secondMDIndex];
        tripletsInGPU.hitIndices[tripletIndex * 6 + 3] = mdsInGPU.outerHitIndices[secondMDIndex];
        tripletsInGPU.hitIndices[tripletIndex * 6 + 4] = mdsInGPU.anchorHitIndices[thirdMDIndex];
        tripletsInGPU.hitIndices[tripletIndex * 6 + 5] = mdsInGPU.outerHitIndices[thirdMDIndex];
#ifdef CUT_VALUE_DEBUG
        tripletsInGPU.zOut[tripletIndex] = zOut;
        tripletsInGPU.rtOut[tripletIndex] = rtOut;
        tripletsInGPU.deltaPhiPos[tripletIndex] = deltaPhiPos;
        tripletsInGPU.deltaPhi[tripletIndex] = deltaPhi;
        tripletsInGPU.zLo[tripletIndex] = zLo;
        tripletsInGPU.zHi[tripletIndex] = zHi;
        tripletsInGPU.rtLo[tripletIndex] = rtLo;
        tripletsInGPU.rtHi[tripletIndex] = rtHi;
        tripletsInGPU.zLoPointed[tripletIndex] = zLoPointed;
        tripletsInGPU.zHiPointed[tripletIndex] = zHiPointed;
        tripletsInGPU.sdlCut[tripletIndex] = sdlCut;
        tripletsInGPU.betaInCut[tripletIndex] = betaInCut;
        tripletsInGPU.betaOutCut[tripletIndex] = betaOutCut;
        tripletsInGPU.deltaBetaCut[tripletIndex] = deltaBetaCut;
        tripletsInGPU.kZ[tripletIndex] = kZ;
#endif
    };

    template<typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passRZConstraint(TAcc const & acc, struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& middleLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex) 
    {
        //get the rt and z
        const float& r1 = mdsInGPU.anchorRt[firstMDIndex];
        const float& r2 = mdsInGPU.anchorRt[secondMDIndex];
        const float& r3 = mdsInGPU.anchorRt[thirdMDIndex];

        const float& z1 = mdsInGPU.anchorZ[firstMDIndex];
        const float& z2 = mdsInGPU.anchorZ[secondMDIndex];
        const float& z3 = mdsInGPU.anchorZ[thirdMDIndex];
            
        //following Philip's layer number prescription
        const int layer1 = modulesInGPU.sdlLayers[innerInnerLowerModuleIndex];
        const int layer2 = modulesInGPU.sdlLayers[middleLowerModuleIndex];
        const int layer3 = modulesInGPU.sdlLayers[outerOuterLowerModuleIndex];

        const float residual = z2 - ( (z3 - z1) / (r3 - r1) * (r2 - r1) + z1);

        if (layer1 == 12 and layer2 == 13 and layer3 == 14)
        {
            return false;
        }
        else if (layer1 == 1 and layer2 == 2 and layer3 == 3)
        {
            return alpaka::math::abs(acc, residual) < 0.53f;
        }
        else if (layer1 == 1 and layer2 == 2 and layer3 == 7)
        {
            return alpaka::math::abs(acc, residual) < 1;
        }
        else if (layer1 == 13 and layer2 == 14 and layer3 == 15)
        {
            return false;
        }
        else if (layer1 == 14 and layer2 == 15 and layer3 == 16)
        {
            return false;
        }
        else if (layer1 == 1 and layer2 == 7 and layer3 == 8)
        {
            return alpaka::math::abs(acc, residual) < 1;
        }
        else if (layer1 == 2 and layer2 == 3 and layer3 == 4)
        {
            return alpaka::math::abs(acc, residual) < 1.21f;
        }
        else if (layer1 == 2 and layer2 == 3 and layer3 == 7)
        {
            return alpaka::math::abs(acc, residual) < 1.f;
        }
        else if (layer1 == 2 and layer2 == 7 and layer3 == 8)
        {
            return alpaka::math::abs(acc, residual) < 1.f;
        }
        else if (layer1 == 3 and layer2 == 4 and layer3 == 5)
        {
            return alpaka::math::abs(acc, residual) < 2.7f;
        }
        else if (layer1 == 4 and layer2 == 5 and layer3 == 6)
        {
            return alpaka::math::abs(acc, residual) < 3.06f;
        }
        else if (layer1 == 7 and layer2 == 8 and layer3 == 9)
        {
            return alpaka::math::abs(acc, residual) < 1;
        }
        else if (layer1 == 8 and layer2 == 9 and layer3 == 10)
        {
            return alpaka::math::abs(acc, residual) < 1;
        }
        else if (layer1 == 9 and layer2 == 10 and layer3 == 11)
        {
            return alpaka::math::abs(acc, residual) < 1;
        }
        else
        {
            return alpaka::math::abs(acc, residual) < 5;
        }
    };

    template<typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPointingConstraintBBB(TAcc const & acc, struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& middleLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex, float& zOut, float& rtOut)
    {
        bool pass = true;
        bool isPSIn = (modulesInGPU.moduleType[innerInnerLowerModuleIndex] == SDL::PS);
        bool isPSOut = (modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::PS);

        float rtIn = mdsInGPU.anchorRt[firstMDIndex];
        float rtMid = mdsInGPU.anchorRt[secondMDIndex];
        rtOut = mdsInGPU.anchorRt[thirdMDIndex];
        
        float zIn = mdsInGPU.anchorZ[firstMDIndex];
        float zMid = mdsInGPU.anchorZ[secondMDIndex];
        zOut = mdsInGPU.anchorZ[thirdMDIndex];

        float alpha1GeVOut = alpaka::math::asin(acc, alpaka::math::min(acc, rtOut * SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax));

        float rtRatio_OutIn = rtOut / rtIn; // Outer segment beginning rt divided by inner segment beginning rt;
        float dzDrtScale = alpaka::math::tan(acc, alpha1GeVOut) / alpha1GeVOut; // The track can bend in r-z plane slightly
        float zpitchIn = (isPSIn ? SDL::pixelPSZpitch : SDL::strip2SZpitch);
        float zpitchOut = (isPSOut ? SDL::pixelPSZpitch : SDL::strip2SZpitch);

        const float zHi = zIn + (zIn + SDL::deltaZLum) * (rtRatio_OutIn - 1.f) * (zIn < 0.f ? 1.f : dzDrtScale) + (zpitchIn + zpitchOut);
        const float zLo = zIn + (zIn - SDL::deltaZLum) * (rtRatio_OutIn - 1.f) * (zIn > 0.f ? 1.f : dzDrtScale) - (zpitchIn + zpitchOut); //slope-correction only on outer end

        //Cut 1 - z compatibility
        pass = pass and ((zOut >= zLo) & (zOut <= zHi));
        if(not pass) return pass;

        float drt_OutIn = (rtOut - rtIn);
        float invRtIn = 1. / rtIn;

        float r3In = alpaka::math::sqrt(acc, zIn * zIn + rtIn * rtIn);
        float drt_InSeg = rtMid - rtIn;
        float dz_InSeg = zMid - zIn;
        float dr3_InSeg = alpaka::math::sqrt(acc, rtMid * rtMid + zMid * zMid) - alpaka::math::sqrt(acc, rtIn * rtIn + zIn + zIn);

        float coshEta = dr3_InSeg/drt_InSeg;
        float dzErr = (zpitchIn + zpitchOut) * (zpitchIn + zpitchOut) * 2.f;

        float sdlThetaMulsF = 0.015f *alpaka::math::sqrt(acc, 0.1f + 0.2f * (rtOut - rtIn) / 50.f) *alpaka::math::sqrt(acc, r3In / rtIn);
        float sdlMuls = sdlThetaMulsF * 3.f / SDL::ptCut * 4.f; // will need a better guess than x4?
        dzErr += sdlMuls * sdlMuls * drt_OutIn * drt_OutIn / 3.f * coshEta * coshEta; //sloppy
        dzErr =alpaka::math::sqrt(acc, dzErr);

        // Constructing upper and lower bound
        const float dzMean = dz_InSeg / drt_InSeg * drt_OutIn;
        const float zWindow = dzErr / drt_InSeg * drt_OutIn + (zpitchIn + zpitchOut); //FIXME for SDL::ptCut lower than ~0.8 need to add curv path correction
        const float zLoPointed = zIn + dzMean * (zIn > 0.f ? 1.f : dzDrtScale) - zWindow;
        const float zHiPointed = zIn + dzMean * (zIn < 0.f ? 1.f : dzDrtScale) + zWindow;

        // Constructing upper and lower bound

        // Cut #2: Pointed Z (Inner segment two MD points to outer segment inner MD)
        pass = pass and ((zOut >= zLoPointed) & (zOut <= zHiPointed));

        return pass;
    };

    template<typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPointingConstraintBBE(TAcc const & acc, struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& middleLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex, float& zOut, float& rtOut)
    {
        bool pass = true;
        //unsigned int outerInnerLowerModuleIndex = middleLowerModuleIndex;

        bool isPSIn = (modulesInGPU.moduleType[innerInnerLowerModuleIndex] == SDL::PS);
        bool isPSOut = (modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::PS);

        float rtIn = mdsInGPU.anchorRt[firstMDIndex];
        float rtMid = mdsInGPU.anchorRt[secondMDIndex];
        rtOut = mdsInGPU.anchorRt[thirdMDIndex];
        
        float zIn = mdsInGPU.anchorZ[firstMDIndex];
        float zMid = mdsInGPU.anchorZ[secondMDIndex];
        zOut = mdsInGPU.anchorZ[thirdMDIndex];

        
        float alpha1GeV_OutLo = alpaka::math::asin(acc, alpaka::math::min(acc, rtOut * SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax));

        float rtRatio_OutIn = rtOut / rtIn; // Outer segment beginning rt divided by inner segment beginning rt;
        float dzDrtScale = alpaka::math::tan(acc, alpha1GeV_OutLo) / alpha1GeV_OutLo; // The track can bend in r-z plane slightly
        float zpitchIn = (isPSIn ? SDL::pixelPSZpitch : SDL::strip2SZpitch);
        float zpitchOut = (isPSOut ? SDL::pixelPSZpitch : SDL::strip2SZpitch);
        float zGeom = zpitchIn + zpitchOut;

        float zLo = zIn + (zIn - SDL::deltaZLum) * (rtRatio_OutIn - 1.f) * (zIn > 0.f ? 1.f : dzDrtScale) - zGeom; 

        // Cut #0: Preliminary (Only here in endcap case)
        pass = pass and (zIn * zOut > 0);
        if(not pass) return pass;

        float dLum = SDL::copysignf(SDL::deltaZLum, zIn);
        bool isOutSgInnerMDPS = modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::PS;
        float rtGeom1 = isOutSgInnerMDPS ? SDL::pixelPSZpitch : SDL::strip2SZpitch;
        float zGeom1 = SDL::copysignf(zGeom,zIn);
        float rtLo = rtIn * (1.f + (zOut - zIn - zGeom1) / (zIn + zGeom1 + dLum) / dzDrtScale) - rtGeom1; //slope correction only on the lower end
        zOut = zOut;
        rtOut = rtOut;

        //Cut #1: rt condition
        float zInForHi = zIn - zGeom1 - dLum;
        if(zInForHi * zIn < 0)
        {
            zInForHi = SDL::copysignf(0.1f,zIn);
        }
        float rtHi = rtIn * (1.f + (zOut - zIn + zGeom1) / zInForHi) + rtGeom1;

        //Cut #2: rt condition
        pass = pass and ((rtOut >= rtLo) & (rtOut <= rtHi));
        if(not pass) return pass;

        float rIn = alpaka::math::sqrt(acc, zIn * zIn + rtIn * rtIn);

        const float drtSDIn = rtMid - rtIn;
        const float dzSDIn = zMid - zIn;
        const float dr3SDIn = alpaka::math::sqrt(acc, rtMid * rtMid + zMid * zMid) - alpaka::math::sqrt(acc, rtIn * rtIn + zIn * zIn);

        const float coshEta = dr3SDIn / drtSDIn; //direction estimate
        const float dzOutInAbs = alpaka::math::abs(acc, zOut - zIn);
        const float multDzDr = dzOutInAbs * coshEta / (coshEta * coshEta - 1.f);
        const float zGeom1_another = SDL::pixelPSZpitch;
        const float kZ = (zOut - zIn) / dzSDIn;
        float drtErr = zGeom1_another * zGeom1_another * drtSDIn * drtSDIn / dzSDIn / dzSDIn * (1.f - 2.f * kZ + 2.f * kZ * kZ);
        const float sdlThetaMulsF = 0.015f * alpaka::math::sqrt(acc, 0.1f + 0.2 * (rtOut - rtIn) / 50.f) * alpaka::math::sqrt(acc, rIn / rtIn);
        const float sdlMuls = sdlThetaMulsF * 3.f / SDL::ptCut * 4.f; //will need a better guess than x4?
        drtErr += sdlMuls * sdlMuls * multDzDr * multDzDr / 3.f * coshEta * coshEta; //sloppy: relative muls is 1/3 of total muls
        drtErr = alpaka::math::sqrt(acc, drtErr);
        const float drtMean = drtSDIn * dzOutInAbs / alpaka::math::abs(acc, dzSDIn);
        const float rtWindow = drtErr + rtGeom1;
        const float rtLo_another = rtIn + drtMean / dzDrtScale - rtWindow;
        const float rtHi_another = rtIn + drtMean + rtWindow;

        //Cut #3: rt-z pointed

        pass = pass and (kZ >=0) & (rtOut >= rtLo) & (rtOut <= rtHi);
        return pass;
    };

    template<typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPointingConstraintEEE(TAcc const & acc, struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& middleLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex, float& zOut, float& rtOut)
    {
        bool pass = true;
        bool isPSIn = (modulesInGPU.moduleType[innerInnerLowerModuleIndex] == SDL::PS);
        bool isPSOut = (modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::PS);

        float rtIn = mdsInGPU.anchorRt[firstMDIndex];
        float rtMid = mdsInGPU.anchorRt[secondMDIndex];
        rtOut = mdsInGPU.anchorRt[thirdMDIndex];
        
        float zIn = mdsInGPU.anchorZ[firstMDIndex];
        float zMid = mdsInGPU.anchorZ[secondMDIndex];
        zOut = mdsInGPU.anchorZ[thirdMDIndex];


        float alpha1GeV_Out = alpaka::math::asin(acc, alpaka::math::min(acc, rtOut * SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax));

        float rtRatio_OutIn = rtOut / rtIn; // Outer segment beginning rt divided by inner segment beginning rt;
        float dzDrtScale = alpaka::math::tan(acc, alpha1GeV_Out) / alpha1GeV_Out; // The track can bend in r-z plane slightly
        float zpitchIn = (isPSIn ? SDL::pixelPSZpitch : SDL::strip2SZpitch);
        float zpitchOut = (isPSOut ? SDL::pixelPSZpitch : SDL::strip2SZpitch);
        float zGeom = zpitchIn + zpitchOut;

        const float zLo = zIn + (zIn - SDL::deltaZLum) * (rtRatio_OutIn - 1.f) * (zIn > 0.f ? 1.f : dzDrtScale) - zGeom; //slope-correction only on outer end


        // Cut #0: Preliminary (Only here in endcap case)
        pass = pass and (zIn * zOut > 0);
        if(not pass) return pass;

        float dLum = SDL::copysignf(SDL::deltaZLum, zIn);
        bool isOutSgOuterMDPS = modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::PS;
        bool isInSgInnerMDPS = modulesInGPU.moduleType[innerInnerLowerModuleIndex] == SDL::PS;

        float rtGeom = (isInSgInnerMDPS and isOutSgOuterMDPS) ? 2.f * SDL::pixelPSZpitch : (isInSgInnerMDPS or isOutSgOuterMDPS) ? SDL::pixelPSZpitch + SDL::strip2SZpitch : 2.f * SDL::strip2SZpitch;

        float zGeom1 = SDL::copysignf(zGeom,zIn);
        float dz = zOut - zIn;
        const float rtLo = rtIn * (1.f + dz / (zIn + dLum) / dzDrtScale) - rtGeom; //slope correction only on the lower end
        const float rtHi = rtIn * (1.f + dz / (zIn - dLum)) + rtGeom;

        //Cut #1: rt condition
        pass = pass and ((rtOut >= rtLo) & (rtOut <= rtHi));
        if(not pass) return pass;
        
        bool isInSgOuterMDPS = modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::PS;

        float drOutIn = rtOut - rtIn;
        float drtSDIn = rtMid - rtIn;
        float dzSDIn = zMid - zIn;
        float dr3SDIn = alpaka::math::sqrt(acc, rtMid * rtMid + zMid * zMid) - alpaka::math::sqrt(acc, rtIn * rtIn + zIn * zIn);

        float coshEta = dr3SDIn / drtSDIn; //direction estimate
        float dzOutInAbs =  alpaka::math::abs(acc, zOut - zIn);
        float multDzDr = dzOutInAbs * coshEta / (coshEta * coshEta - 1.f);

        float kZ = (zOut - zIn) / dzSDIn;
        float sdlThetaMulsF = 0.015f * alpaka::math::sqrt(acc, 0.1f + 0.2f * (rtOut - rtIn) / 50.f);

        float sdlMuls = sdlThetaMulsF * 3.f / SDL::ptCut * 4.f; //will need a better guess than x4?

        float drtErr = alpaka::math::sqrt(acc, SDL::pixelPSZpitch * SDL::pixelPSZpitch * 2.f / (dzSDIn * dzSDIn) * (dzOutInAbs * dzOutInAbs) + sdlMuls * sdlMuls * multDzDr * multDzDr / 3.f * coshEta * coshEta);

        float drtMean = drtSDIn * dzOutInAbs/alpaka::math::abs(acc, dzSDIn);
        float rtWindow = drtErr + rtGeom;
        float rtLo_point = rtIn + drtMean / dzDrtScale - rtWindow;
        float rtHi_point = rtIn + drtMean + rtWindow;

        // Cut #3: rt-z pointed
        // https://github.com/slava77/cms-tkph2-ntuple/blob/superDoubletLinked-91X-noMock/doubletAnalysis.C#L3765

        if (isInSgInnerMDPS and isInSgOuterMDPS) // If both PS then we can point
        {
            pass = pass and ((kZ >= 0) &  (rtOut >= rtLo_point) & (rtOut <= rtHi_point));
        }

        return pass;
    };

    template<typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPointingConstraint(TAcc const & acc, struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& middleLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex, float& zOut, float& rtOut)
    {
        short innerInnerLowerModuleSubdet = modulesInGPU.subdets[innerInnerLowerModuleIndex];
        short middleLowerModuleSubdet = modulesInGPU.subdets[middleLowerModuleIndex];
        short outerOuterLowerModuleSubdet = modulesInGPU.subdets[outerOuterLowerModuleIndex];

        if(innerInnerLowerModuleSubdet == SDL::Barrel
                and middleLowerModuleSubdet == SDL::Barrel
                and outerOuterLowerModuleSubdet == SDL::Barrel)
        {
            return passPointingConstraintBBB(acc, modulesInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, middleLowerModuleIndex, outerOuterLowerModuleIndex, firstMDIndex, secondMDIndex, thirdMDIndex, zOut, rtOut);
        }
        else if(innerInnerLowerModuleSubdet == SDL::Barrel
                and middleLowerModuleSubdet == SDL::Barrel
                and outerOuterLowerModuleSubdet == SDL::Endcap)
        {
            return passPointingConstraintBBE(acc, modulesInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, middleLowerModuleIndex, outerOuterLowerModuleIndex, firstMDIndex, secondMDIndex, thirdMDIndex, zOut, rtOut);
        }
        else if(innerInnerLowerModuleSubdet == SDL::Barrel
                and middleLowerModuleSubdet == SDL::Endcap
                and outerOuterLowerModuleSubdet == SDL::Endcap)
        {
            return passPointingConstraintBBE(acc, modulesInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, middleLowerModuleIndex, outerOuterLowerModuleIndex, firstMDIndex, secondMDIndex, thirdMDIndex, zOut, rtOut);

        }

        else if(innerInnerLowerModuleSubdet == SDL::Endcap
                and middleLowerModuleSubdet == SDL::Endcap
                and outerOuterLowerModuleSubdet == SDL::Endcap)
        {
            return passPointingConstraintEEE(acc, modulesInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, middleLowerModuleIndex, outerOuterLowerModuleIndex, firstMDIndex, secondMDIndex, thirdMDIndex, zOut, rtOut);
        }
        return false; // failsafe    
    };

    void printTriplet(struct triplets& tripletsInGPU, struct segments& segmentsInGPU, struct miniDoublets& mdsInGPU, struct hits& hitsInGPU, struct modules& modulesInGPU, unsigned int tripletIndex);

    template<typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void runDeltaBetaIterationsT3(TAcc const & acc, float& betaIn, float& betaOut, float& betaAv, float & pt_beta, float sdIn_dr, float sdOut_dr, float dr, float lIn)
    {
        if (lIn == 0)
        {
            betaOut += copysign(alpaka::math::asin(acc, alpaka::math::min(acc, sdOut_dr * SDL::k2Rinv1GeVf / alpaka::math::abs(acc, pt_beta), SDL::sinAlphaMax)), betaOut);
            return;
        }

        if (betaIn * betaOut > 0.f and (alpaka::math::abs(acc, pt_beta) < 4.f * SDL::pt_betaMax or (lIn >= 11 and alpaka::math::abs(acc, pt_beta) < 8.f * SDL::pt_betaMax)))   //and the pt_beta is well-defined; less strict for endcap-endcap
        {
            const float betaInUpd  = betaIn + SDL::copysignf(alpaka::math::asin(acc, alpaka::math::min(acc, sdIn_dr * SDL::k2Rinv1GeVf / alpaka::math::abs(acc, pt_beta), SDL::sinAlphaMax)), betaIn); //FIXME: need a faster version
            const float betaOutUpd = betaOut + SDL::copysignf(alpaka::math::asin(acc, alpaka::math::min(acc, sdOut_dr * SDL::k2Rinv1GeVf / alpaka::math::abs(acc, pt_beta), SDL::sinAlphaMax)), betaOut); //FIXME: need a faster version
            betaAv = 0.5f * (betaInUpd + betaOutUpd);

            //1st update
            //pt_beta = dr * k2Rinv1GeVf / alpaka::math::sin(acc, betaAv); //get a better pt estimate
            const float pt_beta_inv = 1.f/alpaka::math::abs(acc, dr * k2Rinv1GeVf / alpaka::math::sin(acc, betaAv)); //get a better pt estimate

            betaIn  += SDL::copysignf(alpaka::math::asin(acc, alpaka::math::min(acc, sdIn_dr * SDL::k2Rinv1GeVf *pt_beta_inv, SDL::sinAlphaMax)), betaIn); //FIXME: need a faster version
            betaOut += SDL::copysignf(alpaka::math::asin(acc, alpaka::math::min(acc, sdOut_dr * SDL::k2Rinv1GeVf *pt_beta_inv, SDL::sinAlphaMax)), betaOut); //FIXME: need a faster version
            //update the av and pt
            betaAv = 0.5f * (betaIn + betaOut);
            //2nd update
            pt_beta = dr * SDL::k2Rinv1GeVf / alpaka::math::sin(acc, betaAv); //get a better pt estimate
        }
        else if (lIn < 11 && alpaka::math::abs(acc, betaOut) < 0.2f * alpaka::math::abs(acc, betaIn) && alpaka::math::abs(acc, pt_beta) < 12.f * SDL::pt_betaMax)   //use betaIn sign as ref
        {
            const float pt_betaIn = dr * k2Rinv1GeVf / alpaka::math::sin(acc, betaIn);

            const float betaInUpd  = betaIn + SDL::copysignf(alpaka::math::asin(acc, alpaka::math::min(acc, sdIn_dr * SDL::k2Rinv1GeVf / alpaka::math::abs(acc, pt_betaIn), SDL::sinAlphaMax)), betaIn); //FIXME: need a faster version
            const float betaOutUpd = betaOut + SDL::copysignf(alpaka::math::asin(acc, alpaka::math::min(acc, sdOut_dr * SDL::k2Rinv1GeVf / alpaka::math::abs(acc, pt_betaIn), SDL::sinAlphaMax)), betaIn); //FIXME: need a faster version
            betaAv = (alpaka::math::abs(acc, betaOut) > 0.2f * alpaka::math::abs(acc, betaIn)) ? (0.5f * (betaInUpd + betaOutUpd)) : betaInUpd;

            //1st update
            pt_beta = dr * SDL::k2Rinv1GeVf / alpaka::math::sin(acc, betaAv); //get a better pt estimate
            betaIn  += SDL::copysignf(alpaka::math::asin(acc, alpaka::math::min(acc, sdIn_dr * SDL::k2Rinv1GeVf / alpaka::math::abs(acc, pt_beta), SDL::sinAlphaMax)), betaIn); //FIXME: need a faster version
            betaOut += SDL::copysignf(alpaka::math::asin(acc, alpaka::math::min(acc, sdOut_dr * SDL::k2Rinv1GeVf / alpaka::math::abs(acc, pt_beta), SDL::sinAlphaMax)), betaIn); //FIXME: need a faster version
            //update the av and pt
            betaAv = 0.5f * (betaIn + betaOut);
            //2nd update
            pt_beta = dr * SDL::k2Rinv1GeVf / alpaka::math::sin(acc, betaAv); //get a better pt estimate
        }
    };

    template<typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runTripletDefaultAlgoBBBB(TAcc const & acc, struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& innerOuterLowerModuleIndex, uint16_t& outerInnerLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex, unsigned int& fourthMDIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& dPhi, float& betaIn, float&betaOut, float& pt_beta, float& zLo, float& zHi, float& zLoPointed, float& zHiPointed, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut)
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

        float alpha1GeV_OutLo = alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax));

        float rtRatio_OutLoInLo = rt_OutLo / rt_InLo; // Outer segment beginning rt divided by inner segment beginning rt;
        float dzDrtScale = alpaka::math::tan(acc, alpha1GeV_OutLo) / alpha1GeV_OutLo; // The track can bend in r-z plane slightly
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
        float r3_InLo = alpaka::math::sqrt(acc, z_InLo * z_InLo + rt_InLo * rt_InLo);
        float drt_InSeg = rt_InOut - rt_InLo;
        float dz_InSeg = z_InOut - z_InLo;
        float dr3_InSeg = alpaka::math::sqrt(acc, rt_InOut * rt_InOut + z_InOut * z_InOut) - alpaka::math::sqrt(acc, rt_InLo * rt_InLo + z_InLo * z_InLo);

        float coshEta = dr3_InSeg/drt_InSeg;
        float dzErr = (zpitch_InLo + zpitch_OutLo) * (zpitch_InLo + zpitch_OutLo) * 2.f;

        float sdlThetaMulsF = 0.015f * alpaka::math::sqrt(acc, 0.1f + 0.2f * (rt_OutLo - rt_InLo) / 50.f) * alpaka::math::sqrt(acc, r3_InLo / rt_InLo);
        float sdlMuls = sdlThetaMulsF * 3.f / SDL::ptCut * 4.f; // will need a better guess than x4?
        dzErr += sdlMuls * sdlMuls * drt_OutLo_InLo * drt_OutLo_InLo / 3.f * coshEta * coshEta; //sloppy
        dzErr = alpaka::math::sqrt(acc, dzErr);

        // Constructing upper and lower bound
        const float dzMean = dz_InSeg / drt_InSeg * drt_OutLo_InLo;
        const float zWindow = dzErr / drt_InSeg * drt_OutLo_InLo + (zpitch_InLo + zpitch_OutLo); //FIXME for SDL::ptCut lower than ~0.8 need to add curv path correction
        zLoPointed = z_InLo + dzMean * (z_InLo > 0.f ? 1.f : dzDrtScale) - zWindow;
        zHiPointed = z_InLo + dzMean * (z_InLo < 0.f ? 1.f : dzDrtScale) + zWindow;

        // Cut #2: Pointed Z (Inner segment two MD points to outer segment inner MD)
        pass =  pass and ((z_OutLo >= zLoPointed) & (z_OutLo <= zHiPointed));
        if(not pass) return pass;

        float sdlPVoff = 0.1f/rt_OutLo;
        sdlCut = alpha1GeV_OutLo + alpaka::math::sqrt(acc, sdlMuls * sdlMuls + sdlPVoff * sdlPVoff);

        deltaPhiPos = SDL::phi_mpi_pi(mdsInGPU.anchorPhi[fourthMDIndex]-mdsInGPU.anchorPhi[secondMDIndex]);
        // Cut #3: FIXME:deltaPhiPos can be tighter
        pass = pass and (alpaka::math::abs(acc, deltaPhiPos) <= sdlCut);
        if(not pass) return pass;

        float midPointX = 0.5f*(mdsInGPU.anchorX[firstMDIndex] + mdsInGPU.anchorX[thirdMDIndex]);
        float midPointY = 0.5f* (mdsInGPU.anchorY[firstMDIndex] + mdsInGPU.anchorY[thirdMDIndex]);
        float midPointZ = 0.5f*(mdsInGPU.anchorZ[firstMDIndex] + mdsInGPU.anchorZ[thirdMDIndex]);
        float diffX = mdsInGPU.anchorX[thirdMDIndex] - mdsInGPU.anchorX[firstMDIndex];
        float diffY = mdsInGPU.anchorY[thirdMDIndex] - mdsInGPU.anchorY[firstMDIndex];
        float diffZ = mdsInGPU.anchorZ[thirdMDIndex] - mdsInGPU.anchorZ[firstMDIndex];

        dPhi = SDL::deltaPhi(acc, midPointX, midPointY, diffX, diffY);

        // Cut #4: deltaPhiChange
        pass = pass and (alpaka::math::abs(acc, dPhi) <= sdlCut);
        //lots of array accesses below. Cut here!
        if(not pass) return pass;

        // First obtaining the raw betaIn and betaOut values without any correction and just purely based on the mini-doublet hit positions

        float alpha_InLo  = __H2F(segmentsInGPU.dPhiChanges[innerSegmentIndex]);
        float alpha_OutLo = __H2F(segmentsInGPU.dPhiChanges[outerSegmentIndex]);

        bool isEC_lastLayer = modulesInGPU.subdets[outerOuterLowerModuleIndex] == SDL::Endcap and modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::TwoS;

        float alpha_OutUp,alpha_OutUp_highEdge,alpha_OutUp_lowEdge;

        alpha_OutUp = SDL::phi_mpi_pi(SDL::phi(mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex], mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex])-mdsInGPU.anchorPhi[fourthMDIndex]);

        alpha_OutUp_highEdge = alpha_OutUp;
        alpha_OutUp_lowEdge = alpha_OutUp;

        float tl_axis_x = mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[firstMDIndex];
        float tl_axis_y = mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[firstMDIndex];
        float tl_axis_z = mdsInGPU.anchorZ[fourthMDIndex] - mdsInGPU.anchorZ[firstMDIndex];
        float tl_axis_highEdge_x = tl_axis_x;
        float tl_axis_highEdge_y = tl_axis_y;
        float tl_axis_lowEdge_x = tl_axis_x;
        float tl_axis_lowEdge_y = tl_axis_y;

        betaIn = alpha_InLo - SDL::phi_mpi_pi(SDL::phi(tl_axis_x, tl_axis_y)-mdsInGPU.anchorPhi[firstMDIndex]);

        float betaInRHmin = betaIn;
        float betaInRHmax = betaIn;
        betaOut = -alpha_OutUp + SDL::phi_mpi_pi(SDL::phi(tl_axis_x, tl_axis_y)-mdsInGPU.anchorPhi[fourthMDIndex]);

        float betaOutRHmin = betaOut;
        float betaOutRHmax = betaOut;

        if(isEC_lastLayer)
        {
            alpha_OutUp_highEdge = SDL::phi_mpi_pi(SDL::phi(mdsInGPU.anchorHighEdgeX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex], mdsInGPU.anchorHighEdgeY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex])-mdsInGPU.anchorHighEdgePhi[fourthMDIndex]);
            alpha_OutUp_lowEdge = SDL::phi_mpi_pi(SDL::phi(mdsInGPU.anchorLowEdgeX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex], mdsInGPU.anchorLowEdgeY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex])-mdsInGPU.anchorLowEdgePhi[fourthMDIndex]);

            tl_axis_highEdge_x = mdsInGPU.anchorHighEdgeX[fourthMDIndex] - mdsInGPU.anchorX[firstMDIndex];
            tl_axis_highEdge_y = mdsInGPU.anchorHighEdgeY[fourthMDIndex] - mdsInGPU.anchorY[firstMDIndex];
            tl_axis_lowEdge_x = mdsInGPU.anchorLowEdgeX[fourthMDIndex] - mdsInGPU.anchorX[firstMDIndex];
            tl_axis_lowEdge_y = mdsInGPU.anchorLowEdgeY[fourthMDIndex] - mdsInGPU.anchorY[firstMDIndex];

            betaOutRHmin = -alpha_OutUp_highEdge + SDL::phi_mpi_pi(SDL::phi(tl_axis_highEdge_x, tl_axis_highEdge_y)-mdsInGPU.anchorHighEdgePhi[fourthMDIndex]);
            betaOutRHmax = -alpha_OutUp_lowEdge + SDL::phi_mpi_pi(SDL::phi(tl_axis_lowEdge_x, tl_axis_lowEdge_y)-mdsInGPU.anchorLowEdgePhi[fourthMDIndex]);
        }

        //beta computation
        float drt_tl_axis = alpaka::math::sqrt(acc, tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);
        float drt_tl_lowEdge = alpaka::math::sqrt(acc, tl_axis_lowEdge_x * tl_axis_lowEdge_x + tl_axis_lowEdge_y * tl_axis_lowEdge_y);
        float drt_tl_highEdge = alpaka::math::sqrt(acc, tl_axis_highEdge_x * tl_axis_highEdge_x + tl_axis_highEdge_y * tl_axis_highEdge_y);

        float corrF = 1.f;
        //innerOuterAnchor - innerInnerAnchor
        const float rt_InSeg = alpaka::math::sqrt(acc, (mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) * (mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) + (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]) * (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]));
        betaInCut = alpaka::math::asin(acc, alpaka::math::min(acc, (-rt_InSeg * corrF + drt_tl_axis) * SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax)) + (0.02f / drt_InSeg);

        //Cut #5: first beta cut
        pass = pass and (alpaka::math::abs(acc, betaInRHmin) < betaInCut);
        if(not pass) return pass;

        float betaAv = 0.5f * (betaIn + betaOut);
        pt_beta = drt_tl_axis * SDL::k2Rinv1GeVf/alpaka::math::sin(acc, betaAv);
        int lIn = 5;
        int lOut = isEC_lastLayer ? 11 : 5;
        float sdOut_dr = alpaka::math::sqrt(acc, (mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex]) * (mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex]) + (mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]) * (mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]));
        float sdOut_d = mdsInGPU.anchorRt[fourthMDIndex] - mdsInGPU.anchorRt[thirdMDIndex];

        const float diffDr = alpaka::math::abs(acc, rt_InSeg - sdOut_dr) / alpaka::math::abs(acc, rt_InSeg + sdOut_dr);

        runDeltaBetaIterationsT3(acc, betaIn, betaOut, betaAv, pt_beta, rt_InSeg, sdOut_dr, drt_tl_axis, lIn);

        const float betaInMMSF = (alpaka::math::abs(acc, betaInRHmin + betaInRHmax) > 0) ? (2.f * betaIn / alpaka::math::abs(acc, betaInRHmin + betaInRHmax)) : 0.f; //mean value of min,max is the old betaIn
        const float betaOutMMSF = (alpaka::math::abs(acc, betaOutRHmin + betaOutRHmax) > 0) ? (2.f * betaOut / alpaka::math::abs(acc, betaOutRHmin + betaOutRHmax)) : 0.f;
        betaInRHmin *= betaInMMSF;
        betaInRHmax *= betaInMMSF;
        betaOutRHmin *= betaOutMMSF;
        betaOutRHmax *= betaOutMMSF;

        const float dBetaMuls = sdlThetaMulsF * 4.f / alpaka::math::min(acc, alpaka::math::abs(acc, pt_beta), SDL::pt_betaMax); //need to confirm the range-out value of 7 GeV


        const float alphaInAbsReg = alpaka::math::max(acc, alpaka::math::abs(acc, alpha_InLo), alpaka::math::asin(acc, alpaka::math::min(acc, rt_InLo * SDL::k2Rinv1GeVf / 3.0f, SDL::sinAlphaMax)));
        const float alphaOutAbsReg = alpaka::math::max(acc, alpaka::math::abs(acc, alpha_OutLo), alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * SDL::k2Rinv1GeVf / 3.0f, SDL::sinAlphaMax)));
        const float dBetaInLum = lIn < 11 ? 0.0f : alpaka::math::abs(acc, alphaInAbsReg*SDL::deltaZLum / z_InLo);
        const float dBetaOutLum = lOut < 11 ? 0.0f : alpaka::math::abs(acc, alphaOutAbsReg*SDL::deltaZLum / z_OutLo);
        const float dBetaLum2 = (dBetaInLum + dBetaOutLum) * (dBetaInLum + dBetaOutLum);
        const float sinDPhi = alpaka::math::sin(acc, dPhi);

        const float dBetaRIn2 = 0; // TODO-RH
        // const float dBetaROut2 = 0; // TODO-RH
        float dBetaROut = 0;
        if(isEC_lastLayer)
        {
            dBetaROut = (alpaka::math::sqrt(acc, mdsInGPU.anchorHighEdgeX[fourthMDIndex] * mdsInGPU.anchorHighEdgeX[fourthMDIndex] + mdsInGPU.anchorHighEdgeY[fourthMDIndex] * mdsInGPU.anchorHighEdgeY[fourthMDIndex]) - alpaka::math::sqrt(acc, mdsInGPU.anchorLowEdgeX[fourthMDIndex] * mdsInGPU.anchorLowEdgeX[fourthMDIndex] + mdsInGPU.anchorLowEdgeY[fourthMDIndex] * mdsInGPU.anchorLowEdgeY[fourthMDIndex])) * sinDPhi / drt_tl_axis;
        }

        const float dBetaROut2 =  dBetaROut * dBetaROut;

        betaOutCut = alpaka::math::asin(acc, alpaka::math::min(acc, drt_tl_axis*SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax)) //FIXME: need faster version
            + (0.02f / sdOut_d) + alpaka::math::sqrt(acc, dBetaLum2 + dBetaMuls*dBetaMuls);

        //Cut #6: The real beta cut
        pass =  pass and ((alpaka::math::abs(acc, betaOut) < betaOutCut));
        if(not pass) return pass;

        float pt_betaIn = drt_tl_axis * SDL::k2Rinv1GeVf/alpaka::math::sin(acc, betaIn);
        float pt_betaOut = drt_tl_axis * SDL::k2Rinv1GeVf / alpaka::math::sin(acc, betaOut);
        float dBetaRes = 0.02f/alpaka::math::min(acc, sdOut_d,drt_InSeg);
        float dBetaCut2 = (dBetaRes*dBetaRes * 2.0f + dBetaMuls * dBetaMuls + dBetaLum2 + dBetaRIn2 + dBetaROut2
                + 0.25f * (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)) * (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)));

        float dBeta = betaIn - betaOut;
        deltaBetaCut = alpaka::math::sqrt(acc, dBetaCut2);
        pass = pass and (dBeta * dBeta <= dBetaCut2);

        return pass;
    };

    template<typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runTripletDefaultAlgoBBEE(TAcc const & acc, struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& innerOuterLowerModuleIndex, uint16_t& outerInnerLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex, unsigned int& fourthMDIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& dPhi, float& betaIn, float&betaOut, float& pt_beta, float& zLo, float& rtLo, float& rtHi, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ)
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

        float alpha1GeV_OutLo = alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax));

        float rtRatio_OutLoInLo = rt_OutLo / rt_InLo; // Outer segment beginning rt divided by inner segment beginning rt;
        float dzDrtScale = alpaka::math::tan(acc, alpha1GeV_OutLo) / alpha1GeV_OutLo; // The track can bend in r-z plane slightly
        float zpitch_InLo = (isPS_InLo ? SDL::pixelPSZpitch : SDL::strip2SZpitch);
        float zpitch_OutLo = (isPS_OutLo ? SDL::pixelPSZpitch : SDL::strip2SZpitch);
        float zGeom = zpitch_InLo + zpitch_OutLo;

        zLo = z_InLo + (z_InLo - SDL::deltaZLum) * (rtRatio_OutLoInLo - 1.f) * (z_InLo > 0.f ? 1.f : dzDrtScale) - zGeom;

        // Cut #0: Preliminary (Only here in endcap case)
        pass = pass and (z_InLo * z_OutLo > 0);
        if(not pass) return pass;

        float dLum = SDL::copysignf(SDL::deltaZLum, z_InLo);
        bool isOutSgInnerMDPS = modulesInGPU.moduleType[outerInnerLowerModuleIndex] == SDL::PS;
        float rtGeom1 = isOutSgInnerMDPS ? SDL::pixelPSZpitch : SDL::strip2SZpitch;
        float zGeom1 = SDL::copysignf(zGeom,z_InLo);
        rtLo = rt_InLo * (1.f + (z_OutLo - z_InLo - zGeom1) / (z_InLo + zGeom1 + dLum) / dzDrtScale) - rtGeom1; //slope correction only on the lower end
        zOut = z_OutLo;
        rtOut = rt_OutLo;

        //Cut #1: rt condition
        pass =  pass and (rtOut >= rtLo);
        if(not pass) return pass;

        float zInForHi = z_InLo - zGeom1 - dLum;
        if(zInForHi * z_InLo < 0)
        {
            zInForHi = SDL::copysignf(0.1f,z_InLo);
        }
        rtHi = rt_InLo * (1.f + (z_OutLo - z_InLo + zGeom1) / zInForHi) + rtGeom1;

        //Cut #2: rt condition
        pass =  pass and ((rt_OutLo >= rtLo) & (rt_OutLo <= rtHi));
        if(not pass) return pass;

        float rIn = alpaka::math::sqrt(acc, z_InLo * z_InLo + rt_InLo * rt_InLo);
        const float drtSDIn = rt_InOut - rt_InLo;
        const float dzSDIn = z_InOut - z_InLo;
        const float dr3SDIn = alpaka::math::sqrt(acc, rt_InOut * rt_InOut + z_InOut * z_InOut) - alpaka::math::sqrt(acc, rt_InLo * rt_InLo + z_InLo * z_InLo);

        const float coshEta = dr3SDIn / drtSDIn; //direction estimate
        const float dzOutInAbs = alpaka::math::abs(acc, z_OutLo - z_InLo);
        const float multDzDr = dzOutInAbs * coshEta / (coshEta * coshEta - 1.f);
        const float zGeom1_another = SDL::pixelPSZpitch;
        kZ = (z_OutLo - z_InLo) / dzSDIn;
        float drtErr = zGeom1_another * zGeom1_another * drtSDIn * drtSDIn / dzSDIn / dzSDIn * (1.f - 2.f * kZ + 2.f * kZ * kZ);
        const float sdlThetaMulsF = 0.015f * alpaka::math::sqrt(acc, 0.1f + 0.2f * (rt_OutLo - rt_InLo) / 50.f) * alpaka::math::sqrt(acc, rIn / rt_InLo);
        const float sdlMuls = sdlThetaMulsF * 3.f / SDL::ptCut * 4.f; //will need a better guess than x4?
        drtErr += sdlMuls * sdlMuls * multDzDr * multDzDr / 3.f * coshEta * coshEta; //sloppy: relative muls is 1/3 of total muls
        drtErr = alpaka::math::sqrt(acc, drtErr);
        const float drtMean = drtSDIn * dzOutInAbs / alpaka::math::abs(acc, dzSDIn);
        const float rtWindow = drtErr + rtGeom1;
        const float rtLo_another = rt_InLo + drtMean / dzDrtScale - rtWindow;
        const float rtHi_another = rt_InLo + drtMean + rtWindow;

        //Cut #3: rt-z pointed
        pass =  pass and ((kZ >= 0) & (rtOut >= rtLo) & (rtOut <= rtHi));
        if(not pass) return pass;

        const float sdlPVoff = 0.1f / rt_OutLo;
        sdlCut = alpha1GeV_OutLo + alpaka::math::sqrt(acc, sdlMuls * sdlMuls + sdlPVoff*sdlPVoff);

        deltaPhiPos = SDL::phi_mpi_pi(mdsInGPU.anchorPhi[fourthMDIndex]-mdsInGPU.anchorPhi[secondMDIndex]);

        //Cut #4: deltaPhiPos can be tighter
        pass =  pass and (alpaka::math::abs(acc, deltaPhiPos) <= sdlCut);
        if(not pass) return pass;

        float midPointX = 0.5f*(mdsInGPU.anchorX[firstMDIndex] + mdsInGPU.anchorX[thirdMDIndex]);
        float midPointY = 0.5f* (mdsInGPU.anchorY[firstMDIndex] + mdsInGPU.anchorY[thirdMDIndex]);
        float midPointZ = 0.5f*(mdsInGPU.anchorZ[firstMDIndex] + mdsInGPU.anchorZ[thirdMDIndex]);
        float diffX = mdsInGPU.anchorX[thirdMDIndex] - mdsInGPU.anchorX[firstMDIndex];
        float diffY = mdsInGPU.anchorY[thirdMDIndex] - mdsInGPU.anchorY[firstMDIndex];
        float diffZ = mdsInGPU.anchorZ[thirdMDIndex] - mdsInGPU.anchorZ[firstMDIndex];

        dPhi = SDL::deltaPhi(acc, midPointX, midPointY, diffX, diffY);
        // Cut #5: deltaPhiChange
        pass =  pass and (alpaka::math::abs(acc, dPhi) <= sdlCut);
        if(not pass) return pass;

        float sdIn_alpha     = __H2F(segmentsInGPU.dPhiChanges[innerSegmentIndex]);
        float sdIn_alpha_min = __H2F(segmentsInGPU.dPhiChangeMins[innerSegmentIndex]);
        float sdIn_alpha_max = __H2F(segmentsInGPU.dPhiChangeMaxs[innerSegmentIndex]);
        float sdOut_alpha = sdIn_alpha; //weird

        float sdOut_alphaOut = SDL::phi_mpi_pi(SDL::phi(mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex], mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex])-mdsInGPU.anchorPhi[fourthMDIndex]);

        float sdOut_alphaOut_min = SDL::phi_mpi_pi(acc, __H2F(segmentsInGPU.dPhiChangeMins[outerSegmentIndex]) - __H2F(segmentsInGPU.dPhiMins[outerSegmentIndex]));
        float sdOut_alphaOut_max = SDL::phi_mpi_pi(acc, __H2F(segmentsInGPU.dPhiChangeMaxs[outerSegmentIndex]) - __H2F(segmentsInGPU.dPhiMaxs[outerSegmentIndex]));

        float tl_axis_x = mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[firstMDIndex];
        float tl_axis_y = mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[firstMDIndex];
        float tl_axis_z = mdsInGPU.anchorZ[fourthMDIndex] - mdsInGPU.anchorZ[firstMDIndex];

        betaIn = sdIn_alpha - SDL::phi_mpi_pi(SDL::phi(tl_axis_x, tl_axis_y)-mdsInGPU.anchorPhi[firstMDIndex]);

        float betaInRHmin = betaIn;
        float betaInRHmax = betaIn;
        betaOut = -sdOut_alphaOut + SDL::phi_mpi_pi(SDL::phi(tl_axis_x, tl_axis_y)-mdsInGPU.anchorPhi[fourthMDIndex]);

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
        if(alpaka::math::abs(acc, betaOutRHmin) > alpaka::math::abs(acc, betaOutRHmax))
        {
            swapTemp = betaOutRHmin;
            betaOutRHmin = betaOutRHmax;
            betaOutRHmax = swapTemp;
        }

        if(alpaka::math::abs(acc, betaInRHmin) > alpaka::math::abs(acc, betaInRHmax))
        {
            swapTemp = betaInRHmin;
            betaInRHmin = betaInRHmax;
            betaInRHmax = swapTemp;
        }

        float sdIn_dr = alpaka::math::sqrt(acc, (mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) * (mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) + (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]) * (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]));
        float sdIn_d = rt_InOut - rt_InLo;

        float dr = alpaka::math::sqrt(acc, tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);
        const float corrF = 1.f;
        betaInCut = alpaka::math::asin(acc, alpaka::math::min(acc, (-sdIn_dr * corrF + dr) * SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax)) + (0.02f / sdIn_d);

        //Cut #6: first beta cut
        pass =  pass and (alpaka::math::abs(acc, betaInRHmin) < betaInCut);
        if(not pass) return pass;

        float betaAv = 0.5f * (betaIn + betaOut);
        pt_beta = dr * SDL::k2Rinv1GeVf / alpaka::math::sin(acc, betaAv);

        float lIn = 5;
        float lOut = 11;

        float sdOut_dr = alpaka::math::sqrt(acc, (mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex]) * (mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex]) + (mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]) * (mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]));
        float sdOut_d = mdsInGPU.anchorRt[fourthMDIndex] - mdsInGPU.anchorRt[thirdMDIndex];

        runDeltaBetaIterationsT3(acc, betaIn, betaOut, betaAv, pt_beta, sdIn_dr, sdOut_dr, dr, lIn);

        const float betaInMMSF = (alpaka::math::abs(acc, betaInRHmin + betaInRHmax) > 0) ? (2.f * betaIn / alpaka::math::abs(acc, betaInRHmin + betaInRHmax)) : 0.; //mean value of min,max is the old betaIn
        const float betaOutMMSF = (alpaka::math::abs(acc, betaOutRHmin + betaOutRHmax) > 0) ? (2.f * betaOut / alpaka::math::abs(acc, betaOutRHmin + betaOutRHmax)) : 0.;
        betaInRHmin *= betaInMMSF;
        betaInRHmax *= betaInMMSF;
        betaOutRHmin *= betaOutMMSF;
        betaOutRHmax *= betaOutMMSF;

        const float dBetaMuls = sdlThetaMulsF * 4.f / alpaka::math::min(acc, alpaka::math::abs(acc, pt_beta), SDL::pt_betaMax); //need to confirm the range-out value of 7 GeV

        const float alphaInAbsReg = alpaka::math::max(acc, alpaka::math::abs(acc, sdIn_alpha), alpaka::math::asin(acc, alpaka::math::min(acc, rt_InLo * SDL::k2Rinv1GeVf / 3.0f, SDL::sinAlphaMax)));
        const float alphaOutAbsReg = alpaka::math::max(acc, alpaka::math::abs(acc, sdOut_alpha), alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * SDL::k2Rinv1GeVf / 3.0f, SDL::sinAlphaMax)));
        const float dBetaInLum = lIn < 11 ? 0.0f : alpaka::math::abs(acc, alphaInAbsReg*SDL::deltaZLum / z_InLo);
        const float dBetaOutLum = lOut < 11 ? 0.0f : alpaka::math::abs(acc, alphaOutAbsReg*SDL::deltaZLum / z_OutLo);
        const float dBetaLum2 = (dBetaInLum + dBetaOutLum) * (dBetaInLum + dBetaOutLum);
        const float sinDPhi = alpaka::math::sin(acc, dPhi);

        const float dBetaRIn2 = 0; // TODO-RH
        // const float dBetaROut2 = 0; // TODO-RH
        float dBetaROut = 0;
        if(modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::TwoS)
        {
            dBetaROut = (alpaka::math::sqrt(acc, mdsInGPU.anchorHighEdgeX[fourthMDIndex] * mdsInGPU.anchorHighEdgeX[fourthMDIndex] + mdsInGPU.anchorHighEdgeY[fourthMDIndex] * mdsInGPU.anchorHighEdgeY[fourthMDIndex]) - alpaka::math::sqrt(acc, mdsInGPU.anchorLowEdgeX[fourthMDIndex] * mdsInGPU.anchorLowEdgeX[fourthMDIndex] + mdsInGPU.anchorLowEdgeY[fourthMDIndex] * mdsInGPU.anchorLowEdgeY[fourthMDIndex])) * sinDPhi / dr;
        }

        const float dBetaROut2 = dBetaROut * dBetaROut;
        betaOutCut = alpaka::math::asin(acc, alpaka::math::min(acc, dr*SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax)) //FIXME: need faster version
            + (0.02f / sdOut_d) + alpaka::math::sqrt(acc, dBetaLum2 + dBetaMuls*dBetaMuls);

        //Cut #6: The real beta cut
        pass =  pass and (alpaka::math::abs(acc, betaOut) < betaOutCut);
        if(not pass) return pass;

        float pt_betaIn = dr * SDL::k2Rinv1GeVf/alpaka::math::sin(acc, betaIn);
        float pt_betaOut = dr * SDL::k2Rinv1GeVf / alpaka::math::sin(acc, betaOut);
        float dBetaRes = 0.02f/alpaka::math::min(acc, sdOut_d,sdIn_d);
        float dBetaCut2 = (dBetaRes*dBetaRes * 2.0f + dBetaMuls * dBetaMuls + dBetaLum2 + dBetaRIn2 + dBetaROut2
                + 0.25f * (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)) * (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)));
        float dBeta = betaIn - betaOut;
        deltaBetaCut = alpaka::math::sqrt(acc, dBetaCut2);
        //Cut #7: Cut on dBet
        pass =  pass and (dBeta * dBeta <= dBetaCut2);

        return pass;
    };

    template<typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runTripletDefaultAlgoEEEE(TAcc const & acc, struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& innerOuterLowerModuleIndex, uint16_t& outerInnerLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex, unsigned int& fourthMDIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& dPhi, float& betaIn, float&betaOut, float& pt_beta, float& zLo, float& rtLo, float& rtHi, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ)
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

        float alpha1GeV_OutLo = alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax));

        float rtRatio_OutLoInLo = rt_OutLo / rt_InLo; // Outer segment beginning rt divided by inner segment beginning rt;
        float dzDrtScale = alpaka::math::tan(acc, alpha1GeV_OutLo) / alpha1GeV_OutLo; // The track can bend in r-z plane slightly
        float zpitch_InLo = (isPS_InLo ? SDL::pixelPSZpitch : SDL::strip2SZpitch);
        float zpitch_OutLo = (isPS_OutLo ? SDL::pixelPSZpitch : SDL::strip2SZpitch);
        float zGeom = zpitch_InLo + zpitch_OutLo;

        zLo = z_InLo + (z_InLo - SDL::deltaZLum) * (rtRatio_OutLoInLo - 1.f) * (z_InLo > 0.f ? 1.f : dzDrtScale) - zGeom; //slope-correction only on outer end

        // Cut #0: Preliminary (Only here in endcap case)
        pass =  pass and ((z_InLo * z_OutLo) > 0);
        if(not pass) return pass;

        float dLum = SDL::copysignf(SDL::deltaZLum, z_InLo);
        bool isOutSgInnerMDPS = modulesInGPU.moduleType[outerInnerLowerModuleIndex] == SDL::PS;
        bool isInSgInnerMDPS = modulesInGPU.moduleType[innerInnerLowerModuleIndex] == SDL::PS;

        float rtGeom = (isInSgInnerMDPS and isOutSgInnerMDPS) ? 2.f * SDL::pixelPSZpitch : (isInSgInnerMDPS or isOutSgInnerMDPS) ? SDL::pixelPSZpitch + SDL::strip2SZpitch : 2.f * SDL::strip2SZpitch;

        float zGeom1 = SDL::copysignf(zGeom,z_InLo);
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
        const float dr3SDIn = alpaka::math::sqrt(acc, rt_InOut * rt_InOut + z_InOut * z_InOut) - alpaka::math::sqrt(acc, rt_InLo * rt_InLo + z_InLo * z_InLo);
        float coshEta = dr3SDIn / drtSDIn; //direction estimate
        float dzOutInAbs =  alpaka::math::abs(acc, z_OutLo - z_InLo);
        float multDzDr = dzOutInAbs * coshEta / (coshEta * coshEta - 1.f);

        kZ = (z_OutLo - z_InLo) / dzSDIn;
        float sdlThetaMulsF = 0.015f * alpaka::math::sqrt(acc, 0.1f + 0.2f * (rt_OutLo - rt_InLo) / 50.f);

        float sdlMuls = sdlThetaMulsF * 3.f / SDL::ptCut * 4.f; //will need a better guess than x4?

        float drtErr = alpaka::math::sqrt(acc, SDL::pixelPSZpitch * SDL::pixelPSZpitch * 2.f / (dzSDIn * dzSDIn) * (dzOutInAbs * dzOutInAbs) + sdlMuls * sdlMuls * multDzDr * multDzDr / 3.f * coshEta * coshEta);

        float drtMean = drtSDIn * dzOutInAbs/alpaka::math::abs(acc, dzSDIn);
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
        sdlCut = alpha1GeV_OutLo + alpaka::math::sqrt(acc, sdlMuls * sdlMuls + sdlPVoff * sdlPVoff);

        deltaPhiPos = SDL::phi_mpi_pi(mdsInGPU.anchorPhi[fourthMDIndex]-mdsInGPU.anchorPhi[secondMDIndex]);

        pass =  pass and (alpaka::math::abs(acc, deltaPhiPos) <= sdlCut);
        if(not pass) return pass;

        float midPointX = 0.5f*(mdsInGPU.anchorX[firstMDIndex] + mdsInGPU.anchorX[thirdMDIndex]);
        float midPointY = 0.5f* (mdsInGPU.anchorY[firstMDIndex] + mdsInGPU.anchorY[thirdMDIndex]);
        float midPointZ = 0.5f*(mdsInGPU.anchorZ[firstMDIndex] + mdsInGPU.anchorZ[thirdMDIndex]);
        float diffX = mdsInGPU.anchorX[thirdMDIndex] - mdsInGPU.anchorX[firstMDIndex];
        float diffY = mdsInGPU.anchorY[thirdMDIndex] - mdsInGPU.anchorY[firstMDIndex];
        float diffZ = mdsInGPU.anchorZ[thirdMDIndex] - mdsInGPU.anchorZ[firstMDIndex];

        dPhi = SDL::deltaPhi(acc, midPointX, midPointY, diffX, diffY);

        // Cut #5: deltaPhiChange
        pass =  pass and ((alpaka::math::abs(acc, dPhi) <= sdlCut));
        if(not pass) return pass;

        float sdIn_alpha = __H2F(segmentsInGPU.dPhiChanges[innerSegmentIndex]);
        float sdOut_alpha = sdIn_alpha; //weird
        float sdOut_dPhiPos = SDL::phi_mpi_pi(mdsInGPU.anchorPhi[fourthMDIndex]-mdsInGPU.anchorPhi[thirdMDIndex]);

        float sdOut_dPhiChange = __H2F(segmentsInGPU.dPhiChanges[outerSegmentIndex]);
        float sdOut_dPhiChange_min = __H2F(segmentsInGPU.dPhiChangeMins[outerSegmentIndex]);
        float sdOut_dPhiChange_max = __H2F(segmentsInGPU.dPhiChangeMaxs[outerSegmentIndex]);

        float sdOut_alphaOutRHmin = SDL::phi_mpi_pi(acc, sdOut_dPhiChange_min - sdOut_dPhiPos);
        float sdOut_alphaOutRHmax = SDL::phi_mpi_pi(acc, sdOut_dPhiChange_max - sdOut_dPhiPos);
        float sdOut_alphaOut = SDL::phi_mpi_pi(acc, sdOut_dPhiChange - sdOut_dPhiPos);

        float tl_axis_x = mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[firstMDIndex];
        float tl_axis_y = mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[firstMDIndex];
        float tl_axis_z = mdsInGPU.anchorZ[fourthMDIndex] - mdsInGPU.anchorZ[firstMDIndex];

        betaIn = sdIn_alpha - SDL::phi_mpi_pi(SDL::phi(tl_axis_x, tl_axis_y)-mdsInGPU.anchorPhi[firstMDIndex]);

        float sdIn_alphaRHmin = __H2F(segmentsInGPU.dPhiChangeMins[innerSegmentIndex]);
        float sdIn_alphaRHmax = __H2F(segmentsInGPU.dPhiChangeMaxs[innerSegmentIndex]);
        float betaInRHmin = betaIn + sdIn_alphaRHmin - sdIn_alpha;
        float betaInRHmax = betaIn + sdIn_alphaRHmax - sdIn_alpha;

        betaOut = -sdOut_alphaOut + SDL::phi_mpi_pi(SDL::phi(tl_axis_x, tl_axis_y)-mdsInGPU.anchorPhi[fourthMDIndex]);

        float betaOutRHmin = betaOut - sdOut_alphaOutRHmin + sdOut_alphaOut;
        float betaOutRHmax = betaOut - sdOut_alphaOutRHmax + sdOut_alphaOut;

        float swapTemp;
        if(alpaka::math::abs(acc, betaOutRHmin) > alpaka::math::abs(acc, betaOutRHmax))
        {
            swapTemp = betaOutRHmin;
            betaOutRHmin = betaOutRHmax;
            betaOutRHmax = swapTemp;
        }

        if(alpaka::math::abs(acc, betaInRHmin) > alpaka::math::abs(acc, betaInRHmax))
        {
            swapTemp = betaInRHmin;
            betaInRHmin = betaInRHmax;
            betaInRHmax = swapTemp;
        }
        float sdIn_dr = alpaka::math::sqrt(acc, (mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) * (mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) + (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]) * (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]));
        float sdIn_d = rt_InOut - rt_InLo;

        float dr = alpaka::math::sqrt(acc, tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);
        const float corrF = 1.f;
        betaInCut = alpaka::math::asin(acc, alpaka::math::min(acc, (-sdIn_dr * corrF + dr) * SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax)) + (0.02f / sdIn_d);

        //Cut #6: first beta cut
        pass =  pass and (alpaka::math::abs(acc, betaInRHmin) < betaInCut);
        if(not pass) return pass;

        float betaAv = 0.5f * (betaIn + betaOut);
        pt_beta = dr * SDL::k2Rinv1GeVf / alpaka::math::sin(acc, betaAv);


        int lIn= 11; //endcap
        int lOut = 13; //endcap

        float sdOut_dr = alpaka::math::sqrt(acc, (mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex]) * (mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex]) + (mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]) * (mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]));
        float sdOut_d = mdsInGPU.anchorRt[fourthMDIndex] - mdsInGPU.anchorRt[thirdMDIndex];

        float diffDr = alpaka::math::abs(acc, sdIn_dr - sdOut_dr)/alpaka::math::abs(acc, sdIn_dr + sdOut_dr);

        runDeltaBetaIterationsT3(acc, betaIn, betaOut, betaAv, pt_beta, sdIn_dr, sdOut_dr, dr, lIn);

        const float betaInMMSF = (alpaka::math::abs(acc, betaInRHmin + betaInRHmax) > 0) ? (2.f * betaIn / alpaka::math::abs(acc, betaInRHmin + betaInRHmax)) : 0.; //mean value of min,max is the old betaIn
        const float betaOutMMSF = (alpaka::math::abs(acc, betaOutRHmin + betaOutRHmax) > 0) ? (2.f * betaOut / alpaka::math::abs(acc, betaOutRHmin + betaOutRHmax)) : 0.;
        betaInRHmin *= betaInMMSF;
        betaInRHmax *= betaInMMSF;
        betaOutRHmin *= betaOutMMSF;
        betaOutRHmax *= betaOutMMSF;

        const float dBetaMuls = sdlThetaMulsF * 4.f / alpaka::math::min(acc, alpaka::math::abs(acc, pt_beta), SDL::pt_betaMax); //need to confirm the range-out value of 7 GeV

        const float alphaInAbsReg = alpaka::math::max(acc, alpaka::math::abs(acc, sdIn_alpha), alpaka::math::asin(acc, alpaka::math::min(acc, rt_InLo * SDL::k2Rinv1GeVf / 3.0f, SDL::sinAlphaMax)));
        const float alphaOutAbsReg = alpaka::math::max(acc, alpaka::math::abs(acc, sdOut_alpha), alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * SDL::k2Rinv1GeVf / 3.0f, SDL::sinAlphaMax)));
        const float dBetaInLum = lIn < 11 ? 0.0f : alpaka::math::abs(acc, alphaInAbsReg*SDL::deltaZLum / z_InLo);
        const float dBetaOutLum = lOut < 11 ? 0.0f : alpaka::math::abs(acc, alphaOutAbsReg*SDL::deltaZLum / z_OutLo);
        const float dBetaLum2 = (dBetaInLum + dBetaOutLum) * (dBetaInLum + dBetaOutLum);
        const float sinDPhi = alpaka::math::sin(acc, dPhi);

        const float dBetaRIn2 = 0; // TODO-RH
        // const float dBetaROut2 = 0; // TODO-RH
        float dBetaROut2 = 0;//TODO-RH
        betaOutCut = alpaka::math::asin(acc, alpaka::math::min(acc, dr*SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax)) //FIXME: need faster version
            + (0.02f / sdOut_d) + alpaka::math::sqrt(acc, dBetaLum2 + dBetaMuls*dBetaMuls);

        //Cut #6: The real beta cut
        pass =  pass and (alpaka::math::abs(acc, betaOut) < betaOutCut);
        if(not pass) return pass;

        float pt_betaIn = dr * SDL::k2Rinv1GeVf/alpaka::math::sin(acc, betaIn);
        float pt_betaOut = dr * SDL::k2Rinv1GeVf / alpaka::math::sin(acc, betaOut);
        float dBetaRes = 0.02f/alpaka::math::min(acc, sdOut_d,sdIn_d);
        float dBetaCut2 = (dBetaRes*dBetaRes * 2.0f + dBetaMuls * dBetaMuls + dBetaLum2 + dBetaRIn2 + dBetaROut2
                + 0.25f * (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)) * (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)));
        float dBeta = betaIn - betaOut;
        //Cut #7: Cut on dBeta
        deltaBetaCut = alpaka::math::sqrt(acc, dBetaCut2);

        pass =  pass and (dBeta * dBeta <= dBetaCut2);

        return pass;
    };

    template<typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runTripletDefaultAlgo(TAcc const & acc, struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& innerOuterLowerModuleIndex, uint16_t& outerInnerLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, unsigned int& firstMDIndex, unsigned int& secondMDIndex, unsigned int& thirdMDIndex, unsigned int& fourthMDIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& deltaPhi, float& betaIn, float&betaOut, float& pt_beta, float& zLo, float& zHi, float& rtLo, float& rtHi, float& zLoPointed, float& zHiPointed, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ)
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
            return runTripletDefaultAlgoBBBB(acc, modulesInGPU,mdsInGPU,segmentsInGPU,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,innerSegmentIndex,outerSegmentIndex,firstMDIndex, secondMDIndex, thirdMDIndex, fourthMDIndex, zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta, zLo, zHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut);
        }

        else if(innerInnerLowerModuleSubdet == SDL::Barrel
                and innerOuterLowerModuleSubdet == SDL::Barrel
                and outerInnerLowerModuleSubdet == SDL::Endcap
                and outerOuterLowerModuleSubdet == SDL::Endcap)
        {
        return runTripletDefaultAlgoBBEE(acc, modulesInGPU,mdsInGPU,segmentsInGPU,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,innerSegmentIndex,outerSegmentIndex,firstMDIndex, secondMDIndex, thirdMDIndex, fourthMDIndex, zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta, zLo, rtLo, rtHi, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ);
        }


        else if(innerInnerLowerModuleSubdet == SDL::Barrel
                and innerOuterLowerModuleSubdet == SDL::Barrel
                and outerInnerLowerModuleSubdet == SDL::Barrel
                and outerOuterLowerModuleSubdet == SDL::Endcap)
        {
            return runTripletDefaultAlgoBBBB(acc, modulesInGPU,mdsInGPU,segmentsInGPU,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,innerSegmentIndex,outerSegmentIndex,firstMDIndex, secondMDIndex, thirdMDIndex, fourthMDIndex, zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta,zLo, zHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut);

        }

        else if(innerInnerLowerModuleSubdet == SDL::Barrel
                and innerOuterLowerModuleSubdet == SDL::Endcap
                and outerInnerLowerModuleSubdet == SDL::Endcap
                and outerOuterLowerModuleSubdet == SDL::Endcap)
        {
            return runTripletDefaultAlgoBBEE(acc, modulesInGPU,mdsInGPU,segmentsInGPU,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,innerSegmentIndex,outerSegmentIndex, firstMDIndex, secondMDIndex, thirdMDIndex, fourthMDIndex, zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta, zLo, rtLo, rtHi, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ);

        }

        else if(innerInnerLowerModuleSubdet == SDL::Endcap
                and innerOuterLowerModuleSubdet == SDL::Endcap
                and outerInnerLowerModuleSubdet == SDL::Endcap
                and outerOuterLowerModuleSubdet == SDL::Endcap)
        {
            return runTripletDefaultAlgoEEEE(acc, modulesInGPU,mdsInGPU,segmentsInGPU,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,innerSegmentIndex,outerSegmentIndex, firstMDIndex, secondMDIndex, thirdMDIndex, fourthMDIndex, zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta, zLo, rtLo, rtHi, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ);
        }

        return pass;
    };

    template<typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runTripletConstraintsAndAlgo(TAcc const & acc, struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, uint16_t& innerInnerLowerModuleIndex, uint16_t& middleLowerModuleIndex, uint16_t& outerOuterLowerModuleIndex, unsigned int& innerSegmentIndex, unsigned int& outerSegmentIndex, float& zOut, float& rtOut, float& deltaPhiPos, float& deltaPhi, float& betaIn, float& betaOut, float& pt_beta, float &zLo, float& zHi, float& rtLo, float& rtHi, float& zLoPointed, float& zHiPointed, float& sdlCut, float& betaInCut, float& betaOutCut, float& deltaBetaCut, float& kZ)
    {
        bool pass = true;

        //this cut reduces the number of candidates by a factor of 4, i.e., 3 out of 4 warps can end right here!
        if(segmentsInGPU.mdIndices[2 * innerSegmentIndex+ 1] != segmentsInGPU.mdIndices[2 * outerSegmentIndex]) return false;

        unsigned int firstMDIndex = segmentsInGPU.mdIndices[2 *innerSegmentIndex];
        unsigned int secondMDIndex = segmentsInGPU.mdIndices[2 * outerSegmentIndex];
        unsigned int thirdMDIndex = segmentsInGPU.mdIndices[2 * outerSegmentIndex + 1];

        pass = pass and (passRZConstraint(acc, modulesInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, middleLowerModuleIndex, outerOuterLowerModuleIndex, firstMDIndex, secondMDIndex, thirdMDIndex));
        if(not pass) return pass;
        pass = pass and (passPointingConstraint(acc, modulesInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, middleLowerModuleIndex, outerOuterLowerModuleIndex, firstMDIndex, secondMDIndex, thirdMDIndex, zOut, rtOut));
        if(not pass) return pass;
        pass = pass and (runTripletDefaultAlgo(acc, modulesInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, middleLowerModuleIndex, middleLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, firstMDIndex, secondMDIndex, secondMDIndex, thirdMDIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, pt_beta, zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ));

        return pass;
    };

    struct createTripletsInGPUv2
    {
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(
                TAcc const & acc,
                struct SDL::modules& modulesInGPU,
                struct SDL::miniDoublets& mdsInGPU,
                struct SDL::segments& segmentsInGPU,
                struct SDL::triplets& tripletsInGPU,
                struct SDL::objectRanges& rangesInGPU,
                uint16_t *index_gpu,
                uint16_t nonZeroModules) const
        {
            using Dim = alpaka::Dim<TAcc>;
            using Idx = alpaka::Idx<TAcc>;
            using Vec = alpaka::Vec<Dim, Idx>;

            Vec const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
            Vec const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

            for(uint16_t innerLowerModuleArrayIdx = globalThreadIdx[0]; innerLowerModuleArrayIdx < nonZeroModules; innerLowerModuleArrayIdx += gridThreadExtent[0])
            {
                uint16_t innerInnerLowerModuleIndex = index_gpu[innerLowerModuleArrayIdx];
                if(innerInnerLowerModuleIndex >= *modulesInGPU.nLowerModules) continue;

                uint16_t nConnectedModules = modulesInGPU.nConnectedModules[innerInnerLowerModuleIndex];
                if(nConnectedModules == 0) continue;

                unsigned int nInnerSegments = segmentsInGPU.nSegments[innerInnerLowerModuleIndex];
                for(int innerSegmentArrayIndex = globalThreadIdx[1]; innerSegmentArrayIndex < nInnerSegments; innerSegmentArrayIndex += gridThreadExtent[1])
                {
                    unsigned int innerSegmentIndex = rangesInGPU.segmentRanges[innerInnerLowerModuleIndex * 2] + innerSegmentArrayIndex;

                    // middle lower module - outer lower module of inner segment
                    uint16_t middleLowerModuleIndex = segmentsInGPU.outerLowerModuleIndices[innerSegmentIndex];

                    unsigned int nOuterSegments = segmentsInGPU.nSegments[middleLowerModuleIndex];
                    for(int outerSegmentArrayIndex = globalThreadIdx[2]; outerSegmentArrayIndex < nOuterSegments; outerSegmentArrayIndex += gridThreadExtent[2])
                    {
                        unsigned int outerSegmentIndex = rangesInGPU.segmentRanges[2 * middleLowerModuleIndex] + outerSegmentArrayIndex;
                    
                        uint16_t outerOuterLowerModuleIndex = segmentsInGPU.outerLowerModuleIndices[outerSegmentIndex];

                        float zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut, pt_beta;
                        float zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ;

                        bool success = runTripletConstraintsAndAlgo(acc, modulesInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, middleLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, pt_beta, zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ);

                        if(success)
                        {
                            unsigned int totOccupancyTriplets = alpaka::atomicOp<alpaka::AtomicAdd>(acc, &tripletsInGPU.totOccupancyTriplets[innerInnerLowerModuleIndex], 1);
                            if(totOccupancyTriplets >= (rangesInGPU.tripletModuleOccupancy[innerInnerLowerModuleIndex]))
                            {
#ifdef Warnings
                                printf("Triplet excess alert! Module index = %d\n",innerInnerLowerModuleIndex);
#endif
                            }
                            else
                            {
                                unsigned int tripletModuleIndex = alpaka::atomicOp<alpaka::AtomicAdd>(acc, &tripletsInGPU.nTriplets[innerInnerLowerModuleIndex], 1);
                                unsigned int tripletIndex = rangesInGPU.tripletModuleIndices[innerInnerLowerModuleIndex] + tripletModuleIndex;
#ifdef CUT_VALUE_DEBUG
                                addTripletToMemory(modulesInGPU, mdsInGPU, segmentsInGPU, tripletsInGPU, innerSegmentIndex, outerSegmentIndex, innerInnerLowerModuleIndex, middleLowerModuleIndex, outerOuterLowerModuleIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut,pt_beta, zLo,zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ, tripletIndex);
#else
                                addTripletToMemory(modulesInGPU, mdsInGPU, segmentsInGPU, tripletsInGPU, innerSegmentIndex, outerSegmentIndex, innerInnerLowerModuleIndex, middleLowerModuleIndex, outerOuterLowerModuleIndex, betaIn, betaOut, pt_beta, tripletIndex);
#endif
                            }
                        }
                    }
                }
            }
        }
    };

    struct createTripletArrayRanges
    {
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(
                TAcc const & acc,
                struct modules& modulesInGPU,
                struct objectRanges& rangesInGPU,
                struct segments& segmentsInGPU) const
        {
            using Dim = alpaka::Dim<TAcc>;
            using Idx = alpaka::Idx<TAcc>;
            using Vec = alpaka::Vec<Dim, Idx>;

            Vec const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
            Vec const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

            // Initialize variables in shared memory and set to 0
            int& nTotalTriplets = alpaka::declareSharedVar<int, __COUNTER__>(acc);
            nTotalTriplets = 0;
            alpaka::syncBlockThreads(acc);

            // Initialize variables outside of the for loop.
            int occupancy, category_number, eta_number;

            for(uint16_t i = globalThreadIdx[2]; i < *modulesInGPU.nLowerModules; i+= gridThreadExtent[2])
            {
                if(segmentsInGPU.nSegments[i] == 0)
                {
                    rangesInGPU.tripletModuleIndices[i] = nTotalTriplets;
                    rangesInGPU.tripletModuleOccupancy[i] = 0;
                    continue;
                }

                short module_rings = modulesInGPU.rings[i];
                short module_layers = modulesInGPU.layers[i];
                short module_subdets = modulesInGPU.subdets[i];
                float module_eta = alpaka::math::abs(acc, modulesInGPU.eta[i]);

                if (module_layers<=3 && module_subdets==5) category_number = 0;
                else if (module_layers>=4 && module_subdets==5) category_number = 1;
                else if (module_layers<=2 && module_subdets==4 && module_rings>=11) category_number = 2;
                else if (module_layers>=3 && module_subdets==4 && module_rings>=8) category_number = 2;
                else if (module_layers<=2 && module_subdets==4 && module_rings<=10) category_number = 3;
                else if (module_layers>=3 && module_subdets==4 && module_rings<=7) category_number = 3;

                if (module_eta<0.75) eta_number = 0;
                else if (module_eta>0.75 && module_eta<1.5) eta_number = 1;
                else if (module_eta>1.5 && module_eta<2.25) eta_number = 2;
                else if (module_eta>2.25 && module_eta<3) eta_number = 3;

                if (category_number == 0 && eta_number == 0) occupancy = 543;
                else if (category_number == 0 && eta_number == 1) occupancy = 235;
                else if (category_number == 0 && eta_number == 2) occupancy = 88;
                else if (category_number == 0 && eta_number == 3) occupancy = 46;
                else if (category_number == 1 && eta_number == 0) occupancy = 755;
                else if (category_number == 1 && eta_number == 1) occupancy = 347;
                else if (category_number == 2 && eta_number == 1) occupancy = 0;
                else if (category_number == 2 && eta_number == 2) occupancy = 0;
                else if (category_number == 3 && eta_number == 1) occupancy = 38;
                else if (category_number == 3 && eta_number == 2) occupancy = 46;
                else if (category_number == 3 && eta_number == 3) occupancy = 39;

                rangesInGPU.tripletModuleOccupancy[i] = occupancy;
                unsigned int nTotT = alpaka::atomicOp<alpaka::AtomicAdd>(acc, &nTotalTriplets, occupancy);
                rangesInGPU.tripletModuleIndices[i] = nTotT;
            }

            // Wait for all threads to finish before reporting final values
            alpaka::syncBlockThreads(acc);
            if(globalThreadIdx[2] == 0)
            {
                *rangesInGPU.device_nTotalTrips = nTotalTriplets;
            }
        }
    };

    struct addTripletRangesToEventExplicit
    {
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(
                TAcc const & acc,
                struct modules& modulesInGPU,
                struct triplets& tripletsInGPU,
                struct objectRanges& rangesInGPU) const
        {
            using Dim = alpaka::Dim<TAcc>;
            using Idx = alpaka::Idx<TAcc>;
            using Vec = alpaka::Vec<Dim, Idx>;

            Vec const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
            Vec const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

            for(uint16_t i = globalThreadIdx[2]; i < *modulesInGPU.nLowerModules; i += gridThreadExtent[2])
            {
                if(tripletsInGPU.nTriplets[i] == 0)
                {
                    rangesInGPU.tripletRanges[i * 2] = -1;
                    rangesInGPU.tripletRanges[i * 2 + 1] = -1;
                }
                else
                {
                    rangesInGPU.tripletRanges[i * 2] = rangesInGPU.tripletModuleIndices[i];
                    rangesInGPU.tripletRanges[i * 2 + 1] = rangesInGPU.tripletModuleIndices[i] +  tripletsInGPU.nTriplets[i] - 1;
                }
            }
        }
    };
}
#endif
