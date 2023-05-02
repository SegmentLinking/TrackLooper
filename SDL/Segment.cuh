#ifndef Segment_cuh
#define Segment_cuh

#include "Constants.cuh"
#include "Constants.h"
#include "EndcapGeometry.cuh"
#include "TiltedGeometry.h"
#include "MiniDoublet.cuh"
#include "Module.cuh"
#include "Hit.cuh"

namespace SDL
{
    struct segments
    {
        unsigned int* nMemoryLocations;

        unsigned int* mdIndices;
        uint16_t* innerLowerModuleIndices;
        uint16_t* outerLowerModuleIndices;
        unsigned int* innerMiniDoubletAnchorHitIndices;
        unsigned int* outerMiniDoubletAnchorHitIndices;
        
        int* nSegments; //number of segments per inner lower module
        int* totOccupancySegments; //number of segments per inner lower module
        FPX* dPhis;
        FPX* dPhiMins;
        FPX* dPhiMaxs;
        FPX* dPhiChanges;
        FPX* dPhiChangeMins;
        FPX* dPhiChangeMaxs;

        //not so optional pixel dudes
        float* ptIn;
        float* ptErr;
        float* px;
        float* py;
        float* pz;
        float* etaErr;
        float* eta;
        float* phi;
        int* charge;
        unsigned int* seedIdx;
        int* superbin;
        int8_t* pixelType;
        char* isQuad;
        bool* isDup;
        float* score;
        float* circleCenterX;
        float* circleCenterY;
        float* circleRadius;
        bool* partOfPT5;
        uint4* pLSHitsIdxs;

        segments();
        ~segments();

	    void freeMemory(cudaStream_t stream);
	    void freeMemoryCache();
        void resetMemory(unsigned int nMemoryLocationsx, unsigned int nModules, unsigned int maxPixelSegments,cudaStream_t stream);
    };

    void createSegmentsInExplicitMemory(struct segments& segmentsInGPU, unsigned int maxSegments, uint16_t nLowerModules, unsigned int maxPixelSegments,cudaStream_t stream);

    void createSegmentArrayRanges(struct modules& modulesInGPU, struct objectRanges& rangesInGPU, struct miniDoublets& mdsinGPU, uint16_t& nLowerModules, int& nSegments, cudaStream_t stream, const uint16_t& maxPixelSegments);

    __global__ void addSegmentRangesToEventExplicit(struct modules& modulesInGPU, struct segments& segmentsInGPU, struct objectRanges& rangesInGPU);

    ALPAKA_FN_ACC ALPAKA_FN_INLINE float isTighterTiltedModules_seg(struct modules& modulesInGPU, unsigned int moduleIndex)
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
    };

    ALPAKA_FN_ACC ALPAKA_FN_INLINE float isTighterTiltedModules_seg(short subdet, short layer, short side, short rod)
    {
        // The "tighter" tilted modules are the subset of tilted modules that have smaller spacing
        // This is the same as what was previously considered as"isNormalTiltedModules"
        // See Figure 9.1 of https://cds.cern.ch/record/2272264/files/CMS-TDR-014.pdf
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
    };

    ALPAKA_FN_ACC ALPAKA_FN_INLINE float moduleGapSize_seg(short layer, short ring, short subdet, short side, short rod)
    {
        float miniDeltaTilted[3] = {0.26f, 0.26f, 0.26f};
        float miniDeltaFlat[6] ={0.26f, 0.16f, 0.16f, 0.18f, 0.18f, 0.18f};
        float miniDeltaLooseTilted[3] = {0.4f,0.4f,0.4f};
        float miniDeltaEndcap[5][15];

        for (size_t i = 0; i < 5; i++)
        {
            for (size_t j = 0; j < 15; j++)
            {
                if (i == 0 || i == 1)
                {
                    if (j < 10)
                    {
                        miniDeltaEndcap[i][j] = 0.4f;
                    }
                    else
                    {
                        miniDeltaEndcap[i][j] = 0.18f;
                    }
                }
                else if (i == 2 || i == 3)
                {
                    if (j < 8)
                    {
                        miniDeltaEndcap[i][j] = 0.4f;
                    }
                    else
                    {
                        miniDeltaEndcap[i][j]  = 0.18f;
                    }
                }
                else
                {
                    if (j < 9)
                    {
                        miniDeltaEndcap[i][j] = 0.4f;
                    }
                    else
                    {
                        miniDeltaEndcap[i][j] = 0.18f;
                    }
                }
            }
        }

        unsigned int iL = layer-1;
        unsigned int iR = ring - 1;

        float moduleSeparation = 0;

        if (subdet == Barrel and side == Center)
        {
            moduleSeparation = miniDeltaFlat[iL];
        }
        else if (isTighterTiltedModules_seg(subdet, layer, side, rod))
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
    };

    ALPAKA_FN_ACC ALPAKA_FN_INLINE float moduleGapSize_seg(struct modules& modulesInGPU, unsigned int moduleIndex)
    {
        float miniDeltaTilted[3] = {0.26f, 0.26f, 0.26f};
        float miniDeltaFlat[6] ={0.26f, 0.16f, 0.16f, 0.18f, 0.18f, 0.18f};
        float miniDeltaLooseTilted[3] = {0.4f,0.4f,0.4f};
        float miniDeltaEndcap[5][15];

        for (size_t i = 0; i < 5; i++)
        {
            for (size_t j = 0; j < 15; j++)
            {
                if (i == 0 || i == 1)
                {
                    if (j < 10)
                    {
                        miniDeltaEndcap[i][j] = 0.4f;
                    }
                    else
                    {
                        miniDeltaEndcap[i][j] = 0.18f;
                    }
                }
                else if (i == 2 || i == 3)
                {
                    if (j < 8)
                    {
                        miniDeltaEndcap[i][j] = 0.4f;
                    }
                    else
                    {
                        miniDeltaEndcap[i][j]  = 0.18f;
                    }
                }
                else
                {
                    if (j < 9)
                    {
                        miniDeltaEndcap[i][j] = 0.4f;
                    }
                    else
                    {
                        miniDeltaEndcap[i][j] = 0.18f;
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
        else if (isTighterTiltedModules_seg(modulesInGPU, moduleIndex))
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
    };

    template<typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void dAlphaThreshold(TAcc const & acc, float* dAlphaThresholdValues, struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, float& xIn, float& yIn, float& zIn, float& rtIn, float& xOut, float& yOut, float& zOut, float& rtOut, uint16_t& innerLowerModuleIndex, uint16_t& outerLowerModuleIndex, unsigned int& innerMDIndex, unsigned int& outerMDIndex)
    {
        float sdMuls = (modulesInGPU.subdets[innerLowerModuleIndex] == SDL::Barrel) ? miniMulsPtScaleBarrel[modulesInGPU.layers[innerLowerModuleIndex]-1] * 3.f/ptCut : miniMulsPtScaleEndcap[modulesInGPU.layers[innerLowerModuleIndex]-1] * 3.f/ptCut;

        //more accurate then outer rt - inner rt
        float segmentDr = alpaka::math::sqrt(acc, (yOut - yIn) * (yOut - yIn) + (xOut - xIn) * (xOut - xIn));

        const float dAlpha_Bfield = alpaka::math::asin(acc, alpaka::math::min(acc, segmentDr * k2Rinv1GeVf/ptCut, sinAlphaMax));

        bool isInnerTilted = modulesInGPU.subdets[innerLowerModuleIndex] == SDL::Barrel and modulesInGPU.sides[innerLowerModuleIndex] != SDL::Center;
        bool isOuterTilted = modulesInGPU.subdets[outerLowerModuleIndex] == SDL::Barrel and modulesInGPU.sides[outerLowerModuleIndex] != SDL::Center;

        float& drdzInner = modulesInGPU.drdzs[innerLowerModuleIndex];
        float& drdzOuter = modulesInGPU.drdzs[outerLowerModuleIndex];
        float innerModuleGapSize = SDL::moduleGapSize_seg(modulesInGPU, innerLowerModuleIndex);
        float outerModuleGapSize = SDL::moduleGapSize_seg(modulesInGPU, outerLowerModuleIndex);
        const float innerminiTilt = isInnerTilted ? (0.5f * pixelPSZpitch * drdzInner / alpaka::math::sqrt(acc, 1.f + drdzInner * drdzInner) / innerModuleGapSize) : 0;

        const float outerminiTilt = isOuterTilted ? (0.5f * pixelPSZpitch * drdzOuter / alpaka::math::sqrt(acc, 1.f + drdzOuter * drdzOuter) / outerModuleGapSize) : 0;

        float miniDelta = innerModuleGapSize; 

        float sdLumForInnerMini;    
        float sdLumForOuterMini;

        if (modulesInGPU.subdets[innerLowerModuleIndex] == SDL::Barrel)
        {
            sdLumForInnerMini = innerminiTilt * dAlpha_Bfield;
        }
        else
        {
            sdLumForInnerMini = mdsInGPU.dphis[innerMDIndex] * 15.0f / mdsInGPU.dzs[innerMDIndex];
        }

        if (modulesInGPU.subdets[outerLowerModuleIndex] == SDL::Barrel)
        {
            sdLumForOuterMini = outerminiTilt * dAlpha_Bfield;
        }
        else
        {
            sdLumForOuterMini = mdsInGPU.dphis[outerMDIndex] * 15.0f / mdsInGPU.dzs[outerMDIndex];
        }

        // Unique stuff for the segment dudes alone
        float dAlpha_res_inner = 0.02f/miniDelta * (modulesInGPU.subdets[innerLowerModuleIndex] == SDL::Barrel ? 1.0f : alpaka::math::abs(acc, zIn)/rtIn);
        float dAlpha_res_outer = 0.02f/miniDelta * (modulesInGPU.subdets[outerLowerModuleIndex] == SDL::Barrel ? 1.0f : alpaka::math::abs(acc, zOut)/rtOut);

        float dAlpha_res = dAlpha_res_inner + dAlpha_res_outer;

        if (modulesInGPU.subdets[innerLowerModuleIndex] == SDL::Barrel and modulesInGPU.sides[innerLowerModuleIndex] == SDL::Center)
        {
            dAlphaThresholdValues[0] = dAlpha_Bfield + alpaka::math::sqrt(acc, dAlpha_res * dAlpha_res + sdMuls * sdMuls);       
        }
        else
        {
            dAlphaThresholdValues[0] = dAlpha_Bfield + alpaka::math::sqrt(acc, dAlpha_res * dAlpha_res + sdMuls * sdMuls + sdLumForInnerMini * sdLumForInnerMini);    
        }

        if(modulesInGPU.subdets[outerLowerModuleIndex] == SDL::Barrel and modulesInGPU.sides[outerLowerModuleIndex] == SDL::Center)
        {
            dAlphaThresholdValues[1] = dAlpha_Bfield + alpaka::math::sqrt(acc, dAlpha_res * dAlpha_res + sdMuls * sdMuls);    
        }
        else
        {
            dAlphaThresholdValues[1] = dAlpha_Bfield + alpaka::math::sqrt(acc, dAlpha_res * dAlpha_res + sdMuls * sdMuls + sdLumForOuterMini * sdLumForOuterMini);
        }

        //Inner to outer 
        dAlphaThresholdValues[2] = dAlpha_Bfield + alpaka::math::sqrt(acc, dAlpha_res * dAlpha_res + sdMuls * sdMuls);
    };

    ALPAKA_FN_ACC ALPAKA_FN_INLINE void addSegmentToMemory(struct segments& segmentsInGPU, unsigned int lowerMDIndex, unsigned int upperMDIndex, uint16_t innerLowerModuleIndex, uint16_t outerLowerModuleIndex, unsigned int innerMDAnchorHitIndex, unsigned int outerMDAnchorHitIndex, float& dPhi, float& dPhiMin, float& dPhiMax, float& dPhiChange, float& dPhiChangeMin, float& dPhiChangeMax, unsigned int idx)
    {
        //idx will be computed in the kernel, which is the index into which the 
        //segment will be written
        //nSegments will be incremented in the kernel
        //printf("seg: %u %u %u %u\n",lowerMDIndex, upperMDIndex,innerLowerModuleIndex,outerLowerModuleIndex);
        segmentsInGPU.mdIndices[idx * 2] = lowerMDIndex;
        segmentsInGPU.mdIndices[idx * 2 + 1] = upperMDIndex;
        segmentsInGPU.innerLowerModuleIndices[idx] = innerLowerModuleIndex;
        segmentsInGPU.outerLowerModuleIndices[idx] = outerLowerModuleIndex;
        segmentsInGPU.innerMiniDoubletAnchorHitIndices[idx] = innerMDAnchorHitIndex;
        segmentsInGPU.outerMiniDoubletAnchorHitIndices[idx] = outerMDAnchorHitIndex;

        segmentsInGPU.dPhis[idx]          = __F2H(dPhi);
        segmentsInGPU.dPhiMins[idx]       = __F2H(dPhiMin);
        segmentsInGPU.dPhiMaxs[idx]       = __F2H(dPhiMax);
        segmentsInGPU.dPhiChanges[idx]    = __F2H(dPhiChange);
        segmentsInGPU.dPhiChangeMins[idx] = __F2H(dPhiChangeMin);
        segmentsInGPU.dPhiChangeMaxs[idx] = __F2H(dPhiChangeMax);
    }

    template<typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void addPixelSegmentToMemory(TAcc const & acc, struct segments& segmentsInGPU, struct miniDoublets& mdsInGPU, unsigned int innerMDIndex, unsigned int outerMDIndex, uint16_t pixelModuleIndex, unsigned int hitIdxs[4], unsigned int innerAnchorHitIndex, unsigned int outerAnchorHitIndex, float dPhiChange, unsigned int idx, unsigned int pixelSegmentArrayIndex, float score)
    {
        segmentsInGPU.mdIndices[idx * 2] = innerMDIndex;
        segmentsInGPU.mdIndices[idx * 2 + 1] = outerMDIndex;
        segmentsInGPU.innerLowerModuleIndices[idx] = pixelModuleIndex;
        segmentsInGPU.outerLowerModuleIndices[idx] = pixelModuleIndex;
        segmentsInGPU.innerMiniDoubletAnchorHitIndices[idx] = innerAnchorHitIndex;
        segmentsInGPU.outerMiniDoubletAnchorHitIndices[idx] = outerAnchorHitIndex;
        segmentsInGPU.dPhiChanges[idx] = __F2H(dPhiChange);
        segmentsInGPU.isDup[pixelSegmentArrayIndex] = false;
        segmentsInGPU.score[pixelSegmentArrayIndex] = score;

        segmentsInGPU.pLSHitsIdxs[pixelSegmentArrayIndex].x = hitIdxs[0];
        segmentsInGPU.pLSHitsIdxs[pixelSegmentArrayIndex].y = hitIdxs[1];
        segmentsInGPU.pLSHitsIdxs[pixelSegmentArrayIndex].z = hitIdxs[2];
        segmentsInGPU.pLSHitsIdxs[pixelSegmentArrayIndex].w = hitIdxs[3];

        //computing circle parameters
        /*
        The two anchor hits are r3PCA and r3LH. p3PCA pt, eta, phi is hitIndex1 x, y, z
        */
        float circleRadius = mdsInGPU.outerX[innerMDIndex] / (2 * k2Rinv1GeVf);
        float circlePhi = mdsInGPU.outerZ[innerMDIndex];
        float candidateCenterXs[] = {mdsInGPU.anchorX[innerMDIndex] + circleRadius * alpaka::math::sin(acc, circlePhi), mdsInGPU.anchorX[innerMDIndex] - circleRadius * alpaka::math::sin(acc, circlePhi)};
        float candidateCenterYs[] = {mdsInGPU.anchorY[innerMDIndex] - circleRadius * alpaka::math::cos(acc, circlePhi), mdsInGPU.anchorY[innerMDIndex] + circleRadius * alpaka::math::cos(acc, circlePhi)};

        //check which of the circles can accommodate r3LH better (we won't get perfect agreement)
        float bestChiSquared = SDL::SDL_INF;
        float chiSquared;
        size_t bestIndex;
        for(size_t i = 0; i < 2; i++)
        {
            chiSquared = alpaka::math::abs(acc, alpaka::math::sqrt(acc, (mdsInGPU.anchorX[outerMDIndex] - candidateCenterXs[i]) * (mdsInGPU.anchorX[outerMDIndex] - candidateCenterXs[i]) + (mdsInGPU.anchorY[outerMDIndex] - candidateCenterYs[i]) * (mdsInGPU.anchorY[outerMDIndex] - candidateCenterYs[i])) - circleRadius);
            if(chiSquared < bestChiSquared)
            {
                bestChiSquared = chiSquared;
                bestIndex = i;
            }
        }
        segmentsInGPU.circleCenterX[pixelSegmentArrayIndex] = candidateCenterXs[bestIndex];
        segmentsInGPU.circleCenterY[pixelSegmentArrayIndex] = candidateCenterYs[bestIndex];
        segmentsInGPU.circleRadius[pixelSegmentArrayIndex] = circleRadius;
    };

    template<typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runSegmentDefaultAlgoBarrel(TAcc const & acc, struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, uint16_t& innerLowerModuleIndex, uint16_t& outerLowerModuleIndex, unsigned int& innerMDIndex, unsigned int& outerMDIndex, float& zIn, float& zOut, float& rtIn, float& rtOut, float& dPhi, float& dPhiMin, float& dPhiMax, float& dPhiChange, float& dPhiChangeMin, float& dPhiChangeMax, float& dAlphaInnerMDSegment, float& dAlphaOuterMDSegment, float&dAlphaInnerMDOuterMD, float& zLo, float& zHi, float& sdCut, float& dAlphaInnerMDSegmentThreshold, float& dAlphaOuterMDSegmentThreshold, float& dAlphaInnerMDOuterMDThreshold)
    {
        bool pass = true;
    
        float sdMuls = (modulesInGPU.subdets[innerLowerModuleIndex] == SDL::Barrel) ? miniMulsPtScaleBarrel[modulesInGPU.layers[innerLowerModuleIndex]-1] * 3.f/ptCut : miniMulsPtScaleEndcap[modulesInGPU.layers[innerLowerModuleIndex]-1] * 3.f/ptCut;

        float xIn, yIn, xOut, yOut;

        xIn = mdsInGPU.anchorX[innerMDIndex];
        yIn = mdsInGPU.anchorY[innerMDIndex];
        zIn = mdsInGPU.anchorZ[innerMDIndex];
        rtIn = mdsInGPU.anchorRt[innerMDIndex];

        xOut = mdsInGPU.anchorX[outerMDIndex];
        yOut = mdsInGPU.anchorY[outerMDIndex];
        zOut = mdsInGPU.anchorZ[outerMDIndex];
        rtOut = mdsInGPU.anchorRt[outerMDIndex];

        float sdSlope = alpaka::math::asin(acc, alpaka::math::min(acc, rtOut * k2Rinv1GeVf / ptCut, sinAlphaMax));
        float sdPVoff = 0.1f/rtOut;
        float dzDrtScale = alpaka::math::tan(acc, sdSlope)/sdSlope; //FIXME: need appropriate value

        const float zGeom = modulesInGPU.layers[innerLowerModuleIndex] <= 2 ? 2.f * pixelPSZpitch : 2.f * strip2SZpitch;

        zLo = zIn + (zIn - deltaZLum) * (rtOut / rtIn - 1.f) * (zIn > 0.f ? 1.f : dzDrtScale) - zGeom; //slope-correction only on outer end
        zHi = zIn + (zIn + deltaZLum) * (rtOut / rtIn - 1.f) * (zIn < 0.f ? 1.f : dzDrtScale) + zGeom;

        pass =  pass and ((zOut >= zLo) & (zOut <= zHi));
        if(not pass) return pass;

        sdCut = sdSlope + alpaka::math::sqrt(acc, sdMuls * sdMuls + sdPVoff * sdPVoff);

        dPhi  = deltaPhi_alpaka(acc, xIn, yIn, xOut, yOut);

        pass =  pass and (alpaka::math::abs(acc, dPhi) <= sdCut);
        if(not pass) return pass;

        dPhiChange = deltaPhiChange_alpaka(acc, xIn, yIn, xOut, yOut);

        pass =  pass and (alpaka::math::abs(acc, dPhiChange) <= sdCut);
        if(not pass) return pass;

        float dAlphaThresholdValues[3];
        dAlphaThreshold(acc, dAlphaThresholdValues, modulesInGPU, mdsInGPU, xIn, yIn, zIn, rtIn, xOut, yOut, zOut, rtOut, innerLowerModuleIndex, outerLowerModuleIndex, innerMDIndex, outerMDIndex);

        float innerMDAlpha = mdsInGPU.dphichanges[innerMDIndex];
        float outerMDAlpha = mdsInGPU.dphichanges[outerMDIndex];
        dAlphaInnerMDSegment = innerMDAlpha - dPhiChange;
        dAlphaOuterMDSegment = outerMDAlpha - dPhiChange;
        dAlphaInnerMDOuterMD = innerMDAlpha - outerMDAlpha;

        dAlphaInnerMDSegmentThreshold = dAlphaThresholdValues[0];
        dAlphaOuterMDSegmentThreshold = dAlphaThresholdValues[1];
        dAlphaInnerMDOuterMDThreshold = dAlphaThresholdValues[2];
        
        pass =  pass and (alpaka::math::abs(acc, dAlphaInnerMDSegment) < dAlphaThresholdValues[0]);
        if(not pass) return pass;
        pass =  pass and (alpaka::math::abs(acc, dAlphaOuterMDSegment) < dAlphaThresholdValues[1]);
        if(not pass) return pass;
        pass =  pass and (alpaka::math::abs(acc, dAlphaInnerMDOuterMD) < dAlphaThresholdValues[2]);

        return pass;
    };

    template<typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runSegmentDefaultAlgoEndcap(TAcc const & acc, struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, uint16_t& innerLowerModuleIndex, uint16_t& outerLowerModuleIndex, unsigned int& innerMDIndex, unsigned int& outerMDIndex, float& zIn, float& zOut, float& rtIn, float& rtOut, float& dPhi, float& dPhiMin, float& dPhiMax, float& dPhiChange, float& dPhiChangeMin, float& dPhiChangeMax, float& dAlphaInnerMDSegment, float& dAlphaOuterMDSegment, float& rtLo, float& rtHi, float& sdCut, float& dAlphaInnerMDSegmentThreshold, float& dAlphaOuterMDSegmentThreshold, float& dAlphaInnerMDOuterMDThreshold, float&dAlphaInnerMDOuterMD)
    {
        bool pass = true;
    
        float xIn, yIn;    
        float xOut, yOut, xOutHigh, yOutHigh, xOutLow, yOutLow;

        xIn = mdsInGPU.anchorX[innerMDIndex];
        yIn = mdsInGPU.anchorY[innerMDIndex];
        zIn = mdsInGPU.anchorZ[innerMDIndex];
        rtIn = mdsInGPU.anchorRt[innerMDIndex];

        xOut = mdsInGPU.anchorX[outerMDIndex];
        yOut = mdsInGPU.anchorY[outerMDIndex];
        zOut = mdsInGPU.anchorZ[outerMDIndex];
        rtOut = mdsInGPU.anchorRt[outerMDIndex];

        xOutHigh = mdsInGPU.anchorHighEdgeX[outerMDIndex];
        yOutHigh = mdsInGPU.anchorHighEdgeY[outerMDIndex];
        xOutLow = mdsInGPU.anchorLowEdgeX[outerMDIndex];
        yOutLow = mdsInGPU.anchorLowEdgeY[outerMDIndex];
        bool outerLayerEndcapTwoS = (modulesInGPU.subdets[outerLowerModuleIndex] == SDL::Endcap) & (modulesInGPU.moduleType[outerLowerModuleIndex] == SDL::TwoS);

        float sdSlope = alpaka::math::asin(acc, alpaka::math::min(acc, rtOut * k2Rinv1GeVf / ptCut, sinAlphaMax));
        float sdPVoff = 0.1/rtOut;
        float disks2SMinRadius = 60.f;

        float rtGeom =  ((rtIn < disks2SMinRadius && rtOut < disks2SMinRadius) ? (2.f * pixelPSZpitch)
                : ((rtIn < disks2SMinRadius || rtOut < disks2SMinRadius) ? (pixelPSZpitch + strip2SZpitch)
                : (2.f * strip2SZpitch)));

        //cut 0 - z compatibility
        pass =  pass and (zIn * zOut >= 0);
        if(not pass) return pass;

        float dz = zOut - zIn;
        // Alpaka: Needs to be moved over
        float dLum = copysignf(deltaZLum, zIn);
        float drtDzScale = sdSlope/alpaka::math::tan(acc, sdSlope);

        rtLo = alpaka::math::max(acc, rtIn * (1.f + dz / (zIn + dLum) * drtDzScale) - rtGeom,  rtIn - 0.5f * rtGeom); //rt should increase
        rtHi = rtIn * (zOut - dLum) / (zIn - dLum) + rtGeom; //dLum for luminous; rGeom for measurement size; no tanTheta_loc(pt) correction

        // Completeness
        pass =  pass and ((rtOut >= rtLo) & (rtOut <= rtHi));
        if(not pass) return pass;

        dPhi = deltaPhi_alpaka(acc, xIn, yIn, xOut, yOut);

        sdCut = sdSlope;
        if(outerLayerEndcapTwoS)
        {
            float dPhiPos_high = deltaPhi_alpaka(acc, xIn, yIn, xOutHigh, yOutHigh);
            float dPhiPos_low = deltaPhi_alpaka(acc, xIn, yIn, xOutLow, yOutLow);
            
            dPhiMax = alpaka::math::abs(acc, dPhiPos_high) > alpaka::math::abs(acc, dPhiPos_low) ? dPhiPos_high : dPhiPos_low;
            dPhiMin = alpaka::math::abs(acc, dPhiPos_high) > alpaka::math::abs(acc, dPhiPos_low) ? dPhiPos_low : dPhiPos_high;
        }
        else
        {
            dPhiMax = dPhi;
            dPhiMin = dPhi;
        }
        pass =  pass and (alpaka::math::abs(acc, dPhi) <= sdCut);
        if(not pass) return pass;

        float dzFrac = dz/zIn;
        dPhiChange = dPhi/dzFrac * (1.f + dzFrac);
        dPhiChangeMin = dPhiMin/dzFrac * (1.f + dzFrac);
        dPhiChangeMax = dPhiMax/dzFrac * (1.f + dzFrac);
        
        pass =  pass and (alpaka::math::abs(acc, dPhiChange) <= sdCut);
        if(not pass) return pass;

        float dAlphaThresholdValues[3];
        dAlphaThreshold(acc, dAlphaThresholdValues, modulesInGPU, mdsInGPU, xIn, yIn, zIn, rtIn, xOut, yOut, zOut, rtOut,innerLowerModuleIndex, outerLowerModuleIndex, innerMDIndex, outerMDIndex);

        dAlphaInnerMDSegmentThreshold = dAlphaThresholdValues[0];
        dAlphaOuterMDSegmentThreshold = dAlphaThresholdValues[1];
        dAlphaInnerMDOuterMDThreshold = dAlphaThresholdValues[2];

        float innerMDAlpha = mdsInGPU.dphichanges[innerMDIndex];
        float outerMDAlpha = mdsInGPU.dphichanges[outerMDIndex];
        dAlphaInnerMDSegment = innerMDAlpha - dPhiChange;
        dAlphaOuterMDSegment = outerMDAlpha - dPhiChange;
        dAlphaInnerMDOuterMD = innerMDAlpha - outerMDAlpha;
    
        pass =  pass and (alpaka::math::abs(acc, dAlphaInnerMDSegment) < dAlphaThresholdValues[0]);
        if(not pass) return pass;
        pass =  pass and (alpaka::math::abs(acc, dAlphaOuterMDSegment) < dAlphaThresholdValues[1]);
        if(not pass) return pass;
        pass =  pass and (alpaka::math::abs(acc, dAlphaInnerMDOuterMD) < dAlphaThresholdValues[2]);

        return pass;
    };

    template<typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runSegmentDefaultAlgo(TAcc const & acc, struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, uint16_t& innerLowerModuleIndex, uint16_t& outerLowerModuleIndex, unsigned int& innerMDIndex, unsigned int& outerMDIndex, float& zIn, float& zOut, float& rtIn, float& rtOut, float& dPhi, float& dPhiMin, float& dPhiMax, float& dPhiChange, float& dPhiChangeMin, float& dPhiChangeMax, float& dAlphaInnerMDSegment, float& dAlphaOuterMDSegment, float&dAlphaInnerMDOuterMD, float& zLo, float& zHi, float& rtLo, float& rtHi, float& sdCut, float& dAlphaInnerMDSegmentThreshold, float& dAlphaOuterMDSegmentThreshold, float& dAlphaInnerMDOuterMDThreshold)
    {
        zLo = -999.f;
        zHi = -999.f;
        rtLo = -999.f;
        rtHi = -999.f;

        if(modulesInGPU.subdets[innerLowerModuleIndex] == SDL::Barrel and modulesInGPU.subdets[outerLowerModuleIndex] == SDL::Barrel)
        {
            return runSegmentDefaultAlgoBarrel(acc, modulesInGPU, mdsInGPU, innerLowerModuleIndex, outerLowerModuleIndex, innerMDIndex, outerMDIndex, zIn, zOut, rtIn, rtOut, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD, zLo, zHi, sdCut, dAlphaInnerMDSegmentThreshold, dAlphaOuterMDSegmentThreshold, dAlphaInnerMDOuterMDThreshold);
        }
        else
        {
            return runSegmentDefaultAlgoEndcap(acc, modulesInGPU, mdsInGPU, innerLowerModuleIndex, outerLowerModuleIndex, innerMDIndex, outerMDIndex, zIn, zOut, rtIn, rtOut, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD, rtLo, rtHi, sdCut, dAlphaInnerMDSegmentThreshold, dAlphaOuterMDSegmentThreshold, dAlphaInnerMDOuterMDThreshold);
        }
    };

    void printSegment(struct segments& segmentsInGPU, struct miniDoublets& mdsInGPU, struct hits& hitsInGPU, struct modules& modulesInGPU, unsigned int segmentIndex);

    struct createSegmentsInGPUv2
    {
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(
                TAcc const & acc,
                struct SDL::modules& modulesInGPU,
                struct SDL::miniDoublets& mdsInGPU,
                struct SDL::segments& segmentsInGPU,
                struct SDL::objectRanges& rangesInGPU) const
        {
            using Dim = alpaka::Dim<TAcc>;
            using Idx = alpaka::Idx<TAcc>;
            using Vec = alpaka::Vec<Dim, Idx>;

            Vec const globalBlockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc);
            Vec const blockThreadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
            Vec const gridBlockExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc);
            Vec const blockThreadExtent = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);

            for(uint16_t innerLowerModuleIndex = globalBlockIdx[2]; innerLowerModuleIndex < (*modulesInGPU.nLowerModules); innerLowerModuleIndex += gridBlockExtent[2])
            {
                unsigned int nConnectedModules = modulesInGPU.nConnectedModules[innerLowerModuleIndex];

                for(uint16_t outerLowerModuleArrayIdx = blockThreadIdx[1]; outerLowerModuleArrayIdx< nConnectedModules; outerLowerModuleArrayIdx+= blockThreadExtent[1])
                {
                    uint16_t outerLowerModuleIndex = modulesInGPU.moduleMap[innerLowerModuleIndex * MAX_CONNECTED_MODULES + outerLowerModuleArrayIdx];

                    unsigned int nInnerMDs = mdsInGPU.nMDs[innerLowerModuleIndex];
                    unsigned int nOuterMDs = mdsInGPU.nMDs[outerLowerModuleIndex];

                    int limit = nInnerMDs*nOuterMDs;

                    if (limit == 0) continue;
                    for(int hitIndex = blockThreadIdx[2]; hitIndex < limit; hitIndex += blockThreadExtent[2])
                    {
                        int innerMDArrayIdx = hitIndex / nOuterMDs;
                        int outerMDArrayIdx = hitIndex % nOuterMDs;
                        if(outerMDArrayIdx >= nOuterMDs) continue;

                        unsigned int innerMDIndex = rangesInGPU.mdRanges[innerLowerModuleIndex * 2] + innerMDArrayIdx;
                        unsigned int outerMDIndex = rangesInGPU.mdRanges[outerLowerModuleIndex * 2] + outerMDArrayIdx;

                        float zIn, zOut, rtIn, rtOut, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD;

                        unsigned int innerMiniDoubletAnchorHitIndex = mdsInGPU.anchorHitIndices[innerMDIndex];
                        unsigned int outerMiniDoubletAnchorHitIndex = mdsInGPU.anchorHitIndices[outerMDIndex];
                        dPhiMin = 0;
                        dPhiMax = 0;
                        dPhiChangeMin = 0;
                        dPhiChangeMax = 0;
                        float zLo, zHi, rtLo, rtHi, sdCut , dAlphaInnerMDSegmentThreshold, dAlphaOuterMDSegmentThreshold, dAlphaInnerMDOuterMDThreshold;
                        bool pass = runSegmentDefaultAlgo(acc, modulesInGPU, mdsInGPU, innerLowerModuleIndex, outerLowerModuleIndex, innerMDIndex, outerMDIndex, zIn, zOut, rtIn, rtOut, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD, zLo, zHi, rtLo, rtHi, sdCut, dAlphaInnerMDSegmentThreshold, dAlphaOuterMDSegmentThreshold, dAlphaInnerMDOuterMDThreshold);

                        if(pass)
                        {
                            unsigned int totOccupancySegments = alpaka::atomicOp<alpaka::AtomicAdd>(acc, &segmentsInGPU.totOccupancySegments[innerLowerModuleIndex], 1);
                            if(totOccupancySegments >= (rangesInGPU.segmentModuleIndices[innerLowerModuleIndex + 1] - rangesInGPU.segmentModuleIndices[innerLowerModuleIndex]))
                            {
#ifdef Warnings
                                printf("Segment excess alert! Module index = %d\n",innerLowerModuleIndex);
#endif
                            }
                            else
                            {
                                unsigned int segmentModuleIdx = alpaka::atomicOp<alpaka::AtomicAdd>(acc, &segmentsInGPU.nSegments[innerLowerModuleIndex], 1);
                                unsigned int segmentIdx = rangesInGPU.segmentModuleIndices[innerLowerModuleIndex] + segmentModuleIdx;

                                addSegmentToMemory(segmentsInGPU,innerMDIndex, outerMDIndex,innerLowerModuleIndex, outerLowerModuleIndex, innerMiniDoubletAnchorHitIndex, outerMiniDoubletAnchorHitIndex, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, segmentIdx);
                            }
                        }
                    }
                }
            }
        }
    };
}

#endif
