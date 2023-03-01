#ifndef MiniDoublet_cuh
#define MiniDoublet_cuh

#include <array>
#include <tuple>
#include <cmath>
#include "Constants.cuh"
#include "EndcapGeometry.cuh"
#include "TiltedGeometry.h"
#include "Module.cuh"
#include "Hit.cuh"
#include "PrintUtil.h"

//CUDA MATH API
#include "math.h"

namespace SDL
{
    struct miniDoublets
    {
        unsigned int* nMemoryLocations;

        unsigned int* anchorHitIndices;
        unsigned int* outerHitIndices;
        uint16_t* moduleIndices;
        int* nMDs; //counter per module
        int* totOccupancyMDs; //counter per module
        float* dphichanges;

        float* dzs; //will store drt if the module is endcap
        float* dphis;

        float* shiftedXs;
        float* shiftedYs;
        float* shiftedZs;
        float* noShiftedDzs; //if shifted module
        float* noShiftedDphis; //if shifted module
        float* noShiftedDphiChanges; //if shifted module

        //hit stuff
        float* anchorX;
        float* anchorY;
        float* anchorZ;
        float* anchorRt;
        float* anchorPhi;
        float* anchorEta;
        float* anchorHighEdgeX;
        float* anchorHighEdgeY;
        float* anchorLowEdgeX;
        float* anchorLowEdgeY;
        float* anchorLowEdgePhi;
        float* anchorHighEdgePhi;
        float* anchorHighEdgePhi;
        float* anchorLowEdgePhi;

        float* outerX;
        float* outerY;
        float* outerZ;
        float* outerRt;
        float* outerPhi;
        float* outerEta;
        float* outerHighEdgeX;
        float* outerHighEdgeY;
        float* outerLowEdgeX;
        float* outerLowEdgeY;

#ifdef CUT_VALUE_DEBUG
        //CUT VALUES
        float* dzCuts;
        float* drtCuts;
        float* drts;
        float* miniCuts;
#endif

        miniDoublets();
        ~miniDoublets();
      	void freeMemory(cudaStream_t stream);
      	void freeMemoryCache();
        void resetMemory(unsigned int nMemoryLocations, unsigned int nModules,cudaStream_t stream);

    };


    void createMDsInExplicitMemory(struct miniDoublets& mdsInGPU, unsigned int maxMDs,uint16_t nLowerModules, unsigned int maxPixelMDs,cudaStream_t stream);


    __global__ void createMDArrayRangesGPU(struct modules& modulesInGPU, struct objectRanges& rangesInGPU);//, unsigned int* nTotalMDs);

    __global__ void addMiniDoubletRangesToEventExplicit(struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, struct objectRanges& rangesInGPU, struct hits& hitsInGPU);

//#ifdef CUT_VALUE_DEBUG
//    ALPAKA_FN_ACC void addMDToMemory(struct miniDoublets& mdsInGPU, struct hits& hitsInGPU, struct modules& modulesInGPU, unsigned int lowerHitIdx, unsigned int upperHitIdx, uint16_t& lowerModuleIdx, float dz, float drt, float dPhi, float dPhiChange, float shiftedX, float shiftedY, float shiftedZ, float noShiftedDz, float noShiftedDphi, float noShiftedDPhiChange, float dzCut, float drtCut, float miniCut, unsigned int idx);
//#else
    //for successful MDs
    ALPAKA_FN_HOST_ACC void addMDToMemory(struct miniDoublets& mdsInGPU, struct hits& hitsInGPU, struct modules& modulesInGPU, unsigned int lowerHitIdx, unsigned int upperHitIdx, uint16_t& lowerModuleIdx, float dz, float dPhi, float dPhiChange, float shiftedX, float shiftedY, float shiftedZ, float noShiftedDz, float noShiftedDphi, float noShiftedDPhiChange, unsigned int idx);
//#endif

    //ALPAKA_FN_ACC float dPhiThreshold(struct hits& hitsInGPU, struct modules& modulesInGPU, unsigned int hitIndex, unsigned int moduleIndex, float dPhi = 0, float dz = 0);
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float isTighterTiltedModules(struct modules& modulesInGPU, uint16_t& moduleIndex)
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
    ALPAKA_FN_ACC void initModuleGapSize();

    ALPAKA_FN_ACC ALPAKA_FN_INLINE float moduleGapSize(struct modules& modulesInGPU, uint16_t& moduleIndex)
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

    template<typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float dPhiThreshold(TAcc const & acc, /*struct hits& hitsInGPU,*/float rt, struct modules& modulesInGPU, /*unsigned int hitIndex,*/ uint16_t& moduleIndex, float dPhi = 0, float dz = 0)
    {
        // =================================================================
        // Various constants
        // =================================================================
        //mean of the horizontal layer position in y; treat this as R below

        // =================================================================
        // Computing some components that make up the cut threshold
        // =================================================================

        unsigned int iL = modulesInGPU.layers[moduleIndex] - 1;
        // Alpaka: Needs to be moved over.
        const float miniSlope = asinf(alpaka::math::min(acc, rt * k2Rinv1GeVf / ptCut, sinAlphaMax));
        const float rLayNominal = ((modulesInGPU.subdets[moduleIndex]== Barrel) ? miniRminMeanBarrel[iL] : miniRminMeanEndcap[iL]);
        const float miniPVoff = 0.1f / rLayNominal;
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
                drdz = modulesInGPU.drdzs[modulesInGPU.partnerModuleIndices[moduleIndex]];
            }
        }
        else
        {
            drdz = 0;
        }
        const float miniTilt = ((isTilted) ? 0.5f * pixelPSZpitch * drdz / alpaka::math::sqrt(acc, 1.f + drdz * drdz) / moduleGapSize(modulesInGPU,moduleIndex) : 0);

        // Compute luminous region requirement for endcap
        const float miniLum = alpaka::math::abs(acc, dPhi * deltaZLum/dz); // Balaji's new error
        // const float miniLum = abs(deltaZLum / lowerHit.z()); // Old error


        // =================================================================
        // Return the threshold value
        // =================================================================
        // Following condition is met if the module is central and flatly lying
        if (modulesInGPU.subdets[moduleIndex] == Barrel and modulesInGPU.sides[moduleIndex] == Center)
        {
            return miniSlope + alpaka::math::sqrt(acc, miniMuls * miniMuls + miniPVoff * miniPVoff);
        }
        // Following condition is met if the module is central and tilted
        else if (modulesInGPU.subdets[moduleIndex] == Barrel and modulesInGPU.sides[moduleIndex] != Center) //all types of tilted modules
        {
            return miniSlope + alpaka::math::sqrt(acc, miniMuls * miniMuls + miniPVoff * miniPVoff + miniTilt * miniTilt * miniSlope * miniSlope);
        }
        // If not barrel, it is Endcap
        else
        {
            return miniSlope + alpaka::math::sqrt(acc, miniMuls * miniMuls + miniPVoff * miniPVoff + miniLum * miniLum);
        }
    }

    ALPAKA_FN_ACC void shiftStripHits(struct modules& modulesInGPU, uint16_t& lowerModuleIndex, uint16_t& upperModuleIndex, unsigned int lowerHitIndex, unsigned int upperHitIndex, float* shiftedCoords,float xLower,float yLower,float zLower,float rtLower,float xUpper,float yUpper,float zUpper,float rtUpper);

    template<typename TAcc>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float ATan2_alpaka(TAcc const & acc, float y, float x)
    {
        // Alpaka: Needs to be moved over.
        if (x != 0) return atan2f(y, x);
        if (y == 0) return  0;
        if (y >  0) return  float(M_PI) / 2.f;
        else        return -float(M_PI) / 2.f;
    }

    template<typename TAcc>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float phi_mpi_pi_alpaka(TAcc const & acc, float x)
    {
        // Alpaka: Needs to be moved over.
        if (std::isnan(x))
        {
            return x;
        }

        if (alpaka::math::abs(acc, x) <= float(M_PI))
            return x;

        constexpr float o2pi = 1.f / (2.f * float(M_PI));
        float n = alpaka::math::round(acc, x * o2pi);
        return x - n * float(2.f * float(M_PI));
    }

    template<typename TAcc>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float phi_alpaka(TAcc const & acc, float x, float y) {
        return phi_mpi_pi_alpaka(acc, float(M_PI) + ATan2_alpaka(acc, -y, -x));
    }

    template<typename TAcc>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float deltaPhi_alpaka(TAcc const & acc, float x1, float y1, float x2, float y2) {
        float phi1 = phi_alpaka(acc, x1,y1);
        float phi2 = phi_alpaka(acc, x2,y2);
        return phi_mpi_pi_alpaka(acc, (phi2 - phi1));
    }

    template<typename TAcc>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float deltaPhiChange_alpaka(TAcc const & acc, float x1, float y1, float x2, float y2) {
        return deltaPhi_alpaka(acc, x1, y1, x2-x1, y2-y1);
    }

    template<typename TAcc>
    ALPAKA_FN_ACC bool runMiniDoubletDefaultAlgo(TAcc const & acc, struct modules& modulesInGPU, /*struct hits& hitsInGPU,*/ uint16_t& lowerModuleIndex, uint16_t& upperModuleIndex, unsigned int lowerHitIndex, unsigned int upperHitIndex, float& dz, float& dPhi, float& dPhiChange, float& shiftedX, float& shiftedY, float& shiftedZ, float& noShiftedDz, float& noShiftedDphi, float& noShiftedDphiChange, float xLower, float yLower, float zLower, float rtLower,float xUpper,float yUpper,float zUpper,float rtUpper)
    {
        if(modulesInGPU.subdets[lowerModuleIndex] == SDL::Barrel)
        {
            return runMiniDoubletDefaultAlgoBarrel(acc, modulesInGPU, /*hitsInGPU,*/ lowerModuleIndex, upperModuleIndex, lowerHitIndex, upperHitIndex, dz, dPhi, dPhiChange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange,xLower,yLower,zLower,rtLower, xUpper,yUpper,zUpper,rtUpper);
        }
        else
        {
            return runMiniDoubletDefaultAlgoEndcap(acc, modulesInGPU, /*hitsInGPU,*/ lowerModuleIndex, upperModuleIndex, lowerHitIndex, upperHitIndex, dz, dPhi, dPhiChange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange,xLower,yLower,zLower,rtLower, xUpper,yUpper,zUpper,rtUpper);
        }
    }

    template<typename TAcc>
    ALPAKA_FN_ACC bool runMiniDoubletDefaultAlgoBarrel(TAcc const & acc, struct modules& modulesInGPU, /*struct hits& hitsInGPU,*/ uint16_t& lowerModuleIndex, uint16_t& upperModuleIndex, unsigned int lowerHitIndex, unsigned int upperHitIndex, float& dz, float& dPhi, float& dPhiChange, float& shiftedX, float& shiftedY, float& shiftedZ, float& noshiftedDz, float& noShiftedDphi, float& noShiftedDphiChange, float xLower,float yLower, float zLower, float rtLower,float xUpper,float yUpper,float zUpper,float rtUpper)
    {

        bool pass = true; 
        dz = zLower - zUpper;     
        const float dzCut = modulesInGPU.moduleType[lowerModuleIndex] == SDL::PS ? 2.f : 10.f;
        //const float sign = ((dz > 0) - (dz < 0)) * ((hitsInGPU.zs[lowerHitIndex] > 0) - (hitsInGPU.zs[lowerHitIndex] < 0));
        const float sign = ((dz > 0) - (dz < 0)) * ((zLower > 0) - (zLower < 0));
        const float invertedcrossercut = (alpaka::math::abs(acc, dz) > 2) * sign;

        pass = pass  and ((alpaka::math::abs(acc, dz) < dzCut) & (invertedcrossercut <= 0));
        if(not pass) return pass;

        float miniCut = 0;

        miniCut = modulesInGPU.moduleLayerType[lowerModuleIndex] == SDL::Pixel ?  dPhiThreshold(acc, rtLower, modulesInGPU, lowerModuleIndex) : dPhiThreshold(acc, rtUpper, modulesInGPU, lowerModuleIndex); 

        // Cut #2: dphi difference
        // Ref to original code: https://github.com/slava77/cms-tkph2-ntuple/blob/184d2325147e6930030d3d1f780136bc2dd29ce6/doubletAnalysis.C#L3085
        float xn = 0.f, yn = 0.f;// , zn = 0;
        float shiftedRt;
        if (modulesInGPU.sides[lowerModuleIndex] != Center) // If barrel and not center it is tilted
        {
            // Shift the hits and calculate new xn, yn position
            float shiftedCoords[3];
            shiftStripHits(modulesInGPU, lowerModuleIndex, upperModuleIndex, lowerHitIndex, upperHitIndex, shiftedCoords,xLower,yLower,zLower,rtLower,xUpper,yUpper,zUpper,rtUpper);
            xn = shiftedCoords[0];
            yn = shiftedCoords[1];

            // Lower or the upper hit needs to be modified depending on which one was actually shifted
            if (modulesInGPU.moduleLayerType[lowerModuleIndex] == SDL::Pixel)
            {
                shiftedX = xn;
                shiftedY = yn;
                shiftedZ = zUpper;
                shiftedRt = alpaka::math::sqrt(acc, xn * xn + yn * yn);

                dPhi = deltaPhi_alpaka(acc, xLower,yLower,shiftedX, shiftedY); //function from Hit.cu
                noShiftedDphi = deltaPhi_alpaka(acc, xLower, yLower, xUpper, yUpper);
            }
            else
            {
                shiftedX = xn;
                shiftedY = yn;
                shiftedZ = zLower;
                shiftedRt = alpaka::math::sqrt(acc, xn * xn + yn * yn);
                dPhi = deltaPhi_alpaka(acc, shiftedX, shiftedY, xUpper, yUpper);
                noShiftedDphi = deltaPhi_alpaka(acc, xLower,yLower,xUpper,yUpper);
            }
        }
        else
        {
            dPhi = deltaPhi_alpaka(acc, xLower, yLower, xUpper, yUpper);
            noShiftedDphi = dPhi;
        }

        pass = pass & (alpaka::math::abs(acc, dPhi) < miniCut);
        if(not pass) return pass;

        // Cut #3: The dphi change going from lower Hit to upper Hit
        // Ref to original code: https://github.com/slava77/cms-tkph2-ntuple/blob/184d2325147e6930030d3d1f780136bc2dd29ce6/doubletAnalysis.C#L3076
        if (modulesInGPU.sides[lowerModuleIndex]!= Center)
        {
            // When it is tilted, use the new shifted positions
            // TODO: This is somewhat of an mystery.... somewhat confused why this is the case
            if (modulesInGPU.moduleLayerType[lowerModuleIndex] != SDL::Pixel)
            {
                // dPhi Change should be calculated so that the upper hit has higher rt.
                // In principle, this kind of check rt_lower < rt_upper should not be necessary because the hit shifting should have taken care of this.
                // (i.e. the strip hit is shifted to be aligned in the line of sight from interaction point to pixel hit of PS module guaranteeing rt ordering)
                // But I still placed this check for safety. (TODO: After cheking explicitly if not needed remove later?)
                // setdeltaPhiChange_alpaka(lowerHit.rt() < upperHitMod.rt() ? lowerHit.deltaPhiChange_alpaka(upperHitMod) : upperHitMod.deltaPhiChange_alpaka(lowerHit));


                dPhiChange = (rtLower < shiftedRt) ? deltaPhiChange_alpaka(acc, xLower, yLower, shiftedX, shiftedY) : deltaPhiChange_alpaka(acc, shiftedX, shiftedY, xLower, yLower); 
                noShiftedDphiChange = rtLower < rtUpper ? deltaPhiChange_alpaka(acc, xLower,yLower, xUpper, yUpper) : deltaPhiChange_alpaka(acc, xUpper, yUpper, xLower, yLower);
            }
            else
            {
                // dPhi Change should be calculated so that the upper hit has higher rt.
                // In principle, this kind of check rt_lower < rt_upper should not be necessary because the hit shifting should have taken care of this.
                // (i.e. the strip hit is shifted to be aligned in the line of sight from interaction point to pixel hit of PS module guaranteeing rt ordering)
                // But I still placed this check for safety. (TODO: After cheking explicitly if not needed remove later?)

                dPhiChange = (shiftedRt < rtUpper) ? deltaPhiChange_alpaka(acc, shiftedX, shiftedY, xUpper, yUpper) : deltaPhiChange_alpaka(acc, xUpper, yUpper, shiftedX, shiftedY);
                noShiftedDphiChange = rtLower < rtUpper ? deltaPhiChange_alpaka(acc, xLower,yLower, xUpper, yUpper) : deltaPhiChange_alpaka(acc, xUpper, yUpper, xLower, yLower);
            }
        }
        else
        {
            // When it is flat lying module, whichever is the lowerSide will always have rt lower
            dPhiChange = deltaPhiChange_alpaka(acc, xLower, yLower, xUpper, yUpper);
            noShiftedDphiChange = dPhiChange;
        }

        pass = pass & (alpaka::math::abs(acc, dPhiChange) < miniCut);

        return pass;
    }

    template<typename TAcc>
    ALPAKA_FN_ACC bool runMiniDoubletDefaultAlgoEndcap(TAcc const & acc, struct modules& modulesInGPU, uint16_t& lowerModuleIndex, uint16_t& upperModuleIndex, unsigned int lowerHitIndex, unsigned int upperHitIndex, float& drt, float& dPhi, float& dPhiChange, float& shiftedX, float& shiftedY, float& shiftedZ, float& noshiftedDz, float& noShiftedDphi, float& noShiftedDphichange,float xLower, float yLower, float zLower, float rtLower,float xUpper,float yUpper,float zUpper,float rtUpper)
    {

        bool pass = true; 

        // There are series of cuts that applies to mini-doublet in a "endcap" region

        // Cut #1 : dz cut. The dz difference can't be larger than 1cm. (max separation is 4mm for modules in the endcap)
        // Ref to original code: https://github.com/slava77/cms-tkph2-ntuple/blob/184d2325147e6930030d3d1f780136bc2dd29ce6/doubletAnalysis.C#L3093
        // For PS module in case when it is tilted a different dz (after the strip hit shift) is calculated later.

        float dz = zLower - zUpper; // Not const since later it might change depending on the type of module

        const float dzCut = 1.f;

        pass = pass & (alpaka::math::abs(acc, dz) < dzCut);
        if(not pass) return pass;
        // Cut #2 : drt cut. The dz difference can't be larger than 1cm. (max separation is 4mm for modules in the endcap)
        // Ref to original code: https://github.com/slava77/cms-tkph2-ntuple/blob/184d2325147e6930030d3d1f780136bc2dd29ce6/doubletAnalysis.C#L3100
        const float drtCut = modulesInGPU.moduleType[lowerModuleIndex] == SDL::PS ? 2.f : 10.f;
        drt = rtLower - rtUpper;
        pass = pass & (alpaka::math::abs(acc, drt) < drtCut);
        if(not pass) return pass;
        // The new scheme shifts strip hits to be "aligned" along the line of sight from interaction point to the pixel hit (if it is PS modules)
        float xn = 0, yn = 0, zn = 0;

        float shiftedCoords[3];
        shiftStripHits(modulesInGPU, /*hitsInGPU,*/ lowerModuleIndex, upperModuleIndex, lowerHitIndex, upperHitIndex, shiftedCoords,xLower,yLower,zLower,rtLower,xUpper,yUpper,zUpper,rtUpper);

        xn = shiftedCoords[0];
        yn = shiftedCoords[1];
        zn = shiftedCoords[2];

        if (modulesInGPU.moduleType[lowerModuleIndex] == SDL::PS)
        {
            // Appropriate lower or upper hit is modified after checking which one was actually shifted
            if (modulesInGPU.moduleLayerType[lowerModuleIndex] == SDL::Pixel)
            {
                // SDL::Hit upperHitMod(upperHit);
                // upperHitMod.setXYZ(xn, yn, upperHit.z());
                // setdeltaPhi_alpaka(lowerHit.deltaPhi_alpaka(upperHitMod));
                shiftedX = xn;
                shiftedY = yn;
                shiftedZ = zUpper;
                dPhi = deltaPhi_alpaka(acc, xLower, yLower, shiftedX, shiftedY);
                noShiftedDphi = deltaPhi_alpaka(acc, xLower, yLower, xUpper, yUpper);
            }
            else
            {
                // SDL::Hit lowerHitMod(lowerHit);
                // lowerHitMod.setXYZ(xn, yn, lowerHit.z());
                // setdeltaPhi_alpaka(lowerHitMod.deltaPhi_alpaka(upperHit));
                shiftedX = xn;
                shiftedY = yn;
                shiftedZ = zLower;
                dPhi = deltaPhi_alpaka(acc, shiftedX, shiftedY, xUpper, yUpper);
                noShiftedDphi = deltaPhi_alpaka(acc, xLower, yLower, xUpper, yUpper);
            }
        }
        else
        {
            shiftedX = xn;
            shiftedY = yn;
            shiftedZ = zUpper;
            dPhi = deltaPhi_alpaka(acc, xLower, yLower, xn, yn);
            noShiftedDphi = deltaPhi_alpaka(acc, xLower, yLower, xUpper, yUpper);
        }

        // dz needs to change if it is a PS module where the strip hits are shifted in order to properly account for the case when a tilted module falls under "endcap logic"
        // if it was an endcap it will have zero effect
        if (modulesInGPU.moduleType[lowerModuleIndex] == SDL::PS)
        {
            dz = modulesInGPU.moduleLayerType[lowerModuleIndex] == SDL::Pixel ? zLower - zn : zUpper - zn; 
        }

        float miniCut = 0;
        miniCut = modulesInGPU.moduleLayerType[lowerModuleIndex] == SDL::Pixel ?  dPhiThreshold(acc, rtLower, modulesInGPU, lowerModuleIndex,dPhi, dz) :  dPhiThreshold(acc, rtUpper, modulesInGPU, lowerModuleIndex, dPhi, dz);

        pass = pass & (alpaka::math::abs(acc, dPhi) < miniCut);
        if(not pass) return pass;

        // Cut #4: Another cut on the dphi after some modification
        // Ref to original code: https://github.com/slava77/cms-tkph2-ntuple/blob/184d2325147e6930030d3d1f780136bc2dd29ce6/doubletAnalysis.C#L3119-L3124

        
        float dzFrac = alpaka::math::abs(acc, dz) / alpaka::math::abs(acc, zLower);
        dPhiChange = dPhi / dzFrac * (1.f + dzFrac);
        noShiftedDphichange = noShiftedDphi / dzFrac * (1.f + dzFrac);
        pass = pass & (alpaka::math::abs(acc, dPhiChange) < miniCut);

        return pass;
    }

    void printMD(struct miniDoublets& mdsInGPU, struct hits& hitsInGPU, SDL::modules& modulesInGPU, unsigned int mdIndex);

    struct createMiniDoubletsInGPUv2
    {
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(
                TAcc const & acc,
                struct SDL::modules& modulesInGPU,
                struct SDL::hits& hitsInGPU,
                struct SDL::miniDoublets& mdsInGPU,
                struct SDL::objectRanges& rangesInGPU) const
        {
            using Dim = alpaka::Dim<TAcc>;
            using Idx = alpaka::Idx<TAcc>;
            using Vec = alpaka::Vec<Dim, Idx>;

            Vec const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
            Vec const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

            for(uint16_t lowerModuleIndex = globalThreadIdx[1]; lowerModuleIndex < (*modulesInGPU.nLowerModules); lowerModuleIndex += gridThreadExtent[1])
            {
                uint16_t upperModuleIndex = modulesInGPU.partnerModuleIndices[lowerModuleIndex];
                int nLowerHits = hitsInGPU.hitRangesnLower[lowerModuleIndex];
                int nUpperHits = hitsInGPU.hitRangesnUpper[lowerModuleIndex];
                if(hitsInGPU.hitRangesLower[lowerModuleIndex] == -1) continue;
                const int maxHits = alpaka::math::max(acc, nUpperHits, nLowerHits);
                unsigned int upHitArrayIndex = hitsInGPU.hitRangesUpper[lowerModuleIndex];
                unsigned int loHitArrayIndex = hitsInGPU.hitRangesLower[lowerModuleIndex];
                int limit = nUpperHits*nLowerHits;

                for(int hitIndex = globalThreadIdx[0]; hitIndex< limit; hitIndex += gridThreadExtent[0])
                {
                    int lowerHitIndex =  hitIndex / nUpperHits;
                    int upperHitIndex =  hitIndex % nUpperHits;
                    if(upperHitIndex >= nUpperHits) continue;
                    if(lowerHitIndex >= nLowerHits) continue;
                    unsigned int lowerHitArrayIndex = loHitArrayIndex + lowerHitIndex;
                    float xLower = hitsInGPU.xs[lowerHitArrayIndex];
                    float yLower = hitsInGPU.ys[lowerHitArrayIndex];
                    float zLower = hitsInGPU.zs[lowerHitArrayIndex];
                    float rtLower = hitsInGPU.rts[lowerHitArrayIndex];
                    unsigned int upperHitArrayIndex = upHitArrayIndex+upperHitIndex;
                    float xUpper = hitsInGPU.xs[upperHitArrayIndex];
                    float yUpper = hitsInGPU.ys[upperHitArrayIndex];
                    float zUpper = hitsInGPU.zs[upperHitArrayIndex];
                    float rtUpper = hitsInGPU.rts[upperHitArrayIndex];

                    float dz, dphi, dphichange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange;
                    //float dzCut, drtCut;//, miniCut;
                    bool success = runMiniDoubletDefaultAlgo(acc, modulesInGPU, lowerModuleIndex, upperModuleIndex, lowerHitArrayIndex, upperHitArrayIndex, dz, dphi, dphichange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange, xLower,yLower,zLower,rtLower,xUpper,yUpper,zUpper,rtUpper);
        if(success)
                    {
                        int totOccupancyMDs = alpaka::atomicOp<alpaka::AtomicAdd>(acc, &mdsInGPU.totOccupancyMDs[lowerModuleIndex], 1);
                        if(totOccupancyMDs >= (rangesInGPU.miniDoubletModuleIndices[lowerModuleIndex + 1] - rangesInGPU.miniDoubletModuleIndices[lowerModuleIndex]))
                        {
        #ifdef Warnings
                            printf("Mini-doublet excess alert! Module index =  %d\n",lowerModuleIndex);
        #endif
                        }
                        else
                        {
                            int mdModuleIndex = alpaka::atomicOp<alpaka::AtomicAdd>(acc, &mdsInGPU.nMDs[lowerModuleIndex], 1);
                            unsigned int mdIndex = rangesInGPU.miniDoubletModuleIndices[lowerModuleIndex] + mdModuleIndex;
        //#ifdef CUT_VALUE_DEBUG
        //                    addMDToMemory(mdsInGPU,hitsInGPU, modulesInGPU, lowerHitArrayIndex, upperHitArrayIndex, lowerModuleIndex, dz,drt, dphi, dphichange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange, dzCut, drtCut, miniCut, mdIndex);
        //#else
                            addMDToMemory(mdsInGPU,hitsInGPU, modulesInGPU, lowerHitArrayIndex, upperHitArrayIndex, lowerModuleIndex, dz, dphi, dphichange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange, mdIndex);
        //#endif
                        }

                    }
                }
            }
        }
    };

    


}

#endif

