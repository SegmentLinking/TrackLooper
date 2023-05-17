#ifndef MiniDoublet_cuh
#define MiniDoublet_cuh

#include "Constants.cuh"
#include "EndcapGeometry.cuh"
#include "TiltedGeometry.h"
#include "Module.cuh"
#include "Hit.cuh"

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

    void printMD(struct miniDoublets& mdsInGPU, struct hits& hitsInGPU, SDL::modules& modulesInGPU, unsigned int mdIndex);

    ALPAKA_FN_ACC ALPAKA_FN_INLINE void addMDToMemory(struct miniDoublets& mdsInGPU, struct hits& hitsInGPU, struct modules& modulesInGPU, unsigned int lowerHitIdx, unsigned int upperHitIdx, uint16_t& lowerModuleIdx, float dz, float dPhi, float dPhiChange, float shiftedX, float shiftedY, float shiftedZ, float noShiftedDz, float noShiftedDphi, float noShiftedDPhiChange, unsigned int idx)
    {
        //the index into which this MD needs to be written will be computed in the kernel
        //nMDs variable will be incremented in the kernel, no need to worry about that here

        mdsInGPU.moduleIndices[idx] = lowerModuleIdx;
        unsigned int anchorHitIndex, outerHitIndex;
        if(modulesInGPU.moduleType[lowerModuleIdx] == PS and modulesInGPU.moduleLayerType[lowerModuleIdx] == Strip)
        {
            mdsInGPU.anchorHitIndices[idx] = upperHitIdx;
            mdsInGPU.outerHitIndices[idx] = lowerHitIdx;

            anchorHitIndex = upperHitIdx;
            outerHitIndex = lowerHitIdx;
        }
        else
        {
            mdsInGPU.anchorHitIndices[idx] = lowerHitIdx;
            mdsInGPU.outerHitIndices[idx] = upperHitIdx;

            anchorHitIndex = lowerHitIdx;
            outerHitIndex = upperHitIdx;
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

        mdsInGPU.anchorX[idx] = hitsInGPU.xs[anchorHitIndex];
        mdsInGPU.anchorY[idx] = hitsInGPU.ys[anchorHitIndex];
        mdsInGPU.anchorZ[idx] = hitsInGPU.zs[anchorHitIndex];
        mdsInGPU.anchorRt[idx] = hitsInGPU.rts[anchorHitIndex];
        mdsInGPU.anchorPhi[idx] = hitsInGPU.phis[anchorHitIndex];
        mdsInGPU.anchorEta[idx] = hitsInGPU.etas[anchorHitIndex];
        mdsInGPU.anchorHighEdgeX[idx] = hitsInGPU.highEdgeXs[anchorHitIndex];
        mdsInGPU.anchorHighEdgeY[idx] = hitsInGPU.highEdgeYs[anchorHitIndex];
        mdsInGPU.anchorLowEdgeX[idx] = hitsInGPU.lowEdgeXs[anchorHitIndex];
        mdsInGPU.anchorLowEdgeY[idx] = hitsInGPU.lowEdgeYs[anchorHitIndex];

        mdsInGPU.outerX[idx] = hitsInGPU.xs[outerHitIndex];
        mdsInGPU.outerY[idx] = hitsInGPU.ys[outerHitIndex];
        mdsInGPU.outerZ[idx] = hitsInGPU.zs[outerHitIndex];
        mdsInGPU.outerRt[idx] = hitsInGPU.rts[outerHitIndex];
        mdsInGPU.outerPhi[idx] = hitsInGPU.phis[outerHitIndex];
        mdsInGPU.outerEta[idx] = hitsInGPU.etas[outerHitIndex];
        mdsInGPU.outerHighEdgeX[idx] = hitsInGPU.highEdgeXs[outerHitIndex];
        mdsInGPU.outerHighEdgeY[idx] = hitsInGPU.highEdgeYs[outerHitIndex];
        mdsInGPU.outerLowEdgeX[idx] = hitsInGPU.lowEdgeXs[outerHitIndex];
        mdsInGPU.outerLowEdgeY[idx] = hitsInGPU.lowEdgeYs[outerHitIndex];
    };

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
    };

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
    };

    template<typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float dPhiThreshold(TAcc const & acc, float rt, struct modules& modulesInGPU, uint16_t& moduleIndex, float dPhi = 0, float dz = 0)
    {
        // =================================================================
        // Various constants
        // =================================================================
        //mean of the horizontal layer position in y; treat this as R below

        // =================================================================
        // Computing some components that make up the cut threshold
        // =================================================================

        unsigned int iL = modulesInGPU.layers[moduleIndex] - 1;
        const float miniSlope = alpaka::math::asin(acc, alpaka::math::min(acc, rt * k2Rinv1GeVf / ptCut, sinAlphaMax));
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
    };

    template<typename TAcc>
    ALPAKA_FN_INLINE ALPAKA_FN_ACC void shiftStripHits(TAcc const & acc, struct modules& modulesInGPU, uint16_t& lowerModuleIndex, uint16_t& upperModuleIndex, unsigned int lowerHitIndex, unsigned int upperHitIndex, float* shiftedCoords, float xLower, float yLower, float zLower, float rtLower,float xUpper,float yUpper,float zUpper,float rtUpper)
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
        float zp; // pixel y (pixel hit y)
        float rtp; // pixel y (pixel hit y)
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
        float moduleSeparation;
        float drprime; // The radial shift size in x-y plane projection
        float drprime_x; // x-component of drprime
        float drprime_y; // y-component of drprime
        float& slope = modulesInGPU.slopes[lowerModuleIndex]; // The slope of the possible strip hits for a given module in x-y plane
        float absArctanSlope;
        float angleM; // the angle M is the angle of rotation of the module in x-y plane if the possible strip hits are along the x-axis, then angleM = 0, and if the possible strip hits are along y-axis angleM = 90 degrees
        float absdzprime; // The distance between the two points after shifting
        float& drdz_ = modulesInGPU.drdzs[lowerModuleIndex];
        // Assign hit pointers based on their hit type
        if (modulesInGPU.moduleType[lowerModuleIndex] == PS)
        {
            // TODO: This is somewhat of an mystery.... somewhat confused why this is the case
            if (modulesInGPU.subdets[lowerModuleIndex] == Barrel ? modulesInGPU.moduleLayerType[lowerModuleIndex] != Pixel : modulesInGPU.moduleLayerType[lowerModuleIndex] == Pixel)
            {
                xo =xUpper;
                yo =yUpper;
                xp =xLower;
                yp =yLower;
                zp =zLower;
                rtp =rtLower;
                xp =xLower;
                yp =yLower;
                zp =zLower;
                rtp =rtLower;
            }
            else
            {
                xo = xLower;
                yo = yLower;
                xp = xUpper;
                yp = yUpper;
                zp = zUpper;
                rtp=rtUpper;
                xp = xUpper;
                yp = yUpper;
                zp = zUpper;
                rtp=rtUpper;
            }
        }
        else
        {
            xo =xUpper;
            yo =yUpper;
            xp =xLower;
            yp =yLower;
            zp =zLower;
            rtp =rtLower;
            xp =xLower;
            yp =yLower;
            zp =zLower;
            rtp =rtLower;
        }

        // If it is endcap some of the math gets simplified (and also computers don't like infinities)
        isEndcap = modulesInGPU.subdets[lowerModuleIndex]== Endcap;

        // NOTE: TODO: Keep in mind that the sin(atan) function can be simplifed to something like x / sqrt(1 + x^2) and similar for cos
        // I am not sure how slow sin, atan, cos, functions are in c++. If x / sqrt(1 + x^2) are faster change this later to reduce arithmetic computation time
        angleA = alpaka::math::abs(acc, alpaka::math::atan(acc, rtp / zp));
        angleB = ((isEndcap) ? float(M_PI) / 2.f : alpaka::math::atan(acc, drdz_)); // The tilt module on the postive z-axis has negative drdz slope in r-z plane and vice versa

        moduleSeparation = moduleGapSize(modulesInGPU, lowerModuleIndex);

        // Sign flips if the pixel is later layer
        if (modulesInGPU.moduleType[lowerModuleIndex] == PS and modulesInGPU.moduleLayerType[lowerModuleIndex] != Pixel)
        {
            moduleSeparation *= -1;
        }

        drprime = (moduleSeparation / alpaka::math::sin(acc, angleA + angleB)) * alpaka::math::sin(acc, angleA);
        
        // Compute arctan of the slope and take care of the slope = infinity case
        absArctanSlope = ((slope != SDL::SDL_INF) ? fabs(alpaka::math::atan(acc, slope)) : float(M_PI) / 2.f);

        // Depending on which quadrant the pixel hit lies, we define the angleM by shifting them slightly differently
        if (xp > 0 and yp > 0)
        {
            angleM = absArctanSlope;
        }
        else if (xp > 0 and yp < 0)
        {
            angleM = float(M_PI) - absArctanSlope;
        }
        else if (xp < 0 and yp < 0)
        {
            angleM = float(M_PI) + absArctanSlope;
        }
        else // if (xp < 0 and yp > 0)
        {
            angleM = 2.f * float(M_PI) - absArctanSlope;
        }

        // Then since the angleM sign is taken care of properly
        drprime_x = drprime * alpaka::math::sin(acc, angleM);
        drprime_y = drprime * alpaka::math::cos(acc, angleM);

        // The new anchor position is
        xa = xp + drprime_x;
        ya = yp + drprime_y;

        // Compute the new strip hit position (if the slope vaule is in special condition take care of the exceptions)
        if (slope == SDL::SDL_INF) // Designated for tilted module when the slope is exactly infinity (module lying along y-axis)
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
        absdzprime = alpaka::math::abs(acc, moduleSeparation / alpaka::math::sin(acc, angleA + angleB) * alpaka::math::cos(acc, angleA)); // module separation sign is for shifting in radial direction for z-axis direction take care of the sign later

        // Depending on which one as closer to the interactin point compute the new z wrt to the pixel properly
        if (modulesInGPU.moduleLayerType[lowerModuleIndex] == Pixel)
        {
            abszn = alpaka::math::abs(acc, zp) + absdzprime;
        }
        else
        {
            abszn = alpaka::math::abs(acc, zp) - absdzprime;
        }

        zn = abszn * ((zp > 0) ? 1 : -1); // Apply the sign of the zn

        shiftedCoords[0] = xn;
        shiftedCoords[1] = yn;
        shiftedCoords[2] = zn;
    };

    template<typename TAcc>
    ALPAKA_FN_ACC bool runMiniDoubletDefaultAlgo(TAcc const & acc, struct modules& modulesInGPU, uint16_t& lowerModuleIndex, uint16_t& upperModuleIndex, unsigned int lowerHitIndex, unsigned int upperHitIndex, float& dz, float& dPhi, float& dPhiChange, float& shiftedX, float& shiftedY, float& shiftedZ, float& noShiftedDz, float& noShiftedDphi, float& noShiftedDphiChange, float xLower, float yLower, float zLower, float rtLower,float xUpper,float yUpper,float zUpper,float rtUpper)
    {
        if(modulesInGPU.subdets[lowerModuleIndex] == SDL::Barrel)
        {
            return runMiniDoubletDefaultAlgoBarrel(acc, modulesInGPU, lowerModuleIndex, upperModuleIndex, lowerHitIndex, upperHitIndex, dz, dPhi, dPhiChange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange,xLower,yLower,zLower,rtLower, xUpper,yUpper,zUpper,rtUpper);
        }
        else
        {
            return runMiniDoubletDefaultAlgoEndcap(acc, modulesInGPU, lowerModuleIndex, upperModuleIndex, lowerHitIndex, upperHitIndex, dz, dPhi, dPhiChange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange,xLower,yLower,zLower,rtLower, xUpper,yUpper,zUpper,rtUpper);
        }
    };

    template<typename TAcc>
    ALPAKA_FN_ACC bool runMiniDoubletDefaultAlgoBarrel(TAcc const & acc, struct modules& modulesInGPU, uint16_t& lowerModuleIndex, uint16_t& upperModuleIndex, unsigned int lowerHitIndex, unsigned int upperHitIndex, float& dz, float& dPhi, float& dPhiChange, float& shiftedX, float& shiftedY, float& shiftedZ, float& noshiftedDz, float& noShiftedDphi, float& noShiftedDphiChange, float xLower,float yLower, float zLower, float rtLower,float xUpper,float yUpper,float zUpper,float rtUpper)
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
            shiftStripHits(acc, modulesInGPU, lowerModuleIndex, upperModuleIndex, lowerHitIndex, upperHitIndex, shiftedCoords,xLower,yLower,zLower,rtLower,xUpper,yUpper,zUpper,rtUpper);
            xn = shiftedCoords[0];
            yn = shiftedCoords[1];

            // Lower or the upper hit needs to be modified depending on which one was actually shifted
            if (modulesInGPU.moduleLayerType[lowerModuleIndex] == SDL::Pixel)
            {
                shiftedX = xn;
                shiftedY = yn;
                shiftedZ = zUpper;
                shiftedRt = alpaka::math::sqrt(acc, xn * xn + yn * yn);

                dPhi = SDL::deltaPhi(acc, xLower,yLower,shiftedX, shiftedY); //function from Hit.cu
                noShiftedDphi = SDL::deltaPhi(acc, xLower, yLower, xUpper, yUpper);
            }
            else
            {
                shiftedX = xn;
                shiftedY = yn;
                shiftedZ = zLower;
                shiftedRt = alpaka::math::sqrt(acc, xn * xn + yn * yn);
                dPhi = SDL::deltaPhi(acc, shiftedX, shiftedY, xUpper, yUpper);
                noShiftedDphi = SDL::deltaPhi(acc, xLower,yLower,xUpper,yUpper);
            }
        }
        else
        {
            dPhi = SDL::deltaPhi(acc, xLower, yLower, xUpper, yUpper);
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
                // setdeltaPhiChange(lowerHit.rt() < upperHitMod.rt() ? lowerHit.deltaPhiChange(upperHitMod) : upperHitMod.deltaPhiChange(lowerHit));

                dPhiChange = (rtLower < shiftedRt) ? SDL::deltaPhiChange(acc, xLower, yLower, shiftedX, shiftedY) : SDL::deltaPhiChange(acc, shiftedX, shiftedY, xLower, yLower); 
                noShiftedDphiChange = rtLower < rtUpper ? SDL::deltaPhiChange(acc, xLower,yLower, xUpper, yUpper) : SDL::deltaPhiChange(acc, xUpper, yUpper, xLower, yLower);
            }
            else
            {
                // dPhi Change should be calculated so that the upper hit has higher rt.
                // In principle, this kind of check rt_lower < rt_upper should not be necessary because the hit shifting should have taken care of this.
                // (i.e. the strip hit is shifted to be aligned in the line of sight from interaction point to pixel hit of PS module guaranteeing rt ordering)
                // But I still placed this check for safety. (TODO: After cheking explicitly if not needed remove later?)

                dPhiChange = (shiftedRt < rtUpper) ? SDL::deltaPhiChange(acc, shiftedX, shiftedY, xUpper, yUpper) : SDL::deltaPhiChange(acc, xUpper, yUpper, shiftedX, shiftedY);
                noShiftedDphiChange = rtLower < rtUpper ? SDL::deltaPhiChange(acc, xLower,yLower, xUpper, yUpper) : SDL::deltaPhiChange(acc, xUpper, yUpper, xLower, yLower);
            }
        }
        else
        {
            // When it is flat lying module, whichever is the lowerSide will always have rt lower
            dPhiChange = SDL::deltaPhiChange(acc, xLower, yLower, xUpper, yUpper);
            noShiftedDphiChange = dPhiChange;
        }

        pass = pass & (alpaka::math::abs(acc, dPhiChange) < miniCut);

        return pass;
    };

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
        shiftStripHits(acc, modulesInGPU, lowerModuleIndex, upperModuleIndex, lowerHitIndex, upperHitIndex, shiftedCoords,xLower,yLower,zLower,rtLower,xUpper,yUpper,zUpper,rtUpper);

        xn = shiftedCoords[0];
        yn = shiftedCoords[1];
        zn = shiftedCoords[2];

        if (modulesInGPU.moduleType[lowerModuleIndex] == SDL::PS)
        {
            // Appropriate lower or upper hit is modified after checking which one was actually shifted
            if (modulesInGPU.moduleLayerType[lowerModuleIndex] == SDL::Pixel)
            {
                shiftedX = xn;
                shiftedY = yn;
                shiftedZ = zUpper;
                dPhi = SDL::deltaPhi(acc, xLower, yLower, shiftedX, shiftedY);
                noShiftedDphi = SDL::deltaPhi(acc, xLower, yLower, xUpper, yUpper);
            }
            else
            {
                shiftedX = xn;
                shiftedY = yn;
                shiftedZ = zLower;
                dPhi = SDL::deltaPhi(acc, shiftedX, shiftedY, xUpper, yUpper);
                noShiftedDphi = SDL::deltaPhi(acc, xLower, yLower, xUpper, yUpper);
            }
        }
        else
        {
            shiftedX = xn;
            shiftedY = yn;
            shiftedZ = zUpper;
            dPhi = SDL::deltaPhi(acc, xLower, yLower, xn, yn);
            noShiftedDphi = SDL::deltaPhi(acc, xLower, yLower, xUpper, yUpper);
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
    };

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

                for(int hitIndex = globalThreadIdx[2]; hitIndex< limit; hitIndex += gridThreadExtent[2])
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
                    bool success = runMiniDoubletDefaultAlgo(acc, modulesInGPU, lowerModuleIndex, upperModuleIndex, lowerHitArrayIndex, upperHitArrayIndex, dz, dphi, dphichange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange, xLower,yLower,zLower,rtLower,xUpper,yUpper,zUpper,rtUpper);
                    if(success)
                    {
                        int totOccupancyMDs = alpaka::atomicOp<alpaka::AtomicAdd>(acc, &mdsInGPU.totOccupancyMDs[lowerModuleIndex], 1);
                        if(totOccupancyMDs >= (rangesInGPU.miniDoubletModuleOccupancy[lowerModuleIndex]))
                        {
#ifdef Warnings
                            printf("Mini-doublet excess alert! Module index =  %d\n",lowerModuleIndex);
#endif
                        }
                        else
                        {
                            int mdModuleIndex = alpaka::atomicOp<alpaka::AtomicAdd>(acc, &mdsInGPU.nMDs[lowerModuleIndex], 1);
                            unsigned int mdIndex = rangesInGPU.miniDoubletModuleIndices[lowerModuleIndex] + mdModuleIndex;

                            addMDToMemory(mdsInGPU,hitsInGPU, modulesInGPU, lowerHitArrayIndex, upperHitArrayIndex, lowerModuleIndex, dz, dphi, dphichange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange, mdIndex);
                        }
                    }
                }
            }
        }
    };

    struct createMDArrayRangesGPU
    {
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(
                TAcc const & acc,
                struct modules& modulesInGPU,
                struct objectRanges& rangesInGPU) const
        {
            using Dim = alpaka::Dim<TAcc>;
            using Idx = alpaka::Idx<TAcc>;
            using Vec = alpaka::Vec<Dim, Idx>;

            Vec const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
            Vec const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

            // Initialize variables in shared memory and set to 0
            int& nTotalMDs = alpaka::declareSharedVar<int, __COUNTER__>(acc); nTotalMDs = 0;
            alpaka::syncBlockThreads(acc);

            // Initialize variables outside of the for loop.
            float module_eta;
            unsigned int nTotMDs;
            int occupancy, category_number, eta_number;
            short module_subdets, module_layers, module_rings;

            for(uint16_t i = globalThreadIdx[2]; i < *modulesInGPU.nLowerModules; i += gridThreadExtent[2])
            {
                module_rings = modulesInGPU.rings[i];
                module_layers = modulesInGPU.layers[i];
                module_subdets = modulesInGPU.subdets[i];
                module_eta = alpaka::math::abs(acc, modulesInGPU.eta[i]);

                if (module_layers<=3 && module_subdets==5) category_number = 0;
                else if (module_layers>=4 && module_subdets==5) category_number = 1;
                else if (module_layers<=2 && module_subdets==4 && module_rings>=11) category_number = 2;
                else if (module_layers>=3 && module_subdets==4 && module_rings>=8) category_number = 2;
                else if (module_layers<=2 && module_subdets==4 && module_rings<=10) category_number = 3;
                else if (module_layers>=3 && module_subdets==4 && module_rings<=7) category_number = 3;

                if (module_eta<0.75) eta_number = 0;
                else if (module_eta>0.75 && module_eta<1.5) eta_number = 1;
                else if (module_eta>1.5  && module_eta<2.25) eta_number = 2;
                else if (module_eta>2.25 && module_eta<3) eta_number = 3;

                if (category_number == 0 && eta_number == 0) occupancy = 49;
                else if (category_number == 0 && eta_number == 1) occupancy = 42;
                else if (category_number == 0 && eta_number == 2) occupancy = 37;
                else if (category_number == 0 && eta_number == 3) occupancy = 41;
                else if (category_number == 1) occupancy = 100;
                else if (category_number == 2 && eta_number == 1) occupancy = 16;
                else if (category_number == 2 && eta_number == 2) occupancy = 19;
                else if (category_number == 3 && eta_number == 1) occupancy = 14;
                else if (category_number == 3 && eta_number == 2) occupancy = 20;
                else if (category_number == 3 && eta_number == 3) occupancy = 25;

                nTotMDs = alpaka::atomicOp<alpaka::AtomicAdd>(acc, &nTotalMDs, occupancy);

                rangesInGPU.miniDoubletModuleIndices[i] = nTotMDs; 
                rangesInGPU.miniDoubletModuleOccupancy[i] = occupancy;
            }

            // Wait for all threads to finish before reporting final values
            alpaka::syncBlockThreads(acc);
            if(globalThreadIdx[2] == 0)
            {
                rangesInGPU.miniDoubletModuleIndices[*modulesInGPU.nLowerModules] = nTotalMDs;
                *rangesInGPU.device_nTotalMDs = nTotalMDs;
            }
        }
    };

    struct addMiniDoubletRangesToEventExplicit
    {
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(
                TAcc const & acc,
                struct modules& modulesInGPU,
                struct miniDoublets& mdsInGPU,
                struct objectRanges& rangesInGPU,
                struct hits& hitsInGPU) const
        {
            using Dim = alpaka::Dim<TAcc>;
            using Idx = alpaka::Idx<TAcc>;
            using Vec = alpaka::Vec<Dim, Idx>;

            Vec const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
            Vec const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

            for(uint16_t i = globalThreadIdx[2]; i < *modulesInGPU.nLowerModules; i += gridThreadExtent[2])
            {
                if(mdsInGPU.nMDs[i] == 0 or hitsInGPU.hitRanges[i * 2] == -1)
                {
                    rangesInGPU.mdRanges[i * 2] = -1;
                    rangesInGPU.mdRanges[i * 2 + 1] = -1;
                }
                else
                {
                    rangesInGPU.mdRanges[i * 2] = rangesInGPU.miniDoubletModuleIndices[i] ;
                    rangesInGPU.mdRanges[i * 2 + 1] = rangesInGPU.miniDoubletModuleIndices[i] + mdsInGPU.nMDs[i] - 1;
                }
            }
        }
    };
}
#endif