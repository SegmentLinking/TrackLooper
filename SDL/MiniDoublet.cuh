#ifndef MiniDoublet_cuh
#define MiniDoublet_cuh

#ifdef __CUDACC__
#define CUDA_HOSTDEV  __host__ __device__
#define CUDA_DEV __device__
#define CUDA_CONST_VAR __device__
#else
#define CUDA_HOSTDEV
#define CUDA_CONST_VAR
#define CUDA_DEV
#endif


#include <array>
#include <tuple>
#include <cmath>
#include "Constants.h"
#include "EndcapGeometry.h"
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
        unsigned int* anchorHitIndices;
        unsigned int* outerHitIndices;
        unsigned int* moduleIndices;
        unsigned int* nMDs; //counter per module
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
        unsigned int* anchorHitIdx; //idx in ntuple
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
        unsigned int* outerHitIdx;
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
        void resetMemory(unsigned int maxMDsPerModule, unsigned int nModules, unsigned int maxPixelMDs,cudaStream_t stream);

    };



    void createMDsInUnifiedMemory(struct miniDoublets& mdsInGPU, unsigned int maxMDs,unsigned int nModules, unsigned int maxPixelMDs, cudaStream_t stream);
    void createMDsInExplicitMemory(struct miniDoublets& mdsInGPU, unsigned int maxMDs,unsigned int nModules, unsigned int maxPixelMDs,cudaStream_t stream);

#ifdef CUT_VALUE_DEBUG
    CUDA_HOSTDEV void addMDToMemory(struct miniDoublets& mdsInGPU, struct hits& hitsInGPU, struct modules& modulesInGPU, unsigned int lowerHitIdx, unsigned int upperHitIdx, unsigned int lowerModuleIdx, float dz, float drt, float dphi, float dphichange, float shfitedX, float shiftedY, float shiftedZ, float noShiftedDz, float noShiftedDphi, float noShiftedDPhiChange, float dzCut, float drtCut, float miniCut, unsigned int idx);

#else
    //for successful MDs
    CUDA_HOSTDEV void addMDToMemory(struct miniDoublets& mdsInGPU, struct hits& hitsInGPU, struct modules& modulesInGPU, unsigned int lowerHitIdx, unsigned int upperHitIdx, unsigned int lowerModuleIdx, float dz, float dphi, float dphichange, float shfitedX, float shiftedY, float shiftedZ, float noShiftedDz, float noShiftedDphi, float noShiftedDPhiChange, unsigned int idx);

#endif

    CUDA_DEV float dPhiThreshold(struct hits& hitsInGPU, struct modules& modulesInGPU, unsigned int hitIndex, unsigned int moduleIndex, float dPhi = 0, float dz = 0);
    CUDA_DEV extern inline float isTighterTiltedModules(struct modules& modulesInGPU, unsigned int moduleIndex);
    CUDA_DEV void initModuleGapSize();

    CUDA_DEV float moduleGapSize(struct modules& modulesInGPU, unsigned int moduleIndex);

    CUDA_DEV void shiftStripHits(struct modules& modulesInGPU, struct hits& hitsInGPU, unsigned int lowerModuleIndex, unsigned int lowerHitIndex, unsigned int upperHitIndex, float* shiftedCoords);

    CUDA_DEV bool runMiniDoubletDefaultAlgo(struct modules& modulesInGPU, struct hits& hitsInGPU, unsigned int lowerModuleIndex, unsigned int lowerHitIndex, unsigned int upperHitIndex, float& dz, float& drt, float& dphi, float& dphichange, float& shiftedX, float& shiftedY, float& shiftedZ, float& noShiftedDz, float& noShiftedDphi, float& noShiftedDphiChange, float& dzCut, float& drtCut, float& miniCut);

    CUDA_DEV bool runMiniDoubletDefaultAlgoBarrel(struct modules& modulesInGPU, struct hits& hitsInGPU, unsigned int lowerModuleIndex, unsigned int lowerHitIndex, unsigned int upperHitIndex, float& dz, float& drt, float& dphi, float& dphichange, float& shiftedX, float& shiftedY, float& shiftedZ, float& noShiftedDz, float& noShiftedDphi, float& noShiftedDphiChange, float& dzCut, float& drtCut, float& miniCut);

    CUDA_DEV bool runMiniDoubletDefaultAlgoEndcap(struct modules& modulesInGPU, struct hits& hitsInGPU, unsigned int lowerModuleIndex, unsigned int lowerHitIndex, unsigned int upperHitIndex, float& dz, float& drt, float& dphi, float& dphichange, float& shiftedX, float& shiftedY, float& shiftedZ, float& noShiftedDz, float& noShiftedDphi, float& noShiftedDphiChange, float& dzCut, float& drtCut, float& miniCut);

    void printMD(struct miniDoublets& mdsInGPU, struct hits& hitsInGPU, SDL::modules& modulesInGPU, unsigned int mdIndex);


    //constant variables
    extern CUDA_CONST_VAR float sinAlphaMax;
    extern CUDA_CONST_VAR float k2Rinv1GeVf;
    extern CUDA_CONST_VAR float ptCut;
    extern CUDA_CONST_VAR float deltaZLum;

    extern CUDA_CONST_VAR float miniMulsPtScaleBarrel[6];
    extern CUDA_CONST_VAR float miniMulsPtScaleEndcap[5];
    extern CUDA_CONST_VAR float miniRminMeanBarrel[6];
    extern CUDA_CONST_VAR float miniRminMeanEndcap[5];
    extern CUDA_CONST_VAR float miniDeltaTilted[3];
    extern CUDA_CONST_VAR float miniDeltaFlat[6];
    extern CUDA_CONST_VAR float miniDeltaLooseTilted[3];
    extern CUDA_CONST_VAR float miniDeltaEndcap[5][15];
    extern CUDA_CONST_VAR float pixelPSZpitch;
    extern CUDA_CONST_VAR float strip2SZpitch;

//CUDA_DEV float inline moduleGapSize(struct modules& modulesInGPU, unsigned int moduleIndex)
//{
//    float miniDeltaTilted[3] = {0.26f, 0.26f, 0.26f};
//    float miniDeltaFlat[6] ={0.26f, 0.16f, 0.16f, 0.18f, 0.18f, 0.18f};
//    float miniDeltaLooseTilted[3] = {0.4f,0.4f,0.4f};
//    float miniDeltaEndcap[5][15];
//
//    for (size_t i = 0; i < 5; i++)
//    {
//        for (size_t j = 0; j < 15; j++)
//        {
//            if (i == 0 || i == 1)
//            {
//                if (j < 10)
//                {
//                    miniDeltaEndcap[i][j] = 0.4f;
//                }
//                else
//                {
//                    miniDeltaEndcap[i][j] = 0.18f;
//                }
//            }
//            else if (i == 2 || i == 3)
//            {
//                if (j < 8)
//                {
//                    miniDeltaEndcap[i][j] = 0.4f;
//                }
//                else
//                {
//                    miniDeltaEndcap[i][j]  = 0.18f;
//                }
//            }
//            else
//            {
//                if (j < 9)
//                {
//                    miniDeltaEndcap[i][j] = 0.4f;
//                }
//                else
//                {
//                    miniDeltaEndcap[i][j] = 0.18f;
//                }
//            }
//        }
//    }
//
//
//    unsigned int iL = modulesInGPU.layers[moduleIndex]-1;
//    unsigned int iR = modulesInGPU.rings[moduleIndex] - 1;
//    short subdet = modulesInGPU.subdets[moduleIndex];
//    short side = modulesInGPU.sides[moduleIndex];
//
//    float moduleSeparation = 0;
//
//    if (subdet == Barrel and side == Center)
//    {
//        moduleSeparation = miniDeltaFlat[iL];
//    }
//    else if (isTighterTiltedModules(modulesInGPU, moduleIndex))
//    {
//        moduleSeparation = miniDeltaTilted[iL];
//    }
//    else if (subdet == Endcap)
//    {
//        moduleSeparation = miniDeltaEndcap[iL][iR];
//    }
//    else //Loose tilted modules
//    {
//        moduleSeparation = miniDeltaLooseTilted[iL];
//    }
//
//    return moduleSeparation;
//}

}

#endif

