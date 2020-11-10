#ifndef MiniDoublet_h
#define MiniDoublet_h

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
        unsigned int* hitIndices;
        unsigned int* moduleIndices;
        short* pixelModuleFlag;
        unsigned int* nMDs; //counter per module
        float* dphichanges;

        float* dzs; //will store drt if the module is endcap
        float*dphis;

        float* shiftedXs;
        float* shiftedYs;
        float* shiftedZs;
        float* noShiftedDzs; //if shifted module
        float* noShiftedDphis; //if shifted module
        float* noShiftedDphiChanges; //if shifted module

        miniDoublets();
      	void freeMemory();
      	void freeMemoryCache();

    };



    void createMDsInUnifiedMemory(struct miniDoublets& mdsInGPU, unsigned int maxMDs,unsigned int nModules);
    void createMDsInExplicitMemory(struct miniDoublets& mdsInGPU,struct miniDoublets& mdsInTemp, unsigned int maxMDs,unsigned int nModules);
    //for successful MDs
    CUDA_DEV void addMDToMemory(struct miniDoublets& mdsInGPU, struct hits& hitsInGPU, struct modules& modulesInGPU, unsigned int lowerHitIdx, unsigned int upperHitIdx, unsigned int lowerModuleIdx, float dz, float dphi, float dphichange, float shfitedX, float shiftedY, float shiftedZ, float noShiftedDz, float noShiftedDphi, float noShiftedDPhiChange, unsigned int idx);

    CUDA_DEV float dPhiThreshold(struct hits& hitsInGPU, struct modules& modulesInGPU, unsigned int hitIndex, unsigned int moduleIndex, float dPhi = 0, float dz = 0);
    CUDA_DEV extern inline float isTighterTiltedModules(struct modules& modulesInGPU, unsigned int moduleIndex);
    CUDA_DEV void initModuleGapSize();

    CUDA_DEV float moduleGapSize(struct modules& modulesInGPU, unsigned int moduleIndex);


    CUDA_DEV bool runMiniDoubletDefaultAlgo(struct modules& modulesInGPU, struct hits& hitsInGPU, float& dz, float& dphi, float& dphichange, float& shiftedX, float& shiftedY, float& shiftedZ, float& noShiftedDz, float& noShiftedDphi, float& noShiftedDphiChange);

    CUDA_DEV void shiftStripHits(struct modules& modulesInGPU, struct hits& hitsInGPU, unsigned int lowerModuleIndex, unsigned int lowerHitIndex, unsigned int upperHitIndex, float* shiftedCoords);


    CUDA_DEV bool runMiniDoubletDefaultAlgo(struct modules& modulesInGPU, struct hits& hitsInGPU, unsigned int lowerModuleIndex, unsigned int lowerHitIndex, unsigned int upperHitIndex, float& dz, float& dphi, float& dphichange, float& shiftedX, float& shiftedY, float& shiftedZ, float& noShiftedDz, float& noShiftedDphi, float& noShiftedDphiChange);

    CUDA_DEV bool runMiniDoubletDefaultAlgoBarrel(struct modules& modulesInGPU, struct hits& hitsInGPU, unsigned int lowerModuleIndex, unsigned int lowerHitIndex, unsigned int upperHitIndex, float& dz, float& dphi, float& dphichange, float& shiftedX, float& shiftedY, float& shiftedZ, float& noShiftedDz, float& noShiftedDphi, float& noShiftedDphiChange);

    CUDA_DEV bool runMiniDoubletDefaultAlgoEndcap(struct modules& modulesInGPU, struct hits& hitsInGPU, unsigned int lowerModuleIndex, unsigned int lowerHitIndex, unsigned int upperHitIndex, float& drt, float& dphi, float& dphichange, float& shiftedX, float& shiftedY, float& shiftedZ, float& noShiftedDz, float& noShiftedDphi, float& noShiftedDphiChange);

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

}

#endif

