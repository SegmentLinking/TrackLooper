#ifndef MiniDoublet_cuh
#define MiniDoublet_cuh

#include <alpaka/alpaka.hpp>

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
        unsigned int* nMDs; //counter per module
        unsigned int* totOccupancyMDs; //counter per module
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


    void createMDArrayRanges(struct modules& modulesInGPU, struct objectRanges& rangesInGPU, uint16_t& nLowerModules, unsigned int& nTotalMDs, cudaStream_t stream, const unsigned int& maxPixelMDs);


//#ifdef CUT_VALUE_DEBUG
//    ALPAKA_FN_ACC void addMDToMemory(struct miniDoublets& mdsInGPU, struct hits& hitsInGPU, struct modules& modulesInGPU, unsigned int lowerHitIdx, unsigned int upperHitIdx, uint16_t& lowerModuleIdx, float dz, float drt, float dPhi, float dPhiChange, float shiftedX, float shiftedY, float shiftedZ, float noShiftedDz, float noShiftedDphi, float noShiftedDPhiChange, float dzCut, float drtCut, float miniCut, unsigned int idx);
//#else
    //for successful MDs
    ALPAKA_FN_HOST_ACC void addMDToMemory(struct miniDoublets& mdsInGPU, struct hits& hitsInGPU, struct modules& modulesInGPU, unsigned int lowerHitIdx, unsigned int upperHitIdx, uint16_t& lowerModuleIdx, float dz, float dPhi, float dPhiChange, float shiftedX, float shiftedY, float shiftedZ, float noShiftedDz, float noShiftedDphi, float noShiftedDPhiChange, unsigned int idx);
//#endif

    //ALPAKA_FN_ACC float dPhiThreshold(struct hits& hitsInGPU, struct modules& modulesInGPU, unsigned int hitIndex, unsigned int moduleIndex, float dPhi = 0, float dz = 0);
    ALPAKA_FN_ACC extern inline float dPhiThreshold(float rt, struct modules& modulesInGPU, uint16_t&  moduleIndex, float dPhi = 0, float dz = 0);
    ALPAKA_FN_ACC extern inline float isTighterTiltedModules(struct modules& modulesInGPU, uint16_t& moduleIndex);
    ALPAKA_FN_ACC void initModuleGapSize();

    ALPAKA_FN_ACC extern inline float moduleGapSize(struct modules& modulesInGPU, uint16_t& moduleIndex);



    ALPAKA_FN_ACC void shiftStripHits(struct modules& modulesInGPU, uint16_t& lowerModuleIndex, uint16_t& upperModuleIndex, unsigned int lowerHitIndex, unsigned int upperHitIndex, float* shiftedCoords,float xLower,float yLower,float zLower,float rtLower,float xUpper,float yUpper,float zUpper,float rtUpper);

    ALPAKA_FN_ACC bool runMiniDoubletDefaultAlgo(struct modules& modulesInGPU, uint16_t& lowerModuleIndex, uint16_t& upperModuleIndex, unsigned int lowerHitIndex, unsigned int upperHitIndex, float& dz, float& dphi, float& dphichange, float& shiftedX, float& shiftedY, float& shiftedZ, float& noShiftedDz, float& noShiftedDphi, float& noShiftedDphiChange,float xLower,float yLower,float zLower,float rtLower,float xUpper,float yUpper,float zUpper,float rtUpper);

    ALPAKA_FN_ACC bool runMiniDoubletDefaultAlgoBarrel(struct modules& modulesInGPU, uint16_t& lowerModuleIndex, uint16_t& upperModuleIndex, unsigned int lowerHitIndex, unsigned int upperHitIndex, float& dz, float& dphi, float& dphichange, float& shiftedX, float& shiftedY, float& shiftedZ, float& noShiftedDz, float& noShiftedDphi, float& noShiftedDphiChange,float xLower,float yLower,float zLower,float rtLower,float xUpper,float yUpper,float zUpper,float rtUpper);

    ALPAKA_FN_ACC bool runMiniDoubletDefaultAlgoEndcap(struct modules& modulesInGPU, uint16_t& lowerModuleIndex, uint16_t& upperModuleIndex, unsigned int lowerHitIndex, unsigned int upperHitIndex, float& drt, float& dphi, float& dphichange, float& shiftedX, float& shiftedY, float& shiftedZ, float& noShiftedDz, float& noShiftedDphi, float& noShiftedDphiChange,float xLower,float yLower,float zLower,float rtLower,float xUpper,float yUpper,float zUpper,float rtUpper);

    void printMD(struct miniDoublets& mdsInGPU, struct hits& hitsInGPU, SDL::modules& modulesInGPU, unsigned int mdIndex);

    __global__ void createMiniDoubletsInGPUv2(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::objectRanges& rangesInGPU);


}

#endif

