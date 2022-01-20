#ifndef Quintuplet_cuh
#define Quintuplet_cuh

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
#include "Triplet.cuh"
#include "Tracklet.cuh"

namespace SDL
{
    struct quintuplets
    {
        unsigned int* tripletIndices;
        unsigned int* lowerModuleIndices;
        unsigned int* nQuintuplets;
        FPX* innerRadius;
        FPX* outerRadius;
        FPX* pt;
        FPX* eta;
        FPX* phi;
        FPX* score_rphisum;
        int* layer;
        bool* isDup;
        bool* partOfPT5;

        float* regressionRadius;
        float* regressionG;
        float* regressionF;

        //for track extensions
        unsigned int* logicalLayers;
        unsigned int* hitIndices;

#ifdef CUT_VALUE_DEBUG
        float* innerRadiusMin;
        float* innerRadiusMax;
        float* outerRadiusMin;
        float* outerRadiusMax;
        float* bridgeRadiusMin;
        float* bridgeRadiusMax;
        float* innerRadiusMin2S;
        float* innerRadiusMax2S;
        float* bridgeRadius;
        float* bridgeRadiusMin2S;
        float* bridgeRadiusMax2S;
        float* outerRadiusMin2S;
        float* outerRadiusMax2S;
        float* chiSquared;
        float* nonAnchorChiSquared;
#endif

        quintuplets();
        ~quintuplets();
        void freeMemory(cudaStream_t stream);
        void freeMemoryCache();
    };

    void createQuintupletsInUnifiedMemory(struct SDL::quintuplets& quintupletsInGPU, const unsigned int& maxQuintuplets, const unsigned int& nLowerModules, const unsigned int& nEligibleModules,cudaStream_t stream);

    void createQuintupletsInExplicitMemory(struct SDL::quintuplets& quintupletsInGPU, const unsigned int& maxQuintuplets, const unsigned int& nLowerModules, const unsigned int& nEligibleModules,cudaStream_t stream);

    void createEligibleModulesListForQuintuplets(struct modules& modulesInGPU, struct triplets& tripletsInGPU, unsigned int& nEligibleModules, unsigned int* indicesOfEligibleModules, unsigned int maxQuintuplets, unsigned int& maxTriplets,cudaStream_t stream, struct objectRanges& rangesInGPU);

  CUDA_DEV void rmQuintupletToMemory(struct SDL::quintuplets& quintupletsInGPU, unsigned int quintupletIndex);

#ifdef CUT_VALUE_DEBUG
    CUDA_DEV void addQuintupletToMemory(struct SDL::triplets& tripletsInGPU, struct SDL::quintuplets& quintupletsInGPU, unsigned int innerTripletIndex, unsigned int outerTripletIndex, unsigned int lowerModule1, unsigned int lowerModule2, unsigned int lowerModule3, unsigned int lowerModule4, unsigned int lowerModule5, float innerRadius, float innerRadiusMin, float innerRadiusMax, float outerRadius, float outerRadiusMin, float outerRadiusMax, float bridgeRadius, float bridgeRadiusMin, float bridgeRadiusMax,
        float innerRadiusMin2S, float innerRadiusMax2S, float bridgeRadiusMin2S, float bridgeRadiusMax2S, float outerRadiusMin2S, float outerRadiusMax2S, float regressionG, float regressionF, float regressionRadius, float chiSquared, float nonAnchorChiSquared, float pt, float eta, float phi, float scores, int layer, unsigned int quintupletIndex);

#else
    CUDA_DEV void addQuintupletToMemory(struct SDL::triplets& tripletsInGPU, struct SDL::quintuplets& quintupletsInGPU, unsigned int innerTripletIndex, unsigned int outerTripletIndex, unsigned int lowerModule1, unsigned int lowerModule2, unsigned int lowerModule3, unsigned int lowerModule4, unsigned int lowerModule5, float innerRadius, float outerRadius, float regressionG, float regressionF, float regressionRadius, float pt, float eta, float phi, float scores, int layer, unsigned int
            quintupletIndex);
#endif
    
    CUDA_DEV bool runQuintupletDefaultAlgo(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct triplets& tripletsInGPU, unsigned int lowerModuleIndex1, unsigned int lowerMoudleIndex2, unsigned int lowerModuleIndex3, unsigned int lowerModuleIndex4, unsigned int lowerModuleIndex5, unsigned int innerTripletIndex, unsigned int outerTripletIndex, float& innerRadius, float& innerRadiusMax, float& innerRadiusMin, float& outerRadius,
        float& outerRadiusMax, float& outerRadiusMin, float& bridgeRadius, float& bridgeRadiusMin, float& bridgeRadiusMax, float& innerRadiusMin2S, float& innerRadiusMax2S, float& bridgeRadiusMin2S, float& bridgeRadiusMax2S, float& outerRadiusMin2S, float& outerRadiusMax2S, float& regressionG, float& regressionF, float& regressionRadius, float& chiSquared, float& nonAnchorChiSquared);

    CUDA_DEV bool passT5RZConstraint(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, unsigned int anchorHitIndex1, unsigned int anchorHitIndex2, unsigned int anchorHitIndex3, unsigned int anchorHitIndex4, unsigned int anchorHitIndex5, unsigned int lowerModuleIndex1, unsigned int lowerModuleIndex2, unsigned int lowerModuleIndex3, unsigned int lowerModuleIndex4, unsigned int lowerModuleIndex5);

    CUDA_DEV bool T5HasCommonMiniDoublet(struct triplets& tripletsInGPU, struct segments& segmentsInGPU, unsigned int innerTripletIndex, unsigned int outerTripletIndex);

    CUDA_DEV float computeRadiusFromThreeAnchorHits(float x1, float y1, float x2, float y2, float x3, float y3, float& g, float& f);

    CUDA_DEV void computeErrorInRadius(float* x1Vec, float* y1Vec, float* x2Vec, float* y2Vec, float* x3Vec, float* y3Vec, float& gError, float& fError);

    CUDA_DEV bool matchRadiiBBBBB(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax);

    CUDA_DEV bool matchRadiiBBBBE(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S, float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax);

    CUDA_DEV bool matchRadiiBBBEE(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S,float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax);

    CUDA_DEV bool matchRadiiBBBEE12378(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S,float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax);

    CUDA_DEV bool matchRadiiBBBEE23478(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S,float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax);

CUDA_DEV bool matchRadiiBBBEE34578(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S,float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax);


    CUDA_DEV bool matchRadiiBBEEE(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S,float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax);

    CUDA_DEV bool matchRadiiBEEEE(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S,float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax);

    CUDA_DEV bool matchRadiiEEEEE(const float& innerRadius, const float& bridgeRadius, const float& outerRadius, const float& innerRadiusMin2S, const float& innerRadiusMax2S, const float& bridgeRadiusMin2S, const float& bridgeRadiusMax2S, const float& outerRadiusMin2S, const float& outerRadiusMax2S,float& innerRadiusMin, float& innerRadiusMax, float& bridgeRadiusMin, float& bridgeRadiusMax, float& outerRadiusMin, float& outerRadiusMax);

    CUDA_DEV bool checkIntervalOverlap(const float& firstMin, const float& firstMax, const float& secondMin, const float& secondMax);

    CUDA_DEV float computeRadiusUsingRegression(int nPoints, float* xs, float* ys, float* delta1, float* delta2, float* slopes, bool* isFlat, float& g, float& f, float* sigmas, float& chiSquared);

    CUDA_DEV void computeSigmasForRegression(SDL::modules& modulesInGPU, const unsigned int* lowerModuleIndices, float* delta1, float* delta2, float* slopes, bool* isFlat, int nPoints = 5, bool anchorHits = true);

    CUDA_DEV bool passChiSquaredConstraint(struct modules& modulesInGPU, unsigned int lowerModuleIndex1, unsigned int lowerModuleIndex2, unsigned int lowerModuleIndex3, unsigned int lowerModuleIndex4, unsigned int lowerModuleIndex5, float& chiSquared);

    CUDA_DEV float computeChiSquared(int nPoints, float* xs, float* ys, float* delta1, float* delta2, float* slopes, bool* isFlat, float g, float f, float radius);

}
#endif
