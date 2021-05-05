#ifndef PixelTriplet_cuh
#define PixelTriplet_cuh

#ifdef __CUDACC__
#define CUDA_HOSTDEV  __host__ __device__
#define CUDA_DEV __device__
#else
#define CUDA_HOSTDEV
#define CUDA_DEV
#endif

#include "Constants.h"
#include "EndcapGeometry.h"
#include "TiltedGeometry.h"
#include "Tracklet.cuh"
#include "Triplet.cuh"
#include "Segment.cuh"
#include "MiniDoublet.cuh"
#include "Module.cuh"
#include "Hit.cuh"
#include "PrintUtil.h"
#include "Quintuplet.cuh"

namespace SDL
{
    struct pixelTriplets //one pixel segment, one outer tracker triplet!
    {
        unsigned int* pixelSegmentIndices;         
        unsigned int* tripletIndices;
        unsigned int* nPixelTriplets; //size 1

        float* pixelRadius;
        float* pixelRadiusError;
        float* tripletRadius;

        pixelTriplets();
        ~pixelTriplets();
        void freeMemory();
        void freeMemoryCache();
    };

    void createPixelTripletsInUnifiedMemory(struct pixelTriplets& pixelTripletsInGPU, unsigned int maxPixelTriplets);

    void createPixelTripletsInExplicitMemory(struct pixelTriplets& pixelTripletsinGPU, unsigned int maxPixelTriplets);

    CUDA_DEV void addPixelTripletToMemory(struct pixelTriplets& pixelTripletsInGPU, unsigned int pixelSegmentIndex, unsigned int tripletIndex, float pixelRadius, float pixelRadiuError, float tripletRadius, unsigned int pixelTripletIndex);


    CUDA_DEV bool runPixelTripletDefaultAlgo(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, struct segments& segmentsInGPU, struct triplets& tripletsInGPU, unsigned int& pixelSegmentIndex, unsigned int tripletIndex, float& pixelRadius, float& pixelRadiusError, float& tripletRadius);

    CUDA_DEV bool passRadiusCriterion(struct modules& modulesInGPU, float& pixelRadius, float& pixelRadiusError, float& tripletRadius, unsigned int lowerModuleIndex, unsigned int middleModuleIndex, unsigned int upperModuleIndex);

    CUDA_DEV bool passRadiusCriterionBBB(float& pixelRadius, float& pixelRadiusError, float& tripletRadius);

    CUDA_DEV bool passRadiusCriterionBBE(float& pixelRadius, float& pixelRadiusError, float& tripletRadius);

    CUDA_DEV bool passRadiusCriterionBEE(float& pixelRadius, float& pixelRadiusError, float& tripletRadius);

    CUDA_DEV bool passRadiusCriterionEEE(float& pixelRadius, float& pixelRadiusError, float& tripletRadius);

}
#endif
