#ifndef Kernels_cuh
#define Kernels_cuh

#include <vector>
#include <list>
#include <map>
#include <cassert>
#include <stdlib.h>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <memory>
#include <algorithm>
#include <cuda_runtime.h>
#include <omp.h>
#include <chrono>
#include "Module.cuh"
#include "Hit.cuh"
#include "MiniDoublet.cuh"
#include "Segment.cuh"
#include "Triplet.cuh"
#include "TrackCandidate.cuh"
#include "Quintuplet.cuh"
#include "PixelTriplet.cuh"
#include "Constants.cuh"

#include "cuda_profiler_api.h"
#ifdef __CUDACC__
#define CUDA_G __global__
#else
#define CUDA_G
#endif



__device__ float scorepT3(struct SDL::modules& modulesInGPU,struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, unsigned int innerPix,unsigned int outerTrip,float pt, float pz);
__global__ void removeDupPixelTripletsInGPUFromMap(struct SDL::pixelTriplets& pixelTripletsInGPU, bool secondPass);
__global__ void removeDupPixelQuintupletsInGPUFromMap(struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, bool secondPass);
__global__ void markUsedObjects(struct SDL::modules& modulesInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, struct SDL::quintuplets& quintupletsInGPU);

__global__ void checkHitspLS(struct SDL::modules& modulesInGPU, struct SDL::objectRanges& rangesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::hits& hitsInGPU, bool secondpass);


__device__ void scoreT5(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU,unsigned int innerTrip,unsigned int outerTrip, int layer, float* scores);
__global__ void removeDupQuintupletsInGPUAfterBuild(struct SDL::modules& modulesInGPU, struct SDL::quintuplets& quintupletsInGPU, struct SDL::objectRanges& rangesInGPU);
__global__ void removeDupQuintupletsInGPUBeforeTC(struct SDL::quintuplets& quintupletsInGPU, struct SDL::objectRanges& rangesInGPU);

__global__ void testMiniDoublets(struct SDL::miniDoublets& mdsInGPU);
__global__ void createMiniDoubletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::objectRanges& rangesInGPU);

    CUDA_DEV void rmPixelQuintupletToMemory(struct pixelQuintuplets& pixelQuintupletsInGPU, unsigned int pixelQuintupletIndex); 
    CUDA_DEV void rmPixelTripletToMemory(struct pixelTriplets& pixelTripletsInGPU, unsigned int pixelTripletIndex);
    CUDA_DEV void rmQuintupletToMemory(struct SDL::quintuplets& quintupletsInGPU, unsigned int quintupletIndex);   
    CUDA_DEV void rmPixelSegmentFromMemory(struct segments& segmentsInGPU, unsigned int pixelSegmentArrayIndex);  
#endif
