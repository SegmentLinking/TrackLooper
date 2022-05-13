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
#include "Tracklet.cuh"
#include "PixelTracklet.cuh"
#include "Triplet.cuh"
#include "TrackCandidate.cuh"
#include "Quintuplet.cuh"
#include "PixelTriplet.cuh"
#include "TrackExtensions.cuh"
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
__global__ void removeDupQuintupletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::quintuplets& quintupletsInGPU, bool secondPass, struct SDL::objectRanges& rangesInGPU);
__global__ void removeDupQuintupletsInGPUv2(struct SDL::quintuplets& quintupletsInGPU, struct SDL::objectRanges& rangesInGPU);

__global__ void testMiniDoublets(struct SDL::miniDoublets& mdsInGPU);
__global__ void createMiniDoubletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::objectRanges& rangesInGPU);

__global__ void createT3T3ExtendedTracksInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::quintuplets& quintupletsInGPU, struct SDL::pixelTriplets& pixelTripletsInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU, struct SDL::trackExtensions& trackExtensionsInGPU, unsigned int nTrackCandidates);

__global__ void cleanDuplicateExtendedTracks(struct SDL::trackExtensions& trackExtensionsInGPU, unsigned int nTrackCandidates);


    CUDA_DEV void rmPixelQuintupletToMemory(struct pixelQuintuplets& pixelQuintupletsInGPU, unsigned int pixelQuintupletIndex); 
    CUDA_DEV void rmPixelTripletToMemory(struct pixelTriplets& pixelTripletsInGPU, unsigned int pixelTripletIndex);
    CUDA_DEV void rmQuintupletToMemory(struct SDL::quintuplets& quintupletsInGPU, unsigned int quintupletIndex);   
    CUDA_DEV void rmPixelSegmentFromMemory(struct segments& segmentsInGPU, unsigned int pixelSegmentArrayIndex);  
#endif
