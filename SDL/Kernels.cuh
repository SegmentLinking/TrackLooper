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
#include "PixelQuintuplet.cuh"
#include "TrackExtensions.cuh"

#include "cuda_profiler_api.h"
#ifdef __CUDACC__
#define CUDA_G __global__
#else
#define CUDA_G
#endif

const unsigned int MAX_BLOCKS = 80;
const unsigned int MAX_CONNECTED_MODULES = 40;
const unsigned int N_MAX_MD_PER_MODULES = 89;
const unsigned int N_MAX_SEGMENTS_PER_MODULE = 537; //Also used in (pixel)trackletDefualtAlgo in header files-> change there as well (hard coded) 
const unsigned int N_MAX_PIXEL_MD_PER_MODULES = 100000;
const unsigned int N_MAX_PIXEL_SEGMENTS_PER_MODULE = 50000;

const unsigned int N_MAX_TRIPLETS_PER_MODULE = 1170;
const unsigned int N_MAX_INWARD_TRIPLETS_PER_MODULE = 1170;

const unsigned int N_MAX_QUINTUPLETS_PER_MODULE = 513;
const unsigned int N_MAX_PIXEL_TRIPLETS = 5000;
const unsigned int N_MAX_PIXEL_QUINTUPLETS = 15000;
const unsigned int N_MAX_TOTAL_TRIPLETS = 200000;

const unsigned int N_MAX_TRACK_CANDIDATES = 1000;
const unsigned int N_MAX_PIXEL_TRACK_CANDIDATES = 4000;

const unsigned int N_MAX_TRACK_CANDIDATE_EXTENSIONS = 200000;  
const unsigned int N_MAX_TRACK_EXTENSIONS_PER_TC = 30;
const unsigned int N_MAX_T3T3_TRACK_EXTENSIONS = 40000;

//old numbers
//const unsigned int N_MAX_QUINTUPLETS_PER_MODULE = 5000;
//const unsigned int N_MAX_PIXEL_TRIPLETS = 250000;
//const unsigned int N_MAX_PIXEL_QUINTUPLETS = 1000000;
//const unsigned int N_MAX_TRACK_CANDIDATES = 10000;
//const unsigned int N_MAX_PIXEL_TRACK_CANDIDATES = 400000;

//const unsigned int N_MAX_TRIPLETS_PER_MODULE = 1000;
//const unsigned int N_MAX_TOTAL_TRIPLETS = 100000;
//const unsigned int N_MAX_QUINTUPLETS_PER_MODULE = 2000;
//const unsigned int N_MAX_PIXEL_TRIPLETS = 2500;
//const unsigned int N_MAX_PIXEL_QUINTUPLETS = 10000;
//const unsigned int N_MAX_TRACK_CANDIDATES = 10000;
//const unsigned int N_MAX_PIXEL_TRACK_CANDIDATES = 10000;
//const unsigned int N_MAX_TOTAL_TRIPLETS = 130000;


__device__ float scorepT3(struct SDL::modules& modulesInGPU,struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, unsigned int innerPix,unsigned int outerTrip,float pt, float pz);
__global__ void removeDupPixelTripletsInGPUFromMap(struct SDL::pixelTriplets& pixelTripletsInGPU, bool secondPass);
//__device__ int* checkHitspT3(unsigned int ix, unsigned int jx,struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::pixelTriplets& pixelTripletsInGPU);
__global__ void removeDupPixelQuintupletsInGPUFromMap(struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, bool secondPass);
__global__ void markUsedObjects(struct SDL::modules& modulesInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, struct SDL::quintuplets& quintupletsInGPU);

__global__ void checkHitspLS(struct SDL::modules& modulesInGPU, struct SDL::objectRanges& rangesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::hits& hitsInGPU, bool secondpass);


__device__ void scoreT5(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU,unsigned int innerTrip,unsigned int outerTrip, int layer, float* scores);
__global__ void removeDupQuintupletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::quintuplets& quintupletsInGPU, bool secondPass, struct SDL::objectRanges& rangesInGPU);
__global__ void removeDupQuintupletsInGPUv2(struct SDL::modules& modulesInGPU, struct SDL::quintuplets& quintupletsInGPU, bool secondPass, struct SDL::objectRanges& rangesInGPU);
//__device__ int checkHitsT5(unsigned int hit1, unsigned int hit2,struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU,struct SDL::triplets& tripletsInGPU, struct SDL::quintuplets& quintupletsInGPU);

__global__ void testMiniDoublets(struct SDL::miniDoublets& mdsInGPU);
__global__ void createMiniDoubletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::objectRanges& rangesInGPU);

__global__ void createSegmentsInGPU(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::objectRanges& rangesInGPU);

 __global__ void createSegmentsFromInnerLowerModule(struct SDL::modules&modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, unsigned int innerLowerModuleIndex, unsigned int nInnerMDs);

__global__ void addpT3asTrackCandidateInGPU(struct SDL::modules& modulesInGPU, struct SDL::objectRanges& rangesInGPU, struct SDL::pixelTriplets& pixelTripletsInGPU,struct SDL::trackCandidates& trackCandidatesInGPU, struct SDL::segments& segmentsInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU);

__global__ void addpLSasTrackCandidateInGPU(struct SDL::modules& modulesInGPU, struct SDL::objectRanges& rangesInGPU, struct SDL::pixelTriplets& pixelTripletsInGPU,struct SDL::trackCandidates& trackCandidatesInGPU, struct SDL::segments& segmentsInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, struct SDL::miniDoublets& mdsInGPU,struct SDL::hits& hitsInGPU, struct SDL::quintuplets& quintupletsInGPU);

__global__ void addT5asTrackCandidateInGPU(struct SDL::modules& modulesInGPU, struct SDL::quintuplets& quintupletsInGPU,struct SDL::trackCandidates& trackCandidatesInGPU,struct SDL::pixelQuintuplets& pixelQuintupletsInGPU,struct SDL::pixelTriplets& pixelTripletsInGPU, struct SDL::objectRanges& rangesInGPU);

__global__ void addpT5asTrackCandidateInGPU(struct SDL::modules& modulesInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU,struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU,struct SDL::quintuplets& quintupletsInGPU);

__global__ void createTripletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::triplets& tripletsInwardInGPU, struct SDL::objectRanges& rangesInGPU, uint16_t* index_gpu, uint16_t nonZeroModules);

__global__ void createTrackCandidatesInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU, unsigned int *threadIdx_gpu, unsigned int* threadIdx_gpu_offset);

__global__ void createQuintupletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::triplets& tripletsInwardInGPU, struct SDL::quintuplets& quintupletsInGPU, unsigned int* threadIdx_gpu, unsigned int* threadIdx_gpu_offset, int nTotalTriplets,struct SDL::objectRanges& rangesInGPU);

__global__ void createPixelTripletsInGPUFromMap(struct SDL::modules& modulesInGPU, struct SDL::objectRanges& rangesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::pixelTriplets& pixelTripletsInGPU, unsigned int* connectedPixelSize, unsigned int* connectedPixelIndex, unsigned int nPixelSegments, unsigned int* seg_pix_gpu, unsigned int* seg_pix_gpu_offset, unsigned int totalSegs);

 __global__ void createExtendedTracksInGPU(struct SDL::modules& modulesInGPU, struct SDL::objectRanges& rangesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::pixelTriplets& pixelTripletsInGPU, struct SDL::quintuplets& quintupletsInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU, struct SDL::trackExtensions& trackExtensionsInGPU);

__global__ void createT3T3ExtendedTracksInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::quintuplets& quintupletsInGPU, struct SDL::pixelTriplets& pixelTripletsInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU, struct SDL::trackExtensions& trackExtensionsInGPU, unsigned int nTrackCandidates);

__global__ void cleanDuplicateExtendedTracks(struct SDL::trackExtensions& trackExtensionsInGPU, unsigned int nTrackCandidates);

__global__ void createPixelQuintupletsInGPUFromMap(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::quintuplets& quintupletsInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, unsigned int* connectedPixelSize, unsigned int* connectedPixelIndex, unsigned int nPixelSegments, unsigned int* seg_pix_gpu, unsigned int* seg_pix_gpu_offset, unsigned int totalSegs,struct SDL::objectRanges& rangesInGPU);
 
#endif
