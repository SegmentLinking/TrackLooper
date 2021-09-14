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

const unsigned int N_MAX_HITS_PER_MODULE = 100;
const unsigned int N_MAX_MD_PER_MODULES = 100;
const unsigned int N_MAX_SEGMENTS_PER_MODULE = 600; //WHY!
const unsigned int MAX_CONNECTED_MODULES = 40;
const unsigned int N_MAX_TRACKLETS_PER_MODULE = 8000;//temporary
const unsigned int N_MAX_TRIPLETS_PER_MODULE = 5000;
const unsigned int N_MAX_TOTAL_TRIPLETS = 200000;
const unsigned int N_MAX_PIXEL_MD_PER_MODULES = 100000;
const unsigned int N_MAX_PIXEL_SEGMENTS_PER_MODULE = 50000;
const unsigned int N_MAX_QUINTUPLETS_PER_MODULE = 5000;
#ifdef FINAL_T3T4
const unsigned int N_MAX_TRACK_CANDIDATES_PER_MODULE = 50000;
const unsigned int N_MAX_PIXEL_TRACKLETS_PER_MODULE = 3000000;
const unsigned int N_MAX_PIXEL_TRACK_CANDIDATES_PER_MODULE = 5000000;
#else
const unsigned int N_MAX_TRACK_CANDIDATES_PER_MODULE = 5000;
const unsigned int N_MAX_PIXEL_TRACKLETS_PER_MODULE = 200000;
const unsigned int N_MAX_PIXEL_TRACK_CANDIDATES_PER_MODULE = 250000;
#endif
const unsigned int N_MAX_PIXEL_TRIPLETS = 250000;
const unsigned int N_MAX_PIXEL_QUINTUPLETS = 1000000;
const unsigned int N_MAX_TRACK_CANDIDATE_EXTENSIONS = 200000;  

__device__ float scorepT3(struct SDL::modules& modulesInGPU,struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, unsigned int innerPix,unsigned int outerTrip,float pt, float pz);
__global__ void removeDupPixelTripletsInGPUFromMap(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::pixelTriplets& pixelTripletsInGPU, struct SDL::triplets& tripletsInGPU, bool secondPass);
__device__ int* checkHitspT3(unsigned int ix, unsigned int jx,struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::pixelTriplets& pixelTripletsInGPU);
__global__ void removeDupPixelQuintupletsInGPUFromMap(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::pixelTriplets& pixelTripletsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, struct SDL::quintuplets& quintupletsInGPU, bool secondPass);
__global__ void markUsedObjects(struct SDL::modules& modulesInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, struct SDL::quintuplets& quintupletsInGPU);

__global__ void checkHitspLS(struct SDL::modules& modulesInGPU,struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::hits& hitsInGPU);


__device__ void scoreT5(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU,unsigned int innerTrip,unsigned int outerTrip, int layer, float* scores);
__global__ void removeDupQuintupletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::quintuplets& quintupletsInGPU, bool secondPass);
__device__ int checkHitsT5(unsigned int hit1, unsigned int hit2,struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU,struct SDL::triplets& tripletsInGPU, struct SDL::quintuplets& quintupletsInGPU);

__global__ void testMiniDoublets(struct SDL::miniDoublets& mdsInGPU);
__global__ void createMiniDoubletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU);

__global__ void createSegmentsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU);

 __global__ void createSegmentsFromInnerLowerModule(struct SDL::modules&modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, unsigned int innerLowerModuleIndex, unsigned int nInnerMDs);

__global__ void addpT2asTrackCandidateInGPU(struct SDL::modules& modulesInGPU,struct SDL::pixelTracklets& pixelTrackletsInGPU,struct SDL::trackCandidates& trackCandidatesInGPU);

__global__ void addpT3asTrackCandidateInGPU(struct SDL::modules& modulesInGPU,struct SDL::pixelTriplets& pixelTripletsInGPU,struct SDL::trackCandidates& trackCandidatesInGPU, struct SDL::segments& segmentsInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU);
__global__ void addpLSasTrackCandidateInGPU(struct SDL::modules& modulesInGPU,struct SDL::pixelTriplets& pixelTripletsInGPU,struct SDL::trackCandidates& trackCandidatesInGPU, struct SDL::segments& segmentsInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, struct SDL::miniDoublets& mdsInGPU,struct SDL::hits& hitsInGPU);

__global__ void addT5asTrackCandidateInGPU(struct SDL::modules& modulesInGPU,struct SDL::quintuplets& quintupletsInGPU,struct SDL::trackCandidates& trackCandidatesInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU,struct SDL::pixelTriplets& pixelTripletsInGPU);

__global__ void addpT5asTrackCandidateInGPU(struct SDL::modules& modulesInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU);
#ifndef NESTED_PARA
#ifdef NEWGRID_Tracklet
__global__ void createTrackletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU, unsigned int *index_gpu);
__global__ void createTrackletsFromTriplets(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::tracklets& trackletsInGPU, unsigned int *threadIdx_gpu, unsigned int * threadIdx_gpu_offset);
#endif
#else
__global__ void createTrackletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU);
__global__ void createTrackletsFromInnerInnerLowerModule(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU, unsigned int innerInnerLowerModuleIndex, unsigned int nInnerSegments, unsigned int innerInnerLowerModuleArrayIndex);

__global__ void createTrackletsFromTriplets(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::tracklets& trackletsInGPU/*, unsigned int *index_gpu*/);
__global__ void createTrackletsFromTripletsP2(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::tracklets& trackletsInGPU/*, unsigned int *index_gpu*/,unsigned int innerInnerLowerModuleArrayIndex, unsigned int nTriplets);
#endif

#ifndef NESTED_PARA
#ifdef NEWGRID_Pixel
__global__ void createPixelTrackletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::pixelTracklets& pixelTrackletsInGPU, unsigned int* threadIdx_gpu, unsigned int* threadIdx_gpu_offset);
__global__ void createPixelTrackletsInGPUFromMap(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::pixelTracklets& pixelTrackletsInGPU, unsigned int* connectedPixelSize, unsigned int* connectedPixelIndex, unsigned int nInnerSegs,unsigned int* seg_pix_gpu, unsigned int* seg_pix_gpu_offset, unsigned int totalSegs);
#endif
#else
__global__ void createPixelTrackletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::pixelTracklets& pixelTrackletsInGPU);


__global__ void createPixelTrackletsFromOuterInnerLowerModule(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::pixelTracklets& pixelTrackletsInGPU, unsigned int outerInnerLowerModuleIndex, unsigned int nInnerSegments, unsigned int nOuterSegments, unsigned int pixelModuleIndex, unsigned int pixelLowerModuleArrayIndex);
#endif
__global__ void createPixelTrackletsFromOuterInnerLowerModulev3(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::pixelTracklets& pixelTrackletsInGPU, unsigned int outerInnerLowerModuleIndex, unsigned int nInnerSegments, unsigned int nOuterSegments, unsigned int pixelModuleIndex, unsigned int pixelLowerModuleArrayIndex);

__global__ void createTrackletsWithAGapInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU);


__global__ void createTrackletsWithAGapFromInnerInnerLowerModule(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU, unsigned int innerInnerLowerModuleIndex, unsigned int nInnerSegments, unsigned int innerInnerLowerModuleArrayIndex);

#ifndef NESTED_PARA
#ifdef NEWGRID_Trips
__global__ void createTripletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, str\
uct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, unsigned int *index_gpu);
#endif
#else
__global__ void createTripletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU);

__global__ void createTripletsFromInnerInnerLowerModule(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, unsigned int innerInnerLowerModuleIndex, unsigned int nInnerSegments, unsigned int innerInnerLowerModuleArrayIndex);
#endif

#ifndef NESTED_PARA
#ifdef NEWGRID_Track
__global__ void createTrackCandidatesInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU, unsigned int *threadIdx_gpu, unsigned int* threadIdx_gpu_offset);

__global__ void createPixelTrackCandidatesInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::pixelTracklets& pixelTrackletsInGPU,struct SDL::tracklets& trackletsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU, unsigned int* threadIdx_gpu, unsigned int *threadIdx_gpu_offset);

#endif
#else
__global__ void createPixelTrackCandidatesInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::pixelTracklets& pixelTrackletsInGPU,struct SDL::tracklets& trackletsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU);

__global__ void createPixelTrackCandidatesFromOuterInnerInnerLowerModule(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU, struct SDL::pixelTracklets& pixelTrackletsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU, unsigned int pixelLowerModuleArrayIndex, unsigned int outerInnerInnerLowerModuleArrayIndex, unsigned int nPixelTracklets, unsigned int nOuterLayerTracklets, unsigned int nOuterLayerTriplets);

__global__ void createTrackCandidatesInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU);

__global__ void createTrackCandidatesFromInnerInnerInnerLowerModule(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU, unsigned int innerInnerInnerLowerModuleArrayIndex, unsigned int nInnerTracklets, unsigned int nInnerTriplets);
#endif

#ifdef NEWGRID_T5
__global__ void createQuintupletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::quintuplets& quintupletsInGPU, unsigned int* threadIdx_gpu, unsigned int* threadIdx_gpu_offset, int nTotalTriplets);

#else
__global__ void createQuintupletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::quintuplets& quintupletsInGPU);
#endif

#ifdef NEWGRID_pT3
__global__ void createPixelTripletsInGPUFromMap(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::pixelTriplets& pixelTripletsInGPU, unsigned int* connectedPixelSize, unsigned int* connectedPixelIndex, unsigned int nPixelSegments, unsigned int* seg_pix_gpu, unsigned int* seg_pix_gpu_offset, unsigned int totalSegs);
#else
__global__ void createPixelTripletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::pixelTriplets& pixelTripletsInGPU);

__global__ void createPixelTripletsFromOuterInnerLowerModule(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::pixelTriplets& pixelTripletsInGPU, unsigned int outerTripletInnerLowerModuleArrayIndex, unsigned int nPixelSegments, unsigned int nOuterTriplets, unsigned int pixelModuleIndex);
#endif

#ifdef NEWGRID_pT5

__global__ void createPixelQuintupletsInGPUFromMap(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::quintuplets& quintupletsInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, unsigned int* connectedPixelSize, unsigned int* connectedPixelIndex, unsigned int nPixelSegments, unsigned int* seg_pix_gpu, unsigned int* seg_pix_gpu_offset, unsigned int totalSegs);

#else
__global__ void createPixelQuintupletsFromFirstModule(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::quintuplets& quintupletsInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, unsigned int nPixelSegments, unsigned int nOuterQuintuplets, unsigned int firstLowerModuleArrayIndex, unsigned int pixelModuleIndex);

__global__ void createPixelQuintupletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::quintuplets& quintupletsInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU);
#endif

 __global__ void createExtendedTracksInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::pixelTriplets& pixelTripletsInGPU, struct SDL::quintuplets& quintupletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU, struct SDL::trackExtensions& trackExtensionsInGPU);

#endif
