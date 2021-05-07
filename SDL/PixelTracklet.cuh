#ifndef Pixel_Tracklet_cuh
#define Pixel_Tracklet_cuh

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
#include "Tracklet.cuh"

#endif
