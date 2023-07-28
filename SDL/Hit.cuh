#ifndef Hit_cuh
#define Hit_cuh

#ifdef __CUDACC__
#define CUDA_HOSTDEV  __host__ __device__
#define CUDA_DEV __device__
#define CUDA_CONST_VAR __device__
#define CUDA_G __global__
#else
#define CUDA_HOSTDEV
#define CUDA_CONST_VAR
#define CUDA_DEV
#define CUDA_G
#endif

#include <iostream>
#include <cmath>
#include <vector>

//#include "PrintUtil.h"
#include "Module.cuh"
#include "allocate.h"

namespace SDL
{
    struct hits
    {
        unsigned int *nHits; //single number
//        unsigned int *n2SHits;
        float *xs;
        float *ys;
        float *zs;

        uint16_t* moduleIndices;
        unsigned int* idxs;
        unsigned int* detid;
        
        float *rts;
        float* phis;
        float* etas;

//        int *edge2SMap;
        float *highEdgeXs;
        float *highEdgeYs;
        float *lowEdgeXs;
        float *lowEdgeYs;

        int* hitRanges;
        int* hitRangesLower;
        int* hitRangesUpper;
        int8_t* hitRangesnLower;
        int8_t* hitRangesnUpper;
        
        hits();
        void freeMemory();
        //void freeMemory(cudaStream_t stream);
        void freeMemoryCache();
        ~hits();

    };

    void createHitsInExplicitMemory(struct hits& hitsInGPU, int nModules, unsigned int maxHits,cudaStream_t stream,unsigned int evtnum);
    CUDA_G void addHitToMemoryKernel(struct hits& hitsInGPU,struct modules& modulesInGPU,const float* x,const float* y, const float* z,const uint16_t* moduleIndex,const float* phis, const int loopsize);
    //CUDA_G void checkHits(struct hits& hitsInGPU, const int loopsize);
    void addHitToMemory(struct hits& hitsInCPU,struct modules& modulesInGPU,float x, float y, float z, unsigned int detId, unsigned int idxInNtuple,cudaStream_t stream,struct objectRanges& rangesInGPU);
    CUDA_G void addHitToMemoryGPU(struct hits& hitsInCPU,struct modules& modulesInGPU,float x, float y, float z, unsigned int detId, unsigned int idxInNtuple,unsigned int moduleIndex, float phis,struct objectRanges& rangesInGPU);
    
    CUDA_HOSTDEV inline float eta(float x, float y, float z) {
      float r3 = std::sqrt( x*x + y*y + z*z );
      float rt = std::sqrt( x*x + y*y );
      float eta = ((z > 0) - ( z < 0)) * acosh( r3 / rt );
      return eta;
    }
    CUDA_HOSTDEV inline float phi_mpi_pi(float x) {
      if (std::isnan(x))
      {
        //printf("phi_mpi_pi() function called with NaN\n");                                                
          return x;
      }

      if (fabsf(x) <= float(M_PI))
        return x;
      constexpr float o2pi = 1.f / (2.f * float(M_PI));
      float n = std::round(x * o2pi);
      return x - n * float(2.f * float(M_PI));
    }
    CUDA_HOSTDEV inline float phi(float x, float y) {
      return phi_mpi_pi(float(M_PI) + atan2f(-y,-x));
    }
    CUDA_HOSTDEV inline float deltaPhi(float x1, float y1, float x2, float y2) {
      float phi1 = phi(x1,y1);
      float phi2 = phi(x2,y2);
      return phi_mpi_pi((phi2 - phi1));
    }
    CUDA_HOSTDEV inline float deltaPhiChange(float x1, float y1, float x2, float y2) {
      return deltaPhi(x1, y1, x2-x1, y2-y1);
    }
    void getEdgeHits(unsigned int detId,float x, float y, float& xhigh, float& yhigh, float& xlow, float& ylow);
    CUDA_DEV void getEdgeHitsK(float phi,float x, float y, float& xhigh, float& yhigh, float& xlow, float& ylow);

    void printHit(struct hits& hitsInGPU, struct modules& modulesInGPU, unsigned int hitIndex);
}
#endif

