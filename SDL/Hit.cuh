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

    void createHitsInUnifiedMemory(struct hits& hitsInGPU, int nModules, unsigned int maxHits, unsigned int max2SHits,cudaStream_t stream,unsigned int evtnum);
    void createHitsInExplicitMemory(struct hits& hitsInGPU, int nModules, unsigned int maxHits,cudaStream_t stream,unsigned int evtnum);
    CUDA_G void addHitToMemoryKernel(struct hits& hitsInGPU,struct modules& modulesInGPU,const float* x,const float* y, const float* z,const uint16_t* moduleIndex,const float* phis, const int loopsize);
    //CUDA_G void checkHits(struct hits& hitsInGPU, const int loopsize);
    void addHitToMemory(struct hits& hitsInCPU,struct modules& modulesInGPU,float x, float y, float z, unsigned int detId, unsigned int idxInNtuple,cudaStream_t stream,struct objectRanges& rangesInGPU);
    CUDA_G void addHitToMemoryGPU(struct hits& hitsInCPU,struct modules& modulesInGPU,float x, float y, float z, unsigned int detId, unsigned int idxInNtuple,unsigned int moduleIndex, float phis,struct objectRanges& rangesInGPU);
    
    CUDA_HOSTDEV inline float ATan2(float y, float x) {
    //if (x != 0) return  x * (float(-0xf.8eed2p-4) + x * x * float(0x3.1238p-4)); // degree 3 7 bit accuracy//atan2f(y, x);
    if (x != 0){ 
      //float a = y/x;
      //float z = a*a;  
      float test = atan2f(y, x);
      //float test1 = a * (float(-0xf.8eed2p-4) + z * float(0x3.1238p-4)); 
      //float test2 = a * (float(-0xf.ecfc8p-4) + z * (float(0x4.9e79dp-4) + z * float(-0x1.44f924p-4)));
      //float test3 =  a * (float(-0xf.fcc7ap-4) + z * (float(0x5.23886p-4) + z * (float(-0x2.571968p-4) + z * float(0x9.fb05p-8))));
      //float test4 = a * (float(-0xf.ff73ep-4) +
      //        z * (float(0x5.48ee1p-4) +
      //             z * (float(-0x2.e1efe8p-4) + z * (float(0x1.5cce54p-4) + z * float(-0x5.56245p-8)))));
      //float test5 = a * (float(-0xf.ffff4p-4) +
      //        z * (float(0x5.552f9p-4 + z * (float(-0x3.30f728p-4) +
      //                                       z * (float(0x2.39826p-4) +
      //                                            z * (float(-0x1.8a880cp-4) +
      //                                                 z * (float(0xe.484d6p-8) +
      //                                                      z * (float(-0x5.93d5p-8) + z * float(0x1.0875dcp-8)))))))));
      //printf("%f %f %f %f %f %f %f %f\n",y,x,test,test1,test2,test3,test4,test5);
      return test;
    }
    if (y == 0) return  0;
    if (y >  0) return  float(M_PI) / 2.f;
    else        return -float(M_PI) / 2.f;
    }
    CUDA_HOSTDEV inline float phi_mpi_pi(float x) {
    if (isnan(x))
    {
      //printf("phi_mpi_pi() function called with NaN\n");                                                
        return x;
    }

    //while (x >= float(M_PI))
    //    x -= 2.f * float(M_PI);

    //while (x < -float(M_PI))
    //    x += 2.f * float(M_PI);

    //return x;
    if (fabsf(x) <= float(M_PI))
      return x;
    constexpr float o2pi = 1.f / (2.f * float(M_PI));
    float n = std::round(x * o2pi);
    return x - n * float(2.f * float(M_PI));
    }
    CUDA_HOSTDEV inline float phi(float x, float y, float z) {
        return phi_mpi_pi(float(M_PI) + ATan2(-y, -x));
    }
    CUDA_HOSTDEV inline float deltaPhi(float x1, float y1, float z1, float x2, float y2, float z2) {
    float phi1 = phi(x1,y1,z1);
    float phi2 = phi(x2,y2,z2);
    return phi_mpi_pi((phi2 - phi1));
    }
    CUDA_HOSTDEV inline float deltaPhiChange(float x1, float y1, float z1, float x2, float y2, float z2) {
    return deltaPhi(x1,y1,z1,x2-x1, y2-y1, z2-z1);
    }
    void getEdgeHits(unsigned int detId,float x, float y, float& xhigh, float& yhigh, float& xlow, float& ylow);
    CUDA_DEV void getEdgeHitsK(float phi,float x, float y, float& xhigh, float& yhigh, float& xlow, float& ylow);

    void printHit(struct hits& hitsInGPU, struct modules& modulesInGPU, unsigned int hitIndex);
}
#endif

