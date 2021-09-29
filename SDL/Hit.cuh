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

namespace SDL
{
    struct hits
    {
        unsigned int *nHits; //single number
//        unsigned int *n2SHits;
        float *xs;
        float *ys;
        float *zs;

        unsigned int* moduleIndices;
        unsigned int* idxs;
        
        float *rts;
        float* phis;
        float* etas;

//        int *edge2SMap;
        float *highEdgeXs;
        float *highEdgeYs;
        float *lowEdgeXs;
        float *lowEdgeYs;
        
        hits();
        void freeMemory();
        void freeMemoryCache();
        ~hits();

    };

    void createHitsInUnifiedMemory(struct hits& hitsInGPU,unsigned int maxHits, unsigned int max2SHits);
    void createHitsInExplicitMemory(struct hits& hitsInGPU, unsigned int maxHits);
    CUDA_G void addHitToMemoryKernel(struct hits& hitsInGPU,struct modules& modulesInGPU,const float* x,const float* y, const float* z,const unsigned int* moduelIndex,const float* phis, const int loopsize);
    //CUDA_G void checkHits(struct hits& hitsInGPU, const int loopsize);
    void addHitToMemory(struct hits& hitsInCPU,struct modules& modulesInGPU,float x, float y, float z, unsigned int detId, unsigned int idxInNtuple);
    CUDA_G void addHitToMemoryGPU(struct hits& hitsInCPU,struct modules& modulesInGPU,float x, float y, float z, unsigned int detId, unsigned int idxInNtuple,unsigned int moduleIndex, float phis);
    
    CUDA_HOSTDEV inline float ATan2(float y, float x) {
    if (x != 0) return  atan2f(y, x);
    if (y == 0) return  0;
    if (y >  0) return  M_PI / 2;
    else        return -M_PI / 2;
    }
    CUDA_HOSTDEV inline float phi_mpi_pi(float x) {
    if (isnan(x))
    {
      //printf("phi_mpi_pi() function called with NaN\n");                                                
        return x;
    }

    while (x >= M_PI)
        x -= 2. * M_PI;

    while (x < -M_PI)
        x += 2. * M_PI;

    return x;
    }
    CUDA_HOSTDEV inline float phi(float x, float y, float z) {
        return phi_mpi_pi(M_PI + ATan2(-y, -x));
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

