#ifndef MathUtil_h
#define MathUtil_h

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

#include <iostream>
#include <cmath>

#include "Hit.cuh"

namespace SDL
{
    class Hit;
}

namespace SDL
{
    // functions for math related operations
    namespace MathUtil
    {

        CUDA_HOSTDEV float Phi_mpi_pi(float phi);
        CUDA_HOSTDEV float ATan2(float y, float x);
        float ptEstimateFromDeltaPhiChangeAndRt(float dphiChange, float rt);
        float ptEstimateFromRadius(float radius);
        float dphiEstimateFromPtAndRt(float pt, float rt);
        SDL::Hit getCenterFromThreePoints(SDL::Hit& hit1, SDL::Hit& hit2, SDL::Hit& hit3);
        float angleCorr(float dr, float pt, float angle);

    }
}

#endif
