#ifndef SDL_MathUtil_h
#define SDL_MathUtil_h

#include <iostream>
#include <cmath>

#include "Hit.h"

namespace SDL
{
    namespace CPU
    {
        class Hit;
    }
}

namespace SDL
{
    namespace CPU
    {
        // functions for math related operations
        namespace MathUtil
        {

            float Phi_mpi_pi(float phi);
            float ATan2(float y, float x);
            float ptEstimateFromDeltaPhiChangeAndRt(float dphiChange, float rt);
            float ptEstimateFromRadius(float radius);
            float dphiEstimateFromPtAndRt(float pt, float rt);
            SDL::CPU::Hit getCenterFromThreePoints(SDL::CPU::Hit& hit1, SDL::CPU::Hit& hit2, SDL::CPU::Hit& hit3);
            float angleCorr(float dr, float pt, float angle);

        }
    }
}

#endif
