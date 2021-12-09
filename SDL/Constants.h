#ifndef Constants_h
#define Constants_h

#define PTCUT 1.0f
#include <cuda_fp16.h>

namespace SDL
{
    namespace CPU
    {
        namespace Constant
        {
            // Luminous region fiducial region. +- 15 cm
            const float deltaZLum = 15.f;

            const float kRinv1GeVf = (2.99792458e-3f * 3.8f);

            const float k2Rinv1GeVf = 0.5f*(2.99792458e-3f * 3.8f);

            const float ptCut = PTCUT;

            const float sinAlphaMax = 0.95f;  // arcsin (0.95) ~= 1.25 rad ~= 72 degrees (1/5th circle)

            const float pixelPSZpitch = 0.15f;

            const float strip2SZpitch = 5.0f;

        }
    }
}


#endif
