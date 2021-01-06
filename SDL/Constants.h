#ifndef Constants_h
#define Constants_h

#define PTCUT 1.0f

namespace SDL
{
    namespace CPU
    {
        namespace Constant
        {
            // Luminous region fiducial region. +- 15 cm
            const float deltaZLum = 15.f;

            const float kRinv1GeVf = (2.99792458e-3 * 3.8);

            const float k2Rinv1GeVf = (2.99792458e-3 * 3.8) / 2.;

            const float ptCut = PTCUT;

            const float sinAlphaMax = 0.95;  // arcsin (0.95) ~= 1.25 rad ~= 72 degrees (1/5th circle)

            const float pixelPSZpitch = 0.15;

            const float strip2SZpitch = 5.0;

        }
    }
}


#endif
