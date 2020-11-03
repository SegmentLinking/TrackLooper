#ifndef GeometryUtil_h
#define GeometryUtil_h

#include <iostream>
#include <cmath>

#include "PrintUtil.h"
#include "Hit.cuh"
#include "Module.cuh"

namespace SDL
{
    class Hit;
    class Module;
}

namespace SDL
{
    // functions for math related operations
    namespace GeometryUtil
    {

        SDL::Hit stripEdgeHit(const SDL::Hit& recohit, bool isup=true);
        SDL::Hit stripHighEdgeHit(const SDL::Hit& recohit);
        SDL::Hit stripLowEdgeHit(const SDL::Hit& recohit);

    }
}

#endif
