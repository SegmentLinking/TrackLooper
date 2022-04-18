#ifndef GeometryUtil_h
#define GeometryUtil_h

#include <iostream>
#include <cmath>

#include "PrintUtil.h"
#include "Hit.h"
#include "Module.h"

namespace SDL
{
    namespace CPU
    {
        class Hit;
        class Module;
    }
}

namespace SDL
{
    namespace CPU
    {
        // functions for math related operations
        namespace GeometryUtil
        {

            SDL::CPU::Hit stripEdgeHit(const SDL::CPU::Hit& recohit, bool isup=true);
            SDL::CPU::Hit stripHighEdgeHit(const SDL::CPU::Hit& recohit);
            SDL::CPU::Hit stripLowEdgeHit(const SDL::CPU::Hit& recohit);

        }
    }
}

#endif
