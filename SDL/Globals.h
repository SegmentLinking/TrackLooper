#ifndef Globals_h
#define Globals_h

#ifdef LST_IS_CMSSW_PACKAGE
#include "RecoTracker/LSTCore/interface/alpaka/Constants.h"
#include "RecoTracker/LSTCore/interface/alpaka/Module.h"
#else
#include "Constants.h"
#include "Module.h"
#endif

#include "TiltedGeometry.h"
#include "EndcapGeometry.h"
#include "ModuleConnectionMap.h"
#include "PixelMap.h"

namespace SDL {
  template <typename>
  struct Globals;
  template <>
  struct Globals<SDL::Dev> {
    static SDL::modulesBuffer<SDL::Dev>* modulesBuffers;
    static SDL::modulesBuffer<SDL::Dev> const* modulesBuffersES;  // not owned const buffers
    static uint16_t nModules;
    static uint16_t nLowerModules;
    static std::shared_ptr<SDL::pixelMap> pixelMapping;

    static EndcapGeometry<SDL::Dev>* endcapGeometry;
    static TiltedGeometry<SDL::Dev> tiltedGeometry;
    static ModuleConnectionMap<SDL::Dev> moduleConnectionMap;

    static void freeEndcap();
    static void freeModules();
  };

}  // namespace SDL

#endif
