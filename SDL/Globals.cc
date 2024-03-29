#include "Globals.h"

SDL::modulesBuffer<SDL::Dev>* SDL::Globals<SDL::Dev>::modulesBuffers = nullptr;
SDL::modulesBuffer<SDL::Dev> const* SDL::Globals<SDL::Dev>::modulesBuffersES = nullptr;
std::shared_ptr<SDL::pixelMap> SDL::Globals<SDL::Dev>::pixelMapping = nullptr;
uint16_t SDL::Globals<SDL::Dev>::nModules;
uint16_t SDL::Globals<SDL::Dev>::nLowerModules;

SDL::EndcapGeometry<SDL::Dev>* SDL::Globals<SDL::Dev>::endcapGeometry = new SDL::EndcapGeometry<SDL::Dev>();

SDL::TiltedGeometry<SDL::Dev> SDL::Globals<SDL::Dev>::tiltedGeometry;
SDL::ModuleConnectionMap<SDL::Dev> SDL::Globals<SDL::Dev>::moduleConnectionMap;

void SDL::Globals<SDL::Dev>::freeEndcap() {
  if (endcapGeometry != nullptr) {
    delete endcapGeometry;
    endcapGeometry = nullptr;
  }
}

// Temporary solution to the global variables. Should be freed with shared_ptr.
void SDL::Globals<SDL::Dev>::freeModules() {
  if (modulesBuffers != nullptr) {
    delete modulesBuffers;
    modulesBuffers = nullptr;
  }
}
