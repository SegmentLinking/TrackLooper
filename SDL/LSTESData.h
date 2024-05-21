#ifndef LSTESData_H
#define LSTESData_H

#ifdef LST_IS_CMSSW_PACKAGE
#include "RecoTracker/LSTCore/interface/alpaka/Constants.h"
#else
#include "Constants.h"
#endif

#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"

#include <filesystem>
#include <memory>

namespace SDL {

  struct pixelMap;

  template <typename>
  class TiltedGeometry;

  template <typename>
  class ModuleConnectionMap;
  using MapPLStoLayer = std::array<std::array<ModuleConnectionMap<Dev>, 4>, 3>;

  template <typename>
  struct modulesBuffer;

  template <typename>
  class EndcapGeometryHost;

  template <typename>
  class EndcapGeometry;

  template <typename TDev>
  struct LSTESHostData;

  // FIXME: This shouldn't be a templated struct
  template <>
  struct LSTESHostData<Dev> {
    const std::shared_ptr<const MapPLStoLayer> mapPLStoLayer;
    const std::shared_ptr<const EndcapGeometryHost<Dev>> endcapGeometry;
    const std::shared_ptr<const TiltedGeometry<Dev>> tiltedGeometry;
    const std::shared_ptr<const ModuleConnectionMap<Dev>> moduleConnectionMap;

    LSTESHostData(std::shared_ptr<MapPLStoLayer> mapPLStoLayerIn,
                  std::shared_ptr<EndcapGeometryHost<Dev>> endcapGeometryIn,
                  std::shared_ptr<TiltedGeometry<Dev>> tiltedGeometryIn,
                  std::shared_ptr<ModuleConnectionMap<Dev>> moduleConnectionMapIn)
        : mapPLStoLayer(std::const_pointer_cast<const MapPLStoLayer>(mapPLStoLayerIn)),
          endcapGeometry(std::const_pointer_cast<const EndcapGeometryHost<Dev>>(endcapGeometryIn)),
          tiltedGeometry(std::const_pointer_cast<const TiltedGeometry<Dev>>(tiltedGeometryIn)),
          moduleConnectionMap(std::const_pointer_cast<const ModuleConnectionMap<Dev>>(moduleConnectionMapIn)) {}
  };

  template <typename TDev>
  struct LSTESDeviceData {
    const uint16_t nModules;
    const uint16_t nLowerModules;
    const unsigned int nPixels;
    const std::shared_ptr<const modulesBuffer<TDev>> modulesBuffers;
    const std::shared_ptr<const EndcapGeometry<TDev>> endcapGeometry;
    const std::shared_ptr<const pixelMap> pixelMapping;

    LSTESDeviceData(uint16_t nModulesIn,
                    uint16_t nLowerModulesIn,
                    unsigned int nPixelsIn,
                    std::shared_ptr<modulesBuffer<TDev>> modulesBuffersIn,
                    std::shared_ptr<EndcapGeometry<TDev>> endcapGeometryIn,
                    std::shared_ptr<pixelMap> pixelMappingIn)
        : nModules(nModulesIn),
          nLowerModules(nLowerModulesIn),
          nPixels(nPixelsIn),
          modulesBuffers(std::const_pointer_cast<const modulesBuffer<TDev>>(modulesBuffersIn)),
          endcapGeometry(std::const_pointer_cast<const EndcapGeometry<TDev>>(endcapGeometryIn)),
          pixelMapping(std::const_pointer_cast<const pixelMap>(pixelMappingIn)) {}
  };

  std::unique_ptr<LSTESHostData<Dev>> loadAndFillESHost();
  std::unique_ptr<LSTESDeviceData<Dev>> loadAndFillESDevice(SDL::QueueAcc& queue, const LSTESHostData<Dev>* hostData);

}  // namespace SDL

namespace cms::alpakatools {
  template <>
  struct CopyToDevice<SDL::LSTESHostData<SDL::Dev>> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, SDL::LSTESHostData<SDL::Dev> const& hostData) {
      return std::make_unique<SDL::LSTESHostData<SDL::Dev>>(hostData);
    }
  };
} // namespace cms::alpakatools

#endif
