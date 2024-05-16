#ifndef EndcapGeometry_h
#define EndcapGeometry_h

#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>

#ifdef LST_IS_CMSSW_PACKAGE
#include "RecoTracker/LSTCore/interface/alpaka/Constants.h"
#else
#include "Constants.h"
#endif

#include "HeterogeneousCore/AlpakaInterface/interface/host.h"

namespace SDL {

  // Only the full one contains alpaka buffers
  template <typename TDev, bool full>
  class EndcapGeometry;

  template <>
  class EndcapGeometry<DevHost, false> {

  public:
    std::map<unsigned int, float> dxdy_slope_;     // dx/dy slope
    std::map<unsigned int, float> centroid_phis_;  // centroid phi

    EndcapGeometry() = default;
    ~EndcapGeometry() = default;

    void load(std::string);
    float getdxdy_slope(unsigned int detid);
  };

  template <>
  class EndcapGeometry<Dev, true> {
  private:
    std::map<unsigned int, float> dxdy_slope_;     // dx/dy slope
    std::map<unsigned int, float> centroid_phis_;  // centroid phi

  public:
    Buf<Dev, unsigned int> geoMapDetId_buf;
    Buf<Dev, float> geoMapPhi_buf;

    unsigned int nEndCapMap;

    EndcapGeometry(Dev const& devAccIn, QueueAcc& queue, SDL::EndcapGeometry<DevHost, false> const& endcapGeometryIn);
    ~EndcapGeometry() = default;

    void fillGeoMapArraysExplicit(QueueAcc& queue);
    float getdxdy_slope(unsigned int detid);
  };
}  // namespace SDL

#endif
