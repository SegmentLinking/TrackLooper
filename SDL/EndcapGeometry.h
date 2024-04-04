#ifndef EndcapGeometry_h
#define EndcapGeometry_h

#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>

#include "Constants.h"

namespace SDL {
  template <typename TDev>
  class EndcapGeometry {};
  template <>
  class EndcapGeometry<SDL::Dev> {
  private:
    std::map<unsigned int, float> dxdy_slope_;     // dx/dy slope
    std::map<unsigned int, float> centroid_phis_;  // centroid phi

  public:
    Buf<SDL::Dev, unsigned int> geoMapDetId_buf;
    Buf<SDL::Dev, float> geoMapPhi_buf;

    unsigned int nEndCapMap;

    EndcapGeometry(SDL::Dev const& devAccIn, unsigned int sizef = endcap_size);
    EndcapGeometry(SDL::Dev const& devAccIn, SDL::QueueAcc& queue, std::string filename, unsigned int sizef = endcap_size);
    ~EndcapGeometry() = default;

    template <typename TQueue>
    void load(TQueue& queue, std::string);

    template <typename TQueue>
    void fillGeoMapArraysExplicit(TQueue& queue);
    float getdxdy_slope(unsigned int detid);
  };
}  // namespace SDL

#endif
