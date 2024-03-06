#ifndef EndcapGeometry_h
#define EndcapGeometry_h

#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "Constants.h"

namespace SDL {
  class EndcapGeometry {
  private:
    std::map<unsigned int, float> sls_;  // lower slope
    std::map<unsigned int, float> centroid_phis_;  // centroid phi

  public:
    Buf<SDL::Dev, unsigned int> geoMapDetId_buf;
    Buf<SDL::Dev, float> geoMapPhi_buf;

    unsigned int nEndCapMap;

    EndcapGeometry(unsigned int sizef = endcap_size);
    EndcapGeometry(std::string filename, unsigned int sizef = endcap_size);
    ~EndcapGeometry();

    void load(std::string);

    void fillGeoMapArraysExplicit();
    void CreateGeoMapArraysExplicit();
    float getAverageR2(unsigned int detid);
    float getYInterceptLower(unsigned int detid);
    float getSlopeLower(unsigned int detid);
  };
  void freeEndcap();
  extern EndcapGeometry* endcapGeometry;
}  // namespace SDL

#endif
