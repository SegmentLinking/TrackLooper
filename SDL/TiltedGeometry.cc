#include "TiltedGeometry.h"

SDL::TiltedGeometry<SDL::Dev>::TiltedGeometry(std::string filename) { load(filename); }

void SDL::TiltedGeometry<SDL::Dev>::load(std::string filename) {
  drdzs_.clear();
  dxdys_.clear();

  std::ifstream ifile(filename);

  std::string line;
  while (std::getline(ifile, line)) {
    unsigned int detid;
    float drdz;
    float dxdy;

    std::stringstream ss(line);

    if (ss >> detid >> drdz >> dxdy) {
      drdzs_[detid] = drdz;
      dxdys_[detid] = dxdy;
    } else {
      throw std::runtime_error("Failed to parse line: " + line);
    }
  }
}

float SDL::TiltedGeometry<SDL::Dev>::getDrDz(unsigned int detid) {
  if (drdzs_.find(detid) != drdzs_.end()) {
    return drdzs_[detid];
  } else {
    return 0;
  }
}

float SDL::TiltedGeometry<SDL::Dev>::getDxDy(unsigned int detid) {
  if (dxdys_.find(detid) != dxdys_.end()) {
    return dxdys_[detid];
  } else {
    return 0;
  }
}
