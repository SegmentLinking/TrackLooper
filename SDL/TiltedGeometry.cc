#include "TiltedGeometry.h"

SDL::TiltedGeometry SDL::globals::tiltedGeometry;

SDL::TiltedGeometry::TiltedGeometry() {}

SDL::TiltedGeometry::TiltedGeometry(std::string filename) { load(filename); }

SDL::TiltedGeometry::~TiltedGeometry() {}

void SDL::TiltedGeometry::load(std::string filename) {
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

float SDL::TiltedGeometry::getDrDz(unsigned int detid) {
  if (drdzs_.find(detid) != drdzs_.end()) {
    return drdzs_[detid];
  } else {
    return 0;
  }
}

float SDL::TiltedGeometry::getDxDy(unsigned int detid) {
  if (dxdys_.find(detid) != dxdys_.end()) {
    return dxdys_[detid];
  } else {
    return 0;
  }
}
