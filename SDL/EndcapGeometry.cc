#include "EndcapGeometry.h"

SDL::EndcapGeometry<SDL::Dev>::EndcapGeometry(unsigned int sizef)
    : geoMapDetId_buf(allocBufWrapper<unsigned int>(devAcc, sizef)),
      geoMapPhi_buf(allocBufWrapper<float>(devAcc, sizef)) {}

SDL::EndcapGeometry<SDL::Dev>::EndcapGeometry(std::string filename, unsigned int sizef)
    : geoMapDetId_buf(allocBufWrapper<unsigned int>(devAcc, sizef)),
      geoMapPhi_buf(allocBufWrapper<float>(devAcc, sizef)) {
  load(filename);
}

void SDL::EndcapGeometry<SDL::Dev>::load(std::string filename) {
  dxdy_slope_.clear();
  centroid_phis_.clear();

  std::ifstream ifile(filename);

  std::string line;
  while (std::getline(ifile, line)) {
    std::istringstream ss(line);
    unsigned int detid;
    float dxdy_slope, centroid_phi;

    if (ss >> detid >> dxdy_slope >> centroid_phi) {
      dxdy_slope_[detid] = dxdy_slope;
      centroid_phis_[detid] = centroid_phi;
    } else {
      throw std::runtime_error("Failed to parse line: " + line);
    }
  }

  fillGeoMapArraysExplicit();
}

void SDL::EndcapGeometry<SDL::Dev>::fillGeoMapArraysExplicit() {
  QueueAcc queue(devAcc);

  int phi_size = centroid_phis_.size();

  // Temporary check for endcap initialization.
  if (phi_size != endcap_size) {
    std::cerr << "\nError: phi_size and endcap_size are not equal.\n";
    std::cerr << "phi_size: " << phi_size << ", endcap_size: " << endcap_size << "\n";
    std::cerr << "Please change endcap_size in Constants.h to make it equal to phi_size.\n";
    throw std::runtime_error("Mismatched sizes");
  }

  // Allocate buffers on host
  auto mapPhi_host_buf = allocBufWrapper<float>(devHost, phi_size);
  auto mapDetId_host_buf = allocBufWrapper<unsigned int>(devHost, phi_size);

  // Access the raw pointers of the buffers
  float* mapPhi = alpaka::getPtrNative(mapPhi_host_buf);
  unsigned int* mapDetId = alpaka::getPtrNative(mapDetId_host_buf);

  unsigned int counter = 0;
  for (auto it = centroid_phis_.begin(); it != centroid_phis_.end(); ++it) {
    unsigned int detId = it->first;
    float Phi = it->second;
    mapPhi[counter] = Phi;
    mapDetId[counter] = detId;
    counter++;
  }

  nEndCapMap = counter;

  // Copy data from host to device buffers
  alpaka::memcpy(queue, geoMapPhi_buf, mapPhi_host_buf, phi_size);
  alpaka::memcpy(queue, geoMapDetId_buf, mapDetId_host_buf, phi_size);
  alpaka::wait(queue);
}

float SDL::EndcapGeometry<SDL::Dev>::getdxdy_slope(unsigned int detid) { return dxdy_slope_[detid]; }
