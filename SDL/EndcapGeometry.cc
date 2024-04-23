#include "EndcapGeometry.h"

SDL::EndcapGeometry<SDL::Dev>::EndcapGeometry(SDL::Dev const& devAccIn, unsigned int sizef)
    : geoMapDetId_buf(allocBufWrapper<unsigned int>(devAccIn, sizef)),
      geoMapPhi_buf(allocBufWrapper<float>(devAccIn, sizef)) {}

SDL::EndcapGeometry<SDL::Dev>::EndcapGeometry(SDL::Dev const& devAccIn,
                                              SDL::QueueAcc& queue,
                                              std::string filename,
                                              unsigned int sizef)
    : geoMapDetId_buf(allocBufWrapper<unsigned int>(devAccIn, sizef)),
      geoMapPhi_buf(allocBufWrapper<float>(devAccIn, sizef)) {
  load(queue, filename);
}

void SDL::EndcapGeometry<SDL::Dev>::load(SDL::QueueAcc& queue, std::string filename) {
  dxdy_slope_.clear();
  centroid_phis_.clear();

  std::ifstream ifile(filename, std::ios::binary);
  if (!ifile.is_open()) {
    throw std::runtime_error("Unable to open file: " + filename);
  }

  while (!ifile.eof()) {
    unsigned int detid;
    float dxdy_slope, centroid_phi;

    // Read the detid, dxdy_slope, and centroid_phi from binary file
    ifile.read(reinterpret_cast<char*>(&detid), sizeof(detid));
    ifile.read(reinterpret_cast<char*>(&dxdy_slope), sizeof(dxdy_slope));
    ifile.read(reinterpret_cast<char*>(&centroid_phi), sizeof(centroid_phi));

    if (ifile) {
      dxdy_slope_[detid] = dxdy_slope;
      centroid_phis_[detid] = centroid_phi;
    } else {
      // End of file or read failed
      if (!ifile.eof()) {
        throw std::runtime_error("Failed to read Endcap Geometry binary data.");
      }
    }
  }

  fillGeoMapArraysExplicit(queue);
}

void SDL::EndcapGeometry<SDL::Dev>::fillGeoMapArraysExplicit(SDL::QueueAcc& queue) {
  unsigned int phi_size = centroid_phis_.size();

  // Temporary check for endcap initialization.
  if (phi_size != endcap_size) {
    std::cerr << "\nError: phi_size and endcap_size are not equal.\n";
    std::cerr << "phi_size: " << phi_size << ", endcap_size: " << endcap_size << "\n";
    std::cerr << "Please change endcap_size in Constants.h to make it equal to phi_size.\n";
    throw std::runtime_error("Mismatched sizes");
  }

  // Allocate buffers on host
  SDL::DevHost const& devHost = cms::alpakatools::host();
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
  alpaka::memcpy(queue, geoMapPhi_buf, mapPhi_host_buf);
  alpaka::memcpy(queue, geoMapDetId_buf, mapDetId_host_buf);
  alpaka::wait(queue);
}

float SDL::EndcapGeometry<SDL::Dev>::getdxdy_slope(unsigned int detid) { return dxdy_slope_[detid]; }
