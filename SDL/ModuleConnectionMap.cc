#include "ModuleConnectionMap.h"

SDL::ModuleConnectionMap<SDL::Dev>::ModuleConnectionMap() {}

SDL::ModuleConnectionMap<SDL::Dev>::ModuleConnectionMap(std::string filename) { load(filename); }

SDL::ModuleConnectionMap<SDL::Dev>::~ModuleConnectionMap() {}

void SDL::ModuleConnectionMap<SDL::Dev>::load(std::string filename) {
  moduleConnections_.clear();

  std::ifstream ifile;
  ifile.open(filename.c_str());
  std::string line;

  while (std::getline(ifile, line)) {
    unsigned int detid;
    int number_of_connections;
    std::vector<unsigned int> connected_detids;
    unsigned int connected_detid;

    std::stringstream ss(line);

    ss >> detid >> number_of_connections;

    for (int ii = 0; ii < number_of_connections; ++ii) {
      ss >> connected_detid;
      connected_detids.push_back(connected_detid);
    }

    moduleConnections_[detid] = connected_detids;
  }
}

void SDL::ModuleConnectionMap<SDL::Dev>::add(std::string filename) {
  std::ifstream ifile;
  ifile.open(filename.c_str());
  std::string line;

  while (std::getline(ifile, line)) {
    unsigned int detid;
    int number_of_connections;
    std::vector<unsigned int> connected_detids;
    unsigned int connected_detid;

    std::stringstream ss(line);

    ss >> detid >> number_of_connections;

    for (int ii = 0; ii < number_of_connections; ++ii) {
      ss >> connected_detid;
      connected_detids.push_back(connected_detid);
    }

    // Concatenate
    moduleConnections_[detid].insert(moduleConnections_[detid].end(), connected_detids.begin(), connected_detids.end());

    // Sort
    std::sort(moduleConnections_[detid].begin(), moduleConnections_[detid].end());

    // Unique
    moduleConnections_[detid].erase(std::unique(moduleConnections_[detid].begin(), moduleConnections_[detid].end()),
                                    moduleConnections_[detid].end());
  }
}

void SDL::ModuleConnectionMap<SDL::Dev>::print() {
  std::cout << "Printing ModuleConnectionMap" << std::endl;
  for (auto& pair : moduleConnections_) {
    unsigned int detid = pair.first;
    std::vector<unsigned int> connected_detids = pair.second;
    std::cout << " detid: " << detid << std::endl;
    for (auto& connected_detid : connected_detids) {
      std::cout << " connected_detid: " << connected_detid << std::endl;
    }
  }
}

const std::vector<unsigned int>& SDL::ModuleConnectionMap<SDL::Dev>::getConnectedModuleDetIds(unsigned int detid) const {
  static const std::vector<unsigned int> dummy;
  auto const mList = moduleConnections_.find(detid);
  return mList != moduleConnections_.end() ? mList->second : dummy;
}
int SDL::ModuleConnectionMap<SDL::Dev>::size() const { return moduleConnections_.size(); }
