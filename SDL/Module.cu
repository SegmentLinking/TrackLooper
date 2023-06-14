#include "Module.cuh"

// TODO: Change this to remove it from global scope.
std::map <unsigned int, uint16_t> *SDL::detIdToIndex;
std::map <unsigned int, float> *SDL::module_x;
std::map <unsigned int, float> *SDL::module_y;
std::map <unsigned int, float> *SDL::module_z;
std::map <unsigned int, unsigned int> *SDL::module_type; // 23 : Ph2PSP, 24 : Ph2PSS, 25 : Ph2SS
// https://github.com/cms-sw/cmssw/blob/5e809e8e0a625578aa265dc4b128a93830cb5429/Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h#L29