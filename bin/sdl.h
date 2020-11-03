#ifndef sdl_h
#define sdl_h

#include <vector>
#include <map>
#include <tuple>

#include "trktree.h"
#include "rooutil.h"
#include "cxxopts.h"

#include "SDL/Event.cuh" // SDL::Event
#include "SDL/Module.cuh" // SDL::Module
#include "SDL/EndcapGeometry.h" // SDL::EndcapGeometr
#include "SDL/ModuleConnectionMap.h" // SDL::ModuleConnectionMap

// Efficiency study modules
#include "Study.h"

#include "constants.h"

#include "AnalysisConfig.h"

#include "trkCore.h"

//#include "src/print_module_centroid.h"
//#include "src/build_module_map.h"
//#include "src/mtv.h"
//#include "src/tracklet.h"
//#include "src/algo_eff.h"
#include "src/write_sdl_ntuple.h"
#include "AnalysisInterface/EventForAnalysisInterface.h"

#endif
