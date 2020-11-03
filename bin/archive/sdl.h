#ifndef sdl_h
#define sdl_h

#include <vector>
#include <map>
#include <tuple>

#include "trktree.h"
#include "rooutil.h"
#include "cxxopts.h"

#include "SDL/Event.h" // SDL::Event
#include "SDL/Module.h" // SDL::Module
#include "SDL/PrintUtil.h" // SDL::out
#include "SDL/EndcapGeometry.h" // SDL::EndcapGeometry
#include "SDL/ModuleConnectionMap.h" // SDL::ModuleConnectionMap

// Efficiency study modules
#include "Study.h"

#include "constants.h"

#include "AnalysisConfig.h"

#include "trkCore.h"

#include "src/print_module_centroid.h"
#include "src/build_module_map.h"
#include "src/mtv.h"
#include "src/tracklet.h"
#include "src/algo_eff.h"

#endif
