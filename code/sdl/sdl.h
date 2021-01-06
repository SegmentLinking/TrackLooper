#ifndef sdl_h
#define sdl_h

#include <vector>
#include <map>
#include <tuple>
#include <string>
#include <fstream>
#include <streambuf>

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
#include "write_sdl_ntuple.h"
#include "AnalysisInterface/EventForAnalysisInterface.h"

#include "TSystem.h"

#endif
