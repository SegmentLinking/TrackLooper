#ifndef sdl_h
#define sdl_h

#include <vector>
#include <map>
#include <tuple>
#include <string>
#include <fstream>
#include <streambuf>
#include <iostream>
#include <unistd.h>

#include "trktree.h"
#include "rooutil.h"
#include "cxxopts.h"

#include "SDL/Event.h" // SDL::Event
#include "SDL/Module.h" // SDL::Module
#include "SDL/EndcapGeometry.h" // SDL::EndcapGeometr
#include "SDL/ModuleConnectionMap.h" // SDL::ModuleConnectionMap
#include "SDL/Event.h"

// Efficiency study modules
#include "AnalysisConfig.h"
#include "trkCore.h"
#include "write_sdl_ntuple.h"

#include "TSystem.h"

// Main code
void run_sdl();

#endif
