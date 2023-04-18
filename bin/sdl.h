#ifndef sdl_h
#define sdl_h

#include <vector>
#include <map>
#include <tuple>
#include <string>
#include <fstream>
#include <streambuf>
#include <iostream>
#include <cppitertools/enumerate.hpp>
#include <unistd.h>

#include "trktree.h"
#include "rooutil.h"
#include "cxxopts.h"

#include "SDL/Event.cuh" // SDL::Event
#include "SDL/Module.cuh" // SDL::Module
#include "SDL/EndcapGeometry.cuh" // SDL::EndcapGeometr
#include "SDL/ModuleConnectionMap.h" // SDL::ModuleConnectionMap
#include "SDL/Event.cuh"

// Efficiency study modules
#include "AnalysisConfig.h"
#include "trkCore.h"
#include "write_sdl_ntuple.h"

#include "TSystem.h"

// Main code
void run_sdl();

#endif
