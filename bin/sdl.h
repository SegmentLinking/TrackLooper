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

#include "trktree.h"
#include "rooutil.h"
#include "cxxopts.h"

#include "SDL/Event.cuh" // SDL::Event
#include "SDL/Module.cuh" // SDL::Module
#include "SDL/EndcapGeometry.h" // SDL::EndcapGeometr
#include "SDL/ModuleConnectionMap.h" // SDL::ModuleConnectionMap
#include "SDL/Event.cuh"
//#include "SDL/MathUtil.h"

// Efficiency study modules
#include "Study.h"
#include "constants.h"
#include "AnalysisConfig.h"
#include "trkCore.h"
#include "WriteSDLNtuplev2.h"
#include "AnalysisInterface/EventForAnalysisInterface.h"
#include "write_sdl_ntuple.h"

#include "TSystem.h"

// Main code
void run_sdl();
// Write metadata
void writeMetaData();

#endif
