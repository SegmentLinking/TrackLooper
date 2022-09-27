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
#include "SDL/EndcapGeometry.cuh" // SDL::EndcapGeometr
#include "SDL/ModuleConnectionMap.h" // SDL::ModuleConnectionMap
//#include "SDL/MathUtil.h"

// Efficiency study modules
//#include "Study.h"
#include "constants.h"
#include "AnalysisConfig.h"
#include "trkCore.h"
//#include "WriteSDLNtuplev2.h"
//#include "AnalysisInterface/EventForAnalysisInterface.h"
#include "write_sdl_ntuple.h"

#include "TSystem.h"

// Main code
/*#ifdef PORTTOCMSSW
void run_sdl(**** event)
#else
void run_sdl()
#endif
*/
void run_sdl();

// Write metadata
void writeMetaData();

//running functions declare
void pre_running();
void verbose_and_write(SDL::Event* events, int evtnum);
void do_output();
void do_delete(std::vector<SDL::Event*> events, cudaStream_t* streams);

#endif
