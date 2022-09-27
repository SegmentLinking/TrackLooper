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
#ifdef PORTTOCMSSW
void run_sdl(
    std::vector<float> this_trkX,
    std::vector<float> this_trkY,
    std::vector<float> this_trkZ,
    std::vector<unsigned int> this_hitId,
    std::vector<unsigned int> this_hitIdxs,
    std::vector<unsigned int> this_hitIndices_vec0,
    std::vector<unsigned int> this_hitIndices_vec1,
    std::vector<unsigned int> this_hitIndices_vec2,
    std::vector<unsigned int> this_hitIndices_vec3,
    std::vector<float> this_deltaPhi_vec,
    std::vector<float> this_ptIn_vec,
    std::vector<float> this_ptErr_vec,
    std::vector<float> this_px_vec,
    std::vector<float> this_py_vec,
    std::vector<float> this_pz_vec,
    std::vector<float> this_eta_vec,
    std::vector<float> this_etaErr_vec,
    std::vector<float> this_phi_vec,
    std::vector<float> this_charge_vec,
    std::vector<int> this_superbin_vec,
    std::vector<int8_t> this_pixelType_vec,
    std::vector<short> this_isQuad_vec);
#else
void run_sdl();
#endif

// Write metadata
void writeMetaData();

//running functions declare
void pre_running();
void verbose_and_write(SDL::Event* events, int evtnum);
void do_output();
void do_delete(std::vector<SDL::Event*> events, cudaStream_t* streams);

#endif
