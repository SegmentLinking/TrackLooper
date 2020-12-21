#ifndef write_sdl_ntuple_h
#define write_sdl_ntuple_h

#include <iostream>
#include "AnalysisConfig.h"
#include "SDL/Event.cuh"
#include "AnalysisInterface/EventForAnalysisInterface.h"
#include <cppitertools/itertools.hpp>
#include "Study.h"
#include "WriteSDLNtuplev2.h"
#include "trkCore.h"

void write_sdl_ntuple();
void createOutputBranches();
void fillOutputBranches(SDL::Event& event);
void printTimingInformation(std::vector<std::vector<float>> timing_information);

#endif
