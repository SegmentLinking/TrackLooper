#ifndef write_sdl_ntuple_h
#define write_sdl_ntuple_h

#include <iostream>
#include "AnalysisConfig.h"
#include "SDL/Event.cuh"
#include "AnalysisInterface/EventForAnalysisInterface.h"
#include <cppitertools/enumerate.hpp>
#include "Study.h"
#include "WriteSDLNtuplev2.h"
#include "trkCore.h"

void write_sdl_ntuple();
void createOutputBranches();
void fillOutputBranches(SDL::Event& event);
void fillOutputBranches_for_CPU(SDL::CPU::Event& event);
void printTimingInformation(std::vector<std::vector<float>> timing_information);
void printQuadrupletMultiplicities(SDL::Event& event);
void printMiniDoubletMultiplicities(SDL::Event& event);

#endif
