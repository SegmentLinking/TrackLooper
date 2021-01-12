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
void printHitMultiplicities(SDL::Event& event);
void printpT4s(SDL::Event& event);
void printpT4s_for_CPU(SDL::CPU::Event& event);
void printMDs(SDL::Event& event);
void printMDs_for_CPU(SDL::CPU::Event& event);
void printLSs(SDL::Event& event);
void printLSs_for_CPU(SDL::CPU::Event& event);
void printpLSs(SDL::Event& event);
void printpLSs_for_CPU(SDL::CPU::Event& event);
void printT3s(SDL::Event& event);
void printT3s_for_CPU(SDL::CPU::Event& event);
void printT4s(SDL::Event& event);
void printT4s_for_CPU(SDL::CPU::Event& event);
void printTCs(SDL::Event& event);
void printTCs_for_CPU(SDL::CPU::Event& event);

#endif
