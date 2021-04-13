#ifndef write_sdl_ntuple_h
#define write_sdl_ntuple_h

#include <iostream>
#include <cppitertools/enumerate.hpp>

#include "SDL/MathUtil.h"
#include "SDL/Event.cuh"

// Efficiency study modules
#include "Study.h"
#include "constants.h"
#include "AnalysisConfig.h"
#include "trkCore.h"
#include "WriteSDLNtuplev2.h"
#include "AnalysisInterface/EventForAnalysisInterface.h"

// Common
void createOutputBranches();
void createLowerLevelOutputBranches();
void createQuintupletCutValueBranches();
void createQuadrupletCutValueBranches();
void createTripletCutValueBranches();
void createSegmentCutValueBranches();
void createMiniDoubletCutValueBranches();
void createOccupancyBranches();
void createPixelQuadrupletCutValueBranches();
// Common
void fillSimTrackOutputBranches();
// GPU
void fillOutputBranches(SDL::Event& event);
void fillTrackCandidateOutputBranches(SDL::Event& event);
void fillLowerLevelOutputBranches(SDL::Event& event);
void fillQuadrupletOutputBranches(SDL::Event& event);
void fillTripletOutputBranches(SDL::Event& event);
void fillQuintupletOutputBranches(SDL::Event& event);
void fillPixelLineSegmentOutputBranches(SDL::Event& event);
void fillOccupancyBranches(SDL::Event& event);
void fillPixelQuadrupletOutputBranches(SDL::Event& event);
// CPU
void fillOutputBranches_for_CPU(SDL::CPU::Event& event);
void fillTrackCandidateOutputBranches_for_CPU(SDL::CPU::Event& event);
void fillLowerLevelOutputBranches_for_CPU(SDL::CPU::Event& event);
void fillQuadrupletOutputBranches_for_CPU(SDL::CPU::Event& event);
void fillTripletOutputBranches_for_CPU(SDL::CPU::Event& event);
void fillPixelQuadrupletOutputBranches_for_CPU(SDL::CPU::Event& event);

// Timing
void printTimingInformation(std::vector<std::vector<float>>& timing_information);

// Print multiplicities
void printQuadrupletMultiplicities(SDL::Event& event);
void printMiniDoubletMultiplicities(SDL::Event& event);
void printHitMultiplicities(SDL::Event& event);

// Print objects (GPU)
void printAllObjects(SDL::Event& event);
void printpT4s(SDL::Event& event);
void printMDs(SDL::Event& event);
void printLSs(SDL::Event& event);
void printpLSs(SDL::Event& event);
void printT3s(SDL::Event& event);
void printT4s(SDL::Event& event);
void printTCs(SDL::Event& event);

// Print objects (CPU)
void printAllObjects_for_CPU(SDL::CPU::Event& event);
void printpT4s_for_CPU(SDL::CPU::Event& event);
void printMDs_for_CPU(SDL::CPU::Event& event);
void printLSs_for_CPU(SDL::CPU::Event& event);
void printpLSs_for_CPU(SDL::CPU::Event& event);
void printT3s_for_CPU(SDL::CPU::Event& event);
void printT4s_for_CPU(SDL::CPU::Event& event);
void printTCs_for_CPU(SDL::CPU::Event& event);

// Print anomalous multiplicities
void debugPrintOutlierMultiplicities(SDL::Event& event);

#endif
