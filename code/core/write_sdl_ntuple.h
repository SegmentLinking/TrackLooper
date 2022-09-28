#ifndef write_sdl_ntuple_h
#define write_sdl_ntuple_h

#include <iostream>
#include <cppitertools/enumerate.hpp>

#include "MathUtil.h"
#include "SDL/Event.cuh"

// Efficiency study modules
//#include "Study.h"
#include "constants.h"
#include "AnalysisConfig.h"
#include "trkCore.h"

// Common
void createOutputBranches();
void createLowerLevelOutputBranches();
void createQuintupletCutValueBranches();
void createPixelQuintupletCutValueBranches();
void createQuadrupletCutValueBranches();
void createTripletCutValueBranches();
void createSegmentCutValueBranches();
void createMiniDoubletCutValueBranches();
void createOccupancyBranches();
void createPixelQuadrupletCutValueBranches();
void createPixelTripletCutValueBranches();
void createTrackExtensionCutValueBranches();
void createPrimitiveBranches();
void createPrimitiveBranches_v1();
void createPrimitiveBranches_v2();
// Common
void fillSimTrackOutputBranches();
// GPU

void fillOutputBranches(SDL::Event* event);
void get_output_CMSSW(SDL::Event* event, std::vector<float>& tc_pt, std::vector<float>& tc_eta, std::vector<float>& tc_phi);
void fillTrackCandidateOutputBranches(SDL::Event* event);
void fillTrackExtensionOutputBranches(SDL::Event* event);
void fillPureTrackExtensionOutputBranches(SDL::Event* event);
void fillT3T3TrackExtensionOutputBranches(SDL::Event* event);
void fillLowerLevelOutputBranches(SDL::Event* event);
void fillQuadrupletOutputBranches(SDL::Event* event);
void fillTripletOutputBranches(SDL::Event* event);
void fillPixelTripletOutputBranches(SDL::Event* event);
void fillQuintupletOutputBranches(SDL::Event* event);
void fillPixelQuintupletOutputBranches(SDL::Event* event);
void fillPixelLineSegmentOutputBranches(SDL::Event* event);
void fillOccupancyBranches(SDL::Event* event);
void fillPixelQuadrupletOutputBranches(SDL::Event* event);
void fillSegmentBranches(SDL::Event* event);
void fillMiniDoubletBranches(SDL::Event* event);


// Timing
void printTimingInformation(std::vector<std::vector<float>>& timing_information, float fullTime, float fullavg);

// Print multiplicities
void printQuadrupletMultiplicities(SDL::Event* event);
void printMiniDoubletMultiplicities(SDL::Event* event);
void printHitMultiplicities(SDL::Event* event);

// Print objects (GPU)
void printAllObjects(SDL::Event* event);
void printpT4s(SDL::Event* event);
void printMDs(SDL::Event* event);
void printLSs(SDL::Event* event);
void printpLSs(SDL::Event* event);
void printT3s(SDL::Event* event);
void printT4s(SDL::Event* event);
void printTCs(SDL::Event* event);

//// Print objects (CPU)
//void printAllObjects_for_CPU(SDL::CPU::Event& event);
//void printpT4s_for_CPU(SDL::CPU::Event& event);
//void printMDs_for_CPU(SDL::CPU::Event& event);
//void printLSs_for_CPU(SDL::CPU::Event& event);
//void printpLSs_for_CPU(SDL::CPU::Event& event);
//void printT3s_for_CPU(SDL::CPU::Event& event);
//void printT4s_for_CPU(SDL::CPU::Event& event);
//void printTCs_for_CPU(SDL::CPU::Event& event);

// Print anomalous multiplicities
void debugPrintOutlierMultiplicities(SDL::Event* event);

#endif
