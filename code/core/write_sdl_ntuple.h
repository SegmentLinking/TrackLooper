#ifndef write_sdl_ntuple_h
#define write_sdl_ntuple_h

#include <iostream>
#include <tuple>
#include <cppitertools/enumerate.hpp>

#include "MathUtil.h"
#include "SDL/Event.cuh"

#include "constants.h"
#include "AnalysisConfig.h"
#include "trkCore.h"
#include "AccessHelper.h"

// Common
void createOutputBranches();
void createOutputBranches_v2();
void createGnnNtupleBranches();

void fillOutputBranches(SDL::Event* event);
void fillOutputBranches_v2(SDL::Event* event);
void fillGnnNtupleBranches(SDL::Event* event);
void fillGnnNtupleMiniDoublet(SDL::Event* event, unsigned int MD);
std::tuple<int, float, float, float, int, vector<int>> parseTrackCandidate(SDL::Event* event, unsigned int);
std::tuple<float, float, float, vector<unsigned int>, vector<unsigned int>> parsepT5(SDL::Event* event, unsigned int);
std::tuple<float, float, float, vector<unsigned int>, vector<unsigned int>> parsepT3(SDL::Event* event, unsigned int);
std::tuple<float, float, float, vector<unsigned int>, vector<unsigned int>> parseT5(SDL::Event* event, unsigned int);
std::tuple<float, float, float, vector<unsigned int>, vector<unsigned int>> parsepLS(SDL::Event* event, unsigned int);

// Print multiplicities
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

// Print anomalous multiplicities
void debugPrintOutlierMultiplicities(SDL::Event* event);

#endif
