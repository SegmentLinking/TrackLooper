#ifndef trkCore_h
#define trkCore_h

#include "trktree.h"
#include "TCanvas.h"
#include "TSystem.h"
#include "Module.h"
#include "AnalysisConfig.h"
#include "SDL/ModuleConnectionMap.h"
#include "SDLMath.h"
#include "SDL/Event.cuh"
#include <cppitertools/enumerate.hpp>
#include <cppitertools/zip.hpp>
#include <numeric>
#include <filesystem>

// --------------------- ======================== ---------------------

void loadMaps();
bool goodEvent();
float runMiniDoublet(SDL::Event* event, int evt);
float runSegment(SDL::Event* event);
float runT4(SDL::Event* event);
float runT4x(SDL::Event* event);
float runpT4(SDL::Event* event);
float runT3(SDL::Event* event);
float runTrackCandidate(SDL::Event* event);
float runTrackCandidateTest_v2(SDL::Event* event);
float runQuintuplet(SDL::Event* event);
float runPixelQuintuplet(SDL::Event* event);
float runPixelLineSegment(SDL::Event* event);
float runpT3(SDL::Event* event);

// --------------------- ======================== ---------------------

std::vector<int> matchedSimTrkIdxs(std::vector<unsigned int> hitidxs, std::vector<unsigned int> hittypes, bool verbose=false);
std::vector<int> matchedSimTrkIdxs(std::vector<int> hitidxs, std::vector<int> hittypes, bool verbose=false);
int getDenomSimTrkType(int isimtrk);

// --------------------- ======================== ---------------------

float drfracSimHitConsistentWithHelix(int isimtrk, int isimhitidx);
float drfracSimHitConsistentWithHelix(SDLMath::Helix& helix, int isimhitidx);
float distxySimHitConsistentWithHelix(int isimtrk, int isimhitidx);
float distxySimHitConsistentWithHelix(SDLMath::Helix& helix, int isimhitidx);
TVector3 calculateR3FromPCA(const TVector3& p3, const float dxy, const float dz);

// --------------------- ======================== ---------------------

void addInputsToLineSegmentTrackingPreLoad(std::vector<std::vector<float>> &out_trkX,
                                           std::vector<std::vector<float>> &out_trkY,
                                           std::vector<std::vector<float>> &out_trkZ,
                                           std::vector<std::vector<unsigned int>> &out_hitId,
                                           std::vector<std::vector<unsigned int>> &out_hitIdxs,
                                           std::vector<std::vector<unsigned int>> &out_hitIndices_vec0,
                                           std::vector<std::vector<unsigned int>> &out_hitIndices_vec1,
                                           std::vector<std::vector<unsigned int>> &out_hitIndices_vec2,
                                           std::vector<std::vector<unsigned int>> &out_hitIndices_vec3,
                                           std::vector<std::vector<float>> &out_deltaPhi_vec,
                                           std::vector<std::vector<float>> &out_ptIn_vec,
                                           std::vector<std::vector<float>> &out_ptErr_vec,
                                           std::vector<std::vector<float>> &out_px_vec,
                                           std::vector<std::vector<float>> &out_py_vec,
                                           std::vector<std::vector<float>> &out_pz_vec,
                                           std::vector<std::vector<float>> &out_eta_vec,
                                           std::vector<std::vector<float>> &out_etaErr_vec,
                                           std::vector<std::vector<float>> &out_phi_vec,
                                           std::vector<std::vector<int>> &out_charge_vec,
                                           std::vector<std::vector<unsigned int>> &out_seedIdx_vec,
                                           std::vector<std::vector<int>> &out_superbin_vec,
                                           std::vector<std::vector<int8_t>> &out_pixelType_vec,
                                           std::vector<std::vector<short>> &out_isQuad_vec);

float addInputsToEventPreLoad(SDL::Event *event,
                              bool useOMP,
                              std::vector<float> trkX,
                              std::vector<float> trkY,
                              std::vector<float> trkZ,
                              std::vector<unsigned int> hitId,
                              std::vector<unsigned int> hitIdxs,
                              std::vector<unsigned int> hitIndices_vec0,
                              std::vector<unsigned int> hitIndices_vec1,
                              std::vector<unsigned int> hitIndices_vec2,
                              std::vector<unsigned int> hitIndices_vec3,
                              std::vector<float> deltaPhi_vec,
                              std::vector<float> ptIn_vec,
                              std::vector<float> ptErr_vec,
                              std::vector<float> px_vec,
                              std::vector<float> py_vec,
                              std::vector<float> pz_vec,
                              std::vector<float> eta_vec,
                              std::vector<float> etaErr_vec,
                              std::vector<float> phi_vec,
                              std::vector<int> charge_vec,
                              std::vector<unsigned int> seedIdx_vec,
                              std::vector<int> superbin_vec,
                              std::vector<int8_t> pixelType_vec,
                              std::vector<short> isQuad_vec);

void printTimingInformation(std::vector<std::vector<float>>& timing_information, float fullTime, float fullavg);

// --------------------- ======================== ---------------------

TString get_absolute_path_after_check_file_exists(const std::string name);
void writeMetaData();

// --------------------- ======================== ---------------------

// DEPRECATED FUNCTION
float addInputsToLineSegmentTrackingUsingExplicitMemory(SDL::Event &event);
float addInputsToLineSegmentTracking(SDL::Event &event, bool useOMP);

#endif
