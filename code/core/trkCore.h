#ifndef trkCore_h
#define trkCore_h

#include "trktree.h"
#include "TCanvas.h"
#include "TSystem.h"
#include "AnalysisInterface/EventForAnalysisInterface.h"
#include "AnalysisInterface/Module.h"
#include "AnalysisConfig.h"
#include "SDL/ModuleConnectionMap.h"
#include "SDLMath.h"
#include "SDL/Event.cuh"
#include "SDL/Event.h"
#include <cppitertools/enumerate.hpp>
#include <cppitertools/zip.hpp>
#include <numeric>

enum TrackletType {
    BB1BB3 = 0,
    BB2BB4,
    BB3BB5,
    BB1BE3,
    BB2BE4,
    BB3BE5,
    BB1EE3,
    BB2EE4,
    BB3EE5,
    BE1EE3,
    BE2EE4,
    BE3EE5,
    EE1EE3,
    EE2EE4,
};

float simhit_p(unsigned int simhitidx);
float hitAngle(unsigned int simhitidx);
bool isMuonCurlingHit(unsigned int isimtrk, unsigned int ith_hit);
bool hasAll12HitsWithNBarrel(unsigned int isimtrk, int nbarrel);
bool hasAll12HitsWithNBarrelUsingModuleMap(unsigned int isimtrk, int nbarrel, bool usesimhits=false);
bool checkModuleConnectionsAreGood(std::array<std::vector<unsigned int>, 6>& layers_good_paired_modules);
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
float runTrackExtensions(SDL::Event* event);
//float runMiniDoublet(SDL::Event& event, int evt);
//float runSegment(SDL::Event& event);
//float runT4(SDL::Event& event);
//float runT4x(SDL::Event& event);
//float runpT4(SDL::Event& event);
//float runT3(SDL::Event& event);
//float runTrackCandidate(SDL::Event& event);
//float runTrackCandidateTest_v2(SDL::Event& event);
//float runQuintuplet(SDL::Event& event);
//float runPixelQuintuplet(SDL::Event& event);
//float runPixelLineSegment(SDL::Event& event);
//float runpT3(SDL::Event& event);

std::vector<float> getPtBounds();
bool inTimeTrackWithPdgId(int isimtrk, int pdgid);
std::vector<int> matchedSimTrkIdxs(std::vector<int> hitidxs, std::vector<int> hittypes, bool verbose=false);
std::vector<int> matchedSimTrkIdxs(SDL::Segment* sg, bool matchOnlyAnchor=false);
std::vector<int> matchedSimTrkIdxs(SDL::Tracklet& tl);

bool isMTVMatch(unsigned int isimtrk, std::vector<unsigned int> hit_idxs, bool verbose=false);
void loadMaps();
float drfracSimHitConsistentWithHelix(int isimtrk, int isimhitidx);
float drfracSimHitConsistentWithHelix(SDLMath::Helix& helix, int isimhitidx);
float distxySimHitConsistentWithHelix(int isimtrk, int isimhitidx);
float distxySimHitConsistentWithHelix(SDLMath::Helix& helix, int isimhitidx);

float addInputsToLineSegmentTrackingUsingUnifiedMemory(SDL::Event &event);
float addInputsToLineSegmentTrackingUsingExplicitMemory(SDL::Event &event);
float addInputsToLineSegmentTracking(SDL::Event &event, bool useOMP);
void addInputsToLineSegmentTrackingPreLoad(std::vector<std::vector<float>>& out_trkX,std::vector<std::vector<float>>& out_trkY,std::vector<std::vector<float>>& out_trkZ,
std::vector<std::vector<unsigned int>>&    out_hitId,
std::vector<std::vector<unsigned int>>&    out_hitIdxs,
std::vector<std::vector<unsigned int>>&    out_hitIndices_vec0,
std::vector<std::vector<unsigned int>>&    out_hitIndices_vec1,
std::vector<std::vector<unsigned int>>&    out_hitIndices_vec2,
std::vector<std::vector<unsigned int>>&    out_hitIndices_vec3,
std::vector<std::vector<float>>&    out_deltaPhi_vec,
std::vector<std::vector<float>>&    out_ptIn_vec,
std::vector<std::vector<float>>&    out_ptErr_vec,
std::vector<std::vector<float>>&    out_px_vec,
std::vector<std::vector<float>>&    out_py_vec,
std::vector<std::vector<float>>&    out_pz_vec,
std::vector<std::vector<float>>&    out_eta_vec,
std::vector<std::vector<float>>&    out_etaErr_vec,
std::vector<std::vector<float>>&    out_phi_vec,
std::vector<std::vector<int>>&    out_superbin_vec,
std::vector<std::vector<int8_t>>&    out_pixelType_vec,
std::vector<std::vector<short>>&    out_isQuad_vec,
std::vector<unsigned int>&    hitOffset
);
//float addInputsToEventPreLoad(SDL::Event& event, bool useOMP,std::vector<float> trkX,std::vector<float> trkY,std::vector<float> trkZ,
float addInputsToEventPreLoad(SDL::Event* event, bool useOMP,std::vector<float> trkX,std::vector<float> trkY,std::vector<float> trkZ,
std::vector<unsigned int>    hitId,
std::vector<unsigned int>    hitIdxs,
std::vector<unsigned int>    hitIndices_vec0,
std::vector<unsigned int>    hitIndices_vec1,
std::vector<unsigned int>    hitIndices_vec2,
std::vector<unsigned int>    hitIndices_vec3,
std::vector<float>    deltaPhi_vec,
std::vector<float>    ptIn_vec,
std::vector<float>    ptErr_vec,
std::vector<float>    px_vec,
std::vector<float>    py_vec,
std::vector<float>    pz_vec,
std::vector<float>    eta_vec,
std::vector<float>    etaErr_vec,
std::vector<float>    phi_vec,
std::vector<int>    superbin_vec,
std::vector<int8_t>    pixelType_vec,
std::vector<short>   isQuad_vec,
unsigned int hitOffset, unsigned int evt, struct SDL::hits* hitsInGPU_event
);

TVector3 calculateR3FromPCA(const TVector3& p3, const float dxy, const float dz);

float addOuterTrackerHits(SDL::CPU::Event& event);
float addOuterTrackerSimHits(SDL::CPU::Event& event);
float addOuterTrackerSimHitsFromPVOnly(SDL::CPU::Event& event);
float addOuterTrackerSimHitsNotFromPVOnly(SDL::CPU::Event& event);
float addPixelSegments(SDL::CPU::Event& event, int isimtrk=-1);
float runMiniDoublet_on_CPU(SDL::CPU::Event& event);
float runSegment_on_CPU(SDL::CPU::Event& event);
float runT4_on_CPU(SDL::CPU::Event& event);
float runT4x_on_CPU(SDL::CPU::Event& event);
float runpT4_on_CPU(SDL::CPU::Event& event);
float runT3_on_CPU(SDL::CPU::Event& event);
float runTrackCandidate_on_CPU(SDL::CPU::Event& event);
float runT5_on_CPU(SDL::CPU::Event& event);
float runpT3_on_CPU(SDL::CPU::Event& event);

// Printing SDL information
void printHitSummary(SDL::CPU::Event& event);
void printMiniDoubletSummary(SDL::CPU::Event& event);
void printSegmentSummary(SDL::CPU::Event& event);
void printTrackletSummary(SDL::CPU::Event& event);
void printTripletSummary(SDL::CPU::Event& event);
void printTrackCandidateSummary(SDL::CPU::Event& event);

// trk tool
bool isDenomSimTrk(int isimtrk);
bool isDenomOfInterestSimTrk(int isimtrk);
int getDenomSimTrkType(int isimtrk);
int bestSimHitMatch(int irecohit);
int logicalLayer(const SDL::CPU::Module& module);
int isAnchorLayer(const SDL::CPU::Module& module);

#endif
