#ifndef AccessHelper_h
#define AccessHelper_h

#include <vector>
#include <tuple>
#include "SDL/Event.cuh"

// ----* pLS *----
std::vector<unsigned int> getPixelHitsFrompLS(SDL::Event* event, unsigned int pLS);
std::vector<unsigned int> getPixelHitIdxsFrompLS(SDL::Event* event, unsigned int pLS);
std::vector<unsigned int> getPixelHitTypesFrompLS(SDL::Event* event, unsigned int pLS);

// ----* MD *----
std::vector<unsigned int> getHitsFromMD(SDL::Event* event, unsigned int MD);

// ----* LS *----
std::vector<unsigned int> getMDsFromLS(SDL::Event* event, unsigned int LS);
std::vector<unsigned int> getHitsFromLS(SDL::Event* event, unsigned int LS);

// ----* T3 *----
std::vector<unsigned int> getLSsFromT3(SDL::Event* event, unsigned int T3);
std::vector<unsigned int> getMDsFromT3(SDL::Event* event, unsigned int T3);
std::vector<unsigned int> getHitsFromT3(SDL::Event* event, unsigned int T3);

// ----* T5 *----
std::vector<unsigned int> getT3sFromT5(SDL::Event* event, unsigned int T5);
std::vector<unsigned int> getLSsFromT5(SDL::Event* event, unsigned int T5);
std::vector<unsigned int> getMDsFromT5(SDL::Event* event, unsigned int T5);
std::vector<unsigned int> getHitsFromT5(SDL::Event* event, unsigned int T5);
std::vector<unsigned int> getHitIdxsFromT5(SDL::Event* event, unsigned int T5);
std::vector<unsigned int> getHitTypesFromT5(SDL::Event* event, unsigned int T5);
std::vector<unsigned int> getModuleIdxsFromT5(SDL::Event* event, unsigned int T5);

// ----* pT3 *----
unsigned int getPixelLSFrompT3(SDL::Event* event, unsigned int pT3);
unsigned int getT3FrompT3(SDL::Event* event, unsigned int pT3);
std::vector<unsigned int> getLSsFrompT3(SDL::Event* event, unsigned int pT3);
std::vector<unsigned int> getMDsFrompT3(SDL::Event* event, unsigned int pT3);
std::vector<unsigned int> getOuterTrackerHitsFrompT3(SDL::Event* event, unsigned int pT3);
std::vector<unsigned int> getPixelHitsFrompT3(SDL::Event* event, unsigned int pT3);
std::vector<unsigned int> getHitsFrompT3(SDL::Event* event, unsigned int pT3);
std::vector<unsigned int> getHitIdxsFrompT3(SDL::Event* event, unsigned int pT3);
std::vector<unsigned int> getHitTypesFrompT3(SDL::Event* event, unsigned int pT3);
std::vector<unsigned int> getModuleIdxsFrompT3(SDL::Event* event, unsigned int pT3);

// ----* pT5 *----
unsigned int getPixelLSFrompT5(SDL::Event* event, unsigned int pT5);
unsigned int getT5FrompT5(SDL::Event* event, unsigned int pT5);
std::vector<unsigned int> getT3sFrompT5(SDL::Event* event, unsigned int pT5);
std::vector<unsigned int> getLSsFrompT5(SDL::Event* event, unsigned int pT5);
std::vector<unsigned int> getMDsFrompT5(SDL::Event* event, unsigned int pT5);
std::vector<unsigned int> getOuterTrackerHitsFrompT5(SDL::Event* event, unsigned int pT5);
std::vector<unsigned int> getPixelHitsFrompT5(SDL::Event* event, unsigned int pT5);
std::vector<unsigned int> getHitsFrompT5(SDL::Event* event, unsigned int pT5);
std::vector<unsigned int> getHitIdxsFrompT5(SDL::Event* event, unsigned int pT5);
std::vector<unsigned int> getHitTypesFrompT5(SDL::Event* event, unsigned int pT5);
std::vector<unsigned int> getModuleIdxsFrompT5(SDL::Event* event, unsigned int pT5);

#endif
