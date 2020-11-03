#ifndef StudyConditionalHitEfficiency_h
#define StudyConditionalHitEfficiency_h

#include "SDL/Event.h"

#include "Study.h"

#include <vector>
#include <tuple>

#include "TString.h"
#include "trktree.h"
#include "constants.h"

#include "AnalysisConfig.h"

#include "trkCore.h"

class StudyConditionalHitEfficiency : public Study
{

public:
    const char* studyname;

    std::vector<float> pt_boundaries;

    int pdgid_of_study;

    std::vector<float> pt_all;
    std::vector<float> eta_all;
    std::vector<float> phi_all;
    std::vector<float> pt_all_w_last_layer;

    std::array<std::vector<float>, 13> pt_w_nmiss_simhits;
    std::array<std::vector<float>, 13> pt_w_nmiss_hits;
    std::array<std::vector<float>, 13> pt_w_miss_layer;
    std::array<std::array<std::vector<float>, 13>, 13> pt_w_nmiss_miss_layer;

    std::array<std::vector<float>, 13> eta_w_nmiss_simhits;
    std::array<std::vector<float>, 13> eta_w_nmiss_hits;
    std::array<std::vector<float>, 13> eta_w_miss_layer;
    std::array<std::array<std::vector<float>, 13>, 13> eta_w_nmiss_miss_layer;

    std::array<std::vector<float>, 13> phi_w_nmiss_simhits;
    std::array<std::vector<float>, 13> phi_w_nmiss_hits;
    std::array<std::vector<float>, 13> phi_w_miss_layer;
    std::array<std::array<std::vector<float>, 13>, 13> phi_w_nmiss_miss_layer;

    std::array<std::vector<float>, 2> ptbin_nhits;
    std::array<std::vector<float>, 2> ptbin_miss_layer;

    std::array<std::vector<float>, 3> prop_phi_2Slayer;
    std::array<std::vector<float>, 3> prop_115_phi_2Slayer;

    std::vector<float> pt_0p95_1p05_hit_miss_study;
    std::array<std::vector<float>, 3> pt_0p95_1p05_nmiss2_prop_phi_2Slayer;
    std::array<std::vector<float>, 3> pt_0p95_1p05_nmiss4_prop_phi_2Slayer_0;
    std::array<std::vector<float>, 3> pt_0p95_1p05_nmiss4_prop_phi_2Slayer_1;

    StudyConditionalHitEfficiency(
        const char* studyName,
        std::vector<float> ptbounds,
        int pdgid
        );
    virtual void bookStudy();
    virtual void doStudy(SDL::Event& recoevent, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents);
    float prop_phi(unsigned int isimtrk, unsigned int originlayer, float targetR);

};

#endif
