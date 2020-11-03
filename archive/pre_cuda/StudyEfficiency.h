#ifndef StudyEfficiency_h
#define StudyEfficiency_h

#include "SDL/Event.h"

#include "Study.h"

#include <vector>
#include <tuple>

#include "TString.h"
#include "trktree.h"
#include "constants.h"

#include "AnalysisConfig.h"

class StudyEfficiency : public Study
{

public:
    enum StudyEfficiencyMode
    {
        kStudyEffAll = 1,
        kStudyEffBarrel,
        kStudyEffBarrelFlat,
        kStudyEffBarrelTilt,
        kStudyEffBarrelTiltHighZ,
        kStudyEffBarrelTiltLowZ,
        kStudyEffEndcap,
        kStudyEffEndcapPS,
        kStudyEffEndcap2S,
        kStudyEffEndcapPSCloseRing,
        kStudyEffEndcapPSLowPt,
    };

    const char* studyname;
    StudyEfficiencyMode mode;
    const char* modename;
    std::vector<float> pt_boundaries;
    std::vector<float> md_matched_track_pt;
    std::vector<float> md_all_track_pt;
    std::vector<float> md_matched_track_eta;
    std::vector<float> md_all_track_eta;
    std::vector<float> md_matched_track_phi;
    std::vector<float> md_all_track_phi;
    std::vector<float> md_matched_track_z;
    std::vector<float> md_all_track_z;
    std::array<std::vector<float>, NLAYERS> md_matched_track_pt_by_layer;
    std::array<std::vector<float>, NLAYERS> md_all_track_pt_by_layer;
    std::array<std::vector<float>, NLAYERS> md_matched_track_eta_by_layer;
    std::array<std::vector<float>, NLAYERS> md_all_track_eta_by_layer;
    std::array<std::vector<float>, NLAYERS> md_matched_track_phi_by_layer;
    std::array<std::vector<float>, NLAYERS> md_all_track_phi_by_layer;
    std::array<std::vector<float>, NLAYERS> md_matched_track_wrapphi_by_layer;
    std::array<std::vector<float>, NLAYERS> md_all_track_wrapphi_by_layer;
    std::array<std::vector<float>, NLAYERS> md_matched_track_z_by_layer;
    std::array<std::vector<float>, NLAYERS> md_all_track_z_by_layer;
    std::vector<float> md_lower_hit_only_track_pt;
    std::vector<float> md_lower_hit_only_track_eta;

    StudyEfficiency(const char* studyName, StudyEfficiencyMode, std::vector<float> ptboundaries);
    virtual void bookStudy();
    virtual void doStudy(SDL::Event& recoevent, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents);
    virtual void doStudy_v1(SDL::Event& recoevent, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents);

};

#endif
