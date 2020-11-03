#ifndef StudyTrackletEfficiency_h
#define StudyTrackletEfficiency_h

#include "SDL/Event.h"

#include "Study.h"

#include <vector>
#include <tuple>

#include "TString.h"
#include "trktree.h"
#include "constants.h"

#include "AnalysisConfig.h"

class StudyTrackletEfficiency : public Study
{

public:
    enum StudyTrackletEfficiencyMode
    {
        kStudyEffAll = 1,
        kStudyEffBarrel1Barrel3,
        kStudyEffBarrel1FlatBarrel3Flat,
        kStudyEffBarrel1TiltBarrel3Flat,
        kStudyEffBarrel1TiltBarrel3Tilt,
        kStudyEffBarrel1TiltBarrel3TiltBarrel4,
        kStudyEffBarrel1TiltBarrel3TiltEndcap1,
        kStudyEffBarrelBarrelBarrelBarrel,
        kStudyEffBarrelBarrelEndcapEndcap,
        kStudyEffBB1BB3,
        kStudyEffBB2BB4,
        kStudyEffBB3BB5,
        kStudyEffSpecific,
    };

    const char* studyname;
    StudyTrackletEfficiencyMode mode;
    const char* modename;
    std::vector<float> pt_boundaries;
    std::vector<float> tl_matched_track_pt;
    std::vector<float> tl_all_track_pt;
    std::vector<float> tl_matched_track_eta;
    std::vector<float> tl_all_track_eta;
    std::vector<float> tl_matched_track_phi;
    std::vector<float> tl_all_track_phi;
    std::vector<float> tl_matched_track_z;
    std::vector<float> tl_all_track_z;
    std::array<std::vector<float>, NLAYERS> tl_matched_track_pt_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_all_track_pt_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_matched_track_eta_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_all_track_eta_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_matched_track_phi_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_all_track_phi_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_matched_track_wrapphi_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_all_track_wrapphi_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_matched_track_z_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_all_track_z_by_layer;

    StudyTrackletEfficiency(const char* studyName, StudyTrackletEfficiencyMode, std::vector<float> ptboundaries);
    virtual void bookStudy();
    virtual void doStudy(SDL::Event& recoevent, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents);
    virtual void doStudy_v1(SDL::Event& recoevent, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents);

};

#endif
