#ifndef StudySDLEfficiencyv2_h
#define StudySDLEfficiencyv2_h

#include "SDL/Event.h"

#include "Study.h"

#include <vector>
#include <tuple>

#include "TString.h"
#include "trktree.h"
#include "constants.h"

#include "AnalysisConfig.h"

#include "trkCore.h"

class StudySDLEfficiencyv2 : public Study
{

public:

    enum StudySDLEfficiencyv2Mode
    {
        kStudySDLBBBBBB = 1,
        kStudySDLBBBBBE,
        kStudySDLBBBBEE,
        kStudySDLBBBEEE,
        kStudySDLBBEEEE,
        kStudySDLBEEEEE,
    };

    const char* studyname;

    StudySDLEfficiencyv2Mode eff_mode;

    const char* eff_modename;

    std::vector<float> pt_boundaries;

    std::array<std::vector<float>, NLAYERS> md_matched_track_pt_by_layer;
    std::array<std::vector<float>, NLAYERS> md_all_track_pt_by_layer;
    std::array<std::vector<float>, NLAYERS> md_matched_track_eta_by_layer;
    std::array<std::vector<float>, NLAYERS> md_all_track_eta_by_layer;
    std::array<std::vector<float>, NLAYERS> md_matched_track_dxy_by_layer;
    std::array<std::vector<float>, NLAYERS> md_all_track_dxy_by_layer;
    std::array<std::vector<float>, NLAYERS> md_matched_track_dz_by_layer;
    std::array<std::vector<float>, NLAYERS> md_all_track_dz_by_layer;
    std::array<std::vector<float>, NLAYERS> md_matched_track_pt_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> md_all_track_pt_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> md_matched_track_eta_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> md_all_track_eta_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> md_matched_track_dxy_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> md_all_track_dxy_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> md_matched_track_dz_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> md_all_track_dz_mtv_by_layer;

    std::array<std::vector<float>, NLAYERS> sg_matched_track_pt_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_all_track_pt_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_matched_track_eta_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_all_track_eta_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_matched_track_dxy_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_all_track_dxy_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_matched_track_dz_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_all_track_dz_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_matched_track_pt_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_all_track_pt_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_matched_track_eta_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_all_track_eta_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_matched_track_dxy_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_all_track_dxy_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_matched_track_dz_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_all_track_dz_mtv_by_layer;

    std::array<std::vector<float>, NLAYERS> tl_matched_track_pt_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_all_track_pt_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_matched_track_eta_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_all_track_eta_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_matched_track_dxy_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_all_track_dxy_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_matched_track_dz_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_all_track_dz_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_matched_track_pt_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_all_track_pt_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_matched_track_eta_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_all_track_eta_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_matched_track_dxy_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_all_track_dxy_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_matched_track_dz_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_all_track_dz_mtv_by_layer;

    std::array<std::vector<float>, NLAYERS> tc_matched_track_pt_by_layer;
    std::array<std::vector<float>, NLAYERS> tc_all_track_pt_by_layer;
    std::array<std::vector<float>, NLAYERS> tc_matched_track_eta_by_layer;
    std::array<std::vector<float>, NLAYERS> tc_all_track_eta_by_layer;
    std::array<std::vector<float>, NLAYERS> tc_matched_track_dxy_by_layer;
    std::array<std::vector<float>, NLAYERS> tc_all_track_dxy_by_layer;
    std::array<std::vector<float>, NLAYERS> tc_matched_track_dz_by_layer;
    std::array<std::vector<float>, NLAYERS> tc_all_track_dz_by_layer;
    std::array<std::vector<float>, NLAYERS> tc_matched_track_pt_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> tc_all_track_pt_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> tc_matched_track_eta_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> tc_all_track_eta_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> tc_matched_track_dxy_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> tc_all_track_dxy_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> tc_matched_track_dz_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> tc_all_track_dz_mtv_by_layer;

    StudySDLEfficiencyv2(
        const char* studyName,
        StudySDLEfficiencyv2Mode eff_mode_,
        std::vector<float> ptbounds);
    virtual void bookStudy();
    virtual void doStudy(SDL::Event& recoevent, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents);

};

#endif
