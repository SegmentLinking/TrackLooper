#ifndef StudySDLEfficiency_h
#define StudySDLEfficiency_h

#include "SDL/Event.h"

#include "Study.h"

#include <vector>
#include <tuple>

#include "TString.h"
#include "trktree.h"
#include "constants.h"

#include "AnalysisConfig.h"

#include "trkCore.h"

class StudySDLEfficiency : public Study
{

public:
    enum StudySDLMiniDoubletEfficiencyMode
    {
        kStudySDLMDEffAll = 1,
        kStudySDLMDEffBarrel,
        kStudySDLMDEffBarrelFlat,
        kStudySDLMDEffBarrelTilt,
        kStudySDLMDEffEndcap,
        kStudySDLMDEffEndcapPS,
        kStudySDLMDEffEndcap2S,
    };

    enum StudySDLSegmentEfficiencyMode
    {
        kStudySDLSGEffAll = 1,
        kStudySDLSGEffBB,
    };

    enum StudySDLTrackletEfficiencyMode
    {
        kStudySDLTLEffAll = 1,
        kStudySDLTLEffBBBB,
    };

    enum StudySDLTrackCandidateEfficiencyMode
    {
        kStudySDLTCEffAll = 1,
        kStudySDLTCEffBBBBBB,
    };

    const char* studyname;

    StudySDLMiniDoubletEfficiencyMode md_mode;
    StudySDLSegmentEfficiencyMode sg_mode;
    StudySDLTrackletEfficiencyMode tl_mode;
    StudySDLTrackCandidateEfficiencyMode tc_mode;

    const char* md_modename;
    const char* sg_modename;
    const char* tl_modename;
    const char* tc_modename;

    std::vector<float> pt_boundaries;

    std::array<std::vector<float>, NLAYERS> md_matched_track_pt_by_layer;
    std::array<std::vector<float>, NLAYERS> md_all_track_pt_by_layer;
    std::array<std::vector<float>, NLAYERS> md_matched_track_eta_by_layer;
    std::array<std::vector<float>, NLAYERS> md_all_track_eta_by_layer;
    std::array<std::vector<float>, NLAYERS> md_matched_track_dxy_by_layer;
    std::array<std::vector<float>, NLAYERS> md_all_track_dxy_by_layer;
    std::array<std::vector<float>, NLAYERS> md_matched_track_pt_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> md_all_track_pt_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> md_matched_track_eta_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> md_all_track_eta_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> md_matched_track_dxy_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> md_all_track_dxy_mtv_by_layer;

    std::array<std::vector<float>, NLAYERS> sg_matched_track_pt_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_all_track_pt_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_matched_track_eta_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_all_track_eta_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_matched_track_dxy_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_all_track_dxy_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_matched_track_pt_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_all_track_pt_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_matched_track_eta_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_all_track_eta_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_matched_track_dxy_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_all_track_dxy_mtv_by_layer;

    std::array<std::vector<float>, NLAYERS> tl_matched_track_pt_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_all_track_pt_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_matched_track_eta_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_all_track_eta_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_matched_track_dxy_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_all_track_dxy_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_matched_track_pt_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_all_track_pt_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_matched_track_eta_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_all_track_eta_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_matched_track_dxy_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_all_track_dxy_mtv_by_layer;

    std::array<std::vector<float>, NLAYERS> tc_matched_track_pt_by_layer;
    std::array<std::vector<float>, NLAYERS> tc_all_track_pt_by_layer;
    std::array<std::vector<float>, NLAYERS> tc_matched_track_eta_by_layer;
    std::array<std::vector<float>, NLAYERS> tc_all_track_eta_by_layer;
    std::array<std::vector<float>, NLAYERS> tc_matched_track_dxy_by_layer;
    std::array<std::vector<float>, NLAYERS> tc_all_track_dxy_by_layer;
    std::array<std::vector<float>, NLAYERS> tc_matched_track_pt_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> tc_all_track_pt_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> tc_matched_track_eta_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> tc_all_track_eta_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> tc_matched_track_dxy_mtv_by_layer;
    std::array<std::vector<float>, NLAYERS> tc_all_track_dxy_mtv_by_layer;

    StudySDLEfficiency(
        const char* studyName,
        StudySDLMiniDoubletEfficiencyMode md_mode_,
        StudySDLSegmentEfficiencyMode sg_mode_,
        StudySDLTrackletEfficiencyMode tl_mode_,
        StudySDLTrackCandidateEfficiencyMode tc_mode_,
        std::vector<float> ptbounds);
    virtual void bookStudy();
    virtual void doStudy(SDL::Event& recoevent, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents);

};

#endif
