#ifndef StudySDLInefficiency_h
#define StudySDLInefficiency_h

#include "SDL/Event.h"

#include "Study.h"

#include <vector>
#include <tuple>

#include "TString.h"
#include "trktree.h"
#include "constants.h"

#include "AnalysisConfig.h"

#include "trkCore.h"

class StudySDLInefficiency : public Study
{

public:
    enum StudySDLMiniDoubletInefficiencyMode
    {
        kStudySDLMDEffAll = 1,
        kStudySDLMDEffBarrel,
        kStudySDLMDEffBarrelFlat,
        kStudySDLMDEffBarrelTilt,
        kStudySDLMDEffEndcap,
        kStudySDLMDEffEndcapPS,
        kStudySDLMDEffEndcap2S,
    };

    enum StudySDLSegmentInefficiencyMode
    {
        kStudySDLSGEffAll = 1,
        kStudySDLSGEffBB,
    };

    enum StudySDLTrackletInefficiencyMode
    {
        kStudySDLTLEffAll = 1,
        kStudySDLTLEffBBBB,
    };

    enum StudySDLTrackCandidateInefficiencyMode
    {
        kStudySDLTCEffAll = 1,
        kStudySDLTCEffBBBBBB,
    };

    const char* studyname;

    StudySDLMiniDoubletInefficiencyMode md_mode;
    StudySDLSegmentInefficiencyMode sg_mode;
    StudySDLTrackletInefficiencyMode tl_mode;
    StudySDLTrackCandidateInefficiencyMode tc_mode;

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

    std::array<std::vector<float>, NLAYERS> sg_matched_track_pt_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_all_track_pt_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_matched_track_eta_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_all_track_eta_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_matched_track_dxy_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_all_track_dxy_by_layer;

    std::array<std::vector<float>, NLAYERS> tl_matched_track_pt_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_all_track_pt_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_matched_track_eta_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_all_track_eta_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_matched_track_dxy_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_all_track_dxy_by_layer;

    std::array<std::vector<float>, NLAYERS> tc_matched_track_pt_by_layer;
    std::array<std::vector<float>, NLAYERS> tc_all_track_pt_by_layer;
    std::array<std::vector<float>, NLAYERS> tc_matched_track_eta_by_layer;
    std::array<std::vector<float>, NLAYERS> tc_all_track_eta_by_layer;
    std::array<std::vector<float>, NLAYERS> tc_matched_track_dxy_by_layer;
    std::array<std::vector<float>, NLAYERS> tc_all_track_dxy_by_layer;

    StudySDLInefficiency(
        const char* studyName,
        StudySDLMiniDoubletInefficiencyMode md_mode_,
        StudySDLSegmentInefficiencyMode sg_mode_,
        StudySDLTrackletInefficiencyMode tl_mode_,
        StudySDLTrackCandidateInefficiencyMode tc_mode_,
        std::vector<float> ptbounds);
    virtual void bookStudy();
    virtual void doStudy(SDL::Event& recoevent, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents);

};

#endif
