#ifndef StudyTrackCandidateSelection_h
#define StudyTrackCandidateSelection_h

#include "SDL/Event.h"

#include "Study.h"

#include <vector>
#include <tuple>

#include "TString.h"
#include "trktree.h"
#include "constants.h"

#include "AnalysisConfig.h"

class StudyTrackCandidateSelection : public Study
{

public:
    enum StudyTrackCandidateSelectionMode
    {
        kStudySelAll = 1,
        kStudySelSpecific,
    };

    const char* studyname;
    StudyTrackCandidateSelectionMode mode;
    const char* modename;
    std::vector<float> tc_cutflow;
    std::vector<float> tc_nocut_outer_tl_deltaBeta;
    std::vector<float> tc_nocut_inner_tl_deltaBeta;
    std::vector<float> tc_outer_tl_deltaBeta;
    std::vector<float> tc_inner_tl_deltaBeta;
    std::vector<float> tc_inner_tl_betaIn_minus_outer_tl_betaOut;
    std::vector<float> tc_inner_tl_betaAv_minus_outer_tl_betaAv;
    std::vector<float> tc_dr;
    std::vector<float> tc_r;
    std::vector<float> truth_tc_dr;
    std::vector<float> truth_tc_r;
    std::vector<float> truth_gt1pt_tc_dr;
    std::vector<float> truth_gt1pt_tc_r;
    std::array<std::vector<float>, NLAYERS> tc_cutflow_by_layer;

    std::vector<float> pt_boundaries;
    std::vector<float> tc_matched_track_pt;
    std::vector<float> tc_all_track_pt;

    StudyTrackCandidateSelection(const char* studyName, StudyTrackCandidateSelectionMode, std::vector<float> ptbounds);
    virtual void bookStudy();
    virtual void doStudy(SDL::Event& recoevent, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents);

};

#endif
