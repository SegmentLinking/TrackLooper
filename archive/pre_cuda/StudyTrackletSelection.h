#ifndef StudyTrackletSelection_h
#define StudyTrackletSelection_h

#include "SDL/Event.h"

#include "Study.h"

#include <vector>
#include <tuple>

#include "TString.h"
#include "trktree.h"
#include "constants.h"

#include "AnalysisConfig.h"

class StudyTrackletSelection : public Study
{

public:
    enum StudyTrackletSelectionMode
    {
        kStudySelAll = 1,
        kStudySelBarrelBarrelBarrelBarrel,
        kStudySelBarrelBarrelEndcapEndcap,
        kStudySelBB1BB3,
        kStudySelBB1BE3,
        kStudySelBB1EE3,
        kStudySelBE1EE3,
        kStudySelEE1EE3,
        kStudySelBB2BB4,
        kStudySelBB2BE4,
        kStudySelBB2EE4,
        kStudySelBE2EE4,
        kStudySelEE2EE4,
        kStudySelBB3BB5,
        kStudySelBB3BE5,
        kStudySelBB3EE5,
        kStudySelBE3EE5,
        kStudySelBB1BB4,
        kStudySelBB1BB5,
        kStudySelEE1EE3AllPS,
        kStudySelEE1EE3All2S,
        kStudySelSpecific,
    };

    const char* studyname;
    StudyTrackletSelectionMode mode;
    const char* modename;
    std::vector<float> tl_deltaBeta_postCut;
    std::vector<float> tl_deltaBeta;
    std::vector<float> tl_deltaBeta_dcut;
    std::vector<float> tl_betaOut;
    std::vector<float> tl_betaOut_dcut;
    std::vector<float> tl_betaOut_cutthresh;
    std::vector<float> tl_betaIn;
    std::vector<float> tl_betaIn_dcut;
    std::vector<float> tl_betaIn_cutthresh;
    std::vector<float> tl_deltaBeta_midpoint;
    std::vector<float> tl_deltaBeta_3rdCorr;
    std::vector<float> tl_deltaBeta_4thCorr;
    std::vector<float> tl_cutflow;
    std::array<std::vector<float>, NLAYERS> tl_deltaBeta_postCut_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_deltaBeta_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_deltaBeta_dcut_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_betaOut_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_betaOut_dcut_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_betaOut_cutthresh_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_betaIn_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_betaIn_dcut_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_betaIn_cutthresh_by_layer;
    std::array<std::vector<float>, NLAYERS> tl_cutflow_by_layer;

    std::vector<float> tl_truth_deltaBeta;
    std::vector<float> tl_truth_deltaBeta_4thCorr;
    std::vector<float> tl_truth_cutflow;

    std::vector<float> tl_matched_track_pt;
    std::vector<float> tl_matched_track_deltaBeta;
    std::vector<float> tl_matched_track_deltaBeta_4thCorr;

    std::vector<float> tl_matched_pt2_track_pt;
    std::vector<float> tl_matched_pt2_track_deltaBeta;
    std::vector<float> tl_matched_pt2_track_deltaBeta_4thCorr;

    std::vector<float> tl_matched_pt1peak_track_pt;
    std::vector<float> tl_matched_pt1peak_track_deltaBeta;
    std::vector<float> tl_matched_pt1peak_track_deltaBeta_4thCorr;

    std::vector<float> tl_matched_pt1p5peak_track_pt;
    std::vector<float> tl_matched_pt1p5peak_track_deltaBeta;
    std::vector<float> tl_matched_pt1p5peak_track_deltaBeta_4thCorr;

    std::vector<float> tl_matched_pt2peak_track_pt;
    std::vector<float> tl_matched_pt2peak_track_deltaBeta;
    std::vector<float> tl_matched_pt2peak_track_deltaBeta_4thCorr;

    std::vector<float> tl_matched_pt2p5peak_track_pt;
    std::vector<float> tl_matched_pt2p5peak_track_deltaBeta;
    std::vector<float> tl_matched_pt2p5peak_track_deltaBeta_4thCorr;

    std::vector<float> tl_matched_pt3peak_track_pt;
    std::vector<float> tl_matched_pt3peak_track_deltaBeta;
    std::vector<float> tl_matched_pt3peak_track_deltaBeta_4thCorr;

    std::vector<float> tl_matched_pos_track_pt;
    std::vector<float> tl_matched_pos_track_deltaBeta;
    std::vector<float> tl_matched_pos_track_deltaBeta_4thCorr;

    std::vector<float> tl_matched_pos_pt1peak_track_pt;
    std::vector<float> tl_matched_pos_pt1peak_track_deltaBeta;
    std::vector<float> tl_matched_pos_pt1peak_track_deltaBeta_4thCorr;

    std::vector<float> tl_matched_pos_pt1p5peak_track_pt;
    std::vector<float> tl_matched_pos_pt1p5peak_track_deltaBeta;
    std::vector<float> tl_matched_pos_pt1p5peak_track_deltaBeta_4thCorr;

    std::vector<float> tl_matched_pos_pt2peak_track_pt;
    std::vector<float> tl_matched_pos_pt2peak_track_deltaBeta;
    std::vector<float> tl_matched_pos_pt2peak_track_deltaBeta_4thCorr;

    std::vector<float> tl_matched_pos_pt2p5peak_track_pt;
    std::vector<float> tl_matched_pos_pt2p5peak_track_deltaBeta;
    std::vector<float> tl_matched_pos_pt2p5peak_track_deltaBeta_4thCorr;

    std::vector<float> tl_matched_pos_pt3peak_track_pt;
    std::vector<float> tl_matched_pos_pt3peak_track_deltaBeta;
    std::vector<float> tl_matched_pos_pt3peak_track_deltaBeta_4thCorr;

    std::vector<float> tl_matched_pos_pt5peak_track_pt;
    std::vector<float> tl_matched_pos_pt5peak_track_deltaBeta;
    std::vector<float> tl_matched_pos_pt5peak_track_deltaBeta_4thCorr;

    std::vector<float> tl_matched_pos_pt10peak_track_pt;
    std::vector<float> tl_matched_pos_pt10peak_track_deltaBeta;
    std::vector<float> tl_matched_pos_pt10peak_track_deltaBeta_4thCorr;

    std::vector<float> tl_matched_pos_pt10_track_pt;
    std::vector<float> tl_matched_pos_pt10_track_deltaBeta;
    std::vector<float> tl_matched_pos_pt10_track_deltaBeta_4thCorr;

    std::vector<float> tl_matched_pos_pt20_track_pt;
    std::vector<float> tl_matched_pos_pt20_track_deltaBeta;
    std::vector<float> tl_matched_pos_pt20_track_deltaBeta_4thCorr;

    std::vector<float> tl_matched_pos_pt50_track_pt;
    std::vector<float> tl_matched_pos_pt50_track_deltaBeta;
    std::vector<float> tl_matched_pos_pt50_track_deltaBeta_4thCorr;

    std::vector<float> tl_matched_neg_track_pt;
    std::vector<float> tl_matched_neg_track_deltaBeta;
    std::vector<float> tl_matched_neg_track_deltaBeta_4thCorr;

    std::vector<float> tl_matched_neg_pt1peak_track_pt;
    std::vector<float> tl_matched_neg_pt1peak_track_deltaBeta;
    std::vector<float> tl_matched_neg_pt1peak_track_deltaBeta_4thCorr;

    std::vector<float> tl_matched_neg_pt1p5peak_track_pt;
    std::vector<float> tl_matched_neg_pt1p5peak_track_deltaBeta;
    std::vector<float> tl_matched_neg_pt1p5peak_track_deltaBeta_4thCorr;

    std::vector<float> tl_matched_neg_pt2peak_track_pt;
    std::vector<float> tl_matched_neg_pt2peak_track_deltaBeta;
    std::vector<float> tl_matched_neg_pt2peak_track_deltaBeta_4thCorr;

    std::vector<float> tl_matched_neg_pt2p5peak_track_pt;
    std::vector<float> tl_matched_neg_pt2p5peak_track_deltaBeta;
    std::vector<float> tl_matched_neg_pt2p5peak_track_deltaBeta_4thCorr;

    std::vector<float> tl_matched_neg_pt3peak_track_pt;
    std::vector<float> tl_matched_neg_pt3peak_track_deltaBeta;
    std::vector<float> tl_matched_neg_pt3peak_track_deltaBeta_4thCorr;

    std::vector<float> tl_matched_neg_pt5peak_track_pt;
    std::vector<float> tl_matched_neg_pt5peak_track_deltaBeta;
    std::vector<float> tl_matched_neg_pt5peak_track_deltaBeta_4thCorr;

    std::vector<float> tl_matched_neg_pt10peak_track_pt;
    std::vector<float> tl_matched_neg_pt10peak_track_deltaBeta;
    std::vector<float> tl_matched_neg_pt10peak_track_deltaBeta_4thCorr;

    std::vector<float> tl_matched_neg_pt10_track_pt;
    std::vector<float> tl_matched_neg_pt10_track_deltaBeta;
    std::vector<float> tl_matched_neg_pt10_track_deltaBeta_4thCorr;

    std::vector<float> tl_matched_neg_pt20_track_pt;
    std::vector<float> tl_matched_neg_pt20_track_deltaBeta;
    std::vector<float> tl_matched_neg_pt20_track_deltaBeta_4thCorr;

    std::vector<float> tl_matched_neg_pt50_track_pt;
    std::vector<float> tl_matched_neg_pt50_track_deltaBeta;
    std::vector<float> tl_matched_neg_pt50_track_deltaBeta_4thCorr;

    std::vector<float> tl_unmatched_track_deltaBeta;
    std::vector<float> tl_unmatched_track_deltaBeta_4thCorr;

    std::array<std::vector<float>, 7> tl_deltaBeta_ptslice;
    std::array<std::array<std::vector<float>, 7>, NLAYERS> tl_deltaBeta_ptslice_by_layer;
    std::array<std::vector<float>, 7> tl_deltaBeta_postCut_ptslice;
    std::array<std::array<std::vector<float>, 7>, NLAYERS> tl_deltaBeta_postCut_ptslice_by_layer;

    StudyTrackletSelection(const char* studyName, StudyTrackletSelectionMode);
    virtual void bookStudy();
    virtual void doStudy(SDL::Event& recoevent, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents);

};

#endif
