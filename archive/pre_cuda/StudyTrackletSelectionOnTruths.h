#ifndef StudyTrackletSelectionOnTruths_h
#define StudyTrackletSelectionOnTruths_h

#include "SDL/Event.h"

#include "Study.h"

#include <vector>
#include <tuple>

#include "TString.h"
#include "trktree.h"
#include "constants.h"

#include "AnalysisConfig.h"

class StudyTrackletSelectionOnTruths : public Study
{

public:
    enum StudyTrackletSelectionOnTruthsMode
    {
        kStudySelAll = 1,
        kStudySelSpecific,
    };

    const char* studyname;
    StudyTrackletSelectionOnTruthsMode mode;
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

    std::array<std::vector<float>, 7> tl_deltaBeta_ptslice;
    std::array<std::array<std::vector<float>, 7>, NLAYERS> tl_deltaBeta_ptslice_by_layer;
    std::array<std::vector<float>, 7> tl_deltaBeta_postCut_ptslice;
    std::array<std::array<std::vector<float>, 7>, NLAYERS> tl_deltaBeta_postCut_ptslice_by_layer;

    StudyTrackletSelectionOnTruths(const char* studyName, StudyTrackletSelectionOnTruthsMode);
    virtual void bookStudy();
    virtual void doStudy(SDL::Event& recoevent, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents);

};

#endif
