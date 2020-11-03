#ifndef StudySegmentSelectionOnTruth_h
#define StudySegmentSelectionOnTruth_h

#include "SDL/Event.h"

#include "Study.h"

#include <vector>
#include <tuple>

#include "TString.h"
#include "trktree.h"
#include "constants.h"

#include "AnalysisConfig.h"

class StudySegmentSelectionOnTruth : public Study
{

public:
    enum StudySegmentSelectionOnTruthMode
    {
        kStudySelAll = 1,
        kStudySelBB12,
        kStudySelBB23,
        kStudySelBB34,
        kStudySelBB45,
        kStudySelBB56,
        kStudySelSpecific,
    };

    const char* studyname;
    StudySegmentSelectionOnTruthMode mode;
    const char* modename;
    std::vector<float> sg_cutflow;
    std::vector<float> sg_zLo_cut;
    std::vector<float> sg_zHi_cut;

    StudySegmentSelectionOnTruth(const char* studyName, StudySegmentSelectionOnTruthMode);
    virtual void bookStudy();
    virtual void doStudy(SDL::Event& recoevent, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents);

};

#endif
