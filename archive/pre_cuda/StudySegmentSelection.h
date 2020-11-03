#ifndef StudySegmentSelection_h
#define StudySegmentSelection_h

#include "SDL/Event.h"

#include "Study.h"

#include <vector>
#include <tuple>

#include "TString.h"
#include "trktree.h"
#include "constants.h"

#include "AnalysisConfig.h"

class StudySegmentSelection : public Study
{

public:
    enum StudySegmentSelectionMode
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
    StudySegmentSelectionMode mode;
    const char* modename;
    std::map<std::string, std::vector<float>> histvars;

    StudySegmentSelection(const char* studyName, StudySegmentSelectionMode);
    virtual void bookStudy();
    virtual void doStudy(SDL::Event& recoevent, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents);

};

#endif
