#ifndef StudyTripletSelection_h
#define StudyTripletSelection_h

#include "SDL/Event.h"

#include "Study.h"

#include <vector>
#include <tuple>

#include "TString.h"
#include "trktree.h"
#include "constants.h"

#include "AnalysisConfig.h"

class StudyTripletSelection : public Study
{

public:
    enum StudyTripletSelectionMode
    {
        kStudySelAll = 1,
        kStudySelBB1BB2,
        kStudySelBB2BB3,
        kStudySelBB3BB4,
        kStudySelBB4BB5,
        kStudySelSpecific,
    };

    const char* studyname;
    StudyTripletSelectionMode mode;
    const char* modename;
    std::vector<float> tp_deltaBeta;
    std::vector<float> tp_deltaBeta_midpoint;
    std::vector<float> tp_deltaBeta_3rdCorr;
    std::vector<float> tp_deltaBeta_4thCorr;

    StudyTripletSelection(const char* studyName, StudyTripletSelectionMode);
    virtual void bookStudy();
    virtual void doStudy(SDL::Event& recoevent, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents);

};

#endif
