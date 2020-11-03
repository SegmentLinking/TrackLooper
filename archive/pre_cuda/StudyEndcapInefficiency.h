#ifndef StudyEndcapInefficiency_h
#define StudyEndcapInefficiency_h

#include "SDL/Event.h"

#include "Study.h"

#include <vector>
#include <tuple>

#include "TString.h"
#include "trktree.h"
#include "constants.h"

#include "AnalysisConfig.h"

class StudyEndcapInefficiency : public Study
{

public:
    enum StudyEndcapInefficiencyMode
    {
        kStudyEndcapIneffAll = 1,
        kStudyEndcapIneffPS = 2,
        kStudyEndcapIneff2S = 3,
        kStudyEndcapIneffPSLowP = 4,
        kStudyEndcapIneffPSLowS = 5,
    };

    const char* studyname;
    StudyEndcapInefficiencyMode mode;
    const char* modename;

    std::vector<float> dzs_passed;
    std::vector<float> drts_passed;
    std::vector<float> fabsdPhis_passed;
    std::vector<float> zs_passed;
    std::vector<float> dzfracs_passed;
    std::vector<float> fabsdPhiMods_passed;
    std::vector<float> fabsdPhiModDiffs_passed;

    std::vector<float> dzs_failed;
    std::vector<float> drts_failed;
    std::vector<float> fabsdPhis_failed;
    std::vector<float> zs_failed;
    std::vector<float> dzfracs_failed;
    std::vector<float> fabsdPhiMods_failed;
    std::vector<float> fabsdPhiModDiffs_failed;

    StudyEndcapInefficiency(const char* studyName, StudyEndcapInefficiencyMode);
    virtual void bookStudy();
    virtual void doStudy(SDL::Event& recoevent, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents);
};

#endif
