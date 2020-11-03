#ifndef StudyMiniDoubletOccupancy_h
#define StudyMiniDoubletOccupancy_h

#include "SDL/Event.h"

#include "Study.h"

#include <vector>
#include <tuple>

#include "TString.h"
#include "trktree.h"
#include "constants.h"

#include "AnalysisConfig.h"

class StudyMiniDoubletOccupancy : public Study
{

public:
    enum StudyMiniDoubletOccupancyMode
    {
        kStudyAll = 1,
    };

    const char* studyname;
    StudyMiniDoubletOccupancyMode mode;
    const char* modename;
    std::array<int, NLAYERS> n_in_lower_modules_by_layer;
    std::array<int, NLAYERS> n_in_upper_modules_by_layer;
    std::array<int, NLAYERS> n_in_both_modules_by_layer;
    std::array<std::vector<float>, NLAYERS> dz_lower_modules_by_layer;
    std::array<std::vector<float>, NLAYERS> dz_upper_modules_by_layer;
    std::array<std::vector<float>, NLAYERS> dz_both_modules_by_layer;
    std::array<std::vector<float>, NLAYERS> n_cross_lower_modules_by_layer;
    std::array<std::vector<float>, NLAYERS> n_cross_upper_modules_by_layer;
    std::array<std::vector<float>, NLAYERS> n_cross_both_modules_by_layer;
    std::array<std::vector<float>, NLAYERS> true_dz_lower_modules_by_layer;
    std::array<std::vector<float>, NLAYERS> true_dz_upper_modules_by_layer;
    std::array<std::vector<float>, NLAYERS> true_dz_both_modules_by_layer;
    std::array<std::vector<float>, NLAYERS> n_true_cross_lower_modules_by_layer;
    std::array<std::vector<float>, NLAYERS> n_true_cross_upper_modules_by_layer;
    std::array<std::vector<float>, NLAYERS> n_true_cross_both_modules_by_layer;

    StudyMiniDoubletOccupancy(const char* studyName, StudyMiniDoubletOccupancy::StudyMiniDoubletOccupancyMode mode_);
    virtual void bookStudy();
    virtual void doStudy(SDL::Event& recoevent, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents);

};

#endif
