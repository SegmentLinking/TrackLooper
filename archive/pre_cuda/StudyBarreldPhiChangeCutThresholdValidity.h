#ifndef StudyBarreldPhiChangeCutThresholdValidity_h
#define StudyBarreldPhiChangeCutThresholdValidity_h

#include "SDL/Event.cuh"

#include "Study.h"

#include <vector>
#include <tuple>

#include "TString.h"
#include "trktree.h"
#include "constants.h"

#include "AnalysisConfig.h"

class StudyBarreldPhiChangeCutThresholdValidity : public Study
{

public:

    std::vector<float> rt_v_phim__rt;
    std::vector<float> rt_v_phim__phim;
    std::array<std::vector<float>, NLAYERS> rt_v_phim__rt_by_layer;
    std::array<std::vector<float>, NLAYERS> rt_v_phim__phim_by_layer;
    std::vector<float> rt_v_dphi__rt;
    std::vector<float> rt_v_dphi__dphi;
    std::array<std::vector<float>, NLAYERS> rt_v_dphi__rt_by_layer;
    std::array<std::vector<float>, NLAYERS> rt_v_dphi__dphi_by_layer;

    virtual void bookStudy();
    virtual void doStudy(SDL::Event& recoevent, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents);

};

#endif
