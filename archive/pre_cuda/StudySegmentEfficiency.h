#ifndef StudySegmentEfficiency_h
#define StudySegmentEfficiency_h

#include "SDL/Event.h"

#include "Study.h"

#include <vector>
#include <tuple>

#include "TString.h"
#include "trktree.h"
#include "constants.h"

#include "AnalysisConfig.h"

class StudySegmentEfficiency : public Study
{

public:
    enum StudySegmentEfficiencyMode
    {
        kStudyEffAll = 1,
        kStudyEffBarrelBarrel,
        kStudyEffBarrelFlatBarrel,
        kStudyEffBarrelTiltBarrel,
        kStudyEffBarrelFlatBarrelFlat,
        kStudyEffBarrelFlatBarrelTilt,
        kStudyEffBarrelTiltBarrelFlat,
        kStudyEffBarrelTiltBarrelTilt,
        kStudyEffBarrelEndcap,
        kStudyEffBarrelTiltEndcap,
        kStudyEffBarrel,
        kStudyEffEndcap,
        kStudyEffEndcapPS,
        kStudyEffEndcapPSPS,
        kStudyEffEndcapPS2S,
        kStudyEffEndcap2S,
    };

    const char* studyname;
    StudySegmentEfficiencyMode mode;
    const char* modename;
    std::vector<float> pt_boundaries;
    std::vector<float> sg_matched_track_pt;
    std::vector<float> sg_all_track_pt;
    std::vector<float> sg_matched_track_eta;
    std::vector<float> sg_all_track_eta;
    std::vector<float> sg_matched_track_phi;
    std::vector<float> sg_all_track_phi;
    std::vector<float> sg_matched_track_z;
    std::vector<float> sg_all_track_z;
    std::vector<float> sg_matched_track_ring;
    std::vector<float> sg_all_track_ring;
    std::vector<float> sg_matched_track_module;
    std::vector<float> sg_all_track_module;
    std::vector<float> sg_matched_track_targ_ring;
    std::vector<float> sg_all_track_targ_ring;
    std::vector<float> sg_matched_track_targ_module;
    std::vector<float> sg_all_track_targ_module;
    std::array<std::vector<float>, NLAYERS> sg_matched_track_pt_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_all_track_pt_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_matched_track_eta_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_all_track_eta_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_matched_track_phi_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_all_track_phi_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_matched_track_wrapphi_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_all_track_wrapphi_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_matched_track_z_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_all_track_z_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_matched_track_ring_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_all_track_ring_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_matched_track_module_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_all_track_module_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_matched_track_targ_ring_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_all_track_targ_ring_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_matched_track_targ_module_by_layer;
    std::array<std::vector<float>, NLAYERS> sg_all_track_targ_module_by_layer;

    StudySegmentEfficiency(const char* studyName, StudySegmentEfficiencyMode, std::vector<float> ptboundaries);
    virtual void bookStudy();
    virtual void doStudy(SDL::Event& recoevent, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents);
    virtual void doStudy_v1(SDL::Event& recoevent, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents);

    virtual void printSegmentDebugInfo(SDL::Segment* sg, float pt);

};

#endif
