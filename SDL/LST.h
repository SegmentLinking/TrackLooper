#include <filesystem>
#include <cstdlib>
#include <numeric>

#include "../code/cppitertools/enumerate.hpp"

#include "TString.h"
#include "Math/Vector3D.h"
#include <Math/Vector4D.h>

#include "Event.cuh"

namespace SDL {
  
  class LST {
  public:
    LST();

    void eventSetup();
    void run(cudaStream_t stream,
             const std::vector<float> see_px,
             const std::vector<float> see_py,
             const std::vector<float> see_pz,
             const std::vector<float> see_dxy,
             const std::vector<float> see_dz,
             const std::vector<float> see_ptErr,
             const std::vector<float> see_etaErr,
             const std::vector<float> see_stateTrajGlbX,
             const std::vector<float> see_stateTrajGlbY,
             const std::vector<float> see_stateTrajGlbZ,
             const std::vector<float> see_stateTrajGlbPx,
             const std::vector<float> see_stateTrajGlbPy,
             const std::vector<float> see_stateTrajGlbPz,
             const std::vector<int> see_q,
             const std::vector<unsigned int> see_algo,
             const std::vector<std::vector<int>> see_hitIdx,
             const std::vector<unsigned int> ph2_detId,
             const std::vector<float> ph2_x,
             const std::vector<float> ph2_y,
             const std::vector<float> ph2_z);
    std::vector<float> pt() { return out_tc_pt_; }
    std::vector<float> eta() { return out_tc_eta_; }
    std::vector<float> phi() { return out_tc_phi_; }
    std::vector<std::vector<unsigned int>> hits() { return out_tc_hitIdxs_; }
    std::vector<unsigned int> len() { return out_tc_len_; }
    std::vector<int> seedIdx() { return out_tc_seedIdx_; }
  private:
    void loadMaps();
    TString get_absolute_path_after_check_file_exists(const std::string name);
    void prepareInput(const std::vector<float> see_px,
                      const std::vector<float> see_py,
                      const std::vector<float> see_pz,
                      const std::vector<float> see_dxy,
                      const std::vector<float> see_dz,
                      const std::vector<float> see_ptErr,
                      const std::vector<float> see_etaErr,
                      const std::vector<float> see_stateTrajGlbX,
                      const std::vector<float> see_stateTrajGlbY,
                      const std::vector<float> see_stateTrajGlbZ,
                      const std::vector<float> see_stateTrajGlbPx,
                      const std::vector<float> see_stateTrajGlbPy,
                      const std::vector<float> see_stateTrajGlbPz,
                      const std::vector<int> see_q,
                      const std::vector<unsigned int> see_algo,
                      const std::vector<std::vector<int>> see_hitIdx,
                      const std::vector<unsigned int> ph2_detId,
                      const std::vector<float> ph2_x,
                      const std::vector<float> ph2_y,
                      const std::vector<float> ph2_z);

    ROOT::Math::XYZVector calculateR3FromPCA(const ROOT::Math::PxPyPzMVector& p3, const float dxy, const float dz);

    void getOutput(SDL::Event& event);
    // Object accessors
    // ----* pLS *----
    std::vector<unsigned int> getPixelHitsFrompLS(const SDL::modules& modulesInGPU, const SDL::objectRanges& rangesInGPU, const SDL::miniDoublets& miniDoubletsInGPU, const SDL::segments& segmentsInGPU, unsigned int pLS);
    std::vector<unsigned int> getPixelHitIdxsFrompLS(const SDL::modules& modulesInGPU, const SDL::objectRanges& rangesInGPU, const SDL::hits& hitsInGPU, const SDL::miniDoublets& miniDoubletsInGPU, const SDL::segments& segmentsInGPU, unsigned int pLS);
    // ----* MD *----
    std::vector<unsigned int> getHitsFromMD(const SDL::miniDoublets& miniDoubletsInGPU, unsigned int MD);
    // ----* LS *----
    std::vector<unsigned int> getMDsFromLS(const SDL::segments& segmentsInGPU, unsigned int LS);
    // ----* T3 *----
    std::vector<unsigned int> getLSsFromT3(const SDL::triplets& tripletsInGPU, unsigned int T3);
    std::vector<unsigned int> getMDsFromT3(const SDL::segments& segmentsInGPU, const SDL::triplets& tripletsInGPU, unsigned int T3);
    std::vector<unsigned int> getHitsFromT3(const SDL::miniDoublets& miniDoubletsInGPU, const SDL::segments& segmentsInGPU, const SDL::triplets& tripletsInGPU, unsigned int T3);
    // ----* T5 *----
    std::vector<unsigned int> getT3sFromT5(const SDL::quintuplets& quintupletsInGPU, unsigned int T5);
    std::vector<unsigned int> getLSsFromT5(const SDL::triplets& tripletsInGPU, const SDL::quintuplets& quintupletsInGPU, unsigned int T5);
    std::vector<unsigned int> getMDsFromT5(const SDL::segments& segmentsInGPU, const SDL::triplets& tripletsInGPU, const SDL::quintuplets& quintupletsInGPU, unsigned int T5);
    std::vector<unsigned int> getHitsFromT5(const SDL::miniDoublets& miniDoubletsInGPU, const SDL::segments& segmentsInGPU, const SDL::triplets& tripletsInGPU, const SDL::quintuplets& quintupletsInGPU, unsigned int T5);
    std::vector<unsigned int> getHitIdxsFromT5(const SDL::hits& hitsInGPU, const SDL::miniDoublets& miniDoubletsInGPU, const SDL::segments& segmentsInGPU, const SDL::triplets& tripletsInGPU, const SDL::quintuplets& quintupletsInGPU, unsigned int T5);
    // ----* pT3 *----
    unsigned int getPixelLSFrompT3(const SDL::modules& modulesInGPU, const SDL::objectRanges& rangesInGPU, const SDL::pixelTriplets& pixelTripletsInGPU, unsigned int pT3);
    unsigned int getT3FrompT3(const SDL::pixelTriplets& pixelTripletsInGPU, unsigned int pT3);
    std::vector<unsigned int> getOuterTrackerHitsFrompT3(const SDL::miniDoublets& miniDoubletsInGPU, const SDL::segments& segmentsInGPU, const SDL::triplets& tripletsInGPU, const SDL::pixelTriplets& pixelTripletsInGPU, unsigned int pT3);
    std::vector<unsigned int> getHitsFrompT3(const SDL::modules& modulesInGPU, const SDL::objectRanges& rangesInGPU, const SDL::miniDoublets& miniDoubletsInGPU, const SDL::segments& segmentsInGPU, const SDL::triplets& tripletsInGPU, const SDL::pixelTriplets& pixelTripletsInGPU, unsigned int pT3);
    std::vector<unsigned int> getHitIdxsFrompT3(const SDL::modules& modulesInGPU, const SDL::objectRanges& rangesInGPU, const SDL::hits& hitsInGPU, const SDL::miniDoublets& miniDoubletsInGPU, const SDL::segments& segmentsInGPU, const SDL::triplets& tripletsInGPU, const SDL::pixelTriplets& pixelTripletsInGPU, unsigned int pT3);
    // ----* pT5 *----
    unsigned int getPixelLSFrompT5(const SDL::modules& modulesInGPU, const SDL::objectRanges& rangesInGPU, const SDL::pixelQuintuplets& pixelQuintupletsInGPU, unsigned int pT5);
    unsigned int getT5FrompT5(const SDL::pixelQuintuplets& pixelQuintupletsInGPU, unsigned int pT5);
    std::vector<unsigned int> getT3sFrompT5(const SDL::quintuplets& quintupletsInGPU, const SDL::pixelQuintuplets& pixelQuintupletsInGPU, unsigned int pT5);
    std::vector<unsigned int> getOuterTrackerHitsFrompT5(const SDL::miniDoublets& miniDoubletsInGPU, const SDL::segments& segmentsInGPU, const SDL::triplets& tripletsInGPU, const SDL::quintuplets& quintupletsInGPU, const SDL::pixelQuintuplets& pixelQuintupletsInGPU, unsigned int pT5);
    std::vector<unsigned int> getHitsFrompT5(const SDL::modules& modulesInGPU, const SDL::objectRanges& rangesInGPU, const SDL::miniDoublets& miniDoubletsInGPU, const SDL::segments& segmentsInGPU, const SDL::triplets& tripletsInGPU, const SDL::quintuplets& quintupletsInGPU, const SDL::pixelQuintuplets& pixelQuintupletsInGPU, unsigned int pT5);
    std::vector<unsigned int> getHitIdxsFrompT5(const SDL::modules& modulesInGPU, const SDL::objectRanges& rangesInGPU, const SDL::hits& hitsInGPU, const SDL::miniDoublets& miniDoubletsInGPU, const SDL::segments& segmentsInGPU, const SDL::triplets& tripletsInGPU, const SDL::quintuplets& quintupletsInGPU, const SDL::pixelQuintuplets& pixelQuintupletsInGPU, unsigned int pT5);

    // Input and output vectors
    TString TrackLooperDir_;
    std::vector<float> in_trkX_;
    std::vector<float> in_trkY_;
    std::vector<float> in_trkZ_;
    std::vector<unsigned int> in_hitId_;
    std::vector<unsigned int> in_hitIdxs_;
    std::vector<unsigned int> in_hitIndices_vec0_;
    std::vector<unsigned int> in_hitIndices_vec1_;
    std::vector<unsigned int> in_hitIndices_vec2_;
    std::vector<unsigned int> in_hitIndices_vec3_;
    std::vector<float> in_deltaPhi_vec_;
    std::vector<float> in_ptIn_vec_;
    std::vector<float> in_ptErr_vec_;
    std::vector<float> in_px_vec_;
    std::vector<float> in_py_vec_;
    std::vector<float> in_pz_vec_;
    std::vector<float> in_eta_vec_;
    std::vector<float> in_etaErr_vec_;
    std::vector<float> in_phi_vec_;
    std::vector<int> in_charge_vec_;
    std::vector<unsigned int> in_seedIdx_vec_;
    std::vector<int> in_superbin_vec_;
    std::vector<int8_t> in_pixelType_vec_;
    std::vector<short> in_isQuad_vec_;
    std::vector<float> out_tc_pt_;
    std::vector<float> out_tc_eta_;
    std::vector<float> out_tc_phi_;
    std::vector<std::vector<unsigned int>> out_tc_hitIdxs_;
    std::vector<unsigned int> out_tc_len_;
    std::vector<int> out_tc_seedIdx_;
  };

} //namespace
