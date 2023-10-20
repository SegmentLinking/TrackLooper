#ifndef LST_H
#define LST_H

#include <filesystem>
#include <cstdlib>
#include <numeric>
#include <mutex>

#include "code/cppitertools/enumerate.hpp"

#include "TString.h"
#include "Math/Vector3D.h"
#include <Math/Vector4D.h>

#include "Event.h"

namespace SDL {
  class LST {
  public:
    LST() = default;

    void eventSetup();
    template <typename TQueue>
    void run(TQueue& queue,
             bool verbose,
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
             const std::vector<std::vector<int>> see_hitIdx,
             const std::vector<unsigned int> ph2_detId,
             const std::vector<float> ph2_x,
             const std::vector<float> ph2_y,
             const std::vector<float> ph2_z) {
      auto event = SDL::Event(verbose, queue);
      prepareInput(see_px,
                   see_py,
                   see_pz,
                   see_dxy,
                   see_dz,
                   see_ptErr,
                   see_etaErr,
                   see_stateTrajGlbX,
                   see_stateTrajGlbY,
                   see_stateTrajGlbZ,
                   see_stateTrajGlbPx,
                   see_stateTrajGlbPy,
                   see_stateTrajGlbPz,
                   see_q,
                   see_hitIdx,
                   ph2_detId,
                   ph2_x,
                   ph2_y,
                   ph2_z);

      event.addHitToEvent(in_trkX_, in_trkY_, in_trkZ_, in_hitId_, in_hitIdxs_);
      event.addPixelSegmentToEvent(in_hitIndices_vec0_,
                                   in_hitIndices_vec1_,
                                   in_hitIndices_vec2_,
                                   in_hitIndices_vec3_,
                                   in_deltaPhi_vec_,
                                   in_ptIn_vec_,
                                   in_ptErr_vec_,
                                   in_px_vec_,
                                   in_py_vec_,
                                   in_pz_vec_,
                                   in_eta_vec_,
                                   in_etaErr_vec_,
                                   in_phi_vec_,
                                   in_charge_vec_,
                                   in_seedIdx_vec_,
                                   in_superbin_vec_,
                                   in_pixelType_vec_,
                                   in_isQuad_vec_);
      event.createMiniDoublets();
      if (verbose) {
        printf("# of Mini-doublets produced: %d\n", event.getNumberOfMiniDoublets());
        printf("# of Mini-doublets produced barrel layer 1: %d\n", event.getNumberOfMiniDoubletsByLayerBarrel(0));
        printf("# of Mini-doublets produced barrel layer 2: %d\n", event.getNumberOfMiniDoubletsByLayerBarrel(1));
        printf("# of Mini-doublets produced barrel layer 3: %d\n", event.getNumberOfMiniDoubletsByLayerBarrel(2));
        printf("# of Mini-doublets produced barrel layer 4: %d\n", event.getNumberOfMiniDoubletsByLayerBarrel(3));
        printf("# of Mini-doublets produced barrel layer 5: %d\n", event.getNumberOfMiniDoubletsByLayerBarrel(4));
        printf("# of Mini-doublets produced barrel layer 6: %d\n", event.getNumberOfMiniDoubletsByLayerBarrel(5));
        printf("# of Mini-doublets produced endcap layer 1: %d\n", event.getNumberOfMiniDoubletsByLayerEndcap(0));
        printf("# of Mini-doublets produced endcap layer 2: %d\n", event.getNumberOfMiniDoubletsByLayerEndcap(1));
        printf("# of Mini-doublets produced endcap layer 3: %d\n", event.getNumberOfMiniDoubletsByLayerEndcap(2));
        printf("# of Mini-doublets produced endcap layer 4: %d\n", event.getNumberOfMiniDoubletsByLayerEndcap(3));
        printf("# of Mini-doublets produced endcap layer 5: %d\n", event.getNumberOfMiniDoubletsByLayerEndcap(4));
      }

      event.createSegmentsWithModuleMap();
      if (verbose) {
        printf("# of Segments produced: %d\n", event.getNumberOfSegments());
        printf("# of Segments produced layer 1-2:  %d\n", event.getNumberOfSegmentsByLayerBarrel(0));
        printf("# of Segments produced layer 2-3:  %d\n", event.getNumberOfSegmentsByLayerBarrel(1));
        printf("# of Segments produced layer 3-4:  %d\n", event.getNumberOfSegmentsByLayerBarrel(2));
        printf("# of Segments produced layer 4-5:  %d\n", event.getNumberOfSegmentsByLayerBarrel(3));
        printf("# of Segments produced layer 5-6:  %d\n", event.getNumberOfSegmentsByLayerBarrel(4));
        printf("# of Segments produced endcap layer 1:  %d\n", event.getNumberOfSegmentsByLayerEndcap(0));
        printf("# of Segments produced endcap layer 2:  %d\n", event.getNumberOfSegmentsByLayerEndcap(1));
        printf("# of Segments produced endcap layer 3:  %d\n", event.getNumberOfSegmentsByLayerEndcap(2));
        printf("# of Segments produced endcap layer 4:  %d\n", event.getNumberOfSegmentsByLayerEndcap(3));
        printf("# of Segments produced endcap layer 5:  %d\n", event.getNumberOfSegmentsByLayerEndcap(4));
      }

      event.createTriplets();
      if (verbose) {
        printf("# of T3s produced: %d\n", event.getNumberOfTriplets());
        printf("# of T3s produced layer 1-2-3: %d\n", event.getNumberOfTripletsByLayerBarrel(0));
        printf("# of T3s produced layer 2-3-4: %d\n", event.getNumberOfTripletsByLayerBarrel(1));
        printf("# of T3s produced layer 3-4-5: %d\n", event.getNumberOfTripletsByLayerBarrel(2));
        printf("# of T3s produced layer 4-5-6: %d\n", event.getNumberOfTripletsByLayerBarrel(3));
        printf("# of T3s produced endcap layer 1-2-3: %d\n", event.getNumberOfTripletsByLayerEndcap(0));
        printf("# of T3s produced endcap layer 2-3-4: %d\n", event.getNumberOfTripletsByLayerEndcap(1));
        printf("# of T3s produced endcap layer 3-4-5: %d\n", event.getNumberOfTripletsByLayerEndcap(2));
        printf("# of T3s produced endcap layer 1: %d\n", event.getNumberOfTripletsByLayerEndcap(0));
        printf("# of T3s produced endcap layer 2: %d\n", event.getNumberOfTripletsByLayerEndcap(1));
        printf("# of T3s produced endcap layer 3: %d\n", event.getNumberOfTripletsByLayerEndcap(2));
        printf("# of T3s produced endcap layer 4: %d\n", event.getNumberOfTripletsByLayerEndcap(3));
        printf("# of T3s produced endcap layer 5: %d\n", event.getNumberOfTripletsByLayerEndcap(4));
      }

      event.createQuintuplets();
      if (verbose) {
        printf("# of Quintuplets produced: %d\n", event.getNumberOfQuintuplets());
        printf("# of Quintuplets produced layer 1-2-3-4-5-6: %d\n", event.getNumberOfQuintupletsByLayerBarrel(0));
        printf("# of Quintuplets produced layer 2: %d\n", event.getNumberOfQuintupletsByLayerBarrel(1));
        printf("# of Quintuplets produced layer 3: %d\n", event.getNumberOfQuintupletsByLayerBarrel(2));
        printf("# of Quintuplets produced layer 4: %d\n", event.getNumberOfQuintupletsByLayerBarrel(3));
        printf("# of Quintuplets produced layer 5: %d\n", event.getNumberOfQuintupletsByLayerBarrel(4));
        printf("# of Quintuplets produced layer 6: %d\n", event.getNumberOfQuintupletsByLayerBarrel(5));
        printf("# of Quintuplets produced endcap layer 1: %d\n", event.getNumberOfQuintupletsByLayerEndcap(0));
        printf("# of Quintuplets produced endcap layer 2: %d\n", event.getNumberOfQuintupletsByLayerEndcap(1));
        printf("# of Quintuplets produced endcap layer 3: %d\n", event.getNumberOfQuintupletsByLayerEndcap(2));
        printf("# of Quintuplets produced endcap layer 4: %d\n", event.getNumberOfQuintupletsByLayerEndcap(3));
        printf("# of Quintuplets produced endcap layer 5: %d\n", event.getNumberOfQuintupletsByLayerEndcap(4));
      }

      event.pixelLineSegmentCleaning();

      event.createPixelQuintuplets();
      if (verbose)
        printf("# of Pixel Quintuplets produced: %d\n", event.getNumberOfPixelQuintuplets());

      event.createPixelTriplets();
      if (verbose)
        printf("# of Pixel T3s produced: %d\n", event.getNumberOfPixelTriplets());

      event.createTrackCandidates();
      if (verbose) {
        printf("# of TrackCandidates produced: %d\n", event.getNumberOfTrackCandidates());
        printf("        # of Pixel TrackCandidates produced: %d\n", event.getNumberOfPixelTrackCandidates());
        printf("        # of pT5 TrackCandidates produced: %d\n", event.getNumberOfPT5TrackCandidates());
        printf("        # of pT3 TrackCandidates produced: %d\n", event.getNumberOfPT3TrackCandidates());
        printf("        # of pLS TrackCandidates produced: %d\n", event.getNumberOfPLSTrackCandidates());
        printf("        # of T5 TrackCandidates produced: %d\n", event.getNumberOfT5TrackCandidates());
      }

      getOutput(event);

      event.resetEvent();
    }
    std::vector<std::vector<unsigned int>> hits() { return out_tc_hitIdxs_; }
    std::vector<unsigned int> len() { return out_tc_len_; }
    std::vector<int> seedIdx() { return out_tc_seedIdx_; }
    std::vector<short> trackCandidateType() { return out_tc_trackCandidateType_; }
    static void loadMaps();

  private:
    static TString get_absolute_path_after_check_file_exists(const std::string name);
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
                      const std::vector<std::vector<int>> see_hitIdx,
                      const std::vector<unsigned int> ph2_detId,
                      const std::vector<float> ph2_x,
                      const std::vector<float> ph2_y,
                      const std::vector<float> ph2_z);

    ROOT::Math::XYZVector calculateR3FromPCA(const ROOT::Math::PxPyPzMVector& p3, const float dxy, const float dz);

    void getOutput(SDL::Event& event);
    std::vector<unsigned int> getHitIdxs(const short trackCandidateType,
                                         const unsigned int TCIdx,
                                         const unsigned int* TCHitIndices,
                                         const unsigned int* hitIndices);

    // Input and output vectors
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
    std::vector<char> in_isQuad_vec_;
    std::vector<std::vector<unsigned int>> out_tc_hitIdxs_;
    std::vector<unsigned int> out_tc_len_;
    std::vector<int> out_tc_seedIdx_;
    std::vector<short> out_tc_trackCandidateType_;
  };

}  // namespace SDL

#endif
