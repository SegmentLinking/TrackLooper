#include "LST.h"

namespace {
  TString trackLooperDir() { return getenv("LST_BASE"); }
}  // namespace

void SDL::LST::eventSetup() {
  static std::once_flag mapsLoaded;
  std::call_once(mapsLoaded, &SDL::LST::loadMaps);
  TString path = get_absolute_path_after_check_file_exists(
      TString::Format("%s/data/centroid_CMSSW_12_2_0_pre2.txt", trackLooperDir().Data()).Data());
  static std::once_flag modulesInited;
  std::call_once(modulesInited, SDL::initModules, path);
}

void SDL::LST::loadMaps() {
  // Module orientation information (DrDz or phi angles)
  auto const& tldir = trackLooperDir();
  TString endcap_geom = get_absolute_path_after_check_file_exists(
      TString::Format("%s/data/endcap_orientation_data_CMSSW_12_2_0_pre2.txt", trackLooperDir().Data()).Data());
  TString tilted_geom = get_absolute_path_after_check_file_exists(
      TString::Format("%s/data/tilted_orientation_data_CMSSW_12_2_0_pre2.txt", trackLooperDir().Data()).Data());
  SDL::endcapGeometry->load(endcap_geom.Data());  // centroid values added to the map
  SDL::tiltedGeometry.load(tilted_geom.Data());

  // Module connection map (for line segment building)
  TString mappath = get_absolute_path_after_check_file_exists(
      TString::Format("%s/data/module_connection_tracing_CMSSW_12_2_0_pre2_merged.txt", trackLooperDir().Data()).Data());
  SDL::moduleConnectionMap.load(mappath.Data());

  TString pLSMapDir = trackLooperDir() + "/data/pixelmaps_CMSSW_12_2_0_pre2_0p8minPt/pLS_map";
  std::string connects[] = {"_layer1_subdet5", "_layer2_subdet5", "_layer1_subdet4", "_layer2_subdet4"};
  TString path;

  for (std::string& connect : connects) {
    auto connectData = connect.data();

    path = TString::Format("%s%s.txt", pLSMapDir.Data(), connectData).Data();
    SDL::moduleConnectionMap_pLStoLayer.emplace_back(
        ModuleConnectionMap(get_absolute_path_after_check_file_exists(path.Data()).Data()));

    path = TString::Format("%s_pos%s.txt", pLSMapDir.Data(), connectData).Data();
    SDL::moduleConnectionMap_pLStoLayer_pos.emplace_back(
        ModuleConnectionMap(get_absolute_path_after_check_file_exists(path.Data()).Data()));

    path = TString::Format("%s_neg%s.txt", pLSMapDir.Data(), connectData).Data();
    SDL::moduleConnectionMap_pLStoLayer_neg.emplace_back(
        ModuleConnectionMap(get_absolute_path_after_check_file_exists(path.Data()).Data()));
  }
}

TString SDL::LST::get_absolute_path_after_check_file_exists(const std::string name) {
  std::filesystem::path fullpath = std::filesystem::absolute(name.c_str());
  if (not std::filesystem::exists(fullpath)) {
    std::cout << "ERROR: Could not find the file = " << fullpath << std::endl;
    exit(2);
  }
  return TString(fullpath.string().c_str());
}

void SDL::LST::prepareInput(const std::vector<float> see_px,
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
  unsigned int count = 0;
  auto n_see = see_stateTrajGlbPx.size();
  std::vector<float> px_vec;
  px_vec.reserve(n_see);
  std::vector<float> py_vec;
  py_vec.reserve(n_see);
  std::vector<float> pz_vec;
  pz_vec.reserve(n_see);
  std::vector<unsigned int> hitIndices_vec0;
  hitIndices_vec0.reserve(n_see);
  std::vector<unsigned int> hitIndices_vec1;
  hitIndices_vec1.reserve(n_see);
  std::vector<unsigned int> hitIndices_vec2;
  hitIndices_vec2.reserve(n_see);
  std::vector<unsigned int> hitIndices_vec3;
  hitIndices_vec3.reserve(n_see);
  std::vector<float> ptIn_vec;
  ptIn_vec.reserve(n_see);
  std::vector<float> ptErr_vec;
  ptErr_vec.reserve(n_see);
  std::vector<float> etaErr_vec;
  etaErr_vec.reserve(n_see);
  std::vector<float> eta_vec;
  eta_vec.reserve(n_see);
  std::vector<float> phi_vec;
  phi_vec.reserve(n_see);
  std::vector<int> charge_vec;
  charge_vec.reserve(n_see);
  std::vector<unsigned int> seedIdx_vec;
  seedIdx_vec.reserve(n_see);
  std::vector<float> deltaPhi_vec;
  deltaPhi_vec.reserve(n_see);
  std::vector<float> trkX = ph2_x;
  std::vector<float> trkY = ph2_y;
  std::vector<float> trkZ = ph2_z;
  std::vector<unsigned int> hitId = ph2_detId;
  std::vector<unsigned int> hitIdxs(ph2_detId.size());

  std::vector<int> superbin_vec;
  std::vector<int8_t> pixelType_vec;
  std::vector<char> isQuad_vec;
  std::iota(hitIdxs.begin(), hitIdxs.end(), 0);
  const int hit_size = trkX.size();

  for (auto&& [iSeed, _] : iter::enumerate(see_stateTrajGlbPx)) {
    ROOT::Math::PxPyPzMVector p3LH(see_stateTrajGlbPx[iSeed], see_stateTrajGlbPy[iSeed], see_stateTrajGlbPz[iSeed], 0);
    ROOT::Math::XYZVector p3LH_helper(see_stateTrajGlbPx[iSeed], see_stateTrajGlbPy[iSeed], see_stateTrajGlbPz[iSeed]);
    float ptIn = p3LH.Pt();
    float eta = p3LH.Eta();
    float ptErr = see_ptErr[iSeed];

    if ((ptIn > 0.8 - 2 * ptErr)) {
      ROOT::Math::XYZVector r3LH(see_stateTrajGlbX[iSeed], see_stateTrajGlbY[iSeed], see_stateTrajGlbZ[iSeed]);
      ROOT::Math::PxPyPzMVector p3PCA(see_px[iSeed], see_py[iSeed], see_pz[iSeed], 0);
      ROOT::Math::XYZVector r3PCA(calculateR3FromPCA(p3PCA, see_dxy[iSeed], see_dz[iSeed]));

      float pixelSegmentDeltaPhiChange = (r3LH - p3LH_helper).Phi();
      float etaErr = see_etaErr[iSeed];
      float px = p3LH.Px();
      float py = p3LH.Py();
      float pz = p3LH.Pz();

      int charge = see_q[iSeed];
      int pixtype = -1;

      if (ptIn >= 2.0)
        pixtype = 0;
      else if (ptIn >= (0.8 - 2 * ptErr) and ptIn < 2.0) {
        if (pixelSegmentDeltaPhiChange >= 0)
          pixtype = 1;
        else
          pixtype = 2;
      } else
        continue;

      unsigned int hitIdx0 = hit_size + count;
      count++;
      unsigned int hitIdx1 = hit_size + count;
      count++;
      unsigned int hitIdx2 = hit_size + count;
      count++;
      unsigned int hitIdx3;
      if (see_hitIdx[iSeed].size() <= 3)
        hitIdx3 = hitIdx2;
      else {
        hitIdx3 = hit_size + count;
        count++;
      }

      trkX.push_back(r3PCA.X());
      trkY.push_back(r3PCA.Y());
      trkZ.push_back(r3PCA.Z());
      trkX.push_back(p3PCA.Pt());
      float p3PCA_Eta = p3PCA.Eta();
      trkY.push_back(p3PCA_Eta);
      float p3PCA_Phi = p3PCA.Phi();
      trkZ.push_back(p3PCA_Phi);
      trkX.push_back(r3LH.X());
      trkY.push_back(r3LH.Y());
      trkZ.push_back(r3LH.Z());
      hitId.push_back(1);
      hitId.push_back(1);
      hitId.push_back(1);
      if (see_hitIdx[iSeed].size() > 3) {
        trkX.push_back(r3LH.X());
        trkY.push_back(see_dxy[iSeed]);
        trkZ.push_back(see_dz[iSeed]);
        hitId.push_back(1);
      }
      px_vec.push_back(px);
      py_vec.push_back(py);
      pz_vec.push_back(pz);

      hitIndices_vec0.push_back(hitIdx0);
      hitIndices_vec1.push_back(hitIdx1);
      hitIndices_vec2.push_back(hitIdx2);
      hitIndices_vec3.push_back(hitIdx3);
      ptIn_vec.push_back(ptIn);
      ptErr_vec.push_back(ptErr);
      etaErr_vec.push_back(etaErr);
      eta_vec.push_back(eta);
      float phi = p3LH.Phi();
      phi_vec.push_back(phi);
      charge_vec.push_back(charge);
      seedIdx_vec.push_back(iSeed);
      deltaPhi_vec.push_back(pixelSegmentDeltaPhiChange);

      hitIdxs.push_back(see_hitIdx[iSeed][0]);
      hitIdxs.push_back(see_hitIdx[iSeed][1]);
      hitIdxs.push_back(see_hitIdx[iSeed][2]);
      char isQuad = false;
      if (see_hitIdx[iSeed].size() > 3) {
        isQuad = true;
        hitIdxs.push_back(see_hitIdx[iSeed][3]);
      }
      float neta = 25.;
      float nphi = 72.;
      float nz = 25.;
      int etabin = (p3PCA_Eta + 2.6) / ((2 * 2.6) / neta);
      int phibin = (p3PCA_Phi + 3.14159265358979323846) / ((2. * 3.14159265358979323846) / nphi);
      int dzbin = (see_dz[iSeed] + 30) / (2 * 30 / nz);
      int isuperbin = (nz * nphi) * etabin + (nz)*phibin + dzbin;
      superbin_vec.push_back(isuperbin);
      pixelType_vec.push_back(pixtype);
      isQuad_vec.push_back(isQuad);
    }
  }

  in_trkX_ = trkX;
  in_trkY_ = trkY;
  in_trkZ_ = trkZ;
  in_hitId_ = hitId;
  in_hitIdxs_ = hitIdxs;
  in_hitIndices_vec0_ = hitIndices_vec0;
  in_hitIndices_vec1_ = hitIndices_vec1;
  in_hitIndices_vec2_ = hitIndices_vec2;
  in_hitIndices_vec3_ = hitIndices_vec3;
  in_deltaPhi_vec_ = deltaPhi_vec;
  in_ptIn_vec_ = ptIn_vec;
  in_ptErr_vec_ = ptErr_vec;
  in_px_vec_ = px_vec;
  in_py_vec_ = py_vec;
  in_pz_vec_ = pz_vec;
  in_eta_vec_ = eta_vec;
  in_etaErr_vec_ = etaErr_vec;
  in_phi_vec_ = phi_vec;
  in_charge_vec_ = charge_vec;
  in_seedIdx_vec_ = seedIdx_vec;
  in_superbin_vec_ = superbin_vec;
  in_pixelType_vec_ = pixelType_vec;
  in_isQuad_vec_ = isQuad_vec;
}

ROOT::Math::XYZVector SDL::LST::calculateR3FromPCA(const ROOT::Math::PxPyPzMVector& p3,
                                                   const float dxy,
                                                   const float dz) {
  const float pt = p3.Pt();
  const float p = p3.P();
  const float vz = dz * pt * pt / p / p;

  const float vx = -dxy * p3.y() / pt - p3.x() / p * p3.z() / p * dz;
  const float vy = dxy * p3.x() / pt - p3.y() / p * p3.z() / p * dz;
  return ROOT::Math::XYZVector(vx, vy, vz);
}

void SDL::LST::getOutput(SDL::Event& event) {
  std::vector<std::vector<unsigned int>> tc_hitIdxs_;
  std::vector<unsigned int> tc_len_;
  std::vector<int> tc_seedIdx_;
  std::vector<short> tc_trackCandidateType_;

  SDL::hitsBuffer<alpaka::DevCpu>& hitsInGPU = (*event.getHitsInCMSSW());
  SDL::trackCandidatesBuffer<alpaka::DevCpu>& trackCandidatesInGPU = (*event.getTrackCandidatesInCMSSW());

  unsigned int nTrackCandidates = *trackCandidatesInGPU.nTrackCandidates;
  for (unsigned int idx = 0; idx < nTrackCandidates; idx++) {
    short trackCandidateType = trackCandidatesInGPU.trackCandidateType[idx];
    std::vector<unsigned int> hit_idx =
        getHitIdxs(trackCandidateType, idx, trackCandidatesInGPU.hitIndices, hitsInGPU.idxs);

    tc_hitIdxs_.push_back(hit_idx);
    tc_len_.push_back(hit_idx.size());
    tc_seedIdx_.push_back(trackCandidatesInGPU.pixelSeedIndex[idx]);
    tc_trackCandidateType_.push_back(trackCandidateType);
  }

  out_tc_hitIdxs_ = tc_hitIdxs_;
  out_tc_len_ = tc_len_;
  out_tc_seedIdx_ = tc_seedIdx_;
  out_tc_trackCandidateType_ = tc_trackCandidateType_;
}

std::vector<unsigned int> SDL::LST::getHitIdxs(const short trackCandidateType,
                                               const unsigned int TCIdx,
                                               const unsigned int* TCHitIndices,
                                               const unsigned int* hitIndices) {
  std::vector<unsigned int> hits;

  unsigned int maxNHits = 0;
  if (trackCandidateType == 7)
    maxNHits = 14;  // pT5
  else if (trackCandidateType == 5)
    maxNHits = 10;  // pT3
  else if (trackCandidateType == 4)
    maxNHits = 10;  // T5
  else if (trackCandidateType == 8)
    maxNHits = 4;  // pLS

  for (unsigned int i = 0; i < maxNHits; i++) {
    unsigned int hitIdxInGPU = TCHitIndices[14 * TCIdx + i];
    unsigned int hitIdx =
        (trackCandidateType == 8)
            ? hitIdxInGPU
            : hitIndices[hitIdxInGPU];  // Hit indices are stored differently in the standalone for pLS.

    // For p objects, the 3rd and 4th hit maybe the same,
    // due to the way pLS hits are stored in the standalone.
    // This is because pixel seeds can be either triplets or quadruplets.
    if (trackCandidateType != 4 && hits.size() == 3 && hits.back() == hitIdx)  // Remove duplicate 4th hits.
      continue;

    hits.push_back(hitIdx);
  }

  return hits;
}
