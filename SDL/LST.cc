#include "LST.h"

SDL::LST::LST() {
  TrackLooperDir_ = getenv("LST_BASE");
}

void SDL::LST::eventSetup() {
  loadMaps();
  TString path = get_absolute_path_after_check_file_exists(TString::Format("%s/data/centroid_CMSSW_12_2_0_pre2.txt",TrackLooperDir_.Data()).Data());
  SDL::initModules(path);
}

void SDL::LST::run(cudaStream_t stream,
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
                   const std::vector<float> ph2_z) {
  auto event = SDL::Event(stream);
  prepareInput(see_px, see_py, see_pz, see_dxy, see_dz, see_ptErr, see_etaErr, see_stateTrajGlbX, see_stateTrajGlbY, see_stateTrajGlbZ, see_stateTrajGlbPx, see_stateTrajGlbPy, see_stateTrajGlbPz, see_q, see_algo, see_hitIdx, ph2_detId, ph2_x, ph2_y, ph2_z);

  event.addHitToEvent(in_trkX_, in_trkY_, in_trkZ_, in_hitId_, in_hitIdxs_); // TODO : Need to fix the hitIdxs
  event.addPixelSegmentToEvent(in_hitIndices_vec0_, in_hitIndices_vec1_, in_hitIndices_vec2_, in_hitIndices_vec3_,
                               in_deltaPhi_vec_,
                               in_ptIn_vec_, in_ptErr_vec_,
                               in_px_vec_, in_py_vec_, in_pz_vec_,
                               in_eta_vec_, in_etaErr_vec_,
                               in_phi_vec_,
                               in_charge_vec_,
                               in_seedIdx_vec_,
                               in_superbin_vec_,
                               in_pixelType_vec_,
                               in_isQuad_vec_);
  event.createMiniDoublets();
  //printf("# of Mini-doublets produced: %d\n",event.getNumberOfMiniDoublets());
  //printf("# of Mini-doublets produced barrel layer 1: %d\n",event.getNumberOfMiniDoubletsByLayerBarrel(0));
  //printf("# of Mini-doublets produced barrel layer 2: %d\n",event.getNumberOfMiniDoubletsByLayerBarrel(1));
  //printf("# of Mini-doublets produced barrel layer 3: %d\n",event.getNumberOfMiniDoubletsByLayerBarrel(2));
  //printf("# of Mini-doublets produced barrel layer 4: %d\n",event.getNumberOfMiniDoubletsByLayerBarrel(3));
  //printf("# of Mini-doublets produced barrel layer 5: %d\n",event.getNumberOfMiniDoubletsByLayerBarrel(4));
  //printf("# of Mini-doublets produced barrel layer 6: %d\n",event.getNumberOfMiniDoubletsByLayerBarrel(5));
  //printf("# of Mini-doublets produced endcap layer 1: %d\n",event.getNumberOfMiniDoubletsByLayerEndcap(0));
  //printf("# of Mini-doublets produced endcap layer 2: %d\n",event.getNumberOfMiniDoubletsByLayerEndcap(1));
  //printf("# of Mini-doublets produced endcap layer 3: %d\n",event.getNumberOfMiniDoubletsByLayerEndcap(2));
  //printf("# of Mini-doublets produced endcap layer 4: %d\n",event.getNumberOfMiniDoubletsByLayerEndcap(3));
  //printf("# of Mini-doublets produced endcap layer 5: %d\n",event.getNumberOfMiniDoubletsByLayerEndcap(4));

  event.createSegmentsWithModuleMap();
  //printf("# of Segments produced: %d\n",event.getNumberOfSegments());
  //printf("# of Segments produced layer 1-2:  %d\n",event.getNumberOfSegmentsByLayerBarrel(0));
  //printf("# of Segments produced layer 2-3:  %d\n",event.getNumberOfSegmentsByLayerBarrel(1));
  //printf("# of Segments produced layer 3-4:  %d\n",event.getNumberOfSegmentsByLayerBarrel(2));
  //printf("# of Segments produced layer 4-5:  %d\n",event.getNumberOfSegmentsByLayerBarrel(3));
  //printf("# of Segments produced layer 5-6:  %d\n",event.getNumberOfSegmentsByLayerBarrel(4));
  //printf("# of Segments produced endcap layer 1:  %d\n",event.getNumberOfSegmentsByLayerEndcap(0));
  //printf("# of Segments produced endcap layer 2:  %d\n",event.getNumberOfSegmentsByLayerEndcap(1));
  //printf("# of Segments produced endcap layer 3:  %d\n",event.getNumberOfSegmentsByLayerEndcap(2));
  //printf("# of Segments produced endcap layer 4:  %d\n",event.getNumberOfSegmentsByLayerEndcap(3));
  //printf("# of Segments produced endcap layer 5:  %d\n",event.getNumberOfSegmentsByLayerEndcap(4));

  event.createTriplets();
  //printf("# of T3s produced: %d\n",event.getNumberOfTriplets());
  //printf("# of T3s produced layer 1-2-3: %d\n",event.getNumberOfTripletsByLayerBarrel(0));
  //printf("# of T3s produced layer 2-3-4: %d\n",event.getNumberOfTripletsByLayerBarrel(1));
  //printf("# of T3s produced layer 3-4-5: %d\n",event.getNumberOfTripletsByLayerBarrel(2));
  //printf("# of T3s produced layer 4-5-6: %d\n",event.getNumberOfTripletsByLayerBarrel(3));
  //printf("# of T3s produced endcap layer 1-2-3: %d\n",event.getNumberOfTripletsByLayerEndcap(0));
  //printf("# of T3s produced endcap layer 2-3-4: %d\n",event.getNumberOfTripletsByLayerEndcap(1));
  //printf("# of T3s produced endcap layer 3-4-5: %d\n",event.getNumberOfTripletsByLayerEndcap(2));
  //printf("# of T3s produced endcap layer 1: %d\n",event.getNumberOfTripletsByLayerEndcap(0));
  //printf("# of T3s produced endcap layer 2: %d\n",event.getNumberOfTripletsByLayerEndcap(1));
  //printf("# of T3s produced endcap layer 3: %d\n",event.getNumberOfTripletsByLayerEndcap(2));
  //printf("# of T3s produced endcap layer 4: %d\n",event.getNumberOfTripletsByLayerEndcap(3));
  //printf("# of T3s produced endcap layer 5: %d\n",event.getNumberOfTripletsByLayerEndcap(4));

  event.createQuintuplets();
  //printf("# of Quintuplets produced: %d\n",event.getNumberOfQuintuplets());
  //printf("# of Quintuplets produced layer 1-2-3-4-5-6: %d\n",event.getNumberOfQuintupletsByLayerBarrel(0));
  //printf("# of Quintuplets produced layer 2: %d\n",event.getNumberOfQuintupletsByLayerBarrel(1));
  //printf("# of Quintuplets produced layer 3: %d\n",event.getNumberOfQuintupletsByLayerBarrel(2));
  //printf("# of Quintuplets produced layer 4: %d\n",event.getNumberOfQuintupletsByLayerBarrel(3));
  //printf("# of Quintuplets produced layer 5: %d\n",event.getNumberOfQuintupletsByLayerBarrel(4));
  //printf("# of Quintuplets produced layer 6: %d\n",event.getNumberOfQuintupletsByLayerBarrel(5));
  //printf("# of Quintuplets produced endcap layer 1: %d\n",event.getNumberOfQuintupletsByLayerEndcap(0));
  //printf("# of Quintuplets produced endcap layer 2: %d\n",event.getNumberOfQuintupletsByLayerEndcap(1));
  //printf("# of Quintuplets produced endcap layer 3: %d\n",event.getNumberOfQuintupletsByLayerEndcap(2));
  //printf("# of Quintuplets produced endcap layer 4: %d\n",event.getNumberOfQuintupletsByLayerEndcap(3));
  //printf("# of Quintuplets produced endcap layer 5: %d\n",event.getNumberOfQuintupletsByLayerEndcap(4));

  event.pixelLineSegmentCleaning();

  event.createPixelQuintuplets();
  //printf("# of Pixel Quintuplets produced: %d\n",event.getNumberOfPixelQuintuplets());

  event.createPixelTriplets();
  //printf("# of Pixel T3s produced: %d\n",event.getNumberOfPixelTriplets());

  event.createTrackCandidates();
  //printf("# of TrackCandidates produced: %d\n",event.getNumberOfTrackCandidates());
  //printf("    # of Pixel TrackCandidates produced: %d\n",event.getNumberOfPixelTrackCandidates());
  //printf("    # of pT5 TrackCandidates produced: %d\n",event.getNumberOfPT5TrackCandidates());
  //printf("    # of pT3 TrackCandidates produced: %d\n",event.getNumberOfPT3TrackCandidates());
  //printf("    # of pLS TrackCandidates produced: %d\n",event.getNumberOfPLSTrackCandidates());
  //printf("    # of T5 TrackCandidates produced: %d\n",event.getNumberOfT5TrackCandidates());

  getOutput(event);
}


void SDL::LST::loadMaps() {
  // Module orientation information (DrDz or phi angles)
  TString endcap_geom = get_absolute_path_after_check_file_exists(TString::Format("%s/data/endcap_orientation_data_CMSSW_12_2_0_pre2.txt", TrackLooperDir_.Data()).Data());
  TString tilted_geom = get_absolute_path_after_check_file_exists(TString::Format("%s/data/tilted_orientation_data_CMSSW_12_2_0_pre2.txt", TrackLooperDir_.Data()).Data());
  SDL::endcapGeometry.load(endcap_geom.Data()); // centroid values added to the map
  SDL::tiltedGeometry.load(tilted_geom.Data());

  // Module connection map (for line segment building)
  TString mappath = get_absolute_path_after_check_file_exists(TString::Format("%s/data/module_connection_tracing_CMSSW_12_2_0_pre2_merged.txt", TrackLooperDir_.Data()).Data());
  SDL::moduleConnectionMap.load(mappath.Data());

  TString pLSMapDir = TrackLooperDir_+"/data/pixelmaps_CMSSW_12_2_0_pre2_0p8minPt";

  TString path;
  path = TString::Format("%s/pLS_map_layer1_subdet5.txt", pLSMapDir.Data()).Data(); SDL::moduleConnectionMap_pLStoLayer1Subdet5.load(get_absolute_path_after_check_file_exists(path.Data()).Data());
  path = TString::Format("%s/pLS_map_layer2_subdet5.txt", pLSMapDir.Data()).Data(); SDL::moduleConnectionMap_pLStoLayer2Subdet5.load(get_absolute_path_after_check_file_exists(path.Data()).Data());
  path = TString::Format("%s/pLS_map_layer1_subdet4.txt", pLSMapDir.Data()).Data(); SDL::moduleConnectionMap_pLStoLayer1Subdet4.load(get_absolute_path_after_check_file_exists(path.Data()).Data());
  path = TString::Format("%s/pLS_map_layer2_subdet4.txt", pLSMapDir.Data()).Data(); SDL::moduleConnectionMap_pLStoLayer2Subdet4.load(get_absolute_path_after_check_file_exists(path.Data()).Data());

  path = TString::Format("%s/pLS_map_neg_layer1_subdet5.txt", pLSMapDir.Data()).Data(); SDL::moduleConnectionMap_pLStoLayer1Subdet5_neg.load(get_absolute_path_after_check_file_exists(path.Data()).Data());
  path = TString::Format("%s/pLS_map_neg_layer2_subdet5.txt", pLSMapDir.Data()).Data(); SDL::moduleConnectionMap_pLStoLayer2Subdet5_neg.load(get_absolute_path_after_check_file_exists(path.Data()).Data());
  path = TString::Format("%s/pLS_map_neg_layer1_subdet4.txt", pLSMapDir.Data()).Data(); SDL::moduleConnectionMap_pLStoLayer1Subdet4_neg.load(get_absolute_path_after_check_file_exists(path.Data()).Data());
  path = TString::Format("%s/pLS_map_neg_layer2_subdet4.txt", pLSMapDir.Data()).Data(); SDL::moduleConnectionMap_pLStoLayer2Subdet4_neg.load(get_absolute_path_after_check_file_exists(path.Data()).Data());

  path = TString::Format("%s/pLS_map_pos_layer1_subdet5.txt", pLSMapDir.Data()).Data(); SDL::moduleConnectionMap_pLStoLayer1Subdet5_pos.load(get_absolute_path_after_check_file_exists(path.Data()).Data());
  path = TString::Format("%s/pLS_map_pos_layer2_subdet5.txt", pLSMapDir.Data()).Data(); SDL::moduleConnectionMap_pLStoLayer2Subdet5_pos.load(get_absolute_path_after_check_file_exists(path.Data()).Data());
  path = TString::Format("%s/pLS_map_pos_layer1_subdet4.txt", pLSMapDir.Data()).Data(); SDL::moduleConnectionMap_pLStoLayer1Subdet4_pos.load(get_absolute_path_after_check_file_exists(path.Data()).Data());
  path = TString::Format("%s/pLS_map_pos_layer2_subdet4.txt", pLSMapDir.Data()).Data(); SDL::moduleConnectionMap_pLStoLayer2Subdet4_pos.load(get_absolute_path_after_check_file_exists(path.Data()).Data());
}

TString SDL::LST::get_absolute_path_after_check_file_exists(const std::string name) {
  std::filesystem::path fullpath = std::filesystem::absolute(name.c_str());
  if (not std::filesystem::exists(fullpath))
  {
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
                            const std::vector<unsigned int> see_algo,
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
  std::vector<short> isQuad_vec;
  std::iota(hitIdxs.begin(), hitIdxs.end(), 0);
  const int hit_size = trkX.size();

  for (auto &&[iSeed, _] : iter::enumerate(see_stateTrajGlbPx)) {
    bool good_seed_type = false;
    if (see_algo[iSeed] == 4) good_seed_type = true;
    if (see_algo[iSeed] == 22) good_seed_type = true;
    if (not good_seed_type) continue;

    ROOT::Math::PxPyPzMVector p3LH(see_stateTrajGlbPx[iSeed], see_stateTrajGlbPy[iSeed], see_stateTrajGlbPz[iSeed], 0);
    ROOT::Math::XYZVector p3LH_helper(see_stateTrajGlbPx[iSeed], see_stateTrajGlbPy[iSeed], see_stateTrajGlbPz[iSeed]);
    float ptIn = p3LH.Pt();
    float eta = p3LH.Eta();
    float ptErr = see_ptErr[iSeed];

    if ((ptIn > 0.8 - 2 * ptErr)) {
      ROOT::Math::XYZVector r3LH(see_stateTrajGlbX[iSeed], see_stateTrajGlbY[iSeed], see_stateTrajGlbZ[iSeed]);
      ROOT::Math::PxPyPzMVector p3PCA(see_px[iSeed], see_py[iSeed], see_pz[iSeed], 0);
      ROOT::Math::XYZVector r3PCA(calculateR3FromPCA(p3PCA, see_dxy[iSeed], see_dz[iSeed]));

      float pixelSegmentDeltaPhiChange = (r3LH-p3LH_helper).Phi();
      float etaErr = see_etaErr[iSeed];
      float px = p3LH.Px();
      float py = p3LH.Py();
      float pz = p3LH.Pz();

      int charge = see_q[iSeed];
      int pixtype = -1;

      if (ptIn >= 2.0) pixtype = 0;
      else if (ptIn >= (0.8 - 2 * ptErr) and ptIn < 2.0) {
        if (pixelSegmentDeltaPhiChange >= 0) pixtype =1;
        else pixtype = 2;
      }
      else continue;

      unsigned int hitIdx0 = hit_size + count;
      count++; 
      unsigned int hitIdx1 = hit_size + count;
      count++;
      unsigned int hitIdx2 = hit_size + count;
      count++;
      unsigned int hitIdx3;
      if (see_hitIdx[iSeed].size() <= 3) hitIdx3 = hitIdx2;
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
      if(see_hitIdx[iSeed].size() > 3) {
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
      bool isQuad = false;
      if(see_hitIdx[iSeed].size() > 3) {
        isQuad = true;
        hitIdxs.push_back(see_hitIdx[iSeed][3]);
      }
      float neta = 25.;
      float nphi = 72.;
      float nz = 25.;
      int etabin = (p3PCA_Eta + 2.6) / ((2*2.6)/neta);
      int phibin = (p3PCA_Phi + 3.14159265358979323846) / ((2.*3.14159265358979323846) / nphi);
      int dzbin = (see_dz[iSeed] + 30) / (2*30 / nz);
      int isuperbin = (nz * nphi) * etabin + (nz) * phibin + dzbin;
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

ROOT::Math::XYZVector SDL::LST::calculateR3FromPCA(const ROOT::Math::PxPyPzMVector& p3, const float dxy, const float dz) {
  const float pt = p3.Pt();
  const float p = p3.P();
  const float vz = dz*pt*pt/p/p;

  const float vx = -dxy*p3.y()/pt - p3.x()/p*p3.z()/p*dz;
  const float vy =  dxy*p3.x()/pt - p3.y()/p*p3.z()/p*dz;
  return ROOT::Math::XYZVector(vx, vy, vz);
}

void SDL::LST::getOutput(SDL::Event& event) {
  std::vector<float> tc_pt_, tc_eta_, tc_phi_;
  std::vector<std::vector<unsigned int>> tc_hitIdxs_;
  std::vector<int> tc_len_, tc_seedIdx_;

  SDL::trackCandidates& trackCandidatesInGPU = (*event.getTrackCandidates());
  SDL::triplets& tripletsInGPU = (*event.getTriplets());
  SDL::segments& segmentsInGPU = (*event.getSegments());
  SDL::hits& hitsInGPU = (*event.getHits());

  const float kRinv1GeVf = (2.99792458e-3 * 3.8);
  const float k2Rinv1GeVf = kRinv1GeVf / 2.;

  unsigned int nTrackCandidates = *trackCandidatesInGPU.nTrackCandidates;
  for (unsigned int idx = 0; idx < nTrackCandidates; idx++) {
    int hit_array_length = 0;
    short trackCandidateType = trackCandidatesInGPU.trackCandidateType[idx];

    std::vector<unsigned int> hit_idx;
    float pt, eta, phi;
    int seedIdx = -1;
    if (trackCandidateType == 8) { // pLS
      unsigned int pLS = trackCandidatesInGPU.directObjectIndices[idx];

      pt = segmentsInGPU.ptIn[pLS];
      eta = segmentsInGPU.eta[pLS];
      phi = segmentsInGPU.phi[pLS];
      seedIdx = segmentsInGPU.seedIdx[pLS];

      hit_idx = getPixelHitIdxsFrompLS(event, pLS);
      hit_array_length = hit_idx.size();
    }
    else if (trackCandidateType == 5) { // pT3
      unsigned int pT3 = trackCandidatesInGPU.directObjectIndices[idx];

      std::vector<unsigned int> Hits = getOuterTrackerHitsFrompT3(event, pT3);
      unsigned int Hit_0 = Hits[0];
      unsigned int Hit_4 = Hits[4];

      unsigned int T3 = getT3FrompT3(event, pT3);
      unsigned int pLS = getPixelLSFrompT3(event, pT3);

      const float dr = sqrt(pow(hitsInGPU.xs[Hit_4] - hitsInGPU.xs[Hit_0], 2) + pow(hitsInGPU.ys[Hit_4] - hitsInGPU.ys[Hit_0], 2));

      float betaIn   = __H2F(tripletsInGPU.betaIn[T3]);
      float betaOut  = __H2F(tripletsInGPU.betaOut[T3]);

      const float pt_T3 = abs(dr * k2Rinv1GeVf / sin((betaIn + betaOut) / 2.));

      const float pt_pLS = segmentsInGPU.ptIn[pLS];
      eta = segmentsInGPU.eta[pLS];
      phi = segmentsInGPU.phi[pLS];
      seedIdx = segmentsInGPU.seedIdx[pLS];

      pt = (pt_pLS + pt_T3) / 2.;

      hit_idx = getHitIdxsFrompT3(event, pT3);
      hit_array_length = hit_idx.size();
    }
    else if (trackCandidateType == 7) { // pT5
      unsigned int pT5 = trackCandidatesInGPU.directObjectIndices[idx];

      std::vector<unsigned int> Hits = getOuterTrackerHitsFrompT5(event, pT5);
      unsigned int Hit_0 = Hits[0];
      unsigned int Hit_4 = Hits[4];
      unsigned int Hit_8 = Hits[8];

      std::vector<unsigned int> T3s = getT3sFrompT5(event, pT5);
      unsigned int T3_0 = T3s[0];
      unsigned int T3_1 = T3s[1];

      unsigned int pLS = getPixelLSFrompT5(event, pT5);

      const float dr_in = sqrt(pow(hitsInGPU.xs[Hit_4] - hitsInGPU.xs[Hit_0], 2) + pow(hitsInGPU.ys[Hit_4] - hitsInGPU.ys[Hit_0], 2));
      const float dr_out = sqrt(pow(hitsInGPU.xs[Hit_8] - hitsInGPU.xs[Hit_4], 2) + pow(hitsInGPU.ys[Hit_8] - hitsInGPU.ys[Hit_4], 2));

      float betaIn_in   = __H2F(tripletsInGPU.betaIn[T3_0]);
      float betaOut_in  = __H2F(tripletsInGPU.betaOut[T3_0]);
      float betaIn_out  = __H2F(tripletsInGPU.betaIn[T3_1]);
      float betaOut_out = __H2F(tripletsInGPU.betaOut[T3_1]);

      const float ptAv_in = abs(dr_in * k2Rinv1GeVf / sin((betaIn_in + betaOut_in) / 2.));
      const float ptAv_out = abs(dr_out * k2Rinv1GeVf / sin((betaIn_out + betaOut_out) / 2.));
      const float pt_T5 = (ptAv_in + ptAv_out) / 2.;

      const float pt_pLS = segmentsInGPU.ptIn[pLS];
      eta = segmentsInGPU.eta[pLS];
      phi = segmentsInGPU.phi[pLS];
      seedIdx = segmentsInGPU.seedIdx[pLS];

      pt = (pt_pLS + pt_T5) / 2.;

      hit_idx = getHitIdxsFrompT5(event, pT5);
      hit_array_length = hit_idx.size();
    }
    else if (trackCandidateType == 4) { // T5
      unsigned int T5 = trackCandidatesInGPU.directObjectIndices[idx];
      std::vector<unsigned int> T3s = getT3sFromT5(event, T5);
      std::vector<unsigned int> hits = getHitsFromT5(event, T5);

      unsigned int Hit_0 = hits[0];
      unsigned int Hit_4 = hits[4];
      unsigned int Hit_8 = hits[8];

      const float dr_in = sqrt(pow(hitsInGPU.xs[Hit_4] - hitsInGPU.xs[Hit_0], 2) + pow(hitsInGPU.ys[Hit_4] - hitsInGPU.ys[Hit_0], 2));
      const float dr_out = sqrt(pow(hitsInGPU.xs[Hit_8] - hitsInGPU.xs[Hit_4], 2) + pow(hitsInGPU.ys[Hit_8] - hitsInGPU.ys[Hit_4], 2));

      float betaIn_in   = __H2F(tripletsInGPU.betaIn [T3s[0]]);
      float betaOut_in  = __H2F(tripletsInGPU.betaOut[T3s[0]]);
      float betaIn_out  = __H2F(tripletsInGPU.betaIn [T3s[1]]);
      float betaOut_out = __H2F(tripletsInGPU.betaOut[T3s[1]]);

      const float ptAv_in = abs(dr_in * k2Rinv1GeVf / sin((betaIn_in + betaOut_in) / 2.));
      const float ptAv_out = abs(dr_out * k2Rinv1GeVf / sin((betaIn_out + betaOut_out) / 2.));

      pt = (ptAv_in + ptAv_out) / 2.;

      eta = SDL::eta(in_trkX_[Hit_8], in_trkY_[Hit_8], in_trkZ_[Hit_8]); // eta from outermost hit
      phi = SDL::phi(in_trkX_[Hit_0], in_trkY_[Hit_0]); // phi from innermost hit

      hit_idx = getHitIdxsFromT5(event, T5);
      hit_array_length = hit_idx.size();
    }

    tc_pt_.push_back(pt);
    tc_eta_.push_back(eta);
    tc_phi_.push_back(phi);
    tc_hitIdxs_.push_back(hit_idx);
    tc_len_.push_back(hit_array_length);
    tc_seedIdx_.push_back(seedIdx);
  }
  out_tc_pt_ = tc_pt_;
  out_tc_eta_ = tc_eta_;
  out_tc_phi_ = tc_phi_;
  out_tc_hitIdxs_ = tc_hitIdxs_;
  out_tc_len_ = tc_len_;
  out_tc_seedIdx_ = tc_seedIdx_;

  //for(auto out : out_tc_pt_) printf("%f\n",out);
  //printf("\n");
  //for(auto out : out_tc_eta_) printf("%f\n",out);
  //printf("\n");
  //for(auto out : out_tc_phi_) printf("%f\n",out);
  //printf("\n");
  //for(auto out : out_tc_len_) printf("%d\n",out);
  //printf("\n");
  //for(auto out : out_tc_seedIdx_) printf("%d\n",out);
}

// Object accessors
// ===============
// ----* pLS *----
// ===============

//____________________________________________________________________________________________
std::vector<unsigned int> SDL::LST::getPixelHitsFrompLS(SDL::Event& event, unsigned int pLS) {
  SDL::segments& segments_ = *(event.getSegments());
  SDL::miniDoublets& miniDoublets_ = *(event.getMiniDoublets());
  SDL::objectRanges& rangesInGPU = (*event.getRanges());
  SDL::modules& modulesInGPU = (*event.getModules());
  const unsigned int pLS_offset = rangesInGPU.segmentModuleIndices[*(modulesInGPU.nLowerModules)];
  unsigned int MD_1 = segments_.mdIndices[2 * (pLS + pLS_offset)];
  unsigned int MD_2 = segments_.mdIndices[2 * (pLS + pLS_offset) + 1];
  unsigned int hit_1 = miniDoublets_.anchorHitIndices[MD_1];
  unsigned int hit_2 = miniDoublets_.outerHitIndices [MD_1];
  unsigned int hit_3 = miniDoublets_.anchorHitIndices[MD_2];
  unsigned int hit_4 = miniDoublets_.outerHitIndices [MD_2];
  if (hit_3 == hit_4)
    return {hit_1, hit_2, hit_3};
  else
    return {hit_1, hit_2, hit_3, hit_4};
}

//____________________________________________________________________________________________
std::vector<unsigned int> SDL::LST::getPixelHitIdxsFrompLS(SDL::Event& event, unsigned int pLS) {
  SDL::hits& hitsInGPU = *(event.getHits());
  std::vector<unsigned int> hits = getPixelHitsFrompLS(event, pLS);
  std::vector<unsigned int> hitidxs;
  for (auto& hit : hits)
    hitidxs.push_back(hitsInGPU.idxs[hit]);
  return hitidxs;
}

//____________________________________________________________________________________________
std::vector<unsigned int> SDL::LST::getPixelHitTypesFrompLS(SDL::Event& event, unsigned int pLS) {
  std::vector<unsigned int> hits = getPixelHitsFrompLS(event, pLS);
  std::vector<unsigned int> hittypes(hits.size(), 0);
  return hittypes;
}

// ==============
// ----* MD *----
// ==============

//____________________________________________________________________________________________
std::vector<unsigned int> SDL::LST::getHitsFromMD(SDL::Event& event, unsigned int MD) {
  SDL::miniDoublets& miniDoublets_ = *(event.getMiniDoublets());
  unsigned int hit_1 = miniDoublets_.anchorHitIndices[MD];
  unsigned int hit_2 = miniDoublets_.outerHitIndices [MD];
  return {hit_1, hit_2};
}

// ==============
// ----* LS *----
// ==============

//____________________________________________________________________________________________
std::vector<unsigned int> SDL::LST::getMDsFromLS(SDL::Event& event, unsigned int LS) {
  SDL::segments& segments_ = *(event.getSegments());
  unsigned int MD_1 = segments_.mdIndices[2 * LS];
  unsigned int MD_2 = segments_.mdIndices[2 * LS + 1];
  return {MD_1, MD_2};
}

//____________________________________________________________________________________________
std::vector<unsigned int> SDL::LST::getHitsFromLS(SDL::Event& event, unsigned int LS) {
  std::vector<unsigned int> MDs = getMDsFromLS(event, LS);
  std::vector<unsigned int> hits_0 = getHitsFromMD(event, MDs[0]);
  std::vector<unsigned int> hits_1 = getHitsFromMD(event, MDs[1]);
  return {hits_0[0], hits_0[1], hits_1[0], hits_1[1]};
}

// ==============
// ----* T3 *----
// ==============

//____________________________________________________________________________________________
std::vector<unsigned int> SDL::LST::getLSsFromT3(SDL::Event& event, unsigned int T3) {
  SDL::triplets& triplets_ = *(event.getTriplets());
  unsigned int LS_1 = triplets_.segmentIndices[2 * T3];
  unsigned int LS_2 = triplets_.segmentIndices[2 * T3 + 1];
  return {LS_1, LS_2};
}

//____________________________________________________________________________________________
std::vector<unsigned int> SDL::LST::getMDsFromT3(SDL::Event& event, unsigned int T3) {
  std::vector<unsigned int> LSs = getLSsFromT3(event, T3);
  std::vector<unsigned int> MDs_0 = getMDsFromLS(event, LSs[0]);
  std::vector<unsigned int> MDs_1 = getMDsFromLS(event, LSs[1]);
  return {MDs_0[0], MDs_0[1], MDs_1[1]};
}

//____________________________________________________________________________________________
std::vector<unsigned int> SDL::LST::getHitsFromT3(SDL::Event& event, unsigned int T3) {
  std::vector<unsigned int> MDs = getMDsFromT3(event, T3);
  std::vector<unsigned int> hits_0 = getHitsFromMD(event, MDs[0]);
  std::vector<unsigned int> hits_1 = getHitsFromMD(event, MDs[1]);
  std::vector<unsigned int> hits_2 = getHitsFromMD(event, MDs[2]);
  return {hits_0[0], hits_0[1], hits_1[0], hits_1[1], hits_2[0], hits_2[1]};
}

// ==============
// ----* T5 *----
// ==============

//____________________________________________________________________________________________
std::vector<unsigned int> SDL::LST::getT3sFromT5(SDL::Event& event, unsigned int T5) {
  SDL::quintuplets& quintuplets_ = *(event.getQuintuplets());
  unsigned int T3_1 = quintuplets_.tripletIndices[2 * T5];
  unsigned int T3_2 = quintuplets_.tripletIndices[2 * T5 + 1];
  return {T3_1, T3_2};
}

//____________________________________________________________________________________________
std::vector<unsigned int> SDL::LST::getLSsFromT5(SDL::Event& event, unsigned int T5) {
  std::vector<unsigned int> T3s = getT3sFromT5(event, T5);
  std::vector<unsigned int> LSs_0 = getLSsFromT3(event, T3s[0]);
  std::vector<unsigned int> LSs_1 = getLSsFromT3(event, T3s[1]);
  return {LSs_0[0], LSs_0[1], LSs_1[0], LSs_1[1]};
}

//____________________________________________________________________________________________
std::vector<unsigned int> SDL::LST::getMDsFromT5(SDL::Event& event, unsigned int T5) {
  std::vector<unsigned int> LSs = getLSsFromT5(event, T5);
  std::vector<unsigned int> MDs_0 = getMDsFromLS(event, LSs[0]);
  std::vector<unsigned int> MDs_1 = getMDsFromLS(event, LSs[1]);
  std::vector<unsigned int> MDs_2 = getMDsFromLS(event, LSs[2]);
  std::vector<unsigned int> MDs_3 = getMDsFromLS(event, LSs[3]);
  return {MDs_0[0], MDs_0[1], MDs_1[1], MDs_2[1], MDs_3[1]};
}

//____________________________________________________________________________________________
std::vector<unsigned int> SDL::LST::getHitsFromT5(SDL::Event& event, unsigned int T5) {
  std::vector<unsigned int> MDs = getMDsFromT5(event, T5);
  std::vector<unsigned int> hits_0 = getHitsFromMD(event, MDs[0]);
  std::vector<unsigned int> hits_1 = getHitsFromMD(event, MDs[1]);
  std::vector<unsigned int> hits_2 = getHitsFromMD(event, MDs[2]);
  std::vector<unsigned int> hits_3 = getHitsFromMD(event, MDs[3]);
  std::vector<unsigned int> hits_4 = getHitsFromMD(event, MDs[4]);
  return {hits_0[0], hits_0[1], hits_1[0], hits_1[1], hits_2[0], hits_2[1], hits_3[0], hits_3[1], hits_4[0], hits_4[1]};
}

//____________________________________________________________________________________________
std::vector<unsigned int> SDL::LST::getHitIdxsFromT5(SDL::Event& event, unsigned int T5) {
  SDL::hits& hitsInGPU = *(event.getHits());
  std::vector<unsigned int> hits = getHitsFromT5(event, T5);
  std::vector<unsigned int> hitidxs;
  for (auto& hit : hits)
    hitidxs.push_back(hitsInGPU.idxs[hit]);
  return hitidxs;
}

//____________________________________________________________________________________________
std::vector<unsigned int> SDL::LST::getHitTypesFromT5(SDL::Event& event, unsigned int T5) {
  return {4, 4, 4, 4, 4, 4, 4, 4, 4, 4};;
}

// ===============
// ----* pT3 *----
// ===============

//____________________________________________________________________________________________
unsigned int SDL::LST::getPixelLSFrompT3(SDL::Event& event, unsigned int pT3) {
  SDL::pixelTriplets& pixelTriplets_ = *(event.getPixelTriplets());
  SDL::objectRanges& rangesInGPU = (*event.getRanges());
  SDL::modules& modulesInGPU = (*event.getModules());
  const unsigned int pLS_offset = rangesInGPU.segmentModuleIndices[*(modulesInGPU.nLowerModules)];
  return pixelTriplets_.pixelSegmentIndices[pT3] - pLS_offset;
}

//____________________________________________________________________________________________
unsigned int SDL::LST::getT3FrompT3(SDL::Event& event, unsigned int pT3) {
  SDL::pixelTriplets& pixelTriplets_ = *(event.getPixelTriplets());
  return pixelTriplets_.tripletIndices[pT3];
}

//____________________________________________________________________________________________
std::vector<unsigned int> SDL::LST::getLSsFrompT3(SDL::Event& event, unsigned int pT3) {
  unsigned int T3 = getT3FrompT3(event, pT3);
  return getLSsFromT3(event, T3);
}

//____________________________________________________________________________________________
std::vector<unsigned int> SDL::LST::getMDsFrompT3(SDL::Event& event, unsigned int pT3) {
  unsigned int T3 = getT3FrompT3(event, pT3);
  return getMDsFromT3(event, T3);
}

//____________________________________________________________________________________________
std::vector<unsigned int> SDL::LST::getOuterTrackerHitsFrompT3(SDL::Event& event, unsigned int pT3) {
  unsigned int T3 = getT3FrompT3(event, pT3);
  return getHitsFromT3(event, T3);
}

//____________________________________________________________________________________________
std::vector<unsigned int> SDL::LST::getPixelHitsFrompT3(SDL::Event& event, unsigned int pT3) {
  unsigned int pLS = getPixelLSFrompT3(event, pT3);
  return getPixelHitsFrompLS(event, pLS);
}

//____________________________________________________________________________________________
std::vector<unsigned int> SDL::LST::getHitsFrompT3(SDL::Event& event, unsigned int pT3) {
  unsigned int pLS = getPixelLSFrompT3(event, pT3);
  unsigned int T3 = getT3FrompT3(event, pT3);
  std::vector<unsigned int> pixelHits = getPixelHitsFrompLS(event, pLS);
  std::vector<unsigned int> outerTrackerHits = getHitsFromT3(event, T3);
  pixelHits.insert(pixelHits.end(), outerTrackerHits.begin(), outerTrackerHits.end());
  return pixelHits;
}

//____________________________________________________________________________________________
std::vector<unsigned int> SDL::LST::getHitIdxsFrompT3(SDL::Event& event, unsigned int pT3) {
  SDL::hits& hitsInGPU = *(event.getHits());
  std::vector<unsigned int> hits = getHitsFrompT3(event, pT3);
  std::vector<unsigned int> hitidxs;
  for (auto& hit : hits)
    hitidxs.push_back(hitsInGPU.idxs[hit]);
  return hitidxs;
}

//____________________________________________________________________________________________
std::vector<unsigned int> SDL::LST::getHitTypesFrompT3(SDL::Event& event, unsigned int pT3) {
  unsigned int pLS = getPixelLSFrompT3(event, pT3);
  std::vector<unsigned int> pixelHits = getPixelHitsFrompLS(event, pLS);
  // pixel Hits list will be either 3 or 4 and depending on it return accordingly
  if (pixelHits.size() == 3)
    return {0, 0, 0, 4, 4, 4, 4, 4, 4};
  else
    return {0, 0, 0, 0, 4, 4, 4, 4, 4, 4};
}

// ===============
// ----* pT5 *----
// ===============

//____________________________________________________________________________________________
unsigned int SDL::LST::getPixelLSFrompT5(SDL::Event& event, unsigned int pT5) {
  SDL::pixelQuintuplets& pixelQuintuplets_ = *(event.getPixelQuintuplets());
  SDL::objectRanges& rangesInGPU = (*event.getRanges());
  SDL::modules& modulesInGPU = (*event.getModules());
  const unsigned int pLS_offset = rangesInGPU.segmentModuleIndices[*(modulesInGPU.nLowerModules)];
  return pixelQuintuplets_.pixelIndices[pT5] - pLS_offset;
}

//____________________________________________________________________________________________
unsigned int SDL::LST::getT5FrompT5(SDL::Event& event, unsigned int pT5) {
  SDL::pixelQuintuplets& pixelQuintuplets_ = *(event.getPixelQuintuplets());
  return pixelQuintuplets_.T5Indices[pT5];
}

//____________________________________________________________________________________________
std::vector<unsigned int> SDL::LST::getT3sFrompT5(SDL::Event& event, unsigned int pT5) {
  unsigned int T5 = getT5FrompT5(event, pT5);
  return getT3sFromT5(event, T5);
}

//____________________________________________________________________________________________
std::vector<unsigned int> SDL::LST::getLSsFrompT5(SDL::Event& event, unsigned int pT5) {
  unsigned int T5 = getT5FrompT5(event, pT5);
  return getLSsFromT5(event, T5);
}

//____________________________________________________________________________________________
std::vector<unsigned int> SDL::LST::getMDsFrompT5(SDL::Event& event, unsigned int pT5) {
  unsigned int T5 = getT5FrompT5(event, pT5);
  return getMDsFromT5(event, T5);
}

//____________________________________________________________________________________________
std::vector<unsigned int> SDL::LST::getOuterTrackerHitsFrompT5(SDL::Event& event, unsigned int pT5) {
  unsigned int T5 = getT5FrompT5(event, pT5);
  return getHitsFromT5(event, T5);
}

//____________________________________________________________________________________________
std::vector<unsigned int> SDL::LST::getPixelHitsFrompT5(SDL::Event& event, unsigned int pT5) {
  unsigned int pLS = getPixelLSFrompT5(event, pT5);
  return getPixelHitsFrompLS(event, pLS);
}


//____________________________________________________________________________________________
std::vector<unsigned int> SDL::LST::getHitsFrompT5(SDL::Event& event, unsigned int pT5) {
  unsigned int pLS = getPixelLSFrompT5(event, pT5);
  unsigned int T5 = getT5FrompT5(event, pT5);
  std::vector<unsigned int> pixelHits = getPixelHitsFrompLS(event, pLS);
  std::vector<unsigned int> outerTrackerHits = getHitsFromT5(event, T5);
  pixelHits.insert(pixelHits.end(), outerTrackerHits.begin(), outerTrackerHits.end());
  return pixelHits;
}

//____________________________________________________________________________________________
std::vector<unsigned int> SDL::LST::getHitIdxsFrompT5(SDL::Event& event, unsigned int pT5) {
  SDL::hits& hitsInGPU = *(event.getHits());
  std::vector<unsigned int> hits = getHitsFrompT5(event, pT5);
  std::vector<unsigned int> hitidxs;
  for (auto& hit : hits)
    hitidxs.push_back(hitsInGPU.idxs[hit]);
  return hitidxs;
}

//____________________________________________________________________________________________
std::vector<unsigned int> SDL::LST::getHitTypesFrompT5(SDL::Event& event, unsigned int pT5) {
  unsigned int pLS = getPixelLSFrompT5(event, pT5);
  std::vector<unsigned int> pixelHits = getPixelHitsFrompLS(event, pLS);
  // pixel Hits list will be either 3 or 4 and depending on it return accordingly
  if (pixelHits.size() == 3)
    return {0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
  else
    return {0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
}
