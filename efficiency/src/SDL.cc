#include "SDL.h"
SDL sdl;

void SDL::Init(TTree *tree) {
  tree->SetMakeClass(1);
  SimTrack_pt_branch = 0;
  if (tree->GetBranch("SimTrack_pt") != 0) {
    SimTrack_pt_branch = tree->GetBranch("SimTrack_pt");
    if (SimTrack_pt_branch) { SimTrack_pt_branch->SetAddress(&SimTrack_pt_); }
  }
  SimTrack_eta_branch = 0;
  if (tree->GetBranch("SimTrack_eta") != 0) {
    SimTrack_eta_branch = tree->GetBranch("SimTrack_eta");
    if (SimTrack_eta_branch) { SimTrack_eta_branch->SetAddress(&SimTrack_eta_); }
  }
  SimTrack_phi_branch = 0;
  if (tree->GetBranch("SimTrack_phi") != 0) {
    SimTrack_phi_branch = tree->GetBranch("SimTrack_phi");
    if (SimTrack_phi_branch) { SimTrack_phi_branch->SetAddress(&SimTrack_phi_); }
  }
  SimTrack_dxy_branch = 0;
  if (tree->GetBranch("SimTrack_dxy") != 0) {
    SimTrack_dxy_branch = tree->GetBranch("SimTrack_dxy");
    if (SimTrack_dxy_branch) { SimTrack_dxy_branch->SetAddress(&SimTrack_dxy_); }
  }
  SimTrack_dz_branch = 0;
  if (tree->GetBranch("SimTrack_dz") != 0) {
    SimTrack_dz_branch = tree->GetBranch("SimTrack_dz");
    if (SimTrack_dz_branch) { SimTrack_dz_branch->SetAddress(&SimTrack_dz_); }
  }
  SimTrack_charge_branch = 0;
  if (tree->GetBranch("SimTrack_charge") != 0) {
    SimTrack_charge_branch = tree->GetBranch("SimTrack_charge");
    if (SimTrack_charge_branch) { SimTrack_charge_branch->SetAddress(&SimTrack_charge_); }
  }
  SimTrack_ppVtx_branch = 0;
  if (tree->GetBranch("SimTrack_ppVtx") != 0) {
    SimTrack_ppVtx_branch = tree->GetBranch("SimTrack_ppVtx");
    if (SimTrack_ppVtx_branch) { SimTrack_ppVtx_branch->SetAddress(&SimTrack_ppVtx_); }
  }
  SimTrack_pdgID_branch = 0;
  if (tree->GetBranch("SimTrack_pdgID") != 0) {
    SimTrack_pdgID_branch = tree->GetBranch("SimTrack_pdgID");
    if (SimTrack_pdgID_branch) { SimTrack_pdgID_branch->SetAddress(&SimTrack_pdgID_); }
  }
  SimTrack_vx_branch = 0;
  if (tree->GetBranch("SimTrack_vx") != 0) {
    SimTrack_vx_branch = tree->GetBranch("SimTrack_vx");
    if (SimTrack_vx_branch) { SimTrack_vx_branch->SetAddress(&SimTrack_vx_); }
  }
  SimTrack_vy_branch = 0;
  if (tree->GetBranch("SimTrack_vy") != 0) {
    SimTrack_vy_branch = tree->GetBranch("SimTrack_vy");
    if (SimTrack_vy_branch) { SimTrack_vy_branch->SetAddress(&SimTrack_vy_); }
  }
  SimTrack_vz_branch = 0;
  if (tree->GetBranch("SimTrack_vz") != 0) {
    SimTrack_vz_branch = tree->GetBranch("SimTrack_vz");
    if (SimTrack_vz_branch) { SimTrack_vz_branch->SetAddress(&SimTrack_vz_); }
  }
  SimTrack_TC_idx_branch = 0;
  if (tree->GetBranch("SimTrack_TC_idx") != 0) {
    SimTrack_TC_idx_branch = tree->GetBranch("SimTrack_TC_idx");
    if (SimTrack_TC_idx_branch) { SimTrack_TC_idx_branch->SetAddress(&SimTrack_TC_idx_); }
  }
  SimTrack_TC_typemask_branch = 0;
  if (tree->GetBranch("SimTrack_TC_typemask") != 0) {
    SimTrack_TC_typemask_branch = tree->GetBranch("SimTrack_TC_typemask");
    if (SimTrack_TC_typemask_branch) { SimTrack_TC_typemask_branch->SetAddress(&SimTrack_TC_typemask_); }
  }
  TC_pt_branch = 0;
  if (tree->GetBranch("TC_pt") != 0) {
    TC_pt_branch = tree->GetBranch("TC_pt");
    if (TC_pt_branch) { TC_pt_branch->SetAddress(&TC_pt_); }
  }
  TC_eta_branch = 0;
  if (tree->GetBranch("TC_eta") != 0) {
    TC_eta_branch = tree->GetBranch("TC_eta");
    if (TC_eta_branch) { TC_eta_branch->SetAddress(&TC_eta_); }
  }
  TC_phi_branch = 0;
  if (tree->GetBranch("TC_phi") != 0) {
    TC_phi_branch = tree->GetBranch("TC_phi");
    if (TC_phi_branch) { TC_phi_branch->SetAddress(&TC_phi_); }
  }
  TC_dxy_branch = 0;
  if (tree->GetBranch("TC_dxy") != 0) {
    TC_dxy_branch = tree->GetBranch("TC_dxy");
    if (TC_dxy_branch) { TC_dxy_branch->SetAddress(&TC_dxy_); }
  }
  TC_dz_branch = 0;
  if (tree->GetBranch("TC_dz") != 0) {
    TC_dz_branch = tree->GetBranch("TC_dz");
    if (TC_dz_branch) { TC_dz_branch->SetAddress(&TC_dz_); }
  }
  TC_charge_branch = 0;
  if (tree->GetBranch("TC_charge") != 0) {
    TC_charge_branch = tree->GetBranch("TC_charge");
    if (TC_charge_branch) { TC_charge_branch->SetAddress(&TC_charge_); }
  }
  TC_type_branch = 0;
  if (tree->GetBranch("TC_type") != 0) {
    TC_type_branch = tree->GetBranch("TC_type");
    if (TC_type_branch) { TC_type_branch->SetAddress(&TC_type_); }
  }
  TC_isFake_branch = 0;
  if (tree->GetBranch("TC_isFake") != 0) {
    TC_isFake_branch = tree->GetBranch("TC_isFake");
    if (TC_isFake_branch) { TC_isFake_branch->SetAddress(&TC_isFake_); }
  }
  TC_isDuplicate_branch = 0;
  if (tree->GetBranch("TC_isDuplicate") != 0) {
    TC_isDuplicate_branch = tree->GetBranch("TC_isDuplicate");
    if (TC_isDuplicate_branch) { TC_isDuplicate_branch->SetAddress(&TC_isDuplicate_); }
  }
  TC_SimTrack_idx_branch = 0;
  if (tree->GetBranch("TC_SimTrack_idx") != 0) {
    TC_SimTrack_idx_branch = tree->GetBranch("TC_SimTrack_idx");
    if (TC_SimTrack_idx_branch) { TC_SimTrack_idx_branch->SetAddress(&TC_SimTrack_idx_); }
  }
  sim_pt_branch = 0;
  if (tree->GetBranch("sim_pt") != 0) {
    sim_pt_branch = tree->GetBranch("sim_pt");
    if (sim_pt_branch) { sim_pt_branch->SetAddress(&sim_pt_); }
  }
  sim_eta_branch = 0;
  if (tree->GetBranch("sim_eta") != 0) {
    sim_eta_branch = tree->GetBranch("sim_eta");
    if (sim_eta_branch) { sim_eta_branch->SetAddress(&sim_eta_); }
  }
  sim_phi_branch = 0;
  if (tree->GetBranch("sim_phi") != 0) {
    sim_phi_branch = tree->GetBranch("sim_phi");
    if (sim_phi_branch) { sim_phi_branch->SetAddress(&sim_phi_); }
  }
  sim_pca_dxy_branch = 0;
  if (tree->GetBranch("sim_pca_dxy") != 0) {
    sim_pca_dxy_branch = tree->GetBranch("sim_pca_dxy");
    if (sim_pca_dxy_branch) { sim_pca_dxy_branch->SetAddress(&sim_pca_dxy_); }
  }
  sim_pca_dz_branch = 0;
  if (tree->GetBranch("sim_pca_dz") != 0) {
    sim_pca_dz_branch = tree->GetBranch("sim_pca_dz");
    if (sim_pca_dz_branch) { sim_pca_dz_branch->SetAddress(&sim_pca_dz_); }
  }
  sim_q_branch = 0;
  if (tree->GetBranch("sim_q") != 0) {
    sim_q_branch = tree->GetBranch("sim_q");
    if (sim_q_branch) { sim_q_branch->SetAddress(&sim_q_); }
  }
  sim_event_branch = 0;
  if (tree->GetBranch("sim_event") != 0) {
    sim_event_branch = tree->GetBranch("sim_event");
    if (sim_event_branch) { sim_event_branch->SetAddress(&sim_event_); }
  }
  sim_pdgId_branch = 0;
  if (tree->GetBranch("sim_pdgId") != 0) {
    sim_pdgId_branch = tree->GetBranch("sim_pdgId");
    if (sim_pdgId_branch) { sim_pdgId_branch->SetAddress(&sim_pdgId_); }
  }
  sim_vx_branch = 0;
  if (tree->GetBranch("sim_vx") != 0) {
    sim_vx_branch = tree->GetBranch("sim_vx");
    if (sim_vx_branch) { sim_vx_branch->SetAddress(&sim_vx_); }
  }
  sim_vy_branch = 0;
  if (tree->GetBranch("sim_vy") != 0) {
    sim_vy_branch = tree->GetBranch("sim_vy");
    if (sim_vy_branch) { sim_vy_branch->SetAddress(&sim_vy_); }
  }
  sim_vz_branch = 0;
  if (tree->GetBranch("sim_vz") != 0) {
    sim_vz_branch = tree->GetBranch("sim_vz");
    if (sim_vz_branch) { sim_vz_branch->SetAddress(&sim_vz_); }
  }
  sim_trkNtupIdx_branch = 0;
  if (tree->GetBranch("sim_trkNtupIdx") != 0) {
    sim_trkNtupIdx_branch = tree->GetBranch("sim_trkNtupIdx");
    if (sim_trkNtupIdx_branch) { sim_trkNtupIdx_branch->SetAddress(&sim_trkNtupIdx_); }
  }
  sim_TC_matched_branch = 0;
  if (tree->GetBranch("sim_TC_matched") != 0) {
    sim_TC_matched_branch = tree->GetBranch("sim_TC_matched");
    if (sim_TC_matched_branch) { sim_TC_matched_branch->SetAddress(&sim_TC_matched_); }
  }
  sim_TC_matched_mask_branch = 0;
  if (tree->GetBranch("sim_TC_matched_mask") != 0) {
    sim_TC_matched_mask_branch = tree->GetBranch("sim_TC_matched_mask");
    if (sim_TC_matched_mask_branch) { sim_TC_matched_mask_branch->SetAddress(&sim_TC_matched_mask_); }
  }
  tc_pt_branch = 0;
  if (tree->GetBranch("tc_pt") != 0) {
    tc_pt_branch = tree->GetBranch("tc_pt");
    if (tc_pt_branch) { tc_pt_branch->SetAddress(&tc_pt_); }
  }
  tc_eta_branch = 0;
  if (tree->GetBranch("tc_eta") != 0) {
    tc_eta_branch = tree->GetBranch("tc_eta");
    if (tc_eta_branch) { tc_eta_branch->SetAddress(&tc_eta_); }
  }
  tc_phi_branch = 0;
  if (tree->GetBranch("tc_phi") != 0) {
    tc_phi_branch = tree->GetBranch("tc_phi");
    if (tc_phi_branch) { tc_phi_branch->SetAddress(&tc_phi_); }
  }
  tc_type_branch = 0;
  if (tree->GetBranch("tc_type") != 0) {
    tc_type_branch = tree->GetBranch("tc_type");
    if (tc_type_branch) { tc_type_branch->SetAddress(&tc_type_); }
  }
  tc_isFake_branch = 0;
  if (tree->GetBranch("tc_isFake") != 0) {
    tc_isFake_branch = tree->GetBranch("tc_isFake");
    if (tc_isFake_branch) { tc_isFake_branch->SetAddress(&tc_isFake_); }
  }
  tc_isDuplicate_branch = 0;
  if (tree->GetBranch("tc_isDuplicate") != 0) {
    tc_isDuplicate_branch = tree->GetBranch("tc_isDuplicate");
    if (tc_isDuplicate_branch) { tc_isDuplicate_branch->SetAddress(&tc_isDuplicate_); }
  }
  tc_matched_simIdx_branch = 0;
  if (tree->GetBranch("tc_matched_simIdx") != 0) {
    tc_matched_simIdx_branch = tree->GetBranch("tc_matched_simIdx");
    if (tc_matched_simIdx_branch) { tc_matched_simIdx_branch->SetAddress(&tc_matched_simIdx_); }
  }
  sim_dummy_branch = 0;
  if (tree->GetBranch("sim_dummy") != 0) {
    sim_dummy_branch = tree->GetBranch("sim_dummy");
    if (sim_dummy_branch) { sim_dummy_branch->SetAddress(&sim_dummy_); }
  }
  tc_dummy_branch = 0;
  if (tree->GetBranch("tc_dummy") != 0) {
    tc_dummy_branch = tree->GetBranch("tc_dummy");
    if (tc_dummy_branch) { tc_dummy_branch->SetAddress(&tc_dummy_); }
  }
  pT5_matched_simIdx_branch = 0;
  if (tree->GetBranch("pT5_matched_simIdx") != 0) {
    pT5_matched_simIdx_branch = tree->GetBranch("pT5_matched_simIdx");
    if (pT5_matched_simIdx_branch) { pT5_matched_simIdx_branch->SetAddress(&pT5_matched_simIdx_); }
  }
  pT5_hitIdxs_branch = 0;
  if (tree->GetBranch("pT5_hitIdxs") != 0) {
    pT5_hitIdxs_branch = tree->GetBranch("pT5_hitIdxs");
    if (pT5_hitIdxs_branch) { pT5_hitIdxs_branch->SetAddress(&pT5_hitIdxs_); }
  }
  sim_pT5_matched_branch = 0;
  if (tree->GetBranch("sim_pT5_matched") != 0) {
    sim_pT5_matched_branch = tree->GetBranch("sim_pT5_matched");
    if (sim_pT5_matched_branch) { sim_pT5_matched_branch->SetAddress(&sim_pT5_matched_); }
  }
  pT5_pt_branch = 0;
  if (tree->GetBranch("pT5_pt") != 0) {
    pT5_pt_branch = tree->GetBranch("pT5_pt");
    if (pT5_pt_branch) { pT5_pt_branch->SetAddress(&pT5_pt_); }
  }
  pT5_eta_branch = 0;
  if (tree->GetBranch("pT5_eta") != 0) {
    pT5_eta_branch = tree->GetBranch("pT5_eta");
    if (pT5_eta_branch) { pT5_eta_branch->SetAddress(&pT5_eta_); }
  }
  pT5_phi_branch = 0;
  if (tree->GetBranch("pT5_phi") != 0) {
    pT5_phi_branch = tree->GetBranch("pT5_phi");
    if (pT5_phi_branch) { pT5_phi_branch->SetAddress(&pT5_phi_); }
  }
  pT5_isFake_branch = 0;
  if (tree->GetBranch("pT5_isFake") != 0) {
    pT5_isFake_branch = tree->GetBranch("pT5_isFake");
    if (pT5_isFake_branch) { pT5_isFake_branch->SetAddress(&pT5_isFake_); }
  }
  pT5_isDuplicate_branch = 0;
  if (tree->GetBranch("pT5_isDuplicate") != 0) {
    pT5_isDuplicate_branch = tree->GetBranch("pT5_isDuplicate");
    if (pT5_isDuplicate_branch) { pT5_isDuplicate_branch->SetAddress(&pT5_isDuplicate_); }
  }
  pT5_score_branch = 0;
  if (tree->GetBranch("pT5_score") != 0) {
    pT5_score_branch = tree->GetBranch("pT5_score");
    if (pT5_score_branch) { pT5_score_branch->SetAddress(&pT5_score_); }
  }
  pT5_layer_binary_branch = 0;
  if (tree->GetBranch("pT5_layer_binary") != 0) {
    pT5_layer_binary_branch = tree->GetBranch("pT5_layer_binary");
    if (pT5_layer_binary_branch) { pT5_layer_binary_branch->SetAddress(&pT5_layer_binary_); }
  }
  pT5_moduleType_binary_branch = 0;
  if (tree->GetBranch("pT5_moduleType_binary") != 0) {
    pT5_moduleType_binary_branch = tree->GetBranch("pT5_moduleType_binary");
    if (pT5_moduleType_binary_branch) { pT5_moduleType_binary_branch->SetAddress(&pT5_moduleType_binary_); }
  }
  pT5_matched_pt_branch = 0;
  if (tree->GetBranch("pT5_matched_pt") != 0) {
    pT5_matched_pt_branch = tree->GetBranch("pT5_matched_pt");
    if (pT5_matched_pt_branch) { pT5_matched_pt_branch->SetAddress(&pT5_matched_pt_); }
  }
  pT5_rzChiSquared_branch = 0;
  if (tree->GetBranch("pT5_rzChiSquared") != 0) {
    pT5_rzChiSquared_branch = tree->GetBranch("pT5_rzChiSquared");
    if (pT5_rzChiSquared_branch) { pT5_rzChiSquared_branch->SetAddress(&pT5_rzChiSquared_); }
  }
  pT5_rPhiChiSquared_branch = 0;
  if (tree->GetBranch("pT5_rPhiChiSquared") != 0) {
    pT5_rPhiChiSquared_branch = tree->GetBranch("pT5_rPhiChiSquared");
    if (pT5_rPhiChiSquared_branch) { pT5_rPhiChiSquared_branch->SetAddress(&pT5_rPhiChiSquared_); }
  }
  pT5_rPhiChiSquaredInwards_branch = 0;
  if (tree->GetBranch("pT5_rPhiChiSquaredInwards") != 0) {
    pT5_rPhiChiSquaredInwards_branch = tree->GetBranch("pT5_rPhiChiSquaredInwards");
    if (pT5_rPhiChiSquaredInwards_branch) { pT5_rPhiChiSquaredInwards_branch->SetAddress(&pT5_rPhiChiSquaredInwards_); }
  }
  sim_pT3_matched_branch = 0;
  if (tree->GetBranch("sim_pT3_matched") != 0) {
    sim_pT3_matched_branch = tree->GetBranch("sim_pT3_matched");
    if (sim_pT3_matched_branch) { sim_pT3_matched_branch->SetAddress(&sim_pT3_matched_); }
  }
  pT3_pt_branch = 0;
  if (tree->GetBranch("pT3_pt") != 0) {
    pT3_pt_branch = tree->GetBranch("pT3_pt");
    if (pT3_pt_branch) { pT3_pt_branch->SetAddress(&pT3_pt_); }
  }
  pT3_isFake_branch = 0;
  if (tree->GetBranch("pT3_isFake") != 0) {
    pT3_isFake_branch = tree->GetBranch("pT3_isFake");
    if (pT3_isFake_branch) { pT3_isFake_branch->SetAddress(&pT3_isFake_); }
  }
  pT3_isDuplicate_branch = 0;
  if (tree->GetBranch("pT3_isDuplicate") != 0) {
    pT3_isDuplicate_branch = tree->GetBranch("pT3_isDuplicate");
    if (pT3_isDuplicate_branch) { pT3_isDuplicate_branch->SetAddress(&pT3_isDuplicate_); }
  }
  pT3_eta_branch = 0;
  if (tree->GetBranch("pT3_eta") != 0) {
    pT3_eta_branch = tree->GetBranch("pT3_eta");
    if (pT3_eta_branch) { pT3_eta_branch->SetAddress(&pT3_eta_); }
  }
  pT3_phi_branch = 0;
  if (tree->GetBranch("pT3_phi") != 0) {
    pT3_phi_branch = tree->GetBranch("pT3_phi");
    if (pT3_phi_branch) { pT3_phi_branch->SetAddress(&pT3_phi_); }
  }
  pT3_score_branch = 0;
  if (tree->GetBranch("pT3_score") != 0) {
    pT3_score_branch = tree->GetBranch("pT3_score");
    if (pT3_score_branch) { pT3_score_branch->SetAddress(&pT3_score_); }
  }
  pT3_foundDuplicate_branch = 0;
  if (tree->GetBranch("pT3_foundDuplicate") != 0) {
    pT3_foundDuplicate_branch = tree->GetBranch("pT3_foundDuplicate");
    if (pT3_foundDuplicate_branch) { pT3_foundDuplicate_branch->SetAddress(&pT3_foundDuplicate_); }
  }
  pT3_matched_simIdx_branch = 0;
  if (tree->GetBranch("pT3_matched_simIdx") != 0) {
    pT3_matched_simIdx_branch = tree->GetBranch("pT3_matched_simIdx");
    if (pT3_matched_simIdx_branch) { pT3_matched_simIdx_branch->SetAddress(&pT3_matched_simIdx_); }
  }
  pT3_hitIdxs_branch = 0;
  if (tree->GetBranch("pT3_hitIdxs") != 0) {
    pT3_hitIdxs_branch = tree->GetBranch("pT3_hitIdxs");
    if (pT3_hitIdxs_branch) { pT3_hitIdxs_branch->SetAddress(&pT3_hitIdxs_); }
  }
  pT3_pixelRadius_branch = 0;
  if (tree->GetBranch("pT3_pixelRadius") != 0) {
    pT3_pixelRadius_branch = tree->GetBranch("pT3_pixelRadius");
    if (pT3_pixelRadius_branch) { pT3_pixelRadius_branch->SetAddress(&pT3_pixelRadius_); }
  }
  pT3_pixelRadiusError_branch = 0;
  if (tree->GetBranch("pT3_pixelRadiusError") != 0) {
    pT3_pixelRadiusError_branch = tree->GetBranch("pT3_pixelRadiusError");
    if (pT3_pixelRadiusError_branch) { pT3_pixelRadiusError_branch->SetAddress(&pT3_pixelRadiusError_); }
  }
  pT3_matched_pt_branch = 0;
  if (tree->GetBranch("pT3_matched_pt") != 0) {
    pT3_matched_pt_branch = tree->GetBranch("pT3_matched_pt");
    if (pT3_matched_pt_branch) { pT3_matched_pt_branch->SetAddress(&pT3_matched_pt_); }
  }
  pT3_tripletRadius_branch = 0;
  if (tree->GetBranch("pT3_tripletRadius") != 0) {
    pT3_tripletRadius_branch = tree->GetBranch("pT3_tripletRadius");
    if (pT3_tripletRadius_branch) { pT3_tripletRadius_branch->SetAddress(&pT3_tripletRadius_); }
  }
  pT3_rPhiChiSquared_branch = 0;
  if (tree->GetBranch("pT3_rPhiChiSquared") != 0) {
    pT3_rPhiChiSquared_branch = tree->GetBranch("pT3_rPhiChiSquared");
    if (pT3_rPhiChiSquared_branch) { pT3_rPhiChiSquared_branch->SetAddress(&pT3_rPhiChiSquared_); }
  }
  pT3_rPhiChiSquaredInwards_branch = 0;
  if (tree->GetBranch("pT3_rPhiChiSquaredInwards") != 0) {
    pT3_rPhiChiSquaredInwards_branch = tree->GetBranch("pT3_rPhiChiSquaredInwards");
    if (pT3_rPhiChiSquaredInwards_branch) { pT3_rPhiChiSquaredInwards_branch->SetAddress(&pT3_rPhiChiSquaredInwards_); }
  }
  pT3_rzChiSquared_branch = 0;
  if (tree->GetBranch("pT3_rzChiSquared") != 0) {
    pT3_rzChiSquared_branch = tree->GetBranch("pT3_rzChiSquared");
    if (pT3_rzChiSquared_branch) { pT3_rzChiSquared_branch->SetAddress(&pT3_rzChiSquared_); }
  }
  pT3_layer_binary_branch = 0;
  if (tree->GetBranch("pT3_layer_binary") != 0) {
    pT3_layer_binary_branch = tree->GetBranch("pT3_layer_binary");
    if (pT3_layer_binary_branch) { pT3_layer_binary_branch->SetAddress(&pT3_layer_binary_); }
  }
  pT3_moduleType_binary_branch = 0;
  if (tree->GetBranch("pT3_moduleType_binary") != 0) {
    pT3_moduleType_binary_branch = tree->GetBranch("pT3_moduleType_binary");
    if (pT3_moduleType_binary_branch) { pT3_moduleType_binary_branch->SetAddress(&pT3_moduleType_binary_); }
  }
  sim_pLS_matched_branch = 0;
  if (tree->GetBranch("sim_pLS_matched") != 0) {
    sim_pLS_matched_branch = tree->GetBranch("sim_pLS_matched");
    if (sim_pLS_matched_branch) { sim_pLS_matched_branch->SetAddress(&sim_pLS_matched_); }
  }
  sim_pLS_types_branch = 0;
  if (tree->GetBranch("sim_pLS_types") != 0) {
    sim_pLS_types_branch = tree->GetBranch("sim_pLS_types");
    if (sim_pLS_types_branch) { sim_pLS_types_branch->SetAddress(&sim_pLS_types_); }
  }
  pLS_isFake_branch = 0;
  if (tree->GetBranch("pLS_isFake") != 0) {
    pLS_isFake_branch = tree->GetBranch("pLS_isFake");
    if (pLS_isFake_branch) { pLS_isFake_branch->SetAddress(&pLS_isFake_); }
  }
  pLS_isDuplicate_branch = 0;
  if (tree->GetBranch("pLS_isDuplicate") != 0) {
    pLS_isDuplicate_branch = tree->GetBranch("pLS_isDuplicate");
    if (pLS_isDuplicate_branch) { pLS_isDuplicate_branch->SetAddress(&pLS_isDuplicate_); }
  }
  pLS_pt_branch = 0;
  if (tree->GetBranch("pLS_pt") != 0) {
    pLS_pt_branch = tree->GetBranch("pLS_pt");
    if (pLS_pt_branch) { pLS_pt_branch->SetAddress(&pLS_pt_); }
  }
  pLS_eta_branch = 0;
  if (tree->GetBranch("pLS_eta") != 0) {
    pLS_eta_branch = tree->GetBranch("pLS_eta");
    if (pLS_eta_branch) { pLS_eta_branch->SetAddress(&pLS_eta_); }
  }
  pLS_phi_branch = 0;
  if (tree->GetBranch("pLS_phi") != 0) {
    pLS_phi_branch = tree->GetBranch("pLS_phi");
    if (pLS_phi_branch) { pLS_phi_branch->SetAddress(&pLS_phi_); }
  }
  pLS_score_branch = 0;
  if (tree->GetBranch("pLS_score") != 0) {
    pLS_score_branch = tree->GetBranch("pLS_score");
    if (pLS_score_branch) { pLS_score_branch->SetAddress(&pLS_score_); }
  }
  sim_T5_matched_branch = 0;
  if (tree->GetBranch("sim_T5_matched") != 0) {
    sim_T5_matched_branch = tree->GetBranch("sim_T5_matched");
    if (sim_T5_matched_branch) { sim_T5_matched_branch->SetAddress(&sim_T5_matched_); }
  }
  t5_isFake_branch = 0;
  if (tree->GetBranch("t5_isFake") != 0) {
    t5_isFake_branch = tree->GetBranch("t5_isFake");
    if (t5_isFake_branch) { t5_isFake_branch->SetAddress(&t5_isFake_); }
  }
  t5_isDuplicate_branch = 0;
  if (tree->GetBranch("t5_isDuplicate") != 0) {
    t5_isDuplicate_branch = tree->GetBranch("t5_isDuplicate");
    if (t5_isDuplicate_branch) { t5_isDuplicate_branch->SetAddress(&t5_isDuplicate_); }
  }
  t5_foundDuplicate_branch = 0;
  if (tree->GetBranch("t5_foundDuplicate") != 0) {
    t5_foundDuplicate_branch = tree->GetBranch("t5_foundDuplicate");
    if (t5_foundDuplicate_branch) { t5_foundDuplicate_branch->SetAddress(&t5_foundDuplicate_); }
  }
  t5_pt_branch = 0;
  if (tree->GetBranch("t5_pt") != 0) {
    t5_pt_branch = tree->GetBranch("t5_pt");
    if (t5_pt_branch) { t5_pt_branch->SetAddress(&t5_pt_); }
  }
  t5_eta_branch = 0;
  if (tree->GetBranch("t5_eta") != 0) {
    t5_eta_branch = tree->GetBranch("t5_eta");
    if (t5_eta_branch) { t5_eta_branch->SetAddress(&t5_eta_); }
  }
  t5_phi_branch = 0;
  if (tree->GetBranch("t5_phi") != 0) {
    t5_phi_branch = tree->GetBranch("t5_phi");
    if (t5_phi_branch) { t5_phi_branch->SetAddress(&t5_phi_); }
  }
  t5_score_rphisum_branch = 0;
  if (tree->GetBranch("t5_score_rphisum") != 0) {
    t5_score_rphisum_branch = tree->GetBranch("t5_score_rphisum");
    if (t5_score_rphisum_branch) { t5_score_rphisum_branch->SetAddress(&t5_score_rphisum_); }
  }
  t5_hitIdxs_branch = 0;
  if (tree->GetBranch("t5_hitIdxs") != 0) {
    t5_hitIdxs_branch = tree->GetBranch("t5_hitIdxs");
    if (t5_hitIdxs_branch) { t5_hitIdxs_branch->SetAddress(&t5_hitIdxs_); }
  }
  t5_matched_simIdx_branch = 0;
  if (tree->GetBranch("t5_matched_simIdx") != 0) {
    t5_matched_simIdx_branch = tree->GetBranch("t5_matched_simIdx");
    if (t5_matched_simIdx_branch) { t5_matched_simIdx_branch->SetAddress(&t5_matched_simIdx_); }
  }
  t5_moduleType_binary_branch = 0;
  if (tree->GetBranch("t5_moduleType_binary") != 0) {
    t5_moduleType_binary_branch = tree->GetBranch("t5_moduleType_binary");
    if (t5_moduleType_binary_branch) { t5_moduleType_binary_branch->SetAddress(&t5_moduleType_binary_); }
  }
  t5_layer_binary_branch = 0;
  if (tree->GetBranch("t5_layer_binary") != 0) {
    t5_layer_binary_branch = tree->GetBranch("t5_layer_binary");
    if (t5_layer_binary_branch) { t5_layer_binary_branch->SetAddress(&t5_layer_binary_); }
  }
  t5_matched_pt_branch = 0;
  if (tree->GetBranch("t5_matched_pt") != 0) {
    t5_matched_pt_branch = tree->GetBranch("t5_matched_pt");
    if (t5_matched_pt_branch) { t5_matched_pt_branch->SetAddress(&t5_matched_pt_); }
  }
  t5_partOfTC_branch = 0;
  if (tree->GetBranch("t5_partOfTC") != 0) {
    t5_partOfTC_branch = tree->GetBranch("t5_partOfTC");
    if (t5_partOfTC_branch) { t5_partOfTC_branch->SetAddress(&t5_partOfTC_); }
  }
  t5_innerRadius_branch = 0;
  if (tree->GetBranch("t5_innerRadius") != 0) {
    t5_innerRadius_branch = tree->GetBranch("t5_innerRadius");
    if (t5_innerRadius_branch) { t5_innerRadius_branch->SetAddress(&t5_innerRadius_); }
  }
  t5_outerRadius_branch = 0;
  if (tree->GetBranch("t5_outerRadius") != 0) {
    t5_outerRadius_branch = tree->GetBranch("t5_outerRadius");
    if (t5_outerRadius_branch) { t5_outerRadius_branch->SetAddress(&t5_outerRadius_); }
  }
  t5_bridgeRadius_branch = 0;
  if (tree->GetBranch("t5_bridgeRadius") != 0) {
    t5_bridgeRadius_branch = tree->GetBranch("t5_bridgeRadius");
    if (t5_bridgeRadius_branch) { t5_bridgeRadius_branch->SetAddress(&t5_bridgeRadius_); }
  }
  t5_chiSquared_branch = 0;
  if (tree->GetBranch("t5_chiSquared") != 0) {
    t5_chiSquared_branch = tree->GetBranch("t5_chiSquared");
    if (t5_chiSquared_branch) { t5_chiSquared_branch->SetAddress(&t5_chiSquared_); }
  }
  t5_rzChiSquared_branch = 0;
  if (tree->GetBranch("t5_rzChiSquared") != 0) {
    t5_rzChiSquared_branch = tree->GetBranch("t5_rzChiSquared");
    if (t5_rzChiSquared_branch) { t5_rzChiSquared_branch->SetAddress(&t5_rzChiSquared_); }
  }
  t5_nonAnchorChiSquared_branch = 0;
  if (tree->GetBranch("t5_nonAnchorChiSquared") != 0) {
    t5_nonAnchorChiSquared_branch = tree->GetBranch("t5_nonAnchorChiSquared");
    if (t5_nonAnchorChiSquared_branch) { t5_nonAnchorChiSquared_branch->SetAddress(&t5_nonAnchorChiSquared_); }
  }
  MD_pt_branch = 0;
  if (tree->GetBranch("MD_pt") != 0) {
    MD_pt_branch = tree->GetBranch("MD_pt");
    if (MD_pt_branch) { MD_pt_branch->SetAddress(&MD_pt_); }
  }
  MD_eta_branch = 0;
  if (tree->GetBranch("MD_eta") != 0) {
    MD_eta_branch = tree->GetBranch("MD_eta");
    if (MD_eta_branch) { MD_eta_branch->SetAddress(&MD_eta_); }
  }
  MD_phi_branch = 0;
  if (tree->GetBranch("MD_phi") != 0) {
    MD_phi_branch = tree->GetBranch("MD_phi");
    if (MD_phi_branch) { MD_phi_branch->SetAddress(&MD_phi_); }
  }
  MD_dphichange_branch = 0;
  if (tree->GetBranch("MD_dphichange") != 0) {
    MD_dphichange_branch = tree->GetBranch("MD_dphichange");
    if (MD_dphichange_branch) { MD_dphichange_branch->SetAddress(&MD_dphichange_); }
  }
  MD_isFake_branch = 0;
  if (tree->GetBranch("MD_isFake") != 0) {
    MD_isFake_branch = tree->GetBranch("MD_isFake");
    if (MD_isFake_branch) { MD_isFake_branch->SetAddress(&MD_isFake_); }
  }
  MD_tpType_branch = 0;
  if (tree->GetBranch("MD_tpType") != 0) {
    MD_tpType_branch = tree->GetBranch("MD_tpType");
    if (MD_tpType_branch) { MD_tpType_branch->SetAddress(&MD_tpType_); }
  }
  MD_detId_branch = 0;
  if (tree->GetBranch("MD_detId") != 0) {
    MD_detId_branch = tree->GetBranch("MD_detId");
    if (MD_detId_branch) { MD_detId_branch->SetAddress(&MD_detId_); }
  }
  MD_layer_branch = 0;
  if (tree->GetBranch("MD_layer") != 0) {
    MD_layer_branch = tree->GetBranch("MD_layer");
    if (MD_layer_branch) { MD_layer_branch->SetAddress(&MD_layer_); }
  }
  MD_0_r_branch = 0;
  if (tree->GetBranch("MD_0_r") != 0) {
    MD_0_r_branch = tree->GetBranch("MD_0_r");
    if (MD_0_r_branch) { MD_0_r_branch->SetAddress(&MD_0_r_); }
  }
  MD_0_x_branch = 0;
  if (tree->GetBranch("MD_0_x") != 0) {
    MD_0_x_branch = tree->GetBranch("MD_0_x");
    if (MD_0_x_branch) { MD_0_x_branch->SetAddress(&MD_0_x_); }
  }
  MD_0_y_branch = 0;
  if (tree->GetBranch("MD_0_y") != 0) {
    MD_0_y_branch = tree->GetBranch("MD_0_y");
    if (MD_0_y_branch) { MD_0_y_branch->SetAddress(&MD_0_y_); }
  }
  MD_0_z_branch = 0;
  if (tree->GetBranch("MD_0_z") != 0) {
    MD_0_z_branch = tree->GetBranch("MD_0_z");
    if (MD_0_z_branch) { MD_0_z_branch->SetAddress(&MD_0_z_); }
  }
  MD_1_r_branch = 0;
  if (tree->GetBranch("MD_1_r") != 0) {
    MD_1_r_branch = tree->GetBranch("MD_1_r");
    if (MD_1_r_branch) { MD_1_r_branch->SetAddress(&MD_1_r_); }
  }
  MD_1_x_branch = 0;
  if (tree->GetBranch("MD_1_x") != 0) {
    MD_1_x_branch = tree->GetBranch("MD_1_x");
    if (MD_1_x_branch) { MD_1_x_branch->SetAddress(&MD_1_x_); }
  }
  MD_1_y_branch = 0;
  if (tree->GetBranch("MD_1_y") != 0) {
    MD_1_y_branch = tree->GetBranch("MD_1_y");
    if (MD_1_y_branch) { MD_1_y_branch->SetAddress(&MD_1_y_); }
  }
  MD_1_z_branch = 0;
  if (tree->GetBranch("MD_1_z") != 0) {
    MD_1_z_branch = tree->GetBranch("MD_1_z");
    if (MD_1_z_branch) { MD_1_z_branch->SetAddress(&MD_1_z_); }
  }
  LS_pt_branch = 0;
  if (tree->GetBranch("LS_pt") != 0) {
    LS_pt_branch = tree->GetBranch("LS_pt");
    if (LS_pt_branch) { LS_pt_branch->SetAddress(&LS_pt_); }
  }
  LS_eta_branch = 0;
  if (tree->GetBranch("LS_eta") != 0) {
    LS_eta_branch = tree->GetBranch("LS_eta");
    if (LS_eta_branch) { LS_eta_branch->SetAddress(&LS_eta_); }
  }
  LS_phi_branch = 0;
  if (tree->GetBranch("LS_phi") != 0) {
    LS_phi_branch = tree->GetBranch("LS_phi");
    if (LS_phi_branch) { LS_phi_branch->SetAddress(&LS_phi_); }
  }
  LS_isFake_branch = 0;
  if (tree->GetBranch("LS_isFake") != 0) {
    LS_isFake_branch = tree->GetBranch("LS_isFake");
    if (LS_isFake_branch) { LS_isFake_branch->SetAddress(&LS_isFake_); }
  }
  LS_MD_idx0_branch = 0;
  if (tree->GetBranch("LS_MD_idx0") != 0) {
    LS_MD_idx0_branch = tree->GetBranch("LS_MD_idx0");
    if (LS_MD_idx0_branch) { LS_MD_idx0_branch->SetAddress(&LS_MD_idx0_); }
  }
  LS_MD_idx1_branch = 0;
  if (tree->GetBranch("LS_MD_idx1") != 0) {
    LS_MD_idx1_branch = tree->GetBranch("LS_MD_idx1");
    if (LS_MD_idx1_branch) { LS_MD_idx1_branch->SetAddress(&LS_MD_idx1_); }
  }
  LS_sim_pt_branch = 0;
  if (tree->GetBranch("LS_sim_pt") != 0) {
    LS_sim_pt_branch = tree->GetBranch("LS_sim_pt");
    if (LS_sim_pt_branch) { LS_sim_pt_branch->SetAddress(&LS_sim_pt_); }
  }
  LS_sim_eta_branch = 0;
  if (tree->GetBranch("LS_sim_eta") != 0) {
    LS_sim_eta_branch = tree->GetBranch("LS_sim_eta");
    if (LS_sim_eta_branch) { LS_sim_eta_branch->SetAddress(&LS_sim_eta_); }
  }
  LS_sim_phi_branch = 0;
  if (tree->GetBranch("LS_sim_phi") != 0) {
    LS_sim_phi_branch = tree->GetBranch("LS_sim_phi");
    if (LS_sim_phi_branch) { LS_sim_phi_branch->SetAddress(&LS_sim_phi_); }
  }
  LS_sim_pca_dxy_branch = 0;
  if (tree->GetBranch("LS_sim_pca_dxy") != 0) {
    LS_sim_pca_dxy_branch = tree->GetBranch("LS_sim_pca_dxy");
    if (LS_sim_pca_dxy_branch) { LS_sim_pca_dxy_branch->SetAddress(&LS_sim_pca_dxy_); }
  }
  LS_sim_pca_dz_branch = 0;
  if (tree->GetBranch("LS_sim_pca_dz") != 0) {
    LS_sim_pca_dz_branch = tree->GetBranch("LS_sim_pca_dz");
    if (LS_sim_pca_dz_branch) { LS_sim_pca_dz_branch->SetAddress(&LS_sim_pca_dz_); }
  }
  LS_sim_q_branch = 0;
  if (tree->GetBranch("LS_sim_q") != 0) {
    LS_sim_q_branch = tree->GetBranch("LS_sim_q");
    if (LS_sim_q_branch) { LS_sim_q_branch->SetAddress(&LS_sim_q_); }
  }
  LS_sim_pdgId_branch = 0;
  if (tree->GetBranch("LS_sim_pdgId") != 0) {
    LS_sim_pdgId_branch = tree->GetBranch("LS_sim_pdgId");
    if (LS_sim_pdgId_branch) { LS_sim_pdgId_branch->SetAddress(&LS_sim_pdgId_); }
  }
  LS_sim_event_branch = 0;
  if (tree->GetBranch("LS_sim_event") != 0) {
    LS_sim_event_branch = tree->GetBranch("LS_sim_event");
    if (LS_sim_event_branch) { LS_sim_event_branch->SetAddress(&LS_sim_event_); }
  }
  LS_sim_bx_branch = 0;
  if (tree->GetBranch("LS_sim_bx") != 0) {
    LS_sim_bx_branch = tree->GetBranch("LS_sim_bx");
    if (LS_sim_bx_branch) { LS_sim_bx_branch->SetAddress(&LS_sim_bx_); }
  }
  LS_sim_vx_branch = 0;
  if (tree->GetBranch("LS_sim_vx") != 0) {
    LS_sim_vx_branch = tree->GetBranch("LS_sim_vx");
    if (LS_sim_vx_branch) { LS_sim_vx_branch->SetAddress(&LS_sim_vx_); }
  }
  LS_sim_vy_branch = 0;
  if (tree->GetBranch("LS_sim_vy") != 0) {
    LS_sim_vy_branch = tree->GetBranch("LS_sim_vy");
    if (LS_sim_vy_branch) { LS_sim_vy_branch->SetAddress(&LS_sim_vy_); }
  }
  LS_sim_vz_branch = 0;
  if (tree->GetBranch("LS_sim_vz") != 0) {
    LS_sim_vz_branch = tree->GetBranch("LS_sim_vz");
    if (LS_sim_vz_branch) { LS_sim_vz_branch->SetAddress(&LS_sim_vz_); }
  }
  LS_isInTrueTC_branch = 0;
  if (tree->GetBranch("LS_isInTrueTC") != 0) {
    LS_isInTrueTC_branch = tree->GetBranch("LS_isInTrueTC");
    if (LS_isInTrueTC_branch) { LS_isInTrueTC_branch->SetAddress(&LS_isInTrueTC_); }
  }
  t5_t3_idx0_branch = 0;
  if (tree->GetBranch("t5_t3_idx0") != 0) {
    t5_t3_idx0_branch = tree->GetBranch("t5_t3_idx0");
    if (t5_t3_idx0_branch) { t5_t3_idx0_branch->SetAddress(&t5_t3_idx0_); }
  }
  t5_t3_idx1_branch = 0;
  if (tree->GetBranch("t5_t3_idx1") != 0) {
    t5_t3_idx1_branch = tree->GetBranch("t5_t3_idx1");
    if (t5_t3_idx1_branch) { t5_t3_idx1_branch->SetAddress(&t5_t3_idx1_); }
  }
  t3_isFake_branch = 0;
  if (tree->GetBranch("t3_isFake") != 0) {
    t3_isFake_branch = tree->GetBranch("t3_isFake");
    if (t3_isFake_branch) { t3_isFake_branch->SetAddress(&t3_isFake_); }
  }
  t3_ptLegacy_branch = 0;
  if (tree->GetBranch("t3_ptLegacy") != 0) {
    t3_ptLegacy_branch = tree->GetBranch("t3_ptLegacy");
    if (t3_ptLegacy_branch) { t3_ptLegacy_branch->SetAddress(&t3_ptLegacy_); }
  }
  t3_pt_branch = 0;
  if (tree->GetBranch("t3_pt") != 0) {
    t3_pt_branch = tree->GetBranch("t3_pt");
    if (t3_pt_branch) { t3_pt_branch->SetAddress(&t3_pt_); }
  }
  t3_eta_branch = 0;
  if (tree->GetBranch("t3_eta") != 0) {
    t3_eta_branch = tree->GetBranch("t3_eta");
    if (t3_eta_branch) { t3_eta_branch->SetAddress(&t3_eta_); }
  }
  t3_phi_branch = 0;
  if (tree->GetBranch("t3_phi") != 0) {
    t3_phi_branch = tree->GetBranch("t3_phi");
    if (t3_phi_branch) { t3_phi_branch->SetAddress(&t3_phi_); }
  }
  t3_0_r_branch = 0;
  if (tree->GetBranch("t3_0_r") != 0) {
    t3_0_r_branch = tree->GetBranch("t3_0_r");
    if (t3_0_r_branch) { t3_0_r_branch->SetAddress(&t3_0_r_); }
  }
  t3_0_dr_branch = 0;
  if (tree->GetBranch("t3_0_dr") != 0) {
    t3_0_dr_branch = tree->GetBranch("t3_0_dr");
    if (t3_0_dr_branch) { t3_0_dr_branch->SetAddress(&t3_0_dr_); }
  }
  t3_0_x_branch = 0;
  if (tree->GetBranch("t3_0_x") != 0) {
    t3_0_x_branch = tree->GetBranch("t3_0_x");
    if (t3_0_x_branch) { t3_0_x_branch->SetAddress(&t3_0_x_); }
  }
  t3_0_y_branch = 0;
  if (tree->GetBranch("t3_0_y") != 0) {
    t3_0_y_branch = tree->GetBranch("t3_0_y");
    if (t3_0_y_branch) { t3_0_y_branch->SetAddress(&t3_0_y_); }
  }
  t3_0_z_branch = 0;
  if (tree->GetBranch("t3_0_z") != 0) {
    t3_0_z_branch = tree->GetBranch("t3_0_z");
    if (t3_0_z_branch) { t3_0_z_branch->SetAddress(&t3_0_z_); }
  }
  t3_2_r_branch = 0;
  if (tree->GetBranch("t3_2_r") != 0) {
    t3_2_r_branch = tree->GetBranch("t3_2_r");
    if (t3_2_r_branch) { t3_2_r_branch->SetAddress(&t3_2_r_); }
  }
  t3_2_dr_branch = 0;
  if (tree->GetBranch("t3_2_dr") != 0) {
    t3_2_dr_branch = tree->GetBranch("t3_2_dr");
    if (t3_2_dr_branch) { t3_2_dr_branch->SetAddress(&t3_2_dr_); }
  }
  t3_2_x_branch = 0;
  if (tree->GetBranch("t3_2_x") != 0) {
    t3_2_x_branch = tree->GetBranch("t3_2_x");
    if (t3_2_x_branch) { t3_2_x_branch->SetAddress(&t3_2_x_); }
  }
  t3_2_y_branch = 0;
  if (tree->GetBranch("t3_2_y") != 0) {
    t3_2_y_branch = tree->GetBranch("t3_2_y");
    if (t3_2_y_branch) { t3_2_y_branch->SetAddress(&t3_2_y_); }
  }
  t3_2_z_branch = 0;
  if (tree->GetBranch("t3_2_z") != 0) {
    t3_2_z_branch = tree->GetBranch("t3_2_z");
    if (t3_2_z_branch) { t3_2_z_branch->SetAddress(&t3_2_z_); }
  }
  t3_4_r_branch = 0;
  if (tree->GetBranch("t3_4_r") != 0) {
    t3_4_r_branch = tree->GetBranch("t3_4_r");
    if (t3_4_r_branch) { t3_4_r_branch->SetAddress(&t3_4_r_); }
  }
  t3_4_dr_branch = 0;
  if (tree->GetBranch("t3_4_dr") != 0) {
    t3_4_dr_branch = tree->GetBranch("t3_4_dr");
    if (t3_4_dr_branch) { t3_4_dr_branch->SetAddress(&t3_4_dr_); }
  }
  t3_4_x_branch = 0;
  if (tree->GetBranch("t3_4_x") != 0) {
    t3_4_x_branch = tree->GetBranch("t3_4_x");
    if (t3_4_x_branch) { t3_4_x_branch->SetAddress(&t3_4_x_); }
  }
  t3_4_y_branch = 0;
  if (tree->GetBranch("t3_4_y") != 0) {
    t3_4_y_branch = tree->GetBranch("t3_4_y");
    if (t3_4_y_branch) { t3_4_y_branch->SetAddress(&t3_4_y_); }
  }
  t3_4_z_branch = 0;
  if (tree->GetBranch("t3_4_z") != 0) {
    t3_4_z_branch = tree->GetBranch("t3_4_z");
    if (t3_4_z_branch) { t3_4_z_branch->SetAddress(&t3_4_z_); }
  }
  tc_lsIdx_branch = 0;
  if (tree->GetBranch("tc_lsIdx") != 0) {
    tc_lsIdx_branch = tree->GetBranch("tc_lsIdx");
    if (tc_lsIdx_branch) { tc_lsIdx_branch->SetAddress(&tc_lsIdx_); }
  }
  tree->SetMakeClass(0);
}
void SDL::GetEntry(unsigned int idx) {
  index = idx;
  SimTrack_pt_isLoaded = false;
  SimTrack_eta_isLoaded = false;
  SimTrack_phi_isLoaded = false;
  SimTrack_dxy_isLoaded = false;
  SimTrack_dz_isLoaded = false;
  SimTrack_charge_isLoaded = false;
  SimTrack_ppVtx_isLoaded = false;
  SimTrack_pdgID_isLoaded = false;
  SimTrack_vx_isLoaded = false;
  SimTrack_vy_isLoaded = false;
  SimTrack_vz_isLoaded = false;
  SimTrack_TC_idx_isLoaded = false;
  SimTrack_TC_typemask_isLoaded = false;
  TC_pt_isLoaded = false;
  TC_eta_isLoaded = false;
  TC_phi_isLoaded = false;
  TC_dxy_isLoaded = false;
  TC_dz_isLoaded = false;
  TC_charge_isLoaded = false;
  TC_type_isLoaded = false;
  TC_isFake_isLoaded = false;
  TC_isDuplicate_isLoaded = false;
  TC_SimTrack_idx_isLoaded = false;
  sim_pt_isLoaded = false;
  sim_eta_isLoaded = false;
  sim_phi_isLoaded = false;
  sim_pca_dxy_isLoaded = false;
  sim_pca_dz_isLoaded = false;
  sim_q_isLoaded = false;
  sim_event_isLoaded = false;
  sim_pdgId_isLoaded = false;
  sim_vx_isLoaded = false;
  sim_vy_isLoaded = false;
  sim_vz_isLoaded = false;
  sim_trkNtupIdx_isLoaded = false;
  sim_TC_matched_isLoaded = false;
  sim_TC_matched_mask_isLoaded = false;
  tc_pt_isLoaded = false;
  tc_eta_isLoaded = false;
  tc_phi_isLoaded = false;
  tc_type_isLoaded = false;
  tc_isFake_isLoaded = false;
  tc_isDuplicate_isLoaded = false;
  tc_matched_simIdx_isLoaded = false;
  sim_dummy_isLoaded = false;
  tc_dummy_isLoaded = false;
  pT5_matched_simIdx_isLoaded = false;
  pT5_hitIdxs_isLoaded = false;
  sim_pT5_matched_isLoaded = false;
  pT5_pt_isLoaded = false;
  pT5_eta_isLoaded = false;
  pT5_phi_isLoaded = false;
  pT5_isFake_isLoaded = false;
  pT5_isDuplicate_isLoaded = false;
  pT5_score_isLoaded = false;
  pT5_layer_binary_isLoaded = false;
  pT5_moduleType_binary_isLoaded = false;
  pT5_matched_pt_isLoaded = false;
  pT5_rzChiSquared_isLoaded = false;
  pT5_rPhiChiSquared_isLoaded = false;
  pT5_rPhiChiSquaredInwards_isLoaded = false;
  sim_pT3_matched_isLoaded = false;
  pT3_pt_isLoaded = false;
  pT3_isFake_isLoaded = false;
  pT3_isDuplicate_isLoaded = false;
  pT3_eta_isLoaded = false;
  pT3_phi_isLoaded = false;
  pT3_score_isLoaded = false;
  pT3_foundDuplicate_isLoaded = false;
  pT3_matched_simIdx_isLoaded = false;
  pT3_hitIdxs_isLoaded = false;
  pT3_pixelRadius_isLoaded = false;
  pT3_pixelRadiusError_isLoaded = false;
  pT3_matched_pt_isLoaded = false;
  pT3_tripletRadius_isLoaded = false;
  pT3_rPhiChiSquared_isLoaded = false;
  pT3_rPhiChiSquaredInwards_isLoaded = false;
  pT3_rzChiSquared_isLoaded = false;
  pT3_layer_binary_isLoaded = false;
  pT3_moduleType_binary_isLoaded = false;
  sim_pLS_matched_isLoaded = false;
  sim_pLS_types_isLoaded = false;
  pLS_isFake_isLoaded = false;
  pLS_isDuplicate_isLoaded = false;
  pLS_pt_isLoaded = false;
  pLS_eta_isLoaded = false;
  pLS_phi_isLoaded = false;
  pLS_score_isLoaded = false;
  sim_T5_matched_isLoaded = false;
  t5_isFake_isLoaded = false;
  t5_isDuplicate_isLoaded = false;
  t5_foundDuplicate_isLoaded = false;
  t5_pt_isLoaded = false;
  t5_eta_isLoaded = false;
  t5_phi_isLoaded = false;
  t5_score_rphisum_isLoaded = false;
  t5_hitIdxs_isLoaded = false;
  t5_matched_simIdx_isLoaded = false;
  t5_moduleType_binary_isLoaded = false;
  t5_layer_binary_isLoaded = false;
  t5_matched_pt_isLoaded = false;
  t5_partOfTC_isLoaded = false;
  t5_innerRadius_isLoaded = false;
  t5_outerRadius_isLoaded = false;
  t5_bridgeRadius_isLoaded = false;
  t5_chiSquared_isLoaded = false;
  t5_rzChiSquared_isLoaded = false;
  t5_nonAnchorChiSquared_isLoaded = false;
  MD_pt_isLoaded = false;
  MD_eta_isLoaded = false;
  MD_phi_isLoaded = false;
  MD_dphichange_isLoaded = false;
  MD_isFake_isLoaded = false;
  MD_tpType_isLoaded = false;
  MD_detId_isLoaded = false;
  MD_layer_isLoaded = false;
  MD_0_r_isLoaded = false;
  MD_0_x_isLoaded = false;
  MD_0_y_isLoaded = false;
  MD_0_z_isLoaded = false;
  MD_1_r_isLoaded = false;
  MD_1_x_isLoaded = false;
  MD_1_y_isLoaded = false;
  MD_1_z_isLoaded = false;
  LS_pt_isLoaded = false;
  LS_eta_isLoaded = false;
  LS_phi_isLoaded = false;
  LS_isFake_isLoaded = false;
  LS_MD_idx0_isLoaded = false;
  LS_MD_idx1_isLoaded = false;
  LS_sim_pt_isLoaded = false;
  LS_sim_eta_isLoaded = false;
  LS_sim_phi_isLoaded = false;
  LS_sim_pca_dxy_isLoaded = false;
  LS_sim_pca_dz_isLoaded = false;
  LS_sim_q_isLoaded = false;
  LS_sim_pdgId_isLoaded = false;
  LS_sim_event_isLoaded = false;
  LS_sim_bx_isLoaded = false;
  LS_sim_vx_isLoaded = false;
  LS_sim_vy_isLoaded = false;
  LS_sim_vz_isLoaded = false;
  LS_isInTrueTC_isLoaded = false;
  t5_t3_idx0_isLoaded = false;
  t5_t3_idx1_isLoaded = false;
  t3_isFake_isLoaded = false;
  t3_ptLegacy_isLoaded = false;
  t3_pt_isLoaded = false;
  t3_eta_isLoaded = false;
  t3_phi_isLoaded = false;
  t3_0_r_isLoaded = false;
  t3_0_dr_isLoaded = false;
  t3_0_x_isLoaded = false;
  t3_0_y_isLoaded = false;
  t3_0_z_isLoaded = false;
  t3_2_r_isLoaded = false;
  t3_2_dr_isLoaded = false;
  t3_2_x_isLoaded = false;
  t3_2_y_isLoaded = false;
  t3_2_z_isLoaded = false;
  t3_4_r_isLoaded = false;
  t3_4_dr_isLoaded = false;
  t3_4_x_isLoaded = false;
  t3_4_y_isLoaded = false;
  t3_4_z_isLoaded = false;
  tc_lsIdx_isLoaded = false;
}
void SDL::LoadAllBranches() {
  if (SimTrack_pt_branch != 0) SimTrack_pt();
  if (SimTrack_eta_branch != 0) SimTrack_eta();
  if (SimTrack_phi_branch != 0) SimTrack_phi();
  if (SimTrack_dxy_branch != 0) SimTrack_dxy();
  if (SimTrack_dz_branch != 0) SimTrack_dz();
  if (SimTrack_charge_branch != 0) SimTrack_charge();
  if (SimTrack_ppVtx_branch != 0) SimTrack_ppVtx();
  if (SimTrack_pdgID_branch != 0) SimTrack_pdgID();
  if (SimTrack_vx_branch != 0) SimTrack_vx();
  if (SimTrack_vy_branch != 0) SimTrack_vy();
  if (SimTrack_vz_branch != 0) SimTrack_vz();
  if (SimTrack_TC_idx_branch != 0) SimTrack_TC_idx();
  if (SimTrack_TC_typemask_branch != 0) SimTrack_TC_typemask();
  if (TC_pt_branch != 0) TC_pt();
  if (TC_eta_branch != 0) TC_eta();
  if (TC_phi_branch != 0) TC_phi();
  if (TC_dxy_branch != 0) TC_dxy();
  if (TC_dz_branch != 0) TC_dz();
  if (TC_charge_branch != 0) TC_charge();
  if (TC_type_branch != 0) TC_type();
  if (TC_isFake_branch != 0) TC_isFake();
  if (TC_isDuplicate_branch != 0) TC_isDuplicate();
  if (TC_SimTrack_idx_branch != 0) TC_SimTrack_idx();
  if (sim_pt_branch != 0) sim_pt();
  if (sim_eta_branch != 0) sim_eta();
  if (sim_phi_branch != 0) sim_phi();
  if (sim_pca_dxy_branch != 0) sim_pca_dxy();
  if (sim_pca_dz_branch != 0) sim_pca_dz();
  if (sim_q_branch != 0) sim_q();
  if (sim_event_branch != 0) sim_event();
  if (sim_pdgId_branch != 0) sim_pdgId();
  if (sim_vx_branch != 0) sim_vx();
  if (sim_vy_branch != 0) sim_vy();
  if (sim_vz_branch != 0) sim_vz();
  if (sim_trkNtupIdx_branch != 0) sim_trkNtupIdx();
  if (sim_TC_matched_branch != 0) sim_TC_matched();
  if (sim_TC_matched_mask_branch != 0) sim_TC_matched_mask();
  if (tc_pt_branch != 0) tc_pt();
  if (tc_eta_branch != 0) tc_eta();
  if (tc_phi_branch != 0) tc_phi();
  if (tc_type_branch != 0) tc_type();
  if (tc_isFake_branch != 0) tc_isFake();
  if (tc_isDuplicate_branch != 0) tc_isDuplicate();
  if (tc_matched_simIdx_branch != 0) tc_matched_simIdx();
  if (sim_dummy_branch != 0) sim_dummy();
  if (tc_dummy_branch != 0) tc_dummy();
  if (pT5_matched_simIdx_branch != 0) pT5_matched_simIdx();
  if (pT5_hitIdxs_branch != 0) pT5_hitIdxs();
  if (sim_pT5_matched_branch != 0) sim_pT5_matched();
  if (pT5_pt_branch != 0) pT5_pt();
  if (pT5_eta_branch != 0) pT5_eta();
  if (pT5_phi_branch != 0) pT5_phi();
  if (pT5_isFake_branch != 0) pT5_isFake();
  if (pT5_isDuplicate_branch != 0) pT5_isDuplicate();
  if (pT5_score_branch != 0) pT5_score();
  if (pT5_layer_binary_branch != 0) pT5_layer_binary();
  if (pT5_moduleType_binary_branch != 0) pT5_moduleType_binary();
  if (pT5_matched_pt_branch != 0) pT5_matched_pt();
  if (pT5_rzChiSquared_branch != 0) pT5_rzChiSquared();
  if (pT5_rPhiChiSquared_branch != 0) pT5_rPhiChiSquared();
  if (pT5_rPhiChiSquaredInwards_branch != 0) pT5_rPhiChiSquaredInwards();
  if (sim_pT3_matched_branch != 0) sim_pT3_matched();
  if (pT3_pt_branch != 0) pT3_pt();
  if (pT3_isFake_branch != 0) pT3_isFake();
  if (pT3_isDuplicate_branch != 0) pT3_isDuplicate();
  if (pT3_eta_branch != 0) pT3_eta();
  if (pT3_phi_branch != 0) pT3_phi();
  if (pT3_score_branch != 0) pT3_score();
  if (pT3_foundDuplicate_branch != 0) pT3_foundDuplicate();
  if (pT3_matched_simIdx_branch != 0) pT3_matched_simIdx();
  if (pT3_hitIdxs_branch != 0) pT3_hitIdxs();
  if (pT3_pixelRadius_branch != 0) pT3_pixelRadius();
  if (pT3_pixelRadiusError_branch != 0) pT3_pixelRadiusError();
  if (pT3_matched_pt_branch != 0) pT3_matched_pt();
  if (pT3_tripletRadius_branch != 0) pT3_tripletRadius();
  if (pT3_rPhiChiSquared_branch != 0) pT3_rPhiChiSquared();
  if (pT3_rPhiChiSquaredInwards_branch != 0) pT3_rPhiChiSquaredInwards();
  if (pT3_rzChiSquared_branch != 0) pT3_rzChiSquared();
  if (pT3_layer_binary_branch != 0) pT3_layer_binary();
  if (pT3_moduleType_binary_branch != 0) pT3_moduleType_binary();
  if (sim_pLS_matched_branch != 0) sim_pLS_matched();
  if (sim_pLS_types_branch != 0) sim_pLS_types();
  if (pLS_isFake_branch != 0) pLS_isFake();
  if (pLS_isDuplicate_branch != 0) pLS_isDuplicate();
  if (pLS_pt_branch != 0) pLS_pt();
  if (pLS_eta_branch != 0) pLS_eta();
  if (pLS_phi_branch != 0) pLS_phi();
  if (pLS_score_branch != 0) pLS_score();
  if (sim_T5_matched_branch != 0) sim_T5_matched();
  if (t5_isFake_branch != 0) t5_isFake();
  if (t5_isDuplicate_branch != 0) t5_isDuplicate();
  if (t5_foundDuplicate_branch != 0) t5_foundDuplicate();
  if (t5_pt_branch != 0) t5_pt();
  if (t5_eta_branch != 0) t5_eta();
  if (t5_phi_branch != 0) t5_phi();
  if (t5_score_rphisum_branch != 0) t5_score_rphisum();
  if (t5_hitIdxs_branch != 0) t5_hitIdxs();
  if (t5_matched_simIdx_branch != 0) t5_matched_simIdx();
  if (t5_moduleType_binary_branch != 0) t5_moduleType_binary();
  if (t5_layer_binary_branch != 0) t5_layer_binary();
  if (t5_matched_pt_branch != 0) t5_matched_pt();
  if (t5_partOfTC_branch != 0) t5_partOfTC();
  if (t5_innerRadius_branch != 0) t5_innerRadius();
  if (t5_outerRadius_branch != 0) t5_outerRadius();
  if (t5_bridgeRadius_branch != 0) t5_bridgeRadius();
  if (t5_chiSquared_branch != 0) t5_chiSquared();
  if (t5_rzChiSquared_branch != 0) t5_rzChiSquared();
  if (t5_nonAnchorChiSquared_branch != 0) t5_nonAnchorChiSquared();
  if (MD_pt_branch != 0) MD_pt();
  if (MD_eta_branch != 0) MD_eta();
  if (MD_phi_branch != 0) MD_phi();
  if (MD_dphichange_branch != 0) MD_dphichange();
  if (MD_isFake_branch != 0) MD_isFake();
  if (MD_tpType_branch != 0) MD_tpType();
  if (MD_detId_branch != 0) MD_detId();
  if (MD_layer_branch != 0) MD_layer();
  if (MD_0_r_branch != 0) MD_0_r();
  if (MD_0_x_branch != 0) MD_0_x();
  if (MD_0_y_branch != 0) MD_0_y();
  if (MD_0_z_branch != 0) MD_0_z();
  if (MD_1_r_branch != 0) MD_1_r();
  if (MD_1_x_branch != 0) MD_1_x();
  if (MD_1_y_branch != 0) MD_1_y();
  if (MD_1_z_branch != 0) MD_1_z();
  if (LS_pt_branch != 0) LS_pt();
  if (LS_eta_branch != 0) LS_eta();
  if (LS_phi_branch != 0) LS_phi();
  if (LS_isFake_branch != 0) LS_isFake();
  if (LS_MD_idx0_branch != 0) LS_MD_idx0();
  if (LS_MD_idx1_branch != 0) LS_MD_idx1();
  if (LS_sim_pt_branch != 0) LS_sim_pt();
  if (LS_sim_eta_branch != 0) LS_sim_eta();
  if (LS_sim_phi_branch != 0) LS_sim_phi();
  if (LS_sim_pca_dxy_branch != 0) LS_sim_pca_dxy();
  if (LS_sim_pca_dz_branch != 0) LS_sim_pca_dz();
  if (LS_sim_q_branch != 0) LS_sim_q();
  if (LS_sim_pdgId_branch != 0) LS_sim_pdgId();
  if (LS_sim_event_branch != 0) LS_sim_event();
  if (LS_sim_bx_branch != 0) LS_sim_bx();
  if (LS_sim_vx_branch != 0) LS_sim_vx();
  if (LS_sim_vy_branch != 0) LS_sim_vy();
  if (LS_sim_vz_branch != 0) LS_sim_vz();
  if (LS_isInTrueTC_branch != 0) LS_isInTrueTC();
  if (t5_t3_idx0_branch != 0) t5_t3_idx0();
  if (t5_t3_idx1_branch != 0) t5_t3_idx1();
  if (t3_isFake_branch != 0) t3_isFake();
  if (t3_ptLegacy_branch != 0) t3_ptLegacy();
  if (t3_pt_branch != 0) t3_pt();
  if (t3_eta_branch != 0) t3_eta();
  if (t3_phi_branch != 0) t3_phi();
  if (t3_0_r_branch != 0) t3_0_r();
  if (t3_0_dr_branch != 0) t3_0_dr();
  if (t3_0_x_branch != 0) t3_0_x();
  if (t3_0_y_branch != 0) t3_0_y();
  if (t3_0_z_branch != 0) t3_0_z();
  if (t3_2_r_branch != 0) t3_2_r();
  if (t3_2_dr_branch != 0) t3_2_dr();
  if (t3_2_x_branch != 0) t3_2_x();
  if (t3_2_y_branch != 0) t3_2_y();
  if (t3_2_z_branch != 0) t3_2_z();
  if (t3_4_r_branch != 0) t3_4_r();
  if (t3_4_dr_branch != 0) t3_4_dr();
  if (t3_4_x_branch != 0) t3_4_x();
  if (t3_4_y_branch != 0) t3_4_y();
  if (t3_4_z_branch != 0) t3_4_z();
  if (tc_lsIdx_branch != 0) tc_lsIdx();
}
const vector<float> &SDL::SimTrack_pt() {
  if (not SimTrack_pt_isLoaded) {
    if (SimTrack_pt_branch != 0) {
      SimTrack_pt_branch->GetEntry(index);
    } else {
      printf("branch SimTrack_pt_branch does not exist!\n");
      exit(1);
    }
    SimTrack_pt_isLoaded = true;
  }
  return *SimTrack_pt_;
}
const vector<float> &SDL::SimTrack_eta() {
  if (not SimTrack_eta_isLoaded) {
    if (SimTrack_eta_branch != 0) {
      SimTrack_eta_branch->GetEntry(index);
    } else {
      printf("branch SimTrack_eta_branch does not exist!\n");
      exit(1);
    }
    SimTrack_eta_isLoaded = true;
  }
  return *SimTrack_eta_;
}
const vector<float> &SDL::SimTrack_phi() {
  if (not SimTrack_phi_isLoaded) {
    if (SimTrack_phi_branch != 0) {
      SimTrack_phi_branch->GetEntry(index);
    } else {
      printf("branch SimTrack_phi_branch does not exist!\n");
      exit(1);
    }
    SimTrack_phi_isLoaded = true;
  }
  return *SimTrack_phi_;
}
const vector<float> &SDL::SimTrack_dxy() {
  if (not SimTrack_dxy_isLoaded) {
    if (SimTrack_dxy_branch != 0) {
      SimTrack_dxy_branch->GetEntry(index);
    } else {
      printf("branch SimTrack_dxy_branch does not exist!\n");
      exit(1);
    }
    SimTrack_dxy_isLoaded = true;
  }
  return *SimTrack_dxy_;
}
const vector<float> &SDL::SimTrack_dz() {
  if (not SimTrack_dz_isLoaded) {
    if (SimTrack_dz_branch != 0) {
      SimTrack_dz_branch->GetEntry(index);
    } else {
      printf("branch SimTrack_dz_branch does not exist!\n");
      exit(1);
    }
    SimTrack_dz_isLoaded = true;
  }
  return *SimTrack_dz_;
}
const vector<int> &SDL::SimTrack_charge() {
  if (not SimTrack_charge_isLoaded) {
    if (SimTrack_charge_branch != 0) {
      SimTrack_charge_branch->GetEntry(index);
    } else {
      printf("branch SimTrack_charge_branch does not exist!\n");
      exit(1);
    }
    SimTrack_charge_isLoaded = true;
  }
  return *SimTrack_charge_;
}
const vector<int> &SDL::SimTrack_ppVtx() {
  if (not SimTrack_ppVtx_isLoaded) {
    if (SimTrack_ppVtx_branch != 0) {
      SimTrack_ppVtx_branch->GetEntry(index);
    } else {
      printf("branch SimTrack_ppVtx_branch does not exist!\n");
      exit(1);
    }
    SimTrack_ppVtx_isLoaded = true;
  }
  return *SimTrack_ppVtx_;
}
const vector<int> &SDL::SimTrack_pdgID() {
  if (not SimTrack_pdgID_isLoaded) {
    if (SimTrack_pdgID_branch != 0) {
      SimTrack_pdgID_branch->GetEntry(index);
    } else {
      printf("branch SimTrack_pdgID_branch does not exist!\n");
      exit(1);
    }
    SimTrack_pdgID_isLoaded = true;
  }
  return *SimTrack_pdgID_;
}
const vector<float> &SDL::SimTrack_vx() {
  if (not SimTrack_vx_isLoaded) {
    if (SimTrack_vx_branch != 0) {
      SimTrack_vx_branch->GetEntry(index);
    } else {
      printf("branch SimTrack_vx_branch does not exist!\n");
      exit(1);
    }
    SimTrack_vx_isLoaded = true;
  }
  return *SimTrack_vx_;
}
const vector<float> &SDL::SimTrack_vy() {
  if (not SimTrack_vy_isLoaded) {
    if (SimTrack_vy_branch != 0) {
      SimTrack_vy_branch->GetEntry(index);
    } else {
      printf("branch SimTrack_vy_branch does not exist!\n");
      exit(1);
    }
    SimTrack_vy_isLoaded = true;
  }
  return *SimTrack_vy_;
}
const vector<float> &SDL::SimTrack_vz() {
  if (not SimTrack_vz_isLoaded) {
    if (SimTrack_vz_branch != 0) {
      SimTrack_vz_branch->GetEntry(index);
    } else {
      printf("branch SimTrack_vz_branch does not exist!\n");
      exit(1);
    }
    SimTrack_vz_isLoaded = true;
  }
  return *SimTrack_vz_;
}
const vector<vector<int> > &SDL::SimTrack_TC_idx() {
  if (not SimTrack_TC_idx_isLoaded) {
    if (SimTrack_TC_idx_branch != 0) {
      SimTrack_TC_idx_branch->GetEntry(index);
    } else {
      printf("branch SimTrack_TC_idx_branch does not exist!\n");
      exit(1);
    }
    SimTrack_TC_idx_isLoaded = true;
  }
  return *SimTrack_TC_idx_;
}
const vector<int> &SDL::SimTrack_TC_typemask() {
  if (not SimTrack_TC_typemask_isLoaded) {
    if (SimTrack_TC_typemask_branch != 0) {
      SimTrack_TC_typemask_branch->GetEntry(index);
    } else {
      printf("branch SimTrack_TC_typemask_branch does not exist!\n");
      exit(1);
    }
    SimTrack_TC_typemask_isLoaded = true;
  }
  return *SimTrack_TC_typemask_;
}
const vector<float> &SDL::TC_pt() {
  if (not TC_pt_isLoaded) {
    if (TC_pt_branch != 0) {
      TC_pt_branch->GetEntry(index);
    } else {
      printf("branch TC_pt_branch does not exist!\n");
      exit(1);
    }
    TC_pt_isLoaded = true;
  }
  return *TC_pt_;
}
const vector<float> &SDL::TC_eta() {
  if (not TC_eta_isLoaded) {
    if (TC_eta_branch != 0) {
      TC_eta_branch->GetEntry(index);
    } else {
      printf("branch TC_eta_branch does not exist!\n");
      exit(1);
    }
    TC_eta_isLoaded = true;
  }
  return *TC_eta_;
}
const vector<float> &SDL::TC_phi() {
  if (not TC_phi_isLoaded) {
    if (TC_phi_branch != 0) {
      TC_phi_branch->GetEntry(index);
    } else {
      printf("branch TC_phi_branch does not exist!\n");
      exit(1);
    }
    TC_phi_isLoaded = true;
  }
  return *TC_phi_;
}
const vector<float> &SDL::TC_dxy() {
  if (not TC_dxy_isLoaded) {
    if (TC_dxy_branch != 0) {
      TC_dxy_branch->GetEntry(index);
    } else {
      printf("branch TC_dxy_branch does not exist!\n");
      exit(1);
    }
    TC_dxy_isLoaded = true;
  }
  return *TC_dxy_;
}
const vector<float> &SDL::TC_dz() {
  if (not TC_dz_isLoaded) {
    if (TC_dz_branch != 0) {
      TC_dz_branch->GetEntry(index);
    } else {
      printf("branch TC_dz_branch does not exist!\n");
      exit(1);
    }
    TC_dz_isLoaded = true;
  }
  return *TC_dz_;
}
const vector<int> &SDL::TC_charge() {
  if (not TC_charge_isLoaded) {
    if (TC_charge_branch != 0) {
      TC_charge_branch->GetEntry(index);
    } else {
      printf("branch TC_charge_branch does not exist!\n");
      exit(1);
    }
    TC_charge_isLoaded = true;
  }
  return *TC_charge_;
}
const vector<int> &SDL::TC_type() {
  if (not TC_type_isLoaded) {
    if (TC_type_branch != 0) {
      TC_type_branch->GetEntry(index);
    } else {
      printf("branch TC_type_branch does not exist!\n");
      exit(1);
    }
    TC_type_isLoaded = true;
  }
  return *TC_type_;
}
const vector<int> &SDL::TC_isFake() {
  if (not TC_isFake_isLoaded) {
    if (TC_isFake_branch != 0) {
      TC_isFake_branch->GetEntry(index);
    } else {
      printf("branch TC_isFake_branch does not exist!\n");
      exit(1);
    }
    TC_isFake_isLoaded = true;
  }
  return *TC_isFake_;
}
const vector<int> &SDL::TC_isDuplicate() {
  if (not TC_isDuplicate_isLoaded) {
    if (TC_isDuplicate_branch != 0) {
      TC_isDuplicate_branch->GetEntry(index);
    } else {
      printf("branch TC_isDuplicate_branch does not exist!\n");
      exit(1);
    }
    TC_isDuplicate_isLoaded = true;
  }
  return *TC_isDuplicate_;
}
const vector<vector<int> > &SDL::TC_SimTrack_idx() {
  if (not TC_SimTrack_idx_isLoaded) {
    if (TC_SimTrack_idx_branch != 0) {
      TC_SimTrack_idx_branch->GetEntry(index);
    } else {
      printf("branch TC_SimTrack_idx_branch does not exist!\n");
      exit(1);
    }
    TC_SimTrack_idx_isLoaded = true;
  }
  return *TC_SimTrack_idx_;
}
const vector<float> &SDL::sim_pt() {
  if (not sim_pt_isLoaded) {
    if (sim_pt_branch != 0) {
      sim_pt_branch->GetEntry(index);
    } else {
      printf("branch sim_pt_branch does not exist!\n");
      exit(1);
    }
    sim_pt_isLoaded = true;
  }
  return *sim_pt_;
}
const vector<float> &SDL::sim_eta() {
  if (not sim_eta_isLoaded) {
    if (sim_eta_branch != 0) {
      sim_eta_branch->GetEntry(index);
    } else {
      printf("branch sim_eta_branch does not exist!\n");
      exit(1);
    }
    sim_eta_isLoaded = true;
  }
  return *sim_eta_;
}
const vector<float> &SDL::sim_phi() {
  if (not sim_phi_isLoaded) {
    if (sim_phi_branch != 0) {
      sim_phi_branch->GetEntry(index);
    } else {
      printf("branch sim_phi_branch does not exist!\n");
      exit(1);
    }
    sim_phi_isLoaded = true;
  }
  return *sim_phi_;
}
const vector<float> &SDL::sim_pca_dxy() {
  if (not sim_pca_dxy_isLoaded) {
    if (sim_pca_dxy_branch != 0) {
      sim_pca_dxy_branch->GetEntry(index);
    } else {
      printf("branch sim_pca_dxy_branch does not exist!\n");
      exit(1);
    }
    sim_pca_dxy_isLoaded = true;
  }
  return *sim_pca_dxy_;
}
const vector<float> &SDL::sim_pca_dz() {
  if (not sim_pca_dz_isLoaded) {
    if (sim_pca_dz_branch != 0) {
      sim_pca_dz_branch->GetEntry(index);
    } else {
      printf("branch sim_pca_dz_branch does not exist!\n");
      exit(1);
    }
    sim_pca_dz_isLoaded = true;
  }
  return *sim_pca_dz_;
}
const vector<int> &SDL::sim_q() {
  if (not sim_q_isLoaded) {
    if (sim_q_branch != 0) {
      sim_q_branch->GetEntry(index);
    } else {
      printf("branch sim_q_branch does not exist!\n");
      exit(1);
    }
    sim_q_isLoaded = true;
  }
  return *sim_q_;
}
const vector<int> &SDL::sim_event() {
  if (not sim_event_isLoaded) {
    if (sim_event_branch != 0) {
      sim_event_branch->GetEntry(index);
    } else {
      printf("branch sim_event_branch does not exist!\n");
      exit(1);
    }
    sim_event_isLoaded = true;
  }
  return *sim_event_;
}
const vector<int> &SDL::sim_pdgId() {
  if (not sim_pdgId_isLoaded) {
    if (sim_pdgId_branch != 0) {
      sim_pdgId_branch->GetEntry(index);
    } else {
      printf("branch sim_pdgId_branch does not exist!\n");
      exit(1);
    }
    sim_pdgId_isLoaded = true;
  }
  return *sim_pdgId_;
}
const vector<float> &SDL::sim_vx() {
  if (not sim_vx_isLoaded) {
    if (sim_vx_branch != 0) {
      sim_vx_branch->GetEntry(index);
    } else {
      printf("branch sim_vx_branch does not exist!\n");
      exit(1);
    }
    sim_vx_isLoaded = true;
  }
  return *sim_vx_;
}
const vector<float> &SDL::sim_vy() {
  if (not sim_vy_isLoaded) {
    if (sim_vy_branch != 0) {
      sim_vy_branch->GetEntry(index);
    } else {
      printf("branch sim_vy_branch does not exist!\n");
      exit(1);
    }
    sim_vy_isLoaded = true;
  }
  return *sim_vy_;
}
const vector<float> &SDL::sim_vz() {
  if (not sim_vz_isLoaded) {
    if (sim_vz_branch != 0) {
      sim_vz_branch->GetEntry(index);
    } else {
      printf("branch sim_vz_branch does not exist!\n");
      exit(1);
    }
    sim_vz_isLoaded = true;
  }
  return *sim_vz_;
}
const vector<float> &SDL::sim_trkNtupIdx() {
  if (not sim_trkNtupIdx_isLoaded) {
    if (sim_trkNtupIdx_branch != 0) {
      sim_trkNtupIdx_branch->GetEntry(index);
    } else {
      printf("branch sim_trkNtupIdx_branch does not exist!\n");
      exit(1);
    }
    sim_trkNtupIdx_isLoaded = true;
  }
  return *sim_trkNtupIdx_;
}
const vector<int> &SDL::sim_TC_matched() {
  if (not sim_TC_matched_isLoaded) {
    if (sim_TC_matched_branch != 0) {
      sim_TC_matched_branch->GetEntry(index);
    } else {
      printf("branch sim_TC_matched_branch does not exist!\n");
      exit(1);
    }
    sim_TC_matched_isLoaded = true;
  }
  return *sim_TC_matched_;
}
const vector<int> &SDL::sim_TC_matched_mask() {
  if (not sim_TC_matched_mask_isLoaded) {
    if (sim_TC_matched_mask_branch != 0) {
      sim_TC_matched_mask_branch->GetEntry(index);
    } else {
      printf("branch sim_TC_matched_mask_branch does not exist!\n");
      exit(1);
    }
    sim_TC_matched_mask_isLoaded = true;
  }
  return *sim_TC_matched_mask_;
}
const vector<float> &SDL::tc_pt() {
  if (not tc_pt_isLoaded) {
    if (tc_pt_branch != 0) {
      tc_pt_branch->GetEntry(index);
    } else {
      printf("branch tc_pt_branch does not exist!\n");
      exit(1);
    }
    tc_pt_isLoaded = true;
  }
  return *tc_pt_;
}
const vector<float> &SDL::tc_eta() {
  if (not tc_eta_isLoaded) {
    if (tc_eta_branch != 0) {
      tc_eta_branch->GetEntry(index);
    } else {
      printf("branch tc_eta_branch does not exist!\n");
      exit(1);
    }
    tc_eta_isLoaded = true;
  }
  return *tc_eta_;
}
const vector<float> &SDL::tc_phi() {
  if (not tc_phi_isLoaded) {
    if (tc_phi_branch != 0) {
      tc_phi_branch->GetEntry(index);
    } else {
      printf("branch tc_phi_branch does not exist!\n");
      exit(1);
    }
    tc_phi_isLoaded = true;
  }
  return *tc_phi_;
}
const vector<int> &SDL::tc_type() {
  if (not tc_type_isLoaded) {
    if (tc_type_branch != 0) {
      tc_type_branch->GetEntry(index);
    } else {
      printf("branch tc_type_branch does not exist!\n");
      exit(1);
    }
    tc_type_isLoaded = true;
  }
  return *tc_type_;
}
const vector<int> &SDL::tc_isFake() {
  if (not tc_isFake_isLoaded) {
    if (tc_isFake_branch != 0) {
      tc_isFake_branch->GetEntry(index);
    } else {
      printf("branch tc_isFake_branch does not exist!\n");
      exit(1);
    }
    tc_isFake_isLoaded = true;
  }
  return *tc_isFake_;
}
const vector<int> &SDL::tc_isDuplicate() {
  if (not tc_isDuplicate_isLoaded) {
    if (tc_isDuplicate_branch != 0) {
      tc_isDuplicate_branch->GetEntry(index);
    } else {
      printf("branch tc_isDuplicate_branch does not exist!\n");
      exit(1);
    }
    tc_isDuplicate_isLoaded = true;
  }
  return *tc_isDuplicate_;
}
const vector<vector<int> > &SDL::tc_matched_simIdx() {
  if (not tc_matched_simIdx_isLoaded) {
    if (tc_matched_simIdx_branch != 0) {
      tc_matched_simIdx_branch->GetEntry(index);
    } else {
      printf("branch tc_matched_simIdx_branch does not exist!\n");
      exit(1);
    }
    tc_matched_simIdx_isLoaded = true;
  }
  return *tc_matched_simIdx_;
}
const vector<float> &SDL::sim_dummy() {
  if (not sim_dummy_isLoaded) {
    if (sim_dummy_branch != 0) {
      sim_dummy_branch->GetEntry(index);
    } else {
      printf("branch sim_dummy_branch does not exist!\n");
      exit(1);
    }
    sim_dummy_isLoaded = true;
  }
  return *sim_dummy_;
}
const vector<float> &SDL::tc_dummy() {
  if (not tc_dummy_isLoaded) {
    if (tc_dummy_branch != 0) {
      tc_dummy_branch->GetEntry(index);
    } else {
      printf("branch tc_dummy_branch does not exist!\n");
      exit(1);
    }
    tc_dummy_isLoaded = true;
  }
  return *tc_dummy_;
}
const vector<vector<int> > &SDL::pT5_matched_simIdx() {
  if (not pT5_matched_simIdx_isLoaded) {
    if (pT5_matched_simIdx_branch != 0) {
      pT5_matched_simIdx_branch->GetEntry(index);
    } else {
      printf("branch pT5_matched_simIdx_branch does not exist!\n");
      exit(1);
    }
    pT5_matched_simIdx_isLoaded = true;
  }
  return *pT5_matched_simIdx_;
}
const vector<vector<int> > &SDL::pT5_hitIdxs() {
  if (not pT5_hitIdxs_isLoaded) {
    if (pT5_hitIdxs_branch != 0) {
      pT5_hitIdxs_branch->GetEntry(index);
    } else {
      printf("branch pT5_hitIdxs_branch does not exist!\n");
      exit(1);
    }
    pT5_hitIdxs_isLoaded = true;
  }
  return *pT5_hitIdxs_;
}
const vector<int> &SDL::sim_pT5_matched() {
  if (not sim_pT5_matched_isLoaded) {
    if (sim_pT5_matched_branch != 0) {
      sim_pT5_matched_branch->GetEntry(index);
    } else {
      printf("branch sim_pT5_matched_branch does not exist!\n");
      exit(1);
    }
    sim_pT5_matched_isLoaded = true;
  }
  return *sim_pT5_matched_;
}
const vector<float> &SDL::pT5_pt() {
  if (not pT5_pt_isLoaded) {
    if (pT5_pt_branch != 0) {
      pT5_pt_branch->GetEntry(index);
    } else {
      printf("branch pT5_pt_branch does not exist!\n");
      exit(1);
    }
    pT5_pt_isLoaded = true;
  }
  return *pT5_pt_;
}
const vector<float> &SDL::pT5_eta() {
  if (not pT5_eta_isLoaded) {
    if (pT5_eta_branch != 0) {
      pT5_eta_branch->GetEntry(index);
    } else {
      printf("branch pT5_eta_branch does not exist!\n");
      exit(1);
    }
    pT5_eta_isLoaded = true;
  }
  return *pT5_eta_;
}
const vector<float> &SDL::pT5_phi() {
  if (not pT5_phi_isLoaded) {
    if (pT5_phi_branch != 0) {
      pT5_phi_branch->GetEntry(index);
    } else {
      printf("branch pT5_phi_branch does not exist!\n");
      exit(1);
    }
    pT5_phi_isLoaded = true;
  }
  return *pT5_phi_;
}
const vector<int> &SDL::pT5_isFake() {
  if (not pT5_isFake_isLoaded) {
    if (pT5_isFake_branch != 0) {
      pT5_isFake_branch->GetEntry(index);
    } else {
      printf("branch pT5_isFake_branch does not exist!\n");
      exit(1);
    }
    pT5_isFake_isLoaded = true;
  }
  return *pT5_isFake_;
}
const vector<int> &SDL::pT5_isDuplicate() {
  if (not pT5_isDuplicate_isLoaded) {
    if (pT5_isDuplicate_branch != 0) {
      pT5_isDuplicate_branch->GetEntry(index);
    } else {
      printf("branch pT5_isDuplicate_branch does not exist!\n");
      exit(1);
    }
    pT5_isDuplicate_isLoaded = true;
  }
  return *pT5_isDuplicate_;
}
const vector<int> &SDL::pT5_score() {
  if (not pT5_score_isLoaded) {
    if (pT5_score_branch != 0) {
      pT5_score_branch->GetEntry(index);
    } else {
      printf("branch pT5_score_branch does not exist!\n");
      exit(1);
    }
    pT5_score_isLoaded = true;
  }
  return *pT5_score_;
}
const vector<int> &SDL::pT5_layer_binary() {
  if (not pT5_layer_binary_isLoaded) {
    if (pT5_layer_binary_branch != 0) {
      pT5_layer_binary_branch->GetEntry(index);
    } else {
      printf("branch pT5_layer_binary_branch does not exist!\n");
      exit(1);
    }
    pT5_layer_binary_isLoaded = true;
  }
  return *pT5_layer_binary_;
}
const vector<int> &SDL::pT5_moduleType_binary() {
  if (not pT5_moduleType_binary_isLoaded) {
    if (pT5_moduleType_binary_branch != 0) {
      pT5_moduleType_binary_branch->GetEntry(index);
    } else {
      printf("branch pT5_moduleType_binary_branch does not exist!\n");
      exit(1);
    }
    pT5_moduleType_binary_isLoaded = true;
  }
  return *pT5_moduleType_binary_;
}
const vector<float> &SDL::pT5_matched_pt() {
  if (not pT5_matched_pt_isLoaded) {
    if (pT5_matched_pt_branch != 0) {
      pT5_matched_pt_branch->GetEntry(index);
    } else {
      printf("branch pT5_matched_pt_branch does not exist!\n");
      exit(1);
    }
    pT5_matched_pt_isLoaded = true;
  }
  return *pT5_matched_pt_;
}
const vector<float> &SDL::pT5_rzChiSquared() {
  if (not pT5_rzChiSquared_isLoaded) {
    if (pT5_rzChiSquared_branch != 0) {
      pT5_rzChiSquared_branch->GetEntry(index);
    } else {
      printf("branch pT5_rzChiSquared_branch does not exist!\n");
      exit(1);
    }
    pT5_rzChiSquared_isLoaded = true;
  }
  return *pT5_rzChiSquared_;
}
const vector<float> &SDL::pT5_rPhiChiSquared() {
  if (not pT5_rPhiChiSquared_isLoaded) {
    if (pT5_rPhiChiSquared_branch != 0) {
      pT5_rPhiChiSquared_branch->GetEntry(index);
    } else {
      printf("branch pT5_rPhiChiSquared_branch does not exist!\n");
      exit(1);
    }
    pT5_rPhiChiSquared_isLoaded = true;
  }
  return *pT5_rPhiChiSquared_;
}
const vector<float> &SDL::pT5_rPhiChiSquaredInwards() {
  if (not pT5_rPhiChiSquaredInwards_isLoaded) {
    if (pT5_rPhiChiSquaredInwards_branch != 0) {
      pT5_rPhiChiSquaredInwards_branch->GetEntry(index);
    } else {
      printf("branch pT5_rPhiChiSquaredInwards_branch does not exist!\n");
      exit(1);
    }
    pT5_rPhiChiSquaredInwards_isLoaded = true;
  }
  return *pT5_rPhiChiSquaredInwards_;
}
const vector<int> &SDL::sim_pT3_matched() {
  if (not sim_pT3_matched_isLoaded) {
    if (sim_pT3_matched_branch != 0) {
      sim_pT3_matched_branch->GetEntry(index);
    } else {
      printf("branch sim_pT3_matched_branch does not exist!\n");
      exit(1);
    }
    sim_pT3_matched_isLoaded = true;
  }
  return *sim_pT3_matched_;
}
const vector<float> &SDL::pT3_pt() {
  if (not pT3_pt_isLoaded) {
    if (pT3_pt_branch != 0) {
      pT3_pt_branch->GetEntry(index);
    } else {
      printf("branch pT3_pt_branch does not exist!\n");
      exit(1);
    }
    pT3_pt_isLoaded = true;
  }
  return *pT3_pt_;
}
const vector<int> &SDL::pT3_isFake() {
  if (not pT3_isFake_isLoaded) {
    if (pT3_isFake_branch != 0) {
      pT3_isFake_branch->GetEntry(index);
    } else {
      printf("branch pT3_isFake_branch does not exist!\n");
      exit(1);
    }
    pT3_isFake_isLoaded = true;
  }
  return *pT3_isFake_;
}
const vector<int> &SDL::pT3_isDuplicate() {
  if (not pT3_isDuplicate_isLoaded) {
    if (pT3_isDuplicate_branch != 0) {
      pT3_isDuplicate_branch->GetEntry(index);
    } else {
      printf("branch pT3_isDuplicate_branch does not exist!\n");
      exit(1);
    }
    pT3_isDuplicate_isLoaded = true;
  }
  return *pT3_isDuplicate_;
}
const vector<float> &SDL::pT3_eta() {
  if (not pT3_eta_isLoaded) {
    if (pT3_eta_branch != 0) {
      pT3_eta_branch->GetEntry(index);
    } else {
      printf("branch pT3_eta_branch does not exist!\n");
      exit(1);
    }
    pT3_eta_isLoaded = true;
  }
  return *pT3_eta_;
}
const vector<float> &SDL::pT3_phi() {
  if (not pT3_phi_isLoaded) {
    if (pT3_phi_branch != 0) {
      pT3_phi_branch->GetEntry(index);
    } else {
      printf("branch pT3_phi_branch does not exist!\n");
      exit(1);
    }
    pT3_phi_isLoaded = true;
  }
  return *pT3_phi_;
}
const vector<float> &SDL::pT3_score() {
  if (not pT3_score_isLoaded) {
    if (pT3_score_branch != 0) {
      pT3_score_branch->GetEntry(index);
    } else {
      printf("branch pT3_score_branch does not exist!\n");
      exit(1);
    }
    pT3_score_isLoaded = true;
  }
  return *pT3_score_;
}
const vector<int> &SDL::pT3_foundDuplicate() {
  if (not pT3_foundDuplicate_isLoaded) {
    if (pT3_foundDuplicate_branch != 0) {
      pT3_foundDuplicate_branch->GetEntry(index);
    } else {
      printf("branch pT3_foundDuplicate_branch does not exist!\n");
      exit(1);
    }
    pT3_foundDuplicate_isLoaded = true;
  }
  return *pT3_foundDuplicate_;
}
const vector<vector<int> > &SDL::pT3_matched_simIdx() {
  if (not pT3_matched_simIdx_isLoaded) {
    if (pT3_matched_simIdx_branch != 0) {
      pT3_matched_simIdx_branch->GetEntry(index);
    } else {
      printf("branch pT3_matched_simIdx_branch does not exist!\n");
      exit(1);
    }
    pT3_matched_simIdx_isLoaded = true;
  }
  return *pT3_matched_simIdx_;
}
const vector<vector<int> > &SDL::pT3_hitIdxs() {
  if (not pT3_hitIdxs_isLoaded) {
    if (pT3_hitIdxs_branch != 0) {
      pT3_hitIdxs_branch->GetEntry(index);
    } else {
      printf("branch pT3_hitIdxs_branch does not exist!\n");
      exit(1);
    }
    pT3_hitIdxs_isLoaded = true;
  }
  return *pT3_hitIdxs_;
}
const vector<float> &SDL::pT3_pixelRadius() {
  if (not pT3_pixelRadius_isLoaded) {
    if (pT3_pixelRadius_branch != 0) {
      pT3_pixelRadius_branch->GetEntry(index);
    } else {
      printf("branch pT3_pixelRadius_branch does not exist!\n");
      exit(1);
    }
    pT3_pixelRadius_isLoaded = true;
  }
  return *pT3_pixelRadius_;
}
const vector<float> &SDL::pT3_pixelRadiusError() {
  if (not pT3_pixelRadiusError_isLoaded) {
    if (pT3_pixelRadiusError_branch != 0) {
      pT3_pixelRadiusError_branch->GetEntry(index);
    } else {
      printf("branch pT3_pixelRadiusError_branch does not exist!\n");
      exit(1);
    }
    pT3_pixelRadiusError_isLoaded = true;
  }
  return *pT3_pixelRadiusError_;
}
const vector<vector<float> > &SDL::pT3_matched_pt() {
  if (not pT3_matched_pt_isLoaded) {
    if (pT3_matched_pt_branch != 0) {
      pT3_matched_pt_branch->GetEntry(index);
    } else {
      printf("branch pT3_matched_pt_branch does not exist!\n");
      exit(1);
    }
    pT3_matched_pt_isLoaded = true;
  }
  return *pT3_matched_pt_;
}
const vector<float> &SDL::pT3_tripletRadius() {
  if (not pT3_tripletRadius_isLoaded) {
    if (pT3_tripletRadius_branch != 0) {
      pT3_tripletRadius_branch->GetEntry(index);
    } else {
      printf("branch pT3_tripletRadius_branch does not exist!\n");
      exit(1);
    }
    pT3_tripletRadius_isLoaded = true;
  }
  return *pT3_tripletRadius_;
}
const vector<float> &SDL::pT3_rPhiChiSquared() {
  if (not pT3_rPhiChiSquared_isLoaded) {
    if (pT3_rPhiChiSquared_branch != 0) {
      pT3_rPhiChiSquared_branch->GetEntry(index);
    } else {
      printf("branch pT3_rPhiChiSquared_branch does not exist!\n");
      exit(1);
    }
    pT3_rPhiChiSquared_isLoaded = true;
  }
  return *pT3_rPhiChiSquared_;
}
const vector<float> &SDL::pT3_rPhiChiSquaredInwards() {
  if (not pT3_rPhiChiSquaredInwards_isLoaded) {
    if (pT3_rPhiChiSquaredInwards_branch != 0) {
      pT3_rPhiChiSquaredInwards_branch->GetEntry(index);
    } else {
      printf("branch pT3_rPhiChiSquaredInwards_branch does not exist!\n");
      exit(1);
    }
    pT3_rPhiChiSquaredInwards_isLoaded = true;
  }
  return *pT3_rPhiChiSquaredInwards_;
}
const vector<float> &SDL::pT3_rzChiSquared() {
  if (not pT3_rzChiSquared_isLoaded) {
    if (pT3_rzChiSquared_branch != 0) {
      pT3_rzChiSquared_branch->GetEntry(index);
    } else {
      printf("branch pT3_rzChiSquared_branch does not exist!\n");
      exit(1);
    }
    pT3_rzChiSquared_isLoaded = true;
  }
  return *pT3_rzChiSquared_;
}
const vector<int> &SDL::pT3_layer_binary() {
  if (not pT3_layer_binary_isLoaded) {
    if (pT3_layer_binary_branch != 0) {
      pT3_layer_binary_branch->GetEntry(index);
    } else {
      printf("branch pT3_layer_binary_branch does not exist!\n");
      exit(1);
    }
    pT3_layer_binary_isLoaded = true;
  }
  return *pT3_layer_binary_;
}
const vector<int> &SDL::pT3_moduleType_binary() {
  if (not pT3_moduleType_binary_isLoaded) {
    if (pT3_moduleType_binary_branch != 0) {
      pT3_moduleType_binary_branch->GetEntry(index);
    } else {
      printf("branch pT3_moduleType_binary_branch does not exist!\n");
      exit(1);
    }
    pT3_moduleType_binary_isLoaded = true;
  }
  return *pT3_moduleType_binary_;
}
const vector<int> &SDL::sim_pLS_matched() {
  if (not sim_pLS_matched_isLoaded) {
    if (sim_pLS_matched_branch != 0) {
      sim_pLS_matched_branch->GetEntry(index);
    } else {
      printf("branch sim_pLS_matched_branch does not exist!\n");
      exit(1);
    }
    sim_pLS_matched_isLoaded = true;
  }
  return *sim_pLS_matched_;
}
const vector<vector<int> > &SDL::sim_pLS_types() {
  if (not sim_pLS_types_isLoaded) {
    if (sim_pLS_types_branch != 0) {
      sim_pLS_types_branch->GetEntry(index);
    } else {
      printf("branch sim_pLS_types_branch does not exist!\n");
      exit(1);
    }
    sim_pLS_types_isLoaded = true;
  }
  return *sim_pLS_types_;
}
const vector<int> &SDL::pLS_isFake() {
  if (not pLS_isFake_isLoaded) {
    if (pLS_isFake_branch != 0) {
      pLS_isFake_branch->GetEntry(index);
    } else {
      printf("branch pLS_isFake_branch does not exist!\n");
      exit(1);
    }
    pLS_isFake_isLoaded = true;
  }
  return *pLS_isFake_;
}
const vector<int> &SDL::pLS_isDuplicate() {
  if (not pLS_isDuplicate_isLoaded) {
    if (pLS_isDuplicate_branch != 0) {
      pLS_isDuplicate_branch->GetEntry(index);
    } else {
      printf("branch pLS_isDuplicate_branch does not exist!\n");
      exit(1);
    }
    pLS_isDuplicate_isLoaded = true;
  }
  return *pLS_isDuplicate_;
}
const vector<float> &SDL::pLS_pt() {
  if (not pLS_pt_isLoaded) {
    if (pLS_pt_branch != 0) {
      pLS_pt_branch->GetEntry(index);
    } else {
      printf("branch pLS_pt_branch does not exist!\n");
      exit(1);
    }
    pLS_pt_isLoaded = true;
  }
  return *pLS_pt_;
}
const vector<float> &SDL::pLS_eta() {
  if (not pLS_eta_isLoaded) {
    if (pLS_eta_branch != 0) {
      pLS_eta_branch->GetEntry(index);
    } else {
      printf("branch pLS_eta_branch does not exist!\n");
      exit(1);
    }
    pLS_eta_isLoaded = true;
  }
  return *pLS_eta_;
}
const vector<float> &SDL::pLS_phi() {
  if (not pLS_phi_isLoaded) {
    if (pLS_phi_branch != 0) {
      pLS_phi_branch->GetEntry(index);
    } else {
      printf("branch pLS_phi_branch does not exist!\n");
      exit(1);
    }
    pLS_phi_isLoaded = true;
  }
  return *pLS_phi_;
}
const vector<float> &SDL::pLS_score() {
  if (not pLS_score_isLoaded) {
    if (pLS_score_branch != 0) {
      pLS_score_branch->GetEntry(index);
    } else {
      printf("branch pLS_score_branch does not exist!\n");
      exit(1);
    }
    pLS_score_isLoaded = true;
  }
  return *pLS_score_;
}
const vector<int> &SDL::sim_T5_matched() {
  if (not sim_T5_matched_isLoaded) {
    if (sim_T5_matched_branch != 0) {
      sim_T5_matched_branch->GetEntry(index);
    } else {
      printf("branch sim_T5_matched_branch does not exist!\n");
      exit(1);
    }
    sim_T5_matched_isLoaded = true;
  }
  return *sim_T5_matched_;
}
const vector<int> &SDL::t5_isFake() {
  if (not t5_isFake_isLoaded) {
    if (t5_isFake_branch != 0) {
      t5_isFake_branch->GetEntry(index);
    } else {
      printf("branch t5_isFake_branch does not exist!\n");
      exit(1);
    }
    t5_isFake_isLoaded = true;
  }
  return *t5_isFake_;
}
const vector<int> &SDL::t5_isDuplicate() {
  if (not t5_isDuplicate_isLoaded) {
    if (t5_isDuplicate_branch != 0) {
      t5_isDuplicate_branch->GetEntry(index);
    } else {
      printf("branch t5_isDuplicate_branch does not exist!\n");
      exit(1);
    }
    t5_isDuplicate_isLoaded = true;
  }
  return *t5_isDuplicate_;
}
const vector<int> &SDL::t5_foundDuplicate() {
  if (not t5_foundDuplicate_isLoaded) {
    if (t5_foundDuplicate_branch != 0) {
      t5_foundDuplicate_branch->GetEntry(index);
    } else {
      printf("branch t5_foundDuplicate_branch does not exist!\n");
      exit(1);
    }
    t5_foundDuplicate_isLoaded = true;
  }
  return *t5_foundDuplicate_;
}
const vector<float> &SDL::t5_pt() {
  if (not t5_pt_isLoaded) {
    if (t5_pt_branch != 0) {
      t5_pt_branch->GetEntry(index);
    } else {
      printf("branch t5_pt_branch does not exist!\n");
      exit(1);
    }
    t5_pt_isLoaded = true;
  }
  return *t5_pt_;
}
const vector<float> &SDL::t5_eta() {
  if (not t5_eta_isLoaded) {
    if (t5_eta_branch != 0) {
      t5_eta_branch->GetEntry(index);
    } else {
      printf("branch t5_eta_branch does not exist!\n");
      exit(1);
    }
    t5_eta_isLoaded = true;
  }
  return *t5_eta_;
}
const vector<float> &SDL::t5_phi() {
  if (not t5_phi_isLoaded) {
    if (t5_phi_branch != 0) {
      t5_phi_branch->GetEntry(index);
    } else {
      printf("branch t5_phi_branch does not exist!\n");
      exit(1);
    }
    t5_phi_isLoaded = true;
  }
  return *t5_phi_;
}
const vector<float> &SDL::t5_score_rphisum() {
  if (not t5_score_rphisum_isLoaded) {
    if (t5_score_rphisum_branch != 0) {
      t5_score_rphisum_branch->GetEntry(index);
    } else {
      printf("branch t5_score_rphisum_branch does not exist!\n");
      exit(1);
    }
    t5_score_rphisum_isLoaded = true;
  }
  return *t5_score_rphisum_;
}
const vector<vector<int> > &SDL::t5_hitIdxs() {
  if (not t5_hitIdxs_isLoaded) {
    if (t5_hitIdxs_branch != 0) {
      t5_hitIdxs_branch->GetEntry(index);
    } else {
      printf("branch t5_hitIdxs_branch does not exist!\n");
      exit(1);
    }
    t5_hitIdxs_isLoaded = true;
  }
  return *t5_hitIdxs_;
}
const vector<vector<int> > &SDL::t5_matched_simIdx() {
  if (not t5_matched_simIdx_isLoaded) {
    if (t5_matched_simIdx_branch != 0) {
      t5_matched_simIdx_branch->GetEntry(index);
    } else {
      printf("branch t5_matched_simIdx_branch does not exist!\n");
      exit(1);
    }
    t5_matched_simIdx_isLoaded = true;
  }
  return *t5_matched_simIdx_;
}
const vector<int> &SDL::t5_moduleType_binary() {
  if (not t5_moduleType_binary_isLoaded) {
    if (t5_moduleType_binary_branch != 0) {
      t5_moduleType_binary_branch->GetEntry(index);
    } else {
      printf("branch t5_moduleType_binary_branch does not exist!\n");
      exit(1);
    }
    t5_moduleType_binary_isLoaded = true;
  }
  return *t5_moduleType_binary_;
}
const vector<int> &SDL::t5_layer_binary() {
  if (not t5_layer_binary_isLoaded) {
    if (t5_layer_binary_branch != 0) {
      t5_layer_binary_branch->GetEntry(index);
    } else {
      printf("branch t5_layer_binary_branch does not exist!\n");
      exit(1);
    }
    t5_layer_binary_isLoaded = true;
  }
  return *t5_layer_binary_;
}
const vector<float> &SDL::t5_matched_pt() {
  if (not t5_matched_pt_isLoaded) {
    if (t5_matched_pt_branch != 0) {
      t5_matched_pt_branch->GetEntry(index);
    } else {
      printf("branch t5_matched_pt_branch does not exist!\n");
      exit(1);
    }
    t5_matched_pt_isLoaded = true;
  }
  return *t5_matched_pt_;
}
const vector<int> &SDL::t5_partOfTC() {
  if (not t5_partOfTC_isLoaded) {
    if (t5_partOfTC_branch != 0) {
      t5_partOfTC_branch->GetEntry(index);
    } else {
      printf("branch t5_partOfTC_branch does not exist!\n");
      exit(1);
    }
    t5_partOfTC_isLoaded = true;
  }
  return *t5_partOfTC_;
}
const vector<float> &SDL::t5_innerRadius() {
  if (not t5_innerRadius_isLoaded) {
    if (t5_innerRadius_branch != 0) {
      t5_innerRadius_branch->GetEntry(index);
    } else {
      printf("branch t5_innerRadius_branch does not exist!\n");
      exit(1);
    }
    t5_innerRadius_isLoaded = true;
  }
  return *t5_innerRadius_;
}
const vector<float> &SDL::t5_outerRadius() {
  if (not t5_outerRadius_isLoaded) {
    if (t5_outerRadius_branch != 0) {
      t5_outerRadius_branch->GetEntry(index);
    } else {
      printf("branch t5_outerRadius_branch does not exist!\n");
      exit(1);
    }
    t5_outerRadius_isLoaded = true;
  }
  return *t5_outerRadius_;
}
const vector<float> &SDL::t5_bridgeRadius() {
  if (not t5_bridgeRadius_isLoaded) {
    if (t5_bridgeRadius_branch != 0) {
      t5_bridgeRadius_branch->GetEntry(index);
    } else {
      printf("branch t5_bridgeRadius_branch does not exist!\n");
      exit(1);
    }
    t5_bridgeRadius_isLoaded = true;
  }
  return *t5_bridgeRadius_;
}
const vector<float> &SDL::t5_chiSquared() {
  if (not t5_chiSquared_isLoaded) {
    if (t5_chiSquared_branch != 0) {
      t5_chiSquared_branch->GetEntry(index);
    } else {
      printf("branch t5_chiSquared_branch does not exist!\n");
      exit(1);
    }
    t5_chiSquared_isLoaded = true;
  }
  return *t5_chiSquared_;
}
const vector<float> &SDL::t5_rzChiSquared() {
  if (not t5_rzChiSquared_isLoaded) {
    if (t5_rzChiSquared_branch != 0) {
      t5_rzChiSquared_branch->GetEntry(index);
    } else {
      printf("branch t5_rzChiSquared_branch does not exist!\n");
      exit(1);
    }
    t5_rzChiSquared_isLoaded = true;
  }
  return *t5_rzChiSquared_;
}
const vector<float> &SDL::t5_nonAnchorChiSquared() {
  if (not t5_nonAnchorChiSquared_isLoaded) {
    if (t5_nonAnchorChiSquared_branch != 0) {
      t5_nonAnchorChiSquared_branch->GetEntry(index);
    } else {
      printf("branch t5_nonAnchorChiSquared_branch does not exist!\n");
      exit(1);
    }
    t5_nonAnchorChiSquared_isLoaded = true;
  }
  return *t5_nonAnchorChiSquared_;
}
const vector<float> &SDL::MD_pt() {
  if (not MD_pt_isLoaded) {
    if (MD_pt_branch != 0) {
      MD_pt_branch->GetEntry(index);
    } else {
      printf("branch MD_pt_branch does not exist!\n");
      exit(1);
    }
    MD_pt_isLoaded = true;
  }
  return *MD_pt_;
}
const vector<float> &SDL::MD_eta() {
  if (not MD_eta_isLoaded) {
    if (MD_eta_branch != 0) {
      MD_eta_branch->GetEntry(index);
    } else {
      printf("branch MD_eta_branch does not exist!\n");
      exit(1);
    }
    MD_eta_isLoaded = true;
  }
  return *MD_eta_;
}
const vector<float> &SDL::MD_phi() {
  if (not MD_phi_isLoaded) {
    if (MD_phi_branch != 0) {
      MD_phi_branch->GetEntry(index);
    } else {
      printf("branch MD_phi_branch does not exist!\n");
      exit(1);
    }
    MD_phi_isLoaded = true;
  }
  return *MD_phi_;
}
const vector<float> &SDL::MD_dphichange() {
  if (not MD_dphichange_isLoaded) {
    if (MD_dphichange_branch != 0) {
      MD_dphichange_branch->GetEntry(index);
    } else {
      printf("branch MD_dphichange_branch does not exist!\n");
      exit(1);
    }
    MD_dphichange_isLoaded = true;
  }
  return *MD_dphichange_;
}
const vector<int> &SDL::MD_isFake() {
  if (not MD_isFake_isLoaded) {
    if (MD_isFake_branch != 0) {
      MD_isFake_branch->GetEntry(index);
    } else {
      printf("branch MD_isFake_branch does not exist!\n");
      exit(1);
    }
    MD_isFake_isLoaded = true;
  }
  return *MD_isFake_;
}
const vector<int> &SDL::MD_tpType() {
  if (not MD_tpType_isLoaded) {
    if (MD_tpType_branch != 0) {
      MD_tpType_branch->GetEntry(index);
    } else {
      printf("branch MD_tpType_branch does not exist!\n");
      exit(1);
    }
    MD_tpType_isLoaded = true;
  }
  return *MD_tpType_;
}
const vector<int> &SDL::MD_detId() {
  if (not MD_detId_isLoaded) {
    if (MD_detId_branch != 0) {
      MD_detId_branch->GetEntry(index);
    } else {
      printf("branch MD_detId_branch does not exist!\n");
      exit(1);
    }
    MD_detId_isLoaded = true;
  }
  return *MD_detId_;
}
const vector<int> &SDL::MD_layer() {
  if (not MD_layer_isLoaded) {
    if (MD_layer_branch != 0) {
      MD_layer_branch->GetEntry(index);
    } else {
      printf("branch MD_layer_branch does not exist!\n");
      exit(1);
    }
    MD_layer_isLoaded = true;
  }
  return *MD_layer_;
}
const vector<float> &SDL::MD_0_r() {
  if (not MD_0_r_isLoaded) {
    if (MD_0_r_branch != 0) {
      MD_0_r_branch->GetEntry(index);
    } else {
      printf("branch MD_0_r_branch does not exist!\n");
      exit(1);
    }
    MD_0_r_isLoaded = true;
  }
  return *MD_0_r_;
}
const vector<float> &SDL::MD_0_x() {
  if (not MD_0_x_isLoaded) {
    if (MD_0_x_branch != 0) {
      MD_0_x_branch->GetEntry(index);
    } else {
      printf("branch MD_0_x_branch does not exist!\n");
      exit(1);
    }
    MD_0_x_isLoaded = true;
  }
  return *MD_0_x_;
}
const vector<float> &SDL::MD_0_y() {
  if (not MD_0_y_isLoaded) {
    if (MD_0_y_branch != 0) {
      MD_0_y_branch->GetEntry(index);
    } else {
      printf("branch MD_0_y_branch does not exist!\n");
      exit(1);
    }
    MD_0_y_isLoaded = true;
  }
  return *MD_0_y_;
}
const vector<float> &SDL::MD_0_z() {
  if (not MD_0_z_isLoaded) {
    if (MD_0_z_branch != 0) {
      MD_0_z_branch->GetEntry(index);
    } else {
      printf("branch MD_0_z_branch does not exist!\n");
      exit(1);
    }
    MD_0_z_isLoaded = true;
  }
  return *MD_0_z_;
}
const vector<float> &SDL::MD_1_r() {
  if (not MD_1_r_isLoaded) {
    if (MD_1_r_branch != 0) {
      MD_1_r_branch->GetEntry(index);
    } else {
      printf("branch MD_1_r_branch does not exist!\n");
      exit(1);
    }
    MD_1_r_isLoaded = true;
  }
  return *MD_1_r_;
}
const vector<float> &SDL::MD_1_x() {
  if (not MD_1_x_isLoaded) {
    if (MD_1_x_branch != 0) {
      MD_1_x_branch->GetEntry(index);
    } else {
      printf("branch MD_1_x_branch does not exist!\n");
      exit(1);
    }
    MD_1_x_isLoaded = true;
  }
  return *MD_1_x_;
}
const vector<float> &SDL::MD_1_y() {
  if (not MD_1_y_isLoaded) {
    if (MD_1_y_branch != 0) {
      MD_1_y_branch->GetEntry(index);
    } else {
      printf("branch MD_1_y_branch does not exist!\n");
      exit(1);
    }
    MD_1_y_isLoaded = true;
  }
  return *MD_1_y_;
}
const vector<float> &SDL::MD_1_z() {
  if (not MD_1_z_isLoaded) {
    if (MD_1_z_branch != 0) {
      MD_1_z_branch->GetEntry(index);
    } else {
      printf("branch MD_1_z_branch does not exist!\n");
      exit(1);
    }
    MD_1_z_isLoaded = true;
  }
  return *MD_1_z_;
}
const vector<float> &SDL::LS_pt() {
  if (not LS_pt_isLoaded) {
    if (LS_pt_branch != 0) {
      LS_pt_branch->GetEntry(index);
    } else {
      printf("branch LS_pt_branch does not exist!\n");
      exit(1);
    }
    LS_pt_isLoaded = true;
  }
  return *LS_pt_;
}
const vector<float> &SDL::LS_eta() {
  if (not LS_eta_isLoaded) {
    if (LS_eta_branch != 0) {
      LS_eta_branch->GetEntry(index);
    } else {
      printf("branch LS_eta_branch does not exist!\n");
      exit(1);
    }
    LS_eta_isLoaded = true;
  }
  return *LS_eta_;
}
const vector<float> &SDL::LS_phi() {
  if (not LS_phi_isLoaded) {
    if (LS_phi_branch != 0) {
      LS_phi_branch->GetEntry(index);
    } else {
      printf("branch LS_phi_branch does not exist!\n");
      exit(1);
    }
    LS_phi_isLoaded = true;
  }
  return *LS_phi_;
}
const vector<int> &SDL::LS_isFake() {
  if (not LS_isFake_isLoaded) {
    if (LS_isFake_branch != 0) {
      LS_isFake_branch->GetEntry(index);
    } else {
      printf("branch LS_isFake_branch does not exist!\n");
      exit(1);
    }
    LS_isFake_isLoaded = true;
  }
  return *LS_isFake_;
}
const vector<int> &SDL::LS_MD_idx0() {
  if (not LS_MD_idx0_isLoaded) {
    if (LS_MD_idx0_branch != 0) {
      LS_MD_idx0_branch->GetEntry(index);
    } else {
      printf("branch LS_MD_idx0_branch does not exist!\n");
      exit(1);
    }
    LS_MD_idx0_isLoaded = true;
  }
  return *LS_MD_idx0_;
}
const vector<int> &SDL::LS_MD_idx1() {
  if (not LS_MD_idx1_isLoaded) {
    if (LS_MD_idx1_branch != 0) {
      LS_MD_idx1_branch->GetEntry(index);
    } else {
      printf("branch LS_MD_idx1_branch does not exist!\n");
      exit(1);
    }
    LS_MD_idx1_isLoaded = true;
  }
  return *LS_MD_idx1_;
}
const vector<float> &SDL::LS_sim_pt() {
  if (not LS_sim_pt_isLoaded) {
    if (LS_sim_pt_branch != 0) {
      LS_sim_pt_branch->GetEntry(index);
    } else {
      printf("branch LS_sim_pt_branch does not exist!\n");
      exit(1);
    }
    LS_sim_pt_isLoaded = true;
  }
  return *LS_sim_pt_;
}
const vector<float> &SDL::LS_sim_eta() {
  if (not LS_sim_eta_isLoaded) {
    if (LS_sim_eta_branch != 0) {
      LS_sim_eta_branch->GetEntry(index);
    } else {
      printf("branch LS_sim_eta_branch does not exist!\n");
      exit(1);
    }
    LS_sim_eta_isLoaded = true;
  }
  return *LS_sim_eta_;
}
const vector<float> &SDL::LS_sim_phi() {
  if (not LS_sim_phi_isLoaded) {
    if (LS_sim_phi_branch != 0) {
      LS_sim_phi_branch->GetEntry(index);
    } else {
      printf("branch LS_sim_phi_branch does not exist!\n");
      exit(1);
    }
    LS_sim_phi_isLoaded = true;
  }
  return *LS_sim_phi_;
}
const vector<float> &SDL::LS_sim_pca_dxy() {
  if (not LS_sim_pca_dxy_isLoaded) {
    if (LS_sim_pca_dxy_branch != 0) {
      LS_sim_pca_dxy_branch->GetEntry(index);
    } else {
      printf("branch LS_sim_pca_dxy_branch does not exist!\n");
      exit(1);
    }
    LS_sim_pca_dxy_isLoaded = true;
  }
  return *LS_sim_pca_dxy_;
}
const vector<float> &SDL::LS_sim_pca_dz() {
  if (not LS_sim_pca_dz_isLoaded) {
    if (LS_sim_pca_dz_branch != 0) {
      LS_sim_pca_dz_branch->GetEntry(index);
    } else {
      printf("branch LS_sim_pca_dz_branch does not exist!\n");
      exit(1);
    }
    LS_sim_pca_dz_isLoaded = true;
  }
  return *LS_sim_pca_dz_;
}
const vector<int> &SDL::LS_sim_q() {
  if (not LS_sim_q_isLoaded) {
    if (LS_sim_q_branch != 0) {
      LS_sim_q_branch->GetEntry(index);
    } else {
      printf("branch LS_sim_q_branch does not exist!\n");
      exit(1);
    }
    LS_sim_q_isLoaded = true;
  }
  return *LS_sim_q_;
}
const vector<int> &SDL::LS_sim_pdgId() {
  if (not LS_sim_pdgId_isLoaded) {
    if (LS_sim_pdgId_branch != 0) {
      LS_sim_pdgId_branch->GetEntry(index);
    } else {
      printf("branch LS_sim_pdgId_branch does not exist!\n");
      exit(1);
    }
    LS_sim_pdgId_isLoaded = true;
  }
  return *LS_sim_pdgId_;
}
const vector<int> &SDL::LS_sim_event() {
  if (not LS_sim_event_isLoaded) {
    if (LS_sim_event_branch != 0) {
      LS_sim_event_branch->GetEntry(index);
    } else {
      printf("branch LS_sim_event_branch does not exist!\n");
      exit(1);
    }
    LS_sim_event_isLoaded = true;
  }
  return *LS_sim_event_;
}
const vector<int> &SDL::LS_sim_bx() {
  if (not LS_sim_bx_isLoaded) {
    if (LS_sim_bx_branch != 0) {
      LS_sim_bx_branch->GetEntry(index);
    } else {
      printf("branch LS_sim_bx_branch does not exist!\n");
      exit(1);
    }
    LS_sim_bx_isLoaded = true;
  }
  return *LS_sim_bx_;
}
const vector<float> &SDL::LS_sim_vx() {
  if (not LS_sim_vx_isLoaded) {
    if (LS_sim_vx_branch != 0) {
      LS_sim_vx_branch->GetEntry(index);
    } else {
      printf("branch LS_sim_vx_branch does not exist!\n");
      exit(1);
    }
    LS_sim_vx_isLoaded = true;
  }
  return *LS_sim_vx_;
}
const vector<float> &SDL::LS_sim_vy() {
  if (not LS_sim_vy_isLoaded) {
    if (LS_sim_vy_branch != 0) {
      LS_sim_vy_branch->GetEntry(index);
    } else {
      printf("branch LS_sim_vy_branch does not exist!\n");
      exit(1);
    }
    LS_sim_vy_isLoaded = true;
  }
  return *LS_sim_vy_;
}
const vector<float> &SDL::LS_sim_vz() {
  if (not LS_sim_vz_isLoaded) {
    if (LS_sim_vz_branch != 0) {
      LS_sim_vz_branch->GetEntry(index);
    } else {
      printf("branch LS_sim_vz_branch does not exist!\n");
      exit(1);
    }
    LS_sim_vz_isLoaded = true;
  }
  return *LS_sim_vz_;
}
const vector<int> &SDL::LS_isInTrueTC() {
  if (not LS_isInTrueTC_isLoaded) {
    if (LS_isInTrueTC_branch != 0) {
      LS_isInTrueTC_branch->GetEntry(index);
    } else {
      printf("branch LS_isInTrueTC_branch does not exist!\n");
      exit(1);
    }
    LS_isInTrueTC_isLoaded = true;
  }
  return *LS_isInTrueTC_;
}
const vector<int> &SDL::t5_t3_idx0() {
  if (not t5_t3_idx0_isLoaded) {
    if (t5_t3_idx0_branch != 0) {
      t5_t3_idx0_branch->GetEntry(index);
    } else {
      printf("branch t5_t3_idx0_branch does not exist!\n");
      exit(1);
    }
    t5_t3_idx0_isLoaded = true;
  }
  return *t5_t3_idx0_;
}
const vector<int> &SDL::t5_t3_idx1() {
  if (not t5_t3_idx1_isLoaded) {
    if (t5_t3_idx1_branch != 0) {
      t5_t3_idx1_branch->GetEntry(index);
    } else {
      printf("branch t5_t3_idx1_branch does not exist!\n");
      exit(1);
    }
    t5_t3_idx1_isLoaded = true;
  }
  return *t5_t3_idx1_;
}
const vector<int> &SDL::t3_isFake() {
  if (not t3_isFake_isLoaded) {
    if (t3_isFake_branch != 0) {
      t3_isFake_branch->GetEntry(index);
    } else {
      printf("branch t3_isFake_branch does not exist!\n");
      exit(1);
    }
    t3_isFake_isLoaded = true;
  }
  return *t3_isFake_;
}
const vector<float> &SDL::t3_ptLegacy() {
  if (not t3_ptLegacy_isLoaded) {
    if (t3_ptLegacy_branch != 0) {
      t3_ptLegacy_branch->GetEntry(index);
    } else {
      printf("branch t3_ptLegacy_branch does not exist!\n");
      exit(1);
    }
    t3_ptLegacy_isLoaded = true;
  }
  return *t3_ptLegacy_;
}
const vector<float> &SDL::t3_pt() {
  if (not t3_pt_isLoaded) {
    if (t3_pt_branch != 0) {
      t3_pt_branch->GetEntry(index);
    } else {
      printf("branch t3_pt_branch does not exist!\n");
      exit(1);
    }
    t3_pt_isLoaded = true;
  }
  return *t3_pt_;
}
const vector<float> &SDL::t3_eta() {
  if (not t3_eta_isLoaded) {
    if (t3_eta_branch != 0) {
      t3_eta_branch->GetEntry(index);
    } else {
      printf("branch t3_eta_branch does not exist!\n");
      exit(1);
    }
    t3_eta_isLoaded = true;
  }
  return *t3_eta_;
}
const vector<float> &SDL::t3_phi() {
  if (not t3_phi_isLoaded) {
    if (t3_phi_branch != 0) {
      t3_phi_branch->GetEntry(index);
    } else {
      printf("branch t3_phi_branch does not exist!\n");
      exit(1);
    }
    t3_phi_isLoaded = true;
  }
  return *t3_phi_;
}
const vector<float> &SDL::t3_0_r() {
  if (not t3_0_r_isLoaded) {
    if (t3_0_r_branch != 0) {
      t3_0_r_branch->GetEntry(index);
    } else {
      printf("branch t3_0_r_branch does not exist!\n");
      exit(1);
    }
    t3_0_r_isLoaded = true;
  }
  return *t3_0_r_;
}
const vector<float> &SDL::t3_0_dr() {
  if (not t3_0_dr_isLoaded) {
    if (t3_0_dr_branch != 0) {
      t3_0_dr_branch->GetEntry(index);
    } else {
      printf("branch t3_0_dr_branch does not exist!\n");
      exit(1);
    }
    t3_0_dr_isLoaded = true;
  }
  return *t3_0_dr_;
}
const vector<float> &SDL::t3_0_x() {
  if (not t3_0_x_isLoaded) {
    if (t3_0_x_branch != 0) {
      t3_0_x_branch->GetEntry(index);
    } else {
      printf("branch t3_0_x_branch does not exist!\n");
      exit(1);
    }
    t3_0_x_isLoaded = true;
  }
  return *t3_0_x_;
}
const vector<float> &SDL::t3_0_y() {
  if (not t3_0_y_isLoaded) {
    if (t3_0_y_branch != 0) {
      t3_0_y_branch->GetEntry(index);
    } else {
      printf("branch t3_0_y_branch does not exist!\n");
      exit(1);
    }
    t3_0_y_isLoaded = true;
  }
  return *t3_0_y_;
}
const vector<float> &SDL::t3_0_z() {
  if (not t3_0_z_isLoaded) {
    if (t3_0_z_branch != 0) {
      t3_0_z_branch->GetEntry(index);
    } else {
      printf("branch t3_0_z_branch does not exist!\n");
      exit(1);
    }
    t3_0_z_isLoaded = true;
  }
  return *t3_0_z_;
}
const vector<float> &SDL::t3_2_r() {
  if (not t3_2_r_isLoaded) {
    if (t3_2_r_branch != 0) {
      t3_2_r_branch->GetEntry(index);
    } else {
      printf("branch t3_2_r_branch does not exist!\n");
      exit(1);
    }
    t3_2_r_isLoaded = true;
  }
  return *t3_2_r_;
}
const vector<float> &SDL::t3_2_dr() {
  if (not t3_2_dr_isLoaded) {
    if (t3_2_dr_branch != 0) {
      t3_2_dr_branch->GetEntry(index);
    } else {
      printf("branch t3_2_dr_branch does not exist!\n");
      exit(1);
    }
    t3_2_dr_isLoaded = true;
  }
  return *t3_2_dr_;
}
const vector<float> &SDL::t3_2_x() {
  if (not t3_2_x_isLoaded) {
    if (t3_2_x_branch != 0) {
      t3_2_x_branch->GetEntry(index);
    } else {
      printf("branch t3_2_x_branch does not exist!\n");
      exit(1);
    }
    t3_2_x_isLoaded = true;
  }
  return *t3_2_x_;
}
const vector<float> &SDL::t3_2_y() {
  if (not t3_2_y_isLoaded) {
    if (t3_2_y_branch != 0) {
      t3_2_y_branch->GetEntry(index);
    } else {
      printf("branch t3_2_y_branch does not exist!\n");
      exit(1);
    }
    t3_2_y_isLoaded = true;
  }
  return *t3_2_y_;
}
const vector<float> &SDL::t3_2_z() {
  if (not t3_2_z_isLoaded) {
    if (t3_2_z_branch != 0) {
      t3_2_z_branch->GetEntry(index);
    } else {
      printf("branch t3_2_z_branch does not exist!\n");
      exit(1);
    }
    t3_2_z_isLoaded = true;
  }
  return *t3_2_z_;
}
const vector<float> &SDL::t3_4_r() {
  if (not t3_4_r_isLoaded) {
    if (t3_4_r_branch != 0) {
      t3_4_r_branch->GetEntry(index);
    } else {
      printf("branch t3_4_r_branch does not exist!\n");
      exit(1);
    }
    t3_4_r_isLoaded = true;
  }
  return *t3_4_r_;
}
const vector<float> &SDL::t3_4_dr() {
  if (not t3_4_dr_isLoaded) {
    if (t3_4_dr_branch != 0) {
      t3_4_dr_branch->GetEntry(index);
    } else {
      printf("branch t3_4_dr_branch does not exist!\n");
      exit(1);
    }
    t3_4_dr_isLoaded = true;
  }
  return *t3_4_dr_;
}
const vector<float> &SDL::t3_4_x() {
  if (not t3_4_x_isLoaded) {
    if (t3_4_x_branch != 0) {
      t3_4_x_branch->GetEntry(index);
    } else {
      printf("branch t3_4_x_branch does not exist!\n");
      exit(1);
    }
    t3_4_x_isLoaded = true;
  }
  return *t3_4_x_;
}
const vector<float> &SDL::t3_4_y() {
  if (not t3_4_y_isLoaded) {
    if (t3_4_y_branch != 0) {
      t3_4_y_branch->GetEntry(index);
    } else {
      printf("branch t3_4_y_branch does not exist!\n");
      exit(1);
    }
    t3_4_y_isLoaded = true;
  }
  return *t3_4_y_;
}
const vector<float> &SDL::t3_4_z() {
  if (not t3_4_z_isLoaded) {
    if (t3_4_z_branch != 0) {
      t3_4_z_branch->GetEntry(index);
    } else {
      printf("branch t3_4_z_branch does not exist!\n");
      exit(1);
    }
    t3_4_z_isLoaded = true;
  }
  return *t3_4_z_;
}
const vector<vector<int> > &SDL::tc_lsIdx() {
  if (not tc_lsIdx_isLoaded) {
    if (tc_lsIdx_branch != 0) {
      tc_lsIdx_branch->GetEntry(index);
    } else {
      printf("branch tc_lsIdx_branch does not exist!\n");
      exit(1);
    }
    tc_lsIdx_isLoaded = true;
  }
  return *tc_lsIdx_;
}
void SDL::progress( int nEventsTotal, int nEventsChain ){
  int period = 1000;
  if(nEventsTotal%1000 == 0) {
    if (isatty(1)) {
      if( ( nEventsChain - nEventsTotal ) > period ){
        float frac = (float)nEventsTotal/(nEventsChain*0.01);
        printf("\015\033[32m ---> \033[1m\033[31m%4.1f%%"
               "\033[0m\033[32m <---\033[0m\015", frac);
        fflush(stdout);
      }
      else {
        printf("\015\033[32m ---> \033[1m\033[31m%4.1f%%"
               "\033[0m\033[32m <---\033[0m\015", 100.);
        cout << endl;
      }
    }
  }
}
namespace tas {
  const vector<float> &SimTrack_pt() { return sdl.SimTrack_pt(); }
  const vector<float> &SimTrack_eta() { return sdl.SimTrack_eta(); }
  const vector<float> &SimTrack_phi() { return sdl.SimTrack_phi(); }
  const vector<float> &SimTrack_dxy() { return sdl.SimTrack_dxy(); }
  const vector<float> &SimTrack_dz() { return sdl.SimTrack_dz(); }
  const vector<int> &SimTrack_charge() { return sdl.SimTrack_charge(); }
  const vector<int> &SimTrack_ppVtx() { return sdl.SimTrack_ppVtx(); }
  const vector<int> &SimTrack_pdgID() { return sdl.SimTrack_pdgID(); }
  const vector<float> &SimTrack_vx() { return sdl.SimTrack_vx(); }
  const vector<float> &SimTrack_vy() { return sdl.SimTrack_vy(); }
  const vector<float> &SimTrack_vz() { return sdl.SimTrack_vz(); }
  const vector<vector<int> > &SimTrack_TC_idx() { return sdl.SimTrack_TC_idx(); }
  const vector<int> &SimTrack_TC_typemask() { return sdl.SimTrack_TC_typemask(); }
  const vector<float> &TC_pt() { return sdl.TC_pt(); }
  const vector<float> &TC_eta() { return sdl.TC_eta(); }
  const vector<float> &TC_phi() { return sdl.TC_phi(); }
  const vector<float> &TC_dxy() { return sdl.TC_dxy(); }
  const vector<float> &TC_dz() { return sdl.TC_dz(); }
  const vector<int> &TC_charge() { return sdl.TC_charge(); }
  const vector<int> &TC_type() { return sdl.TC_type(); }
  const vector<int> &TC_isFake() { return sdl.TC_isFake(); }
  const vector<int> &TC_isDuplicate() { return sdl.TC_isDuplicate(); }
  const vector<vector<int> > &TC_SimTrack_idx() { return sdl.TC_SimTrack_idx(); }
  const vector<float> &sim_pt() { return sdl.sim_pt(); }
  const vector<float> &sim_eta() { return sdl.sim_eta(); }
  const vector<float> &sim_phi() { return sdl.sim_phi(); }
  const vector<float> &sim_pca_dxy() { return sdl.sim_pca_dxy(); }
  const vector<float> &sim_pca_dz() { return sdl.sim_pca_dz(); }
  const vector<int> &sim_q() { return sdl.sim_q(); }
  const vector<int> &sim_event() { return sdl.sim_event(); }
  const vector<int> &sim_pdgId() { return sdl.sim_pdgId(); }
  const vector<float> &sim_vx() { return sdl.sim_vx(); }
  const vector<float> &sim_vy() { return sdl.sim_vy(); }
  const vector<float> &sim_vz() { return sdl.sim_vz(); }
  const vector<float> &sim_trkNtupIdx() { return sdl.sim_trkNtupIdx(); }
  const vector<int> &sim_TC_matched() { return sdl.sim_TC_matched(); }
  const vector<int> &sim_TC_matched_mask() { return sdl.sim_TC_matched_mask(); }
  const vector<float> &tc_pt() { return sdl.tc_pt(); }
  const vector<float> &tc_eta() { return sdl.tc_eta(); }
  const vector<float> &tc_phi() { return sdl.tc_phi(); }
  const vector<int> &tc_type() { return sdl.tc_type(); }
  const vector<int> &tc_isFake() { return sdl.tc_isFake(); }
  const vector<int> &tc_isDuplicate() { return sdl.tc_isDuplicate(); }
  const vector<vector<int> > &tc_matched_simIdx() { return sdl.tc_matched_simIdx(); }
  const vector<float> &sim_dummy() { return sdl.sim_dummy(); }
  const vector<float> &tc_dummy() { return sdl.tc_dummy(); }
  const vector<vector<int> > &pT5_matched_simIdx() { return sdl.pT5_matched_simIdx(); }
  const vector<vector<int> > &pT5_hitIdxs() { return sdl.pT5_hitIdxs(); }
  const vector<int> &sim_pT5_matched() { return sdl.sim_pT5_matched(); }
  const vector<float> &pT5_pt() { return sdl.pT5_pt(); }
  const vector<float> &pT5_eta() { return sdl.pT5_eta(); }
  const vector<float> &pT5_phi() { return sdl.pT5_phi(); }
  const vector<int> &pT5_isFake() { return sdl.pT5_isFake(); }
  const vector<int> &pT5_isDuplicate() { return sdl.pT5_isDuplicate(); }
  const vector<int> &pT5_score() { return sdl.pT5_score(); }
  const vector<int> &pT5_layer_binary() { return sdl.pT5_layer_binary(); }
  const vector<int> &pT5_moduleType_binary() { return sdl.pT5_moduleType_binary(); }
  const vector<float> &pT5_matched_pt() { return sdl.pT5_matched_pt(); }
  const vector<float> &pT5_rzChiSquared() { return sdl.pT5_rzChiSquared(); }
  const vector<float> &pT5_rPhiChiSquared() { return sdl.pT5_rPhiChiSquared(); }
  const vector<float> &pT5_rPhiChiSquaredInwards() { return sdl.pT5_rPhiChiSquaredInwards(); }
  const vector<int> &sim_pT3_matched() { return sdl.sim_pT3_matched(); }
  const vector<float> &pT3_pt() { return sdl.pT3_pt(); }
  const vector<int> &pT3_isFake() { return sdl.pT3_isFake(); }
  const vector<int> &pT3_isDuplicate() { return sdl.pT3_isDuplicate(); }
  const vector<float> &pT3_eta() { return sdl.pT3_eta(); }
  const vector<float> &pT3_phi() { return sdl.pT3_phi(); }
  const vector<float> &pT3_score() { return sdl.pT3_score(); }
  const vector<int> &pT3_foundDuplicate() { return sdl.pT3_foundDuplicate(); }
  const vector<vector<int> > &pT3_matched_simIdx() { return sdl.pT3_matched_simIdx(); }
  const vector<vector<int> > &pT3_hitIdxs() { return sdl.pT3_hitIdxs(); }
  const vector<float> &pT3_pixelRadius() { return sdl.pT3_pixelRadius(); }
  const vector<float> &pT3_pixelRadiusError() { return sdl.pT3_pixelRadiusError(); }
  const vector<vector<float> > &pT3_matched_pt() { return sdl.pT3_matched_pt(); }
  const vector<float> &pT3_tripletRadius() { return sdl.pT3_tripletRadius(); }
  const vector<float> &pT3_rPhiChiSquared() { return sdl.pT3_rPhiChiSquared(); }
  const vector<float> &pT3_rPhiChiSquaredInwards() { return sdl.pT3_rPhiChiSquaredInwards(); }
  const vector<float> &pT3_rzChiSquared() { return sdl.pT3_rzChiSquared(); }
  const vector<int> &pT3_layer_binary() { return sdl.pT3_layer_binary(); }
  const vector<int> &pT3_moduleType_binary() { return sdl.pT3_moduleType_binary(); }
  const vector<int> &sim_pLS_matched() { return sdl.sim_pLS_matched(); }
  const vector<vector<int> > &sim_pLS_types() { return sdl.sim_pLS_types(); }
  const vector<int> &pLS_isFake() { return sdl.pLS_isFake(); }
  const vector<int> &pLS_isDuplicate() { return sdl.pLS_isDuplicate(); }
  const vector<float> &pLS_pt() { return sdl.pLS_pt(); }
  const vector<float> &pLS_eta() { return sdl.pLS_eta(); }
  const vector<float> &pLS_phi() { return sdl.pLS_phi(); }
  const vector<float> &pLS_score() { return sdl.pLS_score(); }
  const vector<int> &sim_T5_matched() { return sdl.sim_T5_matched(); }
  const vector<int> &t5_isFake() { return sdl.t5_isFake(); }
  const vector<int> &t5_isDuplicate() { return sdl.t5_isDuplicate(); }
  const vector<int> &t5_foundDuplicate() { return sdl.t5_foundDuplicate(); }
  const vector<float> &t5_pt() { return sdl.t5_pt(); }
  const vector<float> &t5_eta() { return sdl.t5_eta(); }
  const vector<float> &t5_phi() { return sdl.t5_phi(); }
  const vector<float> &t5_score_rphisum() { return sdl.t5_score_rphisum(); }
  const vector<vector<int> > &t5_hitIdxs() { return sdl.t5_hitIdxs(); }
  const vector<vector<int> > &t5_matched_simIdx() { return sdl.t5_matched_simIdx(); }
  const vector<int> &t5_moduleType_binary() { return sdl.t5_moduleType_binary(); }
  const vector<int> &t5_layer_binary() { return sdl.t5_layer_binary(); }
  const vector<float> &t5_matched_pt() { return sdl.t5_matched_pt(); }
  const vector<int> &t5_partOfTC() { return sdl.t5_partOfTC(); }
  const vector<float> &t5_innerRadius() { return sdl.t5_innerRadius(); }
  const vector<float> &t5_outerRadius() { return sdl.t5_outerRadius(); }
  const vector<float> &t5_bridgeRadius() { return sdl.t5_bridgeRadius(); }
  const vector<float> &t5_chiSquared() { return sdl.t5_chiSquared(); }
  const vector<float> &t5_rzChiSquared() { return sdl.t5_rzChiSquared(); }
  const vector<float> &t5_nonAnchorChiSquared() { return sdl.t5_nonAnchorChiSquared(); }
  const vector<float> &MD_pt() { return sdl.MD_pt(); }
  const vector<float> &MD_eta() { return sdl.MD_eta(); }
  const vector<float> &MD_phi() { return sdl.MD_phi(); }
  const vector<float> &MD_dphichange() { return sdl.MD_dphichange(); }
  const vector<int> &MD_isFake() { return sdl.MD_isFake(); }
  const vector<int> &MD_tpType() { return sdl.MD_tpType(); }
  const vector<int> &MD_detId() { return sdl.MD_detId(); }
  const vector<int> &MD_layer() { return sdl.MD_layer(); }
  const vector<float> &MD_0_r() { return sdl.MD_0_r(); }
  const vector<float> &MD_0_x() { return sdl.MD_0_x(); }
  const vector<float> &MD_0_y() { return sdl.MD_0_y(); }
  const vector<float> &MD_0_z() { return sdl.MD_0_z(); }
  const vector<float> &MD_1_r() { return sdl.MD_1_r(); }
  const vector<float> &MD_1_x() { return sdl.MD_1_x(); }
  const vector<float> &MD_1_y() { return sdl.MD_1_y(); }
  const vector<float> &MD_1_z() { return sdl.MD_1_z(); }
  const vector<float> &LS_pt() { return sdl.LS_pt(); }
  const vector<float> &LS_eta() { return sdl.LS_eta(); }
  const vector<float> &LS_phi() { return sdl.LS_phi(); }
  const vector<int> &LS_isFake() { return sdl.LS_isFake(); }
  const vector<int> &LS_MD_idx0() { return sdl.LS_MD_idx0(); }
  const vector<int> &LS_MD_idx1() { return sdl.LS_MD_idx1(); }
  const vector<float> &LS_sim_pt() { return sdl.LS_sim_pt(); }
  const vector<float> &LS_sim_eta() { return sdl.LS_sim_eta(); }
  const vector<float> &LS_sim_phi() { return sdl.LS_sim_phi(); }
  const vector<float> &LS_sim_pca_dxy() { return sdl.LS_sim_pca_dxy(); }
  const vector<float> &LS_sim_pca_dz() { return sdl.LS_sim_pca_dz(); }
  const vector<int> &LS_sim_q() { return sdl.LS_sim_q(); }
  const vector<int> &LS_sim_pdgId() { return sdl.LS_sim_pdgId(); }
  const vector<int> &LS_sim_event() { return sdl.LS_sim_event(); }
  const vector<int> &LS_sim_bx() { return sdl.LS_sim_bx(); }
  const vector<float> &LS_sim_vx() { return sdl.LS_sim_vx(); }
  const vector<float> &LS_sim_vy() { return sdl.LS_sim_vy(); }
  const vector<float> &LS_sim_vz() { return sdl.LS_sim_vz(); }
  const vector<int> &LS_isInTrueTC() { return sdl.LS_isInTrueTC(); }
  const vector<int> &t5_t3_idx0() { return sdl.t5_t3_idx0(); }
  const vector<int> &t5_t3_idx1() { return sdl.t5_t3_idx1(); }
  const vector<int> &t3_isFake() { return sdl.t3_isFake(); }
  const vector<float> &t3_ptLegacy() { return sdl.t3_ptLegacy(); }
  const vector<float> &t3_pt() { return sdl.t3_pt(); }
  const vector<float> &t3_eta() { return sdl.t3_eta(); }
  const vector<float> &t3_phi() { return sdl.t3_phi(); }
  const vector<float> &t3_0_r() { return sdl.t3_0_r(); }
  const vector<float> &t3_0_dr() { return sdl.t3_0_dr(); }
  const vector<float> &t3_0_x() { return sdl.t3_0_x(); }
  const vector<float> &t3_0_y() { return sdl.t3_0_y(); }
  const vector<float> &t3_0_z() { return sdl.t3_0_z(); }
  const vector<float> &t3_2_r() { return sdl.t3_2_r(); }
  const vector<float> &t3_2_dr() { return sdl.t3_2_dr(); }
  const vector<float> &t3_2_x() { return sdl.t3_2_x(); }
  const vector<float> &t3_2_y() { return sdl.t3_2_y(); }
  const vector<float> &t3_2_z() { return sdl.t3_2_z(); }
  const vector<float> &t3_4_r() { return sdl.t3_4_r(); }
  const vector<float> &t3_4_dr() { return sdl.t3_4_dr(); }
  const vector<float> &t3_4_x() { return sdl.t3_4_x(); }
  const vector<float> &t3_4_y() { return sdl.t3_4_y(); }
  const vector<float> &t3_4_z() { return sdl.t3_4_z(); }
  const vector<vector<int> > &tc_lsIdx() { return sdl.tc_lsIdx(); }
}
