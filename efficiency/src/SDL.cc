#include "SDL.h"
SDL sdl;

void SDL::Init(TTree *tree) {
  tree->SetMakeClass(1);
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
  sim_parentVtxIdx_branch = 0;
  if (tree->GetBranch("sim_parentVtxIdx") != 0) {
    sim_parentVtxIdx_branch = tree->GetBranch("sim_parentVtxIdx");
    if (sim_parentVtxIdx_branch) { sim_parentVtxIdx_branch->SetAddress(&sim_parentVtxIdx_); }
  }
  t3_isDuplicate_branch = 0;
  if (tree->GetBranch("t3_isDuplicate") != 0) {
    t3_isDuplicate_branch = tree->GetBranch("t3_isDuplicate");
    if (t3_isDuplicate_branch) { t3_isDuplicate_branch->SetAddress(&t3_isDuplicate_); }
  }
  pLS_phi_branch = 0;
  if (tree->GetBranch("pLS_phi") != 0) {
    pLS_phi_branch = tree->GetBranch("pLS_phi");
    if (pLS_phi_branch) { pLS_phi_branch->SetAddress(&pLS_phi_); }
  }
  sim_event_branch = 0;
  if (tree->GetBranch("sim_event") != 0) {
    sim_event_branch = tree->GetBranch("sim_event");
    if (sim_event_branch) { sim_event_branch->SetAddress(&sim_event_); }
  }
  sim_pT4_matched_branch = 0;
  if (tree->GetBranch("sim_pT4_matched") != 0) {
    sim_pT4_matched_branch = tree->GetBranch("sim_pT4_matched");
    if (sim_pT4_matched_branch) { sim_pT4_matched_branch->SetAddress(&sim_pT4_matched_); }
  }
  sim_q_branch = 0;
  if (tree->GetBranch("sim_q") != 0) {
    sim_q_branch = tree->GetBranch("sim_q");
    if (sim_q_branch) { sim_q_branch->SetAddress(&sim_q_); }
  }
  sim_eta_branch = 0;
  if (tree->GetBranch("sim_eta") != 0) {
    sim_eta_branch = tree->GetBranch("sim_eta");
    if (sim_eta_branch) { sim_eta_branch->SetAddress(&sim_eta_); }
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
  sim_T5_matched_branch = 0;
  if (tree->GetBranch("sim_T5_matched") != 0) {
    sim_T5_matched_branch = tree->GetBranch("sim_T5_matched");
    if (sim_T5_matched_branch) { sim_T5_matched_branch->SetAddress(&sim_T5_matched_); }
  }
  sim_T5_types_branch = 0;
  if (tree->GetBranch("sim_T5_types") != 0) {
    sim_T5_types_branch = tree->GetBranch("sim_T5_types");
    if (sim_T5_types_branch) { sim_T5_types_branch->SetAddress(&sim_T5_types_); }
  }
  t4_occupancies_branch = 0;
  if (tree->GetBranch("t4_occupancies") != 0) {
    t4_occupancies_branch = tree->GetBranch("t4_occupancies");
    if (t4_occupancies_branch) { t4_occupancies_branch->SetAddress(&t4_occupancies_); }
  }
  t3_occupancies_branch = 0;
  if (tree->GetBranch("t3_occupancies") != 0) {
    t3_occupancies_branch = tree->GetBranch("t3_occupancies");
    if (t3_occupancies_branch) { t3_occupancies_branch->SetAddress(&t3_occupancies_); }
  }
  t5_isDuplicate_branch = 0;
  if (tree->GetBranch("t5_isDuplicate") != 0) {
    t5_isDuplicate_branch = tree->GetBranch("t5_isDuplicate");
    if (t5_isDuplicate_branch) { t5_isDuplicate_branch->SetAddress(&t5_isDuplicate_); }
  }
  sim_pT4_types_branch = 0;
  if (tree->GetBranch("sim_pT4_types") != 0) {
    sim_pT4_types_branch = tree->GetBranch("sim_pT4_types");
    if (sim_pT4_types_branch) { sim_pT4_types_branch->SetAddress(&sim_pT4_types_); }
  }
  sim_pT3_matched_branch = 0;
  if (tree->GetBranch("sim_pT3_matched") != 0) {
    sim_pT3_matched_branch = tree->GetBranch("sim_pT3_matched");
    if (sim_pT3_matched_branch) { sim_pT3_matched_branch->SetAddress(&sim_pT3_matched_); }
  }
  sim_tcIdx_branch = 0;
  if (tree->GetBranch("sim_tcIdx") != 0) {
    sim_tcIdx_branch = tree->GetBranch("sim_tcIdx");
    if (sim_tcIdx_branch) { sim_tcIdx_branch->SetAddress(&sim_tcIdx_); }
  }
  pT3_isFake_branch = 0;
  if (tree->GetBranch("pT3_isFake") != 0) {
    pT3_isFake_branch = tree->GetBranch("pT3_isFake");
    if (pT3_isFake_branch) { pT3_isFake_branch->SetAddress(&pT3_isFake_); }
  }
  t4_isFake_branch = 0;
  if (tree->GetBranch("t4_isFake") != 0) {
    t4_isFake_branch = tree->GetBranch("t4_isFake");
    if (t4_isFake_branch) { t4_isFake_branch->SetAddress(&t4_isFake_); }
  }
  pT3_occupancies_branch = 0;
  if (tree->GetBranch("pT3_occupancies") != 0) {
    pT3_occupancies_branch = tree->GetBranch("pT3_occupancies");
    if (pT3_occupancies_branch) { pT3_occupancies_branch->SetAddress(&pT3_occupancies_); }
  }
  tc_occupancies_branch = 0;
  if (tree->GetBranch("tc_occupancies") != 0) {
    tc_occupancies_branch = tree->GetBranch("tc_occupancies");
    if (tc_occupancies_branch) { tc_occupancies_branch->SetAddress(&tc_occupancies_); }
  }
  simvtx_x_branch = 0;
  if (tree->GetBranch("simvtx_x") != 0) {
    simvtx_x_branch = tree->GetBranch("simvtx_x");
    if (simvtx_x_branch) { simvtx_x_branch->SetAddress(&simvtx_x_); }
  }
  simvtx_y_branch = 0;
  if (tree->GetBranch("simvtx_y") != 0) {
    simvtx_y_branch = tree->GetBranch("simvtx_y");
    if (simvtx_y_branch) { simvtx_y_branch->SetAddress(&simvtx_y_); }
  }
  simvtx_z_branch = 0;
  if (tree->GetBranch("simvtx_z") != 0) {
    simvtx_z_branch = tree->GetBranch("simvtx_z");
    if (simvtx_z_branch) { simvtx_z_branch->SetAddress(&simvtx_z_); }
  }
  sim_T4_matched_branch = 0;
  if (tree->GetBranch("sim_T4_matched") != 0) {
    sim_T4_matched_branch = tree->GetBranch("sim_T4_matched");
    if (sim_T4_matched_branch) { sim_T4_matched_branch->SetAddress(&sim_T4_matched_); }
  }
  sim_TC_matched_branch = 0;
  if (tree->GetBranch("sim_TC_matched") != 0) {
    sim_TC_matched_branch = tree->GetBranch("sim_TC_matched");
    if (sim_TC_matched_branch) { sim_TC_matched_branch->SetAddress(&sim_TC_matched_); }
  }
  pLS_isDuplicate_branch = 0;
  if (tree->GetBranch("pLS_isDuplicate") != 0) {
    pLS_isDuplicate_branch = tree->GetBranch("pLS_isDuplicate");
    if (pLS_isDuplicate_branch) { pLS_isDuplicate_branch->SetAddress(&pLS_isDuplicate_); }
  }
  module_subdets_branch = 0;
  if (tree->GetBranch("module_subdets") != 0) {
    module_subdets_branch = tree->GetBranch("module_subdets");
    if (module_subdets_branch) { module_subdets_branch->SetAddress(&module_subdets_); }
  }
  pT3_pt_branch = 0;
  if (tree->GetBranch("pT3_pt") != 0) {
    pT3_pt_branch = tree->GetBranch("pT3_pt");
    if (pT3_pt_branch) { pT3_pt_branch->SetAddress(&pT3_pt_); }
  }
  t5_occupancies_branch = 0;
  if (tree->GetBranch("t5_occupancies") != 0) {
    t5_occupancies_branch = tree->GetBranch("t5_occupancies");
    if (t5_occupancies_branch) { t5_occupancies_branch->SetAddress(&t5_occupancies_); }
  }
  tc_pt_branch = 0;
  if (tree->GetBranch("tc_pt") != 0) {
    tc_pt_branch = tree->GetBranch("tc_pt");
    if (tc_pt_branch) { tc_pt_branch->SetAddress(&tc_pt_); }
  }
  pT3_isDuplicate_branch = 0;
  if (tree->GetBranch("pT3_isDuplicate") != 0) {
    pT3_isDuplicate_branch = tree->GetBranch("pT3_isDuplicate");
    if (pT3_isDuplicate_branch) { pT3_isDuplicate_branch->SetAddress(&pT3_isDuplicate_); }
  }
  tc_isDuplicate_branch = 0;
  if (tree->GetBranch("tc_isDuplicate") != 0) {
    tc_isDuplicate_branch = tree->GetBranch("tc_isDuplicate");
    if (tc_isDuplicate_branch) { tc_isDuplicate_branch->SetAddress(&tc_isDuplicate_); }
  }
  pLS_pt_branch = 0;
  if (tree->GetBranch("pLS_pt") != 0) {
    pLS_pt_branch = tree->GetBranch("pLS_pt");
    if (pLS_pt_branch) { pLS_pt_branch->SetAddress(&pLS_pt_); }
  }
  sim_T4_types_branch = 0;
  if (tree->GetBranch("sim_T4_types") != 0) {
    sim_T4_types_branch = tree->GetBranch("sim_T4_types");
    if (sim_T4_types_branch) { sim_T4_types_branch->SetAddress(&sim_T4_types_); }
  }
  pT4_isDuplicate_branch = 0;
  if (tree->GetBranch("pT4_isDuplicate") != 0) {
    pT4_isDuplicate_branch = tree->GetBranch("pT4_isDuplicate");
    if (pT4_isDuplicate_branch) { pT4_isDuplicate_branch->SetAddress(&pT4_isDuplicate_); }
  }
  sim_phi_branch = 0;
  if (tree->GetBranch("sim_phi") != 0) {
    sim_phi_branch = tree->GetBranch("sim_phi");
    if (sim_phi_branch) { sim_phi_branch->SetAddress(&sim_phi_); }
  }
  t3_isFake_branch = 0;
  if (tree->GetBranch("t3_isFake") != 0) {
    t3_isFake_branch = tree->GetBranch("t3_isFake");
    if (t3_isFake_branch) { t3_isFake_branch->SetAddress(&t3_isFake_); }
  }
  md_occupancies_branch = 0;
  if (tree->GetBranch("md_occupancies") != 0) {
    md_occupancies_branch = tree->GetBranch("md_occupancies");
    if (md_occupancies_branch) { md_occupancies_branch->SetAddress(&md_occupancies_); }
  }
  t5_isFake_branch = 0;
  if (tree->GetBranch("t5_isFake") != 0) {
    t5_isFake_branch = tree->GetBranch("t5_isFake");
    if (t5_isFake_branch) { t5_isFake_branch->SetAddress(&t5_isFake_); }
  }
  sim_TC_types_branch = 0;
  if (tree->GetBranch("sim_TC_types") != 0) {
    sim_TC_types_branch = tree->GetBranch("sim_TC_types");
    if (sim_TC_types_branch) { sim_TC_types_branch->SetAddress(&sim_TC_types_); }
  }
  sg_occupancies_branch = 0;
  if (tree->GetBranch("sg_occupancies") != 0) {
    sg_occupancies_branch = tree->GetBranch("sg_occupancies");
    if (sg_occupancies_branch) { sg_occupancies_branch->SetAddress(&sg_occupancies_); }
  }
  sim_pca_dz_branch = 0;
  if (tree->GetBranch("sim_pca_dz") != 0) {
    sim_pca_dz_branch = tree->GetBranch("sim_pca_dz");
    if (sim_pca_dz_branch) { sim_pca_dz_branch->SetAddress(&sim_pca_dz_); }
  }
  t4_isDuplicate_branch = 0;
  if (tree->GetBranch("t4_isDuplicate") != 0) {
    t4_isDuplicate_branch = tree->GetBranch("t4_isDuplicate");
    if (t4_isDuplicate_branch) { t4_isDuplicate_branch->SetAddress(&t4_isDuplicate_); }
  }
  pT4_pt_branch = 0;
  if (tree->GetBranch("pT4_pt") != 0) {
    pT4_pt_branch = tree->GetBranch("pT4_pt");
    if (pT4_pt_branch) { pT4_pt_branch->SetAddress(&pT4_pt_); }
  }
  sim_pT3_types_branch = 0;
  if (tree->GetBranch("sim_pT3_types") != 0) {
    sim_pT3_types_branch = tree->GetBranch("sim_pT3_types");
    if (sim_pT3_types_branch) { sim_pT3_types_branch->SetAddress(&sim_pT3_types_); }
  }
  sim_pLS_types_branch = 0;
  if (tree->GetBranch("sim_pLS_types") != 0) {
    sim_pLS_types_branch = tree->GetBranch("sim_pLS_types");
    if (sim_pLS_types_branch) { sim_pLS_types_branch->SetAddress(&sim_pLS_types_); }
  }
  sim_pLS_matched_branch = 0;
  if (tree->GetBranch("sim_pLS_matched") != 0) {
    sim_pLS_matched_branch = tree->GetBranch("sim_pLS_matched");
    if (sim_pLS_matched_branch) { sim_pLS_matched_branch->SetAddress(&sim_pLS_matched_); }
  }
  pT4_isFake_branch = 0;
  if (tree->GetBranch("pT4_isFake") != 0) {
    pT4_isFake_branch = tree->GetBranch("pT4_isFake");
    if (pT4_isFake_branch) { pT4_isFake_branch->SetAddress(&pT4_isFake_); }
  }
  pT4_phi_branch = 0;
  if (tree->GetBranch("pT4_phi") != 0) {
    pT4_phi_branch = tree->GetBranch("pT4_phi");
    if (pT4_phi_branch) { pT4_phi_branch->SetAddress(&pT4_phi_); }
  }
  t4_phi_branch = 0;
  if (tree->GetBranch("t4_phi") != 0) {
    t4_phi_branch = tree->GetBranch("t4_phi");
    if (t4_phi_branch) { t4_phi_branch->SetAddress(&t4_phi_); }
  }
  t5_phi_branch = 0;
  if (tree->GetBranch("t5_phi") != 0) {
    t5_phi_branch = tree->GetBranch("t5_phi");
    if (t5_phi_branch) { t5_phi_branch->SetAddress(&t5_phi_); }
  }
  sim_T3_matched_branch = 0;
  if (tree->GetBranch("sim_T3_matched") != 0) {
    sim_T3_matched_branch = tree->GetBranch("sim_T3_matched");
    if (sim_T3_matched_branch) { sim_T3_matched_branch->SetAddress(&sim_T3_matched_); }
  }
  t5_pt_branch = 0;
  if (tree->GetBranch("t5_pt") != 0) {
    t5_pt_branch = tree->GetBranch("t5_pt");
    if (t5_pt_branch) { t5_pt_branch->SetAddress(&t5_pt_); }
  }
  pT3_phi_branch = 0;
  if (tree->GetBranch("pT3_phi") != 0) {
    pT3_phi_branch = tree->GetBranch("pT3_phi");
    if (pT3_phi_branch) { pT3_phi_branch->SetAddress(&pT3_phi_); }
  }
  t3_pt_branch = 0;
  if (tree->GetBranch("t3_pt") != 0) {
    t3_pt_branch = tree->GetBranch("t3_pt");
    if (t3_pt_branch) { t3_pt_branch->SetAddress(&t3_pt_); }
  }
  module_rings_branch = 0;
  if (tree->GetBranch("module_rings") != 0) {
    module_rings_branch = tree->GetBranch("module_rings");
    if (module_rings_branch) { module_rings_branch->SetAddress(&module_rings_); }
  }
  pT3_eta_branch = 0;
  if (tree->GetBranch("pT3_eta") != 0) {
    pT3_eta_branch = tree->GetBranch("pT3_eta");
    if (pT3_eta_branch) { pT3_eta_branch->SetAddress(&pT3_eta_); }
  }
  t4_eta_branch = 0;
  if (tree->GetBranch("t4_eta") != 0) {
    t4_eta_branch = tree->GetBranch("t4_eta");
    if (t4_eta_branch) { t4_eta_branch->SetAddress(&t4_eta_); }
  }
  sim_pt_branch = 0;
  if (tree->GetBranch("sim_pt") != 0) {
    sim_pt_branch = tree->GetBranch("sim_pt");
    if (sim_pt_branch) { sim_pt_branch->SetAddress(&sim_pt_); }
  }
  pLS_isFake_branch = 0;
  if (tree->GetBranch("pLS_isFake") != 0) {
    pLS_isFake_branch = tree->GetBranch("pLS_isFake");
    if (pLS_isFake_branch) { pLS_isFake_branch->SetAddress(&pLS_isFake_); }
  }
  pT3_tripletRadius_branch = 0;
  if (tree->GetBranch("pT3_tripletRadius") != 0) {
    pT3_tripletRadius_branch = tree->GetBranch("pT3_tripletRadius");
    if (pT3_tripletRadius_branch) { pT3_tripletRadius_branch->SetAddress(&pT3_tripletRadius_); }
  }
  sim_bunchCrossing_branch = 0;
  if (tree->GetBranch("sim_bunchCrossing") != 0) {
    sim_bunchCrossing_branch = tree->GetBranch("sim_bunchCrossing");
    if (sim_bunchCrossing_branch) { sim_bunchCrossing_branch->SetAddress(&sim_bunchCrossing_); }
  }
  tc_isFake_branch = 0;
  if (tree->GetBranch("tc_isFake") != 0) {
    tc_isFake_branch = tree->GetBranch("tc_isFake");
    if (tc_isFake_branch) { tc_isFake_branch->SetAddress(&tc_isFake_); }
  }
  sim_T3_types_branch = 0;
  if (tree->GetBranch("sim_T3_types") != 0) {
    sim_T3_types_branch = tree->GetBranch("sim_T3_types");
    if (sim_T3_types_branch) { sim_T3_types_branch->SetAddress(&sim_T3_types_); }
  }
  t5_eta_branch = 0;
  if (tree->GetBranch("t5_eta") != 0) {
    t5_eta_branch = tree->GetBranch("t5_eta");
    if (t5_eta_branch) { t5_eta_branch->SetAddress(&t5_eta_); }
  }
  t4_pt_branch = 0;
  if (tree->GetBranch("t4_pt") != 0) {
    t4_pt_branch = tree->GetBranch("t4_pt");
    if (t4_pt_branch) { t4_pt_branch->SetAddress(&t4_pt_); }
  }
  pLS_eta_branch = 0;
  if (tree->GetBranch("pLS_eta") != 0) {
    pLS_eta_branch = tree->GetBranch("pLS_eta");
    if (pLS_eta_branch) { pLS_eta_branch->SetAddress(&pLS_eta_); }
  }
  module_layers_branch = 0;
  if (tree->GetBranch("module_layers") != 0) {
    module_layers_branch = tree->GetBranch("module_layers");
    if (module_layers_branch) { module_layers_branch->SetAddress(&module_layers_); }
  }
  pT4_eta_branch = 0;
  if (tree->GetBranch("pT4_eta") != 0) {
    pT4_eta_branch = tree->GetBranch("pT4_eta");
    if (pT4_eta_branch) { pT4_eta_branch->SetAddress(&pT4_eta_); }
  }
  sim_pca_dxy_branch = 0;
  if (tree->GetBranch("sim_pca_dxy") != 0) {
    sim_pca_dxy_branch = tree->GetBranch("sim_pca_dxy");
    if (sim_pca_dxy_branch) { sim_pca_dxy_branch->SetAddress(&sim_pca_dxy_); }
  }
  sim_pdgId_branch = 0;
  if (tree->GetBranch("sim_pdgId") != 0) {
    sim_pdgId_branch = tree->GetBranch("sim_pdgId");
    if (sim_pdgId_branch) { sim_pdgId_branch->SetAddress(&sim_pdgId_); }
  }
  pT3_pixelRadius_branch = 0;
  if (tree->GetBranch("pT3_pixelRadius") != 0) {
    pT3_pixelRadius_branch = tree->GetBranch("pT3_pixelRadius");
    if (pT3_pixelRadius_branch) { pT3_pixelRadius_branch->SetAddress(&pT3_pixelRadius_); }
  }
  tree->SetMakeClass(0);
}
void SDL::GetEntry(unsigned int idx) {
  index = idx;
  t3_eta_isLoaded = false;
  t3_phi_isLoaded = false;
  sim_parentVtxIdx_isLoaded = false;
  t3_isDuplicate_isLoaded = false;
  pLS_phi_isLoaded = false;
  sim_event_isLoaded = false;
  sim_pT4_matched_isLoaded = false;
  sim_q_isLoaded = false;
  sim_eta_isLoaded = false;
  tc_eta_isLoaded = false;
  tc_phi_isLoaded = false;
  sim_T5_matched_isLoaded = false;
  sim_T5_types_isLoaded = false;
  t4_occupancies_isLoaded = false;
  t3_occupancies_isLoaded = false;
  t5_isDuplicate_isLoaded = false;
  sim_pT4_types_isLoaded = false;
  sim_pT3_matched_isLoaded = false;
  sim_tcIdx_isLoaded = false;
  pT3_isFake_isLoaded = false;
  t4_isFake_isLoaded = false;
  pT3_occupancies_isLoaded = false;
  tc_occupancies_isLoaded = false;
  simvtx_x_isLoaded = false;
  simvtx_y_isLoaded = false;
  simvtx_z_isLoaded = false;
  sim_T4_matched_isLoaded = false;
  sim_TC_matched_isLoaded = false;
  pLS_isDuplicate_isLoaded = false;
  module_subdets_isLoaded = false;
  pT3_pt_isLoaded = false;
  t5_occupancies_isLoaded = false;
  tc_pt_isLoaded = false;
  pT3_isDuplicate_isLoaded = false;
  tc_isDuplicate_isLoaded = false;
  pLS_pt_isLoaded = false;
  sim_T4_types_isLoaded = false;
  pT4_isDuplicate_isLoaded = false;
  sim_phi_isLoaded = false;
  t3_isFake_isLoaded = false;
  md_occupancies_isLoaded = false;
  t5_isFake_isLoaded = false;
  sim_TC_types_isLoaded = false;
  sg_occupancies_isLoaded = false;
  sim_pca_dz_isLoaded = false;
  t4_isDuplicate_isLoaded = false;
  pT4_pt_isLoaded = false;
  sim_pT3_types_isLoaded = false;
  sim_pLS_types_isLoaded = false;
  sim_pLS_matched_isLoaded = false;
  pT4_isFake_isLoaded = false;
  pT4_phi_isLoaded = false;
  t4_phi_isLoaded = false;
  t5_phi_isLoaded = false;
  sim_T3_matched_isLoaded = false;
  t5_pt_isLoaded = false;
  pT3_phi_isLoaded = false;
  t3_pt_isLoaded = false;
  module_rings_isLoaded = false;
  pT3_eta_isLoaded = false;
  t4_eta_isLoaded = false;
  sim_pt_isLoaded = false;
  pLS_isFake_isLoaded = false;
  pT3_tripletRadius_isLoaded = false;
  sim_bunchCrossing_isLoaded = false;
  tc_isFake_isLoaded = false;
  sim_T3_types_isLoaded = false;
  t5_eta_isLoaded = false;
  t4_pt_isLoaded = false;
  pLS_eta_isLoaded = false;
  module_layers_isLoaded = false;
  pT4_eta_isLoaded = false;
  sim_pca_dxy_isLoaded = false;
  sim_pdgId_isLoaded = false;
  pT3_pixelRadius_isLoaded = false;
}
void SDL::LoadAllBranches() {
  if (t3_eta_branch != 0) t3_eta();
  if (t3_phi_branch != 0) t3_phi();
  if (sim_parentVtxIdx_branch != 0) sim_parentVtxIdx();
  if (t3_isDuplicate_branch != 0) t3_isDuplicate();
  if (pLS_phi_branch != 0) pLS_phi();
  if (sim_event_branch != 0) sim_event();
  if (sim_pT4_matched_branch != 0) sim_pT4_matched();
  if (sim_q_branch != 0) sim_q();
  if (sim_eta_branch != 0) sim_eta();
  if (tc_eta_branch != 0) tc_eta();
  if (tc_phi_branch != 0) tc_phi();
  if (sim_T5_matched_branch != 0) sim_T5_matched();
  if (sim_T5_types_branch != 0) sim_T5_types();
  if (t4_occupancies_branch != 0) t4_occupancies();
  if (t3_occupancies_branch != 0) t3_occupancies();
  if (t5_isDuplicate_branch != 0) t5_isDuplicate();
  if (sim_pT4_types_branch != 0) sim_pT4_types();
  if (sim_pT3_matched_branch != 0) sim_pT3_matched();
  if (sim_tcIdx_branch != 0) sim_tcIdx();
  if (pT3_isFake_branch != 0) pT3_isFake();
  if (t4_isFake_branch != 0) t4_isFake();
  if (pT3_occupancies_branch != 0) pT3_occupancies();
  if (tc_occupancies_branch != 0) tc_occupancies();
  if (simvtx_x_branch != 0) simvtx_x();
  if (simvtx_y_branch != 0) simvtx_y();
  if (simvtx_z_branch != 0) simvtx_z();
  if (sim_T4_matched_branch != 0) sim_T4_matched();
  if (sim_TC_matched_branch != 0) sim_TC_matched();
  if (pLS_isDuplicate_branch != 0) pLS_isDuplicate();
  if (module_subdets_branch != 0) module_subdets();
  if (pT3_pt_branch != 0) pT3_pt();
  if (t5_occupancies_branch != 0) t5_occupancies();
  if (tc_pt_branch != 0) tc_pt();
  if (pT3_isDuplicate_branch != 0) pT3_isDuplicate();
  if (tc_isDuplicate_branch != 0) tc_isDuplicate();
  if (pLS_pt_branch != 0) pLS_pt();
  if (sim_T4_types_branch != 0) sim_T4_types();
  if (pT4_isDuplicate_branch != 0) pT4_isDuplicate();
  if (sim_phi_branch != 0) sim_phi();
  if (t3_isFake_branch != 0) t3_isFake();
  if (md_occupancies_branch != 0) md_occupancies();
  if (t5_isFake_branch != 0) t5_isFake();
  if (sim_TC_types_branch != 0) sim_TC_types();
  if (sg_occupancies_branch != 0) sg_occupancies();
  if (sim_pca_dz_branch != 0) sim_pca_dz();
  if (t4_isDuplicate_branch != 0) t4_isDuplicate();
  if (pT4_pt_branch != 0) pT4_pt();
  if (sim_pT3_types_branch != 0) sim_pT3_types();
  if (sim_pLS_types_branch != 0) sim_pLS_types();
  if (sim_pLS_matched_branch != 0) sim_pLS_matched();
  if (pT4_isFake_branch != 0) pT4_isFake();
  if (pT4_phi_branch != 0) pT4_phi();
  if (t4_phi_branch != 0) t4_phi();
  if (t5_phi_branch != 0) t5_phi();
  if (sim_T3_matched_branch != 0) sim_T3_matched();
  if (t5_pt_branch != 0) t5_pt();
  if (pT3_phi_branch != 0) pT3_phi();
  if (t3_pt_branch != 0) t3_pt();
  if (module_rings_branch != 0) module_rings();
  if (pT3_eta_branch != 0) pT3_eta();
  if (t4_eta_branch != 0) t4_eta();
  if (sim_pt_branch != 0) sim_pt();
  if (pLS_isFake_branch != 0) pLS_isFake();
  if (pT3_tripletRadius_branch != 0) pT3_tripletRadius();
  if (sim_bunchCrossing_branch != 0) sim_bunchCrossing();
  if (tc_isFake_branch != 0) tc_isFake();
  if (sim_T3_types_branch != 0) sim_T3_types();
  if (t5_eta_branch != 0) t5_eta();
  if (t4_pt_branch != 0) t4_pt();
  if (pLS_eta_branch != 0) pLS_eta();
  if (module_layers_branch != 0) module_layers();
  if (pT4_eta_branch != 0) pT4_eta();
  if (sim_pca_dxy_branch != 0) sim_pca_dxy();
  if (sim_pdgId_branch != 0) sim_pdgId();
  if (pT3_pixelRadius_branch != 0) pT3_pixelRadius();
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
const vector<int> &SDL::sim_parentVtxIdx() {
  if (not sim_parentVtxIdx_isLoaded) {
    if (sim_parentVtxIdx_branch != 0) {
      sim_parentVtxIdx_branch->GetEntry(index);
    } else {
      printf("branch sim_parentVtxIdx_branch does not exist!\n");
      exit(1);
    }
    sim_parentVtxIdx_isLoaded = true;
  }
  return *sim_parentVtxIdx_;
}
const vector<int> &SDL::t3_isDuplicate() {
  if (not t3_isDuplicate_isLoaded) {
    if (t3_isDuplicate_branch != 0) {
      t3_isDuplicate_branch->GetEntry(index);
    } else {
      printf("branch t3_isDuplicate_branch does not exist!\n");
      exit(1);
    }
    t3_isDuplicate_isLoaded = true;
  }
  return *t3_isDuplicate_;
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
const vector<int> &SDL::sim_pT4_matched() {
  if (not sim_pT4_matched_isLoaded) {
    if (sim_pT4_matched_branch != 0) {
      sim_pT4_matched_branch->GetEntry(index);
    } else {
      printf("branch sim_pT4_matched_branch does not exist!\n");
      exit(1);
    }
    sim_pT4_matched_isLoaded = true;
  }
  return *sim_pT4_matched_;
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
const vector<vector<int> > &SDL::sim_T5_types() {
  if (not sim_T5_types_isLoaded) {
    if (sim_T5_types_branch != 0) {
      sim_T5_types_branch->GetEntry(index);
    } else {
      printf("branch sim_T5_types_branch does not exist!\n");
      exit(1);
    }
    sim_T5_types_isLoaded = true;
  }
  return *sim_T5_types_;
}
const vector<int> &SDL::t4_occupancies() {
  if (not t4_occupancies_isLoaded) {
    if (t4_occupancies_branch != 0) {
      t4_occupancies_branch->GetEntry(index);
    } else {
      printf("branch t4_occupancies_branch does not exist!\n");
      exit(1);
    }
    t4_occupancies_isLoaded = true;
  }
  return *t4_occupancies_;
}
const vector<int> &SDL::t3_occupancies() {
  if (not t3_occupancies_isLoaded) {
    if (t3_occupancies_branch != 0) {
      t3_occupancies_branch->GetEntry(index);
    } else {
      printf("branch t3_occupancies_branch does not exist!\n");
      exit(1);
    }
    t3_occupancies_isLoaded = true;
  }
  return *t3_occupancies_;
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
const vector<vector<int> > &SDL::sim_pT4_types() {
  if (not sim_pT4_types_isLoaded) {
    if (sim_pT4_types_branch != 0) {
      sim_pT4_types_branch->GetEntry(index);
    } else {
      printf("branch sim_pT4_types_branch does not exist!\n");
      exit(1);
    }
    sim_pT4_types_isLoaded = true;
  }
  return *sim_pT4_types_;
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
const vector<vector<int> > &SDL::sim_tcIdx() {
  if (not sim_tcIdx_isLoaded) {
    if (sim_tcIdx_branch != 0) {
      sim_tcIdx_branch->GetEntry(index);
    } else {
      printf("branch sim_tcIdx_branch does not exist!\n");
      exit(1);
    }
    sim_tcIdx_isLoaded = true;
  }
  return *sim_tcIdx_;
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
const vector<int> &SDL::t4_isFake() {
  if (not t4_isFake_isLoaded) {
    if (t4_isFake_branch != 0) {
      t4_isFake_branch->GetEntry(index);
    } else {
      printf("branch t4_isFake_branch does not exist!\n");
      exit(1);
    }
    t4_isFake_isLoaded = true;
  }
  return *t4_isFake_;
}
const int &SDL::pT3_occupancies() {
  if (not pT3_occupancies_isLoaded) {
    if (pT3_occupancies_branch != 0) {
      pT3_occupancies_branch->GetEntry(index);
    } else {
      printf("branch pT3_occupancies_branch does not exist!\n");
      exit(1);
    }
    pT3_occupancies_isLoaded = true;
  }
  return pT3_occupancies_;
}
const vector<int> &SDL::tc_occupancies() {
  if (not tc_occupancies_isLoaded) {
    if (tc_occupancies_branch != 0) {
      tc_occupancies_branch->GetEntry(index);
    } else {
      printf("branch tc_occupancies_branch does not exist!\n");
      exit(1);
    }
    tc_occupancies_isLoaded = true;
  }
  return *tc_occupancies_;
}
const vector<float> &SDL::simvtx_x() {
  if (not simvtx_x_isLoaded) {
    if (simvtx_x_branch != 0) {
      simvtx_x_branch->GetEntry(index);
    } else {
      printf("branch simvtx_x_branch does not exist!\n");
      exit(1);
    }
    simvtx_x_isLoaded = true;
  }
  return *simvtx_x_;
}
const vector<float> &SDL::simvtx_y() {
  if (not simvtx_y_isLoaded) {
    if (simvtx_y_branch != 0) {
      simvtx_y_branch->GetEntry(index);
    } else {
      printf("branch simvtx_y_branch does not exist!\n");
      exit(1);
    }
    simvtx_y_isLoaded = true;
  }
  return *simvtx_y_;
}
const vector<float> &SDL::simvtx_z() {
  if (not simvtx_z_isLoaded) {
    if (simvtx_z_branch != 0) {
      simvtx_z_branch->GetEntry(index);
    } else {
      printf("branch simvtx_z_branch does not exist!\n");
      exit(1);
    }
    simvtx_z_isLoaded = true;
  }
  return *simvtx_z_;
}
const vector<int> &SDL::sim_T4_matched() {
  if (not sim_T4_matched_isLoaded) {
    if (sim_T4_matched_branch != 0) {
      sim_T4_matched_branch->GetEntry(index);
    } else {
      printf("branch sim_T4_matched_branch does not exist!\n");
      exit(1);
    }
    sim_T4_matched_isLoaded = true;
  }
  return *sim_T4_matched_;
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
const vector<int> &SDL::module_subdets() {
  if (not module_subdets_isLoaded) {
    if (module_subdets_branch != 0) {
      module_subdets_branch->GetEntry(index);
    } else {
      printf("branch module_subdets_branch does not exist!\n");
      exit(1);
    }
    module_subdets_isLoaded = true;
  }
  return *module_subdets_;
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
const vector<int> &SDL::t5_occupancies() {
  if (not t5_occupancies_isLoaded) {
    if (t5_occupancies_branch != 0) {
      t5_occupancies_branch->GetEntry(index);
    } else {
      printf("branch t5_occupancies_branch does not exist!\n");
      exit(1);
    }
    t5_occupancies_isLoaded = true;
  }
  return *t5_occupancies_;
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
const vector<vector<int> > &SDL::sim_T4_types() {
  if (not sim_T4_types_isLoaded) {
    if (sim_T4_types_branch != 0) {
      sim_T4_types_branch->GetEntry(index);
    } else {
      printf("branch sim_T4_types_branch does not exist!\n");
      exit(1);
    }
    sim_T4_types_isLoaded = true;
  }
  return *sim_T4_types_;
}
const vector<int> &SDL::pT4_isDuplicate() {
  if (not pT4_isDuplicate_isLoaded) {
    if (pT4_isDuplicate_branch != 0) {
      pT4_isDuplicate_branch->GetEntry(index);
    } else {
      printf("branch pT4_isDuplicate_branch does not exist!\n");
      exit(1);
    }
    pT4_isDuplicate_isLoaded = true;
  }
  return *pT4_isDuplicate_;
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
const vector<int> &SDL::md_occupancies() {
  if (not md_occupancies_isLoaded) {
    if (md_occupancies_branch != 0) {
      md_occupancies_branch->GetEntry(index);
    } else {
      printf("branch md_occupancies_branch does not exist!\n");
      exit(1);
    }
    md_occupancies_isLoaded = true;
  }
  return *md_occupancies_;
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
const vector<vector<int> > &SDL::sim_TC_types() {
  if (not sim_TC_types_isLoaded) {
    if (sim_TC_types_branch != 0) {
      sim_TC_types_branch->GetEntry(index);
    } else {
      printf("branch sim_TC_types_branch does not exist!\n");
      exit(1);
    }
    sim_TC_types_isLoaded = true;
  }
  return *sim_TC_types_;
}
const vector<int> &SDL::sg_occupancies() {
  if (not sg_occupancies_isLoaded) {
    if (sg_occupancies_branch != 0) {
      sg_occupancies_branch->GetEntry(index);
    } else {
      printf("branch sg_occupancies_branch does not exist!\n");
      exit(1);
    }
    sg_occupancies_isLoaded = true;
  }
  return *sg_occupancies_;
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
const vector<int> &SDL::t4_isDuplicate() {
  if (not t4_isDuplicate_isLoaded) {
    if (t4_isDuplicate_branch != 0) {
      t4_isDuplicate_branch->GetEntry(index);
    } else {
      printf("branch t4_isDuplicate_branch does not exist!\n");
      exit(1);
    }
    t4_isDuplicate_isLoaded = true;
  }
  return *t4_isDuplicate_;
}
const vector<float> &SDL::pT4_pt() {
  if (not pT4_pt_isLoaded) {
    if (pT4_pt_branch != 0) {
      pT4_pt_branch->GetEntry(index);
    } else {
      printf("branch pT4_pt_branch does not exist!\n");
      exit(1);
    }
    pT4_pt_isLoaded = true;
  }
  return *pT4_pt_;
}
const vector<vector<int> > &SDL::sim_pT3_types() {
  if (not sim_pT3_types_isLoaded) {
    if (sim_pT3_types_branch != 0) {
      sim_pT3_types_branch->GetEntry(index);
    } else {
      printf("branch sim_pT3_types_branch does not exist!\n");
      exit(1);
    }
    sim_pT3_types_isLoaded = true;
  }
  return *sim_pT3_types_;
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
const vector<int> &SDL::pT4_isFake() {
  if (not pT4_isFake_isLoaded) {
    if (pT4_isFake_branch != 0) {
      pT4_isFake_branch->GetEntry(index);
    } else {
      printf("branch pT4_isFake_branch does not exist!\n");
      exit(1);
    }
    pT4_isFake_isLoaded = true;
  }
  return *pT4_isFake_;
}
const vector<float> &SDL::pT4_phi() {
  if (not pT4_phi_isLoaded) {
    if (pT4_phi_branch != 0) {
      pT4_phi_branch->GetEntry(index);
    } else {
      printf("branch pT4_phi_branch does not exist!\n");
      exit(1);
    }
    pT4_phi_isLoaded = true;
  }
  return *pT4_phi_;
}
const vector<float> &SDL::t4_phi() {
  if (not t4_phi_isLoaded) {
    if (t4_phi_branch != 0) {
      t4_phi_branch->GetEntry(index);
    } else {
      printf("branch t4_phi_branch does not exist!\n");
      exit(1);
    }
    t4_phi_isLoaded = true;
  }
  return *t4_phi_;
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
const vector<int> &SDL::sim_T3_matched() {
  if (not sim_T3_matched_isLoaded) {
    if (sim_T3_matched_branch != 0) {
      sim_T3_matched_branch->GetEntry(index);
    } else {
      printf("branch sim_T3_matched_branch does not exist!\n");
      exit(1);
    }
    sim_T3_matched_isLoaded = true;
  }
  return *sim_T3_matched_;
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
const vector<int> &SDL::module_rings() {
  if (not module_rings_isLoaded) {
    if (module_rings_branch != 0) {
      module_rings_branch->GetEntry(index);
    } else {
      printf("branch module_rings_branch does not exist!\n");
      exit(1);
    }
    module_rings_isLoaded = true;
  }
  return *module_rings_;
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
const vector<float> &SDL::t4_eta() {
  if (not t4_eta_isLoaded) {
    if (t4_eta_branch != 0) {
      t4_eta_branch->GetEntry(index);
    } else {
      printf("branch t4_eta_branch does not exist!\n");
      exit(1);
    }
    t4_eta_isLoaded = true;
  }
  return *t4_eta_;
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
const vector<int> &SDL::sim_bunchCrossing() {
  if (not sim_bunchCrossing_isLoaded) {
    if (sim_bunchCrossing_branch != 0) {
      sim_bunchCrossing_branch->GetEntry(index);
    } else {
      printf("branch sim_bunchCrossing_branch does not exist!\n");
      exit(1);
    }
    sim_bunchCrossing_isLoaded = true;
  }
  return *sim_bunchCrossing_;
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
const vector<vector<int> > &SDL::sim_T3_types() {
  if (not sim_T3_types_isLoaded) {
    if (sim_T3_types_branch != 0) {
      sim_T3_types_branch->GetEntry(index);
    } else {
      printf("branch sim_T3_types_branch does not exist!\n");
      exit(1);
    }
    sim_T3_types_isLoaded = true;
  }
  return *sim_T3_types_;
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
const vector<float> &SDL::t4_pt() {
  if (not t4_pt_isLoaded) {
    if (t4_pt_branch != 0) {
      t4_pt_branch->GetEntry(index);
    } else {
      printf("branch t4_pt_branch does not exist!\n");
      exit(1);
    }
    t4_pt_isLoaded = true;
  }
  return *t4_pt_;
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
const vector<int> &SDL::module_layers() {
  if (not module_layers_isLoaded) {
    if (module_layers_branch != 0) {
      module_layers_branch->GetEntry(index);
    } else {
      printf("branch module_layers_branch does not exist!\n");
      exit(1);
    }
    module_layers_isLoaded = true;
  }
  return *module_layers_;
}
const vector<float> &SDL::pT4_eta() {
  if (not pT4_eta_isLoaded) {
    if (pT4_eta_branch != 0) {
      pT4_eta_branch->GetEntry(index);
    } else {
      printf("branch pT4_eta_branch does not exist!\n");
      exit(1);
    }
    pT4_eta_isLoaded = true;
  }
  return *pT4_eta_;
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
  const vector<float> &t3_eta() { return sdl.t3_eta(); }
  const vector<float> &t3_phi() { return sdl.t3_phi(); }
  const vector<int> &sim_parentVtxIdx() { return sdl.sim_parentVtxIdx(); }
  const vector<int> &t3_isDuplicate() { return sdl.t3_isDuplicate(); }
  const vector<float> &pLS_phi() { return sdl.pLS_phi(); }
  const vector<int> &sim_event() { return sdl.sim_event(); }
  const vector<int> &sim_pT4_matched() { return sdl.sim_pT4_matched(); }
  const vector<int> &sim_q() { return sdl.sim_q(); }
  const vector<float> &sim_eta() { return sdl.sim_eta(); }
  const vector<float> &tc_eta() { return sdl.tc_eta(); }
  const vector<float> &tc_phi() { return sdl.tc_phi(); }
  const vector<int> &sim_T5_matched() { return sdl.sim_T5_matched(); }
  const vector<vector<int> > &sim_T5_types() { return sdl.sim_T5_types(); }
  const vector<int> &t4_occupancies() { return sdl.t4_occupancies(); }
  const vector<int> &t3_occupancies() { return sdl.t3_occupancies(); }
  const vector<int> &t5_isDuplicate() { return sdl.t5_isDuplicate(); }
  const vector<vector<int> > &sim_pT4_types() { return sdl.sim_pT4_types(); }
  const vector<int> &sim_pT3_matched() { return sdl.sim_pT3_matched(); }
  const vector<vector<int> > &sim_tcIdx() { return sdl.sim_tcIdx(); }
  const vector<int> &pT3_isFake() { return sdl.pT3_isFake(); }
  const vector<int> &t4_isFake() { return sdl.t4_isFake(); }
  const int &pT3_occupancies() { return sdl.pT3_occupancies(); }
  const vector<int> &tc_occupancies() { return sdl.tc_occupancies(); }
  const vector<float> &simvtx_x() { return sdl.simvtx_x(); }
  const vector<float> &simvtx_y() { return sdl.simvtx_y(); }
  const vector<float> &simvtx_z() { return sdl.simvtx_z(); }
  const vector<int> &sim_T4_matched() { return sdl.sim_T4_matched(); }
  const vector<int> &sim_TC_matched() { return sdl.sim_TC_matched(); }
  const vector<int> &pLS_isDuplicate() { return sdl.pLS_isDuplicate(); }
  const vector<int> &module_subdets() { return sdl.module_subdets(); }
  const vector<float> &pT3_pt() { return sdl.pT3_pt(); }
  const vector<int> &t5_occupancies() { return sdl.t5_occupancies(); }
  const vector<float> &tc_pt() { return sdl.tc_pt(); }
  const vector<int> &pT3_isDuplicate() { return sdl.pT3_isDuplicate(); }
  const vector<int> &tc_isDuplicate() { return sdl.tc_isDuplicate(); }
  const vector<float> &pLS_pt() { return sdl.pLS_pt(); }
  const vector<vector<int> > &sim_T4_types() { return sdl.sim_T4_types(); }
  const vector<int> &pT4_isDuplicate() { return sdl.pT4_isDuplicate(); }
  const vector<float> &sim_phi() { return sdl.sim_phi(); }
  const vector<int> &t3_isFake() { return sdl.t3_isFake(); }
  const vector<int> &md_occupancies() { return sdl.md_occupancies(); }
  const vector<int> &t5_isFake() { return sdl.t5_isFake(); }
  const vector<vector<int> > &sim_TC_types() { return sdl.sim_TC_types(); }
  const vector<int> &sg_occupancies() { return sdl.sg_occupancies(); }
  const vector<float> &sim_pca_dz() { return sdl.sim_pca_dz(); }
  const vector<int> &t4_isDuplicate() { return sdl.t4_isDuplicate(); }
  const vector<float> &pT4_pt() { return sdl.pT4_pt(); }
  const vector<vector<int> > &sim_pT3_types() { return sdl.sim_pT3_types(); }
  const vector<vector<int> > &sim_pLS_types() { return sdl.sim_pLS_types(); }
  const vector<int> &sim_pLS_matched() { return sdl.sim_pLS_matched(); }
  const vector<int> &pT4_isFake() { return sdl.pT4_isFake(); }
  const vector<float> &pT4_phi() { return sdl.pT4_phi(); }
  const vector<float> &t4_phi() { return sdl.t4_phi(); }
  const vector<float> &t5_phi() { return sdl.t5_phi(); }
  const vector<int> &sim_T3_matched() { return sdl.sim_T3_matched(); }
  const vector<float> &t5_pt() { return sdl.t5_pt(); }
  const vector<float> &pT3_phi() { return sdl.pT3_phi(); }
  const vector<float> &t3_pt() { return sdl.t3_pt(); }
  const vector<int> &module_rings() { return sdl.module_rings(); }
  const vector<float> &pT3_eta() { return sdl.pT3_eta(); }
  const vector<float> &t4_eta() { return sdl.t4_eta(); }
  const vector<float> &sim_pt() { return sdl.sim_pt(); }
  const vector<int> &pLS_isFake() { return sdl.pLS_isFake(); }
  const vector<float> &pT3_tripletRadius() { return sdl.pT3_tripletRadius(); }
  const vector<int> &sim_bunchCrossing() { return sdl.sim_bunchCrossing(); }
  const vector<int> &tc_isFake() { return sdl.tc_isFake(); }
  const vector<vector<int> > &sim_T3_types() { return sdl.sim_T3_types(); }
  const vector<float> &t5_eta() { return sdl.t5_eta(); }
  const vector<float> &t4_pt() { return sdl.t4_pt(); }
  const vector<float> &pLS_eta() { return sdl.pLS_eta(); }
  const vector<int> &module_layers() { return sdl.module_layers(); }
  const vector<float> &pT4_eta() { return sdl.pT4_eta(); }
  const vector<float> &sim_pca_dxy() { return sdl.sim_pca_dxy(); }
  const vector<int> &sim_pdgId() { return sdl.sim_pdgId(); }
  const vector<float> &pT3_pixelRadius() { return sdl.pT3_pixelRadius(); }
}
