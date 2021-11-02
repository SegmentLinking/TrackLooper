#include "SDL.h"
SDL sdl;

void SDL::Init(TTree *tree) {
  tree->SetMakeClass(1);
  pT5_occupancies_branch = 0;
  if (tree->GetBranch("pT5_occupancies") != 0) {
    pT5_occupancies_branch = tree->GetBranch("pT5_occupancies");
    if (pT5_occupancies_branch) { pT5_occupancies_branch->SetAddress(&pT5_occupancies_); }
  }
  t3_phi_branch = 0;
  if (tree->GetBranch("t3_phi") != 0) {
    t3_phi_branch = tree->GetBranch("t3_phi");
    if (t3_phi_branch) { t3_phi_branch->SetAddress(&t3_phi_); }
  }
  t4_zLoPointed_branch = 0;
  if (tree->GetBranch("t4_zLoPointed") != 0) {
    t4_zLoPointed_branch = tree->GetBranch("t4_zLoPointed");
    if (t4_zLoPointed_branch) { t4_zLoPointed_branch->SetAddress(&t4_zLoPointed_); }
  }
  t4_kZ_branch = 0;
  if (tree->GetBranch("t4_kZ") != 0) {
    t4_kZ_branch = tree->GetBranch("t4_kZ");
    if (t4_kZ_branch) { t4_kZ_branch->SetAddress(&t4_kZ_); }
  }
  t3_isDuplicate_branch = 0;
  if (tree->GetBranch("t3_isDuplicate") != 0) {
    t3_isDuplicate_branch = tree->GetBranch("t3_isDuplicate");
    if (t3_isDuplicate_branch) { t3_isDuplicate_branch->SetAddress(&t3_isDuplicate_); }
  }
  sim_event_branch = 0;
  if (tree->GetBranch("sim_event") != 0) {
    sim_event_branch = tree->GetBranch("sim_event");
    if (sim_event_branch) { sim_event_branch->SetAddress(&sim_event_); }
  }
  t4_zOut_branch = 0;
  if (tree->GetBranch("t4_zOut") != 0) {
    t4_zOut_branch = tree->GetBranch("t4_zOut");
    if (t4_zOut_branch) { t4_zOut_branch->SetAddress(&t4_zOut_); }
  }
  sim_q_branch = 0;
  if (tree->GetBranch("sim_q") != 0) {
    sim_q_branch = tree->GetBranch("sim_q");
    if (sim_q_branch) { sim_q_branch->SetAddress(&sim_q_); }
  }
  t3_rtHi_branch = 0;
  if (tree->GetBranch("t3_rtHi") != 0) {
    t3_rtHi_branch = tree->GetBranch("t3_rtHi");
    if (t3_rtHi_branch) { t3_rtHi_branch->SetAddress(&t3_rtHi_); }
  }
  sim_eta_branch = 0;
  if (tree->GetBranch("sim_eta") != 0) {
    sim_eta_branch = tree->GetBranch("sim_eta");
    if (sim_eta_branch) { sim_eta_branch->SetAddress(&sim_eta_); }
  }
  pT4_betaOut_branch = 0;
  if (tree->GetBranch("pT4_betaOut") != 0) {
    pT4_betaOut_branch = tree->GetBranch("pT4_betaOut");
    if (pT4_betaOut_branch) { pT4_betaOut_branch->SetAddress(&pT4_betaOut_); }
  }
  pT4_zLo_branch = 0;
  if (tree->GetBranch("pT4_zLo") != 0) {
    pT4_zLo_branch = tree->GetBranch("pT4_zLo");
    if (pT4_zLo_branch) { pT4_zLo_branch->SetAddress(&pT4_zLo_); }
  }
  t5_eta_branch = 0;
  if (tree->GetBranch("t5_eta") != 0) {
    t5_eta_branch = tree->GetBranch("t5_eta");
    if (t5_eta_branch) { t5_eta_branch->SetAddress(&t5_eta_); }
  }
  sim_denom_branch = 0;
  if (tree->GetBranch("sim_denom") != 0) {
    sim_denom_branch = tree->GetBranch("sim_denom");
    if (sim_denom_branch) { sim_denom_branch->SetAddress(&sim_denom_); }
  }
  pLS_phi_branch = 0;
  if (tree->GetBranch("pLS_phi") != 0) {
    pLS_phi_branch = tree->GetBranch("pLS_phi");
    if (pLS_phi_branch) { pLS_phi_branch->SetAddress(&pLS_phi_); }
  }
  t3_layer3_branch = 0;
  if (tree->GetBranch("t3_layer3") != 0) {
    t3_layer3_branch = tree->GetBranch("t3_layer3");
    if (t3_layer3_branch) { t3_layer3_branch->SetAddress(&t3_layer3_); }
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
  t4_deltaPhiPos_branch = 0;
  if (tree->GetBranch("t4_deltaPhiPos") != 0) {
    t4_deltaPhiPos_branch = tree->GetBranch("t4_deltaPhiPos");
    if (t4_deltaPhiPos_branch) { t4_deltaPhiPos_branch->SetAddress(&t4_deltaPhiPos_); }
  }
  pT5_rzChiSquared_branch = 0;
  if (tree->GetBranch("pT5_rzChiSquared") != 0) {
    pT5_rzChiSquared_branch = tree->GetBranch("pT5_rzChiSquared");
    if (pT5_rzChiSquared_branch) { pT5_rzChiSquared_branch->SetAddress(&pT5_rzChiSquared_); }
  }
  pT5_eta_branch = 0;
  if (tree->GetBranch("pT5_eta") != 0) {
    pT5_eta_branch = tree->GetBranch("pT5_eta");
    if (pT5_eta_branch) { pT5_eta_branch->SetAddress(&pT5_eta_); }
  }
  sim_pT3_matched_branch = 0;
  if (tree->GetBranch("sim_pT3_matched") != 0) {
    sim_pT3_matched_branch = tree->GetBranch("sim_pT3_matched");
    if (sim_pT3_matched_branch) { sim_pT3_matched_branch->SetAddress(&sim_pT3_matched_); }
  }
  pT3_matched_pt_branch = 0;
  if (tree->GetBranch("pT3_matched_pt") != 0) {
    pT3_matched_pt_branch = tree->GetBranch("pT3_matched_pt");
    if (pT3_matched_pt_branch) { pT3_matched_pt_branch->SetAddress(&pT3_matched_pt_); }
  }
  t3_zHiPointed_branch = 0;
  if (tree->GetBranch("t3_zHiPointed") != 0) {
    t3_zHiPointed_branch = tree->GetBranch("t3_zHiPointed");
    if (t3_zHiPointed_branch) { t3_zHiPointed_branch->SetAddress(&t3_zHiPointed_); }
  }
  t4_betaOut_branch = 0;
  if (tree->GetBranch("t4_betaOut") != 0) {
    t4_betaOut_branch = tree->GetBranch("t4_betaOut");
    if (t4_betaOut_branch) { t4_betaOut_branch->SetAddress(&t4_betaOut_); }
  }
  t4_isDuplicate_branch = 0;
  if (tree->GetBranch("t4_isDuplicate") != 0) {
    t4_isDuplicate_branch = tree->GetBranch("t4_isDuplicate");
    if (t4_isDuplicate_branch) { t4_isDuplicate_branch->SetAddress(&t4_isDuplicate_); }
  }
  t4_betaOutCut_branch = 0;
  if (tree->GetBranch("t4_betaOutCut") != 0) {
    t4_betaOutCut_branch = tree->GetBranch("t4_betaOutCut");
    if (t4_betaOutCut_branch) { t4_betaOutCut_branch->SetAddress(&t4_betaOutCut_); }
  }
  t3_betaOut_branch = 0;
  if (tree->GetBranch("t3_betaOut") != 0) {
    t3_betaOut_branch = tree->GetBranch("t3_betaOut");
    if (t3_betaOut_branch) { t3_betaOut_branch->SetAddress(&t3_betaOut_); }
  }
  t3_sdlCut_branch = 0;
  if (tree->GetBranch("t3_sdlCut") != 0) {
    t3_sdlCut_branch = tree->GetBranch("t3_sdlCut");
    if (t3_sdlCut_branch) { t3_sdlCut_branch->SetAddress(&t3_sdlCut_); }
  }
  pT4_rtOut_branch = 0;
  if (tree->GetBranch("pT4_rtOut") != 0) {
    pT4_rtOut_branch = tree->GetBranch("pT4_rtOut");
    if (pT4_rtOut_branch) { pT4_rtOut_branch->SetAddress(&pT4_rtOut_); }
  }
  t3_betaInCut_branch = 0;
  if (tree->GetBranch("t3_betaInCut") != 0) {
    t3_betaInCut_branch = tree->GetBranch("t3_betaInCut");
    if (t3_betaInCut_branch) { t3_betaInCut_branch->SetAddress(&t3_betaInCut_); }
  }
  pT4_betaOutCut_branch = 0;
  if (tree->GetBranch("pT4_betaOutCut") != 0) {
    pT4_betaOutCut_branch = tree->GetBranch("pT4_betaOutCut");
    if (pT4_betaOutCut_branch) { pT4_betaOutCut_branch->SetAddress(&pT4_betaOutCut_); }
  }
  pT4_betaInCut_branch = 0;
  if (tree->GetBranch("pT4_betaInCut") != 0) {
    pT4_betaInCut_branch = tree->GetBranch("pT4_betaInCut");
    if (pT4_betaInCut_branch) { pT4_betaInCut_branch->SetAddress(&pT4_betaInCut_); }
  }
  pT3_pixelRadius_branch = 0;
  if (tree->GetBranch("pT3_pixelRadius") != 0) {
    pT3_pixelRadius_branch = tree->GetBranch("pT3_pixelRadius");
    if (pT3_pixelRadius_branch) { pT3_pixelRadius_branch->SetAddress(&pT3_pixelRadius_); }
  }
  sim_pt_branch = 0;
  if (tree->GetBranch("sim_pt") != 0) {
    sim_pt_branch = tree->GetBranch("sim_pt");
    if (sim_pt_branch) { sim_pt_branch->SetAddress(&sim_pt_); }
  }
  pT5_matched_pt_branch = 0;
  if (tree->GetBranch("pT5_matched_pt") != 0) {
    pT5_matched_pt_branch = tree->GetBranch("pT5_matched_pt");
    if (pT5_matched_pt_branch) { pT5_matched_pt_branch->SetAddress(&pT5_matched_pt_); }
  }
  pT4_deltaPhi_branch = 0;
  if (tree->GetBranch("pT4_deltaPhi") != 0) {
    pT4_deltaPhi_branch = tree->GetBranch("pT4_deltaPhi");
    if (pT4_deltaPhi_branch) { pT4_deltaPhi_branch->SetAddress(&pT4_deltaPhi_); }
  }
  t3_zLoPointed_branch = 0;
  if (tree->GetBranch("t3_zLoPointed") != 0) {
    t3_zLoPointed_branch = tree->GetBranch("t3_zLoPointed");
    if (t3_zLoPointed_branch) { t3_zLoPointed_branch->SetAddress(&t3_zLoPointed_); }
  }
  pLS_eta_branch = 0;
  if (tree->GetBranch("pLS_eta") != 0) {
    pLS_eta_branch = tree->GetBranch("pLS_eta");
    if (pLS_eta_branch) { pLS_eta_branch->SetAddress(&pLS_eta_); }
  }
  t3_deltaBetaCut_branch = 0;
  if (tree->GetBranch("t3_deltaBetaCut") != 0) {
    t3_deltaBetaCut_branch = tree->GetBranch("t3_deltaBetaCut");
    if (t3_deltaBetaCut_branch) { t3_deltaBetaCut_branch->SetAddress(&t3_deltaBetaCut_); }
  }
  t3_moduleType_binary_branch = 0;
  if (tree->GetBranch("t3_moduleType_binary") != 0) {
    t3_moduleType_binary_branch = tree->GetBranch("t3_moduleType_binary");
    if (t3_moduleType_binary_branch) { t3_moduleType_binary_branch->SetAddress(&t3_moduleType_binary_); }
  }
  sim_pdgId_branch = 0;
  if (tree->GetBranch("sim_pdgId") != 0) {
    sim_pdgId_branch = tree->GetBranch("sim_pdgId");
    if (sim_pdgId_branch) { sim_pdgId_branch->SetAddress(&sim_pdgId_); }
  }
  t3_eta_branch = 0;
  if (tree->GetBranch("t3_eta") != 0) {
    t3_eta_branch = tree->GetBranch("t3_eta");
    if (t3_eta_branch) { t3_eta_branch->SetAddress(&t3_eta_); }
  }
  t5_bridgeRadiusMax2S_branch = 0;
  if (tree->GetBranch("t5_bridgeRadiusMax2S") != 0) {
    t5_bridgeRadiusMax2S_branch = tree->GetBranch("t5_bridgeRadiusMax2S");
    if (t5_bridgeRadiusMax2S_branch) { t5_bridgeRadiusMax2S_branch->SetAddress(&t5_bridgeRadiusMax2S_); }
  }
  t5_outerRadiusMax2S_branch = 0;
  if (tree->GetBranch("t5_outerRadiusMax2S") != 0) {
    t5_outerRadiusMax2S_branch = tree->GetBranch("t5_outerRadiusMax2S");
    if (t5_outerRadiusMax2S_branch) { t5_outerRadiusMax2S_branch->SetAddress(&t5_outerRadiusMax2S_); }
  }
  t4_occupancies_branch = 0;
  if (tree->GetBranch("t4_occupancies") != 0) {
    t4_occupancies_branch = tree->GetBranch("t4_occupancies");
    if (t4_occupancies_branch) { t4_occupancies_branch->SetAddress(&t4_occupancies_); }
  }
  t5_layer_binary_branch = 0;
  if (tree->GetBranch("t5_layer_binary") != 0) {
    t5_layer_binary_branch = tree->GetBranch("t5_layer_binary");
    if (t5_layer_binary_branch) { t5_layer_binary_branch->SetAddress(&t5_layer_binary_); }
  }
  sim_tcIdx_branch = 0;
  if (tree->GetBranch("sim_tcIdx") != 0) {
    sim_tcIdx_branch = tree->GetBranch("sim_tcIdx");
    if (sim_tcIdx_branch) { sim_tcIdx_branch->SetAddress(&sim_tcIdx_); }
  }
  pT4_layer_binary_branch = 0;
  if (tree->GetBranch("pT4_layer_binary") != 0) {
    pT4_layer_binary_branch = tree->GetBranch("pT4_layer_binary");
    if (pT4_layer_binary_branch) { pT4_layer_binary_branch->SetAddress(&pT4_layer_binary_); }
  }
  pT3_layer_binary_branch = 0;
  if (tree->GetBranch("pT3_layer_binary") != 0) {
    pT3_layer_binary_branch = tree->GetBranch("pT3_layer_binary");
    if (pT3_layer_binary_branch) { pT3_layer_binary_branch->SetAddress(&pT3_layer_binary_); }
  }
  pT3_pix_idx3_branch = 0;
  if (tree->GetBranch("pT3_pix_idx3") != 0) {
    pT3_pix_idx3_branch = tree->GetBranch("pT3_pix_idx3");
    if (pT3_pix_idx3_branch) { pT3_pix_idx3_branch->SetAddress(&pT3_pix_idx3_); }
  }
  pT3_pix_idx2_branch = 0;
  if (tree->GetBranch("pT3_pix_idx2") != 0) {
    pT3_pix_idx2_branch = tree->GetBranch("pT3_pix_idx2");
    if (pT3_pix_idx2_branch) { pT3_pix_idx2_branch->SetAddress(&pT3_pix_idx2_); }
  }
  pT3_pix_idx1_branch = 0;
  if (tree->GetBranch("pT3_pix_idx1") != 0) {
    pT3_pix_idx1_branch = tree->GetBranch("pT3_pix_idx1");
    if (pT3_pix_idx1_branch) { pT3_pix_idx1_branch->SetAddress(&pT3_pix_idx1_); }
  }
  t5_bridgeRadiusMax_branch = 0;
  if (tree->GetBranch("t5_bridgeRadiusMax") != 0) {
    t5_bridgeRadiusMax_branch = tree->GetBranch("t5_bridgeRadiusMax");
    if (t5_bridgeRadiusMax_branch) { t5_bridgeRadiusMax_branch->SetAddress(&t5_bridgeRadiusMax_); }
  }
  t5_bridgeRadiusMin2S_branch = 0;
  if (tree->GetBranch("t5_bridgeRadiusMin2S") != 0) {
    t5_bridgeRadiusMin2S_branch = tree->GetBranch("t5_bridgeRadiusMin2S");
    if (t5_bridgeRadiusMin2S_branch) { t5_bridgeRadiusMin2S_branch->SetAddress(&t5_bridgeRadiusMin2S_); }
  }
  module_subdets_branch = 0;
  if (tree->GetBranch("module_subdets") != 0) {
    module_subdets_branch = tree->GetBranch("module_subdets");
    if (module_subdets_branch) { module_subdets_branch->SetAddress(&module_subdets_); }
  }
  pT3_tripletRadius_branch = 0;
  if (tree->GetBranch("pT3_tripletRadius") != 0) {
    pT3_tripletRadius_branch = tree->GetBranch("pT3_tripletRadius");
    if (pT3_tripletRadius_branch) { pT3_tripletRadius_branch->SetAddress(&pT3_tripletRadius_); }
  }
  pT4_zLoPointed_branch = 0;
  if (tree->GetBranch("pT4_zLoPointed") != 0) {
    pT4_zLoPointed_branch = tree->GetBranch("pT4_zLoPointed");
    if (pT4_zLoPointed_branch) { pT4_zLoPointed_branch->SetAddress(&pT4_zLoPointed_); }
  }
  t3_hit_idx4_branch = 0;
  if (tree->GetBranch("t3_hit_idx4") != 0) {
    t3_hit_idx4_branch = tree->GetBranch("t3_hit_idx4");
    if (t3_hit_idx4_branch) { t3_hit_idx4_branch->SetAddress(&t3_hit_idx4_); }
  }
  t3_hit_idx5_branch = 0;
  if (tree->GetBranch("t3_hit_idx5") != 0) {
    t3_hit_idx5_branch = tree->GetBranch("t3_hit_idx5");
    if (t3_hit_idx5_branch) { t3_hit_idx5_branch->SetAddress(&t3_hit_idx5_); }
  }
  t3_hit_idx6_branch = 0;
  if (tree->GetBranch("t3_hit_idx6") != 0) {
    t3_hit_idx6_branch = tree->GetBranch("t3_hit_idx6");
    if (t3_hit_idx6_branch) { t3_hit_idx6_branch->SetAddress(&t3_hit_idx6_); }
  }
  t3_rtOut_branch = 0;
  if (tree->GetBranch("t3_rtOut") != 0) {
    t3_rtOut_branch = tree->GetBranch("t3_rtOut");
    if (t3_rtOut_branch) { t3_rtOut_branch->SetAddress(&t3_rtOut_); }
  }
  t3_hit_idx1_branch = 0;
  if (tree->GetBranch("t3_hit_idx1") != 0) {
    t3_hit_idx1_branch = tree->GetBranch("t3_hit_idx1");
    if (t3_hit_idx1_branch) { t3_hit_idx1_branch->SetAddress(&t3_hit_idx1_); }
  }
  t3_hit_idx2_branch = 0;
  if (tree->GetBranch("t3_hit_idx2") != 0) {
    t3_hit_idx2_branch = tree->GetBranch("t3_hit_idx2");
    if (t3_hit_idx2_branch) { t3_hit_idx2_branch->SetAddress(&t3_hit_idx2_); }
  }
  t3_hit_idx3_branch = 0;
  if (tree->GetBranch("t3_hit_idx3") != 0) {
    t3_hit_idx3_branch = tree->GetBranch("t3_hit_idx3");
    if (t3_hit_idx3_branch) { t3_hit_idx3_branch->SetAddress(&t3_hit_idx3_); }
  }
  t3_isFake_branch = 0;
  if (tree->GetBranch("t3_isFake") != 0) {
    t3_isFake_branch = tree->GetBranch("t3_isFake");
    if (t3_isFake_branch) { t3_isFake_branch->SetAddress(&t3_isFake_); }
  }
  t5_isFake_branch = 0;
  if (tree->GetBranch("t5_isFake") != 0) {
    t5_isFake_branch = tree->GetBranch("t5_isFake");
    if (t5_isFake_branch) { t5_isFake_branch->SetAddress(&t5_isFake_); }
  }
  t5_bridgeRadiusMin_branch = 0;
  if (tree->GetBranch("t5_bridgeRadiusMin") != 0) {
    t5_bridgeRadiusMin_branch = tree->GetBranch("t5_bridgeRadiusMin");
    if (t5_bridgeRadiusMin_branch) { t5_bridgeRadiusMin_branch->SetAddress(&t5_bridgeRadiusMin_); }
  }
  t4_zLo_branch = 0;
  if (tree->GetBranch("t4_zLo") != 0) {
    t4_zLo_branch = tree->GetBranch("t4_zLo");
    if (t4_zLo_branch) { t4_zLo_branch->SetAddress(&t4_zLo_); }
  }
  md_occupancies_branch = 0;
  if (tree->GetBranch("md_occupancies") != 0) {
    md_occupancies_branch = tree->GetBranch("md_occupancies");
    if (md_occupancies_branch) { md_occupancies_branch->SetAddress(&md_occupancies_); }
  }
  t3_layer_binary_branch = 0;
  if (tree->GetBranch("t3_layer_binary") != 0) {
    t3_layer_binary_branch = tree->GetBranch("t3_layer_binary");
    if (t3_layer_binary_branch) { t3_layer_binary_branch->SetAddress(&t3_layer_binary_); }
  }
  t4_layer_binary_branch = 0;
  if (tree->GetBranch("t4_layer_binary") != 0) {
    t4_layer_binary_branch = tree->GetBranch("t4_layer_binary");
    if (t4_layer_binary_branch) { t4_layer_binary_branch->SetAddress(&t4_layer_binary_); }
  }
  sim_pT3_types_branch = 0;
  if (tree->GetBranch("sim_pT3_types") != 0) {
    sim_pT3_types_branch = tree->GetBranch("sim_pT3_types");
    if (sim_pT3_types_branch) { sim_pT3_types_branch->SetAddress(&sim_pT3_types_); }
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
  t4_isFake_branch = 0;
  if (tree->GetBranch("t4_isFake") != 0) {
    t4_isFake_branch = tree->GetBranch("t4_isFake");
    if (t4_isFake_branch) { t4_isFake_branch->SetAddress(&t4_isFake_); }
  }
  t4_deltaPhi_branch = 0;
  if (tree->GetBranch("t4_deltaPhi") != 0) {
    t4_deltaPhi_branch = tree->GetBranch("t4_deltaPhi");
    if (t4_deltaPhi_branch) { t4_deltaPhi_branch->SetAddress(&t4_deltaPhi_); }
  }
  t4_rtLo_branch = 0;
  if (tree->GetBranch("t4_rtLo") != 0) {
    t4_rtLo_branch = tree->GetBranch("t4_rtLo");
    if (t4_rtLo_branch) { t4_rtLo_branch->SetAddress(&t4_rtLo_); }
  }
  t5_outerRadius_branch = 0;
  if (tree->GetBranch("t5_outerRadius") != 0) {
    t5_outerRadius_branch = tree->GetBranch("t5_outerRadius");
    if (t5_outerRadius_branch) { t5_outerRadius_branch->SetAddress(&t5_outerRadius_); }
  }
  pT5_phi_branch = 0;
  if (tree->GetBranch("pT5_phi") != 0) {
    pT5_phi_branch = tree->GetBranch("pT5_phi");
    if (pT5_phi_branch) { pT5_phi_branch->SetAddress(&pT5_phi_); }
  }
  t4_betaIn_branch = 0;
  if (tree->GetBranch("t4_betaIn") != 0) {
    t4_betaIn_branch = tree->GetBranch("t4_betaIn");
    if (t4_betaIn_branch) { t4_betaIn_branch->SetAddress(&t4_betaIn_); }
  }
  tc_isFake_branch = 0;
  if (tree->GetBranch("tc_isFake") != 0) {
    tc_isFake_branch = tree->GetBranch("tc_isFake");
    if (tc_isFake_branch) { tc_isFake_branch->SetAddress(&tc_isFake_); }
  }
  t3_zOut_branch = 0;
  if (tree->GetBranch("t3_zOut") != 0) {
    t3_zOut_branch = tree->GetBranch("t3_zOut");
    if (t3_zOut_branch) { t3_zOut_branch->SetAddress(&t3_zOut_); }
  }
  t5_outerRadiusMax_branch = 0;
  if (tree->GetBranch("t5_outerRadiusMax") != 0) {
    t5_outerRadiusMax_branch = tree->GetBranch("t5_outerRadiusMax");
    if (t5_outerRadiusMax_branch) { t5_outerRadiusMax_branch->SetAddress(&t5_outerRadiusMax_); }
  }
  pT3_isFake_branch = 0;
  if (tree->GetBranch("pT3_isFake") != 0) {
    pT3_isFake_branch = tree->GetBranch("pT3_isFake");
    if (pT3_isFake_branch) { pT3_isFake_branch->SetAddress(&pT3_isFake_); }
  }
  sim_pLS_types_branch = 0;
  if (tree->GetBranch("sim_pLS_types") != 0) {
    sim_pLS_types_branch = tree->GetBranch("sim_pLS_types");
    if (sim_pLS_types_branch) { sim_pLS_types_branch->SetAddress(&sim_pLS_types_); }
  }
  t3_deltaBeta_branch = 0;
  if (tree->GetBranch("t3_deltaBeta") != 0) {
    t3_deltaBeta_branch = tree->GetBranch("t3_deltaBeta");
    if (t3_deltaBeta_branch) { t3_deltaBeta_branch->SetAddress(&t3_deltaBeta_); }
  }
  sim_pca_dxy_branch = 0;
  if (tree->GetBranch("sim_pca_dxy") != 0) {
    sim_pca_dxy_branch = tree->GetBranch("sim_pca_dxy");
    if (sim_pca_dxy_branch) { sim_pca_dxy_branch->SetAddress(&sim_pca_dxy_); }
  }
  t5_outerRadiusMin_branch = 0;
  if (tree->GetBranch("t5_outerRadiusMin") != 0) {
    t5_outerRadiusMin_branch = tree->GetBranch("t5_outerRadiusMin");
    if (t5_outerRadiusMin_branch) { t5_outerRadiusMin_branch->SetAddress(&t5_outerRadiusMin_); }
  }
  pT4_phi_branch = 0;
  if (tree->GetBranch("pT4_phi") != 0) {
    pT4_phi_branch = tree->GetBranch("pT4_phi");
    if (pT4_phi_branch) { pT4_phi_branch->SetAddress(&pT4_phi_); }
  }
  t3_rtLo_branch = 0;
  if (tree->GetBranch("t3_rtLo") != 0) {
    t3_rtLo_branch = tree->GetBranch("t3_rtLo");
    if (t3_rtLo_branch) { t3_rtLo_branch->SetAddress(&t3_rtLo_); }
  }
  t3_betaOutCut_branch = 0;
  if (tree->GetBranch("t3_betaOutCut") != 0) {
    t3_betaOutCut_branch = tree->GetBranch("t3_betaOutCut");
    if (t3_betaOutCut_branch) { t3_betaOutCut_branch->SetAddress(&t3_betaOutCut_); }
  }
  pT5_isDuplicate_branch = 0;
  if (tree->GetBranch("pT5_isDuplicate") != 0) {
    pT5_isDuplicate_branch = tree->GetBranch("pT5_isDuplicate");
    if (pT5_isDuplicate_branch) { pT5_isDuplicate_branch->SetAddress(&pT5_isDuplicate_); }
  }
  pT4_zHi_branch = 0;
  if (tree->GetBranch("pT4_zHi") != 0) {
    pT4_zHi_branch = tree->GetBranch("pT4_zHi");
    if (pT4_zHi_branch) { pT4_zHi_branch->SetAddress(&pT4_zHi_); }
  }
  t5_moduleType_binary_branch = 0;
  if (tree->GetBranch("t5_moduleType_binary") != 0) {
    t5_moduleType_binary_branch = tree->GetBranch("t5_moduleType_binary");
    if (t5_moduleType_binary_branch) { t5_moduleType_binary_branch->SetAddress(&t5_moduleType_binary_); }
  }
  t3_residual_branch = 0;
  if (tree->GetBranch("t3_residual") != 0) {
    t3_residual_branch = tree->GetBranch("t3_residual");
    if (t3_residual_branch) { t3_residual_branch->SetAddress(&t3_residual_); }
  }
  t3_occupancies_branch = 0;
  if (tree->GetBranch("t3_occupancies") != 0) {
    t3_occupancies_branch = tree->GetBranch("t3_occupancies");
    if (t3_occupancies_branch) { t3_occupancies_branch->SetAddress(&t3_occupancies_); }
  }
  sim_pT4_types_branch = 0;
  if (tree->GetBranch("sim_pT4_types") != 0) {
    sim_pT4_types_branch = tree->GetBranch("sim_pT4_types");
    if (sim_pT4_types_branch) { sim_pT4_types_branch->SetAddress(&sim_pT4_types_); }
  }
  t4_deltaBetaCut_branch = 0;
  if (tree->GetBranch("t4_deltaBetaCut") != 0) {
    t4_deltaBetaCut_branch = tree->GetBranch("t4_deltaBetaCut");
    if (t4_deltaBetaCut_branch) { t4_deltaBetaCut_branch->SetAddress(&t4_deltaBetaCut_); }
  }
  t5_pt_branch = 0;
  if (tree->GetBranch("t5_pt") != 0) {
    t5_pt_branch = tree->GetBranch("t5_pt");
    if (t5_pt_branch) { t5_pt_branch->SetAddress(&t5_pt_); }
  }
  sim_len_branch = 0;
  if (tree->GetBranch("sim_len") != 0) {
    sim_len_branch = tree->GetBranch("sim_len");
    if (sim_len_branch) { sim_len_branch->SetAddress(&sim_len_); }
  }
  sim_lengap_branch = 0;
  if (tree->GetBranch("sim_lengap") != 0) {
    sim_lengap_branch = tree->GetBranch("sim_lengap");
    if (sim_lengap_branch) { sim_lengap_branch->SetAddress(&sim_lengap_); }
  }
  sim_hits_branch = 0;
  if (tree->GetBranch("sim_hits") != 0) {
    sim_hits_branch = tree->GetBranch("sim_hits");
    if (sim_hits_branch) { sim_hits_branch->SetAddress(&sim_hits_); }
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
  t4_rtOut_branch = 0;
  if (tree->GetBranch("t4_rtOut") != 0) {
    t4_rtOut_branch = tree->GetBranch("t4_rtOut");
    if (t4_rtOut_branch) { t4_rtOut_branch->SetAddress(&t4_rtOut_); }
  }
  pT3_pt_branch = 0;
  if (tree->GetBranch("pT3_pt") != 0) {
    pT3_pt_branch = tree->GetBranch("pT3_pt");
    if (pT3_pt_branch) { pT3_pt_branch->SetAddress(&pT3_pt_); }
  }
  tc_pt_branch = 0;
  if (tree->GetBranch("tc_pt") != 0) {
    tc_pt_branch = tree->GetBranch("tc_pt");
    if (tc_pt_branch) { tc_pt_branch->SetAddress(&tc_pt_); }
  }
  pT3_pixelRadiusError_branch = 0;
  if (tree->GetBranch("pT3_pixelRadiusError") != 0) {
    pT3_pixelRadiusError_branch = tree->GetBranch("pT3_pixelRadiusError");
    if (pT3_pixelRadiusError_branch) { pT3_pixelRadiusError_branch->SetAddress(&pT3_pixelRadiusError_); }
  }
  pT5_isFake_branch = 0;
  if (tree->GetBranch("pT5_isFake") != 0) {
    pT5_isFake_branch = tree->GetBranch("pT5_isFake");
    if (pT5_isFake_branch) { pT5_isFake_branch->SetAddress(&pT5_isFake_); }
  }
  pT5_pt_branch = 0;
  if (tree->GetBranch("pT5_pt") != 0) {
    pT5_pt_branch = tree->GetBranch("pT5_pt");
    if (pT5_pt_branch) { pT5_pt_branch->SetAddress(&pT5_pt_); }
  }
  pT4_deltaBeta_branch = 0;
  if (tree->GetBranch("pT4_deltaBeta") != 0) {
    pT4_deltaBeta_branch = tree->GetBranch("pT4_deltaBeta");
    if (pT4_deltaBeta_branch) { pT4_deltaBeta_branch->SetAddress(&pT4_deltaBeta_); }
  }
  t5_innerRadiusMax_branch = 0;
  if (tree->GetBranch("t5_innerRadiusMax") != 0) {
    t5_innerRadiusMax_branch = tree->GetBranch("t5_innerRadiusMax");
    if (t5_innerRadiusMax_branch) { t5_innerRadiusMax_branch->SetAddress(&t5_innerRadiusMax_); }
  }
  sim_phi_branch = 0;
  if (tree->GetBranch("sim_phi") != 0) {
    sim_phi_branch = tree->GetBranch("sim_phi");
    if (sim_phi_branch) { sim_phi_branch->SetAddress(&sim_phi_); }
  }
  t4_betaInCut_branch = 0;
  if (tree->GetBranch("t4_betaInCut") != 0) {
    t4_betaInCut_branch = tree->GetBranch("t4_betaInCut");
    if (t4_betaInCut_branch) { t4_betaInCut_branch->SetAddress(&t4_betaInCut_); }
  }
  t5_innerRadiusMin_branch = 0;
  if (tree->GetBranch("t5_innerRadiusMin") != 0) {
    t5_innerRadiusMin_branch = tree->GetBranch("t5_innerRadiusMin");
    if (t5_innerRadiusMin_branch) { t5_innerRadiusMin_branch->SetAddress(&t5_innerRadiusMin_); }
  }
  pT4_sdlCut_branch = 0;
  if (tree->GetBranch("pT4_sdlCut") != 0) {
    pT4_sdlCut_branch = tree->GetBranch("pT4_sdlCut");
    if (pT4_sdlCut_branch) { pT4_sdlCut_branch->SetAddress(&pT4_sdlCut_); }
  }
  pT3_hit_idx3_branch = 0;
  if (tree->GetBranch("pT3_hit_idx3") != 0) {
    pT3_hit_idx3_branch = tree->GetBranch("pT3_hit_idx3");
    if (pT3_hit_idx3_branch) { pT3_hit_idx3_branch->SetAddress(&pT3_hit_idx3_); }
  }
  pT4_zHiPointed_branch = 0;
  if (tree->GetBranch("pT4_zHiPointed") != 0) {
    pT4_zHiPointed_branch = tree->GetBranch("pT4_zHiPointed");
    if (pT4_zHiPointed_branch) { pT4_zHiPointed_branch->SetAddress(&pT4_zHiPointed_); }
  }
  pT3_hit_idx1_branch = 0;
  if (tree->GetBranch("pT3_hit_idx1") != 0) {
    pT3_hit_idx1_branch = tree->GetBranch("pT3_hit_idx1");
    if (pT3_hit_idx1_branch) { pT3_hit_idx1_branch->SetAddress(&pT3_hit_idx1_); }
  }
  sim_pca_dz_branch = 0;
  if (tree->GetBranch("sim_pca_dz") != 0) {
    sim_pca_dz_branch = tree->GetBranch("sim_pca_dz");
    if (sim_pca_dz_branch) { sim_pca_dz_branch->SetAddress(&sim_pca_dz_); }
  }
  t4_deltaBeta_branch = 0;
  if (tree->GetBranch("t4_deltaBeta") != 0) {
    t4_deltaBeta_branch = tree->GetBranch("t4_deltaBeta");
    if (t4_deltaBeta_branch) { t4_deltaBeta_branch->SetAddress(&t4_deltaBeta_); }
  }
  pT3_hit_idx5_branch = 0;
  if (tree->GetBranch("pT3_hit_idx5") != 0) {
    pT3_hit_idx5_branch = tree->GetBranch("pT3_hit_idx5");
    if (pT3_hit_idx5_branch) { pT3_hit_idx5_branch->SetAddress(&pT3_hit_idx5_); }
  }
  pT3_hit_idx4_branch = 0;
  if (tree->GetBranch("pT3_hit_idx4") != 0) {
    pT3_hit_idx4_branch = tree->GetBranch("pT3_hit_idx4");
    if (pT3_hit_idx4_branch) { pT3_hit_idx4_branch->SetAddress(&pT3_hit_idx4_); }
  }
  pT5_layer_binary_branch = 0;
  if (tree->GetBranch("pT5_layer_binary") != 0) {
    pT5_layer_binary_branch = tree->GetBranch("pT5_layer_binary");
    if (pT5_layer_binary_branch) { pT5_layer_binary_branch->SetAddress(&pT5_layer_binary_); }
  }
  t5_bridgeRadius_branch = 0;
  if (tree->GetBranch("t5_bridgeRadius") != 0) {
    t5_bridgeRadius_branch = tree->GetBranch("t5_bridgeRadius");
    if (t5_bridgeRadius_branch) { t5_bridgeRadius_branch->SetAddress(&t5_bridgeRadius_); }
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
  sim_T3_matched_branch = 0;
  if (tree->GetBranch("sim_T3_matched") != 0) {
    sim_T3_matched_branch = tree->GetBranch("sim_T3_matched");
    if (sim_T3_matched_branch) { sim_T3_matched_branch->SetAddress(&sim_T3_matched_); }
  }
  t3_deltaPhiPos_branch = 0;
  if (tree->GetBranch("t3_deltaPhiPos") != 0) {
    t3_deltaPhiPos_branch = tree->GetBranch("t3_deltaPhiPos");
    if (t3_deltaPhiPos_branch) { t3_deltaPhiPos_branch->SetAddress(&t3_deltaPhiPos_); }
  }
  pT3_phi_branch = 0;
  if (tree->GetBranch("pT3_phi") != 0) {
    pT3_phi_branch = tree->GetBranch("pT3_phi");
    if (pT3_phi_branch) { pT3_phi_branch->SetAddress(&pT3_phi_); }
  }
  t5_matched_pt_branch = 0;
  if (tree->GetBranch("t5_matched_pt") != 0) {
    t5_matched_pt_branch = tree->GetBranch("t5_matched_pt");
    if (t5_matched_pt_branch) { t5_matched_pt_branch->SetAddress(&t5_matched_pt_); }
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
  t3_deltaPhi_branch = 0;
  if (tree->GetBranch("t3_deltaPhi") != 0) {
    t3_deltaPhi_branch = tree->GetBranch("t3_deltaPhi");
    if (t3_deltaPhi_branch) { t3_deltaPhi_branch->SetAddress(&t3_deltaPhi_); }
  }
  pLS_isFake_branch = 0;
  if (tree->GetBranch("pLS_isFake") != 0) {
    pLS_isFake_branch = tree->GetBranch("pLS_isFake");
    if (pLS_isFake_branch) { pLS_isFake_branch->SetAddress(&pLS_isFake_); }
  }
  pT4_betaIn_branch = 0;
  if (tree->GetBranch("pT4_betaIn") != 0) {
    pT4_betaIn_branch = tree->GetBranch("pT4_betaIn");
    if (pT4_betaIn_branch) { pT4_betaIn_branch->SetAddress(&pT4_betaIn_); }
  }
  sim_bunchCrossing_branch = 0;
  if (tree->GetBranch("sim_bunchCrossing") != 0) {
    sim_bunchCrossing_branch = tree->GetBranch("sim_bunchCrossing");
    if (sim_bunchCrossing_branch) { sim_bunchCrossing_branch->SetAddress(&sim_bunchCrossing_); }
  }
  pT4_zOut_branch = 0;
  if (tree->GetBranch("pT4_zOut") != 0) {
    pT4_zOut_branch = tree->GetBranch("pT4_zOut");
    if (pT4_zOut_branch) { pT4_zOut_branch->SetAddress(&pT4_zOut_); }
  }
  pT4_deltaPhiPos_branch = 0;
  if (tree->GetBranch("pT4_deltaPhiPos") != 0) {
    pT4_deltaPhiPos_branch = tree->GetBranch("pT4_deltaPhiPos");
    if (pT4_deltaPhiPos_branch) { pT4_deltaPhiPos_branch->SetAddress(&pT4_deltaPhiPos_); }
  }
  sim_parentVtxIdx_branch = 0;
  if (tree->GetBranch("sim_parentVtxIdx") != 0) {
    sim_parentVtxIdx_branch = tree->GetBranch("sim_parentVtxIdx");
    if (sim_parentVtxIdx_branch) { sim_parentVtxIdx_branch->SetAddress(&sim_parentVtxIdx_); }
  }
  t3_zHi_branch = 0;
  if (tree->GetBranch("t3_zHi") != 0) {
    t3_zHi_branch = tree->GetBranch("t3_zHi");
    if (t3_zHi_branch) { t3_zHi_branch->SetAddress(&t3_zHi_); }
  }
  sim_pT4_matched_branch = 0;
  if (tree->GetBranch("sim_pT4_matched") != 0) {
    sim_pT4_matched_branch = tree->GetBranch("sim_pT4_matched");
    if (sim_pT4_matched_branch) { sim_pT4_matched_branch->SetAddress(&sim_pT4_matched_); }
  }
  t5_innerRadiusMin2S_branch = 0;
  if (tree->GetBranch("t5_innerRadiusMin2S") != 0) {
    t5_innerRadiusMin2S_branch = tree->GetBranch("t5_innerRadiusMin2S");
    if (t5_innerRadiusMin2S_branch) { t5_innerRadiusMin2S_branch->SetAddress(&t5_innerRadiusMin2S_); }
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
  t5_isDuplicate_branch = 0;
  if (tree->GetBranch("t5_isDuplicate") != 0) {
    t5_isDuplicate_branch = tree->GetBranch("t5_isDuplicate");
    if (t5_isDuplicate_branch) { t5_isDuplicate_branch->SetAddress(&t5_isDuplicate_); }
  }
  t4_zHiPointed_branch = 0;
  if (tree->GetBranch("t4_zHiPointed") != 0) {
    t4_zHiPointed_branch = tree->GetBranch("t4_zHiPointed");
    if (t4_zHiPointed_branch) { t4_zHiPointed_branch->SetAddress(&t4_zHiPointed_); }
  }
  pT4_rtHi_branch = 0;
  if (tree->GetBranch("pT4_rtHi") != 0) {
    pT4_rtHi_branch = tree->GetBranch("pT4_rtHi");
    if (pT4_rtHi_branch) { pT4_rtHi_branch->SetAddress(&pT4_rtHi_); }
  }
  t5_outerRadiusMin2S_branch = 0;
  if (tree->GetBranch("t5_outerRadiusMin2S") != 0) {
    t5_outerRadiusMin2S_branch = tree->GetBranch("t5_outerRadiusMin2S");
    if (t5_outerRadiusMin2S_branch) { t5_outerRadiusMin2S_branch->SetAddress(&t5_outerRadiusMin2S_); }
  }
  t3_betaIn_branch = 0;
  if (tree->GetBranch("t3_betaIn") != 0) {
    t3_betaIn_branch = tree->GetBranch("t3_betaIn");
    if (t3_betaIn_branch) { t3_betaIn_branch->SetAddress(&t3_betaIn_); }
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
  t5_innerRadius_branch = 0;
  if (tree->GetBranch("t5_innerRadius") != 0) {
    t5_innerRadius_branch = tree->GetBranch("t5_innerRadius");
    if (t5_innerRadius_branch) { t5_innerRadius_branch->SetAddress(&t5_innerRadius_); }
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
  t5_occupancies_branch = 0;
  if (tree->GetBranch("t5_occupancies") != 0) {
    t5_occupancies_branch = tree->GetBranch("t5_occupancies");
    if (t5_occupancies_branch) { t5_occupancies_branch->SetAddress(&t5_occupancies_); }
  }
  t3_layer1_branch = 0;
  if (tree->GetBranch("t3_layer1") != 0) {
    t3_layer1_branch = tree->GetBranch("t3_layer1");
    if (t3_layer1_branch) { t3_layer1_branch->SetAddress(&t3_layer1_); }
  }
  pT4_kZ_branch = 0;
  if (tree->GetBranch("pT4_kZ") != 0) {
    pT4_kZ_branch = tree->GetBranch("pT4_kZ");
    if (pT4_kZ_branch) { pT4_kZ_branch->SetAddress(&pT4_kZ_); }
  }
  pT3_hit_idx2_branch = 0;
  if (tree->GetBranch("pT3_hit_idx2") != 0) {
    pT3_hit_idx2_branch = tree->GetBranch("pT3_hit_idx2");
    if (pT3_hit_idx2_branch) { pT3_hit_idx2_branch->SetAddress(&pT3_hit_idx2_); }
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
  t4_pt_branch = 0;
  if (tree->GetBranch("t4_pt") != 0) {
    t4_pt_branch = tree->GetBranch("t4_pt");
    if (t4_pt_branch) { t4_pt_branch->SetAddress(&t4_pt_); }
  }
  t4_zHi_branch = 0;
  if (tree->GetBranch("t4_zHi") != 0) {
    t4_zHi_branch = tree->GetBranch("t4_zHi");
    if (t4_zHi_branch) { t4_zHi_branch->SetAddress(&t4_zHi_); }
  }
  sim_TC_types_branch = 0;
  if (tree->GetBranch("sim_TC_types") != 0) {
    sim_TC_types_branch = tree->GetBranch("sim_TC_types");
    if (sim_TC_types_branch) { sim_TC_types_branch->SetAddress(&sim_TC_types_); }
  }
  t3_kZ_branch = 0;
  if (tree->GetBranch("t3_kZ") != 0) {
    t3_kZ_branch = tree->GetBranch("t3_kZ");
    if (t3_kZ_branch) { t3_kZ_branch->SetAddress(&t3_kZ_); }
  }
  t4_moduleType_binary_branch = 0;
  if (tree->GetBranch("t4_moduleType_binary") != 0) {
    t4_moduleType_binary_branch = tree->GetBranch("t4_moduleType_binary");
    if (t4_moduleType_binary_branch) { t4_moduleType_binary_branch->SetAddress(&t4_moduleType_binary_); }
  }
  sg_occupancies_branch = 0;
  if (tree->GetBranch("sg_occupancies") != 0) {
    sg_occupancies_branch = tree->GetBranch("sg_occupancies");
    if (sg_occupancies_branch) { sg_occupancies_branch->SetAddress(&sg_occupancies_); }
  }
  pT4_pt_branch = 0;
  if (tree->GetBranch("pT4_pt") != 0) {
    pT4_pt_branch = tree->GetBranch("pT4_pt");
    if (pT4_pt_branch) { pT4_pt_branch->SetAddress(&pT4_pt_); }
  }
  pT3_hit_idx6_branch = 0;
  if (tree->GetBranch("pT3_hit_idx6") != 0) {
    pT3_hit_idx6_branch = tree->GetBranch("pT3_hit_idx6");
    if (pT3_hit_idx6_branch) { pT3_hit_idx6_branch->SetAddress(&pT3_hit_idx6_); }
  }
  pT3_pix_idx4_branch = 0;
  if (tree->GetBranch("pT3_pix_idx4") != 0) {
    pT3_pix_idx4_branch = tree->GetBranch("pT3_pix_idx4");
    if (pT3_pix_idx4_branch) { pT3_pix_idx4_branch->SetAddress(&pT3_pix_idx4_); }
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
  t4_sdlCut_branch = 0;
  if (tree->GetBranch("t4_sdlCut") != 0) {
    t4_sdlCut_branch = tree->GetBranch("t4_sdlCut");
    if (t4_sdlCut_branch) { t4_sdlCut_branch->SetAddress(&t4_sdlCut_); }
  }
  pT4_rtLo_branch = 0;
  if (tree->GetBranch("pT4_rtLo") != 0) {
    pT4_rtLo_branch = tree->GetBranch("pT4_rtLo");
    if (pT4_rtLo_branch) { pT4_rtLo_branch->SetAddress(&pT4_rtLo_); }
  }
  t5_innerRadiusMax2S_branch = 0;
  if (tree->GetBranch("t5_innerRadiusMax2S") != 0) {
    t5_innerRadiusMax2S_branch = tree->GetBranch("t5_innerRadiusMax2S");
    if (t5_innerRadiusMax2S_branch) { t5_innerRadiusMax2S_branch->SetAddress(&t5_innerRadiusMax2S_); }
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
  t3_zLo_branch = 0;
  if (tree->GetBranch("t3_zLo") != 0) {
    t3_zLo_branch = tree->GetBranch("t3_zLo");
    if (t3_zLo_branch) { t3_zLo_branch->SetAddress(&t3_zLo_); }
  }
  pT4_deltaBetaCut_branch = 0;
  if (tree->GetBranch("pT4_deltaBetaCut") != 0) {
    pT4_deltaBetaCut_branch = tree->GetBranch("pT4_deltaBetaCut");
    if (pT4_deltaBetaCut_branch) { pT4_deltaBetaCut_branch->SetAddress(&pT4_deltaBetaCut_); }
  }
  t4_rtHi_branch = 0;
  if (tree->GetBranch("t4_rtHi") != 0) {
    t4_rtHi_branch = tree->GetBranch("t4_rtHi");
    if (t4_rtHi_branch) { t4_rtHi_branch->SetAddress(&t4_rtHi_); }
  }
  t3_layer2_branch = 0;
  if (tree->GetBranch("t3_layer2") != 0) {
    t3_layer2_branch = tree->GetBranch("t3_layer2");
    if (t3_layer2_branch) { t3_layer2_branch->SetAddress(&t3_layer2_); }
  }
  sim_T3_types_branch = 0;
  if (tree->GetBranch("sim_T3_types") != 0) {
    sim_T3_types_branch = tree->GetBranch("sim_T3_types");
    if (sim_T3_types_branch) { sim_T3_types_branch->SetAddress(&sim_T3_types_); }
  }
  sim_pT5_types_branch = 0;
  if (tree->GetBranch("sim_pT5_types") != 0) {
    sim_pT5_types_branch = tree->GetBranch("sim_pT5_types");
    if (sim_pT5_types_branch) { sim_pT5_types_branch->SetAddress(&sim_pT5_types_); }
  }
  sim_pT5_matched_branch = 0;
  if (tree->GetBranch("sim_pT5_matched") != 0) {
    sim_pT5_matched_branch = tree->GetBranch("sim_pT5_matched");
    if (sim_pT5_matched_branch) { sim_pT5_matched_branch->SetAddress(&sim_pT5_matched_); }
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
  tree->SetMakeClass(0);
}
void SDL::GetEntry(unsigned int idx) {
  index = idx;
  pT5_occupancies_isLoaded = false;
  t3_phi_isLoaded = false;
  t4_zLoPointed_isLoaded = false;
  t4_kZ_isLoaded = false;
  t3_isDuplicate_isLoaded = false;
  sim_event_isLoaded = false;
  t4_zOut_isLoaded = false;
  sim_q_isLoaded = false;
  t3_rtHi_isLoaded = false;
  sim_eta_isLoaded = false;
  pT4_betaOut_isLoaded = false;
  pT4_zLo_isLoaded = false;
  t5_eta_isLoaded = false;
  sim_denom_isLoaded = false;
  pLS_phi_isLoaded = false;
  t3_layer3_isLoaded = false;
  pT3_isDuplicate_isLoaded = false;
  tc_isDuplicate_isLoaded = false;
  t4_deltaPhiPos_isLoaded = false;
  pT5_rzChiSquared_isLoaded = false;
  pT5_eta_isLoaded = false;
  sim_pT3_matched_isLoaded = false;
  pT3_matched_pt_isLoaded = false;
  t3_zHiPointed_isLoaded = false;
  t4_betaOut_isLoaded = false;
  t4_isDuplicate_isLoaded = false;
  t4_betaOutCut_isLoaded = false;
  t3_betaOut_isLoaded = false;
  t3_sdlCut_isLoaded = false;
  pT4_rtOut_isLoaded = false;
  t3_betaInCut_isLoaded = false;
  pT4_betaOutCut_isLoaded = false;
  pT4_betaInCut_isLoaded = false;
  pT3_pixelRadius_isLoaded = false;
  sim_pt_isLoaded = false;
  pT5_matched_pt_isLoaded = false;
  pT4_deltaPhi_isLoaded = false;
  t3_zLoPointed_isLoaded = false;
  pLS_eta_isLoaded = false;
  t3_deltaBetaCut_isLoaded = false;
  t3_moduleType_binary_isLoaded = false;
  sim_pdgId_isLoaded = false;
  t3_eta_isLoaded = false;
  t5_bridgeRadiusMax2S_isLoaded = false;
  t5_outerRadiusMax2S_isLoaded = false;
  t4_occupancies_isLoaded = false;
  t5_layer_binary_isLoaded = false;
  sim_tcIdx_isLoaded = false;
  pT4_layer_binary_isLoaded = false;
  pT3_layer_binary_isLoaded = false;
  pT3_pix_idx3_isLoaded = false;
  pT3_pix_idx2_isLoaded = false;
  pT3_pix_idx1_isLoaded = false;
  t5_bridgeRadiusMax_isLoaded = false;
  t5_bridgeRadiusMin2S_isLoaded = false;
  module_subdets_isLoaded = false;
  pT3_tripletRadius_isLoaded = false;
  pT4_zLoPointed_isLoaded = false;
  t3_hit_idx4_isLoaded = false;
  t3_hit_idx5_isLoaded = false;
  t3_hit_idx6_isLoaded = false;
  t3_rtOut_isLoaded = false;
  t3_hit_idx1_isLoaded = false;
  t3_hit_idx2_isLoaded = false;
  t3_hit_idx3_isLoaded = false;
  t3_isFake_isLoaded = false;
  t5_isFake_isLoaded = false;
  t5_bridgeRadiusMin_isLoaded = false;
  t4_zLo_isLoaded = false;
  md_occupancies_isLoaded = false;
  t3_layer_binary_isLoaded = false;
  t4_layer_binary_isLoaded = false;
  sim_pT3_types_isLoaded = false;
  t4_phi_isLoaded = false;
  t5_phi_isLoaded = false;
  t4_isFake_isLoaded = false;
  t4_deltaPhi_isLoaded = false;
  t4_rtLo_isLoaded = false;
  t5_outerRadius_isLoaded = false;
  pT5_phi_isLoaded = false;
  t4_betaIn_isLoaded = false;
  tc_isFake_isLoaded = false;
  t3_zOut_isLoaded = false;
  t5_outerRadiusMax_isLoaded = false;
  pT3_isFake_isLoaded = false;
  sim_pLS_types_isLoaded = false;
  t3_deltaBeta_isLoaded = false;
  sim_pca_dxy_isLoaded = false;
  t5_outerRadiusMin_isLoaded = false;
  pT4_phi_isLoaded = false;
  t3_rtLo_isLoaded = false;
  t3_betaOutCut_isLoaded = false;
  pT5_isDuplicate_isLoaded = false;
  pT4_zHi_isLoaded = false;
  t5_moduleType_binary_isLoaded = false;
  t3_residual_isLoaded = false;
  t3_occupancies_isLoaded = false;
  sim_pT4_types_isLoaded = false;
  t4_deltaBetaCut_isLoaded = false;
  t5_pt_isLoaded = false;
  sim_len_isLoaded = false;
  sim_lengap_isLoaded = false;
  sim_hits_isLoaded = false;
  simvtx_x_isLoaded = false;
  simvtx_y_isLoaded = false;
  simvtx_z_isLoaded = false;
  sim_T4_matched_isLoaded = false;
  t4_rtOut_isLoaded = false;
  pT3_pt_isLoaded = false;
  tc_pt_isLoaded = false;
  pT3_pixelRadiusError_isLoaded = false;
  pT5_isFake_isLoaded = false;
  pT5_pt_isLoaded = false;
  pT4_deltaBeta_isLoaded = false;
  t5_innerRadiusMax_isLoaded = false;
  sim_phi_isLoaded = false;
  t4_betaInCut_isLoaded = false;
  t5_innerRadiusMin_isLoaded = false;
  pT4_sdlCut_isLoaded = false;
  pT3_hit_idx3_isLoaded = false;
  pT4_zHiPointed_isLoaded = false;
  pT3_hit_idx1_isLoaded = false;
  sim_pca_dz_isLoaded = false;
  t4_deltaBeta_isLoaded = false;
  pT3_hit_idx5_isLoaded = false;
  pT3_hit_idx4_isLoaded = false;
  pT5_layer_binary_isLoaded = false;
  t5_bridgeRadius_isLoaded = false;
  sim_pLS_matched_isLoaded = false;
  pT4_isFake_isLoaded = false;
  sim_T3_matched_isLoaded = false;
  t3_deltaPhiPos_isLoaded = false;
  pT3_phi_isLoaded = false;
  t5_matched_pt_isLoaded = false;
  pT3_eta_isLoaded = false;
  t4_eta_isLoaded = false;
  t3_deltaPhi_isLoaded = false;
  pLS_isFake_isLoaded = false;
  pT4_betaIn_isLoaded = false;
  sim_bunchCrossing_isLoaded = false;
  pT4_zOut_isLoaded = false;
  pT4_deltaPhiPos_isLoaded = false;
  sim_parentVtxIdx_isLoaded = false;
  t3_zHi_isLoaded = false;
  sim_pT4_matched_isLoaded = false;
  t5_innerRadiusMin2S_isLoaded = false;
  tc_eta_isLoaded = false;
  tc_phi_isLoaded = false;
  sim_T5_matched_isLoaded = false;
  sim_T5_types_isLoaded = false;
  t5_isDuplicate_isLoaded = false;
  t4_zHiPointed_isLoaded = false;
  pT4_rtHi_isLoaded = false;
  t5_outerRadiusMin2S_isLoaded = false;
  t3_betaIn_isLoaded = false;
  pT3_occupancies_isLoaded = false;
  tc_occupancies_isLoaded = false;
  t5_innerRadius_isLoaded = false;
  sim_TC_matched_isLoaded = false;
  pLS_isDuplicate_isLoaded = false;
  t5_occupancies_isLoaded = false;
  t3_layer1_isLoaded = false;
  pT4_kZ_isLoaded = false;
  pT3_hit_idx2_isLoaded = false;
  pLS_pt_isLoaded = false;
  sim_T4_types_isLoaded = false;
  pT4_isDuplicate_isLoaded = false;
  t4_pt_isLoaded = false;
  t4_zHi_isLoaded = false;
  sim_TC_types_isLoaded = false;
  t3_kZ_isLoaded = false;
  t4_moduleType_binary_isLoaded = false;
  sg_occupancies_isLoaded = false;
  pT4_pt_isLoaded = false;
  pT3_hit_idx6_isLoaded = false;
  pT3_pix_idx4_isLoaded = false;
  sim_vx_isLoaded = false;
  sim_vy_isLoaded = false;
  sim_vz_isLoaded = false;
  t4_sdlCut_isLoaded = false;
  pT4_rtLo_isLoaded = false;
  t5_innerRadiusMax2S_isLoaded = false;
  t3_pt_isLoaded = false;
  module_rings_isLoaded = false;
  t3_zLo_isLoaded = false;
  pT4_deltaBetaCut_isLoaded = false;
  t4_rtHi_isLoaded = false;
  t3_layer2_isLoaded = false;
  sim_T3_types_isLoaded = false;
  sim_pT5_types_isLoaded = false;
  sim_pT5_matched_isLoaded = false;
  module_layers_isLoaded = false;
  pT4_eta_isLoaded = false;
}
void SDL::LoadAllBranches() {
  if (pT5_occupancies_branch != 0) pT5_occupancies();
  if (t3_phi_branch != 0) t3_phi();
  if (t4_zLoPointed_branch != 0) t4_zLoPointed();
  if (t4_kZ_branch != 0) t4_kZ();
  if (t3_isDuplicate_branch != 0) t3_isDuplicate();
  if (sim_event_branch != 0) sim_event();
  if (t4_zOut_branch != 0) t4_zOut();
  if (sim_q_branch != 0) sim_q();
  if (t3_rtHi_branch != 0) t3_rtHi();
  if (sim_eta_branch != 0) sim_eta();
  if (pT4_betaOut_branch != 0) pT4_betaOut();
  if (pT4_zLo_branch != 0) pT4_zLo();
  if (t5_eta_branch != 0) t5_eta();
  if (sim_denom_branch != 0) sim_denom();
  if (pLS_phi_branch != 0) pLS_phi();
  if (t3_layer3_branch != 0) t3_layer3();
  if (pT3_isDuplicate_branch != 0) pT3_isDuplicate();
  if (tc_isDuplicate_branch != 0) tc_isDuplicate();
  if (t4_deltaPhiPos_branch != 0) t4_deltaPhiPos();
  if (pT5_rzChiSquared_branch != 0) pT5_rzChiSquared();
  if (pT5_eta_branch != 0) pT5_eta();
  if (sim_pT3_matched_branch != 0) sim_pT3_matched();
  if (pT3_matched_pt_branch != 0) pT3_matched_pt();
  if (t3_zHiPointed_branch != 0) t3_zHiPointed();
  if (t4_betaOut_branch != 0) t4_betaOut();
  if (t4_isDuplicate_branch != 0) t4_isDuplicate();
  if (t4_betaOutCut_branch != 0) t4_betaOutCut();
  if (t3_betaOut_branch != 0) t3_betaOut();
  if (t3_sdlCut_branch != 0) t3_sdlCut();
  if (pT4_rtOut_branch != 0) pT4_rtOut();
  if (t3_betaInCut_branch != 0) t3_betaInCut();
  if (pT4_betaOutCut_branch != 0) pT4_betaOutCut();
  if (pT4_betaInCut_branch != 0) pT4_betaInCut();
  if (pT3_pixelRadius_branch != 0) pT3_pixelRadius();
  if (sim_pt_branch != 0) sim_pt();
  if (pT5_matched_pt_branch != 0) pT5_matched_pt();
  if (pT4_deltaPhi_branch != 0) pT4_deltaPhi();
  if (t3_zLoPointed_branch != 0) t3_zLoPointed();
  if (pLS_eta_branch != 0) pLS_eta();
  if (t3_deltaBetaCut_branch != 0) t3_deltaBetaCut();
  if (t3_moduleType_binary_branch != 0) t3_moduleType_binary();
  if (sim_pdgId_branch != 0) sim_pdgId();
  if (t3_eta_branch != 0) t3_eta();
  if (t5_bridgeRadiusMax2S_branch != 0) t5_bridgeRadiusMax2S();
  if (t5_outerRadiusMax2S_branch != 0) t5_outerRadiusMax2S();
  if (t4_occupancies_branch != 0) t4_occupancies();
  if (t5_layer_binary_branch != 0) t5_layer_binary();
  if (sim_tcIdx_branch != 0) sim_tcIdx();
  if (pT4_layer_binary_branch != 0) pT4_layer_binary();
  if (pT3_layer_binary_branch != 0) pT3_layer_binary();
  if (pT3_pix_idx3_branch != 0) pT3_pix_idx3();
  if (pT3_pix_idx2_branch != 0) pT3_pix_idx2();
  if (pT3_pix_idx1_branch != 0) pT3_pix_idx1();
  if (t5_bridgeRadiusMax_branch != 0) t5_bridgeRadiusMax();
  if (t5_bridgeRadiusMin2S_branch != 0) t5_bridgeRadiusMin2S();
  if (module_subdets_branch != 0) module_subdets();
  if (pT3_tripletRadius_branch != 0) pT3_tripletRadius();
  if (pT4_zLoPointed_branch != 0) pT4_zLoPointed();
  if (t3_hit_idx4_branch != 0) t3_hit_idx4();
  if (t3_hit_idx5_branch != 0) t3_hit_idx5();
  if (t3_hit_idx6_branch != 0) t3_hit_idx6();
  if (t3_rtOut_branch != 0) t3_rtOut();
  if (t3_hit_idx1_branch != 0) t3_hit_idx1();
  if (t3_hit_idx2_branch != 0) t3_hit_idx2();
  if (t3_hit_idx3_branch != 0) t3_hit_idx3();
  if (t3_isFake_branch != 0) t3_isFake();
  if (t5_isFake_branch != 0) t5_isFake();
  if (t5_bridgeRadiusMin_branch != 0) t5_bridgeRadiusMin();
  if (t4_zLo_branch != 0) t4_zLo();
  if (md_occupancies_branch != 0) md_occupancies();
  if (t3_layer_binary_branch != 0) t3_layer_binary();
  if (t4_layer_binary_branch != 0) t4_layer_binary();
  if (sim_pT3_types_branch != 0) sim_pT3_types();
  if (t4_phi_branch != 0) t4_phi();
  if (t5_phi_branch != 0) t5_phi();
  if (t4_isFake_branch != 0) t4_isFake();
  if (t4_deltaPhi_branch != 0) t4_deltaPhi();
  if (t4_rtLo_branch != 0) t4_rtLo();
  if (t5_outerRadius_branch != 0) t5_outerRadius();
  if (pT5_phi_branch != 0) pT5_phi();
  if (t4_betaIn_branch != 0) t4_betaIn();
  if (tc_isFake_branch != 0) tc_isFake();
  if (t3_zOut_branch != 0) t3_zOut();
  if (t5_outerRadiusMax_branch != 0) t5_outerRadiusMax();
  if (pT3_isFake_branch != 0) pT3_isFake();
  if (sim_pLS_types_branch != 0) sim_pLS_types();
  if (t3_deltaBeta_branch != 0) t3_deltaBeta();
  if (sim_pca_dxy_branch != 0) sim_pca_dxy();
  if (t5_outerRadiusMin_branch != 0) t5_outerRadiusMin();
  if (pT4_phi_branch != 0) pT4_phi();
  if (t3_rtLo_branch != 0) t3_rtLo();
  if (t3_betaOutCut_branch != 0) t3_betaOutCut();
  if (pT5_isDuplicate_branch != 0) pT5_isDuplicate();
  if (pT4_zHi_branch != 0) pT4_zHi();
  if (t5_moduleType_binary_branch != 0) t5_moduleType_binary();
  if (t3_residual_branch != 0) t3_residual();
  if (t3_occupancies_branch != 0) t3_occupancies();
  if (sim_pT4_types_branch != 0) sim_pT4_types();
  if (t4_deltaBetaCut_branch != 0) t4_deltaBetaCut();
  if (t5_pt_branch != 0) t5_pt();
  if (sim_len_branch != 0) sim_len();
  if (sim_lengap_branch != 0) sim_lengap();
  if (sim_hits_branch != 0) sim_hits();
  if (simvtx_x_branch != 0) simvtx_x();
  if (simvtx_y_branch != 0) simvtx_y();
  if (simvtx_z_branch != 0) simvtx_z();
  if (sim_T4_matched_branch != 0) sim_T4_matched();
  if (t4_rtOut_branch != 0) t4_rtOut();
  if (pT3_pt_branch != 0) pT3_pt();
  if (tc_pt_branch != 0) tc_pt();
  if (pT3_pixelRadiusError_branch != 0) pT3_pixelRadiusError();
  if (pT5_isFake_branch != 0) pT5_isFake();
  if (pT5_pt_branch != 0) pT5_pt();
  if (pT4_deltaBeta_branch != 0) pT4_deltaBeta();
  if (t5_innerRadiusMax_branch != 0) t5_innerRadiusMax();
  if (sim_phi_branch != 0) sim_phi();
  if (t4_betaInCut_branch != 0) t4_betaInCut();
  if (t5_innerRadiusMin_branch != 0) t5_innerRadiusMin();
  if (pT4_sdlCut_branch != 0) pT4_sdlCut();
  if (pT3_hit_idx3_branch != 0) pT3_hit_idx3();
  if (pT4_zHiPointed_branch != 0) pT4_zHiPointed();
  if (pT3_hit_idx1_branch != 0) pT3_hit_idx1();
  if (sim_pca_dz_branch != 0) sim_pca_dz();
  if (t4_deltaBeta_branch != 0) t4_deltaBeta();
  if (pT3_hit_idx5_branch != 0) pT3_hit_idx5();
  if (pT3_hit_idx4_branch != 0) pT3_hit_idx4();
  if (pT5_layer_binary_branch != 0) pT5_layer_binary();
  if (t5_bridgeRadius_branch != 0) t5_bridgeRadius();
  if (sim_pLS_matched_branch != 0) sim_pLS_matched();
  if (pT4_isFake_branch != 0) pT4_isFake();
  if (sim_T3_matched_branch != 0) sim_T3_matched();
  if (t3_deltaPhiPos_branch != 0) t3_deltaPhiPos();
  if (pT3_phi_branch != 0) pT3_phi();
  if (t5_matched_pt_branch != 0) t5_matched_pt();
  if (pT3_eta_branch != 0) pT3_eta();
  if (t4_eta_branch != 0) t4_eta();
  if (t3_deltaPhi_branch != 0) t3_deltaPhi();
  if (pLS_isFake_branch != 0) pLS_isFake();
  if (pT4_betaIn_branch != 0) pT4_betaIn();
  if (sim_bunchCrossing_branch != 0) sim_bunchCrossing();
  if (pT4_zOut_branch != 0) pT4_zOut();
  if (pT4_deltaPhiPos_branch != 0) pT4_deltaPhiPos();
  if (sim_parentVtxIdx_branch != 0) sim_parentVtxIdx();
  if (t3_zHi_branch != 0) t3_zHi();
  if (sim_pT4_matched_branch != 0) sim_pT4_matched();
  if (t5_innerRadiusMin2S_branch != 0) t5_innerRadiusMin2S();
  if (tc_eta_branch != 0) tc_eta();
  if (tc_phi_branch != 0) tc_phi();
  if (sim_T5_matched_branch != 0) sim_T5_matched();
  if (sim_T5_types_branch != 0) sim_T5_types();
  if (t5_isDuplicate_branch != 0) t5_isDuplicate();
  if (t4_zHiPointed_branch != 0) t4_zHiPointed();
  if (pT4_rtHi_branch != 0) pT4_rtHi();
  if (t5_outerRadiusMin2S_branch != 0) t5_outerRadiusMin2S();
  if (t3_betaIn_branch != 0) t3_betaIn();
  if (pT3_occupancies_branch != 0) pT3_occupancies();
  if (tc_occupancies_branch != 0) tc_occupancies();
  if (t5_innerRadius_branch != 0) t5_innerRadius();
  if (sim_TC_matched_branch != 0) sim_TC_matched();
  if (pLS_isDuplicate_branch != 0) pLS_isDuplicate();
  if (t5_occupancies_branch != 0) t5_occupancies();
  if (t3_layer1_branch != 0) t3_layer1();
  if (pT4_kZ_branch != 0) pT4_kZ();
  if (pT3_hit_idx2_branch != 0) pT3_hit_idx2();
  if (pLS_pt_branch != 0) pLS_pt();
  if (sim_T4_types_branch != 0) sim_T4_types();
  if (pT4_isDuplicate_branch != 0) pT4_isDuplicate();
  if (t4_pt_branch != 0) t4_pt();
  if (t4_zHi_branch != 0) t4_zHi();
  if (sim_TC_types_branch != 0) sim_TC_types();
  if (t3_kZ_branch != 0) t3_kZ();
  if (t4_moduleType_binary_branch != 0) t4_moduleType_binary();
  if (sg_occupancies_branch != 0) sg_occupancies();
  if (pT4_pt_branch != 0) pT4_pt();
  if (pT3_hit_idx6_branch != 0) pT3_hit_idx6();
  if (pT3_pix_idx4_branch != 0) pT3_pix_idx4();
  if (sim_vx_branch != 0) sim_vx();
  if (sim_vy_branch != 0) sim_vy();
  if (sim_vz_branch != 0) sim_vz();
  if (t4_sdlCut_branch != 0) t4_sdlCut();
  if (pT4_rtLo_branch != 0) pT4_rtLo();
  if (t5_innerRadiusMax2S_branch != 0) t5_innerRadiusMax2S();
  if (t3_pt_branch != 0) t3_pt();
  if (module_rings_branch != 0) module_rings();
  if (t3_zLo_branch != 0) t3_zLo();
  if (pT4_deltaBetaCut_branch != 0) pT4_deltaBetaCut();
  if (t4_rtHi_branch != 0) t4_rtHi();
  if (t3_layer2_branch != 0) t3_layer2();
  if (sim_T3_types_branch != 0) sim_T3_types();
  if (sim_pT5_types_branch != 0) sim_pT5_types();
  if (sim_pT5_matched_branch != 0) sim_pT5_matched();
  if (module_layers_branch != 0) module_layers();
  if (pT4_eta_branch != 0) pT4_eta();
}
const int &SDL::pT5_occupancies() {
  if (not pT5_occupancies_isLoaded) {
    if (pT5_occupancies_branch != 0) {
      pT5_occupancies_branch->GetEntry(index);
    } else {
      printf("branch pT5_occupancies_branch does not exist!\n");
      exit(1);
    }
    pT5_occupancies_isLoaded = true;
  }
  return pT5_occupancies_;
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
const vector<float> &SDL::t4_zLoPointed() {
  if (not t4_zLoPointed_isLoaded) {
    if (t4_zLoPointed_branch != 0) {
      t4_zLoPointed_branch->GetEntry(index);
    } else {
      printf("branch t4_zLoPointed_branch does not exist!\n");
      exit(1);
    }
    t4_zLoPointed_isLoaded = true;
  }
  return *t4_zLoPointed_;
}
const vector<float> &SDL::t4_kZ() {
  if (not t4_kZ_isLoaded) {
    if (t4_kZ_branch != 0) {
      t4_kZ_branch->GetEntry(index);
    } else {
      printf("branch t4_kZ_branch does not exist!\n");
      exit(1);
    }
    t4_kZ_isLoaded = true;
  }
  return *t4_kZ_;
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
const vector<float> &SDL::t4_zOut() {
  if (not t4_zOut_isLoaded) {
    if (t4_zOut_branch != 0) {
      t4_zOut_branch->GetEntry(index);
    } else {
      printf("branch t4_zOut_branch does not exist!\n");
      exit(1);
    }
    t4_zOut_isLoaded = true;
  }
  return *t4_zOut_;
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
const vector<float> &SDL::t3_rtHi() {
  if (not t3_rtHi_isLoaded) {
    if (t3_rtHi_branch != 0) {
      t3_rtHi_branch->GetEntry(index);
    } else {
      printf("branch t3_rtHi_branch does not exist!\n");
      exit(1);
    }
    t3_rtHi_isLoaded = true;
  }
  return *t3_rtHi_;
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
const vector<float> &SDL::pT4_betaOut() {
  if (not pT4_betaOut_isLoaded) {
    if (pT4_betaOut_branch != 0) {
      pT4_betaOut_branch->GetEntry(index);
    } else {
      printf("branch pT4_betaOut_branch does not exist!\n");
      exit(1);
    }
    pT4_betaOut_isLoaded = true;
  }
  return *pT4_betaOut_;
}
const vector<float> &SDL::pT4_zLo() {
  if (not pT4_zLo_isLoaded) {
    if (pT4_zLo_branch != 0) {
      pT4_zLo_branch->GetEntry(index);
    } else {
      printf("branch pT4_zLo_branch does not exist!\n");
      exit(1);
    }
    pT4_zLo_isLoaded = true;
  }
  return *pT4_zLo_;
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
const vector<int> &SDL::sim_denom() {
  if (not sim_denom_isLoaded) {
    if (sim_denom_branch != 0) {
      sim_denom_branch->GetEntry(index);
    } else {
      printf("branch sim_denom_branch does not exist!\n");
      exit(1);
    }
    sim_denom_isLoaded = true;
  }
  return *sim_denom_;
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
const vector<int> &SDL::t3_layer3() {
  if (not t3_layer3_isLoaded) {
    if (t3_layer3_branch != 0) {
      t3_layer3_branch->GetEntry(index);
    } else {
      printf("branch t3_layer3_branch does not exist!\n");
      exit(1);
    }
    t3_layer3_isLoaded = true;
  }
  return *t3_layer3_;
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
const vector<float> &SDL::t4_deltaPhiPos() {
  if (not t4_deltaPhiPos_isLoaded) {
    if (t4_deltaPhiPos_branch != 0) {
      t4_deltaPhiPos_branch->GetEntry(index);
    } else {
      printf("branch t4_deltaPhiPos_branch does not exist!\n");
      exit(1);
    }
    t4_deltaPhiPos_isLoaded = true;
  }
  return *t4_deltaPhiPos_;
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
const vector<float> &SDL::t3_zHiPointed() {
  if (not t3_zHiPointed_isLoaded) {
    if (t3_zHiPointed_branch != 0) {
      t3_zHiPointed_branch->GetEntry(index);
    } else {
      printf("branch t3_zHiPointed_branch does not exist!\n");
      exit(1);
    }
    t3_zHiPointed_isLoaded = true;
  }
  return *t3_zHiPointed_;
}
const vector<float> &SDL::t4_betaOut() {
  if (not t4_betaOut_isLoaded) {
    if (t4_betaOut_branch != 0) {
      t4_betaOut_branch->GetEntry(index);
    } else {
      printf("branch t4_betaOut_branch does not exist!\n");
      exit(1);
    }
    t4_betaOut_isLoaded = true;
  }
  return *t4_betaOut_;
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
const vector<float> &SDL::t4_betaOutCut() {
  if (not t4_betaOutCut_isLoaded) {
    if (t4_betaOutCut_branch != 0) {
      t4_betaOutCut_branch->GetEntry(index);
    } else {
      printf("branch t4_betaOutCut_branch does not exist!\n");
      exit(1);
    }
    t4_betaOutCut_isLoaded = true;
  }
  return *t4_betaOutCut_;
}
const vector<float> &SDL::t3_betaOut() {
  if (not t3_betaOut_isLoaded) {
    if (t3_betaOut_branch != 0) {
      t3_betaOut_branch->GetEntry(index);
    } else {
      printf("branch t3_betaOut_branch does not exist!\n");
      exit(1);
    }
    t3_betaOut_isLoaded = true;
  }
  return *t3_betaOut_;
}
const vector<float> &SDL::t3_sdlCut() {
  if (not t3_sdlCut_isLoaded) {
    if (t3_sdlCut_branch != 0) {
      t3_sdlCut_branch->GetEntry(index);
    } else {
      printf("branch t3_sdlCut_branch does not exist!\n");
      exit(1);
    }
    t3_sdlCut_isLoaded = true;
  }
  return *t3_sdlCut_;
}
const vector<float> &SDL::pT4_rtOut() {
  if (not pT4_rtOut_isLoaded) {
    if (pT4_rtOut_branch != 0) {
      pT4_rtOut_branch->GetEntry(index);
    } else {
      printf("branch pT4_rtOut_branch does not exist!\n");
      exit(1);
    }
    pT4_rtOut_isLoaded = true;
  }
  return *pT4_rtOut_;
}
const vector<float> &SDL::t3_betaInCut() {
  if (not t3_betaInCut_isLoaded) {
    if (t3_betaInCut_branch != 0) {
      t3_betaInCut_branch->GetEntry(index);
    } else {
      printf("branch t3_betaInCut_branch does not exist!\n");
      exit(1);
    }
    t3_betaInCut_isLoaded = true;
  }
  return *t3_betaInCut_;
}
const vector<float> &SDL::pT4_betaOutCut() {
  if (not pT4_betaOutCut_isLoaded) {
    if (pT4_betaOutCut_branch != 0) {
      pT4_betaOutCut_branch->GetEntry(index);
    } else {
      printf("branch pT4_betaOutCut_branch does not exist!\n");
      exit(1);
    }
    pT4_betaOutCut_isLoaded = true;
  }
  return *pT4_betaOutCut_;
}
const vector<float> &SDL::pT4_betaInCut() {
  if (not pT4_betaInCut_isLoaded) {
    if (pT4_betaInCut_branch != 0) {
      pT4_betaInCut_branch->GetEntry(index);
    } else {
      printf("branch pT4_betaInCut_branch does not exist!\n");
      exit(1);
    }
    pT4_betaInCut_isLoaded = true;
  }
  return *pT4_betaInCut_;
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
const vector<float> &SDL::pT4_deltaPhi() {
  if (not pT4_deltaPhi_isLoaded) {
    if (pT4_deltaPhi_branch != 0) {
      pT4_deltaPhi_branch->GetEntry(index);
    } else {
      printf("branch pT4_deltaPhi_branch does not exist!\n");
      exit(1);
    }
    pT4_deltaPhi_isLoaded = true;
  }
  return *pT4_deltaPhi_;
}
const vector<float> &SDL::t3_zLoPointed() {
  if (not t3_zLoPointed_isLoaded) {
    if (t3_zLoPointed_branch != 0) {
      t3_zLoPointed_branch->GetEntry(index);
    } else {
      printf("branch t3_zLoPointed_branch does not exist!\n");
      exit(1);
    }
    t3_zLoPointed_isLoaded = true;
  }
  return *t3_zLoPointed_;
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
const vector<float> &SDL::t3_deltaBetaCut() {
  if (not t3_deltaBetaCut_isLoaded) {
    if (t3_deltaBetaCut_branch != 0) {
      t3_deltaBetaCut_branch->GetEntry(index);
    } else {
      printf("branch t3_deltaBetaCut_branch does not exist!\n");
      exit(1);
    }
    t3_deltaBetaCut_isLoaded = true;
  }
  return *t3_deltaBetaCut_;
}
const vector<int> &SDL::t3_moduleType_binary() {
  if (not t3_moduleType_binary_isLoaded) {
    if (t3_moduleType_binary_branch != 0) {
      t3_moduleType_binary_branch->GetEntry(index);
    } else {
      printf("branch t3_moduleType_binary_branch does not exist!\n");
      exit(1);
    }
    t3_moduleType_binary_isLoaded = true;
  }
  return *t3_moduleType_binary_;
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
const vector<float> &SDL::t5_bridgeRadiusMax2S() {
  if (not t5_bridgeRadiusMax2S_isLoaded) {
    if (t5_bridgeRadiusMax2S_branch != 0) {
      t5_bridgeRadiusMax2S_branch->GetEntry(index);
    } else {
      printf("branch t5_bridgeRadiusMax2S_branch does not exist!\n");
      exit(1);
    }
    t5_bridgeRadiusMax2S_isLoaded = true;
  }
  return *t5_bridgeRadiusMax2S_;
}
const vector<float> &SDL::t5_outerRadiusMax2S() {
  if (not t5_outerRadiusMax2S_isLoaded) {
    if (t5_outerRadiusMax2S_branch != 0) {
      t5_outerRadiusMax2S_branch->GetEntry(index);
    } else {
      printf("branch t5_outerRadiusMax2S_branch does not exist!\n");
      exit(1);
    }
    t5_outerRadiusMax2S_isLoaded = true;
  }
  return *t5_outerRadiusMax2S_;
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
const vector<int> &SDL::pT4_layer_binary() {
  if (not pT4_layer_binary_isLoaded) {
    if (pT4_layer_binary_branch != 0) {
      pT4_layer_binary_branch->GetEntry(index);
    } else {
      printf("branch pT4_layer_binary_branch does not exist!\n");
      exit(1);
    }
    pT4_layer_binary_isLoaded = true;
  }
  return *pT4_layer_binary_;
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
const vector<int> &SDL::pT3_pix_idx3() {
  if (not pT3_pix_idx3_isLoaded) {
    if (pT3_pix_idx3_branch != 0) {
      pT3_pix_idx3_branch->GetEntry(index);
    } else {
      printf("branch pT3_pix_idx3_branch does not exist!\n");
      exit(1);
    }
    pT3_pix_idx3_isLoaded = true;
  }
  return *pT3_pix_idx3_;
}
const vector<int> &SDL::pT3_pix_idx2() {
  if (not pT3_pix_idx2_isLoaded) {
    if (pT3_pix_idx2_branch != 0) {
      pT3_pix_idx2_branch->GetEntry(index);
    } else {
      printf("branch pT3_pix_idx2_branch does not exist!\n");
      exit(1);
    }
    pT3_pix_idx2_isLoaded = true;
  }
  return *pT3_pix_idx2_;
}
const vector<int> &SDL::pT3_pix_idx1() {
  if (not pT3_pix_idx1_isLoaded) {
    if (pT3_pix_idx1_branch != 0) {
      pT3_pix_idx1_branch->GetEntry(index);
    } else {
      printf("branch pT3_pix_idx1_branch does not exist!\n");
      exit(1);
    }
    pT3_pix_idx1_isLoaded = true;
  }
  return *pT3_pix_idx1_;
}
const vector<float> &SDL::t5_bridgeRadiusMax() {
  if (not t5_bridgeRadiusMax_isLoaded) {
    if (t5_bridgeRadiusMax_branch != 0) {
      t5_bridgeRadiusMax_branch->GetEntry(index);
    } else {
      printf("branch t5_bridgeRadiusMax_branch does not exist!\n");
      exit(1);
    }
    t5_bridgeRadiusMax_isLoaded = true;
  }
  return *t5_bridgeRadiusMax_;
}
const vector<float> &SDL::t5_bridgeRadiusMin2S() {
  if (not t5_bridgeRadiusMin2S_isLoaded) {
    if (t5_bridgeRadiusMin2S_branch != 0) {
      t5_bridgeRadiusMin2S_branch->GetEntry(index);
    } else {
      printf("branch t5_bridgeRadiusMin2S_branch does not exist!\n");
      exit(1);
    }
    t5_bridgeRadiusMin2S_isLoaded = true;
  }
  return *t5_bridgeRadiusMin2S_;
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
const vector<float> &SDL::pT4_zLoPointed() {
  if (not pT4_zLoPointed_isLoaded) {
    if (pT4_zLoPointed_branch != 0) {
      pT4_zLoPointed_branch->GetEntry(index);
    } else {
      printf("branch pT4_zLoPointed_branch does not exist!\n");
      exit(1);
    }
    pT4_zLoPointed_isLoaded = true;
  }
  return *pT4_zLoPointed_;
}
const vector<int> &SDL::t3_hit_idx4() {
  if (not t3_hit_idx4_isLoaded) {
    if (t3_hit_idx4_branch != 0) {
      t3_hit_idx4_branch->GetEntry(index);
    } else {
      printf("branch t3_hit_idx4_branch does not exist!\n");
      exit(1);
    }
    t3_hit_idx4_isLoaded = true;
  }
  return *t3_hit_idx4_;
}
const vector<int> &SDL::t3_hit_idx5() {
  if (not t3_hit_idx5_isLoaded) {
    if (t3_hit_idx5_branch != 0) {
      t3_hit_idx5_branch->GetEntry(index);
    } else {
      printf("branch t3_hit_idx5_branch does not exist!\n");
      exit(1);
    }
    t3_hit_idx5_isLoaded = true;
  }
  return *t3_hit_idx5_;
}
const vector<int> &SDL::t3_hit_idx6() {
  if (not t3_hit_idx6_isLoaded) {
    if (t3_hit_idx6_branch != 0) {
      t3_hit_idx6_branch->GetEntry(index);
    } else {
      printf("branch t3_hit_idx6_branch does not exist!\n");
      exit(1);
    }
    t3_hit_idx6_isLoaded = true;
  }
  return *t3_hit_idx6_;
}
const vector<float> &SDL::t3_rtOut() {
  if (not t3_rtOut_isLoaded) {
    if (t3_rtOut_branch != 0) {
      t3_rtOut_branch->GetEntry(index);
    } else {
      printf("branch t3_rtOut_branch does not exist!\n");
      exit(1);
    }
    t3_rtOut_isLoaded = true;
  }
  return *t3_rtOut_;
}
const vector<int> &SDL::t3_hit_idx1() {
  if (not t3_hit_idx1_isLoaded) {
    if (t3_hit_idx1_branch != 0) {
      t3_hit_idx1_branch->GetEntry(index);
    } else {
      printf("branch t3_hit_idx1_branch does not exist!\n");
      exit(1);
    }
    t3_hit_idx1_isLoaded = true;
  }
  return *t3_hit_idx1_;
}
const vector<int> &SDL::t3_hit_idx2() {
  if (not t3_hit_idx2_isLoaded) {
    if (t3_hit_idx2_branch != 0) {
      t3_hit_idx2_branch->GetEntry(index);
    } else {
      printf("branch t3_hit_idx2_branch does not exist!\n");
      exit(1);
    }
    t3_hit_idx2_isLoaded = true;
  }
  return *t3_hit_idx2_;
}
const vector<int> &SDL::t3_hit_idx3() {
  if (not t3_hit_idx3_isLoaded) {
    if (t3_hit_idx3_branch != 0) {
      t3_hit_idx3_branch->GetEntry(index);
    } else {
      printf("branch t3_hit_idx3_branch does not exist!\n");
      exit(1);
    }
    t3_hit_idx3_isLoaded = true;
  }
  return *t3_hit_idx3_;
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
const vector<float> &SDL::t5_bridgeRadiusMin() {
  if (not t5_bridgeRadiusMin_isLoaded) {
    if (t5_bridgeRadiusMin_branch != 0) {
      t5_bridgeRadiusMin_branch->GetEntry(index);
    } else {
      printf("branch t5_bridgeRadiusMin_branch does not exist!\n");
      exit(1);
    }
    t5_bridgeRadiusMin_isLoaded = true;
  }
  return *t5_bridgeRadiusMin_;
}
const vector<float> &SDL::t4_zLo() {
  if (not t4_zLo_isLoaded) {
    if (t4_zLo_branch != 0) {
      t4_zLo_branch->GetEntry(index);
    } else {
      printf("branch t4_zLo_branch does not exist!\n");
      exit(1);
    }
    t4_zLo_isLoaded = true;
  }
  return *t4_zLo_;
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
const vector<int> &SDL::t3_layer_binary() {
  if (not t3_layer_binary_isLoaded) {
    if (t3_layer_binary_branch != 0) {
      t3_layer_binary_branch->GetEntry(index);
    } else {
      printf("branch t3_layer_binary_branch does not exist!\n");
      exit(1);
    }
    t3_layer_binary_isLoaded = true;
  }
  return *t3_layer_binary_;
}
const vector<int> &SDL::t4_layer_binary() {
  if (not t4_layer_binary_isLoaded) {
    if (t4_layer_binary_branch != 0) {
      t4_layer_binary_branch->GetEntry(index);
    } else {
      printf("branch t4_layer_binary_branch does not exist!\n");
      exit(1);
    }
    t4_layer_binary_isLoaded = true;
  }
  return *t4_layer_binary_;
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
const vector<float> &SDL::t4_deltaPhi() {
  if (not t4_deltaPhi_isLoaded) {
    if (t4_deltaPhi_branch != 0) {
      t4_deltaPhi_branch->GetEntry(index);
    } else {
      printf("branch t4_deltaPhi_branch does not exist!\n");
      exit(1);
    }
    t4_deltaPhi_isLoaded = true;
  }
  return *t4_deltaPhi_;
}
const vector<float> &SDL::t4_rtLo() {
  if (not t4_rtLo_isLoaded) {
    if (t4_rtLo_branch != 0) {
      t4_rtLo_branch->GetEntry(index);
    } else {
      printf("branch t4_rtLo_branch does not exist!\n");
      exit(1);
    }
    t4_rtLo_isLoaded = true;
  }
  return *t4_rtLo_;
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
const vector<float> &SDL::t4_betaIn() {
  if (not t4_betaIn_isLoaded) {
    if (t4_betaIn_branch != 0) {
      t4_betaIn_branch->GetEntry(index);
    } else {
      printf("branch t4_betaIn_branch does not exist!\n");
      exit(1);
    }
    t4_betaIn_isLoaded = true;
  }
  return *t4_betaIn_;
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
const vector<float> &SDL::t3_zOut() {
  if (not t3_zOut_isLoaded) {
    if (t3_zOut_branch != 0) {
      t3_zOut_branch->GetEntry(index);
    } else {
      printf("branch t3_zOut_branch does not exist!\n");
      exit(1);
    }
    t3_zOut_isLoaded = true;
  }
  return *t3_zOut_;
}
const vector<float> &SDL::t5_outerRadiusMax() {
  if (not t5_outerRadiusMax_isLoaded) {
    if (t5_outerRadiusMax_branch != 0) {
      t5_outerRadiusMax_branch->GetEntry(index);
    } else {
      printf("branch t5_outerRadiusMax_branch does not exist!\n");
      exit(1);
    }
    t5_outerRadiusMax_isLoaded = true;
  }
  return *t5_outerRadiusMax_;
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
const vector<float> &SDL::t3_deltaBeta() {
  if (not t3_deltaBeta_isLoaded) {
    if (t3_deltaBeta_branch != 0) {
      t3_deltaBeta_branch->GetEntry(index);
    } else {
      printf("branch t3_deltaBeta_branch does not exist!\n");
      exit(1);
    }
    t3_deltaBeta_isLoaded = true;
  }
  return *t3_deltaBeta_;
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
const vector<float> &SDL::t5_outerRadiusMin() {
  if (not t5_outerRadiusMin_isLoaded) {
    if (t5_outerRadiusMin_branch != 0) {
      t5_outerRadiusMin_branch->GetEntry(index);
    } else {
      printf("branch t5_outerRadiusMin_branch does not exist!\n");
      exit(1);
    }
    t5_outerRadiusMin_isLoaded = true;
  }
  return *t5_outerRadiusMin_;
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
const vector<float> &SDL::t3_rtLo() {
  if (not t3_rtLo_isLoaded) {
    if (t3_rtLo_branch != 0) {
      t3_rtLo_branch->GetEntry(index);
    } else {
      printf("branch t3_rtLo_branch does not exist!\n");
      exit(1);
    }
    t3_rtLo_isLoaded = true;
  }
  return *t3_rtLo_;
}
const vector<float> &SDL::t3_betaOutCut() {
  if (not t3_betaOutCut_isLoaded) {
    if (t3_betaOutCut_branch != 0) {
      t3_betaOutCut_branch->GetEntry(index);
    } else {
      printf("branch t3_betaOutCut_branch does not exist!\n");
      exit(1);
    }
    t3_betaOutCut_isLoaded = true;
  }
  return *t3_betaOutCut_;
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
const vector<float> &SDL::pT4_zHi() {
  if (not pT4_zHi_isLoaded) {
    if (pT4_zHi_branch != 0) {
      pT4_zHi_branch->GetEntry(index);
    } else {
      printf("branch pT4_zHi_branch does not exist!\n");
      exit(1);
    }
    pT4_zHi_isLoaded = true;
  }
  return *pT4_zHi_;
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
const vector<float> &SDL::t3_residual() {
  if (not t3_residual_isLoaded) {
    if (t3_residual_branch != 0) {
      t3_residual_branch->GetEntry(index);
    } else {
      printf("branch t3_residual_branch does not exist!\n");
      exit(1);
    }
    t3_residual_isLoaded = true;
  }
  return *t3_residual_;
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
const vector<float> &SDL::t4_deltaBetaCut() {
  if (not t4_deltaBetaCut_isLoaded) {
    if (t4_deltaBetaCut_branch != 0) {
      t4_deltaBetaCut_branch->GetEntry(index);
    } else {
      printf("branch t4_deltaBetaCut_branch does not exist!\n");
      exit(1);
    }
    t4_deltaBetaCut_isLoaded = true;
  }
  return *t4_deltaBetaCut_;
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
const vector<float> &SDL::sim_hits() {
  if (not sim_hits_isLoaded) {
    if (sim_hits_branch != 0) {
      sim_hits_branch->GetEntry(index);
    } else {
      printf("branch sim_hits_branch does not exist!\n");
      exit(1);
    }
    sim_hits_isLoaded = true;
  }
  return *sim_hits_;
}
const vector<float> &SDL::sim_len() {
  if (not sim_len_isLoaded) {
    if (sim_len_branch != 0) {
      sim_len_branch->GetEntry(index);
    } else {
      printf("branch sim_len_branch does not exist!\n");
      exit(1);
    }
    sim_len_isLoaded = true;
  }
  return *sim_len_;
}
const vector<float> &SDL::sim_lengap() {
  if (not sim_lengap_isLoaded) {
    if (sim_lengap_branch != 0) {
      sim_lengap_branch->GetEntry(index);
    } else {
      printf("branch sim_lengap_branch does not exist!\n");
      exit(1);
    }
    sim_lengap_isLoaded = true;
  }
  return *sim_lengap_;
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
const vector<float> &SDL::t4_rtOut() {
  if (not t4_rtOut_isLoaded) {
    if (t4_rtOut_branch != 0) {
      t4_rtOut_branch->GetEntry(index);
    } else {
      printf("branch t4_rtOut_branch does not exist!\n");
      exit(1);
    }
    t4_rtOut_isLoaded = true;
  }
  return *t4_rtOut_;
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
const vector<float> &SDL::pT4_deltaBeta() {
  if (not pT4_deltaBeta_isLoaded) {
    if (pT4_deltaBeta_branch != 0) {
      pT4_deltaBeta_branch->GetEntry(index);
    } else {
      printf("branch pT4_deltaBeta_branch does not exist!\n");
      exit(1);
    }
    pT4_deltaBeta_isLoaded = true;
  }
  return *pT4_deltaBeta_;
}
const vector<float> &SDL::t5_innerRadiusMax() {
  if (not t5_innerRadiusMax_isLoaded) {
    if (t5_innerRadiusMax_branch != 0) {
      t5_innerRadiusMax_branch->GetEntry(index);
    } else {
      printf("branch t5_innerRadiusMax_branch does not exist!\n");
      exit(1);
    }
    t5_innerRadiusMax_isLoaded = true;
  }
  return *t5_innerRadiusMax_;
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
const vector<float> &SDL::t4_betaInCut() {
  if (not t4_betaInCut_isLoaded) {
    if (t4_betaInCut_branch != 0) {
      t4_betaInCut_branch->GetEntry(index);
    } else {
      printf("branch t4_betaInCut_branch does not exist!\n");
      exit(1);
    }
    t4_betaInCut_isLoaded = true;
  }
  return *t4_betaInCut_;
}
const vector<float> &SDL::t5_innerRadiusMin() {
  if (not t5_innerRadiusMin_isLoaded) {
    if (t5_innerRadiusMin_branch != 0) {
      t5_innerRadiusMin_branch->GetEntry(index);
    } else {
      printf("branch t5_innerRadiusMin_branch does not exist!\n");
      exit(1);
    }
    t5_innerRadiusMin_isLoaded = true;
  }
  return *t5_innerRadiusMin_;
}
const vector<float> &SDL::pT4_sdlCut() {
  if (not pT4_sdlCut_isLoaded) {
    if (pT4_sdlCut_branch != 0) {
      pT4_sdlCut_branch->GetEntry(index);
    } else {
      printf("branch pT4_sdlCut_branch does not exist!\n");
      exit(1);
    }
    pT4_sdlCut_isLoaded = true;
  }
  return *pT4_sdlCut_;
}
const vector<int> &SDL::pT3_hit_idx3() {
  if (not pT3_hit_idx3_isLoaded) {
    if (pT3_hit_idx3_branch != 0) {
      pT3_hit_idx3_branch->GetEntry(index);
    } else {
      printf("branch pT3_hit_idx3_branch does not exist!\n");
      exit(1);
    }
    pT3_hit_idx3_isLoaded = true;
  }
  return *pT3_hit_idx3_;
}
const vector<float> &SDL::pT4_zHiPointed() {
  if (not pT4_zHiPointed_isLoaded) {
    if (pT4_zHiPointed_branch != 0) {
      pT4_zHiPointed_branch->GetEntry(index);
    } else {
      printf("branch pT4_zHiPointed_branch does not exist!\n");
      exit(1);
    }
    pT4_zHiPointed_isLoaded = true;
  }
  return *pT4_zHiPointed_;
}
const vector<int> &SDL::pT3_hit_idx1() {
  if (not pT3_hit_idx1_isLoaded) {
    if (pT3_hit_idx1_branch != 0) {
      pT3_hit_idx1_branch->GetEntry(index);
    } else {
      printf("branch pT3_hit_idx1_branch does not exist!\n");
      exit(1);
    }
    pT3_hit_idx1_isLoaded = true;
  }
  return *pT3_hit_idx1_;
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
const vector<float> &SDL::t4_deltaBeta() {
  if (not t4_deltaBeta_isLoaded) {
    if (t4_deltaBeta_branch != 0) {
      t4_deltaBeta_branch->GetEntry(index);
    } else {
      printf("branch t4_deltaBeta_branch does not exist!\n");
      exit(1);
    }
    t4_deltaBeta_isLoaded = true;
  }
  return *t4_deltaBeta_;
}
const vector<int> &SDL::pT3_hit_idx5() {
  if (not pT3_hit_idx5_isLoaded) {
    if (pT3_hit_idx5_branch != 0) {
      pT3_hit_idx5_branch->GetEntry(index);
    } else {
      printf("branch pT3_hit_idx5_branch does not exist!\n");
      exit(1);
    }
    pT3_hit_idx5_isLoaded = true;
  }
  return *pT3_hit_idx5_;
}
const vector<int> &SDL::pT3_hit_idx4() {
  if (not pT3_hit_idx4_isLoaded) {
    if (pT3_hit_idx4_branch != 0) {
      pT3_hit_idx4_branch->GetEntry(index);
    } else {
      printf("branch pT3_hit_idx4_branch does not exist!\n");
      exit(1);
    }
    pT3_hit_idx4_isLoaded = true;
  }
  return *pT3_hit_idx4_;
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
const vector<float> &SDL::t3_deltaPhiPos() {
  if (not t3_deltaPhiPos_isLoaded) {
    if (t3_deltaPhiPos_branch != 0) {
      t3_deltaPhiPos_branch->GetEntry(index);
    } else {
      printf("branch t3_deltaPhiPos_branch does not exist!\n");
      exit(1);
    }
    t3_deltaPhiPos_isLoaded = true;
  }
  return *t3_deltaPhiPos_;
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
const vector<vector<float> > &SDL::t5_matched_pt() {
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
const vector<float> &SDL::t3_deltaPhi() {
  if (not t3_deltaPhi_isLoaded) {
    if (t3_deltaPhi_branch != 0) {
      t3_deltaPhi_branch->GetEntry(index);
    } else {
      printf("branch t3_deltaPhi_branch does not exist!\n");
      exit(1);
    }
    t3_deltaPhi_isLoaded = true;
  }
  return *t3_deltaPhi_;
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
const vector<float> &SDL::pT4_betaIn() {
  if (not pT4_betaIn_isLoaded) {
    if (pT4_betaIn_branch != 0) {
      pT4_betaIn_branch->GetEntry(index);
    } else {
      printf("branch pT4_betaIn_branch does not exist!\n");
      exit(1);
    }
    pT4_betaIn_isLoaded = true;
  }
  return *pT4_betaIn_;
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
const vector<float> &SDL::pT4_zOut() {
  if (not pT4_zOut_isLoaded) {
    if (pT4_zOut_branch != 0) {
      pT4_zOut_branch->GetEntry(index);
    } else {
      printf("branch pT4_zOut_branch does not exist!\n");
      exit(1);
    }
    pT4_zOut_isLoaded = true;
  }
  return *pT4_zOut_;
}
const vector<float> &SDL::pT4_deltaPhiPos() {
  if (not pT4_deltaPhiPos_isLoaded) {
    if (pT4_deltaPhiPos_branch != 0) {
      pT4_deltaPhiPos_branch->GetEntry(index);
    } else {
      printf("branch pT4_deltaPhiPos_branch does not exist!\n");
      exit(1);
    }
    pT4_deltaPhiPos_isLoaded = true;
  }
  return *pT4_deltaPhiPos_;
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
const vector<float> &SDL::t3_zHi() {
  if (not t3_zHi_isLoaded) {
    if (t3_zHi_branch != 0) {
      t3_zHi_branch->GetEntry(index);
    } else {
      printf("branch t3_zHi_branch does not exist!\n");
      exit(1);
    }
    t3_zHi_isLoaded = true;
  }
  return *t3_zHi_;
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
const vector<float> &SDL::t5_innerRadiusMin2S() {
  if (not t5_innerRadiusMin2S_isLoaded) {
    if (t5_innerRadiusMin2S_branch != 0) {
      t5_innerRadiusMin2S_branch->GetEntry(index);
    } else {
      printf("branch t5_innerRadiusMin2S_branch does not exist!\n");
      exit(1);
    }
    t5_innerRadiusMin2S_isLoaded = true;
  }
  return *t5_innerRadiusMin2S_;
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
const vector<float> &SDL::t4_zHiPointed() {
  if (not t4_zHiPointed_isLoaded) {
    if (t4_zHiPointed_branch != 0) {
      t4_zHiPointed_branch->GetEntry(index);
    } else {
      printf("branch t4_zHiPointed_branch does not exist!\n");
      exit(1);
    }
    t4_zHiPointed_isLoaded = true;
  }
  return *t4_zHiPointed_;
}
const vector<float> &SDL::pT4_rtHi() {
  if (not pT4_rtHi_isLoaded) {
    if (pT4_rtHi_branch != 0) {
      pT4_rtHi_branch->GetEntry(index);
    } else {
      printf("branch pT4_rtHi_branch does not exist!\n");
      exit(1);
    }
    pT4_rtHi_isLoaded = true;
  }
  return *pT4_rtHi_;
}
const vector<float> &SDL::t5_outerRadiusMin2S() {
  if (not t5_outerRadiusMin2S_isLoaded) {
    if (t5_outerRadiusMin2S_branch != 0) {
      t5_outerRadiusMin2S_branch->GetEntry(index);
    } else {
      printf("branch t5_outerRadiusMin2S_branch does not exist!\n");
      exit(1);
    }
    t5_outerRadiusMin2S_isLoaded = true;
  }
  return *t5_outerRadiusMin2S_;
}
const vector<float> &SDL::t3_betaIn() {
  if (not t3_betaIn_isLoaded) {
    if (t3_betaIn_branch != 0) {
      t3_betaIn_branch->GetEntry(index);
    } else {
      printf("branch t3_betaIn_branch does not exist!\n");
      exit(1);
    }
    t3_betaIn_isLoaded = true;
  }
  return *t3_betaIn_;
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
const int &SDL::tc_occupancies() {
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
const vector<int> &SDL::t3_layer1() {
  if (not t3_layer1_isLoaded) {
    if (t3_layer1_branch != 0) {
      t3_layer1_branch->GetEntry(index);
    } else {
      printf("branch t3_layer1_branch does not exist!\n");
      exit(1);
    }
    t3_layer1_isLoaded = true;
  }
  return *t3_layer1_;
}
const vector<float> &SDL::pT4_kZ() {
  if (not pT4_kZ_isLoaded) {
    if (pT4_kZ_branch != 0) {
      pT4_kZ_branch->GetEntry(index);
    } else {
      printf("branch pT4_kZ_branch does not exist!\n");
      exit(1);
    }
    pT4_kZ_isLoaded = true;
  }
  return *pT4_kZ_;
}
const vector<int> &SDL::pT3_hit_idx2() {
  if (not pT3_hit_idx2_isLoaded) {
    if (pT3_hit_idx2_branch != 0) {
      pT3_hit_idx2_branch->GetEntry(index);
    } else {
      printf("branch pT3_hit_idx2_branch does not exist!\n");
      exit(1);
    }
    pT3_hit_idx2_isLoaded = true;
  }
  return *pT3_hit_idx2_;
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
const vector<float> &SDL::t4_zHi() {
  if (not t4_zHi_isLoaded) {
    if (t4_zHi_branch != 0) {
      t4_zHi_branch->GetEntry(index);
    } else {
      printf("branch t4_zHi_branch does not exist!\n");
      exit(1);
    }
    t4_zHi_isLoaded = true;
  }
  return *t4_zHi_;
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
const vector<float> &SDL::t3_kZ() {
  if (not t3_kZ_isLoaded) {
    if (t3_kZ_branch != 0) {
      t3_kZ_branch->GetEntry(index);
    } else {
      printf("branch t3_kZ_branch does not exist!\n");
      exit(1);
    }
    t3_kZ_isLoaded = true;
  }
  return *t3_kZ_;
}
const vector<int> &SDL::t4_moduleType_binary() {
  if (not t4_moduleType_binary_isLoaded) {
    if (t4_moduleType_binary_branch != 0) {
      t4_moduleType_binary_branch->GetEntry(index);
    } else {
      printf("branch t4_moduleType_binary_branch does not exist!\n");
      exit(1);
    }
    t4_moduleType_binary_isLoaded = true;
  }
  return *t4_moduleType_binary_;
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
const vector<int> &SDL::pT3_hit_idx6() {
  if (not pT3_hit_idx6_isLoaded) {
    if (pT3_hit_idx6_branch != 0) {
      pT3_hit_idx6_branch->GetEntry(index);
    } else {
      printf("branch pT3_hit_idx6_branch does not exist!\n");
      exit(1);
    }
    pT3_hit_idx6_isLoaded = true;
  }
  return *pT3_hit_idx6_;
}
const vector<int> &SDL::pT3_pix_idx4() {
  if (not pT3_pix_idx4_isLoaded) {
    if (pT3_pix_idx4_branch != 0) {
      pT3_pix_idx4_branch->GetEntry(index);
    } else {
      printf("branch pT3_pix_idx4_branch does not exist!\n");
      exit(1);
    }
    pT3_pix_idx4_isLoaded = true;
  }
  return *pT3_pix_idx4_;
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
const vector<float> &SDL::t4_sdlCut() {
  if (not t4_sdlCut_isLoaded) {
    if (t4_sdlCut_branch != 0) {
      t4_sdlCut_branch->GetEntry(index);
    } else {
      printf("branch t4_sdlCut_branch does not exist!\n");
      exit(1);
    }
    t4_sdlCut_isLoaded = true;
  }
  return *t4_sdlCut_;
}
const vector<float> &SDL::pT4_rtLo() {
  if (not pT4_rtLo_isLoaded) {
    if (pT4_rtLo_branch != 0) {
      pT4_rtLo_branch->GetEntry(index);
    } else {
      printf("branch pT4_rtLo_branch does not exist!\n");
      exit(1);
    }
    pT4_rtLo_isLoaded = true;
  }
  return *pT4_rtLo_;
}
const vector<float> &SDL::t5_innerRadiusMax2S() {
  if (not t5_innerRadiusMax2S_isLoaded) {
    if (t5_innerRadiusMax2S_branch != 0) {
      t5_innerRadiusMax2S_branch->GetEntry(index);
    } else {
      printf("branch t5_innerRadiusMax2S_branch does not exist!\n");
      exit(1);
    }
    t5_innerRadiusMax2S_isLoaded = true;
  }
  return *t5_innerRadiusMax2S_;
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
const vector<float> &SDL::t3_zLo() {
  if (not t3_zLo_isLoaded) {
    if (t3_zLo_branch != 0) {
      t3_zLo_branch->GetEntry(index);
    } else {
      printf("branch t3_zLo_branch does not exist!\n");
      exit(1);
    }
    t3_zLo_isLoaded = true;
  }
  return *t3_zLo_;
}
const vector<float> &SDL::pT4_deltaBetaCut() {
  if (not pT4_deltaBetaCut_isLoaded) {
    if (pT4_deltaBetaCut_branch != 0) {
      pT4_deltaBetaCut_branch->GetEntry(index);
    } else {
      printf("branch pT4_deltaBetaCut_branch does not exist!\n");
      exit(1);
    }
    pT4_deltaBetaCut_isLoaded = true;
  }
  return *pT4_deltaBetaCut_;
}
const vector<float> &SDL::t4_rtHi() {
  if (not t4_rtHi_isLoaded) {
    if (t4_rtHi_branch != 0) {
      t4_rtHi_branch->GetEntry(index);
    } else {
      printf("branch t4_rtHi_branch does not exist!\n");
      exit(1);
    }
    t4_rtHi_isLoaded = true;
  }
  return *t4_rtHi_;
}
const vector<int> &SDL::t3_layer2() {
  if (not t3_layer2_isLoaded) {
    if (t3_layer2_branch != 0) {
      t3_layer2_branch->GetEntry(index);
    } else {
      printf("branch t3_layer2_branch does not exist!\n");
      exit(1);
    }
    t3_layer2_isLoaded = true;
  }
  return *t3_layer2_;
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
const vector<vector<int> > &SDL::sim_pT5_types() {
  if (not sim_pT5_types_isLoaded) {
    if (sim_pT5_types_branch != 0) {
      sim_pT5_types_branch->GetEntry(index);
    } else {
      printf("branch sim_pT5_types_branch does not exist!\n");
      exit(1);
    }
    sim_pT5_types_isLoaded = true;
  }
  return *sim_pT5_types_;
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
  const int &pT5_occupancies() { return sdl.pT5_occupancies(); }
  const vector<float> &t3_phi() { return sdl.t3_phi(); }
  const vector<float> &t4_zLoPointed() { return sdl.t4_zLoPointed(); }
  const vector<float> &t4_kZ() { return sdl.t4_kZ(); }
  const vector<int> &t3_isDuplicate() { return sdl.t3_isDuplicate(); }
  const vector<int> &sim_event() { return sdl.sim_event(); }
  const vector<float> &t4_zOut() { return sdl.t4_zOut(); }
  const vector<int> &sim_q() { return sdl.sim_q(); }
  const vector<float> &t3_rtHi() { return sdl.t3_rtHi(); }
  const vector<float> &sim_eta() { return sdl.sim_eta(); }
  const vector<float> &pT4_betaOut() { return sdl.pT4_betaOut(); }
  const vector<float> &pT4_zLo() { return sdl.pT4_zLo(); }
  const vector<float> &t5_eta() { return sdl.t5_eta(); }
  const vector<int> &sim_denom() { return sdl.sim_denom(); }
  const vector<float> &pLS_phi() { return sdl.pLS_phi(); }
  const vector<int> &t3_layer3() { return sdl.t3_layer3(); }
  const vector<int> &pT3_isDuplicate() { return sdl.pT3_isDuplicate(); }
  const vector<int> &tc_isDuplicate() { return sdl.tc_isDuplicate(); }
  const vector<float> &t4_deltaPhiPos() { return sdl.t4_deltaPhiPos(); }
  const vector<float> &pT5_rzChiSquared() { return sdl.pT5_rzChiSquared(); }
  const vector<float> &pT5_eta() { return sdl.pT5_eta(); }
  const vector<int> &sim_pT3_matched() { return sdl.sim_pT3_matched(); }
  const vector<vector<float> > &pT3_matched_pt() { return sdl.pT3_matched_pt(); }
  const vector<float> &t3_zHiPointed() { return sdl.t3_zHiPointed(); }
  const vector<float> &t4_betaOut() { return sdl.t4_betaOut(); }
  const vector<int> &t4_isDuplicate() { return sdl.t4_isDuplicate(); }
  const vector<float> &t4_betaOutCut() { return sdl.t4_betaOutCut(); }
  const vector<float> &t3_betaOut() { return sdl.t3_betaOut(); }
  const vector<float> &t3_sdlCut() { return sdl.t3_sdlCut(); }
  const vector<float> &pT4_rtOut() { return sdl.pT4_rtOut(); }
  const vector<float> &t3_betaInCut() { return sdl.t3_betaInCut(); }
  const vector<float> &pT4_betaOutCut() { return sdl.pT4_betaOutCut(); }
  const vector<float> &pT4_betaInCut() { return sdl.pT4_betaInCut(); }
  const vector<float> &pT3_pixelRadius() { return sdl.pT3_pixelRadius(); }
  const vector<float> &sim_pt() { return sdl.sim_pt(); }
  const vector<float> &pT5_matched_pt() { return sdl.pT5_matched_pt(); }
  const vector<float> &pT4_deltaPhi() { return sdl.pT4_deltaPhi(); }
  const vector<float> &t3_zLoPointed() { return sdl.t3_zLoPointed(); }
  const vector<float> &pLS_eta() { return sdl.pLS_eta(); }
  const vector<float> &t3_deltaBetaCut() { return sdl.t3_deltaBetaCut(); }
  const vector<int> &t3_moduleType_binary() { return sdl.t3_moduleType_binary(); }
  const vector<int> &sim_pdgId() { return sdl.sim_pdgId(); }
  const vector<float> &t3_eta() { return sdl.t3_eta(); }
  const vector<float> &t5_bridgeRadiusMax2S() { return sdl.t5_bridgeRadiusMax2S(); }
  const vector<float> &t5_outerRadiusMax2S() { return sdl.t5_outerRadiusMax2S(); }
  const vector<int> &t4_occupancies() { return sdl.t4_occupancies(); }
  const vector<int> &t5_layer_binary() { return sdl.t5_layer_binary(); }
  const vector<vector<int> > &sim_tcIdx() { return sdl.sim_tcIdx(); }
  const vector<int> &pT4_layer_binary() { return sdl.pT4_layer_binary(); }
  const vector<int> &pT3_layer_binary() { return sdl.pT3_layer_binary(); }
  const vector<int> &pT3_pix_idx3() { return sdl.pT3_pix_idx3(); }
  const vector<int> &pT3_pix_idx2() { return sdl.pT3_pix_idx2(); }
  const vector<int> &pT3_pix_idx1() { return sdl.pT3_pix_idx1(); }
  const vector<float> &t5_bridgeRadiusMax() { return sdl.t5_bridgeRadiusMax(); }
  const vector<float> &t5_bridgeRadiusMin2S() { return sdl.t5_bridgeRadiusMin2S(); }
  const vector<int> &module_subdets() { return sdl.module_subdets(); }
  const vector<float> &pT3_tripletRadius() { return sdl.pT3_tripletRadius(); }
  const vector<float> &pT4_zLoPointed() { return sdl.pT4_zLoPointed(); }
  const vector<int> &t3_hit_idx4() { return sdl.t3_hit_idx4(); }
  const vector<int> &t3_hit_idx5() { return sdl.t3_hit_idx5(); }
  const vector<int> &t3_hit_idx6() { return sdl.t3_hit_idx6(); }
  const vector<float> &t3_rtOut() { return sdl.t3_rtOut(); }
  const vector<int> &t3_hit_idx1() { return sdl.t3_hit_idx1(); }
  const vector<int> &t3_hit_idx2() { return sdl.t3_hit_idx2(); }
  const vector<int> &t3_hit_idx3() { return sdl.t3_hit_idx3(); }
  const vector<int> &t3_isFake() { return sdl.t3_isFake(); }
  const vector<int> &t5_isFake() { return sdl.t5_isFake(); }
  const vector<float> &t5_bridgeRadiusMin() { return sdl.t5_bridgeRadiusMin(); }
  const vector<float> &t4_zLo() { return sdl.t4_zLo(); }
  const vector<int> &md_occupancies() { return sdl.md_occupancies(); }
  const vector<int> &t3_layer_binary() { return sdl.t3_layer_binary(); }
  const vector<int> &t4_layer_binary() { return sdl.t4_layer_binary(); }
  const vector<vector<int> > &sim_pT3_types() { return sdl.sim_pT3_types(); }
  const vector<float> &t4_phi() { return sdl.t4_phi(); }
  const vector<float> &t5_phi() { return sdl.t5_phi(); }
  const vector<int> &t4_isFake() { return sdl.t4_isFake(); }
  const vector<float> &t4_deltaPhi() { return sdl.t4_deltaPhi(); }
  const vector<float> &t4_rtLo() { return sdl.t4_rtLo(); }
  const vector<float> &t5_outerRadius() { return sdl.t5_outerRadius(); }
  const vector<float> &pT5_phi() { return sdl.pT5_phi(); }
  const vector<float> &t4_betaIn() { return sdl.t4_betaIn(); }
  const vector<int> &tc_isFake() { return sdl.tc_isFake(); }
  const vector<float> &t3_zOut() { return sdl.t3_zOut(); }
  const vector<float> &t5_outerRadiusMax() { return sdl.t5_outerRadiusMax(); }
  const vector<int> &pT3_isFake() { return sdl.pT3_isFake(); }
  const vector<vector<int> > &sim_pLS_types() { return sdl.sim_pLS_types(); }
  const vector<float> &t3_deltaBeta() { return sdl.t3_deltaBeta(); }
  const vector<float> &sim_pca_dxy() { return sdl.sim_pca_dxy(); }
  const vector<float> &t5_outerRadiusMin() { return sdl.t5_outerRadiusMin(); }
  const vector<float> &pT4_phi() { return sdl.pT4_phi(); }
  const vector<float> &t3_rtLo() { return sdl.t3_rtLo(); }
  const vector<float> &t3_betaOutCut() { return sdl.t3_betaOutCut(); }
  const vector<int> &pT5_isDuplicate() { return sdl.pT5_isDuplicate(); }
  const vector<float> &pT4_zHi() { return sdl.pT4_zHi(); }
  const vector<int> &t5_moduleType_binary() { return sdl.t5_moduleType_binary(); }
  const vector<float> &t3_residual() { return sdl.t3_residual(); }
  const vector<int> &t3_occupancies() { return sdl.t3_occupancies(); }
  const vector<vector<int> > &sim_pT4_types() { return sdl.sim_pT4_types(); }
  const vector<float> &t4_deltaBetaCut() { return sdl.t4_deltaBetaCut(); }
  const vector<float> &t5_pt() { return sdl.t5_pt(); }
  const vector<float> &sim_len() { return sdl.sim_len(); }
  const vector<float> &sim_lengap() { return sdl.sim_lengap(); }
  const vector<float> &sim_hits() { return sdl.sim_hits(); }
  const vector<float> &simvtx_x() { return sdl.simvtx_x(); }
  const vector<float> &simvtx_y() { return sdl.simvtx_y(); }
  const vector<float> &simvtx_z() { return sdl.simvtx_z(); }
  const vector<int> &sim_T4_matched() { return sdl.sim_T4_matched(); }
  const vector<float> &t4_rtOut() { return sdl.t4_rtOut(); }
  const vector<float> &pT3_pt() { return sdl.pT3_pt(); }
  const vector<float> &tc_pt() { return sdl.tc_pt(); }
  const vector<float> &pT3_pixelRadiusError() { return sdl.pT3_pixelRadiusError(); }
  const vector<int> &pT5_isFake() { return sdl.pT5_isFake(); }
  const vector<float> &pT5_pt() { return sdl.pT5_pt(); }
  const vector<float> &pT4_deltaBeta() { return sdl.pT4_deltaBeta(); }
  const vector<float> &t5_innerRadiusMax() { return sdl.t5_innerRadiusMax(); }
  const vector<float> &sim_phi() { return sdl.sim_phi(); }
  const vector<float> &t4_betaInCut() { return sdl.t4_betaInCut(); }
  const vector<float> &t5_innerRadiusMin() { return sdl.t5_innerRadiusMin(); }
  const vector<float> &pT4_sdlCut() { return sdl.pT4_sdlCut(); }
  const vector<int> &pT3_hit_idx3() { return sdl.pT3_hit_idx3(); }
  const vector<float> &pT4_zHiPointed() { return sdl.pT4_zHiPointed(); }
  const vector<int> &pT3_hit_idx1() { return sdl.pT3_hit_idx1(); }
  const vector<float> &sim_pca_dz() { return sdl.sim_pca_dz(); }
  const vector<float> &t4_deltaBeta() { return sdl.t4_deltaBeta(); }
  const vector<int> &pT3_hit_idx5() { return sdl.pT3_hit_idx5(); }
  const vector<int> &pT3_hit_idx4() { return sdl.pT3_hit_idx4(); }
  const vector<int> &pT5_layer_binary() { return sdl.pT5_layer_binary(); }
  const vector<float> &t5_bridgeRadius() { return sdl.t5_bridgeRadius(); }
  const vector<int> &sim_pLS_matched() { return sdl.sim_pLS_matched(); }
  const vector<int> &pT4_isFake() { return sdl.pT4_isFake(); }
  const vector<int> &sim_T3_matched() { return sdl.sim_T3_matched(); }
  const vector<float> &t3_deltaPhiPos() { return sdl.t3_deltaPhiPos(); }
  const vector<float> &pT3_phi() { return sdl.pT3_phi(); }
  const vector<vector<float> > &t5_matched_pt() { return sdl.t5_matched_pt(); }
  const vector<float> &pT3_eta() { return sdl.pT3_eta(); }
  const vector<float> &t4_eta() { return sdl.t4_eta(); }
  const vector<float> &t3_deltaPhi() { return sdl.t3_deltaPhi(); }
  const vector<int> &pLS_isFake() { return sdl.pLS_isFake(); }
  const vector<float> &pT4_betaIn() { return sdl.pT4_betaIn(); }
  const vector<int> &sim_bunchCrossing() { return sdl.sim_bunchCrossing(); }
  const vector<float> &pT4_zOut() { return sdl.pT4_zOut(); }
  const vector<float> &pT4_deltaPhiPos() { return sdl.pT4_deltaPhiPos(); }
  const vector<int> &sim_parentVtxIdx() { return sdl.sim_parentVtxIdx(); }
  const vector<float> &t3_zHi() { return sdl.t3_zHi(); }
  const vector<int> &sim_pT4_matched() { return sdl.sim_pT4_matched(); }
  const vector<float> &t5_innerRadiusMin2S() { return sdl.t5_innerRadiusMin2S(); }
  const vector<float> &tc_eta() { return sdl.tc_eta(); }
  const vector<float> &tc_phi() { return sdl.tc_phi(); }
  const vector<int> &sim_T5_matched() { return sdl.sim_T5_matched(); }
  const vector<vector<int> > &sim_T5_types() { return sdl.sim_T5_types(); }
  const vector<int> &t5_isDuplicate() { return sdl.t5_isDuplicate(); }
  const vector<float> &t4_zHiPointed() { return sdl.t4_zHiPointed(); }
  const vector<float> &pT4_rtHi() { return sdl.pT4_rtHi(); }
  const vector<float> &t5_outerRadiusMin2S() { return sdl.t5_outerRadiusMin2S(); }
  const vector<float> &t3_betaIn() { return sdl.t3_betaIn(); }
  const int &pT3_occupancies() { return sdl.pT3_occupancies(); }
  const int &tc_occupancies() { return sdl.tc_occupancies(); }
  const vector<float> &t5_innerRadius() { return sdl.t5_innerRadius(); }
  const vector<int> &sim_TC_matched() { return sdl.sim_TC_matched(); }
  const vector<int> &pLS_isDuplicate() { return sdl.pLS_isDuplicate(); }
  const vector<int> &t5_occupancies() { return sdl.t5_occupancies(); }
  const vector<int> &t3_layer1() { return sdl.t3_layer1(); }
  const vector<float> &pT4_kZ() { return sdl.pT4_kZ(); }
  const vector<int> &pT3_hit_idx2() { return sdl.pT3_hit_idx2(); }
  const vector<float> &pLS_pt() { return sdl.pLS_pt(); }
  const vector<vector<int> > &sim_T4_types() { return sdl.sim_T4_types(); }
  const vector<int> &pT4_isDuplicate() { return sdl.pT4_isDuplicate(); }
  const vector<float> &t4_pt() { return sdl.t4_pt(); }
  const vector<float> &t4_zHi() { return sdl.t4_zHi(); }
  const vector<vector<int> > &sim_TC_types() { return sdl.sim_TC_types(); }
  const vector<float> &t3_kZ() { return sdl.t3_kZ(); }
  const vector<int> &t4_moduleType_binary() { return sdl.t4_moduleType_binary(); }
  const vector<int> &sg_occupancies() { return sdl.sg_occupancies(); }
  const vector<float> &pT4_pt() { return sdl.pT4_pt(); }
  const vector<int> &pT3_hit_idx6() { return sdl.pT3_hit_idx6(); }
  const vector<int> &pT3_pix_idx4() { return sdl.pT3_pix_idx4(); }
  const vector<float> &sim_vx() { return sdl.sim_vx(); }
  const vector<float> &sim_vy() { return sdl.sim_vy(); }
  const vector<float> &sim_vz() { return sdl.sim_vz(); }
  const vector<float> &t4_sdlCut() { return sdl.t4_sdlCut(); }
  const vector<float> &pT4_rtLo() { return sdl.pT4_rtLo(); }
  const vector<float> &t5_innerRadiusMax2S() { return sdl.t5_innerRadiusMax2S(); }
  const vector<float> &t3_pt() { return sdl.t3_pt(); }
  const vector<int> &module_rings() { return sdl.module_rings(); }
  const vector<float> &t3_zLo() { return sdl.t3_zLo(); }
  const vector<float> &pT4_deltaBetaCut() { return sdl.pT4_deltaBetaCut(); }
  const vector<float> &t4_rtHi() { return sdl.t4_rtHi(); }
  const vector<int> &t3_layer2() { return sdl.t3_layer2(); }
  const vector<vector<int> > &sim_T3_types() { return sdl.sim_T3_types(); }
  const vector<vector<int> > &sim_pT5_types() { return sdl.sim_pT5_types(); }
  const vector<int> &sim_pT5_matched() { return sdl.sim_pT5_matched(); }
  const vector<int> &module_layers() { return sdl.module_layers(); }
  const vector<float> &pT4_eta() { return sdl.pT4_eta(); }
}
