#include "SDL.h"
SDL sdl;

void SDL::Init(TTree *tree) {
  tree->SetMakeClass(1);
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
  sim_vtxperp_branch = 0;
  if (tree->GetBranch("sim_vtxperp") != 0) {
    sim_vtxperp_branch = tree->GetBranch("sim_vtxperp");
    if (sim_vtxperp_branch) { sim_vtxperp_branch->SetAddress(&sim_vtxperp_); }
  }
  sim_trkNtupIdx_branch = 0;
  if (tree->GetBranch("sim_trkNtupIdx") != 0) {
    sim_trkNtupIdx_branch = tree->GetBranch("sim_trkNtupIdx");
    if (sim_trkNtupIdx_branch) { sim_trkNtupIdx_branch->SetAddress(&sim_trkNtupIdx_); }
  }
  sim_tcIdx_branch = 0;
  if (tree->GetBranch("sim_tcIdx") != 0) {
    sim_tcIdx_branch = tree->GetBranch("sim_tcIdx");
    if (sim_tcIdx_branch) { sim_tcIdx_branch->SetAddress(&sim_tcIdx_); }
  }
  sim_tcIdxAll_branch = 0;
  if (tree->GetBranch("sim_tcIdxAll") != 0) {
    sim_tcIdxAll_branch = tree->GetBranch("sim_tcIdxAll");
    if (sim_tcIdxAll_branch) { sim_tcIdxAll_branch->SetAddress(&sim_tcIdxAll_); }
  }
  sim_tcIdxAllFrac_branch = 0;
  if (tree->GetBranch("sim_tcIdxAllFrac") != 0) {
    sim_tcIdxAllFrac_branch = tree->GetBranch("sim_tcIdxAllFrac");
    if (sim_tcIdxAllFrac_branch) { sim_tcIdxAllFrac_branch->SetAddress(&sim_tcIdxAllFrac_); }
  }
  sim_mdIdxAll_branch = 0;
  if (tree->GetBranch("sim_mdIdxAll") != 0) {
    sim_mdIdxAll_branch = tree->GetBranch("sim_mdIdxAll");
    if (sim_mdIdxAll_branch) { sim_mdIdxAll_branch->SetAddress(&sim_mdIdxAll_); }
  }
  sim_mdIdxAllFrac_branch = 0;
  if (tree->GetBranch("sim_mdIdxAllFrac") != 0) {
    sim_mdIdxAllFrac_branch = tree->GetBranch("sim_mdIdxAllFrac");
    if (sim_mdIdxAllFrac_branch) { sim_mdIdxAllFrac_branch->SetAddress(&sim_mdIdxAllFrac_); }
  }
  sim_lsIdxAll_branch = 0;
  if (tree->GetBranch("sim_lsIdxAll") != 0) {
    sim_lsIdxAll_branch = tree->GetBranch("sim_lsIdxAll");
    if (sim_lsIdxAll_branch) { sim_lsIdxAll_branch->SetAddress(&sim_lsIdxAll_); }
  }
  sim_lsIdxAllFrac_branch = 0;
  if (tree->GetBranch("sim_lsIdxAllFrac") != 0) {
    sim_lsIdxAllFrac_branch = tree->GetBranch("sim_lsIdxAllFrac");
    if (sim_lsIdxAllFrac_branch) { sim_lsIdxAllFrac_branch->SetAddress(&sim_lsIdxAllFrac_); }
  }
  sim_t3IdxAll_branch = 0;
  if (tree->GetBranch("sim_t3IdxAll") != 0) {
    sim_t3IdxAll_branch = tree->GetBranch("sim_t3IdxAll");
    if (sim_t3IdxAll_branch) { sim_t3IdxAll_branch->SetAddress(&sim_t3IdxAll_); }
  }
  sim_t3IdxAllFrac_branch = 0;
  if (tree->GetBranch("sim_t3IdxAllFrac") != 0) {
    sim_t3IdxAllFrac_branch = tree->GetBranch("sim_t3IdxAllFrac");
    if (sim_t3IdxAllFrac_branch) { sim_t3IdxAllFrac_branch->SetAddress(&sim_t3IdxAllFrac_); }
  }
  sim_t5IdxAll_branch = 0;
  if (tree->GetBranch("sim_t5IdxAll") != 0) {
    sim_t5IdxAll_branch = tree->GetBranch("sim_t5IdxAll");
    if (sim_t5IdxAll_branch) { sim_t5IdxAll_branch->SetAddress(&sim_t5IdxAll_); }
  }
  sim_t5IdxAllFrac_branch = 0;
  if (tree->GetBranch("sim_t5IdxAllFrac") != 0) {
    sim_t5IdxAllFrac_branch = tree->GetBranch("sim_t5IdxAllFrac");
    if (sim_t5IdxAllFrac_branch) { sim_t5IdxAllFrac_branch->SetAddress(&sim_t5IdxAllFrac_); }
  }
  sim_plsIdxAll_branch = 0;
  if (tree->GetBranch("sim_plsIdxAll") != 0) {
    sim_plsIdxAll_branch = tree->GetBranch("sim_plsIdxAll");
    if (sim_plsIdxAll_branch) { sim_plsIdxAll_branch->SetAddress(&sim_plsIdxAll_); }
  }
  sim_plsIdxAllFrac_branch = 0;
  if (tree->GetBranch("sim_plsIdxAllFrac") != 0) {
    sim_plsIdxAllFrac_branch = tree->GetBranch("sim_plsIdxAllFrac");
    if (sim_plsIdxAllFrac_branch) { sim_plsIdxAllFrac_branch->SetAddress(&sim_plsIdxAllFrac_); }
  }
  sim_pt3IdxAll_branch = 0;
  if (tree->GetBranch("sim_pt3IdxAll") != 0) {
    sim_pt3IdxAll_branch = tree->GetBranch("sim_pt3IdxAll");
    if (sim_pt3IdxAll_branch) { sim_pt3IdxAll_branch->SetAddress(&sim_pt3IdxAll_); }
  }
  sim_pt3IdxAllFrac_branch = 0;
  if (tree->GetBranch("sim_pt3IdxAllFrac") != 0) {
    sim_pt3IdxAllFrac_branch = tree->GetBranch("sim_pt3IdxAllFrac");
    if (sim_pt3IdxAllFrac_branch) { sim_pt3IdxAllFrac_branch->SetAddress(&sim_pt3IdxAllFrac_); }
  }
  sim_pt5IdxAll_branch = 0;
  if (tree->GetBranch("sim_pt5IdxAll") != 0) {
    sim_pt5IdxAll_branch = tree->GetBranch("sim_pt5IdxAll");
    if (sim_pt5IdxAll_branch) { sim_pt5IdxAll_branch->SetAddress(&sim_pt5IdxAll_); }
  }
  sim_pt5IdxAllFrac_branch = 0;
  if (tree->GetBranch("sim_pt5IdxAllFrac") != 0) {
    sim_pt5IdxAllFrac_branch = tree->GetBranch("sim_pt5IdxAllFrac");
    if (sim_pt5IdxAllFrac_branch) { sim_pt5IdxAllFrac_branch->SetAddress(&sim_pt5IdxAllFrac_); }
  }
  sim_simHitX_branch = 0;
  if (tree->GetBranch("sim_simHitX") != 0) {
    sim_simHitX_branch = tree->GetBranch("sim_simHitX");
    if (sim_simHitX_branch) { sim_simHitX_branch->SetAddress(&sim_simHitX_); }
  }
  sim_simHitY_branch = 0;
  if (tree->GetBranch("sim_simHitY") != 0) {
    sim_simHitY_branch = tree->GetBranch("sim_simHitY");
    if (sim_simHitY_branch) { sim_simHitY_branch->SetAddress(&sim_simHitY_); }
  }
  sim_simHitZ_branch = 0;
  if (tree->GetBranch("sim_simHitZ") != 0) {
    sim_simHitZ_branch = tree->GetBranch("sim_simHitZ");
    if (sim_simHitZ_branch) { sim_simHitZ_branch->SetAddress(&sim_simHitZ_); }
  }
  sim_simHitDetId_branch = 0;
  if (tree->GetBranch("sim_simHitDetId") != 0) {
    sim_simHitDetId_branch = tree->GetBranch("sim_simHitDetId");
    if (sim_simHitDetId_branch) { sim_simHitDetId_branch->SetAddress(&sim_simHitDetId_); }
  }
  sim_simHitLayer_branch = 0;
  if (tree->GetBranch("sim_simHitLayer") != 0) {
    sim_simHitLayer_branch = tree->GetBranch("sim_simHitLayer");
    if (sim_simHitLayer_branch) { sim_simHitLayer_branch->SetAddress(&sim_simHitLayer_); }
  }
  sim_simHitDistxyHelix_branch = 0;
  if (tree->GetBranch("sim_simHitDistxyHelix") != 0) {
    sim_simHitDistxyHelix_branch = tree->GetBranch("sim_simHitDistxyHelix");
    if (sim_simHitDistxyHelix_branch) { sim_simHitDistxyHelix_branch->SetAddress(&sim_simHitDistxyHelix_); }
  }
  sim_simHitLayerMinDistxyHelix_branch = 0;
  if (tree->GetBranch("sim_simHitLayerMinDistxyHelix") != 0) {
    sim_simHitLayerMinDistxyHelix_branch = tree->GetBranch("sim_simHitLayerMinDistxyHelix");
    if (sim_simHitLayerMinDistxyHelix_branch) { sim_simHitLayerMinDistxyHelix_branch->SetAddress(&sim_simHitLayerMinDistxyHelix_); }
  }
  sim_recoHitX_branch = 0;
  if (tree->GetBranch("sim_recoHitX") != 0) {
    sim_recoHitX_branch = tree->GetBranch("sim_recoHitX");
    if (sim_recoHitX_branch) { sim_recoHitX_branch->SetAddress(&sim_recoHitX_); }
  }
  sim_recoHitY_branch = 0;
  if (tree->GetBranch("sim_recoHitY") != 0) {
    sim_recoHitY_branch = tree->GetBranch("sim_recoHitY");
    if (sim_recoHitY_branch) { sim_recoHitY_branch->SetAddress(&sim_recoHitY_); }
  }
  sim_recoHitZ_branch = 0;
  if (tree->GetBranch("sim_recoHitZ") != 0) {
    sim_recoHitZ_branch = tree->GetBranch("sim_recoHitZ");
    if (sim_recoHitZ_branch) { sim_recoHitZ_branch->SetAddress(&sim_recoHitZ_); }
  }
  sim_recoHitDetId_branch = 0;
  if (tree->GetBranch("sim_recoHitDetId") != 0) {
    sim_recoHitDetId_branch = tree->GetBranch("sim_recoHitDetId");
    if (sim_recoHitDetId_branch) { sim_recoHitDetId_branch->SetAddress(&sim_recoHitDetId_); }
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
  tc_pt5Idx_branch = 0;
  if (tree->GetBranch("tc_pt5Idx") != 0) {
    tc_pt5Idx_branch = tree->GetBranch("tc_pt5Idx");
    if (tc_pt5Idx_branch) { tc_pt5Idx_branch->SetAddress(&tc_pt5Idx_); }
  }
  tc_pt3Idx_branch = 0;
  if (tree->GetBranch("tc_pt3Idx") != 0) {
    tc_pt3Idx_branch = tree->GetBranch("tc_pt3Idx");
    if (tc_pt3Idx_branch) { tc_pt3Idx_branch->SetAddress(&tc_pt3Idx_); }
  }
  tc_t5Idx_branch = 0;
  if (tree->GetBranch("tc_t5Idx") != 0) {
    tc_t5Idx_branch = tree->GetBranch("tc_t5Idx");
    if (tc_t5Idx_branch) { tc_t5Idx_branch->SetAddress(&tc_t5Idx_); }
  }
  tc_plsIdx_branch = 0;
  if (tree->GetBranch("tc_plsIdx") != 0) {
    tc_plsIdx_branch = tree->GetBranch("tc_plsIdx");
    if (tc_plsIdx_branch) { tc_plsIdx_branch->SetAddress(&tc_plsIdx_); }
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
  tc_simIdx_branch = 0;
  if (tree->GetBranch("tc_simIdx") != 0) {
    tc_simIdx_branch = tree->GetBranch("tc_simIdx");
    if (tc_simIdx_branch) { tc_simIdx_branch->SetAddress(&tc_simIdx_); }
  }
  tc_simIdxAll_branch = 0;
  if (tree->GetBranch("tc_simIdxAll") != 0) {
    tc_simIdxAll_branch = tree->GetBranch("tc_simIdxAll");
    if (tc_simIdxAll_branch) { tc_simIdxAll_branch->SetAddress(&tc_simIdxAll_); }
  }
  tc_simIdxAllFrac_branch = 0;
  if (tree->GetBranch("tc_simIdxAllFrac") != 0) {
    tc_simIdxAllFrac_branch = tree->GetBranch("tc_simIdxAllFrac");
    if (tc_simIdxAllFrac_branch) { tc_simIdxAllFrac_branch->SetAddress(&tc_simIdxAllFrac_); }
  }
  md_pt_branch = 0;
  if (tree->GetBranch("md_pt") != 0) {
    md_pt_branch = tree->GetBranch("md_pt");
    if (md_pt_branch) { md_pt_branch->SetAddress(&md_pt_); }
  }
  md_eta_branch = 0;
  if (tree->GetBranch("md_eta") != 0) {
    md_eta_branch = tree->GetBranch("md_eta");
    if (md_eta_branch) { md_eta_branch->SetAddress(&md_eta_); }
  }
  md_phi_branch = 0;
  if (tree->GetBranch("md_phi") != 0) {
    md_phi_branch = tree->GetBranch("md_phi");
    if (md_phi_branch) { md_phi_branch->SetAddress(&md_phi_); }
  }
  
#ifdef OUTPUT_MD_CUTS
  md_dphi_branch = 0;
  if (tree->GetBranch("md_dphi") != 0) {
    md_dphi_branch = tree->GetBranch("md_dphi");
    if (md_dphi_branch) { md_dphi_branch->SetAddress(&md_dphi_); }
  }
  md_dphichange_branch = 0;
  if (tree->GetBranch("md_dphichange") != 0) {
    md_dphichange_branch = tree->GetBranch("md_dphichange");
    if (md_dphichange_branch) { md_dphichange_branch->SetAddress(&md_dphichange_); }
  }
  md_dz_branch = 0;
  if (tree->GetBranch("md_dz") != 0) {
    md_dz_branch = tree->GetBranch("md_dz");
    if (md_dz_branch) { md_dz_branch->SetAddress(&md_dz_); }
  }
#endif

  md_anchor_x_branch = 0;
  if (tree->GetBranch("md_anchor_x") != 0) {
    md_anchor_x_branch = tree->GetBranch("md_anchor_x");
    if (md_anchor_x_branch) { md_anchor_x_branch->SetAddress(&md_anchor_x_); }
  }
  md_anchor_y_branch = 0;
  if (tree->GetBranch("md_anchor_y") != 0) {
    md_anchor_y_branch = tree->GetBranch("md_anchor_y");
    if (md_anchor_y_branch) { md_anchor_y_branch->SetAddress(&md_anchor_y_); }
  }
  md_anchor_z_branch = 0;
  if (tree->GetBranch("md_anchor_z") != 0) {
    md_anchor_z_branch = tree->GetBranch("md_anchor_z");
    if (md_anchor_z_branch) { md_anchor_z_branch->SetAddress(&md_anchor_z_); }
  }
  md_other_x_branch = 0;
  if (tree->GetBranch("md_other_x") != 0) {
    md_other_x_branch = tree->GetBranch("md_other_x");
    if (md_other_x_branch) { md_other_x_branch->SetAddress(&md_other_x_); }
  }
  md_other_y_branch = 0;
  if (tree->GetBranch("md_other_y") != 0) {
    md_other_y_branch = tree->GetBranch("md_other_y");
    if (md_other_y_branch) { md_other_y_branch->SetAddress(&md_other_y_); }
  }
  md_other_z_branch = 0;
  if (tree->GetBranch("md_other_z") != 0) {
    md_other_z_branch = tree->GetBranch("md_other_z");
    if (md_other_z_branch) { md_other_z_branch->SetAddress(&md_other_z_); }
  }
  md_type_branch = 0;
  if (tree->GetBranch("md_type") != 0) {
    md_type_branch = tree->GetBranch("md_type");
    if (md_type_branch) { md_type_branch->SetAddress(&md_type_); }
  }
  md_layer_branch = 0;
  if (tree->GetBranch("md_layer") != 0) {
    md_layer_branch = tree->GetBranch("md_layer");
    if (md_layer_branch) { md_layer_branch->SetAddress(&md_layer_); }
  }
  md_detId_branch = 0;
  if (tree->GetBranch("md_detId") != 0) {
    md_detId_branch = tree->GetBranch("md_detId");
    if (md_detId_branch) { md_detId_branch->SetAddress(&md_detId_); }
  }
  md_isFake_branch = 0;
  if (tree->GetBranch("md_isFake") != 0) {
    md_isFake_branch = tree->GetBranch("md_isFake");
    if (md_isFake_branch) { md_isFake_branch->SetAddress(&md_isFake_); }
  }
  md_simIdx_branch = 0;
  if (tree->GetBranch("md_simIdx") != 0) {
    md_simIdx_branch = tree->GetBranch("md_simIdx");
    if (md_simIdx_branch) { md_simIdx_branch->SetAddress(&md_simIdx_); }
  }
  md_simIdxAll_branch = 0;
  if (tree->GetBranch("md_simIdxAll") != 0) {
    md_simIdxAll_branch = tree->GetBranch("md_simIdxAll");
    if (md_simIdxAll_branch) { md_simIdxAll_branch->SetAddress(&md_simIdxAll_); }
  }
  md_simIdxAllFrac_branch = 0;
  if (tree->GetBranch("md_simIdxAllFrac") != 0) {
    md_simIdxAllFrac_branch = tree->GetBranch("md_simIdxAllFrac");
    if (md_simIdxAllFrac_branch) { md_simIdxAllFrac_branch->SetAddress(&md_simIdxAllFrac_); }
  }
  ls_pt_branch = 0;
  if (tree->GetBranch("ls_pt") != 0) {
    ls_pt_branch = tree->GetBranch("ls_pt");
    if (ls_pt_branch) { ls_pt_branch->SetAddress(&ls_pt_); }
  }
  ls_eta_branch = 0;
  if (tree->GetBranch("ls_eta") != 0) {
    ls_eta_branch = tree->GetBranch("ls_eta");
    if (ls_eta_branch) { ls_eta_branch->SetAddress(&ls_eta_); }
  }
  ls_phi_branch = 0;
  if (tree->GetBranch("ls_phi") != 0) {
    ls_phi_branch = tree->GetBranch("ls_phi");
    if (ls_phi_branch) { ls_phi_branch->SetAddress(&ls_phi_); }
  }
  ls_mdIdx0_branch = 0;
  if (tree->GetBranch("ls_mdIdx0") != 0) {
    ls_mdIdx0_branch = tree->GetBranch("ls_mdIdx0");
    if (ls_mdIdx0_branch) { ls_mdIdx0_branch->SetAddress(&ls_mdIdx0_); }
  }
  ls_mdIdx1_branch = 0;
  if (tree->GetBranch("ls_mdIdx1") != 0) {
    ls_mdIdx1_branch = tree->GetBranch("ls_mdIdx1");
    if (ls_mdIdx1_branch) { ls_mdIdx1_branch->SetAddress(&ls_mdIdx1_); }
  }
  ls_isFake_branch = 0;
  if (tree->GetBranch("ls_isFake") != 0) {
    ls_isFake_branch = tree->GetBranch("ls_isFake");
    if (ls_isFake_branch) { ls_isFake_branch->SetAddress(&ls_isFake_); }
  }
  ls_simIdx_branch = 0;
  if (tree->GetBranch("ls_simIdx") != 0) {
    ls_simIdx_branch = tree->GetBranch("ls_simIdx");
    if (ls_simIdx_branch) { ls_simIdx_branch->SetAddress(&ls_simIdx_); }
  }

#ifdef OUTPUT_LS_CUTS 
  ls_zLos_branch = 0;
  if (tree->GetBranch("ls_zLos") != 0) {
    ls_zLos_branch = tree->GetBranch("ls_zLos");
    if (ls_zLos_branch) { ls_zLos_branch->SetAddress(&ls_zLos_); }
  }
  ls_zHis_branch = 0;
  if (tree->GetBranch("ls_zHis") != 0) {
    ls_zHis_branch = tree->GetBranch("ls_zHis");
    if (ls_zHis_branch) { ls_zHis_branch->SetAddress(&ls_zHis_); }
  }
  ls_rtLos_branch = 0;
  if (tree->GetBranch("ls_rtLos") != 0) {
    ls_rtLos_branch = tree->GetBranch("ls_rtLos");
    if (ls_rtLos_branch) { ls_rtLos_branch->SetAddress(&ls_rtLos_); }
  }
  ls_dPhis_branch = 0;
  if (tree->GetBranch("ls_dPhis") != 0) {
    ls_dPhis_branch = tree->GetBranch("ls_dPhis");
    if (ls_dPhis_branch) { ls_dPhis_branch->SetAddress(&ls_dPhis_); }
  }
  ls_dPhiMins_branch = 0;
  if (tree->GetBranch("ls_dPhiMins") != 0) {
    ls_dPhiMins_branch = tree->GetBranch("ls_dPhiMins");
    if (ls_dPhiMins_branch) { ls_dPhiMins_branch->SetAddress(&ls_dPhiMins_); }
  }
  ls_dPhiMaxs_branch = 0;
  if (tree->GetBranch("ls_dPhiMaxs") != 0) {
    ls_dPhiMaxs_branch = tree->GetBranch("ls_dPhiMaxs");
    if (ls_dPhiMaxs_branch) { ls_dPhiMaxs_branch->SetAddress(&ls_dPhiMaxs_); }
  }
  ls_dPhiChanges_branch = 0;
  if (tree->GetBranch("ls_dPhiChanges") != 0) {
    ls_dPhiChanges_branch = tree->GetBranch("ls_dPhiChanges");
    if (ls_dPhiChanges_branch) { ls_dPhiChanges_branch->SetAddress(&ls_dPhiChanges_); }
  }
  ls_dPhiChangeMins_branch = 0;
  if (tree->GetBranch("ls_dPhiChangeMins") != 0) {
    ls_dPhiChangeMins_branch = tree->GetBranch("ls_dPhiChangeMins");
    if (ls_dPhiChangeMins_branch) { ls_dPhiChangeMins_branch->SetAddress(&ls_dPhiChangeMins_); }
  }
  ls_dPhiChangeMaxs_branch = 0;
  if (tree->GetBranch("ls_dPhiChangeMaxs") != 0) {
    ls_dPhiChangeMaxs_branch = tree->GetBranch("ls_dPhiChangeMaxs");
    if (ls_dPhiChangeMaxs_branch) { ls_dPhiChangeMaxs_branch->SetAddress(&ls_dPhiChangeMaxs_); }
  }
  ls_dAlphaInners_branch = 0;
  if (tree->GetBranch("ls_dAlphaInners") != 0) {
    ls_dAlphaInners_branch = tree->GetBranch("ls_dAlphaInners");
    if (ls_dAlphaInners_branch) { ls_dAlphaInners_branch->SetAddress(&ls_dAlphaInners_); }
  }
  ls_dAlphaOuters_branch = 0;
  if (tree->GetBranch("ls_dAlphaOuters") != 0) {
    ls_dAlphaOuters_branch = tree->GetBranch("ls_dAlphaOuters");
    if (ls_dAlphaOuters_branch) { ls_dAlphaOuters_branch->SetAddress(&ls_dAlphaOuters_); }
  }
  ls_dAlphaInnerOuters_branch = 0;
  if (tree->GetBranch("ls_dAlphaInnerOuters") != 0) {
    ls_dAlphaInnerOuters_branch = tree->GetBranch("ls_dAlphaInnerOuters");
    if (ls_dAlphaInnerOuters_branch) { ls_dAlphaInnerOuters_branch->SetAddress(&ls_dAlphaInnerOuters_); }
  }
#endif

  ls_simIdxAll_branch = 0;
  if (tree->GetBranch("ls_simIdxAll") != 0) {
    ls_simIdxAll_branch = tree->GetBranch("ls_simIdxAll");
    if (ls_simIdxAll_branch) { ls_simIdxAll_branch->SetAddress(&ls_simIdxAll_); }
  }
  ls_simIdxAllFrac_branch = 0;
  if (tree->GetBranch("ls_simIdxAllFrac") != 0) {
    ls_simIdxAllFrac_branch = tree->GetBranch("ls_simIdxAllFrac");
    if (ls_simIdxAllFrac_branch) { ls_simIdxAllFrac_branch->SetAddress(&ls_simIdxAllFrac_); }
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
  t3_lsIdx0_branch = 0;
  if (tree->GetBranch("t3_lsIdx0") != 0) {
    t3_lsIdx0_branch = tree->GetBranch("t3_lsIdx0");
    if (t3_lsIdx0_branch) { t3_lsIdx0_branch->SetAddress(&t3_lsIdx0_); }
  }
  t3_lsIdx1_branch = 0;
  if (tree->GetBranch("t3_lsIdx1") != 0) {
    t3_lsIdx1_branch = tree->GetBranch("t3_lsIdx1");
    if (t3_lsIdx1_branch) { t3_lsIdx1_branch->SetAddress(&t3_lsIdx1_); }
  }
  t3_isFake_branch = 0;
  if (tree->GetBranch("t3_isFake") != 0) {
    t3_isFake_branch = tree->GetBranch("t3_isFake");
    if (t3_isFake_branch) { t3_isFake_branch->SetAddress(&t3_isFake_); }
  }
  t3_isDuplicate_branch = 0;
  if (tree->GetBranch("t3_isDuplicate") != 0) {
    t3_isDuplicate_branch = tree->GetBranch("t3_isDuplicate");
    if (t3_isDuplicate_branch) { t3_isDuplicate_branch->SetAddress(&t3_isDuplicate_); }
  }
  t3_simIdx_branch = 0;
  if (tree->GetBranch("t3_simIdx") != 0) {
    t3_simIdx_branch = tree->GetBranch("t3_simIdx");
    if (t3_simIdx_branch) { t3_simIdx_branch->SetAddress(&t3_simIdx_); }
  }
  t3_simIdxAll_branch = 0;
  if (tree->GetBranch("t3_simIdxAll") != 0) {
    t3_simIdxAll_branch = tree->GetBranch("t3_simIdxAll");
    if (t3_simIdxAll_branch) { t3_simIdxAll_branch->SetAddress(&t3_simIdxAll_); }
  }
  t3_simIdxAllFrac_branch = 0;
  if (tree->GetBranch("t3_simIdxAllFrac") != 0) {
    t3_simIdxAllFrac_branch = tree->GetBranch("t3_simIdxAllFrac");
    if (t3_simIdxAllFrac_branch) { t3_simIdxAllFrac_branch->SetAddress(&t3_simIdxAllFrac_); }
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
  t5_t3Idx0_branch = 0;
  if (tree->GetBranch("t5_t3Idx0") != 0) {
    t5_t3Idx0_branch = tree->GetBranch("t5_t3Idx0");
    if (t5_t3Idx0_branch) { t5_t3Idx0_branch->SetAddress(&t5_t3Idx0_); }
  }
  t5_t3Idx1_branch = 0;
  if (tree->GetBranch("t5_t3Idx1") != 0) {
    t5_t3Idx1_branch = tree->GetBranch("t5_t3Idx1");
    if (t5_t3Idx1_branch) { t5_t3Idx1_branch->SetAddress(&t5_t3Idx1_); }
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
  t5_simIdx_branch = 0;
  if (tree->GetBranch("t5_simIdx") != 0) {
    t5_simIdx_branch = tree->GetBranch("t5_simIdx");
    if (t5_simIdx_branch) { t5_simIdx_branch->SetAddress(&t5_simIdx_); }
  }
  t5_simIdxAll_branch = 0;
  if (tree->GetBranch("t5_simIdxAll") != 0) {
    t5_simIdxAll_branch = tree->GetBranch("t5_simIdxAll");
    if (t5_simIdxAll_branch) { t5_simIdxAll_branch->SetAddress(&t5_simIdxAll_); }
  }
  t5_simIdxAllFrac_branch = 0;
  if (tree->GetBranch("t5_simIdxAllFrac") != 0) {
    t5_simIdxAllFrac_branch = tree->GetBranch("t5_simIdxAllFrac");
    if (t5_simIdxAllFrac_branch) { t5_simIdxAllFrac_branch->SetAddress(&t5_simIdxAllFrac_); }
  }
  pls_pt_branch = 0;
  if (tree->GetBranch("pls_pt") != 0) {
    pls_pt_branch = tree->GetBranch("pls_pt");
    if (pls_pt_branch) { pls_pt_branch->SetAddress(&pls_pt_); }
  }
  pls_eta_branch = 0;
  if (tree->GetBranch("pls_eta") != 0) {
    pls_eta_branch = tree->GetBranch("pls_eta");
    if (pls_eta_branch) { pls_eta_branch->SetAddress(&pls_eta_); }
  }
  pls_phi_branch = 0;
  if (tree->GetBranch("pls_phi") != 0) {
    pls_phi_branch = tree->GetBranch("pls_phi");
    if (pls_phi_branch) { pls_phi_branch->SetAddress(&pls_phi_); }
  }
  pls_nhit_branch = 0;
  if (tree->GetBranch("pls_nhit") != 0) {
    pls_nhit_branch = tree->GetBranch("pls_nhit");
    if (pls_nhit_branch) { pls_nhit_branch->SetAddress(&pls_nhit_); }
  }
  pls_hit0_x_branch = 0;
  if (tree->GetBranch("pls_hit0_x") != 0) {
    pls_hit0_x_branch = tree->GetBranch("pls_hit0_x");
    if (pls_hit0_x_branch) { pls_hit0_x_branch->SetAddress(&pls_hit0_x_); }
  }
  pls_hit0_y_branch = 0;
  if (tree->GetBranch("pls_hit0_y") != 0) {
    pls_hit0_y_branch = tree->GetBranch("pls_hit0_y");
    if (pls_hit0_y_branch) { pls_hit0_y_branch->SetAddress(&pls_hit0_y_); }
  }
  pls_hit0_z_branch = 0;
  if (tree->GetBranch("pls_hit0_z") != 0) {
    pls_hit0_z_branch = tree->GetBranch("pls_hit0_z");
    if (pls_hit0_z_branch) { pls_hit0_z_branch->SetAddress(&pls_hit0_z_); }
  }
  pls_hit1_x_branch = 0;
  if (tree->GetBranch("pls_hit1_x") != 0) {
    pls_hit1_x_branch = tree->GetBranch("pls_hit1_x");
    if (pls_hit1_x_branch) { pls_hit1_x_branch->SetAddress(&pls_hit1_x_); }
  }
  pls_hit1_y_branch = 0;
  if (tree->GetBranch("pls_hit1_y") != 0) {
    pls_hit1_y_branch = tree->GetBranch("pls_hit1_y");
    if (pls_hit1_y_branch) { pls_hit1_y_branch->SetAddress(&pls_hit1_y_); }
  }
  pls_hit1_z_branch = 0;
  if (tree->GetBranch("pls_hit1_z") != 0) {
    pls_hit1_z_branch = tree->GetBranch("pls_hit1_z");
    if (pls_hit1_z_branch) { pls_hit1_z_branch->SetAddress(&pls_hit1_z_); }
  }
  pls_hit2_x_branch = 0;
  if (tree->GetBranch("pls_hit2_x") != 0) {
    pls_hit2_x_branch = tree->GetBranch("pls_hit2_x");
    if (pls_hit2_x_branch) { pls_hit2_x_branch->SetAddress(&pls_hit2_x_); }
  }
  pls_hit2_y_branch = 0;
  if (tree->GetBranch("pls_hit2_y") != 0) {
    pls_hit2_y_branch = tree->GetBranch("pls_hit2_y");
    if (pls_hit2_y_branch) { pls_hit2_y_branch->SetAddress(&pls_hit2_y_); }
  }
  pls_hit2_z_branch = 0;
  if (tree->GetBranch("pls_hit2_z") != 0) {
    pls_hit2_z_branch = tree->GetBranch("pls_hit2_z");
    if (pls_hit2_z_branch) { pls_hit2_z_branch->SetAddress(&pls_hit2_z_); }
  }
  pls_hit3_x_branch = 0;
  if (tree->GetBranch("pls_hit3_x") != 0) {
    pls_hit3_x_branch = tree->GetBranch("pls_hit3_x");
    if (pls_hit3_x_branch) { pls_hit3_x_branch->SetAddress(&pls_hit3_x_); }
  }
  pls_hit3_y_branch = 0;
  if (tree->GetBranch("pls_hit3_y") != 0) {
    pls_hit3_y_branch = tree->GetBranch("pls_hit3_y");
    if (pls_hit3_y_branch) { pls_hit3_y_branch->SetAddress(&pls_hit3_y_); }
  }
  pls_hit3_z_branch = 0;
  if (tree->GetBranch("pls_hit3_z") != 0) {
    pls_hit3_z_branch = tree->GetBranch("pls_hit3_z");
    if (pls_hit3_z_branch) { pls_hit3_z_branch->SetAddress(&pls_hit3_z_); }
  }
  pls_isFake_branch = 0;
  if (tree->GetBranch("pls_isFake") != 0) {
    pls_isFake_branch = tree->GetBranch("pls_isFake");
    if (pls_isFake_branch) { pls_isFake_branch->SetAddress(&pls_isFake_); }
  }
  pls_isDuplicate_branch = 0;
  if (tree->GetBranch("pls_isDuplicate") != 0) {
    pls_isDuplicate_branch = tree->GetBranch("pls_isDuplicate");
    if (pls_isDuplicate_branch) { pls_isDuplicate_branch->SetAddress(&pls_isDuplicate_); }
  }
  pls_simIdx_branch = 0;
  if (tree->GetBranch("pls_simIdx") != 0) {
    pls_simIdx_branch = tree->GetBranch("pls_simIdx");
    if (pls_simIdx_branch) { pls_simIdx_branch->SetAddress(&pls_simIdx_); }
  }
  pls_simIdxAll_branch = 0;
  if (tree->GetBranch("pls_simIdxAll") != 0) {
    pls_simIdxAll_branch = tree->GetBranch("pls_simIdxAll");
    if (pls_simIdxAll_branch) { pls_simIdxAll_branch->SetAddress(&pls_simIdxAll_); }
  }
  pls_simIdxAllFrac_branch = 0;
  if (tree->GetBranch("pls_simIdxAllFrac") != 0) {
    pls_simIdxAllFrac_branch = tree->GetBranch("pls_simIdxAllFrac");
    if (pls_simIdxAllFrac_branch) { pls_simIdxAllFrac_branch->SetAddress(&pls_simIdxAllFrac_); }
  }
  pt3_pt_branch = 0;
  if (tree->GetBranch("pt3_pt") != 0) {
    pt3_pt_branch = tree->GetBranch("pt3_pt");
    if (pt3_pt_branch) { pt3_pt_branch->SetAddress(&pt3_pt_); }
  }
  pt3_eta_branch = 0;
  if (tree->GetBranch("pt3_eta") != 0) {
    pt3_eta_branch = tree->GetBranch("pt3_eta");
    if (pt3_eta_branch) { pt3_eta_branch->SetAddress(&pt3_eta_); }
  }
  pt3_phi_branch = 0;
  if (tree->GetBranch("pt3_phi") != 0) {
    pt3_phi_branch = tree->GetBranch("pt3_phi");
    if (pt3_phi_branch) { pt3_phi_branch->SetAddress(&pt3_phi_); }
  }
  pt3_plsIdx_branch = 0;
  if (tree->GetBranch("pt3_plsIdx") != 0) {
    pt3_plsIdx_branch = tree->GetBranch("pt3_plsIdx");
    if (pt3_plsIdx_branch) { pt3_plsIdx_branch->SetAddress(&pt3_plsIdx_); }
  }
  pt3_t3Idx_branch = 0;
  if (tree->GetBranch("pt3_t3Idx") != 0) {
    pt3_t3Idx_branch = tree->GetBranch("pt3_t3Idx");
    if (pt3_t3Idx_branch) { pt3_t3Idx_branch->SetAddress(&pt3_t3Idx_); }
  }
  pt3_isFake_branch = 0;
  if (tree->GetBranch("pt3_isFake") != 0) {
    pt3_isFake_branch = tree->GetBranch("pt3_isFake");
    if (pt3_isFake_branch) { pt3_isFake_branch->SetAddress(&pt3_isFake_); }
  }
  pt3_isDuplicate_branch = 0;
  if (tree->GetBranch("pt3_isDuplicate") != 0) {
    pt3_isDuplicate_branch = tree->GetBranch("pt3_isDuplicate");
    if (pt3_isDuplicate_branch) { pt3_isDuplicate_branch->SetAddress(&pt3_isDuplicate_); }
  }
  pt3_simIdx_branch = 0;
  if (tree->GetBranch("pt3_simIdx") != 0) {
    pt3_simIdx_branch = tree->GetBranch("pt3_simIdx");
    if (pt3_simIdx_branch) { pt3_simIdx_branch->SetAddress(&pt3_simIdx_); }
  }
  pt3_simIdxAll_branch = 0;
  if (tree->GetBranch("pt3_simIdxAll") != 0) {
    pt3_simIdxAll_branch = tree->GetBranch("pt3_simIdxAll");
    if (pt3_simIdxAll_branch) { pt3_simIdxAll_branch->SetAddress(&pt3_simIdxAll_); }
  }
  pt3_simIdxAllFrac_branch = 0;
  if (tree->GetBranch("pt3_simIdxAllFrac") != 0) {
    pt3_simIdxAllFrac_branch = tree->GetBranch("pt3_simIdxAllFrac");
    if (pt3_simIdxAllFrac_branch) { pt3_simIdxAllFrac_branch->SetAddress(&pt3_simIdxAllFrac_); }
  }
  pt5_pt_branch = 0;
  if (tree->GetBranch("pt5_pt") != 0) {
    pt5_pt_branch = tree->GetBranch("pt5_pt");
    if (pt5_pt_branch) { pt5_pt_branch->SetAddress(&pt5_pt_); }
  }
  pt5_eta_branch = 0;
  if (tree->GetBranch("pt5_eta") != 0) {
    pt5_eta_branch = tree->GetBranch("pt5_eta");
    if (pt5_eta_branch) { pt5_eta_branch->SetAddress(&pt5_eta_); }
  }
  pt5_phi_branch = 0;
  if (tree->GetBranch("pt5_phi") != 0) {
    pt5_phi_branch = tree->GetBranch("pt5_phi");
    if (pt5_phi_branch) { pt5_phi_branch->SetAddress(&pt5_phi_); }
  }
  pt5_plsIdx_branch = 0;
  if (tree->GetBranch("pt5_plsIdx") != 0) {
    pt5_plsIdx_branch = tree->GetBranch("pt5_plsIdx");
    if (pt5_plsIdx_branch) { pt5_plsIdx_branch->SetAddress(&pt5_plsIdx_); }
  }
  pt5_t5Idx_branch = 0;
  if (tree->GetBranch("pt5_t5Idx") != 0) {
    pt5_t5Idx_branch = tree->GetBranch("pt5_t5Idx");
    if (pt5_t5Idx_branch) { pt5_t5Idx_branch->SetAddress(&pt5_t5Idx_); }
  }
  pt5_isFake_branch = 0;
  if (tree->GetBranch("pt5_isFake") != 0) {
    pt5_isFake_branch = tree->GetBranch("pt5_isFake");
    if (pt5_isFake_branch) { pt5_isFake_branch->SetAddress(&pt5_isFake_); }
  }
  pt5_isDuplicate_branch = 0;
  if (tree->GetBranch("pt5_isDuplicate") != 0) {
    pt5_isDuplicate_branch = tree->GetBranch("pt5_isDuplicate");
    if (pt5_isDuplicate_branch) { pt5_isDuplicate_branch->SetAddress(&pt5_isDuplicate_); }
  }
  pt5_simIdx_branch = 0;
  if (tree->GetBranch("pt5_simIdx") != 0) {
    pt5_simIdx_branch = tree->GetBranch("pt5_simIdx");
    if (pt5_simIdx_branch) { pt5_simIdx_branch->SetAddress(&pt5_simIdx_); }
  }
  pt5_simIdxAll_branch = 0;
  if (tree->GetBranch("pt5_simIdxAll") != 0) {
    pt5_simIdxAll_branch = tree->GetBranch("pt5_simIdxAll");
    if (pt5_simIdxAll_branch) { pt5_simIdxAll_branch->SetAddress(&pt5_simIdxAll_); }
  }
  pt5_simIdxAllFrac_branch = 0;
  if (tree->GetBranch("pt5_simIdxAllFrac") != 0) {
    pt5_simIdxAllFrac_branch = tree->GetBranch("pt5_simIdxAllFrac");
    if (pt5_simIdxAllFrac_branch) { pt5_simIdxAllFrac_branch->SetAddress(&pt5_simIdxAllFrac_); }
  }
  tree->SetMakeClass(0);
}
void SDL::GetEntry(unsigned int idx) {
  index = idx;
  sim_pt_isLoaded = false;
  sim_eta_isLoaded = false;
  sim_phi_isLoaded = false;
  sim_pca_dxy_isLoaded = false;
  sim_pca_dz_isLoaded = false;
  sim_q_isLoaded = false;
  sim_pdgId_isLoaded = false;
  sim_vx_isLoaded = false;
  sim_vy_isLoaded = false;
  sim_vz_isLoaded = false;
  sim_vtxperp_isLoaded = false;
  sim_trkNtupIdx_isLoaded = false;
  sim_tcIdx_isLoaded = false;
  sim_tcIdxAll_isLoaded = false;
  sim_tcIdxAllFrac_isLoaded = false;
  sim_mdIdxAll_isLoaded = false;
  sim_mdIdxAllFrac_isLoaded = false;
  sim_lsIdxAll_isLoaded = false;
  sim_lsIdxAllFrac_isLoaded = false;
  sim_t3IdxAll_isLoaded = false;
  sim_t3IdxAllFrac_isLoaded = false;
  sim_t5IdxAll_isLoaded = false;
  sim_t5IdxAllFrac_isLoaded = false;
  sim_plsIdxAll_isLoaded = false;
  sim_plsIdxAllFrac_isLoaded = false;
  sim_pt3IdxAll_isLoaded = false;
  sim_pt3IdxAllFrac_isLoaded = false;
  sim_pt5IdxAll_isLoaded = false;
  sim_pt5IdxAllFrac_isLoaded = false;
  sim_simHitX_isLoaded = false;
  sim_simHitY_isLoaded = false;
  sim_simHitZ_isLoaded = false;
  sim_simHitDetId_isLoaded = false;
  sim_simHitLayer_isLoaded = false;
  sim_simHitDistxyHelix_isLoaded = false;
  sim_simHitLayerMinDistxyHelix_isLoaded = false;
  sim_recoHitX_isLoaded = false;
  sim_recoHitY_isLoaded = false;
  sim_recoHitZ_isLoaded = false;
  sim_recoHitDetId_isLoaded = false;
  tc_pt_isLoaded = false;
  tc_eta_isLoaded = false;
  tc_phi_isLoaded = false;
  tc_type_isLoaded = false;
  tc_pt5Idx_isLoaded = false;
  tc_pt3Idx_isLoaded = false;
  tc_t5Idx_isLoaded = false;
  tc_plsIdx_isLoaded = false;
  tc_isFake_isLoaded = false;
  tc_isDuplicate_isLoaded = false;
  tc_simIdx_isLoaded = false;
  tc_simIdxAll_isLoaded = false;
  tc_simIdxAllFrac_isLoaded = false;
  md_pt_isLoaded = false;
  md_eta_isLoaded = false;
  md_phi_isLoaded = false;
  #ifdef OUTPUT_MD_CUTS
  md_dphi_isLoaded = false;
  md_dphichange_isLoaded = false;
  md_dz_isLoaded = false;
  #endif
  md_anchor_x_isLoaded = false;
  md_anchor_y_isLoaded = false;
  md_anchor_z_isLoaded = false;
  md_other_x_isLoaded = false;
  md_other_y_isLoaded = false;
  md_other_z_isLoaded = false;
  md_type_isLoaded = false;
  md_layer_isLoaded = false;
  md_detId_isLoaded = false;
  md_isFake_isLoaded = false;
  md_simIdx_isLoaded = false;
  md_simIdxAll_isLoaded = false;
  md_simIdxAllFrac_isLoaded = false;
  ls_pt_isLoaded = false;
  ls_eta_isLoaded = false;
  ls_phi_isLoaded = false;
  ls_mdIdx0_isLoaded = false;
  ls_mdIdx1_isLoaded = false;
  ls_isFake_isLoaded = false;

  #ifdef OUTPUT_LS_CUTS
  ls_zLos_isLoaded = false;
  ls_zHis_isLoaded = false;
  ls_rtLos_isLoaded = false;
  ls_rtHis_isLoaded = false;
  ls_dPhis_isLoaded = false;
  ls_dPhiMins_isLoaded = false;
  ls_dPhiMaxs_isLoaded = false;
  ls_dPhiChanges_isLoaded = false;
  ls_dPhiChangeMins_isLoaded = false;
  ls_dPhiChangeMaxs_isLoaded = false;
  ls_dAlphaInners_isLoaded = false;
  ls_dAlphaOuters_isLoaded = false;
  ls_dAlphaInnerOuters_isLoaded = false;
  #endif

  ls_simIdxAll_isLoaded = false;
  ls_simIdxAllFrac_isLoaded = false;
  t3_pt_isLoaded = false;
  t3_eta_isLoaded = false;
  t3_phi_isLoaded = false;
  t3_lsIdx0_isLoaded = false;
  t3_lsIdx1_isLoaded = false;
  t3_isFake_isLoaded = false;
  t3_isDuplicate_isLoaded = false;
  t3_simIdx_isLoaded = false;
  t3_simIdxAll_isLoaded = false;
  t3_simIdxAllFrac_isLoaded = false;
  t5_pt_isLoaded = false;
  t5_eta_isLoaded = false;
  t5_phi_isLoaded = false;
  t5_t3Idx0_isLoaded = false;
  t5_t3Idx1_isLoaded = false;
  t5_isFake_isLoaded = false;
  t5_isDuplicate_isLoaded = false;
  t5_simIdx_isLoaded = false;
  t5_simIdxAll_isLoaded = false;
  t5_simIdxAllFrac_isLoaded = false;
  pls_pt_isLoaded = false;
  pls_eta_isLoaded = false;
  pls_phi_isLoaded = false;
  pls_nhit_isLoaded = false;
  pls_hit0_x_isLoaded = false;
  pls_hit0_y_isLoaded = false;
  pls_hit0_z_isLoaded = false;
  pls_hit1_x_isLoaded = false;
  pls_hit1_y_isLoaded = false;
  pls_hit1_z_isLoaded = false;
  pls_hit2_x_isLoaded = false;
  pls_hit2_y_isLoaded = false;
  pls_hit2_z_isLoaded = false;
  pls_hit3_x_isLoaded = false;
  pls_hit3_y_isLoaded = false;
  pls_hit3_z_isLoaded = false;
  pls_isFake_isLoaded = false;
  pls_isDuplicate_isLoaded = false;
  pls_simIdx_isLoaded = false;
  pls_simIdxAll_isLoaded = false;
  pls_simIdxAllFrac_isLoaded = false;
  pt3_pt_isLoaded = false;
  pt3_eta_isLoaded = false;
  pt3_phi_isLoaded = false;
  pt3_plsIdx_isLoaded = false;
  pt3_t3Idx_isLoaded = false;
  pt3_isFake_isLoaded = false;
  pt3_isDuplicate_isLoaded = false;
  pt3_simIdx_isLoaded = false;
  pt3_simIdxAll_isLoaded = false;
  pt3_simIdxAllFrac_isLoaded = false;
  pt5_pt_isLoaded = false;
  pt5_eta_isLoaded = false;
  pt5_phi_isLoaded = false;
  pt5_plsIdx_isLoaded = false;
  pt5_t5Idx_isLoaded = false;
  pt5_isFake_isLoaded = false;
  pt5_isDuplicate_isLoaded = false;
  pt5_simIdx_isLoaded = false;
  pt5_simIdxAll_isLoaded = false;
  pt5_simIdxAllFrac_isLoaded = false;
}
void SDL::LoadAllBranches() {
  if (sim_pt_branch != 0) sim_pt();
  if (sim_eta_branch != 0) sim_eta();
  if (sim_phi_branch != 0) sim_phi();
  if (sim_pca_dxy_branch != 0) sim_pca_dxy();
  if (sim_pca_dz_branch != 0) sim_pca_dz();
  if (sim_q_branch != 0) sim_q();
  if (sim_pdgId_branch != 0) sim_pdgId();
  if (sim_vx_branch != 0) sim_vx();
  if (sim_vy_branch != 0) sim_vy();
  if (sim_vz_branch != 0) sim_vz();
  if (sim_vtxperp_branch != 0) sim_vtxperp();
  if (sim_trkNtupIdx_branch != 0) sim_trkNtupIdx();
  if (sim_tcIdx_branch != 0) sim_tcIdx();
  if (sim_tcIdxAll_branch != 0) sim_tcIdxAll();
  if (sim_tcIdxAllFrac_branch != 0) sim_tcIdxAllFrac();
  if (sim_mdIdxAll_branch != 0) sim_mdIdxAll();
  if (sim_mdIdxAllFrac_branch != 0) sim_mdIdxAllFrac();
  if (sim_lsIdxAll_branch != 0) sim_lsIdxAll();
  if (sim_lsIdxAllFrac_branch != 0) sim_lsIdxAllFrac();
  if (sim_t3IdxAll_branch != 0) sim_t3IdxAll();
  if (sim_t3IdxAllFrac_branch != 0) sim_t3IdxAllFrac();
  if (sim_t5IdxAll_branch != 0) sim_t5IdxAll();
  if (sim_t5IdxAllFrac_branch != 0) sim_t5IdxAllFrac();
  if (sim_plsIdxAll_branch != 0) sim_plsIdxAll();
  if (sim_plsIdxAllFrac_branch != 0) sim_plsIdxAllFrac();
  if (sim_pt3IdxAll_branch != 0) sim_pt3IdxAll();
  if (sim_pt3IdxAllFrac_branch != 0) sim_pt3IdxAllFrac();
  if (sim_pt5IdxAll_branch != 0) sim_pt5IdxAll();
  if (sim_pt5IdxAllFrac_branch != 0) sim_pt5IdxAllFrac();
  if (sim_simHitX_branch != 0) sim_simHitX();
  if (sim_simHitY_branch != 0) sim_simHitY();
  if (sim_simHitZ_branch != 0) sim_simHitZ();
  if (sim_simHitDetId_branch != 0) sim_simHitDetId();
  if (sim_simHitLayer_branch != 0) sim_simHitLayer();
  if (sim_simHitDistxyHelix_branch != 0) sim_simHitDistxyHelix();
  if (sim_simHitLayerMinDistxyHelix_branch != 0) sim_simHitLayerMinDistxyHelix();
  if (sim_recoHitX_branch != 0) sim_recoHitX();
  if (sim_recoHitY_branch != 0) sim_recoHitY();
  if (sim_recoHitZ_branch != 0) sim_recoHitZ();
  if (sim_recoHitDetId_branch != 0) sim_recoHitDetId();
  if (tc_pt_branch != 0) tc_pt();
  if (tc_eta_branch != 0) tc_eta();
  if (tc_phi_branch != 0) tc_phi();
  if (tc_type_branch != 0) tc_type();
  if (tc_pt5Idx_branch != 0) tc_pt5Idx();
  if (tc_pt3Idx_branch != 0) tc_pt3Idx();
  if (tc_t5Idx_branch != 0) tc_t5Idx();
  if (tc_plsIdx_branch != 0) tc_plsIdx();
  if (tc_isFake_branch != 0) tc_isFake();
  if (tc_isDuplicate_branch != 0) tc_isDuplicate();
  if (tc_simIdx_branch != 0) tc_simIdx();
  if (tc_simIdxAll_branch != 0) tc_simIdxAll();
  if (tc_simIdxAllFrac_branch != 0) tc_simIdxAllFrac();
  if (md_pt_branch != 0) md_pt();
  if (md_eta_branch != 0) md_eta();
  if (md_phi_branch != 0) md_phi();

  #ifdef OUTPUT_MD_CUTS
  if (md_dphi_branch !=0) md_dphi();
  if (md_dphichange_branch !=0) md_dphichange();
  if (md_dz_branch !=0) md_dz();
  #endif

  if (md_anchor_x_branch != 0) md_anchor_x();
  if (md_anchor_y_branch != 0) md_anchor_y();
  if (md_anchor_z_branch != 0) md_anchor_z();
  if (md_other_x_branch != 0) md_other_x();
  if (md_other_y_branch != 0) md_other_y();
  if (md_other_z_branch != 0) md_other_z();
  if (md_type_branch != 0) md_type();
  if (md_layer_branch != 0) md_layer();
  if (md_detId_branch != 0) md_detId();
  if (md_isFake_branch != 0) md_isFake();
  if (md_simIdx_branch != 0) md_simIdx();
  if (md_simIdxAll_branch != 0) md_simIdxAll();
  if (md_simIdxAllFrac_branch != 0) md_simIdxAllFrac();
  if (ls_pt_branch != 0) ls_pt();
  if (ls_eta_branch != 0) ls_eta();
  if (ls_phi_branch != 0) ls_phi();
  if (ls_mdIdx0_branch != 0) ls_mdIdx0();
  if (ls_mdIdx1_branch != 0) ls_mdIdx1();
  if (ls_isFake_branch != 0) ls_isFake();
  if (ls_simIdx_branch != 0) ls_simIdx();
  #ifdef OUTPUT_LS_CUTS
  if (ls_zLos_branch != 0) ls_zLos();
  if (ls_zHis_branch != 0) ls_zHis();
  if (ls_rtLos_branch != 0) ls_rtLos();
  if (ls_rtHis_branch != 0) ls_rtHis();
  if (ls_dPhis_branch != 0) ls_dPhis();
  if (ls_dPhiMins_branch != 0) ls_dPhiMins();
  if (ls_dPhiMaxs_branch != 0) ls_dPhiMaxs();
  if (ls_dPhiChanges_branch != 0) ls_dPhiChanges();
  if (ls_dPhiChangeMins_branch != 0) ls_dPhiChangeMins();
  if (ls_dPhiChangeMaxs_branch != 0) ls_dPhiChangeMaxs();
  if (ls_dAlphaInners_branch != 0) ls_dAlphaInners();
  if (ls_dAlphaOuters_branch != 0) ls_dAlphaOuters();
  if (ls_dAlphaInnerOuters_branch != 0) ls_dAlphaInnerOuters();
  #endif
  if (ls_simIdxAll_branch != 0) ls_simIdxAll();
  if (ls_simIdxAllFrac_branch != 0) ls_simIdxAllFrac();
  if (t3_pt_branch != 0) t3_pt();
  if (t3_eta_branch != 0) t3_eta();
  if (t3_phi_branch != 0) t3_phi();
  if (t3_lsIdx0_branch != 0) t3_lsIdx0();
  if (t3_lsIdx1_branch != 0) t3_lsIdx1();
  if (t3_isFake_branch != 0) t3_isFake();
  if (t3_isDuplicate_branch != 0) t3_isDuplicate();
  if (t3_simIdx_branch != 0) t3_simIdx();
  if (t3_simIdxAll_branch != 0) t3_simIdxAll();
  if (t3_simIdxAllFrac_branch != 0) t3_simIdxAllFrac();
  if (t5_pt_branch != 0) t5_pt();
  if (t5_eta_branch != 0) t5_eta();
  if (t5_phi_branch != 0) t5_phi();
  if (t5_t3Idx0_branch != 0) t5_t3Idx0();
  if (t5_t3Idx1_branch != 0) t5_t3Idx1();
  if (t5_isFake_branch != 0) t5_isFake();
  if (t5_isDuplicate_branch != 0) t5_isDuplicate();
  if (t5_simIdx_branch != 0) t5_simIdx();
  if (t5_simIdxAll_branch != 0) t5_simIdxAll();
  if (t5_simIdxAllFrac_branch != 0) t5_simIdxAllFrac();
  if (pls_pt_branch != 0) pls_pt();
  if (pls_eta_branch != 0) pls_eta();
  if (pls_phi_branch != 0) pls_phi();
  if (pls_nhit_branch != 0) pls_nhit();
  if (pls_hit0_x_branch != 0) pls_hit0_x();
  if (pls_hit0_y_branch != 0) pls_hit0_y();
  if (pls_hit0_z_branch != 0) pls_hit0_z();
  if (pls_hit1_x_branch != 0) pls_hit1_x();
  if (pls_hit1_y_branch != 0) pls_hit1_y();
  if (pls_hit1_z_branch != 0) pls_hit1_z();
  if (pls_hit2_x_branch != 0) pls_hit2_x();
  if (pls_hit2_y_branch != 0) pls_hit2_y();
  if (pls_hit2_z_branch != 0) pls_hit2_z();
  if (pls_hit3_x_branch != 0) pls_hit3_x();
  if (pls_hit3_y_branch != 0) pls_hit3_y();
  if (pls_hit3_z_branch != 0) pls_hit3_z();
  if (pls_isFake_branch != 0) pls_isFake();
  if (pls_isDuplicate_branch != 0) pls_isDuplicate();
  if (pls_simIdx_branch != 0) pls_simIdx();
  if (pls_simIdxAll_branch != 0) pls_simIdxAll();
  if (pls_simIdxAllFrac_branch != 0) pls_simIdxAllFrac();
  if (pt3_pt_branch != 0) pt3_pt();
  if (pt3_eta_branch != 0) pt3_eta();
  if (pt3_phi_branch != 0) pt3_phi();
  if (pt3_plsIdx_branch != 0) pt3_plsIdx();
  if (pt3_t3Idx_branch != 0) pt3_t3Idx();
  if (pt3_isFake_branch != 0) pt3_isFake();
  if (pt3_isDuplicate_branch != 0) pt3_isDuplicate();
  if (pt3_simIdx_branch != 0) pt3_simIdx();
  if (pt3_simIdxAll_branch != 0) pt3_simIdxAll();
  if (pt3_simIdxAllFrac_branch != 0) pt3_simIdxAllFrac();
  if (pt5_pt_branch != 0) pt5_pt();
  if (pt5_eta_branch != 0) pt5_eta();
  if (pt5_phi_branch != 0) pt5_phi();
  if (pt5_plsIdx_branch != 0) pt5_plsIdx();
  if (pt5_t5Idx_branch != 0) pt5_t5Idx();
  if (pt5_isFake_branch != 0) pt5_isFake();
  if (pt5_isDuplicate_branch != 0) pt5_isDuplicate();
  if (pt5_simIdx_branch != 0) pt5_simIdx();
  if (pt5_simIdxAll_branch != 0) pt5_simIdxAll();
  if (pt5_simIdxAllFrac_branch != 0) pt5_simIdxAllFrac();
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
const vector<float> &SDL::sim_vtxperp() {
  if (not sim_vtxperp_isLoaded) {
    if (sim_vtxperp_branch != 0) {
      sim_vtxperp_branch->GetEntry(index);
    } else {
      printf("branch sim_vtxperp_branch does not exist!\n");
      exit(1);
    }
    sim_vtxperp_isLoaded = true;
  }
  return *sim_vtxperp_;
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
const vector<int> &SDL::sim_tcIdx() {
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
const vector<vector<int> > &SDL::sim_tcIdxAll() {
  if (not sim_tcIdxAll_isLoaded) {
    if (sim_tcIdxAll_branch != 0) {
      sim_tcIdxAll_branch->GetEntry(index);
    } else {
      printf("branch sim_tcIdxAll_branch does not exist!\n");
      exit(1);
    }
    sim_tcIdxAll_isLoaded = true;
  }
  return *sim_tcIdxAll_;
}
const vector<vector<float> > &SDL::sim_tcIdxAllFrac() {
  if (not sim_tcIdxAllFrac_isLoaded) {
    if (sim_tcIdxAllFrac_branch != 0) {
      sim_tcIdxAllFrac_branch->GetEntry(index);
    } else {
      printf("branch sim_tcIdxAllFrac_branch does not exist!\n");
      exit(1);
    }
    sim_tcIdxAllFrac_isLoaded = true;
  }
  return *sim_tcIdxAllFrac_;
}
const vector<vector<int> > &SDL::sim_mdIdxAll() {
  if (not sim_mdIdxAll_isLoaded) {
    if (sim_mdIdxAll_branch != 0) {
      sim_mdIdxAll_branch->GetEntry(index);
    } else {
      printf("branch sim_mdIdxAll_branch does not exist!\n");
      exit(1);
    }
    sim_mdIdxAll_isLoaded = true;
  }
  return *sim_mdIdxAll_;
}
const vector<vector<float> > &SDL::sim_mdIdxAllFrac() {
  if (not sim_mdIdxAllFrac_isLoaded) {
    if (sim_mdIdxAllFrac_branch != 0) {
      sim_mdIdxAllFrac_branch->GetEntry(index);
    } else {
      printf("branch sim_mdIdxAllFrac_branch does not exist!\n");
      exit(1);
    }
    sim_mdIdxAllFrac_isLoaded = true;
  }
  return *sim_mdIdxAllFrac_;
}
const vector<vector<int> > &SDL::sim_lsIdxAll() {
  if (not sim_lsIdxAll_isLoaded) {
    if (sim_lsIdxAll_branch != 0) {
      sim_lsIdxAll_branch->GetEntry(index);
    } else {
      printf("branch sim_lsIdxAll_branch does not exist!\n");
      exit(1);
    }
    sim_lsIdxAll_isLoaded = true;
  }
  return *sim_lsIdxAll_;
}
const vector<vector<float> > &SDL::sim_lsIdxAllFrac() {
  if (not sim_lsIdxAllFrac_isLoaded) {
    if (sim_lsIdxAllFrac_branch != 0) {
      sim_lsIdxAllFrac_branch->GetEntry(index);
    } else {
      printf("branch sim_lsIdxAllFrac_branch does not exist!\n");
      exit(1);
    }
    sim_lsIdxAllFrac_isLoaded = true;
  }
  return *sim_lsIdxAllFrac_;
}
const vector<vector<int> > &SDL::sim_t3IdxAll() {
  if (not sim_t3IdxAll_isLoaded) {
    if (sim_t3IdxAll_branch != 0) {
      sim_t3IdxAll_branch->GetEntry(index);
    } else {
      printf("branch sim_t3IdxAll_branch does not exist!\n");
      exit(1);
    }
    sim_t3IdxAll_isLoaded = true;
  }
  return *sim_t3IdxAll_;
}
const vector<vector<float> > &SDL::sim_t3IdxAllFrac() {
  if (not sim_t3IdxAllFrac_isLoaded) {
    if (sim_t3IdxAllFrac_branch != 0) {
      sim_t3IdxAllFrac_branch->GetEntry(index);
    } else {
      printf("branch sim_t3IdxAllFrac_branch does not exist!\n");
      exit(1);
    }
    sim_t3IdxAllFrac_isLoaded = true;
  }
  return *sim_t3IdxAllFrac_;
}
const vector<vector<int> > &SDL::sim_t5IdxAll() {
  if (not sim_t5IdxAll_isLoaded) {
    if (sim_t5IdxAll_branch != 0) {
      sim_t5IdxAll_branch->GetEntry(index);
    } else {
      printf("branch sim_t5IdxAll_branch does not exist!\n");
      exit(1);
    }
    sim_t5IdxAll_isLoaded = true;
  }
  return *sim_t5IdxAll_;
}
const vector<vector<float> > &SDL::sim_t5IdxAllFrac() {
  if (not sim_t5IdxAllFrac_isLoaded) {
    if (sim_t5IdxAllFrac_branch != 0) {
      sim_t5IdxAllFrac_branch->GetEntry(index);
    } else {
      printf("branch sim_t5IdxAllFrac_branch does not exist!\n");
      exit(1);
    }
    sim_t5IdxAllFrac_isLoaded = true;
  }
  return *sim_t5IdxAllFrac_;
}
const vector<vector<int> > &SDL::sim_plsIdxAll() {
  if (not sim_plsIdxAll_isLoaded) {
    if (sim_plsIdxAll_branch != 0) {
      sim_plsIdxAll_branch->GetEntry(index);
    } else {
      printf("branch sim_plsIdxAll_branch does not exist!\n");
      exit(1);
    }
    sim_plsIdxAll_isLoaded = true;
  }
  return *sim_plsIdxAll_;
}
const vector<vector<float> > &SDL::sim_plsIdxAllFrac() {
  if (not sim_plsIdxAllFrac_isLoaded) {
    if (sim_plsIdxAllFrac_branch != 0) {
      sim_plsIdxAllFrac_branch->GetEntry(index);
    } else {
      printf("branch sim_plsIdxAllFrac_branch does not exist!\n");
      exit(1);
    }
    sim_plsIdxAllFrac_isLoaded = true;
  }
  return *sim_plsIdxAllFrac_;
}
const vector<vector<int> > &SDL::sim_pt3IdxAll() {
  if (not sim_pt3IdxAll_isLoaded) {
    if (sim_pt3IdxAll_branch != 0) {
      sim_pt3IdxAll_branch->GetEntry(index);
    } else {
      printf("branch sim_pt3IdxAll_branch does not exist!\n");
      exit(1);
    }
    sim_pt3IdxAll_isLoaded = true;
  }
  return *sim_pt3IdxAll_;
}
const vector<vector<float> > &SDL::sim_pt3IdxAllFrac() {
  if (not sim_pt3IdxAllFrac_isLoaded) {
    if (sim_pt3IdxAllFrac_branch != 0) {
      sim_pt3IdxAllFrac_branch->GetEntry(index);
    } else {
      printf("branch sim_pt3IdxAllFrac_branch does not exist!\n");
      exit(1);
    }
    sim_pt3IdxAllFrac_isLoaded = true;
  }
  return *sim_pt3IdxAllFrac_;
}
const vector<vector<int> > &SDL::sim_pt5IdxAll() {
  if (not sim_pt5IdxAll_isLoaded) {
    if (sim_pt5IdxAll_branch != 0) {
      sim_pt5IdxAll_branch->GetEntry(index);
    } else {
      printf("branch sim_pt5IdxAll_branch does not exist!\n");
      exit(1);
    }
    sim_pt5IdxAll_isLoaded = true;
  }
  return *sim_pt5IdxAll_;
}
const vector<vector<float> > &SDL::sim_pt5IdxAllFrac() {
  if (not sim_pt5IdxAllFrac_isLoaded) {
    if (sim_pt5IdxAllFrac_branch != 0) {
      sim_pt5IdxAllFrac_branch->GetEntry(index);
    } else {
      printf("branch sim_pt5IdxAllFrac_branch does not exist!\n");
      exit(1);
    }
    sim_pt5IdxAllFrac_isLoaded = true;
  }
  return *sim_pt5IdxAllFrac_;
}
const vector<vector<float> > &SDL::sim_simHitX() {
  if (not sim_simHitX_isLoaded) {
    if (sim_simHitX_branch != 0) {
      sim_simHitX_branch->GetEntry(index);
    } else {
      printf("branch sim_simHitX_branch does not exist!\n");
      exit(1);
    }
    sim_simHitX_isLoaded = true;
  }
  return *sim_simHitX_;
}
const vector<vector<float> > &SDL::sim_simHitY() {
  if (not sim_simHitY_isLoaded) {
    if (sim_simHitY_branch != 0) {
      sim_simHitY_branch->GetEntry(index);
    } else {
      printf("branch sim_simHitY_branch does not exist!\n");
      exit(1);
    }
    sim_simHitY_isLoaded = true;
  }
  return *sim_simHitY_;
}
const vector<vector<float> > &SDL::sim_simHitZ() {
  if (not sim_simHitZ_isLoaded) {
    if (sim_simHitZ_branch != 0) {
      sim_simHitZ_branch->GetEntry(index);
    } else {
      printf("branch sim_simHitZ_branch does not exist!\n");
      exit(1);
    }
    sim_simHitZ_isLoaded = true;
  }
  return *sim_simHitZ_;
}
const vector<vector<int> > &SDL::sim_simHitDetId() {
  if (not sim_simHitDetId_isLoaded) {
    if (sim_simHitDetId_branch != 0) {
      sim_simHitDetId_branch->GetEntry(index);
    } else {
      printf("branch sim_simHitDetId_branch does not exist!\n");
      exit(1);
    }
    sim_simHitDetId_isLoaded = true;
  }
  return *sim_simHitDetId_;
}
const vector<vector<int> > &SDL::sim_simHitLayer() {
  if (not sim_simHitLayer_isLoaded) {
    if (sim_simHitLayer_branch != 0) {
      sim_simHitLayer_branch->GetEntry(index);
    } else {
      printf("branch sim_simHitLayer_branch does not exist!\n");
      exit(1);
    }
    sim_simHitLayer_isLoaded = true;
  }
  return *sim_simHitLayer_;
}
const vector<vector<float> > &SDL::sim_simHitDistxyHelix() {
  if (not sim_simHitDistxyHelix_isLoaded) {
    if (sim_simHitDistxyHelix_branch != 0) {
      sim_simHitDistxyHelix_branch->GetEntry(index);
    } else {
      printf("branch sim_simHitDistxyHelix_branch does not exist!\n");
      exit(1);
    }
    sim_simHitDistxyHelix_isLoaded = true;
  }
  return *sim_simHitDistxyHelix_;
}
const vector<vector<float> > &SDL::sim_simHitLayerMinDistxyHelix() {
  if (not sim_simHitLayerMinDistxyHelix_isLoaded) {
    if (sim_simHitLayerMinDistxyHelix_branch != 0) {
      sim_simHitLayerMinDistxyHelix_branch->GetEntry(index);
    } else {
      printf("branch sim_simHitLayerMinDistxyHelix_branch does not exist!\n");
      exit(1);
    }
    sim_simHitLayerMinDistxyHelix_isLoaded = true;
  }
  return *sim_simHitLayerMinDistxyHelix_;
}
const vector<vector<float> > &SDL::sim_recoHitX() {
  if (not sim_recoHitX_isLoaded) {
    if (sim_recoHitX_branch != 0) {
      sim_recoHitX_branch->GetEntry(index);
    } else {
      printf("branch sim_recoHitX_branch does not exist!\n");
      exit(1);
    }
    sim_recoHitX_isLoaded = true;
  }
  return *sim_recoHitX_;
}
const vector<vector<float> > &SDL::sim_recoHitY() {
  if (not sim_recoHitY_isLoaded) {
    if (sim_recoHitY_branch != 0) {
      sim_recoHitY_branch->GetEntry(index);
    } else {
      printf("branch sim_recoHitY_branch does not exist!\n");
      exit(1);
    }
    sim_recoHitY_isLoaded = true;
  }
  return *sim_recoHitY_;
}
const vector<vector<float> > &SDL::sim_recoHitZ() {
  if (not sim_recoHitZ_isLoaded) {
    if (sim_recoHitZ_branch != 0) {
      sim_recoHitZ_branch->GetEntry(index);
    } else {
      printf("branch sim_recoHitZ_branch does not exist!\n");
      exit(1);
    }
    sim_recoHitZ_isLoaded = true;
  }
  return *sim_recoHitZ_;
}
const vector<vector<int> > &SDL::sim_recoHitDetId() {
  if (not sim_recoHitDetId_isLoaded) {
    if (sim_recoHitDetId_branch != 0) {
      sim_recoHitDetId_branch->GetEntry(index);
    } else {
      printf("branch sim_recoHitDetId_branch does not exist!\n");
      exit(1);
    }
    sim_recoHitDetId_isLoaded = true;
  }
  return *sim_recoHitDetId_;
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
const vector<int> &SDL::tc_pt5Idx() {
  if (not tc_pt5Idx_isLoaded) {
    if (tc_pt5Idx_branch != 0) {
      tc_pt5Idx_branch->GetEntry(index);
    } else {
      printf("branch tc_pt5Idx_branch does not exist!\n");
      exit(1);
    }
    tc_pt5Idx_isLoaded = true;
  }
  return *tc_pt5Idx_;
}
const vector<int> &SDL::tc_pt3Idx() {
  if (not tc_pt3Idx_isLoaded) {
    if (tc_pt3Idx_branch != 0) {
      tc_pt3Idx_branch->GetEntry(index);
    } else {
      printf("branch tc_pt3Idx_branch does not exist!\n");
      exit(1);
    }
    tc_pt3Idx_isLoaded = true;
  }
  return *tc_pt3Idx_;
}
const vector<int> &SDL::tc_t5Idx() {
  if (not tc_t5Idx_isLoaded) {
    if (tc_t5Idx_branch != 0) {
      tc_t5Idx_branch->GetEntry(index);
    } else {
      printf("branch tc_t5Idx_branch does not exist!\n");
      exit(1);
    }
    tc_t5Idx_isLoaded = true;
  }
  return *tc_t5Idx_;
}
const vector<int> &SDL::tc_plsIdx() {
  if (not tc_plsIdx_isLoaded) {
    if (tc_plsIdx_branch != 0) {
      tc_plsIdx_branch->GetEntry(index);
    } else {
      printf("branch tc_plsIdx_branch does not exist!\n");
      exit(1);
    }
    tc_plsIdx_isLoaded = true;
  }
  return *tc_plsIdx_;
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
const vector<int> &SDL::tc_simIdx() {
  if (not tc_simIdx_isLoaded) {
    if (tc_simIdx_branch != 0) {
      tc_simIdx_branch->GetEntry(index);
    } else {
      printf("branch tc_simIdx_branch does not exist!\n");
      exit(1);
    }
    tc_simIdx_isLoaded = true;
  }
  return *tc_simIdx_;
}
const vector<vector<int> > &SDL::tc_simIdxAll() {
  if (not tc_simIdxAll_isLoaded) {
    if (tc_simIdxAll_branch != 0) {
      tc_simIdxAll_branch->GetEntry(index);
    } else {
      printf("branch tc_simIdxAll_branch does not exist!\n");
      exit(1);
    }
    tc_simIdxAll_isLoaded = true;
  }
  return *tc_simIdxAll_;
}
const vector<vector<float> > &SDL::tc_simIdxAllFrac() {
  if (not tc_simIdxAllFrac_isLoaded) {
    if (tc_simIdxAllFrac_branch != 0) {
      tc_simIdxAllFrac_branch->GetEntry(index);
    } else {
      printf("branch tc_simIdxAllFrac_branch does not exist!\n");
      exit(1);
    }
    tc_simIdxAllFrac_isLoaded = true;
  }
  return *tc_simIdxAllFrac_;
}
const vector<float> &SDL::md_pt() {
  if (not md_pt_isLoaded) {
    if (md_pt_branch != 0) {
      md_pt_branch->GetEntry(index);
    } else {
      printf("branch md_pt_branch does not exist!\n");
      exit(1);
    }
    md_pt_isLoaded = true;
  }
  return *md_pt_;
}
const vector<float> &SDL::md_eta() {
  if (not md_eta_isLoaded) {
    if (md_eta_branch != 0) {
      md_eta_branch->GetEntry(index);
    } else {
      printf("branch md_eta_branch does not exist!\n");
      exit(1);
    }
    md_eta_isLoaded = true;
  }
  return *md_eta_;
}
const vector<float> &SDL::md_phi() {
  if (not md_phi_isLoaded) {
    if (md_phi_branch != 0) {
      md_phi_branch->GetEntry(index);
    } else {
      printf("branch md_phi_branch does not exist!\n");
      exit(1);
    }
    md_phi_isLoaded = true;
  }
  return *md_phi_;
}

#ifdef OUTPUT_MD_CUTS
const vector<float> &SDL::md_dphi() {
  if (not md_dphi_isLoaded) {
    if (md_dphi_branch != 0) {
      md_dphi_branch->GetEntry(index);
    } else {
      printf("branch md_dphi_branch does not exist!\n");
      exit(1);
    }
    md_dphi_isLoaded = true;
  }
  return *md_dphichange_;
}const vector<float> &SDL::md_dphichange() {
  if (not md_dphichange_isLoaded) {
    if (md_dphichange_branch != 0) {
      md_dphichange_branch->GetEntry(index);
    } else {
      printf("branch md_dphichange_branch does not exist!\n");
      exit(1);
    }
    md_dphichange_isLoaded = true;
  }
  return *md_dz_;
}const vector<float> &SDL::md_dz() {
  if (not md_dz_isLoaded) {
    if (md_dz_branch != 0) {
      md_dz_branch->GetEntry(index);
    } else {
      printf("branch md_dz_branch does not exist!\n");
      exit(1);
    }
    md_dz_isLoaded = true;
  }
  return *md_dz_;
}

#endif




const vector<float> &SDL::md_anchor_x() {
  if (not md_anchor_x_isLoaded) {
    if (md_anchor_x_branch != 0) {
      md_anchor_x_branch->GetEntry(index);
    } else {
      printf("branch md_anchor_x_branch does not exist!\n");
      exit(1);
    }
    md_anchor_x_isLoaded = true;
  }
  return *md_anchor_x_;
}
const vector<float> &SDL::md_anchor_y() {
  if (not md_anchor_y_isLoaded) {
    if (md_anchor_y_branch != 0) {
      md_anchor_y_branch->GetEntry(index);
    } else {
      printf("branch md_anchor_y_branch does not exist!\n");
      exit(1);
    }
    md_anchor_y_isLoaded = true;
  }
  return *md_anchor_y_;
}
const vector<float> &SDL::md_anchor_z() {
  if (not md_anchor_z_isLoaded) {
    if (md_anchor_z_branch != 0) {
      md_anchor_z_branch->GetEntry(index);
    } else {
      printf("branch md_anchor_z_branch does not exist!\n");
      exit(1);
    }
    md_anchor_z_isLoaded = true;
  }
  return *md_anchor_z_;
}
const vector<float> &SDL::md_other_x() {
  if (not md_other_x_isLoaded) {
    if (md_other_x_branch != 0) {
      md_other_x_branch->GetEntry(index);
    } else {
      printf("branch md_other_x_branch does not exist!\n");
      exit(1);
    }
    md_other_x_isLoaded = true;
  }
  return *md_other_x_;
}
const vector<float> &SDL::md_other_y() {
  if (not md_other_y_isLoaded) {
    if (md_other_y_branch != 0) {
      md_other_y_branch->GetEntry(index);
    } else {
      printf("branch md_other_y_branch does not exist!\n");
      exit(1);
    }
    md_other_y_isLoaded = true;
  }
  return *md_other_y_;
}
const vector<float> &SDL::md_other_z() {
  if (not md_other_z_isLoaded) {
    if (md_other_z_branch != 0) {
      md_other_z_branch->GetEntry(index);
    } else {
      printf("branch md_other_z_branch does not exist!\n");
      exit(1);
    }
    md_other_z_isLoaded = true;
  }
  return *md_other_z_;
}
const vector<int> &SDL::md_type() {
  if (not md_type_isLoaded) {
    if (md_type_branch != 0) {
      md_type_branch->GetEntry(index);
    } else {
      printf("branch md_type_branch does not exist!\n");
      exit(1);
    }
    md_type_isLoaded = true;
  }
  return *md_type_;
}
const vector<int> &SDL::md_layer() {
  if (not md_layer_isLoaded) {
    if (md_layer_branch != 0) {
      md_layer_branch->GetEntry(index);
    } else {
      printf("branch md_layer_branch does not exist!\n");
      exit(1);
    }
    md_layer_isLoaded = true;
  }
  return *md_layer_;
}
const vector<int> &SDL::md_detId() {
  if (not md_detId_isLoaded) {
    if (md_detId_branch != 0) {
      md_detId_branch->GetEntry(index);
    } else {
      printf("branch md_detId_branch does not exist!\n");
      exit(1);
    }
    md_detId_isLoaded = true;
  }
  return *md_detId_;
}
const vector<int> &SDL::md_isFake() {
  if (not md_isFake_isLoaded) {
    if (md_isFake_branch != 0) {
      md_isFake_branch->GetEntry(index);
    } else {
      printf("branch md_isFake_branch does not exist!\n");
      exit(1);
    }
    md_isFake_isLoaded = true;
  }
  return *md_isFake_;
}
const vector<int> &SDL::md_simIdx() {
  if (not md_simIdx_isLoaded) {
    if (md_simIdx_branch != 0) {
      md_simIdx_branch->GetEntry(index);
    } else {
      printf("branch md_simIdx_branch does not exist!\n");
      exit(1);
    }
    md_simIdx_isLoaded = true;
  }
  return *md_simIdx_;
}
const vector<vector<int> > &SDL::md_simIdxAll() {
  if (not md_simIdxAll_isLoaded) {
    if (md_simIdxAll_branch != 0) {
      md_simIdxAll_branch->GetEntry(index);
    } else {
      printf("branch md_simIdxAll_branch does not exist!\n");
      exit(1);
    }
    md_simIdxAll_isLoaded = true;
  }
  return *md_simIdxAll_;
}
const vector<vector<float> > &SDL::md_simIdxAllFrac() {
  if (not md_simIdxAllFrac_isLoaded) {
    if (md_simIdxAllFrac_branch != 0) {
      md_simIdxAllFrac_branch->GetEntry(index);
    } else {
      printf("branch md_simIdxAllFrac_branch does not exist!\n");
      exit(1);
    }
    md_simIdxAllFrac_isLoaded = true;
  }
  return *md_simIdxAllFrac_;
}
const vector<float> &SDL::ls_pt() {
  if (not ls_pt_isLoaded) {
    if (ls_pt_branch != 0) {
      ls_pt_branch->GetEntry(index);
    } else {
      printf("branch ls_pt_branch does not exist!\n");
      exit(1);
    }
    ls_pt_isLoaded = true;
  }
  return *ls_pt_;
}
const vector<float> &SDL::ls_eta() {
  if (not ls_eta_isLoaded) {
    if (ls_eta_branch != 0) {
      ls_eta_branch->GetEntry(index);
    } else {
      printf("branch ls_eta_branch does not exist!\n");
      exit(1);
    }
    ls_eta_isLoaded = true;
  }
  return *ls_eta_;
}
const vector<float> &SDL::ls_phi() {
  if (not ls_phi_isLoaded) {
    if (ls_phi_branch != 0) {
      ls_phi_branch->GetEntry(index);
    } else {
      printf("branch ls_phi_branch does not exist!\n");
      exit(1);
    }
    ls_phi_isLoaded = true;
  }
  return *ls_phi_;
}
const vector<int> &SDL::ls_mdIdx0() {
  if (not ls_mdIdx0_isLoaded) {
    if (ls_mdIdx0_branch != 0) {
      ls_mdIdx0_branch->GetEntry(index);
    } else {
      printf("branch ls_mdIdx0_branch does not exist!\n");
      exit(1);
    }
    ls_mdIdx0_isLoaded = true;
  }
  return *ls_mdIdx0_;
}
const vector<int> &SDL::ls_mdIdx1() {
  if (not ls_mdIdx1_isLoaded) {
    if (ls_mdIdx1_branch != 0) {
      ls_mdIdx1_branch->GetEntry(index);
    } else {
      printf("branch ls_mdIdx1_branch does not exist!\n");
      exit(1);
    }
    ls_mdIdx1_isLoaded = true;
  }
  return *ls_mdIdx1_;
}
const vector<int> &SDL::ls_isFake() {
  if (not ls_isFake_isLoaded) {
    if (ls_isFake_branch != 0) {
      ls_isFake_branch->GetEntry(index);
    } else {
      printf("branch ls_isFake_branch does not exist!\n");
      exit(1);
    }
    ls_isFake_isLoaded = true;
  }
  return *ls_isFake_;
}
const vector<int> &SDL::ls_simIdx() {
  if (not ls_simIdx_isLoaded) {
    if (ls_simIdx_branch != 0) {
      ls_simIdx_branch->GetEntry(index);
    } else {
      printf("branch ls_simIdx_branch does not exist!\n");
      exit(1);
    }
    ls_simIdx_isLoaded = true;
  }
  return *ls_simIdx_;
}

#ifdef OUTPUT_LS_CUTS
const vector<int> &SDL::ls_zLos() {
  if (not ls_zLos_isLoaded) {
    if (ls_zLos_branch != 0) {
      ls_zLos_branch->GetEntry(index);
    } else {
      printf("branch ls_zLos_branch does not exist!\n");
      exit(1);
    }
    ls_zLos_isLoaded = true;
  }
  return *ls_zLos_;
}const vector<int> &SDL::ls_zHis() {
  if (not ls_zHis_isLoaded) {
    if (ls_zHis_branch != 0) {
      ls_zHis_branch->GetEntry(index);
    } else {
      printf("branch ls_zHis_branch does not exist!\n");
      exit(1);
    }
    ls_zHis_isLoaded = true;
  }
  return *ls_zHis_;
}const vector<int> &SDL::ls_rtLos() {
  if (not ls_rtLos_isLoaded) {
    if (ls_rtLos_branch != 0) {
      ls_rtLos_branch->GetEntry(index);
    } else {
      printf("branch ls_rtLos_branch does not exist!\n");
      exit(1);
    }
    ls_rtLos_isLoaded = true;
  }
  return *ls_rtLos_;
}const vector<int> &SDL::ls_rtHis() {
  if (not ls_rtHis_isLoaded) {
    if (ls_rtHis_branch != 0) {
      ls_rtHis_branch->GetEntry(index);
    } else {
      printf("branch ls_rtHis_branch does not exist!\n");
      exit(1);
    }
    ls_rtHis_isLoaded = true;
  }
  return *ls_rtHis_;
}const vector<int> &SDL::ls_dPhis() {
  if (not ls_dPhis_isLoaded) {
    if (ls_dPhis_branch != 0) {
      ls_dPhis_branch->GetEntry(index);
    } else {
      printf("branch ls_dPhis_branch does not exist!\n");
      exit(1);
    }
    ls_dPhis_isLoaded = true;
  }
  return *ls_dPhis_;
}const vector<int> &SDL::ls_dPhiMins() {
  if (not ls_dPhiMins_isLoaded) {
    if (ls_dPhiMins_branch != 0) {
      ls_dPhiMins_branch->GetEntry(index);
    } else {
      printf("branch ls_dPhiMins_branch does not exist!\n");
      exit(1);
    }
    ls_dPhiMins_isLoaded = true;
  }
  return *ls_dPhiMins_;
}const vector<int> &SDL::ls_dPhiMaxs() {
  if (not ls_dPhiMaxs_isLoaded) {
    if (ls_dPhiMaxs_branch != 0) {
      ls_dPhiMaxs_branch->GetEntry(index);
    } else {
      printf("branch ls_dPhiMaxs_branch does not exist!\n");
      exit(1);
    }
    ls_dPhiMaxs_isLoaded = true;
  }
  return *ls_dPhiMaxs_;
}const vector<int> &SDL::ls_dPhiChanges() {
  if (not ls_dPhiChanges_isLoaded) {
    if (ls_dPhiChanges_branch != 0) {
      ls_dPhiChanges_branch->GetEntry(index);
    } else {
      printf("branch ls_dPhiChanges_branch does not exist!\n");
      exit(1);
    }
    ls_dPhiChanges_isLoaded = true;
  }
  return *ls_dPhiChanges_;
}const vector<int> &SDL::ls_dPhiChangeMins() {
  if (not ls_dPhiChangeMins_isLoaded) {
    if (ls_dPhiChangeMins_branch != 0) {
      ls_dPhiChangeMins_branch->GetEntry(index);
    } else {
      printf("branch ls_dPhiChangeMins_branch does not exist!\n");
      exit(1);
    }
    ls_dPhiChangeMins_isLoaded = true;
  }
  return *ls_dPhiChangeMins_;
}const vector<int> &SDL::ls_dPhiChangeMaxs() {
  if (not ls_dPhiChangeMaxs_isLoaded) {
    if (ls_dPhiChangeMaxs_branch != 0) {
      ls_dPhiChangeMaxs_branch->GetEntry(index);
    } else {
      printf("branch ls_dPhiChangeMaxs_branch does not exist!\n");
      exit(1);
    }
    ls_dPhiChangeMaxs_isLoaded = true;
  }
  return *ls_dPhiChangeMaxs_;
}const vector<int> &SDL::ls_dAlphaInners() {
  if (not ls_dAlphaInners_isLoaded) {
    if (ls_dAlphaInners_branch != 0) {
      ls_dAlphaInners_branch->GetEntry(index);
    } else {
      printf("branch ls_dAlphaInners_branch does not exist!\n");
      exit(1);
    }
    ls_dAlphaInners_isLoaded = true;
  }
  return *ls_dAlphaInners_;
}const vector<int> &SDL::ls_dAlphaOuters() {
  if (not ls_dAlphaOuters_isLoaded) {
    if (ls_dAlphaOuters_branch != 0) {
      ls_dAlphaOuters_branch->GetEntry(index);
    } else {
      printf("branch ls_dAlphaOuters_branch does not exist!\n");
      exit(1);
    }
    ls_dAlphaOuters_isLoaded = true;
  }
  return *ls_dAlphaOuters_;
}const vector<int> &SDL::ls_dAlphaInnerOuters() {
  if (not ls_dAlphaInnerOuters_isLoaded) {
    if (ls_dAlphaInnerOuters_branch != 0) {
      ls_dAlphaInnerOuters_branch->GetEntry(index);
    } else {
      printf("branch ls_dAlphaInnerOuters_branch does not exist!\n");
      exit(1);
    }
    ls_dAlphaInnerOuters_isLoaded = true;
  }
  return *ls_dAlphaInnerOuters_;
}
#endif

const vector<vector<int> > &SDL::ls_simIdxAll() {
  if (not ls_simIdxAll_isLoaded) {
    if (ls_simIdxAll_branch != 0) {
      ls_simIdxAll_branch->GetEntry(index);
    } else {
      printf("branch ls_simIdxAll_branch does not exist!\n");
      exit(1);
    }
    ls_simIdxAll_isLoaded = true;
  }
  return *ls_simIdxAll_;
}
const vector<vector<float> > &SDL::ls_simIdxAllFrac() {
  if (not ls_simIdxAllFrac_isLoaded) {
    if (ls_simIdxAllFrac_branch != 0) {
      ls_simIdxAllFrac_branch->GetEntry(index);
    } else {
      printf("branch ls_simIdxAllFrac_branch does not exist!\n");
      exit(1);
    }
    ls_simIdxAllFrac_isLoaded = true;
  }
  return *ls_simIdxAllFrac_;
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
const vector<int> &SDL::t3_lsIdx0() {
  if (not t3_lsIdx0_isLoaded) {
    if (t3_lsIdx0_branch != 0) {
      t3_lsIdx0_branch->GetEntry(index);
    } else {
      printf("branch t3_lsIdx0_branch does not exist!\n");
      exit(1);
    }
    t3_lsIdx0_isLoaded = true;
  }
  return *t3_lsIdx0_;
}
const vector<int> &SDL::t3_lsIdx1() {
  if (not t3_lsIdx1_isLoaded) {
    if (t3_lsIdx1_branch != 0) {
      t3_lsIdx1_branch->GetEntry(index);
    } else {
      printf("branch t3_lsIdx1_branch does not exist!\n");
      exit(1);
    }
    t3_lsIdx1_isLoaded = true;
  }
  return *t3_lsIdx1_;
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
const vector<int> &SDL::t3_simIdx() {
  if (not t3_simIdx_isLoaded) {
    if (t3_simIdx_branch != 0) {
      t3_simIdx_branch->GetEntry(index);
    } else {
      printf("branch t3_simIdx_branch does not exist!\n");
      exit(1);
    }
    t3_simIdx_isLoaded = true;
  }
  return *t3_simIdx_;
}
const vector<vector<int> > &SDL::t3_simIdxAll() {
  if (not t3_simIdxAll_isLoaded) {
    if (t3_simIdxAll_branch != 0) {
      t3_simIdxAll_branch->GetEntry(index);
    } else {
      printf("branch t3_simIdxAll_branch does not exist!\n");
      exit(1);
    }
    t3_simIdxAll_isLoaded = true;
  }
  return *t3_simIdxAll_;
}
const vector<vector<float> > &SDL::t3_simIdxAllFrac() {
  if (not t3_simIdxAllFrac_isLoaded) {
    if (t3_simIdxAllFrac_branch != 0) {
      t3_simIdxAllFrac_branch->GetEntry(index);
    } else {
      printf("branch t3_simIdxAllFrac_branch does not exist!\n");
      exit(1);
    }
    t3_simIdxAllFrac_isLoaded = true;
  }
  return *t3_simIdxAllFrac_;
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
const vector<int> &SDL::t5_t3Idx0() {
  if (not t5_t3Idx0_isLoaded) {
    if (t5_t3Idx0_branch != 0) {
      t5_t3Idx0_branch->GetEntry(index);
    } else {
      printf("branch t5_t3Idx0_branch does not exist!\n");
      exit(1);
    }
    t5_t3Idx0_isLoaded = true;
  }
  return *t5_t3Idx0_;
}
const vector<int> &SDL::t5_t3Idx1() {
  if (not t5_t3Idx1_isLoaded) {
    if (t5_t3Idx1_branch != 0) {
      t5_t3Idx1_branch->GetEntry(index);
    } else {
      printf("branch t5_t3Idx1_branch does not exist!\n");
      exit(1);
    }
    t5_t3Idx1_isLoaded = true;
  }
  return *t5_t3Idx1_;
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
const vector<int> &SDL::t5_simIdx() {
  if (not t5_simIdx_isLoaded) {
    if (t5_simIdx_branch != 0) {
      t5_simIdx_branch->GetEntry(index);
    } else {
      printf("branch t5_simIdx_branch does not exist!\n");
      exit(1);
    }
    t5_simIdx_isLoaded = true;
  }
  return *t5_simIdx_;
}
const vector<vector<int> > &SDL::t5_simIdxAll() {
  if (not t5_simIdxAll_isLoaded) {
    if (t5_simIdxAll_branch != 0) {
      t5_simIdxAll_branch->GetEntry(index);
    } else {
      printf("branch t5_simIdxAll_branch does not exist!\n");
      exit(1);
    }
    t5_simIdxAll_isLoaded = true;
  }
  return *t5_simIdxAll_;
}
const vector<vector<float> > &SDL::t5_simIdxAllFrac() {
  if (not t5_simIdxAllFrac_isLoaded) {
    if (t5_simIdxAllFrac_branch != 0) {
      t5_simIdxAllFrac_branch->GetEntry(index);
    } else {
      printf("branch t5_simIdxAllFrac_branch does not exist!\n");
      exit(1);
    }
    t5_simIdxAllFrac_isLoaded = true;
  }
  return *t5_simIdxAllFrac_;
}
const vector<float> &SDL::pls_pt() {
  if (not pls_pt_isLoaded) {
    if (pls_pt_branch != 0) {
      pls_pt_branch->GetEntry(index);
    } else {
      printf("branch pls_pt_branch does not exist!\n");
      exit(1);
    }
    pls_pt_isLoaded = true;
  }
  return *pls_pt_;
}
const vector<float> &SDL::pls_eta() {
  if (not pls_eta_isLoaded) {
    if (pls_eta_branch != 0) {
      pls_eta_branch->GetEntry(index);
    } else {
      printf("branch pls_eta_branch does not exist!\n");
      exit(1);
    }
    pls_eta_isLoaded = true;
  }
  return *pls_eta_;
}
const vector<float> &SDL::pls_phi() {
  if (not pls_phi_isLoaded) {
    if (pls_phi_branch != 0) {
      pls_phi_branch->GetEntry(index);
    } else {
      printf("branch pls_phi_branch does not exist!\n");
      exit(1);
    }
    pls_phi_isLoaded = true;
  }
  return *pls_phi_;
}
const vector<int> &SDL::pls_nhit() {
  if (not pls_nhit_isLoaded) {
    if (pls_nhit_branch != 0) {
      pls_nhit_branch->GetEntry(index);
    } else {
      printf("branch pls_nhit_branch does not exist!\n");
      exit(1);
    }
    pls_nhit_isLoaded = true;
  }
  return *pls_nhit_;
}
const vector<float> &SDL::pls_hit0_x() {
  if (not pls_hit0_x_isLoaded) {
    if (pls_hit0_x_branch != 0) {
      pls_hit0_x_branch->GetEntry(index);
    } else {
      printf("branch pls_hit0_x_branch does not exist!\n");
      exit(1);
    }
    pls_hit0_x_isLoaded = true;
  }
  return *pls_hit0_x_;
}
const vector<float> &SDL::pls_hit0_y() {
  if (not pls_hit0_y_isLoaded) {
    if (pls_hit0_y_branch != 0) {
      pls_hit0_y_branch->GetEntry(index);
    } else {
      printf("branch pls_hit0_y_branch does not exist!\n");
      exit(1);
    }
    pls_hit0_y_isLoaded = true;
  }
  return *pls_hit0_y_;
}
const vector<float> &SDL::pls_hit0_z() {
  if (not pls_hit0_z_isLoaded) {
    if (pls_hit0_z_branch != 0) {
      pls_hit0_z_branch->GetEntry(index);
    } else {
      printf("branch pls_hit0_z_branch does not exist!\n");
      exit(1);
    }
    pls_hit0_z_isLoaded = true;
  }
  return *pls_hit0_z_;
}
const vector<float> &SDL::pls_hit1_x() {
  if (not pls_hit1_x_isLoaded) {
    if (pls_hit1_x_branch != 0) {
      pls_hit1_x_branch->GetEntry(index);
    } else {
      printf("branch pls_hit1_x_branch does not exist!\n");
      exit(1);
    }
    pls_hit1_x_isLoaded = true;
  }
  return *pls_hit1_x_;
}
const vector<float> &SDL::pls_hit1_y() {
  if (not pls_hit1_y_isLoaded) {
    if (pls_hit1_y_branch != 0) {
      pls_hit1_y_branch->GetEntry(index);
    } else {
      printf("branch pls_hit1_y_branch does not exist!\n");
      exit(1);
    }
    pls_hit1_y_isLoaded = true;
  }
  return *pls_hit1_y_;
}
const vector<float> &SDL::pls_hit1_z() {
  if (not pls_hit1_z_isLoaded) {
    if (pls_hit1_z_branch != 0) {
      pls_hit1_z_branch->GetEntry(index);
    } else {
      printf("branch pls_hit1_z_branch does not exist!\n");
      exit(1);
    }
    pls_hit1_z_isLoaded = true;
  }
  return *pls_hit1_z_;
}
const vector<float> &SDL::pls_hit2_x() {
  if (not pls_hit2_x_isLoaded) {
    if (pls_hit2_x_branch != 0) {
      pls_hit2_x_branch->GetEntry(index);
    } else {
      printf("branch pls_hit2_x_branch does not exist!\n");
      exit(1);
    }
    pls_hit2_x_isLoaded = true;
  }
  return *pls_hit2_x_;
}
const vector<float> &SDL::pls_hit2_y() {
  if (not pls_hit2_y_isLoaded) {
    if (pls_hit2_y_branch != 0) {
      pls_hit2_y_branch->GetEntry(index);
    } else {
      printf("branch pls_hit2_y_branch does not exist!\n");
      exit(1);
    }
    pls_hit2_y_isLoaded = true;
  }
  return *pls_hit2_y_;
}
const vector<float> &SDL::pls_hit2_z() {
  if (not pls_hit2_z_isLoaded) {
    if (pls_hit2_z_branch != 0) {
      pls_hit2_z_branch->GetEntry(index);
    } else {
      printf("branch pls_hit2_z_branch does not exist!\n");
      exit(1);
    }
    pls_hit2_z_isLoaded = true;
  }
  return *pls_hit2_z_;
}
const vector<float> &SDL::pls_hit3_x() {
  if (not pls_hit3_x_isLoaded) {
    if (pls_hit3_x_branch != 0) {
      pls_hit3_x_branch->GetEntry(index);
    } else {
      printf("branch pls_hit3_x_branch does not exist!\n");
      exit(1);
    }
    pls_hit3_x_isLoaded = true;
  }
  return *pls_hit3_x_;
}
const vector<float> &SDL::pls_hit3_y() {
  if (not pls_hit3_y_isLoaded) {
    if (pls_hit3_y_branch != 0) {
      pls_hit3_y_branch->GetEntry(index);
    } else {
      printf("branch pls_hit3_y_branch does not exist!\n");
      exit(1);
    }
    pls_hit3_y_isLoaded = true;
  }
  return *pls_hit3_y_;
}
const vector<float> &SDL::pls_hit3_z() {
  if (not pls_hit3_z_isLoaded) {
    if (pls_hit3_z_branch != 0) {
      pls_hit3_z_branch->GetEntry(index);
    } else {
      printf("branch pls_hit3_z_branch does not exist!\n");
      exit(1);
    }
    pls_hit3_z_isLoaded = true;
  }
  return *pls_hit3_z_;
}
const vector<int> &SDL::pls_isFake() {
  if (not pls_isFake_isLoaded) {
    if (pls_isFake_branch != 0) {
      pls_isFake_branch->GetEntry(index);
    } else {
      printf("branch pls_isFake_branch does not exist!\n");
      exit(1);
    }
    pls_isFake_isLoaded = true;
  }
  return *pls_isFake_;
}
const vector<int> &SDL::pls_isDuplicate() {
  if (not pls_isDuplicate_isLoaded) {
    if (pls_isDuplicate_branch != 0) {
      pls_isDuplicate_branch->GetEntry(index);
    } else {
      printf("branch pls_isDuplicate_branch does not exist!\n");
      exit(1);
    }
    pls_isDuplicate_isLoaded = true;
  }
  return *pls_isDuplicate_;
}
const vector<int> &SDL::pls_simIdx() {
  if (not pls_simIdx_isLoaded) {
    if (pls_simIdx_branch != 0) {
      pls_simIdx_branch->GetEntry(index);
    } else {
      printf("branch pls_simIdx_branch does not exist!\n");
      exit(1);
    }
    pls_simIdx_isLoaded = true;
  }
  return *pls_simIdx_;
}
const vector<vector<int> > &SDL::pls_simIdxAll() {
  if (not pls_simIdxAll_isLoaded) {
    if (pls_simIdxAll_branch != 0) {
      pls_simIdxAll_branch->GetEntry(index);
    } else {
      printf("branch pls_simIdxAll_branch does not exist!\n");
      exit(1);
    }
    pls_simIdxAll_isLoaded = true;
  }
  return *pls_simIdxAll_;
}
const vector<vector<float> > &SDL::pls_simIdxAllFrac() {
  if (not pls_simIdxAllFrac_isLoaded) {
    if (pls_simIdxAllFrac_branch != 0) {
      pls_simIdxAllFrac_branch->GetEntry(index);
    } else {
      printf("branch pls_simIdxAllFrac_branch does not exist!\n");
      exit(1);
    }
    pls_simIdxAllFrac_isLoaded = true;
  }
  return *pls_simIdxAllFrac_;
}
const vector<float> &SDL::pt3_pt() {
  if (not pt3_pt_isLoaded) {
    if (pt3_pt_branch != 0) {
      pt3_pt_branch->GetEntry(index);
    } else {
      printf("branch pt3_pt_branch does not exist!\n");
      exit(1);
    }
    pt3_pt_isLoaded = true;
  }
  return *pt3_pt_;
}
const vector<float> &SDL::pt3_eta() {
  if (not pt3_eta_isLoaded) {
    if (pt3_eta_branch != 0) {
      pt3_eta_branch->GetEntry(index);
    } else {
      printf("branch pt3_eta_branch does not exist!\n");
      exit(1);
    }
    pt3_eta_isLoaded = true;
  }
  return *pt3_eta_;
}
const vector<float> &SDL::pt3_phi() {
  if (not pt3_phi_isLoaded) {
    if (pt3_phi_branch != 0) {
      pt3_phi_branch->GetEntry(index);
    } else {
      printf("branch pt3_phi_branch does not exist!\n");
      exit(1);
    }
    pt3_phi_isLoaded = true;
  }
  return *pt3_phi_;
}
const vector<int> &SDL::pt3_plsIdx() {
  if (not pt3_plsIdx_isLoaded) {
    if (pt3_plsIdx_branch != 0) {
      pt3_plsIdx_branch->GetEntry(index);
    } else {
      printf("branch pt3_plsIdx_branch does not exist!\n");
      exit(1);
    }
    pt3_plsIdx_isLoaded = true;
  }
  return *pt3_plsIdx_;
}
const vector<int> &SDL::pt3_t3Idx() {
  if (not pt3_t3Idx_isLoaded) {
    if (pt3_t3Idx_branch != 0) {
      pt3_t3Idx_branch->GetEntry(index);
    } else {
      printf("branch pt3_t3Idx_branch does not exist!\n");
      exit(1);
    }
    pt3_t3Idx_isLoaded = true;
  }
  return *pt3_t3Idx_;
}
const vector<int> &SDL::pt3_isFake() {
  if (not pt3_isFake_isLoaded) {
    if (pt3_isFake_branch != 0) {
      pt3_isFake_branch->GetEntry(index);
    } else {
      printf("branch pt3_isFake_branch does not exist!\n");
      exit(1);
    }
    pt3_isFake_isLoaded = true;
  }
  return *pt3_isFake_;
}
const vector<int> &SDL::pt3_isDuplicate() {
  if (not pt3_isDuplicate_isLoaded) {
    if (pt3_isDuplicate_branch != 0) {
      pt3_isDuplicate_branch->GetEntry(index);
    } else {
      printf("branch pt3_isDuplicate_branch does not exist!\n");
      exit(1);
    }
    pt3_isDuplicate_isLoaded = true;
  }
  return *pt3_isDuplicate_;
}
const vector<int> &SDL::pt3_simIdx() {
  if (not pt3_simIdx_isLoaded) {
    if (pt3_simIdx_branch != 0) {
      pt3_simIdx_branch->GetEntry(index);
    } else {
      printf("branch pt3_simIdx_branch does not exist!\n");
      exit(1);
    }
    pt3_simIdx_isLoaded = true;
  }
  return *pt3_simIdx_;
}
const vector<vector<int> > &SDL::pt3_simIdxAll() {
  if (not pt3_simIdxAll_isLoaded) {
    if (pt3_simIdxAll_branch != 0) {
      pt3_simIdxAll_branch->GetEntry(index);
    } else {
      printf("branch pt3_simIdxAll_branch does not exist!\n");
      exit(1);
    }
    pt3_simIdxAll_isLoaded = true;
  }
  return *pt3_simIdxAll_;
}
const vector<vector<float> > &SDL::pt3_simIdxAllFrac() {
  if (not pt3_simIdxAllFrac_isLoaded) {
    if (pt3_simIdxAllFrac_branch != 0) {
      pt3_simIdxAllFrac_branch->GetEntry(index);
    } else {
      printf("branch pt3_simIdxAllFrac_branch does not exist!\n");
      exit(1);
    }
    pt3_simIdxAllFrac_isLoaded = true;
  }
  return *pt3_simIdxAllFrac_;
}
const vector<float> &SDL::pt5_pt() {
  if (not pt5_pt_isLoaded) {
    if (pt5_pt_branch != 0) {
      pt5_pt_branch->GetEntry(index);
    } else {
      printf("branch pt5_pt_branch does not exist!\n");
      exit(1);
    }
    pt5_pt_isLoaded = true;
  }
  return *pt5_pt_;
}
const vector<float> &SDL::pt5_eta() {
  if (not pt5_eta_isLoaded) {
    if (pt5_eta_branch != 0) {
      pt5_eta_branch->GetEntry(index);
    } else {
      printf("branch pt5_eta_branch does not exist!\n");
      exit(1);
    }
    pt5_eta_isLoaded = true;
  }
  return *pt5_eta_;
}
const vector<float> &SDL::pt5_phi() {
  if (not pt5_phi_isLoaded) {
    if (pt5_phi_branch != 0) {
      pt5_phi_branch->GetEntry(index);
    } else {
      printf("branch pt5_phi_branch does not exist!\n");
      exit(1);
    }
    pt5_phi_isLoaded = true;
  }
  return *pt5_phi_;
}
const vector<int> &SDL::pt5_plsIdx() {
  if (not pt5_plsIdx_isLoaded) {
    if (pt5_plsIdx_branch != 0) {
      pt5_plsIdx_branch->GetEntry(index);
    } else {
      printf("branch pt5_plsIdx_branch does not exist!\n");
      exit(1);
    }
    pt5_plsIdx_isLoaded = true;
  }
  return *pt5_plsIdx_;
}
const vector<int> &SDL::pt5_t5Idx() {
  if (not pt5_t5Idx_isLoaded) {
    if (pt5_t5Idx_branch != 0) {
      pt5_t5Idx_branch->GetEntry(index);
    } else {
      printf("branch pt5_t5Idx_branch does not exist!\n");
      exit(1);
    }
    pt5_t5Idx_isLoaded = true;
  }
  return *pt5_t5Idx_;
}
const vector<int> &SDL::pt5_isFake() {
  if (not pt5_isFake_isLoaded) {
    if (pt5_isFake_branch != 0) {
      pt5_isFake_branch->GetEntry(index);
    } else {
      printf("branch pt5_isFake_branch does not exist!\n");
      exit(1);
    }
    pt5_isFake_isLoaded = true;
  }
  return *pt5_isFake_;
}
const vector<int> &SDL::pt5_isDuplicate() {
  if (not pt5_isDuplicate_isLoaded) {
    if (pt5_isDuplicate_branch != 0) {
      pt5_isDuplicate_branch->GetEntry(index);
    } else {
      printf("branch pt5_isDuplicate_branch does not exist!\n");
      exit(1);
    }
    pt5_isDuplicate_isLoaded = true;
  }
  return *pt5_isDuplicate_;
}
const vector<int> &SDL::pt5_simIdx() {
  if (not pt5_simIdx_isLoaded) {
    if (pt5_simIdx_branch != 0) {
      pt5_simIdx_branch->GetEntry(index);
    } else {
      printf("branch pt5_simIdx_branch does not exist!\n");
      exit(1);
    }
    pt5_simIdx_isLoaded = true;
  }
  return *pt5_simIdx_;
}
const vector<vector<int> > &SDL::pt5_simIdxAll() {
  if (not pt5_simIdxAll_isLoaded) {
    if (pt5_simIdxAll_branch != 0) {
      pt5_simIdxAll_branch->GetEntry(index);
    } else {
      printf("branch pt5_simIdxAll_branch does not exist!\n");
      exit(1);
    }
    pt5_simIdxAll_isLoaded = true;
  }
  return *pt5_simIdxAll_;
}
const vector<vector<float> > &SDL::pt5_simIdxAllFrac() {
  if (not pt5_simIdxAllFrac_isLoaded) {
    if (pt5_simIdxAllFrac_branch != 0) {
      pt5_simIdxAllFrac_branch->GetEntry(index);
    } else {
      printf("branch pt5_simIdxAllFrac_branch does not exist!\n");
      exit(1);
    }
    pt5_simIdxAllFrac_isLoaded = true;
  }
  return *pt5_simIdxAllFrac_;
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
  const vector<float> &sim_pt() { return sdl.sim_pt(); }
  const vector<float> &sim_eta() { return sdl.sim_eta(); }
  const vector<float> &sim_phi() { return sdl.sim_phi(); }
  const vector<float> &sim_pca_dxy() { return sdl.sim_pca_dxy(); }
  const vector<float> &sim_pca_dz() { return sdl.sim_pca_dz(); }
  const vector<int> &sim_q() { return sdl.sim_q(); }
  const vector<int> &sim_pdgId() { return sdl.sim_pdgId(); }
  const vector<float> &sim_vx() { return sdl.sim_vx(); }
  const vector<float> &sim_vy() { return sdl.sim_vy(); }
  const vector<float> &sim_vz() { return sdl.sim_vz(); }
  const vector<float> &sim_vtxperp() { return sdl.sim_vtxperp(); }
  const vector<float> &sim_trkNtupIdx() { return sdl.sim_trkNtupIdx(); }
  const vector<int> &sim_tcIdx() { return sdl.sim_tcIdx(); }
  const vector<vector<int> > &sim_tcIdxAll() { return sdl.sim_tcIdxAll(); }
  const vector<vector<float> > &sim_tcIdxAllFrac() { return sdl.sim_tcIdxAllFrac(); }
  const vector<vector<int> > &sim_mdIdxAll() { return sdl.sim_mdIdxAll(); }
  const vector<vector<float> > &sim_mdIdxAllFrac() { return sdl.sim_mdIdxAllFrac(); }
  const vector<vector<int> > &sim_lsIdxAll() { return sdl.sim_lsIdxAll(); }
  const vector<vector<float> > &sim_lsIdxAllFrac() { return sdl.sim_lsIdxAllFrac(); }
  const vector<vector<int> > &sim_t3IdxAll() { return sdl.sim_t3IdxAll(); }
  const vector<vector<float> > &sim_t3IdxAllFrac() { return sdl.sim_t3IdxAllFrac(); }
  const vector<vector<int> > &sim_t5IdxAll() { return sdl.sim_t5IdxAll(); }
  const vector<vector<float> > &sim_t5IdxAllFrac() { return sdl.sim_t5IdxAllFrac(); }
  const vector<vector<int> > &sim_plsIdxAll() { return sdl.sim_plsIdxAll(); }
  const vector<vector<float> > &sim_plsIdxAllFrac() { return sdl.sim_plsIdxAllFrac(); }
  const vector<vector<int> > &sim_pt3IdxAll() { return sdl.sim_pt3IdxAll(); }
  const vector<vector<float> > &sim_pt3IdxAllFrac() { return sdl.sim_pt3IdxAllFrac(); }
  const vector<vector<int> > &sim_pt5IdxAll() { return sdl.sim_pt5IdxAll(); }
  const vector<vector<float> > &sim_pt5IdxAllFrac() { return sdl.sim_pt5IdxAllFrac(); }
  const vector<vector<float> > &sim_simHitX() { return sdl.sim_simHitX(); }
  const vector<vector<float> > &sim_simHitY() { return sdl.sim_simHitY(); }
  const vector<vector<float> > &sim_simHitZ() { return sdl.sim_simHitZ(); }
  const vector<vector<int> > &sim_simHitDetId() { return sdl.sim_simHitDetId(); }
  const vector<vector<int> > &sim_simHitLayer() { return sdl.sim_simHitLayer(); }
  const vector<vector<float> > &sim_simHitDistxyHelix() { return sdl.sim_simHitDistxyHelix(); }
  const vector<vector<float> > &sim_simHitLayerMinDistxyHelix() { return sdl.sim_simHitLayerMinDistxyHelix(); }
  const vector<vector<float> > &sim_recoHitX() { return sdl.sim_recoHitX(); }
  const vector<vector<float> > &sim_recoHitY() { return sdl.sim_recoHitY(); }
  const vector<vector<float> > &sim_recoHitZ() { return sdl.sim_recoHitZ(); }
  const vector<vector<int> > &sim_recoHitDetId() { return sdl.sim_recoHitDetId(); }
  const vector<float> &tc_pt() { return sdl.tc_pt(); }
  const vector<float> &tc_eta() { return sdl.tc_eta(); }
  const vector<float> &tc_phi() { return sdl.tc_phi(); }
  const vector<int> &tc_type() { return sdl.tc_type(); }
  const vector<int> &tc_pt5Idx() { return sdl.tc_pt5Idx(); }
  const vector<int> &tc_pt3Idx() { return sdl.tc_pt3Idx(); }
  const vector<int> &tc_t5Idx() { return sdl.tc_t5Idx(); }
  const vector<int> &tc_plsIdx() { return sdl.tc_plsIdx(); }
  const vector<int> &tc_isFake() { return sdl.tc_isFake(); }
  const vector<int> &tc_isDuplicate() { return sdl.tc_isDuplicate(); }
  const vector<int> &tc_simIdx() { return sdl.tc_simIdx(); }
  const vector<vector<int> > &tc_simIdxAll() { return sdl.tc_simIdxAll(); }
  const vector<vector<float> > &tc_simIdxAllFrac() { return sdl.tc_simIdxAllFrac(); }
  const vector<float> &md_pt() { return sdl.md_pt(); }
  const vector<float> &md_eta() { return sdl.md_eta(); }
  const vector<float> &md_phi() { return sdl.md_phi(); }
  #ifdef OUTPUT_MD_CUTS
  const vector<float> &md_dphi() { return sdl.md_dphi(); }
  const vector<float> &md_dphichange() { return sdl.md_dphichange(); }
  const vector<float> &md_dz() { return sdl.md_dz(); }
  #endif
  const vector<float> &md_anchor_x() { return sdl.md_anchor_x(); }
  const vector<float> &md_anchor_y() { return sdl.md_anchor_y(); }
  const vector<float> &md_anchor_z() { return sdl.md_anchor_z(); }
  const vector<float> &md_other_x() { return sdl.md_other_x(); }
  const vector<float> &md_other_y() { return sdl.md_other_y(); }
  const vector<float> &md_other_z() { return sdl.md_other_z(); }
  const vector<int> &md_type() { return sdl.md_type(); }
  const vector<int> &md_layer() { return sdl.md_layer(); }
  const vector<int> &md_detId() { return sdl.md_detId(); }
  const vector<int> &md_isFake() { return sdl.md_isFake(); }
  const vector<int> &md_simIdx() { return sdl.md_simIdx(); }
  const vector<vector<int> > &md_simIdxAll() { return sdl.md_simIdxAll(); }
  const vector<vector<float> > &md_simIdxAllFrac() { return sdl.md_simIdxAllFrac(); }
  const vector<float> &ls_pt() { return sdl.ls_pt(); }
  const vector<float> &ls_eta() { return sdl.ls_eta(); }
  const vector<float> &ls_phi() { return sdl.ls_phi(); }
  const vector<int> &ls_mdIdx0() { return sdl.ls_mdIdx0(); }
  const vector<int> &ls_mdIdx1() { return sdl.ls_mdIdx1(); }
  const vector<int> &ls_isFake() { return sdl.ls_isFake(); }
  const vector<int> &ls_simIdx() { return sdl.ls_simIdx(); }
  #ifdef OUTPUT_LS_CUTS
  const vector<int> &ls_zLos() { return sdl.ls_zLos(); }
  const vector<int> &ls_zHis() { return sdl.ls_zHis(); }
  const vector<int> &ls_rtLos() { return sdl.ls_rtLos(); }
  const vector<int> &ls_rtHis() { return sdl.ls_rtHis(); }
  const vector<int> &ls_dPhis() { return sdl.ls_dPhis(); }
  const vector<int> &ls_dPhiMins() { return sdl.ls_dPhiMins(); }
  const vector<int> &ls_dPhiMaxs() { return sdl.ls_dPhiMaxs(); }
  const vector<int> &ls_dPhiChanges() { return sdl.ls_dPhiChanges(); }
  const vector<int> &ls_dPhiChangeMins() { return sdl.ls_dPhiChangeMins(); }
  const vector<int> &ls_dPhiChangeMaxs() { return sdl.ls_dPhiChangeMaxs(); }
  const vector<int> &ls_dAlphaInners() { return sdl.ls_dAlphaInners(); }
  const vector<int> &ls_dAlphaOuters() { return sdl.ls_dAlphaOuters(); }
  const vector<int> &ls_dAlphaInnerOuters() { return sdl.ls_dAlphaInnerOuters(); }
  #endif
  const vector<vector<int> > &ls_simIdxAll() { return sdl.ls_simIdxAll(); }
  const vector<vector<float> > &ls_simIdxAllFrac() { return sdl.ls_simIdxAllFrac(); }
  const vector<float> &t3_pt() { return sdl.t3_pt(); }
  const vector<float> &t3_eta() { return sdl.t3_eta(); }
  const vector<float> &t3_phi() { return sdl.t3_phi(); }
  const vector<int> &t3_lsIdx0() { return sdl.t3_lsIdx0(); }
  const vector<int> &t3_lsIdx1() { return sdl.t3_lsIdx1(); }
  const vector<int> &t3_isFake() { return sdl.t3_isFake(); }
  const vector<int> &t3_isDuplicate() { return sdl.t3_isDuplicate(); }
  const vector<int> &t3_simIdx() { return sdl.t3_simIdx(); }
  const vector<vector<int> > &t3_simIdxAll() { return sdl.t3_simIdxAll(); }
  const vector<vector<float> > &t3_simIdxAllFrac() { return sdl.t3_simIdxAllFrac(); }
  const vector<float> &t5_pt() { return sdl.t5_pt(); }
  const vector<float> &t5_eta() { return sdl.t5_eta(); }
  const vector<float> &t5_phi() { return sdl.t5_phi(); }
  const vector<int> &t5_t3Idx0() { return sdl.t5_t3Idx0(); }
  const vector<int> &t5_t3Idx1() { return sdl.t5_t3Idx1(); }
  const vector<int> &t5_isFake() { return sdl.t5_isFake(); }
  const vector<int> &t5_isDuplicate() { return sdl.t5_isDuplicate(); }
  const vector<int> &t5_simIdx() { return sdl.t5_simIdx(); }
  const vector<vector<int> > &t5_simIdxAll() { return sdl.t5_simIdxAll(); }
  const vector<vector<float> > &t5_simIdxAllFrac() { return sdl.t5_simIdxAllFrac(); }
  const vector<float> &pls_pt() { return sdl.pls_pt(); }
  const vector<float> &pls_eta() { return sdl.pls_eta(); }
  const vector<float> &pls_phi() { return sdl.pls_phi(); }
  const vector<int> &pls_nhit() { return sdl.pls_nhit(); }
  const vector<float> &pls_hit0_x() { return sdl.pls_hit0_x(); }
  const vector<float> &pls_hit0_y() { return sdl.pls_hit0_y(); }
  const vector<float> &pls_hit0_z() { return sdl.pls_hit0_z(); }
  const vector<float> &pls_hit1_x() { return sdl.pls_hit1_x(); }
  const vector<float> &pls_hit1_y() { return sdl.pls_hit1_y(); }
  const vector<float> &pls_hit1_z() { return sdl.pls_hit1_z(); }
  const vector<float> &pls_hit2_x() { return sdl.pls_hit2_x(); }
  const vector<float> &pls_hit2_y() { return sdl.pls_hit2_y(); }
  const vector<float> &pls_hit2_z() { return sdl.pls_hit2_z(); }
  const vector<float> &pls_hit3_x() { return sdl.pls_hit3_x(); }
  const vector<float> &pls_hit3_y() { return sdl.pls_hit3_y(); }
  const vector<float> &pls_hit3_z() { return sdl.pls_hit3_z(); }
  const vector<int> &pls_isFake() { return sdl.pls_isFake(); }
  const vector<int> &pls_isDuplicate() { return sdl.pls_isDuplicate(); }
  const vector<int> &pls_simIdx() { return sdl.pls_simIdx(); }
  const vector<vector<int> > &pls_simIdxAll() { return sdl.pls_simIdxAll(); }
  const vector<vector<float> > &pls_simIdxAllFrac() { return sdl.pls_simIdxAllFrac(); }
  const vector<float> &pt3_pt() { return sdl.pt3_pt(); }
  const vector<float> &pt3_eta() { return sdl.pt3_eta(); }
  const vector<float> &pt3_phi() { return sdl.pt3_phi(); }
  const vector<int> &pt3_plsIdx() { return sdl.pt3_plsIdx(); }
  const vector<int> &pt3_t3Idx() { return sdl.pt3_t3Idx(); }
  const vector<int> &pt3_isFake() { return sdl.pt3_isFake(); }
  const vector<int> &pt3_isDuplicate() { return sdl.pt3_isDuplicate(); }
  const vector<int> &pt3_simIdx() { return sdl.pt3_simIdx(); }
  const vector<vector<int> > &pt3_simIdxAll() { return sdl.pt3_simIdxAll(); }
  const vector<vector<float> > &pt3_simIdxAllFrac() { return sdl.pt3_simIdxAllFrac(); }
  const vector<float> &pt5_pt() { return sdl.pt5_pt(); }
  const vector<float> &pt5_eta() { return sdl.pt5_eta(); }
  const vector<float> &pt5_phi() { return sdl.pt5_phi(); }
  const vector<int> &pt5_plsIdx() { return sdl.pt5_plsIdx(); }
  const vector<int> &pt5_t5Idx() { return sdl.pt5_t5Idx(); }
  const vector<int> &pt5_isFake() { return sdl.pt5_isFake(); }
  const vector<int> &pt5_isDuplicate() { return sdl.pt5_isDuplicate(); }
  const vector<int> &pt5_simIdx() { return sdl.pt5_simIdx(); }
  const vector<vector<int> > &pt5_simIdxAll() { return sdl.pt5_simIdxAll(); }
  const vector<vector<float> > &pt5_simIdxAllFrac() { return sdl.pt5_simIdxAllFrac(); }
}
