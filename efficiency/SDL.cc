#include "SDL.h"
SDL sdl;

void SDL::Init(TTree *tree) {

  tree->SetMakeClass(1);

  ph2_x_branch = tree->GetBranch("ph2_x");
  if (ph2_x_branch) ph2_x_branch->SetAddress(&ph2_x_);
  ph2_y_branch = tree->GetBranch("ph2_y");
  if (ph2_y_branch) ph2_y_branch->SetAddress(&ph2_y_);
  ph2_z_branch = tree->GetBranch("ph2_z");
  if (ph2_z_branch) ph2_z_branch->SetAddress(&ph2_z_);
  ph2_detId_branch = tree->GetBranch("ph2_detId");
  if (ph2_detId_branch) ph2_detId_branch->SetAddress(&ph2_detId_);
  ph2_simHitIdx_branch = tree->GetBranch("ph2_simHitIdx");
  if (ph2_simHitIdx_branch) ph2_simHitIdx_branch->SetAddress(&ph2_simHitIdx_);
  ph2_simType_branch = tree->GetBranch("ph2_simType");
  if (ph2_simType_branch) ph2_simType_branch->SetAddress(&ph2_simType_);
  ph2_anchorLayer_branch = tree->GetBranch("ph2_anchorLayer");
  if (ph2_anchorLayer_branch) ph2_anchorLayer_branch->SetAddress(&ph2_anchorLayer_);
  simhit_x_branch = tree->GetBranch("simhit_x");
  if (simhit_x_branch) simhit_x_branch->SetAddress(&simhit_x_);
  simhit_y_branch = tree->GetBranch("simhit_y");
  if (simhit_y_branch) simhit_y_branch->SetAddress(&simhit_y_);
  simhit_z_branch = tree->GetBranch("simhit_z");
  if (simhit_z_branch) simhit_z_branch->SetAddress(&simhit_z_);
  simhit_detId_branch = tree->GetBranch("simhit_detId");
  if (simhit_detId_branch) simhit_detId_branch->SetAddress(&simhit_detId_);
  simhit_partnerDetId_branch = tree->GetBranch("simhit_partnerDetId");
  if (simhit_partnerDetId_branch) simhit_partnerDetId_branch->SetAddress(&simhit_partnerDetId_);
  simhit_subdet_branch = tree->GetBranch("simhit_subdet");
  if (simhit_subdet_branch) simhit_subdet_branch->SetAddress(&simhit_subdet_);
  simhit_particle_branch = tree->GetBranch("simhit_particle");
  if (simhit_particle_branch) simhit_particle_branch->SetAddress(&simhit_particle_);
  simhit_hitIdx_branch = tree->GetBranch("simhit_hitIdx");
  if (simhit_hitIdx_branch) simhit_hitIdx_branch->SetAddress(&simhit_hitIdx_);
  simhit_simTrkIdx_branch = tree->GetBranch("simhit_simTrkIdx");
  if (simhit_simTrkIdx_branch) simhit_simTrkIdx_branch->SetAddress(&simhit_simTrkIdx_);
  sim_pt_branch = tree->GetBranch("sim_pt");
  if (sim_pt_branch) sim_pt_branch->SetAddress(&sim_pt_);
  sim_eta_branch = tree->GetBranch("sim_eta");
  if (sim_eta_branch) sim_eta_branch->SetAddress(&sim_eta_);
  sim_phi_branch = tree->GetBranch("sim_phi");
  if (sim_phi_branch) sim_phi_branch->SetAddress(&sim_phi_);
  sim_pca_dxy_branch = tree->GetBranch("sim_pca_dxy");
  if (sim_pca_dxy_branch) sim_pca_dxy_branch->SetAddress(&sim_pca_dxy_);
  sim_pca_dz_branch = tree->GetBranch("sim_pca_dz");
  if (sim_pca_dz_branch) sim_pca_dz_branch->SetAddress(&sim_pca_dz_);
  sim_q_branch = tree->GetBranch("sim_q");
  if (sim_q_branch) sim_q_branch->SetAddress(&sim_q_);
  sim_event_branch = tree->GetBranch("sim_event");
  if (sim_event_branch) sim_event_branch->SetAddress(&sim_event_);
  sim_pdgId_branch = tree->GetBranch("sim_pdgId");
  if (sim_pdgId_branch) sim_pdgId_branch->SetAddress(&sim_pdgId_);
  sim_bunchCrossing_branch = tree->GetBranch("sim_bunchCrossing");
  if (sim_bunchCrossing_branch) sim_bunchCrossing_branch->SetAddress(&sim_bunchCrossing_);
  sim_hasAll12HitsInBarrel_branch = tree->GetBranch("sim_hasAll12HitsInBarrel");
  if (sim_hasAll12HitsInBarrel_branch) sim_hasAll12HitsInBarrel_branch->SetAddress(&sim_hasAll12HitsInBarrel_);
  sim_simHitIdx_branch = tree->GetBranch("sim_simHitIdx");
  if (sim_simHitIdx_branch) sim_simHitIdx_branch->SetAddress(&sim_simHitIdx_);
  sim_simHitLayer_branch = tree->GetBranch("sim_simHitLayer");
  if (sim_simHitLayer_branch) sim_simHitLayer_branch->SetAddress(&sim_simHitLayer_);
  sim_simHitBoth_branch = tree->GetBranch("sim_simHitBoth");
  if (sim_simHitBoth_branch) sim_simHitBoth_branch->SetAddress(&sim_simHitBoth_);
  sim_simHitDrFracWithHelix_branch = tree->GetBranch("sim_simHitDrFracWithHelix");
  if (sim_simHitDrFracWithHelix_branch) sim_simHitDrFracWithHelix_branch->SetAddress(&sim_simHitDrFracWithHelix_);
  sim_simHitDistXyWithHelix_branch = tree->GetBranch("sim_simHitDistXyWithHelix");
  if (sim_simHitDistXyWithHelix_branch) sim_simHitDistXyWithHelix_branch->SetAddress(&sim_simHitDistXyWithHelix_);
  simvtx_x_branch = tree->GetBranch("simvtx_x");
  if (simvtx_x_branch) simvtx_x_branch->SetAddress(&simvtx_x_);
  simvtx_y_branch = tree->GetBranch("simvtx_y");
  if (simvtx_y_branch) simvtx_y_branch->SetAddress(&simvtx_y_);
  simvtx_z_branch = tree->GetBranch("simvtx_z");
  if (simvtx_z_branch) simvtx_z_branch->SetAddress(&simvtx_z_);
  see_stateTrajGlbPx_branch = tree->GetBranch("see_stateTrajGlbPx");
  if (see_stateTrajGlbPx_branch) see_stateTrajGlbPx_branch->SetAddress(&see_stateTrajGlbPx_);
  see_stateTrajGlbPy_branch = tree->GetBranch("see_stateTrajGlbPy");
  if (see_stateTrajGlbPy_branch) see_stateTrajGlbPy_branch->SetAddress(&see_stateTrajGlbPy_);
  see_stateTrajGlbPz_branch = tree->GetBranch("see_stateTrajGlbPz");
  if (see_stateTrajGlbPz_branch) see_stateTrajGlbPz_branch->SetAddress(&see_stateTrajGlbPz_);
  see_stateTrajGlbX_branch = tree->GetBranch("see_stateTrajGlbX");
  if (see_stateTrajGlbX_branch) see_stateTrajGlbX_branch->SetAddress(&see_stateTrajGlbX_);
  see_stateTrajGlbY_branch = tree->GetBranch("see_stateTrajGlbY");
  if (see_stateTrajGlbY_branch) see_stateTrajGlbY_branch->SetAddress(&see_stateTrajGlbY_);
  see_stateTrajGlbZ_branch = tree->GetBranch("see_stateTrajGlbZ");
  if (see_stateTrajGlbZ_branch) see_stateTrajGlbZ_branch->SetAddress(&see_stateTrajGlbZ_);
  see_px_branch = tree->GetBranch("see_px");
  if (see_px_branch) see_px_branch->SetAddress(&see_px_);
  see_py_branch = tree->GetBranch("see_py");
  if (see_py_branch) see_py_branch->SetAddress(&see_py_);
  see_pz_branch = tree->GetBranch("see_pz");
  if (see_pz_branch) see_pz_branch->SetAddress(&see_pz_);
  see_ptErr_branch = tree->GetBranch("see_ptErr");
  if (see_ptErr_branch) see_ptErr_branch->SetAddress(&see_ptErr_);
  see_dxy_branch = tree->GetBranch("see_dxy");
  if (see_dxy_branch) see_dxy_branch->SetAddress(&see_dxy_);
  see_dxyErr_branch = tree->GetBranch("see_dxyErr");
  if (see_dxyErr_branch) see_dxyErr_branch->SetAddress(&see_dxyErr_);
  see_dz_branch = tree->GetBranch("see_dz");
  if (see_dz_branch) see_dz_branch->SetAddress(&see_dz_);
  see_hitIdx_branch = tree->GetBranch("see_hitIdx");
  if (see_hitIdx_branch) see_hitIdx_branch->SetAddress(&see_hitIdx_);
  see_hitType_branch = tree->GetBranch("see_hitType");
  if (see_hitType_branch) see_hitType_branch->SetAddress(&see_hitType_);
  see_simTrkIdx_branch = tree->GetBranch("see_simTrkIdx");
  if (see_simTrkIdx_branch) see_simTrkIdx_branch->SetAddress(&see_simTrkIdx_);
  see_algo_branch = tree->GetBranch("see_algo");
  if (see_algo_branch) see_algo_branch->SetAddress(&see_algo_);
  md_hitIdx_branch = tree->GetBranch("md_hitIdx");
  if (md_hitIdx_branch) md_hitIdx_branch->SetAddress(&md_hitIdx_);
  md_simTrkIdx_branch = tree->GetBranch("md_simTrkIdx");
  if (md_simTrkIdx_branch) md_simTrkIdx_branch->SetAddress(&md_simTrkIdx_);
  md_layer_branch = tree->GetBranch("md_layer");
  if (md_layer_branch) md_layer_branch->SetAddress(&md_layer_);
  md_pt_branch = tree->GetBranch("md_pt");
  if (md_pt_branch) md_pt_branch->SetAddress(&md_pt_);
  md_eta_branch = tree->GetBranch("md_eta");
  if (md_eta_branch) md_eta_branch->SetAddress(&md_eta_);
  md_phi_branch = tree->GetBranch("md_phi");
  if (md_phi_branch) md_phi_branch->SetAddress(&md_phi_);
  md_sim_pt_branch = tree->GetBranch("md_sim_pt");
  if (md_sim_pt_branch) md_sim_pt_branch->SetAddress(&md_sim_pt_);
  md_sim_eta_branch = tree->GetBranch("md_sim_eta");
  if (md_sim_eta_branch) md_sim_eta_branch->SetAddress(&md_sim_eta_);
  md_sim_phi_branch = tree->GetBranch("md_sim_phi");
  if (md_sim_phi_branch) md_sim_phi_branch->SetAddress(&md_sim_phi_);
  md_type_branch = tree->GetBranch("md_type");
  if (md_type_branch) md_type_branch->SetAddress(&md_type_);
  md_dz_branch = tree->GetBranch("md_dz");
  if (md_dz_branch) md_dz_branch->SetAddress(&md_dz_);
  md_dzCut_branch = tree->GetBranch("md_dzCut");
  if (md_dzCut_branch) md_dzCut_branch->SetAddress(&md_dzCut_);
  md_drt_branch = tree->GetBranch("md_drt");
  if (md_drt_branch) md_drt_branch->SetAddress(&md_drt_);
  md_drtCut_branch = tree->GetBranch("md_drtCut");
  if (md_drtCut_branch) md_drtCut_branch->SetAddress(&md_drtCut_);
  md_miniCut_branch = tree->GetBranch("md_miniCut");
  if (md_miniCut_branch) md_miniCut_branch->SetAddress(&md_miniCut_);
  md_dphi_branch = tree->GetBranch("md_dphi");
  if (md_dphi_branch) md_dphi_branch->SetAddress(&md_dphi_);
  md_dphiChange_branch = tree->GetBranch("md_dphiChange");
  if (md_dphiChange_branch) md_dphiChange_branch->SetAddress(&md_dphiChange_);
  sim_mdIdx_branch = tree->GetBranch("sim_mdIdx");
  if (sim_mdIdx_branch) sim_mdIdx_branch->SetAddress(&sim_mdIdx_);
  sim_mdIdx_isMTVmatch_branch = tree->GetBranch("sim_mdIdx_isMTVmatch");
  if (sim_mdIdx_isMTVmatch_branch) sim_mdIdx_isMTVmatch_branch->SetAddress(&sim_mdIdx_isMTVmatch_);
  ph2_mdIdx_branch = tree->GetBranch("ph2_mdIdx");
  if (ph2_mdIdx_branch) ph2_mdIdx_branch->SetAddress(&ph2_mdIdx_);
  sg_hitIdx_branch = tree->GetBranch("sg_hitIdx");
  if (sg_hitIdx_branch) sg_hitIdx_branch->SetAddress(&sg_hitIdx_);
  sg_simTrkIdx_branch = tree->GetBranch("sg_simTrkIdx");
  if (sg_simTrkIdx_branch) sg_simTrkIdx_branch->SetAddress(&sg_simTrkIdx_);
  sg_simTrkIdx_anchorMatching_branch = tree->GetBranch("sg_simTrkIdx_anchorMatching");
  if (sg_simTrkIdx_anchorMatching_branch) sg_simTrkIdx_anchorMatching_branch->SetAddress(&sg_simTrkIdx_anchorMatching_);
  sg_layer_branch = tree->GetBranch("sg_layer");
  if (sg_layer_branch) sg_layer_branch->SetAddress(&sg_layer_);
  sg_pt_branch = tree->GetBranch("sg_pt");
  if (sg_pt_branch) sg_pt_branch->SetAddress(&sg_pt_);
  sg_eta_branch = tree->GetBranch("sg_eta");
  if (sg_eta_branch) sg_eta_branch->SetAddress(&sg_eta_);
  sg_phi_branch = tree->GetBranch("sg_phi");
  if (sg_phi_branch) sg_phi_branch->SetAddress(&sg_phi_);
  sg_sim_pt_branch = tree->GetBranch("sg_sim_pt");
  if (sg_sim_pt_branch) sg_sim_pt_branch->SetAddress(&sg_sim_pt_);
  sg_sim_eta_branch = tree->GetBranch("sg_sim_eta");
  if (sg_sim_eta_branch) sg_sim_eta_branch->SetAddress(&sg_sim_eta_);
  sg_sim_phi_branch = tree->GetBranch("sg_sim_phi");
  if (sg_sim_phi_branch) sg_sim_phi_branch->SetAddress(&sg_sim_phi_);
  sim_sgIdx_branch = tree->GetBranch("sim_sgIdx");
  if (sim_sgIdx_branch) sim_sgIdx_branch->SetAddress(&sim_sgIdx_);
  sim_sgIdx_isMTVmatch_branch = tree->GetBranch("sim_sgIdx_isMTVmatch");
  if (sim_sgIdx_isMTVmatch_branch) sim_sgIdx_isMTVmatch_branch->SetAddress(&sim_sgIdx_isMTVmatch_);
  psg_hitIdx_branch = tree->GetBranch("psg_hitIdx");
  if (psg_hitIdx_branch) psg_hitIdx_branch->SetAddress(&psg_hitIdx_);
  psg_simTrkIdx_branch = tree->GetBranch("psg_simTrkIdx");
  if (psg_simTrkIdx_branch) psg_simTrkIdx_branch->SetAddress(&psg_simTrkIdx_);
  psg_simTrkIdx_anchorMatching_branch = tree->GetBranch("psg_simTrkIdx_anchorMatching");
  if (psg_simTrkIdx_anchorMatching_branch) psg_simTrkIdx_anchorMatching_branch->SetAddress(&psg_simTrkIdx_anchorMatching_);
  psg_layer_branch = tree->GetBranch("psg_layer");
  if (psg_layer_branch) psg_layer_branch->SetAddress(&psg_layer_);
  psg_pt_branch = tree->GetBranch("psg_pt");
  if (psg_pt_branch) psg_pt_branch->SetAddress(&psg_pt_);
  psg_eta_branch = tree->GetBranch("psg_eta");
  if (psg_eta_branch) psg_eta_branch->SetAddress(&psg_eta_);
  psg_phi_branch = tree->GetBranch("psg_phi");
  if (psg_phi_branch) psg_phi_branch->SetAddress(&psg_phi_);
  psg_sim_pt_branch = tree->GetBranch("psg_sim_pt");
  if (psg_sim_pt_branch) psg_sim_pt_branch->SetAddress(&psg_sim_pt_);
  psg_sim_eta_branch = tree->GetBranch("psg_sim_eta");
  if (psg_sim_eta_branch) psg_sim_eta_branch->SetAddress(&psg_sim_eta_);
  psg_sim_phi_branch = tree->GetBranch("psg_sim_phi");
  if (psg_sim_phi_branch) psg_sim_phi_branch->SetAddress(&psg_sim_phi_);
  sim_psgIdx_branch = tree->GetBranch("sim_psgIdx");
  if (sim_psgIdx_branch) sim_psgIdx_branch->SetAddress(&sim_psgIdx_);
  sim_psgIdx_isMTVmatch_branch = tree->GetBranch("sim_psgIdx_isMTVmatch");
  if (sim_psgIdx_isMTVmatch_branch) sim_psgIdx_isMTVmatch_branch->SetAddress(&sim_psgIdx_isMTVmatch_);
  tp_hitIdx_branch = tree->GetBranch("tp_hitIdx");
  if (tp_hitIdx_branch) tp_hitIdx_branch->SetAddress(&tp_hitIdx_);
  tp_simTrkIdx_branch = tree->GetBranch("tp_simTrkIdx");
  if (tp_simTrkIdx_branch) tp_simTrkIdx_branch->SetAddress(&tp_simTrkIdx_);
  tp_layer_branch = tree->GetBranch("tp_layer");
  if (tp_layer_branch) tp_layer_branch->SetAddress(&tp_layer_);
  tp_pt_branch = tree->GetBranch("tp_pt");
  if (tp_pt_branch) tp_pt_branch->SetAddress(&tp_pt_);
  tp_eta_branch = tree->GetBranch("tp_eta");
  if (tp_eta_branch) tp_eta_branch->SetAddress(&tp_eta_);
  tp_phi_branch = tree->GetBranch("tp_phi");
  if (tp_phi_branch) tp_phi_branch->SetAddress(&tp_phi_);
  tp_sim_pt_branch = tree->GetBranch("tp_sim_pt");
  if (tp_sim_pt_branch) tp_sim_pt_branch->SetAddress(&tp_sim_pt_);
  tp_sim_eta_branch = tree->GetBranch("tp_sim_eta");
  if (tp_sim_eta_branch) tp_sim_eta_branch->SetAddress(&tp_sim_eta_);
  tp_sim_phi_branch = tree->GetBranch("tp_sim_phi");
  if (tp_sim_phi_branch) tp_sim_phi_branch->SetAddress(&tp_sim_phi_);
  sim_tpIdx_branch = tree->GetBranch("sim_tpIdx");
  if (sim_tpIdx_branch) sim_tpIdx_branch->SetAddress(&sim_tpIdx_);
  sim_tpIdx_isMTVmatch_branch = tree->GetBranch("sim_tpIdx_isMTVmatch");
  if (sim_tpIdx_isMTVmatch_branch) sim_tpIdx_isMTVmatch_branch->SetAddress(&sim_tpIdx_isMTVmatch_);
  qp_hitIdx_branch = tree->GetBranch("qp_hitIdx");
  if (qp_hitIdx_branch) qp_hitIdx_branch->SetAddress(&qp_hitIdx_);
  qp_simTrkIdx_branch = tree->GetBranch("qp_simTrkIdx");
  if (qp_simTrkIdx_branch) qp_simTrkIdx_branch->SetAddress(&qp_simTrkIdx_);
  qp_layer_branch = tree->GetBranch("qp_layer");
  if (qp_layer_branch) qp_layer_branch->SetAddress(&qp_layer_);
  qp_pt_branch = tree->GetBranch("qp_pt");
  if (qp_pt_branch) qp_pt_branch->SetAddress(&qp_pt_);
  qp_eta_branch = tree->GetBranch("qp_eta");
  if (qp_eta_branch) qp_eta_branch->SetAddress(&qp_eta_);
  qp_phi_branch = tree->GetBranch("qp_phi");
  if (qp_phi_branch) qp_phi_branch->SetAddress(&qp_phi_);
  qp_sim_pt_branch = tree->GetBranch("qp_sim_pt");
  if (qp_sim_pt_branch) qp_sim_pt_branch->SetAddress(&qp_sim_pt_);
  qp_sim_eta_branch = tree->GetBranch("qp_sim_eta");
  if (qp_sim_eta_branch) qp_sim_eta_branch->SetAddress(&qp_sim_eta_);
  qp_sim_phi_branch = tree->GetBranch("qp_sim_phi");
  if (qp_sim_phi_branch) qp_sim_phi_branch->SetAddress(&qp_sim_phi_);
  sim_qpIdx_branch = tree->GetBranch("sim_qpIdx");
  if (sim_qpIdx_branch) sim_qpIdx_branch->SetAddress(&sim_qpIdx_);
  sim_qpIdx_isMTVmatch_branch = tree->GetBranch("sim_qpIdx_isMTVmatch");
  if (sim_qpIdx_isMTVmatch_branch) sim_qpIdx_isMTVmatch_branch->SetAddress(&sim_qpIdx_isMTVmatch_);
  pqp_hitIdx_branch = tree->GetBranch("pqp_hitIdx");
  if (pqp_hitIdx_branch) pqp_hitIdx_branch->SetAddress(&pqp_hitIdx_);
  pqp_simTrkIdx_branch = tree->GetBranch("pqp_simTrkIdx");
  if (pqp_simTrkIdx_branch) pqp_simTrkIdx_branch->SetAddress(&pqp_simTrkIdx_);
  pqp_layer_branch = tree->GetBranch("pqp_layer");
  if (pqp_layer_branch) pqp_layer_branch->SetAddress(&pqp_layer_);
  pqp_pt_branch = tree->GetBranch("pqp_pt");
  if (pqp_pt_branch) pqp_pt_branch->SetAddress(&pqp_pt_);
  pqp_eta_branch = tree->GetBranch("pqp_eta");
  if (pqp_eta_branch) pqp_eta_branch->SetAddress(&pqp_eta_);
  pqp_phi_branch = tree->GetBranch("pqp_phi");
  if (pqp_phi_branch) pqp_phi_branch->SetAddress(&pqp_phi_);
  pqp_sim_pt_branch = tree->GetBranch("pqp_sim_pt");
  if (pqp_sim_pt_branch) pqp_sim_pt_branch->SetAddress(&pqp_sim_pt_);
  pqp_sim_eta_branch = tree->GetBranch("pqp_sim_eta");
  if (pqp_sim_eta_branch) pqp_sim_eta_branch->SetAddress(&pqp_sim_eta_);
  pqp_sim_phi_branch = tree->GetBranch("pqp_sim_phi");
  if (pqp_sim_phi_branch) pqp_sim_phi_branch->SetAddress(&pqp_sim_phi_);
  sim_pqpIdx_branch = tree->GetBranch("sim_pqpIdx");
  if (sim_pqpIdx_branch) sim_pqpIdx_branch->SetAddress(&sim_pqpIdx_);
  sim_pqpIdx_isMTVmatch_branch = tree->GetBranch("sim_pqpIdx_isMTVmatch");
  if (sim_pqpIdx_isMTVmatch_branch) sim_pqpIdx_isMTVmatch_branch->SetAddress(&sim_pqpIdx_isMTVmatch_);
  tc_hitIdx_branch = tree->GetBranch("tc_hitIdx");
  if (tc_hitIdx_branch) tc_hitIdx_branch->SetAddress(&tc_hitIdx_);
  tc_simTrkIdx_branch = tree->GetBranch("tc_simTrkIdx");
  if (tc_simTrkIdx_branch) tc_simTrkIdx_branch->SetAddress(&tc_simTrkIdx_);
  tc_layer_branch = tree->GetBranch("tc_layer");
  if (tc_layer_branch) tc_layer_branch->SetAddress(&tc_layer_);
  tc_pt_branch = tree->GetBranch("tc_pt");
  if (tc_pt_branch) tc_pt_branch->SetAddress(&tc_pt_);
  tc_eta_branch = tree->GetBranch("tc_eta");
  if (tc_eta_branch) tc_eta_branch->SetAddress(&tc_eta_);
  tc_phi_branch = tree->GetBranch("tc_phi");
  if (tc_phi_branch) tc_phi_branch->SetAddress(&tc_phi_);
  tc_sim_pt_branch = tree->GetBranch("tc_sim_pt");
  if (tc_sim_pt_branch) tc_sim_pt_branch->SetAddress(&tc_sim_pt_);
  tc_sim_eta_branch = tree->GetBranch("tc_sim_eta");
  if (tc_sim_eta_branch) tc_sim_eta_branch->SetAddress(&tc_sim_eta_);
  tc_sim_phi_branch = tree->GetBranch("tc_sim_phi");
  if (tc_sim_phi_branch) tc_sim_phi_branch->SetAddress(&tc_sim_phi_);
  sim_tcIdx_branch = tree->GetBranch("sim_tcIdx");
  if (sim_tcIdx_branch) sim_tcIdx_branch->SetAddress(&sim_tcIdx_);
  sim_tcIdx_isMTVmatch_branch = tree->GetBranch("sim_tcIdx_isMTVmatch");
  if (sim_tcIdx_isMTVmatch_branch) sim_tcIdx_isMTVmatch_branch->SetAddress(&sim_tcIdx_isMTVmatch_);

  tree->SetMakeClass(0);
}

void SDL::GetEntry(unsigned int idx) {
  // this only marks branches as not loaded, saving a lot of time
  index = idx;
  ph2_x_isLoaded = false;
  ph2_y_isLoaded = false;
  ph2_z_isLoaded = false;
  ph2_detId_isLoaded = false;
  ph2_simHitIdx_isLoaded = false;
  ph2_simType_isLoaded = false;
  ph2_anchorLayer_isLoaded = false;
  simhit_x_isLoaded = false;
  simhit_y_isLoaded = false;
  simhit_z_isLoaded = false;
  simhit_detId_isLoaded = false;
  simhit_partnerDetId_isLoaded = false;
  simhit_subdet_isLoaded = false;
  simhit_particle_isLoaded = false;
  simhit_hitIdx_isLoaded = false;
  simhit_simTrkIdx_isLoaded = false;
  sim_pt_isLoaded = false;
  sim_eta_isLoaded = false;
  sim_phi_isLoaded = false;
  sim_pca_dxy_isLoaded = false;
  sim_pca_dz_isLoaded = false;
  sim_q_isLoaded = false;
  sim_event_isLoaded = false;
  sim_pdgId_isLoaded = false;
  sim_bunchCrossing_isLoaded = false;
  sim_hasAll12HitsInBarrel_isLoaded = false;
  sim_simHitIdx_isLoaded = false;
  sim_simHitLayer_isLoaded = false;
  sim_simHitBoth_isLoaded = false;
  sim_simHitDrFracWithHelix_isLoaded = false;
  sim_simHitDistXyWithHelix_isLoaded = false;
  simvtx_x_isLoaded = false;
  simvtx_y_isLoaded = false;
  simvtx_z_isLoaded = false;
  see_stateTrajGlbPx_isLoaded = false;
  see_stateTrajGlbPy_isLoaded = false;
  see_stateTrajGlbPz_isLoaded = false;
  see_stateTrajGlbX_isLoaded = false;
  see_stateTrajGlbY_isLoaded = false;
  see_stateTrajGlbZ_isLoaded = false;
  see_px_isLoaded = false;
  see_py_isLoaded = false;
  see_pz_isLoaded = false;
  see_ptErr_isLoaded = false;
  see_dxy_isLoaded = false;
  see_dxyErr_isLoaded = false;
  see_dz_isLoaded = false;
  see_hitIdx_isLoaded = false;
  see_hitType_isLoaded = false;
  see_simTrkIdx_isLoaded = false;
  see_algo_isLoaded = false;
  md_hitIdx_isLoaded = false;
  md_simTrkIdx_isLoaded = false;
  md_layer_isLoaded = false;
  md_pt_isLoaded = false;
  md_eta_isLoaded = false;
  md_phi_isLoaded = false;
  md_sim_pt_isLoaded = false;
  md_sim_eta_isLoaded = false;
  md_sim_phi_isLoaded = false;
  md_type_isLoaded = false;
  md_dz_isLoaded = false;
  md_dzCut_isLoaded = false;
  md_drt_isLoaded = false;
  md_drtCut_isLoaded = false;
  md_miniCut_isLoaded = false;
  md_dphi_isLoaded = false;
  md_dphiChange_isLoaded = false;
  sim_mdIdx_isLoaded = false;
  sim_mdIdx_isMTVmatch_isLoaded = false;
  ph2_mdIdx_isLoaded = false;
  sg_hitIdx_isLoaded = false;
  sg_simTrkIdx_isLoaded = false;
  sg_simTrkIdx_anchorMatching_isLoaded = false;
  sg_layer_isLoaded = false;
  sg_pt_isLoaded = false;
  sg_eta_isLoaded = false;
  sg_phi_isLoaded = false;
  sg_sim_pt_isLoaded = false;
  sg_sim_eta_isLoaded = false;
  sg_sim_phi_isLoaded = false;
  sim_sgIdx_isLoaded = false;
  sim_sgIdx_isMTVmatch_isLoaded = false;
  psg_hitIdx_isLoaded = false;
  psg_simTrkIdx_isLoaded = false;
  psg_simTrkIdx_anchorMatching_isLoaded = false;
  psg_layer_isLoaded = false;
  psg_pt_isLoaded = false;
  psg_eta_isLoaded = false;
  psg_phi_isLoaded = false;
  psg_sim_pt_isLoaded = false;
  psg_sim_eta_isLoaded = false;
  psg_sim_phi_isLoaded = false;
  sim_psgIdx_isLoaded = false;
  sim_psgIdx_isMTVmatch_isLoaded = false;
  tp_hitIdx_isLoaded = false;
  tp_simTrkIdx_isLoaded = false;
  tp_layer_isLoaded = false;
  tp_pt_isLoaded = false;
  tp_eta_isLoaded = false;
  tp_phi_isLoaded = false;
  tp_sim_pt_isLoaded = false;
  tp_sim_eta_isLoaded = false;
  tp_sim_phi_isLoaded = false;
  sim_tpIdx_isLoaded = false;
  sim_tpIdx_isMTVmatch_isLoaded = false;
  qp_hitIdx_isLoaded = false;
  qp_simTrkIdx_isLoaded = false;
  qp_layer_isLoaded = false;
  qp_pt_isLoaded = false;
  qp_eta_isLoaded = false;
  qp_phi_isLoaded = false;
  qp_sim_pt_isLoaded = false;
  qp_sim_eta_isLoaded = false;
  qp_sim_phi_isLoaded = false;
  sim_qpIdx_isLoaded = false;
  sim_qpIdx_isMTVmatch_isLoaded = false;
  pqp_hitIdx_isLoaded = false;
  pqp_simTrkIdx_isLoaded = false;
  pqp_layer_isLoaded = false;
  pqp_pt_isLoaded = false;
  pqp_eta_isLoaded = false;
  pqp_phi_isLoaded = false;
  pqp_sim_pt_isLoaded = false;
  pqp_sim_eta_isLoaded = false;
  pqp_sim_phi_isLoaded = false;
  sim_pqpIdx_isLoaded = false;
  sim_pqpIdx_isMTVmatch_isLoaded = false;
  tc_hitIdx_isLoaded = false;
  tc_simTrkIdx_isLoaded = false;
  tc_layer_isLoaded = false;
  tc_pt_isLoaded = false;
  tc_eta_isLoaded = false;
  tc_phi_isLoaded = false;
  tc_sim_pt_isLoaded = false;
  tc_sim_eta_isLoaded = false;
  tc_sim_phi_isLoaded = false;
  sim_tcIdx_isLoaded = false;
  sim_tcIdx_isMTVmatch_isLoaded = false;
}

void SDL::LoadAllBranches() {
  // load all branches
  if (ph2_x_branch != 0) ph2_x();
  if (ph2_y_branch != 0) ph2_y();
  if (ph2_z_branch != 0) ph2_z();
  if (ph2_detId_branch != 0) ph2_detId();
  if (ph2_simHitIdx_branch != 0) ph2_simHitIdx();
  if (ph2_simType_branch != 0) ph2_simType();
  if (ph2_anchorLayer_branch != 0) ph2_anchorLayer();
  if (simhit_x_branch != 0) simhit_x();
  if (simhit_y_branch != 0) simhit_y();
  if (simhit_z_branch != 0) simhit_z();
  if (simhit_detId_branch != 0) simhit_detId();
  if (simhit_partnerDetId_branch != 0) simhit_partnerDetId();
  if (simhit_subdet_branch != 0) simhit_subdet();
  if (simhit_particle_branch != 0) simhit_particle();
  if (simhit_hitIdx_branch != 0) simhit_hitIdx();
  if (simhit_simTrkIdx_branch != 0) simhit_simTrkIdx();
  if (sim_pt_branch != 0) sim_pt();
  if (sim_eta_branch != 0) sim_eta();
  if (sim_phi_branch != 0) sim_phi();
  if (sim_pca_dxy_branch != 0) sim_pca_dxy();
  if (sim_pca_dz_branch != 0) sim_pca_dz();
  if (sim_q_branch != 0) sim_q();
  if (sim_event_branch != 0) sim_event();
  if (sim_pdgId_branch != 0) sim_pdgId();
  if (sim_bunchCrossing_branch != 0) sim_bunchCrossing();
  if (sim_hasAll12HitsInBarrel_branch != 0) sim_hasAll12HitsInBarrel();
  if (sim_simHitIdx_branch != 0) sim_simHitIdx();
  if (sim_simHitLayer_branch != 0) sim_simHitLayer();
  if (sim_simHitBoth_branch != 0) sim_simHitBoth();
  if (sim_simHitDrFracWithHelix_branch != 0) sim_simHitDrFracWithHelix();
  if (sim_simHitDistXyWithHelix_branch != 0) sim_simHitDistXyWithHelix();
  if (simvtx_x_branch != 0) simvtx_x();
  if (simvtx_y_branch != 0) simvtx_y();
  if (simvtx_z_branch != 0) simvtx_z();
  if (see_stateTrajGlbPx_branch != 0) see_stateTrajGlbPx();
  if (see_stateTrajGlbPy_branch != 0) see_stateTrajGlbPy();
  if (see_stateTrajGlbPz_branch != 0) see_stateTrajGlbPz();
  if (see_stateTrajGlbX_branch != 0) see_stateTrajGlbX();
  if (see_stateTrajGlbY_branch != 0) see_stateTrajGlbY();
  if (see_stateTrajGlbZ_branch != 0) see_stateTrajGlbZ();
  if (see_px_branch != 0) see_px();
  if (see_py_branch != 0) see_py();
  if (see_pz_branch != 0) see_pz();
  if (see_ptErr_branch != 0) see_ptErr();
  if (see_dxy_branch != 0) see_dxy();
  if (see_dxyErr_branch != 0) see_dxyErr();
  if (see_dz_branch != 0) see_dz();
  if (see_hitIdx_branch != 0) see_hitIdx();
  if (see_hitType_branch != 0) see_hitType();
  if (see_simTrkIdx_branch != 0) see_simTrkIdx();
  if (see_algo_branch != 0) see_algo();
  if (md_hitIdx_branch != 0) md_hitIdx();
  if (md_simTrkIdx_branch != 0) md_simTrkIdx();
  if (md_layer_branch != 0) md_layer();
  if (md_pt_branch != 0) md_pt();
  if (md_eta_branch != 0) md_eta();
  if (md_phi_branch != 0) md_phi();
  if (md_sim_pt_branch != 0) md_sim_pt();
  if (md_sim_eta_branch != 0) md_sim_eta();
  if (md_sim_phi_branch != 0) md_sim_phi();
  if (md_type_branch != 0) md_type();
  if (md_dz_branch != 0) md_dz();
  if (md_dzCut_branch != 0) md_dzCut();
  if (md_drt_branch != 0) md_drt();
  if (md_drtCut_branch != 0) md_drtCut();
  if (md_miniCut_branch != 0) md_miniCut();
  if (md_dphi_branch != 0) md_dphi();
  if (md_dphiChange_branch != 0) md_dphiChange();
  if (sim_mdIdx_branch != 0) sim_mdIdx();
  if (sim_mdIdx_isMTVmatch_branch != 0) sim_mdIdx_isMTVmatch();
  if (ph2_mdIdx_branch != 0) ph2_mdIdx();
  if (sg_hitIdx_branch != 0) sg_hitIdx();
  if (sg_simTrkIdx_branch != 0) sg_simTrkIdx();
  if (sg_simTrkIdx_anchorMatching_branch != 0) sg_simTrkIdx_anchorMatching();
  if (sg_layer_branch != 0) sg_layer();
  if (sg_pt_branch != 0) sg_pt();
  if (sg_eta_branch != 0) sg_eta();
  if (sg_phi_branch != 0) sg_phi();
  if (sg_sim_pt_branch != 0) sg_sim_pt();
  if (sg_sim_eta_branch != 0) sg_sim_eta();
  if (sg_sim_phi_branch != 0) sg_sim_phi();
  if (sim_sgIdx_branch != 0) sim_sgIdx();
  if (sim_sgIdx_isMTVmatch_branch != 0) sim_sgIdx_isMTVmatch();
  if (psg_hitIdx_branch != 0) psg_hitIdx();
  if (psg_simTrkIdx_branch != 0) psg_simTrkIdx();
  if (psg_simTrkIdx_anchorMatching_branch != 0) psg_simTrkIdx_anchorMatching();
  if (psg_layer_branch != 0) psg_layer();
  if (psg_pt_branch != 0) psg_pt();
  if (psg_eta_branch != 0) psg_eta();
  if (psg_phi_branch != 0) psg_phi();
  if (psg_sim_pt_branch != 0) psg_sim_pt();
  if (psg_sim_eta_branch != 0) psg_sim_eta();
  if (psg_sim_phi_branch != 0) psg_sim_phi();
  if (sim_psgIdx_branch != 0) sim_psgIdx();
  if (sim_psgIdx_isMTVmatch_branch != 0) sim_psgIdx_isMTVmatch();
  if (tp_hitIdx_branch != 0) tp_hitIdx();
  if (tp_simTrkIdx_branch != 0) tp_simTrkIdx();
  if (tp_layer_branch != 0) tp_layer();
  if (tp_pt_branch != 0) tp_pt();
  if (tp_eta_branch != 0) tp_eta();
  if (tp_phi_branch != 0) tp_phi();
  if (tp_sim_pt_branch != 0) tp_sim_pt();
  if (tp_sim_eta_branch != 0) tp_sim_eta();
  if (tp_sim_phi_branch != 0) tp_sim_phi();
  if (sim_tpIdx_branch != 0) sim_tpIdx();
  if (sim_tpIdx_isMTVmatch_branch != 0) sim_tpIdx_isMTVmatch();
  if (qp_hitIdx_branch != 0) qp_hitIdx();
  if (qp_simTrkIdx_branch != 0) qp_simTrkIdx();
  if (qp_layer_branch != 0) qp_layer();
  if (qp_pt_branch != 0) qp_pt();
  if (qp_eta_branch != 0) qp_eta();
  if (qp_phi_branch != 0) qp_phi();
  if (qp_sim_pt_branch != 0) qp_sim_pt();
  if (qp_sim_eta_branch != 0) qp_sim_eta();
  if (qp_sim_phi_branch != 0) qp_sim_phi();
  if (sim_qpIdx_branch != 0) sim_qpIdx();
  if (sim_qpIdx_isMTVmatch_branch != 0) sim_qpIdx_isMTVmatch();
  if (pqp_hitIdx_branch != 0) pqp_hitIdx();
  if (pqp_simTrkIdx_branch != 0) pqp_simTrkIdx();
  if (pqp_layer_branch != 0) pqp_layer();
  if (pqp_pt_branch != 0) pqp_pt();
  if (pqp_eta_branch != 0) pqp_eta();
  if (pqp_phi_branch != 0) pqp_phi();
  if (pqp_sim_pt_branch != 0) pqp_sim_pt();
  if (pqp_sim_eta_branch != 0) pqp_sim_eta();
  if (pqp_sim_phi_branch != 0) pqp_sim_phi();
  if (sim_pqpIdx_branch != 0) sim_pqpIdx();
  if (sim_pqpIdx_isMTVmatch_branch != 0) sim_pqpIdx_isMTVmatch();
  if (tc_hitIdx_branch != 0) tc_hitIdx();
  if (tc_simTrkIdx_branch != 0) tc_simTrkIdx();
  if (tc_layer_branch != 0) tc_layer();
  if (tc_pt_branch != 0) tc_pt();
  if (tc_eta_branch != 0) tc_eta();
  if (tc_phi_branch != 0) tc_phi();
  if (tc_sim_pt_branch != 0) tc_sim_pt();
  if (tc_sim_eta_branch != 0) tc_sim_eta();
  if (tc_sim_phi_branch != 0) tc_sim_phi();
  if (sim_tcIdx_branch != 0) sim_tcIdx();
  if (sim_tcIdx_isMTVmatch_branch != 0) sim_tcIdx_isMTVmatch();
}

const vector<float> &SDL::ph2_x() {
  if (not ph2_x_isLoaded) {
    if (ph2_x_branch != 0) {
      ph2_x_branch->GetEntry(index);
    } else {
      printf("branch ph2_x_branch does not exist!\n");
      exit(1);
    }
    ph2_x_isLoaded = true;
  }
  return *ph2_x_;
}

const vector<float> &SDL::ph2_y() {
  if (not ph2_y_isLoaded) {
    if (ph2_y_branch != 0) {
      ph2_y_branch->GetEntry(index);
    } else {
      printf("branch ph2_y_branch does not exist!\n");
      exit(1);
    }
    ph2_y_isLoaded = true;
  }
  return *ph2_y_;
}

const vector<float> &SDL::ph2_z() {
  if (not ph2_z_isLoaded) {
    if (ph2_z_branch != 0) {
      ph2_z_branch->GetEntry(index);
    } else {
      printf("branch ph2_z_branch does not exist!\n");
      exit(1);
    }
    ph2_z_isLoaded = true;
  }
  return *ph2_z_;
}

const vector<unsigned int> &SDL::ph2_detId() {
  if (not ph2_detId_isLoaded) {
    if (ph2_detId_branch != 0) {
      ph2_detId_branch->GetEntry(index);
    } else {
      printf("branch ph2_detId_branch does not exist!\n");
      exit(1);
    }
    ph2_detId_isLoaded = true;
  }
  return *ph2_detId_;
}

const vector<vector<int> > &SDL::ph2_simHitIdx() {
  if (not ph2_simHitIdx_isLoaded) {
    if (ph2_simHitIdx_branch != 0) {
      ph2_simHitIdx_branch->GetEntry(index);
    } else {
      printf("branch ph2_simHitIdx_branch does not exist!\n");
      exit(1);
    }
    ph2_simHitIdx_isLoaded = true;
  }
  return *ph2_simHitIdx_;
}

const vector<unsigned int> &SDL::ph2_simType() {
  if (not ph2_simType_isLoaded) {
    if (ph2_simType_branch != 0) {
      ph2_simType_branch->GetEntry(index);
    } else {
      printf("branch ph2_simType_branch does not exist!\n");
      exit(1);
    }
    ph2_simType_isLoaded = true;
  }
  return *ph2_simType_;
}

const vector<int> &SDL::ph2_anchorLayer() {
  if (not ph2_anchorLayer_isLoaded) {
    if (ph2_anchorLayer_branch != 0) {
      ph2_anchorLayer_branch->GetEntry(index);
    } else {
      printf("branch ph2_anchorLayer_branch does not exist!\n");
      exit(1);
    }
    ph2_anchorLayer_isLoaded = true;
  }
  return *ph2_anchorLayer_;
}

const vector<float> &SDL::simhit_x() {
  if (not simhit_x_isLoaded) {
    if (simhit_x_branch != 0) {
      simhit_x_branch->GetEntry(index);
    } else {
      printf("branch simhit_x_branch does not exist!\n");
      exit(1);
    }
    simhit_x_isLoaded = true;
  }
  return *simhit_x_;
}

const vector<float> &SDL::simhit_y() {
  if (not simhit_y_isLoaded) {
    if (simhit_y_branch != 0) {
      simhit_y_branch->GetEntry(index);
    } else {
      printf("branch simhit_y_branch does not exist!\n");
      exit(1);
    }
    simhit_y_isLoaded = true;
  }
  return *simhit_y_;
}

const vector<float> &SDL::simhit_z() {
  if (not simhit_z_isLoaded) {
    if (simhit_z_branch != 0) {
      simhit_z_branch->GetEntry(index);
    } else {
      printf("branch simhit_z_branch does not exist!\n");
      exit(1);
    }
    simhit_z_isLoaded = true;
  }
  return *simhit_z_;
}

const vector<unsigned int> &SDL::simhit_detId() {
  if (not simhit_detId_isLoaded) {
    if (simhit_detId_branch != 0) {
      simhit_detId_branch->GetEntry(index);
    } else {
      printf("branch simhit_detId_branch does not exist!\n");
      exit(1);
    }
    simhit_detId_isLoaded = true;
  }
  return *simhit_detId_;
}

const vector<unsigned int> &SDL::simhit_partnerDetId() {
  if (not simhit_partnerDetId_isLoaded) {
    if (simhit_partnerDetId_branch != 0) {
      simhit_partnerDetId_branch->GetEntry(index);
    } else {
      printf("branch simhit_partnerDetId_branch does not exist!\n");
      exit(1);
    }
    simhit_partnerDetId_isLoaded = true;
  }
  return *simhit_partnerDetId_;
}

const vector<unsigned int> &SDL::simhit_subdet() {
  if (not simhit_subdet_isLoaded) {
    if (simhit_subdet_branch != 0) {
      simhit_subdet_branch->GetEntry(index);
    } else {
      printf("branch simhit_subdet_branch does not exist!\n");
      exit(1);
    }
    simhit_subdet_isLoaded = true;
  }
  return *simhit_subdet_;
}

const vector<int> &SDL::simhit_particle() {
  if (not simhit_particle_isLoaded) {
    if (simhit_particle_branch != 0) {
      simhit_particle_branch->GetEntry(index);
    } else {
      printf("branch simhit_particle_branch does not exist!\n");
      exit(1);
    }
    simhit_particle_isLoaded = true;
  }
  return *simhit_particle_;
}

const vector<vector<int> > &SDL::simhit_hitIdx() {
  if (not simhit_hitIdx_isLoaded) {
    if (simhit_hitIdx_branch != 0) {
      simhit_hitIdx_branch->GetEntry(index);
    } else {
      printf("branch simhit_hitIdx_branch does not exist!\n");
      exit(1);
    }
    simhit_hitIdx_isLoaded = true;
  }
  return *simhit_hitIdx_;
}

const vector<int> &SDL::simhit_simTrkIdx() {
  if (not simhit_simTrkIdx_isLoaded) {
    if (simhit_simTrkIdx_branch != 0) {
      simhit_simTrkIdx_branch->GetEntry(index);
    } else {
      printf("branch simhit_simTrkIdx_branch does not exist!\n");
      exit(1);
    }
    simhit_simTrkIdx_isLoaded = true;
  }
  return *simhit_simTrkIdx_;
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

const vector<int> &SDL::sim_hasAll12HitsInBarrel() {
  if (not sim_hasAll12HitsInBarrel_isLoaded) {
    if (sim_hasAll12HitsInBarrel_branch != 0) {
      sim_hasAll12HitsInBarrel_branch->GetEntry(index);
    } else {
      printf("branch sim_hasAll12HitsInBarrel_branch does not exist!\n");
      exit(1);
    }
    sim_hasAll12HitsInBarrel_isLoaded = true;
  }
  return *sim_hasAll12HitsInBarrel_;
}

const vector<vector<int> > &SDL::sim_simHitIdx() {
  if (not sim_simHitIdx_isLoaded) {
    if (sim_simHitIdx_branch != 0) {
      sim_simHitIdx_branch->GetEntry(index);
    } else {
      printf("branch sim_simHitIdx_branch does not exist!\n");
      exit(1);
    }
    sim_simHitIdx_isLoaded = true;
  }
  return *sim_simHitIdx_;
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

const vector<vector<int> > &SDL::sim_simHitBoth() {
  if (not sim_simHitBoth_isLoaded) {
    if (sim_simHitBoth_branch != 0) {
      sim_simHitBoth_branch->GetEntry(index);
    } else {
      printf("branch sim_simHitBoth_branch does not exist!\n");
      exit(1);
    }
    sim_simHitBoth_isLoaded = true;
  }
  return *sim_simHitBoth_;
}

const vector<vector<float> > &SDL::sim_simHitDrFracWithHelix() {
  if (not sim_simHitDrFracWithHelix_isLoaded) {
    if (sim_simHitDrFracWithHelix_branch != 0) {
      sim_simHitDrFracWithHelix_branch->GetEntry(index);
    } else {
      printf("branch sim_simHitDrFracWithHelix_branch does not exist!\n");
      exit(1);
    }
    sim_simHitDrFracWithHelix_isLoaded = true;
  }
  return *sim_simHitDrFracWithHelix_;
}

const vector<vector<float> > &SDL::sim_simHitDistXyWithHelix() {
  if (not sim_simHitDistXyWithHelix_isLoaded) {
    if (sim_simHitDistXyWithHelix_branch != 0) {
      sim_simHitDistXyWithHelix_branch->GetEntry(index);
    } else {
      printf("branch sim_simHitDistXyWithHelix_branch does not exist!\n");
      exit(1);
    }
    sim_simHitDistXyWithHelix_isLoaded = true;
  }
  return *sim_simHitDistXyWithHelix_;
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

const vector<float> &SDL::see_stateTrajGlbPx() {
  if (not see_stateTrajGlbPx_isLoaded) {
    if (see_stateTrajGlbPx_branch != 0) {
      see_stateTrajGlbPx_branch->GetEntry(index);
    } else {
      printf("branch see_stateTrajGlbPx_branch does not exist!\n");
      exit(1);
    }
    see_stateTrajGlbPx_isLoaded = true;
  }
  return *see_stateTrajGlbPx_;
}

const vector<float> &SDL::see_stateTrajGlbPy() {
  if (not see_stateTrajGlbPy_isLoaded) {
    if (see_stateTrajGlbPy_branch != 0) {
      see_stateTrajGlbPy_branch->GetEntry(index);
    } else {
      printf("branch see_stateTrajGlbPy_branch does not exist!\n");
      exit(1);
    }
    see_stateTrajGlbPy_isLoaded = true;
  }
  return *see_stateTrajGlbPy_;
}

const vector<float> &SDL::see_stateTrajGlbPz() {
  if (not see_stateTrajGlbPz_isLoaded) {
    if (see_stateTrajGlbPz_branch != 0) {
      see_stateTrajGlbPz_branch->GetEntry(index);
    } else {
      printf("branch see_stateTrajGlbPz_branch does not exist!\n");
      exit(1);
    }
    see_stateTrajGlbPz_isLoaded = true;
  }
  return *see_stateTrajGlbPz_;
}

const vector<float> &SDL::see_stateTrajGlbX() {
  if (not see_stateTrajGlbX_isLoaded) {
    if (see_stateTrajGlbX_branch != 0) {
      see_stateTrajGlbX_branch->GetEntry(index);
    } else {
      printf("branch see_stateTrajGlbX_branch does not exist!\n");
      exit(1);
    }
    see_stateTrajGlbX_isLoaded = true;
  }
  return *see_stateTrajGlbX_;
}

const vector<float> &SDL::see_stateTrajGlbY() {
  if (not see_stateTrajGlbY_isLoaded) {
    if (see_stateTrajGlbY_branch != 0) {
      see_stateTrajGlbY_branch->GetEntry(index);
    } else {
      printf("branch see_stateTrajGlbY_branch does not exist!\n");
      exit(1);
    }
    see_stateTrajGlbY_isLoaded = true;
  }
  return *see_stateTrajGlbY_;
}

const vector<float> &SDL::see_stateTrajGlbZ() {
  if (not see_stateTrajGlbZ_isLoaded) {
    if (see_stateTrajGlbZ_branch != 0) {
      see_stateTrajGlbZ_branch->GetEntry(index);
    } else {
      printf("branch see_stateTrajGlbZ_branch does not exist!\n");
      exit(1);
    }
    see_stateTrajGlbZ_isLoaded = true;
  }
  return *see_stateTrajGlbZ_;
}

const vector<float> &SDL::see_px() {
  if (not see_px_isLoaded) {
    if (see_px_branch != 0) {
      see_px_branch->GetEntry(index);
    } else {
      printf("branch see_px_branch does not exist!\n");
      exit(1);
    }
    see_px_isLoaded = true;
  }
  return *see_px_;
}

const vector<float> &SDL::see_py() {
  if (not see_py_isLoaded) {
    if (see_py_branch != 0) {
      see_py_branch->GetEntry(index);
    } else {
      printf("branch see_py_branch does not exist!\n");
      exit(1);
    }
    see_py_isLoaded = true;
  }
  return *see_py_;
}

const vector<float> &SDL::see_pz() {
  if (not see_pz_isLoaded) {
    if (see_pz_branch != 0) {
      see_pz_branch->GetEntry(index);
    } else {
      printf("branch see_pz_branch does not exist!\n");
      exit(1);
    }
    see_pz_isLoaded = true;
  }
  return *see_pz_;
}

const vector<float> &SDL::see_ptErr() {
  if (not see_ptErr_isLoaded) {
    if (see_ptErr_branch != 0) {
      see_ptErr_branch->GetEntry(index);
    } else {
      printf("branch see_ptErr_branch does not exist!\n");
      exit(1);
    }
    see_ptErr_isLoaded = true;
  }
  return *see_ptErr_;
}

const vector<float> &SDL::see_dxy() {
  if (not see_dxy_isLoaded) {
    if (see_dxy_branch != 0) {
      see_dxy_branch->GetEntry(index);
    } else {
      printf("branch see_dxy_branch does not exist!\n");
      exit(1);
    }
    see_dxy_isLoaded = true;
  }
  return *see_dxy_;
}

const vector<float> &SDL::see_dxyErr() {
  if (not see_dxyErr_isLoaded) {
    if (see_dxyErr_branch != 0) {
      see_dxyErr_branch->GetEntry(index);
    } else {
      printf("branch see_dxyErr_branch does not exist!\n");
      exit(1);
    }
    see_dxyErr_isLoaded = true;
  }
  return *see_dxyErr_;
}

const vector<float> &SDL::see_dz() {
  if (not see_dz_isLoaded) {
    if (see_dz_branch != 0) {
      see_dz_branch->GetEntry(index);
    } else {
      printf("branch see_dz_branch does not exist!\n");
      exit(1);
    }
    see_dz_isLoaded = true;
  }
  return *see_dz_;
}

const vector<vector<int> > &SDL::see_hitIdx() {
  if (not see_hitIdx_isLoaded) {
    if (see_hitIdx_branch != 0) {
      see_hitIdx_branch->GetEntry(index);
    } else {
      printf("branch see_hitIdx_branch does not exist!\n");
      exit(1);
    }
    see_hitIdx_isLoaded = true;
  }
  return *see_hitIdx_;
}

const vector<vector<int> > &SDL::see_hitType() {
  if (not see_hitType_isLoaded) {
    if (see_hitType_branch != 0) {
      see_hitType_branch->GetEntry(index);
    } else {
      printf("branch see_hitType_branch does not exist!\n");
      exit(1);
    }
    see_hitType_isLoaded = true;
  }
  return *see_hitType_;
}

const vector<vector<int> > &SDL::see_simTrkIdx() {
  if (not see_simTrkIdx_isLoaded) {
    if (see_simTrkIdx_branch != 0) {
      see_simTrkIdx_branch->GetEntry(index);
    } else {
      printf("branch see_simTrkIdx_branch does not exist!\n");
      exit(1);
    }
    see_simTrkIdx_isLoaded = true;
  }
  return *see_simTrkIdx_;
}

const vector<unsigned int> &SDL::see_algo() {
  if (not see_algo_isLoaded) {
    if (see_algo_branch != 0) {
      see_algo_branch->GetEntry(index);
    } else {
      printf("branch see_algo_branch does not exist!\n");
      exit(1);
    }
    see_algo_isLoaded = true;
  }
  return *see_algo_;
}

const vector<vector<int> > &SDL::md_hitIdx() {
  if (not md_hitIdx_isLoaded) {
    if (md_hitIdx_branch != 0) {
      md_hitIdx_branch->GetEntry(index);
    } else {
      printf("branch md_hitIdx_branch does not exist!\n");
      exit(1);
    }
    md_hitIdx_isLoaded = true;
  }
  return *md_hitIdx_;
}

const vector<vector<int> > &SDL::md_simTrkIdx() {
  if (not md_simTrkIdx_isLoaded) {
    if (md_simTrkIdx_branch != 0) {
      md_simTrkIdx_branch->GetEntry(index);
    } else {
      printf("branch md_simTrkIdx_branch does not exist!\n");
      exit(1);
    }
    md_simTrkIdx_isLoaded = true;
  }
  return *md_simTrkIdx_;
}

const vector<vector<int> > &SDL::md_layer() {
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

const vector<float> &SDL::md_sim_pt() {
  if (not md_sim_pt_isLoaded) {
    if (md_sim_pt_branch != 0) {
      md_sim_pt_branch->GetEntry(index);
    } else {
      printf("branch md_sim_pt_branch does not exist!\n");
      exit(1);
    }
    md_sim_pt_isLoaded = true;
  }
  return *md_sim_pt_;
}

const vector<float> &SDL::md_sim_eta() {
  if (not md_sim_eta_isLoaded) {
    if (md_sim_eta_branch != 0) {
      md_sim_eta_branch->GetEntry(index);
    } else {
      printf("branch md_sim_eta_branch does not exist!\n");
      exit(1);
    }
    md_sim_eta_isLoaded = true;
  }
  return *md_sim_eta_;
}

const vector<float> &SDL::md_sim_phi() {
  if (not md_sim_phi_isLoaded) {
    if (md_sim_phi_branch != 0) {
      md_sim_phi_branch->GetEntry(index);
    } else {
      printf("branch md_sim_phi_branch does not exist!\n");
      exit(1);
    }
    md_sim_phi_isLoaded = true;
  }
  return *md_sim_phi_;
}

const vector<float> &SDL::md_type() {
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

const vector<float> &SDL::md_dz() {
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

const vector<float> &SDL::md_dzCut() {
  if (not md_dzCut_isLoaded) {
    if (md_dzCut_branch != 0) {
      md_dzCut_branch->GetEntry(index);
    } else {
      printf("branch md_dzCut_branch does not exist!\n");
      exit(1);
    }
    md_dzCut_isLoaded = true;
  }
  return *md_dzCut_;
}

const vector<float> &SDL::md_drt() {
  if (not md_drt_isLoaded) {
    if (md_drt_branch != 0) {
      md_drt_branch->GetEntry(index);
    } else {
      printf("branch md_drt_branch does not exist!\n");
      exit(1);
    }
    md_drt_isLoaded = true;
  }
  return *md_drt_;
}

const vector<float> &SDL::md_drtCut() {
  if (not md_drtCut_isLoaded) {
    if (md_drtCut_branch != 0) {
      md_drtCut_branch->GetEntry(index);
    } else {
      printf("branch md_drtCut_branch does not exist!\n");
      exit(1);
    }
    md_drtCut_isLoaded = true;
  }
  return *md_drtCut_;
}

const vector<float> &SDL::md_miniCut() {
  if (not md_miniCut_isLoaded) {
    if (md_miniCut_branch != 0) {
      md_miniCut_branch->GetEntry(index);
    } else {
      printf("branch md_miniCut_branch does not exist!\n");
      exit(1);
    }
    md_miniCut_isLoaded = true;
  }
  return *md_miniCut_;
}

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
  return *md_dphi_;
}

const vector<float> &SDL::md_dphiChange() {
  if (not md_dphiChange_isLoaded) {
    if (md_dphiChange_branch != 0) {
      md_dphiChange_branch->GetEntry(index);
    } else {
      printf("branch md_dphiChange_branch does not exist!\n");
      exit(1);
    }
    md_dphiChange_isLoaded = true;
  }
  return *md_dphiChange_;
}

const vector<vector<int> > &SDL::sim_mdIdx() {
  if (not sim_mdIdx_isLoaded) {
    if (sim_mdIdx_branch != 0) {
      sim_mdIdx_branch->GetEntry(index);
    } else {
      printf("branch sim_mdIdx_branch does not exist!\n");
      exit(1);
    }
    sim_mdIdx_isLoaded = true;
  }
  return *sim_mdIdx_;
}

const vector<vector<int> > &SDL::sim_mdIdx_isMTVmatch() {
  if (not sim_mdIdx_isMTVmatch_isLoaded) {
    if (sim_mdIdx_isMTVmatch_branch != 0) {
      sim_mdIdx_isMTVmatch_branch->GetEntry(index);
    } else {
      printf("branch sim_mdIdx_isMTVmatch_branch does not exist!\n");
      exit(1);
    }
    sim_mdIdx_isMTVmatch_isLoaded = true;
  }
  return *sim_mdIdx_isMTVmatch_;
}

const vector<vector<int> > &SDL::ph2_mdIdx() {
  if (not ph2_mdIdx_isLoaded) {
    if (ph2_mdIdx_branch != 0) {
      ph2_mdIdx_branch->GetEntry(index);
    } else {
      printf("branch ph2_mdIdx_branch does not exist!\n");
      exit(1);
    }
    ph2_mdIdx_isLoaded = true;
  }
  return *ph2_mdIdx_;
}

const vector<vector<int> > &SDL::sg_hitIdx() {
  if (not sg_hitIdx_isLoaded) {
    if (sg_hitIdx_branch != 0) {
      sg_hitIdx_branch->GetEntry(index);
    } else {
      printf("branch sg_hitIdx_branch does not exist!\n");
      exit(1);
    }
    sg_hitIdx_isLoaded = true;
  }
  return *sg_hitIdx_;
}

const vector<vector<int> > &SDL::sg_simTrkIdx() {
  if (not sg_simTrkIdx_isLoaded) {
    if (sg_simTrkIdx_branch != 0) {
      sg_simTrkIdx_branch->GetEntry(index);
    } else {
      printf("branch sg_simTrkIdx_branch does not exist!\n");
      exit(1);
    }
    sg_simTrkIdx_isLoaded = true;
  }
  return *sg_simTrkIdx_;
}

const vector<vector<int> > &SDL::sg_simTrkIdx_anchorMatching() {
  if (not sg_simTrkIdx_anchorMatching_isLoaded) {
    if (sg_simTrkIdx_anchorMatching_branch != 0) {
      sg_simTrkIdx_anchorMatching_branch->GetEntry(index);
    } else {
      printf("branch sg_simTrkIdx_anchorMatching_branch does not exist!\n");
      exit(1);
    }
    sg_simTrkIdx_anchorMatching_isLoaded = true;
  }
  return *sg_simTrkIdx_anchorMatching_;
}

const vector<vector<int> > &SDL::sg_layer() {
  if (not sg_layer_isLoaded) {
    if (sg_layer_branch != 0) {
      sg_layer_branch->GetEntry(index);
    } else {
      printf("branch sg_layer_branch does not exist!\n");
      exit(1);
    }
    sg_layer_isLoaded = true;
  }
  return *sg_layer_;
}

const vector<float> &SDL::sg_pt() {
  if (not sg_pt_isLoaded) {
    if (sg_pt_branch != 0) {
      sg_pt_branch->GetEntry(index);
    } else {
      printf("branch sg_pt_branch does not exist!\n");
      exit(1);
    }
    sg_pt_isLoaded = true;
  }
  return *sg_pt_;
}

const vector<float> &SDL::sg_eta() {
  if (not sg_eta_isLoaded) {
    if (sg_eta_branch != 0) {
      sg_eta_branch->GetEntry(index);
    } else {
      printf("branch sg_eta_branch does not exist!\n");
      exit(1);
    }
    sg_eta_isLoaded = true;
  }
  return *sg_eta_;
}

const vector<float> &SDL::sg_phi() {
  if (not sg_phi_isLoaded) {
    if (sg_phi_branch != 0) {
      sg_phi_branch->GetEntry(index);
    } else {
      printf("branch sg_phi_branch does not exist!\n");
      exit(1);
    }
    sg_phi_isLoaded = true;
  }
  return *sg_phi_;
}

const vector<float> &SDL::sg_sim_pt() {
  if (not sg_sim_pt_isLoaded) {
    if (sg_sim_pt_branch != 0) {
      sg_sim_pt_branch->GetEntry(index);
    } else {
      printf("branch sg_sim_pt_branch does not exist!\n");
      exit(1);
    }
    sg_sim_pt_isLoaded = true;
  }
  return *sg_sim_pt_;
}

const vector<float> &SDL::sg_sim_eta() {
  if (not sg_sim_eta_isLoaded) {
    if (sg_sim_eta_branch != 0) {
      sg_sim_eta_branch->GetEntry(index);
    } else {
      printf("branch sg_sim_eta_branch does not exist!\n");
      exit(1);
    }
    sg_sim_eta_isLoaded = true;
  }
  return *sg_sim_eta_;
}

const vector<float> &SDL::sg_sim_phi() {
  if (not sg_sim_phi_isLoaded) {
    if (sg_sim_phi_branch != 0) {
      sg_sim_phi_branch->GetEntry(index);
    } else {
      printf("branch sg_sim_phi_branch does not exist!\n");
      exit(1);
    }
    sg_sim_phi_isLoaded = true;
  }
  return *sg_sim_phi_;
}

const vector<vector<int> > &SDL::sim_sgIdx() {
  if (not sim_sgIdx_isLoaded) {
    if (sim_sgIdx_branch != 0) {
      sim_sgIdx_branch->GetEntry(index);
    } else {
      printf("branch sim_sgIdx_branch does not exist!\n");
      exit(1);
    }
    sim_sgIdx_isLoaded = true;
  }
  return *sim_sgIdx_;
}

const vector<vector<int> > &SDL::sim_sgIdx_isMTVmatch() {
  if (not sim_sgIdx_isMTVmatch_isLoaded) {
    if (sim_sgIdx_isMTVmatch_branch != 0) {
      sim_sgIdx_isMTVmatch_branch->GetEntry(index);
    } else {
      printf("branch sim_sgIdx_isMTVmatch_branch does not exist!\n");
      exit(1);
    }
    sim_sgIdx_isMTVmatch_isLoaded = true;
  }
  return *sim_sgIdx_isMTVmatch_;
}

const vector<vector<int> > &SDL::psg_hitIdx() {
  if (not psg_hitIdx_isLoaded) {
    if (psg_hitIdx_branch != 0) {
      psg_hitIdx_branch->GetEntry(index);
    } else {
      printf("branch psg_hitIdx_branch does not exist!\n");
      exit(1);
    }
    psg_hitIdx_isLoaded = true;
  }
  return *psg_hitIdx_;
}

const vector<vector<int> > &SDL::psg_simTrkIdx() {
  if (not psg_simTrkIdx_isLoaded) {
    if (psg_simTrkIdx_branch != 0) {
      psg_simTrkIdx_branch->GetEntry(index);
    } else {
      printf("branch psg_simTrkIdx_branch does not exist!\n");
      exit(1);
    }
    psg_simTrkIdx_isLoaded = true;
  }
  return *psg_simTrkIdx_;
}

const vector<vector<int> > &SDL::psg_simTrkIdx_anchorMatching() {
  if (not psg_simTrkIdx_anchorMatching_isLoaded) {
    if (psg_simTrkIdx_anchorMatching_branch != 0) {
      psg_simTrkIdx_anchorMatching_branch->GetEntry(index);
    } else {
      printf("branch psg_simTrkIdx_anchorMatching_branch does not exist!\n");
      exit(1);
    }
    psg_simTrkIdx_anchorMatching_isLoaded = true;
  }
  return *psg_simTrkIdx_anchorMatching_;
}

const vector<vector<int> > &SDL::psg_layer() {
  if (not psg_layer_isLoaded) {
    if (psg_layer_branch != 0) {
      psg_layer_branch->GetEntry(index);
    } else {
      printf("branch psg_layer_branch does not exist!\n");
      exit(1);
    }
    psg_layer_isLoaded = true;
  }
  return *psg_layer_;
}

const vector<float> &SDL::psg_pt() {
  if (not psg_pt_isLoaded) {
    if (psg_pt_branch != 0) {
      psg_pt_branch->GetEntry(index);
    } else {
      printf("branch psg_pt_branch does not exist!\n");
      exit(1);
    }
    psg_pt_isLoaded = true;
  }
  return *psg_pt_;
}

const vector<float> &SDL::psg_eta() {
  if (not psg_eta_isLoaded) {
    if (psg_eta_branch != 0) {
      psg_eta_branch->GetEntry(index);
    } else {
      printf("branch psg_eta_branch does not exist!\n");
      exit(1);
    }
    psg_eta_isLoaded = true;
  }
  return *psg_eta_;
}

const vector<float> &SDL::psg_phi() {
  if (not psg_phi_isLoaded) {
    if (psg_phi_branch != 0) {
      psg_phi_branch->GetEntry(index);
    } else {
      printf("branch psg_phi_branch does not exist!\n");
      exit(1);
    }
    psg_phi_isLoaded = true;
  }
  return *psg_phi_;
}

const vector<float> &SDL::psg_sim_pt() {
  if (not psg_sim_pt_isLoaded) {
    if (psg_sim_pt_branch != 0) {
      psg_sim_pt_branch->GetEntry(index);
    } else {
      printf("branch psg_sim_pt_branch does not exist!\n");
      exit(1);
    }
    psg_sim_pt_isLoaded = true;
  }
  return *psg_sim_pt_;
}

const vector<float> &SDL::psg_sim_eta() {
  if (not psg_sim_eta_isLoaded) {
    if (psg_sim_eta_branch != 0) {
      psg_sim_eta_branch->GetEntry(index);
    } else {
      printf("branch psg_sim_eta_branch does not exist!\n");
      exit(1);
    }
    psg_sim_eta_isLoaded = true;
  }
  return *psg_sim_eta_;
}

const vector<float> &SDL::psg_sim_phi() {
  if (not psg_sim_phi_isLoaded) {
    if (psg_sim_phi_branch != 0) {
      psg_sim_phi_branch->GetEntry(index);
    } else {
      printf("branch psg_sim_phi_branch does not exist!\n");
      exit(1);
    }
    psg_sim_phi_isLoaded = true;
  }
  return *psg_sim_phi_;
}

const vector<vector<int> > &SDL::sim_psgIdx() {
  if (not sim_psgIdx_isLoaded) {
    if (sim_psgIdx_branch != 0) {
      sim_psgIdx_branch->GetEntry(index);
    } else {
      printf("branch sim_psgIdx_branch does not exist!\n");
      exit(1);
    }
    sim_psgIdx_isLoaded = true;
  }
  return *sim_psgIdx_;
}

const vector<vector<int> > &SDL::sim_psgIdx_isMTVmatch() {
  if (not sim_psgIdx_isMTVmatch_isLoaded) {
    if (sim_psgIdx_isMTVmatch_branch != 0) {
      sim_psgIdx_isMTVmatch_branch->GetEntry(index);
    } else {
      printf("branch sim_psgIdx_isMTVmatch_branch does not exist!\n");
      exit(1);
    }
    sim_psgIdx_isMTVmatch_isLoaded = true;
  }
  return *sim_psgIdx_isMTVmatch_;
}

const vector<vector<int> > &SDL::tp_hitIdx() {
  if (not tp_hitIdx_isLoaded) {
    if (tp_hitIdx_branch != 0) {
      tp_hitIdx_branch->GetEntry(index);
    } else {
      printf("branch tp_hitIdx_branch does not exist!\n");
      exit(1);
    }
    tp_hitIdx_isLoaded = true;
  }
  return *tp_hitIdx_;
}

const vector<vector<int> > &SDL::tp_simTrkIdx() {
  if (not tp_simTrkIdx_isLoaded) {
    if (tp_simTrkIdx_branch != 0) {
      tp_simTrkIdx_branch->GetEntry(index);
    } else {
      printf("branch tp_simTrkIdx_branch does not exist!\n");
      exit(1);
    }
    tp_simTrkIdx_isLoaded = true;
  }
  return *tp_simTrkIdx_;
}

const vector<vector<int> > &SDL::tp_layer() {
  if (not tp_layer_isLoaded) {
    if (tp_layer_branch != 0) {
      tp_layer_branch->GetEntry(index);
    } else {
      printf("branch tp_layer_branch does not exist!\n");
      exit(1);
    }
    tp_layer_isLoaded = true;
  }
  return *tp_layer_;
}

const vector<float> &SDL::tp_pt() {
  if (not tp_pt_isLoaded) {
    if (tp_pt_branch != 0) {
      tp_pt_branch->GetEntry(index);
    } else {
      printf("branch tp_pt_branch does not exist!\n");
      exit(1);
    }
    tp_pt_isLoaded = true;
  }
  return *tp_pt_;
}

const vector<float> &SDL::tp_eta() {
  if (not tp_eta_isLoaded) {
    if (tp_eta_branch != 0) {
      tp_eta_branch->GetEntry(index);
    } else {
      printf("branch tp_eta_branch does not exist!\n");
      exit(1);
    }
    tp_eta_isLoaded = true;
  }
  return *tp_eta_;
}

const vector<float> &SDL::tp_phi() {
  if (not tp_phi_isLoaded) {
    if (tp_phi_branch != 0) {
      tp_phi_branch->GetEntry(index);
    } else {
      printf("branch tp_phi_branch does not exist!\n");
      exit(1);
    }
    tp_phi_isLoaded = true;
  }
  return *tp_phi_;
}

const vector<float> &SDL::tp_sim_pt() {
  if (not tp_sim_pt_isLoaded) {
    if (tp_sim_pt_branch != 0) {
      tp_sim_pt_branch->GetEntry(index);
    } else {
      printf("branch tp_sim_pt_branch does not exist!\n");
      exit(1);
    }
    tp_sim_pt_isLoaded = true;
  }
  return *tp_sim_pt_;
}

const vector<float> &SDL::tp_sim_eta() {
  if (not tp_sim_eta_isLoaded) {
    if (tp_sim_eta_branch != 0) {
      tp_sim_eta_branch->GetEntry(index);
    } else {
      printf("branch tp_sim_eta_branch does not exist!\n");
      exit(1);
    }
    tp_sim_eta_isLoaded = true;
  }
  return *tp_sim_eta_;
}

const vector<float> &SDL::tp_sim_phi() {
  if (not tp_sim_phi_isLoaded) {
    if (tp_sim_phi_branch != 0) {
      tp_sim_phi_branch->GetEntry(index);
    } else {
      printf("branch tp_sim_phi_branch does not exist!\n");
      exit(1);
    }
    tp_sim_phi_isLoaded = true;
  }
  return *tp_sim_phi_;
}

const vector<vector<int> > &SDL::sim_tpIdx() {
  if (not sim_tpIdx_isLoaded) {
    if (sim_tpIdx_branch != 0) {
      sim_tpIdx_branch->GetEntry(index);
    } else {
      printf("branch sim_tpIdx_branch does not exist!\n");
      exit(1);
    }
    sim_tpIdx_isLoaded = true;
  }
  return *sim_tpIdx_;
}

const vector<vector<int> > &SDL::sim_tpIdx_isMTVmatch() {
  if (not sim_tpIdx_isMTVmatch_isLoaded) {
    if (sim_tpIdx_isMTVmatch_branch != 0) {
      sim_tpIdx_isMTVmatch_branch->GetEntry(index);
    } else {
      printf("branch sim_tpIdx_isMTVmatch_branch does not exist!\n");
      exit(1);
    }
    sim_tpIdx_isMTVmatch_isLoaded = true;
  }
  return *sim_tpIdx_isMTVmatch_;
}

const vector<vector<int> > &SDL::qp_hitIdx() {
  if (not qp_hitIdx_isLoaded) {
    if (qp_hitIdx_branch != 0) {
      qp_hitIdx_branch->GetEntry(index);
    } else {
      printf("branch qp_hitIdx_branch does not exist!\n");
      exit(1);
    }
    qp_hitIdx_isLoaded = true;
  }
  return *qp_hitIdx_;
}

const vector<vector<int> > &SDL::qp_simTrkIdx() {
  if (not qp_simTrkIdx_isLoaded) {
    if (qp_simTrkIdx_branch != 0) {
      qp_simTrkIdx_branch->GetEntry(index);
    } else {
      printf("branch qp_simTrkIdx_branch does not exist!\n");
      exit(1);
    }
    qp_simTrkIdx_isLoaded = true;
  }
  return *qp_simTrkIdx_;
}

const vector<vector<int> > &SDL::qp_layer() {
  if (not qp_layer_isLoaded) {
    if (qp_layer_branch != 0) {
      qp_layer_branch->GetEntry(index);
    } else {
      printf("branch qp_layer_branch does not exist!\n");
      exit(1);
    }
    qp_layer_isLoaded = true;
  }
  return *qp_layer_;
}

const vector<float> &SDL::qp_pt() {
  if (not qp_pt_isLoaded) {
    if (qp_pt_branch != 0) {
      qp_pt_branch->GetEntry(index);
    } else {
      printf("branch qp_pt_branch does not exist!\n");
      exit(1);
    }
    qp_pt_isLoaded = true;
  }
  return *qp_pt_;
}

const vector<float> &SDL::qp_eta() {
  if (not qp_eta_isLoaded) {
    if (qp_eta_branch != 0) {
      qp_eta_branch->GetEntry(index);
    } else {
      printf("branch qp_eta_branch does not exist!\n");
      exit(1);
    }
    qp_eta_isLoaded = true;
  }
  return *qp_eta_;
}

const vector<float> &SDL::qp_phi() {
  if (not qp_phi_isLoaded) {
    if (qp_phi_branch != 0) {
      qp_phi_branch->GetEntry(index);
    } else {
      printf("branch qp_phi_branch does not exist!\n");
      exit(1);
    }
    qp_phi_isLoaded = true;
  }
  return *qp_phi_;
}

const vector<float> &SDL::qp_sim_pt() {
  if (not qp_sim_pt_isLoaded) {
    if (qp_sim_pt_branch != 0) {
      qp_sim_pt_branch->GetEntry(index);
    } else {
      printf("branch qp_sim_pt_branch does not exist!\n");
      exit(1);
    }
    qp_sim_pt_isLoaded = true;
  }
  return *qp_sim_pt_;
}

const vector<float> &SDL::qp_sim_eta() {
  if (not qp_sim_eta_isLoaded) {
    if (qp_sim_eta_branch != 0) {
      qp_sim_eta_branch->GetEntry(index);
    } else {
      printf("branch qp_sim_eta_branch does not exist!\n");
      exit(1);
    }
    qp_sim_eta_isLoaded = true;
  }
  return *qp_sim_eta_;
}

const vector<float> &SDL::qp_sim_phi() {
  if (not qp_sim_phi_isLoaded) {
    if (qp_sim_phi_branch != 0) {
      qp_sim_phi_branch->GetEntry(index);
    } else {
      printf("branch qp_sim_phi_branch does not exist!\n");
      exit(1);
    }
    qp_sim_phi_isLoaded = true;
  }
  return *qp_sim_phi_;
}

const vector<vector<int> > &SDL::sim_qpIdx() {
  if (not sim_qpIdx_isLoaded) {
    if (sim_qpIdx_branch != 0) {
      sim_qpIdx_branch->GetEntry(index);
    } else {
      printf("branch sim_qpIdx_branch does not exist!\n");
      exit(1);
    }
    sim_qpIdx_isLoaded = true;
  }
  return *sim_qpIdx_;
}

const vector<vector<int> > &SDL::sim_qpIdx_isMTVmatch() {
  if (not sim_qpIdx_isMTVmatch_isLoaded) {
    if (sim_qpIdx_isMTVmatch_branch != 0) {
      sim_qpIdx_isMTVmatch_branch->GetEntry(index);
    } else {
      printf("branch sim_qpIdx_isMTVmatch_branch does not exist!\n");
      exit(1);
    }
    sim_qpIdx_isMTVmatch_isLoaded = true;
  }
  return *sim_qpIdx_isMTVmatch_;
}

const vector<vector<int> > &SDL::pqp_hitIdx() {
  if (not pqp_hitIdx_isLoaded) {
    if (pqp_hitIdx_branch != 0) {
      pqp_hitIdx_branch->GetEntry(index);
    } else {
      printf("branch pqp_hitIdx_branch does not exist!\n");
      exit(1);
    }
    pqp_hitIdx_isLoaded = true;
  }
  return *pqp_hitIdx_;
}

const vector<vector<int> > &SDL::pqp_simTrkIdx() {
  if (not pqp_simTrkIdx_isLoaded) {
    if (pqp_simTrkIdx_branch != 0) {
      pqp_simTrkIdx_branch->GetEntry(index);
    } else {
      printf("branch pqp_simTrkIdx_branch does not exist!\n");
      exit(1);
    }
    pqp_simTrkIdx_isLoaded = true;
  }
  return *pqp_simTrkIdx_;
}

const vector<vector<int> > &SDL::pqp_layer() {
  if (not pqp_layer_isLoaded) {
    if (pqp_layer_branch != 0) {
      pqp_layer_branch->GetEntry(index);
    } else {
      printf("branch pqp_layer_branch does not exist!\n");
      exit(1);
    }
    pqp_layer_isLoaded = true;
  }
  return *pqp_layer_;
}

const vector<float> &SDL::pqp_pt() {
  if (not pqp_pt_isLoaded) {
    if (pqp_pt_branch != 0) {
      pqp_pt_branch->GetEntry(index);
    } else {
      printf("branch pqp_pt_branch does not exist!\n");
      exit(1);
    }
    pqp_pt_isLoaded = true;
  }
  return *pqp_pt_;
}

const vector<float> &SDL::pqp_eta() {
  if (not pqp_eta_isLoaded) {
    if (pqp_eta_branch != 0) {
      pqp_eta_branch->GetEntry(index);
    } else {
      printf("branch pqp_eta_branch does not exist!\n");
      exit(1);
    }
    pqp_eta_isLoaded = true;
  }
  return *pqp_eta_;
}

const vector<float> &SDL::pqp_phi() {
  if (not pqp_phi_isLoaded) {
    if (pqp_phi_branch != 0) {
      pqp_phi_branch->GetEntry(index);
    } else {
      printf("branch pqp_phi_branch does not exist!\n");
      exit(1);
    }
    pqp_phi_isLoaded = true;
  }
  return *pqp_phi_;
}

const vector<float> &SDL::pqp_sim_pt() {
  if (not pqp_sim_pt_isLoaded) {
    if (pqp_sim_pt_branch != 0) {
      pqp_sim_pt_branch->GetEntry(index);
    } else {
      printf("branch pqp_sim_pt_branch does not exist!\n");
      exit(1);
    }
    pqp_sim_pt_isLoaded = true;
  }
  return *pqp_sim_pt_;
}

const vector<float> &SDL::pqp_sim_eta() {
  if (not pqp_sim_eta_isLoaded) {
    if (pqp_sim_eta_branch != 0) {
      pqp_sim_eta_branch->GetEntry(index);
    } else {
      printf("branch pqp_sim_eta_branch does not exist!\n");
      exit(1);
    }
    pqp_sim_eta_isLoaded = true;
  }
  return *pqp_sim_eta_;
}

const vector<float> &SDL::pqp_sim_phi() {
  if (not pqp_sim_phi_isLoaded) {
    if (pqp_sim_phi_branch != 0) {
      pqp_sim_phi_branch->GetEntry(index);
    } else {
      printf("branch pqp_sim_phi_branch does not exist!\n");
      exit(1);
    }
    pqp_sim_phi_isLoaded = true;
  }
  return *pqp_sim_phi_;
}

const vector<vector<int> > &SDL::sim_pqpIdx() {
  if (not sim_pqpIdx_isLoaded) {
    if (sim_pqpIdx_branch != 0) {
      sim_pqpIdx_branch->GetEntry(index);
    } else {
      printf("branch sim_pqpIdx_branch does not exist!\n");
      exit(1);
    }
    sim_pqpIdx_isLoaded = true;
  }
  return *sim_pqpIdx_;
}

const vector<vector<int> > &SDL::sim_pqpIdx_isMTVmatch() {
  if (not sim_pqpIdx_isMTVmatch_isLoaded) {
    if (sim_pqpIdx_isMTVmatch_branch != 0) {
      sim_pqpIdx_isMTVmatch_branch->GetEntry(index);
    } else {
      printf("branch sim_pqpIdx_isMTVmatch_branch does not exist!\n");
      exit(1);
    }
    sim_pqpIdx_isMTVmatch_isLoaded = true;
  }
  return *sim_pqpIdx_isMTVmatch_;
}

const vector<vector<int> > &SDL::tc_hitIdx() {
  if (not tc_hitIdx_isLoaded) {
    if (tc_hitIdx_branch != 0) {
      tc_hitIdx_branch->GetEntry(index);
    } else {
      printf("branch tc_hitIdx_branch does not exist!\n");
      exit(1);
    }
    tc_hitIdx_isLoaded = true;
  }
  return *tc_hitIdx_;
}

const vector<vector<int> > &SDL::tc_simTrkIdx() {
  if (not tc_simTrkIdx_isLoaded) {
    if (tc_simTrkIdx_branch != 0) {
      tc_simTrkIdx_branch->GetEntry(index);
    } else {
      printf("branch tc_simTrkIdx_branch does not exist!\n");
      exit(1);
    }
    tc_simTrkIdx_isLoaded = true;
  }
  return *tc_simTrkIdx_;
}

const vector<vector<int> > &SDL::tc_layer() {
  if (not tc_layer_isLoaded) {
    if (tc_layer_branch != 0) {
      tc_layer_branch->GetEntry(index);
    } else {
      printf("branch tc_layer_branch does not exist!\n");
      exit(1);
    }
    tc_layer_isLoaded = true;
  }
  return *tc_layer_;
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

const vector<float> &SDL::tc_sim_pt() {
  if (not tc_sim_pt_isLoaded) {
    if (tc_sim_pt_branch != 0) {
      tc_sim_pt_branch->GetEntry(index);
    } else {
      printf("branch tc_sim_pt_branch does not exist!\n");
      exit(1);
    }
    tc_sim_pt_isLoaded = true;
  }
  return *tc_sim_pt_;
}

const vector<float> &SDL::tc_sim_eta() {
  if (not tc_sim_eta_isLoaded) {
    if (tc_sim_eta_branch != 0) {
      tc_sim_eta_branch->GetEntry(index);
    } else {
      printf("branch tc_sim_eta_branch does not exist!\n");
      exit(1);
    }
    tc_sim_eta_isLoaded = true;
  }
  return *tc_sim_eta_;
}

const vector<float> &SDL::tc_sim_phi() {
  if (not tc_sim_phi_isLoaded) {
    if (tc_sim_phi_branch != 0) {
      tc_sim_phi_branch->GetEntry(index);
    } else {
      printf("branch tc_sim_phi_branch does not exist!\n");
      exit(1);
    }
    tc_sim_phi_isLoaded = true;
  }
  return *tc_sim_phi_;
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

const vector<vector<int> > &SDL::sim_tcIdx_isMTVmatch() {
  if (not sim_tcIdx_isMTVmatch_isLoaded) {
    if (sim_tcIdx_isMTVmatch_branch != 0) {
      sim_tcIdx_isMTVmatch_branch->GetEntry(index);
    } else {
      printf("branch sim_tcIdx_isMTVmatch_branch does not exist!\n");
      exit(1);
    }
    sim_tcIdx_isMTVmatch_isLoaded = true;
  }
  return *sim_tcIdx_isMTVmatch_;
}


void SDL::progress( int nEventsTotal, int nEventsChain ){
  int period = 1000;
  if (nEventsTotal%1000 == 0) {
    // xterm magic from L. Vacavant and A. Cerri
    if (isatty(1)) {
      if ((nEventsChain - nEventsTotal) > period) {
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

const vector<float> &ph2_x() { return sdl.ph2_x(); }
const vector<float> &ph2_y() { return sdl.ph2_y(); }
const vector<float> &ph2_z() { return sdl.ph2_z(); }
const vector<unsigned int> &ph2_detId() { return sdl.ph2_detId(); }
const vector<vector<int> > &ph2_simHitIdx() { return sdl.ph2_simHitIdx(); }
const vector<unsigned int> &ph2_simType() { return sdl.ph2_simType(); }
const vector<int> &ph2_anchorLayer() { return sdl.ph2_anchorLayer(); }
const vector<float> &simhit_x() { return sdl.simhit_x(); }
const vector<float> &simhit_y() { return sdl.simhit_y(); }
const vector<float> &simhit_z() { return sdl.simhit_z(); }
const vector<unsigned int> &simhit_detId() { return sdl.simhit_detId(); }
const vector<unsigned int> &simhit_partnerDetId() { return sdl.simhit_partnerDetId(); }
const vector<unsigned int> &simhit_subdet() { return sdl.simhit_subdet(); }
const vector<int> &simhit_particle() { return sdl.simhit_particle(); }
const vector<vector<int> > &simhit_hitIdx() { return sdl.simhit_hitIdx(); }
const vector<int> &simhit_simTrkIdx() { return sdl.simhit_simTrkIdx(); }
const vector<float> &sim_pt() { return sdl.sim_pt(); }
const vector<float> &sim_eta() { return sdl.sim_eta(); }
const vector<float> &sim_phi() { return sdl.sim_phi(); }
const vector<float> &sim_pca_dxy() { return sdl.sim_pca_dxy(); }
const vector<float> &sim_pca_dz() { return sdl.sim_pca_dz(); }
const vector<int> &sim_q() { return sdl.sim_q(); }
const vector<int> &sim_event() { return sdl.sim_event(); }
const vector<int> &sim_pdgId() { return sdl.sim_pdgId(); }
const vector<int> &sim_bunchCrossing() { return sdl.sim_bunchCrossing(); }
const vector<int> &sim_hasAll12HitsInBarrel() { return sdl.sim_hasAll12HitsInBarrel(); }
const vector<vector<int> > &sim_simHitIdx() { return sdl.sim_simHitIdx(); }
const vector<vector<int> > &sim_simHitLayer() { return sdl.sim_simHitLayer(); }
const vector<vector<int> > &sim_simHitBoth() { return sdl.sim_simHitBoth(); }
const vector<vector<float> > &sim_simHitDrFracWithHelix() { return sdl.sim_simHitDrFracWithHelix(); }
const vector<vector<float> > &sim_simHitDistXyWithHelix() { return sdl.sim_simHitDistXyWithHelix(); }
const vector<float> &simvtx_x() { return sdl.simvtx_x(); }
const vector<float> &simvtx_y() { return sdl.simvtx_y(); }
const vector<float> &simvtx_z() { return sdl.simvtx_z(); }
const vector<float> &see_stateTrajGlbPx() { return sdl.see_stateTrajGlbPx(); }
const vector<float> &see_stateTrajGlbPy() { return sdl.see_stateTrajGlbPy(); }
const vector<float> &see_stateTrajGlbPz() { return sdl.see_stateTrajGlbPz(); }
const vector<float> &see_stateTrajGlbX() { return sdl.see_stateTrajGlbX(); }
const vector<float> &see_stateTrajGlbY() { return sdl.see_stateTrajGlbY(); }
const vector<float> &see_stateTrajGlbZ() { return sdl.see_stateTrajGlbZ(); }
const vector<float> &see_px() { return sdl.see_px(); }
const vector<float> &see_py() { return sdl.see_py(); }
const vector<float> &see_pz() { return sdl.see_pz(); }
const vector<float> &see_ptErr() { return sdl.see_ptErr(); }
const vector<float> &see_dxy() { return sdl.see_dxy(); }
const vector<float> &see_dxyErr() { return sdl.see_dxyErr(); }
const vector<float> &see_dz() { return sdl.see_dz(); }
const vector<vector<int> > &see_hitIdx() { return sdl.see_hitIdx(); }
const vector<vector<int> > &see_hitType() { return sdl.see_hitType(); }
const vector<vector<int> > &see_simTrkIdx() { return sdl.see_simTrkIdx(); }
const vector<unsigned int> &see_algo() { return sdl.see_algo(); }
const vector<vector<int> > &md_hitIdx() { return sdl.md_hitIdx(); }
const vector<vector<int> > &md_simTrkIdx() { return sdl.md_simTrkIdx(); }
const vector<vector<int> > &md_layer() { return sdl.md_layer(); }
const vector<float> &md_pt() { return sdl.md_pt(); }
const vector<float> &md_eta() { return sdl.md_eta(); }
const vector<float> &md_phi() { return sdl.md_phi(); }
const vector<float> &md_sim_pt() { return sdl.md_sim_pt(); }
const vector<float> &md_sim_eta() { return sdl.md_sim_eta(); }
const vector<float> &md_sim_phi() { return sdl.md_sim_phi(); }
const vector<float> &md_type() { return sdl.md_type(); }
const vector<float> &md_dz() { return sdl.md_dz(); }
const vector<float> &md_dzCut() { return sdl.md_dzCut(); }
const vector<float> &md_drt() { return sdl.md_drt(); }
const vector<float> &md_drtCut() { return sdl.md_drtCut(); }
const vector<float> &md_miniCut() { return sdl.md_miniCut(); }
const vector<float> &md_dphi() { return sdl.md_dphi(); }
const vector<float> &md_dphiChange() { return sdl.md_dphiChange(); }
const vector<vector<int> > &sim_mdIdx() { return sdl.sim_mdIdx(); }
const vector<vector<int> > &sim_mdIdx_isMTVmatch() { return sdl.sim_mdIdx_isMTVmatch(); }
const vector<vector<int> > &ph2_mdIdx() { return sdl.ph2_mdIdx(); }
const vector<vector<int> > &sg_hitIdx() { return sdl.sg_hitIdx(); }
const vector<vector<int> > &sg_simTrkIdx() { return sdl.sg_simTrkIdx(); }
const vector<vector<int> > &sg_simTrkIdx_anchorMatching() { return sdl.sg_simTrkIdx_anchorMatching(); }
const vector<vector<int> > &sg_layer() { return sdl.sg_layer(); }
const vector<float> &sg_pt() { return sdl.sg_pt(); }
const vector<float> &sg_eta() { return sdl.sg_eta(); }
const vector<float> &sg_phi() { return sdl.sg_phi(); }
const vector<float> &sg_sim_pt() { return sdl.sg_sim_pt(); }
const vector<float> &sg_sim_eta() { return sdl.sg_sim_eta(); }
const vector<float> &sg_sim_phi() { return sdl.sg_sim_phi(); }
const vector<vector<int> > &sim_sgIdx() { return sdl.sim_sgIdx(); }
const vector<vector<int> > &sim_sgIdx_isMTVmatch() { return sdl.sim_sgIdx_isMTVmatch(); }
const vector<vector<int> > &psg_hitIdx() { return sdl.psg_hitIdx(); }
const vector<vector<int> > &psg_simTrkIdx() { return sdl.psg_simTrkIdx(); }
const vector<vector<int> > &psg_simTrkIdx_anchorMatching() { return sdl.psg_simTrkIdx_anchorMatching(); }
const vector<vector<int> > &psg_layer() { return sdl.psg_layer(); }
const vector<float> &psg_pt() { return sdl.psg_pt(); }
const vector<float> &psg_eta() { return sdl.psg_eta(); }
const vector<float> &psg_phi() { return sdl.psg_phi(); }
const vector<float> &psg_sim_pt() { return sdl.psg_sim_pt(); }
const vector<float> &psg_sim_eta() { return sdl.psg_sim_eta(); }
const vector<float> &psg_sim_phi() { return sdl.psg_sim_phi(); }
const vector<vector<int> > &sim_psgIdx() { return sdl.sim_psgIdx(); }
const vector<vector<int> > &sim_psgIdx_isMTVmatch() { return sdl.sim_psgIdx_isMTVmatch(); }
const vector<vector<int> > &tp_hitIdx() { return sdl.tp_hitIdx(); }
const vector<vector<int> > &tp_simTrkIdx() { return sdl.tp_simTrkIdx(); }
const vector<vector<int> > &tp_layer() { return sdl.tp_layer(); }
const vector<float> &tp_pt() { return sdl.tp_pt(); }
const vector<float> &tp_eta() { return sdl.tp_eta(); }
const vector<float> &tp_phi() { return sdl.tp_phi(); }
const vector<float> &tp_sim_pt() { return sdl.tp_sim_pt(); }
const vector<float> &tp_sim_eta() { return sdl.tp_sim_eta(); }
const vector<float> &tp_sim_phi() { return sdl.tp_sim_phi(); }
const vector<vector<int> > &sim_tpIdx() { return sdl.sim_tpIdx(); }
const vector<vector<int> > &sim_tpIdx_isMTVmatch() { return sdl.sim_tpIdx_isMTVmatch(); }
const vector<vector<int> > &qp_hitIdx() { return sdl.qp_hitIdx(); }
const vector<vector<int> > &qp_simTrkIdx() { return sdl.qp_simTrkIdx(); }
const vector<vector<int> > &qp_layer() { return sdl.qp_layer(); }
const vector<float> &qp_pt() { return sdl.qp_pt(); }
const vector<float> &qp_eta() { return sdl.qp_eta(); }
const vector<float> &qp_phi() { return sdl.qp_phi(); }
const vector<float> &qp_sim_pt() { return sdl.qp_sim_pt(); }
const vector<float> &qp_sim_eta() { return sdl.qp_sim_eta(); }
const vector<float> &qp_sim_phi() { return sdl.qp_sim_phi(); }
const vector<vector<int> > &sim_qpIdx() { return sdl.sim_qpIdx(); }
const vector<vector<int> > &sim_qpIdx_isMTVmatch() { return sdl.sim_qpIdx_isMTVmatch(); }
const vector<vector<int> > &pqp_hitIdx() { return sdl.pqp_hitIdx(); }
const vector<vector<int> > &pqp_simTrkIdx() { return sdl.pqp_simTrkIdx(); }
const vector<vector<int> > &pqp_layer() { return sdl.pqp_layer(); }
const vector<float> &pqp_pt() { return sdl.pqp_pt(); }
const vector<float> &pqp_eta() { return sdl.pqp_eta(); }
const vector<float> &pqp_phi() { return sdl.pqp_phi(); }
const vector<float> &pqp_sim_pt() { return sdl.pqp_sim_pt(); }
const vector<float> &pqp_sim_eta() { return sdl.pqp_sim_eta(); }
const vector<float> &pqp_sim_phi() { return sdl.pqp_sim_phi(); }
const vector<vector<int> > &sim_pqpIdx() { return sdl.sim_pqpIdx(); }
const vector<vector<int> > &sim_pqpIdx_isMTVmatch() { return sdl.sim_pqpIdx_isMTVmatch(); }
const vector<vector<int> > &tc_hitIdx() { return sdl.tc_hitIdx(); }
const vector<vector<int> > &tc_simTrkIdx() { return sdl.tc_simTrkIdx(); }
const vector<vector<int> > &tc_layer() { return sdl.tc_layer(); }
const vector<float> &tc_pt() { return sdl.tc_pt(); }
const vector<float> &tc_eta() { return sdl.tc_eta(); }
const vector<float> &tc_phi() { return sdl.tc_phi(); }
const vector<float> &tc_sim_pt() { return sdl.tc_sim_pt(); }
const vector<float> &tc_sim_eta() { return sdl.tc_sim_eta(); }
const vector<float> &tc_sim_phi() { return sdl.tc_sim_phi(); }
const vector<vector<int> > &sim_tcIdx() { return sdl.sim_tcIdx(); }
const vector<vector<int> > &sim_tcIdx_isMTVmatch() { return sdl.sim_tcIdx_isMTVmatch(); }

}

#include "rooutil.cc"
