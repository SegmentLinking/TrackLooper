// -*- C++ -*-
#ifndef SDL_H
#define SDL_H
#include "Math/LorentzVector.h"
#include "Math/Point3D.h"
#include "TMath.h"
#include "TBranch.h"
#include "TTree.h"
#include "TH1F.h"
#include "TFile.h"
#include "TBits.h"
#include <vector>
#include <unistd.h>
typedef ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<float> > LorentzVector;

// Generated with file: /mnt/data1/bsathian/tce_work/t3t3/20211202-forSlides/withT3T3/output/outputs_2f04df9DIRTY_PU200/gpu_unified.root

using namespace std;
class SDL {
private:
protected:
  unsigned int index;
  int pT5_occupancies_;
  TBranch *pT5_occupancies_branch;
  bool pT5_occupancies_isLoaded;
  vector<float> *t3_phi_;
  TBranch *t3_phi_branch;
  bool t3_phi_isLoaded;
  vector<float> *t5_score_rphisum_;
  TBranch *t5_score_rphisum_branch;
  bool t5_score_rphisum_isLoaded;
  vector<int> *pT4_isFake_;
  TBranch *pT4_isFake_branch;
  bool pT4_isFake_isLoaded;
  vector<int> *t3_isDuplicate_;
  TBranch *t3_isDuplicate_branch;
  bool t3_isDuplicate_isLoaded;
  vector<int> *sim_event_;
  TBranch *sim_event_branch;
  bool sim_event_isLoaded;
  vector<int> *sim_q_;
  TBranch *sim_q_branch;
  bool sim_q_isLoaded;
  vector<float> *sim_eta_;
  TBranch *sim_eta_branch;
  bool sim_eta_isLoaded;
  vector<int> *T3T3_layer_binary_;
  TBranch *T3T3_layer_binary_branch;
  bool T3T3_layer_binary_isLoaded;
  vector<int> *pT3_foundDuplicate_;
  TBranch *pT3_foundDuplicate_branch;
  bool pT3_foundDuplicate_isLoaded;
  vector<int> *sim_T3T3_matched_;
  TBranch *sim_T3T3_matched_branch;
  bool sim_T3T3_matched_isLoaded;
  vector<float> *sim_len_;
  TBranch *sim_len_branch;
  bool sim_len_isLoaded;
  vector<int> *pureTCE_isDuplicate_;
  TBranch *pureTCE_isDuplicate_branch;
  bool pureTCE_isDuplicate_isLoaded;
  vector<float> *pT3_score_;
  TBranch *pT3_score_branch;
  bool pT3_score_isLoaded;
  vector<float> *t5_eta_;
  TBranch *t5_eta_branch;
  bool t5_eta_isLoaded;
  vector<int> *sim_denom_;
  TBranch *sim_denom_branch;
  bool sim_denom_isLoaded;
  vector<int> *T3T3_anchorIndex_;
  TBranch *T3T3_anchorIndex_branch;
  bool T3T3_anchorIndex_isLoaded;
  vector<int> *pT5_isDuplicate_;
  TBranch *pT5_isDuplicate_branch;
  bool pT5_isDuplicate_isLoaded;
  vector<int> *sim_tce_matched_;
  TBranch *sim_tce_matched_branch;
  bool sim_tce_matched_isLoaded;
  vector<int> *pT3_isDuplicate_;
  TBranch *pT3_isDuplicate_branch;
  bool pT3_isDuplicate_isLoaded;
  vector<int> *tc_isDuplicate_;
  TBranch *tc_isDuplicate_branch;
  bool tc_isDuplicate_isLoaded;
  vector<float> *pT3_eta_2_;
  TBranch *pT3_eta_2_branch;
  bool pT3_eta_2_isLoaded;
  vector<int> *sim_pT3_matched_;
  TBranch *sim_pT3_matched_branch;
  bool sim_pT3_matched_isLoaded;
  vector<float> *pureTCE_rzChiSquared_;
  TBranch *pureTCE_rzChiSquared_branch;
  bool pureTCE_rzChiSquared_isLoaded;
  vector<float> *T3T3_regressionRadius_;
  TBranch *T3T3_regressionRadius_branch;
  bool T3T3_regressionRadius_isLoaded;
  vector<int> *t4_isDuplicate_;
  TBranch *t4_isDuplicate_branch;
  bool t4_isDuplicate_isLoaded;
  vector<float> *pureTCE_eta_;
  TBranch *pureTCE_eta_branch;
  bool pureTCE_eta_isLoaded;
  vector<float> *tce_rPhiChiSquared_;
  TBranch *tce_rPhiChiSquared_branch;
  bool tce_rPhiChiSquared_isLoaded;
  vector<int> *pureTCE_anchorType_;
  TBranch *pureTCE_anchorType_branch;
  bool pureTCE_anchorType_isLoaded;
  vector<float> *pureTCE_pt_;
  TBranch *pureTCE_pt_branch;
  bool pureTCE_pt_isLoaded;
  vector<float> *sim_pt_;
  TBranch *sim_pt_branch;
  bool sim_pt_isLoaded;
  vector<float> *t5_eta_2_;
  TBranch *t5_eta_2_branch;
  bool t5_eta_2_isLoaded;
  vector<float> *T3T3_pt_;
  TBranch *T3T3_pt_branch;
  bool T3T3_pt_isLoaded;
  vector<int> *T3T3_maxHitMatchedCounts_;
  TBranch *T3T3_maxHitMatchedCounts_branch;
  bool T3T3_maxHitMatchedCounts_isLoaded;
  vector<int> *T3T3_anchorType_;
  TBranch *T3T3_anchorType_branch;
  bool T3T3_anchorType_isLoaded;
  vector<vector<int> > *T3T3_nLayerOverlaps_;
  TBranch *T3T3_nLayerOverlaps_branch;
  bool T3T3_nLayerOverlaps_isLoaded;
  vector<float> *pLS_eta_;
  TBranch *pLS_eta_branch;
  bool pLS_eta_isLoaded;
  vector<int> *sim_pdgId_;
  TBranch *sim_pdgId_branch;
  bool sim_pdgId_isLoaded;
  vector<float> *t3_eta_;
  TBranch *t3_eta_branch;
  bool t3_eta_isLoaded;
  vector<int> *tce_layer_binary_;
  TBranch *tce_layer_binary_branch;
  bool tce_layer_binary_isLoaded;
  vector<vector<int> > *sim_T3T3_types_;
  TBranch *sim_T3T3_types_branch;
  bool sim_T3T3_types_isLoaded;
  vector<int> *sim_TC_matched_nonextended_;
  TBranch *sim_TC_matched_nonextended_branch;
  bool sim_TC_matched_nonextended_isLoaded;
  vector<int> *t4_occupancies_;
  TBranch *t4_occupancies_branch;
  bool t4_occupancies_isLoaded;
  vector<float> *tce_eta_;
  TBranch *tce_eta_branch;
  bool tce_eta_isLoaded;
  vector<float> *T3T3_rzChiSquared_;
  TBranch *T3T3_rzChiSquared_branch;
  bool T3T3_rzChiSquared_isLoaded;
  vector<int> *tce_isDuplicate_;
  TBranch *tce_isDuplicate_branch;
  bool tce_isDuplicate_isLoaded;
  vector<vector<int> > *pT5_matched_simIdx_;
  TBranch *pT5_matched_simIdx_branch;
  bool pT5_matched_simIdx_isLoaded;
  vector<vector<int> > *sim_tcIdx_;
  TBranch *sim_tcIdx_branch;
  bool sim_tcIdx_isLoaded;
  vector<float> *t5_phi_2_;
  TBranch *t5_phi_2_branch;
  bool t5_phi_2_isLoaded;
  vector<int> *pureTCE_maxHitMatchedCounts_;
  TBranch *pureTCE_maxHitMatchedCounts_branch;
  bool pureTCE_maxHitMatchedCounts_isLoaded;
  vector<vector<int> > *t5_matched_simIdx_;
  TBranch *t5_matched_simIdx_branch;
  bool t5_matched_simIdx_isLoaded;
  vector<int> *module_subdets_;
  TBranch *module_subdets_branch;
  bool module_subdets_isLoaded;
  vector<int> *tce_anchorType_;
  TBranch *tce_anchorType_branch;
  bool tce_anchorType_isLoaded;
  vector<vector<int> > *tce_nHitOverlaps_;
  TBranch *tce_nHitOverlaps_branch;
  bool tce_nHitOverlaps_isLoaded;
  vector<int> *t3_isFake_;
  TBranch *t3_isFake_branch;
  bool t3_isFake_isLoaded;
  vector<float> *tce_phi_;
  TBranch *tce_phi_branch;
  bool tce_phi_isLoaded;
  vector<int> *t5_isFake_;
  TBranch *t5_isFake_branch;
  bool t5_isFake_isLoaded;
  vector<int> *md_occupancies_;
  TBranch *md_occupancies_branch;
  bool md_occupancies_isLoaded;
  vector<vector<int> > *t5_hitIdxs_;
  TBranch *t5_hitIdxs_branch;
  bool t5_hitIdxs_isLoaded;
  vector<vector<int> > *sim_pT3_types_;
  TBranch *sim_pT3_types_branch;
  bool sim_pT3_types_isLoaded;
  vector<vector<int> > *sim_pureTCE_types_;
  TBranch *sim_pureTCE_types_branch;
  bool sim_pureTCE_types_isLoaded;
  vector<vector<int> > *T3T3_hitIdxs_;
  TBranch *T3T3_hitIdxs_branch;
  bool T3T3_hitIdxs_isLoaded;
  vector<float> *t4_phi_;
  TBranch *t4_phi_branch;
  bool t4_phi_isLoaded;
  vector<float> *t5_phi_;
  TBranch *t5_phi_branch;
  bool t5_phi_isLoaded;
  vector<vector<int> > *pT5_hitIdxs_;
  TBranch *pT5_hitIdxs_branch;
  bool pT5_hitIdxs_isLoaded;
  vector<float> *t5_pt_;
  TBranch *t5_pt_branch;
  bool t5_pt_isLoaded;
  vector<float> *pT5_phi_;
  TBranch *pT5_phi_branch;
  bool pT5_phi_isLoaded;
  vector<int> *pureTCE_isFake_;
  TBranch *pureTCE_isFake_branch;
  bool pureTCE_isFake_isLoaded;
  vector<float> *tce_pt_;
  TBranch *tce_pt_branch;
  bool tce_pt_isLoaded;
  vector<int> *tc_isFake_;
  TBranch *tc_isFake_branch;
  bool tc_isFake_isLoaded;
  vector<int> *pT3_isFake_;
  TBranch *pT3_isFake_branch;
  bool pT3_isFake_isLoaded;
  vector<vector<int> > *tce_nLayerOverlaps_;
  TBranch *tce_nLayerOverlaps_branch;
  bool tce_nLayerOverlaps_isLoaded;
  vector<int> *tc_sim_;
  TBranch *tc_sim_branch;
  bool tc_sim_isLoaded;
  vector<vector<int> > *sim_pLS_types_;
  TBranch *sim_pLS_types_branch;
  bool sim_pLS_types_isLoaded;
  vector<float> *sim_pca_dxy_;
  TBranch *sim_pca_dxy_branch;
  bool sim_pca_dxy_isLoaded;
  vector<int> *T3T3_isDuplicate_;
  TBranch *T3T3_isDuplicate_branch;
  bool T3T3_isDuplicate_isLoaded;
  vector<float> *pT4_phi_;
  TBranch *pT4_phi_branch;
  bool pT4_phi_isLoaded;
  vector<float> *sim_hits_;
  TBranch *sim_hits_branch;
  bool sim_hits_isLoaded;
  vector<float> *pLS_phi_;
  TBranch *pLS_phi_branch;
  bool pLS_phi_isLoaded;
  vector<int> *sim_pureTCE_matched_;
  TBranch *sim_pureTCE_matched_branch;
  bool sim_pureTCE_matched_isLoaded;
  vector<float> *T3T3_eta_;
  TBranch *T3T3_eta_branch;
  bool T3T3_eta_isLoaded;
  vector<int> *t3_occupancies_;
  TBranch *t3_occupancies_branch;
  bool t3_occupancies_isLoaded;
  vector<vector<int> > *T3T3_matched_simIdx_;
  TBranch *T3T3_matched_simIdx_branch;
  bool T3T3_matched_simIdx_isLoaded;
  vector<int> *t5_foundDuplicate_;
  TBranch *t5_foundDuplicate_branch;
  bool t5_foundDuplicate_isLoaded;
  vector<vector<int> > *sim_pT4_types_;
  TBranch *sim_pT4_types_branch;
  bool sim_pT4_types_isLoaded;
  vector<float> *T3T3_matched_pt_;
  TBranch *T3T3_matched_pt_branch;
  bool T3T3_matched_pt_isLoaded;
  vector<int> *t4_isFake_;
  TBranch *t4_isFake_branch;
  bool t4_isFake_isLoaded;
  vector<float> *simvtx_x_;
  TBranch *simvtx_x_branch;
  bool simvtx_x_isLoaded;
  vector<float> *simvtx_y_;
  TBranch *simvtx_y_branch;
  bool simvtx_y_isLoaded;
  vector<float> *simvtx_z_;
  TBranch *simvtx_z_branch;
  bool simvtx_z_isLoaded;
  vector<int> *sim_T4_matched_;
  TBranch *sim_T4_matched_branch;
  bool sim_T4_matched_isLoaded;
  vector<bool> *sim_isGood_;
  TBranch *sim_isGood_branch;
  bool sim_isGood_isLoaded;
  vector<float> *pT3_pt_;
  TBranch *pT3_pt_branch;
  bool pT3_pt_isLoaded;
  vector<float> *tc_pt_;
  TBranch *tc_pt_branch;
  bool tc_pt_isLoaded;
  vector<float> *pT3_phi_2_;
  TBranch *pT3_phi_2_branch;
  bool pT3_phi_2_isLoaded;
  vector<float> *pT5_pt_;
  TBranch *pT5_pt_branch;
  bool pT5_pt_isLoaded;
  vector<int> *T3T3_isFake_;
  TBranch *T3T3_isFake_branch;
  bool T3T3_isFake_isLoaded;
  vector<float> *pureTCE_rPhiChiSquared_;
  TBranch *pureTCE_rPhiChiSquared_branch;
  bool pureTCE_rPhiChiSquared_isLoaded;
  vector<int> *pT5_score_;
  TBranch *pT5_score_branch;
  bool pT5_score_isLoaded;
  vector<float> *sim_phi_;
  TBranch *sim_phi_branch;
  bool sim_phi_isLoaded;
  vector<int> *pT5_isFake_;
  TBranch *pT5_isFake_branch;
  bool pT5_isFake_isLoaded;
  vector<int> *tc_maxHitMatchedCounts_;
  TBranch *tc_maxHitMatchedCounts_branch;
  bool tc_maxHitMatchedCounts_isLoaded;
  vector<float> *T3T3_phi_;
  TBranch *T3T3_phi_branch;
  bool T3T3_phi_isLoaded;
  vector<vector<int> > *pureTCE_nLayerOverlaps_;
  TBranch *pureTCE_nLayerOverlaps_branch;
  bool pureTCE_nLayerOverlaps_isLoaded;
  vector<float> *sim_pca_dz_;
  TBranch *sim_pca_dz_branch;
  bool sim_pca_dz_isLoaded;
  vector<vector<int> > *pureTCE_hitIdxs_;
  TBranch *pureTCE_hitIdxs_branch;
  bool pureTCE_hitIdxs_isLoaded;
  vector<vector<int> > *pureTCE_nHitOverlaps_;
  TBranch *pureTCE_nHitOverlaps_branch;
  bool pureTCE_nHitOverlaps_isLoaded;
  vector<int> *sim_pLS_matched_;
  TBranch *sim_pLS_matched_branch;
  bool sim_pLS_matched_isLoaded;
  vector<vector<int> > *tc_matched_simIdx_;
  TBranch *tc_matched_simIdx_branch;
  bool tc_matched_simIdx_isLoaded;
  vector<int> *sim_T3_matched_;
  TBranch *sim_T3_matched_branch;
  bool sim_T3_matched_isLoaded;
  vector<float> *pLS_score_;
  TBranch *pLS_score_branch;
  bool pLS_score_isLoaded;
  vector<float> *pT3_phi_;
  TBranch *pT3_phi_branch;
  bool pT3_phi_isLoaded;
  vector<float> *pT5_eta_;
  TBranch *pT5_eta_branch;
  bool pT5_eta_isLoaded;
  vector<float> *T3T3_innerT3Radius_;
  TBranch *T3T3_innerT3Radius_branch;
  bool T3T3_innerT3Radius_isLoaded;
  vector<float> *tc_phi_;
  TBranch *tc_phi_branch;
  bool tc_phi_isLoaded;
  vector<float> *t4_eta_;
  TBranch *t4_eta_branch;
  bool t4_eta_isLoaded;
  vector<int> *pLS_isFake_;
  TBranch *pLS_isFake_branch;
  bool pLS_isFake_isLoaded;
  vector<vector<int> > *pureTCE_matched_simIdx_;
  TBranch *pureTCE_matched_simIdx_branch;
  bool pureTCE_matched_simIdx_isLoaded;
  vector<int> *sim_bunchCrossing_;
  TBranch *sim_bunchCrossing_branch;
  bool sim_bunchCrossing_isLoaded;
  vector<int> *tc_partOfExtension_;
  TBranch *tc_partOfExtension_branch;
  bool tc_partOfExtension_isLoaded;
  vector<float> *pT3_eta_;
  TBranch *pT3_eta_branch;
  bool pT3_eta_isLoaded;
  vector<int> *sim_parentVtxIdx_;
  TBranch *sim_parentVtxIdx_branch;
  bool sim_parentVtxIdx_isLoaded;
  vector<int> *pureTCE_layer_binary_;
  TBranch *pureTCE_layer_binary_branch;
  bool pureTCE_layer_binary_isLoaded;
  vector<int> *sim_pT4_matched_;
  TBranch *sim_pT4_matched_branch;
  bool sim_pT4_matched_isLoaded;
  vector<float> *T3T3_rPhiChiSquared_;
  TBranch *T3T3_rPhiChiSquared_branch;
  bool T3T3_rPhiChiSquared_isLoaded;
  vector<float> *tc_eta_;
  TBranch *tc_eta_branch;
  bool tc_eta_isLoaded;
  vector<float> *sim_lengap_;
  TBranch *sim_lengap_branch;
  bool sim_lengap_isLoaded;
  vector<int> *sim_T5_matched_;
  TBranch *sim_T5_matched_branch;
  bool sim_T5_matched_isLoaded;
  vector<vector<int> > *sim_T5_types_;
  TBranch *sim_T5_types_branch;
  bool sim_T5_types_isLoaded;
  vector<vector<int> > *tce_matched_simIdx_;
  TBranch *tce_matched_simIdx_branch;
  bool tce_matched_simIdx_isLoaded;
  vector<int> *t5_isDuplicate_;
  TBranch *t5_isDuplicate_branch;
  bool t5_isDuplicate_isLoaded;
  vector<vector<int> > *pT3_hitIdxs_;
  TBranch *pT3_hitIdxs_branch;
  bool pT3_hitIdxs_isLoaded;
  vector<vector<int> > *tc_hitIdxs_;
  TBranch *tc_hitIdxs_branch;
  bool tc_hitIdxs_isLoaded;
  int pT3_occupancies_;
  TBranch *pT3_occupancies_branch;
  bool pT3_occupancies_isLoaded;
  int tc_occupancies_;
  TBranch *tc_occupancies_branch;
  bool tc_occupancies_isLoaded;
  vector<int> *sim_TC_matched_;
  TBranch *sim_TC_matched_branch;
  bool sim_TC_matched_isLoaded;
  vector<int> *sim_TC_matched_mask_;
  TBranch *sim_TC_matched_mask_branch;
  bool sim_TC_matched_mask_isLoaded;
  vector<int> *pLS_isDuplicate_;
  TBranch *pLS_isDuplicate_branch;
  bool pLS_isDuplicate_isLoaded;
  vector<int> *tce_anchorIndex_;
  TBranch *tce_anchorIndex_branch;
  bool tce_anchorIndex_isLoaded;
  vector<int> *t5_occupancies_;
  TBranch *t5_occupancies_branch;
  bool t5_occupancies_isLoaded;
  vector<float> *T3T3_outerT3Radius_;
  TBranch *T3T3_outerT3Radius_branch;
  bool T3T3_outerT3Radius_isLoaded;
  vector<int> *tc_type_;
  TBranch *tc_type_branch;
  bool tc_type_isLoaded;
  vector<int> *tce_isFake_;
  TBranch *tce_isFake_branch;
  bool tce_isFake_isLoaded;
  vector<float> *pLS_pt_;
  TBranch *pLS_pt_branch;
  bool pLS_pt_isLoaded;
  vector<int> *pureTCE_anchorIndex_;
  TBranch *pureTCE_anchorIndex_branch;
  bool pureTCE_anchorIndex_isLoaded;
  vector<vector<int> > *sim_T4_types_;
  TBranch *sim_T4_types_branch;
  bool sim_T4_types_isLoaded;
  vector<int> *pT4_isDuplicate_;
  TBranch *pT4_isDuplicate_branch;
  bool pT4_isDuplicate_isLoaded;
  vector<float> *t4_pt_;
  TBranch *t4_pt_branch;
  bool t4_pt_isLoaded;
  vector<vector<int> > *T3T3_nHitOverlaps_;
  TBranch *T3T3_nHitOverlaps_branch;
  bool T3T3_nHitOverlaps_isLoaded;
  vector<vector<int> > *sim_TC_types_;
  TBranch *sim_TC_types_branch;
  bool sim_TC_types_isLoaded;
  vector<int> *sg_occupancies_;
  TBranch *sg_occupancies_branch;
  bool sg_occupancies_isLoaded;
  vector<float> *pT4_pt_;
  TBranch *pT4_pt_branch;
  bool pT4_pt_isLoaded;
  vector<float> *pureTCE_phi_;
  TBranch *pureTCE_phi_branch;
  bool pureTCE_phi_isLoaded;
  vector<float> *sim_vx_;
  TBranch *sim_vx_branch;
  bool sim_vx_isLoaded;
  vector<float> *sim_vy_;
  TBranch *sim_vy_branch;
  bool sim_vy_isLoaded;
  vector<float> *sim_vz_;
  TBranch *sim_vz_branch;
  bool sim_vz_isLoaded;
  vector<int> *tce_maxHitMatchedCounts_;
  TBranch *tce_maxHitMatchedCounts_branch;
  bool tce_maxHitMatchedCounts_isLoaded;
  vector<float> *t3_pt_;
  TBranch *t3_pt_branch;
  bool t3_pt_isLoaded;
  vector<int> *module_rings_;
  TBranch *module_rings_branch;
  bool module_rings_isLoaded;
  vector<vector<int> > *sim_T3_types_;
  TBranch *sim_T3_types_branch;
  bool sim_T3_types_isLoaded;
  vector<vector<int> > *sim_pT5_types_;
  TBranch *sim_pT5_types_branch;
  bool sim_pT5_types_isLoaded;
  vector<int> *sim_pT5_matched_;
  TBranch *sim_pT5_matched_branch;
  bool sim_pT5_matched_isLoaded;
  vector<int> *module_layers_;
  TBranch *module_layers_branch;
  bool module_layers_isLoaded;
  vector<float> *pT4_eta_;
  TBranch *pT4_eta_branch;
  bool pT4_eta_isLoaded;
  vector<vector<int> > *sim_tce_types_;
  TBranch *sim_tce_types_branch;
  bool sim_tce_types_isLoaded;
  vector<float> *tce_rzChiSquared_;
  TBranch *tce_rzChiSquared_branch;
  bool tce_rzChiSquared_isLoaded;
  vector<vector<int> > *pT3_matched_simIdx_;
  TBranch *pT3_matched_simIdx_branch;
  bool pT3_matched_simIdx_isLoaded;
public:
  void Init(TTree *tree);
  void GetEntry(unsigned int idx);
  void LoadAllBranches();
  const int &pT5_occupancies();
  const vector<float> &t3_phi();
  const vector<float> &t5_score_rphisum();
  const vector<int> &pT4_isFake();
  const vector<int> &t3_isDuplicate();
  const vector<int> &sim_event();
  const vector<int> &sim_q();
  const vector<float> &sim_eta();
  const vector<int> &T3T3_layer_binary();
  const vector<int> &pT3_foundDuplicate();
  const vector<int> &sim_T3T3_matched();
  const vector<float> &sim_len();
  const vector<int> &pureTCE_isDuplicate();
  const vector<float> &pT3_score();
  const vector<float> &t5_eta();
  const vector<int> &sim_denom();
  const vector<int> &T3T3_anchorIndex();
  const vector<int> &pT5_isDuplicate();
  const vector<int> &sim_tce_matched();
  const vector<int> &pT3_isDuplicate();
  const vector<int> &tc_isDuplicate();
  const vector<float> &pT3_eta_2();
  const vector<int> &sim_pT3_matched();
  const vector<float> &pureTCE_rzChiSquared();
  const vector<float> &T3T3_regressionRadius();
  const vector<int> &t4_isDuplicate();
  const vector<float> &pureTCE_eta();
  const vector<float> &tce_rPhiChiSquared();
  const vector<int> &pureTCE_anchorType();
  const vector<float> &pureTCE_pt();
  const vector<float> &sim_pt();
  const vector<float> &t5_eta_2();
  const vector<float> &T3T3_pt();
  const vector<int> &T3T3_maxHitMatchedCounts();
  const vector<int> &T3T3_anchorType();
  const vector<vector<int> > &T3T3_nLayerOverlaps();
  const vector<float> &pLS_eta();
  const vector<int> &sim_pdgId();
  const vector<float> &t3_eta();
  const vector<int> &tce_layer_binary();
  const vector<vector<int> > &sim_T3T3_types();
  const vector<int> &sim_TC_matched_nonextended();
  const vector<int> &t4_occupancies();
  const vector<float> &tce_eta();
  const vector<float> &T3T3_rzChiSquared();
  const vector<int> &tce_isDuplicate();
  const vector<vector<int> > &pT5_matched_simIdx();
  const vector<vector<int> > &sim_tcIdx();
  const vector<float> &t5_phi_2();
  const vector<int> &pureTCE_maxHitMatchedCounts();
  const vector<vector<int> > &t5_matched_simIdx();
  const vector<int> &module_subdets();
  const vector<int> &tce_anchorType();
  const vector<vector<int> > &tce_nHitOverlaps();
  const vector<int> &t3_isFake();
  const vector<float> &tce_phi();
  const vector<int> &t5_isFake();
  const vector<int> &md_occupancies();
  const vector<vector<int> > &t5_hitIdxs();
  const vector<vector<int> > &sim_pT3_types();
  const vector<vector<int> > &sim_pureTCE_types();
  const vector<vector<int> > &T3T3_hitIdxs();
  const vector<float> &t4_phi();
  const vector<float> &t5_phi();
  const vector<vector<int> > &pT5_hitIdxs();
  const vector<float> &t5_pt();
  const vector<float> &pT5_phi();
  const vector<int> &pureTCE_isFake();
  const vector<float> &tce_pt();
  const vector<int> &tc_isFake();
  const vector<int> &pT3_isFake();
  const vector<vector<int> > &tce_nLayerOverlaps();
  const vector<int> &tc_sim();
  const vector<vector<int> > &sim_pLS_types();
  const vector<float> &sim_pca_dxy();
  const vector<int> &T3T3_isDuplicate();
  const vector<float> &pT4_phi();
  const vector<float> &sim_hits();
  const vector<float> &pLS_phi();
  const vector<int> &sim_pureTCE_matched();
  const vector<float> &T3T3_eta();
  const vector<int> &t3_occupancies();
  const vector<vector<int> > &T3T3_matched_simIdx();
  const vector<int> &t5_foundDuplicate();
  const vector<vector<int> > &sim_pT4_types();
  const vector<float> &T3T3_matched_pt();
  const vector<int> &t4_isFake();
  const vector<float> &simvtx_x();
  const vector<float> &simvtx_y();
  const vector<float> &simvtx_z();
  const vector<int> &sim_T4_matched();
  const vector<bool> &sim_isGood();
  const vector<float> &pT3_pt();
  const vector<float> &tc_pt();
  const vector<float> &pT3_phi_2();
  const vector<float> &pT5_pt();
  const vector<int> &T3T3_isFake();
  const vector<float> &pureTCE_rPhiChiSquared();
  const vector<int> &pT5_score();
  const vector<float> &sim_phi();
  const vector<int> &pT5_isFake();
  const vector<int> &tc_maxHitMatchedCounts();
  const vector<float> &T3T3_phi();
  const vector<vector<int> > &pureTCE_nLayerOverlaps();
  const vector<float> &sim_pca_dz();
  const vector<vector<int> > &pureTCE_hitIdxs();
  const vector<vector<int> > &pureTCE_nHitOverlaps();
  const vector<int> &sim_pLS_matched();
  const vector<vector<int> > &tc_matched_simIdx();
  const vector<int> &sim_T3_matched();
  const vector<float> &pLS_score();
  const vector<float> &pT3_phi();
  const vector<float> &pT5_eta();
  const vector<float> &T3T3_innerT3Radius();
  const vector<float> &tc_phi();
  const vector<float> &t4_eta();
  const vector<int> &pLS_isFake();
  const vector<vector<int> > &pureTCE_matched_simIdx();
  const vector<int> &sim_bunchCrossing();
  const vector<int> &tc_partOfExtension();
  const vector<float> &pT3_eta();
  const vector<int> &sim_parentVtxIdx();
  const vector<int> &pureTCE_layer_binary();
  const vector<int> &sim_pT4_matched();
  const vector<float> &T3T3_rPhiChiSquared();
  const vector<float> &tc_eta();
  const vector<float> &sim_lengap();
  const vector<int> &sim_T5_matched();
  const vector<vector<int> > &sim_T5_types();
  const vector<vector<int> > &tce_matched_simIdx();
  const vector<int> &t5_isDuplicate();
  const vector<vector<int> > &pT3_hitIdxs();
  const vector<vector<int> > &tc_hitIdxs();
  const int &pT3_occupancies();
  const int &tc_occupancies();
  const vector<int> &sim_TC_matched();
  const vector<int> &sim_TC_matched_mask();
  const vector<int> &pLS_isDuplicate();
  const vector<int> &tce_anchorIndex();
  const vector<int> &t5_occupancies();
  const vector<float> &T3T3_outerT3Radius();
  const vector<int> &tc_type();
  const vector<int> &tce_isFake();
  const vector<float> &pLS_pt();
  const vector<int> &pureTCE_anchorIndex();
  const vector<vector<int> > &sim_T4_types();
  const vector<int> &pT4_isDuplicate();
  const vector<float> &t4_pt();
  const vector<vector<int> > &T3T3_nHitOverlaps();
  const vector<vector<int> > &sim_TC_types();
  const vector<int> &sg_occupancies();
  const vector<float> &pT4_pt();
  const vector<float> &pureTCE_phi();
  const vector<float> &sim_vx();
  const vector<float> &sim_vy();
  const vector<float> &sim_vz();
  const vector<int> &tce_maxHitMatchedCounts();
  const vector<float> &t3_pt();
  const vector<int> &module_rings();
  const vector<vector<int> > &sim_T3_types();
  const vector<vector<int> > &sim_pT5_types();
  const vector<int> &sim_pT5_matched();
  const vector<int> &module_layers();
  const vector<float> &pT4_eta();
  const vector<vector<int> > &sim_tce_types();
  const vector<float> &tce_rzChiSquared();
  const vector<vector<int> > &pT3_matched_simIdx();
  static void progress( int nEventsTotal, int nEventsChain );
};

#ifndef __CINT__
extern SDL sdl;
#endif

namespace tas {

  const int &pT5_occupancies();
  const vector<float> &t3_phi();
  const vector<float> &t5_score_rphisum();
  const vector<int> &pT4_isFake();
  const vector<int> &t3_isDuplicate();
  const vector<int> &sim_event();
  const vector<int> &sim_q();
  const vector<float> &sim_eta();
  const vector<int> &T3T3_layer_binary();
  const vector<int> &pT3_foundDuplicate();
  const vector<int> &sim_T3T3_matched();
  const vector<float> &sim_len();
  const vector<int> &pureTCE_isDuplicate();
  const vector<float> &pT3_score();
  const vector<float> &t5_eta();
  const vector<int> &sim_denom();
  const vector<int> &T3T3_anchorIndex();
  const vector<int> &pT5_isDuplicate();
  const vector<int> &sim_tce_matched();
  const vector<int> &pT3_isDuplicate();
  const vector<int> &tc_isDuplicate();
  const vector<float> &pT3_eta_2();
  const vector<int> &sim_pT3_matched();
  const vector<float> &pureTCE_rzChiSquared();
  const vector<float> &T3T3_regressionRadius();
  const vector<int> &t4_isDuplicate();
  const vector<float> &pureTCE_eta();
  const vector<float> &tce_rPhiChiSquared();
  const vector<int> &pureTCE_anchorType();
  const vector<float> &pureTCE_pt();
  const vector<float> &sim_pt();
  const vector<float> &t5_eta_2();
  const vector<float> &T3T3_pt();
  const vector<int> &T3T3_maxHitMatchedCounts();
  const vector<int> &T3T3_anchorType();
  const vector<vector<int> > &T3T3_nLayerOverlaps();
  const vector<float> &pLS_eta();
  const vector<int> &sim_pdgId();
  const vector<float> &t3_eta();
  const vector<int> &tce_layer_binary();
  const vector<vector<int> > &sim_T3T3_types();
  const vector<int> &sim_TC_matched_nonextended();
  const vector<int> &t4_occupancies();
  const vector<float> &tce_eta();
  const vector<float> &T3T3_rzChiSquared();
  const vector<int> &tce_isDuplicate();
  const vector<vector<int> > &pT5_matched_simIdx();
  const vector<vector<int> > &sim_tcIdx();
  const vector<float> &t5_phi_2();
  const vector<int> &pureTCE_maxHitMatchedCounts();
  const vector<vector<int> > &t5_matched_simIdx();
  const vector<int> &module_subdets();
  const vector<int> &tce_anchorType();
  const vector<vector<int> > &tce_nHitOverlaps();
  const vector<int> &t3_isFake();
  const vector<float> &tce_phi();
  const vector<int> &t5_isFake();
  const vector<int> &md_occupancies();
  const vector<vector<int> > &t5_hitIdxs();
  const vector<vector<int> > &sim_pT3_types();
  const vector<vector<int> > &sim_pureTCE_types();
  const vector<vector<int> > &T3T3_hitIdxs();
  const vector<float> &t4_phi();
  const vector<float> &t5_phi();
  const vector<vector<int> > &pT5_hitIdxs();
  const vector<float> &t5_pt();
  const vector<float> &pT5_phi();
  const vector<int> &pureTCE_isFake();
  const vector<float> &tce_pt();
  const vector<int> &tc_isFake();
  const vector<int> &pT3_isFake();
  const vector<vector<int> > &tce_nLayerOverlaps();
  const vector<int> &tc_sim();
  const vector<vector<int> > &sim_pLS_types();
  const vector<float> &sim_pca_dxy();
  const vector<int> &T3T3_isDuplicate();
  const vector<float> &pT4_phi();
  const vector<float> &sim_hits();
  const vector<float> &pLS_phi();
  const vector<int> &sim_pureTCE_matched();
  const vector<float> &T3T3_eta();
  const vector<int> &t3_occupancies();
  const vector<vector<int> > &T3T3_matched_simIdx();
  const vector<int> &t5_foundDuplicate();
  const vector<vector<int> > &sim_pT4_types();
  const vector<float> &T3T3_matched_pt();
  const vector<int> &t4_isFake();
  const vector<float> &simvtx_x();
  const vector<float> &simvtx_y();
  const vector<float> &simvtx_z();
  const vector<int> &sim_T4_matched();
  const vector<bool> &sim_isGood();
  const vector<float> &pT3_pt();
  const vector<float> &tc_pt();
  const vector<float> &pT3_phi_2();
  const vector<float> &pT5_pt();
  const vector<int> &T3T3_isFake();
  const vector<float> &pureTCE_rPhiChiSquared();
  const vector<int> &pT5_score();
  const vector<float> &sim_phi();
  const vector<int> &pT5_isFake();
  const vector<int> &tc_maxHitMatchedCounts();
  const vector<float> &T3T3_phi();
  const vector<vector<int> > &pureTCE_nLayerOverlaps();
  const vector<float> &sim_pca_dz();
  const vector<vector<int> > &pureTCE_hitIdxs();
  const vector<vector<int> > &pureTCE_nHitOverlaps();
  const vector<int> &sim_pLS_matched();
  const vector<vector<int> > &tc_matched_simIdx();
  const vector<int> &sim_T3_matched();
  const vector<float> &pLS_score();
  const vector<float> &pT3_phi();
  const vector<float> &pT5_eta();
  const vector<float> &T3T3_innerT3Radius();
  const vector<float> &tc_phi();
  const vector<float> &t4_eta();
  const vector<int> &pLS_isFake();
  const vector<vector<int> > &pureTCE_matched_simIdx();
  const vector<int> &sim_bunchCrossing();
  const vector<int> &tc_partOfExtension();
  const vector<float> &pT3_eta();
  const vector<int> &sim_parentVtxIdx();
  const vector<int> &pureTCE_layer_binary();
  const vector<int> &sim_pT4_matched();
  const vector<float> &T3T3_rPhiChiSquared();
  const vector<float> &tc_eta();
  const vector<float> &sim_lengap();
  const vector<int> &sim_T5_matched();
  const vector<vector<int> > &sim_T5_types();
  const vector<vector<int> > &tce_matched_simIdx();
  const vector<int> &t5_isDuplicate();
  const vector<vector<int> > &pT3_hitIdxs();
  const vector<vector<int> > &tc_hitIdxs();
  const int &pT3_occupancies();
  const int &tc_occupancies();
  const vector<int> &sim_TC_matched();
  const vector<int> &sim_TC_matched_mask();
  const vector<int> &pLS_isDuplicate();
  const vector<int> &tce_anchorIndex();
  const vector<int> &t5_occupancies();
  const vector<float> &T3T3_outerT3Radius();
  const vector<int> &tc_type();
  const vector<int> &tce_isFake();
  const vector<float> &pLS_pt();
  const vector<int> &pureTCE_anchorIndex();
  const vector<vector<int> > &sim_T4_types();
  const vector<int> &pT4_isDuplicate();
  const vector<float> &t4_pt();
  const vector<vector<int> > &T3T3_nHitOverlaps();
  const vector<vector<int> > &sim_TC_types();
  const vector<int> &sg_occupancies();
  const vector<float> &pT4_pt();
  const vector<float> &pureTCE_phi();
  const vector<float> &sim_vx();
  const vector<float> &sim_vy();
  const vector<float> &sim_vz();
  const vector<int> &tce_maxHitMatchedCounts();
  const vector<float> &t3_pt();
  const vector<int> &module_rings();
  const vector<vector<int> > &sim_T3_types();
  const vector<vector<int> > &sim_pT5_types();
  const vector<int> &sim_pT5_matched();
  const vector<int> &module_layers();
  const vector<float> &pT4_eta();
  const vector<vector<int> > &sim_tce_types();
  const vector<float> &tce_rzChiSquared();
  const vector<vector<int> > &pT3_matched_simIdx();
}
#endif
