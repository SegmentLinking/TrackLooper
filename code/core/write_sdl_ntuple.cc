#include "write_sdl_ntuple.h"

//________________________________________________________________________________________________________________________________
void createOutputBranches()
{
    // Setup output TTree
    ana.tx->createBranch<vector<float>>("sim_pt");
    ana.tx->createBranch<vector<float>>("sim_eta");
    ana.tx->createBranch<vector<float>>("sim_phi");
    ana.tx->createBranch<vector<float>>("sim_pca_dxy");
    ana.tx->createBranch<vector<float>>("sim_pca_dz");
    ana.tx->createBranch<vector<int>>("sim_q");
    ana.tx->createBranch<vector<int>>("sim_event");
    ana.tx->createBranch<vector<int>>("sim_pdgId");
    ana.tx->createBranch<vector<int>>("sim_bunchCrossing");
    ana.tx->createBranch<vector<int>>("sim_parentVtxIdx");
    ana.tx->createBranch<vector<int>>("sim_denom");
    ana.tx->createBranch<vector<float>>("sim_vx");
    ana.tx->createBranch<vector<float>>("sim_vy");
    ana.tx->createBranch<vector<float>>("sim_vz");
    ana.tx->createBranch<vector<bool>>("sim_isGood");

    // Sim vertex
    ana.tx->createBranch<vector<float>>("simvtx_x");
    ana.tx->createBranch<vector<float>>("simvtx_y");
    ana.tx->createBranch<vector<float>>("simvtx_z");
    ana.tx->createBranch<vector<float>>("sim_len");
    ana.tx->createBranch<vector<float>>("sim_lengap");
    ana.tx->createBranch<vector<float>>("sim_hits");

    ana.tx->createBranch<vector<vector<int>>>("sim_tcIdx");

    // Matched to track candidate
    ana.tx->createBranch<vector<int>>("sim_TC_matched");
    ana.tx->createBranch<vector<int>>("sim_TC_matched_nonextended");
    ana.tx->createBranch<vector<vector<int>>>("sim_TC_types");

    // Track candidates
    ana.tx->createBranch<vector<float>>("tc_pt");
    ana.tx->createBranch<vector<float>>("tc_eta");
    ana.tx->createBranch<vector<float>>("tc_phi");
    ana.tx->createBranch<vector<int>>("tc_type");
    ana.tx->createBranch<vector<vector<int>>>("tc_matched_simIdx");
    ana.tx->createBranch<vector<int>>("tc_sim");
    ana.tx->createBranch<vector<int>>("tc_isFake");
    ana.tx->createBranch<vector<int>>("tc_isDuplicate");
    ana.tx->createBranch<vector<int>>("tc_partOfExtension");
    ana.tx->createBranch<vector<vector<int>>>("tc_hitIdxs");

    createOccupancyBranches();
   
    if (ana.do_lower_level)
    {
        createLowerLevelOutputBranches();
    }
}

//________________________________________________________________________________________________________________________________

void createOccupancyBranches()
{
    //one entry per lower module
    ana.tx->createBranch<vector<int>>("module_layers");
    ana.tx->createBranch<vector<int>>("module_subdets");
    ana.tx->createBranch<vector<int>>("module_rings");
    ana.tx->createBranch<vector<int>>("md_occupancies");
    ana.tx->createBranch<vector<int>>("sg_occupancies");
    ana.tx->createBranch<vector<int>>("t4_occupancies");
    ana.tx->createBranch<vector<int>>("t3_occupancies");
    ana.tx->createBranch<int>("tc_occupancies");
    ana.tx->createBranch<vector<int>>("t5_occupancies");
    ana.tx->createBranch<int>("pT3_occupancies");
    ana.tx->createBranch<int>("pT5_occupancies");
}

//________________________________________________________________________________________________________________________________
void createLowerLevelOutputBranches()
{
    // Matched to quadruplet
    ana.tx->createBranch<vector<int>>("sim_T4_matched");
    ana.tx->createBranch<vector<vector<int>>>("sim_T4_types");

    // Matched to pixel quadruplet
    ana.tx->createBranch<vector<int>>("sim_pT4_matched");
    ana.tx->createBranch<vector<vector<int>>>("sim_pT4_types");


    // T4s
    ana.tx->createBranch<vector<float>>("t4_pt");
    ana.tx->createBranch<vector<float>>("t4_eta");
    ana.tx->createBranch<vector<float>>("t4_phi");
    ana.tx->createBranch<vector<int>>("t4_isFake");
    ana.tx->createBranch<vector<int>>("t4_isDuplicate");

    // pT4s
    ana.tx->createBranch<vector<float>>("pT4_pt");
    ana.tx->createBranch<vector<float>>("pT4_eta");
    ana.tx->createBranch<vector<float>>("pT4_phi");
    ana.tx->createBranch<vector<int>>("pT4_isFake");
    ana.tx->createBranch<vector<int>>("pT4_isDuplicate");

    // Matched to triplet
    ana.tx->createBranch<vector<int>>("sim_T3_matched");
    ana.tx->createBranch<vector<vector<int>>>("sim_T3_types");

    // T3
    ana.tx->createBranch<vector<float>>("t3_pt");
    ana.tx->createBranch<vector<float>>("t3_eta");
    ana.tx->createBranch<vector<float>>("t3_phi");
    ana.tx->createBranch<vector<int>>("t3_isFake");
    ana.tx->createBranch<vector<int>>("t3_isDuplicate");

    //T5 - new kid
    ana.tx->createBranch<vector<int>>("sim_T5_matched");
    ana.tx->createBranch<vector<vector<int>>>("sim_T5_types");
    ana.tx->createBranch<vector<int>>("t5_isFake");
    ana.tx->createBranch<vector<int>>("t5_isDuplicate");
    ana.tx->createBranch<vector<int>>("t5_foundDuplicate");
    ana.tx->createBranch<vector<float>>("t5_pt");
    ana.tx->createBranch<vector<float>>("t5_eta");
    ana.tx->createBranch<vector<float>>("t5_phi");
    ana.tx->createBranch<vector<float>>("t5_eta_2");
    ana.tx->createBranch<vector<float>>("t5_phi_2");

    ana.tx->createBranch<vector<float>>("t5_score_rphisum");
    ana.tx->createBranch<vector<vector<int>>>("t5_hitIdxs");
    ana.tx->createBranch<vector<vector<int>>>("t5_matched_simIdx");

    //pLS
    ana.tx->createBranch<vector<int>>("sim_pLS_matched");
    ana.tx->createBranch<vector<vector<int>>>("sim_pLS_types");
    ana.tx->createBranch<vector<int>>("pLS_isFake");
    ana.tx->createBranch<vector<int>>("pLS_isDuplicate");
    ana.tx->createBranch<vector<float>>("pLS_pt");
    ana.tx->createBranch<vector<float>>("pLS_eta");
    ana.tx->createBranch<vector<float>>("pLS_phi");
    ana.tx->createBranch<vector<float>>("pLS_score");

    //pT3
    ana.tx->createBranch<vector<int>>("sim_pT3_matched");
    ana.tx->createBranch<vector<vector<int>>>("sim_pT3_types");
    ana.tx->createBranch<vector<float>>("pT3_pt");
    ana.tx->createBranch<vector<float>>("pT3_eta");
    ana.tx->createBranch<vector<float>>("pT3_phi");
    ana.tx->createBranch<vector<int>>("pT3_isFake");
    ana.tx->createBranch<vector<int>>("pT3_isDuplicate");
    ana.tx->createBranch<vector<float>>("pT3_eta_2");
    ana.tx->createBranch<vector<float>>("pT3_phi_2");
    ana.tx->createBranch<vector<float>>("pT3_score");
    ana.tx->createBranch<vector<int>>("pT3_foundDuplicate");
    ana.tx->createBranch<vector<vector<int>>>("pT3_matched_simIdx");
    ana.tx->createBranch<vector<vector<int>>>("pT3_hitIdxs");

    //pT5
    ana.tx->createBranch<vector<vector<int>>>("pT5_matched_simIdx");
    ana.tx->createBranch<vector<vector<int>>>("pT5_hitIdxs");
    ana.tx->createBranch<vector<int>>("sim_pT5_matched");
    ana.tx->createBranch<vector<vector<int>>>("sim_pT5_types");
    ana.tx->createBranch<vector<float>>("pT5_pt");
    ana.tx->createBranch<vector<float>>("pT5_eta");
    ana.tx->createBranch<vector<float>>("pT5_phi");
    ana.tx->createBranch<vector<int>>("pT5_isFake");
    ana.tx->createBranch<vector<int>>("pT5_isDuplicate");
    ana.tx->createBranch<vector<int>>("pT5_score");

    //TCE
    ana.tx->createBranch<vector<int>>("tce_anchorIndex");
    ana.tx->createBranch<vector<int>>("sim_tce_matched");
    ana.tx->createBranch<vector<vector<int>>>("sim_tce_types");
    ana.tx->createBranch<vector<int>>("tce_isFake");
    ana.tx->createBranch<vector<int>>("tce_isDuplicate");
    ana.tx->createBranch<vector<vector<int>>>("tce_matched_simIdx");
    ana.tx->createBranch<vector<vector<int>>>("tce_nLayerOverlaps");
    ana.tx->createBranch<vector<vector<int>>>("tce_nHitOverlaps");
    ana.tx->createBranch<vector<float>>("tce_pt");
    ana.tx->createBranch<vector<float>>("tce_eta");
    ana.tx->createBranch<vector<float>>("tce_phi");
    ana.tx->createBranch<vector<int>>("tce_layer_binary");
    ana.tx->createBranch<vector<int>>("tce_anchorType");

    ana.tx->createBranch<vector<int>>("pureTCE_anchorIndex");
    ana.tx->createBranch<vector<int>>("sim_pureTCE_matched");
    ana.tx->createBranch<vector<vector<int>>>("sim_pureTCE_types");
    ana.tx->createBranch<vector<int>>("pureTCE_isFake");
    ana.tx->createBranch<vector<int>>("pureTCE_isDuplicate");
    ana.tx->createBranch<vector<vector<int>>>("pureTCE_matched_simIdx");
    ana.tx->createBranch<vector<vector<int>>>("pureTCE_nLayerOverlaps");
    ana.tx->createBranch<vector<vector<int>>>("pureTCE_nHitOverlaps");
    ana.tx->createBranch<vector<float>>("pureTCE_pt");
    ana.tx->createBranch<vector<float>>("pureTCE_eta");
    ana.tx->createBranch<vector<float>>("pureTCE_phi");
    ana.tx->createBranch<vector<int>>("pureTCE_layer_binary");
    ana.tx->createBranch<vector<int>>("pureTCE_anchorType");
    ana.tx->createBranch<vector<vector<int>>>("pureTCE_hitIdxs");
#ifdef T3T3_EXTENSIONS
    ana.tx->createBranch<vector<int>>("T3T3_anchorIndex");
    ana.tx->createBranch<vector<int>>("sim_T3T3_matched");
    ana.tx->createBranch<vector<vector<int>>>("sim_T3T3_types");
    ana.tx->createBranch<vector<int>>("T3T3_isFake");
    ana.tx->createBranch<vector<int>>("T3T3_isDuplicate");
    ana.tx->createBranch<vector<vector<int>>>("T3T3_matched_simIdx");
    ana.tx->createBranch<vector<vector<int>>>("T3T3_nLayerOverlaps");
    ana.tx->createBranch<vector<vector<int>>>("T3T3_nHitOverlaps");
    ana.tx->createBranch<vector<float>>("T3T3_pt");
    ana.tx->createBranch<vector<float>>("T3T3_eta");
    ana.tx->createBranch<vector<float>>("T3T3_phi");
    ana.tx->createBranch<vector<int>>("T3T3_layer_binary");
    ana.tx->createBranch<vector<int>>("T3T3_anchorType");
    ana.tx->createBranch<vector<float>>("T3T3_matched_pt"); 
    ana.tx->createBranch<vector<vector<int>>>("T3T3_hitIdxs");
#endif

#ifdef CUT_VALUE_DEBUG
    createQuadrupletCutValueBranches();
    createTripletCutValueBranches();
    createSegmentCutValueBranches();
    createMiniDoubletCutValueBranches();
    createPixelQuadrupletCutValueBranches();
    createPixelTripletCutValueBranches();
    createQuintupletCutValueBranches();
    createPixelQuintupletCutValueBranches();
    createTrackExtensionCutValueBranches();
#ifdef T3T3_EXTENSIONS
    createT3T3CutvalueBranches();
#endif
#endif

#ifdef PRIMITIVE_STUDY
    createPrimitiveBranches();
#endif
}

void createTrackExtensionCutValueBranches()
{
    ana.tx->createBranch<vector<float>>("tce_rPhiChiSquared");
    ana.tx->createBranch<vector<float>>("tce_rzChiSquared");
    ana.tx->createBranch<vector<float>>("pureTCE_rPhiChiSquared");
    ana.tx->createBranch<vector<float>>("pureTCE_rzChiSquared");

}
void createT3T3CutValueBranches()
{
    ana.tx->createBranch<vector<float>>("T3T3_rPhiChiSquared");
    ana.tx->createBranch<vector<float>>("T3T3_rzChiSquared");
    ana.tx->createBranch<vector<float>>("T3T3_regressionRadius");
    ana.tx->createBranch<vector<float>>("T3T3_innerT3Radius");
    ana.tx->createBranch<vector<float>>("T3T3_outerT3Radius");
}

void createQuintupletCutValueBranches()
{
    ana.tx->createBranch<vector<int>>("t5_layer_binary");
    ana.tx->createBranch<vector<vector<float>>>("t5_matched_pt");
    ana.tx->createBranch<vector<float>>("t5_innerRadius");
    ana.tx->createBranch<vector<float>>("t5_innerRadiusMin");
    ana.tx->createBranch<vector<float>>("t5_innerRadiusMax");
    ana.tx->createBranch<vector<float>>("t5_outerRadius");
    ana.tx->createBranch<vector<float>>("t5_regressionRadius");
    ana.tx->createBranch<vector<float>>("t5_outerRadiusMin");
    ana.tx->createBranch<vector<float>>("t5_outerRadiusMax");
    ana.tx->createBranch<vector<float>>("t5_bridgeRadius");
    ana.tx->createBranch<vector<float>>("t5_bridgeRadiusMin");
    ana.tx->createBranch<vector<float>>("t5_bridgeRadiusMax");
    ana.tx->createBranch<vector<float>>("t5_innerRadiusMin2S");
    ana.tx->createBranch<vector<float>>("t5_innerRadiusMax2S");
    ana.tx->createBranch<vector<float>>("t5_bridgeRadiusMin2S");
    ana.tx->createBranch<vector<float>>("t5_bridgeRadiusMax2S");
    ana.tx->createBranch<vector<float>>("t5_outerRadiusMin2S");
    ana.tx->createBranch<vector<float>>("t5_outerRadiusMax2S");
    ana.tx->createBranch<vector<int>>("t5_moduleType_binary");
    ana.tx->createBranch<vector<float>>("t5_chiSquared");
    ana.tx->createBranch<vector<float>>("t5_nonAnchorChiSquared");
}

void createPixelQuintupletCutValueBranches()
{
    ana.tx->createBranch<vector<int>>("pT5_layer_binary");
    ana.tx->createBranch<vector<int>>("pT5_moduleType_binary");
    ana.tx->createBranch<vector<float>>("pT5_matched_pt");
    ana.tx->createBranch<vector<float>>("pT5_rzChiSquared");
    ana.tx->createBranch<vector<float>>("pT5_rPhiChiSquared");
    ana.tx->createBranch<vector<float>>("pT5_rPhiChiSquaredInwards");
}
void createPixelTripletCutValueBranches()
{
    ana.tx->createBranch<vector<float>>("pT3_pixelRadius");
    ana.tx->createBranch<vector<float>>("pT3_pixelRadiusError");
    ana.tx->createBranch<vector<vector<float>>>("pT3_matched_pt");
    ana.tx->createBranch<vector<float>>("pT3_tripletRadius");
    ana.tx->createBranch<vector<float>>("pT3_rPhiChiSquared");
    ana.tx->createBranch<vector<float>>("pT3_rPhiChiSquaredInwards");
    ana.tx->createBranch<vector<float>>("pT3_rzChiSquared");
    ana.tx->createBranch<vector<int>>("pT3_layer_binary");
    ana.tx->createBranch<vector<int>>("pT3_moduleType_binary");
    ana.tx->createBranch<vector<int>>("pT3_pix_idx1");
    ana.tx->createBranch<vector<int>>("pT3_pix_idx2");
    ana.tx->createBranch<vector<int>>("pT3_pix_idx3");
    ana.tx->createBranch<vector<int>>("pT3_pix_idx4");
    ana.tx->createBranch<vector<int>>("pT3_hit_idx1");
    ana.tx->createBranch<vector<int>>("pT3_hit_idx2");
    ana.tx->createBranch<vector<int>>("pT3_hit_idx3");
    ana.tx->createBranch<vector<int>>("pT3_hit_idx4");
    ana.tx->createBranch<vector<int>>("pT3_hit_idx5");
    ana.tx->createBranch<vector<int>>("pT3_hit_idx6");

}
void createQuadrupletCutValueBranches()
{
    ana.tx->createBranch<vector<float>>("t4_zOut");
    ana.tx->createBranch<vector<float>>("t4_rtOut");
    ana.tx->createBranch<vector<float>>("t4_deltaPhiPos");
    ana.tx->createBranch<vector<float>>("t4_deltaPhi");
    ana.tx->createBranch<vector<float>>("t4_betaIn");
    ana.tx->createBranch<vector<float>>("t4_betaOut");
    ana.tx->createBranch<vector<float>>("t4_deltaBeta");

    ana.tx->createBranch<vector<float>>("t4_zLo");
    ana.tx->createBranch<vector<float>>("t4_zHi");
    ana.tx->createBranch<vector<float>>("t4_rtLo");
    ana.tx->createBranch<vector<float>>("t4_rtHi");
    ana.tx->createBranch<vector<float>>("t4_kZ");
    ana.tx->createBranch<vector<float>>("t4_zLoPointed");
    ana.tx->createBranch<vector<float>>("t4_zHiPointed");
    ana.tx->createBranch<vector<float>>("t4_sdlCut");
    ana.tx->createBranch<vector<float>>("t4_betaInCut");
    ana.tx->createBranch<vector<float>>("t4_betaOutCut");
    ana.tx->createBranch<vector<float>>("t4_deltaBetaCut");
    ana.tx->createBranch<vector<int>>("t4_layer_binary");
    ana.tx->createBranch<vector<int>>("t4_moduleType_binary");

}

void createPixelQuadrupletCutValueBranches()
{
    ana.tx->createBranch<vector<float>>("pT4_zOut");
    ana.tx->createBranch<vector<float>>("pT4_rtOut");
    ana.tx->createBranch<vector<float>>("pT4_deltaPhiPos");
    ana.tx->createBranch<vector<float>>("pT4_deltaPhi");
    ana.tx->createBranch<vector<float>>("pT4_betaIn");
    ana.tx->createBranch<vector<float>>("pT4_betaOut");
    ana.tx->createBranch<vector<float>>("pT4_deltaBeta");

    ana.tx->createBranch<vector<float>>("pT4_zLo");
    ana.tx->createBranch<vector<float>>("pT4_zHi");
    ana.tx->createBranch<vector<float>>("pT4_rtLo");
    ana.tx->createBranch<vector<float>>("pT4_rtHi");
    ana.tx->createBranch<vector<float>>("pT4_kZ");
    ana.tx->createBranch<vector<float>>("pT4_zLoPointed");
    ana.tx->createBranch<vector<float>>("pT4_zHiPointed");
    ana.tx->createBranch<vector<float>>("pT4_sdlCut");
    ana.tx->createBranch<vector<float>>("pT4_betaInCut");
    ana.tx->createBranch<vector<float>>("pT4_betaOutCut");
    ana.tx->createBranch<vector<float>>("pT4_deltaBetaCut");
    ana.tx->createBranch<vector<int>>("pT4_layer_binary");
}

void createTripletCutValueBranches()
{
    ana.tx->createBranch<vector<float>>("t3_zOut");
    ana.tx->createBranch<vector<float>>("t3_rtOut");
    ana.tx->createBranch<vector<float>>("t3_deltaPhiPos");
    ana.tx->createBranch<vector<float>>("t3_deltaPhi");
    ana.tx->createBranch<vector<float>>("t3_betaIn");
    ana.tx->createBranch<vector<float>>("t3_betaOut");
    ana.tx->createBranch<vector<float>>("t3_deltaBeta");

    ana.tx->createBranch<vector<float>>("t3_zLo");
    ana.tx->createBranch<vector<float>>("t3_zHi");
    ana.tx->createBranch<vector<float>>("t3_rtLo");
    ana.tx->createBranch<vector<float>>("t3_rtHi");
    ana.tx->createBranch<vector<float>>("t3_kZ");
    ana.tx->createBranch<vector<float>>("t3_zLoPointed");
    ana.tx->createBranch<vector<float>>("t3_zHiPointed");
    ana.tx->createBranch<vector<float>>("t3_sdlCut");
    ana.tx->createBranch<vector<float>>("t3_betaInCut");
    ana.tx->createBranch<vector<float>>("t3_betaOutCut");
    ana.tx->createBranch<vector<float>>("t3_deltaBetaCut");
    ana.tx->createBranch<vector<int>>("t3_layer_binary");
    ana.tx->createBranch<vector<int>>("t3_moduleType_binary");

    ana.tx->createBranch<vector<float>>("t3_residual");
    ana.tx->createBranch<vector<int>>("t3_layer1");
    ana.tx->createBranch<vector<int>>("t3_layer2");
    ana.tx->createBranch<vector<int>>("t3_layer3");
    ana.tx->createBranch<vector<int>>("t3_hit_idx1");
    ana.tx->createBranch<vector<int>>("t3_hit_idx2");
    ana.tx->createBranch<vector<int>>("t3_hit_idx3");
    ana.tx->createBranch<vector<int>>("t3_hit_idx4");
    ana.tx->createBranch<vector<int>>("t3_hit_idx5");
    ana.tx->createBranch<vector<int>>("t3_hit_idx6");

}

void createSegmentCutValueBranches()
{

}

void createMiniDoubletCutValueBranches()
{
    
}

void createPrimitiveBranches()
{
    createPrimitiveBranches_v2();
}

void createPrimitiveBranches_v1()
{

    ana.tx->createBranch<vector<int>>("prim_detid");
    ana.tx->createBranch<vector<int>>("prim_layer");
    ana.tx->createBranch<vector<int>>("prim_type");
    ana.tx->createBranch<vector<int>>("prim_tilt");
    ana.tx->createBranch<vector<int>>("prim_rod");
    ana.tx->createBranch<vector<int>>("prim_ring");
    ana.tx->createBranch<vector<int>>("prim_module");

    ana.tx->createBranch<vector<vector<int>>>("prim_lower_nonpvsimhit_layer");
    ana.tx->createBranch<vector<vector<int>>>("prim_lower_nonpvsimhit_sim_denom");
    ana.tx->createBranch<vector<vector<int>>>("prim_lower_nonpvsimhit_sim_idx");
    ana.tx->createBranch<vector<vector<int>>>("prim_lower_nonpvsimhit_sim_pdgid");
    ana.tx->createBranch<vector<vector<float>>>("prim_lower_nonpvsimhit_sim_pt");
    ana.tx->createBranch<vector<vector<float>>>("prim_lower_nonpvsimhit_sim_eta");
    ana.tx->createBranch<vector<vector<float>>>("prim_lower_nonpvsimhit_sim_phi");
    ana.tx->createBranch<vector<vector<float>>>("prim_lower_nonpvsimhit_sim_dxy");
    ana.tx->createBranch<vector<vector<float>>>("prim_lower_nonpvsimhit_sim_dz");
    ana.tx->createBranch<vector<vector<float>>>("prim_lower_nonpvsimhit_x");
    ana.tx->createBranch<vector<vector<float>>>("prim_lower_nonpvsimhit_y");
    ana.tx->createBranch<vector<vector<float>>>("prim_lower_nonpvsimhit_z");

    ana.tx->createBranch<vector<vector<int>>>("prim_upper_nonpvsimhit_layer");
    ana.tx->createBranch<vector<vector<int>>>("prim_upper_nonpvsimhit_sim_denom");
    ana.tx->createBranch<vector<vector<int>>>("prim_upper_nonpvsimhit_sim_idx");
    ana.tx->createBranch<vector<vector<int>>>("prim_upper_nonpvsimhit_sim_pdgid");
    ana.tx->createBranch<vector<vector<float>>>("prim_upper_nonpvsimhit_sim_pt");
    ana.tx->createBranch<vector<vector<float>>>("prim_upper_nonpvsimhit_sim_eta");
    ana.tx->createBranch<vector<vector<float>>>("prim_upper_nonpvsimhit_sim_phi");
    ana.tx->createBranch<vector<vector<float>>>("prim_upper_nonpvsimhit_sim_dxy");
    ana.tx->createBranch<vector<vector<float>>>("prim_upper_nonpvsimhit_sim_dz");
    ana.tx->createBranch<vector<vector<float>>>("prim_upper_nonpvsimhit_x");
    ana.tx->createBranch<vector<vector<float>>>("prim_upper_nonpvsimhit_y");
    ana.tx->createBranch<vector<vector<float>>>("prim_upper_nonpvsimhit_z");

    ana.tx->createBranch<vector<vector<int>>>("prim_lower_pvsimhit_layer");
    ana.tx->createBranch<vector<vector<int>>>("prim_lower_pvsimhit_sim_denom");
    ana.tx->createBranch<vector<vector<int>>>("prim_lower_pvsimhit_sim_idx");
    ana.tx->createBranch<vector<vector<int>>>("prim_lower_pvsimhit_sim_pdgid");
    ana.tx->createBranch<vector<vector<float>>>("prim_lower_pvsimhit_sim_pt");
    ana.tx->createBranch<vector<vector<float>>>("prim_lower_pvsimhit_sim_eta");
    ana.tx->createBranch<vector<vector<float>>>("prim_lower_pvsimhit_sim_phi");
    ana.tx->createBranch<vector<vector<float>>>("prim_lower_pvsimhit_sim_dxy");
    ana.tx->createBranch<vector<vector<float>>>("prim_lower_pvsimhit_sim_dz");
    ana.tx->createBranch<vector<vector<float>>>("prim_lower_pvsimhit_x");
    ana.tx->createBranch<vector<vector<float>>>("prim_lower_pvsimhit_y");
    ana.tx->createBranch<vector<vector<float>>>("prim_lower_pvsimhit_z");

    ana.tx->createBranch<vector<vector<int>>>("prim_upper_pvsimhit_layer");
    ana.tx->createBranch<vector<vector<int>>>("prim_upper_pvsimhit_sim_denom");
    ana.tx->createBranch<vector<vector<int>>>("prim_upper_pvsimhit_sim_idx");
    ana.tx->createBranch<vector<vector<int>>>("prim_upper_pvsimhit_sim_pdgid");
    ana.tx->createBranch<vector<vector<float>>>("prim_upper_pvsimhit_sim_pt");
    ana.tx->createBranch<vector<vector<float>>>("prim_upper_pvsimhit_sim_eta");
    ana.tx->createBranch<vector<vector<float>>>("prim_upper_pvsimhit_sim_phi");
    ana.tx->createBranch<vector<vector<float>>>("prim_upper_pvsimhit_sim_dxy");
    ana.tx->createBranch<vector<vector<float>>>("prim_upper_pvsimhit_sim_dz");
    ana.tx->createBranch<vector<vector<float>>>("prim_upper_pvsimhit_x");
    ana.tx->createBranch<vector<vector<float>>>("prim_upper_pvsimhit_y");
    ana.tx->createBranch<vector<vector<float>>>("prim_upper_pvsimhit_z");

    ana.tx->createBranch<vector<vector<int>>>("prim_lower_recohit_layer"); // first match simhit -> sim trk
    ana.tx->createBranch<vector<vector<int>>>("prim_lower_recohit_sim_denom"); // first match simhit -> sim trk
    ana.tx->createBranch<vector<vector<int>>>("prim_lower_recohit_sim_idx"); // first match simhit -> sim trk
    ana.tx->createBranch<vector<vector<int>>>("prim_lower_recohit_sim_pdgid");
    ana.tx->createBranch<vector<vector<float>>>("prim_lower_recohit_sim_pt");
    ana.tx->createBranch<vector<vector<float>>>("prim_lower_recohit_sim_eta");
    ana.tx->createBranch<vector<vector<float>>>("prim_lower_recohit_sim_phi");
    ana.tx->createBranch<vector<vector<float>>>("prim_lower_recohit_sim_dxy");
    ana.tx->createBranch<vector<vector<float>>>("prim_lower_recohit_sim_dz");
    ana.tx->createBranch<vector<vector<float>>>("prim_lower_recohit_x");
    ana.tx->createBranch<vector<vector<float>>>("prim_lower_recohit_y");
    ana.tx->createBranch<vector<vector<float>>>("prim_lower_recohit_z");

    ana.tx->createBranch<vector<vector<int>>>("prim_upper_recohit_layer"); // first match simhit -> sim trk
    ana.tx->createBranch<vector<vector<int>>>("prim_upper_recohit_sim_denom"); // first match simhit -> sim trk
    ana.tx->createBranch<vector<vector<int>>>("prim_upper_recohit_sim_idx"); // first match simhit -> sim trk
    ana.tx->createBranch<vector<vector<int>>>("prim_upper_recohit_sim_pdgid");
    ana.tx->createBranch<vector<vector<float>>>("prim_upper_recohit_sim_pt");
    ana.tx->createBranch<vector<vector<float>>>("prim_upper_recohit_sim_eta");
    ana.tx->createBranch<vector<vector<float>>>("prim_upper_recohit_sim_phi");
    ana.tx->createBranch<vector<vector<float>>>("prim_upper_recohit_sim_dxy");
    ana.tx->createBranch<vector<vector<float>>>("prim_upper_recohit_sim_dz");
    ana.tx->createBranch<vector<vector<float>>>("prim_upper_recohit_x");
    ana.tx->createBranch<vector<vector<float>>>("prim_upper_recohit_y");
    ana.tx->createBranch<vector<vector<float>>>("prim_upper_recohit_z");

    ana.tx->createBranch<vector<vector<int>>>("prim_lower_mdhit_layer"); // first match simhit -> sim trk
    ana.tx->createBranch<vector<vector<int>>>("prim_lower_mdhit_sim_denom"); // first match simhit -> sim trk
    ana.tx->createBranch<vector<vector<int>>>("prim_lower_mdhit_sim_idx"); // first match simhit -> sim trk
    ana.tx->createBranch<vector<vector<int>>>("prim_lower_mdhit_sim_pdgid");
    ana.tx->createBranch<vector<vector<float>>>("prim_lower_mdhit_sim_pt");
    ana.tx->createBranch<vector<vector<float>>>("prim_lower_mdhit_sim_eta");
    ana.tx->createBranch<vector<vector<float>>>("prim_lower_mdhit_sim_phi");
    ana.tx->createBranch<vector<vector<float>>>("prim_lower_mdhit_sim_dxy");
    ana.tx->createBranch<vector<vector<float>>>("prim_lower_mdhit_sim_dz");
    ana.tx->createBranch<vector<vector<float>>>("prim_lower_mdhit_x"); // paired with upper
    ana.tx->createBranch<vector<vector<float>>>("prim_lower_mdhit_y"); // paired with upper
    ana.tx->createBranch<vector<vector<float>>>("prim_lower_mdhit_z"); // paired with upper

    ana.tx->createBranch<vector<vector<int>>>("prim_upper_mdhit_layer"); // first match simhit -> sim trk
    ana.tx->createBranch<vector<vector<int>>>("prim_upper_mdhit_sim_denom"); // first match simhit -> sim trk
    ana.tx->createBranch<vector<vector<int>>>("prim_upper_mdhit_sim_idx"); // first match simhit -> sim trk
    ana.tx->createBranch<vector<vector<int>>>("prim_upper_mdhit_sim_pdgid");
    ana.tx->createBranch<vector<vector<float>>>("prim_upper_mdhit_sim_pt");
    ana.tx->createBranch<vector<vector<float>>>("prim_upper_mdhit_sim_eta");
    ana.tx->createBranch<vector<vector<float>>>("prim_upper_mdhit_sim_phi");
    ana.tx->createBranch<vector<vector<float>>>("prim_upper_mdhit_sim_dxy");
    ana.tx->createBranch<vector<vector<float>>>("prim_upper_mdhit_sim_dz");
    ana.tx->createBranch<vector<vector<float>>>("prim_upper_mdhit_x"); // paired with lower
    ana.tx->createBranch<vector<vector<float>>>("prim_upper_mdhit_y"); // paired with lower
    ana.tx->createBranch<vector<vector<float>>>("prim_upper_mdhit_z"); // paired with lower

    ana.tx->createBranch<vector<vector<int>>>("prim_lower_sghit_layer"); // first match simhit -> sim trk
    ana.tx->createBranch<vector<vector<int>>>("prim_lower_sghit_sim_denom"); // first match simhit -> sim trk
    ana.tx->createBranch<vector<vector<int>>>("prim_lower_sghit_sim_idx"); // first match simhit -> sim trk
    ana.tx->createBranch<vector<vector<int>>>("prim_lower_sghit_sim_pdgid");
    ana.tx->createBranch<vector<vector<float>>>("prim_lower_sghit_sim_pt");
    ana.tx->createBranch<vector<vector<float>>>("prim_lower_sghit_sim_eta");
    ana.tx->createBranch<vector<vector<float>>>("prim_lower_sghit_sim_phi");
    ana.tx->createBranch<vector<vector<float>>>("prim_lower_sghit_sim_dxy");
    ana.tx->createBranch<vector<vector<float>>>("prim_lower_sghit_sim_dz");
    ana.tx->createBranch<vector<vector<float>>>("prim_lower_sghit_x"); // paired with upper
    ana.tx->createBranch<vector<vector<float>>>("prim_lower_sghit_y"); // paired with upper
    ana.tx->createBranch<vector<vector<float>>>("prim_lower_sghit_z"); // paired with upper

    ana.tx->createBranch<vector<vector<int>>>("prim_upper_sghit_layer"); // first match simhit -> sim trk
    ana.tx->createBranch<vector<vector<int>>>("prim_upper_sghit_sim_denom"); // first match simhit -> sim trk
    ana.tx->createBranch<vector<vector<int>>>("prim_upper_sghit_sim_idx"); // first match simhit -> sim trk
    ana.tx->createBranch<vector<vector<int>>>("prim_upper_sghit_sim_pdgid");
    ana.tx->createBranch<vector<vector<float>>>("prim_upper_sghit_sim_pt");
    ana.tx->createBranch<vector<vector<float>>>("prim_upper_sghit_sim_eta");
    ana.tx->createBranch<vector<vector<float>>>("prim_upper_sghit_sim_phi");
    ana.tx->createBranch<vector<vector<float>>>("prim_upper_sghit_sim_dxy");
    ana.tx->createBranch<vector<vector<float>>>("prim_upper_sghit_sim_dz");
    ana.tx->createBranch<vector<vector<float>>>("prim_upper_sghit_x"); // paired with lower
    ana.tx->createBranch<vector<vector<float>>>("prim_upper_sghit_y"); // paired with lower
    ana.tx->createBranch<vector<vector<float>>>("prim_upper_sghit_z"); // paired with lower

}

void createPrimitiveBranches_v2()
{

    vector<TString> categs = {"sim", "nonsim"};

    // HIT
    ana.tx->createBranch<vector<int>>("prim_sim_hit_idx");
    ana.tx->createBranch<vector<int>>("prim_sim_hit_layer");
    ana.tx->createBranch<vector<int>>("prim_sim_hit_subdet");
    ana.tx->createBranch<vector<int>>("prim_sim_hit_side");
    ana.tx->createBranch<vector<int>>("prim_sim_hit_rod");
    ana.tx->createBranch<vector<int>>("prim_sim_hit_ring");
    ana.tx->createBranch<vector<int>>("prim_sim_hit_module");
    ana.tx->createBranch<vector<int>>("prim_sim_hit_detid");
    ana.tx->createBranch<vector<int>>("prim_sim_hit_isanchorlayer");
    ana.tx->createBranch<vector<int>>("prim_sim_hit_islowerlayer");
    ana.tx->createBranch<vector<float>>("prim_sim_hit_x");
    ana.tx->createBranch<vector<float>>("prim_sim_hit_y");
    ana.tx->createBranch<vector<float>>("prim_sim_hit_z");
    ana.tx->createBranch<vector<float>>("prim_sim_hit_sim_pt");
    ana.tx->createBranch<vector<float>>("prim_sim_hit_sim_eta");
    ana.tx->createBranch<vector<float>>("prim_sim_hit_sim_phi");
    ana.tx->createBranch<vector<float>>("prim_sim_hit_sim_vx");
    ana.tx->createBranch<vector<float>>("prim_sim_hit_sim_vy");
    ana.tx->createBranch<vector<float>>("prim_sim_hit_sim_vz");
    ana.tx->createBranch<vector<int>>("prim_sim_hit_sim_idx");
    ana.tx->createBranch<vector<int>>("prim_sim_hit_sim_q");
    ana.tx->createBranch<vector<int>>("prim_sim_hit_sim_pdgid");
    ana.tx->createBranch<vector<int>>("prim_sim_hit_sim_event");
    ana.tx->createBranch<vector<int>>("prim_sim_hit_sim_bunch");
    ana.tx->createBranch<vector<int>>("prim_sim_hit_sim_denom");

    // HIT
    ana.tx->createBranch<vector<int>>("prim_nonsim_hit_idx");
    ana.tx->createBranch<vector<int>>("prim_nonsim_hit_layer");
    ana.tx->createBranch<vector<int>>("prim_nonsim_hit_subdet");
    ana.tx->createBranch<vector<int>>("prim_nonsim_hit_side");
    ana.tx->createBranch<vector<int>>("prim_nonsim_hit_rod");
    ana.tx->createBranch<vector<int>>("prim_nonsim_hit_ring");
    ana.tx->createBranch<vector<int>>("prim_nonsim_hit_module");
    ana.tx->createBranch<vector<int>>("prim_nonsim_hit_detid");
    ana.tx->createBranch<vector<int>>("prim_nonsim_hit_isanchorlayer");
    ana.tx->createBranch<vector<int>>("prim_nonsim_hit_islowerlayer");
    ana.tx->createBranch<vector<float>>("prim_nonsim_hit_x");
    ana.tx->createBranch<vector<float>>("prim_nonsim_hit_y");
    ana.tx->createBranch<vector<float>>("prim_nonsim_hit_z");
    ana.tx->createBranch<vector<float>>("prim_nonsim_hit_sim_pt");
    ana.tx->createBranch<vector<float>>("prim_nonsim_hit_sim_eta");
    ana.tx->createBranch<vector<float>>("prim_nonsim_hit_sim_phi");
    ana.tx->createBranch<vector<float>>("prim_nonsim_hit_sim_vx");
    ana.tx->createBranch<vector<float>>("prim_nonsim_hit_sim_vy");
    ana.tx->createBranch<vector<float>>("prim_nonsim_hit_sim_vz");
    ana.tx->createBranch<vector<int>>("prim_nonsim_hit_sim_idx");
    ana.tx->createBranch<vector<int>>("prim_nonsim_hit_sim_q");
    ana.tx->createBranch<vector<int>>("prim_nonsim_hit_sim_pdgid");
    ana.tx->createBranch<vector<int>>("prim_nonsim_hit_sim_event");
    ana.tx->createBranch<vector<int>>("prim_nonsim_hit_sim_bunch");
    ana.tx->createBranch<vector<int>>("prim_nonsim_hit_sim_denom");

    // MiniDoublet
    ana.tx->createBranch<vector<int>>("prim_sim_md_anchor_idx");
    ana.tx->createBranch<vector<int>>("prim_sim_md_upper_idx");
    ana.tx->createBranch<vector<int>>("prim_sim_md_layer");
    ana.tx->createBranch<vector<int>>("prim_sim_md_subdet");
    ana.tx->createBranch<vector<int>>("prim_sim_md_side");
    ana.tx->createBranch<vector<int>>("prim_sim_md_rod");
    ana.tx->createBranch<vector<int>>("prim_sim_md_ring");
    ana.tx->createBranch<vector<int>>("prim_sim_md_module");
    ana.tx->createBranch<vector<int>>("prim_sim_md_detid");
    ana.tx->createBranch<vector<int>>("prim_sim_md_isanchorlayer");
    ana.tx->createBranch<vector<int>>("prim_sim_md_islowerlayer");
    ana.tx->createBranch<vector<int>>("prim_sim_md_nsim_match");
    ana.tx->createBranch<vector<float>>("prim_sim_md_anchor_x");
    ana.tx->createBranch<vector<float>>("prim_sim_md_anchor_y");
    ana.tx->createBranch<vector<float>>("prim_sim_md_anchor_z");
    ana.tx->createBranch<vector<float>>("prim_sim_md_upper_x");
    ana.tx->createBranch<vector<float>>("prim_sim_md_upper_y");
    ana.tx->createBranch<vector<float>>("prim_sim_md_upper_z");
    ana.tx->createBranch<vector<float>>("prim_sim_md_sim_pt");
    ana.tx->createBranch<vector<float>>("prim_sim_md_sim_eta");
    ana.tx->createBranch<vector<float>>("prim_sim_md_sim_phi");
    ana.tx->createBranch<vector<float>>("prim_sim_md_sim_vx");
    ana.tx->createBranch<vector<float>>("prim_sim_md_sim_vy");
    ana.tx->createBranch<vector<float>>("prim_sim_md_sim_vz");
    ana.tx->createBranch<vector<int>>("prim_sim_md_sim_idx");
    ana.tx->createBranch<vector<int>>("prim_sim_md_sim_q");
    ana.tx->createBranch<vector<int>>("prim_sim_md_sim_pdgid");
    ana.tx->createBranch<vector<int>>("prim_sim_md_sim_event");
    ana.tx->createBranch<vector<int>>("prim_sim_md_sim_bunch");
    ana.tx->createBranch<vector<int>>("prim_sim_md_sim_denom");

    // MiniDoublet
    ana.tx->createBranch<vector<int>>("prim_nonsim_md_anchor_idx");
    ana.tx->createBranch<vector<int>>("prim_nonsim_md_upper_idx");
    ana.tx->createBranch<vector<int>>("prim_nonsim_md_layer");
    ana.tx->createBranch<vector<int>>("prim_nonsim_md_subdet");
    ana.tx->createBranch<vector<int>>("prim_nonsim_md_side");
    ana.tx->createBranch<vector<int>>("prim_nonsim_md_rod");
    ana.tx->createBranch<vector<int>>("prim_nonsim_md_ring");
    ana.tx->createBranch<vector<int>>("prim_nonsim_md_module");
    ana.tx->createBranch<vector<int>>("prim_nonsim_md_detid");
    ana.tx->createBranch<vector<int>>("prim_nonsim_md_isanchorlayer");
    ana.tx->createBranch<vector<int>>("prim_nonsim_md_islowerlayer");
    ana.tx->createBranch<vector<int>>("prim_nonsim_md_nsim_match");
    ana.tx->createBranch<vector<float>>("prim_nonsim_md_anchor_x");
    ana.tx->createBranch<vector<float>>("prim_nonsim_md_anchor_y");
    ana.tx->createBranch<vector<float>>("prim_nonsim_md_anchor_z");
    ana.tx->createBranch<vector<float>>("prim_nonsim_md_upper_x");
    ana.tx->createBranch<vector<float>>("prim_nonsim_md_upper_y");
    ana.tx->createBranch<vector<float>>("prim_nonsim_md_upper_z");
    ana.tx->createBranch<vector<float>>("prim_nonsim_md_sim_pt");
    ana.tx->createBranch<vector<float>>("prim_nonsim_md_sim_eta");
    ana.tx->createBranch<vector<float>>("prim_nonsim_md_sim_phi");
    ana.tx->createBranch<vector<float>>("prim_nonsim_md_sim_vx");
    ana.tx->createBranch<vector<float>>("prim_nonsim_md_sim_vy");
    ana.tx->createBranch<vector<float>>("prim_nonsim_md_sim_vz");
    ana.tx->createBranch<vector<int>>("prim_nonsim_md_sim_idx");
    ana.tx->createBranch<vector<int>>("prim_nonsim_md_sim_q");
    ana.tx->createBranch<vector<int>>("prim_nonsim_md_sim_pdgid");
    ana.tx->createBranch<vector<int>>("prim_nonsim_md_sim_event");
    ana.tx->createBranch<vector<int>>("prim_nonsim_md_sim_bunch");
    ana.tx->createBranch<vector<int>>("prim_nonsim_md_sim_denom");

}


//________________________________________________________________________________________________________________________________
void fillOutputBranches(SDL::Event* event)
{
    fillSimTrackOutputBranches();
    fillTrackCandidateOutputBranches(event);
    fillOccupancyBranches(event);

    if (ana.do_lower_level)
    {
        fillLowerLevelOutputBranches(event);
    }

    ana.tx->fill();
    ana.tx->clear();

}


//________________________________________________________________________________________________________________________________
void fillOccupancyBranches(SDL::Event* event)
{
    SDL::trackCandidates& trackCandidatesInGPU = (*event->getTrackCandidates());
    SDL::triplets& tripletsInGPU = (*event->getTriplets());
    SDL::segments& segmentsInGPU = (*event->getSegments());
    SDL::miniDoublets& mdsInGPU = (*event->getMiniDoublets());
    SDL::hits& hitsInGPU = (*event->getHits());
    SDL::modules& modulesInGPU = (*event->getModules());
    SDL::quintuplets&  quintupletsInGPU = (*event->getQuintuplets());
    SDL::pixelQuintuplets& pixelQuintupletsInGPU = (*event->getPixelQuintuplets());
    SDL::pixelTriplets& pixelTripletsInGPU = (*event->getPixelTriplets());
    //get the occupancies from these dudes
    std::vector<int> moduleLayer;
    std::vector<int> moduleSubdet;
    std::vector<int> moduleRing;
    std::vector<int> trackCandidateOccupancy;
    std::vector<int> tripletOccupancy;
    std::vector<int> segmentOccupancy;
    std::vector<int> mdOccupancy;
    std::vector<int> quintupletOccupancy;

    for(unsigned int idx = 0; idx <= *(modulesInGPU.nLowerModules); idx++)
    {
        //layer = 0, subdet = 0 => pixel module
        //module, md and segment - need some gymnastics
        unsigned int lowerIdx = idx;//modulesInGPU.lowerModuleIndices[idx];
        moduleLayer.push_back(modulesInGPU.layers[lowerIdx]);
        moduleSubdet.push_back(modulesInGPU.subdets[lowerIdx]);
        moduleRing.push_back(modulesInGPU.rings[lowerIdx]);
        segmentOccupancy.push_back(segmentsInGPU.totOccupancySegments[lowerIdx]);
        mdOccupancy.push_back(mdsInGPU.totOccupancyMDs[lowerIdx]);

        if(idx < *(modulesInGPU.nLowerModules))
        {
            quintupletOccupancy.push_back(quintupletsInGPU.totOccupancyQuintuplets[idx]);
            tripletOccupancy.push_back(tripletsInGPU.totOccupancyTriplets[idx]);
        }
    }
    ana.tx->setBranch<vector<int>>("module_layers",moduleLayer);
    ana.tx->setBranch<vector<int>>("module_subdets",moduleSubdet);
    ana.tx->setBranch<vector<int>>("module_rings",moduleRing);
    ana.tx->setBranch<vector<int>>("md_occupancies",mdOccupancy);
    ana.tx->setBranch<vector<int>>("sg_occupancies",segmentOccupancy);
    ana.tx->setBranch<vector<int>>("t3_occupancies",tripletOccupancy);
    ana.tx->setBranch<int>("tc_occupancies",*(trackCandidatesInGPU.nTrackCandidates));
    ana.tx->setBranch<int>("pT3_occupancies", *(pixelTripletsInGPU.totOccupancyPixelTriplets));
    ana.tx->setBranch<vector<int>>("t5_occupancies", quintupletOccupancy);
    ana.tx->setBranch<int>("pT5_occupancies", *(pixelQuintupletsInGPU.nPixelQuintuplets));
}

//________________________________________________________________________________________________________________________________
void fillSimTrackOutputBranches()
{

    // Sim tracks
    ana.tx->setBranch<vector<float>>("sim_pt", trk.sim_pt());
    ana.tx->setBranch<vector<float>>("sim_eta", trk.sim_eta());
    ana.tx->setBranch<vector<float>>("sim_phi", trk.sim_phi());
    ana.tx->setBranch<vector<float>>("sim_pca_dxy", trk.sim_pca_dxy());
    ana.tx->setBranch<vector<float>>("sim_pca_dz", trk.sim_pca_dz());
    ana.tx->setBranch<vector<int>>("sim_q", trk.sim_q());
    ana.tx->setBranch<vector<int>>("sim_event", trk.sim_event());
    ana.tx->setBranch<vector<int>>("sim_pdgId", trk.sim_pdgId());
    ana.tx->setBranch<vector<int>>("sim_bunchCrossing", trk.sim_bunchCrossing());
    ana.tx->setBranch<vector<int>>("sim_parentVtxIdx", trk.sim_parentVtxIdx());
    std::vector<int> sim_denom;
    std::vector<float> sim_vx;
    std::vector<float> sim_vy;
    std::vector<float> sim_vz;
    std::vector<bool> sim_isGood;
    for (unsigned int isimtrk = 0; isimtrk < trk.sim_pt().size(); ++isimtrk)
    {
        sim_denom.push_back(getDenomSimTrkType(isimtrk));
        int vtxidx = trk.sim_parentVtxIdx()[isimtrk];
        sim_vx.push_back(trk.simvtx_x()[vtxidx]);
        sim_vy.push_back(trk.simvtx_y()[vtxidx]);
        sim_vz.push_back(trk.simvtx_z()[vtxidx]);
        bool isgood =0;
        if((abs(trk.sim_eta()[isimtrk]) < 2.4) && (trk.sim_q()[isimtrk] != 0) &&(trk.sim_bunchCrossing()[isimtrk] ==0) && (trk.sim_event()[isimtrk]==0) && (trk.simvtx_z()[vtxidx]<30) ){
          float simvtx_xy2 = trk.simvtx_x()[vtxidx]*trk.simvtx_x()[vtxidx] + trk.simvtx_y()[vtxidx]*trk.simvtx_y()[vtxidx];
          if(simvtx_xy2 < 6.25 ){
            isgood = 1;
          }
        }
        sim_isGood.push_back(isgood);
    }
    ana.tx->setBranch<vector<int>>("sim_denom", sim_denom);
    ana.tx->setBranch<vector<float>>("sim_vx", sim_vx);
    ana.tx->setBranch<vector<float>>("sim_vy", sim_vy);
    ana.tx->setBranch<vector<float>>("sim_vz", sim_vz);
    ana.tx->setBranch<vector<bool>>("sim_isGood", sim_isGood);

    // simvtx
    ana.tx->setBranch<vector<float>>("simvtx_x", trk.simvtx_x());
    ana.tx->setBranch<vector<float>>("simvtx_y", trk.simvtx_y());
    ana.tx->setBranch<vector<float>>("simvtx_z", trk.simvtx_z());

    //const auto simHitIdxs = &trk.sim_simHitIdx();
    const auto simHitLays = &trk.simhit_layer();
        //count++;
        //if(hit.size() ==0){continue;}
        //printf("size: %d\n",hit.size());
        //for(auto lay: hit){
        //  printf("%d\n",simHitLays->at(lay));
        //}
    std::vector<float> sim_len;
    std::vector<float> sim_lengap;
    std::vector<float> sim_hits;
    for(unsigned int isimtrk =0; isimtrk < trk.sim_pt().size(); ++isimtrk)
    {
       //printf("size: %d\n",trk.sim_simHitIdx()[isimtrk].size());
       bool lay1 = 0;
       bool lay2 = 0;
       bool lay3 = 0;
       bool lay4 = 0;
       bool lay5 = 0;
       bool lay6 = 0;
       bool blay1 = 0;
       bool blay2 = 0;
       bool blay3 = 0;
       bool blay4 = 0;
       bool blay5 = 0;
       bool blay6 = 0;
        int len =-1;
        int lengap =-1;
        int hits =0;
       for (unsigned int ith_hit = 0; ith_hit < trk.sim_simHitIdx()[isimtrk].size(); ++ith_hit)
       {
          // Retrieve the sim hit idx
          unsigned int simhitidx = trk.sim_simHitIdx()[isimtrk][ith_hit];
          // Select only the hits in the outer tracker
          if (not (trk.simhit_subdet()[simhitidx] == 4 or trk.simhit_subdet()[simhitidx] == 5))
            continue;

          //if (not (trk.simhit_particle()[simhitidx] == trk.sim_pdgId()[isimtrk]))
          //  continue;

          if (isMuonCurlingHit(isimtrk, ith_hit)){
            len = -2;
            lengap = -2;
            break;
          }
          len =0;
          lengap =0;
          hits++;
        
          //printf("%d\n",simHitLays->at(simhitidx));
          int lay = simHitLays->at(simhitidx);
          if(trk.simhit_subdet()[simhitidx] == 4){
          if(lay ==1){lay1=1;}
          if(lay ==2){lay2=1;}
          if(lay ==3){lay3=1;}
          if(lay ==4){lay4=1;}
          if(lay ==5){lay5=1;}
          if(lay ==6){lay6=1;}
          if(lay >6){printf("high layer: %d\n",lay);}
          }
          if(trk.simhit_subdet()[simhitidx] == 5){
          if(lay ==1){blay1=1;}
          if(lay ==2){blay2=1;}
          if(lay ==3){blay3=1;}
          if(lay ==4){blay4=1;}
          if(lay ==5){blay5=1;}
          if(lay ==6){blay6=1;}
          if(lay >6){printf("high layer: %d\n",lay);}
          }
        }
        if(lay1){
        len++;
        if(lay2){len++;
        if(lay3){len++;
        if(lay4){len++; 
        if(lay5){len++;
        if(lay6){len++;
        }}}}} 
        }
        if(blay1){
        len++;
        if(blay2){len++;
        if(blay3){len++;
        if(blay4){len++; 
        if(blay5){len++;
        if(blay6){len++;
        }}}}} 
        }
        sim_len.push_back(static_cast<float>(len));
        if(lay1){lengap++;} 
        if(lay2){lengap++;}
        if(lay3){lengap++;}
        if(lay4){lengap++;} 
        if(lay5){lengap++;}
        if(lay6){lengap++;}
        if(blay1){lengap++;} 
        if(blay2){lengap++;}
        if(blay3){lengap++;}
        if(blay4){lengap++;} 
        if(blay5){lengap++;}
        if(blay6){lengap++;}
        sim_lengap.push_back(static_cast<float>(lengap));
        sim_hits.push_back(static_cast<float>(hits));
    }
    ana.tx->setBranch<vector<float>>("sim_len", sim_len);
    ana.tx->setBranch<vector<float>>("sim_lengap", sim_lengap);
    ana.tx->setBranch<vector<float>>("sim_hits", sim_hits);
}

//________________________________________________________________________________________________________________________________

void fillTrackCandidateOutputBranches(SDL::Event* event)
{

    SDL::trackCandidates& trackCandidatesInGPU = (*event->getTrackCandidates());
    SDL::triplets& tripletsInGPU = (*event->getTriplets());
    SDL::segments& segmentsInGPU = (*event->getSegments());
    SDL::miniDoublets& miniDoubletsInGPU = (*event->getMiniDoublets());
    SDL::hits& hitsInGPU = (*event->getHits());
    SDL::modules& modulesInGPU = (*event->getModules());
    SDL::quintuplets& quintupletsInGPU = (*event->getQuintuplets());
    SDL::pixelQuintuplets& pixelQuintupletsInGPU = (*event->getPixelQuintuplets());
    SDL::pixelTriplets& pixelTripletsInGPU = (*event->getPixelTriplets());
    // Did it match to track candidate?
    std::vector<int> sim_TC_matched(trk.sim_pt().size());
    std::vector<int> sim_TC_matched_nonextended(trk.sim_pt().size());
    std::vector<vector<int>> sim_TC_types(trk.sim_pt().size());
    std::vector<int> tc_isFake;
    std::vector<vector<int>> tc_matched_simIdx;
    std::vector<float> tc_pt;
    std::vector<float> tc_eta;
    std::vector<float> tc_phi;
    std::vector<int> tc_type;
    std::vector<int> tc_sim;
    std::vector<int> tc_partOfExtension;
    std::vector<vector<int>> tc_hitIdxs;

    unsigned int nTrackCandidates = *trackCandidatesInGPU.nTrackCandidates;
    for (unsigned int jdx = 0; jdx < nTrackCandidates; jdx++)
    {
        short trackCandidateType = trackCandidatesInGPU.trackCandidateType[jdx];
        unsigned int innerTrackletIdx = trackCandidatesInGPU.objectIndices[2 * jdx];
        unsigned int outerTrackletIdx = trackCandidatesInGPU.objectIndices[2 * jdx + 1];

        unsigned int innerTrackletInnerSegmentIndex = -1;
        unsigned int innerTrackletOuterSegmentIndex = -1;
        unsigned int outerTrackletOuterSegmentIndex = -1;
        unsigned int outerTrackletInnerSegmentIndex = -1;
        unsigned int outermostSegmentIndex = -1;

        float betaIn_in = 0;
        float betaOut_in = 0;
        float betaIn_out = 0;
        float betaOut_out = 0;

        std::vector<int> hit_idx;
        std::vector<int> hit_types;
        int layer_binary = 0;
        /*const*/ float pt;
        /*const*/ float eta_pLS = -999;
        /*const*/ float phi_pLS = -999;
        tc_type.emplace_back(trackCandidateType);
        if (trackCandidateType == 8) //pLS
        {
            const unsigned int N_MAX_SEGMENTS_PER_MODULE = 600;
            unsigned int pixelModuleIndex = *(modulesInGPU.nLowerModules);
            unsigned int pixelSegmentIndex = pixelModuleIndex * N_MAX_SEGMENTS_PER_MODULE + innerTrackletIdx;
            pt = segmentsInGPU.ptIn[innerTrackletIdx];
            eta_pLS = segmentsInGPU.eta[innerTrackletIdx];
            phi_pLS = segmentsInGPU.phi[innerTrackletIdx];
            unsigned int innerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * pixelSegmentIndex];
            unsigned int outerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * pixelSegmentIndex + 1];
            unsigned int innerMiniDoubletLowerHitIndex = miniDoubletsInGPU.anchorHitIndices[innerMiniDoubletIndex];
            unsigned int innerMiniDoubletUpperHitIndex = miniDoubletsInGPU.outerHitIndices[innerMiniDoubletIndex];
            unsigned int outerMiniDoubletLowerHitIndex = miniDoubletsInGPU.anchorHitIndices[outerMiniDoubletIndex];
            unsigned int outerMiniDoubletUpperHitIndex = miniDoubletsInGPU.outerHitIndices[outerMiniDoubletIndex];

            /*std::vector<int>*/ hit_idx = {
                (int) hitsInGPU.idxs[innerMiniDoubletLowerHitIndex],
                (int) hitsInGPU.idxs[innerMiniDoubletUpperHitIndex],
                (int) hitsInGPU.idxs[outerMiniDoubletLowerHitIndex],
                (int) hitsInGPU.idxs[outerMiniDoubletUpperHitIndex]
            };

            /*std::vector<int>*/ hit_types = {0,0,0,0};
            std::vector<int> module_idxs(4, pixelModuleIndex);        
            int layer0 = modulesInGPU.layers[module_idxs[0]];
            int layer2 = modulesInGPU.layers[module_idxs[2]];

            int subdet0 = modulesInGPU.subdets[module_idxs[0]];
            int subdet2 = modulesInGPU.subdets[module_idxs[2]];

            int logicallayer0 = 0;
            int logicallayer2 = 0;

        ///*int*/ layer_binary = 0;
        layer_binary |= (1 << logicallayer0);
        layer_binary |= (1 << logicallayer2);
       }else{
            if (trackCandidateType == 5) //pT3
            {
                innerTrackletInnerSegmentIndex = pixelTripletsInGPU.pixelSegmentIndices[innerTrackletIdx];
                innerTrackletOuterSegmentIndex = tripletsInGPU.segmentIndices[2 * pixelTripletsInGPU.tripletIndices[innerTrackletIdx]]; //lower segment of the outer triplet
                
                outerTrackletOuterSegmentIndex = tripletsInGPU.segmentIndices[2 * pixelTripletsInGPU.tripletIndices[innerTrackletIdx] + 1]; //upper segment of the outer triplet

                betaIn_in = 0;
                betaOut_in = 0;
                betaIn_out =  __H2F(tripletsInGPU.betaIn[pixelTripletsInGPU.tripletIndices[innerTrackletIdx]]);
                betaOut_out = __H2F(tripletsInGPU.betaOut[pixelTripletsInGPU.tripletIndices[innerTrackletIdx]]);

            }
            if (trackCandidateType == 4) // T5
            {
                unsigned int innerTrackletIndex = quintupletsInGPU.tripletIndices[2 * innerTrackletIdx];
                unsigned int outerTrackletIndex = quintupletsInGPU.tripletIndices[2 * innerTrackletIdx + 1];
                innerTrackletInnerSegmentIndex = tripletsInGPU.segmentIndices[2 * innerTrackletIndex];
                innerTrackletOuterSegmentIndex = tripletsInGPU.segmentIndices[2 * innerTrackletIndex + 1];
                outerTrackletOuterSegmentIndex = tripletsInGPU.segmentIndices[2 * outerTrackletIndex + 1];
                betaIn_in   = __H2F(tripletsInGPU.betaIn[innerTrackletIndex]);
                betaOut_in  = __H2F(tripletsInGPU.betaOut[innerTrackletIndex]);
                betaIn_out  = __H2F(tripletsInGPU.betaIn[outerTrackletIndex]);
                betaOut_out = __H2F(tripletsInGPU.betaOut[outerTrackletIndex]);
            }
            if(trackCandidateType == 7) //pT5
            {
                //innerTrackletIndex = pLS
                //outerTrackletIndex = T5
                innerTrackletInnerSegmentIndex = innerTrackletIdx;
                //segment number 1 of T5
                innerTrackletOuterSegmentIndex = tripletsInGPU.segmentIndices[2 * quintupletsInGPU.tripletIndices[2 * outerTrackletIdx]];
                //segment number 2 of T5
                outerTrackletInnerSegmentIndex = tripletsInGPU.segmentIndices[2 * quintupletsInGPU.tripletIndices[2 * outerTrackletIdx] + 1];
                //segment number 3 of T5
                outerTrackletOuterSegmentIndex = tripletsInGPU.segmentIndices[2 * quintupletsInGPU.tripletIndices[2 * outerTrackletIdx + 1]];
                //segment number 4 of T5
                outermostSegmentIndex = tripletsInGPU.segmentIndices[2 * quintupletsInGPU.tripletIndices[2 * outerTrackletIdx + 1] + 1];

                //betaIn only has the beta values of the T5s. Use the TC type = 7 criterion to then get the pixel pT value to add together with these later!!!!!!
                betaIn_in   = __H2F(tripletsInGPU.betaIn[quintupletsInGPU.tripletIndices[2 * outerTrackletIdx]]);
                betaOut_in  = __H2F(tripletsInGPU.betaOut[quintupletsInGPU.tripletIndices[2 * outerTrackletIdx]]);

                betaIn_out  = __H2F(tripletsInGPU.betaIn[quintupletsInGPU.tripletIndices[2 * outerTrackletIdx + 1]]);
                betaOut_out = __H2F(tripletsInGPU.betaOut[quintupletsInGPU.tripletIndices[2 * outerTrackletIdx + 1]]);

            }
            unsigned int innerTrackletInnerSegmentInnerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * innerTrackletInnerSegmentIndex];
            unsigned int innerTrackletInnerSegmentOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * innerTrackletInnerSegmentIndex + 1];
            unsigned int innerTrackletOuterSegmentInnerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * innerTrackletOuterSegmentIndex];
            unsigned int innerTrackletOuterSegmentOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * innerTrackletOuterSegmentIndex + 1];
            unsigned int outerTrackletOuterSegmentInnerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * outerTrackletOuterSegmentIndex];
            unsigned int outerTrackletOuterSegmentOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * outerTrackletOuterSegmentIndex + 1];

            unsigned int outermostSegmentInnerMiniDoubletIndex;            
            unsigned int outermostSegmentOuterMiniDoubletIndex;
            if(trackCandidateType == 7)
            {
               outermostSegmentInnerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * outermostSegmentIndex];
               outermostSegmentOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * outermostSegmentIndex + 1];
            }

            unsigned int innerTrackletInnerSegmentInnerMiniDoubletLowerHitIndex = miniDoubletsInGPU.anchorHitIndices[innerTrackletInnerSegmentInnerMiniDoubletIndex];
            unsigned int innerTrackletInnerSegmentInnerMiniDoubletUpperHitIndex = miniDoubletsInGPU.outerHitIndices[innerTrackletInnerSegmentInnerMiniDoubletIndex];
            unsigned int innerTrackletInnerSegmentOuterMiniDoubletLowerHitIndex = miniDoubletsInGPU.anchorHitIndices[ innerTrackletInnerSegmentOuterMiniDoubletIndex];
            unsigned int innerTrackletInnerSegmentOuterMiniDoubletUpperHitIndex = miniDoubletsInGPU.outerHitIndices[ innerTrackletInnerSegmentOuterMiniDoubletIndex];
            unsigned int innerTrackletOuterSegmentInnerMiniDoubletLowerHitIndex = miniDoubletsInGPU.anchorHitIndices[ innerTrackletOuterSegmentInnerMiniDoubletIndex];
            unsigned int innerTrackletOuterSegmentInnerMiniDoubletUpperHitIndex = miniDoubletsInGPU.outerHitIndices[ innerTrackletOuterSegmentInnerMiniDoubletIndex];
            unsigned int innerTrackletOuterSegmentOuterMiniDoubletLowerHitIndex = miniDoubletsInGPU.anchorHitIndices[ innerTrackletOuterSegmentOuterMiniDoubletIndex];
            unsigned int innerTrackletOuterSegmentOuterMiniDoubletUpperHitIndex = miniDoubletsInGPU.outerHitIndices[ innerTrackletOuterSegmentOuterMiniDoubletIndex];
            unsigned int outerTrackletOuterSegmentInnerMiniDoubletLowerHitIndex = miniDoubletsInGPU.anchorHitIndices[ outerTrackletOuterSegmentInnerMiniDoubletIndex];
            unsigned int outerTrackletOuterSegmentInnerMiniDoubletUpperHitIndex = miniDoubletsInGPU.outerHitIndices[ outerTrackletOuterSegmentInnerMiniDoubletIndex];
            unsigned int outerTrackletOuterSegmentOuterMiniDoubletLowerHitIndex = miniDoubletsInGPU.anchorHitIndices[ outerTrackletOuterSegmentOuterMiniDoubletIndex];
            unsigned int outerTrackletOuterSegmentOuterMiniDoubletUpperHitIndex = miniDoubletsInGPU.outerHitIndices[ outerTrackletOuterSegmentOuterMiniDoubletIndex];

            unsigned int outermostSegmentOuterMiniDoubletLowerHitIndex;
            unsigned int outermostSegmentOuterMiniDoubletUpperHitIndex;

            if(trackCandidateType == 7)
            {
                outermostSegmentOuterMiniDoubletLowerHitIndex = miniDoubletsInGPU.anchorHitIndices[ outermostSegmentOuterMiniDoubletIndex];
                outermostSegmentOuterMiniDoubletUpperHitIndex = miniDoubletsInGPU.outerHitIndices[ outermostSegmentOuterMiniDoubletIndex];
            }

            /*std::vector<int>*/ hit_idx = {
                (int) hitsInGPU.idxs[innerTrackletInnerSegmentInnerMiniDoubletLowerHitIndex],
                (int) hitsInGPU.idxs[innerTrackletInnerSegmentInnerMiniDoubletUpperHitIndex],
                (int) hitsInGPU.idxs[innerTrackletInnerSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.idxs[innerTrackletInnerSegmentOuterMiniDoubletUpperHitIndex],
                (int) hitsInGPU.idxs[innerTrackletOuterSegmentInnerMiniDoubletLowerHitIndex],
                (int) hitsInGPU.idxs[innerTrackletOuterSegmentInnerMiniDoubletUpperHitIndex],
                (int) hitsInGPU.idxs[innerTrackletOuterSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.idxs[innerTrackletOuterSegmentOuterMiniDoubletUpperHitIndex],
                (int) hitsInGPU.idxs[outerTrackletOuterSegmentInnerMiniDoubletLowerHitIndex],
                (int) hitsInGPU.idxs[outerTrackletOuterSegmentInnerMiniDoubletUpperHitIndex],
                (int) hitsInGPU.idxs[outerTrackletOuterSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.idxs[outerTrackletOuterSegmentOuterMiniDoubletUpperHitIndex],
            };

            if(trackCandidateType == 7)
            {
                hit_idx.push_back((int)hitsInGPU.idxs[outermostSegmentOuterMiniDoubletLowerHitIndex]);
                hit_idx.push_back((int)hitsInGPU.idxs[outermostSegmentOuterMiniDoubletUpperHitIndex]);
            }

            unsigned int iiia_idx = -1;
            unsigned int iooa_idx = -1;
            unsigned int oiia_idx = -1;
            unsigned int oooa_idx = -1;

            if (trackCandidateType == 5)
            {
                iiia_idx = segmentsInGPU.outerMiniDoubletAnchorHitIndices[innerTrackletInnerSegmentIndex]; // for pLS the innerSegment outerMiniDoublet
                iooa_idx = segmentsInGPU.outerMiniDoubletAnchorHitIndices[innerTrackletOuterSegmentIndex];
                oiia_idx = segmentsInGPU.innerMiniDoubletAnchorHitIndices[innerTrackletOuterSegmentIndex];
                oooa_idx = segmentsInGPU.outerMiniDoubletAnchorHitIndices[outerTrackletOuterSegmentIndex];
            }
            else
            {
                iiia_idx = segmentsInGPU.innerMiniDoubletAnchorHitIndices[innerTrackletInnerSegmentIndex];
                iooa_idx = segmentsInGPU.outerMiniDoubletAnchorHitIndices[innerTrackletOuterSegmentIndex];
                oiia_idx = segmentsInGPU.innerMiniDoubletAnchorHitIndices[innerTrackletOuterSegmentIndex];
                oooa_idx = segmentsInGPU.outerMiniDoubletAnchorHitIndices[outerTrackletOuterSegmentIndex];
            }

            const float dr_in = sqrt(pow(hitsInGPU.xs[iiia_idx] - hitsInGPU.xs[iooa_idx], 2) + pow(hitsInGPU.ys[iiia_idx] - hitsInGPU.ys[iooa_idx], 2));
            const float dr_out = sqrt(pow(hitsInGPU.xs[oiia_idx] - hitsInGPU.xs[oooa_idx], 2) + pow(hitsInGPU.ys[oiia_idx] - hitsInGPU.ys[oooa_idx], 2));

            const float kRinv1GeVf = (2.99792458e-3 * 3.8);
            const float k2Rinv1GeVf = kRinv1GeVf / 2.;

            const float ptAv_in = (trackCandidateType == 7 or trackCandidateType == 5) ? segmentsInGPU.ptIn[innerTrackletInnerSegmentIndex-(*(modulesInGPU.nLowerModules))*600] : dr_in * k2Rinv1GeVf / sin((betaIn_in + betaOut_in) / 2.);

            const float ptAv_out = dr_out * k2Rinv1GeVf / sin((betaIn_out + betaOut_out) / 2.);
            float ptAv;
            if(trackCandidateType == 7)
            {
                float ptAv_outermost = dr_in * k2Rinv1GeVf / sin((betaIn_in + betaOut_in) / 2.);
                ptAv =  (ptAv_in + ptAv_out + ptAv_outermost) / 3.;
            }
            else
            {
                ptAv = (ptAv_in + ptAv_out) / 2.;
            }

            /*std::vector<int>*/ hit_types;
            if (trackCandidateType != 4) // Then this means this track candidate is a pLS-based
            {
                hit_types.push_back(0);
                hit_types.push_back(0);
                hit_types.push_back(0);
                hit_types.push_back(0);
            }
            else
            {
                hit_types.push_back(4);
                hit_types.push_back(4);
                hit_types.push_back(4);
                hit_types.push_back(4);
            }

            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            if(trackCandidateType == 7)
            {
                hit_types.push_back(4);
                hit_types.push_back(4);
            }

            std::vector<int> module_idxs = {
                (int) hitsInGPU.moduleIndices[innerTrackletInnerSegmentInnerMiniDoubletLowerHitIndex],
                (int) hitsInGPU.moduleIndices[innerTrackletInnerSegmentInnerMiniDoubletUpperHitIndex],
                (int) hitsInGPU.moduleIndices[innerTrackletInnerSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.moduleIndices[innerTrackletInnerSegmentOuterMiniDoubletUpperHitIndex],
                (int) hitsInGPU.moduleIndices[innerTrackletOuterSegmentInnerMiniDoubletLowerHitIndex],
                (int) hitsInGPU.moduleIndices[innerTrackletOuterSegmentInnerMiniDoubletUpperHitIndex],
                (int) hitsInGPU.moduleIndices[innerTrackletOuterSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.moduleIndices[innerTrackletOuterSegmentOuterMiniDoubletUpperHitIndex],
                (int) hitsInGPU.moduleIndices[outerTrackletOuterSegmentInnerMiniDoubletLowerHitIndex],
                (int) hitsInGPU.moduleIndices[outerTrackletOuterSegmentInnerMiniDoubletUpperHitIndex],
                (int) hitsInGPU.moduleIndices[outerTrackletOuterSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.moduleIndices[outerTrackletOuterSegmentOuterMiniDoubletUpperHitIndex],
            };
            if(trackCandidateType == 7)
            {
                module_idxs.push_back((int) hitsInGPU.moduleIndices[outermostSegmentOuterMiniDoubletLowerHitIndex]);
                module_idxs.push_back((int) hitsInGPU.moduleIndices[outermostSegmentOuterMiniDoubletUpperHitIndex]);
            }

            bool isPixel0 = (trackCandidateType != 4);
            // bool isPixel1 = (idx == *(modulesInGPU.nLowerModules));
            bool isPixel2 = (trackCandidateType != 4);
            // bool isPixel3 = (idx == *(modulesInGPU.nLowerModules));
            bool isPixel4 = false;
            // bool isPixel5 = false;
            bool isPixel6 = false;
            // bool isPixel7 = false;
            bool isPixel8 = false;
            // bool isPixel9 = false;
            bool isPixel10 = false;
            // bool isPixel11 = false;
            bool isPixel12 = false;

            int layer0 = modulesInGPU.layers[module_idxs[0]];
            // int layer1 = modulesInGPU.layers[module_idxs[1]];
            int layer2 = modulesInGPU.layers[module_idxs[2]];
            // int layer3 = modulesInGPU.layers[module_idxs[3]];
            int layer4 = modulesInGPU.layers[module_idxs[4]];
            // int layer5 = modulesInGPU.layers[module_idxs[5]];
            int layer6 = modulesInGPU.layers[module_idxs[6]];
            // int layer7 = modulesInGPU.layers[module_idxs[7]];
            int layer8 = modulesInGPU.layers[module_idxs[8]];
            // int layer9 = modulesInGPU.layers[module_idxs[9]];
            int layer10 = modulesInGPU.layers[module_idxs[10]];
            // int layer11 = modulesInGPU.layers[module_idxs[11]];
            int layer12 = trackCandidateType == 7 ? modulesInGPU.layers[module_idxs[12]] : 0;


            int subdet0 = modulesInGPU.subdets[module_idxs[0]];
            // int subdet1 = modulesInGPU.subdets[module_idxs[1]];
            int subdet2 = modulesInGPU.subdets[module_idxs[2]];
            // int subdet3 = modulesInGPU.subdets[module_idxs[3]];
            int subdet4 = modulesInGPU.subdets[module_idxs[4]];
            // int subdet5 = modulesInGPU.subdets[module_idxs[5]];
            int subdet6 = modulesInGPU.subdets[module_idxs[6]];
            // int subdet7 = modulesInGPU.subdets[module_idxs[7]];
            int subdet8 = modulesInGPU.subdets[module_idxs[8]];
            // int subdet9 = modulesInGPU.subdets[module_idxs[9]];
            int subdet10 = modulesInGPU.subdets[module_idxs[10]];
            // int subdet11 = modulesInGPU.subdets[module_idxs[11]];
            int subdet12 = trackCandidateType == 7 ? modulesInGPU.subdets[module_idxs[12]] : 0;


            int logicallayer0 = isPixel0 ? 0 : layer0 + 6 * (subdet0 == 4);
            // int logicallayer1 = isPixel1 ? 0 : layer1 + 6 * (subdet1 == 4);
            int logicallayer2 = isPixel2 ? 0 : layer2 + 6 * (subdet2 == 4);
            // int logicallayer3 = isPixel3 ? 0 : layer3 + 6 * (subdet3 == 4);
            int logicallayer4 = isPixel4 ? 0 : layer4 + 6 * (subdet4 == 4);
            // int logicallayer5 = isPixel5 ? 0 : layer5 + 6 * (subdet5 == 4);
            int logicallayer6 = isPixel6 ? 0 : layer6 + 6 * (subdet6 == 4);
            // int logicallayer7 = isPixel7 ? 0 : layer7 + 6 * (subdet7 == 4);
            int logicallayer8 = isPixel8 ? 0 : layer8 + 6 * (subdet8 == 4);
            // int logicallayer9 = isPixel9 ? 0 : layer9 + 6 * (subdet9 == 4);
            int logicallayer10 = isPixel10 ? 0 : layer10 + 6 * (subdet10 == 4);
            // int logicallayer11 = isPixel11 ? 0 : layer11 + 6 * (subdet11 == 4);
            int logicallayer12 = trackCandidateType == 7 ? (isPixel12 == 0 ? 0 : layer12 + 6 * (subdet12 == 4)) : 0;

            //int layer_binary = 0;
            layer_binary |= (1 << logicallayer0);
            layer_binary |= (1 << logicallayer2);
            layer_binary |= (1 << logicallayer4);
            layer_binary |= (1 << logicallayer6);
            layer_binary |= (1 << logicallayer8);
            layer_binary |= (1 << logicallayer10);
            if(trackCandidateType == 7) layer_binary |= (1 << logicallayer12);
            /*const float*/ pt = ptAv;
      }// end !pLS
            tc_hitIdxs.push_back(hit_idx);
            // sim track matched index
            std::vector<int> matched_sim_trk_idxs = matchedSimTrkIdxs(hit_idx, hit_types);

            for (auto &isimtrk : matched_sim_trk_idxs)
            {
                sim_TC_matched[isimtrk]++;
                if(not trackCandidatesInGPU.partOfExtension[jdx])
                {
                    sim_TC_matched_nonextended[isimtrk]++;
                }

            }

            for (auto &isimtrk : matched_sim_trk_idxs)
            {
                sim_TC_types[isimtrk].push_back(layer_binary);
            }

            // Compute pt, eta, phi of TC
            //const float pt = ptAv;
            float eta = -999;
            float phi = -999;
            if (hit_types[0] == 4)
            {
                SDL::CPU::Hit hitA(trk.ph2_x()[hit_idx[0]], trk.ph2_y()[hit_idx[0]], trk.ph2_z()[hit_idx[0]]);
                SDL::CPU::Hit hitB(trk.ph2_x()[hit_idx[11]], trk.ph2_y()[hit_idx[11]], trk.ph2_z()[hit_idx[11]]);
                eta = hitB.eta();
                phi = hitA.phi();
            }
            else if (trackCandidateType == 8) // if pLS
            {
                eta = eta_pLS;
                phi = phi_pLS;
            }
            else
            {
                SDL::CPU::Hit hitA(trk.pix_x()[hit_idx[0]], trk.pix_y()[hit_idx[0]], trk.pix_z()[hit_idx[0]]);
                SDL::CPU::Hit hitB(trk.ph2_x()[hit_idx.back()], trk.ph2_y()[hit_idx.back()], trk.ph2_z()[hit_idx.back()]);
                eta = hitB.eta();
                phi = hitA.phi();
            }

            if(trackCandidateType == 7)
            {
                SDL::CPU::Hit hitB(trk.ph2_x()[hit_idx[13]], trk.ph2_y()[hit_idx[13]], trk.ph2_z()[hit_idx[13]]);
                eta = hitB.eta();
            }

            tc_isFake.push_back(matched_sim_trk_idxs.size() == 0);
            tc_partOfExtension.push_back(trackCandidatesInGPU.partOfExtension[jdx]);
            tc_pt.push_back(pt);
            tc_eta.push_back(eta);
            tc_phi.push_back(phi);
            tc_matched_simIdx.push_back(matched_sim_trk_idxs);
    }
    ana.tx->setBranch<vector<int>>("sim_TC_matched", sim_TC_matched);
    ana.tx->setBranch<vector<int>>("sim_TC_matched_nonextended", sim_TC_matched_nonextended);
    ana.tx->setBranch<vector<vector<int>>>("sim_TC_types", sim_TC_types);

    vector<int> tc_isDuplicate(tc_matched_simIdx.size());

    for (unsigned int i = 0; i < tc_matched_simIdx.size(); ++i)
    {
        bool isDuplicate = false;
        for (unsigned int isim = 0; isim < tc_matched_simIdx[i].size(); ++isim)
        {
            if (sim_TC_matched[tc_matched_simIdx[i][isim]] > 1)
            {
                isDuplicate = true;
            }
        }
        tc_isDuplicate[i] = isDuplicate;
    }

    
    ana.tx->setBranch<vector<float>>("tc_pt", tc_pt);
    ana.tx->setBranch<vector<float>>("tc_eta", tc_eta);
    ana.tx->setBranch<vector<float>>("tc_phi", tc_phi);
    ana.tx->setBranch<vector<int>>("tc_isFake", tc_isFake);
    ana.tx->setBranch<vector<int>>("tc_isDuplicate", tc_isDuplicate);
    ana.tx->setBranch<vector<int>>("tc_type", tc_type);
    ana.tx->setBranch<vector<vector<int>>>("tc_matched_simIdx", tc_matched_simIdx);
    ana.tx->setBranch<vector<int>>("tc_partOfExtension", tc_partOfExtension);
    ana.tx->setBranch<vector<vector<int>>>("tc_hitIdxs", tc_hitIdxs);
}

//________________________________________________________________________________________________________________________________
void fillLowerLevelOutputBranches(SDL::Event* event)
{
    fillTripletOutputBranches(event);
    fillPixelLineSegmentOutputBranches(event);
    fillQuintupletOutputBranches(event);
    fillPixelQuintupletOutputBranches(event);
    fillPixelTripletOutputBranches(event);
    fillTrackExtensionOutputBranches(event);
    fillPureTrackExtensionOutputBranches(event);
#ifdef T3T3_EXTENSIONS
    fillT3T3TrackExtensionOutputBranches(event);
#endif
}

#ifdef T3T3_EXTENSIONS
void fillT3T3TrackExtensionOutputBranches(SDL::Event* event)
{
    SDL::trackExtensions& trackExtensionsInGPU = (*event->getTrackExtensions());
    SDL::trackCandidates& trackCandidatesInGPU = (*event->getTrackCandidates());
    unsigned int nTrackCandidates = *trackCandidatesInGPU.nTrackCandidates;

    SDL::triplets& tripletsInGPU = (*event->getTriplets());
    SDL::hits& hitsInGPU = (*event->getHits());

    std::vector<int> sim_tce_matched(trk.sim_pt().size());
    std::vector<vector<int>> sim_tce_types(trk.sim_pt().size());
    std::vector<int> tce_isFake;
    std::vector<vector<int>> tce_matched_simIdx;
    std::vector<float> tce_pt;
    std::vector<float> tce_eta;
    std::vector<float> tce_phi;
    std::vector<int> tce_type;
    std::vector<int> tce_sim;
    std::vector<int> tce_layer_binary;
    std::vector<int> tce_moduleType_binary;

#ifdef CUT_VALUE_DEBUG
    std::vector<float> tce_rzChiSquared;
    std::vector<float> tce_rPhiChiSquared;
    std::vector<float> tce_innerRadius;
    std::vector<float> tce_outerRadius;
#endif

    std::vector<float> tce_simpt;
    std::vector<std::vector<int>> tce_nLayerOverlaps;
    std::vector<std::vector<int>> tce_nHitOverlaps;
    std::vector<int> tce_anchorIndex;
    std::vector<float> tce_regressionRadius;

    std::vector<float> t3_pt = ana.tx->getBranch<vector<float>>("t3_pt");
    std::vector<float> t3_eta = ana.tx->getBranch<vector<float>>("t3_eta");
    std::vector<float> t3_phi = ana.tx->getBranch<vector<float>>("t3_phi");
    const unsigned int N_MAX_T3T3_TRACK_EXTENSIONS = 40000;

    std::vector<int> tce_anchorType;;
    unsigned int nTrackExtensions = (trackExtensionsInGPU.nTrackExtensions)[nTrackCandidates] > N_MAX_T3T3_TRACK_EXTENSIONS ? N_MAX_T3T3_TRACK_EXTENSIONS : (trackExtensionsInGPU.nTrackExtensions)[nTrackCandidates]; 

    std::vector<std::vector<int>> hitIndices;

    for(size_t j = 0; j < nTrackExtensions; j++)
    {
        unsigned int teIdx = nTrackCandidates * N_MAX_TRACK_EXTENSIONS_PER_TC + j;
        short anchorType = trackExtensionsInGPU.constituentTCTypes[3*teIdx];
        short outerType = trackExtensionsInGPU.constituentTCTypes[3*teIdx + 1];
    
        unsigned int anchorIndex = trackExtensionsInGPU.constituentTCIndices[3*teIdx];
        unsigned int outerIndex = trackExtensionsInGPU.constituentTCIndices[3*teIdx + 1];
        unsigned int layer_binary = 0;
        tce_anchorIndex.push_back(anchorIndex);
        //get the hit indices
        unsigned int* anchorHitIndices;
        unsigned int* outerHitIndices;
        uint8_t* anchorLogicalLayers;
        uint8_t* outerLogicalLayers;
        vector<int> hit_idxs;
        vector<int> module_idxs;

        vector<int> nLayerOverlaps;
        vector<int> nHitOverlaps;
    
        nLayerOverlaps.push_back(trackExtensionsInGPU.nLayerOverlaps[2*teIdx]);
        nHitOverlaps.push_back(trackExtensionsInGPU.nHitOverlaps[2*teIdx]);

        if(trackExtensionsInGPU.isDup[teIdx]) continue;

        tce_nLayerOverlaps.push_back(nLayerOverlaps);
        tce_nHitOverlaps.push_back(nHitOverlaps);
        float regressionRadius = __H2F(trackExtensionsInGPU.regressionRadius[teIdx]);

#ifdef CUT_VALUE_DEBUG
        tce_rPhiChiSquared.push_back(__H2F(trackExtensionsInGPU.rPhiChiSquared[teIdx]));
        tce_rzChiSquared.push_back(__H2F(trackExtensionsInGPU.rzChiSquared[teIdx]));
        tce_regressionRadius.push_back(regressionRadius);
        tce_innerRadius.push_back(trackExtensionsInGPU.innerRadius[teIdx]);
        tce_outerRadius.push_back(trackExtensionsInGPU.outerRadius[teIdx]);
#endif       
        anchorHitIndices = &tripletsInGPU.hitIndices[6 * anchorIndex];
        anchorLogicalLayers = &tripletsInGPU.logicalLayers[3 * anchorIndex];
        outerHitIndices = &tripletsInGPU.hitIndices[6 * outerIndex];        
        outerLogicalLayers = &tripletsInGPU.logicalLayers[3 * outerIndex];
        // Compute pt, eta, phi of T3
        const float kRinv1GeVf = (2.99792458e-3 * 3.8);

        const float pt = kRinv1GeVf * regressionRadius;
        float eta = -999;
        float phi = -999;

        SDL::CPU::Hit hitA(hitsInGPU.xs[anchorHitIndices[0]], hitsInGPU.ys[anchorHitIndices[0]], hitsInGPU.zs[anchorHitIndices[0]]);
        SDL::CPU::Hit hitB(hitsInGPU.xs[outerHitIndices[5]], hitsInGPU.ys[outerHitIndices[5]], hitsInGPU.zs[outerHitIndices[5]]);
        eta = hitB.eta();
        phi = hitA.phi();

        tce_pt.push_back(pt);
        tce_eta.push_back(eta);
        tce_phi.push_back(phi);

        outerHitIndices = &tripletsInGPU.hitIndices[6 * outerIndex];
        outerLogicalLayers = &tripletsInGPU.logicalLayers[3 * outerIndex];

        size_t anchorLimits = 6;
        size_t outerLimits = 6;        
        for(size_t j = 0; j < anchorLimits; j++)
        {
            hit_idxs.push_back(hitsInGPU.idxs[anchorHitIndices[j]]);
            module_idxs.push_back(hitsInGPU.moduleIndices[anchorHitIndices[j]]);
        }
        for(size_t j = 0; j < (anchorLimits / 2); j++)
        {
            layer_binary |= (1 << anchorLogicalLayers[j]);
        }

        for(size_t j = 0; j < outerLimits; j++)
        {
            hit_idxs.push_back(hitsInGPU.idxs[outerHitIndices[j]]);
            module_idxs.push_back(hitsInGPU.moduleIndices[outerHitIndices[j]]);
        }
        for(size_t j = 0; j < (outerLimits/2); j++)
        {
            layer_binary |= (1 << outerLogicalLayers[j]);
        }

        hitIndices.push_back(hit_idxs);

        std::vector<int> hit_types(hit_idxs.size(), 4);
        std::vector<int> matched_sim_trk_idxs = matchedSimTrkIdxs(hit_idxs, hit_types);
        for(auto &isimtrk: matched_sim_trk_idxs)
        {
            sim_tce_matched[isimtrk]++;
            sim_tce_types[isimtrk].push_back(layer_binary);
        }

        if(matched_sim_trk_idxs.size() == 0)
        {
            tce_simpt.push_back(-999);
        }
        else
        {
            tce_simpt.push_back(trk.sim_pt()[matched_sim_trk_idxs[0]]);
        }

        tce_layer_binary.push_back(layer_binary);
        tce_isFake.push_back(matched_sim_trk_idxs.size() == 0);
        tce_matched_simIdx.push_back(matched_sim_trk_idxs); 
        tce_anchorType.push_back(anchorType);
    }
    vector<int> tce_isDuplicate(tce_matched_simIdx.size());
    for (unsigned int i = 0; i < tce_matched_simIdx.size(); ++i)
    {
        bool isDuplicate = false;
        for (unsigned int isim = 0; isim < tce_matched_simIdx[i].size(); ++isim)
        {
            if (sim_tce_matched[tce_matched_simIdx[i][isim]] > 1)
            {
                isDuplicate = true;
            }
        }
        tce_isDuplicate[i] = isDuplicate;
    }

    ana.tx->setBranch<vector<int>>("T3T3_anchorIndex", tce_anchorIndex);
    ana.tx->setBranch<vector<int>>("sim_T3T3_matched", sim_tce_matched);
    ana.tx->setBranch<vector<vector<int>>>("sim_T3T3_types", sim_tce_types);
    ana.tx->setBranch<vector<vector<int>>>("T3T3_matched_simIdx", tce_matched_simIdx);
    ana.tx->setBranch<vector<int>>("T3T3_isFake", tce_isFake);
    ana.tx->setBranch<vector<int>>("T3T3_isDuplicate", tce_isDuplicate);
    ana.tx->setBranch<vector<vector<int>>>("T3T3_nLayerOverlaps", tce_nLayerOverlaps);
    ana.tx->setBranch<vector<vector<int>>>("T3T3_nHitOverlaps", tce_nHitOverlaps);
    ana.tx->setBranch<vector<float>>("T3T3_pt", tce_pt);
    ana.tx->setBranch<vector<float>>("T3T3_eta", tce_eta);
    ana.tx->setBranch<vector<float>>("T3T3_phi", tce_phi);
    ana.tx->setBranch<vector<int>>("T3T3_layer_binary", tce_layer_binary);
    ana.tx->setBranch<vector<int>>("T3T3_anchorType", tce_anchorType);

#ifdef CUT_VALUE_DEBUG
    ana.tx->setBranch<vector<float>>("T3T3_regressionRadius", tce_regressionRadius);
    ana.tx->setBranch<vector<float>>("T3T3_rPhiChiSquared", tce_rPhiChiSquared);
    ana.tx->setBranch<vector<float>>("T3T3_rzChiSquared", tce_rzChiSquared);
    ana.tx->setBranch<vector<float>>("T3T3_innerT3Radius", tce_innerRadius);
    ana.tx->setBranch<vector<float>>("T3T3_outerT3Radius", tce_outerRadius);
#endif
    ana.tx->setBranch<vector<float>>("T3T3_matched_pt", tce_simpt);
    ana.tx->setBranch<vector<vector<int>>>("T3T3_hitIdxs", hitIndices);
}
#endif

void fillPureTrackExtensionOutputBranches(SDL::Event* event)
{
    SDL::trackExtensions& trackExtensionsInGPU = (*event->getTrackExtensions());
    SDL::trackCandidates& trackCandidatesInGPU = (*event->getTrackCandidates());
    unsigned int nTrackCandidates = *trackCandidatesInGPU.nTrackCandidates;

    SDL::triplets& tripletsInGPU = (*event->getTriplets());
    SDL::hits& hitsInGPU = (*event->getHits());

    std::vector<int> sim_tce_matched(trk.sim_pt().size());
    std::vector<vector<int>> sim_tce_types(trk.sim_pt().size());
    std::vector<int> tce_isFake;
    std::vector<vector<int>> tce_matched_simIdx;
    std::vector<float> tce_pt;
    std::vector<float> tce_eta;
    std::vector<float> tce_phi;
    std::vector<int> tce_type;
    std::vector<int> tce_sim;
    std::vector<int> tce_layer_binary;
    std::vector<int> tce_moduleType_binary;
#ifdef CUT_VALUE_DEBUG
    std::vector<float> tce_rzChiSquared;
    std::vector<float> tce_rPhiChiSquared;
#endif
    std::vector<float> tce_simpt;
    std::vector<std::vector<int>> tce_nLayerOverlaps;
    std::vector<std::vector<int>> tce_nHitOverlaps;
    std::vector<int> tce_anchorIndex;
    std::vector<std::vector<int>> tce_hitIndices;
    
    std::vector<float> tc_pt = ana.tx->getBranch<vector<float>>("tc_pt");
    std::vector<float> tc_eta = ana.tx->getBranch<vector<float>>("tc_eta");
    std::vector<float> tc_phi = ana.tx->getBranch<vector<float>>("tc_phi");

    std::vector<float> t3_pt = ana.tx->getBranch<vector<float>>("t3_pt");
    std::vector<float> t3_eta = ana.tx->getBranch<vector<float>>("t3_eta");
    std::vector<float> t3_phi = ana.tx->getBranch<vector<float>>("t3_phi");
    unsigned int N_MAX_TRACK_EXTENSIONS_PER_TC = 30;
    const unsigned int N_MAX_T3T3_TRACK_EXTENSIONS = 40000;

    std::vector<int> tce_anchorType;;
#ifdef T3T3_EXTENSIONS
    for(size_t i = 0; i <= nTrackCandidates; i++) //CHEAT - Include the T3T3 extensions!
#else
    for(size_t i = 0; i < nTrackCandidates; i++)
#endif
    {
        unsigned int nTrackExtensions;
        if(i < nTrackCandidates)
        {
            nTrackExtensions = (trackExtensionsInGPU.nTrackExtensions)[i] > N_MAX_TRACK_EXTENSIONS_PER_TC ? N_MAX_TRACK_EXTENSIONS_PER_TC : (trackExtensionsInGPU.nTrackExtensions)[i];
        }
        else
        {
            nTrackExtensions = (trackExtensionsInGPU.nTrackExtensions)[i] > N_MAX_T3T3_TRACK_EXTENSIONS ? N_MAX_T3T3_TRACK_EXTENSIONS : (trackExtensionsInGPU.nTrackExtensions)[i]; 
        }
        for(size_t j = 0; j < nTrackExtensions; j++)
        {
            unsigned int teIdx = i * N_MAX_TRACK_EXTENSIONS_PER_TC + j;
            short anchorType = trackExtensionsInGPU.constituentTCTypes[3*teIdx];
            short outerType = trackExtensionsInGPU.constituentTCTypes[3*teIdx + 1];
    
            unsigned int anchorIndex = trackExtensionsInGPU.constituentTCIndices[3*teIdx];
            unsigned int outerIndex = trackExtensionsInGPU.constituentTCIndices[3*teIdx + 1];
            unsigned int layer_binary = 0;
            tce_anchorIndex.push_back(anchorIndex);
            //get the hit indices
            unsigned int* anchorHitIndices;
            unsigned int* outerHitIndices;
            uint8_t* anchorLogicalLayers;
            uint8_t* outerLogicalLayers;
            vector<int> hit_idxs;
            vector<int> module_idxs;

            vector<int> nLayerOverlaps;
            vector<int> nHitOverlaps;
    
            nLayerOverlaps.push_back(trackExtensionsInGPU.nLayerOverlaps[2*teIdx]);
            nHitOverlaps.push_back(trackExtensionsInGPU.nHitOverlaps[2*teIdx]);

            if(trackExtensionsInGPU.isDup[teIdx]) continue;

            tce_nLayerOverlaps.push_back(nLayerOverlaps);
            tce_nHitOverlaps.push_back(nHitOverlaps);
#ifdef CUT_VALUE_DEBUG
            tce_rPhiChiSquared.push_back(__H2F(trackExtensionsInGPU.rPhiChiSquared[teIdx]));
            tce_rzChiSquared.push_back(__H2F(trackExtensionsInGPU.rzChiSquared[teIdx]));
#endif
            if(anchorType != 3)
            {
                anchorHitIndices = &trackCandidatesInGPU.hitIndices[14 * anchorIndex];
                anchorLogicalLayers = &trackCandidatesInGPU.logicalLayers[7 * anchorIndex];
                tce_pt.push_back(tc_pt.at(anchorIndex));
                tce_eta.push_back(tc_eta.at(anchorIndex));
                tce_phi.push_back(tc_phi.at(anchorIndex));
            }
        
            else
            {
                anchorHitIndices = &tripletsInGPU.hitIndices[6 * anchorIndex];
                anchorLogicalLayers = &tripletsInGPU.logicalLayers[3 * anchorIndex];
                
                float x1 = hitsInGPU.xs[anchorHitIndices[0]];
                float x2 = hitsInGPU.xs[anchorHitIndices[2]];
                float x3 = hitsInGPU.xs[anchorHitIndices[4]];
                float y1 = hitsInGPU.ys[anchorHitIndices[0]];
                float y2 = hitsInGPU.ys[anchorHitIndices[2]];
                float y3 = hitsInGPU.ys[anchorHitIndices[4]];
                float z1 = hitsInGPU.zs[anchorHitIndices[0]];
                float z2 = hitsInGPU.zs[anchorHitIndices[2]];
                float z3 = hitsInGPU.zs[anchorHitIndices[4]];

                float g, f; // not used
                float innerRadius = SDL::CPU::TrackCandidate::computeRadiusFromThreeAnchorHits(x1, y1, x2, y2, x3, y3, g, f);

                // Compute pt, eta, phi of T3
                const float kRinv1GeVf = (2.99792458e-3 * 3.8);

                const float pt = kRinv1GeVf * innerRadius;
                float eta = -999;
                float phi = -999;
                SDL::CPU::Hit hitA(x1,y1,z1);
                SDL::CPU::Hit hitB(x2,y2,z2);
                eta = hitB.eta();
                phi = hitA.phi();

                tce_pt.push_back(pt);
                tce_eta.push_back(eta);
                tce_phi.push_back(phi);
           }

            if(outerType == 3)
            {
                outerHitIndices = &tripletsInGPU.hitIndices[6 * outerIndex];
                outerLogicalLayers = &tripletsInGPU.logicalLayers[3 * outerIndex];
            }
            else
            {
                outerHitIndices = &trackCandidatesInGPU.hitIndices[14 * anchorIndex];
                outerLogicalLayers = &trackCandidatesInGPU.logicalLayers[3 * outerIndex];
            }
            size_t anchorLimits = anchorType == 7 ? 14 : (anchorType == 3 ? 6 : 10);
            size_t outerLimits = outerType == 3 ? 6 : (outerType == 7 ? 14 : 10);
        
            for(size_t j = 0; j < anchorLimits; j++)
            {

                hit_idxs.push_back(hitsInGPU.idxs[anchorHitIndices[j]]);
                module_idxs.push_back(hitsInGPU.moduleIndices[anchorHitIndices[j]]);
            }
            for(size_t j = 0; j < (anchorLimits / 2); j++)
            {
                layer_binary |= (1 << anchorLogicalLayers[j]);
            }

            for(size_t j = 0; j < outerLimits; j++)
            {
                hit_idxs.push_back(hitsInGPU.idxs[outerHitIndices[j]]);
                module_idxs.push_back(hitsInGPU.moduleIndices[outerHitIndices[j]]);
            }
            for(size_t j = 0; j < (outerLimits/2); j++)
            {
                layer_binary |= (1 << outerLogicalLayers[j]);
            }
            std::vector<int> hit_types(hit_idxs.size(), 4);
            if(anchorType == 7 or anchorType == 5)
            {
                hit_types[0] = 0;
                hit_types[1] = 0;
                hit_types[2] = 0;
                hit_types[3] = 0;
            }
            std::vector<int> matched_sim_trk_idxs = matchedSimTrkIdxs(hit_idxs, hit_types);
            for(auto &isimtrk: matched_sim_trk_idxs)
            {
                sim_tce_matched[isimtrk]++;
                sim_tce_types[isimtrk].push_back(layer_binary);
            }
            tce_layer_binary.push_back(layer_binary);
            tce_isFake.push_back(matched_sim_trk_idxs.size() == 0);
            tce_matched_simIdx.push_back(matched_sim_trk_idxs); 
            tce_anchorType.push_back(anchorType);
            tce_hitIndices.push_back(hit_idxs);
        }
    }

    vector<int> tce_isDuplicate(tce_matched_simIdx.size());
    for (unsigned int i = 0; i < tce_matched_simIdx.size(); ++i)
    {
        bool isDuplicate = false;
        for (unsigned int isim = 0; isim < tce_matched_simIdx[i].size(); ++isim)
        {
            if (sim_tce_matched[tce_matched_simIdx[i][isim]] > 1)
            {
                isDuplicate = true;
            }
        }
        tce_isDuplicate[i] = isDuplicate;
    }

    ana.tx->setBranch<vector<int>>("pureTCE_anchorIndex", tce_anchorIndex);
    ana.tx->setBranch<vector<int>>("sim_pureTCE_matched", sim_tce_matched);
    ana.tx->setBranch<vector<vector<int>>>("sim_pureTCE_types", sim_tce_types);
    ana.tx->setBranch<vector<vector<int>>>("pureTCE_matched_simIdx", tce_matched_simIdx);
    ana.tx->setBranch<vector<int>>("pureTCE_isFake", tce_isFake);
    ana.tx->setBranch<vector<int>>("pureTCE_isDuplicate", tce_isDuplicate);
    ana.tx->setBranch<vector<vector<int>>>("pureTCE_nLayerOverlaps", tce_nLayerOverlaps);
    ana.tx->setBranch<vector<vector<int>>>("pureTCE_nHitOverlaps", tce_nHitOverlaps);
    ana.tx->setBranch<vector<float>>("pureTCE_pt", tce_pt);
    ana.tx->setBranch<vector<float>>("pureTCE_eta", tce_eta);
    ana.tx->setBranch<vector<float>>("pureTCE_phi", tce_phi);
#ifdef CUT_VALUE_DEBUG
    ana.tx->setBranch<vector<float>>("pureTCE_rPhiChiSquared", tce_rPhiChiSquared);
    ana.tx->setBranch<vector<float>>("pureTCE_rzChiSquared", tce_rzChiSquared);
#endif
    ana.tx->setBranch<vector<int>>("pureTCE_layer_binary", tce_layer_binary);
    ana.tx->setBranch<vector<int>>("pureTCE_anchorType", tce_anchorType);
    ana.tx->setBranch<vector<vector<int>>>("pureTCE_hitIdxs", tce_hitIndices);
}

void fillTrackExtensionOutputBranches(SDL::Event* event)
{
    SDL::trackExtensions& trackExtensionsInGPU = (*event->getTrackExtensions());
    SDL::trackCandidates& trackCandidatesInGPU = (*event->getTrackCandidates());
    unsigned int nTrackCandidates = *trackCandidatesInGPU.nTrackCandidates;

    SDL::triplets& tripletsInGPU = (*event->getTriplets());
    SDL::hits& hitsInGPU = (*event->getHits());

    std::vector<int> sim_tce_matched(trk.sim_pt().size());
    std::vector<vector<int>> sim_tce_types(trk.sim_pt().size());
    std::vector<int> tce_isFake;
    std::vector<vector<int>> tce_matched_simIdx;
    std::vector<float> tce_pt;
    std::vector<float> tce_eta;
    std::vector<float> tce_phi;
    std::vector<int> tce_type;
    std::vector<int> tce_sim;
    std::vector<int> tce_layer_binary;
    std::vector<int> tce_moduleType_binary;
#ifdef CUT_VALUE_DEBUG
    std::vector<float> tce_rzChiSquared;
    std::vector<float> tce_rPhiChiSquared;
#endif
    std::vector<float> tce_simpt;
    std::vector<std::vector<int>> tce_nLayerOverlaps;
    std::vector<std::vector<int>> tce_nHitOverlaps;
    std::vector<int> tce_anchorIndex;
    std::vector<float> tc_pt = ana.tx->getBranch<vector<float>>("tc_pt");
    std::vector<float> tc_eta = ana.tx->getBranch<vector<float>>("tc_eta");
    std::vector<float> tc_phi = ana.tx->getBranch<vector<float>>("tc_phi");

    std::vector<float> t3_pt = ana.tx->getBranch<vector<float>>("t3_pt");
    std::vector<float> t3_eta = ana.tx->getBranch<vector<float>>("t3_eta");
    std::vector<float> t3_phi = ana.tx->getBranch<vector<float>>("t3_phi");
    unsigned int N_MAX_TRACK_EXTENSIONS_PER_TC = 30;
    const unsigned int N_MAX_T3T3_TRACK_EXTENSIONS = 40000;

    std::vector<int> tce_anchorType;;
#ifdef T3T3_EXTENSIONS
    for(size_t i = 0; i <= nTrackCandidates; i++) //CHEAT - Include the T3T3 extensions!
#else
    for(size_t i = 0; i < nTrackCandidates; i++)
#endif
    {
        unsigned int nTrackExtensions;
        if(i < nTrackCandidates)
        {
            nTrackExtensions = (trackExtensionsInGPU.nTrackExtensions)[i] > N_MAX_TRACK_EXTENSIONS_PER_TC ? N_MAX_TRACK_EXTENSIONS_PER_TC : (trackExtensionsInGPU.nTrackExtensions)[i];
        }
        else
        {
            nTrackExtensions = (trackExtensionsInGPU.nTrackExtensions)[i] > N_MAX_T3T3_TRACK_EXTENSIONS ? N_MAX_T3T3_TRACK_EXTENSIONS : (trackExtensionsInGPU.nTrackExtensions)[i]; 
        }
        for(size_t j = 0; j < nTrackExtensions; j++)
        {
            unsigned int teIdx = i * N_MAX_TRACK_EXTENSIONS_PER_TC + j;
            short anchorType = trackExtensionsInGPU.constituentTCTypes[3*teIdx];
            short outerType = trackExtensionsInGPU.constituentTCTypes[3*teIdx + 1];
    
            unsigned int anchorIndex = trackExtensionsInGPU.constituentTCIndices[3*teIdx];
            unsigned int outerIndex = trackExtensionsInGPU.constituentTCIndices[3*teIdx + 1];
            unsigned int layer_binary = 0;
            tce_anchorIndex.push_back(anchorIndex);
            //get the hit indices
            unsigned int* anchorHitIndices;
            unsigned int* outerHitIndices;
            uint8_t* anchorLogicalLayers;
            uint8_t* outerLogicalLayers;
            vector<int> hit_idxs;
            vector<int> module_idxs;
            vector<int> nLayerOverlaps;
            vector<int> nHitOverlaps;
    
            nLayerOverlaps.push_back(trackExtensionsInGPU.nLayerOverlaps[2*teIdx]);
            nHitOverlaps.push_back(trackExtensionsInGPU.nHitOverlaps[2*teIdx]);

            if(trackExtensionsInGPU.isDup[teIdx]) continue;

            tce_nLayerOverlaps.push_back(nLayerOverlaps);
            tce_nHitOverlaps.push_back(nHitOverlaps);
#ifdef CUT_VALUE_DEBUG
            tce_rPhiChiSquared.push_back(__H2F(trackExtensionsInGPU.rPhiChiSquared[teIdx]));
            tce_rzChiSquared.push_back(__H2F(trackExtensionsInGPU.rzChiSquared[teIdx]));
#endif
            if(anchorType != 3)
            {
                anchorHitIndices = &trackCandidatesInGPU.hitIndices[14 * anchorIndex];
                anchorLogicalLayers = &trackCandidatesInGPU.logicalLayers[7 * anchorIndex];
                tce_pt.push_back(tc_pt.at(anchorIndex));
                tce_eta.push_back(tc_eta.at(anchorIndex));
                tce_phi.push_back(tc_phi.at(anchorIndex));
            }
        
            else
            {
                anchorHitIndices = &tripletsInGPU.hitIndices[6 * anchorIndex];
                anchorLogicalLayers = &tripletsInGPU.logicalLayers[3 * anchorIndex];
                
                float x1 = hitsInGPU.xs[anchorHitIndices[0]];
                float x2 = hitsInGPU.xs[anchorHitIndices[2]];
                float x3 = hitsInGPU.xs[anchorHitIndices[4]];
                float y1 = hitsInGPU.ys[anchorHitIndices[0]];
                float y2 = hitsInGPU.ys[anchorHitIndices[2]];
                float y3 = hitsInGPU.ys[anchorHitIndices[4]];
                float z1 = hitsInGPU.zs[anchorHitIndices[0]];
                float z2 = hitsInGPU.zs[anchorHitIndices[2]];
                float z3 = hitsInGPU.zs[anchorHitIndices[4]];

                float g, f; // not used
                float regressionRadius = __H2F(trackExtensionsInGPU.regressionRadius[teIdx]);
                // Compute pt, eta, phi of T3
                const float kRinv1GeVf = (2.99792458e-3 * 3.8);

                const float pt = kRinv1GeVf * regressionRadius;
                float eta = -999;
                float phi = -999;
                SDL::CPU::Hit hitA(x1,y1,z1);
                SDL::CPU::Hit hitB(x2,y2,z2);
                eta = hitB.eta();
                phi = hitA.phi();

                tce_pt.push_back(pt);
                tce_eta.push_back(eta);
                tce_phi.push_back(phi);
           }

            if(outerType == 3)
            {
                outerHitIndices = &tripletsInGPU.hitIndices[6 * outerIndex];
                outerLogicalLayers = &tripletsInGPU.logicalLayers[3 * outerIndex];
            }
            else
            {
                outerHitIndices = &trackCandidatesInGPU.hitIndices[14 * anchorIndex];
                outerLogicalLayers = &trackCandidatesInGPU.logicalLayers[3 * outerIndex];
            }
            size_t anchorLimits = anchorType == 7 ? 14 : (anchorType == 3 ? 6 : 10);
            size_t outerLimits = outerType == 3 ? 6 : (outerType == 7 ? 14 : 10);
        
            for(size_t j = 0; j < anchorLimits; j++)
            {
                hit_idxs.push_back(hitsInGPU.idxs[anchorHitIndices[j]]);
                module_idxs.push_back(hitsInGPU.moduleIndices[anchorHitIndices[j]]);
            }
            for(size_t j = 0; j < (anchorLimits / 2); j++)
            {
                layer_binary |= (1 << anchorLogicalLayers[j]);
            }

            for(size_t j = 0; j < outerLimits; j++)
            {
                hit_idxs.push_back(hitsInGPU.idxs[outerHitIndices[j]]);
                module_idxs.push_back(hitsInGPU.moduleIndices[outerHitIndices[j]]);
            }
            for(size_t j = 0; j < (outerLimits/2); j++)
            {
                layer_binary |= (1 << outerLogicalLayers[j]);
            }
            std::vector<int> hit_types(hit_idxs.size(), 4);
            if(anchorType == 7 or anchorType == 5)
            {
                hit_types[0] = 0;
                hit_types[1] = 0;
                hit_types[2] = 0;
                hit_types[3] = 0;
            }
            std::vector<int> matched_sim_trk_idxs = matchedSimTrkIdxs(hit_idxs, hit_types);
            for(auto &isimtrk: matched_sim_trk_idxs)
            {
                sim_tce_matched[isimtrk]++;
                sim_tce_types[isimtrk].push_back(layer_binary);
            }
            tce_layer_binary.push_back(layer_binary);
            tce_isFake.push_back(matched_sim_trk_idxs.size() == 0);
            tce_matched_simIdx.push_back(matched_sim_trk_idxs); 
            tce_anchorType.push_back(anchorType);
        }
    }

    //CHEAT - fill the rest with Track Candidates
    std::vector<int> sim_TC_matched_nonextended = ana.tx->getBranch<vector<int>>("sim_TC_matched_nonextended");
    std::vector<int> tc_isFake = ana.tx->getBranch<vector<int>>("tc_isFake");
    std::vector<std::vector<int>> sim_TC_types = ana.tx->getBranch<vector<vector<int>>>("sim_TC_types");
    std::vector<std::vector<int>> tc_matched_simIdx = ana.tx->getBranch<vector<vector<int>>>("tc_matched_simIdx");
    std::vector<int> tc_partOfExtension = ana.tx->getBranch<vector<int>>("tc_partOfExtension");
    std::vector<int> tc_type = ana.tx->getBranch<vector<int>>("tc_type");

    for(size_t jdx = 0; jdx < sim_tce_matched.size(); jdx++)
    {
        sim_tce_matched[jdx] += sim_TC_matched_nonextended[jdx];
        for(auto &jt:sim_TC_types[jdx])
        {
            sim_tce_types[jdx].push_back(jt);
        }
    }
   
    for(unsigned int jdx = 0; jdx < tc_matched_simIdx.size(); jdx++)
    {
        if(tc_partOfExtension[jdx]) continue;

        tce_isFake.push_back(tc_isFake[jdx]);
        std::vector<int> temp;
        for(auto &jt:tc_matched_simIdx[jdx])
        {
            temp.push_back(jt);
        }
        tce_matched_simIdx.push_back(temp);
        tce_anchorType.push_back(tc_type[jdx]);

        tce_pt.push_back(tc_pt[jdx]);
        tce_eta.push_back(tc_eta[jdx]);
        tce_phi.push_back(tc_phi[jdx]);

    }

    vector<int> tce_isDuplicate(tce_matched_simIdx.size());
    for (unsigned int i = 0; i < tce_matched_simIdx.size(); ++i)
    {
        bool isDuplicate = false;
        for (unsigned int isim = 0; isim < tce_matched_simIdx[i].size(); ++isim)
        {
            if (sim_tce_matched[tce_matched_simIdx[i][isim]] > 1)
            {
                isDuplicate = true;
            }
        }
        tce_isDuplicate[i] = isDuplicate;
    }

    ana.tx->setBranch<vector<int>>("tce_anchorIndex", tce_anchorIndex);
    ana.tx->setBranch<vector<int>>("sim_tce_matched", sim_tce_matched);
    ana.tx->setBranch<vector<vector<int>>>("sim_tce_types", sim_tce_types);
    ana.tx->setBranch<vector<vector<int>>>("tce_matched_simIdx", tce_matched_simIdx);
    ana.tx->setBranch<vector<int>>("tce_isFake", tce_isFake);
    ana.tx->setBranch<vector<int>>("tce_isDuplicate", tce_isDuplicate);
    ana.tx->setBranch<vector<vector<int>>>("tce_nLayerOverlaps", tce_nLayerOverlaps);
    ana.tx->setBranch<vector<vector<int>>>("tce_nHitOverlaps", tce_nHitOverlaps);
    ana.tx->setBranch<vector<float>>("tce_pt", tce_pt);
    ana.tx->setBranch<vector<float>>("tce_eta", tce_eta);
    ana.tx->setBranch<vector<float>>("tce_phi", tce_phi);
    ana.tx->setBranch<vector<int>>("tce_layer_binary", tce_layer_binary);
    ana.tx->setBranch<vector<int>>("tce_anchorType", tce_anchorType);
}

//________________________________________________________________________________________________________________________________
void fillQuintupletOutputBranches(SDL::Event* event)
{
    SDL::quintuplets& quintupletsInGPU = (*event->getQuintuplets());
    SDL::triplets& tripletsInGPU = (*event->getTriplets());
    SDL::segments& segmentsInGPU = (*event->getSegments());
    SDL::miniDoublets& miniDoubletsInGPU = (*event->getMiniDoublets());
    SDL::hits& hitsInGPU = (*event->getHits());
    SDL::modules& modulesInGPU = (*event->getModules());
    SDL::objectRanges& rangesInGPU = (*event->getRanges());

    // Did it match to track candidate?
    std::vector<int> sim_T5_matched(trk.sim_pt().size());
    std::vector<vector<int>> sim_T5_types(trk.sim_pt().size());
    std::vector<int> t5_isFake;
    std::vector<vector<int>> t5_matched_simIdx;
    std::vector<float> t5_pt;
    std::vector<float> t5_eta;
    std::vector<float> t5_phi;
    std::vector<float> t5_eta_2;
    std::vector<float> t5_phi_2;
    std::vector<float> t5_score_rphisum;
    std::vector<vector<int>> t5_hitIdxs;
    std::vector<int> t5_foundDuplicate;

#ifdef CUT_VALUE_DEBUG
    std::vector<float> t5_innerRadius;
    std::vector<float> t5_innerRadiusMin;
    std::vector<float> t5_innerRadiusMax;
    std::vector<float> t5_innerRadiusMin2S;
    std::vector<float> t5_innerRadiusMax2S;
    std::vector<float> t5_outerRadius;
    std::vector<float> t5_regressionRadius;
    std::vector<float> t5_outerRadiusMin;
    std::vector<float> t5_outerRadiusMax;
    std::vector<float> t5_outerRadiusMin2S;
    std::vector<float> t5_outerRadiusMax2S;
    std::vector<float> t5_bridgeRadius;
    std::vector<float> t5_bridgeRadiusMin;
    std::vector<float> t5_bridgeRadiusMax;
    std::vector<float> t5_bridgeRadiusMin2S;
    std::vector<float> t5_bridgeRadiusMax2S;
    std::vector<std::vector<float>> t5_simpt;
    std::vector<float> t5_chiSquared;
    std::vector<float> t5_nonAnchorChiSquared;
    std::vector<int> layer_binaries;
    std::vector<int> moduleType_binaries;
#endif

    const int MAX_NQUINTUPLET_PER_MODULE = 3000;
    const float kRinv1GeVf = (2.99792458e-3 * 3.8);
    const float k2Rinv1GeVf = kRinv1GeVf / 2.;
    
    for(unsigned int idx = 0; idx < *(modulesInGPU.nLowerModules); idx++)
    {
        if(rangesInGPU.quintupletModuleIndices[idx] == -1)
        {
            continue;
        }

        unsigned int nQuintuplets = quintupletsInGPU.nQuintuplets[idx];

        for(unsigned int jdx = 0; jdx < nQuintuplets; jdx++)
        {
            unsigned int quintupletIndex = rangesInGPU.quintupletModuleIndices[idx] + jdx;
            unsigned int innerTripletIndex = quintupletsInGPU.tripletIndices[2 * quintupletIndex];
            unsigned int outerTripletIndex = quintupletsInGPU.tripletIndices[2 * quintupletIndex + 1];

            if(quintupletsInGPU.isDup[quintupletIndex]==1){continue;}
            t5_foundDuplicate.emplace_back(quintupletsInGPU.isDup[quintupletIndex]);
            t5_score_rphisum.emplace_back(__H2F(quintupletsInGPU.score_rphisum[quintupletIndex]));
            t5_eta_2.emplace_back(__H2F(quintupletsInGPU.eta[quintupletIndex]));
            t5_phi_2.emplace_back(__H2F(quintupletsInGPU.phi[quintupletIndex]));

#ifdef CUT_VALUE_DEBUG
            t5_innerRadius.push_back(quintupletsInGPU.innerRadius[quintupletIndex]);
            t5_innerRadiusMin.push_back(quintupletsInGPU.innerRadiusMin[quintupletIndex]);
            t5_innerRadiusMax.push_back(quintupletsInGPU.innerRadiusMax[quintupletIndex]);
            t5_innerRadiusMin2S.push_back(quintupletsInGPU.innerRadiusMin2S[quintupletIndex]);
            t5_innerRadiusMax2S.push_back(quintupletsInGPU.innerRadiusMax2S[quintupletIndex]);

            t5_outerRadius.push_back(quintupletsInGPU.outerRadius[quintupletIndex]);
            t5_regressionRadius.push_back(quintupletsInGPU.regressionRadius[quintupletIndex]);
            t5_outerRadiusMin.push_back(quintupletsInGPU.outerRadiusMin[quintupletIndex]);
            t5_outerRadiusMax.push_back(quintupletsInGPU.outerRadiusMax[quintupletIndex]);
            t5_outerRadiusMin2S.push_back(quintupletsInGPU.outerRadiusMin2S[quintupletIndex]);
            t5_outerRadiusMax2S.push_back(quintupletsInGPU.outerRadiusMax2S[quintupletIndex]);

            t5_bridgeRadius.push_back(quintupletsInGPU.bridgeRadius[quintupletIndex]);
            t5_bridgeRadiusMin.push_back(quintupletsInGPU.bridgeRadiusMin[quintupletIndex]);
            t5_bridgeRadiusMax.push_back(quintupletsInGPU.bridgeRadiusMax[quintupletIndex]);
            t5_bridgeRadiusMin2S.push_back(quintupletsInGPU.bridgeRadiusMin2S[quintupletIndex]);
            t5_bridgeRadiusMax2S.push_back(quintupletsInGPU.bridgeRadiusMax2S[quintupletIndex]);

            t5_chiSquared.push_back(quintupletsInGPU.chiSquared[quintupletIndex]);
            t5_nonAnchorChiSquared.push_back(quintupletsInGPU.nonAnchorChiSquared[quintupletIndex]);
#endif

            unsigned int innerTripletInnerSegmentIndex = tripletsInGPU.segmentIndices[2 * innerTripletIndex];
            unsigned int innerTripletOuterSegmentIndex = tripletsInGPU.segmentIndices[2 * innerTripletIndex + 1];
            unsigned int outerTripletInnerSegmentIndex = tripletsInGPU.segmentIndices[2 * outerTripletIndex];
            unsigned int outerTripletOuterSegmentIndex = tripletsInGPU.segmentIndices[2 * outerTripletIndex + 1];

            unsigned int innerTripletInnerSegmentInnerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * innerTripletInnerSegmentIndex];

             //same as innerTripletOuterSegmentInnerMiniDoubletIndex
            unsigned int innerTripletInnerSegmentOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * innerTripletInnerSegmentIndex + 1];
            
            //same as outerTripletInnerSegmentInnerMiniDoubletIndex
            unsigned int innerTripletOuterSegmentOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * innerTripletOuterSegmentIndex + 1];

            //same as outerTripletOuterSegmentInnerMiniDoubletIndex
            unsigned int outerTripletInnerSegmentOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * outerTripletInnerSegmentIndex + 1];

            unsigned int outerTripletOuterSegmentOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * outerTripletOuterSegmentIndex + 1];

            unsigned int innerTripletInnerSegmentInnerMiniDoubletLowerHitIndex = miniDoubletsInGPU.anchorHitIndices[innerTripletInnerSegmentInnerMiniDoubletIndex];
            unsigned int innerTripletInnerSegmentInnerMiniDoubletUpperHitIndex = miniDoubletsInGPU.outerHitIndices[ innerTripletInnerSegmentInnerMiniDoubletIndex];

            unsigned int innerTripletInnerSegmentOuterMiniDoubletLowerHitIndex = miniDoubletsInGPU.anchorHitIndices[ innerTripletInnerSegmentOuterMiniDoubletIndex];
            unsigned int innerTripletInnerSegmentOuterMiniDoubletUpperHitIndex = miniDoubletsInGPU.outerHitIndices[ innerTripletInnerSegmentOuterMiniDoubletIndex];

            unsigned int innerTripletOuterSegmentOuterMiniDoubletLowerHitIndex = miniDoubletsInGPU.anchorHitIndices[ innerTripletOuterSegmentOuterMiniDoubletIndex];
            unsigned int innerTripletOuterSegmentOuterMiniDoubletUpperHitIndex = miniDoubletsInGPU.outerHitIndices[ innerTripletOuterSegmentOuterMiniDoubletIndex];

            unsigned int outerTripletInnerSegmentOuterMiniDoubletLowerHitIndex = miniDoubletsInGPU.anchorHitIndices[ outerTripletInnerSegmentOuterMiniDoubletIndex];
            unsigned int outerTripletInnerSegmentOuterMiniDoubletUpperHitIndex = miniDoubletsInGPU.outerHitIndices[ outerTripletInnerSegmentOuterMiniDoubletIndex];
    
            unsigned int outerTripletOuterSegmentOuterMiniDoubletLowerHitIndex = miniDoubletsInGPU.anchorHitIndices[ outerTripletOuterSegmentOuterMiniDoubletIndex];
            unsigned int outerTripletOuterSegmentOuterMiniDoubletUpperHitIndex = miniDoubletsInGPU.outerHitIndices[ outerTripletOuterSegmentOuterMiniDoubletIndex];

            std::vector<int> hit_idxs = {
                (int) hitsInGPU.idxs[innerTripletInnerSegmentInnerMiniDoubletLowerHitIndex],
                (int) hitsInGPU.idxs[innerTripletInnerSegmentInnerMiniDoubletUpperHitIndex],
                (int) hitsInGPU.idxs[innerTripletInnerSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.idxs[innerTripletInnerSegmentOuterMiniDoubletUpperHitIndex],
                (int) hitsInGPU.idxs[innerTripletOuterSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.idxs[innerTripletOuterSegmentOuterMiniDoubletUpperHitIndex],
                (int) hitsInGPU.idxs[outerTripletInnerSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.idxs[outerTripletInnerSegmentOuterMiniDoubletUpperHitIndex],
                (int) hitsInGPU.idxs[outerTripletOuterSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.idxs[outerTripletOuterSegmentOuterMiniDoubletUpperHitIndex]
            };
            t5_hitIdxs.emplace_back(hit_idxs);

            std::vector<int> hit_types(hit_idxs.size(), 4);
            std::vector<int> module_idxs = {
                (int) hitsInGPU.moduleIndices[innerTripletInnerSegmentInnerMiniDoubletLowerHitIndex],
                (int) hitsInGPU.moduleIndices[innerTripletInnerSegmentInnerMiniDoubletUpperHitIndex],
                (int) hitsInGPU.moduleIndices[innerTripletInnerSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.moduleIndices[innerTripletInnerSegmentOuterMiniDoubletUpperHitIndex],
                (int) hitsInGPU.moduleIndices[innerTripletOuterSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.moduleIndices[innerTripletOuterSegmentOuterMiniDoubletUpperHitIndex],
                (int) hitsInGPU.moduleIndices[outerTripletInnerSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.moduleIndices[outerTripletInnerSegmentOuterMiniDoubletUpperHitIndex],
                (int) hitsInGPU.moduleIndices[outerTripletOuterSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.moduleIndices[outerTripletOuterSegmentOuterMiniDoubletUpperHitIndex]
            };

            int layer0 = modulesInGPU.layers[module_idxs[0]];
            int layer2 = modulesInGPU.layers[module_idxs[2]];
            int layer4 = modulesInGPU.layers[module_idxs[4]];
            int layer6 = modulesInGPU.layers[module_idxs[6]];
            int layer8 = modulesInGPU.layers[module_idxs[8]];

            int subdet0 = modulesInGPU.subdets[module_idxs[0]];
            int subdet2 = modulesInGPU.subdets[module_idxs[2]];
            int subdet4 = modulesInGPU.subdets[module_idxs[4]];
            int subdet6 = modulesInGPU.subdets[module_idxs[6]];
            int subdet8 = modulesInGPU.subdets[module_idxs[8]];

            int logicallayer0 = layer0 + 6 * (subdet0 == 4);
            int logicallayer2 = layer2 + 6 * (subdet2 == 4);
            int logicallayer4 = layer4 + 6 * (subdet4 == 4);
            int logicallayer6 = layer6 + 6 * (subdet6 == 4);
            int logicallayer8 = layer8 + 6 * (subdet8 == 4);

            int layer_binary = 0;
            layer_binary |= (1 << logicallayer0);
            layer_binary |= (1 << logicallayer2);
            layer_binary |= (1 << logicallayer4);
            layer_binary |= (1 << logicallayer6);
            layer_binary |= (1 << logicallayer8);
            


            float pt = k2Rinv1GeVf * (__H2F(quintupletsInGPU.innerRadius[quintupletIndex]) + __H2F(quintupletsInGPU.outerRadius[quintupletIndex]));

            //copyting stuff from before for eta and phi
            SDL::CPU::Hit hitA(trk.ph2_x()[hit_idxs[0]], trk.ph2_y()[hit_idxs[0]], trk.ph2_z()[hit_idxs[0]]);
            SDL::CPU::Hit hitB(trk.ph2_x()[hit_idxs[9]], trk.ph2_y()[hit_idxs[9]], trk.ph2_z()[hit_idxs[9]]);

            float eta = hitB.eta();
            float phi = hitA.phi();

            t5_pt.push_back(pt);
            t5_eta.push_back(eta);
            t5_phi.push_back(phi);

#ifdef CUT_VALUE_DEBUG

            int moduleType_binary = 0;
            int moduleType0 = modulesInGPU.moduleType[module_idxs[0]];
            int moduleType2 = modulesInGPU.moduleType[module_idxs[2]];
            int moduleType4 = modulesInGPU.moduleType[module_idxs[4]];
            int moduleType6 = modulesInGPU.moduleType[module_idxs[6]];
            int moduleType8 = modulesInGPU.moduleType[module_idxs[8]];
            
            moduleType_binary |= (moduleType0 << 0);
            moduleType_binary |= (moduleType2 << 2);
            moduleType_binary |= (moduleType4 << 4);
            moduleType_binary |= (moduleType6 << 6);
            moduleType_binary |= (moduleType8 << 8);

            layer_binaries.push_back(layer_binary);
            moduleType_binaries.push_back(moduleType_binary);
#endif
            std::vector<int> matched_sim_trk_idxs = matchedSimTrkIdxs(hit_idxs, hit_types);
            for (auto &isimtrk : matched_sim_trk_idxs)
            {
                sim_T5_matched[isimtrk]++;
            }
#ifdef CUT_VALUE_DEBUG
            std::vector<float> sim_pt_per_t5;
            if(matched_sim_trk_idxs.size() == 0)
            {
                sim_pt_per_t5.push_back(-999);
            }
            else
            {
		sim_pt_per_t5.push_back(trk.sim_pt()[matched_sim_trk_idxs[0]]);
            }
            t5_simpt.push_back(sim_pt_per_t5);
#endif

            for (auto &isimtrk : matched_sim_trk_idxs)
            {
                sim_T5_types[isimtrk].push_back(layer_binary);
            }
            t5_isFake.push_back(matched_sim_trk_idxs.size() == 0);
            t5_matched_simIdx.push_back(matched_sim_trk_idxs);
        }
    }

    std::vector<int> t5_isDuplicate(t5_matched_simIdx.size());

    for (unsigned int i = 0; i < t5_matched_simIdx.size(); ++i)
    {
        bool isDuplicate = false;
        for (unsigned int isim = 0; isim < t5_matched_simIdx[i].size(); ++isim)
        {
            if (sim_T5_matched[t5_matched_simIdx[i][isim]] > 1)
            {
                isDuplicate = true;
            }
        }
        t5_isDuplicate[i] = isDuplicate;
    }
    ana.tx->setBranch<vector<int>>("sim_T5_matched", sim_T5_matched);
    ana.tx->setBranch<vector<vector<int>>>("sim_T5_types", sim_T5_types);
    ana.tx->setBranch<vector<int>>("t5_isFake", t5_isFake);
    ana.tx->setBranch<vector<int>>("t5_isDuplicate", t5_isDuplicate);
    ana.tx->setBranch<vector<int>>("t5_foundDuplicate", t5_foundDuplicate);
    ana.tx->setBranch<vector<float>>("t5_pt", t5_pt);
    ana.tx->setBranch<vector<float>>("t5_eta", t5_eta);
    ana.tx->setBranch<vector<float>>("t5_phi", t5_phi);
    ana.tx->setBranch<vector<float>>("t5_eta_2", t5_eta_2);
    ana.tx->setBranch<vector<float>>("t5_phi_2", t5_phi_2);


    ana.tx->setBranch<vector<vector<int>>>("t5_matched_simIdx", t5_matched_simIdx);
    ana.tx->setBranch<vector<vector<int>>>("t5_hitIdxs", t5_hitIdxs);
    ana.tx->setBranch<vector<float>>("t5_score_rphisum", t5_score_rphisum);
#ifdef CUT_VALUE_DEBUG
    ana.tx->setBranch<vector<vector<float>>>("t5_matched_pt",t5_simpt);

    ana.tx->setBranch<vector<float>>("t5_outerRadius",t5_outerRadius);
    ana.tx->setBranch<vector<float>>("t5_regressionRadius", t5_regressionRadius);
    ana.tx->setBranch<vector<float>>("t5_outerRadiusMin",t5_outerRadiusMin);
    ana.tx->setBranch<vector<float>>("t5_outerRadiusMax",t5_outerRadiusMax);
    ana.tx->setBranch<vector<float>>("t5_outerRadiusMin2S",t5_outerRadiusMin2S);
    ana.tx->setBranch<vector<float>>("t5_outerRadiusMax2S",t5_outerRadiusMax2S);
    ana.tx->setBranch<vector<float>>("t5_chiSquared", t5_chiSquared);
    ana.tx->setBranch<vector<float>>("t5_nonAnchorChiSquared", t5_nonAnchorChiSquared);

    ana.tx->setBranch<vector<float>>("t5_innerRadius",t5_innerRadius);
    ana.tx->setBranch<vector<float>>("t5_innerRadiusMin",t5_innerRadiusMin);
    ana.tx->setBranch<vector<float>>("t5_innerRadiusMax",t5_innerRadiusMax);
    ana.tx->setBranch<vector<float>>("t5_innerRadiusMin2S",t5_innerRadiusMin2S);
    ana.tx->setBranch<vector<float>>("t5_innerRadiusMax2S",t5_innerRadiusMax2S);
    ana.tx->setBranch<vector<float>>("t5_bridgeRadius",t5_bridgeRadius);
    ana.tx->setBranch<vector<float>>("t5_bridgeRadiusMin",t5_bridgeRadiusMin);
    ana.tx->setBranch<vector<float>>("t5_bridgeRadiusMax",t5_bridgeRadiusMax);
    ana.tx->setBranch<vector<float>>("t5_bridgeRadiusMin2S",t5_bridgeRadiusMin2S);
    ana.tx->setBranch<vector<float>>("t5_bridgeRadiusMax2S",t5_bridgeRadiusMax2S);
    ana.tx->setBranch<vector<int>>("t5_layer_binary",layer_binaries); 
    ana.tx->setBranch<vector<int>>("t5_moduleType_binary", moduleType_binaries);
#endif

}

void fillPixelTripletOutputBranches(SDL::Event* event)
{
    SDL::pixelTriplets& pixelTripletsInGPU = (*event->getPixelTriplets());
    SDL::triplets& tripletsInGPU = (*event->getTriplets());
    SDL::segments& segmentsInGPU = (*event->getSegments());
    SDL::miniDoublets& mdsInGPU = (*event->getMiniDoublets());
    SDL::hits& hitsInGPU = (*event->getHits());
    SDL::modules& modulesInGPU = (*event->getModules());

    std::vector<int> sim_pT3_matched(trk.sim_pt().size(), 0);
    std::vector<vector<int>> sim_pT3_types(trk.sim_pt().size());
    std::vector<int> pT3_isFake;
    std::vector<vector<int>> pT3_matched_simIdx;
    std::vector<float> pT3_pt;
    std::vector<float> pT3_eta;
    std::vector<float> pT3_phi;
    std::vector<float> pT3_eta_2;
    std::vector<float> pT3_phi_2;
    std::vector<float> pT3_score;
    std::vector<int> pT3_foundDuplicate;
    std::vector<vector<int>> pT3_hitIdxs;

#ifdef CUT_VALUE_DEBUG
    std::vector<float> pT3_pixelRadius;
    std::vector<float> pT3_tripletRadius;
    std::vector<int> pT3_layer_binary;
    std::vector<int> pT3_moduleType_binary;
    std::vector<float> pT3_pixelRadiusError;
    std::vector<std::vector<float>> pT3_simpt;
    std::vector<float> pT3_rPhiChiSquared;
    std::vector<float> pT3_rPhiChiSquaredInwards;
    std::vector<float> pT3_rzChiSquared;
#endif

    const unsigned int N_MAX_PIXEL_TRIPLETS = 5000;

    unsigned int nPixelTriplets = *(pixelTripletsInGPU.nPixelTriplets);

    for(unsigned int jdx = 0; jdx < nPixelTriplets; jdx++)
    {
        unsigned int pixelSegmentIndex = pixelTripletsInGPU.pixelSegmentIndices[jdx];
        unsigned int tripletIndex = pixelTripletsInGPU.tripletIndices[jdx];

        if(pixelTripletsInGPU.isDup[jdx]==1){continue;}
        pT3_eta_2.emplace_back(__H2F(pixelTripletsInGPU.eta[jdx]));
        pT3_phi_2.emplace_back(__H2F(pixelTripletsInGPU.phi[jdx]));
        pT3_score.emplace_back(__H2F(pixelTripletsInGPU.score[jdx]));
        pT3_foundDuplicate.emplace_back(pixelTripletsInGPU.isDup[jdx]);
        
        unsigned int pixelInnerMDIndex = segmentsInGPU.mdIndices[2 * pixelSegmentIndex];
        unsigned int pixelOuterMDIndex = segmentsInGPU.mdIndices[2 * pixelSegmentIndex + 1];
        unsigned int tripletInnerSegmentIndex = tripletsInGPU.segmentIndices[2 * tripletIndex];
        unsigned int tripletOuterSegmentIndex = tripletsInGPU.segmentIndices[2 * tripletIndex + 1];

        unsigned int tripletInnerMDIndex = segmentsInGPU.mdIndices[2 * tripletInnerSegmentIndex];
        unsigned int tripletMiddleMDIndex = segmentsInGPU.mdIndices[2 * tripletInnerSegmentIndex + 1];
        unsigned int tripletOuterMDIndex = segmentsInGPU.mdIndices[2 * tripletOuterSegmentIndex + 1];
        unsigned int pixelInnerMDLowerHitIndex = mdsInGPU.anchorHitIndices[pixelInnerMDIndex];
        unsigned int pixelInnerMDUpperHitIndex = mdsInGPU.outerHitIndices[pixelInnerMDIndex];
        unsigned int pixelOuterMDLowerHitIndex = mdsInGPU.anchorHitIndices[pixelOuterMDIndex];
        unsigned int pixelOuterMDUpperHitIndex = mdsInGPU.outerHitIndices[pixelOuterMDIndex];

        unsigned int tripletInnerMDLowerHitIndex = mdsInGPU.anchorHitIndices[tripletInnerMDIndex];
        unsigned int tripletInnerMDUpperHitIndex = mdsInGPU.outerHitIndices[tripletInnerMDIndex];
        unsigned int tripletMiddleMDLowerHitIndex = mdsInGPU.anchorHitIndices[tripletMiddleMDIndex];
        unsigned int tripletMiddleMDUpperHitIndex = mdsInGPU.outerHitIndices[tripletMiddleMDIndex];
        unsigned int tripletOuterMDLowerHitIndex = mdsInGPU.anchorHitIndices[tripletOuterMDIndex];
        unsigned int tripletOuterMDUpperHitIndex = mdsInGPU.outerHitIndices[tripletOuterMDIndex];

        std::vector<int> hit_idxs = {
            (int) hitsInGPU.idxs[pixelInnerMDLowerHitIndex],
            (int) hitsInGPU.idxs[pixelInnerMDUpperHitIndex],
            (int) hitsInGPU.idxs[pixelOuterMDLowerHitIndex],
            (int) hitsInGPU.idxs[pixelOuterMDUpperHitIndex],
            (int) hitsInGPU.idxs[tripletInnerMDLowerHitIndex],
            (int) hitsInGPU.idxs[tripletInnerMDUpperHitIndex],
            (int) hitsInGPU.idxs[tripletMiddleMDLowerHitIndex],
            (int) hitsInGPU.idxs[tripletMiddleMDUpperHitIndex],
            (int) hitsInGPU.idxs[tripletOuterMDLowerHitIndex],
            (int) hitsInGPU.idxs[tripletOuterMDUpperHitIndex]
        };

        pT3_hitIdxs.emplace_back(hit_idxs);

        std::vector<int> hit_types;
        hit_types.push_back(0);
        hit_types.push_back(0);
        hit_types.push_back(0);
        hit_types.push_back(0);
        hit_types.push_back(4);
        hit_types.push_back(4);
        hit_types.push_back(4);
        hit_types.push_back(4);
        hit_types.push_back(4);
        hit_types.push_back(4);

        std::vector<int> module_idxs = {
            (int) hitsInGPU.moduleIndices[pixelInnerMDLowerHitIndex],
            (int) hitsInGPU.moduleIndices[pixelInnerMDUpperHitIndex],
            (int) hitsInGPU.moduleIndices[pixelOuterMDLowerHitIndex],
            (int) hitsInGPU.moduleIndices[pixelOuterMDUpperHitIndex],
            (int) hitsInGPU.moduleIndices[tripletInnerMDLowerHitIndex],
            (int) hitsInGPU.moduleIndices[tripletInnerMDUpperHitIndex],
            (int) hitsInGPU.moduleIndices[tripletMiddleMDLowerHitIndex],
            (int) hitsInGPU.moduleIndices[tripletMiddleMDUpperHitIndex],
            (int) hitsInGPU.moduleIndices[tripletOuterMDLowerHitIndex],
            (int) hitsInGPU.moduleIndices[tripletOuterMDUpperHitIndex]        };
        int layer0 = modulesInGPU.layers[module_idxs[0]];
        int layer2 = modulesInGPU.layers[module_idxs[2]];
        int layer4 = modulesInGPU.layers[module_idxs[4]];
        int layer6 = modulesInGPU.layers[module_idxs[6]];
        int layer8 = modulesInGPU.layers[module_idxs[8]];

        int subdet0 = modulesInGPU.subdets[module_idxs[0]];
        int subdet2 = modulesInGPU.subdets[module_idxs[2]];
        int subdet4 = modulesInGPU.subdets[module_idxs[4]];
        int subdet6 = modulesInGPU.subdets[module_idxs[6]];
        int subdet8 = modulesInGPU.subdets[module_idxs[8]];


        int logicallayer0 = 0;
        int logicallayer2 = 0;
        int logicallayer4 = layer4 + 6 * (subdet4 == 4);
        int logicallayer6 = layer6 + 6 * (subdet6 == 4);
        int logicallayer8 = layer8 + 6 * (subdet8 == 4);

        int layer_binary = 0;
        layer_binary |= (1 << logicallayer0);
        layer_binary |= (1 << logicallayer2);
        layer_binary |= (1 << logicallayer4);
        layer_binary |= (1 << logicallayer6);
        layer_binary |= (1 << logicallayer8);
          
        //bare bones implementation only
        std::vector<int> matched_sim_trk_idxs = matchedSimTrkIdxs(hit_idxs, hit_types);
        for (auto &isimtrk : matched_sim_trk_idxs)
        {
            sim_pT3_matched[isimtrk]++;
        }
#ifdef CUT_VALUE_DEBUG
        pT3_layer_binary.push_back(layer_binary);
        int moduleType0 = modulesInGPU.moduleType[module_idxs[0]];
        int moduleType2 = modulesInGPU.moduleType[module_idxs[2]];
        int moduleType4 = modulesInGPU.moduleType[module_idxs[4]];
        int moduleType6 = modulesInGPU.moduleType[module_idxs[6]];
        int moduleType8 = modulesInGPU.moduleType[module_idxs[8]];

        int moduleType_binary = 0;
        moduleType_binary |= (moduleType4 << 0);
        moduleType_binary |= (moduleType6 << 2);
        moduleType_binary |= (moduleType8 << 4);

        pT3_moduleType_binary.push_back(moduleType_binary);
        std::vector<float> sim_pt_per_pT3;
        if(matched_sim_trk_idxs.size() == 0)
        {
            sim_pt_per_pT3.push_back(-999);
        }
        else
        {
            sim_pt_per_pT3.push_back(trk.sim_pt()[matched_sim_trk_idxs[0]]);
        }
        pT3_simpt.push_back(sim_pt_per_pT3);
#endif
        for (auto &isimtrk : matched_sim_trk_idxs)
        {
            sim_pT3_types[isimtrk].push_back(layer_binary);
        }
        pT3_isFake.push_back(matched_sim_trk_idxs.size() == 0);
        pT3_matched_simIdx.push_back(matched_sim_trk_idxs);

        float pixelRadius = __H2F(pixelTripletsInGPU.pixelRadius[jdx]);
        float pixelRadiusError = pixelTripletsInGPU.pixelRadiusError[jdx];
        float tripletRadius = __H2F(pixelTripletsInGPU.tripletRadius[jdx]);
        const float kRinv1GeVf = (2.99792458e-3 * 3.8);
        const float k2Rinv1GeVf = kRinv1GeVf / 2.;

        float pt = k2Rinv1GeVf * (pixelRadius + tripletRadius);
        //copyting stuff from before for eta and phi
        SDL::CPU::Hit hitA(trk.pix_x()[hit_idxs[0]], trk.pix_y()[hit_idxs[0]], trk.pix_z()[hit_idxs[0]]);
        SDL::CPU::Hit hitB(trk.ph2_x()[hit_idxs[9]], trk.ph2_y()[hit_idxs[9]], trk.ph2_z()[hit_idxs[9]]);

        float eta = hitB.eta();
        float phi = hitA.phi();

        pT3_pt.push_back(pt);
        pT3_eta.push_back(eta);
        pT3_phi.push_back(phi);
#ifdef CUT_VALUE_DEBUG
        pT3_pixelRadius.push_back(pixelRadius);
        pT3_pixelRadiusError.push_back(pixelRadiusError);
        pT3_tripletRadius.push_back(tripletRadius);
        pT3_rPhiChiSquared.push_back(pixelTripletsInGPU.rPhiChiSquared[jdx]);
        pT3_rPhiChiSquaredInwards.push_back(pixelTripletsInGPU.rPhiChiSquaredInwards[jdx]);
        pT3_rzChiSquared.push_back(pixelTripletsInGPU.rzChiSquared[jdx]);
#endif
    }

    vector<int> pT3_isDuplicate(pT3_matched_simIdx.size());

    for (unsigned int i = 0; i < pT3_matched_simIdx.size(); ++i)
    {
        bool isDuplicate = false;
        for (unsigned int isim = 0; isim < pT3_matched_simIdx[i].size(); ++isim)
        {
            if (sim_pT3_matched[pT3_matched_simIdx[i][isim]] > 1)
            {
                isDuplicate = true;
            }
        }
        pT3_isDuplicate[i] = isDuplicate;
    }

    ana.tx->setBranch<vector<int>>("sim_pT3_matched", sim_pT3_matched);
    ana.tx->setBranch<vector<vector<int>>>("sim_pT3_types", sim_pT3_types);
    ana.tx->setBranch<vector<int>>("pT3_isFake", pT3_isFake);
    ana.tx->setBranch<vector<int>>("pT3_isDuplicate", pT3_isDuplicate);
    ana.tx->setBranch<vector<float>>("pT3_pt", pT3_pt);
    ana.tx->setBranch<vector<float>>("pT3_eta", pT3_eta);
    ana.tx->setBranch<vector<float>>("pT3_phi", pT3_phi);
    ana.tx->setBranch<vector<float>>("pT3_eta_2", pT3_eta_2);
    ana.tx->setBranch<vector<float>>("pT3_phi_2", pT3_phi_2);
    ana.tx->setBranch<vector<float>>("pT3_score", pT3_score);
    ana.tx->setBranch<vector<int>>("pT3_foundDuplicate", pT3_foundDuplicate);
    ana.tx->setBranch<vector<vector<int>>>("pT3_matched_simIdx", pT3_matched_simIdx);
    ana.tx->setBranch<vector<vector<int>>>("pT3_hitIdxs", pT3_hitIdxs);
#ifdef CUT_VALUE_DEBUG
    ana.tx->setBranch<vector<vector<float>>>("pT3_matched_pt", pT3_simpt);
    ana.tx->setBranch<vector<float>>("pT3_pixelRadius", pT3_pixelRadius);
    ana.tx->setBranch<vector<float>>("pT3_pixelRadiusError", pT3_pixelRadiusError);
    ana.tx->setBranch<vector<float>>("pT3_tripletRadius", pT3_tripletRadius);
    ana.tx->setBranch<vector<int>>("pT3_layer_binary", pT3_layer_binary);
    ana.tx->setBranch<vector<int>>("pT3_moduleType_binary", pT3_moduleType_binary);
    ana.tx->setBranch<vector<float>>("pT3_rPhiChiSquared", pT3_rPhiChiSquared);
    ana.tx->setBranch<vector<float>>("pT3_rPhiChiSquaredInwards", pT3_rPhiChiSquaredInwards);
    ana.tx->setBranch<vector<float>>("pT3_rzChiSquared", pT3_rzChiSquared);
#endif

}

void fillPixelQuintupletOutputBranches(SDL::Event* event)
{
    SDL::pixelQuintuplets& pixelQuintupletsInGPU = (*event->getPixelQuintuplets());
    SDL::quintuplets& quintupletsInGPU = (*event->getQuintuplets());
    SDL::triplets& tripletsInGPU = (*event->getTriplets());
    SDL::segments& segmentsInGPU = (*event->getSegments());
    SDL::miniDoublets& mdsInGPU = (*event->getMiniDoublets());
    SDL::hits& hitsInGPU = (*event->getHits());
    SDL::modules& modulesInGPU = (*event->getModules());

    std::vector<int> sim_pT5_matched(trk.sim_pt().size(), 0);
    std::vector<vector<int>> sim_pT5_types(trk.sim_pt().size());
    std::vector<int> pT5_isFake;
    std::vector<int> pT5_score;
    std::vector<float> pT5_pt;
    std::vector<float> pT5_eta;
    std::vector<float> pT5_phi;
    std::vector<vector<int>> pT5_hitIdxs;
    std::vector<vector<int>> pT5_matched_simIdx;

#ifdef CUT_VALUE_DEBUG
    std::vector<int> pT5_layer_binary;
    std::vector<int> pT5_moduleType_binary;
    std::vector<float> pT5_rzChiSquared;
    std::vector<float> pT5_rPhiChiSquared;
    std::vector<float> pT5_rPhiChiSquaredInwards;
    std::vector<float> pT5_simpt;
#endif
    const unsigned int N_MAX_PIXEL_QUINTUPLETS = 15000;
    unsigned int nPixelQuintuplets = *(pixelQuintupletsInGPU.nPixelQuintuplets);

    for(unsigned int jdx = 0; jdx < nPixelQuintuplets; jdx++)
    {
        //obtain the hits
        if(pixelQuintupletsInGPU.isDup[jdx]) {continue;};
        pT5_score.emplace_back(__H2F(pixelQuintupletsInGPU.score[jdx]));
        unsigned int T5Index = pixelQuintupletsInGPU.T5Indices[jdx];
    
        unsigned int T5InnerTripletIndex = quintupletsInGPU.tripletIndices[2 * T5Index];
        unsigned int T5OuterTripletIndex = quintupletsInGPU.tripletIndices[2 * T5Index + 1];
        unsigned int pixelSegmentIndex = pixelQuintupletsInGPU.pixelIndices[jdx];
        unsigned int T5InnerTripletInnerSegmentIndex = tripletsInGPU.segmentIndices[2 * T5InnerTripletIndex]; 
        unsigned int T5InnerTripletOuterSegmentIndex = tripletsInGPU.segmentIndices[2 * T5InnerTripletIndex + 1];
        unsigned int T5OuterTripletInnerSegmentIndex = tripletsInGPU.segmentIndices[2 * T5OuterTripletIndex];
        unsigned int T5OuterTripletOuterSegmentIndex = tripletsInGPU.segmentIndices[2 * T5OuterTripletIndex + 1];

        //7 MDs -> 14 hits!
        unsigned int pixelInnerMDIndex = segmentsInGPU.mdIndices[2 * pixelSegmentIndex];
        unsigned int pixelOuterMDIndex = segmentsInGPU.mdIndices[2 * pixelSegmentIndex + 1];
        unsigned int T5MDIndex1 = segmentsInGPU.mdIndices[2 * T5InnerTripletInnerSegmentIndex];
        unsigned int T5MDIndex2 = segmentsInGPU.mdIndices[2 * T5InnerTripletInnerSegmentIndex + 1];
        unsigned int T5MDIndex3 = segmentsInGPU.mdIndices[2 * T5InnerTripletOuterSegmentIndex + 1];
        unsigned int T5MDIndex4 = segmentsInGPU.mdIndices[2 * T5OuterTripletInnerSegmentIndex + 1];
        unsigned int T5MDIndex5 = segmentsInGPU.mdIndices[2 * T5OuterTripletOuterSegmentIndex + 1];

        unsigned int hitIndex1 = mdsInGPU.anchorHitIndices[pixelInnerMDIndex];
        unsigned int hitIndex2 = mdsInGPU.outerHitIndices[pixelInnerMDIndex];
        unsigned int hitIndex3 = mdsInGPU.anchorHitIndices[pixelOuterMDIndex];
        unsigned int hitIndex4 = mdsInGPU.outerHitIndices[pixelOuterMDIndex];
        unsigned int hitIndex5 = mdsInGPU.anchorHitIndices[T5MDIndex1];
        unsigned int hitIndex6 = mdsInGPU.outerHitIndices[T5MDIndex1];
        unsigned int hitIndex7 = mdsInGPU.anchorHitIndices[T5MDIndex2];
        unsigned int hitIndex8 = mdsInGPU.outerHitIndices[T5MDIndex2];
        unsigned int hitIndex9 = mdsInGPU.anchorHitIndices[T5MDIndex3];
        unsigned int hitIndex10 = mdsInGPU.outerHitIndices[T5MDIndex3];
        unsigned int hitIndex11 = mdsInGPU.anchorHitIndices[T5MDIndex4];
        unsigned int hitIndex12 = mdsInGPU.outerHitIndices[T5MDIndex4];
        unsigned int hitIndex13 = mdsInGPU.anchorHitIndices[T5MDIndex5];
        unsigned int hitIndex14 = mdsInGPU.outerHitIndices[T5MDIndex5];

        std::vector<int> hit_idxs = {
            (int) hitsInGPU.idxs[hitIndex1],
            (int) hitsInGPU.idxs[hitIndex2],
            (int) hitsInGPU.idxs[hitIndex3],
            (int) hitsInGPU.idxs[hitIndex4],
            (int) hitsInGPU.idxs[hitIndex5],
            (int) hitsInGPU.idxs[hitIndex6],
            (int) hitsInGPU.idxs[hitIndex7],
            (int) hitsInGPU.idxs[hitIndex8],
            (int) hitsInGPU.idxs[hitIndex9],
            (int) hitsInGPU.idxs[hitIndex10],
            (int) hitsInGPU.idxs[hitIndex11],
            (int) hitsInGPU.idxs[hitIndex12],
            (int) hitsInGPU.idxs[hitIndex13],
            (int) hitsInGPU.idxs[hitIndex14],
        };

        pT5_hitIdxs.emplace_back(hit_idxs);

        std::vector<int> hit_types;
        hit_types.push_back(0);
        hit_types.push_back(0);
        hit_types.push_back(0);
        hit_types.push_back(0);
        hit_types.push_back(4);
        hit_types.push_back(4);
        hit_types.push_back(4);
        hit_types.push_back(4);
        hit_types.push_back(4);
        hit_types.push_back(4);
        hit_types.push_back(4);
        hit_types.push_back(4);
        hit_types.push_back(4);
        hit_types.push_back(4); 

        std::vector<int> module_idxs = {
            (int) hitsInGPU.moduleIndices[hitIndex1],
            (int) hitsInGPU.moduleIndices[hitIndex2],
            (int) hitsInGPU.moduleIndices[hitIndex3],
            (int) hitsInGPU.moduleIndices[hitIndex4],
            (int) hitsInGPU.moduleIndices[hitIndex5],
            (int) hitsInGPU.moduleIndices[hitIndex6],
            (int) hitsInGPU.moduleIndices[hitIndex7],
            (int) hitsInGPU.moduleIndices[hitIndex8],
            (int) hitsInGPU.moduleIndices[hitIndex9],
            (int) hitsInGPU.moduleIndices[hitIndex10],
            (int) hitsInGPU.moduleIndices[hitIndex11],
            (int) hitsInGPU.moduleIndices[hitIndex12],
            (int) hitsInGPU.moduleIndices[hitIndex13],
            (int) hitsInGPU.moduleIndices[hitIndex14],
        };

        //layer binary -> disregard the 4 pixels!
        int layer0 = modulesInGPU.layers[module_idxs[0]];
        int layer2 = modulesInGPU.layers[module_idxs[2]];
        int layer4 = modulesInGPU.layers[module_idxs[4]];
        int layer6 = modulesInGPU.layers[module_idxs[6]];
        int layer8 = modulesInGPU.layers[module_idxs[8]];
        int layer10 = modulesInGPU.layers[module_idxs[10]];
        int layer12 = modulesInGPU.layers[module_idxs[12]];

        int subdet4 = modulesInGPU.subdets[module_idxs[4]];
        int subdet6 = modulesInGPU.subdets[module_idxs[6]];
        int subdet8 = modulesInGPU.subdets[module_idxs[8]];
        int subdet10 = modulesInGPU.subdets[module_idxs[10]];
        int subdet12 = modulesInGPU.subdets[module_idxs[12]];

        int logicallayer0 = 0;
        int logicallayer2 = 0;
        int logicallayer4 = layer4 + 6 * (subdet4 == 4);
        int logicallayer6 = layer6 + 6 * (subdet6 == 4);
        int logicallayer8 = layer8 + 6 * (subdet8 == 4);
        int logicallayer10 = layer10 + 6 * (subdet10 == 4);
        int logicallayer12 = layer12 + 6 * (subdet12 == 4);

        int layer_binary = 0;
        layer_binary |= (1 << logicallayer0);
        layer_binary |= (1 << logicallayer2);
        layer_binary |= (1 << logicallayer4);
        layer_binary |= (1 << logicallayer6);
        layer_binary |= (1 << logicallayer8);
        layer_binary |= (1 << logicallayer10);
        layer_binary |= (1 << logicallayer12);

        std::vector<int> matched_sim_trk_idxs = matchedSimTrkIdxs(hit_idxs, hit_types);
        for (auto &isimtrk : matched_sim_trk_idxs)
        {
            sim_pT5_matched[isimtrk]++;
        }

#ifdef CUT_VALUE_DEBUG

        int moduleType0 = modulesInGPU.moduleType[module_idxs[0]];
        int moduleType2 = modulesInGPU.moduleType[module_idxs[2]];
        int moduleType4 = modulesInGPU.moduleType[module_idxs[4]];
        int moduleType6 = modulesInGPU.moduleType[module_idxs[6]];
        int moduleType8 = modulesInGPU.moduleType[module_idxs[8]];
        int moduleType10 = modulesInGPU.moduleType[module_idxs[10]];
        int moduleType12 = modulesInGPU.moduleType[module_idxs[12]];

        int moduleType_binary = 0;
        moduleType_binary |= (moduleType4 << 0);
        moduleType_binary |= (moduleType6 << 2);
        moduleType_binary |= (moduleType8 << 4);
        moduleType_binary |= (moduleType10 << 6);
        moduleType_binary |= (moduleType12 << 8);

        pT5_layer_binary.push_back(layer_binary);

        int moduleType0 = modulesInGPU.moduleType[module_idxs[0]];
        int moduleType2 = modulesInGPU.moduleType[module_idxs[2]];
        int moduleType4 = modulesInGPU.moduleType[module_idxs[4]];
        int moduleType6 = modulesInGPU.moduleType[module_idxs[6]];
        int moduleType8 = modulesInGPU.moduleType[module_idxs[8]];
        int moduleType10 = modulesInGPU.moduleType[module_idxs[10]];
        int moduleType12 = modulesInGPU.moduleType[module_idxs[12]];


        int moduleType_binary = 0;
        moduleType_binary |= (moduleType4 << 0);
        moduleType_binary |= (moduleType6 << 2);
        moduleType_binary |= (moduleType8 << 4);
        moduleType_binary |= (moduleType10 << 6);
        moduleType_binary |= (moduleType12 << 8);

        pT5_moduleType_binary.push_back(moduleType_binary);
        pT5_rzChiSquared.push_back(pixelQuintupletsInGPU.rzChiSquared[jdx]);
        pT5_rPhiChiSquared.push_back(pixelQuintupletsInGPU.rPhiChiSquared[jdx]);
        pT5_rPhiChiSquaredInwards.push_back(pixelQuintupletsInGPU.rPhiChiSquaredInwards[jdx]);
        std::vector<float> sim_pt_per_pT5;

        if(matched_sim_trk_idxs.size() == 0)
        {
            sim_pt_per_pT5.push_back(-999);
        }
        else
        {
            pT5_simpt.push_back(trk.sim_pt()[matched_sim_trk_idxs[0]]);
        }

#endif
        for (auto &isimtrk : matched_sim_trk_idxs)
        {
            sim_pT5_types[isimtrk].push_back(layer_binary);
        }
        pT5_isFake.push_back(matched_sim_trk_idxs.size() == 0);
        pT5_matched_simIdx.push_back(matched_sim_trk_idxs);

        const float kRinv1GeVf = (2.99792458e-3 * 3.8);
        float pt = (segmentsInGPU.ptIn[pixelSegmentIndex - (*(modulesInGPU.nLowerModules))*600] +  quintupletsInGPU.regressionRadius[T5Index] * kRinv1GeVf) / 2;

        SDL::CPU::Hit hitA(trk.pix_x()[hit_idxs[0]], trk.pix_y()[hit_idxs[0]], trk.pix_z()[hit_idxs[0]]);
        SDL::CPU::Hit hitB(trk.ph2_x()[hit_idxs[13]], trk.ph2_y()[hit_idxs[13]], trk.ph2_z()[hit_idxs[13]]);

        float eta = hitB.eta();
        float phi = hitA.phi();

        pT5_pt.push_back(pt);
        pT5_eta.push_back(eta);
        pT5_phi.push_back(phi);
        
    }

    vector<int> pT5_isDuplicate(pT5_matched_simIdx.size());

    for (unsigned int i = 0; i < pT5_matched_simIdx.size(); ++i)
    {
        bool isDuplicate = false;
        for (unsigned int isim = 0; isim < pT5_matched_simIdx[i].size(); ++isim)
        {
            if (sim_pT5_matched[pT5_matched_simIdx[i][isim]] > 1)
            {
                isDuplicate = true;
            }
        }
        pT5_isDuplicate[i] = isDuplicate;
    }

    ana.tx->setBranch<vector<int>>("sim_pT5_matched", sim_pT5_matched);
    ana.tx->setBranch<vector<vector<int>>>("sim_pT5_types", sim_pT5_types);
    ana.tx->setBranch<vector<int>>("pT5_isFake", pT5_isFake);
    ana.tx->setBranch<vector<int>>("pT5_isDuplicate", pT5_isDuplicate);
    ana.tx->setBranch<vector<int>>("pT5_score", pT5_score);
    ana.tx->setBranch<vector<float>>("pT5_pt", pT5_pt);
    ana.tx->setBranch<vector<float>>("pT5_eta", pT5_eta);
    ana.tx->setBranch<vector<float>>("pT5_phi", pT5_phi);
    ana.tx->setBranch<vector<vector<int>>>("pT5_matched_simIdx", pT5_matched_simIdx);
    ana.tx->setBranch<vector<vector<int>>>("pT5_hitIdxs", pT5_hitIdxs);

#ifdef CUT_VALUE_DEBUG
    ana.tx->setBranch<vector<int>>("pT5_layer_binary", pT5_layer_binary);
    ana.tx->setBranch<vector<int>>("pT5_moduleType_binary", pT5_moduleType_binary);
    ana.tx->setBranch<vector<float>>("pT5_rzChiSquared", pT5_rzChiSquared);
    ana.tx->setBranch<vector<float>>("pT5_rPhiChiSquared", pT5_rPhiChiSquared);
    ana.tx->setBranch<vector<float>>("pT5_rPhiChiSquaredInwards", pT5_rPhiChiSquaredInwards);
    ana.tx->setBranch<vector<float>>("pT5_matched_pt", pT5_simpt);
#endif
}

//________________________________________________________________________________________________________________________________
void fillPixelLineSegmentOutputBranches(SDL::Event* event)
{
    SDL::segments& segmentsInGPU = (*event->getSegments());
    SDL::miniDoublets& miniDoubletsInGPU = (*event->getMiniDoublets());
    SDL::hits& hitsInGPU = (*event->getHits());
    SDL::modules& modulesInGPU = (*event->getModules());

    std::vector<int> sim_pLS_matched(trk.sim_pt().size(), 0);
    std::vector<vector<int>> sim_pLS_types(trk.sim_pt().size());
    std::vector<int> pLS_isFake;
    std::vector<vector<int>> pLS_matched_simIdx;
    std::vector<float> pLS_pt;
    std::vector<float> pLS_eta;
    std::vector<float> pLS_phi;
    std::vector<float> pLS_score;

    const unsigned int N_MAX_PIXEL_SEGMENTS_PER_MODULE = 50000; 
    const unsigned int N_MAX_SEGMENTS_PER_MODULE = 600;
    unsigned int pixelModuleIndex = *(modulesInGPU.nLowerModules);
    unsigned int nPixelSegments = segmentsInGPU.nSegments[pixelModuleIndex];
    for(unsigned int jdx = 0; jdx < nPixelSegments; jdx++)
    {
        if(segmentsInGPU.isDup[jdx]) {continue;}
        if(!segmentsInGPU.isQuad[jdx]) {continue;}
        pLS_score.push_back(segmentsInGPU.score[jdx]);
        unsigned int pixelSegmentIndex = pixelModuleIndex * N_MAX_SEGMENTS_PER_MODULE + jdx;
        unsigned int innerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * pixelSegmentIndex];
        unsigned int outerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * pixelSegmentIndex + 1];
        unsigned int innerMiniDoubletLowerHitIndex = miniDoubletsInGPU.anchorHitIndices[innerMiniDoubletIndex];
        unsigned int innerMiniDoubletUpperHitIndex = miniDoubletsInGPU.outerHitIndices[innerMiniDoubletIndex];
        unsigned int outerMiniDoubletLowerHitIndex = miniDoubletsInGPU.anchorHitIndices[outerMiniDoubletIndex];
        unsigned int outerMiniDoubletUpperHitIndex = miniDoubletsInGPU.outerHitIndices[outerMiniDoubletIndex];

        std::vector<int> hit_idxs = {
            (int) hitsInGPU.idxs[innerMiniDoubletLowerHitIndex], 
            (int) hitsInGPU.idxs[innerMiniDoubletUpperHitIndex], 
            (int) hitsInGPU.idxs[outerMiniDoubletLowerHitIndex], 
            (int) hitsInGPU.idxs[outerMiniDoubletUpperHitIndex]
        };

        std::vector<int> hit_types = {0,0,0,0};
        std::vector<int> module_idxs = {
            pixelModuleIndex, pixelModuleIndex, pixelModuleIndex, pixelModuleIndex
        };
        int layer0 = modulesInGPU.layers[module_idxs[0]];
        int layer2 = modulesInGPU.layers[module_idxs[2]];

        int subdet0 = modulesInGPU.subdets[module_idxs[0]];
        int subdet2 = modulesInGPU.subdets[module_idxs[2]];

        int logicallayer0 = 0;
        int logicallayer2 = 0;

        int layer_binary = 0;
        layer_binary |= (1 << logicallayer0);
        layer_binary |= (1 << logicallayer2);

        // sim track matched index
        std::vector<int> matched_sim_trk_idxs = matchedSimTrkIdxs(hit_idxs, hit_types);

        for (auto &isimtrk : matched_sim_trk_idxs)
        {
            sim_pLS_matched[isimtrk]++;
        }

        for (auto &isimtrk : matched_sim_trk_idxs)
        {
            sim_pLS_types[isimtrk].push_back(layer_binary);
        }


        pLS_isFake.push_back(matched_sim_trk_idxs.size() == 0);
        pLS_matched_simIdx.push_back(matched_sim_trk_idxs);

        pLS_pt.push_back(segmentsInGPU.ptIn[jdx]);
        pLS_eta.push_back(segmentsInGPU.eta[jdx]);
        pLS_phi.push_back(segmentsInGPU.phi[jdx]);

    }
    vector<int> pLS_isDuplicate(pLS_matched_simIdx.size());

    for (unsigned int i = 0; i < pLS_matched_simIdx.size(); ++i)
    {
        bool isDuplicate = false;
        for (unsigned int isim = 0; isim < pLS_matched_simIdx[i].size(); ++isim)
        {
            if (sim_pLS_matched[pLS_matched_simIdx[i][isim]] > 1)
            {
                isDuplicate = true;
            }
        }
        pLS_isDuplicate[i] = isDuplicate;
    }

    ana.tx->setBranch<vector<int>>("sim_pLS_matched",sim_pLS_matched);
    ana.tx->setBranch<vector<vector<int>>>("sim_pLS_types",sim_pLS_types);
    ana.tx->setBranch<vector<float>>("pLS_pt",pLS_pt);
    ana.tx->setBranch<vector<float>>("pLS_eta",pLS_eta);
    ana.tx->setBranch<vector<float>>("pLS_phi",pLS_phi);
    ana.tx->setBranch<vector<float>>("pLS_score",pLS_score);
    ana.tx->setBranch<vector<int>>("pLS_isFake",pLS_isFake);
    ana.tx->setBranch<vector<int>>("pLS_isDuplicate",pLS_isDuplicate);
}

//________________________________________________________________________________________________________________________________
void fillPixelLineSegmentOutputBranches_for_CPU(SDL::CPU::Event& event)
{
    // Did it match to track candidate?
    std::vector<int> sim_pLS_matched(trk.sim_pt().size());
    std::vector<vector<int>> sim_pLS_types(trk.sim_pt().size());

    // get layer ptrs
    std::vector<SDL::CPU::Layer*> layerPtrs;
    layerPtrs.push_back(&(event.getPixelLayer()));

    std::vector<int> pLS_isFake;
    std::vector<vector<int>> pLS_matched_simIdx;
    std::vector<float> pLS_pt;
    std::vector<float> pLS_eta;
    std::vector<float> pLS_phi;

    // Loop over layers and access track candidates
    for (auto& layerPtr : layerPtrs)
    {

        // Segments ptrs
        const std::vector<SDL::CPU::Segment*>& segmentPtrs = layerPtr->getSegmentPtrs();


        // Loop over trackCandidate ptrs
        for (auto& segmentPtr : segmentPtrs)
        {

            // hit idx
            std::vector<int> hit_idx;
            hit_idx.push_back(segmentPtr->innerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(segmentPtr->innerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(segmentPtr->outerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(segmentPtr->outerMiniDoubletPtr()->upperHitPtr()->idx());

            std::vector<int> hit_types;
            hit_types.push_back(0);
            hit_types.push_back(0);
            hit_types.push_back(0);
            hit_types.push_back(0);

            const SDL::CPU::Module& module0 = segmentPtr->innerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module2 = segmentPtr->outerMiniDoubletPtr()->lowerHitPtr()->getModule();

            bool isPixel0 = false;
            bool isPixel2 = false;

            int layer0 = module0.layer();
            int layer2 = module2.layer();

            int subdet0 = module0.subdet();
            int subdet2 = module2.subdet();

            int logicallayer0 = isPixel0 ? 0 : layer0  + 6 * (subdet0 == 4);
            int logicallayer2 = isPixel2 ? 0 : layer2  + 6 * (subdet2 == 4);

            int layer_binary = 0;
            layer_binary |= (1 << logicallayer0);
            layer_binary |= (1 << logicallayer2);

            int md_layer1 = isPixel0 ? 0 : layer0  + 6 * (subdet0 == 4) + 5 * (subdet0 == 4 and module0.moduleType() == SDL::CPU::Module::TwoS);
            int md_layer2 = isPixel2 ? 0 : layer2  + 6 * (subdet2 == 4) + 5 * (subdet2 == 4 and module2.moduleType() == SDL::CPU::Module::TwoS);

            // std::cout << " " << hit_idx[0] << " " << hit_idx[1] << " " << hit_idx[2] << " " << hit_idx[3] << " " << hit_idx[6] << " " << hit_idx[7] << std::endl;
            // std::cout << " " << hit_types[0] << " " << hit_types[1] << " " << hit_types[2] << " " << hit_types[3] << " " << hit_types[6] << " " << hit_types[7] << std::endl;

            // sim track matched index
            std::vector<int> matched_sim_trk_idxs = matchedSimTrkIdxs(hit_idx, hit_types);

            for (auto& isimtrk : matched_sim_trk_idxs)
            {
                sim_pLS_matched[isimtrk]++;
            }

            for (auto& isimtrk : matched_sim_trk_idxs)
            {
                sim_pLS_types[isimtrk].push_back(layer_binary);
            }

            // Compute pt, eta, phi of pLS
            // const float pt = segmentPtr->tlCand.getRecoVar("pt_beta");
            const float pt = segmentPtr->getRecoVar("ptIn");
            float eta = -999;
            float phi = -999;
            SDL::CPU::Hit hitA(trk.ph2_x()[hit_idx[0]], trk.ph2_y()[hit_idx[0]], trk.ph2_z()[hit_idx[0]]);
            SDL::CPU::Hit hitB(trk.ph2_x()[hit_idx[3]], trk.ph2_y()[hit_idx[3]], trk.ph2_z()[hit_idx[3]]);
            eta = hitB.eta();
            phi = hitA.phi();

            pLS_isFake.push_back(matched_sim_trk_idxs.size() == 0);
            pLS_pt.push_back(pt);
            pLS_eta.push_back(eta);
            pLS_phi.push_back(phi);
            pLS_matched_simIdx.push_back(matched_sim_trk_idxs);

        }

    }

    ana.tx->setBranch<vector<int>>("sim_pLS_matched", sim_pLS_matched);
    ana.tx->setBranch<vector<vector<int>>>("sim_pLS_types", sim_pLS_types);

    vector<int> pLS_isDuplicate(pLS_matched_simIdx.size());

    for (unsigned int i = 0; i < pLS_matched_simIdx.size(); ++i)
    {
        bool isDuplicate = false;
        for (unsigned int isim = 0; isim < pLS_matched_simIdx[i].size(); ++isim)
        {
            if (sim_pLS_matched[pLS_matched_simIdx[i][isim]] > 1)
            {
                isDuplicate = true;
            }
        }
        pLS_isDuplicate[i] = isDuplicate;
    }

    ana.tx->setBranch<vector<float>>("pLS_pt", pLS_pt);
    ana.tx->setBranch<vector<float>>("pLS_eta", pLS_eta);
    ana.tx->setBranch<vector<float>>("pLS_phi", pLS_phi);
    ana.tx->setBranch<vector<int>>("pLS_isFake", pLS_isFake);
    ana.tx->setBranch<vector<int>>("pLS_isDuplicate", pLS_isDuplicate);

}

void fillTripletOutputBranches(SDL::Event* event)
{

    SDL::triplets& tripletsInGPU = (*event->getTriplets());
    SDL::segments& segmentsInGPU = (*event->getSegments());
    SDL::miniDoublets& miniDoubletsInGPU = (*event->getMiniDoublets());
    SDL::hits& hitsInGPU = (*event->getHits());
    SDL::modules& modulesInGPU = (*event->getModules());

    // Did it match to track candidate?
    std::vector<int> sim_T3_matched(trk.sim_pt().size());
    std::vector<vector<int>> sim_T3_types(trk.sim_pt().size());
    std::vector<int> t3_isFake;
    std::vector<vector<int>> t3_matched_simIdx;
    std::vector<float> t3_pt;
    std::vector<float> t3_eta;
    std::vector<float> t3_phi;
#ifdef CUT_VALUE_DEBUG
    std::vector<float> t3_ZOut;
    std::vector<float> t3_RtOut;
    std::vector<float> t3_deltaPhiPos;
    std::vector<float> t3_deltaPhi;
    std::vector<float> t3_betaIn;
    std::vector<float> t3_betaOut;
    std::vector<float> t3_deltaBeta;
    std::vector<float> t3_ZLo;
    std::vector<float> t3_ZHi;
    std::vector<float> t3_RtLo;
    std::vector<float> t3_RtHi;
    std::vector<float> t3_kZ;
    std::vector<float> t3_ZLoPointed;
    std::vector<float> t3_ZHiPointed;
    std::vector<float> t3_sdlCut;
    std::vector<float> t3_betaInCut;
    std::vector<float> t3_betaOutCut;
    std::vector<float> t3_deltaBetaCut;
    std::vector<int> layer_binaries;
    std::vector<int> moduleType_binaries;
#endif

    const int MAX_NTRIPLET_PER_MODULE = 2500;
    for (unsigned int idx = 0; idx < *(modulesInGPU.nLowerModules); idx++) // "<=" because cheating to include pixel track candidate lower module
    {

        unsigned int nTriplets = tripletsInGPU.nTriplets[idx];

        for (unsigned int jdx = 0; jdx < nTriplets; jdx++)
        {
            unsigned int tripletIndex = MAX_NTRIPLET_PER_MODULE * idx + jdx; // this line causes the issue
            unsigned int innerSegmentIndex = -1;
            unsigned int outerSegmentIndex = -1;

            innerSegmentIndex = tripletsInGPU.segmentIndices[2 * tripletIndex];
            outerSegmentIndex = tripletsInGPU.segmentIndices[2 * tripletIndex + 1];
            float betaIn  = __H2F(tripletsInGPU.betaIn[tripletIndex]);
            float betaOut = __H2F(tripletsInGPU.betaOut[tripletIndex]);

            unsigned int innerSegmentInnerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * innerSegmentIndex];
            unsigned int innerSegmentOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * innerSegmentIndex + 1];
            unsigned int outerSegmentInnerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * outerSegmentIndex];
            unsigned int outerSegmentOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * outerSegmentIndex + 1];

            unsigned int innerSegmentInnerMiniDoubletLowerHitIndex = miniDoubletsInGPU.anchorHitIndices[innerSegmentInnerMiniDoubletIndex];
            unsigned int innerSegmentInnerMiniDoubletUpperHitIndex = miniDoubletsInGPU.outerHitIndices[innerSegmentInnerMiniDoubletIndex];
            unsigned int innerSegmentOuterMiniDoubletLowerHitIndex = miniDoubletsInGPU.anchorHitIndices[innerSegmentOuterMiniDoubletIndex];
            unsigned int innerSegmentOuterMiniDoubletUpperHitIndex = miniDoubletsInGPU.outerHitIndices[innerSegmentOuterMiniDoubletIndex];
            unsigned int outerSegmentInnerMiniDoubletLowerHitIndex = miniDoubletsInGPU.anchorHitIndices[outerSegmentInnerMiniDoubletIndex];
            unsigned int outerSegmentInnerMiniDoubletUpperHitIndex = miniDoubletsInGPU.outerHitIndices[outerSegmentInnerMiniDoubletIndex];
            unsigned int outerSegmentOuterMiniDoubletLowerHitIndex = miniDoubletsInGPU.anchorHitIndices[outerSegmentOuterMiniDoubletIndex];
            unsigned int outerSegmentOuterMiniDoubletUpperHitIndex = miniDoubletsInGPU.outerHitIndices[outerSegmentOuterMiniDoubletIndex];

            std::vector<int> hit_idxs = {
                (int) hitsInGPU.idxs[innerSegmentInnerMiniDoubletLowerHitIndex],
                (int) hitsInGPU.idxs[innerSegmentInnerMiniDoubletUpperHitIndex],
                (int) hitsInGPU.idxs[innerSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.idxs[innerSegmentOuterMiniDoubletUpperHitIndex],
                (int) hitsInGPU.idxs[outerSegmentInnerMiniDoubletLowerHitIndex],
                (int) hitsInGPU.idxs[outerSegmentInnerMiniDoubletUpperHitIndex],
                (int) hitsInGPU.idxs[outerSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.idxs[outerSegmentOuterMiniDoubletUpperHitIndex],
            };

            unsigned int iia_idx = segmentsInGPU.innerMiniDoubletAnchorHitIndices[innerSegmentIndex];
            unsigned int ooa_idx = segmentsInGPU.outerMiniDoubletAnchorHitIndices[outerSegmentIndex];

            const float dr = sqrt(pow(hitsInGPU.xs[iia_idx] - hitsInGPU.xs[ooa_idx], 2) + pow(hitsInGPU.ys[iia_idx] - hitsInGPU.ys[ooa_idx], 2));
            const float kRinv1GeVf = (2.99792458e-3 * 3.8);
            const float k2Rinv1GeVf = kRinv1GeVf / 2.;
            const float ptAv = dr * k2Rinv1GeVf / sin((betaIn + betaOut) / 2.);

            std::vector<int> hit_types;
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);

            std::vector<int> module_idxs = {
                (int) hitsInGPU.moduleIndices[innerSegmentInnerMiniDoubletLowerHitIndex],
                (int) hitsInGPU.moduleIndices[innerSegmentInnerMiniDoubletUpperHitIndex],
                (int) hitsInGPU.moduleIndices[innerSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.moduleIndices[innerSegmentOuterMiniDoubletUpperHitIndex],
                (int) hitsInGPU.moduleIndices[outerSegmentInnerMiniDoubletLowerHitIndex],
                (int) hitsInGPU.moduleIndices[outerSegmentInnerMiniDoubletUpperHitIndex],
                (int) hitsInGPU.moduleIndices[outerSegmentOuterMiniDoubletLowerHitIndex],
                (int) hitsInGPU.moduleIndices[outerSegmentOuterMiniDoubletUpperHitIndex],
            };

            int layer0 = modulesInGPU.layers[module_idxs[0]];
            int layer2 = modulesInGPU.layers[module_idxs[2]];
            int layer4 = modulesInGPU.layers[module_idxs[4]];
            int layer6 = modulesInGPU.layers[module_idxs[6]];

            int subdet0 = modulesInGPU.subdets[module_idxs[0]];
            int subdet2 = modulesInGPU.subdets[module_idxs[2]];
            int subdet4 = modulesInGPU.subdets[module_idxs[4]];
            int subdet6 = modulesInGPU.subdets[module_idxs[6]];

            int logicallayer0 = layer0 + 6 * (subdet0 == 4);
            int logicallayer2 = layer2 + 6 * (subdet2 == 4);
            int logicallayer4 = layer4 + 6 * (subdet4 == 4);
            int logicallayer6 = layer6 + 6 * (subdet6 == 4);

            int layer_binary = 0;
            layer_binary |= (1 << logicallayer0);
            layer_binary |= (1 << logicallayer2);
            layer_binary |= (1 << logicallayer4);
            layer_binary |= (1 << logicallayer6);

#ifdef CUT_VALUE_DEBUG
            int moduleType_binary = 0;
            int moduleType0 = modulesInGPU.moduleType[module_idxs[0]];
            int moduleType2 = modulesInGPU.moduleType[module_idxs[2]];
            int moduleType4 = modulesInGPU.moduleType[module_idxs[4]];
            int moduleType6 = modulesInGPU.moduleType[module_idxs[6]];
            
            moduleType_binary |= (moduleType0 << 0);
            moduleType_binary |= (moduleType2 << 2);
            moduleType_binary |= (moduleType4 << 4);
            moduleType_binary |= (moduleType6 << 6);
           
            layer_binaries.push_back(layer_binary);
            moduleType_binaries.push_back(moduleType_binary);

#endif

            // std::cout << " " << hit_idxs[0] << " " << hit_idxs[1] << " " << hit_idxs[2] << " " << hit_idxs[3] << " " << hit_idxs[6] << " " << hit_idxs[7] << std::endl;
            // std::cout << " " << hit_types[0] << " " << hit_types[1] << " " << hit_types[2] << " " << hit_types[3] << " " << hit_types[6] << " " << hit_types[7] << std::endl;

            // sim track matched index
            std::vector<int> matched_sim_trk_idxs = matchedSimTrkIdxs(hit_idxs, hit_types);

            for (auto &isimtrk : matched_sim_trk_idxs)
            {
                sim_T3_matched[isimtrk]++;
            }

            for (auto &isimtrk : matched_sim_trk_idxs)
            {
                sim_T3_types[isimtrk].push_back(layer_binary);
            }

            //radius computation from the three triplet MD anchor hits
            unsigned int innerTripletFirstSegmentAnchorHitIndex = segmentsInGPU.innerMiniDoubletAnchorHitIndices[innerSegmentIndex];
            unsigned int innerTripletSecondSegmentAnchorHitIndex = segmentsInGPU.outerMiniDoubletAnchorHitIndices[innerSegmentIndex]; //same as second segment inner MD anchorhit index
            unsigned int innerTripletThirdSegmentAnchorHitIndex = segmentsInGPU.outerMiniDoubletAnchorHitIndices[outerSegmentIndex]; //same as third segment inner MD anchor hit index

            float x1 = hitsInGPU.xs[innerTripletFirstSegmentAnchorHitIndex];
            float x2 = hitsInGPU.xs[innerTripletSecondSegmentAnchorHitIndex];
            float x3 = hitsInGPU.xs[innerTripletThirdSegmentAnchorHitIndex];

            float y1 = hitsInGPU.ys[innerTripletFirstSegmentAnchorHitIndex];
            float y2 = hitsInGPU.ys[innerTripletSecondSegmentAnchorHitIndex];
            float y3 = hitsInGPU.ys[innerTripletThirdSegmentAnchorHitIndex];

            float g, f; // not used
            float innerRadius = SDL::CPU::TrackCandidate::computeRadiusFromThreeAnchorHits(x1, y1, x2, y2, x3, y3, g, f);

            // Compute pt, eta, phi of T3
            const float pt = kRinv1GeVf * innerRadius;
            float eta = -999;
            float phi = -999;
            SDL::CPU::Hit hitA(trk.ph2_x()[hit_idxs[0]], trk.ph2_y()[hit_idxs[0]], trk.ph2_z()[hit_idxs[0]]);
            SDL::CPU::Hit hitB(trk.ph2_x()[hit_idxs[7]], trk.ph2_y()[hit_idxs[7]], trk.ph2_z()[hit_idxs[7]]);
            eta = hitB.eta();
            phi = hitA.phi();

            t3_isFake.push_back(matched_sim_trk_idxs.size() == 0);
            t3_pt.push_back(pt);
            t3_eta.push_back(eta);
            t3_phi.push_back(phi);
            t3_matched_simIdx.push_back(matched_sim_trk_idxs);

#ifdef CUT_VALUE_DEBUG
            float zOut = tripletsInGPU.zOut[tripletIndex];
            float rtOut = tripletsInGPU.rtOut[tripletIndex];
            float deltaPhiPos = tripletsInGPU.deltaPhiPos[tripletIndex];
            float deltaPhi = tripletsInGPU.deltaPhi[tripletIndex];
            //betaIn and betaOut already defined!
            float deltaBeta = betaIn - betaOut;
            float zLo = tripletsInGPU.zLo[tripletIndex];
            float zHi = tripletsInGPU.zHi[tripletIndex];
            float rtLo = tripletsInGPU.rtLo[tripletIndex];
            float rtHi = tripletsInGPU.rtHi[tripletIndex];
            float kZ = tripletsInGPU.kZ[tripletIndex];
            float zLoPointed = tripletsInGPU.zLoPointed[tripletIndex];
            float zHiPointed = tripletsInGPU.zHiPointed[tripletIndex];
            float sdlCut = tripletsInGPU.sdlCut[tripletIndex];
            float betaInCut =  tripletsInGPU.betaInCut[tripletIndex];
            float betaOutCut = tripletsInGPU.betaOutCut[tripletIndex];
            float deltaBetaCut = tripletsInGPU.deltaBetaCut[tripletIndex];

            t3_ZOut.push_back(zOut);
            t3_RtOut.push_back(rtOut);
            t3_deltaPhiPos.push_back(deltaPhiPos);
            t3_deltaPhi.push_back(deltaPhi);
            t3_betaIn.push_back(betaIn);
            t3_betaOut.push_back(betaOut);
            t3_deltaBeta.push_back(deltaBeta);
            t3_ZLo.push_back(zLo);
            t3_ZHi.push_back(zHi);
            t3_RtLo.push_back(rtLo);
            t3_RtHi.push_back(rtHi);
            t3_kZ.push_back(kZ);
            t3_ZLoPointed.push_back(zLoPointed);
            t3_ZHiPointed.push_back(zHiPointed);
            t3_sdlCut.push_back(sdlCut);
            t3_betaInCut.push_back(betaInCut);
            t3_betaOutCut.push_back(betaOutCut);
            t3_deltaBetaCut.push_back(deltaBetaCut);
#endif


        }

    }

    ana.tx->setBranch<vector<int>>("sim_T3_matched", sim_T3_matched);
    ana.tx->setBranch<vector<vector<int>>>("sim_T3_types", sim_T3_types);

    vector<int> t3_isDuplicate(t3_matched_simIdx.size());

    for (unsigned int i = 0; i < t3_matched_simIdx.size(); ++i)
    {
        bool isDuplicate = false;
        for (unsigned int isim = 0; isim < t3_matched_simIdx[i].size(); ++isim)
        {
            if (sim_T3_matched[t3_matched_simIdx[i][isim]] > 1)
            {
                isDuplicate = true;
            }
        }
        t3_isDuplicate[i] = isDuplicate;
    }

    ana.tx->setBranch<vector<float>>("t3_pt", t3_pt);
    ana.tx->setBranch<vector<float>>("t3_eta", t3_eta);
    ana.tx->setBranch<vector<float>>("t3_phi", t3_phi);
    ana.tx->setBranch<vector<int>>("t3_isFake", t3_isFake);
    ana.tx->setBranch<vector<int>>("t3_isDuplicate", t3_isDuplicate);

#ifdef CUT_VALUE_DEBUG
    ana.tx->setBranch<vector<float>>("t3_zOut",t3_ZOut);
    ana.tx->setBranch<vector<float>>("t3_rtOut",t3_RtOut);
    ana.tx->setBranch<vector<float>>("t3_deltaPhiPos",t3_deltaPhiPos);
    ana.tx->setBranch<vector<float>>("t3_deltaPhi",t3_deltaPhi);
    ana.tx->setBranch<vector<float>>("t3_betaIn",t3_betaIn);
    ana.tx->setBranch<vector<float>>("t3_betaOut",t3_betaOut);
    ana.tx->setBranch<vector<float>>("t3_deltaBeta",t3_deltaBeta);
    ana.tx->setBranch<vector<float>>("t3_zLo",t3_ZLo);
    ana.tx->setBranch<vector<float>>("t3_zHi",t3_ZHi);
    ana.tx->setBranch<vector<float>>("t3_rtLo",t3_RtLo);
    ana.tx->setBranch<vector<float>>("t3_rtHi",t3_RtHi);
    ana.tx->setBranch<vector<float>>("t3_kZ",t3_kZ);
    ana.tx->setBranch<vector<float>>("t3_zLoPointed",t3_ZLoPointed);
    ana.tx->setBranch<vector<float>>("t3_zHiPointed",t3_ZHiPointed);
    ana.tx->setBranch<vector<float>>("t3_sdlCut",t3_sdlCut);
    ana.tx->setBranch<vector<float>>("t3_betaInCut",t3_betaInCut);
    ana.tx->setBranch<vector<float>>("t3_betaOutCut",t3_betaOutCut);
    ana.tx->setBranch<vector<float>>("t3_deltaBetaCut",t3_deltaBetaCut);
    ana.tx->setBranch<vector<int>>("t3_layer_binary",layer_binaries);
    ana.tx->setBranch<vector<int>>("t3_moduleType_binary",moduleType_binaries);
#endif

}


//________________________________________________________________________________________________________________________________
void fillOutputBranches_for_CPU(SDL::CPU::Event& event)
{
    fillSimTrackOutputBranches();
    fillTrackCandidateOutputBranches_for_CPU(event);
    if (ana.do_lower_level)
    {
        fillLowerLevelOutputBranches_for_CPU(event);
    }

#ifdef PRIMITIVE_STUDY
    fillPrimitiveBranches_for_CPU(event);
#endif

    ana.tx->fill();
    ana.tx->clear();

}

//________________________________________________________________________________________________________________________________
void fillTrackCandidateOutputBranches_for_CPU(SDL::CPU::Event& event)
{
    // Did it match to track candidate?
    std::vector<int> sim_TC_matched(trk.sim_pt().size());
    std::vector<vector<int>> sim_TC_types(trk.sim_pt().size());

    // get layer ptrs
    std::vector<SDL::CPU::Layer*> layerPtrs = event.getLayerPtrs();
    layerPtrs.push_back(&(event.getPixelLayer()));

    std::vector<int> tc_isFake;
    std::vector<vector<int>> tc_matched_simIdx;
    std::vector<float> tc_pt;
    std::vector<float> tc_eta;
    std::vector<float> tc_phi;

    const float kRinv1GeVf = (2.99792458e-3 * 3.8);
    const float k2Rinv1GeVf = kRinv1GeVf / 2.;

    // Loop over layers and access track candidates
    for (auto& layerPtr : layerPtrs)
    {

        // Track Candidate ptrs
        const std::vector<SDL::CPU::TrackCandidate*>& trackCandidatePtrs = layerPtr->getTrackCandidatePtrs();


        // Loop over trackCandidate ptrs
        for (auto& trackCandidatePtr : trackCandidatePtrs)
        {

            // hit idx
            std::vector<int> hit_idx;
            hit_idx.push_back(trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->innerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx());

            std::vector<int> hit_types;
            if (trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().detId() == 1)
            {
                hit_types.push_back(0);
                hit_types.push_back(0);
                hit_types.push_back(0);
                hit_types.push_back(0);
            }
            else
            {
                hit_types.push_back(4);
                hit_types.push_back(4);
                hit_types.push_back(4);
                hit_types.push_back(4);
            }

            // If the inner segment and outer segment of the inner tracklet exactly the SAME pointer it means it's a pixel line segment in reality
            if (trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr() == trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr())
            {
                hit_types.push_back(0);
                hit_types.push_back(0);
                hit_types.push_back(0);
                hit_types.push_back(0);
            }
            else
            {
                hit_types.push_back(4);
                hit_types.push_back(4);
                hit_types.push_back(4);
                hit_types.push_back(4);
            }

            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);

            // 0 -- 0
            //      0 -- 0
            //
            //           0 -- 0
            //                0 -- 0
            // 01  23   
            //     45   67
            //          89   1011
            //               1213  1415

            // 0 -- 0
            //      0 -- 0
            //
            //           0 -- 0
            //                0 -- 0
            // 0    2   
            //      4    6
            //           8   10  
            //               1213  1415

            const SDL::CPU::Module& module0  = trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module2  = trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module4  = trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module6  = trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module8  = trackCandidatePtr->outerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module10 = trackCandidatePtr->outerTrackletBasePtr()->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module12 = trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module14 = trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule();

            bool isPixel0  = module0.isPixelLayerModule();
            bool isPixel2  = module2.isPixelLayerModule();
            bool isPixel4  = module4.isPixelLayerModule();
            bool isPixel6  = module6.isPixelLayerModule();
            bool isPixel8  = module8.isPixelLayerModule();
            bool isPixel10 = module10.isPixelLayerModule();
            bool isPixel12 = module12.isPixelLayerModule();
            bool isPixel14 = module14.isPixelLayerModule();

            int layer0  = module0.layer();
            int layer2  = module2.layer();
            int layer4  = module4.layer();
            int layer6  = module6.layer();
            int layer8  = module8.layer();
            int layer10 = module10.layer();
            int layer12 = module12.layer();
            int layer14 = module14.layer();

            int subdet0  = module0.subdet();
            int subdet2  = module2.subdet();
            int subdet4  = module4.subdet();
            int subdet6  = module6.subdet();
            int subdet8  = module8.subdet();
            int subdet10 = module10.subdet();
            int subdet12 = module12.subdet();
            int subdet14 = module14.subdet();

            int logicallayer0  = isPixel0  ? 0 : layer0  + 6 * (subdet0  == 4);
            int logicallayer2  = isPixel2  ? 0 : layer2  + 6 * (subdet2  == 4);
            int logicallayer4  = isPixel4  ? 0 : layer4  + 6 * (subdet4  == 4);
            int logicallayer6  = isPixel6  ? 0 : layer6  + 6 * (subdet6  == 4);
            int logicallayer8  = isPixel8  ? 0 : layer8  + 6 * (subdet8  == 4);
            int logicallayer10 = isPixel10 ? 0 : layer10 + 6 * (subdet10 == 4);
            int logicallayer12 = isPixel12 ? 0 : layer12 + 6 * (subdet12 == 4);
            int logicallayer14 = isPixel14 ? 0 : layer14 + 6 * (subdet14 == 4);

            int layer_binary = 0;
            layer_binary |= (1 << logicallayer0);
            layer_binary |= (1 << logicallayer2);
            layer_binary |= (1 << logicallayer4);
            layer_binary |= (1 << logicallayer6);
            layer_binary |= (1 << logicallayer8);
            layer_binary |= (1 << logicallayer10);
            layer_binary |= (1 << logicallayer12);
            layer_binary |= (1 << logicallayer14);

            bool isInnerTrackletTriplet = (hit_idx[2] == hit_idx[4] and hit_idx[3] == hit_idx[5] and hit_types[2] == hit_types[4] and hit_types[3] == hit_types[5]);
            bool isOuterTrackletTriplet = (hit_idx[10] == hit_idx[12] and hit_idx[11] == hit_idx[13] and hit_types[10] == hit_types[12] and hit_types[11] == hit_types[13]);
            bool isMiddleTrackletTriplet = (hit_idx[6] == hit_idx[8] and hit_idx[7] == hit_idx[9] and hit_types[6] == hit_types[8] and hit_types[7] == hit_types[9]);

            bool isInnerTrackletPixelLineSegment =
                (hit_idx[0] == hit_idx[4] and hit_idx[1] == hit_idx[5] and hit_types[0] == hit_types[4] and hit_types[1] == hit_types[5])
                and
                (hit_idx[2] == hit_idx[6] and hit_idx[3] == hit_idx[7] and hit_types[2] == hit_types[6] and hit_types[3] == hit_types[7]);

            bool isT5 = isInnerTrackletTriplet and isOuterTrackletTriplet and isMiddleTrackletTriplet;
            bool ispT3 = isInnerTrackletPixelLineSegment and isOuterTrackletTriplet;

            if (not (isT5 or ispT3))
                continue;

            // sim track matched index
            std::vector<int> matched_sim_trk_idxs = matchedSimTrkIdxs(hit_idx, hit_types);

            for (auto& isimtrk : matched_sim_trk_idxs)
            {
                sim_TC_matched[isimtrk]++;
            }

            for (auto& isimtrk : matched_sim_trk_idxs)
            {
                sim_TC_types[isimtrk].push_back(layer_binary);
            }

            // Compute pt, eta, phi of TC
            float eta = -999;
            float phi = -999;

            // // if (isInnerTrackletTriplet and isOuterTrackletTriplet)
            // if (true)
            // {
            //     std::cout << "here1" << std::endl;
            //     std::cout <<  " isInnerTrackletTriplet: " << isInnerTrackletTriplet <<  " isOuterTrackletTriplet: " << isOuterTrackletTriplet <<  std::endl;
            //     std::cout <<  " logicallayer0: " << logicallayer0 <<  " logicallayer2: " << logicallayer2 <<  " logicallayer4: " << logicallayer4 <<  " logicallayer6: " << logicallayer6 <<  " logicallayer8: " << logicallayer8 <<  " logicallayer10: " << logicallayer10 <<  std::endl;
            //     for (unsigned int ihit = 0; ihit < hit_idx.size(); ++ihit)
            //     {
            //         std::cout <<  " ihit: " << ihit <<  " hit_idx[ihit]: " << hit_idx[ihit] <<  std::endl;
            //     }
            //     for (unsigned int ihit = 0; ihit < hit_types.size(); ++ihit)
            //     {
            //         std::cout <<  " ihit: " << ihit <<  " hit_types[ihit]: " << hit_types[ihit] <<  std::endl;
            //     }
            //     std::cout << "recovar size: " << ((SDL::CPU::Triplet*) trackCandidatePtr->innerTrackletBasePtr())->tlCand.getRecoVars().size() << std::endl;
            //     for (auto& [k, v]: ((SDL::CPU::Triplet*) trackCandidatePtr->innerTrackletBasePtr())->tlCand.getRecoVars())
            //     {
            //         std::cout <<  " k: " << k <<  std::endl;
            //     }
            //     std::cout << "recovar size: " << ((SDL::CPU::Triplet*) trackCandidatePtr->outerTrackletBasePtr())->tlCand.getRecoVars().size() << std::endl;
            //     for (auto& [k, v]: ((SDL::CPU::Triplet*) trackCandidatePtr->outerTrackletBasePtr())->tlCand.getRecoVars())
            //     {
            //         std::cout <<  " k: " << k <<  std::endl;
            //     }
            // }
            // float pt_in  = isInnerTrackletTriplet ? ((SDL::CPU::Triplet*) trackCandidatePtr->innerTrackletBasePtr())->tlCand.getRecoVar("pt_beta") : trackCandidatePtr->innerTrackletBasePtr()->getRecoVar("pt_beta");
            // // std::cout <<  " pt_in: " << pt_in <<  std::endl;
            // float pt_out = isOuterTrackletTriplet ? ((SDL::CPU::Triplet*) trackCandidatePtr->outerTrackletBasePtr())->tlCand.getRecoVar("pt_beta") : trackCandidatePtr->outerTrackletBasePtr()->getRecoVar("pt_beta");
            // // std::cout <<  " pt_out: " << pt_out <<  std::endl;
            // float pt = (pt_in + pt_out) / 2.;
            // // std::cout << "here2" << std::endl;

            float pt = -999;

            if (isT5)
            {
                // std::cout << "here2" << std::endl;
                // for (unsigned int ihit = 0; ihit < hit_idx.size(); ++ihit)
                // {
                //     std::cout <<  " ihit: " << ihit <<  " hit_idx[ihit]: " << hit_idx[ihit] <<  std::endl;
                // }
                // for (unsigned int ihit = 0; ihit < hit_types.size(); ++ihit)
                // {
                //     std::cout <<  " ihit: " << ihit <<  " hit_types[ihit]: " << hit_types[ihit] <<  std::endl;
                // }
                // std::cout << "recovar size: " << ((SDL::CPU::Triplet*) trackCandidatePtr->innerTrackletBasePtr())->tlCand.getRecoVars().size() << std::endl;
                // for (auto& [k, v]: ((SDL::CPU::Triplet*) trackCandidatePtr->innerTrackletBasePtr())->tlCand.getRecoVars())
                // {
                //     std::cout <<  " k: " << k <<  std::endl;
                // }
                // std::cout << "recovar size: " << ((SDL::CPU::Triplet*) trackCandidatePtr->outerTrackletBasePtr())->tlCand.getRecoVars().size() << std::endl;
                // for (auto& [k, v]: ((SDL::CPU::Triplet*) trackCandidatePtr->outerTrackletBasePtr())->tlCand.getRecoVars())
                // {
                //     std::cout <<  " k: " << k <<  std::endl;
                // }
                // std::cout << "recovar size: " << ((SDL::CPU::Triplet*) trackCandidatePtr->outerTrackletBasePtr())->tlCand.getRecoVars().size() << std::endl;
                // for (auto& [k, v]: trackCandidatePtr->getRecoVars())
                // {
                //     std::cout <<  " k: " << k <<  std::endl;
                // }
                pt = k2Rinv1GeVf * (trackCandidatePtr->getRecoVar("innerRadius") + trackCandidatePtr->getRecoVar("outerRadius"));
            }
            else if (ispT3)
            {
                // std::cout << "here1" << std::endl;
                pt = k2Rinv1GeVf * (trackCandidatePtr->getRecoVar("innerRadius") + trackCandidatePtr->getRecoVar("outerRadius"));
            }
            else
            {
                pt = -999;
                continue;
            }

            // float ptBetaIn_in = isInnerTrackletTriplet ? ((SDL::CPU::Triplet*) trackCandidatePtr->innerTrackletBasePtr())->tlCand.getRecoVar("pt_betaIn") : trackCandidatePtr->innerTrackletBasePtr()->getRecoVar("pt_betaIn");
            // float ptBetaOut_in = isInnerTrackletTriplet ? ((SDL::CPU::Triplet*) trackCandidatePtr->innerTrackletBasePtr())->tlCand.getRecoVar("pt_betaOut") : trackCandidatePtr->innerTrackletBasePtr()->getRecoVar("pt_betaOut");

            // if ((trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().detId() == 1))
            //     std::cout << " " << (trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().detId() == 1) <<  " " << hit_idx[0] <<  " " << hit_idx[1] <<  " " << hit_idx[2] <<  " " << hit_idx[3] <<  " " << hit_idx[4] <<  " " << hit_idx[5] <<  " " << hit_idx[6] <<  " " << hit_idx[7] <<  " " << hit_idx[8] <<  " " << hit_idx[9] <<  " " << hit_idx[10] <<  " " << hit_idx[11] <<  " pt_in: " << pt_in <<  " pt_out: " << pt_out <<  " ptBetaIn_in: " << ptBetaIn_in <<  " ptBetaOut_in: " << ptBetaOut_in << std::endl;

            if (hit_types[0] == 4)
            {
                SDL::CPU::Hit hitA(trk.ph2_x()[hit_idx[0]], trk.ph2_y()[hit_idx[0]], trk.ph2_z()[hit_idx[0]]);
                SDL::CPU::Hit hitB(trk.ph2_x()[hit_idx[15]], trk.ph2_y()[hit_idx[15]], trk.ph2_z()[hit_idx[15]]);
                eta = hitB.eta();
                phi = hitA.phi();
            }
            else
            {
                SDL::CPU::Hit hitA(trk.pix_x()[hit_idx[0]], trk.pix_y()[hit_idx[0]], trk.pix_z()[hit_idx[0]]);
                SDL::CPU::Hit hitB(trk.ph2_x()[hit_idx[15]], trk.ph2_y()[hit_idx[15]], trk.ph2_z()[hit_idx[15]]);
                eta = hitB.eta();
                phi = hitA.phi();
            }

            tc_isFake.push_back(matched_sim_trk_idxs.size() == 0);
            tc_pt.push_back(pt);
            tc_eta.push_back(eta);
            tc_phi.push_back(phi);
            tc_matched_simIdx.push_back(matched_sim_trk_idxs);

        } // trackCandidatePtrs loop

    }


    ana.tx->setBranch<vector<int>>("sim_TC_matched", sim_TC_matched);
    ana.tx->setBranch<vector<vector<int>>>("sim_TC_types", sim_TC_types);

    vector<int> tc_isDuplicate(tc_matched_simIdx.size());

    for (unsigned int i = 0; i < tc_matched_simIdx.size(); ++i)
    {
        bool isDuplicate = false;
        for (unsigned int isim = 0; isim < tc_matched_simIdx[i].size(); ++isim)
        {
            if (sim_TC_matched[tc_matched_simIdx[i][isim]] > 1)
            {
                isDuplicate = true;
            }
        }
        tc_isDuplicate[i] = isDuplicate;
    }

    ana.tx->setBranch<vector<float>>("tc_pt", tc_pt);
    ana.tx->setBranch<vector<float>>("tc_eta", tc_eta);
    ana.tx->setBranch<vector<float>>("tc_phi", tc_phi);
    ana.tx->setBranch<vector<int>>("tc_isFake", tc_isFake);
    ana.tx->setBranch<vector<int>>("tc_isDuplicate", tc_isDuplicate);

}

//________________________________________________________________________________________________________________________________
void fillLowerLevelOutputBranches_for_CPU(SDL::CPU::Event& event)
{
    fillTripletOutputBranches_for_CPU(event);
    fillPixelLineSegmentOutputBranches_for_CPU(event);
    fillQuintupletOutputBranches_for_CPU(event);
    fillPixelTripletOutputBranches_for_CPU(event);

}

//________________________________________________________________________________________________________________________________
void fillTripletOutputBranches_for_CPU(SDL::CPU::Event& event)
{
    // Did it match to track candidate?
    std::vector<int> sim_T3_matched(trk.sim_pt().size());
    std::vector<vector<int>> sim_T3_types(trk.sim_pt().size());

    // get layer ptrs
    std::vector<SDL::CPU::Layer*> layerPtrs = event.getLayerPtrs();
    // layerPtrs.push_back(&(event.getPixelLayer()));

    std::vector<int> t3_isFake;
    std::vector<vector<int>> t3_matched_simIdx;
    std::vector<float> t3_pt;
    std::vector<float> t3_eta;
    std::vector<float> t3_phi;

#ifdef CUT_VALUE_DEBUG
    std::vector<float> residual;
    std::vector<int> layers1;
    std::vector<int> layers2;
    std::vector<int> layers3;
    std::vector<int> hit_idx1;
    std::vector<int> hit_idx2;
    std::vector<int> hit_idx3;
    std::vector<int> hit_idx4;
    std::vector<int> hit_idx5;
    std::vector<int> hit_idx6;
#endif

    const float kRinv1GeVf = (2.99792458e-3 * 3.8);

    // Loop over layers and access track candidates
    for (auto& layerPtr : layerPtrs)
    {

        // Triplets ptrs
        const std::vector<SDL::CPU::Triplet*>& tripletPtrs = layerPtr->getTripletPtrs();


        // Loop over trackCandidate ptrs
        for (auto& tripletPtr : tripletPtrs)
        {

            // hit idx
            std::vector<int> hit_idx;
            hit_idx.push_back(tripletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(tripletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(tripletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(tripletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(tripletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(tripletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(tripletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(tripletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx());

            std::vector<int> hit_types;
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);

            const SDL::CPU::Module& module0 = tripletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module2 = tripletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module4 = tripletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module6 = tripletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule();

            bool isPixel0 = false;
            bool isPixel2 = false;
            bool isPixel4 = false;
            bool isPixel6 = false;

            int layer0 = module0.layer();
            int layer2 = module2.layer();
            int layer4 = module4.layer();
            int layer6 = module6.layer();

            int subdet0 = module0.subdet();
            int subdet2 = module2.subdet();
            int subdet3 = module4.subdet();
            int subdet6 = module6.subdet();

            int logicallayer0 = isPixel0 ? 0 : layer0  + 6 * (subdet0 == 4);
            int logicallayer2 = isPixel2 ? 0 : layer2  + 6 * (subdet2 == 4);
            int logicallayer4 = isPixel4 ? 0 : layer4  + 6 * (subdet3 == 4);
            int logicallayer6 = isPixel6 ? 0 : layer6  + 6 * (subdet6 == 4);

            int layer_binary = 0;
            layer_binary |= (1 << logicallayer0);
            layer_binary |= (1 << logicallayer2);
            layer_binary |= (1 << logicallayer4);
            layer_binary |= (1 << logicallayer6);

            int md_layer1 = isPixel0 ? 0 : layer0  + 6 * (subdet0 == 4) + 5 * (subdet0 == 4 and module0.moduleType() == SDL::CPU::Module::TwoS);
            int md_layer2 = isPixel2 ? 0 : layer2  + 6 * (subdet2 == 4) + 5 * (subdet2 == 4 and module2.moduleType() == SDL::CPU::Module::TwoS);
            int md_layer3 = isPixel6 ? 0 : layer6  + 6 * (subdet6 == 4) + 5 * (subdet6 == 4 and module6.moduleType() == SDL::CPU::Module::TwoS);

            // std::cout << " " << hit_idx[0] << " " << hit_idx[1] << " " << hit_idx[2] << " " << hit_idx[3] << " " << hit_idx[6] << " " << hit_idx[7] << std::endl;
            // std::cout << " " << hit_types[0] << " " << hit_types[1] << " " << hit_types[2] << " " << hit_types[3] << " " << hit_types[6] << " " << hit_types[7] << std::endl;

            // sim track matched index
            std::vector<int> matched_sim_trk_idxs = matchedSimTrkIdxs(hit_idx, hit_types);

            for (auto& isimtrk : matched_sim_trk_idxs)
            {
                sim_T3_matched[isimtrk]++;
            }

            for (auto& isimtrk : matched_sim_trk_idxs)
            {
                sim_T3_types[isimtrk].push_back(layer_binary);
            }

            // Compute pt, eta, phi of T3
            // const float pt = tripletPtr->tlCand.getRecoVar("pt_beta");
            const float pt = kRinv1GeVf * tripletPtr->getRecoVar("tripletRadius");
            float eta = -999;
            float phi = -999;
            SDL::CPU::Hit hitA(trk.ph2_x()[hit_idx[0]], trk.ph2_y()[hit_idx[0]], trk.ph2_z()[hit_idx[0]]);
            SDL::CPU::Hit hitB(trk.ph2_x()[hit_idx[7]], trk.ph2_y()[hit_idx[7]], trk.ph2_z()[hit_idx[7]]);
            eta = hitB.eta();
            phi = hitA.phi();

            t3_isFake.push_back(matched_sim_trk_idxs.size() == 0);
            t3_pt.push_back(pt);
            t3_eta.push_back(eta);
            t3_phi.push_back(phi);
            t3_matched_simIdx.push_back(matched_sim_trk_idxs);
#ifdef CUT_VALUE_DEBUG
            residual.push_back(tripletPtr->getRecoVar("residual"));
            layers1.push_back(md_layer1);
            layers2.push_back(md_layer2);
            layers3.push_back(md_layer3);
            hit_idx1.push_back(hit_idx[0]);
            hit_idx2.push_back(hit_idx[1]);
            hit_idx3.push_back(hit_idx[2]);
            hit_idx4.push_back(hit_idx[3]);
            hit_idx5.push_back(hit_idx[6]);
            hit_idx6.push_back(hit_idx[7]);
#endif

        }

    }

    ana.tx->setBranch<vector<int>>("sim_T3_matched", sim_T3_matched);
    ana.tx->setBranch<vector<vector<int>>>("sim_T3_types", sim_T3_types);

    vector<int> t3_isDuplicate(t3_matched_simIdx.size());

    for (unsigned int i = 0; i < t3_matched_simIdx.size(); ++i)
    {
        bool isDuplicate = false;
        for (unsigned int isim = 0; isim < t3_matched_simIdx[i].size(); ++isim)
        {
            if (sim_T3_matched[t3_matched_simIdx[i][isim]] > 1)
            {
                isDuplicate = true;
            }
        }
        t3_isDuplicate[i] = isDuplicate;
    }

    ana.tx->setBranch<vector<float>>("t3_pt", t3_pt);
    ana.tx->setBranch<vector<float>>("t3_eta", t3_eta);
    ana.tx->setBranch<vector<float>>("t3_phi", t3_phi);
    ana.tx->setBranch<vector<int>>("t3_isFake", t3_isFake);
    ana.tx->setBranch<vector<int>>("t3_isDuplicate", t3_isDuplicate);
#ifdef CUT_VALUE_DEBUG
    ana.tx->setBranch<vector<float>>("t3_residual", residual);
    ana.tx->setBranch<vector<int>>("t3_layer1", layers1);
    ana.tx->setBranch<vector<int>>("t3_layer2", layers2);
    ana.tx->setBranch<vector<int>>("t3_layer3", layers3);
    ana.tx->setBranch<vector<int>>("t3_hit_idx1", hit_idx1);
    ana.tx->setBranch<vector<int>>("t3_hit_idx2", hit_idx2);
    ana.tx->setBranch<vector<int>>("t3_hit_idx3", hit_idx3);
    ana.tx->setBranch<vector<int>>("t3_hit_idx4", hit_idx4);
    ana.tx->setBranch<vector<int>>("t3_hit_idx5", hit_idx5);
    ana.tx->setBranch<vector<int>>("t3_hit_idx6", hit_idx6);
#endif

}

//________________________________________________________________________________________________________________________________
void fillQuintupletOutputBranches_for_CPU(SDL::CPU::Event& event)
{
    // Did it match to track candidate?
    std::vector<int> sim_T5_matched(trk.sim_pt().size());
    std::vector<vector<int>> sim_T5_types(trk.sim_pt().size());

    // get layer ptrs
    std::vector<SDL::CPU::Layer*> layerPtrs = event.getLayerPtrs();
    layerPtrs.push_back(&(event.getPixelLayer()));

    std::vector<int> t5_isFake;
    std::vector<vector<int>> t5_matched_simIdx;
    std::vector<float> t5_pt;
    std::vector<float> t5_eta;
    std::vector<float> t5_phi;
    std::vector<std::vector<float>> t5_simpt;

    const float kRinv1GeVf = (2.99792458e-3 * 3.8);
    const float k2Rinv1GeVf = kRinv1GeVf / 2.;

    // Loop over layers and access track candidates
    for (auto& layerPtr : layerPtrs)
    {

        // Track Candidate ptrs
        const std::vector<SDL::CPU::TrackCandidate*>& trackCandidatePtrs = layerPtr->getTrackCandidatePtrs();


        // Loop over trackCandidate ptrs
        for (auto& trackCandidatePtr : trackCandidatePtrs)
        {

            // hit idx
            std::vector<int> hit_idx;
            hit_idx.push_back(trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->innerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx());

            std::vector<int> hit_types;
            if (trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().detId() == 1)
            {
                hit_types.push_back(0);
                hit_types.push_back(0);
                hit_types.push_back(0);
                hit_types.push_back(0);
            }
            else
            {
                hit_types.push_back(4);
                hit_types.push_back(4);
                hit_types.push_back(4);
                hit_types.push_back(4);
            }

            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);

            // 0 -- 0
            //      0 -- 0
            //
            //           0 -- 0
            //                0 -- 0
            // 01  23   
            //     45   67
            //          89   1011
            //               1213  1415

            // 0 -- 0
            //      0 -- 0
            //
            //           0 -- 0
            //                0 -- 0
            // 0    2   
            //      4    6
            //           8   10  
            //               1213  1415

            bool isInnerTrackletTriplet = (hit_idx[2] == hit_idx[4] and hit_idx[3] == hit_idx[5] and hit_types[2] == hit_types[4] and hit_types[3] == hit_types[5]);
            bool isOuterTrackletTriplet = (hit_idx[10] == hit_idx[12] and hit_idx[11] == hit_idx[13] and hit_types[10] == hit_types[12] and hit_types[11] == hit_types[13]);
            bool isMiddleTrackletTriplet = (hit_idx[6] == hit_idx[8] and hit_idx[7] == hit_idx[9] and hit_types[6] == hit_types[8] and hit_types[7] == hit_types[9]);
            bool isT5 = isInnerTrackletTriplet and isOuterTrackletTriplet and isMiddleTrackletTriplet;

            if (not isT5)
                continue;

            const SDL::CPU::Module& module0  = trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module2  = trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module4  = trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module6  = trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module8  = trackCandidatePtr->outerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module10 = trackCandidatePtr->outerTrackletBasePtr()->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module12 = trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module14 = trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule();

            bool isPixel0  = module0.isPixelLayerModule();
            bool isPixel2  = module2.isPixelLayerModule();
            bool isPixel4  = module4.isPixelLayerModule();
            bool isPixel6  = module6.isPixelLayerModule();
            bool isPixel8  = module8.isPixelLayerModule();
            bool isPixel10 = module10.isPixelLayerModule();
            bool isPixel12 = module12.isPixelLayerModule();
            bool isPixel14 = module14.isPixelLayerModule();

            int layer0  = module0.layer();
            int layer2  = module2.layer();
            int layer4  = module4.layer();
            int layer6  = module6.layer();
            int layer8  = module8.layer();
            int layer10 = module10.layer();
            int layer12 = module12.layer();
            int layer14 = module14.layer();

            int subdet0  = module0.subdet();
            int subdet2  = module2.subdet();
            int subdet4  = module4.subdet();
            int subdet6  = module6.subdet();
            int subdet8  = module8.subdet();
            int subdet10 = module10.subdet();
            int subdet12 = module12.subdet();
            int subdet14 = module14.subdet();

            int logicallayer0  = isPixel0  ? 0 : layer0  + 6 * (subdet0  == 4);
            int logicallayer2  = isPixel2  ? 0 : layer2  + 6 * (subdet2  == 4);
            int logicallayer4  = isPixel4  ? 0 : layer4  + 6 * (subdet4  == 4);
            int logicallayer6  = isPixel6  ? 0 : layer6  + 6 * (subdet6  == 4);
            int logicallayer8  = isPixel8  ? 0 : layer8  + 6 * (subdet8  == 4);
            int logicallayer10 = isPixel10 ? 0 : layer10 + 6 * (subdet10 == 4);
            int logicallayer12 = isPixel12 ? 0 : layer12 + 6 * (subdet12 == 4);
            int logicallayer14 = isPixel14 ? 0 : layer14 + 6 * (subdet14 == 4);

            int layer_binary = 0;
            layer_binary |= (1 << logicallayer0);
            layer_binary |= (1 << logicallayer2);
            layer_binary |= (1 << logicallayer4);
            layer_binary |= (1 << logicallayer6);
            layer_binary |= (1 << logicallayer8);
            layer_binary |= (1 << logicallayer10);
            layer_binary |= (1 << logicallayer12);
            layer_binary |= (1 << logicallayer14);

            // sim track matched index
            std::vector<int> matched_sim_trk_idxs = matchedSimTrkIdxs(hit_idx, hit_types);

            for (auto& isimtrk : matched_sim_trk_idxs)
            {
                sim_T5_matched[isimtrk]++;
            }

            for (auto& isimtrk : matched_sim_trk_idxs)
            {
                sim_T5_types[isimtrk].push_back(layer_binary);
            }

            // Compute pt, eta, phi of T5
            float eta = -999;
            float phi = -999;

            // // if (isInnerTrackletTriplet and isOuterTrackletTriplet)
            // if (true)
            // {
            //     std::cout << "here1" << std::endl;
            //     std::cout <<  " isInnerTrackletTriplet: " << isInnerTrackletTriplet <<  " isOuterTrackletTriplet: " << isOuterTrackletTriplet <<  std::endl;
            //     std::cout <<  " logicallayer0: " << logicallayer0 <<  " logicallayer2: " << logicallayer2 <<  " logicallayer4: " << logicallayer4 <<  " logicallayer6: " << logicallayer6 <<  " logicallayer8: " << logicallayer8 <<  " logicallayer10: " << logicallayer10 <<  std::endl;
            //     for (unsigned int ihit = 0; ihit < hit_idx.size(); ++ihit)
            //     {
            //         std::cout <<  " ihit: " << ihit <<  " hit_idx[ihit]: " << hit_idx[ihit] <<  std::endl;
            //     }
            //     for (unsigned int ihit = 0; ihit < hit_types.size(); ++ihit)
            //     {
            //         std::cout <<  " ihit: " << ihit <<  " hit_types[ihit]: " << hit_types[ihit] <<  std::endl;
            //     }
            //     std::cout << "recovar size: " << ((SDL::CPU::Triplet*) trackCandidatePtr->innerTrackletBasePtr())->tlCand.getRecoVars().size() << std::endl;
            //     for (auto& [k, v]: ((SDL::CPU::Triplet*) trackCandidatePtr->innerTrackletBasePtr())->tlCand.getRecoVars())
            //     {
            //         std::cout <<  " k: " << k <<  std::endl;
            //     }
            //     std::cout << "recovar size: " << ((SDL::CPU::Triplet*) trackCandidatePtr->outerTrackletBasePtr())->tlCand.getRecoVars().size() << std::endl;
            //     for (auto& [k, v]: ((SDL::CPU::Triplet*) trackCandidatePtr->outerTrackletBasePtr())->tlCand.getRecoVars())
            //     {
            //         std::cout <<  " k: " << k <<  std::endl;
            //     }
            // }

            // Previous calculation method
            // float pt_in  = isInnerTrackletTriplet ? ((SDL::CPU::Triplet*) trackCandidatePtr->innerTrackletBasePtr())->tlCand.getRecoVar("pt_beta") : trackCandidatePtr->innerTrackletBasePtr()->getRecoVar("pt_beta");
            // std::cout <<  " pt_in: " << pt_in <<  std::endl;
            // float pt_out = isOuterTrackletTriplet ? ((SDL::CPU::Triplet*) trackCandidatePtr->outerTrackletBasePtr())->tlCand.getRecoVar("pt_beta") : trackCandidatePtr->outerTrackletBasePtr()->getRecoVar("pt_beta");
            // std::cout <<  " pt_out: " << pt_out <<  std::endl;
            // float pt = (pt_in + pt_out) / 2.;
            // std::cout << "here2" << std::endl;

            float pt = k2Rinv1GeVf * (trackCandidatePtr->getRecoVar("innerRadius") + trackCandidatePtr->getRecoVar("outerRadius"));

            // float ptBetaIn_in = isInnerTrackletTriplet ? ((SDL::CPU::Triplet*) trackCandidatePtr->innerTrackletBasePtr())->tlCand.getRecoVar("pt_betaIn") : trackCandidatePtr->innerTrackletBasePtr()->getRecoVar("pt_betaIn");
            // float ptBetaOut_in = isInnerTrackletTriplet ? ((SDL::CPU::Triplet*) trackCandidatePtr->innerTrackletBasePtr())->tlCand.getRecoVar("pt_betaOut") : trackCandidatePtr->innerTrackletBasePtr()->getRecoVar("pt_betaOut");

            // if ((trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().detId() == 1))
            //     std::cout << " " << (trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().detId() == 1) <<  " " << hit_idx[0] <<  " " << hit_idx[1] <<  " " << hit_idx[2] <<  " " << hit_idx[3] <<  " " << hit_idx[4] <<  " " << hit_idx[5] <<  " " << hit_idx[6] <<  " " << hit_idx[7] <<  " " << hit_idx[8] <<  " " << hit_idx[9] <<  " " << hit_idx[10] <<  " " << hit_idx[11] <<  " pt_in: " << pt_in <<  " pt_out: " << pt_out <<  " ptBetaIn_in: " << ptBetaIn_in <<  " ptBetaOut_in: " << ptBetaOut_in << std::endl;

            if (hit_types[0] == 4)
            {
                SDL::CPU::Hit hitA(trk.ph2_x()[hit_idx[0]], trk.ph2_y()[hit_idx[0]], trk.ph2_z()[hit_idx[0]]);
                SDL::CPU::Hit hitB(trk.ph2_x()[hit_idx[15]], trk.ph2_y()[hit_idx[15]], trk.ph2_z()[hit_idx[15]]);
                eta = hitB.eta();
                phi = hitA.phi();
            }
            else
            {
                SDL::CPU::Hit hitA(trk.pix_x()[hit_idx[0]], trk.pix_y()[hit_idx[0]], trk.pix_z()[hit_idx[0]]);
                SDL::CPU::Hit hitB(trk.ph2_x()[hit_idx[15]], trk.ph2_y()[hit_idx[15]], trk.ph2_z()[hit_idx[15]]);
                eta = hitB.eta();
                phi = hitA.phi();
            }

            t5_isFake.push_back(matched_sim_trk_idxs.size() == 0);
            t5_pt.push_back(pt);
            t5_eta.push_back(eta);
            t5_phi.push_back(phi);
            t5_matched_simIdx.push_back(matched_sim_trk_idxs);
            std::vector<float> sim_pt_per_t5;
            if(matched_sim_trk_idxs.size() == 0)
            {
                sim_pt_per_t5.push_back(-999);
            }
            else
            {
		sim_pt_per_t5.push_back(trk.sim_pt()[matched_sim_trk_idxs[0]]);
            }
            t5_simpt.push_back(sim_pt_per_t5);


        }

    }

    ana.tx->setBranch<vector<int>>("sim_T5_matched", sim_T5_matched);
    ana.tx->setBranch<vector<vector<int>>>("sim_T5_types", sim_T5_types);

    vector<int> t5_isDuplicate(t5_matched_simIdx.size());

    for (unsigned int i = 0; i < t5_matched_simIdx.size(); ++i)
    {
        bool isDuplicate = false;
        for (unsigned int isim = 0; isim < t5_matched_simIdx[i].size(); ++isim)
        {
            if (sim_T5_matched[t5_matched_simIdx[i][isim]] > 1)
            {
                isDuplicate = true;
            }
        }
        t5_isDuplicate[i] = isDuplicate;
    }

    ana.tx->setBranch<vector<float>>("t5_pt", t5_pt);
    ana.tx->setBranch<vector<float>>("t5_eta", t5_eta);
    ana.tx->setBranch<vector<float>>("t5_phi", t5_phi);
    ana.tx->setBranch<vector<int>>("t5_isFake", t5_isFake);
    ana.tx->setBranch<vector<int>>("t5_isDuplicate", t5_isDuplicate);
    ana.tx->setBranch<vector<vector<float>>>("t5_matched_pt", t5_simpt);

}

//________________________________________________________________________________________________________________________________
void fillPixelTripletOutputBranches_for_CPU(SDL::CPU::Event& event)
{
    // Did it match to track candidate?
    std::vector<int> sim_pT3_matched(trk.sim_pt().size());
    std::vector<vector<int>> sim_pT3_types(trk.sim_pt().size());

    // get layer ptrs
    std::vector<SDL::CPU::Layer*> layerPtrs;
    layerPtrs.push_back(&(event.getPixelLayer())); // Add only the pixel layers

    std::vector<int> pt3_isFake;
    std::vector<vector<int>> pt3_matched_simIdx;
    std::vector<float> pt3_pt;
    std::vector<float> pt3_eta;
    std::vector<float> pt3_phi;

#ifdef CUT_VALUE_DEBUG
    std::vector<int> pix_idx1;
    std::vector<int> pix_idx2;
    std::vector<int> pix_idx3;
    std::vector<int> pix_idx4;
    std::vector<int> hit_idx1;
    std::vector<int> hit_idx2;
    std::vector<int> hit_idx3;
    std::vector<int> hit_idx4;
    std::vector<int> hit_idx5;
    std::vector<int> hit_idx6;
#endif

    const float kRinv1GeVf = (2.99792458e-3 * 3.8);
    const float k2Rinv1GeVf = kRinv1GeVf / 2.;

    // Loop over layers and access track candidates
    for (auto& layerPtr : layerPtrs) // Only Pixel Layers will be looped over
    {

        // Track Candidate ptrs
        const std::vector<SDL::CPU::TrackCandidate*>& trackCandidatePtrs = layerPtr->getTrackCandidatePtrs();


        // Loop over trackCandidate ptrs (a pT3 is a track Candidate in CPU data format)
        for (auto& trackCandidatePtr : trackCandidatePtrs)
        {

            // hit idx
            std::vector<int> hit_idx;
            // for pT3 it will be a pixel "line segment" for the following 8 hits.
            // i.e. first four hits will be repeat for the latter four hits.
            hit_idx.push_back(trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx());
            // for pT3 it will be a triplet for the following 8 hits
            // i.e. the middle 4 hits will repeat, since the are a shared mini-doublet
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->innerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx());

            std::vector<int> hit_types;
            if (trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().detId() == 1)
            {
                hit_types.push_back(0);
                hit_types.push_back(0);
                hit_types.push_back(0);
                hit_types.push_back(0);
            }
            else
            {
                hit_types.push_back(4);
                hit_types.push_back(4);
                hit_types.push_back(4);
                hit_types.push_back(4);
            }

            // If the inner segment and outer segment of the inner tracklet exactly the SAME pointer it means it's a pixel line segment in reality
            if (trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr() == trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr())
            {
                hit_types.push_back(0);
                hit_types.push_back(0);
                hit_types.push_back(0);
                hit_types.push_back(0);
            }
            else
            {
                hit_types.push_back(4);
                hit_types.push_back(4);
                hit_types.push_back(4);
                hit_types.push_back(4);
            }

            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);

            // a pT3 looks like the following pictorially
            // 0 -- 0
            // 0 -- 0
            //
            //           0 -- 0
            //                0 -- 0
            // 01  23   
            // 45  67
            //          89   1011
            //               1213  1415

            // 0 -- 0
            // 0 -- 0
            //
            //           0 -- 0
            //                0 -- 0
            // 0    2   
            // 4    6
            //           8   10  
            //               1213  1415

            bool isInnerTrackletPixelLineSegment =
                (hit_idx[0] == hit_idx[4] and hit_idx[1] == hit_idx[5] and hit_types[0] == hit_types[4] and hit_types[1] == hit_types[5])
                and
                (hit_idx[2] == hit_idx[6] and hit_idx[3] == hit_idx[7] and hit_types[2] == hit_types[6] and hit_types[3] == hit_types[7]);
            bool isOuterTrackletTriplet = (hit_idx[10] == hit_idx[12] and hit_idx[11] == hit_idx[13] and hit_types[10] == hit_types[12] and hit_types[11] == hit_types[13]);
            bool ispT3 = isInnerTrackletPixelLineSegment and isOuterTrackletTriplet;

            if (not ispT3)
                continue;

            // if (hit_idx[10] == 231 and hit_idx[11] == 232)
            // {
            //     // unsigned int detid10 = trk.ph2_detId()[hit_idx[10]];
            //     // unsigned int detid11 = trk.ph2_detId()[hit_idx[11]];
            //     const SDL::CPU::Module& module10 = trackCandidatePtr->outerTrackletBasePtr()->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule();
            //     const SDL::CPU::Module& module14 = trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule();
            //     std::cout << module10;
            //     std::cout << module14;
            // }

            // std::cout << " " << hit_idx[0] << " " << hit_idx[1] << " " << hit_idx[2] << " " << hit_idx[3] << " " << hit_idx[8] << " " << hit_idx[9] << " " << hit_idx[10] << " " << hit_idx[11] << " " << hit_idx[14] << " " << hit_idx[15] << std::endl;
            // std::cout << " " << hit_types[0] << " " << hit_types[1] << " " << hit_types[2] << " " << hit_types[3] << " " << hit_types[8] << " " << hit_types[9] << " " << hit_types[10] << " " << hit_types[11] << " " << hit_types[14] << " " << hit_types[15] << std::endl;

            const SDL::CPU::Module& module0  = trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module2  = trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module4  = trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module6  = trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module8  = trackCandidatePtr->outerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module10 = trackCandidatePtr->outerTrackletBasePtr()->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module12 = trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module14 = trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule();

            bool isPixel0  = module0.isPixelLayerModule();
            bool isPixel2  = module2.isPixelLayerModule();
            bool isPixel4  = module4.isPixelLayerModule();
            bool isPixel6  = module6.isPixelLayerModule();
            bool isPixel8  = module8.isPixelLayerModule();
            bool isPixel10 = module10.isPixelLayerModule();
            bool isPixel12 = module12.isPixelLayerModule();
            bool isPixel14 = module14.isPixelLayerModule();

            int layer0  = module0.layer();
            int layer2  = module2.layer();
            int layer4  = module4.layer();
            int layer6  = module6.layer();
            int layer8  = module8.layer();
            int layer10 = module10.layer();
            int layer12 = module12.layer();
            int layer14 = module14.layer();

            int subdet0  = module0.subdet();
            int subdet2  = module2.subdet();
            int subdet4  = module4.subdet();
            int subdet6  = module6.subdet();
            int subdet8  = module8.subdet();
            int subdet10 = module10.subdet();
            int subdet12 = module12.subdet();
            int subdet14 = module14.subdet();

            int logicallayer0  = isPixel0  ? 0 : layer0  + 6 * (subdet0  == 4);
            int logicallayer2  = isPixel2  ? 0 : layer2  + 6 * (subdet2  == 4);
            int logicallayer4  = isPixel4  ? 0 : layer4  + 6 * (subdet4  == 4);
            int logicallayer6  = isPixel6  ? 0 : layer6  + 6 * (subdet6  == 4);
            int logicallayer8  = isPixel8  ? 0 : layer8  + 6 * (subdet8  == 4);
            int logicallayer10 = isPixel10 ? 0 : layer10 + 6 * (subdet10 == 4);
            int logicallayer12 = isPixel12 ? 0 : layer12 + 6 * (subdet12 == 4);
            int logicallayer14 = isPixel14 ? 0 : layer14 + 6 * (subdet14 == 4);

            int layer_binary = 0;
            layer_binary |= (1 << logicallayer0);
            layer_binary |= (1 << logicallayer2);
            layer_binary |= (1 << logicallayer4);
            layer_binary |= (1 << logicallayer6);
            layer_binary |= (1 << logicallayer8);
            layer_binary |= (1 << logicallayer10);
            layer_binary |= (1 << logicallayer12);
            layer_binary |= (1 << logicallayer14);

            // sim track matched index
            std::vector<int> matched_sim_trk_idxs = matchedSimTrkIdxs(hit_idx, hit_types);

            for (auto& isimtrk : matched_sim_trk_idxs)
            {
                sim_pT3_matched[isimtrk]++;
            }

            for (auto& isimtrk : matched_sim_trk_idxs)
            {
                sim_pT3_types[isimtrk].push_back(layer_binary);
            }

            // Compute pt, eta, phi of pT3
            float eta = -999;
            float phi = -999;

            // // if (isInnerTrackletTriplet and isOuterTrackletTriplet)
            // if (true)
            // {
            //     std::cout << "here1" << std::endl;
            //     std::cout <<  " isInnerTrackletTriplet: " << isInnerTrackletTriplet <<  " isOuterTrackletTriplet: " << isOuterTrackletTriplet <<  std::endl;
            //     std::cout <<  " logicallayer0: " << logicallayer0 <<  " logicallayer2: " << logicallayer2 <<  " logicallayer4: " << logicallayer4 <<  " logicallayer6: " << logicallayer6 <<  " logicallayer8: " << logicallayer8 <<  " logicallayer10: " << logicallayer10 <<  std::endl;
            //     for (unsigned int ihit = 0; ihit < hit_idx.size(); ++ihit)
            //     {
            //         std::cout <<  " ihit: " << ihit <<  " hit_idx[ihit]: " << hit_idx[ihit] <<  std::endl;
            //     }
            //     for (unsigned int ihit = 0; ihit < hit_types.size(); ++ihit)
            //     {
            //         std::cout <<  " ihit: " << ihit <<  " hit_types[ihit]: " << hit_types[ihit] <<  std::endl;
            //     }
            //     std::cout << "recovar size: " << ((SDL::CPU::Triplet*) trackCandidatePtr->innerTrackletBasePtr())->tlCand.getRecoVars().size() << std::endl;
            //     for (auto& [k, v]: ((SDL::CPU::Triplet*) trackCandidatePtr->innerTrackletBasePtr())->tlCand.getRecoVars())
            //     {
            //         std::cout <<  " k: " << k <<  std::endl;
            //     }
            //     std::cout << "recovar size: " << ((SDL::CPU::Triplet*) trackCandidatePtr->outerTrackletBasePtr())->tlCand.getRecoVars().size() << std::endl;
            //     for (auto& [k, v]: ((SDL::CPU::Triplet*) trackCandidatePtr->outerTrackletBasePtr())->tlCand.getRecoVars())
            //     {
            //         std::cout <<  " k: " << k <<  std::endl;
            //     }
            // }

            // Previous calculation method
            // float pt_in  = isInnerTrackletTriplet ? ((SDL::CPU::Triplet*) trackCandidatePtr->innerTrackletBasePtr())->tlCand.getRecoVar("pt_beta") : trackCandidatePtr->innerTrackletBasePtr()->getRecoVar("pt_beta");
            // std::cout <<  " pt_in: " << pt_in <<  std::endl;
            // float pt_out = isOuterTrackletTriplet ? ((SDL::CPU::Triplet*) trackCandidatePtr->outerTrackletBasePtr())->tlCand.getRecoVar("pt_beta") : trackCandidatePtr->outerTrackletBasePtr()->getRecoVar("pt_beta");
            // std::cout <<  " pt_out: " << pt_out <<  std::endl;
            // float pt = (pt_in + pt_out) / 2.;
            // std::cout << "here2" << std::endl;

            float pt = k2Rinv1GeVf * (trackCandidatePtr->getRecoVar("innerRadius") + trackCandidatePtr->getRecoVar("outerRadius"));

            // float ptBetaIn_in = isInnerTrackletTriplet ? ((SDL::CPU::Triplet*) trackCandidatePtr->innerTrackletBasePtr())->tlCand.getRecoVar("pt_betaIn") : trackCandidatePtr->innerTrackletBasePtr()->getRecoVar("pt_betaIn");
            // float ptBetaOut_in = isInnerTrackletTriplet ? ((SDL::CPU::Triplet*) trackCandidatePtr->innerTrackletBasePtr())->tlCand.getRecoVar("pt_betaOut") : trackCandidatePtr->innerTrackletBasePtr()->getRecoVar("pt_betaOut");

            // if ((trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().detId() == 1))
            //     std::cout << " " << (trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().detId() == 1) <<  " " << hit_idx[0] <<  " " << hit_idx[1] <<  " " << hit_idx[2] <<  " " << hit_idx[3] <<  " " << hit_idx[4] <<  " " << hit_idx[5] <<  " " << hit_idx[6] <<  " " << hit_idx[7] <<  " " << hit_idx[8] <<  " " << hit_idx[9] <<  " " << hit_idx[10] <<  " " << hit_idx[11] <<  " pt_in: " << pt_in <<  " pt_out: " << pt_out <<  " ptBetaIn_in: " << ptBetaIn_in <<  " ptBetaOut_in: " << ptBetaOut_in << std::endl;

            if (hit_types[0] == 4)
            {
                SDL::CPU::Hit hitA(trk.ph2_x()[hit_idx[0]], trk.ph2_y()[hit_idx[0]], trk.ph2_z()[hit_idx[0]]);
                SDL::CPU::Hit hitB(trk.ph2_x()[hit_idx[15]], trk.ph2_y()[hit_idx[15]], trk.ph2_z()[hit_idx[15]]);
                eta = hitB.eta();
                phi = hitA.phi();
            }
            else
            {
                SDL::CPU::Hit hitA(trk.pix_x()[hit_idx[0]], trk.pix_y()[hit_idx[0]], trk.pix_z()[hit_idx[0]]);
                SDL::CPU::Hit hitB(trk.ph2_x()[hit_idx[15]], trk.ph2_y()[hit_idx[15]], trk.ph2_z()[hit_idx[15]]);
                eta = hitB.eta();
                phi = hitA.phi();
            }

            pt3_isFake.push_back(matched_sim_trk_idxs.size() == 0);
            pt3_pt.push_back(pt);
            pt3_eta.push_back(eta);
            pt3_phi.push_back(phi);
            pt3_matched_simIdx.push_back(matched_sim_trk_idxs);

#ifdef CUT_VALUE_DEBUG
            pix_idx1.push_back(hit_idx[0]);
            pix_idx2.push_back(hit_idx[1]);
            pix_idx3.push_back(hit_idx[2]);
            pix_idx4.push_back(hit_idx[3]);
            hit_idx1.push_back(hit_idx[8]);
            hit_idx2.push_back(hit_idx[9]);
            hit_idx3.push_back(hit_idx[10]);
            hit_idx4.push_back(hit_idx[11]);
            hit_idx5.push_back(hit_idx[14]);
            hit_idx6.push_back(hit_idx[15]);
#endif
        }

    }

    ana.tx->setBranch<vector<int>>("sim_pT3_matched", sim_pT3_matched);
    ana.tx->setBranch<vector<vector<int>>>("sim_pT3_types", sim_pT3_types);

    vector<int> pt3_isDuplicate(pt3_matched_simIdx.size());

    for (unsigned int i = 0; i < pt3_matched_simIdx.size(); ++i)
    {
        bool isDuplicate = false;
        for (unsigned int isim = 0; isim < pt3_matched_simIdx[i].size(); ++isim)
        {
            if (sim_pT3_matched[pt3_matched_simIdx[i][isim]] > 1)
            {
                isDuplicate = true;
            }
        }
        pt3_isDuplicate[i] = isDuplicate;
    }

    ana.tx->setBranch<vector<float>>("pT3_pt", pt3_pt);
    ana.tx->setBranch<vector<float>>("pT3_eta", pt3_eta);
    ana.tx->setBranch<vector<float>>("pT3_phi", pt3_phi);
    ana.tx->setBranch<vector<int>>("pT3_isFake", pt3_isFake);
    ana.tx->setBranch<vector<int>>("pT3_isDuplicate", pt3_isDuplicate);
#ifdef CUT_VALUE_DEBUG
    ana.tx->setBranch<vector<int>>("pT3_pix_idx1", pix_idx1);
    ana.tx->setBranch<vector<int>>("pT3_pix_idx2", pix_idx2);
    ana.tx->setBranch<vector<int>>("pT3_pix_idx3", pix_idx3);
    ana.tx->setBranch<vector<int>>("pT3_pix_idx4", pix_idx4);
    ana.tx->setBranch<vector<int>>("pT3_hit_idx1", hit_idx1);
    ana.tx->setBranch<vector<int>>("pT3_hit_idx2", hit_idx2);
    ana.tx->setBranch<vector<int>>("pT3_hit_idx3", hit_idx3);
    ana.tx->setBranch<vector<int>>("pT3_hit_idx4", hit_idx4);
    ana.tx->setBranch<vector<int>>("pT3_hit_idx5", hit_idx5);
    ana.tx->setBranch<vector<int>>("pT3_hit_idx6", hit_idx6);
#endif

}

//________________________________________________________________________________________________________________________________
void fillPrimitiveBranches_for_CPU(SDL::CPU::Event& event)
{
    fillPrimitiveBranches_for_CPU_v2(event);
}

//________________________________________________________________________________________________________________________________
void fillPrimitiveBranches_for_CPU_v1(SDL::CPU::Event& event)
{
    SDL::CPU::Event simhit_event;
    addOuterTrackerSimHitsFromPVOnly(simhit_event);

    SDL::CPU::Event simhit_NotPVevent;
    addOuterTrackerSimHitsNotFromPVOnly(simhit_NotPVevent);

    std::vector<unsigned int> detids;

    for (auto& module : event.getLowerModulePtrs())
    {
        if (module->detId() == 1) // Pixel modules are skipped for now
            continue;
        const unsigned int& detid = module->moduleType() == SDL::CPU::Module::PS ? (module->moduleLayerType() == SDL::CPU::Module::Pixel ? module->detId() : module->partnerDetId()) : module->detId();
        if (std::find(detids.begin(), detids.end(), detid) == detids.end())
            detids.push_back(detid);
    }

    for (auto& module : simhit_event.getLowerModulePtrs())
    {
        const unsigned int& detid = module->moduleType() == SDL::CPU::Module::PS ? (module->moduleLayerType() == SDL::CPU::Module::Pixel ? module->detId() : module->partnerDetId()) : module->detId();
        if (std::find(detids.begin(), detids.end(), detid) == detids.end())
            detids.push_back(detid);
    }

    for (auto& module : simhit_NotPVevent.getLowerModulePtrs())
    {
        const unsigned int& detid = module->moduleType() == SDL::CPU::Module::PS ? (module->moduleLayerType() == SDL::CPU::Module::Pixel ? module->detId() : module->partnerDetId()) : module->detId();
        if (std::find(detids.begin(), detids.end(), detid) == detids.end())
            detids.push_back(detid);
    }
    
    for (auto& detid : detids)
    {

        const SDL::CPU::Module& module = event.getModule(detid);
        const SDL::CPU::Module& simhit_module = simhit_event.getModule(detid);
        const SDL::CPU::Module& simhit_NotPVmodule = simhit_NotPVevent.getModule(detid);
        const SDL::CPU::Module& partner_module = event.getModule(module.partnerDetId());
        const SDL::CPU::Module& partner_simhit_module = simhit_event.getModule(module.partnerDetId());
        const SDL::CPU::Module& partner_simhit_NotPVmodule = simhit_NotPVevent.getModule(module.partnerDetId());

        int layer = module.layer() + 6 * (module.subdet() == 4) + 5 * (module.subdet() == 4 and module.moduleType() == 1);

        ana.tx->pushbackToBranch<int>("prim_detid", detid);
        ana.tx->pushbackToBranch<int>("prim_layer", module.layer() + 6 * (module.subdet() == 4));
        ana.tx->pushbackToBranch<int>("prim_type", module.moduleType());
        ana.tx->pushbackToBranch<int>("prim_tilt", (module.subdet() == 5) and (module.side() != 3));
        ana.tx->pushbackToBranch<int>("prim_rod", module.rod());
        ana.tx->pushbackToBranch<int>("prim_ring", module.ring());
        ana.tx->pushbackToBranch<int>("prim_module", module.module());

        vector<int> prim_lower_nonpvsimhit_layer;
        vector<int> prim_lower_nonpvsimhit_sim_denom;
        vector<int> prim_lower_nonpvsimhit_sim_idx;
        vector<int> prim_lower_nonpvsimhit_sim_pdgid;
        vector<float> prim_lower_nonpvsimhit_sim_pt;
        vector<float> prim_lower_nonpvsimhit_sim_eta;
        vector<float> prim_lower_nonpvsimhit_sim_phi;
        vector<float> prim_lower_nonpvsimhit_sim_dxy;
        vector<float> prim_lower_nonpvsimhit_sim_dz;
        vector<float> prim_lower_nonpvsimhit_x;
        vector<float> prim_lower_nonpvsimhit_y;
        vector<float> prim_lower_nonpvsimhit_z;
        for (auto& hitPtr : simhit_NotPVmodule.getHitPtrs())
        {
            int idx = hitPtr->idx();
            int trkidx = trk.simhit_simTrkIdx()[idx];
            prim_lower_nonpvsimhit_layer.push_back(layer);
            prim_lower_nonpvsimhit_sim_denom.push_back(isDenomOfInterestSimTrk(trkidx));
            prim_lower_nonpvsimhit_sim_idx.push_back(trkidx);
            prim_lower_nonpvsimhit_sim_pdgid.push_back(trk.sim_pdgId()[trkidx]);
            prim_lower_nonpvsimhit_sim_pt.push_back(trk.sim_pt()[trkidx]);
            prim_lower_nonpvsimhit_sim_eta.push_back(trk.sim_eta()[trkidx]);
            prim_lower_nonpvsimhit_sim_phi.push_back(trk.sim_phi()[trkidx]);
            prim_lower_nonpvsimhit_sim_dxy.push_back(trk.sim_pca_dxy()[trkidx]);
            prim_lower_nonpvsimhit_sim_dz.push_back(trk.sim_pca_dz()[trkidx]);
            prim_lower_nonpvsimhit_x.push_back(hitPtr->x());
            prim_lower_nonpvsimhit_y.push_back(hitPtr->y());
            prim_lower_nonpvsimhit_z.push_back(hitPtr->z());
        }

        vector<int> prim_upper_nonpvsimhit_layer;
        vector<int> prim_upper_nonpvsimhit_sim_denom;
        vector<int> prim_upper_nonpvsimhit_sim_idx;
        vector<int> prim_upper_nonpvsimhit_sim_pdgid;
        vector<float> prim_upper_nonpvsimhit_sim_pt;
        vector<float> prim_upper_nonpvsimhit_sim_eta;
        vector<float> prim_upper_nonpvsimhit_sim_phi;
        vector<float> prim_upper_nonpvsimhit_sim_dxy;
        vector<float> prim_upper_nonpvsimhit_sim_dz;
        vector<float> prim_upper_nonpvsimhit_x;
        vector<float> prim_upper_nonpvsimhit_y;
        vector<float> prim_upper_nonpvsimhit_z;
        for (auto& hitPtr : partner_simhit_NotPVmodule.getHitPtrs())
        {
            int idx = hitPtr->idx();
            int trkidx = trk.simhit_simTrkIdx()[idx];
            prim_upper_nonpvsimhit_layer.push_back(layer);
            prim_upper_nonpvsimhit_sim_denom.push_back(isDenomOfInterestSimTrk(trkidx));
            prim_upper_nonpvsimhit_sim_idx.push_back(trkidx);
            prim_upper_nonpvsimhit_sim_pdgid.push_back(trk.sim_pdgId()[trkidx]);
            prim_upper_nonpvsimhit_sim_pt.push_back(trk.sim_pt()[trkidx]);
            prim_upper_nonpvsimhit_sim_eta.push_back(trk.sim_eta()[trkidx]);
            prim_upper_nonpvsimhit_sim_phi.push_back(trk.sim_phi()[trkidx]);
            prim_upper_nonpvsimhit_sim_dxy.push_back(trk.sim_pca_dxy()[trkidx]);
            prim_upper_nonpvsimhit_sim_dz.push_back(trk.sim_pca_dz()[trkidx]);
            prim_upper_nonpvsimhit_x.push_back(hitPtr->x());
            prim_upper_nonpvsimhit_y.push_back(hitPtr->y());
            prim_upper_nonpvsimhit_z.push_back(hitPtr->z());
        }

        vector<int> prim_lower_pvsimhit_layer;
        vector<int> prim_lower_pvsimhit_sim_denom;
        vector<int> prim_lower_pvsimhit_sim_idx;
        vector<int> prim_lower_pvsimhit_sim_pdgid;
        vector<float> prim_lower_pvsimhit_sim_pt;
        vector<float> prim_lower_pvsimhit_sim_eta;
        vector<float> prim_lower_pvsimhit_sim_phi;
        vector<float> prim_lower_pvsimhit_sim_dxy;
        vector<float> prim_lower_pvsimhit_sim_dz;
        vector<float> prim_lower_pvsimhit_x;
        vector<float> prim_lower_pvsimhit_y;
        vector<float> prim_lower_pvsimhit_z;
        for (auto& hitPtr : simhit_module.getHitPtrs())
        {
            int idx = hitPtr->idx();
            int trkidx = trk.simhit_simTrkIdx()[idx];
            prim_lower_pvsimhit_layer.push_back(layer);
            prim_lower_pvsimhit_sim_denom.push_back(isDenomOfInterestSimTrk(trkidx));
            prim_lower_pvsimhit_sim_idx.push_back(trkidx);
            prim_lower_pvsimhit_sim_pdgid.push_back(trk.sim_pdgId()[trkidx]);
            prim_lower_pvsimhit_sim_pt.push_back(trk.sim_pt()[trkidx]);
            prim_lower_pvsimhit_sim_eta.push_back(trk.sim_eta()[trkidx]);
            prim_lower_pvsimhit_sim_phi.push_back(trk.sim_phi()[trkidx]);
            prim_lower_pvsimhit_sim_dxy.push_back(trk.sim_pca_dxy()[trkidx]);
            prim_lower_pvsimhit_sim_dz.push_back(trk.sim_pca_dz()[trkidx]);
            prim_lower_pvsimhit_x.push_back(hitPtr->x());
            prim_lower_pvsimhit_y.push_back(hitPtr->y());
            prim_lower_pvsimhit_z.push_back(hitPtr->z());
        }

        vector<int> prim_upper_pvsimhit_layer;
        vector<int> prim_upper_pvsimhit_sim_denom;
        vector<int> prim_upper_pvsimhit_sim_idx;
        vector<int> prim_upper_pvsimhit_sim_pdgid;
        vector<float> prim_upper_pvsimhit_sim_pt;
        vector<float> prim_upper_pvsimhit_sim_eta;
        vector<float> prim_upper_pvsimhit_sim_phi;
        vector<float> prim_upper_pvsimhit_sim_dxy;
        vector<float> prim_upper_pvsimhit_sim_dz;
        vector<float> prim_upper_pvsimhit_x;
        vector<float> prim_upper_pvsimhit_y;
        vector<float> prim_upper_pvsimhit_z;
        for (auto& hitPtr : partner_simhit_module.getHitPtrs())
        {
            int idx = hitPtr->idx();
            int trkidx = trk.simhit_simTrkIdx()[idx];
            prim_upper_pvsimhit_layer.push_back(layer);
            prim_upper_pvsimhit_sim_denom.push_back(isDenomOfInterestSimTrk(trkidx));
            prim_upper_pvsimhit_sim_idx.push_back(trkidx);
            prim_upper_pvsimhit_sim_pdgid.push_back(trk.sim_pdgId()[trkidx]);
            prim_upper_pvsimhit_sim_pt.push_back(trk.sim_pt()[trkidx]);
            prim_upper_pvsimhit_sim_eta.push_back(trk.sim_eta()[trkidx]);
            prim_upper_pvsimhit_sim_phi.push_back(trk.sim_phi()[trkidx]);
            prim_upper_pvsimhit_sim_dxy.push_back(trk.sim_pca_dxy()[trkidx]);
            prim_upper_pvsimhit_sim_dz.push_back(trk.sim_pca_dz()[trkidx]);
            prim_upper_pvsimhit_x.push_back(hitPtr->x());
            prim_upper_pvsimhit_y.push_back(hitPtr->y());
            prim_upper_pvsimhit_z.push_back(hitPtr->z());
        }

        vector<int> prim_lower_recohit_layer;
        vector<int> prim_lower_recohit_sim_denom;
        vector<int> prim_lower_recohit_sim_idx;
        vector<int> prim_lower_recohit_sim_pdgid;
        vector<float> prim_lower_recohit_sim_pt;
        vector<float> prim_lower_recohit_sim_eta;
        vector<float> prim_lower_recohit_sim_phi;
        vector<float> prim_lower_recohit_sim_dxy;
        vector<float> prim_lower_recohit_sim_dz;
        vector<float> prim_lower_recohit_x;
        vector<float> prim_lower_recohit_y;
        vector<float> prim_lower_recohit_z;
        for (auto& hitPtr : module.getHitPtrs())
        {
            int hitidx = hitPtr->idx();
            bool has_simhit_matched = trk.ph2_simHitIdx()[hitidx].size() > 0;
            if (has_simhit_matched)
            {
                for (auto& idx : trk.ph2_simHitIdx()[hitidx])
                {
                    int trkidx = trk.simhit_simTrkIdx()[idx];
                    prim_lower_recohit_sim_denom.push_back(isDenomOfInterestSimTrk(trkidx));
                    prim_lower_recohit_sim_idx.push_back(trkidx);
                    prim_lower_recohit_sim_pdgid.push_back(trk.sim_pdgId()[trkidx]);
                    prim_lower_recohit_sim_pt.push_back(trk.sim_pt()[trkidx]);
                    prim_lower_recohit_sim_eta.push_back(trk.sim_eta()[trkidx]);
                    prim_lower_recohit_sim_phi.push_back(trk.sim_phi()[trkidx]);
                    prim_lower_recohit_sim_dxy.push_back(trk.sim_pca_dxy()[trkidx]);
                    prim_lower_recohit_sim_dz.push_back(trk.sim_pca_dz()[trkidx]);
                    break;
                }
            }
            else
            {
                prim_lower_recohit_sim_denom.push_back(-999);
                prim_lower_recohit_sim_idx.push_back(-999);
                prim_lower_recohit_sim_pdgid.push_back(-999);
                prim_lower_recohit_sim_pt.push_back(-999);
                prim_lower_recohit_sim_eta.push_back(-999);
                prim_lower_recohit_sim_phi.push_back(-999);
                prim_lower_recohit_sim_dxy.push_back(-999);
                prim_lower_recohit_sim_dz.push_back(-999);
            }
            prim_lower_recohit_layer.push_back(layer);
            prim_lower_recohit_x.push_back(hitPtr->x());
            prim_lower_recohit_y.push_back(hitPtr->y());
            prim_lower_recohit_z.push_back(hitPtr->z());
        }

        vector<int> prim_upper_recohit_layer;
        vector<int> prim_upper_recohit_sim_denom;
        vector<int> prim_upper_recohit_sim_idx;
        vector<int> prim_upper_recohit_sim_pdgid;
        vector<float> prim_upper_recohit_sim_pt;
        vector<float> prim_upper_recohit_sim_eta;
        vector<float> prim_upper_recohit_sim_phi;
        vector<float> prim_upper_recohit_sim_dxy;
        vector<float> prim_upper_recohit_sim_dz;
        vector<float> prim_upper_recohit_x;
        vector<float> prim_upper_recohit_y;
        vector<float> prim_upper_recohit_z;
        for (auto& hitPtr : partner_module.getHitPtrs())
        {
            int hitidx = hitPtr->idx();
            bool has_simhit_matched = trk.ph2_simHitIdx()[hitidx].size() > 0;
            if (has_simhit_matched)
            {
                for (auto& idx : trk.ph2_simHitIdx()[hitidx])
                {
                    int trkidx = trk.simhit_simTrkIdx()[idx];
                    prim_upper_recohit_sim_denom.push_back(isDenomOfInterestSimTrk(trkidx));
                    prim_upper_recohit_sim_idx.push_back(trkidx);
                    prim_upper_recohit_sim_pdgid.push_back(trk.sim_pdgId()[trkidx]);
                    prim_upper_recohit_sim_pt.push_back(trk.sim_pt()[trkidx]);
                    prim_upper_recohit_sim_eta.push_back(trk.sim_eta()[trkidx]);
                    prim_upper_recohit_sim_phi.push_back(trk.sim_phi()[trkidx]);
                    prim_upper_recohit_sim_dxy.push_back(trk.sim_pca_dxy()[trkidx]);
                    prim_upper_recohit_sim_dz.push_back(trk.sim_pca_dz()[trkidx]);
                    break;
                }
            }
            else
            {
                prim_upper_recohit_sim_denom.push_back(-999);
                prim_upper_recohit_sim_idx.push_back(-999);
                prim_upper_recohit_sim_pdgid.push_back(-999);
                prim_upper_recohit_sim_pt.push_back(-999);
                prim_upper_recohit_sim_eta.push_back(-999);
                prim_upper_recohit_sim_phi.push_back(-999);
                prim_upper_recohit_sim_dxy.push_back(-999);
                prim_upper_recohit_sim_dz.push_back(-999);
            }
            prim_upper_recohit_layer.push_back(layer);
            prim_upper_recohit_x.push_back(hitPtr->x());
            prim_upper_recohit_y.push_back(hitPtr->y());
            prim_upper_recohit_z.push_back(hitPtr->z());
        }

        vector<int> prim_lower_mdhit_layer;
        vector<int> prim_lower_mdhit_sim_denom;
        vector<int> prim_lower_mdhit_sim_idx;
        vector<int> prim_lower_mdhit_sim_pdgid;
        vector<float> prim_lower_mdhit_sim_pt;
        vector<float> prim_lower_mdhit_sim_eta;
        vector<float> prim_lower_mdhit_sim_phi;
        vector<float> prim_lower_mdhit_sim_dxy;
        vector<float> prim_lower_mdhit_sim_dz;
        vector<float> prim_lower_mdhit_x;
        vector<float> prim_lower_mdhit_y;
        vector<float> prim_lower_mdhit_z;
        vector<int> prim_upper_mdhit_layer;
        vector<int> prim_upper_mdhit_sim_denom;
        vector<int> prim_upper_mdhit_sim_idx;
        vector<int> prim_upper_mdhit_sim_pdgid;
        vector<float> prim_upper_mdhit_sim_pt;
        vector<float> prim_upper_mdhit_sim_eta;
        vector<float> prim_upper_mdhit_sim_phi;
        vector<float> prim_upper_mdhit_sim_dxy;
        vector<float> prim_upper_mdhit_sim_dz;
        vector<float> prim_upper_mdhit_x;
        vector<float> prim_upper_mdhit_y;
        vector<float> prim_upper_mdhit_z;
        for (auto& mdPtr : module.getMiniDoubletPtrs())
        {
            int hitidx = mdPtr->lowerHitPtr()->idx();
            bool has_simhit_matched = trk.ph2_simHitIdx()[hitidx].size() > 0;
            if (has_simhit_matched)
            {
                for (auto& idx : trk.ph2_simHitIdx()[hitidx])
                {
                    int trkidx = trk.simhit_simTrkIdx()[idx];
                    prim_lower_mdhit_sim_denom.push_back(isDenomOfInterestSimTrk(trkidx));
                    prim_lower_mdhit_sim_idx.push_back(trkidx);
                    prim_lower_mdhit_sim_pdgid.push_back(trk.sim_pdgId()[trkidx]);
                    prim_lower_mdhit_sim_pt.push_back(trk.sim_pt()[trkidx]);
                    prim_lower_mdhit_sim_eta.push_back(trk.sim_eta()[trkidx]);
                    prim_lower_mdhit_sim_phi.push_back(trk.sim_phi()[trkidx]);
                    prim_lower_mdhit_sim_dxy.push_back(trk.sim_pca_dxy()[trkidx]);
                    prim_lower_mdhit_sim_dz.push_back(trk.sim_pca_dz()[trkidx]);
                    break;
                }
            }
            else
            {
                prim_lower_mdhit_sim_denom.push_back(-999);
                prim_lower_mdhit_sim_idx.push_back(-999);
                prim_lower_mdhit_sim_pdgid.push_back(-999);
                prim_lower_mdhit_sim_pt.push_back(-999);
                prim_lower_mdhit_sim_eta.push_back(-999);
                prim_lower_mdhit_sim_phi.push_back(-999);
                prim_lower_mdhit_sim_dxy.push_back(-999);
                prim_lower_mdhit_sim_dz.push_back(-999);
            }
            prim_lower_mdhit_layer.push_back(layer);
            prim_lower_mdhit_x.push_back(mdPtr->lowerHitPtr()->x());
            prim_lower_mdhit_y.push_back(mdPtr->lowerHitPtr()->y());
            prim_lower_mdhit_z.push_back(mdPtr->lowerHitPtr()->z());

            hitidx = mdPtr->upperHitPtr()->idx();
            has_simhit_matched = trk.ph2_simHitIdx()[hitidx].size() > 0;
            if (has_simhit_matched)
            {
                for (auto& idx : trk.ph2_simHitIdx()[hitidx])
                {
                    int trkidx = trk.simhit_simTrkIdx()[idx];
                    prim_upper_mdhit_sim_denom.push_back(isDenomOfInterestSimTrk(trkidx));
                    prim_upper_mdhit_sim_idx.push_back(trkidx);
                    prim_upper_mdhit_sim_pdgid.push_back(trk.sim_pdgId()[trkidx]);
                    prim_upper_mdhit_sim_pt.push_back(trk.sim_pt()[trkidx]);
                    prim_upper_mdhit_sim_eta.push_back(trk.sim_eta()[trkidx]);
                    prim_upper_mdhit_sim_phi.push_back(trk.sim_phi()[trkidx]);
                    prim_upper_mdhit_sim_dxy.push_back(trk.sim_pca_dxy()[trkidx]);
                    prim_upper_mdhit_sim_dz.push_back(trk.sim_pca_dz()[trkidx]);
                    break;
                }
            }
            else
            {
                prim_upper_mdhit_sim_denom.push_back(-999);
                prim_upper_mdhit_sim_idx.push_back(-999);
                prim_upper_mdhit_sim_pdgid.push_back(-999);
                prim_upper_mdhit_sim_pt.push_back(-999);
                prim_upper_mdhit_sim_eta.push_back(-999);
                prim_upper_mdhit_sim_phi.push_back(-999);
                prim_upper_mdhit_sim_dxy.push_back(-999);
                prim_upper_mdhit_sim_dz.push_back(-999);
            }
            prim_upper_mdhit_layer.push_back(layer);
            prim_upper_mdhit_x.push_back(mdPtr->upperHitPtr()->x());
            prim_upper_mdhit_y.push_back(mdPtr->upperHitPtr()->y());
            prim_upper_mdhit_z.push_back(mdPtr->upperHitPtr()->z());
        }

        ana.tx->pushbackToBranch<vector<int>>("prim_lower_nonpvsimhit_layer", prim_lower_nonpvsimhit_layer);
        ana.tx->pushbackToBranch<vector<int>>("prim_lower_nonpvsimhit_sim_denom", prim_lower_nonpvsimhit_sim_denom);
        ana.tx->pushbackToBranch<vector<int>>("prim_lower_nonpvsimhit_sim_idx", prim_lower_nonpvsimhit_sim_idx);
        ana.tx->pushbackToBranch<vector<int>>("prim_lower_nonpvsimhit_sim_pdgid", prim_lower_nonpvsimhit_sim_pdgid);
        ana.tx->pushbackToBranch<vector<float>>("prim_lower_nonpvsimhit_sim_pt", prim_lower_nonpvsimhit_sim_pt);
        ana.tx->pushbackToBranch<vector<float>>("prim_lower_nonpvsimhit_sim_eta", prim_lower_nonpvsimhit_sim_eta);
        ana.tx->pushbackToBranch<vector<float>>("prim_lower_nonpvsimhit_sim_phi", prim_lower_nonpvsimhit_sim_phi);
        ana.tx->pushbackToBranch<vector<float>>("prim_lower_nonpvsimhit_sim_dxy", prim_lower_nonpvsimhit_sim_dxy);
        ana.tx->pushbackToBranch<vector<float>>("prim_lower_nonpvsimhit_sim_dz", prim_lower_nonpvsimhit_sim_dz);
        ana.tx->pushbackToBranch<vector<float>>("prim_lower_nonpvsimhit_x", prim_lower_nonpvsimhit_x);
        ana.tx->pushbackToBranch<vector<float>>("prim_lower_nonpvsimhit_y", prim_lower_nonpvsimhit_y);
        ana.tx->pushbackToBranch<vector<float>>("prim_lower_nonpvsimhit_z", prim_lower_nonpvsimhit_z);

        ana.tx->pushbackToBranch<vector<int>>("prim_upper_nonpvsimhit_layer", prim_upper_nonpvsimhit_layer);
        ana.tx->pushbackToBranch<vector<int>>("prim_upper_nonpvsimhit_sim_denom", prim_upper_nonpvsimhit_sim_denom);
        ana.tx->pushbackToBranch<vector<int>>("prim_upper_nonpvsimhit_sim_idx", prim_upper_nonpvsimhit_sim_idx);
        ana.tx->pushbackToBranch<vector<int>>("prim_upper_nonpvsimhit_sim_pdgid", prim_upper_nonpvsimhit_sim_pdgid);
        ana.tx->pushbackToBranch<vector<float>>("prim_upper_nonpvsimhit_sim_pt", prim_upper_nonpvsimhit_sim_pt);
        ana.tx->pushbackToBranch<vector<float>>("prim_upper_nonpvsimhit_sim_eta", prim_upper_nonpvsimhit_sim_eta);
        ana.tx->pushbackToBranch<vector<float>>("prim_upper_nonpvsimhit_sim_phi", prim_upper_nonpvsimhit_sim_phi);
        ana.tx->pushbackToBranch<vector<float>>("prim_upper_nonpvsimhit_sim_dxy", prim_upper_nonpvsimhit_sim_dxy);
        ana.tx->pushbackToBranch<vector<float>>("prim_upper_nonpvsimhit_sim_dz", prim_upper_nonpvsimhit_sim_dz);
        ana.tx->pushbackToBranch<vector<float>>("prim_upper_nonpvsimhit_x", prim_upper_nonpvsimhit_x);
        ana.tx->pushbackToBranch<vector<float>>("prim_upper_nonpvsimhit_y", prim_upper_nonpvsimhit_y);
        ana.tx->pushbackToBranch<vector<float>>("prim_upper_nonpvsimhit_z", prim_upper_nonpvsimhit_z);

        ana.tx->pushbackToBranch<vector<int>>("prim_lower_pvsimhit_layer", prim_lower_pvsimhit_layer);
        ana.tx->pushbackToBranch<vector<int>>("prim_lower_pvsimhit_sim_denom", prim_lower_pvsimhit_sim_denom);
        ana.tx->pushbackToBranch<vector<int>>("prim_lower_pvsimhit_sim_idx", prim_lower_pvsimhit_sim_idx);
        ana.tx->pushbackToBranch<vector<int>>("prim_lower_pvsimhit_sim_pdgid", prim_lower_pvsimhit_sim_pdgid);
        ana.tx->pushbackToBranch<vector<float>>("prim_lower_pvsimhit_sim_pt", prim_lower_pvsimhit_sim_pt);
        ana.tx->pushbackToBranch<vector<float>>("prim_lower_pvsimhit_sim_eta", prim_lower_pvsimhit_sim_eta);
        ana.tx->pushbackToBranch<vector<float>>("prim_lower_pvsimhit_sim_phi", prim_lower_pvsimhit_sim_phi);
        ana.tx->pushbackToBranch<vector<float>>("prim_lower_pvsimhit_sim_dxy", prim_lower_pvsimhit_sim_dxy);
        ana.tx->pushbackToBranch<vector<float>>("prim_lower_pvsimhit_sim_dz", prim_lower_pvsimhit_sim_dz);
        ana.tx->pushbackToBranch<vector<float>>("prim_lower_pvsimhit_x", prim_lower_pvsimhit_x);
        ana.tx->pushbackToBranch<vector<float>>("prim_lower_pvsimhit_y", prim_lower_pvsimhit_y);
        ana.tx->pushbackToBranch<vector<float>>("prim_lower_pvsimhit_z", prim_lower_pvsimhit_z);

        ana.tx->pushbackToBranch<vector<int>>("prim_upper_pvsimhit_layer", prim_upper_pvsimhit_layer);
        ana.tx->pushbackToBranch<vector<int>>("prim_upper_pvsimhit_sim_denom", prim_upper_pvsimhit_sim_denom);
        ana.tx->pushbackToBranch<vector<int>>("prim_upper_pvsimhit_sim_idx", prim_upper_pvsimhit_sim_idx);
        ana.tx->pushbackToBranch<vector<int>>("prim_upper_pvsimhit_sim_pdgid", prim_upper_pvsimhit_sim_pdgid);
        ana.tx->pushbackToBranch<vector<float>>("prim_upper_pvsimhit_sim_pt", prim_upper_pvsimhit_sim_pt);
        ana.tx->pushbackToBranch<vector<float>>("prim_upper_pvsimhit_sim_eta", prim_upper_pvsimhit_sim_eta);
        ana.tx->pushbackToBranch<vector<float>>("prim_upper_pvsimhit_sim_phi", prim_upper_pvsimhit_sim_phi);
        ana.tx->pushbackToBranch<vector<float>>("prim_upper_pvsimhit_sim_dxy", prim_upper_pvsimhit_sim_dxy);
        ana.tx->pushbackToBranch<vector<float>>("prim_upper_pvsimhit_sim_dz", prim_upper_pvsimhit_sim_dz);
        ana.tx->pushbackToBranch<vector<float>>("prim_upper_pvsimhit_x", prim_upper_pvsimhit_x);
        ana.tx->pushbackToBranch<vector<float>>("prim_upper_pvsimhit_y", prim_upper_pvsimhit_y);
        ana.tx->pushbackToBranch<vector<float>>("prim_upper_pvsimhit_z", prim_upper_pvsimhit_z);

        ana.tx->pushbackToBranch<vector<int>>("prim_lower_recohit_layer", prim_lower_recohit_layer); // first match simhit -> sim trk
        ana.tx->pushbackToBranch<vector<int>>("prim_lower_recohit_sim_denom", prim_lower_recohit_sim_denom); // first match simhit -> sim trk
        ana.tx->pushbackToBranch<vector<int>>("prim_lower_recohit_sim_idx", prim_lower_recohit_sim_idx); // first match simhit -> sim trk
        ana.tx->pushbackToBranch<vector<int>>("prim_lower_recohit_sim_pdgid", prim_lower_recohit_sim_pdgid);
        ana.tx->pushbackToBranch<vector<float>>("prim_lower_recohit_sim_pt", prim_lower_recohit_sim_pt);
        ana.tx->pushbackToBranch<vector<float>>("prim_lower_recohit_sim_eta", prim_lower_recohit_sim_eta);
        ana.tx->pushbackToBranch<vector<float>>("prim_lower_recohit_sim_phi", prim_lower_recohit_sim_phi);
        ana.tx->pushbackToBranch<vector<float>>("prim_lower_recohit_sim_dxy", prim_lower_recohit_sim_dxy);
        ana.tx->pushbackToBranch<vector<float>>("prim_lower_recohit_sim_dz", prim_lower_recohit_sim_dz);
        ana.tx->pushbackToBranch<vector<float>>("prim_lower_recohit_x", prim_lower_recohit_x);
        ana.tx->pushbackToBranch<vector<float>>("prim_lower_recohit_y", prim_lower_recohit_y);
        ana.tx->pushbackToBranch<vector<float>>("prim_lower_recohit_z", prim_lower_recohit_z);

        ana.tx->pushbackToBranch<vector<int>>("prim_upper_recohit_layer", prim_upper_recohit_layer); // first match simhit -> sim trk
        ana.tx->pushbackToBranch<vector<int>>("prim_upper_recohit_sim_denom", prim_upper_recohit_sim_denom); // first match simhit -> sim trk
        ana.tx->pushbackToBranch<vector<int>>("prim_upper_recohit_sim_idx", prim_upper_recohit_sim_idx); // first match simhit -> sim trk
        ana.tx->pushbackToBranch<vector<int>>("prim_upper_recohit_sim_pdgid", prim_upper_recohit_sim_pdgid);
        ana.tx->pushbackToBranch<vector<float>>("prim_upper_recohit_sim_pt", prim_upper_recohit_sim_pt);
        ana.tx->pushbackToBranch<vector<float>>("prim_upper_recohit_sim_eta", prim_upper_recohit_sim_eta);
        ana.tx->pushbackToBranch<vector<float>>("prim_upper_recohit_sim_phi", prim_upper_recohit_sim_phi);
        ana.tx->pushbackToBranch<vector<float>>("prim_upper_recohit_sim_dxy", prim_upper_recohit_sim_dxy);
        ana.tx->pushbackToBranch<vector<float>>("prim_upper_recohit_sim_dz", prim_upper_recohit_sim_dz);
        ana.tx->pushbackToBranch<vector<float>>("prim_upper_recohit_x", prim_upper_recohit_x);
        ana.tx->pushbackToBranch<vector<float>>("prim_upper_recohit_y", prim_upper_recohit_y);
        ana.tx->pushbackToBranch<vector<float>>("prim_upper_recohit_z", prim_upper_recohit_z);

        ana.tx->pushbackToBranch<vector<int>>("prim_lower_mdhit_layer", prim_lower_mdhit_layer); // first match simhit -> sim trk
        ana.tx->pushbackToBranch<vector<int>>("prim_lower_mdhit_sim_denom", prim_lower_mdhit_sim_denom); // first match simhit -> sim trk
        ana.tx->pushbackToBranch<vector<int>>("prim_lower_mdhit_sim_idx", prim_lower_mdhit_sim_idx); // first match simhit -> sim trk
        ana.tx->pushbackToBranch<vector<int>>("prim_lower_mdhit_sim_pdgid", prim_lower_mdhit_sim_pdgid);
        ana.tx->pushbackToBranch<vector<float>>("prim_lower_mdhit_sim_pt", prim_lower_mdhit_sim_pt);
        ana.tx->pushbackToBranch<vector<float>>("prim_lower_mdhit_sim_eta", prim_lower_mdhit_sim_eta);
        ana.tx->pushbackToBranch<vector<float>>("prim_lower_mdhit_sim_phi", prim_lower_mdhit_sim_phi);
        ana.tx->pushbackToBranch<vector<float>>("prim_lower_mdhit_sim_dxy", prim_lower_mdhit_sim_dxy);
        ana.tx->pushbackToBranch<vector<float>>("prim_lower_mdhit_sim_dz", prim_lower_mdhit_sim_dz);
        ana.tx->pushbackToBranch<vector<float>>("prim_lower_mdhit_x", prim_lower_mdhit_x); // paired with upper
        ana.tx->pushbackToBranch<vector<float>>("prim_lower_mdhit_y", prim_lower_mdhit_y); // paired with upper
        ana.tx->pushbackToBranch<vector<float>>("prim_lower_mdhit_z", prim_lower_mdhit_z); // paired with upper

        ana.tx->pushbackToBranch<vector<int>>("prim_upper_mdhit_layer", prim_upper_mdhit_layer); // first match simhit -> sim trk
        ana.tx->pushbackToBranch<vector<int>>("prim_upper_mdhit_sim_denom", prim_upper_mdhit_sim_denom); // first match simhit -> sim trk
        ana.tx->pushbackToBranch<vector<int>>("prim_upper_mdhit_sim_idx", prim_upper_mdhit_sim_idx); // first match simhit -> sim trk
        ana.tx->pushbackToBranch<vector<int>>("prim_upper_mdhit_sim_pdgid", prim_upper_mdhit_sim_pdgid);
        ana.tx->pushbackToBranch<vector<float>>("prim_upper_mdhit_sim_pt", prim_upper_mdhit_sim_pt);
        ana.tx->pushbackToBranch<vector<float>>("prim_upper_mdhit_sim_eta", prim_upper_mdhit_sim_eta);
        ana.tx->pushbackToBranch<vector<float>>("prim_upper_mdhit_sim_phi", prim_upper_mdhit_sim_phi);
        ana.tx->pushbackToBranch<vector<float>>("prim_upper_mdhit_sim_dxy", prim_upper_mdhit_sim_dxy);
        ana.tx->pushbackToBranch<vector<float>>("prim_upper_mdhit_sim_dz", prim_upper_mdhit_sim_dz);
        ana.tx->pushbackToBranch<vector<float>>("prim_upper_mdhit_x", prim_upper_mdhit_x); // paired with lower
        ana.tx->pushbackToBranch<vector<float>>("prim_upper_mdhit_y", prim_upper_mdhit_y); // paired with lower
        ana.tx->pushbackToBranch<vector<float>>("prim_upper_mdhit_z", prim_upper_mdhit_z); // paired with lower

    }

    // std::cout <<  " detids.size(): " << detids.size() <<  std::endl;

}

//________________________________________________________________________________________________________________________________
void fillPrimitiveBranches_for_CPU_v2(SDL::CPU::Event& event)
{
    fillPrimitiveBranches_Hit_for_CPU_v2(event);
    fillPrimitiveBranches_MD_for_CPU_v2(event);
}

//________________________________________________________________________________________________________________________________
void fillPrimitiveBranches_Hit_for_CPU_v2(SDL::CPU::Event& event)
{

    vector<int> prim_sim_hit_idx;
    vector<int> prim_sim_hit_layer;
    vector<int> prim_sim_hit_subdet;
    vector<int> prim_sim_hit_side;
    vector<int> prim_sim_hit_rod;
    vector<int> prim_sim_hit_ring;
    vector<int> prim_sim_hit_module;
    vector<int> prim_sim_hit_detid;
    vector<int> prim_sim_hit_isanchorlayer;
    vector<int> prim_sim_hit_islowerlayer;
    vector<float> prim_sim_hit_x;
    vector<float> prim_sim_hit_y;
    vector<float> prim_sim_hit_z;
    vector<float> prim_sim_hit_sim_pt;
    vector<float> prim_sim_hit_sim_eta;
    vector<float> prim_sim_hit_sim_phi;
    vector<float> prim_sim_hit_sim_vx;
    vector<float> prim_sim_hit_sim_vy;
    vector<float> prim_sim_hit_sim_vz;
    vector<int> prim_sim_hit_sim_idx;
    vector<int> prim_sim_hit_sim_q;
    vector<int> prim_sim_hit_sim_pdgid;
    vector<int> prim_sim_hit_sim_event;
    vector<int> prim_sim_hit_sim_bunch;
    vector<int> prim_sim_hit_sim_denom;

    vector<int> prim_nonsim_hit_idx;
    vector<int> prim_nonsim_hit_layer;
    vector<int> prim_nonsim_hit_subdet;
    vector<int> prim_nonsim_hit_side;
    vector<int> prim_nonsim_hit_rod;
    vector<int> prim_nonsim_hit_ring;
    vector<int> prim_nonsim_hit_module;
    vector<int> prim_nonsim_hit_detid;
    vector<int> prim_nonsim_hit_isanchorlayer;
    vector<int> prim_nonsim_hit_islowerlayer;
    vector<float> prim_nonsim_hit_x;
    vector<float> prim_nonsim_hit_y;
    vector<float> prim_nonsim_hit_z;
    vector<float> prim_nonsim_hit_sim_pt;
    vector<float> prim_nonsim_hit_sim_eta;
    vector<float> prim_nonsim_hit_sim_phi;
    vector<float> prim_nonsim_hit_sim_vx;
    vector<float> prim_nonsim_hit_sim_vy;
    vector<float> prim_nonsim_hit_sim_vz;
    vector<int> prim_nonsim_hit_sim_idx;
    vector<int> prim_nonsim_hit_sim_q;
    vector<int> prim_nonsim_hit_sim_pdgid;
    vector<int> prim_nonsim_hit_sim_event;
    vector<int> prim_nonsim_hit_sim_bunch;
    vector<int> prim_nonsim_hit_sim_denom;

    for (auto& module : event.getModulePtrs())
    {
        if (module->detId() == 1)
            continue;
        for (auto& hit : module->getHitPtrs())
        {
            int simhitidx = bestSimHitMatch(hit->idx());
            if (simhitidx < 0)
            {
                prim_nonsim_hit_idx           . push_back(hit->idx());
                prim_nonsim_hit_layer         . push_back(logicalLayer(*module));
                prim_nonsim_hit_subdet        . push_back(module->subdet());
                prim_nonsim_hit_side          . push_back(module->side());
                prim_nonsim_hit_rod           . push_back(module->rod());
                prim_nonsim_hit_ring          . push_back(module->ring());
                prim_nonsim_hit_module        . push_back(module->module());
                prim_nonsim_hit_detid         . push_back(module->detId());
                prim_nonsim_hit_isanchorlayer . push_back(isAnchorLayer(*module));
                prim_nonsim_hit_islowerlayer  . push_back(module->isLower());
                prim_nonsim_hit_x             . push_back(hit->x());
                prim_nonsim_hit_y             . push_back(hit->y());
                prim_nonsim_hit_z             . push_back(hit->z());
                prim_nonsim_hit_sim_pt    . push_back(-999);
                prim_nonsim_hit_sim_eta   . push_back(-999);
                prim_nonsim_hit_sim_phi   . push_back(-999);
                prim_nonsim_hit_sim_vx    . push_back(-999);
                prim_nonsim_hit_sim_vy    . push_back(-999);
                prim_nonsim_hit_sim_vz    . push_back(-999);
                prim_nonsim_hit_sim_idx   . push_back(-999);
                prim_nonsim_hit_sim_q     . push_back(-999);
                prim_nonsim_hit_sim_pdgid . push_back(-999);
                prim_nonsim_hit_sim_event . push_back(-999);
                prim_nonsim_hit_sim_bunch . push_back(-999);
                prim_nonsim_hit_sim_denom . push_back(-999);
            }
            else
            {
                int simtrkidx = trk.simhit_simTrkIdx()[simhitidx];
                prim_sim_hit_idx           . push_back(hit->idx());
                prim_sim_hit_layer         . push_back(logicalLayer(*module));
                prim_sim_hit_subdet        . push_back(module->subdet());
                prim_sim_hit_side          . push_back(module->side());
                prim_sim_hit_rod           . push_back(module->rod());
                prim_sim_hit_ring          . push_back(module->ring());
                prim_sim_hit_module        . push_back(module->module());
                prim_sim_hit_detid         . push_back(module->detId());
                prim_sim_hit_isanchorlayer . push_back(isAnchorLayer(*module));
                prim_sim_hit_islowerlayer  . push_back(module->isLower());
                prim_sim_hit_x             . push_back(hit->x());
                prim_sim_hit_y             . push_back(hit->y());
                prim_sim_hit_z             . push_back(hit->z());
                prim_sim_hit_sim_pt    . push_back(trk.sim_pt()[simtrkidx]);
                prim_sim_hit_sim_eta   . push_back(trk.sim_eta()[simtrkidx]);
                prim_sim_hit_sim_phi   . push_back(trk.sim_phi()[simtrkidx]);
                int vtxidx = trk.sim_parentVtxIdx()[simtrkidx];
                prim_sim_hit_sim_vx    . push_back(trk.simvtx_x()[vtxidx]);
                prim_sim_hit_sim_vy    . push_back(trk.simvtx_x()[vtxidx]);
                prim_sim_hit_sim_vz    . push_back(trk.simvtx_x()[vtxidx]);
                prim_sim_hit_sim_idx   . push_back(simtrkidx);
                prim_sim_hit_sim_q     . push_back(trk.sim_q()[simtrkidx]);
                prim_sim_hit_sim_pdgid . push_back(trk.sim_pdgId()[simtrkidx]);
                prim_sim_hit_sim_event . push_back(trk.sim_event()[simtrkidx]);
                prim_sim_hit_sim_bunch . push_back(trk.sim_bunchCrossing()[simtrkidx]);
                prim_sim_hit_sim_denom . push_back(getDenomSimTrkType(simtrkidx));
            }
        }
    }

    ana.tx->setBranch<vector<int>>("prim_sim_hit_idx", prim_sim_hit_idx);
    ana.tx->setBranch<vector<int>>("prim_sim_hit_layer", prim_sim_hit_layer);
    ana.tx->setBranch<vector<int>>("prim_sim_hit_subdet", prim_sim_hit_subdet);
    ana.tx->setBranch<vector<int>>("prim_sim_hit_side", prim_sim_hit_side);
    ana.tx->setBranch<vector<int>>("prim_sim_hit_rod", prim_sim_hit_rod);
    ana.tx->setBranch<vector<int>>("prim_sim_hit_ring", prim_sim_hit_ring);
    ana.tx->setBranch<vector<int>>("prim_sim_hit_module", prim_sim_hit_module);
    ana.tx->setBranch<vector<int>>("prim_sim_hit_detid", prim_sim_hit_detid);
    ana.tx->setBranch<vector<int>>("prim_sim_hit_isanchorlayer", prim_sim_hit_isanchorlayer);
    ana.tx->setBranch<vector<int>>("prim_sim_hit_islowerlayer", prim_sim_hit_islowerlayer);
    ana.tx->setBranch<vector<float>>("prim_sim_hit_x", prim_sim_hit_x);
    ana.tx->setBranch<vector<float>>("prim_sim_hit_y", prim_sim_hit_y);
    ana.tx->setBranch<vector<float>>("prim_sim_hit_z", prim_sim_hit_z);
    ana.tx->setBranch<vector<float>>("prim_sim_hit_sim_pt", prim_sim_hit_sim_pt);
    ana.tx->setBranch<vector<float>>("prim_sim_hit_sim_eta", prim_sim_hit_sim_eta);
    ana.tx->setBranch<vector<float>>("prim_sim_hit_sim_phi", prim_sim_hit_sim_phi);
    ana.tx->setBranch<vector<float>>("prim_sim_hit_sim_vx", prim_sim_hit_sim_vx);
    ana.tx->setBranch<vector<float>>("prim_sim_hit_sim_vy", prim_sim_hit_sim_vy);
    ana.tx->setBranch<vector<float>>("prim_sim_hit_sim_vz", prim_sim_hit_sim_vz);
    ana.tx->setBranch<vector<int>>("prim_sim_hit_sim_idx", prim_sim_hit_sim_idx);
    ana.tx->setBranch<vector<int>>("prim_sim_hit_sim_q", prim_sim_hit_sim_q);
    ana.tx->setBranch<vector<int>>("prim_sim_hit_sim_pdgid", prim_sim_hit_sim_pdgid);
    ana.tx->setBranch<vector<int>>("prim_sim_hit_sim_event", prim_sim_hit_sim_event);
    ana.tx->setBranch<vector<int>>("prim_sim_hit_sim_bunch", prim_sim_hit_sim_bunch);
    ana.tx->setBranch<vector<int>>("prim_sim_hit_sim_denom", prim_sim_hit_sim_denom);

    ana.tx->setBranch<vector<int>>("prim_nonsim_hit_idx", prim_nonsim_hit_idx);
    ana.tx->setBranch<vector<int>>("prim_nonsim_hit_layer", prim_nonsim_hit_layer);
    ana.tx->setBranch<vector<int>>("prim_nonsim_hit_subdet", prim_nonsim_hit_subdet);
    ana.tx->setBranch<vector<int>>("prim_nonsim_hit_side", prim_nonsim_hit_side);
    ana.tx->setBranch<vector<int>>("prim_nonsim_hit_rod", prim_nonsim_hit_rod);
    ana.tx->setBranch<vector<int>>("prim_nonsim_hit_ring", prim_nonsim_hit_ring);
    ana.tx->setBranch<vector<int>>("prim_nonsim_hit_module", prim_nonsim_hit_module);
    ana.tx->setBranch<vector<int>>("prim_nonsim_hit_detid", prim_nonsim_hit_detid);
    ana.tx->setBranch<vector<int>>("prim_nonsim_hit_isanchorlayer", prim_nonsim_hit_isanchorlayer);
    ana.tx->setBranch<vector<int>>("prim_nonsim_hit_islowerlayer", prim_nonsim_hit_islowerlayer);
    ana.tx->setBranch<vector<float>>("prim_nonsim_hit_x", prim_nonsim_hit_x);
    ana.tx->setBranch<vector<float>>("prim_nonsim_hit_y", prim_nonsim_hit_y);
    ana.tx->setBranch<vector<float>>("prim_nonsim_hit_z", prim_nonsim_hit_z);
    ana.tx->setBranch<vector<float>>("prim_nonsim_hit_sim_pt", prim_nonsim_hit_sim_pt);
    ana.tx->setBranch<vector<float>>("prim_nonsim_hit_sim_eta", prim_nonsim_hit_sim_eta);
    ana.tx->setBranch<vector<float>>("prim_nonsim_hit_sim_phi", prim_nonsim_hit_sim_phi);
    ana.tx->setBranch<vector<float>>("prim_nonsim_hit_sim_vx", prim_nonsim_hit_sim_vx);
    ana.tx->setBranch<vector<float>>("prim_nonsim_hit_sim_vy", prim_nonsim_hit_sim_vy);
    ana.tx->setBranch<vector<float>>("prim_nonsim_hit_sim_vz", prim_nonsim_hit_sim_vz);
    ana.tx->setBranch<vector<int>>("prim_nonsim_hit_sim_idx", prim_nonsim_hit_sim_idx);
    ana.tx->setBranch<vector<int>>("prim_nonsim_hit_sim_q", prim_nonsim_hit_sim_q);
    ana.tx->setBranch<vector<int>>("prim_nonsim_hit_sim_pdgid", prim_nonsim_hit_sim_pdgid);
    ana.tx->setBranch<vector<int>>("prim_nonsim_hit_sim_event", prim_nonsim_hit_sim_event);
    ana.tx->setBranch<vector<int>>("prim_nonsim_hit_sim_bunch", prim_nonsim_hit_sim_bunch);
    ana.tx->setBranch<vector<int>>("prim_nonsim_hit_sim_denom", prim_nonsim_hit_sim_denom);
}

//________________________________________________________________________________________________________________________________
void fillPrimitiveBranches_MD_for_CPU_v2(SDL::CPU::Event& event)
{

    vector<int> prim_sim_md_anchor_idx;
    vector<int> prim_sim_md_upper_idx;
    vector<int> prim_sim_md_layer;
    vector<int> prim_sim_md_subdet;
    vector<int> prim_sim_md_side;
    vector<int> prim_sim_md_rod;
    vector<int> prim_sim_md_ring;
    vector<int> prim_sim_md_module;
    vector<int> prim_sim_md_detid;
    vector<int> prim_sim_md_isanchorlayer;
    vector<int> prim_sim_md_islowerlayer;
    vector<int> prim_sim_md_nsim_match;
    vector<float> prim_sim_md_anchor_x;
    vector<float> prim_sim_md_anchor_y;
    vector<float> prim_sim_md_anchor_z;
    vector<float> prim_sim_md_upper_x;
    vector<float> prim_sim_md_upper_y;
    vector<float> prim_sim_md_upper_z;
    vector<float> prim_sim_md_sim_pt;
    vector<float> prim_sim_md_sim_eta;
    vector<float> prim_sim_md_sim_phi;
    vector<float> prim_sim_md_sim_vx;
    vector<float> prim_sim_md_sim_vy;
    vector<float> prim_sim_md_sim_vz;
    vector<int> prim_sim_md_sim_idx;
    vector<int> prim_sim_md_sim_q;
    vector<int> prim_sim_md_sim_pdgid;
    vector<int> prim_sim_md_sim_event;
    vector<int> prim_sim_md_sim_bunch;
    vector<int> prim_sim_md_sim_denom;

    vector<int> prim_nonsim_md_anchor_idx;
    vector<int> prim_nonsim_md_upper_idx;
    vector<int> prim_nonsim_md_layer;
    vector<int> prim_nonsim_md_subdet;
    vector<int> prim_nonsim_md_side;
    vector<int> prim_nonsim_md_rod;
    vector<int> prim_nonsim_md_ring;
    vector<int> prim_nonsim_md_module;
    vector<int> prim_nonsim_md_detid;
    vector<int> prim_nonsim_md_isanchorlayer;
    vector<int> prim_nonsim_md_islowerlayer;
    vector<int> prim_nonsim_md_nsim_match;
    vector<float> prim_nonsim_md_anchor_x;
    vector<float> prim_nonsim_md_anchor_y;
    vector<float> prim_nonsim_md_anchor_z;
    vector<float> prim_nonsim_md_upper_x;
    vector<float> prim_nonsim_md_upper_y;
    vector<float> prim_nonsim_md_upper_z;
    vector<float> prim_nonsim_md_sim_pt;
    vector<float> prim_nonsim_md_sim_eta;
    vector<float> prim_nonsim_md_sim_phi;
    vector<float> prim_nonsim_md_sim_vx;
    vector<float> prim_nonsim_md_sim_vy;
    vector<float> prim_nonsim_md_sim_vz;
    vector<int> prim_nonsim_md_sim_idx;
    vector<int> prim_nonsim_md_sim_q;
    vector<int> prim_nonsim_md_sim_pdgid;
    vector<int> prim_nonsim_md_sim_event;
    vector<int> prim_nonsim_md_sim_bunch;
    vector<int> prim_nonsim_md_sim_denom;

    for (auto& module : event.getLowerModulePtrs())
    {
        if (module->detId() == 1)
            continue;
        for (auto& mdPtr : module->getMiniDoubletPtrs())
        {

            SDL::CPU::Hit* lhit = mdPtr->lowerHitPtr();
            SDL::CPU::Hit* uhit = mdPtr->upperHitPtr();
            SDL::CPU::Hit* ahit = mdPtr->anchorHitPtr();
            SDL::CPU::Hit* nahit = lhit == ahit ? uhit : lhit;
            std::vector<int> matchSimTrkIdxs = matchedSimTrkIdxs({lhit->idx(), uhit->idx()}, {4, 4});

            int nsim_match = 0;
            if (trk.ph2_simHitIdx()[lhit->idx()].size() > 0) nsim_match++;
            if (trk.ph2_simHitIdx()[uhit->idx()].size() > 0) nsim_match++;

            if (matchSimTrkIdxs.size() == 0) // no match
            {
                prim_nonsim_md_anchor_idx    . push_back(ahit->idx());
                prim_nonsim_md_upper_idx     . push_back(nahit->idx());
                prim_nonsim_md_layer         . push_back(logicalLayer(*module));
                prim_nonsim_md_subdet        . push_back(module->subdet());
                prim_nonsim_md_side          . push_back(module->side());
                prim_nonsim_md_rod           . push_back(module->rod());
                prim_nonsim_md_ring          . push_back(module->ring());
                prim_nonsim_md_module        . push_back(module->module());
                prim_nonsim_md_detid         . push_back(module->detId());
                prim_nonsim_md_isanchorlayer . push_back(isAnchorLayer(*module));
                prim_nonsim_md_islowerlayer  . push_back(module->isLower());
                prim_nonsim_md_nsim_match    . push_back(nsim_match);
                prim_nonsim_md_anchor_x      . push_back(ahit->x());
                prim_nonsim_md_anchor_y      . push_back(ahit->y());
                prim_nonsim_md_anchor_z      . push_back(ahit->z());
                prim_nonsim_md_upper_x       . push_back(nahit->x());
                prim_nonsim_md_upper_y       . push_back(nahit->y());
                prim_nonsim_md_upper_z       . push_back(nahit->z());
                prim_nonsim_md_sim_pt    . push_back(-999);
                prim_nonsim_md_sim_eta   . push_back(-999);
                prim_nonsim_md_sim_phi   . push_back(-999);
                prim_nonsim_md_sim_vx    . push_back(-999);
                prim_nonsim_md_sim_vy    . push_back(-999);
                prim_nonsim_md_sim_vz    . push_back(-999);
                prim_nonsim_md_sim_idx   . push_back(-999);
                prim_nonsim_md_sim_q     . push_back(-999);
                prim_nonsim_md_sim_pdgid . push_back(-999);
                prim_nonsim_md_sim_event . push_back(-999);
                prim_nonsim_md_sim_bunch . push_back(-999);
                prim_nonsim_md_sim_denom . push_back(-999);
            }
            else
            {
                int simtrkidx = matchSimTrkIdxs.at(0); // Take first match
                prim_sim_md_anchor_idx    . push_back(ahit->idx());
                prim_sim_md_upper_idx     . push_back(nahit->idx());
                prim_sim_md_layer         . push_back(logicalLayer(*module));
                prim_sim_md_subdet        . push_back(module->subdet());
                prim_sim_md_side          . push_back(module->side());
                prim_sim_md_rod           . push_back(module->rod());
                prim_sim_md_ring          . push_back(module->ring());
                prim_sim_md_module        . push_back(module->module());
                prim_sim_md_detid         . push_back(module->detId());
                prim_sim_md_isanchorlayer . push_back(isAnchorLayer(*module));
                prim_sim_md_islowerlayer  . push_back(module->isLower());
                prim_sim_md_nsim_match    . push_back(nsim_match);
                prim_sim_md_anchor_x      . push_back(ahit->x());
                prim_sim_md_anchor_y      . push_back(ahit->y());
                prim_sim_md_anchor_z      . push_back(ahit->z());
                prim_sim_md_upper_x       . push_back(nahit->x());
                prim_sim_md_upper_y       . push_back(nahit->y());
                prim_sim_md_upper_z       . push_back(nahit->z());
                prim_sim_md_sim_pt    . push_back(trk.sim_pt()[simtrkidx]);
                prim_sim_md_sim_eta   . push_back(trk.sim_eta()[simtrkidx]);
                prim_sim_md_sim_phi   . push_back(trk.sim_phi()[simtrkidx]);
                int vtxidx = trk.sim_parentVtxIdx()[simtrkidx];
                prim_sim_md_sim_vx    . push_back(trk.simvtx_x()[vtxidx]);
                prim_sim_md_sim_vy    . push_back(trk.simvtx_x()[vtxidx]);
                prim_sim_md_sim_vz    . push_back(trk.simvtx_x()[vtxidx]);
                prim_sim_md_sim_idx   . push_back(simtrkidx);
                prim_sim_md_sim_q     . push_back(trk.sim_q()[simtrkidx]);
                prim_sim_md_sim_pdgid . push_back(trk.sim_pdgId()[simtrkidx]);
                prim_sim_md_sim_event . push_back(trk.sim_event()[simtrkidx]);
                prim_sim_md_sim_bunch . push_back(trk.sim_bunchCrossing()[simtrkidx]);
                prim_sim_md_sim_denom . push_back(getDenomSimTrkType(simtrkidx));
            }
        }
    }

    ana.tx->setBranch<vector<int>>("prim_sim_md_anchor_idx", prim_sim_md_anchor_idx);
    ana.tx->setBranch<vector<int>>("prim_sim_md_upper_idx", prim_sim_md_upper_idx);
    ana.tx->setBranch<vector<int>>("prim_sim_md_layer", prim_sim_md_layer);
    ana.tx->setBranch<vector<int>>("prim_sim_md_subdet", prim_sim_md_subdet);
    ana.tx->setBranch<vector<int>>("prim_sim_md_side", prim_sim_md_side);
    ana.tx->setBranch<vector<int>>("prim_sim_md_rod", prim_sim_md_rod);
    ana.tx->setBranch<vector<int>>("prim_sim_md_ring", prim_sim_md_ring);
    ana.tx->setBranch<vector<int>>("prim_sim_md_module", prim_sim_md_module);
    ana.tx->setBranch<vector<int>>("prim_sim_md_detid", prim_sim_md_detid);
    ana.tx->setBranch<vector<int>>("prim_sim_md_isanchorlayer", prim_sim_md_isanchorlayer);
    ana.tx->setBranch<vector<int>>("prim_sim_md_islowerlayer", prim_sim_md_islowerlayer);
    ana.tx->setBranch<vector<int>>("prim_sim_md_nsim_match", prim_sim_md_nsim_match);
    ana.tx->setBranch<vector<float>>("prim_sim_md_anchor_x", prim_sim_md_anchor_x);
    ana.tx->setBranch<vector<float>>("prim_sim_md_anchor_y", prim_sim_md_anchor_y);
    ana.tx->setBranch<vector<float>>("prim_sim_md_anchor_z", prim_sim_md_anchor_z);
    ana.tx->setBranch<vector<float>>("prim_sim_md_upper_x", prim_sim_md_upper_x);
    ana.tx->setBranch<vector<float>>("prim_sim_md_upper_y", prim_sim_md_upper_y);
    ana.tx->setBranch<vector<float>>("prim_sim_md_upper_z", prim_sim_md_upper_z);
    ana.tx->setBranch<vector<float>>("prim_sim_md_sim_pt", prim_sim_md_sim_pt);
    ana.tx->setBranch<vector<float>>("prim_sim_md_sim_eta", prim_sim_md_sim_eta);
    ana.tx->setBranch<vector<float>>("prim_sim_md_sim_phi", prim_sim_md_sim_phi);
    ana.tx->setBranch<vector<float>>("prim_sim_md_sim_vx", prim_sim_md_sim_vx);
    ana.tx->setBranch<vector<float>>("prim_sim_md_sim_vy", prim_sim_md_sim_vy);
    ana.tx->setBranch<vector<float>>("prim_sim_md_sim_vz", prim_sim_md_sim_vz);
    ana.tx->setBranch<vector<int>>("prim_sim_md_sim_idx", prim_sim_md_sim_idx);
    ana.tx->setBranch<vector<int>>("prim_sim_md_sim_q", prim_sim_md_sim_q);
    ana.tx->setBranch<vector<int>>("prim_sim_md_sim_pdgid", prim_sim_md_sim_pdgid);
    ana.tx->setBranch<vector<int>>("prim_sim_md_sim_event", prim_sim_md_sim_event);
    ana.tx->setBranch<vector<int>>("prim_sim_md_sim_bunch", prim_sim_md_sim_bunch);
    ana.tx->setBranch<vector<int>>("prim_sim_md_sim_denom", prim_sim_md_sim_denom);

    ana.tx->setBranch<vector<int>>("prim_nonsim_md_anchor_idx", prim_nonsim_md_anchor_idx);
    ana.tx->setBranch<vector<int>>("prim_nonsim_md_upper_idx", prim_nonsim_md_upper_idx);
    ana.tx->setBranch<vector<int>>("prim_nonsim_md_layer", prim_nonsim_md_layer);
    ana.tx->setBranch<vector<int>>("prim_nonsim_md_subdet", prim_nonsim_md_subdet);
    ana.tx->setBranch<vector<int>>("prim_nonsim_md_side", prim_nonsim_md_side);
    ana.tx->setBranch<vector<int>>("prim_nonsim_md_rod", prim_nonsim_md_rod);
    ana.tx->setBranch<vector<int>>("prim_nonsim_md_ring", prim_nonsim_md_ring);
    ana.tx->setBranch<vector<int>>("prim_nonsim_md_module", prim_nonsim_md_module);
    ana.tx->setBranch<vector<int>>("prim_nonsim_md_detid", prim_nonsim_md_detid);
    ana.tx->setBranch<vector<int>>("prim_nonsim_md_isanchorlayer", prim_nonsim_md_isanchorlayer);
    ana.tx->setBranch<vector<int>>("prim_nonsim_md_islowerlayer", prim_nonsim_md_islowerlayer);
    ana.tx->setBranch<vector<int>>("prim_nonsim_md_nsim_match", prim_nonsim_md_nsim_match);
    ana.tx->setBranch<vector<float>>("prim_nonsim_md_anchor_x", prim_nonsim_md_anchor_x);
    ana.tx->setBranch<vector<float>>("prim_nonsim_md_anchor_y", prim_nonsim_md_anchor_y);
    ana.tx->setBranch<vector<float>>("prim_nonsim_md_anchor_z", prim_nonsim_md_anchor_z);
    ana.tx->setBranch<vector<float>>("prim_nonsim_md_upper_x", prim_nonsim_md_upper_x);
    ana.tx->setBranch<vector<float>>("prim_nonsim_md_upper_y", prim_nonsim_md_upper_y);
    ana.tx->setBranch<vector<float>>("prim_nonsim_md_upper_z", prim_nonsim_md_upper_z);
    ana.tx->setBranch<vector<float>>("prim_nonsim_md_sim_pt", prim_nonsim_md_sim_pt);
    ana.tx->setBranch<vector<float>>("prim_nonsim_md_sim_eta", prim_nonsim_md_sim_eta);
    ana.tx->setBranch<vector<float>>("prim_nonsim_md_sim_phi", prim_nonsim_md_sim_phi);
    ana.tx->setBranch<vector<float>>("prim_nonsim_md_sim_vx", prim_nonsim_md_sim_vx);
    ana.tx->setBranch<vector<float>>("prim_nonsim_md_sim_vy", prim_nonsim_md_sim_vy);
    ana.tx->setBranch<vector<float>>("prim_nonsim_md_sim_vz", prim_nonsim_md_sim_vz);
    ana.tx->setBranch<vector<int>>("prim_nonsim_md_sim_idx", prim_nonsim_md_sim_idx);
    ana.tx->setBranch<vector<int>>("prim_nonsim_md_sim_q", prim_nonsim_md_sim_q);
    ana.tx->setBranch<vector<int>>("prim_nonsim_md_sim_pdgid", prim_nonsim_md_sim_pdgid);
    ana.tx->setBranch<vector<int>>("prim_nonsim_md_sim_event", prim_nonsim_md_sim_event);
    ana.tx->setBranch<vector<int>>("prim_nonsim_md_sim_bunch", prim_nonsim_md_sim_bunch);
    ana.tx->setBranch<vector<int>>("prim_nonsim_md_sim_denom", prim_nonsim_md_sim_denom);
}


//________________________________________________________________________________________________________________________________
void printTimingInformation(std::vector<std::vector<float>>& timing_information,float fullTime,float fullavg)
{

    if (ana.verbose == 0)
        return;

    std::cout << showpoint;
    std::cout << fixed;
    std::cout << setprecision(2);
    std::cout << right;
    std::cout << "Timing summary" << std::endl;
    std::cout << "   Evt    Hits       MD       LS      T3       T5       pLS       pT5      pT3      TC      TCE        Total     Total(short)" << std::endl;
    std::vector<float> timing_sum_information(timing_information[0].size());
    std::vector<float> timing_shortlist;
    std::vector<float> timing_list;
    for (auto&& [ievt, timing] : iter::enumerate(timing_information))
    {
        float timing_total = 0.f;
        float timing_total_short = 0.f;
        timing_total += timing[0]*1000; // Hits
        timing_total += timing[1]*1000; // MD
        timing_total += timing[2]*1000; // LS
        timing_total += timing[3]*1000; // T3
        timing_total += timing[4]*1000; // T5
        timing_total += timing[5]*1000; // pLS
        timing_total += timing[6]*1000; // pT5
        timing_total += timing[7]*1000; // pT3
        timing_total += timing[8]*1000; // TC
        timing_total += timing[9]*1000; // TCE
        timing_total_short += timing[1]*1000; // MD
        timing_total_short += timing[2]*1000; // LS
        timing_total_short += timing[3]*1000; // T3
        timing_total_short += timing[4]*1000; // T5
        timing_total_short += timing[6]*1000; //pT5
        timing_total_short += timing[7]*1000; //pT3
        timing_total_short += timing[8]*1000; //TC
        timing_total_short += timing[9]*1000; //TCE
        std::cout << setw(6) << ievt;
        std::cout << "   "<<setw(6) << timing[0]*1000; // Hits
        std::cout << "   "<<setw(6) << timing[1]*1000; // MD
        std::cout << "   "<<setw(6) << timing[2]*1000; // LS
        std::cout << "   "<<setw(6) << timing[3]*1000; // T3
        std::cout << "   "<<setw(6) << timing[4]*1000; // T5
        std::cout << "   "<<setw(6) << timing[5]*1000; //pLS
        std::cout << "   "<<setw(6) << timing[6]*1000; //pT5
        std::cout << "   "<<setw(6) << timing[7]*1000; //pT3
        std::cout << "   "<<setw(6) << timing[8]*1000; //TC
        std::cout << "   "<<setw(6) << timing[9]*1000; //TCE
        std::cout << "   "<<setw(7) << timing_total; // Total time
        std::cout << "   "<<setw(7) << timing_total_short; // Total time
        std::cout << std::endl;
        timing_sum_information[0] += timing[0]*1000; // Hits
        timing_sum_information[1] += timing[1]*1000; // MD
        timing_sum_information[2] += timing[2]*1000; // LS
        timing_sum_information[3] += timing[3]*1000; // T3
        timing_sum_information[4] += timing[4]*1000; // T5
        timing_sum_information[5] += timing[5]*1000; // pLS
        timing_sum_information[6] += timing[6]*1000; // pT5
        timing_sum_information[7] += timing[7]*1000; // pT3
        timing_sum_information[8] += timing[8]*1000; // TC
        timing_sum_information[9] += timing[9]*1000; // TCE
        timing_shortlist.push_back(timing_total_short); //short total
        timing_list.push_back(timing_total); //short total
    }
    timing_sum_information[0] /= timing_information.size(); // Hits
    timing_sum_information[1] /= timing_information.size(); // MD
    timing_sum_information[2] /= timing_information.size(); // LS
    timing_sum_information[3] /= timing_information.size(); // T3
    timing_sum_information[4] /= timing_information.size(); // T5
    timing_sum_information[5] /= timing_information.size(); // pLS
    timing_sum_information[6] /= timing_information.size(); // pT5
    timing_sum_information[7] /= timing_information.size(); // pT3
    timing_sum_information[8] /= timing_information.size(); // TC
    timing_sum_information[9] /= timing_information.size(); // TCE

    float timing_total_avg = 0.0;
    timing_total_avg += timing_sum_information[0]; // Hits
    timing_total_avg += timing_sum_information[1]; // MD
    timing_total_avg += timing_sum_information[2]; // LS
    timing_total_avg += timing_sum_information[3]; // T3
    timing_total_avg += timing_sum_information[4]; // T5
    timing_total_avg += timing_sum_information[5]; // pLS
    timing_total_avg += timing_sum_information[6]; // pT5
    timing_total_avg += timing_sum_information[7]; // pT3
    timing_total_avg += timing_sum_information[8]; // TC
    timing_total_avg += timing_sum_information[9]; // TCE
    float timing_totalshort_avg = 0.0;
    timing_totalshort_avg += timing_sum_information[1]; // MD
    timing_totalshort_avg += timing_sum_information[2]; // LS
    timing_totalshort_avg += timing_sum_information[3]; // T3
    timing_totalshort_avg += timing_sum_information[4]; // T5
    timing_totalshort_avg += timing_sum_information[6]; // pT5
    timing_totalshort_avg += timing_sum_information[7]; // pT3
    timing_totalshort_avg += timing_sum_information[8]; // TC
    timing_totalshort_avg += timing_sum_information[9]; // TCE

    float standardDeviation = 0.0;
    for(auto shorttime: timing_shortlist) {
      standardDeviation += pow(shorttime - timing_totalshort_avg, 2);
    }
    float stdDev = sqrt(standardDeviation/timing_shortlist.size());

    //float standardDeviationFull = 0.0;
    //for(auto time: timing_list) {
    //  standardDeviationFull += pow(time - timing_total_avg, 2);
    //}
    //float stdDevFull = sqrt(standardDeviationFull/timing_list.size());

    float effectiveThroughput = fullavg*timing_totalshort_avg/timing_total_avg;
    //float std_throughput = effectiveThroughput * sqrt(pow(stdDevFull/timing_total_avg,2) + pow(stdDev/timing_totalshort_avg,2));

    std::cout << setprecision(0);
    std::cout << "   Evt    Hits       MD       LS      T3       T5       pLS       pT5      pT3      TC       TCE      Event      Short             Loop      Effective" << std::endl;
    std::cout << setw(6) << "avg";
    std::cout << "   "<<setw(6) << timing_sum_information[0]; // Hits
    std::cout << "   "<<setw(6) << timing_sum_information[1]; // MD
    std::cout << "   "<<setw(6) << timing_sum_information[2]; // LS
    std::cout << "   "<<setw(6) << timing_sum_information[3]; // T3
    std::cout << "   "<<setw(6) << timing_sum_information[4]; // T5
    std::cout << "   "<<setw(6) << timing_sum_information[5]; // pLS
    std::cout << "   "<<setw(6) << timing_sum_information[6]; // pT5
    std::cout << "   "<<setw(6) << timing_sum_information[7]; // pT3
    std::cout << "   "<<setw(6) << timing_sum_information[8]; // TC
    std::cout << "   "<<setw(6) << timing_sum_information[9]; // TCE
    std::cout << "   "<<setw(7) << timing_total_avg; // Average total time
    //std::cout << "+/- "<< stdDevFull;
    std::cout << "   "<<setw(7) << timing_totalshort_avg; // Average total time
    std::cout << "+/- "<< setw(4)<<stdDev; 
    //std::cout << "   "<<setw(7) << fullTime; // Full time
    std::cout << "   "<<setw(7) << fullavg; // Average full time
    std::cout << "   "<<setw(7) << effectiveThroughput; // Effective time
    //std::cout << "+/- "<< std_throughput; 
    std::cout << "   "<<ana.compilation_target;
    std::cout << "[s="<<ana.streams<<"]";
    std::cout << std::endl;

    std::cout << left;

}

//________________________________________________________________________________________________________________________________
void printHitMultiplicities(SDL::Event* event)
{
    SDL::hits& hitsInGPU = (*event->getHits());
    SDL::modules& modulesInGPU = (*event->getModules());
    SDL::objectRanges& rangesInGPU = (*event->getRanges());

    int nHits = 0;
    for (unsigned int idx = 0; idx <= *(modulesInGPU.nLowerModules); idx++) // "<=" because cheating to include pixel track candidate lower module
    {
        nHits += rangesInGPU.hitRanges[4 * idx + 1] - rangesInGPU.hitRanges[4 * idx] + 1;       
        nHits += rangesInGPU.hitRanges[4 * idx + 3] - rangesInGPU.hitRanges[4 * idx + 2] + 1;
    }
    std::cout <<  " nHits: " << nHits <<  std::endl;
}

//________________________________________________________________________________________________________________________________
void printMiniDoubletMultiplicities(SDL::Event* event)
{
    SDL::miniDoublets& miniDoubletsInGPU = (*event->getMiniDoublets());
    SDL::modules& modulesInGPU = (*event->getModules());

    int nMiniDoublets = 0;
    int totOccupancyMiniDoublets = 0;
    for (unsigned int idx = 0; idx <= *(modulesInGPU.nModules); idx++) // "<=" because cheating to include pixel track candidate lower module
    {
        if(modulesInGPU.isLower[idx])
        {
            nMiniDoublets += miniDoubletsInGPU.nMDs[idx];
            totOccupancyMiniDoublets += miniDoubletsInGPU.totOccupancyMDs[idx];
        }
    }
    std::cout <<  " nMiniDoublets: " << nMiniDoublets <<  std::endl;
    std::cout <<  " totOccupancyMiniDoublets (including trucated ones): " << totOccupancyMiniDoublets <<  std::endl;
}

//________________________________________________________________________________________________________________________________
void printAllObjects(SDL::Event* event)
{
    printMDs(event);
    printLSs(event);
    printpLSs(event);
    printT3s(event);
    //printpT4s(event);
}

//________________________________________________________________________________________________________________________________
void printAllObjects_for_CPU(SDL::CPU::Event& event)
{
    printMDs_for_CPU(event);
    printLSs_for_CPU(event);
    printpLSs_for_CPU(event);
    printT3s_for_CPU(event);
    printTCs_for_CPU(event);
}


//________________________________________________________________________________________________________________________________
void printMDs(SDL::Event* event)
{
    SDL::miniDoublets& miniDoubletsInGPU = (*event->getMiniDoublets());
    SDL::hits& hitsInGPU = (*event->getHits());
    SDL::modules& modulesInGPU = (*event->getModules());
    for (unsigned int idx = 0; idx <= *(modulesInGPU.nLowerModules); ++idx)
    {
        for (unsigned int jdx = 0; jdx < miniDoubletsInGPU.nMDs[2*idx]; jdx++)
        {
            unsigned int mdIdx = (2*idx) * 100 + jdx;
            unsigned int LowerHitIndex = miniDoubletsInGPU.anchorHitIndices[mdIdx];
            unsigned int UpperHitIndex = miniDoubletsInGPU.outerHitIndices[mdIdx];
            unsigned int hit0 = hitsInGPU.idxs[LowerHitIndex];
            unsigned int hit1 = hitsInGPU.idxs[UpperHitIndex];
            std::cout <<  "VALIDATION 'MD': " << "MD" <<  " hit0: " << hit0 <<  " hit1: " << hit1 <<  std::endl;
        }
        for (unsigned int jdx = 0; jdx < miniDoubletsInGPU.nMDs[2*idx+1]; jdx++)
        {
            unsigned int mdIdx = (2*idx+1) * 100 + jdx;
            unsigned int LowerHitIndex = miniDoubletsInGPU.anchorHitIndices[mdIdx];
            unsigned int UpperHitIndex = miniDoubletsInGPU.outerHitIndices[mdIdx];
            unsigned int hit0 = hitsInGPU.idxs[LowerHitIndex];
            unsigned int hit1 = hitsInGPU.idxs[UpperHitIndex];
            std::cout <<  "VALIDATION 'MD': " << "MD" <<  " hit0: " << hit0 <<  " hit1: " << hit1 <<  std::endl;
        }
    }
}

//________________________________________________________________________________________________________________________________
void printMDs_for_CPU(SDL::CPU::Event& event)
{
    // get layer ptrs
    const std::vector<SDL::CPU::Layer*> layerPtrs = event.getLayerPtrs();

    // Loop over layers and access minidoublets
    for (auto& layerPtr : layerPtrs)
    {

        // MiniDoublet ptrs
        const std::vector<SDL::CPU::MiniDoublet*>& minidoubletPtrs = layerPtr->getMiniDoubletPtrs();

        // Loop over minidoublet ptrs
        for (auto& minidoubletPtr : minidoubletPtrs)
        {

            // hit idx
            unsigned int hit0 = minidoubletPtr->lowerHitPtr()->idx();
            unsigned int hit1 = minidoubletPtr->upperHitPtr()->idx();

            std::cout <<  "VALIDATION 'MD': " << "MD" <<  " hit0: " << hit0 <<  " hit1: " << hit1 <<  std::endl;

        }

    }

}

//________________________________________________________________________________________________________________________________
void printLSs(SDL::Event* event)
{
    SDL::segments& segmentsInGPU = (*event->getSegments());
    SDL::miniDoublets& miniDoubletsInGPU = (*event->getMiniDoublets());
    SDL::hits& hitsInGPU = (*event->getHits());
    SDL::modules& modulesInGPU = (*event->getModules());
    int nSegments = 0;
    for (unsigned int i = 0; i <  *(modulesInGPU.nLowerModules); ++i)
    {
        unsigned int idx = i;//modulesInGPU.lowerModuleIndices[i];
        nSegments += segmentsInGPU.nSegments[idx];
        for (unsigned int jdx = 0; jdx < segmentsInGPU.nSegments[idx]; jdx++)
        {
            unsigned int sgIdx = idx * 600 + jdx;
            unsigned int InnerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * sgIdx];
            unsigned int OuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * sgIdx + 1];
            unsigned int InnerMiniDoubletLowerHitIndex = miniDoubletsInGPU.anchorHitIndices[InnerMiniDoubletIndex];
            unsigned int InnerMiniDoubletUpperHitIndex = miniDoubletsInGPU.outerHitIndices[InnerMiniDoubletIndex];
            unsigned int OuterMiniDoubletLowerHitIndex = miniDoubletsInGPU.anchorHitIndices[OuterMiniDoubletIndex];
            unsigned int OuterMiniDoubletUpperHitIndex = miniDoubletsInGPU.outerHitIndices[OuterMiniDoubletIndex];
            unsigned int hit0 = hitsInGPU.idxs[InnerMiniDoubletLowerHitIndex];
            unsigned int hit1 = hitsInGPU.idxs[InnerMiniDoubletUpperHitIndex];
            unsigned int hit2 = hitsInGPU.idxs[OuterMiniDoubletLowerHitIndex];
            unsigned int hit3 = hitsInGPU.idxs[OuterMiniDoubletUpperHitIndex];
            std::cout <<  "VALIDATION 'LS': " << "LS" <<  " hit0: " << hit0 <<  " hit1: " << hit1 <<  " hit2: " << hit2 <<  " hit3: " << hit3 <<  std::endl;
        }
    }
    std::cout <<  "VALIDATION nSegments: " << nSegments <<  std::endl;
}

//________________________________________________________________________________________________________________________________
void printLSs_for_CPU(SDL::CPU::Event& event)
{
    // get layer ptrs
    const std::vector<SDL::CPU::Layer*> layerPtrs = event.getLayerPtrs();

    // Loop over layers and access segments
    for (auto& layerPtr : layerPtrs)
    {

        // MiniDoublet ptrs
        const std::vector<SDL::CPU::Segment*>& segmentPtrs = layerPtr->getSegmentPtrs();

        // Loop over segment ptrs
        for (auto& segmentPtr : segmentPtrs)
        {

            // hit idx
            unsigned int hit0 = segmentPtr->innerMiniDoubletPtr()->lowerHitPtr()->idx();
            unsigned int hit1 = segmentPtr->innerMiniDoubletPtr()->upperHitPtr()->idx();
            unsigned int hit2 = segmentPtr->outerMiniDoubletPtr()->lowerHitPtr()->idx();
            unsigned int hit3 = segmentPtr->outerMiniDoubletPtr()->upperHitPtr()->idx();

            std::cout <<  "VALIDATION 'LS': " << "LS" <<  " hit0: " << hit0 <<  " hit1: " << hit1 <<  " hit2: " << hit2 <<  " hit3: " << hit3 <<  std::endl;

        }

    }

}

//________________________________________________________________________________________________________________________________
void printpLSs(SDL::Event* event)
{
    SDL::segments& segmentsInGPU = (*event->getSegments());
    SDL::miniDoublets& miniDoubletsInGPU = (*event->getMiniDoublets());
    SDL::hits& hitsInGPU = (*event->getHits());
    SDL::modules& modulesInGPU = (*event->getModules());
    unsigned int i = *(modulesInGPU.nLowerModules);
    unsigned int idx = i;//modulesInGPU.lowerModuleIndices[i];
    int npLS = segmentsInGPU.nSegments[idx];
    for (unsigned int jdx = 0; jdx < segmentsInGPU.nSegments[idx]; jdx++)
    {
        unsigned int sgIdx = idx * 600 + jdx;
        unsigned int InnerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * sgIdx];
        unsigned int OuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * sgIdx + 1];
        unsigned int InnerMiniDoubletLowerHitIndex = miniDoubletsInGPU.anchorHitIndices[InnerMiniDoubletIndex];
        unsigned int InnerMiniDoubletUpperHitIndex = miniDoubletsInGPU.outerHitIndices[InnerMiniDoubletIndex];
        unsigned int OuterMiniDoubletLowerHitIndex = miniDoubletsInGPU.anchorHitIndices[OuterMiniDoubletIndex];
        unsigned int OuterMiniDoubletUpperHitIndex = miniDoubletsInGPU.outerHitIndices[OuterMiniDoubletIndex];
        unsigned int hit0 = hitsInGPU.idxs[InnerMiniDoubletLowerHitIndex];
        unsigned int hit1 = hitsInGPU.idxs[InnerMiniDoubletUpperHitIndex];
        unsigned int hit2 = hitsInGPU.idxs[OuterMiniDoubletLowerHitIndex];
        unsigned int hit3 = hitsInGPU.idxs[OuterMiniDoubletUpperHitIndex];
        std::cout <<  "VALIDATION 'pLS': " << "pLS" <<  " hit0: " << hit0 <<  " hit1: " << hit1 <<  " hit2: " << hit2 <<  " hit3: " << hit3 <<  std::endl;
    }
    std::cout <<  "VALIDATION npLS: " << npLS <<  std::endl;
}

//________________________________________________________________________________________________________________________________
void printpLSs_for_CPU(SDL::CPU::Event& event)
{
    // get layer ptr
    SDL::CPU::Layer* layerPtr = &(event.getPixelLayer());

    // MiniDoublet ptrs
    const std::vector<SDL::CPU::Segment*>& segmentPtrs = layerPtr->getSegmentPtrs();

    // Number of pLS
    int npLS = segmentPtrs.size();

    // Loop over segment ptrs
    for (auto& segmentPtr : segmentPtrs)
    {

        // hit idx
        unsigned int hit0 = segmentPtr->innerMiniDoubletPtr()->lowerHitPtr()->idx();
        unsigned int hit1 = segmentPtr->innerMiniDoubletPtr()->upperHitPtr()->idx();
        unsigned int hit2 = segmentPtr->outerMiniDoubletPtr()->lowerHitPtr()->idx();
        unsigned int hit3 = segmentPtr->outerMiniDoubletPtr()->upperHitPtr()->idx();

        std::cout <<  "VALIDATION 'pLS': " << "pLS" <<  " hit0: " << hit0 <<  " hit1: " << hit1 <<  " hit2: " << hit2 <<  " hit3: " << hit3 <<  std::endl;

    }
    std::cout <<  "VALIDATION npLS: " << npLS <<  std::endl;

}

//________________________________________________________________________________________________________________________________
void printT3s(SDL::Event* event)
{
    SDL::triplets& tripletsInGPU = (*event->getTriplets());
    SDL::segments& segmentsInGPU = (*event->getSegments());
    SDL::miniDoublets& miniDoubletsInGPU = (*event->getMiniDoublets());
    SDL::hits& hitsInGPU = (*event->getHits());
    SDL::modules& modulesInGPU = (*event->getModules());
    int nTriplets = 0;
    for (unsigned int i = 0; i <  *(modulesInGPU.nLowerModules); ++i)
    {
        // unsigned int idx = SDL::modulesInGPU->lowerModuleIndices[i];
        nTriplets += tripletsInGPU.nTriplets[i];
        unsigned int idx = i;
        for (unsigned int jdx = 0; jdx < tripletsInGPU.nTriplets[idx]; jdx++)
        {
            unsigned int tpIdx = idx * 5000 + jdx;
            unsigned int InnerSegmentIndex = tripletsInGPU.segmentIndices[2 * tpIdx];
            unsigned int OuterSegmentIndex = tripletsInGPU.segmentIndices[2 * tpIdx + 1];
            unsigned int InnerSegmentInnerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * InnerSegmentIndex];
            unsigned int InnerSegmentOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * InnerSegmentIndex + 1];
            unsigned int OuterSegmentOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * OuterSegmentIndex + 1];

            unsigned int hit_idx0 = miniDoubletsInGPU.anchorHitIndices[InnerSegmentInnerMiniDoubletIndex];
            unsigned int hit_idx1 = miniDoubletsInGPU.outerHitIndices[InnerSegmentInnerMiniDoubletIndex];
            unsigned int hit_idx2 = miniDoubletsInGPU.anchorHitIndices[InnerSegmentOuterMiniDoubletIndex];
            unsigned int hit_idx3 = miniDoubletsInGPU.outerHitIndices[InnerSegmentOuterMiniDoubletIndex];
            unsigned int hit_idx4 = miniDoubletsInGPU.anchorHitIndices[OuterSegmentOuterMiniDoubletIndex];
            unsigned int hit_idx5 = miniDoubletsInGPU.outerHitIndices[OuterSegmentOuterMiniDoubletIndex];

            unsigned int hit0 = hitsInGPU.idxs[hit_idx0];
            unsigned int hit1 = hitsInGPU.idxs[hit_idx1];
            unsigned int hit2 = hitsInGPU.idxs[hit_idx2];
            unsigned int hit3 = hitsInGPU.idxs[hit_idx3];
            unsigned int hit4 = hitsInGPU.idxs[hit_idx4];
            unsigned int hit5 = hitsInGPU.idxs[hit_idx5];
            std::cout <<  "VALIDATION 'T3': " << "T3" <<  " hit0: " << hit0 <<  " hit1: " << hit1 <<  " hit2: " << hit2 <<  " hit3: " << hit3 <<  " hit4: " << hit4 <<  " hit5: " << hit5 <<  std::endl;
        }
    }
    std::cout <<  "VALIDATION nTriplets: " << nTriplets <<  std::endl;
}

//________________________________________________________________________________________________________________________________
void printT3s_for_CPU(SDL::CPU::Event& event)
{
    // get layer ptrs
    const std::vector<SDL::CPU::Layer*> layerPtrs = event.getLayerPtrs();

    // Loop over layers and access triplets
    for (auto& layerPtr : layerPtrs)
    {

        // MiniDoublet ptrs
        const std::vector<SDL::CPU::Triplet*>& tripletPtrs = layerPtr->getTripletPtrs();

        // Loop over triplet ptrs
        for (auto& tripletPtr : tripletPtrs)
        {

            // hit idx
            unsigned int hit0 = tripletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx();
            unsigned int hit1 = tripletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx();
            unsigned int hit2 = tripletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx();
            unsigned int hit3 = tripletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx();
            unsigned int hit4 = tripletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx();
            unsigned int hit5 = tripletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx();

            std::cout <<  "VALIDATION 'T3': " << "T3" <<  " hit0: " << hit0 <<  " hit1: " << hit1 <<  " hit2: " << hit2 <<  " hit3: " << hit3 <<  " hit4: " << hit4 <<  " hit5: " << hit5 <<  std::endl;

        }

    }

}

void printTCs_for_CPU(SDL::CPU::Event& event)
{
    // get layer ptrs
    std::vector<SDL::CPU::Layer*> layerPtrs = event.getLayerPtrs();
    layerPtrs.push_back(&(event.getPixelLayer()));

    // Loop over layers and access track candidates
    for (auto& layerPtr : layerPtrs)
    {

        // Track Candidate ptrs
        const std::vector<SDL::CPU::TrackCandidate*>& trackCandidatePtrs = layerPtr->getTrackCandidatePtrs();


        // Loop over trackCandidate ptrs
        for (auto& trackCandidatePtr : trackCandidatePtrs)
        {
            // hit idx
            unsigned int hit0 = trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx();
            unsigned int hit1 = trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx();
            unsigned int hit2 = trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx();
            unsigned int hit3 = trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx();
            unsigned int hit4 = trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx();
            unsigned int hit5 = trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx();
            unsigned int hit6 = trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx();
            unsigned int hit7 = trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx();
            unsigned int hit8 = trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx();
            unsigned int hit9 = trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx();
            unsigned int hit10 = trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx();
            unsigned int hit11 = trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx();

            std::cout <<  "VALIDATION 'TC': " << "TC" <<  " hit0: " << hit0 <<  " hit1: " << hit1 <<  " hit2: " << hit2 <<  " hit3: " << hit3 <<  " hit4: " << hit4 <<  " hit5: " << hit5 <<  " hit6: " << hit6 <<  " hit7: " << hit7 <<  " hit8: " << hit8 <<  " hit9: " << hit9 <<  " hit10: " << hit10 <<  " hit11: " << hit11 <<  std::endl;
        }

    }

}


//________________________________________________________________________________________________________________________________
void debugPrintOutlierMultiplicities(SDL::Event* event)
{
    SDL::trackCandidates& trackCandidatesInGPU = (*event->getTrackCandidates());
    SDL::triplets& tripletsInGPU = (*event->getTriplets());
    SDL::segments& segmentsInGPU = (*event->getSegments());
    SDL::miniDoublets& miniDoubletsInGPU = (*event->getMiniDoublets());
    SDL::hits& hitsInGPU = (*event->getHits());
    SDL::modules& modulesInGPU = (*event->getModules());
    SDL::objectRanges& rangesInGPU = (*event->getRanges());
    int nTrackCandidates = 0;
    for (unsigned int idx = 0; idx <= *(modulesInGPU.nLowerModules); ++idx)
    {
        if (trackCandidatesInGPU.nTrackCandidates[idx] > 50000)
        {
            std::cout <<  " SDL::modulesInGPU->detIds[SDL::modulesInGPU->lowerModuleIndices[idx]]: " << modulesInGPU.detIds[idx] <<  std::endl;
            std::cout <<  " idx: " << idx <<  " trackCandidatesInGPU.nTrackCandidates[idx]: " << trackCandidatesInGPU.nTrackCandidates[idx] <<  std::endl;
            std::cout <<  " idx: " << idx <<  " tripletsInGPU.nTriplets[idx]: " << tripletsInGPU.nTriplets[idx] <<  std::endl;
            unsigned int i = idx;//modulesInGPU.lowerModuleIndices[idx];
            std::cout <<  " idx: " << idx <<  " i: " << i <<  " segmentsInGPU.nSegments[i]: " << segmentsInGPU.nSegments[i] <<  std::endl;
            int nMD = miniDoubletsInGPU.nMDs[2*idx]+miniDoubletsInGPU.nMDs[2*idx+1] ;
            std::cout <<  " idx: " << idx <<  " nMD: " << nMD <<  std::endl;
            int nHits = 0;
            nHits += rangesInGPU.hitRanges[4*idx+1] - rangesInGPU.hitRanges[4*idx] + 1;       
            nHits += rangesInGPU.hitRanges[4*idx+3] - rangesInGPU.hitRanges[4*idx+2] + 1;
            std::cout <<  " idx: " << idx <<  " nHits: " << nHits <<  std::endl;
        }
    }
}
