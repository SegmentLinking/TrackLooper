#include "WriteSDLNtuplev2.h"

//____________________________________________________________________________________________
WriteSDLNtuplev2::WriteSDLNtuplev2(const char* studyName)
{
    studyname = studyName;
}

//____________________________________________________________________________________________
void WriteSDLNtuplev2::bookStudy()
{
    createHitsSimHitsSimTracksBranches();
    createPixelSeedBranches();
    createMiniDoubletBranches();
    createSegmentBranches();
    createPixelSegmentBranches();
    createTripletBranches();
    createQuadrupletBranches();
    createPixelQuadrupletBranches();
    createTrackCandidateBranches();
}

//____________________________________________________________________________________________
void WriteSDLNtuplev2::createHitsSimHitsSimTracksBranches()
{
    // Reco hits
    ana.tx->createBranch<vector<float>>("ph2_x");
    ana.tx->createBranch<vector<float>>("ph2_y");
    ana.tx->createBranch<vector<float>>("ph2_z");
    ana.tx->createBranch<vector<unsigned int>>("ph2_detId");
    ana.tx->createBranch<vector<vector<int>>>("ph2_simHitIdx");
    ana.tx->createBranch<vector<unsigned int>>("ph2_simType");
    ana.tx->createBranch<vector<int>>("ph2_anchorLayer");

    // Sim hits
    ana.tx->createBranch<vector<float>>("simhit_x");
    ana.tx->createBranch<vector<float>>("simhit_y");
    ana.tx->createBranch<vector<float>>("simhit_z");
    ana.tx->createBranch<vector<unsigned int>>("simhit_detId");
    ana.tx->createBranch<vector<unsigned int>>("simhit_partnerDetId");
    ana.tx->createBranch<vector<unsigned int>>("simhit_subdet");
    ana.tx->createBranch<vector<int>>("simhit_particle");
    ana.tx->createBranch<vector<vector<int>>>("simhit_hitIdx");
    ana.tx->createBranch<vector<int>>("simhit_simTrkIdx");

    // Sim tracks
    ana.tx->createBranch<vector<float>>("sim_pt");
    ana.tx->createBranch<vector<float>>("sim_eta");
    ana.tx->createBranch<vector<float>>("sim_phi");
    ana.tx->createBranch<vector<float>>("sim_pca_dxy");
    ana.tx->createBranch<vector<float>>("sim_pca_dz");
    ana.tx->createBranch<vector<int>>("sim_q");
    ana.tx->createBranch<vector<int>>("sim_event");
    ana.tx->createBranch<vector<int>>("sim_pdgId");
    ana.tx->createBranch<vector<int>>("sim_bunchCrossing");
    ana.tx->createBranch<vector<int>>("sim_hasAll12HitsInBarrel");
    ana.tx->createBranch<vector<vector<int>>>("sim_simHitIdx");
    ana.tx->createBranch<vector<vector<int>>>("sim_simHitLayer");
    ana.tx->createBranch<vector<vector<int>>>("sim_simHitBoth");
    ana.tx->createBranch<vector<vector<float>>>("sim_simHitDrFracWithHelix");
    ana.tx->createBranch<vector<vector<float>>>("sim_simHitDistXyWithHelix");

    // Sim vertex
    ana.tx->createBranch<vector<float>>("simvtx_x");
    ana.tx->createBranch<vector<float>>("simvtx_y");
    ana.tx->createBranch<vector<float>>("simvtx_z");
}

//____________________________________________________________________________________________
void WriteSDLNtuplev2::createPixelSeedBranches()
{
    ana.tx->createBranch<vector<float>>("see_stateTrajGlbPx");
    ana.tx->createBranch<vector<float>>("see_stateTrajGlbPy");
    ana.tx->createBranch<vector<float>>("see_stateTrajGlbPz");
    ana.tx->createBranch<vector<float>>("see_stateTrajGlbX");
    ana.tx->createBranch<vector<float>>("see_stateTrajGlbY");
    ana.tx->createBranch<vector<float>>("see_stateTrajGlbZ");
    ana.tx->createBranch<vector<float>>("see_px");
    ana.tx->createBranch<vector<float>>("see_py");
    ana.tx->createBranch<vector<float>>("see_pz");
    ana.tx->createBranch<vector<float>>("see_ptErr");
    ana.tx->createBranch<vector<float>>("see_dxy");
    ana.tx->createBranch<vector<float>>("see_dxyErr");
    ana.tx->createBranch<vector<float>>("see_dz");
    ana.tx->createBranch<vector<vector<int>>>("see_hitIdx");
    ana.tx->createBranch<vector<vector<int>>>("see_hitType");
    ana.tx->createBranch<vector<vector<int>>>("see_simTrkIdx");
    ana.tx->createBranch<vector<unsigned int>>("see_algo");
}

//____________________________________________________________________________________________
void WriteSDLNtuplev2::createMiniDoubletBranches()
{
    // MiniDoublets
    ana.tx->createBranch<vector<vector<int>>>("md_hitIdx");
    ana.tx->createBranch<vector<vector<int>>>("md_simTrkIdx");
    ana.tx->createBranch<vector<vector<int>>>("md_layer");
    // ana.tx->createBranch<vector<int>>("md_simHitIdx");

    // Kinematic quantity
    ana.tx->createBranch<vector<float>>("md_pt");
    ana.tx->createBranch<vector<float>>("md_eta");
    ana.tx->createBranch<vector<float>>("md_phi");
    ana.tx->createBranch<vector<float>>("md_sim_pt");
    ana.tx->createBranch<vector<float>>("md_sim_eta");
    ana.tx->createBranch<vector<float>>("md_sim_phi");

    ana.tx->createBranch<vector<float>>("md_type");
    ana.tx->createBranch<vector<float>>("md_dz");
    ana.tx->createBranch<vector<float>>("md_dzCut");
    ana.tx->createBranch<vector<float>>("md_drt");
    ana.tx->createBranch<vector<float>>("md_drtCut");
    ana.tx->createBranch<vector<float>>("md_miniCut");
    ana.tx->createBranch<vector<float>>("md_dphi");
    ana.tx->createBranch<vector<float>>("md_dphiChange");

    // Sim track to minidoublet matching
    ana.tx->createBranch<vector<vector<int>>>("sim_mdIdx");
    ana.tx->createBranch<vector<vector<int>>>("sim_mdIdx_isMTVmatch");

    // reco hit to minidoublet matching
    ana.tx->createBranch<vector<vector<int>>>("ph2_mdIdx");
}

//____________________________________________________________________________________________
void WriteSDLNtuplev2::createSegmentBranches()
{
    // Segments
    ana.tx->createBranch<vector<vector<int>>>("sg_hitIdx");
    ana.tx->createBranch<vector<vector<int>>>("sg_simTrkIdx");
    ana.tx->createBranch<vector<vector<int>>>("sg_layer");
    // ana.tx->createBranch<vector<int>>("sg_simHitIdx");

    // Kinematic quantity
    ana.tx->createBranch<vector<float>>("sg_pt");
    ana.tx->createBranch<vector<float>>("sg_eta");
    ana.tx->createBranch<vector<float>>("sg_phi");
    ana.tx->createBranch<vector<float>>("sg_sim_pt");
    ana.tx->createBranch<vector<float>>("sg_sim_eta");
    ana.tx->createBranch<vector<float>>("sg_sim_phi");

    // Sim track to Segment matching
    ana.tx->createBranch<vector<vector<int>>>("sim_sgIdx");
    ana.tx->createBranch<vector<vector<int>>>("sim_sgIdx_isMTVmatch");
}

//____________________________________________________________________________________________
void WriteSDLNtuplev2::createPixelSegmentBranches()
{
    // Segments
    ana.tx->createBranch<vector<vector<int>>>("psg_hitIdx");
    ana.tx->createBranch<vector<vector<int>>>("psg_simTrkIdx");
    ana.tx->createBranch<vector<vector<int>>>("psg_layer");
    // ana.tx->createBranch<vector<int>>("psg_simHitIdx");

    // Kinematic quantity
    ana.tx->createBranch<vector<float>>("psg_pt");
    ana.tx->createBranch<vector<float>>("psg_eta");
    ana.tx->createBranch<vector<float>>("psg_phi");
    ana.tx->createBranch<vector<float>>("psg_sim_pt");
    ana.tx->createBranch<vector<float>>("psg_sim_eta");
    ana.tx->createBranch<vector<float>>("psg_sim_phi");

    // Sim track to Segment matching
    ana.tx->createBranch<vector<vector<int>>>("sim_psgIdx");
    ana.tx->createBranch<vector<vector<int>>>("sim_psgIdx_isMTVmatch");
}

//____________________________________________________________________________________________
void WriteSDLNtuplev2::createTripletBranches()
{
    // Triplets
    ana.tx->createBranch<vector<vector<int>>>("tp_hitIdx");
    ana.tx->createBranch<vector<vector<int>>>("tp_simTrkIdx");
    ana.tx->createBranch<vector<vector<int>>>("tp_layer");
    // ana.tx->createBranch<vector<int>>("tp_simHitIdx");

    // Kinematic quantity
    ana.tx->createBranch<vector<float>>("tp_pt");
    ana.tx->createBranch<vector<float>>("tp_eta");
    ana.tx->createBranch<vector<float>>("tp_phi");
    ana.tx->createBranch<vector<float>>("tp_sim_pt");
    ana.tx->createBranch<vector<float>>("tp_sim_eta");
    ana.tx->createBranch<vector<float>>("tp_sim_phi");

    // Sim track to Triplets matching
    ana.tx->createBranch<vector<vector<int>>>("sim_tpIdx");
    ana.tx->createBranch<vector<vector<int>>>("sim_tpIdx_isMTVmatch");
}

//____________________________________________________________________________________________
void WriteSDLNtuplev2::createQuadrupletBranches()
{
    // Quadruplets
    ana.tx->createBranch<vector<vector<int>>>("qp_hitIdx");
    ana.tx->createBranch<vector<vector<int>>>("qp_simTrkIdx");
    ana.tx->createBranch<vector<vector<int>>>("qp_layer");
    // ana.tx->createBranch<vector<int>>("qp_simHitIdx");

    // Kinematic quantity
    ana.tx->createBranch<vector<float>>("qp_pt");
    ana.tx->createBranch<vector<float>>("qp_eta");
    ana.tx->createBranch<vector<float>>("qp_phi");
    ana.tx->createBranch<vector<float>>("qp_sim_pt");
    ana.tx->createBranch<vector<float>>("qp_sim_eta");
    ana.tx->createBranch<vector<float>>("qp_sim_phi");

    // Sim track to Quadruplets matching
    ana.tx->createBranch<vector<vector<int>>>("sim_qpIdx");
    ana.tx->createBranch<vector<vector<int>>>("sim_qpIdx_isMTVmatch");
}

//____________________________________________________________________________________________
void WriteSDLNtuplev2::createPixelQuadrupletBranches()
{
    // Quadruplets
    ana.tx->createBranch<vector<vector<int>>>("pqp_hitIdx");
    ana.tx->createBranch<vector<vector<int>>>("pqp_simTrkIdx");
    ana.tx->createBranch<vector<vector<int>>>("pqp_layer");
    // ana.tx->createBranch<vector<int>>("pqp_simHitIdx");

    // Kinematic quantity
    ana.tx->createBranch<vector<float>>("pqp_pt");
    ana.tx->createBranch<vector<float>>("pqp_eta");
    ana.tx->createBranch<vector<float>>("pqp_phi");
    ana.tx->createBranch<vector<float>>("pqp_sim_pt");
    ana.tx->createBranch<vector<float>>("pqp_sim_eta");
    ana.tx->createBranch<vector<float>>("pqp_sim_phi");

    // Sim track to Quadruplets matching
    ana.tx->createBranch<vector<vector<int>>>("sim_pqpIdx");
    ana.tx->createBranch<vector<vector<int>>>("sim_pqpIdx_isMTVmatch");
}

//____________________________________________________________________________________________
void WriteSDLNtuplev2::createTrackCandidateBranches()
{
    // Track Candidates
    ana.tx->createBranch<vector<vector<int>>>("tc_hitIdx");
    ana.tx->createBranch<vector<vector<int>>>("tc_simTrkIdx");
    ana.tx->createBranch<vector<vector<int>>>("tc_layer");
    // ana.tx->createBranch<vector<int>>("tc_simHitIdx");

    // Kinematic quantity
    ana.tx->createBranch<vector<float>>("tc_pt");
    ana.tx->createBranch<vector<float>>("tc_eta");
    ana.tx->createBranch<vector<float>>("tc_phi");
    ana.tx->createBranch<vector<float>>("tc_sim_pt");
    ana.tx->createBranch<vector<float>>("tc_sim_eta");
    ana.tx->createBranch<vector<float>>("tc_sim_phi");

    // Sim track to Track Candidates matching
    ana.tx->createBranch<vector<vector<int>>>("sim_tcIdx");
    ana.tx->createBranch<vector<vector<int>>>("sim_tcIdx_isMTVmatch");
}

//____________________________________________________________________________________________
void WriteSDLNtuplev2::doStudy(SDL::EventForAnalysisInterface& event, std::vector<std::tuple<unsigned int, SDL::EventForAnalysisInterface*>> simtrkevents)
{
    setHitsSimHitsSimTracksBranches();
    //setPixelSeedBranches();
    setMiniDoubletBranches(event);
    setSegmentBranches(event);
    //setPixelSegmentBranches(event);
    setTripletBranches(event);
    setQuadrupletBranches(event);
    //setPixelQuadrupletBranches(event);
    //setTrackCandidateBranches(event);
    ana.tx->fill();
    ana.tx->clear();
}

//____________________________________________________________________________________________
void WriteSDLNtuplev2::setHitsSimHitsSimTracksBranches()
{

    // Reco hits
    ana.tx->setBranch<vector<float>>("ph2_x", trk.ph2_x());
    ana.tx->setBranch<vector<float>>("ph2_y", trk.ph2_y());
    ana.tx->setBranch<vector<float>>("ph2_z", trk.ph2_z());
    ana.tx->setBranch<vector<unsigned int>>("ph2_detId", trk.ph2_detId());
    ana.tx->setBranch<vector<vector<int>>>("ph2_simHitIdx", trk.ph2_simHitIdx());
    vector<unsigned int> ph2_simType(trk.ph2_simType().begin(), trk.ph2_simType().end());
    ana.tx->setBranch<vector<unsigned int>>("ph2_simType", ph2_simType);
    vector<int> anchorLayer;
    for (unsigned int ihit = 0; ihit < trk.ph2_detId().size(); ++ihit)
    {
        int thisHitanchorLayer = 0;
        SDL::Module module(trk.ph2_detId()[ihit]);
        if (module.moduleType() == SDL::Module::PS and module.moduleLayerType() == SDL::Module::Pixel)
            thisHitanchorLayer = 1;
        if (module.moduleType() == SDL::Module::TwoS and module.isLower())
            thisHitanchorLayer = 1;
        anchorLayer.push_back(thisHitanchorLayer);
    }
    ana.tx->setBranch<vector<int>>("ph2_anchorLayer", anchorLayer);

    // Sim hits
    ana.tx->setBranch<vector<float>>("simhit_x", trk.simhit_x());
    ana.tx->setBranch<vector<float>>("simhit_y", trk.simhit_y());
    ana.tx->setBranch<vector<float>>("simhit_z", trk.simhit_z());
    ana.tx->setBranch<vector<unsigned int>>("simhit_detId", trk.simhit_detId());
    vector<unsigned int> simhit_partnerDetId;
    for (unsigned int imod = 0; imod < trk.simhit_detId().size(); imod++)
    {
        if (trk.simhit_subdet()[imod] == 4 or trk.simhit_subdet()[imod] == 5)
            simhit_partnerDetId.push_back(SDL::Module(trk.simhit_detId()[imod]).partnerDetId());
        else
            simhit_partnerDetId.push_back(-1);
    }
    ana.tx->setBranch<vector<unsigned int>>("simhit_partnerDetId", simhit_partnerDetId);
    vector<unsigned int> simhit_subdet(trk.simhit_subdet().begin(), trk.simhit_subdet().end());
    ana.tx->setBranch<vector<unsigned int>>("simhit_subdet", simhit_subdet);
    ana.tx->setBranch<vector<int>>("simhit_particle", trk.simhit_particle());
    ana.tx->setBranch<vector<vector<int>>>("simhit_hitIdx", trk.simhit_hitIdx());
    ana.tx->setBranch<vector<int>>("simhit_simTrkIdx", trk.simhit_simTrkIdx());

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
    vector<int> hasAll12HitsWithNBarrel;
    for (unsigned int isim = 0; isim < trk.sim_pt().size(); ++isim)
        hasAll12HitsWithNBarrel.push_back(hasAll12HitsWithNBarrelUsingModuleMap(isim, 6));
    ana.tx->setBranch<vector<int>>("sim_hasAll12HitsInBarrel", hasAll12HitsWithNBarrel);
    ana.tx->setBranch<vector<vector<int>>>("sim_simHitIdx", trk.sim_simHitIdx());

    // simvtx
    ana.tx->setBranch<vector<float>>("simvtx_x", trk.simvtx_x());
    ana.tx->setBranch<vector<float>>("simvtx_y", trk.simvtx_y());
    ana.tx->setBranch<vector<float>>("simvtx_z", trk.simvtx_z());

    //--------------------
    // Extra calculations 
    //--------------------

    // layer index with simhits
    for (auto&& [isimtrk, sim_simHitIdx] : iter::enumerate(trk.sim_simHitIdx()))
    {

        std::vector<int> sim_simHitLayer;
        std::vector<int> sim_simHitBoth; // Both side in the pt module
        std::vector<float> sim_simHitDrFracWithHelix;
        std::vector<float> sim_simHitDistXyWithHelix;
        for (auto& isimhitidx : sim_simHitIdx)
        {
            int subdet = trk.simhit_subdet()[isimhitidx];
            int islower = trk.simhit_isLower()[isimhitidx];
            int sign = islower == 1 ? -1 : 1;
            if (subdet != 4 and subdet != 5)
            {
                sim_simHitLayer.push_back(0);
                sim_simHitBoth.push_back(0);
            }
            else
            {
                sim_simHitLayer.push_back( sign * (trk.simhit_layer()[isimhitidx] + 6 * (subdet == 4)) );

                bool bothside = false;
                for (auto& jsimhitidx : sim_simHitIdx)
                {
                    if (SDL::Module(trk.simhit_detId()[isimhitidx]).partnerDetId() == trk.simhit_detId()[jsimhitidx])
                    {
                        bothside = true;
                    }
                }
                
                if (bothside)
                {
                    sim_simHitBoth.push_back(1);
                }
                else
                {
                    sim_simHitBoth.push_back(0);
                }
            }

            sim_simHitDrFracWithHelix.push_back(drfracSimHitConsistentWithHelix(isimtrk, isimhitidx));
            sim_simHitDistXyWithHelix.push_back(distxySimHitConsistentWithHelix(isimtrk, isimhitidx));

        }

        ana.tx->pushbackToBranch<vector<int>>("sim_simHitLayer", sim_simHitLayer);
        ana.tx->pushbackToBranch<vector<int>>("sim_simHitBoth", sim_simHitBoth);
        ana.tx->pushbackToBranch<vector<float>>("sim_simHitDrFracWithHelix", sim_simHitDrFracWithHelix);
        ana.tx->pushbackToBranch<vector<float>>("sim_simHitDistXyWithHelix", sim_simHitDistXyWithHelix);

    }

}

//____________________________________________________________________________________________
/*void WriteSDLNtuplev2::setPixelSeedBranches()
{
    // Reco pixel seeds
    ana.tx->setBranch<vector<float>>("see_stateTrajGlbPx", trk.see_stateTrajGlbPx());
    ana.tx->setBranch<vector<float>>("see_stateTrajGlbPy", trk.see_stateTrajGlbPy());
    ana.tx->setBranch<vector<float>>("see_stateTrajGlbPz", trk.see_stateTrajGlbPz());
    ana.tx->setBranch<vector<float>>("see_stateTrajGlbX", trk.see_stateTrajGlbX());
    ana.tx->setBranch<vector<float>>("see_stateTrajGlbY", trk.see_stateTrajGlbY());
    ana.tx->setBranch<vector<float>>("see_stateTrajGlbZ", trk.see_stateTrajGlbZ());
    ana.tx->setBranch<vector<float>>("see_px", trk.see_px());
    ana.tx->setBranch<vector<float>>("see_py", trk.see_py());
    ana.tx->setBranch<vector<float>>("see_pz", trk.see_pz());
    ana.tx->setBranch<vector<float>>("see_ptErr", trk.see_ptErr());
    ana.tx->setBranch<vector<float>>("see_dxy", trk.see_dxy());
    ana.tx->setBranch<vector<float>>("see_dxyErr", trk.see_dxyErr());
    ana.tx->setBranch<vector<float>>("see_dz", trk.see_dz());
    ana.tx->setBranch<vector<vector<int>>>("see_hitIdx", trk.see_hitIdx());
    ana.tx->setBranch<vector<vector<int>>>("see_hitType", trk.see_hitType());
    ana.tx->setBranch<vector<vector<int>>>("see_simTrkIdx", trk.see_simTrkIdx());
    ana.tx->setBranch<vector<unsigned int>>("see_algo", trk.see_algo());
}*/

//____________________________________________________________________________________________
void WriteSDLNtuplev2::setMiniDoubletBranches(SDL::EventForAnalysisInterface& event)
{

    // get layer ptrs
    const std::vector<SDL::Layer*> layerPtrs = event.getLayerPtrs();

    // sim track to minidoublet matching
    std::vector<vector<int>> sim_mdIdx(trk.sim_pt().size());
    std::vector<vector<int>> sim_mdIdx_isMTVmatch(trk.sim_pt().size());

    // reco hit to minidoublet matching
    std::vector<vector<int>> ph2_mdIdx(trk.ph2_detId().size());

    // Loop over layers and access minidoublets
    for (auto& layerPtr : layerPtrs)
    {

        // MiniDoublet ptrs
        const std::vector<SDL::MiniDoublet*>& minidoubletPtrs = layerPtr->getMiniDoubletPtrs();

        // Loop over minidoublet ptrs
        for (auto& minidoubletPtr : minidoubletPtrs)
        {

            // hit idx
            std::vector<int> hit_idx;
            hit_idx.push_back(minidoubletPtr->lowerHitPtr()->idx());
            hit_idx.push_back(minidoubletPtr->upperHitPtr()->idx());
            ana.tx->pushbackToBranch<vector<int>>("md_hitIdx", hit_idx);

            std::vector<int> hit_types;
            hit_types.push_back(4);
            hit_types.push_back(4);

            // sim track matched index
            std::vector<int> matched_sim_trk_idxs = matchedSimTrkIdxs(hit_idx, hit_types);

            ana.tx->pushbackToBranch<vector<int>>("md_simTrkIdx", matched_sim_trk_idxs);

            // tracklet layers
            std::vector<int> layers;
            layers.push_back(minidoubletPtr->lowerHitPtr()->getModule().layer() + 6 * (minidoubletPtr->lowerHitPtr()->getModule().subdet() == SDL::Module::Endcap));
            layers.push_back(minidoubletPtr->upperHitPtr()->getModule().layer() + 6 * (minidoubletPtr->upperHitPtr()->getModule().subdet() == SDL::Module::Endcap));
            ana.tx->pushbackToBranch<vector<int>>("md_layer", layers);

            // Sim track to mini-doublet matching
            int imd = ana.tx->getBranch<vector<vector<int>>>("md_hitIdx").size() - 1;

            // For matched sim track
            for (auto& matched_sim_trk_idx : matched_sim_trk_idxs)
            {
                sim_mdIdx[matched_sim_trk_idx].push_back(imd);
                std::vector<unsigned int> hit_idx_(hit_idx.begin(), hit_idx.end());
                if (isMTVMatch(matched_sim_trk_idx, hit_idx_))
                    sim_mdIdx_isMTVmatch[matched_sim_trk_idx].push_back(imd);
            }

            // Reco hit to mini-doublet matching
            // For matched sim track
            for (auto& ihit : hit_idx)
            {
                ph2_mdIdx[ihit].push_back(imd);
            }

            SDL::Hit hitA(trk.ph2_x()[hit_idx[0]], trk.ph2_y()[hit_idx[0]], trk.ph2_z()[hit_idx[0]]);
            SDL::Hit hitB(trk.ph2_x()[hit_idx[1]], trk.ph2_y()[hit_idx[1]], trk.ph2_z()[hit_idx[1]]);
            SDL::Hit hitC(0, 0, 0);
            SDL::Hit center = SDL::MathUtil::getCenterFromThreePoints(hitA, hitB, hitC);
            float radius = center.rt();
            float pt = SDL::MathUtil::ptEstimateFromRadius(radius);
            float eta = hitB.eta();
            float phi = hitA.phi();

            float maxsimpt = -999;
            float maxsimeta = 0;
            float maxsimphi = 0;
            for (auto& simtrkidx : matched_sim_trk_idxs)
            {
                float tmpsimpt = trk.sim_pt()[simtrkidx];
                float tmpsimeta = trk.sim_eta()[simtrkidx];
                float tmpsimphi = trk.sim_phi()[simtrkidx];
                if (maxsimpt < tmpsimpt)
                {
                    maxsimpt = tmpsimpt;
                    maxsimeta = tmpsimeta;
                    maxsimphi = tmpsimphi;
                }
            }

            // Kinematic quantity
            ana.tx->pushbackToBranch<float>("md_pt", pt);
            ana.tx->pushbackToBranch<float>("md_eta", eta);
            ana.tx->pushbackToBranch<float>("md_phi", phi);
            ana.tx->pushbackToBranch<float>("md_sim_pt", maxsimpt);
            ana.tx->pushbackToBranch<float>("md_sim_eta", maxsimeta);
            ana.tx->pushbackToBranch<float>("md_sim_phi", maxsimphi);

            float dz = minidoubletPtr->getDz();
            float dzCut = 0;//minidoubletPtr->getRecoVar("dzCut");
            float drt = 0;//minidoubletPtr->getRecoVar("drt");
            float drtCut = 0;//minidoubletPtr->getRecoVar("drtCut");
            float miniCut = 0;//minidoubletPtr->getRecoVar("miniCut");
            float dphi = minidoubletPtr->getDeltaPhi();
            float dphichange = minidoubletPtr->getDeltaPhiChange();
            float type = 0;//minidoubletPtr->getRecoVar("type");

            ana.tx->pushbackToBranch<float>("md_type", type);
            ana.tx->pushbackToBranch<float>("md_dz", dz);
            ana.tx->pushbackToBranch<float>("md_dzCut", dzCut);
            ana.tx->pushbackToBranch<float>("md_drt", drt);
            ana.tx->pushbackToBranch<float>("md_drtCut", drtCut);
            ana.tx->pushbackToBranch<float>("md_miniCut", miniCut);
            ana.tx->pushbackToBranch<float>("md_dphi", dphi);
            ana.tx->pushbackToBranch<float>("md_dphiChange", dphichange);

        }


    }

    ana.tx->setBranch<vector<vector<int>>>("sim_mdIdx", sim_mdIdx);
    ana.tx->setBranch<vector<vector<int>>>("sim_mdIdx_isMTVmatch", sim_mdIdx_isMTVmatch);
    ana.tx->setBranch<vector<vector<int>>>("ph2_mdIdx", ph2_mdIdx);

}

//____________________________________________________________________________________________
void WriteSDLNtuplev2::setSegmentBranches(SDL::EventForAnalysisInterface& event)
{

    // get layer ptrs
    const std::vector<SDL::Layer*> layerPtrs = event.getLayerPtrs();

    // sim track to segment matching
    std::vector<vector<int>> sim_sgIdx(trk.sim_pt().size());
    std::vector<vector<int>> sim_sgIdx_isMTVmatch(trk.sim_pt().size());

    // Loop over layers and access segments
    for (auto& layerPtr : layerPtrs)
    {

        // Segment ptrs
        const std::vector<SDL::Segment*>& segmentPtrs = layerPtr->getSegmentPtrs();

        // Loop over segment ptrs
        for (auto& segmentPtr : segmentPtrs)
        {

            // hit idx
            std::vector<int> hit_idx;
            hit_idx.push_back(segmentPtr->innerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(segmentPtr->innerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(segmentPtr->outerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(segmentPtr->outerMiniDoubletPtr()->upperHitPtr()->idx());
            ana.tx->pushbackToBranch<vector<int>>("sg_hitIdx", hit_idx);

            std::vector<int> hit_types;
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);

            // sim track matched index
            std::vector<int> matched_sim_trk_idxs = matchedSimTrkIdxs(hit_idx, hit_types);
            ana.tx->pushbackToBranch<vector<int>>("sg_simTrkIdx", matched_sim_trk_idxs);

            // tracklet layers
            std::vector<int> layers;
            layers.push_back(segmentPtr->innerMiniDoubletPtr()->lowerHitPtr()->getModule().layer() + 6 * (segmentPtr->innerMiniDoubletPtr()->lowerHitPtr()->getModule().subdet() == SDL::Module::Endcap));
            layers.push_back(segmentPtr->innerMiniDoubletPtr()->upperHitPtr()->getModule().layer() + 6 * (segmentPtr->innerMiniDoubletPtr()->upperHitPtr()->getModule().subdet() == SDL::Module::Endcap));
            layers.push_back(segmentPtr->outerMiniDoubletPtr()->lowerHitPtr()->getModule().layer() + 6 * (segmentPtr->outerMiniDoubletPtr()->lowerHitPtr()->getModule().subdet() == SDL::Module::Endcap));
            layers.push_back(segmentPtr->outerMiniDoubletPtr()->upperHitPtr()->getModule().layer() + 6 * (segmentPtr->outerMiniDoubletPtr()->upperHitPtr()->getModule().subdet() == SDL::Module::Endcap));
            ana.tx->pushbackToBranch<vector<int>>("sg_layer", layers);

            // Sim track to Segment matching
            int isg = ana.tx->getBranch<vector<vector<int>>>("sg_hitIdx").size() - 1;

            // For matched sim track
            for (auto& matched_sim_trk_idx : matched_sim_trk_idxs)
            {
                sim_sgIdx[matched_sim_trk_idx].push_back(isg);
                std::vector<unsigned int> hit_idx_(hit_idx.begin(), hit_idx.end());
                if (isMTVMatch(matched_sim_trk_idx, hit_idx_))
                    sim_sgIdx_isMTVmatch[matched_sim_trk_idx].push_back(isg);
            }

            SDL::Hit hitA(trk.ph2_x()[hit_idx[0]], trk.ph2_y()[hit_idx[0]], trk.ph2_z()[hit_idx[0]]);
            SDL::Hit hitB(trk.ph2_x()[hit_idx[3]], trk.ph2_y()[hit_idx[3]], trk.ph2_z()[hit_idx[3]]);
            SDL::Hit hitC(0, 0, 0);
            SDL::Hit center = SDL::MathUtil::getCenterFromThreePoints(hitA, hitB, hitC);
            float radius = center.rt();
            float pt = SDL::MathUtil::ptEstimateFromRadius(radius);
            float eta = hitB.eta();
            float phi = hitA.phi();

            float maxsimpt = -999;
            float maxsimeta = 0;
            float maxsimphi = 0;
            for (auto& simtrkidx : matched_sim_trk_idxs)
            {
                float tmpsimpt = trk.sim_pt()[simtrkidx];
                float tmpsimeta = trk.sim_eta()[simtrkidx];
                float tmpsimphi = trk.sim_phi()[simtrkidx];
                if (maxsimpt < tmpsimpt)
                {
                    maxsimpt = tmpsimpt;
                    maxsimeta = tmpsimeta;
                    maxsimphi = tmpsimphi;
                }
            }

            // Kinematic quantity
            ana.tx->pushbackToBranch<float>("sg_pt", pt);
            ana.tx->pushbackToBranch<float>("sg_eta", eta);
            ana.tx->pushbackToBranch<float>("sg_phi", phi);
            ana.tx->pushbackToBranch<float>("sg_sim_pt", maxsimpt);
            ana.tx->pushbackToBranch<float>("sg_sim_eta", maxsimeta);
            ana.tx->pushbackToBranch<float>("sg_sim_phi", maxsimphi);

        }

    }

    ana.tx->setBranch<vector<vector<int>>>("sim_sgIdx", sim_sgIdx);
    ana.tx->setBranch<vector<vector<int>>>("sim_sgIdx_isMTVmatch", sim_sgIdx_isMTVmatch);

}

//____________________________________________________________________________________________
/*void WriteSDLNtuplev2::setPixelSegmentBranches(SDL::EventForAnalysisInterface& event)
{

    // sim track to segment matching
    std::vector<vector<int>> sim_psgIdx(trk.sim_pt().size());
    std::vector<vector<int>> sim_psgIdx_isMTVmatch(trk.sim_pt().size());

    SDL::Layer* layerPtr = &event.getPixelLayer();

    // Segment ptrs
    const std::vector<SDL::Segment*>& segmentPtrs = layerPtr->getSegmentPtrs();

    // Loop over segment ptrs
    for (auto& segmentPtr : segmentPtrs)
    {

        // hit idx
        std::vector<int> hit_idx;
        hit_idx.push_back(segmentPtr->innerMiniDoubletPtr()->lowerHitPtr()->idx());
        hit_idx.push_back(segmentPtr->innerMiniDoubletPtr()->upperHitPtr()->idx());
        hit_idx.push_back(segmentPtr->outerMiniDoubletPtr()->lowerHitPtr()->idx());
        hit_idx.push_back(segmentPtr->outerMiniDoubletPtr()->upperHitPtr()->idx());
        ana.tx->pushbackToBranch<vector<int>>("psg_hitIdx", hit_idx);

        std::vector<int> hit_types;
        hit_types.push_back(0);
        hit_types.push_back(0);
        hit_types.push_back(0);
        hit_types.push_back(0);

        // sim track matched index
        std::vector<int> matched_sim_trk_idxs = matchedSimTrkIdxs(hit_idx, hit_types);
        ana.tx->pushbackToBranch<vector<int>>("psg_simTrkIdx", matched_sim_trk_idxs);

        // tracklet layers
        std::vector<int> layers;
        layers.push_back(0);
        layers.push_back(0);
        layers.push_back(0);
        layers.push_back(0);
        ana.tx->pushbackToBranch<vector<int>>("psg_layer", layers);

        // Sim track to Segment matching
        int ipsg = ana.tx->getBranch<vector<vector<int>>>("psg_hitIdx").size() - 1;

        // For matched sim track
        for (auto& matched_sim_trk_idx : matched_sim_trk_idxs)
        {
            sim_psgIdx[matched_sim_trk_idx].push_back(ipsg);
            std::vector<unsigned int> hit_idx_(hit_idx.begin(), hit_idx.end());
            if (isMTVMatch(matched_sim_trk_idx, hit_idx_))
                sim_psgIdx_isMTVmatch[matched_sim_trk_idx].push_back(ipsg);
        }

        SDL::Hit hitA(trk.ph2_x()[hit_idx[0]], trk.ph2_y()[hit_idx[0]], trk.ph2_z()[hit_idx[0]]);
        SDL::Hit hitB(trk.ph2_x()[hit_idx[3]], trk.ph2_y()[hit_idx[3]], trk.ph2_z()[hit_idx[3]]);
        SDL::Hit hitC(0, 0, 0);
        SDL::Hit center = SDL::MathUtil::getCenterFromThreePoints(hitA, hitB, hitC);
        float radius = center.rt();
        float pt = SDL::MathUtil::ptEstimateFromRadius(radius);
        float eta = hitB.eta();
        float phi = hitA.phi();

        float maxsimpt = -999;
        float maxsimeta = 0;
        float maxsimphi = 0;
        for (auto& simtrkidx : matched_sim_trk_idxs)
        {
            float tmpsimpt = trk.sim_pt()[simtrkidx];
            float tmpsimeta = trk.sim_eta()[simtrkidx];
            float tmpsimphi = trk.sim_phi()[simtrkidx];
            if (maxsimpt < tmpsimpt)
            {
                maxsimpt = tmpsimpt;
                maxsimeta = tmpsimeta;
                maxsimphi = tmpsimphi;
            }
        }

        // Kinematic quantity
        ana.tx->pushbackToBranch<float>("psg_pt", pt);
        ana.tx->pushbackToBranch<float>("psg_eta", eta);
        ana.tx->pushbackToBranch<float>("psg_phi", phi);
        ana.tx->pushbackToBranch<float>("psg_sim_pt", maxsimpt);
        ana.tx->pushbackToBranch<float>("psg_sim_eta", maxsimeta);
        ana.tx->pushbackToBranch<float>("psg_sim_phi", maxsimphi);

    }


    ana.tx->setBranch<vector<vector<int>>>("sim_psgIdx", sim_psgIdx);
    ana.tx->setBranch<vector<vector<int>>>("sim_psgIdx_isMTVmatch", sim_psgIdx_isMTVmatch);

}*/

//____________________________________________________________________________________________
void WriteSDLNtuplev2::setTripletBranches(SDL::EventForAnalysisInterface& event)
{

    // get layer ptrs
    const std::vector<SDL::Layer*> layerPtrs = event.getLayerPtrs();

    // sim track to triplet matching
    std::vector<vector<int>> sim_tpIdx(trk.sim_pt().size());
    std::vector<vector<int>> sim_tpIdx_isMTVmatch(trk.sim_pt().size());

    // Loop over layers and access triplets
    for (auto& layerPtr : layerPtrs)
    {

        // Triplet ptrs
        const std::vector<SDL::Triplet*>& tripletPtrs = layerPtr->getTripletPtrs();

        // Loop over triplet ptrs
        for (auto& tripletPtr : tripletPtrs)
        {

            // hit idx
            std::vector<int> hit_idx;
            hit_idx.push_back(tripletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(tripletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(tripletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(tripletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(tripletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(tripletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx());
            ana.tx->pushbackToBranch<vector<int>>("tp_hitIdx", hit_idx);

            std::vector<int> hit_types;
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);

            // sim track matched index
            std::vector<int> matched_sim_trk_idxs = matchedSimTrkIdxs(hit_idx, hit_types);
            ana.tx->pushbackToBranch<vector<int>>("tp_simTrkIdx", matched_sim_trk_idxs);

            // tracklet layers
            std::vector<int> layers;
            layers.push_back(tripletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().layer() + 6 * (tripletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().subdet() == SDL::Module::Endcap));
            layers.push_back(tripletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->getModule().layer() + 6 * (tripletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->getModule().subdet() == SDL::Module::Endcap));
            layers.push_back(tripletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule().layer() + 6 * (tripletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule().subdet() == SDL::Module::Endcap));
            layers.push_back(tripletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->getModule().layer() + 6 * (tripletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->getModule().subdet() == SDL::Module::Endcap));
            layers.push_back(tripletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule().layer() + 6 * (tripletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule().subdet() == SDL::Module::Endcap));
            layers.push_back(tripletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->getModule().layer() + 6 * (tripletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->getModule().subdet() == SDL::Module::Endcap));
            ana.tx->pushbackToBranch<vector<int>>("tp_layer", layers);

            // Sim track to Triplet matching
            int itp = ana.tx->getBranch<vector<vector<int>>>("tp_hitIdx").size() - 1;

            // For matched sim track
            for (auto& matched_sim_trk_idx : matched_sim_trk_idxs)
            {
                sim_tpIdx[matched_sim_trk_idx].push_back(itp);
                std::vector<unsigned int> hit_idx_(hit_idx.begin(), hit_idx.end());
                if (isMTVMatch(matched_sim_trk_idx, hit_idx_))
                    sim_tpIdx_isMTVmatch[matched_sim_trk_idx].push_back(itp);
            }

            SDL::Hit hitA(trk.ph2_x()[hit_idx[0]], trk.ph2_y()[hit_idx[0]], trk.ph2_z()[hit_idx[0]]);
            SDL::Hit hitB(trk.ph2_x()[hit_idx[5]], trk.ph2_y()[hit_idx[5]], trk.ph2_z()[hit_idx[5]]);
            SDL::Hit hitC(0, 0, 0);
            SDL::Hit center = SDL::MathUtil::getCenterFromThreePoints(hitA, hitB, hitC);
            float radius = center.rt();
            float pt = SDL::MathUtil::ptEstimateFromRadius(radius);
            float eta = hitB.eta();
            float phi = hitA.phi();

            float maxsimpt = -999;
            float maxsimeta = 0;
            float maxsimphi = 0;
            for (auto& simtrkidx : matched_sim_trk_idxs)
            {
                float tmpsimpt = trk.sim_pt()[simtrkidx];
                float tmpsimeta = trk.sim_eta()[simtrkidx];
                float tmpsimphi = trk.sim_phi()[simtrkidx];
                if (maxsimpt < tmpsimpt)
                {
                    maxsimpt = tmpsimpt;
                    maxsimeta = tmpsimeta;
                    maxsimphi = tmpsimphi;
                }
            }

            // Kinematic quantity
            ana.tx->pushbackToBranch<float>("tp_pt", pt);
            ana.tx->pushbackToBranch<float>("tp_eta", eta);
            ana.tx->pushbackToBranch<float>("tp_phi", phi);
            ana.tx->pushbackToBranch<float>("tp_sim_pt", maxsimpt);
            ana.tx->pushbackToBranch<float>("tp_sim_eta", maxsimeta);
            ana.tx->pushbackToBranch<float>("tp_sim_phi", maxsimphi);

        }

    }

    ana.tx->setBranch<vector<vector<int>>>("sim_tpIdx", sim_tpIdx);
    ana.tx->setBranch<vector<vector<int>>>("sim_tpIdx_isMTVmatch", sim_tpIdx_isMTVmatch);

}

//____________________________________________________________________________________________
void WriteSDLNtuplev2::setQuadrupletBranches(SDL::EventForAnalysisInterface& event)
{

    // get layer ptrs
    const std::vector<SDL::Layer*> layerPtrs = event.getLayerPtrs();

    // sim track to tracklet matching
    std::vector<vector<int>> sim_qpIdx(trk.sim_pt().size());
    std::vector<vector<int>> sim_qpIdx_isMTVmatch(trk.sim_pt().size());

    // Loop over layers and access tracklets
    for (auto& layerPtr : layerPtrs)
    {

        // Quadruplet ptrs
        const std::vector<SDL::Tracklet*>& trackletPtrs = layerPtr->getTrackletPtrs();

        // Loop over tracklet ptrs
        for (auto& trackletPtr : trackletPtrs)
        {

            // hit idx
            std::vector<int> hit_idx;
            hit_idx.push_back(trackletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx());
            ana.tx->pushbackToBranch<vector<int>>("qp_hitIdx", hit_idx);

            std::vector<int> hit_types;
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);
            hit_types.push_back(4);

            // sim track matched index
            std::vector<int> matched_sim_trk_idxs = matchedSimTrkIdxs(hit_idx, hit_types);
            ana.tx->pushbackToBranch<vector<int>>("qp_simTrkIdx", matched_sim_trk_idxs);

            // tracklet layers
            std::vector<int> layers;
            layers.push_back(trackletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().layer() + 6 * (trackletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().subdet() == SDL::Module::Endcap));
            layers.push_back(trackletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule().layer() + 6 * (trackletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule().subdet() == SDL::Module::Endcap));
            layers.push_back(trackletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().layer() + 6 * (trackletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().subdet() == SDL::Module::Endcap));
            layers.push_back(trackletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule().layer() + 6 * (trackletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule().subdet() == SDL::Module::Endcap));
            ana.tx->pushbackToBranch<vector<int>>("qp_layer", layers);

            // Sim track to Quadruplet matching
            int iqp = ana.tx->getBranch<vector<vector<int>>>("qp_hitIdx").size() - 1;

            // For matched sim track
            for (auto& matched_sim_trk_idx : matched_sim_trk_idxs)
            {
                sim_qpIdx[matched_sim_trk_idx].push_back(iqp);
                std::vector<unsigned int> hit_idx_(hit_idx.begin(), hit_idx.end());
                if (isMTVMatch(matched_sim_trk_idx, hit_idx_))
                    sim_qpIdx_isMTVmatch[matched_sim_trk_idx].push_back(iqp);
            }

            SDL::Hit hitA(trk.ph2_x()[hit_idx[0]], trk.ph2_y()[hit_idx[0]], trk.ph2_z()[hit_idx[0]]);
            SDL::Hit hitB(trk.ph2_x()[hit_idx[7]], trk.ph2_y()[hit_idx[7]], trk.ph2_z()[hit_idx[7]]);
            SDL::Hit hitC(0, 0, 0);
            SDL::Hit center = SDL::MathUtil::getCenterFromThreePoints(hitA, hitB, hitC);
            float radius = center.rt();
            float pt = SDL::MathUtil::ptEstimateFromRadius(radius);
            float eta = hitB.eta();
            float phi = hitA.phi();

            float maxsimpt = -999;
            float maxsimeta = 0;
            float maxsimphi = 0;
            for (auto& simtrkidx : matched_sim_trk_idxs)
            {
                float tmpsimpt = trk.sim_pt()[simtrkidx];
                float tmpsimeta = trk.sim_eta()[simtrkidx];
                float tmpsimphi = trk.sim_phi()[simtrkidx];
                if (maxsimpt < tmpsimpt)
                {
                    maxsimpt = tmpsimpt;
                    maxsimeta = tmpsimeta;
                    maxsimphi = tmpsimphi;
                }
            }

            // Kinematic quantity
            ana.tx->pushbackToBranch<float>("qp_pt", pt);
            ana.tx->pushbackToBranch<float>("qp_eta", eta);
            ana.tx->pushbackToBranch<float>("qp_phi", phi);
            ana.tx->pushbackToBranch<float>("qp_sim_pt", maxsimpt);
            ana.tx->pushbackToBranch<float>("qp_sim_eta", maxsimeta);
            ana.tx->pushbackToBranch<float>("qp_sim_phi", maxsimphi);

        }

    }

    ana.tx->setBranch<vector<vector<int>>>("sim_qpIdx", sim_qpIdx);
    ana.tx->setBranch<vector<vector<int>>>("sim_qpIdx_isMTVmatch", sim_qpIdx_isMTVmatch);

    // // Tracklets
    // ana.tx->setBranch<vector<int>>("qp_hitIdx");
    // ana.tx->setBranch<vector<int>>("qp_simHitIdx");
    // ana.tx->setBranch<vector<int>>("qp_simTrkIdx");

}

//____________________________________________________________________________________________
/*void WriteSDLNtuplev2::setPixelQuadrupletBranches(SDL::EventForAnalysisInterface& event)
{

    // get pixel layer ptrs
    const SDL::Layer& pixelLayer = event.getPixelLayer();

    // sim track to tracklet matching
    std::vector<vector<int>> sim_pqpIdx(trk.sim_pt().size());
    std::vector<vector<int>> sim_pqpIdx_isMTVmatch(trk.sim_pt().size());

    // Quadruplet ptrs
    const std::vector<SDL::Tracklet*>& trackletPtrs = pixelLayer.getTrackletPtrs();

    // Loop over tracklet ptrs
    for (auto& trackletPtr : trackletPtrs)
    {

        // hit idx
        std::vector<int> hit_idx;
        hit_idx.push_back(trackletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx());
        hit_idx.push_back(trackletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx());
        hit_idx.push_back(trackletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx());
        hit_idx.push_back(trackletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx());
        hit_idx.push_back(trackletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx());
        hit_idx.push_back(trackletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx());
        hit_idx.push_back(trackletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx());
        hit_idx.push_back(trackletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx());
        ana.tx->pushbackToBranch<vector<int>>("pqp_hitIdx", hit_idx);

        std::vector<int> hit_types;
        hit_types.push_back(0);
        hit_types.push_back(0);
        hit_types.push_back(0);
        hit_types.push_back(0);
        hit_types.push_back(4);
        hit_types.push_back(4);
        hit_types.push_back(4);
        hit_types.push_back(4);

        // sim track matched index
        std::vector<int> matched_sim_trk_idxs = matchedSimTrkIdxs(hit_idx, hit_types);
        ana.tx->pushbackToBranch<vector<int>>("pqp_simTrkIdx", matched_sim_trk_idxs);

        // tracklet layers
        std::vector<int> layers;
        layers.push_back(trackletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().layer() + 6 * (trackletPtr->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().subdet() == SDL::Module::Endcap));
        layers.push_back(trackletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule().layer() + 6 * (trackletPtr->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule().subdet() == SDL::Module::Endcap));
        layers.push_back(trackletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().layer() + 6 * (trackletPtr->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().subdet() == SDL::Module::Endcap));
        layers.push_back(trackletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule().layer() + 6 * (trackletPtr->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule().subdet() == SDL::Module::Endcap));
        ana.tx->pushbackToBranch<vector<int>>("pqp_layer", layers);

        // Sim track to Quadruplet matching
        int ipqp = ana.tx->getBranch<vector<vector<int>>>("pqp_hitIdx").size() - 1;

        // For matched sim track
        for (auto& matched_sim_trk_idx : matched_sim_trk_idxs)
        {
            sim_pqpIdx[matched_sim_trk_idx].push_back(ipqp);
            std::vector<unsigned int> hit_idx_(hit_idx.begin(), hit_idx.end());
            if (isMTVMatch(matched_sim_trk_idx, hit_idx_))
                sim_pqpIdx_isMTVmatch[matched_sim_trk_idx].push_back(ipqp);
        }

        SDL::Hit hitA(trk.ph2_x()[hit_idx[0]], trk.ph2_y()[hit_idx[0]], trk.ph2_z()[hit_idx[0]]);
        SDL::Hit hitB(trk.ph2_x()[hit_idx[7]], trk.ph2_y()[hit_idx[7]], trk.ph2_z()[hit_idx[7]]);
        SDL::Hit hitC(0, 0, 0);
        SDL::Hit center = SDL::MathUtil::getCenterFromThreePoints(hitA, hitB, hitC);
        float radius = center.rt();
        float pt = SDL::MathUtil::ptEstimateFromRadius(radius);
        float eta = hitB.eta();
        float phi = hitA.phi();

        float maxsimpt = -999;
        float maxsimeta = 0;
        float maxsimphi = 0;
        for (auto& simtrkidx : matched_sim_trk_idxs)
        {
            float tmpsimpt = trk.sim_pt()[simtrkidx];
            float tmpsimeta = trk.sim_eta()[simtrkidx];
            float tmpsimphi = trk.sim_phi()[simtrkidx];
            if (maxsimpt < tmpsimpt)
            {
                maxsimpt = tmpsimpt;
                maxsimeta = tmpsimeta;
                maxsimphi = tmpsimphi;
            }
        }

        // Kinematic quantity
        ana.tx->pushbackToBranch<float>("pqp_pt", pt);
        ana.tx->pushbackToBranch<float>("pqp_eta", eta);
        ana.tx->pushbackToBranch<float>("pqp_phi", phi);
        ana.tx->pushbackToBranch<float>("pqp_sim_pt", maxsimpt);
        ana.tx->pushbackToBranch<float>("pqp_sim_eta", maxsimeta);
        ana.tx->pushbackToBranch<float>("pqp_sim_phi", maxsimphi);

    }


    ana.tx->setBranch<vector<vector<int>>>("sim_pqpIdx", sim_pqpIdx);
    ana.tx->setBranch<vector<vector<int>>>("sim_pqpIdx_isMTVmatch", sim_pqpIdx_isMTVmatch);

}

//____________________________________________________________________________________________
void WriteSDLNtuplev2::setTrackCandidateBranches(SDL::EventForAnalysisInterface& event)
{

    // get layer ptrs
    const std::vector<SDL::Layer*> layerPtrs = event.getLayerPtrs();

    // sim track to track candidate matching
    std::vector<vector<int>> sim_tcIdx(trk.sim_pt().size());
    std::vector<vector<int>> sim_tcIdx_isMTVmatch(trk.sim_pt().size());

    // Loop over layers and access track candidates
    for (auto& layerPtr : layerPtrs)
    {

        // Track Candidate ptrs
        const std::vector<SDL::TrackCandidate*>& trackCandidatePtrs = layerPtr->getTrackCandidatePtrs();

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
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx());
            hit_idx.push_back(trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx());
            ana.tx->pushbackToBranch<vector<int>>("tc_hitIdx", hit_idx);

            std::vector<int> hit_types;
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

            // sim track matched index
            std::vector<int> matched_sim_trk_idxs = matchedSimTrkIdxs(hit_idx, hit_types);
            ana.tx->pushbackToBranch<vector<int>>("tc_simTrkIdx", matched_sim_trk_idxs);

            // trackCandidate layers
            std::vector<int> layers;
            layers.push_back(trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().layer() + 6 * (trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().subdet() == SDL::Module::Endcap));
            layers.push_back(trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule().layer() + 6 * (trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule().subdet() == SDL::Module::Endcap));
            layers.push_back(trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().layer() + 6 * (trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().subdet() == SDL::Module::Endcap));
            layers.push_back(trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule().layer() + 6 * (trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule().subdet() == SDL::Module::Endcap));
            layers.push_back(trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().layer() + 6 * (trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().subdet() == SDL::Module::Endcap));
            layers.push_back(trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule().layer() + 6 * (trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule().subdet() == SDL::Module::Endcap));
            ana.tx->pushbackToBranch<vector<int>>("tc_layer", layers);

            // Sim track to Track Candidate matching
            int itc = ana.tx->getBranch<vector<vector<int>>>("tc_hitIdx").size() - 1;

            // For matched sim track
            for (auto& matched_sim_trk_idx : matched_sim_trk_idxs)
            {
                sim_tcIdx[matched_sim_trk_idx].push_back(itc);
                std::vector<unsigned int> hit_idx_(hit_idx.begin(), hit_idx.end());
                if (isMTVMatch(matched_sim_trk_idx, hit_idx_))
                    sim_tcIdx_isMTVmatch[matched_sim_trk_idx].push_back(itc);
            }

            SDL::Hit hitA(trk.ph2_x()[hit_idx[0]], trk.ph2_y()[hit_idx[0]], trk.ph2_z()[hit_idx[0]]);
            SDL::Hit hitB(trk.ph2_x()[hit_idx[11]], trk.ph2_y()[hit_idx[11]], trk.ph2_z()[hit_idx[11]]);
            SDL::Hit hitC(0, 0, 0);
            SDL::Hit center = SDL::MathUtil::getCenterFromThreePoints(hitA, hitB, hitC);
            float radius = center.rt();
            float pt = SDL::MathUtil::ptEstimateFromRadius(radius);
            float eta = hitB.eta();
            float phi = hitA.phi();

            float maxsimpt = -999;
            float maxsimeta = 0;
            float maxsimphi = 0;
            for (auto& simtrkidx : matched_sim_trk_idxs)
            {
                float tmpsimpt = trk.sim_pt()[simtrkidx];
                float tmpsimeta = trk.sim_eta()[simtrkidx];
                float tmpsimphi = trk.sim_phi()[simtrkidx];
                if (maxsimpt < tmpsimpt)
                {
                    maxsimpt = tmpsimpt;
                    maxsimeta = tmpsimeta;
                    maxsimphi = tmpsimphi;
                }
            }

            // Kinematic quantity
            ana.tx->pushbackToBranch<float>("tc_pt", pt);
            ana.tx->pushbackToBranch<float>("tc_eta", eta);
            ana.tx->pushbackToBranch<float>("tc_phi", phi);
            ana.tx->pushbackToBranch<float>("tc_sim_pt", maxsimpt);
            ana.tx->pushbackToBranch<float>("tc_sim_eta", maxsimeta);
            ana.tx->pushbackToBranch<float>("tc_sim_phi", maxsimphi);

        }

    }

    ana.tx->setBranch<vector<vector<int>>>("sim_tcIdx", sim_tcIdx);
    ana.tx->setBranch<vector<vector<int>>>("sim_tcIdx_isMTVmatch", sim_tcIdx_isMTVmatch);

}*/
