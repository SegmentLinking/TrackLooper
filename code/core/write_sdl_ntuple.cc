#include "write_sdl_ntuple.h"

//________________________________________________________________________________________________________________________________
void createOutputBranches()
{
    createRequiredOutputBranches();
    createOptionalOutputBranches();
}

//________________________________________________________________________________________________________________________________
void fillOutputBranches(SDL::Event* event)
{
    setOutputBranches(event);
    setOptionalOutputBranches(event);
    if (ana.gnn_ntuple)
        setGnnNtupleBranches(event);

    // Now actually fill the ttree
    ana.tx->fill();

    // Then clear the branches to default values (e.g. -999, or clear the vectors to empty vectors)
    ana.tx->clear();
}

//________________________________________________________________________________________________________________________________
void createRequiredOutputBranches()
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
    ana.tx->createBranch<vector<float>>("sim_vx");
    ana.tx->createBranch<vector<float>>("sim_vy");
    ana.tx->createBranch<vector<float>>("sim_vz");
    ana.tx->createBranch<vector<float>>("sim_trkNtupIdx");
    ana.tx->createBranch<vector<int>>("sim_TC_matched");
    ana.tx->createBranch<vector<int>>("sim_TC_matched_mask");

    // Track candidates
    ana.tx->createBranch<vector<float>>("tc_pt");
    ana.tx->createBranch<vector<float>>("tc_eta");
    ana.tx->createBranch<vector<float>>("tc_phi");
    ana.tx->createBranch<vector<int>>("tc_type");
    ana.tx->createBranch<vector<int>>("tc_isFake");
    ana.tx->createBranch<vector<int>>("tc_isDuplicate");
    ana.tx->createBranch<vector<vector<int>>>("tc_matched_simIdx");
}

//________________________________________________________________________________________________________________________________
void createOptionalOutputBranches()
{
#ifdef CUT_VALUE_DEBUG
    // Event-wide branches
    // ana.tx->createBranch<float>("evt_dummy");

    // Sim Track branches
    // NOTE: Must sync with main tc branch in length!!
    ana.tx->createBranch<vector<float>>("sim_dummy");

    // Track Candidate branches
    // NOTE: Must sync with main tc branch in length!!
    ana.tx->createBranch<vector<float>>("tc_dummy");

    // pT5 branches
    ana.tx->createBranch<vector<vector<int>>>("pT5_matched_simIdx");
    ana.tx->createBranch<vector<vector<int>>>("pT5_hitIdxs");
    ana.tx->createBranch<vector<int>>("sim_pT5_matched");
    ana.tx->createBranch<vector<float>>("pT5_pt");
    ana.tx->createBranch<vector<float>>("pT5_eta");
    ana.tx->createBranch<vector<float>>("pT5_phi");
    ana.tx->createBranch<vector<int>>("pT5_isFake");
    ana.tx->createBranch<vector<int>>("pT5_isDuplicate");
    ana.tx->createBranch<vector<int>>("pT5_score");
    ana.tx->createBranch<vector<int>>("pT5_layer_binary");
    ana.tx->createBranch<vector<int>>("pT5_moduleType_binary");
    ana.tx->createBranch<vector<float>>("pT5_matched_pt");
    ana.tx->createBranch<vector<float>>("pT5_rzChiSquared");
    ana.tx->createBranch<vector<float>>("pT5_rPhiChiSquared");
    ana.tx->createBranch<vector<float>>("pT5_rPhiChiSquaredInwards");

    // pT3 branches
    ana.tx->createBranch<vector<int>>("sim_pT3_matched");
    ana.tx->createBranch<vector<float>>("pT3_pt");
    ana.tx->createBranch<vector<int>>("pT3_isFake");
    ana.tx->createBranch<vector<int>>("pT3_isDuplicate");
    ana.tx->createBranch<vector<float>>("pT3_eta");
    ana.tx->createBranch<vector<float>>("pT3_phi");
    ana.tx->createBranch<vector<float>>("pT3_score");
    ana.tx->createBranch<vector<int>>("pT3_foundDuplicate");
    ana.tx->createBranch<vector<vector<int>>>("pT3_matched_simIdx");
    ana.tx->createBranch<vector<vector<int>>>("pT3_hitIdxs");
    ana.tx->createBranch<vector<float>>("pT3_pixelRadius");
    ana.tx->createBranch<vector<float>>("pT3_pixelRadiusError");
    ana.tx->createBranch<vector<vector<float>>>("pT3_matched_pt");
    ana.tx->createBranch<vector<float>>("pT3_tripletRadius");
    ana.tx->createBranch<vector<float>>("pT3_rPhiChiSquared");
    ana.tx->createBranch<vector<float>>("pT3_rPhiChiSquaredInwards");
    ana.tx->createBranch<vector<float>>("pT3_rzChiSquared");
    ana.tx->createBranch<vector<int>>("pT3_layer_binary");
    ana.tx->createBranch<vector<int>>("pT3_moduleType_binary");

    // pLS branches
    ana.tx->createBranch<vector<int>>("sim_pLS_matched");
    ana.tx->createBranch<vector<vector<int>>>("sim_pLS_types");
    ana.tx->createBranch<vector<int>>("pLS_isFake");
    ana.tx->createBranch<vector<int>>("pLS_isDuplicate");
    ana.tx->createBranch<vector<float>>("pLS_pt");
    ana.tx->createBranch<vector<float>>("pLS_eta");
    ana.tx->createBranch<vector<float>>("pLS_phi");
    ana.tx->createBranch<vector<float>>("pLS_score");

    // T5 branches
    ana.tx->createBranch<vector<int>>("sim_T5_matched");
    ana.tx->createBranch<vector<int>>("t5_isFake");
    ana.tx->createBranch<vector<int>>("t5_isDuplicate");
    ana.tx->createBranch<vector<int>>("t5_foundDuplicate");
    ana.tx->createBranch<vector<float>>("t5_pt");
    ana.tx->createBranch<vector<float>>("t5_eta");
    ana.tx->createBranch<vector<float>>("t5_phi");
    ana.tx->createBranch<vector<float>>("t5_score_rphisum");
    ana.tx->createBranch<vector<vector<int>>>("t5_hitIdxs");
    ana.tx->createBranch<vector<vector<int>>>("t5_matched_simIdx");
    ana.tx->createBranch<vector<int>>("t5_moduleType_binary");
    ana.tx->createBranch<vector<int>>("t5_layer_binary");
    ana.tx->createBranch<vector<float>>("t5_matched_pt");
    ana.tx->createBranch<vector<int>>("t5_partOfTC");
    ana.tx->createBranch<vector<float>>("t5_innerRadius");
    ana.tx->createBranch<vector<float>>("t5_outerRadius");
    ana.tx->createBranch<vector<float>>("t5_bridgeRadius");
    ana.tx->createBranch<vector<float>>("t5_chiSquared");
    ana.tx->createBranch<vector<float>>("t5_rzChiSquared");
    ana.tx->createBranch<vector<float>>("t5_nonAnchorChiSquared");

#endif
}

//________________________________________________________________________________________________________________________________
void createGnnNtupleBranches()
{
    // Mini Doublets
    ana.tx->createBranch<vector<float>>("MD_pt");
    ana.tx->createBranch<vector<float>>("MD_eta");
    ana.tx->createBranch<vector<float>>("MD_phi");
    ana.tx->createBranch<vector<float>>("MD_dphichange");
    ana.tx->createBranch<vector<int>>("MD_isFake");
    ana.tx->createBranch<vector<int>>("MD_tpType");
    ana.tx->createBranch<vector<int>>("MD_detId");
    ana.tx->createBranch<vector<int>>("MD_layer");
    ana.tx->createBranch<vector<float>>("MD_0_r");
    ana.tx->createBranch<vector<float>>("MD_0_x");
    ana.tx->createBranch<vector<float>>("MD_0_y");
    ana.tx->createBranch<vector<float>>("MD_0_z");
    ana.tx->createBranch<vector<float>>("MD_1_r");
    ana.tx->createBranch<vector<float>>("MD_1_x");
    ana.tx->createBranch<vector<float>>("MD_1_y");
    ana.tx->createBranch<vector<float>>("MD_1_z");

    // Line Segments
    ana.tx->createBranch<vector<float>>("LS_pt");
    ana.tx->createBranch<vector<float>>("LS_eta");
    ana.tx->createBranch<vector<float>>("LS_phi");
    ana.tx->createBranch<vector<int>>("LS_isFake");
    ana.tx->createBranch<vector<int>>("LS_MD_idx0");
    ana.tx->createBranch<vector<int>>("LS_MD_idx1");
    ana.tx->createBranch<vector<float>>("LS_sim_pt");
    ana.tx->createBranch<vector<float>>("LS_sim_eta");
    ana.tx->createBranch<vector<float>>("LS_sim_phi");
    ana.tx->createBranch<vector<float>>("LS_sim_pca_dxy");
    ana.tx->createBranch<vector<float>>("LS_sim_pca_dz");
    ana.tx->createBranch<vector<int>>("LS_sim_q");
    ana.tx->createBranch<vector<int>>("LS_sim_pdgId");
    ana.tx->createBranch<vector<int>>("LS_sim_event");
    ana.tx->createBranch<vector<int>>("LS_sim_bx");
    ana.tx->createBranch<vector<float>>("LS_sim_vx");
    ana.tx->createBranch<vector<float>>("LS_sim_vy");
    ana.tx->createBranch<vector<float>>("LS_sim_vz");
    ana.tx->createBranch<vector<int>>("LS_isInTrueTC");

    // T3 branches
    ana.tx->createBranch<vector<int>>("t3_isFake");
    ana.tx->createBranch<vector<int>>("t5_t3_idx0");
    ana.tx->createBranch<vector<int>>("t5_t3_idx1");
    ana.tx->createBranch<vector<float>>("t3_ptAv");
    ana.tx->createBranch<vector<float>>("t3_pt");
    ana.tx->createBranch<vector<float>>("t3_eta");
    ana.tx->createBranch<vector<float>>("t3_phi");
    ana.tx->createBranch<vector<float>>("t3_0_r");
    ana.tx->createBranch<vector<float>>("t3_0_dr");
    ana.tx->createBranch<vector<float>>("t3_0_x");
    ana.tx->createBranch<vector<float>>("t3_0_y");
    ana.tx->createBranch<vector<float>>("t3_0_z");
    ana.tx->createBranch<vector<float>>("t3_2_r");
    ana.tx->createBranch<vector<float>>("t3_2_dr");
    ana.tx->createBranch<vector<float>>("t3_2_x");
    ana.tx->createBranch<vector<float>>("t3_2_y");
    ana.tx->createBranch<vector<float>>("t3_2_z");
    ana.tx->createBranch<vector<float>>("t3_4_r");
    ana.tx->createBranch<vector<float>>("t3_4_dr");
    ana.tx->createBranch<vector<float>>("t3_4_x");
    ana.tx->createBranch<vector<float>>("t3_4_y");
    ana.tx->createBranch<vector<float>>("t3_4_z");

    // TC's LS
    ana.tx->createBranch<vector<vector<int>>>("tc_lsIdx");
}

//________________________________________________________________________________________________________________________________
void setOutputBranches(SDL::Event* event)
{

    // ============ Sim tracks =============
    int n_accepted_simtrk = 0;
    for (unsigned int isimtrk = 0; isimtrk < trk.sim_pt().size(); ++isimtrk)
    {
        // Skip out-of-time pileup
        if (trk.sim_bunchCrossing()[isimtrk] != 0)
            continue;

        // Skip non-hard-scatter
        if (trk.sim_event()[isimtrk] != 0)
            continue;

        ana.tx->pushbackToBranch<float>("sim_pt", trk.sim_pt()[isimtrk]);
        ana.tx->pushbackToBranch<float>("sim_eta", trk.sim_eta()[isimtrk]);
        ana.tx->pushbackToBranch<float>("sim_phi", trk.sim_phi()[isimtrk]);
        ana.tx->pushbackToBranch<float>("sim_pca_dxy", trk.sim_pca_dxy()[isimtrk]);
        ana.tx->pushbackToBranch<float>("sim_pca_dz", trk.sim_pca_dz()[isimtrk]);
        ana.tx->pushbackToBranch<int>("sim_q", trk.sim_q()[isimtrk]);
        ana.tx->pushbackToBranch<int>("sim_event", trk.sim_event()[isimtrk]);
        ana.tx->pushbackToBranch<int>("sim_pdgId", trk.sim_pdgId()[isimtrk]);

        // For vertex we need to look it up from simvtx info
        int vtxidx = trk.sim_parentVtxIdx()[isimtrk];
        ana.tx->pushbackToBranch<float>("sim_vx", trk.simvtx_x()[vtxidx]);
        ana.tx->pushbackToBranch<float>("sim_vy", trk.simvtx_y()[vtxidx]);
        ana.tx->pushbackToBranch<float>("sim_vz", trk.simvtx_z()[vtxidx]);

        // The trkNtupIdx is the idx in the trackingNtuple
        ana.tx->pushbackToBranch<float>("sim_trkNtupIdx", isimtrk);

        // Increase the counter for accepted simtrk
        n_accepted_simtrk++;
    }

    // Intermediate variables to keep track of matched track candidates for a given sim track
    std::vector<int> sim_TC_matched(n_accepted_simtrk);
    std::vector<int> sim_TC_matched_mask(n_accepted_simtrk);

    // Intermediate variables to keep track of matched sim tracks for a given track candidate
    std::vector<std::vector<int>> tc_matched_simIdx;

    // ============ Track candidates =============
    SDL::trackCandidates& trackCandidatesInGPU = (*event->getTrackCandidates());
    unsigned int nTrackCandidates = *trackCandidatesInGPU.nTrackCandidates;
    for (unsigned int idx = 0; idx < nTrackCandidates; idx++)
    {
        // Compute reco quantities of track candidate based on final object
        int type, isFake;
        float pt, eta, phi;
        std::vector<int> simidx;
        std::tie(type, pt, eta, phi, isFake, simidx) = parseTrackCandidate(event, idx);
        ana.tx->pushbackToBranch<float>("tc_pt", pt);
        ana.tx->pushbackToBranch<float>("tc_eta", eta);
        ana.tx->pushbackToBranch<float>("tc_phi", phi);
        ana.tx->pushbackToBranch<int>("tc_type", type);
        ana.tx->pushbackToBranch<int>("tc_isFake", isFake);
        tc_matched_simIdx.push_back(simidx);

        // Loop over matched sim idx and increase counter of TC_matched
        for (auto& idx : simidx)
        {
            // NOTE Important to note that the idx of the std::vector<> is same
            // as the tracking-ntuple's sim track idx ONLY because event==0 and bunchCrossing==0 condition is applied!!
            // Also do not try to access beyond the event and bunchCrossing
            if (idx < n_accepted_simtrk)
            {
                sim_TC_matched.at(idx) += 1;
                sim_TC_matched_mask.at(idx) |= (1 << type);
            }
        }
    }

    // Using the intermedaite variables to compute whether a given track candidate is a duplicate
    vector<int> tc_isDuplicate(tc_matched_simIdx.size());
    // Loop over the track candidates
    for (unsigned int i = 0; i < tc_matched_simIdx.size(); ++i)
    {
        bool isDuplicate = false;
        // Loop over the sim idx matched to this track candidate
        for (unsigned int isim = 0; isim < tc_matched_simIdx[i].size(); ++isim)
        {
            // Using the sim_TC_matched to see whether this track candidate is matched to a sim track that is matched to more than one
            int simidx = tc_matched_simIdx[i][isim];
            if (simidx < n_accepted_simtrk)
            {
                if (sim_TC_matched[simidx] > 1)
                {
                    isDuplicate = true;
                }
            }
        }
        tc_isDuplicate[i] = isDuplicate;
    }

    // Now set the last remaining branches
    ana.tx->setBranch<vector<int>>("sim_TC_matched", sim_TC_matched);
    ana.tx->setBranch<vector<int>>("sim_TC_matched_mask", sim_TC_matched_mask);
    ana.tx->setBranch<vector<vector<int>>>("tc_matched_simIdx", tc_matched_simIdx);
    ana.tx->setBranch<vector<int>>("tc_isDuplicate", tc_isDuplicate);
}

//________________________________________________________________________________________________________________________________
void setOptionalOutputBranches(SDL::Event* event)
{
#ifdef CUT_VALUE_DEBUG

    setPixelQuintupletOutputBranches(event);
    setQuintupletOutputBranches(event);
    setPixelTripletOutputBranches(event);

#endif
}

//________________________________________________________________________________________________________________________________
void setPixelQuintupletOutputBranches(SDL::Event* event)
{
    // ============ pT5 =============
    SDL::pixelQuintuplets& pixelQuintupletsInGPU = (*event->getPixelQuintuplets());
    SDL::quintuplets& quintupletsInGPU = (*event->getQuintuplets());
    SDL::segments& segmentsInGPU = (*event->getSegments());
    SDL::modules& modulesInGPU = (*event->getModules());
    int n_accepted_simtrk = ana.tx->getBranch<vector<int>>("sim_TC_matched").size();

    const float kRinv1GeVf = (2.99792458e-3 * 3.8);

    unsigned int nPixelQuintuplets = *pixelQuintupletsInGPU.nPixelQuintuplets; // size of this nPixelTriplets array is 1 (NOTE: parallelism lost here.)
    std::vector<int> sim_pT5_matched(n_accepted_simtrk);
    std::vector<std::vector<int>> pT5_matched_simIdx;

    for (unsigned int pT5 = 0; pT5 < nPixelQuintuplets; pT5++)
    {
        unsigned int T5Index = getT5FrompT5(event, pT5);
        unsigned int pLSIndex = getPixelLSFrompT5(event, pT5);
        float pt = (__H2F(quintupletsInGPU.innerRadius[T5Index]) * kRinv1GeVf + segmentsInGPU.ptIn[pLSIndex]) / 2;
        float eta = segmentsInGPU.eta[pLSIndex];
        float phi = segmentsInGPU.phi[pLSIndex];

        std::vector<unsigned int> hit_idx = getHitIdxsFrompT5(event, pT5);
        std::vector<unsigned int> module_idx = getModuleIdxsFrompT5(event, pT5);
        std::vector<unsigned int> hit_type = getHitTypesFrompT5(event, pT5);

        int layer_binary = 1;
        int moduleType_binary = 0;
        for (size_t i = 0; i < module_idx.size(); i += 2)
        {
            layer_binary |= (1 << (modulesInGPU.layers[module_idx[i]] + 6 * (modulesInGPU.subdets[module_idx[i]] == 4)));
            moduleType_binary |=  (modulesInGPU.moduleType[module_idx[i]] << i);  
        }
        std::vector<int> simidx = matchedSimTrkIdxs(hit_idx, hit_type);
        ana.tx->pushbackToBranch<int>("pT5_isFake", static_cast<int>(simidx.size() == 0)); 
        ana.tx->pushbackToBranch<float>("pT5_pt", pt);
        ana.tx->pushbackToBranch<float>("pT5_eta", eta);
        ana.tx->pushbackToBranch<float>("pT5_phi", phi);
        ana.tx->pushbackToBranch<int>("pT5_layer_binary", layer_binary);
        ana.tx->pushbackToBranch<int>("pT5_moduleType_binary", moduleType_binary);

        pT5_matched_simIdx.push_back(simidx);

        // Loop over matched sim idx and increase counter of pT5_matched
        for (auto& idx : simidx)
        {
            // NOTE Important to note that the idx of the std::vector<> is same
            // as the tracking-ntuple's sim track idx ONLY because event==0 and bunchCrossing==0 condition is applied!!
            // Also do not try to access beyond the event and bunchCrossing
            if (idx < n_accepted_simtrk)
            {
                sim_pT5_matched.at(idx) += 1;
            }
        }
    }

    // Using the intermedaite variables to compute whether a given track candidate is a duplicate
    vector<int> pT5_isDuplicate(pT5_matched_simIdx.size());
    // Loop over the track candidates
    for (unsigned int i = 0; i < pT5_matched_simIdx.size(); ++i)
    {
        bool isDuplicate = false;
        // Loop over the sim idx matched to this track candidate
        for (unsigned int isim = 0; isim < pT5_matched_simIdx[i].size(); ++isim)
        {
            // Using the sim_pT5_matched to see whether this track candidate is matched to a sim track that is matched to more than one
            int simidx = pT5_matched_simIdx[i][isim];
            if (simidx < n_accepted_simtrk)
            {
                if (sim_pT5_matched[simidx] > 1)
                {
                    isDuplicate = true;
                }
            }
        }
        pT5_isDuplicate[i] = isDuplicate;
    }

    // Now set the last remaining branches
    ana.tx->setBranch<vector<int>>("sim_pT5_matched", sim_pT5_matched);
    ana.tx->setBranch<vector<vector<int>>>("pT5_matched_simIdx", pT5_matched_simIdx);
    ana.tx->setBranch<vector<int>>("pT5_isDuplicate", pT5_isDuplicate);

}

//________________________________________________________________________________________________________________________________
void setQuintupletOutputBranches(SDL::Event* event)
{
    SDL::quintuplets& quintupletsInGPU = (*event->getQuintuplets());
    SDL::objectRanges& rangesInGPU = (*event->getRanges());
    SDL::modules& modulesInGPU = (*event->getModules());
    const float kRinv1GeVf = (2.99792458e-3 * 3.8);
    int n_accepted_simtrk = ana.tx->getBranch<vector<int>>("sim_TC_matched").size();

    std::vector<int> sim_t5_matched(n_accepted_simtrk);
    std::vector<std::vector<int>> t5_matched_simIdx;

    for (unsigned int lowerModuleIdx = 0; lowerModuleIdx < *(modulesInGPU.nLowerModules); ++lowerModuleIdx)
    {
        unsigned int nQuintuplets = quintupletsInGPU.nQuintuplets[lowerModuleIdx];
        for (unsigned int idx = 0; idx < nQuintuplets; idx++)
        {
            unsigned int quintupletIndex = rangesInGPU.quintupletModuleIndices[lowerModuleIdx] + idx;
            float pt = quintupletsInGPU.innerRadius[quintupletIndex] * kRinv1GeVf;
            float eta = __H2F(quintupletsInGPU.eta[quintupletIndex]);
            float phi = __H2F(quintupletsInGPU.phi[quintupletIndex]);
            
            std::vector<unsigned int> hit_idx = getHitIdxsFromT5(event, quintupletIndex);
            std::vector<unsigned int> hit_type = getHitTypesFromT5(event, quintupletIndex);
            std::vector<unsigned int> module_idx = getModuleIdxsFromT5(event, quintupletIndex);

            int layer_binary = 0;
            int moduleType_binary = 0;
            for (size_t i = 0; i < module_idx.size(); i += 2)
            {
                layer_binary |= (1 << (modulesInGPU.layers[module_idx[i]] + 6 * (modulesInGPU.subdets[module_idx[i]] == 4)));
                moduleType_binary |=  (modulesInGPU.moduleType[module_idx[i]] << i);  
            }

            std::vector<int> simidx = matchedSimTrkIdxs(hit_idx, hit_type);

            ana.tx->pushbackToBranch<int>("t5_isFake", static_cast<int>(simidx.size() == 0));
            ana.tx->pushbackToBranch<float>("t5_pt", pt);
            ana.tx->pushbackToBranch<float>("t5_eta", eta);
            ana.tx->pushbackToBranch<float>("t5_phi", phi);
            ana.tx->pushbackToBranch<float>("t5_innerRadius", __H2F(quintupletsInGPU.innerRadius[quintupletIndex]));
            ana.tx->pushbackToBranch<float>("t5_bridgeRadius", __H2F(quintupletsInGPU.bridgeRadius[quintupletIndex]));
            ana.tx->pushbackToBranch<float>("t5_outerRadius", __H2F(quintupletsInGPU.outerRadius[quintupletIndex]));
            ana.tx->pushbackToBranch<float>("t5_chiSquared", quintupletsInGPU.chiSquared[quintupletIndex]);
            ana.tx->pushbackToBranch<float>("t5_rzChiSquared", quintupletsInGPU.rzChiSquared[quintupletIndex]);
            ana.tx->pushbackToBranch<int>("t5_layer_binary", layer_binary);
            ana.tx->pushbackToBranch<int>("t5_moduleType_binary", moduleType_binary);

            t5_matched_simIdx.push_back(simidx);

            for (auto &simtrk : simidx)
            {
               if(simtrk < n_accepted_simtrk)
               {
                    sim_t5_matched.at(simtrk) += 1;
               }
            }
        }
    }

    vector<int> t5_isDuplicate(t5_matched_simIdx.size());
    for (unsigned int i = 0; i < t5_matched_simIdx.size(); i++)
    {
        bool isDuplicate = false;
        for (unsigned int isim = 0; isim < t5_matched_simIdx[i].size(); isim++)
        {
            int simidx = t5_matched_simIdx[i][isim];
            if(simidx < n_accepted_simtrk)
            {
                if(sim_t5_matched[simidx] > 1)
                {
                    isDuplicate = true;
                }
            }
        }
        t5_isDuplicate[i] = isDuplicate;
    }
    ana.tx->setBranch<vector<int>>("sim_T5_matched", sim_t5_matched);
    ana.tx->setBranch<vector<vector<int>>>("t5_matched_simIdx", t5_matched_simIdx);
    ana.tx->setBranch<vector<int>>("t5_isDuplicate", t5_isDuplicate);
}

//________________________________________________________________________________________________________________________________
void setPixelTripletOutputBranches(SDL::Event* event)
{
    SDL::pixelTriplets& pixelTripletsInGPU = (*event->getPixelTriplets());
    SDL::triplets& tripletsInGPU = *(event->getTriplets());
    SDL::modules& modulesInGPU = *(event->getModules());
    SDL::segments& segmentsInGPU = *(event->getSegments());
    SDL::hits& hitsInGPU = *(event->getHits());
    int n_accepted_simtrk = ana.tx->getBranch<vector<int>>("sim_TC_matched").size();

    unsigned int nPixelTriplets = *pixelTripletsInGPU.nPixelTriplets;
    std::vector<int> sim_pT3_matched(n_accepted_simtrk);
    std::vector<std::vector<int>> pT3_matched_simIdx;
    const float kRinv1GeVf = (2.99792458e-3 * 3.8);
    const float k2Rinv1GeVf = kRinv1GeVf / 2.;

    for (unsigned int pT3 = 0; pT3 < nPixelTriplets; pT3++)
    {
        unsigned int T3Index = getT3FrompT3(event, pT3);
        unsigned int pLSIndex = getPixelLSFrompT3(event, pT3);
        std::vector<unsigned int> Hits = getOuterTrackerHitsFrompT3(event, pT3);
        unsigned int Hit_0 = Hits[0];
        unsigned int Hit_4 = Hits[4];
        const float dr = sqrt(pow(hitsInGPU.xs[Hit_4] - hitsInGPU.xs[Hit_0], 2) + pow(hitsInGPU.ys[Hit_4] - hitsInGPU.ys[Hit_0], 2));
        float betaIn   = __H2F(tripletsInGPU.betaIn[T3Index]);
        float betaOut  = __H2F(tripletsInGPU.betaOut[T3Index]);
        const float pt_T3 = abs(dr * k2Rinv1GeVf / sin((betaIn + betaOut) / 2.));

        const float pt_pLS = segmentsInGPU.ptIn[pLSIndex];
        const float pt = (pt_pLS + pt_T3) / 2.;

        float eta = segmentsInGPU.eta[pLSIndex];
        float phi = segmentsInGPU.phi[pLSIndex];
        std::vector<unsigned int> hit_idx = getHitIdxsFrompT3(event, pT3);
        std::vector<unsigned int> hit_type = getHitTypesFrompT3(event, pT3);

        std::vector<int> simidx = matchedSimTrkIdxs(hit_idx, hit_type);
        std::vector<unsigned int> module_idx = getModuleIdxsFrompT3(event, pT3);
        int layer_binary = 1;
        int moduleType_binary = 0;
        for (size_t i = 0; i < module_idx.size(); i += 2)
        {
            layer_binary |= (1 << (modulesInGPU.layers[module_idx[i]] + 6 * (modulesInGPU.subdets[module_idx[i]] == 4)));
            moduleType_binary |=  (modulesInGPU.moduleType[module_idx[i]] << i);  
        }
        ana.tx->pushbackToBranch<int>("pT3_isFake", static_cast<int>(simidx.size() == 0));
        ana.tx->pushbackToBranch<float>("pT3_pt", pt);
        ana.tx->pushbackToBranch<float>("pT3_eta", eta);
        ana.tx->pushbackToBranch<float>("pT3_phi", phi);
        ana.tx->pushbackToBranch<int>("pT3_layer_binary", layer_binary);
        ana.tx->pushbackToBranch<int>("pT3_moduleType_binary", moduleType_binary);

        pT3_matched_simIdx.push_back(simidx);

        for (auto &idx : simidx)
        {
            if (idx < n_accepted_simtrk)
            {
                sim_pT3_matched.at(idx) += 1;
            }
        }
    }

    vector<int> pT3_isDuplicate(pT3_matched_simIdx.size());
    for (unsigned int i = 0; i < pT3_matched_simIdx.size(); i++)
    {
        bool isDuplicate = true;
        for (unsigned int isim = 0; isim < pT3_matched_simIdx[i].size(); isim++)
        {
            int simidx = pT3_matched_simIdx[i][isim];
            if (simidx < n_accepted_simtrk)
            {
                if (sim_pT3_matched[simidx] > 1)
                {
                    isDuplicate = true;
                }
            }
        }
        pT3_isDuplicate[i] = isDuplicate;
    }
    ana.tx->setBranch<vector<int>>("sim_pT3_matched", sim_pT3_matched);
    ana.tx->setBranch<vector<vector<int>>>("pT3_matched_simIdx", pT3_matched_simIdx);
    ana.tx->setBranch<vector<int>>("pT3_isDuplicate", pT3_isDuplicate);
}

//________________________________________________________________________________________________________________________________
void setGnnNtupleBranches(SDL::Event* event)
{
    // Get relevant information
    SDL::segments& segmentsInGPU = (*event->getSegments());
    SDL::miniDoublets& miniDoubletsInGPU = (*event->getMiniDoublets());
    SDL::quintuplets& quintupletsInGPU = (*event->getQuintuplets());
    SDL::hits& hitsInGPU = (*event->getHits());
    SDL::modules& modulesInGPU = (*event->getModules());
    SDL::objectRanges& rangesInGPU = (*event->getRanges());
    SDL::trackCandidates& trackCandidatesInGPU = (*event->getTrackCandidates());

    std::set<unsigned int> mds_used_in_sg;
    std::map<unsigned int, unsigned int> md_index_map;
    std::map<unsigned int, unsigned int> sg_index_map;

    std::set<unsigned int> T3s_used_in_T5;
    std::map<unsigned int, unsigned int> T3_index_map;
    // std::map<unsigned int, unsigned int> T5_index_map; // not used

    std::set<unsigned int> lss_used_in_true_tc;
    unsigned int nTrackCandidates = *trackCandidatesInGPU.nTrackCandidates;
    for (unsigned int idx = 0; idx < nTrackCandidates; idx++)
    {

        // Only consider true track candidates
        std::vector<unsigned int> hitidxs;
        std::vector<unsigned int> hittypes;
        std::tie(hitidxs, hittypes) = getHitIdxsAndHitTypesFromTC(event, idx);
        std::vector<int> simidxs = matchedSimTrkIdxs(hitidxs, hittypes);
        if (simidxs.size() == 0)
            continue;

        std::vector<unsigned int> LSs = getLSsFromTC(event, idx);
        for (auto& LS: LSs)
        {
            if (lss_used_in_true_tc.find(LS) == lss_used_in_true_tc.end())
            {
                lss_used_in_true_tc.insert(LS);
            }
        }
    }

    std::cout <<  " lss_used_in_true_tc.size(): " << lss_used_in_true_tc.size() <<  std::endl;

    // std::cout <<  " nTotalMD: " << nTotalMD <<  std::endl;
    // std::cout <<  " nTotalLS: " << nTotalLS <<  std::endl;

    // Loop over modules (lower ones where the MDs are saved)
    unsigned int nTotalMD = 0;
    unsigned int nTotalLS = 0;
    for (unsigned int idx = 0; idx < *(modulesInGPU.nLowerModules); ++idx)
    {
        /* We use getMDsFromLS instead, as we only want MDs associated w/ LS candidates
        // Loop over minidoublets
        nTotalMD += miniDoubletsInGPU.nMDs[idx];
        for (unsigned int jdx = 0; jdx < miniDoubletsInGPU.nMDs[idx]; jdx++)
        {
            // Get the actual index to the mini-doublet using rangesInGPU
            unsigned int mdIdx = rangesInGPU.miniDoubletModuleIndices[idx] + jdx;
            setGnnNtupleMiniDoublet(event, mdIdx);
        }
        */

        // Loop over segments
        nTotalLS += segmentsInGPU.nSegments[idx];
        for (unsigned int jdx = 0; jdx < segmentsInGPU.nSegments[idx]; jdx++)
        {
            // Get the actual index to the segments using rangesInGPU
            unsigned int sgIdx = rangesInGPU.segmentModuleIndices[idx] + jdx;

            // Get the hit indices
            std::vector<unsigned int> MDs = getMDsFromLS(event, sgIdx);

            if (mds_used_in_sg.find(MDs[0]) == mds_used_in_sg.end())
            {
                mds_used_in_sg.insert(MDs[0]);
                md_index_map[MDs[0]] = mds_used_in_sg.size() - 1;
                setGnnNtupleMiniDoublet(event, MDs[0]);
            }

            if (mds_used_in_sg.find(MDs[1]) == mds_used_in_sg.end())
            {
                mds_used_in_sg.insert(MDs[1]);
                md_index_map[MDs[1]] = mds_used_in_sg.size() - 1;
                setGnnNtupleMiniDoublet(event, MDs[1]);
            }

            ana.tx->pushbackToBranch<int>("LS_MD_idx0", md_index_map[MDs[0]]);
            ana.tx->pushbackToBranch<int>("LS_MD_idx1", md_index_map[MDs[1]]);

            std::vector<unsigned int> hits = getHitsFromLS(event, sgIdx);

            // Computing line segment pt estimate (assuming beam spot is at zero)
            SDL::CPU::Hit hitA(0, 0, 0);
            SDL::CPU::Hit hitB(hitsInGPU.xs[hits[0]], hitsInGPU.ys[hits[0]], hitsInGPU.zs[hits[0]]);
            SDL::CPU::Hit hitC(hitsInGPU.xs[hits[2]], hitsInGPU.ys[hits[2]], hitsInGPU.zs[hits[2]]);
            SDL::CPU::Hit center = SDL::CPU::MathUtil::getCenterFromThreePoints(hitA, hitB, hitC);
            float pt = SDL::CPU::MathUtil::ptEstimateFromRadius(center.rt());
            float eta = hitC.eta();
            float phi = hitB.phi();

            ana.tx->pushbackToBranch<float>("LS_pt", pt);
            ana.tx->pushbackToBranch<float>("LS_eta", eta);
            ana.tx->pushbackToBranch<float>("LS_phi", phi);
            // ana.tx->pushbackToBranch<int>("LS_layer0", layer0);
            // ana.tx->pushbackToBranch<int>("LS_layer1", layer1);

            std::vector<unsigned int> hitidxs;
            std::vector<unsigned int> hittypes;
            std::tie(hitidxs, hittypes) = getHitIdxsAndHitTypesFromLS(event, sgIdx);
            std::vector<int> simidxs = matchedSimTrkIdxs(hitidxs, hittypes);

            ana.tx->pushbackToBranch<int>  ("LS_isFake"     , simidxs.size() == 0);
            ana.tx->pushbackToBranch<float>("LS_sim_pt"     , simidxs.size() > 0 ? trk.sim_pt           ()[simidxs[0]] : -999);
            ana.tx->pushbackToBranch<float>("LS_sim_eta"    , simidxs.size() > 0 ? trk.sim_eta          ()[simidxs[0]] : -999);
            ana.tx->pushbackToBranch<float>("LS_sim_phi"    , simidxs.size() > 0 ? trk.sim_phi          ()[simidxs[0]] : -999);
            ana.tx->pushbackToBranch<float>("LS_sim_pca_dxy", simidxs.size() > 0 ? trk.sim_pca_dxy      ()[simidxs[0]] : -999);
            ana.tx->pushbackToBranch<float>("LS_sim_pca_dz" , simidxs.size() > 0 ? trk.sim_pca_dz       ()[simidxs[0]] : -999);
            ana.tx->pushbackToBranch<int>  ("LS_sim_q"      , simidxs.size() > 0 ? trk.sim_q            ()[simidxs[0]] : -999);
            ana.tx->pushbackToBranch<int>  ("LS_sim_event"  , simidxs.size() > 0 ? trk.sim_event        ()[simidxs[0]] : -999);
            ana.tx->pushbackToBranch<int>  ("LS_sim_bx"     , simidxs.size() > 0 ? trk.sim_bunchCrossing()[simidxs[0]] : -999);
            ana.tx->pushbackToBranch<int>  ("LS_sim_pdgId"  , simidxs.size() > 0 ? trk.sim_pdgId        ()[simidxs[0]] : -999);
            ana.tx->pushbackToBranch<float>("LS_sim_vx"     , simidxs.size() > 0 ? trk.simvtx_x         ()[trk.sim_parentVtxIdx()[simidxs[0]]] : -999);
            ana.tx->pushbackToBranch<float>("LS_sim_vy"     , simidxs.size() > 0 ? trk.simvtx_y         ()[trk.sim_parentVtxIdx()[simidxs[0]]] : -999);
            ana.tx->pushbackToBranch<float>("LS_sim_vz"     , simidxs.size() > 0 ? trk.simvtx_z         ()[trk.sim_parentVtxIdx()[simidxs[0]]] : -999);
            ana.tx->pushbackToBranch<int>  ("LS_isInTrueTC" , lss_used_in_true_tc.find(sgIdx) != lss_used_in_true_tc.end());

            sg_index_map[sgIdx] = ana.tx->getBranch<vector<int>>("LS_isFake").size() - 1;

            // // T5 eta and phi are computed using outer and innermost hits
            // SDL::CPU::Hit hitA(trk.ph2_x()[anchitidx], trk.ph2_y()[anchitidx], trk.ph2_z()[anchitidx]);
            // const float phi = hitA.phi();
            // const float eta = hitA.eta();
        }

        // Loop over quintuplets
        for (unsigned int jdx = 0; jdx < quintupletsInGPU.nQuintuplets[idx]; jdx++)
        {
            /*
                Pictorial representation of a T5
               
                inner tracker        outer tracker
                -------------  --------------------------
                               01    23    45    67    89   (anchor hit of a minidoublet is always the first of the pair)
                 (none)        oo -- oo -- oo -- oo -- oo   T5
                               |____________|
                                   T3s[0]  |____________|
                                  inner T3     T3s[1]
                                              outer T3
            */
            unsigned int T5_idx = rangesInGPU.quintupletModuleIndices[idx] + jdx;

            std::vector<unsigned int> T3s = getT3sFromT5(event, T5_idx);
            // Inner T3
            if (T3s_used_in_T5.find(T3s[0]) == T3s_used_in_T5.end())
            {
                T3s_used_in_T5.insert(T3s[0]);
                T3_index_map[T3s[0]] = T3s_used_in_T5.size() - 1;
                setGnnNtupleTriplet(event, T3s[0]);
            }
            // Outer T3
            if (T3s_used_in_T5.find(T3s[1]) == T3s_used_in_T5.end())
            {
                T3s_used_in_T5.insert(T3s[1]);
                T3_index_map[T3s[1]] = T3s_used_in_T5.size() - 1;
                setGnnNtupleTriplet(event, T3s[1]);
            }

            ana.tx->pushbackToBranch<int>("t5_t3_idx0", T3_index_map[T3s[0]]);
            ana.tx->pushbackToBranch<int>("t5_t3_idx1", T3_index_map[T3s[1]]);
        }
    }

    for (unsigned int idx = 0; idx < nTrackCandidates; idx++)
    {
        std::vector<unsigned int> LSs = getLSsFromTC(event, idx);
        std::vector<int> lsIdx;
        for (auto& LS : LSs)
        {
            lsIdx.push_back(sg_index_map[LS]);
        }
        ana.tx->pushbackToBranch<vector<int>>("tc_lsIdx", lsIdx);
    }

    std::cout <<  " ana.tx->getBranch<vector<vector<int>>>('tc_lsIdx').size(): " << ana.tx->getBranch<vector<vector<int>>>("tc_lsIdx").size() <<  std::endl;


    std::cout <<  " mds_used_in_sg.size(): " << mds_used_in_sg.size() <<  std::endl;

    // std::cout <<  " ana.tx->getBranchLazy<vector<float>>('MD_pt').size(): " << ana.tx->getBranchLazy<vector<float>>("MD_pt").size() <<  std::endl;
    // std::cout <<  " mds_used_in_sg.size(): " << mds_used_in_sg.size() <<  std::endl;
}

//________________________________________________________________________________________________________________________________
void setGnnNtupleMiniDoublet(SDL::Event* event, unsigned int MD)
{
    // Get relevant information
    SDL::miniDoublets& miniDoubletsInGPU = (*event->getMiniDoublets());
    SDL::hits& hitsInGPU = (*event->getHits());

    // Get the hit indices
    unsigned int hit0 = miniDoubletsInGPU.anchorHitIndices[MD];
    unsigned int hit1 = miniDoubletsInGPU.outerHitIndices[MD];

    // Get the hit infos
    const float hit0_x = hitsInGPU.xs[hit0];
    const float hit0_y = hitsInGPU.ys[hit0];
    const float hit0_z = hitsInGPU.zs[hit0];
    const float hit0_r = sqrt(hit0_x * hit0_x + hit0_y * hit0_y);
    const float hit1_x = hitsInGPU.xs[hit1];
    const float hit1_y = hitsInGPU.ys[hit1];
    const float hit1_z = hitsInGPU.zs[hit1];
    const float hit1_r = sqrt(hit1_x * hit1_x + hit1_y * hit1_y);

    // Do sim matching
    std::vector<unsigned int> hit_idx = {hitsInGPU.idxs[hit0], hitsInGPU.idxs[hit1]};
    std::vector<unsigned int> hit_type = {4, 4};
    std::vector<int> simidxs = matchedSimTrkIdxs(hit_idx, hit_type);

    bool isFake = simidxs.size() == 0;
    int tp_type = getDenomSimTrkType(simidxs);

    // Obtain where the actual hit is located in terms of their layer, module, rod, and ring number
    unsigned int anchitidx = hitsInGPU.idxs[hit0];
    int subdet = trk.ph2_subdet()[hitsInGPU.idxs[anchitidx]];
    int is_endcap = subdet == 4;
    int layer = trk.ph2_layer()[anchitidx] + 6 * (is_endcap); // this accounting makes it so that you have layer 1 2 3 4 5 6 in the barrel, and 7 8 9 10 11 in the endcap. (becuase endcap is ph2_subdet == 4)
    int detId = trk.ph2_detId()[anchitidx];

    // Obtaining dPhiChange
    float dphichange = miniDoubletsInGPU.dphichanges[MD];

    // Computing pt
    const float kRinv1GeVf = (2.99792458e-3 * 3.8);
    const float k2Rinv1GeVf = kRinv1GeVf / 2.;
    float pt = hit0_r * k2Rinv1GeVf / sin(dphichange);

    // T5 eta and phi are computed using outer and innermost hits
    SDL::CPU::Hit hitA(trk.ph2_x()[anchitidx], trk.ph2_y()[anchitidx], trk.ph2_z()[anchitidx]);
    const float phi = hitA.phi();
    const float eta = hitA.eta();

    // Mini Doublets
    ana.tx->pushbackToBranch<float>("MD_pt", pt);
    ana.tx->pushbackToBranch<float>("MD_eta", eta);
    ana.tx->pushbackToBranch<float>("MD_phi", phi);
    ana.tx->pushbackToBranch<float>("MD_dphichange", dphichange);
    ana.tx->pushbackToBranch<int>("MD_isFake", isFake);
    ana.tx->pushbackToBranch<int>("MD_tpType", tp_type);
    ana.tx->pushbackToBranch<int>("MD_detId", detId);
    ana.tx->pushbackToBranch<int>("MD_layer", layer);
    ana.tx->pushbackToBranch<float>("MD_0_r", hit0_r);
    ana.tx->pushbackToBranch<float>("MD_0_x", hit0_x);
    ana.tx->pushbackToBranch<float>("MD_0_y", hit0_y);
    ana.tx->pushbackToBranch<float>("MD_0_z", hit0_z);
    ana.tx->pushbackToBranch<float>("MD_1_r", hit1_r);
    ana.tx->pushbackToBranch<float>("MD_1_x", hit1_x);
    ana.tx->pushbackToBranch<float>("MD_1_y", hit1_y);
    ana.tx->pushbackToBranch<float>("MD_1_z", hit1_z);
    // ana.tx->pushbackToBranch<int>("MD_sim_idx", simidxs.size() > 0 ? simidxs[0] : -999);
}

//________________________________________________________________________________________________________________________________
void setGnnNtupleTriplet(SDL::Event* event, unsigned int T3)
{
    /*
        Pictorial representation of a T3
       
        inner tracker  outer tracker
        -------------  --------------
                       01    23    45   (anchor hit of a minidoublet is always the first of the pair)
         (none)        oo -- oo -- oo   T3
    */

    // Get relevant information
    SDL::hits& hitsInGPU = (*event->getHits());
    SDL::modules& modulesInGPU = (*event->getFullModules());
    SDL::triplets& tripletsInGPU = (*event->getTriplets());

    // Hits
    std::vector<unsigned int> hits = getHitsFromT3(event, T3);
    unsigned int hit0 = hits[0];
    unsigned int hit2 = hits[2];
    unsigned int hit4 = hits[4];

    // Hit locations
    const float hit0_x = hitsInGPU.xs[hit0];
    const float hit0_y = hitsInGPU.ys[hit0];
    const float hit0_z = hitsInGPU.zs[hit0];
    const float hit0_r = sqrt(hit0_x * hit0_x + hit0_y * hit0_y);
    const float hit2_x = hitsInGPU.xs[hit2];
    const float hit2_y = hitsInGPU.ys[hit2];
    const float hit2_z = hitsInGPU.zs[hit2];
    const float hit2_r = sqrt(hit2_x * hit2_x + hit2_y * hit2_y);
    const float hit4_x = hitsInGPU.xs[hit4];
    const float hit4_y = hitsInGPU.ys[hit4];
    const float hit4_z = hitsInGPU.zs[hit4];
    const float hit4_r = sqrt(hit4_x * hit4_x + hit4_y * hit4_y);
    ana.tx->pushbackToBranch<float>("t3_0_r", hit0_r);
    ana.tx->pushbackToBranch<float>("t3_0_x", hit0_x);
    ana.tx->pushbackToBranch<float>("t3_0_y", hit0_y);
    ana.tx->pushbackToBranch<float>("t3_0_z", hit0_z);
    ana.tx->pushbackToBranch<float>("t3_2_r", hit2_r);
    ana.tx->pushbackToBranch<float>("t3_2_x", hit2_x);
    ana.tx->pushbackToBranch<float>("t3_2_y", hit2_y);
    ana.tx->pushbackToBranch<float>("t3_2_z", hit2_z);
    ana.tx->pushbackToBranch<float>("t3_4_r", hit4_r);
    ana.tx->pushbackToBranch<float>("t3_4_x", hit4_x);
    ana.tx->pushbackToBranch<float>("t3_4_y", hit4_y);
    ana.tx->pushbackToBranch<float>("t3_4_z", hit4_z);

    /* Sigmas for chi2 calculation 
     * (stolen from SDL::computeSigmasForRegression and SDL::computeRadiusUsingRegressionk) */
    std::vector<float> sigmas;
    float inv1 = 0.01f/0.009f;
    float inv2 = 0.15f/0.009f;
    // float inv3 = 2.4f/0.009f; // not used
    for (auto hit : {hit0, hit2, hit4})
    {
        // Get module info
        unsigned int module = hitsInGPU.moduleIndices[hit];
        SDL::ModuleType module_type = modulesInGPU.moduleType[module];
        short module_subdet = modulesInGPU.subdets[module];
        short module_side = modulesInGPU.sides[module];
        float module_drdz = modulesInGPU.drdzs[module];
        float module_slope = modulesInGPU.slopes[module];
        // Get deltas for sigma calculation
        float delta1, delta2;
        bool is_flat;
        // Category 1: barrel PS flat
        if (module_subdet == SDL::Barrel and module_type == SDL::PS and module_side == SDL::Center)
        {
            delta1 = inv1;//1.1111f;//0.01;
            delta2 = inv1;//1.1111f;//0.01;
            module_slope = -999.f;
            is_flat = true;
        }
        // Category 2: barrel 2S
        else if (module_subdet == SDL::Barrel and module_type == SDL::TwoS)
        {
            delta1 = 1.f;//0.009;
            delta2 = 1.f;//0.009;
            module_slope = -999.f;
            is_flat = true;
        }
        // Category 3: barrel PS tilted
        else if (module_subdet == SDL::Barrel and module_type == SDL::PS and module_side != SDL::Center)
        {

            delta1 = inv1;//1.1111f;//0.01;
            is_flat = false;

            delta2 = (inv2 * module_drdz/sqrtf(1 + module_drdz * module_drdz));
        }
        // Category 4: endcap PS
        else if (module_subdet == SDL::Endcap and module_type == SDL::PS)
        {
            delta1 = inv1;//1.1111f;//0.01;
            is_flat = false;

            /* Despite the type of the module layer of the lower module index,
             * all anchor hits are on the pixel side and all non-anchor hits are
             * on the strip side! */
            delta2 = inv2;//16.6666f;//0.15f;
        }
        // Category 5: endcap 2S
        else if (module_subdet == SDL::Endcap and module_type == SDL::TwoS)
        {
            delta1 = 1.f;//0.009;
            delta2 = 500.f*inv1;//555.5555f;//5.f;
            is_flat = false;
        }
        else
        {
            printf("ERROR!!!!! I SHOULDN'T BE HERE!!!! subdet = %d, type = %d, side = %d\n", module_subdet, module_type, module_side);
        }

        /* Since C++ can't represent infinity, SDL_INF = 123456789 was used 
         * to represent infinity in the data table */
        float absArctanSlope = ((module_slope != 123456789) ? fabs(atanf(module_slope)) : 0.5f*float(M_PI));

        float hit_x = hitsInGPU.xs[hit];
        float hit_y = hitsInGPU.ys[hit];
        float angleM;
        if (hit_x > 0 and hit_y > 0)
        {
            angleM = 0.5f*float(M_PI) - absArctanSlope;
        }
        else if (hit_x < 0 and hit_y > 0)
        {
            angleM = absArctanSlope + 0.5f*float(M_PI);
        }
        else if (hit_x < 0 and hit_y < 0)
        {
            angleM = -(absArctanSlope + 0.5f*float(M_PI));
        }
        else if (hit_x > 0 and hit_y < 0)
        {
            angleM = -(0.5f*float(M_PI) - absArctanSlope);
        }

        float xPrime, yPrime;
        if(not is_flat)
        {
            xPrime = hit_x * cosf(angleM) + hit_y * sinf(angleM);
            yPrime = hit_y * cosf(angleM) - hit_x * sinf(angleM);
        }
        else
        {
            xPrime = hit_x;
            yPrime = hit_y;
        }
        sigmas.push_back(2 * sqrtf((xPrime * delta1) * (xPrime * delta1) + (yPrime * delta2) * (yPrime * delta2)));
    }

    ana.tx->pushbackToBranch<float>("t3_0_dr", sigmas[0]);
    ana.tx->pushbackToBranch<float>("t3_2_dr", sigmas[1]);
    ana.tx->pushbackToBranch<float>("t3_4_dr", sigmas[2]);

    // Constants
    const float kRinv1GeVf = (2.99792458e-3 * 3.8);
    const float k2Rinv1GeVf = kRinv1GeVf / 2.;

    // Radial distance
    const float dr = sqrt(pow(hitsInGPU.xs[hit4] - hitsInGPU.xs[hit0], 2) + pow(hitsInGPU.ys[hit4] - hitsInGPU.ys[hit0], 2));

    // Beta angles
    float betaIn  = __H2F(tripletsInGPU.betaIn[T3]);
    float betaOut = __H2F(tripletsInGPU.betaOut[T3]);

    // Legacy T4 pt estimate
    const float ptAv = abs(dr * k2Rinv1GeVf / sin((betaIn + betaOut) / 2.));
    SDL::CPU::Hit hitA(trk.ph2_x()[hit0], trk.ph2_y()[hit0], trk.ph2_z()[hit0]);
    SDL::CPU::Hit hitB(trk.ph2_x()[hit2], trk.ph2_y()[hit2], trk.ph2_z()[hit2]);
    SDL::CPU::Hit hitC(trk.ph2_x()[hit4], trk.ph2_y()[hit4], trk.ph2_z()[hit4]);
    ana.tx->pushbackToBranch<float>("t3_ptAv", ptAv);
    // More accurate pt estimate
    SDL::CPU::Hit center = SDL::CPU::MathUtil::getCenterFromThreePoints(hitA, hitB, hitC);
    ana.tx->pushbackToBranch<float>("t3_pt", SDL::CPU::MathUtil::ptEstimateFromRadius(center.rt()));
    // Angles
    ana.tx->pushbackToBranch<float>("t3_eta", hitA.phi());
    ana.tx->pushbackToBranch<float>("t3_phi", hitC.eta());

    // Truth information
    std::vector<unsigned int> hitidxs;
    std::vector<unsigned int> hittypes;
    std::tie(hitidxs, hittypes) = getHitIdxsAndHitTypesFromT3(event, T3);
    std::vector<int> simidxs = matchedSimTrkIdxs(hitidxs, hittypes);
    ana.tx->pushbackToBranch<int>("t3_isFake", simidxs.size() == 0);
}

//________________________________________________________________________________________________________________________________
std::tuple<int, float, float, float, int, vector<int>> parseTrackCandidate(SDL::Event* event, unsigned int idx)
{
    // Get the type of the track candidate
    SDL::trackCandidates& trackCandidatesInGPU = (*event->getTrackCandidates());
    short type = trackCandidatesInGPU.trackCandidateType[idx];

    enum
    {
        pT5 = 7,
        pT3 = 5,
        T5 = 4,
        pLS = 8
    };

    // Compute pt eta phi and hit indices that will be used to figure out whether the TC matched
    float pt, eta, phi;
    std::vector<unsigned int> hit_idx, hit_type;
    switch (type)
    {
        case pT5: std::tie(pt, eta, phi, hit_idx, hit_type) = parsepT5(event, idx); break;
        case pT3: std::tie(pt, eta, phi, hit_idx, hit_type) = parsepT3(event, idx); break;
        case T5:  std::tie(pt, eta, phi, hit_idx, hit_type) = parseT5(event, idx); break;
        case pLS: std::tie(pt, eta, phi, hit_idx, hit_type) = parsepLS(event, idx); break;

    }

    // Perform matching
    std::vector<int> simidx = matchedSimTrkIdxs(hit_idx, hit_type);
    int isFake = simidx.size() == 0;

    return {type, pt, eta, phi, isFake, simidx};
}

//________________________________________________________________________________________________________________________________
std::tuple<float, float, float, vector<unsigned int>, vector<unsigned int>> parsepT5(SDL::Event* event, unsigned int idx)
{
    // Get relevant information
    SDL::trackCandidates& trackCandidatesInGPU = (*event->getTrackCandidates());
    SDL::triplets& tripletsInGPU = (*event->getTriplets());
    SDL::segments& segmentsInGPU = (*event->getSegments());
    SDL::hits& hitsInGPU = (*event->getHits());

    //
    // pictorial representation of a pT5
    //
    // inner tracker        outer tracker
    // -------------  --------------------------
    // pLS            01    23    45    67    89   (anchor hit of a minidoublet is always the first of the pair)
    // ****           oo -- oo -- oo -- oo -- oo   pT5
    //                oo -- oo -- oo               first T3 of the T5
    //                            oo -- oo -- oo   second T3 of the T5
    unsigned int pT5 = trackCandidatesInGPU.directObjectIndices[idx];
    std::vector<unsigned int> Hits = getOuterTrackerHitsFrompT5(event, pT5);
    unsigned int Hit_0 = Hits[0];
    unsigned int Hit_4 = Hits[4];
    unsigned int Hit_8 = Hits[8];

    std::vector<unsigned int> T3s = getT3sFrompT5(event, pT5);
    unsigned int T3_0 = T3s[0];
    unsigned int T3_1 = T3s[1];

    unsigned int pLS = getPixelLSFrompT5(event, pT5);

    const float kRinv1GeVf = (2.99792458e-3 * 3.8);
    const float k2Rinv1GeVf = kRinv1GeVf / 2.;

    //=================================================================================
    // Some history and geometry lesson...
    // For a given T3, we compute two angles. (NOTE: This is a bit weird!)
    // Historically, T3 were created out of T4, which we used to build a long time ago.
    // So for the sake of argument let's discuss T4 first.
    // For a T4, we have 4 mini-doublets.
    // Therefore we have 4 "anchor hits".
    // Therefore we have 4 xyz points.
    //
    //
    //       *
    //       |\
    //       | \
    //       |1 \
    //       |   \
    //       |  * \
    //       |
    //       |
    //       |
    //       |
    //       |
    //       |  * /
    //       |   /
    //       |2 /
    //       | /
    //       |/
    //       *
    //
    //
    // Then from these 4 points, one can approximate a some sort of "best" fitted circle trajectory,
    // and obtain "tangential" angles from 1st and 4th hits.
    // See the carton below.
    // The "*" are the 4 physical hit points
    // angle 1 and 2 are the "tangential" angle for a "circle" from 4 * points.
    // Please note, that a straight line from first two * and the latter two * are NOT the
    // angle 1 and angle 2. (they were called "beta" angles)
    // But rather, a slightly larger angle.
    // Because 4 * points would be on a circle, and a tangential line on the circles
    // would deviate from the points on circles.
    //
    // In the early days of LST, there was an iterative algorithm (devised by Slava) to
    // obtain the angle beta1 and 2 _without_ actually performing a 4 point circle fit.
    // Hence, the beta1 and beta2 were quickly estimated without too many math operations
    // and afterwards (beta1-beta2) was computed to obtain what we call a "delta-beta" values.
    //
    // For a real track, the deltabeta ~ 0, for fakes, it'd have a flat distribution.
    //
    // However, after some time we abandonded the T4s, and moved to T3s.
    // In T3, however, now we have the following cartoon:
    //
    //       *
    //       |\
    //       | \
    //       |1 \
    //       |   \
    //       |  * X   (* here are "two" MDs but really just one)
    //       |   /
    //       |2 /
    //       | /
    //       |/
    //       *
    //
    // With the "four" *'s (really just "three") you can still run the iterative beta calculation,
    // which is what we still currently do, we still get two beta1 and beta2
    // But! high school geometry tells us that 3 points = ONLY 1 possible CIRCLE!
    // There is really nothing to "fit" here.
    // YET we still compute these in T3, out of legacy method of how we used to treat T4s.
    //
    // Hence, in the below code, "betaIn_in" and "betaOut_in" if we performed
    // a circle fit they would come out by definition identical values.
    // But due to our approximate iterative beta calculation method, they come out different values.
    // So if we are "cutting on" abs(deltaBeta) = abs(betaIn_in - betaOut_in) < threshold,
    // what does that even mean?
    //
    // Anyhow, as of now, we compute 2 beta's for T3s, and T5 has two T3s.
    // And from there we estimate the pt's and we compute pt_T5.

    // Compute the radial distance between first mini-doublet to third minidoublet
    const float dr_in = sqrt(pow(hitsInGPU.xs[Hit_4] - hitsInGPU.xs[Hit_0], 2) + pow(hitsInGPU.ys[Hit_4] - hitsInGPU.ys[Hit_0], 2));
    // Compute the radial distance between third mini-doublet to fifth minidoublet
    const float dr_out = sqrt(pow(hitsInGPU.xs[Hit_8] - hitsInGPU.xs[Hit_4], 2) + pow(hitsInGPU.ys[Hit_8] - hitsInGPU.ys[Hit_4], 2));
    float betaIn_in   = __H2F(tripletsInGPU.betaIn[T3_0]);
    float betaOut_in  = __H2F(tripletsInGPU.betaOut[T3_0]);
    float betaIn_out  = __H2F(tripletsInGPU.betaIn[T3_1]);
    float betaOut_out = __H2F(tripletsInGPU.betaOut[T3_1]);
    const float ptAv_in = abs(dr_in * k2Rinv1GeVf / sin((betaIn_in + betaOut_in) / 2.));
    const float ptAv_out = abs(dr_out * k2Rinv1GeVf / sin((betaIn_out + betaOut_out) / 2.));
    const float pt_T5 = (ptAv_in + ptAv_out) / 2.;

    // pixel pt
    const float pt_pLS = segmentsInGPU.ptIn[pLS];
    const float eta_pLS = segmentsInGPU.eta[pLS];
    const float phi_pLS = segmentsInGPU.phi[pLS];

    // average pt
    const float pt = (pt_pLS + pt_T5) / 2.;

    // Form the hit idx/type vector
    std::vector<unsigned int> hit_idx = getHitIdxsFrompT5(event, pT5);
    std::vector<unsigned int> hit_type = getHitTypesFrompT5(event, pT5);

    return {pt, eta_pLS, phi_pLS, hit_idx, hit_type};

}

//________________________________________________________________________________________________________________________________
std::tuple<float, float, float, vector<unsigned int>, vector<unsigned int>> parsepT3(SDL::Event* event, unsigned int idx)
{
    // Get relevant information
    SDL::trackCandidates& trackCandidatesInGPU = (*event->getTrackCandidates());
    SDL::triplets& tripletsInGPU = (*event->getTriplets());
    SDL::segments& segmentsInGPU = (*event->getSegments());
    SDL::hits& hitsInGPU = (*event->getHits());

    //
    // pictorial representation of a pT3
    //
    // inner tracker        outer tracker
    // -------------  --------------------------
    // pLS            01    23    45               (anchor hit of a minidoublet is always the first of the pair)
    // ****           oo -- oo -- oo               pT3
    unsigned int pT3 = trackCandidatesInGPU.directObjectIndices[idx];
    std::vector<unsigned int> Hits = getOuterTrackerHitsFrompT3(event, pT3);
    unsigned int Hit_0 = Hits[0];
    unsigned int Hit_4 = Hits[4];

    unsigned int T3 = getT3FrompT3(event, pT3);

    unsigned int pLS = getPixelLSFrompT3(event, pT3);

    const float kRinv1GeVf = (2.99792458e-3 * 3.8);
    const float k2Rinv1GeVf = kRinv1GeVf / 2.;
    const float dr = sqrt(pow(hitsInGPU.xs[Hit_4] - hitsInGPU.xs[Hit_0], 2) + pow(hitsInGPU.ys[Hit_4] - hitsInGPU.ys[Hit_0], 2));
    float betaIn   = __H2F(tripletsInGPU.betaIn[T3]);
    float betaOut  = __H2F(tripletsInGPU.betaOut[T3]);
    const float pt_T3 = abs(dr * k2Rinv1GeVf / sin((betaIn + betaOut) / 2.));

    // pixel pt
    const float pt_pLS = segmentsInGPU.ptIn[pLS];
    const float eta_pLS = segmentsInGPU.eta[pLS];
    const float phi_pLS = segmentsInGPU.phi[pLS];

    // average pt
    const float pt = (pt_pLS + pt_T3) / 2.;

    // Form the hit idx/type vector
    std::vector<unsigned int> hit_idx = getHitIdxsFrompT3(event, pT3);
    std::vector<unsigned int> hit_type = getHitTypesFrompT3(event, pT3);

    return {pt, eta_pLS, phi_pLS, hit_idx, hit_type};

}

//________________________________________________________________________________________________________________________________
std::tuple<float, float, float, vector<unsigned int>, vector<unsigned int>> parseT5(SDL::Event* event, unsigned int idx)
{
    SDL::trackCandidates& trackCandidatesInGPU = (*event->getTrackCandidates());
    SDL::triplets& tripletsInGPU = (*event->getTriplets());
    SDL::hits& hitsInGPU = (*event->getHits());
    unsigned int T5 = trackCandidatesInGPU.directObjectIndices[idx];
    std::vector<unsigned int> T3s = getT3sFromT5(event, T5);
    std::vector<unsigned int> hits = getHitsFromT5(event, T5);

    //
    // pictorial representation of a T5
    //
    // inner tracker        outer tracker
    // -------------  --------------------------
    //                01    23    45    67    89   (anchor hit of a minidoublet is always the first of the pair)
    //  (none)        oo -- oo -- oo -- oo -- oo   T5
    unsigned int Hit_0 = hits[0];
    unsigned int Hit_4 = hits[4];
    unsigned int Hit_8 = hits[8];

    // radial distance
    const float dr_in = sqrt(pow(hitsInGPU.xs[Hit_4] - hitsInGPU.xs[Hit_0], 2) + pow(hitsInGPU.ys[Hit_4] - hitsInGPU.ys[Hit_0], 2));
    const float dr_out = sqrt(pow(hitsInGPU.xs[Hit_8] - hitsInGPU.xs[Hit_4], 2) + pow(hitsInGPU.ys[Hit_8] - hitsInGPU.ys[Hit_4], 2));

    // beta angles
    float betaIn_in   = __H2F(tripletsInGPU.betaIn [T3s[0]]);
    float betaOut_in  = __H2F(tripletsInGPU.betaOut[T3s[0]]);
    float betaIn_out  = __H2F(tripletsInGPU.betaIn [T3s[1]]);
    float betaOut_out = __H2F(tripletsInGPU.betaOut[T3s[1]]);

    // constants
    const float kRinv1GeVf = (2.99792458e-3 * 3.8);
    const float k2Rinv1GeVf = kRinv1GeVf / 2.;

    // Compute pt estimates from inner and outer triplets
    const float ptAv_in = abs(dr_in * k2Rinv1GeVf / sin((betaIn_in + betaOut_in) / 2.));
    const float ptAv_out = abs(dr_out * k2Rinv1GeVf / sin((betaIn_out + betaOut_out) / 2.));

    // T5 pt is average of the two pt estimates
    const float pt = (ptAv_in + ptAv_out) / 2.;     // this is deprecated, c.f. setQuintupletOutputBranches

    // T5 eta and phi are computed using outer and innermost hits
    SDL::CPU::Hit hitA(trk.ph2_x()[Hit_0], trk.ph2_y()[Hit_0], trk.ph2_z()[Hit_0]);
    SDL::CPU::Hit hitB(trk.ph2_x()[Hit_8], trk.ph2_y()[Hit_8], trk.ph2_z()[Hit_8]);
    const float phi = hitA.phi();
    const float eta = hitB.eta();

    std::vector<unsigned int> hit_idx = getHitIdxsFromT5(event, T5);
    std::vector<unsigned int> hit_type = getHitTypesFromT5(event, T5);

    return {pt, eta, phi, hit_idx, hit_type};
}

//________________________________________________________________________________________________________________________________
std::tuple<float, float, float, vector<unsigned int>, vector<unsigned int>> parsepLS(SDL::Event* event, unsigned int idx)
{
    SDL::trackCandidates& trackCandidatesInGPU = (*event->getTrackCandidates());
    SDL::segments& segmentsInGPU = (*event->getSegments());

    // Getting pLS index
    unsigned int pLS = trackCandidatesInGPU.directObjectIndices[idx];

    // Getting pt eta and phi
    float pt = segmentsInGPU.ptIn[pLS];
    float eta = segmentsInGPU.eta[pLS];
    float phi = segmentsInGPU.phi[pLS];

    // Getting hit indices and types
    std::vector<unsigned int> hit_idx = getPixelHitIdxsFrompLS(event, pLS);
    std::vector<unsigned int> hit_type = getPixelHitTypesFrompLS(event, pLS);

    return {pt, eta, phi, hit_idx, hit_type};
}

//________________________________________________________________________________________________________________________________
float computeRadiusFromThreeAnchorHits(float x1, float y1, float x2, float y2, float x3, float y3, float& g, float& f)
{
   float radius = 0;
   if ((y1 - y3) * (x2 - x3) - (x1 - x3) * (y2 - y3) == 0)
    {
        return -1; // WTF man three collinear points!
    }

    float denom = ((y1 - y3) * (x2 - x3) - (x1 - x3) * (y2 - y3));

    g = 0.5 * ((y3 - y2) * (x1 * x1 + y1 * y1) + (y1 - y3) * (x2 * x2 + y2 * y2) + (y2 - y1) * (x3 * x3 + y3 * y3)) / denom;

    f = 0.5 * ((x2 - x3) * (x1 * x1 + y1 * y1) + (x3 - x1) * (x2 * x2 + y2 * y2) + (x1 - x2) * (x3 * x3 + y3 * y3)) / denom;

    float c = ((x2 * y3 - x3 * y2) * (x1 * x1 + y1 * y1) + (x3 * y1 - x1 * y3) * (x2 * x2 + y2 * y2) + (x1 * y2 - x2 * y1) * (x3 * x3 + y3 * y3)) / denom;

    if (g * g + f * f - c < 0)
    {
        std::cout << "FATAL! r^2 < 0!" << std::endl;
        return -1;
    }

    radius = sqrtf(g * g + f * f - c);
    return radius;
}

//________________________________________________________________________________________________________________________________
void printHitMultiplicities(SDL::Event* event)
{
    //SDL::hits& hitsInGPU = (*event->getHits());
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
}

//________________________________________________________________________________________________________________________________
void printMDs(SDL::Event* event)
{
    SDL::miniDoublets& miniDoubletsInGPU = (*event->getMiniDoublets());
    SDL::hits& hitsInGPU = (*event->getHits());
    SDL::modules& modulesInGPU = (*event->getModules());
    SDL::objectRanges& rangesInGPU = (*event->getRanges());

    // Then obtain the lower module index
    for (unsigned int idx = 0; idx <= *(modulesInGPU.nLowerModules); ++idx)
    {
        for (unsigned int iMD = 0; iMD < miniDoubletsInGPU.nMDs[idx]; iMD++)
        {
            unsigned int mdIdx = rangesInGPU.miniDoubletModuleIndices[idx] + iMD;
            unsigned int LowerHitIndex = miniDoubletsInGPU.anchorHitIndices[mdIdx];
            unsigned int UpperHitIndex = miniDoubletsInGPU.outerHitIndices[mdIdx];
            unsigned int hit0 = hitsInGPU.idxs[LowerHitIndex];
            unsigned int hit1 = hitsInGPU.idxs[UpperHitIndex];
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
    SDL::objectRanges& rangesInGPU = (*event->getRanges());

    int nSegments = 0;
    for (unsigned int i = 0; i <  *(modulesInGPU.nLowerModules); ++i)
    {
        unsigned int idx = i;//modulesInGPU.lowerModuleIndices[i];
        nSegments += segmentsInGPU.nSegments[idx];
        for (unsigned int jdx = 0; jdx < segmentsInGPU.nSegments[idx]; jdx++)
        {
            unsigned int sgIdx = rangesInGPU.segmentModuleIndices[idx] + jdx;
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
void printpLSs(SDL::Event* event)
{
    SDL::segments& segmentsInGPU = (*event->getSegments());
    SDL::miniDoublets& miniDoubletsInGPU = (*event->getMiniDoublets());
    SDL::hits& hitsInGPU = (*event->getHits());
    SDL::modules& modulesInGPU = (*event->getModules());
    SDL::objectRanges& rangesInGPU = (*event->getRanges());

    unsigned int i = *(modulesInGPU.nLowerModules);
    unsigned int idx = i;//modulesInGPU.lowerModuleIndices[i];
    int npLS = segmentsInGPU.nSegments[idx];
    for (unsigned int jdx = 0; jdx < segmentsInGPU.nSegments[idx]; jdx++)
    {
        unsigned int sgIdx = rangesInGPU.segmentModuleIndices[idx] + jdx;
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
void debugPrintOutlierMultiplicities(SDL::Event* event)
{
    SDL::trackCandidates& trackCandidatesInGPU = (*event->getTrackCandidates());
    SDL::triplets& tripletsInGPU = (*event->getTriplets());
    SDL::segments& segmentsInGPU = (*event->getSegments());
    SDL::miniDoublets& miniDoubletsInGPU = (*event->getMiniDoublets());
    //SDL::hits& hitsInGPU = (*event->getHits());
    SDL::modules& modulesInGPU = (*event->getModules());
    SDL::objectRanges& rangesInGPU = (*event->getRanges());
    //int nTrackCandidates = 0;
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
