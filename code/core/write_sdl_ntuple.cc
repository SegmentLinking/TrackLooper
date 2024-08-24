#include "write_sdl_ntuple.h"
//________________________________________________________________________________________________________________________________
void createOutputBranches()
{
    createOutputBranches_v2();
}

//________________________________________________________________________________________________________________________________
void createOutputBranches_v1()
{
    createRequiredOutputBranches();
    createOptionalOutputBranches();
}

//________________________________________________________________________________________________________________________________
void fillOutputBranches(SDL::Event* event)
{
    fillOutputBranches_v2(event);
}

//________________________________________________________________________________________________________________________________
void fillOutputBranches_v1(SDL::Event* event)
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
    
    //hubert
    ana.tx->createBranch<vector<float>>("MD_dphi");
    ana.tx->createBranch<vector<float>>("MD_dz");
    
    
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
    std::vector<int> sim_TC_matched_for_duplicate(trk.sim_pt().size());

    // Intermediate variables to keep track of matched sim tracks for a given track candidate
    std::vector<std::vector<int>> tc_matched_simIdx;

    // ============ Track candidates =============
    SDL::trackCandidatesBuffer<alpaka::DevCpu>& trackCandidatesInGPU = (*event->getTrackCandidates());
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
            sim_TC_matched_for_duplicate.at(idx) += 1;
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
            if (sim_TC_matched_for_duplicate[simidx] > 1)
            {
                isDuplicate = true;
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
    SDL::pixelQuintupletsBuffer<alpaka::DevCpu>& pixelQuintupletsInGPU = (*event->getPixelQuintuplets());
    SDL::quintupletsBuffer<alpaka::DevCpu>& quintupletsInGPU = (*event->getQuintuplets());
    SDL::segmentsBuffer<alpaka::DevCpu>& segmentsInGPU = (*event->getSegments());
    SDL::modulesBuffer<alpaka::DevCpu>& modulesInGPU = (*event->getModules());
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
    SDL::quintupletsBuffer<alpaka::DevCpu>& quintupletsInGPU = (*event->getQuintuplets());
    SDL::objectRangesBuffer<alpaka::DevCpu>& rangesInGPU = (*event->getRanges());
    SDL::modulesBuffer<alpaka::DevCpu>& modulesInGPU = (*event->getModules());
    const float kRinv1GeVf = (2.99792458e-3 * 3.8);
    int n_accepted_simtrk = ana.tx->getBranch<vector<int>>("sim_TC_matched").size();

    std::vector<int> sim_t5_matched(n_accepted_simtrk);
    std::vector<std::vector<int>> t5_matched_simIdx;

    for (unsigned int lowerModuleIdx = 0; lowerModuleIdx < *(modulesInGPU.nLowerModules); ++lowerModuleIdx)
    {
        int nQuintuplets = quintupletsInGPU.nQuintuplets[lowerModuleIdx];
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
    SDL::pixelTripletsBuffer<alpaka::DevCpu>& pixelTripletsInGPU = (*event->getPixelTriplets());
    SDL::tripletsBuffer<alpaka::DevCpu>& tripletsInGPU = *(event->getTriplets());
    SDL::modulesBuffer<alpaka::DevCpu>& modulesInGPU = *(event->getModules());
    SDL::segmentsBuffer<alpaka::DevCpu>& segmentsInGPU = *(event->getSegments());
    SDL::hitsBuffer<alpaka::DevCpu>& hitsInGPU = *(event->getHits());
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
    SDL::segmentsBuffer<alpaka::DevCpu>& segmentsInGPU = (*event->getSegments());
    SDL::miniDoubletsBuffer<alpaka::DevCpu>& miniDoubletsInGPU = (*event->getMiniDoublets());
    SDL::hitsBuffer<alpaka::DevCpu>& hitsInGPU = (*event->getHits());
    SDL::modulesBuffer<alpaka::DevCpu>& modulesInGPU = (*event->getModules());
    SDL::objectRangesBuffer<alpaka::DevCpu>& rangesInGPU = (*event->getRanges());
    SDL::trackCandidatesBuffer<alpaka::DevCpu>& trackCandidatesInGPU = (*event->getTrackCandidates());

    std::set<unsigned int> mds_used_in_sg;
    std::map<unsigned int, unsigned int> md_index_map;
    std::map<unsigned int, unsigned int> sg_index_map;

    // Loop over modules (lower ones where the MDs are saved)
    unsigned int nTotalMD = 0;
    unsigned int nTotalLS = 0;
    for (unsigned int idx = 0; idx < *(modulesInGPU.nLowerModules); ++idx)
    {
        nTotalMD += miniDoubletsInGPU.nMDs[idx];
        nTotalLS += segmentsInGPU.nSegments[idx];
    }

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
    for (unsigned int idx = 0; idx < *(modulesInGPU.nLowerModules); ++idx)
    {
        // // Loop over minidoublets
        // for (unsigned int jdx = 0; jdx < miniDoubletsInGPU.nMDs[idx]; jdx++)
        // {
        //     // Get the actual index to the mini-doublet using rangesInGPU
        //     unsigned int mdIdx = rangesInGPU.miniDoubletModuleIndices[idx] + jdx;

        //     setGnnNtupleMiniDoublet(event, mdIdx);
        // }

        // Loop over segments
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

            ana.tx->pushbackToBranch<int>("LS_isFake", simidxs.size() == 0);
            ana.tx->pushbackToBranch<float>("LS_sim_pt"      , simidxs.size() > 0 ? trk.sim_pt           ()[simidxs[0]] : -999);
            ana.tx->pushbackToBranch<float>("LS_sim_eta"     , simidxs.size() > 0 ? trk.sim_eta          ()[simidxs[0]] : -999);
            ana.tx->pushbackToBranch<float>("LS_sim_phi"     , simidxs.size() > 0 ? trk.sim_phi          ()[simidxs[0]] : -999);
            ana.tx->pushbackToBranch<float>("LS_sim_pca_dxy" , simidxs.size() > 0 ? trk.sim_pca_dxy      ()[simidxs[0]] : -999);
            ana.tx->pushbackToBranch<float>("LS_sim_pca_dz"  , simidxs.size() > 0 ? trk.sim_pca_dz       ()[simidxs[0]] : -999);
            ana.tx->pushbackToBranch<int>  ("LS_sim_q"       , simidxs.size() > 0 ? trk.sim_q            ()[simidxs[0]] : -999);
            ana.tx->pushbackToBranch<int>  ("LS_sim_event"   , simidxs.size() > 0 ? trk.sim_event        ()[simidxs[0]] : -999);
            ana.tx->pushbackToBranch<int>  ("LS_sim_bx"      , simidxs.size() > 0 ? trk.sim_bunchCrossing()[simidxs[0]] : -999);
            ana.tx->pushbackToBranch<int>  ("LS_sim_pdgId"   , simidxs.size() > 0 ? trk.sim_pdgId        ()[simidxs[0]] : -999);
            ana.tx->pushbackToBranch<float>("LS_sim_vx"      , simidxs.size() > 0 ? trk.simvtx_x         ()[trk.sim_parentVtxIdx()[simidxs[0]]] : -999);
            ana.tx->pushbackToBranch<float>("LS_sim_vy"      , simidxs.size() > 0 ? trk.simvtx_y         ()[trk.sim_parentVtxIdx()[simidxs[0]]] : -999);
            ana.tx->pushbackToBranch<float>("LS_sim_vz"      , simidxs.size() > 0 ? trk.simvtx_z         ()[trk.sim_parentVtxIdx()[simidxs[0]]] : -999);
            ana.tx->pushbackToBranch<int>  ("LS_isInTrueTC"  , lss_used_in_true_tc.find(sgIdx) != lss_used_in_true_tc.end());

            sg_index_map[sgIdx] = ana.tx->getBranch<vector<int>>("LS_isFake").size() - 1;

            // // T5 eta and phi are computed using outer and innermost hits
            // SDL::CPU::Hit hitA(trk.ph2_x()[anchitidx], trk.ph2_y()[anchitidx], trk.ph2_z()[anchitidx]);
            // const float phi = hitA.phi();
            // const float eta = hitA.eta();

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

    std::cout <<  " mds_used_in_sg.size(): " << mds_used_in_sg.size() <<  std::endl;
}

//________________________________________________________________________________________________________________________________
void setGnnNtupleMiniDoublet(SDL::Event* event, unsigned int MD)
{
    // Get relevant information
    SDL::miniDoubletsBuffer<alpaka::DevCpu>& miniDoubletsInGPU = (*event->getMiniDoublets());
    SDL::hitsBuffer<alpaka::DevCpu>& hitsInGPU = (*event->getHits());

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
    
    //Huberts addition:
    float dphi  =  miniDoubletsInGPU.dphis[MD];
    float dz = miniDoubletsInGPU.dzs[MD];

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
    
    // hubert
    ana.tx->pushbackToBranch<float>("MD_dz", dz);
    ana.tx->pushbackToBranch<float>("MD_dphi", dphi);
    
    // ana.tx->pushbackToBranch<int>("MD_sim_idx", simidxs.size() > 0 ? simidxs[0] : -999);
}

//________________________________________________________________________________________________________________________________
std::tuple<int, float, float, float, int, vector<int>> parseTrackCandidate(SDL::Event* event, unsigned int idx)
{
    // Get the type of the track candidate
    SDL::trackCandidatesBuffer<alpaka::DevCpu>& trackCandidatesInGPU = (*event->getTrackCandidates());
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
    std::vector<int> simidx = matchedSimTrkIdxs(hit_idx, hit_type, true);
    int isFake = simidx.size() == 0;

    return {type, pt, eta, phi, isFake, simidx};
}

//________________________________________________________________________________________________________________________________
std::tuple<int, float, float, float, int, vector<int>, vector<float>> parseTrackCandidateAllMatch(SDL::Event* event, unsigned int idx)
{
    // Get the type of the track candidate
    SDL::trackCandidatesBuffer<alpaka::DevCpu>& trackCandidatesInGPU = (*event->getTrackCandidates());
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
    std::vector<int> simidx;
    std::vector<float> simidxfrac;
    std::tie(simidx, simidxfrac) = matchedSimTrkIdxsAndFracs(hit_idx, hit_type, false, 0);
    int isFake = true;
    for (size_t is = 0; is < simidx.size(); ++is)
    {
        if (simidxfrac.at(is) > 0.75)
        {
            isFake = false;
            break;
        }
    }

    return {type, pt, eta, phi, isFake, simidx, simidxfrac};
}

//________________________________________________________________________________________________________________________________
std::tuple<float, float, float, vector<unsigned int>, vector<unsigned int>> parsepT5(SDL::Event* event, unsigned int idx)
{
    // Get relevant information
    SDL::trackCandidatesBuffer<alpaka::DevCpu>& trackCandidatesInGPU = (*event->getTrackCandidates());
    SDL::tripletsBuffer<alpaka::DevCpu>& tripletsInGPU = (*event->getTriplets());
    SDL::segmentsBuffer<alpaka::DevCpu>& segmentsInGPU = (*event->getSegments());
    SDL::hitsBuffer<alpaka::DevCpu>& hitsInGPU = (*event->getHits());

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
    const int seedIdx = segmentsInGPU.seedIdx[pLS];
    const float pt_pLS = trk.see_pt()[seedIdx];
    const float eta_pLS = trk.see_eta()[seedIdx];
    const float phi_pLS = trk.see_phi()[seedIdx];

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
    SDL::trackCandidatesBuffer<alpaka::DevCpu>& trackCandidatesInGPU = (*event->getTrackCandidates());
    SDL::tripletsBuffer<alpaka::DevCpu>& tripletsInGPU = (*event->getTriplets());
    SDL::segmentsBuffer<alpaka::DevCpu>& segmentsInGPU = (*event->getSegments());
    SDL::hitsBuffer<alpaka::DevCpu>& hitsInGPU = (*event->getHits());

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
    const int seedIdx = segmentsInGPU.seedIdx[pLS];
    const float pt_pLS = trk.see_pt()[seedIdx];
    const float eta_pLS = trk.see_eta()[seedIdx];
    const float phi_pLS = trk.see_phi()[seedIdx];

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
    SDL::trackCandidatesBuffer<alpaka::DevCpu>& trackCandidatesInGPU = (*event->getTrackCandidates());
    SDL::tripletsBuffer<alpaka::DevCpu>& tripletsInGPU = (*event->getTriplets());
    SDL::hitsBuffer<alpaka::DevCpu>& hitsInGPU = (*event->getHits());
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
    const float pt = (ptAv_in + ptAv_out) / 2.;

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
    SDL::trackCandidatesBuffer<alpaka::DevCpu>& trackCandidatesInGPU = (*event->getTrackCandidates());
    SDL::segmentsBuffer<alpaka::DevCpu>& segmentsInGPU = (*event->getSegments());

    // Getting pLS index
    unsigned int pLS = trackCandidatesInGPU.directObjectIndices[idx];

    // Getting pt eta and phi
    const int seedIdx = segmentsInGPU.seedIdx[pLS];
    const float pt = trk.see_pt()[seedIdx];
    const float eta = trk.see_eta()[seedIdx];
    const float phi = trk.see_phi()[seedIdx];

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
    SDL::modulesBuffer<alpaka::DevCpu>& modulesInGPU = (*event->getModules());
    SDL::objectRangesBuffer<alpaka::DevCpu>& rangesInGPU = (*event->getRanges());

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
    SDL::miniDoubletsBuffer<alpaka::DevCpu>& miniDoubletsInGPU = (*event->getMiniDoublets());
    SDL::modulesBuffer<alpaka::DevCpu>& modulesInGPU = (*event->getModules());

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
    SDL::miniDoubletsBuffer<alpaka::DevCpu>& miniDoubletsInGPU = (*event->getMiniDoublets());
    SDL::hitsBuffer<alpaka::DevCpu>& hitsInGPU = (*event->getHits());
    SDL::modulesBuffer<alpaka::DevCpu>& modulesInGPU = (*event->getModules());
    SDL::objectRangesBuffer<alpaka::DevCpu>& rangesInGPU = (*event->getRanges());

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
    SDL::segmentsBuffer<alpaka::DevCpu>& segmentsInGPU = (*event->getSegments());
    SDL::miniDoubletsBuffer<alpaka::DevCpu>& miniDoubletsInGPU = (*event->getMiniDoublets());
    SDL::hitsBuffer<alpaka::DevCpu>& hitsInGPU = (*event->getHits());
    SDL::modulesBuffer<alpaka::DevCpu>& modulesInGPU = (*event->getModules());
    SDL::objectRangesBuffer<alpaka::DevCpu>& rangesInGPU = (*event->getRanges());

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
    SDL::segmentsBuffer<alpaka::DevCpu>& segmentsInGPU = (*event->getSegments());
    SDL::miniDoubletsBuffer<alpaka::DevCpu>& miniDoubletsInGPU = (*event->getMiniDoublets());
    SDL::hitsBuffer<alpaka::DevCpu>& hitsInGPU = (*event->getHits());
    SDL::modulesBuffer<alpaka::DevCpu>& modulesInGPU = (*event->getModules());
    SDL::objectRangesBuffer<alpaka::DevCpu>& rangesInGPU = (*event->getRanges());

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
    SDL::tripletsBuffer<alpaka::DevCpu>& tripletsInGPU = (*event->getTriplets());
    SDL::segmentsBuffer<alpaka::DevCpu>& segmentsInGPU = (*event->getSegments());
    SDL::miniDoubletsBuffer<alpaka::DevCpu>& miniDoubletsInGPU = (*event->getMiniDoublets());
    SDL::hitsBuffer<alpaka::DevCpu>& hitsInGPU = (*event->getHits());
    SDL::modulesBuffer<alpaka::DevCpu>& modulesInGPU = (*event->getModules());
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
    SDL::trackCandidatesBuffer<alpaka::DevCpu>& trackCandidatesInGPU = (*event->getTrackCandidates());
    SDL::tripletsBuffer<alpaka::DevCpu>& tripletsInGPU = (*event->getTriplets());
    SDL::segmentsBuffer<alpaka::DevCpu>& segmentsInGPU = (*event->getSegments());
    SDL::miniDoubletsBuffer<alpaka::DevCpu>& miniDoubletsInGPU = (*event->getMiniDoublets());
    SDL::modulesBuffer<alpaka::DevCpu>& modulesInGPU = (*event->getModules());
    SDL::objectRangesBuffer<alpaka::DevCpu>& rangesInGPU = (*event->getRanges());
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

//    (# of simulated track) && (is reconstructed)
// --------------------------------------------------
//    (# of simulated track)

//    (# of simulated track w/ pt > 0.9 GeV and pt < 1.0 GeV and "some selection requirement on the sim track") && (is reconstructed)
// ----------------------------------------------------------------------------------- --------------------------------------------------
//    (# of simulated track w/ pt > 0.9 GeV and pt < 1.0 GeV and "some selection requirement on the sim track")

//    (# of reconstructed track w/ pt > 0.9 GeV and pt < 1.0 GeV and "some selection requirement on the reconstructed track") && (is_fake)
// ----------------------------------------------------------------------------------- --------------------------------------------------
//    (# of reconstructed track w/ pt > 0.9 GeV and pt < 1.0 GeV and "some selection requirement on the reconstructed track")

//________________________________________________________________________________________________________________________________
void createOutputBranches_v2()
{
    //---------------------------------------------------------------------------------------------------------------------------------
    // Simulated Track Container
    //
    //    The container will hold per entry a simulated track in the event. Only the current bunch crossing, and
    //    primary vertex (hard-scattered) tracks will be saved to reduce the size of the output.
    //
    ana.tx->createBranch<vector<float>>        ("sim_pt");                          // pt
    ana.tx->createBranch<vector<float>>        ("sim_eta");                         // eta
    ana.tx->createBranch<vector<float>>        ("sim_phi");                         // phi
    ana.tx->createBranch<vector<float>>        ("sim_pca_dxy");                     // dxy of point of closest approach
    ana.tx->createBranch<vector<float>>        ("sim_pca_dz");                      // dz of point of clossest approach
    ana.tx->createBranch<vector<int>>          ("sim_q");                           // charge +1, -1, 0
    ana.tx->createBranch<vector<int>>          ("sim_pdgId");                       // pdgId
    ana.tx->createBranch<vector<float>>        ("sim_vx");                          // production vertex x position (values are derived from simvtx_* and sim_parentVtxIdx branches in the tracking ntuple)
    ana.tx->createBranch<vector<float>>        ("sim_vy");                          // production vertex y position (values are derived from simvtx_* and sim_parentVtxIdx branches in the tracking ntuple)
    ana.tx->createBranch<vector<float>>        ("sim_vz");                          // production vertex z position (values are derived from simvtx_* and sim_parentVtxIdx branches in the tracking ntuple)
    ana.tx->createBranch<vector<float>>        ("sim_vtxperp");                     // production vertex r (sqrt(x**2 + y**2)) position (values are derived from simvtx_* and sim_parentVtxIdx branches in the tracking ntuple)
    ana.tx->createBranch<vector<float>>        ("sim_trkNtupIdx");                  // idx of sim_* in the tracking ntuple (N.B. this may be redundant)
    ana.tx->createBranch<vector<int>>          ("sim_tcIdxBest");                   // idx to the best match (highest nhit match) tc_* container
    ana.tx->createBranch<vector<float>>        ("sim_tcIdxBestFrac");               // match fraction to the best match (highest nhit match) tc_* container
    ana.tx->createBranch<vector<int>>          ("sim_tcIdx");                       // idx to the best match (highest nhit match and > 75%) tc_* container
    ana.tx->createBranch<vector<vector<int>>>  ("sim_tcIdxAll");                    // list of idx to any matches (> 0%) to tc_* container
    ana.tx->createBranch<vector<vector<float>>>("sim_tcIdxAllFrac");                // list of match fraction for each match (> 0%) to tc_* container
    ana.tx->createBranch<vector<vector<int>>>  ("sim_mdIdxAll");                    // list of idx to matches (> 0%) to md_* container
    ana.tx->createBranch<vector<vector<float>>>("sim_mdIdxAllFrac");                // list of match fraction for each match (> 0%) to md_* container
    ana.tx->createBranch<vector<vector<int>>>  ("sim_lsIdxAll");                    // list of idx to matches (> 0%) to ls_* container
    ana.tx->createBranch<vector<vector<float>>>("sim_lsIdxAllFrac");                // list of match fraction for each match (> 0%) to ls_* container
    ana.tx->createBranch<vector<vector<int>>>  ("sim_t3IdxAll");                    // list of idx to matches (> 0%) to t3_* container
    ana.tx->createBranch<vector<vector<float>>>("sim_t3IdxAllFrac");                // list of match fraction for each match (> 0%) to t3_* container
    ana.tx->createBranch<vector<vector<int>>>  ("sim_t5IdxAll");                    // list of idx to matches (> 0%) to t5_* container
    ana.tx->createBranch<vector<vector<float>>>("sim_t5IdxAllFrac");                // list of match fraction for each match (> 0%) to t5_* container
    ana.tx->createBranch<vector<vector<int>>>  ("sim_plsIdxAll");                   // list of idx to matches (> 0%) to pls_* container
    ana.tx->createBranch<vector<vector<float>>>("sim_plsIdxAllFrac");               // list of match fraction for each match (> 0%) to pls_* container
    ana.tx->createBranch<vector<vector<int>>>  ("sim_pt3IdxAll");                   // list of idx to matches (> 0%) to pt3_* container
    ana.tx->createBranch<vector<vector<float>>>("sim_pt3IdxAllFrac");               // list of match fraction for each match (> 0%) to pt3_* container
    ana.tx->createBranch<vector<vector<int>>>  ("sim_pt5IdxAll");                   // list of idx to matches (> 0%) to pt5_* container
    ana.tx->createBranch<vector<vector<float>>>("sim_pt5IdxAllFrac");               // list of match fraction for each match (> 0%) to pt5_* container
    ana.tx->createBranch<vector<vector<float>>>("sim_simHitX");                     // list of simhit's X positions
    ana.tx->createBranch<vector<vector<float>>>("sim_simHitY");                     // list of simhit's Y positions
    ana.tx->createBranch<vector<vector<float>>>("sim_simHitZ");                     // list of simhit's Z positions
    ana.tx->createBranch<vector<vector<int>>>  ("sim_simHitDetId");                 // list of simhit's detId
    ana.tx->createBranch<vector<vector<int>>>  ("sim_simHitLayer");                 // list of simhit's layers (N.B. layer is numbered 1 2 3 4 5 6 for barrel, 7 8 9 10 11 for endcaps)
    ana.tx->createBranch<vector<vector<float>>>("sim_simHitDistxyHelix");           // list of simhit's distance in xy-plane to the expected point based on simhit's z position and helix formed from pt,eta,phi,vx,vy,vz,q of the simulated track
    ana.tx->createBranch<vector<vector<float>>>("sim_simHitLayerMinDistxyHelix");   // length of 11 float numbers with min(simHitDistxyHelix) value for each layer. Useful for finding e.g. "sim tracks that traversed barrel detector entirelyand left a reasonable hit in layer 1 2 3 4 5 6 layers."
    ana.tx->createBranch<vector<vector<float>>>("sim_simHitLayerMinDistxyPrevHit"); // length of 11 float numbers with min(simHitDistxyHelix) value for each layer. Useful for finding e.g. "sim tracks that traversed barrel detector entirelyand left a reasonable hit in layer 1 2 3 4 5 6 layers."
    ana.tx->createBranch<vector<vector<float>>>("sim_recoHitX");                    // list of recohit's X positions
    ana.tx->createBranch<vector<vector<float>>>("sim_recoHitY");                    // list of recohit's Y positions
    ana.tx->createBranch<vector<vector<float>>>("sim_recoHitZ");                    // list of recohit's Z positions
    ana.tx->createBranch<vector<vector<int>>>  ("sim_recoHitDetId");                // list of recohit's detId

    //---------------------------------------------------------------------------------------------------------------------------------
    // Track Candidates
    //
    //    The container will hold per entry a track candidate built by LST in the event.
    //
    ana.tx->createBranch<vector<float>>        ("tc_pt");                    // pt
    ana.tx->createBranch<vector<float>>        ("tc_eta");                   // eta
    ana.tx->createBranch<vector<float>>        ("tc_phi");                   // phi
    ana.tx->createBranch<vector<int>>          ("tc_type");                  // type = 7 (pT5), 5 (pT3), 4 (T5), 8 (pLS)
    ana.tx->createBranch<vector<int>>          ("tc_pt5Idx");                // index to the pt5_* if it is the said type, if not set to -999
    ana.tx->createBranch<vector<int>>          ("tc_pt3Idx");                // index to the pt3_* if it is the said type, if not set to -999
    ana.tx->createBranch<vector<int>>          ("tc_t5Idx");                 // index to the t5_*  if it is the said type, if not set to -999
    ana.tx->createBranch<vector<int>>          ("tc_plsIdx");                // index to the pls_* if it is the said type, if not set to -999
    ana.tx->createBranch<vector<int>>          ("tc_isFake");                // 1 if tc is fake 0 other if not
    ana.tx->createBranch<vector<int>>          ("tc_isDuplicate");           // 1 if tc is duplicate 0 other if not
    ana.tx->createBranch<vector<int>>          ("tc_simIdx");                // idx of best matched (highest nhit and > 75%) simulated track
    ana.tx->createBranch<vector<vector<int>>>  ("tc_simIdxAll");             // list of idx of all matched (> 0%) simulated track
    ana.tx->createBranch<vector<vector<float>>>("tc_simIdxAllFrac");         // list of idx of all matched (> 0%) simulated track

    //---------------------------------------------------------------------------------------------------------------------------------
    // Mini-Doublets (i.e. Two reco hits paired in a single pT-module of Outer Tracker of CMS, a.k.a. MD)
    //
    //    The container will hold per entry a mini-doublet built by LST in the event.
    //
    ana.tx->createBranch<vector<float>>        ("md_pt");                    // pt (computed based on delta phi change)
    ana.tx->createBranch<vector<float>>        ("md_eta");                   // eta (computed based on anchor hit's eta)
    ana.tx->createBranch<vector<float>>        ("md_phi"); // phi (computed based on anchor hit's phi)
    
#ifdef OUTPUT_MD_CUTS
    ana.tx->createBranch<vector<float>>        ("md_dphi"); 
    ana.tx->createBranch<vector<float>>        ("md_dphichange"); 
    ana.tx->createBranch<vector<float>>        ("md_dz"); 
#endif
    
    ana.tx->createBranch<vector<float>>        ("md_anchor_x");              // anchor hit x
    ana.tx->createBranch<vector<float>>        ("md_anchor_y");              // anchor hit y
    ana.tx->createBranch<vector<float>>        ("md_anchor_z");              // anchor hit z
    ana.tx->createBranch<vector<float>>        ("md_other_x");               // other hit x
    ana.tx->createBranch<vector<float>>        ("md_other_y");               // other hit y
    ana.tx->createBranch<vector<float>>        ("md_other_z");               // other hit z
    ana.tx->createBranch<vector<int>>          ("md_type");                  // type of the module where the mini-doublet sit (type = 1 (PS), 0 (2S))
    ana.tx->createBranch<vector<int>>          ("md_layer");                 // layer index of the module where the mini-doublet sit (layer = 1 2 3 4 5 6 (barrel) 7 8 9 10 11 (endcap))
    ana.tx->createBranch<vector<int>>          ("md_detId");                 // detId = detector unique ID that contains a lot of information that can be parsed later if needed
    ana.tx->createBranch<vector<int>>          ("md_isFake");                // 1 if md is fake 0 other if not
    ana.tx->createBranch<vector<int>>          ("md_simIdx");                // idx of best matched (highest nhit and > 75%) simulated track
    ana.tx->createBranch<vector<vector<int>>>  ("md_simIdxAll");             // list of idx of all matched (> 0%) simulated track
    ana.tx->createBranch<vector<vector<float>>>("md_simIdxAllFrac");         // list of idx of all matched (> 0%) simulated track

    //---------------------------------------------------------------------------------------------------------------------------------
    // Line Segments (i.e. Two mini-doublets, a.k.a. LS)
    //
    //    The container will hold per entry a line-segment built by LST in the event.
    //
    ana.tx->createBranch<vector<float>>        ("ls_pt");                    // pt (computed based on radius of the circle formed by three points: (origin), (anchor hit 1), (anchor hit 2))
    ana.tx->createBranch<vector<float>>        ("ls_eta");                   // eta (computed based on last anchor hit's eta)
    ana.tx->createBranch<vector<float>>        ("ls_phi");                   // phi (computed based on first anchor hit's phi)
    ana.tx->createBranch<vector<int>>          ("ls_mdIdx0");                // index to the first MD
    ana.tx->createBranch<vector<int>>          ("ls_mdIdx1");                // index to the second MD
    ana.tx->createBranch<vector<int>>          ("ls_isFake");                // 1 if md is fake 0 other if not
    ana.tx->createBranch<vector<int>>          ("ls_simIdx");                // idx of best matched (highest nhit and > 75%) simulated track

#ifdef OUTPUT_LS_CUTS
    ana.tx->createBranch<vector<float>>        ("ls_zLos");
    ana.tx->createBranch<vector<float>>        ("ls_zHis");
    ana.tx->createBranch<vector<float>>        ("ls_rtLos");
    ana.tx->createBranch<vector<float>>        ("ls_rtHis");
    ana.tx->createBranch<vector<float>>        ("ls_dPhis");
    ana.tx->createBranch<vector<float>>        ("ls_dPhiMins");
    ana.tx->createBranch<vector<float>>        ("ls_dPhiMaxs");
    ana.tx->createBranch<vector<float>>        ("ls_dPhiChanges");
    ana.tx->createBranch<vector<float>>        ("ls_dPhiChangeMins");
    ana.tx->createBranch<vector<float>>        ("ls_dPhiChangeMaxs");
    ana.tx->createBranch<vector<float>>        ("ls_dAlphaInners");
    ana.tx->createBranch<vector<float>>        ("ls_dAlphaOuters");
    ana.tx->createBranch<vector<float>>        ("ls_dAlphaInnerOuters");
#endif   
     
    ana.tx->createBranch<vector<vector<int>>>  ("ls_simIdxAll");             // list of idx of all matched (> 0%) simulated track
    ana.tx->createBranch<vector<vector<float>>>("ls_simIdxAllFrac");         // list of idx of all matched (> 0%) simulated track

    //---------------------------------------------------------------------------------------------------------------------------------
    // Triplets (i.e. Three mini-doublets, a.k.a. T3)
    //
    //    The container will hold per entry a triplets built by LST in the event.
    //
    ana.tx->createBranch<vector<float>>        ("t3_pt");                    // pt (computed based on radius of the circle formed by three points: anchor hit 1, 2, 3
    ana.tx->createBranch<vector<float>>        ("t3_eta");                   // eta (computed based on last anchor hit's eta)
    ana.tx->createBranch<vector<float>>        ("t3_phi");                   // phi (computed based on first anchor hit's phi)
    ana.tx->createBranch<vector<int>>          ("t3_lsIdx0");                // index to the first LS
    ana.tx->createBranch<vector<int>>          ("t3_lsIdx1");                // index to the second LS
    ana.tx->createBranch<vector<int>>          ("t3_isFake");                // 1 if t3 is fake 0 other if not
    ana.tx->createBranch<vector<int>>          ("t3_isDuplicate");           // 1 if t3 is duplicate 0 other if not
    ana.tx->createBranch<vector<int>>          ("t3_simIdx");                // idx of best matched (highest nhit and > 75%) simulated track
    ana.tx->createBranch<vector<vector<int>>>  ("t3_simIdxAll");             // list of idx of all matched (> 0%) simulated track
    ana.tx->createBranch<vector<vector<float>>>("t3_simIdxAllFrac");         // list of idx of all matched (> 0%) simulated track

    //---------------------------------------------------------------------------------------------------------------------------------
    // Quintuplets (i.e. Five mini-doublets, a.k.a. T5)
    //
    //    The container will hold per entry a quintuplet built by LST in the event.
    //
    ana.tx->createBranch<vector<float>>        ("t5_pt");                    // pt (computed based on average of the 4 circles formed by, (1, 2, 3), (2, 3, 4), (3, 4, 5), (1, 3, 5)
    ana.tx->createBranch<vector<float>>        ("t5_eta");                   // eta (computed based on last anchor hit's eta)
    ana.tx->createBranch<vector<float>>        ("t5_phi");                   // phi (computed based on first anchor hit's phi)
    ana.tx->createBranch<vector<int>>          ("t5_t3Idx0");                // index of first T3
    ana.tx->createBranch<vector<int>>          ("t5_t3Idx1");                // index of second T3
    ana.tx->createBranch<vector<int>>          ("t5_isFake");                // 1 if t5 is fake 0 other if not
    ana.tx->createBranch<vector<int>>          ("t5_isDuplicate");           // 1 if t5 is duplicate 0 other if not
    ana.tx->createBranch<vector<int>>          ("t5_simIdx");                // idx of best matched (highest nhit and > 75%) simulated track
    ana.tx->createBranch<vector<vector<int>>>  ("t5_simIdxAll");             // list of idx of all matched (> 0%) simulated track
    ana.tx->createBranch<vector<vector<float>>>("t5_simIdxAllFrac");         // list of idx of all matched (> 0%) simulated track

    //---------------------------------------------------------------------------------------------------------------------------------
    // Pixel Line Segments (a.k.a pLS)
    //
    //    The container will hold per entry a pixel line segment (built by an external algo, e.g. patatrack) accepted by LST in the event.
    //
    ana.tx->createBranch<vector<float>>        ("pls_pt");                    // pt (taken from pt of the 3-vector from see_stateTrajGlbPx/Py/Pz)
    ana.tx->createBranch<vector<float>>        ("pls_eta");                   // eta (taken from eta of the 3-vector from see_stateTrajGlbPx/Py/Pz)
    ana.tx->createBranch<vector<float>>        ("pls_phi");                   // phi (taken from phi of the 3-vector from see_stateTrajGlbPx/Py/Pz)
    ana.tx->createBranch<vector<int>>          ("pls_nhit");                  // Number of actual hit: 3 if triplet, 4 if quadruplet
    ana.tx->createBranch<vector<float>>        ("pls_hit0_x");                // pLS's reco hit0 x
    ana.tx->createBranch<vector<float>>        ("pls_hit0_y");                // pLS's reco hit0 y
    ana.tx->createBranch<vector<float>>        ("pls_hit0_z");                // pLS's reco hit0 z
    ana.tx->createBranch<vector<float>>        ("pls_hit1_x");                // pLS's reco hit1 x
    ana.tx->createBranch<vector<float>>        ("pls_hit1_y");                // pLS's reco hit1 y
    ana.tx->createBranch<vector<float>>        ("pls_hit1_z");                // pLS's reco hit1 z
    ana.tx->createBranch<vector<float>>        ("pls_hit2_x");                // pLS's reco hit2 x
    ana.tx->createBranch<vector<float>>        ("pls_hit2_y");                // pLS's reco hit2 y
    ana.tx->createBranch<vector<float>>        ("pls_hit2_z");                // pLS's reco hit2 z
    ana.tx->createBranch<vector<float>>        ("pls_hit3_x");                // pLS's reco hit3 x (if triplet, this is set to -999)
    ana.tx->createBranch<vector<float>>        ("pls_hit3_y");                // pLS's reco hit3 y (if triplet, this is set to -999)
    ana.tx->createBranch<vector<float>>        ("pls_hit3_z");                // pLS's reco hit3 z (if triplet, this is set to -999)
    ana.tx->createBranch<vector<int>>          ("pls_isFake");                // 1 if pLS is fake 0 other if not
    ana.tx->createBranch<vector<int>>          ("pls_isDuplicate");           // 1 if pLS is duplicate 0 other if not
    ana.tx->createBranch<vector<int>>          ("pls_simIdx");                // idx of best matched (highest nhit and > 75%) simulated track
    ana.tx->createBranch<vector<vector<int>>>  ("pls_simIdxAll");             // list of idx of all matched (> 0%) simulated track
    ana.tx->createBranch<vector<vector<float>>>("pls_simIdxAllFrac");         // list of idx of all matched (> 0%) simulated track

    //---------------------------------------------------------------------------------------------------------------------------------
    // pLS + T3 (i.e. an object where a pLS is linked with a T3, a.k.a. pT3)
    //
    //    The container will hold per entry a pT3 built by LST in the event.
    //
    ana.tx->createBranch<vector<float>>        ("pt3_pt");                    // pt (taken from the pLS)
    ana.tx->createBranch<vector<float>>        ("pt3_eta");                   // eta (taken from the pLS)
    ana.tx->createBranch<vector<float>>        ("pt3_phi");                   // phi (taken from the pLS)
    ana.tx->createBranch<vector<int>>          ("pt3_plsIdx");                // idx to pLS
    ana.tx->createBranch<vector<int>>          ("pt3_t3Idx");                 // idx to T3
    ana.tx->createBranch<vector<int>>          ("pt3_isFake");                // 1 if pT3 is fake 0 other if not
    ana.tx->createBranch<vector<int>>          ("pt3_isDuplicate");           // 1 if pT3 is duplicate 0 other if not
    ana.tx->createBranch<vector<int>>          ("pt3_simIdx");                // idx of best matched (highest nhit and > 75%) simulated track
    ana.tx->createBranch<vector<vector<int>>>  ("pt3_simIdxAll");             // list of idx of all matched (> 0%) simulated track
    ana.tx->createBranch<vector<vector<float>>>("pt3_simIdxAllFrac");         // list of idx of all matched (> 0%) simulated track

    //---------------------------------------------------------------------------------------------------------------------------------
    // pLS + T5 (i.e. an object where a pLS is linked with a T5, a.k.a. pT5)
    //
    //    The container will hold per entry a pT5 built by LST in the event.
    //
    ana.tx->createBranch<vector<float>>        ("pt5_pt");                    // pt (taken from the pLS)
    ana.tx->createBranch<vector<float>>        ("pt5_eta");                   // eta (taken from the pLS)
    ana.tx->createBranch<vector<float>>        ("pt5_phi");                   // phi (taken from the pLS)
    ana.tx->createBranch<vector<int>>          ("pt5_plsIdx");                // idx to pLS
    ana.tx->createBranch<vector<int>>          ("pt5_t5Idx");                 // idx to T5
    ana.tx->createBranch<vector<int>>          ("pt5_isFake");                // 1 if pT5 is fake 0 other if not
    ana.tx->createBranch<vector<int>>          ("pt5_isDuplicate");           // 1 if pT5 is duplicate 0 other if not
    ana.tx->createBranch<vector<int>>          ("pt5_simIdx");                // idx of best matched (highest nhit and > 75%) simulated track
    ana.tx->createBranch<vector<vector<int>>>  ("pt5_simIdxAll");             // list of idx of all matched (> 0%) simulated track
    ana.tx->createBranch<vector<vector<float>>>("pt5_simIdxAllFrac");         // list of idx of all matched (> 0%) simulated track

}

//________________________________________________________________________________________________________________________________
void fillOutputBranches_v2(SDL::Event* event)
{

    // This function will go through each object and fill in the branches defined in createOutputBranches_v2 function.

    //--------------------------------------------
    //
    //
    // Sim Tracks
    //
    //
    //--------------------------------------------

    // Total number of simulated tracks stored in the tracking ntuple
    // The entire list contains a lot more simulated tracks than we care. (e.g. simulated tracks from pileup, or non-current-bunch-crossing particles)
    int n_total_simtrk = trk.sim_pt().size();

    // Total number of simulated tracks with the condition that the simulated track came from a particle produced in the hard scattering and from the current bunch-crossing)
    // "accepted" here would mean that in the tracking ntuple (sim_bunchCrossing == 0 and sim_event == 0)
    int n_accepted_simtrk = 0;

    // Looping over the simulated tracks in the tracking ntuple
    for (unsigned int isimtrk = 0; isimtrk < trk.sim_pt().size(); ++isimtrk)
    {
        // Skip out-of-time pileup
        if (trk.sim_bunchCrossing()[isimtrk] != 0)
            continue;

        // Skip non-hard-scatter
        if (trk.sim_event()[isimtrk] != 0)
            continue;

        // Now we have a list of "accepted" tracks (no condition on vtx_z/perp, nor pt, eta etc are applied yet)

        // Fill the branch with simulated tracks.
        // N.B. these simulated tracks are looser than MTV denominator
        ana.tx->pushbackToBranch<float>("sim_pt", trk.sim_pt()[isimtrk]);
        ana.tx->pushbackToBranch<float>("sim_eta", trk.sim_eta()[isimtrk]);
        ana.tx->pushbackToBranch<float>("sim_phi", trk.sim_phi()[isimtrk]);
        ana.tx->pushbackToBranch<float>("sim_pca_dxy", trk.sim_pca_dxy()[isimtrk]);
        ana.tx->pushbackToBranch<float>("sim_pca_dz", trk.sim_pca_dz()[isimtrk]);
        ana.tx->pushbackToBranch<int>("sim_q", trk.sim_q()[isimtrk]);
        ana.tx->pushbackToBranch<int>("sim_pdgId", trk.sim_pdgId()[isimtrk]);

        // For vertex we need to look it up from simvtx info for the given simtrack
        int vtxidx = trk.sim_parentVtxIdx()[isimtrk]; // for each simulated track, there is an index that points to the production vertex
        ana.tx->pushbackToBranch<float>("sim_vx", trk.simvtx_x()[vtxidx]); // using the index we retrieve xyz position
        ana.tx->pushbackToBranch<float>("sim_vy", trk.simvtx_y()[vtxidx]);
        ana.tx->pushbackToBranch<float>("sim_vz", trk.simvtx_z()[vtxidx]);
        ana.tx->pushbackToBranch<float>("sim_vtxperp", sqrt(trk.simvtx_x()[vtxidx]*trk.simvtx_x()[vtxidx] + trk.simvtx_y()[vtxidx]*trk.simvtx_y()[vtxidx]));

        // Retrieve some track parameter information so we can build a helix
        float pt = trk.sim_pt()[isimtrk];
        float eta = trk.sim_eta()[isimtrk];
        float phi = trk.sim_phi()[isimtrk];
        float vx = trk.simvtx_x()[vtxidx];
        float vy = trk.simvtx_y()[vtxidx];
        float vz = trk.simvtx_z()[vtxidx];
        float charge = trk.sim_q()[isimtrk];

        // Build the helix model. This model is useful to compute some specific expected hits.
        SDLMath::Helix helix(pt, eta, phi, vx, vy, vz, charge);

        // Information to keep track of so we can save to output
        std::vector<int> simHitLayer;
        std::vector<float> simHitDistxyHelix;
        std::vector<float> simHitX;
        std::vector<float> simHitY;
        std::vector<float> simHitZ;
        std::vector<int> simHitDetId;
        std::vector<float> recoHitX;
        std::vector<float> recoHitY;
        std::vector<float> recoHitZ;
        std::vector<int> recoHitDetId;
        std::vector<float> simHitLayerMinDistxyHelix(11, 999);

        std::vector<std::vector<int>> simHitIdxs(11);

        // Loop over the simhits (truth hits)
        for (size_t isimhit = 0; isimhit < trk.sim_simHitIdx()[isimtrk].size(); ++isimhit)
        {

            // Retrieve the actual index to the simhit_* container of the tracking ntuple
            int isimhitidx = trk.sim_simHitIdx()[isimtrk][isimhit];

            // Following computes the distance of the simhit's actual positionin xy to the "expected" xy position based on simhit's z position.
            // i.e. Take simhit's z position -> plug them into helix parametric function to obtain the xy position for that given z.
            // Then compare the computed xy position from the helix to the simhit's actualy xy position.
            // This is a measure of "how off from the original trajectory the simhits are?"
            // For example, if the particle got deflected early on due to material, then the xy position distance would be large.
            float distxyconsistent = distxySimHitConsistentWithHelix(helix, isimhitidx);

            // Also retrieve some basic information about the simhit's location (layers, isbarrel?, etc.)
            int subdet = trk.simhit_subdet()[isimhitidx]; // subdet == 4 means endcap of the outer tracker, subdet == 5 means barrel of the outer tracker)
            int is_endcap = subdet == 4;

            // Now compute "logical layer" index
            // N.B. if a hit is in the inner tracker, layer would be staying at layer = 0
            int layer = 0;
            if (subdet == 4 or subdet == 5) // this is not an outer tracker hit
                layer = trk.simhit_layer()[isimhitidx] + 6 * (is_endcap); // this accounting makes it so that you have layer 1 2 3 4 5 6 in the barrel, and 7 8 9 10 11 in the endcap. (becuase endcap is ph2_subdet == 4)

            // keep track of isimhits in each layers so we can compute mindistxy from previous hit in previous layer
            if (subdet == 4 or subdet == 5)
                simHitIdxs[layer-1].push_back(isimhitidx);

            // For this hit, now we push back to the vector that we are keeping track of
            simHitLayer.push_back(layer);
            simHitDistxyHelix.push_back(distxyconsistent);
            simHitX.push_back(trk.simhit_x()[isimhitidx]);
            simHitY.push_back(trk.simhit_y()[isimhitidx]);
            simHitZ.push_back(trk.simhit_z()[isimhitidx]);
            simHitDetId.push_back(trk.simhit_detId()[isimhitidx]);

            // Also retrieve all the reco-hits matched to this simhit and also aggregate them
            for (size_t irecohit = 0; irecohit < trk.simhit_hitIdx()[isimhitidx].size(); ++irecohit)
            {
                recoHitX.push_back(trk.ph2_x()[trk.simhit_hitIdx()[isimhitidx][irecohit]]);
                recoHitY.push_back(trk.ph2_y()[trk.simhit_hitIdx()[isimhitidx][irecohit]]);
                recoHitZ.push_back(trk.ph2_z()[trk.simhit_hitIdx()[isimhitidx][irecohit]]);
                recoHitDetId.push_back(trk.ph2_detId()[trk.simhit_hitIdx()[isimhitidx][irecohit]]);
            }

            // If the given simhit that we are dealing with is not in the outer tracker (i.e. layer == 0. see few lines above.)
            // then, skip this simhit and go to the next hit.
            if (layer == 0)
                continue;

            // If it is a outer tracker hit, then we keep track of out of the 11 layers, what is the minimum "DistxyHelix" (distance to the expected point in the helix in xy)
            // This variable will have a fixed 11 float numbers, and using this to restrict "at least one hit that is not too far from the expected helix" can be useful to select some interesting denominator tracks.
            if (distxyconsistent < simHitLayerMinDistxyHelix[layer-1])
            {
                simHitLayerMinDistxyHelix[layer-1] = distxyconsistent;
            }
        }

        std::vector<float> simHitLayerMinDistxyHelixPrevHit(11, 999);
        std::vector<float> simHitLayeriSimHitMinDixtxyHelixPrevHit(11, -999);

        // // The algorithm will be to start with the main helix from the sim information and get the isimhit with least distxy.
        // // Then, from that you find the min distxy and repeat
        // for (int ilogicallayer = 0; ilogicallayer < 11; ++ilogicallayer)
        // {
        //     int ilayer = ilogicallayer - 1;

        //     float prev_pt, prev_eta, prev_phi, prev_vx, prev_vy, prev_vz;

        //     if (ilayer == 0)
        //     {
        //         prev_pt = pt;
        //         prev_eta = eta;
        //         prev_phi = phi;
        //         prev_vx = vx;
        //         prev_vy = vy;
        //         prev_vz = vz;
        //     }
        //     else
        //     {
        //         int isimhitidx = simHitLayeriSimHitMinDixtxyHelixPrevHit[ilayer - 1];
        //         TVector3 pp(trk.simhit_px()[isimhitidx], trk.simhit_py()[isimhitidx], trk.simhit_pz()[isimhitidx]);
        //         prev_pt = pp.Pt();
        //         prev_eta = pp.Eta();
        //         prev_phi = pp.Phi();
        //         prev_vx = trk.simhit_x()[isimhitidx];
        //         prev_vy = trk.simhit_y()[isimhitidx];
        //         prev_vz = trk.simhit_z()[isimhitidx];
        //     }

        //     SDLMath::Helix prev_helix(prev_pt, prev_eta, prev_phi, prev_vx, prev_vy, prev_vz, charge);

        //     for (int isimhit = 0; isimhit < simHitIdxs[ilayer].size(); ++isimhit)
        //     {
        //         int isimhitidx = simHitIdxs[ilayer][isimhit];
        //         float distxyconsistent = distxySimHitConsistentWithHelix(prev_helix, isimhitidx);
        //         if (simHitLayerMinDistxyHelixPrevHit[ilayer] > distxyconsistent)
        //         {
        //             simHitLayerMinDistxyHelixPrevHit[ilayer] = distxyconsistent;
        //             simHitLayeriSimHitMinDixtxyHelixPrevHit[ilayer] = isimhitidx;
        //         }
        //     }
        // }

        // Now we fill the branch
        ana.tx->pushbackToBranch<vector<int>>  ("sim_simHitLayer", simHitLayer);
        ana.tx->pushbackToBranch<vector<float>>("sim_simHitDistxyHelix", simHitDistxyHelix);
        ana.tx->pushbackToBranch<vector<float>>("sim_simHitLayerMinDistxyHelix", simHitLayerMinDistxyHelix);
        ana.tx->pushbackToBranch<vector<float>>("sim_simHitLayerMinDistxyPrevHit", simHitLayerMinDistxyHelixPrevHit);
        ana.tx->pushbackToBranch<vector<float>>("sim_simHitX", simHitX);
        ana.tx->pushbackToBranch<vector<float>>("sim_simHitY", simHitY);
        ana.tx->pushbackToBranch<vector<float>>("sim_simHitZ", simHitZ);
        ana.tx->pushbackToBranch<vector<int>>("sim_simHitDetId", simHitDetId);
        ana.tx->pushbackToBranch<vector<float>>("sim_recoHitX", recoHitX);
        ana.tx->pushbackToBranch<vector<float>>("sim_recoHitY", recoHitY);
        ana.tx->pushbackToBranch<vector<float>>("sim_recoHitZ", recoHitZ);
        ana.tx->pushbackToBranch<vector<int>>("sim_recoHitDetId", recoHitDetId);

        // The trkNtupIdx is the idx in the trackingNtuple
        ana.tx->pushbackToBranch<float>("sim_trkNtupIdx", isimtrk);

        // Increase the counter for accepted simtrk
        n_accepted_simtrk++;
    }





    // From GPU get some information
    SDL::modulesBuffer          <alpaka::DevCpu>& modulesInGPU          = (*event->getModules());
    SDL::hitsBuffer             <alpaka::DevCpu>& hitsInGPU             = (*event->getHits());
    SDL::miniDoubletsBuffer     <alpaka::DevCpu>& miniDoubletsInGPU     = (*event->getMiniDoublets());
    SDL::segmentsBuffer         <alpaka::DevCpu>& segmentsInGPU         = (*event->getSegments());
    SDL::trackCandidatesBuffer  <alpaka::DevCpu>& trackCandidatesInGPU  = (*event->getTrackCandidates());
    SDL::objectRangesBuffer     <alpaka::DevCpu>& rangesInGPU           = (*event->getRanges());
    SDL::pixelQuintupletsBuffer <alpaka::DevCpu>& pixelQuintupletsInGPU = (*event->getPixelQuintuplets());
    SDL::quintupletsBuffer      <alpaka::DevCpu>& quintupletsInGPU      = (*event->getQuintuplets());
    SDL::tripletsBuffer         <alpaka::DevCpu>& tripletsInGPU         = (*event->getTriplets());
    SDL::pixelTripletsBuffer    <alpaka::DevCpu>& pixelTripletsInGPU    = (*event->getPixelTriplets());



    //--------------------------------------------
    //
    //
    // Mini-Doublets
    //
    //
    //--------------------------------------------

    // Following are some vectors to keep track of the information to write to the ntuple
    // N.B. following two branches have a length for the entire sim track, but what actually will be written in sim_mdIdxAll branch is NOT that long
    // Later in the code, it will restrict to only the ones to write out.
    // The reason at this stage, the entire mdIdxAll is being tracked is to compute duplicate properly later on
    // When computing a duplicate object it is important to consider all simulated tracks including pileup tracks
    std::vector<std::vector<int>> sim_mdIdxAll(n_total_simtrk);
    std::vector<std::vector<float>> sim_mdIdxAllFrac(n_total_simtrk);
    std::vector<std::vector<int>> md_simIdxAll;
    std::vector<std::vector<float>> md_simIdxAllFrac;

    // global md index that will be used to keep track of md being outputted to the ntuple
    // each time a md is written out the following will be counted up
    unsigned int md_idx = 0;

    // map to keep track of (GPU mdIdx) -> (md_idx in ntuple output)
    // There is a specific mdIdx used to navigate the GPU array of mini-doublets
    std::map<unsigned int, unsigned int> md_idx_map;

    // First loop over the modules (roughly there are ~13k pair of pt modules)
    for (unsigned int idx = 0; idx < *(modulesInGPU.nLowerModules); ++idx)
    {
        // For each pt module pair, we loop over mini-doublets created
        for (unsigned int iMD = 0; iMD < miniDoubletsInGPU.nMDs[idx]; iMD++)
        {

            // Compute the specific MD index to access specific spot in the array of GPU memory
            unsigned int mdIdx = rangesInGPU.miniDoubletModuleIndices[idx] + iMD;

            // From that gpu memory index "mdIdx" -> output ntuple's md index is mapped
            // This is useful later when connecting higher level objects to point to specific one in the ntuple
            md_idx_map[mdIdx] = md_idx;

            // Access the list of hits in the mini-doublets (there are only two in this case)
            std::vector<unsigned int> hit_idx, hit_type;
            std::tie(hit_idx, hit_type) = getHitIdxsAndHitTypesFromMD(event, mdIdx);

            // And then compute matching between simtrack and the mini-doublets
            std::vector<int> simidx;
            std::vector<float> simidxfrac;
            std::tie(simidx, simidxfrac) = matchedSimTrkIdxsAndFracs(hit_idx, hit_type, false/*=verbose*/, /*match fraction greater than X% = */0);

            // Obtain the lower and upper hit information to compute some basic property of the mini-doublets
            unsigned int LowerHitIndex = miniDoubletsInGPU.anchorHitIndices[mdIdx];
            unsigned int UpperHitIndex = miniDoubletsInGPU.outerHitIndices[mdIdx];
            unsigned int hit0 = hitsInGPU.idxs[LowerHitIndex];
            unsigned int hit1 = hitsInGPU.idxs[UpperHitIndex];
            float anchor_x = hitsInGPU.xs[LowerHitIndex];
            float anchor_y = hitsInGPU.ys[LowerHitIndex];
            float anchor_z = hitsInGPU.zs[LowerHitIndex];
            float other_x = hitsInGPU.xs[UpperHitIndex];
            float other_y = hitsInGPU.ys[UpperHitIndex];
            float other_z = hitsInGPU.zs[UpperHitIndex];

            // Construct the anchor hit 3 vector
            SDL::CPU::Hit anchor_hit(anchor_x, anchor_y, anchor_z, LowerHitIndex);

            // Pt is computed via dphichange and the eta and phi are computed based on anchor hit
            float dphichange = miniDoubletsInGPU.dphichanges[mdIdx];
            float dphi = miniDoubletsInGPU.dphis[mdIdx];
            float dz = miniDoubletsInGPU.dzs[mdIdx];
            float k2Rinv1GeVf = (2.99792458e-3 * 3.8) / 2;
            float pt = anchor_hit.rt() * k2Rinv1GeVf / sin(dphichange);
            float eta = anchor_hit.eta();
            float phi = anchor_hit.phi();

            // Obtain where the actual hit is located in terms of their layer, module, rod, and ring number
            int subdet = trk.ph2_subdet()[hit0];
            int is_endcap = subdet == 4;
            int layer = trk.ph2_layer()[hit0] + 6 * (is_endcap); // this accounting makes it so that you have layer 1 2 3 4 5 6 in the barrel, and 7 8 9 10 11 in the endcap. (becuase endcap is ph2_subdet == 4)
            int detId = trk.ph2_detId()[hit0];
            int ring = (detId & (15 << 12)) >> 12; // See https://github.com/SegmentLinking/TrackLooper/blob/158804cab7fd0976264a7bc4cee236f4986328c2/SDL/Module.cc and Module.h
            int isPS = is_endcap ? (layer <= 2 ? ring <= 10 : ring <= 7) : layer <= 3;

            // Write out the ntuple
            ana.tx->pushbackToBranch<float>("md_pt", pt);
            ana.tx->pushbackToBranch<float>("md_eta", eta);
            ana.tx->pushbackToBranch<float>("md_phi", phi);
            
#ifdef OUTPUT_MD_CUTS
            ana.tx->pushbackToBranch<float>("md_dphichange", dphichange);
            ana.tx->pushbackToBranch<float>("md_dphi", dphi);
            ana.tx->pushbackToBranch<float>("md_dz", dz);
#endif
            ana.tx->pushbackToBranch<float>("md_anchor_x", anchor_x);
            ana.tx->pushbackToBranch<float>("md_anchor_y", anchor_y);
            ana.tx->pushbackToBranch<float>("md_anchor_z", anchor_z);
            ana.tx->pushbackToBranch<float>("md_other_x", other_x);
            ana.tx->pushbackToBranch<float>("md_other_y", other_y);
            ana.tx->pushbackToBranch<float>("md_other_z", other_z);
            ana.tx->pushbackToBranch<int>("md_type", isPS);
            ana.tx->pushbackToBranch<int>("md_layer", layer);
            ana.tx->pushbackToBranch<int>("md_detId", detId);

            // Compute whether this is a fake
            bool isfake = true;
            for (size_t isim = 0; isim < simidx.size(); ++isim)
            {
                if (simidxfrac[isim] > 0.75)
                {
                    isfake = false;
                    break;
                }
            }
            ana.tx->pushbackToBranch<int>("md_isFake", isfake);

            // For this md, keep track of all the simidx that are matched
            md_simIdxAll.push_back(simidx);
            md_simIdxAllFrac.push_back(simidxfrac);

            // The book keeping of opposite mapping is done here
            // For each matched sim idx, we go back and keep track of which obj it is matched to.
            // Loop over all the matched sim idx
            for (size_t is = 0; is < simidx.size(); ++is)
            {
                // For this matched sim index keep track (sim -> md) mapping
                int sim_idx = simidx.at(is);
                float sim_idx_frac = simidxfrac.at(is);
                sim_mdIdxAll.at(sim_idx).push_back(md_idx);
                sim_mdIdxAllFrac.at(sim_idx).push_back(sim_idx_frac);
            }

            // Also, among the simidx matches, find the best match (highest fractional match)
            // N.B. the simidx is already returned sorted by highest number of "nhits" match
            // So as it loops over, the condition will ensure that the highest fraction with highest nhits will be matched with the priority given to highest fraction
            int md_simIdx = -999;
            float md_simIdxBestFrac = 0;
            for (size_t isim = 0; isim < simidx.size(); ++isim)
            {
                int thisidx = simidx[isim];
                float thisfrac = simidxfrac[isim];
                if (thisfrac > md_simIdxBestFrac and thisfrac > 0.75)
                {
                    md_simIdxBestFrac = thisfrac;
                    md_simIdx = thisidx;
                }
            }

            // the best match index will then be saved here
            ana.tx->pushbackToBranch<int>("md_simIdx", md_simIdx);

            // Count up the md_idx
            md_idx++;
        }
    }

    // Now save the (obj -> simidx) mapping
    ana.tx->setBranch<vector<vector<int>>>("md_simIdxAll", md_simIdxAll);
    ana.tx->setBranch<vector<vector<float>>>("md_simIdxAllFrac", md_simIdxAllFrac);

    // Not all (sim->objIdx) will be saved but only for the sim that is from hard scatter and current bunch crossing
    // So a restriction up to only "n_accepted_simtrk" done by chopping off the rest
    // N.B. the reason we can simply take the first "n_accepted_simtrk" is because the tracking ntuple is organized such that those sim tracks show up on the first "n_accepted_simtrk" of tracks.
    std::vector<std::vector<int>> sim_mdIdxAll_to_write;
    std::vector<std::vector<float>> sim_mdIdxAllFrac_to_write;
    std::copy(sim_mdIdxAll.begin(), sim_mdIdxAll.begin() + n_accepted_simtrk, std::back_inserter(sim_mdIdxAll_to_write));
    std::copy(sim_mdIdxAllFrac.begin(), sim_mdIdxAllFrac.begin() + n_accepted_simtrk, std::back_inserter(sim_mdIdxAllFrac_to_write));
    ana.tx->setBranch<vector<vector<int>>>("sim_mdIdxAll", sim_mdIdxAll_to_write);
    ana.tx->setBranch<vector<vector<float>>>("sim_mdIdxAllFrac", sim_mdIdxAllFrac_to_write);






    //--------------------------------------------
    //
    //
    // Line Segments
    //
    //
    //--------------------------------------------


    // Following are some vectors to keep track of the information to write to the ntuple
    // N.B. following two branches have a length for the entire sim track, but what actually will be written in sim_objIdxAll branch is NOT that long
    // Later in the code, it will restrict to only the ones to write out.
    // The reason at this stage, the entire objIdxAll is being tracked is to compute duplicate properly later on
    // When computing a duplicate object it is important to consider all simulated tracks including pileup tracks
    std::vector<std::vector<int>> sim_lsIdxAll(n_total_simtrk);
    std::vector<std::vector<float>> sim_lsIdxAllFrac(n_total_simtrk);
    std::vector<std::vector<int>> ls_simIdxAll;
    std::vector<std::vector<float>> ls_simIdxAllFrac;

    // global index that will be used to keep track of obj being outputted to the ntuple
    // each time a obj is written out the following will be counted up
    unsigned int ls_idx = 0;

    // map to keep track of (GPU objIdx) -> (obj_idx in ntuple output)
    // There is a specific objIdx used to navigate the GPU array of mini-doublets
    std::map<unsigned int, unsigned int> ls_idx_map;

    // First loop over the modules (roughly there are ~13k pair of pt modules)
    for (unsigned int idx = 0; idx < *(modulesInGPU.nLowerModules); ++idx)
    {
        // For each pt module pair, we loop over objects created
        for (unsigned int iLS = 0; iLS < segmentsInGPU.nSegments[idx]; iLS++)
        {

            // Compute the specific obj index to access specific spot in the array of GPU memory
            unsigned int lsIdx = rangesInGPU.segmentModuleIndices[idx] + iLS;

            // From that gpu memory index "objIdx" -> output ntuple's obj index is mapped
            // This is useful later when connecting higher level objects to point to specific one in the ntuple
            ls_idx_map[lsIdx] = ls_idx;

            // Access the list of hits in the objects (there are only two in this case)
            std::vector<unsigned int> hit_idx, hit_type;
            std::tie(hit_idx, hit_type) = getHitIdxsAndHitTypesFromLS(event, lsIdx);

            // And then compute matching between simtrack and the objects
            std::vector<int> simidx;
            std::vector<float> simidxfrac;
            std::tie(simidx, simidxfrac) = matchedSimTrkIdxsAndFracs(hit_idx, hit_type, false, 0);
            std::vector<unsigned int> mdIdxs = getMDsFromLS(event, lsIdx);

            // Computing line segment pt estimate (assuming beam spot is at zero)
            SDL::CPU::Hit hitA(0, 0, 0);
            SDL::CPU::Hit hitB(hitsInGPU.xs[hit_idx[0]], hitsInGPU.ys[hit_idx[0]], hitsInGPU.zs[hit_idx[0]]);
            SDL::CPU::Hit hitC(hitsInGPU.xs[hit_idx[2]], hitsInGPU.ys[hit_idx[2]], hitsInGPU.zs[hit_idx[2]]);
            SDL::CPU::Hit center = SDL::CPU::MathUtil::getCenterFromThreePoints(hitA, hitB, hitC);
            float pt = SDL::CPU::MathUtil::ptEstimateFromRadius(center.rt());
            float eta = hitC.eta();
            float phi = hitB.phi();
            
            
#ifdef OUTPUT_LS_CUTS
            float zHi = segmentsInGPU.zHis[lsIdx];
            float zLo = segmentsInGPU.zLos[lsIdx];
            float rtHi = segmentsInGPU.rtHis[lsIdx];
            float rtLo = segmentsInGPU.rtLos[lsIdx];
	    float dAlphaInner = segmentsInGPU.dAlphaInners[lsIdx];
	    float dAlphaOuter = segmentsInGPU.dAlphaOuters[lsIdx];
	    float dAlphaInnerOuter = segmentsInGPU.dAlphaInnerOuters[lsIdx];
#endif
            float dPhi = segmentsInGPU.dPhis[lsIdx];
            float dPhiMin = segmentsInGPU.dPhiMins[lsIdx];
            float dPhiMax = segmentsInGPU.dPhiMaxs[lsIdx];
            float dPhiChange = segmentsInGPU.dPhiChanges[lsIdx];
            float dPhiChangeMin = segmentsInGPU.dPhiChangeMins[lsIdx];
            float dPhiChangeMax = segmentsInGPU.dPhiChangeMaxs[lsIdx];

            // Write out the ntuple
            ana.tx->pushbackToBranch<float>("ls_pt", pt);
            ana.tx->pushbackToBranch<float>("ls_eta", eta);
            ana.tx->pushbackToBranch<float>("ls_phi", phi);
            
#ifdef OUTPUT_LS_CUTS
            ana.tx->pushbackToBranch<float>("ls_zHis",zHi);
            ana.tx->pushbackToBranch<float>("ls_zLos",zLo);
            ana.tx->pushbackToBranch<float>("ls_rtHis",rtHi);
            ana.tx->pushbackToBranch<float>("ls_rtLos",rtLo);
            ana.tx->pushbackToBranch<float>("ls_dPhis",dPhi);
            ana.tx->pushbackToBranch<float>("ls_dPhiMins",dPhiMin);
            ana.tx->pushbackToBranch<float>("ls_dPhiMaxs",dPhiMax);
            ana.tx->pushbackToBranch<float>("ls_dPhiChanges",dPhiChange);
            ana.tx->pushbackToBranch<float>("ls_dPhiChangeMins",dPhiChangeMin);
            ana.tx->pushbackToBranch<float>("ls_dPhiChangeMaxs",dPhiChangeMax);
	    ana.tx->pushbackToBranch<float>("ls_dAlphaInners",dAlphaInner);
	    ana.tx->pushbackToBranch<float>("ls_dAlphaOuters",dAlphaOuter);
	    ana.tx->pushbackToBranch<float>("ls_dAlphaInnerOuters",dAlphaInnerOuter);

#endif
            
            
            ana.tx->pushbackToBranch<int>("ls_mdIdx0", md_idx_map[mdIdxs[0]]);
            ana.tx->pushbackToBranch<int>("ls_mdIdx1", md_idx_map[mdIdxs[1]]);

            // Compute whether this is a fake
            bool isfake = true;
            for (size_t isim = 0; isim < simidx.size(); ++isim)
            {
                if (simidxfrac[isim] > 0.75)
                {
                    isfake = false;
                    break;
                }
            }
            ana.tx->pushbackToBranch<int>("ls_isFake", isfake);

            // For this obj, keep track of all the simidx that are matched
            ls_simIdxAll.push_back(simidx);
            ls_simIdxAllFrac.push_back(simidxfrac);

            // The book keeping of opposite mapping is done here
            // For each matched sim idx, we go back and keep track of which obj it is matched to.
            // Loop over all the matched sim idx
            for (size_t is = 0; is < simidx.size(); ++is)
            {
                int sim_idx = simidx.at(is);
                float sim_idx_frac = simidxfrac.at(is);
                if (sim_idx < n_total_simtrk)
                {
                    sim_lsIdxAll.at(sim_idx).push_back(ls_idx);
                    sim_lsIdxAllFrac.at(sim_idx).push_back(sim_idx_frac);
                }
            }

            // Also, among the simidx matches, find the best match (highest fractional match)
            // N.B. the simidx is already returned sorted by highest number of "nhits" match
            // So as it loops over, the condition will ensure that the highest fraction with highest nhits will be matched with the priority given to highest fraction
            int ls_simIdx = -999;
            float ls_simIdxBestFrac = 0;
            for (size_t isim = 0; isim < simidx.size(); ++isim)
            {
                int thisidx = simidx[isim];
                float thisfrac = simidxfrac[isim];
                if (thisfrac > ls_simIdxBestFrac and thisfrac > 0.75)
                {
                    ls_simIdxBestFrac = thisfrac;
                    ls_simIdx = thisidx;
                }
            }

            // the best match index will then be saved here
            ana.tx->pushbackToBranch<int>("ls_simIdx", ls_simIdx);

            // Count up the index
            ls_idx++;
        }
    }

    // Now save the (obj -> simidx) mapping
    ana.tx->setBranch<vector<vector<int>>>("ls_simIdxAll", ls_simIdxAll);
    ana.tx->setBranch<vector<vector<float>>>("ls_simIdxAllFrac", ls_simIdxAllFrac);

    // Not all (sim->objIdx) will be saved but only for the sim that is from hard scatter and current bunch crossing
    // So a restriction up to only "n_accepted_simtrk" done by chopping off the rest
    // N.B. the reason we can simply take the first "n_accepted_simtrk" is because the tracking ntuple is organized such that those sim tracks show up on the first "n_accepted_simtrk" of tracks.
    std::vector<std::vector<int>> sim_lsIdxAll_to_write;
    std::vector<std::vector<float>> sim_lsIdxAllFrac_to_write;
    std::copy(sim_lsIdxAll.begin(), sim_lsIdxAll.begin() + n_accepted_simtrk, std::back_inserter(sim_lsIdxAll_to_write));
    std::copy(sim_lsIdxAllFrac.begin(), sim_lsIdxAllFrac.begin() + n_accepted_simtrk, std::back_inserter(sim_lsIdxAllFrac_to_write));
    ana.tx->setBranch<vector<vector<int>>>("sim_lsIdxAll", sim_lsIdxAll_to_write);
    ana.tx->setBranch<vector<vector<float>>>("sim_lsIdxAllFrac", sim_lsIdxAllFrac_to_write);







    //--------------------------------------------
    //
    //
    // Triplet
    //
    //
    //--------------------------------------------

    std::vector<std::vector<int>> sim_t3IdxAll(n_total_simtrk);
    std::vector<std::vector<float>> sim_t3IdxAllFrac(n_total_simtrk);
    std::vector<std::vector<int>> t3_simIdxAll;
    std::vector<std::vector<float>> t3_simIdxAllFrac;
    // Then obtain the lower module index
    unsigned int t3_idx = 0; // global t3 index that will be used to keep track of t3 being outputted to the ntuple
    // map to keep track of (GPU t3Idx) -> (t3_idx in ntuple output)
    std::map<unsigned int, unsigned int> t3_idx_map;
    // printT3s(event);
    for (unsigned int idx = 0; idx < *(modulesInGPU.nLowerModules); ++idx)
    {
        unsigned int nmods = *(modulesInGPU.nLowerModules);
        for (unsigned int iT3 = 0; iT3 < tripletsInGPU.nTriplets[idx]; iT3++)
        {
            unsigned int t3Idx = rangesInGPU.tripletModuleIndices[idx] + iT3;
            t3_idx_map[t3Idx] = t3_idx;
            std::vector<unsigned int> hit_idx, hit_type;
            std::tie(hit_idx, hit_type) = getHitIdxsAndHitTypesFromT3(event, t3Idx);
            std::vector<int> simidx;
            std::vector<float> simidxfrac;
            std::tie(simidx, simidxfrac) = matchedSimTrkIdxsAndFracs(hit_idx, hit_type, false, 0);
            std::vector<unsigned int> lsIdxs = getLSsFromT3(event, t3Idx);
            ana.tx->pushbackToBranch<int>("t3_lsIdx0", ls_idx_map[lsIdxs[0]]);
            ana.tx->pushbackToBranch<int>("t3_lsIdx1", ls_idx_map[lsIdxs[1]]);
            // Computing line segment pt estimate (assuming beam spot is at zero)
            SDL::CPU::Hit hitA(hitsInGPU.xs[hit_idx[0]], hitsInGPU.ys[hit_idx[0]], hitsInGPU.zs[hit_idx[0]]);
            SDL::CPU::Hit hitB(hitsInGPU.xs[hit_idx[2]], hitsInGPU.ys[hit_idx[2]], hitsInGPU.zs[hit_idx[2]]);
            SDL::CPU::Hit hitC(hitsInGPU.xs[hit_idx[4]], hitsInGPU.ys[hit_idx[4]], hitsInGPU.zs[hit_idx[4]]);
            SDL::CPU::Hit center = SDL::CPU::MathUtil::getCenterFromThreePoints(hitA, hitB, hitC);
            float pt = SDL::CPU::MathUtil::ptEstimateFromRadius(center.rt());
            float eta = hitC.eta();
            float phi = hitA.phi();
            ana.tx->pushbackToBranch<float>("t3_pt", pt);
            ana.tx->pushbackToBranch<float>("t3_eta", eta);
            ana.tx->pushbackToBranch<float>("t3_phi", phi);
            bool isfake = true;
            for (size_t isim = 0; isim < simidx.size(); ++isim)
            {
                if (simidxfrac[isim] > 0.75)
                {
                    isfake = false;
                    break;
                }
            }
            ana.tx->pushbackToBranch<int>("t3_isFake", isfake);
            t3_simIdxAll.push_back(simidx);
            t3_simIdxAllFrac.push_back(simidxfrac);
            for (size_t is = 0; is < simidx.size(); ++is)
            {
                int sim_idx = simidx.at(is);
                float sim_idx_frac = simidxfrac.at(is);
                if (sim_idx < n_total_simtrk)
                {
                    sim_t3IdxAll.at(sim_idx).push_back(t3_idx);
                    sim_t3IdxAllFrac.at(sim_idx).push_back(sim_idx_frac);
                }
            }
            int t3_simIdx = -999;
            float t3_simIdxBestFrac = 0;
            for (size_t isim = 0; isim < simidx.size(); ++isim)
            {
                int thisidx = simidx[isim];
                float thisfrac = simidxfrac[isim];
                if (thisfrac > t3_simIdxBestFrac and thisfrac > 0.75)
                {
                    t3_simIdxBestFrac = thisfrac;
                    t3_simIdx = thisidx;
                }
            }
            ana.tx->pushbackToBranch<int>("t3_simIdx", t3_simIdx);
            // count global
            t3_idx++;
        }
    }
    ana.tx->setBranch<vector<vector<int>>>("t3_simIdxAll", t3_simIdxAll);
    ana.tx->setBranch<vector<vector<float>>>("t3_simIdxAllFrac", t3_simIdxAllFrac);
    std::vector<std::vector<int>> sim_t3IdxAll_to_write;
    std::vector<std::vector<float>> sim_t3IdxAllFrac_to_write;
    std::copy(sim_t3IdxAll.begin(), sim_t3IdxAll.begin() + n_accepted_simtrk, std::back_inserter(sim_t3IdxAll_to_write));
    std::copy(sim_t3IdxAllFrac.begin(), sim_t3IdxAllFrac.begin() + n_accepted_simtrk, std::back_inserter(sim_t3IdxAllFrac_to_write));
    ana.tx->setBranch<vector<vector<int>>>("sim_t3IdxAll", sim_t3IdxAll_to_write);
    ana.tx->setBranch<vector<vector<float>>>("sim_t3IdxAllFrac", sim_t3IdxAllFrac_to_write);

    // Using the intermedaite variables to compute whether a given object is a duplicate
    vector<int> t3_isDuplicate(t3_simIdxAll.size());

    // Loop over the objects
    for (unsigned int t3_idx = 0; t3_idx < t3_simIdxAll.size(); ++t3_idx)
    {
        bool isDuplicate = false;
        // Loop over the sim idx matched to this object
        for (unsigned int isim = 0; isim < t3_simIdxAll[t3_idx].size(); ++isim)
        {
            int sim_idx = t3_simIdxAll[t3_idx][isim];
            int n_sim_matched = 0;
            for (size_t ism = 0; ism < sim_t3IdxAll.at(sim_idx).size(); ++ism)
            {
                if (sim_t3IdxAllFrac.at(sim_idx).at(ism) > 0.75)
                {
                    n_sim_matched += 1;
                    if (n_sim_matched > 1)
                    {
                        isDuplicate = true;
                        break;
                    }
                }
            }
        }
        t3_isDuplicate[t3_idx] = isDuplicate;
    }
    ana.tx->setBranch<vector<int>>("t3_isDuplicate", t3_isDuplicate);







    //--------------------------------------------
    //
    //
    // Quintuplet
    //
    //
    //--------------------------------------------

    std::vector<std::vector<int>> sim_t5IdxAll(n_total_simtrk);
    std::vector<std::vector<float>> sim_t5IdxAllFrac(n_total_simtrk);
    std::vector<std::vector<int>> t5_simIdxAll;
    std::vector<std::vector<float>> t5_simIdxAllFrac;
    // Then obtain the lower module index
    unsigned int t5_idx = 0; // global t5 index that will be used to keep track of t5 being outputted to the ntuple
    // map to keep track of (GPU t5Idx) -> (t5_idx in ntuple output)
    std::map<unsigned int, unsigned int> t5_idx_map;
    // printT3s(event);
    for (unsigned int idx = 0; idx < *(modulesInGPU.nLowerModules); ++idx)
    {
        unsigned int nmods = *(modulesInGPU.nLowerModules);
        for (unsigned int iT5 = 0; iT5 < quintupletsInGPU.nQuintuplets[idx]; iT5++)
        {
            unsigned int t5Idx = rangesInGPU.quintupletModuleIndices[idx] + iT5;
            t5_idx_map[t5Idx] = t5_idx;
            std::vector<unsigned int> hit_idx, hit_type;
            std::tie(hit_idx, hit_type) = getHitIdxsAndHitTypesFromT5(event, t5Idx);
            std::vector<int> simidx;
            std::vector<float> simidxfrac;
            std::tie(simidx, simidxfrac) = matchedSimTrkIdxsAndFracs(hit_idx, hit_type, false, 0);
            std::vector<unsigned int> t3Idxs = getT3sFromT5(event, t5Idx);
            ana.tx->pushbackToBranch<int>("t5_t3Idx0", t3_idx_map[t3Idxs[0]]);
            ana.tx->pushbackToBranch<int>("t5_t3Idx1", t3_idx_map[t3Idxs[1]]);
            // Computing line segment pt estimate (assuming beam spot is at zero)
            SDL::CPU::Hit hitA(hitsInGPU.xs[hit_idx[0]], hitsInGPU.ys[hit_idx[0]], hitsInGPU.zs[hit_idx[0]]);
            SDL::CPU::Hit hitB(hitsInGPU.xs[hit_idx[2]], hitsInGPU.ys[hit_idx[2]], hitsInGPU.zs[hit_idx[2]]);
            SDL::CPU::Hit hitC(hitsInGPU.xs[hit_idx[4]], hitsInGPU.ys[hit_idx[4]], hitsInGPU.zs[hit_idx[4]]);
            SDL::CPU::Hit hitD(hitsInGPU.xs[hit_idx[6]], hitsInGPU.ys[hit_idx[6]], hitsInGPU.zs[hit_idx[6]]);
            SDL::CPU::Hit hitE(hitsInGPU.xs[hit_idx[8]], hitsInGPU.ys[hit_idx[8]], hitsInGPU.zs[hit_idx[8]]);
            SDL::CPU::Hit center = SDL::CPU::MathUtil::getCenterFromThreePoints(hitA, hitB, hitC);
            float pt = SDL::CPU::MathUtil::ptEstimateFromRadius(center.rt());
            SDL::CPU::Hit center2 = SDL::CPU::MathUtil::getCenterFromThreePoints(hitB, hitC, hitD);
            float pt2 = SDL::CPU::MathUtil::ptEstimateFromRadius(center2.rt());
            SDL::CPU::Hit center3 = SDL::CPU::MathUtil::getCenterFromThreePoints(hitC, hitD, hitE);
            float pt3 = SDL::CPU::MathUtil::ptEstimateFromRadius(center3.rt());
            SDL::CPU::Hit center4 = SDL::CPU::MathUtil::getCenterFromThreePoints(hitA, hitC, hitE);
            float pt4 = SDL::CPU::MathUtil::ptEstimateFromRadius(center4.rt());
            float ptavg = (pt + pt2 + pt3 + pt4) / 4;
            float eta = hitE.eta();
            float phi = hitA.phi();
            ana.tx->pushbackToBranch<float>("t5_pt", ptavg);
            ana.tx->pushbackToBranch<float>("t5_eta", eta);
            ana.tx->pushbackToBranch<float>("t5_phi", phi);
            bool isfake = true;
            for (size_t isim = 0; isim < simidx.size(); ++isim)
            {
                if (simidxfrac[isim] > 0.75)
                {
                    isfake = false;
                    break;
                }
            }
            ana.tx->pushbackToBranch<int>("t5_isFake", isfake);
            t5_simIdxAll.push_back(simidx);
            t5_simIdxAllFrac.push_back(simidxfrac);
            for (size_t is = 0; is < simidx.size(); ++is)
            {
                int sim_idx = simidx.at(is);
                float sim_idx_frac = simidxfrac.at(is);
                if (sim_idx < n_total_simtrk)
                {
                    sim_t5IdxAll.at(sim_idx).push_back(t5_idx);
                    sim_t5IdxAllFrac.at(sim_idx).push_back(sim_idx_frac);
                }
            }
            int t5_simIdx = -999;
            float t5_simIdxBestFrac = 0;
            for (size_t isim = 0; isim < simidx.size(); ++isim)
            {
                int thisidx = simidx[isim];
                float thisfrac = simidxfrac[isim];
                if (thisfrac > t5_simIdxBestFrac and thisfrac > 0.75)
                {
                    t5_simIdxBestFrac = thisfrac;
                    t5_simIdx = thisidx;
                }
            }
            ana.tx->pushbackToBranch<int>("t5_simIdx", t5_simIdx);
            // count global
            t5_idx++;
        }
    }
    ana.tx->setBranch<vector<vector<int>>>("t5_simIdxAll", t5_simIdxAll);
    ana.tx->setBranch<vector<vector<float>>>("t5_simIdxAllFrac", t5_simIdxAllFrac);
    std::vector<std::vector<int>> sim_t5IdxAll_to_write;
    std::vector<std::vector<float>> sim_t5IdxAllFrac_to_write;
    std::copy(sim_t5IdxAll.begin(), sim_t5IdxAll.begin() + n_accepted_simtrk, std::back_inserter(sim_t5IdxAll_to_write));
    std::copy(sim_t5IdxAllFrac.begin(), sim_t5IdxAllFrac.begin() + n_accepted_simtrk, std::back_inserter(sim_t5IdxAllFrac_to_write));
    ana.tx->setBranch<vector<vector<int>>>("sim_t5IdxAll", sim_t5IdxAll_to_write);
    ana.tx->setBranch<vector<vector<float>>>("sim_t5IdxAllFrac", sim_t5IdxAllFrac_to_write);

    // Using the intermedaite variables to compute whether a given object is a duplicate
    vector<int> t5_isDuplicate(t5_simIdxAll.size());
    // Loop over the objects
    for (unsigned int t5_idx = 0; t5_idx < t5_simIdxAll.size(); ++t5_idx)
    {
        bool isDuplicate = false;
        // Loop over the sim idx matched to this object
        for (unsigned int isim = 0; isim < t5_simIdxAll[t5_idx].size(); ++isim)
        {
            int sim_idx = t5_simIdxAll[t5_idx][isim];
            int n_sim_matched = 0;
            for (size_t ism = 0; ism < sim_t5IdxAll.at(sim_idx).size(); ++ism)
            {
                if (sim_t5IdxAllFrac.at(sim_idx).at(ism) > 0.75)
                {
                    n_sim_matched += 1;
                    if (n_sim_matched > 1)
                    {
                        isDuplicate = true;
                        break;
                    }
                }
            }
        }
        t5_isDuplicate[t5_idx] = isDuplicate;
    }
    ana.tx->setBranch<vector<int>>("t5_isDuplicate", t5_isDuplicate);







    //--------------------------------------------
    //
    //
    // pLS
    //
    //
    //--------------------------------------------

    std::vector<std::vector<int>> sim_plsIdxAll(n_total_simtrk);
    std::vector<std::vector<float>> sim_plsIdxAllFrac(n_total_simtrk);
    std::vector<std::vector<int>> pls_simIdxAll;
    std::vector<std::vector<float>> pls_simIdxAllFrac;
    // Then obtain the lower module index
    unsigned int pls_idx = 0; // global pls index that will be used to keep track of pls being outputted to the ntuple
    // map to keep track of (GPU plsIdx) -> (pls_idx in ntuple output)
    std::map<unsigned int, unsigned int> pls_idx_map;
    // printT3s(event);
    for (unsigned int idx = *(modulesInGPU.nLowerModules); idx <= *(modulesInGPU.nLowerModules); ++idx)
    {
        unsigned int nmods = *(modulesInGPU.nLowerModules);
        for (unsigned int ipLS = 0; ipLS < segmentsInGPU.nSegments[idx]; ipLS++)
        {
            unsigned int plsIdx = rangesInGPU.segmentModuleIndices[idx] + ipLS;
            pls_idx_map[plsIdx] = pls_idx;
            std::vector<unsigned int> hit_idx, hit_type;
            std::tie(hit_idx, hit_type) = getHitIdxsAndHitTypesFrompLS(event, ipLS);
            std::vector<int> simidx;
            std::vector<float> simidxfrac;
            std::tie(simidx, simidxfrac) = matchedSimTrkIdxsAndFracs(hit_idx, hit_type, false, 0);
            int seedIdx = segmentsInGPU.seedIdx[ipLS];
            ana.tx->pushbackToBranch<float>("pls_pt", trk.see_pt()[seedIdx]);
            ana.tx->pushbackToBranch<float>("pls_eta", trk.see_eta()[seedIdx]);
            ana.tx->pushbackToBranch<float>("pls_phi", trk.see_phi()[seedIdx]);
            ana.tx->pushbackToBranch<int>("pls_nhit", hit_idx.size());
            for (size_t ihit = 0; ihit < trk.see_hitIdx()[seedIdx].size(); ++ihit)
            {
                int hitidx = trk.see_hitIdx()[seedIdx][ihit];
                int hittype = trk.see_hitType()[seedIdx][ihit];
                int x = trk.pix_x()[hitidx];
                int y = trk.pix_y()[hitidx];
                int z = trk.pix_z()[hitidx];
                ana.tx->pushbackToBranch<float>(TString::Format("pls_hit%d_x", ihit), x);
                ana.tx->pushbackToBranch<float>(TString::Format("pls_hit%d_y", ihit), y);
                ana.tx->pushbackToBranch<float>(TString::Format("pls_hit%d_z", ihit), z);
            }
            if (trk.see_hitIdx()[seedIdx].size() == 3)
            {
                ana.tx->pushbackToBranch<float>("pls_hit3_x", -999);
                ana.tx->pushbackToBranch<float>("pls_hit3_y", -999);
                ana.tx->pushbackToBranch<float>("pls_hit3_z", -999);
            }
            bool isfake = true;
            for (size_t isim = 0; isim < simidx.size(); ++isim)
            {
                if (simidxfrac[isim] > 0.75)
                {
                    isfake = false;
                    break;
                }
            }
            ana.tx->pushbackToBranch<int>("pls_isFake", isfake);
            pls_simIdxAll.push_back(simidx);
            pls_simIdxAllFrac.push_back(simidxfrac);
            for (size_t is = 0; is < simidx.size(); ++is)
            {
                int sim_idx = simidx.at(is);
                float sim_idx_frac = simidxfrac.at(is);
                if (sim_idx < n_total_simtrk)
                {
                    sim_plsIdxAll.at(sim_idx).push_back(pls_idx);
                    sim_plsIdxAllFrac.at(sim_idx).push_back(sim_idx_frac);
                }
            }
            int pls_simIdx = -999;
            float pls_simIdxBestFrac = 0;
            for (size_t isim = 0; isim < simidx.size(); ++isim)
            {
                int thisidx = simidx[isim];
                float thisfrac = simidxfrac[isim];
                if (thisfrac > pls_simIdxBestFrac and thisfrac > 0.75)
                {
                    pls_simIdxBestFrac = thisfrac;
                    pls_simIdx = thisidx;
                }
            }
            ana.tx->pushbackToBranch<int>("pls_simIdx", pls_simIdx);
            // count global
            pls_idx++;
        }
    }
    ana.tx->setBranch<vector<vector<int>>>("pls_simIdxAll", pls_simIdxAll);
    ana.tx->setBranch<vector<vector<float>>>("pls_simIdxAllFrac", pls_simIdxAllFrac);
    std::vector<std::vector<int>> sim_plsIdxAll_to_write;
    std::vector<std::vector<float>> sim_plsIdxAllFrac_to_write;
    std::copy(sim_plsIdxAll.begin(), sim_plsIdxAll.begin() + n_accepted_simtrk, std::back_inserter(sim_plsIdxAll_to_write));
    std::copy(sim_plsIdxAllFrac.begin(), sim_plsIdxAllFrac.begin() + n_accepted_simtrk, std::back_inserter(sim_plsIdxAllFrac_to_write));
    ana.tx->setBranch<vector<vector<int>>>("sim_plsIdxAll", sim_plsIdxAll_to_write);
    ana.tx->setBranch<vector<vector<float>>>("sim_plsIdxAllFrac", sim_plsIdxAllFrac_to_write);

    // Using the intermedaite variables to compute whether a given object is a duplicate
    vector<int> pls_isDuplicate(pls_simIdxAll.size());
    // Loop over the objects
    for (unsigned int pls_idx = 0; pls_idx < pls_simIdxAll.size(); ++pls_idx)
    {
        bool isDuplicate = false;
        // Loop over the sim idx matched to this object
        for (unsigned int isim = 0; isim < pls_simIdxAll[pls_idx].size(); ++isim)
        {
            int sim_idx = pls_simIdxAll[pls_idx][isim];
            int n_sim_matched = 0;
            for (size_t ism = 0; ism < sim_plsIdxAll.at(sim_idx).size(); ++ism)
            {
                if (sim_plsIdxAllFrac.at(sim_idx).at(ism) > 0.75)
                {
                    n_sim_matched += 1;
                    if (n_sim_matched > 1)
                    {
                        isDuplicate = true;
                        break;
                    }
                }
            }
        }
        pls_isDuplicate[pls_idx] = isDuplicate;
    }
    ana.tx->setBranch<vector<int>>("pls_isDuplicate", pls_isDuplicate);







    //--------------------------------------------
    //
    //
    // pT3
    //
    //
    //--------------------------------------------

    std::vector<std::vector<int>> sim_pt3IdxAll(n_total_simtrk);
    std::vector<std::vector<float>> sim_pt3IdxAllFrac(n_total_simtrk);
    std::vector<std::vector<int>> pt3_simIdxAll;
    std::vector<std::vector<float>> pt3_simIdxAllFrac;
    // Then obtain the lower module index
    unsigned int pt3_idx = 0; // global pt3 index that will be used to keep track of pt3 being outputted to the ntuple
    // map to keep track of (GPU pt3Idx) -> (pt3_idx in ntuple output)
    std::map<unsigned int, unsigned int> pt3_idx_map;
    // printT3s(event);
    unsigned int nPixelTriplets = *pixelTripletsInGPU.nPixelTriplets;
    for (unsigned int ipT3 = 0; ipT3 < nPixelTriplets; ipT3++)
    {
        unsigned int pt3Idx = ipT3;
        pt3_idx_map[pt3Idx] = pt3_idx;
        std::vector<unsigned int> hit_idx, hit_type;
        std::tie(hit_idx, hit_type) = getHitIdxsAndHitTypesFrompT3(event, ipT3);
        std::vector<int> simidx;
        std::vector<float> simidxfrac;
        std::tie(simidx, simidxfrac) = matchedSimTrkIdxsAndFracs(hit_idx, hit_type, false, 0);
        // // Computing line segment pt estimate (assuming beam spot is at zero)
        unsigned int ipLS = getPixelLSFrompT3(event, ipT3);
        unsigned int plsIdx = rangesInGPU.segmentModuleIndices[*(modulesInGPU.nLowerModules)] + ipLS;
        unsigned int pls_idx = pls_idx_map[plsIdx];
        float pt = segmentsInGPU.ptIn[ipLS];
        float eta = segmentsInGPU.eta[ipLS];
        float phi = segmentsInGPU.phi[ipLS];
        ana.tx->pushbackToBranch<float>("pt3_pt", pt);
        ana.tx->pushbackToBranch<float>("pt3_eta", eta);
        ana.tx->pushbackToBranch<float>("pt3_phi", phi);
        ana.tx->pushbackToBranch<int>("pt3_plsIdx", pls_idx);
        unsigned int t3Idx = getT3FrompT3(event, ipT3);
        unsigned int t3_idx = t3_idx_map[t3Idx];
        ana.tx->pushbackToBranch<int>("pt3_t3Idx", t3_idx);
        bool isfake = true;
        for (size_t isim = 0; isim < simidx.size(); ++isim)
        {
            if (simidxfrac[isim] > 0.75)
            {
                isfake = false;
                break;
            }
        }
        ana.tx->pushbackToBranch<int>("pt3_isFake", isfake);
        pt3_simIdxAll.push_back(simidx);
        pt3_simIdxAllFrac.push_back(simidxfrac);
        for (size_t is = 0; is < simidx.size(); ++is)
        {
            int sim_idx = simidx.at(is);
            float sim_idx_frac = simidxfrac.at(is);
            if (sim_idx < n_total_simtrk)
            {
                sim_pt3IdxAll.at(sim_idx).push_back(pt3_idx);
                sim_pt3IdxAllFrac.at(sim_idx).push_back(sim_idx_frac);
            }
        }
        int pt3_simIdx = -999;
        float pt3_simIdxBestFrac = 0;
        for (size_t isim = 0; isim < simidx.size(); ++isim)
        {
            int thisidx = simidx[isim];
            float thisfrac = simidxfrac[isim];
            if (thisfrac > pt3_simIdxBestFrac and thisfrac > 0.75)
            {
                pt3_simIdxBestFrac = thisfrac;
                pt3_simIdx = thisidx;
            }
        }
        ana.tx->pushbackToBranch<int>("pt3_simIdx", pt3_simIdx);
        // count global
        pt3_idx++;
    }
    ana.tx->setBranch<vector<vector<int>>>("pt3_simIdxAll", pt3_simIdxAll);
    ana.tx->setBranch<vector<vector<float>>>("pt3_simIdxAllFrac", pt3_simIdxAllFrac);
    std::vector<std::vector<int>> sim_pt3IdxAll_to_write;
    std::vector<std::vector<float>> sim_pt3IdxAllFrac_to_write;
    std::copy(sim_pt3IdxAll.begin(), sim_pt3IdxAll.begin() + n_accepted_simtrk, std::back_inserter(sim_pt3IdxAll_to_write));
    std::copy(sim_pt3IdxAllFrac.begin(), sim_pt3IdxAllFrac.begin() + n_accepted_simtrk, std::back_inserter(sim_pt3IdxAllFrac_to_write));
    ana.tx->setBranch<vector<vector<int>>>("sim_pt3IdxAll", sim_pt3IdxAll_to_write);
    ana.tx->setBranch<vector<vector<float>>>("sim_pt3IdxAllFrac", sim_pt3IdxAllFrac_to_write);

    // Using the intermedaite variables to compute whether a given object is a duplicate
    vector<int> pt3_isDuplicate(pt3_simIdxAll.size());
    // Loop over the objects
    for (unsigned int pt3_idx = 0; pt3_idx < pt3_simIdxAll.size(); ++pt3_idx)
    {
        bool isDuplicate = false;
        // Loop over the sim idx matched to this object
        for (unsigned int isim = 0; isim < pt3_simIdxAll[pt3_idx].size(); ++isim)
        {
            int sim_idx = pt3_simIdxAll[pt3_idx][isim];
            int n_sim_matched = 0;
            for (size_t ism = 0; ism < sim_pt3IdxAll.at(sim_idx).size(); ++ism)
            {
                if (sim_pt3IdxAllFrac.at(sim_idx).at(ism) > 0.75)
                {
                    n_sim_matched += 1;
                    if (n_sim_matched > 1)
                    {
                        isDuplicate = true;
                        break;
                    }
                }
            }
        }
        pt3_isDuplicate[pt3_idx] = isDuplicate;
    }
    ana.tx->setBranch<vector<int>>("pt3_isDuplicate", pt3_isDuplicate);






    //--------------------------------------------
    //
    //
    // pT5
    //
    //
    //--------------------------------------------

    std::vector<std::vector<int>> sim_pt5IdxAll(n_total_simtrk);
    std::vector<std::vector<float>> sim_pt5IdxAllFrac(n_total_simtrk);
    std::vector<std::vector<int>> pt5_simIdxAll;
    std::vector<std::vector<float>> pt5_simIdxAllFrac;
    // Then obtain the lower module index
    unsigned int pt5_idx = 0; // global pt5 index that will be used to keep track of pt5 being outputted to the ntuple
    // map to keep track of (GPU pt5Idx) -> (pt5_idx in ntuple output)
    std::map<unsigned int, unsigned int> pt5_idx_map;
    // printT5s(event);
    unsigned int nPixelQuintuplets = *pixelQuintupletsInGPU.nPixelQuintuplets;
    for (unsigned int ipT5 = 0; ipT5 < nPixelQuintuplets; ipT5++)
    {
        unsigned int pt5Idx = ipT5;
        pt5_idx_map[pt5Idx] = pt5_idx;
        std::vector<unsigned int> hit_idx, hit_type;
        std::tie(hit_idx, hit_type) = getHitIdxsAndHitTypesFrompT5(event, ipT5);
        std::vector<int> simidx;
        std::vector<float> simidxfrac;
        std::tie(simidx, simidxfrac) = matchedSimTrkIdxsAndFracs(hit_idx, hit_type, false, 0);
        // // Computing line segment pt estimate (assuming beam spot is at zero)
        unsigned int ipLS = getPixelLSFrompT5(event, ipT5);
        unsigned int plsIdx = rangesInGPU.segmentModuleIndices[*(modulesInGPU.nLowerModules)] + ipLS;
        unsigned int pls_idx = pls_idx_map[plsIdx];
        float pt = segmentsInGPU.ptIn[ipLS];
        float eta = segmentsInGPU.eta[ipLS];
        float phi = segmentsInGPU.phi[ipLS];
        ana.tx->pushbackToBranch<float>("pt5_pt", pt);
        ana.tx->pushbackToBranch<float>("pt5_eta", eta);
        ana.tx->pushbackToBranch<float>("pt5_phi", phi);
        ana.tx->pushbackToBranch<int>("pt5_plsIdx", pls_idx);
        unsigned int t5Idx = getT5FrompT5(event, ipT5);
        unsigned int t5_idx = t5_idx_map[t5Idx];
        ana.tx->pushbackToBranch<int>("pt5_t5Idx", t5_idx);
        bool isfake = true;
        for (size_t isim = 0; isim < simidx.size(); ++isim)
        {
            if (simidxfrac[isim] > 0.75)
            {
                isfake = false;
                break;
            }
        }
        ana.tx->pushbackToBranch<int>("pt5_isFake", isfake);
        pt5_simIdxAll.push_back(simidx);
        pt5_simIdxAllFrac.push_back(simidxfrac);
        for (size_t is = 0; is < simidx.size(); ++is)
        {
            int sim_idx = simidx.at(is);
            float sim_idx_frac = simidxfrac.at(is);
            if (sim_idx < n_total_simtrk)
            {
                sim_pt5IdxAll.at(sim_idx).push_back(pt5_idx);
                sim_pt5IdxAllFrac.at(sim_idx).push_back(sim_idx_frac);
            }
        }
        int pt5_simIdx = -999;
        float pt5_simIdxBestFrac = 0;
        for (size_t isim = 0; isim < simidx.size(); ++isim)
        {
            int thisidx = simidx[isim];
            float thisfrac = simidxfrac[isim];
            if (thisfrac > pt5_simIdxBestFrac and thisfrac > 0.75)
            {
                pt5_simIdxBestFrac = thisfrac;
                pt5_simIdx = thisidx;
            }
        }
        ana.tx->pushbackToBranch<int>("pt5_simIdx", pt5_simIdx);
        // count global
        pt5_idx++;
    }
    ana.tx->setBranch<vector<vector<int>>>("pt5_simIdxAll", pt5_simIdxAll);
    ana.tx->setBranch<vector<vector<float>>>("pt5_simIdxAllFrac", pt5_simIdxAllFrac);
    std::vector<std::vector<int>> sim_pt5IdxAll_to_write;
    std::vector<std::vector<float>> sim_pt5IdxAllFrac_to_write;
    std::copy(sim_pt5IdxAll.begin(), sim_pt5IdxAll.begin() + n_accepted_simtrk, std::back_inserter(sim_pt5IdxAll_to_write));
    std::copy(sim_pt5IdxAllFrac.begin(), sim_pt5IdxAllFrac.begin() + n_accepted_simtrk, std::back_inserter(sim_pt5IdxAllFrac_to_write));
    ana.tx->setBranch<vector<vector<int>>>("sim_pt5IdxAll", sim_pt5IdxAll_to_write);
    ana.tx->setBranch<vector<vector<float>>>("sim_pt5IdxAllFrac", sim_pt5IdxAllFrac_to_write);

    // Using the intermedaite variables to compute whether a given object is a duplicate
    vector<int> pt5_isDuplicate(pt5_simIdxAll.size());
    // Loop over the objects
    for (unsigned int pt5_idx = 0; pt5_idx < pt5_simIdxAll.size(); ++pt5_idx)
    {
        bool isDuplicate = false;
        // Loop over the sim idx matched to this object
        for (unsigned int isim = 0; isim < pt5_simIdxAll[pt5_idx].size(); ++isim)
        {
            int sim_idx = pt5_simIdxAll[pt5_idx][isim];
            int n_sim_matched = 0;
            for (size_t ism = 0; ism < sim_pt5IdxAll.at(sim_idx).size(); ++ism)
            {
                if (sim_pt5IdxAllFrac.at(sim_idx).at(ism) > 0.75)
                {
                    n_sim_matched += 1;
                    if (n_sim_matched > 1)
                    {
                        isDuplicate = true;
                        break;
                    }
                }
            }
        }
        pt5_isDuplicate[pt5_idx] = isDuplicate;
    }
    ana.tx->setBranch<vector<int>>("pt5_isDuplicate", pt5_isDuplicate);







    //--------------------------------------------
    //
    //
    // Track Candidates
    //
    //
    //--------------------------------------------

    // Following are some vectors to keep track of the information to write to the ntuple
    // N.B. following two branches have a length for the entire sim track, but what actually will be written in sim_tcIdxAll branch is NOT that long
    // Later in the code, it will restrict to only the ones to write out.
    // The reason at this stage, the entire tcIdxAll is being tracked is to compute duplicate properly later on
    // When computing a duplicate object it is important to consider all simulated tracks including pileup tracks
    std::vector<std::vector<int>> sim_tcIdxAll(n_total_simtrk);
    std::vector<std::vector<float>> sim_tcIdxAllFrac(n_total_simtrk);
    std::vector<std::vector<int>> tc_simIdxAll;
    std::vector<std::vector<float>> tc_simIdxAllFrac;

    // Number of total track candidates created in this event
    unsigned int nTrackCandidates = *trackCandidatesInGPU.nTrackCandidates;

    // Looping over each track candidate
    for (unsigned int tc_idx = 0; tc_idx < nTrackCandidates; tc_idx++)
    {
        // Compute reco quantities of track candidate based on final object
        int type, isFake;
        float pt, eta, phi;
        std::vector<int> simidx; // list of all the matched sim idx
        std::vector<float> simidxfrac; // list of match fraction for each matched sim idx

        // The following function reads off and computes the matched sim track indices
        std::tie(type, pt, eta, phi, isFake, simidx, simidxfrac) = parseTrackCandidateAllMatch(event, tc_idx);

        // Fill some branches for this track candidate
        ana.tx->pushbackToBranch<float>("tc_pt", pt);
        ana.tx->pushbackToBranch<float>("tc_eta", eta);
        ana.tx->pushbackToBranch<float>("tc_phi", phi);
        ana.tx->pushbackToBranch<int>("tc_type", type);
        enum
        {
            pT5 = 7,
            pT3 = 5,
            T5 = 4,
            pLS = 8
        };
        if (type == pT5)
        {
            ana.tx->pushbackToBranch<int>("tc_pt5Idx", pt5_idx_map[trackCandidatesInGPU.directObjectIndices[tc_idx]]);
            ana.tx->pushbackToBranch<int>("tc_pt3Idx", -999);
            ana.tx->pushbackToBranch<int>("tc_t5Idx" , -999);
            ana.tx->pushbackToBranch<int>("tc_plsIdx", -999);
        }
        else if (type == pT3)
        {
            ana.tx->pushbackToBranch<int>("tc_pt5Idx", -999);
            ana.tx->pushbackToBranch<int>("tc_pt3Idx", pt3_idx_map[trackCandidatesInGPU.directObjectIndices[tc_idx]]);
            ana.tx->pushbackToBranch<int>("tc_t5Idx" , -999);
            ana.tx->pushbackToBranch<int>("tc_plsIdx", -999);
        }
        else if (type == T5 )
        {
            ana.tx->pushbackToBranch<int>("tc_pt5Idx", -999);
            ana.tx->pushbackToBranch<int>("tc_pt3Idx", -999);
            ana.tx->pushbackToBranch<int>("tc_t5Idx" , t5_idx_map[trackCandidatesInGPU.directObjectIndices[tc_idx]]);
            ana.tx->pushbackToBranch<int>("tc_plsIdx", -999);
        }
        else if (type == pLS)
        {
            ana.tx->pushbackToBranch<int>("tc_pt5Idx", -999);
            ana.tx->pushbackToBranch<int>("tc_pt3Idx", -999);
            ana.tx->pushbackToBranch<int>("tc_t5Idx" , -999);
            ana.tx->pushbackToBranch<int>("tc_plsIdx", pls_idx_map[rangesInGPU.segmentModuleIndices[*(modulesInGPU.nLowerModules)] + trackCandidatesInGPU.directObjectIndices[tc_idx]]);
        }

        ana.tx->pushbackToBranch<int>("tc_isFake", isFake);

        // For this tc, keep track of all the simidx that are matched
        tc_simIdxAll.push_back(simidx);
        tc_simIdxAllFrac.push_back(simidxfrac);

        // The book keeping of opposite mapping is done here
        // For each matched sim idx, we go back and keep track of which tc it is matched to.
        // Loop over all the matched sim idx
        for (size_t is = 0; is < simidx.size(); ++is)
        {
            // For this matched sim index keep track (sim -> tc) mapping
            int sim_idx = simidx.at(is);
            float sim_idx_frac = simidxfrac.at(is);
            sim_tcIdxAll.at(sim_idx).push_back(tc_idx);
            sim_tcIdxAllFrac.at(sim_idx).push_back(sim_idx_frac);
        }

        // Also, among the simidx matches, find the best match (highest fractional match)
        // N.B. the simidx is already returned sorted by highest number of "nhits" match
        // So as it loops over, the condition will ensure that the highest fraction with highest nhits will be matched with the priority given to highest fraction
        int tc_simIdx = -999;
        float tc_simIdxBestFrac = 0;
        for (size_t isim = 0; isim < simidx.size(); ++isim)
        {
            int thisidx = simidx[isim];
            float thisfrac = simidxfrac[isim];
            if (thisfrac > tc_simIdxBestFrac and thisfrac > 0.75)
            {
                tc_simIdxBestFrac = thisfrac;
                tc_simIdx = thisidx;
            }
        }

        // the best match index will then be saved here
        ana.tx->pushbackToBranch<int>("tc_simIdx", tc_simIdx);
    }

    // Now save the (tc -> simidx) mapping
    ana.tx->setBranch<vector<vector<int>>>("tc_simIdxAll", tc_simIdxAll);
    ana.tx->setBranch<vector<vector<float>>>("tc_simIdxAllFrac", tc_simIdxAllFrac);

    // Not all (sim->tcIdx) will be saved but only for the sim that is from hard scatter and current bunch crossing
    // So a restriction up to only "n_accepted_simtrk" done by chopping off the rest
    // N.B. the reason we can simply take the first "n_accepted_simtrk" is because the tracking ntuple is organized such that those sim tracks show up on the first "n_accepted_simtrk" of tracks.
    std::vector<std::vector<int>> sim_tcIdxAll_to_write;
    std::vector<std::vector<float>> sim_tcIdxAllFrac_to_write;
    std::copy(sim_tcIdxAll.begin(), sim_tcIdxAll.begin() + n_accepted_simtrk, std::back_inserter(sim_tcIdxAll_to_write)); // this is where the vector is only copying the first "n_accepted_simtrk"
    std::copy(sim_tcIdxAllFrac.begin(), sim_tcIdxAllFrac.begin() + n_accepted_simtrk, std::back_inserter(sim_tcIdxAllFrac_to_write)); // ditto
    ana.tx->setBranch<vector<vector<int>>>("sim_tcIdxAll", sim_tcIdxAll_to_write);
    ana.tx->setBranch<vector<vector<float>>>("sim_tcIdxAllFrac", sim_tcIdxAllFrac_to_write);


    // Using the intermedaite variables to compute whether a given track candidate is a duplicate
    vector<int> tc_isDuplicate(tc_simIdxAll.size());

    // Loop over the track candidates
    for (unsigned int tc_idx = 0; tc_idx < tc_simIdxAll.size(); ++tc_idx)
    {
        bool isDuplicate = false;
        // Loop over the sim idx matched to this track candidate
        for (unsigned int isim = 0; isim < tc_simIdxAll[tc_idx].size(); ++isim)
        {
            int sim_idx = tc_simIdxAll[tc_idx][isim];
            int n_sim_matched = 0;
            for (size_t ism = 0; ism < sim_tcIdxAll.at(sim_idx).size(); ++ism)
            {
                if (sim_tcIdxAllFrac.at(sim_idx).at(ism) > 0.75)
                {
                    n_sim_matched += 1;
                    if (n_sim_matched > 1)
                    {
                        isDuplicate = true;
                        break;
                    }
                }
            }
        }
        tc_isDuplicate[tc_idx] = isDuplicate;
    }
    ana.tx->setBranch<vector<int>>("tc_isDuplicate", tc_isDuplicate);

    // Similarly, the best match for the (sim -> tc is computed)
    // TODO: Is this redundant? I am not sure if it is guaranteed that sim_tcIdx will have same result with tc_simIdx.
    // I think it will be, but I have not rigorously checked. I only checked about first few thousands and it was all true. as long as tc->sim was pointing to a sim that is among the n_accepted.
    // For the most part I think this won't be a problem.
    for (size_t i = 0; i < sim_tcIdxAll_to_write.size(); ++ i)
    {
        // bestmatch is not always the first one
        int bestmatch_idx = -999;
        float bestmatch_frac = -999;
        for (size_t jj = 0; jj < sim_tcIdxAll_to_write.at(i).size(); ++jj)
        {
            int idx = sim_tcIdxAll_to_write.at(i).at(jj);
            float frac = sim_tcIdxAllFrac_to_write.at(i).at(jj);
            if (bestmatch_frac < frac)
            {
                bestmatch_idx = idx;
                bestmatch_frac = frac;
            }
        }
        ana.tx->pushbackToBranch<int>("sim_tcIdxBest", bestmatch_idx);
        ana.tx->pushbackToBranch<float>("sim_tcIdxBestFrac", bestmatch_frac);
        if (bestmatch_frac > 0.75) // then this is a good match according to MTV
            ana.tx->pushbackToBranch<int>("sim_tcIdx", bestmatch_idx);
        else
            ana.tx->pushbackToBranch<int>("sim_tcIdx", -999);
    }





    //--------------------------------------------
    //
    //
    // Write the output
    //
    //
    //--------------------------------------------

    // Now actually fill the ttree
    ana.tx->fill();

    // Then clear the branches to default values (e.g. -999, or clear the vectors to empty vectors)
    ana.tx->clear();

}