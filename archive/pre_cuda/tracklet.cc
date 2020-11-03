#include "tracklet.h"

void tracklet()
{
    // List of studies to perform
    std::vector<Study*> studies;

    // studies.push_back(new StudyTrackletSelection("studySelTlBB1BB3", StudyTrackletSelection::kStudySelBB1BB3));
    // studies.push_back(new StudyTrackletSelection("studySelTlBB1BE3", StudyTrackletSelection::kStudySelBB1BE3));
    // studies.push_back(new StudyTrackletSelection("studySelTlBB1EE3", StudyTrackletSelection::kStudySelBB1EE3));
    // studies.push_back(new StudyTrackletSelection("studySelTlBE1EE3", StudyTrackletSelection::kStudySelBE1EE3));
    // studies.push_back(new StudyTrackletSelection("studySelTlEE1EE3", StudyTrackletSelection::kStudySelEE1EE3));

    // studies.push_back(new StudyTrackletSelection("studySelTlBB2BB4", StudyTrackletSelection::kStudySelBB2BB4));
    // studies.push_back(new StudyTrackletSelection("studySelTlBB2BE4", StudyTrackletSelection::kStudySelBB2BE4));
    // studies.push_back(new StudyTrackletSelection("studySelTlBB2EE4", StudyTrackletSelection::kStudySelBB2EE4));
    // studies.push_back(new StudyTrackletSelection("studySelTlBE2EE4", StudyTrackletSelection::kStudySelBE2EE4));
    // studies.push_back(new StudyTrackletSelection("studySelTlEE2EE4", StudyTrackletSelection::kStudySelEE2EE4));

    // studies.push_back(new StudyTrackletSelection("studySelTlBB3BB5", StudyTrackletSelection::kStudySelBB3BB5));
    // studies.push_back(new StudyTrackletSelection("studySelTlBB3BE5", StudyTrackletSelection::kStudySelBB3BE5));
    // studies.push_back(new StudyTrackletSelection("studySelTlBB3EE5", StudyTrackletSelection::kStudySelBB3EE5));
    // studies.push_back(new StudyTrackletSelection("studySelTlBE3EE5", StudyTrackletSelection::kStudySelBE3EE5));

    // // EE3EE5 is impossible

    // // studies.push_back(new StudyTrackletSelection("studySelTlEE1EE3AllPS", StudyTrackletSelection::kStudySelEE1EE3AllPS));
    // // studies.push_back(new StudyTrackletSelection("studySelTlEE1EE3All2S", StudyTrackletSelection::kStudySelEE1EE3All2S));

    ana.tx->createBranch<int>("event");
    ana.tx->createBranch<int>("category");
    ana.tx->createBranch<int>("nPS");
    ana.tx->createBranch<float>("deltaBetaDefault");
    ana.tx->createBranch<float>("betaInDefault");
    ana.tx->createBranch<float>("betaOutDefault");
    ana.tx->createBranch<int>("betacormode");
    ana.tx->createBranch<int>("getPassBitsDefaultAlgo");
    ana.tx->createBranch<int>("passDeltaAlphaOut");
    ana.tx->createBranch<vector<float>>("betaIn");
    ana.tx->createBranch<vector<float>>("betaOut");
    ana.tx->createBranch<vector<float>>("betaAv");
    ana.tx->createBranch<vector<float>>("betaPt");
    ana.tx->createBranch<vector<float>>("dBeta");
    ana.tx->createBranch<float>("dBetaCut2");
    ana.tx->createBranch<float>("matched_trk_pt");
    ana.tx->createBranch<int>("matched_trk_charge");
    ana.tx->createBranch<float>("matched_trk_eta");
    ana.tx->createBranch<float>("matched_trk_phi");
    ana.tx->createBranch<float>("matched_trk_pca_pt");
    ana.tx->createBranch<float>("matched_trk_pca_dz");
    ana.tx->createBranch<float>("matched_trk_pca_dxy");
    ana.tx->createBranch<float>("matched_trk_pca_phi");
    ana.tx->createBranch<float>("matched_trk_pca_eta");
    ana.tx->createBranch<float>("matched_trk_pca_cotTheta");
    ana.tx->createBranch<float>("matched_trk_pca_lambda");
    ana.tx->createBranch<float>("matched_trk_px");
    ana.tx->createBranch<float>("matched_trk_py");
    ana.tx->createBranch<float>("matched_trk_pz");
    ana.tx->createBranch<float>("simvtx_x");
    ana.tx->createBranch<float>("simvtx_y");
    ana.tx->createBranch<float>("simvtx_z");
    ana.tx->createBranch<vector<int>>("module_detId");
    ana.tx->createBranch<vector<int>>("module_subdet");
    ana.tx->createBranch<vector<int>>("module_layer");
    ana.tx->createBranch<vector<int>>("module_ring");
    ana.tx->createBranch<vector<int>>("module_module");
    ana.tx->createBranch<vector<int>>("module_rod");
    ana.tx->createBranch<vector<int>>("module_isPS");
    ana.tx->createBranch<vector<int>>("azimuthal_direction_module_idx");
    ana.tx->createBranch<vector<int>>("transverse_direction_module_idx");
    ana.tx->createBranch<vector<float>>("x");
    ana.tx->createBranch<vector<float>>("y");
    ana.tx->createBranch<vector<float>>("z");
    ana.tx->createBranch<vector<float>>("true_x");
    ana.tx->createBranch<vector<float>>("true_y");
    ana.tx->createBranch<vector<float>>("true_z");

    // book the studies
    for (auto& study : studies)
    {
        study->bookStudy();
    }

    // Book Histograms
    ana.cutflow.bookHistograms(ana.histograms); // if just want to book everywhere

    // Load various maps used in the SDL reconstruction
    loadMaps();

    // Looping input file
    while (ana.looper.nextEvent())
    {

        if (not goodEvent())
            continue;

        // Main instance that will hold modules, hits, minidoublets, etc. (i.e. main data structure)
        SDL::Event event;

        // Add hits to the event
        addOuterTrackerHits(event);

        printHitSummary(event);
        runMiniDoublet(event);
        printMiniDoubletSummary(event);
        runSegment(event);
        printSegmentSummary(event);

        TStopwatch my_timer;
        if (ana.verbose != 0) std::cout << "Reco Tracklet start" << std::endl;
        my_timer.Start(kFALSE);
        // event.createTrackletsWithModuleMap();
        // event.createTrackletsViaNavigation();
        event.createTrackletsWithModuleMap(SDL::DefaultNm1_TLAlgo);
        // event.createTrackletsViaNavigation(SDL::DefaultNm1_TLAlgo);
        float tl_elapsed = my_timer.RealTime();
        if (ana.verbose != 0) std::cout << "Reco Tracklet processing time: " << tl_elapsed << " secs" << std::endl;

        // std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents;

        // for (auto& study : studies)
        // {
        //     study->doStudy(event, simtrkevents);
        // }


        // Loop over tracklets in event
        for (auto& layerPtr : event.getLayerPtrs())
        {

            // Parse the layer index later to be used for indexing
            int layer_idx = layerPtr->layerIdx() - 1;

            for (auto& tl : layerPtr->getTrackletPtrs())
            {
                // ana.tx->setBranch<int>("event", event_idx);
                ana.tx->setBranch<int>("category", getTrackletCategory(*tl));
                ana.tx->setBranch<int>("nPS", getNPSModules(*tl));
                ana.tx->setBranch<float>("deltaBetaDefault", tl->getDeltaBeta());
                ana.tx->setBranch<float>("betaInDefault", tl->getRecoVar("betaIn"));
                ana.tx->setBranch<float>("betaOutDefault", tl->getRecoVar("betaOut"));
                ana.tx->setBranch<int>("betacormode", tl->getRecoVar("betacormode"));
                ana.tx->setBranch<int>("getPassBitsDefaultAlgo", tl->getPassBitsDefaultAlgo());
                ana.tx->setBranch<int>("passDeltaAlphaOut", tl->getPassBitsDefaultAlgo() & (1 << SDL::Tracklet::TrackletSelection::dBeta));
                ana.tx->setBranch<float>("dBetaCut2", tl->getRecoVar("dBetaCut2"));
                ana.tx->pushbackToBranch<float>("betaIn", tl->getRecoVar("betaIn_0th"));
                ana.tx->pushbackToBranch<float>("betaOut", tl->getRecoVar("betaOut_0th"));
                ana.tx->pushbackToBranch<float>("betaAv", tl->getRecoVar("betaAv_0th"));
                ana.tx->pushbackToBranch<float>("betaPt", tl->getRecoVar("betaPt_0th"));
                ana.tx->pushbackToBranch<float>("dBeta", tl->getRecoVar("dBeta_0th"));
                ana.tx->pushbackToBranch<float>("betaIn", tl->getRecoVar("betaIn_1st"));
                ana.tx->pushbackToBranch<float>("betaOut", tl->getRecoVar("betaOut_1st"));
                ana.tx->pushbackToBranch<float>("betaAv", tl->getRecoVar("betaAv_1st"));
                ana.tx->pushbackToBranch<float>("betaPt", tl->getRecoVar("betaPt_1st"));
                ana.tx->pushbackToBranch<float>("dBeta", tl->getRecoVar("dBeta_1st"));
                ana.tx->pushbackToBranch<float>("betaIn", tl->getRecoVar("betaIn_2nd"));
                ana.tx->pushbackToBranch<float>("betaOut", tl->getRecoVar("betaOut_2nd"));
                ana.tx->pushbackToBranch<float>("betaAv", tl->getRecoVar("betaAv_2nd"));
                ana.tx->pushbackToBranch<float>("betaPt", tl->getRecoVar("betaPt_2nd"));
                ana.tx->pushbackToBranch<float>("dBeta", tl->getRecoVar("dBeta_2nd"));
                ana.tx->pushbackToBranch<float>("betaIn", tl->getRecoVar("betaIn_3rd"));
                ana.tx->pushbackToBranch<float>("betaOut", tl->getRecoVar("betaOut_3rd"));
                ana.tx->pushbackToBranch<float>("betaAv", tl->getRecoVar("betaAv_3rd"));
                ana.tx->pushbackToBranch<float>("betaPt", tl->getRecoVar("betaPt_3rd"));
                ana.tx->pushbackToBranch<float>("dBeta", tl->getRecoVar("dBeta_3rd"));
                ana.tx->pushbackToBranch<float>("betaIn", tl->getRecoVar("betaIn_4th"));
                ana.tx->pushbackToBranch<float>("betaOut", tl->getRecoVar("betaOut_4th"));
                ana.tx->pushbackToBranch<float>("betaAv", tl->getRecoVar("betaAv_4th"));
                ana.tx->pushbackToBranch<float>("betaPt", tl->getRecoVar("betaPt_4th"));
                ana.tx->pushbackToBranch<float>("dBeta", tl->getRecoVar("dBeta_4th"));
                std::vector<int> matched_sim_trk_idxs = matchedSimTrkIdxs(*tl);
                if (matched_sim_trk_idxs.size() > 0)
                {
                    ana.tx->setBranch<float>("matched_trk_pt", trk.sim_pt()[matched_sim_trk_idxs[0]]);
                    ana.tx->setBranch<int>("matched_trk_charge", trk.sim_q()[matched_sim_trk_idxs[0]]);
                    ana.tx->setBranch<float>("matched_trk_eta", trk.sim_eta()[matched_sim_trk_idxs[0]]);
                    ana.tx->setBranch<float>("matched_trk_phi", trk.sim_phi()[matched_sim_trk_idxs[0]]);
                    ana.tx->setBranch<float>("matched_trk_pca_pt", trk.sim_pca_pt()[matched_sim_trk_idxs[0]]);
                    ana.tx->setBranch<float>("matched_trk_pca_dz", trk.sim_pca_dz()[matched_sim_trk_idxs[0]]);
                    ana.tx->setBranch<float>("matched_trk_pca_dxy", trk.sim_pca_dxy()[matched_sim_trk_idxs[0]]);
                    ana.tx->setBranch<float>("matched_trk_pca_phi", trk.sim_pca_phi()[matched_sim_trk_idxs[0]]);
                    ana.tx->setBranch<float>("matched_trk_pca_eta", trk.sim_pca_eta()[matched_sim_trk_idxs[0]]);
                    ana.tx->setBranch<float>("matched_trk_pca_cotTheta", trk.sim_pca_cotTheta()[matched_sim_trk_idxs[0]]);
                    ana.tx->setBranch<float>("matched_trk_pca_lambda", trk.sim_pca_lambda()[matched_sim_trk_idxs[0]]);
                    ana.tx->setBranch<float>("matched_trk_px", trk.sim_px()[matched_sim_trk_idxs[0]]);
                    ana.tx->setBranch<float>("matched_trk_py", trk.sim_py()[matched_sim_trk_idxs[0]]);
                    ana.tx->setBranch<float>("matched_trk_pz", trk.sim_pz()[matched_sim_trk_idxs[0]]);
                    ana.tx->setBranch<float>("simvtx_x", trk.simvtx_x()[trk.sim_parentVtxIdx()[matched_sim_trk_idxs[0]]]);
                    ana.tx->setBranch<float>("simvtx_y", trk.simvtx_y()[trk.sim_parentVtxIdx()[matched_sim_trk_idxs[0]]]);
                    ana.tx->setBranch<float>("simvtx_z", trk.simvtx_z()[trk.sim_parentVtxIdx()[matched_sim_trk_idxs[0]]]);
                }

                vector<const SDL::Module*> modules = {
                    &(tl->innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()),
                    &(tl->outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule()),
                    &(tl->innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule()),
                    &(tl->outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule())};

                for (auto& modptr : modules)
                {
                    ana.tx->pushbackToBranch<int>("module_detId" , modptr->detId() ) ;
                    ana.tx->pushbackToBranch<int>("module_subdet" , modptr->subdet() ) ;
                    ana.tx->pushbackToBranch<int>("module_layer" , modptr->layer() ) ;
                    ana.tx->pushbackToBranch<int>("module_ring" , modptr->ring() ) ;
                    ana.tx->pushbackToBranch<int>("module_module" , modptr->module() ) ;
                    ana.tx->pushbackToBranch<int>("module_rod" , modptr->rod() ) ;
                    ana.tx->pushbackToBranch<int>("module_isPS" , modptr->moduleType() == SDL::Module::PS ) ;
                    ana.tx->pushbackToBranch<int>("azimuthal_direction_module_idx", modptr->subdet() == 5 ? modptr->rod() : modptr->module());
                    ana.tx->pushbackToBranch<int>("transverse_direction_module_idx", modptr->subdet() == 5 ? modptr->layer() : modptr->ring());
                }

                vector<const SDL::Hit*> hits = {
                    tl->innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr(),
                    tl->outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr(),
                    tl->innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr(),
                    tl->outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()};

                for (auto& hitptr : hits)
                {
                    ana.tx->pushbackToBranch<float>("x", hitptr->x());
                    ana.tx->pushbackToBranch<float>("y", hitptr->y());
                    ana.tx->pushbackToBranch<float>("z", hitptr->z());

                    int hitidx = hitptr->idx();

                    if (trk.ph2_simHitIdx()[hitidx].size() > 0)
                    {

                        ana.tx->pushbackToBranch<float>("true_x", trk.simhit_x()[trk.ph2_simHitIdx()[hitidx][0]]);
                        ana.tx->pushbackToBranch<float>("true_y", trk.simhit_y()[trk.ph2_simHitIdx()[hitidx][0]]);
                        ana.tx->pushbackToBranch<float>("true_z", trk.simhit_z()[trk.ph2_simHitIdx()[hitidx][0]]);

                    }
                }

                ana.tx->fill();
                ana.tx->clear();
            }

        }


        // ************************************************
        // Now fill all the histograms booked by each study
        // ************************************************

        // Fill all the histograms
        ana.cutflow.fill();

        // <--------------------------
        // <--------------------------
        // <--------------------------
    }

    // Writing output file
    ana.cutflow.saveOutput();

    // Writing ttree output to file
    ana.output_ttree->Write();

    // The below can be sometimes crucial
    delete ana.output_tfile;

}
