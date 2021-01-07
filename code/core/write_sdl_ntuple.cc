#include "write_sdl_ntuple.h"

void write_sdl_ntuple()
{

    // Load various maps used in the SDL reconstruction
    loadMaps();

    if (not ana.do_run_cpu)
        SDL::initModules();

    createOutputBranches();

    // Timing average information
    std::vector<std::vector<float>> timing_information;

    // Looping input file
    while (ana.looper.nextEvent())
    {
        std::cout<<"event number = "<<ana.looper.getCurrentEventIndex()<<std::endl;

        if (not goodEvent())
            continue;

        if (not ana.do_run_cpu)
        {
            // Main instance that will hold modules, hits, minidoublets, etc. (i.e. main data structure)
            SDL::Event event;

            // Add hits to the event
            float timing_input_loading = addInputsToLineSegmentTrackingUsingUnifiedMemory(event);
            printHitMultiplicities(event);

            // Run Mini-doublet
            float timing_MD = runMiniDoublet(event);
            printMiniDoubletMultiplicities(event);

            // Run Segment
            float timing_LS = runSegment(event);

            // Run pT4
            float timing_pT4 = runpT4(event);
            //printQuadrupletMultiplicities(event);

            // Run T4x
            float timing_T4x = runT4x(event);
            //printQuadrupletMultiplicities(event);

            // Run T4
            float timing_T4 = runT4(event);
            printQuadrupletMultiplicities(event);

            // Run T3
            float timing_T3 = runT3(event);

            // Run TC
            float timing_TC = runTrackCandidate(event);

            timing_information.push_back({ timing_input_loading,
                    timing_MD,
                    timing_LS,
                    timing_T4,
                    timing_T4x,
                    timing_pT4,
                    timing_T3,
                    timing_TC});

            fillOutputBranches(event);
        }
        else
        {
            // Main instance that will hold modules, hits, minidoublets, etc. (i.e. main data structure)
            SDL::CPU::Event event;

            // event.setLogLevel(SDL::CPU::Log_Debug3);

            // Add hits to the event
            float timing_input_loading = addOuterTrackerHits(event);

            // Add pixel segments
            timing_input_loading += addPixelSegments(event);

            // Print hit summary
            printHitSummary(event);

            // Run Mini-doublet
            float timing_MD = runMiniDoublet_on_CPU(event);
            printMiniDoubletSummary(event);

            // Run Segment
            float timing_LS = runSegment_on_CPU(event);
            printSegmentSummary(event);

            // Run Tracklet
            float timing_T4 = runT4_on_CPU(event);
            printTrackletSummary(event);
            float timing_T4x = runT4x_on_CPU(event);
            printTrackletSummary(event);
            float timing_pT4 = runpT4_on_CPU(event);
            printTrackletSummary(event);

            // Run Triplet
            float timing_T3 = runT3_on_CPU(event);
            printTripletSummary(event);

            // Run TrackCandidate
            float timing_TC = runTrackCandidate_on_CPU(event);
            printTrackCandidateSummary(event);

            timing_information.push_back({ timing_input_loading,
                    timing_MD,
                    timing_LS,
                    timing_T4,
                    timing_T4x,
                    timing_pT4,
                    timing_T3,
                    timing_TC});

            fillOutputBranches_for_CPU(event);

        }

    }

    printTimingInformation(timing_information);

    // Writing ttree output to file
    ana.output_tfile->cd();
    ana.output_ttree->Write();

    // The below can be sometimes crucial
    delete ana.output_tfile;

}

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

    // Sim vertex
    ana.tx->createBranch<vector<float>>("simvtx_x");
    ana.tx->createBranch<vector<float>>("simvtx_y");
    ana.tx->createBranch<vector<float>>("simvtx_z");

    ana.tx->createBranch<vector<vector<int>>>("sim_tcIdx");

    // Matched to track candidate
    ana.tx->createBranch<vector<int>>("sim_TC_matched");

    ana.tx->createBranch<vector<vector<int>>>("sim_TC_types");
}

void fillOutputBranches(SDL::Event& event)
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

    // simvtx
    ana.tx->setBranch<vector<float>>("simvtx_x", trk.simvtx_x());
    ana.tx->setBranch<vector<float>>("simvtx_y", trk.simvtx_y());
    ana.tx->setBranch<vector<float>>("simvtx_z", trk.simvtx_z());

    SDL::trackCandidates& trackCandidatesInGPU = (*event.getTrackCandidates());
    SDL::tracklets& trackletsInGPU = (*event.getTracklets());
    SDL::triplets& tripletsInGPU = (*event.getTriplets());
    SDL::segments& segmentsInGPU = (*event.getSegments());
    SDL::miniDoublets& miniDoubletsInGPU = (*event.getMiniDoublets());
    SDL::hits& hitsInGPU = (*event.getHits());

    // Did it match to track candidate?
    std::vector<int> sim_TC_matched(trk.sim_pt().size());

    for (unsigned int idx = 0; idx <= *(SDL::modulesInGPU->nLowerModules); idx++) // "<=" because cheating to include pixel track candidate lower module
    {
        for (unsigned int jdx = 0; jdx < trackCandidatesInGPU.nTrackCandidates[idx]; jdx++)
        {
            unsigned int trackCandidateIndex = idx * 50000/*_N_MAX_TRACK_CANDIDATES_PER_MODULE*/ + jdx;

            short trackCandidateType = trackCandidatesInGPU.trackCandidateType[trackCandidateIndex];

            unsigned int innerTrackletIdx = trackCandidatesInGPU.objectIndices[2 * trackCandidateIndex];
            unsigned int outerTrackletIdx = trackCandidatesInGPU.objectIndices[2 * trackCandidateIndex + 1];

            unsigned int innerTrackletInnerSegmentIndex = -1;
            unsigned int innerTrackletOuterSegmentIndex = -1;
            unsigned int outerTrackletOuterSegmentIndex = -1;

            if (trackCandidateType == 0) // T4T4
            {
                innerTrackletInnerSegmentIndex = trackletsInGPU.segmentIndices[2 * innerTrackletIdx];
                innerTrackletOuterSegmentIndex = trackletsInGPU.segmentIndices[2 * innerTrackletIdx + 1];
                outerTrackletOuterSegmentIndex = trackletsInGPU.segmentIndices[2 * outerTrackletIdx + 1];
            }
            else if (trackCandidateType == 1) // T4T3
            {
                innerTrackletInnerSegmentIndex = trackletsInGPU.segmentIndices[2 * innerTrackletIdx];
                innerTrackletOuterSegmentIndex = trackletsInGPU.segmentIndices[2 * innerTrackletIdx + 1];
                outerTrackletOuterSegmentIndex = tripletsInGPU.segmentIndices[2 * outerTrackletIdx + 1];
            }
            else if (trackCandidateType == 2) // T3T4
            {
                innerTrackletInnerSegmentIndex = tripletsInGPU.segmentIndices[2 * innerTrackletIdx];
                innerTrackletOuterSegmentIndex = tripletsInGPU.segmentIndices[2 * innerTrackletIdx + 1];
                outerTrackletOuterSegmentIndex = trackletsInGPU.segmentIndices[2 * outerTrackletIdx + 1];
            }

            unsigned int innerTrackletInnerSegmentInnerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * innerTrackletInnerSegmentIndex];
            unsigned int innerTrackletInnerSegmentOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * innerTrackletInnerSegmentIndex + 1];
            unsigned int innerTrackletOuterSegmentInnerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * innerTrackletOuterSegmentIndex];
            unsigned int innerTrackletOuterSegmentOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * innerTrackletOuterSegmentIndex + 1];
            unsigned int outerTrackletOuterSegmentInnerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * outerTrackletOuterSegmentIndex];
            unsigned int outerTrackletOuterSegmentOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * outerTrackletOuterSegmentIndex + 1];
            unsigned int innerTrackletInnerSegmentInnerMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * innerTrackletInnerSegmentInnerMiniDoubletIndex];
            unsigned int innerTrackletInnerSegmentInnerMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * innerTrackletInnerSegmentInnerMiniDoubletIndex + 1];
            unsigned int innerTrackletInnerSegmentOuterMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * innerTrackletInnerSegmentOuterMiniDoubletIndex];
            unsigned int innerTrackletInnerSegmentOuterMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * innerTrackletInnerSegmentOuterMiniDoubletIndex + 1];
            unsigned int innerTrackletOuterSegmentInnerMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * innerTrackletOuterSegmentInnerMiniDoubletIndex];
            unsigned int innerTrackletOuterSegmentInnerMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * innerTrackletOuterSegmentInnerMiniDoubletIndex + 1];
            unsigned int innerTrackletOuterSegmentOuterMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * innerTrackletOuterSegmentOuterMiniDoubletIndex];
            unsigned int innerTrackletOuterSegmentOuterMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * innerTrackletOuterSegmentOuterMiniDoubletIndex + 1];
            unsigned int outerTrackletOuterSegmentInnerMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * outerTrackletOuterSegmentInnerMiniDoubletIndex];
            unsigned int outerTrackletOuterSegmentInnerMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * outerTrackletOuterSegmentInnerMiniDoubletIndex + 1];
            unsigned int outerTrackletOuterSegmentOuterMiniDoubletLowerHitIndex = miniDoubletsInGPU.hitIndices[2 * outerTrackletOuterSegmentOuterMiniDoubletIndex];
            unsigned int outerTrackletOuterSegmentOuterMiniDoubletUpperHitIndex = miniDoubletsInGPU.hitIndices[2 * outerTrackletOuterSegmentOuterMiniDoubletIndex + 1];

            std::vector<int> hit_idxs = {
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

            std::vector<int> hit_types;
            if (idx == *(SDL::modulesInGPU->nLowerModules)) // Then this means this track candidate is a pLS-based
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

            // sim track matched index
            std::vector<int> matched_sim_trk_idxs = matchedSimTrkIdxs(hit_idxs, hit_types);

            for (auto& isimtrk : matched_sim_trk_idxs)
            {
                sim_TC_matched[isimtrk]++;
            }
        }
    }

    ana.tx->setBranch<vector<int>>("sim_TC_matched", sim_TC_matched);

    ana.tx->fill();
    ana.tx->clear();

}

void fillOutputBranches_for_CPU(SDL::CPU::Event& event)
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

    // simvtx
    ana.tx->setBranch<vector<float>>("simvtx_x", trk.simvtx_x());
    ana.tx->setBranch<vector<float>>("simvtx_y", trk.simvtx_y());
    ana.tx->setBranch<vector<float>>("simvtx_z", trk.simvtx_z());

    // Did it match to track candidate?
    std::vector<int> sim_TC_matched(trk.sim_pt().size());
    std::vector<vector<int>> sim_TC_types(trk.sim_pt().size());

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

            const SDL::CPU::Module& module0 = trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module1 = trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->getModule();
            const SDL::CPU::Module& module2 = trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module3 = trackCandidatePtr->innerTrackletBasePtr()->innerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->getModule();
            const SDL::CPU::Module& module4 = trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module5 = trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->getModule();
            const SDL::CPU::Module& module6 = trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module7 = trackCandidatePtr->innerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->getModule();
            const SDL::CPU::Module& module8 = trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module9 = trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->getModule();
            const SDL::CPU::Module& module10 = trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule();
            const SDL::CPU::Module& module11 = trackCandidatePtr->outerTrackletBasePtr()->outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->getModule();

            bool isPixel0 = module0.isPixelLayerModule();
            bool isPixel1 = module1.isPixelLayerModule();
            bool isPixel2 = module2.isPixelLayerModule();
            bool isPixel3 = module3.isPixelLayerModule();
            bool isPixel4 = module4.isPixelLayerModule();
            bool isPixel5 = module5.isPixelLayerModule();
            bool isPixel6 = module6.isPixelLayerModule();
            bool isPixel7 = module7.isPixelLayerModule();
            bool isPixel8 = module8.isPixelLayerModule();
            bool isPixel9 = module9.isPixelLayerModule();
            bool isPixel10 =module10.isPixelLayerModule();
            bool isPixel11 =module11.isPixelLayerModule();

            int layer0 = module0.layer();
            int layer1 = module1.layer();
            int layer2 = module2.layer();
            int layer3 = module3.layer();
            int layer4 = module4.layer();
            int layer5 = module5.layer();
            int layer6 = module6.layer();
            int layer7 = module7.layer();
            int layer8 = module8.layer();
            int layer9 = module9.layer();
            int layer10 =module10.layer();
            int layer11 =module11.layer();

            int subdet0 = module0.subdet();
            int subdet1 = module1.subdet();
            int subdet2 = module2.subdet();
            int subdet3 = module3.subdet();
            int subdet4 = module4.subdet();
            int subdet5 = module5.subdet();
            int subdet6 = module6.subdet();
            int subdet7 = module7.subdet();
            int subdet8 = module8.subdet();
            int subdet9 = module9.subdet();
            int subdet10 =module10.subdet();
            int subdet11 =module11.subdet();

            int logicallayer0 = isPixel0 ? 0 : layer0  + 6 * (subdet0 == 4);
            int logicallayer1 = isPixel1 ? 0 : layer1  + 6 * (subdet1 == 4);
            int logicallayer2 = isPixel2 ? 0 : layer2  + 6 * (subdet2 == 4);
            int logicallayer3 = isPixel3 ? 0 : layer3  + 6 * (subdet3 == 4);
            int logicallayer4 = isPixel4 ? 0 : layer4  + 6 * (subdet4 == 4);
            int logicallayer5 = isPixel5 ? 0 : layer5  + 6 * (subdet5 == 4);
            int logicallayer6 = isPixel6 ? 0 : layer6  + 6 * (subdet6 == 4);
            int logicallayer7 = isPixel7 ? 0 : layer7  + 6 * (subdet7 == 4);
            int logicallayer8 = isPixel8 ? 0 : layer8  + 6 * (subdet8 == 4);
            int logicallayer9 = isPixel9 ? 0 : layer9  + 6 * (subdet9 == 4);
            int logicallayer10 =isPixel10 ? 0 : layer10 + 6 * (subdet10 == 4);
            int logicallayer11 =isPixel11 ? 0 : layer11 + 6 * (subdet11 == 4);

            int layer_binary = 0;
            layer_binary |= (1 << logicallayer0);
            layer_binary |= (1 << logicallayer2);
            layer_binary |= (1 << logicallayer4);
            layer_binary |= (1 << logicallayer6);
            layer_binary |= (1 << logicallayer8);
            layer_binary |= (1 << logicallayer10);

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

        }

    }

    ana.tx->setBranch<vector<int>>("sim_TC_matched", sim_TC_matched);
    ana.tx->setBranch<vector<vector<int>>>("sim_TC_types", sim_TC_types);

    ana.tx->fill();
    ana.tx->clear();

}

void printTimingInformation(std::vector<std::vector<float>> timing_information)
{

    if (ana.verbose == 0)
        return;

    std::cout << showpoint;
    std::cout << fixed;
    std::cout << setprecision(2);
    std::cout << right;
    std::cout << "Timing summary" << std::endl;
    std::cout << "Evt     Hits  MD    LS    T4    T4x   pT4   T3    TC" << std::endl;
    std::vector<float> timing_sum_information(7);
    for (auto&& [ievt, timing] : iter::enumerate(timing_information))
    {
        std::cout << setw(6) << ievt;
        std::cout << setw(6) << timing[0]; // Hits
        std::cout << setw(6) << timing[1]; // MD
        std::cout << setw(6) << timing[2]; // LS
        std::cout << setw(6) << timing[3]; // T4
        std::cout << setw(6) << timing[4]; // T4x
        std::cout << setw(6) << timing[5]; // pT4
        std::cout << setw(6) << timing[6]; // T3
        std::cout << setw(6) << timing[7]; // TC
        std::cout << std::endl;
        timing_sum_information[0] += timing[0]; // Hits
        timing_sum_information[1] += timing[1]; // MD
        timing_sum_information[2] += timing[2]; // LS
        timing_sum_information[3] += timing[3]; // T4
        timing_sum_information[4] += timing[4]; // T4x
        timing_sum_information[5] += timing[5]; // pT4
        timing_sum_information[6] += timing[6]; // T3
        timing_sum_information[7] += timing[7]; // TC
    }
    timing_sum_information[0] /= timing_information.size(); // Hits
    timing_sum_information[1] /= timing_information.size(); // MD
    timing_sum_information[2] /= timing_information.size(); // LS
    timing_sum_information[3] /= timing_information.size(); // T4
    timing_sum_information[4] /= timing_information.size(); // T4x
    timing_sum_information[5] /= timing_information.size(); // pT4
    timing_sum_information[6] /= timing_information.size(); // T3
    timing_sum_information[7] /= timing_information.size(); // TC
    std::cout << setw(6) << "avg";
    std::cout << setw(6) << timing_sum_information[0]; // Hits
    std::cout << setw(6) << timing_sum_information[1]; // MD
    std::cout << setw(6) << timing_sum_information[2]; // LS
    std::cout << setw(6) << timing_sum_information[3]; // T4
    std::cout << setw(6) << timing_sum_information[4]; // T4x
    std::cout << setw(6) << timing_sum_information[5]; // pT4
    std::cout << setw(6) << timing_sum_information[6]; // T3
    std::cout << setw(6) << timing_sum_information[7]; // T3
    std::cout << std::endl;

    std::cout << left;

}

void printQuadrupletMultiplicities(SDL::Event& event)
{
    SDL::tracklets& trackletsInGPU = (*event.getTracklets());

    int nTracklets = 0;
    for (unsigned int idx = 0; idx <= *(SDL::modulesInGPU->nLowerModules); idx++) // "<=" because cheating to include pixel track candidate lower module
    {
        nTracklets += trackletsInGPU.nTracklets[idx];
    }
    std::cout <<  " nTracklets: " << nTracklets <<  std::endl;
}

void printHitMultiplicities(SDL::Event& event)
{
    SDL::hits& hitsInGPU = (*event.getHits());

    int nHits = 0;
    for (unsigned int idx = 0; idx <= *(SDL::modulesInGPU->nLowerModules); idx++) // "<=" because cheating to include pixel track candidate lower module
    {
        nHits += SDL::modulesInGPU->hitRanges[4 * idx + 1] - SDL::modulesInGPU->hitRanges[4 * idx] + 1;       
        nHits += SDL::modulesInGPU->hitRanges[4 * idx + 3] - SDL::modulesInGPU->hitRanges[4 * idx + 2] + 1;
    }
    std::cout <<  " nHits: " << nHits <<  std::endl;
}

void printMiniDoubletMultiplicities(SDL::Event& event)
{
    SDL::miniDoublets& miniDoubletsInGPU = (*event.getMiniDoublets());

    int nMiniDoublets = 0;
    for (unsigned int idx = 0; idx <= *(SDL::modulesInGPU->nModules); idx++) // "<=" because cheating to include pixel track candidate lower module
    {
        nMiniDoublets += miniDoubletsInGPU.nMDs[idx];
    }
    std::cout <<  " nMiniDoublets: " << nMiniDoublets <<  std::endl;
}
