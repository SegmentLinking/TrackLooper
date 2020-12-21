#include "write_sdl_ntuple.h"

void write_sdl_ntuple()
{

    // Load various maps used in the SDL reconstruction
    loadMaps();

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

        // Main instance that will hold modules, hits, minidoublets, etc. (i.e. main data structure)
        SDL::Event event;

        // Add hits to the event
        float timing_input_loading = addInputsToLineSegmentTrackingUsingUnifiedMemory(event);

        // Run Mini-doublet
        float timing_MD = runMiniDoublet(event);

        // Run Segment
        float timing_LS = runSegment(event);

        // Run T4
        float timing_T4 = runT4(event);

        // Run T4x
        float timing_T4x = runT4x(event);

        // Run pT4
        float timing_pT4 = runpT4(event);

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

    // printTimingInformation(timing_information);

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

void printTimingInformation(std::vector<std::vector<float>> timing_information)
{
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
