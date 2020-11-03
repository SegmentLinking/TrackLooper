#include "StudyTrackCandidateSelection.h"

StudyTrackCandidateSelection::StudyTrackCandidateSelection(const char* studyName, StudyTrackCandidateSelection::StudyTrackCandidateSelectionMode mode_, std::vector<float> ptbounds)
{

    studyname = studyName;
    mode = mode_;
    switch (mode)
    {
        case kStudySelAll: modename = "all"; break;
        case kStudySelSpecific: modename = "specific"; break;
        default: modename = "UNDEFINED"; break;
    }
    pt_boundaries = ptbounds;

}

void StudyTrackCandidateSelection::bookStudy()
{
    // Book Histograms
    ana.histograms.addVecHistogram(TString::Format("tc_%s_cutflow", modename), 8, 0, 8, [&]() { return tc_cutflow; } );
    ana.histograms.addVecHistogram(TString::Format("tc_%s_outer_tl_deltaBeta", modename), 180, -0.15, 0.15, [&]() { return tc_outer_tl_deltaBeta; } );
    ana.histograms.addVecHistogram(TString::Format("tc_%s_inner_tl_deltaBeta", modename), 180, -0.15, 0.15, [&]() { return tc_inner_tl_deltaBeta; } );
    ana.histograms.addVecHistogram(TString::Format("tc_%s_nocut_outer_tl_deltaBeta", modename), 180, -0.15, 0.15, [&]() { return tc_nocut_outer_tl_deltaBeta; } );
    ana.histograms.addVecHistogram(TString::Format("tc_%s_nocut_inner_tl_deltaBeta", modename), 180, -0.15, 0.15, [&]() { return tc_nocut_inner_tl_deltaBeta; } );
    ana.histograms.addVecHistogram(TString::Format("tc_%s_inner_tl_betaIn_minus_outer_tl_betaOut", modename), 180, -0.15, 0.15, [&]() { return tc_inner_tl_betaIn_minus_outer_tl_betaOut; } );
    ana.histograms.addVecHistogram(TString::Format("tc_%s_inner_tl_betaAv_minus_outer_tl_betaAv", modename), 180, -0.15, 0.15, [&]() { return tc_inner_tl_betaAv_minus_outer_tl_betaAv; } );
    ana.histograms.addVecHistogram(TString::Format("tc_%s_dr", modename), 180, -1.5, 1.5, [&]() { return tc_dr; } );
    ana.histograms.add2DVecHistogram(TString::Format("tc_%s_dr", modename), 180, -1.5, 1.5, TString::Format("tc_%s_r", modename), 180, 0., 15000., [&]() { return tc_dr; }, [&]() { return tc_r; } );
    ana.histograms.add2DVecHistogram(TString::Format("truth_tc_%s_dr", modename), 180, -1.5, 1.5, TString::Format("truth_tc_%s_r", modename), 180, 0., 15000., [&]() { return truth_tc_dr; }, [&]() { return truth_tc_r; } );
    ana.histograms.add2DVecHistogram(TString::Format("truth_gt1pt_tc_%s_dr", modename), 180, -1.5, 1.5, TString::Format("truth_gt1pt_tc_%s_r", modename), 180, 0., 15000., [&]() { return truth_gt1pt_tc_dr; }, [&]() { return truth_gt1pt_tc_r; } );

    ana.histograms.addVecHistogram(TString::Format("tc_%s_matched_track_pt", modename), pt_boundaries, [&]() { return tc_matched_track_pt; } );
    ana.histograms.addVecHistogram(TString::Format("tc_%s_all_track_pt", modename), pt_boundaries, [&]() { return tc_all_track_pt; } );

    const int nlayers = NLAYERS;
    for (int ii = 0; ii < nlayers; ++ii)
    {
        ana.histograms.addVecHistogram(TString::Format("tc_%s_cutflow_by_layer%d", modename, ii), 8, 0, 8, [&, ii]() { return tc_cutflow_by_layer[ii]; } );
    }

}

void StudyTrackCandidateSelection::doStudy(SDL::Event& event, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents)
{

    // Each do study is performed per event

    // The study assumes that the createTrackCandidates has been run with "AllComb" algorithm
    // The DefaultAlgo will be run individually here

    // First clear all the output variables that will be used to fill the histograms for this event
    tc_cutflow.clear();
    tc_outer_tl_deltaBeta.clear();
    tc_inner_tl_deltaBeta.clear();
    tc_nocut_outer_tl_deltaBeta.clear();
    tc_nocut_inner_tl_deltaBeta.clear();
    tc_inner_tl_betaIn_minus_outer_tl_betaOut.clear();
    tc_inner_tl_betaAv_minus_outer_tl_betaAv.clear();
    tc_dr.clear();
    tc_r.clear();
    truth_tc_dr.clear();
    truth_tc_r.clear();
    truth_gt1pt_tc_dr.clear();
    truth_gt1pt_tc_r.clear();
    for (int ii = 0; ii < NLAYERS; ++ii)
    {
        tc_cutflow_by_layer[ii].clear();
    }

    tc_matched_track_pt.clear();
    tc_all_track_pt.clear();

    //***********************
    // Studying selections and cutflows for the recoed events
    //***********************

    //// Loop over tracklets in event
    //for (auto& layerPtr : event.getLayerPtrs())
    //{

    //    // Parse the layer index later to be used for indexing
    //    int layer_idx = layerPtr->layerIdx() - 1;

    //    // This means no tracklets in this layer
    //    if (layerPtr->getTrackCandidatePtrs().size() == 0)
    //    {
    //        continue;
    //    }

    //    // Assuming I have at least one tracklets from this track
    //    std::vector<SDL::TrackCandidate*> tcs_of_interest;
    //    for (auto& tc : layerPtr->getTrackCandidatePtrs())
    //    {
    //        // Depending on the mode, only include a subset of interested tracklets
    //        switch (mode)
    //        {
    //            case kStudySelAll: /* do nothing */ break;
    //            case kStudySelSpecific: break;
    //            default: /* skip everything should not be here anyways...*/ continue; break;
    //        }

    //        // If this tracklet passes the condition that it is of interest then, add to the list of tracklets of interest
    //        tcs_of_interest.push_back(tc);

    //    }

    //    // If no tcs of interest are found then skip
    //    if (tcs_of_interest.size() == 0)
    //        continue;

    //    // The tcs_of_interest holds only the tc "candidate" that we think are of interest for the given study mode
    //    for (auto& tc : tcs_of_interest)
    //    {

    //        tc->runTrackCandidateAlgo(SDL::Default_TCAlgo);

    //        // Cutflow
    //        //------------------------
    //        tc_cutflow.push_back(0);
    //        tc_cutflow_by_layer[layer_idx].push_back(0);

    //        const int& passbit = tc->getPassBitsDefaultAlgo();

    //        for (unsigned int i = 0; i < SDL::TrackCandidate::TrackCandidateSelection::nCut; ++i)
    //        {
    //            if (passbit & (1 << i))
    //            {
    //                tc_cutflow.push_back(i + 1);
    //                tc_cutflow_by_layer[layer_idx].push_back(i + 1);
    //            }
    //            else
    //            {
    //                break;
    //            }
    //        }

    //        tc_nocut_outer_tl_deltaBeta.push_back(tc->outerTrackletPtr()->getDeltaBeta());
    //        tc_nocut_inner_tl_deltaBeta.push_back(tc->innerTrackletPtr()->getDeltaBeta());

    //        if (passbit & ( 1 << SDL::TrackCandidate::TrackCandidateSelection::commonSegment))
    //        {
    //            tc_outer_tl_deltaBeta.push_back(tc->outerTrackletPtr()->getDeltaBeta());
    //            tc_inner_tl_deltaBeta.push_back(tc->innerTrackletPtr()->getDeltaBeta());

    //            tc_inner_tl_betaIn_minus_outer_tl_betaOut.push_back(tc->innerTrackletPtr()->getBetaIn() - tc->outerTrackletPtr()->getBetaOut());
    //            tc_inner_tl_betaAv_minus_outer_tl_betaAv.push_back(tc->innerTrackletPtr()->getRecoVar("betaAv") - tc->outerTrackletPtr()->getRecoVar("betaAv"));
    //            tc_dr.push_back(tc->getRecoVar("dR"));
    //            tc_r.push_back(tc->getRecoVar("innerR"));

    //        }

    //    }

    //}

    // Loop over track events
    unsigned int isimtrkevent = 0;
    for (auto& simtrkevent : simtrkevents)
    {

        isimtrkevent++;

        // Unpack the tuple (sim_track_index, SDL::Event containing reco hits only matched to the given sim track)
        unsigned int& isimtrk = std::get<0>(simtrkevent);
        SDL::Event& trackevent = *(std::get<1>(simtrkevent));

        bool trackmatch = false;

        // Loop over the layer that contains tracklets for this track
        for (auto& layerPtr_Track : trackevent.getLayerPtrs())
        {

            // Parse the layer index later to be used for indexing
            int layer_idx = layerPtr_Track->layerIdx() - 1;

            // This means no tracklets in this layer
            if (layerPtr_Track->getTrackCandidatePtrs().size() == 0)
            {
                continue;
            }

            // Assuming I have at least one tracklets from this track
            std::vector<SDL::TrackCandidate*> tcs_of_interest;
            for (auto& tc : layerPtr_Track->getTrackCandidatePtrs())
            {
                // Depending on the mode, only include a subset of interested tracklets
                switch (mode)
                {
                    case kStudySelAll: /* do nothing */ break;
                    case kStudySelSpecific: break;
                    default: /* skip everything should not be here anyways...*/ continue; break;
                }

                // If this tracklet passes the condition that it is of interest then, add to the list of tracklets of interest
                tcs_of_interest.push_back(tc);

            }

            // If no tcs of interest are found then skip
            if (tcs_of_interest.size() == 0)
                continue;

            float dr_with_min_dr = 999;
            float r_with_min_dr = 10000000;

            bool match = false;
            for (auto& tc_Track : tcs_of_interest)
            {

                const SDL::Module& innerTlinnerSgInnerMDLowerHitModule = tc_Track->innerTrackletPtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule();

                // Loop over the tc reconstructed from with proper SDL algorithm and if the index of the tracklets match (i.e. if the 8 hits match)
                // Then we have found at least one tracklets associated to this track's reco-hits in this module
                // Therefore flag the match boolean
                SDL::Layer::SubDet innerLayerSubDet = innerTlinnerSgInnerMDLowerHitModule.subdet() == SDL::Module::Barrel ? SDL::Layer::Barrel : SDL::Layer::Endcap;
                // for (auto& tc : event.getLayer(innerTlinnerSgInnerMDLowerHitModule.layer(), innerLayerSubDet).getTrackCandidatePtrs())
                for (auto& tc : event.getLayer(1, SDL::Layer::Barrel).getTrackCandidatePtrs())
                {
                    if (tc_Track->isIdxMatched(*tc))
                    {
                        match = true;
                    }
                }
            }

            // The tcs_of_interest holds only the tc "candidate" that we think are of interest for the given study mode
            for (auto& tc : tcs_of_interest)
            {

                const int& passbit = tc->getPassBitsDefaultAlgo();

                if (passbit & ( 1 << SDL::TrackCandidate::TrackCandidateSelection::commonSegment))
                {

                    if (dr_with_min_dr > tc->getRecoVar("dR"))
                    {
                        dr_with_min_dr = tc->getRecoVar("dR");
                        r_with_min_dr = tc->getRecoVar("innerR");
                    }

                }

            }

            truth_tc_dr.push_back(dr_with_min_dr);
            truth_tc_r.push_back(r_with_min_dr);

            if (trk.sim_pt()[isimtrk] > 1.)
            {
                truth_gt1pt_tc_dr.push_back(dr_with_min_dr);
                truth_gt1pt_tc_r.push_back(r_with_min_dr);
            }

            if (match)
                trackmatch = true;

        }

        float pt = std::min((double) trk.sim_pt()[isimtrk], 49.999);

        if (trackmatch)
        {
            tc_matched_track_pt.push_back(pt);
        }
        tc_all_track_pt.push_back(pt);


    }

}
