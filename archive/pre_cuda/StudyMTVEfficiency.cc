#include "StudyMTVEfficiency.h"

StudyMTVEfficiency::StudyMTVEfficiency(
        const char* studyName,
        std::vector<float> ptbounds,
        int pdgid
        )
{

    studyname = studyName;

    pt_boundaries = ptbounds;

    pdgid_of_study = pdgid;

    dxy_thresh = 2.5;

    dz_thresh = 30;

    pt_thresh = 0.9;

}

void StudyMTVEfficiency::bookStudy()
{
    bookStudy_v2();
}

void StudyMTVEfficiency::bookStudy_v1()
{
    // Book Histograms
    const int nlayers = NLAYERS;

    const float etamax = 2.5;

    const int eta_nbins = 50;

    ana.histograms.addVecHistogram("tc_matched_track_pt_mtv" , pt_boundaries, [&]()                { return tc_matched_track_pt_mtv[0];  } );
    ana.histograms.addVecHistogram("tc_all_track_pt_mtv"     , pt_boundaries, [&]()                { return tc_all_track_pt_mtv[0];      } );
    ana.histograms.addVecHistogram("tc_matched_track_eta_mtv", eta_nbins, -1*etamax, etamax, [&]() { return tc_matched_track_eta_mtv[0]; } );
    ana.histograms.addVecHistogram("tc_all_track_eta_mtv"    , eta_nbins, -1*etamax, etamax, [&]() { return tc_all_track_eta_mtv[0];     } );
    ana.histograms.addVecHistogram("tc_matched_track_dxy_mtv", 50, -10, 10, [&]()                  { return tc_matched_track_dxy_mtv[0]; } );
    ana.histograms.addVecHistogram("tc_all_track_dxy_mtv"    , 50, -10, 10, [&]()                  { return tc_all_track_dxy_mtv[0];     } );

    ana.histograms.addVecHistogram("tc_matched_track_pt_mtv_eta0_0p4" , pt_boundaries, [&]()                { return tc_matched_track_pt_mtv[1];  } );
    ana.histograms.addVecHistogram("tc_all_track_pt_mtv_eta0_0p4"     , pt_boundaries, [&]()                { return tc_all_track_pt_mtv[1];      } );
    ana.histograms.addVecHistogram("tc_matched_track_eta_mtv_eta0_0p4", eta_nbins, -1*etamax, etamax, [&]() { return tc_matched_track_eta_mtv[1]; } );
    ana.histograms.addVecHistogram("tc_all_track_eta_mtv_eta0_0p4"    , eta_nbins, -1*etamax, etamax, [&]() { return tc_all_track_eta_mtv[1];     } );
    ana.histograms.addVecHistogram("tc_matched_track_dxy_mtv_eta0_0p4", 50, -10, 10, [&]()                  { return tc_matched_track_dxy_mtv[1]; } );
    ana.histograms.addVecHistogram("tc_all_track_dxy_mtv_eta0_0p4"    , 50, -10, 10, [&]()                  { return tc_all_track_dxy_mtv[1];     } );

    ana.histograms.addVecHistogram("tc_matched_track_pt_mtv_eta0p4_0p8" , pt_boundaries, [&]()                { return tc_matched_track_pt_mtv[2];  } );
    ana.histograms.addVecHistogram("tc_all_track_pt_mtv_eta0p4_0p8"     , pt_boundaries, [&]()                { return tc_all_track_pt_mtv[2];      } );
    ana.histograms.addVecHistogram("tc_matched_track_eta_mtv_eta0p4_0p8", eta_nbins, -1*etamax, etamax, [&]() { return tc_matched_track_eta_mtv[2]; } );
    ana.histograms.addVecHistogram("tc_all_track_eta_mtv_eta0p4_0p8"    , eta_nbins, -1*etamax, etamax, [&]() { return tc_all_track_eta_mtv[2];     } );
    ana.histograms.addVecHistogram("tc_matched_track_dxy_mtv_eta0p4_0p8", 50, -10, 10, [&]()                  { return tc_matched_track_dxy_mtv[2]; } );
    ana.histograms.addVecHistogram("tc_all_track_dxy_mtv_eta0p4_0p8"    , 50, -10, 10, [&]()                  { return tc_all_track_dxy_mtv[2];     } );

}

void StudyMTVEfficiency::bookStudy_v2()
{
    // Book Histograms
    const int nlayers = NLAYERS;

    const float etamax = 2.5;

    const int eta_nbins = 50;

    const float phimax = 3.1416;

    const int phi_nbins = 1080;

    ana.histograms.addVecHistogram("tc_matched_track_pt_mtv" , pt_boundaries, [&]()                { return tc_matched_track_pt_mtv[0];  } );
    ana.histograms.addVecHistogram("tc_all_track_pt_mtv"     , pt_boundaries, [&]()                { return tc_all_track_pt_mtv[0];      } );
    ana.histograms.addVecHistogram("tc_matched_track_eta_mtv", eta_nbins, -1*etamax, etamax, [&]() { return tc_matched_track_eta_mtv[0]; } );
    ana.histograms.addVecHistogram("tc_all_track_eta_mtv"    , eta_nbins, -1*etamax, etamax, [&]() { return tc_all_track_eta_mtv[0];     } );
    ana.histograms.addVecHistogram("tc_matched_track_phi_mtv", phi_nbins, -1*phimax, phimax, [&]() { return tc_matched_track_phi_mtv[0]; } );
    ana.histograms.addVecHistogram("tc_all_track_phi_mtv"    , phi_nbins, -1*phimax, phimax, [&]() { return tc_all_track_phi_mtv[0];     } );
    ana.histograms.addVecHistogram("tc_matched_track_dxy_mtv", 50, -10, 10, [&]()                  { return tc_matched_track_dxy_mtv[0]; } );
    ana.histograms.addVecHistogram("tc_all_track_dxy_mtv"    , 50, -10, 10, [&]()                  { return tc_all_track_dxy_mtv[0];     } );

    ana.histograms.addVecHistogram("tc_notmatched_trackcandidate_pt_mtv" , pt_boundaries, [&]()                { return tc_matched_track_pt_mtv[1];  } ); // NOTE: Name of the variable is a MISNOMER!!
    ana.histograms.addVecHistogram("tc_all_trackcandidate_pt_mtv"        , pt_boundaries, [&]()                { return tc_all_track_pt_mtv[1];      } ); // NOTE: Name of the variable is a MISNOMER!!
    ana.histograms.addVecHistogram("tc_notmatched_trackcandidate_eta_mtv", eta_nbins, -1*etamax, etamax, [&]() { return tc_matched_track_eta_mtv[1]; } ); // NOTE: Name of the variable is a MISNOMER!!
    ana.histograms.addVecHistogram("tc_all_trackcandidate_eta_mtv"       , eta_nbins, -1*etamax, etamax, [&]() { return tc_all_track_eta_mtv[1];     } ); // NOTE: Name of the variable is a MISNOMER!!
    ana.histograms.addVecHistogram("tc_notmatched_trackcandidate_dxy_mtv", 50, -10, 10, [&]()                  { return tc_matched_track_dxy_mtv[1]; } ); // NOTE: Name of the variable is a MISNOMER!!
    ana.histograms.addVecHistogram("tc_all_trackcandidate_dxy_mtv"       , 50, -10, 10, [&]()                  { return tc_all_track_dxy_mtv[1];     } ); // NOTE: Name of the variable is a MISNOMER!!

}

void StudyMTVEfficiency::doStudy(SDL::Event& event, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents)
{
    doStudy_v2(event, simtrkevents);
}

void StudyMTVEfficiency::doStudy_v2(SDL::Event& event, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents)
{
    // Each do study is performed per event

    // First clear all the output variables that will be used to fill the histograms for this event

    for (unsigned int i = 0; i < 3; ++i)
    {
        tc_matched_track_pt_mtv [i].clear();
        tc_all_track_pt_mtv     [i].clear();
        tc_matched_track_eta_mtv[i].clear();
        tc_all_track_eta_mtv    [i].clear();
        tc_matched_track_phi_mtv[i].clear();
        tc_all_track_phi_mtv    [i].clear();
        tc_matched_track_dxy_mtv[i].clear();
        tc_all_track_dxy_mtv    [i].clear();
    }

    // Renaming the variables so I don't get confused
    vector<float>& tc_matched_sim_track_pt_mtv  = tc_matched_track_pt_mtv [0];
    vector<float>& tc_all_sim_track_pt_mtv      = tc_all_track_pt_mtv     [0];
    vector<float>& tc_matched_sim_track_eta_mtv = tc_matched_track_eta_mtv[0];
    vector<float>& tc_all_sim_track_eta_mtv     = tc_all_track_eta_mtv    [0];
    vector<float>& tc_matched_sim_track_phi_mtv = tc_matched_track_phi_mtv[0];
    vector<float>& tc_all_sim_track_phi_mtv     = tc_all_track_phi_mtv    [0];
    vector<float>& tc_matched_sim_track_dxy_mtv = tc_matched_track_dxy_mtv[0];
    vector<float>& tc_all_sim_track_dxy_mtv     = tc_all_track_dxy_mtv    [0];

    // Renaming the variables so I don't get confused
    vector<float>& tc_notmatched_trackcandidate_pt_mtv  = tc_matched_track_pt_mtv [1];
    vector<float>& tc_all_trackcandidate_pt_mtv         = tc_all_track_pt_mtv     [1];
    vector<float>& tc_notmatched_trackcandidate_eta_mtv = tc_matched_track_eta_mtv[1];
    vector<float>& tc_all_trackcandidate_eta_mtv        = tc_all_track_eta_mtv    [1];
    vector<float>& tc_notmatched_trackcandidate_phi_mtv = tc_matched_track_phi_mtv[1];
    vector<float>& tc_all_trackcandidate_phi_mtv        = tc_all_track_phi_mtv    [1];
    vector<float>& tc_notmatched_trackcandidate_dxy_mtv = tc_matched_track_dxy_mtv[1];
    vector<float>& tc_all_trackcandidate_dxy_mtv        = tc_all_track_dxy_mtv    [1];

    // The denominators are selected here
    for (unsigned int isimtrk = 0; isimtrk < trk.sim_pt().size(); ++isimtrk)
    {
        float pt = std::min((double) trk.sim_pt()[isimtrk], 49.999);
        float eta = trk.sim_eta()[isimtrk];
        float phi = trk.sim_phi()[isimtrk];
        float dxy = trk.sim_pca_dxy()[isimtrk];
        float dz = trk.sim_pca_dz()[isimtrk];

        if (abs(dz) > dz_thresh)
            continue;

        if (goodBarrelTrack(isimtrk, pdgid_of_study))
        {
            if (abs(dxy) < dxy_thresh and abs(eta) < 0.8)
                tc_all_sim_track_pt_mtv.push_back(pt);
            if (abs(dxy) < dxy_thresh and abs(pt) > pt_thresh)
                tc_all_sim_track_eta_mtv.push_back(eta);
            if (abs(eta) < 0.8 and abs(pt) > pt_thresh)
                tc_all_sim_track_dxy_mtv.push_back(dxy);
            if (abs(eta) < 0.8 and abs(dxy) < dxy_thresh)
                tc_all_sim_track_phi_mtv.push_back(phi);
        }

    }

    // The numerators are selected here
    vector<int> numer_trk_idxs; // For efficiency
    for (auto& tc : event.getLayer(1, SDL::Layer::Barrel).getTrackCandidatePtrs())
    {
        vector<int> matched_simtrk_idxs = matchedSimTrkIdxs(tc);

        // If there is a good matched sim track
        for (auto& matched_simtrk_idx : matched_simtrk_idxs)
        {
            if (std::find(numer_trk_idxs.begin(), numer_trk_idxs.end(), matched_simtrk_idx) == numer_trk_idxs.end())
                numer_trk_idxs.push_back(matched_simtrk_idx);
        }

        float inner_betaPt_2nd = tc->innerTrackletPtr()->getRecoVar("betaPt_2nd");
        float outer_betaPt_2nd = tc->outerTrackletPtr()->getRecoVar("betaPt_2nd");
        float pt_estimate = (inner_betaPt_2nd + outer_betaPt_2nd) / 2.;

        SDL::Hit innermosthit = (*(tc->innerTrackletPtr()->innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()));
        SDL::Hit outermosthit = (*(tc->outerTrackletPtr()->outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()));

        SDL::Hit fullseg = outermosthit - innermosthit;

        float eta_estimate = copysign(-std::log(std::tan(std::atan(fullseg.rt() / abs(fullseg.z())) / 2.)), fullseg.z());

        tc_all_trackcandidate_pt_mtv.push_back(pt_estimate);
        tc_all_trackcandidate_eta_mtv.push_back(eta_estimate);

        // if there is a matched one
        if (matched_simtrk_idxs.size() == 0)
        {
            tc_notmatched_trackcandidate_pt_mtv.push_back(pt_estimate);
            tc_notmatched_trackcandidate_eta_mtv.push_back(eta_estimate);
        }

    }

    // Loop over to fill the efficiency related numerator object
    for (auto& numer_trk_idx : numer_trk_idxs)
    {

        // Then get the values and fill
        float pt = std::min((double) trk.sim_pt()[numer_trk_idx], 49.999);
        float eta = trk.sim_eta()[numer_trk_idx];
        float phi = trk.sim_phi()[numer_trk_idx];
        float dxy = trk.sim_pca_dxy()[numer_trk_idx];
        float dz = trk.sim_pca_dz()[numer_trk_idx];

        if (abs(dz) > dz_thresh)
            continue;

        // And is a denominator track
        if (goodBarrelTrack(numer_trk_idx, pdgid_of_study))
        {
            if (abs(dxy) < dxy_thresh and abs(eta) < 0.8)
                tc_matched_sim_track_pt_mtv.push_back(pt);
            if (abs(dxy) < dxy_thresh and abs(pt) > pt_thresh)
                tc_matched_sim_track_eta_mtv.push_back(eta);
            if (abs(eta) < 0.8 and abs(pt) > pt_thresh)
                tc_matched_sim_track_dxy_mtv.push_back(dxy);
            if (abs(eta) < 0.8 and abs(dxy) < dxy_thresh)
                tc_matched_sim_track_phi_mtv.push_back(phi);
        }

    }

    // Loop over to 

}

void StudyMTVEfficiency::doStudy_v1(SDL::Event& event, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents)
{
    // Each do study is performed per event

    // First clear all the output variables that will be used to fill the histograms for this event

    for (unsigned int i = 0; i < 3; ++i)
    {
        tc_matched_track_pt_mtv [i].clear();
        tc_all_track_pt_mtv     [i].clear();
        tc_matched_track_eta_mtv[i].clear();
        tc_all_track_eta_mtv    [i].clear();
        tc_matched_track_dxy_mtv[i].clear();
        tc_all_track_dxy_mtv    [i].clear();
    }

    // Barrel layer layer 0
    SDL::Layer barrelLayer0 = event.getLayer(1, SDL::Layer::Barrel);

    //***************************************************************
    // Preselecting Reco Track Candidates that are worth looking into
    //***************************************************************

    // track candidates that match the particle of interest
    std::vector<unsigned int> good_track_candidates;

    std::cout <<  " barrelLayer0.getTrackCandidatePtrs().size(): " << barrelLayer0.getTrackCandidatePtrs().size() <<  std::endl;

    // Loop over the track candidates and ask whether the hits are matched with this track (MTV like)
    for (unsigned int itc = 0; itc < barrelLayer0.getTrackCandidatePtrs().size(); ++itc)
    {

        SDL::TrackCandidate* tcCandPtr = barrelLayer0.getTrackCandidatePtrs()[itc];
        std::vector<unsigned int> hitidxs;
        hitidxs.push_back(tcCandPtr->innerTrackletPtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx());
        hitidxs.push_back(tcCandPtr->innerTrackletPtr()->innerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx());
        hitidxs.push_back(tcCandPtr->innerTrackletPtr()->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx());
        hitidxs.push_back(tcCandPtr->innerTrackletPtr()->innerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx());
        hitidxs.push_back(tcCandPtr->innerTrackletPtr()->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx());
        hitidxs.push_back(tcCandPtr->innerTrackletPtr()->outerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx());
        hitidxs.push_back(tcCandPtr->innerTrackletPtr()->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx());
        hitidxs.push_back(tcCandPtr->innerTrackletPtr()->outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx());
        hitidxs.push_back(tcCandPtr->outerTrackletPtr()->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx());
        hitidxs.push_back(tcCandPtr->outerTrackletPtr()->outerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx());
        hitidxs.push_back(tcCandPtr->outerTrackletPtr()->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx());
        hitidxs.push_back(tcCandPtr->outerTrackletPtr()->outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx());
        if (is75percentFromSimMatchedHits(hitidxs, pdgid_of_study))
        {
            good_track_candidates.push_back(itc);
        }
    }

    std::cout <<  " good_track_candidates.size(): " << good_track_candidates.size() <<  std::endl;

    //***********************
    // Efficiency calculation
    //***********************

    std::array<float, 6> etabounds = {2.2, 1.8, 1.45, 1.2, 1, 0.8};

    std::cout <<  " simtrkevents.size(): " << simtrkevents.size() <<  std::endl;

    int nprocessed = 0;

    // Loop over track events
    for (auto& simtrkevent : simtrkevents)
    {

        std::cout <<  " nprocessed: " << nprocessed <<  std::endl;
        nprocessed++;

        // Unpack the tuple (sim_track_index, SDL::Event containing reco hits only matched to the given sim track)
        unsigned int& isimtrk = std::get<0>(simtrkevent);
        SDL::Event& trackevent = *(std::get<1>(simtrkevent));

        // Create list of booleans
        bool tc_trackmatch_mtv = false;;

        // Parse pt and eta of this track
        float pt = std::min((double) trk.sim_pt()[isimtrk], 49.999);
        float eta = trk.sim_eta()[isimtrk];
        float dxy = trk.sim_pca_dxy()[isimtrk];
        float dz = trk.sim_pca_dz()[isimtrk];

        if (abs(dz) > dz_thresh)
            continue;

        // Loop over the track candidates and ask whether the hits are matched with this track (MTV like)
        for (auto& igood_tc_idxs : good_track_candidates)
        {

            SDL::TrackCandidate* tcCandPtr = barrelLayer0.getTrackCandidatePtrs()[igood_tc_idxs];
            std::vector<unsigned int> hitidxs;
            hitidxs.push_back(tcCandPtr->innerTrackletPtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx());
            hitidxs.push_back(tcCandPtr->innerTrackletPtr()->innerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx());
            hitidxs.push_back(tcCandPtr->innerTrackletPtr()->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx());
            hitidxs.push_back(tcCandPtr->innerTrackletPtr()->innerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx());
            hitidxs.push_back(tcCandPtr->innerTrackletPtr()->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx());
            hitidxs.push_back(tcCandPtr->innerTrackletPtr()->outerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx());
            hitidxs.push_back(tcCandPtr->innerTrackletPtr()->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx());
            hitidxs.push_back(tcCandPtr->innerTrackletPtr()->outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx());
            hitidxs.push_back(tcCandPtr->outerTrackletPtr()->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx());
            hitidxs.push_back(tcCandPtr->outerTrackletPtr()->outerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx());
            hitidxs.push_back(tcCandPtr->outerTrackletPtr()->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx());
            hitidxs.push_back(tcCandPtr->outerTrackletPtr()->outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx());
            if (isMTVMatch(isimtrk, hitidxs))
            {
                tc_trackmatch_mtv = true;
            }

        }

        // *****************
        // Track Candidate
        // *****************

        float etamaxbound = etabounds[5];

        if (abs(eta) < etamaxbound and abs(dxy) < dxy_thresh)
        {

            tc_all_track_pt_mtv[0].push_back(pt);
            if (tc_trackmatch_mtv) tc_matched_track_pt_mtv[0].push_back(pt);

            if (abs(eta) < 0.4)
            {
                tc_all_track_pt_mtv[1].push_back(pt);
                if (tc_trackmatch_mtv) tc_matched_track_pt_mtv[1].push_back(pt);
            }
            if (abs(eta) > 0.4 and abs(eta) < 0.8)
            {
                tc_all_track_pt_mtv[2].push_back(pt);
                if (tc_trackmatch_mtv) tc_matched_track_pt_mtv[2].push_back(pt);
            }
        }

        if (pt > pt_thresh and abs(dxy) < dxy_thresh and abs(eta) < etamaxbound)
        {

            tc_all_track_eta_mtv[0].push_back(eta);
            if (tc_trackmatch_mtv) tc_matched_track_eta_mtv[0].push_back(eta);

            if (abs(eta) < 0.4)
            {
                tc_all_track_eta_mtv[1].push_back(eta);
                if (tc_trackmatch_mtv) tc_matched_track_eta_mtv[1].push_back(eta);
            }
            if (abs(eta) > 0.4 and abs(eta) < 0.8)
            {
                tc_all_track_eta_mtv[2].push_back(eta);
                if (tc_trackmatch_mtv) tc_matched_track_eta_mtv[2].push_back(eta);
            }
        }

        if (pt > pt_thresh and abs(eta) < etamaxbound)
        {

            tc_all_track_dxy_mtv[0].push_back(dxy);
            if (tc_trackmatch_mtv) tc_matched_track_dxy_mtv[0].push_back(dxy);

            if (abs(eta) < 0.4)
            {
                tc_all_track_dxy_mtv[1].push_back(dxy);
                if (tc_trackmatch_mtv) tc_matched_track_dxy_mtv[1].push_back(dxy);
            }
            if (abs(eta) > 0.4 and abs(eta) < 0.8)
            {
                tc_all_track_dxy_mtv[2].push_back(dxy);
                if (tc_trackmatch_mtv) tc_matched_track_dxy_mtv[2].push_back(dxy);
            }
        }

    }

}

