#include "StudyTrackletEfficiency.h"

StudyTrackletEfficiency::StudyTrackletEfficiency(const char* studyName, StudyTrackletEfficiency::StudyTrackletEfficiencyMode mode_, std::vector<float> ptbounds)
{

    studyname = studyName;
    mode = mode_;
    switch (mode)
    {
        case kStudyEffAll: modename = "all"; break;
        case kStudyEffBarrel1Barrel3: modename = "barrel1barrel3"; break;
        case kStudyEffBarrel1FlatBarrel3Flat: modename = "barrel1flatbarrel3flat"; break;
        case kStudyEffBarrel1TiltBarrel3Flat: modename = "barrel1tiltbarrel3flat"; break;
        case kStudyEffBarrel1TiltBarrel3Tilt: modename = "barrel1tiltbarrel3tilt"; break;
        case kStudyEffBarrel1TiltBarrel3TiltBarrel4: modename = "barrel1tiltbarrel3tiltbarrel4"; break;
        case kStudyEffBarrel1TiltBarrel3TiltEndcap1: modename = "barrel1tiltbarrel3tiltendcap1"; break;
        case kStudyEffBarrelBarrelBarrelBarrel: modename = "barrelbarrelbarrelbarrel"; break;
        case kStudyEffBarrelBarrelEndcapEndcap: modename = "barrelbarrelendcapendcap"; break;
        case kStudyEffBB1BB3: modename = "bb1bb3"; break;
        case kStudyEffBB2BB4: modename = "bb2bb4"; break;
        case kStudyEffBB3BB5: modename = "bb3bb5"; break;
        case kStudyEffSpecific: modename = "specific"; break;
        default: modename = "UNDEFINED"; break;
    }
    pt_boundaries = ptbounds;

}

void StudyTrackletEfficiency::bookStudy()
{
    // Book Histograms
    ana.histograms.addVecHistogram(TString::Format("tl_%s_matched_track_pt", modename), pt_boundaries, [&]() { return tl_matched_track_pt; } );
    ana.histograms.addVecHistogram(TString::Format("tl_%s_all_track_pt", modename), pt_boundaries, [&]() { return tl_all_track_pt; } );
    ana.histograms.addVecHistogram(TString::Format("tl_%s_matched_track_eta", modename), 50, -4, 4, [&]() { return tl_matched_track_eta; } );
    ana.histograms.addVecHistogram(TString::Format("tl_%s_all_track_eta", modename), 50, -4, 4, [&]() { return tl_all_track_eta; } );
    ana.histograms.addVecHistogram(TString::Format("tl_%s_matched_track_phi", modename), 200, -3.1416, 3.1416, [&]() { return tl_matched_track_phi; } );
    ana.histograms.addVecHistogram(TString::Format("tl_%s_all_track_phi", modename), 200, -3.1416, 3.1416, [&]() { return tl_all_track_phi; } );
    ana.histograms.addVecHistogram(TString::Format("tl_%s_matched_track_z", modename), 200, -300, 300, [&]() { return tl_matched_track_z; } );
    ana.histograms.addVecHistogram(TString::Format("tl_%s_all_track_z", modename), 200, -300, 300, [&]() { return tl_all_track_z; } );

    const int nlayers = NLAYERS;
    for (int ii = 0; ii < nlayers; ++ii)
    {
        ana.histograms.addVecHistogram(TString::Format("tl_%s_matched_track_pt_by_layer%d", modename, ii), pt_boundaries, [&, ii]() { return tl_matched_track_pt_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("tl_%s_all_track_pt_by_layer%d", modename, ii), pt_boundaries, [&, ii]() { return tl_all_track_pt_by_layer[ii]; } );
    }
    for (int ii = 0; ii < nlayers; ++ii)
    {
        ana.histograms.addVecHistogram(TString::Format("tl_%s_matched_track_eta_by_layer%d", modename, ii), 50, -4, 4, [&, ii]() { return tl_matched_track_eta_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("tl_%s_all_track_eta_by_layer%d", modename, ii), 50, -4, 4, [&, ii]() { return tl_all_track_eta_by_layer[ii]; } );
    }
    for (int ii = 0; ii < nlayers; ++ii)
    {
        ana.histograms.addVecHistogram(TString::Format("tl_%s_matched_track_phi_by_layer%d", modename, ii), 200, -3.1416, 3.1416, [&, ii]() { return tl_matched_track_phi_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("tl_%s_all_track_phi_by_layer%d", modename, ii), 200, -3.1416, 3.1416, [&, ii]() { return tl_all_track_phi_by_layer[ii]; } );
    }
    for (int ii = 0; ii < nlayers; ++ii)
    {
        ana.histograms.addVecHistogram(TString::Format("tl_%s_matched_track_wrapphi_by_layer%d", modename, ii), 200, -3.1416, 3.1416, [&, ii]() { return tl_matched_track_wrapphi_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("tl_%s_all_track_wrapphi_by_layer%d", modename, ii), 200, -3.1416, 3.1416, [&, ii]() { return tl_all_track_wrapphi_by_layer[ii]; } );
    }
    for (int ii = 0; ii < nlayers; ++ii)
    {
        ana.histograms.addVecHistogram(TString::Format("tl_%s_matched_track_z_by_layer%d", modename, ii), 200, -300, 300, [&, ii]() { return tl_matched_track_z_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("tl_%s_all_track_z_by_layer%d", modename, ii), 200, -300, 300, [&, ii]() { return tl_all_track_z_by_layer[ii]; } );
    }
}

void StudyTrackletEfficiency::doStudy(SDL::Event& event, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents)
{
    // Each do study is performed per event

    // First clear all the output variables that will be used to fill the histograms for this event
    tl_matched_track_pt.clear();
    tl_all_track_pt.clear();
    tl_matched_track_eta.clear();
    tl_all_track_eta.clear();
    tl_matched_track_phi.clear();
    tl_all_track_phi.clear();
    tl_matched_track_z.clear();
    tl_all_track_z.clear();
    for (int ii = 0; ii < NLAYERS; ++ii)
    {
        tl_matched_track_pt_by_layer[ii].clear();
        tl_all_track_pt_by_layer[ii].clear();
        tl_matched_track_eta_by_layer[ii].clear();
        tl_all_track_eta_by_layer[ii].clear();
        tl_matched_track_phi_by_layer[ii].clear();
        tl_all_track_phi_by_layer[ii].clear();
        tl_matched_track_wrapphi_by_layer[ii].clear();
        tl_all_track_wrapphi_by_layer[ii].clear();
        tl_matched_track_z_by_layer[ii].clear();
        tl_all_track_z_by_layer[ii].clear();
    }

    //***********************
    // Efficiency calculation
    //***********************

    // Loop over track events
    for (auto& simtrkevent : simtrkevents)
    {

        // Unpack the tuple (sim_track_index, SDL::Event containing reco hits only matched to the given sim track)
        unsigned int& isimtrk = std::get<0>(simtrkevent);
        SDL::Event& trackevent = *(std::get<1>(simtrkevent));

        bool trackmatch = false;

        // Loop over the layer that contains tracklets for this track
        for (auto& layerPtr_Track : trackevent.getLayerPtrs())
        {

            // Parse the layer index later to be used for indexing
            int layer_idx = layerPtr_Track->layerIdx() - 1;

            // Parse pt and eta of this track
            float rawpt = trk.sim_pt()[isimtrk];
            float pt = std::min((double) trk.sim_pt()[isimtrk], 49.999);
            float eta = trk.sim_eta()[isimtrk];
            float phi = trk.sim_phi()[isimtrk];

            // This means no tracklets in this layer
            if (layerPtr_Track->getTrackletPtrs().size() == 0)
            {
                continue;
            }

            // Assuming I have at least one tracklets from this track
            std::vector<SDL::Tracklet*> tls_of_interest;
            for (auto& tl_Track : layerPtr_Track->getTrackletPtrs())
            {
                const SDL::Module& innerSgInnerMDLowerHitModule = tl_Track->innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule();
                const SDL::Module& outerSgInnerMDLowerHitModule = tl_Track->outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule();
                const SDL::Module& innerSgOuterMDLowerHitModule = tl_Track->innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule();
                const SDL::Module& outerSgOuterMDLowerHitModule = tl_Track->outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule();
                const int innerSgInnerLayerIdx = innerSgInnerMDLowerHitModule.layer();
                const int outerSgInnerLayerIdx = outerSgInnerMDLowerHitModule.layer();
                const bool innerSgInnerLayerBarrel = innerSgInnerMDLowerHitModule.subdet() == SDL::Module::Barrel;
                const bool outerSgInnerLayerBarrel = outerSgInnerMDLowerHitModule.subdet() == SDL::Module::Barrel;
                const bool innerSgOuterLayerBarrel = innerSgOuterMDLowerHitModule.subdet() == SDL::Module::Barrel;
                const bool outerSgOuterLayerBarrel = outerSgOuterMDLowerHitModule.subdet() == SDL::Module::Barrel;
                const bool outerSgInnerLayerEndcap = outerSgInnerMDLowerHitModule.subdet() == SDL::Module::Endcap;
                const bool innerSgInnerLayerEndcap = innerSgInnerMDLowerHitModule.subdet() == SDL::Module::Endcap;
                const bool outerSgOuterLayerEndcap = outerSgOuterMDLowerHitModule.subdet() == SDL::Module::Endcap;
                const bool innerSgOuterLayerEndcap = innerSgOuterMDLowerHitModule.subdet() == SDL::Module::Endcap;
                const bool innerSgInnerLayerBarrelFlat = innerSgInnerMDLowerHitModule.subdet() == SDL::Module::Barrel and innerSgInnerMDLowerHitModule.side() == SDL::Module::Center;
                const bool outerSgInnerLayerBarrelFlat = outerSgInnerMDLowerHitModule.subdet() == SDL::Module::Barrel and outerSgInnerMDLowerHitModule.side() == SDL::Module::Center;
                const bool innerSgInnerLayerBarrelTilt = innerSgInnerMDLowerHitModule.subdet() == SDL::Module::Barrel and innerSgInnerMDLowerHitModule.side() != SDL::Module::Center;
                const bool outerSgInnerLayerBarrelTilt = outerSgInnerMDLowerHitModule.subdet() == SDL::Module::Barrel and outerSgInnerMDLowerHitModule.side() != SDL::Module::Center;
                const bool innerSgInnerLayerPS = innerSgInnerMDLowerHitModule.moduleType() == SDL::Module::PS;
                const bool innerSgOuterLayerPS = innerSgOuterMDLowerHitModule.moduleType() == SDL::Module::PS;
                const bool outerSgInnerLayerPS = outerSgInnerMDLowerHitModule.moduleType() == SDL::Module::PS;
                const bool outerSgOuterLayerPS = outerSgOuterMDLowerHitModule.moduleType() == SDL::Module::PS;
                const bool outerSgInnerLayer2S = outerSgInnerMDLowerHitModule.moduleType() == SDL::Module::TwoS;
                const bool outerSgOuterLayer2S = outerSgOuterMDLowerHitModule.moduleType() == SDL::Module::TwoS;

                // Depending on the mode, only include a subset of interested tracklets
                switch (mode)
                {
                    case kStudyEffAll: /* do nothing */ break;
                    case kStudyEffBarrel1Barrel3:  if (not (innerSgInnerLayerIdx == 1 and outerSgInnerLayerIdx == 3 and innerSgInnerLayerBarrel and outerSgInnerLayerBarrel)) continue; break;
                    case kStudyEffBarrel1FlatBarrel3Flat:  if (not (innerSgInnerLayerIdx == 1 and outerSgInnerLayerIdx == 3 and innerSgInnerLayerBarrelFlat and outerSgInnerLayerBarrelFlat)) continue; break;
                    case kStudyEffBarrel1TiltBarrel3Flat:  if (not (innerSgInnerLayerIdx == 1 and outerSgInnerLayerIdx == 3 and innerSgInnerLayerBarrelTilt and outerSgInnerLayerBarrelFlat)) continue; break;
                    case kStudyEffBarrel1TiltBarrel3Tilt:  if (not (innerSgInnerLayerIdx == 1 and outerSgInnerLayerIdx == 3 and innerSgInnerLayerBarrelTilt and outerSgInnerLayerBarrelTilt)) continue; break;
                    case kStudyEffBarrel1TiltBarrel3TiltBarrel4:  if (not (innerSgInnerLayerIdx == 1 and outerSgInnerLayerIdx == 3 and innerSgInnerLayerBarrelTilt and outerSgInnerLayerBarrelTilt and outerSgOuterLayerBarrel)) continue; break;
                    case kStudyEffBarrel1TiltBarrel3TiltEndcap1:  if (not (innerSgInnerLayerIdx == 1 and outerSgInnerLayerIdx == 3 and innerSgInnerLayerBarrelTilt and outerSgInnerLayerBarrelTilt and outerSgOuterLayerEndcap)) continue; break;
                    case kStudyEffBarrelBarrelBarrelBarrel:
                        if (not (innerSgInnerLayerBarrel
                             and outerSgInnerLayerBarrel
                             and innerSgOuterLayerBarrel
                             and outerSgOuterLayerBarrel))
                            continue;
                        break;
                    case kStudyEffBarrelBarrelEndcapEndcap:
                        if (not (innerSgInnerLayerBarrel
                             and outerSgInnerLayerEndcap
                             and innerSgOuterLayerBarrel
                             and outerSgOuterLayerEndcap
                             and innerSgInnerLayerPS
                             and innerSgOuterLayerPS
                             and outerSgInnerLayer2S
                             and outerSgOuterLayer2S
                             ))
                            continue;
                        break;
                    case kStudyEffBB1BB3:
                        if (not (
                                    innerSgInnerLayerIdx == 1
                                    and outerSgInnerLayerIdx == 3
                                    and innerSgInnerLayerBarrel
                                    and outerSgInnerLayerBarrel
                                    and innerSgOuterLayerBarrel
                                    and outerSgOuterLayerBarrel
                                ))
                            continue;
                        break;
                    case kStudyEffBB2BB4:
                        if (not (
                                    innerSgInnerLayerIdx == 2
                                    and outerSgInnerLayerIdx == 4
                                    and innerSgInnerLayerBarrel
                                    and outerSgInnerLayerBarrel
                                    and innerSgOuterLayerBarrel
                                    and outerSgOuterLayerBarrel
                                ))
                            continue;
                        break;
                    case kStudyEffBB3BB5:
                        if (not (
                                    innerSgInnerLayerIdx == 3
                                    and outerSgInnerLayerIdx == 5
                                    and innerSgInnerLayerBarrel
                                    and outerSgInnerLayerBarrel
                                    and innerSgOuterLayerBarrel
                                    and outerSgOuterLayerBarrel
                                ))
                            continue;
                        break;
                    case kStudyEffSpecific:
                        if (not (
                                 layer_idx == 0
                             and innerSgInnerLayerIdx == 1
                             and outerSgInnerLayerIdx == 3
                             and innerSgInnerLayerBarrel
                             and outerSgInnerLayerBarrel
                             and innerSgOuterLayerBarrel
                             and outerSgOuterLayerBarrel))
                            continue;
                        break;
                    default: /* skip everything should not be here anyways...*/ continue; break;
                }

                // If this tracklet passes the condition that it is of interest then, add to the list of tracklets of interest
                tls_of_interest.push_back(tl_Track);

            }

            // If no tls of interest are found then skip
            if (tls_of_interest.size() == 0)
                continue;

            // Boolean to test whether for this module that a track passed through, whether it found a matched mini-doublet
            bool match = false;

            // Loop over the tl "candidate" from the module that a sim-track passed through and left at least one mini-doublet in each module
            // The tls_of_interest holds only the tl "candidate" that we think are of interest for the given study mode
            float z; // z position of lowest hit in the tracklet candidate
            for (auto& tl_Track : tls_of_interest)
            {

                const SDL::Module& innerSgInnerMDLowerHitModule = tl_Track->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule();
                const SDL::Module& outerSgInnerMDLowerHitModule = tl_Track->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule();

                // get z position of one of the tl
                z = tl_Track->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->z();

                // Loop over the tl reconstructed from with proper SDL algorithm and if the index of the tracklets match (i.e. if the 8 hits match)
                // Then we have found at least one tracklets associated to this track's reco-hits in this module
                // Therefore flag the match boolean
                SDL::Layer::SubDet innerLayerSubDet = innerSgInnerMDLowerHitModule.subdet() == SDL::Module::Barrel ? SDL::Layer::Barrel : SDL::Layer::Endcap;
                for (auto& tl : event.getLayer(innerSgInnerMDLowerHitModule.layer(), innerLayerSubDet).getTrackletPtrs())
                {
                    if (tl_Track->isIdxMatched(*tl))
                    {
                        match = true;
                    }
                }
            }

            if (match)
                trackmatch = true;

        }

        float pt = std::min((double) trk.sim_pt()[isimtrk], 49.999);
        float eta = trk.sim_eta()[isimtrk];

        // Denominator for all layers pt efficiency
        tl_all_track_pt.push_back(pt);

        // Denominator for all layers eta efficiency (notice the 1 GeV cut)
        if (trk.sim_pt()[isimtrk] > 1.)
            tl_all_track_eta.push_back(eta);

        // Numerators
        if (trackmatch)
        {

            // Numerators for matched all layers pt efficiency
            tl_matched_track_pt.push_back(pt);

            // Numeratosr for matched all layers eta efficiency (notice the 1 GeV cut)
            if (trk.sim_pt()[isimtrk] > 1.)
                tl_matched_track_eta.push_back(eta);

        }

    }

}

void StudyTrackletEfficiency::doStudy_v1(SDL::Event& event, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents)
{
    // Each do study is performed per event

    // First clear all the output variables that will be used to fill the histograms for this event
    tl_matched_track_pt.clear();
    tl_all_track_pt.clear();
    tl_matched_track_eta.clear();
    tl_all_track_eta.clear();
    tl_matched_track_phi.clear();
    tl_all_track_phi.clear();
    tl_matched_track_z.clear();
    tl_all_track_z.clear();
    for (int ii = 0; ii < NLAYERS; ++ii)
    {
        tl_matched_track_pt_by_layer[ii].clear();
        tl_all_track_pt_by_layer[ii].clear();
        tl_matched_track_eta_by_layer[ii].clear();
        tl_all_track_eta_by_layer[ii].clear();
        tl_matched_track_phi_by_layer[ii].clear();
        tl_all_track_phi_by_layer[ii].clear();
        tl_matched_track_wrapphi_by_layer[ii].clear();
        tl_all_track_wrapphi_by_layer[ii].clear();
        tl_matched_track_z_by_layer[ii].clear();
        tl_all_track_z_by_layer[ii].clear();
    }

    //***********************
    // Efficiency calculation
    //***********************

    // Loop over track events
    for (auto& simtrkevent : simtrkevents)
    {

        // Unpack the tuple (sim_track_index, SDL::Event containing reco hits only matched to the given sim track)
        unsigned int& isimtrk = std::get<0>(simtrkevent);
        SDL::Event& trackevent = *(std::get<1>(simtrkevent));

        // Loop over the layer that contains tracklets for this track
        for (auto& layerPtr_Track : trackevent.getLayerPtrs())
        {

            // Parse the layer index later to be used for indexing
            int layer_idx = layerPtr_Track->layerIdx() - 1;

            // Parse pt and eta of this track
            float rawpt = trk.sim_pt()[isimtrk];
            float pt = std::min((double) trk.sim_pt()[isimtrk], 49.999);
            float eta = trk.sim_eta()[isimtrk];
            float phi = trk.sim_phi()[isimtrk];

            // This means no tracklets in this layer
            if (layerPtr_Track->getTrackletPtrs().size() == 0)
            {
                continue;
            }

            // Assuming I have at least one tracklets from this track
            std::vector<SDL::Tracklet*> tls_of_interest;
            for (auto& tl_Track : layerPtr_Track->getTrackletPtrs())
            {
                const SDL::Module& innerSgInnerMDLowerHitModule = tl_Track->innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule();
                const SDL::Module& outerSgInnerMDLowerHitModule = tl_Track->outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule();
                const SDL::Module& innerSgOuterMDLowerHitModule = tl_Track->innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule();
                const SDL::Module& outerSgOuterMDLowerHitModule = tl_Track->outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule();
                const int innerSgInnerLayerIdx = innerSgInnerMDLowerHitModule.layer();
                const int outerSgInnerLayerIdx = outerSgInnerMDLowerHitModule.layer();
                const bool innerSgInnerLayerBarrel = innerSgInnerMDLowerHitModule.subdet() == SDL::Module::Barrel;
                const bool outerSgInnerLayerBarrel = outerSgInnerMDLowerHitModule.subdet() == SDL::Module::Barrel;
                const bool innerSgOuterLayerBarrel = innerSgOuterMDLowerHitModule.subdet() == SDL::Module::Barrel;
                const bool outerSgOuterLayerBarrel = outerSgOuterMDLowerHitModule.subdet() == SDL::Module::Barrel;
                const bool outerSgInnerLayerEndcap = outerSgInnerMDLowerHitModule.subdet() == SDL::Module::Endcap;
                const bool innerSgInnerLayerEndcap = innerSgInnerMDLowerHitModule.subdet() == SDL::Module::Endcap;
                const bool outerSgOuterLayerEndcap = outerSgOuterMDLowerHitModule.subdet() == SDL::Module::Endcap;
                const bool innerSgOuterLayerEndcap = innerSgOuterMDLowerHitModule.subdet() == SDL::Module::Endcap;
                const bool innerSgInnerLayerBarrelFlat = innerSgInnerMDLowerHitModule.subdet() == SDL::Module::Barrel and innerSgInnerMDLowerHitModule.side() == SDL::Module::Center;
                const bool outerSgInnerLayerBarrelFlat = outerSgInnerMDLowerHitModule.subdet() == SDL::Module::Barrel and outerSgInnerMDLowerHitModule.side() == SDL::Module::Center;
                const bool innerSgInnerLayerBarrelTilt = innerSgInnerMDLowerHitModule.subdet() == SDL::Module::Barrel and innerSgInnerMDLowerHitModule.side() != SDL::Module::Center;
                const bool outerSgInnerLayerBarrelTilt = outerSgInnerMDLowerHitModule.subdet() == SDL::Module::Barrel and outerSgInnerMDLowerHitModule.side() != SDL::Module::Center;
                const bool innerSgInnerLayerPS = innerSgInnerMDLowerHitModule.moduleType() == SDL::Module::PS;
                const bool innerSgOuterLayerPS = innerSgOuterMDLowerHitModule.moduleType() == SDL::Module::PS;
                const bool outerSgInnerLayerPS = outerSgInnerMDLowerHitModule.moduleType() == SDL::Module::PS;
                const bool outerSgOuterLayerPS = outerSgOuterMDLowerHitModule.moduleType() == SDL::Module::PS;
                const bool outerSgInnerLayer2S = outerSgInnerMDLowerHitModule.moduleType() == SDL::Module::TwoS;
                const bool outerSgOuterLayer2S = outerSgOuterMDLowerHitModule.moduleType() == SDL::Module::TwoS;

                // Depending on the mode, only include a subset of interested tracklets
                switch (mode)
                {
                    case kStudyEffAll: /* do nothing */ break;
                    case kStudyEffBarrel1Barrel3:  if (not (innerSgInnerLayerIdx == 1 and outerSgInnerLayerIdx == 3 and innerSgInnerLayerBarrel and outerSgInnerLayerBarrel)) continue; break;
                    case kStudyEffBarrel1FlatBarrel3Flat:  if (not (innerSgInnerLayerIdx == 1 and outerSgInnerLayerIdx == 3 and innerSgInnerLayerBarrelFlat and outerSgInnerLayerBarrelFlat)) continue; break;
                    case kStudyEffBarrel1TiltBarrel3Flat:  if (not (innerSgInnerLayerIdx == 1 and outerSgInnerLayerIdx == 3 and innerSgInnerLayerBarrelTilt and outerSgInnerLayerBarrelFlat)) continue; break;
                    case kStudyEffBarrel1TiltBarrel3Tilt:  if (not (innerSgInnerLayerIdx == 1 and outerSgInnerLayerIdx == 3 and innerSgInnerLayerBarrelTilt and outerSgInnerLayerBarrelTilt)) continue; break;
                    case kStudyEffBarrel1TiltBarrel3TiltBarrel4:  if (not (innerSgInnerLayerIdx == 1 and outerSgInnerLayerIdx == 3 and innerSgInnerLayerBarrelTilt and outerSgInnerLayerBarrelTilt and outerSgOuterLayerBarrel)) continue; break;
                    case kStudyEffBarrel1TiltBarrel3TiltEndcap1:  if (not (innerSgInnerLayerIdx == 1 and outerSgInnerLayerIdx == 3 and innerSgInnerLayerBarrelTilt and outerSgInnerLayerBarrelTilt and outerSgOuterLayerEndcap)) continue; break;
                    case kStudyEffBarrelBarrelBarrelBarrel:
                        if (not (innerSgInnerLayerBarrel
                             and outerSgInnerLayerBarrel
                             and innerSgOuterLayerBarrel
                             and outerSgOuterLayerBarrel))
                            continue;
                        break;
                    case kStudyEffBarrelBarrelEndcapEndcap:
                        if (not (innerSgInnerLayerBarrel
                             and outerSgInnerLayerEndcap
                             and innerSgOuterLayerBarrel
                             and outerSgOuterLayerEndcap
                             and innerSgInnerLayerPS
                             and innerSgOuterLayerPS
                             and outerSgInnerLayer2S
                             and outerSgOuterLayer2S
                             ))
                            continue;
                        break;
                    case kStudyEffBB1BB3:
                        if (not (
                                    innerSgInnerLayerIdx == 1
                                    and outerSgInnerLayerIdx == 3
                                    and innerSgInnerLayerBarrel
                                    and outerSgInnerLayerBarrel
                                    and innerSgOuterLayerBarrel
                                    and outerSgOuterLayerBarrel
                                ))
                            continue;
                        break;
                    case kStudyEffBB2BB4:
                        if (not (
                                    innerSgInnerLayerIdx == 2
                                    and outerSgInnerLayerIdx == 4
                                    and innerSgInnerLayerBarrel
                                    and outerSgInnerLayerBarrel
                                    and innerSgOuterLayerBarrel
                                    and outerSgOuterLayerBarrel
                                ))
                            continue;
                        break;
                    case kStudyEffBB3BB5:
                        if (not (
                                    innerSgInnerLayerIdx == 3
                                    and outerSgInnerLayerIdx == 5
                                    and innerSgInnerLayerBarrel
                                    and outerSgInnerLayerBarrel
                                    and innerSgOuterLayerBarrel
                                    and outerSgOuterLayerBarrel
                                ))
                            continue;
                        break;
                    case kStudyEffSpecific:
                        if (not (
                                 layer_idx == 0
                             and innerSgInnerLayerIdx == 1
                             and outerSgInnerLayerIdx == 3
                             and innerSgInnerLayerBarrel
                             and outerSgInnerLayerBarrel
                             and innerSgOuterLayerBarrel
                             and outerSgOuterLayerBarrel))
                            continue;
                        break;
                    default: /* skip everything should not be here anyways...*/ continue; break;
                }

                // If this tracklet passes the condition that it is of interest then, add to the list of tracklets of interest
                tls_of_interest.push_back(tl_Track);

            }

            // If no tls of interest are found then skip
            if (tls_of_interest.size() == 0)
                continue;

            // Boolean to test whether for this module that a track passed through, whether it found a matched mini-doublet
            bool match = false;

            // Loop over the tl "candidate" from the module that a sim-track passed through and left at least one mini-doublet in each module
            // The tls_of_interest holds only the tl "candidate" that we think are of interest for the given study mode
            float z; // z position of lowest hit in the tracklet candidate
            for (auto& tl_Track : tls_of_interest)
            {

                const SDL::Module& innerSgInnerMDLowerHitModule = tl_Track->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule();
                const SDL::Module& outerSgInnerMDLowerHitModule = tl_Track->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule();

                // get z position of one of the tl
                z = tl_Track->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->z();

                // Loop over the tl reconstructed from with proper SDL algorithm and if the index of the tracklets match (i.e. if the 8 hits match)
                // Then we have found at least one tracklets associated to this track's reco-hits in this module
                // Therefore flag the match boolean
                SDL::Layer::SubDet innerLayerSubDet = innerSgInnerMDLowerHitModule.subdet() == SDL::Module::Barrel ? SDL::Layer::Barrel : SDL::Layer::Endcap;
                for (auto& tl : event.getLayer(innerSgInnerMDLowerHitModule.layer(), innerLayerSubDet).getTrackletPtrs())
                {
                    if (tl_Track->isIdxMatched(*tl))
                    {
                        match = true;
                    }
                }
            }

            // At this stage, we have either found a tracklets in this module either matched to the track or not.

            // Denominator for all layers pt efficiency
            tl_all_track_pt.push_back(pt);

            // Denominator for all layers eta efficiency (notice the 1 GeV cut)
            if (trk.sim_pt()[isimtrk] > 1.)
                tl_all_track_eta.push_back(eta);

            // Denominator for all layers phi efficiency (notice the 1 GeV cut)
            if (trk.sim_pt()[isimtrk] > 1.)
                tl_all_track_phi.push_back(phi);

            // Denominator for all layers z efficiency (notice the 1 GeV cut)
            if (trk.sim_pt()[isimtrk] > 1.)
                tl_all_track_z.push_back(z);

            // Numerators
            if (match)
            {

                // Numerators for matched all layers pt efficiency
                tl_matched_track_pt.push_back(pt);

                // Numeratosr for matched all layers eta efficiency (notice the 1 GeV cut)
                if (trk.sim_pt()[isimtrk] > 1.)
                    tl_matched_track_eta.push_back(eta);

                // Numeratosr for matched all layers phi efficiency (notice the 1 GeV cut)
                if (trk.sim_pt()[isimtrk] > 1.)
                    tl_matched_track_phi.push_back(phi);

                // Numeratosr for matched all layers z efficiency (notice the 1 GeV cut)
                if (trk.sim_pt()[isimtrk] > 1.)
                    tl_matched_track_z.push_back(z);

                // if (mode == kStudyEffBarrel1FlatBarrel3Flat and pt > 1.)
                // {
                //     for (auto& tl_Track : tls_of_interest)
                //     {
                //         tl_Track->runTrackletAlgo(SDL::Default_TLAlgo, SDL::Log_Debug3);
                //     }
                // }
            }
            // Failed tracks for all layers
            else
            {
                // Doing nothing for now ...
                // if (mode == kStudyEffSpecific and pt > 1. and layer_idx == 1)
                if (mode == kStudyEffBarrelBarrelBarrelBarrel and pt > 1. and layer_idx == 1)
                {
                    SDL::cout << "****************************************************************" << std::endl;
                    SDL::cout << "Failed Track : " << rawpt << std::endl;
                    SDL::cout << "****************************************************************" << std::endl;
                    SDL::cout << std::endl;
                    for (auto& tl_Track : tls_of_interest)
                    {
                        tl_Track->runTrackletAlgo(SDL::Default_TLAlgo, SDL::Log_Debug3);
                        SDL::cout << tl_Track << std::endl;
                    }
                }
            }

            // Denominator for specific layers pt efficiency
            tl_all_track_pt_by_layer[layer_idx].push_back(pt);

            // Denominator for specific layers eta efficiency (notice the 1 GeV cut)
            if (trk.sim_pt()[isimtrk] > 1.)
                tl_all_track_eta_by_layer[layer_idx].push_back(eta);

            // Denominator for specific layers phi efficiency (notice the 1 GeV cut)
            if (trk.sim_pt()[isimtrk] > 1.)
                tl_all_track_phi_by_layer[layer_idx].push_back(phi);

            // Denominator for specific layers z efficiency (notice the 1 GeV cut)
            if (trk.sim_pt()[isimtrk] > 1.)
                tl_all_track_z_by_layer[layer_idx].push_back(z);

            // Denominator for specific layers wrapphi efficiency (notice the 1 GeV cut)
            float wrapphi = 0;
            if (layer_idx == 0)
                wrapphi = fmod(fabs(phi), 2*TMath::Pi() / 18);
            else if (layer_idx == 1)
                wrapphi = fmod(fabs(phi), 2*TMath::Pi() / 26);
            else
                wrapphi = fmod(fabs(phi), 2*TMath::Pi() / 36);
            if (trk.sim_pt()[isimtrk] > 5.)
                tl_all_track_wrapphi_by_layer[layer_idx].push_back(wrapphi);

            // Numerators
            if (match)
            {
                // Numerators for matched specific layer pt effciency
                tl_matched_track_pt_by_layer[layer_idx].push_back(pt);

                // Numerators for matched specific layer eta effciency (notice the 1 GeV cut)
                if (trk.sim_pt()[isimtrk] > 1.)
                    tl_matched_track_eta_by_layer[layer_idx].push_back(eta);

                // Numerators for matched specific layer phi effciency (notice the 1 GeV cut)
                if (trk.sim_pt()[isimtrk] > 1.)
                    tl_matched_track_phi_by_layer[layer_idx].push_back(phi);

                // Numerators for matched specific layer wrapphi effciency (notice the 1 GeV cut)
                if (trk.sim_pt()[isimtrk] > 5.)
                    tl_matched_track_wrapphi_by_layer[layer_idx].push_back(wrapphi);

                // Numerators for matched specific layer z effciency (notice the 1 GeV cut)
                if (trk.sim_pt()[isimtrk] > 1.)
                    tl_matched_track_z_by_layer[layer_idx].push_back(z);

            }
            // Failed tracks for specific layers
            else
            {
            }

        }

    }

}
