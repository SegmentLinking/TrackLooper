#include "StudyTrackletSelectionOnTruths.h"

StudyTrackletSelectionOnTruths::StudyTrackletSelectionOnTruths(const char* studyName, StudyTrackletSelectionOnTruths::StudyTrackletSelectionOnTruthsMode mode_)
{

    studyname = studyName;
    mode = mode_;
    switch (mode)
    {
        case kStudySelAll: modename = "all"; break;
        case kStudySelSpecific: modename = "specific"; break;
        default: modename = "UNDEFINED"; break;
    }

}

void StudyTrackletSelectionOnTruths::bookStudy()
{
    // Book Histograms
    for (int ii = 0; ii < 7; ++ii)
    {
        ana.histograms.addVecHistogram(TString::Format("tl_truth_%s_deltaBeta_ptbin%d", modename, ii), 180, -0.06, 0.06, [&, ii]() { return tl_deltaBeta_ptslice[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("tl_truth_%s_deltaBeta_postCut_ptbin%d", modename, ii), 180, -0.06, 0.06, [&, ii]() { return tl_deltaBeta_postCut_ptslice[ii]; } );
    }
    ana.histograms.addVecHistogram(TString::Format("tl_truth_%s_deltaBeta", modename), 180, -0.06, 0.06, [&]() { return tl_deltaBeta; } );
    ana.histograms.addVecHistogram(TString::Format("tl_truth_%s_deltaBeta_postCut", modename), 180, -0.06, 0.06, [&]() { return tl_deltaBeta_postCut; } );
    ana.histograms.addVecHistogram(TString::Format("tl_truth_%s_deltaBeta_dcut", modename), 180, -0.06, 0.06, [&]() { return tl_deltaBeta_dcut; } );
    ana.histograms.addVecHistogram(TString::Format("tl_truth_%s_betaOut", modename), 180, -0.06, 0.06, [&]() { return tl_betaOut; } );
    ana.histograms.addVecHistogram(TString::Format("tl_truth_%s_betaOut_wide", modename), 180, -0.6, 0.6, [&]() { return tl_betaOut; } );
    ana.histograms.addVecHistogram(TString::Format("tl_truth_%s_betaOut_dcut", modename), 180, -0.06, 0.06, [&]() { return tl_betaOut_dcut; } );
    ana.histograms.addVecHistogram(TString::Format("tl_truth_%s_betaOut_dcut_wide", modename), 180, -0.6, 0.6, [&]() { return tl_betaOut_dcut; } );
    ana.histograms.addVecHistogram(TString::Format("tl_truth_%s_betaOut_cutthresh", modename), 180, 0., 0.6, [&]() { return tl_betaOut_cutthresh; } );
    ana.histograms.addVecHistogram(TString::Format("tl_truth_%s_betaIn", modename), 180, -0.06, 0.06, [&]() { return tl_betaIn; } );
    ana.histograms.addVecHistogram(TString::Format("tl_truth_%s_betaIn_wide", modename), 180, -0.6, 0.6, [&]() { return tl_betaIn; } );
    ana.histograms.addVecHistogram(TString::Format("tl_truth_%s_betaIn_dcut", modename), 180, -0.06, 0.06, [&]() { return tl_betaIn_dcut; } );
    ana.histograms.addVecHistogram(TString::Format("tl_truth_%s_betaIn_dcut_wide", modename), 180, -0.6, 0.6, [&]() { return tl_betaIn_dcut; } );
    ana.histograms.addVecHistogram(TString::Format("tl_truth_%s_betaIn_cutthresh", modename), 180, 0., 0.6, [&]() { return tl_betaIn_cutthresh; } );
    ana.histograms.addVecHistogram(TString::Format("tl_truth_%s_cutflow", modename), 8, 0, 8, [&]() { return tl_cutflow; } );

    const int nlayers = NLAYERS;
    for (int ii = 0; ii < nlayers; ++ii)
    {
        for (int jj = 0; jj < 7; ++jj)
        {
            ana.histograms.addVecHistogram(TString::Format("tl_truth_%s_deltaBeta_ptbin%d_by_layer%d", modename, jj, ii), 180, -0.06, 0.06, [&, ii, jj]() { return tl_deltaBeta_ptslice_by_layer[ii][jj]; } );
            ana.histograms.addVecHistogram(TString::Format("tl_truth_%s_deltaBeta_postCut_ptbin%d_by_layer%d", modename, jj, ii), 180, -0.06, 0.06, [&, ii, jj]() { return tl_deltaBeta_postCut_ptslice_by_layer[ii][jj]; } );
        }
        ana.histograms.addVecHistogram(TString::Format("tl_truth_%s_deltaBeta_by_layer%d", modename, ii), 180, -0.06, 0.06, [&, ii]() { return tl_deltaBeta_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("tl_truth_%s_deltaBeta_postCut_by_layer%d", modename, ii), 180, -0.06, 0.06, [&, ii]() { return tl_deltaBeta_postCut_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("tl_truth_%s_deltaBeta_dcut_by_layer%d", modename, ii), 180, -0.06, 0.06, [&, ii]() { return tl_deltaBeta_dcut_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("tl_truth_%s_betaOut_by_layer%d", modename, ii), 180, -0.06, 0.06, [&, ii]() { return tl_betaOut_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("tl_truth_%s_betaOut_wide_by_layer%d", modename, ii), 180, -0.6, 0.6, [&, ii]() { return tl_betaOut_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("tl_truth_%s_betaOut_dcut_by_layer%d", modename, ii), 180, -0.06, 0.06, [&, ii]() { return tl_betaOut_dcut_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("tl_truth_%s_betaOut_dcut_wide_by_layer%d", modename, ii), 180, -0.6, 0.6, [&, ii]() { return tl_betaOut_dcut_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("tl_truth_%s_betaOut_cutthresh_by_layer%d", modename, ii), 180, 0., 0.6, [&, ii]() { return tl_betaOut_cutthresh_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("tl_truth_%s_betaIn_by_layer%d", modename, ii), 180, -0.06, 0.06, [&, ii]() { return tl_betaIn_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("tl_truth_%s_betaIn_wide_by_layer%d", modename, ii), 180, -0.6, 0.6, [&, ii]() { return tl_betaIn_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("tl_truth_%s_betaIn_dcut_by_layer%d", modename, ii), 180, -0.06, 0.06, [&, ii]() { return tl_betaIn_dcut_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("tl_truth_%s_betaIn_dcut_wide_by_layer%d", modename, ii), 180, -0.6, 0.6, [&, ii]() { return tl_betaIn_dcut_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("tl_truth_%s_betaIn_cutthresh_by_layer%d", modename, ii), 180, 0., 0.6, [&, ii]() { return tl_betaIn_cutthresh_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("tl_truth_%s_cutflow_by_layer%d", modename, ii), 8, 0, 8, [&, ii]() { return tl_cutflow_by_layer[ii]; } );
    }
}

void StudyTrackletSelectionOnTruths::doStudy(SDL::Event& event, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents)
{

    // Each do study is performed per event

    // The study assumes that the createTracklets has been run with "AllComb" algorithm
    // The DefaultAlgo will be run individually here

    // First clear all the output variables that will be used to fill the histograms for this event
    tl_deltaBeta.clear();
    tl_deltaBeta_postCut.clear();
    tl_deltaBeta_dcut.clear();
    tl_betaOut.clear();
    tl_betaOut_dcut.clear();
    tl_betaOut_cutthresh.clear();
    tl_betaIn.clear();
    tl_betaIn_dcut.clear();
    tl_betaIn_cutthresh.clear();
    tl_cutflow.clear();
    for (int ii = 0; ii < 7; ++ii)
    {
        tl_deltaBeta_ptslice[ii].clear();
        tl_deltaBeta_postCut_ptslice[ii].clear();
    }
    for (int ii = 0; ii < NLAYERS; ++ii)
    {
        tl_deltaBeta_postCut_by_layer[ii].clear();
        tl_deltaBeta_by_layer[ii].clear();
        tl_deltaBeta_dcut_by_layer[ii].clear();
        tl_betaOut_by_layer[ii].clear();
        tl_betaOut_dcut_by_layer[ii].clear();
        tl_betaOut_cutthresh_by_layer[ii].clear();
        tl_betaIn_by_layer[ii].clear();
        tl_betaIn_dcut_by_layer[ii].clear();
        tl_betaIn_cutthresh_by_layer[ii].clear();
        tl_cutflow_by_layer[ii].clear();
        for (int jj = 0; jj < 7; ++jj)
        {
            tl_deltaBeta_ptslice_by_layer[ii][jj].clear();
            tl_deltaBeta_postCut_ptslice_by_layer[ii][jj].clear();
        }
    }

    //***********************
    // Studying selections and cutflows for the truth events (i.e. probably just simulated muons)
    //***********************

    // Loop over track events
    for (auto& simtrkevent : simtrkevents)
    {

        // Unpack the tuple (sim_track_index, SDL::Event containing reco hits only matched to the given sim track)
        unsigned int& isimtrk = std::get<0>(simtrkevent);
        SDL::Event& trackevent = *(std::get<1>(simtrkevent));

        // Loop over tracklets in event
        for (auto& layerPtr : trackevent.getLayerPtrs())
        {

            // Parse the layer index later to be used for indexing
            int layer_idx = layerPtr->layerIdx() - 1;

            // This means no tracklets in this layer
            if (layerPtr->getTrackletPtrs().size() == 0)
            {
                continue;
            }

            // Assuming I have at least one tracklets from this track
            std::vector<SDL::Tracklet*> tls_of_interest;
            for (auto& tl : layerPtr->getTrackletPtrs())
            {
                const SDL::Module& innerSgInnerMDLowerHitModule = tl->innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule();
                const SDL::Module& outerSgInnerMDLowerHitModule = tl->outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule();
                const SDL::Module& innerSgOuterMDLowerHitModule = tl->innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule();
                const SDL::Module& outerSgOuterMDLowerHitModule = tl->outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule();
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
                const bool innerSgOuterLayerBarrelFlat = innerSgOuterMDLowerHitModule.subdet() == SDL::Module::Barrel and innerSgOuterMDLowerHitModule.side() == SDL::Module::Center;
                const bool outerSgOuterLayerBarrelFlat = outerSgOuterMDLowerHitModule.subdet() == SDL::Module::Barrel and outerSgOuterMDLowerHitModule.side() == SDL::Module::Center;
                const bool innerSgOuterLayerBarrelTilt = innerSgOuterMDLowerHitModule.subdet() == SDL::Module::Barrel and innerSgOuterMDLowerHitModule.side() != SDL::Module::Center;
                const bool outerSgOuterLayerBarrelTilt = outerSgOuterMDLowerHitModule.subdet() == SDL::Module::Barrel and outerSgOuterMDLowerHitModule.side() != SDL::Module::Center;
                const bool innerSgInnerLayerPS = innerSgInnerMDLowerHitModule.moduleType() == SDL::Module::PS;
                const bool innerSgOuterLayerPS = innerSgOuterMDLowerHitModule.moduleType() == SDL::Module::PS;
                const bool outerSgInnerLayerPS = outerSgInnerMDLowerHitModule.moduleType() == SDL::Module::PS;
                const bool outerSgOuterLayerPS = outerSgOuterMDLowerHitModule.moduleType() == SDL::Module::PS;
                const bool outerSgInnerLayer2S = outerSgInnerMDLowerHitModule.moduleType() == SDL::Module::TwoS;
                const bool outerSgOuterLayer2S = outerSgOuterMDLowerHitModule.moduleType() == SDL::Module::TwoS;

                // Depending on the mode, only include a subset of interested tracklets
                switch (mode)
                {
                    case kStudySelAll: /* do nothing */ break;
                    case kStudySelSpecific:
                                                        if (not (
                                                                    layer_idx == 0
                                                                    and innerSgInnerLayerIdx == 1
                                                                    and outerSgInnerLayerIdx == 3
                                                                    and innerSgInnerLayerBarrel
                                                                    and outerSgInnerLayerBarrel
                                                                    and innerSgOuterLayerBarrel
                                                                    and outerSgOuterLayerBarrel
                                                                    and innerSgInnerLayerBarrelFlat
                                                                    and outerSgInnerLayerBarrelFlat
                                                                    and innerSgOuterLayerBarrelFlat
                                                                    and outerSgOuterLayerBarrelFlat
                                                                ))
                                                            continue;
                                                        break;
                    default: /* skip everything should not be here anyways...*/ continue; break;
                }

                // If this tracklet passes the condition that it is of interest then, add to the list of tracklets of interest
                tls_of_interest.push_back(tl);

            }

            // If no tls of interest are found then skip
            if (tls_of_interest.size() == 0)
                continue;

            // // Ghost removing
            // std::vector<int> tls_of_interest_mask;
            // std::vector<SDL::Tracklet*> tls_of_interest_ghost_removed;

            // for (auto& tl : tls_of_interest)
            // {
            //     tls_of_interest_mask.push_back(1);
            // }

            // for (unsigned int ii = 0; ii < tls_of_interest.size(); ++ii)
            // {
            //     if (tls_of_interest_mask[ii])
            //     {
            //         for (unsigned int jj = ii + 1; jj < tls_of_interest.size(); ++jj)
            //         {

            //             if (tls_of_interest[ii]->isAnchorHitIdxMatched(*tls_of_interest[jj]))
            //             {
            //                 tls_of_interest_mask[jj] = 0;
            //             }

            //         }
            //     }
            // }

            // for (unsigned int ii = 0; ii < tls_of_interest.size(); ++ii)
            // {
            //     if (tls_of_interest_mask[ii])
            //     {
            //         tls_of_interest_ghost_removed.push_back(tls_of_interest[ii]);
            //     }
            // }

            // Evaluate which pt bin it is
            // iptbin = -1;

            // The tls_of_interest holds only the tl "candidate" that we think are of interest for the given study mode
            for (auto& tl : tls_of_interest)
            {

                tl->runTrackletAlgo(SDL::Default_TLAlgo);

                // Cutflow
                //------------------------
                tl_cutflow.push_back(0);
                tl_cutflow_by_layer[layer_idx].push_back(0);

                const int& passbit = tl->getPassBitsDefaultAlgo();

                for (unsigned int i = 0; i < SDL::Tracklet::TrackletSelection::nCut; ++i)
                {
                    if (passbit & (1 << i))
                    {
                        tl_cutflow.push_back(i + 1);
                        tl_cutflow_by_layer[layer_idx].push_back(i + 1);
                    }
                    else
                    {
                        break;
                    }
                }

                // DeltaBeta post cut
                //------------------------
                if (passbit & (1 << SDL::Tracklet::TrackletSelection::dBeta))
                {

                    const float deltaBeta = tl->getDeltaBeta();

                    tl_deltaBeta_postCut.push_back(deltaBeta);

                    tl_deltaBeta_postCut_by_layer[layer_idx].push_back(deltaBeta);
                }

                // DeltaBeta
                //------------------------
                if (passbit & (1 << SDL::Tracklet::TrackletSelection::dAlphaOut))
                {

                    const float deltaBeta = tl->getDeltaBeta();
                    const float deltaBetaCut = fabs(deltaBeta) - tl->getDeltaBetaCut();

                    tl_deltaBeta.push_back(deltaBeta);
                    tl_deltaBeta_dcut.push_back(deltaBetaCut);

                    tl_deltaBeta_by_layer[layer_idx].push_back(deltaBeta);
                    tl_deltaBeta_dcut_by_layer[layer_idx].push_back(deltaBetaCut);
                }

                // betaOut
                //------------------------
                if (passbit & (1 << SDL::Tracklet::TrackletSelection::dAlphaIn))
                {

                    const float betaOut = tl->getBetaOut();
                    const float betaOutCut = fabs(betaOut) - tl->getBetaOutCut();
                    const float betaOutCutThresh = tl->getBetaOutCut();

                    tl_betaOut.push_back(betaOut);
                    tl_betaOut_dcut.push_back(betaOutCut);
                    tl_betaOut_cutthresh.push_back(betaOutCutThresh);

                    tl_betaOut_by_layer[layer_idx].push_back(betaOut);
                    tl_betaOut_dcut_by_layer[layer_idx].push_back(betaOutCut);
                    tl_betaOut_cutthresh_by_layer[layer_idx].push_back(betaOutCutThresh);
                }

                // betaIn
                //------------------------
                if (passbit & (1 << SDL::Tracklet::TrackletSelection::slope))
                {

                    const float betaIn = tl->getBetaIn();
                    const float betaInCut = fabs(betaIn) - tl->getBetaInCut();
                    const float betaInCutThresh = tl->getBetaInCut();

                    tl_betaIn.push_back(betaIn);
                    tl_betaIn_dcut.push_back(betaInCut);
                    tl_betaIn_cutthresh.push_back(betaInCutThresh);

                    tl_betaIn_by_layer[layer_idx].push_back(betaIn);
                    tl_betaIn_dcut_by_layer[layer_idx].push_back(betaInCut);
                    tl_betaIn_cutthresh_by_layer[layer_idx].push_back(betaInCutThresh);
                }

            }

        }

    }

}
