#include "StudyTripletSelection.h"

StudyTripletSelection::StudyTripletSelection(const char* studyName, StudyTripletSelection::StudyTripletSelectionMode mode_)
{

    studyname = studyName;
    mode = mode_;
    switch (mode)
    {
        case kStudySelAll: modename = "all"; break;
        case kStudySelBB1BB2: modename = "bb1bb2"; break;
        case kStudySelBB2BB3: modename = "bb2bb3"; break;
        case kStudySelBB3BB4: modename = "bb3bb4"; break;
        case kStudySelBB4BB5: modename = "bb4bb5"; break;
        case kStudySelSpecific: modename = "specific"; break;
        default: modename = "UNDEFINED"; break;
    }

}

void StudyTripletSelection::bookStudy()
{
    // Book Histograms
    ana.histograms.addVecHistogram(TString::Format("tp_%s_deltaBeta"                   , modename) , 180 , -0.15 , 0.15 , [&]() { return tp_deltaBeta;          } );
    ana.histograms.addVecHistogram(TString::Format("tp_%s_deltaBeta_standard"          , modename) , 180 , -0.15 , 0.15 , [&]() { return tp_deltaBeta;          } );
    ana.histograms.addVecHistogram(TString::Format("tp_%s_deltaBeta_zoom"              , modename) , 180 , -0.06 , 0.06 , [&]() { return tp_deltaBeta;          } );
    ana.histograms.addVecHistogram(TString::Format("tp_%s_deltaBeta_maxzoom"           , modename) , 180 , -0.04 , 0.04 , [&]() { return tp_deltaBeta;          } );
    ana.histograms.addVecHistogram(TString::Format("tp_%s_deltaBeta_slava"             , modename) , 400 , -0.15 , 0.15 , [&]() { return tp_deltaBeta;          } );
    ana.histograms.addVecHistogram(TString::Format("tp_%s_deltaBeta_midpoint"          , modename) , 180 , -0.30 , 0.30 , [&]() { return tp_deltaBeta_midpoint; } );
    ana.histograms.addVecHistogram(TString::Format("tp_%s_deltaBeta_3rdCorr"           , modename) , 180 , -0.06 , 0.06 , [&]() { return tp_deltaBeta_3rdCorr;  } );
    ana.histograms.addVecHistogram(TString::Format("tp_%s_deltaBeta_4thCorr"           , modename) , 180 , -0.03 , 0.03 , [&]() { return tp_deltaBeta_4thCorr;  } );
    ana.histograms.addVecHistogram(TString::Format("tp_%s_deltaBeta_midpoint_standard" , modename) , 180 , -0.15 , 0.15 , [&]() { return tp_deltaBeta_midpoint; } );
    ana.histograms.addVecHistogram(TString::Format("tp_%s_deltaBeta_3rdCorr_standard"  , modename) , 180 , -0.15 , 0.15 , [&]() { return tp_deltaBeta_3rdCorr;  } );
    ana.histograms.addVecHistogram(TString::Format("tp_%s_deltaBeta_4thCorr_standard"  , modename) , 180 , -0.15 , 0.15 , [&]() { return tp_deltaBeta_4thCorr;  } );
    ana.histograms.addVecHistogram(TString::Format("tp_%s_deltaBeta_midpoint_zoom"     , modename) , 180 , -0.06 , 0.06 , [&]() { return tp_deltaBeta_midpoint; } );
    ana.histograms.addVecHistogram(TString::Format("tp_%s_deltaBeta_3rdCorr_zoom"      , modename) , 180 , -0.06 , 0.06 , [&]() { return tp_deltaBeta_3rdCorr;  } );
    ana.histograms.addVecHistogram(TString::Format("tp_%s_deltaBeta_4thCorr_zoom"      , modename) , 180 , -0.06 , 0.06 , [&]() { return tp_deltaBeta_4thCorr;  } );
    ana.histograms.addVecHistogram(TString::Format("tp_%s_deltaBeta_midpoint_maxzoom"  , modename) , 180 , -0.04 , 0.04 , [&]() { return tp_deltaBeta_midpoint; } );
    ana.histograms.addVecHistogram(TString::Format("tp_%s_deltaBeta_3rdCorr_maxzoom"   , modename) , 180 , -0.04 , 0.04 , [&]() { return tp_deltaBeta_3rdCorr;  } );
    ana.histograms.addVecHistogram(TString::Format("tp_%s_deltaBeta_4thCorr_maxzoom"   , modename) , 180 , -0.04 , 0.04 , [&]() { return tp_deltaBeta_4thCorr;  } );

}

void StudyTripletSelection::doStudy(SDL::Event& event, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents)
{

    // Each do study is performed per event

    // First clear all the output variables that will be used to fill the histograms for this event
    tp_deltaBeta.clear();
    tp_deltaBeta_midpoint.clear();
    tp_deltaBeta_3rdCorr.clear();
    tp_deltaBeta_4thCorr.clear();

    //***********************
    // Studying selections and cutflows for the recoed events
    //***********************

    // Loop over tracklets in event
    for (auto& layerPtr : event.getLayerPtrs())
    {

        // Parse the layer index later to be used for indexing
        int layer_idx = layerPtr->layerIdx() - 1;

        // This means no tracklets in this layer
        if (layerPtr->getTripletPtrs().size() == 0)
        {
            continue;
        }

        // Assuming I have at least one tracklets from this track
        std::vector<SDL::Triplet*> tps_of_interest;
        for (auto& tp : layerPtr->getTripletPtrs())
        {
            const SDL::Module& innerSgInnerMDLowerHitModule = tp->innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule();
            const SDL::Module& outerSgInnerMDLowerHitModule = tp->outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule();
            const SDL::Module& innerSgOuterMDLowerHitModule = tp->innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule();
            const SDL::Module& outerSgOuterMDLowerHitModule = tp->outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule();
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
                case kStudySelBB1BB2:
                                                                  if (not (
                                                                              innerSgInnerLayerIdx == 1
                                                                              and outerSgInnerLayerIdx == 2
                                                                              and innerSgInnerLayerBarrel
                                                                              and outerSgInnerLayerBarrel
                                                                              and innerSgOuterLayerBarrel
                                                                              and outerSgOuterLayerBarrel
                                                                              ))
                                                                      continue;
                                                                  break;
                case kStudySelBB2BB3:
                                                                  if (not (
                                                                              innerSgInnerLayerIdx == 2
                                                                              and outerSgInnerLayerIdx == 3
                                                                              and innerSgInnerLayerBarrel
                                                                              and outerSgInnerLayerBarrel
                                                                              and innerSgOuterLayerBarrel
                                                                              and outerSgOuterLayerBarrel
                                                                              ))
                                                                      continue;
                                                                  break;
                case kStudySelBB3BB4:
                                                                  if (not (
                                                                              innerSgInnerLayerIdx == 3
                                                                              and outerSgInnerLayerIdx == 4
                                                                              and innerSgInnerLayerBarrel
                                                                              and outerSgInnerLayerBarrel
                                                                              and innerSgOuterLayerBarrel
                                                                              and outerSgOuterLayerBarrel
                                                                              ))
                                                                      continue;
                                                                  break;
                case kStudySelBB4BB5:
                                                                  if (not (
                                                                              innerSgInnerLayerIdx == 4
                                                                              and outerSgInnerLayerIdx == 5
                                                                              and innerSgInnerLayerBarrel
                                                                              and outerSgInnerLayerBarrel
                                                                              and innerSgOuterLayerBarrel
                                                                              and outerSgOuterLayerBarrel
                                                                              ))
                                                                      continue;
                                                                  break;
                case kStudySelSpecific:
                                                                  if (not (
                                                                              layer_idx == 0
                                                                              and innerSgInnerLayerIdx == 1
                                                                              and outerSgInnerLayerIdx == 3
                                                                              and innerSgInnerLayerBarrel
                                                                              and outerSgInnerLayerBarrel
                                                                              and innerSgOuterLayerBarrel
                                                                              and outerSgOuterLayerBarrel
                                                                              ))
                                                                      continue;
                                                                  break;
                default: /* skip everything should not be here anyways...*/ continue; break;
            }

            // If this tracklet passes the condition that it is of interest then, add to the list of tracklets of interest
            tps_of_interest.push_back(tp);

        }

        // If no tps of interest are found then skip
        if (tps_of_interest.size() == 0)
            continue;

        // The tps_of_interest holds only the tp "candidate" that we think are of interest for the given study mode
        for (auto& tp : tps_of_interest)
        {

            // DeltaBeta
            //------------------------

            const float deltaBeta = tp->tlCand.getDeltaBeta();
            const float deltaBetaCut = fabs(deltaBeta) - tp->tlCand.getDeltaBetaCut();

            tp_deltaBeta.push_back(deltaBeta);
            tp_deltaBeta_midpoint.push_back(tp->tlCand.getRecoVar("dBeta_midPoint"));
            tp_deltaBeta_3rdCorr.push_back(tp->tlCand.getRecoVar("dBeta_3rd"));
            tp_deltaBeta_4thCorr.push_back(tp->tlCand.getRecoVar("dBeta_4th"));

        }

    }

}
