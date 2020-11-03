#include "StudySegmentEfficiency.h"

StudySegmentEfficiency::StudySegmentEfficiency(const char* studyName, StudySegmentEfficiency::StudySegmentEfficiencyMode mode_, std::vector<float> ptbounds)
{

    studyname = studyName;
    mode = mode_;
    switch (mode)
    {
        case kStudyEffAll: modename = "all"; break;
        case kStudyEffBarrelBarrel: modename = "barrelbarrel"; break;
        case kStudyEffBarrelFlatBarrel: modename = "barrelflatbarrel"; break;
        case kStudyEffBarrelTiltBarrel: modename = "barreltiltbarrel"; break;
        case kStudyEffBarrelFlatBarrelFlat: modename = "barrelflatbarrelflat"; break;
        case kStudyEffBarrelFlatBarrelTilt: modename = "barrelflatbarreltilt"; break;
        case kStudyEffBarrelTiltBarrelFlat: modename = "barreltiltbarrelflat"; break;
        case kStudyEffBarrelTiltBarrelTilt: modename = "barreltiltbarreltilt"; break;
        case kStudyEffBarrelEndcap: modename = "barrelendcap"; break;
        case kStudyEffBarrelTiltEndcap: modename = "barreltiltendcap"; break;
        case kStudyEffBarrel: modename = "barrel"; break;
        case kStudyEffEndcap: modename = "endcap"; break;
        case kStudyEffEndcapPS: modename = "endcapPS"; break;
        case kStudyEffEndcapPSPS: modename = "endcapPSPS"; break;
        case kStudyEffEndcapPS2S: modename = "endcapPS2S"; break;
        case kStudyEffEndcap2S: modename = "endcap2S"; break;
        default: modename = "UNDEFINED"; break;
    }
    pt_boundaries = ptbounds;

}

void StudySegmentEfficiency::bookStudy()
{
    // Book Histograms
    ana.histograms.addVecHistogram(TString::Format("sg_%s_matched_track_pt", modename), pt_boundaries, [&]() { return sg_matched_track_pt; } );
    ana.histograms.addVecHistogram(TString::Format("sg_%s_all_track_pt", modename), pt_boundaries, [&]() { return sg_all_track_pt; } );
    ana.histograms.addVecHistogram(TString::Format("sg_%s_matched_track_eta", modename), 50, -4, 4, [&]() { return sg_matched_track_eta; } );
    ana.histograms.addVecHistogram(TString::Format("sg_%s_all_track_eta", modename), 50, -4, 4, [&]() { return sg_all_track_eta; } );
    ana.histograms.addVecHistogram(TString::Format("sg_%s_matched_track_phi", modename), 200, -3.1416, 3.1416, [&]() { return sg_matched_track_phi; } );
    ana.histograms.addVecHistogram(TString::Format("sg_%s_all_track_phi", modename), 200, -3.1416, 3.1416, [&]() { return sg_all_track_phi; } );
    ana.histograms.addVecHistogram(TString::Format("sg_%s_matched_track_z", modename), 200, -300, 300, [&]() { return sg_matched_track_z; } );
    ana.histograms.addVecHistogram(TString::Format("sg_%s_all_track_z", modename), 200, -300, 300, [&]() { return sg_all_track_z; } );
    ana.histograms.addVecHistogram(TString::Format("sg_%s_matched_track_ring", modename), 18, 0, 18, [&]() { return sg_matched_track_ring; } );
    ana.histograms.addVecHistogram(TString::Format("sg_%s_all_track_ring", modename), 18, 0, 18, [&]() { return sg_all_track_ring; } );
    ana.histograms.addVecHistogram(TString::Format("sg_%s_matched_track_module", modename), 40, 0, 40, [&]() { return sg_matched_track_module; } );
    ana.histograms.addVecHistogram(TString::Format("sg_%s_all_track_module", modename), 40, 0, 40, [&]() { return sg_all_track_module; } );
    ana.histograms.addVecHistogram(TString::Format("sg_%s_matched_track_targ_ring", modename), 18, 0, 18, [&]() { return sg_matched_track_targ_ring; } );
    ana.histograms.addVecHistogram(TString::Format("sg_%s_all_track_targ_ring", modename), 18, 0, 18, [&]() { return sg_all_track_targ_ring; } );
    ana.histograms.addVecHistogram(TString::Format("sg_%s_matched_track_targ_module", modename), 40, 0, 40, [&]() { return sg_matched_track_targ_module; } );
    ana.histograms.addVecHistogram(TString::Format("sg_%s_all_track_targ_module", modename), 40, 0, 40, [&]() { return sg_all_track_targ_module; } );

    const int nlayers = NLAYERS;
    for (int ii = 0; ii < nlayers; ++ii)
    {
        ana.histograms.addVecHistogram(TString::Format("sg_%s_matched_track_pt_by_layer%d", modename, ii), pt_boundaries, [&, ii]() { return sg_matched_track_pt_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("sg_%s_all_track_pt_by_layer%d", modename, ii), pt_boundaries, [&, ii]() { return sg_all_track_pt_by_layer[ii]; } );
    }
    for (int ii = 0; ii < nlayers; ++ii)
    {
        ana.histograms.addVecHistogram(TString::Format("sg_%s_matched_track_eta_by_layer%d", modename, ii), 50, -4, 4, [&, ii]() { return sg_matched_track_eta_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("sg_%s_all_track_eta_by_layer%d", modename, ii), 50, -4, 4, [&, ii]() { return sg_all_track_eta_by_layer[ii]; } );
    }
    for (int ii = 0; ii < nlayers; ++ii)
    {
        ana.histograms.addVecHistogram(TString::Format("sg_%s_matched_track_phi_by_layer%d", modename, ii), 200, -3.1416, 3.1416, [&, ii]() { return sg_matched_track_phi_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("sg_%s_all_track_phi_by_layer%d", modename, ii), 200, -3.1416, 3.1416, [&, ii]() { return sg_all_track_phi_by_layer[ii]; } );
    }
    for (int ii = 0; ii < nlayers; ++ii)
    {
        ana.histograms.addVecHistogram(TString::Format("sg_%s_matched_track_wrapphi_by_layer%d", modename, ii), 200, -3.1416, 3.1416, [&, ii]() { return sg_matched_track_wrapphi_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("sg_%s_all_track_wrapphi_by_layer%d", modename, ii), 200, -3.1416, 3.1416, [&, ii]() { return sg_all_track_wrapphi_by_layer[ii]; } );
    }
    for (int ii = 0; ii < nlayers; ++ii)
    {
        ana.histograms.addVecHistogram(TString::Format("sg_%s_matched_track_z_by_layer%d", modename, ii), 200, -300, 300, [&, ii]() { return sg_matched_track_z_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("sg_%s_all_track_z_by_layer%d", modename, ii), 200, -300, 300, [&, ii]() { return sg_all_track_z_by_layer[ii]; } );
    }
    for (int ii = 0; ii < nlayers; ++ii)
    {
        ana.histograms.addVecHistogram(TString::Format("sg_%s_matched_track_ring_by_layer%d", modename, ii), 18, 0, 18, [&, ii]() { return sg_matched_track_ring_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("sg_%s_all_track_ring_by_layer%d", modename, ii), 18, 0, 18, [&, ii]() { return sg_all_track_ring_by_layer[ii]; } );
    }
    for (int ii = 0; ii < nlayers; ++ii)
    {
        ana.histograms.addVecHistogram(TString::Format("sg_%s_matched_track_module_by_layer%d", modename, ii), 40, 0, 40, [&, ii]() { return sg_matched_track_module_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("sg_%s_all_track_module_by_layer%d", modename, ii), 40, 0, 40, [&, ii]() { return sg_all_track_module_by_layer[ii]; } );
    }
    for (int ii = 0; ii < nlayers; ++ii)
    {
        ana.histograms.addVecHistogram(TString::Format("sg_%s_matched_track_targ_ring_by_layer%d", modename, ii), 18, 0, 18, [&, ii]() { return sg_matched_track_targ_ring_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("sg_%s_all_track_targ_ring_by_layer%d", modename, ii), 18, 0, 18, [&, ii]() { return sg_all_track_targ_ring_by_layer[ii]; } );
    }
    for (int ii = 0; ii < nlayers; ++ii)
    {
        ana.histograms.addVecHistogram(TString::Format("sg_%s_matched_track_targ_module_by_layer%d", modename, ii), 40, 0, 40, [&, ii]() { return sg_matched_track_targ_module_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("sg_%s_all_track_targ_module_by_layer%d", modename, ii), 40, 0, 40, [&, ii]() { return sg_all_track_targ_module_by_layer[ii]; } );
    }
}

void StudySegmentEfficiency::doStudy(SDL::Event& event, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents)
{
    // Each do study is performed per event

    // First clear all the output variables that will be used to fill the histograms for this event
    sg_matched_track_pt.clear();
    sg_all_track_pt.clear();
    sg_matched_track_eta.clear();
    sg_all_track_eta.clear();
    sg_matched_track_phi.clear();
    sg_all_track_phi.clear();
    sg_matched_track_z.clear();
    sg_all_track_z.clear();
    sg_matched_track_ring.clear();
    sg_all_track_ring.clear();
    sg_matched_track_module.clear();
    sg_all_track_module.clear();
    sg_matched_track_targ_ring.clear();
    sg_all_track_targ_ring.clear();
    sg_matched_track_targ_module.clear();
    sg_all_track_targ_module.clear();
    for (int ii = 0; ii < NLAYERS; ++ii)
    {
        sg_matched_track_pt_by_layer[ii].clear();
        sg_all_track_pt_by_layer[ii].clear();
        sg_matched_track_eta_by_layer[ii].clear();
        sg_all_track_eta_by_layer[ii].clear();
        sg_matched_track_phi_by_layer[ii].clear();
        sg_all_track_phi_by_layer[ii].clear();
        sg_matched_track_wrapphi_by_layer[ii].clear();
        sg_all_track_wrapphi_by_layer[ii].clear();
        sg_matched_track_z_by_layer[ii].clear();
        sg_all_track_z_by_layer[ii].clear();
        sg_matched_track_ring_by_layer[ii].clear();
        sg_all_track_ring_by_layer[ii].clear();
        sg_matched_track_module_by_layer[ii].clear();
        sg_all_track_module_by_layer[ii].clear();
        sg_matched_track_targ_ring_by_layer[ii].clear();
        sg_all_track_targ_ring_by_layer[ii].clear();
        sg_matched_track_targ_module_by_layer[ii].clear();
        sg_all_track_targ_module_by_layer[ii].clear();
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

        std::array<bool, 6> trackmatch_by_layer;
        trackmatch_by_layer[0] = false;
        trackmatch_by_layer[1] = false;
        trackmatch_by_layer[2] = false;
        trackmatch_by_layer[3] = false;
        trackmatch_by_layer[4] = false;

        // Loop over the lower modules that contains hits for this track
        for (auto& lowerModulePtr_Track : trackevent.getLowerModulePtrs())
        {

            // Parse the layer index later to be used for indexing
            int layer_idx = lowerModulePtr_Track->layer() - 1;

            // Parse pt and eta of this track
            float rawpt = trk.sim_pt()[isimtrk];
            float pt = std::min((double) trk.sim_pt()[isimtrk], 49.999);
            float eta = trk.sim_eta()[isimtrk];
            float phi = trk.sim_phi()[isimtrk];

            // For this module that a sim-track supposedly passed through if there are no more than 1 segment
            // it means that this track did not leave at least one hit each in each side of the module + the same for the other module
            // The getSegmentPtrs() will return ALL pairs of minidoublets between the modules.
            // That's because the simtrkevent would have createSegments called with SDL::AllComb_SGAlgo option.
            // This option loops over all hits in inner lower module and outer lower module and forms every possible pair of mini-doublets to form segments.
            // So if the following condition of size() == 0 is true, it means this sim-track didn't leave at least one minidoublet in each side.
            if (lowerModulePtr_Track->getSegmentPtrs().size() == 0)
            {
                continue;
            }

            // Assuming that I do have more than one segment for this module, restrict the phase-space based on the study mode
            // if the study mode is for example barrel-bareel, then ask whether there is a segment that passes barrel-barrel
            // So among the "getSegmentPtrs()" list, there should be at least one that has mini-doublets in both barrel modules
            // So we will do a for loop over the segments, and depending on the study mode, we will save interested segments.
            // If the number of interested segments is = 0, then we will "continue" from this track.
            // And later the list of saved interested segments will be used to compare against the true reco-ed segments.
            // If we find a match, then that passes the numerator as well
            std::vector<SDL::Segment*> sgs_of_interest;
            for (auto& sg_Track : lowerModulePtr_Track->getSegmentPtrs())
            {
                const SDL::Module& innerLowerModule = sg_Track->innerMiniDoubletPtr()->lowerHitPtr()->getModule();
                const SDL::Module& outerLowerModule = sg_Track->outerMiniDoubletPtr()->lowerHitPtr()->getModule();
                bool isInnerMiniDoubletBarrel = innerLowerModule.subdet() == SDL::Module::Barrel;
                bool isOuterMiniDoubletBarrel = outerLowerModule.subdet() == SDL::Module::Barrel;
                bool isInnerMiniDoubletCenter = innerLowerModule.side() == SDL::Module::Center;
                bool isOuterMiniDoubletCenter = outerLowerModule.side() == SDL::Module::Center;
                bool isInnerMiniDoubletBarrelFlat = isInnerMiniDoubletBarrel and isInnerMiniDoubletCenter;
                bool isOuterMiniDoubletBarrelFlat = isOuterMiniDoubletBarrel and isOuterMiniDoubletCenter;
                bool isInnerMiniDoubletBarrelTilt = isInnerMiniDoubletBarrel and not isInnerMiniDoubletCenter;
                bool isOuterMiniDoubletBarrelTilt = isOuterMiniDoubletBarrel and not isOuterMiniDoubletCenter;
                bool isInnerMiniDoubletPS = innerLowerModule.moduleType() == SDL::Module::PS;
                bool isOuterMiniDoubletPS = outerLowerModule.moduleType() == SDL::Module::PS;

                // Depending on the mode, only include a subset of interested segments
                switch (mode)
                {
                    case kStudyEffAll: /* do nothing */ break;
                    case kStudyEffBarrelBarrel: if (not (isInnerMiniDoubletBarrel and isOuterMiniDoubletBarrel)) { continue; } break;
                    case kStudyEffBarrelFlatBarrel: if (not (isInnerMiniDoubletBarrelFlat and isOuterMiniDoubletBarrel)) { continue; } break;
                    case kStudyEffBarrelTiltBarrel: if (not (isInnerMiniDoubletBarrelTilt and isOuterMiniDoubletBarrel)) { continue; } break;
                    case kStudyEffBarrelFlatBarrelFlat: if (not (isInnerMiniDoubletBarrelFlat and isOuterMiniDoubletBarrelFlat)) { continue; } break;
                    case kStudyEffBarrelFlatBarrelTilt: if (not (isInnerMiniDoubletBarrelFlat and isOuterMiniDoubletBarrelTilt)) { continue; } break;
                    case kStudyEffBarrelTiltBarrelFlat: if (not (isInnerMiniDoubletBarrelTilt and isOuterMiniDoubletBarrelFlat)) { continue; } break;
                    case kStudyEffBarrelTiltBarrelTilt: if (not (isInnerMiniDoubletBarrelTilt and isOuterMiniDoubletBarrelTilt)) { continue; } break;
                    case kStudyEffBarrelEndcap: if (not (isInnerMiniDoubletBarrel and not isOuterMiniDoubletBarrel)) { continue; } break;
                    case kStudyEffBarrelTiltEndcap: if (not (isInnerMiniDoubletBarrel and isInnerMiniDoubletBarrelTilt and not isOuterMiniDoubletBarrel)) { continue; } break;
                    case kStudyEffBarrel: if (not (isInnerMiniDoubletBarrel)) { continue; } break;
                    case kStudyEffEndcap: if (not (not isInnerMiniDoubletBarrel)) { continue; } break;
                    case kStudyEffEndcapPS: if (not (not isInnerMiniDoubletBarrel and isInnerMiniDoubletPS)) { continue; } break;
                    case kStudyEffEndcapPSPS: if (not (not isInnerMiniDoubletBarrel and isInnerMiniDoubletPS and isOuterMiniDoubletPS)) { continue; } break;
                    case kStudyEffEndcapPS2S: if (not (not isInnerMiniDoubletBarrel and isInnerMiniDoubletPS and not isOuterMiniDoubletPS)) { continue; } break;
                    case kStudyEffEndcap2S: if (not (not isInnerMiniDoubletBarrel and not isInnerMiniDoubletPS)) { continue; } break;
                    default: /* skip everything should not be here anyways...*/ continue; break;
                }

                // If this segment passes the condition that it is of interest then, add to the list of segements of interest
                sgs_of_interest.push_back(sg_Track);

            }

            // If no sgs of interest are found then skip
            if (sgs_of_interest.size() == 0)
                continue;

            // Boolean to test whether for this module that a track passed through, whether it found a matched mini-doublet
            bool match = false;

            // Loop over the sg "candidate" from the module that a sim-track passed through and left at least one mini-doublet in each module
            // The sgs_of_interest holds only the sg "candidate" that we think are of interest for the given study mode
            float z; // the average value of z for the matched inner layer truth matched mini-doublet's lower hit

            int targ_ring = (sgs_of_interest[0]->outerMiniDoubletPtr()->lowerHitPtr()->getModule()).ring();
            int targ_module = (sgs_of_interest[0]->outerMiniDoubletPtr()->lowerHitPtr()->getModule()).module();
            for (auto& sg_Track : sgs_of_interest)
            {

                // Sum all the hits' z position of the inner md's lower hit z's. then after the loop divide by total number to get the average value
                z  = sg_Track->innerMiniDoubletPtr()->lowerHitPtr()->z();

                // Loop over the sg reconstructed from with proper SDL algorithm and if the index of the mini-doublets match (i.e. if the 4 hits match)
                // Then we have found at least one segment associated to this track's reco-hits in this module
                // Therefore flag the match boolean
                for (auto& sg : event.getModule(lowerModulePtr_Track->detId()).getSegmentPtrs())
                {
                    if (sg_Track->isIdxMatched(*sg))
                    {
                        match = true;
                        targ_ring = (sg_Track->outerMiniDoubletPtr()->lowerHitPtr()->getModule()).ring();
                        targ_module = (sg_Track->outerMiniDoubletPtr()->lowerHitPtr()->getModule()).module();
                        // if (mode == kStudyEffBarrelFlatBarrelFlat)
                        // {
                        //     SDL::cout << "***************************************************************************" << std::endl;
                        //     SDL::cout << "Good segment" << std::endl;
                        //     SDL::cout << "***************************************************************************" << std::endl;
                        //     SDL::cout <<  " rawpt: " << rawpt <<  " eta: " << eta <<  " phi: " << phi <<  std::endl;
                        //     printSegmentDebugInfo(sg_Track, rawpt);
                        // }
                    }
                }
            }

            if (match)
                trackmatch_by_layer[layer_idx] = true;
        }

        float pt = std::min((double) trk.sim_pt()[isimtrk], 49.999);
        float eta = trk.sim_eta()[isimtrk];

        for (unsigned int i = 0; i < 5; ++i)
        {

            // Denominator for specific layers pt efficiency
            sg_all_track_pt_by_layer[i].push_back(pt);

            // Denominator for specific layers eta efficiency (notice the 1 GeV cut)
            if (trk.sim_pt()[isimtrk] > 1.)
                sg_all_track_eta_by_layer[i].push_back(eta);

            // Numerators
            if (trackmatch_by_layer[i])
            {
                // Numerators for matched specific layer pt effciency
                sg_matched_track_pt_by_layer[i].push_back(pt);

                // Numerators for matched specific layer eta effciency (notice the 1 GeV cut)
                if (trk.sim_pt()[isimtrk] > 1.)
                    sg_matched_track_eta_by_layer[i].push_back(eta);

            }

        }

    }

}

void StudySegmentEfficiency::printSegmentDebugInfo(SDL::Segment* sg_Track, float pt)
{
    sg_Track->runSegmentAlgo(SDL::Default_SGAlgo, SDL::Log_Debug3);
    SDL::cout <<  " sg_Track->getDeltaPhiChange(): " << sg_Track->getDeltaPhiChange() <<  std::endl;
    SDL::cout <<  " sg_Track->passesSegmentAlgo(SDL::Default_SGAlgo): " << sg_Track->passesSegmentAlgo(SDL::Default_SGAlgo) <<  std::endl;
    // SDL::cout << sg_Track << std::endl;
    const SDL::MiniDoublet& innerMiniDoublet = (*sg_Track->innerMiniDoubletPtr());
    const SDL::MiniDoublet& outerMiniDoublet = (*sg_Track->outerMiniDoubletPtr());
    SDL::cout <<  " innerMiniDoublet.getDeltaPhiChange(): " << innerMiniDoublet.getDeltaPhiChange() <<  std::endl;
    SDL::cout <<  " outerMiniDoublet.getDeltaPhiChange(): " << outerMiniDoublet.getDeltaPhiChange() <<  std::endl;
    SDL::cout <<  " innerMiniDoublet.getDeltaPhiChangeNoShift(): " << innerMiniDoublet.getDeltaPhiChangeNoShift() <<  std::endl;
    SDL::cout <<  " outerMiniDoublet.getDeltaPhiChangeNoShift(): " << outerMiniDoublet.getDeltaPhiChangeNoShift() <<  std::endl;
    const SDL::Hit& lowerHitInnerMiniDoublet = (*innerMiniDoublet.lowerHitPtr());
    const SDL::Hit& upperHitInnerMiniDoublet = (*innerMiniDoublet.upperHitPtr());
    const SDL::Hit& lowerHitOuterMiniDoublet = (*outerMiniDoublet.lowerHitPtr());
    const SDL::Hit& upperHitOuterMiniDoublet = (*outerMiniDoublet.upperHitPtr());
    SDL::cout <<  " SDL::MathUtil::ptEstimateFromDeltaPhiChangeAndRt(innerMiniDoublet.getDeltaPhiChange(),lowerHitInnerMiniDoublet.rt()): " << SDL::MathUtil::ptEstimateFromDeltaPhiChangeAndRt(innerMiniDoublet.getDeltaPhiChange(),lowerHitInnerMiniDoublet.rt()) <<  std::endl;
    SDL::cout <<  " SDL::MathUtil::ptEstimateFromDeltaPhiChangeAndRt(outerMiniDoublet.getDeltaPhiChange(),lowerHitOuterMiniDoublet.rt()): " << SDL::MathUtil::ptEstimateFromDeltaPhiChangeAndRt(outerMiniDoublet.getDeltaPhiChange(),lowerHitOuterMiniDoublet.rt()) <<  std::endl;
    // const float innerMiniDoubletDPhiEstimate = SDL::MathUtil::dphiEstimateFromPtAndRt(pt, lowerHitInnerMiniDoublet.rt());
    // const float outerMiniDoubletDPhiEstimate = SDL::MathUtil::dphiEstimateFromPtAndRt(pt, lowerHitOuterMiniDoublet.rt());
    // SDL::cout <<  " innerMiniDoubletDPhiEstimate: " << innerMiniDoubletDPhiEstimate <<  std::endl;
    // SDL::cout <<  " outerMiniDoubletDPhiEstimate: " << outerMiniDoubletDPhiEstimate <<  std::endl;
    const SDL::Module& innerLowerModule = lowerHitInnerMiniDoublet.getModule();
    const SDL::Module& outerLowerModule = lowerHitOuterMiniDoublet.getModule();
    const SDL::Module& innerUpperModule = upperHitInnerMiniDoublet.getModule();
    const SDL::Module& outerUpperModule = upperHitOuterMiniDoublet.getModule();
    SDL::cout << innerLowerModule << std::endl;
    SDL::cout << innerUpperModule << std::endl;
    SDL::cout << outerLowerModule << std::endl;
    SDL::cout << outerUpperModule << std::endl;
    SDL::cout << innerMiniDoublet << std::endl;
    SDL::cout << outerMiniDoublet << std::endl;
    SDL::MiniDoublet::shiftStripHits(lowerHitInnerMiniDoublet, upperHitInnerMiniDoublet, innerLowerModule, SDL::Log_Debug3);
}

void StudySegmentEfficiency::doStudy_v1(SDL::Event& event, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents)
{
    // Each do study is performed per event

    // First clear all the output variables that will be used to fill the histograms for this event
    sg_matched_track_pt.clear();
    sg_all_track_pt.clear();
    sg_matched_track_eta.clear();
    sg_all_track_eta.clear();
    sg_matched_track_phi.clear();
    sg_all_track_phi.clear();
    sg_matched_track_z.clear();
    sg_all_track_z.clear();
    sg_matched_track_ring.clear();
    sg_all_track_ring.clear();
    sg_matched_track_module.clear();
    sg_all_track_module.clear();
    sg_matched_track_targ_ring.clear();
    sg_all_track_targ_ring.clear();
    sg_matched_track_targ_module.clear();
    sg_all_track_targ_module.clear();
    for (int ii = 0; ii < NLAYERS; ++ii)
    {
        sg_matched_track_pt_by_layer[ii].clear();
        sg_all_track_pt_by_layer[ii].clear();
        sg_matched_track_eta_by_layer[ii].clear();
        sg_all_track_eta_by_layer[ii].clear();
        sg_matched_track_phi_by_layer[ii].clear();
        sg_all_track_phi_by_layer[ii].clear();
        sg_matched_track_wrapphi_by_layer[ii].clear();
        sg_all_track_wrapphi_by_layer[ii].clear();
        sg_matched_track_z_by_layer[ii].clear();
        sg_all_track_z_by_layer[ii].clear();
        sg_matched_track_ring_by_layer[ii].clear();
        sg_all_track_ring_by_layer[ii].clear();
        sg_matched_track_module_by_layer[ii].clear();
        sg_all_track_module_by_layer[ii].clear();
        sg_matched_track_targ_ring_by_layer[ii].clear();
        sg_all_track_targ_ring_by_layer[ii].clear();
        sg_matched_track_targ_module_by_layer[ii].clear();
        sg_all_track_targ_module_by_layer[ii].clear();
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

        // Loop over the lower modules that contains hits for this track
        for (auto& lowerModulePtr_Track : trackevent.getLowerModulePtrs())
        {

            // Parse the layer index later to be used for indexing
            int layer_idx = lowerModulePtr_Track->layer() - 1;

            // Parse pt and eta of this track
            float rawpt = trk.sim_pt()[isimtrk];
            float pt = std::min((double) trk.sim_pt()[isimtrk], 49.999);
            float eta = trk.sim_eta()[isimtrk];
            float phi = trk.sim_phi()[isimtrk];

            // For this module that a sim-track supposedly passed through if there are no more than 1 segment
            // it means that this track did not leave at least one hit each in each side of the module + the same for the other module
            // The getSegmentPtrs() will return ALL pairs of minidoublets between the modules.
            // That's because the simtrkevent would have createSegments called with SDL::AllComb_SGAlgo option.
            // This option loops over all hits in inner lower module and outer lower module and forms every possible pair of mini-doublets to form segments.
            // So if the following condition of size() == 0 is true, it means this sim-track didn't leave at least one minidoublet in each side.
            if (lowerModulePtr_Track->getSegmentPtrs().size() == 0)
            {
                continue;
            }

            // Assuming that I do have more than one segment for this module, restrict the phase-space based on the study mode
            // if the study mode is for example barrel-bareel, then ask whether there is a segment that passes barrel-barrel
            // So among the "getSegmentPtrs()" list, there should be at least one that has mini-doublets in both barrel modules
            // So we will do a for loop over the segments, and depending on the study mode, we will save interested segments.
            // If the number of interested segments is = 0, then we will "continue" from this track.
            // And later the list of saved interested segments will be used to compare against the true reco-ed segments.
            // If we find a match, then that passes the numerator as well
            std::vector<SDL::Segment*> sgs_of_interest;
            for (auto& sg_Track : lowerModulePtr_Track->getSegmentPtrs())
            {
                const SDL::Module& innerLowerModule = sg_Track->innerMiniDoubletPtr()->lowerHitPtr()->getModule();
                const SDL::Module& outerLowerModule = sg_Track->outerMiniDoubletPtr()->lowerHitPtr()->getModule();
                bool isInnerMiniDoubletBarrel = innerLowerModule.subdet() == SDL::Module::Barrel;
                bool isOuterMiniDoubletBarrel = outerLowerModule.subdet() == SDL::Module::Barrel;
                bool isInnerMiniDoubletCenter = innerLowerModule.side() == SDL::Module::Center;
                bool isOuterMiniDoubletCenter = outerLowerModule.side() == SDL::Module::Center;
                bool isInnerMiniDoubletBarrelFlat = isInnerMiniDoubletBarrel and isInnerMiniDoubletCenter;
                bool isOuterMiniDoubletBarrelFlat = isOuterMiniDoubletBarrel and isOuterMiniDoubletCenter;
                bool isInnerMiniDoubletBarrelTilt = isInnerMiniDoubletBarrel and not isInnerMiniDoubletCenter;
                bool isOuterMiniDoubletBarrelTilt = isOuterMiniDoubletBarrel and not isOuterMiniDoubletCenter;
                bool isInnerMiniDoubletPS = innerLowerModule.moduleType() == SDL::Module::PS;
                bool isOuterMiniDoubletPS = outerLowerModule.moduleType() == SDL::Module::PS;

                // Depending on the mode, only include a subset of interested segments
                switch (mode)
                {
                    case kStudyEffAll: /* do nothing */ break;
                    case kStudyEffBarrelBarrel: if (not (isInnerMiniDoubletBarrel and isOuterMiniDoubletBarrel)) { continue; } break;
                    case kStudyEffBarrelFlatBarrel: if (not (isInnerMiniDoubletBarrelFlat and isOuterMiniDoubletBarrel)) { continue; } break;
                    case kStudyEffBarrelTiltBarrel: if (not (isInnerMiniDoubletBarrelTilt and isOuterMiniDoubletBarrel)) { continue; } break;
                    case kStudyEffBarrelFlatBarrelFlat: if (not (isInnerMiniDoubletBarrelFlat and isOuterMiniDoubletBarrelFlat)) { continue; } break;
                    case kStudyEffBarrelFlatBarrelTilt: if (not (isInnerMiniDoubletBarrelFlat and isOuterMiniDoubletBarrelTilt)) { continue; } break;
                    case kStudyEffBarrelTiltBarrelFlat: if (not (isInnerMiniDoubletBarrelTilt and isOuterMiniDoubletBarrelFlat)) { continue; } break;
                    case kStudyEffBarrelTiltBarrelTilt: if (not (isInnerMiniDoubletBarrelTilt and isOuterMiniDoubletBarrelTilt)) { continue; } break;
                    case kStudyEffBarrelEndcap: if (not (isInnerMiniDoubletBarrel and not isOuterMiniDoubletBarrel)) { continue; } break;
                    case kStudyEffBarrelTiltEndcap: if (not (isInnerMiniDoubletBarrel and isInnerMiniDoubletBarrelTilt and not isOuterMiniDoubletBarrel)) { continue; } break;
                    case kStudyEffBarrel: if (not (isInnerMiniDoubletBarrel)) { continue; } break;
                    case kStudyEffEndcap: if (not (not isInnerMiniDoubletBarrel)) { continue; } break;
                    case kStudyEffEndcapPS: if (not (not isInnerMiniDoubletBarrel and isInnerMiniDoubletPS)) { continue; } break;
                    case kStudyEffEndcapPSPS: if (not (not isInnerMiniDoubletBarrel and isInnerMiniDoubletPS and isOuterMiniDoubletPS)) { continue; } break;
                    case kStudyEffEndcapPS2S: if (not (not isInnerMiniDoubletBarrel and isInnerMiniDoubletPS and not isOuterMiniDoubletPS)) { continue; } break;
                    case kStudyEffEndcap2S: if (not (not isInnerMiniDoubletBarrel and not isInnerMiniDoubletPS)) { continue; } break;
                    default: /* skip everything should not be here anyways...*/ continue; break;
                }

                // If this segment passes the condition that it is of interest then, add to the list of segements of interest
                sgs_of_interest.push_back(sg_Track);

            }

            // If no sgs of interest are found then skip
            if (sgs_of_interest.size() == 0)
                continue;

            // Boolean to test whether for this module that a track passed through, whether it found a matched mini-doublet
            bool match = false;

            // Loop over the sg "candidate" from the module that a sim-track passed through and left at least one mini-doublet in each module
            // The sgs_of_interest holds only the sg "candidate" that we think are of interest for the given study mode
            float z; // the average value of z for the matched inner layer truth matched mini-doublet's lower hit

            int targ_ring = (sgs_of_interest[0]->outerMiniDoubletPtr()->lowerHitPtr()->getModule()).ring();
            int targ_module = (sgs_of_interest[0]->outerMiniDoubletPtr()->lowerHitPtr()->getModule()).module();
            for (auto& sg_Track : sgs_of_interest)
            {

                // Sum all the hits' z position of the inner md's lower hit z's. then after the loop divide by total number to get the average value
                z  = sg_Track->innerMiniDoubletPtr()->lowerHitPtr()->z();

                // Loop over the sg reconstructed from with proper SDL algorithm and if the index of the mini-doublets match (i.e. if the 4 hits match)
                // Then we have found at least one segment associated to this track's reco-hits in this module
                // Therefore flag the match boolean
                for (auto& sg : event.getModule(lowerModulePtr_Track->detId()).getSegmentPtrs())
                {
                    if (sg_Track->isIdxMatched(*sg))
                    {
                        match = true;
                        targ_ring = (sg_Track->outerMiniDoubletPtr()->lowerHitPtr()->getModule()).ring();
                        targ_module = (sg_Track->outerMiniDoubletPtr()->lowerHitPtr()->getModule()).module();
                        // if (mode == kStudyEffBarrelFlatBarrelFlat)
                        // {
                        //     SDL::cout << "***************************************************************************" << std::endl;
                        //     SDL::cout << "Good segment" << std::endl;
                        //     SDL::cout << "***************************************************************************" << std::endl;
                        //     SDL::cout <<  " rawpt: " << rawpt <<  " eta: " << eta <<  " phi: " << phi <<  std::endl;
                        //     printSegmentDebugInfo(sg_Track, rawpt);
                        // }
                    }
                }
            }
            // z /= sgs_of_interest.size();

            // // Debugging high pt inefficiency
            // if (pt > 5 and not match and mode == kStudyEffBarrelTiltBarrelTilt)
            // {
            //     // Among the sg "candidate" of interest (i.e. the ones that passes a module phase-space of interest
            //     // Why did it fail?
            //     SDL::cout << "Studying failed segment" << std::endl;
            //     for (auto& sg_Track : sgs_of_interest)
            //     {
            //         const SDL::MiniDoublet& innerMiniDoublet = *sg_Track->innerMiniDoubletPtr();
            //         const SDL::MiniDoublet& outerMiniDoublet = *sg_Track->outerMiniDoubletPtr();
            //         SDL::Segment::isMiniDoubletPairASegmentBarrel(innerMiniDoublet, outerMiniDoublet, SDL::Default_SGAlgo, SDL::Log_Debug3);
            //     }
            // }

            // At this stage, we have either found a segment in this module either matched to the track or not.

            // Denominator for all layers pt efficiency
            sg_all_track_pt.push_back(pt);

            // Denominator for all layers eta efficiency (notice the 1 GeV cut)
            if (trk.sim_pt()[isimtrk] > 1.)
                sg_all_track_eta.push_back(eta);

            // Denominator for all layers phi efficiency (notice the 1 GeV cut)
            if (trk.sim_pt()[isimtrk] > 1.)
                sg_all_track_phi.push_back(phi);

            // Denominator for all layers z efficiency (notice the 1 GeV cut)
            if (trk.sim_pt()[isimtrk] > 1.)
                sg_all_track_z.push_back(z);

            // Denominator for all layers ring efficiency (notice the 1 GeV cut)
            if (trk.sim_pt()[isimtrk] > 1.)
                sg_all_track_ring.push_back(lowerModulePtr_Track->ring());

            // Denominator for all layers module efficiency (notice the 1 GeV cut)
            if (trk.sim_pt()[isimtrk] > 1.)
                sg_all_track_module.push_back(lowerModulePtr_Track->module());

            // Denominator for all layers ring efficiency (notice the 1 GeV cut)
            if (trk.sim_pt()[isimtrk] > 1.)
                sg_all_track_targ_ring.push_back(targ_ring);

            // Denominator for all layers module efficiency (notice the 1 GeV cut)
            if (trk.sim_pt()[isimtrk] > 1.)
                sg_all_track_targ_module.push_back(targ_module);

            // Numerators
            if (match)
            {

                // Numerators for matched all layers pt efficiency
                sg_matched_track_pt.push_back(pt);

                // Numeratosr for matched all layers eta efficiency (notice the 1 GeV cut)
                if (trk.sim_pt()[isimtrk] > 1.)
                    sg_matched_track_eta.push_back(eta);

                // Numeratosr for matched all layers phi efficiency (notice the 1 GeV cut)
                if (trk.sim_pt()[isimtrk] > 1.)
                    sg_matched_track_phi.push_back(phi);

                // Numeratosr for matched all layers z efficiency (notice the 1 GeV cut)
                if (trk.sim_pt()[isimtrk] > 1.)
                    sg_matched_track_z.push_back(z);

                // Numeratosr for matched all layers ring efficiency (notice the 1 GeV cut)
                if (trk.sim_pt()[isimtrk] > 1.)
                    sg_matched_track_ring.push_back(lowerModulePtr_Track->ring());

                // Numeratosr for matched all layers module efficiency (notice the 1 GeV cut)
                if (trk.sim_pt()[isimtrk] > 1.)
                    sg_matched_track_module.push_back(lowerModulePtr_Track->module());

                // Numeratosr for matched all layers ring efficiency (notice the 1 GeV cut)
                if (trk.sim_pt()[isimtrk] > 1.)
                    sg_matched_track_targ_ring.push_back(targ_ring);

                // Numeratosr for matched all layers module efficiency (notice the 1 GeV cut)
                if (trk.sim_pt()[isimtrk] > 1.)
                    sg_matched_track_targ_module.push_back(targ_module);
            }
            // Failed tracks for all layers
            else
            {
                // Doing nothing for now ...
            }

            // Denominator for specific layers pt efficiency
            sg_all_track_pt_by_layer[layer_idx].push_back(pt);

            // Denominator for specific layers eta efficiency (notice the 1 GeV cut)
            if (trk.sim_pt()[isimtrk] > 1.)
                sg_all_track_eta_by_layer[layer_idx].push_back(eta);

            // Denominator for specific layers phi efficiency (notice the 1 GeV cut)
            if (trk.sim_pt()[isimtrk] > 1.)
                sg_all_track_phi_by_layer[layer_idx].push_back(phi);

            // Denominator for specific layers z efficiency (notice the 1 GeV cut)
            if (trk.sim_pt()[isimtrk] > 1.)
                sg_all_track_z_by_layer[layer_idx].push_back(z);

            // Denominator for specific layers ring efficiency (notice the 1 GeV cut)
            if (trk.sim_pt()[isimtrk] > 1.)
                sg_all_track_ring_by_layer[layer_idx].push_back(lowerModulePtr_Track->ring());

            // Denominator for specific layers module efficiency (notice the 1 GeV cut)
            if (trk.sim_pt()[isimtrk] > 1.)
                sg_all_track_module_by_layer[layer_idx].push_back(lowerModulePtr_Track->module());

            // Denominator for specific layers ring efficiency (notice the 1 GeV cut)
            if (trk.sim_pt()[isimtrk] > 1.)
                sg_all_track_targ_ring_by_layer[layer_idx].push_back(targ_ring);

            // Denominator for specific layers module efficiency (notice the 1 GeV cut)
            if (trk.sim_pt()[isimtrk] > 1.)
                sg_all_track_targ_module_by_layer[layer_idx].push_back(targ_module);

            // Denominator for specific layers wrapphi efficiency (notice the 1 GeV cut)
            float wrapphi = 0;
            if (layer_idx == 0)
                wrapphi = fmod(fabs(phi), 2*TMath::Pi() / 18);
            else if (layer_idx == 1)
                wrapphi = fmod(fabs(phi), 2*TMath::Pi() / 26);
            else
                wrapphi = fmod(fabs(phi), 2*TMath::Pi() / 36);
            if (trk.sim_pt()[isimtrk] > 5.)
                sg_all_track_wrapphi_by_layer[layer_idx].push_back(wrapphi);

            // Numerators
            if (match)
            {
                // Numerators for matched specific layer pt effciency
                sg_matched_track_pt_by_layer[layer_idx].push_back(pt);

                // Numerators for matched specific layer eta effciency (notice the 1 GeV cut)
                if (trk.sim_pt()[isimtrk] > 1.)
                    sg_matched_track_eta_by_layer[layer_idx].push_back(eta);

                // Numerators for matched specific layer phi effciency (notice the 1 GeV cut)
                if (trk.sim_pt()[isimtrk] > 1.)
                    sg_matched_track_phi_by_layer[layer_idx].push_back(phi);

                // Numerators for matched specific layer wrapphi effciency (notice the 1 GeV cut)
                if (trk.sim_pt()[isimtrk] > 5.)
                    sg_matched_track_wrapphi_by_layer[layer_idx].push_back(wrapphi);

                // Numerators for matched specific layer z effciency (notice the 1 GeV cut)
                if (trk.sim_pt()[isimtrk] > 1.)
                    sg_matched_track_z_by_layer[layer_idx].push_back(z);

                // Numerators for matched specific layer ring effciency (notice the 1 GeV cut)
                if (trk.sim_pt()[isimtrk] > 1.)
                    sg_matched_track_ring_by_layer[layer_idx].push_back(lowerModulePtr_Track->ring());

                // Numerators for matched specific layer module effciency (notice the 1 GeV cut)
                if (trk.sim_pt()[isimtrk] > 1.)
                    sg_matched_track_module_by_layer[layer_idx].push_back(lowerModulePtr_Track->module());

                // Numerators for matched specific layer ring effciency (notice the 1 GeV cut)
                if (trk.sim_pt()[isimtrk] > 1.)
                    sg_matched_track_targ_ring_by_layer[layer_idx].push_back(targ_ring);

                // Numerators for matched specific layer module effciency (notice the 1 GeV cut)
                if (trk.sim_pt()[isimtrk] > 1.)
                    sg_matched_track_targ_module_by_layer[layer_idx].push_back(targ_module);

            }
            // Failed tracks for specific layers
            else
            {
                // Studying inefficiency in barreltilt to barreltilt
                if (mode == kStudyEffEndcap2S and pt > 5.)
                {

                    for (auto& sg_Track : sgs_of_interest)
                    {
                        SDL::cout << "***************************************************************************" << std::endl;
                        SDL::cout << "Bad segment" << std::endl;
                        SDL::cout << "***************************************************************************" << std::endl;
                        SDL::cout <<  " rawpt: " << rawpt <<  " eta: " << eta <<  " phi: " << phi <<  std::endl;
                        printSegmentDebugInfo(sg_Track, rawpt);
                    }

                }
            }

        }

    }

}
