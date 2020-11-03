#include "StudyEfficiency.h"

StudyEfficiency::StudyEfficiency(const char* studyName, StudyEfficiency::StudyEfficiencyMode mode_, std::vector<float> ptbounds)
{

    studyname = studyName;
    mode = mode_;
    switch (mode)
    {
        case kStudyEffAll: modename = "all"; break;
        case kStudyEffBarrel: modename = "barrel"; break;
        case kStudyEffBarrelFlat: modename = "barrelflat"; break;
        case kStudyEffBarrelTilt: modename = "barreltilt"; break;
        case kStudyEffBarrelTiltHighZ: modename = "barreltilthighz"; break;
        case kStudyEffBarrelTiltLowZ: modename = "barreltiltlowz"; break;
        case kStudyEffEndcap: modename = "endcap"; break;
        case kStudyEffEndcapPS: modename = "endcapPS"; break;
        case kStudyEffEndcap2S: modename = "endcap2S"; break;
        case kStudyEffEndcapPSCloseRing: modename = "endcapPSCloseRing"; break;
        case kStudyEffEndcapPSLowPt: modename = "endcapPSLowPt"; break;
        default: modename = "UNDEFINED"; break;
    }
    pt_boundaries = ptbounds;

}

void StudyEfficiency::bookStudy()
{
    // Book Histograms
    ana.histograms.addVecHistogram(TString::Format("md_%s_matched_track_pt", modename), pt_boundaries, [&]() { return md_matched_track_pt; } );
    ana.histograms.addVecHistogram(TString::Format("md_%s_all_track_pt", modename), pt_boundaries, [&]() { return md_all_track_pt; } );
    ana.histograms.addVecHistogram(TString::Format("md_%s_matched_track_eta", modename), 50, -4, 4, [&]() { return md_matched_track_eta; } );
    ana.histograms.addVecHistogram(TString::Format("md_%s_all_track_eta", modename), 50, -4, 4, [&]() { return md_all_track_eta; } );
    ana.histograms.addVecHistogram(TString::Format("md_%s_matched_track_phi", modename), 200, -3.1416, 3.1416, [&]() { return md_matched_track_phi; } );
    ana.histograms.addVecHistogram(TString::Format("md_%s_all_track_phi", modename), 200, -3.1416, 3.1416, [&]() { return md_all_track_phi; } );
    ana.histograms.addVecHistogram(TString::Format("md_%s_matched_track_z", modename), 200, -300, 300, [&]() { return md_matched_track_z; } );
    ana.histograms.addVecHistogram(TString::Format("md_%s_all_track_z", modename), 200, -300, 300, [&]() { return md_all_track_z; } );

    const int nlayers = NLAYERS;
    for (int ii = 0; ii < nlayers; ++ii)
    {
        ana.histograms.addVecHistogram(TString::Format("md_%s_matched_track_pt_by_layer%d", modename, ii), pt_boundaries, [&, ii]() { return md_matched_track_pt_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("md_%s_all_track_pt_by_layer%d", modename, ii), pt_boundaries, [&, ii]() { return md_all_track_pt_by_layer[ii]; } );
    }
    for (int ii = 0; ii < nlayers; ++ii)
    {
        ana.histograms.addVecHistogram(TString::Format("md_%s_matched_track_eta_by_layer%d", modename, ii), 50, -4, 4, [&, ii]() { return md_matched_track_eta_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("md_%s_all_track_eta_by_layer%d", modename, ii), 50, -4, 4, [&, ii]() { return md_all_track_eta_by_layer[ii]; } );
    }
    for (int ii = 0; ii < nlayers; ++ii)
    {
        ana.histograms.addVecHistogram(TString::Format("md_%s_matched_track_phi_by_layer%d", modename, ii), 200, -3.1416, 3.1416, [&, ii]() { return md_matched_track_phi_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("md_%s_all_track_phi_by_layer%d", modename, ii), 200, -3.1416, 3.1416, [&, ii]() { return md_all_track_phi_by_layer[ii]; } );
    }
    for (int ii = 0; ii < nlayers; ++ii)
    {
        ana.histograms.addVecHistogram(TString::Format("md_%s_matched_track_wrapphi_by_layer%d", modename, ii), 200, -3.1416, 3.1416, [&, ii]() { return md_matched_track_wrapphi_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("md_%s_all_track_wrapphi_by_layer%d", modename, ii), 200, -3.1416, 3.1416, [&, ii]() { return md_all_track_wrapphi_by_layer[ii]; } );
    }
    for (int ii = 0; ii < nlayers; ++ii)
    {
        ana.histograms.addVecHistogram(TString::Format("md_%s_matched_track_z_by_layer%d", modename, ii), 200, -300, 300, [&, ii]() { return md_matched_track_z_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("md_%s_all_track_z_by_layer%d", modename, ii), 200, -300, 300, [&, ii]() { return md_all_track_z_by_layer[ii]; } );
    }
    ana.histograms.addVecHistogram(TString::Format("md_%s_lower_hit_only_track_pt", modename), pt_boundaries, [&]() { return md_lower_hit_only_track_pt; } );
    ana.histograms.addVecHistogram(TString::Format("md_%s_lower_hit_only_track_eta", modename), 50, -4, 4, [&]() { return md_lower_hit_only_track_eta; } );
}

void StudyEfficiency::doStudy(SDL::Event& event, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents)
{
    // Each do study is performed per event

    // First clear all the output variables that will be used to fill the histograms for this event
    md_matched_track_pt.clear();
    md_all_track_pt.clear();
    md_matched_track_eta.clear();
    md_all_track_eta.clear();
    md_matched_track_phi.clear();
    md_all_track_phi.clear();
    md_matched_track_z.clear();
    md_all_track_z.clear();
    for (int ii = 0; ii < NLAYERS; ++ii)
    {
        md_matched_track_pt_by_layer[ii].clear();
        md_all_track_pt_by_layer[ii].clear();
        md_matched_track_eta_by_layer[ii].clear();
        md_all_track_eta_by_layer[ii].clear();
        md_matched_track_phi_by_layer[ii].clear();
        md_all_track_phi_by_layer[ii].clear();
        md_matched_track_wrapphi_by_layer[ii].clear();
        md_all_track_wrapphi_by_layer[ii].clear();
        md_matched_track_z_by_layer[ii].clear();
        md_all_track_z_by_layer[ii].clear();
    }
    md_lower_hit_only_track_pt.clear();
    md_lower_hit_only_track_eta.clear();

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
        trackmatch_by_layer[5] = false;

        // Loop over the lower modules that contains hits for this track
        for (auto& lowerModulePtr_Track : trackevent.getLowerModulePtrs())
        {

            // Depending on the mode, only run a subset of interested modules
            switch (mode)
            {
                case kStudyEffAll: /* do nothing */ break;
                case kStudyEffBarrel: if (not (lowerModulePtr_Track->subdet() == SDL::Module::Barrel)) { continue; } break;
                case kStudyEffBarrelFlat: if (not (lowerModulePtr_Track->subdet() == SDL::Module::Barrel and lowerModulePtr_Track->side() == SDL::Module::Center)) { continue; } break;
                case kStudyEffBarrelTilt: if (not (lowerModulePtr_Track->subdet() == SDL::Module::Barrel and lowerModulePtr_Track->side() != SDL::Module::Center)) { continue; } break;
                case kStudyEffBarrelTiltHighZ: if (not (lowerModulePtr_Track->subdet() == SDL::Module::Barrel and lowerModulePtr_Track->side() != SDL::Module::Center and ((lowerModulePtr_Track->rod() < 3 and lowerModulePtr_Track->side() == SDL::Module::NegZ) or (lowerModulePtr_Track->rod() > 10 and lowerModulePtr_Track->side() == SDL::Module::PosZ)))) { continue; } break;
                case kStudyEffBarrelTiltLowZ: if (not (lowerModulePtr_Track->subdet() == SDL::Module::Barrel and lowerModulePtr_Track->side() != SDL::Module::Center and ((lowerModulePtr_Track->rod() > 5 and lowerModulePtr_Track->side() == SDL::Module::NegZ) or (lowerModulePtr_Track->rod() < 8 and lowerModulePtr_Track->side() == SDL::Module::PosZ)))) { continue; } break;
                case kStudyEffEndcap: if (not (lowerModulePtr_Track->subdet() == SDL::Module::Endcap)) { continue; } break;
                case kStudyEffEndcapPS: if (not (lowerModulePtr_Track->subdet() == SDL::Module::Endcap and lowerModulePtr_Track->moduleType() == SDL::Module::PS)) { continue; } break;
                case kStudyEffEndcap2S: if (not (lowerModulePtr_Track->subdet() == SDL::Module::Endcap and lowerModulePtr_Track->moduleType() == SDL::Module::TwoS)) { continue; } break;
                case kStudyEffEndcapPSCloseRing: if (not (lowerModulePtr_Track->subdet() == SDL::Module::Endcap and lowerModulePtr_Track->moduleType() == SDL::Module::PS and lowerModulePtr_Track->ring() <= 1 and lowerModulePtr_Track->layer() < 3)) { continue; } break;
                case kStudyEffEndcapPSLowPt: if (not (lowerModulePtr_Track->subdet() == SDL::Module::Endcap and lowerModulePtr_Track->moduleType() == SDL::Module::PS and lowerModulePtr_Track->ring() <= 1 and lowerModulePtr_Track->layer() < 3 and trk.sim_pt()[isimtrk] > 1 and trk.sim_pt()[isimtrk] < 2)) { continue; } break;
                default: /* skip everything should not be here anyways...*/ continue; break;
            }

            // Parse the layer index later to be used for indexing
            int layer_idx = lowerModulePtr_Track->layer() - 1;

            // Parse pt and eta of this track
            float pt = std::min((double) trk.sim_pt()[isimtrk], 49.999);
            float eta = trk.sim_eta()[isimtrk];

            // For this module that a sim-track supposedly passed through if there are no more than 1 mini-doublets
            // it means that this track did not leave at least one hit each in each side of the module
            // The getMiniDoubletPtrs() will return ALL pairs of hits between the modules.
            // That's because the simtrkevent would have createMiniDoublets called with SDL::AllComb_MDAlgo option.
            // This option loops over all hits in lower module and upper module and forms every possible pair.
            // So if the following condition of size() == 0 is true, it means this sim-track didn't leave at least one hit in each side.
            if (lowerModulePtr_Track->getMiniDoubletPtrs().size() == 0)
            {
                md_lower_hit_only_track_pt.push_back(pt);
                md_lower_hit_only_track_pt.push_back(eta);
                continue;
            }

            // Boolean to test whether for this module that a track passed through, whether it found a matched mini-doublet
            bool match = false;

            // Loop over the md "candidate" from the module that a sim-track passed through and left at least one hit in each module
            float z = 0; // The z position of this "truth candidate" mini-doublet will be calculated by the average of all combos
            for (auto& md_Track : lowerModulePtr_Track->getMiniDoubletPtrs())
            {

                // To ge
                z += md_Track->lowerHitPtr()->z();

                // Loop over the md reconstructed from with proper SDL algorithm and if the index of the hits match
                // Then we have found at least one mini-doublet associated to this track's reco-hits in this module
                // Therefore flag the match boolean
                for (auto& md : event.getModule(lowerModulePtr_Track->detId()).getMiniDoubletPtrs())
                {
                    if (md_Track->isIdxMatched(*md))
                    {
                        match = true;
                    }
                }
            }

            if (lowerModulePtr_Track->getMiniDoubletPtrs().size() > 0)
                z /= lowerModulePtr_Track->getMiniDoubletPtrs().size();

            if (match)
                trackmatch_by_layer[layer_idx] = true;

        }

        // Parse pt and eta of this track
        float pt = std::min((double) trk.sim_pt()[isimtrk], 49.999);
        float eta = trk.sim_eta()[isimtrk];

        for (unsigned int i = 0; i < 6; ++i)
        {

            // Denominator for specific layers pt efficiency
            md_all_track_pt_by_layer[i].push_back(pt);

            // Denominator for specific layers eta efficiency (notice the 1 GeV cut)
            if (trk.sim_pt()[isimtrk] > 1.)
                md_all_track_eta_by_layer[i].push_back(eta);

            // Numerators
            if (trackmatch_by_layer[i])
            {
                // Numerators for matched specific layer pt effciency
                md_matched_track_pt_by_layer[i].push_back(pt);

                // Numerators for matched specific layer eta effciency (notice the 1 GeV cut)
                if (trk.sim_pt()[isimtrk] > 1.)
                    md_matched_track_eta_by_layer[i].push_back(eta);

            }

        }

    }

}

//======================================================================================================================
void StudyEfficiency::doStudy_v1(SDL::Event& event, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents)
{
    // Each do study is performed per event

    // First clear all the output variables that will be used to fill the histograms for this event
    md_matched_track_pt.clear();
    md_all_track_pt.clear();
    md_matched_track_eta.clear();
    md_all_track_eta.clear();
    md_matched_track_phi.clear();
    md_all_track_phi.clear();
    md_matched_track_z.clear();
    md_all_track_z.clear();
    for (int ii = 0; ii < NLAYERS; ++ii)
    {
        md_matched_track_pt_by_layer[ii].clear();
        md_all_track_pt_by_layer[ii].clear();
        md_matched_track_eta_by_layer[ii].clear();
        md_all_track_eta_by_layer[ii].clear();
        md_matched_track_phi_by_layer[ii].clear();
        md_all_track_phi_by_layer[ii].clear();
        md_matched_track_wrapphi_by_layer[ii].clear();
        md_all_track_wrapphi_by_layer[ii].clear();
        md_matched_track_z_by_layer[ii].clear();
        md_all_track_z_by_layer[ii].clear();
    }
    md_lower_hit_only_track_pt.clear();
    md_lower_hit_only_track_eta.clear();

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

            // Depending on the mode, only run a subset of interested modules
            switch (mode)
            {
                case kStudyEffAll: /* do nothing */ break;
                case kStudyEffBarrel: if (not (lowerModulePtr_Track->subdet() == SDL::Module::Barrel)) { continue; } break;
                case kStudyEffBarrelFlat: if (not (lowerModulePtr_Track->subdet() == SDL::Module::Barrel and lowerModulePtr_Track->side() == SDL::Module::Center)) { continue; } break;
                case kStudyEffBarrelTilt: if (not (lowerModulePtr_Track->subdet() == SDL::Module::Barrel and lowerModulePtr_Track->side() != SDL::Module::Center)) { continue; } break;
                case kStudyEffBarrelTiltHighZ: if (not (lowerModulePtr_Track->subdet() == SDL::Module::Barrel and lowerModulePtr_Track->side() != SDL::Module::Center and ((lowerModulePtr_Track->rod() < 3 and lowerModulePtr_Track->side() == SDL::Module::NegZ) or (lowerModulePtr_Track->rod() > 10 and lowerModulePtr_Track->side() == SDL::Module::PosZ)))) { continue; } break;
                case kStudyEffBarrelTiltLowZ: if (not (lowerModulePtr_Track->subdet() == SDL::Module::Barrel and lowerModulePtr_Track->side() != SDL::Module::Center and ((lowerModulePtr_Track->rod() > 5 and lowerModulePtr_Track->side() == SDL::Module::NegZ) or (lowerModulePtr_Track->rod() < 8 and lowerModulePtr_Track->side() == SDL::Module::PosZ)))) { continue; } break;
                case kStudyEffEndcap: if (not (lowerModulePtr_Track->subdet() == SDL::Module::Endcap)) { continue; } break;
                case kStudyEffEndcapPS: if (not (lowerModulePtr_Track->subdet() == SDL::Module::Endcap and lowerModulePtr_Track->moduleType() == SDL::Module::PS)) { continue; } break;
                case kStudyEffEndcap2S: if (not (lowerModulePtr_Track->subdet() == SDL::Module::Endcap and lowerModulePtr_Track->moduleType() == SDL::Module::TwoS)) { continue; } break;
                case kStudyEffEndcapPSCloseRing: if (not (lowerModulePtr_Track->subdet() == SDL::Module::Endcap and lowerModulePtr_Track->moduleType() == SDL::Module::PS and lowerModulePtr_Track->ring() <= 1 and lowerModulePtr_Track->layer() < 3)) { continue; } break;
                case kStudyEffEndcapPSLowPt: if (not (lowerModulePtr_Track->subdet() == SDL::Module::Endcap and lowerModulePtr_Track->moduleType() == SDL::Module::PS and lowerModulePtr_Track->ring() <= 1 and lowerModulePtr_Track->layer() < 3 and trk.sim_pt()[isimtrk] > 1 and trk.sim_pt()[isimtrk] < 2)) { continue; } break;
                default: /* skip everything should not be here anyways...*/ continue; break;
            }

            // Parse the layer index later to be used for indexing
            int layer_idx = lowerModulePtr_Track->layer() - 1;

            // Parse pt and eta of this track
            float pt = std::min((double) trk.sim_pt()[isimtrk], 49.999);
            float eta = trk.sim_eta()[isimtrk];
            float phi = trk.sim_phi()[isimtrk];

            // For this module that a sim-track supposedly passed through if there are no more than 1 mini-doublets
            // it means that this track did not leave at least one hit each in each side of the module
            // The getMiniDoubletPtrs() will return ALL pairs of hits between the modules.
            // That's because the simtrkevent would have createMiniDoublets called with SDL::AllComb_MDAlgo option.
            // This option loops over all hits in lower module and upper module and forms every possible pair.
            // So if the following condition of size() == 0 is true, it means this sim-track didn't leave at least one hit in each side.
            if (lowerModulePtr_Track->getMiniDoubletPtrs().size() == 0)
            {
                md_lower_hit_only_track_pt.push_back(pt);
                md_lower_hit_only_track_pt.push_back(eta);
                continue;
            }

            // Boolean to test whether for this module that a track passed through, whether it found a matched mini-doublet
            bool match = false;

            // Loop over the md "candidate" from the module that a sim-track passed through and left at least one hit in each module
            float z = 0; // The z position of this "truth candidate" mini-doublet will be calculated by the average of all combos
            for (auto& md_Track : lowerModulePtr_Track->getMiniDoubletPtrs())
            {

                // To ge
                z += md_Track->lowerHitPtr()->z();

                // Loop over the md reconstructed from with proper SDL algorithm and if the index of the hits match
                // Then we have found at least one mini-doublet associated to this track's reco-hits in this module
                // Therefore flag the match boolean
                for (auto& md : event.getModule(lowerModulePtr_Track->detId()).getMiniDoubletPtrs())
                {
                    if (md_Track->isIdxMatched(*md))
                    {
                        match = true;
                    }
                }
            }

            if (lowerModulePtr_Track->getMiniDoubletPtrs().size() > 0)
                z /= lowerModulePtr_Track->getMiniDoubletPtrs().size();

            // At this stage, we have either found a mini-doublet in this module matched to the track or not.

            // Denominator for all layers pt efficiency
            md_all_track_pt.push_back(pt);

            // Denominator for all layers eta efficiency (notice the 1 GeV cut)
            if (trk.sim_pt()[isimtrk] > 1.)
                md_all_track_eta.push_back(eta);

            // Denominator for all layers phi efficiency (notice the 1 GeV cut)
            if (trk.sim_pt()[isimtrk] > 1.)
                md_all_track_phi.push_back(phi);

            // Denominator for all layers z efficiency (notice the 1 GeV cut)
            if (trk.sim_pt()[isimtrk] > 1.)
                md_all_track_z.push_back(z);

            // Numerators
            if (match)
            {

                // Numerators for matched all layers pt efficiency
                md_matched_track_pt.push_back(pt);

                // Numeratosr for matched all layers eta efficiency (notice the 1 GeV cut)
                if (trk.sim_pt()[isimtrk] > 1.)
                    md_matched_track_eta.push_back(eta);

                // Numeratosr for matched all layers phi efficiency (notice the 1 GeV cut)
                if (trk.sim_pt()[isimtrk] > 1.)
                    md_matched_track_phi.push_back(phi);

                // Numeratosr for matched all layers z efficiency (notice the 1 GeV cut)
                if (trk.sim_pt()[isimtrk] > 1.)
                    md_matched_track_z.push_back(z);

            }
            // Failed tracks for all layers
            else
            {
                // Doing nothing for now ...
            }

            // Denominator for specific layers pt efficiency
            md_all_track_pt_by_layer[layer_idx].push_back(pt);

            // Denominator for specific layers eta efficiency (notice the 1 GeV cut)
            if (trk.sim_pt()[isimtrk] > 1.)
                md_all_track_eta_by_layer[layer_idx].push_back(eta);

            // Denominator for specific layers phi efficiency (notice the 1 GeV cut)
            if (trk.sim_pt()[isimtrk] > 1.)
                md_all_track_phi_by_layer[layer_idx].push_back(phi);

            // Denominator for specific layers z efficiency (notice the 1 GeV cut)
            if (trk.sim_pt()[isimtrk] > 1.)
                md_all_track_z_by_layer[layer_idx].push_back(z);

            // Denominator for specific layers wrapphi efficiency (notice the 1 GeV cut)
            float wrapphi = 0;
            if (layer_idx == 0)
                wrapphi = fmod(fabs(phi), 2*TMath::Pi() / 18);
            else if (layer_idx == 1)
                wrapphi = fmod(fabs(phi), 2*TMath::Pi() / 26);
            else
                wrapphi = fmod(fabs(phi), 2*TMath::Pi() / 36);
            if (trk.sim_pt()[isimtrk] > 5.)
                md_all_track_wrapphi_by_layer[layer_idx].push_back(wrapphi);

            // Numerators
            if (match)
            {
                // Numerators for matched specific layer pt effciency
                md_matched_track_pt_by_layer[layer_idx].push_back(pt);

                // Numerators for matched specific layer eta effciency (notice the 1 GeV cut)
                if (trk.sim_pt()[isimtrk] > 1.)
                    md_matched_track_eta_by_layer[layer_idx].push_back(eta);

                // Numerators for matched specific layer phi effciency (notice the 1 GeV cut)
                if (trk.sim_pt()[isimtrk] > 1.)
                    md_matched_track_phi_by_layer[layer_idx].push_back(phi);

                // Numerators for matched specific layer wrapphi effciency (notice the 1 GeV cut)
                if (trk.sim_pt()[isimtrk] > 5.)
                    md_matched_track_wrapphi_by_layer[layer_idx].push_back(wrapphi);

                // Numerators for matched specific layer z effciency (notice the 1 GeV cut)
                if (trk.sim_pt()[isimtrk] > 1.)
                    md_matched_track_z_by_layer[layer_idx].push_back(z);

            }
            // Failed tracks for specific layers
            else
            {
            }

        }

    }

}
