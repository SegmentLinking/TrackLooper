#include "StudyMiniDoubletEfficiency.h"

StudyMiniDoubletEfficiency::StudyMiniDoubletEfficiency(const char* studyName, StudyMiniDoubletEfficiency::StudyMiniDoubletEfficiencyMode mode_, std::vector<float> ptbounds)
{

    studyname = studyName;
    mode = mode_;
    switch (mode)
    {
        case kStudyEffAll: modename = "specific"; break;
        default: modename = "UNDEFINED"; break;
    }
    pt_boundaries = ptbounds;

}

void StudyMiniDoubletEfficiency::bookStudy()
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

void StudyMiniDoubletEfficiency::doStudy(SDL::Event& event, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents)
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

        // Parse pt and eta of this track
        float pt = std::min((double) trk.sim_pt()[isimtrk], 49.999);
        float eta = trk.sim_eta()[isimtrk];
        float phi = trk.sim_phi()[isimtrk];

        bool match5 = false;
        bool match6 = false;

        // Loop over the lower modules that contains hits for this track
        bool has5barrel = false;
        bool has6barrel = false;
        for (auto& lowerModulePtr_Track : trackevent.getLowerModulePtrs())
        {

            // Parse the layer index later to be used for indexing
            int layer_idx = lowerModulePtr_Track->layer();
            bool isbarrel = lowerModulePtr_Track->subdet() == SDL::Module::Barrel;

            if (isbarrel and layer_idx == 6)
            {
                has6barrel = true;

                // Loop over the md reconstructed from with proper SDL algorithm and if the index of the hits match
                // Then we have found at least one mini-doublet associated to this track's reco-hits in this module
                // Therefore flag the match boolean
                for (auto& md_Track : lowerModulePtr_Track->getMiniDoubletPtrs())
                {
                    for (auto& md : event.getModule(lowerModulePtr_Track->detId()).getMiniDoubletPtrs())
                    {
                        if (md_Track->isIdxMatched(*md))
                        {
                            match6 = true;
                        }
                    }
                }

            }
            if (isbarrel and layer_idx == 5)
            {
                has5barrel = true;

                // Loop over the md reconstructed from with proper SDL algorithm and if the index of the hits match
                // Then we have found at least one mini-doublet associated to this track's reco-hits in this module
                // Therefore flag the match boolean
                for (auto& md_Track : lowerModulePtr_Track->getMiniDoubletPtrs())
                {
                    for (auto& md : event.getModule(lowerModulePtr_Track->detId()).getMiniDoubletPtrs())
                    {
                        if (md_Track->isIdxMatched(*md))
                        {
                            match5 = true;
                        }
                    }
                }

            }

        }

        if (not has5barrel or not has6barrel)
            continue;

        if (has5barrel)
        {
            if (match5)
                md_matched_track_pt_by_layer[4].push_back(pt);
            md_all_track_pt_by_layer[4].push_back(pt);
        }

        if (has6barrel)
        {
            if (match6)
                md_matched_track_pt_by_layer[5].push_back(pt);
            md_all_track_pt_by_layer[5].push_back(pt);
        }

    }

}
