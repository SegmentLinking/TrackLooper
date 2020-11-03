#include "StudyMiniDoubletOccupancy.h"

StudyMiniDoubletOccupancy::StudyMiniDoubletOccupancy(const char* studyName, StudyMiniDoubletOccupancy::StudyMiniDoubletOccupancyMode mode_)
{

    studyname = studyName;
    mode = mode_;
    switch (mode)
    {
        case kStudyAll: modename = "all"; break;
        default: modename = "UNDEFINED"; break;
    }

}

void StudyMiniDoubletOccupancy::bookStudy()
{
    // Book Histograms
    const int nlayers = NLAYERS;
    for (int ii = 0; ii < nlayers; ++ii)
    {
        ana.histograms.addHistogram(TString::Format("n_md_lower_by_layer%d", ii), 56, 0, 14000, [&, ii]() { return n_in_lower_modules_by_layer[ii]; } );
        ana.histograms.addHistogram(TString::Format("n_md_upper_by_layer%d", ii), 56, 0, 14000, [&, ii]() { return n_in_upper_modules_by_layer[ii]; } );
        ana.histograms.addHistogram(TString::Format("n_md_both_by_layer%d", ii),112, 0, 28000, [&, ii]() { return n_in_both_modules_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("dz_md_lower_by_layer%d", ii), 180, -5, 5, [&, ii]() { return dz_lower_modules_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("dz_md_upper_by_layer%d", ii), 180, -5, 5, [&, ii]() { return dz_upper_modules_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("dz_md_both_by_layer%d", ii), 180, -5, 5, [&, ii]() { return dz_both_modules_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("n_crosser_md_lower_by_layer%d", ii), 3, -1, 2, [&, ii]() { return n_cross_lower_modules_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("n_crosser_md_upper_by_layer%d", ii), 3, -1, 2, [&, ii]() { return n_cross_upper_modules_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("n_crosser_md_both_by_layer%d", ii), 3, -1, 2, [&, ii]() { return n_cross_both_modules_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("dz_true_md_lower_by_layer%d", ii), 180, -5, 5, [&, ii]() { return true_dz_lower_modules_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("dz_true_md_upper_by_layer%d", ii), 180, -5, 5, [&, ii]() { return true_dz_upper_modules_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("dz_true_md_both_by_layer%d", ii), 180, -5, 5, [&, ii]() { return true_dz_both_modules_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("n_true_crosser_md_lower_by_layer%d", ii), 3, -1, 2, [&, ii]() { return n_true_cross_lower_modules_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("n_true_crosser_md_upper_by_layer%d", ii), 3, -1, 2, [&, ii]() { return n_true_cross_upper_modules_by_layer[ii]; } );
        ana.histograms.addVecHistogram(TString::Format("n_true_crosser_md_both_by_layer%d", ii), 3, -1, 2, [&, ii]() { return n_true_cross_both_modules_by_layer[ii]; } );
    }

}

void StudyMiniDoubletOccupancy::doStudy(SDL::Event& event, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents)
{

    // Each do study is performed per event

    // First clear all the output variables that will be used to fill the histograms for this event
    for (int ii = 0; ii < NLAYERS; ++ii)
    {
        n_in_lower_modules_by_layer[ii] = 0;
        n_in_upper_modules_by_layer[ii] = 0;
        n_in_both_modules_by_layer[ii] = 0;
        dz_lower_modules_by_layer[ii].clear();
        dz_upper_modules_by_layer[ii].clear();
        dz_both_modules_by_layer[ii].clear();
        n_cross_lower_modules_by_layer[ii].clear();
        n_cross_upper_modules_by_layer[ii].clear();
        n_cross_both_modules_by_layer[ii].clear();
        true_dz_lower_modules_by_layer[ii].clear();
        true_dz_upper_modules_by_layer[ii].clear();
        true_dz_both_modules_by_layer[ii].clear();
        n_true_cross_lower_modules_by_layer[ii].clear();
        n_true_cross_upper_modules_by_layer[ii].clear();
        n_true_cross_both_modules_by_layer[ii].clear();
    }

    //***********************
    // Studying Hit Occupancy
    //***********************

    std::vector<SDL::Module*> moduleList = event.getModulePtrs();

    // Loop over tracklets in event
    for (auto& modulePtr : moduleList)
    {
        int subdet = modulePtr->subdet();
        if (subdet != 5)
            continue;

        // Get mini-doublet pointers
        const std::vector<SDL::MiniDoublet*>& mdPtrs = modulePtr->getMiniDoubletPtrs();

        // Get the nhit (i.e. minidoublerts)
        int nhit_in_module = mdPtrs.size();

        // Get the layer index
        int ilayer = modulePtr->layer();

        // Is Lower
        int isLower = modulePtr->isLower();

        // Add to the counter
        if (isLower)
            n_in_lower_modules_by_layer[ilayer-1] += nhit_in_module;
        else
            n_in_upper_modules_by_layer[ilayer-1] += nhit_in_module;
        n_in_both_modules_by_layer[ilayer-1] += nhit_in_module;

        for (auto& mdPtr : mdPtrs)
        {

            const float& dz = mdPtr->getDz();
            const float sign = ((dz > 0) - (dz < 0)) * ((mdPtr->lowerHitPtr()->z() > 0) - (mdPtr->lowerHitPtr()->z() < 0));

            // Add to the vector
            if (isLower)
                dz_lower_modules_by_layer[ilayer-1].push_back(dz);
            else
                dz_upper_modules_by_layer[ilayer-1].push_back(dz);
            dz_both_modules_by_layer[ilayer-1].push_back(dz);

            // Add to the vector
            if (isLower)
                n_cross_lower_modules_by_layer[ilayer-1].push_back((abs(dz) > 2) * sign);
            else
                n_cross_upper_modules_by_layer[ilayer-1].push_back((abs(dz) > 2) * sign);
            n_cross_both_modules_by_layer[ilayer-1].push_back((abs(dz) > 2) * sign);

        }

    }

    //*******************************
    // Studying Hit Occupancy (Truth)
    //*******************************

    // Loop over sim events
    for (auto& simtrkevent : simtrkevents)
    {
        // Unpack the tuple (sim_track_index, SDL::Event containing reco hits only matched to the given sim track)
        unsigned int& isimtrk = std::get<0>(simtrkevent);
        SDL::Event& trackevent = *(std::get<1>(simtrkevent));

        std::vector<SDL::Module*> moduleList = trackevent.getModulePtrs();

        // Loop over tracklets in event
        for (auto& modulePtr : moduleList)
        {
            int subdet = modulePtr->subdet();
            if (subdet != 5)
                continue;

            // Get mini-doublet pointers
            const std::vector<SDL::MiniDoublet*>& mdPtrs = modulePtr->getMiniDoubletPtrs();

            // Get the layer index
            int ilayer = modulePtr->layer();

            // Is Lower
            int isLower = modulePtr->isLower();

            for (auto& mdPtr : mdPtrs)
            {

                const float& dz = mdPtr->getDz();
                const float sign = ((dz > 0) - (dz < 0)) * ((mdPtr->lowerHitPtr()->z() > 0) - (mdPtr->lowerHitPtr()->z() < 0));

                // Add to the vector
                if (isLower)
                    true_dz_lower_modules_by_layer[ilayer-1].push_back(dz);
                else
                    true_dz_upper_modules_by_layer[ilayer-1].push_back(dz);
                true_dz_both_modules_by_layer[ilayer-1].push_back(dz);

                // Add to the vector
                if (isLower)
                    n_true_cross_lower_modules_by_layer[ilayer-1].push_back((abs(dz) > 2) * sign);
                else
                    n_true_cross_upper_modules_by_layer[ilayer-1].push_back((abs(dz) > 2) * sign);
                n_true_cross_both_modules_by_layer[ilayer-1].push_back((abs(dz) > 2) * sign);

            }

        }

    }

    // First clear all the output variables that will be used to fill the histograms for this event
    for (int ii = 0; ii < NLAYERS; ++ii)
    {
        // std::cout <<  " n_in_lower_modules_by_layer[ii]: " << n_in_lower_modules_by_layer[ii] <<  std::endl;
        // std::cout <<  " n_in_upper_modules_by_layer[ii]: " << n_in_upper_modules_by_layer[ii] <<  std::endl;
        // std::cout <<  " n_in_both_modules_by_layer[ii]: " << n_in_both_modules_by_layer[ii] <<  std::endl;
        // std::cout <<  " dz_lower_modules_by_layer[ii].size(): " << dz_lower_modules_by_layer[ii].size() <<  std::endl;
        // for (auto& dz : dz_lower_modules_by_layer[ii])
        //     std::cout <<  " dz: " << dz <<  std::endl;
    }

}
