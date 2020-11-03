#include "algo_eff.h"

void algo_eff()
{
    // List of studies to perform
    std::vector<Study*> studies;

    // pt_boundaries
    std::vector<float> pt_boundaries = getPtBounds();

    studies.push_back(new StudySDLEfficiencyv2("eff_bbbbbb", StudySDLEfficiencyv2::kStudySDLBBBBBB, pt_boundaries));
    studies.push_back(new StudySDLEfficiencyv2("eff_bbbbbe", StudySDLEfficiencyv2::kStudySDLBBBBBE, pt_boundaries));
    studies.push_back(new StudySDLEfficiencyv2("eff_bbbbee", StudySDLEfficiencyv2::kStudySDLBBBBEE, pt_boundaries));
    studies.push_back(new StudySDLEfficiencyv2("eff_bbbeee", StudySDLEfficiencyv2::kStudySDLBBBEEE, pt_boundaries));
    studies.push_back(new StudySDLEfficiencyv2("eff_bbeeee", StudySDLEfficiencyv2::kStudySDLBBEEEE, pt_boundaries));
    studies.push_back(new StudySDLEfficiencyv2("eff_beeeee", StudySDLEfficiencyv2::kStudySDLBEEEEE, pt_boundaries));

    // book the studies
    for (auto& study : studies)
    {
        study->bookStudy();
    }

    // Book Histograms
    ana.cutflow.bookHistograms(ana.histograms); // if just want to book everywhere

    // Load various maps used in the SDL reconstruction
    loadMaps();

    // Looping input file
    while (ana.looper.nextEvent())
    {

        if (not goodEvent())
            continue;

        // *************************************************
        // Reconstructed hits and formation of mini-doublets
        // *************************************************

        // Main instance that will hold modules, hits, minidoublets, etc. (i.e. main data structure)
        SDL::Event event;

        // Add hits to the event
        addOuterTrackerHits(event);

        // Run the SDL reconstruction on the event
        runSDL(event);

        // *********************************************************************
        // SDL::Event from each sim track (using only hits from given sim track)
        // *********************************************************************

        // Each SDL::Event object in simtrkevents will hold single sim-track related hits
        // It will be a vector of tuple of <sim_track_index, SDL::Event*>.
        std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents;

        // Loop over sim-tracks that is from in time (bx = 0) tracks with pdgid matching (against ana.pdg_id) and per sim-track aggregate reco hits
        // and only use those hits, and run SDL on them
        for (auto&& isimtrk : iter::filter([&](int i) { return inTimeTrackWithPdgId(i, ana.pdg_id); }, iter::range(trk.sim_pt().size())))
        {

            // event just for this track
            SDL::Event* trackevent = new SDL::Event();

            // Add reco hits associated with the given sim track to the SDL::Event container
            addOuterTrackerHitsFromSimTrack((*trackevent), isimtrk);

            // Run SDL reconstruction on the event
            runSDL((*trackevent));

            // Push to the vector so we have a data-base of per hit, mini-doublets
            simtrkevents.push_back(std::make_tuple(isimtrk, trackevent));

        }


        // ********************************************************************************************
        // Perform various studies with reco events and sim-track-matched-reco-hits-based mini-doublets
        // ********************************************************************************************

        for (auto& study : studies)
        {
            study->doStudy(event, simtrkevents);
        }


        // ************************************************
        // Now fill all the histograms booked by each study
        // ************************************************

        // Fill all the histograms
        ana.cutflow.fill();

        // <--------------------------
        // <--------------------------
        // <--------------------------
    }

    // Writing output file
    ana.cutflow.saveOutput();

    // Writing ttree output to file
    ana.output_ttree->Write();

    // The below can be sometimes crucial
    delete ana.output_tfile;

}
