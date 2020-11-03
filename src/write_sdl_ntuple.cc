#include "write_sdl_ntuple.h"

void write_sdl_ntuple()
{
    // List of studies to perform
    std::vector<Study*> studies;

    // pt_boundaries
    std::vector<float> pt_boundaries = getPtBounds();

    studies.push_back(new WriteSDLNtuplev2("WriteSDLNtuple"));

    // book the studies
    for (auto& study : studies)
    {
        study->bookStudy();
    }

    // Book Histograms
    ana.cutflow.bookHistograms(ana.histograms); // if just want to book everywhere

    // Load various maps used in the SDL reconstruction
    loadMaps();
    SDL::initModules();
    // Looping input file
    while (ana.looper.nextEvent())
    {
        std::cout<<"event number = "<<ana.looper.getCurrentEventIndex()<<std::endl;

        if (not goodEvent())
            continue;

        // Main instance that will hold modules, hits, minidoublets, etc. (i.e. main data structure)
        SDL::Event event;

        // Add hits to the event
        addOuterTrackerHits(event);

        // Add pixel segments
        //addPixelSegments(event);

        // Print hit summary
//        printHitSummary(event);

        // Run Mini-doublet
        runMiniDoublet(event);

        // Run Segment
        runSegment(event);

        // Run Tracklet
        runTracklet(event);

        // Run Triplet
        runTriplet(event);
//        printTripletSummary(event);

        // Run TrackCandidate
//        runTrackCandidateTest_v2(event);
//        printTrackCandidateSummary(event);

        // *********************************************************************
        // SDL::Event from each sim track (using only hits from given sim track)
        // *********************************************************************

        // Each SDL::Event object in simtrkevents will hold single sim-track related hits
        // It will be a vector of tuple of <sim_track_index, SDL::Event*>.
        std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents;
        std::vector<std::tuple<unsigned int, SDL::EventForAnalysisInterface*>> simtrkeventsForAnalysisInterface;

        // Loop over sim-tracks that is from in time (bx = 0) tracks with pdgid matching (against ana.pdg_id) and per sim-track aggregate reco hits
        // and only use those hits, and run SDL on them
        for (auto&& isimtrk : iter::filter([&](int i) { return inTimeTrackWithPdgId(i, ana.pdg_id); }, iter::range(trk.sim_pt().size())))
        {

            // event just for this track
            SDL::Event* trackevent = new SDL::Event();

            // // Add reco hits associated with the given sim track to the SDL::Event container
            // addOuterTrackerHitsFromSimTrack((*trackevent), isimtrk);

            // // Run SDL reconstruction on the event
            // runSDL((*trackevent));

            // Push to the vector so we have a data-base of per hit, mini-doublets
            simtrkevents.push_back(std::make_tuple(isimtrk, trackevent));
            SDL::EventForAnalysisInterface* trackeventForAnalysisInterface = new SDL::EventForAnalysisInterface(SDL::modulesInGPU, trackevent->getHits(), trackevent->getMiniDoublets(), trackevent->getSegments(), trackevent->getTracklets(), trackevent->getTriplets());
            simtrkeventsForAnalysisInterface.push_back(std::make_tuple(isimtrk,trackeventForAnalysisInterface));

        }


        // ********************************************************************************************
        // Perform various studies with reco events and sim-track-matched-reco-hits-based mini-doublets
        // ********************************************************************************************
        //analysis interface
        SDL::EventForAnalysisInterface* eventForAnalysisInterface = new SDL::EventForAnalysisInterface(SDL::modulesInGPU, event.getHits(), event.getMiniDoublets(), event.getSegments(), event.getTracklets(), event.getTriplets());
        for (auto& study : studies)
        {
            study->doStudy(*eventForAnalysisInterface, simtrkeventsForAnalysisInterface);
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
