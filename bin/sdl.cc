#include "sdl.h"

// ./process INPUTFILEPATH OUTPUTFILE [NEVENTS]
//_______________________________________________________________________________
int main(int argc, char** argv)
{

//********************************************************************************
//
// 0. Preliminary operations
//
//********************************************************************************

    // Checking the TRACKLOOPERDIR is set
    ana.track_looper_dir_path = gSystem->Getenv("TRACKLOOPERDIR");
    if (ana.track_looper_dir_path.IsNull())
    {
        RooUtil::error("TRACKLOOPERDIR is not set! Did you run $ source setup.sh from TrackLooper/ main repository directory?");
    }
    RooUtil::print(TString::Format("TRACKLOOPERDIR=%s", ana.track_looper_dir_path.Data()));

    // Write the command line used to run it
    // N.B. This needs to be before the argument parsing as it will change some values
    std::vector<std::string> allArgs(argv, argv + argc);
    ana.full_cmd_line = "";
    for (auto& str : allArgs)
    {
        ana.full_cmd_line += TString::Format(" %s", str.c_str());
    }

//********************************************************************************
//
// 1. Parsing options
//
//********************************************************************************

    // cxxopts is just a tool to parse argc, and argv easily

    // Grand option setting
    cxxopts::Options options("\n  $ sdl",  "\n         **********************\n         *                    *\n         *       Looper       *\n         *                    *\n         **********************\n");

    // Read the options
    options.add_options()
        ("m,mode"           , "Run mode (NOT DEFINED)", cxxopts::value<int>()->default_value("5"))
        ("i,input"          , "Comma separated input file list OR if just a directory is provided it will glob all in the directory BUT must end with '/' for the path", cxxopts::value<std::string>()->default_value("muonGun"))
        ("t,tree"           , "Name of the tree in the root file to open and loop over"                                             , cxxopts::value<std::string>()->default_value("trackingNtuple/tree"))
        ("o,output"         , "Output file name"                                                                                    , cxxopts::value<std::string>())
        ("N,nmatch"         , "N match for MTV-like matching"                                                                       , cxxopts::value<int>()->default_value("9"))
        ("n,nevents"        , "N events to loop over"                                                                               , cxxopts::value<int>()->default_value("-1"))
        ("x,event_index"    , "specific event index to process"                                                                     , cxxopts::value<int>()->default_value("-1"))
        ("g,pdg_id"         , "The simhit pdgId match option (default = 0)"                                                         , cxxopts::value<int>()->default_value("0"))
        ("v,verbose"        , "Verbose mode (0: no print, 1: only final timing, 2: object multiplitcity"                            , cxxopts::value<int>()->default_value("0"))
        ("w,write_ntuple"   , "Write Ntuple"                                                                                        , cxxopts::value<int>()->default_value("1"))
        ("d,debug"          , "Run debug job. i.e. overrides output option to 'debug.root' and 'recreate's the file.")
        ("c,cpu"            , "Run CPU version of the code.")
        ("p,optimization"   , "write cut optimization ntuple")
        ("l,lower_level"    , "write lower level objects ntuple results")
        ("j,nsplit_jobs"    , "Enable splitting jobs by N blocks (--job_index must be set)", cxxopts::value<int>())
        ("I,job_index"      , "job_index of split jobs (--nsplit_jobs must be set. index starts from 0. i.e. 0, 1, 2, 3, etc...)", cxxopts::value<int>())
        ("h,help"           , "Print help")
        ;

    auto result = options.parse(argc, argv);

    // NOTE: When an option was provided (e.g. -i or --input), then the result.count("<option name>") is more than 0
    // Therefore, the option can be parsed easily by asking the condition if (result.count("<option name>");
    // That's how the several options are parsed below

    //_______________________________________________________________________________
    // --help
    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        exit(1);
    }

    //_______________________________________________________________________________
    // --input
    ana.input_raw_string = result["input"].as<std::string>();

    // A default value one
    if (ana.input_raw_string.EqualTo("muonGun"))
        ana.input_file_list_tstring = "/data2/segmentlinking/trackingNtuple_100_pt0p5_2p0.root";
    else if (ana.input_raw_string.EqualTo("muonGun_highE"))
        ana.input_file_list_tstring = "/data2/segmentlinking/trackingNtuple_10MuGun.root";
    else if (ana.input_raw_string.EqualTo("pionGun"))
        ana.input_file_list_tstring = "/data2/segmentlinking/trackingNtuple_1pion_10k_pt0p5_50p0.root";
    else if (ana.input_raw_string.EqualTo("PU200"))
        ana.input_file_list_tstring = "/data2/segmentlinking/trackingNtuple_with_PUinfo_500_evts.root";
    else if (ana.input_raw_string.EqualTo("cube"))
        ana.input_file_list_tstring = "/data2/segmentlinking/trackingNtuple_10_pt0p5_50_5cm_cube.root";
    else if (ana.input_raw_string.EqualTo("cube50cm"))
        ana.input_file_list_tstring = "/data2/segmentlinking/trackingNtuple_10_pt0p5_50_50cm_cube.root";
    else
        ana.input_file_list_tstring = ana.input_raw_string;

    //_______________________________________________________________________________
    // --tree
    ana.input_tree_name = result["tree"].as<std::string>();

    //_______________________________________________________________________________
    // --debug
    if (result.count("debug"))
    {
        ana.output_tfile = new TFile("debug.root", "recreate");
    }
    else
    {
        //_______________________________________________________________________________
        // --output
        if (result.count("output"))
        {
            ana.output_tfile = new TFile(result["output"].as<std::string>().c_str(), "create");
            if (not ana.output_tfile->IsOpen())
            {
                std::cout << options.help() << std::endl;
                std::cout << "ERROR: output already exists! provide new output name or delete old file. OUTPUTFILE=" << result["output"].as<std::string>() << std::endl;
                exit(1);
            }
        }
        else
        {
            std::cout << "Warning: Output file name is not provided! Check your arguments. Output file will be set to 'debug.root'" << std::endl;
            ana.output_tfile = new TFile("debug.root", "recreate");
        }
    }

    //_______________________________________________________________________________
    // --nmatch
    ana.nmatch_threshold = result["nmatch"].as<int>();

    //_______________________________________________________________________________
    // --nevents
    ana.n_events = result["nevents"].as<int>();
    ana.specific_event_index = result["event_index"].as<int>();

    //_______________________________________________________________________________
    // --pdg_id
    ana.pdg_id = result["pdg_id"].as<int>();

    //_______________________________________________________________________________
    // --nsplit_jobs
    if (result.count("nsplit_jobs"))
    {
        ana.nsplit_jobs = result["nsplit_jobs"].as<int>();
        if (ana.nsplit_jobs <= 0)
        {
            std::cout << options.help() << std::endl;
            std::cout << "ERROR: option string --nsplit_jobs" << ana.nsplit_jobs << " has zero or negative value!" << std::endl;
            std::cout << "I am not sure what this means..." << std::endl;
            exit(1);
        }
    }
    else
    {
        ana.nsplit_jobs = -1;
    }

    //_______________________________________________________________________________
    // --job_index
    if (result.count("job_index"))
    {
        ana.job_index = result["job_index"].as<int>();
        if (ana.job_index < 0)
        {
            std::cout << options.help() << std::endl;
            std::cout << "ERROR: option string --job_index" << ana.job_index << " has negative value!" << std::endl;
            std::cout << "I am not sure what this means..." << std::endl;
            exit(1);
        }
    }
    else
    {
        ana.job_index = -1;
    }

    // Sanity check for split jobs (if one is set the other must be set too)
    if (result.count("job_index") or result.count("nsplit_jobs"))
    {
        // If one is not provided then throw error
        if ( not (result.count("job_index") and result.count("nsplit_jobs")))
        {
            std::cout << options.help() << std::endl;
            std::cout << "ERROR: option string --job_index and --nsplit_jobs must be set at the same time!" << std::endl;
            exit(1);
        }
        // If it is set then check for sanity
        else
        {
            if (ana.job_index >= ana.nsplit_jobs)
            {
                std::cout << options.help() << std::endl;
                std::cout << "ERROR: --job_index >= --nsplit_jobs ! This does not make sense..." << std::endl;
                exit(1);
            }
        }
    }

    //_______________________________________________________________________________
    // --verbose
    ana.verbose = result["verbose"].as<int>();

    //_______________________________________________________________________________
    // --mode
    ana.mode = result["mode"].as<int>();

    //_______________________________________________________________________________
    // --write_ntuple
    ana.do_write_ntuple = result["write_ntuple"].as<int>();

    //_______________________________________________________________________________
    // --cpu
    if (result.count("cpu"))
    {
        ana.do_run_cpu = true;
    }
    else
    {
        ana.do_run_cpu = false;
    }

    //_______________________________________________________________________________
    // --optimization
    if (result.count("optimization"))
    {
        ana.do_cut_value_ntuple = true;
    }
    else
    {
        ana.do_cut_value_ntuple = false;
    }

    //_______________________________________________________________________________
    // --lower_level
#ifdef CUT_VALUE_DEBUG
    ana.do_lower_level = true;
#else
    if (result.count("lower_level"))
    {
        ana.do_lower_level = true;
    }
    else
    {
        ana.do_lower_level = false;
    }
#endif

    // Printing out the option settings overview
    std::cout <<  "=========================================================" << std::endl;
    std::cout <<  " Setting of the analysis job based on provided arguments " << std::endl;
    std::cout <<  "---------------------------------------------------------" << std::endl;
    std::cout <<  " ana.input_file_list_tstring: " << ana.input_file_list_tstring <<  std::endl;
    std::cout <<  " ana.output_tfile: " << ana.output_tfile->GetName() <<  std::endl;
    std::cout <<  " ana.n_events: " << ana.n_events <<  std::endl;
    std::cout <<  " ana.nsplit_jobs: " << ana.nsplit_jobs <<  std::endl;
    std::cout <<  " ana.job_index: " << ana.job_index <<  std::endl;
    std::cout <<  " ana.specific_event_index: " << ana.specific_event_index <<  std::endl;
    std::cout <<  " ana.do_cut_value_ntuple: " << ana.do_cut_value_ntuple <<  std::endl;
    std::cout <<  " ana.do_run_cpu: " << ana.do_run_cpu <<  std::endl;
    std::cout <<  " ana.do_write_ntuple: " << ana.do_write_ntuple <<  std::endl;
    std::cout <<  " ana.mode: " << ana.mode <<  std::endl;
    std::cout <<  " ana.verbose: " << ana.verbose <<  std::endl;
    std::cout <<  " ana.nmatch_threshold: " << ana.nmatch_threshold <<  std::endl;
    std::cout <<  "=========================================================" << std::endl;

    // Create the TChain that holds the TTree's of the baby ntuples
    ana.events_tchain = RooUtil::FileUtil::createTChain(ana.input_tree_name, ana.input_file_list_tstring);
    ana.looper.init(ana.events_tchain, &trk, ana.n_events);

    // Set the cutflow object output file
    ana.cutflow.setTFile(ana.output_tfile);

    // Create ttree to output to the ana.output_tfile
    ana.output_ttree = new TTree("tree", "tree");

    // Create TTreeX instance that will take care of the interface part of TTree
    ana.tx = new RooUtil::TTreeX(ana.output_ttree);

    // Write metadata related to this run
    writeMetaData();

    // Run the code
    run_sdl();

    return 0;
}

//________________________________________________________________________________________________________________________________
void run_sdl()
{

    // Load various maps used in the SDL reconstruction
    loadMaps();
    Study* study;

    if (not ana.do_run_cpu)
        SDL::initModules(TString::Format("%s/data/centroid.txt", gSystem->Getenv("TRACKLOOPERDIR")));

    if (not ana.do_cut_value_ntuple or ana.do_run_cpu)
    {
        createOutputBranches();
    }
    else
    {
        //call the function from WriteSDLNtuplev2.cc
        study = new WriteSDLNtuplev2("WriteSDLNtuple");
        study->bookStudy();
        ana.cutflow.bookHistograms(ana.histograms);
    }

    // Timing average information
    std::vector<std::vector<float>> timing_information;

    // Looping input file
    while (ana.looper.nextEvent())
    {

        std::cout << "event number = " << ana.looper.getCurrentEventIndex() << std::endl;

        if (not goodEvent())
            continue;

        if (not ana.do_run_cpu)
        {
            //*******************************************************
            // GPU VERSION RUN
            //*******************************************************

            // Main instance that will hold modules, hits, minidoublets, etc. (i.e. main data structure)
            SDL::Event event;

            // Add hits to the event
            float timing_input_loading = 0;
            if (ana.compilation_target.find("explicit") != std::string::npos)
                timing_input_loading = addInputsToLineSegmentTrackingUsingExplicitMemory(event);
            else
                timing_input_loading = addInputsToLineSegmentTrackingUsingExplicitMemory(event);
                //timing_input_loading = addInputsToLineSegmentTrackingUsingUnifiedMemory(event);

            // Run Mini-doublet
            float timing_MD = runMiniDoublet(event);

            // Run Segment
            float timing_LS = runSegment(event);

            // Run pT4
            float timing_pT4 = runpT4(event);

            // Run T4x
            float timing_T4x = 0; // runT4x(event);

            // Run T3
            float timing_T3 = runT3(event);

            //Run pT3
            float timing_pT3 = runpT3(event);

#ifdef DO_QUADRUPLET
            // Run T4
            float timing_T4 = runT4(event);
#else
            //Don't run T4
            float timing_T4 = 0;
#endif
#ifdef DO_QUINTUPLET
            float timing_T5 = runQuintuplet(event);
            float timing_pT5 = runPixelQuintuplet(event);
#else
            float timing_T5 = 0;
            float timing_pT5 = 0;
#endif
            // Run TC
            float timing_TC = runTrackCandidate(event);

            timing_information.push_back({ timing_input_loading,
                    timing_MD,
                    timing_LS,
                    timing_T4,
                    timing_T4x,
                    timing_pT4,
                    timing_T3,
                    timing_TC,
                    timing_T5,
                    timing_pT3,
                    timing_pT5});

            if (ana.verbose == 4)
            {
                printAllObjects(event);
            }

            if (ana.verbose == 5)
            {
                debugPrintOutlierMultiplicities(event);
            }

            if (ana.do_write_ntuple)
            {
                if (not ana.do_cut_value_ntuple)
                {
                    fillOutputBranches(event);
                }
            }

        }
        else
        {
            //*******************************************************
            // CPU VERSION RUN
            //*******************************************************

            // Main instance that will hold modules, hits, minidoublets, etc. (i.e. main data structure)
            SDL::CPU::Event event;

            // event.setLogLevel(SDL::CPU::Log_Debug3);

            // Add hits to the event
            float timing_input_loading = addOuterTrackerHits(event);

            // Add pixel segments
            timing_input_loading += addPixelSegments(event);

            // Print hit summary
            printHitSummary(event);

            // Run Mini-doublet
            float timing_MD = runMiniDoublet_on_CPU(event);
            printMiniDoubletSummary(event);

            // Run Segment
            float timing_LS = runSegment_on_CPU(event);
            printSegmentSummary(event);

            // Run Triplet
            float timing_T3 = runT3_on_CPU(event);
            printTripletSummary(event);

            // Run Tracklet
            float timing_T4 = 0; // runT4_on_CPU(event);
            printTrackletSummary(event);
            float timing_T4x = 0; // runT4x_on_CPU(event); // T4x's are turned off right now
            printTrackletSummary(event);
            float timing_pT4 = runpT4_on_CPU(event);
            printTrackletSummary(event);
            float timing_pT3 = runpT3_on_CPU(event);
            printTrackletSummary(event);

            // Run T5s
            float timing_T5 = runT5_on_CPU(event);
            // Run TrackCandidate
            float timing_TC = 0; // runTrackCandidate_on_CPU(event); // {T4, T3 based TC's, and no T5};
            printTrackCandidateSummary(event);

            timing_information.push_back({ timing_input_loading,
                    timing_MD,
                    timing_LS,
                    timing_T4,
                    timing_T4x,
                    timing_pT3,
                    timing_T3,
                    timing_TC,
                    timing_T5});

            if (ana.verbose == 4)
            {
                printAllObjects_for_CPU(event);
            }

            if (ana.do_write_ntuple)
            {
                fillOutputBranches_for_CPU(event);
            }

        }

    }

    printTimingInformation(timing_information);

    if (not ana.do_run_cpu)
        SDL::cleanModules();

    // Writing ttree output to file
    ana.output_tfile->cd();
    if (not ana.do_cut_value_ntuple) 
    {
        ana.cutflow.saveOutput();
    }

    ana.output_ttree->Write();

    // The below can be sometimes crucial
    delete ana.output_tfile;

}

//_______________________________________________________________________________
void writeMetaData()
{

    // Write out metadata of the code to the output_tfile
    ana.output_tfile->cd();
    gSystem->Exec(TString::Format("echo '' > %s.gitversion.txt", ana.output_tfile->GetName()));
    gSystem->Exec(TString::Format("git rev-parse HEAD >> %s.gitversion.txt", ana.output_tfile->GetName()));
    gSystem->Exec(TString::Format("echo 'git status' >> %s.gitversion.txt", ana.output_tfile->GetName()));
    gSystem->Exec(TString::Format("git status >> %s.gitversion.txt", ana.output_tfile->GetName()));
    gSystem->Exec(TString::Format("echo 'git log -n5' >> .%s.gitversion.txt", ana.output_tfile->GetName()));
    gSystem->Exec(TString::Format("git log >> %s.gitversion.txt", ana.output_tfile->GetName()));
    gSystem->Exec(TString::Format("echo 'git diff' >> %s.gitversion.txt", ana.output_tfile->GetName()));
    gSystem->Exec(TString::Format("git diff >> %s.gitversion.txt", ana.output_tfile->GetName()));
    std::ifstream t(TString::Format("%s.gitversion.txt", ana.output_tfile->GetName()));
    std::string str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
    TString tstr = str.c_str();
    TObjString tobjstr("code_tag_data");
    tobjstr.SetString(tstr.Data());
    ana.output_tfile->WriteObject(&tobjstr, "code_tag_data");
    gSystem->Exec(TString::Format("rm %s.gitversion.txt", ana.output_tfile->GetName()));
    TString make_log_path = TString::Format("%s/.make.log", ana.track_looper_dir_path.Data());
    std::ifstream makelog(make_log_path.Data());
    std::string makestr((std::istreambuf_iterator<char>(makelog)), std::istreambuf_iterator<char>());
    TString maketstr = makestr.c_str();
    TObjString maketobjstr("make_log");
    maketobjstr.SetString(maketstr.Data());
    ana.output_tfile->WriteObject(&maketobjstr, "make_log");

    // Write git diff output in a separate string to gauge the difference
    gSystem->Exec(TString::Format("git diff > %s.gitdiff.txt", ana.output_tfile->GetName()));
    std::ifstream gitdiff(TString::Format("%s.gitdiff.txt", ana.output_tfile->GetName()));
    std::string strgitdiff((std::istreambuf_iterator<char>(gitdiff)), std::istreambuf_iterator<char>());
    TString tstrgitdiff = strgitdiff.c_str();
    TObjString tobjstrgitdiff("gitdiff");
    tobjstrgitdiff.SetString(tstrgitdiff.Data());
    ana.output_tfile->WriteObject(&tobjstrgitdiff, "gitdiff");
    gSystem->Exec(TString::Format("rm %s.gitdiff.txt", ana.output_tfile->GetName()));

    // Parse from makestr the TARGET
    TString rawstrdata = maketstr.ReplaceAll("MAKETARGET=", "%");
    TString targetrawdata = RooUtil::StringUtil::rsplit(rawstrdata,"%")[1];
    TString targetdata = RooUtil::StringUtil::split(targetrawdata)[0];
    ana.compilation_target = targetdata.Data();

    // Write out input sample or file name
    TObjString input;
    input.SetString(ana.input_raw_string.Data());
    ana.output_tfile->WriteObject(&input, "input");

    // Write out whether it's GPU or CPU
    TObjString version;
    if (ana.do_run_cpu)
        version.SetString("CPU");
    else
        version.SetString(TString::Format("GPU_%s", targetdata.Data()));
    ana.output_tfile->WriteObject(&version, "version");

    // Write the full command line used
    TObjString full_cmd_line_to_be_written;
    full_cmd_line_to_be_written.SetString(ana.full_cmd_line.Data());
    ana.output_tfile->WriteObject(&full_cmd_line_to_be_written, "full_cmd_line");

    // Write the TRACKLOOPERDIR
    TObjString tracklooperdirpath_to_be_written;
    tracklooperdirpath_to_be_written.SetString(ana.track_looper_dir_path.Data());
    ana.output_tfile->WriteObject(&tracklooperdirpath_to_be_written, "tracklooper_path");

}
