#include "sdl.h"

// ./process INPUTFILEPATH OUTPUTFILE [NEVENTS]
int main(int argc, char** argv)
{

//********************************************************************************
//
// 0. Preliminary operations
//
//********************************************************************************

    // Write the command line used to run it
    // N.B. This needs to be before the argument parsing as it will change some values
    std::vector<std::string> allArgs(argv, argv + argc);
    TString full_cmd_line;
    for (auto& str : allArgs)
    {
        full_cmd_line += TString::Format(" %s", str.c_str());
    }

    // Checking the TRACKLOOPERDIR is set
    TString TrackLooperDir = gSystem->Getenv("TRACKLOOPERDIR");
    if (TrackLooperDir.IsNull())
    {
        RooUtil::error("TRACKLOOPERDIR is not set! Did you run $ source setup.sh from TrackLooper/ main repository directory?");
    }
    RooUtil::print(TString::Format("TRACKLOOPERDIR=%s", TrackLooperDir.Data()));

//********************************************************************************
//
// 1. Parsing options
//
//********************************************************************************

    // cxxopts is just a tool to parse argc, and argv easily

    // Grand option setting
    cxxopts::Options options("\n  $ doAnalysis",  "\n         **********************\n         *                    *\n         *       Looper       *\n         *                    *\n         **********************\n");

    // Read the options
    options.add_options()
        ("m,mode"           , "Run mode (0=build_module_map, 1=print_module_centroid, 2=mtv, 3=algo_eff, 4=tracklet, 5=write_sdl_ntuple, 6=pixel_tracklet_eff)", cxxopts::value<int>()->default_value("5"))
        ("i,input"          , "Comma separated input file list OR if just a directory is provided it will glob all in the directory BUT must end with '/' for the path", cxxopts::value<std::string>()->default_value("muonGun"))
        ("t,tree"           , "Name of the tree in the root file to open and loop over"                                             , cxxopts::value<std::string>()->default_value("trackingNtuple/tree"))
        ("o,output"         , "Output file name"                                                                                    , cxxopts::value<std::string>())
        ("N,nmatch"         , "N match for MTV-like plots"                                                                          , cxxopts::value<int>()->default_value("9"))
        ("n,nevents"        , "N events to loop over"                                                                               , cxxopts::value<int>()->default_value("-1"))
        ("x,event_index"    , "specific event index to process"                                                                     , cxxopts::value<int>()->default_value("-1"))
        ("p,ptbound_mode"   , "Pt bound mode (i.e. 0 = default, 1 = pt~1, 2 = pt~0.95-1.5, 3 = pt~0.5-1.5, 4 = pt~0.5-2.0"          , cxxopts::value<int>()->default_value("0"))
        ("g,pdg_id"         , "The simhit pdgId match option (default = 0)"                                                         , cxxopts::value<int>()->default_value("0"))
        ("v,verbose"        , "Verbose mode"                                                                                        , cxxopts::value<int>()->default_value("0"))
        ("d,debug"          , "Run debug job. i.e. overrides output option to 'debug.root' and 'recreate's the file.")
        ("c,cpu"            , "Run CPU version of the code.")
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
    ana.input_file_list_tstring = result["input"].as<std::string>();

    // A default value one
    if (ana.input_file_list_tstring.EqualTo("muonGun"))
        ana.input_file_list_tstring = "/data2/segmentlinking/trackingNtuple_100_pt0p5_2p0.root";
    else if (ana.input_file_list_tstring.EqualTo("muonGun_highE"))
        ana.input_file_list_tstring = "/data2/segmentlinking/trackingNtuple_10MuGun.root";
    else if (ana.input_file_list_tstring.EqualTo("pionGun"))
        ana.input_file_list_tstring = "/data2/segmentlinking/trackingNtuple_1pion_10k_pt0p5_50p0.root";
    else if (ana.input_file_list_tstring.EqualTo("PU200"))
        ana.input_file_list_tstring = "/data2/segmentlinking/trackingNtuple_with_PUinfo_500_evts.root";

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

    // -1 upto mini-doublet is all-comb
    // -2 upto segment is all-comb
    // -3 upto tracklet is all-comb NOTE: MEMORY WILL BLOW UP FOR HIGH PU
    // -4 upto trackcandidate is all-comb NOTE: MEMORY WILL BLOW UP FOR HIGH PU
    //  0 nothing
    //  1 upto mini-doublet is all-comb
    //  2 upto mini-doublet is default segment is all-comb
    //  3 upto segment is default tracklet is all-comb
    //  4 upto tracklet is default trackcandidate is all-comb
    ana.ptbound_mode = result["ptbound_mode"].as<int>();

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
    // --cpu
    if (result.count("cpu"))
    {
        ana.do_run_cpu = true;
    }
    else
    {
        ana.do_run_cpu = false;
    }

    // Printing out the option settings overview
    std::cout <<  "=========================================================" << std::endl;
    std::cout <<  " Setting of the analysis job based on provided arguments " << std::endl;
    std::cout <<  "---------------------------------------------------------" << std::endl;
    std::cout <<  " ana.input_file_list_tstring: " << ana.input_file_list_tstring <<  std::endl;
    std::cout <<  " ana.output_tfile: " << ana.output_tfile->GetName() <<  std::endl;
    std::cout <<  " ana.n_events: " << ana.n_events <<  std::endl;
    std::cout <<  " ana.ptbound_mode: " << ana.ptbound_mode <<  std::endl;
    std::cout <<  " ana.nsplit_jobs: " << ana.nsplit_jobs <<  std::endl;
    std::cout <<  " ana.job_index: " << ana.job_index <<  std::endl;
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

    // Write out metadata of the code to the output_tfile
    ana.output_tfile->cd();
    gSystem->Exec("echo '' > .gitversion.txt");
    gSystem->Exec("git rev-parse HEAD >> .gitversion.txt");
    gSystem->Exec("echo 'git status' >> .gitversion.txt");
    gSystem->Exec("git status >> .gitversion.txt");
    gSystem->Exec("echo 'git log -n5' >> .gitversion.txt");
    gSystem->Exec("git log >> .gitversion.txt");
    gSystem->Exec("echo 'git diff' >> .gitversion.txt");
    gSystem->Exec("git diff >> .gitversion.txt");
    std::ifstream t(".gitversion.txt");
    std::string str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
    TString tstr = str.c_str();
    TObjString tobjstr("code_tag_data");
    tobjstr.SetString(tstr.Data());
    ana.output_tfile->WriteObject(&tobjstr, "code_tag_data");
    TString make_log_path = TString::Format("%s/.make.log", TrackLooperDir.Data());
    std::ifstream makelog(make_log_path.Data());
    std::string makestr((std::istreambuf_iterator<char>(makelog)), std::istreambuf_iterator<char>());
    TString maketstr = makestr.c_str();
    TObjString maketobjstr("make_log");
    maketobjstr.SetString(maketstr.Data());
    ana.output_tfile->WriteObject(&maketobjstr, "make_log");

    // Write git diff output in a separate string to gauge the difference
    gSystem->Exec("git diff > .gitdiff.txt");
    std::ifstream gitdiff(".gitdiff.txt");
    std::string strgitdiff((std::istreambuf_iterator<char>(gitdiff)), std::istreambuf_iterator<char>());
    TString tstrgitdiff = strgitdiff.c_str();
    TObjString tobjstrgitdiff("gitdiff");
    tobjstrgitdiff.SetString(tstrgitdiff.Data());
    ana.output_tfile->WriteObject(&tobjstrgitdiff, "gitdiff");

    // Parse from makestr the TARGET
    TString rawstrdata = maketstr.ReplaceAll("MAKETARGET=", "%");
    TString targetrawdata = RooUtil::StringUtil::rsplit(rawstrdata,"%")[1];
    TString targetdata = RooUtil::StringUtil::split(targetrawdata)[0];

    // Write out input sample or file name
    TObjString input;
    input.SetString(result["input"].as<std::string>().c_str());
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
    full_cmd_line_to_be_written.SetString(full_cmd_line.Data());
    ana.output_tfile->WriteObject(&full_cmd_line_to_be_written, "full_cmd_line");

    // Write the TRACKLOOPERDIR
    TObjString tracklooperdirpath_to_be_written;
    tracklooperdirpath_to_be_written.SetString(TrackLooperDir.Data());
    ana.output_tfile->WriteObject(&tracklooperdirpath_to_be_written, "tracklooper_path");

    // Run depending on the mode
    switch (ana.mode)
    {
        //case 0: build_module_map(); break;
        //case 1: print_module_centroid(); break;
        //case 2: mtv(); break;
        //case 3: algo_eff(); break;
        //case 4: tracklet(); break;
        case 5: write_sdl_ntuple(false,true); break;
        case 6 : write_sdl_ntuple(true,true); break;
        case 7 : write_sdl_ntuple(false,false,targetdata.Data()); break; // quick run, not validation
        //case 6: pixel_tracklet_eff(); break;
        default:
                std::cout << options.help() << std::endl;
                std::cout << "ERROR: --mode was not provided! Check your arguments." << std::endl;
                exit(1);
                break;
    }

    return 0;
}

