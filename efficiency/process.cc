#include "process.h"

// Global variable
AnalysisConfig ana;

#include "sdl_types.h"

// ./process INPUTFILEPATH OUTPUTFILE [NEVENTS]
int main(int argc, char** argv)
{

    // Parse arguments
    parseArguments(argc, argv);

    // Initialize input and output root files
    initializeInputsAndOutputs();

    // Create various bits important for each track
    createSDLVariables();

    // creating a set of efficiency plots
    std::vector<EfficiencySetDefinition> list_effSetDef;

    // Book MD
    for (auto& md_type : MD_types)
        list_effSetDef.push_back(EfficiencySetDefinition(TString::Format("MD_%s", md_type.Data()), 13, [&, md_type](int isim) {return ana.tx.getBranch<vector<vector<int>>>(TString::Format("mtv_match_idxs_MD_%s", md_type.Data()))[isim].size() > 0;}));

    for (auto& ls_type : LS_types)
        list_effSetDef.push_back(EfficiencySetDefinition(TString::Format("LS_%s", ls_type.Data()), 13, [&, ls_type](int isim) {return ana.tx.getBranch<vector<vector<int>>>(TString::Format("mtv_match_idxs_LS_%s", ls_type.Data()))[isim].size() > 0;}));

    list_effSetDef.push_back(EfficiencySetDefinition("pLS_P", 13, [&](int isim) {return ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_pLS_P")[isim].size() > 0;}));

    for (auto& pt4_type : pT4_types)
        list_effSetDef.push_back(EfficiencySetDefinition(TString::Format("pT4_%s", pt4_type.Data()), 13, [&, pt4_type](int isim) {return ana.tx.getBranch<vector<vector<int>>>(TString::Format("mtv_match_idxs_pT4_%s", pt4_type.Data()))[isim].size() > 0;}));
    list_effSetDef.push_back(EfficiencySetDefinition("pT4_AllTypes", 13, [&](int isim) {return ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_pT4_AllTypes")[isim].size() > 0;}));

    for (auto& t4_type : T4_types)
        list_effSetDef.push_back(EfficiencySetDefinition(TString::Format("T4_%s", t4_type.Data()), 13, [&, t4_type](int isim) {return ana.tx.getBranch<vector<vector<int>>>(TString::Format("mtv_match_idxs_T4_%s", t4_type.Data()))[isim].size() > 0;}));
    list_effSetDef.push_back(EfficiencySetDefinition("T4_AllTypes", 13, [&](int isim) {return ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_T4_AllTypes")[isim].size() > 0;}));

    for (auto& t4x_type : T4x_types)
        list_effSetDef.push_back(EfficiencySetDefinition(TString::Format("T4x_%s", t4x_type.Data()), 13, [&, t4x_type](int isim) {return ana.tx.getBranch<vector<vector<int>>>(TString::Format("mtv_match_idxs_T4x_%s", t4x_type.Data()))[isim].size() > 0;}));
    list_effSetDef.push_back(EfficiencySetDefinition("T4x_AllTypes", 13, [&](int isim) {return ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_T4x_AllTypes")[isim].size() > 0;}));
    list_effSetDef.push_back(EfficiencySetDefinition("AllT4s_AllTypes", 13, [&](int isim) {return ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_AllT4s_AllTypes")[isim].size() > 0;}));

    for (auto& t3_type : T3_types)
        list_effSetDef.push_back(EfficiencySetDefinition(TString::Format("T3_%s", t3_type.Data()), 13, [&, t3_type](int isim) {return ana.tx.getBranch<vector<vector<int>>>(TString::Format("mtv_match_idxs_T3_%s", t3_type.Data()))[isim].size() > 0;}));
    list_effSetDef.push_back(EfficiencySetDefinition("T3_AllTypes", 13, [&](int isim) {return ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_T3_AllTypes")[isim].size() > 0;}));

    for (auto& tc_type : TC_types)
        list_effSetDef.push_back(EfficiencySetDefinition(TString::Format("TC_%s", tc_type.Data()), 13, [&, tc_type](int isim) {return ana.tx.getBranch<vector<vector<int>>>(TString::Format("mtv_match_idxs_TC_%s", tc_type.Data()))[isim].size() > 0;}));
    list_effSetDef.push_back(EfficiencySetDefinition("TC_AllTypes", 13, [&](int isim) {return ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_TC_AllTypes")[isim].size() > 0;}));
    list_effSetDef.push_back(EfficiencySetDefinition("TC_Set1Types", 13, [&](int isim) {return ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_TC_Set1Types")[isim].size() > 0;}));
    list_effSetDef.push_back(EfficiencySetDefinition("TC_Set2Types", 13, [&](int isim) {return ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_TC_Set2Types")[isim].size() > 0;}));
    list_effSetDef.push_back(EfficiencySetDefinition("TC_Set3Types", 13, [&](int isim) {return ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_TC_Set3Types")[isim].size() > 0;}));


    bookEfficiencySets(list_effSetDef);

    ana.cutflow.bookCutflows();

    // Book Histograms
    ana.cutflow.bookHistograms(ana.histograms); // if just want to book everywhere

    // Looping input file
    while (ana.looper.nextEvent())
    {

        // If splitting jobs are requested then determine whether to process the event or not based on remainder
        if (ana.job_index != -1 and ana.nsplit_jobs != -1)
        {
            if (ana.looper.getNEventsProcessed() % ana.nsplit_jobs != (unsigned int) ana.job_index)
                continue;
        }

        // Reset all variables
        ana.tx.clear();

        setSDLVariables();

        fillEfficiencySets(list_effSetDef);

        //Do what you need to do in for each event here
        //To save use the following function
        ana.cutflow.fill();
    }

    // Writing output file
    ana.cutflow.saveOutput();

    // The below can be sometimes crucial
    delete ana.output_tfile;
}

AnalysisConfig::AnalysisConfig() : tx("variable", "variable")
{
}

void parseArguments(int argc, char** argv)
{
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
        ("i,input"       , "Comma separated input file list OR if just a directory is provided it will glob all in the directory BUT must end with '/' for the path", cxxopts::value<std::string>())
        ("t,tree"        , "Name of the tree in the root file to open and loop over"                                             , cxxopts::value<std::string>())
        ("o,output"      , "Output file name"                                                                                    , cxxopts::value<std::string>())
        ("n,nevents"     , "N events to loop over"                                                                               , cxxopts::value<int>()->default_value("-1"))
        ("j,nsplit_jobs" , "Enable splitting jobs by N blocks (--job_index must be set)"                                         , cxxopts::value<int>())
        ("I,job_index"   , "job_index of split jobs (--nsplit_jobs must be set. index starts from 0. i.e. 0, 1, 2, 3, etc...)"   , cxxopts::value<int>())
        ("p,ptbound_mode"   , "Pt bound mode (i.e. 0 = default, 1 = pt~1, 2 = pt~0.95-1.5, 3 = pt~0.5-1.5, 4 = pt~0.5-2.0"          , cxxopts::value<int>()->default_value("0"))
        ("d,debug"       , "Run debug job. i.e. overrides output option to 'debug.root' and 'recreate's the file.")
        ("h,help"        , "Print help")
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
    if (result.count("input"))
    {
        ana.input_file_list_tstring = result["input"].as<std::string>();
    }
    else
    {
        std::cout << options.help() << std::endl;
        std::cout << "ERROR: Input list is not provided! Check your arguments" << std::endl;
        exit(1);
    }

    //_______________________________________________________________________________
    // --tree
    if (result.count("tree"))
    {
        ana.input_tree_name = result["tree"].as<std::string>();
    }
    else
    {
        std::cout << options.help() << std::endl;
        std::cout << "ERROR: Input tree name is not provided! Check your arguments" << std::endl;
        exit(1);
    }

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
            std::cout << options.help() << std::endl;
            std::cout << "ERROR: Output file name is not provided! Check your arguments" << std::endl;
            exit(1);
        }
    }

    //_______________________________________________________________________________
    // --nevents
    ana.n_events = result["nevents"].as<int>();

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
    // --nsplit_jobs
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

    //
    // Printing out the option settings overview
    //
    std::cout <<  "=========================================================" << std::endl;
    std::cout <<  " Setting of the analysis job based on provided arguments " << std::endl;
    std::cout <<  "---------------------------------------------------------" << std::endl;
    std::cout <<  " ana.input_file_list_tstring: " << ana.input_file_list_tstring <<  std::endl;
    std::cout <<  " ana.output_tfile: " << ana.output_tfile->GetName() <<  std::endl;
    std::cout <<  " ana.n_events: " << ana.n_events <<  std::endl;
    std::cout <<  " ana.nsplit_jobs: " << ana.nsplit_jobs <<  std::endl;
    std::cout <<  " ana.job_index: " << ana.job_index <<  std::endl;
    std::cout <<  " ana.ptbound_mode: " << ana.ptbound_mode <<  std::endl;
    std::cout <<  "=========================================================" << std::endl;
}

void initializeInputsAndOutputs()
{
    // Create the TChain that holds the TTree's of the baby ntuples
    ana.events_tchain = RooUtil::FileUtil::createTChain(ana.input_tree_name, ana.input_file_list_tstring);

    // This is a standard thing SNT does pretty much every looper we use
    ana.looper.init(ana.events_tchain, &sdl, ana.n_events);

    // Set the cutflow object output file
    ana.cutflow.setTFile(ana.output_tfile);
}

void createSDLVariables()
{

    // hasAll12Hits
    ana.tx.createBranch<vector<int>>("sim_hasAll12HitsInBarrel");

    // Mini-Doublets
    for (auto& md_type : MD_types)
        ana.tx.createBranch<vector<vector<int>>>(TString::Format("mtv_match_idxs_MD_%s", md_type.Data()));

    // Line segments
    for (auto& ls_type : LS_types)
        ana.tx.createBranch<vector<vector<int>>>(TString::Format("mtv_match_idxs_LS_%s", ls_type.Data()));

    // Pixel Line segments
    ana.tx.createBranch<vector<vector<int>>>("mtv_match_idxs_pLS_P");

    // Tracklets with Pixel
    for (auto& pt4_type : pT4_types)
        ana.tx.createBranch<vector<vector<int>>>(TString::Format("mtv_match_idxs_pT4_%s", pt4_type.Data()));
    ana.tx.createBranch<vector<vector<int>>>("mtv_match_idxs_pT4_AllTypes");

    // Tracklets
    for (auto& t4_type : T4_types)
        ana.tx.createBranch<vector<vector<int>>>(TString::Format("mtv_match_idxs_T4_%s", t4_type.Data()));
    ana.tx.createBranch<vector<vector<int>>>("mtv_match_idxs_T4_AllTypes");

    // Tracklets with a gap
    for (auto& t4x_type : T4x_types)
        ana.tx.createBranch<vector<vector<int>>>(TString::Format("mtv_match_idxs_T4x_%s", t4x_type.Data()));
    ana.tx.createBranch<vector<vector<int>>>("mtv_match_idxs_T4x_AllTypes");
    ana.tx.createBranch<vector<vector<int>>>("mtv_match_idxs_AllT4s_AllTypes");

    // Triplets
    for (auto& t3_type : T3_types)
        ana.tx.createBranch<vector<vector<int>>>(TString::Format("mtv_match_idxs_T3_%s", t3_type.Data()));
    ana.tx.createBranch<vector<vector<int>>>("mtv_match_idxs_T3_AllTypes");

    // Track Candidates
    for (auto& tc_type : TC_types)
        ana.tx.createBranch<vector<vector<int>>>(TString::Format("mtv_match_idxs_TC_%s", tc_type.Data()));
    ana.tx.createBranch<vector<vector<int>>>("mtv_match_idxs_TC_AllTypes");
    ana.tx.createBranch<vector<vector<int>>>("mtv_match_idxs_TC_Set1Types");
    ana.tx.createBranch<vector<vector<int>>>("mtv_match_idxs_TC_Set2Types");
    ana.tx.createBranch<vector<vector<int>>>("mtv_match_idxs_TC_Set3Types");

}

void setSDLVariables()
{

    for (unsigned int isim = 0; isim < sdl.sim_pt().size(); ++isim)
    {

        // hasAll12Hits
        ana.tx.pushbackToBranch<int>("sim_hasAll12HitsInBarrel", sdl.sim_hasAll12HitsInBarrel()[isim]);

        // Mini-Doublets
        std::array<std::vector<int>, n_MD_types> MD_idxs;
        for (auto& mdIdx : sdl.sim_mdIdx()[isim])
        {
            const int& layer = sdl.md_layer()[mdIdx][0];
            MD_idxs[layer-1].push_back(mdIdx);
        }

        // Set the MD idxs variables
        for (unsigned int imd = 0; imd < MD_types.size(); ++imd)
        {
            ana.tx.pushbackToBranch<vector<int>>(TString::Format("mtv_match_idxs_MD_%s", MD_types.at(imd).Data()), MD_idxs.at(imd));
        }

        // Line Segments
        std::array<std::vector<int>, n_LS_types> LS_idxs;
        for (auto& sgIdx : sdl.sim_sgIdx()[isim])
        {
            const int& layerIn = sdl.sg_layer()[sgIdx][0];
            const int& layerOut = sdl.sg_layer()[sgIdx][2];
            LS_idxs[LS_types_map[std::make_pair(layerIn, layerOut)]].push_back(sgIdx);
        }

        // Set the LS idxs variables
        for (unsigned int ils = 0; ils < LS_types.size(); ++ils)
        {
            ana.tx.pushbackToBranch<vector<int>>(TString::Format("mtv_match_idxs_LS_%s", LS_types.at(ils).Data()), LS_idxs.at(ils));
        }

        // Pixel Line segments
        std::vector<int> pLS_idxs;
        for (auto& psgIdx : sdl.sim_psgIdx()[isim])
        {
            pLS_idxs.push_back(psgIdx);
        }

        ana.tx.pushbackToBranch<vector<int>>("mtv_match_idxs_pLS_P", pLS_idxs);

        // Pixel (pT4)
        std::array<std::vector<int>, n_pT4_types> pT4_idxs;
        std::vector<int> pT4_idxs_all;
        for (auto& pqpIdx : sdl.sim_pqpIdx()[isim])
        {
            const int& layerOutLo = sdl.pqp_layer()[pqpIdx][2];
            const int& layerOutUp = sdl.pqp_layer()[pqpIdx][3];
            pT4_idxs[pT4_types_map[std::make_pair(layerOutLo, layerOutUp)]].push_back(pqpIdx);
            pT4_idxs_all.push_back(pqpIdx);
        }

        // Set the pT4 idxs variables
        for (unsigned int ipt4 = 0; ipt4 < pT4_types.size(); ++ipt4)
        {
            ana.tx.pushbackToBranch<vector<int>>(TString::Format("mtv_match_idxs_pT4_%s", pT4_types.at(ipt4).Data()), pT4_idxs.at(ipt4));
        }
        ana.tx.pushbackToBranch<vector<int>>("mtv_match_idxs_pT4_AllTypes", pT4_idxs_all);

        // Any T4 (not including pT4)
        std::vector<int> AllT4s_idxs_all;

        // Tracklet (T4)
        std::array<std::vector<int>, n_T4_types> T4_idxs;
        std::vector<int> T4_idxs_all;
        for (auto& qpIdx : sdl.sim_qpIdx()[isim])
        {
            std::vector<int> layers = {sdl.qp_layer()[qpIdx][0], sdl.qp_layer()[qpIdx][1], sdl.qp_layer()[qpIdx][2], sdl.qp_layer()[qpIdx][3]};
            if (T4_types_map.find(layers) != T4_types_map.end())
            {
                T4_idxs[T4_types_map.at(layers)].push_back(qpIdx);
                T4_idxs_all.push_back(qpIdx);
                AllT4s_idxs_all.push_back(qpIdx);
            }
        }

        // Set the T4 idxs variables
        for (unsigned int it4 = 0; it4 < T4_types.size(); ++it4)
        {
            ana.tx.pushbackToBranch<vector<int>>(TString::Format("mtv_match_idxs_T4_%s", T4_types.at(it4).Data()), T4_idxs.at(it4));
        }
        ana.tx.pushbackToBranch<vector<int>>("mtv_match_idxs_T4_AllTypes", T4_idxs_all);

        // Tracklet (T4x)
        std::array<std::vector<int>, n_T4x_types> T4x_idxs;
        std::vector<int> T4x_idxs_all;
        for (auto& qpIdx : sdl.sim_qpIdx()[isim])
        {
            std::vector<int> layers = {sdl.qp_layer()[qpIdx][0], sdl.qp_layer()[qpIdx][1], sdl.qp_layer()[qpIdx][2], sdl.qp_layer()[qpIdx][3]};
            if (T4x_types_map.find(layers) != T4x_types_map.end())
            {
                T4x_idxs[T4x_types_map.at(layers)].push_back(qpIdx);
                T4x_idxs_all.push_back(qpIdx);
                AllT4s_idxs_all.push_back(qpIdx);
            }
        }

        // Set the T4x idxs variables
        for (unsigned int it4x = 0; it4x < T4x_types.size(); ++it4x)
        {
            ana.tx.pushbackToBranch<vector<int>>(TString::Format("mtv_match_idxs_T4x_%s", T4x_types.at(it4x).Data()), T4x_idxs.at(it4x));
        }
        ana.tx.pushbackToBranch<vector<int>>("mtv_match_idxs_T4x_AllTypes", T4x_idxs_all);

        // Any T4s (not including pT4)
        ana.tx.pushbackToBranch<vector<int>>("mtv_match_idxs_AllT4s_AllTypes", AllT4s_idxs_all);

        // Triplet (T3)
        std::array<std::vector<int>, n_T3_types> T3_idxs;
        std::vector<int> T3_idxs_all;
        for (auto& tpIdx : sdl.sim_tpIdx()[isim])
        {
            std::vector<int> layers = {sdl.tp_layer()[tpIdx][0], sdl.tp_layer()[tpIdx][2], sdl.tp_layer()[tpIdx][4]};
            if (T3_types_map.find(layers) != T3_types_map.end())
            {
                T3_idxs[T3_types_map.at(layers)].push_back(tpIdx);
                T3_idxs_all.push_back(tpIdx);
            }
        }

        // Set the T3 idxs variables
        for (unsigned int it3 = 0; it3 < T3_types.size(); ++it3)
        {
            ana.tx.pushbackToBranch<vector<int>>(TString::Format("mtv_match_idxs_T3_%s", T3_types.at(it3).Data()), T3_idxs.at(it3));
        }
        ana.tx.pushbackToBranch<vector<int>>("mtv_match_idxs_T3_AllTypes", T3_idxs_all);

        // Track Candidates (TC)
        std::array<std::vector<int>, n_TC_types> TC_idxs;
        std::vector<int> TC_idxs_all;
        std::vector<int> TC_set1_idxs_all;
        std::vector<int> TC_set2_idxs_all;
        std::vector<int> TC_set3_idxs_all;
        for (auto& tcIdx : sdl.sim_tcIdx()[isim])
        {
            const std::vector<int>& layers = sdl.tc_layer()[tcIdx];
            if (TC_types_map.find(layers) != TC_types_map.end())
            {
                TC_idxs[TC_types_map.at(layers)].push_back(tcIdx);
                TC_idxs_all.push_back(tcIdx);
            }
            if (TC_set1_types_map.find(layers) != TC_set1_types_map.end())
            {
                TC_set1_idxs_all.push_back(tcIdx);
            }
            if (TC_set2_types_map.find(layers) != TC_set2_types_map.end())
            {
                TC_set2_idxs_all.push_back(tcIdx);
            }
            if (TC_set3_types_map.find(layers) != TC_set3_types_map.end())
            {
                TC_set3_idxs_all.push_back(tcIdx);
            }
        }

        // Set the TC idxs variables
        for (unsigned int itc = 0; itc < TC_types.size(); ++itc)
        {
            ana.tx.pushbackToBranch<vector<int>>(TString::Format("mtv_match_idxs_TC_%s", TC_types.at(itc).Data()), TC_idxs.at(itc));
        }
        ana.tx.pushbackToBranch<vector<int>>("mtv_match_idxs_TC_AllTypes", TC_idxs_all);
        ana.tx.pushbackToBranch<vector<int>>("mtv_match_idxs_TC_Set1Types", TC_set1_idxs_all);
        ana.tx.pushbackToBranch<vector<int>>("mtv_match_idxs_TC_Set2Types", TC_set2_idxs_all);
        ana.tx.pushbackToBranch<vector<int>>("mtv_match_idxs_TC_Set3Types", TC_set3_idxs_all);

    }

}

void printSDLVariables()
{
    for (unsigned int isimtrk = 0; isimtrk < sdl.sim_pt().size(); ++isimtrk)
        printSDLVariablesForATrack(isimtrk);
}

void printSDLVariablesForATrack(int isimtrk)
{

    // hasAll12Hits
    const int& hasAll12Hits = ana.tx.getBranch<vector<int>>("sim_hasAll12HitsInBarrel")[isimtrk];

    // Mini-Doublets
    bool hasMD_B1 = ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_MD_B1")[isimtrk].size() > 0;
    bool hasMD_B2 = ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_MD_B2")[isimtrk].size() > 0;
    bool hasMD_B3 = ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_MD_B3")[isimtrk].size() > 0;
    bool hasMD_B4 = ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_MD_B4")[isimtrk].size() > 0;
    bool hasMD_B5 = ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_MD_B5")[isimtrk].size() > 0;
    bool hasMD_B6 = ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_MD_B6")[isimtrk].size() > 0;
    bool hasMD_E1 = ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_MD_E1")[isimtrk].size() > 0;
    bool hasMD_E2 = ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_MD_E2")[isimtrk].size() > 0;
    bool hasMD_E3 = ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_MD_E3")[isimtrk].size() > 0;
    bool hasMD_E4 = ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_MD_E4")[isimtrk].size() > 0;
    bool hasMD_E5 = ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_MD_E5")[isimtrk].size() > 0;

    const float& pt = sdl.sim_pt()[isimtrk];
    const float& eta = sdl.sim_eta()[isimtrk];
    const float& dz = sdl.sim_pca_dz()[isimtrk];
    const float& dxy = sdl.sim_pca_dxy()[isimtrk];
    const float& phi = sdl.sim_phi()[isimtrk];
    // const int& bunch = sdl.sim_bunchCrossing()[isimtrk];
    const int& pdgid = sdl.sim_pdgId()[isimtrk];

    std::cout << "isimtrk : " << isimtrk << std::endl;
    std::cout <<  " pt: " << pt <<  " eta: " << eta <<  " pdgid: " << pdgid <<  " dz: " << dz <<  " dxy: " << dxy <<  " phi: " << phi <<  std::endl;
    std::cout << "hasAll12Hits : " << hasAll12Hits << std::endl;
    std::cout <<  " hasMD_B1: " << hasMD_B1 <<  " hasMD_B2: " << hasMD_B2 <<  " hasMD_B3: " << hasMD_B3 <<  " hasMD_B4: " << hasMD_B4 <<  " hasMD_B5: " << hasMD_B5 <<  " hasMD_B6: " << hasMD_B6 <<  " hasMD_E1: " << hasMD_E1 <<  " hasMD_E2: " << hasMD_E2 <<  " hasMD_E3: " << hasMD_E3 <<  " hasMD_E4: " << hasMD_E4 <<  " hasMD_E5: " << hasMD_E5 <<  std::endl;

}

EfficiencySetDefinition::EfficiencySetDefinition(TString set_name_, int pdgid_, std::function<bool(int)> pass_)
{
    set_name = set_name_;
    pdgid = pdgid_;
    pass = pass_;
}

void bookEfficiencySets(std::vector<EfficiencySetDefinition>& effsets)
{
    for (auto& effset : effsets)
        bookEfficiencySet(effset);
}

void bookEfficiencySet(EfficiencySetDefinition& effset)
{

    std::vector<float> pt_boundaries = getPtBounds();

    TString category_name = effset.set_name;

    // Denominator tracks' quantities
    ana.tx.createBranch<vector<float>>(category_name + "_denom_pt");
    ana.tx.createBranch<vector<float>>(category_name + "_denom_eta");
    ana.tx.createBranch<vector<float>>(category_name + "_denom_dxy");
    ana.tx.createBranch<vector<float>>(category_name + "_denom_dz");
    ana.tx.createBranch<vector<float>>(category_name + "_denom_phi");

    // Numerator tracks' quantities
    ana.tx.createBranch<vector<float>>(category_name + "_numer_pt");
    ana.tx.createBranch<vector<float>>(category_name + "_numer_eta");
    ana.tx.createBranch<vector<float>>(category_name + "_numer_dxy");
    ana.tx.createBranch<vector<float>>(category_name + "_numer_dz");
    ana.tx.createBranch<vector<float>>(category_name + "_numer_phi");

    // Histogram utility object that is used to define the histograms
    ana.histograms.addVecHistogram(category_name + "_h_denom_pt"  , pt_boundaries     , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_denom_pt"); } );
    ana.histograms.addVecHistogram(category_name + "_h_numer_pt"  , pt_boundaries     , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_numer_pt"); } );
    ana.histograms.addVecHistogram(category_name + "_h_denom_eta" , 180 , -2.5  , 2.5  , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_denom_eta"); } );
    ana.histograms.addVecHistogram(category_name + "_h_numer_eta" , 180 , -2.5  , 2.5  , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_numer_eta"); } );
    ana.histograms.addVecHistogram(category_name + "_h_denom_dxy" , 180 , -10.  , 10.  , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_denom_dxy"); } );
    ana.histograms.addVecHistogram(category_name + "_h_numer_dxy" , 180 , -10.  , 10.  , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_numer_dxy"); } );
    ana.histograms.addVecHistogram(category_name + "_h_denom_dz"  , 180 , -30.  , 30.  , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_denom_dz"); } );
    ana.histograms.addVecHistogram(category_name + "_h_numer_dz"  , 180 , -30.  , 30.  , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_numer_dz"); } );
    ana.histograms.addVecHistogram(category_name + "_h_denom_phi" , 180 , -M_PI , M_PI , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_denom_phi"); } );
    ana.histograms.addVecHistogram(category_name + "_h_numer_phi" , 180 , -M_PI , M_PI , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_numer_phi"); } );

}

void fillEfficiencySets(std::vector<EfficiencySetDefinition>& effsets)
{
    for (unsigned int isimtrk = 0; isimtrk < sdl.sim_pt().size(); ++isimtrk)
    {
        for (auto& effset : effsets)
        {
            fillEfficiencySet(isimtrk, effset);
        }
    }
}

void fillEfficiencySet(int isimtrk, EfficiencySetDefinition& effset)
{
    const float& pt = sdl.sim_pt()[isimtrk];
    const float& eta = sdl.sim_eta()[isimtrk];
    const float& dz = sdl.sim_pca_dz()[isimtrk];
    const float& dxy = sdl.sim_pca_dxy()[isimtrk];
    const float& phi = sdl.sim_phi()[isimtrk];
    const int& bunch = sdl.sim_bunchCrossing()[isimtrk];
    const int& pdgid = sdl.sim_pdgId()[isimtrk];

    // if (abs(dz) > 30 or abs(dxy) > 2.5 or bunch != 0 or abs(pdgid) != 13)
    //     return;
    if (bunch != 0 or abs(pdgid) != 13)
        return;

    TString category_name = effset.set_name;

    if (category_name.Contains("B6"))
    {
    }

    if (pt > 1.5 and abs(dz) < 30 and abs(dxy) < 2.5)
        ana.tx.pushbackToBranch<float>(category_name + "_denom_eta", eta);
    if (abs(eta) < 2.4 and abs(dz) < 30 and abs(dxy) < 2.5)
        ana.tx.pushbackToBranch<float>(category_name + "_denom_pt", pt);
    if (abs(eta) < 2.4 and pt > 1.5 and abs(dz) < 30 and abs(dxy) < 2.5)
        ana.tx.pushbackToBranch<float>(category_name + "_denom_phi", phi);
    if (abs(eta) < 2.4 and pt > 1.5 and abs(dz) < 30)
        ana.tx.pushbackToBranch<float>(category_name + "_denom_dxy", dxy);
    if (abs(eta) < 2.4 and pt > 1.5 and abs(dxy) < 2.5)
        ana.tx.pushbackToBranch<float>(category_name + "_denom_dz", dz);

    if (effset.pass(isimtrk))
    {
        if (pt > 1.5 and abs(dz) < 30 and abs(dxy) < 2.5)
            ana.tx.pushbackToBranch<float>(category_name + "_numer_eta", eta);
        if (abs(eta) < 2.4 and abs(dz) < 30 and abs(dxy) < 2.5)
            ana.tx.pushbackToBranch<float>(category_name + "_numer_pt", pt);
        if (abs(eta) < 2.4 and pt > 1.5 and abs(dz) < 30 and abs(dxy) < 2.5)
            ana.tx.pushbackToBranch<float>(category_name + "_numer_phi", phi);
        if (abs(eta) < 2.4 and pt > 1.5 and abs(dz) < 30)
            ana.tx.pushbackToBranch<float>(category_name + "_numer_dxy", dxy);
        if (abs(eta) < 2.4 and pt > 1.5 and abs(dxy) < 2.5)
            ana.tx.pushbackToBranch<float>(category_name + "_numer_dz", dz);
    }
}

std::vector<float> getPtBounds()
{
    std::vector<float> pt_boundaries;
    if (ana.ptbound_mode == 0)
        pt_boundaries = {0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0, 10, 15., 25, 50};
    else if (ana.ptbound_mode == 1)
        pt_boundaries = {0.988, 0.99, 0.992, 0.994, 0.996, 0.998, 1.0, 1.002, 1.004, 1.006, 1.008, 1.01, 1.012}; // lowpt
    else if (ana.ptbound_mode == 2)
        pt_boundaries = {0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995, 1.00, 1.005, 1.01, 1.015, 1.02, 1.025, 1.03, 1.035, 1.04, 1.045, 1.05}; // pt 0p95 1p05
    else if (ana.ptbound_mode == 3)
        pt_boundaries = {0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06, 1.08, 1.1, 1.2, 1.5}; // lowpt
    else if (ana.ptbound_mode == 4)
        pt_boundaries = {0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06, 1.08, 1.1, 1.12, 1.14, 1.16, 1.18, 1.2, 1.22, 1.24, 1.26, 1.28, 1.3, 1.32, 1.34, 1.36, 1.38, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0}; // lowpt
    else if (ana.ptbound_mode == 5)
        pt_boundaries = {0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06, 1.08, 1.1, 1.12, 1.14, 1.16, 1.18, 1.2, 1.24, 1.28, 1.32, 1.36, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0}; // lowpt
    else if (ana.ptbound_mode == 6)
        pt_boundaries = {0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 3.0, 4.0, 5.0}; // lowpt
    else if (ana.ptbound_mode == 7)
        pt_boundaries = {0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06, 1.08, 1.1, 1.12, 1.14, 1.16, 1.18, 1.2, 1.22, 1.24, 1.26, 1.28, 1.3, 1.32, 1.34, 1.36, 1.38, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.4, 2.6, 2.8, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50}; // lowpt
    else if (ana.ptbound_mode == 8)
        pt_boundaries = {0, 0.5, 1.0, 3.0, 5.0, 10, 15., 25, 50};
    return pt_boundaries;
}
