#ifndef AnalysisConfig_h
#define AnalysisConfig_h

#include "TString.h"
#include "rooutil.h"
#include "trktree.h"
#include "SDL/ModuleConnectionMap.h"

class AnalysisConfig
{

public:

    // Analysis run mode
    int mode;

    // TString that holds the input file list (comma separated)
    TString input_file_list_tstring;

    // TString that holds the name of the TTree to open for each input files
    TString input_tree_name;

    // Output TFile
    TFile* output_tfile;

    // Number of events to loop over
    int n_events;

    // specific event_index to process
    int specific_event_index;

    // run efficiency study
    bool run_eff_study;

    // run inefficiency study
    bool run_ineff_study;

    // run inefficiency study
    int mode_write_ineff_study_debug_ntuple; // 0 = MDs, 1 = SGs, 2 = TLs, 3 = TCs

    // run MTV study
    bool run_mtv_study;

    // pt binning options
    int ptbound_mode;

    // pdg id of the particles to compute efficincies on
    int pdg_id;

    // verbose of the particles to compute efficincies on
    int verbose;

    // to print module connection info
    bool print_conn;

    // to print module boundary info
    bool print_boundary;

    // to print centroid
    bool print_centroid;

    // Debug boolean
    bool debug;

    // TChain that holds the input TTree's
    TChain* events_tchain;

    // Jobs to split (if this number is positive, then we will skip certain number of events)
    // If there are N events, and was asked to split 2 ways, then depending on job_index, it will run over first half or latter half
    int nsplit_jobs;

    // Job index (assuming nsplit_jobs is set, the job_index determine where to loop over)
    int job_index;

    // Custom Looper object to facilitate looping over many files
    RooUtil::Looper<trktree> looper;

    // Custom Cutflow framework
    RooUtil::Cutflow cutflow;

    // Custom Histograms object compatible with RooUtil::Cutflow framework
    RooUtil::Histograms histograms;

    // TTree output for studies
    TTree* output_ttree;

    // TTreeX to facilitate writing output to ttree
    RooUtil::TTreeX* tx;

    // Module boundaries related info
    std::map<int, std::array<float, 6>> moduleBoundaries;
    std::map<int, std::vector<std::vector<float>>> moduleSimHits;
    std::map<int, int> modulePopulation;

    SDL::ModuleConnectionMap moduleConnectiongMapLoose;
};

extern AnalysisConfig ana;

#endif
