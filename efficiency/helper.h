#ifndef helper_h
#define helper_h

#include "SDL.h"
#include "cxxopts.h"
#include "rooutil.h"
#include <../cppitertools/itertools.hpp>

class AnalysisConfig {

public:

    // TString that holds the input file list (comma separated)
    TString input_file_list_tstring;

    // TString that holds the name of the TTree to open for each input files
    TString input_tree_name;

    // Output TFile
    TFile* output_tfile;

    // Number of events to loop over
    int n_events;

    // Jobs to split (if this number is positive, then we will skip certain number of events)
    // If there are N events, and was asked to split 2 ways, then depending on job_index, it will run over first half or latter half
    int nsplit_jobs;

    // Job index (assuming nsplit_jobs is set, the job_index determine where to loop over)
    int job_index;

    // Debug boolean
    bool debug;

    // TChain that holds the input TTree's
    TChain* events_tchain;

    // Custom Looper object to facilitate looping over many files
    RooUtil::Looper<SDL> looper;

    // Custom Cutflow framework
    RooUtil::Cutflow cutflow;

    // Custom Histograms object compatible with RooUtil::Cutflow framework
    RooUtil::Histograms histograms;

    // Custom TTree object to hold intermediate variables
    RooUtil::TTreeX tx;

    // pt binning options
    int ptbound_mode;

    // pdgid
    int pdgid;

    AnalysisConfig();

};

extern AnalysisConfig ana;

class EfficiencySetDefinition {
public:
    TString set_name;
    int pdgid;
    std::function<bool(int)> pass;
    EfficiencySetDefinition(TString, int, std::function<bool(int)>);
};


void parseArguments(int argc, char** argv);
void initializeInputsAndOutputs();
std::vector<float> getPtBounds();
std::vector<int> matchedSimTrkIdxs(std::vector<int> hitidxs, std::vector<int> hittypes);

#endif
