//  .
// ..: P. Chang, philip@physics.ucsd.edu

#ifndef scripts_h
#define scripts_h

#include "dorky.h"
#include "TChain.h"
#include "TTree.h"
#include "TFile.h"

#include <iostream>

namespace RooUtil
{
    void remove_duplicate(TChain* chain, TString output, const char* run_bname, const char* lumi_bname, const char* evt_bname, int size=0);
    void split_files(TChain* chain, TString output, int size=5000000);
}

#endif
