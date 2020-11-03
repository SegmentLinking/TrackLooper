//  .
// ..: P. Chang, philip@physics.ucsd.edu

#ifndef varmap_h
#define varmap_h

// C/C++
#include <algorithm>
#include <fstream>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
#include <stdarg.h>
#include <functional>
#include <cmath>
#include <sstream>

// ROOT
#include "TBenchmark.h"
#include "TBits.h"
#include "TChain.h"
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TLeaf.h"
#include "TH1.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TChainElement.h"
#include "TTreeCache.h"
#include "TTreePerfStats.h"
#include "TStopwatch.h"
#include "TSystem.h"
#include "TString.h"
#include "TLorentzVector.h"
#include "Math/LorentzVector.h"

#include "stringutil.h"
#include "printutil.h"

using namespace std;

namespace RooUtil
{

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Var Map class
    ///////////////////////////////////////////////////////////////////////////////////////////////
    class VarMap
    {
        private:
            TString filename_;
        public:
            std::map<std::vector<int>, std::vector<float>> varmap_;
            VarMap();
            VarMap( TString filename, TString delim, int nkeys );
            ~VarMap();
            void load( TString, TString, int );
            std::vector<float> get( std::vector<int> );
    };

}

#endif
