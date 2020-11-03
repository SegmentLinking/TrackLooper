//  .
// ..: P. Chang, philip@physics.ucsd.edu

#ifndef cuttool_h
#define cuttool_h

// C/C++
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
#include <stdarg.h>
#include <functional>
#include <cmath>

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

#include "printutil.h"

namespace RooUtil
{
    class CutTool
    {
        private:
            std::map<TString, bool> cache;
        public:
            CutTool() {}
            ~CutTool() {}
            void clearCache() { cache.clear(); }
            bool passesCut(TString name, std::function<bool()> cut, std::vector<TString> nm1=std::vector<TString>());
            bool passesCut(TString name);
    };
}

#endif
