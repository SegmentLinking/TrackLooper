//  .
// ..: P. Chang, philip@physics.ucsd.edu

#ifndef eventlist_cc
#define eventlist_cc

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

//#define MAP std::unordered_map
//#define STRING std::string
#define MAP std::map
#define STRING TString

using namespace std;

namespace RooUtil
{

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Event List class
    ///////////////////////////////////////////////////////////////////////////////////////////////
    class EventList
    {
        public:
            std::vector<std::vector<int>> event_list;
            EventList();
            EventList( TString filename, TString delim=":" );
            ~EventList();
            void load( TString, TString=":" );
            bool has( int, int, int );
    };

}


#endif
