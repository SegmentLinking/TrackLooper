//  .
// ..: P. Chang, philip@physics.ucsd.edu

#ifndef autohist_cc
#define autohist_cc

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

//#define MAP std::unordered_map
//#define STRING std::string
#define MAP std::map
#define STRING TString

#include "printutil.h"

namespace RooUtil
{

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Auto histogram maker
    ///////////////////////////////////////////////////////////////////////////////////////////////
    // The concept is to keep the histogram at precision of 0.001.
    // So far in my experience of HEP, I never had to deal with precision more than 1/1000th
    // This means that if I keep every histogram at the precision of Range / N bin = 0.001,
    // I can always reproduce any other bin size histogram offline.
    // So, all I have to do is start with some small number of histograms binned in 0.001,
    // And slowly expand as the numbers come in.
    // For small range histograms (ip, or isolation), we can still create at 1/1000-th precision.
    // Then when the values that come in hits 50 or 100 (configurable), we blow it up.
    class AutoHist
    {

        public:
            int resolution;
            MAP<STRING, TH1*> histdb;

            //////////////////////////////////////////////////////////////////////////////////////////////////
            // Functions
            //////////////////////////////////////////////////////////////////////////////////////////////////

            AutoHist();
            ~AutoHist();
            // user interface
            void fill( double xval, STRING name, double wgt = 1 );
            void fill( double xval, STRING name, double wgt, int nbin, double xmin, double xmax,
                    std::vector<TString> = std::vector<TString>() );
            void fill( double xval, double yval, STRING name, double wgt, int, double, double, int, double, double,
                    std::vector<TString> = std::vector<TString>() );
            void fill( double xval, STRING name, double wgt, int nbin, double* );
            void fill( double xval, double yval, STRING name, double wgt, int, double*, int, double* );
            void save( TString ofilename, TString option = "recreate" );
            void save( TFile* ofile );
            // under the hood (but not private...)
            void fill( double xval, TH1*& h, double wgt = 1, bool norebinning = false );
            TH1* hadd( TH1*, TH1* );
            TH1* get( STRING );
            void print();
            static int getRes( double range );
            static int getRes( TH1* h );
            static void transfer( TH1*, TH1* );
            static TH1* crop( TH1*, int, double, double );
            static TH1* createHist( double xval, TString name, double wgt = 1, bool alreadyneg = false, int forceres = -1 );
            static TH1* createFixedBinHist( double, TString, double, int, double, double );
            static TH1* createFixedBinHist( double, TString, double, int, double* );
            static TH1* createFixedBinHist( double, double, TString, double, int, double, double, int, double, double );
            static TH1* createFixedBinHist( double, double, TString, double, int, double*, int, double* );

    };

}

#endif
