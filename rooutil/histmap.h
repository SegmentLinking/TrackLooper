#ifndef histmap_h
#define histmap_h

#include "TFile.h"
#include "TH1.h"
#include "stringutil.h"
#include "printutil.h"

namespace RooUtil
{
    class HistMap
    {
        public:
            TFile* file;
            TH1* hist;
            int dimension;
            HistMap(TString histpath);
            ~HistMap();
            double eval(double);
            double eval(double, double);
            double eval(double, double, double);
            double eval_up(double);
            double eval_up(double, double);
            double eval_up(double, double, double);
            double eval_down(double);
            double eval_down(double, double);
            double eval_down(double, double, double);

    };
}


#endif
