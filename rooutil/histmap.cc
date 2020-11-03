#include "histmap.h"

//_________________________________________________________________________________________________
RooUtil::HistMap::HistMap(TString histpath)
{
    std::vector<TString> parsed = RooUtil::StringUtil::split(histpath, ":");
    if (parsed.size() != 2)
        error(TString::Format("HistMap instantitation should have a valid string HistMap(\"/path/to/rootfile.root:h_2dhist\"). The last field is the dimension.\nYou asked = %s", histpath.Data()));
    file = new TFile(parsed[0]);
    hist = (TH1*) file->Get(parsed[1]);
//    dimension = parsed[2].Atoi();
}

//_________________________________________________________________________________________________
RooUtil::HistMap::~HistMap()
{
    // if (file)
    //     file->Close();
}

//_________________________________________________________________________________________________
double RooUtil::HistMap::eval(double x)                     { return hist->GetBinContent(hist->FindBin(x));       }
double RooUtil::HistMap::eval(double x, double y)           { return hist->GetBinContent(hist->FindBin(x, y));    }
double RooUtil::HistMap::eval(double x, double y, double z) { return hist->GetBinContent(hist->FindBin(x, y, z)); }

//_________________________________________________________________________________________________
double RooUtil::HistMap::eval_up(double x)                     { return hist->GetBinContent(hist->FindBin(x))       + hist->GetBinError(hist->FindBin(x));       }
double RooUtil::HistMap::eval_up(double x, double y)           { return hist->GetBinContent(hist->FindBin(x, y))    + hist->GetBinError(hist->FindBin(x, y));    }
double RooUtil::HistMap::eval_up(double x, double y, double z) { return hist->GetBinContent(hist->FindBin(x, y, z)) + hist->GetBinError(hist->FindBin(x, y, z)); }

//_________________________________________________________________________________________________
double RooUtil::HistMap::eval_down(double x)                     { return hist->GetBinContent(hist->FindBin(x))       - hist->GetBinError(hist->FindBin(x));       }
double RooUtil::HistMap::eval_down(double x, double y)           { return hist->GetBinContent(hist->FindBin(x, y))    - hist->GetBinError(hist->FindBin(x, y));    }
double RooUtil::HistMap::eval_down(double x, double y, double z) { return hist->GetBinContent(hist->FindBin(x, y, z)) - hist->GetBinError(hist->FindBin(x, y, z)); }
