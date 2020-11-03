#include <TString.h>
#include <TFile.h>
#include <TChain.h>

TChain* rch(TString input, TString treeName="t")
{
    TChain *t = new TChain(treeName);
    t->Add(input);
    return t;
}
