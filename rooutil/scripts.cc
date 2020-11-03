#include "scripts.h"

void RooUtil::remove_duplicate(TChain* chain, TString output, const char* run_bname, const char* lumi_bname, const char* evt_bname, int size)
{
    using namespace std;
    cout << chain->GetEntries() << endl;
    TFile* ofile = TFile::Open(output, "RECREATE");
    TTree* otree = chain->CloneTree(0);

    if (size)
    {
        otree->SetMaxTreeSize(size);
    }

    int run = 0;
    int lumi = 0;
    unsigned long long evt = 0;

    chain->SetBranchAddress(run_bname, &run);
    chain->SetBranchAddress(lumi_bname, &lumi);
    chain->SetBranchAddress(evt_bname, &evt);

    Long64_t totalnevents = chain->GetEntries();
    for (Long64_t ientry = 0; ientry < totalnevents; ++ientry)
    {
        if (ientry % 10000 == 0)
            cout << "Processed " << ientry << " events out of " << totalnevents << endl;
        chain->GetEntry(ientry);
        duplicate_removal::DorkyEventIdentifier id(run, evt, lumi);
        if (duplicate_removal::is_duplicate(id))
            continue; 
        otree->Fill();
    }

    otree->Write();
    ofile->Close();
}

void RooUtil::split_files(TChain* chain, TString output, int size)
{
    using namespace std;
    cout << chain->GetEntries() << endl;
    TFile* ofile = TFile::Open(output, "RECREATE");
    TTree* otree = chain->CloneTree(0);

    if (size)
    {
        otree->SetMaxTreeSize(size);
    }

    Long64_t totalnevents = chain->GetEntries();
    for (Long64_t ientry = 0; ientry < totalnevents; ++ientry)
    {
        if (ientry % 10000 == 0)
            cout << "Processed " << ientry << " events out of " << totalnevents << endl;
        chain->GetEntry(ientry);
        otree->Fill();
    }

    otree->Write();
    if (ofile)
        ofile->Close();
}
