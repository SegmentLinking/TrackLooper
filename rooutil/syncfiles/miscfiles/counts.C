#include <TString.h>

void counts(TString input, TString treeName="Events", TString directory="default")
{
    gErrorIgnoreLevel=kWarning;
    if(!directory.Contains("default")) {
        treeName = directory + "/" + treeName;
    }
    TChain * ch = new TChain(treeName);
    if(input.Contains(".root")) {
        ch->Add(input);
    } else {
        ch->Add(input+"/*.root");
    }
    std::cout << "Events: " << ch->GetEntries() << std::endl;
}
