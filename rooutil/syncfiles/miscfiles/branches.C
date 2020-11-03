#include <TString.h>
#include <TFile.h>

void branches(TString input, TString treeName="Events")
{
    TFile *file = new TFile(input);
    TTree *tree = (TTree*)file->Get(treeName);
    if(treeName.Contains("Events")) {
        for(int i = 0; i < tree->GetListOfAliases()->LastIndex(); i++) 
            std::cout << "branch: " << tree->GetListOfAliases()->At(i)->GetName() << std::endl;
    } else {
        for(int i = 0; i < tree->GetListOfBranches()->LastIndex(); i++) 
            std::cout << "branch: " << tree->GetListOfBranches()->At(i)->GetName() << std::endl;
    }
}
