#!/bin/bash

files=""
tree="Events"
scanner="scanner.C"
outtxt="runLumisOutput.txt"
outjson="runLumisOutput.json"

if [ $# -lt 1 ]; then
    echo "findLumis -t [treeName] -i \"/path/to/files/*.root\""
    exit
fi

OPTIND=1
while getopts "i:t:" opt; do
  case $opt in
      i) files=$OPTARG;;
      t) tree=$OPTARG;;
  esac
done
shift $((OPTIND-1))

# make scanner

echo "" > $scanner
echo "#include <iostream>
#include <fstream>
#include \"TChain.h\"
#include \"TFile.h\"
#include \"TROOT.h\"
using namespace std;
// gErrorIgnoreLevel=kError;
void ${scanner%.*}() {
    TChain *chain = new TChain(\"$tree\");
    chain->Add(\"$files\");
    TObjArray *listOfFiles = chain->GetListOfFiles();
    TIter fileIter(listOfFiles);
    TFile *currentFile = 0;
    ofstream runLumiOutput;
    runLumiOutput.open(\"$outtxt\");
    int prevRun = -1, prevLumi = -1;
    while ( (currentFile = (TFile*)fileIter.Next()) ) {
        TFile *file = new TFile( currentFile->GetTitle() );
        TTree *tree = (TTree*)file->Get(\"$tree\");
        TString filename(currentFile->GetTitle());
        int run, lumi;
        TBranch *run_branch = tree->GetBranch(\"evt_run\");
        run_branch->SetAddress(&run);
        tree->GetBranch(\"run\"); 
        TBranch *lumi_branch = tree->GetBranch(\"evt_lumiBlock\");
        lumi_branch->SetAddress(&lumi);
        tree->GetBranch(\"lumi\"); 
        for( unsigned int event = 0; event < tree->GetEntriesFast(); ++event) {
            tree->GetEntry(event);
            if(run != prevRun || lumi != prevLumi) {
                runLumiOutput << run << \":\" << lumi << \"\n\";
                prevRun = run;
                prevLumi = lumi;
            }
        }
        delete tree;
        file->Close();
        delete file;
    }
    runLumiOutput.close();
}" >> $scanner

CMSSW_VERSION=CMSSW_7_4_1
cd /cvmfs/cms.cern.ch/slc6_amd64_gcc491/cms/cmssw/$CMSSW_VERSION/src
eval `scramv1 runtime -sh`
cd -

root -l -b -q $scanner
csv2json.py $outtxt > $outjson

mkdir -p runcsv
rsync -rP --ignore-existing  lxplus.cern.ch:/afs/cern.ch/user/m/marlow/public/lcr2/lcr2.py lxplus.cern.ch:/afs/cern.ch/user/m/marlow/public/lcr2/runcsv/ runcsv
mv runcsv/lcr2.py .

python lcr2.py -i $outjson

