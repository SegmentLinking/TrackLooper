#include "TTree.h"
#include "TTreeFormula.h"
#include "TH1F.h"

#include "QFramework/TQSampleVisitor.h"
#include "QFramework/TQAnalysisSampleVisitorBase.h"
#include "QFramework/TQCounter.h"
#include "QFramework/TQUtils.h"
#include "QFramework/TQSample.h"
#include "QFramework/TQStringUtils.h"
#include "QFramework/TQIterator.h"

#include "TList.h"
#include "TObjString.h"
#include "TStopwatch.h"

//#define _DEBUG_
#include "QFramework/TQLibrary.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQAnalysisSampleVisitorBase:
//
// An abstract base class for various types of analysis sample visitors.
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQAnalysisSampleVisitorBase)

//__________________________________________________________________________________|___________


TQAnalysisSampleVisitorBase::TQAnalysisSampleVisitorBase(const TString& name, bool verbose) : 
  TQSampleVisitor(name),
  fUseBranches(UseBranches::ReducedBranches),
  fMaxEvents(LLONG_MAX)
{
  // constructor with name and verbosity setting
  this->setVerbose(verbose);
}


//__________________________________________________________________________________|___________

int TQAnalysisSampleVisitorBase::initialize(TQSampleFolder */*sampleFolder*/, TString& message) {
  // initialize a sample folder
  message.Append(" ");
  message.Append(TQStringUtils::fixedWidth("# Entries", 12,"r"));
  message.Append(TQStringUtils::fixedWidth("Time [sec]", 12,"r"));
  message.Append(" ");
  message.Append(TQStringUtils::fixedWidth("Message", 40, "l")); 

  return visitOK;
 
}

//__________________________________________________________________________________|___________

void TQAnalysisSampleVisitorBase::setUseBranches(UseBranches branchSetting){
  // set the branch policy
  this->fUseBranches = branchSetting;
}

//__________________________________________________________________________________|___________

void TQAnalysisSampleVisitorBase::setMaxEvents(Long64_t max){
  // set the maximum number of events on to analyse per sample (debugging option)
  this->fMaxEvents = max;
}

//__________________________________________________________________________________|___________

TQAnalysisSampleVisitorBase::~TQAnalysisSampleVisitorBase() {
  // default destructor
}

//__________________________________________________________________________________|___________

bool TQAnalysisSampleVisitorBase::setupBranches(TTree* tree, TCollection* branchNames){
	// setup all the branches in the given tree, depending on the branch policy and the given list of branch names
  if(!tree || !branchNames) return false;
  if (fUseBranches == UseBranches::ReducedBranches) {
    // use only branches found in cuts and jobs 
    DEBUGclass("enabling requested branches");
    TQIterator itr(branchNames);
    while(itr.hasNext()){
      TObject* bName = itr.readNext();
      if(!bName) continue;
      TString name(bName->GetName());
      if (name.First('*') != kNPOS || tree->FindBranch(name)){
        tree->SetBranchStatus(name, 1);
      }
    }
    return true;
  } else if(fUseBranches == UseBranches::TTreeCache){
    // use TTreeCache 
    WARNclass("TTreeCache branch management is not yet implemented!");
    // TODO!
    tree->SetBranchStatus("*", 1);
    return true;
  } else {
    /* use all branches */
    DEBUGclass("enabling all branches");
    tree->SetBranchStatus("*", 1);
    return true;
  }
}
