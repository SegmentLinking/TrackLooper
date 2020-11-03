#include "QFramework/TQAnalysisSampleVisitor.h"

#include "QFramework/TQCounter.h"
#include "QFramework/TQUtils.h"
#include "QFramework/TQSample.h"
#include "QFramework/TQCut.h"
#include "QFramework/TQStringUtils.h"
#include "QFramework/TQIterator.h"

#include "TList.h"
#include "TObjString.h"
#include "TStopwatch.h"

#include "QFramework/TQAlgorithm.h"

// #define _DEBUG_
#include "QFramework/TQLibrary.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQAnalysisSampleVisitor:
//
// Visit samples and execute analysis jobs at cuts. 
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQAnalysisSampleVisitor)


//__________________________________________________________________________________|___________

TQAnalysisSampleVisitor::TQAnalysisSampleVisitor() : 
TQAnalysisSampleVisitorBase("asv",false)
{
  // Default constructor of the TQAnalysisSampleVisitor class
  this->setVisitTraceID("analysis");
}

//__________________________________________________________________________________|___________

TQAnalysisSampleVisitor::~TQAnalysisSampleVisitor(){
  // default destructor
}

//__________________________________________________________________________________|___________

TQAnalysisSampleVisitor::TQAnalysisSampleVisitor(TQCut* base, bool verbose) : 
  TQAnalysisSampleVisitorBase("asv",verbose),
  fBaseCut(base)
{
  // constructor with base cut
  this->setVisitTraceID("analysis");
}



//__________________________________________________________________________________|___________

int TQAnalysisSampleVisitor::visitFolder(TQSampleFolder * sampleFolder, TString& /*message*/) {
  // visit an instance of TQSampleFolder
  if(!fBaseCut)
    return visitFAILED;

  if(!fBaseCut->initializeSampleFolder(sampleFolder))
    return visitFAILED;

  this->stamp(sampleFolder);

  return visitLISTONLY;
}

//__________________________________________________________________________________|___________

int TQAnalysisSampleVisitor::revisitSample(TQSample * /*sample*/, TString& /*message*/) {
  // revisit an instance of TQSample on the way out
  return visitIGNORE;
}



//__________________________________________________________________________________|___________

int TQAnalysisSampleVisitor::revisitFolder(TQSampleFolder * sampleFolder, TString& message) {
  // revisit an instance of TQSampleFolder on the way out
  bool finalized = fBaseCut->finalizeSampleFolder(sampleFolder);
 
  bool generalizeHistograms = false;
  bool generalizeCounter = false;
  //@tag: [.asv.generalize.<objectType>] These folder tags determine if objects of type "<objectType>" (=histograms,counter,countergrid) should be generalized using TQSampleFolder::generalizeObjects. Default: true
  sampleFolder->getTagBool("generalize.histograms", generalizeHistograms, true);
  sampleFolder->getTagBool("generalize.counter", generalizeCounter, true);

  bool generalized = false;
  bool doGeneralize = generalizeHistograms || generalizeCounter;

  if (doGeneralize){

    /* start the stop watch */
    TStopwatch * timer = new TStopwatch();

    /* generalize all histograms */
    int nHistos = 0;
    if (generalizeHistograms)
      nHistos = sampleFolder->generalizeHistograms();

    /* generalize all counter */
    int nCounter = 0;
    if (generalizeCounter)
      nCounter = sampleFolder->generalizeCounters();

    /* stop the timer */
    timer->Stop();

    message.Append(" ");
    message.Append(TQStringUtils::fixedWidth("--", 12,"r"));
    message.Append(TQStringUtils::fixedWidth(TString::Format("%.2f", timer->RealTime()), 12,"r"));
    message.Append(" ");
    
    if (nHistos == 0 && nCounter == 0)
      message.Append("nothing to generalize");
    else{
      message.Append("generalized: ");
      if(nHistos > 0){
        if (generalizeHistograms) message.Append(TString::Format("%d histograms", nHistos));
      }
      if(nCounter > 0 && nHistos > 0){
        message.Append(", ");
      }
      if(nCounter > 0){
        message.Append(TString::Format("%d counters", nCounter));
      }
    }
 
    /* delete the timer */
    delete timer;

    if (nHistos > 0 || nCounter > 0)
      generalized = true;

  } else {
    if(finalized)
      return visitIGNORE;
  }
  if(generalized || !doGeneralize){
    if(finalized){
      return visitOK;
    } else
      return visitWARN;
  } else {
    if(finalized)
      return visitWARN;
  }
  return visitFAILED;
}


//__________________________________________________________________________________|___________

int TQAnalysisSampleVisitor::visitSample(TQSample * sample, TString& message) {
  // visit an instance of TQSample
  TStopwatch * timer = new TStopwatch();

  /* analyse the tree */
  TString analysisMessage;
  #ifndef _DEBUG_
  TQLibrary::redirect_stderr("/dev/null");
  #endif
  DEBUGclass("analysing tree");
  int nEntries = analyseTree(sample, analysisMessage);
  #ifndef _DEBUG_
  TQLibrary::restore_stderr();
  #endif
  /* stop the timer */
  timer->Stop();

  /* compile the message */
  message.Append(" ");

  //@tag: [.asv.analysis.nentries] This sample tag is set by TQAnalysisSampleVisitor, containing the (raw) number of events analyzed in a tree.
  /* save the number of entries in tree analyzed */
  sample->setTagInteger(".asv.analysis.nentries", nEntries);

  if (nEntries >= 0) {
    message.Append(TQStringUtils::fixedWidth(TQStringUtils::getThousandsSeparators(nEntries), 12,"r"));
  } else {
    message.Append(TQStringUtils::fixedWidth("--", 12,"r"));
  }
 
  message.Append(TQStringUtils::fixedWidth(TString::Format("%.2f", timer->RealTime()), 12,"r"));
  message.Append(" ");
  message.Append(TQStringUtils::fixedWidth(analysisMessage, 40, "l"));

  /* delete the timer */
  delete timer;

  if(nEntries > 0){
    this->stamp(sample);
    return visitOK;
  }
  else if (nEntries == 0)
    return visitWARN;

  return visitFAILED;
 
}


//__________________________________________________________________________________|___________

void TQAnalysisSampleVisitor::setBaseCut(TQCut * baseCut) {
  // set the base cut
  if (baseCut) fBaseCut = baseCut; 
  else WARNclass("attempt to set baseCut=NULL");
}


//__________________________________________________________________________________|___________

TQCut * TQAnalysisSampleVisitor::getBaseCut() {
  // get the base cut
  return fBaseCut;
}


//__________________________________________________________________________________|___________

int TQAnalysisSampleVisitor::analyseTree(TQSample * sample, TString& message) {
  // analyse the tree in this sample
  DEBUGclass("entering function");
  DEBUGclass("testing sample");
  if (!sample) {
    message = TString("sample is NULL");
    DEBUGclass(message);
    return -1;
  }

  DEBUGclass("testing basecut");
  if (!fBaseCut) {
    message = TString("no base cut given!");
    DEBUGclass(message);
    return -1;
  }

  /* try to get tree token */
  DEBUGclass("obtaining tree token");
  TQToken * treeToken = sample->getTreeToken();
  if(!treeToken){
    message = TString("failed to get tree token");
    DEBUGclass(message);
    return -1;
  }
  DEBUGclass("retrieving tree");
  TTree * tree = (TTree*)(treeToken->getContent());
  if(!tree){
    message = TString("failed to retrieve tree");
    DEBUGclass(message);
    sample->returnToken(treeToken);
    return -1;
  }

  DEBUGclass("owning tree token");
  treeToken->setOwner(this);
 
  int nEntries = -1;

  DEBUGclass("initializing sample '%s'",sample->GetName());
  if (fBaseCut->initialize(sample)) {
    /* initialize the tree */
    TObjArray* branchNames = fBaseCut->getListOfBranches();
    branchNames->SetOwner(true);
    tree->SetBranchStatus("*", 0);
    this->setupBranches(tree,branchNames);
    delete branchNames;

    DEBUGclass("retrieving number of entries");
    Long64_t nEntriesOrig = tree->GetEntries();
    nEntries = std::min(nEntriesOrig,this->fMaxEvents);

    /* check wether to use MC weights */
    DEBUGclass("testing for usemcweights tag");
    bool useWeights = false;
    //@tag: [usemcweights] This sample tag controlls if (MC) weights are applied for the respective sample.
    sample->getTagBool("usemcweights", useWeights, true);

    if(useWeights && nEntries!=nEntriesOrig){
      // due to popular request, we want to normalization be correct even when *only doing debugging*
      double norm = double(nEntriesOrig)/double(nEntries);
      sample->setNormalisation(sample->getNormalisation() * norm);
    }
    
    if(this->initializeAlgorithms(sample)){
      /* loop over tree entries */
      DEBUGclass("entering event loop");
      for (int i = 0; i < nEntries; ++i) {
        tree->GetEntry(i);
        DEBUGclass(" visiting entry %d, executing algorithms",i);
        if(!this->executeAlgorithms()){
          nEntries = -1;
          message = TString("failure in algorithm execute");
        } else {
          DEBUGclass(" analysing entry %d",i);
          this->fBaseCut->analyse(1., useWeights);
          this->cleanupAlgorithms();
        }
      }
      if(!this->finalizeAlgorithms()){
        message = TString("failure in algorithm finalize");
      }
    } else {
      message = TString("failure in algorithm initialize");
    }
    
      
#ifdef _DEBUG_
    TQObservable::printObservables();
    TQUtils::printActiveBranches(tree);
    fBaseCut->printCuts();
#endif

  } else {
    message =TString("failed to initialize analysis chain");
    DEBUGclass(message);
  }
 
  if (fBaseCut && !fBaseCut->finalize()) {
    message = TString("failed to finalize analysis chain");
  }
 
  sample->returnTreeToken(treeToken);
 
  DEBUGclass("finished analyzing sample '%s'",sample->GetName());

  if (sample->getNTreeTokens() > 0) {
    std::cout << std::endl;
    message=TString("sample left with # tree tokens > 0");
    sample->printTreeTokens();
    std::cout << std::endl;
  }
 
  return nEntries;
}

void TQAnalysisSampleVisitor::setReduceBranches(bool reduce){
  // set the branch usage policy to reduced branches (true) or all branches (false)
  this->setUseBranches(reduce ? ReducedBranches : AllBranches);
}
