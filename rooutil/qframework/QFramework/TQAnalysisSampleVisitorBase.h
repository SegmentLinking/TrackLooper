//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQAnalysisSampleVisitorBase__
#define __TQAnalysisSampleVisitorBase__

#include "QFramework/TQSampleVisitor.h"
#include "QFramework/TQTaggable.h"
#include "TTree.h"
#include <climits>

class TQAnalysisSampleVisitorBase : public TQSampleVisitor, public TQTaggable {
 
public:
  enum UseBranches {
    AllBranches = 0,
    ReducedBranches = 1,
    TTreeCache = 2
  };
 
protected:
 
  UseBranches fUseBranches;
  Long64_t fMaxEvents;
  bool setupBranches(TTree* tree, TCollection* branchNames);

  virtual int visitFolder(TQSampleFolder * sampleFolder, TString& message)  override = 0;
  virtual int revisitFolder(TQSampleFolder * sampleFolder, TString& message) override = 0;
  virtual int visitSample(TQSample * sample, TString& message) override = 0;
  virtual int revisitSample(TQSample * sample, TString& message) override = 0;
  
public: 
 
  TQAnalysisSampleVisitorBase(const TString& name, bool verbose = false);
  virtual ~TQAnalysisSampleVisitorBase();
  
  virtual int initialize(TQSampleFolder * sampleFolder, TString& message) override;
 
  void setUseBranches(UseBranches branchSetting = ReducedBranches);

  void setMaxEvents(Long64_t max = LLONG_MAX);
 
  ClassDefOverride(TQAnalysisSampleVisitorBase, 0); // base class for analysis sample visitors
 
};

#endif


