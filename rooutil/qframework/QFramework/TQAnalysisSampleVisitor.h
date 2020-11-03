//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQAnalysisSampleVisitor__
#define __TQAnalysisSampleVisitor__

#include "QFramework/TQCut.h"

#include "QFramework/TQAnalysisSampleVisitorBase.h"
#include "QFramework/TQAlgorithm.h"

class TQAnalysisSampleVisitor : public TQAnalysisSampleVisitorBase, public TQAlgorithm::Manager {
 
protected:
  
  TQCut * fBaseCut;

  int analyseTree(TQSample * sample, TString& message);

  virtual int visitFolder(TQSampleFolder * sampleFolder, TString& message) override;
  virtual int revisitFolder(TQSampleFolder * sampleFolder, TString& message) override;
  virtual int visitSample(TQSample * sample, TString& message) override;
  virtual int revisitSample(TQSample * sample, TString& message) override;
  
public: 
 
  TQAnalysisSampleVisitor();
  TQAnalysisSampleVisitor(TQCut* base, bool verbose = false);

  void setBaseCut(TQCut * baseCut);
  TQCut * getBaseCut();

  void setReduceBranches(bool reduce=true);
 
  virtual ~TQAnalysisSampleVisitor();
 
  ClassDefOverride(TQAnalysisSampleVisitor, 0); // visitor that performs a physics analysis with one or more analysis jobs
 
};

#endif


