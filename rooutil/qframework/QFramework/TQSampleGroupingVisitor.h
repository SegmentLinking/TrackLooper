//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQSAMPLEGROUPINGVISITOR__
#define __TQSAMPLEGROUPINGVISITOR__

#include "QFramework/TQSampleVisitor.h"
#include <TString.h>
#ifndef __CINT__
#include <unordered_set>
#include <string>
#endif

class TQSampleDataReader;

class TQSampleGroupingVisitor : public TQSampleVisitor {
protected:
  TQSampleDataReader* fReader = NULL;
  TString fCounterName = "initial";
  int fEventLimit = 10e6;
  TString fActiveItemName = "";
  int fActiveItemCount = 0;
  #ifndef __CINT__
  std::unordered_set<std::string> fPaths;
  #endif

  virtual int initialize(TQSampleFolder * sampleFolder, TString& message) override;
  virtual int finalize() override;
  virtual int visitFolder(TQSampleFolder * sampleFolder, TString& message) override;
  virtual int revisitFolder(TQSampleFolder * sampleFolder, TString& message) override;
  virtual int visitSample(TQSample * sample, TString& message) override;
  
public: 

  void setCounterName(const TString& name);
  void setEventLimit(int nEvents);
  TString getCounterName();
  int getEventLimit();

  std::vector<TString> getPaths();

  TQSampleGroupingVisitor(const char* counterName, int nEvents);
  virtual ~TQSampleGroupingVisitor();
  
  ClassDefOverride(TQSampleGroupingVisitor, 0); // sample visitor for grouping samples
 
};

#endif


