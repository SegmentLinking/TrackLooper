//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQSAMPLEREVISITOR__
#define __TQSAMPLEREVISITOR__

#include "QFramework/TQSampleVisitor.h"
#include <TString.h>

class TQFolder;

class TQSampleRevisitor : public TQSampleVisitor {
protected:
  virtual int visitFolder(TQSampleFolder * sampleFolder, TString& message) override;
  virtual int visitSample(TQSample * sample, TString& message) override;
  virtual int revisitFolder(TQSampleFolder * sampleFolder, TString& message) override;
  virtual int revisitSample(TQSample * sample, TString& message) override;

  int readTrace(TQFolder* f, const TString& prefix, TString& message);
  
public: 

  TQSampleRevisitor();
  TQSampleRevisitor(const char* name);
  virtual ~TQSampleRevisitor();
  
  ClassDefOverride(TQSampleRevisitor, 0); // sample visitor to replay the activities of another one
 
};

#endif


