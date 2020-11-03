//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQSampleInitializer__
#define __TQSampleInitializer__

#include "TList.h"
#include "QFramework/TQSampleInitializerBase.h"
#include "QFramework/TQSampleVisitor.h"
#include "QFramework/TQSample.h"

class TQSampleInitializer : public TQSampleVisitor, public TQSampleInitializerBase {
protected:
    virtual int visitSample(TQSample * sample, TString& message) override;
public:
  TQSampleInitializer();
  TQSampleInitializer(const TString& dirname, int depth);
  virtual ~TQSampleInitializer();
 
  bool getExitOnFail();
  void setExitOnFail(bool ex);

  ClassDefOverride(TQSampleInitializer, 0); // sample visitor to initialize samples
};

#endif
