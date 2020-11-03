//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQSampleListInitializer__
#define __TQSampleListInitializer__

#include "QFramework/TQSampleInitializerBase.h"
#include "QFramework/TQSampleFolder.h"

class TQSampleListInitializer : public TQSampleInitializerBase {
 protected:
  TQSampleFolder* fSampleFolder = NULL;

 public:
  
  TQSampleListInitializer(TQSampleFolder* sf);
  virtual ~TQSampleListInitializer();
  
  bool initializeSampleForFile(const TString& filepath);
  int initializeSamplesForFiles(TCollection* c);
  int initializeSamples();

  ClassDefOverride(TQSampleListInitializer, 0); // helper class to initialize samples from a list of files
};

#endif
