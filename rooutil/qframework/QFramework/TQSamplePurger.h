//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQSAMPLEPURGER__
#define __TQSAMPLEPURGER__

#include "QFramework/TQSampleVisitor.h"

class TQSamplePurger : public TQSampleVisitor {
protected:
  bool fPurgeFolders = true;
  bool fShallow = true;
  
  virtual int revisitFolder(TQSampleFolder * sampleFolder, TString& message) override;
  virtual int visitSample(TQSample * sample, TString& message) override;
  
public: 

  void setPurgeFolders(bool purge);
  void setShallow(bool shallow);
  
  TQSamplePurger(const TString& name);
  virtual ~TQSamplePurger();
  
  ClassDefOverride(TQSamplePurger, 0); // sample visitor to clean up a sample folder
 
};

#endif


