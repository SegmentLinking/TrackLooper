//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQSAMPLENORMALIZATIONOBSERVABLE__
#define __TQSAMPLENORMALIZATIONOBSERVABLE__
#include "QFramework/TQObservable.h"

class TQSampleNormalizationObservable : public TQObservable {
protected:
  // put here any data members your class might need
  
public:
  virtual double getValue() const override;
  virtual TObjArray* getBranchNames() const override;
  virtual Long64_t getCurrentEntry() const override;
protected:
  virtual bool initializeSelf() override;
  virtual bool finalizeSelf() override;
public:
  TQSampleNormalizationObservable();
  TQSampleNormalizationObservable(const TString& name);
  virtual ~TQSampleNormalizationObservable();
  ClassDefOverride(TQSampleNormalizationObservable, 1); //QFramework class


};
#endif
