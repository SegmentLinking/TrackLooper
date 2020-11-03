//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQTreeObservable__
#define __TQTreeObservable__

#include "QFramework/TQObservable.h"
#include "TTree.h"

class TQTreeObservable : public TQObservable {
protected:
  TQToken* fTreeToken = NULL;
  TTree* fTree = NULL;

public:

  TQTreeObservable();
  TQTreeObservable(const TString& expression);
  virtual ~TQTreeObservable();

  virtual bool initialize(TQSample * sample) override;
  virtual bool finalize() override;

  virtual Long64_t getCurrentEntry() const override;
  virtual void print() const override;

  ClassDefOverride(TQTreeObservable, 0); //QFramework class
};

#endif
