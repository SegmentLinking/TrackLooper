//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQEVENTINDEXOBSERVABLE__
#define __TQEVENTINDEXOBSERVABLE__
#include "QFramework/TQTreeObservable.h"

class TQEventIndexObservable : public TQTreeObservable {
protected:
  virtual bool initializeSelf() override;
  virtual bool finalizeSelf() override;
  
public:
  virtual double getValue() const override;
  virtual TObjArray* getBranchNames() const override;
public:
  TQEventIndexObservable();
  TQEventIndexObservable(const TString& name);
  virtual ~TQEventIndexObservable();

  ClassDefOverride(TQEventIndexObservable, 0); // observable that returns the event index in the tree
};
#endif
