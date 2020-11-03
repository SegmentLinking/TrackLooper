//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQWWWMTONELEP__
#define __TQWWWMTONELEP__
#include "QFramework/TQTreeObservable.h"
#include "TTreeFormula.h"

class TQWWWMTOneLep : public TQTreeObservable {
protected:
  // put here any data members your class might need
  TTreeFormula* fFormula;
 
public:
  virtual double getValue() const override;
  virtual TObjArray* getBranchNames() const override;
protected:
  virtual bool initializeSelf() override;
  virtual bool finalizeSelf() override;
public:
  TQWWWMTOneLep();
  TQWWWMTOneLep(const TString& name);
  virtual ~TQWWWMTOneLep();
  ClassDefOverride(TQWWWMTOneLep, 1);


};
#endif
