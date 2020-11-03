//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQCONSTTREEOBSERVABLE__
#define __TQCONSTTREEOBSERVABLE__

#include "QFramework/TQObservable.h"

class TQConstObservable : public TQObservable {

protected:
  TString fExpression = "";
  double fValue = 0;

  virtual bool initializeSelf() override;
  virtual bool finalizeSelf() override;

  virtual Long64_t getCurrentEntry() const override;

public:

  TQConstObservable();
  TQConstObservable(const TString& expression);

  virtual double getValue() const override;
 
  virtual TObjArray* getBranchNames() const override;
 
  virtual bool hasExpression() const override;

  virtual const TString& getExpression() const override;
  virtual void setExpression(const TString& expr) override;

  virtual ~TQConstObservable();

  DECLARE_OBSERVABLE_FACTORY(TQConstObservable,TString expression)
  
  ClassDefOverride(TQConstObservable, 0); // observable for constant values
 
};

#endif
