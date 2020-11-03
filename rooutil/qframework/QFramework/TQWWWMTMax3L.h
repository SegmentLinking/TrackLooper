//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQWWWMTMAX3L__
#define __TQWWWMTMAX3L__
#include "QFramework/TQTreeObservable.h"
#include "TTreeFormula.h"

class TQWWWMTMax3L : public TQTreeObservable {
protected:
  // put here any data members your class might need
  TTreeFormula* fFormula;
 
public:
  virtual double getValue() const override;
  virtual TObjArray* getBranchNames() const override;
protected:
  virtual bool initializeSelf() override;
  virtual bool finalizeSelf() override;
protected:
  TString fExpression = "";

public:
  virtual bool hasExpression() const override;
  virtual const TString& getExpression() const override;
  virtual void setExpression(const TString& expr) override;

  TQWWWMTMax3L();
  TQWWWMTMax3L(const TString& expression);
  virtual ~TQWWWMTMax3L();
  ClassDefOverride(TQWWWMTMax3L, 1);


};
#endif
