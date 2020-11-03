//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQTREEFORMULAOBSERVABLE__
#define __TQTREEFORMULAOBSERVABLE__

#include "TTreeFormula.h"
#include "QFramework/TQTreeObservable.h"

class TQTreeFormulaObservable : public TQTreeObservable {

protected:
  TString fExpression = "";
  TTreeFormula * fFormula = NULL;
  bool fcallGetNdata = false;
  bool fVectorObs = false;
  mutable TString fFullExpression; //helper variable for getExpression since we need to prepend the prefix

  virtual double getValueAt(int index) const override;
  virtual int getNevaluations() const override;
  inline virtual TQObservable::ObservableType
    getObservableType() const override {
        if (this->fVectorObs) {
          return TQObservable::ObservableType::vector;
        } else {
          return TQObservable::ObservableType::scalar;
        }
    }

  virtual bool initializeSelf() override;
  virtual bool finalizeSelf() override;

public:

  TQTreeFormulaObservable();
  TQTreeFormulaObservable(const TString& expression, bool vectorObs=false);

  virtual double getValue() const override;

  virtual bool hasExpression() const override;
  virtual const TString& getExpression() const override;
  virtual void setExpression(const TString& expr) override;
  virtual TString getActiveExpression() const override;

  virtual TObjArray* getBranchNames() const override;
 
  virtual ~TQTreeFormulaObservable();

  DECLARE_OBSERVABLE_FACTORY(TQTreeFormulaObservable, TString expr)
  
  ClassDefOverride(TQTreeFormulaObservable, 0); // observable to access data from a tree using a TTreeFormula

};

#endif
