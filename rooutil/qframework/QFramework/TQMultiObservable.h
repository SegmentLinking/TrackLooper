//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQMULTIOBSERVABLE__
#define __TQMULTIOBSERVABLE__
#include "QFramework/TQObservable.h"
#include "TFormula.h"
#include "TObjArray.h"

class TQMultiObservable : public TQObservable {
protected:
  virtual bool initializeSelf() override;
  virtual bool finalizeSelf() override;
  
public:
  virtual int getNevaluations() const override;
  virtual double getValueAt(int index) const override;
  virtual double getValue() const override;
  
  inline virtual TQObservable::ObservableType getObservableType() const override {return this->fObservableType;}

  virtual TObjArray* getBranchNames() const override;
  virtual Long64_t getCurrentEntry() const override;
protected:
  TString fExpression = "";
  TString fActiveExpression = "";
  TString fParsedExpression = "";
  std::vector<TQObservable*> fObservables;
  TQObservable::ObservableType fObservableType = TQObservable::ObservableType::unknown;
  TFormula* fFormula = NULL; //!
  mutable Long64_t fCachedEntry; //!
  mutable std::vector<double> fCachedValue; //!

public:
  virtual bool hasExpression() const override;
  virtual const TString& getExpression() const override;
  virtual void setExpression(const TString& expr) override;

  TQMultiObservable();
  TQMultiObservable(const TString& expression);
  virtual ~TQMultiObservable();
public:
  bool parseExpression(const TString& expr);
  void clearParsedExpression();
  TString getParsedExpression();
  TQObservable* getObservable(int idx);
  
  virtual TString getActiveExpression() const override;

  DECLARE_OBSERVABLE_FACTORY(TQMultiObservable,TString expression)
  
  ClassDefOverride(TQMultiObservable, 0); // meta-observable combining data from other observables

};
#endif
