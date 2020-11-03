//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQFILTEROBSERVALBLE__
#define __TQFILTEROBSERVALBLE__
#include "QFramework/TQObservable.h"

class TQFilterObservable : public TQObservable {
protected:
  // put here any data members your class might need
 
public:
  virtual double getValue() const override;
  virtual TObjArray* getBranchNames() const override;

  virtual double getValueAt(int index) const override;
  virtual int getNevaluations() const override;
  inline virtual TQObservable::ObservableType
    getObservableType() const override {
        return TQObservable::ObservableType::vector;
    }

protected:
  virtual bool initializeSelf() override;
  virtual bool finalizeSelf() override;
protected:
  TString fExpression = "";
  TString fActiveExpression = "";
  TString fCutString = "";
  TString fValueString = "";
  
  mutable Long64_t fCachedEntry = -1;//!
  mutable std::vector<double> fCachedValues; //!
  bool makeCache() const;

  TQObservable* fCutObs = NULL;
  TQObservable* fValueObs = NULL;

public:
  virtual bool hasExpression() const override;
  virtual const TString& getExpression() const override;
  virtual void setExpression(const TString& expr) override;

  TQFilterObservable();
  TQFilterObservable(const TString& expression);
  virtual ~TQFilterObservable();
public:
  bool parseExpression(const TString& expr);
  void clearParsedExpression();

  virtual TString getActiveExpression() const override;
  virtual Long64_t getCurrentEntry() const override;
  static int registerFactory();

public:
  DECLARE_OBSERVABLE_FACTORY(TQFilterObservable,TString expr)

  ClassDefOverride(TQFilterObservable, 1); //QFramework class

};
#endif
