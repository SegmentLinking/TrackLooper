//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQHISTOGRAMOBSERVABLE__
#define __TQHISTOGRAMOBSERVABLE__
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TFile.h"

#include "QFramework/TQObservable.h"

template <class T> class TQHistogramObservable : public TQObservable {
protected:
  virtual bool initializeSelf() override;
  virtual bool finalizeSelf() override;
  bool makeCache() const;
  mutable std::vector<double> fCachedValues;
  double findValue(int index) const;
  
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
  TString fFileName = "";
  TFile* fFile = NULL;
  std::vector<TQObservable*> fObservables;
  T* fHistogram = NULL;
  mutable Long64_t fCachedEntry; //!
  mutable double fCachedValue; //!
  TQObservable::ObservableType fObservableType = TQObservable::ObservableType::unknown;  

public:
  virtual bool hasExpression() const override;
  virtual const TString& getExpression() const override;
  virtual void setExpression(const TString& expr) override;

public:
  bool parseExpression(const TString& expr);
  void clearParsedExpression();
  //TString getParsedExpression();
  TQObservable* getObservable(int idx);

  virtual TString getActiveExpression() const override;
public:

  TQHistogramObservable();
  TQHistogramObservable(const TString& expression);
  virtual ~TQHistogramObservable();

  DECLARE_OBSERVABLE_FACTORY(TQHistogramObservable<T>,TString expression);
  
  ClassDefT(TQHistogramObservable<T>,0) // observable to facilitate lookup of values in histograms
};

ClassDefT2(TQHistogramObservable,T)

typedef TQHistogramObservable<TH1> TQTH1Observable;
typedef TQHistogramObservable<TH2> TQTH2Observable;
typedef TQHistogramObservable<TH3> TQTH3Observable;
#endif
