//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQMVATREEOBSERVABLE__
#define __TQMVATREEOBSERVABLE__

#include "QFramework/TQObservable.h"
#include "TMVA/Reader.h"
#include "QFramework/TQNamedTaggable.h"

class TQMVAObservable : public TQObservable {
 
public:
  static TQTaggable globalAliases;
  
  class Reader { //nested
  protected:
    TString fFileName = "";
    TString fMethodName = "";
    TMVA::Reader* fMVAReader = NULL;
    TMVA::IMethod* fMVAMethod = NULL;
    TObjArray* fVariables = NULL;
    mutable std::vector<float> fValues;
    std::vector<TString> fExpressions;

    int parseVariables();
    void assignVariables();
    void clearVariables();
    void printVariables() const;
    void print() const;

  public:
    Reader(const char* filename, const char* methodname);

    size_t size() const;
    const TString& getExpression(size_t i) const;
    double getValue() const;
    void fillValue(size_t i,double val) const ;
      
    static bool getExpression(TQTaggable* var, TString& result);
  };

protected:
  Reader* fReader = NULL;
  
  TString fExpression = "";
  std::vector<TQObservable*> fObservables;
  mutable Long64_t fCachedEntry = -1; //!
  mutable double fCachedValue = -1; //!

  virtual bool initializeSelf() override;
  virtual bool finalizeSelf() override;

public:
  static TQMVAObservable::Reader* getReader(const TString& expression);

  virtual Long64_t getCurrentEntry() const override;

  virtual bool hasExpression() const override;
  virtual const TString& getExpression() const override;
  virtual void setExpression(const TString& expr) override;
  virtual TString getActiveExpression() const override;

  TQMVAObservable();
  TQMVAObservable(const TString& expression);

  virtual double getValue() const override;

  virtual TObjArray* getBranchNames() const override;
 
  virtual ~TQMVAObservable();

  DECLARE_OBSERVABLE_FACTORY(TQMVAObservable,TString expression)
  
  ClassDefOverride(TQMVAObservable, 0);

};

#endif
