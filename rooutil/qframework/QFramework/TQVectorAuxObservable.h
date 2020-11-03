//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQVECTORAUXOBSERVABLE__
#define __TQVECTORAUXOBSERVABLE__
#include "QFramework/TQObservable.h"

class TQVectorAuxObservable : public TQObservable {
protected:
  // put here any data members your class might need
  TQObservable* fSubObservable = NULL; //!
  TQObservable* fIndexObservable = NULL; //! //only to be used for 'AT' mode indicating the index of fSubObservable to be returned
  mutable Long64_t fCachedEntry; //!
  mutable double fCachedValue; //!
  mutable TString fFullExpression; //helper variable for getExpression since we need to prepend the prefix
  
public:
  virtual double getValue() const override;
  virtual TObjArray* getBranchNames() const override;
  virtual Long64_t getCurrentEntry() const override;
  
  enum Operation {AND,OR,SUM,SUMABS,AVG,LEN,MAX,MIN,MAXABS,MINABS,NORM,PROD,NTRUE,NFALSE,AT,invalid};

protected:
  virtual bool initializeSelf() override;
  virtual bool finalizeSelf() override;
protected:
  TString fExpression = ""; //full, "raw" expression
  TString fVecExpression = ""; //in any case the first sub-expression
  TString fIndexExpression = ""; //only used for 'AT' mode
  TQVectorAuxObservable::Operation fOperation = TQVectorAuxObservable::Operation::invalid;
public:
  inline TQVectorAuxObservable::Operation getOperation() const {return this->fOperation;}
  inline void setOperation(TQVectorAuxObservable::Operation op) {this->fOperation = op; return;}
  static TString getOperationName(TQVectorAuxObservable::Operation op);
  static TString getPrefix(TQVectorAuxObservable::Operation op);
  static TQVectorAuxObservable::Operation readPrefix(TString& expression);
  
  virtual bool hasExpression() const override;
  virtual const TString& getExpression() const override;
  virtual void setExpression(const TString& expr) override;
  
  inline virtual bool isPrefixRequired() const override {return true;} ; //needs to be overridden if an observable (with a factory) should only be matched if it's prefix is present. In this case the observable must ensure to include its prefix in strings returned by getExpression, getActiveExpression,...
  
  TQVectorAuxObservable();
  TQVectorAuxObservable(const TString& expression);
  virtual ~TQVectorAuxObservable();
public:

  virtual TString getActiveExpression() const override;
  
  
public:
  DECLARE_OBSERVABLE_FACTORY(TQVectorAuxObservable,TString expr)

  ClassDefOverride(TQVectorAuxObservable, 0); //QFramework class

};
#endif
