//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQCutFactory__
#define __TQCutFactory__

#include "TObject.h"
#include "TString.h"
#include "QFramework/TQCompiledCut.h"
#include "TList.h"

class TQCutFactory : public TObject {

protected:

  TList * fCuts;

  TList * fTreeObservableTemplates;

public:

  static TString evaluate(const TString& input, const TString& parameter = "");
  static TString evaluateSubExpression(const TString& input, const TString& parameter = "");



  TQCutFactory();

  void setTreeObservableTemplates(TList * treeObservableTemplates);

  void addCut(TString definition);

  TString findCut(TString name);
  TString removeCut(TString name);

  void print();

  bool isEmpty();

  void orderCutDefs();
  TQCompiledCut * compileCutsWithoutEvaluation();
  TQCompiledCut * compileCuts(TString parameter = "");
 
  virtual ~TQCutFactory();
 
  ClassDefOverride(TQCutFactory, 1); // deprecated

};

#endif
