//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQNFCustomCalculator__
#define __TQNFCustomCalculator__

#include "QFramework/TQNamedTaggable.h"
#include "QFramework/TQNFBase.h"
#include "TFormula.h"

class TQNFCustomCalculator : public TQNFBase {
protected:
  TString fPath;
  TString fPathData;

  double fResult;
  bool fSuccess = false;

  std::vector<TString> vBkgPaths;
  std::vector<TString> fPaths;
  std::vector<TString> fCuts;
  std::vector<TString> fTypes;
  std::vector<bool> fSubtractBkg;
  std::vector<double> fValues;
  TFormula* fFormula = NULL;
  TString fExpression;
  
  double getValue(const TString& path, const TString& cut, TQTaggable& tags, bool subtractBkg = false);
  
  virtual bool initializeSelf() override; 
  virtual bool finalizeSelf() override; 

public:
  TQNFCustomCalculator();
  TQNFCustomCalculator(TQSampleDataReader* rd);
  TQNFCustomCalculator(TQSampleFolder* sf);
  virtual ~TQNFCustomCalculator();

  TString getPathMC();
  TString getPathData();
  void setPathMC(const TString& path);
  void setPathData(const TString& path);

  bool readConfiguration(TQFolder* f ) override;

  bool calculate();
  void printResult();
 
  double getResult();

  int execute(int itrNumber = -1) override;
  bool success() override;

  int deployResult(const std::vector<TString>& startCutNames, const std::vector<TString>& stopAtCut, int overwrite) override;

  ClassDefOverride(TQNFCustomCalculator,1) // calculator for normalization factors allowing for custom formulae
};

#endif
