//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQABCDCALCULATOR__
#define __TQABCDCALCULATOR__

#include "QFramework/TQNamedTaggable.h"
#include "QFramework/TQNFBase.h"

class TQABCDCalculator : public TQNFBase {
protected:
  TString fPathMC;
  TString fPathData;
  TString fCutTarget;
  TString fCutSource;
  TString fCutNumerator;
  TString fCutDenominator;
  
  std::vector<TString> vBkgPaths;
  std::vector<TQCounter*> vBkgCountersB;
  std::vector<TQCounter*> vBkgCountersC;
  std::vector<TQCounter*> vBkgCountersD;

  double fResult;
  double fResultErr2;
  bool fSuccess = false;
  bool doNonClosureCorrection = false;

  TQCounter* cnt_mc_a = NULL;
  TQCounter* cnt_mc_b = NULL;
  TQCounter* cnt_mc_c = NULL;
  TQCounter* cnt_mc_d = NULL;
  TQCounter* cnt_data_b = NULL;
  TQCounter* cnt_data_c = NULL;
  TQCounter* cnt_data_d = NULL;

  virtual bool initializeSelf() override; 
  virtual bool finalizeSelf() override; 

public:
  TQABCDCalculator();
  TQABCDCalculator(TQSampleDataReader* rd);
  TQABCDCalculator(TQSampleFolder* sf);
  virtual ~TQABCDCalculator();

  TString getPathMC();
  void setPathMC(const TString& path);
  TString getPathData();
  void setPathData(const TString& path);

  TString getA();
  TString getB();
  TString getC();
  TString getD();
  void setA(const TString& region);
  void setB(const TString& region);
  void setC(const TString& region);
  void setD(const TString& region);
  TString getTarget();
  TString getSource();
  TString getNumerator();
  TString getDenominator();
  void setTarget(const TString& region);
  void setSource(const TString& region);
  void setNumerator(const TString& region);
  void setDenominator(const TString& region);
 
  bool readConfiguration(TQFolder* f ) override;

  bool calculate();
  void printResult();
 
  double getResult();
  double getResultUncertainty();
  double getResultVariance();

  int execute(int itrNumber = -1) override;
  bool success() override;

  int deployResult(const std::vector<TString>& startAtCutNames, const std::vector<TString>& stopAtCut, int overwrite) override;

  ClassDefOverride(TQABCDCalculator,1) // helper class to facilitate an ABCD-estimate
};

#endif
