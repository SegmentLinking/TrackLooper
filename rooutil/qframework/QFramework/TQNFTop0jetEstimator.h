//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQNFTOP0JETESTIMATOR__
#define __TQNFTOP0JETESTIMATOR__

#include "QFramework/TQNamedTaggable.h"
#include "QFramework/TQNFBase.h"

class TQNFTop0jetEstimator : public TQNFBase {
protected:
  TString fPath;
  TString fPathData;
  TString fCutCR;
  TString fCutJetEffNumerator;
  TString fCutJetEffDenominator;

  double fResult;
  double fResultXSec;
  double fResultExtrapolation;
  double fResultAlphaMC;
  double fResultAlphaData;
  
  bool fSuccess = false;

  TQCounter* cnt_mc_cr = NULL;
  TQCounter* cnt_data_cr = NULL;
  TQCounter* cnt_mc_numerator = NULL;
  TQCounter* cnt_data_numerator = NULL;
  TQCounter* cnt_mc_denominator = NULL;
  TQCounter* cnt_data_denominator = NULL;
  
  std::vector<TString> vBkgPaths;
  std::vector<TQCounter*> vBkgCountersCR;
  std::vector<TQCounter*> vBkgCountersNumerator;
  std::vector<TQCounter*> vBkgCountersDenominator;
  
  int deployResultInternal(const std::vector<TString>& startCutNames, const std::vector<TString>& stopAtCut, int overwrite, int mode);
  
  TString getPostfix(int mode = 0);
  
  virtual bool initializeSelf() override; 
  virtual bool finalizeSelf() override; 

public:
  TQNFTop0jetEstimator();
  TQNFTop0jetEstimator(TQSampleDataReader* rd);
  TQNFTop0jetEstimator(TQSampleFolder* sf);
  virtual ~TQNFTop0jetEstimator();

  TString getPathMC();
  TString getPathData();
  void setPathMC(const TString& path);
  void setPathData(const TString& path);

  TString getControlRegion();
  TString getJetEffNumerator();
  TString getJetEffDenominator();
  void setControlRegion(const TString& region);
  void setJetEffNumerator(const TString& region);
  void setJetEffDenominator(const TString& region);
 
  bool readConfiguration(TQFolder* f ) override;

  bool calculate();
  void printResult(int mode = 0);
 
  double getResult(int mode = 0);

  int execute(int itrNumber = -1) override;
  bool success() override;

  int deployResult(const std::vector<TString>& startCutNames, const std::vector<TString>& stopAtCut, int overwrite) override;
  
  ClassDefOverride(TQNFTop0jetEstimator,1) // implements the HWW top0jet estimate
};

#endif
