//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQNFTOP1JETESTIMATOR__
#define __TQNFTOP1JETESTIMATOR__

#include "QFramework/TQNamedTaggable.h"
#include "QFramework/TQNFBase.h"

class TQNFTop1jetEstimator : public TQNFBase {
protected:
  TString fPath;
  TString fPathData;
  TString fCut10;
  TString fCut11;
  TString fCut21;
  TString fCut22;
  double fResult;
  double fResultExtrapolation; //full extrapolation factor (1-epsilon)/epsilon
  double fResultEpsilon1Jet; //b-tag eff. measured in data and extrapolated to 1jet region using MC
  double fResultEpsilon2JetData; //b-tag eff. measured in data (2jets, 1 or 2 btags)
  double fResultGammaMC; //MC based extrapol. factor for b-tag eff. 2jet->1jet
  bool fSuccess = false;

  TQCounter* cnt_mc_10 = 0;
  TQCounter* cnt_mc_11 = 0;
  TQCounter* cnt_mc_21 = 0;
  TQCounter* cnt_mc_22 = 0;
  TQCounter* cnt_data_11 = 0;
  TQCounter* cnt_data_21 = 0;
  TQCounter* cnt_data_22 = 0;
  
  std::vector<TString> vBkgPaths;
  std::vector<TQCounter*> vBkgCounters11;
  std::vector<TQCounter*> vBkgCounters21;
  std::vector<TQCounter*> vBkgCounters22;
  
  int deployResultInternal(const std::vector<TString>& startCutNames, const std::vector<TString>& stopAtCut, int overwrite, int mode = 0);
  
  TString getPostfix(int mode=0);
  
  virtual bool initializeSelf() override; 
  virtual bool finalizeSelf() override; 

public:
  TQNFTop1jetEstimator();
  TQNFTop1jetEstimator(TQSampleDataReader* rd);
  TQNFTop1jetEstimator(TQSampleFolder* sf);
  virtual ~TQNFTop1jetEstimator();

  TString getPathMC();
  TString getPathData();
  void setPathMC(const TString& path);
  void setPathData(const TString& path);

  TString getRegion1j0b();
  TString getRegion1j1b();
  TString getRegion2j1b();
  TString getRegion2j2b();
  void setRegion1j0b(const TString& region);
  void setRegion1j1b(const TString& region);
  void setRegion2j1b(const TString& region);
  void setRegion2j2b(const TString& region);
  
  bool readConfiguration(TQFolder* f ) override;

  bool calculate();
  void printResult(int mode = 0);
 
  double getResult(int mode = 0);

  int execute(int itrNumber = -1) override;
  bool success() override;

  int deployResult(const std::vector<TString>& startCutNames, const std::vector<TString>& stopAtCut, int overwrite) override;

  ClassDefOverride(TQNFTop1jetEstimator,1) // implements the HWW top1jet estimate
};

#endif
