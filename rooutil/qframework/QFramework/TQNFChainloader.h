//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQ_NFCHAINLOADER__
#define __TQ_NFCHAINLOADER__

#include "QFramework/TQSampleFolder.h"
#include "QFramework/TQSampleDataReader.h"
#include "QFramework/TQMessageStream.h"
#include "QFramework/TQNFBase.h"
#include "TRandom3.h"
#include <iostream>
#include "TString.h"

#ifdef __CINT__ 
#define override
#endif
class TQNFChainloader : public TQNamedTaggable {
protected:
  TQSampleDataReader* fReader;
  TQTaggable fVariationTags;
  
  std::vector<TString> vNFconfig;
  std::vector<TQNFBase*> vNFcalc; //!
  std::vector<TQNFBase*> vNFpostCalc; //!
  std::map<TString,double> relativeVariationMap;
  std::map<TString,bool> targetNFpaths;
  std::map<TString,std::vector<double>> mNFvectors;
  TRandom3 rnd;
  int registerTargetPaths(const std::vector<TString>& vec);
  int collectNFs();
  int deployNFs();
  bool initialized = false;
 
public:
  enum Mode {
    Manual,
    NFCalculator,
    ABCD
  };
 
  int status;
  int verbosity;
 
  TQNFChainloader();
  TQNFChainloader(const std::vector<TString>& nfconfigs, TQSampleFolder* f);
  virtual ~TQNFChainloader();
 
  int addNFconfigs(const std::vector<TString>& nfconfigs);
  int setNFconfigs(const std::vector<TString>& nfconfigs);
  void setSampleFolder(TQSampleFolder* f);
  double getRelVariation(const TString& key, double value, double uncertainty);
 
  bool initialize();
  bool finalize();
 
  TList* getListOfFolders();

  int execute();
  
  int setVariationTags(const TQTaggable& tags);
  int setVariationTags(TQTaggable* tags);

  ClassDefOverride(TQNFChainloader,1) // high-level class to manage multi-stage normalization factor computations
};

#endif
