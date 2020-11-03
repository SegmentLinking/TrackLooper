//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQMultiChannelAnalysisSampleVisitor__
#define __TQMultiChannelAnalysisSampleVisitor__

#include <map>

#include "TTree.h"

#include "QFramework/TQCut.h"
#include "QFramework/TQAnalysisSampleVisitorBase.h"
#include "QFramework/TQAlgorithm.h"

class TQMultiChannelAnalysisSampleVisitor : public TQAnalysisSampleVisitorBase, public TQAlgorithm::Manager {
 
protected:

  bool fUseObservableSets = true;
  std::map<TString,TQCut*> fChannels;

  bool checkCut(TQCut * baseCut);
  bool checkChannel(const TString& pathPattern);
  
  int analyseTree(TQSample * sample, TString& message);

  virtual int visitFolder(TQSampleFolder * sampleFolder, TString& message) override;
  virtual int revisitFolder(TQSampleFolder * sampleFolder, TString& message) override;
  virtual int visitSample(TQSample * sample, TString& message) override;
  virtual int revisitSample(TQSample * sample, TString& message) override;
  
  bool stampAllFriends(TQSample* sample) const;
  void updateFriends(TQSampleFolder* sf);
  
public: 
 
  TQMultiChannelAnalysisSampleVisitor();

  void printChannels();
  void useObservableSets(bool useSets);
  void addChannel(const TString& pathPattern, TQCut * baseCut);
  TQCut * getBaseCut(const TString& path);

  virtual ~TQMultiChannelAnalysisSampleVisitor();
 
  ClassDefOverride(TQMultiChannelAnalysisSampleVisitor, 0); // optimized analysis visitor for parallel reading of multiple channels
 
};

#endif


