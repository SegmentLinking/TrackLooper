//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQSampleInitializerBase__
#define __TQSampleInitializerBase__

#include "QFramework/TQSample.h"

class TQSampleInitializerBase : public TQTaggable {

protected: 
 
  TQFolder* fPaths = NULL;
  virtual bool getTreeInformation(TQSample* sample, const TString& filename, TString& treeName, double& nEvents, int& nEntries, TString& message);
  virtual bool initializeSample(TQSample* sample, const TString& fullpath, TString& message);
  virtual bool setSampleNormalization(TQSample* sample,double samplefraction = 1.);

  TQSampleInitializerBase();
  
public:

  static TQFolder* extractCounters(TFile* file,double scale=1);
  static bool extractCountersFromSample(TQSample* sf);
  static bool extractCounters(TFile* file, TQFolder* cutflow, double scale=1);
  
  virtual void reset();
  
  void readDirectory(const TString& path, int maxdepth=999);
  void printDirectory(const TString& opts = "");
  TQFolder* getDirectory();
  bool readInputFilesList(const TString& path = "input.txt", bool verbose = false);

  virtual ~TQSampleInitializerBase();
 
  ClassDefOverride(TQSampleInitializerBase, 0); // base class for sample initializers

};

#endif
