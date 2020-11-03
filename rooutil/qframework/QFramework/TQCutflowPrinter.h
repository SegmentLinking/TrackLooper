//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQCutflowPrinter__
#define __TQCutflowPrinter__

#include "QFramework/TQPresenter.h"
#include "QFramework/TQTable.h"

class TQCutflowPrinter : public TQPresenter {

protected:
 
  bool getValues (TQTaggable& tags, TQNamedTaggable* cut, TQNamedTaggable* process, double& value, double& error, int& raw, TString& info, TString& defaultTitle);
  bool getScaleFactors (TQTaggable& tags, TQNamedTaggable* cut, TQNamedTaggable* process, double& value, double& error, bool& applied, bool& equal, TString& info);
  TString makeCellContents (TQTaggable& tags, TQNamedTaggable* cut, TQNamedTaggable* process);
  TString makeSFCellContents(TQTaggable& tags, TQNamedTaggable* cut, TQNamedTaggable* process, TString& info);
 
public:

  TQCutflowPrinter();
  TQCutflowPrinter(TQSampleFolder * samples);
  TQCutflowPrinter(TQSampleDataReader * reader);

  bool readProcessesFromFile(TString fileName, TString channel);
  bool readCutsFromFile (TString fileName, bool addscline=false);

  void addCutflowCut (TString cutName, TString cutTitle, int sfPolicy=0);
  void addCutflowProcess(TString processName, TString processTitle);

  int sanitizeProcesses();
  int sanitizeCuts();

  TQTable * createTable(const TString& tags = "");
  TQTable * createTable(TQTaggable* tags);
  TQTable * createTable(TQTaggable tags);
 
  void dumpProcesses(std::ostream& out);
  void dumpCuts(std::ostream& out);

  virtual ~TQCutflowPrinter();
 
  ClassDefOverride(TQCutflowPrinter, 0); // presenter class to print cutflows

};

#endif
