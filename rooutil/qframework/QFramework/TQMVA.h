//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQMVA__
#define __TQMVA__

#include "QFramework/TQNamedTaggable.h"
#include "TObject.h"

#include "QFramework/TQSampleFolder.h"
#include "QFramework/TQSample.h"
#include "QFramework/TQToken.h"
#include "QFramework/TQCut.h"

#include "RVersion.h"
#if ROOT_VERSION_CODE < ROOT_VERSION(6,07,00)
#define LEGACY_INTERFACE_PRE_607
#endif

namespace TMVA { //EXCLUDE
  class Factory;
#ifndef LEGACY_INTERFACE_PRE_607
  class DataLoader;
#endif
}

class TQSample;

class TQMVA : public TQNamedTaggable {

protected:
  std::vector<TString> fSigPaths;
  std::vector<TString> fBkgPaths;

  TMVA::Factory* fMVA = NULL;
#ifndef LEGACY_INTERFACE_PRE_607
  TMVA::DataLoader* fDataLoader = NULL;
#endif
  TQSampleFolder* fSampleFolder = NULL;
  TQCut* fBaseCut = NULL;
  TQCut* fActiveCut = NULL;
  TFile* fOutputFile = NULL;
  int fVerbose = 1;
  TQTaggable* fAliases = NULL;

  std::vector<TString> fNames;
  std::vector<TString> fExpressions;
  void init();
  void addVariable(const TString& name, const TString& title, const TString& expression, const TString& unit, char vtype, double min, double max, bool spectator);

public:

  enum SampleType { // nested
    Signal, Background
  };

  class EventSelector { // nested
    friend class TQMVA;
    virtual bool selectEvent(int evtno) const = 0;
  };
  class AllTrainEventSelector : public EventSelector { // nested
    virtual bool selectEvent(int /*evtno*/) const { return true; };
  };
  class AllTestEventSelector : public EventSelector { // nested
    virtual bool selectEvent(int /*evtno*/) const { return false; };
  };
  class EvenOddEventSelector : public EventSelector { // nested
    virtual bool selectEvent(int evtno) const { return ((evtno % 2) == 0); };
  };
  class OddEvenEventSelector : public EventSelector { // nested
    virtual bool selectEvent(int evtno) const { return ((evtno % 2) == 1); };
  };
 
  TQMVA();
  TQMVA(const TString& name_);
  TQMVA(TQSampleFolder* sf);
  TQMVA(const TString& name_, TQSampleFolder* sf);
 
  void addSignal(const TString& path);
  void addBackground(const TString& path);
  void clearSignal();
  void clearBackground();

  void setVerbose(int verbose);

  void bookVariable(const TString& name, const TString& expression, double min = 0, double max = 0);
  void bookVariable(const TString& name, const TString& expression, const TString& title, double min = 0, double max = 0);
  void bookVariable(const TString& name, const TString& expression, const TString& title, const TString& unit, double min = 0, double max = 0);

  void bookVariable(const char* name, const char* expression, double min = 0, double max = 0);
  void bookVariable(const char* name, const char* expression, const char* title, double min = 0, double max = 0);
  void bookVariable(const char* name, const char* expression, const char* title, const char* unit, double min = 0, double max = 0);

  void bookSpectator(const TString& name, const TString& expression, double min = 0, double max = 0);
  void bookSpectator(const TString& name, const TString& expression, const TString& title, double min = 0, double max = 0);
  void bookSpectator(const TString& name, const TString& expression, const TString& title, const TString& unit, double min = 0, double max = 0);

  void bookSpectator(const char* name, const char* expression, double min = 0, double max = 0);
  void bookSpectator(const char* name, const char* expression, const char* title, double min = 0, double max = 0);
  void bookSpectator(const char* name, const char* expression, const char* title, const char* unit, double min = 0, double max = 0);


  void printInternalVariables() const; 
  void printListOfVariables() const;

  TString getVariableExpression(const TString& var);
  TString getVariableExpression(const char* var);

  virtual TQCut* useCut(const TString& name);
  virtual void setBaseCut(TQCut* cut);
  virtual TQCut* getBaseCut();

  TList* getListOfSamples(TQMVA::SampleType type);
  void printListOfSamples(TQMVA::SampleType type);

  virtual ~TQMVA();
 
  TMVA::Factory* getFactory();
  void setFactory(TMVA::Factory* mva);
  TQSampleFolder* getSampleFolder();
  void setSampleFolder(TQSampleFolder* sf);
  bool createFactory(const TString& outFileName, const TString& options = "");
  void deleteFactory();
  void closeOutputFile();

  TQTaggable* getAliases();
  void setAliases(TQTaggable* aliases);
 
  int readSamplesOfType(TQMVA::SampleType type, const TQMVA::EventSelector& sel);
  int readSamplesOfType(TQMVA::SampleType type);
  int readSamples(const TQMVA::EventSelector& evtsel);
  int readSamples();
  int readSample(TQSample* s, TQMVA::SampleType type, const TQMVA::EventSelector& sel);
 
  void prepareTrees(const TString& options = "");

  ClassDefOverride(TQMVA, 1); //QFramework class
 
};

#endif
