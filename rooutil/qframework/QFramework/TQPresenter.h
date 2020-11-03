//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQPresenter__
#define __TQPresenter__

#include "QFramework/TQNamedTaggable.h"
#include "QFramework/TQSampleDataReader.h"

class TQPresenter : public TQTaggable {

protected:

  TQSampleDataReader * fReader;
  bool fOwnerOfReader;

  TQFolder* fNormalizationInfo;

  TObjArray * fCuts; 
  TObjArray * fProcesses;

  TQFolder* fSystematics; 

  virtual TQNamedTaggable* getProcess(const TString& name);
  virtual TQNamedTaggable* getCut(const TString& name);

public:

  virtual bool hasProcess(const TString& name);
  virtual bool hasCut(const TString& name);

  TQPresenter();
  TQPresenter(TQSampleDataReader* reader);
  TQPresenter(TQSampleFolder* samples);
  virtual ~TQPresenter();

  virtual void reset();

  virtual void resetProcesses();
  virtual void clearProcesses();
  virtual void printProcesses();

  virtual int nProcesses();
  virtual int nCuts();

  virtual void resetCuts();
  virtual void clearCuts();
  virtual void printCuts();

  virtual bool loadSystematics(const TString& path, const TString& id);
  virtual bool loadSystematics(const TString& importPath);
  TQFolder* getSystematics(const TString& id);

  TQFolder* setNormalizationInfo(const TString& path);
  void setNormalizationInfo(TQFolder* f);
  TQFolder* getNormalizationInfo();

  virtual int sanitizeProcesses();
  virtual int sanitizeCuts();

  virtual TQSampleDataReader * getReader();
  virtual void setReader(TQSampleDataReader * reader);

  virtual void setSampleFolder(TQSampleFolder* sf);
  virtual TQSampleFolder* getSampleFolder();

  virtual void addCut (const TString& cutName, const TString& cutTitle, const TString& tags);
  virtual void addProcess(const TString& processName, const TString& processTitle, const TString& tags);
  virtual void addCut (const TString& cutName, const TString& cutTitle, TQTaggable& tags);
  virtual void addProcess(const TString& processName, const TString& processTitle, TQTaggable& tags);
  virtual void addCut (const TString& cutName, const TString& tags);
  virtual void addProcess(const TString& processName, const TString& tags);
  virtual void addCut (const TString& cutName, TQTaggable& tags);
  virtual void addProcess(const TString& processName, TQTaggable& tags);
  virtual void addProcess(const TString& processName, TQTaggable* tags);
  virtual void addCut (const TString& tags);
  virtual void addProcess(const TString& tags);
  virtual void addCut (TQTaggable& tags);
  virtual void addProcess(TQTaggable& tags);

  virtual TString getCutTitle(const TString& cutName);
  virtual TString getProcessTitle(const TString& processName);
  virtual TString getCutTags(const TString& cutName);
  virtual TString getProcessTags(const TString& processName);

  TCollection* getListOfProcessNames();
  TCollection* getListOfCutNames();
  TString getProcessPath(const TString& processName);
  
  virtual void exportScheme(TQTaggable* tags);
  virtual void importScheme(TQTaggable* tags);
  virtual void importSchemeFromPath(const TString& path);

  virtual int importProcessesFromFile(const TString& fileName,const TString& tags = "");
  virtual int importCutsFromFile(const TString& fileName, const TString& tags = "");
  virtual int importProcessesFromFile(const TString& fileName,TQTaggable& tags);
  virtual int importCutsFromFile(const TString& fileName, TQTaggable& tags);
  virtual int exportProcessesToFile(const TString& fileName,const TString& tags = "");
  virtual int exportCutsToFile(const TString& fileName, const TString& tags = "");
  virtual int exportProcessesToFile(const TString& fileName,TQTaggable& tags);
  virtual int exportCutsToFile(const TString& fileName, TQTaggable& tags);

  void removeProcessesByName(const TString& nameFilter);
  void removeCutsByName(const TString& nameFilter);

  ClassDefOverride(TQPresenter,0) // abstract base class for data visualizations

};

#endif
