//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQSampleDataReader__
#define __TQSampleDataReader__

class TH1;
class THnBase;
class TGraph;
class TGraph2D;
class TTree;
class TProfile;

#include "QFramework/TQSampleFolder.h"
#include "QFramework/TQTaggable.h"
#include "QFramework/TQCounter.h"
#include "QFramework/TQPCA.h"
#include "QFramework/TQTable.h"
#include "QFramework/TQMessageStream.h"
#include <sstream>

class TQSampleDataReader {
 
protected:
 
  TQSampleFolder * f_baseSampleFolder;
  bool f_localMode;
  TQMessageStream f_errMsg;

  TString f_styleScheme;
  TString f_filterScheme;
  TString f_normScheme;
  TString f_scaleScheme;
 
  TString f_pathHistograms;
  TString f_pathGraphs;
  TString f_pathGridScan;
  TString f_pathCutflow;
  TString f_pathEventlists;
  TString f_pathScaleFactors;
  TString f_pathPCA;
  TString f_pathTrees;

  int f_Verbose;
  int f_maxCorrWarnings = 25;
  
  std::map<TString,TQFolder*> f_identifierToFolderMap;

  void setErrorMessage(const TString& fname, const TString& message);

  template<class T> TList * collectElements(TList* paths, TList* elements, const TString& subPath, /*TQCounter * scale,*/ TList * sfList, TQTaggable * options);
  template<class T> int collectElementsWorker(TQSampleFolder * sampleFolder, const TString& elementName, const TString& subPath, TList* list, TList * baseScaleList, TList * sfList, TQTaggable * options, int indent);

  template<class T> int sumElements( TList * list, T * &histo, TQTaggable * options );
  int sumElements( TList * list, TQCounter* &counter, TQTaggable * options );
  int sumElements( TList * list, TQTable * &table, TQTaggable * options );
  int sumElements( TList * list, TGraph * &graph, TQTaggable * options );
  int sumElements( TList * list, TGraph2D * &graph, TQTaggable * options );
  int sumElements( TList * list, TTree * &tree, TQTaggable * options );
  int sumElements( TList * list, TQPCA * &tree, TQTaggable * options );
  int sumElements( TList * list, THnBase * &histo, TQTaggable * options );
  
  void addObjectNames(TQSampleFolder * sampleFolder, const TString& objectPath, TClass * objectClass,
                      TList * &objectNames, TList * sfList, const TString& filterScheme);
 
  void addObjectNames(const TString& path, const TString& objectPath, TClass * objectClass,
                      TList * &objectNames, TList * sfList, const TString& filterScheme);

  TList * getBaseScaleFactors(TQSampleFolder * sampleFolder, const TString& path, const TString& scaleScheme);
  TList * getListOfSampleFoldersTrivial(TString path, TClass* tclass=NULL);

  template<class T>
  T * getElement(const TString& path, const TString& name, const TString& subPath, TQTaggable * options, TList * sfList = NULL);
 
  template<class T>
  int getElementWorker(TList* paths, TList* elements, T*&element, const TString& subPath, TList* sfList, TQTaggable* options);

 
public:

  void setDefaultScaleScheme(const TString& defaultScheme);
  
  void applyStyleToElement(TObject* element, TCollection* sfList, TQTaggable* options = NULL);

  TList * parsePaths(TString paths, TList * inputTokens = 0, TString pathPrefix = "");

  TList * getListOfSampleFolders(const TString& path, TClass* tclass = TQSampleFolder::Class());

  TObject * getElement(const TString& path, const TString& name, TClass* objClass, const TString& subPath, TQTaggable * options, TList * sfList = NULL);

  TQSampleDataReader();
  TQSampleDataReader(TQSampleFolder * sampleFolder);

  void printPaths(TString paths);

  void reset();

  void setLocalMode(bool localMode = false);
  bool getLocalMode();

  void setVerbose(int verbose);
  int getVerbose();

  TString getErrorMessage();

  TQSampleFolder* getSampleFolder();

  bool compareHistograms(const TString& histName1, const TString& histName2, const TString path = "*", double maxdiff = 0.01, bool print = true);
  
  bool areFoldersCorrelated(TQFolder* f1, TQFolder* f2, double& correlation, bool reversed = false);
  bool areFoldersCorrelated(TQFolder* f1, TQFolder* f2);
  
  TString getStoragePath(TQFolder* f);
  
  TQFolder* getCorrelationFolderByIdentifier(const TString& id, bool forceUpdate=false);

  virtual bool passesFilter(TQSampleFolder * sampleFolder, TString filterName);

  virtual TH1 * getRatesHistogram(const TString& path, TString name, TQTaggable * options, TList * sfList = 0);
  virtual TH1 * getRatesHistogram(const TString& path, const TString& name, const TString& options = "", TList * sfList = 0);
  virtual TProfile * getProfile(const TString& path, TString name, TQTaggable * options, TList * sfList = 0);
  virtual TProfile * getProfile(const TString& path, const TString& name, const TString& options = "", TList * sfList = 0);
  virtual TH1 * getHistogram(const TString& path, TString name, TQTaggable * options, TList * sfList = 0);
  virtual TH1 * getHistogram(const TString& path, const TString& name, const TString& options = "", TList * sfList = 0);
  virtual TGraph * getGraph(const TString& path, TString name, TQTaggable * options, TList * sfList = 0);
  virtual TGraph * getGraph(const TString& path, const TString& name, const TString& options = "", TList * sfList = 0);
  virtual TGraph2D * getGraph2D(const TString& path, TString name, TQTaggable * options, TList * sfList = 0);
  virtual TGraph2D * getGraph2D(const TString& path, const TString& name, const TString& options = "", TList * sfList = 0);
  virtual THnBase * getTHnBase(const TString& path, TString name, TQTaggable * options, TList * sfList = 0);
  virtual THnBase * getTHnBase(const TString& path, const TString& name, const TString& options = "", TList * sfList = 0);
  virtual TQCounter * getCounter (const TString& path, const TString& name, const TString& options = "", TList * sfList = 0);
  virtual TQCounter * getCounter (const TString& path, const TString& name, TQTaggable* options, TList * sfList = 0);
  virtual TQPCA*getPCA (const TString& path, const TString& name, const TString& options = "", TList * sfList = 0);
  virtual TQPCA*getPCA (const TString& path, const TString& name, TQTaggable* options, TList * sfList = 0);
  virtual TQTable * getEventlist(const TString& path, const TString& name, TQTaggable * options, TList * sfList = 0);
  virtual TQTable * getEventlist(const TString& path, const TString& name, const TString& options = "", TList * sfList = 0);

  virtual TTree * getTree(const TString& path, const TString& name, TQTaggable * options, TList * sfList = 0);
  virtual TTree * getTree(const TString& path, const TString& name, const TString& options = "", TList * sfList = 0);


  virtual bool hasHistogram(const TString& path, const TString& name, const TString& options = "");
  virtual bool hasCounter (const TString& path, const TString& name, const TString& options = "");

  virtual TList * getListOfHistogramNames (const TString& path = ".", TList * sfList = 0);
  virtual TList * getListOfCounterNames (const TString& path = ".", TList * sfList = 0);
  virtual TList * getListOfEventlistNames (const TString& path = ".", TList * sfList = 0);
  virtual TList * getListOfObjectNames (TClass* objClass, const TString& subpath, const TString& path = ".", TList * sfList=0);

  virtual void printListOfHistograms (TString options = "");
  virtual void printListOfHistogramLocations(const TString& name);
  virtual void printListOfCounters (TString options = "");
  virtual void printListOfCounterLocations(const TString& name);

  virtual bool exportHistograms(TDirectory* d, const TString& sfpath, const TString& tags = "");
  virtual bool exportHistograms(TDirectory* d, const TString& sfpath, TQTaggable& tags);
  virtual TFolder* exportHistograms(const TString& sfpath, const TString& tags = "");
  virtual TFolder* exportHistograms(const TString& sfpath, TQTaggable& tags);
  virtual bool exportHistogramsToFile(const TString& fname, const TString& sfpath, const TString& tags = "");
  virtual bool exportHistogramsToFile(const TString& fname, const TString& sfpath, TQTaggable& tags);

  virtual void copyData(const TString& source, const TString&target, const TString&options = "");

  virtual ~TQSampleDataReader();
 
  ClassDef(TQSampleDataReader, 1); // helper class for data retrieval from sample folders

};

#endif
