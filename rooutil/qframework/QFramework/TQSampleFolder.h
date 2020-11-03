//this file looks like plain C, but it's actually -*- c++ -*-
// this file looks like it's c, but it's actually -*- C++ -*-
#ifndef __TQSampleFolder__
#define __TQSampleFolder__

#include "QFramework/TQCounter.h"
#include "QFramework/TQTaggable.h"
#include "QFramework/TQFolder.h"
#include "TList.h"
#include "TH1.h"

#include "QFramework/TQFlags.h"

class TQSampleVisitor;
class TQSample;

class TQSampleFolder : public TQFolder {

protected:

  void init();

  TQFolder * getTemplate();
 
  bool mergeAsSampleFolder(TQSampleFolder* f, const TString& traceID, MergeMode mode, bool verbose);

  //TList* fFriends = NULL; //!
  std::shared_ptr<std::set<TQSampleFolder*>> fFriends = nullptr; //!
  
  virtual void findFriendsInternal(TQSampleFolder* otherSF, bool forceUpdateSubFolders = false);
  int generalizeObjectsPrivate    (TList * names, TClass* objClass, TQTaggable* options, const TString& paths, const TString& subpath);
  int generalizeObjectsPrivate    (TList * names, TClass* objClass, TQTaggable* options, const TString& subpath);
  bool fIsFindingFriends = false; //! temporary flag to prevent infinite recursions in friend finding
  
public:
  static const TString restrictSelectionTagName;
  
  void convertLegacyNFs();
  void purgeWithoutTag(const TString& tag);
 
  static TQSampleFolder * newSampleFolder(TString name);
  static TQSampleFolder * loadSampleFolder(TString path, bool lazy=false);
  static TQSampleFolder * loadLazySampleFolder(const TString& path);

  static TQSampleFolder * newSampleList(const TString& name, const TString& treeLocation, double normalization = 1.);


  TQSampleFolder();
  TQSampleFolder(TString name_);

  virtual TQFolder * newInstance(const TString& name) override;

  TList * getListOfSampleFolders(const TString& path_, TQSampleFolder * template_, bool toplevelOnly = false);
  TList * getListOfSampleFolders(const TString& path_ = "?", TClass* tclass = TQSampleFolder::Class(), bool toplevelOnly = false);
  TList * getListOfSamples(const TString& path = "*");
  int printListOfSamples(const TString& path = "*");
  
  std::vector<TString> getSampleFolderPaths(const TString& path_ = "?", TClass * tClass = NULL, bool toplevelOnly = false);
  std::vector<TString> getSampleFolderPathsWildcarded(const TString& path_ = "?", TClass * tClass = NULL, bool toplevelOnly = false);
  std::vector<TString> getSamplePaths(const TString& path_ = "?", TClass * tClass = NULL, bool toplevelOnly = false);
  std::vector<TString> getSamplePathsWildcarded(const TString& path_ = "?", TClass * tClass = NULL, bool toplevelOnly = false);

  TQSampleFolder * getSampleFolder(TString path_, TQSampleFolder * template_, int * nSampleFolders_ = 0);
  TQSampleFolder * getSampleFolder(TString path_, TClass * tclass = 0, int * nSampleFolders_ = 0);
  TQSampleFolder * addSampleFolder(TQSampleFolder * sampleFolder_, TString path_, TQSampleFolder * template_);
  TQSampleFolder * addSampleFolder(TQSampleFolder * sampleFolder_, TString path_ = "", TClass* tclass = 0);

  TQSample * getSample(const TString& path);

  TQSampleFolder * getBaseSampleFolder();
  TQSampleFolder * getRootSampleFolder();

  int getNSampleFolders(bool recursive = false);
  int getNSamples(bool recursive = false);

  TH1 * getHistogram(TString path, TString name, TQTaggable * options, TList * sfList = 0);
  TH1 * getHistogram(TString path, TString name, TString options = "", TList * sfList = 0);
  TQCounter * getCounter (TString path, TString name, TString options = "", TList * sfList = 0);

  bool hasHistogram(TString path, TString name, TString options = "");
  bool hasCounter (TString path, TString name, TString options = "");

  bool renameLocalObject(TString category, TClass * classType, TString oldName, TString newName);
  bool renameLocalHistogram(TString oldName, TString newName);
  bool renameLocalCounter(TString oldName, TString newName);

  int renameHistogram(TString oldName, TString newName);
  int renameCounter(TString oldName, TString newName);

  bool deleteLocalObject(TString category, TString name);
  bool deleteLocalHistogram(TString name);
  bool deleteLocalCounter(TString name);

  int deleteHistogram(TString name);
  int deleteHistograms(TString filter);
  int deleteSingleCounter(TString name);
  int deleteCounter(TString filter);

  int copyHistogram(TString source, TString destination, TString options = "");
  int copyHistograms(TString sourceFilter, TString appendix, TString options = "");

  int copyHistogramToCounter(TString source, TString destination = "");

  TList * getListOfHistogramNames (const TString& path = ".", TList * sfList = 0);
  TList * getListOfCounterNames (const TString& path = ".", TList * sfList = 0);

  void printListOfHistograms(const TString& options = "");
  void printListOfCounters (const TString& options = "");

  bool validateAllCounter(TString path1, TString path2, TString options = "");
  bool validateAllCounter(TQSampleFolder * sampleFolder, TString options = "");
  bool validateCounter(TString path1, TString path2, TString counterName, TString options = "");
  bool validateCounter(TQSampleFolder * sampleFolder, TString counterName, TString options = "", int indent = 0);

  int setScaleFactor(const TString& name, double scaleFactor, double uncertainty = 0);
  int setScaleFactor(const char* name, double scaleFactor, double uncertainty = 0);
  int setScaleFactor(const TString& name, double scaleFactor, const TString& sampleFolders);
  int setScaleFactor(const TString& name, double scaleFactor, const char* sampleFolders);
  int setScaleFactor(const TString& name, double scaleFactor, double uncertainty, const TString& sampleFolders);
  int setScaleFactor(TString name, const TString& title, double scaleFactor, double uncertainty = 0);
 
  bool getScaleFactor(const TString& path, double& scale, double& uncertainty, bool recursive = false);
  double getScaleFactor(const TString& name, bool recursive = false);
  TQCounter* getScaleFactorCounter(const TString& name);
  TQCounter* getScaleFactorCounterRecursive(const TString& name);
  TQCounter* getScaleFactorCounterInternal(TString name);
  void printScaleFactors(TString filter = "");

  int visitMe(TQSampleVisitor * visitor, bool requireSelectionTag=false);
  int visitSampleFolders(TQSampleVisitor * visitor, const TString& category = "?");

  int generalizeHistograms (TList * names, const TString& paths, const TString& options);
  int generalizeCounters   (TList * names, const TString& paths, const TString& options);
  int generalizeHistograms (const TString& paths = "", const TString& options= "");
  int generalizeCounters   (const TString& paths = "", const TString& options= "");

  int generalizeObjects(const TString& prefix, const TString& options = "");
  int generalizeObjects(TList * names, TClass* objClass, const TString& options, const TString& paths, const TString& subpath);

  void findFriends(const TString& pathpattern, bool forceUpdate = false);
  void findFriends(TQSampleFolder* otherSF, bool forceUpdate = false);
  void findFriends(bool forceUpdate = false);

  void befriend(TQSampleFolder* other);
  void clearFriends();
  void printFriends();
  int countFriends();
  bool hasFriends();
  std::shared_ptr<std::set<TQSampleFolder*>> getFriends();
  bool isFriend(TQSampleFolder* other);
  
  bool merge(TQSampleFolder * f, bool sumElements = false, bool verbose = false);
  bool merge(TQSampleFolder * f, const TString& traceID, bool sumElements = false, bool verbose = false);

  virtual ~TQSampleFolder();
 
  ClassDefOverride(TQSampleFolder, 3); // derived container class for data and monte carlo samples

};

#endif
