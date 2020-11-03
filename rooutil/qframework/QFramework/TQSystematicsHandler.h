//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQSystematicsHandler__
#define __TQSystematicsHandler__

#include "QFramework/TQFolder.h"
#include "QFramework/TQTable.h"
#include "QFramework/TQSampleDataReader.h"

class TQSystematicsHandler : public TQFolder {
protected:

  TString fTotalPath;
  TString fNominalFilePath;
  std::map<const TString, std::vector<std::pair<TString, TString>> > fSystematics;
  std::map<const TString, TQSampleFolder*> fInputs;

  void collectSystematic(const TString& systname);
  int createVariationHistograms(TQFolder* syst, TQFolder* nominal);
  void createVariationYield (TQFolder* syst, TQFolder* nominal);
  void exportObjects(TQFolder* cut,TQFolder* target);
  void collectHistograms(TQSampleDataReader* rd, const TString& subpath, const TString& sysfolderpath);
  void collectCounters(TQSampleDataReader* rd, const TString& subpath, const TString& sysfolderpath);
  bool addSystematic(const TString& name, const std::vector<std::pair<TString, TString>>& path_folder_pairs);

public:

  TQSystematicsHandler(const TString& name = "systematics");
  ~TQSystematicsHandler();

  TQFolder* addCut(const TString& id);

  bool hasSystematic(const TString& name) const;
  TQSampleFolder* getSampleFolder(const TString& systname,int i);

  void setNominal (const TString& path);
  void setNominalFilePath (const TString& path);
  void setTotalPath(const TString& path);
  void setNominalSampleFolderPath(const TString& path);
  bool addSystematic(const TString& name, const std::pair<TString, TString>& path);
  bool addSystematic(const TString& name, const std::pair<TString, TString>& path1, const std::pair<TString, TString>& path2);
  bool addSystematic(const TString& name, const TString& path);
  bool addSystematic(const TString& name, const TString& path1, const TString& path2);
  bool addSystematicFromSampleFolderPath(const TString& name, const TString& path);
  bool addSystematicFromSampleFolderPath(const TString& name, const TString& path1, const TString& path2);
  void printSystematics() const;

  void collectSystematics();

  TQFolder* exportSystematics();

  TList* getRanking(const TString& cutname);
  TQTable* getTable(const TString& cutname);
 
  ClassDefOverride(TQSystematicsHandler,0) // helper class to prepare systematic variations for plotting

};

#endif
