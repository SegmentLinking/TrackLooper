//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQNTUPLEDUMPERANALYSISJOB__
#define __TQNTUPLEDUMPERANALYSISJOB__

#include "QFramework/TQObservable.h"
#include "QFramework/TQAnalysisJob.h"
#include "QFramework/TQMessageStream.h"
#include "QFramework/TQTaggable.h"
#include "QFramework/TQCut.h"
#include "QFramework/TQUtils.h"

#include "TList.h"
#include "TFile.h"
#include "TTree.h"


namespace TQNTupleDumperAnalysisJobHelpers {
  class FileHandle;
  class TreeHandle;
  class BranchHandle;
}

class TQNTupleDumperAnalysisJob : public TQAnalysisJob {
 public:
  enum VarType { UNKNOWN=0, INT=1, FLOAT=2, DOUBLE=3 , ULL=4, VECTORINT=5, VECTORFLOAT=6, VECTORDOUBLE=7, VECTORULL=8};
  
  static TQNTupleDumperAnalysisJob::VarType getVarType (TString typestr);
  static TString getTypeString (TQNTupleDumperAnalysisJob::VarType type);
  static int importJobsFromTextFiles(const std::vector<TString>& filenames, TQCut* basecut, TQTaggable* aliases = NULL, const TString& channelFilter = "*", bool verbose = false);
  static TString getErrorMessage();
  static void setErrorMessage(const TString& message);


  TQToken* getTreeToken(const TString& filename, const TString& treename);
  bool returnTreeToken(const TString& filename, const TString& treename, TQToken* tok);
  
 protected:
  
  static const std::map<VarType,bool> isVectorBranchMap; //!
  
  static TQMessageStream fErrMsg;
  
  static std::map<TString,TFile*> fFiles; //!
  static std::map<TString,int> nUsers; //!
  static TObjString fTimestamp; //!
  
  std::vector<TString> fVarNames;
  std::vector<TString> fExpressions;
  std::vector<VarType> fTypes;

  std::vector<TQObservable*> fObservables;

  TString fTreeName = "nTuple";
  TString fFileName = "data";
  TString fActiveTreeName = ""; //!
  TString fActiveFileName = ""; //!
  
  bool fWriteWeight = true;
  TString fWeightName = "weight";
  TQNTupleDumperAnalysisJobHelpers::BranchHandle* fWeightBranch = 0; //!

  TQNTupleDumperAnalysisJobHelpers::TreeHandle * fTreeHandler = NULL; //!
  TQToken* fTreeToken = NULL; //!
  std::vector<TQNTupleDumperAnalysisJobHelpers::BranchHandle*> fBranches; //!
  
  TQSampleFolder* poolAt = NULL;

  bool initializeTree(TQTaggable* tags);
  bool finalizeTree();

  virtual bool initializeSelf() override;
  virtual bool finalizeSelf() override;
  
  int getNentriesToCreate() const;
  mutable bool fExpectSingleEntryPerEvent = false;
  
 public:

  TQNTupleDumperAnalysisJob(const TString& name);
  TQNTupleDumperAnalysisJob(TQNTupleDumperAnalysisJob* other);

  bool bookVariable(const TString& type, const TString& name, const TString& definition);
  bool bookVariable(VarType type, const TString& name, const TString& definition);
  int nVariables();
  
  virtual TQAnalysisJob * getClone() override;
  virtual bool initializeSampleFolder(TQSampleFolder* sf) override;
  virtual bool finalizeSampleFolder(TQSampleFolder* sf) override;
  virtual bool execute(double weight) override;

  virtual TObjArray * getBranchNames() override;

  void writeWeights(bool write = true, const TString& name = "weight");
  
  TString getTreeName() const ;
  void setTreeName (const TString& treename);
  TString getFileName() const ;
  void setFileName (const TString& filename);

  void printBranches();


  ClassDefOverride(TQNTupleDumperAnalysisJob,1) // analysis job allowing to dump flat nTuples

};



#endif
