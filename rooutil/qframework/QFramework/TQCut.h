//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQCut__
#define __TQCut__

#include "TNamed.h"
#include "TObjArray.h"
#include "TTree.h"

#include "QFramework/TQIterator.h"

class TQAnalysisJob;
class TQSample;
class TQFolder;
class TQObservable;
class TQAnalysisJob;
class TQToken;
class TQAnalysisSampleVisitorBase;

class TQCut : public TNamed {

protected:

  TString fCutExpression = "";
  TQObservable * fCutObservable = NULL; //!
 
  TString fWeightExpression = "";
  TQObservable * fWeightObservable = NULL; //!

  TQCut * fBase = NULL;
  TObjArray * fCuts = new TObjArray();
  TObjArray * fAnalysisJobs = new TObjArray(); //!

  mutable TQCutIterator fCutItr; //!
  mutable TQAnalysisJobIterator fJobItr; //!
  
  TQToken * fTreeToken = NULL; //!
  TQSample * fSample = NULL; //!
  std::vector<TQSampleFolder*> fInitializationHistory; //!
  TTree * fTree = NULL; //!
 
  bool fSkipAnalysisJobs = false; //set based on tags on sample folders
  bool fSkipAnalysisJobsGlobal = false; //set based on tags during cut creation (from TQFolder)

  void setBase(TQCut * base_);

  bool skipAnalysisJobs(TQSampleFolder * sf);

  virtual bool initializeObservables();
  virtual bool finalizeObservables();
  virtual bool initializeSelfSampleFolder(TQSampleFolder* sf);
  virtual bool finalizeSelfSampleFolder  (TQSampleFolder* sf);

  virtual TObjArray* getOwnBranches();

  bool printEvaluationStep(size_t indent) const;

  void printInternal(const TString& options, int indent);

  static TQCut* createFromFolderInternal(TQFolder* folder, TQTaggable* tags);
  void importFromFolderInternal(TQFolder * folder, TQTaggable* tags);
  
public:

  int Compare(const TObject* obj) const override;
  void sort();
  int getDepth() const;
  int getWidth() const;

  void printEvaluation() const;
  void printEvaluation(Long64_t iEvent) const;
  void printWeightComponents() const;
  void printWeightComponents(Long64_t iEvent) const;
 
  static void writeDiagramHeader(std::ostream & os, TQTaggable& tags);
  static void writeDiagramFooter(std::ostream & os, TQTaggable& tags);
  int writeDiagramText(std::ostream& os, TQTaggable& tags, TString pos = "");
  bool writeDiagramToFile(const TString& filename, const TString& options = "");
  bool writeDiagramToFile(const TString& filename, TQTaggable& tags);
  bool printDiagram(TQTaggable& options);
  bool printDiagram(const TString& options);
  TString writeDiagramToString(TQTaggable& tags);
  TString getNodeName();

  TList * exportDefinitions(bool terminatingColon = false);

  static bool isValidName(const TString& name_);

  static bool parseCutDefinition(const TString& definition_, TString * name_, 
                                 TString * baseCutName_, TString * cutExpr_, TString * weightExpr_);

  static bool parseCutDefinition(TString definition_, TString& name_, 
                                 TString& baseCutName_, TString& cutExpr_, TString& weightExpr_);

  static TQCut * createCut(const TString& definition_);
  static TQCut * createCut(const TString& name, const TString& cutExpr, const TString& weightExpr = "");

  static TQCut * importFromFolder(TQFolder * folder, TQTaggable* tags = NULL);


  TQCut();
  TQCut(const TString& name_, const TString& title_ = "", const TString& cutExpression = "", const TString& weightExpression = "");

  TQCut * getBase();
  TString getPath();
  const TQCut * getBaseConst() const;
  TQCut * getRoot();

  bool isDescendantOf(TString cutName);

  static bool isTrivialTrue(const TString& expr);
  static bool isTrivialFalse(const TString& expr);

  TString getActiveCutExpression() const;
  TString getActiveWeightExpression() const;

  void printActiveCutExpression(size_t indent = 0) const;

  void setCutExpression(const TString& cutExpression);
  const TString& getCutExpression() const;
  TString getCompiledCutExpression(TQTaggable* tags);
  TString getGlobalCutExpression(TQTaggable* tags = NULL);
 
  void setWeightExpression(const TString& weightExpression);
  const TString& getWeightExpression() const;
  TString getCompiledWeightExpression(TQTaggable* tags);
  TString getGlobalWeightExpression(TQTaggable* tags = NULL);
  
  inline void setSkipAnalysisJobsGlobal(bool skip=true) {this->fSkipAnalysisJobsGlobal = skip;}
  inline bool getSkipAnalysisJobsGlobal() {return this->fSkipAnalysisJobsGlobal;}
  
  TQObservable* getCutObservable();
  TQObservable* getWeightObservable();

  bool addCut(const TString& definition_);
  bool addCut(TQCut * cut);
  bool addCut(TQCut * cut, const TString& baseCutName);
 
  TQCut * addAndReturnCut(const TString& definition_);

  bool includeBase();
  bool removeCut(const TString& name);

  void printCuts(const TString& options = "r");
  void printCut(const TString& options = "r");
  void print(const TString& options = "r");
  void printAnalysisJobs(const TString& options = "");
  
  int dumpToFolder(TQFolder * folder);

  virtual bool isMergeable() const;

  void consolidate();

  TQCut * getCut(const TString& name);
  void getMatchingCuts(TObjArray& matchingCuts, const TString& name);
  void getMatchingCuts(TObjArray& matchingCuts, const TObjArray& name_sep, int offset=0);
  void propagateMatchingCuts(TObjArray& matchingCuts, const TObjArray& name_sep, int offset=0);
  bool isResidualMatchingSegmentOptional(const TObjArray& name_segments, int offset=0);
  TObjArray * getCuts();
  TObjArray * getListOfCuts();
  virtual TObjArray * getListOfBranches();
  TQCut * getSingleCut(TString name, TString excl_pattern = "PATTERNYOUWON'TSEE");

  void setCuts(TObjArray* cuts);

  virtual bool initialize(TQSample * sample);
  virtual bool finalize();
  TQSample* getSample();
  inline const std::vector<TQSampleFolder*>& getInitializationHistory() const { return this->fInitializationHistory;} 
  
  bool canInitialize(TQSampleFolder* sf) const;
  bool canFinalize(TQSampleFolder* sf) const;
  bool initializeSampleFolder(TQSampleFolder* sf);
  bool finalizeSampleFolder (TQSampleFolder* sf);
 
  bool addAnalysisJob(TQAnalysisJob * newJob_, const TString& cuts_ = "");
  int getNAnalysisJobs();
  void clearAnalysisJobs();
  TObjArray* getJobs();

  TQCut* getClone();
  TQCut* getCompiledClone(TQTaggable* tags);

  virtual bool passed() const;
  virtual double getWeight() const;

  virtual bool passedGlobally() const;
  virtual double getGlobalWeight() const;

  bool executeAnalysisJobs(double weight);
  void analyse(double weight, bool useWeights);
  
  virtual ~TQCut();
 
  ClassDefOverride(TQCut, 1); // a cut to be applied during an analysis

};

#endif
