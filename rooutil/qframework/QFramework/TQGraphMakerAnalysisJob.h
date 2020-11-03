//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQGraphMakerAnalysisJob__
#define __TQGraphMakerAnalysisJob__

#include "QFramework/TQObservable.h"
#include "QFramework/TQAnalysisJob.h"
#include "QFramework/TQMessageStream.h"
#include "TList.h"
#include "QFramework/TQTaggable.h"
#include "QFramework/TQCut.h"

class TQGraphMakerAnalysisJob : public TQAnalysisJob {
 
protected:

  static TQMessageStream f_ErrMsg;
  int f_Verbose;
 
  std::vector<TNamed*> fGraphTemplates;
  std::vector<std::vector<TString> > fExpressions; 
  std::vector<std::vector<TQObservable*> > fObservables; //!
  std::vector<TClass*> fGraphTypes;
 
  /* actual graphs */
  // IMPORTANT NOTE:
  // fGraphs is declared a transient data member
  // hence, it is not serialized and deserialized by calls of TObject::Clone()
  // this is of critical importance, since copying would lead to invalid pointers 
  // in the field of the cloned object -- NEVER CHANGE THIS!!!
  std::vector<TNamed*> fGraphs; //!
 
  void setErrorMessage(TString message);
  void initializeGraphs();
  bool finalizeGraphs();
 
  TQSampleFolder* poolAt;

public:

  TQGraphMakerAnalysisJob();
  TQGraphMakerAnalysisJob(TQGraphMakerAnalysisJob* other);

  static const TString& getValidNameCharacters();
 
  static int importJobsFromTextFiles(const TString& files, TQCut* basecut, const TString& channelfilter = "*", bool verbose = false);
  static int importJobsFromTextFiles(const TString& files, TQCut* basecut, TQTaggable* aliases, const TString& channelfilter = "*", bool verbose = false);
  static int importJobsFromTextFiles(const std::vector<TString>& filenames, TQCut* basecut, const TString& channelFilter = "*", bool verbose = false);
  static int importJobsFromTextFiles(const std::vector<TString>& filenames, TQCut* basecut, TQTaggable* aliases, const TString& channelFilter = "*", bool verbose = false);

  bool initializeSampleFolder(TQSampleFolder* sf) override;
  bool finalizeSampleFolder (TQSampleFolder* sf) override;

  bool bookGraph(TString definition, TQTaggable* aliases = NULL);
  void cancelGraph(const TString& name);
  void printBooking(const TString& moretext);
  void printBookingTeX(const TString& moretext);
  
  static void clearMessages();

  void setVerbose(int verbose);
  int getVerbose();

  virtual TQAnalysisJob * getClone() override;

  virtual void reset() override;
  virtual void print(const TString& options) override;

  
  static TString getErrorMessage();

  virtual bool initializeSelf() override;
  virtual bool finalizeSelf() override;

  virtual bool execute(double weight) override;

  virtual TObjArray * getBranchNames() override;

  virtual ~TQGraphMakerAnalysisJob();
 
  ClassDefOverride(TQGraphMakerAnalysisJob, 0); // analysis job to create graphs

};

#endif
