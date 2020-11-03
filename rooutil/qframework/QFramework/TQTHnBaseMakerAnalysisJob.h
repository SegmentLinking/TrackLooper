//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQTHnBaseMakerAnalysisJob__
#define __TQTHnBaseMakerAnalysisJob__

#include "QFramework/TQObservable.h"
#include "QFramework/TQAnalysisJob.h"
#include "QFramework/TQMessageStream.h"
#include "THnBase.h"
#include "QFramework/TQTaggable.h"
#include "QFramework/TQCut.h"

class TQTHnBaseMakerAnalysisJob : public TQAnalysisJob {
 
protected:

  static bool g_useHistogramObservableNames;
  static TQMessageStream f_ErrMsg;
  int f_Verbose;
 
  /* template histograms (will be cloned) */
  std::vector<THnBase*> fHistogramTemplates;
  std::vector<bool> fFillSynchronized;
  std::vector<bool> fFillRaw;
 
  /* tree observables */
  std::vector<std::vector<TString> > fExpressions; 
  std::vector<std::vector<TQObservable*> > fObservables; //!
  /* histo types */
  std::vector<int> fHistoTypes;
 
  /* actual histograms */
  // IMPORTANT NOTE:
  // fHistograms is declared a transient data member
  // hence, it is not serialized and deserialized by calls of TObject::Clone()
  // this is of critical importance, since copying would lead to invalid pointers 
  // in the field of the cloned object -- NEVER CHANGE THIS!!!
  std::vector<THnBase*> fHistograms; //!
 
  void setErrorMessage(TString message);
  void initializeHistograms();
  bool finalizeHistograms();
 
  TQSampleFolder* poolAt;

public:

  TQTHnBaseMakerAnalysisJob();
  TQTHnBaseMakerAnalysisJob(TQTHnBaseMakerAnalysisJob* other);

  static const TString& getValidNameCharacters();
 
  static int importJobsFromTextFiles(const TString& files, TQCut* basecut, const TString& channelfilter = "*", bool verbose = false);
  static int importJobsFromTextFiles(const TString& files, TQCut* basecut, TQTaggable* aliases, const TString& channelfilter = "*", bool verbose = false);
  static int importJobsFromTextFiles(const std::vector<TString>& filenames, TQCut* basecut, const TString& channelFilter = "*", bool verbose = false);
  static int importJobsFromTextFiles(const std::vector<TString>& filenames, TQCut* basecut, TQTaggable* aliases, const TString& channelFilter = "*", bool verbose = false);

  bool initializeSampleFolder(TQSampleFolder* sf) override;
  bool finalizeSampleFolder (TQSampleFolder* sf) override;

  bool bookHistogram(TString definition, TQTaggable* aliases = NULL);
  void cancelHistogram(const TString& name);
  
  static void clearMessages();

  void setVerbose(int verbose);
  int getVerbose();

  virtual TQAnalysisJob * getClone() override;

  virtual void reset() override;
  
  static TString getErrorMessage();

  virtual bool initializeSelf() override;
  virtual bool finalizeSelf() override;

  virtual bool execute(double weight) override;

  virtual ~TQTHnBaseMakerAnalysisJob();
 
  ClassDefOverride(TQTHnBaseMakerAnalysisJob, 0); // analysis job to book histograms of arbitrary dimension

};

#endif
