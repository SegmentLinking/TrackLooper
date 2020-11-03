//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQHistoMakerAnalysisJob__
#define __TQHistoMakerAnalysisJob__

#include "QFramework/TQObservable.h"
#include "QFramework/TQAnalysisJob.h"
#include "QFramework/TQMessageStream.h"
#include "TList.h"
#include "QFramework/TQTaggable.h"
#include "QFramework/TQCut.h"

class TQHistoMakerAnalysisJob : public TQAnalysisJob {
 
protected:

  static bool g_useHistogramObservableNames;
  static TQMessageStream f_ErrMsg;
  int f_Verbose;
 
  /* template histograms (will be cloned) */
  std::vector<TH1*> fHistogramTemplates;
  
  /* histogram specific options*/
  std::vector<bool> fFillSynchronized;
  std::vector<bool> fFillRaw;
 
  /* observables */
  std::vector<std::vector<TString> > fExpressions; 
  std::vector<TString> fWeightExpressions;
  std::vector<std::vector<TQObservable*> > fObservables; //!
  std::vector<TQObservable*> fWeightObservables; //!
  
  /* histo types */
  std::vector<int> fHistoTypes;
 
  /* actual histograms */
  // IMPORTANT NOTE:
  // fHistograms is declared a transient data member
  // hence, it is not serialized and deserialized by calls of TObject::Clone()
  // this is of critical importance, since copying would lead to invalid pointers 
  // in the field of the cloned object -- NEVER CHANGE THIS!!!
  std::vector<TH1*> fHistograms; //!
 
  void setErrorMessage(TString message);
  void initializeHistograms();
  bool finalizeHistograms();
 
  TQSampleFolder* poolAt;

public:

  TQHistoMakerAnalysisJob();
  TQHistoMakerAnalysisJob(TQHistoMakerAnalysisJob* other);

  static const TString& getValidNameCharacters();
 
  static int importJobsFromTextFiles(const TString& files, TQCut* basecut, const TString& channelfilter = "*", bool verbose = false);
  static int importJobsFromTextFiles(const TString& files, TQCut* basecut, TQTaggable* aliases, const TString& channelfilter = "*", bool verbose = false);
  static int importJobsFromTextFiles(const std::vector<TString>& filenames, TQCut* basecut, const TString& channelFilter = "*", bool verbose = false);
  static int importJobsFromTextFiles(const std::vector<TString>& filenames, TQCut* basecut, TQTaggable* aliases, const TString& channelFilter = "*", bool verbose = false);

  bool initializeSampleFolder(TQSampleFolder* sf) override;
  bool finalizeSampleFolder (TQSampleFolder* sf) override;

  bool bookHistogram(TString definition, TQTaggable* aliases = NULL);
  void cancelHistogram(const TString& name);
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

  virtual ~TQHistoMakerAnalysisJob();
 
  ClassDefOverride(TQHistoMakerAnalysisJob, 0); // analysis job to create histograms

};

#endif
