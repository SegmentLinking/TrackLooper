//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQEventlistAnalysisJob__
#define __TQEventlistAnalysisJob__

#include "QFramework/TQAnalysisJob.h"
#include "QFramework/TQObservable.h"
#include "QFramework/TQTable.h"
#include "TObject.h"
#include "TList.h"
#include "QFramework/TQMessageStream.h"

#include <vector>

class TQCounter;
class TQSample;


class TQEventlistAnalysisJob : public TQAnalysisJob {
protected:
 
  std::vector<TString> fTitles;
  std::vector<TString> fExpressions;
  std::vector<TQObservable*> fObservables;
  int nFormulas;
 
  TQTable * fEventlist; //!
  Long64_t fEventIndex;
 
  bool showWeightColumn;

  static TQMessageStream f_ErrMsg;
  int f_Verbose;

  void setErrorMessage(TString message);
 
public:

  TQEventlistAnalysisJob();
  TQEventlistAnalysisJob(const TString& name_);

  virtual void reset() override;
  void addColumn(const TString& expression, const TString& label = "");
  void setWeightColumn(bool weight = true);

  static TString getErrorMessage();
  static void clearMessages();

  virtual TObjArray * getBranchNames() override;
  virtual TQEventlistAnalysisJob* getClone() override;

  virtual bool initializeSelf() override;
  virtual bool finalizeSelf() override;

  int nColumns() const;

  static int importJobsFromTextFiles(const TString& files, TQCut* basecut, const TString& channelFilter="*", bool verbose=false);
  static int importJobsFromTextFiles(const std::vector<TString>& filenames, TQCut* basecut, const TString& channelFilter="*", bool verbose=false);
  static int importJobsFromTextFiles(const TString& files, TQCut* basecut, TQTaggable* aliases, const TString& channelFilter="*", bool verbose=false);
  static int importJobsFromTextFiles(const std::vector<TString>& filenames, TQCut* basecut, TQTaggable* aliases, const TString& channelFilter="*", bool verbose=false);

  virtual bool execute(double weight) override;
 
  virtual ~TQEventlistAnalysisJob();
 
  ClassDefOverride(TQEventlistAnalysisJob, 0); // analysis job that creates an event list

};

#endif
