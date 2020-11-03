//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQCutflowAnalysisJob__
#define __TQCutflowAnalysisJob__

#include "QFramework/TQAnalysisJob.h"
#include "TObject.h"
#include "QFramework/TQCounter.h"

class TQSample;

class TQCutflowAnalysisJob : public TQAnalysisJob {

protected:

  TQCounter * fCounter = NULL;
  TQSampleFolder* poolAt = NULL;

	bool finalizeCounter();

public:

  TQCutflowAnalysisJob();
  TQCutflowAnalysisJob(const TString& name_);

  bool initializeSampleFolder(TQSampleFolder* sf) override;
  bool finalizeSampleFolder (TQSampleFolder* sf) override;

  using TQAnalysisJob::getBranchNames;

  virtual bool initializeSelf() override;
  virtual bool finalizeSelf() override;
  virtual bool execute(double weight) override;
 
  virtual ~TQCutflowAnalysisJob();
 
  ClassDefOverride(TQCutflowAnalysisJob, 1); // analysis job that creates counters for a cutflow

};

#endif
