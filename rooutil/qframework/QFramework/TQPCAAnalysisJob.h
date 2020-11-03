//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQPCAAnalysisJob__
#define __TQPCAAnalysisJob__

#include "QFramework/TQAnalysisJob.h"
#include "TObject.h"
#include "QFramework/TQPCA.h"
#include "QFramework/TQObservable.h"

class TQSample;

class TQPCAAnalysisJob : public TQAnalysisJob {

protected:
 
  TQPCA* fPCA = NULL;
  double* fValues = NULL;
  std::vector<TString> fNames;
  std::vector<TString> fTitles;
  std::vector<TString> fExpressions;
  std::vector<TQObservable*> fObservables; //!
  TQSampleFolder* poolAt = NULL;
  double weightCutoff = 0;

  bool initializePCA();
  bool finalizePCA();
  bool checkValues();
 
public:
 
  TQPCAAnalysisJob();
  TQPCAAnalysisJob(TString name_);
 
  virtual TQPCAAnalysisJob* copy();
  virtual TQPCAAnalysisJob* getClone() override;
  TQPCAAnalysisJob(TQPCAAnalysisJob &job);
  TQPCAAnalysisJob(TQPCAAnalysisJob* job);

  virtual void bookVariable(const TString& name, const TString& title, const TString& expression);
  virtual void bookVariable(const TString& expression);
 
  virtual bool initializeSampleFolder(TQSampleFolder* sf) override;
  virtual bool finalizeSampleFolder(TQSampleFolder* sf) override;
  virtual bool initializeSelf() override;
  virtual bool finalizeSelf() override;
  virtual bool execute(double weight) override;
 
  virtual void setWeightCutoff(double cutoff);
  virtual double getWeightCutoff();

  virtual ~TQPCAAnalysisJob();

  virtual TObjArray * getBranchNames() override;
 
  ClassDefOverride(TQPCAAnalysisJob, 1); // analysis job preparing a principal component analysis
 
};

#endif
