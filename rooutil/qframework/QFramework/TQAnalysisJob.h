//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQAnalysisJob__
#define __TQAnalysisJob__

#include "TNamed.h"
#include "QFramework/TQSampleFolder.h"


class TQSample;
class TQCut;

class TQAnalysisJob : public TNamed {
 
protected:
 
  TQCut * fCut; //!
 
  TQSample * fSample; //!
 
  void copyTransientMembersFrom(TQAnalysisJob* other);

public:

  TQAnalysisJob();
  TQAnalysisJob(const TString& name_);
 
  virtual void reset();

  virtual void print(const TString& options = "");
  
  virtual bool initializeSampleFolder(TQSampleFolder* sf);
  virtual bool finalizeSampleFolder (TQSampleFolder* sf);
 
  virtual TQAnalysisJob * getClone();

  virtual TString getDescription();

  void setCut(TQCut * cut_);
  TQCut * getCut();
  int addToCuts(TList* cuts, const TString& cutname = "*");
 
  virtual TObjArray * getBranchNames();
 
  bool initialize(TQSample * sample);
  bool finalize();

  virtual bool initializeSelf() = 0;
  virtual bool finalizeSelf() = 0;
 
  virtual bool execute(double weight);
 
  virtual ~TQAnalysisJob();
 
  ClassDefOverride(TQAnalysisJob, 1); // analysis job to be run during the event loop on every event

};

#endif
