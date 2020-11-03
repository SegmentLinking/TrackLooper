//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQHistComparer__
#define __TQHistComparer__

#include "QFramework/TQPresenter.h"
#include "TLegend.h"

class TQHistComparer : public TQPresenter {
protected:

  TObjArray * fHists;
  TObjArray * fSummaryHists;
  TObjArray * fDistNames;
  TLegend * makeLegend(TQTaggable& tags, TObjArray* histos);



public:

  TQHistComparer(TQSampleFolder* sf);
  TQHistComparer(TQSampleDataReader* reader);

  void addDistribution (TString name, TString title="");
  bool resetDistributions();

  bool writeToFile(const TString& filename, const TString& filename_summary, const TString& tags);
  bool writeToFile(const TString& filename, const TString& filename_summary, TQTaggable& tags);

  ClassDefOverride(TQHistComparer,0) //QFramework class

};

#endif
