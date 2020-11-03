//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQ_GRIDSCANPOINT__
#define __TQ_GRIDSCANPOINT__

#include "TH1.h"
#include "TString.h"
#include "QFramework/TQSignificanceEvaluator.h"

class TQGridScanPoint : public TObject {
public:
  std::vector<TString> variables;
  std::vector<double> coordinates;
  std::vector<TString> switchStatus;
  double significance;
  TString evalInfoStr;
  Long64_t id;

  TQGridScanPoint();
  TQGridScanPoint(std::vector<TString>* vars, std::vector<double>& coords, std::vector<TString> switchStatus);
  TQGridScanPoint(std::vector<TString>& vars, std::vector<double>& coords, std::vector<TString> switchStatus);
  ~TQGridScanPoint();

  void clear();

  static bool greater(const TQGridScanPoint* first, const TQGridScanPoint* second);
  static bool smaller(const TQGridScanPoint* first, const TQGridScanPoint* second);

  friend bool operator < (const TQGridScanPoint& first, const TQGridScanPoint& second);
  friend bool operator > (const TQGridScanPoint& first, const TQGridScanPoint& second);
 
  ClassDefOverride(TQGridScanPoint,2)  // auxiliary class for the TQGridScanner
};

#endif

