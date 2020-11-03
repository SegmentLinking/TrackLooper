//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQCompPlotter__
#define __TQCompPlotter__

#include "TH1.h"
#include "THStack.h"
#include "TNamed.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TString.h"
#include "QFramework/TQTaggable.h"
#include "QFramework/TQNamedTaggable.h"
#include "QFramework/TQSampleFolder.h"
#include "TObjArray.h"
#include "QFramework/TQPlotter.h"

#include "TGraph.h"
#include "TGraphErrors.h"
#include "TGraphAsymmErrors.h"

#include <math.h>
#include <limits>

class TQSampleDataReader;

class TQCompPlotter : public TQPlotter {

protected:
  virtual TCanvas * makePlot(TString histogram, TQTaggable& inputTags);
  int global_hist_counter;
 
  void setStyle(TQTaggable& tags);
  TObjArray* collectHistograms(TQTaggable& tags);
  void drawRatio(TQTaggable& tags);
  void drawLabels(TQTaggable& tags);
  void makeLegend(TQTaggable& tags, TObjArray* histos);
  bool calculateAxisRanges1D(TQTaggable& tags);
  void drawLegend(TQTaggable& tags);
  void drawArrows(TQTaggable &tags,TGraphErrors *ratioGraph, double min, double max, bool verbose);
  bool drawHistograms(TQTaggable& tags);
  TString getScaleFactorList(TString histname);

public:


  virtual void reset();
 
  TQCompPlotter();
  TQCompPlotter(TQSampleFolder * baseSampleFolder);
  TQCompPlotter(TQSampleDataReader * dataSource);
  
  virtual ~TQCompPlotter();
 
  ClassDefOverride(TQCompPlotter, 1); //QFramework class
 
};

//typedef TQCompPlotter TQCompPlotter2;

#endif
