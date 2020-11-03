//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQHWWPlotter__
#define __TQHWWPlotter__

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

class TQHWWPlotter : public TQPlotter {

protected:
  virtual TCanvas * makePlot(TString histogram, TQTaggable& inputTags);
  int global_hist_counter;
 
  void setStyle(TQTaggable& tags);
  void setStyleIllinois(TQTaggable& tags);
  TObjArray* collectHistograms(TQTaggable& tags);
  void drawRatio(TQTaggable& tags);
  void drawPull(TQTaggable& tags);
  void drawMultiRatio(TQTaggable& tags);
  void drawSignificance(TQTaggable& tags);
  void drawLabels(TQTaggable& tags);
  void makeLegend(TQTaggable& tags, TObjArray* histos);
  void stackHistograms(TQTaggable& tags,const TString& stackname);
  bool calculateAxisRanges1D(TQTaggable& tags);
  bool drawStack(TQTaggable& tags);
  bool drawProfiles(TQTaggable& tags);
  void drawLegend(TQTaggable& tags);
  void drawContours(TQTaggable& tags);
  void drawHeatmap(TQTaggable& tags);
  void drawMigration(TQTaggable& tags);
  void drawOptScan(TQTaggable& tags);
  void drawDataMinusBackground(TQTaggable& tags);
  void drawDataMinusBackgroundOverData(TQTaggable& tags);
  void drawCustomSubPlot(TQTaggable& tags);
  void drawArrows(TQTaggable &tags,TGraph *ratioGraph, double min, double max, bool verbose);
  TString getScaleFactorList(TString histname);

  bool sysOk;

  bool includeSystematicsLegacy(TQTaggable& tags, TQTaggable& aliases);

public:
 
  static TCanvas * plotHistogram(TH1 * histo, const TString& options = "");
  static bool plotAndSaveHistogram(TH1 * histo, const TString& saveAs, const TString& options = "");
  static TCanvas* plotHistograms(TH1 * data, TH1* mc, const TString& options = "style.showRatio=true, histstyles.mc.histFillColor=kRed");
  static bool plotAndSaveHistograms(TH1 * histo, TH1* mc, const TString& saveAs, const TString& options = "style.showRatio=true, histstyles.mc.histFillColor=kRed");
 
  virtual bool setTotalBkgSystematics(const TString& path);


  virtual void reset();
 
  TQHWWPlotter();
  TQHWWPlotter(TQSampleFolder * baseSampleFolder);
  TQHWWPlotter(TQSampleDataReader * dataSource);
 
  bool setScheme(TString scheme);
 
  virtual ~TQHWWPlotter();
 
  ClassDefOverride(TQHWWPlotter, 3); //QFramework class
 
};

typedef TQHWWPlotter TQHWWPlotter2;

#endif
