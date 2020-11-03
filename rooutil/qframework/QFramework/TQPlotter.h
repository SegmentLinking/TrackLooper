//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQPlotter__
#define __TQPlotter__

#include "TH1.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "QFramework/TQPresenter.h"
#include "QFramework/TQStringUtils.h"

#include "TGraph.h"
#include "TGraphErrors.h"
#include "TGraphAsymmErrors.h"
#include "TProfile.h"
#include "TProfile2D.h"

#include <math.h>
#include <limits>

class TQSampleDataReader;

class TQPlotter : public TQPresenter {

protected:

  TObjArray* pads; //!
  TDirectory* objects; //!

  void drawCutLines1D(TQTaggable& tags);
  int drawAdditionalAxes(TQTaggable& tags);
  int drawHeightLines(TQTaggable& tags);
  void setAxisLabels(TQTaggable& tags);
  TCanvas* createCanvas(TQTaggable& tags);

  TString makeHistogramIdentifier(TQNamedTaggable* process);

  virtual double getHistogramUpperLimit(TQTaggable& tags, TList * histograms, double lower, bool includeErrors = false);
  virtual bool checkConsistency(TH1 * &hMaster, TObjArray * histograms);

  bool includeSystematics(TQTaggable& tags);
  void setTotalBackgroundErrors(TQTaggable& tags, bool showSysMC = false, bool showStatMC = true);

  virtual void addAllHistogramsToLegend(TQTaggable& tags, TLegend * legend, const TString& processFilter, const TString& options = "", bool reverse=false);
  virtual void addHistogramToLegend(TQTaggable& tags, TLegend * legend, const TString& identifier, TQTaggable& options);
  virtual void addHistogramToLegend(TQTaggable& tags, TLegend * legend, TH1* histo, TQTaggable& options);
  virtual void addHistogramToLegend(TQTaggable& tags, TLegend * legend, TQNamedTaggable* process, const TString& options = "");
  virtual void addHistogramToLegend(TQTaggable& tags, TLegend * legend, TH1* histo, const TString& options = "");

  virtual TObjArray * getHistograms(TObjArray* processes, const TString& tagFilter, const TString& histogramName, const TString& namePrefix, TQTaggable& aliases, TQTaggable& options);
  virtual TCanvas * makePlot(TString histogram, TQTaggable& tags) = 0;
 
  virtual void addObject(TNamed* obj, const TString& key = "");
  virtual void addObject(TGraph* obj, TString key = "");
  virtual void addObject(TH1* obj, const TString& key = "");
  virtual void addObject(TCollection* obj, const TString& key = "");
  virtual void addObject(TLegend* obj, const TString& key = "");
  virtual void removeObject(const TString& key, bool deleteObject = false);
  virtual void clearObjects();
  virtual void deleteObjects();
 
  virtual void applyStyle(TQTaggable& tags, TAxis* a, const TString& key, double distscaling = 1., double sizescaling = 1.);
  virtual void applyStyle(TQTaggable& tags, TH1* hist, const TString& key, double xscaling = 1., double yscaling = 1.);
  virtual void applyGeometry(TQTaggable& tags, TAxis* a, const TString& key, double distscaling = 1., double sizescaling = 1.);
  virtual void applyGeometry(TQTaggable& tags, TH1* hist, const TString& key, double xscaling = 1., double yscaling = 1.);
  virtual void applyStyle(TQTaggable& tags, TGraph* g, const TString& key, double xscaling = 1., double yscaling = 1.);
  virtual void applyGeometry(TQTaggable& tags, TGraph* g, const TString& key, double xscaling = 1., double yscaling = 1.);
 
  virtual TPad* createPad(TQTaggable& tags, const TString& key);

public:

  virtual int getNProcesses(const TString& tagFilter);
 
  TPad* getPad(const TString& name);
  virtual void printObjects();
  template <class Type>
  Type* getObject(const TString& key){
    // retrieve a graphics object by name and cast to the given class
    return dynamic_cast<Type*>(this->getTObject(key));
  }
  TObject* getTObject(const TString& key);
  
  TQPlotter();
  TQPlotter(TQSampleFolder * baseSampleFolder);
  TQPlotter(TQSampleDataReader * dataSource);

  virtual bool addData (TString path, TString options = "");
  virtual bool addBackground(TString path, TString options = "");
  virtual bool addSignal (TString path, TString options = "");

  virtual TCanvas * plot(TString histogram, const TString& inputTags);
  virtual TCanvas * plot(TString histogram, const char* inputTags);
  virtual TCanvas * plot(TString histogram, TQTaggable * inputTags = 0);
  virtual TCanvas * plot(TString histogram, TQTaggable& inputTags);
  virtual bool plotAndSaveAs(const TString& histogram, const TString& saveAs, const char* inputTags);
  virtual bool plotAndSaveAs(const TString& histogram, const TString& saveAs, const TString& inputTags);
  virtual bool plotAndSaveAs(const TString& histogram, const TString& saveAs, TQTaggable * inputTags = 0);
  virtual bool plotAndSaveAs(const TString& histogram, const TString& saveAs, TQTaggable& inputTags);

  static void setStyleAtlas();

  virtual ~TQPlotter(); 

  virtual void reset();

  int sanitizeProcesses();
 
  static TString createAxisTagsAsString(const TString& prefix, const TString& title, double xCoeff, double yCoeff, double constCoeff, double wMin, double wMax, double xCoord, double yCoord, int nDiv = 101);
  static bool createAxisTags(TQTaggable& tags, const TString& prefix, const TString& title, double xCoeff, double yCoeff, double constCoeff, double wMin, double wMax, double xCoord, double yCoord, int nDiv = 101);
  static TString createAxisTagsAsConfigString(const TString& prefix, const TString& title, double xCoeff, double yCoeff, double constCoeff, double wMin, double wMax, double xCoord, double yCoord, int nDiv = 101);
  static TQTaggable* createAxisTags(const TString& prefix, const TString& title, double xCoeff, double yCoeff, double constCoeff, double wMin, double wMax, double xCoord, double yCoord, int nDiv = 101);

  static void estimateRangeY(TH1* h, double& min, double &max, double tolerance = std::numeric_limits<double>::infinity());
  static void estimateRangeY(TGraphErrors* g, double& min, double &max, double tolerance = std::numeric_limits<double>::infinity());
  static void estimateRangeY(TGraphAsymmErrors* g, double& min, double &max, double tolerance = std::numeric_limits<double>::infinity());
  static void getRange(TGraphErrors* g, double &xlow, double &xhigh, double &ylow, double &yhigh, bool get_xrange=true, bool get_yrange=true, double maxQerr = std::numeric_limits<double>::infinity());
 
  ClassDefOverride(TQPlotter, 0); //QFramework class

};

#endif
