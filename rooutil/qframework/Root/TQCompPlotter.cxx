#include "TCanvas.h"
#include "TH1.h"
#include "TStyle.h"
#include "TROOT.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TLegend.h"
#include "TLatex.h"
#include "THStack.h"
#include "TParameter.h"
#include "TMap.h"
#include "TMath.h"
#include "TGraphAsymmErrors.h"
#include "TGraphErrors.h"
#include "TArrow.h"
#include "TLine.h"
#include "TH2.h"
#include "TH3.h"
#include "TGaxis.h"
#include "TFormula.h"
#include "TF1.h"
#include "TF2.h"
#include "TFitResult.h"
#include "TRandom3.h"
#include "TObjArray.h"

#include "QFramework/TQNamedTaggable.h"
#include "QFramework/TQCompPlotter.h"
#include "QFramework/TQSampleDataReader.h"
#include "QFramework/TQHistogramUtils.h"
#include "QFramework/TQStringUtils.h"
#include "QFramework/TQUtils.h"
#include "QFramework/TQIterator.h"

// #define _DEBUG_
#include "QFramework/TQLibrary.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <stdlib.h>
#include <cmath>

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQCompPlotter:
//
// The TQCompPlotter provides advanced plotting features for the H->WW->lvlv analysis.
// It inherits basic plotting functionality from the abstract TQPlotter base class.
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQCompPlotter)

//__________________________________________________________________________________|___________

TQCompPlotter::TQCompPlotter() : TQPlotter() {
  // Default constructor of TQCompPlotter class
}

//__________________________________________________________________________________|___________

TQCompPlotter::TQCompPlotter(TQSampleFolder * baseSampleFolder) : TQPlotter(baseSampleFolder) {
  // Constructor of TQCompPlotter class
}

//__________________________________________________________________________________|___________

TQCompPlotter::TQCompPlotter(TQSampleDataReader * dataSource) : TQPlotter(dataSource) {
  // Constructor of TQCompPlotter clas
}

//______________________________________________________________________________________________

TQCompPlotter::~TQCompPlotter() {
  // Destructor of TQCompPlotter class:
}

//__________________________________________________________________________________|___________

void TQCompPlotter::reset() {
  // reset all the processes 
  TQPlotter::reset();
}

//__________________________________________________________________________________|___________

void TQCompPlotter::setStyle(TQTaggable& tags){
  // setup the default style tags
  double padScaling = 1.;
  double ratioPadScaling = 1.;
  double additionalTopMargin = 0.;
  bool showSub = tags.getTagBoolDefault("style.showSub",false);
  double ratioPadRatio = tags.getTagDoubleDefault("geometry.sub.height",0.35);
  //@tag: [style.topLegend] If this argument tag is set to true, the legend is shown at the top of the plot. Default: false.
  if(tags.getTagBoolDefault("style.topLegend",false)){
    tags.setTagDouble("geometry.main.margins.top",0.25);
    tags.setTagInteger("style.legend.fillColor",kWhite);
    tags.setTagInteger("style.legend.fillStyle",1001);
    tags.setTagDouble("labels.info.yPos",0.9);
    tags.setTagDouble("style.labels.yPos",0.9);
    tags.setTagBool("style.showMissing",false);
    tags.setTagDouble("legend.yMin",0.75);
    tags.setTagDouble("legend.yMax",0.95);
  }

  tags.setTagDouble("geometry.main.margins.top", 0.05);
  if((tags.hasTag("style.heatmap")|| tags.hasTag("style.migration")) && tags.getTagIntegerDefault("style.nDim",1) == 2){ //only change the margin if it's an actual 2D plot.
      tags.setTagDouble("geometry.main.margins.right", 0.12);
  } else {
    tags.setTagDouble("geometry.main.margins.right", 0.05);
  }
  tags.setTagDouble("geometry.main.margins.left", 0.16);
 
  tags.setTagDouble("geometry.main.xMin",0.);
  tags.setTagDouble("geometry.main.xMax",1.);
  tags.setTagDouble("geometry.main.yMax",1.);

  tags.setTagInteger("geometry.canvas.height",600);
  if (showSub){
    //@tag: [style.ratioTopMargin] This argument tag sets the top margin for ratio plots. Default: 0.015
    tags.setTagDouble("geometry.sub.margins.top", tags.getTagDoubleDefault("style.ratioTopMargin",0.015));
    tags.setTagDouble("geometry.sub.margins.bottom",0.16);
    tags.setTagDouble("geometry.sub.margins.left", 0.16);
    tags.setTagDouble("geometry.sub.margins.right", 0.05);
    tags.setTagInteger("geometry.canvas.width",600);
    padScaling = 1. / 8. * 6.;
    ratioPadScaling = (1. / ratioPadRatio) / 8. * 6.;
    additionalTopMargin = 0.1 * (padScaling - 1);
    tags.setTagDouble("geometry.main.additionalTopMargin",additionalTopMargin);
    tags.setTagDouble("geometry.main.scaling",padScaling);
    tags.setTagDouble("geometry.sub.scaling",ratioPadScaling);
 
    tags.setTagDouble("geometry.sub.xMin",0.);
    tags.setTagDouble("geometry.sub.yMin",0.);
    tags.setTagDouble("geometry.sub.xMax",1.);
    tags.setTagDouble("geometry.sub.yMax",ratioPadRatio);
    tags.setTagDouble("geometry.main.yMin",0);
    tags.setTagDouble("geometry.main.margins.bottom",ratioPadRatio/padScaling);
 
    tags.setTagBool("style.main.xAxis.showLabels",false);
    tags.setTagBool("style.main.xAxis.showTitle",false);
  } else {
    tags.setTagDouble("geometry.main.margins.bottom",0.16);
    tags.setTagDouble("geometry.main.yMin",0.);
    tags.setTagInteger("geometry.canvas.width",800);
  }

  tags.setTagInteger("style.text.font",42);
  tags.setTagDouble("style.textSize",0.05);
  // tags.setTagDouble("style.markerSize",1.2);
  tags.setTagInteger("style.markerType",20);

  tags.setTagInteger("style.main.lineColor",0);
  tags.setTagInteger("style.main.markerColor",0);
  tags.setTagInteger("style.main.fillColor",0);
  tags.setTagInteger("style.main.fillStyle",0);

  tags.setTagInteger("style.main.totalBkg.lineColor",kBlue);
  tags.setTagInteger("style.main.totalBkg.lineWidth",1);
  tags.setTagInteger("style.main.totalBkg.fillColor",0);
  tags.setTagInteger("style.main.totalBkg.fillStyle",0);
  tags.setTagInteger("style.main.totalBkgError.fillColor",14);
  tags.setTagInteger("style.main.totalBkgError.fillStyle",3254);

  tags.setTagInteger("style.ratio.mcErrorBand.fillColor",kOrange -2);
  tags.setTagInteger("style.ratio.mcErrorBand.fillStyle",1001);
  tags.setTagInteger("style.optScan.default.fillColor",kOrange -2);
  tags.setTagInteger("style.optScan.default.fillStyle",1001);
  tags.setTagInteger("style.optScan.left.fillColor",kRed);
  tags.setTagInteger("style.optScan.left.fillStyle",1001);
  tags.setTagInteger("style.optScan.right.fillColor",kBlue);
  tags.setTagInteger("style.optScan.right.fillStyle",1001);
 
  tags.setTagInteger("style.main.data.lineWidth",2);
  // tags.setTagDouble ("style.main.data.markerSize",1.0);
  // tags.setTagInteger("style.main.data.lineColor",kBlack);

  tags.setTagInteger("style.significance.fillColor",kRed);
  tags.setTagInteger("style.significance.fillStyle",1001);
  tags.setTagInteger("style.significance.lineColor",0);
  tags.setTagInteger("style.significance.lineStyle",0);

  //@tag:[style.tickLength] This argument tag controls the length of the x- and y-axis ticks. Default: 0.03
  double tickLength = tags.getTagDoubleDefault("style.tickLength",0.03);
  tags.setTagDouble("geometry.main.yAxis.tickLength",tickLength);
  tags.setTagDouble("geometry.main.xAxis.tickLength",tickLength);
  tags.setTagDouble("geometry.sub.yAxis.tickLength", tickLength*0.8); 
  // I found that the y ticks need to be scaled by 0.8 in the ratio plot in order to match the main plot
  // if anybody understands the reason behind this, please add a comment or eMail me: cburgard@cern.ch
  tags.setTagDouble("geometry.sub.xAxis.tickLength",tickLength);

  //@tag:[style.sub.yAxis.nDiv] This tag controls the number of divisions/ticks of the sub plot. The number of top level ticks is given by the two least significant digits (in decimal notation). The second two least significant digits determine the number of sub divisions (smaller ticks), the thrid least significant set of two digits controls the sub-sub-devisions. Default: 510 (10 top level divisions, 5 sub divisions)
  tags.setTagInteger("style.sub.yAxis.nDiv",tags.getTagIntegerDefault("style.ratio.nYdiv",510));

  tags.setTagDouble("legend.xMin",0.59);
  tags.setTagDouble("legend.xMax",0.90);
  tags.setTagDouble("legend.yMin",0.70);
  tags.setTagDouble("legend.yMax",0.92);

  tags.setTagBool("errors.showX",true);
  tags.setTagDouble("erros.widthX",0.5);
}

//__________________________________________________________________________________|___________

TObjArray* TQCompPlotter::collectHistograms(TQTaggable& tags){
  // use the TQSampleDataReader to retrieve all histograms from the sample folder
  
  bool verbose = tags.getTagBoolDefault("verbose",false );
  //@tag:[normalization] This argument tag allows to normalize histograms. Integral is normalized to the given value.
  bool normalize = tags.hasTag("normalization");
  tags.setTag("normalize",normalize);

  //@tag: [style.showUnderflow,style.showOverflow] This argument tag controls if under/overflow bins are shown in the histogram. Default: false.
  bool showUnderflow = tags.getTagBoolDefault ("style.showUnderflow",false);
  bool showOverflow = tags.getTagBoolDefault ("style.showOverflow",false );
  tags.setTagBool("includeOverflow",showOverflow);
  tags.setTagBool("includeUnderflow",showUnderflow);

  double normalization = tags.getTagDoubleDefault("normalization", 1.);
  bool includeUnderOverFlow = tags.getTagBoolDefault("includeUnderOverFlow",false);
  TString integralOptions = tags.getTagStringDefault("integralOptions",""); //set this to "width" to include the bin width when computing the histogram integral for normalization
  
  std::vector<TString> paths = std::vector<TString>();
  std::vector<TString> cuts = std::vector<TString>();
  if (tags.getTag("path",paths) < 1) {
    ERRORclass("no paths defined (path.0=\"processPath\")");
    tags.printTags();
    return NULL;
  }
  if (tags.getTag("name",cuts) < 1) {
    ERRORclass("no histograms defined (name.0=\"cut/variable\")");
    tags.printTags();
    return NULL;
  }
  if (paths.size() != cuts.size()) {
    ERRORclass("lists of paths and histograms have different lengths");
    return NULL;
  }
  TObjArray* histos = new TObjArray();
  
  for (size_t i=0; i<paths.size(); ++i) {
    TQTaggable tmpTags;
    tmpTags.importTagsWithoutPrefix(tags,TString::Format("style.%d.",(int)i),true);
    tmpTags.importTagsWithoutPrefix(tags,"style.",false);
    tmpTags.importTagsWithoutPrefix(this,"style.",false);
    tmpTags.setTagBool("includeOverflow",showOverflow);
    tmpTags.setTagBool("includeUnderflow",showUnderflow);
    TH1* h = this->fReader->getHistogram(paths.at(i),cuts.at(i),&tmpTags);
    if (!h) {
      WARNclass("Failed to retrieve histogram");
      continue;
    }
    if (h->GetDimension() > 1) {
      WARNclass("Only one dimensional histograms are supported, skipping...");
      continue;
    }
    h->SetName(TString::Format("hist.%d",(int)i));
    if (normalize) h->Scale(normalization/h->Integral( (includeUnderOverFlow ? 0 : 1) , h->GetNbinsX() + (includeUnderOverFlow ? 1 : 0)  , integralOptions));
    h->SetTitle(tmpTags.getTagStringDefault(TString::Format("label.%d",(int)(i)),h->GetTitle()));
    histos->Add(h);
  }
  
  // check consistency and create master histogram 
  TH1* hMaster = NULL;
  bool consistent = checkConsistency(hMaster, histos);
 
  // stop if there is no valid histogram or histograms are invalid
  if (!consistent){
    if(verbose) VERBOSEclass("consistency check failed");
    delete histos;
    return NULL;
  }
  if (!hMaster){
    if(verbose) VERBOSEclass("no histograms found");
    delete histos;
    return NULL;
  }
  
  hMaster->Reset();
  hMaster->SetTitle(tags.getTagStringDefault("title",hMaster->GetTitle()));
  TQTH1Iterator itr(histos);
  TH1* hMax = TQHistogramUtils::copyHistogram(hMaster,"totalBkg");
  while(itr.hasNext()) {
    TH1* h = itr.readNext();
    for (int i=0; i<std::min(TQHistogramUtils::getNBins(h),TQHistogramUtils::getNBins(hMax)); ++i) {
      hMax->SetBinContent(i,std::max(h->GetBinContent(i),hMax->GetBinContent(i)));
    }
  }
  this->addObject(hMax,"totalBkg");
  this->addObject(hMaster,"master");
  this->addObject(histos,"histos");

  return histos;
}

//__________________________________________________________________________________|___________

void TQCompPlotter::makeLegend(TQTaggable& tags, TObjArray* histos){
  // create a legend including the given list of histograms
  // @tags:style.showEventYields: show event yields (integral counts) in the legend
  // @tags:style.showEventYields.useUnderOverflow: include underflow and overflow in the event yields displayed in the legend (default:true)
  // @tags:style.nLegendCols: number of columns to be shown in legend
  // @tags:style.legendHeight: scaling factor for height of the legend
  bool showEventYields = tags.getTagBoolDefault ("style.showEventYields",false);
  int nLegendCols = tags.getTagIntegerDefault ("style.nLegendCols",showEventYields ? 1 : 2);
  double legendHeight = tags.getTagDoubleDefault ("style.legendHeight",1. );

  // the nominal coordinates of the legend

  // @tags:[geometry.legend.xMin,geometry.legend.xMax,geometry.legend.yMin,geometry.legend.yMax]: control the geometry of the legend in relative coordinates
  double x1 = tags.getTagDoubleDefault("geometry.legend.xMin",0.59);
  double y1 = tags.getTagDoubleDefault("geometry.legend.yMin",0.70) - tags.getTagDoubleDefault("geometry.main.additionalTopMargin",0.);
  double x2 = tags.getTagDoubleDefault("geometry.legend.xMax",0.90);
  double y2 = tags.getTagDoubleDefault("geometry.legend.yMax",0.92) - tags.getTagDoubleDefault("geometry.main.additionalTopMargin",0.);

  // if we plot the ratio, the canvas has to be divided which results in a
  // scaling of the legend. To avoid this, we have to rescale the legend's
  // position
  y1 = y2 - (y2 - y1) * tags.getTagDoubleDefault("geometry.main.scaling",1.);

  // calculate the number of entries
  int nEntries = 0;
  //@tag: style.showMissing: show empty legend entries where histogram is empty or could not be retrieved (default:true)
  bool showMissing = tags.getTagBoolDefault ("style.showMissing",true );
 
  nEntries += (showMissing ? histos->GetEntriesFast() : histos->GetEntries());

  // calculate the height of the legend
  int nLegendRows = (int)nEntries / nLegendCols + ((nEntries % nLegendCols) > 0 ? 1 : 0);
  legendHeight *= (y2 - y1) * (double)nLegendRows / 5.;

  // set the height of the legend
  y1 = y2 - legendHeight;
  // create the legend and set some attributes
  double tmpx1 = x1; 
  double tmpx2 = x2; 
  double tmpy1 = y1; 
  double tmpy2 = y2; 
 
  TLegend* legend = new TLegend(tmpx1, tmpy1, tmpx2, tmpy2);
  this->addObject(legend,"legend");
  legend->SetBorderSize(0);
  legend->SetNColumns(nLegendCols);
  if(tags.getTagBoolDefault("style.useLegendPad",false)){
    //@tags:style.useLegendPad: put the legend on a separate pad on the side of the plot
    legend->SetFillColor(0);
    legend->SetFillStyle(0);
  } else {
    //@tags:[style.legend.fillColor,style.legend.fillStyle]: control color and style of the legend with TLegend::SetFillColor and TLegend::SetFillStyle. defaults are 0.
    legend->SetFillColor(tags.getTagIntegerDefault("style.legend.fillColor",0));
    legend->SetFillStyle(tags.getTagIntegerDefault("style.legend.fillStyle",0));
  }
  //@tags: style.legend.textSize: control the font size (floating point number, default is 0.032)
  //@tags: style.legend.textSizeFixed: boolean to control whether the text size will be interpreted relative to canvas size (default) or absolute
  double textsize = tags.getTagDoubleDefault("style.legend.textSize",0.032);
  if (tags.getTagBoolDefault ("style.legend.textSizeFixed", false))
    legend->SetTextSize(textsize);
  else
    legend->SetTextSize(textsize * tags.getTagDoubleDefault("geometry.main.scaling",1.));
 
  
    //addAllHistogramsToLegend(tags,legend, ".isBackground", tags.getTagStringDefault("legend.dataDisplayType",".legendOptions='lep'"));
  
  
  // add the total background histogram to the legend
  TQTH1Iterator itr(histos);
  while (itr.hasNext()) {
    TH1* h = itr.readNext();
    if (!h) continue;
    TString id = h->GetName();
    TQStringUtils::removeLeadingText(id,"hist.");
    legend->AddEntry(h,tags.getTagStringDefault("label."+id,id));
  }
}

//__________________________________________________________________________________|___________

void TQCompPlotter::drawLegend(TQTaggable& tags){
  // draw the legend produced by TQCompPlotter::makeLegend
  bool verbose = tags.getTagBoolDefault("verbose",false);
  if(verbose) VERBOSEclass("drawing legend");
  TLegend* legend = this->getObject<TLegend>("legend");
  // draw the legend
  if(tags.getTagBoolDefault("style.useLegendPad",false)){
    this->getPad("legend");
    legend->Draw();
  } else {
    legend->Draw("same");
  }
}

//__________________________________________________________________________________|___________

TCanvas * TQCompPlotter::makePlot(TString/*histogram*/, TQTaggable& tags) {
  // master-function controlling the plotting
  bool verbose = tags.getTagBoolDefault("verbose",false);
 
  if(verbose) VERBOSEclass("entering function");
  
  gStyle->SetOptStat(false);
  gStyle->SetOptFit(false);
  gStyle->SetOptTitle(false);

  //////////////////////////////////////////////////////
  // obtain the histograms
  //////////////////////////////////////////////////////

  if(verbose) VERBOSEclass("collecting histograms");
  TObjArray* histos = this->collectHistograms(tags);
  if(!histos) return NULL;

  /*
  TQTH1Iterator histitr(histos);
  int nEntries = 0;
  bool is2D = false;
  while(histitr.hasNext()){
    TH1* hist = histitr.readNext();
    nEntries += hist->GetEntries();
    if(dynamic_cast<TH2*>(hist)) is2D=true;
  }
  if(nEntries < 1){
    WARNclass("refusing to plot histogram '%s' - no entries!",histogram.Data());
    return NULL;
  }
*/
  //////////////////////////////////////////////////////
  // the ratio plot can only be shown if there is a valid
  // data histogram and at least one MC background histogram 
  //////////////////////////////////////////////////////

  //if(verbose) VERBOSEclass("sanitizing tags");
  
  // @tag:style.overrideTotalBkgRequirement: usually, 1D plots without background are skipped. force plotting data/signal only with this option.
  
  TH1* hMaster = this->getObject<TH1>("master");
  if (!hMaster){
    if(verbose) VERBOSEclass("no master histogram found, quitting");
    return NULL;
  }
  TH1* hTotalBkg = this->getObject<TH1>("totalBkg");
  if (!hTotalBkg){
    if(verbose) VERBOSEclass("no maximum-bin-entries histogram found, quitting");
    return NULL;
  }

  /*@tag:[style.showRatio]
    
    control what is shown in the sub-plot. all of these default to 'false', only showing the main plot.
    if any of these are set to true, the corresponding sub plot is shown. only one sub plot can be shown at a time.

    style.showRatio: show the ratio between data and signal+bkg
   */
  bool showRatio = tags.getTagBoolDefault ("style.showRatio",false);
  int nDim = tags.getTagIntegerDefault("style.nDim",1);
  if(nDim != 1){
    showRatio = false;
  }
  
  //////////////////////////////////////////////////////
  // set and apply the style
  //////////////////////////////////////////////////////

  if(verbose) VERBOSEclass("applying style");
  this->setStyle(tags);

  double xscaling = tags.getTagDoubleDefault("geometry.xscaling",1.);
  //@tags:[geometry.main.*] control geometry parameters of the main pad
  this->applyGeometry(tags,hMaster, "main", xscaling,tags.getTagDoubleDefault("geometry.main.scaling",1.));
  this->applyStyle (tags,hMaster, "main", xscaling,tags.getTagDoubleDefault("geometry.main.scaling",1.));
  this->applyStyle (tags,hTotalBkg,"main.totalBkg",xscaling,tags.getTagDoubleDefault("geometry.main.scaling",1.));
 
  //////////////////////////////////////////////////////
  // canvas and pads
  //////////////////////////////////////////////////////

  if(verbose) VERBOSEclass("creating canvas");
  TCanvas * canvas = this->createCanvas(tags);
  if(!canvas) return NULL;
  canvas->Draw();
  TPad* pad = this->getPad("main");
  if(!pad){
    delete canvas;
    return NULL;
  }
  canvas->cd();

  bool axisOK = this->calculateAxisRanges1D(tags);
  if(!axisOK){
    if(verbose) VERBOSEclass("encountered invalid axis ranges, using defaults");
  }
  //////////////////////////////////////////////////////
  // create the stack
  //////////////////////////////////////////////////////
 /*
  if(hTotalBkg && nDim == 1){
    if(verbose) VERBOSEclass("stacking histograms");
    this->stackHistograms(tags,"stack");
  }
*/
  //////////////////////////////////////////////////////
  // legend
  //////////////////////////////////////////////////////

  if(verbose) VERBOSEclass("creating legend");
  if(tags.getTagBoolDefault("style.useLegendPad",false)){
    canvas->cd();
    TPad * legendPad = this->createPad(tags,"legend");
    legendPad->Draw();
    this->getPad("legend");
    if(!legendPad) return NULL;
    this->makeLegend(tags,histos);
  } else {
    this->makeLegend(tags,histos);
  }

  //////////////////////////////////////////////////////
  // basic label setup
  //////////////////////////////////////////////////////

  if(verbose) VERBOSEclass("setting labels");
  TString label;
  tags.setGlobalOverwrite(true);
  /*
  int labelidx = 0;  
  if (tags.getTagString("labels.lumi", label)){
    labelidx++;
    tags.setTagString(TString::Format("labels.%d",labelidx), label);
  }
  if (tags.getTagString("labels.process", label)){
    labelidx++;
    tags.setTagString(TString::Format("labels.%d",labelidx), label);
  }
  */

//  tags.setGlobalOverwrite(false);

  //////////////////////////////////////////////////////
  // draw main pad
  //////////////////////////////////////////////////////

  if(verbose) VERBOSEclass("drawing main pad");
  this->getPad("main");
  if(nDim == 1){
    if(!tags.getTagBoolDefault("allow1D",true))	return NULL;
    bool ok = this->drawHistograms(tags);
    if(!ok){
      return NULL;
    }
    this->drawLegend(tags);
    if(verbose) VERBOSEclass("drawing cut lines");
    this->drawCutLines1D(tags);
  } else {
    ERRORclass("unsupported dimensionality (nDim=%d)!",nDim);
    return NULL;
  }

  //////////////////////////////////////////////////////
  // draw the labels
  //////////////////////////////////////////////////////

  if(verbose) VERBOSEclass("drawing labels");
  this->setAxisLabels(tags);
  this->drawLabels(tags);

  //////////////////////////////////////////////////////
  // redraw main pad
  //////////////////////////////////////////////////////
  
  if(verbose) VERBOSEclass("refreshing drawing area");
  pad->RedrawAxis();
  pad->Update();
  pad->cd();

  //////////////////////////////////////////////////////
  // draw sub pad
  //////////////////////////////////////////////////////

  if(tags.getTagBoolDefault("style.showSub",false)){
    if(verbose) VERBOSEclass("drawing subplot");
    canvas->cd();
    
    if (showRatio){
      if(verbose) VERBOSEclass("drawing ratio");
      this->drawRatio(tags);
    } 
  }

  if(verbose) VERBOSEclass("all done!");
  // return the canvas
  return canvas;
}

//__________________________________________________________________________________|___________

bool TQCompPlotter::drawHistograms(TQTaggable& tags){
  // draw the stack produced by TQHWWPlotter::stackHistograms
  
  
  bool verbose = tags.getTagBoolDefault("verbose",false);

  TH1* hMaster = this->getObject<TH1>("master");
  if(!hMaster) return false;

  // the first histogram to draw is the SM histogram.
  hMaster->Draw("hist");
 
  
  if(tags.getTagBoolDefault("errors.showX",true)){
    double errWidthX = 0.5;
    if(tags.getTagDouble("errors.widthX", errWidthX))
      gStyle->SetErrorX(errWidthX);
  } else {
    gStyle->SetErrorX(0.);
  }

  //////////////////////////////////////////////////////
  // calculate axis ranges
  // rescale to avoid graphical collisions
  //////////////////////////////////////////////////////

  if(verbose) VERBOSEclass("calculating axis ranges & rescaling histograms");
  bool axisOK = this->calculateAxisRanges1D(tags);
  if(!axisOK){
    if(verbose) VERBOSEclass("encountered invalid axis ranges, using defaults");
  }

  //////////////////////////////////////////////////////
  // draw everything
  //////////////////////////////////////////////////////

  //@tag:style.*.drawOptions: control the draw options of this process (default: 'hist' for MC, 'ep' for data)
  
  if(verbose) VERBOSEclass("drawing histograms");
  // draw signal
  TQTH1Iterator itr(this->getObject<TObjArray>("histos"));
  while(itr.hasNext()){
    TH1* h = itr.readNext();
    if(!h) continue;
    TString id = h->GetName();
    TQStringUtils::removeLeadingText(id,"hist.");
    TQTaggable tmpTags;
    tmpTags.importTagsWithoutPrefix(tags,"style."+id+".");
    //set all kinds of drawing options for the histogram
    h->SetFillStyle(tmpTags.getTagIntegerDefault("fillStyle",0));
    h->SetFillColorAlpha(tmpTags.getTagIntegerDefault("fillColor",0),tmpTags.getTagDoubleDefault("fillColorAlpha",1.));
    h->SetLineColorAlpha(tmpTags.getTagIntegerDefault("lineColor",1),tmpTags.getTagDoubleDefault("lineColorAlpha",1.));
    h->SetLineWidth(tmpTags.getTagIntegerDefault("lineWidth",1));
    h->SetLineStyle(tmpTags.getTagIntegerDefault("lineStyle",1));
    h->SetMarkerColorAlpha(tmpTags.getTagIntegerDefault("markerColor",1),tmpTags.getTagDoubleDefault("makerColorAlpha",1.));
    h->SetMarkerSize(tmpTags.getTagDoubleDefault("markerSize",1.0));
    h->SetMarkerStyle(tmpTags.getTagIntegerDefault("markerStyle",1));
    
    h->Draw(tmpTags.getTagStringDefault("drawOptions", "hist") + " same");
  }
  
  return true;
}

//__________________________________________________________________________________|___________

bool TQCompPlotter::calculateAxisRanges1D(TQTaggable& tags){
  // calculate the axis ranges, taking into account the given block tags
  bool logScale = tags.getTagBoolDefault ("style.logScale",false );
  bool verbose = tags.getTagBoolDefault("verbose",false);
 
  TH1* hTotalBkg = this->getObject<TH1>("totalBkg");
  TList* histograms = new TList();
  if (hTotalBkg) histograms->Add(hTotalBkg);

  double min = std::numeric_limits<double>::infinity();
  TQTH1Iterator itr(this->getObject<TObjArray>("histos"));
  while(itr.hasNext()){
    TH1* h = itr.readNext();
    if(!h) continue;
    histograms->Add(h);
    double tmpmin = TQHistogramUtils::getMin(h, true);
    if(tmpmin < min) min = tmpmin;
  }
  
  if(logScale && min < tags.getTagDoubleDefault("style.logMinMin",1e-9) ) min = tags.getTagDoubleDefault("style.logMinMin",1e-9);

  bool showSub = tags.getTagBoolDefault ("style.showSub",false);
  double yVetoLeft = 0.84;
  TList* labels = tags.getTagList("labels");
  if (labels) yVetoLeft -= (showSub ? 0.08 : 0.09) * (double)labels->GetEntries() * tags.getTagDoubleDefault("geometry.main.scaling",1.);
  delete labels;

  double yVetoRight = tags.getTagDoubleDefault("legend.yMin",0.5) - tags.getTagDoubleDefault("legend.margin.right",0.05);
 
  tags.setTagDouble("blocks.x.0",0.5); tags.setTagDouble("blocks.y.0",yVetoLeft);
  tags.setTagDouble("blocks.x.1",1.0); tags.setTagDouble("blocks.y.1",yVetoRight);

  double max_precise = this->getHistogramUpperLimit(tags, histograms,min,true);
  delete histograms;

  tags.getTagDouble("style.min", min);
  if(logScale){
    tags.getTagDoubleDefault("style.logMin",min);
  } else {
    tags.getTagDoubleDefault("style.linMin",min);
  }

  double max;
  if(max_precise <= 0 || !TQUtils::isNum(max_precise) || max_precise < min){
    max = std::max(2*min,10.);
    if(verbose) VERBOSEclass("using default range");
  } else {
    if(verbose) VERBOSEclass("using rounded range");
    max = TQUtils::roundAutoUp(max_precise);
  }
 
  if(verbose) VERBOSEclass("calculated y-axis range is %g < y < %g (%g)",min,max,max_precise);
 
  // the user might want to overwrite the automatic upper limit on the y axis
  tags.getTagDouble("style.max", max);
  double maxscale = 1.0;
  tags.getTagDouble("style.max.scale", maxscale);

  TH1* hMaster = this->getObject<TH1>("master");
  hMaster->SetMinimum(min);
  hMaster->SetMaximum(max * maxscale);

  return !(max == 0);
}

//__________________________________________________________________________________|___________

void TQCompPlotter::drawLabels(TQTaggable& tags){
  // draw the labels given by the tags

  if(!tags.getTagBoolDefault("style.showLabels",true)) return;

  double scaling = tags.getTagDoubleDefault("geometry.main.scaling",1.);
  double textsize = tags.getTagDoubleDefault("style.textSize",0.05);
  int font = tags.getTagDoubleDefault("style.text.font",42);
  int color = tags.getTagDoubleDefault("style.text.color",1);
  double xscaling = tags.getTagDoubleDefault("geometry.xscaling",1.);

  double x = tags.getTagDoubleDefault("style.labels.xOffset",0.2)*xscaling;
 
  double y = tags.getTagDoubleDefault("style.labels.yPos",0.86 - tags.getTagDoubleDefault("geometry.main.additionalTopMargin",0.));
 
  bool drawATLAS = tags.getTagBoolDefault ("labels.drawATLAS",true);
 
  TString nfLabel = "";
  if(tags.getTagBoolDefault ("labels.drawNFInfo",false)){
    TString tmpLabel = tags.getTagStringDefault("labels.nfInfo","#color[2]{(NF applied for %s)}");
    if(TQStringUtils::countText(tmpLabel,"%s") == 1){
      TString nflist = this->getScaleFactorList(tags.getTagStringDefault("input.bkg",""));
      if(!nflist.IsNull()){
        nfLabel = TString::Format(tmpLabel.Data(),nflist.Data());
      }
    }
  }

  TString infoLabel = tags.getTagBoolDefault ("labels.drawInfo",true) ? tags.getTagStringDefault ("labels.info",TString::Format("Plot: \"%s\"", tags.getTagStringDefault("input.name","histogram").Data())) : "";
  TString atlasLabel = tags.getTagStringDefault ("labels.atlas","Private");
  TString stickerLabel = tags.getTagStringDefault ("labels.sticker","");
 
  if (drawATLAS) {
    // draw the ATLAS label
    TLatex l;
    l.SetNDC();
    l.SetTextFont(72);
    l.SetTextSize(textsize * tags.getTagDoubleDefault("labels.drawATLAS.scale",1.25) * scaling);
    l.SetTextColor(1);
    l.DrawLatex(x, y, tags.getTagStringDefault("labels.drawATLAS.text","ATLAS"));
  }
 
  if (!atlasLabel.IsNull()){
    // draw the ATLAS label addition
    TLatex p;
    p.SetNDC();
    p.SetTextFont(font);
    p.SetTextColor(color);
    p.SetTextSize(textsize * tags.getTagDoubleDefault("labels.atlas.scale",1.25) * scaling);
    p.DrawLatex(x + tags.getTagDoubleDefault("labels.atlas.xOffset",0.16)*xscaling, y, atlasLabel.Data());
  }
 
  if (!infoLabel.IsNull()){
    // draw the info label
    if(!nfLabel.IsNull()){
      infoLabel.Prepend(" ");
      infoLabel.Prepend(nfLabel);
    }
    TLatex l0;
    l0.SetNDC();
    l0.SetTextFont(font);
    bool newPlotStyle = tags.getTagBoolDefault ("style.newPlotStyle", false);
    if (newPlotStyle)
      l0.SetTextSize(textsize * tags.getTagDoubleDefault("labels.info.size",0.6) * scaling * 0.7);
    else
      l0.SetTextSize(textsize * tags.getTagDoubleDefault("labels.info.size",0.6) * scaling);
    l0.SetTextColor(color);
    double xpos = tags.getTagDoubleDefault("geometry.main.margins.left",0.16) + tags.getTagDoubleDefault("labels.info.xPos",1.)*(1. - tags.getTagDoubleDefault("geometry.main.margins.right",0.05) - tags.getTagDoubleDefault("geometry.main.margins.left",0.16));
    double ypos = 1. - scaling*(1.-tags.getTagDoubleDefault("labels.info.yPos",0.2))*tags.getTagDoubleDefault("geometry.main.margins.top",0.05);
    l0.SetTextAlign(tags.getTagIntegerDefault("labels.info.align",31));
    l0.DrawLatex(xpos, ypos, infoLabel.Data());
  }

  // draw additional labels
  TQIterator itr(tags.getTagList("labels"),true);
  double marginStep = tags.getTagDoubleDefault("style.labels.marginStep",0.06);
  double labelTextScale = tags.getTagDoubleDefault("style.labels.scale",0.85);
  size_t index = 1;
  while(itr.hasNext()){
    TObject* obj = itr.readNext();
    if(!obj) break;
    TLatex latex;
    latex.SetNDC();
    latex.SetTextFont(font);
    latex.SetTextSize(textsize * labelTextScale * scaling);
    latex.SetTextColor(color);
    latex.DrawLatex(x, y - marginStep * index * scaling,obj->GetName());
    index++;
  }
}

//__________________________________________________________________________________|___________

//__________________________________________________________________________________|___________

void TQCompPlotter::drawRatio(TQTaggable& tags){
  // draw a ratio-plot in the sub-pad
  double ratioMax = tags.getTagDoubleDefault ("style.ratioMax",1000.);
  double ratioMin = tags.getTagDoubleDefault ("style.ratioMin",0.);
  double ratioMaxQerr = tags.getTagDoubleDefault ("style.ratioMaxQerr",std::numeric_limits<double>::infinity());
  bool forceRatioLimits = tags.getTagBoolDefault ("style.forceRatioLimits",false );
  //bool asymmSysErrorBand = tags.getTagBoolDefault("errors.drawAsymmSysMC", false);
  bool verbose = tags.getTagBoolDefault("verbose",false);
  TObjArray* hArr = this->getObject<TObjArray>("histos");
  if (!hArr) {
    throw std::runtime_error("Failed to retrieve histogram array!");
  }
  int denom_index = tags.getTagIntegerDefault("denominatorIndex",0);
  std::vector<int> num_index = tags.getTagVInteger("numeratorIndex");
  if (num_index.size() < 1) {
    num_index.push_back(1);
  }
  
  TH1* h_denominator = dynamic_cast<TH1*>(hArr->FindObject(  ("hist."+std::to_string(denom_index)).c_str()  ));
  if(!h_denominator) return;
  if (verbose) VERBOSEclass("Found denominator histogram");
  
  //TObjArray* histosAsymmSys = 0;
  //if (asymmSysErrorBand) {
  //  histosAsymmSys = this->getObject<TObjArray>("asymmSys");
  //}
 
  TPad* ratioPad = this->getPad("sub");
  if(!ratioPad) return;
  ratioPad->cd();
 
  int nBins = h_denominator->GetNbinsX();

  int nPoints = 0;
  for (int i = 1; i <= nBins; i++) {
    if (h_denominator->GetBinContent(i) != 0.) {
      nPoints++;
    }
  }

  // the graph used to draw the error band on the ratio
  if(verbose) VERBOSEclass("generating ratio error graphs");
  TGraphAsymmErrors * ratioErrorGraph = new TGraphAsymmErrors(nPoints);
  ratioErrorGraph->SetTitle("Monte Carlo ratio error band");
  this->addObject(ratioErrorGraph,"ratioErrorGraph");
  //TGraphAsymmErrors * asymmErrorGraph;
  //if (asymmSysErrorBand){
  //  asymmErrorGraph = TQHistogramUtils::getGraph(h_denominator, histosAsymmSys);
  //  this->addObject(asymmErrorGraph,"asymmSysErrorBand");
  //}

  int iPoint = 0;
  for (int iBin = 1; iBin <= nBins; iBin++) {
    double MC = h_denominator->GetBinContent(iBin);
    double MCErr = h_denominator->GetBinError(iBin);
    double MCErrUpper = MCErr;
    double MCErrLower = MCErr;
    //if (asymmSysErrorBand) {
    //  MCErrUpper = asymmErrorGraph->GetErrorYhigh(iBin);
    //  MCErrLower = asymmErrorGraph->GetErrorYlow(iBin);
    //}
    if(MCErr == 0 || MC == 0) continue;
    double ratioBandErrorUpper = MCErrUpper / MC;
    double ratioBandErrorLower = MCErrLower / MC;
    // set the position and the width of the ratio error band
    ratioErrorGraph->SetPoint(iPoint, h_denominator->GetBinCenter(iBin), 1.);
 
    ratioErrorGraph->SetPointError(iPoint, h_denominator->GetBinWidth(iBin) / 2.,
                                   h_denominator->GetBinWidth(iBin) / 2.,
                                   ratioBandErrorLower, ratioBandErrorUpper);
    // if shape sys turned on we will have asymmetric error
    iPoint++;
  }

  if(verbose) VERBOSEclass("calculating geometry and axis ranges");
  // set the x range of the ratio graph to match the one of the main histogram
  double xLowerLimit = h_denominator->GetBinLowEdge(1);
  double xUpperLimit = h_denominator->GetBinLowEdge(nBins) + h_denominator->GetBinWidth(nBins);
  ratioErrorGraph->GetXaxis()->SetLimits(xLowerLimit, xUpperLimit);
  double ratioPadScaling = tags.getTagDoubleDefault("geometry.sub.scaling",1.);
  double xscaling = tags.getTagDoubleDefault("geometry.xscaling",1.);
  this->applyStyle (tags,ratioErrorGraph,"ratio.mcErrorBand",xscaling,ratioPadScaling);
  this->applyGeometry(tags,ratioErrorGraph,"sub" ,xscaling,ratioPadScaling);

  if(verbose) VERBOSEclass("drawing ratio error graph");
  ratioErrorGraph->Draw("A2");
  TH1* hMaster = this->getObject<TH1>("master");
  ratioErrorGraph->GetHistogram()->GetXaxis()->SetTitle(hMaster->GetXaxis()->GetTitle());
 
  TString dataLabel("");
  TList* graphs = new TList();
  TList* hists = new TList();
  for (size_t i=0; i<num_index.size();++i) {
    if (verbose) VERBOSEclass("trying to add 'hist.%d'",num_index[i]);
    TH1* h = dynamic_cast<TH1*>( hArr->FindObject(("hist."+std::to_string(num_index[i])).c_str()));
    if (!h) continue;
    hists->Add(h);
  }
  
  // graphs->SetOwner(true);

  // actual minimum and maximum ratio
  double actualRatioMin = 1.;
  double actualRatioMax = 1.;

  if(verbose) VERBOSEclass("generating ratio graphs");
  // loop over data histograms
  TQTH1Iterator itr(hists);

  /*if (tags.getTagBoolDefault("useToyData",false))
  {
    // set the x range of the dmb graph to match the one of the main histogram
    TH1 * h_numerator = TQHistogramUtils::copyHistogram(this->getObject<TH1>("toyData"),"tmp");
    TQTaggableIterator itr2(fProcesses);
    if(dataLabel.IsNull()) dataLabel = h_numerator->GetTitle();
    hists->Add(h_numerator);
  }*/

  while(itr.hasNext()){
    if (verbose) VERBOSEclass("generating numerator variant");
    // get the data histogram
    TH1* h_numerator = itr.readNext();
    //if(!hist) continue;
    //if(!hist->getTagBoolDefault(".isData",false)) continue;
    //TH1 * h_numerator = this->getObject<TH1>(this->makeHistogramIdentifier(process));
    if(!h_numerator) continue;
    if(dataLabel.IsNull()) dataLabel = h_numerator->GetTitle();
    //hists->Add(h_numerator);

    // calculate the number of valid ratio points: ratio points are considered
    // valid if they have a finite value (MC prediction != 0) (--> nPoints) and
    // the observed data is greater than zero (--> nRatioPoints)
    int nRatioPoints = 0;
    for (int i = 1; i <= nBins; i++) {
      double mcVal = h_denominator->GetBinContent(i);
      double dataVal = h_numerator->GetBinContent(i);
      if(mcVal == 0) continue;
      if(dataVal == 0) continue;
      if(!TQUtils::isNum(mcVal)){
        WARNclass("encountered non-numeric MC value: %f",mcVal);
        continue;
      }
      if(!TQUtils::isNum(dataVal)){
        WARNclass("encountered non-numeric data value: %f",dataVal);
        continue;
      }
      nRatioPoints++;
    }
 
    if(nRatioPoints < 1){
      // there is nothing to draw -- well, let's do nothing, then
      continue;
    }
 
    // the graph used to draw the ratio points
    TGraphErrors * ratioGraph = new TGraphErrors(nRatioPoints);
    this->addObject(ratioGraph,TString::Format("ratioGraph_%s",h_numerator->GetName()));
    ratioGraph->SetTitle(TString::Format("%s (ratio)",h_numerator->GetTitle()));
    ratioGraph->SetLineColor(h_numerator->GetLineColor());
    ratioGraph->SetMarkerSize(h_numerator->GetMarkerSize());
    ratioGraph->SetMarkerStyle(h_numerator->GetMarkerStyle());
    ratioGraph->SetMarkerColor(h_numerator->GetMarkerColor());
 
    int iRatioPoint = 0;
 
    // loop over all bins of the histogram
    for (int iBin = 1; iBin <= nBins; iBin++) {
      double x = h_denominator->GetBinCenter(iBin);
      // get the values and errors of data and MC for this bin
      //"numerator" = "data" in HWWPlotter (h_denominator = hTotalBackground)
      double numerator = h_numerator ->GetBinContent(iBin);
      double numeratorErr = h_numerator ->GetBinError (iBin);
      double denominator = h_denominator->GetBinContent(iBin);
      // cannot do anything if MC expectation is zero
      if (denominator == 0. || numerator <= 0.) continue;
 
      double ratio = numerator / denominator;
      double ratioError = ratio * numeratorErr / numerator;
      if(verbose) VERBOSEclass("adding ratio point with x=%f, y=%f (numerator=%f, denominator=%f)",x,ratio,numerator,denominator);
      ratioGraph->SetPoint(iRatioPoint, x, ratio);
      ratioGraph->SetPointError(iRatioPoint, 0., ratioError);
      iRatioPoint++;
    }
 
    this->applyStyle(tags   ,ratioGraph,"sub.data",1.,ratioPadScaling);
    
    double ratioMinAllowed = tags.getTagDoubleDefault ("style.ratioMinAllowed",ratioMin);
    double ratioMaxAllowed = tags.getTagDoubleDefault ("style.ratioMaxAllowed",ratioMax);
    actualRatioMin=ratioMinAllowed;
    actualRatioMax=ratioMaxAllowed;
    if(verbose) VERBOSEclass("drawRatio: allowed range of ratio graph: %f -- %f",actualRatioMin,actualRatioMax);

    this->estimateRangeY(ratioGraph,actualRatioMin,actualRatioMax,ratioMaxQerr);
 
    if(verbose) VERBOSEclass("drawRatio: estimated range of ratio graph: %f -- %f (ratioMaxQerr=%f)",actualRatioMin,actualRatioMax,ratioMaxQerr);

    if(actualRatioMin == actualRatioMax){
      if(verbose) VERBOSEclass("expanding ratio to not be empty");
      actualRatioMin *= 0.9;
      actualRatioMax *= 1.1;
    }
    
    if (forceRatioLimits)
      actualRatioMin = ratioMin;
    else 
      actualRatioMin = actualRatioMin-0.1*(actualRatioMax-actualRatioMin);
 
    if (forceRatioLimits)
      actualRatioMax = ratioMax;
    else
      actualRatioMax = actualRatioMax+0.1*(actualRatioMax-actualRatioMin);
 
    if(verbose) VERBOSEclass("drawRatio: final of ratio graph: %f -- %f",actualRatioMin,actualRatioMax);
 
    ratioErrorGraph->GetHistogram()->SetMaximum(actualRatioMax);
    ratioErrorGraph->GetHistogram()->SetMinimum(actualRatioMin);
 
    graphs->Add(ratioGraph);
  }

  TString totalBkgLabel = tags.getTagStringDefault ("labels.denominator", "SM");
  tags.getTagString("labels.numerator",dataLabel);

  ratioErrorGraph->GetHistogram()->GetXaxis()->SetTitle(hMaster->GetXaxis()->GetTitle());
  ratioErrorGraph->GetYaxis()->SetTitle(dataLabel + " / "+ tags.getTagStringDefault ("labels.denominator", "denominator") +" ");
 
  gStyle->SetEndErrorSize(0); 


  if(verbose) VERBOSEclass("drawing lines markers");
  // if 1. is included in the range of the y axis of the ratio plot...
  this->applyStyle(tags,ratioErrorGraph->GetHistogram()->GetYaxis(),"sub.yAxis");
  if ((ratioErrorGraph->GetHistogram()->GetMinimum() <= 1) && (ratioErrorGraph->GetHistogram()->GetMaximum() >= 1.)) {
    // draw the red line indicating 1 in the ratio plot and around 0 in case of
    // significance
    TLine * line = new TLine(xLowerLimit, 1., xUpperLimit, 1.);
    line->SetLineColor(kRed);
    line->Draw();
  }


  double textsize = tags.getTagDoubleDefault("style.textSize",0.05)* ratioPadScaling * tags.getTagDoubleDefault("style.ratio.fitSlope.printResults.textSize",0.5);
  TLatex l;
  l.SetNDC();
  l.SetTextSize(textsize);
  double fitResultPrintPosX = tags.getTagDoubleDefault("style.ratio.fitSlope.printResults.posX",0.2);
  double fitResultPrintPosY = tags.getTagDoubleDefault("style.ratio.fitSlope.printResults.posY",0.85);
  double fitResultPrintStepY = tags.getTagDoubleDefault("style.ratio.fitSlope.printResults.stepY",0.5);

  if(verbose) VERBOSEclass("drawing additional markers");
  TQGraphErrorsIterator itr2(graphs);
  while(itr2.hasNext()){
    TGraphErrors* ratioGraph = itr2.readNext();
    if(!ratioGraph) continue;

    if(tags.getTagBoolDefault("style.ratio.fitSlope",false)){
      ratioGraph->Fit("pol1","Q","",xLowerLimit,xUpperLimit);
      TF1* f = ratioGraph->GetFunction("pol1");
      f->SetLineColor(ratioGraph->GetLineColor());
      f->SetLineWidth(tags.getTagIntegerDefault("style.ratio.fitSlope.lineWidth",1));
      f->SetLineStyle(tags.getTagIntegerDefault("style.ratio.fitSlope.lineStyle",2));
 
      if (tags.getTagBoolDefault("style.ratio.fitSlope.printResults",false)) {
        l.SetTextColor(ratioGraph->GetLineColor());
        double slope = TQUtils::roundAuto(f->GetParameter(1));
        double slopeErr = TQUtils::roundAuto(f->GetParError(1));
        double chi2 = TQUtils::roundAuto(f->GetChisquare());
        TString s = TString::Format("slope #approx %g #pm %g (#chi^{2}#approx%g)",slope,slopeErr,chi2);
        l.DrawLatex(fitResultPrintPosX,fitResultPrintPosY,s);
        fitResultPrintPosY -= fitResultPrintStepY * textsize;
      }
 
    }
 
    ratioGraph->Draw("P SAME"); 

    this->drawArrows(tags,ratioGraph, actualRatioMin,actualRatioMax,verbose);

  }
} 



//__________________________________________________________________________________|___________


void TQCompPlotter::drawArrows(TQTaggable &tags,TGraphErrors *ratioGraph, double actualRatioMin, double actualRatioMax, bool verbose = false){
  Int_t nBins = ratioGraph->GetN();

  double arrowLength = tags.getTagDoubleDefault ("style.arrowLength",0.12 ); // fraction of the y-range
  double arrowOffset = tags.getTagDoubleDefault ("style.arrowOffset",0.08 ); // fraction of the y-range
  int arrowLineWidth = tags.getTagIntegerDefault ("style.arrowLineWidth",2 );
  double arrowHeadSize = tags.getTagDoubleDefault ("style.arrowHeadSize",0.03 );
  double padRatio = tags.getTagDoubleDefault("geometry.sub.height",0.35);

  double canvasHeight = tags.getTagDoubleDefault("geometry.canvas.height",600.);
  double canvasWidth = tags.getTagDoubleDefault("geometry.canvas.width",600.);
  double frameWidthFrac = 1. - tags.getTagDoubleDefault("geometry.sub.margins.right",0.1) - tags.getTagDoubleDefault("geometry.sub.margins.left",0.1);
  double frameWidth = frameWidthFrac * canvasWidth;
  double arrowHeight = arrowHeadSize * canvasHeight;
  double binWidth = frameWidth / nBins;
  double alpha = 2*std::atan(binWidth/(2*arrowHeight)) * 180 / 3.1415926;
 
  double arrowHeadAngle = tags.getTagDoubleDefault ("style.arrowHeadAngle",std::min(60.,alpha));
  TString arrowType = tags.getTagStringDefault ("style.arrowType", "|>" ); // refer to TArrow
  int arrowColor = tags.getTagIntegerDefault ("style.arrowColor",kRed);
 
  TArrow marker;
  marker.SetLineWidth(arrowLineWidth);
  marker.SetLineColor(arrowColor);
  marker.SetFillColor(arrowColor);
  marker.SetAngle(arrowHeadAngle);
  for(size_t i=0; i < (size_t)(ratioGraph->GetN()); i++){
    double x; double y;
    if( i != (size_t)(ratioGraph->GetPoint((int)i, x, y)))
      continue;
    double plrange = actualRatioMax - actualRatioMin;
    if(y > actualRatioMax){
      marker.DrawArrow(x,
                       actualRatioMax - (arrowOffset+arrowLength)*plrange,
                       x,
                       actualRatioMax - (arrowOffset)*plrange,
                       arrowHeadSize*padRatio,
                       arrowType);
      if(verbose) VERBOSEclass("drawing marker for point %i, y > %f",i,actualRatioMax);
    }
    if(y < actualRatioMin){
      marker.DrawArrow(x,
                       actualRatioMin + (arrowOffset+arrowLength)*plrange,
                       x,
                       actualRatioMin + arrowOffset*plrange,
                       arrowHeadSize*padRatio,
                       arrowType);
      if(verbose) VERBOSEclass("drawing marker for point %i, y < %f",i,actualRatioMin);
    }
  }
}

//__________________________________________________________________________________|___________

TString TQCompPlotter::getScaleFactorList(TString histname){
  // retrieve a comma-separated list of the scaled contributions (titles only)
  TString cutname;
  if(!TQStringUtils::readUpTo(histname,cutname,"/")) return "";
  if(!this->getNormalizationInfo()) return "";
  TQFolder* f = this->getNormalizationInfo()->getFolder(TString::Format(".cut.%s",cutname.Data()));
  if(!f) return "";
  TString retval = "";
  TQIterator itr(f->getListOfKeys(),true);
  while(itr.hasNext()){
    TObject* obj = itr.readNext();
    if(!obj) continue;
    if(!retval.IsNull()) retval.Append(",");
    retval.Append(f->getTagStringDefault(obj->GetName(),obj->GetName()));
  }
  return retval;
}

