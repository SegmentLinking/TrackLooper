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
#include "TProfile.h"
#include "TProfile2D.h"
#include "TGaxis.h"
#include "TFormula.h"
#include "TF1.h"
#include "TF2.h"
#include "TFitResult.h"
#include "TRandom3.h"
#include "TObjArray.h"

#include "QFramework/TQNamedTaggable.h"
#include "QFramework/TQHWWPlotter.h"
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
// TQHWWPlotter:
//
// The TQHWWPlotter provides advanced plotting features for the H->WW->lvlv analysis.
// It inherits basic plotting functionality from the abstract TQPlotter base class.
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQHWWPlotter)

//__________________________________________________________________________________|___________

TQHWWPlotter::TQHWWPlotter() : TQPlotter() {
  // Default constructor of TQHWWPlotter class
}

//__________________________________________________________________________________|___________

TQHWWPlotter::TQHWWPlotter(TQSampleFolder * baseSampleFolder) : TQPlotter(baseSampleFolder) {
  // Constructor of TQHWWPlotter class
}

//__________________________________________________________________________________|___________

TQHWWPlotter::TQHWWPlotter(TQSampleDataReader * dataSource) : TQPlotter(dataSource) {
  // Constructor of TQHWWPlotter clas
}

//______________________________________________________________________________________________

TQHWWPlotter::~TQHWWPlotter() {
  // Destructor of TQHWWPlotter class:
}

//__________________________________________________________________________________|___________

void TQHWWPlotter::reset() {
  // reset all the processes 
  TQPlotter::reset();
}


//__________________________________________________________________________________|___________

bool TQHWWPlotter::setScheme(TString scheme) {
  // set the plotting scheme
  // this is legacy functionality, please use explicit calls to addProcess instead
  //@tag:[energy,luminosity] (legacy!) This argument tag sets the center-of-mass energy and integrated luminosity denoted on the plots. Default: "(?)".
  TString energy = this->getTagStringDefault("energy","(?)");
  TString lumi = this->getTagStringDefault("luminosity","(?)");
 
  if(this->fReader && this->fReader->getSampleFolder()){
    this->fReader->getSampleFolder()->setTagBool("wildcarded?",true,"sig/?");
    this->fReader->getSampleFolder()->setTagBool("wildcarded?",true,"bkg/?");
    this->fReader->getSampleFolder()->setTagBool("wildcarded?",true,"data/?");
  }

  if (scheme.CompareTo("hww.minimal.uniflavour") == 0){
    this->resetProcesses();
    this->addBackground("bkg");
    this->addData("data");
    this->addSignal("sig/mh$(mh)");
    return true;
  }
  if (scheme.CompareTo("hww.vbf") == 0){
    resetProcesses();

    addData("data/$(lepch)");
    addBackground("bkg/$(lepch)/diboson/WW");
    addBackground("bkg/$(lepch)/diboson/NonWW");
    addBackground("bkg/$(lepch)/top/ttbar");
    addBackground("bkg/$(lepch)/top/singletop");
    addBackground("bkg/$(lepch)/Zjets");
    addBackground("bkg/$(lepch)/Wjets");
    addBackground("sig/$(lepch)/mh$(mh)/ggf");
    addSignal("sig/$(lepch)/mh$(mh)/vbf");
    setTotalBkgSystematics("bkg/$(lepch)");

    clear();
    setTagBool("style.autoStack", true);
    setTagString("labels.0", "#sqrt{s} = "+energy+" TeV, #lower[-0.2]{#scale[0.6]{#int}} Ldt = "+lumi+" fb^{-1}");
    setTagString("labels.1", "H#rightarrowWW^{*}#rightarrowl#nul#nu");
    setTagBool("style.stackSignal", (scheme.CompareTo("hww.default.sigStacked") == 0));
    return true;
  }
  if (scheme.CompareTo("hww.default") == 0 ||
      scheme.CompareTo("hww.default.sigStacked") == 0) {

    resetProcesses();
    addData("data/$(lepch)");
    addBackground("bkg/$(lepch)/diboson/WW");
    addBackground("bkg/$(lepch)/diboson/NonWW");
    addBackground("bkg/$(lepch)/top/ttbar");
    addBackground("bkg/$(lepch)/top/singletop");
    addBackground("bkg/$(lepch)/Zjets");
    addBackground("bkg/$(lepch)/Wjets");
    addBackground("");
    addSignal("sig/$(lepch)/mh$(mh)");
    setTotalBkgSystematics("bkg/$(lepch)");

    clear();
    setTagBool("style.autoStack", true);
    setTagString("labels.0", "#sqrt{s} = "+energy+" TeV, #lower[-0.2]{#scale[0.6]{#int}} Ldt = "+lumi+" fb^{-1}");
    setTagString("labels.1", "H#rightarrowWW^{*}#rightarrowl#nul#nu");
    setTagBool("style.stackSignal", (scheme.CompareTo("hww.default.sigStacked") == 0));

    return true;
  } else if (scheme.CompareTo("hww.bkgsig") == 0 ||
             scheme.CompareTo("hww.bkgsig.normalize") == 0 ||
             scheme.CompareTo("hww.bkgggfvbf") == 0 ||
             scheme.CompareTo("hww.bkgggfvbf.normalize") == 0) {

    resetProcesses();
    addBackground("bkg/$(lepch)");
    if (scheme.Contains(".bkgsig")) {
      addSignal("sig/$(lepch)/mh$(mh)");
    } else if (scheme.Contains(".bkgggfvbf")) {
      addSignal("sig/$(lepch)/mh$(mh)/ggf");
      addSignal("sig/$(lepch)/mh$(mh)/vbf");
    }
    setTotalBkgSystematics("bkg/$(lepch)");

    clear();
    setTagBool("style.autoStack", true);
    setTagBool("style.showTotalBkg", false);
    setTagBool("normalize", scheme.EndsWith(".normalize"));
    setTagInteger("style.nLegendCols", 1);
    setTagDouble("style.legendHeight", 1.75);
    setTagString("labels.0", "#sqrt{s} = "+energy+" TeV, #lower[-0.2]{#scale[0.6]{#int}} Ldt = "+lumi+" fb^{-1}");
    setTagString("labels.1", "H#rightarrowWW^{*}#rightarrowl#nul#nu");

    return true;
  } else if (scheme.CompareTo("hww.databkgsig") == 0) {

    resetProcesses();
    addData("data/$(lepch)");
    addBackground("bkg/$(lepch)");
    addSignal("sig/$(lepch)/mh$(mh)");
    setTotalBkgSystematics("bkg/$(lepch)");

    clear();
    setTagBool("style.autoStack", true);
    setTagBool("style.showTotalBkg", false);
    setTagBool("style.stackSignal", true);
    setTagInteger("style.nLegendCols", 1);
    setTagDouble("style.legendHeight", 1.75);
    setTagString("labels.0", "#sqrt{s} = "+energy+" TeV, #lower[-0.2]{#scale[0.6]{#int}} Ldt = "+lumi+" fb^{-1}");
    setTagString("labels.1", "H#rightarrowWW^{*}#rightarrowl#nul#nu");

    return true;
  } else if (scheme.CompareTo("comparePlots") == 0) {
    setTagBool("normalize",true);
    setTagBool("style.showRatio",true);
    setTagBool("style.showSub",true);
    setTagString("style.stackDrawOptions", "ep");
    setTagBool("style.showTotalBkg", false);
    setTagBool("errors.showX", false);
    TQTaggableIterator itr(this->fProcesses);
    while(itr.hasNext()){
      TQNamedTaggable * process = itr.readNext();
      if(!process) continue;
      process->setTagBool(".ignoreProcessName",true);
      process->setTagInteger("histLineWidth", 2);
      process->setTagInteger("histMarkerStyle", 20);

      if(getNProcesses(".isData") < 2)
        process->setTagDouble( "histMarkerSize", 0.9);
      else if(getNProcesses(".isData") < 4)
        process->setTagDouble( "histMarkerSize", 0.8);
      else
        process->setTagDouble( "histMarkerSize", 0.7);
      //@tag:[color] (legacy!) The value of this process tag is copied to the process tags "histMarkerColor" and "histLineColor". Defaults are 0 and 1, respectively.
      process->setTagInteger("histMarkerColor", process->getTagIntegerDefault("color",0) );
      process->setTagInteger("histLineColor", process->getTagIntegerDefault("color",1) );
      if(process->getTagBoolDefault(".isBackground",false))
        {
          if(getNProcesses(".isBackground") == 1)
            //@tag:[title] (legacy!) This process tag is copied to the process tag "labels.totalBkg" ("labels.data") if the process is the only background (data) process. Default: "Y" ("X")
            this->setTagString("labels.totalBkg",process->getTagStringDefault("title","Y"));
          else
            setTagString("labels.totalBkg","sum");
        }
      else if(process->getTagBoolDefault(".isData",false))
        {
          if(getNProcesses("isData") == 1)
            setTagString("labels.data",process->getTagStringDefault("title","X"));
          else
            setTagString("labels.data","X");
        }
      process->printTags();
    }
    return true;
  } else {
    return false;
  }
}


//__________________________________________________________________________________|___________

TCanvas * TQHWWPlotter::plotHistogram(TH1 * histo, const TString& options) {
  // plot a single histogram with the given options
  TQSampleFolder * sfTmp = TQSampleFolder::newSampleFolder("sfTmp");
  sfTmp->addObject(TQHistogramUtils::copyHistogram(histo), ".histograms+::histo");

  // parse options
  TQTaggable tags(options);

  // initialize instance of plotter
  TQHWWPlotter pl(sfTmp);
  pl.resetProcesses();
  pl.addData(".", TString::Format("drawOptions = '%s'",
                                  //@tag:[drawOptions] This argument tag is forwarded by TQHWWPlotter::plotHistogram to TQPlotter::addData. Default: "hist".
                                  tags.getTagStringDefault("drawOptions", "hist").Data()));
  tags.removeTag("drawOptions");
  pl.importTags(&tags);

  // plot
  TCanvas * cv = pl.plot("histo");

  delete sfTmp;
  return cv;
}


//__________________________________________________________________________________|___________

TCanvas * TQHWWPlotter::plotHistograms(TH1 * data, TH1* mc, const TString& options) {
  // plot a pair of histograms histograms (data and MC) with the given options
  // set individual styling options with "histstyles.data.XXX" and "histstyles.mc.XXX"

  TQTaggable tags(options);
  TQSampleFolder * sfTmp = TQSampleFolder::newSampleFolder("sfTmp");

  TQSampleFolder* dataSF = sfTmp->getSampleFolder("data+");
  TQTaggable dataTags; 
  //@tag: [histstyles.data.,histstyles.mc.; style.default.] The first two argument tag prefixes are used to apply tags either to plotting of data or plotting of MC in TQHWWPlotter::plotHistograms. Only tags continuing with "style.default." are then actually used.
  dataTags.importTagsWithoutPrefix(tags,"histstyles.data.");
  dataSF->importTagsWithPrefix(dataTags,"style.default.");
  dataSF->addObject(TQHistogramUtils::copyHistogram(data), ".histograms+::histogram");

  TQSampleFolder* mcSF = sfTmp->getSampleFolder("mc+");
  TQTaggable mcTags; 
  mcTags.importTagsWithoutPrefix(tags,"histstyles.mc.");
  mcSF->importTagsWithPrefix(mcTags,"style.default.");
  mcSF->addObject(TQHistogramUtils::copyHistogram(mc), ".histograms+::histogram");

  TQHWWPlotter pl(sfTmp);
  pl.resetProcesses();
  pl.addData ("data", TString::Format("title='%s'",data->GetTitle()));
  pl.addBackground("mc", TString::Format("title='%s'",mc->GetTitle()));

  TCanvas * cv = pl.plot("histogram", options);

  delete sfTmp;
  return cv;
}



//__________________________________________________________________________________|___________

bool TQHWWPlotter::plotAndSaveHistogram(TH1 * histo, const TString& saveAs, const TString& options) {
  // quick-access-function to plot one histogram and save it
  TCanvas * cv = TQHWWPlotter::plotHistogram(histo, options);
  if (cv) {
    cv->SaveAs(saveAs.Data());
    delete cv;
    return true;
  } else {
    return false;
  }
  delete cv;
}

//__________________________________________________________________________________|___________

bool TQHWWPlotter::plotAndSaveHistograms(TH1 * data, TH1* mc, const TString& saveAs, const TString& options) {
  // quick-access-function to plot two histograms and save them
  TCanvas * cv = TQHWWPlotter::plotHistograms(data,mc, options);
  if (cv) {
    cv->SaveAs(saveAs.Data());
    delete cv;
    return true;
  } else {
    return false;
  }
  delete cv;
}

//__________________________________________________________________________________|___________


void TQHWWPlotter::setStyleIllinois(TQTaggable& tags){
  // setup the "new" (Illinois) style tags
  double padScaling = 1.;
  double ratioPadScaling = 1.;
  double additionalTopMargin = 0.;
  //@tag: [geometry.sub.height] This argument tag (TQHWWPlotter::setStyleIllinois) controls the height of the sub plot (e.g. ratio plot)
  double ratioPadRatio = tags.getTagDoubleDefault("geometry.sub.height",0.35);
  //@tag:[style.titleOffset] This argument tag (TQHWWPlotter::setStyleIllinois) controls the offset of axis lables. Default: 1.
  double titleOffset = tags.getTagDoubleDefault("style.titleOffset",1);
  //@tag:[style.textSize] 
  double textsize = tags.getTagDoubleDefault("style.textSize",0.05);
  //@tag:[style.showSub] This argument tag controls if a sub plot (e.g. ratio plot) is shown or not. Default: false.
  bool showSub = tags.getTagBoolDefault ("style.showSub",false);
  //@tag:[geometry.legendPadRatio] This argument tag (TQHWWPlotter::setStyleIllinois) controls the width of the legend on plots. Default: 0.25
  double legend_divider_x = 1. - tags.getTagDoubleDefault("geoemetry.legendPadRatio",0.25);

  double geometry_left_margin = 0.20;
 
  tags.setTagDouble("geometry.main.margins.right", 0.05);
  tags.setTagDouble("geometry.main.margins.left", geometry_left_margin);
  tags.setTagDouble("geometry.main.margins.top", textsize * 1.6 * padScaling);

  tags.setTagDouble("geometry.main.xMin",0.);
  tags.setTagDouble("geometry.main.xMax",legend_divider_x);
  tags.setTagDouble("geometry.main.yMax",1.);

  tags.setTagInteger("style.tickx",0);
  tags.setTagInteger("style.ticky",0);

  //tags.setTagDouble("style.legendHeight",0.35 * (showSub ? 1.0 : 1.1));
  tags.setTagDouble("style.legendHeight",0.33 * (showSub ? 1.0 : 1.1));
  tags.setTagDouble("geometry.legend.margins.right", 0.05);
  tags.setTagDouble("geometry.legend.margins.top", textsize * 1.6 * padScaling);
  tags.setTagDouble("geometry.legend.margins.left",0.16);
  tags.setTagDouble("geometry.legend.margins.bottom", 0.);

  tags.setTagInteger("geometry.canvas.height",600);
  if (showSub){
    tags.setTagInteger("geometry.canvas.width",700);
    padScaling = 1. / (1. - ratioPadRatio) / 8. * 6.;
    ratioPadScaling = (1. / ratioPadRatio) / 8. * 6.;
    double legendPadScaling = 1.;//1/legend_divider_x;
    additionalTopMargin = 0.1 * (padScaling - 1);
    tags.setTagDouble("geometry.sub.margins.top", 0.);
    tags.setTagDouble("geometry.sub.margins.bottom",0.16);
    tags.setTagDouble("geometry.sub.margins.left", geometry_left_margin);
    tags.setTagDouble("geometry.sub.margins.right",0.05);

    tags.setTagDouble("geometry.main.additionalTopMargin",additionalTopMargin);
    tags.setTagDouble("geometry.main.scaling",padScaling);
    tags.setTagDouble("geometry.sub.scaling",ratioPadScaling);
    tags.setTagDouble("geometry.xscaling",legendPadScaling);
 
    tags.setTagDouble("geometry.sub.xMin",0.);
    tags.setTagDouble("geometry.sub.yMin",0.);
    tags.setTagDouble("geometry.sub.xMax",legend_divider_x);
    tags.setTagDouble("geometry.sub.yMax",ratioPadRatio);

    tags.setTagDouble("geometry.main.yMin",ratioPadRatio);
    tags.setTagDouble("geometry.main.margins.bottom", 0.);

    tags.setTagBool("style.main.xAxis.showLabels",false);
    tags.setTagBool("style.main.xAxis.showTitle",false);
  } else {
    tags.setTagInteger("geometry.canvas.width",800);
    tags.setTagDouble("geometry.main.margins.bottom", 0.18);
    tags.setTagDouble("geometry.main.yMin",0.);
  }

  tags.setTagDouble("geometry.legend.xMin",legend_divider_x);
  tags.setTagDouble("geometry.legend.yMin", 0.);
  tags.setTagDouble("geometry.legend.xMax",1.);
  tags.setTagDouble("geometry.legend.yMax",1. - (1. - (showSub?ratioPadRatio:0)) *tags.getTagDoubleDefault("geometry.main.margins.top",0.3));
 
  tags.setTagInteger("style.text.font",42);
  tags.setTagDouble("style.textSize",0.05);
	// tags.setTagDouble("style.markerSize",1.2);
  tags.setTagInteger("style.markerType",20); 
  tags.setTagBool("errors.showX", false);
  //@tag: [style.showEventYields] If this argument tag is set to true, integrated event yields for each process are shown in the legend.
  if (tags.getTagBoolDefault ("style.showEventYields",false))
    tags.setTagDouble("legend.textSize",0.080);
    
  tags.setTagBool("style.useLegendPad",true);
  tags.setTagDouble("style.logMin",0.011);
  tags.setTagDouble("legend.margin.right",-0.2);
 
  tags.setTagDouble("geometry.main.xAxis.titleOffset",1.2*titleOffset);
  tags.setTagDouble("geometry.main.xAxis.labelOffset",0.03);
  tags.setTagDouble("geometry.main.xAxis.titleSize",showSub ? 0. : textsize*1.2);
  tags.setTagDouble("geometry.main.xAxis.labelSize",showSub ? 0. : textsize);
  tags.setTagDouble("geometry.main.yAxis.titleOffset",(showSub? 1.5:1.7)*titleOffset);
  tags.setTagDouble("geometry.main.yAxis.labelOffset",0.03);
  tags.setTagDouble("geometry.main.yAxis.titleSize",textsize*1.2);
  tags.setTagDouble("geometry.main.yAxis.labelSize",textsize);

  if(showSub){
    tags.setTagDouble("geometry.sub.xAxis.titleOffset",1.1*titleOffset);
    tags.setTagDouble("geometry.sub.xAxis.labelOffset",0.04);
    tags.setTagDouble("geometry.sub.xAxis.titleSize",textsize*1.2);
    tags.setTagDouble("geometry.sub.xAxis.labelSize",textsize);
    tags.setTagDouble("geometry.sub.yAxis.titleOffset",1.5*titleOffset);
    tags.setTagDouble("geometry.sub.yAxis.labelOffset",0.03);
    tags.setTagDouble("geometry.sub.yAxis.titleSize",textsize*1.2);
    tags.setTagDouble("geometry.sub.yAxis.labelSize",textsize);
  }

  double tickLength = tags.getTagDoubleDefault("style.tickLength",0.02);
  tags.setTagDouble("geometry.main.yAxis.tickLength",-tickLength);
  tags.setTagDouble("geometry.main.xAxis.tickLength",-tickLength);
  tags.setTagDouble("geometry.sub.yAxis.tickLength", -tickLength*0.8); 
  // I found that the y ticks need to be scaled by 0.8 in the ratio plot in order to match the main plot
  // if anybody understands the reason behind this, please add a comment or eMail me: cburgard@cern.ch
  tags.setTagDouble("geometry.sub.xAxis.tickLength",-tickLength);

  tags.setTagInteger("style.main.xAxis.nDiv",50008);
  tags.setTagInteger("style.main.yAxis.nDiv",50004);

  tags.setTagInteger("style.main.lineColor",0);
  tags.setTagInteger("style.main.markerColor",0);

  tags.setTagInteger("style.ratio.mcErrorBand.fillColor",14);
  tags.setTagInteger("style.ratio.mcErrorBand.fillStyle",3254);

  tags.setTagInteger("style.significance.fillColor",kRed);
  tags.setTagInteger("style.significance.fillStyle",3254);
  tags.setTagInteger("style.significance.lineColor",0);
  tags.setTagInteger("style.significance.lineStyle",0);

  tags.setTagInteger("style.main.data.lineWidth",2);
  tags.setTagInteger("style.main.data.lineColor",kBlack);
  // tags.setTagDouble ("style.main.data.markerSize",1.0); 

  tags.setTagInteger("style.main.totalBkg.fillColor",0);
  tags.setTagInteger("style.main.totalBkg.fillStyle",0);
  tags.setTagInteger("style.main.totalBkgError.fillColor",14);
  tags.setTagInteger("style.main.totalBkgError.fillStyle",3254);
 
  tags.setTagInteger("style.sub.xAxis.nDiv",50008);

  tags.setTagInteger("style.nLegendCols",1);

  tags.setTagDouble("legend.xMin",0.0);
  tags.setTagDouble("legend.yMin",0.0);
  tags.setTagDouble("legend.xMax",1.0);
  tags.setTagDouble("legend.yMax",1.0);

  tags.setTagInteger("labels.info.align",11);
  tags.setTagDouble("labels.info.size",0.8);
  tags.setTagDouble("labels.info.xPos",0.);
  tags.setTagDouble("labels.atlas.scale",1.);
  if (showSub) // the following controls the xoffset of "internal" in ATLAS "internal".
    tags.setTagDouble("labels.atlas.xOffset",0.16);
  else
    tags.setTagDouble("labels.atlas.xOffset",0.20);
  tags.setTagDouble("labels.drawATLAS.scale",1.21);
  tags.setTagDouble("style.labels.scale",1.1);
  tags.setTagDouble("style.labels.xOffset",geometry_left_margin+0.015);
  tags.setTagBool("style.legend.textSizeFixed",true);
  tags.setTagDouble("legend.textSize",0.132);
 
  tags.setTagBool("legend.showTotalBkgErrorType",false);
  tags.setTagString("legend.dataDisplayType","ex1y2p");
  tags.setTagBool("style.manualStacking",false);
  tags.setTagBool("style.autoStackLegend",true);

  double yVetoLeft = 0.70;
  double yVetoRight = 0.70;
  //@tag: [style.logScale] If this argument tag is set to true, the y axis is shown in log scale. Default: false. 
  if (tags.getTagBoolDefault ("style.logScale",false )) {
    yVetoLeft -= 0.15;
    yVetoRight -= 0.15;
  }
  tags.setTagDouble("blocks.x.0",0.5); tags.setTagDouble("blocks.y.0",yVetoLeft);
  tags.setTagDouble("blocks.x.1",1.0); tags.setTagDouble("blocks.y.1",yVetoRight);
}

//__________________________________________________________________________________|___________

void TQHWWPlotter::setStyle(TQTaggable& tags){
  //this method is still in use
  // setup the default style tags
  double padScaling = 1.;
  double ratioPadScaling = 1.;
  double additionalTopMargin = 0.;
  //tag doc see setStyleIllinois
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
  } else {
    tags.setTagDouble("geometry.main.margins.top", 0.05);
    tags.setTagDouble("geometry.legend.margins.top", 0.05);
  }
  if((tags.hasTag("style.heatmap") || tags.hasTag("style.migration"))&& tags.getTagIntegerDefault("style.nDim",1) == 2){ //only change the margin if it's an actual 2D plot.
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
    tags.setTagDouble("geometry.legend.margins.bottom",ratioPadRatio/padScaling);
 
    tags.setTagBool("style.main.xAxis.showLabels",false);
    tags.setTagBool("style.main.xAxis.showTitle",false);
  } else {
    tags.setTagDouble("geometry.main.margins.bottom",0.16);
    tags.setTagDouble("geometry.legend.margins.bottom",0.16);
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
	tags.setTagDouble ("style.main.data.markerStyle",20);
	if (tags.hasTag("style.markerSize") && !tags.hasTag("style.main.data.markerSize")) {
		tags.setTagDouble("style.main.data.markerSize",tags.getTagDoubleDefault("style.markerSize", 1.));
	}

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

  //@tag:[style.sub.yAxis.nDiv, style.ratio.nYdiv] This tag controls the number of divisions/ticks of the sub plot. The number of top level ticks is given by the two least significant digits (in decimal notation). The second two least significant digits determine the number of sub divisions (smaller ticks), the thrid least significant set of two digits controls the sub-sub-devisions. When "setStyle" is called, the first tag (...yAxis.nDiv) defaults to the value determined from latter tag (...nYdiv) or 510 (10 top level divisions, 5 sub divisions) if the latter one is not set.
  tags.setTagInteger("style.sub.yAxis.nDiv",tags.getTagIntegerDefault("style.ratio.nYdiv",510));

  tags.setTagDouble("legend.xMin",0.59);
  tags.setTagDouble("legend.xMax",0.90);
  tags.setTagDouble("legend.yMin",0.70);
  tags.setTagDouble("legend.yMax",0.92);

  tags.setTagBool("errors.showX",true);
  tags.setTagDouble("erros.widthX",0.5);

  tags.setTagDouble("blocks.x.0",0.5);
  tags.setTagDouble("blocks.x.1",1.0);
  if(tags.getTagBoolDefault("style.showLabels",true)){
    double yVetoLeft = 0.84;
    TList* labels = tags.getTagList("labels");
    if (labels) yVetoLeft -= (showSub ? 0.08 : 0.09) * (double)labels->GetEntries() * tags.getTagDoubleDefault("geometry.main.scaling",1.);
    delete labels;
    tags.setTagDouble("blocks.y.0",yVetoLeft);
  } else {
    tags.setTagDouble("blocks.y.0",1);
  }
  if(!tags.getTagBoolDefault("style.useLegendPad",false)){
    double yVetoRight = tags.getTagDoubleDefault("legend.yMin",0.5) - tags.getTagDoubleDefault("legend.margin.right",0.05);
    tags.setTagDouble("blocks.y.1",yVetoRight);
  } else {
    tags.setTagDouble("blocks.y.1",1);
  }
}

//__________________________________________________________________________________|___________

bool TQHWWPlotter::includeSystematicsLegacy(TQTaggable& tags,TQTaggable& aliases){
  // legacy function for systematics handling
  // the new version is to be found in the TQPlotter base class
  
  //@tag:[verbose] This argument tag enables verbosity. Default: false.
  bool verbose = tags.getTagBoolDefault("verbose",false);
  if(verbose) VERBOSEclass("collecting uncertainty band data");

  // get the systematic uncertainty histogram
  TH1 * hTotalBkgSys = 0;
  //@tag: [errors.drawSysMC,errors.drawAsymmSysMC] These argument tags enable drawing of (asymmetric) systematic uncertainties for MC. Default: false
  bool sysMcErrors = tags.getTagBoolDefault("errors.drawSysMC",false );
  bool asymmSysMcErrors = tags.getTagBoolDefault("errors.drawAsymmSysMC", false);

  if (sysMcErrors && !asymmSysMcErrors) {
    //@tag: [input.bkg] This argument tag specifies the path of the total background. Default: "bkg"
    TObjArray * histoSystematics = getHistograms(this->fProcesses,"isTotalBkgSys", tags.getTagStringDefault("input.bkg", "bkg"),".systematics/", aliases, tags);
    if (histoSystematics && histoSystematics->GetEntries() > 0){
      hTotalBkgSys = (TH1*)histoSystematics->First();
      this->addObject(hTotalBkgSys,"totalBkgSys");
    }
  } else {
    TObjArray * histoSystematicsAsymm = 0;
    if (asymmSysMcErrors) {
      TObjArray * procSystematicsAsymm = new TObjArray();
      procSystematicsAsymm->SetOwner(true);
      TQTaggableIterator sysItr(this->fProcesses);
      TObjArray* processes = NULL;
      //@tag: [errors.drawAsymmSysList] This argument tag contains a comma seperated list of systematic uncertainties.
      TObjArray* systList = (tags.getTagStringDefault("errors.drawAsymmSysList", "")).Tokenize(",");
      while(!processes && sysItr.hasNext()){
        TQTaggable* process = sysItr.readNext();
        if(!process) continue;
        //@tag: [isTotalBkgSys] This process tag labels the process as the "process" representing the total systematic uncertainties.
        if(!process->getTagBoolDefault("isTotalBkgSys")) continue;
        //@tag: [.path] This process tag contains the path(s) (possibly including path arithmetics) of the corresponding sample folder(s).
        processes = process->getTagStringDefault(".path").Tokenize("+");
      }
      for (int iSys = 0; iSys < systList->GetEntries(); iSys++) {
        TString entry = "";
        for (int iProc = 0; iProc < processes->GetEntries(); iProc++) {
          TString process = TQStringUtils::trim(processes->At(iProc)->GetName());
          TString syst = TQStringUtils::trim(systList->At(iSys)->GetName());
          if (iProc)
            entry += "+" + process + "/" + syst;
          else
            entry = process + "/" + syst;
        }
        TQNamedTaggable* tmp = new TQNamedTaggable();
        tmp->setTagString(".path",entry);
        tmp->setTagBool("isBkgSys",true);
        procSystematicsAsymm->Add(tmp);
 
      }
      histoSystematicsAsymm = getHistograms(processes, "isBkgSys", tags.getTagStringDefault("input.bkg", "bkg"), ".asymmsystematics/", aliases, tags);
      delete procSystematicsAsymm;
      delete processes;
      delete systList;
    }
    this->addObject(histoSystematicsAsymm,"asymmSys"); 
  }

  return true;
}

//__________________________________________________________________________________|___________

bool TQHWWPlotter::setTotalBkgSystematics(const TString& path) {
  // set the total background systematics to be retrieved from the given path
  TQNamedTaggable* totalBkgSys = new TQNamedTaggable("totalBkgSys");
  totalBkgSys->setTagBool("isTotalBkgSys",true);
  totalBkgSys->setTagString(".path",path);
  this->fProcesses->Add(totalBkgSys);
  return true;
}

//__________________________________________________________________________________|___________

TObjArray* TQHWWPlotter::collectHistograms(TQTaggable& tags){
  // use the TQSampleDataReader to retrieve all histograms from the sample folder
  
  //@tag: [style.showUnderflow,style.showOverflow] This argument tag controls if under/overflow bins are shown in the histogram. Default: false.
  bool showUnderflow = tags.getTagBoolDefault ("style.showUnderflow",false);
  bool showOverflow = tags.getTagBoolDefault ("style.showOverflow",false );
  tags.setTagBool("includeOverflow",showOverflow);
  tags.setTagBool("includeUnderflow",showUnderflow);
  //@tag: [input.mh] (HWW legacy) This argument tag defines the higgs-boson mass. Default: 125
  int mh = tags.getTagIntegerDefault ("input.mh", 125);
  bool verbose = tags.getTagBoolDefault("verbose",false );
  //tags.setTagBool("norm",normalize); //this is problematic! this causes every histogram individually to be normalized in the SampleDataReader (which changes the shape of the histogram stack and is very misleading) TODO: remove this line after some checks/time

  // get the histograms
  TQTaggable aliases;
  //@tag: [input.lepch,input.channel] These argument tags do something (TODO)
  aliases.setTagString("lepch",tags.getTagStringDefault("input.lepch","?")); //in case it's not specified, we set a default in this way
  aliases.setTagString("channel",tags.getTagStringDefault("input.channel","?"));
  aliases.setTagString("datachannel",tags.getTagStringDefault("input.datachannel","?"));
  aliases.setTagInteger("mh",mh);
  aliases.importTagsWithoutPrefix(tags,"alias.");//import alias and input tags
  aliases.importTagsWithoutPrefix(tags,"input.");
  
  if(verbose) VERBOSEclass("getting data histograms");
  TObjArray* histosData = getHistograms(this->fProcesses,".isData", tags.getTagStringDefault("input.data", "histogram"), "", aliases, tags);
  if(verbose) VERBOSEclass("getting background histograms");
  TObjArray* histosBkg = getHistograms(this->fProcesses,".isBackground", tags.getTagStringDefault("input.bkg", "histogram"), "", aliases, tags);
  if(verbose) VERBOSEclass("getting signal histograms");
  TObjArray* histosSig = getHistograms(this->fProcesses,".isSignal", tags.getTagStringDefault("input.sig", "histogram"), "", aliases, tags);
 
  TObjArray* histos = new TObjArray();
  
  histos->AddAll(histosData);
  histos->AddAll(histosBkg);
  histos->AddAll(histosSig);
  if(histos->GetEntries() < 1){
    delete histos;
    ERRORclass("no histograms found: "+tags.exportTagsAsString("input.*"));
    return NULL;
  }
  if(!histosData || histosData->GetEntries() < 1){
    if(verbose) VERBOSEclass("no data histograms found, disabling data");
    tags.setTagBool("style.drawData",false);
  } else {
    if(verbose) VERBOSEclass("found %d data histograms",histosData->GetEntries());
  }
  if(histosData) delete histosData;
  if(histosBkg) delete histosBkg;
  if(histosSig) delete histosSig;
  
  TQTH1Iterator itr(histos);
  double maxint = 0;
  while(itr.hasNext()){
    TH1* hist = itr.readNext();
    double integral = hist->Integral();
    maxint = std::max(integral,maxint);
  }
  if(verbose) VERBOSEclass("highest integral of histogram set is %g",maxint);
  if( maxint < tags.getTagDoubleDefault("skipEmptyHistograms",1e-5) && !tags.getTagDoubleDefault("skipEmptyHistograms",1e-5) == 0 ){  //@tag: [skipEmptyHistograms] Skip histograms with integral below given value. This is set to 1e-5 by default!
    delete histos;
    return NULL;
  }

  //////////////////////////////////////////////////////
  // check the consistency
  //////////////////////////////////////////////////////

  if(verbose) VERBOSEclass("checking histogram consistency");
  // expect at least one histogram to be available 
  bool consistent = (histos->GetEntries() > 0);
  // check consistency and create master histogram 
  TH1* hMaster = NULL;
  consistent = checkConsistency(hMaster, histos) && consistent;
 
  // stop if there is no valid histogram or histograms are invalid
  if (!consistent){
    if(verbose) VERBOSEclass("consistency check failed");
    delete histos;
    return NULL;
  }
  hMaster->Reset();
  this->addObject(histos,"histos");

  tags.setTagInteger("style.nDim",TQHistogramUtils::getDimension(hMaster));

  //////////////////////////////////////////////////////
  // initialize background histogram
  //////////////////////////////////////////////////////
 
  // prepare the total background histogram
  if(verbose) VERBOSEclass("preparing total background histogram");
  TH1* hTotalBkg = TQHistogramUtils::copyHistogram(hMaster,"totalBkg");
  hTotalBkg->SetTitle(tags.getTagStringDefault ("labels.totalBkg", "SM"));
  TH1* hTotalSig = TQHistogramUtils::copyHistogram(hMaster,"totalSig");
  hTotalSig->SetTitle(tags.getTagStringDefault ("labels.totalSig", "H#rightarrowWW#rightarrowl#nul#nu"));
 
  // reset the histogram, because it is a clone of any of the subprocess
  // histograms and we are only interested in the the binning. Reset() resets
  // bin content and errors.
  hTotalBkg->Reset();
  hTotalSig->Reset();

  // sum all background contributions by hand and (in the same step) check
  // whether there is any background process to be plotted. If so, the SM
  // background histogram will be shown (including the entry in the legend).
  bool hasTotalBkg = false;
  TString stackFilter = tags.getTagStringDefault ("style.stackFilter","" );
  TQTaggableIterator itr_bkg(this->fProcesses);
  while(itr_bkg.hasNext()){
    TQNamedTaggable* process = itr_bkg.readNext();
    if(!process) continue;
    if(!process->getTagBoolDefault(".isBackground",false)) continue;
    TH1 * h = this->getObject<TH1>(this->makeHistogramIdentifier(process));
    if(!h) continue;
    if (!stackFilter.IsNull()) {
      TString title = h->GetTitle();
      if (!stackFilter.Contains(title))
        continue;
    }
    hTotalBkg->Add(h);
    hasTotalBkg = true;
  }

  if(hTotalBkg){
    TH1* hTotalBkgOnly = TQHistogramUtils::copyHistogram(hTotalBkg,"totalBkgOnly");
    hTotalBkgOnly->SetDirectory(NULL);
    this->addObject(hTotalBkgOnly);
  }
  
  // If stacking signal, add it to total background so that it shows up as a
  // part of the ratio
  bool hasTotalSig = false;
  bool stackSignal = tags.getTagBoolDefault ("style.stackSignal",false);
  if (hTotalBkg) {
    TQTaggableIterator itr_sig(this->fProcesses);
    while(itr_sig.hasNext()){
      TQNamedTaggable* process = itr_sig.readNext();
      if(!process) continue;
      if(!process->getTagBoolDefault(".isSignal",false)) continue;
      TH1 * h = this->getObject<TH1>(this->makeHistogramIdentifier(process));
      if(!h) continue;
      if(process->getTagBoolDefault("stack",stackSignal)){
        hTotalBkg->Add(h);
        hasTotalSig = true;
      } else {
        hTotalSig->Add(h);
        hasTotalSig = true;
      }
    }
  }

  if(hTotalBkg && !hasTotalBkg){
    this->removeObject("totalBkg",true); 
  } 

  if(hTotalSig && !hasTotalSig){
    this->removeObject("totalSig",true); 
  } 

  //////////////////////////////////////////////////////
  // systematics handling
  //////////////////////////////////////////////////////

  if(tags.getTagBoolDefault("errors.drawSysMC",false ) || tags.getTagBoolDefault("errors.drawAsymmSysMC", false)){
    // the "old" way
    this->includeSystematicsLegacy(tags,aliases);
  }
  if(tags.hasTag("errors.showSys")){
    // the "new" way
    this->sysOk = this->includeSystematics(tags);
  }

  //////////////////////////////////////////////////////
  // rebinning options
  //////////////////////////////////////////////////////

  if(this->getNProcesses(".isData") == 0 || !hasTotalBkg) tags.setTagBool("style.showSub",false);

  return histos;
}

//__________________________________________________________________________________|___________

void TQHWWPlotter::makeLegend(TQTaggable& tags, TObjArray* histos){
  // create a legend including the given list of histograms
  // @tags:style.showEventYields: show event yields (integral counts) in the legend
  // @tags:style.showEventYields.useUnderOverflow: include underflow and overflow in the event yields displayed in the legend (default:true)
  // @tags:style.nLegendCols: number of columns to be shown in legend
  // @tags:style.legendHeight: scaling factor for height of the legend
  bool showEventYields = tags.getTagBoolDefault ("style.showEventYields",false);
  int nLegendCols = tags.getTagIntegerDefault ("style.nLegendCols",showEventYields ? 1 : 2);
  bool showTotalBkg = tags.getTagBoolDefault ("style.showTotalBkg",true);
  double legendHeight = tags.getTagDoubleDefault ("style.legendHeight",1. );
  bool drawData = tags.getTagBoolDefault ("style.drawData",true);
  bool verbose = tags.getTagBoolDefault("verbose",false);

  // the nominal coordinates of the legend

  bool legpad = tags.getTagBoolDefault("style.useLegendPad",false);
  
  // calculate the number of entries
  int nEntries = 0;
  //@tag: style.showMissing: show empty legend entries where histogram is empty or could not be retrieved (default:true)
  bool showMissing = tags.getTagBoolDefault ("style.showMissing",true );
  
  nEntries += (showMissing ? histos->GetEntriesFast() : histos->GetEntries());
  TH1* hTotalBkg = this->getObject<TH1>("totalBkg");
  if (showTotalBkg && (hTotalBkg || (showMissing && this->getNProcesses(".isBackground") > 0))) nEntries++;
  
  // calculate the height of the legend
  int nLegendRows = (int)nEntries / nLegendCols + ((nEntries % nLegendCols) > 0 ? 1 : 0);
  
  TLegend* legend = NULL;
  if(legpad){
    //@tags:style.useLegendPad: put the legend on a separate pad on the side of the plot
    if(verbose) VERBOSEclass("creating legend with unity coordinates");
    legend = new TLegend(tags.getTagDoubleDefault("geometry.legend.margins.left",0),
                         tags.getTagDoubleDefault("geometry.legend.margins.bottom",0),
                         1.-tags.getTagDoubleDefault("geometry.legend.margins.right",0),
                         1.-tags.getTagDoubleDefault("geometry.legend.margins.top",0));
    legend->SetFillColor(0);
    legend->SetFillStyle(0);
  } else {
    // if we plot the ratio, the canvas has to be divided which results in a
    // scaling of the legend. To avoid this, we have to rescale the legend's
    // position
  // @tags:[geometry.legend.xMin,geometry.legend.xMax,geometry.legend.yMin,geometry.legend.yMax]: control the geometry of the legend in relative coordinates
    double x1 = tags.getTagDoubleDefault("geometry.legend.xMin",0.59);
    double y1 = tags.getTagDoubleDefault("geometry.legend.yMin",0.70) - tags.getTagDoubleDefault("geometry.main.additionalTopMargin",0.);
    double x2 = tags.getTagDoubleDefault("geometry.legend.xMax",0.90);
    double y2 = tags.getTagDoubleDefault("geometry.legend.yMax",0.92) - tags.getTagDoubleDefault("geometry.main.additionalTopMargin",0.);
    
    y1 = y2 - (y2 - y1) * tags.getTagDoubleDefault("geometry.main.scaling",1.);
    legendHeight *= (y2 - y1) * (double)nLegendRows / tags.getTagDoubleDefault("geometry.legend.nRows",5.);
    
    // set the height of the legend
    y1 = y2 - legendHeight;
    // create the legend and set some attributes
    double tmpx1 = x1; 
    double tmpx2 = x2; 
    double tmpy1 = y1; 
    double tmpy2 = y2; 
    if(verbose) VERBOSEclass("creating legend with coordinates %g/%g - %g/%g",tmpx1,tmpy1,tmpx2,tmpy2);
    legend = new TLegend(tmpx1, tmpy1, tmpx2, tmpy2);
    //@tags:[style.legend.fillColor,style.legend.fillStyle]: control color and style of the legend with TLegend::SetFillColor and TLegend::SetFillStyle. defaults are 0.
    legend->SetFillColor(tags.getTagIntegerDefault("style.legend.fillColor",0));
    legend->SetFillStyle(tags.getTagIntegerDefault("style.legend.fillStyle",0));
  }
  this->addObject(legend,"legend");
  legend->SetBorderSize(0);
  legend->SetNColumns(nLegendCols);

  //@tags: style.legend.textSize: control the font size (floating point number, default is 0.032)
  //@tags: style.legend.textSizeFixed: boolean to control whether the text size will be interpreted relative to canvas size (default) or absolute
  double textsize = tags.getTagDoubleDefault("style.legend.textSize",0.032);
  if (tags.getTagBoolDefault ("style.legend.textSizeFixed", false))
    legend->SetTextSize(textsize);
  else
    legend->SetTextSize(textsize * tags.getTagDoubleDefault("geometry.main.scaling",1.));
  
  // show the error band on SM MC backgrounds in the legend. We have to use a
  // dummy histogram for the legend to get the correct appearance
  bool statMcErrors = tags.getTagBoolDefault("errors.drawStatMC",true );
  bool sysMcErrors = tags.getTagBoolDefault("errors.drawSysMC",false ) || tags.hasTag("errors.showSys");
  sysMcErrors = sysMcErrors && sysOk;
 
  TH1* hTotalBkgError = TQHistogramUtils::copyHistogram(hTotalBkg,"totalBkgError");
  if(hTotalBkgError){
    hTotalBkgError->Reset();
    hTotalBkgError->SetTitle("total background error (legend dummy)");
    if (statMcErrors || sysMcErrors) {
      this->applyStyle (tags,hTotalBkgError,"main.totalBkgError");
    }
    if(verbose){ DEBUGclass("totalBkgError style: %s",TQHistogramUtils::getDetailsAsString(hTotalBkgError,5).Data()); }
  }
 
  // create the SM legend entry label depending on which error is shown as a
  // band around it
  TString totalBkgLabel = tags.getTagStringDefault ("labels.totalBkg", "SM");
  TString legendTotalBkgLabel = tags.getTagStringDefault ("labels.totalBkg", "SM");
  legendTotalBkgLabel = " " + tags.getTagStringDefault ("labels.legendTotalBkg", legendTotalBkgLabel); // overwrite if you explicitly want it different in legend
  if(tags.getTagBoolDefault("legend.showTotalBkgErrorType",true)){
    if (statMcErrors && sysMcErrors)
      legendTotalBkgLabel.Append(" (sys #oplus stat)");
    else if (sysMcErrors)
      legendTotalBkgLabel.Append(" (sys)");
    else if (statMcErrors)
      legendTotalBkgLabel.Append(" (stat)");
  }
  if (tags.getTagBoolDefault("isCompPlot",false))
    addAllHistogramsToLegend(tags,legend, ".isBackground", tags.getTagStringDefault("legend.dataDisplayType",".legendOptions='lep'"));

  if(!tags.getTagBoolDefault("style.unsorted",false)){
    // add the data processes
    if (drawData) {
      addAllHistogramsToLegend(tags,legend, ".isData", tags.getTagStringDefault("legend.dataDisplayType",".legendOptions='lep'"));
    }
    if (tags.getTagBoolDefault("useToyData",false))
      this->addHistogramToLegend(tags,legend,this->getObject<TH1>("toyData"), tags.getTagStringDefault("legend.dataDisplayType",".legendOptions='lep'"));
  }

  // add the total background histogram to the legend
  if (showTotalBkg) {
    if (hTotalBkgError)
      legend->AddEntry(hTotalBkgError, legendTotalBkgLabel, "lf");
    else if (showMissing && this->getNProcesses(".isBackground") > 0)
      legend->AddEntry((TObject*)NULL,"","");
  }
 
  bool stackSignal = tags.getTagBoolDefault ("style.stackSignal",false);
  bool autoStackSignal = tags.getTagBoolDefault ("style.autoStackSignal",false);
  bool listSignalFirst = tags.getTagBoolDefault ("style.listSignalFirst",false);
  bool showSignal = tags.getTagBoolDefault ("style.showSignalInLegend",true);
  if(tags.getTagBoolDefault("style.unsorted",false)){
    //@tag:style.unsorted: do not apply any sorting whatsoever, list all processes in the order in which they were added
    addAllHistogramsToLegend(tags,legend, "");
  } else {
    if (!tags.getTagBoolDefault ("style.manualStacking", false)) {
      if(!tags.getTagBoolDefault("style.autoStackLegend",false)){
	// add the background and signal processes
	if(verbose) VERBOSEclass("generating legend in default mode");
	if(listSignalFirst){
	  if (showSignal)
	    addAllHistogramsToLegend(tags,legend, ".isSignal");
	  if (!tags.getTagBoolDefault("isCompPlot",false))
	    addAllHistogramsToLegend(tags,legend, ".isBackground");
	} else {
	  if (!tags.getTagBoolDefault("isCompPlot",false))
	    addAllHistogramsToLegend(tags,legend, ".isBackground");
	  if (showSignal)
	    addAllHistogramsToLegend(tags,legend, ".isSignal");
	}
      } else {
	THStack* stack = this->getObject<THStack>("stack");
	if(!stack){
	  if(verbose){
	    VERBOSEclass("cannot generate legend in auto-stack mode - no stack!");
	  } 
	  return;
	} else {
	  if(verbose) VERBOSEclass("generating legend in auto-stack mode");
	  if (!stackSignal && listSignalFirst && showSignal && !autoStackSignal) {
	    addAllHistogramsToLegend(tags,legend, ".isSignal");
	  }
	  TQTH1Iterator itr(stack->GetHists()->MakeIterator(kIterBackward),true);
	  while(itr.hasNext()){
	    TH1* hist = itr.readNext();
	    addHistogramToLegend(tags,legend,hist);
	  }
	  if (!stackSignal && !listSignalFirst && showSignal  && !autoStackSignal) {
	    addAllHistogramsToLegend(tags,legend, ".isSignal");
	  }
	}
      }
    } else {
      if (stackSignal) {
	if(verbose) VERBOSEclass("generating legend in manual stacking mode - stackSignal=true");
	if (listSignalFirst) {
	  if (showSignal)
	    addAllHistogramsToLegend(tags,legend, ".isSignal","",true);
	  addAllHistogramsToLegend(tags,legend, ".isBackground","",true);
	} else {
	  addAllHistogramsToLegend(tags,legend, ".isBackground","",true);
	  addAllHistogramsToLegend(tags,legend, ".isSignal","",true);
	}
      } else {
	if(verbose) VERBOSEclass("generating legend in manual stacking mode - stackSignal=false");
	if (listSignalFirst) {
        if (showSignal)
          addAllHistogramsToLegend(tags,legend, ".isSignal");
        addAllHistogramsToLegend(tags,legend, ".isBackground","",true);
	} else {
	  addAllHistogramsToLegend(tags,legend, ".isBackground","",true);
	  if (showSignal)
	    addAllHistogramsToLegend(tags,legend, ".isSignal");
	}
      }
    }
  }
}

//__________________________________________________________________________________|___________

void TQHWWPlotter::stackHistograms(TQTaggable& tags, const TString& stackname){
  // create the histogram stack
  bool normalize = tags.getTagBoolDefault("normalize",false );
  //@tag:normalizeWithoutOverUnderflow: disable using underflow and overflow for normalization purposes
  bool normalizeWithoutOverUnderflow = !tags.getTagBoolDefault("normalizeWithoutOverUnderflow",false );
  TString stackFilter = tags.getTagStringDefault ("style.stackFilter","");
  TH1* hTotalBkg = this->getObject<TH1>("totalBkg");

  // scale to normalize total background
  double totalBkgScale = TQHistogramUtils::getIntegral(hTotalBkg, normalizeWithoutOverUnderflow);
  if (hTotalBkg && normalize) {
    if (totalBkgScale != 0.)
      hTotalBkg->Scale(1. / totalBkgScale);
  }
  
  // the list of histograms to be stacked
  TObjArray * histStackList = new TObjArray();
  TObjArray * processStackList = new TObjArray();
  TQTaggableIterator itr(this->fProcesses);
  while(itr.hasNext()){
    TQNamedTaggable* process = itr.readNext();
    if(!process) continue;
    if(    (process->getTagBoolDefault(".isBackground",false) && process->getTagBoolDefault("stack", true))   // background
        || (process->getTagBoolDefault(".isSignal",false) &&  tags.getTagBoolDefault ("style.autoStackSignal",false)) // signal
           ){
      TH1 * h = this->getObject<TH1>(this->makeHistogramIdentifier(process));
      if (!h) continue;
      if (!stackFilter.IsNull()) {
        TString title = h->GetTitle();
        if (!stackFilter.Contains(title))
          continue;
      }
      if(totalBkgScale > 0 && normalize){
        h->Scale(1. / totalBkgScale);
      }
      histStackList->Add(h);
      processStackList->Add(process);
    } else if (totalBkgScale > 0 && normalize) {
      //also normalize non-stacked histograms such that their integral matches the normalized stack ("totalBackground")
      TH1 * h = this->getObject<TH1>(this->makeHistogramIdentifier(process));
      if (TQHistogramUtils::getIntegral(h, normalizeWithoutOverUnderflow) > 0) h->Scale(1/TQHistogramUtils::getIntegral(h, normalizeWithoutOverUnderflow));
    }
  }
 
  // sort the histograms to be stacked by ascending integral (sum of weights)
  if (!tags.getTagBoolDefault("style.manualStacking",false) && tags.getTagBoolDefault("style.autoStack",tags.getTagBoolDefault("style.logScale",false))){
    for (int iHist = 0; iHist < histStackList->GetEntries(); iHist++) {
      int iHistMin = iHist;
      double intMin = ((TH1*)histStackList->At(iHistMin))->GetSumOfWeights();
      for (int iHist2 = iHist + 1; iHist2 < histStackList->GetEntries(); iHist2++) {
        double intMin2 = ((TH1*)histStackList->At(iHist2))->GetSumOfWeights();
        if (intMin2 < intMin) {
          iHistMin = iHist2;
          intMin = intMin2;
        }
      }
      if (iHistMin != iHist) {
        TH1 * temp = (TH1*)(*histStackList)[iHist];
        (*histStackList)[iHist] = (*histStackList)[iHistMin];
        (*histStackList)[iHistMin] = temp;
        TQNamedTaggable * temptag = (TQNamedTaggable*)(*processStackList)[iHist];
        (*processStackList)[iHist] = (*processStackList)[iHistMin];
        (*processStackList)[iHistMin] = temptag;
      }
    }
  }
 
  // create the stack
  THStack * stack = new THStack(stackname, tags.getTagBoolDefault("style.stackSignal",false) ? "Signal+Background Stack" : "Background Stack");
 
  // add the histograms to the stack (following the order in the histStackList)
  if (tags.getTagBoolDefault ("style.reverseStacking",false )) {
    for (int iHist = histStackList->GetEntries(); iHist >= 0 ; iHist--){
      TH1* h = dynamic_cast<TH1*>(histStackList->At(iHist));
      stack->Add(h);
    }
  } else {
    for (int iHist = 0; iHist < histStackList->GetEntries(); iHist++){
      TH1* h = dynamic_cast<TH1*>(histStackList->At(iHist));
      stack->Add(h);
    }
  }
 
  // the histStackList was only a vehicle for an easy implementation of the loop
  delete histStackList;
  delete processStackList;
 
  TQTaggableIterator sitr(this->fProcesses);
  bool stackSignal = tags.getTagBoolDefault ("style.stackSignal",false);
  if (tags.getTagBoolDefault ("style.autoStackSignal",false)) stackSignal = false;
  if(hTotalBkg){
    while(sitr.hasNext()){
      TQNamedTaggable* process = sitr.readNext();
      if(!process) continue;
      if(!process->getTagBoolDefault(".isSignal",false)) continue;
      TH1 * h = this->getObject<TH1>(this->makeHistogramIdentifier(process));
      if(!h) continue;
      if(totalBkgScale > 0 && process->getTagBoolDefault("normalizeToTotalBkg",false) && !normalize){ //don't scale again if everything was already normalized to unity (see above)
        //@tag:normalizeToTotalBkg: process tag to normalize individual signal contributions to the total background
        h->Scale(totalBkgScale / TQHistogramUtils::getIntegral(h));
      }
      if (process->getTagBoolDefault("stack", stackSignal)){
        stack->Add(h);
      }
    }
  }
  
  this->addObject(stack,stackname);
}

//__________________________________________________________________________________|___________

void TQHWWPlotter::drawHeatmap(TQTaggable& tags){
  // draw a heatmap 
  // this is a TH2 plotting mode
  TString heatmap = "totalBkg";
  TString overlay = "totalSig";
  //@tag:[style.heatmap] If this tag is set a heatmap ('colz') style 2D histogram is drawn for the histogram (read: process, e.g., "hist_Zjets") instead of contour ones
  if(!tags.getTagString("style.heatmap",heatmap)) return;
  TPad* pad = this->getPad("main");
  pad->cd();
  //@tag:[style.drawGrid] Draws a grid on heatmap plots if set to true. Default: false.
  if (tags.getTagBoolDefault("style.drawGrid",false)) pad->SetGrid(1,1);
  TH2* hMaster = this->getObject<TH2>("master");
  TH2* hHeatmap = this->getObject<TH2>(heatmap);
  TH2* hOverlay = NULL;
  bool verbose = tags.getTagBoolDefault("verbose",false);
  bool doOverlay = tags.getTagString("style.heatmapoverlay",overlay);
  
  bool logScale = tags.getTagBoolDefault("style.logScale",false);
  if (logScale) pad->SetLogz();
    
  if(doOverlay){
    if(verbose) VERBOSEclass("retrieving overlay histogram by name '%s'",overlay.Data());
    hOverlay = this->getObject<TH2>(overlay);
  }
 
  if(verbose) VERBOSEclass("drawing master histogram");
  if(hOverlay){
    hMaster->SetMaximum(hOverlay->GetMaximum());
    hMaster->SetMinimum(hOverlay->GetMinimum());
  }

  hMaster->Draw("HIST");

  if(hHeatmap){
    if(verbose) VERBOSEclass("drawing histogram '%s' as heatmap",heatmap.Data());
    hHeatmap->Draw("COLZSAME");
  } else {
    if(verbose){
      VERBOSEclass("cannot draw '%s' as heatmap - object not found",heatmap.Data());
      this->printObjects();
    }
    return;
  }
  
  if(hOverlay){
    if(verbose) VERBOSEclass("drawing histogram '%s' as heatmap overlay",overlay.Data());
    hOverlay->Draw("BOXSAME");
  } else if(doOverlay){
    if(verbose){
      VERBOSEclass("cannot draw '%s' as heatmap overlay - object not found",overlay.Data());
      this->printObjects();
    }
  }

  return;
}

//__________________________________________________________________________________|___________

void TQHWWPlotter::drawContours(TQTaggable& tags){
  // draw a contour plot
  // this is a TH2 plotting mode
  TPad* pad = this->getPad("main");
  TH2* hMaster = this->getObject<TH2>("master");
  TObjArray* histos = this->getObject<TObjArray>("histos");
  bool verbose = tags.getTagBoolDefault("verbose",false);

  // select the z-values of the contours to be drawn
  //@tag:[style.logMin,style.logMinRel] These argument tags determine the minimum value contour plots in logarithmic scale. Default of "logMin" is 1. If the latter one is set, the minimum of the scale is taken as the product of the "logMinRel" tag's value and the highest bin content of the underlying histogram.
  double logMin = tags.getTagDoubleDefault("style.logMin",1.);
  bool logScale = tags.getTagIntegerDefault("style.logScale",false);
  double max = TQHistogramUtils::getMax(histos,false);
  //if the user has set a relative (to the maximum value) lower boundary for log scaled histograms use that one
  if (logScale && tags.hasTagDouble("style.logMinRel") ) logMin = max*tags.getTagDoubleDefault("style.logMinRel",42.); //the default value should never do anything, and if it does we want it to be obvious that something is wrong
  double min = logScale ? std::max(logMin,TQHistogramUtils::getMin(histos,false)) : TQHistogramUtils::getMin(histos,false);
  
  size_t nContours = tags.getTagIntegerDefault("style.nContours",6);
  double step = logScale ? (log(max) - log(min))/(nContours+1) : (max-min)/(nContours+1);
  std::vector<double> contours;
  for(size_t i=0; i<nContours; i++){
    double z_orig = logScale ? min*exp((i+1)*step) : min+(i+1)*step;
    double z = TQUtils::roundAuto(z_orig,1);
    contours.push_back(z);
  }
 
  // create the contour graphs
  TObjArray* contourGraphs = new TObjArray();
  std::vector<double> contourVals;
  //@tag:[style.minContourArea] This argument tag determines up to what size contours are omitted. Default is three time the minimum bin area of the master histogram. Removing this limit would create PDFs easily reaching ~100MByte!
  double minContourArea = tags.getTagDoubleDefault("style.minContourArea",3*TQHistogramUtils::getMinBinArea(hMaster));
  bool batch = gROOT->IsBatch();
  gROOT->SetBatch(true);
  //@tag:[style.doContourLevelsPerHistogram] If this argument tag is set to true countour levels (in contour plots) are determined for each process separately. Default: false.
  bool contourLevelsPerHistogram = tags.getTagBoolDefault("style.doContourLevelsPerHistogram",false);
  TQIterator histItr(histos);
  while(histItr.hasNext()){
    // create a temporary canvas and draw the histogram to create the contours
    TCanvas* tmp = new TCanvas("tmp","tmp");
    TH2* hist = dynamic_cast<TH2*>(histItr.readNext());
    if(tags.getTagBoolDefault("style.smooth",false)){
      hist->Smooth(1,"k5a");
    }
    if(!hist) continue;
    if (contourLevelsPerHistogram) {
      contours.clear();
      max = TQHistogramUtils::getMax(hist,false,false);
      //if the user has set a relative (to the maximum value) lower boundary for log scaled histograms use that one
      if (logScale && tags.hasTagDouble("style.logMinRel") ) logMin = max*tags.getTagDoubleDefault("style.logMinRel",42.); //the default value should never do anything, and if we want it to be obvious that something is wrong
      min = logScale ? std::max(logMin,TQHistogramUtils::getMin(hist,false,false)) : TQHistogramUtils::getMin(hist,false,false);
      //@tag:[style.nContours] This argument tag sets the number of contour levels shown in contour plots. Default is 6 unless "style.doContourLevelsPerHistogram" is set to true, in which case default is 2.
      size_t nContours = tags.getTagIntegerDefault("style.nContours",2);
      double step = logScale ? (log(max) - log(min))/(nContours+1) : (max-min)/(nContours+1);
      for(size_t i=0; i<nContours; i++){
        double z_orig = logScale ? min*exp((i+1)*step) : min+(i+1)*step;
        double z = TQUtils::roundAuto(z_orig,1);
        contours.push_back(z);
      }
    }
    hist->SetContour(contours.size(), &contours[0]);
    if(verbose) VERBOSEclass("drawing contours for %s",hist->GetName());
    hist->Draw("CONT Z LIST");
    // Needed to force the plotting and retrieve the contours in TGraphs
    tmp->Update(); 
    // retrieve the contours
    TQIterator contItr2(dynamic_cast<TObjArray*>(gROOT->GetListOfSpecials()->FindObject("contours")));
    while(contItr2.hasNext()){
      // contours by level
      TList* contLevel = dynamic_cast<TList*>(contItr2.readNext());
      int idx = contItr2.getLastIndex();
      double z0 = contours[idx];
      if(verbose) VERBOSEclass("\tretrieving %d contours for level %f",contLevel->GetEntries(),z0);
      int nGraphs = 0;
      std::vector<double> contourAreas;
      std::vector<double> contourIndices;
      TQIterator contItr3(contLevel);
      while(contItr3.hasNext()){
        // individual graphs per contour level
        TGraph* curv = dynamic_cast<TGraph*>(contItr3.readNext());
        // veto against non-existant contours
        if(!curv) continue;
        double area = fabs(TQHistogramUtils::getContourArea(curv));
        double triangle = 0.5*pow(TQHistogramUtils::getContourJump(curv),2);
        double val = std::max(area,triangle);
        if(verbose) VERBOSEclass("\t\tcontour %d has area=%f and triangle jump=%f -- contour value is %f",nGraphs,area,triangle,val);
        contourAreas.push_back(val);
        contourIndices.push_back(contItr3.getLastIndex());
        nGraphs++;
      }
      if(verbose) VERBOSEclass("identified %i non-vanishing contours",contourAreas.size());
      nGraphs = 0;
      int nContoursMax = tags.getTagIntegerDefault("style.contourLimit",7);
      while(nGraphs < nContoursMax){
        size_t index = 0;
        double max = 0;
        for(size_t i=0; i<contourAreas.size(); i++){
          if(contourAreas[i] > max){
            max = contourAreas[i];
            index = contourIndices[i];
            contourAreas[i] = 0;
          }
        }
        // veto against micro-blob contours
        TGraph* curv = dynamic_cast<TGraph*>(contLevel->At(index));
        if(max < minContourArea && nGraphs > 0) {
          DEBUGclass("removing micro-blob");
          break;
        }
        
        if(max <= 0) break;
        // individual graphs per contour level
        // veto against non-existant contours
        if(!curv) continue;
        // create clones of the graphs to avoid deletions
        TGraph* gc = dynamic_cast<TGraph*>(curv->Clone());
        if(!gc) continue;
        // apply the styles to the graphs
        int color = hist->GetFillColor();
        int style = 1;
        if((color == kWhite) || (color == 0)){
          color = hist->GetLineColor();
          style = TQStringUtils::equal("hist_data",hist->GetName()) ? 3 : 7;
        } 
        gc->SetLineColor(color);
        gc->SetLineStyle(style);
        if(tags.getTagBoolDefault("style.contourLines.shade",false)){
          gc->SetFillStyle(3004+ (histItr.getLastIndex() % 4));
          gc->SetLineWidth(-100
                           *TQUtils::sgn(tags.getTagIntegerDefault("style.contourLines.shadeDirection",1)) 
                           *tags.getTagIntegerDefault("style.contourLines.shadeWidth",3) 
                           + tags.getTagIntegerDefault("style.contourLines.width",1));
        } else {
          gc->SetLineWidth(tags.getTagIntegerDefault("style.contourLines.width",1));
        }
        gc->SetTitle(TString::Format("%s: contour #%d to level %g",hist->GetTitle(),(int)nGraphs,z0));
        this->addObject(gc,TString::Format("contour_%d_%s_%g_%d",contourGraphs->GetEntries(),hist->GetName(),z0,(int)nGraphs));
        contourGraphs->Add(gc);
        contourVals.push_back(z0);
        nGraphs++;
      }
      if(verbose) VERBOSEclass("\tretrieved %d (dismissed all others)",nGraphs);
    }
    // delete the temporary canvas
    delete tmp;
  }
  this->addObject(contourGraphs,"contours");
  if(!batch) gROOT->SetBatch(false);

  // switch to the correct pad
  pad->cd();
  hMaster->Draw("hist");

  // prepare the TLatex object for drawing the labels
  TLatex l;
  bool autoColor = tags.getTagBoolDefault("style.contourLabels.autocolor",false);
  int color = tags.getTagIntegerDefault("style.contourLabels.color",kBlack);
  double size = tags.getTagDoubleDefault("style.contourLabels.size",0.03);
  l.SetTextSize(size); 
  l.SetNDC(true);
  if(!autoColor) l.SetTextColor(color);

  std::vector<double> labelSpotsX;
  std::vector<double> labelSpotsY;
  TQIterator itrGraphs(contourGraphs);
  while(itrGraphs.hasNext()){
    // iterate over the contour graphs
    TGraph* gc = dynamic_cast<TGraph*>(itrGraphs.readNext());
    if(!gc) continue;
    // actually draw the contour graph
    gc->Draw("C");
    // choose a "central" point of the graph
    // to retrieve coordinates for the label 
    int index = 0.5*gc->GetN();
    int indexStep = 1;
    double x0, y0, z0;
    double xNDC, yNDC;
    // create the label text
    z0 = contourVals[itrGraphs.getLastIndex()];
    TString val = TString::Format("%g",z0);
    double minDistX = 0.5*size*TQStringUtils::getWidth(val);
    double minDistY = size;
    // find a location to draw the label
    bool acceptPosition = false;
    if(tags.getTagBoolDefault("style.contourLabels.avoidCollisions",true)){
      while(index > 0 && index < gc->GetN()){
        acceptPosition = true;
        gc->GetPoint(index, x0, y0);
        xNDC = TQUtils::convertXtoNDC(x0);
        yNDC = TQUtils::convertYtoNDC(y0);
        for(size_t i=0; i<labelSpotsX.size(); i++){
          double dx = fabs(xNDC - labelSpotsX[i]);
          double dy = fabs(yNDC - labelSpotsY[i]);
          if((dx < minDistX) && (dy < minDistY)) acceptPosition = false;
        }
        if(acceptPosition) break;
        index += indexStep;
        indexStep = -TQUtils::sgn(indexStep)*(abs(indexStep)+1);
      }
    }
    if(!acceptPosition){
      if(verbose) VERBOSEclass("did not find any suitable label position, using default");
      gc->GetPoint((int)(0.5*gc->GetN()), x0, y0);
      xNDC = TQUtils::convertXtoNDC(x0);
      yNDC = TQUtils::convertYtoNDC(y0);
    }
    labelSpotsX.push_back(xNDC);
    labelSpotsY.push_back(yNDC);
    // choose a color
    if(autoColor) l.SetTextColor(gc->GetLineColor());
    // draw the label
    l.DrawLatex(xNDC,yNDC,val);
    if(verbose) VERBOSEclass("drawing label '%s' at (%f,%f) == (%f,%f) with minDist=(%f,%f)",val.Data(),x0,y0,xNDC,yNDC,minDistX,minDistY);
  }
}

//__________________________________________________________________________________|___________

void TQHWWPlotter::drawMigration(TQTaggable& tags){
  // draw a migration plot
  // this is a TH2 plotting mode
  TString migration = "ggf";
  //@tag:[style.migration] If this tag is set a migration ('col4z') style 2D histogram is drawn for the histogram (read: process, e.g., "hist_Zjets") instead of contour or heatmap ones
  std::cout<< "in migration function" << std::endl;
  if(!tags.getTagString("style.migration",migration)) return;
  std::cout<< "migration is " << migration << std::endl; 
  //std::cout<< "histogram identifier " << this->makeHistogramIdentifier(migration) << std::endl;
  TPad* pad = this->getPad("main");
  TH2* hMaster = this->getObject<TH2>("master");
  TQTaggableIterator itr_sig(fProcesses);
  TQNamedTaggable* process_mig = NULL;
  while(itr_sig.hasNext()){
    TQNamedTaggable* process = itr_sig.readNext();
    if(!process) continue;
    if(process->getTagBoolDefault(".isData",false)) continue;
    if(process->getName() == migration)
      process_mig = process;
  }
  TH2* hMigration = this->getObject<TH2>(this->makeHistogramIdentifier(process_mig));
  bool verbose = tags.getTagBoolDefault("verbose",false);
  bool logScale = tags.getTagBoolDefault("style.logScale",false);
  if (logScale) pad->SetLogz();

  hMaster->Draw("HIST");

  if(hMigration){
    std::cout << "drawing as migration: " << migration.Data() << std::endl;
    if(verbose) VERBOSEclass("drawing histogram '%s' as migration",migration.Data());
    for(int i=0; i <= hMigration->GetNbinsY(); i++) {
      double integral = hMigration->Integral(0,hMigration->GetNbinsY()+1,i,i);
      if(integral > 0){
        for(int j=0; j <= hMigration->GetNbinsX(); j++) {
          double bincontent = hMigration->GetBinContent(j,i)/integral*100;
          hMigration->SetBinContent(j,i,bincontent);
        }
      }
    }
    gStyle->SetPaintTextFormat("4.1f");
    hMigration->SetContour(99);
    hMigration->Draw("col4zsame");
    hMigration->SetMarkerColor(kBlack);
    hMigration->SetMarkerSize(1.4);
    hMigration->Draw("textsame");
  } else {
    std::cout << "Object not found "<< migration.Data() << std::endl;
    if(verbose){
      VERBOSEclass("cannot draw '%s' as migration - object not found",migration.Data());
      this->printObjects();
    }
    return;
  }

  return;

}

//__________________________________________________________________________________|___________

bool TQHWWPlotter::drawStack(TQTaggable& tags){
  // draw the stack produced by TQHWWPlotter::stackHistograms
  // @tag:errors.drawStatMC: control drawing of statistical MC errors (default:true)
  // @tag:errors.drawSysMC: control drawing of statistical MC errors (default:false)
  // @tag:errors.showSys: control drawing of statistical MC errors (deprecated, use errors.drawSysMC instead)
  // @tag:style.showTotalBkg: control display of additional line (and legend entry) for total background (default:true)
  // @tag:style.drawData: control whether data points will be shown (default:true)
  // @tag:style.data.asymErrors: show asymmetric errors (poisson errors) for data points
  
  this->getPad("main");  
  bool statMcErrors = tags.getTagBoolDefault("errors.drawStatMC",true );
  bool sysMcErrors = tags.getTagBoolDefault("errors.drawSysMC",false );
  bool showTotalBkg = tags.getTagBoolDefault ("style.showTotalBkg",true);
  bool drawData = tags.getTagBoolDefault ("style.drawData",true);
  bool asymErrorsData = tags.getTagBoolDefault("style.data.asymErrors",false);
  bool verbose = tags.getTagBoolDefault("verbose",false);
  

  TH1* hMaster = this->getObject<TH1>("master");
  TH1* hTotalBkg = this->getObject<TH1>("totalBkg");
  if(!hMaster) return false;

  // the first histogram to draw is the SM histogram.
  hMaster->Draw("hist");
 
  //////////////////////////////////////////////////////
  // error band on background MC
  //////////////////////////////////////////////////////

  if(verbose) VERBOSEclass("creating MC error band");
  TGraphAsymmErrors * errorGraph = 0;
  // all this only makes sense if we have a histogram representing the "total
  // background"
  if (hTotalBkg) {
    if (statMcErrors || sysMcErrors) {
      errorGraph = TQHistogramUtils::getGraph(hTotalBkg);
      errorGraph->SetTitle("mc error band");
      this->applyStyle(tags,errorGraph,"main.totalBkgError",1.,tags.getTagDoubleDefault("geometry.main.scaling",1.));
    }
  }

  // if we have shapeSys defined it will overwrite errorgraph
  TObjArray* histosAsymmSys = this->getObject<TObjArray>("asymmSys");
  if (hTotalBkg && histosAsymmSys) {
    if (statMcErrors || sysMcErrors) {
      errorGraph = TQHistogramUtils::getGraph(hTotalBkg, histosAsymmSys);
      errorGraph->SetTitle("mc error band");
      this->applyStyle(tags,errorGraph,"main.totalBkgError",1.,tags.getTagDoubleDefault("geometry.main.scaling",1.));
    }
  }
  this->addObject(errorGraph,"totalBkgErr");

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

  if(verbose) VERBOSEclass("drawing backgrounds");

  THStack* stack = this->getObject<THStack>("stack");
  TString stackDrawOptions = tags.getTagStringDefault ("style.stackDrawOptions","hist");
  bool stackSignal = tags.getTagBoolDefault ("style.stackSignal",false);

  // draw the backgrounds
  if (hTotalBkg && stack) {
    if(!stackSignal){
      stack->Draw(stackDrawOptions + " same");
    }
    // if stack signal draw stack above the total bkg line
    if (showTotalBkg) {
      stack->Draw(stackDrawOptions + " same");
      if (errorGraph)  errorGraph->Draw("2");
    }
    if (stackSignal) {
      stack->Draw(stackDrawOptions + " same");
      if (errorGraph)  errorGraph->Draw("2");
    }
  }

  //@tag:stack: process tag to steer whether this process will be added to the stack (default: true for bkg, else false)
  //@tag:drawOptions: control the draw options of this process (default: 'hist' for MC, 'ep' for data)
  
  if(verbose) VERBOSEclass("drawing signal (and other non-stacked histograms)");
  // draw signal
  TQTaggableIterator itr_sig(fProcesses);
  while(itr_sig.hasNext()){
    TQNamedTaggable* process = itr_sig.readNext();
    if(!process) continue;
    if(process->getTagBoolDefault(".isData",false)) continue;
    if(!(process->getTagBoolDefault("stack",process->getTagBoolDefault(".isBackground") || stackSignal))){
      TH1 * h = this->getObject<TH1>(this->makeHistogramIdentifier(process));
      if(!h) continue;
      TString drawOpt = process->getTagStringDefault("drawOptions", "hist") + " same";
      if(process->getTagBoolDefault("stackShift",false)){
        TH1* hcopy = TQHistogramUtils::copyHistogram(h,TString::Format("%s_shifte",h->GetName()));
        hcopy->Add(hTotalBkg);
        hcopy->Draw(drawOpt);
      } else {
        h->Draw(drawOpt);
      }
    }
  }

  // draw data
  if (drawData) {
    if(verbose) VERBOSEclass("drawing data");
    TQTaggableIterator itr_data(fProcesses);
    while(itr_data.hasNext()){
      TQNamedTaggable* process = itr_data.readNext();
      if(!process) continue;
      if(!process->getTagBoolDefault(".isData",false)) continue;
      TH1 * h = this->getObject<TH1>(this->makeHistogramIdentifier(process));
      if(!h) continue;
      this->applyStyle(tags,h,"main.data");
      if (asymErrorsData) {
        h->Sumw2(false); //only do this for data, since uncertainty is sqrt(n). However, it is needed for kPoisson to take effect
        h->SetBinErrorOption(TH1::kPoisson);
      }  
      TString drawOpt = process->getTagStringDefault("drawOptions","ep") + " same";
      if(verbose)
	VERBOSEclass("drawing data histogram '%s' with option '%s'", h->GetName(),drawOpt.Data());
      h->Draw(drawOpt);
    }
  }

  if (tags.getTagBoolDefault("useToyData",false))
  {
    TH1 * h = this->getObject<TH1>("toyData");
    if (h)
    {
      this->applyStyle(tags,h,"main.data");
      h->Draw("ep same");
    }
  }
  
  /*
  //disabled since aparently not working and part of a regression.
  if(!tags.getTagBoolDefault("style.drawDataFront",true)){
    // draw the backgrounds
    if (hTotalBkg && stack) {
      // if stack signal draw stack above the total bkg line
      if (showTotalBkg) {
	if (errorGraph)  errorGraph->Draw("2");
	stack->Draw(stackDrawOptions + " same");
      }
    }
  }
  */
  if (tags.getTagBoolDefault("isCompPlot",false))
    stack->Draw(stackDrawOptions + " same");
  return true;
}

//__________________________________________________________________________________|___________
bool TQHWWPlotter::drawProfiles(TQTaggable& tags){
  this->getPad("main");
  bool verbose = tags.getTagBoolDefault("verbose",false);
  
  TProfile* hMaster = this->getObject<TProfile>("master");
  TProfile* hTotalBkg = this->getObject<TProfile>("totalBkg");
  if(!hMaster || !hTotalBkg) return false;
  //hOverlay = this->getObject<TProfile>(overlay);
  hMaster->SetMaximum(hMaster->GetYmax());
  hMaster->SetMinimum(hMaster->GetYmin());
  
  // the first histogram to draw is the SM histogram.
  hMaster->Draw("hist");
  
  //////////////////////////////////////////////////////
  // calculate axis ranges
  // rescale to avoid graphical collisions
  //////////////////////////////////////////////////////

  /*if(verbose) VERBOSEclass("calculating axis ranges & rescaling histograms");
  bool axisOK = this->calculateAxisRanges1D(tags);
  if(!axisOK){
    if(verbose) VERBOSEclass("encountered invalid axis ranges, using defaults");
  }*/
  
  //////////////////////////////////////////////////////
  // draw everything
  //////////////////////////////////////////////////////

  if(verbose) VERBOSEclass("drawing backgrounds");

  // draw the backgrounds
  TQTaggableIterator itr_bkg(fProcesses);
  while(itr_bkg.hasNext()){
    TQNamedTaggable* process = itr_bkg.readNext();
    TString tmp1 = process->getName(); 
    WARNclass("process %s",tmp1.Data());
    /*if(TQStringUtils::equal(tmp1, "top")) continue;
    if(TQStringUtils::equal(tmp1, "diboson")) continue;
    if(TQStringUtils::equal(tmp1, "H [125]")) continue;
    if(TQStringUtils::equal(tmp1, "data")) continue;
    if(TQStringUtils::equal(tmp1, "Zjets")) continue;
    if(TQStringUtils::equal(tmp1, "Wjets")) continue;
    if(TQStringUtils::equal(tmp1, "H [300]")) continue;*/
    //if(TQStringUtils::equal(tmp1, "H [600]")) continue;
    //if(TQStringUtils::equal(tmp1, "H [900]")) continue;
    if(!process) continue;
    //if(!process->getTagBoolDefault(".isBkg",false)) continue;
    TProfile * h = this->getObject<TProfile>(this->makeHistogramIdentifier(process));
    if(!h) continue;
    this->applyStyle(tags,h,"main.bkg");
    h->Draw(process->getTagStringDefault("drawOptions", "ep") + " same");
    WARNclass("draw");
  }
  /*
  if (hTotalBkg && stack) {
    if(!stackSignal){
      stack->Draw(stackDrawOptions + " same");
    }
    if (showTotalBkg) {
      hTotalBkg->Draw("hist same");
    }
    // if stack signal draw stack above the total bkg line
    if (stackSignal) {
      stack->Draw(stackDrawOptions + " same");
    }
  }
*/
  //@tag:stack: process tag to steer whether this process will be added to the stack (default: true for bkg, else false)
  //@tag:drawOptions: control the draw options of this process (default: 'hist' for MC, 'ep' for data)
  /*
  if(verbose) VERBOSEclass("drawing signal (and other non-stacked histograms)");
  // draw signal
  TQTaggableIterator itr_sig(fProcesses);
  while(itr_sig.hasNext()){
    TQNamedTaggable* process = itr_sig.readNext();
    if(!process) continue;
    if(process->getTagBoolDefault(".isData",false)) continue;
    if(!(process->getTagBoolDefault("stack",process->getTagBoolDefault(".isBackground") || stackSignal))){
      TProfile * h = this->getObject<TProfile>(this->makeHistogramIdentifier(process));
      if(!h) continue;
      h->Draw(process->getTagStringDefault("drawOptions", "ep") + " same");
    }
  }

  // draw data
  if (drawData) {
    if(verbose) VERBOSEclass("drawing data");
    TQTaggableIterator itr_data(fProcesses);
    while(itr_data.hasNext()){
      TQNamedTaggable* process = itr_data.readNext();
      if(!process) continue;
      if(!process->getTagBoolDefault(".isData",false)) continue;
      TProfile * h = this->getObject<TProfile>(this->makeHistogramIdentifier(process));
      if(!h) continue;
      this->applyStyle(tags,h,"main.data");
      h->Draw(process->getTagStringDefault("drawOptions","ep") + " same");
    }
  }
  */
  return true;
}
//__________________________________________________________________________________|___________

void TQHWWPlotter::drawLegend(TQTaggable& tags){
  // draw the legend produced by TQHWWPlotter::makeLegend
  bool verbose = tags.getTagBoolDefault("verbose",false);
  bool legpad = tags.getTagBoolDefault("style.useLegendPad",false);
  //@tag:[style.showLegend] Controls if legend is shown. Default: true
  if (!tags.getTagBoolDefault("style.showLegend",true)) return;
  TLegend* legend = this->getObject<TLegend>("legend");
  if(legpad){
    if(verbose) VERBOSEclass("drawing legend pad");
    if(!this->getPad("legend")){
      ERRORclass("error retrievling legend pad!");
    }
    legend->Draw();
  } else {
    if(verbose) VERBOSEclass("drawing legend on-pad");
    this->getPad("main");    
    legend->Draw("same");
  }
}

//__________________________________________________________________________________|___________

TCanvas * TQHWWPlotter::makePlot(TString histogram, TQTaggable& tags) {
  // master-function controlling the plotting
  bool verbose = tags.getTagBoolDefault("verbose",false);
 
  if(verbose) VERBOSEclass("entering function");
  TString lepch;
 
  TString histname_data = "";
  TString histname_bkg = "";
  TString histname_sig = "";
  if(histogram.First('=') != kNPOS){
    TQTaggable names(histogram);
    names.getTagString("bkg",histname_bkg);
    names.getTagString("sig",histname_sig);
    names.getTagString("data",histname_data);
    if(histname_bkg.IsNull()) getTagString("sig" ,histname_bkg);
    if(histname_bkg.IsNull()) getTagString("data",histname_bkg);
    if(histname_bkg.IsNull()) return NULL;
    if(histname_data.IsNull()) histname_data = histname_bkg;
    if(histname_sig.IsNull()) histname_sig = histname_bkg;
    names.getTagString("lepch",lepch);
    names.getTagString("channel",lepch);
    histogram = "";
    //@tag:.isSignal: process tag to identify signal
    if(this->getNProcesses(".isSignal") > 0) histogram+="sig:"+histname_sig+",";
    //@tag:.isBackground: process tag to identify background
    if(this->getNProcesses(".isBackground") > 0) histogram+="bkg:"+histname_bkg+",";
    //@tag:.isData: process tag to identify data
    if(this->getNProcesses(".isData") > 0) histogram+="data:"+histname_data+",";
    TQStringUtils::removeTrailing(histogram,",");
  } else {
    lepch = TQStringUtils::readPrefix(histogram, ":", "?");
    histname_bkg = histogram;
    histname_sig = histogram;
    histname_data = histogram;
  }

  gStyle->SetOptStat(false);
  gStyle->SetOptFit(false);
  gStyle->SetOptTitle(false);

  //@tag:[input.bkg]: overwrite the sample folder path used for background
  //@tag:[input.sig]: overwrite the sample folder path used for signal
  //@tag:[input.data]: overwrite the sample folder path used for data
  //@tag:[input.lepch,input.chname]: overwrite the lepton channel used (deprecated, use 'input.channel' instead)
  //@tag:[input.channel]: overwrite the channel
  //@tag:[input.sys]: choose the path from which to read the systematics
  
  tags.getTagString("input.lepch", lepch);
  tags.setTagString("input.channel", lepch);
  tags.setTagString("input.chname",lepch == "?" ? "all" : lepch);
  tags.setTagString("input.name",histogram);
  tags.setTagString("input.bkg", histname_bkg);
  tags.setTagString("input.data",histname_data);
  tags.setTagString("input.sig", histname_sig);
  
  tags.setTagString("input.sys", histname_bkg);
 
  if(verbose) VERBOSEclass("bkg: %s",histname_bkg.Data());
  if(verbose) VERBOSEclass("sig: %s",histname_sig.Data());
  if(verbose) VERBOSEclass("data: %s",histname_data.Data());
  if(verbose) VERBOSEclass("sys: %s",histname_data.Data());

  //////////////////////////////////////////////////////
  // obtain the histograms
  //////////////////////////////////////////////////////

  if(verbose) VERBOSEclass("collecting histograms");
  TObjArray* histos = this->collectHistograms(tags);
  if(!histos) return NULL;

  // @tag:style.drawData: select whether to draw data points (default:true)
  bool drawData = tags.getTagBoolDefault ("style.drawData",true);
  if (!drawData) {
    tags.setGlobalOverwrite(true);  
    tags.setTagBool("style.showKS",false);
    tags.setTagBool("style.showRatio",false);
    tags.setTagBool("style.showDoverB",false);
    tags.setTagBool("style.showDminusB",false);
    tags.setTagBool("style.showSignificance",false);
    tags.setGlobalOverwrite(false);
  }

  // @tag:style.useToyData: if set to true, additional toy (asimov) data is drawn 
  if (tags.getTagBoolDefault("useToyData",false))
  {
    TRandom3 * rand = new TRandom3(0);
    TH1* hTotalBkg = this->getObject<TH1>("totalBkg");
    TH1* hTotalSig = this->getObject<TH1>("totalSig");
    TH1* hTotalBkgPlusSig = TQHistogramUtils::copyHistogram(hTotalBkg,"totalBkgPlusSig");
    if (hTotalSig && (tags.getTagBoolDefault("style.stackSignal",false) == false))
      hTotalBkgPlusSig->Add(hTotalSig);
    TH1 *histoToyData = ((TH1*)(hTotalBkgPlusSig)->Clone());
    histoToyData->SetNameTitle("toy data","toy data");
    // @tag: toyDataMarkerStyle: marker style for toy data is set using TH1::SetMarkerStyle (default=20)
    histoToyData->SetMarkerStyle(tags.getTagIntegerDefault("style.toyDataMarkerStyle",20));
    // @tag: toyDataMarkerSize: marker size for toy data is set using TH1::SetMarkerSize (default=0.8)
    histoToyData->SetMarkerSize(tags.getTagDoubleDefault("style.toyDataMarkerSize",0.8));
    histoToyData->Reset();
    histoToyData->FillRandom(hTotalBkgPlusSig,hTotalBkgPlusSig->Integral());
    histoToyData->Scale(hTotalBkgPlusSig->Integral()/histoToyData->Integral());
    for (Int_t bin_i=1; bin_i<histoToyData->GetNbinsX()+1; bin_i++)
    {
      histoToyData->SetBinContent(bin_i,rand->Poisson(hTotalBkgPlusSig->GetBinContent(bin_i)));
      histoToyData->SetBinError(bin_i,TMath::Sqrt(histoToyData->GetBinContent(bin_i)));
    }
    this->addObject(histoToyData,"toyData");
  }
  TQTH1Iterator histitr(histos);
  int nEntries = 0;
  bool is2D = false;
  bool isTProfile = false;
  while(histitr.hasNext()){
    TH1* hist = histitr.readNext();
    nEntries += hist->GetEntries();
    if(dynamic_cast<TH2*>(hist)) is2D=true;
    if(dynamic_cast<TProfile*>(hist)) isTProfile=true;
  }
  if(nEntries < 1){
    WARNclass("refusing to plot histogram '%s' - no entries!",histogram.Data());
    return NULL;
  }

  //////////////////////////////////////////////////////
  // the ratio plot can only be shown if there is a valid
  // data histogram and at least one MC background histogram 
  //////////////////////////////////////////////////////

  if(verbose) VERBOSEclass("sanitizing tags");
  
  // @tag:style.overrideTotalBkgRequirement: usually, 1D plots without background are skipped. force plotting data/signal only with this option.
  bool overrideTotalBkgRequirement = tags.getTagBoolDefault("style.overrideTotalBkgRequirement", is2D);

  TH1* hTotalBkg = this->getObject<TH1>("totalBkg");
  if ((!hTotalBkg || hTotalBkg->GetEntries() <= 0) && !overrideTotalBkgRequirement){
    if(verbose) VERBOSEclass("no total background histogram found, quitting");
    return NULL;
  }
  TH1* hMaster = this->getObject<TH1>("master");
  if (!hMaster){
    if(verbose) VERBOSEclass("no master histogram found, quitting");
    return NULL;
  }

  /*@tag:[style.showRatio,style.showSignificance,style.showDoverB,style.showDminusB,style.showDminusBoverD,style.showOptScan,style.showCustom]
    
    control what is shown in the sub-plot. all of these default to 'false', only showing the main plot.
    if any of these are set to true, the corresponding sub plot is shown. only one sub plot can be shown at a time.

    style.showRatio: show the ratio between data and signal+bkg
    style.showMultiRatio: show the ratio between pairs of data and background processes in order of adding
    style.showSignificance: show the poisson significance of the data over the background in each  bin
    style.showDoverB: show the ratio between data and background
    style.showDminusB: show the difference between data and background
    style.showDminusBoverD: show the the ratio between the data-bkg (signal estimate) and the data
    style.showOptScan: show an optimization scan. several figure-of-merit (FOM) options are supported via style.FOMmode and style.FOMbbb
    style.showCustom, style.ratioFormula: customized subplot. can be controlled with style.ratioFormula
   */
  bool showRatio = tags.getTagBoolDefault ("style.showRatio",false);
  bool showPull = tags.getTagBoolDefault ("style.showPull",false);
  bool showMultiRatio = tags.getTagBoolDefault ("style.showMultiRatio",false);
  bool showSignificance = tags.getTagBoolDefault ("style.showSignificance",false);
  bool showDoverB = tags.getTagBoolDefault ("style.showDoverB",false);
  bool showDminusB = tags.getTagBoolDefault ("style.showDminusB",false);
  bool showDminusBoverD = tags.getTagBoolDefault ("style.showDminusBoverD",false);
  bool showOptScan = tags.getTagBoolDefault ("style.showOptScan",false);
  bool showCustom = tags.getTagBoolDefault ("style.showCustom",false) && tags.hasTagString("style.ratioFormula") ;
  int nDim = tags.getTagIntegerDefault("style.nDim",1);
  if(nDim != 1){
    showPull = false;
    showRatio = false;
    showMultiRatio = false;
    showDoverB = false;
    showSignificance = false;
    showDminusB = false;
    showDminusBoverD = false;
    showOptScan = false;
    showCustom = false;
    tags.setTagBool("style.showTotalBkg",false);
  }
  if(showRatio && !showDoverB) showDoverB = true;
  if(!hTotalBkg){
    showPull = false;
    showSignificance = false;
    showDoverB = false;
    showDminusB = false;
    showDminusBoverD = false;
    showSignificance = false;
    showCustom = false;
  }
  if(showSignificance || showDoverB || showOptScan || showDminusB || showDminusBoverD || showCustom || showMultiRatio || showPull){
    tags.setGlobalOverwrite(true);
    tags.setTagBool("style.showSub",true);
    tags.setGlobalOverwrite(false);
  } else {
    tags.setTagBool("style.showSub",false);
  }
  if (showOptScan)
    showDoverB = false;
  if (showDminusBoverD)
    showDoverB = false;

  //////////////////////////////////////////////////////
  // set and apply the style
  //////////////////////////////////////////////////////

  if(verbose) VERBOSEclass("applying style");
  bool newPlotStyle = tags.getTagBoolDefault ("style.newPlotStyle", false);
  if (newPlotStyle){
    this->setStyleIllinois(tags);
  } else {
    this->setStyle(tags);
  }

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
  //create the legend before the stack! (stack creation also involves normalization, i.e., after the stack is created we don't have the original event yields at hand anymore!)
  //////////////////////////////////////////////////////
  // create the stack
  //////////////////////////////////////////////////////
 
  if(hTotalBkg && nDim == 1){
    if(verbose) VERBOSEclass("stacking histograms");
    this->stackHistograms(tags,"stack");
  }

  //////////////////////////////////////////////////////
  // basic label setup
  //////////////////////////////////////////////////////

  if(verbose) VERBOSEclass("setting labels");
  TString label;
  tags.setGlobalOverwrite(true);

  int labelidx = 0;  
  if (tags.getTagString("labels.lumi", label)){
    labelidx++;
    tags.setTagString(TString::Format("labels.%d",labelidx), label);
  }
  if (tags.getTagString("labels.process", label)){
    labelidx++;
    tags.setTagString(TString::Format("labels.%d",labelidx), label);
  }
  
  //////////////////////////////////////////////////////
  // calculate advanced labels
  //////////////////////////////////////////////////////

  if(verbose) VERBOSEclass("calculating tests and other labels");

  if(hTotalBkg){
    if(tags.getTagBoolDefault("style.showBkgRMS",false)){
      if(verbose) VERBOSEclass("calculating total background RMS");
      double RMS = hTotalBkg->GetRMS();
      tags.setTagString(TString::Format("labels.%d",labelidx),TString::Format("#font[72]{RMS(%s) = %g}", hTotalBkg->GetTitle(), RMS));
      labelidx++;
    }
    TQTaggableIterator itr(fProcesses);
    while(itr.hasNext()){
      // get the data histogram
      TQNamedTaggable* process = itr.readNext();
      if(!process) continue;
      if(!process->getTagBoolDefault(".isData",false)) continue;
      TH1 * h_data = this->getObject<TH1>(this->makeHistogramIdentifier(process));
      if(!h_data || h_data->GetEntries() == 0) continue;
      if(tags.getTagBoolDefault ("style.showKS",false)){
        if(verbose) VERBOSEclass("calculating KS test for '%s'",h_data->GetTitle());
        float ks = std::numeric_limits<double>::quiet_NaN();
        if(h_data->GetEntries() > 0 && hTotalBkg->GetEntries() > 0 && h_data->Integral() > 0 && hTotalBkg->Integral() > 0){
          ks = hTotalBkg->KolmogorovTest(h_data);
        }
        TString id = TString::Format("labels.%d",labelidx);
        if(TQUtils::isNum(ks)){
          float roundedKS = (float) int(ks*10000.+0.5)/10000.;
          tags.setTagString(id,TString::Format("#color[2]{#font[72]{KS Prob = %2.1f%%}}", roundedKS*100.));
        } else {
          tags.setTagString(id,"#color[2]{#font[72]{KS Prob = N.A.}}");
        }
        labelidx++;
      }
      if(tags.getTagBoolDefault ("style.showChi2",false)){
        if(verbose) VERBOSEclass("calculating Chi2 test for '$s'",h_data->GetTitle());
        Double_t chi2 = 0;
        Int_t ndf = 0, igood = 0;
        TH1F* h_data_tmp = (TH1F*) h_data->Clone("h_data_tmp");
        TH1F* hTotalBkg_tmp = (TH1F*) hTotalBkg->Clone("hTotalBkg_tmp");
        double maxSanitization = 0.;
        if (tags.getTagDouble("style.chi2.maxSanitization",maxSanitization)) {
          for(int i_tmp=1;i_tmp<=h_data_tmp->GetNbinsX()+1;i_tmp++){
            if( h_data_tmp->GetBinContent(i_tmp)==0 && hTotalBkg_tmp->GetBinContent(i_tmp)<maxSanitization ){
              hTotalBkg_tmp->SetBinContent(i_tmp, 0);
              hTotalBkg_tmp->SetBinError(i_tmp, 0);
            }
            if( hTotalBkg_tmp->GetBinContent(i_tmp)==0 && h_data_tmp->GetBinContent(i_tmp)<maxSanitization ){
              h_data_tmp->SetBinContent(i_tmp, 0);
              h_data_tmp->SetBinError(i_tmp, 0);
            }
          }
        }
        double prob = h_data_tmp->Chi2TestX(hTotalBkg_tmp,chi2,ndf,igood,"UW UF OF");
        float roundedChi2Prob = (float) int(prob*10000.+0.5)/10000.;
        tags.setTagString(TString::Format("labels.%d",labelidx),TString::Format("#color[2]{#font[72]{Chi2 Prob = %2.1f%%}}", roundedChi2Prob*100.));
        labelidx++;
      }
      if(tags.getTagBoolDefault ("style.showChi2P",false)){
        TH1F* h_data_tmp = (TH1F*) h_data->Clone("h_data_tmp");
        TH1F* hTotalBkg_tmp = (TH1F*) hTotalBkg->Clone("hTotalBkg_tmp");
        double maxSanitization = 0.;
        if (tags.getTagDouble("style.chi2.maxSanitization",maxSanitization)) {
          for(int i_tmp=1;i_tmp<=h_data_tmp->GetNbinsX()+1;i_tmp++){
            if( h_data_tmp->GetBinContent(i_tmp)==0 && hTotalBkg_tmp->GetBinContent(i_tmp)<maxSanitization ){
              hTotalBkg_tmp->SetBinContent(i_tmp, 0);
              hTotalBkg_tmp->SetBinError(i_tmp, 0);
            }
            if( hTotalBkg_tmp->GetBinContent(i_tmp)==0 && h_data_tmp->GetBinContent(i_tmp)<maxSanitization ){
              h_data_tmp->SetBinContent(i_tmp, 0);
              h_data_tmp->SetBinError(i_tmp, 0);
            }
          }
        }
        double p_value = h_data_tmp->Chi2Test(hTotalBkg_tmp,"UW UF OF");
        tags.setTagString(TString::Format("labels.%d",labelidx),TString::Format("#color[2]{#font[72]{#chi^{2} p-value = %f}}", p_value));
        labelidx++;
      }
      if(tags.getTagBoolDefault ("style.showDataRMS",false)){
        if(verbose) VERBOSEclass("calculating RMS for '$s'",h_data->GetTitle());
        double RMS = h_data->GetRMS();
        tags.setTagString(TString::Format("labels.%d",labelidx),TString::Format("#color[2]{#font[72]{RMS(%s) = %2.1f%%}}", h_data->GetTitle(),RMS));
        labelidx++;
      }
    }
  } else {
    if(verbose) VERBOSEclass("no total background histogram found!");
  }


  //////////////////////////////////////////////////////
  // manage the total background error bars
  //////////////////////////////////////////////////////

  bool sysMcErrors = tags.getTagBoolDefault("errors.drawSysMC",false ) || tags.hasTag("errors.showSys");
  bool statMcErrors = tags.getTagBoolDefault("errors.drawStatMC",true );
  this->setTotalBackgroundErrors(tags,sysMcErrors,statMcErrors);

  tags.setGlobalOverwrite(false);

  //////////////////////////////////////////////////////
  // draw main pad
  //////////////////////////////////////////////////////

  if(verbose) VERBOSEclass("drawing main pad");
  if(isTProfile){
    this->drawProfiles(tags);
    this->drawLegend(tags);
  } else if(nDim == 1){
    if(!tags.getTagBoolDefault("allow1D",true))	return NULL;
    bool ok = this->drawStack(tags);
    if(!ok){
      return NULL;
    }
    this->drawLegend(tags);
    if(verbose) VERBOSEclass("drawing cut lines");
    this->drawCutLines1D(tags);
  } else if(nDim == 2){
    if(!tags.getTagBoolDefault("allow2D",true)) return NULL;
    if(tags.hasTagString("style.heatmap")){
      this->drawHeatmap(tags);
    } else if (tags.hasTagString("style.migration")){
      std::cout<< "call migration plot" << std::endl;
      this->drawMigration(tags);
    } else {
      this->drawContours(tags);
      this->drawLegend(tags);
    }
    if(verbose) VERBOSEclass("drawing decorations");
    this->drawAdditionalAxes(tags);
    this->drawHeightLines(tags);
  } else {
    ERRORclass("unsupported dimensionality (nDim=%d)!",nDim);
    return NULL;
  }

  //////////////////////////////////////////////////////
  // do some advanced magic
  //////////////////////////////////////////////////////

  TString fit;
  if(hTotalBkg && nDim == 1 && tags.getTagString("totalBkgFit.function",fit)){
    TFitResultPtr p = hTotalBkg->Fit(fit,"Q");
    TF1* func = dynamic_cast<TF1*>(hTotalBkg->GetListOfFunctions()->Last());
    if(func){
      func->SetLineColor(tags.getTagIntegerDefault("totalBkgFit.color",kRed));
      func->SetLineStyle(tags.getTagIntegerDefault("totalBkgFit.style",1));
      func->Draw("SAME");
      if(tags.getTagBoolDefault("totalBkgFit.showResults",true)){
        TString info = "";
        for(Int_t i=0; i<func->GetNpar(); ++i){
          TString name = func->GetParName(i);
          tags.getTagString(TString::Format("totalBkgFit.parameter.%d.name",i),name);
          double val = func->GetParameter(i);
          double uncertainty = func->GetParError(i);
          tags.setTagString(TString::Format("labels.%d",labelidx),TString::Format("%s = %g #pm %g",name.Data(),val,uncertainty));
          labelidx++;
        }
      }
    }
  }

  if (drawData) {
    if(verbose) VERBOSEclass("fitting data slopes");
    TQTaggableIterator itr_data(fProcesses);
    while(itr_data.hasNext()){
      TQNamedTaggable* process = itr_data.readNext();
      if(!process) continue;
      if(!process->getTagBoolDefault(".isData",false)) continue;
      TH1 * h = this->getObject<TH1>(this->makeHistogramIdentifier(process));
      if(!h) continue;
      if(tags.getTagBoolDefault("style.data.fitSlope",false)){
        h->Fit("pol1","QG","",hMaster->GetXaxis()->GetXmin(),hMaster->GetXaxis()->GetXmax());
        TF1* f = h->GetFunction("pol1");
        f->SetName(TString::Format("%s_fit",h->GetName()));
        //      this->addObject(f);
        f->SetLineColor(h->GetMarkerColor());
        f->SetLineWidth(tags.getTagIntegerDefault("style.data.fitSlope.lineWidth",1));
        f->SetLineStyle(tags.getTagIntegerDefault("style.data.fitSlope.lineStyle",2));
        //@tag:style.data.fitSlope.exportResults: export the fit results as tags on the plotter
        if (tags.getTagBoolDefault("style.data.fitSlope.exportResults",false)) {
          this->setTagDouble(TString::Format("export.fitSlope.%s.slope",h->GetName()),f->GetParameter(1));
          this->setTagDouble(TString::Format("export.fitSlope.%s.slopeError",h->GetName()),f->GetParError(1));
          this->setTagDouble(TString::Format("export.fitSlope.%s.chi2",h->GetName()),f->GetChisquare());
        }
        if (tags.getTagBoolDefault("style.data.fitSlope.printResults",true)) {
          double slope = TQUtils::roundAuto(f->GetParameter(1));
          double slopeErr = TQUtils::roundAuto(f->GetParError(1));
          double chi2 = TQUtils::roundAuto(f->GetChisquare());
          h->SetTitle(TString::Format("%s (slope #approx %g #pm %g with #chi^{2}#approx%g)",h->GetTitle(),slope,slopeErr,chi2));
        }
      }
    }
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
    
    if (showDoverB){
      if(verbose) VERBOSEclass("drawing ratio");
      this->drawRatio(tags);
    } else if(showPull){
      if(verbose) VERBOSEclass("drawing pull");
      this->drawPull(tags);
    } else if(showMultiRatio){
      if(verbose) VERBOSEclass("drawing multi ratio");
      this->drawMultiRatio(tags);
    } else if(showSignificance){
      if(verbose) VERBOSEclass("drawing significance");
      this->drawSignificance(tags);
    } else if(showOptScan){
      if(verbose) VERBOSEclass("drawing optimization scan");
      this->drawOptScan(tags);
    } else if(showDminusB){
      if(verbose) VERBOSEclass("drawing data minus background");
      this->drawDataMinusBackground(tags);
    } else if(showDminusBoverD){
      if(verbose) VERBOSEclass("drawing data minus background over data");
      this->drawDataMinusBackgroundOverData(tags);
    } else if(showCustom) {
      if(verbose) VERBOSEclass("drawing custom subplot");
      this->drawCustomSubPlot(tags);
    }
  }

  if(verbose) VERBOSEclass("all done!");
  // return the canvas
  return canvas;
}

//__________________________________________________________________________________|___________

bool TQHWWPlotter::calculateAxisRanges1D(TQTaggable& tags){
  // calculate the axis ranges, taking into account the given block tags
  bool logScale = tags.getTagBoolDefault ("style.logScale",false );
  bool drawData = tags.getTagBoolDefault ("style.drawData",true);
  bool stackSignal = tags.getTagBoolDefault ("style.stackSignal",false);
  bool verbose = tags.getTagBoolDefault("verbose",false);
 
  TH1* hTotalBkg = this->getObject<TH1>("totalBkg");
  TList* histograms = new TList();
  if (hTotalBkg) histograms->Add(hTotalBkg);


  double min = std::numeric_limits<double>::infinity();
  TQTaggableIterator itr(fProcesses);
  while(itr.hasNext()){
    TQNamedTaggable* process = itr.readNext();
    if(!process) continue;
    if(process->getTagBoolDefault(".isData",false) && !drawData) continue;
    TH1 * h = this->getObject<TH1>(this->makeHistogramIdentifier(process));
    if(!h) continue;
    histograms->Add(h);
    double tmpmin = TQHistogramUtils::getMin(h, true, true, logScale ? tags.getTagDoubleDefault("style.logMinMin",1e-9) : -std::numeric_limits<double>::infinity() ); //ignore empty/very small bins for log scale plots when trying to automatically determine axis range
    if(tmpmin < min) min = tmpmin;
  }
  //@tag:[style.min,style.logMin,style.linMin,(style.logMinMin)] These argument tags determine the lower bound of the y axis in 1D plots. "style.min" is used unless the specific tag for the plot type (lin/log) is set. Additionally, "style.logMinMin" defaults to 1e-9 and acts as an additional lower bound; use with great care!
  tags.getTagDouble("style.min", min);
  if(logScale){
    tags.getTagDouble("style.logMin",min);
  } else {
    tags.getTagDouble("style.linMin",min);
  }

  if(logScale && min < tags.getTagDoubleDefault("style.logMinMin",1e-9) ) min = tags.getTagDoubleDefault("style.logMinMin",1e-9);
  
  itr.reset();
  TH1* hBkgPlusSig = NULL;
  while(itr.hasNext()){
    TQNamedTaggable * process = itr.readNext();
    if(!process) continue;
    if(!process->getTagBoolDefault(".isSignal",false)) continue;
    TH1* h = this->getObject<TH1>(this->makeHistogramIdentifier(process));
    if (hTotalBkg && process->getTagBoolDefault("stack", stackSignal)) {
      if(!hBkgPlusSig){
        hBkgPlusSig = TQHistogramUtils::copyHistogram(hTotalBkg,"hTotalBkgPlusSig");
        hBkgPlusSig->SetTitle("Total Background + Signal");
        if(hBkgPlusSig) histograms->Add(hBkgPlusSig);
      }
      if(h) hBkgPlusSig->Add(h);
    } else {
      if(h) histograms->Add(h);
    }
  }
 
  double max_precise = this->getHistogramUpperLimit(tags, histograms,min,true);
  delete histograms;

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

TString TQHWWPlotter::getScaleFactorList(TString histname){
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

//__________________________________________________________________________________|___________

void TQHWWPlotter::drawLabels(TQTaggable& tags){
  // draw the labels given by the tags
  this->getPad("main");
  
  bool drawlabels = tags.getTagBoolDefault("style.showLabels",true);
  
  double scaling = tags.getTagDoubleDefault("geometry.main.scaling",1.);
  double textsize = tags.getTagDoubleDefault("style.textSize",0.05);
  int font = tags.getTagDoubleDefault("style.text.font",42);
  int color = tags.getTagDoubleDefault("style.text.color",1);
  double xscaling = tags.getTagDoubleDefault("geometry.xscaling",1.);

  double x = tags.getTagDoubleDefault("style.labels.xOffset",0.2)*xscaling;
 
  double y = tags.getTagDoubleDefault("style.labels.yPos",0.86 - tags.getTagDoubleDefault("geometry.main.additionalTopMargin",0.));
 
  //@tag:labels.drawATLAS: decide whether to draw the 'ATLAS' label
  bool drawATLAS = tags.getTagBoolDefault ("labels.drawATLAS",drawlabels);
 
  TString nfLabel = "";
  //@tag:labels.drawNFInfo: decide whether to draw information on which NFs were applied
  if(tags.getTagBoolDefault ("labels.drawNFInfo",false)){
    TString tmpLabel = tags.getTagStringDefault("labels.nfInfo","#color[2]{(NF applied for %s)}");
    if(TQStringUtils::countText(tmpLabel,"%s") == 1){
      TString nflist = this->getScaleFactorList(tags.getTagStringDefault("input.bkg",""));
      if(!nflist.IsNull()){
        nfLabel = TString::Format(tmpLabel.Data(),TQStringUtils::convertLaTeX2ROOTTeX(nflist).Data());
      }
    }
  }

  //@tag:labels.drawInfo: decide whether to draw the technical info tag on the top right of the plot
  TString infoLabel = tags.getTagBoolDefault ("labels.drawInfo",drawlabels) ? tags.getTagStringDefault ("labels.info",TString::Format("Plot: \"%s\"", tags.getTagStringDefault("input.name","histogram").Data())) : "";
  //@tag:labels.atlas: which ATLAS label to use (Private, work in progress, Internal, Preliminary, ... - default:'Private')
  TString atlasLabel = tags.getTagStringDefault ("labels.atlas","Private");
  TString stickerLabel = tags.getTagStringDefault ("labels.sticker","");
 
  TPad* pad = this->getPad("main");
  pad->cd(1);

  if (drawATLAS) {
    // draw the ATLAS label
    TLatex l;
    l.SetNDC();
    l.SetTextFont(72);
    //@tag:labels.drawATLAS.scale: scale of the ATLAS label. Defaults to the scale set by 'labels.atlas.scale' or 1.25 if neither of the two tags are present
    l.SetTextSize(textsize * tags.getTagDoubleDefault("labels.drawATLAS.scale",tags.getTagDoubleDefault("labels.atlas.scale",1.25)) * scaling);
    l.SetTextColor(1);
    //@tag:labels.drawATLAS.text: text of the ATLAS tag (default: ATLAS)
    l.DrawLatex(x, y, tags.getTagStringDefault("labels.drawATLAS.text","ATLAS"));
  }
 
  if (drawATLAS && !atlasLabel.IsNull()){
    // draw the ATLAS label addition
    TLatex p;
    p.SetNDC();
    p.SetTextFont(font);
    p.SetTextColor(color);
    //@tag:labels.atlas.scale: scale of the addition to the ATLAS label (Internal, Private,...). Defaults to the scale set by 'labels.drawATLAS.scale' or 1.25 if neither of the two tags are present
    p.SetTextSize(textsize * tags.getTagDoubleDefault("labels.atlas.scale",tags.getTagDoubleDefault("labels.drawATLAS.scale",1.25)) * scaling);
    //@tag:labels.atlas.xOffset : horizontal offset between ATLAS label and its addition. (default: 0.16)
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

  // draw  labels
  double marginStep = tags.getTagDoubleDefault("style.labels.marginStep",0.06);
  double labelTextScale = tags.getTagDoubleDefault("style.labels.scale",0.85);
  if(drawlabels){
    TQIterator itr(tags.getTagList("labels"),true);
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

  // draw extra labels
  if(true){
    int index = 0;
    while(true){
      TString key = TString::Format("extralabels.%d",index);
      TString labeltext;
      if(!tags.getTagString(key,labeltext)){
        break;
      }
      TLatex latex;
      latex.SetNDC();
      latex.SetTextFont(font);
      latex.SetTextSize(textsize * labelTextScale * scaling);
      latex.SetTextColor(color);
      latex.DrawLatex(tags.getTagDoubleDefault(key+".x",x),
                      tags.getTagDoubleDefault(key+".y", y - marginStep * index * scaling),
                      labeltext);
      index++;
    }
  }
}
  
//__________________________________________________________________________________|___________

void TQHWWPlotter::drawSignificance(TQTaggable& tags){
  // draw a significance plot in the sub-pad
  TString totalBkgLabel = tags.getTagStringDefault ("labels.totalBkg", "SM");
 
  TPad* sigPad = this->getPad("sub");
  if(!sigPad) return;

  TH1* hMaster = this->getObject<TH1>("master");
  TH1* hTotalBkg = this->getObject<TH1>("totalBkg");
  if(!hTotalBkg) return;

  int nBins = hMaster->GetNbinsX();
 
  // loop over all histograms
  TQTaggableIterator itr(fProcesses);
  while(itr.hasNext()){
    // get the data histograms only
    TQNamedTaggable* process = itr.readNext();
    if(!process) continue;
    if(!process->getTagBoolDefault(".isData",false)) continue;
    TH1 * h_data = this->getObject<TH1>(this->makeHistogramIdentifier(process));
    if (!h_data) continue;
 
    // calculate the number of valid ratio points: ratio points are considered
    // valid if they have a finite value (MC prediction != 0) (--> nPoints) and
    // the observed data is greater than zero (--> nRatioPoints)
    int nPoints = 0;
    int nRatioPoints = 0;
    for (int i = 1; i <= nBins; i++) {
      if (hTotalBkg->GetBinContent(i) != 0.) {
        nPoints++;
        if (h_data->GetBinContent(i) > 0)
          nRatioPoints++;
      }
    }
 
    if(nRatioPoints < 1){
      // there is nothing to draw -- well, let's do nothing, then
      continue;
    }
 
    // the graph used to draw the error band on the ratio
    TGraphAsymmErrors * significanceGraph = new TGraphAsymmErrors(nPoints);
    this->addObject(significanceGraph,TString::Format("significanceGraph_%s",h_data->GetName()));
 
    int iPoint = 0;
 
    // actual minimum and maximum ratio
    double actualSigMin = 0.;
    double actualSigMax = 0.;
 
    bool first = true;
    // loop over all bins of the histogram
    for (int iBin = 1; iBin <= nBins; iBin++) {
 
      // get the values and errors of data and MC for this bin
      double data = h_data ->GetBinContent(iBin);
      double MC = hTotalBkg->GetBinContent(iBin);
      // cannot do anything if MC expectation is zero
      if (MC == 0.)
        continue;
 
      double sig = 0.;
      double pValue = 0.;

      // set the position and the width of the significance band
      significanceGraph->SetPoint(iPoint, hTotalBkg->GetBinCenter(iBin), 0.);
 
      pValue = TQHistogramUtils::pValuePoisson((unsigned)data, MC);
      if (pValue < 0.5)
        sig = TQHistogramUtils::pValueToSignificance(pValue, (data > MC));
 
      if (sig < 0.) {
        significanceGraph->SetPointError(
                                         iPoint, hTotalBkg->GetBinWidth(iBin) / 2.,
                                         hTotalBkg->GetBinWidth(iBin) / 2., -sig, 0.);
      } else {
        significanceGraph->SetPointError(
                                         iPoint, hTotalBkg->GetBinWidth(iBin) / 2.,
                                         hTotalBkg->GetBinWidth(iBin) / 2., 0., sig);
      }
 
      if (sig < actualSigMin){
        actualSigMin = sig;
      }
      if (sig > actualSigMax){
        actualSigMax = sig;
      }
 
      // set the position and the width of the significance band
      significanceGraph->SetPoint(iPoint, hTotalBkg->GetBinCenter(iBin), 0.);
 
      pValue = TQHistogramUtils::pValuePoisson((unsigned)data, MC);
      if (pValue < 0.5)
        sig = TQHistogramUtils::pValueToSignificance(pValue, (data > MC));
 
      if (sig < 0.) {
        significanceGraph->SetPointError(
                                         iPoint, hTotalBkg->GetBinWidth(iBin) / 2.,
                                         hTotalBkg->GetBinWidth(iBin) / 2., -sig, 0.);
      } else {
        significanceGraph->SetPointError(
                                         iPoint, hTotalBkg->GetBinWidth(iBin) / 2.,
                                         hTotalBkg->GetBinWidth(iBin) / 2., 0., sig);
      }
 
      if (sig < actualSigMin){
        actualSigMin = sig;
      }
      if (sig > actualSigMax){
        actualSigMax = sig;
      }

      iPoint++;
 
    }
 
    double xscaling = tags.getTagDoubleDefault("geometry.xscaling",1.);
    double sigPadScaling = tags.getTagDoubleDefault("geometry.sub.scaling",1.);
    this->applyStyle(tags,significanceGraph,"significance",xscaling,sigPadScaling);
    this->applyGeometry(tags,significanceGraph,"sub" ,xscaling,sigPadScaling);
 
    // set the x range of the ratio graph to match the one of the main histogram
    double xLowerLimit = hTotalBkg->GetBinLowEdge(1);
    double xUpperLimit = hTotalBkg->GetBinLowEdge(nBins) + hTotalBkg->GetBinWidth(nBins);
    significanceGraph->GetXaxis()->SetLimits(xLowerLimit, xUpperLimit);
 
    // set the titles of the axis of the ratio graph
    significanceGraph->GetXaxis()->SetTitle(hTotalBkg->GetXaxis()->GetTitle());
 
    // confine the y axis of the ratio plot
    significanceGraph->GetYaxis()->SetTitle("Significance");
 
    actualSigMin = TMath::Abs(actualSigMin);
    int y1 = TMath::Nint(actualSigMin);
    if (y1 < actualSigMin)
      actualSigMin = y1 + 0.5;
    else
      actualSigMin = y1;
 
    if (fmod(actualSigMin, 1) == 0)
      actualSigMin += 0.5;
    int y2 = TMath::Nint(actualSigMax);
    if (y2 < actualSigMax)
      actualSigMax = y2 + 0.5;
    else
      actualSigMax = y2;
    if (fmod(actualSigMax, 1) == 0)
      actualSigMax += 0.5;
 
    significanceGraph->GetHistogram()->SetMinimum(-actualSigMin);
    significanceGraph->GetHistogram()->SetMaximum(actualSigMax);
 
    // draw the ratio/significance graph
    if (first)
      significanceGraph->Draw("A2");
    else 
      significanceGraph->Draw("2");

    // if 1. is included in the range of the y axis of the ratio plot...
    if ((significanceGraph->GetHistogram()->GetMinimum() <= 0.) && (significanceGraph->GetHistogram()->GetMaximum() >= 0.)) {
      // draw the red line indicating 1 in the ratio plot and around 0 in case
      // of significance
      TLine * line = new TLine(xLowerLimit, 0, xUpperLimit, 0);
      line->SetLineColor(kRed);
      line->Draw();
    }
 
    sigPad->RedrawAxis();
    sigPad->Update();
    first = false;
  }
}

//__________________________________________________________________________________|___________

void TQHWWPlotter::drawRatio(TQTaggable& tags){
  // draw a ratio-plot in the sub-pad
  double ratioMax = tags.getTagDoubleDefault ("style.ratioMax",1000.);
  double ratioMin = tags.getTagDoubleDefault ("style.ratioMin",0.);
  double ratioMaxQerr = tags.getTagDoubleDefault ("style.ratioMaxQerr",std::numeric_limits<double>::infinity());
  bool forceRatioLimits = tags.getTagBoolDefault ("style.forceRatioLimits",false );
  bool asymmSysErrorBand = tags.getTagBoolDefault("errors.drawAsymmSysMC", false);
  bool asymmStatErrorData = tags.getTagBoolDefault("style.data.asymErrors", false);
  bool verbose = tags.getTagBoolDefault("verbose",false);
  bool showXErrors = tags.getTagBoolDefault("style.ratio.showXErrors", false);
  bool showYErrors = tags.getTagBoolDefault("style.ratio.showYErrors", true);

  TH1* hTotalBkg = this->getObject<TH1>("totalBkg");
  if(!hTotalBkg) return;
  TObjArray* histosAsymmSys = 0;
  if (asymmSysErrorBand) {
    histosAsymmSys = this->getObject<TObjArray>("asymmSys");
  }
 
  TPad* ratioPad = this->getPad("sub");
  if(!ratioPad) return;
  ratioPad->cd();
 
  int nBins = hTotalBkg->GetNbinsX();

  int nPoints = 0;
  for (int i = 1; i <= nBins; i++) {
    if (hTotalBkg->GetBinContent(i) != 0.) {
      nPoints++;
    }
  }

  // the graph used to draw the error band on the ratio
  if(verbose) VERBOSEclass("generating ratio error graphs");
  TGraphAsymmErrors * ratioErrorGraph = new TGraphAsymmErrors(nPoints);
  ratioErrorGraph->SetTitle("Monte Carlo ratio error band");
  this->addObject(ratioErrorGraph,"ratioErrorGraph");
  TGraphAsymmErrors * asymmErrorGraph;
  if (asymmSysErrorBand){
    asymmErrorGraph = TQHistogramUtils::getGraph(hTotalBkg, histosAsymmSys);
    this->addObject(asymmErrorGraph,"asymmSysErrorBand");
  }

  bool invertRatio = tags.getTagBoolDefault("style.invertRatio",false);

  int iPoint = 0;
  for (int iBin = 1; iBin <= nBins; iBin++) {
    double MC = hTotalBkg->GetBinContent(iBin);
    double MCErr = hTotalBkg->GetBinError(iBin);
    double MCErrUpper = MCErr;
    double MCErrLower = MCErr;
    if (asymmSysErrorBand) {
      MCErrUpper = asymmErrorGraph->GetErrorYhigh(iBin);
      MCErrLower = asymmErrorGraph->GetErrorYlow(iBin);
    }
    if(MCErrUpper == 0 || MCErrLower == 0 || MC == 0) continue;
    double ratioBandErrorUpper =  MCErrUpper / MC;
    double ratioBandErrorLower =  MCErrLower / MC;
    // set the position and the width of the ratio error band
    ratioErrorGraph->SetPoint(iPoint, hTotalBkg->GetBinCenter(iBin), 1.);
 
    ratioErrorGraph->SetPointError(iPoint, hTotalBkg->GetBinWidth(iBin) / 2.,
                                   hTotalBkg->GetBinWidth(iBin) / 2.,
                                   ratioBandErrorLower, ratioBandErrorUpper);
    // if shape sys turned on we will have asymmetric error
    iPoint++;
  }

  if(verbose) VERBOSEclass("calculating geometry and axis ranges");
  // set the x range of the ratio graph to match the one of the main histogram
  double xLowerLimit = hTotalBkg->GetBinLowEdge(1);
  double xUpperLimit = hTotalBkg->GetBinLowEdge(nBins) + hTotalBkg->GetBinWidth(nBins);
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
  tags.getTagString("labels.numerator",dataLabel);
  TList* graphs = new TList();
  TList* hists = new TList();
  // graphs->SetOwner(true);

  // actual minimum and maximum ratio
  double actualRatioMin = 1.;
  double actualRatioMax = 1.;

  if(verbose) VERBOSEclass("generating ratio graphs");
  // loop over data histograms
  TQTaggableIterator itr(fProcesses);

  if (tags.getTagBoolDefault("useToyData",false))
  {
    // set the x range of the dmb graph to match the one of the main histogram
    TH1 * h_data = TQHistogramUtils::copyHistogram(this->getObject<TH1>("toyData"),"tmp");
    TQTaggableIterator itr2(fProcesses);
    if(dataLabel.IsNull()) dataLabel = h_data->GetTitle();
    hists->Add(h_data);
  }

  //@tag:style.ratio.dropYieldsBelow: suppress points in the ratio graph where either MC or data yield are below this number
  double ratioContentThreshold = tags.getTagDoubleDefault("style.ratio.dropYieldsBelow",0.0001);
  
  while(itr.hasNext()){
    // get the data histogram
    DEBUGclass("next process...");
    TQNamedTaggable* process = itr.readNext();
    if(!process) continue;
    if(!process->getTagBoolDefault(".isData",false)) continue;
    TH1 * h_data = this->getObject<TH1>(this->makeHistogramIdentifier(process));
    if(!h_data) continue;
    if(dataLabel.IsNull()) dataLabel = h_data->GetTitle();
    hists->Add(h_data);

    // calculate the number of valid ratio points: ratio points are considered
    // valid if they have a finite value (MC prediction != 0) (--> nPoints) and
    // the observed data is greater than zero (--> nRatioPoints)
    int nRatioPoints = 0;
    for (int i = 1; i <= nBins; i++) {
      double mcVal = hTotalBkg->GetBinContent(i);
      double dataVal = h_data->GetBinContent(i);
      if (mcVal < ratioContentThreshold || dataVal < ratioContentThreshold) continue;
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
    TGraphAsymmErrors * ratioGraph = new TGraphAsymmErrors(nRatioPoints);
    this->addObject(ratioGraph,TString::Format("ratioGraph_%s",h_data->GetName()));
    ratioGraph->SetTitle(TString::Format("%s (ratio)",h_data->GetTitle()));
    ratioGraph->SetLineColor(h_data->GetLineColor());
    ratioGraph->SetMarkerSize(h_data->GetMarkerSize());
    ratioGraph->SetMarkerStyle(h_data->GetMarkerStyle());
    ratioGraph->SetMarkerColor(h_data->GetMarkerColor());
 
    int iRatioPoint = 0;
 
    // loop over all bins of the histogram
    for (int iBin = 1; iBin <= nBins; iBin++) {
      double x = hTotalBkg->GetBinCenter(iBin);
      // get the values and errors of data and MC for this bin
      double data    = h_data ->GetBinContent(iBin);
      
      if (asymmStatErrorData) {
        h_data->Sumw2(false); //only do this on data, all sum-of-squared-weights information is deleted (sqrt(n) will be used instead)
        h_data->SetBinErrorOption(TH1::kPoisson);
      }
      double dataErrUp = asymmStatErrorData ? h_data->GetBinErrorUp(iBin) : h_data ->GetBinError (iBin);
      double dataErrDown = asymmStatErrorData ? h_data->GetBinErrorLow(iBin) : h_data ->GetBinError (iBin);
      double MC      = hTotalBkg->GetBinContent(iBin);
      // cannot do anything if MC expectation is zero
      if (MC < ratioContentThreshold || data < ratioContentThreshold) continue;
      
      double ratio = invertRatio ? MC / data : data / MC;
      double ratioErrorUp = dataErrUp / MC;
      double ratioErrorDown = dataErrDown / MC;
      if(verbose) VERBOSEclass("adding ratio point with x=%f, y=%f (data=%f, MC=%f)",x,ratio,data,MC);
      ratioGraph->SetPoint(iRatioPoint, x, ratio);
      ratioGraph->SetPointError(iRatioPoint, 
                                showXErrors ? hTotalBkg->GetBinWidth(iBin) / 2. : 0., showXErrors ? hTotalBkg->GetBinWidth(iBin) / 2. : 0.,
                                showYErrors ? ratioErrorDown : 0., showYErrors ? ratioErrorUp : 0.
                                );
      iRatioPoint++;
    }
    if(verbose) VERBOSEclass("completed ratio graph with %d (%d) points",iRatioPoint,ratioGraph->GetN());
    
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

  TString totalBkgLabel = tags.getTagStringDefault ("labels.totalBkg", "SM");
  tags.getTagString("labels.data",dataLabel);

  ratioErrorGraph->GetHistogram()->GetXaxis()->SetTitle(hMaster->GetXaxis()->GetTitle());
  ratioErrorGraph->GetYaxis()->SetTitle(tags.getTagStringDefault("labels.ratio",dataLabel + " / "+ tags.getTagStringDefault ("labels.totalBkg", "SM") +" "));
 
  gStyle->SetEndErrorSize(0); 


  if(verbose) VERBOSEclass("drawing lines");
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
  double fitResultPrintStepY = tags.getTagDoubleDefault("style.ratio.fitSlope.printResults.stepY",1.);
  
  TString ratioDrawStyle("SAME ");
  ratioDrawStyle.Append(tags.getTagStringDefault("style.ratio.drawOption","P e0"));
  

  TQGraphIterator itr2(graphs);
  while(itr2.hasNext()){
    TGraph* ratioGraph = itr2.readNext();
    if(!ratioGraph) continue;

    if(tags.getTagBoolDefault("style.ratio.fitSlope",false) && ratioGraph->GetN() > 1){
      if(verbose) VERBOSEclass("running fit on %s",ratioGraph->GetName());
      ratioGraph->Fit("pol1","EQF","",xLowerLimit,xUpperLimit);
      TF1* f = ratioGraph->GetFunction("pol1");
      f->SetName(TString::Format("%s_fit",ratioGraph->GetName()));
      //      this->addObject(f);
      f->SetLineColor(ratioGraph->GetLineColor());
      f->SetLineWidth(tags.getTagIntegerDefault("style.ratio.fitSlope.lineWidth",1));
      f->SetLineStyle(tags.getTagIntegerDefault("style.ratio.fitSlope.lineStyle",2));
      //@tag:style.ratio.fitSlope.printResults: print the fit results on the ratio canvas
      if (tags.getTagBoolDefault("style.ratio.fitSlope.printResults",false)) {
        l.SetTextColor(ratioGraph->GetLineColor());
        double slope = TQUtils::roundAuto(f->GetParameter(1));
        double slopeErr = TQUtils::roundAuto(f->GetParError(1));
        double chi2 = TQUtils::roundAuto(f->GetChisquare());
        TString s = TString::Format("slope #approx %g #pm %g (#chi^{2}#approx%g)",slope,slopeErr,chi2);
        l.DrawLatex(fitResultPrintPosX,fitResultPrintPosY,s);
        fitResultPrintPosY -= fitResultPrintStepY * textsize;
      }
      //@tag:style.ratio.fitSlope.exportResults: export the fit results as tags on the plotter
      if (tags.getTagBoolDefault("style.ratio.fitSlope.exportResults",false)) {
        this->setTagDouble(TString::Format("export.fitSlope.%s.slope",ratioGraph->GetName()),f->GetParameter(1));
        this->setTagDouble(TString::Format("export.fitSlope.%s.slopeError",ratioGraph->GetName()),f->GetParError(1));
        this->setTagDouble(TString::Format("export.fitSlope.%s.chi2",ratioGraph->GetName()),f->GetChisquare());
      }
    }
    
    ratioGraph->Draw(ratioDrawStyle); 
    
    if(verbose) VERBOSEclass("drawing additional markers");

    this->drawArrows(tags,ratioGraph, actualRatioMin,actualRatioMax,verbose);

  }

  // redraw pad to draw axis on top of error band in ratio plot
  if (ratioPad) {
    ratioPad->RedrawAxis();
  }
} 


//__________________________________________________________________________________|___________


void TQHWWPlotter::drawMultiRatio(TQTaggable& tags){
  // draw a ratio-plot in the sub-pad
  double ratioMax = tags.getTagDoubleDefault ("style.ratioMax",1000.);
  double ratioMin = tags.getTagDoubleDefault ("style.ratioMin",0.);
  double ratioMaxQerr = tags.getTagDoubleDefault ("style.ratioMaxQerr",std::numeric_limits<double>::infinity());
  bool forceRatioLimits = tags.getTagBoolDefault ("style.forceRatioLimits",false );
  bool verbose = tags.getTagBoolDefault("verbose",false);

  double ratioPadScaling = tags.getTagDoubleDefault("geometry.sub.scaling",1.);
  double xscaling = tags.getTagDoubleDefault("geometry.xscaling",1.);
  
  TPad* ratioPad = this->getPad("sub");
  if(!ratioPad) return;
  ratioPad->cd();
 
  if(verbose) VERBOSEclass("calculating geometry and axis ranges"); 
  TString dataLabel("");
  TList* graphs = new TList();
  // graphs->SetOwner(true);

  // actual minimum and maximum ratio
  double actualRatioMin = 1.;
  double actualRatioMax = 1.;

  if(verbose) VERBOSEclass("generating ratio graphs");
  // loop over data histograms
  TQTaggableIterator numerator_processes(fProcesses);
  TQTaggableIterator denominator_processes(fProcesses);
  TString axtitle;
  double xmin = 0;
  double xmax = 0;
  int nbins = 0;
  while(numerator_processes.hasNext() && denominator_processes.hasNext()){
    DEBUGclass(" in the loop");
    // get the data histogram
    TQNamedTaggable* denominator = NULL;
    TQNamedTaggable* numerator = NULL;
    while(!numerator &&  numerator_processes.hasNext()){
      DEBUGclass("Numinator");
      TQNamedTaggable* next = numerator_processes.readNext();
      //if(next->getTagBoolDefault(".isData",false)) data = next;
      if(next->getTagBoolDefault(".isNumerator",false)) numerator = next;
    }
    
    while(!denominator && denominator_processes.hasNext()){
    //while(!prediction){
      DEBUGclass(" Denominator ");
      TQNamedTaggable* next = denominator_processes.readNext();
      if(next->getTagBoolDefault(".isDenominator",false)) denominator = next;
      // if(next->getTagBoolDefault(".isSignal",false)) prediction = next;
      //if(next->getTagBoolDefault(".isBackground",false)) prediction = next;
      //prediction_processes.reset();
    }

    // TQNamedTaggable* denominator = next_1;
    if(!numerator || !denominator) continue;

    if(verbose) VERBOSEclass("drawing comparsion between %s and %s",numerator->GetName(), denominator->GetName());
    
    TH1 * h_numerator = this->getObject<TH1>(this->makeHistogramIdentifier(numerator));
    TH1 * h_denominator = this->getObject<TH1>(this->makeHistogramIdentifier(denominator));
    if(!h_numerator) continue;
    if(!h_denominator) continue;
    if(axtitle.IsNull()){
      axtitle = h_numerator->GetXaxis()->GetTitle();
      xmin = h_numerator->GetXaxis()->GetXmin();
      xmax = h_numerator->GetXaxis()->GetXmax();
      nbins = h_numerator->GetNbinsX();
    }

    // calculate the number of valid ratio points: ratio points are considered
    // valid if they have a finite value (MC prediction != 0) (--> nPoints) and
    // the observed data is greater than zero (--> nRatioPoints)
    int nRatioPoints = 0;
    for (int i = 1; i <= h_numerator->GetNbinsX(); i++) {
      double denVal = h_denominator->GetBinContent(i);
      double numVal = h_numerator->GetBinContent(i);
      if(denVal == 0) continue;
      if(numVal == 0) continue;
      if(!TQUtils::isNum(denVal)){
        WARNclass("encountered non-numeric denominator value: %f",denVal);
        continue;
      }
      if(!TQUtils::isNum(numVal)){
        WARNclass("encountered non-numeric numerator value: %f",numVal);
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
    for (int iBin = 1; iBin <= h_numerator->GetNbinsX(); iBin++) {
      double x = h_denominator->GetBinCenter(iBin);
      // get the values and errors of data and MC for this bin
      double num = h_numerator ->GetBinContent(iBin);
      double numErr = h_numerator ->GetBinError (iBin);
      double den = h_denominator->GetBinContent(iBin);
      // cannot do anything if MC expectation is zero
      if (den == 0. || num <= 0.) continue;
 
      double ratio = num / den;
      double ratioError = ratio * numErr / num;
      if(verbose) VERBOSEclass("adding ratio point with x=%f, y=%f (numerator=%f, denominator=%f)",x,ratio,num,den);
      ratioGraph->SetPoint(iRatioPoint, x, ratio);
      ratioGraph->SetPointError(iRatioPoint, 0., ratioError);
      iRatioPoint++;
    }
 
    this->applyStyle(tags   ,ratioGraph,"sub.data",1.,ratioPadScaling);
    
    double ratioMinAllowed = tags.getTagDoubleDefault ("style.ratioMinAllowed",ratioMin);
    double ratioMaxAllowed = tags.getTagDoubleDefault ("style.ratioMaxAllowed",ratioMax);
    actualRatioMin=ratioMinAllowed;
    actualRatioMax=ratioMaxAllowed;
    if(verbose) VERBOSEclass("drawMultiRatio: allowed range of ratio graph: %f -- %f",actualRatioMin,actualRatioMax);

    this->estimateRangeY(ratioGraph,actualRatioMin,actualRatioMax,ratioMaxQerr);
 
    if(verbose) VERBOSEclass("drawMultiRatio: estimated range of ratio graph: %f -- %f (ratioMaxQerr=%f)",actualRatioMin,actualRatioMax,ratioMaxQerr);

    if(actualRatioMin == actualRatioMax){
      if(verbose) VERBOSEclass("expanding multi ratio to not be empty");
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
 
    if(verbose) VERBOSEclass("drawMultiRatio: final of ratio graph: %f -- %f",actualRatioMin,actualRatioMax);
 
    DEBUGclass(" ratio ");
    graphs->Add(ratioGraph);
  }

  if(verbose) VERBOSEclass("built %d ratio graphs",graphs->GetEntries());

  TString label = tags.getTagStringDefault("labels.ratio","ratio");

  gStyle->SetEndErrorSize(0); 

  if(verbose) VERBOSEclass("drawing graphs, range is %g < x %g",xmin,xmax);

  TQGraphErrorsIterator itr2(graphs);
  bool first = true;
  while(itr2.hasNext()){
    TGraphErrors* ratioGraph = itr2.readNext();
    if(!ratioGraph) continue;
    ratioGraph->SetMinimum(actualRatioMin);
    ratioGraph->SetMaximum(actualRatioMax);
    DEBUGclass(" in the loop iiter next");
    ratioGraph->Draw(first ? "AP" : "P SAME");
    if(first){
      ratioGraph->GetYaxis()->SetTitle(label);
      ratioGraph->GetXaxis()->SetTitle(axtitle);
      ratioGraph->GetXaxis()->Set(nbins,xmin,xmax);
      this->applyGeometry(tags,ratioGraph,"sub" ,xscaling,ratioPadScaling);
    }
    first = false;
    this->drawArrows(tags,ratioGraph, actualRatioMin,actualRatioMax,verbose);
  }
} 

/*
void TQHWWPlotter::drawMultiRatio(TQTaggable& tags){
  // draw a ratio-plot in the sub-pad
  double ratioMax = tags.getTagDoubleDefault ("style.ratioMax",1000.);
  double ratioMin = tags.getTagDoubleDefault ("style.ratioMin",0.);
  double ratioMaxQerr = tags.getTagDoubleDefault ("style.ratioMaxQerr",std::numeric_limits<double>::infinity());
  bool forceRatioLimits = tags.getTagBoolDefault ("style.forceRatioLimits",false );
  bool verbose = tags.getTagBoolDefault("verbose",false);

  double ratioPadScaling = tags.getTagDoubleDefault("geometry.sub.scaling",1.);
  double xscaling = tags.getTagDoubleDefault("geometry.xscaling",1.);
  
  TPad* ratioPad = this->getPad("sub");
  if(!ratioPad) return;
  ratioPad->cd();
 
  if(verbose) VERBOSEclass("calculating geometry and axis ranges");
 
  TString dataLabel("");
  TList* graphs = new TList();
  // graphs->SetOwner(true);

  // actual minimum and maximum ratio
  double actualRatioMin = 1.;
  double actualRatioMax = 1.;

  if(verbose) VERBOSEclass("generating ratio graphs");
  // loop over data histograms
  TQTaggableIterator data_processes(fProcesses);
  TQTaggableIterator prediction_processes(fProcesses);
  TString axtitle;
  double xmin = 0;
  double xmax = 0;
  
  while(data_processes.hasNext() && prediction_processes.hasNext()){
    // get the data histogram
    TQNamedTaggable* data = NULL;
    while(!data &&  data_processes.hasNext()){
      TQNamedTaggable* next = data_processes.readNext();
      if(next->getTagBoolDefault(".isData",false)) data = next;
    }
    TQNamedTaggable* prediction = NULL;
    while(!prediction && prediction_processes.hasNext()){
      TQNamedTaggable* next = prediction_processes.readNext();
      if(next->getTagBoolDefault(".isSignal",false)) prediction = next;
    }
    if(!data || !prediction) continue;

    if(verbose) VERBOSEclass("drawing comparsion between %s and %s",data->GetName(), prediction->GetName());
    
    TH1 * h_data = this->getObject<TH1>(this->makeHistogramIdentifier(data));
    TH1 * h_prediction = this->getObject<TH1>(this->makeHistogramIdentifier(prediction));
    if(!h_data) continue;
    if(!h_prediction) continue;

    if(axtitle.IsNull()){
      axtitle = h_data->GetXaxis()->GetTitle();
      xmin = h_data->GetXaxis()->GetXmin();
      xmax = h_data->GetXaxis()->GetXmax();
    }

    // calculate the number of valid ratio points: ratio points are considered
    // valid if they have a finite value (MC prediction != 0) (--> nPoints) and
    // the observed data is greater than zero (--> nRatioPoints)
    int nRatioPoints = 0;
    for (int i = 1; i <= h_data->GetNbinsX(); i++) {
      double mcVal = h_prediction->GetBinContent(i);
      double dataVal = h_data->GetBinContent(i);
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
    this->addObject(ratioGraph,TString::Format("ratioGraph_%s",h_data->GetName()));
    ratioGraph->SetTitle(TString::Format("%s (ratio)",h_data->GetTitle()));
    ratioGraph->SetLineColor(h_data->GetLineColor());
    ratioGraph->SetMarkerSize(h_data->GetMarkerSize());
    ratioGraph->SetMarkerStyle(h_data->GetMarkerStyle());
    ratioGraph->SetMarkerColor(h_data->GetMarkerColor());
 
    int iRatioPoint = 0;
 
    // loop over all bins of the histogram
    for (int iBin = 1; iBin <= h_data->GetNbinsX(); iBin++) {
      double x = h_prediction->GetBinCenter(iBin);
      // get the values and errors of data and MC for this bin
      double data = h_data ->GetBinContent(iBin);
      double dataErr = h_data ->GetBinError (iBin);
      double MC = h_prediction->GetBinContent(iBin);
      // cannot do anything if MC expectation is zero
      if (MC == 0. || data <= 0.) continue;
 
      double ratio = data / MC;
      double ratioError = ratio * dataErr / data;
      if(verbose) VERBOSEclass("adding ratio point with x=%f, y=%f (data=%f, MC=%f)",x,ratio,data,MC);
      ratioGraph->SetPoint(iRatioPoint, x, ratio);
      ratioGraph->SetPointError(iRatioPoint, 0., ratioError);
      iRatioPoint++;
    }
 
    this->applyStyle(tags   ,ratioGraph,"sub.data",1.,ratioPadScaling);
    
    double ratioMinAllowed = tags.getTagDoubleDefault ("style.ratioMinAllowed",ratioMin);
    double ratioMaxAllowed = tags.getTagDoubleDefault ("style.ratioMaxAllowed",ratioMax);
    actualRatioMin=ratioMinAllowed;
    actualRatioMax=ratioMaxAllowed;
    if(verbose) VERBOSEclass("drawMultiRatio: allowed range of ratio graph: %f -- %f",actualRatioMin,actualRatioMax);

    this->estimateRangeY(ratioGraph,actualRatioMin,actualRatioMax,ratioMaxQerr);
 
    if(verbose) VERBOSEclass("drawMultiRatio: estimated range of ratio graph: %f -- %f (ratioMaxQerr=%f)",actualRatioMin,actualRatioMax,ratioMaxQerr);

    if(actualRatioMin == actualRatioMax){
      if(verbose) VERBOSEclass("expanding multi ratio to not be empty");
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
 
    if(verbose) VERBOSEclass("drawMulti Ratio: final of ratio graph: %f -- %f",actualRatioMin,actualRatioMax);
 
    graphs->Add(ratioGraph);
  }

  if(verbose) VERBOSEclass("built %d ratio graphs",graphs->GetEntries());

  TString label = tags.getTagStringDefault("labels.ratio","ratio");

  gStyle->SetEndErrorSize(0); 

  if(verbose) VERBOSEclass("drawing line");

  TQGraphErrorsIterator itr2(graphs);
  bool first = true;
  while(itr2.hasNext()){
    TGraphErrors* ratioGraph = itr2.readNext();
    if(!ratioGraph) continue;
    ratioGraph->SetMinimum(actualRatioMin);
    ratioGraph->SetMaximum(actualRatioMax);
    ratioGraph->Draw(first ? "AL" : "L SAME");
    if(first){
      ratioGraph->GetYaxis()->SetTitle(label);
      ratioGraph->GetXaxis()->SetTitle(axtitle);
      ratioGraph->GetXaxis()->SetRangeUser(xmin,xmax);
      this->applyGeometry(tags,ratioGraph,"sub" ,xscaling,ratioPadScaling);
    }
    first = false;
    this->drawArrows(tags,ratioGraph, actualRatioMin,actualRatioMax,verbose);
  }
} 

*/

void TQHWWPlotter::drawArrows(TQTaggable &tags,TGraph *ratioGraph, double actualRatioMin, double actualRatioMax, bool verbose = false){
  // Check if the red arrows should be drawn
  if(!tags.getTagBoolDefault("style.showArrows",true)) return;

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

void TQHWWPlotter::drawOptScan(TQTaggable& tags){
  // draw a cut optimization scan in the sub-pad
  double xscaling = tags.getTagDoubleDefault("geometry.xscaling",1.);
  double subPadScaling = tags.getTagDoubleDefault("geometry.sub.scaling",1.);
  bool verbose = tags.getTagBoolDefault("verbose",false);

  TH1* hTotalBkg = this->getObject<TH1>("totalBkg");
  TQTaggableIterator itr(this->fProcesses);
  TH1* hSig = NULL;
  while(itr.hasNext() && !hSig){
    TQNamedTaggable* process = itr.readNext();
    if(!process) continue;
    if(process->getTagBoolDefault(".isSignal",false)){
      hSig = this->getObject<TH1>(this->makeHistogramIdentifier(process));
    }
  }
 

  if(!hTotalBkg){
    return;
  }
  if(!hSig){ 
    return;
  }

  TPad* optscanPad = this->getPad("sub");
  if(!optscanPad){
    ERRORclass("cannot draw, no subpad!");
    return;
  }
  optscanPad->cd();

  // no grid
  optscanPad->SetGridy(0);

  // figure of merit (FOM) plot. e.g. S/sqrt(B) and etc. one can implement more
  TH1* hFOMl = 0;
  TH1* hFOMr = 0;
  TH1* hFOM = 0;

  //@tag:optScan.FOMmode: figure of merit to be used. currently available are: s/sqrt(s+b),s/b,poisson,s/sqrt(b),s/sqrt(b+db2)
  //@tag:optScan.FOMbbb: evaluate the figure-of-merit bin-by-bin instead of integrated left and right (default:false)
  //@tag:style.FOMmode: deprecated, use optScan.FOMmode
  //@tag:style.FOMbbb: deprecated, use optScan.FOMbbb
  TString fommodestr = tags.getTagStringDefault("optScan.FOMmode",tags.getTagStringDefault ("style.FOMmode","s/sqrt(b)"));
  TQHistogramUtils::FOM FOMmode = TQHistogramUtils::readFOM(fommodestr);
  if(FOMmode == TQHistogramUtils::kUndefined){
    WARNclass("unknown figure of merit '%s'!",fommodestr.Data());
    return;
  }
  bool binByBin = tags.getTagBoolDefault("optScan.FOMbbb",tags.getTagBoolDefault ("style.FOMbbb",false));
  bool drawLegend = !binByBin;

  double actualmax = 0;
  if(binByBin){
    if(verbose){
      VERBOSEclass("drawing bin-by-bin significances with FOM=%s for S=%s and B=%s",TQHistogramUtils::getFOMTitle(FOMmode).Data(),hSig->GetTitle(),hTotalBkg->GetTitle());
    }
    hFOM = TQHistogramUtils::getFOMHistogram(FOMmode,hSig, hTotalBkg);
    actualmax = hFOM->GetMaximum() * tags.getTagDoubleDefault("optScan.enlargeY",1.3);
    this->addObject(hFOM,"hist_FOM_bbb");
  } else {
    if(verbose){
      VERBOSEclass("drawing optimization scan with FOM=%s for S=%s and B=%s",TQHistogramUtils::getFOMTitle(FOMmode).Data(),hSig->GetTitle(),hTotalBkg->GetTitle());
    }
    hFOMl = TQHistogramUtils::getFOMScan(FOMmode,hSig, hTotalBkg, true,0.05,verbose);
    hFOMr = TQHistogramUtils::getFOMScan(FOMmode,hSig, hTotalBkg, false,0.05,verbose);
    //@tag:optScan.autoselect: select the optimization scan (left,right) that is better suited for every histogram and only show that one (default:false)
    if(tags.getTagBoolDefault("optScan.autoselect",false)){
      if(verbose) VERBOSEclass("autoselecting opt scan");
      if(TQHistogramUtils::isGreaterThan(hFOMr,hFOMl)){
        if(verbose) VERBOSEclass("removing left-hand FOM histogram");
        this->addObject(hFOMr,"hist_FOM");
        delete hFOMl;
        hFOMl = NULL;
        actualmax = hFOMr->GetMaximum() * tags.getTagDoubleDefault("optScan.enlargeY",1.3);
      } else if(TQHistogramUtils::isGreaterThan(hFOMl,hFOMr)){
        if(verbose) VERBOSEclass("removing right-hand FOM histogram");
        this->addObject(hFOMl,"hist_FOM");
        delete hFOMr;
        hFOMr = NULL;
        actualmax = hFOMl->GetMaximum() * tags.getTagDoubleDefault("optScan.enlargeY",1.3);
      } else {
        if(verbose) VERBOSEclass("not removing FOM histogram");
        this->addObject(hFOMl,"hist_FOM_left");
        this->addObject(hFOMr,"hist_FOM_right");
        actualmax = std::max(hFOMl->GetMaximum(),hFOMr->GetMaximum()) * tags.getTagDoubleDefault("optScan.enlargeY",1.3);
      }
    } else {
      if(verbose) VERBOSEclass("using all opt scans");
      this->addObject(hFOMl,"hist_FOM_left");
      this->addObject(hFOMr,"hist_FOM_right");
      actualmax = std::max(hFOMl->GetMaximum(),hFOMr->GetMaximum()) * tags.getTagDoubleDefault("optScan.enlargeY",1.3);
    }
  }

  bool first = true;
  // set style
  if (hFOM) {
    //@tag:style.optScan.default.*: control styling of the auto-selected FOM graph
    if(verbose) VERBOSEclass("drawing FOM histogram");
    this->applyStyle (tags,hFOM,"optScan.default",xscaling,subPadScaling);
    this->applyGeometry(tags,hFOM,"sub" ,xscaling,subPadScaling);
    hFOM->SetMaximum(actualmax);
    hFOM->SetMinimum(0);
    hFOM->SetNdivisions(50008);
    hFOM->GetYaxis()->SetNdivisions(50004);
    hFOM->Draw(first ? "HIST" : "HIST SAME");
    first = false;
  }
  if (hFOMl) {
    //@tag:style.optScan.left.*: control styling of the left-hand-side (lower) FOM graph
    if(verbose) VERBOSEclass("drawing FOM histogram (lhs)");
    this->applyStyle (tags,hFOMl,"optScan.left",xscaling,subPadScaling);
    this->applyGeometry(tags,hFOMl,"sub" ,xscaling,subPadScaling);
    hFOMl->SetFillStyle(0);
    hFOMl->SetMaximum(actualmax);
    hFOMl->SetNdivisions(50008);
    hFOMl->GetYaxis()->SetNdivisions(50004);
    hFOMl->SetMinimum(0);
    hFOMl->Draw(first ? "HIST" : "HIST SAME");
    first = false;
  }
  if (hFOMr) {
    //@tag:style.optScan.right.*: control styling of the right-hand-side (upper) FOM graph
    if(verbose) VERBOSEclass("drawing FOM histogram (rhs)");
    this->applyStyle (tags,hFOMr,"optScan.right",xscaling,subPadScaling);
    this->applyGeometry(tags,hFOMr,"sub" ,xscaling,subPadScaling);
    hFOMr->SetFillStyle(0);
    hFOMr->SetMaximum(actualmax);
    hFOMr->SetNdivisions(50008);
    hFOMr->GetYaxis()->SetNdivisions(50004);
    hFOMr->SetMinimum(0);
    hFOMr->Draw(first ? "HIST" : "HIST SAME");
    first = false;
  }

  // TLegend
  if(drawLegend){
    TLegend * leg = new TLegend(0.21,0.85,0.93,0.95);
    leg->SetNColumns(2);
    leg->SetFillColor(0);
    leg->SetFillStyle(0);
    leg->SetLineColor(0);
    leg->SetLineWidth(0);
    
    if (FOMmode == TQHistogramUtils::kSoSqB){
      if(hFOMr){
        double rmax = hFOMr->GetBinContent(hFOMr->GetMaximumBin());
        double rmaxxval = hFOMr->GetBinLowEdge(hFOMr->GetMaximumBin()) + hFOMr->GetBinWidth(hFOMr->GetMaximumBin());
        leg->AddEntry(hFOMr,TString::Format("#leftarrow cut Max=%.2f (%.2f)",rmax,rmaxxval), "l");
      }
      if(hFOMl){
        double lmax = hFOMl->GetBinContent(hFOMl->GetMaximumBin());
        double lmaxxval = hFOMl->GetBinLowEdge(hFOMl->GetMaximumBin());
        leg->AddEntry(hFOMl,TString::Format("#rightarrow cut Max=%.2f (%.2f)",lmax,lmaxxval), "l");
      }
    } else {
      if(hFOMl) leg->AddEntry(hFOMl,"#rightarrow cut", "l");
      if(hFOMr) leg->AddEntry(hFOMr,"#leftarrow cut" , "l");
    }
    leg->Draw("SAME");
  }

}

//__________________________________________________________________________________|___________

void TQHWWPlotter::drawDataMinusBackground(TQTaggable& tags){
  // draw a data-minus-background plot in the sub-pad

  TH1* hTotalBkg = this->getObject<TH1>("totalBkgOnly");
  if(!hTotalBkg) return;
 
  TPad* dmbPad = this->getPad("sub");
  if(!dmbPad) return;
  dmbPad->cd();
 
  // set the x range of the dmb graph to match the one of the main histogram
  double dmbPadScaling = tags.getTagDoubleDefault("geometry.sub.scaling",1.);
 
  TString dataLabel("");
  TList* graphs = new TList();
  std::vector<TString> drawOptions;
  // graphs->SetOwner(true);
 
  // loop over data histograms
  TQTaggableIterator itr(fProcesses);
  while(itr.hasNext()){
    // get the data histogram
    TQNamedTaggable* process = itr.readNext();
    if(!process) continue;
    if(!process->getTagBoolDefault(".isData",false)) continue;
    TH1 * h_data = this->getObject<TH1>(this->makeHistogramIdentifier(process));
    h_data = TQHistogramUtils::copyHistogram(h_data,"h_data_minus_bkg");
    if(!h_data) continue;
    this->applyStyle(tags,h_data,"sub.data",1.,dmbPadScaling);
    this->applyGeometry(tags,h_data,"sub",1.,dmbPadScaling);
    h_data->Add(hTotalBkg,-1.);
    graphs->Add(h_data);
    drawOptions.push_back("PE");
    if(dataLabel.IsNull()) dataLabel = h_data->GetTitle();
  }

  TQTaggableIterator itrSig(fProcesses);
  while(itrSig.hasNext()){
    TQNamedTaggable* process = itrSig.readNext();
    if(!process) continue;
    if(!process->getTagBoolDefault(".isSignal",false)) continue;
    TH1* h_sig = this->getObject<TH1>(this->makeHistogramIdentifier(process));
    if(!h_sig) continue;
    TH1* h_sig_new = TQHistogramUtils::copyHistogram(h_sig,TString::Format("%s_dminusb",h_sig->GetName()));
    if(!h_sig_new) continue;
    graphs->Add(h_sig);
    drawOptions.push_back("HIST");
  }
  
  gStyle->SetEndErrorSize(0); 
  
  int i=0;
  TQIterator itr2(graphs);
  while(itr2.hasNext()){
    TH1* hist = dynamic_cast<TH1*>(itr2.readNext());
    if(!hist) continue;
    if(i==0){
      hist->GetYaxis()->SetTitle("Data-Bkg.");
      hist->Draw(drawOptions[0]);
    } else {
      hist->Draw(drawOptions[i]+" SAME");
    }
    i++;
  }
  
  // if 0. is included in the range of the y axis of the dmb plot...
  if (TQHistogramUtils::getMin(graphs) <= 0. && TQHistogramUtils::getMax(graphs) >= 0.){
    // draw the red line indicating 0 in the dmb
    TLine * line = new TLine(TQHistogramUtils::getAxisXmin(hTotalBkg), 0., TQHistogramUtils::getAxisXmax(hTotalBkg), 0.);
    line->SetLineColor(kRed);
    line->Draw();
  }


} 


//__________________________________________________________________________________|___________

void TQHWWPlotter::drawDataMinusBackgroundOverData(TQTaggable& tags){
  // draw a data-minus-background plot in the sub-pad

  TH1* hTotalBkg = TQHistogramUtils::copyHistogram(this->getObject<TH1>("totalBkgOnly"),"tmpBkg");
  if(!hTotalBkg) return;
 
  TPad* dmbPad = this->getPad("sub");
  if(!dmbPad) return;
  dmbPad->cd();
 
  // set the x range of the dmb graph to match the one of the main histogram
  double dmbPadScaling = tags.getTagDoubleDefault("geometry.sub.scaling",1.);
 
  TString dataLabel("");
  TList* graphs = new TList();
  // graphs->SetOwner(true);
 

  double ratioMax = tags.getTagDoubleDefault ("style.ratioMax",1000.);
  double ratioMin = tags.getTagDoubleDefault ("style.ratioMin",0.);
  double ratioMaxQerr = tags.getTagDoubleDefault ("style.ratioMaxQerr",std::numeric_limits<double>::infinity());
  double actualRatioMin = 1.;
  double actualRatioMax = 1.;
    double ratioMinAllowed = tags.getTagDoubleDefault ("style.ratioMinAllowed",ratioMin);
    double ratioMaxAllowed = tags.getTagDoubleDefault ("style.ratioMaxAllowed",ratioMax);
    actualRatioMin=ratioMinAllowed;
    actualRatioMax=ratioMaxAllowed;
  bool forceRatioLimits = tags.getTagBoolDefault ("style.forceRatioLimits",false );

  // loop over data histograms
  TQTaggableIterator itr(fProcesses);
  if (tags.getTagBoolDefault("useToyData",false))
  {
    TH1 * h_data = TQHistogramUtils::copyHistogram(this->getObject<TH1>("toyData"),"tmp");
    this->applyStyle(tags,h_data,"sub.data",1.,dmbPadScaling);
    this->applyGeometry(tags,h_data,"sub",1.,dmbPadScaling);
    TQTaggableIterator itr2(fProcesses);
    if (tags.getTagBoolDefault("style.stackSignal",false))
    {
      while(itr2.hasNext()){
        TQNamedTaggable* process2 = itr2.readNext();
        if(process2->getTagBoolDefault(".isSignal",false) == false) continue;
          hTotalBkg->Add(this->getObject<TH1>(this->makeHistogramIdentifier(process2)),-1.);
      }
    }
    h_data->Add(hTotalBkg,-1.);
    h_data->Divide(this->getObject<TH1>("toyData"));
    graphs->Add(h_data);
    if(dataLabel.IsNull()) dataLabel = h_data->GetTitle();
  }

  else
  {
  while(itr.hasNext()){
    // get the data histogram
    TQNamedTaggable* process = itr.readNext();
    if(!process) continue;
    if(!process->getTagBoolDefault(".isData",false)) continue;
    TH1 * h_data = this->getObject<TH1>(this->makeHistogramIdentifier(process));
    h_data = TQHistogramUtils::copyHistogram(h_data,"h_data_minus_bkg");
    if(!h_data) continue;
    this->applyStyle(tags,h_data,"sub.data",1.,dmbPadScaling);
    this->applyGeometry(tags,h_data,"sub",1.,dmbPadScaling);
    TQTaggableIterator itr2(fProcesses);
    if (tags.getTagBoolDefault("style.stackSignal",false))
    {
      while(itr2.hasNext()){
        TQNamedTaggable* process2 = itr2.readNext();
        if(!process2->getTagBoolDefault(".isSignal",false)) continue;
          hTotalBkg->Add(this->getObject<TH1>(this->makeHistogramIdentifier(process2)),-1.);
      }
    }
    h_data->Add(hTotalBkg,-1.);
    h_data->Divide(this->getObject<TH1>(this->makeHistogramIdentifier(process)));
    graphs->Add(h_data);
    if(dataLabel.IsNull()) dataLabel = h_data->GetTitle();
  }
  }

  TQTaggableIterator itrSig(fProcesses);
  while(itrSig.hasNext()){
    TQNamedTaggable* process = itrSig.readNext();
    if(!process) continue;
    if(process->getTagBoolDefault(".isSignal",false) == false) continue;
    TH1* h_sig_old = this->getObject<TH1>(this->makeHistogramIdentifier(process));
    if(!h_sig_old) continue;
    // get the data histogram
    TH1* h_sig = TQHistogramUtils::copyHistogram(h_sig_old,"signal_over_bkg");
    if(!h_sig) continue;
    h_sig->Divide(hTotalBkg);
    graphs->Add(h_sig);
  }

  gStyle->SetEndErrorSize(0); 

  bool first = true;


  TQIterator itr2(graphs);
  while(itr2.hasNext()){
    TH1* hist = dynamic_cast<TH1*>(itr2.readNext());
    if(!hist) continue;
    this->estimateRangeY(hist,actualRatioMin,actualRatioMax,ratioMaxQerr);
    if (forceRatioLimits)
      actualRatioMin = ratioMin;
    else 
      actualRatioMin = actualRatioMin-0.1*(actualRatioMax-actualRatioMin);
 
    if (forceRatioLimits)
      actualRatioMax = ratioMax;
    else
      actualRatioMax = actualRatioMax+0.1*(actualRatioMax-actualRatioMin);

    TGraphErrors * ratioGraph = new TGraphErrors(hist);
    if(first){
      hist->GetYaxis()->SetRangeUser(actualRatioMin,actualRatioMax);
      hist->GetYaxis()->SetNdivisions(5);
      hist->GetYaxis()->SetTitle("(Data-Bkg)/Data");
      hist->Draw("PE");
      first = false;
    }
    hist->Draw("PE SAME"); 
    this->drawArrows(tags,ratioGraph, actualRatioMin,actualRatioMax,false);
  }

  // if 0. is included in the range of the y axis of the dmb plot...
  if (TQHistogramUtils::getMin(graphs) <= 0. && TQHistogramUtils::getMax(graphs) >= 0.){
    // draw the red line indicating 0 in the dmb
    TLine * line = new TLine(TQHistogramUtils::getAxisXmin(hTotalBkg), 0., TQHistogramUtils::getAxisXmax(hTotalBkg), 0.);
    line->SetLineColor(kRed);
    line->Draw();
  }


}



void TQHWWPlotter::drawCustomSubPlot(TQTaggable& tags){
  // draw a significance plot in the sub-pad
  TString totalBkgLabesl = tags.getTagStringDefault ("labels.totalBkg", "SM");
  TString formulaString = tags.getTagStringDefault ("style.ratioFormula","d/b");
  TString formulaName = tags.getTagStringDefault ("style.ratioName",formulaString);
  double min = tags.getTagDoubleDefault("style.customSubMin",0.);
  double max = tags.getTagDoubleDefault("style.customSubMax",0.);
  
  TPad* sigPad = this->getPad("sub");
  if(!sigPad) return;

  TH1* hMaster = this->getObject<TH1>("master");
  TH1* hTotalBkg = this->getObject<TH1>("totalBkg");
  TH1* hTotalSig = this->getObject<TH1>("totalSig");
  if(!hTotalBkg) return;

  int nBins = hMaster->GetNbinsX();
  
  // loop over all histograms
  TQTaggableIterator itr(fProcesses);
  while(itr.hasNext()){
    // get the data histograms only
    TQNamedTaggable* process = itr.readNext();
    if(!process) continue;
    if(!process->getTagBoolDefault(".isData",false)) continue;
    TH1 * hdata = this->getObject<TH1>(this->makeHistogramIdentifier(process));
    if (!hdata) continue;
    
    
    
    // the graph used to draw the values (errors may be suported at a later time)
    //TGraphAsymmErrors * customGraph = new TGraphAsymmErrors(nBins);
    TGraph * customGraph = new TGraph(nBins);
    customGraph->SetLineColor(hdata->GetLineColor());
    customGraph->SetMarkerSize(hdata->GetMarkerSize());
    customGraph->SetMarkerStyle(hdata->GetMarkerStyle());
    customGraph->SetMarkerColor(hdata->GetMarkerColor());
 
    this->addObject(customGraph,TString::Format("custom_%s",hdata->GetName()));
 
    int iPoint = 0;
 
    // actual minimum and maximum value
    double actualSigMin = 0.;
    double actualSigMax = 0.;
 
    bool first = true;
    // loop over all bins of the histogram
    for (int iBin = 1; iBin <= nBins; iBin++) {
      TString localFormula = formulaString;
      
      // retrieve required values and fill them into the formula provided

      //watch the order in which expressions are replaced!!
      //@tags:style.ratioFormula: arbitrary formula that can be defined using the following placeholders: sHighXbinWidth,sLowXbinWidth,sHighXbinWidth,sLow,sHigh,sig,bHighXbinWidth,bLowXbinWidth,bLow,bHigh,bkg,dHighXbinWidth,dLowXbinWidthdLow,dHigh,data,binN,binX,binWidth
      //@tags:style.ratioName: name of the custom subplot to be shown
      localFormula = TQStringUtils::replace(localFormula,"sLowXbinWidth",std::to_string(hTotalSig->Integral(0,iBin,"width")));
      localFormula = TQStringUtils::replace(localFormula,"sHighXbinWidth",std::to_string(hTotalSig->Integral(iBin,nBins+1,"width")));
      localFormula = TQStringUtils::replace(localFormula,"sLow",std::to_string(hTotalSig->Integral(0,iBin)));
      localFormula = TQStringUtils::replace(localFormula,"sHigh",std::to_string(hTotalSig->Integral(iBin,nBins+1)));
      localFormula = TQStringUtils::replace(localFormula,"sig",std::to_string(hTotalSig->GetBinContent(iBin)));
                                                                         
      localFormula = TQStringUtils::replace(localFormula,"bLowXbinWidth",std::to_string(hTotalBkg->Integral(0,iBin,"width")));
      localFormula = TQStringUtils::replace(localFormula,"bHighXbinWidth",std::to_string(hTotalBkg->Integral(iBin,nBins+1,"width")));
      localFormula = TQStringUtils::replace(localFormula,"bLow",std::to_string(hTotalBkg->Integral(0,iBin)));
      localFormula = TQStringUtils::replace(localFormula,"bHigh",std::to_string(hTotalBkg->Integral(iBin,nBins+1)));
      localFormula = TQStringUtils::replace(localFormula,"bkg",std::to_string(hTotalBkg->GetBinContent(iBin)));
                                                                         
      localFormula = TQStringUtils::replace(localFormula,"dLowXbinWidth",std::to_string(hdata->Integral(0,iBin,"width")));
      localFormula = TQStringUtils::replace(localFormula,"dHighXbinWidth",std::to_string(hdata->Integral(iBin,nBins+1,"width")));
      localFormula = TQStringUtils::replace(localFormula,"dLow",std::to_string(hdata->Integral(0,iBin)));
      localFormula = TQStringUtils::replace(localFormula,"dHigh",std::to_string(hdata->Integral(iBin,nBins+1)));
      localFormula = TQStringUtils::replace(localFormula,"data",std::to_string(hdata->GetBinContent(iBin)));
                                                                         
      localFormula = TQStringUtils::replace(localFormula,"binN",std::to_string(iBin));
      localFormula = TQStringUtils::replace(localFormula,"binX",std::to_string(hTotalBkg->GetBinCenter(iBin)));
      localFormula = TQStringUtils::replace(localFormula,"binWidth",std::to_string(hTotalBkg->GetBinWidth(iBin)));

      TFormula frml(formulaName,localFormula);
      if (0 != frml.Compile()) {
        WARNclass("failed to compile formula %s (raw: %s)",localFormula.Data(),formulaString.Data());
        continue;
      }
      double value = frml.Eval(0.);
      // set the position and the width of the significance band
      customGraph->SetPoint(iPoint, hTotalBkg->GetBinCenter(iBin), value);

      //this part might be properly implemented at some time, for now it's irrelevant
      /*
      if (value < 0.) {
        customGraph->SetPointError(
                                         iPoint, hTotalBkg->GetBinWidth(iBin) / 2.,
                                         hTotalBkg->GetBinWidth(iBin) / 2., -value, 0.);
      } else {
        customGraph->SetPointError(
                                         iPoint, hTotalBkg->GetBinWidth(iBin) / 2.,
                                         hTotalBkg->GetBinWidth(iBin) / 2., 0., value);
      }
      */
      if (value < actualSigMin){
        actualSigMin = value;
      }
      if (value > actualSigMax){
        actualSigMax = value;
      }
 
      iPoint++;
 
    }
    if (min < max) {
      actualSigMin = std::max(actualSigMin,min);
      actualSigMax = std::min(actualSigMax,max);
    }
    //@tag: [geometry.xscaling,geomatry.sub.scaling] These argument tags influence the appearance of the subplot. The precise effect is considered 'black magic'.
    double xscaling = tags.getTagDoubleDefault("geometry.xscaling",1.);
    double sigPadScaling = tags.getTagDoubleDefault("geometry.sub.scaling",1.);
    this->applyStyle(tags,customGraph,formulaName,xscaling,sigPadScaling);
    this->applyGeometry(tags,customGraph,"sub" ,xscaling,sigPadScaling);
 
    // set the x range of the ratio graph to match the one of the main histogram
    double xLowerLimit = hTotalBkg->GetBinLowEdge(1);
    double xUpperLimit = hTotalBkg->GetBinLowEdge(nBins) + hTotalBkg->GetBinWidth(nBins);
    customGraph->GetXaxis()->SetLimits(xLowerLimit, xUpperLimit);
 
    // set the titles of the axis of the ratio graph
    customGraph->GetXaxis()->SetTitle(hTotalBkg->GetXaxis()->GetTitle());
 
    // confine the y axis of the ratio plot
    customGraph->GetYaxis()->SetTitle(formulaName);
 
    actualSigMin = TMath::Abs(actualSigMin);
    int y1 = TMath::Nint(actualSigMin);
    if (y1 < actualSigMin)
      actualSigMin = y1 + 0.5;
    else
      actualSigMin = y1;
 
    if (fmod(actualSigMin, 1) == 0)
      actualSigMin += 0.5;
    int y2 = TMath::Nint(actualSigMax);
    if (y2 < actualSigMax)
      actualSigMax = y2 + 0.5;
    else
      actualSigMax = y2;
    if (fmod(actualSigMax, 1) == 0)
      actualSigMax += 0.5;
 
    customGraph->GetHistogram()->SetMinimum(actualSigMin);
    customGraph->GetHistogram()->SetMaximum(actualSigMax);
 
    // draw the graph
    if (first)
      customGraph->Draw("AP2");
    else 
      customGraph->Draw("P2");

    
    sigPad->RedrawAxis();
    sigPad->Update();
    first = false;
  }
} 

//__________________________________________________________________________________|___________

void TQHWWPlotter::drawPull(TQTaggable& tags){
  // draw a pull-plot in the sub-pad
  
  //@tag:[style.pullMax,style.pullMin,style.pullMinAllowed,style.pullMaxAllowed] These argument tags give a suggested range of the y-axis for pull sub-plots. pullMin/MaxAllowed override the other two, unless 'style.forcePullLimits' is set to true, in which case pullMin/Max are hard limits for the subplot's y-axis.
  double pullMax = tags.getTagDoubleDefault ("style.pullMax",1000.);
  double pullMin = tags.getTagDoubleDefault ("style.pullMin",0.);
  //@tag:[style.pullMaxQerr] This argument tag specifies a tolerance to include outlying points when estimating the y-axis range (passed as 'tolerance' argument to TQPlotter::estimateRangeY). Default: std::numeric_limits<double>::infinity.
  double pullMaxQerr = tags.getTagDoubleDefault ("style.pullMaxQerr",std::numeric_limits<double>::infinity());
  //@tag:[style.forcePullLimits] If this argument tag is set to true the y-axis range of the pull sub-plot is enforced to the values given by the tags 'style.pullMin' and 'style.pullMax'. Default: false
  bool forcePullLimits = tags.getTagBoolDefault ("style.forcePullLimits",false );
  bool verbose = tags.getTagBoolDefault("verbose",false);

  TH1* hTotalBkg = this->getObject<TH1>("totalBkg");
  if(!hTotalBkg) return;
 
  TPad* pullPad = this->getPad("sub");
  if(!pullPad) return;
  pullPad->cd();
 
  int nBins = hTotalBkg->GetNbinsX();

  int nPoints = 0;
  for (int i = 1; i <= nBins; i++) {
    if (hTotalBkg->GetBinContent(i) != 0.) {
      nPoints++;
    }
  }

  if(verbose) VERBOSEclass("calculating geometry and axis ranges");
  // set the x range of the pull graph to match the one of the main histogram
  double xLowerLimit = hTotalBkg->GetBinLowEdge(1);
  double xUpperLimit = hTotalBkg->GetBinLowEdge(nBins) + hTotalBkg->GetBinWidth(nBins);
  //@tag:[geometry.sub.scaling] This argument tag sets a scaling factor for the pull sub-plot. Default: 1.
  double pullPadScaling = tags.getTagDoubleDefault("geometry.sub.scaling",1.);

  TString dataLabel("");
  TList* graphs = new TList();
  TList* hists = new TList();
  // graphs->SetOwner(true);

  // actual minimum and maximum pull
  double actualPullMin = 1.;
  double actualPullMax = 1.;

  if(verbose) VERBOSEclass("generating pull graphs");
  // loop over data histograms
  TQTaggableIterator itr(fProcesses);

  while(itr.hasNext()){
    // get the data histogram
    TQNamedTaggable* process = itr.readNext();
    if(!process) continue;
    //@tag:[.isData] This process tag identifies the corresponding histograms as data histograms. Default: false.
    if(!process->getTagBoolDefault(".isData",false)) continue;
    TH1 * h_data = this->getObject<TH1>(this->makeHistogramIdentifier(process));
    if(!h_data) continue;
    if(dataLabel.IsNull()) dataLabel = h_data->GetTitle();
    hists->Add(h_data);

    // calculate the number of valid pull points: pull points are considered
    // valid if they have a finite value (MC prediction != 0) (--> nPoints) and
    // the observed data is greater than zero (--> nPullPoints)
    int nPullPoints = 0;
    for (int i = 1; i <= nBins; i++) {
      double mcErr = hTotalBkg->GetBinError(i);
      double dataErr = h_data->GetBinError(i);
      if(mcErr == 0) continue;
      if(dataErr == 0) continue;
      if(!TQUtils::isNum(hTotalBkg->GetBinContent(i))){
        WARNclass("encountered non-numeric MC value: %f",mcErr);
        continue;
      }
      if(!TQUtils::isNum(h_data->GetBinContent(i))){
        WARNclass("encountered non-numeric data value: %f",dataErr);
        continue;
      }
      nPullPoints++;
    }
 
    if(nPullPoints < 1){
      // there is nothing to draw -- well, let's do nothing, then
      continue;
    }
 
    // the graph used to draw the pull points
    TGraphErrors * pullGraph = new TGraphErrors(nPullPoints);
    this->addObject(pullGraph,TString::Format("pullGraph_%s",h_data->GetName()));
    pullGraph->SetTitle(TString::Format("%s (pull)",h_data->GetTitle()));
    pullGraph->SetLineColor(h_data->GetLineColor());
    pullGraph->SetMarkerSize(h_data->GetMarkerSize());
    pullGraph->SetMarkerStyle(h_data->GetMarkerStyle());
    pullGraph->SetMarkerColor(h_data->GetMarkerColor());
 
    int iPullPoint = 0;
 
    // loop over all bins of the histogram
    for (int iBin = 1; iBin <= nBins; iBin++) {
      double x = hTotalBkg->GetBinCenter(iBin);
      // get the values and errors of data and MC for this bin
      double data = h_data ->GetBinContent(iBin);
      double mc = hTotalBkg->GetBinContent(iBin);
      double value = data - mc;
      double error2 = pow(h_data->GetBinContent(iBin),2) + pow(hTotalBkg->GetBinContent(iBin),2);
      if(verbose) VERBOSEclass("adding pull point with x=%f, v=%f, e=%f (data=%f, MC=%f)",x,value,sqrt(error2),data,mc);
      pullGraph->SetPoint(iPullPoint, x, value/sqrt(error2));
      iPullPoint++;
    }
 
    this->applyStyle(tags   ,pullGraph,"sub.data",1.,pullPadScaling);
    //tag documentation see above
    double pullMinAllowed = tags.getTagDoubleDefault ("style.pullMinAllowed",pullMin);
    double pullMaxAllowed = tags.getTagDoubleDefault ("style.pullMaxAllowed",pullMax);
    actualPullMin=pullMinAllowed;
    actualPullMax=pullMaxAllowed;
    if(verbose) VERBOSEclass("drawPull: allowed range of pull graph: %f -- %f",actualPullMin,actualPullMax);

    this->estimateRangeY(pullGraph,actualPullMin,actualPullMax,pullMaxQerr);
 
    if(verbose) VERBOSEclass("drawPull: estimated range of pull graph: %f -- %f (pullMaxQerr=%f)",actualPullMin,actualPullMax,pullMaxQerr);

    if(actualPullMin == actualPullMax){
      if(verbose) VERBOSEclass("expanding pull to not be empty");
      //TODO: this is not how this works. this is not how any of this works...
      actualPullMin *= 1.1;
      actualPullMax *= 1.1;
    }
    
    if (forcePullLimits)
      actualPullMin = pullMin;
    else 
      actualPullMin = actualPullMin-0.1*(actualPullMax-actualPullMin);
 
    if (forcePullLimits)
      actualPullMax = pullMax;
    else
      actualPullMax = actualPullMax+0.1*(actualPullMax-actualPullMin);
 
    if(verbose) VERBOSEclass("drawPull: final of pull graph: %f -- %f",actualPullMin,actualPullMax);
 
    graphs->Add(pullGraph);
  }
  //@tag:[labels.totalBkg] This argument tag determines the label used for the total (MC) background. Default: "SM".
  TString totalBkgLabel = tags.getTagStringDefault ("labels.totalBkg", "SM");
  //@tag:[labels.data] This argument tag determines the label for data. Default is the title of the data histogram.
  tags.getTagString("labels.data",dataLabel);

  gStyle->SetEndErrorSize(0); 


  if(verbose) VERBOSEclass("drawing line");
  // if 1. is included in the range of the y axis of the pull plot...
  TLine * line = new TLine(xLowerLimit, 0., xUpperLimit, 0.);
  line->SetLineColor(kRed);
  line->Draw();

  if(verbose) VERBOSEclass("drawing additional markers");
  TQGraphErrorsIterator itr2(graphs);
  bool first = true;
  while(itr2.hasNext()){
    TGraphErrors* pullGraph = itr2.readNext();
    if(!pullGraph) continue;
    if(first){
      pullGraph->Draw("AP");
      pullGraph->GetXaxis()->SetRangeUser(hTotalBkg->GetXaxis()->GetXmin(),hTotalBkg->GetXaxis()->GetXmax());
    } else {
      pullGraph->Draw("P SAME");
    }

    this->drawArrows(tags,pullGraph, actualPullMin,actualPullMax,verbose);
  }
} 

