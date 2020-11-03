#include "QFramework/TQPlotter.h"
#include "QFramework/TQSampleDataReader.h"
#include "TCanvas.h"
#include "TH1.h"
#include "TStyle.h"
#include "TROOT.h"
#include "TFile.h"
#include "TDirectory.h"
#include "QFramework/TQNamedTaggable.h"
#include "TLegend.h"
#include "TLegendEntry.h"
#include "TLatex.h"
#include "THStack.h"
#include "TParameter.h"
#include "TMap.h"
#include "TMath.h"
#include "QFramework/TQHistogramUtils.h"
#include "TGraphAsymmErrors.h"
#include "TGraphErrors.h"
#include "QFramework/TQStringUtils.h"
#include "QFramework/TQUtils.h"
#include "TObjArray.h"
#include "TArrow.h"
#include "TLine.h"
#include "TH2.h"
#include "TH3.h"
#include "TGaxis.h"
#include "QFramework/TQIterator.h"
#include "QFramework/TQLibrary.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cmath>

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQPlotter:
//
// The abstract TQPlotter class provides a base class for custom, 
// analysis-specific plotters like the TQHWWPlotter.
// By inheriting from the TQPlotter, a base plotting interface is provided
// for the daughter classes. The only purely virtual function
// that needs to be implemented by the user is the 
// TCanvas* TQPlotter::makePlot
// Other functionality like data management is provided by this base class.
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQPlotter)


//__________________________________________________________________________________|___________

TQPlotter::TQPlotter() :
TQPresenter(),
  pads(new TObjArray()),
  objects(new TDirectory("plotter_tmp",""))
{
  // Constructor of TQPlotter class
}

//__________________________________________________________________________________|___________

TQPlotter::TQPlotter(TQSampleFolder * baseSampleFolder) : 
  TQPresenter(baseSampleFolder),
  pads(new TObjArray()),
  objects(new TDirectory("plotter_tmp",""))
{
  // Constructor of TQPlotter class
}

//__________________________________________________________________________________|___________

TQPlotter::TQPlotter(TQSampleDataReader * dataSource) : 
  TQPresenter(dataSource),
  pads(new TObjArray()),
  objects(new TDirectory("plotter_tmp",""))
{
  // Constructor of TQPlotter class
}


//__________________________________________________________________________________|___________

void TQPlotter::reset() {
  TQPresenter::reset();

  // Reset the plotter
  this->clearObjects();

  /* set the official ATLAS style */
  this->setStyleAtlas();
}

//__________________________________________________________________________________|___________

TString TQPlotter::makeHistogramIdentifier(TQNamedTaggable* process){
  TString name = process->getTagStringDefault(".name",process->getTagStringDefault("name",process->GetName()));
  if(TQStringUtils::isEmpty(name) || process->getTagBoolDefault(".ignoreProcessName",false)){
    return "hist_"+TQStringUtils::makeValidIdentifier(process->exportTagsAsString(),
                                                      TQStringUtils::letters+TQStringUtils::numerals+"_","_");
  } else {
    return "hist_"+TQStringUtils::makeValidIdentifier(name,
                                                      TQStringUtils::letters+TQStringUtils::numerals+"_","_");
  }
}

//__________________________________________________________________________________|___________

bool TQPlotter::addData(TString path, TString options) {
  // add a new data process to the plotter
  TQNamedTaggable* data = new TQNamedTaggable(path);
  data->setTagBool(".isData",true);
  data->setTagString(".legendOptions","lep");
  data->setTagString(".path",path);
  data->importTags(options,true);
  this->fProcesses->Add(data);
  return true;
}


//__________________________________________________________________________________|___________

bool TQPlotter::addBackground(TString path, TString options) {
  // add a new background process to the plotter
  TQNamedTaggable* bkg = new TQNamedTaggable(path);
  bkg->setTagBool(".isBackground",true);
  bkg->setTagString(".path",path);
  bkg->importTags(options,true);
  this->fProcesses->Add(bkg);
  return true;
}


//__________________________________________________________________________________|___________

bool TQPlotter::addSignal(TString path, TString options) {
  // add a new signal process to the plotter
  TQNamedTaggable* sig = new TQNamedTaggable(path);
  sig->setTagBool(".isSignal",true);
  sig->setTagString(".path",path);
  sig->importTags(options,true);
  this->fProcesses->Add(sig);
  return true;
}

//__________________________________________________________________________________|___________

bool TQPlotter::includeSystematics(TQTaggable& tags){
  // include the systematics entry from the systematics folder 
  TString sysID = "";
  bool verbose = tags.getTagBoolDefault("verbose",false);
  if(!tags.getTagString("errors.showSys",sysID)) return false;
  TString histName = "";
  if(!tags.getTagString("input.sys",histName)) return false;
  TQFolder* sysFolder = this->fSystematics->getFolder(sysID);
  if(!sysFolder){
    if(verbose){
      VERBOSEclass("unable to retrieve systematics folder '%s'",sysID.Data());
      this->fSystematics->print();
    }
    return false;
  }
  else if(verbose) VERBOSEclass("successfully retrieved systematics folder '%s'",sysID.Data());
  TH1* hTotalBkg = this->getObject<TH1>("totalBkg");
  if(!hTotalBkg){
    if(verbose) VERBOSEclass("unable to retrieve totalBkg histogram!");
    return false;
  }
  TH1* hSys = TQHistogramUtils::copyHistogram(hTotalBkg,"totalBkgSys");
  TH1* hSysOrig = tags.getTagBoolDefault("errors.shapeSys",true) ? dynamic_cast<TH1*>(sysFolder->getObject(histName)) : NULL;
  if (hSysOrig && TQHistogramUtils::checkConsistency(hTotalBkg,hSysOrig,verbose)) {
    hSys->Multiply(hSysOrig);
  } else {
    if (hSysOrig && !TQHistogramUtils::checkConsistency(hTotalBkg,hSysOrig,verbose)) {
      WARNclass("nominal and systematics histograms are inconsistent!");
    }
    TString cutname = TQFolder::getPathHead(histName);
    TQFolder* cutfolder= sysFolder->getFolder(cutname);
    if(!cutfolder) cutfolder = sysFolder;
    if (!cutfolder->hasTag("~yield")) {
        ERRORclass("unable to retrieve neither shape systematic histogram nor yield for '%s/%s'.", cutname.Data(), histName.Data());
        hSys->Scale(0);
        return false;
    }
    double sys = cutfolder->getTagDoubleDefault("~yield",0);
    hSys->Scale(sys);
    WARNclass("unable to retrieve shape systematic histogram for '%s/%s', using normalization systematic of ~%f instead",cutname.Data(),histName.Data(),sys);
    return true;
  }
  if(verbose) VERBOSEclass("successfully created total background systematics histogram '%s' with integral '%f'",hSys->GetName(),TQHistogramUtils::getIntegral(hSys));
  return true;
}

//__________________________________________________________________________________|___________

TObjArray * TQPlotter::getHistograms(TObjArray* processes, const TString& tagFilter, const TString& histName, const TString& namePrefix, TQTaggable& aliases,TQTaggable& options){
  // retrieve histograms using the internal reader
  // histogram options can be controlled by tags
  
  /* stop if the reader is invalid */
  if (!fReader)
    return 0;
  bool verbose = options.getTagBoolDefault("verbose",false);

  
  /* create the list of histograms */
  TObjArray * histograms = new TObjArray();

  /* loop over the list of processes and get histograms */
  int i = 0;
  TQTaggableIterator itr(processes);
  while(itr.hasNext()){
    /* get the process properties */
    TQNamedTaggable * process = itr.readNext();
    if(!process) continue;
    if(!process->getTagBoolDefault(tagFilter,false)) continue;
    i++;
    
    /* create some consistency */
    //this might look a little weird but does make sense!
//    double scale = options.getTagDoubleDefault(".scale",options.getTagDoubleDefault("scale",1.));
//    options.setTagDouble("scale",scale);
//    options.setTagDouble(".scale",scale);
    
    /* get the path of the histogram */
    TString path = process->getTagStringDefault(".path", "");
    path = aliases.replaceInText(path);
 
    /* check for a predefined histogram name in the process info */
    TString histogramName = process->replaceInText(process->getTagStringDefault("input",histName));
    histogramName = aliases.replaceInText(process->getTagStringDefault("input",histogramName));
 
    if (path.IsNull() || histogramName.IsNull()) continue;
 
    TQTaggable histoOptions;
    if (!namePrefix.IsNull()) {
      histoOptions.setTagString("prefix.name", namePrefix);
    }
    //the order of the following two lines is important, since TQTaggable::exportTags does not overwrite existing tags. However it is desirable to prioritize process specific tags over global ones.
    /* import process specific options */
    process->exportTags(&histoOptions);
    /* import global options */
    options.exportTags(&histoOptions);
    
 
    /* get the histogram */
    TList* sfList = new TList();
    if(verbose) VERBOSEclass("retrieving histogram '%s' from '%s' with options '%s''",histogramName.Data(),path.Data(),histoOptions.exportTagsAsString().Data());
    TH1 * histo = fReader->getHistogram(path, histogramName, &histoOptions,sfList);
    if(histo){
      this->addObject(histo,this->makeHistogramIdentifier(process));
      TString histTitle = "";
      if(process->getTagString( "title",histTitle)) histo->SetTitle(histTitle);
      if(process->getTagString(".title",histTitle)) histo->SetTitle(histTitle);
      TQHistogramUtils::applyStyle(histo,process);
    
      if(TQUtils::isNum(TQHistogramUtils::getIntegral(histo)))
        /* add the histogram to the list */
        histograms->Add(histo);
    } else if(verbose){
      VERBOSEclass("failed to retrieve histogram, skipping");
    }
    if(sfList) delete sfList;
  }
 
  /* return the list of histograms */
  return histograms;
}


//__________________________________________________________________________________|___________

bool TQPlotter::checkConsistency(TH1 * &hMaster, TObjArray * histograms) {
  // check the consistency of an array of histograms with the master histogram
  // will create the master histogram if not present

  TQIterator itr(histograms);
  while(itr.hasNext()){
    // iterate over list and check consistency of histograms
    TH1 * h = dynamic_cast<TH1*>(itr.readNext());
    if (!h) continue;
    if (hMaster) {
      if (!TQHistogramUtils::checkConsistency(hMaster, h) || TMath::IsNaN(TQHistogramUtils::getIntegral(h)))
        return false;
    } else {
      hMaster = TQHistogramUtils::copyHistogram(h,"master");
      hMaster->SetTitle("Main Coordinate System");
    }
  }
 
  /* return if histograms are consistent */
  return true;
}


//__________________________________________________________________________________|___________

void TQPlotter::addHistogramToLegend(TQTaggable& tags, TLegend * legend, TQNamedTaggable* process, const TString& options){
  // add a single histogram to the legend
  if(!process) return;
  // transfer process tags to the options container locally to avoid having to pass around spurious data
  TQTaggable opts(options);
  opts.importTags(process);
  this->addHistogramToLegend(tags,legend,this->makeHistogramIdentifier(process),opts);
}

//__________________________________________________________________________________|___________

void TQPlotter::addHistogramToLegend(TQTaggable& tags, TLegend * legend, const TString& identifier, TQTaggable& options){
  // add a single histogram to the legend
  this->addHistogramToLegend(tags,legend,this->getObject<TH1>(identifier),options);
}


//__________________________________________________________________________________|___________

void TQPlotter::addHistogramToLegend(TQTaggable& tags, TLegend * legend, TH1* histo, const TString& options){
  // add a single histogram to the legend
  TQTaggable opts(options);
  this->addHistogramToLegend(tags,legend,histo,opts);
}

//__________________________________________________________________________________|___________

void TQPlotter::addHistogramToLegend(TQTaggable& tags, TLegend * legend, TH1* histo, TQTaggable& options){
  // add a single histogram to the legend
  bool showMissing = tags.getTagBoolDefault("style.showMissing",true);
  bool showEventYields = tags.getTagBoolDefault("style.showEventYields",false);
  bool showMean = tags.getTagBoolDefault("style.showMean");
  bool showRMS = tags.getTagBoolDefault("style.showRMS");
  bool showUnderOverflow = tags.getTagBoolDefault("style.showEventYields.useUnderOverflow",true);
  bool showEventYieldErrors = tags.getTagBoolDefault("style.showEventYields.showErrors",false);
  bool verbose = tags.getTagBoolDefault("verbose",false);
 
  TString title = options.getTagStringDefault("defaultTitle", "");
  if (histo) title = histo->GetTitle();
  options.getTagString("title", title);
  title = TQStringUtils::convertLaTeX2ROOTTeX(title);

  /* add an entry to the legend */
  if (options.getTagBoolDefault("showInLegend", true)) {
    if (histo) {
      if (showEventYields){
        double err;
        double val = TQHistogramUtils::getIntegralAndError(histo,err,showUnderOverflow);
        if(showEventYieldErrors){
          title.Append(TString::Format(" {%.3g #pm %.3g}", val, err));
        } else {
          title.Append(TString::Format(" {%.5g}", val ));
        }
      }
      if (showMean)
        title.Append(TString::Format("#mu=%.3g", histo->GetMean()));
      if (showRMS)
        title.Append(TString::Format("(%.3g)", histo->GetRMS()));
      if(verbose) VERBOSEclass("adding legend entry '%s', attributed to histogram '%s'",title.Data(),histo->GetName());
      title.Prepend(" ");
      legend->AddEntry(histo, title, options.getTagStringDefault(".legendOptions", "f"));
    } else {
      if (showMissing){
        if(verbose) VERBOSEclass("adding empty legend entry for missing histogram (showMissing=true)");
        title.Prepend(" ");
        legend->AddEntry(new TObject(), title, "");
      }
    }
  } else {
    DEBUGclass("process '%s' is not added to legend (showInLegend=false)",title.Data());
  }
}

//__________________________________________________________________________________|___________

void TQPlotter::addAllHistogramsToLegend(TQTaggable& tags, TLegend * legend, const TString& processFilter,
                                         const TString& options, bool reverse){

  // add all histograms matching the process filter to the legend
  bool verbose = tags.getTagBoolDefault("verbose",false);
  if(verbose) VERBOSEclass("entering function, processFilter='%s'",processFilter.Data());
 
  TQTaggableIterator itr(this->fProcesses->MakeIterator(reverse ? kIterBackward : kIterForward),true);
  while(itr.hasNext()){
    TQNamedTaggable * process = itr.readNext();
    if(!process){
      if(verbose) VERBOSEclass("skipping NULL entry");
    } else {
      if(!processFilter.IsNull() && !process->getTagBoolDefault(processFilter,false)){
        if(verbose) VERBOSEclass("skipping empty legend entry for '%s' - does not match filter '%s'",process->getTagStringDefault(".path",process->GetName()).Data(),processFilter.Data());
      } else {
        this->addHistogramToLegend(tags,legend,process,options);
      }
    }
  }
}


//__________________________________________________________________________________|___________

double TQPlotter::getHistogramUpperLimit(TQTaggable& tags, TList * histograms, double lower, bool includeErrors){
  // calculate the "blocks" (x-axis ranges and corresponding y-values)
  // these are employed to avoid collisions of the bins with labels and other graphic elements (i.e. the legend)

  if (!histograms)
    return 0;
 
  bool logScale = tags.getTagBoolDefault ("style.logScale",false );
 
  double left = 0;
  double maxUpperLimit = 0.;

  int iBlock = 0;
  double block_x = 0;
  double block_y = 100;

  TH1* exampleHist = dynamic_cast<TH1*>(histograms->At(0));
  double xmin = TQHistogramUtils::getAxisXmin(exampleHist);
  double xmax = TQHistogramUtils::getAxisXmax(exampleHist);
 
  if(!(TQUtils::isNum(xmin) && TQUtils::isNum(xmax))) return std::numeric_limits<double>::quiet_NaN();
#ifdef _DEBUG_
  histograms->Print();
#endif

  while(tags.getTag(TString::Format("blocks.x.%d",iBlock),block_x) && tags.getTag(TString::Format("blocks.y.%d",iBlock),block_y)){
    double right = block_x;
    double vetoFrac = block_y;

    double block_min = xmin+left*(xmax-xmin);
    double block_max = xmin+right*(xmax-xmin);

    double max = TQHistogramUtils::getMaximumBinValue(histograms, block_min, block_max, includeErrors);

    double upperLimit = 0.;
    // std::cout << max << ", " << lower << ", " << vetoFrac << std::endl;
    if (logScale) upperLimit =  exp(log(max/lower) / vetoFrac ) * lower;
    else          upperLimit = (max - lower) / vetoFrac + lower;

 
    if (upperLimit > maxUpperLimit) maxUpperLimit = upperLimit;
    left = right;
    iBlock++;
  }

  return maxUpperLimit;
}

//__________________________________________________________________________________|___________

TCanvas * TQPlotter::plot(TString histogram, const TString& inputTags) {
  // plot the given histogram using the given tags
  // the tags are forwarded to and interpreted by the makeplot function
  TQTaggable taggable(inputTags);
  return plot(histogram, taggable);
}

//__________________________________________________________________________________|___________

TCanvas * TQPlotter::plot(TString histogram, TQTaggable* inputTags) {
  // plot the given histogram using the given tags
  // the tags are forwarded to and interpreted by the makeplot function
  TQTaggable tags;
  tags.importTags(inputTags);
  TCanvas* c = this->plot(histogram, tags);
  return c;
}

//__________________________________________________________________________________|___________

TCanvas * TQPlotter::plot(TString histogram, TQTaggable& tags) {
  // plot the given histogram using the given tags
  // the tags are forwarded to and interpreted by the makeplot function
  this->clearObjects();
  //@tags: printProcesses: print processes to the console before plotting
  if(tags.getTagBoolDefault("printProcesses",false)){
    this->printProcesses();
  }
  //@tags: useNamePrefix: prefix histogram names with the variable names. will not affect the look of the plot, but possibly required for elaborate plots to be saved in .C format
  if(tags.getTagBoolDefault("useNamePrefix",true)){
    TString tmp(histogram);
    TString prefix = TQFolder::getPathTail(tmp);
    TQStringUtils::ensureTrailingText(prefix,".");
    tags.importTagsWithoutPrefix(tags,prefix);
  }
  tags.importTags(this);
  TCanvas* c = this->makePlot(histogram, tags);
  //@tags: printObjects: print the objects created for plotting
  if(tags.getTagBoolDefault("printObjects",false)){
    std::cout << TQStringUtils::makeBoldBlue(this->Class()->GetName()) << TQStringUtils::makeBoldWhite(" - objects:") << std::endl;
    this->printObjects();
  }
  //@tags: printLegend: print the legend entries
  if(tags.getTagBoolDefault("printLegend",false)){
    std::cout << TQStringUtils::makeBoldBlue("TQPlotter") << TQStringUtils::makeBoldWhite(" - legend entries:") << std::endl;
    TLegend* leg = this->getObject<TLegend>("legend");
    if(!leg){
      ERRORclass("no legend found!");
    } else {
      TQLegendEntryIterator itr(leg->GetListOfPrimitives());
      while(itr.hasNext()){
        TLegendEntry* entry = itr.readNext();
        if(!entry) continue;
        TObject* obj = entry->GetObject();
        if(obj){
          std::cout << TQStringUtils::makeBoldBlue(TQStringUtils::fixedWidth(obj->Class()->GetName(),20)) << TQStringUtils::makeBoldWhite(TQStringUtils::fixedWidth(obj->GetName(),20))<< '"' << TQStringUtils::makeBoldWhite(entry->GetLabel()) << '"' << std::endl;
        } else {
          std::cout << TQStringUtils::makeBoldRed(TQStringUtils::fixedWidth("NULL",40)) << '"' << TQStringUtils::makeBoldWhite(entry->GetLabel()) << '"' << std::endl;
        }
      }
    }
  }
  //@tags: printStyle: print the style tags active after plotting (includes temporary tags set by the plotter itself)
  if(tags.getTagBoolDefault("printStyle",false))
    tags.printTags();

  return c;
}

//__________________________________________________________________________________|___________

TPad * TQPlotter::getPad(const TString& name){
  // retrieve a pad by name
  if(!this->pads) return NULL;
  if(name.IsNull()) return NULL;
  TQIterator itr(this->pads);
  while(itr.hasNext()){
    TObject* obj = itr.readNext();
    if(!obj) continue;
    if(!TQStringUtils::matches(obj->GetName(),name)) continue;
    TPad* p = dynamic_cast<TPad*>(obj);
    if(!p) return NULL;
    p->cd();
    return p;
  }
  return NULL;
}

//__________________________________________________________________________________|___________

TCanvas * TQPlotter::plot(TString histogram, const char* inputTags) {
  // plot the given histogram using the given tags
  // the tags are forwarded to and interpreted by the makeplot function
  TQTaggable taggable(inputTags);
  TCanvas * c = plot(histogram, taggable);
  return c;
}

//______________________________________________________________________________________________

bool TQPlotter::plotAndSaveAs(const TString& histogram, const TString& saveAs, const TString& inputTags) {
  // plot and save the given histogram using the given tags
  // the tags are forwarded to and interpreted by the makeplot function

  TQTaggable tags(inputTags);
  return this->plotAndSaveAs(histogram,saveAs,tags);
}

//______________________________________________________________________________________________

bool TQPlotter::plotAndSaveAs(const TString& histogram, const TString& saveAs, const char* inputTags) {
  // plot and save the given histogram using the given tags
  // the tags are forwarded to and interpreted by the makeplot function
 
  TQTaggable tags((const TString)(inputTags));
  return this->plotAndSaveAs(histogram,saveAs,tags);
}

//______________________________________________________________________________________________

bool TQPlotter::plotAndSaveAs(const TString& histogram, const TString& saveAs, TQTaggable& inputTags) {
  // plot and save the given histogram using the given tags
  // the tags are forwarded to and interpreted by the makeplot function
  return this->plotAndSaveAs(histogram,saveAs,&inputTags);
}


//______________________________________________________________________________________________

bool TQPlotter::plotAndSaveAs(const TString& histogram, const TString& saveAs, TQTaggable * inputTags) {
  // plot and save the given histogram using the given tags
  // the tags are forwarded to and interpreted by the makeplot function

  TQTaggable tags(inputTags);
  tags.importTags(this);
  tags.setGlobalOverwrite(false);


  //@tags: ensureDirectory: create directories to ensure target path exists
  if(tags.getTagBoolDefault("ensureDirectory",false))
    TQUtils::ensureDirectoryForFile(saveAs);

  TDirectory* tmpObjects = NULL;
  TDirectory* oldDir = gDirectory;
  if(saveAs.EndsWith(".root")){
    tmpObjects = this->objects;
    this->objects = TFile::Open(saveAs,"RECREATE");
  }
  gDirectory = oldDir;
 
  TCanvas * canvas = plot(histogram, tags);
 
  bool success = canvas;
 
  if (success && !tmpObjects) {
    this->pads->SetOwner(true);
    canvas->SaveAs(saveAs.Data());
    //@tags: embedfonts: run external font embedding command on created pdf plots
    if(saveAs.EndsWith(".pdf") && tags.getTagBoolDefault("embedfonts",false)){
      TQLibrary::embedFonts(saveAs);
    }
    if(saveAs.EndsWith(".pdf") || saveAs.EndsWith(".jpg") || saveAs.EndsWith(".png")){
      TString exifinfostring = histogram;
      //@tags: exiftitle: set meta-information as exif string on pdf,jpg and png files
      this->getTagString("exiftitle",exifinfostring);
      tags.getTagString("exiftitle",exifinfostring);
      if(TQLibrary::hasEXIFsupport() && !TQLibrary::setEXIF(saveAs,exifinfostring)){
        ERRORclass("setting EXIF meta-information on %s failed!",saveAs.Data());
      }
    }
  }
  if(success && tmpObjects){
    this->objects->Add(canvas);
    this->objects->Add(this->pads);
    this->objects->Write();
    this->objects->Close();
    this->objects = tmpObjects;
    this->pads = new TObjArray();
    std::cout << "Info in " << this->IsA()->GetName() << ": created file " << saveAs << std::endl;
    this->clearObjects();
  } else if(tags.getTagBoolDefault("deleteObjects",true)){
    //@tags: deleteObjects: control whether plotting objects will be kept in memory after plotting (default: false for plotAndSaveAs, true for plot)
    this->deleteObjects();
    delete canvas;
  } else {
    this->clearObjects();
  }

  return success;
}


//______________________________________________________________________________________________

TQPlotter::~TQPlotter() {
  // Destructor of TQPlotter class:
  // this->clearObjects();
  // if(this->objects) delete this->objects;
  // if(this->pads) delete this->pads;
}


//__________________________________________________________________________________|___________

void TQPlotter::setStyleAtlas() {
  // apply atlas style options to gStyle

  int icol = 0;
  gStyle->SetFrameBorderMode(icol);
  gStyle->SetFrameFillColor(icol);
  gStyle->SetCanvasBorderMode(icol);
  gStyle->SetCanvasColor(icol);
  gStyle->SetPadBorderMode(icol);
  gStyle->SetPadColor(icol);
  gStyle->SetStatColor(icol);

  // set the paper & margin sizes
  gStyle->SetPaperSize(20,26);

  // set margin sizes
  gStyle->SetPadTopMargin(0.05);
  gStyle->SetPadRightMargin(0.05);
  gStyle->SetPadBottomMargin(0.16);
  gStyle->SetPadLeftMargin(0.16);

  // set title offsets (for axis label)
  gStyle->SetTitleXOffset(1.4);
  gStyle->SetTitleYOffset(1.4);

  // use large fonts
  //int font=72; // Helvetica italics
  int font=42; // Helvetica
  double tsize=0.05;
  gStyle->SetTextFont(font);

  gStyle->SetTextSize(tsize);
  gStyle->SetLabelFont(font,"x");
  gStyle->SetTitleFont(font,"x");
  gStyle->SetLabelFont(font,"y");
  gStyle->SetTitleFont(font,"y");
  gStyle->SetLabelFont(font,"z");
  gStyle->SetTitleFont(font,"z");
 
  gStyle->SetLabelSize(tsize,"x");
  gStyle->SetTitleSize(tsize,"x");
  gStyle->SetLabelSize(tsize,"y");
  gStyle->SetTitleSize(tsize,"y");
  gStyle->SetLabelSize(tsize,"z");
  gStyle->SetTitleSize(tsize,"z");

  // use bold lines and markers
  gStyle->SetMarkerStyle(20);
  gStyle->SetMarkerSize(1.2);
  gStyle->SetHistLineWidth((Width_t)2.);
  gStyle->SetLineStyleString(2,"[12 12]"); // postscript dashes
 
  // get rid of X error bars 
  //gStyle->SetErrorX(0.001);
  // get rid of error bar caps
  //gStyle->SetEndErrorSize(0.);

  // do not display any of the standard histogram decorations
  gStyle->SetOptTitle(0);
  //gStyle->SetOptStat(1111);
  gStyle->SetOptStat(0);
  //gStyle->SetOptFit(1111); 
  gStyle->SetOptFit(0);

  // put tick marks on top and RHS of plots
  gStyle->SetPadTickX(1);
  gStyle->SetPadTickY(1);

}


//__________________________________________________________________________________|___________

void TQPlotter::estimateRangeY(TH1* h, double& min, double &max, double tolerance){
  TGraphErrors * g = new TGraphErrors(h);
  estimateRangeY(g,min,max,tolerance);
  delete g;
}

void TQPlotter::estimateRangeY(TGraphErrors* g, double& min, double &max, double tolerance){
  // estimate the y-range of a TGraphErrors
  if(tolerance < 0) tolerance = std::numeric_limits<double>::infinity();
  if(g->GetN() < 1){
    // we can't estimate the range of an empty graph
    return;
  }
  if(g->GetN() < 2){
    // if there's only one point, that is the range;
    double x,y;
    g->GetPoint(0,x,y);
    min = y - g->GetErrorY(0);
    max = y + g->GetErrorY(0);
    return;
  }
  double sumwy = 0;
  double sumw = 0;
  for(size_t i=0; i < (size_t)(g->GetN()); i++){
    double x, y;
    if( i != (size_t)(g->GetPoint((int)i, x, y))) continue;
    DEBUGclass("looking at point %d: x=%f, y=%f",(int)i,x,y);
    if(y < min) continue;
    if(y > max) continue;
    double err = g->GetErrorY(i);
    if(TQUtils::isNum(err) && err > 0){
      double w = pow(err,-2);
      sumw += w;
      sumwy += w*y;
    }
  }
  double ym = sumwy/sumw;
  DEBUGclass("found ym=%f (sumwy=%f, sumw=%f)", ym, sumwy, sumw);
  double sumsigma = 0;
  double sumw2 = 0;
  for(size_t i=0; i < (size_t)(g->GetN()); i++){
    double x, y;
    if( i != (size_t)(g->GetPoint((int)i, x, y))) continue;
    if(y < min) continue;
    if(y > max) continue;
    double err = g->GetErrorY(i);
    if(TQUtils::isNum(err) && err > 0){
      double w = pow(err,-2);
      sumsigma += w * pow(y - ym,2);
      sumw2 += w*w;
    }
  }
  double sy2 = sumw / (sumw * sumw - sumw2) * sumsigma;
  double sy = sqrt(sy2);
  DEBUGclass("found sy2=%f, sy=%f",sy2,sy);


  double tmpmin = ym;
  double tmpmax = ym;
  for(size_t i=0; i < (size_t)(g->GetN()); i++){
    double x, y;
    if( i != (size_t)(g->GetPoint((int)i, x, y))) continue;
    if(y > max) continue;
    if(y < min) continue;
    if(y > ym + tolerance * sy) continue;
    if(y < ym - tolerance * sy) continue;
    if(y > tmpmax) tmpmax = y+g->GetErrorY(i);
    if(y < tmpmin) tmpmin = y-g->GetErrorY(i);
  }
  min = tmpmin;
  max = tmpmax;
}


//__________________________________________________________________________________|___________


void TQPlotter::estimateRangeY(TGraphAsymmErrors* g, double& min, double &max, double tolerance){
  // estimate the y-range of a TGraphErrors
  if(tolerance < 0) tolerance = std::numeric_limits<double>::infinity();
  if(g->GetN() < 1){
    // we can't estimate the range of an empty graph
    return;
  }
  if(g->GetN() < 2){
    // if there's only one point, that is the range;
    double x,y;
    g->GetPoint(0,x,y);
    min = y - g->GetErrorYlow(0);
    max = y + g->GetErrorYhigh(0);
    return;
  }
  double sumwy = 0;
  double sumw = 0;
  for(size_t i=0; i < (size_t)(g->GetN()); i++){
    double x, y;
    if( i != (size_t)(g->GetPoint((int)i, x, y))) continue;
    DEBUGclass("looking at point %d: x=%f, y=%f",(int)i,x,y);
    if(y < min) continue;
    if(y > max) continue;
    double err = sqrt(pow(g->GetErrorYlow(i),2)+pow(g->GetErrorYhigh(i),2));
    if(TQUtils::isNum(err) && err > 0){
      double w = pow(err,-2);
      sumw += w;
      sumwy += w*y;
    }
  }
  double ym = sumwy/sumw;
  DEBUGclass("found ym=%f (sumwy=%f, sumw=%f)", ym, sumwy, sumw);
  double sumsigma = 0;
  double sumw2 = 0;
  for(size_t i=0; i < (size_t)(g->GetN()); i++){
    double x, y;
    if( i != (size_t)(g->GetPoint((int)i, x, y))) continue;
    if(y < min) continue;
    if(y > max) continue;
    double err = sqrt(pow(g->GetErrorYlow(i),2)+pow(g->GetErrorYhigh(i),2));
    if(TQUtils::isNum(err) && err > 0){
      double w = pow(err,-2);
      sumsigma += w * pow(y - ym,2);
      sumw2 += w*w;
    }
  }
  double sy2 = sumw / (sumw * sumw - sumw2) * sumsigma;
  double sy = sqrt(sy2);
  DEBUGclass("found sy2=%f, sy=%f",sy2,sy);


  double tmpmin = ym;
  double tmpmax = ym;
  for(size_t i=0; i < (size_t)(g->GetN()); i++){
    double x, y;
    if( i != (size_t)(g->GetPoint((int)i, x, y))) continue;
    if(y > max) continue;
    if(y < min) continue;
    if(y > ym + tolerance * sy) continue;
    if(y < ym - tolerance * sy) continue;
    if(y > tmpmax) tmpmax = y+g->GetErrorY(i);
    if(y < tmpmin) tmpmin = y-g->GetErrorY(i);
  }
  min = tmpmin;
  max = tmpmax;
}


//__________________________________________________________________________________|___________

void TQPlotter::getRange(TGraphErrors* g, double &xlow, double &xhigh, double &ylow, double &yhigh, bool get_xrange, bool get_yrange, double maxQerr){
  // extract the range from a TGraphErrors
  if(maxQerr < 0) maxQerr = std::numeric_limits<double>::infinity();
  int nx = 0;
  int ny = 0;
  double x;
  double y;
  double sumx = 0;
  double sumx2 = 0;
  double sumy = 0;
  double sumy2 = 0;
  for(size_t i=0; i < (size_t)(g->GetN()); i++){
    if( i == (size_t)(g->GetPoint((int)i, x, y))){
      if((get_xrange || TQUtils::inRange(x, xlow , xhigh)) && (get_yrange || TQUtils::inRange(y, ylow , yhigh))){
        if(get_xrange){
          nx++;
          sumx += x;
          sumx2 += x*x;
        }
        if(get_yrange){
          ny++;
          sumy += y;
          sumy2 += y*y;
        }
      }
    }
  }
  double xmean = sumx/nx;
  double ymean = sumy/ny;
  double xvar = sumx2/nx - pow(sumx/nx,2);
  double yvar = sumy2/ny - pow(sumy/ny,2);
  for(size_t i=0; i < (size_t)(g->GetN()); i++){
    if( i == (size_t)(g->GetPoint((int)i, x, y))){
      if((get_xrange || TQUtils::inRange(x, xlow , xhigh)) && (get_yrange || TQUtils::inRange(y, ylow , yhigh))){
        if(get_xrange){
          if(!TQUtils::isNum(xlow)) xlow = xmean-sqrt(xvar);
          if(!TQUtils::isNum(xhigh)) xhigh = xmean+sqrt(xvar);
          double xm = 0.5*(xhigh + xlow);
          double xd = (xhigh-xlow);
          if(xd < 2*std::numeric_limits<double>::epsilon()) xd = std::numeric_limits<double>::infinity();
          if(TQUtils::inRange(x-g->GetErrorX(i),xm-(xd*maxQerr),xlow)) { xlow = x-g->GetErrorX(i); }
          if(TQUtils::inRange(x+g->GetErrorX(i),yhigh,xm+(xd*maxQerr))){ xhigh = x+g->GetErrorX(i); }
        }
        if(get_yrange){
          if(!TQUtils::isNum(ylow)) ylow = ymean-sqrt(yvar);
          if(!TQUtils::isNum(yhigh)) yhigh = ymean+sqrt(yvar);
          double ym = 0.5*(yhigh + ylow);
          double yd = (yhigh-ylow);
          if(yd < 2*std::numeric_limits<double>::epsilon()) yd = std::numeric_limits<double>::infinity();
          if(TQUtils::inRange(y-g->GetErrorY(i),ym-(yd*maxQerr),ylow)) { ylow = y-g->GetErrorY(i); }
          if(TQUtils::inRange(y+g->GetErrorY(i),yhigh,ym+(yd*maxQerr))){ yhigh = y+g->GetErrorY(i); }
        }
      }
    }
  }
}

//__________________________________________________________________________________|___________

void TQPlotter::setTotalBackgroundErrors(TQTaggable& tags, bool showSysMC, bool showStatMC){
  // the histograms retrieved using the TQSampleDataReader class
  // have the statistical uncertainty filled into bin errors by default
  // this function either sets them to zero (for statMCerrors=false)
  // or adds the systematic uncertainties in quadrature (for sysMCerrors=true)
  bool verbose = tags.getTagBoolDefault("verbose",false);
  TH1* hTotalBkg = this->getObject<TH1>("totalBkg");
  TH1* hTotalBkgSys = this->getObject<TH1>("totalBkgSys");
  if(!hTotalBkg){
    if(verbose) VERBOSEclass("no total background histogram found");
    return;
  }
  if (!showStatMC) {
    /* set bin errors to zero */
    if(verbose) VERBOSEclass("removing statistical errors from total background histogram");
    TQHistogramUtils::resetBinErrors(hTotalBkg);
  }
 
  if (showSysMC){
    if(!hTotalBkgSys) {
      if(verbose) VERBOSEclass("no total background systematics found");
      return;
    }
    /* include systematics */
    if(verbose) VERBOSEclass("adding systematic errors to total background histogram");
    for (int iBin = 1; iBin <= hTotalBkgSys->GetNbinsX(); iBin++) {
      double stat = hTotalBkg->GetBinError(iBin);
      double sysLin = hTotalBkgSys->GetBinContent(iBin);
      double sysSq = hTotalBkgSys->GetBinError(iBin);
      double total = TMath::Sqrt(stat*stat + sysLin*sysLin + sysSq*sysSq);
      hTotalBkg->SetBinError(iBin, total);
    }
  } 
}

//__________________________________________________________________________________|___________

void TQPlotter::addObject(TNamed* obj, const TString& key){
  // add an object to the list of graphics objects maintained by the plotter
  if(!obj) return;
  if(!key.IsNull()) obj->SetName(key);
  if(this->objects->FindObject(obj->GetName())){
    ERRORclass("cannot add object '%s' - an object of this name already exists!",obj->GetName());
  }
  this->objects->Add(obj);
}

//__________________________________________________________________________________|___________

void TQPlotter::addObject(TGraph* obj, TString key){
  // add an object to the list of graphics objects maintained by the plotter
  if(!obj) return;
  if(key.IsNull()) key = obj->GetName();
  obj->SetName("tmpgraph");
  obj->GetHistogram()->SetName(TString::Format("h_%s",key.Data()));
  obj->SetName(key.Data());
  if(this->objects->FindObject(obj->GetName())){
    ERRORclass("cannot add object '%s' - an object of this name already exists!",obj->GetName());
  }
  this->objects->Add(obj);
  obj->GetHistogram()->SetDirectory(NULL);
  DEBUGclass("%s@%#x <=> %s@%#x",obj->GetName(),this->objects,obj->GetHistogram()->GetName(),obj->GetHistogram()->GetDirectory());
}

//__________________________________________________________________________________|___________

void TQPlotter::addObject(TCollection* obj, const TString& key){
  // add a collection to the list of graphics objects maintained by the plotter
  if(!obj) return;
  if(!key.IsNull()) obj->SetName(key);
  this->objects->Add(obj);
}

//__________________________________________________________________________________|___________

void TQPlotter::addObject(TLegend* obj, const TString& key){
  // add a legend to the list of graphics objects maintained by the plotter
  if(!obj) return;
  if(!key.IsNull()) obj->SetName(key);
  this->objects->Add(obj);
}

//__________________________________________________________________________________|___________

void TQPlotter::addObject(TH1* obj, const TString& key){
  // add a histogram to the list of graphics objects maintained by the plotter
  if(!obj) return;
  if(!key.IsNull()) obj->SetName(key);
  if(this->objects->FindObject(obj->GetName())){
    ERRORclass("cannot add histogram '%s' - an object of this name already exists!",obj->GetName());
  }
  obj->SetDirectory(this->objects);
}

//__________________________________________________________________________________|___________

void TQPlotter::removeObject(const TString& key, bool deleteObject){
  // remove an object from the list of graphics object maintained by the plotter
  TObject* obj = this->objects->FindObject(key);
  if(!obj) return;
  this->objects->Remove(obj);
  if(deleteObject) delete obj;
}

//__________________________________________________________________________________|___________

void TQPlotter::clearObjects(){
  // clear all objects maintained by the plotter
  this->pads->SetOwner(false);
  this->pads->Clear();
  this->objects->Clear();
}

//__________________________________________________________________________________|___________

void TQPlotter::deleteObjects(){
  // clear all objects maintained by the plotter
  this->objects->DeleteAll();
  this->objects->Clear();
  this->pads->SetOwner(true);
  this->pads->Clear();
}

//__________________________________________________________________________________|___________

void TQPlotter::printObjects(){
  // print all objects maintained by the plotter
  TQIterator itr(this->objects->GetList());
  while(itr.hasNext()){
    TObject* obj = itr.readNext();
    if(!obj) continue;
    std::cout << TQStringUtils::makeBoldBlue(TQStringUtils::fixedWidth(obj->ClassName(),15));
    std::cout << " ";
    std::cout << TQStringUtils::makeBoldWhite(TQStringUtils::fixedWidth(obj->GetName(),50,"l"));
    std::cout << " ";
    std::cout << TQStringUtils::makeBoldWhite(TQStringUtils::fixedWidth(obj->GetTitle(),50,"l"));
    std::cout << " ";
    TGraph* g = dynamic_cast<TGraph*>(obj);
    if(g){
      std::cout << TQHistogramUtils::getDetailsAsString(g);
      if(TQStringUtils::matches(g->GetName(),"contour_*")) std::cout << ", contour area=" << fabs(TQHistogramUtils::getContourArea(g));
    } else if (obj->InheritsFrom(TH1::Class())) {
      std::cout << TQHistogramUtils::getDetailsAsString((TH1*)obj,4);
    } else {
      std::cout << TQStringUtils::getDetails(obj);
    }
    std::cout << std::endl;
  }
}

//__________________________________________________________________________________|___________

TObject* TQPlotter::getTObject(const TString& key){
  // retrieve a graphics object by name
  TQIterator itr(this->objects->GetList());
  while(itr.hasNext()){
    TObject* obj = itr.readNext();
    if(!obj) continue;
    if(TQStringUtils::matches(obj->GetName(),key)){
      return obj;
    }
  }
  return NULL;
}


//__________________________________________________________________________________|___________

void TQPlotter::applyStyle(TQTaggable& tags, TAxis* a, const TString& key, double/*distscaling*/, double/*sizescaling*/){
  // apply a tag-defined style to an axis
  // tags are read by the given key
  // sizes and distances are scaled according to the given parameters
  //@tags: style.*.showTitle: control whether axis title will be shown
  //@tags: style.*.showLabels: control whether axis labels will be shown
  //@tags: style.*.showTicks: control whether axis ticks
  //@tags: style.*.nDiv: number of divisions (encoded in 5 digit number, default 00510) for this axis
  if(!a) return;
  if(!tags.getTagBoolDefault("style."+key+".showTitle",true)){
    a->SetTitleOffset(0.);
    a->SetTitleSize (0.);
  }
  if(!tags.getTagBoolDefault("style."+key+".showLabels",true)){
    a->SetLabelOffset(0.);
    a->SetLabelSize (0.);
  }
  if(!tags.getTagBoolDefault("style."+key+".showTicks",true)){
    a->SetTickLength (0.);
  }
  if(!tags.getTagBoolDefault("style."+key+".allowExponent",true)){
    a->SetNoExponent(true);
  }
  if(tags.hasTag("style."+key+".nDiv")) a->SetNdivisions (tags.getTagIntegerDefault("style."+key+".nDiv",510));
}

//__________________________________________________________________________________|___________

void TQPlotter::applyGeometry(TQTaggable& tags, TAxis* a, const TString& key, double distscaling, double sizescaling){
  // apply a tag-defined geometry to an axis
  // tags are read by the given key
  // sizes and distances are scaled according to the given parameters
  //@tags: geometry.*.textSize: text size (default 0.05)
  //@tags: geometry.*.titleSize: title size (default:textSize)
  //@tags: geometry.*.labelSize: label size (default:textSize)
  //@tags: geometry.*.titleOffset: title offset
  //@tags: geometry.*.labelOffset: label offset
  //@tags: geometry.*.tickLength: label offset
  if(!a) return;
  double textSize = tags.getTagDoubleDefault("style.textSize",0.05);
  double titleSize = tags.getTagDoubleDefault ("geometry."+key+".titleSize",textSize);
  double labelSize = tags.getTagDoubleDefault ("geometry."+key+".labelSize",textSize);
  a->SetTitleOffset(distscaling*tags.getTagDoubleDefault ("geometry."+key+".titleOffset", 0.07)/titleSize);
  a->SetTitleSize (sizescaling*titleSize);
  a->SetLabelOffset(tags.getTagDoubleDefault ("geometry."+key+".labelOffset",0.00025)/labelSize);
  a->SetLabelSize (sizescaling*labelSize);
  a->SetTickLength (sizescaling*tags.getTagDoubleDefault ("geometry."+key+".tickLength",0.03));
}

//__________________________________________________________________________________|___________

void TQPlotter::applyGeometry(TQTaggable& tags, TH1* hist, const TString& key, double xscaling, double yscaling){
  // apply a tag-defined geometry to a histogram
  // tags are read by the given key
  // sizes and distances are scaled according to the given parameters
  if(!hist) return;
  TAxis* xAxis = hist->GetXaxis();
  // *jedi move* you do not want to know of the handling of xscaling vs. yscaling
  applyGeometry(tags,xAxis,key+".xAxis",1/xscaling,yscaling/xscaling);
  TAxis* yAxis = hist->GetYaxis();
  // *jedi move* you do not want to know of the handling of xscaling vs. yscaling
  applyGeometry(tags,yAxis,key+".yAxis",1/yscaling,yscaling/xscaling);
}

//__________________________________________________________________________________|___________

void TQPlotter::applyGeometry(TQTaggable& tags, TGraph* g, const TString& key, double xscaling, double yscaling){
  // apply a tag-defined geometry to a graph
  // tags are read by the given key
  // sizes and distances are scaled according to the given parameters
  if(!g) return;
  TAxis* xAxis = g->GetXaxis();
  // *jedi move* you do not want to know of the handling of xscaling vs. yscaling
  applyGeometry(tags,xAxis,key+".xAxis",1/xscaling,yscaling/xscaling);
  TAxis* yAxis = g->GetYaxis();
  // *jedi move* you do not want to know of the handling of xscaling vs. yscaling
  applyGeometry(tags,yAxis,key+".yAxis",1/yscaling,yscaling/xscaling);
}

//__________________________________________________________________________________|___________

void TQPlotter::applyStyle(TQTaggable& tags, TH1* hist, const TString& key, double xscaling, double yscaling){
  // apply a tag-defined style to a histogram
  // tags are read by the given key
  // sizes and distances are scaled according to the given parameters
  //@tags: style.*.fillColor: set fill color using TH1::SetFillColor
  //@tags: style.*.fillStyle: set fill style using TH1::SetFillStyle
  //@tags: style.*.lineColor: set line color using TH1::SetLineColor
  //@tags: style.*.lineStyle: set line style using TH1::SetLineStyyle
  //@tags: style.*.markerColor: set marker color using TH1::SetMarkerColor
  //@tags: style.*.markerSize: set marker size using TH1::SetMarkerSize
  //@tags: style.*.markerStyle: set marker size using TH1::SetMarkerStyle
  if(!hist) return;
  TAxis* xAxis = hist->GetXaxis();
  // *jedi move* you do not want to know of the handling of xscaling vs. yscaling
  applyStyle(tags,xAxis,key+".xAxis",yscaling/xscaling,xscaling/yscaling);
  TAxis* yAxis = hist->GetYaxis();
  // *jedi move* you do not want to know of the handling of xscaling vs. yscaling
  applyStyle(tags,yAxis,key+".yAxis",xscaling/yscaling,yscaling/xscaling);
  int fillColor = hist->GetFillColor (); tags.getTagInteger("style."+key+".fillColor", fillColor); hist->SetFillColor (fillColor);
  int fillStyle = hist->GetFillStyle (); tags.getTagInteger("style."+key+".fillStyle", fillStyle); hist->SetFillStyle (fillStyle);
  int lineColor = hist->GetLineColor (); tags.getTagInteger("style."+key+".lineColor", lineColor); hist->SetLineColor (lineColor);
  double lineWidth = hist->GetLineWidth (); tags.getTagDouble ("style."+key+".lineWidth", lineWidth); hist->SetLineWidth (lineWidth);
  int lineStyle = hist->GetLineStyle (); tags.getTagInteger("style."+key+".lineStyle", lineStyle); hist->SetLineStyle (lineStyle);
  int markerColor = hist->GetMarkerColor(); tags.getTagInteger("style.markerColor",markerColor); tags.getTagInteger("style."+key+".markerColor", markerColor); hist->SetMarkerColor(markerColor);
  double markerSize = hist->GetMarkerSize (); tags.getTagDouble ("style.markerSize", markerSize ); tags.getTagDouble ("style."+key+".markerSize" , markerSize ); hist->SetMarkerSize (markerSize );
  int markerStyle = hist->GetMarkerStyle(); tags.getTagInteger("style.markerStyle",markerStyle); tags.getTagInteger("style."+key+".markerStyle", markerStyle); hist->SetMarkerStyle(markerStyle);
  if(tags.getTagBoolDefault ("style.binticks",false )){
    hist->GetXaxis()->SetNdivisions(hist->GetNbinsX(),0,0);
  }
}

//__________________________________________________________________________________|___________

void TQPlotter::applyStyle(TQTaggable& tags, TGraph* g, const TString& key, double xscaling, double yscaling){
  // apply a tag-defined style to a graph
  // tags are read by the given key
  // sizes and distances are scaled according to the given parameters
  if(!g) return;
  TAxis* xAxis = g->GetXaxis();
  applyStyle(tags,xAxis,key+".xAxis",yscaling/xscaling,xscaling/yscaling);
  TAxis* yAxis = g->GetYaxis();
  applyStyle(tags,yAxis,key+".yAxis",xscaling/yscaling,yscaling/xscaling);
  int fillColor = 0; if(tags.getTagInteger("style."+key+".fillColor", fillColor) || tags.getTagInteger("style.fillColor", fillColor)) g->SetFillColor(fillColor);
  int fillStyle = 0; if(tags.getTagInteger("style."+key+".fillStyle", fillStyle)  || tags.getTagInteger("style.fillStyle", fillStyle)) g->SetFillStyle(fillStyle);
  int lineColor = 0; if(tags.getTagInteger("style."+key+".lineColor", lineColor) || tags.getTagInteger("style.lineColor", lineColor)) g->SetLineColor(lineColor);
  int lineStyle = 0; if(tags.getTagInteger("style."+key+".lineStyle", lineStyle) || tags.getTagInteger("style.lineStyle", lineStyle)) g->SetLineStyle(lineStyle);
  double lineWidth = 0; if(tags.getTagDouble("style."+key+".lineWidth", lineWidth) || tags.getTagDouble("style.lineWidth", lineWidth)) g->SetLineWidth(lineWidth);
  int markerColor = 0; if(tags.getTagInteger("style."+key+".markerColor", markerColor) || tags.getTagInteger("style.markerColor", markerColor)) g->SetMarkerColor(markerColor);
  double markerSize = 0; if(tags.getTagDouble("style."+key+".markerSize", markerSize) || tags.getTagDouble("style.markerSize", markerSize)) g->SetMarkerSize(markerSize);
  int markerStyle = 0; if(tags.getTagInteger("style."+key+".markerStyle", markerStyle) || tags.getTagInteger("style.markerStyle", markerStyle)) g->SetMarkerStyle(markerStyle);
  if(tags.getTagBoolDefault ("style.binticks",false )){
    g->GetXaxis()->SetNdivisions(g->GetHistogram()->GetNbinsX(),0,0);
  }
}

//__________________________________________________________________________________|___________

TPad* TQPlotter::createPad(TQTaggable& tags, const TString& key){
  // create a pad and add it to the list
  // geometry and styling parameters will be read from the given tag set
  // the tags according to the given key will be read
  //@tags: geometry.*.scaling: scaling of a pad
  //@tags: [geometry.*.xMin,geometry.*.xMax,geometry.*.yMin,geometry.*.yMax]: set the geometry parameters of a pad
  //@tags: [style.*.fillStyle, style.*.fillColor]: control fill style and color of a pad
  //@tags: [geometry.*.margins.left,geometry.*.margins.right,geometry.*.margins.top,geometry.*.margins.bottom]: control margins of a pad
  //@tags: [style.*.tickx,style.*.ticky]: control whether ticks are shown on x and y axes of this pad
  //@tags: [style.tickx,style.ticky]: control whether ticks are shown on x and y axes of all pads
  //@tags: [style.*.borderSize,style.*.borderMode] control appearance of the borders of this pad
  double padscaling = tags.getTagDoubleDefault("geometry."+key+".scaling",1.);
  TPad * pad = new TPad(key,key,
                        tags.getTagDoubleDefault("geometry."+key+".xMin",0.), 
                        tags.getTagDoubleDefault("geometry."+key+".yMin",0.), 
                        tags.getTagDoubleDefault("geometry."+key+".xMax",1.), 
                        tags.getTagDoubleDefault("geometry."+key+".yMax",1.));
  pad->SetFillStyle(tags.getTagIntegerDefault("style."+key+".fillStyle",0));
  pad->SetFillColor(tags.getTagIntegerDefault("style."+key+".fillColor",0));
  pad->SetMargin(tags.getTagDoubleDefault("geometry."+key+".margins.left" ,0.16),
                 tags.getTagDoubleDefault("geometry."+key+".margins.right" ,0.05),
                 padscaling*tags.getTagDoubleDefault("geometry."+key+".margins.bottom",0.16),
                 padscaling*tags.getTagDoubleDefault("geometry."+key+".margins.top" ,0.05));
  pad->SetTickx(tags.getTagIntegerDefault("style."+key+"tickx",tags.getTagIntegerDefault("style.tickx",1)));
  pad->SetTicky(tags.getTagIntegerDefault("style."+key+"ticky",tags.getTagIntegerDefault("style.ticky",1)));
  pad->SetBorderSize(tags.getTagIntegerDefault("style."+key+".borderSize",0));
  pad->SetBorderMode(tags.getTagIntegerDefault("style."+key+".borderMode",0));
  this->pads->Add(pad);
  return pad;
}

//__________________________________________________________________________________|___________

TCanvas* TQPlotter::createCanvas(TQTaggable& tags){
  /* general layout */
  /* prepare the name of the canvas */
  bool verbose = tags.getTagBoolDefault("verbose",false);
  TString canvasName = tags.getTagStringDefault("input.chname");
  canvasName.Append("_");
  canvasName.Append(tags.getTagStringDefault("input.name","histogram"));
  canvasName.ReplaceAll("/", "_");
  canvasName=TQStringUtils::makeValidIdentifier(canvasName,"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890._","");
 
  /* don't make the new canvas kill another with the same
   * name: append an increasing number until it's unique */
  int iCanvas = 1;
  TCollection * listOfCanvases = gROOT->GetListOfCanvases();
  const TString origCanvasName(canvasName);
  while (listOfCanvases && listOfCanvases->FindObject(canvasName.Data()))
    canvasName = TString::Format("%s_n%d", origCanvasName.Data(), iCanvas++);

  // @tags:geometry.canvas.width: set the width of the canvas
  // @tags:geometry.canvas.height: set the height of the canvas
  if(verbose) VERBOSEclass("creating canvas with name '%s'",canvasName.Data());
  TCanvas* canvas = new TCanvas(TQFolder::makeValidIdentifier(canvasName),canvasName,0,0,
                                tags.getTagIntegerDefault("geometry.canvas.width",800),
                                tags.getTagIntegerDefault("geometry.canvas.height",600));
  canvas->SetMargin(0.,0.,0.,0);
  canvas->cd();
 
  TPad* pad = this->createPad(tags,"main");
  // @tags:style.logScale: control whether the main plot will be shown in log scale (default:false)
  if (tags.getTagBoolDefault ("style.logScale",false ) && (tags.getTagIntegerDefault("style.nDim",1) == 1)){
    pad->SetLogy();
  }
  // @tags:style.logScaleX: control whether the main plot will be shown in logX scale (default:false)
  if (tags.getTagBoolDefault ("style.logScaleX",false )){
    pad->SetLogx();
  }
  pad->Draw();
  // @tags:style.showSub: control whether any subplot will be shown. overwrites subplots defined elsewhere.
  if (tags.getTagBoolDefault ("style.showSub",false)){
    canvas->cd();
    TPad * ratioPad = this->createPad(tags,"sub");
    ratioPad->SetGridy(true);
    // @tags:style.logScaleX: control whether the subplot will be shown in logX scale (default:false)
    if (tags.getTagBoolDefault ("style.logScaleX",false )){
      ratioPad->SetLogx();
    }
    ratioPad->Draw();
  }
 
  canvas->cd();
  return canvas;
}

//__________________________________________________________________________________|___________

void TQPlotter::setAxisLabels(TQTaggable& tags){
  TH1* hMaster = this->getObject<TH1>("master");
  // @tags: labels.axes.mainX: control the x axis labels of the main frame
  TString xLabel = tags.getTagStringDefault("labels.axes.mainX", hMaster->GetXaxis()->GetTitle());
  hMaster->GetXaxis()->SetTitle(xLabel);

  if(tags.getTagIntegerDefault("style.nDim",1) == 1){
    TString xUnit = TQStringUtils::getUnit(TString(hMaster->GetXaxis()->GetTitle()));
    double binWidth = (hMaster->GetBinLowEdge(hMaster->GetNbinsX() + 1) -
                       hMaster->GetBinLowEdge(1)) / hMaster->GetNbinsX();

    // We're using variable width binning (should still work with fixed width)
    int densityBin = 0;
    if (tags.getTagInteger("scaleDensityToBin",densityBin)){
      binWidth = hMaster->GetXaxis()->GetBinWidth(densityBin);
    }

    bool isInteger = ((binWidth - (int)binWidth) < 0.0001);
 
 
    bool normalize = tags.getTagBoolDefault("normalize",false );

    TString yLabel = "Events";
    
    if(normalize){
      /* "Events" or "arbitrary units"? */
      yLabel = "arbitrary units";
    } else if (TQHistogramUtils::hasUniformBinning(hMaster) && !(xUnit.IsNull() && TMath::AreEqualRel(binWidth, 1., 1E-6))) {
    /* there is only one case in which we don't have to add anything to the
     * "Events" label: no unit on the x axis and a bin width of exactly 1 */
      // also, if the binning is irregular, we skip the bin width addition

      if (isInteger)
        yLabel.Append(TString::Format(" / %.0f", binWidth));
      else
        yLabel.Append(TString::Format(" / %.2g", binWidth));
      /* append the unit of the x axis */
      if (xUnit.Length() > 0)
        yLabel.Append(TString::Format(" %s", xUnit.Data()));
    }
    // if (normalize) 

    // @tags: labels.axes.mainY: control the y axis labels of the main frame (usually 'Events' or 'arbitrary units')
    tags.getTagString("labels.axes.mainY", yLabel);

    hMaster->GetYaxis()->SetTitle(yLabel.Data());
  } else {
    TString yLabel = tags.getTagStringDefault("labels.axes.mainY", hMaster->GetYaxis()->GetTitle());
    hMaster->GetYaxis()->SetTitle(yLabel);
  }
}

//__________________________________________________________________________________|___________

void TQPlotter::drawCutLines1D(TQTaggable& tags){
  TH1* hMaster = this->getObject<TH1>("master");
  double upper = hMaster->GetMaximum();
  double lower = hMaster->GetMinimum();
  bool logScale = tags.getTagBoolDefault ("style.logScale",false );

  /* read the list of cut thresholds to display */
  int iCut = 0;
  double threshold = 0.;
  //@tags: cuts.*: a list of (vertical) cut lines to be drawn. value will be x-value of the vertical line
  while (tags.getTagDouble(TString::Format("cuts.%d", iCut++), threshold)) {
    int iBlock = 0;
    double block_x = 0; 
    double block_x_old = 0;
    double block_y = 100;
    while(tags.getTag(TString::Format("blocks.x.%d",iBlock),block_x) && tags.getTag(TString::Format("blocks.y.%d",iBlock),block_y)){
      if(threshold > block_x_old && threshold < block_x){
        //        std::cout << threshold << ":" << block_y << std::endl;
        break;
      }
      block_x_old = block_x;
      iBlock++;
    }
    double max = logScale ? TMath::Exp((TMath::Log(upper) - TMath::Log(lower)) * block_y + TMath::Log(lower)) : (upper - lower) * block_y + lower;
 
    TLine * line = new TLine(threshold, lower, threshold, max);
    //@tags:[style.cutLineStyle,style.cutLineWidth,style.cutLineColor]: control appearance of cutlines (TLine::SetLineStyle,TLine::SetLineWidth,TLine::SetLineColor)
    line->SetLineStyle(tags.getTagIntegerDefault("style.cutLineStyle",7));
    line->SetLineWidth(tags.getTagIntegerDefault("style.cutLineWidth",2));
    line->SetLineColor(tags.getTagIntegerDefault("style.cutLineColor",kRed));
    line->Draw();
  }
}

//__________________________________________________________________________________|___________

TString TQPlotter::createAxisTagsAsString(const TString& prefix, const TString& title, double xCoeff, double yCoeff, double constCoeff, double wMin, double wMax, double xCoord, double yCoord, int /*nDiv*/){
  // convert a variable definition into tags for an additional axis
  TQTaggable tags;
  if(TQPlotter::createAxisTags(tags,prefix,title,xCoeff, yCoeff, constCoeff, wMin, wMax, xCoord, yCoord)){
    return tags.exportTagsAsString();
  }
  return "";
}

//__________________________________________________________________________________|___________

TString TQPlotter::createAxisTagsAsConfigString(const TString& prefix, const TString& title, double xCoeff, double yCoeff, double constCoeff, double wMin, double wMax, double xCoord, double yCoord, int /*nDiv*/){
  // convert a variable definition into tags for an additional axis
  TQTaggable tags;
  if(TQPlotter::createAxisTags(tags,prefix,title,xCoeff, yCoeff, constCoeff, wMin, wMax, xCoord, yCoord)){
    return tags.exportTagsAsConfigString("");
  }
  return "";
}

//__________________________________________________________________________________|___________

TQTaggable* TQPlotter::createAxisTags(const TString& prefix, const TString& title, double xCoeff, double yCoeff, double constCoeff, double wMin, double wMax, double xCoord, double yCoord, int /*nDiv*/){
  // convert a variable definition into tags for an additional axis
  TQTaggable* tags = new TQTaggable();;
  TQPlotter::createAxisTags(*tags,prefix,title,xCoeff, yCoeff, constCoeff, wMin, wMax, xCoord, yCoord);
  return tags;
}

//__________________________________________________________________________________|___________

bool TQPlotter::createAxisTags(TQTaggable& tags, const TString& prefix, const TString& title, double xCoeff, double yCoeff, double constCoeff, double wMin, double wMax, double xCoord, double yCoord, int nDiv){
  // convert a variable definition into tags for an additional axis
  double wCoord = xCoeff*xCoord + yCoeff*yCoord + constCoeff;
  double coeff2 = xCoeff* xCoeff + yCoeff * yCoeff;

  double tmin = (wMin - wCoord)/coeff2;
  double tmax = (wMax - wCoord)/coeff2;

  double xmin = xCoord + xCoeff * tmin;
  double xmax = xCoord + xCoeff * tmax;
  double ymin = yCoord + yCoeff * tmin;
  double ymax = yCoord + yCoeff * tmax;

  tags.setTagBool(prefix+"show",true);
  tags.setTagDouble(prefix+"xMin",xmin);
  tags.setTagDouble(prefix+"xMax",xmax);
  tags.setTagDouble(prefix+"yMin",ymin);
  tags.setTagDouble(prefix+"yMax",ymax);
  tags.setTagDouble(prefix+"wMin",wMin);
  tags.setTagDouble(prefix+"wMax",wMax);
  tags.setTagInteger(prefix+"nDiv",nDiv);
  tags.setTagString(prefix+"title",title);

  return true;
}

//__________________________________________________________________________________|___________


int TQPlotter::drawHeightLines(TQTaggable& tags){
  // draw height lines onto the canvas
  if(!tags.getTagBoolDefault("heightlines.show",false))
    return 0;

  bool verbose = tags.getTagBoolDefault("verbose",false);
  if(verbose) VERBOSEclass("attempting to draw height lines");

  TH1* hMaster = this->getObject<TH1>("master");

  //@tags:[heightlines.show] draw additional diagonal height lines in 2D plots
  //@tags:[heightlines.xCoeff,heightlines.yCoeff,heightlines.constCoeff]  control the slope of diagonal height lines in 2D plots
  double xCoeff = tags.getTagDoubleDefault("heightlines.xCoeff",0.);
  double yCoeff = tags.getTagDoubleDefault("heightlines.yCoeff",0.);
  double constCoeff = tags.getTagDoubleDefault("heightlines.constCoeff",0.);

  bool rotate = tags.getTagBoolDefault("heightlines.rotateLabels",true);
  //@tags:[heightlines.rotateLabels] rotate labels according to the inclination angle of height lines in 2D plots

  double labelSize = tags.getTagDoubleDefault("style.axes.labelSize",0.03);
  double labelOffset = tags.getTagDoubleDefault("style.axes.labelOffset",0.005);

  //@tags:[heightlines.color,heightlines.style]  control the visuals of diagonal height lines in 2D plots
  //@tags:[heightlines.values] list of values along the x axis at which height lines should appear
  int color = tags.getTagIntegerDefault("heightlines.color",kBlack);
  int linestyle = tags.getTagIntegerDefault("heightlines.style",1.);

  std::vector<double> vals = tags.getTagVDouble("heightlines.values");
  double xmin = TQHistogramUtils::getAxisXmin(hMaster);
  double xmax = TQHistogramUtils::getAxisXmax(hMaster);

  int n = 0;

  double slope = - xCoeff/yCoeff;


  TLatex latex;
  latex.SetTextColor(color);
  latex.SetTextSize(labelSize);
  double latexXOffset, latexYOffset;

  if(rotate){
    double visualSlope = - TQUtils::convertdYtoPixels(xCoeff)/TQUtils::convertdXtoPixels(yCoeff);
    double visualAngle = atan(visualSlope) * 180./TMath::Pi();
    latex.SetTextAngle(visualAngle);
    latexXOffset = TQUtils::convertdXfromNDC(labelOffset * sin(-visualSlope));
    latexYOffset = TQUtils::convertdYfromNDC(labelOffset * cos( visualSlope));
  } else {
    latexXOffset = 0.;
    latexYOffset = TQUtils::convertdYfromNDC(labelOffset);
  }

  for(size_t i=0; i<vals.size(); i++){
    if(verbose) VERBOSEclass("drawing height line for z = %g",vals[i]);

    double offset = (vals[i] - constCoeff) / yCoeff;

    double y0 = offset + slope * xmin;
    double y1 = offset + slope * xmax;
    double x0 = xmin;
    double x1 = xmax;

    if(verbose) VERBOSEclass("pre-crop coordinates are x0=%g, x1=%g, y0=%g, y1=%g",x0,x1,y0,y1);

    TLine* l = new TLine(x0,y0,x1,y1);
    if(TQHistogramUtils::cropLine(hMaster,l)){
      if(verbose) VERBOSEclass("post-crop coordinates are x0=%g, x1=%g, y0=%g, y1=%g",l->GetX1(),l->GetX2(),l->GetY1(),l->GetY2());
      l->SetLineColor(color);
      l->SetLineStyle(linestyle);
      l->Draw();
      latex.DrawLatex(latexXOffset + 0.5*(l->GetX2()+l->GetX1()),latexYOffset + 0.5*(l->GetY2()+l->GetY1()),TString::Format("%g",vals[i]));
      n++;
    } else {
      if(verbose) VERBOSEclass("line-crop failed - no line plotted");
      delete l;
    }
  }

  return n;
}

//__________________________________________________________________________________|___________


int TQPlotter::drawAdditionalAxes(TQTaggable& tags){
  // draw an additional axis onto the canvas
  double defaultLabelSize = tags.getTagDoubleDefault("style.axes.labelSize",0.03);
  double defaultTitleSize = tags.getTagDoubleDefault("style.axes.titleSize",0.03);
  double defaultTitleOffset = tags.getTagDoubleDefault("style.axes.titleOffset",1.0);

  /* read the list of cut thresholds to display */
  int iAxis = -1;
  bool show = false;

  //@tags:axis.IDX.show: draw an additional axis with index IDX
  //@tags:[axis.IDX.xMin,axis.IDX.xMax,axis.IDX.yMin,axis.IDX.yMax]: control geometry of additional axis IDX to be drawn
  //@tags:[axis.IDX.wMin,axis.IDX.wMax,axis.IDX.nDiv,axis.IDX.title]: control labeling of additional axis IDX to be drawn
  //@tags:[axis.*.labelSize,axis.IDX.titleSize,axis.IDX.titleOffset]: control style of additional axis IDX to be drawn
  
  while (tags.getTagBool(TString::Format("axis.%d.show", ++iAxis), show)) {
    if(!show) continue;

    double xmin, xmax, ymin, ymax;
    if(!tags.getTagDouble(TString::Format("axis.%d.xMin", iAxis), xmin)) continue;
    if(!tags.getTagDouble(TString::Format("axis.%d.xMax", iAxis), xmax)) continue;
    if(!tags.getTagDouble(TString::Format("axis.%d.yMin", iAxis), ymin)) continue;
    if(!tags.getTagDouble(TString::Format("axis.%d.yMax", iAxis), ymax)) continue;

    double wmin = tags.getTagDoubleDefault (TString::Format("axis.%d.wMin" , iAxis), 0);
    double wmax = tags.getTagDoubleDefault (TString::Format("axis.%d.wMax" , iAxis), 1);
    int nDiv = tags.getTagIntegerDefault (TString::Format("axis.%d.nDiv" , iAxis), 110);
    TString title = tags.getTagStringDefault(TString::Format("axis.%d.title", iAxis), "");


    double labelSize = tags.getTagDoubleDefault (TString::Format("axis.%d.labelSize", iAxis), defaultLabelSize);
    double titleSize = tags.getTagDoubleDefault (TString::Format("axis.%d.titleSize", iAxis), defaultTitleSize);
    double titleOffset = tags.getTagDoubleDefault (TString::Format("axis.%d.titleOffset", iAxis), defaultTitleOffset);

    TGaxis *addAxis = new TGaxis(xmin, ymin, xmax, ymax,wmin,wmax,nDiv);
    if(!title.IsNull()) addAxis->SetTitle(title);
    addAxis->SetLabelSize(labelSize);
    addAxis->SetTitleSize(titleSize);
    addAxis->SetTitleOffset(titleOffset);
    addAxis->Draw();
    show = false;
  }
  return iAxis;
}


//__________________________________________________________________________________|___________

int TQPlotter::getNProcesses(const TString& tagFilter){
  // return the number of added processes matching the tag filter
  TQTaggableIterator itr(this->fProcesses);
  int retval = 0;
  while(itr.hasNext()){
    TQNamedTaggable* tags = itr.readNext();
    if(!tags) continue;
    if(tags->getTagBoolDefault(tagFilter,false)) retval++;
  }
  return retval;
}

//__________________________________________________________________________________|___________

int TQPlotter::sanitizeProcesses() {
  // sanitize all processes 
  TQTaggableIterator itr(fProcesses);
  std::vector<TQNamedTaggable*> removals;
  int retval = 0;
  while(itr.hasNext()){
    TQNamedTaggable* process = itr.readNext();
    if(!process) continue;
    if(process->getTagStringDefault(".path","").Contains("|")){
      removals.push_back(process);
    }
  }
  for(size_t i=0; i<removals.size(); i++){
    fProcesses->Remove(removals[i]);
    delete removals[i];
    retval++;
  }
  return retval;
}
