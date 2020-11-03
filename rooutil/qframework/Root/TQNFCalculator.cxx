#include "QFramework/TQNFCalculator.h"
#include "QFramework/TQStringUtils.h"
#include "QFramework/TQIterator.h"
#include "QFramework/TQUtils.h"
#include "TMath.h"
#include "TRandom.h"
#include "QFramework/TQHistogramUtils.h"
#include "QFramework/TQNFChainloader.h"


#include <limits>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <fstream>

#include "TFractionFitter.h"

//#define _DEBUG_

#include "QFramework/TQLibrary.h"

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQNFCalculator
//
// The TQNFCalculator is a class that automates calculation of
// normalization factors based on control regions.
// It works based on a TQSampleDataReader.
//
// The TQNFCalculator supports two different operational modes
// 1) manual subtraction of unscaled samples 
// To use the TQNFCalculator in this mode, you should
// set the default data path to your data path 
// subtracting all samples that have no NF applied, e.g.
// NFCalculator::setDefaultDataPath("data - sig - bkg/unscaled")
// 2) automatic handling of fixed samples
// To use the TQNFCalculator in this mode, you should call
// TQNFCalculator::addFixSample(path)
// you can choose to either use the combined path, e.g.
// "sig - bkg/unscaled"
// or make several calls for individual sample groups, e.g.
// TQNFCalculator::addFixSample("sig")
// TQNFCalculator::addFixSample("bkg/unscaled")
// The central values should be the same for both methods,
// but the error calculation is more correct when using automatic mode.
// 
// It is possible to set bounds on individual NFs for floating (non-fixed)
// samples by using the variant
// TQNFCalculator::addSample(name,path,lowerBound,upperBound);
// This should be done whenever possible, since the precision of some methods
// relies heavily on sensible bounds.
// 
// Additionally, the TQNFCalculator supports different methods calculation
// - 'simple' or 'single mode': in cases when there is only one floating
// sample this method is not using any elaborate technique, but instead
// calculates the NF arithmeticall ('by hand')
// - 'TFractionFitter': a TFractionFitter is used to simulatenously fit all
// NFs in all regions. The error calculation is not entirely correct since
// event weights are not handled propery
// - 'MatrixInversion' : this mode is only possible if the number of floating
// samples equals the number of regions. Here, all are filled into a matrix
// which is then inverted. While this yields correct central values in principle,
// the precision and especially the correctness of the error propagation depends 
// heavily on the condition of the matrix. For optimal results, lower and upper bounds
// should be set for each NF and the NFs and regions should be appended in the correct
// order, i.e. such that the matrix is mostly diagonal
// It is possible to have the TQNFCalculator try different methods successively, via
// TQNFCalculator::addMethod(methodName)
// For this purpose, two 'fallback'-methods exist:
// - 'Unity' will set all NFs to unity (NF=1)
// - 'FAIL' will unset all NFs and report an error
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQNFCalculator)


TQNFCalculator::TQNFCalculator(TQSampleFolder* f):
TQNFBase("TQNF"),
  status(-999),
  defaultDataPath("data"),
  data(NULL),
  mc(NULL),
  epsilon(std::numeric_limits<double>::epsilon())
{
  // default constructor taking base sample folder
  this->setSampleFolder(f);
}

TQNFCalculator::TQNFCalculator(TQSampleDataReader* rd):
  TQNFBase("TQNF"),
  status(-999),
  defaultDataPath("data"),
  data(NULL),
  mc(NULL),
  epsilon(std::numeric_limits<double>::epsilon())
{
  // default constructor taking base sample folder
  this->setReader(rd);
}

TString TQNFCalculator::getDefaultDataPath(){
  // retrieve the default data path currently set
  return this->defaultDataPath;
}

TQNFCalculator::~TQNFCalculator(){
  // default destructor, cleaning up all pointer fields
  this->finalize();
  this->clear();
}

int TQNFCalculator::deployNF(const TString& name, const std::vector<TString>& startAtCutNames,  const std::vector<TString>& stopAtCutNames, int overwrite){
  // deploys a selected NF in all known locations
  // stop if this cut is an point of tree of cuts on which to apply the NFs
  /*
  std::cout<<"deployNF called with cutName = "<< cutName.Data()<<", stopAtCutNames:"<<std::endl;
  TQListUtils::print(stopAtCutNames);
  std::cout<<"This is where I'd apply NFs with the new code:"<<std::endl;
  std::vector<TString> startCuts;
  startCuts.push_back(cutName);
  std::vector<TString> targets = this->getTargetCuts(startCuts,stopAtCutNames);
  TQListUtils::print(targets);
  std::cout<<"Now go and compare if I did it right ;)"<<std::endl;
  */
  if(!this->success()){ //cutName.Data(),
    WARNclassargs(TQStringUtils::concatenate(3,name.Data(),TQStringUtils::concat(stopAtCutNames).Data(),TQStringUtils::getStringFromBool(overwrite).Data()),"cannot deploy NF, no results available, status is %d!",this->status);
    return -1;
  }
  /*
  bool last = false;
  for (size_t i=0; i<stopAtCutNames.size(); ++i) {
    if (TQStringUtils::matches(cutName, stopAtCutNames[i])) {
      last = true;
      break;
    }
  }
  */
  
  // get the NF
  double nf = 0;
  double sigma = 0;
  int retval = 0;
  this->getNFandUncertainty(name,nf,sigma);
  //if((overwrite > 1) || std::fabs(nf-1.) > this->epsilon){
    // get the monte carlo path for this sample
    TString mcpath = this->getMCPath(name);
    // set the NF according to the desired scale scheme(s)
    //@tag:[writeScaleScheme] This object tag determines the list of scale schemes the results of the NF calculation are written to. Default: ".default"
    std::vector<TString> writeScaleSchemes = this->getTagVString("writeScaleScheme");
    if(writeScaleSchemes.size() < 1){
      writeScaleSchemes.push_back(this->getTagStringDefault("writeScaleScheme",".default"));
    }
    std::vector<TString> targets = this->getTargetCuts(startAtCutNames,stopAtCutNames);
    //--------------------------------------------------------
    for (size_t c = 0; c<targets.size(); ++c) {
      TString cutName = targets.at(c);
      DEBUGclass("deploying NF for %s at %s (overwrite=%d)",name.Data(),cutName.Data(),overwrite);
      TQSampleFolderIterator itr(this->fReader->getListOfSampleFolders(mcpath),true);
      while(itr.hasNext()){
        TQSampleFolder* s = itr.readNext();
        if(!s) continue;
        for(size_t k=0; k<writeScaleSchemes.size(); k++){
          int n = s->setScaleFactor(writeScaleSchemes[k]+":"+cutName+(overwrite>0?"":"<<"), nf,sigma);
          if(n == 0){
            ERRORclass("unable to set scale factor for cut '%s' on path '%s' with scheme '%s'",cutName.Data(),s->getPath().Data(),writeScaleSchemes[k].Data());
          } else {
          DEBUG("Set scale factor for cut '%s' on path '%s' with scheme '%s' (value: %.2f +/- %.2f, overwrite: %s) ",cutName.Data(),s->getPath().Data(),writeScaleSchemes[k].Data(),nf,sigma, (overwrite>0?"true":"false") );
          }
          //keep track where a NF has been written to as required by TQNFBase for use in TQNFChainloader
          this->addNFPath(s->getPath(),cutName,writeScaleSchemes[k]);
          retval += n;
        } 
      }
      // if the info folder is set and valid, we should keep track of all the processes that have NFs applied
      if(this->infoFolder){
        // get the folder which contains the list of processes for which we have NFs
        //@tag:[nfListPattern] This object tag determines the format how the existence of NFs for the target paths/cuts is written to the info folder (if present). Default: ".cut.%s+"
        TQFolder * sfProcessList = this->infoFolder->getFolder(TString::Format(this->getTagStringDefault("nfListPattern",".cut.%s+").Data(),cutName.Data()));
        // get the sample folder which contains the samples for this process
        TList* sflist = this->fReader->getListOfSampleFolders(mcpath);
        TQSampleFolder * processSampleFolder = ( sflist && sflist->GetEntries() > 0 ) ? (TQSampleFolder*)(sflist->First()) : NULL;
        if(sflist) delete sflist;
        // retrieve the correct title of the process from this folder
        // if there is no process title set, we will use the process name instead
        TString processTitle = name;
        if (processSampleFolder)
          //@tag:[processTitleKey] This object tag determines the name of the process tag used to retrieve the process title from. Default: "style.default.title".
          processSampleFolder->getTagString(this->getTagStringDefault("processTitleKey","style.default.title"), processTitle);
        // after we have aquired all necessary information, we add a new entry 
        // to the list of processes to which NFs have been applied
        sfProcessList->setTagString(TQFolder::makeValidIdentifier(processTitle),processTitle);
      }
    //}
  }
  //------------------------------------------
  /* 
  // if no recursion was required or we arrived at a stop cut, we can stop here
  if(stopAtCutNames.size() == 0 || !this->cutInfoFolder || last)
    return retval;
  // if stopAtCutNames is set, we need to recurse over the cut structure
  // therefore, we first need to find out how the cuts are structured
  */
  //TList * cuts = this->cutInfoFolder->getListOfFolders(TString::Format("*/%s/?",cutName.Data()));//HERE!!!
  /*
  if(!cuts) return retval;
  TQIterator iter(cuts,true);
  // iterate over all the cuts below the one we are investigating now
  while(iter.hasNext()){
    TQFolder* f = dynamic_cast<TQFolder*>(iter.readNext());
    if(!f) continue;
    // and deploy NFs at the corresponding cuts
    retval += this->deployNF(name,f->GetName(),stopAtCutNames,overwrite);
  }
  */
  return retval;
}

int TQNFCalculator::deployResult(const std::vector<TString>& startAtCutNames,  const std::vector<TString>& stopAtCutNames, int overwrite){
  if(!this->success()){
    WARNclassargs(TQStringUtils::concatenate(3,TQListUtils::makeCSV(startAtCutNames).Data(),TQStringUtils::concat(stopAtCutNames).Data(),TQStringUtils::getStringFromBool(overwrite).Data()),"cannot deploy NFs, no results available, status is %d!",this->status);
    return -1;
  }
  // deploys all NFs in all known locations
  int retval = 0;
  for(size_t i=0; i<mcNames.size(); i++){
    if(mcFixed[i]) continue;
    retval += this->deployNF(mcNames[i],startAtCutNames, stopAtCutNames,overwrite);
  }
  return retval;
}

void TQNFCalculator::clear(){
  // clear all names, paths and results
  this->dataPaths.clear();
  this->cutNames.clear();
  this->mcNames.clear();
  this->mcPaths.clear();
  this->mcFixed.clear();
  this->NFs.clear();
  this->NFuncertainties.clear();
  this->nfBoundUpper.clear();
  this->nfBoundLower.clear();
  this->initialized = false;
  this->status = -999;
  if(data){
    delete this->data;
    this->data = NULL;
  }
  if(mc){
    delete this->mc;
    this->mc = NULL;
  }
} 

void TQNFCalculator::setDefaultDataPath(TString path){
  // set the default data path 
  // this path is going to be used if no path is supplied 
  // when adding regions via addRegion(...)
  this->defaultDataPath = path;
}

bool TQNFCalculator::addRegion(TString cutName, TString myDataPath){
  // add a new region. the region name will be the cut name
  // optional argument: supply a data path from which data will be obtained
  if(this->initialized){
    ERRORclass("cowardly refusing to add region to already initialized instance - please call TQNFCalculator::clear() before reusing this instance!");
    return false;
  }
  if(myDataPath.IsNull())
    myDataPath=defaultDataPath;
  // let's first see if there is already a region with this cut name
  // if so, we just update the data path
  for(size_t i=0; i<this->cutNames.size(); i++){
    if(cutNames[i] == cutName){
      dataPaths[i] = myDataPath;
      return false;
    }
  }
  // otherwise, we add it
  this->cutNames.push_back(cutName);
  this->dataPaths.push_back(myDataPath);
  return true;
}

bool TQNFCalculator::addSample(TString mcPath, TString name, bool fixed){
  // add a new monte carlo sample group from the given location
  // optional argument: supply a name for easy reference
  if(this->initialized){
    ERRORclass("cowardly refusing to add sample to already initialized instance - please call TQNFCalculator::clear() before reusing this instance!");
    return false;
  }
  if(name.IsNull()) name = mcPath;
  // let's first see if there is already a sample group with this name
  // if so, we update the path
  for(size_t i=0; i<this->mcNames.size(); i++){
    if(mcNames[i] == name){
      mcPaths[i] = mcPath;
      mcFixed[i] = fixed;
      return false;
    }
  }
  // otherwise, we add it
  this->mcPaths.push_back(mcPath);
  this->mcNames.push_back(name);
  this->mcFixed.push_back(fixed);
  //@tag:[defaultBoundLower,defaultBoundUpper] When adding individual samples via addSample, these object tags determine the minimum/maximum allowed value for the NF of the process unless process specific limits are set. Defaults: 0., 2. .
  this->nfBoundLower.push_back(this->getTagDoubleDefault("defaultBoundLower",0.));
  this->nfBoundUpper.push_back(this->getTagDoubleDefault("defaultBoundUpper",2.));
  return true;
}

bool TQNFCalculator::addSample(TString mcPath, TString name, double boundLower, double boundUpper){
  // add a new monte carlo sample group from the given location
  // optional argument: supply a name for easy reference
  if(this->initialized){
    ERRORclass("cowardly refusing to add sample to already initialized instance - please call TQNFCalculator::clear() before reusing this instance!");
    return false;
  }
  if(name.IsNull()) name = mcPath;
  // let's first see if there is already a sample group with this name
  // if so, we update the path
  for(size_t i=0; i<this->mcNames.size(); i++){
    if(mcNames[i] == name){
      mcPaths[i] = mcPath;
      mcFixed[i] = false;
      nfBoundLower[i] = boundLower;
      nfBoundUpper[i] = boundUpper;
      return false;
    }
  }
  // otherwise, we add it
  this->mcPaths.push_back(mcPath);
  this->mcNames.push_back(name);
  this->mcFixed.push_back(false);
  this->nfBoundLower.push_back(boundLower);
  this->nfBoundUpper.push_back(boundUpper);
  return true;
}

bool TQNFCalculator::addFixSample(TString mcPath, TString name){
  // add a new monte carlo sample group from the given location
  // as opposed to TQNFCalculator::addSample, no NF will be fitted
  // instead, the NF will be fixed to 1
  return this->addSample(mcPath,name,true);
}


bool TQNFCalculator::initializeData(){
  // initialize the data histogram
  if(!this->fReader) return false;
  if(this->data) delete this->data;
  // create the nominal/observed/data histogram
  this->data = new TH1F("data","data",dataPaths.size(),0,dataPaths.size());
  this->data->SetDirectory(NULL);
  // fill it with the nominal values from the default data path
  // here, each region has a separate bin
  //@tag:[readScaleScheme] This object tag determines which scale scheme is used when retrieving values entering the NF calculation (e.g. to include results from previous NF calculation steps). Default: ".nfc.read"
  TString scaleScheme = this->getTagStringDefault("readScaleScheme",".nfc.read");
  scaleScheme.Prepend("scaleScheme=");
  for(size_t i=0; i<this->dataPaths.size(); i++){
    TQCounter* c = this->fReader->getCounter(dataPaths[i],cutNames[i],scaleScheme);
    if(!c){
      ERRORclass("unable to obtain counter '%s' from '%s'!",cutNames[i].Data(),dataPaths[i].Data());
      return false;
    }
    if (chainLoader && iterationNumber > -1) {
      const TString path(dataPaths[i]+":"+cutNames[i]);
      const double variation = chainLoader->getRelVariation(path,c->getCounter(),c->getError());
      DEBUG("variation of path '%s' is '%f'",path.Data(),variation);
      //do not change the uncertainty of c.
      c->setCounter(c->getCounter()*variation);
    }
    this->data->SetBinContent(i+1,c->getCounter());
    this->data->SetBinError(i+1,c->getError());
    this->data->GetXaxis()->SetBinLabel(i+1,cutNames[i]);
    delete c;
  }
  return true;
}

bool TQNFCalculator::ensurePositiveMC(){
  // ensure positive mc for all samples
  bool action = false;
  for(int j=0; j<this->mc->GetEntries(); j++){
    TH1* hist = dynamic_cast<TH1*>(this->mc->At(j));
    for(int i=0; i<hist->GetNbinsX(); i++){
      double val = hist->GetBinContent(i);
      if(val < 0){
        hist->SetBinContent(i,0);
        hist->SetBinError(i,std::max(val,hist->GetBinError(i)));
        action = true;
      }
    }
  }
  return action;
}

bool TQNFCalculator::initializeMC(){
  // initialize all mc histograms
  if(!fReader) return false;
  if(this->mc) delete this->mc;
  this->mc = new TObjArray();
  this->mc->SetOwner(true);
  //tag documentation see initializeData()
  TString scaleScheme = this->getTagStringDefault("readScaleScheme",".nfc.read");
  scaleScheme.Prepend("scaleScheme=");
  for(size_t j=0; j<this->mcPaths.size(); j++){
    // create the histogram corresponding to this sample group
    TH1F* hist = new TH1F(mcNames[j],mcNames[j],cutNames.size(),0,cutNames.size());
    hist->SetDirectory(NULL);
    // fill it with the mc predictions from each sample group
    // here, each region has a separate bin
    for(size_t i=0; i<this->dataPaths.size(); i++){
      TQCounter* c = this->fReader->getCounter(mcPaths[j],cutNames[i],scaleScheme);//running on nominal value
      if(!c){
        this->messages.sendMessage(TQMessageStream::ERROR,"unable to obtain counter '%s' from '%s'!" ,cutNames[i].Data(),mcPaths[j].Data());
        return false;
      }
      if (this->chainLoader && iterationNumber> -1) {// If a TQNFchainloader has been set, we (possibly) run with variations 
        //apply variation without changing the uncertainty of c.
        const TString path(mcPaths[j]+":"+cutNames[i]);
        const double variation = chainLoader->getRelVariation(path,c->getCounter(),c->getError());
        DEBUG("variation of path '%s' is '%f'",path.Data(),variation);
        c->setCounter(c->getCounter()*variation);
      }
      DEBUG("Retrieved counter from path '%s' with value %.2f +/- %.2f",TString(mcPaths[j]+":"+cutNames[i]).Data(),c->getCounter(),c->getError());
      hist->SetBinContent(i+1,c->getCounter());
      hist->SetBinError(i+1,c->getError());
      hist->GetXaxis()->SetBinLabel(i+1,cutNames[i]);
      delete c;
    }
    this->mc->Add(hist);
  }
  return true;
}


bool TQNFCalculator::finalizeSelf(){
  if(this->mc) delete this->mc;
  this->mc = NULL;
  if(this->data) delete this->data;
  this->data = NULL;
  this->NFs.clear();
  this->NFuncertainties.clear();
  return true;
}

bool TQNFCalculator::initializeSelf(){
  // initialize the NF Calculator
  // - initializes the data histogram
  // - initializeds the mc histograms
  // will set the initialize flag to true
  // further calls of initialize will have no effect
  // until clear is called
  if(!this->fReader)
    return false;
  DEBUGclass("initializing data histogram");
  if(!this->initializeData())
    return false;
  DEBUGclass("initializing mc histograms");
  if(!this->initializeMC())
    return false;
  return true;
}

size_t TQNFCalculator::getFloatingSamples(){
  // return the number of "floating" (i.e. non-fixed) samples
  size_t n = 0;
  for(size_t i=0; i<this->mcFixed.size(); i++){
    if(!this->mcFixed[i]){
      n++;
    }
  }
  return n;
}

void TQNFCalculator::calculateNFs_MatrixMode(){
  // in the case where the number of samples
  // and the number of regions are identical
  // the TFractionFitter cannot be used
  // hence, this function implements a matrix calculation
  // for this particular case
  const unsigned int n(data->GetNbinsX());
  if(n != this->getFloatingSamples()){
    this->messages.sendMessage(TQMessageStream::WARNING,"cannot use matrix mode for %d samples in %d regions - only possible for equal N", this->getFloatingSamples(),n);
    this->status = -10;
    return;
  }

  if(verbosity > 1){
    this->messages.sendMessage(TQMessageStream::INFO,"entering matrix mode");
  }
  // setup the data vector and monte carlo matrix
  std::vector<TString> vSampleNames;
  TVectorD vData(n);
  TVectorD vDataErr(n);
  TMatrixD mMC(n,n);
  TMatrixD mMCerr(n,n);
  double max = 1;
  for(size_t i=0; i<n; i++){
    double dataVal = data->GetBinContent(i+1);
    double dataErr2 = pow(data->GetBinError(i+1),2);
    int j = 0;
    for(size_t idx=0; idx<this->mcFixed.size(); idx++){
      if(this->mcFixed[idx]){
        TH1* hist = (TH1*)(this->mc->At(idx));
        dataVal -= hist->GetBinContent(i+1);
        dataErr2 += pow(hist->GetBinError(i+1),2);
      } else {
        TH1* hist = (TH1*)(this->mc->At(idx));
        mMC[i][j] = hist->GetBinContent(i+1);
        if(verbosity > 1) this->messages.sendMessage(TQMessageStream::INFO,"%s in %s is %.3f",mcNames[idx].Data(),cutNames[i].Data(),mMC[i][j]);
        vSampleNames.push_back(mcNames[idx]);
        mMCerr[i][j] = hist->GetBinError(i+1);
        j++;
      }
    }
    max = std::max(max,dataVal);
    vData [i] = dataVal;
    vDataErr[i] = sqrt(dataErr2);
  }
  
  // do the error calculation
  TH1F** tmphists = (TH1F**)malloc(n*sizeof(TH1F*));
  int i = 0;
  for(size_t idx=0; idx<this->mcFixed.size(); idx++){
    if(mcFixed[idx]) continue;
    tmphists[i] = new TH1F(mcNames[idx],TString::Format("Distribution for NF[%s]",mcNames[idx].Data()),100,nfBoundLower[idx],nfBoundUpper[idx]);
    tmphists[i]->SetDirectory(NULL);
    i++;
  }
 
  //@tag:[mode.matrix.nToyHits] (legacy!) This object tag sets the number of toys created when using the matrix inversion method to estimate the uncertainty and improve numerical stability. This is deprecated, use the TQNFChainloader and it's toy capabilities instead! Default: 100
  size_t nSamples = this->getTagIntegerDefault("mode.matrix.nToyHits",100);
  TVectorD vecNFErrs(n);
  TVectorD vecNFs(n);
  if(nSamples > 1){
    // do internal error calculation via random sampling
    TRandom rand;
    TVectorD vtmp(vData);
    for(size_t i=0; i<n; ++i){// loop over MC rows
      for(size_t j=0; j<n; ++j){// loop over MC columns
        for(size_t x = 0; x<nSamples; ++x){// sample mc random hits
          TMatrixD mattmp(mMC);
          mattmp[i][j] = rand.Gaus(mMC[i][j],mMCerr[i][j]);
          mattmp.Invert();
          for(size_t k=0; k<n; ++k){// loop over data entries
            for(size_t y = 0; y<nSamples; ++y){// sample data random hits
              vtmp[k] = rand.Gaus(vData[k],vDataErr[k]);
              TVectorD tmpNFs(mattmp * vtmp);
              for(size_t l=0; l<n; l++){// fill the results
                tmphists[l]->Fill(tmpNFs[l]);
              }
              vtmp[k] = vData[k];
            }
          }
        }
      }
    }
    for(size_t i=0; i<n; ++i){
      vecNFs[i] = tmphists[i]->GetMean();
      vecNFErrs[i] = tmphists[i]->GetRMS();
      if(verbosity > 2){
        this->messages.newline();
        TQHistogramUtils::printHistogramASCII(this->messages.activeStream(),tmphists[i],"");
        this->messages.newline();
      }
      delete tmphists[i];
    }
    free(tmphists);
  } else {
    // do not estimate errors, this was not requested / will be done from outside somehow
    TVectorD vtmp(vData);
    TMatrixD mattmp(mMC);
    #ifdef _DEBUG_
    DEBUG("Printing MC matrix before inversion");
    mattmp.Print();
    #endif
    mattmp.Invert();
    #ifdef _DEBUG_
    DEBUG("Printing MC matrix after inversion");
    mattmp.Print();
    #endif
    TVectorD tmpNFs(mattmp * vtmp);
    for(size_t l=0; l<n; l++){// fill the results
      vecNFs[l] = tmpNFs[l];
      vecNFErrs[l] = 0;
    }
  }

  // test the results and calculate closure
  bool ok = true;
  TVectorD vClosure = mMC * vecNFs;
  double closure = 0;
  for(size_t i=0; i<n; i++){
    /* this check is disabled since sometimes negative results can occur (in particular during toy creation) and not storing anything can mess up the chainloader (to be FIXME'd, this also includes fixes in this class like registering destination paths despite a failed calculation!)
    if(vecNFs[i] <= 0 ){ //
      if(verbosity > 0) this->messages.sendMessage(TQMessageStream::ERROR,"NF[%s] = %.2f <= 0",mcNames[i].Data(),vecNFs[i]);
      ok = false;
    }
    */
    closure += fabs(vClosure[i] - vData[i]);
  }

  this->messages.newline();
  // some printout
  if(verbosity > 2){              
    const int numwidth = log10(max)+3;
    for(size_t i=0; i<n; i++){
      TString line = "( ";
      for(size_t j=0; j<n; j++){
        line += TString::Format("%*.1f ",numwidth,mMC[i][j]);
      }
      line += ") ";
      if(i==0) line += "*";
      else line += " ";
      line += TString::Format(" ( %*.1f ) ",4,vecNFs[i]);
      if(i==0) line += "=";
      else line += " ";
      line += TString::Format(" ( %*.1f ) ",numwidth,vData[i]);
      this->messages.sendMessage(TQMessageStream::INFO,line);
    }
  }

  if(verbosity > 1){
    this->messages.sendMessage(TQMessageStream::INFO,"closure = %g",closure);
    int j = 0;
    for(size_t idx=0; idx<this->mcFixed.size(); idx++){
      if(this->mcFixed[idx]) continue;
      this->messages.sendMessage(TQMessageStream::INFO,"NF[%s] = %.3f +/- %.3f",mcNames[idx].Data(),vecNFs[j],vecNFErrs[j]);
      j++;
    }
  }

  if(!ok){
    this->status = 20;
    return;
  }

  double maxClosure;
  //@tag:[mode.matrix.maxClosure] This object tag determines the maximum closure value allowed (results are discarded if this value is exceeded). Closure is ignored if tag is not set.
  if(this->getTagDouble("mode.matrix.maxClosure",maxClosure) && closure > maxClosure){
    if(verbosity > 1) this->messages.sendMessage(TQMessageStream::WARNING,"closure exceeds maximum '%g', discarding results",maxClosure);
    status = 1;
    return;
  }

  // save the results
  int j = 0;
  for(size_t idx=0; idx<this->mcFixed.size(); idx++){
    if(this->mcFixed[idx]){
      this->NFs.push_back (1);
      this->NFuncertainties.push_back(0);
    } else {
      DEBUG("saving result entry %d : %.2f",j,vecNFs[j]);
      this->NFs.push_back (vecNFs[j]);
      this->NFuncertainties.push_back(vecNFErrs[j]);
      j++;
    }
  }
  status = 0;
}

void TQNFCalculator::calculateNFs_singleMode(){
  // in the case of only one sample group
  // the TFractionFitter cannot be used
  // hence, this function implements a manual calculation
  // for the case of only one sample group
  if(this->getFloatingSamples() > 1){
    this->messages.sendMessage(TQMessageStream::WARNING,"cannot use single mode for %d samples - only possible for nSamples=1", this->getFloatingSamples());
    this->status = -5;
    return;
  }
  double sum = 0;
  double sumw = 0;
  TH1* mchist = NULL;
  if(verbosity > 1){
    this->messages.sendMessage(TQMessageStream::INFO,"entering single mode");
  }
  for(int i=0; i<this->data->GetNbinsX(); i++){
    // we sum up all the different NFs from all regions
    // and calculate the weighted average of the NF 
    // for our one and only sample group
    if(verbosity > 1) this->messages.sendMessage(TQMessageStream::INFO,"looking at region %s",cutNames[i].Data());
    double targetVal = data->GetBinContent(i+1);
    if(verbosity > 1) this->messages.sendMessage(TQMessageStream::INFO,"data is %.3f +/- %.3f",targetVal,data->GetBinError(i+1));
    double targetErr2 = pow(data->GetBinError(i+1),2);
    // first we need to calculate the background-subtracted data value
    // that is, the "target value" we want to use as a reference 
    // to calculate the normalization scale factor.
    for(int j=0; j<this->mc->GetEntries(); j++){
      // fixed samples get subtracted from data
      // while the first (and only) non-fixed sample
      // will become the value we try to scale accordingly
      if(this->mcFixed[j]){
        TH1* hist = (TH1*)(this->mc->At(j));
        if(verbosity > 1) this->messages.sendMessage(TQMessageStream::INFO,"%s is %.3f +/- %.3f",mcNames[j].Data(),hist->GetBinContent(i+1),hist->GetBinError(i+1));
        targetVal -= hist->GetBinContent(i+1);
        targetErr2 += pow(hist->GetBinError(i+1),2);
      } else mchist = (TH1*)(this->mc->At(j));
    }
    if(verbosity > 1){
      this->messages.sendMessage(TQMessageStream::INFO,"background subtracted data: %.3f +/- %.3f",targetVal,sqrt(targetErr2));
      this->messages.sendMessage(TQMessageStream::INFO,"contribution from %s is %.3f +/- %.3f",mchist->GetName(),mchist->GetBinContent(i+1),mchist->GetBinError(i+1));
    }
    // we calculate the weighted average over all contributions we collected
    // including the uncertainty on this value
    // hence, we keep track of the weighted values and the sum of weights
    double val = targetVal/mchist->GetBinContent(i+1);
    double weight = 1./(val*val * (targetErr2 / pow(targetVal,2)
                                   + pow(mchist->GetBinError(i+1),2) / pow(mchist->GetBinContent(i+1),2)));
    if(verbosity > 1) this->messages.sendMessage(TQMessageStream::INFO,"collecting NF = %.3f +/- %.3f from %s",val,1.0/sqrt(weight),cutNames[i].Data());
    sum += val * weight;
    sumw += weight;
  }
  if(verbosity > 1) this->messages.sendMessage(TQMessageStream::INFO,"total resulting NF = %.3f +/- %.3f",sum/sumw,1.0/sqrt(sumw));
  // after the calculation is finished, all we need to do is set the scale factor
  // according to our calculation result
  // for consistency, we need to push an NF of 1 +/- 0 for all fixed samples
  for(size_t i=0; i<this->mcFixed.size(); i++){
    if(this->mcFixed[i]){
      this->NFs.push_back (1);
      this->NFuncertainties.push_back(0);
    } else {
      this->NFs.push_back (sum/sumw);
      this->NFuncertainties.push_back(1.0/sqrt(sumw));
    }
  }
  this->status = 0;
  if(verbosity > 0){
    this->messages.sendMessage(TQMessageStream::INFO,"Results of the NF Calculation are as follows:");
    this->printResults(&(this->messages.activeStream()));
  }
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-security"

void TQNFCalculator::calculateNFs_TFractionFitterMode(){
  // this is the worker function 
  // actually performing the NF calculation
  // in the default case of more than one sample group
  if(this->getFloatingSamples() < 2){
    this->messages.sendMessage(TQMessageStream::WARNING,"cannot use TFractionFitter mode for %d samples - only possible for nSamples>1, please use single mode instead", this->getFloatingSamples());
    this->status = -3;
    return;
  }
  const unsigned int n(data->GetNbinsX());
  if(n <= this->getFloatingSamples()){
    this->messages.sendMessage(TQMessageStream::WARNING,"cannot use TFractionFitter mode for %d samples in %d regions - only possible for #regions > #samples, please use MatrixInversion for #regions == #samples or Simple mode instead", this->getFloatingSamples(),n);
    this->status = -5;
    return;
  }
  if(this->verbosity > 1){
    this->messages.sendMessage(TQMessageStream::INFO,"entering fraction fitter mode");
  }
  if(this->ensurePositiveMC()){
    this->messages.sendMessage(TQMessageStream::INFO,"one or more bins was shifted to yield positive bin contents");
  }
  if(this->messages.isFile()){
    TQLibrary::redirect(this->messages.getFilename(),true);
    this->messages.close();
  } 
  TString voption = "Q";
  if(this->verbosity == 1) voption = "";
  if(this->verbosity > 1) voption = "V";
  TFractionFitter fitter(this->data, this->mc, voption);
  int nFloat = 0;
  for(int i=0; i<this->mc->GetEntries(); i++){
    TH1* mchist = (TH1*)(this->mc->At(i));
    double fraction = mchist->Integral() / this->data->Integral();
    if(mcFixed[i]){
      if(this->verbosity > 0){
        printf("\n\n");
        TString msg = this->messages.formatMessage(TQMessageStream::INFO,"fraction for %s is %.3f (fixed=true)\n",mcNames[i].Data(),fraction);
        printf(msg.Data());
      }
      // for fix samples, we constrain the fraction to the current value
      // watch out, the TFractionFitter starts counting MC samples at i=1
      fitter.Constrain(i+1,std::max(fraction - epsilon,0.),std::min(fraction + epsilon,1.));
    } else {
      if(this->verbosity > 0){
        printf("\n\n");
        TString msg = this->messages.formatMessage(TQMessageStream::INFO,"fraction for %s is %.3f (fixed=false)\n",mcNames[i].Data(),fraction);
        printf(msg.Data());
      }
      // else, we allow it to float in the range [0,1]
      // watch out, the TFractionFitter starts counting MC samples at i=1
      fitter.Constrain(i+1,fraction*this->nfBoundLower[i],fraction*this->nfBoundUpper[i]);
      nFloat++;
    }
  }
  if(this->verbosity > 0){
    printf("\n\n");
    TString msg = this->messages.formatMessage(TQMessageStream::INFO,"running simultaneous fit of %d NFs in %d regions\n",nFloat,(int)cutNames.size()).Data();
    printf(msg.Data());
  }
  this->status = fitter.Fit();
  if(this->messages.isFile()){
    TQLibrary::restore();
    this->messages.reopen();
  }
  if(this->verbosity > 0){
    this->messages.newlines(2);
    this->messages.sendMessage(TQMessageStream::INFO,"performed fit, status is %d\n",this->status);
  }
  if(this->success()){
    for(size_t i=0; i<this->mcNames.size(); i++){
      DEBUGclass("retrieving NF for '%s'",this->mcNames[i].Data());
      double nf = 0;
      double sigma = 0;
      fitter.GetResult(i,nf,sigma);
      // the TFractionFitter returns the relative fraction
      // the given histogram contributes to the data
      // thus, to obtain the NFs, we need to divide the result
      // by the current fraction the histogram contributes to data
      TH1* mchist = dynamic_cast<TH1*>(this->mc->At(i));
      double fraction = 1;
      if(!mchist){
        ERRORclass("internal error: invalid histogram pointer for '%s' at position %d/%d!",this->mcNames[i].Data(),(int)i,this->mc->GetEntries());
      } else {
        fraction = mchist->Integral() / this->data->Integral();
      }
      this->NFs.push_back(nf / fraction);
      this->NFuncertainties.push_back(sigma / fraction);
    }
  }
}

#pragma GCC diagnostic pop

int TQNFCalculator::calculateNFs(const TString& modename){
  // calculate the NFs with a certain method (given by the string)
  // the following tags control the side-effects of this function:
  // - saveLog: string, name of the log file to be used
  // if not given, print to the console
  // - writePreFitHistograms: string, name of the root file to be created
  // if not given, nothing will be written
  // - writePostFitHistograms: string, name of the root file to be created
  // if not given, nothing will be written
  // - saveResults: string, name of the text file to be used for result output
  // if not given, nothing will be written

  if(!this->initialize()){
    this->status = -10;
    return this->status;
  }

  TString rootfilename;
  //@tag:[writePreFitHistograms] This object tag determines if and where (filename) pre-fit histograms are written to.
  if(this->getTagString("writePreFitHistograms",rootfilename)){
    this->exportHistogramsToFile(rootfilename);
  }

  this->calculateNFs(this->getMethod(modename));
  if(this->success()){
    //@tag:[writePostFitHistograms] This object tag determines if and where (filename) post-fit histograms are written to.
    if(this->getTagString("writePostFitHistograms",rootfilename)){
      this->exportHistogramsToFile(rootfilename);
    }
    TString resultfilename;
    //@tag:[saveResults] This object tag determines if and where (filename) results are written to (this is independent from storing the results in the TQSampleFolder structure).
    if(this->getTagString("saveResults",resultfilename)){
      this->writeResultsToFile(resultfilename);
    }
  }
  return this->status;
}

int TQNFCalculator::calculateNFs(){
  // actually perform the NF calculation
  // based on all information supplied earlier
  // the following tags control the side-effects of this function:
  // - saveLog: string, name of the log file to be used
  // if not given, print to the console
  // - writePreFitHistograms: string, name of the root file to be created
  // if not given, nothing will be written
  // - writePostFitHistograms: string, name of the root file to be created
  // if not given, nothing will be written
  // - saveResults: string, name of the text file to be used for result output
  // if not given, nothing will be written
  this->messages.sendMessage(TQMessageStream::INFO,"initializing");
  this->messages.newline();
  if(!this->initialize()){
    this->status = -10;
    return this->status;
  }
  TString prefitrootfilename = this->getPathTag("writePreFitHistograms");
  if(!prefitrootfilename.IsNull()){
    this->exportHistogramsToFile(prefitrootfilename);
  }
  this->messages.sendMessage(TQMessageStream::INFO,"calculateNFs entering evaluation");
  for(size_t i=0; i<this->methods.size(); i++){
    this->messages.newline();
    this->messages.sendMessage(TQMessageStream::INFO,TQStringUtils::repeat("-",40));
    this->messages.newline();
    this->calculateNFs(methods[i]); 
    if(this->success()){
      if(verbosity > 1){
        this->messages.sendMessage(TQMessageStream::INFO,"succeeded in calculating NFs with method '%s'",this->getMethodName(methods[i]).Data());
      }
      break;
    } else {
      this->messages.activeStream().flush();
    }
  }
  if(this->success()){
    TString postrootfilename = this->getPathTag("writePostFitHistograms");
    if(!postrootfilename.IsNull()){
      this->exportHistogramsToFile(postrootfilename);
    }
    TString resultfilename = this->getPathTag("saveResults");
    if(!resultfilename.IsNull()){
      this->writeResultsToFile(resultfilename);
    }
  }
  this->messages.activeStream().flush();
  return status;
}

void TQNFCalculator::setNFsUnity(){
  // set all the NFs to one
  this->NFs.clear();
  this->NFuncertainties.clear();
  this->messages.sendMessage(TQMessageStream::WARNING,"TQNFCalculator::setNFsUnity() was called, all NFs are set to unity");
  for(size_t i=0; i<this->mcPaths.size(); i++){
    this->NFs.push_back(1);
    this->NFuncertainties.push_back(0);
  }
  this->status = 0;
}

void TQNFCalculator::fail(){
  // acquire failure state
  this->NFs.clear();
  this->NFuncertainties.clear();
  this->messages.sendMessage(TQMessageStream::WARNING,"TQNFCalculator::fail() was called, all NFs erased, failure state acquired");
  this->status = -1;
}

int TQNFCalculator::calculateNFs(TQNFCalculator::Method method){
  // actually perform the NF calculation
  // based on all information supplied earlier
  if(!this->initialize()){
    this->status = -10;
    return this->status;
  }
  int nsamples = this->getFloatingSamples();
  if(this->mc->GetEntries() == 0 || nsamples == 0){
    // check if we have anything to do
    // if not, well, let's do nothing at all
    this->status = -100;
    return this->status;
  }
  switch(method){
  case Single:
    // if we only have one sample group
    // we need not (and in fact cannot)
    // use the TFractionFitter
    // instead, we calculate the result manually
    this->calculateNFs_singleMode();
    break;
  case MatrixInversion:
    // if we have one region per sample
    // we can use matrix inversion to obtain
    // the NFs
    this->calculateNFs_MatrixMode();
    break;
  case FractionFitter:
    // if everything went smooth up to here
    // we have at least two sample groups
    // and can invoke the TFractionFitter
    this->calculateNFs_TFractionFitterMode();
    break;
  case Unity:
    this->setNFsUnity();
    break;
  default:
    this->fail();
  }
  this->messages.activeStream().flush();
  return this->status;
}

size_t TQNFCalculator::findIndex(TString name){
  // get the contribution index coresponding to the given NF name
  for(size_t i=0; i<this->mcNames.size(); i++){
    if(mcNames[i] == name)
      return i;
  }
  return -1;
}

double TQNFCalculator::getNF(TString name){
  // get the value of the NF with the given name
  size_t n = this->findIndex(name);
  if(!this->success() || (n > this->mcNames.size())) return std::numeric_limits<double>::quiet_NaN();
  return this->NFs[n];
}


double TQNFCalculator::getNFUncertainty(TString name){
  // get the uncertainty of the NF with the given name
  size_t n = this->findIndex(name);
  if(!this->success() || (n > this->mcNames.size())) return std::numeric_limits<double>::quiet_NaN();
  return this->NFuncertainties[n];
}


bool TQNFCalculator::getNFandUncertainty(TString name, double& nf, double& sigma){
  // get the value and uncertainty of the NF with the given name
  // store both in the values supplied as second and third argument
  size_t n = this->findIndex(name);
  if(n > this->mcNames.size()) return false;
  if(this->mcFixed[n]) return false;
  nf = this->NFs[n];
  sigma = this->NFuncertainties[n];
  //ensure that the NF is within the configured limits. TODO: What to do with the uncertainty in this case?
  nf = std::max(nfBoundLower[n],std::min(nfBoundUpper[n],nf));
  return true;
}

void TQNFCalculator::printRegions(std::ostream* os){
  // print all regions scheduled for NF calculation
  if(!os) return;
  for(size_t i=0; i<cutNames.size(); i++){
    *os << cutNames[i] << "\t" << dataPaths[i] << std::endl;
  }
}

void TQNFCalculator::printSamples(std::ostream* os){
  // print all sample groups scheduled for NF calculatrion
  if(!os) return;
  for(size_t i=0; i<mcNames.size(); i++){
    *os << mcNames[i] << "\t" << mcPaths[i];
    if(this->mcFixed[i]) *os << "\t(fixed)";
    else *os << "(" << this->nfBoundLower[i] << "<NF<" << this->nfBoundUpper[i] << ")";
    *os << std::endl;
  }
}

void TQNFCalculator::printResults(std::ostream* os){
  // print the results of the evaluation
  if(!os || !os->good()) return;
  if(!this->success()){
    WARNclass("no results obtained, status is %d",this->status);
    return;
  }
  for(size_t i=0; i<this->NFs.size(); i++){
    double percent = 100*(this->NFs[i]-1);
    *os << this->mcNames.at(i) << "\t" << this->NFs.at(i) << " +/- " << this->NFuncertainties.at(i) << "\t~ ";
    if(TMath::AreEqualAbs(percent,0,0.05))
      *os << "no NF ";
    if(this->mcFixed.at(i))
      *os << "(fixed)";
    else {
      if(percent > 0)
        *os << "+";
      *os << TString::Format("%.2f",percent) << " %";
    }
    *os << std::endl;
  }
}

TH1* TQNFCalculator::getHistogram(const TString& name){
  // obtain the histogram associated with the given sample group
  // will accept all mc names as well as "data" (for the data histogram)
  if(!this->initialize()) return NULL;
  if(name=="data")
    return this->data;
  for(size_t i=0; i<this->mcNames[i]; i++){
    if(name == mcNames[i])
      return (TH1*)(this->mc->At(i));
  }
  return NULL;
}

bool TQNFCalculator::scaleSample(const TString& name, double val){
  // scale the histogram associated to the sample with the given name by the
  // given value
  if(!TQUtils::isNum(val)) return false;
  TH1* hist = this->getHistogram(name);
  if(!hist) return false;
  hist->Scale(val);
  return true;
}

const TString& TQNFCalculator::getMCPath(const TString& name){
  // retrieve the sample path 
  // corresponding to the sample
  // with the given name
  for(size_t i=0; i<this->mcPaths.size(); i++){
    if(mcNames[i] == name)
      return mcPaths[i];
  }
  return TQStringUtils::emptyString;
}
 
int TQNFCalculator::getStatus(){
  // retrieve the status
  return this->status;
}

TString TQNFCalculator::getStatusMessage(){
  // decode the integer status code
  switch(this->status){
  case -999:
    return "uninitialized";
  case -100: 
    return "no data";
  case -50:
    return "method not implemented";
  case -10:
    return "error during initialization";
  case -5:
    return "method not applicable";
  case -1: 
    return "no method succeeded - fail() called";
  case 0:
    return "all OK";
  case 4: 
    return "TFractionFitter failed to converge";
  case 20: 
    return "matrix method encountered negative NFs";
  default:
    return "unkown error";
  }
}

void TQNFCalculator::printStatus(){
  // print the status
  INFOclass("current status of instance '%s' is '%d': %s",this->GetName(),(int)(this->status),this->getStatusMessage().Data());
}

int TQNFCalculator::execute(int itrNumber) {
  //start calculation
  this->iterationNumber = itrNumber;
  return calculateNFs();
}

bool TQNFCalculator::success(){
  // return true if calculation was successful
  // false otherwise
  if(this->initialized && (this->status == 0))
    return true;
  return false;
}

void TQNFCalculator::setEpsilon(double e){
  // set the epsilon deviation allowed for fixed samples
  // smaller values will increase the accuracy of the calculation
  // but also increase the probability of the fit to fail
  this->epsilon = e;
}

double TQNFCalculator::getEpsilon(){
  // get the current value of the epsilon deviation
  // allowed for fixed samples
  return this->epsilon;
}

int TQNFCalculator::writeResultsToFile(const TString& filename){
  // dump the results of the NF calculation to a text file
  if(!this->success()) return -1;
  TQUtils::ensureDirectoryForFile(filename);
  std::ofstream outfile(filename.Data());
  if(!outfile.is_open()){
    return -1;
  }
  int retval = this->writeResultsToStream(&outfile);
  outfile.close();
  return retval;
}

TString TQNFCalculator::getResultsAsString(const TString& name){
  // dump the results of the NF calculation for a particular contribution
  // into a string that is formatted according to the TQTaggable syntax
  if(!this->success()) return "";
  for(size_t i=0; i<mcNames.size(); i++){
    if(name == mcNames[i]){
      TQTaggable* tags = this->getResultsAsTags(i);
      TString s = tags->exportTagsAsString();
      delete tags;
      return s;
    }
  }
  return "";
}

TQTaggable* TQNFCalculator::getResultsAsTags(const TString& name){
  // dump the results of the NF calculation for a particular contribution
  // into an object of the TQTaggable type
  if(!this->success()) return NULL;
  for(size_t i=0; i<mcNames.size(); i++){
    if(name == mcNames[i]){
      return this->getResultsAsTags(i);
    }
  }
  return NULL;
}

TQTaggable* TQNFCalculator::getResultsAsTags(size_t i){
  // dump the results of the NF calculation for a particular entry
  // into an object of the TQTaggable type
  if(!this->success()) return NULL;
  double nf = 0;
  double sigma = 0;
  if (!this->getNFandUncertainty(mcNames[i],nf,sigma)) return 0;
  TQTaggable* tags = new TQTaggable();
  tags->setTagDouble("normalization",nf);
  tags->setTagDouble("normalizationUncertainty",sigma);
  tags->setTagString("path",this->mcPaths[i]);
  if(!this->mcNames[i].IsNull() && (this->mcNames[i] != this->mcPaths[i])){
    tags->setTagString("name",mcNames[i]);
  }
  TQSampleFolder* sf = this->fReader->getSampleFolder();
  TQSampleFolder * processSampleFolder = sf->getSampleFolder(this->mcPaths[i]);
  TString processTitle = "";
  if (processSampleFolder)
  //@tag:[processTitleKey] This object tag determines the name of the process tag used to retrieve the process title from. Default: "style.default.title".
    processSampleFolder->getTagString(this->getTagStringDefault("processTitleKey","style.default.title"), processTitle);
  if(!processTitle.IsNull()){
    tags->setTagString("title",mcNames[i]);
  }
  if(this->cutInfoFolder && sf && (this->cutInfoFolder->getRoot() == sf->getRoot())){
    tags->setTagString("info",this->cutInfoFolder->getPath());
  }
  return tags;
}

int TQNFCalculator::writeResultsToStream(std::ostream* out){
  // dump the results of the NF calculation to an output stream
  if(!this->success()) return -1;
  TQTaggable* tags;
  for(size_t i=0; i<mcNames.size(); i++){
    tags = this->getResultsAsTags(i);
    if (!tags) continue;
    *out << tags->exportTagsAsString() << std::endl;
    delete tags;
  }
  return 1;
}
 

bool TQNFCalculator::readConfiguration(TQFolder* f){
  // read a configuration from a TQFolder instance
  // the following tags are interpreted:
  // verbosity: integer, increasing value increases verbosity
  // epsilon: double, the 'epsilon'-value for the calculation
  // defaultDataPath: string, the default data path
  // methods: list, methods to be used in order of priority
  // all tags are copied to the calculator
  // 
  // a subfolder 'samples' is expected
  // each folder in 'samples/?' is interpreted as one sample group
  // the following parameters are accepted and interpreted
  // name: string, the name of this group
  // path: string, the path of this group
  // fixed: bool, true for fixed samples, false for floating ones
  // upper: upper bound for this NF (recommended, but not required)
  // lower: lower bound for this NF (recommended, but not required)
  //
  // a subfolder 'regions' is expected
  // each folder in 'regions/?' is interpreted as one sample group
  // the following parameters are accepted and interpreted
  // name: string, the name of this region (name of the counter)
  // datapath: string, the data path to be used for this region
  // if not set, the defaultDataPath will be used
  // active: bool. if false, this region is ignored. default is true.
  // 
  // this function returns true if at least one region and at least one floating
  // sample group have been found and added successfully.
  if(!f) return false;
  this->SetName(f->GetName());
  //@tag:[defaultDataPath] This argument tag sets the default TQFolder path for data (can be overwritten for each region, see tag "datapath").
  this->setDefaultDataPath(f->getTagStringDefault("defaultDataPath","data"));
  //@tag:[epsilon] This argument tag sets the maximum allowed margin inevitable for some numerical methods.
  this->setEpsilon(f->getTagDoubleDefault("epsilon",std::numeric_limits<double>::epsilon()));
  //@tag:[verbosity] This object tag sets the objects verbosity. Imported from argument in TQNFTop1jetEstimator::readConfiguration unless already present. Default: 5 .
  this->setVerbosity(f->getTagIntegerDefault("verbosity",5));
  //@tag:[methods] This argument tag determines the method (and their order) used by the TQNFCalculator. The result is the one by the first successful method.
  std::vector<TString> methodStrings = f->getTagVString("methods");
  for(size_t i=0; i<methodStrings.size(); i++){
    this->addMethod(methodStrings[i]);
  }

  this->importTags(f);

  TQFolderIterator sitr(f->getListOfFolders("samples/?"),true);
  int nSamples = 0;
  while(sitr.hasNext()){
    TQFolder* sample = sitr.readNext();
    if(!sample) continue;
    //@tag:[name] This sub-folder("samples/?") argument tag sets the name of the sample. Defaults to the name of the folder it is read from.
    TString name = sample->getTagStringDefault("name",sample->getName());
    //@tag:[path] This sub-folder("samples/?") argument tag sets the path of the sample. Defaults to the value of the "name" tag.
    TString path = sample->getTagStringDefault("path",name);
    //@tag:[fixed] If this sub-folder("samples/?") argument tag is set to true the sample is left unscaled. Instead it is subtracted from data for the calculation.
    bool fixed = sample->getTagBoolDefault("fixed",false);
    //@tag:[lower,upper] These sub-folder("samples/?") argument tags set the lowest/highest allowed value for the NF corresponding to the sample. Defaults: 0., 2. .
    double boundLower = sample->getTagDoubleDefault("lower",this->getTagDoubleDefault("defaultBoundLower",0.));
    double boundUpper = sample->getTagDoubleDefault("upper",this->getTagDoubleDefault("defaultBoundUpper",2.));
    nSamples += !fixed;
    if(fixed) this->addFixSample(path,name);
    else this->addSample(path,name,boundLower,boundUpper);
  }
  if(nSamples < 1) return false;
 
  int nRegions = 0;
  TQFolderIterator ritr(f->getListOfFolders("regions/?"),true);
  while(ritr.hasNext()){
    TQFolder* region = ritr.readNext();
    if(!region) continue;
    //@tag:[name] This sub-folder("regions/?") argument tag sets the name of the (control) region. Defaults to the name of the respective folder it is read from.
    TString name = region->getTagStringDefault("name",region->getName());
    //@tag:[datapath] This sub-folder("regions/?") argument tag determines the TQFolder path where data for this region should be retrieved from. Default: "".
    TString mydatapath = region->getTagStringDefault("datapath","");
    //@tag:[active] This sub-folder("regions/?") argument tag allows to disable a region by setting the tag to false. Default: true.
    bool active = region->getTagBoolDefault("active",true);
    if(active){
      this->addRegion(name,mydatapath);
      nRegions++;
    }
  }
  if(nRegions < 1) return false;

  return true;
}

TH1* TQNFCalculator::getMCHistogram(const TString& name){
  // retrieve the histogram 
  // corresponding to the sample
  // with the given name
  if(!this->initialize()) return NULL;
  for(size_t i=0; i<this->mcPaths.size(); i++){
    if(mcNames[i] == name)
      return dynamic_cast<TH1*>(mc->At(i));
  }
  return NULL;
} 

TQSampleFolder* TQNFCalculator::exportHistograms(bool postFit){
  // export the histograms to a new sample folder
  if(!this->initialize()) return NULL;
  TQSampleFolder* folder = TQSampleFolder::newSampleFolder(this->GetName());
  if(!this->exportHistogramsToSampleFolder(folder,postFit)){
    delete folder;
    return NULL;
  }
  return folder;
}

bool TQNFCalculator::exportHistogramsToFile(const TString& destination, bool recreate, bool postFit){
  // export the histograms to some root file
  // the 'recreate' flag controls if the file is opened in APPEND or RECREATE mode
  // if postfit=true, the post-fit histograms are saved (if possible)
  // if the calculation has not yet been performed or has failed, the pre-fit ones are saved instead
  TQSampleFolder* folder = this->exportHistograms(postFit);
  if(!folder) return false;
  if(!TQUtils::ensureDirectoryForFile(destination)){
    ERRORclass("unable to access or create directory for file '%s'",destination.Data());
    return false;
  }
  if(!folder->writeToFile(destination,recreate)){
    ERRORclass("unable to write folder '%s' to file '%s'",folder->GetName(),destination.Data());
    delete folder;
    return false;
  }
  delete folder;
  return true;
}

bool TQNFCalculator::exportHistogramsToSampleFolder(TQSampleFolder* folder, bool postFit){
  // export the histograms to an existing sample folder
  // if postfit=true, the post-fit histograms are saved (if possible)
  // if the calculation has not yet been performed or has failed, the pre-fit ones are saved instead
  if(!this->initialize() || !folder) return false;
  TQFolder* plotconf = folder->getFolder("scheme+");
  TQSampleFolder* s = folder->getSampleFolder("data+");
  TList* sources = this->fReader->getListOfSampleFolders(defaultDataPath);
  if(sources && sources->GetEntries() > 0){
    TQFolder* source = dynamic_cast<TQFolder*>(sources->First());
    source->exportTags(s,"","style.*",true);
  }
  delete sources;
  TQFolder* f = s->getFolder(".histograms+");
  TH1* hist = dynamic_cast<TH1*>(this->data->Clone());
  if(!hist) return false;
  hist->SetName(this->GetName());
  f->addObject(hist);
  plotconf->setTagString(".processes.0..path","data");
  plotconf->setTagString(".processes.0..name","data");
  plotconf->setTagBool(".processes.0..isData",true);
  for(size_t i=0; i<this->mcPaths.size(); i++){
    TString processTag = TString::Format(".processes.%d.",(int)i+1);
    TQSampleFolder* s = folder->getSampleFolder(mcNames[i]+"+");
    TList* sources = this->fReader->getListOfSampleFolders(mcPaths[i]);
    plotconf->setTagString(processTag+".path",mcNames[i]);
    plotconf->setTagString(processTag+".name",mcNames[i]);
    plotconf->setTagBool(processTag+".isBackground",true);
    plotconf->setTagBool(processTag+".isFixed",mcFixed[i]);
    if(sources && sources->GetEntries() > 0){
      TQFolder* source = dynamic_cast<TQFolder*>(sources->First());
      source->exportTags(s,"","style.*",true);
    }
    delete sources;
    TQFolder* f = s->getFolder(".histograms+");
    TH1* hist = dynamic_cast<TH1*>(mc->At(i)->Clone());
    if(this->success() && postFit) hist->Scale(this->NFs[i]);
    hist->SetName(this->GetName());
    f->addObject(hist);
  }
  return true;
}


bool TQNFCalculator::addMethod(const TString& methodName){
  // add a method at the back of the queue
  Method method = this->getMethod(methodName);
  if(method != UNDEFINED){
    this->methods.push_back(method);
    return true;
  }
  return false;
}
void TQNFCalculator::clearMethods(){
  // empty the method queue
  this->methods.clear();
}
void TQNFCalculator::printMethods(){
  // print all currently scheduled methods in order of priority
  for(size_t i=0; i<this->methods.size(); i++){
    std::cout << i+1 << ") " << this->getMethodName(methods[i]);
  }
}
 
