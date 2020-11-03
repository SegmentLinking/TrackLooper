#include "QFramework/TQNFChainloader.h"
#include "QFramework/TQStringUtils.h"
#include "QFramework/TQIterator.h"
#include "QFramework/TQUtils.h"
#include "TMath.h"
#include "TRandom.h"
#include "QFramework/TQHistogramUtils.h"
#include "QFramework/TQNFCalculator.h"
#include "QFramework/TQABCDCalculator.h"
#include "QFramework/TQNFManualSetter.h"
#include "QFramework/TQNFTop0jetEstimator.h"
#include "QFramework/TQNFTop1jetEstimator.h"
#include "QFramework/TQNFCustomCalculator.h"
#include "QFramework/TQNFUncertaintyScaler.h"
#include "TMatrixD.h"
#include "TFile.h"

#include <limits>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <fstream>

//#define _DEBUG_
#include "QFramework/TQLibrary.h"

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQNFChainloader
//
// This class provides a single interface to all available NF calculation
// methods. Given one or more apropriate configurations it will run the
// corresponding NF calculators in an order according to the configurations
// while keeping track of all NFs saved to the sample folder provided to the
// chainloader. The NF calculators used in this class must inherit from TQNFBase.
//
// The execution of the specified NF calculators can be iterated by setting the
// integer tag "numberOfIterations" (default 1) to the desired number of iterative 
// executions.
//
// NF uncertainties and correlations are computed when the integer tag "toySize" 
// (default 1) is set and differs from its default value. The value of this tag
// determines how many toy NFs are created in order to compute sample 
// (co)variances used as uncertainty estimates. The toy NFs are created by randomly 
// varying input counters (data and MC) according to a gaussian distribution using
// the uncertainties of the input quantities. In order to have consistent variations
// within the calculation process of one toy NF the TQNFChainloader provides these 
// variations to the NF calculators upon request through getRelVariation
// (const TString& key, double value, double uncertainty).
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQNFChainloader)

TQNFChainloader::TQNFChainloader():
TQNamedTaggable(),
  fReader(NULL),
  status(-999),
  verbosity(0)
{
  // default constructor
  rnd=TRandom3();
}

TQNFChainloader::TQNFChainloader(const std::vector<TString>& nfconfigs, TQSampleFolder* f ):
  TQNamedTaggable(),
  fReader(new TQSampleDataReader(f)),
  vNFconfig(nfconfigs),
  status(-999),
  verbosity(0)
{
  // constructor taking vector of nf config file names and a base sample folder
  rnd=TRandom3();
}

TQNFChainloader::~TQNFChainloader(){
  // default destructor, cleaning up all pointer fields
  if(fReader)
    delete fReader;
  this->finalize();
  this->clear();
}

int TQNFChainloader::addNFconfigs(const std::vector<TString>& nfconfigs) {
  this->vNFconfig.insert( this->vNFconfig.end(), nfconfigs.begin(), nfconfigs.end() );
  return 0;
}

int TQNFChainloader::setNFconfigs(const std::vector<TString>& nfconfigs){
  this->finalize();
  this->addNFconfigs(nfconfigs);
  return 0;
}

void TQNFChainloader::setSampleFolder(TQSampleFolder* f){
  if (!f) return;
  if(!this->fReader){
    this->fReader = new TQSampleDataReader(f);
  }
}


bool TQNFChainloader::initialize(){
  if(!this->fReader) return false;
  //@tag: [pathprefix] This object tag is forwarded to the individual NF calculators as "saveResults.prefix", "writePostFitHistograms.prefix", "writePreFitHistograms.prefix" and  "saveLog.prefix".
  TString prefix = this->getTagStringDefault("pathprefix","");
  TQSampleFolder* sf = this->fReader->getSampleFolder();
  TQFolder * nfinfo = sf->getFolder("info/normalization+");
  TQFolder * cutinfo = sf->getFolder("info/cuts");
  for (uint v=0; v<this->vNFconfig.size(); v++) {
    bool execAtEnd = false; //will be set to true, if a method is configured which should be executed after the normal calculation cycle (e.g. TQNFUncertaintyScaler)
    TString nfconfigname = this->vNFconfig.at(v);
    TString errMsg;
    TQFolder * nfconfig = TQFolder::loadFromTextFile(nfconfigname,errMsg);
    if (!nfconfig){
      ERRORclass("unable to import configuration from '%s', error message was '%s'",nfconfigname.Data(),errMsg.Data());
      continue;
    }
    nfconfig->replaceInFolderTags(this->fVariationTags,"*");
    //tmp
    //std::cout<<"----------------------------------\nPrinting NF config after replacements:"<<std::endl;
    //nfconfig->print("rdt");
    //tmp
    INFOclass("setting up normalization factor configuration '%s'",nfconfigname.Data());
    TQFolderIterator itr(nfconfig->getListOfFolders("?"),true);
    while(itr.hasNext()){
      TQFolder * conf = itr.readNext();
      if (!conf) continue;
      //@tag: [mode] This config tag determines the type of NF calculator to be called for a calculation step. Default: "TQNF", other values: "TOP0JET", "TOP1JET", "MANUAL", "ABCD", "CUSTOM", "ERRSCALE"(special, see TQNFUncertaintyScaler!)
      TString mode = conf->getTagStringDefault("mode","TQNF");
      #ifdef _DEBUG_
      std::cout<<"mode = "<<mode<<std::endl;
      #endif
      //@tag: [applyToCut,stopAtCut] The "applyToCut" config tag determines the cut at which the calculated NFs are applied. It is also applied to all subsequent cuts, untill a cut with name equal to the value listed in "stopAtCut" is reached. Please note that some NF calculators do not have a single set of start/stop cuts and need a slightly different configuration.
      if (!(conf->hasTag("applyToCut") || conf->hasTag("applyToCut.0") ) && !TQStringUtils::equal(mode,"MANUAL") && !TQStringUtils::equal(mode,"ERRSCALE") && !TQStringUtils::equal(mode,"UNCSCALE") && !TQStringUtils::equal(mode,"UNCERTSCALE")) { //TQNFManualSetter and TQNFUncertaintyScaler do not have a single starting cut
        WARNclass("skipping calculation of NFs for instance '%s - no target cut given!",conf->getName().Data());
        continue;
      }

      TString stopatcutname = conf->getTagStringDefault("stopAtCut","");
      
      TQNFBase* nfbase = NULL;
      DEBUGclass("instantiating TQNFBase of type '%s'",mode.Data());
      if(TQStringUtils::equal(mode,"TQNF")){
        TQNFCalculator* tqnf = new TQNFCalculator(this->fReader);
        if (!tqnf->readConfiguration(conf)) {
          ERRORclass("cannot calculate NFs for instance '%s' - there was an error parsing the configuration!",conf->getName().Data());
          continue;
        }
        tqnf->setTag("mode.matrix.nToyHits",1);
        nfbase = tqnf;
      } else if (TQStringUtils::equal(mode,"ABCD")) {
        TQABCDCalculator * abcd = new TQABCDCalculator(this->fReader);
        if (!abcd->readConfiguration(conf)) {
          ERRORclass("cannot perform ABCD for instance '%s' - there was an error parsing the configuration!",conf->getName().Data());
          continue;
        }
        nfbase=abcd;
      } else if (TQStringUtils::equal(mode,"TOP0JET")) {
        TQNFTop0jetEstimator * top0jet = new TQNFTop0jetEstimator(this->fReader);
        if (!top0jet->readConfiguration(conf)) {
          ERRORclass("cannot perform TOP0JET for instance '%s' - there was an error parsing the configuration!",conf->getName().Data());
          continue;
        }
        nfbase=top0jet;
      } else if (TQStringUtils::equal(mode,"TOP1JET")) {
        TQNFTop1jetEstimator * top1jet = new TQNFTop1jetEstimator(this->fReader);
        if (!top1jet->readConfiguration(conf)) {
          ERRORclass("cannot perform TOP1JET for instance '%s' - there was an error parsing the configuration!",conf->getName().Data());
          continue;
        }
        nfbase=top1jet;
      } else if (TQStringUtils::equal(mode,"MANUAL")) {
        TQNFManualSetter * manSetter = new TQNFManualSetter(this->fReader);
        if (!manSetter->readConfiguration(conf)) {
          ERRORclass("cannot perform MANUAL for instance '%s' - there was an error parsing the configuration!",conf->getName().Data());
          continue;
        }
        nfbase=manSetter;
      } else if (TQStringUtils::equal(mode,"CUSTOM")) {
        TQNFCustomCalculator * custCalc = new TQNFCustomCalculator(this->fReader);
        if (!custCalc->readConfiguration(conf)) {
          ERRORclass("cannot perform MANUAL for instance '%s' - there was an error parsing the configuration!",conf->getName().Data());
          continue;
        }
        nfbase=custCalc;
      } else if (TQStringUtils::equal(mode,"ERRSCALE") || TQStringUtils::equal(mode,"UNCSCALE") || TQStringUtils::equal(mode,"UNCERTSCALE")) {
        TQNFUncertaintyScaler * uncScaler = new TQNFUncertaintyScaler(this->fReader);
        if (!uncScaler->readConfiguration(conf)) {
          ERRORclass("cannot perform ERRSCALE/UNCSCALE/UNCERTSCALE for instance '%s' - there was an error parsing the configuration!",conf->getName().Data());
          continue;
        }
        nfbase=uncScaler;
        execAtEnd = true;
      }
      DEBUGclass("done");
      if(nfbase){
        DEBUGclass("performing final setup");
	//        nfbase->setTagString("applyToCut",cutname);
	//        nfbase->setTagString("stopAtCut",stopatcutname);
	nfbase->importTags(conf);
        nfbase->setTagString("saveLog.prefix",prefix);
        nfbase->setTagString("writePreFitHistograms.prefix",prefix);
        nfbase->setTagString("writePostFitHistograms.prefix",prefix);
        nfbase->setTagString("saveResults.prefix",prefix);
        nfbase->setChainLoader(this);
        nfbase->setCutInfoFolder(cutinfo);
        nfbase->setInfoFolder(nfinfo);
        if (!execAtEnd) {
        this->vNFcalc.push_back(nfbase);
        } else {
        this->vNFpostCalc.push_back(nfbase);
        }
        DEBUGclass("done");
      } else {
        WARNclass("unable to initialize NF calculator '%s'",conf->GetName());
      }
    }
  }
  this->initialized = true;
  return true;
}


bool TQNFChainloader::finalize(){
  if(this->initialized){
    for(size_t i=0; i<vNFcalc.size(); ++i){
      #ifdef _DEBUG_
      DEBUGclass("finalizing NFBase '%s'",vNFcalc[i]->GetName());
      #endif
      vNFcalc[i]->finalize();
      #ifdef _DEBUG_
      DEBUGclass("deleting");
      #endif
      delete vNFcalc[i];
      #ifdef _DEBUG_
      DEBUGclass("done");
      #endif
    }
    this->vNFcalc.clear();
    if (this->vNFpostCalc.size()>0) {
      #ifdef _DEBUG_
      DEBUGclass("Executing post-calculation methods)");
      #endif
      for (size_t i=0; i<vNFpostCalc.size(); ++i) {
        //########################################
        TQNFBase* nfcalc = this->vNFpostCalc[i];
        #ifdef _DEBUG_
        DEBUGclass("initializing '%s'",nfcalc->GetName());
        #endif
        if(!nfcalc->initialize()){
          ERRORclass("unable to initialize calculator '%s'",nfcalc->GetName());
          continue;
        }
        #ifdef _DEBUG_
        DEBUGclass("executing '%s'",nfcalc->GetName());
        #endif
        int retval = nfcalc->execute(-1);
        if (0 != retval){
          ERRORclass("an error occured while performing calculation '%s' (return value: %d)!",nfcalc->GetName(),retval);
          continue;
        }
        std::vector<TString> startAtCutNames = nfcalc->getTagVString("applyToCut");
        std::vector<TString> stopatcutname = nfcalc->getTagVString("stopAtCut");
        if (!nfcalc->success()){
          WARN("TQNFBase '%s' failed to calculate NFs for cut(s) '%s' with status %d: %s!",nfcalc->GetName(),TQListUtils::makeCSV(startAtCutNames).Data(),nfcalc->getStatus(),nfcalc->getStatusMessage().Data());
          return false;
        } else if (!(nfcalc->deployResult(startAtCutNames,stopatcutname)>0)){
          ERRORclass("unable to deploy results of NF calculation!");
          return false;
        }
        #ifdef _DEBUG_
        DEBUGclass("finalizing '%s'",nfcalc->GetName());
        #endif
        nfcalc->finalize();
        //########################################################
      }
    }
  }
  for(size_t i=0; i<vNFpostCalc.size(); ++i){
      #ifdef _DEBUG_
      DEBUGclass("finalizing NFBase '%s'",vNFpostCalc[i]->GetName());
      #endif
      vNFpostCalc[i]->finalize();
      #ifdef _DEBUG_
      DEBUGclass("deleting");
      #endif
      delete vNFpostCalc[i];
      #ifdef _DEBUG_
      DEBUGclass("done");
      #endif
    }
  this->vNFpostCalc.clear();
  //perform post calculation operations  (e.g. uncertainty scaling via TQNFUncertaintyScaler)
  this->initialized = false;
  return true;
}

int TQNFChainloader::execute() {
  if(!this->initialized){
    ERRORclass("cannot bootstrap NF calculation uninitialized!");
    return -1;
  }
  //@tag: [numberOfIterations] This object tag specifies how often the serial calculation of NFs --as specified in the respective config file-- should be repeated. Since the order in which the calculations are performed generally influences the result, this can be used this option can be used to possibly make the resulting NFs converge, i.e. become independent from the order of calculation. Default: 1
  //@tag: [toySize] This specifies the number of toy NFs created to estimate the uncertainty of the resulting NFs (and their correlation). For each toy NF all input quantities are varied within their uncertainties as present in the sample file at the time of calculation. This usually does not include systematic uncertainties! The variations are ensured to be constant during the calculation of one toy. If the default value of 1 is not changed, no variations and no uncertainty estimate is performed.
  int iterations = this->getTagDefault("numberOfIterations",1);
  int nToy = this->getTagDefault("toySize",1);
  bool doVariations = (nToy>1);
  #ifdef _DEBUG_
  DEBUGclass("bootstrapping nf calculations");
  #endif
  for (int t = 0;t<nToy;t++) {
    #ifdef _DEBUG_
    DEBUGclass("rolling toy %d/%d",t+1,nToy);
    #endif
    for (int i=0;i<iterations;++i) {
      #ifdef _DEBUG_
      DEBUGclass("undergoing iteration %d/%d",i+1,iterations);
      #endif
      for(size_t k=0; k<this->vNFcalc.size(); k++){
        TQNFBase* nfcalc = this->vNFcalc[k];
        #ifdef _DEBUG_
        DEBUGclass("initializing '%s'",nfcalc->GetName());
        #endif
        if(!nfcalc->initialize()){
          ERRORclass("unable to initialize calculator '%s'",nfcalc->GetName());
          continue;
        }
        #ifdef _DEBUG_
        DEBUGclass("executing '%s'",nfcalc->GetName());
        #endif
        int retval = nfcalc->execute(doVariations ? t : -1);
        if (0 != retval){
          ERRORclass("an error occured while performing calculation '%s' (return value: %d)!",nfcalc->GetName(),retval);
          continue;
        }

        std::vector<TString> startAtCutNames = nfcalc->getTagVString("applyToCut");
        std::vector<TString> stopatcutname = nfcalc->getTagVString("stopAtCut");
        if (!nfcalc->success()){
          WARN("TQNFBase '%s' failed to calculate NFs for cut '%s' with status %d: %s!",nfcalc->GetName(),TQListUtils::makeCSV(startAtCutNames).Data(),nfcalc->getStatus(),nfcalc->getStatusMessage().Data());
          return -1;
        } else if (!nfcalc->deployResult(startAtCutNames,stopatcutname)){
          ERRORclass("unable to deploy results of NF calculation!");
          return -1;
        } else {
          DEBUGclass("successfully deployed results for '%s' to cuts '%s'-'%s'",nfcalc->GetName(),TQStringUtils::concat(startAtCutNames,",").Data(),TQStringUtils::concat(stopatcutname,",").Data());
        }
        if (t==0 && doVariations){
          this->registerTargetPaths(nfcalc->getNFpaths()); //only required during first toy-iteration
        }
        #ifdef _DEBUG_
        DEBUGclass("finalizing '%s'",nfcalc->GetName());
        #endif
        nfcalc->finalize();
        #ifdef _DEBUG_
        DEBUGclass("done!");
        #endif
      }
    }
    if (doVariations) {
      this->relativeVariationMap.clear();
      this->collectNFs();
    }
    #ifdef _DEBUG_
    DEBUGclass("end of iteration");
    #endif
  }
  if (doVariations) {
#ifdef _DEBUG_
    DEBUGclass("deploying NFs");
    for(size_t i=0; i<this->vNFcalc.size(); ++i){
      this->vNFcalc[i]->printNFPaths();
    }
#endif
    this->deployNFs();
  }
  DEBUGclass("end of function");
  return 0;
}

double TQNFChainloader::getRelVariation(const TString& key, double value, double uncertainty) {
  // key = [path to counter]:[cut name]
  // This function allows for NF calculators to retrieve consistent random variations of
  // their input (event)counters in order to create toy samples and estimate NF uncertainties.
  // The returned value is a multiplicative factor, allowing for it to be applied to the nominal
  // counter even if some (preliminary) scale factors have already been applied.
  // If no variation is present for key, a new one is generated using 
  // TRandom3::Gaus(value,uncertainty) and stored unless value == 0.
  if (key=="") {
    WARN("No valid key has been given! Returning 0!");
    return 0;
  }
 
  if (this->relativeVariationMap.count(key)==0) { //entry does not yet exist, so we create a new one
    if (value == 0) return 0; //in case the value is zero, the relative variation does not matter anyways. However, this might be due to some scale factor, so we do not want to store a variation.
    this->relativeVariationMap[key] = (rnd.Gaus(value,uncertainty))/value;
  }
  double retval = this->relativeVariationMap[key];
  return retval;
 
}

int TQNFChainloader::registerTargetPaths(const std::vector<TString>& vec){
  int npre = this->targetNFpaths.size();
  for (uint i=0;i<vec.size();i++){
    this->targetNFpaths[vec[i]] = true;
  }
  return this->targetNFpaths.size()-npre;
}

int TQNFChainloader::collectNFs() {
  int nNFs = 0;
  for (auto& item : this->targetNFpaths) {
    TString path = item.first;
    TString cut = TQFolder::getPathTail(path);
    TString scheme = TQStringUtils::readPrefix(cut, ":", ".default");
    double nf = 1;
    TQSampleFolder* sf = this->fReader->getSampleFolder();
    if (sf->getSampleFolder(path)){
      nf = sf->getSampleFolder(path)->getScaleFactor(scheme+":"+cut);
      sf->getSampleFolder(path)->setScaleFactor(scheme+":"+cut,1.,0.);
    }else {
      ERRORclass("Failed to retrieve scale factor, results may be unreliable!");
    }
    mNFvectors[item.first].push_back(nf);
    nNFs++;
  }
  return nNFs;
}

int TQNFChainloader::deployNFs() {
  int nNFs = 0;
  std::vector<TString> names;
  std::vector<TString> identifier;
  std::vector<std::vector<double>> toys;
  for (auto& item : this->mNFvectors) {
    names.push_back(item.first);
    toys.push_back(item.second);
  }
  //@tag: [doNFplots] If this object tag is set to true, a root file is produced with distributions of the obtained toy NFs. Default: false.
  //@tag: [NFplotsPattern,NFplotsFile] The vector valued object tag "NFplotsPattern" specifies filtes specifying which NF toy distributions are to be exported to the root file specified in the object tag "NFplotsFile". The distribution of an NF is exported if it matches any of the comma seperated filter expressions. Example: *bkg/ee/*/.default:CutWeights (syntax: path/scaleScheme:cut)
  if (this->getTagBoolDefault("doNFplots",false)) {
    std::vector<TString> NFpatterns;
    this->getTag("NFplotsPattern",NFpatterns);
    if (NFpatterns.size()>0) {
      //we first collect the indices of the NFs we want to plot
      std::vector<uint> indicesToPlot;
      for (uint i=0; i<names.size(); ++i) {
        for (uint j=0;j<NFpatterns.size();++j) {
          if (TQStringUtils::matches(names.at(i),NFpatterns.at(j))) {
            indicesToPlot.push_back(i);
            break;
          }
        }
      }
      TString NFPlotsFileName = this->getTagStringDefault("NFplotsFile","NFplots.root");
      TQUtils::ensureDirectoryForFile(NFPlotsFileName);
      TFile* f = TFile::Open(NFPlotsFileName,"RECREATE");
      for (uint i=0; i<indicesToPlot.size();++i) {
        uint index = indicesToPlot.at(i);
        double avg =  TQUtils::getAverage(toys.at(index));
        double stddev = sqrt(TQUtils::getSampleVariance(toys.at(index)));
        //@tag: [NFplotsEntriesPerBin] When histograms of NF distributions are produced, this object tag determines the binning of the histograms. It is chosen such, that the average number of toy NFs per bin is (roughly) equal to the value specified in this tag. Default: 10.
        //generic histograms showing +/-3sigma around average and 
        TH1F* hist1 = new TH1F(names.at(index),names.at(index), (int)ceil(this->getTagIntegerDefault("toySize",1)/this->getTagDoubleDefault("NFplotsEntriesPerBin",10.)) ,avg-3*stddev , avg+3*stddev);
        for (uint j=0; j<toys.at(index).size();++j) {
          hist1->Fill(toys[index][j]);
        }
        hist1->Write();
        delete hist1;
        //@tag: [NFplotsLow,NFplotsHigh] These object tags specify custom lower and upper boundaries for histograms of the NF toy distributions if both are set. Please note that dynamic versions showing a +/- 3 sigma interval are produced independently from whether these tags are seet or not.
        //if xmin and xmax for NF plots has been specified we also make plots according to these values:
        if (this->hasTagDouble("NFplotsLow") && this->hasTagDouble("NFplotsHigh") ) {
          TH1F* hist2 = new TH1F(names.at(index),names.at(index), (int)ceil(this->getTagIntegerDefault("toySize",1)/this->getTagDoubleDefault("NFplotsEntriesPerBin",10.)) ,this->getTagDoubleDefault("NFplotsLow",0.) , this->getTagDoubleDefault("NFplotsHigh",2.));
          for (uint j=0; j<toys.at(index).size();++j) {
            hist2->Fill(toys[index][j]);
          }
          hist2->Write();
          delete hist2;
        }
        
        //for correlation (scatter) plots we provide only one version for now
        //@tag: [doNFcorrelationPlots] If this object tag is set to true, 2D correlation plots (scatter plots) of all combinations of NFs passing the filter specified via the object tag "NFplotsPattern" are produced. Default: false.
        if (this->getTagBoolDefault("doNFcorrelationPlots",false) && i<indicesToPlot.size()-1) {
          for (uint j=i+1; j<indicesToPlot.size(); ++j) {
            uint index2 = indicesToPlot.at(j);
            TGraph* gr = TQHistogramUtils::scatterPlot(names.at(index) + " vs " + names.at(index2), toys.at(index), toys.at(index2), names.at(index), names.at(index2) );
            gr->Write();
            delete gr;
          }
        } 
      }
      f->Close();
    } else {
      WARN("No pattern for NF plots given, skipping!");
    }
    
  }
  
  
  for (uint i=0; i<names.size(); ++i) {
    identifier.push_back(TString::Format("%lu_%d",TQUtils::getCurrentTime(),i)); //create a unique identifier for each NF (used for correlation treatment)
  }
#ifdef _DEBUG_
  for(size_t i=0; i<names.size(); ++i){
    std::cout << names[i] << "\t";
    for(size_t j = 0; j < toys[i].size(); ++j){
      std::cout << toys[i][j] << " ";
    }
    std::cout << std::endl;
  }
#endif

  TMatrixD correlations(names.size(),names.size());
  bool ok = true;
  for (uint i=0;i<names.size();i++) {
    for (uint j=0;j<names.size();j++) {
      double val = TQUtils::getSampleCorrelation(toys.at(i),toys.at(j),ok);
      if(TQUtils::isNum(val)){
        correlations[i][j] = val;
      } else {
        ok = false;
        correlations[i][j] = 0;
      } 
    }
  }
  TQSampleFolder* sf = this->fReader->getSampleFolder();
  TQSampleFolder* writeOut = NULL;
  if (this->hasTagString("exportNFsTo")) writeOut = TQSampleFolder::newSampleFolder("writeOut");
  
  //TString savePlotsTo;
  //TFile* f = NULL;
  //this needs to be reimplemented in a reasonable way: (saving NF (toy-)distribution plots somewhere)
  //if (this->getTagString("saveNFplots",savePlotsTo) ) f = TFile::Open(savePlotsTo,"RECREATE");
  //else std::cout<<"No target file specified for NF plots!"<<std::endl;
  for (uint i=0;i<names.size();i++) {
    TString path = names.at(i); //format: path/.scalescheme:cut
    TString cut = TQFolder::getPathTail(path);
    TString scheme = TQStringUtils::readPrefix(cut, ":", ".default");
    TQSampleFolder* subFolder = sf->getSampleFolder(path);
    TQSampleFolder* subWriteOut = NULL;
    if (writeOut) subWriteOut = writeOut->getSampleFolder(path+"+");
    
    if (!subFolder || (writeOut && !subWriteOut) ) {
      ERRORclass("Failed to deploy NF!");
      break;
    }
    double average = TQUtils::getAverage(toys.at(i));
    double uncertainty = sqrt(TQUtils::getSampleVariance(toys.at(i)));
    #ifdef _DEBUG_
    DEBUGclass("setting scale factor '%s:%s' = %f +/- %f with ID='%s' on '%s'",scheme.Data(),cut.Data(),average,uncertainty,identifier.at(i).Data(),subFolder->getPath().Data());
    #endif
    int n = subFolder->setScaleFactor(scheme+":"+cut,identifier.at(i),average,uncertainty);
    if (subWriteOut) subWriteOut->setScaleFactor(scheme+":"+cut,identifier.at(i),average,uncertainty);
    if(n == 0){
      WARNclass("failure to set scale factor on '%s', skipping",subFolder->getPath().Data());
    } else {
      //save  correlation information
      TQFolder* corrFolder = subFolder->getFolder(TQFolder::concatPaths(".scalefactors/",scheme,".correlations/",identifier.at(i))+"+");
      TQFolder* corrWriteOut = NULL;
      if (subWriteOut) corrWriteOut = subWriteOut->getFolder(TQFolder::concatPaths(".scalefactors/",scheme,".correlations/",identifier.at(i))+"+");
      for(uint j=0; j<identifier.size();++j) {
        TQCounter* corrCounter = new TQCounter(identifier.at(j),correlations[i][j],0.);
        corrFolder->addObject(corrCounter,"");
        if (corrWriteOut) {
          TQCounter* corrCounterWriteOut = new TQCounter(identifier.at(j),correlations[i][j],0.);
          corrWriteOut->addObject(corrCounterWriteOut,"");
        }
      }
    }
    nNFs+=n;
  }
  if (writeOut) {
    //@tag: [exportNFsTo] Sets the name of a file to which the calculated NFs should be exported (in addition to storing them in the regular sample folder).
    TString exportFileName =  this->getTagStringDefault("exportNFsTo","");
    if (exportFileName.EndsWith(".root",TString::kIgnoreCase)) writeOut->writeToFile(exportFileName, true,-1,true);
    else writeOut->exportToTextFile(exportFileName,true);
    delete writeOut;
  }
   
  return nNFs;
}


TList* TQNFChainloader::getListOfFolders(){
  TQSampleFolder* sf = this->fReader->getSampleFolder();
  TList* l = new TList();
  for(auto itr=this->targetNFpaths.begin(); itr!=this->targetNFpaths.end(); ++itr){
    TQFolder* f = sf->getFolder(itr->first);
    if(f){
      l->Add(f);
    }
  }
  return l;
}

int TQNFChainloader::setVariationTags(const TQTaggable& tags) {
  this->fVariationTags.clear();
  return this->fVariationTags.importTags(tags);
}

int TQNFChainloader::setVariationTags(TQTaggable* tags) {
  this->fVariationTags.clear();
  return this->fVariationTags.importTags(tags);
}

