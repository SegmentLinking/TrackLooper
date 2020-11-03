#include "QFramework/TQNFManualSetter.h"
#include "QFramework/TQStringUtils.h"
#include "QFramework/TQIterator.h"
#include "QFramework/TQUtils.h"
#include "QFramework/TQFolder.h"
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

//#define _DEBUG_

#include "QFramework/TQLibrary.h"

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQNFManualSetter
//
// This class allows to manually set a normalization scale factor (NF).
// While, in principle, this class can be used on its own, it is highly recommended
// to use the TQNFChainloader instead. An example for a configuration to run the 
// TQNFManualSetter via the TQNFChainloader is provided in the HWWlvlv2015 package
// under HWWlvlv2015/share/normalization/example.cfg
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQNFManualSetter)


TQNFManualSetter::TQNFManualSetter(TQSampleFolder* f):
TQNFBase("TQNFManual"),
  status(-999)
{
  // default constructor taking base sample folder
  this->setSampleFolder(f);
}

TQNFManualSetter::TQNFManualSetter(TQSampleDataReader* rd):
  TQNFBase("TQNFManual"),
  status(-999)
{
  // default constructor taking base sample folder
  this->setReader(rd);
}

TQNFManualSetter::~TQNFManualSetter(){
  // default destructor, cleaning up all pointer fields
  this->finalize();
  this->clear();
}

int TQNFManualSetter::deployNF(const TString& name, const TString& cutName, TQFolder* config) {
  std::vector<TString> vecStart;
  vecStart.push_back(cutName);
  return this->deployNF(name, vecStart, std::vector<TString>(), config);
}

int TQNFManualSetter::deployNF(const TString& name, const std::vector<TString>& startAtCutNames, const std::vector<TString>& stopAtCutNames, TQFolder* config){
  // deploys a selected NF in all known locations
  // stop if this cut is an point of tree of cuts on which to apply the NFs
  if (!config) {
    ERRORclass("Invalid pointer to configuration object!");
    return -1;
  }

  if(!this->success()){
    //WARNclassargs(TQStringUtils::concatenate(3,name.Data(),cutName.Data(),TQStringUtils::getStringFromBool(config->getTagBoolDefault("overwrite",true)).Data()),"cannot deploy NF, no results available, status is %d!",this->status);
    WARNclassargs(TQStringUtils::concatenate(3,name.Data(),TQListUtils::makeCSV(startAtCutNames).Data(),(config->getTagBoolDefault("overwrite",true) ? "true":"false")),"cannot deploy NF, no results available, status is %d!",this->status);
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
  //@tag:[overwrite] This (sub-folder) argument tag determines if the result of the calculator replaces existing values or if it is multiplied with the previously existing value. Default: true
  bool overwrite = config->getTagBoolDefault("overwrite",true);
  DEBUGclass("deploying NF for %s (overwrite=%d)",name.Data(),overwrite?1:0);
  // get the NF
  //@tag:[value,uncertainty] These (sub-folder) argument tags determine the value and the uncertainty of the NF being set (or modified, see "overwrite")
  double nf = config->getTagDoubleDefault("value",1.);
  double sigma = config->getTagDoubleDefault("uncertainty",0.);
  if (! (iterationNumber < 0)) {
    double variation = this->chainLoader->getRelVariation(TQStringUtils::concatenate(3,"NFManualSetter::",config->getPath().Data(),config->getName().Data()),nf,sigma); //we are using a random variation to propagate the uncertainty of manually set NFs assuming no correlation to anything else. 
    nf *=variation;
  }
  int retval = 0;
  
  TString readScheme;
  bool hasReadScheme = config->getTagString("readScaleScheme",readScheme);

    if (name.IsNull()) {
      ERRORclass("No target path given for config %s",config->getName().Data());
      return -1;
    }
    // set the NF according to the desired scale scheme(s)
    //@tag:[writeScaleScheme] This (sub-folder) argument tag determines the list of scale schemes the results of the NF calculation are written to. Default: ".default"
    std::vector<TString> writeScaleSchemes = config->getTagVString("writeScaleScheme");
    if(writeScaleSchemes.size() < 1){
      writeScaleSchemes.push_back(config->getTagStringDefault("writeScaleScheme",".default"));
    }
    std::vector<TString> targets = this->getTargetCuts(startAtCutNames,stopAtCutNames);
    //---------------------------------------------
    for (size_t c=0; c<targets.size(); ++c) {
      TString cutName = targets.at(c);
      TQSampleFolderIterator sItr(this->fReader->getListOfSampleFolders(name),true);
      while(sItr.hasNext()){
        TQSampleFolder* s = sItr.readNext();
        if(!s) continue;
        double readFactor = 1.;
        double readSigma = 0.;
        if (hasReadScheme) {
          if (!s->getScaleFactor(readScheme+":"+cutName,readFactor,readSigma,false)) {
            WARNclass("Failed to read scheme '%s' at cut '%s'",readScheme.Data(),cutName.Data());
            readFactor = 1.;
            readSigma = 0.;
          }
        }
        for(size_t k=0; k<writeScaleSchemes.size(); k++){
          int n = s->setScaleFactor(writeScaleSchemes[k]+":"+cutName+(overwrite>0?"":"<<"), nf*readFactor,sqrt(pow(sigma*readFactor,2) + pow(readSigma*nf,2)) );
          #ifdef _DEBUG_
          if (s->getFolder(".scalefactors")) s->getFolder(".scalefactors")->print();
          #endif
          if(n == 0){
            ERRORclass("unable to set scale factor for cut '%s' on path '%s' with scheme '%s'",cutName.Data(),s->getPath().Data(),writeScaleSchemes[k].Data());
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
        TList* sflist = this->fReader->getListOfSampleFolders(name);
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
    }
    //---------------------------
  /*
  // if no recursion was required, we can stop here
  if(stopAtCutNames.size() == 0 || !this->cutInfoFolder || last)
    return retval;
  // if stopAtCutNames is set, we need to recurse over the cut structure
  // therefore, we first need to find out how the cuts are structured
  */
  //TList * cuts = this->cutInfoFolder->getListOfFolders(TString::Format("*/%s/?",cutName.Data()));
  /*if(!cuts) return retval;
  TQIterator itr(cuts,true);
  // iterate over all the cuts below the one we are investigating now
  while(itr.hasNext()){
    TQFolder* f = dynamic_cast<TQFolder*>(itr.readNext());
    if(!f) continue;
    // and deploy NFs at the corresponding cuts
    retval += this->deployNF(name,f->GetName(),stopAtCutNames,config);
  }
  */
  return retval;
}

int TQNFManualSetter::deployResult(const std::vector<TString>& startAtCutNames,  const std::vector<TString>& stopAtCutNames, int overwrite){
  // The arguments are ignored, they are just there to comply with TQNFBase requirements. The individual
  // start/stop cuts and overwrite are read from the sub configs.
  if(!this->success()){
    WARNclassargs(TQStringUtils::concatenate(3,TQListUtils::makeCSV(startAtCutNames).Data(),TQStringUtils::concat(stopAtCutNames).Data(),TQStringUtils::getStringFromBool(overwrite).Data()),"cannot deploy NFs, no results available, status is %d!",this->status);
    return -1;
  }
  // deploys all NFs in all known locations
  int retval = 0;
  for(int i=0; i<configs->GetEntries(); i++){
    TQFolder* config = dynamic_cast<TQFolder*> (configs->At(i));
    //@tag:[path,applyToCut,stopAtCut] These sub-folder argument tags determine at which paths and cuts the NF is deployed. "stopAtCut" and applyToCut support list context.
    retval += this->deployNF(config->getTagStringDefault("path",""), config->getTagVString("applyToCut"), config->getTagVString("stopAtCut"), config);
  }
  return retval;
}

void TQNFManualSetter::clear(){
  // clear all names, paths and results
  this->initialized = false;
  this->status = -999;
} 


bool TQNFManualSetter::finalizeSelf(){
  return true;
}

bool TQNFManualSetter::initializeSelf(){
  // initialize the NF Calculator
  // - initializes the data histogram
  // - initializeds the mc histograms
  // will set the initialize flag to true
  // further calls of initialize will have no effect
  // until clear is called
  if(!this->fReader){
    return false;
  }
    
  this->status = 0;
  return true;
}

int TQNFManualSetter::getStatus(){
  // retrieve the status
  return this->status;
}

TString TQNFManualSetter::getStatusMessage(){
  // decode the integer status code
  switch(this->status){
  case -999:
    return "uninitialized";
  case -10:
    return "error during initialization";
  case 0:
    return "all OK";
  default:
    return "unkown error";
  }
}

void TQNFManualSetter::printStatus(){
  // print the status
  INFOclass("current status of instance '%s' is '%d': %s",this->GetName(),(int)(this->status),this->getStatusMessage().Data());
}

int TQNFManualSetter::execute(int itrNumber) {
  //start calculation
  this->iterationNumber = itrNumber;
  return 0; //unless sth. went wrong, then possibly return sth. else
}

bool TQNFManualSetter::success(){
  // return true if calculation was successful
  // false otherwise
  if(this->initialized && (this->status == 0))
    return true;
  return false;
}

bool TQNFManualSetter::readConfiguration(TQFolder* f){ 
  // Sets the configuration folder. The folder is expected to contain subfolders
  // having the following tags applied (* = required)
  // ...to be written
  if(!f) return false;
  this->configs = f->getListOfFolders("?");
  if(!configs || configs->GetEntries()<1) {
    ERRORclass("Invalid configuration for TQNFManualSetter");
    return false;
  }
  this->initialized = true;
  this->status = 0;
  
  return true;
}

