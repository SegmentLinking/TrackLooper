#include "QFramework/TQNFUncertaintyScaler.h"
#include "QFramework/TQStringUtils.h"
#include "QFramework/TQIterator.h"
#include "QFramework/TQUtils.h"
#include "QFramework/TQFolder.h"
#include "TMath.h"
//#include "TRandom.h"
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
// TQNFUncertaintyScaler
//
// work in progress
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQNFUncertaintyScaler)


TQNFUncertaintyScaler::TQNFUncertaintyScaler(TQSampleFolder* f):
TQNFBase("TQNFUncertScaler"),
  status(-999)
{
  // default constructor taking base sample folder
  this->setSampleFolder(f);
}

TQNFUncertaintyScaler::TQNFUncertaintyScaler(TQSampleDataReader* rd):
  TQNFBase("TQNFUncertScaler"),
  status(-999)
{
  // default constructor taking base sample folder
  this->setReader(rd);
}

TQNFUncertaintyScaler::~TQNFUncertaintyScaler(){
  // default destructor, cleaning up all pointer fields
  this->finalize();
  this->clear();
}

int TQNFUncertaintyScaler::deployNF(const TString& name, const TString& cutName, TQFolder* config) {
  std::vector<TString> vec;
  vec.push_back(cutName);
  return this->deployNF(name, vec, std::vector<TString>(), config);
}


int TQNFUncertaintyScaler::deployNF(const TString& name, const std::vector<TString>& startAtCutNames, const std::vector<TString>& stopAtCutNames, TQFolder* config){
  // deploys a selected NF in all known locations
  // stop if this cut is an point of tree of cuts on which to apply the NFs
  if (!config) {
    ERRORclass("Invalid pointer to configuration object!");
    return -1;
  }
  
  if (config->hasTag("relvalue") && config->hasTag("absvalue")) {
    WARNclass("Config folder %s has both value tags (\"relvalue\" and \"absvalue\"). This is ambiguous, defaulting to use \"relvalue\"!",config->GetName());
  } else if (!config->hasTag("relvalue") && !config->hasTag("absvalue")) {
    WARNclass("Config folder %s has no value tag (neither \"relvalue\" nor \"absvalue\").",config->GetName());
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
  
  TString path = config->getTagStringDefault("path","");
  if (path.IsNull()) {
    WARNclassargs(TQStringUtils::concatenate(3,name.Data(),TQListUtils::makeCSV(startAtCutNames).Data(),(config->getTagBoolDefault("overwrite",true) ? "true":"false")),"cannot modify NF uncertainty, no path given!",this->status);
    return -1;
  }
  DEBUGclass("modifying NF unertainty for %s at {%s} ",name.Data(),TQListUtils::makeCSV(startAtCutNames).Data());
  
  int retval = 0;
  
  //this is duplicate code, "name" and "path" should contain the same data... 
  if (name.IsNull()) {
    ERRORclass("No target path given for config %s",config->getName().Data());
    return -1;
  }
  // set the NF according to the desired scale scheme(s)
  std::vector<TString> writeScaleSchemes = config->getTagVString("writeScaleScheme");
  std::vector<TString> tmp = config->getTagVString("scaleScheme");
  writeScaleSchemes.insert(writeScaleSchemes.end(),tmp.begin(),tmp.end());
  if(writeScaleSchemes.size() < 1){
    writeScaleSchemes.push_back(config->getTagStringDefault("writeScaleScheme",".default"));
  }
  std::vector<TString> targets = this->getTargetCuts(startAtCutNames,stopAtCutNames);
  for (size_t c=0; c<targets.size(); ++c) {
    TString cutName = targets.at(c);
    TQSampleFolderIterator sItr(this->fReader->getListOfSampleFolders(name),true);
    while(sItr.hasNext()){
      TQSampleFolder* s = sItr.readNext();
      if(!s) continue;
      
      for(size_t k=0; k<writeScaleSchemes.size(); k++){
        TQCounter* counter = s->getScaleFactorCounterInternal(writeScaleSchemes[k]+":"+cutName); //this returns the internal pointer, so handle with care!
        if (!counter){
          ERRORclass("unable to modify scale factor uncertainty for cut '%s' on path '%s' with scheme '%s'",cutName.Data(),s->getPath().Data(),writeScaleSchemes[k].Data());
        } 
        double modifier;
        if (config->getTagDouble("relvalue",modifier)) {
          counter->setError(counter->getError()+counter->getCounter()*modifier);
        } else if (config->getTagDouble("absvalue",modifier)){
          counter->setError(counter->getError()+modifier);
        }
        retval++;
      } 
    }
  }
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

int TQNFUncertaintyScaler::deployResult(const std::vector<TString>& startAtCutNames, const std::vector<TString>& stopAtCutNames, int overwrite){
  //The arguments are ignored, they are just there to comply with TQNFBase requirements. The individual
  //start/stop cuts and overwrite are read from the sub configs.
  if(!this->success()){
    WARNclassargs(TQStringUtils::concatenate(3,TQListUtils::makeCSV(startAtCutNames).Data(),TQStringUtils::concat(stopAtCutNames).Data(),TQStringUtils::getStringFromBool(overwrite).Data()),"cannot deploy NFs, no results available, status is %d!",this->status);
    return -1;
  }
  // deploys all NFs in all known locations
  int retval = 0;
  for(int i=0; i<configs->GetEntries(); i++){
    TQFolder* config = dynamic_cast<TQFolder*> (configs->At(i));
    retval += this->deployNF(config->getTagStringDefault("path",""), config->getTagVString("applyToCut"), config->getTagVString("stopAtCut"), config);
  }
  return retval;
}

void TQNFUncertaintyScaler::clear(){
  // clear all names, paths and results
  this->initialized = false;
  this->status = -999;
} 


bool TQNFUncertaintyScaler::finalizeSelf(){
  return true;
}

bool TQNFUncertaintyScaler::initializeSelf(){
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

int TQNFUncertaintyScaler::getStatus(){
  // retrieve the status
  return this->status;
}

TString TQNFUncertaintyScaler::getStatusMessage(){
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

void TQNFUncertaintyScaler::printStatus(){
  // print the status
  INFOclass("current status of instance '%s' is '%d': %s",this->GetName(),(int)(this->status),this->getStatusMessage().Data());
}

int TQNFUncertaintyScaler::execute(int itrNumber) {
  //start calculation
  this->iterationNumber = itrNumber;
  return 0; //unless sth. went wrong, then possibly return sth. else
}

bool TQNFUncertaintyScaler::success(){
  // return true if calculation was successful
  // false otherwise
  if(this->initialized && (this->status == 0))
    return true;
  return false;
}

bool TQNFUncertaintyScaler::readConfiguration(TQFolder* f){ 
  // Sets the configuration folder. The folder is expected to contain subfolders
  // having the following tags applied (* = required)
  // ...to be written
  if(!f) return false;
  this->configs = f->getListOfFolders("?");
  if(!configs || configs->GetEntries()<1) {
    ERRORclass("Invalid configuration for TQNFUncertaintyScaler");
    return false;
  }
  this->initialized = true;
  this->status = 0;
  
  return true;
}

