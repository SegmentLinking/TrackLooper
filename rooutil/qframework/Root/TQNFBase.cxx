#include "QFramework/TQNFBase.h"
#include "QFramework/TQNFChainloader.h"
#include "TROOT.h"
#include "QFramework/TQNamedTaggable.h"
#include "TMath.h"
#include "QFramework/TQUtils.h"
#include "QFramework/TQIterator.h"
#include "QFramework/TQLibrary.h"


#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cmath>

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQNFBase:
//
// The abstract TQNFBase class provides a common interface for custom NF calculators
// like TQNFCalculator and TQABCDCalculator. All future NF calculators should inherit
// from this class and their usage should be implemented in the TQNFChainloader.
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQNFBase)

TQNFBase::TQNFBase() : TQNamedTaggable(),
  fReader(NULL),
  fOwnerOfReader(false),
  verbosity(1),
  infoFolder(NULL),
  cutInfoFolder(NULL),
  initialized(false),
  iterationNumber(-1)
{

}

TQNFBase::TQNFBase(const TString& name) : TQNamedTaggable(name),
                                          fReader(NULL),
                                          fOwnerOfReader(false),
                                          verbosity(1),
                                          infoFolder(NULL),
                                          cutInfoFolder(NULL),
                                          initialized(false),
                                          iterationNumber(-1)
{
  //dummy constructor
}

void TQNFBase::addNFPath(const TString& path, const TString& cutname, const TString& scalescheme){
  if(this->iterationNumber != 0) return;
  this->vNFpaths.push_back(path+"/"+scalescheme+":"+cutname); 
}

const std::vector<TString>& TQNFBase::getNFpaths() {
  // returns a reference to the current list of NF paths 
  // the actual implementation of this class must keep std::vector<TString> vNFpaths 
  // updated to contain the full paths where NFs have been deployed.
  return vNFpaths;
}

void TQNFBase::printNFPaths(){
  std::cout << TQStringUtils::makeBoldWhite(this->GetName()) << std::endl;
  for(size_t i=0; i<vNFpaths.size(); ++i){
    std::cout << vNFpaths[i] << std::endl;
  }
}

void TQNFBase::setChainLoader(TQNFChainloader* loader) {
  this->chainLoader = loader; 
}

TString TQNFBase::getPathTag(const TString& tagname){
  TString path;
  if(!this->getTagString(tagname,path)) return "";
  if(!path.BeginsWith("/")){
    return TQFolder::concatPaths(this->getTagStringDefault(tagname+".prefix", TQLibrary::getWorkingDirectory()),path);
  }
  return path;
}

bool TQNFBase::initialize(){
  if(this->initialized) return true;
  this->initialized = this->initializeSelf();
  TString logfilename = this->getPathTag("saveLog");
  if(!logfilename.IsNull()){
    this->setOutputStream(logfilename,this->verbosity);
  }
  return this->initialized;
}


bool TQNFBase::finalize(){
  this->initialized=false;
  if(this->finalizeSelf()){
    if(this->hasTag("saveLog")) this->closeOutputStream();
    return true;
  }
  if(this->hasTag("saveLog")) this->closeOutputStream();
  return false;
}

int TQNFBase::getStatus(){
  return 0;
}

TString TQNFBase::getStatusMessage(){
  return "<status unknown>";
}

int TQNFBase::deployResult(const std::vector<TString>& startAtCutNames, const std::vector<TString>& stopAtCuts ){
  // set the NF according to the desired scale scheme
  // if the tag 'targetPath' is set on this instance, it will be used to deploy the NF
  // if this tag is not set, the value of sample path used in the calculation will be used.
  //@tag:[overwrite] This object tag determines if the NFCalculator's result should overwrite an existing value (1) or if it should be set to the product of the existing and the new value(0). Default: 1 .
  return this->deployResult(startAtCutNames,stopAtCuts,this->getTagIntegerDefault("overwrite",1));
}

void TQNFBase::setInfoFolder(TQFolder* f){
  // set the pointer to the info folder
  // where information about NFs and their respective processes
  // will be stored by the TQNFCalculator
  this->infoFolder = f;
}

void TQNFBase::setCutInfoFolder(TQFolder* f){
  // set the pointer to the cut info folder
  // where information about the cut hierarchy is stored
  // and can be read by the TQNFCalculator
  // the content of the cut info folder will not be modified
  this->cutInfoFolder = f;
}

TQNFBase::~TQNFBase(){
  if(this->fReader && this->fOwnerOfReader) delete fReader;
}

void TQNFBase::setSampleFolder(TQSampleFolder* sf){
  // set/change the base sample folder
  if(this->fReader && this->fOwnerOfReader){
    delete this->fReader;
  }
  if(sf){
    this->fReader = new TQSampleDataReader(sf);
    this->fOwnerOfReader = true;
  } else {
    this->fReader = NULL;
  } 
}

void TQNFBase::setReader(TQSampleDataReader* rd){
  // set/change the sample data reader
  if(this->fOwnerOfReader){
    delete this->fReader;
  }
  this->fReader = rd;
  this->fOwnerOfReader = false;
}

TQSampleFolder* TQNFBase::getSampleFolder(){
  // retrieve the base sample folder
  if(!this->fReader) return NULL;
  return this->fReader->getSampleFolder();
}

TQSampleDataReader* TQNFBase::getReader(){
  // retrieve the sample data reader
  return this->fReader;
}

void TQNFBase::setVerbosity(int verbosity){
  // set the verbosity level
  // 0 : silent
  // 1 : some output
  // 2 : all output
  this->verbosity = verbosity;
}

void TQNFBase::setOutputStream(const TString& outfile, int verbosity){
  // this function allows to redirect the output of the TFractionFitter
  // to an arbitrary file or device on your disk
  // including /dev/null in order to dismiss the output irreversibly
  bool ok = this->messages.open(outfile);
  if(ok) this->messages.sendMessage(TQMessageStream::INFO,"%s '%s' opening log file",this->Class()->GetName(),this->GetName());
  this->messages.sendMessage(TQMessageStream::INFO,TQUtils::getTimeStamp());
  this->messages.newline();
  this->verbosity = verbosity;
}

void TQNFBase::closeOutputStream(){
  // close the output stream assigned to this instance
  this->messages.close();
}

std::vector<TString> TQNFBase::getTargetCuts(const std::vector<TString>& startCuts, const std::vector<TString>& stopCuts) {
  // returns a list of all individual cuts which are in startCuts or cuts derived
  // from these, but not derived from any of stopCuts. The entries in stopCuts are
  // inclusive, i.e., these cuts are still added to the returned vector given they
  // are derived from one of the start cuts.
  std::vector<TString> retVec;
  if (!this->cutInfoFolder) {
    WARNclass("Missing cut info folder, returning an empty vector of target cuts!");
    return retVec;
  }

  TList* stopFolders = NULL;
  TList* startFolders = NULL;
  for(size_t i=0; i<startCuts.size(); ++i) {
    if (!startFolders) startFolders = new TList();
    startFolders->AddAll(this->cutInfoFolder->getListOfFolders(TString::Format("*/%s",startCuts.at(i).Data())));
  }
  for(size_t i=0; i<stopCuts.size(); ++i) {
    if (!stopFolders) stopFolders = new TList();
    stopFolders->AddAll(this->cutInfoFolder->getListOfFolders(TString::Format("*/%s",stopCuts.at(i).Data())));
  }
  
  TList* lTargets = new TList();
  TQFolderIterator itr(startFolders);
  while(itr.hasNext()) {//call worker for each starting cut
    TQFolder* f = itr.readNext();
    if (!f) continue;
    this->getTargetCutsWorker(lTargets, f, stopFolders);
  }

  //create a std::map<TQFolder*,bool> to remove duplicates
  std::map<TQFolder*,bool> map;  
  TQFolderIterator targetItr(lTargets);
  while(targetItr.hasNext()) {
    TQFolder* f = targetItr.readNext();
    if (!f) continue;
    map[f] = true;
  }
  delete lTargets;
  
  for(std::map<TQFolder*,bool>::iterator iter = map.begin(); iter != map.end(); ++iter) {
    if (!iter->first) continue;
    retVec.push_back((iter->first)->GetName());
  }
  
  return retVec;  
  
}

void TQNFBase::getTargetCutsWorker(TList* targets, TQFolder* startCut, TList* stopCuts) {
  //if stopCuts is a null pointer no recursion will be performed, instead startCut is simply added to targets.
  if (!startCut) return;

  if (!targets) targets = new TList();
  targets->Add(startCut);
  if (!stopCuts) return; //if there is no recursion required we stop here

  //If the current start cut is one of the stop cuts, we won't iterate any further
  TQFolderIterator stopItr(stopCuts);
  while(stopItr.hasNext()) {
    if (stopItr.readNext() == startCut) {
      return;
    } 
  }
  //otherwise we iterate over subfolders:
  TList * subFolder = startCut->getListOfFolders();
  TQFolderIterator subItr(subFolder);
  while(subItr.hasNext()) {
    TQFolder* f = subItr.readNext();
    this->getTargetCutsWorker(targets, f, stopCuts);
  }
  return;
}












