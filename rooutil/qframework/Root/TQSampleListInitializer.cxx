#include "QFramework/TQSampleListInitializer.h"
#include "QFramework/TQUtils.h"
#include "QFramework/TQLibrary.h"
#include "QFramework/TQIterator.h"
#include "TFile.h"
#include "TTree.h"

#include <stdexcept>

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQSampleListInitializer:
//
// A simple sample initializer that uses a list of files to initialize
// all samples matching these files.
// 
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQSampleListInitializer)

TQSampleListInitializer::TQSampleListInitializer(TQSampleFolder* sf) :
fSampleFolder(sf)
{
  // default constructor
}

TQSampleListInitializer::~TQSampleListInitializer(){
  // default destructor
}

#ifdef ASG_RELEASE
#define XAOD_STANDALONE 1
#include "xAODRootAccess/TEvent.h"
#include "xAODRootAccess/tools/TReturnCode.h"
#include "xAODEventInfo/EventInfo.h"
#include "xAODCutFlow/CutBookkeeper.h"
#include "xAODCutFlow/CutBookkeeperContainer.h"
#include "xAODCutFlow/CutBookkeeperAuxContainer.h"
namespace TQUtils {
xAOD::TEvent xAODMakeEvent(TFile* file);
xAOD::CutBookkeeper::Payload xAODMetaDataGetCutBookkeeper(xAOD::TEvent& event);
}
#endif

bool TQSampleListInitializer::initializeSampleForFile(const TString& filepath_orig){
	// initialize the sample for the given file
  TString filepath = TQStringUtils::trim(filepath_orig,"\n \t");
  TString fpath(filepath);
  TString fname = TQFolder::getPathTail(fpath);

#ifndef ASG_RELEASE
  ERRORclass("this class only supports ASG release setups!");
#else
  TFile* f = TFile::Open(filepath,"READ");
  if(!f){
    ERRORclass("no such file: '%s'",filepath.Data());
    return false;
  }
  if(!f->IsOpen()){
    ERRORclass("unable to open file: '%s'",filepath.Data());
    delete f;
    return false;
  }
  TTree* t = dynamic_cast<TTree*>(f->Get("CollectionTree"));
  if(!t){
    ERRORclass("unable to read CollectionTree in '%s'",filepath.Data());
    f->Close();
    delete f;
    return false;
  }
  try {
    xAOD::TEvent event(TQUtils::xAODMakeEvent(f));
    event.getEntry(0);
    const xAOD::EventInfo* evtinfo = NULL;
    if(event.retrieve(evtinfo, "EventInfo").isFailure()){
      throw std::runtime_error(TString::Format("unable to retrieve event info for file '%s'",filepath.Data()).Data());
    }
    int mcChannelNumber = evtinfo->mcChannelNumber();
    TCollection* c = this->fSampleFolder->getListOfSamples(TString::Format("*/%d",mcChannelNumber));
    if(!c || c->GetEntries() < 1){
      if(c) delete c;
      throw(std::runtime_error(TString::Format("unable to initialize sample '%d' - not found in tree!",mcChannelNumber).Data()));
    }
    TQSampleIterator itr(c,true);
    bool ok = true;
    while(itr.hasNext()){
      TQSample* parentsample = itr.readNext();
      TQSample* s = parentsample->getSample(TQFolder::makeValidIdentifier(fname)+"+");
      if(s){
        TString message;
        if(this->initializeSample(s,filepath,message)){
          INFOclass("initialized sample '%s': %s (%d/%d)",s->getPath().Data(),message.Data(),itr.getLastIndex()+1,c->GetEntries());
        } else {
          ERRORclass("failed to initialize sample '%s': %s",s->GetName(),message.Data());
          ok = false;
        }
      } else {
        throw std::runtime_error(TString::Format("unable to create sample '%s'",fname.Data()).Data());
      }
    }
    f->Close();
    delete f;
    return ok;
  } catch (const std::exception& e){
    ERRORclass(e.what());
    f->Close();
    delete f;
    return false;
  }
#endif
  return false;
}

int TQSampleListInitializer::initializeSamplesForFiles(TCollection* c){
	// initialize all samples in the given list
  int ok = 0;
  TQIterator itr(c);
  c->Print();
  while(itr.hasNext()){
    TObject* obj = itr.readNext();
    ok += this->initializeSampleForFile(obj->GetName());
  }
  return ok;
}

int TQSampleListInitializer::initializeSamples(){
	// initialze all samples
  return this->initializeSamplesForFiles(this->fPaths->getObjectPaths("*","*.root",TObjString::Class()));
}

