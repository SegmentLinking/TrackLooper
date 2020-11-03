#include "QFramework/TQSamplePurger.h"
#include "QFramework/TQSampleFolder.h"
#include "QFramework/TQFolder.h"
#include "QFramework/TQSample.h"
#include "QFramework/TQIterator.h"

// #define _DEBUG_
#include "QFramework/TQLibrary.h"

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQSamplePurger:
//
// Purge a sample folder by removing all samples that do not point to a valid tree.
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQSamplePurger)

void TQSamplePurger::setPurgeFolders(bool purge){
  // control whether entire folders will be purged (as opposed to only their contents)
  this->fPurgeFolders = purge;
}

void TQSamplePurger::setShallow(bool shallow){
  // control whether the purger will shallowly believe the tags on the samples
  // as opposed to checking if the sample is actually (still?) available
  this->fShallow = shallow;
}

int TQSamplePurger::revisitFolder(TQSampleFolder * sampleFolder, TString& /*message*/){
  // revisit a folder, purging the contents
  TCollection* l = sampleFolder->GetListOfFolders();
  TQIterator itr(l);
  bool purge = true;
  while(itr.hasNext()){
    TObject* obj = itr.readNext();
    TQFolder* f = dynamic_cast<TQFolder*>(obj);
    TQSample* s = dynamic_cast<TQSample*>(f);
    if(s){
      if(s->getTagBoolDefault("purge",false)){
        s->detachFromBase();
        delete s;
      } else {
        purge = false;
      }
    } else if(f){
      if(fPurgeFolders && f->getTagBoolDefault("purge",false)){
        f->detachFromBase();
        delete f;
      } else {
        purge = false;
      }
    } else {
      purge = false;
    }
  }
  sampleFolder->setTagBool("purge",purge);
  return visitOK;
}
      
int TQSamplePurger::visitSample(TQSample * sample, TString& /*message*/){
  // visit a sample to determine whether it can be purged
  if(this->fShallow){
    bool purge = true;
    int nEvents = 0;
    if(sample->getTagInteger(".init.nEvents",nEvents)) {
      if (nEvents>0) purge = false; 
    } else if (sample->getTagBoolDefault("isInitialized",false)) purge = false;
    else if (sample->hasTagString(".init.filepath") || sample->hasTagString(".xsp.filepath")) purge = false;
    
    //mark samples as to-be-purged if none of conditions above indicate it should not be purged
    if (purge) sample->setTagBool("purge",true);
  } else {
    TQToken* tok = sample->getTreeToken();
    TTree* t = (TTree*)( tok ? tok->getContent() : NULL );
    int nEntries = ( t ? t->GetEntries() : 0);
    if(nEntries < 1){
      sample->setTagBool("purge",true);
    }
    sample->returnToken(tok);
  }
  return visitOK;
}
 
TQSamplePurger::TQSamplePurger(const TString& name) : TQSampleVisitor(name){
}
TQSamplePurger::~TQSamplePurger(){
}

