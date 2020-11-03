#include "QFramework/TQStringUtils.h"
#include "QFramework/TQSample.h"
#include "QFramework/TQUtils.h"
#include "QFramework/TQIterator.h"
#include "TFile.h"

#include "definitions.h"

// #define _DEBUG_
#include "QFramework/TQLibrary.h"

#ifdef HAS_XAOD
#define XAOD_STANDALONE 1
#include "TChain.h"
#include "xAODRootAccessInterfaces/TActiveEvent.h"
#include "xAODRootAccess/TEvent.h"
#include "xAODRootAccessInterfaces/TVirtualEvent.h"
#include "xAODRootAccess/MakeTransientTree.h"
bool TQSample::gUseTransientTree(true);
#else 
bool TQSample::gUseTransientTree(false);
#endif
bool TQSample::gUseAthenaAccessMode(false);

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQSample:
//
// The TQSample class is a representation of a sample. It is a subclass of a TQSampleFolder and
// can therefore additionally serve as container for objects (e.g. analysis results) associated
// with it. An instance of this class can point to a TTree object which will be used as the
// source of event data. This tree may either be contained in this sample itself, or it may be
// contained in an external ROOT file. Either way, you tell the sample where to find the tree
// by passing an argument to setTreeLocation(...).
//
// As the tree will probably be used by several classes, accessing the tree is managed by the
// sample. To get a pointer to the tree, use getTreeToken(...)
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQSample)


//__________________________________________________________________________________|___________

TList * TQSample::splitTreeLocations(TString treeLocations) {

  /* the list of tree locations to return */
  TList * list = 0;

  while (!treeLocations.IsNull()) {
    /* read one tree location */
    TString location;
    TQStringUtils::readUpTo(treeLocations, location, "<");
    TQStringUtils::readBlanks(treeLocations);

    if (TQStringUtils::removeLeading(treeLocations, "<", 2) != 1) {
      /* create the list */
      if (!list) {
        list = new TList();
        list->SetOwner(true);
      }
      /* add the new item to the list */
      list->Add(new TObjString(TQStringUtils::trim(location).Data()));
    } else {
      /* found a single "<" character: stop */
      if (list)
        delete list;
      return 0;
    }
  }

  /* return the list */
  return list;
}


//__________________________________________________________________________________|___________

TString TQSample::extractFilename(const TString& treeLocation) {

  // find last occurence of ":" ...
  int pos = treeLocation.Last(':');
  if (pos != kNPOS) {
    // ... return string before it
    return treeLocation(0, pos);
  } else {
    // ... or an empty string in case no ':' has been found
    return TString("");
  }
}


//__________________________________________________________________________________|___________

TString TQSample::extractTreename(const TString& treeLocation) {

  // find last occurence of ":" ...
  int pos = treeLocation.Last(':');
  if (pos != kNPOS) {
    // ... return string after it
    return treeLocation(pos + 1, treeLocation.Length());
  } else {
    // ... or the full string in case no ':' has been found
    return treeLocation;
  }
}


//__________________________________________________________________________________|___________

TQSample::TQSample() : TQSampleFolder() {
  // Default constructor of TQSample class: a new instance of TQSample is created
  // and initialized. It will be emtpy and have no base folder and its name will
  // be set to "unkown"
}


//__________________________________________________________________________________|___________

TQSample::TQSample(const TString& name) : TQSampleFolder(name) {
  // Constructor of TQSample class: a new instance of TQSample is created and
  // initialized. It will be emtpy and have no base folder and its name will
  // be set to the value of the parameter "name_" if it is a valid name and
  // "unknown" otherwise
}

//__________________________________________________________________________________|___________

void TQSample::setNormalisation(double normalisation_) {
  // Set the normalisation factor of this sample. This factor will be used by any
  // analysis to scale its results

  fNormalisation = normalisation_;
}


//__________________________________________________________________________________|___________

double TQSample::getNormalisation() {
  // Get the normalisation factor of this sample (= 1. by default)
 
  return fNormalisation;
}

//__________________________________________________________________________________|___________

void TQSample::setTree(TFile* f, const TString& treename){
  // set the TTree of this sample from the file and the treename given
  DEBUGclass("checking status of file");
  if(!f || f->IsZombie() || !f->IsOpen()){
    INFO("Encountered possible zombie file in sample '%s'!",this->getPath().Data());
    this->fTree = NULL;
    return;
  }
  DEBUGclass("attempting to retrieve tree");
  TTree* tree = dynamic_cast<TTree*>(fFile->Get(treename.Data()));
  //don't return just yet. In case of xAODs we might have  file without a CollectionTree but we might still need the TEvent for handling some meta data 
  /*
  if(!tree){
    this->fTree = NULL;
    return;
  }
  */
  DEBUGclass("function called on file '%s' for tree '%s'",f->GetName(),treename.Data());
  // this function is used to encapsule TTree post-processing
#ifdef HAS_XAOD
#warning "using ASG_RELEASE compilation scheme"
  // if we are inside an ASG release, this tree might require special treatment
  // we check this by looking if any branch type includes the string "xAOD"
  
  bool isxAOD = false;
  TTree* testTree = tree;
  if (!testTree) { //we might not have a CollectionTree, so let's try to use the MetaData tree to identify the file as an xAOD.
    testTree = dynamic_cast<TTree*>(fFile->Get("MetaData"));
  }
  if (!testTree) return; //if there's also no MetaData tree, we're done.
  
  TQIterator itr(testTree->GetListOfBranches());
  while(itr.hasNext()){
    TBranch* obj = (TBranch*)(itr.readNext());
    if(TQStringUtils::find(obj->GetClassName(),"xAOD") != kNPOS){
      isxAOD = true;
      break;
    }
  }
  // if we found anything, we use the xAOD::MakeTransientTree method
  if(isxAOD){
    DEBUGclass("identified tree of sample '%s' as xAOD, making transient",this->getPath().Data());
    DEBUGclass("creating new xAOD::TEvent");
    xAOD::TVirtualEvent* evt = xAOD::TActiveEvent::event();
    if(evt){
      DEBUGclass("re-using existing instance of xAOD::TEvent");
      this->fEvent = dynamic_cast<xAOD::TEvent*>(evt);
      if(!this->fEvent) throw std::runtime_error("active instance of TVirtualEvent is not of type TEvent!");
    } else {
      DEBUGclass("creating new instance of xAOD::TEvent");
      this->fEvent = new xAOD::TEvent(gUseAthenaAccessMode? xAOD::TEvent::kAthenaAccess : xAOD::TEvent::kClassAccess);
      this->fEvent->setActive();
    }
    bool ok = (this->fEvent->readFrom( this->fFile, kTRUE, treename ).isSuccess());
    DEBUGclass("calling xAOD::MakeTransientTree on event %x with treename '%s'",this->fEvent,treename.Data());
    if(ok){
      if(tree && TQSample::gUseTransientTree){ //don't try to create a transTree if there isn't even a CollectionTree
        int oldErrorIgnoreLevel = gErrorIgnoreLevel;
        gErrorIgnoreLevel = 2000;
        this->fTree = xAOD::MakeTransientTree( *this->fEvent, treename );
        gErrorIgnoreLevel = oldErrorIgnoreLevel;
        DEBUGclass("retrieved transient tree %x",this->fTree);
        this->fTreeIsTransient = true;
      } else {
        this->fTree = tree;
        this->fTreeIsTransient = false;
      }
    } else {
      WARNclass("TEvent failed to read from input file!");
      //      delete this->fEvent;
      this->fEvent = 0;
    }
    return;
  }
#else
#warning "using plain ROOT compilation scheme"
#endif
  // if control reaches this point, we are either in plain ROOT or at least
  // have a plain ROOT compatible tree. there's not much to do in that case...
  DEBUGclass("identified tree of sample '%s' as plain ROOT, setting data member",this->getPath().Data());
  this->fTree = tree;
  fTreeIsTransient = false;
}

//__________________________________________________________________________________|___________

void TQSample::clearTree(){
  // clear the tree belonging to this sample
  // is called upon returning all tree tokens
  if(fFile){
    this->retractTreeFromFriends();
    if (fFile->IsOpen()){
      DEBUGclass("closing file");
      fFile->Close();
    }
    DEBUGclass("deleting file pointer");
    delete fFile;
  }
#ifdef HAS_XAOD
  fEvent = NULL;
#endif
  fFile = NULL;
  fTree = NULL;
}

//__________________________________________________________________________________|___________

bool TQSample::getTree(){
  // retrieves the tree from the file to store it in the sample
    DEBUGclass("trying to retrieve tree");
    /* get and split the tree locations */
    TList * treeLocations = splitTreeLocations(getTreeLocation());

    if (!treeLocations) return false;

    /* the default file- and treename */
    TString defFilename;
    TString defTreename;
 
    /* iterate over tree locations */
    int nTreeLocations = treeLocations->GetEntries();
    for (int i = 0; i < nTreeLocations && (fTree || i == 0); i++) {
 
      /* the tree location */
      TString location = treeLocations->At(i)->GetName();
      /* extract file- and treename */
      TString filename = extractFilename(location);
      TString treename = extractTreename(location);
 
      /* set default file- and treenames */
      if (filename.IsNull() && !defFilename.IsNull())
        filename = defFilename;
      else if (!filename.IsNull() && defFilename.IsNull())
        defFilename = filename;
      if (treename.IsNull() && !defTreename.IsNull())
        treename = defTreename;
      else if (!treename.IsNull() && defTreename.IsNull())
        defTreename = treename;
 
 
      if (!filename.IsNull()) {

        // returns NULL if file pointer is 0 or !file->isOpen()
        TFile* myOpenFile = TQUtils::openFile(filename);

        if (myOpenFile) {
          // therefore, we only get here
          // if myOpenFile is valid and open 
          if (i == 0) {
            fFile = myOpenFile;
            DEBUGclass("setting tree");
            this->setTree(fFile, treename);
            DEBUGclass("got tree %x",this->fTree);
            if (!fTree) {
              //FIXME: is this really what we want? We might need the file open even if no tree is present (reading meta data!!)
              /*
              fFile->Close();
              delete fFile;
              fFile = 0;
              */
              INFOclass("Failed to retrieve tree from file");
              //return false;
            }
          } else {
            if (fTree) fTree->AddFriend(treename.Data(), myOpenFile);
            else ERRORclass("Cannot add tree from file '%s' as a friend tree as no valid tree could be obtained from the first file '%s'",myOpenFile->GetPath(),fFile->GetPath());
          }
        } else { // not a valid and open file
          ERRORclass("Failed to open file while retrieving tree!");
          return false;
        }
      } else {
        /* look for the tree inside this folder */
        TTree* tree = dynamic_cast<TTree*>(getObject(treename.Data()));
        if (tree){
          /* we found the tree: keep the pointer */
          this->fTree = tree;
          this->fTreeIsTransient = false;
        } else {
          /* we couldn't find the tree */
          fTree = NULL;
          return false;
        }
      }
    }
 
    delete treeLocations;
    DEBUGclass("successfully completed function");
    return true;
}

//__________________________________________________________________________________|___________

void TQSample::promoteTreeToFriends(){
  // promote the tree pointer to some friends
  //TQSampleFolderIterator itr(this->fFriends);
  
  //while(itr.hasNext()){
  
  if (this->fFriends == nullptr) return; // when running in single channel mode we never look for friends
  
  for (auto myFriendSF : (*this->fFriends)) {
    if (! myFriendSF->InheritsFrom(TQSample::Class()) ) { throw std::runtime_error("Attempt to promote tree to friend which is not of type TQSample. A SampleFolder within a Sample should never happen, check your analysis setup!"); return; }
    TQSample* myFriend = static_cast<TQSample*>(myFriendSF);
  //  TQSample* myFriend = dynamic_cast<TQSample*>(itr.readNext());
    DEBUG("promoting tree from sample '%s' to sample '%s'",this->getPath().Data(),myFriend->getPath().Data());
    if(!myFriend->getFile()){
      myFriend->fTree = this->fTree;
#ifdef HAS_XAOD
      myFriend->fEvent = this->fEvent;
#endif
    }
  }
}

void TQSample::retractTreeFromFriends(){
  // retract a tree (and TEvent) pointer from friends
  //TQSampleFolderIterator itr(this->fFriends);
  //while(itr.hasNext()){
  //  TQSample* myFriend = dynamic_cast<TQSample*>(itr.readNext());
  //  if (!myFriend) throw std::runtime_error("Attempt to promote tree to friend which is not of type TQSample. A SampleFolder within a Sample should never happen, check your analysis setup!");
  
  if (this->fFriends == nullptr) return; // when running in single channel mode we never look for friends
  
  for (auto myFriendSF : (*this->fFriends)) {
    if (! myFriendSF->InheritsFrom(TQSample::Class()) ) { throw std::runtime_error("Attempt to retract tree from friend which is not of type TQSample. A SampleFolder within a Sample should never happen, check your analysis setup!"); return; }
    TQSample* myFriend = static_cast<TQSample*>(myFriendSF);
    if (myFriend==this) continue; //don't remove the pointer of this instance (yet)
    DEBUG("retracting tree of sample '%s' from sample '%s'",this->getPath().Data(),myFriend->getPath().Data());
    if(!myFriend->getFile()){
      myFriend->fTree = NULL;
#ifdef HAS_XAOD
      myFriend->fEvent = NULL;
#endif
    }
  }
  
}

//__________________________________________________________________________________|___________

TQToken * TQSample::getTreeToken() {
  // Request a tree token (TQToken). Return a tree token containing a pointer to
  // the TTree object if the tree is accessible, return a null pointer otherwise.
  // Use returnTreeToken(...) to return the tree token after you don't need the
  // tree anymore. The first request of a tree token triggers reading of the tree
  // (opening the ROOT file, in case it is contained in an external file). After
  // every tree token was returned, the tree is released (closing the ROOT file,
  // in case it is contained in an external file).
  //
  // Your code using the tree might look like this:
  //
  // /* try to get a tree token */
  // treeToken = sample->getTreeToken();
  //
  // if (treeToken) {
  // /* get the pointer to the tree */
  // TTree* tree = (TTree*)treeToken->getContent();
  // /* optional: the sample should know who is using the token */
  // treeToken->setOwner(this);
  //
  // /* <use your tree here> */
  //
  // /* return the tree token */
  // sample->returnTreeToken(treeToken);
  // } else {
  // /* for any reason we didn't get a tree
  // * token, so we cannot use the tree */
  // }
  DEBUGclass("entering function");
  
  bool createdFile = !(this->fFile); //check if we already had an open file in the beginning or if we have opened it during getTree
  if (!fTree) {
    if(this->getTree()){
      this->promoteTreeToFriends();
    }
  }

  if (fTree) {
    DEBUGclass("creating tree token");
    /* create a new tree token */
    TQToken * treeToken = new TQToken();
    treeToken->setContent(fTree);
    /* create the list of tree tokens */
    if (!fTokens) {
      fTokens = new TList();
      fTokens->SetOwner(true);
    }
    /* keep it in our list */
    fTokens->Add(treeToken);
    /* dispense the token */
    DEBUGclass("returning tree token");
    return treeToken;
  } 
  
  if (createdFile && this->fFile) { //close the input file if we just caused it to be opened and didn't get our desired object
    this->clearTree();
  }
  /* no tree and thus no tree token to dispense */
  return NULL;
}

//__________________________________________________________________________________|___________

TQToken * TQSample::getFileToken() {
  // Request a file token (TQToken). Return a file token containing a pointer to
  // the TFile object if it is accessible, return a null pointer otherwise.
  // Use returnToken(...) to return the token after you don't need the file anymore.
  DEBUGclass("entering function");
  
  bool createdFile = !(this->fFile); //check if we already had an open file in the beginning or if we have opened it during getTree
  if (!fFile) {
    if(this->getTree()){
      this->promoteTreeToFriends();
    }
  }
  if (fFile) {
    DEBUGclass("returning file");
    TQToken * fileToken = new TQToken();
    fileToken->setContent(fFile);
    if (!fTokens) {
      fTokens = new TList();
      fTokens->SetOwner(true);
    }
    fTokens->Add(fileToken);
    return fileToken;
  } 
  
  
  if (createdFile && this->fFile) { //close the input file if we just caused it to be opened and didn't get our desired object
    this->clearTree();
  }
  return NULL;
}


//__________________________________________________________________________________|___________

TQToken * TQSample::getEventToken() {
  // Request an event token (TQToken). Return a tree token containing a pointer to
  // the TEvent object if the tree is accessible, return a null pointer otherwise.
  // Use returnToken(...) to return the token after you don't need the
  // TEvent anymore. The first request of a tree token triggers reading of the tree
  // (opening the ROOT file, in case it is contained in an external file). After
  // every tree token was returned, the tree is released (closing the ROOT file,
  // in case it is contained in an external file).
#ifdef HAS_XAOD
  DEBUGclass("entering function");
  bool createdFile = !(this->fFile); //check if we already had an open file in the beginning or if we have opened it during getTree
  
  if (!fEvent) {
    if(this->getTree()){
      this->promoteTreeToFriends();
    }
  }

  if (fEvent) {
    DEBUGclass("returning TEvent");
    /* create a new token */
    TQToken * eventToken = new TQToken();
    eventToken->setContent(fEvent);
    /* create the list of event tokens */
    if (!fTokens) {
      fTokens = new TList();
      fTokens->SetOwner(true);
    }
    /* keep it in our list */
    fTokens->Add(eventToken);
    /* dispense the token */
    return eventToken;
  } 
  
  if (createdFile && this->fFile) { //close the input file if we just caused it to be opened and didn't get our desired object
    this->clearTree();
  }
  /* no tree and thus no event token to dispense */
  return NULL;
  #else
  ERRORclass("event token was requested, but TEvent class unavailable in standalone release!");
  return NULL;
  #endif
}

//__________________________________________________________________________________|___________

bool TQSample::returnTreeToken(TQToken * &token_) {
  // Return and delete a tree token that is not needed anymore (for
  // details check documentation of getTreeToken()).
  return this->returnToken(token_);
}

//__________________________________________________________________________________|___________

bool TQSample::returnToken(TQToken * &token_) {
  // Return and delete a tree or event token that is not needed
  // anymore
  if (fTokens && fTokens->FindObject(token_)) {

    /* remove and delete this token
     * and reset the pointer to it */
    fTokens->Remove(token_);
    token_ = 0;

    /* close file if all tokens have been returned */
    if (getNTreeTokens() == 0) {
      DEBUGclass("returning tree token - no tokens left, clearing tree");
      this->clearTree();
    } else {
      DEBUGclass("returning tree token - there are still other active tokens");
    }

    /* successfully returned tree token */
    return true;
  } else {
    /* we don't know this token, so ignore it */
    return false;
  }
}


//__________________________________________________________________________________|___________

bool TQSample::setTreeLocation(TString treeLocation_) {
  // Set the tree location. Setting a new tree location will fail, if there are
  // still unreturned tree tokens. In that case, false is returned, and true other-
  // wise.
  //
  // A tree may either be contained in this sample it self, or it may be contained
  // in an external ROOT file:
  //
  // - if the tree is contained in the sample use
  //
  // setTreeLocation("<name_of_the_tree>");
  //
  // - if the tree is contained in an external ROOT file, use:
  //
  // setTreeLocation("<name_of_the_file>:<name_of_the_tree>");
  //
  // Please note: the name of the tree is NOT the name of the pointer, but it is
  // the name of the object returned by tree->GetName() (or in case of an external
  // file, the name of the key).

  if (getNTreeTokens() == 0) {
    /* set the new tree location */
    fTreeLocation = treeLocation_;
    /* we successfully set the tree location */
    return true;
  } else {
    /* there are still tree tokens unreturned */
    return false;
  }

}


//__________________________________________________________________________________|___________

int TQSample::getNTreeTokens() {
  // Return the number of unreturned tree tokens

  if (fTokens)
    return fTokens->GetEntries();
  else
    return 0;
}


//__________________________________________________________________________________|___________

void TQSample::printTreeTokens() {
  // Print a summary of tree tokens

  TQIterator itr(fTokens);
  while (itr.hasNext()) {
    TQToken * token = dynamic_cast<TQToken*>(itr.readNext());
    if (token) {
      token->print();
    }
  }
}


//__________________________________________________________________________________|___________

bool TQSample::checkTreeAccessibility() {
  // Return true if the TTree is accessible and false otherwise

  bool isAccessible = false;

  /* try to access the tree */
  TQToken * token = getTreeToken();
 
  if (token) {
    isAccessible = true;
    returnTreeToken(token);
  }

  return isAccessible;
}


//__________________________________________________________________________________|___________

TString TQSample::getTreeLocation() {
  // Return the tree location
  if(fTreeLocation.IsNull()){
    TString filepath,treename;
    if(this->getTagString(".xsp.filepath",filepath) && this->getTagString(".xsp.treename",treename)){
      this->fTreeLocation = filepath + ":" + treename;
    }
  }
  return fTreeLocation;
}


//__________________________________________________________________________________|___________

TString TQSample::getFilename() {
  // retrieve the filename associated with this sample
  TList * treeLocations = splitTreeLocations(fTreeLocation);
  TString treeLocation;
  if (treeLocations) {
    if (treeLocations->GetEntries() > 0)
      treeLocation = TString(treeLocations->First()->GetName());
    delete treeLocations;
  }
 
  return extractFilename(treeLocation);
}


//__________________________________________________________________________________|___________

TString TQSample::getTreename() {
  // retrieve the treename associated with this sample
  TList * treeLocations = splitTreeLocations(fTreeLocation);
  TString treeLocation;
  if (treeLocations) {
    if (treeLocations->GetEntries() > 0)
      treeLocation = TString(treeLocations->First()->GetName());
    delete treeLocations;
  }

  return extractTreename(treeLocation);
}


//__________________________________________________________________________________|___________

bool TQSample::addSubSample(TQSample * subSample) {
  // add a subsample to this sample
  // can only be done if this sample is not already a subsample
  if(this->isSubSample()){
    return false;
  }
  if (!subSample) {
    return false;
  }
  if(!this->addFolder(subSample)) {
    return false;
  }
  return true;
}


//__________________________________________________________________________________|___________

TQSample* TQSample::addSelfAsSubSample(const TString& name) {
  // add a copy of yourself as a subsample

  if(name.IsNull()) return NULL;
  TQSample* subSample = new TQSample(name);
  if(!this->addSubSample(subSample)){
    delete subSample;
    return NULL;
  }
 
  subSample->importTags(this);
  return subSample;
}



//__________________________________________________________________________________|___________

TQSample * TQSample::getSubSample(const TString& path) {
  // get the subsample matching a given path

  TQSampleFolder * sampleFolder = getSampleFolder(path);
  if (sampleFolder && sampleFolder->InheritsFrom(TQSample::Class()))
    return static_cast<TQSample*>(sampleFolder);
  else 
    return 0;
}

//__________________________________________________________________________________|___________

bool TQSample::isSubSample(){
  // return true if this sample is the direct child of some other sample
  // false otherwise
  return this->getBaseSample();
}

//__________________________________________________________________________________|___________

TQSample * TQSample::getBaseSample() {
  // retrieve the base sample of this sample
  return dynamic_cast<TQSample*>(this->getBase());
}

//__________________________________________________________________________________|___________

TFile* TQSample::getFile(){
  // retrieve the file pointer of this sample
  return this->fFile;
}

//__________________________________________________________________________________|___________

TQSample::~TQSample() {
  // default destructor
  if (fTokens)
    delete fTokens;
}

//__________________________________________________________________________________|___________


bool TQSample::hasSubSamples() {
  // determine if this sample has any sub-samples
  bool retval = false;
  /* try to find a sub sample */
  TCollection * subSamples = this->getListOfSamples("?");
  DEBUGclass("subsamples: %x (len=%d)",subSamples,(subSamples ? subSamples->GetEntries() : 0));
  #ifdef _DEBUG_
  this->print("d");
  #endif
  if(subSamples){
    if(subSamples->GetEntries() > 0){
      retval = true;
    }
    delete subSamples;
  }
  return retval;
}

//__________________________________________________________________________________|___________
bool TQSample::createFriendLinksForSamplesWithTreeLocation(TQSampleFolder* otherSF) {
  // crawl all samples in otherSF at once, creating friend links between all of them
  // returns true if 'this' sample is amongst the processed TQSample instances
  // and false otherwise.
  if (!otherSF) otherSF = this;
  bool isIncluded = false;
  TQSampleIterator itr(otherSF->getListOfSamples("*"));
  std::map<TString,TQSample*> locationMap;
  while(itr.hasNext()) {
    TQSample* thisSample = itr.readNext();
    if (!thisSample) continue;
    if (thisSample == this) isIncluded = true;
    TString location = thisSample->getTreeLocation();
    if (location.Length() > 0 || !thisSample->hasSubSamples()) {
      if (location.Length() == 0) { //all samples without subsamples need to be in one group in order to ensure that the backpropagation to higher level sample(folder)s works!
        location = TString("<noFile>");
      }
      TQSample* existing = locationMap[location];
      if (existing) { //there was already a sample with this tree location at some point, let's befriend with it.
        existing->befriend(thisSample);
      } else { //we have not yet encountered a sample with this tree location
        thisSample->befriend(thisSample);
        locationMap[location] = thisSample;
      }
    }
  }
  return isIncluded;
}

//__________________________________________________________________________________|___________

void TQSample::findFriendsInternal(TQSampleFolder* otherSF, bool forceUpdateSubsamples){
  // find the friends of this sample folder in some other sample folder
  
  //create all links between samples in otherSF: (reduced complexity ( N^2 -> N*log(N) ) compared to individual iterations!)
  //note: the actual "filter" which reduces the complexity is the calling "findFriends" method in TQSampleFolder which, 
  //unless 
  bool isLinked = this->createFriendLinksForSamplesWithTreeLocation(otherSF);
  bool hasSubSamples = this->hasSubSamples(); 
  if (!isLinked || hasSubSamples) {
    TQSampleIterator itr(otherSF->getListOfSampleFolders("*",TQSample::Class()));
    if (!isLinked) { //this part is only relevant if this sample hasn't been processed yet.
      const TString myTreeLocation = this->getTreeLocation();
      if (myTreeLocation.Length()>0 || !hasSubSamples) { //we don't want to consider super-samples which don't have a tree themselves (this would cause a lot of mis-matching!)
        
        while(itr.hasNext()){
          TQSample* s = itr.readNext();
          if (!s || s->hasFriends()) continue; //if the other sample already has friends we don't need to bother, it's already done
          const TString treelocation = s->getTreeLocation();
          DEBUGclass("looking at friend candidate '%s'",s->getPath().Data());
          if(TQStringUtils::equal(treelocation,myTreeLocation) ){
            DEBUGclass("found friend '%s'",s->getPath().Data());
            //INFOclass("befriending fundamental samples");
            this->befriend(s);
            break; //we're already done here as friend lists are shared pointers, i.e., if we befriend one sample we automatically befriend all with the same treeLocation
          }
        }
      }
    }
    //now check if all sub-samples are friends and if so, befriend with their sup-folder
    if (hasSubSamples) {
      itr.reset();
      TQSampleIterator subitr(this->getListOfSamples("?"));
      while(itr.hasNext()){
        TQSample* other = itr.readNext();
        //we apply the requirements: existence (not a null pointer), the other sample has not already found its set of friends, has sub samples, same number of (sub) samples as this sample.
        if (!other || other->hasFriends() || !other->hasSubSamples() || this->getNSamples() != other->getNSamples() ) continue;
        bool befriend = true;
        while(subitr.hasNext() && befriend) {
          TQSample* thisSub = subitr.readNext();
          if (!thisSub) continue;
          TQSample* otherSub = other->getSubSample(thisSub->GetName());
          if (!otherSub) {befriend = false; break;}
          
          if (forceUpdateSubsamples || (!thisSub->hasFriends()) ) thisSub->findFriends(otherSF, forceUpdateSubsamples);
          if (forceUpdateSubsamples || (!otherSub->hasFriends()) ) otherSub->findFriends(otherSF, forceUpdateSubsamples);
          
          befriend = befriend && thisSub->isFriend(otherSub);
        }
        subitr.reset();
        if (befriend) {
          //INFOclass("Befriending super samples ");
          this->befriend(other);
        }
      }
    }
  }
  
}




