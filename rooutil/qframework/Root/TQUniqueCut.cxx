#include "QFramework/TQUniqueCut.h"
#include "QFramework/TQStringUtils.h"
#include <algorithm>

// #define _DEBUG_
#include "QFramework/TQLibrary.h"

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQUniqueCut:
//
// This class is a special version of the TQCompiledCut.
// The intention of this class is to provide functionality to remove
// overlap between a set of samples.
// 
// This cut is most sensibly used as the root of a hierarchy of TQCompiledCuts.
// As such, it will take care that every event will only be considered once.
// All RunNumbers and EventNumbers are stored and compared with to achieve this,
// any events recognized as duplicates will be rejected.
// Please note that this procedure is very demanding in terms of CPU and RAM,
// and might easily lead to severe problems when analysing large sample sets.
//
// In such cases, please consider removing the overlap on nTuple code level.
// 
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQUniqueCut)

// this is necessary because std::is_sorted is defined in c++11
// in order for this code to compile with older compilers as well,
// this function needs to be provided
#if __cplusplus >= 201103L
#define IS_SORTED(first,last) std::is_sorted(first,last)
#else 
template<class ForwardIt>
bool is_sorted(ForwardIt first, ForwardIt last){
  if (first != last) {
    ForwardIt next = first;
    while (++next != last) {
      if (*next < *first)
        return ( next == last );
      first = next;
    }
  }
  return true;
}
#define IS_SORTED(first,last) is_sorted(first,last);
#endif

bool TQUniqueCut::isMergeable() const {
  // returns false, since TQUniqueCuts are never mergeable
  return false;
}

bool TQUniqueCut::setBranchNames(const TString& runBranch, const TString& evtBranch){
  // set the branch names used for run number and event number
  this->eventNumberBranch = evtBranch;
  this->runNumberBranch = runBranch;
  return true;
}

void TQUniqueCut::initUniqueCut(){
  // initialize this instance of TQUniqueCut
  this->SetName("UniqueCut");
  this->enabled = true;
}

TQUniqueCut::TQUniqueCut() :
  runNumberBranch("run"),
  eventNumberBranch("event")
{
  // default constructor of the TQUniqueCut
  this->initUniqueCut();
}

TQUniqueCut::TQUniqueCut(const TString& name) :
  runNumberBranch("run"),
  eventNumberBranch("event")
{
  // default constructor of the TQUniqueCut
  this->initUniqueCut();
  this->SetName(name);
}

TQUniqueCut::TQUniqueCut(const TString& runBranch, const TString& evtBranch) :
  runNumberBranch(runBranch),
  eventNumberBranch(evtBranch)
{
  // constructor of the TQUniqueCut class, taking branch names for run and event number
  this->initUniqueCut();
}
TQUniqueCut::TQUniqueCut(const TString& name, const TString& runBranch, const TString& evtBranch) :
  runNumberBranch(runBranch),
  eventNumberBranch(evtBranch)
{
  // constructor of the TQUniqueCut class, taking a name as well as branch names for run and event number
  this->initUniqueCut();
  this->SetName(name);
}

void TQUniqueCut::clear(){
  // clear all saved information
  this->runNumbers.clear();
  this->eventNumbers.clear();
}

void TQUniqueCut::setActive(bool active){
  // activate/deactivate this instance
  this->enabled = active;
}

TObjArray* TQUniqueCut::getOwnBranches(){
  // add the branches required by this instance of TQUniqueCut to the internal list of branches
  TObjArray* bNames = new TObjArray();
  bNames->Add(new TObjString(this->runNumberBranch));
  bNames->Add(new TObjString(this->eventNumberBranch));
  return bNames;
}

bool TQUniqueCut::checkUnique(std::vector<int>& entries, int newEntry){
  // check if an entry is new to a sorted vector
  // if so, add it at the appropriate place and return true
  // else return false
  DEBUGclass("entering function");
  int idx = entries.size() -1;
  DEBUGclass("starting search at index %d",idx);
  if(idx < 0){
    entries.push_back(newEntry);
    DEBUG("adding %d as first event for this run",newEntry);
    return true;
  }
  DEBUGclass("checking index at %d",idx);
  if(newEntry > entries[idx]){
    DEBUG("adding %d ",newEntry);
    entries.push_back(newEntry);
    return true;
  }
  DEBUG("searching for event %d ",newEntry);
  while(idx >= 0 && newEntry > entries[idx]){
    idx--;
  }
  if((idx >= 0) && (newEntry == entries[idx])){
    WARNclass("event %d already seen, discarding (%d found at %d)",newEntry,entries[idx],idx);
    return false;
  } else {
    DEBUG("didnt find event %d, adding ",newEntry);
    int newidx = idx+1;
    entries.insert(entries.begin()+newidx,newEntry);
  }
  return true;
}

int TQUniqueCut::getIndex(std::vector<int>& entries, int entry){
  // retrieve the index of an entry in a sorted vector
  // if the entry is not in the vector yet, add it appropriately
  // return the index of the element
  DEBUGclass("attempting to find index of %d",entry);
  if(entries.size() < 1){
    DEBUGclass("adding %d as first entry",entry);
    entries.insert(entries.begin(),entry);
    return 0;
  }
  int idx = entries.size() -1;
  DEBUGclass("checking entry %d against %d",entry,entries[idx]);
  if(entry > entries[idx]){
    DEBUGclass("adding entry %d at %d",entry,idx);
    entries.push_back(entry);
    return idx+1;
  }
  DEBUGclass("searching for entry %d",entry);
  while(idx >= 0 && entry < entries[idx]){
    idx--;
  }
  DEBUGclass("stopping search at index %d",idx);
  if(idx < 0 || entry > entries[idx]){
    int newidx = 1+idx;
    DEBUGclass("did not find entry %d, inserting  at %d",entry,newidx);
    entries.insert(entries.begin()+newidx,entry);
    return newidx;
  }
  DEBUGclass("found %d==%d at %d",entry,entries[idx],idx);
  return idx;
}


bool TQUniqueCut::passed() const {
  // checks the run and event numbers of the current event
  // returns true if they are unique, false if this combination 
  // has already been encountered before
  if(!this->enabled) return true;

  int event = this->eventNumberObservable->getValue();
  int run = this->runNumberObservable->getValue();

  DEBUGclass("checking unique event %d : %d",run,event);
  int runIdx = getIndex(runNumbers,run);

  if(runNumbers.size() >= eventNumbers.size()){
    DEBUGclass("extending eventnumber list");
    auto b = eventNumbers.begin();
    std::vector<int> newelem;
    eventNumbers.insert(b+runIdx,newelem);
  }
  #ifdef _DEBUG_
  if(runIdx >= (int)(eventNumbers.size())){
    throw std::runtime_error("event number list has insufficient length!");
  } else {
    DEBUGclass("event number list length %d is suffucient for run index %d",(int)(eventNumbers.size()),runIdx);
  }
  #endif
  
  DEBUGclass("checking unique for run %d at %d (%d events)",runNumbers[runIdx],runIdx,eventNumbers[runIdx].size());
  const bool unique = TQUniqueCut::checkUnique(eventNumbers[runIdx],event);
  DEBUGclass("returning %d",unique);
  if(!unique){
    DEBUGclass("event known: %d : %d",run,event);
    #ifdef _DEBUG_
    this->printLists();
    #endif
  } else {
    DEBUGclass("new event: %d : %d",run,event);
  }
  return unique;
}

bool TQUniqueCut::initializeObservables() {
  // initialize the observables required by this TQUniqueCut
  this->eventNumberObservable = TQObservable::getObservable(this->eventNumberBranch,this->fSample);
  this->runNumberObservable = TQObservable::getObservable(this->runNumberBranch,this->fSample);

  if (!runNumberObservable || !eventNumberObservable) return false;
  bool ok = true;
  if (!runNumberObservable ->initialize(this->fSample)) {
    ERRORclass("Failed to initialize runNumberObservable obtained from expression '%s' in TQUniqueCut '%s' for sample '%s'",this->runNumberBranch.Data(), this->GetName(), this->fSample->getPath().Data()); 
    ok = false;
  }
  
  if (!eventNumberObservable->initialize(this->fSample)){
    ok = false;
  }
  
  if (!ok) {
    this->finalizeObservables();
    return false;
  }
  return true;
}

bool TQUniqueCut::finalizeObservables() {
  // finalize the observables required by this TQUniqueCut
  if (runNumberObservable && eventNumberObservable){
    return runNumberObservable ->finalize() && eventNumberObservable ->finalize();
  }
  if(runNumberObservable) runNumberObservable ->finalize();
  if(eventNumberObservable) eventNumberObservable ->finalize();
  return false;
}

void TQUniqueCut::printLists() const {
  // print the internal lists of run and event numbers
  // WARNING: these lists can be excessively long!
  for(size_t i=0; i<runNumbers.size(); i++){
    std::cout << "===" << runNumbers[i] << "===" << std::endl;
    for(size_t j=0; j<eventNumbers[i].size()-1; j++){
      std::cout << eventNumbers[i][j] << ",";
    }
    std::cout << eventNumbers[i][eventNumbers[i].size()-1] << std::endl;
  }
}

bool TQUniqueCut::isSorted(int verbose) const {
  // check if the internal lists are properly sorted
  // this function is for debugging purposes only
  // under any reasonable cirumstances, it should return true
  if(!IS_SORTED(runNumbers.begin(),runNumbers.end())){
    if(verbose > 0) ERRORclass("run number list not sorted");
    if(verbose > 1) TQStringUtils::printVector<int>(runNumbers);
    return false;
  }
  bool sorted = true;
  for(size_t i=0; i<runNumbers.size(); i++){
    if(!IS_SORTED(eventNumbers[i].begin(),eventNumbers[i].end())){
      if(verbose) ERRORclass("event number list for run %d not sorted!",runNumbers[i]);
      if(verbose > 1){
        for(size_t j=0; j<eventNumbers[i].size()-1; j++){ std::cout << eventNumbers[i][j] << " "; }
        std::cout << eventNumbers[i][eventNumbers.size()-1] << std::endl;
      }
      sorted = false;
    }
  }
  if(!sorted) return false;
  if(verbose) VERBOSEclass("all lists sorted");
  return true;
}


bool TQUniqueCut::initializeSelfSampleFolder(TQSampleFolder* sf){
  //@tag:[resetUniqueCut] If this boolean sample folder tag is set to true the run and event numbers of this cut are cleared upon initialization and finalization of the unique cut.
  if(sf->getTagBoolDefault("resetUniqueCut",false)) this->clear();
  return true;
}

bool TQUniqueCut::finalizeSelfSampleFolder(TQSampleFolder* sf){
  if(sf->getTagBoolDefault("resetUniqueCut",false)) this->clear();
  return true;                                        
}
