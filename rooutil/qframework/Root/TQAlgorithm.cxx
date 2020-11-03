#include "QFramework/TQAlgorithm.h"
#include "QFramework/TQSample.h"

#include "QFramework/TQLibrary.h"

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQAlgorithm
//
// The TQAlgorithm class is a hook in order to precompute arbitrary
// event-based quantities and cache their values. The entire list of
// algorithms is executed on every event.
//
// every algorithm should implement the initialize, finalize and execute and cleanup methods.
// initialize will be called at the beginning of every sample and will take the sample as an argument
// finalize, execute and cleanup will not receive arguments and are called
// at the end of each sample (finalize), before each event (execute) and after each 
// event (cleanup).
// All four functions should return true in case of success, false in case of failure
// failure of any algorithm will cause the event loop to halt,
// additional error handling may be done by raising exceptions.
//
// All input/output managing is entirely up to the user and may be
// done via filling the TStore, creating decorations to objects in
// the TEvent or any other available operation.
//
// Please note that for use of TQAlgorithms with the TQMultiChannelAnalysisSampleVisitor
// must be streamable, i.e., correctly work with the TObject::Clone() method. This implies
// that all transient members (typically marked with the '//!' suffix in the header file)
// must be set in the 'TQAlgorithm::initialize(TQSample* s)' method! This also applies to 
// TQAlgorithms which are supposedly singletons (i.e. returning 'true' in TQAlgorithm::isSingleton())
// for which 'TQAlgorithm::initializeSingleton(const std::vector<TQSample*>& samples)' should 
// take the place of 'initialize(TQSample* s)'.
// If an implementation of an algorithm is marked as a singleton but should still be useable
// with SCASV as well as MCASV (or similar classes) BOTH initialization methods must be
// overridden!
//
////////////////////////////////////////////////////////////////////////////////////////////////

bool TQAlgorithm::Manager::addAlgorithm(TQAlgorithm* newAlgorithm){
  // add a new cache
  // the function takes ownership of the object, you shall not delete it!
  for(size_t i=0; i<this->gAlgorithmList.size(); ++i){
    if(TQStringUtils::equal(gAlgorithmList[i]->GetName(),newAlgorithm->GetName())){
      return false;
    }
  }
  this->gAlgorithmList.push_back(newAlgorithm);
  return true;
}

void TQAlgorithm::Manager::clearAlgorithms(){
  // clear the list of caches, deleting every one of them
  for(size_t i=0; i< this->gAlgorithmList.size(); ++i){
    delete this->gAlgorithmList[i];
  }
  this->gAlgorithmList.clear();
}

void TQAlgorithm::Manager::clearClonedAlgorithms() {
  // clear the list (of lists) of active caches, deleting every one of them
  for (auto& origAlgAndSubMapPair: this->gAlgorithmStore) {
    for (auto& stringAndClonedAlgPair : origAlgAndSubMapPair.second) {
      delete stringAndClonedAlgPair.second;
    }
    origAlgAndSubMapPair.second.clear();
  }
  this->gAlgorithmStore.clear();
}

const std::vector<TQAlgorithm*>& TQAlgorithm::Manager::getAlgorithms(){
  // retrieve a const reference to the list of all caches
  return this->gAlgorithmList;
}


void TQAlgorithm::Manager::printAlgorithms() const {
	// print the list of all currently registered algorithms
  std::cout<<"Registered algorithms:"<<std::endl;
  for (TQAlgorithm* const& alg : gAlgorithmList) {
    std::cout<<"  "<<(alg?alg->GetName():"NULL")<<std::endl;
  }
}

void TQAlgorithm::Manager::printActiveAlgorithms() const {
	// print the list of all currently active algorithms
  std::cout<<"Active algorithms:"<<std::endl;
  for (TQAlgorithm* const& alg : gActiveAlgorithms) {
    std::cout<<"  "<<(alg?alg->GetName():"NULL")<<std::endl;
  }
}
/*
bool TQAlgorithm::Manager::cloneAlgorithms(int n) {
  // create n clones of each cache unless the cache is a singleton (in which case only one clone is created)
  if (this->gClonedAlgorithmList.size() != 0) {
    WARNfunc("The list of cloned algorithms is not empty. Will implicitly call TQAlgorithm::Manager::clearClonedAlgorithms() before creating a fresh set of clones! Please consider performing this call explicitly before requesting a new set of clones to be created.");
    this->clearClonedAlgorithms();
  }
  for (const auto& origAlg: this->gAlgorithmList) {
    this->gClonedAlgorithmList.push_back(std::vector<TQAlgorithm*>());
    const size_t index = this->gClonedAlgorithmList.size()-1;
    if (origAlg->isSingleton()) { //single clone if algorithm should run once per event
      this->gClonedAlgorithmList[index].push_back( static_cast<TQAlgorithm*>( origAlg->Clone() ) );
    } else {
      for (int i=0;i<n;++i) { //n clones if n instances of the algorithm should run per event (typically n channels)
        this->gClonedAlgorithmList[index].push_back( static_cast<TQAlgorithm*>( origAlg->Clone() ) );
      }
    }
  }
  return true;
  
}
*/

bool TQAlgorithm::Manager::initializeAlgorithms(TQSample*s){
  // initialize all algorithms
  for(size_t i=0; i<this->gAlgorithmList.size(); ++i){
    if(! this->gAlgorithmList[i]->initialize(s)) return false;
  }
  return true;
}

void TQAlgorithm::Manager::resetActiveAlgorithms() {
	// clear the list of currently active algorithms
  gActiveAlgorithms.clear();
}


bool TQAlgorithm::Manager::initializeClonedAlgorithms(std::vector<TQSample*> & samples, const TString& tagKey){
  // initialize all already cloned or to-be-cloned algorithms (multi-channel variant)
  // creates clones of original algorithms if needed.
  this->resetActiveAlgorithms();
  
  for (TQAlgorithm* & origAlg: this->gAlgorithmList) {
    if (!origAlg) continue;
    if (origAlg->isSingleton()) {
      //the simple case, we just add it to the active algorithms and initialize it on all (active) samples at once
      if (!origAlg->initializeSingleton(samples)) {
        throw std::runtime_error(TString::Format("Failed to initialize algorithm '%s' as singleton",origAlg->GetName()).Data());
        return false;
      }
      gActiveAlgorithms.push_back(origAlg);
      continue;//we're done with this algorithm for now
    }
    
    if (gAlgorithmStore.count(origAlg) == 0) {//there are no clones at all yet for this algorithm (and it's not a singleton)
      std::map<TString,TQAlgorithm*> algMap;
      for ( TQSample*& s  : samples) { //we had nothing before so we need to create the whole structure for each sample we are provided with
        TString key;
        if (!s->getTagString(tagKey,key)) {//cannot identify channel -> error (wouldn't know where to store it)
          throw std::runtime_error(TString::Format("Could not obtain channel identifier from sample '%s' using key '%s'",s->getPath().Data(), tagKey.Data()).Data());
          return false;
        }
        //clone origAlg, initialize clone and add to map and active algorithms
        TQAlgorithm* clone = static_cast<TQAlgorithm*>( origAlg->Clone() );
        clone->initialize(s);
        algMap[key] = clone; 
        gActiveAlgorithms.push_back(clone); //ownership belongs to algMap!
      }
      //add algMap to gAlgorithmStore 
      gAlgorithmStore[origAlg] = algMap;
    } else {//there are already at least some clones of this algorithm, let's re-use them where possible
      std::map<TString,TQAlgorithm*>& algMap = gAlgorithmStore[origAlg]; 
      for ( TQSample*& s : samples) {
        TString key;
        if (!s->getTagString(tagKey,key)) {//cannot identify channel -> error (wouldn't know where to look for/store clones of the algorithm)
          throw std::runtime_error(TString::Format("Could not obtain channel identifier from sample '%s' using key '%s'",s->getPath().Data(), tagKey.Data()).Data());
          return false;
        }
        if (algMap.count(tagKey) == 0) { //for this particular channel no clone exists yet
          TQAlgorithm* clone = static_cast<TQAlgorithm*>( origAlg->Clone() );
          clone->initialize(s);
          algMap[key] = clone; 
          gActiveAlgorithms.push_back(clone); //ownership belongs to algMap!
        } else { //there is already a clone for this channel, we use it
          TQAlgorithm* clone = algMap[tagKey];
          clone->initialize(s);
          gActiveAlgorithms.push_back(clone);
        }
      }
      
    }
    
  }
  
  /*
  for(const std::vector<TQAlgorithm*>& clones : this->gClonedAlgorithmList){
    if (clones.size()==1  && clones[0]->isSingleton()) {
      if (!clones[0]->initializeSingleton(samples)) return false; //initialize singleton algorithm with list of all samples
    } else if (clones.size()==samples.size()) {
      //everything should be fine now, let's actually initialize the algorithms
      for (size_t i=0;i<clones.size();++i) {
        if (!clones[i]->initialize(samples[i]) ) return false;
      }
    } else { //we neither have a singleton algorithm nor does the number of clones match the number of samples -> ERROR
      ERRORclass("Failed to initialize cloned algorithms: number of clones differs from number of TQSamples provided!");
      return false;
    }
  }
  */
  return true;
} //end initializeClonedAlgorithms


bool TQAlgorithm::Manager::finalizeAlgorithms(){
  // finalize all algorithms
  
    for (const auto& alg : gAlgorithmList) {
      if (!alg->finalize()) return false;
    }
//  for(size_t i=0; i<this->gAlgorithmList.size(); ++i){
//    if(! this->gAlgorithmList[i]->finalize()) return false;
//  }
  return true;
}

bool TQAlgorithm::Manager::finalizeClonedAlgorithms(){
  // finalize all algorithms (multi-channel variant)
  for (TQAlgorithm*const & alg : gActiveAlgorithms) {
    if (!alg->finalize()) return false;
  }
  return true;
}

bool TQAlgorithm::Manager::executeAlgorithms(){
  // execute all algorithms
  //for(size_t i=0; i<this->gAlgorithmList.size(); ++i){
    for (const auto& alg : gAlgorithmList) {
      if (!alg->execute()) return false;
    }
    //if(! this->gAlgorithmList[i]->execute()) return false;
  return true;
}

bool TQAlgorithm::Manager::executeClonedAlgorithms(){
  // execute all algorithms (multi-channel variant)
  for (TQAlgorithm* const & alg : gActiveAlgorithms) {
    if (!alg->execute()) return false;
  }
  return true;
}

bool TQAlgorithm::Manager::cleanupAlgorithms(){
  // cleanup all algorithms (multi channel variant)
  // post event method
    for (TQAlgorithm* const & alg : gAlgorithmList) {
      if (!alg->cleanup()) return false;
  }
  return true;
}

bool TQAlgorithm::Manager::cleanupClonedAlgorithms(){
  // cleanup all algorithms (multi channel variant)
  // post event method
  // actually a stupid name
  for ( TQAlgorithm* const & alg : gActiveAlgorithms) {
      if (!alg->cleanup()) return false;
    }
  return true;
}

bool TQAlgorithm::isSingleton() const {
	// return true if this algorithm is a singleton, false otherwise
  return false;
}

bool TQAlgorithm::initializeSingleton(const std::vector<TQSample*>& samples) {
	// default implementations simply report an error, it is the responsibility of the author of the derived class to implement at least one of these
  ERRORclass("initialization with a list of samples is not implemented for TQAlgorithm with name '%s'!", this->GetName());
  return false;
}

bool TQAlgorithm::initialize(TQSample* s) {
	// default implementations simply report an error, it is the responsibility of the author of the derived class to implement at least one of these
  ERRORclass("initialization with single TQSample is not implemented for TQAlgorithm with name '%s'!",this->GetName());
  return false;
}

