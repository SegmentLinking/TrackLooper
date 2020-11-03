#include "QFramework/TQMultiChannelAnalysisSampleVisitor.h"

#include "QFramework/TQCounter.h"
#include "QFramework/TQUtils.h"
#include "QFramework/TQSample.h"
#include "QFramework/TQCut.h"
#include "QFramework/TQStringUtils.h"
#include "QFramework/TQIterator.h"

#include "TList.h"
#include "TObjString.h"
#include "TStopwatch.h"

// #define _DEBUG_
#include "QFramework/TQLibrary.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstdlib>

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQMultiChannelAnalysisSampleVisitor:
//
// Visit samples and execute analysis jobs at cuts in a parallelized way.
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQMultiChannelAnalysisSampleVisitor)


//__________________________________________________________________________________|___________

TQMultiChannelAnalysisSampleVisitor::TQMultiChannelAnalysisSampleVisitor() : 
TQAnalysisSampleVisitorBase("mcasv",false)
{
  // Default constructor of the TQMultiChannelAnalysisSampleVisitor class
  this->setVisitTraceID("analysis");
  this->setTagString("tagKey","~.mcasv.channel");
}

//__________________________________________________________________________________|___________

TQMultiChannelAnalysisSampleVisitor::~TQMultiChannelAnalysisSampleVisitor(){
  // default destructor
}

//__________________________________________________________________________________|___________

int TQMultiChannelAnalysisSampleVisitor::visitFolder(TQSampleFolder * sampleFolder, TString&/*message*/) {
  //if this was already visited, we don't need to do so again
  if(!sampleFolder) return visitFAILED;
  DEBUG("checking visit on '%s'",sampleFolder->getPath().Data());
  if (this->checkVisit(sampleFolder)){
    DEBUG("folder '%s' has been visited before...",sampleFolder->getPath().Data());
    return visitIGNORE;
  }
  const bool requireRestrictionTag = TQSampleVisitor::checkRestrictionTag(sampleFolder);
  const TString tagKey = this->getTagStringDefault("tagKey","~.mcasv.channel");
  updateFriends(sampleFolder);
  std::set<TString> foundChannels;
  std::shared_ptr<std::set<TQSampleFolder*>> friends = sampleFolder->getFriends();
  //while(itr.hasNext()) {
  for (auto sf : (*friends)) {
  //  TQSampleFolder* sf = itr.readNext();
    DEBUG("checking if '%s' has been visited before...",sf->getPath().Data());
    if(this->checkVisit(sf)) continue; // this line highly depends on the correct implementation of the friend finding. We skip initialization of cuts (and subsequent objects) without knowing if really all samples further down the hierarchy have been visited before (via friends). If the friend finding works correctly two sample folders can be friends if their substructure is identical
    //-> additional safe guard in revisitFolder: check if all subfolders have been visited!
    if (requireRestrictionTag && !TQSampleVisitor::checkRestrictionTag(sf)) continue; //don't process friends which are not selected
    TString channel = ""; 
    if (!sf->getTagString(tagKey,channel) ) {
      DEBUG("skipping '%s' due to missing channel tag (might not be a channel specific folder)",sf->getPath().Data());
      continue;
    }
    if(foundChannels.find(channel) != foundChannels.end()){
      DEBUG("skipping '%s' due to channel conflict in channel '%s'...",sf->getPath().Data(),channel.Data());
      continue;
    }
    TQCut* cut = this->fChannels[channel];
    if(!cut){
      DEBUG("no cut avialable for channel '%s' required by '%s'- skipping",channel.Data(),sf->getPath().Data());
      return visitFAILED;
    }
    if (!cut->canInitialize(sf)) continue; //don't use this friend if it is on a different path than one the cut was previously intitialized on (prevent cross-talk between different paths!)
    DEBUG("trying to initialize SampleFolder with path '%s'",sf->getPath().Data());
    if(!cut->initializeSampleFolder(sf)) return visitFAILED;
    //this->stamp(sf); //moved to revisitFolder (only stamp once we know all subfolders are processed)
    foundChannels.insert(channel);
  }
  return visitLISTONLY;
}

//__________________________________________________________________________________|___________

int TQMultiChannelAnalysisSampleVisitor::revisitSample(TQSample */*sample*/, TString&/*message*/) {
  // revisit an instance of TQSample on the way out
  return visitIGNORE;
}



//__________________________________________________________________________________|___________

int TQMultiChannelAnalysisSampleVisitor::revisitFolder(TQSampleFolder * sampleFolder, TString&/*message*/) {
  
  if (!sampleFolder) return visitFAILED;
  const TString tagKey = this->getTagStringDefault("tagKey","~.mcasv.channel");
  bool finalized = true;
  const bool requireRestrictionTag = TQSampleVisitor::checkRestrictionTag(sampleFolder);
  //TQSampleFolderIterator itr(sampleFolder->getFriends());
  std::shared_ptr<std::set<TQSampleFolder*>> friends = sampleFolder->getFriends();
  
  //while(itr.hasNext()) {
    //TQSampleFolder* sf = itr.readNext();
  for (auto sf : (*friends) ) {  
    TString channel = sf->getTagStringDefault(tagKey,"");
    TQCut* cut = this->fChannels[channel];
    if(!cut) continue;
    if (!cut->canFinalize(sf)) continue; //only finalize on sample folders which we have actually initialized before
    bool thisFinalized = cut->finalizeSampleFolder(sf);
    //std::cout<<"result of finalization was "<<(thisFinalized?"true":"false")<<std::endl;
    //check if all subfolders have been visited
    if ( thisFinalized ) {
      TQSampleFolderIterator itr(sf->getListOfSampleFolders("?"), true);
      while (itr.hasNext()) {
        TQSampleFolder* sub = itr.readNext();
        bool subFinalized = ( this->checkVisit(sub)/*either visited*/ || (requireRestrictionTag && !TQSampleVisitor::checkRestrictionTag(sub) ) /*or not scheduled for processing*/ )   ; //check if the subfolder has been visited
        if (!subFinalized) WARNclass("This sample(folder) '%s' was apparently not visited despite being scheduled. Please verify that it was processed at a later point!",sub->getPath().Data());
        thisFinalized = thisFinalized && subFinalized;
      }
      if ( thisFinalized ) this->stamp(sf);  //stamp if everything is still fine
    }
    
    finalized = finalized && thisFinalized; 
    
    
    //generalization of objects is not yet ported to TQMultiChannelAnalysisSampleVisitor. the Following lines are just a copy from TQAnalysisSampleVisitor to give a rough guideline
    /* f
    bool generalizeHistograms = false;
    bool generalizeCounter = false;
    bool generalizeCounterGrids = false;
    //@tag: [.asv.generalize.<objectType>] These folder tags determine if objects of type "<objectType>" (=histograms,counter,countergrid) should be generalized using TQSampleFolder::generalizeObjects. Default: true
    sampleFolder->getTagBool(".asv.generalize.histograms", generalizeHistograms, true);
    sampleFolder->getTagBool(".asv.generalize.counter", generalizeCounter, true);
    sampleFolder->getTagBool(".asv.generalize.countergrid", generalizeCounterGrids, true);
  
    bool generalized = false;
    bool doGeneralize = generalizeHistograms || generalizeCounter || generalizeCounterGrids;
  
    if (doGeneralize){
  
      // start the stop watch 
      TStopwatch * timer = new TStopwatch();
  
      // generalize all histograms 
      int nHistos = 0;
      if (generalizeHistograms)
        nHistos = sampleFolder->generalizeHistograms();
  
      // generalize all counter 
      int nCounter = 0;
      if (generalizeCounter)
        nCounter = sampleFolder->generalizeCounter();
  
      // generalize all counter 
      int nCounterGrid = 0;
      if (generalizeCounterGrids)
        nCounterGrid = sampleFolder->generalizeCounterGrid();
  
      // stop the timer 
      timer->Stop();
  
      message.Append(" ");
      message.Append(TQStringUtils::fixedWidth("--", 12,"r"));
      message.Append(TQStringUtils::fixedWidth(TString::Format("%.2f", timer->RealTime()), 12,"r"));
      message.Append(" ");
      
      if (nHistos == 0 && nCounter == 0 && nCounterGrid == 0)
        message.Append("nothing to generalize");
      else{
        message.Append("generalized: ");
        if(nHistos > 0){
          if (generalizeHistograms) message.Append(TString::Format("%d histograms", nHistos));
        }
        if(nCounter + nCounterGrid > 0 && nHistos > 0){
          message.Append(", ");
        }
        if(nCounter > 0){
          message.Append(TString::Format("%d counters", nCounter));
        }
        if(nCounterGrid > 0 && nCounter > 0){
          message.Append(", ");
        }
        if(nCounterGrid > 0){
          message.Append(TString::Format("%d grids", nCounterGrid));
        }
      }
   
      // delete the timer 
      delete timer;
  
      if (nHistos > 0 || nCounter > 0 || nCounterGrid > 0)
        generalized = true;
  
    } else {
      if(finalized)
        return visitIGNORE;
    }
    if(generalized || !doGeneralize){
      if(finalized){
        return visitOK;
      } else
        return visitWARN;
    } else {
      if(finalized)
        return visitWARN;
    }
    return visitFAILED;
    */

  }

  return finalized ? visitLISTONLY : visitWARN;

}


//__________________________________________________________________________________|___________

int TQMultiChannelAnalysisSampleVisitor::visitSample(TQSample * sample, TString& message) {
  // Run the analysis jobs on a sample
  if(this->checkVisit(sample)){
    return visitSKIPPED;
  } else {
    #ifdef _DEBUG_
    sample->printTags();
    #endif
  }
    
  TStopwatch * timer = new TStopwatch();

  /* analyse the tree */
  TString analysisMessage;
  #ifndef _DEBUG_
  //TQLibrary::redirect_stderr("/dev/null");
  #endif
  DEBUGclass("analysing tree");
  int nEntries = this->analyseTree(sample, analysisMessage);
  #ifndef _DEBUG_
  TQLibrary::restore_stderr();
  #endif
  /* stop the timer */
  timer->Stop();

  /* compile the message */
  message.Append(" ");

  /* save the number of entries in tree analyzed */
  sample->setTagInteger(TString::Format(".%s.analysis.nentries",this->GetName()),nEntries);

  if (nEntries >= 0) {
    message.Append(TQStringUtils::fixedWidth(TQStringUtils::getThousandsSeparators(nEntries), 12,"r"));
  } else {
    message.Append(TQStringUtils::fixedWidth("--", 12,"r"));
  }
 
  message.Append(TQStringUtils::fixedWidth(TString::Format("%.2f", timer->RealTime()), 12,"r"));
  message.Append(" ");
  message.Append(TQStringUtils::fixedWidth(analysisMessage, 40, "l"));

  /* delete the timer */
  delete timer;

  if(nEntries > 0){
    return visitOK;
  }
  else if (nEntries == 0)
    return visitWARN;

  return visitFAILED;
 
}

//__________________________________________________________________________________|___________

bool TQMultiChannelAnalysisSampleVisitor::checkCut(TQCut * baseCut) {
  // checks if a given cut is already added to some channel
  for (auto it=this->fChannels.begin(); it!=this->fChannels.end(); ++it){
    if(it->second == baseCut) return true;
  }
  return false;
}

//__________________________________________________________________________________|___________

bool TQMultiChannelAnalysisSampleVisitor::checkChannel(const TString& channelName) {
  // checks if a channel is already added to the iterator
  auto it= this->fChannels.find(channelName);
  if(it==this->fChannels.end()){
    return false;
  }
  return true;
}

//__________________________________________________________________________________|___________

void TQMultiChannelAnalysisSampleVisitor::printChannels() {
  // print the currently scheduled channel
  for (auto it=this->fChannels.begin(); it!=this->fChannels.end(); ++it){
    std::cout << TQStringUtils::fixedWidth(it->first,40) << " " << TQStringUtils::fixedWidth(it->second->GetName(),20) << " @" << it->second << std::endl;
  }
}

//__________________________________________________________________________________|___________

void TQMultiChannelAnalysisSampleVisitor::addChannel(const TString& channelName, TQCut * baseCut) {
  // add a new channel to this visitor, scheduling the visit of the given channelName with the given basecut
  if(channelName.IsNull()){
    ERRORclass("unable to use empty channel name");
    return;
  }
  if(baseCut){
    if(this->checkCut(baseCut)){
      ERRORclass("not adding basecut '%s' -- this cut was already scheduled for another channel!",baseCut->GetName());
      return;
    }
    if(this->checkChannel(channelName)){
      ERRORclass("channel '%s' already added to this visitor!",channelName.Data());
      return;
    }
    this->fChannels[channelName] = baseCut;
  } else {
    if(this->checkChannel(channelName)){
      this->fChannels.erase(channelName);
      return;
    } else {
      WARNclass("attempt to add new channel '%s' with baseCut=NULL",channelName.Data());
      return;
    }
  }
}


//__________________________________________________________________________________|___________

TQCut * TQMultiChannelAnalysisSampleVisitor::getBaseCut(const TString& channelName) {
  // retrieve the basecut corresponding to that channelName
  if(this->checkChannel(channelName)){
    return this->fChannels[channelName];
  } else {
    return NULL;
  }
}


//__________________________________________________________________________________|___________

bool TQMultiChannelAnalysisSampleVisitor::stampAllFriends(TQSample* sample) const {
  // stamps all friends as being visited. Note: this should only be used when there
  // is no chance that any of the friends can be visited successfully as it will
  // stamp ALL friends! (i.e. potentially also ones that weren't explicitly checked/
  // visited)
  if (!sample) return false;
  if(!sample->hasFriends()){
    sample->findFriends();
  }
  if(!sample->hasFriends()){
    return false;
  }
  WARNclass("No events could be read for the sample with path '%s'. All samples using the same data source (same input file and tree) will be ignored!",sample->getPath().Data());
  std::shared_ptr<std::set<TQSampleFolder*>> friends = sample->getFriends();
  if (friends == nullptr) return false;
  for (auto fr : (*friends)) {
    if (!fr) continue;
    this->stamp(fr);
  }
  return true;
}

int TQMultiChannelAnalysisSampleVisitor::analyseTree(TQSample * sample, TString& message) {
  // analyse the tree in this sample
  DEBUGclass("entering function");
  DEBUGclass("testing sample");
  if (!sample) {
    message = "sample is NULL";
    DEBUGclass(message);
    return -1;
  }
  
  // let the sample find some friends
  if(!sample->hasFriends()){
    sample->findFriends();
  }
  if(!sample->hasFriends()){
    message = "no friends found";
    WARN("Sample '%s' has no friends. Please check for problems in your setup!",sample->getPath().Data());
  }
  TQToken* tok = sample->getTreeToken();
  TQToken* fileTok = sample->getFileToken();
  if(!tok && !fileTok){

    // .xsp.filepath should exist for data, .init.filepath for mc
    if( sample->hasTagString(".xsp.filepath") || sample->hasTagString(".init.filepath") ) {
      // we have a true error where the file was expected to be retrieved ok
      throw std::runtime_error(TString::Format("Sample '%s' has a .filepath, but file and tree tokens can't be read!",sample->getPath().Data()).Data());
    }

    message="failed to obtain tree and file token, probably because file was never found for the sample";
    this->stampAllFriends(sample); //mark all friends as done
    return -1;
  }
  TTree* tree = nullptr;
  if (tok) tree = static_cast<TTree*>(tok->getContent());
  if(!tree){
    message="failed to retrieve shared tree but sample is readable (it might just be empty)";
    //this->stampAllFriends(sample); //mark all friends as done
    //return -1;
  }
  //WARNING: we do not perform an early exit if there is not tree available (we have to initialize everything in case meta data handling is required in some algorithm for example!). This also means, that 'tree' might be a nullptr at this point!
  
  //check if we should try to initialize cuts (and therefore observables). If the tree is empty, we should not do so as there are likely no branches (i.e. TQTreeFormulaObservables will cause an error)
  
  const bool requireRestrictionTag = TQSampleVisitor::checkRestrictionTag(sample);
  
  //TQSampleIterator itr(sample->getFriends());
  std::shared_ptr<std::set<TQSampleFolder*>> friends = sample->getFriends();
  DEBUGclass("retrieving number of entries");
  const Long64_t nEntries = std::min(tree?tree->GetEntries():0,this->fMaxEvents);

  //@tag:tagKey: control which tag on the sample folders will be used to identify the cut set to be used (default: ~.mcasv.channel)
  const TString tagKey = this->getTagStringDefault("tagKey","~.mcasv.channel");
  if (tree && nEntries>0) tree->SetBranchStatus("*", 1);
  //while(itr.hasNext()){
    //TQSample* s = itr.readNext();
    
  //ensure that for all friends a matching channel exists and create the coresponding observable sets if needed
  for (auto sf : (*friends)) {
    if (!sf || !sf->InheritsFrom(TQSample::Class()) ) continue;
    //if(!s) continue;
    TQSample* s = static_cast<TQSample*>(sf);
    if(requireRestrictionTag && !TQSampleVisitor::checkRestrictionTag(s)) continue; //ignore samples which are not selected
  
    TString channel;
    if(!s->getTagString(tagKey,channel)){
      message = TString::Format("sample '%s' has no channel set as tag with key '%s'.",s->getPath().Data(),tagKey.Data());
      throw std::runtime_error(message.Data()); //critical error, abort!
      return -1;
    }
    if(!this->checkChannel(channel)){
      message = TString::Format("channel '%s' is unknown from %s:%s",channel.Data(),s->getPath().Data(),tagKey.Data());
      throw std::runtime_error(message.Data()); //critical error, abort!
      return -1;
    }
    if(this->fUseObservableSets){
      if(!TQObservable::manager.setActiveSet(channel)){
        TQObservable::manager.cloneActiveSet(channel);
      }
    }
  }
  
  //itr.reset();
  std::vector<bool> useMCweights;
  std::vector<TQCut*> cuts;
  
  std::set<TString> foundChannels;
  std::vector<TQSample*> runningSamples; //all samples being activated for the upcoming event loop
  //while(itr.hasNext()){
    //TQSample* s = itr.readNext();
    //if(!s) continue;
  for (auto sf : (*friends)) {
    if (!sf || !sf->InheritsFrom(TQSample::Class()) ) continue;
    //if(!s) continue;
    TQSample* s = static_cast<TQSample*>(sf);  
    if (this->checkVisit(s)) continue;
    if(requireRestrictionTag && !TQSampleVisitor::checkRestrictionTag(s)) continue; //ignore samples which are not selected
    TString channel;
    if(!s->getTagString(tagKey,channel)){
      throw std::runtime_error(TString::Format("no channel information set on sample '%s'",s->getPath().Data()).Data());
    }
    if(foundChannels.find(channel)!=foundChannels.end()){
      continue;
    }
    TQObservable::manager.setActiveSet(channel);
    TQCut* basecut = this->fChannels[channel];
    //consistency check
    if (!basecut->canInitialize(s)) continue; //check if this sample can currently be initialized (validates that the sample is a subfolder of the TQSampleFolder the cut was last initialized on if any)
    DEBUGclass("initializing sample '%s' with basecut '%p' for channel '%s'",s->getPath().Data(),basecut,channel.Data());
    if (tree && nEntries>0 && !basecut->initialize(s)) { //only initialize if we actually have a tree (but still set the sample!!! (will early exit if tree is a nullptr, i.e., won't attempt to initialize cut)
      throw std::runtime_error(TString::Format("failed to initialize cuts for channel '%s' on sample '%s'",channel.Data(),s->getPath().Data()).Data());
    }
    // check wether to use MC weights
    DEBUGclass("testing for usemcweights tag");
    bool useWeights = false;
    s->getTagBool("usemcweights", useWeights, true);
    
    // push all required information into vectors
    useMCweights.push_back(useWeights);
    cuts.push_back(basecut);
    runningSamples.push_back(s);
    foundChannels.insert(channel);
  }
  // handle branch status
  if (tree && nEntries>0) {
    tree->SetBranchStatus("*", 0);
    for(size_t i=0; i<cuts.size(); ++i){
      TCollection* bnames = cuts[i]->getListOfBranches();
      if(!this->setupBranches(tree,bnames)){
        throw std::runtime_error(TString::Format("failed to setup branches for sample '%s'",cuts[i]->getSample()->getPath().Data()).Data());
      }
      bnames->SetOwner(true);
      delete bnames;
    }
  }

  // loop over tree entries
  DEBUGclass("entering event loop");
  #ifdef _DEBUG_
  std::cout<<"Tree adress: "<<tree<<std::endl;
  #endif
  const size_t n = cuts.size();
  if (n==0) { //something really stupid must have happened. Formally we should still clean up a bit
    ERRORclass("No cuts were activated for processing sample '%s'",sample->getPath().Data());
    sample->returnToken(tok);
    sample->returnToken(fileTok);
    return -1;
  }
  
  //this->cloneAlgorithms(n); //create clones for each channel
  
  
  if (! this->initializeClonedAlgorithms(runningSamples,tagKey)) {
    ERRORclass("Failed to initialize algorithms for processing sample '%s'",sample->getPath().Data());
    sample->returnToken(tok);
    sample->returnToken(fileTok);
    return -1;
  }
  
  if (tree && nEntries>0) { //only execute the actual event loop part if there is an event tree (we might have nothing but a MetaData tree in case of xAODs for example)
    const Long64_t nEventsPerPercent = ceil(nEntries/100.); 
    const Long64_t progressInterval = ceil(nEventsPerPercent*this->getTagDoubleDefault("progressInterval",0.));
    for (Long64_t i = 0; i < nEntries; ++i) {
      DEBUGclass(" visiting entry %d/%d",i,nEntries);
      tree->GetEntry(i);
      this->executeClonedAlgorithms(); //run pre-event part of algorithms
      for(size_t j=0; j<n; ++j){
        DEBUGclass("  friend %d",j);
        TRY(
          cuts[j]->analyse(1., useMCweights[j]);
        ,TString::Format("An error occured while evaluating entry %d of sample '%s'.",i,(cuts[j]->getSample()!=0?cuts[j]->getSample()->getPath().Data():"<undefined>"))
        )
        DEBUGclass("  friend %d done",j);
      }
      this->cleanupClonedAlgorithms(); //execute post-event part of algorithms
      
      DEBUGclass(" done visiting entry %d/%d",i,nEntries);
      if ( progressInterval>0 && (i%progressInterval == 0) ) {
        this->updateLine(fStatusLine,message,visitPROGRESS,progressInterval<=0,((double)i)/std::max(nEntries,(Long64_t)1)  );
      }
    }
  }
  //finalize algorithms
  if (!this->finalizeClonedAlgorithms()) {
    ERRORclass("failed to finalize algorithms after processing sample '%s'",sample->getPath().Data());
    sample->returnToken(tok);
    sample->returnToken(fileTok);
    return -1;
  }
  // finalize the cuts
  for(size_t j=0; j<n; ++j){
    TQSampleFolder* sf = cuts[j]->getSample();
    bool fromCut = sf; //did we get 'sf' from the cut?
    if (!sf && runningSamples.size()>j) sf = runningSamples[j];
    if (!sf) {
      throw std::runtime_error("The internal logic of TQMultiChannelAnalysisSampleVisitor seems to be broken. Please inform the CAFCore developers and try to provide a test case reproducing this error message!");
      return -1;
    }
    this->stamp(sf);
    if (fromCut) cuts[j]->finalize();
  }

  
  sample->returnToken(tok);
  sample->returnToken(fileTok);
 
  DEBUGclass("finished analyzing sample '%s'",sample->GetName());

  if (sample->getNTreeTokens() > 0) {
    std::cout << std::endl;
    message="sample left with # tree tokens > 0";
    sample->printTreeTokens();
    std::cout << std::endl;
    std::cout << "Message: " <<message<<std::endl;
  }
 
  return nEntries;
}

void TQMultiChannelAnalysisSampleVisitor::useObservableSets(bool useSets){
  // decide whether this visitor should use separate observable sets
  this->fUseObservableSets = useSets;
}

void TQMultiChannelAnalysisSampleVisitor::updateFriends(TQSampleFolder* sf) {
  if (!sf) return;
  if (sf->countFriends() > 0) return;
  sf->findFriends();
  return;
}


