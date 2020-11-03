#include "QFramework/TQSampleGroupingVisitor.h"
#include "QFramework/TQSampleDataReader.h"
#include "QFramework/TQSample.h"

// #define _DEBUG_
#include "QFramework/TQLibrary.h"

#include <algorithm>    // std::sort

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQSampleGroupingVisitor:
//
// Walk through a sample folder hierarchy to create groups of samples with equal size.
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQSampleGroupingVisitor)

int TQSampleGroupingVisitor::initialize(TQSampleFolder * sampleFolder, TString& /*message*/){
  this->fReader = new TQSampleDataReader(sampleFolder);
  this->fPaths.clear();
  this->fActiveItemName = "";
  this->fActiveItemCount = 0;
  return visitLISTONLY;
}

int TQSampleGroupingVisitor::finalize(){
  delete this->fReader;
  return visitLISTONLY;
}

int TQSampleGroupingVisitor::visitSample(TQSample * sample, TString& message){
  TQCounter* cnt = this->fReader->getCounter(sample->getPath(),this->fCounterName);
  if(!cnt) return visitFAILED;
  int count = cnt->getRawCounter();
  TString path = sample->getPathWildcarded();
  delete cnt;
  if(count + this->fActiveItemCount < this->getEventLimit()){
    if(!fActiveItemName.IsNull()){
      this->fActiveItemName.Append(",");
    } 
    this->fActiveItemName.Append(path);
    this->fActiveItemCount += count;
    message = TString::Format("count is %d",this->fActiveItemCount); 
    return visitSKIPPED;
  } else {
    if(!this->fActiveItemName.IsNull()){
      this->fPaths.insert(this->fActiveItemName.Data());
      this->fActiveItemName = path;
      this->fActiveItemCount = count;
    }
    return visitOK;
  }
}

int TQSampleGroupingVisitor::visitFolder(TQSampleFolder * sample, TString& message){
  TQCounter* cnt = this->fReader->getCounter(sample->getPath(),this->fCounterName);
  if(!cnt) return visitFAILED;
  int count = cnt->getRawCounter();
  TString path = sample->getPathWildcarded();
  delete cnt;
  if(count + this->fActiveItemCount < this->getEventLimit()){
    if(!fActiveItemName.IsNull()){
      this->fActiveItemName.Append(",");
    } 
    this->fActiveItemName.Append(path);
    this->fActiveItemCount += count;
    message = TString::Format("count is %d",this->fActiveItemCount); 
    return visitSKIPPED;
  } else {
    if(!this->fActiveItemName.IsNull()){
      this->fPaths.insert(this->fActiveItemName.Data());
      this->fActiveItemName.Clear();
      this->fActiveItemCount = 0;
    }
    return visitOK;
  }
}

int TQSampleGroupingVisitor::revisitFolder(TQSampleFolder * sampleFolder, TString& /*message*/){
  if(!this->fActiveItemName.IsNull() && sampleFolder->getTagIntegerDefault(".sv.statusID",0) == visitOK){
    this->fPaths.insert(this->fActiveItemName.Data());
    this->fActiveItemName.Clear();
    this->fActiveItemCount = 0;
  }
  return visitLISTONLY;
}

 
TQSampleGroupingVisitor::TQSampleGroupingVisitor(const char* counterName, int nEvents) :
  fCounterName(counterName),
  fEventLimit(nEvents)
{
  // constructor with name argument
}

TQSampleGroupingVisitor::~TQSampleGroupingVisitor(){
  // default constructor
}


void TQSampleGroupingVisitor::setCounterName(const TString& name){
  // set the name of the counter to be used
  this->fCounterName = name;
}

void TQSampleGroupingVisitor::setEventLimit(int nEvents){
  // set the limit in the number of events
  this->fEventLimit = nEvents;
}

TString TQSampleGroupingVisitor::getCounterName(){
  // retrieve the name of the counter to be used
  return this->fCounterName;
}

int TQSampleGroupingVisitor::getEventLimit(){
  // retrieve the limit in the number of events
  return this->fEventLimit;
}

std::vector<TString> TQSampleGroupingVisitor::getPaths(){
  // retrieve the list of paths collected
  std::vector<TString> retval;
  for(auto elem:this->fPaths){
    retval.push_back(elem.c_str());
  }
  std::sort(retval.begin(),retval.end());
  return retval;
}
