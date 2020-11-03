#include "QFramework/TQSampleRevisitor.h"
#include "QFramework/TQSample.h"

// #define _DEBUG_
#include "QFramework/TQLibrary.h"

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQSampleRevisitor:
//
// Repeat what a previos sample visitor did and print its output.
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQSampleRevisitor)


int TQSampleRevisitor::visitSample(TQSample * sample, TString& message){
  // visit a sample, reading its trace
  return this->readTrace(sample,".sv.visit",message);
}

int TQSampleRevisitor::visitFolder(TQSampleFolder * sample, TString& message){
  // visit a folder, reading its trace
  return this->readTrace(sample,".sv.visit",message);
}

int TQSampleRevisitor::revisitSample(TQSample * sample, TString& message){
  // visit a sample, reading its trace
  return this->readTrace(sample,".sv.revisit",message);
}

int TQSampleRevisitor::revisitFolder(TQSampleFolder * sample, TString& message){
  // visit a folder, reading its trace
  return this->readTrace(sample,".sv.revisit",message);
}



TQSampleRevisitor::TQSampleRevisitor() {
  // default constructor
}

TQSampleRevisitor::TQSampleRevisitor(const char* name) {
  // constructor with name argument
  this->setVisitTraceID(name);
}

TQSampleRevisitor::~TQSampleRevisitor(){
  // default destructor
}

int TQSampleRevisitor::readTrace(TQFolder* f, const TString& prefix, TString& message ){
  // extract the trace information
  TString id(this->getVisitTraceIDConst());
  id.Prepend(".");
  id.Append(".");
  id.Prepend(prefix);
  int status;
  f->getTagInteger(id+"statusID",status);
  f->getTagString(id+"message",message);
  return status;
}
