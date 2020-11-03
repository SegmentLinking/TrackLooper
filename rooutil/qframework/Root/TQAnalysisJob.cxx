#include "QFramework/TQAnalysisJob.h"
#include "QFramework/TQCompiledCut.h"
#include "QFramework/TQIterator.h"
#include "QFramework/TQSample.h"
#include "QFramework/TQLibrary.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

#include <stdexcept>

using std::cout;
using std::endl;

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQAnalysisJob:
//
// The TQAnalysisJob class is the base class for every implementation of an analysis running
// on a set of samples. The idea of this class is to have a common class for encapsulating 
// analysis implementations. The most important methods for the user it introduces are:
//
// - initialize(...) called before a new sample is analyzed
// - execute(...) called for every event in the sample
// - finalize() called after the last event in the sample was analyzed
//
// In this sense, it is similar to a TSelector (Begin(...) <-> initialize(...), Process(...)
// <-> execute(...), Terminate(...) <-> finalize()). One important difference is, that you
// don't need to do your event selection in the implementation of this class, but you rather
// attach an anaylsis job to an instance of TQCompiledCut, which is responsible for the event
// selection on a more general level. You can attach more than one analysis job to a cut and
// you can attach one analysis job to more than one cuts (the analysis job is then cloned for
// each cut it is attached to).
//
// In your individual implementation of a subclass of this class you may access after init-
// ialization the TQSample or its TTree by using the protected member fields fTree and fSample
// (please take care not to change these pointers).
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQAnalysisJob)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

//__________________________________________________________________________________|___________

TQAnalysisJob::TQAnalysisJob() : TNamed("TQAnalysisJob", "") {
  // Default constructor of TQAnalysisJob class: a new instance of TQAnalysisJob
  // is created and reset. The new instance will be named "TQAnalysisJob".
  reset();
}


//__________________________________________________________________________________|___________

TQAnalysisJob::TQAnalysisJob(const TString& name_) : TNamed(name_.Data(), "") {
  // Constructor of TQAnalysisJob class: a new instance of TQFolder is created and
  // reset. The new instance will be named as specified by the parameter "name_"

  reset();
}


//__________________________________________________________________________________|___________

void TQAnalysisJob::reset() {
  // Reset this analysis job (this method may be overriden by a specific implemen-
  // tation of an analysis job). This method is called after an analysis job was
  // cloned
  fCut = 0;
  fSample = 0;
}

//__________________________________________________________________________________|___________

void TQAnalysisJob::copyTransientMembersFrom(TQAnalysisJob* other) {
  // copy the transient data members from another analysis job
  this->fCut = other->fCut;
}


//__________________________________________________________________________________|___________

TQAnalysisJob * TQAnalysisJob::getClone() {
  // Clone and reset (by calling reset()) this analysis job. This method is called
  // when an analysis job is attached to a cut (instance of TQCompiledCut). It
  // makes sure that an analysis job running at different cut stages is represented
  // by different instances of this class, even if the same instance was passed to
  // be attached to different cuts.

  /* clone this job */ 
  TQAnalysisJob * clone = (TQAnalysisJob*)this->Clone();

  /* reset the clone */
  clone->reset();

  clone->copyTransientMembersFrom(this);

  /* return the clone */
  return clone;

}


//__________________________________________________________________________________|___________

bool TQAnalysisJob::execute(double weight) {
  // Execute this analysis job on one event (this method is to be overwritten by a
  // specific implementation of an analysis job). This method is called for every
  // event of the sample to be analyzed (please note: as in most of the cases an
  // analysis job will be attached to a certain cut, this method will only be
  // called for events passing its selection criteria). You may indicate success
  // or failure by returning true or false respectively

  return true;
}


//__________________________________________________________________________________|___________

TString TQAnalysisJob::getDescription() {
  // Return a short description of this analysis job

  return "an analysis job template doing nothing";
}


//__________________________________________________________________________________|___________

void TQAnalysisJob::setCut(TQCompiledCut * cut_) {
  // Set the cut instance this analysis job is attached to. Please note: an
  // analysis job has to be attached to exactly one cut instance

  fCut = cut_;
}


//__________________________________________________________________________________|___________

TQCompiledCut * TQAnalysisJob::getCut() {
  // Return the cut instance this analysis job is attached to
  return fCut;
}


//______________________________________________________________________________________________

void TQAnalysisJob::print(const TString& options){
	// print the name of this analysis job
  std::cout << this->GetName() << ": " << "no details available";
}

//______________________________________________________________________________________________

TObjArray * TQAnalysisJob::getBranchNames() {
  // Return all used TTree branches in a TObjArray* of TString*
  // The strings should include all used branch names,
  // but can also include formulas, number, etc.

  TObjArray * bNames = new TObjArray();

  // Example:

  //TString * names = new TString("(sqrt(2 * lepPt1 * MET * (1 - cos(MET_phi - lepPhi1))) / 1000.");
  //bNames -> Add((TObject*) names);

  // This would be ok to book the branches lepPt1, MET, MET_phi and lepPhi1

  return bNames;
}

//__________________________________________________________________________________|___________

bool TQAnalysisJob::initialize(TQSample * sample) {
  // Initialize this analysis job on a sample to be analyzed. Try to get a tree
  // token from the sample (by calling TQSample::getTreeToken()) and return true
  // if this initialization process succeeded, return false otherwise. Set the
  // protected variables fSample (TQSample*, pointing to the sample to be
  // analyzed), fTree (TTree*, pointing to the ROOT tree of this sample) to be
  // accessible in implementations of a subclass. For a specific implemenation of
  // an analysis job, this method has to be called before performing specific
  // initialization. In an inherited class implementation do for example
  //
  // bool initialize(TQSample * sample) {
  // if (TQAnalysisJob::initialize(sample)) {
  // /* make your specific initialization here */
  // return true;
  // } else {
  // return false;
  // }
  // }

  if (!sample) {
    return false;
  }
  
  this->fSample = sample;
  bool retval = this->initializeSelf();
  if(!retval){
    this->finalizeSelf();
    return false;
  }
  return true;
}

//__________________________________________________________________________________|___________

bool TQAnalysisJob::finalize() {
  // Finalize this analysis job. Return the tree token and reset the protected
  // variables fSample and fTree. For a specific implemenation of an analysis job,
  // this method has to be called after performing specific finalization. In an 
  // inherited class implementation do for example
  //
  // bool finalize() {
  // /* make your finalization here (e.g write your analysis results) */
  // return TQAnalysisJob::finalize();
  // }

  if(!this->fSample) return true;
  bool retval = this->finalizeSelf();
  
  this->fSample = 0;

  return retval;
}

//__________________________________________________________________________________|___________

bool TQAnalysisJob::initializeSampleFolder(TQSampleFolder* sf){
	// initiallize this algorithm on a sample folder
  return true;
}

//__________________________________________________________________________________|___________

bool TQAnalysisJob::finalizeSampleFolder(TQSampleFolder* sf){
	// finalize this algorithm on a sample folder
  return true;
}

//__________________________________________________________________________________|___________

int TQAnalysisJob::addToCuts(TList* cuts, const TString& cutname){
	// add this algorithm to one (or more) cuts
  if(!cuts || cutname.IsNull()) return 0;
  TQIterator itr(cuts);
  int retval = 0;
  while(itr.hasNext()){
    TQCompiledCut* cc = dynamic_cast<TQCompiledCut*>(itr.readNext());
    if(!cc) continue;
    cc->addAnalysisJob(this,cutname);
    retval++;
  }
  return retval;
}

//__________________________________________________________________________________|___________

TQAnalysisJob::~TQAnalysisJob() {
  // default destructor
}

#pragma GCC diagnostic pop
