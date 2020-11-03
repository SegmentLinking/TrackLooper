#include "QFramework/TQCutflowAnalysisJob.h"
#include "QFramework/TQCompiledCut.h"
#include "QFramework/TQAnalysisJob.h"
#include "QFramework/TQCounter.h"
#include "QFramework/TQSample.h"

// #define _DEBUG_
#include "QFramework/TQLibrary.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

#include <stdexcept>

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQCutflowAnalysisJob:
//
// This analysis job allows to book TQCounters at cuts. It works
// similar to the TQHistoMakerAnalysisJob, but as TQCounters don't
// need any configuration options, no configuration strings are
// required.
//
////////////////////////////////////////////////////////////////////////////////////////////////


ClassImp(TQCutflowAnalysisJob)


//______________________________________________________________________________________________

TQCutflowAnalysisJob::TQCutflowAnalysisJob() : 
TQAnalysisJob("TQCutflowAnalysisJob")
{
	// default constructor
}


//______________________________________________________________________________________________

TQCutflowAnalysisJob::TQCutflowAnalysisJob(const TString& name_) : 
  TQAnalysisJob(name_)
{
	// constructor with name
}


//______________________________________________________________________________________________

bool TQCutflowAnalysisJob::initializeSelf() {
  // initialize this analysis job 
  DEBUGclass("initializing analysis job '%s'",this->GetName());

  // we need a parent cut defining the name of the counter 
  if (!this->getCut()) {
    throw std::runtime_error("this analysis job has no cut assigned");
  }

  if ( !this->fCounter ) {
    this->poolAt = this->fSample;
    
    // create a new counter 
    TQCut* c = this->getCut();
    fCounter = new TQCounter(c->GetName(),c->GetTitle());
  }
  DEBUGclass("finished initializing cutflow analysis job");
  return true;
}


//______________________________________________________________________________________________

bool TQCutflowAnalysisJob::finalizeSelf() {
  // get the cutflow folder 
  DEBUGclass("attempting to create .cutflow folder in sample '%s'",this->fSample->getPath().Data());
	
	if(this->poolAt == this->fSample)
		if(!this->finalizeCounter())
			return false;
	
  /* finalize this analysis job */
  return true;

}


//__________________________________________________________________________________|___________

bool TQCutflowAnalysisJob::initializeSampleFolder(TQSampleFolder* sf){
  // initialize this job on a sample folder (taking care of pooling)
  bool pool = false;
  sf->getTagBool(".aj.pool.counters",pool);
  // std::cout << std::endl << "initialize samplefolder called on " << sf->GetName() << " pool=" << pool << ", fHistograms=" << fHistograms << std::endl << std::endl;
  if(pool && !this->fCounter){
    /* create a new counter */
    TQCut* c = this->getCut();
    fCounter = new TQCounter(c->GetName(),c->GetTitle());
    this->poolAt = sf;
  }

  return true;
}

//______________________________________________________________________________________________

bool TQCutflowAnalysisJob::execute(double weight) {
	// count this event: add its weight to the counter 
  fCounter->add(weight*fSample->getNormalisation());

  return true;

}

//______________________________________________________________________________________________

bool TQCutflowAnalysisJob::finalizeCounter(){
	// finalize the counter
	TQFolder * cfFolder = this->poolAt->getFolder(".cutflow+");
	
	// stop if we failed to get the folder 
	if (!cfFolder) { return false; }

	TObject* c = cfFolder->FindObject(fCounter->GetName());
	if(c) cfFolder->Remove(c);

	// add the counter 
	cfFolder->addObject(fCounter);
  
	// remove the pointer to the counter 
	this->fCounter = 0;
	this->poolAt = NULL;
	return true;
}


//__________________________________________________________________________________|___________

bool TQCutflowAnalysisJob::finalizeSampleFolder(TQSampleFolder* sf){
	// finalize the sample folder
  if (!sf) { return false; }
  // finalize this job on a sample folder (taking care of pooling)
	if(sf == this->poolAt)
		return this->finalizeCounter();
	
  return true;
}

//______________________________________________________________________________________________

TQCutflowAnalysisJob::~TQCutflowAnalysisJob() {
	// default destructor
  if (fCounter) { delete fCounter; }
}


