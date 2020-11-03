#include "QFramework/TQEventlistAnalysisJob.h"
#include "QFramework/TQStringUtils.h"
#include "QFramework/TQCut.h"
#include "QFramework/TQAnalysisJob.h"
#include "QFramework/TQCounter.h"
#include "QFramework/TQSample.h"
#include "QFramework/TQTable.h"
#include "TIterator.h"
#include "TList.h"

// #define _DEBUG_
#include "QFramework/TQLibrary.h"

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <cstdlib>

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQEventlistAnalysisJob:
//
// For each event list scheduled, a new TQEventlistAnalysisJob will be
// created. The syntax for these files is similar to the one used for
// histogramming with the TQHistoMakerAnalysisJob.
//
// In this example, an event list called "evtlist" is defined with four columns,
// containing run and event numbers, invariant dijet mass (in GeV) and
// pseudorapidity gap between the jets.
//
//    evtlist: Run << RunNumber, Event << EventNumber, \ensuremath{m_{jj}} << Mjj/1000. , \ensuremath{\Delta y_{jj}} << DYjj; 
//
//    @Cut_2jetincl: evtlist; The event list "evtlist" is appended to the cut named "Cut_2jetincl". 
//
// Caution: The TGraphs can become extremely large for large numbers
// of events!
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQEventlistAnalysisJob)

TQMessageStream TQEventlistAnalysisJob::f_ErrMsg(new std::stringstream());


//______________________________________________________________________________________________

TQEventlistAnalysisJob::TQEventlistAnalysisJob() : TQAnalysisJob(),
                                                   fEventlist(0),
                                                   fEventIndex(0),
                                                   showWeightColumn(false),
                                                   f_Verbose(false)
{
	// default constructor
}


//______________________________________________________________________________________________

TQEventlistAnalysisJob::TQEventlistAnalysisJob(const TString& name_) : TQAnalysisJob(name_),
                                                                fEventlist(0),
                                                                fEventIndex(0),
                                                                showWeightColumn(false),
                                                                f_Verbose(false)
{
	// default constructor with name
}

//______________________________________________________________________________________________

TQEventlistAnalysisJob * TQEventlistAnalysisJob::getClone(){
  // clone this TQEventlistAnalysisJob
  TQEventlistAnalysisJob* newInstance = new TQEventlistAnalysisJob(this->GetName());
  newInstance->showWeightColumn = this->showWeightColumn;
  for(size_t i=0; i<this->fExpressions.size(); ++i){
    newInstance->fExpressions.push_back(this->fExpressions[i]);
    newInstance->fTitles.push_back(this->fTitles[i]);
  }
  return newInstance;
}

//______________________________________________________________________________________________

void TQEventlistAnalysisJob::reset() {
	// reset this analysis job
  this->fEventIndex = 0;

  TQAnalysisJob::reset();

  this->fExpressions.clear();
  this->fTitles.clear();
  if(fEventlist) delete this->fEventlist;
}


//______________________________________________________________________________________________

void TQEventlistAnalysisJob::addColumn(const TString& expression, const TString& label) {
  // set up tree observables corresponding to expressions
  this->fExpressions.push_back(expression);
  if(!label.IsNull()) this->fTitles.push_back(label);
  else this->fTitles.push_back(expression);
}

//______________________________________________________________________________________________

int TQEventlistAnalysisJob::nColumns() const {
	// get the number of columsn currently booked
  return this->fExpressions.size();
}

//______________________________________________________________________________________________

void TQEventlistAnalysisJob::setWeightColumn(bool weight){
	// toggle the use of a weight column
  this->showWeightColumn = weight;
}

//__________________________________________________________________________________|___________

TObjArray * TQEventlistAnalysisJob::getBranchNames() {
  // return all observable expressions (containing branch names)
  if(!this->fSample){
    throw std::runtime_error("cannot retrieve branches on uninitialized object!");
  }

  TObjArray * bNames = new TObjArray();
  for(size_t i=0; i<this->fObservables.size(); i++){
    TCollection* c = this->fObservables[i]->getBranchNames();
    bNames->AddAll(c);
    delete c;
  }
 
  return bNames;
}


//______________________________________________________________________________________________

bool TQEventlistAnalysisJob::initializeSelf() {
  // initialize this analysis job 
  DEBUGclass("initializing analysis job '%s'",this->GetName());

  // we need a parent cut defining the name of the event list 
  if (!this->getCut()) return false;

  // create a new event list 
  DEBUGclass("allocating table");
  fEventlist = new TQTable(this->GetName());
  fEventlist->clearVlines();
  fEventlist->expand(1000,this->fExpressions.size()+this->showWeightColumn);

  // initialize TQObservables 
  DEBUGclass("preparing observables");
  for (size_t i = 0; i < this->fExpressions.size(); ++i) {
    fEventlist->setEntry(0,i,this->fTitles[i]);
    TQObservable* obs = TQObservable::getObservable(this->fExpressions[i],this->fSample);
    if (!obs) {
      ERRORclass("Failed to obtain observable with expression '%s' for sample '%s' in analysis job '%s'",this->fExpressions[i].Data(),this->fSample->getPath().Data(),this->GetName());
      return false;
    }
    if (!obs->initialize(this->fSample)) {
      ERRORclass("Failed to initialize observable created from expression '%s' for sample '%s' in analysis job '%s'" ,this->fExpressions[i].Data(),this->fSample->getPath().Data(),this->GetName());
      return false;    
    }
    this->fObservables.push_back(obs);
  }
  if(this->showWeightColumn){
    DEBUGclass("handling weight column");
    fEventlist->setEntry(0,this->fObservables.size(),"weight");
  }
 
  // we believe everything is fine 
  return true;
}


//______________________________________________________________________________________________

bool TQEventlistAnalysisJob::finalizeSelf() {
  // finalize this analysis job
  for (unsigned int i = 0; i < this->fObservables.size(); i++) {
    this->fObservables[i]->finalize();
  }
  this->fObservables.clear();

	// clear unused space from the table
  fEventlist->shrink();
 
  // get the cutflow folder 
  TQFolder * evlFolder = this->fSample->getFolder(TString::Format(".eventlists/%s/+",this->getCut()->GetName()));
 
  // stop if we failed to get the folder 
  if (!evlFolder)
    return false;
 
  // remove existing list 
  evlFolder->Remove(evlFolder->FindObject(fEventlist->GetName()));
  // add the list 
  evlFolder->Add(fEventlist);
  fEventlist = NULL;

  return true;
}


//______________________________________________________________________________________________

bool TQEventlistAnalysisJob::execute(double weight) {
	// execute this analysis job on an event and add it to the list
  this->fEventIndex++;
  if(fEventlist->getNrows() <= this->fEventIndex){
    fEventlist->expand(2*this->fEventIndex,this->fObservables.size()+this->showWeightColumn);
  }
  for (size_t i = 0; i < this->fObservables.size(); ++i) {
    double val = 0;
    TRY(
    val = this->fObservables[i]->getValue();
    ,TString::Format("Failed to evaluate observable '%s' at cut '%s'.", fObservables[i]->GetName(), this->getCut()->GetName())
    )
    DEBUGclass("setting entry %d/%d from '%s' to %g",this->fEventIndex,i,this->fObservables[i]->getActiveExpression().Data(),val);
    this->fEventlist->setEntryValue(this->fEventIndex,i,val);
  }
  if(this->showWeightColumn){
    DEBUGclass("setting (weight) entry %d/%d to %g",this->fEventIndex,this->fObservables.size(),weight);
    this->fEventlist->setEntryValue(this->fEventIndex,this->fObservables.size(),weight);
  }
  return true;
}


//__________________________________________________________________________________|___________

void TQEventlistAnalysisJob::setErrorMessage(TString message) {
	// send an error message
  f_ErrMsg.sendClassFunctionMessage(TQMessageStream::INFO,this->Class(),"<anonymous>",message);
	// print the error message if in verbose mode 
  if (f_Verbose > 0) INFOclass(message);
}

//__________________________________________________________________________________|___________

void TQEventlistAnalysisJob::clearMessages(){
  // clear the error messages
  f_ErrMsg.clearMessages();
}

//__________________________________________________________________________________|___________

TString TQEventlistAnalysisJob::getErrorMessage() {
  // Return the latest error message
  return f_ErrMsg.getMessages();
}

//______________________________________________________________________________________________

TQEventlistAnalysisJob::~TQEventlistAnalysisJob() {
	// standard destructor
  this->reset();
}

//__________________________________________________________________________________|___________

int TQEventlistAnalysisJob::importJobsFromTextFiles(const TString& files, TQCut* basecut, const TString& channelFilter, bool verbose){
  // open a list of files (comma-separated), parse all eventlist definitions inside
  // for each assigment encountered matching the channelfilter (or with no channel),
  // create an eventlist job for each eventlist and add it to the basecut1
  std::vector <TString> filenames = TQStringUtils::split(files,",");
  return TQEventlistAnalysisJob::importJobsFromTextFiles(filenames,basecut,channelFilter,verbose);
}

//__________________________________________________________________________________|___________

int TQEventlistAnalysisJob::importJobsFromTextFiles(const TString& files, TQCut* basecut, TQTaggable* aliases, const TString& channelFilter, bool verbose){
  // open a list of files (comma-separated), parse all eventlist definitions inside
  // for each assigment encountered matching the channelfilter (or with no channel),
  // create an eventlist job for each eventlist and add it to the basecut1
  std::vector <TString> filenames = TQStringUtils::split(files,",");
  return TQEventlistAnalysisJob::importJobsFromTextFiles(filenames,basecut,aliases,channelFilter,verbose);
}

//__________________________________________________________________________________|___________

int TQEventlistAnalysisJob::importJobsFromTextFiles(const std::vector<TString>& filenames, TQCut* basecut, const TString& channelFilter, bool verbose){
  // open a list of files (std::vector), parse all eventlist definitions inside
  // for each assigment encountered matching the channelfilter (or with no channel),
  // create an eventlist job for each eventlist and add it to the basecut
  return TQEventlistAnalysisJob::importJobsFromTextFiles(filenames, basecut, NULL, channelFilter, verbose);
}

//__________________________________________________________________________________|___________

int TQEventlistAnalysisJob::importJobsFromTextFiles(const std::vector<TString>& filenames, TQCut* basecut, TQTaggable* aliases, const TString& channelFilter, bool verbose){
  // open a list of files (std::vector), parse all eventlist definitions inside
  // for each assigment encountered matching the channelfilter (or with no channel),
  // create an eventlist job for each eventlist and add it to the basecut
  if(filenames.size() < 1){
    ERRORfunc("importing no eventlists from empty files list!");
    return -1;
  }
  std::map<TString,std::vector<TString> > eventlistDefinitions;
  std::vector<TString> assignments;
  TString buffer;
  for(size_t i=0; i<filenames.size(); i++){
    std::vector<TString>* lines = TQStringUtils::readFileLines(filenames[i]);
    if(!lines){
      if(verbose) ERRORfunc("unable to open file '%s'",filenames[i].Data());
      continue;
    }
    for(size_t j=0; j<lines->size(); j++){
      TString line(lines->at(j));
      TQStringUtils::readBlanks(line);
      if(line.IsNull()) continue;
      if(!line.BeginsWith("@")){
        TString name, def;
        if(!TQStringUtils::readUpTo(line,name,":")){
          if(verbose) ERRORfunc("unable to parse eventlist definition '%s'",line.Data());
          continue;
        }
        TQStringUtils::removeLeading(line,": ");
        TQStringUtils::readUpTo(line,def,";");
        DEBUGclass("found definition: '%s', assigning as '%s'",def.Data(),name.Data());
        eventlistDefinitions[TQStringUtils::trim(name)] = TQStringUtils::split(def,",","{","}");
      } else if(TQStringUtils::removeLeading(line,"@") == 1){ 
        DEBUGclass("found assignment: '%s'",line.Data());
        assignments.push_back(line);
      } else {
        if(verbose) WARNfunc("encountered unknown token: '%s'",line.Data());
      }
    }
    delete lines;
  }

  int retval = 0;
  for(size_t i=0; i<assignments.size(); i++){
    TString assignment = assignments[i];
    DEBUGclass("looking at assignment '%s'",assignment.Data());
    TString channel;
    if(TQStringUtils::readBlock(assignment,channel) && !channel.IsNull() && !TQStringUtils::matches(channel,channelFilter)) continue;
    TString cuts,eventlists;
    TQStringUtils::readUpTo(assignment,cuts,":");
    TQStringUtils::readToken(assignment,buffer," :");
    TQStringUtils::readUpTo(assignment,eventlists,";");
    TQStringUtils::readToken(assignment,buffer,"; ");
    DEBUGclass("eventlists: '%s'",eventlists.Data());
    DEBUGclass("cuts: '%s'",cuts.Data());
    DEBUGclass("spare symbols: '%s'",buffer.Data());
    std::vector<TString> vEvtlists = TQStringUtils::split(eventlists,",");
    if(vEvtlists.size() < 1){
      if(verbose) ERRORfunc("no eventlists listed in assignment '%s'",assignments[i].Data());
      continue;
    }
    for(size_t j=0; j<vEvtlists.size(); j++){
      TString evtlist(vEvtlists[j]);
      bool showWeight = (TQStringUtils::removeTrailing(evtlist,"+") > 0);
      std::vector<TString> def = eventlistDefinitions[evtlist];
      if(def.empty()){
        if(verbose) ERRORfunc("unable to find eventlist definition for name '%s', skipping",vEvtlists[j].Data());
        continue;
      }
      TQEventlistAnalysisJob* job = new TQEventlistAnalysisJob(evtlist);
      job->setWeightColumn(showWeight);
      for(size_t i=0; i<def.size(); i++){
        TString col(aliases ? aliases->replaceInTextRecursive(def[i]) : def[i]);
        TString title;
        TQStringUtils::readUpToText(col,title,"<<");
        TQStringUtils::removeLeading(col," <");
        if(!title.IsNull() && ! col.IsNull()){
          job->addColumn(col,title);
        }
      }
      if(job->nColumns() < 1){
        DEBUGclass("error booking eventlist for '%s', function says '%s'",f_ErrMsg.getMessages().Data());
      } else {
        basecut->addAnalysisJob(job,cuts);
        retval += 1;
      }
      delete job;
    }
  } 

  DEBUGclass("end of function call, found %d event lists",retval);
  return retval;
}
