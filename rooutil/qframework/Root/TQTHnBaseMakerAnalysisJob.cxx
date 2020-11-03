#include "QFramework/TQTHnBaseMakerAnalysisJob.h"
#include "QFramework/TQIterator.h"
#include "QFramework/TQTHnBaseUtils.h"
#include "QFramework/TQObservable.h"
#include "QFramework/TQStringUtils.h"
#include "QFramework/TQCompiledCut.h"
#include "QFramework/TQTaggable.h"
#include "TObjArray.h"
#include "TList.h"
#include "QFramework/TQSample.h"
#include "THnBase.h"
#include "THn.h"
#include "THnSparse.h"

// #define _DEBUG_
#include "QFramework/TQLibrary.h"

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <cstdlib>

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQTHnBaseMakerAnalysisJob:
//
// TQTHnBaseMakerAnalysisJob provides the possibility to use
// multidimensional histograms with its base class THnBase within the
// QFramework.  This class is highly derived from the
// TQHistoMakerAnalysisJob, where more information can be found.
//
////////////////////////////////////////////////////////////////////////////////////////////////

TQMessageStream TQTHnBaseMakerAnalysisJob::f_ErrMsg(new std::stringstream());
bool TQTHnBaseMakerAnalysisJob::g_useHistogramObservableNames(false);

ClassImp(TQTHnBaseMakerAnalysisJob)

//__________________________________________________________________________________|___________

TQTHnBaseMakerAnalysisJob::TQTHnBaseMakerAnalysisJob() : 
TQAnalysisJob("TQTHnBaseMakerAnalysisJob"),
  f_Verbose(0),
  poolAt(NULL)
{
  // standard constructor
}

//__________________________________________________________________________________|___________

TQTHnBaseMakerAnalysisJob::TQTHnBaseMakerAnalysisJob(TQTHnBaseMakerAnalysisJob* other) :
  TQAnalysisJob(other ? other->GetName() : "TQTHnBaseMakerAnalysisJob"),
  f_Verbose(other ? other->f_Verbose : 0),
  poolAt(other ? other->poolAt : NULL)
{
  // copy constructor
  for(size_t i=0; i<other->fHistogramTemplates.size(); ++i){
    this->fHistogramTemplates.push_back(TQTHnBaseUtils::copyHistogram(other->fHistogramTemplates[i]));
    this->fFillSynchronized.push_back(other->fFillSynchronized[i]);
    this->fFillRaw.push_back(other->fFillRaw[i]);
    this->fHistoTypes.push_back(other->fHistoTypes[i]);
    this->fExpressions.push_back(std::vector<TString>());
    for(size_t j=0; j<other->fExpressions[i].size(); ++j){
      this->fExpressions[i].push_back(other->fExpressions[i][j]);
    }
  }
}

//__________________________________________________________________________________|___________

void TQTHnBaseMakerAnalysisJob::setVerbose(int verbose) {
  // set verbosity
  f_Verbose = verbose;
}


//__________________________________________________________________________________|___________

int TQTHnBaseMakerAnalysisJob::getVerbose() {
  // retrieve verbosity
  return f_Verbose;
}

//__________________________________________________________________________________|___________

void TQTHnBaseMakerAnalysisJob::setErrorMessage(TString message) {
 
  /* update the error message */
  f_ErrMsg.sendClassFunctionMessage(TQMessageStream::INFO,this->Class(),"<anonymous>",message);
 
  /* print the error message if in verbose mode */
  if (f_Verbose > 0) INFOclass(message);
}

//__________________________________________________________________________________|___________

void TQTHnBaseMakerAnalysisJob::clearMessages(){
  // clear the error messages
  f_ErrMsg.clearMessages();
}

//__________________________________________________________________________________|___________

TString TQTHnBaseMakerAnalysisJob::getErrorMessage() {
  // Return the latest error message
  return f_ErrMsg.getMessages();
}


//__________________________________________________________________________________|___________

const TString& TQTHnBaseMakerAnalysisJob::getValidNameCharacters() {
  // retrieve a string with valid name characters
  return TQFolder::getValidNameCharacters();
}


//__________________________________________________________________________________|___________

void TQTHnBaseMakerAnalysisJob::cancelHistogram(const TString& name) {
  // cancel the histgogram with the given name
  for(size_t i=fHistogramTemplates.size(); i >0 ; i--){
    if(fHistogramTemplates.at(i)->GetName() == name){
      fHistogramTemplates.erase(fHistogramTemplates.begin()+i-1);
      fFillSynchronized.erase(fFillSynchronized.begin()+i-1);
      fFillRaw.erase(fFillRaw.begin()+i-1);
      fExpressions.at(i-1).clear();
      fExpressions.erase(fExpressions.begin()+i-1);
      fHistoTypes.erase(fHistoTypes.begin()+i-1);
      return;
    }
  }
}

//__________________________________________________________________________________|___________

namespace {
  void setupAxis(TAxis* axis, const TString& title, const TString& expr, const std::vector<TString>& labels){
    axis->SetTitle(title);
    axis->SetName(expr);
    for(size_t i=0; i<labels.size(); ++i){
      axis->SetBinLabel(i+1,labels[i]);
    }
  }
}
//__________________________________________________________________________________|___________

bool TQTHnBaseMakerAnalysisJob::bookHistogram(TString definition, TQTaggable* aliases) {
  DEBUGclass("entering function - booking histogram '%s'",definition.Data());

  if(definition.IsNull()){
    this->f_ErrMsg.sendClassFunctionMessage(TQMessageStream::INFO,this->Class(),__FUNCTION__,"obtained empty histogram definition");
    return false;
  }
  // histogram definition
  TString histoDef;
  TQStringUtils::readUpTo(definition, histoDef, "<", "()[]{}", "''\"\"");

  // create histogram template from definition
  TString msg;
  DEBUGclass("creating histogram '%s'",histoDef.Data());

  if(histoDef.IsNull()){
    this->f_ErrMsg.sendClassFunctionMessage(TQMessageStream::INFO,this->Class(),__FUNCTION__,"histogram constructor is empty, remainder is '%s'",definition.Data());
    return false;
  }

  THnBase * histo = TQTHnBaseUtils::createHistogram(histoDef, msg);
  DEBUGclass(histo ? "success" : "failure");

  if (!histo) {
    this->f_ErrMsg.sendClassFunctionMessage(TQMessageStream::ERROR,this->Class(),__FUNCTION__,msg);
    return false;
  }

  // invalid histogram name?
  if (!TQFolder::isValidName(histo->GetName())) {
    this->f_ErrMsg.sendClassFunctionMessage(TQMessageStream::INFO,this->Class(),__FUNCTION__,"'%s' is an invalid histogram name", histo->GetName());
    delete histo;
    return false;
  }

  // read "<<" operator
  if (TQStringUtils::removeLeading(definition, "<") != 2) {
    this->f_ErrMsg.sendClassFunctionMessage(TQMessageStream::INFO,this->Class(),__FUNCTION__,"Operator '<<' expected after histogram definition");
    delete histo;
    return false;
  }

  //split off a possile option block
  std::vector<TString> settingTokens = TQStringUtils::split(definition, "<<", "([{'\"", ")]}'\"");
  if (settingTokens.size()<1) {
    this->f_ErrMsg.sendClassFunctionMessage(TQMessageStream::INFO,this->Class(),__FUNCTION__,"cannot parse definition block '%s'",definition.Data());
    delete histo; 
    return false;
  }
  definition = settingTokens[0];
  TQTaggable options;
  if (settingTokens.size()>1) {
    TString optionBlock;
    TQStringUtils::readBlanksAndNewlines(settingTokens[1]);
    if (!TQStringUtils::readBlock(settingTokens[1], optionBlock, "()[]{}", "''\"\"")) {
      this->f_ErrMsg.sendClassFunctionMessage(TQMessageStream::INFO,this->Class(),__FUNCTION__,"Failed to parse histogram option block '%s'", settingTokens[1].Data());
      delete histo;
      return false;
    }
    options.importTags(optionBlock);
  }
  
  // read expression block
  TString expressionBlock;
  TQStringUtils::readBlanksAndNewlines(definition);
  if (!TQStringUtils::readBlock(definition, expressionBlock, "()[]{}", "''\"\"")) {
    this->f_ErrMsg.sendClassFunctionMessage(TQMessageStream::INFO,this->Class(),__FUNCTION__,"Missing expression block after '<<' operator");
    delete histo;
    return false;
  }

  // tokenize expression block (one token per histogram dimension)
  TList * expressionTokens = TQStringUtils::tokenize(expressionBlock, ",", true, "()[]{}", "''\"\"");
  if(expressionTokens->GetEntries() < 1){
    this->f_ErrMsg.sendClassFunctionMessage(TQMessageStream::INFO,this->Class(),__FUNCTION__,"cannot parse expression block '%s'",expressionBlock.Data());
    delete histo;
    return false;
  }

  DEBUGclass("parsing expression block '%s', found %d entries",expressionBlock.Data(),expressionTokens->GetEntries());

  // read expression block tokens
  std::vector<TString> exprs;
  std::vector<TString> histnames;
  std::vector<TString> titles;
  std::vector<std::vector<TString> > labels ;
  TQIterator itr(expressionTokens);
  while (itr.hasNext()) {
    TString token(itr.readNext()->GetName());
    // read expression 
    TString expr;
    TString histname;
    int nColon = 0;
    while(true){
      TQStringUtils::readUpTo(token, histname, "\\:", "()[]{}", "''\"\"");
      nColon = TQStringUtils::countLeading(token, ":");
      if (nColon == 1) {
        if (TQStringUtils::removeLeading(token,"\\:")) {
          TQStringUtils::readUpTo(token, expr,"\\:", "()[]{}", "''\"\"");
        }
      }
      nColon = TQStringUtils::countLeading(token, ":");
      if (nColon == 1) {
				TQStringUtils::removeLeading(token,"\\:");
      }
      nColon = TQStringUtils::countLeading(token, ":");
      if (nColon == 0) {
        break;
      } else {
				this->f_ErrMsg.sendClassFunctionMessage(TQMessageStream::INFO,this->Class(),__FUNCTION__,"histogram definitions could not be read - check your input!");
        DEBUGclass("histogram definitions could not be read - check your input!");
        delete histo;
        return false;
      }
    }
    TQStringUtils::readBlanksAndNewlines(token);
    TQStringUtils::removeLeading(token,":");
    TQStringUtils::readBlanksAndNewlines(token);
    
    // use TQTaggable to read title (handling of e.g. quotes)
    TString buffer;
    TString title;
    std::vector<TString> binLabels;
    if(TQStringUtils::readBlock(token,buffer,"''","",false,false) > 0 || TQStringUtils::readBlock(token,buffer,"\"\"","",false,false) > 0){
      title = buffer;
      buffer.Clear();
      TQStringUtils::readBlanksAndNewlines(token);
      if(TQStringUtils::removeLeading(token,":") == 1){
        TQStringUtils::readBlanksAndNewlines(token);
        TQStringUtils::readBlock(token,buffer,"(){}[]","",false,false);
        binLabels = TQStringUtils::split(buffer,",");
        for(size_t i=0; i<binLabels.size(); ++i){
          TQStringUtils::removeLeading(binLabels[i]," ");
          TQStringUtils::removeTrailing(binLabels[i]," ");
          TQStringUtils::unquoteInPlace(binLabels[i]);
        }
      }
    } else {
      title = TQStringUtils::unquote(token);
    }
    
    // store expression and title
    const TString expression(aliases ? aliases->replaceInTextRecursive(expr) : expr);
    exprs.push_back(TQStringUtils::trim(expression));
    titles.push_back(TQStringUtils::trim(title));
    histnames.push_back(TQStringUtils::trim(histname));
    labels.push_back(binLabels);
    DEBUGclass("found expression and title: '%s' and '%s'",expr.Data(),title.Data());
  }

  // histogram properties
  TString name = histo->GetName();
  int dim = TQTHnBaseUtils::getDimension(histo);

  // check dimension of histogram and expression block
  if ( ( dim > 0 && dim != (int)exprs.size() ) || ( dim < 0 && (int)exprs.size() != abs(dim)+1) ) {
    // last ist the TProfile case
    this->f_ErrMsg.sendClassFunctionMessage(TQMessageStream::INFO,this->Class(),__FUNCTION__,"Dimensionality of histogram (%d) and expression block (%d) don't match", dim, (int)(exprs.size()));
    delete histo;
    DEBUGclass("Dimensionality of histogram (%d) and expression block (%d) don't match", dim, (int)(exprs.size()));
    return false;
  }

  // check name of histogram
  if (!TQStringUtils::isValidIdentifier(name, getValidNameCharacters())) {
    this->f_ErrMsg.sendClassFunctionMessage(TQMessageStream::INFO,this->Class(),__FUNCTION__,"Invalid histogram name '%s'", name.Data());
    DEBUGclass("Invalid histogram name '%s'", name.Data());
    delete histo;
    return false;
  }

  // stop if histogram with 'name' already has been booked
  bool exists = false;
  int i = 0;
  while (!exists && i < (int)(fHistogramTemplates.size()))
    exists = (name.CompareTo(fHistogramTemplates.at(i++)->GetName()) == 0);
  if (exists) {
    this->f_ErrMsg.sendClassFunctionMessage(TQMessageStream::INFO,this->Class(),__FUNCTION__,"Histogram with name '%s' has already been booked", name.Data());
    DEBUGclass("Histogram with name '%s' has already been booked", name.Data());
    delete histo;
    return false;
  }
 
  // set up tree observables corresponding to expressions
  for (int i = 0; i < (int)exprs.size(); i++) {
    setupAxis(histo->GetAxis(i), titles[i], histnames[i], labels[i]);
  }

  fExpressions.push_back(exprs);
  fHistoTypes.push_back(TQTHnBaseUtils::getDimension(histo));
  fHistogramTemplates.push_back(histo);
  //@tag: [fillSynchronized] This tag is read from an additional option block in the histogram definition, e.g. TH2(<histogram definition>) << (<variable definitions>) << (fillSynchronized = true) . This tag defaults to false and is only relevant if vector observables are used in multi dimensional histograms. By default all combinations of values from the different observables are filled. If this tag is set to 'true', however, all vector valued observables are required to have the same number of evaluations (same dimensionality). Only combinations of values with the same index in the respective vector observables are filled. If a vector observable is used in combination with a non-vector observable the latter one is evaluated the usual way for each entry of the vector observable.
  fFillSynchronized.push_back(options.getTagBoolDefault("fillSynchronized",false));
  fFillRaw.push_back(options.getTagBoolDefault("fillRaw",false));
  
  return true;
}


//__________________________________________________________________________________|___________

bool TQTHnBaseMakerAnalysisJob::execute(double weight) {
  DEBUGclass("filling histograms for event...");
  // execute this analysis job, filling all histograms
  int nEvals = 0;
  for (unsigned int i = 0; i < fHistograms.size(); ++i) {
#ifdef _DEBUG_
    if(this->fObservables.size() < i){
      throw std::runtime_error("insufficient size of observable vector!");
    }
#endif
    
    const int dim = fObservables[i].size();
    if (dim != TQTHnBaseUtils::getDimension(fHistograms[i])) {
      ERRORclass("Dimension of Observable does not agree with dimension of histogram named %s. No filling of histogram possible", this->fHistograms[i]->GetName());
    }
    TRY(
	std::vector<double> fillvector;
	for (unsigned int a = 0; a<fObservables[i].size(); a++) {
	  fillvector.push_back(fObservables[i][a]->getValue());
	}
	((THnSparse*)(fHistograms[i]))->Fill(fillvector.data(),
																			 this->fFillRaw[i]? 1. : weight*fSample->getNormalisation());
	,TString::Format("Failed to fill histogram '%s' using the observables '%s', '%s', '%s' at cut '%s'.", fHistograms[i]->GetName(), fObservables[i][0]->GetName(), fObservables[i][1]->GetName(), fObservables[i][2]->GetName(), this->getCut()->GetName())
	)
			
      }
	
  return true;
}

//__________________________________________________________________________________|___________

bool TQTHnBaseMakerAnalysisJob::initializeSelf() {
  // initialize this analysis job
  DEBUGclass("initializing analysis job '%s'",this->GetName());
	
  if(fHistograms.size() < 1){
    this->poolAt = this->fSample;
    DEBUGclass("initializing histograms");
    this->initializeHistograms();
  }
	
  bool success = true;
  /* initialize TQObservables */
  DEBUGclass("initializing observables");
  for (unsigned int i = 0; i < fExpressions.size(); ++i) {
    std::vector<TQObservable*> observables;
    for (unsigned int j = 0; j < fExpressions[i].size(); ++j) {
      TQObservable* obs = TQObservable::getObservable(fExpressions[i][j],this->fSample);
      if(obs && success){
        DEBUGclass("initializing...");
        if (!obs->initialize(this->fSample)) {
          ERRORclass("Failed to initialize observable created from expression '%s' for sample '%s' in TQHistomakerAnalysisJob '%s' for histogram named '%s'",this->fExpressions[i][j].Data(), this->fSample->getPath().Data(), this->GetName(), this->fHistograms[i]->GetName());
          success=false;
        }
        DEBUGclass("initialized observable '%s' of type '%s' with '%s'",
                   obs->getExpression().Data(),
                   obs->ClassName(),
                   obs->getActiveExpression().Data());
      }
      if(!obs){
        DEBUGclass("creating const observable");
        obs = TQObservable::getObservable("Const:nan",this->fSample);
        obs->initialize(this->fSample);
      }
      observables.push_back(obs);
    }
    this->fObservables.push_back(observables);
  }
  DEBUG("successfully initialized histogram job");
  return success;
}

//__________________________________________________________________________________|___________

void TQTHnBaseMakerAnalysisJob::initializeHistograms(){
  // create histograms from templates */
  DEBUGclass("Size of histogram template vector : %i", fHistogramTemplates.size());
  for (unsigned int i = 0; i < fHistogramTemplates.size(); i++) {
    
    /* copy/clone the template histogram */
    //TH1 * histo = (TH1*)(*fHistogramTemplates)[i]->Clone();
    THnBase * histo = TQTHnBaseUtils::copyHistogram((fHistogramTemplates)[i]);
    // std::cout << "initialized " << histo->GetName() << std::endl;
    // histo->SetDirectory(0);
    fHistograms.push_back(histo);
  }
}
 

//__________________________________________________________________________________|___________

bool TQTHnBaseMakerAnalysisJob::finalizeSelf() {
  // finalize TQObservables
  for (unsigned int i = 0; i < fObservables.size(); ++i) {
    for (unsigned int j = 0; j < fObservables[i].size(); ++j) {
      fObservables[i][j]->finalize();
    }
  }
  this->fObservables.clear();

  if(this->poolAt == this->fSample)
    if(!this->finalizeHistograms())
      return false;
  
  return true;
}

//__________________________________________________________________________________|___________

bool TQTHnBaseMakerAnalysisJob::finalizeHistograms(){
  // store the histograms in the sample folder hierarchy
  if (!getCut()) { return false; }

  /* get the histogram folder */
  TQFolder * folder = this->poolAt->getFolder(TString::Format(
                                                              ".histograms/%s+", getCut()->GetName()));
  if (!folder) { return false; }
  DEBUGclass("successfully created folder for cut %s", getCut()->GetName());

  /* scale and store histograms */
  DEBUGclass("length of histogram list is %i", fHistograms.size());
  for (unsigned int i = 0; i < fHistograms.size(); i++) {
    THnBase * histo = (fHistograms)[i];
    if (!histo){ DEBUGclass("Histogram is 0!"); };
    /* delete existing histogram */
    TObject *h = folder->FindObject(histo->GetName());
    if (h)
      {
        DEBUGclass("removing previous object %s", h->GetName());
        folder->Remove(h);
      }
    /* save the new histogram */
    DEBUGclass("saving histogram %s", histo->GetName());
    folder->Add(histo);
  }

  /* delete the list of histograms */
  this->fHistograms.clear();
  this->poolAt = NULL;

  return true;
}

//__________________________________________________________________________________|___________

bool TQTHnBaseMakerAnalysisJob::initializeSampleFolder(TQSampleFolder* sf){
  // initialize this job on a sample folder (taking care of pooling)
  bool pool = false;
  sf->getTagBool(".aj.pool.histograms",pool);
  // std::cout << std::endl << "initialize samplefolder called on " << sf->GetName() << " pool=" << pool << ", fHistograms=" << fHistograms << std::endl << std::endl;

  if(pool && (this->fHistograms.size() == 0)){
    this->initializeHistograms();
    this->poolAt = sf;
  }

  return true;
}

//__________________________________________________________________________________|___________

bool TQTHnBaseMakerAnalysisJob::finalizeSampleFolder(TQSampleFolder* sf){
  // finalize this job on a sample folder (taking care of pooling)
  if(sf == this->poolAt)
    return this->finalizeHistograms();
  return true;
}

//__________________________________________________________________________________|___________

TQTHnBaseMakerAnalysisJob::~TQTHnBaseMakerAnalysisJob() {
  // destructor
  for (unsigned int i = 0; i < fHistogramTemplates.size(); i++) {
    delete (fHistogramTemplates)[i]; }
}

//__________________________________________________________________________________|___________

TQAnalysisJob* TQTHnBaseMakerAnalysisJob::getClone(){
  // retrieve a clone of this job
  TQTHnBaseMakerAnalysisJob* newJob = new TQTHnBaseMakerAnalysisJob(this);
  return newJob;
}

//__________________________________________________________________________________|___________

void TQTHnBaseMakerAnalysisJob::reset() {
  // Reset this analysis job. This method is called after an analysis job was
  // cloned.

  // call the reset function of the parent class
  TQAnalysisJob::reset();
  // do class-specific stuff
  fHistograms.clear();
}

//__________________________________________________________________________________|___________

int TQTHnBaseMakerAnalysisJob::importJobsFromTextFiles(const TString& files, TQCompiledCut* basecut, const TString& channelFilter, bool verbose){
  // open a list of files (comma-separated), parse all histogram definitions inside
  // for each assigment encountered matching the channelfilter (or with no channel),
  // create a histogram job, fill it with all appropriate histograms and add it to the basecut
  std::vector <TString> filenames = TQStringUtils::split(files,",");
  return TQTHnBaseMakerAnalysisJob::importJobsFromTextFiles(filenames,basecut,channelFilter,verbose);
}

//__________________________________________________________________________________|___________

int TQTHnBaseMakerAnalysisJob::importJobsFromTextFiles(const TString& files, TQCompiledCut* basecut, TQTaggable* aliases, const TString& channelFilter, bool verbose){
  // open a list of files (comma-separated), parse all histogram definitions inside
  // for each assigment encountered matching the channelfilter (or with no channel),
  // create a histogram job, fill it with all appropriate histograms and add it to the basecut
  std::vector <TString> filenames = TQStringUtils::split(files,",");
  return TQTHnBaseMakerAnalysisJob::importJobsFromTextFiles(filenames,basecut,aliases,channelFilter,verbose);
}

//__________________________________________________________________________________|___________

int TQTHnBaseMakerAnalysisJob::importJobsFromTextFiles(const std::vector<TString>& filenames, TQCompiledCut* basecut, const TString& channelFilter, bool verbose){
  // open a list of files (std::vector), parse all histogram definitions inside
  // for each assigment encountered matching the channelfilter (or with no channel),
  // create a histogram job, fill it with all appropriate histograms and add it to the basecut
  return TQTHnBaseMakerAnalysisJob::importJobsFromTextFiles(filenames, basecut, NULL, channelFilter, verbose);
}

//__________________________________________________________________________________|___________

int TQTHnBaseMakerAnalysisJob::importJobsFromTextFiles(const std::vector<TString>& filenames, TQCompiledCut* basecut, TQTaggable* aliases, const TString& channelFilter, bool verbose){
  // open a list of files (std::vector), parse all histogram definitions inside
  // for each assigment encountered matching the channelfilter (or with no channel),
  // create a histogram job, fill it with all appropriate histograms and add it to the basecut
  if(filenames.size() < 1){
    ERRORfunc("importing no histograms from empty files list!");
    return -1;
  }
  std::map<TString,TString> histogramDefinitions;
  std::vector<TString> assignments;
  TString buffer;
  for(size_t i=0; i<filenames.size(); i++){
    std::vector<TString>* lines = TQStringUtils::readFileLines(filenames[i],2048);
    if(!lines){
      ERRORfunc("unable to open file '%s'",filenames[i].Data());
      continue;
    }
    for(size_t j=0; j<lines->size(); ++j){
      TString line(TQStringUtils::trim(lines->at(j)));
      DEBUGclass("looking at line '%s'",line.Data());
      if(line.IsNull()) continue;
      if(line.BeginsWith("T")){
        size_t namestart = TQStringUtils::findFirstOf(line,"'\"",0)+1;
        size_t nameend = TQStringUtils::findFirstOf(line,"'\"",namestart);
        if(namestart == 0 || namestart > (size_t)line.Length() || nameend > (size_t)line.Length() || nameend == namestart){
          ERRORfunc("unable to parse histogram definition '%s'",line.Data());
          continue;
        }
        TString name(TQStringUtils::trim(line(namestart,nameend-namestart),"\t ,"));
        DEBUGclass("found definition: '%s', assigning as '%s'",line.Data(),name.Data());
        histogramDefinitions[name] = line;
      } else if(TQStringUtils::removeLeading(line,"@") == 1){ 
        DEBUGclass("found assignment: '%s'",line.Data());
        assignments.push_back(line);
      } else {
        WARNfunc("encountered unknown token: '%s'",line.Data());
      }
    }
    delete lines;
  }
  if(verbose) VERBOSEfunc("going to create '%d' jobs",(int)(assignments.size()));
  int retval = 0;
  for(size_t i=0; i<assignments.size(); i++){
    TString assignment = assignments[i];
    DEBUGclass("looking at assignment '%s'",assignment.Data());
    TString channel;
    if(TQStringUtils::readBlock(assignment,channel) && !channel.IsNull() && !TQStringUtils::matches(channel,channelFilter)) continue;
    TString cuts,histograms;
    TQStringUtils::readUpTo(assignment,cuts,":");
    TQStringUtils::readToken(assignment,buffer," :");
    TQStringUtils::readUpTo(assignment,histograms,";");
    TQStringUtils::readToken(assignment,buffer,"; ");
    DEBUGclass("histograms: '%s'",histograms.Data());
    DEBUGclass("cuts: '%s'",cuts.Data());
    if(verbose) VERBOSEfunc("building job for cuts '%s'",cuts.Data());
    DEBUGclass("spare symbols: '%s'",buffer.Data());
    std::vector<TString> vHistos = TQStringUtils::split(histograms,",");
    if(vHistos.size() < 1){
      ERRORfunc("no histograms listed in assignment '%s'",assignments[i].Data());
      continue;
    }
    TQTHnBaseMakerAnalysisJob* job = new TQTHnBaseMakerAnalysisJob();
    for(size_t j=0; j<vHistos.size(); ++j){
      const TString def = histogramDefinitions[TQStringUtils::trim(vHistos[j],"\t ,")];
      if(def.IsNull()){
        ERRORfunc("unable to find histogram definition for name '%s', skipping",TQStringUtils::trim(vHistos[j],"\t ,").Data());
        continue;
      }
      bool ok = job->bookHistogram(def,aliases);
      if(ok){
        if(verbose) VERBOSEfunc("\tbooked histogram '%s'",def.Data());
      } else {
        retval += 1;
        if(verbose) std::cout << f_ErrMsg.getMessages() << std::endl;
        DEBUGclass("error booking histogram for '%s', function says '%s'",def.Data(),f_ErrMsg.getMessages().Data());
      }
    }
    // if(verbose) job->printBooking(cuts);
    basecut->addAnalysisJob(job,cuts);
    delete job;
  }   

  DEBUGclass("end of function call, encountered %d error messages",retval);
  return retval;
}
