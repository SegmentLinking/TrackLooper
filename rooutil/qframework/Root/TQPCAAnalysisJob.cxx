#include "QFramework/TQPCAAnalysisJob.h"
#include "QFramework/TQCompiledCut.h"
#include "QFramework/TQAnalysisJob.h"
#include "QFramework/TQCounter.h"
#include "QFramework/TQSample.h"
#include "QFramework/TQUtils.h"

#include <QFramework/TQLibrary.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQPCAAnalysisJob
//
// The TQPCAAnalysisJob allow to schedule the creation of PCA objects
// for each and every sample visitied during the analysis
// This information can later be retrieved from a .pca subfolder 
// of the analysis root file folder structure
// to perform a linear decorrelation of a set of variables via a PCA
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQPCAAnalysisJob)

//______________________________________________________________________________________________

TQPCAAnalysisJob::TQPCAAnalysisJob() : 
TQAnalysisJob("PCA"), 
  weightCutoff(std::numeric_limits<double>::epsilon())
{
  // default constructor
}


//______________________________________________________________________________________________

TQPCAAnalysisJob::TQPCAAnalysisJob(TString name_) : 
  TQAnalysisJob(name_),
  weightCutoff(std::numeric_limits<double>::epsilon())
{
  // create a new TQPCAAnalysisJob
  // take care of choosing an appropriate name (default "PCA")
  // as this will determine the name under which the resulting grids
  // will be saved in the folder structure

}

//______________________________________________________________________________________________

TQPCAAnalysisJob::TQPCAAnalysisJob(TQPCAAnalysisJob &job) : 
  TQAnalysisJob(job.GetName()),
  poolAt(job.poolAt),
  weightCutoff(job.weightCutoff)
{
  // copy constructor
  for(size_t i=0; i<job.fObservables.size(); i++){
    this->fObservables.push_back(fObservables[i]);
  }
}

//______________________________________________________________________________________________

TQPCAAnalysisJob::TQPCAAnalysisJob(TQPCAAnalysisJob *job) : 
  TQAnalysisJob(job ? job->GetName() : "PCA"),
  poolAt(job ? job->poolAt : NULL),
  weightCutoff(job ? job->weightCutoff : std::numeric_limits<double>::epsilon())
{
  // copy constructor
  if(job){
    for(size_t i=0; i<job->fObservables.size(); i++){
      this->fObservables.push_back(fObservables[i]);
    }
  }
}

//______________________________________________________________________________________________

TQPCAAnalysisJob* TQPCAAnalysisJob::copy(){
  // creates an exact copy of this job
  // neglecting temporary data members
  return new TQPCAAnalysisJob(this);
}

//______________________________________________________________________________________________

TQPCAAnalysisJob* TQPCAAnalysisJob::getClone(){
  // creates an exact copy of this job
  // neglecting temporary data members
  return new TQPCAAnalysisJob(this);
}

//______________________________________________________________________________________________

bool TQPCAAnalysisJob::initializePCA() {
  // initialize the PCA object properly
  if(this->fPCA) delete this->fPCA;
  int nVars = this->fObservables.size();
  if(!this->fValues){
    this->fValues = (double*)calloc(nVars,sizeof(double));
  }
  this->fPCA = new TQPCA(this->GetName(),nVars);
  for(size_t i=0; i<this->fNames.size(); i++){
    this->fPCA->setTagString(TString::Format("varname.%lu", (long unsigned int)i),fNames[i].Data());
    this->fPCA->setTagString(TString::Format("vartitle.%lu", (long unsigned int)i),fTitles[i].Data());
    this->fPCA->setTagString(TString::Format("varexpression.%lu",(long unsigned int)i),fExpressions[i].Data());
  }
  return true;
}

//______________________________________________________________________________________________

bool TQPCAAnalysisJob::finalizePCA() {
  // initialize the PCA object properly
  // and write it to the pool folder
  if(!this->poolAt) return false;
 
  /* get the cutflow folder */
  TString folderName = TString::Format(".pca/%s+",this->getCut()->GetName());
  TQFolder * cfFolder = this->poolAt->getFolder(folderName);
 
  /* stop if we failed to get the folder */
  if (!cfFolder) { return false; }
 
  /* remove existing grid */ 
  TObject* o = cfFolder->FindObject(this->GetName());
  if(o){
    cfFolder->Remove(o);
  } 

  /* add the pca object */
  if(!cfFolder->addObject(this->fPCA)){
    return false;
  }

  this->fPCA = NULL;

  return true;
}

//______________________________________________________________________________________________

bool TQPCAAnalysisJob::initializeSelf() {
  // initialize the job on the given sample

  if(!this->fPCA) this->initializePCA();
 
  /* initialize TQObservables */
  for (unsigned int i = 0; i < fNames.size(); i++) {
    TQObservable* obs = TQObservable::getObservable(this->fExpressions[i],this->fSample);
    if (!obs->initialize(this->fSample)) {
      ERRORclass("Failed to initialize observable obtained from expression '%s' in TQPCAAnalysisJob '%s' for sample '%s'",this->fExpressions[i].Data(),this->GetName(),this->fSample->getPath().Data());
      return false;
    }
    this->fObservables.push_back(obs);
  }
  return true;
}

//______________________________________________________________________________________________

bool TQPCAAnalysisJob::finalizeSelf() {
  // finalize the job, writing all results to the pool folder
 
  if(this->fSample == this->poolAt)
    this->finalizePCA();

  /* initialize TQObservables */
  for (unsigned int i = 0; i < fObservables.size(); i++) {
    this->fObservables[i]->finalize();
  }

  this->fObservables.clear();

  /* finalize this analysis job */
  return true;
}

//______________________________________________________________________________________________

bool TQPCAAnalysisJob::initializeSampleFolder(TQSampleFolder * sf) {
  // initalize the job on a sample folder
  // check for pool tag

  bool pool = false;
  sf->getTagBool(".aj.pool.pca",pool);
 
  if(pool && !this->fPCA){
    this->poolAt = sf;
  }
 
  return true;
 
}




//______________________________________________________________________________________________

bool TQPCAAnalysisJob::finalizeSampleFolder(TQSampleFolder* sf) {
  // finalize the job on a sample folder
  // check for pool tag 

  bool pool =(sf == this->poolAt);
 
  /* stop if we no sample is defined */
  if (!sf) { return false; }
 
  if(pool)
    return this->finalizePCA();
 
  return true;
}


//______________________________________________________________________________________________

bool TQPCAAnalysisJob::execute(double weight) {
  // execute the job on a given event
 
  if(weight < this->weightCutoff) return true;
  // std::cout << "filling with weight " << weight << " (cutoff = " << this->weightCutoff << ")" << std::endl;

  for(size_t i=0; i<this->fObservables.size(); i++){
    this->fValues[i] = this->fObservables[i]->getValue();
  }

  if(this->checkValues()){
    this->fPCA->fill(weight,this->fValues);
  }

  return true;
}



//______________________________________________________________________________________________

bool TQPCAAnalysisJob::checkValues() {
  // check if the values are valid to avoid inf/NaN
  // being propagated to the PCA object

  for(size_t i=0; i<this->fObservables.size(); i++){
    if(!TQUtils::isNum(this->fValues[i])) return false;
  }
 
  return true;
}



//______________________________________________________________________________________________

TQPCAAnalysisJob::~TQPCAAnalysisJob() {
  // destructor
  if(this->fPCA)
    delete this->fPCA;
  for(size_t i=0; i<this->fObservables.size(); i++){
    delete this->fObservables[i];
  }
  if(this->fValues)
    free(this->fValues);
}


//______________________________________________________________________________________________

TObjArray * TQPCAAnalysisJob::getBranchNames() {
  // Return all used TTree branches in a TObjArray* 
  // The strings should include all used branch names,
  // but can also include formulas, number, etc.
 
  TObjArray * bNames = new TObjArray();
 
  for(size_t i=0; i<this->fObservables.size(); i++){
    TObjArray* branchNames = this->fObservables[i]->getBranchNames();
    bNames->AddAll(branchNames);
    delete branchNames;
  }
 
  return bNames;
}

//______________________________________________________________________________________________

void TQPCAAnalysisJob::bookVariable(const TString& name, const TString& title, const TString& expression){
  // book a variable with the given name, title and expression
  this->fNames.push_back(name);
  this->fTitles.push_back(title);
  this->fExpressions.push_back(expression);
}


//______________________________________________________________________________________________

void TQPCAAnalysisJob::bookVariable(const TString& expression){
  // book a variable with the given expression
  // it will also be used as name and title of variable
  this->fNames.push_back(expression);
  this->fTitles.push_back(expression);
  this->fExpressions.push_back(expression);
}

//______________________________________________________________________________________________

void TQPCAAnalysisJob::setWeightCutoff(double cutoff){
  this->weightCutoff = cutoff;
}

//______________________________________________________________________________________________

double TQPCAAnalysisJob::getWeightCutoff(){
  return this->weightCutoff;
}
