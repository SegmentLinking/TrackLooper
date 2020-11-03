#include <sstream>
#include "RVersion.h"
#if ROOT_VERSION_CODE < ROOT_VERSION(6,07,00)
#define private public
#define protected public
#include "TMVA/Factory.h"
#include "TMVA/DataSetInfo.h"
#undef private
#undef protected
#else
#define private public
#define protected public
#include "TMVA/Factory.h"
#include "TMVA/DataSetInfo.h"
#include "TMVA/DataLoader.h"
#undef private
#undef protected
#endif

#include "QFramework/TQMVA.h"
#include "QFramework/TQIterator.h"
#include "TFile.h"
#include "QFramework/TQUtils.h"

// #define _DEBUG_

#include "QFramework/TQLibrary.h"

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQMVA
//
// The TQMVA is a wrapper class providing an interface of the TQSampleFolder structure
// to a TMVA object.
//
// This class can be used as follows:
// 1. Instantiate an object of TQMVA, providing a TQSampleFolder
// either directly to the constructor or via setSampleFolder
// 2. Call the member function createFactory, providing the name of 
// the desired output file and the TMVA options to be passed through.
// You may also set the factory manually via setFactory instead.
// The factory may at any time be retrieved via getFactory.
// 3. Setup the TQMVA
// - Aquire or define an instance of TQCut and assign it
// to the TQMVA via the member function setBaseCut.
// You may instruct the TQMVA to use a derived cut by calling
// the TQMVA::useCut function on some cut name
// - Add the variables using TQMVA::bookVariable
// - Call the addSignal and addBackground functions to announce 
// paths under which signal/background trees may be found.
// This method is superior to adding the trees directly because
// the trees will automatically be obtained from the samples.
// 4. Call TQMVA::readSamples to propagate the setup to the TMVA::Factory
// and read the input samples. As an argument, you may pass an EventSelector,
// a simple class-type object that will sort your events into a 
// 'training' and a 'testing' category. By default, an even-odd-rule will 
// be used.
// 5. Call TQMVA::prepareTrees("XYZ") to replace TMVA::prepareTrainingAndTestTree,
// where the option string "XYZ" will be forwarded to the latter function.
// 6. Retrieve the TMVA::Factory via TQMVA::getFactory. 
// 7. Perform your analysis on the TMVA::Factory directly.
//
// Please note that after retrieving the TMVA::Factory, you may delete the TQMVA object.
// This will NOT delete your TMVA::Factory, which you can use for further analysis.
// It is the users responsibility to delete the TMVA::Factory and close and delete
// the output file. If the TMVA::Factory was created with TQMVA::createFactory, 
// this can be achieved by calling TQMVA::deleteFactory and TQMVA::closeOutputFile.
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQMVA)

TQMVA::TQMVA() :
TQNamedTaggable("TQMVA")
{
  // default constructor
  init();
}

TQMVA::TQMVA(const TString& name_) : 
  TQNamedTaggable(TQStringUtils::replace(name_,"-","_"))
{
  // constructor taking a job name
  init();
}

TQMVA::TQMVA(TQSampleFolder* sf) : 
  TQNamedTaggable("TQMVA"),
  fSampleFolder(sf)
{
  // constructor taking a sample folder
  init();
}

TQMVA::TQMVA(const TString& name_, TQSampleFolder* sf) : 
  TQNamedTaggable(TQStringUtils::replace(name_,"-","_")),
  fSampleFolder(sf)
{
  // constructor taking a job name and sample folder
  init();
}

void TQMVA::init(){
  this->SetTitle("TQMVA");
  #ifndef LEGACY_INTERFACE_PRE_607
  this->fDataLoader = new TMVA::DataLoader("TQMVADataLoader");
  #endif
}

TQMVA::~TQMVA(){
  // default destructor
#ifndef LEGACY_INTERFACE_PRE_607
  delete this->fDataLoader;
#endif
}
 
void TQMVA::printListOfVariables() const {
  // print the list of currently booked variables
  for(size_t i=0; i<this->fNames.size(); i++){
    std::cout << this->fNames[i] << "\t" << this->fExpressions[i] << std::endl;
  }
}

TString TQMVA::getVariableExpression(const TString& var){
  // return the full expression for the variable with a given name
  for(size_t i=0; i<this->fExpressions.size(); i++){
    if(TQStringUtils::equal(var,this->fNames[i])){
      return this->fExpressions[i];
    }
  }
  return TQStringUtils::emptyString;
}

TString TQMVA::getVariableExpression(const char* var){
  // return the full expression for the variable with a given name
  return this->getVariableExpression((TString)(var));
}

void TQMVA::setBaseCut(TQCut* cut){
  // set the base cut
  this->fBaseCut = cut;
}

TQCut* TQMVA::getBaseCut(){
  // get the base cut
  return this->fBaseCut;
}

TQCut* TQMVA::useCut(const TString& name){
  // set and return the active cut (by name)
  this->fActiveCut = this->fBaseCut->getCut(name);
  return this->fActiveCut;
}

TMVA::Factory* TQMVA::getFactory(){
  // get the TMVA::Factory embedded in this class
  return this->fMVA;
}
void TQMVA::setFactory(TMVA::Factory* mva){
  // set the embedded TMVA::Factory
  this->fMVA = mva;
}
TQSampleFolder* TQMVA::getSampleFolder(){
  // get the TQSampleFolder embedded in this class
  return this->fSampleFolder;
}
void TQMVA::setSampleFolder(TQSampleFolder* sf){
  // set the embedded sample folder
  this->fSampleFolder = sf;
}

void TQMVA::deleteFactory(){
  // delete the TMVA::Factory embedded in this class
  if(this->fMVA) delete this->fMVA;
}
void TQMVA::closeOutputFile(){
  // close the output file of the TMVA::Factory
  // will only work as desired if the factory
  // was created with TQMVA::createFactory
  if(this->fOutputFile){
    this->fOutputFile->Close();
    delete this->fOutputFile;
    this->fOutputFile = NULL;
  }
}
bool TQMVA::createFactory(const TString& filename, const TString& options){
  // create an instance of TMVA::Factory
  // will open an output file in RECREATE mode
  // neither the TMVA nor the TFile will be deleted
  // when the TQMVA is deleted, they need to be
  // closed/deleted with closeOutputFile and deleteFactory
  if(!TQUtils::ensureDirectoryForFile(filename)){
    TQLibrary::ERRORclass("unable to access directory for file '%s'",filename.Data());
    return false;
  }
  this->fOutputFile = TFile::Open(filename,"RECREATE");
  if(!fOutputFile || !fOutputFile->IsOpen()){
    TQLibrary::ERRORclass("unable to open file '%s'",filename.Data());
    if(fOutputFile) delete fOutputFile;
    return false;
  }
  this->setTagString("outputFileName",filename);
  this->fMVA = new TMVA::Factory(this->GetName(),fOutputFile,options.Data());
  return true;
}

void TQMVA::addVariable(const TString& name, const TString& title, const TString& expression, const TString& unit, char vtype, double min, double max, bool spectator){
  // internal function mapping to the TMVA::Factory/DataLoader management functions
  if(!this->fMVA){
    throw std::runtime_error("unable to book variable without instance of TMVA!");
  }
#ifdef LEGACY_INTERFACE_PRE_607
  if(spectator) this->fMVA->AddSpectator(TString::Format("%s := %s",name.Data(),name.Data()),title,unit,min,max);
  else this->fMVA->AddVariable(TString::Format("%s := %s",name.Data(),name.Data()),title,unit,vtype,min,max);
  this->fNames.push_back(name);
  this->fExpressions.push_back(expression);
  std::vector<TMVA::VariableInfo>& variables = this->fMVA->DefaultDataSetInfo().GetVariableInfos();
  for(size_t i=0; i<this->fNames.size(); i++){
    if(TQStringUtils::equal(variables[i].GetInternalName(),name)){
      variables[i].SetLabel(expression);
    }
  }
#else
  if(spectator) this->fDataLoader->AddSpectator(TString::Format("%s := %s",name.Data(),name.Data()),title,unit,min,max);
  else this->fDataLoader->AddVariable(TString::Format("%s := %s",name.Data(),name.Data()),title,unit,vtype,min,max);
  this->fNames.push_back(name);
  this->fExpressions.push_back(expression);
  std::vector<TMVA::VariableInfo>& variables = this->fDataLoader->DefaultDataSetInfo().GetVariableInfos();
  for(size_t i=0; i<this->fNames.size(); i++){
    if(TQStringUtils::equal(variables[i].GetInternalName(),name)){
      variables[i].SetLabel(expression);
    }
  }
#endif
}

void TQMVA::printInternalVariables() const {
  // print the list of currently booked variables
  // this function accesses the internal variables of the TMVA::Factory
  #ifdef LEGACY_INTERFACE_PRE_607
  std::vector<TMVA::VariableInfo>& variables = this->fMVA->DefaultDataSetInfo().GetVariableInfos();
  for(size_t i=0; i<variables.size(); i++){
    std::cout << variables[i].GetInternalName() << ":" << variables[i].GetLabel() << ":" << variables[i].GetExpression() << " (" << variables[i].GetVarType() << "/" << variables[i].GetUnit() << ")" << std::endl;
  }
  #else 
  std::vector<TMVA::VariableInfo>& variables = this->fDataLoader->DefaultDataSetInfo().GetVariableInfos();
  for(size_t i=0; i<variables.size(); i++){
    std::cout << variables[i].GetInternalName() << ":" << variables[i].GetLabel() << ":" << variables[i].GetExpression() << " (" << variables[i].GetVarType() << "/" << variables[i].GetUnit() << ")" << std::endl;
  }
  #endif
}

void TQMVA::prepareTrees(const TString& options){
  // wrapper for TMVA::prepareTrainingAndTestTree (with empty cut)
  TCut cut;
#ifdef LEGACY_INTERFACE_PRE_607
  if(this->fMVA) this->fMVA->PrepareTrainingAndTestTree(cut,options);
# else 
  if(this->fDataLoader) this->fDataLoader->PrepareTrainingAndTestTree(cut,options);
#endif
}


void TQMVA::bookVariable(const TString& name_, const TString& expression_, double min, double max){
  // book a variable
  // will internally call TMVA::Factory::addVariable
  TString name = TQFolder::makeValidIdentifier(name_);
  TString expression(this->fAliases ? this->fAliases->replaceInText(expression_) : expression_);
  this->addVariable(name,name,expression,"",'F',min,max,false);
}

void TQMVA::bookVariable(const TString& name_, const TString& expression_, const TString& title, const TString& unit, double min, double max){
  // book a variable
  // will internally call TMVA::Factory::addVariable
  TString name = TQFolder::makeValidIdentifier(name_);
  TString expression(this->fAliases ? this->fAliases->replaceInText(expression_) : expression_);
  this->addVariable(name,title,expression,unit,'F',min,max,false);
}

void TQMVA::bookVariable(const TString& name_, const TString& expression_, const TString& title, double min, double max){
  // book a variable
  // will internally call TMVA::Factory::addVariable
  TString name = TQFolder::makeValidIdentifier(name_);
  TString expression(this->fAliases ? this->fAliases->replaceInText(expression_) : expression_);
  this->addVariable(name,title,expression,"",'F',min,max,false);
}

void TQMVA::bookVariable(const char* name_, const char* expression_, double min, double max){
  // book a variable
  // will internally call TMVA::Factory::addVariable
  TString name = TQFolder::makeValidIdentifier(name_);
  TString expression(this->fAliases ? this->fAliases->replaceInText(expression_).Data() : expression_);
  this->addVariable(name,name,expression,"",'F',min,max,false);
}

void TQMVA::bookVariable(const char* name_, const char* expression_, const char* title, const char* unit, double min, double max){
  // book a variable
  // will internally call TMVA::Factory::addVariable
  TString name = TQFolder::makeValidIdentifier(name_);
  TString expression(this->fAliases ? this->fAliases->replaceInText(expression_).Data() : expression_);
  this->addVariable(name,title,expression,unit,'F',min,max,false);
}

void TQMVA::bookVariable(const char* name_, const char* expression_, const char* title, double min, double max){
  // book a variable
  // will internally call TMVA::Factory::addVariable
  TString name = TQFolder::makeValidIdentifier(name_);
  TString expression = (this->fAliases ? this->fAliases->replaceInText(expression_).Data() : expression_);
  this->addVariable(name,title,expression,"",'F',min,max,false);
}

//--------------------------------------

void TQMVA::bookSpectator(const TString& name_, const TString& expression_, double min, double max){
  // book a variable
  // will internally call TMVA::Factory::addVariable
  TString name = TQFolder::makeValidIdentifier(name_);
  TString expression(this->fAliases ? this->fAliases->replaceInText(expression_) : expression_);
  this->addVariable(name,name,expression,"",'F',min,max,true);
}

void TQMVA::bookSpectator(const TString& name_, const TString& expression_, const TString& title, const TString& unit, double min, double max){
  // book a variable
  // will internally call TMVA::Factory::addVariable
  TString name = TQFolder::makeValidIdentifier(name_);
  TString expression(this->fAliases ? this->fAliases->replaceInText(expression_) : expression_);
  this->addVariable(name,title,expression,unit,'F',min,max,true);
}

void TQMVA::bookSpectator(const TString& name_, const TString& expression_, const TString& title, double min, double max){
  // book a variable
  // will internally call TMVA::Factory::addVariable
  TString name = TQFolder::makeValidIdentifier(name_);
  TString expression(this->fAliases ? this->fAliases->replaceInText(expression_) : expression_);
  this->addVariable(name,title,expression,"",'F',min,max,true);
}

void TQMVA::bookSpectator(const char* name_, const char* expression_, double min, double max){
  // book a variable
  // will internally call TMVA::Factory::addVariable
  TString name = TQFolder::makeValidIdentifier(name_);
  TString expression(this->fAliases ? this->fAliases->replaceInText(expression_).Data() : expression_);
  this->addVariable(name,name,expression,"",'F',min,max,true);
}

void TQMVA::bookSpectator(const char* name_, const char* expression_, const char* title, const char* unit, double min, double max){
  // book a variable
  // will internally call TMVA::Factory::addVariable
  TString name = TQFolder::makeValidIdentifier(name_);
  TString expression(this->fAliases ? this->fAliases->replaceInText(expression_).Data() : expression_);
  this->addVariable(name,title,expression,unit,'F',min,max,true);
}

void TQMVA::bookSpectator(const char* name_, const char* expression_, const char* title, double min, double max){
  // book a variable
  // will internally call TMVA::Factory::addVariable
  TString name = TQFolder::makeValidIdentifier(name_);
  TString expression = (this->fAliases ? this->fAliases->replaceInText(expression_).Data() : expression_);
  this->addVariable(name,title,expression,"",'F',min,max,true);
}

//-------------------------------------

void TQMVA::addSignal(const TString& path){
  // add a signal path
  this->fSigPaths.push_back(path);
}
void TQMVA::addBackground(const TString& path){
  // add a background path
  this->fBkgPaths.push_back(path);
}
void TQMVA::clearSignal(){
  // clear all signal paths
  this->fSigPaths.clear();
}
void TQMVA::clearBackground(){
  // clear all background paths
  this->fBkgPaths.clear();
}

int TQMVA::readSamples(){
  // initialize the TMVA::Factory instance
  // this will open all required files and access the trees
  // and return the number of successfully read events
  return this->readSamples(TQMVA::EvenOddEventSelector());
}

int TQMVA::readSamples(const TQMVA::EventSelector& evtsel){
  // initialize the TMVA::Factory instance
  // this will open all required files and access the trees
  // and return the number of successfully read events
  if(!this->fMVA){
    ERRORclass("cannot initialze - no TMVA::Factory assigned!");
    return 0;
  }
  if(!this->fBaseCut){
    ERRORclass("cannot initialze - no TQCut assigned!");
    return 0;
  }
 
  int retval = 0;
  retval += this->readSamplesOfType(TQMVA::Signal,evtsel);
  retval += this->readSamplesOfType(TQMVA::Background,evtsel);
 
  return retval;
}

int TQMVA::readSamplesOfType(TQMVA::SampleType type){
  // read all samples of the given type (Signal or Background), using the default (Even-Odd) event selector
  return this->readSamplesOfType(type,TQMVA::EvenOddEventSelector());
}

int TQMVA::readSamplesOfType(TQMVA::SampleType type, const TQMVA::EventSelector& sel){
  // read all samples of the given type (Signal or Background), using the
  // given event selector
  DEBUGclass("function called for '%s'",type == TQMVA::Signal ? "Signal" : "Background");
  TQSampleIterator sitr(this->getListOfSamples(type),true);
  int retval = 0;
  while(sitr.hasNext()){
    TQSample* s = sitr.readNext();
    this->readSample(s, type,sel);
    retval++;
  }
  return retval;
}

int TQMVA::readSample(TQSample* s, TQMVA::SampleType type, const TQMVA::EventSelector& sel){
  // read a specific sample, assigning it to the given type (Signal or
  // Background), using the given event selector
  if(fOutputFile) this->fOutputFile->cd();
  if(!fActiveCut) fActiveCut = fBaseCut;
  if(!s){
    ERRORclass("sample is NULL");
    return -1;
  }
  if(!fActiveCut){
    ERRORclass("cannot read sample '%s' without active cut, please use TQMVA::setBaseCut(...) to set a base cut");
    return -1;
  }

  DEBUGclass("reading sample '%s'",s->getPath().Data());

  std::vector<double> vars(this->fExpressions.size());

  TQToken* tok = s->getTreeToken();
  if(!tok){
    ERRORclass("unable to obtain tree token for sample '%s'",s->getPath().Data());
    return -1;
  }
  TTree* t = (TTree*)(tok->getContent());
 
  DEBUGclass("initializing cut '%s'",fBaseCut->GetName());
  this->fBaseCut->initialize(s);
  
  // the following two loops seem strangely chopped apart and they are indeed but for now this should make it work untill a better fix is available
  std::vector<TQObservable*> observables;
  for(size_t i=0; i<this->fExpressions.size(); i++){
    TQObservable* obs = TQObservable::getObservable(this->fExpressions[i],s);
    if (!obs->initialize(s)) {
      ERRORclass("Failed to initialize observable obtained from expression '%s' in TQMVA for sample '%s'",this->fExpressions[i].Data(),s->getPath().Data());
    }
    observables.push_back(obs);
  }
    
  t->SetBranchStatus("*", 0);

  for(size_t i=0; i<this->fExpressions.size(); i++){  
    TQObservable* obs = observables.at(i);
    DEBUGclass("activating branches for variable '%s'",this->fNames[i].Data());
    TObjArray* bNames = obs->getBranchNames();
    if (bNames) bNames->SetOwner(true);
#ifdef _DEBUG_
    DEBUGclass("enabling the following branches:");
    TQListUtils::printList(bNames);
#endif
    TQListUtils::setBranchStatus(t,bNames,1);
    delete bNames; 
  }

  {
    DEBUGclass("activating branches for cut '%s'",this->fBaseCut->GetName());
    TObjArray* branchNames = this->fBaseCut->getListOfBranches();
    branchNames->SetOwner(true);
#ifdef _DEBUG_
    DEBUGclass("enabling the following branches:");
    TQListUtils::printList(branchNames);
#endif
    TQListUtils::setBranchStatus(t,branchNames,1);
  }

  int nTrainEvent = 0;
  double sumWeightsTrain = 0;
  int nTestEvent = 0;
  double sumWeightsTest = 0;
  int nEvent = 0;
 
  DEBUGclass("entering event loop using cut '%s'",fActiveCut->GetName());
  
  if(this->fVerbose) TQLibrary::msgStream.startProcessInfo(TQMessageStream::INFO,80,"r",TString::Format("%s: %s",(type == TQMVA::Signal ? "sig" : "bkg"),s->getPath().Data()));
  
  TQCounter* cnt_train = NULL;
  TQCounter* cnt_test = NULL;
  TQCounter* cnt_total = NULL;
  
  //@tag: [makeCounters] If set to true, counters will be added to the cutflow counters, containing the number of training, testing and training+testing events  
  bool makeCounters = this->getTagBoolDefault("makeCounters",false);
  if(makeCounters){
    cnt_train = new TQCounter(TString::Format("TQMVA_%s_testing", this->GetName()),TString::Format("%s testing events", this->GetTitle()));
    cnt_test = new TQCounter(TString::Format("TQMVA_%s_training",this->GetName()),TString::Format("%s training events",this->GetTitle()));
    cnt_total = new TQCounter(TString::Format("TQMVA_%s_total", this->GetName()),TString::Format("%s testing+training events",this->GetTitle()));
  }

#ifndef _DEBUG_
  TQLibrary::redirect_stdout("/dev/null");
#endif
  Long64_t nEntries = t->GetEntriesFast();
  for(Long64_t iEvent = 0; iEvent < nEntries; ++iEvent){
    t->GetEntry(iEvent);
#ifdef _DEBUG_
    //if(this->fActiveCut) this->fActiveCut->printEvaluation();
    //else DEBUGclass("No active cut present");
#endif
    if(!this->fActiveCut->passedGlobally()){
      continue;
    }
    nEvent++;
 
    for(size_t i=0; i<observables.size(); ++i){
#ifdef _DEBUG_
      try {
#endif
        vars[i] = observables[i]->getValue();
#ifdef _DEBUG_  
      } catch(const std::exception& e){
        BREAK("ERROR in TQMVA: observable '%s' with expression '%s' encountered error '%s'",observables[i]->GetName(),observables[i]->getActiveExpression().Data(),e.what());
      }
#endif
    }
    double weight = this->fActiveCut->getGlobalWeight() * s->getNormalisation();
 
    if(sel.selectEvent(iEvent)){
#ifdef LEGACY_INTERFACE_PRE_607
      if(type==TQMVA::Signal) this->fMVA->AddSignalTrainingEvent(vars,weight);
      else this->fMVA->AddBackgroundTrainingEvent(vars,weight);
#else
      if(type==TQMVA::Signal) this->fDataLoader->AddSignalTrainingEvent(vars,weight);
      else this->fDataLoader->AddBackgroundTrainingEvent(vars,weight);
#endif
      if(makeCounters){
        cnt_train->add(weight);
        cnt_total->add(weight);
      }
      nTrainEvent++;
      sumWeightsTrain += weight;
    } else {
#ifdef LEGACY_INTERFACE_PRE_607
      if(type==TQMVA::Signal) this->fMVA->AddSignalTestEvent(vars,weight);
      else this->fMVA->AddBackgroundTestEvent(vars,weight);
#else
      if(type==TQMVA::Signal) this->fDataLoader->AddSignalTestEvent(vars,weight);
      else this->fDataLoader->AddBackgroundTestEvent(vars,weight);
#endif
      if(makeCounters){
        cnt_test->add(weight);
        cnt_total->add(weight);
      }
      nTestEvent++;
      sumWeightsTest += weight;
    }
  }
#ifndef _DEBUG_
  TQLibrary::restore_stdout();
#endif

  if(this->fVerbose){
    if(nTrainEvent == 0 || nTestEvent == 0 || nEvent == 0){
      TQLibrary::msgStream.endProcessInfo(TQMessageStream::WARN);
    } else {
      TQLibrary::msgStream.endProcessInfo(TQMessageStream::OK);
    } 
    if(nEntries == 0){
      WARNclass("this sample was empty (tree had no entries)");
    } else if(nEvent == 0){
      WARNclass("no events from this sample passed the cut '%s' (from a total of %lld events)!",this->fActiveCut->GetName(),nEntries);
#ifdef _DEBUG_
      DEBUGclass("cut expression is as follows:");
      this->fActiveCut->printActiveCutExpression();
#endif
    } else {
      if(nTrainEvent == 0){
        WARNclass("event selector did not select any training events (from a total of %d selected events)!",nEvent);
      } 
      if(nTestEvent == 0){
        WARNclass("event selector did not select any testing events (from a total of %d selected events)!",nEvent);
      }
    }
    if(this->fVerbose > 1){
      INFO("number of read events: %d training (%.1f weighted), %d test (%.1f weighted)",nTrainEvent,sumWeightsTrain,nTestEvent,sumWeightsTest);
    }
  }

  DEBUGclass("finalizing cut '%s'",fBaseCut->GetName());
  this->fBaseCut->finalize();
  for(size_t i=0; i<observables.size(); ++i){
    DEBUGclass("finalizing variable '%s'",this->fNames[i].Data());
    observables[i]->finalize();
  }

  if(makeCounters){
    TQFolder* counters = s->getFolder(".cutflow+");
    if(counters){
      counters->addObject(cnt_train);
      counters->addObject(cnt_test);
      counters->addObject(cnt_total);
    }
  }
  
  s->returnTreeToken(tok);
  return nEvent;
}

void TQMVA::printListOfSamples(TQMVA::SampleType type){
  // print the list of all samples of the given type (Signal or Background)
  std::vector<TString>* vec = (type == TQMVA::Signal ? &(this->fSigPaths) : &(this->fBkgPaths));
  std::cout << TQStringUtils::makeBoldBlue(this->GetName()) << TQStringUtils::makeBoldWhite(": samples of type '") << TQStringUtils::makeBoldBlue(type == TQMVA::Signal ? "Signal" : "Background") << TQStringUtils::makeBoldWhite("'") << std::endl;
  if(vec->size() < 1){
    std::cout << TQStringUtils::makeBoldRed("<no paths listed>") << std::endl;
    return;
  }
  for(size_t i=0; i<vec->size(); i++){
    TString path(vec->at(i));
    TQStringUtils::ensureTrailingText(path,"/*");
    std::cout << "\t" << TQStringUtils::makeBoldWhite(path) << std::endl;
    TList* l = this->fSampleFolder->getListOfSamples(path);
    if(l && l->GetEntries() > 0){
      TQSampleIterator itr(l,true);
      while(itr.hasNext()){
        std::cout << "\t\t";
        TQSample* s = itr.readNext();
        if(!s) std::cout << TQStringUtils::makeBoldRed("<NULL>");
        else std::cout << s->getPath();
        std::cout << std::endl;
      }
    } else {
      if(l) delete l;
      std::cout << "\t\t" << TQStringUtils::makeBoldRed("<no samples found under this path>" ) << std::endl;
    }
  }
}

TList* TQMVA::getListOfSamples(TQMVA::SampleType type){
  // get the list of all samples of the given type (Signal or Background)
  std::vector<TString>* vec = (type == TQMVA::Signal ? &(this->fSigPaths) : &(this->fBkgPaths));
  TList* retval = new TList();
  for(size_t i=0; i<vec->size(); i++){
    TString path(vec->at(i));
    TQStringUtils::ensureTrailingText(path,"/*");
    TQSampleIterator itr(this->fSampleFolder->getListOfSamples(path),true);
    while(itr.hasNext()) {
      TQSample* s = itr.readNext();
      if (s && !s->hasSubSamples()) retval->Add(s);
    }
  }
  return retval;
}

void TQMVA::setVerbose(int verbose){
  // control verbosity
  this->fVerbose = verbose;
}

TQTaggable* TQMVA::getAliases(){
  // retrieve a pointer to the alias container
  return this->fAliases;
}

void TQMVA::setAliases(TQTaggable* aliases){
  // set the alias container
  this->fAliases = aliases;
}

