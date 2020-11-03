#include "QFramework/TQObservable.h"
#include "TTreeFormula.h"
#include "QFramework/TQStringUtils.h"
#include "QFramework/TQSample.h"
#include "QFramework/TQToken.h"
#include "QFramework/TQUtils.h"
#include "TTree.h"
#include "QFramework/TQIterator.h"
#include "QFramework/TQFilterObservable.h"

// #define _DEBUG_
#include "QFramework/TQLibrary.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

TQObservable::Manager& TQObservable::getManager(){
	// return a reference to the manager
	return TQObservable::manager;
}

TQObservable::Manager::Manager(){
  // create an observable manager
  this->activeSet = new TFolder("default","default observable set");
  this->sets->Add(this->activeSet);
}
bool TQObservable::Manager::setActiveSet(const TString& name){
  // set/change the active set of observables that is currently used
  TFolder * set = dynamic_cast<TFolder*>(this->sets->FindObject(name));
  if(set) {
    this->activeSet = set;
    return true;
  } else {
    return false;
  }
}
void TQObservable::Manager::cloneActiveSet(const TString& newName){
  // clone the currently active observable set to a new instance with a new name
  this->createEmptySet(newName);
  TFolder* newset = dynamic_cast<TFolder*>(this->sets->FindObject(newName));
  TQObservableIterator itr(this->activeSet->GetListOfFolders());
  while(itr.hasNext()){
    TQObservable* obs = itr.readNext();
    if(!obs) continue;
    newset->Add(obs->getClone());
  }
}
void TQObservable::Manager::clearSet(const TString& name){
  // clear and empty the observable set with the given name
  TFolder* set = dynamic_cast<TFolder*>(this->sets->FindObject(name));
  TQObservableIterator itr(set->GetListOfFolders());
  while(itr.hasNext()){
    TQObservable* obs = itr.readNext();
    if(obs) delete obs;
  }
  this->sets->Remove(set);
  delete set;
}
void TQObservable::Manager::createEmptySet(const TString& name){
  // create a new empty set of observables with the given name
  this->sets->Add(new TFolder(name,name));
}
TCollection* TQObservable::Manager::listOfSets(){
  // retrieve the list of known sets
  return this->sets->GetListOfFolders();
}
TFolder* TQObservable::Manager::getActiveSet(){
	// retrieve the currently active set
	return this->activeSet;
}



TQObservable::Manager TQObservable::manager = TQObservable::Manager();
bool TQObservable::gAllowErrorMessages = false;




////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQObservable
//
// The TQObservable class acts on a TQSample with the aim of extracting
// numerical data from a TTree inside the event loop.
//
// TQObservable itself is an abstract base class that is inherited from by
// different observable variants. The most notable ones are
//
// - TQTreeFormulaObservable: A wrapper around TTreeFormula, capable of
// evaluating arithmetic expressions on the TTree branches.
// 
// - TQConstObservable: A trivial observable that has the same value for
// all events of a tree. While generally being of limited use, the existence
// of this observable type allows fast and convenient definition of cuts by
// providing an extremely fast way of implementing trivial cut or weight
// expressions.
//
// - TQMVATreeObservable: A more elaborate observable type that acts as a
// wrapper around a TMVA::Reader object and is capable of evaluating
// multivariate methods on the fly while only requiring a filepath to a
// valid TMVA weights.xml file as input.
//
// - TQMultiTreeObservable: A wrapper for an expression that consists of only
// constant values and names of other TQObservables, where the latter
// are required to be wrapped in brackets [...]. This observable type is
// required to cut on values provided by TQMVATreeObservable, but can also
// be used to combine outputs from different MVA methods or custom
// observable types.
//
// While any type of TQObservable may also be instantiated directly, it is
// generally advisable to use the factory functions
// 
// TQObservable::getTreeObservable(name,expression): creates a
// TQObservable with the given name and expression. The type of
// observable created depends on the expression and may be indicated by a
// prefix if necessary (see function documentation of
// TQObservable::createObservable for details). The observable is
// automatically added to the observable database and is only created if
// necessary - if an observable with the same name and expression already
// exists, it is retrieved instead. If an observable with the same name, but
// a different expression exists, an error message is generated.
//
// TQObservable::getTreeObservable(expression): same as above, but the
// expression is also used as the name.
//
// The pure getter functions TQObservable::getTreeObservableByName(name)
// and TQObservable::getTreeObservableByExpression(expression) do not
// create observables, but only retrieve them from the central database.
//
// Defining Custom Observable Classes
// 
// The TQObservable class can and should be used to define custom observable
// classes to implement a more complex behaviour. In order to have your custom
// observable class operate as expected, you need to implement the following
// methods:
//
// double getValue() const;
// This method returns the value of this observable. When implementing this
// method, please bear in mind that it will be called once for each event
// in all trees, possibly various times at each cut step. Try to implement
// this method as efficient as possible, even if this means spending more
// computation time in the other functions.
//
// bool initializeSelf();
// This method is supposed to initialize your observable in such a way that
// your getValue can safely be called. When this method is called, the
// values of the fields fSample, and fToken are already set and
// initialized, and you may access them in any way you seem fitting.
// If all went well, this method should return true. If an error occurred,
// you need to undo any changes performed previously and return false.
//
// bool finalizeSelf();
// This method is supposed to finalize your observable in such a way that
// all changes performed by initialzeSelf are undone and all data members
// allocated by initializeSelf are properly freed. The values of the
// fields fSample and fToken are still set and valid, and you
// may access them in any way you seem fitting. If all went well, this
// method should return true. In the unlikely case of an error, this method
// should return false.
// 
// TObjArray* getBranchNames(TQSample* s) const;
// This method is supposed to return a TObjArray containing TObjStrings,
// each representing the name of a branch of a tree. The argument with
// which this function is called is a pointer to a TQSample object which
// contains the tree and possibly meta-information in the form of tags. A
// typical implementation of this method may look like this:
// TObjArray* getBranchNames(TQSample* s) const {
// return TQUtils::getBranchNames(this->getCompiledExpression(s));
// }
// Here, the called to TQObservable::getCompiledExpression produces the
// final expression string, taking into account all meta-information of the
// sample, and TQUtils::getBranchNames parses the expression to extract all
// literal strings that may represent branch names.
// 
// Depending on the complexity of your observable class, it may be required to
// implement 
// void setExpression(const TString&) as well. If you
// re-implement this method, you should call it in your constructor to ensure
// consistent behaviour of your class.
//
// For examples on how an actual implementation might look like, you may want
// to browse the source code of any of the aforementioned observable types.
//
// You may also choose to modify TQObservable::createObservable to be
// capable of creating your own observable type. For this purpose, you might
// choose a prefix string to help identifying your observable type.
//
// When you have implemented and created an instance of your class, you may use
// it as follows:
// - call TQObservable::printObservables() to see if it's in the database
// - add your observable to a cut by either of the following methods:
// - call TQCut::setCutObservableName or TQCut::setWeightObservableName
// with the name of your observable
// - call TQCut::setCutObservableName or TQCut::setWeightObservableName
// with the name of your observable
// - provide the name at construction time when creating via
// TQFolder::importFolder by setting the tags .cutObservableName or
// .weightObservableName of the folder in question
// - choose the name of your observable as CUTNAME_cut or CUTNAME_weight
// in the first place, where CUTNAME is the name of the cut in
// question
// - add your observable to a histogram, eventlist or anything else by
// providing the name or expression of the cut as an argument to the
// associated analysis job
//
// Of course, if you have modified TQObservable::createObservable to
// support your observable type, you don't have to add the observable yourself
// but can instead rely on automatic creation of your observable at the time it
// is requested by the analysis chain. You may, however, want to check if
// everything works as expected by calling
// TQObservable::printObservables()
// You may be surpised that the number of observables in the database is
// small. This is because most observables are only created when they are first
// requested. Usually, this happens when an analysis job is configured or when
// a cut is assigned to a sample visitor. In the case of the cut, however, you
// may force premature creation of the observables by calling
// TQCut::setupObservables() on the instance in question.
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQObservable)

//______________________________________________________________________________________________

void TQObservable::clearAll(){
  TQIterator itr(getManager().listOfSets());
	while(itr.hasNext()){
		TFolder* set = dynamic_cast<TFolder*>(itr.readNext());
		if(!set) continue;
		TQObservableIterator obsitr(set->GetListOfFolders());
		while(obsitr.hasNext()){
			TQObservable* obs = obsitr.readNext();
			if(obs){
				set->Remove(obs);
				delete obs;
			}
		}
	}
}

//______________________________________________________________________________________________

TQObservable::TQObservable() : 
  TNamed("TQObservable","TQObservable")
{
  DEBUGclass("default constructor called");
  // default constructor, setting name and title
}

//______________________________________________________________________________________________

TQObservable::TQObservable(const TString& expression) :
  TQObservable()
{
  // constructor setting the name and expression
  DEBUGclass("constructor called with expression '%s'",expression.Data());
  this->SetName(TQObservable::makeObservableName(expression));
}

//______________________________________________________________________________________________

void TQObservable::Manager::registerFactory(const TQObservable::FactoryBase* factory, bool putFirst){
  // add a new factory to the TQObservable manager
  if(putFirst){
    this->observableFactories.insert(this->observableFactories.begin(),factory);
  } else {
    this->observableFactories.push_back(factory);
  }
}

//______________________________________________________________________________________________

void TQObservable::Manager::clearFactories(){
  // clear all factories from the list
  this->observableFactories.clear();
}

#include <typeinfo>

void TQObservable::Manager::printFactories(){
  for(size_t i=0; i<this->observableFactories.size(); ++i){
    std::cout << this->observableFactories[i]->className() << std::endl;
  }
}


#include "QFramework/TQMultiObservable.h"
#include "QFramework/TQMVAObservable.h"
#include "QFramework/TQConstObservable.h"
#include "QFramework/TQTreeFormulaObservable.h"
#include "QFramework/TQHistogramObservable.h"
#include "QFramework/TQVectorAuxObservable.h"

//______________________________________________________________________________________________

bool TQObservableFactory::setupDefault(){
  // clear all factories from the list
  TQObservable::manager.clearFactories();
  TQObservable::manager.registerFactory(TQConstObservable::getFactory(),false);
  TQObservable::manager.registerFactory(TQHistogramObservable<TH3>::getFactory(),false);
  TQObservable::manager.registerFactory(TQHistogramObservable<TH2>::getFactory(),false);
  TQObservable::manager.registerFactory(TQHistogramObservable<TH1>::getFactory(),false);
  TQObservable::manager.registerFactory(TQFilterObservable::getFactory(),false);
  TQObservable::manager.registerFactory(TQVectorAuxObservable::getFactory(),false);
  TQObservable::manager.registerFactory(TQMultiObservable::getFactory(),false);
  TQObservable::manager.registerFactory(TQMVAObservable::getFactory(),false);
  TQObservable::manager.registerFactory(TQTreeFormulaObservable::getFactory(),false);
  return true;
}

namespace {
  const bool _TQObservableFactory__setupDefault = TQObservableFactory::setupDefault();
}

//______________________________________________________________________________________________

TQObservable* TQObservable::Manager::createObservable(TString expression, TQTaggable* tags){
  // this is a factory function for TQObservables. 
  //  TQStringUtils::removeLeadingBlanks(expression);
  //  TQStringUtils::removeTrailingBlanks(expression);
  //  expression = TQObservable::compileExpression(expression,tags,false);
  TQStringUtils::removeLeadingBlanks(expression);
  TQStringUtils::removeTrailingBlanks(expression);
  for(size_t i=0; i<this->observableFactories.size(); ++i){
    TQObservable* obs = this->observableFactories[i]->tryCreateInstance(expression);
    if(obs){
      TString obsname(TQObservable::makeObservableName(expression));
      TQObservable* other = (TQObservable*)(this->activeSet->FindObject(obsname));
      if(other){
        ERRORclass("These are the tags used when attempting to create a new observable instance:");
        tags->printTags();
        ERRORclass("These are the already existing observables:");
        TQObservable::printObservables();
        ERRORclass("An observable was created via its factory from expression '%s'. However, there is already another observable with the same name in the current set of observables. The list of already existing observables has been printed above. The duplicate name is '%s'. if this happens with the latest version of QFramework pease report it to qframework-users@cern.ch and try to provide a test case to reproduce the issue!", expression.Data(), obsname.Data());
        throw std::runtime_error(TString::Format("Internal Error: cannot add observable with name '%s' (generated from expression '%s') to set '%s' -- name is already in use, observable has expression '%s'!",obsname.Data(),expression.Data(),this->activeSet->GetName(),other->getExpression().Data()).Data());
      } else {
        DEBUG(TString::Format("adding observable '%s' to set '%s'",obsname.Data(),this->activeSet->GetName()).Data());
        obs->SetName(obsname);
        this->activeSet->Add(obs);
      }
      return obs;
    }
  }
  return NULL;
}

//______________________________________________________________________________________________

bool TQObservable::addObservable(TQObservable* obs){
  // add an existing tree observable to the pool
  if(!obs) return false;
  if(dynamic_cast<TQObservable*>(TQObservable::manager.activeSet->FindObject(obs->GetName()))){
    WARNclass("Failed to add observable '%s': an observable with this name already present",obs->GetName());
    return false;
  }
  TQObservable::manager.activeSet->Add(obs);
  obs->fIsManaged = true;
  return true;
}

//______________________________________________________________________________________________

bool TQObservable::addObservable(TQObservable* obs, const TString& name){
  // add an existing tree observable to the pool
  // giving it a new name
  if(!obs) return false;
  if(dynamic_cast<TQObservable*>(TQObservable::manager.activeSet->FindObject(name))){
    return false;
  }
  obs->SetName(name);
  TQObservable::manager.activeSet->Add(obs);
  obs->fIsManaged = true;
  return true;
}

//______________________________________________________________________________________________

void TQObservable::printObservables(const TString& filter){
  // retrieve a TQObservable by expression
  // create it if necessary
  int width = TQLibrary::getConsoleWidth();
  TQIterator itr(TQObservable::manager.listOfSets());
  while(itr.hasNext()){
    TFolder* f = dynamic_cast<TFolder*>(itr.readNext());
    TQObservableIterator obsitr(f->GetListOfFolders());
    std::cout << TQStringUtils::makeBoldWhite(TQStringUtils::fixedWidth(f->GetName(),.2*width,"l")) << " " << TQStringUtils::makeBoldWhite(TQStringUtils::fixedWidth("name",.3*width,"l")) << " " << TQStringUtils::makeBoldWhite("expression") << std::endl;
    std::cout << TQStringUtils::makeBoldWhite(TQStringUtils::repeat("=",width)) << std::endl;
    while(obsitr.hasNext()){
      TQObservable* obs = obsitr.readNext();
      if(!obs) continue;
      if(!TQStringUtils::matches(obs->GetName(),filter)) continue;
      std::cout << TQStringUtils::makeBoldBlue(TQStringUtils::fixedWidth(obs->IsA()->GetName(),.2*width,"l"));
      std::cout << " ";
      std::cout << TQStringUtils::makeBoldWhite(TQStringUtils::fixedWidth(obs->GetName(),.3*width,"l"));
      std::cout << " ";
      std::cout << TQStringUtils::fixedWidth(obs->isInitialized() ? TQStringUtils::makeBoldRed(obs->getActiveExpression()) : obs->getExpression(),.5*width,"l.");
      std::cout << std::endl;
    }
  }
}

//______________________________________________________________________________________________

bool TQObservable::matchExpressions(const TString& ex1, const TString& ex2, bool requirePrefix){
  DEBUGclass("trying to match expressions '%s' and '%s'. Prefix matching: %s",ex1.Data(),ex2.Data(),requirePrefix?"required":"ignored");
  //simple parsing
  if(ex1.Length() == ex2.Length()) return TQStringUtils::equal(ex1,ex2);
  
  //try matching without prefix. Originally intended to match expressions like 'Multi:[x]+[y]' and '[x]+[y]'. Can be disabled (requirePrefix=true) as it can cause infinite recursions for observables which explicity require a prefix, e.g. VecAND:x. Prefix requirement is controled by the observable instance a user-expression is compared against:
  if (requirePrefix) {
    DEBUGclass("prefix is required but lengths don't match. Expressions do not match.");
    return false;
  }
  
  if(ex1.Length() > ex2.Length()){
    int first = ex1.First(":");
    DEBUGclass("comparing: '%s' vs '%s'",ex1(first+1,ex1.Length()-first-1).Data(),ex2.Data());
    return (first < .5*ex1.Length()) && TQStringUtils::isValidIdentifier(ex1(0,first),TQStringUtils::alphanum) && TQStringUtils::equal(ex1(first+1,ex1.Length()-first-1),ex2);
  }
  if(ex1.Length() < ex2.Length()){
    int first = ex2.First(":");
    DEBUGclass("comparing: '%s' vs '%s'",ex2(first+1,ex2.Length()-first-1).Data(),ex1.Data());
    return (first < .5*ex2.Length()) && TQStringUtils::isValidIdentifier(ex2(0,first),TQStringUtils::alphanum) && TQStringUtils::equal(ex2(first+1,ex2.Length()-first-1),ex1);
  }
  DEBUGclass("finished matching, no attempt succeeded");
  return false;
}

//______________________________________________________________________________________________

TQObservable* TQObservable::getObservable(const TString& origexpr, TQTaggable* tags){
  // retrieve an incarnation of a TQObservable
  // the observable is identified by its name 'expression'
  // also, the incarnation of this observable needs to match the compiled expression
  // retrieved by compiling expression using the tags provided
  DEBUGclass("trying to obtain observable with origexpr '%s'",origexpr.Data());
  TString incarnation = TQObservable::unreplaceBools(TQObservable::compileExpression(origexpr,tags,true));
  if(incarnation.Contains("$")){
    tags->printTags();
    ERRORclass("there seem to be unresolved aliases/tags in expression '%s', available aliases are listed above.",incarnation.Data());
    return NULL;
  }
  TString expression(origexpr);
  expression.ReplaceAll("\\","");
  TString obsname(TQObservable::makeObservableName(expression));
  TString incname(TQObservable::makeObservableName(incarnation));
  DEBUGclass("attempting to retrieve observable '%s' (raw name)/'%s' (compiled name) with expression '%s'",obsname.Data(),incname.Data(),incarnation.Data());
  if(obsname.IsNull()) {ERRORclass("Failed to retrieve observable: Observable name is empty! Original expression was '%s'",expression.Data()); return NULL;}
  TQObservableIterator itr(TQObservable::manager.activeSet->GetListOfFolders());
  while(itr.hasNext()){
    TQObservable* obs = itr.readNext();
    if(!obs) continue;
    
    DEBUGclass("next object is: '%s' ",obs->GetName());
    
    if(TQObservable::matchExpressions(obs->GetName(),incname, obs->isPrefixRequired() )){
      DEBUGclass("found perfect match '%s' for '%s'",obs->GetName(),incname.Data());
      return obs;
    }
    
    DEBUGclass("comparing '%s' vs. '%s' with expression=%d, initialized=%d",obs->getActiveExpression().Data(),incarnation.Data(),obs->hasExpression(),obs->isInitialized());
      
    if(obs->hasExpression()){
      if(obs->isInitialized() && TQObservable::matchExpressions(TQObservable::unreplaceBools(obs->getActiveExpression()),incarnation,obs->isPrefixRequired() )){
        DEBUGclass("found pre-initialized match '%s' for '%s'",obs->getActiveExpression().Data(),incarnation.Data());
        return obs;
      }
      if(!obs->isInitialized() && TQObservable::matchExpressions(TQObservable::unreplaceBools(obs->getExpression()),expression,obs->isPrefixRequired() )){
        DEBUGclass("found free match '%s' for '%s'",obs->getExpression().Data(),expression.Data());
        return obs;
      }
    }
  }
  itr.reset();
  TQObservable* match = NULL;
  size_t rating = -1;
  while(itr.hasNext()){
    TQObservable* obs = itr.readNext();
    if(!obs) continue;
    DEBUGclass("comparing observable's compiled expression to incarnation: '%s' vs. '%s'",TQObservable::unreplaceBools(obs->getCompiledExpression(tags)).Data(),incarnation.Data());
    if(TQObservable::matchExpressions(TQObservable::unreplaceBools(obs->getCompiledExpression(tags)),incarnation,obs->isPrefixRequired())){
      size_t obsrating = TQStringUtils::countText(obs->getExpression(),"$");//TODO: improve! this is mostly useless, since the default (i.e. non-overridden) implementation of getExpression returns the name of the observable. Some observables' names do not contain "$" anymore since (correctly) TQObservable::makeObservableName is used before setting fName. For now we need this, e.g., for HWWEventWeight with systematics where getExpression() is overridden. 
      DEBUGclass("found general match '%s' for '%s' with rating %d (expression is '%s')",obs->GetName(),incname.Data(),(int)(obsrating),obs->getExpression().Data());
      if(!match || obsrating < rating){
        match = obs;
        rating = obsrating;
      }
    }
  }
  if(match){
    if(match->isInitialized()){
      if(!(match->hasFactory())){
        TQObservable* clone = match->getClone();
        if(!clone){
          throw std::runtime_error(TString::Format("tried to retrieve observable '%s' with expression '%s' in incarnation '%s' and found a near match '%s' that can't be cloned...",obsname.Data(),expression.Data(),incarnation.Data(),match->getActiveExpression().Data()).Data());
        }      
        clone->SetName(incname);
        clone->fIsManaged = true;
        clone->finalize();
        TQObservable::manager.activeSet->Add(clone);
        DEBUGclass("cloned '%s' for '%s'",match->GetName(),clone->GetName());
        return clone;
      }
      DEBUGclass("existing observable '%s' is initialized but factory exists",match->GetName());
    } else {
      DEBUGclass("returning unused match with name '%s' and compiled expression '%s'",match->GetName(),match->getCompiledExpression(NULL).Data());
      return match;
    }
  }
  DEBUGclass("no match found for '%s' in set '%s' -- creating new observable from expression '%s' (incarnation is '%s')",obsname.Data(),TQObservable::manager.activeSet->GetName(),expression.Data(),incarnation.Data());
  
  TQObservable* retObs = TQObservable::manager.createObservable(origexpr, tags); 
  if (!retObs) {
    tags->printTags();
    ERRORclass("Failed to get observable from expression '%s'. Tags/aliases used have been printed above.",origexpr.Data());
    ERRORclass("The following steps have been attempted:");
    ERRORclass("1) Trying to find a) existing observable with name matching '%s' or",incname.Data());
    ERRORclass("                  b) already existing and initialized observable with activeExpression matching '%s' or",incarnation.Data()); 
    ERRORclass("                  c) already existing and UNinitialized observable expression matching '%s' or",expression.Data()); 
    ERRORclass("2) Trying to find existing observable instances which would yield a compiledExpression matching '%s' if initialized with the provided tags (tags see above). either such an observable was not found or the best match has a factory (in which case we do not clone an existing instance but ask the factory to create a new instance.",incarnation.Data());
    ERRORclass("3) No factory managed to produce a working observable instance");   
  }
  return retObs;
  
}

//______________________________________________________________________________________________

bool TQObservable::hasFactory() const {
  return false;
}

//______________________________________________________________________________________________

TString TQObservable::makeObservableName(const TString& name){
  // convert a string into a valid observable name
  TString tmpname(name);
  tmpname.ReplaceAll("!","NOT");
  tmpname.ReplaceAll("&&","AND");
  tmpname.ReplaceAll("||","OR");
  tmpname.ReplaceAll("[","_");
  tmpname.ReplaceAll("]","_");
  tmpname.ReplaceAll("(","_");
  tmpname.ReplaceAll(")","_");
  tmpname.ReplaceAll("/",":");
  return TQStringUtils::makeValidIdentifier(tmpname,"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890._*+-><=:/");
}

//______________________________________________________________________________________________

bool TQObservable::initialize(TQSample * sample){
  // initialize this observable
  if(this->isInitialized()) return true;
  if (fSample){
    // we can't do anything if we already have a sample
    if(fSample == sample) return true;
    else throw std::runtime_error(TString::Format("observable '%s' already initialized to sample '%s'",this->GetName(),fSample->getPath().Data()).Data());
  }

  // the sample to use has to be valid
  if(!sample) return false;
  fSample = sample;
 
  if(!this->initializeSelf()){
    this->fIsInitialized = false;
    this->fSample = NULL;
    throw std::runtime_error(TString::Format("unable to initialize observable '%s'",this->GetName()).Data());
  }

  this->fIsInitialized = true;
  return true;
}

//______________________________________________________________________________________________

bool TQObservable::isInitialized() const {
  // return true if this observable is initialized, false otherwise
  return this->fIsInitialized;
}

//______________________________________________________________________________________________

bool TQObservable::finalize() {
  // finalize this observable, return the tree token
  DEBUGclass("finalizing '%s'",this->GetName());
  if(!this->isInitialized()){
    return true;
  }
 
  if (!this->fSample){
    ERRORclass("no sample assigned to this observable: '%s'",this->GetName());
    return false;
  }
 
  this->fIsInitialized = !(this->finalizeSelf());
  if(this->fIsInitialized){
    ERRORclass("unable to finalize observable '%s'",this->GetName());
  }

  this->fSample = 0;
 
  return (!this->fIsInitialized);
}


//______________________________________________________________________________________________

TQObservable::~TQObservable() {
  // standard destructor
  if(this->fIsManaged){
    TQObservable::manager.activeSet->Remove(this);
  }
}

//______________________________________________________________________________________________

void TQObservable::print() const {
  // print the contents of this observable and its associated branches
  std::cout << TQStringUtils::makeBoldYellow(this->getExpression()) << std::endl;
}

//______________________________________________________________________________________________

TString TQObservable::getCompiledExpression(TQTaggable* tags) const {
  // retrieve the expression string
  return this->compileExpression(getExpression(),tags);
}

//______________________________________________________________________________________________

TString TQObservable::replaceBools(TString expression){
  expression = TQStringUtils::replaceEnclosed(expression,"true", "(1==1)"," /*+-?!|&\n\t ()[]{}=:");
  expression = TQStringUtils::replaceEnclosed(expression,"false","(1==0)"," /*+-?!|&\n\t ()[]{}=:");
  return expression;
}

//______________________________________________________________________________________________

TString TQObservable::unreplaceBools(TString expression){
  expression.ReplaceAll("(1==1)","true");
  expression.ReplaceAll("(1==0)","false");
  return expression;
}

//______________________________________________________________________________________________

TString TQObservable::compileExpression(const TString& input, TQTaggable* tags, bool replaceBools) {
  // compile an expression, evaluating all evaluable expressions beforehand
  // the following examples document the behaviour of this function:
  //
  // - newlines and tabs, and double spaces are removed, e.g.
  // "Mjj > 600." 
  // becomes
  // "Mjj > 600."
  //
  // - placeholders are replaced by their respective values based on the
  // instance of TQTaggable provided as second argument and all its base
  // objects, e.g.
  // "$(bTagWeightName)"
  // becomes
  // "bTagEventWeight"
  // if the TQTaggable object or any of its parent objects carries the tag
  // bTagWeightName = bTagEventWeight
  //
  // - boolean strings are replaced by their numerical equivalents, i.e.
  // "true" is replaced by "1."
  // "false" is replaced by "0."
  // 
  // - inline conditionals in curly braces are evaluated if possible, e.g.
  // "{ $(channel) == 'ee' ? true : false }"
  // becomes "1." if the tag "channel" has the value "ee", false otherwise.
  // if evaluation is not possible because the conditional contains
  // unresolved strings, the curly braces are replaced by round ones.
  // 
  // - single quotes are replaced by double quotes, e.g.
  // "{ $(channel) == "ee" ? true : false }"
  // is identical to 
  // "{ $(channel) == 'ee' ? true : false }"
  // 
  // if the compilation fails and the result still contains unresolved
  // variables (or any dollar signs, to that extend) an error is thrown.
  //
  if(input.IsNull()) return input;
  TString expression = tags ? tags->replaceInTextRecursive(input,"~",false) : input;
  expression.ReplaceAll("'","\"");
  if(replaceBools){
    expression = TQObservable::replaceBools(expression);
  }

  //  DEBUGclass("input is '%s', obtained from '%s'",expression.Data(),input.Data());

  TString retval("");
  while (!expression.IsNull()) {
    // loop over the expressions '{ ... }' until end of string is reached
    //    DEBUGclass("next iteration, remainder is '%s'",expression.Data());
    TQStringUtils::readUpTo(expression,retval,"{");
 
    if(expression.Length() < 1) return TQStringUtils::minimize(retval);
 
    size_t endpos = TQStringUtils::findParenthesisMatch(expression,0,"{","}");
    if(endpos > (size_t)expression.Length()){
      ERRORclass("parenthesis mismatch in expression '%s'",expression.Data());
      return "";
    }
 
    size_t posIf = TQStringUtils::find(expression,"?");
    if (posIf < (size_t)expression.Length()){
      // if the expression contains '?', we assume it's an if-statement
      //      DEBUGclass("found sub-expression: '%s'",TString(expression(0,endpos+1)).Data());
      TString ifExpr(expression(1, posIf-1));
      if(TQStringUtils::hasUnquotedStrings(ifExpr)){
        // if there are unquoted strings, we can't compile it
        // for now, we assume that they are branch names
        // we replace the curly braces with round ones and skip evaluation
        // the TTreeFormula will hopefully deal with it
        //        DEBUGclass("'if'-statement '%s' seems to have unquoted strings, skipping",ifExpr.Data());
        retval.Append("(");
        TString subExpr(expression(1,endpos-1));
        retval.Append(compileExpression(subExpr,NULL,replaceBools));
        retval.Append(")");
      } else {
        // we have found something valid that we can evaluate
        //        DEBUGclass("testing 'if'-statement: '%s'",ifExpr.Data());
        TFormula formula("if", ifExpr);
        TString thenExpr, elseExpr;
        size_t posElse = posIf;
        while(posElse == posIf || expression[posElse-1] == '\\'){
          posElse = TQStringUtils::findFree(expression,":","{}()[]",posElse+1);
        }
        if (posElse < (size_t)expression.Length()) {
          // we have found an ':', indicating an "else"-block
          thenExpr = expression(posIf+1, posElse-posIf-1);
          elseExpr = expression(posElse+1, endpos-posElse-1);
          //          DEBUGclass("found 'then' and 'else' statements: '%s'/'%s'",thenExpr.Data(),elseExpr.Data());
        } else {
          // no "else"-block was found, only "then"
          thenExpr = expression(posIf+1,endpos-posIf-1);
          elseExpr = "";
          //          DEBUGclass("found 'then'-statement: '%s'",thenExpr.Data());
        }
        if (formula.Eval(0.)){
          // the formula evaluated to "true" - we can use the "then"-block
          //          DEBUGclass("using THEN statement");
          retval.Append(compileExpression(thenExpr,NULL,replaceBools));
        } else {
          // the formula evaluated to "false" - we should use the "else"-block
          //          DEBUGclass("using ELSE statement");
          retval.Append(compileExpression(elseExpr,NULL,replaceBools));
        }
      }
    } else {
      // this is odd - we have curly braces, but nothing that looks like an if-statement
      // we assume that the user meant to use round braces instead
      // we replace them and continue
      retval.Append("(");
      TString subExpr(expression(1,endpos-1));
      retval.Append(compileExpression(subExpr,NULL,replaceBools));
      retval.Append(")");
    }
 
    expression.Remove(0,endpos+1);
    TQStringUtils::removeLeadingBlanks(expression);
  }

  if(retval.Index("$") != kNPOS){
    size_t start = retval.Index("$");
    size_t stop;
    if(retval[start+1] == '('){
      stop = TQStringUtils::findParenthesisMatch(retval,start+1,"(",")")+1;
    } else {
      stop = TQStringUtils::findFirstNotOf(retval,TQStringUtils::defaultIDchars,start+1);
    }
    TString var(retval(start,stop-start));
    TString err = TString::Format("unresolved variable '%s' in expression '%s'!",var.Data(),retval.Data());
    if(tags){
      ERRORclass(err+" Available tags are listed above.");
      tags->printTags("l");
    }
    throw std::runtime_error(err.Data());
  }
 
  //  DEBUGclass("expression compilation complete");
  retval.ReplaceAll("\\","");
  return TQStringUtils::minimize(retval);
}

//__________________________________________________________________________________|___________

TString TQObservable::getName() const {
  // retrieve the name of this object
  return this->fName;
}

//__________________________________________________________________________________|___________

void TQObservable::setName(const TString& newName) {
  // set the name of this object
  this->fName = newName;
}

//__________________________________________________________________________________|___________

const TString& TQObservable::getNameConst() const {
  // retrieve a const reference to the name of this object
  return this->fName;
}


//__________________________________________________________________________________|___________

bool TQObservable::isSetup() const {
  // return true if this observable is setup and ready to initialize
  // false otherwise
  return true;
}

//__________________________________________________________________________________|___________

void TQObservable::allowErrorMessages(bool val){
  TQObservable::gAllowErrorMessages = val;
}

//______________________________________________________________________________________________

const TString& TQObservable::getExpression() const {
  // retrieve the expression associated with this observable
  return this->fName;
}

//______________________________________________________________________________________________

bool TQObservable::hasExpression() const {
  // check if this observable type knows expressions (default false)
  DEBUGclass("doesn't know expressions");
  return false;
}

//______________________________________________________________________________________________

TQObservable* TQObservable::getClone() const {
  // retrieve a clone of this observable
  TQObservable* obs = NULL;
  if(this->hasExpression()){
    obs = this->tryCreateInstanceVirtual(this->getExpression());
  }
  if(!obs){
    obs = (TQObservable*)(this->Clone());
  }
  if(obs->isInitialized()) obs->finalize();
  return obs;
}

//______________________________________________________________________________________________

void TQObservable::setExpression(const TString&/*expr*/){
  // set the expression to a given string
  // not implemented for the base class
  ERRORclass("this function is not implemented. did you forget to implement 'virtual void YOURCLASS::setExpression(const TString)'?");
}

//______________________________________________________________________________________________

TQObservable* TQObservable::tryCreateInstanceVirtual (const TString&) const {
  // this is the factory function for each observable type
  // in the default implementation, this always returns NULL
  // because there is no fully general observable factory
  return NULL;
}

//______________________________________________________________________________________________

TString TQObservable::getActiveExpression() const {
  // retrieve the expression string
  return this->getExpression();
}
