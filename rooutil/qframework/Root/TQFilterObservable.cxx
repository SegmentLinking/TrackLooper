#include "QFramework/TQFilterObservable.h"
#include <limits>
#include <vector>
#include "TTreeFormula.h"
#include "TString.h"
#include "QFramework/TQStringUtils.h"
#include "QFramework/TQUtils.h"

// uncomment the following line to enable debug printouts
//#define _DEBUG_
// you can perform debug printouts with statements like this
// DEBUG("error number %d occurred",someInteger);

// be careful to not move the _DEBUG_ flag behind the following line
// otherwise, it will show no effect
#include "QFramework/TQLibrary.h"

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQFilterObservable
//
// TODO: write documentation
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQFilterObservable)

//______________________________________________________________________________________________

TQFilterObservable::TQFilterObservable(){
  // default constructor
  DEBUGclass("default constructor called");
}

//______________________________________________________________________________________________

TQFilterObservable::~TQFilterObservable(){
  // default destructor
  DEBUGclass("destructor called");
} 


//______________________________________________________________________________________________

TObjArray* TQFilterObservable::getBranchNames() const {
  // retrieve the list of branch names 
  // ownership of the list belongs to the caller of the function
  DEBUGclass("retrieving branch names");

  TObjArray* bnames = new TObjArray();
  if (!this->fCutObs) {
    ERROR("there is no fCutObs");
  } else {
      TCollection* c = this->fCutObs->getBranchNames();
      if(c){
        c->SetOwner(false);
        bnames->AddAll(c);
        delete c;
      }
  }
  if (!this->fValueObs) {
    ERROR("there is no fValueObs");
  } else {
      TCollection* c = this->fValueObs->getBranchNames();
      if(c){
        c->SetOwner(false);
        bnames->AddAll(c);
        delete c;
      }
  }
  return bnames;
}

//______________________________________________________________________________________________
bool TQFilterObservable::makeCache() const {
  // determine all return values for getValueAt and getNevaluations if this hasn't been done yet
  
  if (this->fCachedEntry == this->getCurrentEntry() && this->fCachedEntry>0) return true; //nothing to do here, the cache is already up-to-date
  
  this->fCachedValues.clear();
  //we have to consider three cases: scalar value + scalar cut, vector value + scalar cut, vector value + vector cut
  if (!this->fCutObs) {
    ERRORclass("No cut observable present! (Was this observable correctly initialized?)");
    return false;
  }
  if (!this->fValueObs) {
    ERRORclass("No value observable present! (Was this observable correctly initialized?)");
    return false;
  }
  
  //case 1+2:
  if (this->fCutObs->getObservableType() == TQObservable::ObservableType::scalar) {
    if (std::fabs(this->fCutObs->getValue()) > 2*std::numeric_limits<double>::epsilon()) { //check for cut being passed
      for (int i=0; i<this->fValueObs->getNevaluations(); i++) {
        this->fCachedValues.push_back(this->fValueObs->getValueAt(i));
      }
    }
  } else if (this->fValueObs->getObservableType() == TQObservable::ObservableType::vector) {//case 3: vector value + vector cut
    int nEval = this->fCutObs->getNevaluations();
    if (nEval != this->fValueObs->getNevaluations()) {
      ERRORclass(TString::Format("Cut and value observables have different number of evaluations in TQFilterObservable with expression '%s'",this->getExpression().Data()).Data());
      return false;
    }
    for (int i=0; i<nEval; i++) {
      if (std::fabs(this->fCutObs->getValueAt(i)) > 2*std::numeric_limits<double>::epsilon()) { //check for cut being passed
        this->fCachedValues.push_back(this->fValueObs->getValueAt(i));
      }
    }
  } else { //none of the cases 1-3 applies (think of scalar value + vector cut: what should we do in this case?)
    ERRORclass("Illegal combination of observable types: Trying to filter a scalar value based on a vector type cut is highly ambiguous!");
    return false;
  }
  this->fCachedEntry = this->getCurrentEntry();
  return true;
}


double TQFilterObservable::getValue() const {
  // in the rest of this function, you should retrieve the data and calculate your return value
  // here is the place where most of your custom code should go
  // a couple of comments should guide you through the process
  // when writing your code, please keep in mind that this code can be executed several times on every event
  // make your code efficient. catch all possible problems. when in doubt, contact experts!
  
  // here, you should calculate your return value
  // of course, you can use other data members of your observable at any time
  /* example block for TTreeFormula method:
  const double retval = this->fFormula->Eval(0.);
  */
  /* exmple block for TTree::SetBranchAddress method:
  const double retval = this->fBranch1 + this->fBranch2;
  */

  throw std::runtime_error("Called vector type observable TQFilterObservable in scalar context.");
  return -999.;
}

int TQFilterObservable::getNevaluations() const {
  
  if (!this->makeCache()) {
    throw std::runtime_error(TString::Format("Failed to create return value(s) in TQFilterObservable with expression '%s'",this->getExpression().Data()).Data());
    return -1;
  }
  return this->fCachedValues.size();
  /* Frank's original code
  if (this->fCutObs->getValue() > 0) {
    return 0;
  } else {
    return 1;
  }
  */
}

double TQFilterObservable::getValueAt(int index) const {
  if (index >= TQFilterObservable::getNevaluations()) {
    throw std::runtime_error("Caught attempt to evaluate TQFilterObservable out of bounds!");
    return -999.;
  } else {
    return this->fCachedValues.at(index);
  }
}
//______________________________________________________________________________________________

TQFilterObservable::TQFilterObservable(const TString& expression):
TQObservable(expression)
{
  // constructor with expression argument
  DEBUGclass("constructor called with '%s'",expression.Data());
  // the predefined string member "expression" allows your observable to store an expression of your choice
  // this string will be the verbatim argument you passed to the constructor of your observable
  // you can use it to choose between different modes or pass configuration options to your observable
  this->SetName(TQObservable::makeObservableName(expression));
  this->setExpression(expression);
}

//______________________________________________________________________________________________

const TString& TQFilterObservable::getExpression() const {
  // retrieve the expression associated with this observable
  
  return this->fExpression; 
}

//______________________________________________________________________________________________

bool TQFilterObservable::hasExpression() const {
  // check if this observable type knows expressions
  return true;
}

//______________________________________________________________________________________________
void TQFilterObservable::setExpression(const TString& expr){
  // set the expression to a given string
  if (!this->parseExpression(expr)) this->fExpression = expr;
  else this->fExpression = TString("Filter("+this->fValueString+","+this->fCutString+")");
}
//______________________________________________________________________________________________

bool TQFilterObservable::parseExpression(const TString& expr){
  // parse the expression
  if (!expr.BeginsWith("Filter(") || !expr.EndsWith(")")) {
    return false;
  }

  TString exprArgs = expr(7, expr.Length() - 8); // remove opening and closing parenthesis
  std::vector<TString> tokens = TQStringUtils::tokenizeVector(exprArgs, ",", true, "()[]{}", "'\"");
  if (tokens.size() != 2) {
    return false;
  }
  this->fValueString = tokens[0];
  this->fCutString = tokens[1];
  
  return true;
}

//______________________________________________________________________________________________

void TQFilterObservable::clearParsedExpression(){
  this->fValueString.Clear();
  this->fCutString.Clear();
}

//______________________________________________________________________________________________

TString TQFilterObservable::getActiveExpression() const {
  // retrieve the expression associated with this incarnation
  return this->fActiveExpression;
}

//______________________________________________________________________________________________

Long64_t TQFilterObservable::getCurrentEntry() const {
  if (this->fCutObs && this->fCutObs->getCurrentEntry()>0) return this->fCutObs->getCurrentEntry();
  if (this->fValueObs && this->fValueObs->getCurrentEntry()>0) return this->fValueObs->getCurrentEntry();
  return -1; //fallback if we can't determine the current entry
}

//______________________________________________________________________________________________

bool TQFilterObservable::initializeSelf(){
  // initialize self - compile container name, construct accessor
  this->fActiveExpression = TQObservable::compileExpression(this->fExpression,this->fSample);
  if (!this->parseExpression(this->fExpression)) {
    return false;
  }
  DEBUGclass("Initializing observable with active expression '%s'",this->fActiveExpression.Data());
  this->fValueObs = TQObservable::getObservable(this->fValueString, this->fSample);
  if (!this->fValueObs) {
      ERROR("Failed to retrieve value-observable.");
      return false;
  }
  if (!this->fValueObs->initialize(this->fSample)) {
      ERROR("Failed to initialize value-observable.");
      return false;
  }

  this->fCutObs = TQObservable::getObservable(this->fCutString, this->fSample);
  if (!this->fCutObs) {
      ERROR("Failed to retrieve cut-observable.");
      return false;
  }
  if (!this->fCutObs->initialize(this->fSample)) {
      ERROR("Failed to initialize cut-observable.");
      return false;
  }


  return true;
}
 
//______________________________________________________________________________________________

bool TQFilterObservable::finalizeSelf(){
  // finalize self - delete accessor
  if (this->fCutObs != NULL) {
        if (!this->fCutObs->finalize()) {
            ERROR("finalizing CutObs failed");
        }
        this->fCutObs = NULL;
  }
  if (this->fValueObs != NULL) {
        if (!this->fValueObs->finalize()) {
            ERROR("finalizing ValueObs failed");
        }
        this->fValueObs = NULL;
  }
  this->clearParsedExpression();
  this->fActiveExpression.Clear();
  return true;
}
//______________________________________________________________________________________________
int TQFilterObservable::registerFactory() {
  TQObservable::manager.registerFactory(TQFilterObservable::getFactory(),true);
  ERROR("registerFactory");
  return 0;
}


// the following preprocessor macro defines an "observable factory"
// that is supposed to create instances of your class based on input
// expressions.

// it should receive an expression as an input and decide whether this
// expression should be used to construct an instance of your class,
// in which case it should take care of this, or return NULL.

// in addition to defining your observable factory here, you need to
// register it with the TQObservable manager. This can either be done
// from C++ code, using the line
//   TQObservable::manager.registerFactory(TQFilterObservable::getFactory(),true);
// or from python code, using the line
//   TQObservable.manager.registerFactory(TQFilterObservable.getFactory(),True)
// Either of these lines need to be put in a location where they are
// executed before observables are retrieved. You might want to 'grep'
// for 'registerFactory' in the package you are adding your observable
// to in order to get some ideas on where to put these!

DEFINE_OBSERVABLE_FACTORY(TQFilterObservable,TString expr){
  // try to create an instance of this observable from the given expression
  // return the newly created observable upon success
  // or NULL upon failure

  // first, check if the expression fits your observable type
  // for example, you can grab all expressions that begin wth "TQFilterObservable:"
  // if this is the case, then we call the expression-constructor
  if (!expr.BeginsWith("Filter(") || !expr.EndsWith(")")) {
    return NULL;
  }

  TString exprArgs = expr(7, expr.Length() - 8); // remove opening and closing parenthesis
  std::vector<TString> tokens = TQStringUtils::tokenizeVector(exprArgs, ",", true, "()[]{}", "'\"");

  if (tokens.size() != 2) {
    return NULL;
  }
  return new TQFilterObservable(expr);
  
  // else, that is, if the expression doesn't match the pattern we
  // expect, we return this is important, because only in doing so we
  // give other observable types the chance to check for themselves
  return NULL;
  // if you do not return NULL here, your observable will become the
  // default observable type for all expressions that don't match
  // anything else, which is probably not what you want...
}

// int _dummy_TQFilterObservableRegisterFactory = TQFilterObservable::registerFactory();
