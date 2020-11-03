#include "QFramework/TQMultiObservable.h"
#include "TROOT.h"
#include <limits>
#include <stdexcept>

// uncomment the following line to enable debug printouts
//#define _DEBUG_
// you can perform debug printouts with statements like this
// DEBUG("error number %d occurred",someInteger);

// be careful to not move the _DEBUG_ flag behind the following line
// otherwise, it will show no effect
#include "QFramework/TQLibrary.h"

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQMultiObservable
//
// The TQMultiObservable is a sort of meta-observable that will act
// upon other observables and combine their respective
// output. Creation of such an observable can be triggered by
// prepending 'Multi:' to any expression, or by using brackets [...]
// in such a way that they enclose an unquoted string. Each segment
// enclosed in such brackets will be interpreted as the name of some
// other observable, which will be evaluated and inserted in this
// expression at runtime. Please not that TQMultiObservable does not
// provide any direct data access, which means that you shall not use
// any branch names in this expression directly, but only other
// observable names.
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQMultiObservable)

//______________________________________________________________________________________________

TQMultiObservable::TQMultiObservable(){
  // default constructor
  DEBUGclass("default constructor called");
}

//______________________________________________________________________________________________

TQMultiObservable::~TQMultiObservable(){
  // default destructor
  DEBUGclass("destructor called");
} 

//______________________________________________________________________________________________

int TQMultiObservable::getNevaluations() const {
  // returns the number of valid evaluations of this observable for the current event
  // returns -1 if a problem is found. It is left to the calling code to decide if an error should be thrown
  DEBUGclass("getNevaluations called");
  if (this->fObservableType != TQObservable::ObservableType::vector) return 1; //for non-vector valued instances we know the number of evaluations
  int nEvals = -1;
  for (size_t i=0; i<this->fObservables.size(); ++i) {
    if (this->fObservables[i]->getObservableType() != TQObservable::ObservableType::vector) continue;
    int iEvals = this->fObservables[i]->getNevaluations();
    if (nEvals < 0) nEvals = iEvals;
    if (nEvals != iEvals) return -1; //this should not happen, only combinations with vector observables of equal "length" are allowed!
  }
  DEBUGclass("returning");
  return nEvals; //might be -1, which means we have an error, e.g. this instance is marked as a vector observable but none of its sub observables is.
}

//______________________________________________________________________________________________

double TQMultiObservable::getValueAt(int index) const {
  // retrieve the value of this observable
  DEBUGclass("entering function");
  if (index!=0 && this->fObservableType != TQObservable::ObservableType::vector) {throw std::runtime_error("Attempt to retrieve value from illegal index in observable."); return -999.;} //protection for scalar variants
  const int nEntries = this->getNevaluations();
  if (index>=nEntries) throw std::runtime_error(TString::Format("Requested index '%d' is not available for TQMultiObservable with expression '%s' (%d entries available)",index,this->getExpression().Data(),nEntries).Data());
    
  const int entry = this->getCurrentEntry();
  if(entry < 0 || entry != this->fCachedEntry){
    DEBUGclass("calculating value of '%s'",this->getActiveExpression().Data());
    
    if (nEntries < 0) throw std::runtime_error(TString::Format("Illegal number of evaluations for TQMultiObservable with expression '%s'. Are you sure all vector observables are guaranteed to have the same number of evaluations?",this->getExpression().Data()).Data() );
    
    this->fCachedValue.resize(std::max(nEntries,(int)this->fCachedValue.size())); //ensure that the caching vector is sufficiently large
    //calculate+cache all elements of the results vector
    for (int myIndex = 0; myIndex<nEntries; myIndex++) {
      for(size_t i=0; i<this->fObservables.size(); ++i){
        DEBUGclass("evaluating observable %d: %s",(int)(i),this->fObservables[i]->getActiveExpression().Data());
        this->fFormula->SetParameter(i,this->fObservables[i]->getValueAt( this->fObservables[i]->getObservableType() == TQObservable::ObservableType::vector ? myIndex : 0 ));
        DEBUGclass("setting parameter [%d]=%g (from '%s')",(int)(i),this->fObservables[i]->getValueAt(this->fObservables[i]->getObservableType() == TQObservable::ObservableType::vector ? myIndex : 0),this->fObservables[i]->getActiveExpression().Data());
      }
      this->fCachedValue[myIndex] = this->fFormula->Eval(0.);
      DEBUGclass("value of '%s' is %g",this->fFormula->GetTitle(),this->fCachedValue[myIndex]);
    }
    this->fCachedEntry = entry; 
  } else {
    DEBUGclass("skipping reevalution for event %d",entry);
  }
  return this->fCachedValue.at(index);
  
}



double TQMultiObservable::getValue() const {
  //forward to the implementation capable of handling multiple values.
  if (this->fObservableType != TQObservable::ObservableType::vector) return this->getValueAt(0);
  else throw std::runtime_error(TString::Format( "Caught attempt to perform scalar evaluation on vector valued instance of TQMultiObservable with expression '%s'",this->getExpression().Data() ).Data());
}

//______________________________________________________________________________________________

bool TQMultiObservable::initializeSelf(){
  // initialize this observable
  DEBUGclass("starting initialization");
  this->fActiveExpression = TQObservable::unreplaceBools(this->compileExpression(this->fExpression,this->fSample,true));
  DEBUGclass("initializing observable with expression '%s'",this->fActiveExpression.Data()); 
  this->parseExpression(this->fActiveExpression);
  bool retval = true;
  if(!this->fParsedExpression.IsNull()){
    const TString expr(this->compileExpression(this->fParsedExpression,this->fSample,true));
    if(!gAllowErrorMessages) TQLibrary::redirect_stdout("/dev/null");
    //printf("\n<TQMultiObservalbe::initializeSelf>   expr = %s\n\n", expr.Data());
    this->fFormula = new TFormula(this->GetName(),expr);
    if(!gAllowErrorMessages) TQLibrary::restore_stdout();
#if ROOT_VERSION_CODE >= ROOT_VERSION(6,04,00)
    if(!this->fFormula->IsValid()){
      throw std::runtime_error(TString::Format("unable to initialize formula with expression '%s'",expr.Data()).Data());
      return false;
    }
#endif
  } else {
    WARNclass("observable '%s' has Null-Expression!",this->GetName());
    return false;
  }
  this->fObservableType = TQObservable::ObservableType::scalar; //default after initialization, is set to ::vector in the following if needed
  for(size_t i=0; i<this->fObservables.size(); ++i){
    if(!this->fObservables[i]->initialize(this->fSample)) {
      ERRORclass("Failed to initialize sub-observable created from expression '%s'",this->fObservables[i]->getExpression().Data());
      retval = false;
    }
    if (this->fObservables[i]->getObservableType() == TQObservable::ObservableType::vector) this->fObservableType = TQObservable::ObservableType::vector; //if at least one sub-observable is a vector observable then this also becomes a vector observable.
  }
  this->fCachedEntry = -999;
  DEBUGclass("successfully initialized");
  return retval;
}

//______________________________________________________________________________________________

bool TQMultiObservable::finalizeSelf(){
  // initialize this observable
  DEBUGclass("finalizing '%s'",this->GetName());
  bool ok = true;
  for(size_t i=0; i<this->fObservables.size(); ++i){
    if(!(this->fObservables[i]->finalize())){
      ok = false;
    }
  }
  if(this->fFormula) delete this->fFormula;
  this->fFormula = NULL;
  this->fActiveExpression.Clear();
  this->fObservables.clear();
  return ok;
}

//______________________________________________________________________________________________

Long64_t TQMultiObservable::getCurrentEntry() const {
  // retrieve the current entry from the tree
  if(this->fObservables.size() == 0) return -1;
  for (size_t i = 0; i<this->fObservables.size(); i++) {
    if (this->fObservables[i]->getCurrentEntry() >= 0) return this->fObservables[i]->getCurrentEntry();
  }
  return -1;
}

//______________________________________________________________________________________________
TObjArray* TQMultiObservable::getBranchNames() const {
  // retrieve the list of branch names for this observable
  // ownership of the list belongs to the caller of the function
  DEBUGclass("retrieving branch names");
  TObjArray* bnames = new TObjArray();
  for(size_t i=0; i<this->fObservables.size(); ++i){
    TQObservable* obs =this->fObservables[i];
    if(!obs) throw std::runtime_error("encountered invalid sub-observable!");
    DEBUGclass("retrieving branches of observable '%s' of class '%s'",obs->getExpression().Data(),obs->ClassName());
    TCollection* c = obs->getBranchNames();
    if(c){
      c->SetOwner(false);
      bnames->AddAll(c);
      delete c;
    }
  }
  DEBUGclass("returning");
  return bnames;
}

//______________________________________________________________________________________________

TQMultiObservable::TQMultiObservable(const TString& expression):
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

const TString& TQMultiObservable::getExpression() const {
  // retrieve the expression associated with this observable
  return this->fExpression;
}

//______________________________________________________________________________________________

bool TQMultiObservable::hasExpression() const {
  // check if this observable type knows expressions
  return true;
}

//______________________________________________________________________________________________

void TQMultiObservable::setExpression(const TString& expr){
  // set the expression to a given string
  this->fExpression = expr;
}
//______________________________________________________________________________________________

bool TQMultiObservable::parseExpression(const TString& expr){
  // parse the expression
  this->clearParsedExpression();
  TString expression(expr);
  int parcnt = 0;
  while(!expression.IsNull()){
    DEBUGclass("entering next parsing cycle: expr='%s', remainder='%s'",this->fParsedExpression.Data(),expression.Data());
    TQStringUtils::readUpTo(expression,this->fParsedExpression,"[");
    if(expression.IsNull()) break;
    TString subExpr = "";
    if(TQStringUtils::readBlock(expression,subExpr,"[]") < 1){
      throw std::runtime_error(TString::Format("no sub expression block [...] found in '%s'",expression.Data()).Data());
      return false;
    }
    TQObservable* obs = TQObservable::getObservable(subExpr,this->fSample);
    if(obs){
      this->fParsedExpression.Append(TString::Format("[%d]",parcnt));
      this->fObservables.push_back(obs);
      parcnt++;
    } else {
      this->fParsedExpression.Append(TString::Format("0."));
      throw std::runtime_error(TString::Format("cannot parse expression '%s', unable to retrieve observable with expression '%s'",this->fParsedExpression.Data(),subExpr.Data()).Data());
    }
    DEBUGclass("result of parsing cycle: expr='%s', subExpr='%s', remainder='%s'",this->fParsedExpression.Data(),subExpr.Data(),expression.Data());
  }
  if(!expression.IsNull()){
    throw std::runtime_error(TString::Format("there was an undefined error while parsing expression '%s'",this->fParsedExpression.Data()).Data());
    return false;
  }

  return true;
}

//______________________________________________________________________________________________

void TQMultiObservable::clearParsedExpression(){
  // clear the current expression
  this->fParsedExpression.Clear();
  this->fObservables.clear();
}

//______________________________________________________________________________________________

TString TQMultiObservable::getParsedExpression(){
  // clear the current expression
  return this->fParsedExpression;
}

//______________________________________________________________________________________________

TString TQMultiObservable::getActiveExpression() const {
  // retrieve the expression associated with this incarnation
  return this->fActiveExpression;
}

//______________________________________________________________________________________________

TQObservable* TQMultiObservable::getObservable(int idx){
  // return the sub-observable with the given index
  return this->fObservables.at(idx);
}

//______________________________________________________________________________________________

DEFINE_OBSERVABLE_FACTORY(TQMultiObservable,TString expression){
  // try to create an instance of this observable from the given expression
  if(TQStringUtils::removeLeadingText(expression,"Multi:") ||
     TQStringUtils::hasTFormulaParameters(expression)) { //check if there are [] with non-numerical characters (0-9) in between
    return new TQMultiObservable(expression);
  }
  return NULL;
}

