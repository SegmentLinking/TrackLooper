#include "QFramework/TQConstObservable.h"

#include "TFormula.h"
#include "QFramework/TQStringUtils.h"

// #define _DEBUG_
#include "QFramework/TQLibrary.h"

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQConstObservable
//
// The TQConstObservable is a variant of TQObservable that uses a
// constant expression that does not depend on the event visited and
// does not need any tree access.
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQConstObservable)

//______________________________________________________________________________________________

bool TQConstObservable::initializeSelf(){
	// initialize self
  TString expr = this->getCompiledExpression(this->fSample);
  DEBUGclass("initializing observable with expression '%s'",expr.Data());
  if(TQStringUtils::equal(expr,"nan")){
    this->fValue = std::numeric_limits<double>::quiet_NaN();
  } else {
    TFormula f("const",expr);
    this->fValue = f.Eval(1.);
  }
  return true;
}

//______________________________________________________________________________________________

bool TQConstObservable::finalizeSelf(){
  // finalize self - nothing to do
  return true;
}

//______________________________________________________________________________________________

TQConstObservable::TQConstObservable(){
  // default constructor
}

//______________________________________________________________________________________________

TQConstObservable::TQConstObservable(const TString& expression) :
  TQObservable(expression)
{
  // constructor taking expression arguments
  DEBUGclass("constructor called with expression '%s'",expression.Data());
  this->setExpression(expression);
}

//______________________________________________________________________________________________

double TQConstObservable::getValue() const {
  // retrieve the constant value
  return this->fValue;
}

//______________________________________________________________________________________________

TObjArray* TQConstObservable::getBranchNames() const {
  // the const observable is const, it doesn't need any branches
  return NULL;
}

//______________________________________________________________________________________________

TQConstObservable::~TQConstObservable(){
  // default destructor
}

//______________________________________________________________________________________________

Long64_t TQConstObservable::getCurrentEntry() const {
  // the const observable doesn't know about the entry, this function always returns -1
  return -1;
}

//______________________________________________________________________________________________

const TString& TQConstObservable::getExpression() const {
  // retrieve the expression associated with this observable
  DEBUGclass("retrieving expression '%s'",this->fExpression.Data());
  return this->fExpression;
}

//______________________________________________________________________________________________

bool TQConstObservable::hasExpression() const {
  // check if this observable type knows expressions
  return true;
}

//______________________________________________________________________________________________

void TQConstObservable::setExpression(const TString& expr){
  // set the expression to a given string
  DEBUGclass("setting expression to '%s'",expr.Data());
  this->fExpression = TQStringUtils::compactify(expr);
}


//______________________________________________________________________________________________

DEFINE_OBSERVABLE_FACTORY(TQConstObservable,TString expression){
  // try to create an instance of this observable from the given expression
  if(TQStringUtils::removeLeadingText(expression,"Const:") || TQStringUtils::isValidIdentifier(expression,"0123456789./*+- \t<>=")){
    return new TQConstObservable(expression);
  }
  return NULL;
}
