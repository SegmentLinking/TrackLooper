#include "QFramework/TQVectorAuxObservable.h"
#include <limits>

// uncomment the following line to enable debug printouts
// #define _DEBUG_
// you can perform debug printouts with statements like this
// DEBUG("error number %d occurred",someInteger);

// be careful to not move the _DEBUG_ flag behind the following line
// otherwise, it will show no effect
#include "QFramework/TQLibrary.h"


////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQVectorAuxObservable
//
// TODO: write documenation
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQVectorAuxObservable)

//______________________________________________________________________________________________

TQVectorAuxObservable::TQVectorAuxObservable(){
  // default constructor
  DEBUGclass("default constructor called");
}

//______________________________________________________________________________________________

TQVectorAuxObservable::~TQVectorAuxObservable(){
  // default destructor
  DEBUGclass("destructor called");
} 

//______________________________________________________________________________________________

double TQVectorAuxObservable::getValue() const {
  // value retrieval function, called on every event for every cut and histogram this observable is used in
  DEBUGclass("entering function");

  if (!this->fSubObservable) {throw std::runtime_error("Cannot retrieve value without valid sub observable"); return -99999.;}
  
  if ( !(this->getCurrentEntry() < 0) && this->getCurrentEntry() == this->fCachedEntry) {
    //we can simply use the cached value
    return this->fCachedValue;
  }
  double retval;
  int nEntries = this->fSubObservable->getNevaluations();
  if (nEntries < 1) {
    this->fCachedValue = 0.;
    this->fCachedEntry = this->getCurrentEntry();
    return this->fCachedValue;
  }
  
  int at = -1; //only used for 'AT' mode
  switch (this->fOperation) {
    case AND: 
      retval = 1.;
      for (int i=0; i<nEntries; i++) {
        if (std::fabs(this->fSubObservable->getValueAt(i)) < 2* std::numeric_limits<double>::epsilon()) {retval = 0.; break;}
      }
      break;
    case OR: 
      retval = 0.;
      for (int i=0; i<nEntries; i++) {
        if (std::fabs(this->fSubObservable->getValueAt(i)) > 2* std::numeric_limits<double>::epsilon()) {retval = 1.; break;}
      }
      break;
    case SUM: 
      retval = 0.;
      for (int i=0; i<nEntries; i++) {
        retval += this->fSubObservable->getValueAt(i);
      }
      break;
    case SUMABS: 
      retval = 0.;
      for (int i=0; i<nEntries; i++) {
        retval += std::fabs(this->fSubObservable->getValueAt(i));
      }
      break;
    case AVG: 
      retval = 0.;
      for (int i=0; i<nEntries; i++) {
        retval += std::fabs(this->fSubObservable->getValueAt(i));
      }
      retval /= nEntries;
      break;
    case LEN: 
      retval = nEntries;
      break;
    case MAX: 
      retval = std::numeric_limits<double>::min();
      for (int i=0; i<nEntries; i++) {
        retval = std::max(this->fSubObservable->getValueAt(i),retval);
      }
      break;
    case MIN: 
      retval = std::numeric_limits<double>::max();
      for (int i=0; i<nEntries; i++) {
        retval = std::min(this->fSubObservable->getValueAt(i),retval);
      }
      break;
    case MAXABS: 
      retval = std::numeric_limits<double>::min();
      for (int i=0; i<nEntries; i++) {
        retval = std::max(std::fabs(this->fSubObservable->getValueAt(i)),retval);
      }
      break;
    case MINABS: 
      retval = std::numeric_limits<double>::max();
      for (int i=0; i<nEntries; i++) {
        retval = std::min(std::fabs(this->fSubObservable->getValueAt(i)),retval);
      }
      break;
    case NORM: 
      retval = 0;
      for (int i=0; i<nEntries; i++) {
        retval += pow(this->fSubObservable->getValueAt(i),2.);
      }
      retval = sqrt(retval);
      break;
    case PROD: 
      retval = 1.;
      for (int i=0; i<nEntries; i++) {
        retval *= this->fSubObservable->getValueAt(i);
      }
      break;
    case NTRUE: 
      retval = 0;
      for (int i=0; i<nEntries; i++) {
        retval += (std::fabs(this->fSubObservable->getValueAt(i)) > 2* std::numeric_limits<double>::epsilon());
      }
      break;
    case NFALSE: 
      retval = 0;
      for (int i=0; i<nEntries; i++) {
        retval += (std::fabs(this->fSubObservable->getValueAt(i)) < 2* std::numeric_limits<double>::epsilon());
      }
      break;
    case AT:
      at = int(this->fIndexObservable->getValue()); //index observable must be scalar!
      if (at<nEntries && at>=0) {
        retval = this->fSubObservable->getValueAt(at);
      } else {
        throw std::runtime_error(TString::Format("Caught attempt to evaluate the vector observable with expression '%s' out of bounds (requested index: %d, max index: %d)",this->fSubObservable->getExpression().Data(),at,nEntries-1).Data());
        retval = std::numeric_limits<double>::quiet_NaN();
      }
      break;
    default:
      retval = -999999.;
      throw std::runtime_error(TString::Format("Cannot evaluate VectorAux observable with expression '%s': Unknown mode/enum value",this->getExpression().Data()).Data());
  };
  
  this->fCachedValue = retval;
  this->fCachedEntry = this->getCurrentEntry();
  return fCachedValue;
}

//______________________________________________________________________________________________

Long64_t TQVectorAuxObservable::getCurrentEntry() const {
  // retrieve the current entry from the sub observable

  return this->fSubObservable? this->fSubObservable->getCurrentEntry() : (this->fIndexObservable? this->fIndexObservable->getCurrentEntry() : -1);
}

//______________________________________________________________________________________________

TObjArray* TQVectorAuxObservable::getBranchNames() const {
  // retrieve the list of branch names for this observable
  // ownership of the list belongs to the caller of the function
  DEBUGclass("retrieving branch names");
  
  return this->fSubObservable? this->fSubObservable->getBranchNames() : NULL;
}
//______________________________________________________________________________________________

TQVectorAuxObservable::TQVectorAuxObservable(const TString& expression):
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

const TString& TQVectorAuxObservable::getExpression() const {
  // retrieve the expression associated with this observable
  this->fFullExpression = this->fSubObservable? this->fSubObservable->getExpression() : (this->fVecExpression.IsNull()? this->fExpression : this->fVecExpression);
  DEBUGclass("Prepending prefix");
  this->fFullExpression.Prepend("Vec"+TQVectorAuxObservable::getOperationName(this->fOperation) + "(");
  if (this->fIndexObservable) this->fFullExpression.Append(","+this->fIndexObservable->getExpression()); //for 'AT' mode we need a second observable determining the index to retrieve
  else if (!this->fIndexExpression.IsNull()) fFullExpression.Append(","+this->fIndexExpression);
  this->fFullExpression.Append(")");
  DEBUGclass("Returning full expression");
  return this->fFullExpression;
}

//______________________________________________________________________________________________

bool TQVectorAuxObservable::hasExpression() const {
  // check if this observable type knows expressions
  return true;
}

//______________________________________________________________________________________________

void TQVectorAuxObservable::setExpression(const TString& expr){
  // set the expression to a given string
  DEBUGclass("Setting expression for VectorAuxObservable based on specified expression '%s'",expr.Data());
  TString myExpr = expr;
  TQVectorAuxObservable::Operation op = TQVectorAuxObservable::readPrefix(myExpr);
  if (op != TQVectorAuxObservable::Operation::invalid) this->setOperation(op);
  this->fExpression = myExpr;
  DEBUGclass("Set expression for VectorAuxObservable to '%s'",this->fExpression.Data());
}
//______________________________________________________________________________________________

//______________________________________________________________________________________________

TString TQVectorAuxObservable::getActiveExpression() const {
  // retrieve the expression associated with this incarnation
  DEBUGclass("Creating active expression");
  TString aExpr = this->fSubObservable? this->fSubObservable->getActiveExpression() : (this->fVecExpression.IsNull() ? this->fVecExpression : this->fVecExpression);
  aExpr.Prepend(TQVectorAuxObservable::getPrefix(this->fOperation)+"(");
  if (this->fIndexObservable) aExpr.Append(","+this->fIndexObservable->getActiveExpression()); //for 'AT' mode we need a second observable determining the index to retrieve
  else if (!this->fIndexExpression.IsNull()) aExpr.Append(","+this->fIndexExpression);
  aExpr.Append(")");
  DEBUGclass("returning active expression");
  return aExpr;
}

//______________________________________________________________________________________________

bool TQVectorAuxObservable::initializeSelf(){
  // initialize self - compile container name, construct accessor
  DEBUGclass("Initializing VectorAuxObservable with expression '%s'",this->fExpression.Data());
  //the following block should only be relevant for the 'AT' mode:
  if (this->fOperation == AT) {
    std::vector<TString> expressions = TQStringUtils::split(this->fExpression,",","([{\"'",")]}\"'"); //split the expression for the index observable from the rest of the expression
    if (expressions.size()<1) {
      ERRORclass("Failed to parse (split) expression '%s'",this->fExpression.Data());
      throw std::runtime_error("This should never happen, even if there's no separator in the expression TQStringUtils::split should not return an empty vector! Please report this to qframework-users@cern.ch!");
      return false;
    }
    if (expressions.size()==2) {
      this->fVecExpression = expressions[0];
      this->fIndexExpression = expressions[1];
    } else {
      ERRORclass("Illegal number of sub-expressions (%d) for TQVectorAuxObservable with expression '%s', must be 2 for mode 'AT'!",expressions.size(),this->fExpression.Data());
      throw std::runtime_error("Failed to parse expression in TQVectorAuxObservable::initializeSelf for 'AT' mode."); 
    }
    this->fIndexObservable = TQObservable::getObservable(this->fIndexExpression,this->fSample);
    if (!this->fIndexObservable) {
      ERRORclass("Failed to obtain index-observable!");
      throw std::runtime_error(TString::Format("Failed to obtain observable from expression '%s'",this->fIndexExpression.Data()).Data());
    }
    DEBUGclass("Going to initialize index-observable");
    if (!this->fIndexObservable->initialize(this->fSample)) {
      throw std::runtime_error(TString::Format("Failed to initialize observable with name '%s' created from expression '%s' for sample '%s'",this->fIndexObservable->GetName(), this->fIndexExpression.Data(), this->fSample->getPath().Data()).Data());
    }
      
  } else {
    this->fVecExpression = this->fExpression; //just use the full expression as the vector (sub) observable expression
  }
    
  this->fSubObservable = TQObservable::getObservable(this->fVecExpression,this->fSample);
  if (!this->fSubObservable) {
    ERRORclass("Failed to obtain sub-observable!");
    throw std::runtime_error(TString::Format("Failed to obtain observable from expression '%s'",this->fVecExpression.Data()).Data());
  }
  DEBUGclass("Going to initialize sub-observable");
  if (!this->fSubObservable->initialize(this->fSample)) {
    throw std::runtime_error(TString::Format("Failed to initialize observable with name '%s' created from expression '%s' for sample '%s'",this->fSubObservable->GetName(), this->fVecExpression.Data(), this->fSample->getPath().Data()).Data());
  }
  this->fCachedEntry = -999;
  DEBUGclass("Done initializing instance with expression '%s'",this->getExpression().Data());
  return true;
}
 
//______________________________________________________________________________________________

bool TQVectorAuxObservable::finalizeSelf(){
  // finalize self - delete accessor
  this->fVecExpression = ""; this->fIndexExpression = "";
  return this->fSubObservable ? this->fSubObservable->finalize() : true;
  
}

//______________________________________________________________________________________________

TString TQVectorAuxObservable::getOperationName(TQVectorAuxObservable::Operation op) {
  switch(op) {
    //returning the prefix for a certain operation type without the preceeding 'Vec' and the following ':'
    case AND: return TString("AND"); break;
    case OR: return TString("OR"); break;
    case SUM: return TString("SUM"); break;
    case SUMABS: return TString("SUMABS"); break;
    case AVG: return TString("AVG"); break;
    case LEN: return TString("LEN"); break;
    case MAX: return TString("MAX"); break;
    case MIN: return TString("MIN"); break;
    case MAXABS: return TString("MAXABS"); break;
    case MINABS: return TString("MINABS"); break;
    case NORM: return TString("NORM"); break;
    case PROD: return TString("PROD"); break;
    case NTRUE: return TString("NTRUE"); break;
    case NFALSE: return TString("NFALSE"); break;
    case AT: return TString("AT"); break;
    case invalid: return TString("invalid"); break;
  };
  return TString("nonExistingEnum"); 
}

TString TQVectorAuxObservable::getPrefix(TQVectorAuxObservable::Operation op) {
  return TString("Vec"+TQVectorAuxObservable::getOperationName(op));
}

TQVectorAuxObservable::Operation TQVectorAuxObservable::readPrefix(TString& expr) {
  // try to read a matching prefix and return a corresponding enum.
  // If found, the prefix is removed from expression.
  DEBUGclass("Trying to read prefix from expression '%s'",expr.Data());
  TString copyExpr = expr;
  TQVectorAuxObservable::Operation op = TQVectorAuxObservable::Operation::invalid;
  if(TQStringUtils::removeLeadingText(expr,"Vec")){
    if (TQStringUtils::removeLeadingText(expr,"AND")) {
      op = TQVectorAuxObservable::Operation::AND;
    } else if (TQStringUtils::removeLeadingText(expr,"OR")) {
      op = TQVectorAuxObservable::Operation::OR;
    } else if (TQStringUtils::removeLeadingText(expr,"SUMABS")) {
      op = TQVectorAuxObservable::Operation::SUMABS;
    } else if (TQStringUtils::removeLeadingText(expr,"SUM")) {
      op = TQVectorAuxObservable::Operation::SUM;
    } else if (TQStringUtils::removeLeadingText(expr,"AVG")) {
      op = TQVectorAuxObservable::Operation::AVG;
    } else if (TQStringUtils::removeLeadingText(expr,"LEN")) {
      op = TQVectorAuxObservable::Operation::LEN;
    } else if (TQStringUtils::removeLeadingText(expr,"MIN")) {
      op = TQVectorAuxObservable::Operation::MIN;
    } else if (TQStringUtils::removeLeadingText(expr,"MAX")) {
      op = TQVectorAuxObservable::Operation::MAX;
    } else if (TQStringUtils::removeLeadingText(expr,"MINABS")) {
      op = TQVectorAuxObservable::Operation::MINABS;
    } else if (TQStringUtils::removeLeadingText(expr,"MAXABS")) {
      op = TQVectorAuxObservable::Operation::MAXABS;
    } else if (TQStringUtils::removeLeadingText(expr,"NORM")) {
      op = TQVectorAuxObservable::Operation::NORM;
    } else if (TQStringUtils::removeLeadingText(expr,"PROD")) {
      op = TQVectorAuxObservable::Operation::PROD;
    } else if (TQStringUtils::removeLeadingText(expr,"NTRUE")) {
      op = TQVectorAuxObservable::Operation::NTRUE;
    } else if (TQStringUtils::removeLeadingText(expr,"NFALSE")) {
      op = TQVectorAuxObservable::Operation::NFALSE;
    } else if (TQStringUtils::removeLeadingText(expr,"AT")) {
      op = TQVectorAuxObservable::Operation::AT;
    }
    if (op != TQVectorAuxObservable::Operation::invalid) {
      TQStringUtils::readBlanksAndNewlines(expr);
      TString subExpr;
      if (TQStringUtils::readBlock(expr,subExpr,"()[]{}","\"\"''")>0) {
        //check if remaining subExpr is empty (up to whitespaces), otherwise there is likely an error and we need to throw! (unless we find a clean way to automatically fix the expression and re-inject it.
        if (!TQStringUtils::isEmpty(expr,true/*ignore blanks*/)) {
          //print an error to inform the user
          ERRORfunc("A known prefix for the VectorAux observable has been found in the expression '%s'. However, there seems to be a remainder after its expression block: '%s'. Did you forget to encapsulate the VecAux expression in square brackets '[...]'?",copyExpr.Data(),expr.Data());
        } else {
          expr=subExpr; //copy the now reduced expression back
          return op;
        }
      }
      
    }
    
    
    expr = copyExpr; //in case nothing matched, restore the original state
  }
  DEBUGclass("No valid TQVectorAuxObservable prefix found!");
  //fallback if no known operation matches:
  return TQVectorAuxObservable::Operation::invalid;
}
//______________________________________________________________________________________________

// the following preprocessor macro defines an "observable factory"
// that is supposed to create instances of your class based on input
// expressions.

// it should receive an expression as an input and decide whether this
// expression should be used to construct an instance of your class,
// in which case it should take care of this, or return NULL.

// in addition to defining your observable factory here, you need to
// register it with the TQObservable manager. This can either be done
// from C++ code, using the line
//   TQObservable::manager.registerFactory(TQVectorAuxObservable::getFactory(),true);
// or from python code, using the line
//   TQObservable.manager.registerFactory(TQVectorAuxObservable.getFactory(),True)
// Either of these lines need to be put in a location where they are
// executed before observables are retrieved. You might want to 'grep'
// for 'registerFactory' in the package you are adding your observable
// to in order to get some ideas on where to put these!

DEFINE_OBSERVABLE_FACTORY(TQVectorAuxObservable,TString expr){
  // try to create an instance of this observable from the given expression
  // return the newly created observable upon success
  // or NULL upon failure
  //std::cout<<"Trying to match prefix of '"<<expr.Data() <<"'"<<std::endl;
  TQVectorAuxObservable::Operation op = TQVectorAuxObservable::readPrefix(expr);
  if (op != TQVectorAuxObservable::Operation::invalid ) {
    //std::cout<<"Found match!"<<std::endl;
    TQVectorAuxObservable* obs = new TQVectorAuxObservable(expr);
    obs->setOperation(op);
    return obs;    
  }
  // check if the expression fits your observable type
  //std::cout<<"Failed to find a match!"<<std::endl;
  return NULL;
}

