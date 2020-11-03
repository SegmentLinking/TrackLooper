#include "QFramework/TQHistogramObservable.h"
#include "TH1.h"
#include "TH3.h"
#include "TH2.h"
#include <limits>
#include <stdexcept>
#include <algorithm>

// uncomment the following line to enable debug printouts
// #define _DEBUG_
// you can perform debug printouts with statements like this
//DEBUG("error number %d occurred",someInteger);

#include "QFramework/TQUtils.h"
#include "QFramework/TQHistogramUtils.h"
// be careful to not move the _DEBUG_ flag behind the following line
// otherwise, it will show no effect
#include "QFramework/TQLibrary.h"

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQHistogramObservable
//
// The TQHistogramObservable is a templated class that allows to
// extract information from histograms based on quantities found in
// the event. A typical application might be to retrieve efficiency
// scale factors from a 2-dimensional histogram based on a binning in
// pT and eta.
//
////////////////////////////////////////////////////////////////////////////////////////////////

templateClassImp(TQHistogramObservable<T>)

//______________________________________________________________________________________________

template <class T>
TQHistogramObservable<T>::TQHistogramObservable(){
  // default constructor
  DEBUGclass("default constructor called");
}

//______________________________________________________________________________________________

template <class T>
TQHistogramObservable<T>::~TQHistogramObservable(){
  // default destructor
  DEBUGclass("destructor called");
} 

//______________________________________________________________________________________________
template <class T>
int TQHistogramObservable<T>::getNevaluations() const {
  // returns the number of valid evaluations of this observable for the current event
  // returns -1 if a problem is found. It is left to the calling code to decide if an error should be thrown
  DEBUGclass("getNevaluations called");
  if (this->fObservableType != TQObservable::ObservableType::vector) return 1; //for non-vector valued instances we know the number of evaluations
  if (!this->makeCache()) return -1; //something went wrong creating the cache
  DEBUGclass("returning");
  return this->fCachedValues.size();

  
}

//______________________________________________________________________________________________

template <class T>
double TQHistogramObservable<T>::getValue() const {
  //forward to the implementation capable of handling multiple values.
  if (this->fObservableType != TQObservable::ObservableType::vector) return this->getValueAt(0);
  else throw std::runtime_error(TString::Format( "Caught attempt to perform scalar evaluation on vector valued instance of TQMultiObservable with expression '%s'",this->getExpression().Data() ).Data());
  return std::numeric_limits<double>::quiet_NaN();
}

//______________________________________________________________________________________________

template <class T>
double TQHistogramObservable<T>::getValueAt(int index) const {
  // retrieve the value of this observable
  if (!this->makeCache()) {
    throw std::runtime_error(TString::Format("Error in TQHistogramObservable with active expression '%s': Failed to make cache!",this->getActiveExpression().Data()).Data()); 
  }
  if (index<(int)this->fCachedValues.size()) {
    return this->fCachedValues[index];
  } else {
    throw std::runtime_error(TString::Format("Caught attempt to evaluate TQHistogramObservable with active expression '%s' out of bounds (requested index is %d, max index available is %d)",this->getActiveExpression().Data(),index,(int)fCachedValues.size()-1).Data());
  }
  return std::numeric_limits<double>::quiet_NaN();
}
//______________________________________________________________________________________________

template <class T>
TQHistogramObservable<T>::TQHistogramObservable(const TString& expression):
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

template <class T>
bool TQHistogramObservable<T>::initializeSelf(){
  // initialize this observable
  this->fActiveExpression = this->compileExpression(this->fExpression,this->fSample,false);
  DEBUGclass("initializing observable with expression '%s'",this->fActiveExpression.Data()); 
  bool retval = this->parseExpression(this->fActiveExpression);
  this->fObservableType = TQObservable::ObservableType::scalar;
  this->fCachedValues.resize(1);//make sure we have at least one 'slot'. This helps making the scalar usage a bit more efficient
  for(size_t i=0; i<this->fObservables.size(); ++i){
    if(!this->fObservables[i]->initialize(this->fSample)) {
      ERRORclass("Failed to initialize sub-observable with expression '%s' for sample '%s' in TQHistogramObservable with expression '%s'.",this->fObservables[i]->getExpression().Data(),this->fSample->getPath().Data(),this->fExpression.Data());
      retval = false;
    }
    if (this->fObservables[i]->getObservableType() == TQObservable::ObservableType::vector) this->fObservableType = TQObservable::ObservableType::vector;
  }
  return retval;
}

//______________________________________________________________________________________________

template <class T>
bool TQHistogramObservable<T>::finalizeSelf(){
  // finalize this observable
  DEBUGclass("finalizing '%s'",this->GetName());
  bool ok = true;
  for(size_t i=0; i<this->fObservables.size(); ++i){
    if(!(this->fObservables[i]->finalize())){
      ok = false;
    }
  }
  //if(this->fHistogram) delete this->fHistogram; //why does this cause segfaults every now and then? ROOT must be doing some really strange black magic ...
  //if (this->fFile && this->fFile->IsOpen()) this->fFile->Close();
  //if (this->fFile) delete this->fFile;
  this->fHistogram = NULL;
  this->fActiveExpression.Clear();
  this->fObservables.clear();
  this->fFileName = "";
  this->fActiveExpression = "";
  this->fCachedEntry = -1;
  this->fCachedValue = 0.;
  this->fObservableType = TQObservable::ObservableType::scalar;
  return ok;
}

//______________________________________________________________________________________________

template <class T>
Long64_t TQHistogramObservable<T>::getCurrentEntry() const {
  // retrieve the current entry from the tree
  if(this->fObservables.size() == 0) return -1;
  return this->fObservables[0]->getCurrentEntry();
}

//______________________________________________________________________________________________

template <class T>
TObjArray* TQHistogramObservable<T>::getBranchNames() const {
  //retrieve the list of branch names for this observable
  // ownership of the list belongs to the caller of the function
  DEBUGclass("retrieving branch names for %p",this);
  TObjArray* bnames = new TObjArray();
  for(size_t i=0; i<this->fObservables.size(); ++i){
    TQObservable* obs = this->fObservables[i];
    if(!obs){
      throw std::runtime_error("encountered invalid subobservable");
    }
    DEBUG("retrieving branches from observable '%s' of class '%s'",obs->getExpression().Data(),obs->ClassName());
    TCollection* c = this->fObservables[i]->getBranchNames();
    if(c){
      c->SetOwner(false);
      bnames->AddAll(c);
      delete c;
    }
  }
  return bnames;
}
      
//______________________________________________________________________________________________

template <class T>
const TString& TQHistogramObservable<T>::getExpression() const {
  // retrieve the expression associated with this observable
  return this->fExpression;
}

//______________________________________________________________________________________________

template <class T>
bool TQHistogramObservable<T>::hasExpression() const {
  // check if this observable type knows expressions
  return true;
}

//______________________________________________________________________________________________

template <class T>
void TQHistogramObservable<T>::setExpression(const TString& expr){
  // set the expression to a given string
  this->fExpression = expr;
}
//______________________________________________________________________________________________

template <class T>
bool TQHistogramObservable<T>::parseExpression(const TString& expr){
  // parse the expression
  this->clearParsedExpression();
  TString expression(expr);
  TQStringUtils::readToken(expression,this->fFileName,"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890/:.-_+");
  TQStringUtils::removeLeadingBlanks(expression);
  TQStringUtils::removeTrailingBlanks(expression);
  TString fileName = "";
  TString objName = this->fFileName;
  TQStringUtils::readUpTo(objName,fileName,":");
  TQStringUtils::removeLeading(objName,":");
  this->fFile = TFile::Open(fileName,"READ");
  if (!this->fFile || !this->fFile->IsOpen()){
    throw(std::runtime_error(TString::Format("Failed to open file '%s'!",fileName.Data()).Data()));
    return false;
  }
  T* tmpHist = dynamic_cast<T*>(this->fFile->Get(objName));
  if (!tmpHist){
    throw(std::runtime_error(TString::Format("Failed to retrieve object of type '%s' and name '%s' from file '%s'!",T::Class() == TH1::Class() ? "TH1" : T::Class() == TH2::Class() ? "TH2": T::Class() == TH3::Class() ? "TH3" : "unknown", objName.Data(), fileName.Data()).Data()));
    return false;
  }
  this->fHistogram = dynamic_cast<T*>(TQHistogramUtils::copyHistogram(tmpHist));
  if (!this->fHistogram) {
    throw(std::runtime_error(TString::Format("Failed to retrieve object of type '%s' and name '%s' from file '%s'!",T::Class() == TH1::Class() ? "TH1" : T::Class() == TH2::Class() ? "TH2": T::Class() == TH3::Class() ? "TH3" : "unknown", objName.Data(), fileName.Data()).Data()));
    return false;
  }
  this->fHistogram->SetDirectory(0);
  this->fFile->Close();
  delete this->fFile;
  
  TString block;
  if(expression.Length() < 3){
    throw(std::runtime_error(TString::Format("unable to parse expression for HistogramObservable '%s' - empty argument block!",this->GetName()).Data()));
    return false;
  }
  TQStringUtils::readBlock(expression,block,"()");
  std::vector<TString> vDef = TQStringUtils::tokenizeVector(block,",",true,"()[]{}","\"\"''");
  if (vDef.size()<1) {
    throw(std::runtime_error(TString::Format("unable to create HistogramObservable '%s'- not enough arguments!",this->GetName()).Data())); 
    return false;
  }
  for (size_t i = 0; i<vDef.size(); ++i) {
    DEBUGclass("attempting to retrieve subobservable with name '%s'",vDef.at(i).Data());
    TQObservable* obs = TQObservable::getObservable(vDef.at(i),this->fSample);
    if (!obs) {
      throw(std::runtime_error(TString::Format("unable to create observable from expression '%s' for HistogramObservable '%s' - empty argument block!",vDef.at(i).Data(),this->GetName()).Data())); 
      return false;
    } else if(obs == this){
      throw(std::runtime_error(TString::Format("parsing error: expression '%s' yielded observable '%s'",vDef.at(i).Data(),this->getExpression().Data()).Data()));
    } else {
      this->fObservables.push_back(obs);
    }
  }
  return true;
}

//______________________________________________________________________________________________

template <class T>
void TQHistogramObservable<T>::clearParsedExpression(){
  // clear the current expression
  //if (this->fFile && this->fFile->IsOpen()) this->fFile->Close();
  //if (this->fFile) delete this->fFile;
  //if (this->fHistogram) delete this->fHistogram;
  this->fHistogram = NULL;
  this->fObservables.clear();
}

//______________________________________________________________________________________________

template <class T>
TString TQHistogramObservable<T>::getActiveExpression() const {
  // retrieve the expression associated with this incarnation
  
  //rebuild the expression to ensure consistency
  TString retval = this->fFileName+"("; 
  for (size_t i = 0; i<fObservables.size(); ++i) {
    retval += "["+fObservables.at(i)->getExpression()+"]"+(i<fObservables.size()-1 ? "," : "");
  }
  retval += ")";
  return retval;
  //return this->fActiveExpression;
}

//______________________________________________________________________________________________

template <class T>
TQObservable* TQHistogramObservable<T>::getObservable(int idx){
  // return the sub-observable with the given index
  return this->fObservables.at(idx);
}

//______________________________________________________________________________________________

template <class T>
bool TQHistogramObservable<T>::makeCache() const {
  const int entry = this->getCurrentEntry();
  if(entry == this->fCachedEntry) return true; //nothing to do here, cache is already up-to-date
  
  if (!this->fHistogram) { //no need to do anything if we'll run into a problem anyways (also saves us from performing this check for each entry in case of vector observable mode)
    throw (std::runtime_error("Missing histogram to obtain values from!"));
    return false;
  }
  
  if (this->fObservableType == TQObservable::ObservableType::scalar) {
    //shortcut for scalar case
    if (this->fCachedValues.size() != 1) {
      throw std::runtime_error(TString::Format("Error while trying to cache evaluation result of TQHistogramObservable with active expression '%s': cache vector size is not 1",this->getActiveExpression().Data()).Data());
    }
    this->fCachedValues[0] = this->findValue(0);
    this->fCachedEntry = entry;
    return true;
  }
  int nEvals = -1;
  for (size_t i=0; i<this->fObservables.size(); ++i) {
    if (this->fObservables[i]->getObservableType() != TQObservable::ObservableType::vector) continue;
    int iEvals = this->fObservables[i]->getNevaluations();
    if (nEvals < 0) nEvals = iEvals;
    if (nEvals != iEvals) return false; //this should not happen, only combinations with vector observables of equal "length" are allowed!
  }

  this->fCachedValues.resize(nEvals);
  for (int index=0; index<nEvals; ++index) {
    fCachedValues[index] = this->findValue(index);
  }
  this->fCachedEntry = entry;
  return true;  
  
}

//______________________________________________________________________________________________
template <>
double TQHistogramObservable<TH1>::findValue(int index) const {
  if (!this->fHistogram) {
    throw (std::runtime_error("Missing histogram to obtain values from!"));
  }
  
  return this->fHistogram->GetBinContent(this->fHistogram->FindBin(
                                                  this->fObservables.at(0)->getObservableType() == TQObservable::ObservableType::vector ? this->fObservables.at(0)->getValueAt(index) :  this->fObservables.at(0)->getValue() ));
}

//______________________________________________________________________________________________
template <>
double TQHistogramObservable<TH2>::findValue(int index) const {
  if (!this->fHistogram) {
    throw (std::runtime_error("Missing histogram to obtain values from!"));
  }
  
  return this->fHistogram->GetBinContent(this->fHistogram->FindBin(
                                                  this->fObservables.at(0)->getObservableType() == TQObservable::ObservableType::vector ? this->fObservables.at(0)->getValueAt(index) :  this->fObservables.at(0)->getValue(),
                                                  this->fObservables.at(1)->getObservableType() == TQObservable::ObservableType::vector ? this->fObservables.at(1)->getValueAt(index) :  this->fObservables.at(1)->getValue() ));
}
//______________________________________________________________________________________________
template <>
double TQHistogramObservable<TH3>::findValue(int index) const {
  if (!this->fHistogram) {
    throw (std::runtime_error("Missing histogram to obtain values from!"));
  }
  
  return this->fHistogram->GetBinContent(this->fHistogram->FindBin(
                                                  this->fObservables.at(0)->getObservableType() == TQObservable::ObservableType::vector ? this->fObservables.at(0)->getValueAt(index) :  this->fObservables.at(0)->getValue(),
                                                  this->fObservables.at(1)->getObservableType() == TQObservable::ObservableType::vector ? this->fObservables.at(1)->getValueAt(index) :  this->fObservables.at(1)->getValue(),
                                                  this->fObservables.at(2)->getObservableType() == TQObservable::ObservableType::vector ? this->fObservables.at(2)->getValueAt(index) :  this->fObservables.at(2)->getValue() ));
}

//______________________________________________________________________________________________

DEFINE_TEMPLATE_OBSERVABLE_FACTORY_SPECIALIZATION(TQHistogramObservable,TH1,TString expression){
  if(TQStringUtils::removeLeadingText(expression,"TH1Map:")){
    return new TQHistogramObservable<TH1>(expression);
  }
  return NULL;
}

//______________________________________________________________________________________________

DEFINE_TEMPLATE_OBSERVABLE_FACTORY_SPECIALIZATION(TQHistogramObservable,TH2,TString expression){
  if(TQStringUtils::removeLeadingText(expression,"TH2Map:")){
    return new TQHistogramObservable<TH2>(expression);
  }
  return NULL;
}

//______________________________________________________________________________________________

DEFINE_TEMPLATE_OBSERVABLE_FACTORY_SPECIALIZATION(TQHistogramObservable,TH3,TString expression){
  if(TQStringUtils::removeLeadingText(expression,"TH3Map:")){
    return new TQHistogramObservable<TH3>(expression);
  }
  return NULL;
}

template class TQHistogramObservable<TH1>;
template class TQHistogramObservable<TH2>;
template class TQHistogramObservable<TH3>;


