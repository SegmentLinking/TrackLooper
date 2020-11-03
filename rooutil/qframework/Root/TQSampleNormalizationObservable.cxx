#include "QFramework/TQSampleNormalizationObservable.h"
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
// TQSampleNormalizationObservable
//
// The TQSampleNormalizationObservable is a variant of TQObservable
// that simply always returns the sample normaliztion.
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQSampleNormalizationObservable)

//______________________________________________________________________________________________

TQSampleNormalizationObservable::TQSampleNormalizationObservable(){
  // default constructor
  DEBUGclass("default constructor called");
}

//______________________________________________________________________________________________

TQSampleNormalizationObservable::~TQSampleNormalizationObservable(){
  // default destructor
  DEBUGclass("destructor called");
} 

//______________________________________________________________________________________________

double TQSampleNormalizationObservable::getValue() const {
  // value retrieval function, called on every event for every cut and histogram
  DEBUGclass("entering function");

  // the contents of this function highly depend on the way your
  // observable is supposed to retrieve (or create?) data
  // good luck, you're on your own here!
  
  return this->fSample->getNormalisation();
}

//______________________________________________________________________________________________

Long64_t TQSampleNormalizationObservable::getCurrentEntry() const {
  // retrieve the current entry from the tree

  // since we don't have any tree or event pointer, there is usually
  // no way for us to know what entry we are currently looking
  // at. hence, we return -1 by default
  
  return -1;
}

//______________________________________________________________________________________________

TObjArray* TQSampleNormalizationObservable::getBranchNames() const {
  // retrieve the list of branch names for this observable
  // ownership of the list belongs to the caller of the function
  DEBUGclass("retrieving branch names");

  // since we don't have a tree pointer, we probably also don't need any branches
  return NULL;
}
//______________________________________________________________________________________________

bool TQSampleNormalizationObservable::initializeSelf(){
  DEBUGclass("initializing");
  // initialize this observable
  return true;
}

//______________________________________________________________________________________________

bool TQSampleNormalizationObservable::finalizeSelf(){
  // initialize this observable
  DEBUGclass("finalizing");
  // remember to undo anything you did in initializeSelf() !
  return true;
}
//______________________________________________________________________________________________

TQSampleNormalizationObservable::TQSampleNormalizationObservable(const TString& name):
TQObservable(name)
{
  // constructor with name argument
  DEBUGclass("constructor called with '%s'",name.Data());
}
