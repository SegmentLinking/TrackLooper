#include "QFramework/TQEventIndexObservable.h"
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
// TQEventIndexObservable
//
// The TQEventIndexObservable is a variant of TQObservable that simply
// returns the index of the current event in the tree.
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQEventIndexObservable)

//______________________________________________________________________________________________

TQEventIndexObservable::TQEventIndexObservable(){
  // default constructor
  DEBUGclass("default constructor called");
}

//______________________________________________________________________________________________

TQEventIndexObservable::~TQEventIndexObservable(){
  // default destructor
  DEBUGclass("destructor called");
} 

//______________________________________________________________________________________________

bool TQEventIndexObservable::initializeSelf(){
  // initialize this observable on a sample/tree
  DEBUGclass("initializing");

  // since this function is only called once per sample, we can
  // perform any checks that seem necessary
  if(!this->fTree){
    DEBUGclass("no tree, terminating");
    return false;
  }
  return true;
}

//______________________________________________________________________________________________

bool TQEventIndexObservable::finalizeSelf(){
  // finalize this observable on a sample/tree
  DEBUGclass("finalizing");
  return true;
}

//______________________________________________________________________________________________

TObjArray* TQEventIndexObservable::getBranchNames() const {
  // retrieve the list of branch names 
  // ownership of the list belongs to the caller of the function
  DEBUGclass("retrieving branch names");
  return NULL;
}

//______________________________________________________________________________________________

double TQEventIndexObservable::getValue() const {
  // retrieve the index of the current event in the tree
  DEBUGclass("returning");
  return this->getCurrentEntry();
}
//______________________________________________________________________________________________

TQEventIndexObservable::TQEventIndexObservable(const TString& name):
TQTreeObservable(name)
{
  // constructor with name argument
  DEBUGclass("constructor called with '%s'",name.Data());
}
