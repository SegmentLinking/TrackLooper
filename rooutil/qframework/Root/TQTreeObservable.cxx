#include "QFramework/TQTreeObservable.h"
#include "QFramework/TQStringUtils.h"
#include "QFramework/TQSample.h"
#include "QFramework/TQToken.h"
#include "QFramework/TQUtils.h"
#include "TTree.h"
#include "QFramework/TQIterator.h"

// #define _DEBUG_
#include "QFramework/TQLibrary.h"

#include <iostream>

ClassImp(TQTreeObservable)

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQTreeObservable
//
// The TQTreeObservable is a (still abstract) specialization of the abstract class TQObservable.
// It provides derived classes with an automated mechanism to access the Tree.
//
////////////////////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________________________

TQTreeObservable::TQTreeObservable() : 
TQObservable()
{
  // default constructor
}

//______________________________________________________________________________________________

TQTreeObservable::TQTreeObservable(const TString& expression) :
  TQObservable(expression)
{
  // constructor with expression argument
}

//______________________________________________________________________________________________

bool TQTreeObservable::initialize(TQSample * sample){
  // initialize this observable, obtain a tree token
  if(this->fIsInitialized) return true;
  /* we can't do anything if we already own a tree token */
  if (this->fTreeToken || this->fTree) return false;

  /* the sample to use has to be valid */
  if(!sample) return false;

  /* try to get a tree token */
  this->fTreeToken = sample->getTreeToken();
  if (!this->fTreeToken) return false;

  this->fSample = sample;
  this->fTreeToken->setOwner(this);
  this->fTree = static_cast<TTree*>(this->fTreeToken->getContent());
 
  this->fIsInitialized = this->initializeSelf();

  if(!this->fIsInitialized){
    if(this->fTreeToken && this->fSample) this->fSample->returnToken(this->fTreeToken);
    this->fTreeToken = 0;
    this->fSample = 0;
    this->fTree = 0;
  }

  return this->fIsInitialized;
}


//______________________________________________________________________________________________

bool TQTreeObservable::finalize() {
  // finalize this observable, return the tree token
  if(!this->fIsInitialized){
    return true;
  }
 
  if (!this->fTreeToken) return false;
  if (!this->fSample) return false;
  if (!this->fTree) return false;
 
  bool ok = this->finalizeSelf();
  this->fIsInitialized = !(this->fSample->returnToken(this->fTreeToken) && ok);
 
  this->fTreeToken = 0;
  this->fSample = 0;
  this->fTree = 0;
 
  return (!this->fIsInitialized);

}


//______________________________________________________________________________________________

TQTreeObservable::~TQTreeObservable() {
  // standard destructor
}

//______________________________________________________________________________________________

void TQTreeObservable::print() const {
  // print the contents of this observable and its associated branches
  std::cout << TQStringUtils::makeBoldYellow(this->getExpression()) << std::endl;
  TQUtils::printBranches(this->fTree);
}

//______________________________________________________________________________________________

Long64_t TQTreeObservable::getCurrentEntry() const {
  // retrieve the current entry from the tree
  if(!this->fTree) return -1;
  return this->fTree->GetReadEntry();
}
