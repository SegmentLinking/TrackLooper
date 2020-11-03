#include "TString.h"
#include "TClass.h"
#include "QFramework/TQToken.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

#include "QFramework/TQLibrary.h"

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQToken:
//
// A TQToken is a container for a single object that can help manage
// its ownership.  Tokens can be retrieved from certain functions,
// which are the owners of the contained objects. Upon return of the
// token, the owner can then delete the object without user
// intervention.
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQToken)


//______________________________________________________________________________________________

TQToken::TQToken() : TObject() {
  // empty constructor
}


//______________________________________________________________________________________________

void TQToken::print() {
  // print the contents of this token (debugging functionality)
  TString name = "<empty>";
  if (fOwner) {
    name = TString::Format("%s::%s", fOwner->IsA()->GetName(), fOwner->GetName());
    if (fOwner->InheritsFrom(TNamed::Class())) {
      name.Append(TString::Format("(%s)", ((TNamed*)fOwner)->GetTitle()));
    }
  }
  INFO("TQToken has owner '%s'", name.Data());
}


//______________________________________________________________________________________________

void TQToken::setContent(void * obj_) {
  // set the contents to a given pointer
  // ownership of the object stays with the caller
  fContent = obj_;
}


//______________________________________________________________________________________________

void * TQToken::getContent() {
  // retrieve the content of the token as a void pointer
  return fContent;
}

TObject * TQToken::getContentAsTObject() {
  // retrieve the content of the token as a TObject pointer
  return (TObject*)fContent;
}

//______________________________________________________________________________________________

void TQToken::setOwner(TObject * obj_) {
  // set the owner of the content to the given object
  fOwner = obj_;
}


//______________________________________________________________________________________________

TObject * TQToken::getOwner() {
  // retrieve the owner of this object
  return fOwner;
}


//______________________________________________________________________________________________

TQToken::~TQToken() {
  // default destructor
  // will not delete content, since ownership stays with owner
  
}


