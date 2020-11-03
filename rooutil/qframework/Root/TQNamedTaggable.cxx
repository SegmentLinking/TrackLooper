#include "QFramework/TQNamedTaggable.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

using std::cout;
using std::endl;

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQNamedTaggable:
//
// A base class for all named instances of TQTaggable objects.
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQNamedTaggable)


//__________________________________________________________________________________|___________

TQNamedTaggable::TQNamedTaggable() : TNamed(), TQTaggable() {
  // Default constructor of TQNamedTaggable class:

}


//__________________________________________________________________________________|___________

TQNamedTaggable::TQNamedTaggable(TString name) : TNamed(name.Data(), ""), TQTaggable() {
  // Constructor of TQNamedTaggable class:

}


//__________________________________________________________________________________|___________

TQNamedTaggable::TQNamedTaggable(TString name, TString tags) :
  TNamed(name.Data(), ""), TQTaggable(tags) {
  // Constructor of TQNamedTaggable class:

}


//__________________________________________________________________________________|___________

TQNamedTaggable::TQNamedTaggable(TString name, TQTaggable * tags) :
  TNamed(name.Data(), ""), TQTaggable(tags) {
  // Constructor of TQNamedTaggable class:

}


//__________________________________________________________________________________|___________

TQNamedTaggable::TQNamedTaggable(TQNamedTaggable * tags) : TNamed(), TQTaggable(tags) {
  // Constructor of TQNamedTaggable class:

  if (tags) {
    SetName(TQStringUtils::replace(tags->GetName(),"-","_"));
    SetTitle(tags->GetTitle());
  }
}


//__________________________________________________________________________________|___________

TQNamedTaggable::~TQNamedTaggable() {
  // Destructor of TQNamedTaggable class:

}


//__________________________________________________________________________________|___________

TString TQNamedTaggable::getName() {
  // retrieve the name of this object
  return this->fName;
}

//__________________________________________________________________________________|___________

void TQNamedTaggable::setName(const TString& newName) {
  // set the name of this object
  // note that any occurence of '-' in the name will be substituted by '_' in order
  // to comply with the naming policy for object stored inside TQFolders
  this->fName = TQStringUtils::replace(newName,"-","_");
}

//__________________________________________________________________________________|___________

const TString& TQNamedTaggable::getNameConst() const {
  // retrieve a const reference to the name of this object
  return this->fName;
}

//__________________________________________________________________________________|___________

TString TQNamedTaggable::getTitle() {
  // retrieve the title of this object
  return this->fTitle;
}

//__________________________________________________________________________________|___________

void TQNamedTaggable::setTitle(const TString& newTitle) {
  // set the title of this object
  this->fTitle = newTitle;
}

//__________________________________________________________________________________|___________

const TString& TQNamedTaggable::getTitleConst() const {
  // retrieve a const reference to the title of this object
  return this->fTitle;
}


