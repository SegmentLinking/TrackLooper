#include "QFramework/TQImportLink.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

using std::cout;
using std::endl;

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQImportLink:
//
// A variant of TQLink that is capable of importing objects from other files.
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQImportLink)


//__________________________________________________________________________________|___________

TQImportLink::TQImportLink() : TQLink() {
  // Default constructor

  fImportPath.Clear();
}


//__________________________________________________________________________________|___________

TQImportLink::TQImportLink(TString name) : TQLink(name) {
  // Default constructor

}


//__________________________________________________________________________________|___________

TQImportLink::TQImportLink(TString name, TString importPath) : TQLink(name) {
  // Default constructor
  fImportPath = importPath;
}


//__________________________________________________________________________________|___________

TString TQImportLink::getImportPath() {
	// return the import path
  return fImportPath;
}


//__________________________________________________________________________________|___________

TString TQImportLink::getDestAsString() {
	// return the destination string (that is, the import path)
  return getImportPath();
}


//__________________________________________________________________________________|___________

TQImportLink::~TQImportLink() {
  // Destructor of TQImportLink class:

}


