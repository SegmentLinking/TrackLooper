#include "QFramework/TQLink.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQLink:
//
// TQLink is an abstract base class for a placeholder object.
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQLink)


//__________________________________________________________________________________|___________

TQLink::TQLink() : TNamed("Link", "Link") {
  // Default constructor of TQLink class:

}


//__________________________________________________________________________________|___________

TQLink::TQLink(const TString& name) : TNamed(name.Data(), name.Data()) {
  // Default constructor of TQLink class:

}


//__________________________________________________________________________________|___________

TQLink::~TQLink() {
  // Destructor of TQLink class:

}


