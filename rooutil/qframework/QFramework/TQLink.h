//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQLink__
#define __TQLink__

#include "TNamed.h"

class TQLink : public TNamed {

public:

  TQLink();
  TQLink(const TString& name);

  virtual TString getDestAsString() = 0;
 
  virtual ~TQLink();
 
  ClassDefOverride(TQLink, 1); // abstract placeholder class

};

#endif
