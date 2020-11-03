//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQImportLink__
#define __TQImportLink__

#include "QFramework/TQLink.h"
#include "TString.h"

class TQImportLink : public TQLink {

protected:

  TString fImportPath;


public:

  TQImportLink();
  TQImportLink(TString name);
  TQImportLink(TString name, TString importPath);

  virtual TString getImportPath();
  virtual TString getDestAsString();

  virtual ~TQImportLink();
 
  ClassDefOverride(TQImportLink, 1); // placeholder class to allow spreading data over several files

};

#endif
