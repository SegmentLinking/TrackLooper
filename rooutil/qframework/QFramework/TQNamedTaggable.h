//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQNamedTaggable__
#define __TQNamedTaggable__

#include "TNamed.h"
#include "QFramework/TQStringUtils.h"
#include "QFramework/TQTaggable.h"

class TQNamedTaggable : public TNamed, public TQTaggable {

public:

  TQNamedTaggable();
  TQNamedTaggable(TString name);
  TQNamedTaggable(TString name, TString tags);
  TQNamedTaggable(TString name, TQTaggable * tags);
  TQNamedTaggable(TQNamedTaggable * tags);
 
  virtual ~TQNamedTaggable();
 
  virtual TString getName();
  virtual void setName(const TString& newName);
  virtual const TString& getNameConst() const;

  virtual TString getTitle();
  virtual void setTitle(const TString& newTitle);
  virtual const TString& getTitleConst() const;

  ClassDefOverride(TQNamedTaggable, 1); // base class for named taggable objects

};

#endif
