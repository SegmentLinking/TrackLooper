//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQToken__
#define __TQToken__

#include "TObject.h"

class TQToken : public TObject {

protected:
 
  void * fContent = NULL;
  TObject * fOwner = NULL;

public:

  TQToken();

  void print();

  void setContent(void * obj_);
  void * getContent();
  TObject* getContentAsTObject();
  void setOwner(TObject * obj_);
  TObject * getOwner();
 
  virtual ~TQToken();
 
  ClassDefOverride(TQToken, 0); // container class to handle ownership of objects

};

#endif
