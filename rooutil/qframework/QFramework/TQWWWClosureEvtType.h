//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQWWWCLOSUREEVTTYPE__
#define __TQWWWCLOSUREEVTTYPE__
#include "QFramework/TQTreeObservable.h"

#include <vector>

class TQWWWClosureEvtType : public TQTreeObservable {
protected:
  // put here any data members your class might need
  std::vector<int>* genPart_motherId;
  std::vector<int>* genPart_pdgId;
 
public:
  virtual double getValue() const override;
  virtual TObjArray* getBranchNames() const override;
protected:
  virtual bool initializeSelf() override;
  virtual bool finalizeSelf() override;
public:
  TQWWWClosureEvtType();
  TQWWWClosureEvtType(const TString& name);
  virtual ~TQWWWClosureEvtType();
  ClassDefOverride(TQWWWClosureEvtType, 1);


};
#endif
