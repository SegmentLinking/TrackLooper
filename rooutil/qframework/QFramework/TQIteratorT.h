//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQIteratorT__
#define __TQIteratorT__

#include "TObject.h"
#include "TIterator.h"

#include <TObject.h>
#include <TList.h>
#include <TCollection.h>
#include <TObjArray.h>


//////////////////////////////////////////////////////////////////////
//
// IMPORTANT NOTICE for adding custom class iterators:
//
// If you want to add a typedef and/or be able to access a TQ...Iterator
// fromt the ROOT shell, you need to undertake the following steps:
// - add a line 
// typedef TQIteratorT<MyClass> TQMyClassIterator; 
// to TQIterator.h
// - add an include statement
// #include <MyClass.h> 
// in the 'include'-section of TQIteratorT.h
// - add the line 
// #pragma link C++ class TQIteratorT<TQValue>+;
// to TQIteratorTLinkDef.h.add
// 
//////////////////////////////////////////////////////////////////////

#include <TObjString.h>
#include <TH1.h>
#include <TH2.h>
#include <TGraph.h>
#include <TLegendEntry.h>
#include <TGraphErrors.h>
#include <TGraphAsymmErrors.h>
#include <TBranch.h>

class TQPCA;
class TQCounter;
class TQCounterGrid;
class TQFolder;
class TQSampleFolder;
class TQSample;
class TQNamedTaggable;
class TQCut;
class TQObservable;
class TQAnalysisJob;
class TQValue;

template<class T> class TQIteratorT : public TObject {

protected:

  bool fOwnsCollection = false;
  bool fOwnsIterator = false;
  TCollection * fCollection = NULL;
  TIterator * fIterator = NULL;
  T * fNext = NULL;
  TString fFilter = "";
 
  int fCounter = 0;
  int fPreviousCounter = -1;
  int fGlobalCounter = 0;
  int fPreviousGlobalCounter = -1;
  int fNCycles = 1;
  int fCycle = 1;
  int fPreviousCycle = -1;
  int fCurrentIndex = -1;
  int fLastIndex = -1;
 
  bool gotoNext(bool switchCycle=true);


public:

  TQIteratorT();
  TQIteratorT(TIterator * itr, bool ownIterator = false);
  TQIteratorT(TIterator * itr, const TString& filter, bool ownIterator = false);
  TQIteratorT(TCollection * c, bool ownCollection = false);
  TQIteratorT(TCollection * c, const TString& filter, bool ownCollection = false);

  TCollection* getCollection();
  void setCollection(TCollection* c, bool ownCollection = false);
  void clear();
  
  bool hasNext();
  T * getNext();
  T * readNext();
  T * cast(TObject* obj);

  bool isValid() const;
 
  bool reset();
  bool resetCycle();
  int flush();
  bool sort();
 
  int scan(bool doPrintCounter = false);
  void printCounterHeadline(TString prefix = "");
  void printCounter(bool printHeadline = true);
 
  bool gotoElement(T * obj);
  bool gotoElement(const TString& name);
 
  void setNCycles(int nCycles);
  int getNCycles();
  int getCycle();
  int getPreviousCycle();
 
  int getCounter();
  int getPreviousCounter();
  int getGlobalCounter();
  int getPreviousGlobalCounter();
  int getLastIndex();
  int getCurrentIndex();

  void printContents() const;

  ~TQIteratorT();
 
  ClassDefT(TQIteratorT<T>,0) // iterator template class
 
};

ClassDefT2(TQIteratorT,T) 


#ifndef __CINT__
#ifndef ONLY_DECLARE_TPP
#include "QFramework/TQIteratorT.tpp"
#endif
#endif
#endif
