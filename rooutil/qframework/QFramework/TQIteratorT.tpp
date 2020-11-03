// -*- mode: c++ -*-

#include "QFramework/TQListUtils.h"
#include "QFramework/TQStringUtils.h"
#include <iostream>

#include "QFramework/TQPCA.h"
#include "QFramework/TQCounter.h"
#include "QFramework/TQFolder.h"
#include "QFramework/TQSampleFolder.h"
#include "QFramework/TQSample.h"
#include "QFramework/TQNamedTaggable.h"
#include "QFramework/TQCut.h"
#include "QFramework/TQObservable.h"
#include "QFramework/TQAnalysisJob.h"
#include "QFramework/TQValue.h"

templateClassImp(TQIteratorT)
ClassImpT(TQIteratorT,T)

//__________________________________________________________________________________|___________

template<class T>
inline void TQIteratorT<T>::clear() {
  // clear the collection and iterator inside this class
  if (fIterator && fOwnsIterator) {
    delete fIterator;
  }
  if (fOwnsCollection && fCollection) {
    delete fCollection;
  }
}

//__________________________________________________________________________________|___________

template<class T>
TCollection* TQIteratorT<T>::getCollection(){
  // retrieve the collection inside this iterator
  return this->fCollection;
}

//__________________________________________________________________________________|___________

template<class T>
void TQIteratorT<T>::setCollection(TCollection* c, bool ownCollection){
  // set the collection in this iterator
  this->clear();
  this->fCollection = c;
  this->fOwnsCollection = ownCollection;
}

//__________________________________________________________________________________|___________

template<class T>
TQIteratorT<T>::~TQIteratorT() { 
  // Destructor of TQIteratorT class:
  this->clear();
}

//__________________________________________________________________________________|___________

template<class T>
TQIteratorT<T>::TQIteratorT(){
  // default constructor of TQIteratorT class:
}

//__________________________________________________________________________________|___________
 
template<class T>
TQIteratorT<T>::TQIteratorT(TIterator * itr, bool ownIterator) :
  fOwnsIterator(ownIterator),
  fIterator(itr)
{
  // Constructor of TQIteratorT class
  this->gotoNext();
}

//__________________________________________________________________________________|___________
 
template<class T>
TQIteratorT<T>::TQIteratorT(TIterator * itr, const TString& filter, bool ownIterator): 
  fOwnsIterator(ownIterator),
  fIterator(itr),
  fFilter(filter)
{
  // Constructor of TQIteratorT class
  this->gotoNext();
}

//__________________________________________________________________________________|___________

template<class T>
TQIteratorT<T>::TQIteratorT(TCollection * c, bool ownCollection): 
  fOwnsCollection(ownCollection),
  fOwnsIterator(c),
  fCollection(c),
  fIterator(c ? c->MakeIterator() : NULL)
{
  // Constructor of TQIteratorT class
  this->gotoNext();
}

//__________________________________________________________________________________|___________

template<class T>
TQIteratorT<T>::TQIteratorT(TCollection * c, const TString& filter, bool ownCollection):
  fOwnsCollection(ownCollection),
  fOwnsIterator(c),
  fCollection(c),
  fIterator(c ? c->MakeIterator() : NULL),
  fFilter(filter)
{
  // Constructor of TQIteratorT class
  this->gotoNext();
}


//__________________________________________________________________________________|___________

template<class T>
inline T* TQIteratorT<T>::cast(TObject* obj){
  // cast any TObject pointer to the template class
  return dynamic_cast<T*>(obj);
}

//__________________________________________________________________________________|___________

template< >
inline TObject* TQIteratorT<TObject>::cast(TObject* obj){
  // cast any TObject pointer to the template class (TObject)
  return obj;
}

//__________________________________________________________________________________|___________

template<class T>
bool TQIteratorT<T>::gotoNext(bool switchCycle){
  // load the next element in the list

  // fCycle: number of times the list has been cycled through already
  // fPreviousCycle: active cycle of the last successfully loaded element
  // fNCycles: maximum number of cycles allowed
  // fLastInex: list index of the last successfully loaded list entry
  // fCurrentIndex: list index of the currently loaded list entry
  // fCounter: number of successfully retrieved elements in the current cycle
  // fGlobalCounter: number of successfully retrieved elements (overall)
  // fPreviousCounter: number of successfully retrieved elements in the current cycle, excluding the current one
  // fPreviousGlobalCounter: number of successfully retrieved elements in the current cycle, excluding the current one (overall)

  fLastIndex = fCurrentIndex;
 
  if (!this->fIterator) {
    this->fNext = NULL;
    return false;
  }
  this->fPreviousCounter = this->fCounter;
  this->fPreviousGlobalCounter = this->fGlobalCounter;
  this->fPreviousCycle = this->fCycle;
  do {
    TObject* obj = this->fIterator->Next();
    if(!obj){
      /* enter another cycle? */
      if (switchCycle) {
        this->fCycle++;
        if (this->fNCycles >= this->fCycle) {
          this->fIterator->Reset();
          this->fCounter = 0;
          this->fCurrentIndex = -1;
          continue;
        } else {
          this->fNext = NULL;
          return false;
        }
      }
    }
    this->fCurrentIndex++;
    if(!this->fFilter.IsNull() && !TQStringUtils::matchesFilter(obj->GetName(), this->fFilter, ",", true))
      {
        this->fNext = 0;
      }
    else
      {
        this->fNext = this->cast(obj);
      }
 
  } while (!this->fNext && (this->fNCycles >= this->fCycle || !switchCycle));
 
  if (!this->fNext) {
    return false;
  }
 
  fCounter++;
  fGlobalCounter++;
  return true;
}

//__________________________________________________________________________________|___________

template<class T>
bool TQIteratorT<T>::hasNext() {
  // return true if there is another element available, false otherwise
  return getNext() != NULL;
}


//__________________________________________________________________________________|___________

template<class T>
T * TQIteratorT<T>::getNext() {
  // retrieve next element in the list (without proceeding)
  return fNext;
}


//__________________________________________________________________________________|___________

template<class T>
T * TQIteratorT<T>::readNext() {
  // retrieve next element in the list (pre-loading the succeeding one)
  T * next = getNext();
  this->gotoNext();
  return next;
}


//__________________________________________________________________________________|___________

template<class T>
bool TQIteratorT<T>::reset() {
  // reset the iterator to the beginning of the list, clearing cycle history
  fCycle = 1;
  return resetCycle();
}


//__________________________________________________________________________________|___________

template<class T>
bool TQIteratorT<T>::resetCycle() {
  // reset the iterator to the beginning of the list, going back to the start of the current cycle
  if (fIterator) {
    fIterator->Reset();
    fNext = NULL;
    fCounter = 0;
    fPreviousCounter = -1;
    fGlobalCounter = 0;
    fPreviousGlobalCounter = -1;
    fLastIndex = -1;
    return this->gotoNext();
  } else {
    return false;
  }
}


//__________________________________________________________________________________|___________

template<class T>
int TQIteratorT<T>::flush() {
  // iterate over the entire list, returing the number of accepted elements
  int n = 0;
  while (this->readNext()) {
    n++;
  }
  return n;
}


//__________________________________________________________________________________|___________

template<class T>
bool TQIteratorT<T>::sort() {
  // sort the underlying list by name
  // does not change the original list
 
  this->reset();
 
  TList * sorted = new TList();
  sorted->SetOwner(false);
 
  bool stop = false;
  while (!stop && this->hasNext()) {
    TObject * obj = this->readNext();
    TString name = obj->GetName();
    if (!sorted->FindObject(name.Data())) {
      sorted->AddLast(obj);
    } else {
      stop = true;
    }
  }
 
  if (stop) {
    this->reset();
    return false;
  } else {
    if (fIterator && fOwnsIterator) {
      delete fIterator;
    }
    if (fOwnsCollection && fCollection) {
      delete fCollection;
    }
    fOwnsCollection = true;
    fOwnsIterator = true;
    fCollection = sorted;
    fIterator = sorted->MakeIterator();
    return this->gotoNext();
  }
}


//__________________________________________________________________________________|___________

template<class T>
int TQIteratorT<T>::scan(bool doPrintCounter) {
  // iterate over the entire list, printing the iterator status at each step (debugging functionality)

  TString headline = TString::Format("%-30s ", "Element Name");
  if (doPrintCounter) {
    printCounterHeadline(headline);
  } else {
    std::cout << TQStringUtils::makeBoldWhite(headline).Data() << std::endl;
    std::cout << TQStringUtils::makeBoldWhite(
                                              TQStringUtils::repeat("=", headline.Length())).Data() << std::endl;
  }
 
  if (doPrintCounter) {
    std::cout << TString::Format("%-30s ", ">> Post First Read <<").Data();
    if (doPrintCounter) {
      printCounter(false);
    } else {
      std::cout << std::endl;
    }
  }
 
  while (this->hasNext()) {
    TObject * obj = this->readNext();
    std::cout << TString::Format("%-30s ", obj->GetName()).Data();
    if (doPrintCounter) {
      printCounter(false);
    } else {
      std::cout << std::endl;
    }
  }
 
  return 0;
}


//__________________________________________________________________________________|___________

template<class T>
void TQIteratorT<T>::printCounterHeadline(TString prefix) {
  // print headline for the iterator status (debugging functionality)

  TString headline = prefix + TString::Format("%8s %8s %15s %10s %12s %12s %20s",
                                              "Counter",
                                              "Cycle",
                                              "GlobalCounter",
                                              "LastIndex",
                                              "PrevCounter",
                                              "PrevCycle",
                                              "PrevGlobalCounter");
 
  std::cout << TQStringUtils::makeBoldWhite(headline).Data() << std::endl;
  std::cout << TQStringUtils::makeBoldWhite(
                                            TQStringUtils::repeat("=", headline.Length())).Data() << std::endl;
}


//__________________________________________________________________________________|___________

template<class T>
void TQIteratorT<T>::printCounter(bool printHeadline) {
  // print the current iterator status (debugging functionality)

  if (printHeadline) {
    printCounterHeadline();
  }
 
  std::cout << TString::Format("%8s %8s %15s %10s %12s %12s %20s",
                               TString::Format("%d", getCounter()).Data(),
                               TString::Format("%d", getCycle()).Data(),
                               TString::Format("%d", getGlobalCounter()).Data(),
                               TString::Format("%d", getLastIndex()).Data(),
                               TString::Format("%d", getPreviousCounter()).Data(),
                               TString::Format("%d", getPreviousCycle()).Data(),
                               TString::Format("%d", getPreviousGlobalCounter()).Data()).Data() << std::endl;
}

//__________________________________________________________________________________|___________

template<class T>
void TQIteratorT<T>::setNCycles(int nCycles) {
  // set the maximum number of cycles allowed
  if (nCycles > 0) {
    fNCycles = nCycles;
  }
}


//__________________________________________________________________________________|___________

template<class T>
int TQIteratorT<T>::getNCycles() {
  // get the maximum number of cycles allowed
  return fNCycles;
}


//__________________________________________________________________________________|___________

template<class T>
int TQIteratorT<T>::getCycle() {
  // retrieve the number of times the list has been cycled through already
  return fCycle;
}


//__________________________________________________________________________________|___________

template<class T>
int TQIteratorT<T>::getPreviousCycle() {
  // retrieve the active cycle of the last successfully loaded element
  return fPreviousCycle;
}


//__________________________________________________________________________________|___________

template<class T>
int TQIteratorT<T>::getCounter() {
  // return the number of successfully retrieved elements in the current cycle
  return fCounter;
}


//__________________________________________________________________________________|___________

template<class T>
int TQIteratorT<T>::getPreviousCounter() {
  // return the number of successfully retrieved elements in the current cycle, excluding the current one
  return fPreviousCounter;
}

//__________________________________________________________________________________|___________

template<class T>
int TQIteratorT<T>::getGlobalCounter() {
  // return the number of successfully retrieved elements (overall)
  return fGlobalCounter;
}


//__________________________________________________________________________________|___________

template<class T>
int TQIteratorT<T>::getPreviousGlobalCounter() {
  // return the number of successfully retrieved elements in the current cycle, excluding the current one (overall)
  return fPreviousGlobalCounter;
}


//__________________________________________________________________________________|___________

template<class T>
int TQIteratorT<T>::getLastIndex() {
  // return the list index of the last successfully loaded list entry
  return fLastIndex;
}

//__________________________________________________________________________________|___________

template<class T>
int TQIteratorT<T>::getCurrentIndex() {
  // return the list index of the currently loaded list entry
  return fCurrentIndex;
}

//__________________________________________________________________________________|___________



template<class T>
bool TQIteratorT<T>::isValid() const{
  // return true if the iterator was initialize correctly (with a valid list or TIterator), false otherwise
  if (this->fIterator) return true;
  return false;
}


//__________________________________________________________________________________|___________

template<class T>
void TQIteratorT<T>::printContents() const{
  // print the contents of the underlying list (debugging functionality)
  TQListUtils::printList(this->fCollection);
}

//__________________________________________________________________________________|___________

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

template<class T>
bool TQIteratorT<T>::gotoElement(T * obj) {
  // currently not implemented
  return false;
}

template<class T>
bool TQIteratorT<T>::gotoElement(const TString& name) {
  // currently not implemented
  return false;
}

#pragma GCC diagnostic pop

