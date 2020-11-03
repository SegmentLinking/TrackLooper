#ifdef _UNICODE
typedef wchar_t TCHAR;
#else
typedef char TCHAR;
#endif

#include "QFramework/TQListUtils.h"
#include "QFramework/TQStringUtils.h"
#include "QFramework/TQHistogramUtils.h"
#include "QFramework/TQIterator.h"
#include "TObjString.h"
#include "QFramework/TQLibrary.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

using std::vector;
using std::string;


////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQListUtils:
//
// The TQListUtils namespace provides a set of static utility methods related to lists.
//
////////////////////////////////////////////////////////////////////////////////////////////////

void TQListUtils::print(TList * list, const TString& options) {
	// print a list with a given set of options
  TQTaggable * opts = TQTaggable::parseFlags(options);
  if (!opts) {
    ERRORfunc("Failed to parse options '%s'", options.Data());
    return;
  }
 
  // now print the list
  print(list, *opts);
}


//__________________________________________________________________________________|___________

void TQListUtils::print(TList * list, TQTaggable& options) {
	// print a list with a given set of options
  TString errMsg = "option";
  if (!options.claimTags("[f:s!],[s:b!],!!", errMsg)) {
    ERRORfunc("Failed to parse options '%s'", errMsg.Data());
    return;
  }
 
  bool ownList = false;
  if (list && options.getTagBoolDefault("s", false)) {
    list = getListOfNames(list);
    list->Sort();
    ownList = true;
  }

  TQIterator itr(list, options.getTagStringDefault("f"), ownList);
  while (itr.hasNext()) {
    TString name = itr.readNext()->GetName();
 
    // print element
    std::cout << name.Data() << std::endl;
  }
}


//__________________________________________________________________________________|___________

void TQListUtils::print(const std::vector<TString> & vec) {
	// print a vector of strings
  for (size_t i = 0; i< vec.size() ; ++i) {
    std::cout<< "["<<i<<"]"<<vec.at(i).Data()<<std::endl;
  }
}
//__________________________________________________________________________________|___________

TList * TQListUtils::getListOfNames(TCollection * collection, bool ownCollection) {
  // Returns a list (pointer to instance of TList) of instances of TObjString of
  // names of elements in <collection> or a null pointer in case no element is
  // present (the user is responsible for deleting the returned instance). If
  // <ownCollection> is true object <collection> will be deleted after having been
  // read.
 
  // will be the list to return
  TList * result = NULL;
 
  TQIterator itr(collection, ownCollection);
  while (itr.hasNext()) {
    // create new instance of TList for first element to add
    if (!result) {
      result = new TList();
      result->SetOwner(true);
    }
    result->Add(new TObjString(itr.readNext()->GetName()));
  }
 
  return result;
}


//__________________________________________________________________________________|___________

TList * TQListUtils::getMergedListOfNames(TCollection * c1, TCollection * c2, bool ownCollections) {
  // Returns a sorted list (pointer to instance of TList) of instances of
  // TObjString of names of elements present in at least one of the two collections
  // <c1> and <c2> (the user is responsible for deleting the returned instance). If
  // <ownCollections> is true objects <c1> and <c2> will be deleted after having
  // been read. 
 
  // get list of names of elements in collections
  TList * l1 = getListOfNames(c1, ownCollections);
  TList * l2 = getListOfNames(c2, ownCollections);
 
  // will be the resulting list
  TList * result = NULL;
 
  // the number of entries in each list
  int n1 = 0;
  int n2 = 0;

  // sort the lists by name
  if (l1) {
    n1 = l1->GetEntries();
    l1->Sort();
  }
  if (l2) {
    n2 = l2->GetEntries();
    l2->Sort();
  }
 
  // process the two lists by incrementing list indices in a synchronized way
  int i1 = 0;
  int i2 = 0;
  while (i1 < n1 || i2 < n2) {
 
    // get the two objects at the current list indices
    // (use NULL pointer in case a list has reached its end)
    TObject * obj1 = (i1 < n1) ? l1->At(i1) : NULL;
    TObject * obj2 = (i2 < n2) ? l2->At(i2) : NULL;

    TObject * obj = NULL;
 
    if (obj1 && obj2) {
      // Compare objects based on their Compare(...) method (usually compares names)
      int compare = obj1->Compare(obj2);
      // go to next element in list 1 if list 2 is ahead or both are at same level
      if (compare <= 0) {
        obj = obj1;
        i1++;
      }
      // go to next element in list 2 if list 1 is ahead or both are at same level
      if (compare >= 0) {
        obj = obj2;
        i2++;
      }
    } else if (obj1) {
      // list 2 has reached its end
      obj = obj1;
      i1++;
    } else if (obj2) {
      // list 1 has reached its end
      obj = obj2;
      i2++;
    }

    if (!obj) {
      continue;
    }
    if (!result) {
      result = new TList();
      result->SetOwner(true);
    }
    result->Add(new TObjString(obj->GetName()));
  }

  // clean up
  if (l1) {
    delete l1;
  }
  if (l2) {
    delete l2;
  }
 
  return result;
}


//__________________________________________________________________________________|___________

int TQListUtils::removeElements(TList * l, const TString& filter) {
  // Removes all elements from input list <l> with names (string returned by
  // GetName()) matching the string filter pattern <filter> and returns the number
  // of elements that have been removed. If the list <l> is the owner of its objects
  // (l->IsOwner() is true) elements will also be deleted.
 
  // get list of names in list
  TList * names = getListOfNames(l, false);
 
  // number of objects removed from list
  int n = 0;
 
  // iterate over object names matching filter
  TQIterator itr(names, filter, true);
  while (itr.hasNext()) {
    TString name = itr.readNext()->GetName();
    TObject * obj = l->FindObject(name.Data());
    if (!obj) {
      // should not happen
      continue;
    }
    l->Remove(obj);
    n++;
    if (l->IsOwner()) {
      delete obj;
    }
  }

  // return the number of elements that have been removed
  return n;
}


//__________________________________________________________________________________|___________

bool TQListUtils::reorder(TList * list, TList * order) {
 
  if (!list || !order || !areIsomorphic(list, order)) {
    // invalid or non-matching input lists
    return false;
  }
 
  // temporarily make the list not being the owner of
  // objects but remeber owner setting to restore it later
  bool isOwner = list->IsOwner();
  list->SetOwner(false);
 
  TQIterator itr(order);
  while (itr.hasNext()) {
    TString name = itr.readNext()->GetName();
    // move object
    TObject * obj = list->FindObject(name.Data());
    if (!obj) {
      // should not happen
      return false;
    }
    list->Remove(obj);
    list->AddLast(obj);
  }
 
  // restore original owner setting
  list->SetOwner(isOwner);
 
  // successfully done!
  return true;
}


//__________________________________________________________________________________|___________

bool TQListUtils::isSubsetOf(TList * l1, TList * l2) {
  // Returns true if for each element in list <l1> there is an element with the same
  // name in list <l2> and false otherwise.
 
  // iterate over objects in list <l1>
  TQIterator itr(l1);
  while (itr.hasNext()) {
    TString name = itr.readNext()->GetName();
    if (!l2 || !l2->FindObject(name.Data())) {
      // no matching element found
      return false;
    }
  }

  // found match for every element
  return true;
} 


//__________________________________________________________________________________|___________

bool TQListUtils::areIsomorphic(TList * l1, TList * l2) {
  // Returns true if the two input lists <l1> and <l2> are isomorphic and false
  // otherwise. Isomorphism in this context means that for element in one list there
  // is exactly one object with the same name in the other list (implying in neither
  // of the two lists there are two or more objects with the same name). Please note:
  // in case both <l1> and <l2> are invalid pointer true is returned.
 
  if (!l1 || !l2) {
    // return true if both input lists are invalid
    return !l1 && !l2;
  }
  if (hasDuplicates(l1) || hasDuplicates(l2)) {
    // don't allow duplicates in any list
    return false;
  }
 
  // list match if <l1> and <l2> are subsets of each other
  return isSubsetOf(l1, l2) && isSubsetOf(l2, l1);
}


//__________________________________________________________________________________|___________

bool TQListUtils::hasDuplicates(TList * list) {
  // Returns true if there are two or more objects in the input list <list> with
  // the same name and false otherwise.
 
  // remember object names already found in the list
  TList found;
  found.SetOwner(true);
 
  // iterate over objects in list
  TQIterator itr(list);
  while (itr.hasNext()) {
    TString name = itr.readNext()->GetName();
    if (found.FindObject(name.Data())) {
      // already had the same name
      return true;
    } else {
      found.Add(new TObjString(name.Data()));
    }
  }

  // no duplicates found
  return false;
}


//__________________________________________________________________________________|___________

bool TQListUtils::sortByStringLength(TList * list, bool ascending) {  // TODO: still needs to be implemented
  return true;
}

//__________________________________________________________________________________|___________

int TQListUtils::addClones(TCollection* from, TCollection* to){
  // clone all objects contained in one list and append them to the other
  if(!to) return -1;
  TQIterator itr(from);
  size_t n=0;
  while(itr.hasNext()){
    n++;
    TObject* obj = itr.readNext();
    to->Add(obj->Clone());
  }
  return n;
}

//__________________________________________________________________________________|___________

int TQListUtils::moveListContents(TList* origin, TList* target){
  // move all objects contained in one list to the other
  if(!origin || !target) return 0;
  TQIterator itr(origin);
  int nMoved = 0;
  while(itr.hasNext()){
    TObject* obj = itr.readNext();
    if(!obj) continue;
    target->Add(obj);
    nMoved++;
  }
  return nMoved;
}

//__________________________________________________________________________________|___________

TObject* TQListUtils::findInList(TList* l, const TString& name){
  // find the first object in a list matching the given name
  // name pattern may contain wildcars
  if(!l) return NULL;
  TQIterator itr(l);
  while(itr.hasNext()){
    TObject* obj = itr.readNext();
    if(!obj) continue;
    if(TQStringUtils::matches(obj->GetName(),name))
      return obj;
  }
  return NULL;
}

//__________________________________________________________________________________|___________

TQTaggable* TQListUtils::findInListByTag(TList* l, const TString& key, const TString& value){
  // find the first TQTaggable object 
  // that contains the given key-value-pair 
  // value may contain wildcards.
  if(!l) return NULL;
  TQIterator itr(l);
  while(itr.hasNext()){
    TQTaggable* obj = dynamic_cast<TQTaggable*>(itr.readNext());
    if(!obj) continue;
    if(TQStringUtils::matches(obj->getTagStringDefault(key,""),value))
      return obj;
  }
  return NULL;
}

//__________________________________________________________________________________|___________

int TQListUtils::getMaxNameLength(TCollection* c){
  // retrieve the maximum object name length from some collection
  int max = -1;
  TQIterator itr(c);
  while(itr.hasNext()){
    TObject* obj = itr.readNext();
    max = std::max(max,TQStringUtils::getWidth(obj->GetName()));
  }
  return max;
}

//__________________________________________________________________________________|___________

void TQListUtils::printList(TCollection* c){
  // print any type of TCollection nicely formatted
  if(!c){
    ERRORfunc("cannot print NULL list");
    return;
  }
  std::cout << TQStringUtils::makeBoldWhite(c->GetName()) << " has " << c->GetEntries() << " entries:" << std::endl;
  TQIterator itr(c);
  while(itr.hasNext()){
    TObject* obj = dynamic_cast<TObject*>(itr.readNext());
    if(!obj){
      std::cout << TQStringUtils::fixedWidth("TObject",20,false) << " " << TQStringUtils::fixedWidth("NULL",60,false) << std::endl;
      continue;
    }
    std::cout << TQStringUtils::fixedWidth(obj->ClassName(),20,false) << " " << TQStringUtils::fixedWidth(obj->GetName(),60,false) << std::endl;
  }
}

//__________________________________________________________________________________|___________

int TQListUtils::collectContentNames(TCollection* origin, TList* target, const TString& pattern){
  // create a list of all names ob objects contained in some collection
  // that match a given pattern. returns number of matches found.
  if(!origin || !target) return 0;
  TQIterator itr(origin);
  int nMoved = 0;
  while(itr.hasNext()){
    TNamed* named = dynamic_cast<TNamed*>(itr.readNext());
    if(!named) continue;
    if(!TQStringUtils::matches(named->GetName(),pattern))
      continue;
    if(target->Contains(named->GetName()))
      continue;
    target->Add(new TObjString(named->GetName()));
    nMoved++;
  }
  return nMoved;
}

//__________________________________________________________________________________|___________

int TQListUtils::setBranchStatus(TTree* tree, TCollection* list, int status){
  // loop over a list of objects, setting the status of all branches matching the list entry names
  TQIterator itr(list);
  int count = 0;
  while(itr.hasNext()){
    TObject* bName = itr.readNext();
    if(!bName) continue;
    TString name(bName->GetName());
    if (name.First('*') != kNPOS || tree->FindBranch(name)) tree->SetBranchStatus(name, status);
    count++;
  }
  return count;
}

//__________________________________________________________________________________|___________

TString TQListUtils::makeCSV(const std::vector<TString>& vec) {
  TString str = "";
  for (size_t i=0; i<vec.size();++i) {
    str+=vec.at(i)+", ";
  }
  TQStringUtils::removeTrailingText(str,", ");
  return str;
}
