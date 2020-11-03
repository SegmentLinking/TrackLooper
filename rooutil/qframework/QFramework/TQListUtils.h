//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQListUtils__
#define __TQListUtils__

#include "TString.h"
#include "TList.h"
#include "QFramework/TQTaggable.h"
#include "TTree.h"


namespace TQListUtils {

  void print(TList * list, const TString& options = "");
  void print(TList * list, TQTaggable& options);
  void print(const std::vector<TString> & vec);
  
  TString makeCSV(const std::vector<TString> & vec);
 
  int removeElements(TList * l, const TString& filter);

  TList * getListOfNames(TCollection * collection, bool ownCollection = false);
 
  TList * getMergedListOfNames(TCollection * l1, TCollection * l2, bool ownCollections = false);

  bool reorder(TList * list, TList * order);
  bool isSubsetOf(TList * l1, TList * l2);
  bool areIsomorphic(TList * l1, TList * l2);
  bool hasDuplicates(TList * list);
  bool sortByStringLength(TList * list, bool ascending = true);

  int addClones(TCollection* from, TCollection* to);
  int moveListContents(TList* origin, TList* target);
  int collectContentNames(TCollection* origin, TList* target, const TString& pattern);
 
  TObject* findInList(TList* l, const TString& name);
  TQTaggable* findInListByTag(TList* l, const TString& key, const TString& value);

  void printList(TCollection* c);
  int getMaxNameLength(TCollection* c);

  int setBranchStatus(TTree* tree, TCollection* list, int status);

}

#endif
