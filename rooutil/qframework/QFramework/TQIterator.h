//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQIterator__
#define __TQIterator__

#define ONLY_DECLARE_TPP
#include "QFramework/TQIteratorT.h"
#undef ONLY_DECLARE_TPP

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

typedef TQIteratorT<TObject> TQIterator;
typedef TQIteratorT<TQValue> TQValueIterator;
typedef TQIteratorT<TQCounter> TQCounterIterator;
typedef TQIteratorT<TQCut> TQCutIterator;
typedef TQIteratorT<TObjString> TQStringIterator;
typedef TQIteratorT<TQFolder> TQFolderIterator;
typedef TQIteratorT<TQSampleFolder> TQSampleFolderIterator;
typedef TQIteratorT<TQSample> TQSampleIterator;
typedef TQIteratorT<TH1> TQTH1Iterator;
typedef TQIteratorT<TQNamedTaggable> TQTaggableIterator;
typedef TQIteratorT<TQObservable> TQObservableIterator;
typedef TQIteratorT<TQAnalysisJob> TQAnalysisJobIterator;
typedef TQIteratorT<TGraphErrors> TQGraphErrorsIterator;
typedef TQIteratorT<TGraph> TQGraphIterator;
typedef TQIteratorT<TGraphAsymmErrors> TQGraphAsymmErrorsIterator;
typedef TQIteratorT<TCollection> TQCollectionIterator;
typedef TQIteratorT<TList> TQListIterator;
typedef TQIteratorT<TLegendEntry> TQLegendEntryIterator;
typedef TQIteratorT<TBranch> TQBranchIterator;

#ifndef __CINT__
#include "QFramework/TQIteratorT.tpp"
#endif

#endif
