//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQUniqueCut__
#define __TQUniqueCut__

#include "QFramework/TQObservable.h"
#include "QFramework/TQCompiledCut.h"

class TQUniqueCut : public TQCompiledCut {
protected:
  TString runNumberBranch;
  TString eventNumberBranch;
  TQObservable* runNumberObservable;
  TQObservable* eventNumberObservable;

  mutable std::vector<int> runNumbers;
  mutable std::vector<std::vector<int> >eventNumbers;

  bool enabled;

  void initUniqueCut();

  virtual bool initializeObservables() override;
  virtual bool finalizeObservables() override;
  virtual bool initializeSelfSampleFolder(TQSampleFolder* sf) override;
  virtual bool finalizeSelfSampleFolder(TQSampleFolder* sf) override;
  
  virtual TObjArray* getOwnBranches() override;

public:
  static bool checkUnique(std::vector<int>& entries, int newEntry);
  static int getIndex(std::vector<int>& entries, int entry);

  void clear();
  void setActive(bool active);

  bool isMergeable() const override;
  bool setBranchNames(const TString& runBranch, const TString& evtBranch);
  TQUniqueCut();
  TQUniqueCut(const TString& name);
  TQUniqueCut(const TString& runBranch, const TString& evtBranch);
  TQUniqueCut(const TString& name, const TString& runBranch, const TString& evtBranch);

  virtual bool passed() const override;

  void printLists() const;
  bool isSorted(int verbose=1) const;

  ClassDefOverride(TQUniqueCut, 0); // cut to remove duplicate events
};

 
#endif
