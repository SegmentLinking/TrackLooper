//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQAlgorithm__
#define __TQAlgorithm__

#include "TQNamedTaggable.h"
#include <map>

class TQSample;

class TQAlgorithm : public TQNamedTaggable {
public:
  class Manager { // nested
  protected:
    std::vector<TQAlgorithm*> gAlgorithmList; //list of algorithm "templates", will be cloned for multiple channels
//    std::vector<std::vector<TQAlgorithm*>> gClonedAlgorithmList;
    std::vector<TQAlgorithm*> gActiveAlgorithms; //list of algs activated for current set of samples (MCASV)
    std::map<TQAlgorithm*,std::map<TString,TQAlgorithm*>> gAlgorithmStore; //list of all algs ever cloned including currently not active ones. First index is the original algorithm from which the channel specific ones are cloned. string is the name of the channel
    
  public:
    bool addAlgorithm(TQAlgorithm* newAlgorithm);
    //bool cloneAlgorithms(int n);
    void clearAlgorithms();
    void clearClonedAlgorithms();
    void resetActiveAlgorithms();
    const std::vector<TQAlgorithm*>& getAlgorithms();

    void printAlgorithms() const;
    void printActiveAlgorithms() const;
    bool initializeAlgorithms(TQSample*s);
    bool initializeClonedAlgorithms(std::vector<TQSample*>& samples, const TString& tagKey);
    bool finalizeAlgorithms();
    bool finalizeClonedAlgorithms();
    bool executeAlgorithms();
    bool executeClonedAlgorithms();
    bool cleanupAlgorithms();
    bool cleanupClonedAlgorithms();
  };
  
  virtual bool initializeSingleton(const std::vector<TQSample*>& samples); //at least one of the initialization methods must be overridden!
  virtual bool initialize(TQSample* s);
  virtual bool finalize() = 0;
  virtual bool execute() = 0; //called at the beginning of each event (i.e. before the first cut)
  
  virtual bool cleanup() = 0; //called at the end of each event (i.e. after all cuts)
  
  virtual bool isSingleton() const; //can be used to indicate that an algorithm should not be cloned when running multiple channels at once
  
  ClassDefOverride(TQAlgorithm,0) // algorithm to be run on an event before the event loop reads it

};
#endif
