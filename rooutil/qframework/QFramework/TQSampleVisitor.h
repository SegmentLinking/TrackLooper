//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQSampleVisitor__
#define __TQSampleVisitor__

#include "TNamed.h"
#include "TQStringUtils.h"

class TQSample;
class TQSampleFolder;
class TQTaggable;

class TQSampleVisitor : public TNamed {

protected:
 
  int fSampleColWidth; 
  bool fVerbose;
 
  TString fVisitTraceID;
  TString fStatusLine; //store the last printed status line to allow updates from child classes (e.g. indicating event loop progress)

  static const TString statusSKIPPED;
  static const TString statusOK;
  static const TString statusFAILED;
  static const TString statusWARN;
  static const TString statusRUNNING;
  inline static TString statusPROGRESS(double fraction) {return TQStringUtils::makeBoldWhite("[ ")+TQStringUtils::makeBoldWhite (TQStringUtils::fixedWidth(TString::Format("%.0f%%",fraction*100.),4,'c') )+TQStringUtils::makeBoldWhite(" ]");}; // %% is escaped '%' character
  

  virtual int visitFolder(TQSampleFolder * sampleFolder, TString& message);
  virtual int visitSample(TQSample * sample, TString& message);
  virtual int revisitSample(TQSample * sample, TString& message);
  virtual int revisitFolder(TQSampleFolder * sampleFolder, TString& message);

  inline TString printLine(TQSampleFolder* f, int level, bool isSample, const TString& bullet);
  void updateLine(const TString& line, const TString& message, int result, bool ignore=false, double progress = 0.);
  inline void leaveTrace(TQSampleFolder* sf, TString prefix, int result, const TString& message);

  bool callInitialize(TQSampleFolder * sampleFolder);
  int callVisit(TQSampleFolder * sampleFolder, int level, bool requireSelectionTag = false);
  bool callFinalize();
  
  
public:

  enum visitSTATUS {
    visitIGNORE = 9,
    visitLISTONLY = 8,
    visitPROGRESS = 7,
    
    visitSKIPPED = 0,
    visitOK = 1,
    visitWARN = 2,
    visitFAILED = 3
  };
 
  static TString getStatusString(int status, double progress=0.);

  TQSampleVisitor(const TString& name = "vis");
 
  void setVerbose(bool verbose = true);

  /* called before the first element is visited */
  virtual int initialize(TQSampleFolder * sampleFolder, TString& message);
  /* called after last element was visited */
  virtual int finalize();

  int visit(TQSampleFolder * sampleFolder, bool requireSelectionTag = false);

  bool setVisitTraceID(TString id);
  TString getVisitTraceID() const ;
  const char* getVisitTraceIDConst() const;

  void stamp(TQTaggable* obj) const;
  bool checkVisit(TQTaggable* obj) const;
  static bool checkRestrictionTag(TQSampleFolder* sf);
  
  virtual ~TQSampleVisitor();
 
  ClassDefOverride(TQSampleVisitor, 0); //QFramework class

};

#endif
