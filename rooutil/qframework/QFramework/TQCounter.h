//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQCounter__
#define __TQCounter__

#include "TNamed.h"

class TQCounter : public TNamed {

protected:
 
  double fCounter; // counter
  double fErrorSquared; // error
  int fRawCounter; // raw counter

  // TODO: remove
  bool fWarning;


public:

  static TString getComparison(TQCounter * cnt1, TQCounter * cnt2, bool colored = false, double order = 1E-7);

  TQCounter();
  TQCounter(const char * name);
  TQCounter(const char * name, const char* title);
  TQCounter(const TString& name, double cnt, double err = 0.);
  TQCounter(const TString& name, double cnt, double err, int raw);
  TQCounter(TQCounter * counter);

  double getCounter();
  void setCounter(double counter);
  int getRawCounter();
  void setRawCounter(int raw);

  double getError();
  double getErrorSquared();
  void setErrorSquared(double errorSquared);
  void setError(double error);

  double getStatError();
  double getStatErrorSquared();

  double getSysError();
  double getSysErrorSquared();

 
  void reset();

  TString getAsString(const TString& options = "");
  void print(const TString& options = "");
  void printCounter(const TString& options = "");

  void setWarning(bool warning = true);
  bool getWarning();
 
  void add(double weight);
 
 
  void add(TQCounter * counter, double scale = 1., double scaleUncertainty = 0., double correlation = 0., bool includeScaleUncertainty = true);
  void add(TQCounter * counter, TQCounter * scale, double correlation = 0., bool includeScaleUncertainty = false);
  void subtract(TQCounter * counter, double scale = 1., double scaleUncertainty = 0., double correlation = 0., bool includeScaleUncertainty = true);
  void subtract(TQCounter * counter, TQCounter* scale, double correlation = 0., bool includeScaleUncertainty = false);
  void multiply(double factor, double uncertainty = 0., double correlation = 0., bool includeScaleUncertainty = true);
  void multiply(TQCounter * counter, double correlation = 0., bool includeScaleUncertainty = false);
  void divide(double denominator, double uncertainty = 0., double correlation = 0., bool includeScaleUncertainty = true);
  void divide(TQCounter * counter, double correlation = 0, bool includeScaleUncertainty = false);
 
  void scale(double factor, double scaleUncertainty = 0., bool includeScaleUncertainty = true);
  void scale(TQCounter * scale, bool includeScaleUncertainty = true);

  bool isEqualTo(TQCounter * counter, double order = 1E-7);
  bool isRawEqualTo(TQCounter * counter);

  virtual ~TQCounter();

  ClassDefOverride(TQCounter, 1); // event counter
};

#endif

