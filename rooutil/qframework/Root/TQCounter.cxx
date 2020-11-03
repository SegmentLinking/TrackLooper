#include <iostream>

#include "QFramework/TQCounter.h"
#include "QFramework/TQStringUtils.h"
#include "TMath.h"

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQCounter
//
// A TQCounter is a counter for events, counting weighted und
// unweighted event numbers. In essence, it is the same as a histogram
// with a single bin, with a simplified interface.
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQCounter)


//__________________________________________________________________________________|___________

TQCounter::TQCounter() : TNamed() {
  // default constructor, required by ROOT streamer
  reset();
}


//__________________________________________________________________________________|___________

TQCounter::TQCounter(const char * name) : TNamed(name, "") {
  // constructor taking only the object name
  reset();
}

//__________________________________________________________________________________|___________

TQCounter::TQCounter(const char * name, const char* title) : TNamed(name, title) {
  // constructor taking only the object name and title
  reset();
}


//__________________________________________________________________________________|___________

TQCounter::TQCounter(const TString& name, double cnt, double err) : TNamed(name.Data(), "") {
  // constructor taking object name as well as value and uncertainty
  reset();

  setCounter(cnt);
  setErrorSquared(err * err);
}

//__________________________________________________________________________________|___________

TQCounter::TQCounter(const TString& name, double cnt, double err, int raw) : TNamed(name.Data(), "") {
  // constructor taking object name as well as value and uncertainty
  reset();

  setCounter(cnt);
  setErrorSquared(err * err);
  setRawCounter(raw);

}


//__________________________________________________________________________________|___________

TQCounter::TQCounter(TQCounter * counter) : TNamed(counter ? counter->GetName() : "TQCounter", counter ? counter->GetTitle() : "") {
  // Create an exact copy of a counter
 
  reset();
 
  /* get the status of the foreign counter */
  if (counter) {
    add(counter);
  }
 
}


//__________________________________________________________________________________|___________

void TQCounter::reset() {
  // Reset this counter

  fCounter = 0.;
  fErrorSquared = 0.;
  fRawCounter = 0;
  fWarning = false;
}


//__________________________________________________________________________________|___________

TString TQCounter::getAsString(const TString& options) {
  // retrieve the contents of this counter 
  // as a nicely formatted string

  TString flags;
  TString localOptions = options;

  int precision = -1;
  bool hasPrecision = false;

  /* read flags */
  bool stop = false;
  while (!stop) {
    /* read precision flag "p" */
    if (TQStringUtils::readToken(localOptions, flags, "p") > 0) {
      /* read precision definition after 'p' option */
      TString precisionStr;
      if (TQStringUtils::readToken(localOptions, precisionStr,
                                   TQStringUtils::getNumerals()) > 0 && !hasPrecision) {
        precision = precisionStr.Atoi();
        hasPrecision = true;
        continue;
      } else {
        return "";
      }
    }

    /* no valid tokens left to parse */
    stop = true;
  }

  /* unexpected options left? */
  if (localOptions.Length() > 0)
    return "";

  /* prepare string of numbers */
  TString cnt = TString::Format("%.*g", precision, (double)getCounter());
  TString err = TString::Format("%.*g", precision, (double)getError());
  TString raw = TString::Format("%.*g", precision, (double)getRawCounter());

  if (precision > 0)
    precision += 1;

  /* return a string representing this counter */
  return TString::Format("%*s +/- %*s [raw: %*s]", precision, cnt.Data(),
                         precision, err.Data(), precision, raw.Data());
}


//__________________________________________________________________________________|___________

void TQCounter::print(const TString& options) {
  // print a string representing this counter 
  this->printCounter(options);
}

//__________________________________________________________________________________|___________

void TQCounter::printCounter(const TString& options) {
  // print a string representing this counter 
  std::cout << TString::Format("TQCounter('%s'): %s\n", GetName(),getAsString(options).Data()).Data();
}

//__________________________________________________________________________________|___________

double TQCounter::getCounter() {
  // retrieve the value of this counter
  return fCounter;
}


//__________________________________________________________________________________|___________

void TQCounter::setCounter(double counter) {
  // set the value of this counter
  fCounter = counter;
}


//__________________________________________________________________________________|___________

void TQCounter::setRawCounter(int raw) {
  // set the raw value of this counter
  fRawCounter = raw;
}


//__________________________________________________________________________________|___________

double TQCounter::getError() {
  // return the total error (uncertainty)

  return TMath::Sqrt(fErrorSquared);
}


//__________________________________________________________________________________|___________

double TQCounter::getErrorSquared() {
  // return the total variance (square uncertainty)

  return fErrorSquared;
}


//__________________________________________________________________________________|___________

void TQCounter::setErrorSquared(double errorSquared) {
  // set the total variance (square uncertainty)
  if (errorSquared >= 0.)
    fErrorSquared = errorSquared;
}

//__________________________________________________________________________________|___________

void TQCounter::setError(double error) {
  // set the total uncertainty (sqrt(variance)). Please note that internally the 
  // squared value is stored, signs are therefore dropped
  fErrorSquared = pow(error,2);
}

//__________________________________________________________________________________|___________

double TQCounter::getStatError() {
  // retrieve the statistical error
  // currently not implemented
  return 0.;
}


//__________________________________________________________________________________|___________

double TQCounter::getStatErrorSquared() {
  // retrieve the statistical error
  // currently not implemented
  return 0.;
}


//__________________________________________________________________________________|___________

double TQCounter::getSysError() {
  // retrieve the statistical error
  // currently not implemented
  return 0.;
}


//__________________________________________________________________________________|___________

double TQCounter::getSysErrorSquared() {
  // retrieve the systematic error
  // currently not implemented
  return 0.;
}


//__________________________________________________________________________________|___________

int TQCounter::getRawCounter() {
  // Return the number of raw counts

  return fRawCounter;
}


//__________________________________________________________________________________|___________

void TQCounter::setWarning(bool warning) {
  // set the 'warning' flag on this counter
  fWarning = warning;
}


//__________________________________________________________________________________|___________

bool TQCounter::getWarning() {
  // retrieve the value of the 'warning' flag
  return fWarning;
}


//__________________________________________________________________________________|___________

void TQCounter::add(double weight) {
  // add one entry with a given weight to this counter
  fCounter += weight;
  fErrorSquared += pow(weight, 2);
  fRawCounter += 1;
}


//__________________________________________________________________________________|___________

void TQCounter::add(TQCounter * counter, double scale, double scaleUncertainty, double correlation, bool includeScaleUncertainty) {
  // add the contents of another counter to this one optionally including scale uncertainty and a possible correlation between the two counters.
  // "scale" is treated as being uncorrelated to "counter" and the TQCounter this method is called on.
  double scaleUncertSq = includeScaleUncertainty ? TMath::Power(scaleUncertainty,2) : 0. ;
  if (counter) {
    fCounter += scale * counter->getCounter();
    fErrorSquared += TMath::Power(scale, 2.) * counter->getErrorSquared() + TMath::Power(counter->getCounter(),2) * scaleUncertSq + 2. * correlation * scale * TMath::Sqrt(counter->getErrorSquared()) * TMath::Sqrt(fErrorSquared) ;
    fRawCounter += counter->getRawCounter();
    fWarning |= counter->getWarning();
  }
}

//__________________________________________________________________________________|___________

void TQCounter::add(TQCounter * counter, TQCounter * scale, double correlation, bool includeScaleUncertainty) {
  // add the contents of another counter to this one optionally including scale uncertainty and a possible correlation between the two counters.
  // "scale" is treated as being uncorrelated to "counter" and the TQCounter this method is called on.
  if (counter) {
    this->add(counter, scale->getCounter(), scale->getError(), correlation, includeScaleUncertainty);
  }
}

//__________________________________________________________________________________|___________

void TQCounter::subtract(TQCounter * counter, double scale, double scaleUncertainty, double correlation, bool includeScaleUncertainty) {
  // subtract the contents of another counter from this one optionally including scale uncertainty and a possible correlation between the two counters.
  // "scale" is treated as being uncorrelated to "counter" and the TQCounter this method is called on.
  double scaleUncertSq = includeScaleUncertainty ? TMath::Power(scaleUncertainty,2) : 0.;
  if (counter) {
    fCounter -= scale * counter->getCounter();
    fErrorSquared += TMath::Power(scale, 2.) * counter->getErrorSquared() + TMath::Power(counter->getCounter(),2) * scaleUncertSq - 2. * correlation * scale * TMath::Sqrt(counter->getErrorSquared()) * TMath::Sqrt(fErrorSquared) ; //note the -2*corrleation compared to +2*correlation in TQCounter::add(...) !
    fRawCounter += counter->getRawCounter();
    fWarning |= counter->getWarning();
  }
}

//__________________________________________________________________________________|___________

void TQCounter::subtract(TQCounter * counter, TQCounter* scale, double correlation, bool includeScaleUncertainty) {
  // subtract the contents of another counter from this one optionally including scale uncertainty and a possible correlation between the two counters.
  // "scale" is treated as being uncorrelated to "counter" and the TQCounter this method is called on.
  if (counter) {
    this->subtract(counter, scale->getCounter(), scale->getError(), correlation, includeScaleUncertainty);
  }
}

//__________________________________________________________________________________|___________

void TQCounter::multiply(double factor, double uncertainty, double correlation, bool includeScaleUncertainty) {
  // multiply the contents of this counter with the contents of another one including correlation for the error propagation if given.
  double uncertSq = includeScaleUncertainty ? TMath::Power(uncertainty,2) : 0.;
  fErrorSquared = uncertSq*TMath::Power(fCounter,2) + TMath::Power(factor,2)*fErrorSquared + 2* correlation * factor * fCounter * TMath::Sqrt(uncertSq*fErrorSquared); //do not use the "easy to remember" version here, you might divide by zero!
  fCounter *= factor;
  fRawCounter = -1;
}

//__________________________________________________________________________________|___________

void TQCounter::multiply(TQCounter * counter, double correlation, bool includeScaleUncertainty) {
  // multiply the contents of this counter with the contents of another one including correlation for the error propagation if given.
  if (counter) {
    this->multiply(counter->getCounter(), counter->getError(), correlation, includeScaleUncertainty);
    fWarning |= counter->getWarning();
  }
}

//__________________________________________________________________________________|___________

void TQCounter::divide(double denominator, double uncertainty, double correlation, bool includeScaleUncertainty) {
  // divide the contents of this counter by the contents of another one
  double uncertSq = includeScaleUncertainty ? TMath::Power(uncertainty,2) : 0.;
 
  fErrorSquared = TMath::Power(denominator,-2)*fErrorSquared + TMath::Power(fCounter/TMath::Power(denominator,2),2)*uncertSq - 2 * correlation * fCounter * TMath::Power(denominator,3) * TMath::Sqrt(fErrorSquared*uncertSq); //do not use the "easy to remember" version here, might divide by zero! (this version works if denominator != 0)
  fCounter /= denominator;
  fRawCounter = -1;
}

//__________________________________________________________________________________|___________

void TQCounter::divide(TQCounter * counter, double correlation, bool includeScaleUncertainty) {
  // divide the contents of this counter by the contents of another one
  if (counter) {
    this->divide(counter->getCounter(), counter->getError(), correlation, includeScaleUncertainty);
    fWarning |= counter->getWarning();
  }
}

//__________________________________________________________________________________|___________

void TQCounter::scale(double factor, double scaleUncertainty, bool includeScaleUncertainty) {
  // scale this counter with some factor
  double uncertSq = includeScaleUncertainty ? TMath::Power(scaleUncertainty,2) : 0.;
  fErrorSquared = fErrorSquared * TMath::Power(factor,2) + uncertSq * TMath::Power(fCounter,2);
  fCounter *= factor;
}

//__________________________________________________________________________________|___________

void TQCounter::scale(TQCounter * scalefactor, bool includeScaleUncertainty) {
  // scale this counter with some factor
  if (scalefactor) {
    this->scale(scalefactor->getCounter(), scalefactor->getError(), includeScaleUncertainty);
    fWarning |= scalefactor->getWarning();
  }
}
//__________________________________________________________________________________|___________

TString TQCounter::getComparison(TQCounter * cnt1, TQCounter * cnt2,
                                 bool colored, double order) {
  // compare to counters, retrieving an info string as result

  // the comparison string to return 
  TString comparison;

  // the color to apply 
  bool yellow = false;
  bool red = false;

  if (cnt1)
    comparison.Append("(1) ");
  else
    comparison.Append("( ) ");

  if (cnt1 && cnt2) {
    if (cnt1->isEqualTo(cnt2, order)) {
      comparison.Append("=");
    } else if (cnt1->isRawEqualTo(cnt2)) {
      comparison.Append("~");
      yellow = true;
    } else if (cnt1->getCounter() > cnt2->getCounter()) {
      comparison.Append(">");
      red = true;
    } else if (cnt1->getCounter() < cnt2->getCounter()) {
      comparison.Append("<");
      red = true;
    } else {
      comparison.Append("?");
      red = true;
    }
  } else {
    comparison.Append(" ");
    red = true;
  }

  if (cnt2)
    comparison.Append(" (2)");
  else
    comparison.Append(" ( )");

  // return the comparison string 
  if (!colored)
    return comparison;
  else if (yellow)
    return TQStringUtils::makeBoldYellow(comparison);
  else if (red)
    return TQStringUtils::makeBoldRed(comparison);
  else 
    return TQStringUtils::makeBoldGreen(comparison);
}


//__________________________________________________________________________________|___________

bool TQCounter::isEqualTo(TQCounter * counter, double order) {
  // check if two counters are equal (up to some precision)

  // stop if raw numbers are not equal 
  if (!isRawEqualTo(counter))
    return false;

  // stop if weighted numbers are not equal ~O(order) 
  if (!TMath::AreEqualRel(getCounter(), counter->getCounter(), order))
    return false;

  // stop if errors are not equal ~O(order) 
  if (!TMath::AreEqualRel(getError(), counter->getError(), order))
    return false;

  // counter seem to be equal 
  return true;
}


//__________________________________________________________________________________|___________

bool TQCounter::isRawEqualTo(TQCounter * counter) {
  // check if the raw counts of two counters are the same

  // stop if counter is invalid 
  if (!counter)
    return false;

  // stop if raw numbers are not equal 
  if (getRawCounter() != counter->getRawCounter())
    return false;

  // raw counter seem to be equal 
  return true;
}


//__________________________________________________________________________________|___________

TQCounter::~TQCounter() {
  // default destructor
}
