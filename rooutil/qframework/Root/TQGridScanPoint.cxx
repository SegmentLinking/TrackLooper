#include "QFramework/TQGridScanPoint.h"
#include "QFramework/TQHistogramUtils.h"
#include <iostream>

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQGridScanPoint
//
// a single point in configuration space
// can be manipulated by any TQSignificanceEvaluator 
// to store all necessary information for later investigation
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQGridScanPoint)

TQGridScanPoint::TQGridScanPoint():
// default constructor for an empty TQGridScanPoint
significance(0)
{}

TQGridScanPoint::TQGridScanPoint(std::vector<TString>* vars, std::vector<double>& coords, std::vector<TString> switchStatus):
  // constructor for an empty TQGridScanPoint
  // setting the variables and coordinates to the supplied vectors
  variables(*vars),
  coordinates(coords),
	switchStatus(switchStatus),
  significance(0),
  id(0)
{}

TQGridScanPoint::TQGridScanPoint(std::vector<TString>& vars, std::vector<double>& coords, std::vector<TString> switchStatus):
  variables(vars),
  coordinates(coords),
	switchStatus(switchStatus),
  significance(0),
  id(0)
{
  // constructor for an empty TQGridScanPoint
  // setting the variables and coordinates to the supplied vectors
}

void TQGridScanPoint::clear(){
  // delete all information on this point
  variables.clear();
  this->significance = 0;
  this->id = 0;
  coordinates.clear();
  switchStatus.clear();
}

TQGridScanPoint::~TQGridScanPoint(){
  // standard destructor
  this->clear();
}

bool TQGridScanPoint::greater(const TQGridScanPoint* first, const TQGridScanPoint* second){
  // compare two TQGridScanPoints
  // returns true if the significance of the first one is greater
  return (first->significance > second->significance);
}
bool TQGridScanPoint::smaller(const TQGridScanPoint* first, const TQGridScanPoint* second){
  // compare two TQGridScanPoints
  // returns true if the significance of the first one is smaller
  return (first->significance < second->significance);
}

bool operator < (const TQGridScanPoint& first, const TQGridScanPoint& second){
  // compares the significances of two TQGridScanPoints
  return (first.significance < second.significance);
}
bool operator > (const TQGridScanPoint& first, const TQGridScanPoint& second){
  // compares the significances of two TQGridScanPoints
  return (first.significance > second.significance);
}
