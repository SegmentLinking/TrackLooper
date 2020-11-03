#include "TAxis.h"
#include "QFramework/TQHistogramUtils.h"
#include "QFramework/TQTHnBaseUtils.h"
#include "QFramework/TQStringUtils.h"
#include "QFramework/TQTaggable.h"
#include "QFramework/TQCounter.h"
#include "QFramework/TQIterator.h"
#include "QFramework/TQUtils.h"
#include "TMath.h"
#include "TList.h"
#include "THn.h"
#include "THnBase.h"
#include "THnSparse.h"
#include "THashList.h"
#include "TROOT.h"

// #define _DEBUG_

#include "QFramework/TQLibrary.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <limits>

#include <stdio.h> 
#include <stdarg.h> 

#include "Math/Math.h"
#include "Math/QuantFuncMathCore.h"
#include "Math/SpecFuncMathCore.h"

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQTHnBaseUtils:
//
// TQTHnBaseUtils is a namespace providing a set of static utility
// methods for multidimensional histograms with its base class
// THnBase.  It is highly derived from the TQHistogramUtils class,
// where more information can be found.
//
////////////////////////////////////////////////////////////////////////////////////////////////

TString TQTHnBaseUtils::getHistogramDefinition(THnBase * histo) {
  // Returns a string representing the definition of the histogram following a syntax
  // compatible with TQTHnBaseUtils::createHistogram(...) and being similar to the
  // constructor of the corresponding histogram class. 
  // e.g. for a 3-dim histogram:
  // "THnSparseF('histo', 'title', 3, {34, 28, 24}, {0., 10., 0.}, {170., 290., 3.14})"
  // An empty string is returned in case an invalid histogram is passed.

  // invalid histogram?
  if (!histo) {
    return "";
  }
 
  // the following line is because the output of histo->IsA()->GetName() delivers
  // "THnSparseT<TArrayF>" for THnSparseF and THnSparseD respectively...
  TString histoType = histo->IsA()->GetName();
  if (!TQStringUtils::compare(histoType, "THnSparseT<TArrayF>")) {
    histoType = "THnSparseF";
  }
  else if (!TQStringUtils::compare(histoType,"THnSparseT<TArrayD>")) {
    histoType = "THnSparseD";
  }
  
  TString def = TString::Format("%s(\"%s\", ", histoType.Data(), histo->GetName());
  
  // will become full title of histogram (including axis titles)
  TString titleDef = "";//histo->GetTitle();

  // will become binning definition
  TString binDef;
 
  // iterating over dimensions of histogram
  int dim = getDimension(histo);

  for (int i = 0; i < dim; ++i) {
    TAxis * axis = NULL;
    axis = histo->GetAxis(i);
    if (!axis) {
      // should never happen
      break;
    }
 
    // embed axis title in histogram title
    TString title = axis->GetTitle();
    if (!title.IsNull()) {
      TQStringUtils::append(titleDef, title, ";");
    }
    // compile binning definition string
    TQStringUtils::append(binDef, TQHistogramUtils::getBinningDefinition(axis), ", ");
  } 
  
  // format binning definition for use with THnBase
  TQTaggable * binDefTags = TQTaggable::parseParameterList(binDef);
  TString nBins = "";
  TString min = "";
  TString max = "";
  for (int i = 0; i < dim; ++i) {
    nBins.Append(binDefTags->getTagStringDefault(TString::Format("%d", 3*i)));
    min.Append(binDefTags->getTagStringDefault(TString::Format("%d", 3*i+1)));
    max.Append(binDefTags->getTagStringDefault(TString::Format("%d", 3*i+2)));
    if (i < dim-1) {
      nBins.Append(", ");
      min.Append(", ");
      max.Append(", ");
    }
  }
  binDef = TString::Format("{%s}, {%s}, {%s}", nBins.Data(), min.Data(), max.Data());
  
  // now combine all parts to one definition string
  def.Append(TString::Format("\"%s\", ", titleDef.Data()));
  def.Append(TString::Format("%d, ", dim));
  def.Append(binDef + ")");
  
  return def;
}

//__________________________________________________________________________________|___________

THnBase * TQTHnBaseUtils::createHistogram(TString definition, bool printErrMsg) {

  // create histogram
  TString errMsg;
  THnBase * histo = createHistogram(definition, errMsg);

  if (!histo && printErrMsg) {
    // print error message
    std::cout << TQStringUtils::makeBoldRed(errMsg.Prepend(
							   "TQTHnBaseUtils::createHistogram(...): ")).Data() << std::endl;
  }

  // return histogram
  return histo;
}

//__________________________________________________________________________________|___________

THnBase * TQTHnBaseUtils::createHistogram(TString definition, TString &errMsg) {
  // Creates a new instance of a THnBase histogram from a definition string
  // that uses a similar syntax as the constructor of the corresponding histogram
  // class. Currently only THnSparseF and THnSparseD is supported
  //
  // Examples:
  //
  // - a THnSparse with 3 dimensions:
  //
  // createTHnBase("THnSparseF('histo', 'title', 3, {34, 28, 24}, {0., 10., 0.}, {170., 290., 3.14})")
  //
  
  // read histogram type (e.g. "THnBaseF", ...)
  TString type;
  TQStringUtils::readBlanksAndNewlines(definition);
  if (TQStringUtils::readToken(definition,type,TQStringUtils::getLetters() + TQStringUtils::getNumerals()) == 0) {
    errMsg = TString::Format("Missing valid histogram type, received '%s' from '%s'",type.Data(),definition.Data());
    return NULL;
  }
  
  // histogram type to create
  bool isTHnSparseF = (type.CompareTo("THnSparseF") == 0);
  bool isTHnSparseD = (type.CompareTo("THnSparseD") == 0);

  if (!isTHnSparseF && !isTHnSparseD) {
    errMsg = TString::Format("Unknown histogram type '%s'", type.Data());
    return NULL;
  }

  // read parameter block
  TString parameter;
  TQStringUtils::readBlanksAndNewlines(definition);
  if (TQStringUtils::readBlock(definition, parameter, "()", "''\"\"", false) == 0) {
    errMsg = TString::Format("Missing parameter block '(...)' after '%s'", type.Data());
    return NULL;
  }

  // make sure there is nothing left after the parameter block
  TQStringUtils::readBlanksAndNewlines(definition);
  if (!definition.IsNull()) {
    errMsg = TString::Format("Unrecognized token '%s'", definition.Data());
    return NULL;
  }

  // parse parameter block
  TQTaggable * pars = TQTaggable::parseParameterList(parameter, ",", true, "{}[]()", "''\"\"");
  if (!pars) {
    errMsg = TString::Format("Failed to parse parameters '%s'", parameter.Data());
    return NULL;
  }

  // keep track of parameters read (to find unexpected parameters)
  pars->resetReadFlags();

  // name of histogram
  if (!pars->tagIsOfTypeString("0")) {
    errMsg = "Missing valid histogram name";
    delete pars;
    return NULL;
  }
  TString name = pars->getTagStringDefault("0");

  // title of histogram
  if (!pars->tagIsOfTypeString("1")) {
    errMsg = "Missing valid histogram title";
    delete pars;
    return NULL;
  }
  TString title = pars->getTagStringDefault("1");

  // the number of bins on axes
  int dimension = 0;  
  std::vector<int> nBins;
  
  // left and right bound on axes
  std::vector<double> min; 
  std::vector<double> max; 

  // for THnSparse
  if (!extractBinning(pars, dimension, nBins, min, max, errMsg)) {
    errMsg.Append(" when creating THnSparse");
    delete pars;
    return NULL;
  }

  // unread parameters left?
  if (pars->hasUnreadKeys()) {
    errMsg.Append(TString::Format("Too many parameters for '%s'", type.Data()));
    delete pars;
    return NULL;
  }

  // the current directory might contain a histogram with the same name
  int i = 2;
  TString finalName = name;
  while (gDirectory && gDirectory->FindObject(name.Data())) {
    name = TString::Format("%s_i%d", finalName.Data(), i++);
  }

  // now create the histogram calling the corresponding constructor
  THnBase * histo = NULL;
  if (isTHnSparseF) {
    histo = new THnSparseF(name.Data(), title.Data(), dimension, 
			   nBins.data(), min.data(), max.data());
  }
  else if (isTHnSparseD) {
    histo = new THnSparseD(name.Data(), title.Data(), dimension,
			   nBins.data(), min.data(), max.data());
  }

  if (histo) {
    // histo->SetDirectory(NULL); not callable with THnBase, not necessary?
    // set original (final) name
    histo->SetName(finalName.Data());
    // sum weights
    histo->Sumw2();
  }

  delete pars;
  
  // finally return the histogram
  if(!histo){
    errMsg = "unknown error: histogram type is '"+type+"'";
  }
  
  return histo;
}
 
//__________________________________________________________________________________|___________
 
bool TQTHnBaseUtils::extractBinning(TQTaggable * p, int &dimension, std::vector<int> &nBins,std::vector<double> &min,std::vector<double> &max, TString &errMsg) {
	// extract the binning definition of the THnBase object. This must
	// exactly fit the syntax of the histogram definition file.
	
	p->getTagInteger("2", dimension); // get dimension of THnSparse
  
  // loop over THnSparse definitions of nBins, max values and min values
  for (int p_bin = 3; p_bin<6; p_bin++) {  
    double edge = 0.;
    // loop through arrays of nBins, max values and min values
    for (int i=0; i<dimension; i++) {
      if (!p->hasTag(TString::Format("%d.%d", p_bin, i))) {
				errMsg = "Invalid array of bins";
      }
      if (!p->getTagDouble(TString::Format("%d.%d", p_bin, i), edge)) {
				errMsg = "Invalid array of bins";
				return false;
      }
      if (p_bin == 3)
				nBins.push_back(edge);
      if (p_bin == 4)
				min.push_back(edge);
      if (p_bin == 5)
				max.push_back(edge);
    }
  }
  return true;
}

//__________________________________________________________________________________|___________

THnBase * TQTHnBaseUtils::copyHistogram(THnBase * histo, const TString& newName) {
  // Creates an independent copy of the input histogram <histo> which is a THnBase
  // and returns a pointer to the copy or a NULL pointer in case of failure. 

  // stop if the histogram to copy is invalid
  if (!histo) {
    return NULL;
  }
  // make a copy of the input histogram
  THnBase * copy = TQTHnBaseUtils::createHistogram(TQTHnBaseUtils::getHistogramDefinition(histo));
  if (!copyAxisNames(histo, copy)) {
    WARN("failed copying axis names");
  }

  if (!copy) {
    return NULL;
  }
  
  TQTHnBaseUtils::addHistogram(copy,histo);
  
  int dim = getDimension(histo);

  for (int i = 0; i < dim; ++i) {
    if (histo->GetAxis(i)) copy->GetAxis(i)->SetTitle(histo->GetAxis(i)->GetTitle());
  }

  // Remark: The following commented lines are from the TQHistogramUtils package.
	// THnBase does not support SetDirectory/GetDirectory which leads to the assumption
	// that the next lines of code are not needed, however, this is not fully validated.
	
  // set the directory of the original histogram
  // TDirectory * histoDir = histo->GetDirectory();
  // if (!histoDir || newName == "NODIR") {
  //   copy->SetDirectory(NULL);
  //   copy->SetName(histo->GetName());
  // } else {
  //   if (newName.IsNull()) {
  //     // find a unique name
  //     TString prefix = histo->GetName();
  //     TString name = prefix;
  //     int i = 2;
  //     while (histoDir->FindObject(name.Data())) {
  //       name = TString::Format("%s_i%d", prefix.Data(), i++);
  //     }
  //     copy->SetName(name.Data());
  //   } else {
  //     copy->SetName(newName);
  //   }
  //   copy->SetDirectory(histoDir);
  // }
  
  // return the copied histogram
  return copy;
}

//__________________________________________________________________________________|___________

bool TQTHnBaseUtils::copyAxisNames(THnBase * histo, THnBase * copy) {
  // simply copy the axis names
  if (!histo || !copy) {
    return false;
  }
  for (int i=0; i<histo->GetNdimensions(); i++) {
    copy->GetAxis(i)->SetName(histo->GetAxis(i)->GetName());
  }
  return true;
}

//__________________________________________________________________________________|___________

bool TQTHnBaseUtils::addHistogram(THnBase * histo1, THnBase * histo2,
				  TQCounter* scale, double corr12, bool includeScaleUncertainty) {
  // adds two THnBase histograms with built-in Root function THnBase::Add()
  // Remark: This means there is no support for scale uncertainties!
  
  if(scale)
    return TQTHnBaseUtils::addHistogram(histo1,histo2,scale->getCounter(),scale->getError(),corr12,includeScaleUncertainty);
  else
    return TQTHnBaseUtils::addHistogram(histo1,histo2,1.,0.,corr12,false);
}


//__________________________________________________________________________________|___________


bool TQTHnBaseUtils::addHistogram(THnBase * histo1, THnBase * histo2,
				  double scale, double scaleUncertainty,
				  double corr12, bool includeScaleUncertainty) {
  // adds two THnBase histograms with built-in Root function THnBase::Add()
  // Remark: This means there is no support for scale uncertainties!

	// check validity of input histograms
  if (!histo1 || !histo2 || !checkConsistency(histo1, histo2)) {
    return false;
  }
  
  histo1->Add(histo2, scale);
  
  return true;
}


//__________________________________________________________________________________|___________


bool TQTHnBaseUtils::scaleHistogram(THnBase * histo, TQCounter* scale, bool includeScaleUncertainty) {
  // use built in Root function for scaling
  // Remark: This means there is no support for scale uncertainties!  
  if(scale)
    return TQTHnBaseUtils::scaleHistogram(histo,scale->getCounter(),scale->getError(),includeScaleUncertainty);
  else
    return TQTHnBaseUtils::scaleHistogram(histo,1.,0.,false);
}


//__________________________________________________________________________________|___________

bool TQTHnBaseUtils::scaleHistogram(THnBase * histo, double scale, double scaleUncertainty,
				    bool includeScaleUncertainty) {
  // use built in Root function for scaling
  // Remark: This means there is no support for scale uncertainties!  
  
  histo->Scale(scale);

  return true;
}
  
//__________________________________________________________________________________|___________

int TQTHnBaseUtils::getDimension(THnBase * histo) {
	
  if (!histo) {
    // is an invalid pointer
    return 0;
  }
  
  return histo->GetNdimensions();
}

//__________________________________________________________________________________|___________

bool TQTHnBaseUtils::checkConsistency(THnBase * histo1, THnBase * histo2, bool verbose) {
  // Check if two histograms have consistent binning
  // Remark: This is code from built-in Root function THnBase::CheckConsistency()
  
  // stop unless both histograms are valid
  if (!histo1 || !histo2) {
    if(verbose) ERRORfunc("received NULL pointer");
    return false;
  }
  
  if (histo1->GetNdimensions() != histo2->GetNdimensions()) {
    if(verbose) ERRORfunc("different number of dimensions, cannot carry out operation on the histograms");
    return false;
  }
  
  DEBUGfunc("function called on histograms '%s' and '%s'",histo1->GetName(),histo2->GetName());
  
  for (int dim = 0; dim < histo1->GetNdimensions(); dim++) {
    if (histo1->GetAxis(dim)->GetNbins() != histo2->GetAxis(dim)->GetNbins()) {
      if(verbose) ERRORfunc("Different number of bins on axis %i, cannot carry out operation on the histograms", dim);
      return false;
    }
  }
  
  return true;
}

