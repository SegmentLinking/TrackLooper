//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQTHnBaseUtils__
#define __TQTHnBaseUtils__

#include "QFramework/TQTaggable.h"
#include "QFramework/TQSampleFolder.h"
#include "TObject.h"
#include "THnBase.h"
#include "TAxis.h"
#include "QFramework/TQCounter.h"

#include <vector>
#include <limits>

namespace TQTHnBaseUtils {

  bool extractBinning(TQTaggable * p, int &dimension, std::vector<int> &nBins,
											std::vector<double> &min,	std::vector<double> &max,
											TString &errMsg);
	
  TString getHistogramDefinition(THnBase * histo);
 
  int getDimension(THnBase * histo);

  THnBase * createHistogram(TString definition, bool printErrMsg = false);
  THnBase * createHistogram(TString definition, TString &errMsg);

  THnBase * copyHistogram(THnBase * histo, const TString& newName = "");

  bool copyAxisNames(THnBase * histo, THnBase * copy);
 
  bool addHistogram(THnBase * histo1, THnBase * histo2, double scale = 1.,
										double scaleUncertainty = 0., double corr12 = 0.,
										bool includeScaleUncertainty=true);
  bool addHistogram(THnBase * histo1, THnBase * histo2, TQCounter* scale,
										double corr12 = 0., bool includeScaleUncertainty=false);
	
  bool scaleHistogram(THnBase * histo1, double scale = 1., double scaleUncertainty = 0.,
											bool includeScaleUncertainty = true);
  bool scaleHistogram(THnBase * histo1, TQCounter* scale, bool includeScaleUncertainty = false);

  bool checkConsistency(THnBase * histo1, THnBase * histo2, bool verbose=false);


}


#endif
