#include "TMVA/Timer.h"
#include "QFramework/TQGridScanner.h"
#include "QFramework/TQGridScanPoint.h"
#include "TLine.h"
#include <algorithm>
#include <iostream>
#include "TCanvas.h"
#include "QFramework/TQSignificanceEvaluator.h"
#include "QFramework/TQHistogramUtils.h"
#include "QFramework/TQStringUtils.h"
#include "QFramework/TQFolder.h"
#include "QFramework/TQUtils.h"
#include "QFramework/TQIterator.h"
#include "QFramework/TQPlotter.h"
#include "TLatex.h"
#include "TPaletteAxis.h"
#include "TLegend.h"
#include "TSystem.h"
#include "TStyle.h"
#include "TROOT.h"
#include "THnBase.h"
#include "QFramework/TQLibrary.h"
#include <math.h>
#include <limits>


////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQGridScanner
//
// The TQGridScanner is the last element in the chain of classes 
// that can be used for a full scan cut optimization.
// 
// the workflow here could be as follows
// - create a multidimensional histogram during runAnalysis with the desired variables
//   for optimization
// - run the runSignificanceScan.py script for creating a TQGridScanner
//   and supplying it with a TQSignificanceEvaluator
//   that fits your needs regarding precision/runtime requirements
// - sort the results using the TQGridScanner::sortPoints method
// - retrieve the first (best) entry with TQGridScanner::printPoint(0)
//
// Plotting options:
// - ext=pdf,ps,... 
// file extension. Works for all "plotAndSave..."-functions
// accepts all extension that are accepted by TCanvas->SaveAs(...)
// - showmax=true,false
// show the maximum. works for plotAndSave[All]SignficanceProfile[s]
// will additionally show the absolute maximum for each bin
// this option is ignored if topNumber is 1
// - cut.$VARNAME=X. Works if set on the GridScanner itself. 
// Will draw a vertical line at X. Can be used to mark the baseline cut value.
//
////////////////////////////////////////////////////////////////////////////////////////////////

#define PRECISION 0.000000001 // for double/float comparison 

ClassImp(TQGridScanner)

TQGridScanner::TQGridScanner(const TString& name):
TNamed(name,name),
  nPointsProcessed(0),
  targetVarName(""),
  targetVarTitle(""),
  evaluator(NULL),
  verbose(false),
  sorted(false),
  nPointsTotal(0),
  formatString("%d.%d"),
  splitConfigInfoStr("")
{
  // construct a TQGridScanner with the given name (default "gridscan")
  // the name should match the name of the TQGridScanAnalysisJob
  // that you used to create the grids
}

TQGridScanner::TQGridScanner():
  TNamed("gridscan","gridscan"),
  nPointsProcessed(0),
  targetVarName(""),
  targetVarTitle(""),
  evaluator(NULL),
  verbose(false),
  sorted(false),
  nPointsTotal(0),
  formatString("%d.%d"),
  splitConfigInfoStr("")
{
  // construct a TQGridScanner without a name
}

TQGridScanner::~TQGridScanner(){
  // delete a TQGridScanner and remove all data stored therein
  // this will invalidate all pointers to points created/visited
  // and grids read from the folder structure
  // but not to histograms produced thereof
  this->variables.clear();
  this->splitVars.clear();
  this->currentVals.clear();
  this->currentBins.clear();
  this->boundTypes.clear();
  this->points.clear();
  this->clearRequirements();
  delete this->runTimer;
}

void TQGridScanner::addRequirement(const TString& varname, const TString& type, double val){
  // add a requirement that will restrict variable ranges for printing and plotting
  // has no effect whatsoever on the scan itself
  this->requirements.push_back(varname);
  this->requirements_types.push_back(type);
  this->requirements_values.push_back(val);
}

void TQGridScanner::clearRequirements(){
  // clear requirements set via
  // TQGridScanner::addRequirement(...)
  this->requirements_types.clear();
  this->requirements.clear();
  this->requirements_values.clear();
} 

void TQGridScanner::addVariableUpper(const TString& varname){
  // add a variable to the scanner with an upper cut
  // internally, TQGridScanner::addVariable(varname,UPPER) is called
  this->addVariable(varname, UPPER);
}

void TQGridScanner::addVariableLower(const TString& varname){
  // add a variable to the scanner with a lower cut
  // internally, TQGridScanner::addVariable(varname,LOWER) is called
  this->addVariable(varname, LOWER);
}

void TQGridScanner::addVariableSplit(const TString& varname){
  // add a variable to the scanner with a split-cut
  // internally, TQGridScanner::addVariable(varname,SPLIT) is called
  this->addVariable(varname, SPLIT);
}

void TQGridScanner::addVariableUpperSwitch(const TString& varname){
  // internally, TQGridScanner::addVariable(varname,UPPERSWITCH) is called
  this->addVariable(varname, UPPERSWITCH);
}

void TQGridScanner::addVariableLowerSwitch(const TString& varname){
  // internally, TQGridScanner::addVariable(varname,LOWERSWITCH) is called
  this->addVariable(varname, LOWERSWITCH);
}

void TQGridScanner::addVariableUpperFixed(const TString& varname, double value){
  // internally, TQGridScanner::addVariable(varname,UPPERFIXED) is called
  this->addVariable(varname, UPPERFIXED, value);
  this->setFixedBounds(varname, value);
}

void TQGridScanner::addVariableLowerFixed(const TString& varname, double value){
  // internally, TQGridScanner::addVariable(varname,LOWERFIXED) is called
  this->addVariable(varname, LOWERFIXED, value);
  this->setFixedBounds(varname, value);
}

void TQGridScanner::addVariableWindowFixed(const TString& varname, double low, double up){
  // internally, TQGridScanner::addVariable(varname,WINDOWFIXED) is called
  this->addVariable(varname, WINDOWFIXED);
  this->setFixedLowerBound(varname, low);
  this->setFixedUpperBound(varname, up);
}

void TQGridScanner::addVariable(const TString& varname, BoundType type, double value){
  // each variable will be scanned over the full range unless not particular specified
	// in the config options.
  THnBase* g = this->ndimHists[0];
  if(!g) {
    ERRORclass("pointer on multidimensional histogram is NULL!");
  }
  TString vname = TQStringUtils::trim(varname);
  this->variables.push_back(vname);
  this->boundTypes.push_back(type);
  int idx = this->getAxisIndex(g, vname);
  this->varIdx.push_back(idx);  // get the right axis out of the multi-dim histogram
  this->varTitle.push_back(g->GetAxis(idx)->GetTitle());
  this->axisMin.push_back(g->GetAxis(idx)->GetXmin());
  this->axisMax.push_back(g->GetAxis(idx)->GetXmax());
  this->nBins.push_back(g->GetAxis(idx)->GetNbins());
  this->stepsize.push_back( (this->axisMax.back()-this->axisMin.back()) / this->nBins.back());
  // prepare some vectors for the actual scan later on
  if (type==UPPERFIXED || type==LOWERFIXED) {
    this->scanLowerBounds.push_back(value);
    this->scanUpperBounds.push_back(value);
  }	else {
    this->scanLowerBounds.push_back(axisMin.back());
    this->scanUpperBounds.push_back(axisMax.back());
  }
  this->scanLowerBins.push_back(0);
  this->scanUpperBins.push_back(nBins.back());
  this->currentVals.push_back(0);
  this->currentBins.push_back(0);
  if (type==SPLIT) {
    this->splitVars.push_back(variables.size()-1);
  }
  if (type==UPPERSWITCH || type==LOWERSWITCH) {
    this->switchVars.push_back(1);
    this->currentSwitchStatus.push_back("on");
  } else {
    this->switchVars.push_back(0);
    this->currentSwitchStatus.push_back("");
  }
  
  // this->resetBounds();
}

void TQGridScanner::reconfigureVariables() {
  // this function is called from the significance evaluator. It ensures that the appropriate
  // axes are evaluated after reducing the size of the multidimensional histogram
  // (see TQSimpleSignificanceEvaluator::updateHists())

  THnBase* g = this->ndimHists[0];
  if(!g) {
    ERRORclass("pointer on multidimensional histogram is NULL!");
  }
  std::vector<TString> new_variables;
  std::vector<BoundType> new_boundTypes;
  std::vector<int> new_varIdx;
  std::vector<TString> new_varTitle;
  std::vector<double> new_axisMin;
  std::vector<double> new_axisMax;
  std::vector<size_t> new_nBins;
  std::vector<double> new_stepsize;
  std::vector<double> new_scanLowerBounds;
  std::vector<double> new_scanUpperBounds;
  std::vector<int> new_scanLowerBins;
  std::vector<int> new_scanUpperBins;
  std::vector<double> new_currentVals; 
  std::vector<size_t> new_currentBins;
  std::vector<size_t> new_splitVars;
  std::vector<bool> new_switchVars; 
  std::vector<TString> new_currentSwitchStatus;

  for (unsigned int i=0; i<this->axesToScan.size(); i++) {
    new_variables.push_back(this->variables[i]);
    new_boundTypes.push_back(this->boundTypes[i]);
    int idx = this->getAxisIndex(g, this->variables[i]);
    new_varIdx.push_back(idx);  // get the right axis out of the multi-dim histogram
    new_varTitle.push_back(g->GetAxis(idx)->GetTitle());
    new_axisMin.push_back(g->GetAxis(idx)->GetXmin());
    new_axisMax.push_back(g->GetAxis(idx)->GetXmax());
    new_nBins.push_back(g->GetAxis(idx)->GetNbins());
    new_stepsize.push_back( (new_axisMax.back()-new_axisMin.back()) / new_nBins.back());
    // prepare some vectors for the actual scan later on
    new_scanLowerBounds.push_back(new_axisMin.back());
    new_scanUpperBounds.push_back(new_axisMax.back());
    new_scanLowerBins.push_back(0);
    new_scanUpperBins.push_back(new_nBins.back());
    new_currentVals.push_back(0);
    new_currentBins.push_back(0);
    if (this->boundTypes[i]==SPLIT) {
      new_splitVars.push_back(new_variables.size()-1);
    }
    if (this->boundTypes[i]==UPPERSWITCH || this->boundTypes[i]==LOWERSWITCH) {
      new_switchVars.push_back(1);
      new_currentSwitchStatus.push_back("on");
    } else {
      new_switchVars.push_back(0);
      new_currentSwitchStatus.push_back("");
    }
    
  }
  this->variables = new_variables;
  this->boundTypes = new_boundTypes;
  this->varIdx = new_varIdx;
  this->varTitle = new_varTitle;
  this->axisMin = new_axisMin;
  this->axisMax = new_axisMax;
  this->nBins = new_nBins;
  this->stepsize = new_stepsize;
  this->scanLowerBounds = new_scanLowerBounds;
  this->scanUpperBounds = new_scanUpperBounds;
  this->scanLowerBins = new_scanLowerBins;
  this->scanUpperBins = new_scanUpperBins;
  this->currentVals = new_currentVals; 
  this->currentBins = new_currentBins; 
  this->splitVars = new_splitVars; 
  this->switchVars = new_switchVars; 
  this->currentSwitchStatus = new_currentSwitchStatus;
}


void TQGridScanner::setCorrectBinBound(TAxis* axis, double& bound, TString boundtype) {
	// check if bound is a bin boundary, correct it otherwise to the nearest bin boundary
	// print out the corrected boundary
	
	// Bin with nr. 0 is underflow, Bin with nr. nbins+1 is overflow
	std::vector<double> binbounds;
	double max = axis->GetXmax();
	double min = axis->GetXmin();
	int nbins = axis->GetNbins();
	double stepsize = (max - min) / nbins;

	double lowerbound = 0; 
	double upperbound = 0;
	for (int i=0; i<=nbins; i++) {
		double binbound = min+stepsize*i;
		if (std::fabs(bound-binbound) < PRECISION) {
			// bound is a valid bin bound of histogram
			return;
		}
		else if (bound > binbound) {
			lowerbound = binbound;
		}
		else if (bound < binbound) {
			upperbound = binbound;
			break;
		}
		binbounds.push_back(binbound);
	}
	// bound doesn't fit bin boundary, choose the closest binboundary
	double newbound = 0;
	if ((bound-lowerbound) < (upperbound-bound)) newbound = lowerbound;
	else newbound = upperbound;
	WARN(TString::Format("specified %s scan range for variable %s doesn't fit a bin boundary. Correct the value from %f to %f!", boundtype.Data(), axis->GetName(), bound, newbound));
	bound = newbound;
	return;
}

void TQGridScanner::setFixedLowerBound(const TString& varname, double bound) {
	int index = this->getVariablePosition(varname);
  TAxis* axis = this->ndimHists[0]->GetAxis(this->varIdx[index]);
  this->setCorrectBinBound(axis, bound, "lower");
  int binbound = axis->FindBin(bound+2*std::numeric_limits<double>::epsilon());
  for(size_t i=0; i<boundedVariables.size(); i++){
    if(boundedVariables[i] == varname){
      this->lowerBoundedBins[i] = binbound; 
      this->lowerBounds[i] = bound;
      return;
    }
  }
  this->boundedVariables.push_back(varname);
  this->lowerBounds.push_back(bound);
  this->lowerBoundedBins.push_back(binbound);
	this->upperBounds.push_back(axis->GetXmax());
	this->upperBoundedBins.push_back(axis->GetNbins());
}

void TQGridScanner::setFixedUpperBound(const TString& varname, double bound) {
	int index = this->getVariablePosition(varname);
  TAxis* axis = this->ndimHists[0]->GetAxis(this->varIdx[index]);
  this->setCorrectBinBound(axis, bound, "upper");
  int binbound = axis->FindBin(bound+2*std::numeric_limits<double>::epsilon());
  for(size_t i=0; i<boundedVariables.size(); i++){
    if(boundedVariables[i] == varname){
      this->upperBoundedBins[i] = binbound; 
      this->upperBounds[i] = bound;
      return;
    }
  }
  this->boundedVariables.push_back(varname);
	this->upperBounds.push_back(bound);
	this->upperBoundedBins.push_back(bound);
  this->lowerBounds.push_back(axis->GetXmin());
  this->lowerBoundedBins.push_back(0);
}

void TQGridScanner::setFixedBounds(const TString& varname, double bound) {
	int index = this->getVariablePosition(varname);
  TAxis* axis = this->ndimHists[0]->GetAxis(this->varIdx[index]);
  this->setCorrectBinBound(axis, bound, "fixed");
  int binbound = axis->FindBin(bound+2*std::numeric_limits<double>::epsilon());
  this->boundedVariables.push_back(varname);
  this->lowerBounds.push_back(bound);
  this->lowerBoundedBins.push_back(binbound);
	this->upperBounds.push_back(bound);
	this->upperBoundedBins.push_back(binbound);
}

void TQGridScanner::setLowerScanRange(const TString& varname, double bound){
  // save the lower boundary for scanning, which was specified in the config.
  // it is important that this function is only executed once per variable!
	int index = this->getVariablePosition(varname);
	if (this->boundTypes[index] == UPPERFIXED || this->boundTypes[index] == LOWERFIXED ||
			this->boundTypes[index] == WINDOWFIXED) {
		return;
	}
  TAxis* axis = this->ndimHists[0]->GetAxis(this->varIdx[index]);
  this->setCorrectBinBound(axis, bound, "lower");
  int binbound = axis->FindBin(bound+2*std::numeric_limits<double>::epsilon());
  // + epsilon() because otherwise the finite numeric precisision can cause getting the wrong bin here.
  for(size_t i=0; i<boundedVariables.size(); i++){
    if(boundedVariables[i] == varname){
      this->lowerBoundedBins[i] = binbound; 
      this->lowerBounds[i] = bound;
      return;
    }
  }
  this->boundedVariables.push_back(varname);
  this->lowerBounds.push_back(bound);
  this->lowerBoundedBins.push_back(binbound);
	this->upperBounds.push_back(axis->GetXmax());
	this->upperBoundedBins.push_back(axis->GetNbins());
}

void TQGridScanner::setUpperScanRange(const TString& varname, double bound) {
  // save the upper boundary, which was specified in the config, of the given variable
  // if the variable matches one of the scheduled bounded variables, 
  // the scan range will be restricted as well
  // it is important that this function is only executed once per variable!
	int index = this->getVariablePosition(varname);
	if (this->boundTypes[index] == UPPERFIXED || this->boundTypes[index] == LOWERFIXED || 
			this->boundTypes[index] == WINDOWFIXED) {
		return;
	}
  TAxis* axis = this->ndimHists[0]->GetAxis(this->varIdx[index]);
  this->setCorrectBinBound(axis, bound, "upper");
  int binbound = axis->FindBin(bound+2*std::numeric_limits<double>::epsilon());
  // + epsilon() because otherwise the finite numeric precisision can cause getting the wrong bin here.
  for(size_t i=0; i<boundedVariables.size(); i++){
    if(boundedVariables[i] == varname){
      this->upperBoundedBins[i] = binbound; 
      this->upperBounds[i] = bound;
      return;
    }
  }
  this->boundedVariables.push_back(varname);
  this->upperBounds.push_back(bound);
  this->upperBoundedBins.push_back(binbound);
	this->lowerBounds.push_back(axis->GetXmin());
	this->lowerBoundedBins.push_back(0);
}

void TQGridScanner::printVariableConfiguration(){
  // print the internal variable definitions
  this->sortVariables();
  if (this->variables.size() == 0) {
    std::cout << " < no variables scheduled > " << std::endl;
  }
  else {
    for(size_t i=0; i<this->variables.size(); i++){
      size_t index = this->variableOrdering[i];
      std::cout << this->variables[index] << ": ";
      if(boundTypes[index] == UPPER) std::cout << " upper cut, ";
      if(boundTypes[index] == LOWER) std::cout << " lower cut, ";
      if(boundTypes[index] == SPLIT) std::cout << " split, ";
      if(boundTypes[index] == UPPERSWITCH) std::cout << " upper switch, ";
      if(boundTypes[index] == LOWERSWITCH) std::cout << " lower switch, ";
      if(boundTypes[index] == UPPERFIXED || boundTypes[index] == LOWERFIXED) { 
	std::cout << "fixed value of " << this->variables[index] << " ";
	std::cout << this->getBoundOperator(index) << " ";
	std::cout << this->scanLowerBounds[index] << std::endl;
      }
      else if (boundTypes[index] == UPPERSWITCH || boundTypes[index] == LOWERSWITCH) {
	std::cout << "switch cut " << this->variables[index] << this->getBoundOperator(index);
	std::cout << this->scanLowerBounds[index] << " on/off" << std::endl;
      }
      else if (boundTypes[index] == WINDOWFIXED) {
	std::cout << "applying fixed window cut of " << this->scanLowerBounds[index];
	std::cout << " < " << this->variables[index] << " < ";
	std::cout << this->scanUpperBounds[index] << std::endl;
      }
      else {
	std::cout << " scanning from " << this->scanLowerBounds[index] << " to ";
	std::cout << this->scanUpperBounds[index] << " in ";
	std::cout << this->scanUpperBins[index]-this->scanLowerBins[index];
	std::cout << " steps of " << this->stepsize[index] << " each.";
	std::cout << std::endl;
      }
    }
  }
}

size_t TQGridScanner::getMaxBin(size_t varno){
  // retrieve the maximum bin for the given variable which is set for scanning 
	// through all cut values
  return this->scanUpperBins[varno];
}

size_t TQGridScanner::getMinBin(size_t varno){
  // retrieve the minimum bin for the given variable which is set for scanning
	// through all cut values
  return this->scanLowerBins[varno];
}

double TQGridScanner::getBinMax(size_t varno, size_t bin){
  // return the upper boundary of the bin of the given number
  // for variable with index varno
  if(bin > this->nBins[varno]) {
    return std::numeric_limits<double>::infinity();
  }
  return this->ndimHists[0]->GetAxis(this->varIdx[varno])->GetBinUpEdge(bin);
}

double TQGridScanner::getBinMin(size_t varno, size_t bin){
  // return the lower boundary of the bin of the given number
  // for variable with index varno
  if(bin == 0) {
    return -std::numeric_limits<double>::infinity();
  }
  return this->ndimHists[0]->GetAxis(this->varIdx[varno])->GetBinLowEdge(bin);
}

void TQGridScanner::setUpperRange(size_t varno){
  // set temporary the maximum bin for this axis only
	// Remark: When using SetRangeUser the following behaviour holds
	// For the histogram h = TH1F("h", "h", 10, 0, 10) with bins 2 and 3 Filled with Content 1.
	// h.SetRangeUser(2., 3.); h.Integral()
	// will give the output 1.
  bool doCut = this->switchVars[varno];
  for (size_t i=0; i<ndimHists.size(); i++) {
    if (boundTypes[varno] == UPPERSWITCH) {
      // SetRangeUser at border with +1/-1 for inclusion of overflow/underflow bin
      if (doCut) {
	// make cut!
	this->ndimHists[i]->GetAxis(this->varIdx[varno])->SetRangeUser(this->axisMin[varno]-1, this->currentVals[varno]);
	this->switchVars[varno] = 0;
      }
      else {
	this->ndimHists[i]->GetAxis(this->varIdx[varno])->SetRangeUser(this->axisMin[varno]-1, this->axisMax[varno]+1);
	this->switchVars[varno] = 1;
      }
    }
    else {
      this->ndimHists[i]->GetAxis(this->varIdx[varno])->SetRangeUser(this->axisMin[varno]-1, this->currentVals[varno]);
    }
  }
}

void TQGridScanner::setLowerRange(size_t varno){
  // set the temporary minimum bin for this axis only
  bool doCut = this->switchVars[varno];
  for (size_t i=0; i<ndimHists.size(); i++) {
    if (boundTypes[varno] == LOWERSWITCH) {
      // SetRangeUser at border with +1/-1 for inclusion of overflow/underflow bin
      if (doCut) {
	// make cut!
	this->ndimHists[i]->GetAxis(varIdx[varno])->SetRangeUser(this->currentVals[varno], this->axisMax[varno]+1);
	this->switchVars[varno] = 0; // switch off
      }
      else {
	// no cut!
	this->ndimHists[i]->GetAxis(varIdx[varno])->SetRangeUser(this->axisMin[varno]-1, this->axisMax[varno]+1);
	this->switchVars[varno] = 1;
      }
    }
    else {
      this->ndimHists[i]->GetAxis(varIdx[varno])->SetRangeUser(this->currentVals[varno], this->axisMax[varno]+1);
    }
  }
}

void TQGridScanner::setWindowRange(size_t varno){
  // set the window for windowfixed variable
  for (size_t i=0; i<ndimHists.size(); i++) {
		this->ndimHists[i]->GetAxis(varIdx[varno])->SetRangeUser(this->scanLowerBounds[varno], this->scanUpperBounds[varno]);
  }
}

void TQGridScanner::setVariableBoundType(const TString& varname, BoundType type){
  // set/change the variable title of the given variable
  size_t i=this->findVariable(varname);
  if(i > this->variables.size()) return;
  while(this->boundTypes.size() <= i){
    this->boundTypes.push_back(UNDEFINED);
  }
  if(boundTypes.size() == i)
    this->boundTypes.push_back(type);
  else 
    this->boundTypes[i] = type;
}

TString TQGridScanner::getVariableTitle(const TString& varname, bool includeBound){
  // set/change the variable title of the given variable
  return this->getVariableTitle(this->findVariable(varname),includeBound);
}

TString TQGridScanner::getVariableTitle(size_t index, bool includeBound){
  // set the variable title of the given variable
	TString title = this->varTitle[index];
  if(!includeBound) return title;
  int removed = TQStringUtils::removeTrailing(title,"}");
  if(removed > 0) title += ",";
  else title += "_{";
  title += this->boundTypeToVarString(this->boundTypes[index]);
  return title+TQStringUtils::repeat("}",std::max(removed,1));
}

TString TQGridScanner::boundTypeToVarString(BoundType t){
  if(t==UPPER) return "max";
  if(t==LOWER) return "min";
  if(t==SPLIT) return "split";
  if(t==UPPERFIXED) return "maxfixed";
  if(t==LOWERFIXED) return "minfixed";
  if(t==UPPERSWITCH) return "maxswitch";
  if(t==LOWERSWITCH) return "minswitch";
  return "?";
}

TString TQGridScanner::boundTypeToCutString(BoundType t){
  if(t==UPPER) return "upper cut";
  if(t==LOWER) return "lower cut";
  if(t==SPLIT) return "split";
  return "(unknown)";
}
 
bool TQGridScanner::setEvaluator(TQSignificanceEvaluator* e){
  // set the significance evaluation method 
  // by adding a TQSignificianceEvaluator to the GridScanner
  this->evaluator = e;
  return e->initialize(this);
}

bool TQGridScanner::updateEvaluatorHists(TString axisToEvaluate){
  // Find all variables which should be scanned and project them out into a new
  // multidimensional histogram. This will lead to a better performance during scanning
  INFOclass("applying fixed cuts on variables:");
  for (unsigned int i=0; i<variables.size(); i++) {
    if (boundTypes[i] == LOWERFIXED) {
      double low = this->lowerBounds[this->getBoundedVariablePosition(variables[i])];
      this->evaluator->setRangeAxis(this->varIdx[i], low, this->axisMax[i]);
      INFO(TString::Format("%s > %.2f", variables[i].Data(), low));
    }
    else if (boundTypes[i] == UPPERFIXED) {
      double up = this->upperBounds[this->getBoundedVariablePosition(variables[i])];
      this->evaluator->setRangeAxis(this->varIdx[i], this->axisMin[i], up);
      INFO(TString::Format("%s < %.2f", variables[i].Data(), up));
    }
    else if (boundTypes[i] == WINDOWFIXED) {
      double low = this->lowerBounds[this->getBoundedVariablePosition(variables[i])];
      double up = this->upperBounds[this->getBoundedVariablePosition(variables[i])];
      this->evaluator->setRangeAxis(this->varIdx[i],low, up);
      INFO(TString::Format("%.2f < %s < %.2f", low, variables[i].Data(), up));
    }
    else if (boundTypes[i] == UPPER || boundTypes[i] == LOWER ||
	     boundTypes[i] == UPPERSWITCH || boundTypes[i] == LOWERSWITCH) {
      this->axesToScan.push_back(this->varIdx[i]);
    }
    else if (boundTypes[i] == SPLIT) {
      this->axesToScan.push_back(this->varIdx[i]);
    }
    else if (boundTypes[i] == UNDEFINED) {
      WARNclass(TString::Format("found a UNDEFINED boundType for variable %s", variables[i].Data()));
    }
    else {
      WARNclass("unkown bound type");
    }
  }
  int int_axisToEvaluate = this->getAxisIndex(this->ndimHists[0], axisToEvaluate);
  return this->evaluator->updateHists(this->axesToScan, this, int_axisToEvaluate);
}

void TQGridScanner::resetNdimHists() {
	// reset list of histograms
	this->ndimHists.clear();
}
void TQGridScanner::addNdimHist(THnBase* hist){
  // add a multidimensional histogram to the GridScanner
  // YOU SHOULD NEVER BE REQUIRED TO DO THIS MANUALLY
  // the significance evaluator should take care of this
  if(hist)
    this->ndimHists.push_back(hist);
}

void TQGridScanner::setVerbose(bool v){
  // set the verbosity
  // if set to true and the full frequentist/CL/Likelihood fit Machinery
  // is used, you will be left with a whole buch of output!
  this->verbose = v;
}

int TQGridScanner::getAxisIndex(THnBase* h, const TString& varname) {
  // get Axis index of variable with name varname in grid g
  int retval = -1;
  for (int i=0; i<h->GetNdimensions(); i++) {
    if (h->GetAxis(i)->GetName() == varname) {
      retval = i;
      break;
    }
  }
  if (retval < 0) {
    ERRORclass("No variable with name %s found in multidimensional histogram with name %s. Check your configuration!", varname.Data(), h->GetName());
    return -1;
  }
  return retval;
}

void TQGridScanner::setupRangesToScan() {
  // set the ranges to scan by checking if some variables have restricted range conditions
  for (size_t v=0; v<this->variables.size(); v++) {
    for (size_t bv=0; bv<this->boundedVariables.size(); bv++) {
      if (this->variables[v] == this->boundedVariables[bv]) {
				this->scanLowerBounds[v] = this->lowerBounds[bv];
				this->scanLowerBins[v] = this->lowerBoundedBins[bv];
				this->scanUpperBounds[v] = this->upperBounds[bv];
				this->scanUpperBins[v] = this->upperBoundedBins[bv];
				if (std::fabs(this->scanLowerBounds[v] - this->scanUpperBounds[v]) < PRECISION){
					if (this->boundTypes[v] == UPPER) {this->boundTypes[v] = UPPERFIXED;}
					if (this->boundTypes[v] == LOWER) {this->boundTypes[v] = LOWERFIXED;}
				}
      }
    }
  }
}

bool TQGridScanner::run(){
  // run the scan by scanning all variables
  // and calling the TQSignificanceEvaluator::evaluate() function
  // on each and every configuration visited
  // saving the result in an array of TQGridScanPoints

  this->setupRangesToScan();
  // if(this->verbose){
  VERBOSEclass("running over variable configuration:");
  this->printVariableConfiguration();
  //}
  this->nPointsProcessed = 0;
  this->nPointsTotal = 1;
  if(this->ndimHists.size() > 0) {
    for(size_t i=0; i<this->variables.size(); i++) {
      if (this->boundTypes[i] == UPPERSWITCH || this->boundTypes[i] == LOWERSWITCH) {
	this->nPointsTotal *= 2;
      }
      else if (this->boundTypes[i] == UPPER || this->boundTypes[i] == LOWER) {
	this->nPointsTotal *= (this->scanUpperBins[i]-this->scanLowerBins[i])+1;
      }
    }
  }
  INFOclass("estimated total of %i points", this->nPointsTotal);
  this->formatString = TString("%0")+TString::Format("%d",(int)(ceil(log10(this->nPointsTotal))))+"d.%d";
  this->heartbeat = TQUtils::getCurrentTime();
  this->runTimer->Init(this->nPointsTotal);
  bool retval = this->run(0);
  return retval;
}

bool TQGridScanner::run(size_t varno){
  // this is the worker function, recursing over all variables
  // iterating over all bins for each variable, configuring the grids
  // and calling the evaluator upon the respective configuration
  size_t max = this->scanUpperBins[varno];
  if (boundTypes[varno] == UPPERSWITCH || boundTypes[varno] == LOWERSWITCH) {
    // make sure the following loop iterates once
    max = this->scanLowerBins[varno]+1;
  }
  if (boundTypes[varno] == WINDOWFIXED) {
    // make sure that the fixed window cut does not iterate
    max = this->scanLowerBins[varno];
  }
  for(size_t i=this->scanLowerBins[varno]; i <= max; i++){
    this->currentBins[varno] = i;
    // if (boundTypes[varno] != UPPERFIXED && boundTypes[varno] != LOWERFIXED &&
    //  		boundTypes[varno] != WINDOWFIXED) {
    this->currentVals[varno] = this->getBinMin(varno, i);
    // }
    if (boundTypes[varno] == UPPER || boundTypes[varno] == UPPERFIXED) {
      this->setUpperRange(varno);
    } else if (boundTypes[varno] == LOWER || boundTypes[varno] == LOWERFIXED) {
      this->setLowerRange(varno);
    } else if (boundTypes[varno] == UPPERSWITCH) {
      this->currentSwitchStatus[varno] = this->switchVars[varno] ? "on" : "off";
      this->setUpperRange(varno);
    } else if (boundTypes[varno] == LOWERSWITCH) {
      this->currentSwitchStatus[varno] = this->switchVars[varno] ? "on" : "off";
      this->setLowerRange(varno);
    } else if (boundTypes[varno] == WINDOWFIXED) {
      this->setWindowRange(varno);
    } else if(boundTypes[varno] == SPLIT){
      // if the variable is a SPLIT, under/overflow bins are meaningless
      // in this case, we also use the bin maximum
      // to implicitly exclude the underflow
      // but we also skip the last iteration
      // to exclude the overflow as well
			// this->currentVals[varno] = this->grids[0]->getBinMax(variables[varno],i);
    } else {
      WARNclass("unkown bound type");
      return false;
    }
    if(varno+1 < variables.size()){
      // else, we simply increase the depth
      if(!this->run(varno+1))
        return false;
			//this->switchVars[varno] = 1;
      continue;
    }
    this->nPointsProcessed++;
    // this->printConfiguration();
    if(!this->evaluator) return false;
    this->updateTime();
    double significance = 0;
    bool success = false;
    int iConf = 0;
    TString info = "";
    this->evaluator->info = "";
    if(splitVars.size() == 0){
      this->evaluator->setFileIdentifier(TString::Format(this->formatString.Data(),(int)(this->nPointsProcessed),iConf));
      significance = this->evaluator->evaluate();
      if(significance < 0){
        ERRORclass("significance evaluator '%s' returned invalid result, quitting evaluation now!",this->evaluator->GetName());
        return false;
      }
      success = TQUtils::isNum(significance);
      info = this->evaluator->info;
    } else {
      // we use a while loop to iterate over all splitting variables
      // and set the bounds in upper/lower mode, accordingly
      if(this->evaluator->hasNativeRegionSetHandling()){
        this->splitConfigInfoStr = "";
        while(this->getSplitConfiguration(iConf)){
          if(!this->evaluator->prepareNextRegionSet(this->splitConfigInfoStr)){
            ERRORclass("significance evaluator announced an error during preparation of region %d: %s", this->evaluator->GetName(),iConf,this->splitConfigInfoStr.Data());
            return false;
          }
          iConf++;
          this->splitConfigInfoStr = "";
        }
        this->evaluator->setFileIdentifier(TString::Format(this->formatString.Data(),(int)(this->nPointsProcessed),0));
        if(!this->evaluator->isPrepared()){
          ERRORclass("significance evaluator '%s' is unprepared after preparing %i regions!",this->evaluator->GetName(),iConf);
          return false;
        }
        significance = this->evaluator->evaluate();
        success = true;
        info += this->evaluator->info;
      } else {
        double sig2 = 0;
        while(this->getSplitConfiguration(iConf)){
          this->evaluator->setFileIdentifier(TString::Format(this->formatString.Data(),(int)(this->nPointsProcessed),iConf));
          double sig = this->evaluator->evaluate();
          iConf++;
          if(!TQUtils::isNum(sig))
            continue;
          if(sig < 0){
            ERRORclass("significance evaluator '%s' returned invalid result, quitting evaluation now!",this->evaluator->GetName());
            return false;
          }
          sig2 += pow(sig,2);
          success = true;
          info += this->evaluator->info;
        }
        significance = sqrt(sig2);
	info += TString::Format("Z(combined) = %.4f", significance);
      }
    }
    if(this->verbose) std::cout << "processing point " << this->nPointsProcessed << " : ";
    else this->runTimer->DrawProgressBar(this->nPointsProcessed);
    if(!success){
      std::cout << "no significance measured: fit diverged or no evaluation requested. evaluator feedback was: " << this->evaluator->info << std::endl;
      continue;
    } 
    // if this is the maximum depth, we should create a new point
    // supply it with all necessary information
    TQGridScanPoint* p = new TQGridScanPoint(this->variables,currentVals,currentSwitchStatus);
    // and evaluate the TQSignificaneEvaluator on it
    p->significance = significance;
    p->evalInfoStr = info;
    p->id=this->nPointsProcessed;
    if(this->verbose){
      //std::cout << "Zexp(" << this->evaluator->GetName() << ")=" << p->significance << "\t";
      for(size_t j=0; j<this->variables.size(); j++){
	if (boundTypes[j]==UPPERSWITCH || boundTypes[j]==LOWERSWITCH) {
	  // TString status = this->switchVars[j] ? "on" : "off";
	  std::cout << this->variables[j] << " cut " << this->currentSwitchStatus[j];
	}
	else if (boundTypes[j]==WINDOWFIXED) {
	  std::cout << this->scanLowerBounds[j] << "<" << this->variables[j];
	  std::cout << "<" << this->scanUpperBounds[j];
	} else {
	  std::cout << this->variables[j] << this->getBoundOperator(j) << this->currentVals[j];
	}
        if(j != this->variables.size()-1)
          std::cout << ", ";
        else 
          std::cout << "\t";
      }
      std::cout << p->evalInfoStr << std::endl;
    }
    this->points.push_back(p);
  }
  return true;
}

void TQGridScanner::setHeartbeat(TString cmd, Long64_t time){
  // set the "heartbeat" command that will be executed in regular intervals
  // during the evaluation
  // this capability can be used to renew AFS tickets or perform similar tasks
  // that are required to keep the evaluation "alive" for longer runs
  this->heartBeatCommand = cmd;
  this->heartBeatInterval = time;
}

bool TQGridScanner::getSplitConfiguration(int n, size_t varno){
  // this function iterates recursively over all split configurations
  // for a given set of cut and split values
  // that is, switches upper and lower bounds of split variables
  // until all combinations have been traversed
  // the first argument n is the index of the configuration to be aquired
  // if the function succeeded in aquiring configuration n, true is returned
  // if n exceeds the total number of available configurations, false is returned
  int operand = pow(2,splitVars.size() - varno -1) ;
  int myval = n / operand;
  if(myval > 1)
    return false;
  if(myval == 0){
    this->splitConfigInfoStr += "_low"+this->variables[this->splitVars[varno]];
  } else {
    this->splitConfigInfoStr += "_high"+this->variables[this->splitVars[varno]];
  }
  
  for(size_t k=0; k<this->ndimHists.size(); k++){
    if(myval == 0){
      this->ndimHists[k]->GetAxis(this->varIdx[this->splitVars[varno]])->SetRangeUser(this->axisMin[this->splitVars[varno]]-1, this->currentVals[this->splitVars[varno]]);
      // this->grids[k]->setRangeBins(this->variables[this->splitVars[varno]],
      //                              this->getMinBin(this->variables[this->splitVars[varno]]),
      //                              this->currentBins[this->splitVars[varno]]);
    } else {
      this->ndimHists[k]->GetAxis(this->varIdx[this->splitVars[varno]])->SetRangeUser(this->currentVals[this->splitVars[varno]], this->axisMax[this->splitVars[varno]]+1);
      // this->grids[k]->setRangeBins(this->variables[this->splitVars[varno]],
      //                              this->currentBins[this->splitVars[varno]]+1,
      //                              this->getMaxBin(this->variables[this->splitVars[varno]]));
    }
  }

  if(varno+1 < splitVars.size())
    return this->getSplitConfiguration(n % operand,varno+1);
  return true;
}

bool TQGridScanner::updateTime(){
  // this routine will submit the heartbeat if neccessary
  if(this->heartBeatInterval == 0)
    return false;
  if(this->heartBeatCommand.IsNull())
    return false;
  unsigned long t = TQUtils::getCurrentTime();
  if(this->heartbeat + this->heartBeatInterval > t)
    return false;
  this->heartbeat = t;
  gSystem->Exec(this->heartBeatCommand.Data());
  return true;
}
 
void TQGridScanner::sortPoints(){
  // sort the points by their respective significance
  if(!(this->sorted))
    sort(points.begin(), points.end(), TQGridScanPoint::greater);
  this->sorted = true;
}

void TQGridScanner::sortVariables(TString ordering){
  // sort the variables according to the given ordering
  if(ordering.IsNull() && (this->variableOrdering.size() == this->variables.size())) return;
  this->variableOrdering.clear();
  TList* vars = TQStringUtils::tokenize(ordering,",");
  if(vars) vars->SetOwner(true);
  TQIterator itr(vars,true);
  std::vector<bool> moved(variables.size(),false);
  while(itr.hasNext()){
    TObject* obj = itr.readNext();
    if(!obj) continue;
    TString v = TQStringUtils::trim(obj->GetName());
    size_t i = this->findVariable(v);
    if(i > variables.size()) continue;
    if(moved[i]) continue;
    this->variableOrdering.push_back(i);
    moved[i]=true;
  }
  for(size_t i=0; i<this->variables.size(); i++){
    if(moved[i]) continue;
    variableOrdering.push_back(i);
  }
}

size_t TQGridScanner::nPoints(){
  // get the number of points visited by this GridScanner
  return points.size();
}

bool TQGridScanner::isInVariables(const TString& varname) {
	for (unsigned int i=0; i<this->variables.size(); i++) {
		if (varname == this->variables[i]) return 1;
	}
	return 0;
}

size_t TQGridScanner::getVariablePosition(const TString& varname) {
	int retval = -1;
	for (unsigned int i=0; i<this->variables.size(); i++) {
		if (varname == this->variables[i]) return i;
	}
	WARNclass(TString::Format("no variable %s found in available variables!", varname.Data()));
	return retval;
}

size_t TQGridScanner::getBoundedVariablePosition(const TString& varname) {
	int retval = -1;
	for (unsigned int i=0; i<this->boundedVariables.size(); i++) {
		if (varname == this->boundedVariables[i]) return i;
	}
	WARNclass(TString::Format("no bounded variable %s found in available bounded variables!", varname.Data()));
	return retval;
}

size_t TQGridScanner::findVariable(const TString& varname, BoundType type){
  for(size_t i=0; i<this->variables.size(); i++){
    if(varname != variables[i])
      continue;
    if((type == UNDEFINED) || (type == boundTypes[i]))
      return i;
  }
  return -1;
} 

size_t TQGridScanner::findVariable(const TString& varname){
  // return the index of the variable with the given name
  size_t sepindex = varname.Last(':');
  if(sepindex > (size_t)(varname.Length())) return this->findVariable(varname,UNDEFINED);
  TString vname = varname(0,sepindex);
  BoundType type = UNDEFINED;
  TString tname = varname(sepindex+1,varname.Length());
  if(tname == "min") type = LOWER;
  if(tname == "max") type = UPPER;
  if(tname == "minfixed") type = LOWERFIXED;
  if(tname == "maxfixed") type = UPPERFIXED;
  if(tname == "minswitch") type = LOWERSWITCH;
  if(tname == "maxswitch") type = UPPERSWITCH;
  if(tname == "split") type = SPLIT;
  size_t retval = this->findVariable(vname,type);
  return retval;
} 

TString TQGridScanner::getVariableName(size_t index, bool includeBound){
  // return the name of the variable with the given index
  // if includeBound is true, the returned string
  // will have either of the following appended
  // - 'min' if the variable has a lower bound
  // - 'max' if the variable has a lower bound
  // - 'split' if the variable is a splitting criterion
  TString vname = this->variables[index];
  while(index >= boundTypes.size()){
    boundTypes.push_back(UNDEFINED);
  }
  if(!includeBound) return vname;
  if(boundTypes[index] != UNDEFINED) vname += ":";
  if(boundTypes[index] == LOWER) return vname + "min";
  if(boundTypes[index] == UPPER) return vname + "max";
  if(boundTypes[index] == LOWERFIXED) return vname + "minfixed";
  if(boundTypes[index] == UPPERFIXED) return vname + "maxfixed";
  if(boundTypes[index] == LOWERSWITCH) return vname + "minswitch";
  if(boundTypes[index] == UPPERSWITCH) return vname + "maxswitch";
  if(boundTypes[index] == SPLIT) return vname + "split";
  return vname+"_undefined";
}

bool TQGridScanner::isAcceptedPoint(size_t i){
  // check if the point meets the current requirements
  if(this->points.size() < i)
    return false;
  for(size_t j=0; j<this->requirements.size(); j++){
    size_t kVar = this->findVariable(this->requirements[j]);
    if(kVar > points[i]->variables.size()) continue;
    if(this->requirements_types[j] == "="){
      if(this->points[i]->coordinates[kVar] != this->requirements_values[j]){
        return false;
      }
    }
    if(this->requirements_types[j] == "<"){
      if(this->points[i]->coordinates[kVar] > this->requirements_values[j]){
        return false;
      }
    }
    if(this->requirements_types[j] == ">"){
      if(this->points[i]->coordinates[kVar] < this->requirements_values[j]){
        return false;
      }
    }
    if(this->requirements_types[j] == "!="){
      if(this->points[i]->coordinates[kVar] == this->requirements_values[j]){
        return false;
      }
    }
  }
  return true;
}

void TQGridScanner::printConfiguration(){
  // print the current variable cut configuration
  for(size_t i=0; i<this->variables.size(); i++){
		std::cout << this->variables[i] << "=" << this->currentVals[i] << " " << "(bin:" << this->currentBins[i] << ")" << std::endl;
  }
}

void TQGridScanner::printPoints(size_t first, size_t last){
  // print all points witz indices ranging between first and last
  // will only print points that meet the requirements
  if(last >= this->points.size())
    last = this->points.size() -1;
  for(size_t i = first; i<last; i++){
    if(isAcceptedPoint(i))
      this->printPoint(i);
  }
}


void TQGridScanner::printPoints(size_t last){
  // print all points up to given index
  // will only print points that meet the requirements
  this->printPoints(0,last);
}
 
TString TQGridScanner::getBoundOperator(const TString& varname){
  // return a string version of the bound operator for the given variable name
  for(size_t i=0; i<variables.size(); i++){
    if(variables[i] == varname)
      return this->getBoundOperator(i);
  }
  return "?";
}


TString TQGridScanner::getBoundOperator(size_t i){
  // return a string version of the bound operator for the given variable
  if(this->boundTypes[i]==UPPER || this->boundTypes[i]==UPPERFIXED || this->boundTypes[i]==UPPERSWITCH)
    return "<";
  if(this->boundTypes[i]==LOWER || this->boundTypes[i]==LOWERFIXED || this->boundTypes[i]==LOWERSWITCH)
    return ">";
  if(this->boundTypes[i]==SPLIT)
    return "|";
  return "?";
}
 
void TQGridScanner::printPoint(size_t n){
  // print all available information on point number n
  if(this->points.size() < 1){
    ERRORclass("cannot print point: point array is empty!");
    return;
  }
  if(n >= this->points.size())
    n = this->points.size() -1;
  std::cout << "point #" << n << ": ";
  for(size_t i=0; i<this->variables.size(); i++){
    if (this->boundTypes[i]==UPPERFIXED || this->boundTypes[i]==LOWERFIXED ||
	this->boundTypes[i]==WINDOWFIXED) continue;
    if (this->boundTypes[i]==UPPERSWITCH || this->boundTypes[i]==LOWERSWITCH) {
      std::cout << this->variables[i] << " cut " << this->points[n]->switchStatus[i];
    } else {
      std::cout << this->variables[i] << this->getBoundOperator(i) << this->points[n]->coordinates[i];
      if (this->points[n]->coordinates[i] == this->scanLowerBounds[0] ||
	  this->points[n]->coordinates[i] == this->scanLowerBounds.back() ||
	  this->points[n]->coordinates[i] == this->scanUpperBounds[0] ||
	  this->points[n]->coordinates[i] == this->scanUpperBounds.back()) {
	std::cout << " (limited)";
      }
    }
    if(i != this->variables.size()-1)
      std::cout << ", ";
    else 
      std::cout << "\t";
  }
  if(this->points[n]->significance > 0)
    //std::cout << "Zexp(" << (this->evaluator ? this->evaluator->GetName() : "???") << ") = " << this->points[n]->significance;
    std::cout << "\t" << points[n]->evalInfoStr << std::endl;
}

TQGridScanPoint* TQGridScanner::point(size_t n){
  // retrieve a pointer to point number n from the array
  if(n >= this->points.size())
    n = this->points.size() -1;
  return points[n];
}

TLine* TQGridScanner::drawCutLine(TH1* hist){
  // draw the vertical line denoting the current cut value
  TString vname = TString::Format("cut.%s",hist->GetName());
  if(!this->hasTag(vname)) vname.ReplaceAll(":","");
  if(!this->hasTag(vname)){
    size_t idx = this->findVariable(hist->GetName());
    if(idx > variables.size()) return NULL;
    vname = TString::Format("cut.%s",variables[idx].Data());
  }
  double cut = this->getTagDoubleDefault(vname,std::numeric_limits<double>::quiet_NaN());
  if(!TQUtils::isNum(cut))
    return NULL;
  double xmax = TQHistogramUtils::getAxisXmax(hist);
  double xmin = TQHistogramUtils::getAxisXmin(hist);
  if(cut > xmax || cut < xmin) return NULL;
  gPad->Modified();
  gPad->Update();
  double ymin = gPad->GetUymin();
  double ymax = gPad->GetUymax();
  TLine* cutline = new TLine(cut,ymin,cut,ymax);
  int cutlinecolor = this->getTagIntegerDefault("cutLine.color",kRed);
  int cutlinestyle = this->getTagIntegerDefault("cutLine.style",7);
  int cutlinewidth = this->getTagIntegerDefault("cutLine.width",2);
  cutline->SetLineStyle(cutlinestyle);
  cutline->SetLineWidth(cutlinewidth);
  cutline->SetLineColor(cutlinecolor);
  cutline->Draw();
  return cutline;
}

int TQGridScanner::drawCutLines(TH2* hist){
  // draw the lines denoting the current cut value
  TString vname1 = TString::Format("cut.%s",hist->GetYaxis()->GetName());
  TString vname2 = TString::Format("cut.%s",hist->GetXaxis()->GetName());
  if(!this->hasTag(vname1)) vname1.ReplaceAll(":","");
  if(!this->hasTag(vname2)) vname2.ReplaceAll(":","");
  if(!this->hasTag(vname1)){
    size_t idx = this->findVariable(hist->GetYaxis()->GetName());
    if(idx > variables.size()) return 0;
    vname1 = TString::Format("cut.%s",variables[idx].Data());
  }
  if(!this->hasTag(vname2)){
    size_t idx = this->findVariable(hist->GetXaxis()->GetName());
    if(idx > variables.size()) return 0;
    vname2 = TString::Format("cut.%s",variables[idx].Data());
  }
  double cut1 = this->getTagDoubleDefault(vname1,std::numeric_limits<double>::quiet_NaN());
  double cut2 = this->getTagDoubleDefault(vname2,std::numeric_limits<double>::quiet_NaN());
  if(!TQUtils::isNum(cut1) && !TQUtils::isNum(cut2))
    return 0;
  gPad->Modified();
  gPad->Update();
  double ymin = TQHistogramUtils::getAxisYmin(hist);
  double ymax = TQHistogramUtils::getAxisYmax(hist);
  double xmin = TQHistogramUtils::getAxisXmin(hist);
  double xmax = TQHistogramUtils::getAxisXmax(hist);
  int cutlinecolor = this->getTagIntegerDefault("cutLine2D.color",kBlack);
  int cutlinestyle = this->getTagIntegerDefault("cutLine2D.style",7);
  int cutlinewidth = this->getTagIntegerDefault("cutLine2D.width",2);
  int nLines = 0;
  TLatex l;
  l.SetTextSize(this->getTagDoubleDefault("cutLine2D.textSize",0.03) * this->getTagDoubleDefault("cutLine2D.textScale",1.));
  l.SetTextColor(1);
  if(TQUtils::isNum(cut1) && cut1 < ymax && cut1 > ymin){
    TLine* cutline1 = new TLine(xmin,cut1,xmax,cut1);
    cutline1->SetLineStyle(cutlinestyle);
    cutline1->SetLineWidth(cutlinewidth);
    cutline1->SetLineColor(cutlinecolor);
    cutline1->Draw();
    nLines++;
    TString cutLineLabel;
    if(this->getTagString("cutLine2D.text",cutLineLabel)){
      l.DrawLatex(xmin + this->getTagDoubleDefault("cutLine2D.xPos",0.05) * (xmax - xmin), 
                  cut1 + this->getTagDoubleDefault("cutLine2D.yShift",0.01) * (ymax - ymin), 
                  cutLineLabel);
    };
  }
  if(TQUtils::isNum(cut2) && cut2 < xmax && cut2 > xmin){
    TLine* cutline2 = new TLine(cut2,ymin,cut2,ymax);
    cutline2->SetLineStyle(cutlinestyle);
    cutline2->SetLineWidth(cutlinewidth);
    cutline2->SetLineColor(cutlinecolor);
    cutline2->Draw();
    TString cutLineLabel;
    if(this->getTagString("cutLine2D.text",cutLineLabel)){
      l.SetTextAngle(90.);
      l.DrawLatex(cut2 - this->getTagDoubleDefault("cutLine2D.xShift",0.01) * (xmax - xmin),
                  ymin + this->getTagDoubleDefault("cutLine2D.yPos",0.05) * (ymax - ymin), 
                  cutLineLabel);
    };
    nLines++;
  }
  return nLines;
}

void TQGridScanner::drawColAxisTitle(TQTaggable& tags, TH2* hist, const TString& title){
  // draw the color axis title for 2D significance profiles
  gPad->Modified(); 
  gPad->Update();
  TLatex l;
  double label_x = TQHistogramUtils::getAxisXmax(hist);
  TPaletteAxis* palette = dynamic_cast<TPaletteAxis*>(hist->GetListOfFunctions()->FindObject("palette"));
  if(palette) {
    label_x = 0.5*(palette->GetX1() + palette->GetX2());
    l.SetTextAlign(21);
  } else {
    ERRORclass("no pallette found");
  }
  double label_y = tags.getTagDoubleDefault("style.axis.titleOffset",1.0)*0.04*(TQHistogramUtils::getAxisYmax(hist) - TQHistogramUtils::getAxisYmin(hist)) + TQHistogramUtils::getAxisYmax(hist);
  l.SetTextFont(hist->GetYaxis()->GetTitleFont());
  l.SetTextSize(hist->GetYaxis()->GetTitleSize());
  l.SetNDC(false);
  l.DrawLatex(label_x,label_y,title);
  gPad->Modified(); 
  gPad->Update();
}

void TQGridScanner::drawLabels(TQTaggable& tags){
  // draw some of the official atlas labels
  double textsize = tags.getTagDoubleDefault("style.textSize",0.05);
  int font = tags.getTagDoubleDefault("style.text.font",42);
  int color = tags.getTagDoubleDefault("style.text.color",1);
  double x = tags.getTagDoubleDefault("style.labels.xOffset",0.2);
  double y = tags.getTagDoubleDefault("style.labels.yPos",0.92);
  if (tags.getTagBoolDefault("style.drawATLAS",false)) {
    /* draw the ATLAS label */
    TLatex l;
    l.SetNDC();
    l.SetTextFont(72);
    l.SetTextSize(textsize * tags.getTagDoubleDefault("labels.drawATLAS.scale",1.25));
    l.SetTextColor(1);
    l.DrawLatex(x, y, tags.getTagStringDefault("labels.drawATLAS.text","ATLAS"));
    TString atlasLabel;
    if(tags.getTagString("labels.atlas.text",atlasLabel)){
      /* draw the ATLAS label addition */
      TLatex p;
      p.SetNDC();
      p.SetTextFont(font);
      p.SetTextColor(color);
      p.SetTextSize(textsize * tags.getTagDoubleDefault("labels.atlas.scale",1.25));
      p.DrawLatex(x + tags.getTagDoubleDefault("labels.atlas.xOffset",0.16), y, atlasLabel.Data());
    }
  }
}

void TQGridScanner::setStyle1D(TPad* pad, TH1* hist,TQTaggable& tags, TH1* histmax){
  // set the style for 1D plotting
  this->setStyle(pad,hist,tags);
  TQPlotter::setStyleAtlas();
  // pad->SetRightMargin(0.01);
  bool showTitle = tags.getTagBoolDefault("style.showTitle",true);
  if(!showTitle){
    gStyle->SetOptTitle(false);
    // pad->SetTopMargin(0.01);
  }
  hist->UseCurrentStyle();
  pad->UseCurrentStyle();
  // histogram style
  hist->SetMarkerStyle(20);
  hist->SetMarkerColor(kBlack);
  hist->SetLineColor(kBlack);
  if (histmax) {
    histmax->SetMarkerStyle(21);
    histmax->SetMarkerColor(kBlue);
  }
  gPad->Modified(); 
  gPad->Update();
}

void TQGridScanner::setStyle(TPad*/*pad*/, TH1*/*hist*/, TQTaggable& tags){
  // set the general plotting style
	// gROOT->LoadMacro("AtlasUtils.C");
  double axisFontSize = tags.getTagDoubleDefault("style.axis.fontSize",0.04); 
  double axisTitleSize = tags.getTagDoubleDefault("style.axis.titleSize",0.04);
  gStyle->SetLabelSize(axisFontSize,"XY");
  gStyle->SetTitleSize(axisTitleSize,"XY");
  // pad->SetBottomMargin((hist->GetXaxis()->GetTitleOffset()*axisFontSize + axisTitleSize)*1.3);
  // pad->SetLeftMargin((hist->GetYaxis()->GetTitleOffset()*axisFontSize + axisTitleSize)*1.1);
}

// void TQGridScanner::setStyle2D(TPad* pad, TH2* hist,TQTaggable& tags){
//   // set the style for 2D plotting
//   this->setStyle(pad,hist,tags);
//   gStyle->SetPadTopMargin(0.1);
//   gStyle->SetPadRightMargin(0.1);
//   gPad->Modified(); 
//   gPad->Update();
// }

TLegend* TQGridScanner::drawLegend(TQTaggable& tags, TH1F* hist, const TString& histlabel, TH1F* histmax, TLine* cutline){
  // draw the legend for this plot
  if(!hist && !histmax) return NULL;
  gPad->Modified(); 
  gPad->Update();
  double ndc_hmargin = tags.getTagDoubleDefault("style.legend.hmargin",0.11);
  double ndc_width = tags.getTagDoubleDefault("style.legend.width",0.3);
  double ndc_height = tags.getTagDoubleDefault("style.legend.height",0.3);
  double ndc_vmargin = tags.getTagDoubleDefault("style.legend.vmargin",0.02);
  double xmax = TQHistogramUtils::getAxisXmax(hist);
  double xmin = TQHistogramUtils::getAxisXmin(hist);
  double ymax = TQHistogramUtils::getHistogramMaximum(2,hist,histmax);
  double ymin = TQHistogramUtils::getHistogramMinimum(2,hist,histmax);
  double width = ndc_width * (xmax - xmin);
  double height = ndc_height * (ymax - ymin);
  double hmargin = ndc_hmargin * (xmax - xmin);
  double hmargin_left = (ndc_hmargin-0.105) * (xmax - xmin);
  double vmargin = ndc_vmargin * (ymax - ymin);
  double left = xmin + hmargin;
  double right = xmax - hmargin;
  double bottom = ymin + vmargin;
  double top = ymax - vmargin;
  // first try: lets fit the legend on the right side
  double min,max;
  bool drawn = false;
  if(!drawn){
    min = std::min(TQHistogramUtils::getMinimumBinValue(hist,xmax-width-hmargin,xmax),TQHistogramUtils::getMinimumBinValue(histmax,xmax-width-hmargin,xmax));
    max = std::max(TQHistogramUtils::getMaximumBinValue(hist,xmax-width-hmargin,xmax),TQHistogramUtils::getMaximumBinValue(histmax,xmax-width-hmargin,xmax));
    if(max < top-height-vmargin){
      // does it fit at the top?
      bottom = top-height;
      drawn = true;
    } else if(min > bottom+height+vmargin){
      // does it fit at the bottom?
      top = bottom+height;
      drawn = true;
    }
    if(drawn) left = right-width;
  }
  if(!drawn){
    // next try: lets fit the legend on the left side
    min = std::min(TQHistogramUtils::getMinimumBinValue(hist,xmin,xmin+width+hmargin_left),TQHistogramUtils::getMinimumBinValue(histmax,xmin,xmin+width+hmargin_left));
    max = std::max(TQHistogramUtils::getMaximumBinValue(hist,xmin,xmin+width+hmargin_left),TQHistogramUtils::getMaximumBinValue(histmax,xmin,xmin+width+hmargin_left));
    if(max < top-height-vmargin){
      // does it fit at the top?
      bottom = top-height;
      drawn = true;
    } else if(min > bottom+height+vmargin){
      // does it fit at the bottom?
      top = bottom+height;
      drawn = true;
    }
    if(drawn) right = left+width;
  }
  TLegend* legend = NULL;
  if(drawn){
    legend = new TLegend(TQUtils::convertXtoNDC(left),TQUtils::convertYtoNDC(bottom),TQUtils::convertXtoNDC(right),TQUtils::convertYtoNDC(top));
  } else {
    legend = new TLegend(TQUtils::convertXtoNDC(right-width),TQUtils::convertYtoNDC(top-height),TQUtils::convertXtoNDC(right),TQUtils::convertYtoNDC(top));
    // we're unable to fit the legend, we need to rescale 
    if(histmax) histmax->SetMaximum(1.3*histmax->GetMaximum());
    if(hist) hist->SetMaximum(1.3*hist->GetMaximum());
  }
  legend->SetTextSize(tags.getTagDoubleDefault("style.legend.fontSize",0.04));
  legend->SetFillStyle(3001);
  legend->SetFillColor(kWhite);
  legend->SetBorderSize(0);
  if(histmax) legend->AddEntry(histmax,tags.getTagStringDefault("style.optimum.label","profiled optimum"), "L");
  if(hist) legend->AddEntry(hist, histlabel, "LE");
  if(cutline) legend->AddEntry(cutline,tags.getTagStringDefault("style.cutline.label","current cut value"),"L");
  if(legend) legend->Draw();
  return legend;
}

void TQGridScanner::setYaxisTitleOffset(TQTaggable& tags, TH1* hist){
  // set the title offset of the vertical axis
  double max = hist->GetYaxis()->GetXmax();
  hist->GetYaxis()->SetTitleOffset(tags.getTagDoubleDefault("style.axis.titleOffset",1.0) * (0.9+(0.15*ceil(fabs(log10(max))))));
}

void TQGridScanner::setXaxisTitleOffset(TQTaggable& tags, TH1* hist){
  // set the title offset for the horizontal axis
  hist->GetXaxis()->SetTitleOffset(tags.getTagDoubleDefault("style.axis.titleOffset",1.0)*1.1);
}

void TQGridScanner::setAxisTitleOffset(TQTaggable& tags, TH1* hist){
  // set the title offset for both axis
  this->setXaxisTitleOffset(tags,hist);
  this->setYaxisTitleOffset(tags,hist);
}

















void TQGridScanner::deployHistogram(const TString& varname, TQFolder* f, int topNumber){
  // deploy a histogram of the variable varname 
  // containing the best topNumber points
  // in a given TQFolder 
  TQFolder* subf = f->getFolder(TQFolder::makeValidIdentifier(TString::Format("topn_%d",topNumber))+"+");
  TH1F* h = this->getHistogram(varname, topNumber);
  subf->addObject(h, ".histograms+::"+TQFolder::makeValidIdentifier(varname));
}

void TQGridScanner::deployHistogram(const TString& varname, TQFolder* f, double topFraction){
  // deploy a histogram of the variable varname 
  // containing the best (100*topFraction)% of points
  // in a given TQFolder 
  TQFolder* subf = f->getFolder(TQFolder::makeValidIdentifier(TString::Format("topf_%g",topFraction))+"+");
  TH1F* h = this->getHistogram(varname, topFraction);
  subf->addObject(h, ".histograms+::"+TQFolder::makeValidIdentifier(varname));
}

TH1F* TQGridScanner::getHistogram(const TString& varname, double topFraction){
  // obtain a histogram of the variable varname 
  // containing the best (100*topFraction)% of points
  return this->getHistogram(varname, (int)ceil(topFraction*(this->points.size())));
  //  ^^^ please note that this cast is important
  //  to avoid an infinite loop of-self-calls
  //  of this function
}

bool TQGridScanner::hasOtherVariable(size_t index){
  for(size_t i=0; i<this->variables.size(); i++){ 
    if(i == index) continue;
    if(this->variables[i]==variables[index])
      return true;
  }
  return false;
}
 

TH1F* TQGridScanner::getHistogram(const TString& varname, int topNumber){
  // obtain a histogram of the variable varname 
  // containing the best topNumber points
  this->sortPoints();
  size_t index = this->findVariable(varname);
  if(index >= variables.size())
    return NULL;
  //  bool showExtraBins = this->getTagBoolDefault("histogram.showExtraBins",false);
  //  bool showUnderflow = showExtraBins && (this->boundTypes[index] == LOWER);
  //  bool showOverflow = showExtraBins && (this->boundTypes[index] == UPPER);
  //  bool showTopBin = (this->boundTypes[index] != LOWER) || !this->hasOtherVariable(index); 
  TH1F* hist = new TH1F(varname, "title;title",
			//this->getVariableTitle(varname)+";"+this->getVariableTitle(varname),
			this->nBins[index],
			this->axisMin[index],
			this->axisMax[index]);
  // this->getVariableBinNumber(index)+(showTopBin)+(showExtraBins || (this->boundTypes[index] == SPLIT)),
  // this->getVariableMin(index) - (0.5+showUnderflow ) * this->getVariableStepWidth(index),
  // this->getVariableMax(index) + (0.5+showOverflow - !showTopBin) * this->getVariableStepWidth(index));
  hist->SetStats(kFALSE);
  hist->SetDirectory(NULL);
  if(!hist)
    return NULL;
  for(size_t i=0; i<std::min(this->points.size(),(size_t)(topNumber)); i++){
    if(!this->points[i] || !this->isAcceptedPoint(i))
      continue;
    double val = this->points[i]->coordinates[index];
    // if(showUnderflow && val < this->gridVars[index].min)
    //   val = this->gridVars[index].min - std::numeric_limits<double>::epsilon();
    // if(showOverflow && val > this->gridVars[index].max)
    //   val = this->gridVars[index].max + std::numeric_limits<double>::epsilon();
    hist->Fill(val);
  }
  hist->SetEntries(topNumber);
  TString plotTitle = this->getTagStringDefault("histogram.title");
  // if(plotTitle.IsNull()){
  //   plotTitle.ReplaceAll("$(VAR)",this->getVariableTitle(varname));
  //   hist->SetTitle(plotTitle);
  // }
  // if(showUnderflow)
  //   hist->GetXaxis()->SetTitle(TString(hist->GetXaxis()->GetTitle()) + " - lowest bin is underflow");
  // else if(showOverflow)
  //   hist->GetXaxis()->SetTitle(TString(hist->GetXaxis()->GetTitle()) + " - highest bin is overflow");
  return hist;
}

void TQGridScanner::plotAndSaveHistogram(const TString& varname, const TString& path, double topFraction, const TString& options){
  // obtain a histogram of the variable varname 
  // containing the best (100*topFraction)% of points
  // plot and save it under the given path with the given options
  TQTaggable tags(options);
  tags.importTags(this);
  TString extension = tags.getTagStringDefault("ext","pdf");
  TH1F* hist = this->getHistogram(varname, topFraction);
  if(!hist) return;
  TCanvas* c = new TCanvas("c", "c", 600, 600);
  gStyle->SetOptTitle(tags.getTagBoolDefault("style.showTitle",false));
  c->cd();
  this->setStyle1D(c,hist,tags);
  hist->Draw();
  this->drawCutLine(hist);
  gPad->Modified(); 
  gPad->Update();
  c->SaveAs(TQFolder::concatPaths(path,TQFolder::makeValidIdentifier(varname))+TQFolder::makeValidIdentifier(TString::Format("_topf_%g",topFraction))+"."+extension, extension);
	delete c;
}

void TQGridScanner::plotAndSaveHistogram(const TString& varname, const TString& path, int topNumber, const TString& options){
  // obtain a histogram of the variable varname 
  // containing the best topNumber points
  // plot and save it under the given path with the given options
  TQTaggable tags(options);
  tags.importTags(this);
  TString extension = tags.getTagStringDefault("ext","pdf");
  TH1F* hist = this->getHistogram(varname, topNumber);
  TCanvas* c = new TCanvas("c", "c", 600, 600);
  gStyle->SetOptTitle(tags.getTagBoolDefault("style.showTitle",false));
  c->cd();
  this->setStyle1D(c,hist,tags);
  hist->Draw();
  this->drawCutLine(hist);
  gPad->Modified(); 
  gPad->Update();
  c->SaveAs(TQFolder::concatPaths(path,TQFolder::makeValidIdentifier(varname))+TString::Format("_topn_%d",topNumber )+"."+extension, extension);
	delete c;
}

TList* TQGridScanner::getAllHistograms(double topFraction){
  // obtain the histograms for all variables containing the best (100*topFraction)% of points
  TList* list = new TList();
  this->sortVariables();
  for(size_t i=0; i<this->variableOrdering.size(); i++){
    list->Add(this->getHistogram(getVariableName(variableOrdering[i],true),topFraction));
  }
  return list;
}

TList* TQGridScanner::getAllHistograms(int topNumber){
  // obtain the histograms for all variables containing the best topNumber points
  TList* list = new TList();
  this->sortVariables();
  for(size_t i=0; i<this->variables.size(); i++){
    list->Add(this->getHistogram(getVariableName(variableOrdering[i],true),topNumber));
  }
  return list;
}

void TQGridScanner::deployAllHistograms(TQFolder* f, double topFraction){
  // deploy the histograms for all variables containing the best (100*topFraction)% of points 
  // in a given TQFolder
  this->sortVariables(); 
  for(size_t i=0; i<this->variables.size(); i++){
    this->deployHistogram(getVariableName(variableOrdering[i],true),f,topFraction);
  }
}

void TQGridScanner::deployAllHistograms(TQFolder* f, int topNumber){
  // deploy the histograms for all variables containing the best topNumber points
  // in a given TQFolder
  this->sortVariables(); 
  for(size_t i=0; i<this->variables.size(); i++){
    this->deployHistogram(getVariableName(variableOrdering[i],true),f,topNumber);
  }
}

void TQGridScanner::plotAndSaveAllHistograms(const TString& path, double topFraction, const TString& options){
  // obtain the histograms for all variables containing the best (100*topFraction)% of points
  // plot and save them under the given path with the given options
	gROOT->SetBatch(true);
  this->sortVariables();
  for(size_t i=0; i<this->variables.size(); i++){
    this->plotAndSaveHistogram(getVariableName(variableOrdering[i],true),path,topFraction,options);
  }
	gROOT->SetBatch(false);
}
void TQGridScanner::plotAndSaveAllHistograms(const TString& path, int topNumber, const TString& options){
  // obtain the histograms for all variables containing the best topNumber points
  // plot and save them under the given path with the given options
	gROOT->SetBatch(true);
  this->sortVariables();
  for(size_t i=0; i<this->variables.size(); i++){
    this->plotAndSaveHistogram(getVariableName(variableOrdering[i],true),path,topNumber,options);
  }
	gROOT->SetBatch(false);
}














/////////////////////////////////////////////////////////////////////////////////////////////////
// The following commented lines are not yet supported and need to be adjusted
// before being able to plot 2D histograms



// void TQGridScanner::deployHistogram2D(const TString& varname1, const TString& varname2, TQFolder* f, int topNumber){
//   // deploy a 2D histogram of the variables varname1 and varname2
//   // containing the best topNumber points
//   // in a given TQFolder 
//   TQFolder* subf = f->getFolder(TQFolder::makeValidIdentifier(TString::Format("topn_%d",topNumber))+"+");
//   TH2F* h = this->getHistogram2D(varname1, varname2, topNumber);
//   if(h)
//     subf->addObject(h, ".histograms+::"+TQFolder::makeValidIdentifier(varname1+"_"+varname2));
// }

// void TQGridScanner::deployHistogram2D(const TString& varname1, const TString& varname2, TQFolder* f, double topFraction){
//   // deploy a 2D histogram of the variables varname1 and varname2
//   // containing the best (100*topFraction)% of points
//   // in a given TQFolder 
//   TQFolder* subf = f->getFolder(TQFolder::makeValidIdentifier(TString::Format("topf_%g",topFraction))+"+");
//   TH2F* h = this->getHistogram2D(varname1, varname2, topFraction);
//   if(h)
//     subf->addObject(h, ".histograms+::"+TQFolder::makeValidIdentifier(varname1+"_"+varname2));
// }

// TH2F* TQGridScanner::getHistogram2D(const TString& varname1, const TString& varname2, double topFraction){
//   // obtain a 2D histogram of the variables varname1 and varname2
//   // containing the best (100*topFraction)% of points
//   return this->getHistogram2D(varname1, varname2, (int)ceil(topFraction*(this->points.size())));
//   //  ^^^ please note that this cast is important
//   //  to avoid an infinite loop of-self-calls
//   //  of this function
// }

// TH2F* TQGridScanner::getHistogram2D(const TString& varname1, const TString& varname2, int topNumber){
//   // obtain a 2D histogram of the variables varname1 and varname2
//   // containing the best topNumber points
//   this->sortPoints();
//   size_t index1 = this->findVariable(varname1);
//   size_t index2 = this->findVariable(varname2);
//   if(index1 >= gridVars.size() || index2 >= gridVars.size())
//     return NULL;
//   bool showExtraBins = this->getTagBoolDefault("histogram2D.showExtraBins",this->getTagBoolDefault("histogram.showExtraBins",false));
//   bool showUnderflow1 = showExtraBins && (this->boundTypes[index1]==LOWER);
//   bool showOverflow1 = showExtraBins && (this->boundTypes[index1]==UPPER);
//   bool showUnderflow2 = showExtraBins && (this->boundTypes[index2]==LOWER);
//   bool showOverflow2 = showExtraBins && (this->boundTypes[index2]==UPPER);
//   bool showTopBin1 = (this->boundTypes[index1] != LOWER) || !this->hasOtherVariable(index1); 
//   bool showTopBin2 = (this->boundTypes[index2] != LOWER) || !this->hasOtherVariable(index2); 
//   TH2F* hist = new TH2F(varname1+"_"+varname2, 
//                         this->getVariableTitle(varname1) + " vs. " + this->getVariableTitle(varname2)+";"+this->getVariableTitle(varname2)+";"+this->getVariableTitle(varname1),
//                         this->getVariableBinNumber(index2)+(showTopBin2)+(showExtraBins || (this->boundTypes[index2] == SPLIT)),
//                         this->getVariableMin(index2)-(0.5+showUnderflow2 )*this->getVariableStepWidth(index2),
//                         this->getVariableMax(index2)+(0.5+ showOverflow2-!showTopBin2)*this->getVariableStepWidth(index2),
//                         this->getVariableBinNumber(index1)+(showTopBin1)+(showExtraBins || (this->boundTypes[index1] == SPLIT)),
//                         this->getVariableMin(index1)-(0.5+showUnderflow1 )*this->getVariableStepWidth(index1),
//                         this->getVariableMax(index1)+(0.5+ showOverflow1-!showTopBin1)*this->getVariableStepWidth(index1));
//   hist->SetStats(kFALSE);
//   hist->SetDirectory(NULL);
//   if(!hist)
//     return NULL;
//   for(size_t i=0; i<std::min(this->points.size(),(size_t)(topNumber)); i++){
//     if(!this->points[i] || !this->isAcceptedPoint(i))
//       continue;
//     double val1 = this->points[i]->coordinates[index1];
//     double val2 = this->points[i]->coordinates[index2];
//     if(showUnderflow1 && val1 < this->gridVars[index1].min)
//       val1 = this->gridVars[index1].min - this->gridVars[index1].step;
//     if(showOverflow1 && val1 > this->gridVars[index1].max)
//       val1 = this->gridVars[index1].max + this->gridVars[index1].step;
//     if(showUnderflow2 && val2 < this->gridVars[index2].min)
//       val2 = this->gridVars[index2].min - this->gridVars[index2].step;
//     if(showOverflow2 && val2 > this->gridVars[index2].max)
//       val2 = this->gridVars[index2].max + this->gridVars[index2].step;
//     hist->Fill(val2,val1);
//   }
//   TString plotTitle = this->getTagStringDefault("histogram2D.title",this->getTagStringDefault("histogram.title"));
//   if(!plotTitle.IsNull()){
//     plotTitle.ReplaceAll("$(VARY)",this->getVariableTitle(varname1));
//     plotTitle.ReplaceAll("$(VARX)",this->getVariableTitle(varname2));
//     hist->SetTitle(plotTitle);
//   }
//   if(showUnderflow1)
//     hist->GetYaxis()->SetTitle(TString(hist->GetYaxis()->GetTitle()) + " - lowest bin is underflow");
//   else if(showOverflow1)
//     hist->GetYaxis()->SetTitle(TString(hist->GetYaxis()->GetTitle()) + " - highest bin is overflow");
//   if(showUnderflow2)
//     hist->GetXaxis()->SetTitle(TString(hist->GetXaxis()->GetTitle()) + " - lowest bin is underflow");
//   else if(showOverflow2)
//     hist->GetXaxis()->SetTitle(TString(hist->GetXaxis()->GetTitle()) + " - highest bin is overflow");
//   return hist;
// }

// void TQGridScanner::plotAndSaveHistogram2D(const TString& varname1, const TString& varname2, const TString& path, double topFraction, const TString& options){
//   // obtain a 2D histogram of the variables varname1 and varname2
//   // containing the best (100*topFraction)% of points
//   // plot and save it under the given path with the given options
//   TQTaggable tags(options);
//   tags.importTags(this);
//   TString extension = tags.getTagStringDefault("ext","pdf");
//   TH2F* hist = this->getHistogram2D(varname1, varname2, topFraction);
//   if(!hist) return;
//   TCanvas* c = new TCanvas("c", "c", 600, 600);
//   gStyle->SetOptTitle(tags.getTagBoolDefault("style.showTitle",false));
//   c->cd();
//   this->setStyle2D(c,hist,tags);
//   hist->Draw("colz");
//   this->drawLabels(tags);
//   gPad->Modified(); 
//   gPad->Update();
//   c->SaveAs(TQFolder::concatPaths(path,TQFolder::makeValidIdentifier(varname1+"_"+varname2))+TQFolder::makeValidIdentifier(TString::Format("_topf_%g",topFraction))+"."+extension, extension);
// 	delete c;
// }

// void TQGridScanner::plotAndSaveHistogram2D(const TString& varname1, const TString& varname2, const TString& path, int topNumber, const TString& options){
//   // obtain a 2D histogram of the variables varname1 and varname2
//   // containing the best topNumber points
//   // plot and save it under the given path with the given options
//   TQTaggable tags(options);
//   tags.importTags(this);
//   TString extension = tags.getTagStringDefault("ext","pdf");
//   TH2F* hist = this->getHistogram2D(varname1, varname2, topNumber);
//   if(!hist) return;
//   TCanvas* c = new TCanvas("c", "c", 600, 600);
//   gStyle->SetOptTitle(tags.getTagBoolDefault("style.showTitle",false));
//   c->cd();
//   this->setStyle2D(c,hist,tags);
//   hist->Draw("colz");
//   this->drawLabels(tags);
//   gPad->Modified(); 
//   gPad->Update();
//   c->SaveAs(TQFolder::concatPaths(path,TQFolder::makeValidIdentifier(varname1+"_"+varname2))+TString::Format("_topn_%d",topNumber )+"."+extension, extension);
// 	delete c;
// }

// TList* TQGridScanner::getAllHistograms2D(double topFraction){
//   // obtain the 2D histograms for all variable pairs
//   // containing the best (100*topFraction)% of points
//   TList* list = new TList();
//   this->sortVariables();
//   for(size_t i=0; i<this->variables.size(); i++){
//     for(size_t j=i+1; j<this->variables.size(); j++){
//       TH2F* hist = this->getHistogram2D(getVariableName(variableOrdering[i],true),getVariableName(variableOrdering[j],true),topFraction);
//       if(hist)
//         list->Add(this->getHistogram2D(getVariableName(variableOrdering[i],true),getVariableName(variableOrdering[j],true),topFraction));
//     }
//   }
//   return list;
// }

// TList* TQGridScanner::getAllHistograms2D(int topNumber){
//   // obtain the 2D histograms for all variable pairs 
//   // containing the best topNumber points
//   TList* list = new TList();
//   this->sortVariables();
//   for(size_t i=0; i<this->variables.size(); i++){
//     for(size_t j=i+1; j<this->variables.size(); j++){
//       TH2F* hist = this->getHistogram2D(getVariableName(variableOrdering[i],true),getVariableName(variableOrdering[j],true),topNumber);
//       if(hist)
//         list->Add(this->getHistogram2D(getVariableName(variableOrdering[i],true),getVariableName(variableOrdering[j],true),topNumber));
//     }
//   }
//   return list;
// }

// void TQGridScanner::deployAllHistograms2D(TQFolder* f, double topFraction){
//   // deploy the 2D histograms for all variable pairs 
//   // containing the best (100*topFraction)% of points 
//   // in a given TQFolder 
//   this->sortVariables();
//   for(size_t i=0; i<this->variables.size(); i++){
//     for(size_t j=i+1; j<this->variables.size(); j++){
//       this->deployHistogram2D(getVariableName(variableOrdering[i],true),getVariableName(variableOrdering[j],true),f,topFraction);
//     }
//   }
// }

// void TQGridScanner::deployAllHistograms2D(TQFolder* f, int topNumber){
//   // deploy the 2D histograms for all variable pairs 
//   // containing the best topNumber points 
//   // in a given TQFolder 
//   this->sortVariables();
//   for(size_t i=0; i<this->variables.size(); i++){
//     for(size_t j=i+1; j<this->variables.size(); j++){
//       this->deployHistogram2D(getVariableName(variableOrdering[i],true),getVariableName(variableOrdering[j],true),f,topNumber);
//     }
//   }
// }

// void TQGridScanner::plotAndSaveAllHistograms2D(const TString& path, double topFraction, const TString& options){
//   // obtain the 2D histograms for all variable pairs 
//   // containing the best (100*topFraction)% of points
//   // plot and save them under the given path with the given options
//   this->sortVariables();
//   for(size_t i=0; i<this->variables.size(); i++){
//     for(size_t j=i+1; j<this->variables.size(); j++){
//       this->plotAndSaveHistogram2D(getVariableName(variableOrdering[i],true),getVariableName(variableOrdering[j],true),path,topFraction,options);
//     }
//   }
// }
// void TQGridScanner::plotAndSaveAllHistograms2D(const TString& path, int topNumber, const TString& options){
//   // obtain the 2D histograms for all variable pairs 
//   // containing the best topNumber points
//   // plot and save them under the given path with the given options
//   this->sortVariables();
//   for(size_t i=0; i<this->variables.size(); i++){
//     for(size_t j=i+1; j<this->variables.size(); j++){
//       this->plotAndSaveHistogram2D(getVariableName(variableOrdering[i],true),getVariableName(variableOrdering[j],true),path,topNumber,options);
//     }
//   }
// }



TH1F* TQGridScanner::getSignificanceProfile (const TString& varname, int topNumber){
  // obtain a histogram-like plot
  // showing the significance as a function of varname
  // optimized over all invisible dimesions to the average
  // of the best topNumber points
  this->sortPoints();
  size_t index = this->findVariable(varname);
  if(index >= variables.size())
    return NULL;
  bool showExtraBins = this->getTagBoolDefault("profile.showExtraBins",false);
  bool showUnderflow = showExtraBins && (this->boundTypes[index]==LOWER);
  bool showOverflow = showExtraBins && (this->boundTypes[index]==UPPER);
  // bool showTopBin = (this->boundTypes[index] != LOWER) || !this->hasOtherVariable(index);
  //TString title = this->ndimHists[0]->GetAxis(this->getAxisIndex(ndimHists[0], varname))->GetTitle();
  //TString title = "title";
  TString title = this->getVariableTitle(index);
  TH1F* hist = new TH1F(varname, title+";"+title+";Z_{exp}",
			this->nBins[index],
			// this->scanLowerBounds[index],
			// this->scanUpperBounds[index]);
			this->axisMin[index],
			this->axisMax[index]);
  // this->getVariableTitle(varname)+";"+this->getVariableTitle(varname)+";Z_{exp}",
  // this->getVariableBinNumber(index)+(showTopBin)+(showExtraBins || this->boundTypes[index] == SPLIT),
  // this->getVariableMin(index)-(0.5+showUnderflow )*this->getVariableStepWidth(index),
  // this->getVariableMax(index)+(0.5+showOverflow -!showTopBin)*this->getVariableStepWidth(index));
  hist->SetStats(kFALSE);
  hist->SetDirectory(NULL);
  TH1F* hist_fills = (TH1F*)(hist->Clone());
  TH1F* hist_squares = (TH1F*)(hist->Clone());
  if(!hist)
    return NULL;
  for(size_t i=0; i<this->points.size(); i++){
    if(!this->points[i] || !this->isAcceptedPoint(i))
      continue;
    int nBin = hist->FindBin(points[i]->coordinates[index]+2*std::numeric_limits<double>::epsilon());
    if(nBin < 1 && showUnderflow) nBin=1;
    if(nBin > hist->GetNbinsX() && showOverflow) nBin=hist->GetNbinsX();
    if(hist_fills->GetBinContent(nBin) < topNumber){
      hist->AddBinContent(nBin,points[i]->significance);
      hist_squares->AddBinContent(nBin,pow(points[i]->significance,2));
      hist_fills->AddBinContent(nBin,1);
    }
  }
  hist->Divide(hist_fills);
  hist_squares->Divide(hist_fills);
  for(int i=0; i<hist->GetNbinsX()+2; i++){
    if(hist_fills->GetBinContent(i) > 1){
      double err = sqrt(hist_squares->GetBinContent(i) - pow(hist->GetBinContent(i),2));
      // check for nan in the case where the sum of squares and the square sum
      // are identical enough so that err will be the sqrt of a negative (yet small) number
      if(err != err) hist->SetBinError(i,0);
      // cut off exceedingly large error bars
      // relative errors on Z_exp larger than 100% are not sensible anymore
      else if(err > hist->GetBinContent(i)) hist->SetBinError(i, hist->GetBinContent(i));
      // set the error
      else hist->SetBinError(i,err);
    } else {
      hist->SetBinError(i,0);
    }
  }
  hist->SetEntries(topNumber);
  hist->SetMinimum(this->getTagDoubleDefault("profile.sigMin",4.));
  TString plotTitle = this->getTagStringDefault("profile.title");
  // if(!plotTitle.IsNull()){
	// 	plotTitle.ReplaceAll("$(VAR)",this->getVariableTitle(varname));
  //   hist->SetTitle(plotTitle);
  // }
  // if(showUnderflow)
  //   hist->GetXaxis()->SetTitle(TString(hist->GetXaxis()->GetTitle()) + " - lowest bin is underflow");
  // else if(showOverflow)
  //   hist->GetXaxis()->SetTitle(TString(hist->GetXaxis()->GetTitle()) + " - highest bin is overflow");
  delete hist_squares;
  delete hist_fills;
  return hist;
}

void TQGridScanner::plotAndSaveSignificanceProfile (const TString& varname, const TString& path, int topNumber, const TString& options){
  // obtain a histogram-like plot
  // showing the significance as a function of varname
  // optimized over all invisible dimesions to the average
  // of the best topNumber points
  // plot and save it under the given path with the given options
  TQTaggable tags(options);
  tags.importTags(this);
  bool showmax = tags.getTagBoolDefault("showmax",this->getTagBoolDefault("profile.showmax",false));
  TString extension = tags.getTagStringDefault("ext","pdf");
  TH1F* hist = this->getSignificanceProfile(varname, topNumber);
  if(!hist) return;
  TCanvas* c = new TCanvas("c", "c", 600, 600);
  gStyle->SetOptTitle(tags.getTagBoolDefault("style.showTitle",false));
  c->cd();
  TLine* cutline= NULL;
  TH1F* histmax = NULL;
  hist->SetLineColor(tags.getTagIntegerDefault("style.profile.color",this->getTagIntegerDefault("color",kBlue+4)));
  TQHistogramUtils::edge(hist,this->getTagDoubleDefault("profile.sigMin",0.));
  if(showmax && topNumber > 1){
    histmax = this->getSignificanceProfile(varname, 1);
    this->setStyle1D(c,hist,tags, histmax);
    histmax->SetLineColor(tags.getTagIntegerDefault("style.optimum.color",kAzure));
    hist->SetMaximum(0.05*(histmax->GetMaximum() - hist->GetMinimum())+histmax->GetMaximum()); 
    histmax->SetMaximum(hist->GetMaximum());
    TQHistogramUtils::edge(histmax,this->getTagDoubleDefault("profile.sigMin",0.));
    hist->Draw("");
    histmax->Draw("histsame");
    this->drawLabels(tags);
    cutline= this->drawCutLine(histmax);
  } else {
    this->setStyle1D(c,hist,tags);
    hist->Draw();
    this->drawLabels(tags);
    cutline= this->drawCutLine(hist);
  }
  if(topNumber > 1) this->drawLegend(tags,hist,TString::Format("top %d average",topNumber),histmax,cutline);
  else this->drawLegend(tags,NULL,"",hist,cutline);
  gPad->Modified(); 
  gPad->Update();
  c->SaveAs(TQFolder::concatPaths(path,"sig_"+TQFolder::makeValidIdentifier(varname))+TQFolder::makeValidIdentifier(TString::Format("_topn_%d",topNumber))+"."+extension, extension);
  delete c;
}

void TQGridScanner::deploySignificanceProfile (const TString& varname, TQFolder* f, int topNumber){
  // deploy a histogram-like plot
  // showing the significance as a function of varname
  // optimized over all invisible dimesions to the average
  // of the best topNumber points
  // in the given sample folder
  TQFolder* subf = f->getFolder(TQFolder::makeValidIdentifier(TString::Format("sig_n%d",topNumber))+"+");
  TH1F* h = this->getSignificanceProfile(varname, topNumber);
  if(h)
    subf->addObject(h, ".histograms+::sig_"+TQFolder::makeValidIdentifier(varname));
}

TList* TQGridScanner::getAllSignificanceProfiles (int topNumber){
  // obtain histogram-like plots
  // showing the significance as a function of each varname
  // optimized over all invisible dimesions to the average
  // of the best topNumber points
  TList* list = new TList();
  this->sortVariables();
  for(size_t i=0; i<this->variables.size(); i++){
    TH1F* hist = this->getSignificanceProfile(getVariableName(variableOrdering[i],true),topNumber);
    if(hist)
      list->Add(hist);
  }
  return list;
}
void TQGridScanner::plotAndSaveAllSignificanceProfiles (const TString& path, int topNumber, const TString& options){
  // obtain histogram-like plots
  // showing the significance as a function of each varname
  // optimized over all invisible dimesions to the average
  // of the best topNumber points
  // plot and save them under the given path with the given options
  gROOT->SetBatch(true);
  this->sortVariables();
  for(size_t i=0; i<this->variables.size(); i++){
    this->plotAndSaveSignificanceProfile(getVariableName(variableOrdering[i],true),path,topNumber,options);
  }
  gROOT->SetBatch(false);
}

void TQGridScanner::deployAllSignificanceProfiles (TQFolder* f, int topNumber){
  // deploy histogram-like plots
  // showing the significance as a function of each varname
  // optimized over all invisible dimesions to the average
  // of the best topNumber points
  // in the given sample folder
  this->sortVariables();
  for(size_t i=0; i<this->variables.size(); i++){
    this->deploySignificanceProfile(getVariableName(variableOrdering[i],true),f,topNumber);
  }
}



// TH2F* TQGridScanner::getSignificanceProfile2D (const TString& varname1, const TString& varname2, int topNumber){
//   // obtain a 2D histogram-like plot
//   // showing the significance as a function of varname1 and varname2
//   // optimized over all invisible dimesions to the average
//   // of the best topNumber points
//   this->sortPoints();
//   size_t index1 = this->findVariable(varname1);
//   size_t index2 = this->findVariable(varname2);
//   if(index1 >= variables.size() || index2 >= variables.size()) {
//     return NULL;
// 	}
//   bool showExtraBins = this->getTagBoolDefault("profile2D.showExtraBins",this->getTagBoolDefault("profile.showExtraBins",false));
//   bool showUnderflow1 = showExtraBins && (this->boundTypes[index1]==LOWER);
//   bool showOverflow1 = showExtraBins && (this->boundTypes[index1]==UPPER);
//   bool showUnderflow2 = showExtraBins && (this->boundTypes[index2]==LOWER);
//   bool showOverflow2 = showExtraBins && (this->boundTypes[index2]==UPPER);
//   bool showTopBin1 = (this->boundTypes[index1] != LOWER) || !this->hasOtherVariable(index1); 
//   bool showTopBin2 = (this->boundTypes[index2] != LOWER) || !this->hasOtherVariable(index2); 
//   TH2F* hist = new TH2F(varname1+"_"+varname2,
// 												varname1+" vs "+varname2+"; "+varname1+" vs "+varname2,
// 												this->nBins[index1],
// 												this->axisMin[index1],
// 												this->axisMax[index1],
// 												this->nBins[index2],
// 												this->axisMin[index2],
// 												this->axisMax[index2]);
//                         // this->getVariableTitle(varname1) + " vs. " + this->getVariableTitle(varname2)+";"+this->getVariableTitle(varname2)+";"+this->getVariableTitle(varname1),
//                         // this->getVariableBinNumber(index2)+(showTopBin2)+(showExtraBins || (this->boundTypes[index2] == SPLIT)),
//                         // this->getVariableMin(index2)-(0.5+showUnderflow2 )*this->getVariableStepWidth(index2),
//                         // this->getVariableMax(index2)+(0.5+ showOverflow2-!showTopBin2)*this->getVariableStepWidth(index2),
//                         // this->getVariableBinNumber(index1)+(showTopBin1)+(showExtraBins || (this->boundTypes[index1] == SPLIT)),
//                         // this->getVariableMin(index1)-(0.5+showUnderflow1 )*this->getVariableStepWidth(index1),
//                         // this->getVariableMax(index1)+(0.5+ showOverflow1-!showTopBin1)*this->getVariableStepWidth(index1));
//   hist->SetStats(kFALSE);
//   hist->SetDirectory(NULL);
//   hist->GetXaxis()->SetName(varname2);
//   hist->GetYaxis()->SetName(varname1);
//   TH2F* hist_fills = (TH2F*)(hist->Clone());
//   if(!hist) {
//     WARNclass(TString::Format("Failed to initialize 2d-histogram for variables %s and %s", varname1.Data(), varname2.Data()));
//     return NULL;
//   }
//   for(size_t i=0; i<this->points.size(); i++){
//     if(!this->points[i] || !this->isAcceptedPoint(i))
//       continue;
//     double val1 = this->points[i]->coordinates[index1];
//     double val2 = this->points[i]->coordinates[index2];
//     // if(showUnderflow1 && (val1 < this->gridVars[index1].min))
//     //   val1 = this->gridVars[index1].min - this->gridVars[index1].step;
//     // if( showOverflow1 && (val1 > this->gridVars[index1].max))
//     //   val1 = this->gridVars[index1].max + this->gridVars[index1].step;
//     // if(showUnderflow2 && (val2 < this->gridVars[index2].min))
//     //   val2 = this->gridVars[index2].min - this->gridVars[index2].step;
//     // if( showOverflow2 && (val2 > this->gridVars[index2].max))
//     //   val2 = this->gridVars[index2].max + this->gridVars[index2].step;
//     int nBin = hist->FindBin(val2,val1);
//     if(hist_fills->GetBinContent(nBin) < topNumber){
//       hist->AddBinContent(nBin,points[i]->significance);
//       hist_fills->AddBinContent(nBin,1);
//     }
//   }
//   hist->Divide(hist_fills);
//   TString plotTitle = this->getTagStringDefault("profile2D.title",this->getTagStringDefault("profile.title"));
//   // if(!plotTitle.IsNull()){
//   //   plotTitle.ReplaceAll("$(VARY)",this->getVariableTitle(varname1));
//   //   plotTitle.ReplaceAll("$(VARX)",this->getVariableTitle(varname2));
//   //   hist->SetTitle(plotTitle);
//   // }
//   hist->SetMinimum(this->getTagDoubleDefault("profile.sigMin",0.));
//   // if(showUnderflow1)
//   //   hist->GetYaxis()->SetTitle(TString(hist->GetYaxis()->GetTitle()) + " - lowest bin is underflow");
//   // else if(showOverflow1)
//   //   hist->GetYaxis()->SetTitle(TString(hist->GetYaxis()->GetTitle()) + " - highest bin is overflow");
//   // if(showUnderflow2)
//   //   hist->GetXaxis()->SetTitle(TString(hist->GetXaxis()->GetTitle()) + " - lowest bin is underflow");
//   // else if(showOverflow2)
//   //   hist->GetXaxis()->SetTitle(TString(hist->GetXaxis()->GetTitle()) + " - highest bin is overflow");
//   delete hist_fills;
//   return hist;
// }

// void TQGridScanner::plotAndSaveSignificanceProfile2D (const TString& varname1, const TString& varname2, const TString& path, int topNumber, const TString& options){
//   // obtain a 2D histogram-like plot
//   // showing the significance as a function of varname1 and varname2
//   // optimized over all invisible dimesions to the average
//   // of the best topNumber points
//   // plot and save it under the given path with the given options
//   TQTaggable tags(options);
//   tags.importTags(this);
//   TString extension = tags.getTagStringDefault("ext","pdf");
//   TH2F* hist = this->getSignificanceProfile2D(varname1, varname2, topNumber);
//   if(!hist) {
// 		WARNclass(TString::Format("Failed to get significance profile for variables %s and %s!", varname1.Data(), varname2.Data()));
// 		return;
// 	}
//   TQHistogramUtils::edge(hist,tags.getTagDoubleDefault("profile.sigMin",0.));
//   TCanvas* c = new TCanvas("c", "c", 600, 600);
//   gStyle->SetOptTitle(tags.getTagBoolDefault("style.showTitle",false));
//   c->cd();
//   this->setStyle2D(c,hist,tags);
//   hist->Draw("colz");
//   this->drawColAxisTitle(tags,hist,"Z_{exp}");
//   this->drawLabels(tags);
//   this->drawCutLines(hist);
//   gPad->Modified(); 
//   gPad->Update();
//   c->SaveAs(TQFolder::concatPaths(path,"sig_"+TQFolder::makeValidIdentifier(varname1+"_"+varname2))+TString::Format("_topn_%d",topNumber )+"."+extension, extension);
// 	delete c;
// }

// void TQGridScanner::deploySignificanceProfile2D (const TString& varname1, const TString& varname2, TQFolder* f, int topNumber){
//   // deploy a 2D histogram-like plot
//   // showing the significance as a function of varname1 and varname2
//   // optimized over all invisible dimesions to the average
//   // of the best topNumber points
//   // in the given sample folder
//   TQFolder* subf = f->getFolder(TQFolder::makeValidIdentifier(TString::Format("sig_n%d",topNumber))+"+");
//   TH2F* h = this->getSignificanceProfile2D(varname1, varname2, topNumber);
//   if(h)
//     subf->addObject(h, ".histograms+::sig_"+TQFolder::makeValidIdentifier(varname1+"_"+varname2));
// }

// TList* TQGridScanner::getAllSignificanceProfiles2D (int topNumber){
//   // get 2D histogram-like plots
//   // showing the significance as a function of varname1 and varname2
//   // for all variable combinations
//   // optimized over all invisible dimesions to the average
//   // of the best topNumber points
//   TList* list = new TList();
//   this->sortVariables();
//   for(size_t i=0; i<this->variables.size(); i++){
//     for(size_t j=i+1; j<this->variables.size(); j++){
//       TH2F* hist = this->getSignificanceProfile2D(getVariableName(variableOrdering[i],true),getVariableName(variableOrdering[j],true),topNumber);
//       if(hist)
//         list->Add(hist);
//     }
//   }
//   return list;
// }

// void TQGridScanner::plotAndSaveAllSignificanceProfiles2D(const TString& path, int topNumber, const TString& options){
//   // get 2D histogram-like plots
//   // showing the significance as a function of varname1 and varname2
//   // for all variable combinations
//   // optimized over all invisible dimesions to the average
//   // of the best topNumber points
//   // plot and save them under the given path with the given options
// 	gROOT->SetBatch(true);
//   this->sortVariables();
//   for(size_t i=0; i<this->variables.size(); i++){
//     for(size_t j=i+1; j<this->variables.size(); j++){
//       this->plotAndSaveSignificanceProfile2D(getVariableName(variableOrdering[i],true),getVariableName(variableOrdering[j],true),path,topNumber,options);
//     }
//   }
// 	gROOT->SetBatch(false);
// }


// void TQGridScanner::deployAllSignificanceProfiles2D (TQFolder* f, int topNumber){
//   // deploy 2D histogram-like plots
//   // showing the significance as a function of varname1 and varname2
//   // for all variable combinations
//   // optimized over all invisible dimesions to the average
//   // of the best topNumber points
//   // in the given sample folder
//   this->sortVariables();
//   for(size_t i=0; i<this->variables.size(); i++){
//     for(size_t j=i+1; j<this->variables.size(); j++){
//       this->deploySignificanceProfile2D(getVariableName(variableOrdering[i],true),getVariableName(variableOrdering[j],true),f,topNumber);
//     }
//   }
// }







TH1F* TQGridScanner::getSignificanceProfile (const TString& varname, double topFraction){
  // obtain a histogram-like plot
  // showing the significance as a function of varname
  // optimized over all invisible dimesions to the average
  // of the best (100*topFraction)% of points
  return this->getSignificanceProfile(varname, (int)ceil(topFraction*(this->points.size())));
}

void TQGridScanner::plotAndSaveSignificanceProfile (const TString& varname, const TString& path, double topFraction, const TString& options){
  // obtain a histogram-like plot
  // showing the significance as a function of varname
  // optimized over all invisible dimesions to the average
  // of the best topFraction points
  // plot and save it under the given path with the given options
  TQTaggable tags(options);
  tags.importTags(this);
  bool showmax = tags.getTagBoolDefault("showmax",this->getTagBoolDefault("profile.showmax",false));
  TString extension = tags.getTagStringDefault("ext","pdf");
  TH1F* hist = this->getSignificanceProfile(varname, topFraction);
  if(!hist) return;
  TCanvas* c = new TCanvas("c", "c", 600, 600);
  gStyle->SetOptTitle(tags.getTagBoolDefault("style.showTitle",false));
  c->cd();
  TH1F* histmax = NULL;
  TLine* cutline = NULL;
  hist->SetLineColor(tags.getTagIntegerDefault("profile.color",this->getTagIntegerDefault("color",kBlue+4)));
  TQHistogramUtils::edge(hist,this->getTagDoubleDefault("profile.sigMin",0.));
  if(showmax){
    histmax = this->getSignificanceProfile(varname, 1);
    this->setStyle1D(c,hist,tags,histmax);
    histmax->SetLineColor(tags.getTagIntegerDefault("profile.maxcolor",this->getTagIntegerDefault("maxcolor",kAzure)));
    hist->SetMaximum(0.05*(histmax->GetMaximum() - hist->GetMinimum())+histmax->GetMaximum());
    histmax->SetMaximum(hist->GetMaximum());
    TQHistogramUtils::edge(histmax,this->getTagDoubleDefault("profile.sigMin",0.));
    hist->Draw("");
    histmax->Draw("histsame");
    this->drawLabels(tags);
    cutline = this->drawCutLine(histmax);
  } else {
    this->setStyle1D(c,hist,tags);
    hist->Draw();
    this->drawLabels(tags);
    cutline = this->drawCutLine(hist);
  }
  this->drawLegend(tags,hist,TString::Format("top %g%% average",100*topFraction),histmax,cutline);
  gPad->Modified(); 
  gPad->Update();
  c->SaveAs(TQFolder::concatPaths(path,"sig_"+TQFolder::makeValidIdentifier(varname))+TQFolder::makeValidIdentifier(TString::Format("_topf_%g",topFraction))+"."+extension, extension);
  delete c;
}

void TQGridScanner::deploySignificanceProfile (const TString& varname, TQFolder* f, double topFraction){
  // deploy a histogram-like plot
  // showing the significance as a function of varname
  // optimized over all invisible dimesions to the average
  // of the best topFraction points
  // in the given sample folder
  TQFolder* subf = f->getFolder(TQFolder::makeValidIdentifier(TString::Format("sig_f%g",topFraction))+"+");
  TH1F* h = this->getSignificanceProfile(varname, topFraction);
  if(h)
    subf->addObject(h, ".histograms+::sig_"+TQFolder::makeValidIdentifier(varname));
}

TList* TQGridScanner::getAllSignificanceProfiles (double topFraction){
  // obtain histogram-like plots
  // showing the significance as a function of each varname
  // optimized over all invisible dimesions to the average
  // of the best topFraction points
  TList* list = new TList();
  this->sortVariables();
  for(size_t i=0; i<this->variables.size(); i++){
    TH1F* hist = this->getSignificanceProfile(getVariableName(variableOrdering[i],true),topFraction);
    if(hist)
      list->Add(hist);
  }
  return list;
}
void TQGridScanner::plotAndSaveAllSignificanceProfiles (const TString& path, double topFraction, const TString& options){
  // obtain histogram-like plots
  // showing the significance as a function of each varname
  // optimized over all invisible dimesions to the average
  // of the best topFraction points
  // plot and save them under the given path with the given options
	gROOT->SetBatch(true);
  this->sortVariables();
  for(size_t i=0; i<this->variables.size(); i++){
    this->plotAndSaveSignificanceProfile(getVariableName(variableOrdering[i],true),path,topFraction,options);
  }
	gROOT->SetBatch(false);
}

void TQGridScanner::deployAllSignificanceProfiles (TQFolder* f, double topFraction){
  // deploy histogram-like plots
  // showing the significance as a function of each varname
  // optimized over all invisible dimesions to the average
  // of the best topFraction points
  // in the given sample folder
  for(size_t i=0; i<this->variables.size(); i++){
    this->deploySignificanceProfile(getVariableName(variableOrdering[i],true),f,topFraction);
  }
}

// TH2F* TQGridScanner::getSignificanceProfile2D (const TString& varname1, const TString& varname2, double topFraction){
//   // obtain a 2D histogram-like plot
//   // showing the significance as a function of varname1 and varname2
//   // optimized over all invisible dimesions to the average
//   // of the best topFraction points
//   return this->getSignificanceProfile2D(varname1, varname2, (int)ceil(topFraction/(this->getVariableBinNumber(varname1)+2)/(this->getVariableBinNumber(varname2)+2)*(this->points.size())));
//   //  ^^^ please note that this cast is important
//   //  to avoid an infinite loop of-self-calls
//   //  of this function
// }

// void TQGridScanner::plotAndSaveSignificanceProfile2D (const TString& varname1, const TString& varname2, const TString& path, double topFraction, const TString& options){
//   // obtain a 2D histogram-like plot
//   // showing the significance as a function of varname1 and varname2
//   // optimized over all invisible dimesions to the average
//   // of the best topFraction points
//   // plot and save it under the given path with the given options
//   TQTaggable tags(options);
//   tags.importTags(this);
//   TString extension = tags.getTagStringDefault("ext","pdf");
//   TH2F* hist = this->getSignificanceProfile2D(varname1, varname2, topFraction);
//   if(!hist) return;
//   TQHistogramUtils::edge(hist,this->getTagDoubleDefault("profile.sigMin",0.));
//   TCanvas* c = new TCanvas("c", "c", 600, 600);
//   gStyle->SetOptTitle(tags.getTagBoolDefault("style.showTitle",false));
//   c->cd();
//   this->setStyle2D(c,hist,tags);
//   hist->Draw("colz");
//   this->drawColAxisTitle(tags,hist,"Z_{exp}");
//   this->drawLabels(tags);
//   this->drawCutLines(hist);
//   gPad->Modified(); 
//   gPad->Update();
//   c->SaveAs(TQFolder::concatPaths(path,"sig_"+TQFolder::makeValidIdentifier(varname1+"_"+varname2))+TString::Format("_topf_%g",topFraction )+"."+extension, extension);
// 	delete c;
// }

// void TQGridScanner::deploySignificanceProfile2D (const TString& varname1, const TString& varname2, TQFolder* f, double topFraction){
//   // deploy a 2D histogram-like plot
//   // showing the significance as a function of varname1 and varname2
//   // optimized over all invisible dimesions to the average
//   // of the best topFraction points
//   // in the given sample folder
//   TQFolder* subf = f->getFolder(TQFolder::makeValidIdentifier(TString::Format("sig_f%g",topFraction))+"+");
//   TH2F* h = this->getSignificanceProfile2D(varname1, varname2, topFraction);
//   if(h)
//     subf->addObject(h, ".histograms+::sig_"+TQFolder::makeValidIdentifier(varname1+"_"+varname2));
// }

// TList* TQGridScanner::getAllSignificanceProfiles2D (double topFraction){
//   // get 2D histogram-like plots
//   // showing the significance as a function of varname1 and varname2
//   // for all variable combinations
//   // optimized over all invisible dimesions to the average
//   // of the best topFraction points
//   TList* list = new TList();
//   this->sortVariables();
//   for(size_t i=0; i<this->variables.size(); i++){
//     for(size_t j=i+1; j<this->variables.size(); j++){
//       TH2F* hist = this->getSignificanceProfile2D(getVariableName(variableOrdering[i],true),getVariableName(variableOrdering[j],true),topFraction);
//       if(hist)
//         list->Add(hist);
//     }
//   }
//   return list;
// }

// void TQGridScanner::plotAndSaveAllSignificanceProfiles2D(const TString& path, double topFraction, const TString& options){
//   // get 2D histogram-like plots
//   // showing the significance as a function of varname1 and varname2
//   // for all variable combinations
//   // optimized over all invisible dimesions to the average
//   // of the best topFraction points
//   // plot and save them under the given path with the given options
// 	gROOT->SetBatch(true);
//   this->sortVariables();
//   for(size_t i=0; i<this->variables.size(); i++){
//     for(size_t j=i+1; j<this->variables.size(); j++){
//       this->plotAndSaveSignificanceProfile2D(getVariableName(variableOrdering[i],true),getVariableName(variableOrdering[j],true),path,topFraction,options);
//     }
//   }
// 	gROOT->SetBatch(false);
// }

// void TQGridScanner::deployAllSignificanceProfiles2D (TQFolder* f, double topFraction){
//   // deploy 2D histogram-like plots
//   // showing the significance as a function of varname1 and varname2
//   // for all variable combinations
//   // optimized over all invisible dimesions to the average
//   // of the best topFraction points
//   // in the given sample folder
//   this->sortVariables();
//   for(size_t i=0; i<this->variables.size(); i++){
//     for(size_t j=i+1; j<this->variables.size(); j++){
//       this->deploySignificanceProfile2D(getVariableName(variableOrdering[i],true),getVariableName(variableOrdering[j],true),f,topFraction);
//     }
//   }
// }


////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQGridScanPoint
//
// a single point in configuration space
// can be manipulated by any TQSignificanceEvaluator 
// to store all necessary information for later investigation
//
////////////////////////////////////////////////////////////////////////////////////////////////
// ClassImp(TQGridScanner::TQGridScanPoint)

// TQGridScanner::TQGridScanPoint::TQGridScanPoint():
// // default constructor for an empty TQGridScanPoint
// significance(0)
// {}

// TQGridScanner::TQGridScanPoint::TQGridScanPoint(std::vector<TString>* vars, std::vector<double>& coords, std::vector<TString> switchStatus):
//   // constructor for an empty TQGridScanPoint
//   // setting the variables and coordinates to the supplied vectors
//   variables(*vars),
//   coordinates(coords),
// 	switchStatus(switchStatus),
//   significance(0),
//   id(0)
// {}

// TQGridScanner::TQGridScanPoint::TQGridScanPoint(std::vector<TString>& vars, std::vector<double>& coords, std::vector<TString> switchStatus):
//   variables(vars),
//   coordinates(coords),
// 	switchStatus(switchStatus),
//   significance(0),
//   id(0)
// {
//   // constructor for an empty TQGridScanPoint
//   // setting the variables and coordinates to the supplied vectors
// }

// void TQGridScanner::TQGridScanPoint::clear(){
//   // delete all information on this point
//   variables.clear();
//   this->significance = 0;
//   this->id = 0;
//   coordinates.clear();
//   switchStatus.clear();
// }

// TQGridScanner::TQGridScanPoint::~TQGridScanPoint(){
//   // standard destructor
//   this->clear();
// }

// bool TQGridScanner::TQGridScanPoint::greater(const TQGridScanPoint* first, const TQGridScanPoint* second){
//   // compare two TQGridScanPoints
//   // returns true if the significance of the first one is greater
//   return (first->significance > second->significance);
// }

// bool TQGridScanner::TQGridScanPoint::smaller(const TQGridScanPoint* first, const TQGridScanPoint* second){
//   // compare two TQGridScanPoints
//   // returns true if the significance of the first one is smaller
//   return (first->significance < second->significance);
// }

