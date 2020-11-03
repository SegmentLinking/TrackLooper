#include "QFramework/TQSignificanceEvaluator.h"
#include "QFramework/TQHistogramUtils.h"
#include <iostream>
#include <math.h>
#include "QFramework/TQGridScanner.h"
#include "QFramework/TQSampleFolder.h"
#include "QFramework/TQSampleDataReader.h"
#include "QFramework/TQLibrary.h"

#include "THnBase.h"

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQSignificanceEvaluator
//
// The TQSignificaneEvaluator is an abstract class that provides an interface
// for significance evaluators, that is, classes which provide functionality 
// to calculate a significance for the purpose of optimization studies
// currently (Apr/May 2013) these are primarily aimed at supporting the mechanism
// provided by the TQGridScanner -- but there is really nothing that should stop you from 
// changing these classes to support other optimization techniques
// 
// the interface is kept as general as possible to provide maximum flexibility
// thus, please read the documentation to the class members carefully before
// deciding on how to proceed implementing compatibility for different
// optimization techniques
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQSignificanceEvaluator)

TQSignificanceEvaluator::TQSignificanceEvaluator(const TString& name) : 
TNamed(name,name),
  initialization(NULL),
  reader(NULL),
  luminosity(0),
  luminosityScale(1),
  fileIdentifier(name),
  verbose(false)
{}

void TQSignificanceEvaluator::setFileIdentifier(TString s){
  // set a file identifider
  this->fileIdentifier = s;
}

TQSampleDataReader* TQSignificanceEvaluator::getReader(){
  // retrieve TQSampleDataReader object
  return this->reader;
}

bool TQSignificanceEvaluator::setRangeAxis(int /*axis*/, double /*low*/, double /*up*/) {
  return true;
}

bool TQSignificanceEvaluator::updateHists(std::vector<int> /*axesToScan*/, TQGridScanner* /*scanner*/, int /*axisToEvaluate*/) {
  return true;
}

void TQSignificanceEvaluator::setVerbose(bool v){
  // toggle verbosity
  this->verbose = v;
}

void TQSignificanceEvaluator::bookNF(const TString& path){
  // book an NF
  this->autoNFs.push_back(path);
}

void TQSignificanceEvaluator::addRegion(const TString& cutname){
  // add a region (cut)
  this->regions.push_back(cutname);
}

bool TQSignificanceEvaluator::hasNativeRegionSetHandling(){
  // returns true if this evaluator type supports region set handling
  return false;
}
bool TQSignificanceEvaluator::prepareNextRegionSet(const TString& /*suffix*/){
  // prepare the next region set
  WARNclass("prepareRegion was called - this evaluator type does not support native region handling. Something went wrong!");
  return false;
}
bool TQSignificanceEvaluator::isPrepared(){
  // returns true if the evaluator is prepared, false otherwise
  return false;
}

void TQSimpleSignificanceEvaluator::printHistogramAxis() {
  int nHists = this->signalHists.size();
  INFO(TString::Format("provided %d signal/backgroud hists with axis:", nHists));
  for (int i=0; i<nHists; i++) {
    INFO(TString::Format("histogram %d", i));
    for (int j=0; j<this->signalHists[i]->GetNdimensions(); j++) {
      INFO(TString::Format("name of axis %d: %s", j, this->signalHists[i]->GetAxis(j)->GetName()));
    }
  }
}

TQSignificanceEvaluator::TQSignificanceEvaluator(const TString& name, TQSampleFolder* sf) : 
  TNamed(name,name),
  initialization(NULL),
  reader(sf ? new TQSampleDataReader(sf) : NULL),
  luminosity(0),
  luminosityScale(1),
  fileIdentifier(name)
{
  // In this constructor variant, the significance evaluator is created
  // with a pointer to a TQSampleFolder which can be used
  // to retrieve the data necessary for significance evaluatioin
  // the mechanisms of retrievial and calculation are entirely up to the derived class
  this->getLuminosity();
}

double TQSignificanceEvaluator::getLuminosity(TString folderName, TString tagName){
  // retrieve the luminosity value from the given location (default info/luminosity)
  // within the sample folder structure (and save it internally)
  if(this->reader && this->reader->getSampleFolder() && this->reader->getSampleFolder()->getFolder(folderName))
    this->reader->getSampleFolder()->getFolder(folderName)->getTagDouble(tagName,this->luminosity);
  return this->luminosity;
}

bool TQSignificanceEvaluator::scaleLuminosity(double lumi){
  // set the luminosity scale to an arbitrary value -- this will affect the significnce!
  // this will only have an effect if TQSignificanceEvaluator::getLuminosity was called previously!
  if(this->luminosity <= 0)
    this->getLuminosity();
  if(this->luminosity > 0){
    this->luminosityScale = lumi/(this->luminosity);
    return true;
  }
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQSimpleSignificanceEvaluator
//
// The TQSimpleSignificanceCalculator is a very basic implementation of a 
// TQSignificanceEvaluator that simply calculates the signal/sqrt(background) ratio
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQSimpleSignificanceEvaluator)

TQSimpleSignificanceEvaluator::TQSimpleSignificanceEvaluator(TQSampleFolder* sf, TString signal, TString background, TString name) : TQSignificanceEvaluator(name,sf),
  signalPath(signal),
  backgroundPath(background)
{
  // in this constructor variant, the simple significance calculator takes 
  // strings pointing to the signal and background folders
  // as these are the only information needed for the calculation
}

double TQSimpleSignificanceEvaluator::getSignificance2(size_t i){
  // retrieve the square significance
  // double s = this->signalGrids[i]->value()*(this->luminosityScale);
  // double b = this->backgroundGrids[i]->value()*(this->luminosityScale);
  TH1F* sigproj = (TH1F*)this->signalHists[i]->Projection(0);
	
  double s = sigproj->Integral(-1,sigproj->GetXaxis()->GetNbins()+1);
  delete sigproj;
  TH1F* bkgproj = (TH1F*)this->backgroundHists[i]->Projection(0);
  double b = bkgproj->Integral(-1,bkgproj->GetXaxis()->GetNbins()+1);
  delete bkgproj;
  // !GridScan introduce relUncertainty
  if(b < this->getTagDoubleDefault("cutoff",0)) {// || this->backgroundGrids[i]->relUncertainty() > this->getTagDoubleDefault("relErrCutoff",1.0)){
    this->info += regions[i] + " skipped: low stats ; ";
    return 0;
  }
  double sig2 = s*s/b;
  this->info += TString::Format("%s: s=%g, b=%g, Z(%s)=%g ; ", regions[i].Data(), s, b, this->GetName(), sqrt(sig2));
  return sig2;
}

double TQSimpleSignificanceEvaluator::getSignificance(size_t i){
  // retrieve the significance
  return sqrt(this->getSignificance2(i));
}

double TQSimpleSignificanceEvaluator::evaluate(){
  // in the evaluation method, we will ask for different types of initialzation
  // and perform the corresponding steps needed to return a value
  if(this->initialization == TQGridScanner::Class()){
    // if the initialization was done with a GridScanner
    // we have pointers to grids containing the information we need
    double significance = 0;
    // TMatrix<double>* m_NF = this->calculateNFs();
    this->info = "";
    for(size_t i=0; i<regions.size(); i++){
      significance += this->getSignificance2(i);
    }
    return sqrt(significance);
  }
  // if we didn't find a valid initialization method
  // we return zero and issue an error message
  ERRORclass("cannot evaluate without valid initialization!");
  return 0;
}

bool TQSimpleSignificanceEvaluator::initialize(TQGridScanner* scanner){
  // if the initialization was done with a TQGridScanner
  // we remember this and retrieve the grids from the reader now
  // to be able to use them later on
  if(!scanner){
    ERRORclass("no TQGridScanner appointed to initialization!");
    return false;
  }
  if(!reader){
    ERRORclass("no TQSampleDataReader available!");
    return false;
  }
  this->initialization = scanner->Class();
  this->scanner = scanner;
  if(this->regions.size() < 1){
    ERRORclass("no regions set!");
    return false;
  }
  if(this->verbose) VERBOSEclass("initializing with %i regions",this->regions.size());
  for(size_t i=0; i<this->regions.size(); i++){
    THnBase* s = this->reader->getTHnBase(this->signalPath, TQFolder::concatPaths(regions[i],scanner->GetName()));
    THnBase* b = this->reader->getTHnBase(this->backgroundPath, TQFolder::concatPaths(regions[i],scanner->GetName())); 
    if(!s || !b){
      ERRORclass("unable to retrieve THnBase '%s/%s' from '%s' and/or '%s' in folder '%s'",regions[i].Data(),scanner->GetName(),this->signalPath.Data(),this->backgroundPath.Data(),this->reader->getSampleFolder()->GetName());
      return false;
    } 
    this->signalHists.push_back(s);
    this->backgroundHists.push_back(b);
    scanner->addNdimHist(s);
    scanner->addNdimHist(b);
		if(this->verbose) printHistogramAxis();
  }
  return true;
}

bool TQSimpleSignificanceEvaluator::setRangeAxis(int axis, double low, double up) {
	// restrict the range for the specified axis in
	for (unsigned int i=0; i<this->signalHists.size(); i++) {
		if (!this->signalHists[i] || !this->backgroundHists[i]) {
			WARNclass(TString::Format("cannot set range for axis %d for histograms", axis));
			return false;
		}
		this->signalHists[i]->GetAxis(axis)->SetRangeUser(low, up);
		this->backgroundHists[i]->GetAxis(axis)->SetRangeUser(low, up);
	}
	return true;
}
																								
bool TQSimpleSignificanceEvaluator::updateHists(std::vector<int> axesToScan, TQGridScanner* scanner,
						int axisToEvaluate) {
  // Project out the axis which are actually scanned for optimization.
  // This will improve the performance during the actual scan
  
  // always calculate the significance on the same axis to not risk any discrepancies when
  // calculating the significance in different distributions
  std::vector<int> axis;
  axis.push_back(axisToEvaluate);  // must be set in first position of vector!
  for (int i : axesToScan) {
    if (i != axisToEvaluate) axis.push_back(i);
  }
  axesToScan = axis;
  for (unsigned int i=0; i<signalHists.size(); i++) {
    if (!this->signalHists[i] || !this->backgroundHists[i]) {
      WARNclass("cannot update hist");
      return false;
    }
    this->signalHists[i] = this->signalHists[i]->ProjectionND(axesToScan.size(), axesToScan.data(), "E");
    this->backgroundHists[i] = this->backgroundHists[i]->ProjectionND(axesToScan.size(), axesToScan.data(), "E");
    // update histograms in scanner
    scanner->resetNdimHists();
    scanner->addNdimHist(this->signalHists[i]);
    scanner->addNdimHist(this->backgroundHists[i]);
    scanner->reconfigureVariables();
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQPoissonSignificanceEvaluator
//
// The TQPoissonSignificanceCalculator is a rather basic implementation of a 
// TQSignificanceEvaluator that calculates the Poisson significance
// Since the initialization is identical to the TQSimpleSignificanceEvaluator
// we inherit from that as a base class and only replace the name
// and the evaluation method.
// 
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQPoissonSignificanceEvaluator)

TQPoissonSignificanceEvaluator::TQPoissonSignificanceEvaluator(TQSampleFolder* sf, TString signal, TString background, TString name) : TQSimpleSignificanceEvaluator(sf,signal,background,name)
{
  // in this constructor variant, the poisson significance calculator takes 
  // strings pointing to the signal and background folders
  // as these are the only information needed for the calculation
}

double TQPoissonSignificanceEvaluator::getSignificance2(size_t i){
  // retrieve the square significance
  TH1F* sigproj = (TH1F*)this->signalHists[i]->Projection(0);
  double s = sigproj->Integral(-1,sigproj->GetXaxis()->GetNbins()+1);
  delete sigproj;
  TH1F* bkgproj = (TH1F*)this->backgroundHists[i]->Projection(0);
  double b = bkgproj->Integral(-1,bkgproj->GetXaxis()->GetNbins()+1);
  delete bkgproj;
  if(b < this->getTagDoubleDefault("cutoff",0)) {// || this->backgroundGrids[i]->relUncertainty() > this->getTagDoubleDefault("relErrCutoff",1.0)){
    this->info += regions[i] + " skipped: low stats ; ";
    return 0;
  }
  double sig = TQHistogramUtils::getPoisson(b,s);
  this->info += TString::Format("%s: s=%g, b=%g, Z(%s)=%g ; ", regions[i].Data(), s, b, this->GetName(), sig);
  return sig*sig;
}


ClassImp(TQSimpleSignificanceEvaluator2)

TQSimpleSignificanceEvaluator2::TQSimpleSignificanceEvaluator2(TQSampleFolder* sf, TString signal, TString background, TString name) : TQSimpleSignificanceEvaluator(sf,signal,background,name)
{
}

double TQSimpleSignificanceEvaluator2::getSignificance2(size_t i){
  // retrieve the square significance
  TH1F* sigproj = (TH1F*)this->signalHists[i]->Projection(0);
  double s = sigproj->Integral(-1,sigproj->GetXaxis()->GetNbins()+1);
  delete sigproj;
  TH1F* bkgproj = (TH1F*)this->backgroundHists[i]->Projection(0);
  double b = bkgproj->Integral(-1,bkgproj->GetXaxis()->GetNbins()+1);
  delete bkgproj;
  if(b < this->getTagDoubleDefault("cutoff",0)) { // || this->backgroundGrids[i]->relUncertainty() > this->getTagDoubleDefault("relErrCutoff",1.0)){
    this->info += regions[i] + " skipped: low stats ; ";
    return 0;
  }
  double sig2 = s*s/(s+b);
  this->info += TString::Format("%s: s=%g, b=%g, Z(%s)=%g ; ", regions[i].Data(), s, b, this->GetName(), sqrt(sig2));
  return sig2;
}


ClassImp(TQSimpleSignificanceEvaluator3)

TQSimpleSignificanceEvaluator3::TQSimpleSignificanceEvaluator3(TQSampleFolder* sf, TString signal, TString background, TString name) : TQSimpleSignificanceEvaluator(sf,signal,background,name)
{
  _name = name;
}

double TQSimpleSignificanceEvaluator3::getSignificance2(size_t i){
  // retrieve the square significance
  TH1F* sigproj = (TH1F*)this->signalHists[i]->Projection(0);
  double s = sigproj->Integral(-1,sigproj->GetXaxis()->GetNbins()+1);
  delete sigproj;
  TH1F* bkgproj = (TH1F*)this->backgroundHists[i]->Projection(0);
  double b = bkgproj->Integral(-1,bkgproj->GetXaxis()->GetNbins()+1);
  delete bkgproj;
  if(b < this->getTagDoubleDefault("cutoff",0)) { // || this->backgroundGrids[i]->relUncertainty() > this->getTagDoubleDefault("relErrCutoff",1.0)){
    this->info += regions[i] + " skipped: low stats ; ";
    return 0;
  }
  double sig2 = (s*s)/(b*b);
  this->info += TString::Format("%s: s=%g, b=%g, Z=%g (%s); ", regions[i].Data(), s, b, sqrt(sig2), _name.Data());
  return sig2;
}

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
// !GridScan: the following lines need to be adjusted for use with THnBase
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

// ClassImp(TQSeparationSignificanceEvaluator)

// TQSeparationSignificanceEvaluator::TQSeparationSignificanceEvaluator(TQSampleFolder* sf, TString signal, TString background, TString varName, TString name) : TQSimpleSignificanceEvaluator(sf,signal,background,name)
// {
//   _name = name;
//   _varName = varName;
// }

// double TQSeparationSignificanceEvaluator::getSignificance2(size_t i){
//   // retrieve the square significance
//   TH1F * h_s = this->signalGrids[i]->getTH1F(_varName);
//   if (h_s == NULL)
//   {
//     ERRORclass("Cannot load histogram %s for separation evaluator!",(_varName.Data()));
//     return -999;
//   }
    
//   h_s->Scale((this->luminosityScale));
//   TH1F * h_b = this->backgroundGrids[i]->getTH1F(_varName);
//   h_b->Scale((this->luminosityScale));
//   if(h_b->Integral() < this->getTagDoubleDefault("cutoff",0) || this->backgroundGrids[i]->relUncertainty() > this->getTagDoubleDefault("relErrCutoff",1.0)){
//     this->info += regions[i] + " skipped: low stats ; ";
//     return 0;
//   }
//   double sig2 = 0;
//   int nstep = h_s->GetNbinsX(); 
//   double intBin = (h_s->GetXaxis()->GetXmax() - h_s->GetXaxis()->GetXmin())/nstep; 
//   double nS = h_s->GetSumOfWeights()*intBin; 
//   double nB = h_b->GetSumOfWeights()*intBin; 
//   if (nS > 0 && nB > 0) { 
//     for (int bin=1; bin<nstep+1; bin++) { 
//       double s = h_s->GetBinContent( bin )/double(nS); 
//       double b = h_b->GetBinContent( bin )/double(nB); 
//       // separation 
//       if (s + b > 0) sig2 += 0.5*(s - b)*(s - b)/(s + b); 
//     } 
//     sig2 *= intBin; 
//   } 
//   else { 
//     this->info+= TString::Format("histograms with zero entries: %g : %g cannot compute separation", nS, nB);
//     sig2 = 0; 
//   }
//   sig2 = sig2*sig2;
//   this->info += TString::Format("%s: s=%g, b=%g, Z=%g (%s); ", regions[i].Data(), nS, nB, sqrt(sig2), _name.Data());
//   return sig2;
// }

// ClassImp(TQWeightedSignificanceEvaluator)

// TQWeightedSignificanceEvaluator::TQWeightedSignificanceEvaluator(TQSampleFolder* sf, TString signal, TString background, TString name) : TQSimpleSignificanceEvaluator(sf,signal,background,name)
// {
// }

// double TQWeightedSignificanceEvaluator::getSignificance2(size_t i){
//   // retrieve the square significance
//   double s = this->signalGrids[i]->value()*(this->luminosityScale);
//   double db = this->backgroundGrids[i]->uncertainty()*(this->luminosityScale);
//   if(db < this->getTagDoubleDefault("cutoff",0) || this->backgroundGrids[i]->relUncertainty() > this->getTagDoubleDefault("relErrCutoff",1.0)){
//     this->info += regions[i] + " skipped: low stats ; ";
//     return 0;
//   }
//   double sig = s/db;
//   this->info += TString::Format("%s: s=%g, db=%g, Z=%g ; ", regions[i].Data(), s, db, sig);
//   return sig*sig;
// }



// ClassImp(TQWeightedSignificanceEvaluator2)

// TQWeightedSignificanceEvaluator2::TQWeightedSignificanceEvaluator2(TQSampleFolder* sf, TString signal, TString background, TString name) : TQSimpleSignificanceEvaluator(sf,signal,background,name)
// {
// }

// double TQWeightedSignificanceEvaluator2::getSignificance2(size_t i){
//   // retrieve the square significance
//   double s = this->signalGrids[i]->value()*(this->luminosityScale);
//   double db = this->backgroundGrids[i]->uncertainty()*(this->luminosityScale);
//   double ds = this->signalGrids[i]->uncertainty()*(this->luminosityScale);
//   if(db < this->getTagDoubleDefault("cutoff",0) || this->backgroundGrids[i]->relUncertainty() > this->getTagDoubleDefault("relErrCutoff",1.0)){
//     this->info += regions[i] + " skipped: low stats ; ";
//     return 0;
//   }
//   double sig2 = s*s/((ds*ds)+(db*db));
//   this->info += TString::Format("%s: s=%g, ds=%g, db=%g, Z=%g ; ", regions[i].Data(), s, ds, db, sqrt(sig2));
//   return sig2;
// }

