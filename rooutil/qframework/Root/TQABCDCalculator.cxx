#include "QFramework/TQABCDCalculator.h"
#include "QFramework/TQIterator.h"
#include "QFramework/TQCounter.h"
#include "QFramework/TQUtils.h"
#include "QFramework/TQStringUtils.h"

//#define _DEBUG_

#include "QFramework/TQLibrary.h"
#include "QFramework/TQNFChainloader.h"


ClassImp(TQABCDCalculator)

#define NaN std::numeric_limits<double>::quiet_NaN()

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQABCDCalculator
//
// The TQABCDCalculator automates calculation of normalization factors
// for a single sample group based on the ABCD-method, i.e.
//
// NF = A<sub>data</sub>/A<sub>MC</sub>
//
// where the data-prediction is calculated as 
//
// A<sub>data</sub> = B * C/D
//
// The counter names (or cut names) for the four regions A, B, C and D can be set with
// setA/setTarget (MC)
// setB/setSource (data)
// setC/setNumerator (data)
// setD/setDenominator (data)
//
// The path of the sample group for which the correction should be calculated is set via
// setPathMC(path)
// Usually, 'path' is a path to a MC sample for which the correction should be applied.
// For a fully data-driven calculation not using the MC in question (e.g., 'bkg/sampleY').
// The TQFolder passed to the readConfiguration(TQFolder* f) method should contain a subfolder
// which itself contains a set of subfolders with "path" tags applied. These paths usually
// correspond to (MC) background samples that are subtracted from data in regions B,C and D
// before the actual ABCD calculation is performed.
//
////////////////////////////////////////////////////////////////////////////////////////////////

TQABCDCalculator::TQABCDCalculator() : TQNFBase("ABCD"),
  fResult(NaN),
  fResultErr2(NaN)
{
  // default constructor
}

TQABCDCalculator::TQABCDCalculator(TQSampleDataReader* rd) : 
  TQNFBase("ABCD"),
  fResult(NaN),
  fResultErr2(NaN)
{
  // constructor on reader
  this->setReader(rd);
}

TQABCDCalculator::TQABCDCalculator(TQSampleFolder* sf) : 
  TQNFBase("ABCD"),
  fResult(NaN),
  fResultErr2(NaN)
{
  // constructor on reader
  this->setSampleFolder(sf);
}

TQABCDCalculator::~TQABCDCalculator(){
  // default destructor
  this->finalize();
}

void TQABCDCalculator::setTarget(const TString& region){
  // set the target or "A"-region
  this->fCutTarget = region;
}

void TQABCDCalculator::setA(const TString& region){
  // set the target or "A"-region
  this->setTarget(region);
}

void TQABCDCalculator::setSource(const TString& region){
  // set the source or "B"-region
  this->fCutSource = region;
}

void TQABCDCalculator::setB(const TString& region){
  // set the source or "B"-region
  this->setSource(region);
}

void TQABCDCalculator::setNumerator(const TString& region){
  // set the numerator or "C"-region
  this->fCutNumerator = region;
}

void TQABCDCalculator::setC(const TString& region){
  // set the numerator or "C"-region
  this->setNumerator(region);
}

void TQABCDCalculator::setDenominator(const TString& region){
  // set the numerator or "D"-region
  this->fCutDenominator = region;
}

void TQABCDCalculator::setD(const TString& region){
  // set the numerator or "D"-region
  return this->setDenominator(region);
}

TString TQABCDCalculator::getTarget(){
  // get the target or "A"-region
  return this->fCutTarget;
}

TString TQABCDCalculator::getA(){
  // get the target or "A"-region
  return this->getTarget();
}

TString TQABCDCalculator::getSource(){
  // get the source or "B"-region
  return this->fCutSource;
}

TString TQABCDCalculator::getB(){
  // get the source or "B"-region
  return this->getTarget();
}

TString TQABCDCalculator::getNumerator(){
  // get the numerator or "C"-region
  return this->fCutNumerator;
}

TString TQABCDCalculator::getC(){
  // get the numerator or "C"-region
  return this->getNumerator();
}

TString TQABCDCalculator::getDenominator(){
  // get the numerator or "D"-region
  return this->fCutDenominator;
}

TString TQABCDCalculator::getD(){
  // get the numerator or "D"-region
  return this->getDenominator();
}

TString TQABCDCalculator::getPathMC(){
  // return the MC path from which the information is retrieved
  return this->fPathMC;
}

void TQABCDCalculator::setPathMC(const TString& path){
  // set/change the MC path from which the information is retrieved
  this->fPathMC = path;
}


TString TQABCDCalculator::getPathData(){
  // return the MC path from which the information is retrieved
  return this->fPathData;
}

void TQABCDCalculator::setPathData(const TString& path){
  // set/change the MC path from which the information is retrieved
  this->fPathData = path;
}

bool TQABCDCalculator::readConfiguration(TQFolder*f){
  // read a configuration from a TQFolder
  // the following tags are read and interpreted:
  // - verbosity: integer, increasing number increases verbosity
  // - cutA: string, name of counter for region A (the 'target'-region)
  // - cutB: string, name of counter for region B (the 'source'-region)
  // - cutC: string, name of counter for region C (the 'numerator'-region)
  // - cutD: string, name of counter for region D (the 'denominator'-region)
  // - pathMC: the path to be used for the MC sample to be normalized (regions used: A)
  // - pathData: the path of the data sample to be used for estimation (regions used: B,C,D)
  // all tags are copied to the calculator
  if(!f){
    if(verbosity > 0) messages.sendMessage(TQMessageStream::ERROR,"unable to configure from NULL folder");
    return false;
  }
  this->importTags(f);
  //@tag: verbosity This argument tag sets the verbosity level of the TQABCDCalculator object.
  this->getTagInteger("verbosity",this->verbosity);
  TString cutname;
  int n = 0;
  //@tag: [pathMC,pathData] This argument tag defines the sample folder paths of the MC and data samples used for the calculation.
  if(!f->getTagString("pathMC",this->fPathMC)){
    if(verbosity > 0) messages.sendMessage(TQMessageStream::ERROR,"unable to retrieve mc path");
    return false;
  }
  if(!f->getTagString("pathData",this->fPathData)){
    if(verbosity > 0) messages.sendMessage(TQMessageStream::ERROR,"unable to retrieve data path");
    return false;
  }
  //@tag: [cutA,cutB,cutC,cutD] These argument tags set the name of the cut used as the definition of the phase space regions A,B,C and D, respectively.
  if(f->getTagString("cutA",cutname)){
    this->setA(cutname);
    n++;
  } else {
    if(verbosity > 0) messages.sendMessage(TQMessageStream::ERROR,"unable to retrieve cut name for cutA");
  }
  if(f->getTagString("cutB",cutname)){
    this->setB(cutname);
    n++;
  } else {
    if(verbosity > 0) messages.sendMessage(TQMessageStream::ERROR,"unable to retrieve cut name for cutB");
  }
  if(f->getTagString("cutC",cutname)){
    this->setC(cutname);
    n++;
  } else {
    if(verbosity > 0) messages.sendMessage(TQMessageStream::ERROR,"unable to retrieve cut name for cutC");
  }
  if(f->getTagString("cutD",cutname)){
    this->setD(cutname);
    n++;
  } else {
    if(verbosity > 0) messages.sendMessage(TQMessageStream::ERROR,"unable to retrieve cut name for cutD");
  }
  
  vBkgPaths.clear();
  TQFolderIterator sitr(f->getListOfFolders("samples/?"),true);
  while(sitr.hasNext()){
    TQFolder* sample = sitr.readNext();
    if(!sample) continue;
    //@tag: [name,path] These argument tags (set on subfolders of the TQFolder passed to TQABCDCalculator::readConfiguration) define the names and paths of the MC samples which are subtracted from data in the ABCD regions. Defaults: name: name of the subfolder, path: name
    TString name = sample->getTagStringDefault("name",sample->getName());
    TString path = sample->getTagStringDefault("path",name);
    vBkgPaths.push_back(path);
  }
  
  if(n==4){
    return true;
  }
  std::cout << n << std::endl;
  messages.activeStream().flush();
  return false;
}

double TQABCDCalculator::getResult(){
  // retrieve the Result
  return this->fResult;
}

double TQABCDCalculator::getResultUncertainty(){
  // retrieve the uncertainty of the result
  return sqrt(this->fResultErr2);
}

void TQABCDCalculator::printResult(){
  // print the result to the console
  if(TQUtils::isNum(this->getResult())){
    std::cout << TQStringUtils::formatValueError(this->getResult(),this->getResultUncertainty(), "%g +/- %g") << std::endl;
    return;
  }
  std::cout << "<invalid result>" << std::endl;
}

double TQABCDCalculator::getResultVariance(){
  // retrieve the uncertainty of the result
  return this->fResultErr2;
}

bool TQABCDCalculator::finalizeSelf(){
  // perform the finalization and cleanup
  // return true if everything went file, false if an error occured
  if(this->cnt_mc_a){ delete cnt_mc_a; this->cnt_mc_a = NULL; }
  if(this->cnt_mc_b){ delete cnt_mc_b; this->cnt_mc_b = NULL; }
  if(this->cnt_mc_c){ delete cnt_mc_c; this->cnt_mc_c = NULL; }
  if(this->cnt_mc_d){ delete cnt_mc_d; this->cnt_mc_d = NULL; }
  if(this->cnt_data_b){ delete cnt_data_b; this->cnt_data_b = NULL; }
  if(this->cnt_data_c){ delete cnt_data_c; this->cnt_data_c = NULL; }
  if(this->cnt_data_d){ delete cnt_data_d; this->cnt_data_d = NULL; }
  for (uint i=0; i<vBkgCountersB.size(); ++i) {
    if (vBkgCountersB.at(i)) delete vBkgCountersB.at(i);
  }
  vBkgCountersB.clear();
  for (uint i=0; i<vBkgCountersC.size(); ++i) {
    if (vBkgCountersC.at(i)) delete vBkgCountersC.at(i);
  }
  vBkgCountersC.clear();
  for (uint i=0; i<vBkgCountersD.size(); ++i) {
    if (vBkgCountersD.at(i)) delete vBkgCountersD.at(i);
  }
  vBkgCountersD.clear();
  return true;
}

bool TQABCDCalculator::initializeSelf(){
  // perform the initialization
  // return true if everything went file, false if an error occured
  if(!this->fReader || !this->fReader->getSampleFolder()){
    messages.sendMessage(TQMessageStream::ERROR,"cannot perform calculation without valid sample folder!");
    return false;
  }
  if(fPathMC.IsNull() || fPathData.IsNull()){
    messages.sendMessage(TQMessageStream::ERROR,"cannot perform calculation without valid path!");
    return false;
  }
  TQTaggable tmp;
  //@tag: [readScaleScheme] This object tag defines the name of the scale scheme used when retrieving the input quantities. Default: ".abcd.read" .
  tmp.setTagString("scaleScheme",this->getTagStringDefault("readScaleScheme",".abcd.read"));
  //@tag: [applyNonClosureCorrection] If this object tag is set to true, an additional non-closure factor (A/B) / (C/D) (all from MC) is applied. Default: false
  this->doNonClosureCorrection = this->getTagBoolDefault("applyNonClosureCorrection",false);
  this->cnt_mc_a = this->fReader->getCounter(fPathMC,fCutTarget ,&tmp);
  this->cnt_mc_b = this->fReader->getCounter(fPathMC,fCutSource ,&tmp);
  this->cnt_mc_c = this->fReader->getCounter(fPathMC,fCutNumerator ,&tmp);
  this->cnt_mc_d = this->fReader->getCounter(fPathMC,fCutDenominator,&tmp);
  this->cnt_data_b = this->fReader->getCounter(fPathData,fCutSource ,&tmp);
  this->cnt_data_c = this->fReader->getCounter(fPathData,fCutNumerator ,&tmp);
  this->cnt_data_d = this->fReader->getCounter(fPathData,fCutDenominator,&tmp);
  //throw an error if some required counters could not be retrieved
  if(!(cnt_mc_a&&cnt_data_b&&cnt_data_c&&cnt_data_d&& ( doNonClosureCorrection ? (cnt_mc_b && cnt_mc_c && cnt_mc_d) : true   ) )){
    if(cnt_mc_a) delete cnt_mc_a;
    else if(verbosity > 0) messages.sendMessage(TQMessageStream::ERROR,"unable to obtain counter '%s' for region A from path '%s'",fCutTarget.Data(),fPathMC.Data());
    if(cnt_mc_b) delete cnt_mc_b;
    else if(verbosity > 0) messages.sendMessage(TQMessageStream::ERROR,"unable to obtain counter '%s' for region B from path '%s'",fCutSource.Data(),fPathMC.Data());
    if(cnt_mc_c) delete cnt_mc_c;
    else if(verbosity > 0) messages.sendMessage(TQMessageStream::ERROR,"unable to obtain counter '%s' for region C from path '%s'",fCutNumerator.Data(),fPathMC.Data());
    if(cnt_mc_d) delete cnt_mc_d;
    else if(verbosity > 0) messages.sendMessage(TQMessageStream::ERROR,"unable to obtain counter '%s' for region D from path '%s'",fCutDenominator.Data(),fPathMC.Data());
    if(cnt_data_b) delete cnt_data_b;
    else if(verbosity > 0) messages.sendMessage(TQMessageStream::ERROR,"unable to obtain counter '%s' for region B from path '%s'",fCutSource.Data(),fPathData.Data());
    if(cnt_data_c) delete cnt_data_c;
    else if(verbosity > 0) messages.sendMessage(TQMessageStream::ERROR,"unable to obtain counter '%s' for region C from path '%s'",fCutNumerator.Data(),fPathData.Data());
    if(cnt_data_d) delete cnt_data_d;
    else if(verbosity > 0) messages.sendMessage(TQMessageStream::ERROR,"unable to obtain counter '%s' for region D from path '%s'",fCutDenominator.Data(),fPathData.Data());
    return false;
  }
  
  for (uint i=0; i<vBkgCountersB.size(); ++i) {
    if (vBkgCountersB.at(i)) delete vBkgCountersB.at(i);
  }
  vBkgCountersB.clear();
  for (uint i=0; i<vBkgCountersC.size(); ++i) {
    if (vBkgCountersC.at(i)) delete vBkgCountersC.at(i);
  }
  vBkgCountersC.clear();
  for (uint i=0; i<vBkgCountersD.size(); ++i) {
    if (vBkgCountersD.at(i)) delete vBkgCountersD.at(i);
  }
  vBkgCountersD.clear();
  for (uint i=0; i<vBkgPaths.size(); ++i) {
    vBkgCountersB.push_back(this->fReader->getCounter(vBkgPaths.at(i),fCutSource ,&tmp));
    vBkgCountersC.push_back(this->fReader->getCounter(vBkgPaths.at(i),fCutNumerator ,&tmp));
    vBkgCountersD.push_back(this->fReader->getCounter(vBkgPaths.at(i),fCutDenominator ,&tmp));
  }
  
  return true;
}

bool TQABCDCalculator::calculate(){
  // actually perform the calculation of the ABCD estimate and save the results
  // return true if everything went file, false if an error occured
  if (!chainLoader) iterationNumber = -1;
  //copy data (B,C,D) and mc (A) counters to temporary ones 
  TQCounter* ca = new TQCounter(cnt_mc_a);
  TQCounter* cb = new TQCounter(cnt_data_b);
  TQCounter* cc = new TQCounter(cnt_data_c);
  TQCounter* cd = new TQCounter(cnt_data_d);
  DEBUGclass("Starting with counters A(MC) = %g, B(data) = %g, C(data) = %g, D(data) = %g",ca->getCounter(),cb->getCounter(),cc->getCounter(),cd->getCounter());
  if (! (iterationNumber < 0)) {
    if (verbosity > 1) messages.sendMessage(TQMessageStream::VERBOSE,"applying random variations to counters");
    ca->scale(this->chainLoader->getRelVariation((fPathMC+":"+fCutTarget),ca->getCounter(),ca->getError() ) );
    cb->scale(this->chainLoader->getRelVariation((fPathData+":"+fCutSource),cb->getCounter(),cb->getError() ) );
    cc->scale(this->chainLoader->getRelVariation((fPathData+":"+fCutNumerator),cc->getCounter(),cc->getError() ) );
    cd->scale(this->chainLoader->getRelVariation((fPathData+":"+fCutDenominator),cd->getCounter(),cd->getError() ) );
  }
  DEBUGclass("Scaled counters A(MC) = %g, B(data) = %g, C(data) = %g, D(data) = %g",ca->getCounter(),cb->getCounter(),cc->getCounter(),cd->getCounter());
  for (uint i=0; i<vBkgPaths.size(); ++i) {
      if (vBkgCountersB.at(i)) cb->subtract(vBkgCountersB.at(i), ((iterationNumber < 0) ? 1. : this->chainLoader->getRelVariation((vBkgPaths.at(i)+":"+fCutSource),vBkgCountersB.at(i)->getCounter(),vBkgCountersB.at(i)->getError() ) ) );
      else WARNclass("Missing MC background counter in region B (%s)",vBkgPaths.at(i).Data());
      if (vBkgCountersC.at(i)) cc->subtract(vBkgCountersC.at(i), ((iterationNumber < 0) ? 1. : this->chainLoader->getRelVariation((vBkgPaths.at(i)+":"+fCutNumerator),vBkgCountersC.at(i)->getCounter(),vBkgCountersC.at(i)->getError() ) ) );
      else WARNclass("Missing MC background counter in region C (%s)",vBkgPaths.at(i).Data());
      if (vBkgCountersD.at(i)) cd->subtract(vBkgCountersD.at(i), ((iterationNumber < 0) ? 1. : this->chainLoader->getRelVariation((vBkgPaths.at(i)+":"+fCutDenominator),vBkgCountersC.at(i)->getCounter(),vBkgCountersD.at(i)->getError() ) ) );
      else WARNclass("Missing MC background counter in region D (%s)",vBkgPaths.at(i).Data());
    }
  DEBUGclass("Counters after scaling and other background subtraction: A(MC) = %g, B(data-MC') = %g, C(data-MC') = %g, D(data-MC') = %g",ca->getCounter(),cb->getCounter(),cc->getCounter(),cd->getCounter());
  double a = ca->getCounter();
  double b = cb->getCounter();
  double c = cc->getCounter();
  double d = cd->getCounter();
  double sa2 = ca->getErrorSquared();
  double sb2 = cb->getErrorSquared();
  double sc2 = cc->getErrorSquared();
  double sd2 = cd->getErrorSquared();
  delete ca;
  delete cb;
  delete cc;
  delete cd;
  if(verbosity > 1){
    messages.sendMessage(TQMessageStream::VERBOSE,"retrieved mc counter for region A: %g +/- %g",a,sqrt(sa2));
    messages.sendMessage(TQMessageStream::VERBOSE,"calculated value for region B: %g +/- %g",b,sqrt(sb2));
    messages.sendMessage(TQMessageStream::VERBOSE,"calculated value for region C: %g +/- %g",c,sqrt(sc2));
    messages.sendMessage(TQMessageStream::VERBOSE,"calculated value for region D: %g +/- %g",d,sqrt(sd2));
  }
  //this is just to make the following lines a little more readable, ownership is untouched
  TQCounter* ncb = this->cnt_mc_b;
  TQCounter* ncc = this->cnt_mc_c;
  TQCounter* ncd = this->cnt_mc_d;

  double a1 = b * c/d;
  double a1err2 = sb2*pow(c/d,2) + sc2*pow(b/d,2) + sd2*pow((b*c)/(d*d),2);
  //non closure correction factor is (all mc): (A/B)/(C/D)
  this->fResult = doNonClosureCorrection ? 
     a1 / (ncb->getCounter()*chainLoader->getRelVariation(fPathMC+":"+fCutSource,ncb->getCounter(), ncb->getError())) /
    (ncc->getCounter()*chainLoader->getRelVariation(fPathMC+":"+fCutNumerator,ncc->getCounter(), ncc->getError()) ) *
    ncd->getCounter()*chainLoader->getRelVariation(fPathMC+":"+fCutDenominator,ncd->getCounter(), ncd->getError()) 
   : a1/a;
  this->fResultErr2 = doNonClosureCorrection ? 
    (pow(ncb->getCounter()*ncc->getCounter()*ncd->getCounter(),2)*a1err2 + pow(a1*ncc->getCounter()*ncd->getCounter(),2)* ncb->getErrorSquared() + pow(a1*ncb->getCounter()*ncd->getCounter(),2)* ncc->getErrorSquared() + pow(a1*ncc->getCounter()*ncb->getCounter(),2)* ncd->getErrorSquared()) / pow(ncb->getCounter()*ncc->getCounter(),4)
   : a1err2/(a*a) + sa2*pow(a1/(a*a),2);
  messages.sendMessage(TQMessageStream::VERBOSE,"calculated ABCD NF: %g +/- %g",this->fResult,sqrt(this->fResultErr2));
  return true;
}

int TQABCDCalculator::deployResult(const std::vector<TString>& startAtCutNames, const std::vector<TString>& stopAtCutNames, int doOverwrite){
  // set the NF according to the desired scale scheme
  // the NF will be deployed at the given path
  if(!TQUtils::isNum(this->fResult)) return false;
  bool overwrite = (doOverwrite == 1);
  
  /*
  bool last = false;
  for (size_t i=0; i<stopAtCutNames.size(); ++i) {
    if (TQStringUtils::matches(cutName, stopAtCutNames[i])) {
      last = true;
      break;
    }
  }
  */
  
  //@tag: [writeScaleScheme] This vector valued object tag defines the names of the scale schemes the result of the calculation is written to. Default: ".default"
  // set the NF according to the desired scale scheme(s)
  std::vector<TString> writeScaleSchemes = this->getTagVString("writeScaleScheme");
  if(writeScaleSchemes.size() < 1){
    writeScaleSchemes.push_back(this->getTagStringDefault("writeScaleScheme",".default"));
  }
  int retval = 0;
  //@tag: [targetPath] This object tag defines the sample folder path, the calculated NF is applied/written to. Default: path set via config passed to TQABCDCalculator::readConfiguration, argument tag "pathMC".
  TString path = this->getTagStringDefault("targetPath",this->fPathMC);
  //--------------------------------------------------
  std::vector<TString> targets = this->getTargetCuts(startAtCutNames,stopAtCutNames);
  for (size_t c=0; c<targets.size(); ++c) {
    TString cutName = targets.at(c);
    TQSampleFolderIterator itr(this->fReader->getListOfSampleFolders(path),true);
    while(itr.hasNext()){
      TQSampleFolder* sf = itr.readNext();
      if(!sf) continue;
      for(size_t k=0; k<writeScaleSchemes.size(); k++){
        int n = sf->setScaleFactor(writeScaleSchemes[k]+":"+cutName+(overwrite?"":"<<"), this->getResult(), this->getResultUncertainty());
        this->addNFPath(sf->getPath(),cutName,writeScaleSchemes[k]);
        if(n == 0){
          ERRORclass("unable to set scale factor for cut '%s' on path '%s' with scheme '%s'",cutName.Data(),sf->getPath().Data(),writeScaleSchemes[k].Data());
        }
        retval += n;
      }
    }
    // if the info folder is set and valid, we should keep track of all the processes that have NFs applied
    if(this->infoFolder){
      // get the folder which contains the list of processes for which we have NFs
      //@tag: [nfListPattern] This object tag specifies the pattern of the folder name (inside the "info" folder) where information about the existence of the NF is stored. Default: ".cut.%s+" (the "+" enforces creation of the folder if it is not existing yet)
      TQFolder * sfProcessList = this->infoFolder->getFolder(TString::Format(this->getTagStringDefault("nfListPattern",".cut.%s+").Data(),cutName.Data()));
      // get the sample folder which contains the samples for this process
      TList* sflist = this->fReader->getListOfSampleFolders(path);
      TQSampleFolder * processSampleFolder = ( sflist && sflist->GetEntries() > 0 ) ? (TQSampleFolder*)(sflist->First()) : NULL;
      if(sflist) delete sflist;
      // retrieve the correct title of the process from this folder
      // if there is no process title set, we will use the process name instead
      TString processTitle = path;
      if (processSampleFolder)
      //@tag: [processTitleKey] This object tag defines the name of the process tag from which the title of the normalized process is obtained. Default: "style.default.title" 
        processSampleFolder->getTagString(this->getTagStringDefault("processTitleKey","style.default.title"), processTitle);
      // after we have aquired all necessary information, we add a new entry 
      // to the list of processes to which NFs have been applied
      sfProcessList->setTagString(TQFolder::makeValidIdentifier(processTitle),processTitle);
    }
  }
  //--------------------
  
  // if no recursion was required, we can stop here
  /*
  if(stopAtCutNames.size() == 0 || !this->cutInfoFolder || last)
    return retval;
  // if stopAtCutNames is set, we need to recurse over the cut structure
  // therefore, we first need to find out how the cuts are structured
  */
  //TList * cuts = this->cutInfoFolder->getListOfFolders(TString::Format("*/%s/?",cutName.Data()));
  /*
  if(!cuts) return retval;
  TQFolderIterator cutitr(cuts,true);
  // iterate over all the cuts below the one we are investigating now
  while(cutitr.hasNext()){
    TQFolder* f = cutitr.readNext();
    if(!f) continue;
    // and deploy NFs at the corresponding cuts
    retval += this->deployResult(f->GetName(),stopAtCutNames,doOverwrite);
  }
  */
  return retval;
}

int TQABCDCalculator::execute(int itrNumber) {
	// execute the iteration with the given number
  this->iterationNumber = itrNumber;
  fSuccess = this->calculate();
  if (fSuccess) return 0;
  return -1;
}

bool TQABCDCalculator::success () {
	// return true if successful, false otherwise
  return fSuccess;
}
