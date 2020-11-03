#include "QFramework/TQNFTop0jetEstimator.h"
#include "QFramework/TQIterator.h"
#include "QFramework/TQCounter.h"
#include "QFramework/TQUtils.h"
#include "QFramework/TQStringUtils.h"
#include "QFramework/TQLibrary.h"
#include "QFramework/TQNFChainloader.h"

ClassImp(TQNFTop0jetEstimator)

#define NaN std::numeric_limits<double>::quiet_NaN()

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQNFTop0jetEstimator
//
// (to be written)
// note: uncertainties of counters in the jet efficiency correction region are considered
// partially correlated within one sample
//
////////////////////////////////////////////////////////////////////////////////////////////////

TQNFTop0jetEstimator::TQNFTop0jetEstimator() : TQNFBase("TOP0JET"),
  fResult(NaN)
{
  // default constructor
}

TQNFTop0jetEstimator::TQNFTop0jetEstimator(TQSampleDataReader* rd) : 
  TQNFBase("TOP0JET"),
  fResult(NaN)
{
  // constructor on reader
  this->setReader(rd);
}

TQNFTop0jetEstimator::TQNFTop0jetEstimator(TQSampleFolder* sf) : 
  TQNFBase("TOP0JET"),
  fResult(NaN)
{
  // constructor on reader
  this->setSampleFolder(sf);
}

TQNFTop0jetEstimator::~TQNFTop0jetEstimator(){
  // default destructor
  this->finalize();
}

void TQNFTop0jetEstimator::setControlRegion(const TString& region){
  // set the control region
  this->fCutCR = region;
}

void TQNFTop0jetEstimator::setJetEffNumerator(const TString& region){
  // set the numerator region for calculation of the jet efficiency correction
  // (e.g. name (not expression!) of the cut to select "no additional jets")
  this->fCutJetEffNumerator = region;
}

void TQNFTop0jetEstimator::setJetEffDenominator(const TString& region){
  // set the denominator region for calculation of the jet efficiency correction
  // (e.g. name (not expression!) of the cut to select "at least one b-jet")
  this->fCutJetEffDenominator = region;
}

TString TQNFTop0jetEstimator::getControlRegion(){
  // get the control region
  return this->fCutCR;
}

TString TQNFTop0jetEstimator::getJetEffNumerator(){
  // get the numerator region for calculation of the jet efficiency correction
  return this->fCutJetEffNumerator;
}

TString TQNFTop0jetEstimator::getJetEffDenominator(){
  // get the denominator region for calculation of the jet efficiency correction
  return this->fCutJetEffDenominator;
}

TString TQNFTop0jetEstimator::getPathMC(){
  // return the (MC)-Path from which the information is retrieved
  return this->fPath;
}

TString TQNFTop0jetEstimator::getPathData(){
  // return the (MC)-Path from which the information is retrieved
  return this->fPathData;
}

void TQNFTop0jetEstimator::setPathMC(const TString& path){
  // set/change the MC path from which the information is retrieved
  this->fPath = path;
}

void TQNFTop0jetEstimator::setPathData(const TString& path){
  // set/change the data path from which the information is retrieved
  this->fPathData = path;
}


bool TQNFTop0jetEstimator::readConfiguration(TQFolder*f){
  // read a configuration from a TQFolder
  // the following tags are read and interpreted:
  // - verbosity: integer, increasing number increases verbosity
  // - cutCR: string, name of cut for the control region
  // - cutJetEffNumerator: string, name of cut for the jet efficiency correction region (nominator, e.g. selecting "no additional jets")
  // - cutJetEffDenominator: string, name of cut for the jet efficiency correction region (denominator, e.g. selecting "at least 1 b-jet")
  // - path: the path to be used for the MC sample to be estimated
  // all tags are copied to the calculator
  if(!f){
    if(verbosity > 0) messages.sendMessage(TQMessageStream::ERROR,"unable to configure from NULL folder");
    return false;
  }
  this->importTags(f);
  //@tag:[verbosity] This object tag sets the objects verbosity. Imported from argument in TQNFTop1jetEstimator::readConfiguration unless already present.
  this->getTagInteger("verbosity",this->verbosity);
  TString cutname;
  int n = 0;
  //@tag:[pathMC] This argument tag sets the path to retrieve MC counters from. This tag is obligatory!
  if(!f->getTagString("pathMC",this->fPath)){
    if(verbosity > 0) messages.sendMessage(TQMessageStream::ERROR,"unable to retrieve MC path");
    return false;
  }
  //@tag:[pathData] This argument tag sets the path to retrieve data counters from. This tag is obligatory!
  if (!f->getTagString("pathData",this->fPathData)) {
    if(verbosity > 0) messages.sendMessage(TQMessageStream::ERROR,"unable to retrieve data path");
    return false;
  }
  //@tag:[cutCR,cutJetEffNumberator,cutJetEffDenominator] These argument tags set the cut names for the different control regions used in the calculation. These tags are obligatory!
  if(f->getTagString("cutCR",cutname)){
    this->setControlRegion(cutname);
    n++;
  } else {
    if(verbosity > 0) messages.sendMessage(TQMessageStream::ERROR,"unable to retrieve cut name for cutCR");
  }
  if(f->getTagString("cutJetEffNumerator",cutname)){
    this->setJetEffNumerator(cutname);
    n++;
  } else {
    if(verbosity > 0) messages.sendMessage(TQMessageStream::ERROR,"unable to retrieve cut name for cutJetEffNumerator");
  }
  if(f->getTagString("cutJetEffDenominator",cutname)){
    this->setJetEffDenominator(cutname);
    n++;
  } else {
    if(verbosity > 0) messages.sendMessage(TQMessageStream::ERROR,"unable to retrieve cut name for cutJetEffDenominator");
  }
  if(n!=3){
    std::cout << "Only found "<<n<<" out of 3 required regions!"<< std::endl;
    messages.activeStream().flush();
    return false;
  }
  
  //read list of background paths to be subtracted from data before the actual calculation is done
  vBkgPaths.clear();
  TQFolderIterator sitr(f->getListOfFolders("samples/?"),true);
  while(sitr.hasNext()){
    TQFolder* sample = sitr.readNext();
    if(!sample) continue;
    //@tag:[name,path] These sub-folder argument tags determine the name and path of background samples which are to be subtracted from data. "name" defaults to the sample name and serves itself as a fallback for the "path" tag.
    TString name = sample->getTagStringDefault("name",sample->getName());
    TString path = sample->getTagStringDefault("path",name);
    vBkgPaths.push_back(path);
  }
  
  return true;
}

double TQNFTop0jetEstimator::getResult(int mode){
  // retrieve the Result
  switch(mode) {
    case 0:
      return this->fResult;
    case 1:
      return this->fResultXSec;
    case 2:
      return this->fResultExtrapolation;
    case 3:
      return this->fResultAlphaMC;
    case 4:
      return this->fResultAlphaData;
  }
  ERRORclass("Invalid mode '%d', returning NaN.",mode);
  return NaN;
}

void TQNFTop0jetEstimator::printResult(int mode){
  // print the result to the console
  if(TQUtils::isNum(this->getResult(mode))){
    std::cout << this->getResult(mode) << std::endl;
    return;
  }
  std::cout << "<invalid result>" << std::endl;
}

bool TQNFTop0jetEstimator::finalizeSelf(){
  if(this->cnt_mc_cr){ delete cnt_mc_cr; this->cnt_mc_cr = NULL; }
  if(this->cnt_data_cr){ delete cnt_data_cr; this->cnt_data_cr = NULL; }
  if(this->cnt_mc_numerator){ delete cnt_mc_numerator; this->cnt_mc_numerator = NULL; }
  if(this->cnt_data_numerator){ delete cnt_data_numerator; this->cnt_data_numerator = NULL; }
  if(this->cnt_mc_denominator){ delete cnt_mc_denominator; this->cnt_mc_denominator = NULL; }
  if(this->cnt_data_denominator){ delete cnt_data_denominator; this->cnt_data_denominator = NULL; }
  for (size_t i=0; i< this->vBkgCountersCR.size(); ++i) {
    if (this->vBkgCountersCR.at(i)) { delete this->vBkgCountersCR.at(i); this->vBkgCountersCR.at(i) = NULL; }
  }
  this->vBkgCountersCR.clear();
  for (size_t i=0; i< this->vBkgCountersNumerator.size(); ++i) {
    if (this->vBkgCountersNumerator.at(i)) { delete this->vBkgCountersNumerator.at(i); this->vBkgCountersNumerator.at(i) = NULL; }
  }
  this->vBkgCountersNumerator.clear();
  for (size_t i=0; i< this->vBkgCountersDenominator.size(); ++i) {
    if (this->vBkgCountersDenominator.at(i)) { delete this->vBkgCountersDenominator.at(i); this->vBkgCountersDenominator.at(i) = NULL; }
  }
  this->vBkgCountersDenominator.clear();
  return true;
}

bool TQNFTop0jetEstimator::initializeSelf(){
  // actually perform the calculation and save the results
  // return true if everything went file, false if an error occured
  if(!this->fReader || !this->fReader->getSampleFolder()){
    messages.sendMessage(TQMessageStream::ERROR,"cannot perform calculation without valid sample folder!");
    return false;
  }
  if(fPath.IsNull()){
    messages.sendMessage(TQMessageStream::ERROR,"cannot perform calculation without valid path!");
    return false;
  }
  TQTaggable tmp;
  //@tag:[readScaleScheme] This object tag determines which scale scheme is used when retrieving values entering the NF calculation (e.g. to include results from previous NF calculation steps). Default: ".top0jet.read"
  tmp.setTagString("scaleScheme",this->getTagStringDefault("readScaleScheme",".top0jet.read"));
  this->cnt_mc_cr = this->fReader->getCounter(fPath,fCutCR ,&tmp);
  this->cnt_mc_numerator = this->fReader->getCounter(fPath,fCutJetEffNumerator ,&tmp);
  this->cnt_mc_denominator = this->fReader->getCounter(fPath,fCutJetEffDenominator ,&tmp);
  this->cnt_data_cr = this->fReader->getCounter(fPathData,fCutCR ,&tmp);
  this->cnt_data_numerator = this->fReader->getCounter(fPathData,fCutJetEffNumerator ,&tmp);
  this->cnt_data_denominator = this->fReader->getCounter(fPathData,fCutJetEffDenominator ,&tmp);
  if(!(cnt_mc_cr && cnt_mc_numerator && cnt_mc_denominator && cnt_data_cr && cnt_data_numerator && cnt_data_denominator)){
    if(cnt_mc_cr) {delete cnt_mc_cr; cnt_mc_cr=NULL;}
    else if(verbosity > 0) messages.sendMessage(TQMessageStream::ERROR,"unable to obtain counter '%s' for control region from path '%s'",fCutCR.Data(),fPath.Data());
    if(cnt_mc_numerator) {delete cnt_mc_numerator; cnt_mc_numerator=NULL;}
    else if(verbosity > 0) messages.sendMessage(TQMessageStream::ERROR,"unable to obtain counter '%s' for jet efficiency numerator region from path '%s'",fCutJetEffNumerator.Data(),fPath.Data());
    if(cnt_mc_denominator) {delete cnt_mc_denominator; cnt_mc_denominator=NULL;}
    else if(verbosity > 0) messages.sendMessage(TQMessageStream::ERROR,"unable to obtain counter '%s' for jet efficiency denominator region from path '%s'",fCutJetEffDenominator.Data(),fPath.Data());
    if(cnt_data_cr) {delete cnt_data_cr; cnt_data_cr=NULL;}
    else if(verbosity > 0) messages.sendMessage(TQMessageStream::ERROR,"unable to obtain counter '%s' for control region from path '%s'",fCutCR.Data(),fPathData.Data());
    if(cnt_data_numerator) {delete cnt_data_numerator; cnt_data_numerator=NULL;}
    else if(verbosity > 0) messages.sendMessage(TQMessageStream::ERROR,"unable to obtain counter '%s' for jet efficiency numerator region from path '%s'",fCutJetEffNumerator.Data(),fPathData.Data());
    if(cnt_data_denominator) {delete cnt_data_denominator; cnt_data_denominator=NULL;}
    else if(verbosity > 0) messages.sendMessage(TQMessageStream::ERROR,"unable to obtain counter '%s' for jet efficiency denominator region from path '%s'",fCutJetEffDenominator.Data(),fPathData.Data());
    return false;
  }
  vBkgCountersCR.clear();
  vBkgCountersNumerator.clear();
  vBkgCountersDenominator.clear();
  for (uint i=0; i<vBkgPaths.size(); ++i) {
    vBkgCountersCR.push_back(this->fReader->getCounter(vBkgPaths.at(i),fCutCR ,&tmp));
    vBkgCountersNumerator.push_back(this->fReader->getCounter(vBkgPaths.at(i),fCutJetEffNumerator ,&tmp));
    vBkgCountersDenominator.push_back(this->fReader->getCounter(vBkgPaths.at(i),fCutJetEffDenominator ,&tmp));
  }
  
  
  return true;
}

bool TQNFTop0jetEstimator::calculate(){
  if (!this->chainLoader) iterationNumber = -1;
  //double zratio = 0;
  
  //OLD:
  /*
  double mc_cr = this->cnt_mc_cr->getCounter() * (iterationNumber < 0 ? 1. : this->chainLoader->getRelVariation((this->fPath+":"+this->fCutCR),this->cnt_mc_cr->getCounter() ,this->cnt_mc_cr->getError() ) );
  zratio = (iterationNumber < 0 || this->cnt_mc_denominator->getError() == 0) ? 0. : (this->chainLoader->getRelVariation((this->fPath+":"+this->fCutJetEffDenominator),this->cnt_mc_denominator->getCounter() ,this->cnt_mc_denominator->getError() ) -1) * this->cnt_mc_denominator->getCounter() / this->cnt_mc_denominator->getError() ;
  double mc_numerator = this->cnt_mc_numerator->getCounter() + zratio * this->cnt_mc_numerator->getError() ;
  double mc_denominator = this->cnt_mc_denominator->getCounter() + zratio * this->cnt_mc_denominator->getError();
  */
  
  //more correct correlation treatment: in order to get relative variations we don't pass the full count to the chain loader, but reduce the more inclusive regions by the counts in the less inclusive ones. Effectively this creates event counts from orthogonal (i.e. statistically independent) regions, which we later recombine.
  double val_num = this->cnt_mc_numerator->getCounter();
  double err2_num = this->cnt_mc_numerator->getErrorSquared(); //here, we don't have to change anything
  double val_den = this->cnt_mc_denominator->getCounter() - val_num; //remove the sub region
  double err2_den = this->cnt_mc_denominator->getErrorSquared() - err2_num; //adjust uncertainty for removal
  double val_cr = this->cnt_mc_cr->getCounter() - val_num - val_den; //as before
  double err2_cr = this->cnt_mc_cr->getErrorSquared() - err2_num - err2_den;
  //only apply variations if running toys (and chainloader present)
  val_num *= iterationNumber<0 ? 1. : this->chainLoader->getRelVariation((this->fPath+":"+this->fCutJetEffNumerator), val_num, sqrt(err2_num) );
  val_den *= iterationNumber<0 ? 1. : this->chainLoader->getRelVariation((this->fPath+":"+this->fCutJetEffDenominator), val_den, sqrt(err2_den) );
  val_cr *= iterationNumber<0 ? 1. : this->chainLoader->getRelVariation((this->fPath+":"+this->fCutCR), val_cr, sqrt(err2_cr) );
  
  //store in "final" variables for the actual calculation later
  double mc_numerator = val_num;
  double mc_denominator = val_num + val_den;
  double mc_cr = val_num + val_den + val_cr;
  
  //OLD:
  /*
  double data_cr = this->cnt_data_cr->getCounter() * ( iterationNumber < 0 ? 1. : this->chainLoader->getRelVariation((this->fPathData+":"+this->fCutCR),this->cnt_data_cr->getCounter() ,this->cnt_data_cr->getError() ) );
  zratio = (iterationNumber < 0 || this->cnt_data_denominator->getError() == 0) ? 0. : (this->chainLoader->getRelVariation((this->fPathData+":"+this->fCutJetEffDenominator),this->cnt_data_denominator->getCounter() ,this->cnt_data_denominator->getError() )-1) * this->cnt_data_denominator->getCounter() / this->cnt_data_denominator->getError() ;
  double data_numerator = this->cnt_data_numerator->getCounter() + zratio * this->cnt_data_numerator->getError() ;
  double data_denominator = this->cnt_data_denominator->getCounter() + zratio * this->cnt_data_denominator->getError();
  */
  
  //same trick for data
  val_num = this->cnt_data_numerator->getCounter();
  err2_num = this->cnt_data_numerator->getErrorSquared(); //here, we don't have to change anything
  val_den = this->cnt_data_denominator->getCounter() - val_num; //remove the sub region
  err2_den = this->cnt_data_denominator->getErrorSquared() - err2_num; //adjust uncertainty for removal
  val_cr = this->cnt_data_cr->getCounter() - val_num - val_den; //as before
  err2_cr = this->cnt_data_cr->getErrorSquared() - err2_num - err2_den;
  //only apply variations if running toys (and chainloader present)
  val_num *= iterationNumber<0 ? 1. : this->chainLoader->getRelVariation((this->fPathData+":"+this->fCutJetEffNumerator), val_num, sqrt(err2_num) );
  val_den *= iterationNumber<0 ? 1. : this->chainLoader->getRelVariation((this->fPathData+":"+this->fCutJetEffDenominator), val_den, sqrt(err2_den) );
  val_cr *= iterationNumber<0 ? 1. : this->chainLoader->getRelVariation((this->fPathData+":"+this->fCutCR), val_cr, sqrt(err2_cr) );
  
  //store in "final" variables for the actual calculation later
  double data_numerator = val_num;
  double data_denominator = val_num + val_den;
  double data_cr = val_num + val_den + val_cr;


  //subtract other backgrounds
  for (uint i=0; i<vBkgPaths.size(); ++i) {
    //OLD:
    /*
    data_cr -= vBkgCountersCR.at(i) ? vBkgCountersCR.at(i)->getCounter() * (iterationNumber < 0 ? 1. : this->chainLoader->getRelVariation((vBkgPaths.at(i)+":"+fCutCR),vBkgCountersCR.at(i)->getCounter() ,vBkgCountersCR.at(i)->getError() ) ) : 0.;
    if (!vBkgCountersNumerator.at(i) || !vBkgCountersDenominator.at(i) ) continue;
    zratio = iterationNumber < 0 || this->vBkgCountersDenominator.at(i)->getError() == 0 ? 0. : (this->chainLoader->getRelVariation((this->vBkgPaths.at(i)+":"+this->fCutJetEffDenominator),this->vBkgCountersDenominator.at(i)->getCounter() ,this->vBkgCountersDenominator.at(i)->getError() ) -1) * this->vBkgCountersDenominator.at(i)->getCounter() / this->vBkgCountersDenominator.at(i)->getError() ;
    data_numerator -= vBkgCountersNumerator.at(i) ? vBkgCountersNumerator.at(i)->getCounter() + zratio * vBkgCountersNumerator.at(i)->getError() : 0.;
    data_denominator -= vBkgCountersDenominator.at(i) ? vBkgCountersDenominator.at(i)->getCounter() + zratio * vBkgCountersDenominator.at(i)->getError()  : 0.;
    */
    
    if (! (this->vBkgCountersNumerator.at(i) && this->vBkgCountersDenominator.at(i) && this->vBkgCountersCR.at(i)) ) {
      ERRORclass("Missing counter for subtraction of process '%s' from data in at least one of the control regions",this->vBkgPaths.at(i).Data());
      throw std::runtime_error("Failed to calculate top0jet NF! Check your configuration files and input sample folder!");
    }
    //same trick for subtracted backgrounds (some counters might not be available, e.g. if no event survived the selection for these regions)
    val_num = this->vBkgCountersNumerator.at(i)->getCounter();
    err2_num = this->vBkgCountersNumerator.at(i)->getErrorSquared(); //here, we don't have to change anything
    val_den = this->vBkgCountersDenominator.at(i)->getCounter() - val_num; //remove the sub region
    err2_den = this->vBkgCountersDenominator.at(i)->getErrorSquared() - err2_num; //adjust uncertainty for removal
    val_cr = this->vBkgCountersCR.at(i)->getCounter() - val_num - val_den; //as before
    err2_cr = this->vBkgCountersCR.at(i)->getErrorSquared() - err2_num - err2_den;
    //only apply variations if running toys (and chainloader present)
    val_num *= iterationNumber<0 ? 1. : this->chainLoader->getRelVariation((this->vBkgPaths.at(i)+":"+this->fCutJetEffNumerator), val_num, sqrt(err2_num) );
    val_den *= iterationNumber<0 ? 1. : this->chainLoader->getRelVariation((this->vBkgPaths.at(i)+":"+this->fCutJetEffDenominator), val_den, sqrt(err2_den) );
    val_cr *= iterationNumber<0 ? 1. : this->chainLoader->getRelVariation((this->vBkgPaths.at(i)+":"+this->fCutCR), val_cr, sqrt(err2_cr) );
    
    //store in "final" variables for the actual calculation later. here, we need to subtract all regions from the original data count:
    data_numerator -= val_num;
    data_denominator -= (val_num + val_den);
    data_cr -= (val_num + val_den + val_cr);
    
  }
  
  this->fResult = data_cr/mc_cr * pow( data_numerator*mc_denominator/(data_denominator*mc_numerator) ,2);
  this->fResultXSec = data_cr/mc_cr;
  this->fResultExtrapolation = pow( data_numerator*mc_denominator/(data_denominator*mc_numerator) ,2);
  this->fResultAlphaMC = mc_numerator / mc_denominator;
  this->fResultAlphaData = data_numerator / data_denominator;
  messages.sendMessage(TQMessageStream::VERBOSE,"calculated Top0jetEstimator NF: %g ",this->fResult);
  return true;
}

int TQNFTop0jetEstimator::deployResult(const std::vector<TString>& startAtCutNames, const std::vector<TString>& stopAtCutNames, int doOverwrite){
  // public interface to deploy calculated NFs.
  int retval = 0;
  //set the "regular" (full=XSecCorr*extrapolation) NF (mode=0):
  retval += this->deployResultInternal(startAtCutNames, stopAtCutNames, doOverwrite, 0);
  if (this->hasTag("applyToCut.XSec") || this->hasTag("applyToCut.XSec.0")) {
    //if there's an additional tag present to deploy the XSec correction seperatelly, we do so (mode = 1):
    retval += this->deployResultInternal(this->getTagVString("applyToCut.XSec"), this->getTagVString("stopAtCut.XSec"),this->getTagBoolDefault("overwrite.XSec",doOverwrite), 1);
  }
  if (this->hasTag("applyToCut.aux") || this->hasTag("applyToCut.aux.0")) {
    //if there's an additional tag present to deploy extrapolation factor seperatelly, we do so (mode = 2,3,4):
    retval += this->deployResultInternal(this->getTagVString("applyToCut.aux"), this->getTagVString("stopAtCut.aux"),this->getTagBoolDefault("overwrite.aux",doOverwrite), 2);
    retval += this->deployResultInternal(this->getTagVString("applyToCut.aux"), this->getTagVString("stopAtCut.aux"),this->getTagBoolDefault("overwrite.aux",doOverwrite), 3);
    retval += this->deployResultInternal(this->getTagVString("applyToCut.aux"), this->getTagVString("stopAtCut.aux"),this->getTagBoolDefault("overwrite.aux",doOverwrite), 4);
  }
  return retval;
}


int TQNFTop0jetEstimator::deployResultInternal(const std::vector<TString>& startAtCutNames, const std::vector<TString>& stopAtCutNames, int doOverwrite, int mode){
  // set the NF according to the desired scale scheme
  // the NF will be deployed at the given path
  // modes: 0 = deploy full result, 1 = deploy XSec correction only (data/MC in CR), 2 = deploy extrapolation factor (JVSP)
  if(!TQUtils::isNum(this->fResult)) return false;
  bool overwrite = (doOverwrite == 1);
  // set the NF according to the desired scale scheme(s)
  //@tag:[writeScaleScheme] This object tag determines the list of scale schemes the results of the NF calculation are written to. Default: ".default"
  TString postfix =  this->getPostfix(mode);
  
  std::vector<TString> writeScaleSchemes = this->getTagVString(TString::Format("writeScaleScheme%s",postfix.Data() ) );
  //we might run on mode != 0, so we use the default (mode == 0) tag version as a fall back:
  if(writeScaleSchemes.size() < 1){
    if (mode <= 1) {
      writeScaleSchemes = this->getTagVString("writeScaleScheme");  
    } else { //if we're not deploying the standard result or the pure CR NF, we should not use the default schemes but tell the user to specify a dedicated scale scheme for the auxilary results.
      WARNclass("No output scale scheme defined for auxilary results, please fix your config (needs writeScaleScheme.aux, writeScaleScheme.aux.MC, writeScaleScheme.aux.Data)! Skipping this variant...");
      return 0;
    }
  }
  if(writeScaleSchemes.size() < 1){
    writeScaleSchemes.push_back(this->getTagStringDefault("writeScaleScheme",".default"));
  }
  
  int retval = 0;
  //@tag:[targetPath] This object tag determines the TQFolder path(s) the result of the NF calculation is written to. Multiple folders can be specified using the TQFolder path arithmetics ("path1+path2")
  TString path = this->getTagStringDefault("targetPath",this->fPath);
  std::vector<TString> targets = this->getTargetCuts(startAtCutNames,stopAtCutNames);
  for (size_t c=0; c<targets.size(); ++c) {
    TString cutName = targets.at(c);
    TQSampleFolderIterator itr(this->fReader->getListOfSampleFolders(path),true);
    while(itr.hasNext()){
      TQSampleFolder* sf = itr.readNext();
      if(!sf) continue;
      for(size_t k=0; k<writeScaleSchemes.size(); k++){
        int n = sf->setScaleFactor(writeScaleSchemes[k]+":"+cutName+(overwrite?"":"<<"), this->getResult(mode), 0.);
        //INFOclass("Set top0jet NF at '%s' to value '%.2f' (mode = %d)",TString(writeScaleSchemes[k]+":"+cutName+(overwrite?"":"<<")).Data(), this->getResult(mode), mode );
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
      //@tag:[nfListPattern] This object tag determines the format how the existence of NFs for the target paths/cuts is written to the info folder (if present). Default: ".cut.%s+"
      TQFolder * sfProcessList = this->infoFolder->getFolder(TString::Format(this->getTagStringDefault("nfListPattern",".cut.%s+").Data(),cutName.Data()));
      // get the sample folder which contains the samples for this process
      TList* sflist = this->fReader->getListOfSampleFolders(path);
      TQSampleFolder * processSampleFolder = ( sflist && sflist->GetEntries() > 0 ) ? (TQSampleFolder*)(sflist->First()) : NULL;
      if(sflist) delete sflist;
      // retrieve the correct title of the process from this folder
      // if there is no process title set, we will use the process name instead
      TString processTitle = path;
      if (processSampleFolder)
        //@tag:[processTitleKey] This object tag determines the name of the process tag used to retrieve the process title from. Default: "style.default.title".
        processSampleFolder->getTagString(this->getTagStringDefault("processTitleKey","style.default.title"), processTitle);
      // after we have aquired all necessary information, we add a new entry 
      // to the list of processes to which NFs have been applied
      sfProcessList->setTagString(TQFolder::makeValidIdentifier(processTitle),processTitle);
    }
  }
  
  return retval;
}

int TQNFTop0jetEstimator::execute(int itrNumber) {
  this->iterationNumber = itrNumber;
  fSuccess = this->calculate();
  if (fSuccess) return 0;
  return -1;
}

bool TQNFTop0jetEstimator::success () {
  return fSuccess;
}

TString TQNFTop0jetEstimator::getPostfix(int mode) {

  switch (mode) {
    case 0:
      return TString(""); 
    case 1:
      return TString(".XSec"); 
    case 2:
      return TString(".aux"); 
    case 3:
      return TString(".aux.MC"); 
    case 4:
      return TString(".aux.Data"); 
  }
  WARN("Requested postfix for unsuported mode %d",mode);
  return TString("");
}  

