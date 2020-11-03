#include "QFramework/TQNFTop1jetEstimator.h"
#include "QFramework/TQIterator.h"
#include "QFramework/TQCounter.h"
#include "QFramework/TQUtils.h"
#include "QFramework/TQStringUtils.h"
#include "QFramework/TQLibrary.h"
#include "QFramework/TQNFChainloader.h"

ClassImp(TQNFTop1jetEstimator)

#define NaN std::numeric_limits<double>::quiet_NaN()

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQNFTop1jetEstimator
//
// This class is intended to calculate a normalization factor for the top background in the (ggF) 
// H->WW->lvlv analysis in the 1jet channel. The implementation tries to reproduce the method used in the ATLAS 
// Higgs->WW publication from 2014/2015 (http://arxiv.org/pdf/1412.2641v1.pdf, page 30f)
// (more is still to be written here)
//
////////////////////////////////////////////////////////////////////////////////////////////////

TQNFTop1jetEstimator::TQNFTop1jetEstimator() : TQNFBase("TOP1JET"),
  fResult(NaN)
{
  // default constructor
}

TQNFTop1jetEstimator::TQNFTop1jetEstimator(TQSampleDataReader* rd) : 
  TQNFBase("TOP1JET"),
  fResult(NaN)
{
  // constructor on reader
  this->setReader(rd);
}

TQNFTop1jetEstimator::TQNFTop1jetEstimator(TQSampleFolder* sf) : 
  TQNFBase("Top1jet"),
  fResult(NaN)
{
  // constructor on reader
  this->setSampleFolder(sf);
}

TQNFTop1jetEstimator::~TQNFTop1jetEstimator(){
  // default destructor
  this->finalize();
}

void TQNFTop1jetEstimator::setRegion1j0b(const TString& region){
  // set the control region (1 jet, 0 b-jets)
  this->fCut10 = region;
}

void TQNFTop1jetEstimator::setRegion1j1b(const TString& region){
  // set the control region (1 jet, 0 b-jets)
  this->fCut11 = region;
}

void TQNFTop1jetEstimator::setRegion2j1b(const TString& region){
  // set the control region (1 jet, 0 b-jets)
  this->fCut21 = region;
}

void TQNFTop1jetEstimator::setRegion2j2b(const TString& region){
  // set the control region (1 jet, 0 b-jets)
  this->fCut22 = region;
}

TString TQNFTop1jetEstimator::getRegion1j0b(){
  // get the control region (1 jet, 0 b-jets)
  return this->fCut10;
}

TString TQNFTop1jetEstimator::getRegion1j1b(){
  // get the control region (1 jet, 1(+) b-jet(s))
  return this->fCut11;
}

TString TQNFTop1jetEstimator::getRegion2j1b(){
  // get the control region (2 jets, 1 b-jet)
  return this->fCut21;
}

TString TQNFTop1jetEstimator::getRegion2j2b(){
  // get the control region (2 jets, 2(+) b-jets)
  return this->fCut22;
}

TString TQNFTop1jetEstimator::getPathMC(){
  // return the (MC)-Path from which the information is retrieved
  return this->fPath;
}

TString TQNFTop1jetEstimator::getPathData(){
  // return the (MC)-Path from which the information is retrieved
  return this->fPathData;
}

void TQNFTop1jetEstimator::setPathMC(const TString& path){
  // set/change the MC path from which the information is retrieved
  this->fPath = path;
}

void TQNFTop1jetEstimator::setPathData(const TString& path){
  // set/change the data path from which the information is retrieved
  this->fPathData = path;
}


bool TQNFTop1jetEstimator::readConfiguration(TQFolder*f){
  // read a configuration from a TQFolder
  // the following tags are read and interpreted:
  // - verbosity: integer, increasing number increases verbosity
  // - cut1j0b: string, name of cut for the 1jet, 0 b-jets region
  // - cut1j1b: string, name of cut for the 1jet, 1 (or more) b-jet(s) region
  // - cut2j1b: string, name of cut for the 2jet, 1 b-jets region
  // - cut2j2b: string, name of cut for the 2jet, 2 (or more) b-jet(s) region
  // - path: the path to be used for the MC sample to be estimated
  // For how the different regions are used, see 
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
  //@tag:[cut1j0b,cut1j1b,cut2j1b,cut2j2b] These argument tags set the cut names for the 1jet, 0b-tag region (cut1j0b, rest analogously). These tags are obligatory!
  if(f->getTagString("cut1j0b",cutname)){
    this->setRegion1j0b(cutname);
    n++;
  } else {
    if(verbosity > 0) messages.sendMessage(TQMessageStream::ERROR,"unable to retrieve cut name for cut1j0b");
  }
  if(f->getTagString("cut1j1b",cutname)){
    this->setRegion1j1b(cutname);
    n++;
  } else {
    if(verbosity > 0) messages.sendMessage(TQMessageStream::ERROR,"unable to retrieve cut name for cut1j1b");
  }
  if(f->getTagString("cut2j1b",cutname)){
    this->setRegion2j1b(cutname);
    n++;
  } else {
    if(verbosity > 0) messages.sendMessage(TQMessageStream::ERROR,"unable to retrieve cut name for cut2j1b");
  }
  if(f->getTagString("cut2j2b",cutname)){
    this->setRegion2j2b(cutname);
    n++;
  } else {
    if(verbosity > 0) messages.sendMessage(TQMessageStream::ERROR,"unable to retrieve cut name for cut2j2b");
  }
  if(n!=4){
    std::cout << "Only found "<<n<<" out of 4 required regions!"<< std::endl;
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

double TQNFTop1jetEstimator::getResult(int mode){
  // retrieve the Result
  switch(mode) {
    case 0: 
      return this->fResult;
    case 2:
      return this->fResultExtrapolation;
    case 3:
      return this->fResultEpsilon1Jet;
    case 4:
      return this->fResultGammaMC;
    case 5:
      return this->fResultEpsilon2JetData;
  }
  ERRORclass("Invalid mode '%d', returning NaN.",mode);
  return NaN;
}

void TQNFTop1jetEstimator::printResult(int mode){
  // print the result to the console
  if(TQUtils::isNum(this->getResult(mode))){
    std::cout << this->getResult(mode) << std::endl;
    return;
  }
  std::cout << "<invalid result>" << std::endl;
}

bool TQNFTop1jetEstimator::finalizeSelf(){
  if(this->cnt_mc_10){ delete this->cnt_mc_10; this->cnt_mc_10 = NULL; }
  if(this->cnt_mc_11){ delete this->cnt_mc_11; this->cnt_mc_11 = NULL; }
  if(this->cnt_mc_21){ delete this->cnt_mc_21; this->cnt_mc_21 = NULL; }
  if(this->cnt_mc_22){ delete this->cnt_mc_22; this->cnt_mc_22 = NULL; }
  if(this->cnt_data_11){ delete this->cnt_data_11; this->cnt_data_11 = NULL; }
  if(this->cnt_data_21){ delete this->cnt_data_21; this->cnt_data_21 = NULL; }
  if(this->cnt_data_22){ delete this->cnt_data_22; this->cnt_data_22 = NULL; }
  for (size_t i=0; i< this->vBkgCounters11.size(); ++i) {
    if (this->vBkgCounters11.at(i)) { delete this->vBkgCounters11.at(i); this->vBkgCounters11.at(i) = NULL; }
  }
  this->vBkgCounters11.clear();
  for (size_t i=0; i< this->vBkgCounters21.size(); ++i) {
    if (this->vBkgCounters21.at(i)) { delete this->vBkgCounters21.at(i); this->vBkgCounters21.at(i) = NULL; }
  }
  this->vBkgCounters21.clear();
  for (size_t i=0; i< this->vBkgCounters22.size(); ++i) {
    if (this->vBkgCounters22.at(i)) { delete this->vBkgCounters22.at(i); this->vBkgCounters22.at(i) = NULL; }
  }
  this->vBkgCounters22.clear();
  return true;
}

bool TQNFTop1jetEstimator::initializeSelf(){
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
  //@tag:[readScaleScheme] This object tag determines which scale scheme is used when retrieving values entering the NF calculation (e.g. to include results from previous NF calculation steps). Default: ".top1jet.read"
  tmp.setTagString("scaleScheme",this->getTagStringDefault("readScaleScheme",".top1jet.read"));
  this->cnt_mc_10 = this->fReader->getCounter(fPath,fCut10 ,&tmp);
  this->cnt_mc_11 = this->fReader->getCounter(fPath,fCut11 ,&tmp);
  this->cnt_mc_21 = this->fReader->getCounter(fPath,fCut21 ,&tmp);
  this->cnt_mc_22 = this->fReader->getCounter(fPath,fCut22 ,&tmp);
  this->cnt_data_11 = this->fReader->getCounter(fPathData,fCut11 ,&tmp);
  this->cnt_data_21 = this->fReader->getCounter(fPathData,fCut21 ,&tmp);
  this->cnt_data_22 = this->fReader->getCounter(fPathData,fCut22 ,&tmp);
  if(!(cnt_mc_10 && cnt_mc_11 && cnt_mc_21 && cnt_mc_22 && cnt_data_11 && cnt_data_21 && cnt_data_22)){
    if(cnt_mc_10) {delete cnt_mc_10; cnt_mc_10 = NULL;}
    else if(verbosity > 0) messages.sendMessage(TQMessageStream::ERROR,"unable to obtain counter '%s' from path '%s'",fCut10.Data(),fPath.Data());
    if(cnt_mc_11) {delete cnt_mc_11; cnt_mc_11 = NULL;}
    else if(verbosity > 0) messages.sendMessage(TQMessageStream::ERROR,"unable to obtain counter '%s' from path '%s'",fCut11.Data(),fPath.Data());
    if(cnt_mc_21) {delete cnt_mc_21; cnt_mc_21 = NULL;}
    else if(verbosity > 0) messages.sendMessage(TQMessageStream::ERROR,"unable to obtain counter '%s' region from path '%s'",fCut21.Data(),fPath.Data());
    if(cnt_mc_22) {delete cnt_mc_22; cnt_mc_22 = NULL;}
    else if(verbosity > 0) messages.sendMessage(TQMessageStream::ERROR,"unable to obtain counter '%s' region from path '%s'",fCut22.Data(),fPath.Data());
    if(cnt_data_11) {delete cnt_data_11; cnt_data_11 = NULL;}
    else if(verbosity > 0) messages.sendMessage(TQMessageStream::ERROR,"unable to obtain counter '%s' from path '%s'",fCut11.Data(),fPathData.Data());
    if(cnt_data_21) {delete cnt_data_21; cnt_data_21 = NULL;}
    else if(verbosity > 0) messages.sendMessage(TQMessageStream::ERROR,"unable to obtain counter '%s' from path '%s'",fCut21.Data(),fPathData.Data());
    if(cnt_data_22) {delete cnt_data_22; cnt_data_22 = NULL;}
    else if(verbosity > 0) messages.sendMessage(TQMessageStream::ERROR,"unable to obtain counter '%s' from path '%s'",fCut22.Data(),fPathData.Data());
    return false;
  }
  this->vBkgCounters11.clear();
  this->vBkgCounters21.clear();
  this->vBkgCounters22.clear();
  for (uint i=0; i< this->vBkgPaths.size(); ++i) {
    vBkgCounters11.push_back(this->fReader->getCounter(vBkgPaths.at(i),fCut11 ,&tmp));
    vBkgCounters21.push_back(this->fReader->getCounter(vBkgPaths.at(i),fCut21 ,&tmp));
    vBkgCounters22.push_back(this->fReader->getCounter(vBkgPaths.at(i),fCut22 ,&tmp));
  }
  
  return true;
}

bool TQNFTop1jetEstimator::calculate(){
  if (!this->chainLoader) iterationNumber = -1;
  double mc_10 = this->cnt_mc_10->getCounter() ;//* (iterationNumber < 0 ? 1. : this->chainLoader->getRelVariation((this->fPath+":"+this->fCut10),this->cnt_mc_10->getCounter() ,this->cnt_mc_10->getError() ) ); //this should cancel anyways, so we don't want to blow up the uncertainty even more!
  double mc_11 = this->cnt_mc_11->getCounter() * (iterationNumber < 0 ? 1. : this->chainLoader->getRelVariation((this->fPath+":"+this->fCut11),this->cnt_mc_11->getCounter() ,this->cnt_mc_11->getError() ) );
  double mc_21 = this->cnt_mc_21->getCounter() * (iterationNumber < 0 ? 1. : this->chainLoader->getRelVariation((this->fPath+":"+this->fCut21),this->cnt_mc_21->getCounter() ,this->cnt_mc_21->getError() ) );
  double mc_22 = this->cnt_mc_22->getCounter() * (iterationNumber < 0 ? 1. : this->chainLoader->getRelVariation((this->fPath+":"+this->fCut22),this->cnt_mc_22->getCounter() ,this->cnt_mc_22->getError() ) );
  
  
  double data_11 = this->cnt_data_11->getCounter() * ( iterationNumber < 0 ? 1. : this->chainLoader->getRelVariation((this->fPathData+":"+this->fCut11),this->cnt_data_11->getCounter() ,this->cnt_data_11->getError() ) );
  double data_21 = this->cnt_data_21->getCounter() * ( iterationNumber < 0 ? 1. : this->chainLoader->getRelVariation((this->fPathData+":"+this->fCut21),this->cnt_data_21->getCounter() ,this->cnt_data_21->getError() ) );
  double data_22 = this->cnt_data_22->getCounter() * ( iterationNumber < 0 ? 1. : this->chainLoader->getRelVariation((this->fPathData+":"+this->fCut22),this->cnt_data_22->getCounter() ,this->cnt_data_22->getError() ) );
  
  
  //subtract other backgrounds
  for (uint i=0; i<vBkgPaths.size(); ++i) {
    data_11 -= vBkgCounters11.at(i) ? vBkgCounters11.at(i)->getCounter() * (iterationNumber < 0 ? 1. : this->chainLoader->getRelVariation((vBkgPaths.at(i)+":"+fCut11),vBkgCounters11.at(i)->getCounter() ,vBkgCounters11.at(i)->getError() ) ) : 0.;
    data_21 -= vBkgCounters21.at(i) ? vBkgCounters21.at(i)->getCounter() * (iterationNumber < 0 ? 1. : this->chainLoader->getRelVariation((vBkgPaths.at(i)+":"+fCut21),vBkgCounters21.at(i)->getCounter() ,vBkgCounters21.at(i)->getError() ) ) : 0.;
    data_22 -= vBkgCounters22.at(i) ? vBkgCounters22.at(i)->getCounter() * (iterationNumber < 0 ? 1. : this->chainLoader->getRelVariation((vBkgPaths.at(i)+":"+fCut22),vBkgCounters22.at(i)->getCounter() ,vBkgCounters22.at(i)->getError() ) ) : 0.;
  }
  if (data_21+data_22 == 0.) {
    messages.sendMessage(TQMessageStream::ERROR,"Cannot calculate NF, no data events in counters '%s' and '%s' from path '%s'",fCut21.Data(),fCut22.Data(),fPathData.Data());
    return false;
  }
  if (mc_10 == 0.) {
    messages.sendMessage(TQMessageStream::ERROR,"Cannot calculate NF, no MC events in counter '%s' from path '%s'",fCut10.Data(),fPathData.Data());
    return false;
  }
  if (mc_22 == 0.) {
    messages.sendMessage(TQMessageStream::ERROR,"Cannot calculate NF, no MC events in counter '%s' from path '%s'",fCut22.Data(),fPathData.Data());
    return false;
  }
  
  this->fResultEpsilon2JetData = data_22/(0.5*data_21+data_22);
  this->fResultGammaMC = (mc_11*(0.5*mc_21+mc_22)) / ((mc_10+mc_11)*mc_22); //efficiency needs to include all objects in the denominator!
  
  this->fResultEpsilon1Jet = this->fResultGammaMC * this->fResultEpsilon2JetData;
  this->fResultExtrapolation = ( 1-this->fResultEpsilon1Jet)/this->fResultEpsilon1Jet;
  this->fResult = data_11/mc_10 * this->fResultExtrapolation;
  
  messages.sendMessage(TQMessageStream::VERBOSE,"calculated Top1jetEstimator NF: %g ",this->fResult);
  return true;
}

int TQNFTop1jetEstimator::deployResult(const std::vector<TString>& startAtCutNames, const std::vector<TString>& stopAtCutNames, int doOverwrite){
  // public interface to deploy calculated NFs.
  int retval = 0;
  //set the "regular" (full=XSecCorr*extrapolation) NF (mode=0):
  retval += this->deployResultInternal(startAtCutNames, stopAtCutNames, doOverwrite, 0);
  if (this->hasTag("applyToCut.aux") || this->hasTag("applyToCut.aux.0")) {
    //if there's an additional tag present to deploy extrapolation factor seperatelly, we do so (mode = 2):
    retval += this->deployResultInternal(this->getTagVString("applyToCut.aux"), this->getTagVString("stopAtCut.aux"),this->getTagBoolDefault("overwrite.aux",doOverwrite), 2); //extrapolation factor
    retval += this->deployResultInternal(this->getTagVString("applyToCut.aux"), this->getTagVString("stopAtCut.aux"),this->getTagBoolDefault("overwrite.aux",doOverwrite), 3); //estimated b-tag eff. in 1jet region
    retval += this->deployResultInternal(this->getTagVString("applyToCut.aux"), this->getTagVString("stopAtCut.aux"),this->getTagBoolDefault("overwrite.aux",doOverwrite), 4); //2jet->1jet transfer factor (MC)
    retval += this->deployResultInternal(this->getTagVString("applyToCut.aux"), this->getTagVString("stopAtCut.aux"),this->getTagBoolDefault("overwrite.aux",doOverwrite), 5); //2jet b-tag eff. from data
  }
  return retval;
}


int TQNFTop1jetEstimator::deployResultInternal(const std::vector<TString>& startAtCutNames, const std::vector<TString>& stopAtCutNames, int doOverwrite, int mode){

  // set the NF according to the desired scale scheme
  // the NF will be deployed at the given path
  if(!TQUtils::isNum(this->fResult)) return false;
  bool overwrite = (doOverwrite == 1);
  // set the NF according to the desired scale scheme(s)
  //@tag:[writeScaleScheme] This object tag determines the list of scale schemes the results of the NF calculation are written to. Default: ".default"
  //mode==1 is not in use here (keeping the enumeration consistent with top0jet estimator)
  TString postfix = this->getPostfix(mode);
  std::vector<TString> writeScaleSchemes = this->getTagVString(TString::Format("writeScaleScheme%s",postfix.Data() ) );
  //we might run on mode != 0, so we use the default (mode == 0) tag version as a fall back:
  if(writeScaleSchemes.size() < 1){
    if (mode <= 0) {
      writeScaleSchemes = this->getTagVString("writeScaleScheme");  
    } else {
      WARNclass("No output scale scheme defined for auxilary results, please fix your config (needs writeScaleScheme.aux.extrapolation, .epsilon1Jet, .gamma, .epsilon2Jet)! Skipping this variant...");
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
      if(sflist) {delete sflist; sflist = NULL;}
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

int TQNFTop1jetEstimator::execute(int itrNumber) {
  this->iterationNumber = itrNumber;
  fSuccess = this->calculate();
  if (fSuccess) return 0;
  return -1;
}

bool TQNFTop1jetEstimator::success () {
  return fSuccess;
}

TString TQNFTop1jetEstimator::getPostfix(int mode) {

  switch (mode) {
    case 0:
      return TString(""); 
    //case 1 doesn't exist in top1jet estimation (see top0jet estimator)
    case 2:
      return TString(".aux.extrapolation"); 
    case 3:
      return TString(".aux.epsilon1Jet"); 
    case 4:
      return TString(".aux.gamma"); 
    case 5:
      return TString(".aux.epsilon2Jet");
  }
  WARN("Requested postfix for unsuported mode %d",mode);
  return TString("");
}  

