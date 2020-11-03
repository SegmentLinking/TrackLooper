#include "QFramework/TQNFCustomCalculator.h"
#include "QFramework/TQIterator.h"
#include "QFramework/TQCounter.h"
#include "QFramework/TQUtils.h"
#include "QFramework/TQStringUtils.h"
#include "QFramework/TQLibrary.h"
#include "QFramework/TQNFChainloader.h"

ClassImp(TQNFCustomCalculator)

#define NaN std::numeric_limits<double>::quiet_NaN()

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQNFCustomCalculator
//
// This class allows to easily calculate NFs using a custom formula. When creating toy
// NFs via the TQNFChainloader, all input quantities are assumed uncorrelated unless
// their path and cut are identical in which case they are fully correlated.
//
////////////////////////////////////////////////////////////////////////////////////////////////

TQNFCustomCalculator::TQNFCustomCalculator() : TQNFBase("CUSTOM"),
  fResult(NaN)
{
  // default constructor
}

TQNFCustomCalculator::TQNFCustomCalculator(TQSampleDataReader* rd) : 
  TQNFBase("CUSTOM"),
  fResult(NaN)
{
  // constructor on reader
  this->setReader(rd);
}

TQNFCustomCalculator::TQNFCustomCalculator(TQSampleFolder* sf) : 
  TQNFBase("CUSTOM"),
  fResult(NaN)
{
  // constructor on reader
  this->setSampleFolder(sf);
}

TQNFCustomCalculator::~TQNFCustomCalculator(){
  // default destructor
  this->fPaths.clear();
  this->fCuts.clear();
  this->fTypes.clear();
  this->fSubtractBkg.clear();
  this->fExpression = "";
  this->fPath = "";
  this->fPathData = "";
  delete this->fFormula;

  this->finalize();
}

TString TQNFCustomCalculator::getPathMC(){
  // return the default MC path from which the information is retrieved
  return this->fPath;
}

TString TQNFCustomCalculator::getPathData(){
  // return the default data path from which the information is retrieved
  return this->fPathData;
}

void TQNFCustomCalculator::setPathMC(const TString& path){
  // set/change the default MC path from which the information is retrieved
  this->fPath = path;
}

void TQNFCustomCalculator::setPathData(const TString& path){
  // set/change the default data path from which the information is retrieved
  this->fPathData = path;
}


bool TQNFCustomCalculator::readConfiguration(TQFolder*f){
  // read a configuration from a TQFolder
  // the following tags are read and interpreted:
  // - verbosity: integer, increasing number increases verbosity
  // - pathMC,pathData: (default) paths used to retrieve MC/data counters from
  // - expression: expression to be evaluated (via TFormula)
  // - cut.i: Name of the cut used to obtain the value of parameter i.
  // - type.i: Type of the counter used for parameter i. Possible values: "MC",
  //   "data", "dataRaw". In case of "data", background samples (as listed in 
  //    +samples{...}) are automatically subtracted.
  // - path.i: (optional) override pathMC/pathData for this parameter
  // - samples to be subtracted from data before entering the expression can 
  //    be specified in the following way:
  //   + samples {
	//     + Zjets {
	//       <path = "bkg/[ee+mm]/Zjets">
	//     }
	//     ...
	//   }
	
  if(!f){
    if(verbosity > 0) messages.sendMessage(TQMessageStream::ERROR,"unable to configure from NULL folder");
    return false;
  }
  //@tag:[verbosity] This object/argument tag sets the objects verbosity. Object tag overrides argument tag.
  this->importTags(f);
  this->getTagInteger("verbosity",this->verbosity);
  TString cutname;
  //@tag:[pathMC] This object/argument tag sets the default MC path for the NF calculation. Object tag overrides argument tag.
  if(!f->getTagString("pathMC",this->fPath)){
    if(verbosity > 0) messages.sendMessage(TQMessageStream::WARNING,"unable to retrieve MC path");//this is only a warning, a path might still be specified for each individual cut
  }
  //@tag:[pathData] This object/argument tag sets the default data path for the NF calculation. Object tag overrides argument tag.
  if (!f->getTagString("pathData",this->fPathData)) {
    if(verbosity > 0) messages.sendMessage(TQMessageStream::WARNING,"unable to retrieve data path");//this is only a warning, a path might still be specified for each individual cut
  }
  //@tag:[expression] This object/argument tag sets the expression evaluated for the NF calculation. Must be TFormula compatible, the counts obtained from the different control regions are set as arguments on the TFormula. Object tag overrides argument tag.
  if (!f->getTagString("expression",this->fExpression)) {
    if(verbosity > 0) messages.sendMessage(TQMessageStream::ERROR,"unable to retrieve expression");
    return false;
  }
  //read lists of cuts and types (paths are to be read later since they are optional; TQTaggable only adds tags to the returned vectors as long as no element with name "key.number" (number starting at 0) is missing!)
  this->fCuts.clear();
  this->fTypes.clear();
  //@tag:[cut.0,cut.1,...] These object/argument tags set the cut names for the different regions. The index corresponds to the index used in the expression (e.g. cut.0 is associated with [0]). Object tag overrides argument tag.
  if (f->getTag("cut",this->fCuts)<1) {
    if(verbosity > 0) messages.sendMessage(TQMessageStream::ERROR,"No cuts specified! (e.g. <cut.0=\"myCut\",...>)");
    return false;
  }
  //@tag:[type.0,type.1,...] These object/argument tags types of counters ("MC","data","dataRaw" (no background subtraction)) for the different regions. The index corresponds to the index used in the expression (e.g. cut.0 is associated with [0]). Unless a path is explicitly given, the corresponding default path (see "pathData", "pathMC") is used. Object tag overrides argument tag.
  if (f->getTag("type",this->fTypes)<1) {
    if(verbosity > 0) messages.sendMessage(TQMessageStream::ERROR,"No types specified! (e.g. <type.0=\"MC\",...>)");
    return false;
  }
  if (this->fCuts.size() != this->fTypes.size()) {
    if(verbosity > 0) messages.sendMessage(TQMessageStream::ERROR,"The number of cuts does not match the number of types!");
    return false;
  }
  //prepare as much as possible, since this step is only called once, while initialize() is called for every iteration/toy!
  this->fPaths.clear();
  this->fSubtractBkg.clear();
  for (size_t i=0; i<this->fTypes.size(); ++i) {
    if (TQStringUtils::equal(this->fTypes.at(i),"MC")) {
    //@tag:[path.0,path.1,...] This object/argument tag can be used to override the default TQFolder path for this one counter from. If it is not set, it will default to the values of "pathMC" or "pathData" dependent on the type (see "type.0,type.1,..."). Object tag overrides argument tag.
      this->fPaths.push_back(f->getTagStringDefault(TQStringUtils::format("path.%d",i),this->fPath));
      this->fSubtractBkg.push_back(false);
    } else if (TQStringUtils::equal(this->fTypes.at(i),"data")) {
      this->fPaths.push_back(f->getTagStringDefault(TQStringUtils::format("path.%d",i),this->fPathData));
      this->fSubtractBkg.push_back(true);
    } else if (TQStringUtils::equal(this->fTypes.at(i),"dataRaw")) {
      this->fPaths.push_back(f->getTagStringDefault(TQStringUtils::format("path.%d",i),this->fPathData));
      this->fSubtractBkg.push_back(false);
    }
  }
  
  
  //read list of background paths to be subtracted from data before the actual calculation is done
  this->vBkgPaths.clear();
  TQFolderIterator sitr(f->getListOfFolders("samples/?"),true);
  while(sitr.hasNext()){
    TQFolder* sample = sitr.readNext();
    if(!sample) continue;
    //@tag:[name] This sub-folder("samples/?") argument tag sets the name of the sample. Defaults to the name of the folder it is read from.
    TString name = sample->getTagStringDefault("name",sample->getName());
    //@tag:[path] This sub-folder("samples/?") argument tag sets the path of the sample. Defaults to the value of the "name" tag.
    TString path = sample->getTagStringDefault("path",name);
    this->vBkgPaths.push_back(path);
  }
  if (this->fFormula) delete this->fFormula;
  this->fFormula = new TFormula(TString(this->GetName())+TString("_formula"),this->fExpression);
  
  return true;
}

double TQNFCustomCalculator::getResult(){
  // retrieve the Result
  return this->fResult;
}

void TQNFCustomCalculator::printResult(){
  // print the result to the console
  if(TQUtils::isNum(this->getResult())){
    std::cout << this->getResult() << std::endl;
    return;
  }
  std::cout << "<invalid result>" << std::endl;
}

double TQNFCustomCalculator::getValue(const TString& path, const TString& cut, TQTaggable& tags, bool subtractBkg) {
  TQCounter* c = this->fReader->getCounter(path,cut ,&tags);
  if (!c) {
    messages.sendMessage(TQMessageStream::ERROR,TString::Format("Failed to retrieve counter for '%s:%s'",path.Data(),cut.Data()).Data());
    return std::numeric_limits<double>::quiet_NaN();
  }
  double val = c->getCounter() * (this->iterationNumber < 0 ? 1. : this->chainLoader->getRelVariation((path+":"+cut),c->getCounter() ,c->getError() ) );
  delete c;
  if (subtractBkg) {
    for (size_t i=0; i<this->vBkgPaths.size(); ++i) {
      val -= this->getValue(vBkgPaths.at(i),cut,tags,false);
    }
  }
  return val;
}

bool TQNFCustomCalculator::finalizeSelf(){
  this->fValues.clear();
  return true;
}

bool TQNFCustomCalculator::initializeSelf(){
  // actually perform the calculation and save the results
  // return true if everything went file, false if an error occured
  if(!this->fReader || !this->fReader->getSampleFolder()){
    messages.sendMessage(TQMessageStream::ERROR,"cannot perform calculation without valid sample folder!");
    return false;
  }


  if (!this->chainLoader) this->iterationNumber = -1;
  TQTaggable tmp;
  //@tag:[readScaleScheme] This object tag determines which scale scheme is used when retrieving values entering the NF calculation (e.g. to include results from previous NF calculation steps). Default: ".custom.read"
  tmp.setTagString("scaleScheme",this->getTagStringDefault("readScaleScheme",".custom.read"));
  this->fValues.clear();
  for (size_t i=0; i<this->fCuts.size(); ++i) {
    this->fValues.push_back(this->getValue(
      this->fPaths.at(i),
      this->fCuts.at(i),
      tmp,
      this->fSubtractBkg.at(i)
    ));
  }
  return true;
}

bool TQNFCustomCalculator::calculate(){
  if (!this->fFormula) {
    messages.sendMessage(TQMessageStream::ERROR,"Not initialized (missing TFormula object)");
    return false;
  }
  for (size_t i = 0; i< this->fValues.size(); ++i) {
    this->fFormula->SetParameter(i,this->fValues.at(i));
  }
  this->fResult = this->fFormula->Eval(0);
  messages.sendMessage(TQMessageStream::VERBOSE,"calculated custom NF: %g ",this->fResult);
  return true;
}

int TQNFCustomCalculator::deployResult(const std::vector<TString>& startAtCutNames, const std::vector<TString>& stopAtCutNames, int doOverwrite){
  // set the NF according to the desired scale scheme
  // the NF will be deployed at the given path
  if(!TQUtils::isNum(this->fResult)) return false;
  bool overwrite = (doOverwrite == 1);
  // set the NF according to the desired scale scheme(s)
  //@tag:[writeScaleScheme] This object tag determines the list of scale schemes the results of the NF calculation are written to. Default: ".default"
  std::vector<TString> writeScaleSchemes = this->getTagVString("writeScaleScheme");
  if(writeScaleSchemes.size() < 1){
    writeScaleSchemes.push_back(this->getTagStringDefault("writeScaleScheme",".default"));
  }
  /*
  bool last = false;
  for (size_t i=0; i<stopAtCutNames.size(); ++i) {
    if (TQStringUtils::matches(cutName, stopAtCutNames[i])) {
      last = true;
      break;
    }
  }
  */
  int retval = 0;
  //@tag:[targetPath] This object tag determines the TQFolder path(s) the result of the NF calculation is written to. Multiple folders can be specified using the TQFolder path arithmetics ("path1+path2")
  TString path = this->getTagStringDefault("targetPath",this->fPath);
  //-------------------------------------
  std::vector<TString> targets = this->getTargetCuts(startAtCutNames,stopAtCutNames);
  for (size_t c=0; c<targets.size(); ++c) {
    TString cutName = targets.at(c);
    TQSampleFolderIterator itr(this->fReader->getListOfSampleFolders(path),true);
    while(itr.hasNext()){
      TQSampleFolder* sf = itr.readNext();
      if(!sf) continue;
      for(size_t k=0; k<writeScaleSchemes.size(); k++){
        int n = sf->setScaleFactor(writeScaleSchemes[k]+":"+cutName+(overwrite?"":"<<"), this->getResult(), 0.);
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
  //---------------------------
  /*
  // if no recursion was required, we can stop here
  if(stopAtCutNames.size() == 0 || !this->cutInfoFolder || last)
    return retval;
  // if stopAtCutNames is set, we need to recurse over the cut structure
  // therefore, we first need to find out how the cuts are structured
  */
  //TList * cuts = this->cutInfoFolder->getListOfFolders(TString::Format("*/%s/?",cutName.Data()));
  /*if(!cuts) return retval;
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

int TQNFCustomCalculator::execute(int itrNumber) {
  this->iterationNumber = itrNumber;
  fSuccess = this->calculate();
  if (fSuccess) return 0;
  return -1;
}

bool TQNFCustomCalculator::success () {
  return fSuccess;
}
