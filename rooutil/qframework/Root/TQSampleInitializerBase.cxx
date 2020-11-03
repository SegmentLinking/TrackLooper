#include "QFramework/TQSampleInitializerBase.h"
#include "QFramework/TQSample.h"
#include "QFramework/TQUtils.h"
#include "QFramework/TQIterator.h"
#include "QFramework/TQXSecParser.h"

#include "definitions.h"

// #define _DEBUG_
#include "QFramework/TQLibrary.h"

#include "TFile.h"

#include <iostream>
#include <string>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <cstdlib>

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQSampleInitializerBase:
//
// A base class that provides some helper functions for different types of sample initializers.
// Do not instantiate this class!
// 
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQSampleInitializerBase)

TQSampleInitializerBase::TQSampleInitializerBase(){
  // default constructor
}

void TQSampleInitializerBase::reset(){
  // resets the initializer
  if(this->fPaths) delete this->fPaths;
  this->fPaths = NULL;
}

TQSampleInitializerBase::~TQSampleInitializerBase(){
  // default destructor
  if(this->fPaths) delete this->fPaths;
}


#ifdef ROOTCORE
#include <RootCore/Packages.h>
#endif

#ifdef ROOTCORE_PACKAGE_xAODRootAccess
#define ASG_RELEASE 1
namespace TQUtils {
  xAOD::TEvent* xAODMakeEvent(TFile* file);
}
#define XAOD_STANDALONE 1
#include "xAODRootAccess/TEvent.h"
#include "xAODCutFlow/CutBookkeeper.h"
#include "xAODCutFlow/CutBookkeeperContainer.h"
#include "xAODCutFlow/CutBookkeeperAuxContainer.h"
#include "xAODRootAccess/tools/TReturnCode.h"

namespace {
  bool extractCountersHelper(xAOD::TEvent& event, const char* cbkname, TQFolder* folder, double scale){
    if(!folder){
      throw std::runtime_error("TQSampleInitializerBase::extractCountersHelper called with invalid folder!");
    }
    const xAOD::CutBookkeeperContainer* constCont = NULL;
    if(event.retrieveMetaInput(constCont, cbkname).isSuccess()){
      xAOD::CutBookkeeperContainer* cont = const_cast<xAOD::CutBookkeeperContainer*>(constCont);
      if(!cont){
        return false;
      }
      int ok = 0;
      for(size_t i=0; i<cont->size(); ++i){
        const xAOD::CutBookkeeper* cbk = cont->at(i);
        if(!cbk) {
          WARNfunc("Found invalid CutBookkeeper in container.");
          continue;
        }
        xAOD::CutBookkeeper::Payload payload = cbk->payload();
        TQCounter* cnt = new TQCounter(cbk->name(),scale*payload.sumOfEventWeights,scale*sqrt(payload.sumOfEventWeightsSquared),payload.nAcceptedEvents);
        cnt->SetTitle(cbk->description().c_str());
        const TString cycleName(TString::Format("cycle%d+",cbk->cycle()));
        TQFolder* cycle = folder->getFolder(cycleName);
        if(!cycle){
          throw std::runtime_error(TString::Format("TQSampleInitializerBase::extractCountersHelper unable to create subfolder %s!",cycleName.Data()).Data());
        }
        // Adding input stream as folder to circumvent the "same cycle" bug for CutBookkeepers
        TString streamName(cbk->inputStream().c_str());
        if(streamName.IsNull()){
          streamName="UNKNOWN";
        }
        streamName.Append("+");
        TQFolder* stream = cycle->getFolder(streamName);
        if(!stream){
          throw std::runtime_error(TString::Format("TQSampleInitializerBase::extractCountersHelper unable to create subfolder %s!",streamName.Data()).Data());
        }
        stream->addObject(cnt);
        ok++;
      }
      if(ok > 0) return true;
    }
    return false;
  }
}

#endif



TQFolder* TQSampleInitializerBase::extractCounters(TFile* file, double scale){
  // extract thecounters from the CutBookkeepers of an xAOD MetaData tree and return the cutflow folder
  TQFolder* cutflow = new TQFolder(".cutflow");
  if(TQSampleInitializerBase::extractCounters(file,cutflow,scale)){
    return cutflow;
  } else {
    delete cutflow;
    return NULL;
  }
}

bool TQSampleInitializerBase::extractCountersFromSample(TQSample* sf){
  // Extract the counters from the CutBookkeepers of the input file and deposit them in a sample
  TQFolder* cutflow = sf->getFolder(".cutflow+");
  TString fileName = sf->getFilename();

  TFile* f = TFile::Open(fileName,"READ");
  if(!f) {
    throw std::runtime_error(TString::Format("something went wrong when trying to read file for sample '%s'",sf->getPath().Data()).Data());
    return false; 
  }
  if(!f->IsOpen()){
    throw std::runtime_error(TString::Format("unable to open file for sample '%s'",sf->getPath().Data()).Data());
    delete f;
    return false;
  }

  bool ok = TQSampleInitializerBase::extractCounters(f,cutflow,sf->getNormalisation());
  f->Close();
  delete f;
  return ok;
}

bool TQSampleInitializerBase::extractCounters(TFile* file, TQFolder* cutflow, double scale){
  // extract thecounters from the CutBookkeepers of an xAOD MetaData tree and deposit them in the cutflow folder
#ifndef ASG_RELEASE
  ERRORclass("can only extract counters from xAOD::CutBookkeeper in ASG Release!");
  return false;
#else
  if (!cutflow) return false;
  
  xAOD::TEvent* event = TQUtils::xAODMakeEvent(file);
  bool incompleteOK = extractCountersHelper(*event, "IncompleteCutBookkeepers", cutflow->getFolder("IncompleteCutBookkeepers+"),scale);
  bool completeOK   = extractCountersHelper(*event, "CutBookkeepers", cutflow->getFolder("CutBookkeepers+"),scale);
  if(completeOK) return true;
  if(incompleteOK){
    WARNclass("received only incomplete cutbookkeepers");
    return true;
  }
  ERRORclass("unable to read cutbookkeepers");
  return false;
#endif
}


bool TQSampleInitializerBase::getTreeInformation(TQSample* sample, const TString& filename, TString& treeName, double& sumOfWeights, int& nEntries, TString& message){
  // retrieve the tree information on the given sample
  TQLibrary::redirect_stderr("/dev/null");
  TFile* f = TFile::Open(filename,"READ");
  TQLibrary::restore_stderr();
  if(!f){
    message = TString::Format("no such file: %s",filename.Data());
    return false;
  }
  if(!f->IsOpen()){
    message = TString::Format("unable to open file: %s",filename.Data());
    delete f;
    return false;
  }
  TString treeNamePattern;
  std::vector<TString> treeNamePatterns;
  if(this->getTagString("treeName",treeNamePattern)){
    treeNamePatterns.push_back(treeNamePattern);
  }
  if(sample && sample->getTagString("~.xsp.treename",treeNamePattern)){
    treeNamePatterns.push_back(treeNamePattern);
  }
  if(!treeName.IsNull()) 
    treeNamePatterns.push_back(treeName);
  
  // Going to do a quite dirty way to avoid interference between xAOD and NTuple handling (needed for proper MetaData storing in xAODs)
  // Check if any of the treeNamePatterns is "CollectionTree", if yes set isxAOD true
  bool isxAOD = false;
  for( TString& name : treeNamePatterns ) {
    isxAOD = this->getTagBoolDefault("xAOD",TQStringUtils::equal(name,"CollectionTree"));
    if( isxAOD ) break;
  }
  // Tell the sample if it is an xAOD or not
  sample->setTagBool("isxAOD",isxAOD);

  // For now excluding md5sum calculation, it's simply too slow
  // TString md5sum = TQUtils::getMD5(filename);
  // sample->setTagString(".init.filestamp.md5sum",md5sum);
  // Just store file size as a (arguably poor) substitute
  Long64_t fileSize = f->GetSize();
  int fileSize_MB = fileSize/1e6;
  sample->setTagDouble(".init.filestamp.size",fileSize_MB);
  TString moddate = TQUtils::getModificationDate(f);
  sample->setTagString(".init.filestamp.moddate",moddate);

  TTree* t = NULL;
  // Do the next steps only if we are using NTuples as input
  if( !isxAOD ) {
    for(size_t i=0; i<treeNamePatterns.size(); i++){
      treeNamePattern = this->replaceInText(treeNamePatterns[i]);
      DEBUG("trying tree name pattern '%s'",treeNamePattern.Data());
      TList* l = TQUtils::getObjects(treeNamePattern,TTree::Class(), f);
      if(l->GetEntries() == 0){
        DEBUG("failure - no such tree");
        delete l;
      } else if(l->GetEntries() > 1){
        message = TString::Format("ambiguous tree name specification '%s', candidates are %s",treeNamePattern.Data(),TQStringUtils::concat(l,", ","''").Data());
        DEBUG("failure - multiple matches");
        delete l;
        f->Close();
        delete f;
        return false;
      } else {
        DEBUG("success");
        t = dynamic_cast<TTree*>(f->Get(l->At(0)->GetName()));
        delete l;
        break;
      }
    }
    if(!t){
      message = TString::Format("no tree matching %s in file: '%s'",TQStringUtils::concat(treeNamePatterns,"'").Data(),filename.Data());
      f->Close();
      delete f;
      return false;
    }
    treeName = t->GetName();
    nEntries = t->GetEntries();
  }
  else {
    treeName = "CollectionTree";
    nEntries = 0;
  }

  bool extractBookkeepingInformation = sample->getTagBoolDefault("~usemcweights",false);
  if(extractBookkeepingInformation){
    if(this->getTagBoolDefault("useSummarytree",false)){
      TTree* summary = dynamic_cast<TTree*>(f->Get(this->getTagStringDefault("summaryTree.name","summary")));
      if(!summary){
          message = "error retrieving summary tree with name '";
          message += this->getTagStringDefault("summaryTree.name","summary");
          message += "', using tree entry count instead";
          sumOfWeights = nEntries;
      } else {
        sumOfWeights = TQUtils::getSum(summary,this->getTagStringDefault("summaryTree.expression","initialSumW"));
        if(!TQUtils::isNum(sumOfWeights)){
          message = "unable to retrieve sum of weights from summary tree, using tree count instead";
          sumOfWeights = nEntries;
        } else {
          message = TString::Format("successfully retrieved sumOfWeights from summary tree %s : %f",summary->GetName(),sumOfWeights);
        }
      }
    } else if(this->getTagBoolDefault("useCountHistogram",false)){
      TH1* hist = dynamic_cast<TH1*>(f->Get(sample->getTagStringDefault("~.init.countHistogram.name",this->getTagStringDefault("countHistogram.name","Count"))));
      if(!hist){
        message = "error retrieving count histogram from file, using tree entry count instead";
        sumOfWeights = nEntries;
      } else {
        sumOfWeights = hist->GetBinContent(sample->getTagIntegerDefault("~.init.countHistogram.bin.offset",0) + sample->getTagIntegerDefault("~.init.countHistogram.bin",this->getTagIntegerDefault("countHistogram.bin",1)));
        message = "successfully retrieved sumOfWeights from count histogram";
      }
    } else if(sample->getTagDouble("~.xsp.sumOfWeights",sumOfWeights)){
      // nothing to do here, as the operation in the if-statement has already stored the value
      message = "successfully retrieved sumOfWeights from sample tag";
    } else if( isxAOD ){
#ifdef ASG_RELEASE
#warning "using ASG_RELEASE compilation scheme"
      // Make a TEvent object and connect it with the input file
      DEBUGclass("Trying to create TEvent from file '%s'",f->GetPath());
      xAOD::TEvent* event = TQUtils::xAODMakeEvent(f);
      
      // this is the proper way of reading the book-keeping information from the xAOD, available from ATHENA 20.X on
      double averageWeight;
      TString contname = sample->getTagStringDefault("~xAODCutBookKeeperContainer",this->getTagStringDefault("xAODCutBookKeeperContainer","StreamAOD"));
      TString bkname = sample->getTagStringDefault("~xAODCutBookKeeperName",this->getTagStringDefault("xAODCutBookKeeperName","AllExecutedEvents"));
      TString kernelname = sample->getTagStringDefault("~xAODCutBookKeeperKernel",this->getTagStringDefault("xAODCutBookKeeperKernel","HIGG3D1Kernel"));
      nEntries = TQUtils::xAODMetaDataGetNEventsInitialFromCutBookkeeper(*event,contname.Data(),bkname.Data(),kernelname.Data());    
      TString averageWeightName = this->getTagStringDefault("averageWeightName","~.xsp.averageWeight");
      if(sample->getTagDouble(averageWeightName,averageWeight)){
        DEBUGclass("averageWeight=%f",averageWeight);
        sumOfWeights = averageWeight * nEntries;
        message = "successfully used '" + averageWeightName + "' with event count from cutBookkeeper";
      } else {
        sumOfWeights = TQUtils::xAODMetaDataGetSumWeightsInitialFromCutBookkeeper(*event,contname.Data(),bkname.Data(),kernelname.Data());
        message = "successfully retrieved sumOfWeights from cutBookkeeper";
      }
      if(!TQUtils::isNum(sumOfWeights)){
        ERRORclass("no valid sumOfWeights could be retrieved from cutBookkeepers for file '%s'. If the file is expected to be a valid xAOD file it might have been corrupted.",filename.Data());
        message = "unable to retrieve xAOD MetaData information, using tree event count instead";
        sumOfWeights = nEntries;
      }
#else
#warning "using plain ROOT compilation scheme"
      message = "unable to retrieve xAOD MetaData information, using tree event count instead";
      ERRORclass("unable to retrieve xAOD MetaData information: please recompile inside an ASG release environment!");
      sumOfWeights = nEntries;
#endif
    } else {
      sumOfWeights = nEntries;
      message = "successfully retrieved sumOfWeights from tree entry count";
    }
  } else {
    message="not normalizing";
  }
  f->Close();
  delete f;
  if(message.IsNull()){
    throw std::runtime_error("passed through initialization without well-defined normalization scheme!");
  }
  return true;
}
 
void TQSampleInitializerBase::readDirectory(const TString& path, int maxdepth){
  // read a directory from some given path
  DEBUGclass("entering function");
  this->reset();
  DEBUGclass("reading directory '%s'",path.Data());
  TQFolder * directory = TQFolder::copyDirectoryStructure(path, maxdepth);
  if (!this->fPaths) this->fPaths = new TQFolder();
  this->fPaths->addFolder(directory);
  this->fPaths->SetName(TQFolder::makeValidIdentifier(path));
  DEBUGclass("leaving function");
}


bool TQSampleInitializerBase::readInputFilesList(const TString& listpath, bool verbose){
  // read a directory from some given path
  this->reset();
  // read a string containing the list of input files via file since long string causes memory error in CINT when it is read via stdin
  std::ifstream ifs(listpath.Data());
  if(!ifs.good()) return false;
  // split by ','
  std::vector<TString> fileList;
  std::stringstream ss;
  TString delims(",\n\0");
  while(ifs.good()){
    char next;
    ifs.get(next);
    if(delims.Contains(next) || !ifs.good()){
      TString s(ss.str().c_str());
      if(s.IsNull()) continue;
      if(verbose){ VERBOSEclass("adding input: %s",s.Data()); }
      fileList.push_back(s);
      ss.str("");
    } else {
      ss << next;
    }
  }
  // open input files
  if (!this->fPaths) this->fPaths = new TQFolder();
  this->setTagString("filenamePrefix","*");	
  for (auto path : fileList){
    TString name = TQFolder::getPathTail(path);
    TQFolder* fnew = nullptr;
    TQFolder* storageInstance = nullptr;
    if(path.IsNull()){
      fnew = this->fPaths->getFolder("local+"); //store the file name amongst the local files (i.e. for mounted file systems)
      if(verbose){ VERBOSEclass("adding file '%s' in './'",name.Data()); };
    } else {
      TString pathToFind = TQStringUtils::trim(path);
      TString basepath = "";
      size_t protpos = TQStringUtils::find(path,"://",0);
      if(protpos < (size_t)(path.Length())){
	      size_t protend = TQStringUtils::find(path,"/",protpos+3);
	      size_t pathstart = TQStringUtils::findFirstNotOf(path,"/",protend);
	      pathToFind = pathToFind(pathstart,pathToFind.Length()-pathstart);
	      basepath = path(0,pathstart);
	      storageInstance = this->fPaths->getFolder(TQFolder::makeValidIdentifier(basepath,"_").Append("+"));
	      storageInstance->setTagString("basepath",basepath);
      } else if (TQStringUtils::countLeading(pathToFind,"/",1) > 0) { //slightly special treatment for absolute paths as storing a leading "/" in a TQFolder path is not possible
        basepath = TQFolder::getPathHead(pathToFind);
        storageInstance = this->fPaths->getFolder(basepath+"+");
        basepath = "/"+basepath+"/"; //make sure the physical path will be correctly reconstructed (TQFolder::concatPaths will simply ignore a single "/" as one of it's arguments!)
        storageInstance->setTagString("basepath",basepath);
      }
      
      if (!storageInstance) storageInstance = this->fPaths->getFolder("local+"); //if no explicit storage instance is set, we use the default (for files on a local (mounted) file system)
      fnew = storageInstance->getFolder(pathToFind+"+!");      
      if(verbose){ VERBOSEclass("adding file '%s' in '%s'",name.Data(),TQFolder::concatPaths(basepath,pathToFind).Data()); };
    }    
    if(fnew) {
      fnew->addObject(new TObjString(name));
      if(verbose){ VERBOSEclass("succesfully added '%s'/'%s' in %s",path.Data(),name.Data(),fnew->getPath().Data()); };
    }
  }//end of loop over input files
  return true;
}


TQFolder* TQSampleInitializerBase::getDirectory(){
  // retrieve the directory of this sample initializer
  // NOTE: the folder is owned by the sample initializer - do not delete!
  return this->fPaths;
}

void TQSampleInitializerBase::printDirectory(const TString& opts){
  // print the currently active directory for sample localization
  if(this->fPaths){
    this->fPaths->print(opts);
  } else {
    std::cout << TQStringUtils::makeBoldWhite("< no active directory, read with TQSampleInitializerBase::readDirectory(\"...\") >") << std::endl;
  }
}

bool TQSampleInitializerBase::setSampleNormalization(TQSample* sample, double samplefraction){
  // set the sample normalization from the tags already present on the sample
  // if a samplefraction is given, it will be applied as a multiplicative factor (default is unity)
  // The idea behind the normalization is to multiply the initial unskimmed number of events in analysed
  // sample (usually provided by some book-keeping tools through the different skimming steps) by the
  // SumOfWeights/#events computed on the full unskimmed sample of events 
  // (see https://its.cern.ch/jira/secure/attachment/52844/OArnaez_SumOfWeights.pdf)
  bool success = true;
  //@tag:[ignoreSampleNormalization] Determines if a (global) normalization should be applied for the sample (based on cross section, luminosity, k-factor,...) or ignored. Intended to explicity disable sample normalization. Default: false, value is being searched for recursively towards the root (sample) folder.
  if(sample->getTagBoolDefault("~ignoreSampleNormalization",false)) return true;
  if(sample->getTagBoolDefault("~isInitialized",false)) return true;
  double sumOfWeights = 0.;
  if(!sample->getTagDouble(".init.sumOfWeights",sumOfWeights) || sumOfWeights == 0.) success = false;
  double xsec = 0.;
  if (!sample->getTagDouble("~.xsp.xSection", xsec) && !sample->getTagDouble("~.xsection",xsec)) success = false;
  double kfactor = sample->getTagDoubleDefault("~kfactor",sample->getTagDoubleDefault("~.xsp.kFactor",1.));
  double filtereff = sample->getTagDoubleDefault("~filtereff", sample->getTagDoubleDefault("~.xsp.filterEfficiency", 1.));
  double luminosity = sample->getTagDoubleDefault("~luminosity",this->getTagDoubleDefault("luminosity",1.));
  const TString lumiUnitStr = sample->getTagStringDefault("~luminosityUnit",this->getTagStringDefault("luminosityUnit","pb"));
  double lumi = luminosity / TQXSecParser::unit(lumiUnitStr);
  double xSecScale = sample->getTagDoubleDefault("~.xsp.xSecScale",1.);
  
  if (!success) {
    //if we did not retrieve one of the required values we only set a sample normalization according to the xSecScale
    sample->setNormalisation(xSecScale);
    return false;
  }
  DEBUGclass("sampleName=%s, path=%s",sample->getName().Data(),sample->getPath().Data());
  DEBUGclass("xsec=%f, xsecscale=%f",xsec,xSecScale);
  DEBUGclass("filtereff=%f,kfactor=%f",filtereff,kfactor);
  DEBUGclass("sumOfWeights(unskimmed)=%f,samplefraction=%f",sumOfWeights,samplefraction);
  DEBUGclass("luminosity=%f",lumi);
  double nomval = samplefraction * lumi * xsec * xSecScale * kfactor * filtereff;
  double normalization =  nomval / sumOfWeights; 
  DEBUGclass("nominalValue=%f,normalization=%f",nomval,normalization);
  sample->setNormalisation(normalization);
  
  if(sample->getTagBoolDefault("isxAOD",false) && this->getTagBoolDefault("extractCounters",true)){
    bool ok = TQSampleInitializerBase::extractCountersFromSample(sample);
    if(!ok) this->setTagBool("extractCounters",false);
  }
  return true;
}


bool TQSampleInitializerBase::initializeSample(TQSample* sample, const TString& fullpath, TString& message){
  // initialize the sample, given a known file path
  TString treeName = "*";
  double sumOfWeights = 0;
  int nEntries = 0;
  TString fullpath_ = TQStringUtils::makeASCII(fullpath);
  
  bool initialize = this->getTagBoolDefault("initialize",true);
  bool success = false;
  if(initialize) success = this->getTreeInformation(sample,fullpath_,treeName,sumOfWeights,nEntries,message);
  sample->setTagString(".init.filepath",fullpath_);
  sample->setTagString(".init.treename",treeName);

  if(!success){
    return !initialize;
  }
  
  sample->setTreeLocation(fullpath_ + ":" + treeName);
  sample->setTagDouble(".init.sumOfWeights", sumOfWeights);
  sample->setTagDouble(".init.nEvents", nEntries);
  
  TString countername;
  //@tag: makeNominalCounter, makeSWCounter: string tag that, if 
  //@tag    present, will create counters containing the initial, nominal sum
  //@tag    of weights with a name given by the argument.
  if(this->getTagString("makeNominalCounter",countername) || this->getTagString("makeSWCounter",countername)){
    TString foldername = TQStringUtils::readPrefix(countername,":",".cutflow");
    TQFolder* f = sample->getFolder(foldername+"+");
    if(f){
      TQCounter* c = new TQCounter(countername,sumOfWeights,0,nEntries);
      f->addObject(c);
    }
  }
  return true;
}
