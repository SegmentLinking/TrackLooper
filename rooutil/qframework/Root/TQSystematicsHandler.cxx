#include "QFramework/TQSystematicsHandler.h"
#include "QFramework/TQHistogramUtils.h"
#include "QFramework/TQUtils.h"
#include "QFramework/TQIterator.h"

// #define _DEBUG_

#include "QFramework/TQLibrary.h"

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQSystematicsHandler:
//
// With the advent of the TQSysetmaticsHandler, it is now easily
// possible to include not only normalization, but also shape
// systematics into histograms produced by a plotter or add systematic
// uncertainties to the cutflow uncertainties.
//
////////////////////////////////////////////////////////////////////////////////////////////////

using PathFolderPair = std::pair<TString, TString>;

ClassImp(TQSystematicsHandler)

TQSystematicsHandler::TQSystematicsHandler(const TString& name) : 
TQFolder(name),
  fTotalPath("")
{
	// constructor with name
}

TQFolder* TQSystematicsHandler::addCut(const TString& id){
  // add a cut to the systematics handler
  // the added cuts can be looked 
  if(!TQFolder::isValidName(id)){
    ERRORclass("'%s' is not a valid identifier!",id.Data());
    return NULL;
  }
  TQFolder* cut = this->getFolder(id+"+");
  return cut;
}

bool TQSystematicsHandler::hasSystematic(const TString& name) const {
  // check if a systematic of this name has already been added
  return (this->fSystematics.find( name ) != this->fSystematics.end());
}

bool TQSystematicsHandler::addSystematic(const TString& name, const PathFolderPair& path){
  // add a single (i.e. resolution) systematic
  std::vector<PathFolderPair> path_folder_pairs = { path };
  return addSystematic(name, path_folder_pairs);
}

bool TQSystematicsHandler::addSystematic(const TString& name, const PathFolderPair& path1, const PathFolderPair& path2){
  // add a double (i.e. scale up/down) systematic
  std::vector<PathFolderPair> path_folder_pairs = { path1, path2 };
  return addSystematic(name, path_folder_pairs);
}

bool TQSystematicsHandler::addSystematic(const TString& name, const TString& path){
  // add a single (i.e. resolution) systematic from a different file using nominal folder path
  std::vector<PathFolderPair> path_folder_pairs = { {path, ""} };
  return addSystematic(name, path_folder_pairs);
}

bool TQSystematicsHandler::addSystematic(const TString& name, const TString& path1, const TString& path2){
  // add a double (i.e. scale up/down) systematic from a different file using nominal folder path
  std::vector<PathFolderPair> path_folder_pairs = { {path1, ""}, {path2, ""} };
  return addSystematic(name, path_folder_pairs);
}

bool TQSystematicsHandler::addSystematic(const TString& name, const std::vector<PathFolderPair>& path_folder_pairs){
  // Generic systematics adding routine
  // Protected for now, but may go public if we decide to support combining arbitrary folders
  if(this->hasSystematic(name)){
    ERRORclass("not adding systematic '%s': a sysetmatic of this name has already been added");
    return false;
  }
  for(const auto& p:path_folder_pairs){
    if(p.first.Contains("$") || p.second.Contains("$")){
      throw std::runtime_error(TString::Format("the expression '%s:%s' contains unresolved aliases!",p.first.Data(),p.second.Data()).Data());
    }
  }
  this->fSystematics[name] = path_folder_pairs;
  return true;
}

bool TQSystematicsHandler::addSystematicFromSampleFolderPath(const TString& name, const TString& path){
  // add a single (i.e. resolution) systematic
  std::vector<PathFolderPair> path_folder_pairs = { {"", path} };
  return addSystematic(name, path_folder_pairs);
}

bool TQSystematicsHandler::addSystematicFromSampleFolderPath(const TString& name, const TString& path1, const TString& path2){
  // add a double (i.e. scale up/down) systematic
  std::vector<PathFolderPair> path_folder_pairs = { {"", path1}, {"", path2} };
  return this->addSystematic(name, path_folder_pairs);
}

void TQSystematicsHandler::setNominal(const TString& path){
  // set the nominal file path
  this->setNominalFilePath(path);
}

void TQSystematicsHandler::setNominalFilePath(const TString& path){
  // set the nominal file path
  this->addSystematic("Nominal",path);
  this->fNominalFilePath = path;
}

void TQSystematicsHandler::setTotalPath(const TString& path){
  // set the total sample folder path for which the systematics should be computed
  setNominalSampleFolderPath(path);
}

void TQSystematicsHandler::setNominalSampleFolderPath(const TString& path){
  // set the total sample folder path for which the systematics should be computed
  this->fTotalPath = path;
}

void TQSystematicsHandler::collectHistograms(TQSampleDataReader* rd, const TString& subpath, const TString& sfolderpath){
  // collect the histograms from a reader for one subpath of the type systname/file.root/
  TQIterator histItr(rd->getListOfHistogramNames(sfolderpath),true);
  DEBUGclass("retrieved list of histogram names");
  // loop over all histogram known to the sample folder
  while(histItr.hasNext()){
    TObject* alias = histItr.readNext();
    if(!alias) continue;
    TString histname(alias->GetName());
    TString cutname = TQFolder::getPathHead(histname);
    // check if the histogram name starts a cut that is known to the systematics handler
    TQFolder* cut = this->getFolder(cutname);
    if(!cut) continue;
    TQFolder* target = cut->getFolder(subpath+"+");
    DEBUGclass("attempting to retrieve histogram '%s' for '%s' to store in '%s'",histname.Data(),cutname.Data(),target->getPath().Data());
    TH1* hist = rd->getHistogram(sfolderpath,TQFolder::concatPaths(cutname,histname));
    if(hist){
      DEBUGclass("success!");
      if(!target->addObject(hist)){
        WARNclass("unable to add histogram for '%s'",alias->GetName());
      }
      cut->setTagBool("hasHistograms",true);
    }
  }
}

void TQSystematicsHandler::collectCounters(TQSampleDataReader* rd, const TString& subpath, const TString& sfolderpath){
  // collect the counters from a reader for one subpath of the type systname/file.root/
  TQIterator cntItr(rd->getListOfCounterNames(sfolderpath),true);
  DEBUGclass("retrieved list of counter names");
  while(cntItr.hasNext()){
    TObject* alias = cntItr.readNext();
    if(!alias) continue;
    TString cntname(alias->GetName());
    // check if the counter name is equal to a cut that is known to the systematics handler
    if(!this->getFolder(cntname)) continue;
    TQFolder* target = this->getFolder(TQFolder::concatPaths(cntname,subpath)+"+");
    TQCounter* cnt = rd->getCounter(sfolderpath,cntname);
    cnt->SetName("yield");
    if(!target->addObject(cnt)){
      WARNclass("unable to add counter for '%s'",alias->GetName());
      target->print();
    }
  }
}

TQSampleFolder* TQSystematicsHandler::getSampleFolder(const TString& systname,int i){
  bool cache = this->getTagBoolDefault("useCache",true);
  TString sfpath(this->fSystematics[systname][i].first);
  if (sfpath.IsNull()) sfpath = fNominalFilePath;
  TString filepath,sfname;
  TQFolder::parseLocation(sfpath,filepath,sfname);
  if(sfname.IsNull()) sfname="*";
  TString sfstring(filepath + ":" + sfname);
  DEBUGclass("trying to retrieve '%s'",sfstring.Data());
  //@tag:[lazy] This object tag determines if sample folders should be loaded in 'lazy' mode or not. Default: false.
  TQSampleFolder* sf = NULL;
  if(fInputs.find(sfstring) == fInputs.end()){
    sf = TQSampleFolder::loadSampleFolder(sfstring,this->getTagBoolDefault("lazy",false));
    if(cache) fInputs[sfstring] = sf;
  } else {
    sf = fInputs[sfstring];
  }
  return sf;
}

void TQSystematicsHandler::collectSystematic(const TString& systname){
  // collect counters and histograms for one systematic 
  //@tag: [verbose] This object tag enables verbosity.
  if(this->getTagBoolDefault("verbose",false))
    INFO("retrieving objects for '%s'",systname.Data());
  bool cache = this->getTagBoolDefault("useCache",true);
  // loop over the systematics added via addSystematics
  for(size_t i=0; i<this->fSystematics[systname].size(); ++i){
    TString subpath = TQFolder::concatPaths(TQFolder::makeValidIdentifier(systname,"_"),TString::Format("variation%d",(int)i));
    TString sfolderpath(this->fSystematics[systname][i].second);
    if(this->getTagBoolDefault("verbose",false))
      INFO("\tfrom '%s:%s'",this->fSystematics[systname][i].first.Data(),sfolderpath.Data());
    TQSampleFolder* sf = this->getSampleFolder(systname,i);
    if(!sf){
      ERRORclass("unable to load sample folder from file '%s'",this->fSystematics[systname][i].first.Data());
      continue;
    }
    DEBUGclass("opened sample folder '%s' from '%s'",systname.Data(),fSystematics[systname][i].first.Data());
    TQSampleDataReader* rd = new TQSampleDataReader(sf);
    if(!rd){
      ERRORclass("unable to create reader for sample folder '%s'",sf->GetName());
      continue;
    }
    if (sfolderpath.IsNull()) sfolderpath = fTotalPath;
    DEBUGclass("created reader, retrieving histograms from '%s'",sfolderpath.Data());
    this->collectCounters(rd,subpath,sfolderpath);
    this->collectHistograms(rd,subpath,sfolderpath);
    delete rd;
    if(!cache) delete sf;
  }
  DEBUGclass("done with systematic '%s'",systname.Data());
}

void TQSystematicsHandler::collectSystematics(){
  // collect counters and histograms for all registered systematics
  if(this->fTotalPath.IsNull()){
    ERRORclass("cannot collect systematics without path, please call TQSystematicsHandler::setTotalPath('/path/to/my/samples')!");
    return;
  }
  DEBUGclass("starting collection of systematics");
  for(auto sitr = this->fSystematics.begin(); sitr!=this->fSystematics.end(); ++sitr){
    this->collectSystematic(sitr->first);
  }
}

void TQSystematicsHandler::printSystematics() const{
  // print a list of all registered systematics
  for(auto sitr = this->fSystematics.begin(); sitr!=this->fSystematics.end(); ++sitr){
    TString systname(sitr->first);
    std::cout << TQStringUtils::makeBoldWhite(systname) << std::endl;
    for(size_t i=0; i<sitr->second.size(); i++){
      std::cout << "\t" << sitr->second[i].first << "\t" << sitr->second[i].second << std::endl;
    }
  }
}

int TQSystematicsHandler::createVariationHistograms(TQFolder* syst, TQFolder* nominal){
  // create the variation histograms for one systematic 
  if(this->getTagBoolDefault("verbose",false))
    INFO("creating histograms for variation '%s'",syst->GetName());
  int nHistograms = 0;
  TQFolder* firstsyst = syst->getFolder("?");
  if(!firstsyst) return 0;
  TCollection* hists = firstsyst->getListOfObjects("?",TH1::Class());
  TQTH1Iterator histograms(hists);
  TQFolderIterator variants(syst->getListOfFolders("?"),true);
  while(histograms.hasNext()){
    TH1* hist = histograms.readNext();
    if(!hist) continue;
    if(syst->getObject(hist->GetName())){
      DEBUGclass("variation '%s' already has a histogram named '%s', skipping",syst->getPath().Data(),hist->GetName());
      continue;
    }
    TH1* nomhist = dynamic_cast<TH1*>(nominal->getObject(hist->GetName()));
    if(!nomhist) continue;
    TH1* systhist = TQHistogramUtils::copyHistogram(hist);
    systhist->Reset();
    systhist->SetDirectory(NULL);
    DEBUGclass("handling histogram '%s'",hist->GetName());
    int n = 0;
    while(variants.hasNext()){
      TQFolder* variant = variants.readNext();
      if(!variant) continue;
      TH1* h = dynamic_cast<TH1*>(variant->getObject(systhist->GetName()));
      if(!h){
        DEBUGclass("no histogram named '%s' in '%s'",systhist->GetName(),variant->GetName());
        continue;
      }
      n++;
      DEBUGclass("retrieved histogram for variant '%s'",variant->GetName());
      for(int i=0; i<=systhist->GetNbinsX()+1; ++i){
        // Make sure that we don't perform a 0 division here, doesn't make sense anyway to evaluate the bin uncertainty if the bin entry is 0 or negative
        if( !(nomhist->GetBinContent(i) > 0.) ) continue;
        double c = fabs( (h->GetBinContent(i) - nomhist->GetBinContent(i))/nomhist->GetBinContent(i) );
        if(c > systhist->GetBinContent(i)) systhist->SetBinContent(i,c);
      }
    }
    if(systhist){
      DEBUGclass("created histogram '%s' for variation '%s'",systhist->GetName(),syst->GetName());
      systhist->SetEntries(n);
      if(!syst->addObject(systhist)){
        WARNclass("unable to add histogram '%s'",systhist->GetName());
      }
      nHistograms++;
    }
    variants.reset();
  }
  return nHistograms;
}

void TQSystematicsHandler::createVariationYield(TQFolder* syst, TQFolder* nominal){
  // create the variation counters for one systematic
  if(!syst){
    ERRORclass("invalid systematic given!");
    return;
  }
  if(!nominal){
    ERRORclass("invalid nominal given!");
    return;
  }
  if(this->getTagBoolDefault("verbose",false))
    INFO("creating yield counters for variation '%s'",syst->GetName());
  TQFolderIterator variants(syst->getListOfFolders("?"),true);
  double yield = 0;
  TQCounter* nomcount = dynamic_cast<TQCounter*>(nominal->getObject("yield"));
  while(variants.hasNext()){
    TQFolder* variant = variants.readNext();
    if(!variant) continue;
    TQCounter* systcnt = dynamic_cast<TQCounter*>(variant->getObject("yield"));
    if(!systcnt){
      DEBUGclass("no yield counter in '%s'",variant->GetName());
      continue;
    }
    yield = std::max(fabs(systcnt->getCounter() - nomcount->getCounter()),yield);
  }
  DEBUGclass("setting yield for systematic '%s'",syst->GetName());
  yield /= (nomcount->getCounter());
  syst->setTagDouble("yield",yield);
}

void TQSystematicsHandler::exportObjects(TQFolder* cut,TQFolder* target){
  // export the created objects for one cut
  if(!cut || !target) return;
  TQTH1Iterator histograms(cut->getListOfObjects("Nominal/?/?",TH1::Class()),true);
  TQFolderIterator systItr(cut->getListOfFolders("?"),true);
  while(histograms.hasNext()){
    TH1* h = histograms.readNext();
    if(!h) continue;
    TH1* systhist = TQHistogramUtils::copyHistogram(h);
    if(!systhist) continue;
    systhist->Reset();
    systhist->SetDirectory(NULL);
    int n = 0;
    while(systItr.hasNext()){
      TQFolder* syst = systItr.readNext();
      if(!syst) continue;
      TH1* h = dynamic_cast<TH1*>(syst->getObject(systhist->GetName()));
      if(!h) continue;
      n++;
      for(int i=0; i<systhist->GetNbinsX()+1; ++i){
        systhist->AddBinContent(i,pow(h->GetBinContent(i),2));
      }
    }
    systItr.reset();
    for(int i=0; i<systhist->GetNbinsX()+1; ++i){
      systhist->SetBinContent(i,sqrt(systhist->GetBinContent(i)));
      systhist->SetBinError(i,0);
    }
    systhist->SetEntries(n);
    if(!target->addObject(systhist)){
      WARNclass("unable to export histogram '%s'",systhist->GetName());
    }
  }
  double totalyield = 0.;
  while(systItr.hasNext()){
    TQFolder* syst = systItr.readNext();
    if(!syst) continue;
    double yield = 0.;
    //@tag:[yield] Folder tag (TODO)
    if(syst->getTagDouble("yield",yield) && TQUtils::isNum(yield)){
      totalyield += yield*yield;
    }
  }
  target->setTagDouble("yield",sqrt(fabs(totalyield)));
}

TQFolder* TQSystematicsHandler::exportSystematics(){
  // create and export variation histograms and counters for all systematics
  TQFolder* systematics = new TQFolder(this->GetName());
  TQFolderIterator cuts(this->getListOfFolders(),true);
  while(cuts.hasNext()){
    TQFolder* cut = cuts.readNext();
    if(!cut) continue;
    TQFolder* nominal = cut->getFolder("Nominal/?");
    if(!nominal){
      ERROR("unable to locate nominal for cut '%s'!",cut->GetName());
      continue;
    }
    TQFolder* target = systematics->getFolder(TString(cut->GetName())+"+");
    TQFolderIterator systs(cut->getListOfFolders("?"),true);
    while(systs.hasNext()){
      TQFolder* syst = systs.readNext();
      if(!syst) continue;
      if(TQStringUtils::equal(syst->GetName(),"Nominal")) continue;
      //@tag:[hasHistograms] Folder tag (TODO)
      if(cut->getTagBoolDefault("hasHistograms",false)){
        DEBUGclass("starting to create variation histograms at cut '%s'",cut->GetName());
        this->createVariationHistograms(syst,nominal);
      }
      DEBUGclass("starting to create variation yield at cut '%s'",cut->GetName());
      this->createVariationYield(syst,nominal);
    }
    if(this->getTagBoolDefault("verbose",false))
      INFO("exporting objects for cut '%s'",cut->GetName());
    TQSystematicsHandler::exportObjects(cut,target);
  }
  return systematics;
}

TList* TQSystematicsHandler::getRanking(const TString& cutname){
  // return the systematics ranking for one specific cut
  TQFolder* cut = this->getFolder(cutname);
  if(!cut) return NULL;
  TQFolderIterator systematics(cut->getListOfFolders("?"),true);
  TQFolder* nominal = cut->getFolder("Nominal/?");
  while(systematics.hasNext()){
    TQFolder* systematic = systematics.readNext();
    if(!systematic) continue;
    if(TQStringUtils::equal(systematic->GetName(),"Nominal")) continue;
    if(!systematic->hasTag("yield")){
      this->createVariationYield(systematic,nominal);
    }
  }
  systematics.reset();
  TList* folders = new TList();
  // insertion sort
  while(systematics.hasNext()){
    TQFolder* systematic = systematics.readNext();
    if(!systematic) continue;
    if(TQStringUtils::equal(systematic->GetName(),"Nominal")) continue;
    double yield = systematic->getTagDoubleDefault("yield",0);
    TQFolderIterator itr(folders);
    int pos = 0;
    while(itr.hasNext()){
      TQFolder* f = itr.readNext();
      if(f->getTagDoubleDefault("yield",0) < yield){
        //        std::cout << systematic->GetName() << " outranks " << f->GetName() << ", putting at position " << pos << std::endl;
        break;
      }
      pos++;
    }
    //    std::cout << "adding " << systematic->GetName() << " at " << pos << " (yield=" << yield << ")" << std::endl;
    folders->AddAt(systematic,pos);
  }
  return folders;
}

TQTable* TQSystematicsHandler::getTable(const TString& cutname){
  // write the systematics table
  TList* ranking = this->getRanking(cutname);
  TQTable* table = new TQTable(cutname);
  table->expand(ranking->GetEntries()+1,2);
  TQFolderIterator systematics(ranking);
  table->setEntry(0,0,"Systematic");
  table->setEntry(0,1,"\\% yield","latex");
  table->setEntry(0,1,"% yield","plain");
  table->setEntry(0,1,"% yield","html");
  while(systematics.hasNext()){
    TQFolder* systematic = systematics.readNext();
    if(!systematic) continue;
    double yield = systematic->getTagDoubleDefault("yield",0);
    size_t row = systematics.getLastIndex()+1;
    table->setEntry(row,0,systematic->GetName(),"ascii");
    //@tag: [title.latex,title.html] This folder tag is used to set the LaTeX/HTML representations of the systematic's title.
    table->setEntry(row,0,systematic->getTagStringDefault("title.latex",systematic->GetName()),"latex");
    table->setEntry(row,0,systematic->getTagStringDefault("title.html",systematic->GetName()),"html");
    table->setEntryValue(row,1,yield*100);
  }
  return table;
}

TQSystematicsHandler::~TQSystematicsHandler(){
  for(auto it:fInputs){
    delete it.second;
  }
}
