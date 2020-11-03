#include "QFramework/TQXSecParser.h"
#include "QFramework/TQIterator.h"
#include "QFramework/TQUtils.h"

 // #define _DEBUG_

#include "QFramework/TQLibrary.h"

ClassImp(TQXSecParser)

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQXSecParser:
//
// The TQXSecParser is capable of reading crosssection information from a text file.
// Multiple input formattings are supported, the functionality relies on TQTable.
//
// The first row of the text file is interpreted as column labels.
// The first column will be taken as sample name.
// 
// The other colum labels are expected to match the following patterns unless
// specified differently by setting the corresponding tags:
// 
// column content    | default pattern | tag name
// --------------------------------------------------------
// cross section     | *xsec*          | xSecColName 
// filter efficiency | *filt*eff*      | xFilterEffColName
// kFactor           | *k*fac*         | kFactorColName
// process info      | *proc*          | processInfoColName
// generator info    | *gen*           | generatorInfoColName
// simulation info   | *sim*           | simulationInfoColName
//
// For the column tag name xyz, the case sensitivity of the pattern
// can be controlled by setting the boolean tag "xyzCaseSensitive" on
// the TQXSecParser instance at hand, where the default is 'false' in
// all cases.
//
// If the cross section column title contains any of the following
// strings, this will affect the way the cross sections are
// implemented:
//  'mb': cross section interpreted in mb
//  'µb': cross section interpreted in µb
//  'nb': cross section interpreted in nb
//  'pb': cross section interpreted in pb (default)
//  'fb': cross section interpreted in fb
//  'ab': cross section interpreted in ab
//
// This information is used to construct a TQSampleFolder hierarchy.
//
// Columns not matching any of these patterns are ignored. However,
// the mapping of sample identifiers to paths in the sample folder
// hierarchy can explicitly be read in from a column of arbirtrary
// title (e.g. 'path') using the function
// TQXSecParser::readMappingFromColumn('path'). Alternatively, the
// mapping can be read in from an external file that contains lines of
// the type 'sampleID /path/to/this/sample'.
//
// Similarly, samples can be enabled or disabled based on a numeric
// priority column of arbitrary title (e.g. 'priority') by using the
// functions
// [enable|disable]SamplesWithPriority[Less|Greater]Than('priority',N),
// where N is the threshold of choice.
//
// It is also possible to add chunks of nTuples from a given path in
// the file system, regardless of their appearance in the cross
// section file, using the function
// TQXSecParser::addAllSamplesFromPath(...).
//
////////////////////////////////////////////////////////////////////////////////////////////////

TQXSecParser::TQXSecParser(TQSampleFolder* sf, const TString& xsecfilename) :
TQTable(xsecfilename),
  fSampleFolder(sf),
  fPathVariants(new TObjArray())
{
  // constructor receiving a sample folder and reading a tab-separated file
  this->setTagString("readFormatPrior","verbatim");
  this->setTagString(".xsecfilename",xsecfilename);
  this->setTagBool("adjustColWidth",true);
  this->readTSVfile(xsecfilename);
  this->fPathVariants->SetOwner(true);
}

TQXSecParser::TQXSecParser(const TString& xsecfilename) :
  TQTable(xsecfilename),
  fSampleFolder(NULL),
  fPathVariants(new TObjArray())
{
  // constructor reading a tab-separated file
  this->setTagString("readFormatPrior","verbatim");
  this->setTagString(".xsecfilename",xsecfilename);
  this->setTagBool("adjustColWidth",true);
  this->readTSVfile(xsecfilename);
  this->fPathVariants->SetOwner(true);
}

TQXSecParser::TQXSecParser(TQTable& tab) :
  TQTable(tab),
  fSampleFolder(NULL),
  fPathVariants(new TObjArray())
{
  // copy constructor based on TQTable
  this->setTagString("readFormatPrior","verbatim");
  this->setTagBool("adjustColWidth",true);
  this->cloneSettingsFrom(dynamic_cast<TQXSecParser*>(&tab));
}


TQXSecParser::TQXSecParser(TQTable* tab) :
  TQTable(tab),
  fSampleFolder(NULL),
  fPathVariants(new TObjArray())
{
  // copy constructor based on TQTable (pointer variant)
  this->setTagBool("adjustColWidth",true);
  this->cloneSettingsFrom(dynamic_cast<TQXSecParser*>(tab));
}

void TQXSecParser::cloneSettingsFrom(TQXSecParser* parser){
  // internal cloning helper
  if(!parser) return;
  this->fPathVariants->SetOwner(true);
  this->fPathVariants->Clear();
  this->importTags(parser);
  this->fSampleFolder = parser->fSampleFolder;
  TQIterator itr(parser->fPathVariants);
  while(itr.hasNext()){
    this->fPathVariants->Add(itr.readNext()->Clone());
  }
}

TQXSecParser::TQXSecParser(TQXSecParser* parser) :
  TQTable(parser),
  fPathVariants(new TObjArray())
{
  // copy constructor based on TQTable (pointer variant)
  this->fPathVariants->SetOwner(true);
  this->cloneSettingsFrom(parser);
}

TQXSecParser::TQXSecParser(TQXSecParser& parser) :
  TQTable(parser),
  fPathVariants(new TObjArray())
{
  // copy constructor based on TQTable
  this->fPathVariants->SetOwner(true);
  this->cloneSettingsFrom(&parser);
}


TQXSecParser::TQXSecParser(TQSampleFolder* sf) :
  TQTable("TQXSecParser"),
  fSampleFolder(sf),
  fPathVariants(new TObjArray())
{
  // constructor reading a tab-separated file
  this->fPathVariants->SetOwner(true);
  this->setTagBool("adjustColWidth",true);
}

TQXSecParser::TQXSecParser() :
  TQTable("TQXSecParser"),
  fSampleFolder(NULL),
  fPathVariants(new TObjArray())
{
  // default constructor
  this->setTagBool("adjustColWidth",true);
  this->fPathVariants->SetOwner(true);
}

TQXSecParser::~TQXSecParser(){
  // default destructor
  delete this->fPathVariants;
}

void TQXSecParser::addPathVariant(const TString& replacements){
  // add a path variant to the list
  TQNamedTaggable* variant = new TQNamedTaggable("variant");
  variant->importTags(replacements);
  this->fPathVariants->Add(variant);
}

void TQXSecParser::addPathVariant(const TString& key, const TString& value){
  // add a path variant to the list
  TQNamedTaggable* variant = new TQNamedTaggable(value);
  variant->setTagString(key,value);
  this->fPathVariants->Add(variant);
}

void TQXSecParser::printPathVariants(){
  // print all registered path variants
  std::cout << TQStringUtils::makeBoldWhite(this->GetName()) << " - known path variants:" << std::endl;
  TQTaggableIterator itr(this->fPathVariants);
  while(itr.hasNext()){
    TQNamedTaggable* variant = itr.readNext();
    std::cout << "\t" << variant->exportTagsAsString() << std::endl;
  }
}

void TQXSecParser::clearPathVariants(){
  // clear known path variants
  this->fPathVariants->Clear();
}

TQSampleFolder* TQXSecParser::getSampleFolder(){
  // get the sample folder associated to this parser
  return this->fSampleFolder;
}

void TQXSecParser::setSampleFolder(TQSampleFolder* sf){
  // set the sample folder associated to this parser
  this->fSampleFolder = sf;
}

int TQXSecParser::readMapping(const TString& fname, bool print){
  // add mapping from an external TSV-formatted file
  TQTable tmp;
  if(!tmp.readTSVfile(fname)) return false;
  if(print) tmp.printPlain();
  int retval = this->readMapping(tmp);
  if(retval > 0) this->setTagBool(".hasMapping",true);
  return retval;
}

int TQXSecParser::writeMappingToColumn(const TString& colname){
  // copy paths to the given column
  // creates a column with this name if necessary
  this->shrink();
  int colidx = this->findColumn(colname);
  if(colidx < 0){
    this->setAutoExpand(true);
    colidx = this->ncols;
    if(!this->setEntry(0,colidx,colname)){
      ERRORclass("unable to allocate '%s' column!",colname.Data());
      return -1;
    }
    if(colidx != this->findColumn(colname)){
      ERRORclass("unable to find '%s' column!",colname.Data());
      return -1;
    }
  }
  this->setColAlign(colidx,"l");
  int count = 0;
  for(size_t i=1; i<this->nrows; i++){
    TQTaggable* entry = this->getEntryInternal(i,0);
    if(!entry) continue;
    TString path = entry->getTagStringDefault("path","");
    if(path.IsNull()) continue;
    this->setEntry(i,colidx,path);
    count++;
  }
  this->shrink();
  return count;
}

int TQXSecParser::readMappingFromColumn(const TString& colname){
  // copy paths from the given column
  int colidx = this->findColumn(colname);
  if(colidx < 0) return -1;
  this->setColAlign(colidx,"l");
  int count = 0;
  for(size_t i=1; i<this->nrows; i++){
    TString path = this->getEntryVerbatim(i,colidx);
    if(path.IsNull()) continue;
    this->setProperty(i,0,"path",path);
    count++;
  }
  if(count > 0){
    this->setTagBool(".hasMapping",true);
  }
  return count;
}

int TQXSecParser::readMapping(TQTable& tmp){
  // add a mapping from an external table
  int count = 0;
  if(tmp.getNcols() < 2){
    return -1;
  }
  int pathcol = tmp.findColumn("path");
  if(pathcol < 0) pathcol = 1;
  std::map<TString,TString> map = tmp.getMap(0,pathcol,"verbatim","verbatim",false);
  std::map<TString,TString>::iterator it;
  for(size_t i=1; i<this->nrows; i++){
    TQTaggable* thisEntry = this->getEntryInternal(i,0);
    if(!thisEntry) continue;
    TString thisKey;
    if(!thisEntry->getTagString("content.verbatim",thisKey)) continue;
    it = map.find(thisKey);
    if(it==map.end()) continue;
    TString path(it->second);
    thisEntry->setTagString("path",path);
    count++;
  }
  return count;
}

int TQXSecParser::enableSamplesWithColumnStringMatch(const TString& colname, const TString& pattern, bool verbose){
  // enables all samples with string match in column
  int prioidx = this->findColumn(colname);
  if(prioidx < 0) return -1;
  int count = 0;
  for(size_t i=1; i<this->nrows; i++){
    TString entry = this->getEntryPlain(i,prioidx);
    if(TQStringUtils::matches(entry,pattern)){
      if(verbose) INFOclass("enabling sample '%s', '%s' matches '%s'",this->getEntry(i,0).Data(),entry.Data(),pattern.Data());
      this->setProperty(i,0,"enabled",true);
      this->setProperty(i,0,"bold",true);
      count++;
    }
  }
  return count;
}

int TQXSecParser::disableSamplesWithColumnStringMatch(const TString& colname, const TString& pattern, bool verbose){
  // disables all samples with string match in column
  int prioidx = this->findColumn(colname);
  if(prioidx < 0) return -1;
  int count = 0;
  for(size_t i=1; i<this->nrows; i++){
    TString entry = this->getEntryPlain(i,prioidx);
    if(TQStringUtils::matches(entry,pattern)){
      if(verbose) INFOclass("disabling sample '%s', '%s' matches '%s'",this->getEntry(i,0).Data(),entry.Data(),pattern.Data());
      this->setProperty(i,0,"enabled",false);
      this->setProperty(i,0,"bold",false);
      count++;
    }
  }
  return count;
}


int TQXSecParser::enableSamplesWithPriorityLessThan(const TString& colname, int val, bool verbose){
  // enables all samples with priority (given by colname) less than the given value
  int prioidx = this->findColumn(colname);
  if(prioidx < 0) return -1;
  int count = 0;
  for(size_t i=1; i<this->nrows; i++){
    int p = this->getEntry(i,prioidx).Atoi();
    if(p < val){
      if(verbose) INFOclass("enabling sample '%s', priority=%d",this->getEntry(i,0).Data(),p);
      this->setProperty(i,0,"enabled",true);
      this->setProperty(i,0,"bold",true);
      count++;
    }
  }
  return count;
}

int TQXSecParser::disableSamplesWithPriorityLessThan(const TString& colname, int val, bool verbose){
  // disables all samples with priority (given by colname) less than the given value
  int prioidx = this->findColumn(colname);
  if(prioidx < 0) return -1;
  int count = 0;
  for(size_t i=1; i<this->nrows; i++){
    int p = this->getEntry(i,prioidx).Atoi();
    if(p < val){
      if(verbose) INFOclass("disabling sample '%s', priority=%d",this->getEntry(i,0).Data(),p);
      this->setProperty(i,0,"enabled",false);
      this->setProperty(i,0,"bold",false);
      count++;
    }
  }
  return count;
}

int TQXSecParser::enableSamplesWithPriorityGreaterThan(const TString& colname, int val, bool verbose){
  // enables all samples with priority (given by colname) greater than the given value
  int prioidx = this->findColumn(colname);
  if(prioidx < 0) return -1;
  int count =0;
  for(size_t i=1; i<this->nrows; i++){
    int p = this->getEntry(i,prioidx).Atoi();
    if(p > val){
      if(verbose) INFOclass("enabling sample '%s', priority=%d",this->getEntry(i,0).Data(),p);
      this->setProperty(i,0,"enabled",true);
      this->setProperty(i,0,"bold",true);
      count++;
    }
  }
  return count;
}

int TQXSecParser::disableSamplesWithPriorityGreaterThan(const TString& colname, int val, bool verbose){
  // disables all samples with priority (given by colname) greater than the given value
  int prioidx = this->findColumn(colname);
  if(prioidx < 0) return -1;
  int count = 0;
  for(size_t i=1; i<this->nrows; i++){
    int p = this->getEntry(i,prioidx).Atoi();
    if(p > val){
      if(verbose) INFOclass("disabling sample '%s', priority=%d",this->getEntry(i,0).Data(),p);
      this->setProperty(i,0,"enabled",false);
      this->setProperty(i,0,"bold",false);
      count++;
    }
  }
  return count;
}

int TQXSecParser::applyWhitelist(const TString& filename){
  // enables all listed samples
  std::vector<TString>* lines =  TQStringUtils::readFileLines(filename);
  if (!lines) {
    ERRORclass("Failed to read whitelist file '%s'. Does the file exist?",filename.Data());
    return 0;
  }
  int nActivated = this->applyWhitelist(lines);
  delete lines;
  return nActivated;
}

int TQXSecParser::applyWhitelist(std::vector<TString>* lines){
  // enables all listed samples (all non-listed ones will be disabled)
  // returns the number of enabled samples
  if(!lines) return false;
  this->disableAllSamples();
  int count = 0;
  for(const auto& it:*lines){
    TString line(it);
    TString name;
    TQStringUtils::readUpTo(line,name," \t\n");
    TQStringUtils::removeLeading(line," \t\n");
    const bool replace = !line.IsNull();
    if(replace){
      line.ReplaceAll("$",name);
    }
    for(size_t i=1; i<this->nrows; i++){
      if(TQStringUtils::equal(name,this->getEntry(i,0))){
        if(replace) this->setProperty(i,0,"matchingName",line);
        this->setProperty(i,0,"enabled",true);
        this->setProperty(i,0,"bold",true);
        count++;
      }
    }
  }
  if (count==0) {
    WARNclass("No samples passed the whitelist selection!");
  }  
  
  return count;
}

void TQXSecParser::selectFirstColumn(std::vector<TString>* lines){
  // remove all but the first column in a list of strings
  if(!lines) return;
  for(auto& line:*lines){
    TQStringUtils::removeLeading(line," \t");
    int pos = TQStringUtils::findFirstOf(line," \t\n");
    if(pos > 0 && pos < line.Length()){
      line.Remove(pos,line.Length()-pos);
    }
  }
}

int TQXSecParser::applyBlacklist(const TString& filename){
  // disables all listed samples
  std::vector<TString>* lines =  TQStringUtils::readFileLines(filename);
  TQXSecParser::selectFirstColumn(lines);
  return this->applyBlacklist(lines);
}

int TQXSecParser::applyBlacklist(std::vector<TString>* lines){
  // disables all listed samples
  if(!lines) return false;
  int count = this->enableAllSamples();
  for(size_t i=1; i<this->nrows; i++){
    TString name = this->getEntry(i,0);
    auto it = TQStringUtils::find(name,*lines);
    if(it != -1){
      this->setProperty(i,0,"enabled",false);
      this->setProperty(i,0,"bold",false);
      count--;
    }
  }
  return count;
}

int TQXSecParser::enableAllSamples(){
  // enables all listed samples
  int count = 0;
  for(size_t i=1; i<this->nrows; i++){
    this->setProperty(i,0,"enabled",true);
    this->setProperty(i,0,"bold",true);
    count++;
  }
  return count;
}

int TQXSecParser::disableAllSamples(){
  // enables all listed samples
  int count = 0;
  for(size_t i=1; i<this->nrows; i++){
    this->setProperty(i,0,"enabled",false);
    this->setProperty(i,0,"bold",false);
    count++;
  }
  return count;
}

int TQXSecParser::addAllEnabledSamples(){
  // add all mapped samples to the folder 
  // an enabled sample is a mapped sample
  // that has the 'enabled' flag set
  return this->addAllSamples(true,true);
}

int TQXSecParser::addAllMappedSamples(){
  // add all mapped samples to the folder 
  // a mapped sample is one that has a path assigned to it
  return this->addAllSamples(false,true);
}

int TQXSecParser::addAllListedSamples(){
  // add all listed samples to the folder
  return this->addAllSamples(false,false);
}


bool TQXSecParser::hasCompleteMapping(bool requireEnabled){
  // return true if all samples are mapped to a path
  // if requireEnabled=true, only enabled samples will be checked
  for(size_t i=1; i<this->nrows; i++){
    TQTaggable* entry = this->getEntryInternal(i,0);
    if(requireEnabled && !entry->getTagBoolDefault("enabled",false)) continue;
    if(!entry->hasTagString("path")) return false;
  }
  return true;
}

int TQXSecParser::addAllSamples(bool requireEnabled, bool requirePath){
  // add all samples to the folder
  DEBUGclass("entering function, requireEnabled=%d, requirePath=%d",(int)requireEnabled,(int)requirePath);
  if(!this->fSampleFolder) return -1;
  int retval = 0;
  if(this->fPathVariants->GetEntries() < 1){
    this->addPathVariant("");
  }
  int treeNameCol = this->findColumn("treename");
  int sowCol = this->findColumn(this->getTagStringDefault("xAverageWeightColName","*sum*of*weight*"),this->getTagBoolDefault("xAverageWeightColNameCaseSensitive",false));
  int xSecCol = this->findColumn(this->getTagStringDefault("xSecColName","*xsec*"),this->getTagBoolDefault("xSecColNameCaseSensitive",false));
  int filterEffCol = this->findColumn(this->getTagStringDefault("xFilterEffColName","*filt*eff*"),this->getTagBoolDefault("xFilterEffColNameCaseSensitive",false));
  int kFactorCol = this->findColumn(this->getTagStringDefault("kFactorColName","*k*fac*"),this->getTagBoolDefault("kFactorColNameCaseSensitive",false));
  int procInfCol = this->findColumn(this->getTagStringDefault("processInfoColName","*proc*"),this->getTagBoolDefault("processInfoColNameCaseSensitive",false));
  int genInfCol = this->findColumn(this->getTagStringDefault("generatorInfoColName","*gen*"),this->getTagBoolDefault("generatorInfoColNameCaseSensitive",false));
  int simInfCol = this->findColumn(this->getTagStringDefault("simulationInfoColName","*sim*"),this->getTagBoolDefault("simulationInfoColNameCaseSensitive",false));

  TString unitstr = this->getEntryPlain(0,xSecCol);

  this->getTagString("xSectionUnit",unitstr);
  TQXSecParser::Unit unit = TQXSecParser::unit(unitstr);
  if (unit == TQXSecParser::Units::UNKNOWN || unit != unit) unit = TQXSecParser::Units::picobarn; //set default value if no valid value has been found. The check here actualy makes sense as UNKNOWN is currently defined as std::numeric_limits<double>::quiet_NaN(). We still keep the first part in case this gets changed (preventing a potential, very subtle bug...)
  DEBUGclass("using unit '%s', factor is '%f",unitstr.Data(),unit);

  if(!this->getTagBoolDefault(".hasMapping",false)){
    if(this->readMappingFromColumn("*path*") < 1){
      ERRORclass("refusing to add samples - no mapping specified, no adequate column found");
      return -1;
    }
  }
  TString treeName;
  for(size_t i=1; i<this->nrows; i++){
    DEBUGclass("investigating row %d...",(int)i);
    TQTaggable* entry = this->getEntryInternal(i,0);
    if(!entry){
      ERRORclass("encountered empty row at index %d - this should never have happened!",(int)i);
      continue;
    }
    TString samplenameorig;
    TString pathpattern = "uncategorized";
    if(!entry->getTagString("matchingName", samplenameorig) && !entry->getTagString("content.verbatim", samplenameorig)) continue; //if no whitelist has been applied, we use the regular content of the entry (i.e. the sample name)
    TString samplename = TQFolder::makeValidIdentifier(samplenameorig);
    if(requireEnabled && !entry->getTagBoolDefault("enabled",false)) continue;
    if(!entry->getTagString("path",pathpattern) && requirePath) continue;
    DEBUGclass("adding sample '%s'",samplename.Data());
    TQTaggableIterator itr(this->fPathVariants);
    while(itr.hasNext()){
      TQNamedTaggable* pathVariant = itr.readNext();
      TString path = pathVariant ? pathVariant->replaceInText(pathpattern) : pathpattern;
      DEBUGclass("path is '%s'",path.Data());
      TQSampleFolder* sf = this->fSampleFolder->getSampleFolder(path+"+");
      if(!sf){
        ERRORclass("cannot create sample folder at location '%s' - skipping sample '%s'",path.Data(),samplename.Data());
        continue;
      }
      if(sf->hasObject(samplename)){
        WARNclass("cowardly refusing to overwrite sample '%s' in folder '%s' - please check your cross section file for path conflicts!",samplename.Data(),path.Data());
        continue;
      } 
      TQSample* s = new TQSample(samplename);
      if(!s){
        ERRORclass("unable to allocate sample with name '%s' - this should never have happened!",samplename.Data());
      }
      s->setTagString("name",samplename);
      s->setTagInteger("dsid",atof(samplename));
      s->setTagString(".xsp.sampleName",samplenameorig);
      for(size_t j=1; j<this->ncols; j++){
        TString key = TQFolder::makeValidIdentifier(this->getEntryPlain(0,j));
        TString value = this->getEntryPlain(i,j);
        if (!key.IsNull())
          {
            DEBUGclass("copying tag '%s'='%s'",key.Data(),value.Data());
            s->setTagString(key,value);
          }
      }
      double xSecScale;
      if(this->getTagDouble("xSecScale",xSecScale)){
        s->setTagDouble(".xsp.xSecScale",xSecScale);
      }
      if(xSecCol > 0){
        double xsec = this->getEntryDouble(i,xSecCol);
        s->setTagDouble(".xsp.xSection",TQXSecParser::convertUnit(xsec,unit,TQXSecParser::Units::picobarn));
      }
      if(filterEffCol > 0){
        double filtereff = this->getEntryDouble(i,filterEffCol);
        s->setTagDouble(".xsp.filterEfficiency",filtereff);
      }
      if(kFactorCol > 0){
        double kfactor = this->getEntryDouble(i,kFactorCol);
        s->setTagDouble(".xsp.kFactor",kfactor);
      }
      if(procInfCol > 0){
        TString processInfo = this->getEntryPlain(i,procInfCol);
        s->setTagString(".xsp.process",processInfo);
      }
      if(genInfCol > 0){
        TString generatorInfo = this->getEntryPlain(i,genInfCol);
        s->setTagString(".xsp.generator",generatorInfo);
      }
      if(simInfCol > 0){
        TString simulationInfo = this->getEntryPlain(i,simInfCol);
        s->setTagString(".xsp.simulation",simulationInfo);
      }
      if(treeNameCol > 0){
        s->setTagString(".xsp.treename",pathVariant->replaceInText(this->getEntryPlain(i,treeNameCol)));
      } else if(this->getTagString("treeName",treeName)){
        s->setTagString(".xsp.treename",pathVariant->replaceInText(treeName));
      }
      if(sowCol > 0){
        TString sumOfWeightsInfo(this->getEntryPlain(i,sowCol,false));
        if(!sumOfWeightsInfo.IsNull()){
          double sumOfWeightsPerEvent = this->getEntryDouble(i,sowCol); //don't use the string representation as getEntryDouble has a more robust string->double conversion
          DEBUGclass("setting tag averageWeight=%f",sumOfWeightsPerEvent);
          s->setTagDouble(".xsp.averageWeight",sumOfWeightsPerEvent);
        }
      } 

      DEBUGclass("treeName = %s", s->getTagStringDefault(".xsp.treename", "").Data());
      TString filepath;
      if(this->getTagString("fileNamePattern",filepath)){
        s->setTagString(".xsp.filepath",filepath);
      }
      DEBUGclass("adding sample");
      if(!sf->addObject(s)){
        ERRORclass("unable to add sample '%s' to folder '%s' - this should never have happened!",samplename.Data(),sf->GetName());
        delete s;
        continue;
      }
      retval++;
      DEBUGclass("done adding sample '%s' at path '%s'",samplename.Data(),sf->getPath().Data());
    }
  }
  return retval;
}

int TQXSecParser::addAllSamplesFromPath(const TString& filesystempath, const TString& folderpath, const TString& namefilter, const TString& pathfilter, const TString& tagstring){
  // search a file system path for all ntuples and add them to the sample folder
  TQFolder* f = TQFolder::copyDirectoryStructure(filesystempath);
  int retval = this->addAllSamplesFromFolder(f,folderpath,namefilter,pathfilter,tagstring);
  delete f;
  return retval;
}

int TQXSecParser::addAllSamplesFromList(const TString& inputfile, const TString& folderpath, const TString& tagstring){
  // add the contents of a text file as samples to a sample folder
  TQSampleFolder* sf =this->fSampleFolder->getSampleFolder(folderpath+"+");
  if(!sf){
    ERRORclass("cannot create sample folder at path '%s'",folderpath.Data());
    return -1;
  }
  TString treeName;
  bool hasTreeName = this->getTagString("treeName",treeName);
  std::vector<TString> lines;
  int retval = 0;
  if(!TQStringUtils::readFileLines(&lines,inputfile,1024,true)) return -1;
  for(auto line:lines){
    TString path = TQStringUtils::makeASCII(line);
    DEBUGclass("adding file %s",path.Data());
    const TString fname = TQFolder::getPathTail(path);
    TString name(fname);
    TQStringUtils::removeTrailing(name,".123456789");
    TQStringUtils::removeTrailingText(name,".root");
    TQSample* s = new TQSample(TQFolder::makeValidIdentifier(name));
    TString fileName = TQFolder::concatPaths(path,fname);
    s->setTagString(".xsp.filepath", fileName);
    if(hasTreeName){
      s->setTagString(".xsp.treename",treeName);
    }
    s->importTags(tagstring);
    sf->addObject(s);
    retval++;
  }
  return retval;
}


int TQXSecParser::addAllSamplesFromFolder(TQFolder* f, const TString& folderpath, const TString& namefilter, const TString& pathfilter, const TString& tagstring){
  // search a file system path for all ntuples and add them to the sample folder
  if(!f) return -1;
  TList* l = f->getObjectPaths(namefilter,pathfilter,TObjString::Class());
  TQStringIterator itr(l,true);
  int retval = 0;
  TString treeName;
  TQTaggable tags(tagstring);
#ifdef _DEBUG_
  DEBUGclass("adding all samples from folder '%s'",f->GetName());
  f->print("rdt");
#endif
  while(itr.hasNext()){
    TObjString* str = itr.readNext();
    if(!str) continue;
    TString path(str->GetName());
    path = TQStringUtils::makeASCII(path);
    const TString fname = TQFolder::getPathTail(path);
    TString name(fname);
    TQStringUtils::removeTrailingText(name,".root");
    TQSampleFolder* sf =this->fSampleFolder->getSampleFolder(folderpath+"+");
    if(!sf){
      ERRORclass("cannot create sample folder at path '%s', skipping",folderpath.Data());
      continue;
    }
    TQSample* s = new TQSample(TQFolder::makeValidIdentifier(name));
    TString filePath = TQFolder::concatPaths(path,fname);
    s->setTagString(".xsp.filepath",filePath);
    if(this->getTagString("treeName",treeName)){
      s->setTagString(".xsp.treename",treeName);
    }
    double fileSizeInMB = TQUtils::getFileSize(filePath)/1e6;
    if (fileSizeInMB>0.) {
      s->setTagDouble(".xsp.fileSize",fileSizeInMB);
    }
    
    s->importTags(tagstring);
    sf->addObject(s);
    retval++;
  }
  return retval;
}
 
bool TQXSecParser::isGood(){
  // return true if this reader was properly initialized and is good to go
  if(this->getNcols() < 2) return false;
  if(!this->fSampleFolder) return false;
  return true;
}

//__________________________________________________________________________________|___________


TQXSecParser::Unit TQXSecParser::unit(const TString& in) {
  // search a string for a cross section unit
  if(in.Contains("mb")) return TQXSecParser::Units::millibarn;
  if(in.Contains("µb")) return TQXSecParser::Units::microbarn;
  if(in.Contains("nb")) return TQXSecParser::Units::nanobarn;
  if(in.Contains("pb")) return TQXSecParser::Units::picobarn;
  if(in.Contains("fb")) return TQXSecParser::Units::femtobarn;
  if(in.Contains("ab")) return TQXSecParser::Units::attobarn;
  return TQXSecParser::Units::UNKNOWN;
}

//__________________________________________________________________________________|___________

TString TQXSecParser::unitName(TQXSecParser::Unit in){
  // convert a unit into a string
  if(in == TQXSecParser::Units::millibarn) return "mb";
  if(in == TQXSecParser::Units::microbarn) return "µb";
  if(in == TQXSecParser::Units::nanobarn ) return "nb";
  if(in == TQXSecParser::Units::picobarn ) return "pb";
  if(in == TQXSecParser::Units::femtobarn) return "fb";
  if(in == TQXSecParser::Units::attobarn ) return "ab";
  return "";
}

//__________________________________________________________________________________|___________

double TQXSecParser::convertUnit(double in, const TString& inUnit, const TString& outUnit) {
  // convert a cross section value from one unit to the other
  return TQXSecParser::convertUnit(in,TQXSecParser::unit(inUnit),TQXSecParser::unit(outUnit));
}

//__________________________________________________________________________________|___________

double TQXSecParser::convertUnit(double in, TQXSecParser::Unit inUnit, TQXSecParser::Unit outUnit) {
  // convert a cross section value from one unit to the other
  return in * inUnit / outUnit;
}

int TQXSecParser::readFilePatternFromColumn(const TString& colname){
  // copy paths from the given column
  int colidx = this->findColumn(colname);
  if(colidx < 0) return -1;
  this->setColAlign(colidx,"l");
  int count = 0;
  for(size_t i=1; i<this->nrows; i++){
    TString path = this->getEntryVerbatim(i,colidx);
    if(path.IsNull()) continue;

    TQTaggable* entry = this->getEntryInternal(i,0);
    if(entry && !entry->getTagBoolDefault("enabled",false)) continue; // avoid that the sample get's enabled if it wasn't before, e.g. because of low priority
    this->setProperty(i,0,"matchingName",path);
    this->setProperty(i,0,"enabled",true);
    this->setProperty(i,0,"bold",true);
    count++;
  }
  return count;
}
