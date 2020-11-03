#include "QFramework/TQConfigReader.h"
#include "QFramework/TQIterator.h"
#include "QFramework/TQUtils.h"
#include "TEnv.h"
#include "TOrdCollection.h"
#include "TObjArray.h"
#include "TString.h"
#include "TObjString.h"
#include "THashList.h"
#include "TDirectory.h"
#include "TSystem.h"
#include "QFramework/TQLibrary.h"

#include <memory>
#include <iostream>
#include <set>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include "QFramework/TQStringUtils.h"

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQConfigReader
//
// The TQConfigReader is essentially a wrapper for the TEnv class
// allowing to read config-files and environment variables
// 
// The values may either be retrieved directly
// or exported to an incance of the TQTaggable class
//
// Please note that due to restrictions of th TEnv class,
// only one TQConfigReader may be instantiated at any time
//
// To use it, just instantiate an object of this type. All lookups
// will be prepended with the 'name' given.
//
// TQConfigReader config("MyTest"); TString value =
// config.get("MyValue","MyDefault");
//
// will look up 'Mytest.MyValue: ...' in the configuration file.
//
////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

std::vector<TString> getIncludesRecursive(const TString& filename) {
  // Crawls the includes tree and returns a vector of names of included files.
  // Resolution order is depth-first and then in the order of inclusion.
  // Does not guard against double includes, but double including should not have a negative effect
  //   beyond wasting a bit of CPU time.
  if(gSystem->AccessPathName(filename.Data(), kReadPermission)) {
    throw std::runtime_error(TString("Include file does not exist: " + filename).Data());
  }
  TEnv env;
  env.ReadFile(filename.Data(), kEnvChange);

  auto includes_str = TString(env.GetValue("config.Include", ""));
  auto result = std::vector<TString>();

  // Iterates over filenames to include and call this function on them, recursively appending their
  //   includes
  auto includes = std::unique_ptr<TObjArray>(includes_str.Tokenize(" ,\t"));
  auto iter = TIter(includes.get());
  while(auto p = static_cast<TObjString*>(iter.Next())) {
    auto include_filename = p->GetString();
    auto sub_includes = getIncludesRecursive(include_filename);
    result.insert(
        result.end(),
        std::make_move_iterator(sub_includes.begin()),
        std::make_move_iterator(sub_includes.end())
    );
  }
  // Finally, adds this file itself to the includes
  result.push_back(filename);
  return result;
}

}

TQConfigReader::TQConfigReader(const TString& name, const TString& filepath): _env(NULL), _name(name), _filename(filepath)
{
  // constructor taking a name and filename
  // the name identifier will act as a prefix to keys in the config file
  // the filepath identifies the path of the config file to be read
}

TQConfigReader::~TQConfigReader(){
  // if(this->_env) delete this->_env; // <- TODO: find out why this does not work
  // default destructor
}

bool TQConfigReader::isValid(){
	// check if the given configuration is valid, that is, readable
  if(this->getEnv()) return true;
  return false;
}

TString TQConfigReader::replace(TEnv *env, const TString& input){
  // replace all instances of '%{key}' with 'value' in the given input string
  // keys and values will be taken from the given TEnv
  std::string result(input.Data());
 
  std::string::size_type pos = result.rfind("%{");
  if(pos != std::string::npos) {
    std::string::size_type end = result.find('}', pos);
    if(end != std::string::npos && result.find('.', pos) < end) {
      std::string newkey = result.substr(pos + 2, end - pos - 2);
      s_accessed.insert(newkey);
      std::string newvalue = env->GetValue(newkey.c_str(), "");
      if (newvalue == "") { 
        throw std::runtime_error("No value for " + newkey + " defined in the config files."); 
      }
      result.replace(pos, end - pos + 1, newvalue);
 
      // recursive replace if the new std::string doesn't point back to itself.
      if ( newvalue.find(newkey) == std::string::npos ) {
        result = replace(env, result);
      }
    }
    pos = end;
  }
  return result; 
}

TQTaggable* TQConfigReader::exportAsTags(){
  // export all variables as a TQTaggable object
  if(!this->isValid()) return NULL;
  TQTaggable* tags = new TQTaggable();
  this->exportAsTags(tags);
  return tags;
}

bool TQConfigReader::exportAsTags(TQTaggable* tags){
  // export all variables to a TQTaggable object
  if (!this->getEnv()) return false;
  TQIterator itr(_env->GetTable());
  //@tag: [.filename,.configname] These argument tags are set by TQConfigReader when exporting to a TQTaggable object. They contain the filename of the configuration file and the name of the config reader.
  tags->setTagString(".filename",this->getFilename());
  tags->setTagString(".configname",this->getName());


  while (itr.hasNext()){
    TEnvRec *er = dynamic_cast<TEnvRec*>(itr.readNext());
    if(!er) continue;
    TString key = er->GetName();
    TString value = er->GetValue();
    if(TQStringUtils::removeLeadingText(key,this->_name+".")){
      tags->setTagAuto(key,value);
    }
  }
  return true;
}

TString TQConfigReader::get(const TString& key, const TString& def){
  // retrieve a string-valued variable with the given fallback value
  if(!this->getEnv()){
    return def;
  }
  return get(key, def.Data());
}

TString TQConfigReader::get(const TString& key, const char *def){
  // retrieve a string-valued variable with the given fallback value
  TString full = _name + '.' + key;
  s_accessed.insert(full);
  if(!getEnv()){
    return TString(def);
  }
  return TQStringUtils::trim(this->_env->GetValue(full, def));
}

int TQConfigReader::get(const TString& key, int def){
  // retrieve an integer-valued variable with the given fallback value
  TString full = _name + '.' + key;
  s_accessed.insert(full);
  if(!getEnv())
    return def;
  return this->_env->GetValue(full, def);
}

double TQConfigReader::get(const TString& key, double def){
  // retrieve an double-valued variable with the given fallback value
  TString full = _name + '.' + key;
  s_accessed.insert(full);
  if(!getEnv())
    return def;
  return this->_env->GetValue(full, def); 
}

bool TQConfigReader::get(const TString& key, bool def){
  // retrieve an boolean variable with the given fallback value
  TString full = _name + '.' + key;
  s_accessed.insert(full);
  if(!getEnv())
    return def;
  return this->_env->GetValue(full, def);
}

std::vector<TString> TQConfigReader::getVString(const TString& key, const TString& delim){
  // retrieve a list of strings from the given variable, using the given delimiter
  s_accessed.insert(_name + '.' + key);
 
  TString s(get(key, "").Data());
  std::vector<TString> result;
  TObjArray *tokens = s.Tokenize(delim.Data());
  {
    TIter iter(tokens);
    while(TObject *p = iter.Next()) {
      TObjString *item = (TObjString*)p;
      result.push_back(TQStringUtils::trim(item->GetString()));
    }
  }
  delete tokens;
  return result;
}


std::vector<float> TQConfigReader::getVFloat(const TString& key, const TString& delim){
  // retrieve a list of floating point numbers from the given variable, using the given delimiter
  s_accessed.insert(_name + '.' + key);
 
  TString s(get(key, "").Data());
  std::vector<float> result;
  TObjArray *tokens = s.Tokenize(delim.Data());
  TIter iter(tokens);
  while(TObject *p = iter.Next()) {
    TObjString *item = (TObjString*)p;
    result.push_back(atof(item->GetString().Data()));
  }
  delete tokens;
  return result;
}

std::vector<double> TQConfigReader::getVDouble(const TString& key, const TString& delim){
  // retrieve a list of double-precision floating point numbers from the given variable, using the given delimiter
  s_accessed.insert(_name + '.' + key);
 
  TString s(get(key, "").Data());
  std::vector<double> result;
  TObjArray *tokens = s.Tokenize(delim.Data());
  TIter iter(tokens);
  while(TObject *p = iter.Next()) {
    TObjString *item = (TObjString*)p;
    result.push_back(atof(item->GetString().Data()));
  }
  delete tokens;
  return result;
}

std::vector<int> TQConfigReader::getVInt(const TString& key, const TString& delim){
  // retrieve a list of integer numbers from the given variable, using the given delimiter
  s_accessed.insert(_name + '.' + key);
 
  TString s(get(key, "").Data());
  std::vector<int> result;
  TObjArray *tokens = s.Tokenize(delim.Data());
  TIter iter(tokens);
  while(TObject *p = iter.Next()) {
    TObjString *item = (TObjString*)p;
    result.push_back(strtol(item->GetString().Data(),0,0));
  }
  delete tokens;
  return result;
}

void TQConfigReader::set(const TString& key, const TString& value){
  // set a variable to the given value
  if(this->getEnv()) this->_env->SetValue(key.Data(), value.Data());
}


void TQConfigReader::dumpConfiguration(){
  // dump the configuration to stdout
  this->dumpConfiguration(std::cout);
}

void TQConfigReader::dumpConfiguration(std::ostream& stream){
  // dump the current contents to a stream
  if(!this->getEnv()) return;
  TIter next(this->_env->GetTable());
  TEnvRec *er;
  static const char *lc[] = { "Global", "User", "Local", "Changed" };
 
  stream << "TQConfigReader: Dumping configuration settings" << std::endl;
  while ((er = (TEnvRec*) next()))
    stream << Form("%-25s: %-30s [%s]",
                   er->GetName(),er->GetValue(),lc[er->GetLevel()]) << std::endl;
  stream << "TQConfigReader: End of configuration settings" << std::endl;
 
  return;
}

void TQConfigReader::dumpConfiguration(TDirectory* const dir, TString const name){
  // dump the current contents to a directory
  if (dir && this->getEnv()) {
    TObjArray array(_env->GetTable()->GetSize());
    array.SetOwner(kTRUE);
 
    TIter next(_env->GetTable());
    TEnvRec *er;
    static const char *lc[] = { "Global", "User", "Local", "Changed" };
 
    size_t index = 0;
    while ((er = (TEnvRec*) next())) {
      TObjString* str = new TObjString(Form("%-25s: %-30s [%s]",
                                            er->GetName(),
                                            er->GetValue(),
                                            lc[er->GetLevel()]));
      array[index] = str;
      ++index;
    } 
 
    dir->cd();
    array.Write(name.Data(),TObject::kSingleKey);
  }
 
  return;
}

namespace {
  std::unique_ptr<TEnv> _cleanup;
}

void TQConfigReader::overrideInclude (TString& includePaths) {
// If a non-empty string is passed, overrides the config.Include parameter with the given string when a file is read.
// This method must be called after construction of TQConfigReader but before any other method is called!
  _includePaths = includePaths;
}

const TString& TQConfigReader::getFilename(){
  // retrieve the filename (filepath)
  return this->_filename;
}

const TString& TQConfigReader::getName(){
  // retrieve the object's name
  return this->_name;
}

TEnv* TQConfigReader::getEnv(){
  // retrieve the internal TEnv object
  if(this->_env != NULL) 
    return this->_env;

  // in case we just want a dummy TEnv
  if (this->_filename=="") {
	this->_env = new TEnv(".tmp.invalid");
	_cleanup = std::unique_ptr<TEnv>(_env);
	return this->_env;
  }

  bool ok = TQUtils::fileExists(this->_filename);
  if(!ok) return NULL;
  TQUtils::ensureTrailingNewline(this->_filename.Data());
  this->_env = new TEnv(".tmp.invalid");
  if(_env->ReadFile(this->_filename.Data(), kEnvLocal) < 0){
    delete this->_env;
    std::cout << "unable to open file " << this->_filename.Data() << std::endl;
    delete this->_env;
    this->_env = NULL;
    return NULL;
  }
  if(_env->GetTable()->GetSize() == 0) {
    std::cout << TQStringUtils::makeBoldRed("ERROR in TQConfigReader::getEnv()") << " : unable to obtain HashTable (size=0). Is your input file " << TQStringUtils::makeBoldWhite(this->_filename) << " valid?" << std::endl;
    delete this->_env;
    this->_env = NULL;
    return NULL;
  }
 
  auto files_to_include = getIncludesRecursive(_filename);
  for (const auto& filename: files_to_include) {
    _env->ReadFile(filename, kEnvChange);
    // Can't use msgStream since there's no call to ClassImp for this class
    // Any particular reason why not?
    std::cout << "Including file '" << filename.Data() << std::endl;
  }
 
  s_accessed.insert("config.Include");
  _cleanup = std::unique_ptr<TEnv>(_env);

  return this->_env; 
}

void TQConfigReader::resolve(){
  /// Re-run the variable expansion process.
  ///
  /// Normally this is done automatically when all input files have 
  /// been read, so you don't need to call this. 
  if(!this->getEnv()) return;
  TIter next(this->_env->GetTable());
  while(TEnvRec *rec = (TEnvRec *)next()) {
    const char *str = rec->GetValue();
    if(strstr(str, "%{")) {
      TString new_value = replace(this->_env, str);
      this->_env->SetValue(rec->GetName(), new_value.Data());
    }
    // remove space in the end of the config parameter
    std::string st = rec->GetValue() ;
    if (st.rfind(" ") == st.size()-1) {
      std::string::size_type pos ;
      while((pos = st.rfind(" ")) == st.size()-1 && st.size() > 0) 
        st.erase(pos) ;
      this->_env->SetValue(rec->GetName(), st.c_str());
    }
  }
}

std::vector<TString> TQConfigReader::checkConfiguration(){
  // Check that every processor P has accessed all the
  // variables of the form 'P.X: ...'
  // 
  // Report any variables that have not been accessed.
  std::set<TString> processors;
  // Pass 1, find all Processor 'P'
  for(std::set<TString>::iterator it = s_accessed.begin();
      it != s_accessed.end();
      ++it) {
    int dot = (*it).First('.');
    if(dot != kNPOS) {
      TString processor_name = (*it)(0, dot);
      processors.insert(processor_name);
 
    } // else ignore anything without a '.'
  }
  // Pass 2, 
  std::vector<TString> result;
  if(!this->getEnv())
    return result;
  TIter next(this->_env->GetTable());
  while (TEnvRec *rec = (TEnvRec*) next()) {
    TString name(rec->GetName());
    int dot = name.First('.');
    if(dot != kNPOS) {
      if(processors.find(name(0, dot)) != processors.end()) {
        if(s_accessed.find(name) == s_accessed.end()) {
          result.push_back(name);
        }
      }
    }
  }
 
  return result;
}


std::vector<TString> TQConfigReader::variables(const TString& name) {
  // return list of all variables specified for the processor with name "name"
  std::vector<TString> result;
  if(!this->getEnv())
    return result;
  TIter next(this->_env->GetTable());
  while (TEnvRec *rec = (TEnvRec*) next()) {
    TString pname(rec->GetName());
    int dot = pname.First('.');
    if(dot != kNPOS && name == pname(0, dot))
      result.push_back(pname(dot+1,kNPOS));
  } 
  return result ;
}
