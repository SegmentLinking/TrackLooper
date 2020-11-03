//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQTaggable__
#define __TQTaggable__

#include "TList.h"
#include "TString.h"
#include "QFramework/TQValue.h"
#include "TObject.h"

class TQTaggable {

protected:

  TList * fTags;
  mutable TList * fReadFlags;
 
  bool fGlobalIgnoreCase;//!
  bool fGlobalOverwrite;//!
 
  int setTag(TQValue * tag, const TString& destination, bool overwrite);
  inline int setTag(TQValue * tag, const TString& destination = ""){
    return this->setTag(tag,destination,fGlobalOverwrite);
  }

  TQValue * findTag(TString name);
 
  bool getTag(const TString& key, TQValue * &tag);
 
  bool getOp(const TString& op, int &opCode);
 
  int countTagUp (const TString& key);
  int countTagDown(const TString& key);
 
  enum {
    kOpNone = 0,
    kOpRec = 1,
    kOpOR = 2,
    kOpAND = 3,
    kOpADD = 4,
    kOpMULT = 5,
    kOpCNT = 6
  };

  TList * getListOfTags();

public:

  TList* getListOfTagNames();

  static TList * makeListOfTags(TList* unTags);

  static const TString& getValidKeyCharacters();
  static bool isValidKey(const TString& key);

  static TQTaggable * parseFlags(const TString& flags);

  static TQTaggable * parseParameterList(const TString& parameter, const TString& sep = ",", bool trim = true, const TString& blocks = "", const TString& quotes = "");

  bool parseKey(TString key, TString &bareKey, int &opUp, int &opDown);


  TQTaggable();
  TQTaggable(const TString& tags);
  TQTaggable(const char* tags);
  TQTaggable(TQTaggable * tags);
  TQTaggable(const TQTaggable& tags);

  void setGlobalIgnoreCase(bool globalIgnoreCase = true);
  bool getGlobalIgnoreCase() const;

  void setGlobalOverwrite(bool globalOverwrite = true);
  bool getGlobalOverwrite() const;

  virtual void resetReadFlags();
  virtual bool hasUnreadKeys(const TString& filter = "");
  virtual TList * getListOfUnreadKeys(const TString& filter = "");

  virtual TString getFlags();
 
  virtual void onAccess(TQValue * tag);
  virtual void onRead(TQValue * tag);
  virtual void onWrite(TQValue * tag);

  virtual TQTaggable * getBaseTaggable() const;
  virtual TList * getDescendantTaggables();
  virtual TList * getTaggablesByName(const TString& name);
  virtual TList * getListOfTaggables(const TString& name);

  int getNTags();
  void printTags(TString options = "r");

  TList * getListOfKeys(const TString& filter = "");

  bool tagsAreEquivalentTo(TQTaggable * tags, const TString& filter = "");
  bool printDiffOfTags(TQTaggable * tags, const TString& options = "");
  bool printDiffOfTags(TQTaggable * tags, TQTaggable& options);
 
  int removeTag(const TString& key);
  int removeTags(const TString& key);
  void clear();
  int clearTags();

  bool renameTag(const TString& oldKey, const TString& newKey);
  int renameTags(const TString& oldPrefix, const TString& newPrefix);

  /* ----- write tags ----- */

  int importTag(TString tag, bool overwrite = true, bool keepStringQuotes = false);
  int importTags(TString tags, bool overwrite = true, bool keepStringQuotes = false);
  int importTags(const TQTaggable * tags, bool overwrite = true, bool recursive = false);
  int importTags(const TQTaggable& tags, bool overwrite = true, bool recursive = false);

  int importTagWithPrefix (const TString& tag, const TString& prefix, bool overwrite = true, TString fallbackKey = "", bool keepStringQuotes = false);
  int importTagsWithPrefix(TString tags, const TString& prefix, bool overwrite = true, bool keepStringQuotes = false);
  int importTagsWithPrefix(const TQTaggable* tags, const TString& prefix, bool overwrite = true, bool recursive = false);
  int importTagsWithPrefix(const TQTaggable& tags, const TString& prefix, bool overwrite = true, bool recursive = false);
  int importTagsWithoutPrefix(const TQTaggable* tags, const TString& prefix, bool overwrite = true, bool recursive = false);
  int importTagsWithoutPrefix(const TQTaggable& tags, const TString& prefix, bool overwrite = true, bool recursive = false);

  int setTagAuto(const TString& key, TString value, const TString& destination = "");
 
  /* these four methods do the same ... */
  int setTag (const TString& key, double value, const TString& destination = "");
  int setTag (const TString& key, int value, const TString& destination = "");
  int setTag (const TString& key, bool value, const TString& destination = "");
  int setTag (const TString& key, const TString& value, const TString& destination = "");
  int setTag (const TString& key, const char* value, const TString& destination = "");

  /* ... like these four methods */
  int setTagDouble  (TString key, double value, const TString& destination = "");
  int setTagInteger (TString key, int value, const TString& destination = "");
  int setTagBool    (TString key, bool value, const TString& destination = "");
  int setTagString  (TString key, const TString& value, const TString& destination = "");

  // this one is a little different
  int setTagList (const TString& key, TString value, const TString& destination = "");
  template<class T> int setTagList (const TString& key, const std::vector<T>& list, const TString& destination = "");

  /* ----- check tags ----- */

  int printClaim(const TString& definition);
  int claimTags(const TString& definition, bool printErrMsg = false);
  int claimTags(const TString& definition, TString& message);
  int claimTags(const TString& definition, TString& missing, TString& invalid, TString& unexpected);

  bool hasTag (const TString& key);
  bool hasTagDouble (const TString& key);
  bool hasTagInteger (const TString& key);
  bool hasTagBool (const TString& key);
  bool hasTagString (const TString& key);
  bool hasMatchingTag(const TString& name);

  bool tagIsOfTypeDouble (const TString& key);
  bool tagIsOfTypeInteger (const TString& key);
  bool tagIsOfTypeBool (const TString& key);
  bool tagIsOfTypeString (const TString& key);

  bool allTagsValidDoubles();
  bool allTagsValidIntegers();
  bool allTagsValidBools();

  bool allTagsOfTypeDouble();
  bool allTagsOfTypeInteger();
  bool allTagsOfTypeBool();
  bool allTagsOfTypeString();

  /* ----- read tags ----- */

  int exportTags(TQTaggable * dest, const TString& subDest = "", const TString& filter = "", bool recursive = false);
  TString exportTagsAsString(const TString& filter = "", bool xmlStyle = false);
  TString exportTagsAsConfigString(const TString& prefix, const TString& filter = "");
  std::string exportTagsAsStandardString(const TString& filter = "", bool xmlStyle = false);
  std::string exportTagsAsStandardConfigString(const TString& prefix, const TString& filter = "");

  TString replaceInTextRecursive(TString in, const TString& prefix = "", bool keepQuotes = false);
  TString replaceInText(const TString& in, const char* prefix, bool keepQuotes = false);
  TString replaceInText(const TString& in, const TString& prefix, bool keepQuotes = false);
  TString replaceInText(const TString& in, bool keepQuotes = false);
  TString replaceInText(const TString& in, int &nReplaced, int &nFailed, bool keepQuotes = false);
  TString replaceInText(TString in, int &nReplaced, int &nFailed, const TString& prefix, bool keepQuotes = false);

  std::string replaceInStandardStringRecursive(TString in, const TString& prefix = "", bool keepQuotes = false);
  std::string replaceInStandardString(const TString& in, const char* prefix, bool keepQuotes = false);
  std::string replaceInStandardString(const TString& in, const TString& prefix, bool keepQuotes = false);
  std::string replaceInStandardString(const TString& in, bool keepQuotes = false);
  
  bool getTag (const TString& key, double &value);
  bool getTag (const TString& key, int &value);
  bool getTag (const TString& key, bool &value);
  bool getTag (const TString& key, TString &value);

  bool getTagDouble (const TString& key, double &value);
  bool getTagInteger (const TString& key, int &value);
  bool getTagBool (const TString& key, bool &value);
  bool getTagString (const TString& key, TString &value);

  double  getTagDefault (const TString& key, double defaultVal);
  int     getTagDefault (const TString& key, int defaultVal);
  bool    getTagDefault (const TString& key, bool defaultVal);
  TString getTagDefault (const TString& key, const TString& defaultVal);
  TString getTagDefault (const TString& key, const char* defaultVal);

  double      getTagDoubleDefault (const TString& key, double defaultVal = 0.);
  int         getTagIntegerDefault (const TString& key, int defaultVal = 0);
  bool        getTagBoolDefault (const TString& key, bool defaultVal = false);
  TString     getTagStringDefault (const TString& key, const TString& defaultVal = "");
  std::string getTagStandardStringDefault (const TString& key, const TString& defaultVal = "");

  bool getTagAsString (const TString& key, TString &tag);
  bool getTypeOfTagAsString (const TString& key, TString &type);
  bool getValueOfTagAsString (const TString& key, TString &value);

  int getTag(const TString& key, std::vector<TString >& vec);
  int getTag(const TString& key, std::vector<int >& vec);
  int getTag(const TString& key, std::vector<double>& vec);
  int getTag(const TString& key, std::vector<bool >& vec);

  int getTag(const TString& key, TList* l);

  std::vector<TString > getTagVString (const TString& key);
  std::vector<std::string > getTagVStandardString (const TString& key);
  std::vector<int > getTagVInt (const TString& key);
  std::vector<int > getTagVInteger(const TString& key);
  std::vector<double> getTagVDouble (const TString& key);
  std::vector<bool > getTagVBool (const TString& key);

  TString getValuesOfTags(const TString& keys, const TString& sep = ", ");
 
  TList* getTagList(const TString& key);
  int getTagListLength(const TString& key);

  virtual ~TQTaggable();
 
  bool getTag (const TString& key, double &value, bool recursive);
  bool getTag (const TString& key, int &value, bool recursive);
  bool getTag (const TString& key, bool &value, bool recursive);
  bool getTag (const TString& key, TString &value, bool recursive);

  bool getTagDouble (const TString& key, double &value, bool recursive);
  bool getTagInteger (const TString& key, int &value, bool recursive);
  bool getTagBool (const TString& key, bool &value, bool recursive);
  bool getTagString (const TString& key, TString &value, bool recursive);

  TQTaggable& operator=(const TString& s){
    this->importTags(s);
    return *this;
  }
  TQTaggable& operator=(const char* s){
    TString str(s);
    this->importTags(str);
    return *this;
  }

  TQTaggable& operator=( const TQTaggable& other ) {
    this->importTags(other);
    return *this;
  }

  bool exportConfigFile(const TString& filename, const TString& prefix, bool writeUnreadKeys = true);
  bool exportConfigFile(const TString& filename, bool writeUnreadKeys = true);
  bool exportConfigFile(bool writeUnreadKeys = true);
  
  int replaceInTags(TQTaggable& params, const TString& tagFilter = "*");

  ClassDef(TQTaggable, 5); // storage class for meta-information

};

#endif
