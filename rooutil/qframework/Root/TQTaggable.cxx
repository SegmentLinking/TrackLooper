#include "QFramework/TQTaggable.h"
#include "TString.h"
#include "TNamed.h"
#include "TClass.h"
#include "TList.h"
#include "TParameter.h"
#include "TIterator.h"
#include "TCollection.h"
#include "QFramework/TQStringUtils.h"
#include "QFramework/TQListUtils.h"
#include "QFramework/TQUtils.h"
#include "QFramework/TQIterator.h"
#include "QFramework/TQFolder.h"

#include "QFramework/TQLibrary.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQTaggable: 
//
// The TQTaggable class represents a set of tags and introduces a methods to read and write
// tags from and to it. A tag is a key-value-pair, where the key is a unique string and the
// value might be a string, a bool, an integer or a double.
//
// Tags are set using either of the methods:
//
// - TQTaggable::setTagString("<key>", "<value>")
// - TQTaggable::setTagInteger("<key>", <value>)
// - TQTaggable::setTagDouble("<key>", <value>)
// - TQTaggable::setTagBool("<key>", <value>)
//
// To retrieve the value of a tag one can use
//
// - TQTaggable::getTagStringDefault("<key>", "<default>")
// - TQTaggable::getTagIntegerDefault("<key>", <default>)
// - TQTaggable::getTagDoubleDefault("<key>", <default>)
// - TQTaggable::getTagBoolDefault("<key>", <default>)
//
// where the default value <default> is used if the tag is not present or
//
// - TQTaggable::getTagString("<key>", <value>)
// - TQTaggable::getTagInteger("<key>", <value>)
// - TQTaggable::getTagDouble("<key>", <value>)
// - TQTaggable::getTagBool("<key>", <value>)
//
// where <value> is a variable set to the value of the tag if it is present (return value of
// corresponding function indicates if tag is present). In each case data types are converted
// if necessary and possible.
//
// The TQTaggable class is prepared to allow for an hierarchical tree-like structure of
// instances of TQTaggable (if implemented by a descendant class).
//
// Some special symbols allow to modify the way in which tags are retrieved:
// "~tag" search for the first occurence of tag 'tag' upwards the tree
// "tag~" search for the first occurence of tag 'tag' downwards
//
// Some special symbols allow to modify the way in which tags are set:
// "tag?" only set the tag if it does not exist yet, i.e. don't overwrite
//
// Renaming/removing tags can be done using
//
// - TQTaggable::removeTag(...) remove one tag
//
// - TQTaggable::clearTags() remove all tags
//
// - TQTaggable::renameTag(...) rename one tag
//
// - TQTaggable::renameTags(...) rename all tags with certain key prefix
//
//
// The presence (or absence) and the validity (in terms of data types and simple numerical
// requirements) of tags may be tested using
//
// - TQTaggable::claimTags(...) test validity of tags
//
//
// Useful string parsing utilities provided by the TQTaggable class are
//
// - TQTaggable::importTags(...) parses a string with comma-separated key-value
//  assignments, e.g. "hello = world, x = 5"
//
// - TQTaggable::parseParameterList(...) parses a string with (not necessarily comma-)
//  separated values, e.g. "'test', 4, 1"
//
// - TQTaggable::parseFlags(...) parses a string with flags and optional
//  parameters, e.g. "abc4"
//
// Would be nice to have at some point but has not yet been finalized:
//
// "||key" logical OR of all tags 'key' upwards
// "&&key" logical AND of all tags 'key' upwards
// "key||" logical OR of all tags 'key' downwards
// "key&&" logical AND of all tags 'key' downwards
//
// "+key" SUM of all tags 'key' upwards
// "*key" PRODUCT of all tags 'key' upwards
// "key+" SUM of all tags 'key' downwards
// "key*" PRODUCT of all tags 'key' downwards
//
// "#key" number of occurences of tags 'key' upwards
// "key#" number of occurences of tags 'key' downwards
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQTaggable)


//__________________________________________________________________________________|___________

TQTaggable::TQTaggable() : 
  fTags(0),
  fReadFlags(0),
  fGlobalIgnoreCase(false),
  fGlobalOverwrite(true)
{
  // Creates a new instance of the TQTaggable class without any tags set.
}


//__________________________________________________________________________________|___________

TQTaggable::TQTaggable(const TString& tags) :
  TQTaggable(tags.Data())
{
  // Creates a new instance of TQTaggable and tries to interpret the input string
  // <tags> as comma-separated key-value assignments (see TQTaggable::importTags(...)).
  // In case "--" is prepended the following string will be interpreted as flags
  // (see TQTaggable:parseFlags(...)).
}


//__________________________________________________________________________________|___________

TQTaggable::TQTaggable(const char* str) :
  TQTaggable()
{
  // Creates a new instance of TQTaggable and tries to interpret the input string
  // <tags> as comma-separated key-value assignments (see TQTaggable::importTags(...)).
  // In case "--" is prepended the following string will be interpreted as flags
  // (see TQTaggable:parseFlags(...)).
 
  TString tags(str);

  if (TQStringUtils::removeLeading(tags, "-") == 2) {
    // interpret and import as flags
    TQTaggable * flags = TQTaggable::parseFlags(tags);
    if (flags) {
      flags->exportTags(this);
      delete flags;
    }
  } else {
    // simply import tags
    importTags(tags);
  }
}


//__________________________________________________________________________________|___________

TQTaggable::TQTaggable(TQTaggable * tags) :
  TQTaggable()
{
  // Creates a new instance of TQTaggable and imports tags assigned to <tags> to
  // this instance.
  this->importTags(tags);
}

//__________________________________________________________________________________|___________

TQTaggable::TQTaggable(const TQTaggable& tags) :
  TQTaggable()
{
  // Creates a new instance of TQTaggable and imports tags assigned to <tags> to
  // this instance.
  this->importTags(tags);
}

//__________________________________________________________________________________|___________

TQTaggable * TQTaggable::getBaseTaggable() const {
  // Returns a pointer to the base instance of TQTaggable in a tree of TQTaggables.
  // This method is supposed to be overwritten by a descendant class implementing
  // the management of a tree-like structure of TQTaggables.
 
  // default: no base taggable present
  return 0;
}


//__________________________________________________________________________________|___________

TList * TQTaggable::getDescendantTaggables() {
  // default: no descendant taggables 
  return 0;
}


//__________________________________________________________________________________|___________

TList * TQTaggable::getTaggablesByName(const TString& /*name*/) {
  // default: no taggables around 
  return 0;
}

//__________________________________________________________________________________|___________

TList * TQTaggable::getListOfTaggables(const TString& /*name*/) {
  // default: no taggables around 
  return 0;
}


//__________________________________________________________________________________|___________

const TString& TQTaggable::getValidKeyCharacters() {
  // return the list of all valid key characters
  return TQValue::getValidNameCharacters();
}


//__________________________________________________________________________________|___________

bool TQTaggable::isValidKey(const TString& key) {
  // Returns true if <key> is a valid key and false otherwise.
 
  return TQValue::isValidName(key);
}


//__________________________________________________________________________________|___________

void TQTaggable::resetReadFlags() {
  // reset the counting of which flags have been read already
  if (fReadFlags) {
    delete fReadFlags;
  }
  fReadFlags = new TList();
  fReadFlags->SetOwner(true);
}


//__________________________________________________________________________________|___________

bool TQTaggable::hasUnreadKeys(const TString& filter) {
  // get the list of unread keys
  TList * unreadKeys = getListOfUnreadKeys(filter);

  if (unreadKeys) {
    delete unreadKeys;
    return true;
  } else {
    return false;
  }
}


//__________________________________________________________________________________|___________

TList * TQTaggable::getListOfUnreadKeys(const TString& filter) {

  // get the list of all keys matching <filter> and ...
  TList * unreadKeys = getListOfKeys(filter);
  if (!unreadKeys) {
    return NULL;
  }

  // ... remove the flagged ones
  TQIterator itr(fReadFlags);
  while (itr.hasNext()) {
    TObject * tag = unreadKeys->FindObject(itr.readNext()->GetName());
    if (tag) {
      unreadKeys->Remove(tag);
      delete tag;
    }
  }

  if (unreadKeys->GetEntries() == 0) {
    delete unreadKeys;
    unreadKeys = NULL;
  }

  return unreadKeys;
}


//__________________________________________________________________________________|___________

void TQTaggable::onAccess(TQValue * tag) {
}


//__________________________________________________________________________________|___________

void TQTaggable::onRead(TQValue * tag) {

  if (fReadFlags) {
    if (!fReadFlags->FindObject(tag->GetName())) {
      fReadFlags->Add(new TObjString(tag->GetName()));
    }
  }
}


//__________________________________________________________________________________|___________

void TQTaggable::onWrite(TQValue * tag) {
}


//__________________________________________________________________________________|___________

TQTaggable * TQTaggable::parseFlags(const TString& flags) {
  // Reads the input string <flags>, interprets it as a set of flags with optional
  // parameters, and in case of success returns an instance of TQTaggable with tags
  // set corresponding to the flags and its parameters listed in the input string
  // (the user is responsible for deleting the returned instance). Returns a null
  // pointer in case of failure. Please note: this method returns a new instance of
  // TQTaggable even if there are no flags, e.g. if an empty string has been passed,
  // parseFlags(""). A flag is usually represented by a single letter, also allowing
  // for flags with more than one letter. Optionally, the flag might come with an
  // additional parameter, either of type string or integer.
  //
  // "a" occurence of flag "a", resulting in tag "a = true"
  // "a6" flag "a" coming with integer parameter (6), resulting in tag "a = 6"
  // "a[test]" flag "a" coming with string parameter ("test"), resulting in tag
  // "a = 'test'"
  //
  // Boolean flags (simple occurence) can be negated by prepending "!":
  //
  // "!a" negated occurence of flag "a", resulting in tag "a = false"
  //
  // Flags may be made of more than one single letter by enclosing its full name in "<>":
  //
  // "<myFlag>" occurence of flag "myFlag", resulting in tag "myFlag = true"
  //
  // Multiple flags are listed by simple string concatenation:
  //
  // "abc" occurence of flags "a", "b", and "c", resulting in tags "a = true,
  // b = true, c = true"
  // "ab5c[hello]" occurence of flag "a", flag "b" coming with integer parameter (5),
  // and flag "c" coming with string parameter ("hello"), resulting in
  // tags "a = true, b = 5, c = 'hello'"
  // "a-2<flag>5" flag "a" coming with integer parameter (-2) and flag "flag" coming
  // with integer parameter (5), resulting in tags "a = -2, flag = 5"
  //
  // This method is used to parse option flags to several functions of TQx classes.
  //
  // The inverse of TQTaggable::parseFlags(...) is TQTaggable::getFlags().
 
  // input string will be manipulated during parsing
  TString myFlags = flags;
 
  // will be the instance to return
  TQTaggable * tags = new TQTaggable();

  // parse the input string
  bool stop = false;
  while (!myFlags.IsNull() && !stop) {
    // negate a boolean flag?
    bool negate = false;
    if (TQStringUtils::removeLeading(myFlags, "!", 1)) {
      negate = true;
    }
    TString flag;
    // read one flag from head of string
    if (!TQStringUtils::readToken(myFlags, flag, TQStringUtils::getLetters(), 1) &&
        !TQStringUtils::readBlock(myFlags, flag, "<>[]{}", "\"\"''")) {
      stop = true;
      continue;
    }
    if (tags->hasTag(flag)) {
      // multiple occurence of flag => ERROR
      stop = true;
      continue;
    }

    // flag with string parameter
    TString block;
    if (TQStringUtils::readBlock(myFlags, block, "[]<>{}", "\"\"''")) {
      if (negate || !tags->setTagString(flag, block)) {
        // negating string flag or failed to set tag => ERROR
        stop = true;
      }
      continue;
    }

    // flag with numberic parameter
    TString number;
    if (TQStringUtils::readToken(myFlags, number, TQStringUtils::getNumerals() + "-")) {
      if (!TQStringUtils::isInteger(number)) {
        // not a valid integer found after integer flag => ERROR
        stop = true;
        continue;
      }
      if (negate || !tags->setTagInteger(flag, number.Atoi())) {
        // negating integer flag or failed to set tag => ERROR
        stop = true;
      }
      continue;
    }

    // flag without parameter
    tags->setTagBool(flag, !negate);
  }

  if (stop) {
    // cleanup in case there was an error
    delete tags;
    tags = NULL;
  }
 
  // return instance of TQTaggable with tags corresponding to flags
  return tags;
}


//__________________________________________________________________________________|___________

TString TQTaggable::getFlags() {
  // Creates and returns a string representing the tags associated to this instance
  // of TQTaggable as flags (see TQTaggable::parseFlags(...) for detailed information).
 
  // will be the string to return
  TString flags;
 
  // iterate over tags of this instance
  TQIterator itr(this->getListOfKeys(), true);
  while (itr.hasNext()) {
    TString name = itr.readNext()->GetName();
    TString flag;
    if (name.Length() == 1 && TQStringUtils::getLetters().Index(name) != kNPOS) {
      // single letter flag
      flag.Append(name);
    } else {
      // flag not made of one single letter
      flag.Append(TString("<") + name + ">");
    }

    TQValue * val = this->findTag(name);
    if (!val) {
      // should never happen
      continue;
    }
    if (!val->isBool()) {
      if (val->isInteger()) {
        // integer flag
        flag.Append(TString::Format("%d", val->getInteger()));
      } else {
        // string flag (here: also double flag possible)
        flag.Append(TString::Format("[%s]", val->getString().Data()));
      }
    } else {
      // boolean flag
      if (!val->getBool()) {
        // negate if tag is false
        flag.Prepend("!");
      }
    }
 
    // compile final string
    flags.Append(flag);
  }
 
  return flags;
}


//__________________________________________________________________________________|___________

int TQTaggable::setTag(TQValue * tag, const TString& destination, bool overwrite) {

  /* stop if no tag to add was given */
  if (!tag)
    return 0;

  /* if no destination was specified explicitly, add the tag to this taggable */
  if (destination.IsNull()) {

    // make sure the list exists and remove a potentially
    // existing tag with the same name (if overwrite == true)
    if (!fTags) {
      fTags = new TList();
    } else {
      TObject * existing = fTags->FindObject(tag->GetName());
      if (existing) {
        if (!overwrite) {
          // not allowed to overwrite existing tag
          return 0;
        }
        fTags->Remove(existing);
        delete existing;
      }
    }

    /* now add the tag: we add the instance that was passed
     * to this function, thus no need to delete anything */
    this->fTags->Add(tag);
    this->onWrite(tag);

    /* we added one tag */
    return 1;

    /* if a destination was specified we need to get
     * the list of taggables matching the destination */
  } else {

    /* get the list of taggables matching the destination
     * (assuming we have to delete the list returned) */
    TList * taggables = this->getTaggablesByName(destination);

    /* if there is no matching element we are done */
    if (!taggables)
      return 0;

    /* the number of taggables the tag is added to */
    int nTags = 0;

    /* loop over taggables in the list */
    TQIterator itr(taggables);

    while (itr.hasNext()) {
      /* sanity check: we don't expect anything else but instances of
       * the TQTaggable class, but nevertheless make sure this is true */
      TQTaggable* t = dynamic_cast<TQTaggable*>(itr.readNext());
      if(!t) continue;
 
      /* add a copy of the tag to the taggable from the list */
      if (nTags == 0) 
        nTags += t->setTag(tag, "", overwrite);
      else
        nTags += t->setTag(tag->copy(), "", overwrite);

    }

    /* delete the list (but not the objects in the list) */
    delete taggables;

    /* return the number of taggables the tag was added to */
    return nTags;
  }
}


//__________________________________________________________________________________|___________

void TQTaggable::setGlobalIgnoreCase(bool globalIgnoreCase) {

  fGlobalIgnoreCase = globalIgnoreCase;
}


//__________________________________________________________________________________|___________

bool TQTaggable::getGlobalIgnoreCase() const {

  return fGlobalIgnoreCase;
}


//__________________________________________________________________________________|___________

void TQTaggable::setGlobalOverwrite(bool globalOverwrite) {

  fGlobalOverwrite = globalOverwrite;
}


//__________________________________________________________________________________|___________

bool TQTaggable::getGlobalOverwrite() const {
  TQTaggable* base = this->getBaseTaggable();
  return base ? base->getGlobalOverwrite() : fGlobalOverwrite;
}

//__________________________________________________________________________________|___________

TQValue * TQTaggable::findTag(TString name) {

  bool ignoreCase = (TQStringUtils::removeLeading(name, "^", 1) > 0) || fGlobalIgnoreCase;

  // the TQValue object to return
  TQValue * tag = NULL;

  TQIterator itr(fTags);
  while (itr.hasNext() && !tag) {
    TObject * thisObj = itr.readNext();
    // is this check really needed?
    if (!thisObj) {
      continue;
    }
    TString thisName = thisObj->GetName();
    if ((name.CompareTo(thisName, ignoreCase ? TString::kIgnoreCase : TString::kExact) == 0)
        && thisObj->InheritsFrom(TQValue::Class())) {
      tag = (TQValue*)thisObj;
    }
  }

  // NULL if no matching object has been found
  return tag;
}


//__________________________________________________________________________________|___________

bool TQTaggable::hasMatchingTag(const TString& name) {
  TQIterator itr(fTags);
  while (itr.hasNext()) {
    TObject * thisObj = itr.readNext();
    // is this check really needed?
    if (!thisObj) {
      continue;
    }
    if(TQStringUtils::matches(thisObj->GetName(),name)) return true;
  }
  return false;
}


//__________________________________________________________________________________|___________

bool TQTaggable::getTag(const TString& key, TQValue * &tag) {

  TString myKey(key);

  // get tag recursively?
  bool recursiveUp = TQStringUtils::removeLeading(myKey, "~", 1) > 0;
  bool recursiveDown = TQStringUtils::removeTrailing(myKey, "~", 1) > 0;

  // try to find the tag
  TQValue * value = findTag(myKey);

  if (value) {

    /* we found the tag: return it */
    tag = (TQValue*)value;
    onAccess(tag);
    return true;

  } else {

    if (recursiveUp) {
      /* get the base taggable */
      TQTaggable * baseTaggable = getBaseTaggable();
      if (baseTaggable) {
        /* search the base taggable for the tag */
        return baseTaggable->getTag(key, tag);
      }
    }
    
    if (recursiveDown){
      /* get descendant taggables */
      TList * descendantTaggables = this->getDescendantTaggables();
      TQIterator itr(descendantTaggables,true);
      while(itr.hasNext()){
        TQTaggable* child = dynamic_cast<TQTaggable*>(itr.readNext());
        if(!child) continue;
        if(child->getTag(key,tag)){
          DEBUGclass("found tag '%s' with value '%s' on %p",key.Data(),tag->getValueAsString().Data(),child);
          return true;
        } 
      }
    }
    
    /* we couldn't find the tag */
    return false;
  }
}


//__________________________________________________________________________________|___________

TList * TQTaggable::getListOfKeys(const TString& filter) {

  /* the final list to return */
  TList * list = NULL;

  if (!fTags) return NULL;

  TQValueIterator itr(fTags);
  while(itr.hasNext()){
    TQValue* obj = itr.readNext();
    if(!obj) continue;
    if (!filter.IsNull() && !TQStringUtils::matchesFilter(obj->GetName(), filter, ",", true)) {
      continue;
    }
    
    if (!list) {
      /* create the list */
      list = new TList();
      list->SetOwner(true);
    }
    list->Add(new TObjString(obj->GetName()));
  }

  /* return the final list */
  return list;
}


//__________________________________________________________________________________|___________

bool TQTaggable::tagsAreEquivalentTo(TQTaggable * tags, const TString& filter) {
 
  if (!tags) {
    return false;
  }
 
  TList * keys = TQListUtils::getMergedListOfNames(
                                                   this->getListOfKeys(filter), tags->getListOfKeys(filter), true);
 
  TQIterator itr(keys, true);
  while (itr.hasNext()) {
    TString name = itr.readNext()->GetName();
 
    TQValue * myTag = this->findTag(name);
    TQValue * theirTag = tags->findTag(name);
 
    if (!myTag || !theirTag || !myTag->isEquivalentTo(theirTag)) {
      return false;
    }
  }
 
  return true;
}


//__________________________________________________________________________________|___________

bool TQTaggable::printDiffOfTags(TQTaggable * tags, const TString& options) {
 
  // parse options flags
  TQTaggable * opts = TQTaggable::parseFlags(options);
  if (!opts) {
    std::cout << TQStringUtils::makeBoldRed(TString::Format("TQTaggable::printDiffOfTags(...): "
                                                       "Failed to parse options '%s'", options.Data())).Data() << std::endl;
    return false;
  }
 
  bool result = false;
  result = this->printDiffOfTags(tags, *opts);
  delete opts;
  return result;
}


//__________________________________________________________________________________|___________

bool TQTaggable::printDiffOfTags(TQTaggable * tags, TQTaggable& options) {

  const int cColWidth_Name = 50;
  const int cColWidth_Comp = 20;
  const int cColWidth_Details = 30;
 
  if (!tags) {
    return false;
  }
  //@tag:[i] This argument tag sets the indentation when printing a tag diff. Default: 0
  int indent = options.getTagIntegerDefault("i", 0) * 2;
  //@tag:[z] This argument tag enables printing folder diff. Default: false
  bool folderPrintDiff = options.getTagBoolDefault("z", false);
  //@tag:[m] This argument tag enables printing only mismatched in tag diff. Default: false
  bool onlyListMismatches = options.getTagBoolDefault("m", false);
  //@tag:[d] this argument tag enables printing details in tag diff. Default: false
  bool printDetails = options.getTagBoolDefault("d", false);
 
  bool equivalent = true;
 
  TList * keys = TQListUtils::getMergedListOfNames(
                                                   this->getListOfKeys(), tags->getListOfKeys(), true);
 
  TString line;
 
  // print headline
  if (!folderPrintDiff) {
    line = TQStringUtils::fixedWidth("Tag", cColWidth_Name, "l");
    line.Append(TQStringUtils::fixedWidth("Comparison", cColWidth_Comp, "l"));
    if (printDetails) {
      line.Append(TQStringUtils::fixedWidth("Details (1)", cColWidth_Details, "l"));
      line.Append(TQStringUtils::fixedWidth("Details (2)", cColWidth_Details, "l"));
    }
    std::cout << TQStringUtils::makeBoldWhite(line) << std::endl;
    std::cout << TQStringUtils::makeBoldWhite(TQStringUtils::repeat("=", line.Length())) << std::endl;
  }
 
  TQIterator itr(keys, true);
  while (itr.hasNext()) {
    TString name = itr.readNext()->GetName();
 
    TQValue * myTag = this->findTag(name);
    TQValue * theirTag = tags->findTag(name);

    if (folderPrintDiff) {
      line = TQStringUtils::fixedWidth(TQStringUtils::repeatSpaces(indent) +
                                       TString::Format("\033[0;36m<%s>\033[0m", name.Data()), cColWidth_Name, "l");
    } else {
      line = TQStringUtils::fixedWidth(TQStringUtils::repeatSpaces(indent) + name, cColWidth_Name, "l");
    }

    TString comp;
    bool thisEquivalent = true;
    if (myTag && theirTag) {
      if (myTag->isEquivalentTo(theirTag)) {
        comp = TQStringUtils::makeBoldGreen("<1> == <2>");
      } else {
        thisEquivalent = false;
        comp = TQStringUtils::makeBoldRed("<1> != <2>");
      }
    } else if (myTag) {
      thisEquivalent = false;
      comp = TQStringUtils::makeBoldRed("<1> - ");
    } else if (theirTag) {
      thisEquivalent = false;
      comp = TQStringUtils::makeBoldRed(" - <2>");
    }
 
    if (!onlyListMismatches || !thisEquivalent) {
      line.Append(TQStringUtils::fixedWidth(comp, cColWidth_Comp, "l"));
      if (printDetails) {
        TString detail1;
        TString detail2;
        if (myTag) {
          detail1 = myTag->getValueAsString();
        }
        if (theirTag) {
          detail2 = theirTag->getValueAsString();
        }
        line.Append(TQStringUtils::fixedWidth(detail1, cColWidth_Details, "l"));
        line.Append(TQStringUtils::fixedWidth(detail2, cColWidth_Details, "l"));
      }
      std::cout << line.Data() << std::endl;
    }
    if (!thisEquivalent) {
      equivalent = false;
    }
  }
 
  return equivalent;
}


//__________________________________________________________________________________|___________

int TQTaggable::removeTag(const TString& key) {

  // the number of tags removed
  int nTags = 0;

  if (!fTags) {
    // not a single tag present
    return 0;
  }

  // find tag
  TString myKey = key;
  TQValue * tag = findTag(myKey);
  if (tag) {
    // remove and delete tag
    fTags->Remove(tag);
    delete tag;
    nTags++;
  }

  // delete the list itself, if the last tag was removed
  if (fTags->GetEntries() == 0) {
    delete fTags;
    fTags = NULL;
  }

  // return the number of tags removed
  return nTags;
}

//__________________________________________________________________________________|___________

int TQTaggable::removeTags(const TString& key) {
  // remove all tags matching the given key and return their number
  int nTags = 0;

  if (!fTags) {
    // not a single tag present
    return 0;
  }

  // find tag
  TQValueIterator itr(this->fTags);
  while(itr.hasNext()){
    TQValue* tag = itr.readNext();
    if(TQStringUtils::matches(tag->GetName(),key)){
      // remove and delete tag
      fTags->Remove(tag);
      delete tag;
      nTags++;
    }
  }

  // delete the list itself, if the last tag was removed
  if (fTags->GetEntries() == 0) {
    delete fTags;
    fTags = NULL;
  }

  // return the number of tags removed
  return nTags;
}


//__________________________________________________________________________________|___________

bool TQTaggable::renameTag(const TString& oldKey, const TString& newKey) {
  // Renames tag with key <oldKey> to key <newKey> and returns true if renaming was
  // successful and false otherwise. Renaming fails if <newKey> is not a valid key
  // or a tag with key <newKey> already exists.
 
  // check new name
  if (this->hasTag(newKey) || !TQValue::isValidName(newKey)) {
    return false;
  }

  // try to find tag
  TQValue * tag = findTag(oldKey);
  if (tag) {
    // rename tag
    tag->setName(newKey.Data());
    return true;
  } else {
    // couldn't find tag
    return false;
  }
}


//__________________________________________________________________________________|___________

int TQTaggable::renameTags(const TString& oldPrefix, const TString& newPrefix) {
  // Renames all tags with keys that begin with <oldPrefix> replacing <oldPrefix>
  // by <newPrefix> and returns the number of tags that were successfully renamed.
 
  // will be the number of tags renamed
  int n = 0;

  // iterate over all tags present
  TQIterator itr(this->getListOfKeys(), true);
  while (itr.hasNext()) {
    TString key = itr.readNext()->GetName();
    TString oldKey = key;
    // remove old key prefix
    if (!TQStringUtils::removeLeadingText(key, oldPrefix)) {
      // skip if key does not begin with <oldPrefix>
      continue;
    }
    // rename tag
    if (renameTag(oldKey, newPrefix + key)) {
      n++;
    }
  }

  return n;
}

//__________________________________________________________________________________|___________

void TQTaggable::clear() {
  // Removes and deletes all tags of this instance of TQTaggable ansd returns the
  // number of tags that have been removed.
  this->clearTags();
}

//__________________________________________________________________________________|___________

int TQTaggable::clearTags() {
  // Removes and deletes all tags of this instance of TQTaggable ansd returns the
  // number of tags that have been removed.
 
  int n = 0;
 
  // delete all tags
  if (fTags) {
    n = fTags->GetEntries();
    fTags->Delete();
    delete fTags;
    fTags = NULL;
  }

  return n;
}


//__________________________________________________________________________________|___________

int TQTaggable::printClaim(const TString& definition) {
 
  TString missing;
  TString invalid;
  TString unexpected;
 
  int result = claimTags(definition, missing, invalid, unexpected);
 
  std::cout << "Missing : " << missing.Data() << std::endl;
  std::cout << "Invalid : " << invalid.Data() << std::endl;
  std::cout << "Unexpected: " << unexpected.Data() << std::endl;
 
  return result;
}


//__________________________________________________________________________________|___________

int TQTaggable::claimTags(const TString& definition, bool printErrMsg) {
 
  TString errMsg;
 
  int result = claimTags(definition, errMsg);
 
  if (printErrMsg && result < 1) {
    std::cout << "TQTaggable::claimTags(...): " << errMsg.Data() << std::endl;
  }
 
  return result;
}


//__________________________________________________________________________________|___________

int TQTaggable::claimTags(const TString& definition, TString& message) {
 
  if (message.IsNull()) {
    message = "tag";
  }
 
  // dummy strings
  TString missing;
  TString invalid;
  TString unexpected;
 
  int result = claimTags(definition, missing, invalid, unexpected);
 
  if (result < 0) {
    message = "Failed to parse definition";
  } else if (!missing.IsNull()) {
    message = TString::Format("Missing %s '%s'", message.Data(), 
                              TQStringUtils::getFirstToken(missing).Data());
  } else if (!invalid.IsNull()) {
    message = TString::Format("Invalid %s '%s'", message.Data(), 
                              TQStringUtils::getFirstToken(invalid).Data());
  } else if (!unexpected.IsNull()) {
    message = TString::Format("Unexpected %s '%s'", message.Data(), 
                              TQStringUtils::getFirstToken(unexpected).Data());
  }
 
  return result;
}


//__________________________________________________________________________________|___________

int TQTaggable::claimTags(const TString& definition, TString& missing,
                          TString& invalid, TString& unexpected) {
  // Tests the compatibility of the tags associated to this instance of TQTaggable
  // with an expectation encoded in the input string <definition> and returns 1
  // if tags are compatible and 0 otherwise. Basically, <definition> is a 
  // comma-separated list of keys that need to exist as tags. Wildcards "?" and "*"
  // may be used, in which case at least one tag with a key matching the pattern
  // has to exist:
  //
  // - expected tag(s): "key, myTag, style.*"
  // 
  // If a "!" is prepended to a key name a corresponding tag is expected to not be
  // present:
  // 
  // - unaccepted tag: "!key"
  // 
  // Tags may be expected to have a certain data type or to be convertable to
  // certain data types (i = integer, d = double, b = bool, s = string):
  // 
  // - expect data type for tag: "key:d!"
  // - expect tag to be convertable to data types: "key:id"
  // 
  // Here, the "!" appended to a data type (or a list of data types) indicates that
  // the tag has to have one of the listed data types (logical OR). In case no "!" 
  // is appended the tag is required to be convertable to every listed data type
  // (logical AND).
  // 
  // Tags may be marked as optional by enclosing the corresponding part in "[]":
  // 
  // - optional tags: "[key], [style.color.*:i!]"
  // 
  // If "!!" appears as one entry in the comma-separated list <definition> no tag
  // with a key other than the listed ones is expected:
  //
  // - expect only listed keys: "key, myTag, style.*, !!"
  //
  // Numerical tags may be required to fulfill a certain condition, e.g. must be
  // positive (operators "==", "!=", ">=", ">", "<=", and "<" are supported):
  //
  // - Tags must be positive: "myKeys*:d:>0"
  // - Tags must not be zero: "myKeys*:d:!=0"
  //
  // The value -1 is returned in case the definition string could not be parsed
  // properly.
 
  // taggable fulfills claim? (will be set to zero in the course of
  // this function once there is a tag not fulfilling the definition)
  bool fulfilled = true;
 
  // list of keys not listed in claim definition
  TList * unlisted = this->getListOfKeys();
 
  // expect no tags but the ones listed in claim definition
  bool nothingElse = false;
 
  // iterate over (comma-separated) elements in definition
  bool error = false;
  TQIterator itr(TQStringUtils::tokenize(definition, ",", true, "[]"), true);
  while (!error && itr.hasNext()) {
    TString token = itr.readNext()->GetName();

    if (token.CompareTo("!!") == 0) {
      // don't accept unlisted keys
      nothingElse = true;
      continue;
    }
 
    // unaccepted?
    bool unaccepted = TQStringUtils::removeLeading(token, "!", 1);
 
    // optional?
    bool optional = false;
 
    TString block;
    if (TQStringUtils::readBlock(token, block, "[]")) {
      if (unaccepted || !token.IsNull()) {
        // invalid expectations (unaccepted && optional doesn't make sense)
        error = true;
        continue;
      }
      optional = true;
      token = block;
    }
 
    // read key filter (allowing wildcards)
    TString keyFilter;
    if (!TQStringUtils::readToken(token, keyFilter, TQValue::getValidNameCharacters() + "?*")) {
      // invalid expectations (missing key filter)
      error = true;
      continue;
    }
 
    // read type def
    TString typeFilter;
    bool forceType = false;
    if (TQStringUtils::removeLeading(token, ":", 1)) {
      if (!TQStringUtils::readToken(token, typeFilter, "ibds")) {
        // invalid expectation
        error = true;
        continue;
      }
      forceType = (TQStringUtils::removeLeading(token, "!", 1) > 0);
    }
 
    // read numerical constraint
    TString numConstraint;
    if (TQStringUtils::removeLeading(token, ":", 1)) {
      if (TQStringUtils::isEmpty(token, true)) {
        // invalid expectations (expecting constraint definition)
        error = true;
        continue;
      }
      numConstraint = token;
    } else if (!TQStringUtils::isEmpty(token, true)) {
      // invalid expectations (unexpected stuff in definition);
      error = true;
      continue;
    }
 
    bool found = false;
 
    // remove listed keys from list of unlisted keys
    TQListUtils::removeElements(unlisted, keyFilter);
 
    TQIterator itrKeys(this->getListOfKeys(keyFilter), true);
    while (itrKeys.hasNext()) {
      TString key = itrKeys.readNext()->GetName();
      TQValue * tag = this->findTag(key);
      if (!tag) {
        // should not happen
        continue;
      }
 
      // found a key matching key filter
      found = true;
 
      if (unaccepted) {
        // unexpected tag found
        fulfilled = false;
        TQStringUtils::append(unexpected, key, ", ");
        continue;
      }

      if (!typeFilter.IsNull() && ((forceType && (
                                                  // typeFilter lists accepted tag types
                                                  (tag->isInteger() && !typeFilter.Contains("i")) ||
                                                  (tag->isBool() && !typeFilter.Contains("b")) ||
                                                  (tag->isDouble() && !typeFilter.Contains("d")) ||
                                                  (tag->isString() && !typeFilter.Contains("s")))) || 
                                   // typeFilter lists tag types tag needs to be convertable to
                                   ( !forceType && (
                                                    (typeFilter.Contains("i") && !tag->isValidInteger()) ||
                                                    (typeFilter.Contains("b") && !tag->isValidBool()) ||
                                                    (typeFilter.Contains("d") && !tag->isValidDouble()))))) {
 
        // invalid tag type
        fulfilled = false;
        TQStringUtils::append(invalid, key, ", ");
        continue;
      }

      if (!numConstraint.IsNull()) {
        int test = TQStringUtils::testNumber(tag->getDouble(), numConstraint);
        if (test < 0) {
          // invalid expectations (invalid test expression)
          error = true;
          continue;
        } else if (!tag->isValidDouble() || test == 0) {
          // numerically invalid tag
          fulfilled = false;
          TQStringUtils::append(invalid, key, ", ");
          continue;
        }
      }
    }
 
    if (!found && !optional && !unaccepted) {
      // couldn't find expected tag
      fulfilled = false;
      TQStringUtils::append(missing, keyFilter, ", ");
    }
  } 
 
  if (!error) {
    if (nothingElse) {
      if (unlisted && unlisted->GetEntries() > 0) {
        // unexpected tags found
        fulfilled = false;
      }
      TQStringUtils::append(unexpected, TQStringUtils::concat(unlisted, ", "), ", ");
    }
  } 

  if (unlisted) {
    delete unlisted;
  }
 
  if (!error) {
    return fulfilled ? 1 : 0;
  } else {
    return -1;
  }
}


//__________________________________________________________________________________|___________

bool TQTaggable::hasTag(const TString& key) {
 
  /* a dummy (won't be used) */
  TQValue * dummy = 0;

  /* try to find the tag */
  return getTag(key, dummy);
}


//__________________________________________________________________________________|___________

bool TQTaggable::hasTagDouble(const TString& key) {
 
  /* a dummy (won't be used) */
  double dummy = 0.;

  /* try to find the tag */
  return getTagDouble(key, dummy);
}


//__________________________________________________________________________________|___________

bool TQTaggable::hasTagInteger(const TString& key) {
 
  /* a dummy (won't be used) */
  int dummy = 0;

  /* try to find the tag */
  return getTagInteger(key, dummy);
}


//__________________________________________________________________________________|___________

bool TQTaggable::hasTagBool(const TString& key) {
 
  /* a dummy (won't be used) */
  bool dummy = false;

  /* try to find the tag */
  return getTagBool(key, dummy);
}


//__________________________________________________________________________________|___________

bool TQTaggable::hasTagString(const TString& key) {
 
  /* a dummy (won't be used) */
  TString dummy;

  /* try to find the tag */
  return getTagString(key, dummy);
}


//__________________________________________________________________________________|___________

bool TQTaggable::tagIsOfTypeDouble(const TString& key) {

  // a dummy (won't be used)
  TQValue * dummy = NULL;

  // try to find the tag
  getTag(key, dummy);

  // return true if tag exists and is of type double
  return (dummy != NULL) && dummy->isDouble();
}


//__________________________________________________________________________________|___________

bool TQTaggable::tagIsOfTypeInteger(const TString& key) {

  // a dummy (won't be used)
  TQValue * dummy = NULL;

  // try to find the tag
  getTag(key, dummy);

  // return true if tag exists and is of type integer
  return (dummy != NULL) && dummy->isInteger();
}


//__________________________________________________________________________________|___________

bool TQTaggable::tagIsOfTypeBool(const TString& key) {

  // a dummy (won't be used)
  TQValue * dummy = NULL;

  // try to find the tag
  getTag(key, dummy);

  // return true if tag exists and is of type bool
  return (dummy != NULL) && dummy->isBool();
}


//__________________________________________________________________________________|___________

bool TQTaggable::tagIsOfTypeString(const TString& key) {

  // a dummy (won't be used)
  TQValue * dummy = NULL;

  // try to find the tag
  getTag(key, dummy);

  // return true if tag exists and is of type string
  return (dummy != NULL) && dummy->isString();
}


//__________________________________________________________________________________|___________

bool TQTaggable::allTagsValidDoubles() {

  // loop over all present tags
  TQIterator itr(getListOfKeys(), true);
  while (itr.hasNext()) {
    if (!hasTagDouble(itr.readNext()->GetName())) {
      // one tag not a valid double
      return false;
    }
  }

  // no tags at all or all tags valid doubles
  return true;
}


//__________________________________________________________________________________|___________

bool TQTaggable::allTagsValidIntegers() {

  // loop over all present tags
  TQIterator itr(getListOfKeys(), true);
  while (itr.hasNext()) {
    if (!hasTagDouble(itr.readNext()->GetName())) {
      // one tag not a valid integer
      return false;
    }
  }

  // no tags at all or all tags valid integers
  return true;
}


//__________________________________________________________________________________|___________

bool TQTaggable::allTagsValidBools() {

  // loop over all present tags
  TQIterator itr(getListOfKeys(), true);
  while (itr.hasNext()) {
    if (!hasTagBool(itr.readNext()->GetName())) {
      // one tag not a valid bool
      return false;
    }
  }

  // no tags at all or all tags valid bools
  return true;
}


//__________________________________________________________________________________|___________

bool TQTaggable::allTagsOfTypeDouble() {

  // loop over all present tags
  TQIterator itr(getListOfKeys(), true);
  while (itr.hasNext()) {
    if (!tagIsOfTypeDouble(itr.readNext()->GetName())) {
      // one tag not a double
      return false;
    }
  }

  // no tags at all or all tags of type double
  return true;
}


//__________________________________________________________________________________|___________

bool TQTaggable::allTagsOfTypeInteger() {

  // loop over all present tags
  TQIterator itr(getListOfKeys(), true);
  while (itr.hasNext()) {
    if (!tagIsOfTypeInteger(itr.readNext()->GetName())) {
      // one tag not a integer
      return false;
    }
  }

  // no tags at all or all tags of type integer
  return true;
}


//__________________________________________________________________________________|___________

bool TQTaggable::allTagsOfTypeBool() {

  // loop over all present tags
  TQIterator itr(getListOfKeys(), true);
  while (itr.hasNext()) {
    if (!tagIsOfTypeBool(itr.readNext()->GetName())) {
      // one tag not a bool
      return false;
    }
  }

  // no tags at all or all tags of type bool
  return true;
}


//__________________________________________________________________________________|___________

bool TQTaggable::allTagsOfTypeString() {

  // loop over all present tags
  TQIterator itr(getListOfKeys(), true);
  while (itr.hasNext()) {
    if (!tagIsOfTypeString(itr.readNext()->GetName())) {
      // one tag not a string
      return false;
    }
  }

  // no tags at all or all tags of type string
  return true;
}


//__________________________________________________________________________________|___________

int TQTaggable::countTagUp(const TString& key) {

  /* the number of tags upwards */
  int nTags = 0;

  /* count tag of this taggable */
  if (hasTag(key))
    nTags++;

  /* count tags of base taggable */
  TQTaggable * baseTaggable = getBaseTaggable();
  if (baseTaggable)
    nTags += baseTaggable->countTagUp(key);

  return nTags;
}


//__________________________________________________________________________________|___________

int TQTaggable::countTagDown(const TString& key) {

  /* the number of tags upwards */
  int nTags = 0;

  /* count tag of this taggable */
  if (hasTag(key))
    nTags++;

  /* count tags of descendant taggables */
  TList * taggables = getDescendantTaggables();
  if (taggables) {
 
    TIterator * itr = taggables->MakeIterator();
    TObject * obj;
    while ((obj = itr->Next())) {
      if (obj->InheritsFrom(TQTaggable::Class()))
        nTags += ((TQTaggable*)obj)->countTagDown(key);
    }

    /* delete iterator */
    delete itr;
    /* delete the list */
    delete taggables;

  }

  return nTags;
}


//__________________________________________________________________________________|___________

bool TQTaggable::getOp(const TString& op, int &opCode) {

  if (op.Length() == 0)
    opCode = kOpNone;
  else if (op.CompareTo("~") == 0)
    opCode = kOpRec;
  else if (op.CompareTo("||") == 0)
    opCode = kOpOR;
  else if (op.CompareTo("&&") == 0)
    opCode = kOpAND;
  else if (op.CompareTo("+") == 0)
    opCode = kOpADD;
  else if (op.CompareTo("*") == 0)
    opCode = kOpMULT;
  else if (op.CompareTo("#") == 0)
    opCode = kOpCNT;
  else
    return false;

  return true;
}


//__________________________________________________________________________________|___________

bool TQTaggable::parseKey(TString key, TString &bareKey, int &opUp, int &opDown) {

  TString prefix;
  TString keyName;
  TString appendix;

  TQStringUtils::readToken(key, prefix, "~|&+*#");
  TQStringUtils::readToken(key, keyName, TQValue::getValidNameCharacters());
  TQStringUtils::readToken(key, appendix, "~|&+*#");

  /* some unexpected characters: stop */
  if (key.Length() != 0)
    return false;

  int tmpOpUp = kOpNone;
  int tmpOpDown = kOpNone;

  if (!getOp(prefix, tmpOpUp) || !getOp(appendix, tmpOpDown))
    return false;

  opUp = tmpOpUp;
  opDown = tmpOpDown;
  bareKey = keyName;
  return true;
}


//__________________________________________________________________________________|___________

int TQTaggable::getNTags() {
  // Return the number of tags

  if (fTags)
    return fTags->GetEntries();
  else
    return 0;
}


//__________________________________________________________________________________|___________

void TQTaggable::printTags(TString options) {

  /* the taggable to list tags of */
  TQTaggable * taggable = this;

  /* remember the tags already printed */
  TList * printed = new TList();

  /* define the width of table columns */
  const int cColWidth_Key = 40;
  const int cColWidth_Type = 10;
  const int cColWidth_Value = 40;

  bool printRecursive = options.Contains("r");

  /* print the headline */
  TString line;
  line.Append(TQStringUtils::fixedWidth("Key", cColWidth_Key));
  line.Append(TQStringUtils::fixedWidth("Type", cColWidth_Type));
  line.Append(TQStringUtils::fixedWidth("Value", cColWidth_Value));
  std::cout << TQStringUtils::makeBoldWhite(line) << std::endl;
  std::cout << TQStringUtils::makeBoldWhite(TQStringUtils::repeat("=", TQStringUtils::getWidth(line))) << std::endl;

  while (taggable) {

    /* iterate over every tag */
    TQIterator itr(taggable->getListOfTags());
    while(itr.hasNext()){
      TObject* obj = itr.readNext();
      if(!obj) continue;
 
      /* skip this tag if it was already printed */
      if (printed->FindObject(obj->GetName()))
        continue;
 
      /* remember the tag */
      printed->Add(obj);
 
      /* get the tag */
      TString key = obj->GetName();
      TString value, type;
      if (taggable->getValueOfTagAsString(key, value)
          && taggable->getTypeOfTagAsString(key, type)) {
 
        /* mark tags that were propagated to this taggable with ~ */
        if (taggable != this)
          key.Prepend("\033[1;31m~\033[0m");
 
        line.Clear();
        line.Append(TQStringUtils::fixedWidth(key, cColWidth_Key));
        line.Append(TQStringUtils::fixedWidth(type, cColWidth_Type));
        line.Append(TQStringUtils::fixedWidth(value, cColWidth_Value));
        std::cout << line.Data() << std::endl;
 
      } 
 
    }

    if (printRecursive)
      taggable = taggable->getBaseTaggable();
    else
      taggable = 0;

  }

  delete printed;
}


//__________________________________________________________________________________|___________

TList * TQTaggable::getListOfTags() {

  return fTags;
}


//__________________________________________________________________________________|___________

TList * TQTaggable::getListOfTagNames() {

  TList* l = new TList();
  TQValueIterator itr(fTags);
  while(itr.hasNext()){
    TQValue* v = itr.readNext();
    TObjString* s = new TObjString(v->GetName());
    l->Add(s);
  }
  l->SetOwner(true);
  return l;
}


//__________________________________________________________________________________|___________

TQTaggable * TQTaggable::parseParameterList(const TString& parameter, const TString& sep, bool trim, const TString& blocks, const TString& quotes) {
  // Tries to interpret the input string <parameter> as a comma-separated list of
  // values and returns a new instance of TQTaggable with the parameter values
  // represented as tags with the keys corresponding to the index of occurence of
  // the value in the list (the user is responsible for deleting the returned
  // instance). Returns a null pointer in case of failure.
  //
  // "1, 6, 4" results in tags "0 = 1, 1 = 6, 2 = 4",
  // "1, 'hello', 4" results in tags "0 = 1, 1 = 'hello', 2 = 4".
 
  // create a new instance of TQTaggable
  TQTaggable * pars = new TQTaggable();

  int i = 0;
  bool error = false;

  // tokenize values and iterate over list
  TQIterator itr(TQStringUtils::tokenize(parameter, sep, trim, blocks, quotes), true);
  while (itr.hasNext() && !error) {
    TString thisPar = itr.readNext()->GetName();
    if (!pars->importTagWithPrefix(thisPar, "", false, TString::Format("%d", i++))) {
      error = true;
    }
  }

  // return NULL pointer in case no parameters were parsed
  if (pars->getNTags() == 0 || error) {
    delete pars;
    pars = NULL;
  }

  // return the parameter list
  return pars;
}


//__________________________________________________________________________________|___________

int TQTaggable::importTagsWithPrefix(TString tags, const TString& prefix,
                                     bool overwrite, bool keepStringQuotes) {

  /* the number of tags imported */
  int nTags = 0;

  /* import the tags */
  TString tag;
  bool stop = false;
  while (!tags.IsNull() && !stop) {
    /* read the next tag */
    tag.Clear();
    int nChars = 0;
    nChars += TQStringUtils::readUpTo(tags, tag, ",", "[]{}", "''\"\"");
    nChars += TQStringUtils::removeLeading(tags, ",", 1);

    /* stop if no valid tag string is left */
    stop = (nChars == 0);

    /* == test for prefix block == */
    TString probe = tag;
    TString newPrefix;
    TString block;

    /* read the prefix */
    TQStringUtils::readBlanksAndNewlines(probe);
    TQStringUtils::readToken(probe, newPrefix, getValidKeyCharacters());

    /* read the prefix block */
    TQStringUtils::readBlanksAndNewlines(probe);
    int nBlockChar = TQStringUtils::readBlock(probe, block, "[]{}", "''\"\"");
    TQStringUtils::readBlanksAndNewlines(probe);

    if (!newPrefix.IsNull() && probe.IsNull() && nBlockChar > 0)
      /* import prefix block */
      nTags += importTagsWithPrefix(block, prefix + newPrefix, overwrite, keepStringQuotes);
    else
      /* import the tag */
      nTags += importTagWithPrefix(tag, prefix, overwrite, "", keepStringQuotes);
  }
 
  /* return the number of tags imported */
  return nTags;
}


//__________________________________________________________________________________|___________

int TQTaggable::importTags(TString tags, bool overwrite, bool keepStringQuotes) {

  return importTagsWithPrefix(tags, "", overwrite, keepStringQuotes);
}


//__________________________________________________________________________________|___________

int TQTaggable::importTagsWithPrefix(const TQTaggable * tags, const TString& prefix, bool overwrite, bool recursive) {
  // import all tags from another TQTaggable object, appending a certain prefix
  if(!tags) return 0;
  return this->importTagsWithPrefix(*tags,prefix,overwrite,recursive);
}

//__________________________________________________________________________________|___________

int TQTaggable::importTagsWithPrefix(const TQTaggable& tags, const TString& prefix, bool overwrite, bool recursive) {
  // import all tags from another TQTaggable object, appending a certain prefix

  /* the number of imported tags */
  int nTags = 0;

  /* loop over tags to import */
  TQIterator itr(tags.fTags);
  while (itr.hasNext()){
    TQValue* val = dynamic_cast<TQValue*>(itr.readNext());
    if(!val) continue;
    if (overwrite || !hasTag(val->getNameConst())) {
      /* make a copy of the tag */
      TQValue * newTag = 0;
      if (prefix.IsNull())
        newTag = val->copy();
      else
        newTag = val->copy(prefix + val->getName());
      /* set the tag */
      int nSubTags = setTag(newTag);
      if (nSubTags > 0)
        nTags += nSubTags;
      else
        delete newTag;
    }
  }
 
  /* import tags recursively */
  if (recursive) {
    TQTaggable * base = tags.getBaseTaggable();
    nTags += importTagsWithPrefix(base, prefix, false, true);
  }

  /* return the number of imported tags */
  return nTags;
}

////__________________________________________________________________________________|___________

int TQTaggable::importTagsWithoutPrefix(const TQTaggable& tags, const TString& prefix, bool overwrite, bool recursive) {
  // import all tags from another TQTaggable object, filtering and removing a certain prefix

  /* the number of imported tags */
  int nTags = 0;
  /* loop over tags to import */
  TQValueIterator itr(tags.fTags);
  while (itr.hasNext()){
    TQValue* val = itr.readNext();
    if(!val) continue;
    if (overwrite || !this->hasTag(val->getNameConst())) {
      /* make a copy of the tag */
      TQValue * newTag = 0;
      if (prefix.IsNull())
        newTag = val->copy();
      else {
        TString name(val->getNameConst());
        if(TQStringUtils::removeLeadingText(name,prefix)){
          newTag = val->copy(name);
        }
      }
      /* set the tag */
      int nSubTags = setTag(newTag);
      if (nSubTags > 0)
        nTags += nSubTags;
      else
        delete newTag;
    }
  }
 
  /* import tags recursively */
  if (recursive) {
    TQTaggable * base = tags.getBaseTaggable();
    nTags += importTagsWithoutPrefix(base, prefix, false, true);
  }

  return nTags;
}


////__________________________________________________________________________________|___________

int TQTaggable::importTagsWithoutPrefix(const TQTaggable* tags, const TString& prefix, bool overwrite, bool recursive) {
  // import all tags from another TQTaggable object, filtering and removing a certain prefix
  if(!tags) return 0;
  return this->importTagsWithoutPrefix(*tags,prefix,overwrite,recursive);

}


////__________________________________________________________________________________|___________

TList* TQTaggable::makeListOfTags(TList* unTags){
  // convert a list of objects into a list of TQValue objects -- legacy converter function
  TList* tags = new TList();

  if (!unTags)
    return NULL;

  /* loop over tags to import */
  TQIterator itr(unTags);
  while (itr.hasNext()){
    TObject * obj = itr.readNext();
 
    TQValue * newTag = NULL;
    TQValue* oldTag = dynamic_cast<TQValue*>(obj);
    if(oldTag) 
      tags->Add(oldTag->copy());
    if (obj->InheritsFrom(TNamed::Class()))
      newTag = TQValue::newString(
                                  obj->GetName(), ((TNamed*)obj)->GetTitle());
    else if (obj->InheritsFrom(TParameter<double>::Class()))
      newTag = TQValue::newDouble(
                                  obj->GetName(), ((TParameter<double>*)obj)->GetVal());
    else if (obj->InheritsFrom(TParameter<int>::Class()))
      newTag = TQValue::newInteger(
                                   obj->GetName(), ((TParameter<int>*)obj)->GetVal());
    else if (obj->InheritsFrom(TParameter<float>::Class()))
      newTag = TQValue::newBool(
                                obj->GetName(), ((TParameter<float>*)obj)->GetVal() != 0.);
 
    if(newTag){
      tags->Add(newTag);
    }
    delete obj;
  }
  return tags;
}


//__________________________________________________________________________________|___________

int TQTaggable::importTags(const TQTaggable * tags, bool overwrite, bool recursive) {
  // import a list of tags
  return importTagsWithPrefix(tags, "", overwrite, recursive);
}

//__________________________________________________________________________________|___________

int TQTaggable::importTags(const TQTaggable& tags, bool overwrite, bool recursive) {
  // import a list of tags
  return importTagsWithPrefix(&tags, "", overwrite, recursive);
}


//__________________________________________________________________________________|___________

int TQTaggable::importTagWithPrefix(const TString& tagBackup, const TString& prefix,
                                    bool overwrite, TString fallbackKey, bool keepStringQuotes) {
  // import a tag with a prefix
  TString tag(tagBackup);

  /* read key */
  TString key = prefix;
  TQStringUtils::readBlanksAndNewlines(tag);
  int nKey = TQStringUtils::readToken(tag, key, getValidKeyCharacters());

  /* read assignment character "=" */
  TString assign;
  TQStringUtils::readBlanksAndNewlines(tag);
  TQStringUtils::readToken(tag, assign, "=");

  /* if there is no valid key name definition ... */
  bool usingFallbackKey = false;
  if (nKey == 0 || assign.CompareTo("=") != 0) {
    if (fallbackKey.IsNull()) {
      /* no key given */
      return 0;
    } else {
      /* use fallback key */
      key = fallbackKey;
      tag = tagBackup;
      usingFallbackKey = true;
    }
  }

  TQStringUtils::readBlanksAndNewlines(tag);

  if (usingFallbackKey && tag.IsNull()) {
    return 0;
  }

  /* stop if overwriting is disabled and the tag already exists */
  if (!overwrite && hasTag(key))
    return 0;

  TString val;
  // try to read string quoted either with '' or ""
  bool singleQuotes = (TQStringUtils::readBlock(tag, val, "''") > 0);
  bool doubleQuotes = false;
  if (!singleQuotes) {
    doubleQuotes = (TQStringUtils::readBlock(tag, val, "\"\"") > 0);
  }
  if (singleQuotes || doubleQuotes) {
    /* we only expect blanks beyond the end of the string */
    val = TQStringUtils::trim(val);
    if (keepStringQuotes) {
      if (singleQuotes) {
        val = TString("'") + val + "'";
      } else if (doubleQuotes) {
        val = TString("\"") + val + "\"";
      }
    }
    return setTagString(key, val);
  }
 
  // is the tag a list enclosed in "{}" ?
  if (TQStringUtils::readBlock(tag, val, "{}", "''\"\"") > 0) {
    return this->setTagList(key,val);
  }
 
  /* read value */
  val = TQStringUtils::trim(tag);
  if (TQStringUtils::isDouble(val)) {
    /* >> double tag */
    return setTagDouble(key, val.Atof());
  } else if (TQStringUtils::isInteger(val)) {
    /* >> integer tag */
    return setTagInteger(key, (int)val.Atof());
  } else if (TQStringUtils::isBool(val)) {
    /* >> bool tag */
    return setTagBool(key, TQStringUtils::getBoolFromString(val));
  } else if (val.CompareTo("!") == 0) {
    /* >> remove tag */
    if (removeTag(key)) {
      return 1;
    }
  } else {
    /* >> string tag */
    return setTagString(key, val);
  }
  return 0;
}


//__________________________________________________________________________________|___________

int TQTaggable::importTag(TString tag, bool overwrite, bool keepStringQuotes) {
  // import a tag
  return importTagWithPrefix(tag, "", overwrite, keepStringQuotes);
}

//__________________________________________________________________________________|___________


int TQTaggable::setTag(const TString& key, const TString& value, const TString& destination) {
  // set a single string tag
  return setTagString(key, value, destination);
}

//__________________________________________________________________________________|___________


int TQTaggable::setTag(const TString& key, const char* value, const TString& destination) {
  // set a single string tag
  TString s(value);
  return setTagString(key, s, destination);
}


//__________________________________________________________________________________|___________

int TQTaggable::setTag(const TString& key, double value, const TString& destination) {
  // set a single double-precision floating point tag
  return setTagDouble(key, value, destination);
}


//__________________________________________________________________________________|___________

int TQTaggable::setTag(const TString& key, int value, const TString& destination) {
  // set a single integer tag
  return setTagInteger(key, value, destination);
}


//__________________________________________________________________________________|___________

int TQTaggable::setTag(const TString& key, bool value, const TString& destination) {
  // set a single bool tag
  return setTagBool(key, value, destination);
}


//__________________________________________________________________________________|___________

int TQTaggable::setTagAuto(const TString& key, TString value, const TString& destination) {
  // set a tag, automatically picking the type
  TString val;
  int nTags = 0;
  // try to read string quoted either with '' or ""
  if ((TQStringUtils::readBlock(value, val, "''")) > 0 || (TQStringUtils::readBlock(value, val, "\"\"") > 0)) {
    /* we only expect blanks after the end of the string */
    val = TQStringUtils::trim(val);
    if (!val.IsNull()) {
      /* >> string tag */
      nTags = setTagString(key, val);
    }
    return nTags;
  } 
  // is the tag a list enclosed in "{}" ?
  if (TQStringUtils::readBlock(value, val, "{}", "''\"\"")) {
    return this->setTagList(key,val);
  }
  val = TQStringUtils::trim(value);
  if(TQStringUtils::findFree(val,',',"()[]{}") <(size_t)val.Length()){
    nTags = this->setTagList(key,val);
    }
  if(nTags > 0) return nTags;

  if (TQStringUtils::isDouble(val)) {
    /* >> double tag */
    nTags = setTagDouble(key, val.Atof());
  } else if (TQStringUtils::isInteger(val)) {
    /* >> integer tag */
    nTags = setTagInteger(key, (int)val.Atof());
  } else if (TQStringUtils::isBool(val)) {
    /* >> bool tag */
    nTags = setTagBool(key, TQStringUtils::getBoolFromString(val));
  } else if (val.CompareTo("!") == 0) {
    /* >> remove tag */
    if (removeTag(key)) {
      nTags++;
    }
  } else {
    /* >> string tag */
    nTags = setTagString(key, val);
  }
  return nTags;
}

//__________________________________________________________________________________|___________

int TQTaggable::setTagList(const TString& key, TString value, const TString& destination){
  // set a list of tags
  TQTaggable * list = TQTaggable::parseParameterList(value, ",", true, "{}()[]", "''\"\"");
  if (!list) return 0;
  int nTags = 0;
  TQValue * val = NULL;
  TQIterator itr(list->getListOfKeys(), true);
  while (itr.hasNext()) {
    TString listKeyName = itr.readNext()->GetName();
    list->getTag(listKeyName, val);
    nTags += setTag(val->copy(key + "." + listKeyName), destination);
  }
  delete list;
 
  return nTags;
}

//__________________________________________________________________________________|___________

template<class T>
int TQTaggable::setTagList(const TString& key, const std::vector<T>& list, const TString& destination){
  // set a list of tags
  int nTags = 0;
  for(const auto& x:list){
    nTags += this->setTag(TString::Format("%s.%d",key.Data(),nTags), x, destination);
  }
  return nTags;
}

template int TQTaggable::setTagList<bool>(const TString& key, const std::vector<bool>& list, const TString& destination);
template int TQTaggable::setTagList<int>(const TString& key, const std::vector<int>& list, const TString& destination);
template int TQTaggable::setTagList<double>(const TString& key, const std::vector<double>& list, const TString& destination);
template int TQTaggable::setTagList<TString>(const TString& key, const std::vector<TString>& list, const TString& destination);
template int TQTaggable::setTagList<const char*>(const TString& key, const std::vector<const char*>& list, const TString& destination);

//__________________________________________________________________________________|___________

int TQTaggable::setTagDouble(TString key, double value, const TString& destination) {
  // set a single double-precision floating point number tag
  bool dontOverwrite = TQStringUtils::removeTrailing(key, "?", 1);

  /* create the tag */
  TQValue * tag = TQValue::newDouble(key, value);

  /* stop if we failed to create the tag */
  if (!tag)
    return 0;

  /* add/set the tag */
  int nTags = setTag(tag, destination, this->getGlobalOverwrite() && !dontOverwrite);

  /* delete the tag if it wasn't used */
  if (nTags == 0)
    delete tag;

  /* return the number of tags added */
  return nTags;
}


//__________________________________________________________________________________|___________

int TQTaggable::setTagInteger(TString key, int value, const TString& destination) {
  // set a single integer tag
  bool dontOverwrite = TQStringUtils::removeTrailing(key, "?", 1);

  /* create the tag */
  TQValue * tag = TQValue::newInteger(key, value);

  /* stop if we failed to create the tag */
  if (!tag)
    return 0;

  /* add/set the tag */
  int nTags = setTag(tag, destination, this->getGlobalOverwrite() && !dontOverwrite);

  /* delete the tag if it wasn't used */
  if (nTags == 0)
    delete tag;

  /* return the number of tags added */
  return nTags;
}


//__________________________________________________________________________________|___________

int TQTaggable::setTagBool(TString key, bool value, const TString& destination) {
  // set a single bool tag
  bool dontOverwrite = TQStringUtils::removeTrailing(key, "?", 1);

  /* create the tag */
  TQValue * tag = TQValue::newBool(key, value);

  /* stop if we failed to create the tag */
  if (!tag)
    return 0;

  /* add/set the tag */
  int nTags = setTag(tag, destination, this->getGlobalOverwrite() && !dontOverwrite);

  /* delete the tag if it wasn't used */
  if (nTags == 0)
    delete tag;

  /* return the number of tags added */
  return nTags;
}


//__________________________________________________________________________________|___________

int TQTaggable::setTagString(TString key, const TString& value, const TString& destination) {
  // set a single string tag
  bool dontOverwrite = TQStringUtils::removeTrailing(key, "?", 1);

  /* create the tag */
  TQValue * tag = TQValue::newString(key, value);

  /* stop if we failed to create the tag */
  if (!tag)
    return 0;

  /* add/set the tag */
  int nTags = setTag(tag, destination, this->getGlobalOverwrite() && !dontOverwrite);

  /* delete the tag if it wasn't used */
  if (nTags == 0)
    delete tag;

  /* return the number of tags added */
  return nTags;
}


//__________________________________________________________________________________|___________

int TQTaggable::exportTags(TQTaggable * dest, const TString& subDest, const TString& filter, bool recursive) {
  // export the tags to another taggable object
  if (!dest)
    return 0;

  /* the number of tags exported */
  int nTags = 0;

  /* loop over tags */
  TQIterator itr(fTags);
  while (itr.hasNext()){
    TQValue* val = dynamic_cast<TQValue*>(itr.readNext());
    if(!val) continue;
    /* apply name filter */
    if (!filter.IsNull() && !TQStringUtils::matchesFilter(val->GetName(), filter, ",", true)) {
      continue;
    }
    TQValue* newTag = val->copy();
    if (dest->setTag(newTag, subDest, dest->getGlobalOverwrite()))
      nTags++;
  }
  
  if(recursive && this->getBaseTaggable()){
    nTags += this->getBaseTaggable()->exportTags(dest,subDest,filter,recursive);
  }

  /* return the number of tags exported */
  return nTags;
}


//__________________________________________________________________________________|___________

TString TQTaggable::exportTagsAsString(const TString& filter, bool xmlStyle) {
  // export the tags as a string
  TString result;

  /* loop over tags */
  bool first = true;
  TQIterator itr(fTags);
  while(itr.hasNext()){
    TQValue* val = dynamic_cast<TQValue*>(itr.readNext());
    if(!val) continue;
    /* apply name filter */
    if (!filter.IsNull() && !TQStringUtils::matchesFilter(val->GetName(), filter, ",", true)) {
      continue;
    }
    /* comma separation */
    if (!first) {
      if (xmlStyle)
        result.Append(" ");
      else
        result.Append(", ");
    }
    first = false;
    /* append the tag */
    result.Append(val->getAsString(xmlStyle));
  }
 
  /* return the final string */
  return result;
}

//__________________________________________________________________________________|___________

TString TQTaggable::exportTagsAsConfigString(const TString& prefix, const TString& filter){
  // export the tags to a config string
  TString result;

  /* loop over tags */
  TQIterator itr(fTags);
  while(itr.hasNext()){
    TQValue* val = dynamic_cast<TQValue*>(itr.readNext());
    if(!val) continue;
    /* apply name filter */
    if (!filter.IsNull() && !TQStringUtils::matchesFilter(val->GetName(), filter, ",", true)) {
      continue;
    }
    /* append the tag */
    result.Append(prefix);
    result.Append(val->getName());
    result.Append(": ");
    result.Append(val->getValueAsString());
    result.Append("\n");
  }
 
  /* return the final string */
  return result;
}


//__________________________________________________________________________________|___________

TString TQTaggable::replaceInText(const TString& in, const TString& prefix, bool keepQuotes) {
  // replace all placeholders $(xyz) by their respective tagged values "xyz = abc"
  int nReplaced = 0;
  int nFailed = 0;
  return replaceInText(in, nReplaced, nFailed, prefix, keepQuotes);
}

//__________________________________________________________________________________|___________

TString TQTaggable::replaceInTextRecursive(TString in, const TString& prefix, bool keepQuotes) {
  // replace all placeholders $(xyz) by their respective tagged values "xyz = abc"
  // this version of the function recursively replaces the string until no resolvable tag is left
  int nReplaced = 0;
  int nFailed = 0;
  while(true){
    in = replaceInText(in, nReplaced, nFailed, prefix, keepQuotes);
    if(nReplaced == 0) return in;
  }
}

//__________________________________________________________________________________|___________

TString TQTaggable::replaceInText(const TString& in, const char* prefix, bool keepQuotes){
  // replace all placeholders $(xyz) by their respective tagged values "xyz = abc"
  return this->replaceInText(in,(TString)prefix,keepQuotes);
}

//__________________________________________________________________________________|___________

TString TQTaggable::replaceInText(const TString& in, bool keepQuotes) {
  // replace all placeholders $(xyz) by their respective tagged values "xyz = abc"
  return replaceInText(in, "", keepQuotes);
}

//__________________________________________________________________________________|___________

TString TQTaggable::replaceInText(const TString& in, int &nReplaced, int &nFailed, bool keepQuotes) {
  // replace all placeholders $(xyz) by their respective tagged values "xyz = abc"
  return this->replaceInText(in, nReplaced, nFailed, "", keepQuotes);
}

//__________________________________________________________________________________|___________

TString TQTaggable::replaceInText(TString in, int &nReplaced, int &nFailed, const TString& prefix, bool keepQuotes) {
  // replace all placeholders $(xyz) by their respective tagged values "xyz = abc"
  TString out;
  nReplaced = 0;
  nFailed = 0;

  while (!in.IsNull()) {

    /* read text up to control character '$' */
    TQStringUtils::readUpTo(in, out, "$");
    if (in.IsNull())
      continue;

    /* read text up to control character '$' */
    TQStringUtils::readUpTo(in, out, "$");
    if (in.IsNull())
      continue;

    /* read control character */
    TString field;
    int nMarker = TQStringUtils::readToken(in, field, "$");

    /* read replacement definition block */
    TString def;
    int nDef = 0;
    if (in.BeginsWith("(")) {
      nDef = TQStringUtils::readBlock(in, def, "()[]{}", "''\"\"");
      if (nDef > 0)
        field.Append(TString::Format("(%s)", def.Data()));
    } else {
      nDef = TQStringUtils::readToken(in, def, getValidKeyCharacters());
      if (nDef > 0)
        field.Append(def);
    }

    if (!def.IsNull()) {
      /* split definition: get key name */
      TString keyName = prefix;
      TQStringUtils::readUpTo(def, keyName, ",");
      /* read options */
      TQTaggable tmpTags;
      bool missingOptions = false;
      if (!def.IsNull()) {
        TQStringUtils::removeLeading(def, ",", 1);
        missingOptions = (tmpTags.importTagWithPrefix(
                                                      def, "", false, "default") == 0);
      }
      bool hasMatchingTag = hasTag(keyName);
      if (nMarker == 1 && !keyName.IsNull() && !missingOptions &&
          (hasMatchingTag || tmpTags.hasTag("default"))) {
        TString tag;
        if (hasMatchingTag) {
          if (keepQuotes && tagIsOfTypeString(keyName))
            getValueOfTagAsString(keyName, tag);
          else
            getTagString(keyName, tag);
        } else {
          if (keepQuotes && tmpTags.tagIsOfTypeString("default"))
            tmpTags.getValueOfTagAsString("default", tag);
          else
            //TODO write tag documentation for this magic tag!
            tmpTags.getTagString("default", tag);
        }
        out.Append(tag);
        nReplaced++;
      } else {
        nFailed++;
        out.Append(field);
      }
    } else {
      nFailed++;
      out.Append(field);
    }
  }

  return out;
}


//__________________________________________________________________________________|___________

TString TQTaggable::getValuesOfTags(const TString& keys, const TString& sep) {
  // replace keys by values in a comma-separated string list
  TString values;
 
  TQIterator itr(this->getListOfKeys(keys), true);
  while (itr.hasNext()) {
    TQStringUtils::append(values, this->getTagStringDefault(itr.readNext()->GetName()), sep);
  }
 
  return values;
}


//__________________________________________________________________________________|___________


std::vector<TString > TQTaggable::getTagVString(const TString& key){
  // retrieve a list of tags as a vector of strings
  std::vector<TString> vec;
  this->getTag(key,vec);
  return vec;
}

//__________________________________________________________________________________|___________


TList* TQTaggable::getTagList(const TString& key){
  // retrieve a list of tags
  TList* l = new TList();
  l->SetOwner(true);
  this->getTag(key,l);
  return l;
}

//__________________________________________________________________________________|___________


std::vector<int > TQTaggable::getTagVInt (const TString& key){
  // retrieve a list of tags as a vector of integers
  std::vector<int> vec;
  this->getTag(key,vec);
  return vec;
}

//__________________________________________________________________________________|___________


std::vector<int > TQTaggable::getTagVInteger(const TString& key){
  // retrieve a list of tags as a vector of integers
  std::vector<int> vec;
  this->getTag(key,vec);
  return vec;
}

//__________________________________________________________________________________|___________


std::vector<double> TQTaggable::getTagVDouble(const TString& key){
  // retrieve a list of tags as a vector of doubles
  std::vector<double> vec;
  this->getTag(key,vec);
  return vec;
}

//__________________________________________________________________________________|___________

std::vector<bool > TQTaggable::getTagVBool (const TString& key){
  // retrieve a list of tags as a vector of bool
  std::vector<bool> vec;
  this->getTag(key,vec);
  return vec;
}

//__________________________________________________________________________________|___________

int TQTaggable::getTag(const TString& key, std::vector<TString>& vec){
  // get a list of tags
  int idx = 0;
  TString entry;
  while(this->getTag(TString::Format("%s.%d",key.Data(),idx),entry)){
    vec.push_back(entry);
    idx++;
  }
  if(idx==0 && this->hasTagString(key))
    vec.push_back(this->getTagStringDefault(key,""));
  return idx;
}

//__________________________________________________________________________________|___________

int TQTaggable::getTag(const TString& key, TList* l){
  // get a list of tags
  if(!l) return -1;
  int idx = 0;
  TString entry;
  while(this->getTag(TString::Format("%s.%d",key.Data(),idx),entry)){
    l->Add(new TObjString(entry));
    idx++;
  }
  if(idx==0 && this->hasTagString(key))
    l->Add(new TObjString(this->getTagStringDefault(key,"")));
  return idx;
}

//__________________________________________________________________________________|___________

int TQTaggable::getTagListLength(const TString& key){
  // get the length of a list of tags
  int idx = 0;
  while(this->hasTag(TString::Format("%s.%d",key.Data(),idx))){
    idx++;
  }
  if(idx==0 && this->hasTagString(key))
    idx = 1;
  return idx;
}

//__________________________________________________________________________________|___________

int TQTaggable::getTag(const TString& key, std::vector<double>& vec){
  // get a list of double-precision floating point tags
  int idx = 0;
  double entry;
  while(this->getTag(TString::Format("%s.%d",key.Data(),idx),entry)){
    vec.push_back(entry);
    idx++;
  }
  if(idx==0 && this->hasTagDouble(key)){
    vec.push_back(this->getTagDoubleDefault(key,0));
    idx++;
  }
  return idx;
}

//__________________________________________________________________________________|___________

int TQTaggable::getTag(const TString& key, std::vector<int>& vec){
  // get a list of integer tags
  int idx = 0;
  int entry;
  while(this->getTag(TString::Format("%s.%d",key.Data(),idx),entry)){
    vec.push_back(entry);
    idx++;
  }
  if(idx==0 && this->hasTagInteger(key)){
    vec.push_back(this->getTagIntegerDefault(key,0));
    idx++;
  }
  return idx;
}

//__________________________________________________________________________________|___________

int TQTaggable::getTag(const TString& key, std::vector<bool>& vec){
  // get a list of boolean tags
  int idx = 0;
  bool entry;
  while(this->getTag(TString::Format("%s.%d",key.Data(),idx),entry)){
    vec.push_back(entry);
    idx++;
  }
  if(idx==0 && this->hasTagBool(key)){
    vec.push_back(this->getTagBoolDefault(key,false));
    idx++;
  }
  return idx;
}

//__________________________________________________________________________________|___________

bool TQTaggable::getTag(const TString& key, double &value) {
  // get a single double-precision floating point tag
  return getTagDouble(key, value);
}


//__________________________________________________________________________________|___________

bool TQTaggable::getTag(const TString& key, int &value) {
  // get a single integer tag
  return getTagInteger(key, value);
}


//__________________________________________________________________________________|___________

bool TQTaggable::getTag(const TString& key, bool &value) {
  // get a single boolean tag
  return getTagBool(key, value);
}


//__________________________________________________________________________________|___________

bool TQTaggable::getTag(const TString& key, TString &value) {
  // get a single string tag
  return getTagString(key, value);
}


//__________________________________________________________________________________|___________

bool TQTaggable::getTagDouble(const TString& key, double &value) {
  // get a single double-precision floating point tag
  TQValue * val = 0;

  if (getTag(key, val) && val->isValidDouble()) {
    value = val->getDouble();
    onRead(val);
    return true;
  } else {
    return false;
  }
}


//__________________________________________________________________________________|___________

bool TQTaggable::getTagInteger(const TString& key, int &value) {
  // get a single integer tag
  TQValue * val = 0;

  if (getTag(key, val) && val->isValidInteger()) {
    value = val->getInteger();
    onRead(val);
    return true;
  } else {
    return false;
  }
}


//__________________________________________________________________________________|___________

bool TQTaggable::getTagBool(const TString& key, bool &value) {
  // get single boolean tag
  TQValue * val = 0;

  if (getTag(key, val) && val->isValidBool()) {
    value = val->getBool();
    onRead(val);
    return true;
  } else {
    return false;
  }
}


//__________________________________________________________________________________|___________

bool TQTaggable::getTagString(const TString& key, TString &value) {
  // get a single string tag
  TQValue * val = 0;

  if (getTag(key, val)) {
    value = val->getString();
    onRead(val);
    return true;
  } else {
    return false;
  }
}


//__________________________________________________________________________________|___________

double TQTaggable::getTagDefault(const TString& key, double defaultVal) {
  // get a single double-precision floating point tag
  return getTagDoubleDefault(key, defaultVal);
}


//__________________________________________________________________________________|___________

int TQTaggable::getTagDefault(const TString& key, int defaultVal) {
  // get a single integer tag
  return getTagIntegerDefault(key, defaultVal);
}


//__________________________________________________________________________________|___________

bool TQTaggable::getTagDefault(const TString& key, bool defaultVal) {
  // get a single boolean tag
  return getTagBoolDefault(key, defaultVal);
}


//__________________________________________________________________________________|___________

TString TQTaggable::getTagDefault(const TString& key, const TString& defaultVal) {
  // get a single string tag
  return getTagStringDefault(key, defaultVal);
}

//__________________________________________________________________________________|___________

TString TQTaggable::getTagDefault(const TString& key, const char* defaultVal) {
  // get a single string tag
  return getTagStringDefault(key, defaultVal);
}


//__________________________________________________________________________________|___________

double TQTaggable::getTagDoubleDefault(const TString& key, double defaultVal) {
  // get a single double-precision floating point tag
  double value = defaultVal;
  getTagDouble(key, value);
  return value;
}


//__________________________________________________________________________________|___________

int TQTaggable::getTagIntegerDefault(const TString& key, int defaultVal) {
  // get a single integer tag
  int value = defaultVal;
  getTagInteger(key, value);
  return value;
}


//__________________________________________________________________________________|___________

bool TQTaggable::getTagBoolDefault(const TString& key, bool defaultVal) {
  // get a single boolean tag
  bool value = defaultVal;
  getTagBool(key, value);
  return value;
}


//__________________________________________________________________________________|___________

TString TQTaggable::getTagStringDefault(const TString& key, const TString& defaultVal) {
  // get a single string tag
  TString value = defaultVal;
  getTagString(key, value);
  return value;
}


//__________________________________________________________________________________|___________

bool TQTaggable::getTagAsString(const TString& key, TString &tag) {
  // get single string tag
  TQValue * val = 0;

  if (getTag(key, val)) {
    tag = val->getAsString();
    onRead(val);
    return true;
  } else {
    return false;
  }
}


//__________________________________________________________________________________|___________

bool TQTaggable::getTypeOfTagAsString(const TString& key, TString &type) {
  // get the type of a tag as a string
  TQValue * val = 0;

  if (getTag(key, val)) {
    type = val->getTypeAsString();
    onRead(val);
    return true;
  } else {
    return false;
  }
}


//__________________________________________________________________________________|___________

bool TQTaggable::getValueOfTagAsString(const TString& key, TString &value) {
  // get the value of a tag as a string
  TQValue * val = 0;

  if (getTag(key, val)) {
    value = val->getValueAsString();
    onRead(val);
    return true;
  } else {
    return false;
  }
}


//__________________________________________________________________________________|___________

TQTaggable::~TQTaggable() {
  // standard destructor
  clear();

  // delete flags
  if (fReadFlags) {
    delete fReadFlags;
  }
}


bool TQTaggable::getTag (const TString& key, double &value, bool recursive){
  // get a single double-precision floating point tag
  if(recursive) 
    return this->getTag(TString::Format("~%s",key.Data()),value);
  return this->getTag(key,value);
}
bool TQTaggable::getTag (const TString& key, int &value, bool recursive){
  // get a single integer tag
  if(recursive) 
    return this->getTag(TString::Format("~%s",key.Data()),value);
  return this->getTag(key,value);
}
bool TQTaggable::getTag (const TString& key, bool &value, bool recursive){
  // get a single boolean tag
  if(recursive) 
    return this->getTag(TString::Format("~%s",key.Data()),value);
  return this->getTag(key,value);
}
bool TQTaggable::getTag (const TString& key, TString &value, bool recursive){
  // get a single string tag
  if(recursive) 
    return this->getTag(TString::Format("~%s",key.Data()),value);
  return this->getTag(key,value);
}

bool TQTaggable::getTagDouble (const TString& key, double &value, bool recursive){
  // get a single double-precision floating point tag
  if(recursive) 
    return this->getTag(TString::Format("~%s",key.Data()),value);
  return this->getTag(key,value);
}
bool TQTaggable::getTagInteger (const TString& key, int &value, bool recursive){
  // get a single integer tag
  if(recursive) 
    return this->getTag(TString::Format("~%s",key.Data()),value);
  return this->getTag(key,value);
}
bool TQTaggable::getTagBool (const TString& key, bool &value, bool recursive){
  // get a single boolean tag
  if(recursive) 
    return this->getTag(TString::Format("~%s",key.Data()),value);
  return this->getTag(key,value);
}
bool TQTaggable::getTagString (const TString& key, TString &value, bool recursive){
  // get a single string tag
  if(recursive) 
    return this->getTag(TString::Format("~%s",key.Data()),value);
  return this->getTag(key,value);
}

bool TQTaggable::exportConfigFile(const TString& filename, const TString& prefix, bool writeUnreadKeys){
  // export the tags to a config file
  TQUtils::ensureDirectoryForFile(filename);

  std::ofstream of(filename);
  if(!of.good()) return false;

  TQValueIterator itr(fTags);
  while (itr.hasNext()){
    TQValue* val = itr.readNext();
    TString key(val->GetName());
    if(TQStringUtils::removeLeading(key,".") == 1){
      TString value(val->getString());
      of << "# " << key << ": " << value << std::endl;
    }
  }
  
  itr.reset();
  while (itr.hasNext()){
    TQValue* val = itr.readNext();
    /* apply name filter */
    TString key(val->GetName());
    TString value(val->getString());
    if(key.BeginsWith(".")) continue;
    if(writeUnreadKeys || fReadFlags->FindObject(key)){
      of << prefix << "." << key << ":" << "\t" << value << std::endl;
    }
  }
  of.close();
  return true;
}

bool TQTaggable::exportConfigFile(const TString& filename, bool writeUnreadKeys){
  //@tag:[.configname] This object tag determines the prefix when exporting to a configuration file. Default: "Config"
  return this->exportConfigFile(filename, this->getTagStringDefault(".configname","Config"),writeUnreadKeys);
}
bool TQTaggable::exportConfigFile(bool writeUnreadKeys){
  //@tag:[.filename] This object tag determines the filename when exporting to a configuration file. Default: "config.cfg"
  TString fname(this->getTagStringDefault(".filename","config.cfg"));
  return this->exportConfigFile(TQFolder::getPathTail(fname), this->getTagStringDefault(".configname","Config"), writeUnreadKeys);
}

int TQTaggable::replaceInTags(TQTaggable& params, const TString& tagFilter) {
  // Replace tag placeholders in tags set on this instance of TQTaggable
  // according to the tags set on the TQTaggable object provided. A tag filter
  // can be specified to only perform the replacement where the key of a tag
  // matches the filter expression.
  TList* lTags = this->getListOfKeys(tagFilter);
  if (!lTags) return -1;
  TIterator* itr = lTags->MakeIterator();
  TObjString* ostr;
  while ((ostr = (dynamic_cast<TObjString*>(itr->Next())))) {
    if (!ostr) continue;
    if (!this->tagIsOfTypeString(ostr->GetString())) continue;
    this->setTagString(ostr->GetString(), params.replaceInText(this->getTagStringDefault(ostr->GetString(),"")));
  }
  delete lTags;
  delete itr;
  
  return 0;
}



// these are just wrappers for std::string
std::string TQTaggable::exportTagsAsStandardString(const TString& filter, bool xmlStyle){
  // wrapper for the TString variant of this function
  return this->exportTagsAsString(filter,xmlStyle).Data();
}
std::string TQTaggable::exportTagsAsStandardConfigString(const TString& prefix, const TString& filter){
  // wrapper for the TString variant of this function
  return this->exportTagsAsConfigString(prefix,filter).Data();
}
std::string TQTaggable::replaceInStandardStringRecursive(TString in, const TString& prefix, bool keepQuotes){
  // wrapper for the TString variant of this function
  return this->replaceInTextRecursive(in,prefix,keepQuotes).Data();
}
std::string TQTaggable::replaceInStandardString(const TString& in, const char* prefix, bool keepQuotes){
  // wrapper for the TString variant of this function
  return this->replaceInText(in,prefix,keepQuotes).Data();
}
std::string TQTaggable::replaceInStandardString(const TString& in, const TString& prefix, bool keepQuotes){
  // wrapper for the TString variant of this function
  return this->replaceInText(in,prefix,keepQuotes).Data();
}
std::string TQTaggable::replaceInStandardString(const TString& in, bool keepQuotes){
  // wrapper for the TString variant of this function
  return this->replaceInText(in,keepQuotes).Data();
}
std::string TQTaggable::getTagStandardStringDefault (const TString& key, const TString& defaultVal){
  // wrapper for the TString variant of this function
  return this->getTagStringDefault (key,defaultVal).Data();
}
std::vector<std::string > TQTaggable::getTagVStandardString (const TString& key){
  // wrapper for the TString variant of this function
  std::vector<std::string> retval;
  for(const auto& v:this->getTagVString (key)){
    retval.push_back(v.Data());
  }
  return retval;
}





