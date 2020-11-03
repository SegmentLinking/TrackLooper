#include "TObject.h"
#include "TFile.h"
#include "TKey.h"
#include "QFramework/TQFolder.h"
#include "QFramework/TQTaggable.h"
#include "QFramework/TQUtils.h"
#include "TClass.h"
#include "QFramework/TQCounter.h"
#include "QFramework/TQIterator.h"
#include "QFramework/TQSample.h"
#include "QFramework/TQValue.h"
#include "TParameter.h"
#include "TCollection.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "THnBase.h"
#include "QFramework/TQHistogramUtils.h"
#include "QFramework/TQTHnBaseUtils.h"
#include "QFramework/TQStringUtils.h"
#include "QFramework/TQListUtils.h"
#include "THashList.h"
#include "QFramework/TQLink.h"
#include "QFramework/TQImportLink.h"
#include "QFramework/TQTable.h"
#include "QFramework/TQXSecParser.h"

// #define _DEBUG_

#include "QFramework/TQLibrary.h"

#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <map>
#include "dirent.h"

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQFolder:
//
// The TQFolder class is the basic building block for data modeling within the analysis code.
// It is a subclass of ROOT's TFolder class and thereby serves as a container for any object
// based on the TObject class. Objects are identified by their name (the name returned by
// obj->GetName()). The TQFolder class allows to recursively build up a tree-like folder
// hierarchy by adding instances of TQFolder (simply referred to as 'folder') to existing
// folders. These nested folders can be accessed via a Unix-like path scheme. Instances of
// TQFolder can be browsed using the TBrowser.
//
// Please note: the TQFolder class imposes quite strict requirements on names used to identify
// folders or objects within folders (only small and capital letters, numerals, underscore, and
// dot are valid characters to compose a valid name).
//
// Additionally, the TQFolder class inherits from the TQTaggable class which provides features
// to assign tags (key-value-pairs).
//
// A new instance of TQFolder can be created using the static method TQFolder::newFolder(...):
//
// TQFolder * f = TQFolder::newFolder("f");
//
// Objects are added to the folder using TQFolder::addObject(...):
//
// TH1F * h = new TH1F("h", "h", 10, 0., 1.);
// f->addObject(h);
//
// [Please note: histograms are special in the sense that they are by default associated to a
// directory (instance of TDirectory). This is no problem unless you want to stream (using
// TObject::Write()) a folder containing histograms to a file. You should remove histograms
// before adding them to a folder from their directory using TH1::SetDirectory(NULL).]
//
// [Please note: the user is still able to use TFolder::Add(...) to add objects to a folder,
// however, one should never use this method because it might not treat TQFolder specific
// aspects correctly and result in inconsistencies of the data structure.]
//
// The contents of a folder can be printed using TQFolder::print(...):
//
// f->print("trd");
//
// Objects are retrieved from a folder using TQFolder::getObject(...):
//
// TH1F * h = (TH1F*)f->getObject("h");
//
// Existing subfolders are retrieved using TQFolder::getFolder(...):
//
// TQFolder * subf = f->getFolder("subf");
//
// New subfolders can also be created using TQFolder::getFolder(...):
//
// TQFolder * subf = f->getFolder("subf+");
//
// Tags (see also TQTaggable class) are set using e.g. TQFolder::setTagInteger(...)
//
// f->setTagInteger("number", 6);
//
// and retrieved using e.g. TQFolder::getTagInteger(...)
//
// int number = 0;
// f->getTagInteger("number", number);
//
// Folders can be moved and copied using e.g. TQFolder::moveTo(...) and TQFolder::copyTo(...).
// Folders can be streamed (using TObject::Write()) to ROOT files and retrieved from ROOT files
// using e.g.
//
// TQFolder * f = TQFolder::loadFolder("filename.root:f");
//
// The contents (folder hiearchy + tags) of a folder can be exported to human-readable text or
// written directly into a text file:
//
// TString text = f->exportToText();
//
// f->exportToTextFile("text.txt");
//
// The corresponding syntax also allows to import subfolders and tags from a text or a text
// file:
//
// f->importFromText(text);
//
// f->importFromTextFile("text.txt");
//
// The equivalence of two instances of the TQFolder class (including tags) can be assessed:
//
// f->isEquivalentTo(f2);
//
// Two instances of the TQFolder class can be compared by identifying and printing non-
// equivalent elements:
//
// f->printDiff(f2);
//
// f->printDiff("f1", "f2");
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQFolder)


//__________________________________________________________________________________|___________

TQFolder::TQFolder() :
TFolder("unknown", ""),
  TQTaggable(),
  fMyDir(gDirectory),
  fOwnMyDir(false),
  fBase(NULL),
  isFullyResolved(false){
  // Default constructor of TQFolder class: a new and empty instance of TQFolder is
  // created and initialized. Its name will be set to "unkown". Please note: users
  // should not use this constructor but the static factory method
  // TQFolder::newFolder(...). This default constructor has to be present to allow
  // ROOT's CINT to stream instances of TQFolder.
}


//__________________________________________________________________________________|___________

TQFolder::TQFolder(const TString& name) :
  TFolder(TQFolder::isValidName(name) ? name : "invalid_name", ""),
  TQTaggable(),
  fMyDir(gDirectory),
  fOwnMyDir(false),
  fBase(NULL),
  isFullyResolved(false){
  // Constructor of TQFolder class: a new and empty instance of TQFolder is created
  // and initialized. Its name will be set to the value of the parameter "name_" if
  // it is a valid name and "invalid_name" otherwise. Please refer to the
  // documentation of the static method TQFolder::isValidName(...) for details on
  // valid folder names. Please note: users should not use this constructor but the
  // static factory method TQFolder::newFolder(...).
}


//__________________________________________________________________________________|___________

bool TQFolder::IsSortable() const {
  // This method is a reimplementation of ROOT's TObject::IsSortable() and will
  // return true indicating that instances of the TQFolder class can be sorted
  // reasonably.

  // instances of TQFolder are sortable
  return true;
}


//__________________________________________________________________________________|___________

int TQFolder::Compare(const TObject * obj) const {
  // This method is a reimplementation of ROOT's TObject::Compare(...) and allows
  // to compare an instance of TObject to this instance of TQFolder. It will
  // return 0 if an instance of TQFolder with the same name as this instance is
  // passed as an argument and +1 (-1) if the name of this instance is
  // alphabetically classified after (before) the name of the instance passed as an
  // argument. In case of passing an instance that is not a valid sub-class of
  // TQFolder -1 is returned.

  if (obj && obj->InheritsFrom(TQFolder::Class())) {
    TString thisName = GetName();
    TString thatName = obj->GetName();
    return thisName.CompareTo(thatName);
  } else {
    return -1;
  }
}


//__________________________________________________________________________________|___________

TQFolder * TQFolder::newFolder(TString name) {
  // Returns a new and empty instance of the TQFolder class with name <name> if
  // <name> is a valid name and a null pointer otherwise. Please refer to the
  // documentation of the static method TQFolder::isValidName(...) for details on
  // valid folder names.

  // check if <name> is a valid name for instances of TQFolder
  if (isValidName(name)) {
    // create and return a new instance of TQFolder with name <name>
    return new TQFolder(name);
  } else {
    // return NULL pointer since <name> is not a valid folder name
    return NULL;
  }
}


//__________________________________________________________________________________|___________

bool TQFolder::isLazy(){
  // return true if this folder is 'lazy'.
  // lazy folders are not fully resolved, i.e. components of the folder
  // structure still reside on disk and have not been loaded into memory.
  return !(this->isFullyResolved);
}

//__________________________________________________________________________________|___________

TQFolder * TQFolder::newInstance(const TString& name) {
  // Returns a new and empty instance of the TQFolder class with name <name> if
  // <name> is a valid name and a null pointer otherwise. Please note: this method
  // does exactly the same as TQFolder::newFolder(...) but may be overwritten by
  // sub-classes of TQFolder.

  // create and return new instance of TQFolder with name <name>
  return TQFolder::newFolder(name);
}

//__________________________________________________________________________________|___________

bool TQFolder::parseDestination(TString dest, TString &path, TString &newName) {
  // parse a destination string, i.e. 'path::newname'
  // if the pattern was recognized, path and newName will be set accordingly.
  // else, the entire intput will be used as path and newName will be empty
  // this function will return true in either case and only return false if
  // something about the input string was horribly wrong, e.g. if the newName
  // component has length zero
  int colonpos = TQStringUtils::find(dest,"::");
  if((colonpos >= 0) && (colonpos < dest.Length())){
    path = dest(0,colonpos);
    if(colonpos+2 < dest.Length()){
      newName = dest(colonpos+2, dest.Length()-(colonpos+2));
      return true;
    } else {
      return false;
    }
  } else {
    path = dest;
    newName.Clear();
    return true;
  }
  return false;
}

//__________________________________________________________________________________|___________

bool TQFolder::parseLocation(TString importPath, TString& filename, TString& objname){
  // parse a "location string", i.e. 'filename:objname'
  // if the pattern was recognized, filename and objname will be set accordingly.
  // else, the entire input will be used as filename, and objname will be empty.
  // this function shoudl return true in any case.
  filename.Clear();
  objname.Clear();
  size_t pipepos = TQStringUtils::rfind(importPath,">>");
  size_t pos = TQStringUtils::rfind(importPath,":",(pipepos < (size_t)importPath.Length()) ? pipepos : (size_t)importPath.Length());
  while(pos > 2 && importPath[pos-1] == ':')
    pos = TQStringUtils::rfind(importPath,":",pos-2);
  if(pos < (size_t)importPath.Length() && importPath[pos+1] != '/'){
    filename = TQStringUtils::trim(importPath(0,pos));
    objname = importPath(pos+1,importPath.Length()-pos);
    DEBUGclass("filename='%s', objname='%s'",filename.Data(),objname.Data());
    return true;
  } else {
    DEBUGclass("unable to parse location '%s', >>@%d, :@%d",importPath.Data(),pipepos,pos);
    filename = importPath;
    objname = "";
    return true;
  }
  return false;
}

//__________________________________________________________________________________|___________

TFile* TQFolder::openFile(TString& importPath,const TString& opt){
  // open a file for retrieval of a TQFolder instance. the importPath can be
  // given as 'filename:objname' for details on the parsing, see
  // TQFolder::parseLocation. the opt string can be used to specify the opening
  // mode (like in TFile::Open). the importPath argument will be stripped of
  // the filename component and the separating colon by this function. this
  // function will not take ownage of the file pointer, and it's the users
  // responsibility to close and delete it.
  TString filename,objname;
  DEBUGclass("function called on path '%s'",importPath.Data());
  if(!parseLocation(importPath,filename,objname)) return NULL;
  DEBUGclass("parsed location: filename='%s', objname='%s'",filename.Data(),objname.Data());
  importPath = objname;
  // check the file's existence and stop if it doesn't exist
  if (filename.IsNull() || !TQUtils::fileExists(filename)) return NULL;
  // try to open the file
  TFile* file = TFile::Open(filename.Data(), opt);
  if (!file) return NULL;
  if (!file->IsOpen()) {
    delete file;
    return NULL;
  }
  return file;
}

//__________________________________________________________________________________|___________

TObject * TQFolder::importObject(TString importPath,bool recurse) {
  // import an object from the given path, to be given in the typical
  // 'filename:objname' notation. if the filename component is left empty, the
  // current gDirectory will be used.
  TDirectory* dir = this->fMyDir;
  importPath = TQStringUtils::trim(importPath);
  bool local = false;
  if (TQStringUtils::countLeading(importPath,":")>0) local = true;
  DEBUGclass("function called on path '%s'",importPath.Data());
  TFile* file = this->openFile(importPath,"READ");
  if(file)
    dir = file;
  else if (!local) {
    ERRORclass("Failed to retrieve object: File not found!");
    return NULL;
  } else if(!dir)
    dir = gDirectory;


  TObject * imported = NULL;

  if(importPath.IsNull()){
    ERRORclass("cannot import object without name or path - did you forget the ':[name]'?");
  } else {
    DEBUGclass("importing object to path '%s'",importPath.Data());
    imported = importObjectFromDirectory(dir, importPath,recurse);
  }

  // close file and delete file pointer
  if(file){
    file->Close();
    delete file;
  }

  return imported;
}

//__________________________________________________________________________________|___________

const TString& TQFolder::getValidNameCharacters() {
  // Returns a string containing all valid characters that can be used to build
  // valid names of instances of TQFolder or TObjects stored inside an instance of
  // TQFolder.

  // Valid characters are the ones defined as 'default
  // ID characters' in the TQStringUtils class.
  return TQStringUtils::getDefaultIDCharacters();
}


//__________________________________________________________________________________|___________

bool TQFolder::isValidName(const TString& name) {
  // Checks whether the name <name> passed as an argument is a valid name for an
  // instance of TQFolder and return true if <name> is valid and false otherwise.
  // Valid names may consist of letters a..z, A..Z, numerals 0..9, underscores "_"
  // and dots ".". Please note: names only consisting of dots (e.g. "." or "..")
  // are not considered as valid names as these have special meanings.

  // don't allow names like ".." or "."
  if (name.CountChar('.') == name.Length()) {
    return false;
  } else {
    return TQStringUtils::isValidIdentifier(name, getValidNameCharacters(), 1, -1);
  }
}


//__________________________________________________________________________________|___________

bool TQFolder::isValidPath(TString path, bool allowRelativePaths, bool allowWildcards, bool allowArithmeticStringExpressions) {
  // Checks whether <path> is a valid path name. A path is considered as valid if
  // all its path tokens (components between slashes "/") are valid names for
  // instances of TQFolder or TObjects stored inside an instance of TQFolder (please
  // refer to the documentation of the static method TQFolder::isValidName(...)
  // for details on valid folder names). If <allowRelativePaths> is true (true by
  // default) relative references like "." or ".." are also considered as valid path
  // tokens. If <allowWildcards> is true (true by default) wildcards like "?" or "*"
  // are also considered as valid path tokens.


  // a valid path has to consist of at least one character
  bool isValid = !path.IsNull();

  // loop over path tokens and test their validity
  while (isValid && !path.IsNull()) {
    // read one path token
    TString pathToken = getPathHead(path);
    // check the token's validity: it has to be a valid name or ...
    isValid &= isValidName(pathToken)
      // a relative reference in case <allowRelativePaths> is true or ...
      || (allowRelativePaths &&
          (pathToken.CompareTo(".") == 0 || pathToken.CompareTo("..") == 0))
      // a wildcard in case <allowWildcards> is true
      || (allowWildcards &&
          (pathToken.CompareTo("*") == 0 || pathToken.CompareTo("?") == 0))
      // some arithmetic string expression in case <allowArithmeticStringExpressions> is true
      || (allowArithmeticStringExpressions &&
          TQStringUtils::matches(pathToken,"[*]"));
  }

  // return true if the path is valid
  return isValid;
}


//__________________________________________________________________________________|___________

TString TQFolder::makeValidIdentifier(TString identifier, TString replacement) {
  // Makes the name <identifier> a valid name of instances of TQFolder or TObjects
  // stored inside an instance of TQFolder (please refer to the documentation of the
  // static method TQFolder::isValidName(...) for details on valid folder names).
  // Invalid name characters in <identifier> are replaced by <replacement> (default
  // is an empty string effectively removing such characters). If <replacement>
  // itself does not contain invalid name characters the string returned is guaranteed
  // to be a valid name of instances of TQFolder or TObjects stored inside an instance
  // of TQFolder.

  return TQStringUtils::makeValidIdentifier(identifier, getValidNameCharacters(), replacement);
}


//__________________________________________________________________________________|___________

TString TQFolder::makeValidPath(TString path, TString replacement,
                                bool allowRelativePaths, bool allowWildcards) {
  // Make the path <path> a valid path (please refer to the static method
  // TQFolder::isValidPath(...) for details on valid paths). Invalid name characters
  // in <path> are replaced by <replacement> (default is an empty string effectively
  // removing such characters). If <replacement> itself does not contain invalid name
  // characters the string returned is guaranteed to be a valid path.
  // If <allowRelativePaths> is true (true by default) relative references like "."
  // or ".." are kept and if <allowWildcards> is true (true by default) wildcards like
  // "?" or "*" are also kept.

  // keep track of leading and trailing slahes "/" since these will undesirably be
  // removed by TQFolder::getPathHead(...) in the following and thus need to be added
  // later again
  bool leadingSlash = TQStringUtils::removeLeading(path, "/", 1);
  bool trailingSlash = TQStringUtils::removeTrailing(path, "/", 1);

  TString result;
  while (!path.IsNull()) {
    TString subPath = TQFolder::getPathHead(path);
    if ((allowRelativePaths &&
         (subPath.CompareTo(".") == 0 || subPath.CompareTo("..") == 0))
        || (allowWildcards &&
            (subPath.CompareTo("*") == 0 || subPath.CompareTo("?") == 0))) {
      result = TQFolder::concatPaths(result, subPath);
    } else {
      result = TQFolder::concatPaths(result,
                                     TQFolder::makeValidIdentifier(subPath, replacement));
    }
  }

  // add leading and trailing slashes again
  // if these were present at the beginning
  if (leadingSlash) {
    result.Prepend("/");
  }
  if (trailingSlash) {
    result.Append("/");
  }

  // return the cleaned path
  return result;
}


//__________________________________________________________________________________|___________

TString TQFolder::getPathHead(TString &path) {
  // Removes the first path token (head) of <path> and returns it. (content of <path>
  // will be changed). Examples:
  //
  // - TString path = "hello";
  // getPathTail(path) returns "hello" and <path> becomes an empty string.
  // - TString path = "hello/world";
  // getPathTail(path) returns "hello" and <path> becomes "world".
  // - TString path = "hello/world/test";
  // getPathTail(path) returns "hello" and <path> becomes "world/test".

  // find the first occurence of a slash "/" in <path>
  // but excluding the very first character (index = 0)
  int pos = 1;
  while (pos < path.Length() && path[pos] != '/') {
    pos++;
  }

  int start = path.BeginsWith("/") ? 1 : 0;
  TString result = path(start, pos - start);
  path.Remove(0, pos + 1);
  return result;
}


//__________________________________________________________________________________|___________

TString TQFolder::getPathTail(TString &path) {
  // Removes the last path token (tail) of <path> and returns it (content of <path>
  // will be changed). Examples:
  //
  // - TString path = "hello";
  // getPathTail(path) returns "hello" and <path> becomes an empty string.
  // - TString path = "hello/world";
  // getPathTail(path) returns "world" and <path> becomes "hello".
  // - TString path = "hello/world/test";
  // getPathTail(path) returns "test" and <path> becomes "hello/world".

  // find the last occurence of a slash "/" in <path>
  // but excluding the very last character (index = length - 1)
  int pos = path.Length() - 2;
  while (pos >= 0 && path[pos] != '/') {
    pos--;
  }

  if (pos >= 0) {
    TString result = path(pos + 1,
                          path.Length() - pos - 1 - (path.EndsWith("/") ? 1 : 0));
    path.Remove(pos);
    return result;
  } else {
    TString result = path;
    path.Clear();
    return result;
  }
}


//__________________________________________________________________________________|___________

TString TQFolder::getPathWithoutHead(TString path) {
  // Returns <path> without the first path token (head). Examples:
  //
  // getPathWithoutHead("hello") returns an empty string
  // getPathWithoutHead("hello/world") returns "world"
  // getPathWithoutHead("hello/world/test") returns "world/test"

  // remove the head
  getPathHead(path);
  // return what's left
  return path;
}


//__________________________________________________________________________________|___________

TString TQFolder::getPathWithoutTail(TString path) {
  // Returns <path> without the last path token (tail). Examples:
  //
  // getPathWithoutTail("hello") returns an empty string
  // getPathWithoutTail("hello/world") returns "hello"
  // getPathWithoutTail("hello/world/test") returns "hello/world"

  // remove the tail
  getPathTail(path);
  // return what's left
  return path;
}

//__________________________________________________________________________________|___________

TString TQFolder::concatPaths(TString path1, TString path2) {
  // Concatenates the two paths <path1> and <path2> ensuring exactly one intermediate
  // slash "/" between the two in the resulting path. In case one path is an empty
  // string the respective other one is returned. Examples:
  //
  // - concatPaths("a", "b") returns "a/b".
  // - concatPaths("a/", "/b") returns "a/b".
  // - concatPaths("a/", "b") returns "a/b".
  // - concatPaths("a/b", "c") returns "a/b/c".

  // remove trailing (leading) slahes of the first (second) path
  TQStringUtils::removeTrailingText(path2,"/.");
  TQStringUtils::removeTrailingText(path1,"/.");
  TQStringUtils::removeLeadingText(path2,"./");
  TQStringUtils::removeTrailing(path1, "/");
  if (!path1.IsNull()) TQStringUtils::removeLeading(path2, "/");

  // concatenate the two paths ...
  if (!path1.IsNull() && !path2.IsNull()) {
    // ... adding one intermediate "/" in case neither of them is empty
    return path1 + "/" + path2;
  } else {
    return path1 + path2;
  }
}


//__________________________________________________________________________________|___________

TString TQFolder::concatPaths(TString path1, TString path2, TString path3) {
  // Concatenates the three paths <path1>, <path2>, and <path3> ensuring exactly one
  // intermediate slash "/" between each of them in the resulting path. In case one
  // path is an empty string it is ignored. Examples:
  //
  // - concatPaths("a", "b", "c") returns "a/b/c".
  // - concatPaths("a/", "b", "/c") returns "a/b/c".
  // - concatPaths("a/", "", "/c") returns "a/c".

  // nest fundamental version of TQFolder::concatPaths(...) to obtain the result
  return concatPaths(concatPaths(path1, path2), path3);
}


//__________________________________________________________________________________|___________

TString TQFolder::concatPaths(TString path1, TString path2, TString path3, TString path4) {
  // Concatenates the four paths <path1>, <path2>, <path3>, and <path3> ensuring
  // exactly one intermediate slash "/" between each of them in the resulting path.
  // In case one path is an empty string it is ignored. Examples:
  //
  // - concatPaths("a", "b", "c", "d") returns "a/b/c/d".

  // nest fundamental version of TQFolder::concatPaths(...) to obtain the result
  return concatPaths(concatPaths(path1, path2), concatPaths(path3, path4));
}


//__________________________________________________________________________________|___________

int TQFolder::countPathLevels(TString path, bool checkPathTokens) {
  // Counts the number of path levels (valid names for instances of TQFolder separated
  // by slashes "/") in input path <path>. If <path> is not a valid path -1 is returned
  // (relative paths, e.g. "..", and wildcards, e.g. "?", are not considered as valid
  // and will result in -1 as return value). Examples:
  //
  // - countPathLevels("") returns 0.
  // - countPathLevels("hello") returns 1.
  // - countPathLevels("hello/world") returns 2.
  // - countPathLevels("hello/?") returns -1.

  // the number of path levels
  int levels = 0;

  // count path levels by looping and removing heads one by one
  while (!path.IsNull()) {
    // extract next level
    TString subPath = getPathHead(path);
    // check if the path token is a valid name ...
    if (isValidName(subPath) || !checkPathTokens) {
      // ... increase number if yes
      levels++;
    } else {
      // ... return -1 if path token is not a valid name
      return -1;
    }
  }

  // return the number of path levels
  return levels;
}

//__________________________________________________________________________________|___________

bool TQFolder::isEquivalentTo(TQFolder * f, const TString& options) {
  // returns true if the folder is equivalent to another one, e.g. if all
  // elements present in one have a corresponding element of the same name
  // present in the other as well.

  TQTaggable * opts = TQTaggable::parseFlags(options);

  if (!opts) {
    ERRORclass("Failed to parse options '%s'", options.Data());
    return false;
  }

  bool result = false;
  result = this->isEquivalentTo(f, *opts);
  delete opts;
  return result;
}


//__________________________________________________________________________________|___________

bool TQFolder::isEquivalentTo(TQFolder * f, TQTaggable& options) {
  // returns true if the folder is equivalent to another one, e.g. if all
  // elements present in one have a corresponding element of the same name
  // present in the other as well.
  if (!f) {
    return false;
  }

  bool equivalent = true;

  if (!this->tagsAreEquivalentTo(f)) {
    equivalent = false;
  }

  // get sorted list of object names
  TList * objects = TQListUtils::getMergedListOfNames(
                                                      this->GetListOfFolders(), f->GetListOfFolders(), false);

  // iterate over all elements present in at least one of the two instances of TQFolder
  // (in order to set fIsEquivalentToSnapshot (see below) properly on the whole TQFolder
  // tree the iteration does not stop even if an inequivalence has been found)
  TQIterator itrObjects(objects, true);
  while (itrObjects.hasNext()) {
    TString name = itrObjects.readNext()->GetName();

    // try to get elements from both instances
    TObject * obj1 = this->FindObject(name);
    TObject * obj2 = f->FindObject(name);

    if (!obj1 || !obj2) {
      // element is not present in both instances => not equivalent
      equivalent = false;
      continue;
    }

    if (obj1->InheritsFrom(TQFolder::Class()) && obj2->InheritsFrom(TQFolder::Class())) {
      // comparing two instances of the TQFolder class
      if (!((TQFolder*)obj1)->isEquivalentTo((TQFolder*)obj2, options)) {
        equivalent = false;
      }
    } else if (!TQUtils::areEquivalent(obj1, obj2)) {
      equivalent = false;
    }
  }

  // fIsEquivalentToSnapshot is an internal boolean representing the result of the
  // last TQFolder::isEquivalentTo(...) call. It is read by TQFolder::printDiff(...)
  // to avoid a qudratically scaling number of calls to TQFolder::isEquivalentTo(...).
  fIsEquivalentToSnapshot = equivalent;

  return equivalent;
}


//__________________________________________________________________________________|___________

bool TQFolder::printDiff(const TString& path1, const TString& path2, const TString& options) {
  // print the difference between two folders, comparing them recursively
  TQFolder * f1 = this->getFolder(path1);
  TQFolder * f2 = this->getFolder(path2);

  if (f1 && f2) {
    return f1->printDiff(f2, options);
  } else {
    ERRORclass("Failed to find folder '%s'", f1 ? path2.Data() : path1.Data());
    return false;
  }
}


//__________________________________________________________________________________|___________

bool TQFolder::printDiff(TQFolder * f, const TString& options) {
  // print the difference between two folders, comparing them recursively
  TQTaggable * opts = TQTaggable::parseFlags(options);

  if (!opts) {
    ERRORclass("Failed to parse options '%s'", options.Data());
    return false;
  }

  bool result = false;
  result = this->printDiff(f, *opts, 0);
  delete opts;
  return result;
}


//__________________________________________________________________________________|___________

bool TQFolder::printDiff(TQFolder * f, TQTaggable& options, int indent) {
  // print the difference between two folders, comparing them recursively
  const int cColWidth_Total = TQLibrary::getConsoleWidth();

  const int cColWidth_Name = 0.5*cColWidth_Total;
  const int cColWidth_Comp = 0.2*cColWidth_Total;
  const int cColWidth_Details = cColWidth_Total - cColWidth_Name - cColWidth_Comp;

  if (!f) {
    return false;
  }

  if (indent == 0) {
    // if this is the root of printDiff(...) scan folders and fill fIsEquivalentToSnapshot
    this->isEquivalentTo(f, options);
  }
  if (fIsEquivalentToSnapshot) {
    // stop here if everything downstream is equivalent
    return true;
  }

  // print details of individual elements?
  //@tag: [d] If this argument tag is set to true, details on individual elements are printed. Default: false.
  bool printDetails = options.getTagBoolDefault("d", false);

  // get sorted list of object names: this is a list of all names
  // of objects appearing in any of the two folders but sorted and
  // without duplicates
  TList * objects = TQListUtils::getMergedListOfNames(
                                                      this->GetListOfFolders(), f->GetListOfFolders(), false);

  TString line;
  TString comp;

  if (indent == 0) {
    // print headline
    line = TQStringUtils::fixedWidth("Name", cColWidth_Name, "l");
    line.Append(TQStringUtils::fixedWidth("Comparison", cColWidth_Comp, "l"));
    if (printDetails) {
      line.Append(TQStringUtils::fixedWidth("Details (1)", cColWidth_Details, "l"));
      line.Append(TQStringUtils::fixedWidth("Details (2)", cColWidth_Details, "l"));
    }
    std::cout << TQStringUtils::makeBoldWhite(line) << std::endl;
    std::cout << TQStringUtils::makeBoldWhite(TQStringUtils::repeat("=", line.Length())) << std::endl;
  } else {
    // list my name
    line = TQStringUtils::fixedWidth(TQStringUtils::repeatSpaces((indent - 1) * 2) + TQStringUtils::makeBoldBlue(this->GetName()) + "/", cColWidth_Name, "l");
    std::cout << line << std::endl;
  }

  // print diff of tags associated to these folders
  // (prints nothing in case tags are equivalent)
  TQTaggable tagDiffOpts;
  tagDiffOpts.setTagBool("z", true);
  tagDiffOpts.setTagBool("m", true);
  tagDiffOpts.setTagInteger("i", indent);
  tagDiffOpts.setTagBool("d", printDetails);
  this->printDiffOfTags(f, tagDiffOpts);

  TQIterator itrObjects(objects, true);
  while (itrObjects.hasNext()) {
    TString name = itrObjects.readNext()->GetName();

    // grab object from both folders
    TObject * obj1 = this->FindObject(name);
    TObject * obj2 = f->FindObject(name);

    if (obj1 && obj2) {
      if (obj1->InheritsFrom(TQFolder::Class()) && obj2->InheritsFrom(TQFolder::Class())) {
        // print diff of matching subfolders
        ((TQFolder*)obj1)->printDiff((TQFolder*)obj2, options, indent + 1);
        continue;
      }
      if (TQUtils::areEquivalent(obj1, obj2)) {
        continue;
      }
      if (obj1->IsA() == obj2->IsA()) {
        // objects are of same class type but cannot be compared
        comp = TQStringUtils::makeBoldYellow("(1) =? (2)");
      } else if (obj1->InheritsFrom(obj2->IsA())) {
        comp = TQStringUtils::makeBoldYellow("(1) -> (2)");
      } else if (obj2->InheritsFrom(obj1->IsA())) {
        comp = TQStringUtils::makeBoldYellow("(1) <- (2)");
      } else {
        comp = TQStringUtils::makeBoldRed("(1) != (2)");
      }
    } else if (!obj2) {
      if (obj1->InheritsFrom(TQFolder::Class())) {
        name = TQStringUtils::makeBoldBlue(name) + "/";
      }
      comp = TQStringUtils::makeBoldRed("(1) - ");
    } else if (!obj1) {
      if (obj2->InheritsFrom(TQFolder::Class())) {
        name = TQStringUtils::makeBoldBlue(name) + "/";
      }
      comp = TQStringUtils::makeBoldRed(" - (2)");
    }

    line = TQStringUtils::fixedWidth(TQStringUtils::repeatSpaces(indent * 2) + name, cColWidth_Name, "l");
    line.Append(TQStringUtils::fixedWidth(comp, cColWidth_Comp, "l"));
    std::cout << line.Data() << std::endl;
  }

  return false;
}

//__________________________________________________________________________________|___________

void TQFolder::printContents(const TString& options) {
  // this is an alias for the 'print' function
  this->printInternal(options,0,true);
}

//__________________________________________________________________________________|___________

void TQFolder::print(const TString& options) {
  // Prints the contents of this instance of TQFolder via std::cout. The following options
  // may be specified:
  //
  // - "r" recursively print contents of subfolders
  // - "r<N>" (<N> replaced by a positive integer number) like "r" but
  // does not go deeper than <N> levels
  // - "d" prints an additional column summarizing details of the
  // corresponding objects
  // - "h" do not print contents of folders starting with a dot (".")
  // recursively (if used with option "r")
  // - "H" like "h" but do not even list folders starting with a dot (".")
  // - "c" count and display the number of elements in each sub-folder
  // - "C" like "c" but count the number of elements recursively
  // - "l" additionally shows indentation lines
  // - "f[<filter>]" only shows elements whose names match <filter> (allows use
  // of wildcards "*" and "?" in the usual "ls-like" way)
  // - "t" additionally prints tags associated to instances of TQFolder
  // - "t[<filter>]" like "t" but only prints tags whose keys match <filter>
  //
  // The print command can be redirected to another folder by prepending the
  // corresponding path followed by a colon to the list of options, e.g.
  // "hello/world:trd". If wildcards are used in this path the print command will
  // be redirected to the first match only.
  this->printInternal(options,0,true);
}

//__________________________________________________________________________________|___________

void TQFolder::printInternal(TString options, int indent,bool resolve) {
  // internal pretty-print function
  // for documentation, please refer to the public variant

  // define the width of table columns as constants
  // TODO: make these parameters changeable options
  const int cColWidth_Total = TQLibrary::getConsoleWidth();
  const int cColWidth_Name = std::min((int)(0.5*cColWidth_Total),60);
  const int cColWidth_Class = std::min((int)(0.2*cColWidth_Total),30);
  const int cColWidth_Details = cColWidth_Total - cColWidth_Name - cColWidth_Class;

  /* parse the options string
   * =================================================== */

  // ===== redirection =====

  // read the path to print
  TString path = TQStringUtils::readPrefix(options, ":");

  // redirect print
  if (!path.IsNull()) {

    // get the folder to print
    TQFolder * folderToPrint = this->getFolder(path);

    // print if folder exists or throw an error message if not
    if (folderToPrint) {
      folderToPrint->printInternal(options,indent,resolve);
    } else {
      WARNclass("unknown path '%s'",path.Data());
    }

    // we are done here
    return;
  }


  TString flags;
  TString objectFilter;
  TString tagFilter;
  TString localOptions = options;

  int maxDepth = 0;
  bool hasMaxDepth = false;
  bool hasTagFlag = false;
  bool makeTeX = false;

  // read flags and filter definition
  bool stop = false;
  while (!stop) {

    // read flags without parameter
    if (TQStringUtils::readToken(localOptions, flags, "dhHcCl") > 0)
      continue;

    if (TQStringUtils::removeLeadingText(localOptions, "TeX")){
      makeTeX = true;
    }

    // read object filter flag "f" and filter definition
    if (TQStringUtils::readToken(localOptions, flags, "f", 1) > 0) {

      // don't allow multiple filters
      if (objectFilter.Length() > 0) {
        ERRORclass("cannot define more than one object filter using 'f'");
        return;
      }

      // expect filter definition after 'f' option
      if (!(TQStringUtils::readBlock(localOptions, objectFilter, "[]") > 0
            && objectFilter.Length() > 0)) {
        ERRORclass("filter definition expected after option 'f'");
        return;
      }

      continue;
    }

    // read recursion flag "r" and max depth
    if (TQStringUtils::readToken(localOptions, flags, "r", 1) > 0) {

      // read max depth definition after 'r' option
      TString maxDepthStr;
      if (TQStringUtils::readToken(localOptions, maxDepthStr,
                                   TQStringUtils::getNumerals()) > 0) {
        // don't allow multiple maximum recursion depths
        if (hasMaxDepth) {
          ERRORclass("cannot define more than one maximum recursion depth using 'r'");
          return;
        } else {
          maxDepth = maxDepthStr.Atoi();
          if(maxDepth == 0) resolve = false;
          hasMaxDepth = true;
        }
      } else {
        maxDepth = -1;
        hasMaxDepth = true;
      }

      continue;
    }

    // read tag flag "t" and filter definition
    if (TQStringUtils::readToken(localOptions, flags, "t", 1) > 0) {

      // don't allow multiple filters
      if (hasTagFlag) {
        ERRORclass("cannot define more than one tag filter using 't'");
        return;
      } else {
        hasTagFlag = true;
      }

      // read filter definition after 't' option
      TQStringUtils::readBlock(localOptions, tagFilter, "[]");

      continue;
    }

    // no valid tokens left to parse
    stop = true;
  }

  // unexpected options left?
  if (localOptions.Length() > 0) {
    ERRORclass("unknown option '%c'", localOptions[0]);
    return;
  }

  // take care of import links
  if(resolve) this->resolveImportLinks(false);

  // parse the flags
  bool flagDetails = flags.Contains("d");
  bool flagRecursive = flags.Contains("r");
  bool hideElements = flags.Contains("h");
  bool hideAll = flags.Contains("H");
  bool countElements = flags.Contains("c");
  bool countElementsRec = flags.Contains("C");
  bool flagIndentLines = flags.Contains("l");

  /* print the headline if level of indentation is zero
   * =================================================== */

  if (indent == 0) {
    if(!makeTeX){
      TString headline;
      headline.Append(TQStringUtils::fixedWidth("Name", cColWidth_Name, "l"));
      headline.Append(TQStringUtils::fixedWidth("Class", cColWidth_Class, "l"));
      if (flagDetails)
        headline.Append(TQStringUtils::fixedWidth("Details", cColWidth_Details, "l"));
      std::cout << TQStringUtils::makeBoldWhite(headline) << std::endl;
      std::cout << TQStringUtils::makeBoldWhite(TQStringUtils::repeat("=", TQStringUtils::getWidth(headline))) << std::endl;
    } else {
      std::cout << "\\documentclass{standalone}\n";
      std::cout << "\\usepackage[T1]{fontenc}\n";
      std::cout << "\\usepackage{pxfonts}\n";
      std::cout << "\\usepackage[html]{xcolor}\n";

      std::cout << "\\definecolor{folder} {HTML}{5555FF}\n";
      std::cout << "\\definecolor{samplefolder}{HTML}{55FF55}\n";
      std::cout << "\\definecolor{hidden} {HTML}{FF55FF}\n";
      std::cout << "\\definecolor{tag} {HTML}{00AAAA}\n";
      std::cout << "\\providecommand{\\tabularnewline}{\\\\}\n";
      std::cout << "\\newcommand\\hindent{\\hspace{2em}}\n";
      std::cout << "\\newcommand\\stylefolder[1]{{\\bfseries\\color{folder}#1}}\n";
      std::cout << "\\newcommand\\stylesamplefolder[1]{{\\bfseries\\color{samplefolder}#1}}\n";
      std::cout << "\\newcommand\\styletag[1]{{\\color{tag}#1}}\n";
      std::cout << "\\newcommand\\stylehidden[1]{{\\bfseries\\color{hidden}#1}}\n";

      std::cout << "\\begin{document}\\ttfamily\n";
      std::cout << "\\begin{tabular}{l l";
      if(flagDetails) std::cout << " l";
      std::cout << "}\n";
      std::cout << "\\bfseries Name & \\bfseries Class";
      if(flagDetails) std::cout << " & \\bfseries Details";
      std::cout << "\\tabularnewline\\hline\\hline\n";
    }
  }


  TString indentStr;
  if (flagIndentLines)
    indentStr = TQStringUtils::getIndentationLines(indent);
  else if(makeTeX){
    indentStr = TQStringUtils::repeat("\\hindent{}",indent);
  } else {
    indentStr = TQStringUtils::repeatSpaces(indent * 2);
  }


  if (hasTagFlag && getNTags() > 0) {
    // import tags
    TList * tagKeys = this->getListOfKeys(tagFilter);
    if (tagKeys) {
      TQIterator itr(tagKeys);
      while(itr.hasNext()){
        TObject* obj = itr.readNext();

        TString key = obj->GetName();
        TString type;
        this->getTypeOfTagAsString(key, type);
        TString details;
        if (flagDetails) {
          this->getValueOfTagAsString(key, details);
        }

        if(makeTeX){
          std::cout << indentStr;
	  key.ReplaceAll("_","\\_");
          std::cout << "\\styletag{<" << key << ">}";
          std::cout << " & ";
          std::cout << "<Tag:" << type << ">";
          if(flagDetails){
            std::cout << " & ";
	    details.ReplaceAll("$","\\$");
	    details.ReplaceAll("_","\\_");
            std::cout << details;
          }
          std::cout << "\\tabularnewline";
          std::cout << std::endl;
        } else {
          std::cout << indentStr;
          std::cout << TQStringUtils::fixedWidth(TQStringUtils::makeTurquoise((TString)"<"+key+">"), cColWidth_Name - 2*indent, "l");
          std::cout << TQStringUtils::fixedWidth(TString::Format("<Tag:%s>",type.Data()), cColWidth_Class, "l");
          if(flagDetails)
            std::cout << TQStringUtils::fixedWidth(details, cColWidth_Details, "l");
          std::cout << std::endl;
        }
      }
    }
  }


  // iterate over every element in the folder
  TQIterator itr(this->GetListOfFolders());
  while(itr.hasNext()){
    TObject* obj = itr.readNext();

    // get object and class name
    TString objName = obj->GetName();
    TString className = obj->IsA()->GetName();

    // skip elements starting with "." if "H" options was specified
    if (hideAll && objName.BeginsWith("."))
      continue;

    TString bareName = objName;
    TString nameAppendix;

    if (obj->InheritsFrom(TQFolder::Class())) {

      // skip folder if a filter is specified which the folder doesn't pass
      if (!objectFilter.IsNull()
          && (!flagRecursive || ((TQFolder*)obj)->getNObjects(objectFilter, true) == 0)
          && !TQStringUtils::matchesFilter(objName, objectFilter, ",", true))
        continue;

      nameAppendix.Append("/");

      // append the number of elements in the folder (if requested)
      if (countElements || countElementsRec) {
        // count the elements
        int nElements = -1;
        int nElementsRec = -1;
        if (countElements)
          nElements = ((TQFolder*)obj)->getNElements(false);
        if (countElementsRec)
          nElementsRec = ((TQFolder*)obj)->getNElements(true);
        // append the number of elements as string
        if (countElements && countElementsRec && (nElements != nElementsRec))
          nameAppendix = TString::Format(" [c:%d,C:%d]", nElements, nElementsRec);
        else if (countElements)
          nameAppendix = TString::Format(" [%d]", nElements);
        else if (countElementsRec)
          nameAppendix = TString::Format(" [%d]", nElementsRec);
      }
    } else {
      // skip element if a filter is specified which the element doesn't pass
      if (objectFilter.Length() > 0 && !TQStringUtils::matches(objName, objectFilter))
        continue;
    }

    TString details;
    if (flagDetails) {
      details = TQStringUtils::getDetails(obj);
    }

    std::cout << indentStr;

    if(makeTeX){
      bareName.ReplaceAll("_","\\_");
      if (obj->InheritsFrom(TQFolder::Class())) {
        if (obj->InheritsFrom(TQSampleFolder::Class())) {
          std::cout << "\\stylesamplefolder{" << bareName << "}";
        } else if (objName.BeginsWith(".")) {
          std::cout << "\\stylehidden{" << bareName << "}";
        } else {
          std::cout << "\\stylefolder{" << bareName << "}";
        }
      } else {
        std::cout << bareName;
      }
      std::cout << nameAppendix;
      std::cout << " & " << className;
      if (flagDetails)
        std::cout << " & " << details;
      std::cout << "\\tabularnewline";
      std::cout << std::endl;
    } else {
      if (obj->InheritsFrom(TQFolder::Class())) {
        if (obj->InheritsFrom(TQSampleFolder::Class())) {
          bareName = TQStringUtils::makeBoldGreen(bareName);
        } else if (objName.BeginsWith(".")) {
          bareName = TQStringUtils::makeBoldPink(bareName);
        } else {
          bareName = TQStringUtils::makeBoldBlue(bareName);
        }
      }
      std::cout << TQStringUtils::fixedWidth(bareName+nameAppendix, cColWidth_Name-2*indent, "l.");
      std::cout << TQStringUtils::fixedWidth(className, cColWidth_Class, "l");
      if (flagDetails)
        std::cout << TQStringUtils::fixedWidth(details, cColWidth_Details, "l");
      std::cout << std::endl;
    }

    // print the contents of subfolders recursively
    if ((maxDepth == -1 || maxDepth > indent) && obj->InheritsFrom(TQFolder::Class())
        && !(hideElements && objName.BeginsWith(".")))
      ((TQFolder*)obj)->printInternal(options, indent + 1,true);

  }
  if(indent == 0 && makeTeX){
    std::cout << "\\end{tabular}\n\\end{document}" << std::endl;
  }

}


//__________________________________________________________________________________|___________

bool TQFolder::checkConsistency(bool verbose) {
  // Checks the consistency of system of bidirectional pointers between instances of
  // TQFolder and their base folders and returns true if all references are consistent
  // and false otherwise. It prints an error message in case <verbose> is true and
  // an inconsistency is found.

  bool failed = false;

  // loop over all subfolders in this folder
  TQIterator itr(this->getListOfFolders(), true);
  while (itr.hasNext()) {
    TObject * obj = itr.readNext();
    if (!obj->InheritsFrom(TQFolder::Class())) {
      // ignore objects not an instance of TQFolder
      continue;
    }
    TQFolder * folder = (TQFolder*)obj;
    // check if base folder of subfolder is this folder
    if (folder->getBase() != this) {
      if (verbose) {
        ERRORclass("Folder '%s' in folder '%s' has pointer to wrong base folder", folder->GetName(),this->getPath().Data());
      }
      failed = true;
      // check consistency recursively
    } else if (!folder->checkConsistency(verbose)) {
      failed = true;
    }
  }

  return !failed;
}


//__________________________________________________________________________________|___________

bool TQFolder::isBaseOf(TQFolder * folder) {
  // Returns true if this instance of TQFolder is a (not necessarily direct) base
  // folder of <folder> and false otherwise.

  // stop if invalid folder given: no is-base-of relation
  if (!folder) {
    return false;
  }

  // get base folder of <folder> and stop if
  // it has no base folder (returning false)
  TQFolder * base = folder->getBase();
  if (!base) {
    return false;
  }

  // Check if direct base of <folder> is this folder and ...
  if (this == base) {
    // ... return true if yes and ...
    return true;
  } else {
    // ... recursively check isBaseOf(...) otherwise
    return isBaseOf(base);
  }
}


//__________________________________________________________________________________|___________

bool TQFolder::removeObject(const TString& name) {
  // Removes the reference of the object referred to by <path> from the corresponding
  // instance of TQFolder and returns true if has been removed successfully. The
  // object itself is not affected. [Please note: this method is a wrapper to
  // TQFolder::deleteObject(name, true).]

  return (deleteObject(name, true) > 0);
}


//__________________________________________________________________________________|___________

int TQFolder::deleteObject(TString path, bool removeOnly) {
  // Deletes/removes the object referred to by <path> from the corresponding
  // instance of TQFolder and return the number of deleted/removed objects. If
  // <removeOnly> is false (this is the default) the corresponding objected is deleted
  // and its memory is freed. If <removeOnly> is true only the object's reference is
  // removed. If a dash "-" is appended to the object's reference the containing
  // instances of TQFolder are deleted/removed recursively as well if and only if by
  // the operation an instance of TQFolder has become empty. [Please note: if
  // <removeOnly> is true the corresponding instances of TQFolder won't be deleted but
  // only detached from its base folders. One should take care in this case to not
  // loose the pointers to these folders since the inter-reference between them is
  // lost.] If an exclamation mark "!" is appended to the object's reference this
  // method can be used to delete non-empty instances of TQFolder including its
  // content. Please note: if "-" and "!" are combined one needs to append "!-"
  // (and NOT "-!").
  //

  // search for the local object to delete (or to contain the object to delete)

  const bool collapse = (TQStringUtils::removeTrailing(path,"-")>0);
  const bool force = (TQStringUtils::removeTrailing(path,"!")>0);

  TQIterator itr(this->getListOfObjectPaths(path),true);
  // the number of objects deleted/removed
  int nDel = 0;
  while(itr.hasNext()){
    TObject* name = itr.readNext();
    TString objpath(name->GetName());
    TString objname = TQFolder::getPathTail(objpath);

    TQFolder* f = objpath.IsNull() ? this : this->getFolder(objpath);
    if(!f) continue;

    TObject* obj = f->getObject(objname);
    TQFolder* objf = dynamic_cast<TQFolder*>(obj);

    if(objf){
      if(objf->isEmpty() || force){
        DEBUGclass("deleting folder '%s' in '%s' %s",objf->GetName(),objf->getPath().Data(),(objf->isEmpty() ? "(empty)" : "(forced)"));
        objf->detachFromBase();
        nDel++;
        if(!removeOnly) delete objf;
      }
    } else {
      DEBUGclass("deleting object '%s' in '%s' (class: %s)",obj->GetName(),objf->getPath().Data(),obj->Class()->GetName());
      f->Remove(obj);
      nDel++;
      if(!removeOnly) delete obj;
    }
    while(collapse && f->isEmpty()){
      TQFolder* tmpf = f->getBase();
      f->detachFromBase();
      delete f;
      f = tmpf;
    }
  }

  // return the total number of objects deleted
  return nDel;
}


//__________________________________________________________________________________|___________

int TQFolder::deleteAll() {
  // Deletes all elements of this folder. This will also delete subfolders including
  // their elements recursively. Returns the total number of deleted objects.
  // Remark: If the folder contains TQImportLinks, only the links will be deleted.
  // The corresponding content that still resides on disk will not be touched
  // and can still be accessed via TQFolder::loadFolder()

  // the number of objects that have been deleted
  int nDel = 0;

  // loop over elements in the folder and delete them
  TQIterator itr(GetListOfFolders());
  while (itr.hasNext()) {
    // the object
    TObject * obj = itr.readNext();

    // remove the element from this folder
    Remove(obj);

    // delete contents of local object if it is a folder
    // (only needed to get the correct number of objects deleted)
    if (obj->InheritsFrom(TQFolder::Class())) {
      nDel += ((TQFolder*)obj)->deleteAll();
    }

    // delete the object itself
    delete obj;
    nDel++;
  }

  // return the number of objects that have been deleted
  return nDel;
}


//__________________________________________________________________________________|___________

void TQFolder::sortByNameRecursive() {
  // Sorts the elements of this folder and all its subfolders by
  // ascending alphabetical order. This does not affect the objects
  // itself but only the order of references (pointers to these
  // objects) kept by this instance of TQFolder.
  TQFolderIterator itr(this->getListOfFolders("*"),true);
  while(itr.hasNext()){
    TQFolder* f = itr.readNext();
    if(!f) continue;
    if(f->isEmpty()) continue;
    f->sortByName();
  }
}

//__________________________________________________________________________________|___________

int TQFolder::sortByName() {
  // Sorts the elements of this folder by ascending alphabetical order and returns
  // the number of elements processed. This does not affect the objects itself but
  // only the order of references (pointers to these objects) kept by this instance
  // of TQFolder.

  // create a list (TList) of names of elements in
  // this folder (stored as instances of TObjString)
  std::vector<TString> names;
  TQIterator itr1(this->GetListOfFolders());
  while (itr1.hasNext()) {
    names.push_back(itr1.readNext()->GetName());
  }

  // sort the list of names
  std::sort(names.begin(),names.end());

  // iterate over the (sorted) list of names and remove the corresponding
  // object from the list of elements and add it back again (this effectively
  // results in sorting the elements by name since these references are
  // ordered according to the order of insertion)
  for(auto name:names){
    TObject * obj = this->getObject(name);
    this->Remove(obj);
    this->Add(obj);
  }

  // return the number of elements processed
  return names.size();
}


//__________________________________________________________________________________|___________

TQFolder * TQFolder::detachFromBase() {
  // Detaches this instance of TQFolder from its base folder and returns a pointer
  // to this folder (which became the root of this tree of folders by this operation).
  // This method ensures that the bidirectional reference between base folder and
  // this folder are updated properly.

  // if this instance of TQFolder has a base folder ...
  if (fBase) {
    // ... remove this folder from the list of folders of the base folder and ...
    fBase->Remove(this);
    // ... remove the reference to the base folder
    fBase = 0;
  }

  // return a pointer to this instance of TQFolder
  return this;
}


//__________________________________________________________________________________|___________

bool TQFolder::moveTo(TQFolder * dest, const TString& newName) {
  // Moves this instance of TQFolder to the folder <dest> and returns true in case
  // of success and false otherwise. The operation is not performed in case this
  // instance of TQFolder is a base folder of the destination folder. Optionally, a
  // new name <newName> of this instance of TQFolder can be specified to be applied
  // upon successfully having moved it to the destination folder. This method ensures
  // that the bidirectional reference between base folder and this folder are updated
  // properly.



  // need a valid destionation folder
  if (!dest) {
    DEBUGclass("Null pointer TQFolder destination in moveTo");
    return false;
  }
  DEBUGclass(TString::Format("Will now try to move this folder to destination folder %s", dest->getPath().Data()));

  // cannot move folder to itself
  if (dest == this) {
    DEBUGclass("Cannot move folder to itself");
    return false;
  }

  // in case the destination folder is the same as the current base folder ...
  if (dest == fBase) {
    // ... just perform the renaming
    if (!newName.IsNull() && isValidName(newName)) {
      this->SetName(newName.Data());
      return true;
    } else {
      DEBUGclass("newName is null or is not valid");
      return false;
    }
  }

  // this instance of TQFolder must not be a base folder of the destination folder
  if (this->isBaseOf(dest)) {
    DEBUGclass("this instance of TQFolder is a base folder of the destination folder!");
    return false;
  }

  // try to move this instance of TQFolder by temporarily removing its
  // reference to its base folder and adding it to the destination folder
  TQFolder * tmpBase = fBase;
  fBase = NULL;
  TObject * add;
  if (newName.IsNull()) {
    add = dest->addObject(this);
  } else {
    // additionally set a new name
    add = dest->addObject(this, newName + "::");
  }

  if (add) {
    // successfully moved this instance of TQFolder to destination folder
    // => remove reference of former base folder to this instance of TQFolder
    if (tmpBase) {
      tmpBase->Remove(this);
    }
    return true;
  } else {
    // failed to move this instance of TQFolder
    // => restore reference to base folder
    fBase = tmpBase;
    DEBUGclass("Failed to move instance of TQFolder");
    return false;
  }
}


//__________________________________________________________________________________|___________

bool TQFolder::moveTo(const TString& dest) {
  // Moves this instance of TQFolder to the one referred to by <dest> and returns
  // true in case of success and false otherwise. The operation is not performed in
  // case this instance of TQFolder is a base folder of the destination folder.
  // Optionally, the folder can be renamed upon successfully moving it to its new
  // location by appending "::" followed by the new name to <dest>. This method
  // ensures that the bidirectional reference between base folder and this folder
  // are updated properly. Examples:
  //
  // - moveTo("/hello") moves this folder to "/hello"
  // - moveTo("/hello :: myNewName") moves this folder to "/hello" and renames it
  // to "myNewName".
  //
  // [Please note: this is a wrapper to TQFolder::moveTo(TQFolder * dest, ...).]

  // split up the destination path into <path> and <newName> (separated by "::")
  DEBUGclass(TString::Format("moveTo called with argument %s", dest.Data()));
  TString path;
  TString newName;
  if (!parseDestination(dest, path, newName)) {
    DEBUGclass("Failed to parse destination in moveTo(dest)");
    return false;
  }

  // now get the destination folder and move this instance to it and rename it
  DEBUGclass(TString::Format("Will now try to move this folder to path %s and with new name %s", path.Data(), newName.Data()));
  return moveTo(this->getFolder(path), newName);
}


//__________________________________________________________________________________|___________

bool TQFolder::moveFolder(const TString& source, const TString& dest) {
  // Moves the instance of TQFolder referred to by <source> to the one referred to
  // by <dest> and returns true in case of success and false otherwise. The operation
  // is not performed in case the source instance of TQFolder is a base folder of
  // the destination folder. Optionally, the folder can be renamed upon successfully
  // moving it to its new location by appending "::" followed by the new name to <dest>.
  // This method ensures that the bidirectional reference between base folder and this
  // folder are updated properly. Please note that this method does NOT behave
  // exactly as the shell command "mv" does. Examples:
  //
  // - moveFolder("/hello", "/world") moves the folder "/hello" to "/world" (resulting
  // in its new path "/world/hello")
  // - moveFolder("/hello", "/world ::myNewName") moves the folder "/hello" to "/world"
  // and renames it to "myNewName" (resulting in its new path "/world/myNewName")
  //
  // [Please note: this is a wrapper to TQFolder::moveTo(TQFolder * dest, ...).]

  // get the source folder
  TQFolder * src = this->getFolder(source);
  if (!src) {
    return false;
  }

  // split up the destination path into <path> and <newName> (separated by "::")
  TString path;
  TString newName;
  if (!parseDestination(dest, path, newName)) {
    return false;
  }

  // now get the destination folder and move the source folder to it and rename it
  return src->moveTo(this->getFolder(path), newName);
}


//__________________________________________________________________________________|___________

TQFolder * TQFolder::copy(const TString& newName) {
  // Creates an independent copy of this instance of TQFolder including all its
  // sub-elements and return a pointer to it. Optionally, a new name <newName> can
  // be assigned to the copy.

  if (!newName.IsNull() && !TQFolder::isValidName(newName)) {
    return NULL;
  }

  // temporarily hide the base folder because ROOT's streamer facility
  // (invoked by Clone()) would stream the base folder as well
  TQFolder * tmpBase = fBase;
  fBase = 0;

  // clone this folder
  TQFolder * copy = (TQFolder*)this->Clone();

  // restore the reference to the base folder
  fBase = tmpBase;

  // return the copy with new name
  copy->SetName(newName.IsNull() ? this->GetName() : newName.Data());
  return copy;
}


//__________________________________________________________________________________|___________

TQFolder * TQFolder::copyTo(TQFolder * dest) {
  // Creates an independet copy of this instance of TQFolder (including all its
  // sub-elements), adds it to <dest>, and returns a pointer to the copy in case of
  // success and a NULL pointer otherwise.

  // need a valid instance of TQFolder to copy this folder to
  if (!dest) {
    return NULL;
  }

  // make a copy of this instance of TQFolder ...
  TQFolder * copy = this->copy();

  // ... and try to add it to the destination folder
  if (dest->addObject(copy)) {
    // return a pointer to the copy in case of success
    return copy;
  } else {
    // delete the copy again and return a NULL pointer in case of failure
    delete copy;
    return NULL;
  }
}


//__________________________________________________________________________________|___________

TQFolder * TQFolder::copyTo(const TString& dest) {
  // Creates an independet copy of this instance of TQFolder (including all its
  // sub-elements), adds it to the instance of TQFolder referred to by <dest>, and
  // returns a pointer to the copy in case of success and a NULL pointer otherwise.
  //
  // [Please note: this is a wrapper to TQFolder::copyTo(TQFolder * dest).]

  // make copy
  return this->copyTo(this->getFolder(dest));
}


//__________________________________________________________________________________|___________

TQFolder * TQFolder::copyFolder(const TString& source, const TString& dest) {
  // Creates an independet copy of this instance of the TQFolder at the path
  // <source> (including all its sub-elements), adds it to the instance of
  // TQFolder referred to by <dest>, and returns a pointer to the copy in case
  // of success and a NULL pointer otherwise.
  //
  // [Please note: this is a wrapper to TQFolder::copyTo(TQFolder * dest).]

  TQFolder * src = this->getFolder(source);
  if (!src) {
    return NULL;
  }

  return src->copyTo(this->getFolder(dest));
}


//__________________________________________________________________________________|___________

int TQFolder::getNElements(bool recursive, TClass * tclass) {
  // Returns the number of elements (objects and subfolders) in this folder. If
  // recursive = true, the number of elements in subfolders is added recursively.
  // If a class is specified, elements are only counted if they inherit from that
  // class

  this->resolveImportLinks(recursive);

  // the number of elements to be returned
  int nElements = 0;

  // loop over every element in the folder
  TIterator * itr = GetListOfFolders()->MakeIterator();
  TObject * obj;
  while ((obj = itr->Next())) {

    // count this element
    if (!tclass || obj->InheritsFrom(tclass))
      nElements++;

    // count the subelements recursively
    if (recursive && obj->InheritsFrom(TQFolder::Class()))
      nElements += ((TQFolder*)obj)->getNElements(true, tclass);

  }

  // delete the iterator
  delete itr;

  return nElements;

}


//__________________________________________________________________________________|___________

int TQFolder::getNObjects(const TString& nameFilter, bool recursive) {
  // return the number of objects matching the given name filter
  this->resolveImportLinks(recursive);
  // the number of objects
  int nObjects = 0;
  // loop over every element in the folder
  TQIterator itr(GetListOfFolders());
  while(itr.hasNext()){
    TObject* obj = itr.readNext();
    if (nameFilter.IsNull() || TQStringUtils::matches(TString(obj->GetName()), nameFilter))
      nObjects++;
    if (recursive && obj->InheritsFrom(TQFolder::Class()))
      nObjects += ((TQFolder*)obj)->getNObjects(nameFilter, recursive);
  }
  return nObjects;
}


//__________________________________________________________________________________|___________

bool TQFolder::isEmpty() const {
  // Returns true if this instance of TQFolder does not contain any element and
  // false otherwise.
  TCollection* c = this->GetListOfFolders();
  if(!c) return true;
  return (c->GetEntries() == 0);
}


//__________________________________________________________________________________|___________

int TQFolder::getSize(bool memoryOnly) {
  // Returns an estimated size (in bytes) of this instance of TQFolder including
  // its sub-elemets in memory. The number returned is mainly calculated from the
  // estimated size of histograms stored within the folder hierarchy (using the
  // method TQHistogramUtils::estimateSize(...)) and a contribution for any other
  // type of object obtained from the C++ function sizeof(...). Please note that
  // this does determine the size of the pointer rather than the size of the object
  // itself. Thus, TQFolder::getSize() does only provide reasonable results if most
  // elements within the folder hierarchy are histograms (instances of TH1).

  if(!memoryOnly)
    this->resolveImportLinks(true);

  // the size of this folder
  int size = sizeof(*this);

  // loop over every element in the folder
  TQIterator itr(GetListOfFolders());
  while(itr.hasNext()){
    TObject* obj = itr.readNext();
    // add the size of every element
    if (obj->InheritsFrom(TQFolder::Class()))
      size += ((TQFolder*)obj)->getSize();
    else if (obj->InheritsFrom(TH1::Class()))
      size += TQHistogramUtils::estimateSize((TH1*)obj);
    else
      size += sizeof(*obj);

  }
  // return the size of this hierachy in bytes
  return size;
}


//__________________________________________________________________________________|___________

TString TQFolder::getSizeAsString(bool memoryOnly) {
  // TODO: move this to TQStringUtils

  // the size
  double size = (double)getSize(memoryOnly);

  // make the size a readable number
  int power = 0;
  while ((size >= 1000.)) {
    power++;
    size /= 1000.;
  }

  TString unit = "B";
  if (power == 1)
    unit.Prepend("k");
  else if (power == 2)
    unit.Prepend("M");
  else if (power == 3)
    unit.Prepend("G");
  else if (power == 4)
    unit.Prepend("T");
  else
    unit.Prepend("?");

  return TString::Format("%.1f %s", size, unit.Data());
}


//__________________________________________________________________________________|___________

void TQFolder::setBase(TQFolder * base_) {
  // set the base folder
  fBase = base_;
}


//__________________________________________________________________________________|___________

TQTaggable * TQFolder::getBaseTaggable() const {
  // retrieve the base folder as TQTaggable
  return fBase;
}


//__________________________________________________________________________________|___________

TList * TQFolder::getDescendantTaggables() {
  return this->getListOfFolders("?");
}


//__________________________________________________________________________________|___________

TList * TQFolder::getListOfTaggables(const TString& taggables) {
  // retrieve list of subfolders
  return getListOfFolders(taggables);
}

//__________________________________________________________________________________|___________

TList * TQFolder::getTaggablesByName(const TString& taggables) {
  // retrieve list of subfolders
  return getListOfFolders(taggables);
}


//__________________________________________________________________________________|___________

TList * TQFolder::getListOfFolders(const TString& path_, TQFolder * template_, bool toplevelOnly) {
  // retrieve a list of folders under the given path matching the template
  return this->getListOfFolders(path_,template_ ? template_->IsA() : TQFolder::Class(),toplevelOnly);
}

//__________________________________________________________________________________|___________

TList * TQFolder::getListOfFolders(const TString& path_, TClass * tClass, bool toplevelOnly) {
  // retrieve a list of folders under the given path matching the class
  // TODO: handle tag requirements

  // check for comma-separated lists
  if(path_.Contains(",")){
    TList* retval = new TList();
    std::vector<TString> paths = TQStringUtils::split(path_,",");
    for(size_t i=0; i<paths.size(); i++){
      TList* sublist = this->getListOfFolders(paths[i], tClass, toplevelOnly);
      if(sublist){
        retval->AddAll(sublist);
        delete sublist;
      }
    }
    if(retval->GetEntries() < 1){
      delete retval;
      return NULL;
    }
    return retval;
  }

  // remove leading and trailing spaces and tabs
  TString path = TQStringUtils::trim(path_);

  // start at root folder if path starts with "/"
  if (TQStringUtils::removeLeading(path,"/") > 0) {
    // get the root folder
    TQFolder * root = getRoot();
    // return folders relative to root folder
    return root->getListOfFolders(path, tClass, toplevelOnly);
  }

  // stop if path is empty
  if (path.IsNull()) return 0;

  TString find(path);
  TString findNext;

  // split path in this level and next level
  Ssiz_t splitPos = find.First('/');
  if (splitPos != kNPOS) {
    // split in <find>/<findNext>
    findNext = find(splitPos + 1, find.Length());
    find.Remove(splitPos);
    // stop if there is another slash ("//" before)
    if (findNext.BeginsWith("/")) return 0;
  }

  // resulting list of folders
  TList * list = NULL;

  // handle wildcards
  if (find.EqualTo("?") || find.EqualTo("*")) {
    bool stop = false;
    // in case of a "*" wildcard: consider also this folder
    if (find.EqualTo("*")) {
      this->resolveImportLinks(true);

      if (!findNext.IsNull()) {
        // get the list of folders recursively
        list = getListOfFolders(findNext, tClass, toplevelOnly);
        //if we only want the topmost folders we might have to skipp some of the lines below
        stop = toplevelOnly && list && (list->GetEntries() > 0);
      } else if (!tClass || this->InheritsFrom(tClass)){
        /* add this folder to the list if its a valid element*/
        list = new TList();
        list->Add(this);
        stop = toplevelOnly; //if we added this folder to the list and toplevelOnly is true, we should not iterate over subfolders anymore, e.g., to avoid multiple counting
      }

      findNext = path;
    } else {
      this->resolveImportLinks(false);
    }

    // loop over all sub folders
    if (!stop) {
      TQFolderIterator itr(GetListOfFolders());
      while (itr.hasNext()){
        /* consider only valid element: valid pointer to
         * subclass of the template_ (default is TQFolder) */
        TQFolder* obj = itr.readNext();
        if(!obj) continue;
        if (!findNext.IsNull()){
          // get the list of folders recursively
          TList * sublist = obj->getListOfFolders(findNext, tClass, toplevelOnly);
          if (sublist) {
            if (list) {
              list->AddAll(sublist);
              delete sublist;
            } else {
              list = sublist;
            }
          }
        } else {
          if(!tClass || obj->InheritsFrom(tClass)){
            // add this folder to the list
            if (!list) list = new TList();
            list->Add(obj);
          }
        }
      }
    }
  } else if(find.Contains("*")){
    this->resolveImportLinks(false);

    TQFolderIterator itr(GetListOfFolders());
    while (itr.hasNext()){
      TQFolder* obj = itr.readNext();
      if(!obj) continue;
      if(!TQStringUtils::matches(obj->GetName(),find)) continue;
      if (!findNext.IsNull()){
        // get the list of folders recursively
        TList * sublist = obj->getListOfFolders(findNext, tClass, toplevelOnly);
        if (sublist) {
          if (list) {
            list->AddAll(sublist);
            delete sublist;
          } else {
            list = sublist;
          }
        }
      } else {
        if(!tClass || obj->InheritsFrom(tClass)){
          // add this folder to the list
          if (!list) list = new TList();
          list->Add(obj);
        }
      }
    }
  } else {
    TQFolder * element = 0;

    if (find.EqualTo("..")) {
      // requested element is the base folder
      element = getBase();
    } else if (find.EqualTo(".")) {
      // requested element is this folder
      element = this;
    } else {
      // requested element is called <find>
      DEBUGclass("attempting to resolve import link '%s' in '%s'",find.Data(),this->getPath().Data());
      this->resolveImportLink(find,false);
      element = dynamic_cast<TQFolder*>(FindObject(find.Data()));
    }

    /* consider only valid elements: valid pointer to
     * subclass of the template_ (default is TQFolder) */
    if (element){
      if (!findNext.IsNull()) {
        // get the list of folders recursively
        list = element->getListOfFolders(findNext, tClass, toplevelOnly);
      } else {
        // add this folder to the list
        if(tClass ? element->InheritsFrom(tClass) : element->InheritsFrom(TQFolder::Class())){
          list = new TList();

          list->Add(element);
        }
      }

    }

  }
  return list;
}

//__________________________________________________________________________________|___________

std::vector<TString> TQFolder::getFolderPaths(const TString& path_, TClass * tClass, bool toplevelOnly) {
  // returns a std::vector<TString> containing the full paths of folders matching
  // the given requirements.
  std::vector<TString> vec;
  TQFolderIterator itr(this->getListOfFolders(path_,tClass,toplevelOnly),true);
  while (itr.hasNext()) {
    TQFolder * f = itr.readNext();
    if (!f) continue;
    vec.push_back(f->getPath());
  }
  return vec;
}

std::vector<TString> TQFolder::getFolderPathsWildcarded(const TString& path_, TClass * tClass, bool toplevelOnly) {
  // returns a std::vector<TString> containing the full paths of folders matching
  // the given requirements. This variant returns the paths in their wildcarded
  // representation (using the "wildcarded" tag on the respective folders). If
  // multiple folders matching the requirements with the same wildcarded path are
  // found, the path is added to the vector only once.
  std::vector<TString> vec;
  std::map<TString,bool> map;
  TQFolderIterator itr(this->getListOfFolders(path_,tClass,toplevelOnly),true);
  while (itr.hasNext()) {
    TQFolder * f = itr.readNext();
    if (!f) continue;
    map[f->getPathWildcarded()] = true;
  }
  for(std::map<TString,bool>::iterator it = map.begin(); it != map.end(); ++it) {
    vec.push_back(it->first);
  }
  return vec;
}


//__________________________________________________________________________________|___________

TList * TQFolder::getListOfObjects(TString path, TClass * tclass){
  // retrieve a list of objects matching the given path
  TString tail = TQFolder::getPathTail(path);
  TList* retval = new TList();
  if(!path.IsNull()){
    TList* l = this->getListOfFolders(path);
    TQFolderIterator itr(l,true);
    while(itr.hasNext()){
      TQFolder* f = itr.readNext();
      if(!f) continue;
      TList* sublist = f->getListOfObjects(tail,tclass);
      retval->AddAll(sublist);
      delete sublist;
    }
  } else {
    TQIterator objItr(this->GetListOfFolders());
    while(objItr.hasNext()){
      TObject* obj = objItr.readNext();
      if(!obj) continue;
      if(tail.IsNull() || (tail == "?") || (TQStringUtils::matches(obj->GetName(),tail))){
        if(!tclass || obj->InheritsFrom(tclass))
          retval->Add(obj);
      }
    }
  }
  return retval;
}

//__________________________________________________________________________________|___________

TList * TQFolder::getListOfObjectPaths(TString path, TClass * tclass){
  // retrieve a list of objects matching the given path
  TString tail = TQFolder::getPathTail(path);
  TList* retval = new TList();
  retval->SetOwner(true);
  if(!path.IsNull()){
    TList* l = this->getListOfFolders(path);
    TQFolderIterator itr(l,true);
    while(itr.hasNext()){
      TQFolder* f = itr.readNext();
      if(!f) continue;
      TList* sublist = f->getListOfObjectPaths(tail,tclass);
      sublist->SetOwner(false);
      retval->AddAll(sublist);
      delete sublist;
    }
  } else {
    TQIterator objItr(this->GetListOfFolders());
    while(objItr.hasNext()){
      TObject* obj = objItr.readNext();
      if(!obj) continue;
      if(tail.IsNull() || (tail == "?") || (TQStringUtils::matches(obj->GetName(),tail))){
        if(!tclass || obj->InheritsFrom(tclass))
          retval->Add(new TObjString(TQFolder::concatPaths(this->getPath(),obj->GetName())));
      }
    }
  }
  return retval;
}

//__________________________________________________________________________________|___________

TObject * TQFolder::getCopyOfObject(const TString& name_, const TString& path_) {
  TObject* obj = this->getObject(name_,path_);
  if(!obj) return NULL;
  TH1* hist = dynamic_cast<TH1*>(obj);
  if(hist){
    return TQHistogramUtils::copyHistogram(hist,"NODIR");
  }
  return obj->Clone();
}

//__________________________________________________________________________________|___________

TObject * TQFolder::getObject(const TString& name_, const TString& path_) {
  // retrieve an object with the given name from a path
  DEBUGclass("called on '%s' with name='%s', path='%s'",this->getPath().Data(),name_.Data(),path_.Data());
  if (path_.Length() == 0) {
    // find the object in this folder
    return this->FindObject(name_.Data());
  } else if(path_ == "?" || path_ == "*"){ //get the first matching object from the first matching folder which has the requested object 
    TList* l = this->getListOfFolders(path_);
    TQFolderIterator itr(l,true);
    while(itr.hasNext()){
      TQFolder* f = itr.readNext();
      if(!f) continue;
      TObject* obj = f->getObject(name_);
      if(obj) return obj;
    }
    return NULL;
  } else {
    // find the object in the folder called <path_>
    TQFolder * folder = this->getFolder(path_);
    if (folder)
      return folder->getObject(name_);
    else
      return NULL;
  }

}

//__________________________________________________________________________________|___________

TString TQFolder::getObjectPath(TString path) {
  // expand and return the path of some object
  TString name = TQFolder::getPathTail(path);
  TList* l = this->getListOfFolders(TQFolder::concatPaths("*",path));
  TQFolderIterator itr(l,true);
  while(itr.hasNext()){
    TQFolder* f = itr.readNext();
    if(!f) continue;
    TQIterator subitr(f->GetListOfFolders());
    while(subitr.hasNext()){
      TObject* obj = subitr.readNext();
      if(TQStringUtils::matches(obj->GetName(),name)){
        //@tag: [~basepath] This folder tag contains the basepath of the folder, which is prepended to the return value of folder->getPath()
	const TString basepath = f->getTagStringDefault("~basepath",".");
	return basepath + TQFolder::concatPaths(f->getPath(),obj->GetName());

      }
    }
  }
  return "";
}

//__________________________________________________________________________________|___________

TList* TQFolder::getObjectPaths(TString namepattern, TString pathpattern, TClass* objClass) {
  // return the paths of all objects matching the pattern
  // The ownership of the list elements is with the returned list itself.
  
  //the following lines shift any folder name like part of namepattern to pathpattern with a wildcard * in between
  TQStringUtils::ensureTrailingText(pathpattern,"/*");
  pathpattern = TQFolder::concatPaths(pathpattern,namepattern);
  namepattern = TQFolder::getPathTail(pathpattern);

  TQFolderIterator fitr(this->getListOfFolders(pathpattern),true);
  TList* retval = new TList();
  while(fitr.hasNext()){
    TQFolder* f = fitr.readNext();
    TQIterator oitr(f->GetListOfFolders());
    while(oitr.hasNext()){
      TObject* obj = oitr.readNext();
      if(TQStringUtils::matches(obj->GetName(),namepattern) && obj->InheritsFrom(objClass)){
        //@tag: [~basepath] This folder tag contains the basepath of the folder, which is prepended to the return value of folder->getPath()
	const TString basepath = f->getTagStringDefault("~basepath",".");
        retval->Add(new TObjString(basepath + TQFolder::concatPaths(f->getPath(),obj->GetName())));
      }
    }
  }
  return retval;
}

//__________________________________________________________________________________|___________

TString TQFolder::getName() const {
  // retrieve the name of this object
  return this->fName;
}

//__________________________________________________________________________________|___________

void TQFolder::setName(const TString& newName) {
  // set the name of this object
  this->fName = newName;
}

//__________________________________________________________________________________|___________

const TString& TQFolder::getNameConst() const {
  // retrieve a const reference to the name of this object
  return this->fName;
}

//__________________________________________________________________________________|___________

TQFolder * TQFolder::getFolder(TString path_, TQFolder * template_, int * nFolders_) {
  // legacy wrapper for TClass variant
  return this->getFolder(path_,template_ ? template_->Class() : NULL,nFolders_);
}

//__________________________________________________________________________________|___________

TQFolder * TQFolder::getFolder(const TString& path){
  // wrapper to help resolve ambiguous calls
  return this->getFolder(path,NULL);
}

//__________________________________________________________________________________|___________

TQFolder * TQFolder::getFolder(const char* path){
  // wrapper to help resolve ambiguous calls
  return this->getFolder(path,NULL);
}


//__________________________________________________________________________________|___________

TQFolder * TQFolder::getFolder(TString path_, TClass * tclass, int * nFolders_) {
  // Returns the folder (instance of TQFolder) that matches the path pattern <path_>
  // and a NULL pointer in case no match can be found. If more than one folder matches
  // <path_> (because wildcards are used) the first match is returned. Additionally,
  // a new instance of TQFolder is created as requested if it does not already
  // exist and a "+" has been appended to <path_>. The path <path_> may be built
  // up from any number of path levels in either case and an arbitrary number of
  // nested folders may be created by appending one "+" to the full path.
  //
  // Examples:
  //
  // - getFolder("subfolder") returns a pointer to the instance of TQFolder named
  // "subfolder" within this instance of TQFolder if it does exist.
  // - getFolder("subfolder+") returns a pointer to the instance of TQFolder named
  // "subfolder" within this instance of TQFolder and does create it if it does
  // not exist
  // - getFolder("subfolder/fol2+") returns a pointer to the instance of TQFolder
  // named "fol2" within "subfolder" (in turn within this instance of TQFolder)
  // and does create it if it does not exist
  //
  // [Please note: the additional parameters <template_> and <nFolders_> are for
  // internal use only and should not be used by the user.]

  // remove leading and trailing spaces and tabs
  TString path = TQStringUtils::trim(path_);
  if(path.IsNull()) return this;

  DEBUGclass("attempting to retrieve folder '%s' in '%s'",path.Data(),this->getPath().Data());

  // check auto-create option: path ends with "+"
  TQStringUtils::removeTrailing(path,"/");
  bool force = ( TQStringUtils::removeTrailing(path,"!") == 1);
  bool autoCreate = ( TQStringUtils::removeTrailing(path,"+") == 1);
  TQStringUtils::removeTrailing(path,"/");

  // get the list of folders matching path_
  TList * list = getListOfFolders(path, tclass);

  if (list && list->GetEntries() > 0) {

    // assign the number of folders matching <path>
    if (nFolders_)
      *nFolders_ = list->GetEntries();

    // get the first folder in the list
    TQFolder * firstFolder = (TQFolder*)list->At(0);

    // delete the list
    delete list;

    // return the first folder
    return firstFolder;

  } else if (autoCreate) {

    // delete the list
    delete list;

    TString nameToCreate = path;
    TQFolder * baseFolder = this;

    Ssiz_t splitPos = path.Last('/');

    /* the folder to create has subfolders which
     * also have to be created: create them first */
    if (splitPos != kNPOS) {

      // split the path to create into <path>/<name>
      nameToCreate = path(splitPos + 1, path.Length());

      if (splitPos > 0)
        path = path(0, splitPos);
      else
        path = "/.";

      // try to find/create the base
      int * nFolders = new int(0);
      baseFolder = getFolder(path + (force ? "+!" : "+"), tclass, nFolders);

      // we need exatcly one folder matching the base path
      if (*nFolders != 1) { baseFolder = 0; }

      // delete nFolders
      delete nFolders;

    }

    if (baseFolder) {
      /* create a new instance of a folder: clone the
       * template or create a new TQFolder */
      TQFolder * newInstance;
      if (tclass){
        newInstance = (TQFolder*)(tclass->New());
        newInstance->SetName(nameToCreate);
      } else {
        newInstance = new TQFolder(nameToCreate);
      }

      // add this new instance to the base folder
      if (newInstance) {
        if (baseFolder->addFolder(newInstance)) {
          newInstance->setDirectory(this->fMyDir);
          newInstance->autoSetExportName();
          if(force) newInstance->SetName(nameToCreate);
          // we succeeded to create the path
          if (nFolders_) { *nFolders_ = 1; }
          return newInstance;
        } else {
          /* we actually didn't succeed
           * to add the new instance */
          delete newInstance;
        }
      }
    }
  }

  // we didn't find any matching element
  if (nFolders_) { *nFolders_ = 0; }
  return 0;

}

//__________________________________________________________________________________|___________

TQFolder * TQFolder::addFolder(TQFolder * folder_, TString path_, TQFolder * template_) {
  // legacy wrapper for TClass() variant
  return this->addFolder(folder_,path_,template_ ? template_->Class() : NULL);
}

//__________________________________________________________________________________|___________

TQFolder * TQFolder::addFolder(TQFolder * folder_, TString path_, TClass * tclass) {
  // add a folder at the given path

  if (!folder_) return 0;

  if (path_.Length() == 0) {

    // make sure the new sample folder has a valid name
    if (!isValidName(TString(folder_->GetName()))) { return 0; }

    // make sure the new sample folder has no base sample folder
    if (folder_->getBase()) { return 0; }

    // make sure there is no other element with the same name
    if (FindObject(folder_->GetName())) { return 0; }

    // set the base of the new sample folder before adding it
    folder_->setBase(this);

    /* add the new folder and return a
     * pointer on its new base folder */
    Add(folder_);
    return this;

  } else {

    // get the folder to contain the new folder
    TQFolder * folder = getFolder(path_, tclass);

    /* add the new folder and return a
     * pointer on its new base folder */
    if (folder) { return folder->addFolder(folder_, "", tclass); }

  }

  // something went wrong
  return 0;

}


//__________________________________________________________________________________|___________

TList * TQFolder::getListOfObjectNames(TClass * class_, bool recursive, TString path_) {
  // Return the list of names of objects in a folder

  this->resolveImportLinks(recursive);

  if (path_.Length() == 0) {

    // prepare the list to return
    TList * result = new THashList();

    // loop over all elements in the folder
    TIterator * itr = GetListOfFolders()->MakeIterator();
    TObject * obj;
    while ((obj = itr->Next())) {

      /* the element might be a folder, we apply a special
       * treatment because we are only interesed in objects
       * that are not a folder. */
      if (obj->InheritsFrom(TQFolder::Class())) {

        /* folders only have to be considered if the
         * request is recursive (recursive = true). */
        if (recursive) {

          TList * subList = ((TQFolder*)obj)->getListOfObjectNames(class_, true);
          if (subList) {

            // iterate over every element and prepend name of folder
            TIterator * subItr = subList->MakeIterator();
            TObject * subObj;
            while ((subObj = subItr->Next())) {
              // we know the subList only contains TObjStrings
              TString objectName = ((TObjString*)subObj)->GetString();
              // prepend the name of the containing folder
              objectName.Prepend(TString(obj->GetName()) + "/");
              ((TObjString*)subObj)->SetString(objectName.Data());
              // add this to the resulting list
              result->Add(subObj);
            }

            // delete the sublist
            delete subList;
            // delete the iterator
            delete subItr;

          }

        }

      } else if (!class_ || obj->InheritsFrom(class_)) {

        result->Add(new TObjString(obj->GetName()));

      }
    }
    delete itr;

    // return the list
    return result;

  } else {
    TQFolder * folder = getFolder(path_);
    if (folder){
      return folder->getListOfObjectNames(class_, recursive);
    } else {
      return 0;
    }
  }

}


//__________________________________________________________________________________|___________

TString TQFolder::getPath() {
  // Returns the full path of this instance of TQFolder in the hierarchy. For the
  // root folder "/." is returned.
  TString name(this->GetName());
  // to cope with folders that have been auto-renamed to yield valid identifiers, we ask for this "magic tag"
  // which allows us to instead return any other valid name
  //@tag: [.realname] This object tag contains the original name of the object in case it needed to be automatically renamed in order to make a valid name (e.g. replacement of "-" by "_".
  this->getTagString(".realname",name);
  if (fBase) {
    if (fBase == getRoot()) {
      // the base folder is the root
      name.Prepend("/");
      return name;
    } else {
      // the base folder is not the root folder
      return TQFolder::concatPaths(fBase->getPath(), name);
    }
  } else {
    // if there is no base folder, this is the root: return "/."
    return TString("/.");
  }
}

//__________________________________________________________________________________|___________

TString TQFolder::getPathWildcarded() {
  // Returns the full path of this instance of TQFolder in the hierarchy. For the
  // root folder "/" is returned.
  // this version of getPath replaces the names of folders that have the tag
  // wildcarded=true
  // set with a wildcard ('?')

  //@tag: [wildcarded] If this object tag is set to true, the part of the path corresponding to this object in the return value of getPathWildcarded() is replaced with a "?".
  TString name = (this->getTagBoolDefault("wildcarded",false) ? "?" : this->getName());

  if (fBase) {
    if (fBase == getRoot()) {
      // the base folder is the root
      return TString::Format("/%s/", name.Data());
    } else {
      // the base folder is not the root folder
      return TQFolder::concatPaths(fBase->getPathWildcarded(), name)+"/";
    }
  } else {
    // if there is no base folder, this is the root: return "/"
    return TString("/");
  }
}

//__________________________________________________________________________________|___________

TQFolder * TQFolder::getBase(int depth) const {
  // Returns a pointer to the base folder of this instance of TQFolder and a NULL
  // pointer in case it is the root folder of the folder hierarchy. In general, if
  // <depth> is larger than 1 (it is 1 by default) a pointer to the instance of
  // TQFolder <depth> levels up the hierarchy is returned and a NULL pointer if the
  // root folder is traversed before. If <depth> is zero a pointer to this instance
  // of TQFolder is returned.

  if (depth == 1) {
    // the default case: return pointer to base folder
    return fBase;
  } else if (depth == 0) {
    // the trivial case: return a pointer to this folder
    return (TQFolder*)this;
  } else if (depth > 1 && fBase) {
    // the general case: recursively get pointer to requested base folder
    return fBase->getBase(depth - 1);
  } else {
    // either negative <depth> or folder hierarchy not deep enough: return NULL pointer
    return NULL;
  }
}

//__________________________________________________________________________________|___________

int TQFolder::areRelated(const TQFolder* other) const {
  // checks if this folder or the other folder are subfolders of the respective 
  // other. Returns +1*nSteps if 'other' is a subfolder of 'this', -1*nSteps in the inverse case
  // and 0 otherwise. nSteps is the distance between the two folders starting at 1
  // if the two folders are equal. Example: If 'this' is a direct subfolder of 
  // 'other' then -2 is returned
  if (!other) return 0;
  if (this==other) return 1;
  int thisDist = this->getDistToRoot();
  int otherDist = other->getDistToRoot();
  if (thisDist>otherDist) return -1*other->areRelated(this);
  else if (thisDist<otherDist) {
    int tmp = this->areRelated(other->getBase()); 
    return tmp!=0 ? 1+tmp : 0;
  }
  //folders have the same distance from the root node but are not equal -> not related
  return 0;
  
}

//__________________________________________________________________________________|___________

TQFolder * TQFolder::getRoot() {
  // Returns the a pointer to the instance of TQFolder which is the root of this
  // folder hierarchy.

  if (fBase) {
    // if there is a base folder, recursively return root of it
    return fBase->getRoot();
  } else {
    // if there is no base folder, this is the root: return it
    return this;
  }
}


//__________________________________________________________________________________|___________

bool TQFolder::isRoot() {
  // Returns true if this instance of TQFolder is the root of the folder hierarchy
  // (there is no base folder) and false otherwise.

  // true if there is no base folder
  return (fBase == NULL);
}


//__________________________________________________________________________________|___________

int TQFolder::getDistToRoot() const {
  // Returns the number of hierarchy levels to be traversed up to the root folder
  // of this hierarchy.

  if (fBase) {
    // there is a base folder: recursively return distance to root (but increased by one)
    return fBase->getDistToRoot() + 1;
  } else {
    // this is the root folder: return 0
    return 0;
  }
}


//__________________________________________________________________________________|___________

int TQFolder::getDepth() {
  // Returns the number of levels to be traversed down to the deepest element. For
  // an instance of TQFolder that does not contain "sub-instances" of TQFolder the
  // return value is 1.

  this->resolveImportLinks(true);

  // the maximum distance to an element
  int depth = 0;

  // loop over elements in this folder
  TQIterator itr(GetListOfFolders());
  while (itr.hasNext()) {
    TObject * obj = itr.readNext();

    // the local depth of current object (1 by default)
    int localDepth = 1;
    if (obj->InheritsFrom(TQFolder::Class())) {
      // ... but determined recursively in case of instances of TQFolder
      localDepth += ((TQFolder*)obj)->getDepth();
    }

    // keep maximum
    depth = (localDepth > depth) ? localDepth : depth;
  }

  // return total depth (maximum of all local depths found so far)
  return depth;
}


//__________________________________________________________________________________|___________

TList * TQFolder::getTraceToRoot(bool startAtRoot) {
  // retrieve a list of folders containing all hierarchy levels up to the root
  // folder
  TList * list = NULL;
  if (fBase) {
    list = fBase->getTraceToRoot(startAtRoot);
  }

  if (!list) {
    list = new TList();
    list->SetOwner(false);
  }
  if (startAtRoot) {
    list->AddLast(this);
  } else {
    list->AddFirst(this);
  }

  return list;
}

//__________________________________________________________________________________|___________

TQFolder * TQFolder::loadLazyFolder(TString path) {
  // this function works exactly as TQFolder::loadFolder, with the exception that
  // all folder branches that have been externalized/splitted when writing the file
  // will remain in collapsed state until they are accessed.
  // for large files with an excessive use of externalization/splitting, this will
  // significantly speed up accessing the data, since the branches are only expanded
  // on-demand.
  //
  // please note, however, that you will only experience a total speed gain if you
  // only access small fractions of your data. if you plan to read most of the file's data
  // at some point, requiring the expansion of all branches, this 'lazy' feature
  // will only postpone the work of loading the data into RAM to the point where it is accessed
  // bringing no total speed gain.
  return TQFolder::loadFolder(path,true);
}


//__________________________________________________________________________________|___________

TQFolder * TQFolder::loadFolder(TString path, bool lazy) {
  // Loads an instance of TQFolder from an external ROOT file and returns a pointer
  // to it and a NULL pointer in case of failure. The ROOT file and the key within
  // the key within the ROOT file to be read is identified by <path> which has to
  // have the form "filename:keyname", e.g. "file.root:folder".
  // If the folder to load is stored within a structre of TDirectory in the file
  // it can be accessed by prepending the corresponding path to the folder name,
  // e.g. "file.root:dir1/dir2/folder". To load only a subfolder of an instance of
  // TQFolder from the ROOT file one can append the corresponding path to the folder
  // name, e.g. "file.root:folder/subfolder". In this case a pointer to "subfolder"
  // is returned which is made the root folder before
  // the 'lazy' flag will trigger lazy loading, please refer to TQFolder::loadLazyFolder
  // for documentation of this feature.

  // use a dummy folder to import the external folder using TQFolder::importObject(...)
  TQFolder * dummy = TQFolder::newFolder("dummy");
  TString pathname = path;

  TFile* file = TQFolder::openFile(path,"READ");
  if(!file || !file->IsOpen()){
    if(file) delete file;
    ERRORclassargs(path.Data(),"unable to open file!");
    return NULL;
  }

  TObject * imported = dummy->importObjectFromDirectory(file, path,!lazy);

  // the folder to load and return
  TQFolder * folder = NULL;

  // check the folder that has been imported and get the one to return
  if (imported && imported->InheritsFrom(TQFolder::Class())) {

    TQFolder * importedFolder = (TQFolder*)imported;
    while (importedFolder->getBase() != dummy)
      importedFolder = importedFolder->getBase();

    // detach imported folder from dummy folder
    folder = importedFolder->detachFromBase();
  }

  // delete the dummy folder
  delete dummy;

  // close file and delete file pointer
  if(folder){
    if(lazy){
      folder->setDirectoryInternal(file);
      folder->fOwnMyDir = true;
    } else {
      folder->setDirectoryInternal(NULL);
      file->Close();
      delete file;
    }
    folder->autoSetExportName();
  } else {
    file->Close();
    delete file;
  }

  // return the folder
  return folder;
}


//__________________________________________________________________________________|___________

TQFolder * TQFolder::loadFromTextFile(TString filename, bool showErrorMessage) {
  // Creates a new instance of TQFolder, performs an import from an external text
  // file <filename> on it, and returns a pointer to it in case of success and a
  // NULL pointer in case of failure. If <showErrorMessage> is true and an error
  // occurs during the import process the corresponding error message is displayed
  // via std::cout. If the name of the instance returned is set to the filename of the
  // text file with invalid characters replaced by "_".
  //
  // [Please note: this is a wrapper to TQFolder::loadFromTextFile(TString filename,
  // TString &errorMessage).]

  // string to assign a message related to a potential error to
  TString errMsg;

  // try to load a new instance of TQFolder from a text file
  TQFolder * folder = loadFromTextFile(filename, errMsg);

  // in case of an error ...
  if (!folder && showErrorMessage) {
    // print-out the error message provided by TQFolder::loadFromTextFile(...)
    ERRORclass(errMsg);
  }

  // return a pointer to the freshly loaded folder
  return folder;
}


//__________________________________________________________________________________|___________

TQFolder * TQFolder::loadFromTextFile(TString filename, TString &errorMessage) {
  // Creates a new instance of TQFolder, performs an import from an external text
  // file <filename> on it, and returns a pointer to it in case of success and a
  // NULL pointer in case of failure.
  // error messages will be fed into the string given as second argument

  TString name = filename;
  // extract the name of the file (without full path) to obtain
  // a sensible name for the instance of TQFolder to return
  name = TQFolder::getPathTail(name);

  // create a new instance of TQFolder
  TQFolder * folder = TQFolder::newFolder(TQFolder::makeValidIdentifier(name, "_"));

  // import folder from text file
  if (folder->importFromTextFile(filename, errorMessage)) {
    // ... and return it in case of success
    return folder;
  } else {
    // ... or delete it and return a NULL pointer in case of failure
    delete folder;
    return NULL;
  }
}


//__________________________________________________________________________________|___________

TQFolder * TQFolder::addObject(TObject * object, TString destination) {
  // Adds the object <object> to this instance of TQFolder or to the one referred to
  // by <destination> and returns a pointer to the destination folder in case of
  // success or a NULL pointer in case of failure. If destination ends on "!" an
  // existing object with the same name will be replaced by the new one. Default is
  // to keep the old object and return 0 if one exists already.
  DEBUGclass("entering function");
  if (!object) {
    // stop if object to add is invalid
    return NULL;
  }
  //enforce overwrite if destination ends on "!"
  bool overwrite = (TQStringUtils::removeTrailing(destination,"!") >0);
  #ifdef _DEBUG_
  if(overwrite) DEBUGclass("found overwrite flag");
  #endif

  // parse destination path
  TString path;
  if (!destination.IsNull()) {
    DEBUGclass("reading destination");
    TQStringUtils::readUpTo(destination, path, ":");
    path = TQStringUtils::trim(path);
  }

  // parse new name for object and stop if syntax is invalid
  bool newName = false;
  TString name = object->GetName();
  if (!destination.IsNull()) {
    // stop if we won't be able to rename the object
    if (!object->InheritsFrom(TNamed::Class())){
      ERRORclass("object '%s' cannot be renamed!",object->GetName());
      return 0;
    }

    // expect exactly two ":"
    if (TQStringUtils::removeLeading(destination, ":") != 2){
      ERRORclass("invalid number of ':'!");
      return 0;
    }

    // update the name
    name = TQStringUtils::trim(destination);
    newName = true;
  }

  DEBUGclass("in '%s': adding object '%s' to path '%s' with new name '%s'",this->getPath().Data(),object->GetName(),path.Data(),name.Data());

  // stop if the object has an invalid name
  if (!isValidName(name) && !object->InheritsFrom(TObjString::Class())){
    WARNclass("cannot attach object '%s' to path '%s' - invalid name!",object->GetName(),path.Data());
    return NULL;
  }

  /* the object to add might be a folder which is
   * already part of another folder. Stop if so */
  TQFolder* otherF = dynamic_cast<TQFolder*>(object);
  if(otherF && !otherF->isRoot()){
    WARNclass("cannot add subfolder of other folder structure!");
    return NULL;
  }

  // get the destination folder
  TQFolder * folder = this;
  if (!path.IsNull()){
    folder = getFolder(path);
  }

  if (!folder) {
    WARNclass("cannot attach object '%s' to path '%s' - unable to retrieve folder! did you forget the '+' or the '::'?",object->GetName(),path.Data());
    return NULL;
  }

  // stop if the destination folder already contains an object with the same name
  if (folder->hasObject(name)){
    if (overwrite){
      folder->deleteObject(name);
    } else {
      WARNclass("not adding object '%s' to folder '%s' - an object of this name is already present",name.Data(),folder->getPath().Data());
      return NULL;
    }
  }
    // rename and add object
  if (newName)
    ((TNamed*)object)->SetName(name.Data());

  folder->Add(object);

  // set base folder if a folder was added
  if (otherF){
    otherF->setBase(folder);
  }

  // return the folder the object was added to
  return folder;
}

//__________________________________________________________________________________|___________

TDirectory * TQFolder::getDirectory(){
  // retrieve the directory wherein this folder resides
  return this->fMyDir;
}

//__________________________________________________________________________________|___________

int TQFolder::setDirectory(TDirectory* dir, bool own){
  // set the directory of this folder
  // will migrate all data to the new directory, might be slow
  // second argument decides whether the folder owns the directory handed
  int retval = this->resolveImportLinks(this->fMyDir);
  this->clearDirectoryInternal();
  this->setDirectoryInternal(dir);
  this->fOwnMyDir=own;
  return retval;
}

//__________________________________________________________________________________|___________

void TQFolder::clearDirectoryInternal(){
  // clear and delete the directory associated to this folder if owned
  if(this->fOwnMyDir){
    TFile* f = dynamic_cast<TFile*>(this->fMyDir);
    if(f) f->Close();
    delete this->fMyDir;
  }
}

//__________________________________________________________________________________|___________

void TQFolder::setDirectoryInternal(TDirectory* dir){
  // set the directory of this folder
  // internal variant, will not migrate any data
  // DO NOT USE, UNLESS YOU KNOW WHAT YOU ARE DOING
  this->fMyDir = dir;
  TCollection* l = TFolder::GetListOfFolders();
  if(!l) return;
  TQIterator itr(l);
  while (itr.hasNext()) {
    TQFolder* f = dynamic_cast<TQFolder*>(itr.readNext());
    if(!f) continue;
    f->setDirectoryInternal(dir);
  }
}

//__________________________________________________________________________________|___________

TQFolder * TQFolder::addCopyOfObject(TObject * object, TString destination) {
  // Adds an independent copy of the object <object> to this the instance of
  // TQFolder or to the one referred to by <destination> and returns a pointer to
  // the destination folder in case of success or a NULL pointer in case of failure.

  if (!object) {
    // need a valid pointer to an object to add
    // return a NULL pointer in case there is no object to add
    return NULL;
  }

  // use different cloning strategies depending on the type of object
  TObject * copy;
  if (object->InheritsFrom(TQFolder::Class())) {
    // copy instances of TQFolder using its built-in TQFolder::copy() method
    copy = ((TQFolder*)object)->copy();
  } else if (object->InheritsFrom(TH1::Class())) {
    // copy histograms using dedicated TQHistogramUtils::copyHistogram(...) method
    copy = TQHistogramUtils::copyHistogram((TH1*)object,"NODIR");
  } else {
    // copy (clone) object using ROOT's default streamer facility
    copy = object->Clone();
  }
  // copying the object might have failed for some reason
  if (!copy) {
    // ... return a NULL pointer in this case
    return NULL;
  }

  // try to add the new copy to the folder
  TQFolder * folder = addObject(copy, destination);
  // ... this might fail for some reasons as well
  if (!folder) {
    // ... delete the copy again (and return a NULL pointer) in this case
    delete copy;
  }

  // return a pointer to the destination folder (obtained from TQFolder::addObject(...)
  return folder;
}


//__________________________________________________________________________________|___________

bool TQFolder::hasObject(TString name) {
  // return true if the folder has an object of the given name/path in its
  // folder structure
  if (TQStringUtils::removeLeading(name, "/") > 0)
    return getRoot()->hasObject(name);

  // extract the object name
  TString objName = TQFolder::getPathTail(name);

  // search for an object / objects with a name matching <objName>
  if (name.IsNull()) {
    /* search this folder,
     * TODO: allow use of wildcards */
    return (FindObject(objName.Data()) != 0);
  } else {
    /* search another folder,
     * TODO: allow for wildcards
     * TODO: don't allow constructive search ("+") */
    TQFolder * folder = getFolder(name);
    if (folder)
      return folder->hasObject(objName);
    else
      return false;
  }
}

//__________________________________________________________________________________|___________

int TQFolder::resolveImportLinks(bool recurse) {
  // resolve all import links and load all missing components from disk to
  // memory, converting a "lazy" sample folder into a fully expanded one
  return this->resolveImportLinks(NULL,recurse);
}

//__________________________________________________________________________________|___________

int TQFolder::resolveImportLinks(TDirectory * dir,bool recurse) {
  // resolve all import links and load all missing components the given
  // directory to memory, converting a "lazy" sample folder into a fully
  // expanded one
  if(this->isFullyResolved) return 0;
  int nLinks = 0;
  // collect the list of import link names in this folder
  TQIterator itr1(this->GetListOfFolders());
  while (itr1.hasNext()) {
    TObject * lnk = itr1.readNext();
    if(recurse){
      TQFolder* f = dynamic_cast<TQFolder*>(lnk);
      if(f) f->resolveImportLinks(dir,recurse);
    }
    if (lnk->InheritsFrom(TQImportLink::Class())) {
      if (this->resolveImportLink(lnk->GetName(), dir,recurse)) {
        nLinks++;
      }
    }
  }
  this->isFullyResolved = recurse;
  return nLinks;
}


//__________________________________________________________________________________|___________

TObject * TQFolder::resolveImportLink(const TString& linkName, bool recurse) {
  // resolve one specific import link given by name
  return this->resolveImportLink(linkName,NULL,recurse);
}

//__________________________________________________________________________________|___________

TObject * TQFolder::resolveImportLink(const TString& linkName, TDirectory * dir, bool recurse) {
  // resolve one specific import link given by name from a given TDirectory
  if (!TQFolder::isValidName(linkName)) {
    return NULL;
  }
  DEBUGclass("attempting to resolve import link '%s' in '%s' from directory '%p'", linkName.Data(),this->getPath().Data(),dir);
  TObject * obj = this->getObject(linkName);
  TQImportLink * lnk = dynamic_cast<TQImportLink*>(obj);
  if (!lnk) return NULL;


  // get import path
  TString importPath(TQStringUtils::trim(lnk->getImportPath()));

  // detach link
  this->Remove(lnk);

  TObject * imported = NULL;
  if ((TQStringUtils::removeLeading(importPath, ":", 1) != 0)) {
    if(dir){
      imported = this->importObjectFromDirectory(dir,                  importPath,recurse);
    } else {
      imported = this->importObjectFromDirectory(this->getDirectory(), importPath,recurse);
    }
  } else {
    imported = this->importObject(importPath,recurse);
  }

  if (imported) {
    delete lnk;
  } else {
    ERRORclass("unable to resolve import ImportLink '%s' from directory '%s'",lnk->getImportPath().Data(),this->getDirectory()->GetName());
    this->addObject(lnk);
  }

  return imported;
}

//__________________________________________________________________________________|___________

void TQFolder::autoSetExportName(){
  // automatically generate an export name for this folder the export name is
  // required in order to allow writing this folder to disk with a non-zero
  // split value. however, the user should NEVER need to call this function, as
  // this should be handled automatically
  if(this->isRoot()){
    this->fExportName = this->GetName();
  } else {
    TString name(this->getBase()->getExportName());
    if(name.IsNull()){
      this->fExportName = this->GetName();
    } else {
      name += "-" + this->getName();
      TQStringUtils::ensureLeadingText(name,"--");
      this->fExportName = name;
    }
  }
  this->autoSetExportNames();
}

//__________________________________________________________________________________|___________

void TQFolder::autoSetExportNames(){
  // automatically generate an export names for this folder and all its subfolders
  // for further details, refer to TQFolder::autoSetExportName
  TQFolderIterator itr(this->GetListOfFolders());
  while(itr.hasNext()){
    TQFolder* f = itr.readNext();
    if(!f) continue;
    f->autoSetExportName();
  }
}

//__________________________________________________________________________________|___________

void TQFolder::setExportName(const TString& name){
  // set a specific export name for this folder and auto-generate ones for all
  // its subfolders
  // for further details, refer to TQFolder::autoSetExportName
  this->fExportName = name;
  this->autoSetExportNames();
}

//__________________________________________________________________________________|___________

TObject * TQFolder::importObjectFromDirectory(TDirectory * dir, TString importPath, bool recurse) {
  // import an object from a TDirectory and deploy it at the given importPath
  if(importPath.IsNull()){
    ERRORclass("cannot import object without name or path!");
    return NULL;
  }

  // parsing the input path
  // /some/path/objname >> destination
  TString path;
  TQStringUtils::readUpTo(importPath, path, ">");
  path = TQStringUtils::trim(path);
  int nl = TQStringUtils::removeLeading(importPath, ">");
  TString destination = TQStringUtils::trim(importPath);
  if ((destination.IsNull() && nl != 0) || (!destination.IsNull() && nl != 2)) {
    ERRORclass("invalid destination string, please use exactly two pipes '>>' to signify destination!");
    return NULL;
  }

  // retrieving object
  DEBUGclass("attempting to import object from '%s' to '%s' at '%s'",path.Data(),destination.Data(),this->getPath().Data());
  TString objname(TQFolder::getPathHead(path));
  TQIterator itr(dir->GetListOfKeys());
  TObject * object = NULL;
  while(itr.hasNext()){
    TObject* key = itr.readNext();
    TString keyname(key->GetName());
    if( TQStringUtils::matches(keyname, objname ) && !(objname[0] == '*' && keyname[0] == '-')){
      // the second requirement is needed to avoid "*" matching to spare parts starting with "--"
      object = dir->Get(keyname);
      DEBUGclass("matched '%s' to '%s', obtained object '%s'",key->GetName(),objname.Data(),object->GetName());
      break;
    }
  }
  if (!object) {
    DEBUGclass("failed to find object matching '%s' in '%s'",objname.Data(),dir->GetName());
    return NULL;
  }
  DEBUGclass("success!");

  // if the object we got is a TQFolder, we need to look inside
  TQFolder * folder = dynamic_cast<TQFolder*>(object);
  if (folder){
    folder->detachFromBase();
    folder->fMyDir = dir;
    folder->autoSetExportName();
    if(recurse)
      folder->resolveImportLinks(true);
    folder->isFullyResolved = recurse;
    if(!path.IsNull()){
      INFOclass("attempting to retrieve subfolder from '%s' in '%s'",path.Data(),folder->GetName());
      TQFolder* f = folder->getFolder(path);
      if(!f) return NULL;
      f->detachFromBase();
      delete folder;
      folder = f;
      object = f;
    }
  }

  // we can't add TDirectories
  if (object->InheritsFrom(TDirectory::Class())) {
    WARNclass("cannot attach objects of class '%s' to folder!",object->ClassName());
    return NULL;
  }

  // if its a histogram, we need to detach it from the file
  if (object->InheritsFrom(TH1::Class())) {
    ((TH1*)object)->SetDirectory(NULL);
  }

  // add the object
  if (!this->addObject(object, destination)) {
    // we failed to add the object
    WARNclass("failed to attach object '%s' to destination '%s'!",object->GetName(), destination.Data());
    delete object;
    return NULL;
  }

  // return the object imported
  return object;
}

//__________________________________________________________________________________|___________

bool TQFolder::writeFolder(TDirectory * dir, TString name, int depth, bool keepInMemory){
  // writes a TQFolder to a given directory
  // the precise functionality of this function can be customized using optional parameters
  // in the default settings,
  // - dir is the directory to which the folder will be written
  // - exportName is the key under which the folder will be deployed in the directory
  //
  // after this function has been successfully executed, the the given directory will be associated
  // to this folder. further writes via writeFolder or writeUpdate will use this directory
  // if no other (new) directory is given
  //
  // also, all currently collapsed branches of the folder hierarchy will be expanded and rewritten
  // to the new directory.
  //
  // since ROOT has problems with too large TObjects, the optional depth argument allows automatic splitting
  // of a large folder hierarchy into parts. this splitting will on the top level and on each succeeding
  // level of the folder hierarchy, until the depth exceeds the optional depth parameter.
  //
  // the keepInMemory=false flag can be used to automatically remove all folders from memory
  // that have been linked/externalized with the depth-splitting procedure
  // this causes the corresponding branhces of the folder hierarchy to be collapsed
  // these branches are expanded (and the contents are then automatically reloaded from disk)
  // when they are accessed
  // with keepInMemory=false, reading/writing folder contents will take significantly longer
  // however, lots of RAM space will be freed whenever writeFolder or writeUpdate are called
  // with keepInMemory=false
  //
  if (name.IsNull()) {
    name = GetName();
  }
  return writeFolderInternal(dir, name, depth, keepInMemory);
}

//__________________________________________________________________________________|___________

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

bool TQFolder::writeFolderMaxSize(TDirectory * dir, TString name, int maxSizeInMB, bool keepInMemory){
  // this function is still under developement, please use the usual TQFolder::writeFolder function instead
  return writeFolderInternal(dir, name, -1, keepInMemory);
}

//__________________________________________________________________________________|___________

bool TQFolder::writeFolder(TDirectory * dir, int depth, bool keepInMemory){
  // as opposed to TQFolder::writeFolder(TDirectory*dir,TString name, int depth, bool keepInMemory)
  // this function does not take the 'name' argument and automatically used the folder name instead
  // for further documentation on the functionality, please refer to the aforementioned function
  TString name = this->GetName();
  return writeFolderInternal(dir, name, depth, keepInMemory);
}

//__________________________________________________________________________________|___________

bool TQFolder::writeFolderMaxSize(TDirectory * dir, int maxSizeInMB, bool keepInMemory){
  // this function is still under developement, please use the usual TQFolder::writeFolder function instead
  return writeFolderInternal(dir, this->GetName(), -1, keepInMemory);
}

#pragma GCC diagnostic pop

//__________________________________________________________________________________|___________

TString TQFolder::makeExportName(TString exportName){
  // this function will transform the given string into a valid export name
  // for further information, please refer to TQFolder::writeFolder
  // THIS FUNCTION IS FOR INTERNAL USE ONLY
  exportName.ReplaceAll("/", "-");
  exportName.Prepend("--");
  return exportName;
}

//__________________________________________________________________________________|___________


const TString& TQFolder::getExportName(){
  // return the currently set export name of this folder
  // for further information, please refer to TQFolder::writeFolder
  return this->fExportName;
}

//__________________________________________________________________________________|___________

bool TQFolder::split(TDirectory * dir, int depth){
  this->resolveImportLinks();
  this->autoSetExportName();
  TQFolderIterator itr(this->GetListOfFolders());
  if(depth > 0){
    DEBUGclass("@%s: depth > 0, splitting subfolders",this->getPath().Data());
    // collect the list of subfolders
    bool failed = false;
    while (itr.hasNext()) {
      TQFolder* subfolder = itr.readNext();
      if(!subfolder) continue;
      if(!subfolder->split(dir,depth-1)){
        failed = true;
        break;
      }
    }
    if(failed) this->resolveImportLinks(true);
    return !failed;
  } else {
    DEBUGclass("@%s: depth <= 0, splitting here",this->getPath().Data());
    DEBUGclass("setting directory to %x",dir);
    if(dir)
      this->setDirectoryInternal(dir);
    bool failed = false;

    // stop if base directory is invalid
    if (!this->fMyDir) {
      ERRORclass("cannot write to NULL directory");
      return false;
    }

    // collect the list of subfolders
    DEBUGclass("looping over subfolders...");
    while (itr.hasNext()) {
      TQFolder* subFolder = itr.readNext();
      if(!subFolder) continue;
      DEBUGclass("detaching '%s'",subFolder->getPath().Data());
      subFolder->autoSetExportName();
      const TString exportName(subFolder->getExportName());
      subFolder->detachFromBase();
      DEBUGclass("writing '%s' (basefolder is now %p)",subFolder->GetName(),(void*)subFolder->getBase());
      this->fMyDir->WriteTObject(subFolder, exportName, "Overwrite");
      DEBUGclass("adding link for '%s' at '%s'",subFolder->GetName(),exportName.Data());
      TQImportLink * importLink = new TQImportLink(subFolder->GetName(), TString(":") + exportName);
      this->addObject(importLink);
      if(failed) break;
    }
    return !failed;
  }
}


//__________________________________________________________________________________|___________

bool TQFolder::writeFolderInternal(TDirectory * dir, const TString& exportName, int depth, bool keepInMemory){
  // this private function implements the TQFolder::writeFolder functionality
  // other flavours of this function merely use the functionality of this one
  // however, for documentation, please refer to the public variants
  DEBUGclass("entering function");
  this->resolveImportLinks(true);
  this->setDirectoryInternal(dir);

  DEBUGclass("considering exportName '%s'",exportName.Data()) ;
  if(exportName.IsNull())
    this->fExportName = TQFolder::makeExportName(this->getPath());
  else
    this->fExportName = exportName;
  if(this->fExportName.IsNull())
    this->autoSetExportName();

  DEBUGclass("exportName is now '%s'",this->fExportName.Data()) ;

  bool failed = false;
  if(depth >= 0){
    DEBUGclass("splitting at level %d (target is %p)",depth,(void*)(dir)) ;
    if(!this->split(dir,depth)){
      ERRORclass("unable to split folder!");
      failed = true;
    } else {
      DEBUGclass("split successful") ;
    }
  }

  // write this object
  DEBUGclass("writing object to directory '%s'@%p",this->fMyDir->GetName(),(void*)(this->fMyDir)) ;
  #ifdef _DEBUG_
  this->printInternal("rdt",0,false);
  #endif
  this->fMyDir->WriteTObject(this, this->fExportName, "Overwrite");

  if(keepInMemory){
    DEBUGclass("resolving split") ;
    this->resolveImportLinks(true);
  }
  DEBUGclass("all done") ;
  this->isFullyResolved = keepInMemory;
  return !failed;
}

//__________________________________________________________________________________|___________

bool TQFolder::isOnDisk(){
  // check if the TQFolder has a valid directory set
  // and is available to be re-read from this directory
  return this->isOnDisk(this->fMyDir);
}

//__________________________________________________________________________________|___________

bool TQFolder::isOnDisk(TDirectory* dir){
  // check if the TQFolder is available from the given directory
  if(!dir) dir=this->fMyDir;
  if(!dir) return false;
  TList* keys = dir->GetListOfKeys();
  if(!keys) return false;
  TQIterator itr(keys);
  while(itr.hasNext()){
    TObject* key = itr.readNext();
    if( this->fExportName.CompareTo(key->GetName()) == 0){
      return true;
    }
  }
  return false;
}

//__________________________________________________________________________________|___________


bool TQFolder::collapse(){
  // collapse the folder down to its TQImportLinks
  // this is useful in cases where you read a large TQFolder structure
  // only accessing a tiny proportion of the entire structure at a time
  // and only reading (not writing) any data
  // in this case, occasional calles of TQFolder::collapse()
  // will free the memory space used by the TQFolder branch in question
  // and resinsert a TQImportLink at the corresponding points
  // to allow re-reading the folder contents if need be
  // ideally, this will drastically reduce the memory usage of your applicaton
  // however, any modification made to the folder structure will be lost!
  bool retval = false;
  TQIterator itr(this->GetListOfFolders());
  while (itr.hasNext()) {
    TQFolder* f = dynamic_cast<TQFolder*>(itr.readNext());
    if(!f) continue;
    if(!f->isOnDisk()){
      if(f->collapse()) retval = true;
      continue;
    } else {
      TQImportLink * importLink = new TQImportLink(f->GetName(), TString(":") + f->getExportName());
      f->deleteAll();
      this->deleteObject(f->getName());
      this->addObject(importLink);
      retval = true;
    }
  }
  return retval;
}


//__________________________________________________________________________________|___________



bool TQFolder::writeUpdate(int depth, bool keepInMemory){
  // this function, just like TQFolder::writeFolder, will write the folder contents to disk
  // as opposed to the aforementioned function, this will only write an incremental update
  // of the folder, that is, only write those chunks of data that have been accessed since
  // the last write has been performed
  //
  // to determine this, writeUpdate will check for expanded branches of the folder hierarchy
  // and will save them into an external object in the current directory
  // this directory will be either of the following (with descending priority)
  // - the directory to which the folder has been written at the last call of writeFolder
  // - the directory from which the folder has been retrieved originally, if writeable
  // - the current working directory
  //
  // please note that the inremential nature of the updates is only achieved if
  // - previous calls of writeUpdate and/or writeFolder have used the keepInMemory=false flag, and
  // - the depth argument for previous writeFolder and/or writeUpdate calls has been >0
  // since this is the only case in which there will be collapsed branches in the folder hierarchy
  // alternatively, incremental updates will also work if the folder is opened in lazy mode, see
  // TQFolder::loadLazyFolder
  //
  if(!this->fMyDir){
    ERRORclass("cannot write update without active directory!");
    return false;
  }
  if(!this->fMyDir->IsWritable()){
    ERRORclass("cannot write update: active directory '%s' is not writeable!",this->fMyDir->GetName());
    return false;
  }

  // collect the list of subfolders
  std::vector<TQFolder*> subfolders;
  if (depth > 0) {
    TQIterator itr(this->GetListOfFolders());
    while (itr.hasNext()) {
      TQFolder* f = dynamic_cast<TQFolder*>(itr.readNext());
      if(!f) continue;
      subfolders.push_back(f);
    }
  }

  bool failed = false;
  // iterate over subfolders and replace them by import links
  for(size_t i=0; i<subfolders.size(); i++ ){
    TQFolder * subFolder = subfolders[i];
    if (!subFolder->writeUpdate(depth - 1, keepInMemory)){
      failed = true;
    }
    subFolder->detachFromBase();
    TQImportLink * importLink = new TQImportLink(subFolder->GetName(), TString(":") + subFolder->getExportName());
    this->addObject(importLink);
  }
  // write this object
  TQFolder* oldBase = this->getBase();
  this->fBase = NULL;
  this->fMyDir->WriteTObject(this, this->fExportName, "Overwrite");
  this->fBase = oldBase;

  // delete import links again and restore original subfolders
  for(size_t i=0; i<subfolders.size(); i++){
    TQFolder * subFolder = subfolders[i];
    if(keepInMemory){
      this->deleteObject(subFolder->GetName());
      this->addObject(subFolder);
    } else {
      delete subFolder;
    }
  }
  this->isFullyResolved = keepInMemory;
  return !failed;
}




//__________________________________________________________________________________|___________

bool TQFolder::writeDirectory(TDirectory * baseDir) {
  // write this TQFolder >>as a TDirectory<< to the given base directory this
  // function is intended to allow easy export of TQFolders to co-workers which
  // are not using this framework

  this->resolveImportLinks(true);
  // stop if base directory is invalid
  if (!baseDir) {
    return false;
  }

  // check if the directory to create already exists
  TObject * dirObj = baseDir->Get(GetName());
  if (dirObj) {
    // there is another object (TDirecory?) with the same name
    return false;
  }

  // create a new directory
  TDirectory * dir = baseDir->mkdir(GetName(), GetTitle());
  if (!dir) {
    // creating the directory failed for some reason
    return false;
  }

  /* now loop on the contents of this folder
   * and write everything to the new directory */
  bool success = true;
  TQIterator itr(GetListOfFolders());
  while (itr.hasNext()) {
    TObject * obj = itr.readNext();
    if(obj->GetName()[0]=='.') continue;
    if (obj->InheritsFrom(TQFolder::Class())) {
      if (!((TQFolder*)obj)->writeDirectory(dir)) {
        // writing sub folder failed for some reason
        success = false;
      }
    } else {
      dir->WriteTObject(obj);
    }
  }

  return success;
}


//__________________________________________________________________________________|___________

bool TQFolder::importFromTextFiles(const TString& filePattern) {
  // applies the textual definitions from one or more text files to this instance
  // for details, please refer to TQFolder::loadFromTextFile
  TString errMsg;
  bool success = importFromTextFiles(filePattern, errMsg);
  if (!success) {
    ERRORclass(errMsg);
  }
  return success;
}


//__________________________________________________________________________________|___________

bool TQFolder::importFromTextFiles(const TString& filePattern, TString &errorMessage) {
  // applies the textual definitions from one or more text files to this instance
  // for details, please refer to TQFolder::loadFromTextFile
  // error messages will be fed into the string given as second argument
  TList * files = TQUtils::getListOfFilesMatching(filePattern);

  TQIterator itr(files, true);
  while (itr.hasNext()) {
    TString filename = itr.readNext()->GetName();
    TString folderName = TQFolder::makeValidPath(filename, "_", false, false);
    TQFolder * folder = this->getFolder(folderName + "+");
    if (!folder) {
      errorMessage = TString::Format("Failed to create subfolder '%s'",
                                     folderName.Data());
      return false;
    }
    if (!folder->importFromTextFile(filename, errorMessage)) {
      return false;
    }
  }

  if (itr.getCounter() == 0) {
    errorMessage = TString::Format("Cound not find any file matching '%s'",
                                   filePattern.Data());
    return false;
  }

  return true;
}


//__________________________________________________________________________________|___________

bool TQFolder::importFromTextFile(const TString& filename) {
  // applies the textual definitions from exaclty one text file to this instance
  // for details, please refer to TQFolder::loadFromTextFile

  TString errMsg;
  bool success = importFromTextFile(filename, errMsg);
  if (!success) {
    ERRORclass(errMsg);
  }
  return success;
}


//__________________________________________________________________________________|___________

bool TQFolder::importFromTextFile(const TString& filename, TString &errorMessage) {
  // applies the textual definitions from exaclty one text file to this instance
  // for details, please refer to TQFolder::loadFromTextFile
  // error messages will be fed into the string given as second argument
  int nNewlines = 1;
  TString errFile;
  TString errMsg;

  // now import from text file
  bool success = importFromTextFilePrivate(filename, nNewlines, errFile, errMsg);

  // display error message in case an error occured
  if (!success) {
    if (nNewlines < 0) {
      errorMessage = TString::Format("Error related to file '%s': %s",
                                     errFile.Data(), errMsg.Data());
    } else {
      errorMessage = TString::Format("Error in line %d of file '%s': %s",
                                     nNewlines, errFile.Data(), errMsg.Data());
    }
  }

  // return true if no error occured and false otherwise
  return success;
}


//__________________________________________________________________________________|___________

bool TQFolder::importFromTextFilePrivate(const TString& filename_, int &nNewlines, TString &errFile, TString &errMsg) {

  /* in case of an error it occured in this
   * file unless another file has been included */
  TString filename = TQLibrary::getAbsolutePath(filename_);

  errFile = filename;

  // open input file
  std::ifstream file(filename.Data());
  if (file.fail()) {
    nNewlines = -1;
    errMsg = "Failed to open file";
    return false;
  }

  TString text = TQStringUtils::readTextFromFile(&file,"#","#*","*#");
  // close file
  file.close();
  text.ReplaceAll("\r","");
  return importFromTextPrivate(text, nNewlines, errFile, errMsg);
}


//__________________________________________________________________________________|___________

TList * TQFolder::exportToText(bool includeUntextables, int indent) {
  // export this instance of TQFolder to a list of strings
  // the result can be used to create a new instance of TQFolder via
  // TQFolder::loadFolderFromText or to patch an existing folder by calling
  // TQFolder::importFromText
  this->resolveImportLinks(true);

  // the text to return
  TList * text = 0;

  // indentation prefix
  TString indentStr = TQStringUtils::repeatTabs(indent);

  // export tags
  if (this->getNTags() > 0) {
    text = new TList();
    text->SetOwner(true);
    text->AddLast(new TObjString(TString::Format("%s<%s>",
                                                 indentStr.Data(), this->exportTagsAsString().Data()).Data()));
  }

  // export objects: loop over objects in folder
  TQIterator itr(GetListOfFolders());
  while (itr.hasNext()) {
    TObject * obj = itr.readNext();
    if (!text) {
      text = new TList();
      text->SetOwner(true);
    }
    if (obj->InheritsFrom(TQFolder::Class())) {
      TList * subText = ((TQFolder*)obj)->exportToText(includeUntextables, indent + 1);
      if (subText) {
        text->AddLast(new TObjString(TString::Format("%s+%s {",
                                                     indentStr.Data(), obj->GetName()).Data()));
        text->AddAll(subText);
        text->AddLast(new TObjString(TString::Format("%s}",
                                                     indentStr.Data()).Data()));

        subText->SetOwner(false);
        delete subText;
      } else {
        text->AddLast(new TObjString(TString::Format("%s+%s;",
                                                     indentStr.Data(), obj->GetName()).Data()));
      }
    } else if (obj->InheritsFrom(TObjString::Class())){
      text->AddLast(new TObjString(TString::Format("%s\"%s\";",indentStr.Data(),obj->GetName())));
    } else if(includeUntextables) {
      TString details = TQStringUtils::getDetails(obj);
      if (!details.IsNull()) {
        details.Prepend(" {");
        details.Append("}");
      }
      if(obj->InheritsFrom(TH1::Class())){
        text->AddLast(new TObjString(indentStr+TQHistogramUtils::convertToText((TH1*)obj,2)+";"));
      } else {
        text->AddLast(new TObjString(TString::Format("%s#+%s::%s%s;", indentStr.Data(),
                                                     obj->IsA()->GetName(), obj->GetName(), details.Data()).Data()));
      }
    }
  }

  return text;
}


//__________________________________________________________________________________|___________

bool TQFolder::writeContentsToHTML(std::ostream& out, int expandDepth, bool includeUntextables) {
  // creat an HTML-view of this TQFolder instance
  // please be aware that the result can be unmanagably large for large folder hierarchies
  out << "<div class=\"listing\" style=\"display:" << (expandDepth > 0 ? "block" : "none") << "\">" << std::endl;

  // export tags
  TQIterator itrTags(this->getListOfKeys(),true);
  while(itrTags.hasNext()){
    out << "<div class=\"tag\">";
    TObject* obj = itrTags.readNext();
    TString name(obj->GetName());
    TString value(this->getTagStringDefault(name,""));
    out << "<span class=\"tagKey\">" << TQStringUtils::convertPlain2HTML(name) << "</span>";
    out << "<span class=\"tagValue\">" << TQStringUtils::convertPlain2HTML(value) << "</span>";
    out << "</div>" << std::endl;
  }
  // export objects: loop over objects in folder
  TQIterator itr(GetListOfFolders());
  while (itr.hasNext()) {
    TObject * obj = itr.readNext();
    TQFolder* f = dynamic_cast<TQFolder*>(obj);
    if (f){
      out << "<div>" << std::endl;
      out << "<div class=\"folder\" onclick=\"toggleDiv(this.nextElementSibling)\"><span class=\"foldername\">" << f->GetName() << "</span></div>" << std::endl;
      f->writeContentsToHTML(out,expandDepth-1,includeUntextables);
      out << "</div>" << std::endl;
    } else if (includeUntextables) {
      TString details;
      if (obj->InheritsFrom(TH1::Class())) {
        details = TQHistogramUtils::getDetailsAsString((TH1*)obj, 2);
      } else if (obj->InheritsFrom(TQCounter::Class())) {
        details = ((TQCounter*)obj)->getAsString();
      }
      TString className(obj->IsA()->GetName());
      out << "<div class=\"object\"><span class=\"objecttype\"><a style=\"color:inherit; text-decoration:none\" target=\"_blank\" href=\"" << TQFolder::concatPaths(TQLibrary::getWebsite(),className) << ".html\">" << className << "</a></span><span class=\"objectname\">" << obj->GetName() << "</span>";
      if(!details.IsNull()) out << "<span class=objectdetails\">" << TQStringUtils::convertPlain2HTML(details) << "</span>";
      out << "</div>";
    }
  }

  out << "</div>" << std::endl;
  return true;
}

//__________________________________________________________________________________|___________

bool TQFolder::exportToHTMLFile(const TString& filename, int expandDepth, bool includeUntextables) {
  // creat an HTML-view of this TQFolder instance and write it to a file
  // please be aware that the result can be unmanagably large for large folder hierarchies
  if(expandDepth < 1) expandDepth = std::numeric_limits<int>::infinity();
  std::ofstream out(filename);
  if(!out.is_open()) return false;

  out << "<html>" << std::endl << "<head>" << std::endl;
  out << "<meta http-equiv=\"Content-Type\" content=\"text/html; charset=utf-8\">" << std::endl;
  out << "<title>" << this->GetName() << "</title>" << std::endl;
  out << "<style id=\"style\" type=\"text/css\">" << std::endl;
  out << ".folder { width:100%; background-image:url(" << '"' << TQStringUtils::readSVGtoDataURI(TQFolder::concatPaths(TQLibrary::getTQPATH(),"share/nuoveXT-icons/folder.svg")) << '"' << "); background-size:contain; background-repeat: no-repeat; cursor: hand; }" << std::endl;
  out << ".foldername { margin-left:1.5em; font-weight:bold; }" << std::endl;
  out << ".listing { margin-left:3em; }" << std::endl;
  out << ".tag { display:inline-block; font-size:12pt; font-style:italic; width:100%; }" << std::endl;
  out << ".tagkey { display:inline-block; min-width: 200px; color:purple; margin-right:1em; }" << std::endl;
  out << ".tagvalue { display: inline-block; width: 80%; vertical-align: top; };" << std::endl;
  out << ".object { display:inline-block; }" << std::endl;
  out << ".objecttype { display:inline-block; width: 100px; color: darkblue; font-weight:bold; text-decoration:none; }" << std::endl;
  out << ".objectname { display:inline-block; width: 200px; }" << std::endl;
  out << ".objectdetails { };" << std::endl;
  out << "a { font-weight:bold; text-decoration:none; }" << std::endl;
  out << "</style>" << std::endl;

  out << "<script type=\"text/javascript\">" << std::endl;
  out << "function toggleDiv(obj){ if(!obj) return; if(obj.style.display==\"none\") obj.style.display=\"block\"; else obj.style.display=\"none\"; }" << std::endl;
  out << "</script>" << std::endl;
  out << "</head>" << std::endl;
  out << "<body>" << std::endl;
  this->writeContentsToHTML(out,expandDepth,includeUntextables);
  out << "<hr>" << std::endl;
  out << "<div style=\"font-size:12px\">This page was automatically generated by the <a href=\"" << TQLibrary::getWebsite() << "\">HWWAnalysisCode</a> software library. The icons displayed are part of the <a href=\"http://nuovext.pwsp.net/\">nuoveXT</a> icon scheme, licensed under <a href=\"http://www.gnu.org/licenses/lgpl.html\">LGPL</a> (2013).</div>" << std::endl;
  out << "</body>" << std::endl;
  out << "</html>" << std::endl;
  return true;
}

//__________________________________________________________________________________|___________

bool TQFolder::exportToTextFile(const TString& filename, bool includeUntextables) {
  // export this instance of TQFolder to a text file
  // the result can be used to create a new instance of TQFolder via
  // TQFolder::loadFolderFromTextFile or to patch an existing folder by calling
  // TQFolder::importFromTextFile
  TList * text = this->exportToText(includeUntextables);

  if (text) {
    text->AddFirst(new TObjString("# -*- mode: tqfolder -*-"));
    bool success = TQStringUtils::writeTextToFile(text, filename);
    delete text;
    return success;
  } else {
    return false;
  }
}


//__________________________________________________________________________________|___________

bool TQFolder::importFromText(const TString& input) {
  // Import folders and tags from text. The syntax is built from three basic elements:
  //
  // 1) creating new folders (instances of TQFolder),
  // 2) setting tags on folders,
  // 3) executing commands.
  //
  // Lines starting with "#" will be considered a comment and will be ignored. In
  // the following the syntax is explain my means of an example text:
  //
  // # Create a new sub-folder ("mySubfolder"):
  // +mySubfolder;
  //
  // # Create a new sub-folder ("mySubfolder") and import recursively:
  // +mySubfolder {
  //
  // }
  //
  // # Please note: If a recursive import block is opened "{...}" no
  // # terminating ";" is needed.
  //
  // # Create new nested sub-folders ("mySubfolder/anotherSubFolder"):
  // +mySubfolder/anotherSubFolder;
  //
  // # This one does also work with recursive import:
  // +mySubfolder/anotherSubFolder {
  //
  // }
  //
  // # Set a tag ("myTag") with some value (myValue) on this folder:
  // <myTag = myValue>
  //
  // # Set more than one tag:
  // <myTag = myValue, anotherTag = false, hello = "world">
  //
  // # Set a tag ("myTag") with some value (myValue) on a sub-folder ("mySubfolder"):
  // <myTag = myValue> @ mySubfolder;
  //
  // # Include text to import from an external text file ("myImportText.txt"):
  // $include("myImportText.txt");
  //
  // # Delete a folder ("mySubfolder"):
  // $delete("mySubfolder");
  //
  // # (this one simply calls TQFolder::deleteObject(...)).
  //
  // # Ignore text inside a block:
  // $ignore() {
  //
  // }
  //
  // # Please note: the text to be ignored should be a valid text block to import.
  //
  // # Copy a folder ("mySubfolder") and rename the new instance (to "mySubfolder2")
  // $copy("mySubfolder >>:: mySubfolder2");

  TString errMsg;
  bool success = importFromText(input, errMsg);
  if (!success) {
    ERRORclass(errMsg);
  }
  return success;
}


//__________________________________________________________________________________|___________

bool TQFolder::importFromText(const TString& input, TString &errorMessage) {
  // Import folders and tags from text.
  // for details, please refer to the wrapper function of the same name
  // occurring errors will be fed into the string given as second argument
  int nNewlines = 1;
  TString errFile;
  TString errMsg;
  bool success = importFromTextPrivate(input, nNewlines, errFile, errMsg);
  if (!success) {
    if (errFile.IsNull()) {
      errorMessage = TString::Format("Error in line %d: %s", nNewlines, errMsg.Data());
    } else {
      errorMessage = TString::Format("Error in line %d of file '%s': %s",
                                     nNewlines, errFile.Data(), errMsg.Data());
    }
  }
  return success;
}

bool TQFolder::executeCopyCommand(TString object, TString& errMsg, bool moveOnly, const TString& destPrefix){
  // read object source
  TString source;
  object.ReplaceAll("$(BASEFOLDERNAME)",this->GetName());
  TQStringUtils::removeLeadingBlanks(object);
  if (!TQStringUtils::readToken(object, source, TQStringUtils::getDefaultIDCharacters() + "*?/")) {
    errMsg = "Expecting object source for command 'copy'";
    return false;
  }
  TQStringUtils::removeLeadingBlanks(object);
  TString dest;
  if (!object.IsNull()) {
    if (TQStringUtils::removeLeading(object, ">") != 2) {
      errMsg = "Expecting destination operator '>>' after source for command 'copy', but found '" + object + "'";
      return false;
    }
    TQStringUtils::removeLeadingBlanks(object);
    if (object.IsNull()) {
      errMsg = "Expecting destination after operator '>>' for command 'copy', but received no input";
      return false;
    }
    dest = TQFolder::concatPaths(destPrefix,this->replaceInText(object,"~"));
  }
  DEBUGclass("evaluating $copy/$move operator in '%s'",this->getPath().Data());
  // get source of copy
  TQIterator objitr(this->getListOfObjectPaths(source),true);
  int nFound = 0;
  while(objitr.hasNext()){
    TObject * objSourcePath = objitr.readNext();
    if (!objSourcePath) continue;
    TString srcPath(objSourcePath->GetName());
    TString srcName(TQFolder::getPathTail(srcPath));
    TObject* objSource = this->getObject(srcName,srcPath);
    if(!objSource) continue;
    nFound++;

    if(!moveOnly){
      // make copy
      if (objSource->InheritsFrom(TQFolder::Class())) {
        objSource = ((TQFolder*)objSource)->copy();
      } else {
        objSource = objSource->Clone();
      }
      // add copy
      if (!addObject(objSource, dest)) {
        delete objSource;
        errMsg = TString::Format("Failed to copy object '%s'", source.Data());
        return false;
      }
    } else {
      // make copy
      if (objSource->InheritsFrom(TQFolder::Class())) {
        objSource = ((TQFolder*)objSource)->detachFromBase();
      } else {
        this->getFolder(TQFolder::getPathTail(srcPath))->Remove(objSource);
      }
      // add copy
      if (!addObject(objSource, dest)) {
        errMsg = TString::Format("Failed to copy object '%s'", source.Data());
        return false;
      }
    }
  }
  if(nFound < 1){
    errMsg = TString::Format("Couldn't find object matching '%s' in '%s'", source.Data(), this->getPath().Data());
    return false;
  }
  return true;
}

//__________________________________________________________________________________|___________

bool TQFolder::importFromTextPrivate(TString input, int &nNewlines, TString &errFile, TString &errMsg) {
  // worker function to import a TQFolder from a text file

  // read input string
  while (!input.IsNull()) {

    // read leading blanks and line breaks
    TQStringUtils::readBlanksAndNewlines(input, nNewlines);

    // switch according to token
    TString token;
    if (input.BeginsWith("#")) {
      // ==> comment: ignore line
      TQStringUtils::readUpTo(input, token, "\n");
    } else if (input.BeginsWith("@")) {
      // read "@"
      int np = TQStringUtils::removeLeading(input, "@");
      if (np > 1) {
        errMsg = TString::Format("Wrong operator '%s'", TQStringUtils::repeat("@", np).Data());
        return false;
      }

      // read name of object
      int nNewlinesTmp = nNewlines;
      TQStringUtils::readBlanksAndNewlines(input, nNewlines);
      if (TQStringUtils::readToken(input, token,
                                   TQStringUtils::getDefaultIDCharacters() + ",/:*?$()") == 0) {
        nNewlines = nNewlinesTmp;
        errMsg = "Expect object definition after '@'";
        return false;
      }

      // create new folder
      TString path = this->replaceInText(token,"~",false);
      TCollection* c = this->getListOfFolders(path);
      nNewlinesTmp = nNewlines;
      TQStringUtils::readBlanksAndNewlines(input, nNewlinesTmp);
      if (input.BeginsWith("{")) {
        token.Clear();
        if (TQStringUtils::readBlock(input, token, "{}[]()", "''\"\"#\n", false, 2) == 0) {
          errMsg = TString::Format("Block opened here '%s' not closed properly",TQStringUtils::maxLength(TQStringUtils::compactify(input), 40).Data());
          return false;
        }
      }
      if(c && c->GetEntries() > 0){
        TQFolderIterator itr(c,true);
        while(itr.hasNext()){
          TQFolder* f = itr.readNext();
          if(!f) continue;
          //if (!
          f->importFromTextPrivate(token, nNewlines, errFile, errMsg);
          //) {
          //  return false;
          //}
        }
        nNewlines = nNewlinesTmp;
      } else {
        errMsg = TString::Format("[WARNING] Operation '@%s' produced no matches!",path.Data());
        //return false;
      }
    } else if (input.BeginsWith("+")) {
      // ==> new object

      // read "+"
      int np = TQStringUtils::removeLeading(input, "+");
      if (np > 1) {
        errMsg = TString::Format("Wrong operator '%s'", TQStringUtils::repeat("+", np).Data());
        return false;
      }

      // read name of object
      int nNewlinesTmp = nNewlines;
      TQStringUtils::readBlanksAndNewlines(input, nNewlines);
      if (TQStringUtils::readToken(input, token,
                                   TQStringUtils::getDefaultIDCharacters() + "/:*?$()") == 0) {
        nNewlines = nNewlinesTmp;
        errMsg = "Expect object definition after '+'";
        return false;
      }

      // create new folder
      TString newname = this->replaceInText(token,"~",false);
      newname.ReplaceAll("$(BASEFOLDERNAME)",this->GetName());
      TQFolder * newObj = this->getFolder(newname + "+");
      if (!newObj) {
        errMsg = TString::Format("Failed to create object '%s'", token.Data());
        return false;
      }

      // nested definition?
      nNewlinesTmp = nNewlines;
      TQStringUtils::readBlanksAndNewlines(input, nNewlines);
      if (input.BeginsWith("{")) {
        token.Clear();
        if (TQStringUtils::readBlock(input, token, "{}[]()", "''\"\"#\n", false, 2) == 0) {
          errMsg = TString::Format("Block opened here '%s' not closed properly",
                                   TQStringUtils::maxLength(TQStringUtils::compactify(input), 40).Data());
          return false;
        }
        if (!newObj->importFromTextPrivate(token, nNewlines, errFile, errMsg)) {
          return false;
        }
      } else if (!TQStringUtils::removeLeading(input, ";", 1)) {
        nNewlines = nNewlinesTmp;
        errMsg = TString::Format("Missing terminating ';' after object definition '%s'", token.Data());
        return false;
      }

    } else if (input.BeginsWith("<")) {
      // ==> set tags

      // read tags
      TString tags;
      if (TQStringUtils::readBlock(input, tags, "<>[](){}", "''\"\"#\n", false, 2) == 0) {
        errMsg = TString::Format("Tag definition block opened here '%s' not closed properly",
                                 TQStringUtils::maxLength(input, 10).ReplaceAll("\n", " ").Data());
        return false;
      }

      // read (optional) destination
      TString dest;
      TQStringUtils::readBlanksAndNewlines(input, nNewlines);
      if (TQStringUtils::removeLeading(input, "@", 1) > 0) {
        // read destination
        int nNewlinesTmp = nNewlines;
        TQStringUtils::readBlanksAndNewlines(input, nNewlines);
        if (TQStringUtils::readToken(input, dest,
                                     TQStringUtils::getDefaultIDCharacters() + "/*?, ") == 0) {
          nNewlines = nNewlinesTmp;
          errMsg = "Expecting tag destination path after '@'";
          return false;
        }
        // read terminating ";"
        nNewlinesTmp = nNewlines;
        TQStringUtils::readBlanksAndNewlines(input, nNewlines);
        if (!TQStringUtils::removeLeading(input, ";", 1)) {
          nNewlines = nNewlinesTmp;
          errMsg = "Missing terminating ';' after tag destination path";
          return false;
        }
      }

      // import tags
      tags.ReplaceAll("$(BASEFOLDERNAME)",this->GetName());
      TQTaggable tagReader(tags);
      tagReader.exportTags(this, dest);
    } else if (input.BeginsWith("$")) {
      // ==> command
      // remove leading "$"
      TQStringUtils::removeLeading(input, "$", 1);

      // read command
      TString cmd;
      if (!TQStringUtils::readToken(input, cmd, TQStringUtils::getLetters())) {
        // ==> missing command: stop
        errMsg = "Missing command after '$'";
        return false;
      }

      // read parameter
      TString strParameter;

      int nNewlinesTmp = nNewlines;
      TQStringUtils::readBlanksAndNewlines(input, nNewlines);
      if (input.BeginsWith("(")) {
        if (TQStringUtils::readBlock(input, strParameter, "()", "''\"\"#\n", false, 2) == 0) {
          errMsg = TString::Format("Parameter block opened here '%s' not closed properly",
                                   TQStringUtils::maxLength(input, 10).ReplaceAll("\n", " ").Data());
          return false;
        }
      } else {
        // ==> missing parameter: stop
        nNewlines = nNewlinesTmp;
        errMsg = TString::Format("Missing '(...)' after command '%s'", cmd.Data());
        return false;
      }

      bool isCmdInclude = (cmd.CompareTo("include", TString::kIgnoreCase) == 0);
      bool isCmdImport = (cmd.CompareTo("import", TString::kIgnoreCase) == 0);
      bool isCmdCopy = (cmd.CompareTo("copy", TString::kIgnoreCase) == 0);
      bool isCmdMove = (cmd.CompareTo("move", TString::kIgnoreCase) == 0);
      bool isCmdDelete = (cmd.CompareTo("delete", TString::kIgnoreCase) == 0);
      bool isCmdEscape = (cmd.CompareTo("escape", TString::kIgnoreCase) == 0);
      bool isCmdIgnore = (cmd.CompareTo("ignore", TString::kIgnoreCase) == 0);
      bool isCmdModify = (cmd.CompareTo("modify", TString::kIgnoreCase) == 0);
      bool isCmdCreate = (cmd.CompareTo("create", TString::kIgnoreCase) == 0);
      bool isCmdFor = (cmd.CompareTo("for", TString::kIgnoreCase) == 0);
      bool isCmdReplace = (cmd.CompareTo("replace", TString::kIgnoreCase) == 0);
      bool isCmdPrint = (cmd.CompareTo("print", TString::kIgnoreCase) == 0); //execute 'print' command with given argument on curent folder location
      bool isCmdPrintLine (cmd.CompareTo("printline", TString::kIgnoreCase) == 0); //print argument to console
      bool isCmdWrite = (cmd.CompareTo("write", TString::kIgnoreCase) == 0); //write folder to disk
      bool acceptNestedBlock = isCmdCopy;
      bool expectNestedBlock = isCmdIgnore || isCmdFor;

      // read terminating ";"
      nNewlinesTmp = nNewlines;
      TQStringUtils::readBlanksAndNewlines(input, nNewlines);
      TString nestedBlock;
      if (input.BeginsWith("{") && (acceptNestedBlock || expectNestedBlock)) {
        if (TQStringUtils::readBlock(input, nestedBlock, "{}[]()", "''\"\"#\n", false, 2) == 0) {
          errMsg = TString::Format("Block opened here '%s' not closed properly",
                                   TQStringUtils::maxLength(input, 10).ReplaceAll("\n", " ").Data());
          return false;
        }
      } else if (!input.BeginsWith("{") && expectNestedBlock) {
        // ==> missing terminating ";": stop
        nNewlines = nNewlinesTmp;
        errMsg = TString::Format("Expecting nested block after command '%s'", cmd.Data());
        return false;
      } else if (!TQStringUtils::removeLeading(input, ";", 1)) {
        // ==> missing terminating ";": stop
        nNewlines = nNewlinesTmp;
        errMsg = TString::Format("Missing terminating ';' after command '%s'", cmd.Data());
        return false;
      }

      strParameter.ReplaceAll("$(BASEFOLDERNAME)",this->GetName());

      if (isCmdInclude) {
        // read filename
        TQTaggable param;
        param.importTagWithPrefix(strParameter, "", false, "filename");
        TString includeFilename;

        int nNewlines2 = 1;
        TString errFile2;
        TString errMsg2;
        //@tag: [filename] This parameter tag states the file to be included by $include(filename=myfile) when importing a folder structure from text. Specifying the name of this tag ("filename=") is optional.
        if (param.getNTags() != 1 || !param.getTagString("filename", includeFilename)) {
          // ==> error reading file: stop
          errMsg = "Wrong number of arguments or invalid argument to command 'include', filename needed.";
          return false;

        } else if (!this->importFromTextFilePrivate(includeFilename, nNewlines2, errFile2, errMsg2)) {
          // ==> error reading file: stop
          if (nNewlines2 < 0) {
            errMsg = TString::Format("Failed to include file '%s': %s", includeFilename.Data(), errMsg2.Data());
          } else {
            nNewlines = nNewlines2;
            errFile = errFile2;
            errMsg = errMsg2;
          }
          return false;
        }
      } else if (isCmdImport) {
        // import a ROOT file
        TQStringUtils::unquoteInPlace(strParameter,"\"'");
        if (!this->importObject(strParameter)){
          errMsg = TString::Format("Failed to import '%s'", strParameter.Data());
          return false;
        }
      } else if (isCmdDelete) {
        // read object
        TQTaggable param;
        //@tag: [object] This parameter tag states the object to be deleted/copied by $delete(object=someObject) or $copy("path/to/oldObj >> new/Path/::newName"); when importing a folder structure from text. Specifying the name of this tag ("object=") is optional.
        param.importTagWithPrefix(strParameter, "", false, "object");
        TString object;

        if (param.getNTags() < 1){
          // ==> error copying: stop
          errMsg = "Wrong number of arguments to command 'delete'";
          return false;
        } else if(param.getTagString("object", object)){
          object = TQStringUtils::trim(object);
          if (!this->deleteObject(object)) {
            // ==> error deleting object: stop
            errMsg = TString::Format("[mostly harmless] in %s: Failed to delete object '%s'", this->getPath().Data(),object.Data());
            //return false;
          }
        } else if(param.getTagString("tags", object)){
          TString path;
          TQTaggable* obj=NULL;
          if(param.getTagString("path",path)){
            obj = dynamic_cast<TQTaggable*>(this->getObject(path));
          } else {
            obj = this;
          }
          obj->removeTags(object);
        }
      } else if (isCmdCopy || isCmdMove) {
        // read object
        TQTaggable param;
        param.importTagWithPrefix(strParameter, "", false, "object");
        TString object;

        if (param.getNTags() != 1 || !param.getTagString("object", object)) {
          // ==> error copying: stop
          errMsg = "Wrong number of arguments to command 'copy'";
          return false;

        } else {
          bool ok = this->executeCopyCommand(object,errMsg,isCmdMove);
          if(!ok) return false;
        }
      } else if (isCmdIgnore) {
        // ==> simply don't do anything
      } else if (isCmdFor) {
        /*@tag:[step,pad] edit the behavior of the for-loop, enabling skipping a few entries (step) or padding the string of the key to more or less than the number of digits of the final value*/
        TString key,strItBegin,strItEnd;
        if(TQStringUtils::readUpTo(strParameter,key,",")<1 || TQStringUtils::removeLeading(strParameter,",") !=1 ||
           TQStringUtils::readUpTo(strParameter,strItBegin,",")<1 || TQStringUtils::removeLeading(strParameter,",") !=1 ||
           TQStringUtils::readUpTo(strParameter,strItEnd,",")<1){
          errMsg = "for-command must have syntax '$for(key,begin,end){ ... }'";
          return false;
        }
        TQTaggable param(strParameter);
        int itBegin = atoi(strItBegin.Data());
        int itEnd = atoi(strItEnd.Data());
        int step = param.getTagIntegerDefault("step",itEnd >= itBegin ? 1 : -1);
        int pad = param.getTagIntegerDefault("pad",1+log10(std::max(itBegin,itEnd-1)));
        int tmpNewlines(nNewlines);
        TQTaggable helper;
        for(int it=itBegin; it<itEnd; it+=step){
          helper.setTagString(key,TQStringUtils::padNumber(it,pad));
          TString tmpBlock = helper.replaceInText(nestedBlock);
          tmpNewlines = nNewlines;
          if (!this->importFromTextPrivate(tmpBlock, tmpNewlines, errFile, errMsg)){
            return false;
          }
        }
        nNewlines = tmpNewlines;
      } else if (isCmdEscape) {
        TQStringUtils::removeLeadingBlanks(strParameter);
        bool inverted = (TQStringUtils::removeLeading(strParameter,"!") > 0); //inverted selection, trigger on everything BUT folders matching the filter if name is prefixed by "!".
        if (TQStringUtils::equal(strParameter,"") || TQStringUtils::matches(this->getName(), strParameter)) {
          if (!inverted) return true;
        } else {
          if (inverted) return true;
        }
      } else if (isCmdModify) {
        TQTaggable param;
        param.importTags(strParameter);
        #ifdef _DEBUG_
        param.printTags();
        #endif
        /*@tag:[tag,operator,value,path,filter,create,force,override] These text argument tags are used for the $modify command when importing from text. They specify which tag ("tag") at which folder (matching the value of "path", wildcarding is possible) to modify using the value of the tag "value" and the operator from "operator". "tag", "operator" and "value" are required, "path" is optional. A "filter" can be specified to only apply the operation to instances of TQSampleFolder ("sf", "samplefolder") or TQSample ("s", "sample"); please note that the filter is inclusive for instances of classes inheriting from the one specified by the filter (i.e. "sf" also includes instances of TQSample).

        Supported operators are dependent on the type of tag: string: "+" (append), "-" (remove trailing), "=" (set); int/double: "+","-","*","/","^"(exponentiate, old^new),"="(set); bool: "=","==","!=","&&","||". Additionally, operators for non-boolean values (except "=") can be prefixed with an exclamation mark "!" interchanging the existing (old) value and the new value in the corresponding operation for numerical values. For strings, an exclamation mark before the actual operator causes the operation to be performed at the beginning of the existing string. Examples:
        "/" calculates <old>/<new>, "!/" calculates <new>/<old>
        "-" removes the string <new> from the end of <old> if possible, "!-" does the same thing at the beginning of <old>.

        The boolean tag "create" specifies is a new tag should be created, if none of the given name is present so far (default: false). If "force" is set to true, existing tags of different types are overwritten (default: false). In both cases (non-existing tag or overwriting tag of different type), an initial value of 0., false or "" is used before the specified operation is performed. The boolean tag "override" allows to specify is existing tags (of same type) should be replaced. This can be used to initalize tag at all placed which do not have a value set yet. Default is !create.
        */
        if ( !(param.hasTagString("tag") && param.hasTagString("operator") && param.hasTag("value")) ) {
        errMsg = TString("Missing options! Required values are 'tag=\"tagToModify\", operator=\"+-*/=\", value=someValue'. Optional parameters are 'path=\"path/to/folder\", filter=\"s/sf\", create=true/false, override=false/true, force=true/false (later values are default)");
        }
        TString path = param.getTagStringDefault("path",".");
        TString tag = param.getTagStringDefault("tag","");
        //retrieve filter (allows to modify only TQSamples, TQSampleFolders and TQSamples or all TQFolders
        TClass* filter = TQFolder::Class();
        if (TQStringUtils::compare(param.getTagStringDefault("filter",""),"s") == 0 || TQStringUtils::compare(param.getTagStringDefault("filter",""),"sample") == 0) {
          filter = TQSample::Class();
        } else if (TQStringUtils::compare(param.getTagStringDefault("filter",""),"sf") == 0 || TQStringUtils::compare(param.getTagStringDefault("filter",""),"samplefolder") == 0) {
          filter = TQSampleFolder::Class();
        }
        TString op = param.getTagStringDefault("operator","=");
        //only create a non-existent tag if enforced
        bool create = param.getTagBoolDefault("create",false);
        bool force = param.getTagBoolDefault("force",false);
        bool replace = param.getTagBoolDefault("override",!create); //if we are not allowed to create new tags and operator is '=' we wouldn't do anything unless replace is true.
        //retrieve type of tag
        bool isString = param.tagIsOfTypeString("value");
        bool isDouble = param.tagIsOfTypeDouble("value");
        bool isInt = param.tagIsOfTypeInteger("value");
        bool isBool = param.tagIsOfTypeBool("value");
        if (!isString && !isDouble && !isInt && !isBool) {
          errMsg = TString("Cannot modify! Unsupported tag type.");
          return false;
        }
        bool inverted = false;//if this is true, the order of the arguments is inverted (i.e. the left hand argument of the operator becomes the right hand one and vice versa.
        if (!isBool) {
          inverted = TQStringUtils::removeLeading(op,"!",1) == 1;
        }
        if (inverted && TQStringUtils::equal(op,"=")) {
          errMsg = TString::Format("Are you trying to set a value in inverted mode (operator '!=' for non-boolean value)? This does not makesense!");
          return false;
        }
        bool silent = param.getTagBoolDefault("quiet",param.getTagBoolDefault("silent",false));
        if (isString && !( TQStringUtils::equal(op,"=") || TQStringUtils::equal(op,"+") || TQStringUtils::equal(op,"-") ) ) {
          errMsg = TString::Format("Unsupported modification operator '%s' for value of type %s",op.Data(),"string");
          return false;
        }
        if (isDouble && !( TQStringUtils::equal(op,"=") || TQStringUtils::equal(op,"+") || TQStringUtils::equal(op,"-") || TQStringUtils::equal(op,"*") || TQStringUtils::equal(op,"/") || TQStringUtils::equal(op,"^") ) ) {
          errMsg = TString::Format("Unsupported modification operator '%s' for value of type %s",op.Data(),"double");
          return false;
        }
        if (isInt && !( TQStringUtils::equal(op,"=") || TQStringUtils::equal(op,"+") || TQStringUtils::equal(op,"-") || TQStringUtils::equal(op,"*") || TQStringUtils::equal(op,"/") || TQStringUtils::equal(op,"^") ) ) {
          errMsg = TString::Format("Unsupported modification operator '%s' for value of type %s",op.Data(),"int");
          return false;
        }
        if (isBool && !( TQStringUtils::equal(op,"=") || TQStringUtils::equal(op,"==") || TQStringUtils::equal(op,"!=") || TQStringUtils::equal(op,"&&") || TQStringUtils::equal(op,"||") ) ) {
          errMsg = TString::Format("Unsupported modification operator '%s' for value of type %s",op.Data(),"bool");
          return false;
        }

        if ( TQStringUtils::equal(op,"/") && (!inverted && ((isDouble && param.getTagDoubleDefault("value",0.) == 0.) || (isInt && param.getTagIntegerDefault("vaue",0) == 0) ) ) ) {
          errMsg = TString("Cannot modify tags! Division by zero is not available yet. ETA: 1/0. days");
          return false;
        }

        TList* targetList = this->getListOfFolders(path,filter);
        if (!targetList) {
          errMsg = TString::Format("No matching folders found for pattern '%s'",path.Data());
          return false;
        }


        TQFolderIterator itr(targetList);
        while (itr.hasNext()) {
          TQFolder* folder = itr.readNext();
          if (!folder) continue;
          //catch division by zero (for "inverted" mode) TODO: some more checks might be required here
          if (inverted && (TQStringUtils::equal(op,"/") && (folder->tagIsOfTypeDouble(tag) || folder->tagIsOfTypeInteger(tag)) && folder->getTagDoubleDefault(tag,0.) == 0. )) {
            WARNclass(TString::Format("Cannot modify tags for folder '%s' (inverted mode)! Division by zero is not available yet. ETA: 1/0. days; skipping this folder",folder->getName().Data() ).Data());
            continue;
          }

          if (isString) {
            if (!folder->tagIsOfTypeString(tag) && !(create||force) ) {
              //don't create a new tag
              continue;
            }
            if (folder->hasTag(tag) && !folder->tagIsOfTypeString(tag)) {
              if (!force) {
                WARNclass("Tag '%s' at '%s' is already present but of different type!", tag.Data(), folder->getPath().Data());
                continue;
              } else {
                if (!silent) WARNclass("Replacing existing tag '%s' of different type at '%s'!", tag.Data(), folder->getPath().Data());
                TString val;
                if (!folder->getTagString(tag,val)) {WARNclass("Failed to convert existing tag to string type!"); continue;}
                folder->removeTag(tag);
                folder->setTagString(tag,val);
              }
            }
            if (TQStringUtils::equal(op,"=") && (replace || !folder->hasTagString(tag) ) ) {folder->setTag(tag,param.getTagStringDefault("value",""));}
            if (TQStringUtils::equal(op,"+")) {folder->setTag(tag,inverted ? param.getTagStringDefault("value","") + folder->getTagStringDefault(tag,"") : folder->getTagStringDefault(tag,"") + param.getTagStringDefault("value","")); continue;}
            if (TQStringUtils::equal(op,"-")) {
              TString tmp = folder->getTagStringDefault(tag,"");
              if (inverted) {
                TQStringUtils::removeLeadingText(tmp,param.getTagStringDefault("value",""));
              } else {
                TQStringUtils::removeTrailingText(tmp,param.getTagStringDefault("value",""));
              }
              folder->setTag(tag,tmp);
              continue;
            }
          } else if (isDouble) {
            if (!folder->tagIsOfTypeDouble(tag) && !(create||force) ) {
              //don't create a new tag
              continue;
            }
            if (folder->hasTag(tag) && !folder->tagIsOfTypeDouble(tag)) {
              if (!force) {
                WARNclass("Tag '%s' at '%s' is already present but of different type!", tag.Data(), folder->getPath().Data());
                continue;
              } else {
                if (!silent) WARNclass("Replacing existing tag '%s' of different type at '%s'!", tag.Data(), folder->getPath().Data());
                double val;
                if (!folder->getTagDouble(tag,val)) {WARNclass("Failed to convert existing tag to double type!"); continue;}
                folder->removeTag(tag);
                folder->setTagDouble(tag,val);
              }
            }
            if (TQStringUtils::equal(op,"=") && (replace || !folder->hasTagDouble(tag) ) ) {folder->setTag(tag,param.getTagDoubleDefault("value",0.)); continue;}
            //+ operator for double is abelian, so we silently ignore the "inverted" flag
            if (TQStringUtils::equal(op,"+")) {folder->setTag(tag,folder->getTagDoubleDefault(tag,0.) + param.getTagDoubleDefault("value",0.)); continue;}
            if (TQStringUtils::equal(op,"-")) {folder->setTag(tag,inverted ? param.getTagDoubleDefault("value",0.) - folder->getTagDoubleDefault(tag,0.) : folder->getTagDoubleDefault(tag,0.) - param.getTagDoubleDefault("value",0.)); continue;}
            //* operator for double is abelian, so we silently ignore the "inverted" flag
            if (TQStringUtils::equal(op,"*")) {folder->setTag(tag,folder->getTagDoubleDefault(tag,0.) * param.getTagDoubleDefault("value",0.)); continue;}
            if (TQStringUtils::equal(op,"/")) {folder->setTag(tag,inverted ? param.getTagDoubleDefault("value",0.) / folder->getTagDoubleDefault(tag,0.) : folder->getTagDoubleDefault(tag,0.) / param.getTagDoubleDefault("value",0.)); continue;} //division by zero should be caught at an earlier stage.
            if (TQStringUtils::equal(op,"^")) {folder->setTag(tag,inverted ? pow(param.getTagDoubleDefault("value",0.) , folder->getTagDoubleDefault(tag,0.) ) : pow( folder->getTagDoubleDefault(tag,0.) , param.getTagDoubleDefault("value",0.) ) ); continue;}
          }  else if (isInt) {
            if (!folder->tagIsOfTypeInteger(tag) && !(create||force) ) {
              //don't create a new tag
              continue;
            }
            if (folder->hasTag(tag) && !folder->tagIsOfTypeInteger(tag)) {
              if (!force) {
                WARNclass("Tag '%s' at '%s' is already present but of different type!", tag.Data(), folder->getPath().Data());
                continue;
              } else {
                if (!silent) WARNclass("Replacing existing tag '%s' of different type at '%s'!", tag.Data(), folder->getPath().Data());
                int val;
                if (!folder->getTagInteger(tag,val)) {WARNclass("Failed to convert existing tag to integer type!"); continue;}
                folder->removeTag(tag);
                folder->setTagInteger(tag,val);
              }
            }
            if (TQStringUtils::equal(op,"=") && (replace || !folder->hasTagInteger(tag) ) ) {folder->setTag(tag,param.getTagIntegerDefault("value",0)); continue;}
            if (TQStringUtils::equal(op,"+")) {folder->setTag(tag,folder->getTagIntegerDefault(tag,0) + param.getTagIntegerDefault("value",0)); continue;}
            if (TQStringUtils::equal(op,"-")) {folder->setTag(tag,inverted ? param.getTagIntegerDefault("value",0) - folder->getTagIntegerDefault(tag,0) : folder->getTagIntegerDefault(tag,0) - param.getTagIntegerDefault("value",0)); continue;}
            if (TQStringUtils::equal(op,"*")) {folder->setTag(tag,folder->getTagIntegerDefault(tag,0) * param.getTagIntegerDefault("value",0)); continue;}
            if (TQStringUtils::equal(op,"/")) {
              WARNclass("Performing interger division, results may be unexpected!");
              folder->setTag(tag,inverted ? param.getTagIntegerDefault("value",0) / folder->getTagIntegerDefault(tag,0) : folder->getTagIntegerDefault(tag,0) / param.getTagIntegerDefault("value",0)); //division by zero should be caught at an earlier stage.
              continue;
              }
            if (TQStringUtils::equal(op,"^")) { folder->setTag(tag,inverted ? pow( param.getTagIntegerDefault("value",0) , folder->getTagIntegerDefault(tag,0) ) : pow( folder->getTagIntegerDefault(tag,0) , param.getTagIntegerDefault("value",0) ) ); continue; }
          }else if (isBool) {
            if (!folder->hasTagBool(tag) && !(create||force) ) {
              //don't create a new tag
              continue;
            }
            if (folder->hasTag(tag) && !folder->hasTagBool(tag)) {
              if (!force) {
                WARNclass("Tag '%s' at '%s' is already present but of different type!", tag.Data(), folder->getPath().Data());
                continue;
              } else {
                if (!silent) WARNclass("Replacing existing tag '%s' of different type at '%s'!", tag.Data(), folder->getPath().Data());
                bool val;
                if (!folder->getTagBool(tag,val)) {WARNclass("Failed to convert existing tag to boolean type!"); continue;}
                folder->removeTag(tag);
                folder->setTagBool(tag,val);
              }
            }
            if (TQStringUtils::equal(op,"=") && (replace || !folder->hasTagBool(tag) ) ) {folder->setTag(tag,param.getTagBoolDefault("value",false)); continue;}
            if (TQStringUtils::equal(op,"==")) {folder->setTag(tag,folder->getTagBoolDefault(tag,false) == param.getTagBoolDefault("value",false)); continue;}
            if (TQStringUtils::equal(op,"!=")) {folder->setTag(tag,folder->getTagBoolDefault(tag,false) != param.getTagBoolDefault("value",false)); continue;}
            if (TQStringUtils::equal(op,"&&")) {folder->setTag(tag,folder->getTagBoolDefault(tag,false) && param.getTagBoolDefault("value",false)); continue;}
            if (TQStringUtils::equal(op,"||")) {folder->setTag(tag,folder->getTagBoolDefault(tag,false) || param.getTagBoolDefault("value",false)); continue;}
          }

        }
        delete targetList;

      } else if (isCmdReplace) {
        TString raw = strParameter;
        TString pathFilter, tagFilter, typeFilter;
        TQTaggable param;
        TQStringUtils::removeLeadingBlanks(raw);
        if (TQStringUtils::findFree(raw,"=","()[]{}\"\"''") < TQStringUtils::findFree(raw,",","()[]{}\"\"''") ) {
        //default path and tag filter
          pathFilter = ".";
          tagFilter = "*";
          typeFilter = "all";
        } else {
        //Full command treatment (non-default folder/tag filter)
          if (TQStringUtils::readBlock(raw,tagFilter,"\"\"''") > 0 ) {
            TQStringUtils::removeLeadingBlanks(raw);
          } else {
            TQStringUtils::readUpTo(raw,tagFilter,",");
          }
          if (TQStringUtils::removeLeading(raw,",") < 1) {
            errMsg = TString::Format("Failed to parse line $replace(%s), should be $replace(\"(typeFilter)folderFilter:tagFilter\",tag1=\"value1\",tag2=\"value2\",...)",strParameter.Data());
            return false;
          }
          if (TQStringUtils::readBlock(tagFilter,typeFilter,"()") == 0) typeFilter = "all";
          TQStringUtils::readUpTo(tagFilter,pathFilter,":");
          TQStringUtils::removeLeading(tagFilter,":",1);
          TQStringUtils::removeLeadingBlanks(typeFilter);
          TQStringUtils::removeTrailingBlanks(typeFilter);
          TQStringUtils::removeTrailingBlanks(pathFilter);
        }
        param.importTags(this->replaceInText(raw));
        TClass* cFilter;
        if (TQStringUtils::equal(typeFilter,"sf") || TQStringUtils::equal(typeFilter,"samplefolder") ) cFilter = TQSampleFolder::Class();
        else if (TQStringUtils::equal(typeFilter,"s") || TQStringUtils::equal(typeFilter,"sample") ) cFilter = TQSample::Class();
        else cFilter = TQFolder::Class();
        this->replaceInFolderTags(param,pathFilter,tagFilter,cFilter);
        /*
        TList* targetList = this->getListOfFolders(pathFilter,cFilter);
        if (!targetList) {
          errMsg = TString::Format("No matching folders found for pattern '%s'",pathFilter.Data());
          return false;
        }

        TQFolderIterator itr(targetList);
        while (itr.hasNext()) {
          TQFolder* folder = itr.readNext();
          if (!folder) continue;
          TList* lTags = folder->getListOfKeys(tagFilter);
          if (!lTags) continue;
          TIterator* itr = lTags->MakeIterator();
          TObjString* ostr;
          while ((ostr = (dynamic_cast<TObjString*>(itr->Next())))) {
            if (!ostr) continue;
            if (!folder->tagIsOfTypeString(ostr->GetString())) continue;
            folder->setTagString(ostr->GetString(), param.replaceInText(folder->getTagStringDefault(ostr->GetString(),"")));
          }
          delete lTags;
          delete itr;
        }
        delete targetList;
        */
      } else if (isCmdCreate) {
        // read object
        TQTaggable param;
        param.importTags(strParameter);
        TString path;
        //@tag: [path,type] These text argument tags are use with the command $create. The value of "path" determines the elements created. All missing elements in the given path are created as instances of the class specified via the "type" tag: if it is set to "s" or "sample", instances of TQSample are created, if the value of "type" is "sf" or "samplefolder", instances of TQSampleFolder are created. If no (or any invalid) value for "type" is given, instances of TQFolder are created.
        if (param.getNTags() < 1 || !param.getTagString("path", path)) {
          // ==> error copying: stop
          errMsg = "Wrong number of arguments to command 'create'";
          return false;
        } else {
          TString type;
          param.getTagString("type",type);
          type = TQStringUtils::makeLowercase(type);
          TQStringUtils::removeLeadingBlanks(path);
          bool useRoot = TQStringUtils::removeLeading(path,"/") > 0;
          std::vector<TString> vPath = TQStringUtils::split(path,"/");
          std::vector<TQFolder*> currentFolders;
          currentFolders.push_back(useRoot ? this->getRoot() : this);
          for (uint i=0; i<vPath.size(); ++i) {
            if (vPath.at(i).Length() < 1) continue;
            std::vector<TQFolder*> newFolders;
            for(auto currentFolder:currentFolders){
              TQFolderIterator folders(currentFolder->getListOfFolders(vPath.at(i)),true);
              TQFolder* tmpFolder = NULL;
              while(folders.hasNext()){
                tmpFolder = folders.readNext();
                newFolders.push_back(tmpFolder);
              }
              if(!tmpFolder){
                if (TQStringUtils::equal(type,"s") || TQStringUtils::equal(type,"sample") ) {
                  tmpFolder = new TQSample(vPath.at(i));

                } else if (TQStringUtils::equal(type,"sf") || TQStringUtils::equal(type,"samplefolder") ) {
                  tmpFolder = new TQSampleFolder(vPath.at(i));
                } else {
                  tmpFolder = new TQFolder(vPath.at(i));
                }
                currentFolder->addFolder(tmpFolder);
                newFolders.push_back(tmpFolder);
              }
            }
            currentFolders = newFolders;
          }
        }
      } else if (isCmdPrint) { //execute print method on current folder
        this->print(TQStringUtils::unquote(strParameter));
      } else if (isCmdPrintLine) { //print argument to console
        INFO( TString::Format("@%s: '%s'" , this->getPath().Data(), this->replaceInText(TQStringUtils::unquote(strParameter)).Data()) );
      } else if (isCmdWrite) {
        TQTaggable param;
        param.importTagWithPrefix(strParameter, "", false, "filename");
        TQFolder* target = NULL;
        TString filename = param.getTagStringDefault("filename","");
        if (filename.Length()==0) {
          errMsg = TString::Format("no file name specified");
          return false;
        }
        if (!TQUtils::ensureDirectoryForFile(filename)) {
          errMsg = TString::Format("Failed to ensure existance of directory for file '%s'",filename.Data());
          return false;
        }
        if (param.hasTagString("target")) {
          target = this->getFolder(param.getTagStringDefault("target",""));
        } else {
          target = this;
        }
        if (!target) {
          errMsg = TString::Format("could not find target folder '%s'", param.getTagStringDefault("target","").Data());
          return false;
        }
        target->exportToTextFile(filename);
      } else {
        // ==> unknown command: stop
        errMsg = TString::Format("Unknown command '%s'", cmd.Data());
        return false;
      }

    } else if (input.BeginsWith("\"")){
      TQStringUtils::removeLeading(input, "\"");
      TQStringUtils::readUpTo(input,token,"\"");
      TQStringUtils::removeLeading(input, "\"");
      if(!TQStringUtils::removeLeading(input, ";")){
        errMsg = TString::Format("Missing terminating ';' after string '%s'", token.Data());
      }
      TObjString* str = new TObjString(token);
      this->addObject(str);
    } else if (input.BeginsWith("TH")){
      TQStringUtils::readUpTo(input,token,";","()[]{}","\"\"''");
      if(!TQStringUtils::removeLeading(input, ";")){
        errMsg = TString::Format("Missing terminating ';' after histogram '%s'", token.Data());
      }
      TH1* hist = TQHistogramUtils::convertFromText(token);
      this->addObject(hist);
    } else if (!input.IsNull()) {
      // ==> unknown token: stop
      errMsg = TString::Format("Unknown token near '%s'",
                               TQStringUtils::maxLength(input, 30).ReplaceAll("\n", " ").Data());
      return false;
    }
  }

  return true;
}


//__________________________________________________________________________________|___________

TQFolder::~TQFolder() {
  // Deletes this instance of TQFolder and all its objects and sub-folders recursively

  // remove this folder from its base folder
  this->detachFromBase();

  // delete all objects of this folder
  this->deleteAll();

  // take care of the directory
  this->clearDirectoryInternal();
}

//__________________________________________________________________________________|___________

TQFolder* TQFolder::copyDirectoryStructure(const TString& basepath, int maxdepth){
  // return a full TQFolder copy of some actual file system structure
  if(basepath.BeginsWith("root://")){
    DEBUGclass("detected eos head");
    size_t pathpos = basepath.Index("/eos/");
    TString eosprefix = basepath(0,pathpos);
    TString eosurl(eosprefix);
    TQStringUtils::removeTrailing(eosurl,"/");
    TQLibrary::getQLibrary()->setEOSurl(eosurl+".cern.ch");
    TString path = basepath(pathpos,basepath.Length());
    TQFolder* f = TQFolder::copyDirectoryStructureEOS(path,maxdepth);
    if(!f) return NULL;
    f->setTagBool("eos",true);
    f->setTagString("eosprefix",eosprefix);
    f->setTagString("eospath",path);
    f->setTagString("basepath",basepath);
    return f;
  } else {
    DEBUGclass("using local variant");
    TQFolder* f = TQFolder::copyDirectoryStructureLocal(basepath,maxdepth);
    if(!f) return NULL;
    f->setTagBool("eos",false);
    f->setTagString("basepath",basepath);
    return f;
  }
  return NULL;
}

//__________________________________________________________________________________|___________

TQFolder* TQFolder::copyDirectoryStructureLocal(const TString& basepath, int maxdepth){
  // traverse a folder structure of the local physical file system and create a TQFolder-image thereof
  const TString dircmd(TString::Format("find -L %s -maxdepth %d ! -readable -prune -o -type d -print ", basepath.Data(),maxdepth));
  const TString filecmd(TString::Format("find -L %s -maxdepth %d ! -readable -prune -o -type f -print ", basepath.Data(),maxdepth+1));
  DEBUGclass(dircmd);
  TList* dirs = TQUtils::execute(dircmd,4096);
#ifdef _DEBUG_
  dirs->Print();
#endif
  DEBUGclass(filecmd);
  TList* files = TQUtils::execute(filecmd,4096);
#ifdef _DEBUG_
  files->Print();
#endif
  TString path(basepath);
  TString tmppath(path);
  TQFolder* f = new TQFolder("tmp");
  f->SetName(TQFolder::getPathTail(tmppath));
  if(dirs){
    dirs->SetOwner(true);
    TQIterator ditr(dirs);
    while(ditr.hasNext()){
      TObject* obj = ditr.readNext();
      if(!obj) continue;
      TString name = obj->GetName();
      TQStringUtils::removeLeadingText(name,basepath);
      if(!TQFolder::isValidPath(name)) continue;
      f->getFolder(TString::Format("%s+",name.Data()));
    }
    delete dirs;
  }
  if(files){
    files->SetOwner(true);
    TQIterator fitr(files);
    while(fitr.hasNext()){
      TObject* obj = fitr.readNext();
      if(!obj) continue;
      TString path(obj->GetName());
      TQStringUtils::removeLeadingText(path,basepath);
      TString name = TQFolder::getPathTail(path);
      DEBUGfunc("adding file '%s' to '%s'",name.Data(),path.Data());
      if(path.IsNull()){
        f->addObject(new TObjString(name));
      } else {
        TQFolder* newf = f->getFolder(path + "+!");
        if(!newf){
          DEBUGclass("using invalid_name");
          newf = f->getFolder("invalid_name+");
        }
        if(newf){
          DEBUGclass("adding object to '%s'@%x",newf->GetName(),newf);
	  if(!newf->addObject(new TObjString(name))){
	    ERRORclass("cannot add object '%s' to '%s'",name.Data(),newf->getPath().Data());
	  }
	}
	else ERRORfunc("unable to create '%s'",name.Data(),path.Data());
      }
    }
    delete files;
  }
  return f;
}

//__________________________________________________________________________________|___________

TQFolder* TQFolder::copyDirectoryStructureEOS(const TString& basepath, int maxdepth){
  // traverse a folder structure of the some EOS file system and create a TQFolder-image thereof
  DEBUGclass("copying directory structure '%s'",basepath.Data());
  TString path = basepath;
  TString foldername = TQFolder::getPathTail(path);
  TQFolder* f = new TQFolder(foldername);
  f->SetName(foldername);
  if(!f) return NULL;
  TQIterator itr(TQUtils::execute(TQLibrary::getEOScmd()+" ls "+basepath,1024),true);
  while(itr.hasNext()){
    TObject* obj = itr.readNext();
    TString name = TQStringUtils::makeASCII(obj->GetName());
    path = TQFolder::concatPaths(basepath,name);
    DEBUGclass("looking at '%s'",path.Data());
    TString type = "file";
    if(maxdepth > 0){
      TList* l = TQUtils::execute(TQLibrary::getEOScmd()+" stat "+path,1024);
      if(!l){
        ERRORclass("unable to execute '%s stat %s', skipping",TQLibrary::getEOScmd().Data(),path.Data());
        continue;
      }
      TString status = TQStringUtils::concat(l,"");
      delete l;
      if(status.IsNull()){
        ERRORclass("unable to retrieve status of object '%s', skipping",path.Data());
        continue;
      }
      TObjArray* a = status.Tokenize(" ");
      if(!a){
        ERRORclass("unable to tokenize '%s', skipping",status.Data());
        continue;
      }
      a->SetOwner(true);
      if(a->IsEmpty()){
        ERRORclass("no file type flag found in '%s', skipping",status.Data());
        delete a;
        continue;
      }
      type = a->Last()->GetName();
      delete a;
    }
    if(type == "directory"){
      TQFolder* subf = TQFolder::copyDirectoryStructureEOS(path,maxdepth-1);
      // in this special case, we want to allow otherwise invalid names
      // hence, we exceptionally call the underlying routine instead of addObject
      if(subf){
	f->Add(subf);
	subf->setBase(f);
      }
    } else if(type == "file"){
      f->addObject(new TObjString(name));
    }
  }
  return f;
}

//__________________________________________________________________________________|___________

int TQFolder::writeToFile(const TString& filename, bool overwrite, int depth, bool keepInMemory){
  // write this folder to a file of the given name, splitting at the given depth value
  // the name of the folder will be used as a key
  DEBUGclass("opening file '%s'",filename.Data());
  TFile* f = TFile::Open(filename,overwrite ? "RECREATE" : "UPDATE");
  if(!f) return -1;
  if(!f->IsOpen()){
    delete f;
    return -2;
  }
  DEBUGclass("writing to file");
  bool retval = this->writeFolderInternal(f,this->GetName(),depth,keepInMemory);
  DEBUGclass("closing file");
  f->Close();
  return (int)(retval);
}

//__________________________________________________________________________________|___________

void TQFolder::setInfoTags(){
  // deposit a couple of tags with timestamp and meta-information about software versions
  this->setTagString(".creationDate",TQUtils::getTimeStamp());
  this->setTagString(".createdBy",TQLibrary::getApplicationName());
  this->setTagString(".libVersion",TQLibrary::getVersion());
  this->setTagString(".rootVersion",TQLibrary::getROOTVersion());
  this->setTagString(".gccVersion",TQLibrary::getGCCVersion());
}

//__________________________________________________________________________________|___________

TQFolder* TQFolder::findCommonBaseFolder(TCollection* fList, bool allowWildcards){
  // find a common base folder of some list
  TString base = "";
  bool first = true;
  TQFolderIterator itr(fList);
  while(itr.hasNext()){
    TQFolder* f = itr.readNext();
    const TString tmppath( allowWildcards ? f->getPathWildcarded() : f->getPath());
    if(first){
      base = tmppath;
      first=false;
    } else {
      if(TQStringUtils::reduceToCommonPrefix(base,tmppath) < 1) return NULL;
    }
  }
  if(base.IsNull()) return NULL;
  if(base.EndsWith("/")){
    TQFolder* retval = this->getFolder(base);
    if(retval) return retval;
  }
  this->getPathTail(base);
  return this->getFolder(base);
}

//__________________________________________________________________________________|___________

TList* TQFolder::exportTagsToText(const TString& filter){
  // create a TList with strings for all tags matching the filter in this folder structure
  // the result can be applied to some folder via TQFolder::importFromTextFile
  TList* folders = this->getListOfFolders("*");
  TList* retval = new TList();
  TQFolderIterator itr(folders,true);
  std::map<TString,TString> tagMap;
  while(itr.hasNext()){
    TQFolder* obj = itr.readNext();
    if(!obj) continue;
    TString tags = obj->exportTagsAsString(filter,true);
    if(tags.IsNull()) continue;
    TString path = obj->getPathWildcarded();
    tagMap[path] = tags;
  }
  for(std::map<TString,TString>::iterator it = tagMap.begin(); it != tagMap.end(); ++it){
    TString line = "<" + it->second + "> @ " + it->first + ";";
    TObjString* s = new TObjString(line);
    retval->Add(s);
  }
  return retval;
}

//__________________________________________________________________________________|___________

bool TQFolder::exportTagsToTextFile(const TString& filename, const TString& filter){
  // write to a file the strings for all tags matching the filter in this folder structure
  // the result can be applied to some folder via TQFolder::importFromTextFile
  TList * text = this->exportTagsToText(filter);
  text->Print();
  if (text) {
    text->AddFirst(new TObjString("# -*- mode: tqfolder -*-"));
    bool success = TQStringUtils::writeTextToFile(text, filename);
    delete text;
    return success;
  } else {
    return false;
  }
}

//__________________________________________________________________________________|___________

bool TQFolder::merge(TQFolder* other, bool sumElements){
  // merge another instance of TQFolder into this one. this function will
  // traverse the folder structure recursively and collect all existing
  // subfolders from both instances and merge them into one. in the case of a
  // conflict, it will always use the subfolder with the more recent time stamp
  if(this->Class() == TQFolder::Class()){
    return this->mergeAsFolder(other,sumElements ? MergeMode::SumElements : MergeMode::PreferOther);
  } else {
    ERRORclass("unable to merge '%s' with 'TQFolder'",this->Class()->GetName());
    return false;
  }
}

//__________________________________________________________________________________|___________

bool TQFolder::mergeAsFolder(TQFolder* other, MergeMode mode){
  // simply merge to folders, merging all objects, taggs and subfolders
  this->mergeTags(other);
  // merge folders
  TQFolderIterator itr(other->getListOfFolders("?"));
  while(itr.hasNext()){
    TQFolder* f = itr.readNext();
    if(!f) continue;
    TQFolder* thisF = this->getFolder(f->GetName());
    if(thisF){
      thisF->mergeAsFolder(f,mode);
    } else {
      f->detachFromBase();
      this->addObject(f);
    }
  }
  // merge objects
  this->mergeObjects(other,mode);
  // return
  return true;
}

//__________________________________________________________________________________|___________

bool TQFolder::mergeTags(TQFolder* other){
  // merge (copy) the tags from another folder to this one
  bool overwrite = this->getGlobalOverwrite();
  this->setGlobalOverwrite(false);
  this->importTags(other);
  if(overwrite) this->setGlobalOverwrite(true);
  return true;
}

//__________________________________________________________________________________|___________

void TQFolder::mergeObjects(TQFolder* other, MergeMode mode){
  // merge (move) the objects from another folder to this one
  TQIterator itr(other->GetListOfFolders());
  
  //create a helper map for faster retrieval of objects in this folder (bringing down the complexity from linear to logarithmic for that part)
  std::map<TString,TObject*> helperMap;
  std::map<TString,TObject*>::iterator helperIt;
  for (TObject*  obj: (*(this->GetListOfFolders())) ) {
    if (!obj) continue;
    //TString is well enough desinged to allow us to use the const char* from GetName() here directly:
    helperMap[obj->GetName()] = obj;
  }
 
  while(itr.hasNext()){
    TObject* obj = itr.readNext();
    if(!obj) continue;
    // folders are not handled
    if(obj->InheritsFrom(TQFolder::Class())) continue;
    //TObject* thisObj = this->getObject(obj->GetName());
    TObject* thisObj = nullptr;
    helperIt = helperMap.find(obj->GetName());
    if (helperIt != helperMap.end()) thisObj = helperIt->second; //get the TObject pointer
    
    if(!thisObj) {
      DEBUGclass("grabbing object '%s'",thisObj->GetName());
      other->Remove(obj);
      this->addObject(obj);
    } else {
      if(mode == PreferThis){
        DEBUGclass("leaving object '%s'",thisObj->GetName());
        // do nothing 
      } else if(mode == PreferOther){
        DEBUGclass("grabbing & replacing object '%s'",thisObj->GetName());
        other->Remove(obj);
        this->Remove(thisObj);
        delete thisObj;
        this->addObject(obj);
      } else if(mode == SumElements){
      if ( obj->InheritsFrom(TH1::Class()) && thisObj->InheritsFrom(TH1::Class()) ) {
        TH1* hist = static_cast<TH1*>(obj);
        TH1* thisHist = static_cast<TH1*>(thisObj);
        if(TQHistogramUtils::checkConsistency(hist,thisHist)){
          DEBUGclass("summing histogram '%s'",thisObj->GetName());
          thisHist->Add(hist);
          continue;
        }
      } else if ( obj->InheritsFrom(THnBase::Class()) && thisObj->InheritsFrom(THnBase::Class()) ) {
        THnBase* ndimHist = static_cast<THnBase*>(obj);
        THnBase* thisndimHist = static_cast<THnBase*>(thisObj);
        if(TQTHnBaseUtils::checkConsistency(ndimHist,thisndimHist)){
          DEBUGclass("summing n-dim histogram '%s'",thisObj->GetName());
          thisndimHist->Add(ndimHist);
          continue;
        }
      } else if ( obj->InheritsFrom(TQCounter::Class()) && thisObj->InheritsFrom(TQCounter::Class()) ) {
        TQCounter* counter = static_cast<TQCounter*>(obj);
        TQCounter* thisCounter = static_cast<TQCounter*>(thisObj);
        if(counter && thisCounter){
          DEBUGclass("summing counter '%s'",thisObj->GetName());
          thisCounter->add(counter);
          continue;
        }
      } else if ( obj->InheritsFrom(TQTable::Class()) && thisObj->InheritsFrom(TQTable::Class()) ) {
        if ( obj->InheritsFrom(TQXSecParser::Class()) || thisObj->InheritsFrom(TQXSecParser::Class()) ) continue; //skip XSPs, this would lead to an incredibly stupid object as you'd typically end up with sample folders with an XSP which is a 10^wayTooMuch fold copy of the one created in makeSampleFile
        TQTable* tbl = static_cast<TQTable*>(obj);
        TQTable* thisTbl = static_cast<TQTable*>(thisObj);
        if(tbl && thisTbl){
          DEBUGclass("appending table '%s'",thisObj->GetName());
          thisTbl->merge(tbl);
          continue;
        }
      } else if ( obj->InheritsFrom(TObjString::Class()) && thisObj->InheritsFrom(TObjString::Class()) ) {
        TObjString* str = static_cast<TObjString*>(obj);
        TObjString* thisStr = static_cast<TObjString*>(thisObj);
        if(str && thisStr){
          if(TQStringUtils::equal(str->String(),thisStr->String())){
            continue;
          } else {
            ERRORclass("cannot merge two TObjStrings with different content!");
            continue;
          }
        }
      }
      ERRORclass("cannot merge objects '%s' of type '%s' and '%s'",obj->GetName(),obj->ClassName(),thisObj->ClassName());
      }
    }
  }
}

//__________________________________________________________________________________|___________

int TQFolder::replaceInFolderTags(TQTaggable& params, const TString& path, const TString& tagFilter, TClass* typeFilter ){
  TList* targetList = this->getListOfFolders(path,typeFilter);
  if (!targetList) {
    WARNclass("No matching folders found for pattern '%s'",path.Data());
    return -1;
  }
  TQFolderIterator itr(targetList);
  while (itr.hasNext()) {
    TQFolder* folder = itr.readNext();
    if (!folder) continue;
    folder->replaceInTags(params,tagFilter);
  }
  delete targetList;
  return 0;
}
