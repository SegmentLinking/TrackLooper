// this file looks like it's c, but it's actually -*- C++ -*-
#ifndef __TQFolder__
#define __TQFolder__

#include "TFolder.h"
#include "TClass.h"
#include "QFramework/TQTaggable.h"
#include "TList.h"
#include "TDirectory.h"
#include "TKey.h"
#include <limits>

#include "QFramework/TQFlags.h"

class TQFolder : public TFolder, public TQTaggable {
 
protected:

  enum MergeMode {
    PreferThis = 0,
    SumElements = 1,
    PreferOther = 2
  };
  
  TDirectory* fMyDir; //!
  bool fOwnMyDir; //!
  TString fExportName; //!
  TQFolder * fBase;
  bool isFullyResolved; //!

  bool fIsEquivalentToSnapshot; //!
 
  virtual TObject * importObjectFromDirectory(TDirectory * dir, TString importPath, bool recurse=true);

  virtual bool split(TDirectory * dir, int depth);
  virtual bool writeFolderInternal(TDirectory * dir, const TString& exportName, int depth, bool keepInMemory);
 
  virtual bool importFromTextPrivate(TString input, int &nNewlines, TString &errFile, TString &errMsg);
  virtual bool importFromTextFilePrivate(const TString& filename_, int &nNewlines, TString &errFile, TString &errMsg);
 
  void setBase(TQFolder * base_);
  void setDirectoryInternal(TDirectory* dir);
  void clearDirectoryInternal();
  bool writeContentsToHTML(std::ostream& out, int expandDepth, bool includeUntextables);

  void printInternal(TString options, int indent, bool resolve=true);
  bool printDiff(TQFolder * f, TQTaggable& options, int indent);
  void mergeObjects(TQFolder* other, MergeMode mode);
  bool mergeAsFolder(TQFolder* other, MergeMode mode);

public:

  // static string processing
  static bool parseDestination(TString dest, TString &path, TString &newName);
  static bool parseLocation(TString importpath, TString& filename, TString& objname);
  static TString makeExportName(TString exportName);

  static const TString& getValidNameCharacters();
  static bool isValidName(const TString& name);
  static bool isValidPath(TString path, bool allowRelativePaths = true, bool allowWildcards = true, bool allowArithmeticStringExpressions = false);
  static TString makeValidIdentifier(TString identifier, TString replacement = "");
  static TString makeValidPath(TString path, TString replacement = "", bool allowRelativePaths = true, bool allowWildcards = true);
 
  static TString getPathHead (TString &path);
  static TString getPathTail (TString &path);
  static TString getPathWithoutHead (TString path);
  static TString getPathWithoutTail (TString path);
 
  static TString concatPaths(TString path1, TString path2);
  static TString concatPaths(TString path1, TString path2, TString path3);
  static TString concatPaths(TString path1, TString path2, TString path3, TString path4);
 
  static int countPathLevels(TString path, bool checkPathTokens = true);
 
  // directory handling and laziness
  void autoSetExportName();
  void autoSetExportNames();
  void setExportName(const TString& name);
  const TString& getExportName();
  int setDirectory(TDirectory* dir = gDirectory, bool own=false);
  TDirectory* getDirectory();
  bool isOnDisk(TDirectory* dir);
  bool isOnDisk();
  bool isLazy();
  bool collapse();
 
  int resolveImportLinks(bool recurse=true);
  TObject * resolveImportLink(const TString& linkName, bool recurse=true);
  int resolveImportLinks(TDirectory * dir, bool recurse=true);
  TObject * resolveImportLink(const TString& linkName, TDirectory * dir, bool recurse=true);

  // static helpers
  static TFile* openFile(TString& importPath, const TString& opt = "READ");
  static TQFolder * newFolder(TString name);
  static TQFolder * loadLazyFolder(TString path);
  static TQFolder * loadFolder(TString path, bool lazy=false);
  static TQFolder * loadFromTextFile(TString filename, bool showErrorMessage = false);
  static TQFolder * loadFromTextFile(TString filename, TString &errorMessage);
 
  static TQFolder * copyDirectoryStructureLocal(const TString& basepath, int maxdepth = 999);
  static TQFolder * copyDirectoryStructureEOS (const TString& basepath, int maxdepth = 999);
  static TQFolder * copyDirectoryStructure (const TString& basepath, int maxdepth = 999);

  // basic class functionality
  TQFolder();
  TQFolder(const TString& name);
  virtual ~TQFolder();

  virtual TQFolder * newInstance(const TString& name);

  TString getName() const;
  void setName(const TString& newName);
  const TString& getNameConst() const;

  TString getPath();
  TString getPathWildcarded();
  TQFolder * getBase(int depth = 1) const;
  TQFolder * getRoot();
  bool isRoot();
  int getDistToRoot() const;
  int getDepth();
  int areRelated(const TQFolder* other) const;
  TList * getTraceToRoot(bool startAtRoot = false);
  bool isBaseOf(TQFolder * folder);
  TQFolder * detachFromBase();

  bool checkConsistency(bool verbose = false);
  virtual int sortByName();
  virtual void sortByNameRecursive();

  virtual int getSize(bool memoryOnly = true);
  virtual TString getSizeAsString(bool memoryOnly = true);
  bool isEmpty() const;
 
  // tag handling
  virtual TQTaggable * getBaseTaggable() const override;
  virtual TList * getDescendantTaggables() override;
  virtual TList * getListOfTaggables(const TString& taggables) override;
  virtual TList * getTaggablesByName(const TString& taggables) override;

  bool executeCopyCommand(TString object, TString& errMsg, bool moveOnly, const TString& destPrefix = "");

  void setInfoTags();
 
  // advanced functionality
  virtual bool isEquivalentTo(TQFolder * f, const TString& options = "");
  virtual bool isEquivalentTo(TQFolder * f, TQTaggable& options);

  bool printDiff(const TString& path1, const TString& path2, const TString& options = "");
  bool printDiff(TQFolder * f, const TString& options = "");

  bool merge(TQFolder* other, bool sumElements=false);
  bool mergeTags(TQFolder* other);

  TQFolder* findCommonBaseFolder(TCollection* fList, bool allowWildcards=true);

  // pretty-print
  void print(const TString& options = "");
  void printContents(const TString& options = "");

  // working with the folder structure
  TList * getListOfFolders(const TString& path_ = "?", TClass * tclass = TQFolder::Class(), bool toplevelOnly = false);
  TList * getListOfObjects(TString path_ = "?", TClass * tclass = TObject::Class());
  TList * getListOfObjectPaths(TString path_ = "?", TClass * tclass = TObject::Class());
  TList * getListOfFolders(const TString& path_, TQFolder * template_, bool toplevelOnly = false);

  std::vector<TString> getFolderPaths(const TString& path_ = "?", TClass * tClass = TQFolder::Class(), bool toplevelOnly = false);
  std::vector<TString> getFolderPathsWildcarded(const TString& path_ = "?", TClass * tClass = TQFolder::Class(), bool toplevelOnly = false);
  
  TQFolder * getFolder(TString path_, TQFolder * template_, int * nFolders_);
  TQFolder * getFolder(TString path_, TClass * tclass, int * nFolders_ = 0);
  TQFolder * getFolder(const TString& path);
  TQFolder * getFolder(const char* path);
  TQFolder * addFolder(TQFolder * folder_, TString path_, TQFolder * template_);
  TQFolder * addFolder(TQFolder * folder_, TString path_ = "", TClass * tclass = 0);
 
  TList * getListOfObjectNames(TClass * class_ = 0, bool recursive = false, TString path_ = "");
  TObject * getObject(const TString& name_, const TString& path_ = "");
  TObject * getCopyOfObject(const TString& name_, const TString& path_ = "");
  TString getObjectPath(TString name_);
  TList* getObjectPaths(TString namepattern, TString pathpattern = "./", TClass* objClass = TObject::Class()); 

  virtual TQFolder * addObject(TObject * object, TString destination = "");
  virtual TQFolder * addCopyOfObject(TObject * object, TString destination = "");

  virtual bool hasObject(TString name);

  virtual bool removeObject(const TString& name);
  virtual int deleteObject(TString name, bool removeOnly = false);
  virtual int deleteAll();

  virtual bool moveTo(TQFolder * dest, const TString& newName = "");
  virtual bool moveTo(const TString& dest);
  virtual bool moveFolder(const TString& source, const TString& dest);

  virtual TQFolder * copy(const TString& newName = "");

  virtual TQFolder * copyTo(TQFolder * dest);
  virtual TQFolder * copyTo(const TString& dest);
  virtual TQFolder * copyFolder(const TString& source, const TString& dest);

  virtual int getNElements(bool recursive = false, TClass * class_ = 0);
  virtual int getNObjects(const TString& nameFilter = "", bool recursive = false);
  

  // file I/O
  virtual bool writeDirectory(TDirectory * baseDir);
  virtual bool writeUpdate(int depth = -1, bool keepInMemory = true);
  virtual bool writeFolder(TDirectory * dir, TString name, int depth = -1, bool keepInMemory = true);
  virtual bool writeFolderMaxSize(TDirectory * dir, TString name, int maxSizeInMB = 10, bool keepInMemory = true);
  virtual bool writeFolder(TDirectory * dir, int depth = -1, bool keepInMemory = true);
  virtual bool writeFolderMaxSize(TDirectory * dir, int maxSizeInMB = 10, bool keepInMemory = true);
  virtual int writeToFile(const TString& filename, bool overwrite = 0, int depth=-1, bool keepInMemory = true);

  virtual TList * exportToText(bool includeUntextables = false, int indent = 0);
  virtual bool exportToTextFile(const TString& filename, bool includeUntextables = false);
  virtual TList * exportTagsToText(const TString& filter);
  virtual bool exportTagsToTextFile(const TString& filename, const TString& filter);

  virtual TObject * importObject(TString importPath, bool recurse=true);

  bool exportToHTMLFile(const TString& filename, int expandDepth = 1, bool includeUntextables=true);

  virtual bool importFromText(const TString& input);
  virtual bool importFromText(const TString& input, TString &errorMessage);
  virtual bool importFromTextFile(const TString& filename);
  virtual bool importFromTextFile(const TString& filename, TString &errorMessage);
  virtual bool importFromTextFiles(const TString& filePattern);
  virtual bool importFromTextFiles(const TString& filePattern, TString &errorMessage);

  //expanded functionality for TQFolder (basic version can be found in TQTaggable)
  int replaceInFolderTags(TQTaggable& params, const TString& path, const TString& tagFilter = "*", TClass* typeFilter = TQFolder::Class() );

  // ROOT related functions

  virtual int Compare(const TObject * obj) const override;
  virtual bool IsSortable() const override;
  ClassDefOverride(TQFolder, 1); // container class for all types of objects and meta-information

};

#endif

