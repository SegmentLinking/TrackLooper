#include "QFramework/TQFolder.h"
#include "QFramework/TQSampleFolder.h"
#include "QFramework/TQSampleDataReader.h"
#include "QFramework/TQSampleVisitor.h"
#include "QFramework/TQStringUtils.h"
#include "QFramework/TQHistogramUtils.h"
#include "QFramework/TQListUtils.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "QFramework/TQIterator.h"
#include "TIterator.h"
#include "QFramework/TQSample.h"
#include "QFramework/TQValue.h"
#include "QFramework/TQUtils.h"
#include "TList.h"
#include "TFile.h"
#include "TMath.h"
#include "TH1.h"

// #define _DEBUG_

#include "QFramework/TQLibrary.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQSampleFolder:
//
// The TQSampleFolder class is a representation of a certain group/category of event samples
// and by inheriting from the TQFolder class allows to build up a tree-like structure. Analysis 
// results (e.g. histograms or cutflow counters) corresponding to the group of samples
// represented by an instance of the TQSampleFolder class may be stored in specific sub-folders
// (instances of TQFolder) of that instance (e.g. ".histograms" or ".cutflow"). Analysis results
// are retrieved from instances of TQSampeFolder using the TQSampleDataReader class, with
// wrappers implemented in the TQSampleFolder class:
//
// - TQSampleFolder::getHistogram("<path>", "<name>", ...)
//
// - TQSampleFolder::getCounter("<path>", "<name>", ...)
//
// A new and independent instance of TQSampleFolder is created using the static method
// TQSampleFolder::newSampleFolder(...):
//
// TQSampleFolder * sf = TQSampleFolder::newSampleFolder("sf");
//
// whereas a new instance inside an existing sample folder tree can be created using
// TQSampleFolder::getSampleFolder(...): 
//
// TQSampleFolder * sf = sf->getSampleFolder("newSf+");
//
// similar to TQFolder::getFolder(...).
//
// Histograms and counter can be deleted recursively using
//
// - TQSampleFolder::deleteHistogram(...)
// - TQSampleFolder::deleteSingleCounter(...)
// - TQSampleFolder::deleteHistograms(...)
// - TQSampleFolder::deleteCounter(...)
//
// and renamed recursively using
//
// - TQSampleFolder::renameHistogram(...)
// - TQSampleFolder::renameCounter(...)
//
// Histograms and counter can be copied (optionally performing additional operations on these
// objects) using:
//
// - TQSampleFolder::copyHistogram(...)
// - TQSampleFolder::copyHistograms(...)
//
// The TQSampleFolder class provides powerful features to validate cutflow counter between
// two different instances of TQSampleFolder:
//
// - TQSampleFolder::validateAllCounter(...)
// - TQSampleFolder::validateCounter(...)
//
// Generalization of histograms and counter:
//
// Storing histograms or other analysis result objects in a sample folder tree can become very
// memory extensive if for each sample an individual instance is stored. To reduce the
// granularity of object storage and thereby the memory usage the concept of "generalization"
// has been introduced. Generalizing e.g. a histogram at some node in the sample folder tree
// will sum up and remove individual contributions within the corresponding sub-tree and store
// the final sum in the respective sample folder the generalization has taken place at:
//
// - TQSampleFolder::generalizeHistograms(...)
// - TQSampleFolder::generalizeCounter(...)
//
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQSampleFolder)

const TString TQSampleFolder::restrictSelectionTagName = TString("restrict.selected");
//__________________________________________________________________________________|___________

TQSampleFolder::TQSampleFolder() : TQFolder() {
  // Default constructor of TQSampleFolder class: a new and empty instance of
  // TQSampleFolder is created and initialized. Its name will be set to "unkown".
  // Please note: users should not use this constructor but the static factory method
  // TQSampleFolder::newSampleFolder(...). This default constructor has to be present
  // to allow ROOT's CINT to stream instances of TQSampleFolder.

  SetName("unknown");

  init();
}


//__________________________________________________________________________________|___________

TQSampleFolder::TQSampleFolder(TString name) : TQFolder(name) {
  // Default constructor of TQSampleFolder class: a new and empty instance of
  // TQSampleFolder is created and initialized. Its name will be set to the value of
  // the parameter <name> if it is a valid name.
  // Please note: users should not use this constructor but the static factory method
  // TQSampleFolder::newSampleFolder(...).

  init();
}


//__________________________________________________________________________________|___________

TQSampleFolder * TQSampleFolder::newSampleFolder(TString name) {
  // Returns a new and empty instance of the TQSampleFolder class with name <name>
  // if <name> is a valid name and a null pointer otherwise. Please refer to the
  // documentation of the static method TQSampleFolder::isValidName(...) for details
  // on valid folder names.

  // check if <name> is a valid name for instances of TQSampleFolder
  if (isValidName(name)) {
    // create and return a new instance of TQSampleFolder with name <name>
    return new TQSampleFolder(name);
  } else {
    // return NULL pointer since <name> is not a valid folder name
    return NULL;
  }
}


//__________________________________________________________________________________|___________

TQFolder * TQSampleFolder::newInstance(const TString& name) {
  // Returns a new and empty instance of the TQSampleFolder class with name <name>
  // if <name> is a valid name and a null pointer otherwise. Please note: this method
  // does exactly the same as TQSampleFolder::newSampleFolder(...) but may be over-
  // written by sub-classes of TQSampleFolder.

  // create and return new instance of TQFolder with name <name>
  return TQSampleFolder::newSampleFolder(name);
}


//__________________________________________________________________________________|___________

TQSampleFolder * TQSampleFolder::loadLazySampleFolder(const TString& path) {
  // this function works exactly as TQSampleFolder::loadSampleFolder, with the
  // exception that all folder branches that have been externalized/splitted
  // when writing the file will remain in collapsed state until they are
  // accessed. for large files with an excessive use of
  // externalization/splitting, this will significantly speed up accessing the
  // data, since the branches are only expanded on-demand.
  //
  // please note, however, that you will only experience a total speed gain if
  // you only access small fractions of your data. if you plan to read most of
  // the file's data at some point, requiring the expansion of all branches,
  // this 'lazy' feature will only postpone the work of loading the data into
  // RAM to the point where it is accessed bringing no total speed gain.
  return TQSampleFolder::loadSampleFolder(path,true);
}


//__________________________________________________________________________________|___________

TQSampleFolder * TQSampleFolder::loadSampleFolder(TString path, bool lazy) {
  // Loads an instance of TQSampleFolder from an external ROOT file and returns a
  // pointer to it and a NULL pointer in case of failure. The ROOT file and the key
  // within the key within the ROOT file to be read is identified by <path> which
  // has to have the form "filename:keyname", e.g. "file.root:folder".
  // If the sample folder to load is stored within a structre of TDirectory in the
  // file it can be accessed by prepending the corresponding path to the sample
  // folder name, e.g. "file.root:dir1/dir2/folder". To load only a subfolder of an
  // instance of TQSampleFolder from the ROOT file one can append the corresponding
  // path to the folder name, e.g. "file.root:folder/subfolder". In this case a
  // pointer to "subfolder" is returned which is made the root folder before.
  // 
  // the 'lazy' flag will trigger lazy loading, please refer to TQFolder::loadLazySampleFolder
  // for documentation of this feature.
  TQSampleFolder * dummy = TQSampleFolder::newSampleFolder("dummy");
  TString pathname = path;

  DEBUG("attempting to open file at path '%s'",path.Data());

  TFile* file = TQFolder::openFile(path,"READ");
  if(!file || !file->IsOpen()){
    if(file) delete file;
    ERRORclass("unable to open file '%s'", pathname.Data());
    return NULL;
  }

  TQSampleFolder * imported = dynamic_cast<TQSampleFolder*>(dummy->importObjectFromDirectory(file, path,!lazy));

  // the sample folder to load and return
  TQSampleFolder * folder = NULL;

  // check the sample folder that has been imported and get the one to return
  if (imported){
    while (imported->getBaseSampleFolder() != dummy)
      imported = imported->getBaseSampleFolder();

    /* detach imported folder from dummy folder */
    folder = dynamic_cast<TQSampleFolder*>(imported->detachFromBase());
  }

  // delete the dummy sample folder
  delete dummy;

  /* close file and delete file pointer */
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

  // return the sample folder
  return folder;
}


//__________________________________________________________________________________|___________

TQSampleFolder * TQSampleFolder::newSampleList(const TString& name,
                                               const TString& treeLocation, double normalization) {
  // Scans the file system for files matching the string pattern of file part in
  // <treeLocation> and returns an instance of TQSampleFolder with name <name> and
  // instances of TQSample referring to each matching file. The tree name is set
  // to the tree part in <treeLocation> and the sample normalization is set to
  // <normalization>.
  //
  // Examples:
  //
  // - TQSampleFolder * samples =
  // TQSampleFolder::newSampleList("data", "data_*.root:Tree", 1.) will create
  // a sample folder of samples for each "data_*.root" ntuple using the TTree
  // "Tree".

  // make sure the user requested a valid name
  if (!TQSampleFolder::isValidName(name)) {
    return NULL;
  }

  // read tree location
  TList * treeLocations = TQSample::splitTreeLocations(treeLocation);
  if (!treeLocations)
    return NULL;
  if (treeLocations->GetEntries() != 1) {
    delete treeLocations;
    return NULL;
  }
 
  TString thisTreeLocation = treeLocations->First()->GetName();
  delete treeLocations;

  TString pattern = TQSample::extractFilename(thisTreeLocation);
  TString treeName = TQSample::extractTreename(thisTreeLocation);

  // the sample folder to return
  TQSampleFolder * sampleFolder = NULL;

  // get the list of files matching the string pattern
  TList * files = TQUtils::getListOfFilesMatching(pattern);

  if (files) {
    sampleFolder = TQSampleFolder::newSampleFolder(name);

    // loop over the list of matching files
    TQIterator itr(files);
    while (itr.hasNext()) {

      /* full filename */
      TString fullFilename = itr.readNext()->GetName();

      /* the filename without path */
      TString filename = fullFilename;
      filename = TQFolder::getPathTail(filename);

      /* remove trailing ".root" */
      if (filename.EndsWith(".root"))
        filename.Remove(filename.Length() - 5, 5);

      /* ensure a valid name */
      TString name = TQStringUtils::makeValidIdentifier(
                                                        filename, TQSampleFolder::getValidNameCharacters(), "_");

      /* ensure a unique name */
      TString namePrefix = name;
      int i = 2;
      while (sampleFolder->hasObject(name))
        name = TString::Format("%s_n%d", namePrefix.Data(), i++);

      /* create a new sample */
      TQSample * sample = new TQSample(name);
      /* set parameter */
      sample->setTreeLocation(fullFilename + ":" + treeName);
      sample->setNormalisation(normalization);
      /* add to folder */
      sampleFolder->addSampleFolder(sample);
    }
  }

  // sort the elements (samples) within the new sample folder by name
  if (sampleFolder) {
    sampleFolder->sortByName();
  }

  // now return the new sample folder
  return sampleFolder;
}


//__________________________________________________________________________________|___________

void TQSampleFolder::init() {
  // Nothing happens here

}


//__________________________________________________________________________________|___________

TQFolder * TQSampleFolder::getTemplate() {
  return new TQSampleFolder("template");
}


//__________________________________________________________________________________|___________

TQSampleFolder * TQSampleFolder::addSampleFolder(TQSampleFolder * sampleFolder_, TString path_, TQSampleFolder * template_) {
  // legacy wrapper for the TClass variant
  return this->addSampleFolder(sampleFolder_,path_,template_ ? template_->Class() : NULL);
}

//__________________________________________________________________________________|___________

TQSampleFolder * TQSampleFolder::addSampleFolder(TQSampleFolder * sampleFolder_, TString path_, TClass * tclass) {

  /* add the new sample folder and return a pointer on its new base samplefolder */
  if (tclass) {
    return (TQSampleFolder*)addFolder(sampleFolder_, path_, tclass);
  } else {
    /* add the sample folder */
    TQSampleFolder * sampleFolder = (TQSampleFolder*)addFolder(sampleFolder_, path_, TQSampleFolder::Class());
    /* return the sample folder */
    return sampleFolder;
  }
 
}

//__________________________________________________________________________________|___________

int TQSampleFolder::visitMe(TQSampleVisitor * visitor, bool requireSelectionTag) {
  // recursively visit this sample (folder) with the given visitor
  return visitor->visit(this,requireSelectionTag);
}


//__________________________________________________________________________________|___________

int TQSampleFolder::visitSampleFolders(TQSampleVisitor * visitor, const TString& category_) {
  // Lets the sample visitor (instance of TQSampleVisitor) <visitor> visit those
  // sample folders that match the path pattern <category> and returns the number
  // visits. <category> may consists of multiple comma-separated paths and wildcards
  // "?" and "*" may be used. If an exclamation mark "!" is appended to a path pattern
  // the corresponding sample folders are visited non-recursively. This "!" feature doesn't actually work-> TODO: fix or remove comment

  // as a debugging special, it is possible to pass NULL as a visitor. in this
  // case, this function will simply print a warning for each sample folder
  // that would have been visited if a valid visitor had been provided
  
  TString category = TQStringUtils::trim(category_,TQStringUtils::getAllBlanks());//remove whitespaces at beginning and end
  /* allow multiple categories */
  //not needed, getListOf(Sample)Folders already takes care of comma separated lists
  std::vector<TString> categories = TQStringUtils::tokenizeVector(category, ",", true);
  if(categories.size() < 1) categories.push_back("?");
  
  // loop over all sample folders matching <category>
  TCollection* c = this->getListOfSampleFolders(category);
  if(!c) return 0;
  if(c->GetEntries() < 1) c->Add(this);
  // let the list die with the iterator (..., true)
  TQSampleFolderIterator itr(c,true);
  while (itr.hasNext()) {
    // visit the sample folder
    TQSampleFolder* sf = itr.readNext();
    if(!sf) continue;

    if(visitor){
      sf->setTagBool(TQSampleFolder::restrictSelectionTagName,true);
      //TRY(
      //  nVisits += sf->visitMe(visitor);
      //,TString::Format("An error occured while visiting the sample folder at path '%s'.",sf->getPath().Data())
      //)
    } else {
      WARNclass("cannot visit '%s' with NULL visitor!",sf->getPath().Data());
    }
  }
  
  // warning if no matching sample folder could be found
  if (itr.getCounter() == 0) {
    WARNclass("No matching sample folder for '%s' found", category.Data());
  }
	
  
  //we delay the call to the actual visiting in order to ensure all folders are visited which lead up to a selected folder (instead of calling the visitMe method on the selected ones directly, see HWWATLAS-138)
	int visited = 0;
  if (visitor) {
    TRY(
				visited = this->visitMe(visitor,true);
				,TString::Format("An error occured while visiting the sample folders with root node '%s'.",this->getPath().Data())
				)
			//do some cleanup to not spam the output files with too many technical tags:
			itr.reset();
    while(itr.hasNext()) {
      TQSampleFolder* sf = itr.readNext();
      if (!sf) continue;
      sf->removeTag(TQSampleFolder::restrictSelectionTagName);
    }
  }
  
  // return the number of visits
  return visited;
}


//__________________________________________________________________________________|___________

TList * TQSampleFolder::getListOfSampleFolders(const TString& path_, TQSampleFolder* template_, bool toplevelOnly) {
  /* get a list of sample folders */
  return this->getListOfSampleFolders(path_,template_ ? template_->Class() : TQSampleFolder::Class(), toplevelOnly);
}

//__________________________________________________________________________________|___________

TList * TQSampleFolder::getListOfSampleFolders(const TString& path_, TClass * tclass, bool toplevelOnly) {
  /* get a list of sample folders */
  if (tclass) {
    return getListOfFolders(path_, tclass, toplevelOnly);
  } else {
    /* get the list */
    TList * list = getListOfFolders(path_, TQSampleFolder::Class(), toplevelOnly);
    /* return the list */
    return list;
  }
 
}

//__________________________________________________________________________________|___________

std::vector<TString> TQSampleFolder::getSampleFolderPaths(const TString& path_, TClass * tClass, bool toplevelOnly) {
  return getFolderPaths(path_,tClass ? tClass : TQSampleFolder::Class() , toplevelOnly);
}

//__________________________________________________________________________________|___________

std::vector<TString> TQSampleFolder::getSampleFolderPathsWildcarded(const TString& path_, TClass * tClass, bool toplevelOnly) {
  return getFolderPathsWildcarded(path_,tClass ? tClass : TQSampleFolder::Class() , toplevelOnly);
}

//__________________________________________________________________________________|___________

std::vector<TString> TQSampleFolder::getSamplePaths(const TString& path_, TClass * tClass, bool toplevelOnly) {
  return getFolderPaths(path_,tClass ? tClass : TQSample::Class() , toplevelOnly);
}

//__________________________________________________________________________________|___________

std::vector<TString> TQSampleFolder::getSamplePathsWildcarded(const TString& path_, TClass * tClass, bool toplevelOnly) {
  return getFolderPathsWildcarded(path_,tClass ? tClass : TQSample::Class() , toplevelOnly);
}

//__________________________________________________________________________________|___________

TQSample * TQSampleFolder::getSample(const TString& path) {
  // Returns the sample (instance of TQSample) that matches the path pattern <path>
  // and a NULL pointer in case no match can be found. If more than one sample matches
  // <path> (because wildcards are used) the first match is returned.

  return (TQSample*)getFolder(path, TQSample::Class());
}


//__________________________________________________________________________________|___________

TQSampleFolder * TQSampleFolder::getBaseSampleFolder() {
  // Returns a pointer to the base sample folder and a NULL pointer if either there
  // is no base folder or the base folder is not an instance of TQSampleFolder.

  // get the base folder
  TQFolder * base = getBase();

  // check if the base folder is an instance of TQSampleFolder
  if (base && base->InheritsFrom(TQSampleFolder::Class())) {
    // the base folder is a sample folder: return a pointer to it
    return (TQSampleFolder*)base;
  } else {
    // there is no base folder or it is no sample folder: return a NULL pointer
    return NULL;
  }
}

//__________________________________________________________________________________|___________

TQSampleFolder * TQSampleFolder::getRootSampleFolder() {
  // Returns the a pointer to the instance of TQSampleFolder which is the root of
  // this samplefolder hierarchy (not necessarily the root of the folder hierarchy).

  // get the base sample folder
  TQSampleFolder * base = getBaseSampleFolder();

  if (base) {
    // there is a base sample folder: return the root sample folder of it
    return base->getRootSampleFolder();
  } else {
    // there is no base sample folder: this is the root
    return this;
  }
}


//__________________________________________________________________________________|___________

TList * TQSampleFolder::getListOfSamples(const TString& path) {
  // Returns a list (instance of TList) of samples (instances of TQSample) withing
  // this sample folder matching the path pattern <path>. If no matching sample
  // can be found a NULL pointer is returned.

  // get the list of samples folders matching <path> (this also includes
  // the sample we are interested in since TQSampleFolder <-- TQSample)
  TQSampleDataReader rd(this);
  TList * samples = rd.getListOfSampleFolders(path,TQSample::Class());
 
  // return the list of samples
  return samples;
}

//__________________________________________________________________________________|___________

int TQSampleFolder::printListOfSamples(const TString& path) {
  // prints a list of samples (instances of TQSample) withing
  // this sample folder matching the path pattern <path>. 
  // the number of matched samples is returned
  TList* samples = this->getListOfSamples(path);
  if(!samples) return -1;
  int retval = samples->GetEntries();
  TQSampleIterator itr(samples,true);
  while(itr.hasNext()){
    TQSample* s = itr.readNext();
    if(!s) continue;
    std::cout << s->getName() << "\t" << s->getPath() << std::endl;
  }
  return retval;
}

//__________________________________________________________________________________|___________

TQSampleFolder * TQSampleFolder::getSampleFolder(TString path_, TQSampleFolder * template_, int * nSampleFolders_) {
  // legacy wrapper for the TClass variant
  return this->getSampleFolder(path_,template_ ? template_->Class() : NULL, nSampleFolders_);
}

//__________________________________________________________________________________|___________

TQSampleFolder * TQSampleFolder::getSampleFolder(TString path_, TClass * tclass, int * nSampleFolders_) {
  // Returns the sample folder (instance of TQSampleFolder) that matches the path
  // pattern <path_> and a NULL pointer in case no match can be found. If more than
  // one sample folder matches <path_> (because wildcards are used) the first match
  // is returned. Additionally, a new instance of TQSampleFolder is created as
  // requested if it does not already exist and a "+" has been appended to <path_>.
  // The path <path_> may be built up from any number of path levels in either case
  // and an arbitrary number of nested sample folders may be created by appending
  // one "+" to the full path.
  //
  // Examples:
  //
  // - getSampleFolder("subfolder") returns a pointer to the instance of TQSampleFolder
  // named "subfolder" within this instance of TQSampleFolder if it does exist.
  // - getSampleFolder("subfolder+") returns a pointer to the instance of TQSampleFolder
  // named "subfolder" within this instance of TQSampleFolder and does create it if
  // it does not exist
  // - getSampleFolder("subfolder/fol2+") returns a pointer to the instance of
  // TQSampleFolder named "fol2" within "subfolder" (in turn within this instance of
  // TQSampleFolder) and does create it if it does not exist
  //
  // [Please note: the additional parameters <template_> and <nSampleFolders_> are
  // for internal use only and should not be used by the user.]

  if (tclass) {
    return (TQSampleFolder*)getFolder(path_, tclass, nSampleFolders_);
  } else {
    /* get the sample folder */
    TQSampleFolder * sampleFolder = (TQSampleFolder*)getFolder(path_, TQSampleFolder::Class(), nSampleFolders_);
    /* return the sample folder */
    return sampleFolder;
  }
}


//__________________________________________________________________________________|___________

int TQSampleFolder::getNSampleFolders(bool recursive) {
  // Returns the number of sample folders (instances of TQSampleFolder) within this
  // instance of TQSampleFolder. If <recursive> == true not only sample folders
  // within this instance but also recursively within sub sample folder will be
  // counted.

  // return the number of sample folders in this sample folder
  return getNElements(recursive, TQSampleFolder::Class());
}


//__________________________________________________________________________________|___________

int TQSampleFolder::getNSamples(bool recursive) {
  // Returns the number of samples (instances of TQSample) within this instance of
  // TQSampleFolder. If <recursive> == true not only samples within this instance
  // but also recursively within sub sample folder will be counted.

  // return the number of samples in this sample folder
  return getNElements(recursive, TQSample::Class());
}


//__________________________________________________________________________________|___________

bool TQSampleFolder::hasHistogram(TString path, TString name, TString options) {
  // Returns true if a histogram <name> in <path> can be retrieved with options
  // <options> using TQSampleFolder::getHistogram(...).
  // [Please note: this is a wrapper to TQSampleDataReader::hasHistogram(...).]

  // the reader to retrieve histograms
  TQSampleDataReader rd(this);

  // return true if the histogram exists
  return rd.hasHistogram(path, name, options);
}


//__________________________________________________________________________________|___________

bool TQSampleFolder::hasCounter(TString path, TString name, TString options) {
  // Returns true if a counter <name> in <path> can be retrieved with options
  // <options> using TQSampleFolder::getCounter(...).
  // [Please note: this is a wrapper to TQSampleDataReader::hasCounter(...).]

  // the reader to retrieve counter
  TQSampleDataReader rd(this);

  // return true if the counter exists
  return rd.hasCounter(path, name, options);
}


//__________________________________________________________________________________|___________

bool TQSampleFolder::renameLocalObject(TString category,
                                       TClass * classType, TString oldName, TString newName) {
  // Renames a local object within subfolder <category> and name <oldName> into
  // <newName> also allowing to move the object to a different subfolder and returns
  // true in case of success and false otherwise. This method is used by
  // TQSampleFolder::renameLocalHistogram(...) and TQSampleFolder::
  // renameLocalCounter(...) to rename local histograms and counters.

  // get the local object
  TObject * obj = getObject(TQFolder::concatPaths(category, oldName));
  if (obj && obj->InheritsFrom(classType)) {

    // make sure the new object name is valid
    if (!TQFolder::isValidPath(newName, false, false)) {
      return false;
    }

    // remove the old local object (don't delete it)
    removeObject(TQFolder::concatPaths(category, oldName));

    // delete a potentially existing local object with newName
    deleteLocalObject(category, newName);

    // store the old object under new name
    TQFolder * folder = getFolder(TQFolder::concatPaths(category,
                                                        TQFolder::getPathWithoutTail(newName)) + "+");
    if (folder) {
      folder->addObject(obj, "::" + TQFolder::getPathTail(newName));
      return true;
    } else {
      // unknown problem (we failed to obtain the new folder to put the histogram in
      delete obj;
      return false;
    }
  } else {
    // couldn't find the object to rename or it is of wrong class type
    return false;
  }
}


//__________________________________________________________________________________|___________

bool TQSampleFolder::renameLocalHistogram(TString oldName, TString newName) {
  // Renames the local histogram (instance of TH1 wihtin ".histograms" subfolder)
  // with name <oldName> to <newName> also allowing to move the histogram to a new
  // subfolder and returns true in case of success and false otherwise. In case a
  // histogram is moved to a new subfolder it is created unless it already exist.
  // This method does not affect histograms within sub sample folders.
  //
  // Examples:
  //
  // - renameLocalHistogram("CutMETRel/MT", "CutMETRel/MT_2") renames the histogram
  // "MT" in "CutMETRel" to "MT_2".
  // - renameLocalHistogram("CutMETRel/MT", "CutMETRel_2/MT") moves the histogram
  // "MT" in "CutMETRel" to "CutMETRel_2".

  // rename local histogram as instance of TH1 within ".histograms"
  return renameLocalObject(".histograms", TH1::Class(), oldName, newName);
}


//__________________________________________________________________________________|___________

bool TQSampleFolder::renameLocalCounter(TString oldName, TString newName) {
  // Renames the local counter (instance of TQCounter wihtin ".cutflow" subfolder)
  // with name <oldName> to <newName> also allowing to move the counter to a new
  // subfolder and returns true in case of success and false otherwise. In case a
  // counter is moved to a new subfolder it is created unless it already exist.
  // This method does not affect counters within sub sample folders.
  //
  // Examples:
  //
  // - renameLocalCounter("CutMETRel", "CutMETRel_2") renames the counter
  // "CutMETRel" to "CutMETRel_2".
  // - renameLocalCounter("CutMETRel", "test/CutMETRel") moves the counter
  // "CutMETRel" to "test".

  // rename local counter as instance of TQCounter within ".cutflow"
  return renameLocalObject(".cutflow", TQCounter::Class(), oldName, newName);
}


//__________________________________________________________________________________|___________

int TQSampleFolder::renameHistogram(TString oldName, TString newName) {
  // Renames the histogram with name <oldName> into <newName> and returns the number
  // of contributing histograms that have been renamed. This method also allows
  // to move histograms to new subfolders without the need to take additional
  // action. Examples:
  //
  // - renameHistogram("Cut_0jet/MT", "Cut_0jet/MT_2") renames the histogram
  // "Cut_0jet/MT" into "Cut_0jet/MT_2"
  // - renameHistogram("Cut_0jet/MT", "Cut_0jet_test/MT") moves the histogram "MT"
  // from "Cut_0jet" to "Cut_0jet_test" (with the latter being created if it does
  // not exists prior to the operation
  // - renameHistogram("Cut_0jet/MT", "1/2/3/MT") moves the histogram "MT"
  // from "Cut_0jet" to "1/2/3" (with the latter being created if it does
  // not exists prior to the operation

  // the number of renamed histogram contributions
  int nHistos = 0;

  // the list to keep track of contributing histograms
  TList * sfList = new TList();
  sfList->SetOwner(false);

  // this while loop is necesarry since renaming a histogram might uncover
  // histograms with the old name stored 'below' the one that has been renamed
  TH1 * histo = NULL;
  while ((histo = getHistogram(".", oldName, "", sfList))) {
    // we are not really interested in the histogram, only the list of contributions
    delete histo;
    // iterate over list of contributing sample folders
    TQIterator itr(sfList);
    while (itr.hasNext()) {
      // for each contributing sample folder rename the local histogram
      if (((TQSampleFolder*)itr.readNext())->renameLocalHistogram(oldName, newName)) {
        nHistos++;
      }
    }
    // next iteration should not use sample folders of this iteration
    sfList->Clear();
  }
  delete sfList;

  // return the number of contributing histograms renamed
  return nHistos;
}


//__________________________________________________________________________________|___________

int TQSampleFolder::renameCounter(TString oldName, TString newName) {
  // Renames the counter with name <oldName> into <newName> and returns the number
  // of contributing counter that have been renamed. This method also allows
  // to move counter to new subfolders without the need to take additional
  // action. Examples:
  //
  // - renameCounter("Cut_0jet", "Cut_0jet_2") renames the counter "Cut_0jet" into
  // "Cut_0jet_2"
  // - renameCounter("Cut_0jet", "1/2/3/Cut_0jet") moves the counter "Cut_0jet"
  // to "1/2/3" (with the latter being created if it does not exists prior to the
  // operation

  // the number of renamed counter contributions
  int nCounter = 0;

  // the list to keep track of contributing counter
  TList * sfList = new TList();
  sfList->SetOwner(false);

  // this while loop is necesarry since renaming a counter might uncover
  // counter with the old name stored 'below' the one that has been renamed
  TQCounter * counter = NULL;
  while ((counter = getCounter(".", oldName, "", sfList))) {
    // we are not really interested in the counter, only the list of contributions
    delete counter;
    // iterate over list of contributing sample folders
    TQIterator itr(sfList);
    while (itr.hasNext()) {
      // for each contributing sample folder rename the local counter
      if (((TQSampleFolder*)itr.readNext())->renameLocalCounter(oldName, newName)) {
        nCounter++;
      }
    }
    // next iteration should not use sample folders of this iteration
    sfList->Clear();
  }
  delete sfList;

  // return the number of contributing counter renamed
  return nCounter;
}


//__________________________________________________________________________________|___________

bool TQSampleFolder::deleteLocalObject(TString category, TString name) {

  /* delete the local object */
  return deleteObject(TQFolder::concatPaths(category, name) + "-") > 0;
}


//__________________________________________________________________________________|___________

bool TQSampleFolder::deleteLocalHistogram(TString name) {

  /* delete the local histogram */
  return deleteLocalObject(".histograms", name);
}


//__________________________________________________________________________________|___________

bool TQSampleFolder::deleteLocalCounter(TString name) {

  /* delete the local counter */
  return deleteLocalObject(".cutflow", name);
}


//__________________________________________________________________________________|___________

int TQSampleFolder::deleteHistogram(TString name) {
  // Deletes all contributing histograms to histogram <name> within this instance of
  // TQSampleFolder and all sub sample folders and returns the number of contributing
  // histograms that have been deleted.

  /* remove all contributions */
  int nHistos = 0;
  TH1 * histo = 0;
  TList * sfList = new TList();
  while ((histo = getHistogram(".", name, "", sfList))) {
    delete histo;
    /* iterate over list of contributions */
    TIterator * itr = sfList->MakeIterator();
    TObject * obj;
    while ((obj = itr->Next())) {
      if (((TQSampleFolder*)obj)->deleteLocalHistogram(name))
        nHistos++;
    }
    sfList->Clear();
    delete itr;
  }
  delete sfList;

  // return the number of contributing histograms deleted
  return nHistos;
}


//__________________________________________________________________________________|___________

int TQSampleFolder::deleteSingleCounter(TString name) {
  // Deletes all contributing counter to counter <name> within this instance of
  // TQSampleFolder and all sub sample folders and returns the number of contributing
  // counter that have been deleted.

  /* remove all contributions */
  int nCounter = 0;
  TQCounter * counter = 0;
  TList * sfList = new TList();
  while ((counter = getCounter(".", name, "", sfList))) {
    delete counter;
    /* iterate over list of contributions */
    TIterator * itr = sfList->MakeIterator();
    TObject * obj;
    while ((obj = itr->Next())) {
      if (((TQSampleFolder*)obj)->deleteLocalCounter(name))
        nCounter++;
    }
    sfList->Clear();
    delete itr;
  }
  delete sfList;

  // return the number of contributing counter deleted
  return nCounter;
}


//__________________________________________________________________________________|___________

int TQSampleFolder::deleteHistograms(TString filter) {
  // Deletes all contributing histograms to histograms matching the string pattern
  // <filter> within this instance of TQSampleFolder and all sub sample folders and
  // returns the number of histograms that have been deleted. The string pattern
  // is matched using TQStringUtils::matchesFilter(...) and allows the use of wildcards
  // "*" and "?", a negation by prepending "!", and an OR of multiple comma-separated
  // string patterns.

  // the number of histograms that have been deleted
  int nHistos = 0;

  // iterate over the list of available histograms
  TQIterator itr(getListOfHistogramNames(), true);
  while (itr.hasNext()) {
    // the name of the current histogram
    TString name = itr.readNext()->GetName();
    // delete histogram if it matches the filter
    if (TQStringUtils::matchesFilter(name, filter, ",", true)) {
      deleteHistogram(name);
      nHistos++;
    }
  }

  // return the number of histograms that have been deleted
  return nHistos;
}


//__________________________________________________________________________________|___________

int TQSampleFolder::deleteCounter(TString filter) {
  // Deletes all contributing counter to counter matching the string pattern <filter>
  // within this instance of TQSampleFolder and all sub sample folders and returns
  // the number of counter that have been deleted. The string pattern is matched
  // using TQStringUtils::matchesFilter(...) and allows the use of wildcards "*" and
  // "?", a negation by prepending "!", and an OR of multiple comma-separated string
  // patterns.

  // the number of counter that have been deleted
  int nCounter = 0;

  // iterate over the list of available counter
  TQIterator itr(getListOfCounterNames(), true);
  while (itr.hasNext()) {
    // the name of the current histogram
    TString name = itr.readNext()->GetName();
    // delete counter if it matches the filter
    if (TQStringUtils::matchesFilter(name, filter, ",", true)) {
      deleteSingleCounter(name);
      nCounter++;
    }
  }

  // return the number of counter that have benn deleted
  return nCounter;
}


//__________________________________________________________________________________|___________

int TQSampleFolder::copyHistograms(TString sourceFilter, TString appendix, TString options) {
 
  // the number of histograms that have been copied
  int nCopied = 0;
 
  // loop over all histograms matching <sourceFilter>
  TList * histograms = this->getListOfHistogramNames();
  TQIterator itr(histograms, sourceFilter, true);
  while (itr.hasNext()) {
    TString name = itr.readNext()->GetName();
    TString newName = name + appendix;
    // make sure histogram does not yet exist
    if (histograms->FindObject(newName.Data())) {
      continue;
    }
    // make a copy
    if (copyHistogram(name, newName, options) > 0) {
      nCopied++;
    }
  }
 
  return nCopied;
}


//__________________________________________________________________________________|___________

int TQSampleFolder::copyHistogram(TString source, TString destination, TString options) {
  // Retrieves the histogram <source> applying options <options>, stores the resulting
  // histgram as <destination>, and returns the number of contributions to the new
  // histogram.
  //
  // Examples:
  // - copyHistogram("Cut_0jet/MT + Cut_1jet/MT", "Cut_Sum01jet/MT") creates a new
  // histogram "Cut_Sum01jet/MT" from the sum of "Cut_0jet/MT" and "Cut_1jet/MT"
  // - copyHistogram("Cut/MT_vs_Mll", "Cut/MT", "projX = true") creates a new
  // histogram "Cut/MT" from the projection of "Cut/MT_vs_Mll" onto its X axis

  // the number of new histogram contributions
  int nHistos = 0;

  // get the source histogram and the list of contributing sample folders
  TList * sfList = new TList();
  delete getHistogram(".", source, options, sfList);

  // the new name of the histogram is the last path token of <destination>
  TString newName = TQFolder::getPathTail(destination);

  // a list to keep track of sample folders that have already been handled
  TList * done = new TList();

  TQIterator itr(sfList, true);
  while (itr.hasNext()) {
    // the next object (sample folder) to handle
    TQSampleFolder * sf = (TQSampleFolder*)itr.readNext();

    // skip sample folders that have already been handled
    if (done->FindObject(sf)) {
      continue;
    }
    done->Add(sf);

    // get the individual source histogram contribution
    TH1 * h_source = sf->getHistogram(".", source, options);
    if (!h_source) {
      // we failed for some reason to obtain it
      continue;
    }

    // store as new histogram
    if (sf->addObject(h_source, TQFolder::concatPaths(".histograms", destination)
                      + "+::" + newName)) {
      // successfully added the new histogram: increment <nHistos>
      nHistos++;
    } else {
      // failed to add new histogram: delete it again
      delete h_source;
    }
  }
  delete done;

  // return the number of new histogram contributions
  return nHistos;
}


//__________________________________________________________________________________|___________

int TQSampleFolder::copyHistogramToCounter(TString source, TString destination) {

  /* default counter name is histogram name */
  if (destination.IsNull()) {
    destination = source;
  }

  /* the number of new counter */
  int nCounter = 0;

  /* get the source histogram and the contributions */
  TList * sfList = new TList();
  TH1 * histo = getHistogram(".", source, "", sfList);
  if (histo) {
    delete histo;
  }

  TString newName = TQFolder::getPathTail(destination);
  TList * done = new TList();

  TQIterator itr(sfList, true);
  while (itr.hasNext()) {
    TObject * obj = itr.readNext();
    /* skip sample folders already handled */
    if (done->FindObject(obj))
      continue;
    done->Add(obj);
    TQSampleFolder * sf = (TQSampleFolder*)obj;

    /* === create counter from histogram === */

    /* get the source histogram */
    TH1 * h_source = sf->getHistogram(".", source);
    if (!h_source) {
      continue;
    }

    /* create counter */
    TQCounter * cnt = new TQCounter("counter");
    double error = 0.;
    cnt->setCounter(TQHistogramUtils::getIntegralAndError(h_source, error));
    cnt->setErrorSquared(TMath::Power(error, 2.));
    delete h_source;

    /* store the new counter */
    if (sf->addObject(cnt, TQFolder::concatPaths(".cutflow", destination)
                      + "+::" + newName)) {
      nCounter++;
    } else {
      delete cnt;
    }
  }

  delete done;

  return nCounter;
}


//__________________________________________________________________________________|___________

TH1 * TQSampleFolder::getHistogram(TString path, TString name,
                                   TQTaggable * options, TList * sfList) {
  // Returns a histogram (pointer to an instance of TH1) obtained from summing
  // contributions matching <name> from sample folders matching <path>. Additional
  // options may be passed via <options> and a list of contribution sample folders
  // is returned if <sfList> is valid pointer to an instance of TList.
  //
  // [Please note: this is a wrapper to TQSampleDataReader::getHistogram(...).
  // Please check the corresponding documentation for details.]

  // the reader to retrieve histograms
  TQSampleDataReader rd(this);

  // get the histogram and return it
  return rd.getHistogram(path, name, options, sfList);
}


//__________________________________________________________________________________|___________

TH1 * TQSampleFolder::getHistogram(TString path, TString name,
                                   TString options, TList * sfList) {
  // Returns a histogram (pointer to an instance of TH1) obtained from summing
  // contributions matching <name> from sample folders matching <path>. Additional
  // options may be passed via <options> and a list of contributing sample folders
  // is returned if <sfList> is a valid pointer to an instance of TList.
  //
  // [Please note: this is a wrapper to TQSampleDataReader::getHistogram(...).
  // Please check the corresponding documentation for details.]

  // the reader to retrieve histograms
  TQSampleDataReader rd(this);

  // get the histogram and return it
  return rd.getHistogram(path, name, options, sfList);
}


//__________________________________________________________________________________|___________

TQCounter * TQSampleFolder::getCounter(TString path, TString name,
                                       TString options, TList * sfList) {
  // Returns a counter (pointer to an instance of TQCounter) obtained from summing
  // contributions matching <name> from sample folders matching <path>. Additional
  // options may be passed via <options> and a list of contributing sample folders
  // is returned if <sfList> is a valid pointer to an instance of TList.
  //
  // [Please note: this is a wrapper to TQSampleDataReader::getCounter(...).
  // Please check the corresponding documentation for details.]

  // the reader to retrieve counter
  TQSampleDataReader rd(this);

  // get the counter and return it
  return rd.getCounter(path, name, options, sfList);
}


//__________________________________________________________________________________|___________

TList * TQSampleFolder::getListOfHistogramNames(const TString& path, TList * sfList) {
  // Returns a list (pointer to instance of TList) of histogram names (as instances
  // of TObjString) present within the sample folder referred to by <path> and a
  // NULL pointer if no histogram could be found. Additionally, a list of contributing
  // sample folders is returned if <sfList> is a valid pointer to an instance of TList.

  // the reader to retrieve the list of histogram names
  TQSampleDataReader rd(this);

  // return the list of histogram names
  return rd.getListOfHistogramNames(path, sfList);
}


//__________________________________________________________________________________|___________

void TQSampleFolder::printListOfHistograms(const TString& options) {
  // Prints a list of histogram names present within the sample folder

  // the reader to print the list of histogram names
  TQSampleDataReader rd(this);

  // print the list of histogram names
  rd.printListOfHistograms(options);
}


//__________________________________________________________________________________|___________

TList * TQSampleFolder::getListOfCounterNames(const TString& path, TList * sfList) {
  // Returns a list (pointer to instance of TList) of counter names (as instances
  // of TObjString) present within the sample folder referred to by <path> and a
  // NULL pointer if no counter could be found. Additionally, a list of contributing
  // sample folders is returned if <sfList> is a valid pointer to an instance of TList.

  // the reader to retrieve the list of counter names
  TQSampleDataReader rd(this);

  // return the list of counter names
  return rd.getListOfCounterNames(path, sfList);
}


//__________________________________________________________________________________|___________

void TQSampleFolder::printListOfCounters(const TString& options) {
  // Prints a list of counters present within the sample folder

  // the reader to print the list of histogram names
  TQSampleDataReader rd(this);

  // print the list of histogram names
  rd.printListOfCounters(options);
}

//__________________________________________________________________________________|___________

bool TQSampleFolder::validateAllCounter(TString path1, TString path2, TString options) {
  // Validates all available counter in sample folders referred to by <path1> and
  // <path2> by comparing instances with matching names in these two sample folders,
  // prints out the result, and returns true if all counter agree or false otherwise.
  // The default output is a list of all counter names (present in any of the two
  // sample folders) with the second column indicating the result of the comparison
  // of the corresponding counter retrieved from the two input sample folders:
  //
  // (1) = (2) [green] counter agree within chosen accuracy (weighted number,
  // uncertainty, and raw number)
  // (1) ~ (2) [yellow] counter agree only in raw number
  // (1) > (2) [red] counter retrieved from sample folder 1 has a larger
  // weighted number than the one retrieved from sample folder 2
  // (1) < (2) [red] similarly
  // (1) - [red] counter does not exist in sample folder 2
  // - (2) [red] similarly
  //
  // Additional options may be passed via <options>. The following options are
  // available:
  //
  // - "d" additionally trace mismatch of counter that do not agree.
  // - "m" only list counter that do not match
  // - "e" only list counter that exists in both sample folders
  // - "c" print counter states in additional columns
  // - "r" show ratio of weighted numbers of the two counter
  // - "a[<accuracy>]" set relative accuracy used for comparing counter
  // to <accuracy> (default is 1E-7)
  // - "f[<filter>]" only list counter whose name match <filter> (allows use
  // of wildcards "*" and "?" in the usual "ls-like" way)

  // get the samples folders to compare
  TQSampleFolder * sampleFolder1 = this->getSampleFolder(path1);
  TQSampleFolder * sampleFolder2 = this->getSampleFolder(path2);

  // validate all counter
  if (sampleFolder1 && sampleFolder2) {
    return sampleFolder1->validateAllCounter(sampleFolder2, options);
  } else {
    if (!sampleFolder1) {
      // error message since sample folder 1 does not exist
      std::cout << TQStringUtils::makeBoldRed(TString::Format(
                                                              "TQSampleFolder::validateAllCounter(...): Failed to find "
                                                              "sample folder '%s'", path1.Data())).Data() << std::endl;
    } else if (!sampleFolder2) {
      // error message since sample folder 2 does not exist
      std::cout << TQStringUtils::makeBoldRed(TString::Format(
                                                              "TQSampleFolder::validateAllCounter(...): Failed to find "
                                                              "sample folder '%s'", path2.Data())).Data() << std::endl;
    }
    return false;
  }
}


//__________________________________________________________________________________|___________

bool TQSampleFolder::validateAllCounter(TQSampleFolder * sampleFolder, TString options) {

  /* column widths */
  const int cWidthName = 50;
  const int cWidthComp = 12;
  const int cWidthCnt = 40;
  const int cWidthRatio = 10;

  /* stop if input sample folder is invalid */
  if (!sampleFolder)
    return false;

  /* ===== read the options ===== */

  TString flags;
  TString localOptions = options;
  TString accuracyOption;
  TString filter;

  /* read flags */
  bool stop = false;
  while (!stop) {
    /* read flags without parameter */
    if (TQStringUtils::readToken(localOptions, flags, "dmecr") > 0)
      continue;

    /* read accuracy option */
    if (TQStringUtils::readToken(localOptions, flags, "a", 1) > 0) {

      /* don't allow multiple accurary definitions */
      if (accuracyOption.Length() > 0) {
        std::cout << "TQSampleFolder::validateAllCounter(...): cannot define "
          "more than one accuracy using 'a'\n";
        return false;
      }

      /* expect accuracy definition after 'a' option */
      if (!(TQStringUtils::readBlock(localOptions, accuracyOption, "[]") > 0
            && accuracyOption.Length() > 0)) {
        std::cout << "TQSampleFolder::validateAllCounter(...): accuracy "
          "definition expected after option 'a'\n";
        return false;
      }

      if (!TQStringUtils::isNumber(accuracyOption)) {
        std::cout << "TQSampleFolder::validateAllCounter(...): expect "
          "number after option 'a'\n";
        return false;
      }

      continue;
    }

    /* read object filter flag "f" and filter definition */
    if (TQStringUtils::readToken(localOptions, flags, "f", 1) > 0) {

      /* don't allow multiple filters */
      if (filter.Length() > 0) {
        std::cout << "TQSampleFolder::validateAllCounter(...): cannot define "
          "more than one filter using 'f'\n";
        return false;
      }

      /* expect filter definition after 'f' option */
      if (!(TQStringUtils::readBlock(localOptions, filter, "[]") > 0
            && filter.Length() > 0)) {
        std::cout << "TQSampleFolder::validateAllCounter(...): filter "
          "definition expected after option 'f'\n";
        return false;
      }

      continue;
    }

    /* no valid tokens left to parse */
    stop = true;
  }

  /* unexpected options left? */
  if (localOptions.Length() > 0) {
    std::cout << TString::Format("TQSampleFolder::validateAllCounter(...):"
                                 " unknown option '%c'\n", localOptions[0]);
    return false;
  }

  /* parse the flags */
  bool flagDetails = flags.Contains("d");
  bool flagMismatch = flags.Contains("m");
  bool flagExisting = flags.Contains("e");
  bool flagShowCounter = flags.Contains("c");
  bool flagAccuracy = flags.Contains("a");
  bool flagShowRatio = flags.Contains("r");

  double accuracy = 1E-7;
  if (!accuracyOption.IsNull())
    accuracy = accuracyOption.Atof();


  /* ===== make the actual comparison ===== */

  /* true if counter in sample folders are equal */
  bool equal = true;

  /* print headline */
  TString line;
  line.Append(TQStringUtils::fixedWidth("Counter / sample folder", cWidthName, true));
  line.Append(TQStringUtils::fixedWidth("Comparison", cWidthComp));
  if (flagShowCounter) {
    line.Append(TQStringUtils::fixedWidth("Counter 1", cWidthCnt));
    line.Append(TQStringUtils::fixedWidth("Counter 2", cWidthCnt));
  }
  if (flagShowRatio) {
    line.Append(TQStringUtils::fixedWidth("(1)/(2)", cWidthRatio));
  }
  std::cout << TQStringUtils::makeBoldWhite(line).Data() << std::endl;
  std::cout << TQStringUtils::makeBoldWhite(TQStringUtils::repeat("=", line.Length())).Data() << std::endl;

  /* get the list of counter names */
  TList * l1 = this->getListOfCounterNames();
  TList * l2 = sampleFolder->getListOfCounterNames();

  /* the number of entries in the list */
  int n1 = 0;
  int n2 = 0;

  if (l1) {
    n1 = l1->GetEntries();
    l1->Sort();
  }
  if (l2) {
    n2 = l2->GetEntries();
    l2->Sort();
  }

  /* the number of entries in the list and the list indices */
  int i1 = 0;
  int i2 = 0;

  /* process counter in both sample folders */
  while (i1 < n1 || i2 < n2) {

    /* get the two heading sample folders */
    TObjString * obj1 = (i1 < n1) ? (TObjString*)l1->At(i1) : 0;
    TObjString * obj2 = (i2 < n2) ? (TObjString*)l2->At(i2) : 0;

    int compare = -1;
    if (obj1 && obj2) {
      compare = obj1->Compare(obj2);
      i1 += (compare <= 0) ? 1 : 0;
      i2 += (compare >= 0) ? 1 : 0;
    } else if (obj1) {
      compare = -1;
      i1++;
    } else if (obj2) {
      compare = 1;
      i2++;
    } else {
      /* something went totally wrong */
      return false;
    }

    TString name;
    TString comparison;
    bool mismatch = false;

    /* the two counter */
    TQCounter * c1 = 0;
    TQCounter * c2 = 0;

    /* matching counter: compare them */
    if (compare == 0) {
      name = obj1->GetName();
      if (!filter.IsNull() && !TQStringUtils::matchesFilter(
                                                            name, filter, ",", true))
        continue;

      /* get the two counter */
      c1 = this->getCounter(".", name);
      c2 = sampleFolder->getCounter(".", name);

      /* compare counter */
      comparison = TQCounter::getComparison(c1, c2, true, accuracy);
      if (!(c1 && c2 && c1->isEqualTo(c2, accuracy)))
        mismatch = true;
    } else {
      /* counter don't match */
      mismatch = true;
      if (compare < 0) {
        name = obj1->GetName();
        if (!filter.IsNull() && !TQStringUtils::matchesFilter(
                                                              name, filter, ",", true))
          continue;
        comparison = TQStringUtils::makeBoldRed("(1) - ");
      } else {
        name = obj2->GetName();
        if (!filter.IsNull() && !TQStringUtils::matchesFilter(
                                                              name, filter, ",", true))
          continue;
        comparison = TQStringUtils::makeBoldRed(" - (2)");
      }
    }

    if ((mismatch || !flagMismatch) && !(compare != 0 && flagExisting)) {
      line.Clear();
      line.Append(TQStringUtils::fixedWidth(name, cWidthName, true));
      line.Append(TQStringUtils::fixedWidth(comparison, cWidthComp));
      if (flagShowCounter) {
        if (c1)
          line.Append(TQStringUtils::fixedWidth(c1->getAsString(), cWidthCnt));
        else
          line.Append(TQStringUtils::fixedWidth("--", cWidthCnt));
        if (c2)
          line.Append(TQStringUtils::fixedWidth(c2->getAsString(), cWidthCnt));
        else
          line.Append(TQStringUtils::fixedWidth("--", cWidthCnt));
      }
      if (flagShowRatio) {
        TString ratio = "--";
        if (c1 && c2)
          ratio = TString::Format("%.3f", c1->getCounter() / c2->getCounter());
        line.Append(TQStringUtils::fixedWidth(ratio, cWidthRatio));
      }
      std::cout << line.Data() << std::endl;
    }

    /* delete counter */
    delete c1;
    delete c2;

    equal &= !mismatch;
    if (mismatch && compare == 0 && flagDetails) {
      TString myOptions;
      if (flagShowCounter)
        myOptions.Append("c");
      if (flagShowRatio)
        myOptions.Append("r");
      if (flagAccuracy)
        myOptions.Append(TString::Format("a[%f]", accuracy));
      validateCounter(sampleFolder, name, myOptions, -1);
    }
  }

  /* delete list of counter names */
  if (l1)
    delete l1;
  if (l2)
    delete l2;

  /* return true if counter in sample folders are equal */
  return equal;
}


//__________________________________________________________________________________|___________

bool TQSampleFolder::validateCounter(TString path1, TString path2,
                                     TString counterName, TString options) {

  /* get the samples folders to compare */
  TQSampleFolder * sampleFolder1 = this->getSampleFolder(path1);
  TQSampleFolder * sampleFolder2 = this->getSampleFolder(path2);

  /* validate counter */
  if (sampleFolder1 && sampleFolder2)
    return sampleFolder1->validateCounter(sampleFolder2, counterName, options);
  else
    return false;
}


//__________________________________________________________________________________|___________

bool TQSampleFolder::validateCounter(TQSampleFolder * sampleFolder,
                                     TString counterName, TString options, int indent) {

  /* column widths */
  const int cWidthName = 50;
  const int cWidthComp = 12;
  const int cWidthCnt = 40;
  const int cWidthRatio = 10;

  /* stop if input sample folder is invalid */
  if (!sampleFolder)
    return false;

  /* ===== read the options ===== */

  TString flags;
  TString localOptions = options;
  TString accuracyOption;

  /* read flags */
  bool stop = false;
  while (!stop) {
    /* read flags without parameter */
    if (TQStringUtils::readToken(localOptions, flags, "cr") > 0)
      continue;

    /* read accuracy option */
    if (TQStringUtils::readToken(localOptions, flags, "a", 1) > 0) {

      /* don't allow multiple accurary definitions */
      if (accuracyOption.Length() > 0) {
        std::cout << "TQSampleFolder::validateCounter(...): cannot define "
          "more than one accuracy using 'a'\n";
        return false;
      }

      /* expect accuracy definition after 'a' option */
      if (!(TQStringUtils::readBlock(localOptions, accuracyOption, "[]") > 0
            && accuracyOption.Length() > 0)) {
        std::cout << "TQSampleFolder::validateCounter(...): accuracy "
          "definition expected after option 'a'\n";
        return false;
      }

      if (!TQStringUtils::isNumber(accuracyOption)) {
        std::cout << "TQSampleFolder::validateCounter(...): expect "
          "number after option 'a'\n";
        return false;
      }

      continue;
    }

    /* no valid tokens left to parse */
    stop = true;
  }

  /* unexpected options left? */
  if (localOptions.Length() > 0) {
    std::cout << TString::Format("TQSampleFolder::validateCounter(...):"
                                 " unknown option '%c'\n", localOptions[0]);
    return false;
  }

  /* parse the flags */
  bool flagShowCounter = flags.Contains("c");
  bool flagShowRatio = flags.Contains("r");

  double accuracy = 1E-7;
  if (!accuracyOption.IsNull())
    accuracy = accuracyOption.Atof();


  /* ===== make the actual comparison ===== */

  /* true if counter in sample folders are equal */
  bool equal = false;

  TString line;
  if (indent == 0) {
    /* print headline */
    line.Append(TQStringUtils::fixedWidth("Sample folder", cWidthName, true));
    line.Append(TQStringUtils::fixedWidth("Comparison", cWidthComp));
    if (flagShowCounter) {
      line.Append(TQStringUtils::fixedWidth("Counter 1", cWidthCnt));
      line.Append(TQStringUtils::fixedWidth("Counter 2", cWidthCnt));
    }
    if (flagShowRatio) {
      line.Append(TQStringUtils::fixedWidth("(1)/(2)", cWidthRatio));
    }
    std::cout << TQStringUtils::makeBoldWhite(line) << std::endl;
    std::cout << TQStringUtils::makeBoldWhite(TQStringUtils::repeat("=", TQStringUtils::getWidth(line))) << std::endl;
  }

  /* get the two counter */
  TQCounter * c1 = this->getCounter(".", counterName);
  TQCounter * c2 = sampleFolder->getCounter(".", counterName);

  /* the columns to print */
  TString name;
  TString comparison;

  /* matching counter */
  if ((c1 && c2 && c1->isEqualTo(c2, accuracy)) || (!c1 && !c2))
    equal = true;

  if (indent == 0 || (indent >= 0 && !equal)) {
    name = TQStringUtils::repeatSpaces(indent * 2);
    comparison = TQCounter::getComparison(c1, c2, true);

    if (indent == 0)
      name.Append(TQStringUtils::makeBoldBlue(GetName()) + "/, " +
                  TQStringUtils::makeBoldBlue(sampleFolder->GetName()) + "/");
    else 
      name.Append(TQStringUtils::makeBoldBlue(sampleFolder->GetName()) + "/");

    line.Clear();
    line.Append(TQStringUtils::fixedWidth(name, cWidthName, true));
    line.Append(TQStringUtils::fixedWidth(comparison, cWidthComp));
    if (flagShowCounter) {
      if (c1)
        line.Append(TQStringUtils::fixedWidth(c1->getAsString(), cWidthCnt));
      else
        line.Append(TQStringUtils::fixedWidth("--", cWidthCnt));
      if (c2)
        line.Append(TQStringUtils::fixedWidth(c2->getAsString(), cWidthCnt));
      else
        line.Append(TQStringUtils::fixedWidth("--", cWidthCnt));
    }
    if (flagShowRatio) {
      TString ratio = "--";
      if (c1 && c2)
        ratio = TString::Format("%.3f", c1->getCounter() / c2->getCounter());
      line.Append(TQStringUtils::fixedWidth(ratio, cWidthRatio));
    }
    std::cout << line.Data() << std::endl;
  }

  /* we are done if the counter agree */
  if (equal) {
    /* delete counter */
    delete c1;
    delete c2;
    return true;
  }

  /* ===== ===== */

  if (indent < 0)
    indent = 0;

  /* get the list of sub sample folders */
  TList * l1 = this->getListOfSampleFolders("?");
  TList * l2 = sampleFolder->getListOfSampleFolders("?");

  /* the number of entries in the list */
  int n1 = 0;
  int n2 = 0;

  if (l1) {
    n1 = l1->GetEntries();
    l1->Sort();
  }
  if (l2) {
    n2 = l2->GetEntries();
    l2->Sort();
  }

  /* the number of entries in the list and the list indices */
  int i1 = 0;
  int i2 = 0;

  /* process sample folder in both lists */
  while (i1 < n1 || i2 < n2) {

    /* get the two heading sample folders */
    TQSampleFolder * sf1 = (i1 < n1) ? (TQSampleFolder*)l1->At(i1) : 0;
    TQSampleFolder * sf2 = (i2 < n2) ? (TQSampleFolder*)l2->At(i2) : 0;

    int compare = -1;
    if (sf1 && sf2) {
      compare = sf1->Compare(sf2);
      i1 += (compare <= 0) ? 1 : 0;
      i2 += (compare >= 0) ? 1 : 0;
    } else if (sf1) {
      compare = -1;
      i1++;
    } else if (sf2) {
      compare = 1;
      i2++;
    } else {
      /* something went totally wrong */
      return false;
    }

    if (compare == 0) {
      /* matching sample folder: compare sub counter */
      sf1->validateCounter(sf2, counterName, options, indent + 1);
    } else {
      /* sample folder don't match */

      TQCounter * cnt = 0;
      if (compare < 0) {
        cnt = sf1->getCounter(".", counterName);
        name = sf1->GetName();
        comparison = "(1) - ";
      } else if (compare > 0) {
        cnt = sf2->getCounter(".", counterName);
        name = sf2->GetName();
        comparison = " - (2)";
      }

      if (cnt) {
        line.Clear();
        line.Append(TQStringUtils::fixedWidth(TQStringUtils::repeatSpaces(indent * 2 + 2) +
                                              TQStringUtils::makeBoldBlue(name) + "/", cWidthName, true));
        line.Append(TQStringUtils::fixedWidth(TQStringUtils::makeBoldRed(comparison), cWidthComp));
        if (flagShowCounter) {
          if (compare < 0) {
            line.Append(TQStringUtils::fixedWidth(cnt->getAsString(), cWidthCnt));
            line.Append(TQStringUtils::fixedWidth("--", cWidthCnt));
          } else {
            line.Append(TQStringUtils::fixedWidth("--", cWidthCnt));
            line.Append(TQStringUtils::fixedWidth(cnt->getAsString(), cWidthCnt));
          }
        }
        std::cout << line.Data() << std::endl;
        delete cnt;
      }
    }
  }

  /* delete list of counter names */
  if (l1)
    delete l1;
  if (l2)
    delete l2;

  /* delete counter */
  delete c1;
  delete c2;

  /* there was a dismessage */
  return false;
}


//__________________________________________________________________________________|___________

int TQSampleFolder::setScaleFactor(const char* name, double scaleFactor, double uncertainty) {
  // Set a scale factor (better: normalization factor) and its uncertainty.
  // The 'path' is the name of the corresponding scale factor (typically name of the cut)
  // and may be prefixed with "schemeName:" to use a the scale factor scheme 'schemeName'. 
  return this->setScaleFactor((TString)name,name,scaleFactor,uncertainty);
}

//__________________________________________________________________________________|___________

int TQSampleFolder::setScaleFactor(const TString& name, double scaleFactor, double uncertainty) {
  // Set a scale factor (better: normalization factor) and its uncertainty.
  // The 'path' is the name of the corresponding scale factor (typically name of the cut)
  // and may be prefixed with "schemeName:" to use a the scale factor scheme 'schemeName'. 
  return this->setScaleFactor(name,name,scaleFactor,uncertainty);
}

//__________________________________________________________________________________|___________

int TQSampleFolder::setScaleFactor(TString name, const TString& title, double scaleFactor, double uncertainty) {
  // Set a scale factor (better: normalization factor) and its uncertainty.
  // The 'path' is the name of the corresponding scale factor (typically name of the cut)
  // and may be prefixed with "schemeName:" to use a the scale factor scheme 'schemeName'. 

  /* cumulate (multiply) existing and new scale factors if "<<" was appended */
  if (TQStringUtils::removeTrailing(name, "<") == 2)
    scaleFactor *= getScaleFactor(name);
 
  /* extract the scale scheme (default is '.default') */
  TString scheme = TQStringUtils::readPrefix(name, ":", ".default");
 
  /* get the name of the TQValue object */
  TString objName = getPathTail(name);
 
  /* compile the total path to the scale factor TQValue object */
  TString totalPath = TQFolder::concatPaths(".scalefactors", scheme, name);
 
  /* stop if involved names are invalid */
  if (!TQFolder::isValidPath(totalPath) || !TQValue::isValidName(objName))
    return 0;
 
  if (scaleFactor == 1. && uncertainty == 0.) {
 
    // get/create the folder to contain the tag 
    TQFolder * folder = getFolder(totalPath + "+");
    if(!folder) return 0;

    /* delete existing scale factor and remove folders recursively */
    this->deleteObject(TQFolder::concatPaths(totalPath, objName) + "-");
 
  } else {
 
    /* get/create the folder to contain the TQValue object */
    TQFolder * folder = getFolder(totalPath + "+");
 
    if(!folder) return 0;
    TQCounter* cnt = new TQCounter(objName,scaleFactor,uncertainty);
    cnt->SetTitle(title);
    folder->addObject(cnt,"!");//force possibly existing counter to be overwritten
 
  }
 
  return 1;
}

//__________________________________________________________________________________|___________

int TQSampleFolder::setScaleFactor(const TString& name, double scaleFactor, double uncertainty, const TString& sampleFolders) {
  // Set a scale factor (better: normalization factor) and its uncertainty.
  // The 'path' is the name of the corresponding scale factor (typically name of the cut)
  // and may be prefixed with "schemeName:" to use a the scale factor scheme 'schemeName'. 
  // The default scheme is '.default'.

  // The 'sampleFolders' argument may contain a list or arithmetic string expression of sample folders
  // at which the scale factor should be deployed.

  if (sampleFolders.IsNull())
    return this->setScaleFactor(name,name,scaleFactor,uncertainty);

  /* the number of scale factors added */
  int nScaleFactors = 0;
 
  /* the list of sample folders to add the scale factor to */
  TQSampleDataReader reader(this);
  TList * list = reader.getListOfSampleFolders(sampleFolders);
 
  if (!list) return 0;
 
  /* loop over sample folders */
  TQSampleFolderIterator itr(list,true);
  while(itr.hasNext()){
    TQSampleFolder* sf = itr.readNext();
    if(!sf) continue;
    nScaleFactors += sf->setScaleFactor(name,name,scaleFactor,uncertainty);
  }
 
  /* return the number of scale factors added */
  return nScaleFactors;
}

//__________________________________________________________________________________|___________

int TQSampleFolder::setScaleFactor(const TString& name, double scaleFactor, const TString& sampleFolders) {
  // Set a scale factor (better: normalization factor).
  // The 'path' is the name of the corresponding scale factor (typically name of the cut)
  // and may be prefixed with "schemeName:" to use a the scale factor scheme 'schemeName'. 
  // The default scheme is '.default'.

  // The 'sampleFolders' argument may contain a list or arithmetic string expression of sample folders
  // at which the scale factor should be deployed.

  return this->setScaleFactor(name, scaleFactor, 0., sampleFolders);
}

//__________________________________________________________________________________|___________

int TQSampleFolder::setScaleFactor(const TString& name, double scaleFactor, const char* sampleFolders) {
  // Set a scale factor (better: normalization factor).
  // The 'path' is the name of the corresponding scale factor (typically name of the cut)
  // and may be prefixed with "schemeName:" to use a the scale factor scheme 'schemeName'. 
  // The default scheme is '.default'.

  // The 'sampleFolders' argument may contain a list or arithmetic string expression of sample folders
  // at which the scale factor should be deployed.

  return this->setScaleFactor(name, scaleFactor, 0., sampleFolders);
}


//__________________________________________________________________________________|___________

void TQSampleFolder::printScaleFactors(TString filter) {
  // Prints a summary of scale factors (better: normalization factors) associated
  // to this instance of TQSampleFolder. If no normalization factors are associated
  // to it a corresponding message is printed.

  // get the scale factor folder
  TQFolder * folder = getFolder(".scalefactors");

  // no scale factors available if the folder is missing or if it is empty
  if (!folder || folder->isEmpty()) {
    std::cout << "TQSampleFolder::printScaleFactors(...): No scale factor(s) associated to "
      "this sample folder" << std::endl;
    return;
  }

  // print elements recursively including details
  TString options = "rd";
  if (!filter.IsNull()) {
    // filter options for the print command
    options.Append(TString::Format("f[%s]", filter.Data()));
  }

  // print scale factors
  folder->print(options);
}

//__________________________________________________________________________________|___________

bool TQSampleFolder::getScaleFactor(const TString& path, double& scale, double& uncertainty, bool recursive) {
  // Returns a scale factor (better: normalization factor)
  // The 'path' is the name of the corresponding scale factor (typically name of the cut)
  // and may be prefixed with "schemeName:" to use a the scale factor scheme 'schemeName'. 
  // The default scheme is '.default'.
  //
  // The value of the scale factor will be deployed in the 'scale' argument,
  // The uncertainty value will be deployed in the 'uncertainty' argument.
  //
  // The recursive flag can be used to collect and multiply corresponding scale factors 
  // from parent directories in an upward-recursive fashion (default: false).
 
  /* the default scale factor is of course 1. */
  scale = 1.;
  uncertainty = 0.;
  if(recursive){
    // the recursive call transmits ownership, here we need to delete the counter afterwards
    TQCounter* cnt = this->getScaleFactorCounterRecursive(path);
    if(!cnt) return false;
    scale = cnt->getCounter();
    uncertainty = cnt->getError();
    delete cnt;
  } else {
    // the non-recursive call returns the internal pointer, we should leave it alone
    TQCounter* cnt = this->getScaleFactorCounterInternal(path);
    if(!cnt) return false;
    scale = cnt->getCounter();
    uncertainty = cnt->getError();
  }
  return true;
}

//__________________________________________________________________________________|___________

void TQSampleFolder::convertLegacyNFs(){
  // convert legacy NFs into the most recent format
  TCollection* sfs = this->getListOfFolders("*/.scalefactors");
  TQFolderIterator itr(sfs,true);
  // std::cout << "converting legacy NFs on sample folder " << this->getPath() << std::endl;
  while(itr.hasNext()){
    TQFolder* sff = itr.readNext();
    TCollection* subsfs = sff->getListOfFolders("*");
    TQFolderIterator sitr(subsfs,true);
    while(sitr.hasNext()){
      TQFolder* f = sitr.readNext();
      // std::cout << f->getPath() << std::endl;
      // f->printTags();
      TQIterator vals(f->getListOfTagNames(),true);
      while(vals.hasNext()){
        TObject* v = vals.readNext();
        TString s(v->GetName());
        TString name;
        TQStringUtils::readUpTo(s,name,".");
        // std::cout << "\t" << name << std::endl;
        double scale = f->getTagDoubleDefault(name+".value",f->getTagDoubleDefault(name,1.));
        double unc = f->getTagDoubleDefault(name+".uncertainty",1.);
        if(scale != 1. || unc != 0.){
          TQSampleFolder* base = dynamic_cast<TQSampleFolder*>(sff->getBase());
          if(base){
            // std::cout << "\tsetting NF: " << scale << " +/- " << unc << std::endl;
            base->setScaleFactor(name,name,scale,unc);
          }
        }
      }
      TQIterator objs(f->getListOfObjects());
      while(objs.hasNext()){
        TObject* obj = objs.readNext();
        if(obj->InheritsFrom(TQValue::Class())){
          TQValue* v = (TQValue*)obj;
          TQSampleFolder* base = dynamic_cast<TQSampleFolder*>(sff->getBase());
          if(base) base->setScaleFactor(v->GetName(),v->GetName(),v->getDouble(),0);
        }
      }
    }
  }
}

//__________________________________________________________________________________|___________

TQCounter* TQSampleFolder::getScaleFactorCounter(const TString& path) {
  // Returns a scale factor (better: normalization factor) as the underlying TQCounter object
  // the path is the name of the corresponding scale factor (typically name of the cut)
  // and may be prefixed with "schemeName:" to use a the scale factor scheme 'schemeName'. 
  // The default scheme is '.default'.
  TQCounter* cnt = this->getScaleFactorCounterInternal(path);
  if(!cnt) return NULL;
  return new TQCounter(cnt);
}

 
//__________________________________________________________________________________|___________

TQCounter* TQSampleFolder::getScaleFactorCounterInternal(TString path) {
  // Returns a scale factor (better: normalization factor) as the underlying TQCounter object
  // the path is the name of the corresponding scale factor (typically name of the cut)
  // and may be prefixed with "schemeName:" to use a the scale factor scheme 'schemeName'. 
  // The default scheme is '.default'.
  TString scheme = TQStringUtils::readPrefix(path, ":", ".default");
  TString name = TQFolder::getPathTail(path);
  TString subPath = TQFolder::concatPaths(".scalefactors",scheme,path);
  TQFolder* f = this->getFolder(subPath);
  if(!f) return NULL;
  TQCounter* cnt = dynamic_cast<TQCounter*>(f->getObject(name));

  // NOTE: this function does *not* support legacy sample folders that do not
  // have the NF saved as a counter.
 
  return cnt;
}

//__________________________________________________________________________________|___________

TQCounter* TQSampleFolder::getScaleFactorCounterRecursive(const TString& path) {
  // Returns a scale factor (better: normalization factor) as the underlying TQCounter object
  // the path is the name of the corresponding scale factor (typically name of the cut)
  // and may be prefixed with "schemeName:" to use a the scale factor scheme 'schemeName'. 
  // The default scheme is '.default'.
 
  TQSampleFolder* base = this->getBaseSampleFolder();
  if(base){
    return this->getScaleFactorCounter(path);
  } else {
    TQCounter* master = base->getScaleFactorCounterRecursive(path);
    TQCounter* local = this->getScaleFactorCounterInternal(path);
    master->multiply(local);
    return master;
  }
}


//__________________________________________________________________________________|___________

double TQSampleFolder::getScaleFactor(const TString& path, bool recursive) {
  // Returns a scale factor (better: normalization factor)
  // the path is the name of the corresponding scale factor (typically name of the cut)
  // and may be prefixed with "schemeName:" to use a the scale factor scheme 'schemeName'. 
  // The default scheme is '.default'.
  //
  // The value of the scale factor will be returned.
  //
  // The recursive flag can be used to collect and multiply corresponding scale factors 
  // from parent directories in an upward-recursive fashion (default: false).
  double scale, uncertainty;
  this->getScaleFactor(path,scale,uncertainty,recursive);
  return scale;
}

//__________________________________________________________________________________|___________

int TQSampleFolder::generalizeObjectsPrivate(TList * names, TClass* objClass, TQTaggable* options, const TString& paths, const TString& subpath) {
  // Generalizes (merges) contributions to histograms
  int nObjects=0;
  /* allow multiple categories */
  TQIterator cats(TQStringUtils::tokenize(paths),true);
  while(cats.hasNext()){
    TString cat(cats.readNext()->GetName());
    /* generalize the histograms in given paths */
    TQSampleFolderIterator itr(getListOfSampleFolders(cat),true);
    while(itr.hasNext()){
      TQSampleFolder * obj = itr.readNext();
      if(!obj) continue;
      nObjects += obj->generalizeObjectsPrivate(names, objClass, options, subpath);
    }
  }
  /* return the number of histograms generalized */
  return nObjects;
}

//__________________________________________________________________________________|___________

int TQSampleFolder::generalizeObjects(const TString& prefix, const TString& options) {
  // Generalizes (merges) contributions to histograms
  TQTaggable opttags(options);
  TQSampleFolderIterator itr(this->getListOfSampleFolders("?"),true); //DO NOT iterate over all sample folders (path "*" instead of "?"), instead use recursive call below! Iterating over all sample folders at this point can cause a crash if an element in the collection of the iterator gets removed (due to becoming empty) before being processed itself.
  int n=0;
  while(itr.hasNext()){
    TQSampleFolder* sf = itr.readNext();
    if (!sf) continue;
    //TQIterator tags(sf->getListOfKeys(prefix+".*"),true);
    TQIterator tags(sf->getListOfTags(),false); //list is owned by taggable object
    DEBUGclass("searching '%s' for tags '%s', found %d",sf->getPath().Data(),prefix.Data(),tags.getCollection() ? tags.getCollection()->GetEntries() : 0);
    while(tags.hasNext()){
      TString name(tags.readNext()->GetName());
      /*
      if ( this->hasTag("~"+name) ) {
        std::cout<<"Skipping this tag"<<std::endl;
        continue;//we already generalized this type of objects at a more general position
      } else {
        std::cout<<"Continuing to generalize"<<std::endl;
      }
      */
      
      if ( TQStringUtils::removeLeadingText(name,prefix) ) {
        n += sf->generalizeObjectsPrivate(0 /*everyhwere*/,0 /*all objects*/,&opttags,name);
      }
    }
    n+= sf->generalizeObjects(prefix,options); //recursive call on subfolders.
  }
  return n;
}

//__________________________________________________________________________________|___________

int TQSampleFolder::generalizeObjectsPrivate(TList * names, TClass* objClass, TQTaggable* options, const TString& subpath) {
  // Generalizes (merges) contributions to histograms

  /* remember the the number of histograms generalized */
  int nObjects = 0;
 
  /* we need a data reader to read the counter from the sub folders */
  TQSampleDataReader rd(this);

  /* generalize all counter if no specific list was given */
  bool owningNames = false;
  if (!names) {
    names = rd.getListOfObjectNames(objClass, subpath);
    owningNames = true;
  }

  DEBUGclass("using %d names of type %s in %s:%s",
            names ? names->GetEntries() : 0,objClass ? objClass->GetName() : "(NULL)",
            this->getPath().Data(),subpath.Data());

  if(!names){
    return 0;
  }
  /* loop over every element in the list */
  TQIterator itr(names,owningNames);
  while (itr.hasNext()){
    TObject* obj = itr.readNext();
    /* the element name to generalize */
    TString name = obj->GetName();
    /* the generalized histgram */
    TObject* element = NULL;
    /* the list of samples folders contributing to the element */
    TList * list = new TList();
    /* get the element */
    element = rd.getElement(".", name, objClass, subpath, options, list);
    
    // skip this element if we failed to get the generalized element
    if (!element) {
      ERRORclass("unable to access element '%s' of type '%s' in %s:%s",name.Data(),objClass?objClass->GetName():"(NULL)",this->getPath().Data(),subpath.Data());
      delete list;
      continue;
    }
    /* skip this element if the only contribution is from this sample folder */
    if (list->GetEntries() == 1 && list->First() == this) {
      DEBUGclass("not generalizing -- object already at target location!");
      delete list;
      continue;
    }
 
    /* store the generalized element
     * =================================================== */
 
    /* get/create the folder to store the element */
    TString path(TQFolder::concatPaths(subpath,TQFolder::getPathWithoutTail(name)));
    TQFolder * folder = getFolder(path + "+");
    if (folder) {
      /* delete existing element */
      folder->deleteObject(element->GetName());
      /* store the generalized element */
      folder->Add(element);
      DEBUGclass("adding object '%s' to '%s'",element->GetName(),folder->getPath().Data());
      nObjects++;
      //convert TList to std::vector, since the TList destructor can segfault if its contents are already gone...
      std::vector<TQSampleFolder*> vList;
      vList.reserve(list->GetSize()); 
      TQSampleFolderIterator itr2(list);
      while(itr2.hasNext()){
        vList.push_back(itr2.readNext());
      }
      /* delete the list of contributing sample folders */
      delete list;
      /* loop over contributing sample folders and delete the source elements */
      for(size_t i=0; i<vList.size(); i++) {
        TQSampleFolder* sf = vList[i];
        if(!sf || sf == this) continue;
        sf->deleteObject(TQFolder::concatPaths(subpath,name)+"-");
      }
    } else {
      ERRORclass("unable to make folder '%s'!",path.Data());
      delete element;
      /* delete the list of contributing sample folders */
      delete list;
    }
  }
  if(nObjects > 0){
    //@tag:[.generalize.visited] This tag is set when objects are generalized ('downmerged') at the folder the objects are combined at. For merging sample folders with tqmerge this means the trace ID should be set to 'generalize'.
    TQFolder* sf = this;
    while(sf){
      sf->setTagBool(".generalize.visited",true);
      sf = sf->getBase();
    }
  }
  
  /* the number of generalized elements */
  return nObjects;
}

//__________________________________________________________________________________|___________

int TQSampleFolder::generalizeHistograms (const TString& paths, const TString& options){
  // Generalizes (merges) contributions to histogram
  return this->generalizeHistograms(0, paths, options);
}

//__________________________________________________________________________________|___________

int TQSampleFolder::generalizeCounters    (const TString& paths, const TString& options){
  // Generalizes (merges) contributions to counter
  return this->generalizeCounters(0, paths, options);
}

//__________________________________________________________________________________|___________

int TQSampleFolder::generalizeHistograms(TList * names, const TString& paths, const TString& options) {
  // Generalizes (merges) contributions to histogram
  TQTaggable tags(options);
  return this->generalizeObjectsPrivate(names, TH1::Class(), &tags, paths, ".histograms");
}

//__________________________________________________________________________________|___________

int TQSampleFolder::generalizeCounters(TList * names, const TString& paths, const TString& options) {
  // Generalizes (merges) contributions to counter
  TQTaggable tags(options);
  return this->generalizeObjectsPrivate(names, TQCounter::Class(), &tags, paths, ".cutflow");
}

//__________________________________________________________________________________|___________

TQSampleFolder::~TQSampleFolder() {
}

//__________________________________________________________________________________|___________

bool TQSampleFolder::merge(TQSampleFolder* f, bool sumElements, bool verbose){
  // merge two sample folders
  return this->merge(f,"asv",sumElements,verbose);
}

//__________________________________________________________________________________|___________

bool TQSampleFolder::merge(TQSampleFolder* f, const TString& traceID, bool sumElements, bool verbose){
  // merge two sample folders
  if(this->Class() == TQSampleFolder::Class()){
    return this->mergeAsSampleFolder(f,traceID,sumElements ? MergeMode::SumElements : MergeMode::PreferOther,verbose);
  } else {
    ERRORclass("unable to merge '%s' with 'TQSampleFolder'",this->Class()->GetName());
    return false;
  }
}

//__________________________________________________________________________________|___________

bool TQSampleFolder::mergeAsSampleFolder(TQSampleFolder* f, const TString& traceID, MergeMode mode, bool verbose){
  // merge two sample folders
  if(!f) return false;
  this->resolveImportLinks();
  bool didMerge = false;
  TQFolderIterator itr(f->getListOfFolders("?"));
  // iterate over all the folders
  while(itr.hasNext()){
    TQFolder* otherF = itr.readNext();
    if(!otherF) continue;
    TQSampleFolder* otherSF = dynamic_cast<TQSampleFolder*>(otherF);
    TQSample* otherSample = dynamic_cast<TQSample*>(otherSF);
    TObject* thisObj = this->getObject(otherF->GetName());
    if(!thisObj){
      // if we don't have an object of that type, we can just steal it
      otherF->detachFromBase();
      this->addObject(otherF);
      didMerge = true;
      if(verbose) VERBOSEclass("grabbing object '%s'",otherF->getPath().Data());
    } else {
      // if we have an object of that type, we need to check which one it is
      otherF->resolveImportLinks();
      if(otherSample){
        // the other one is a sample
        TQSample* thisSample = dynamic_cast<TQSample*>(thisObj);
        if(!thisSample){
          // there is a corresponding object in this sample folder, but it's not a sample
          ERRORclass("cannot merge sample '%s' with other object of type '%s'",otherSample->GetName(),thisObj->Class()->GetName());
        } else {
          // there is a corresponding object in this sample folder, and it's a sample
          if(!otherSample->hasSubSamples() && !thisSample->hasSubSamples()){
            // neither of the two samples has a subsample
            // in this case, we can simply add them as if they were folders
            int thisDate, otherDate;
            bool otherHasTimestamp = false; bool thisHasTimestamp = false;
            if (mode != SumElements) { //only read those tags if actually relevant
              otherHasTimestamp = otherSF->getTagInteger("."+traceID+".timestamp.machine",otherDate);
              thisHasTimestamp = thisSample->getTagInteger("."+traceID+".timestamp.machine",thisDate);
            }
            //we want to perform merging if one of the following conditions is true
            // -sumElement
            // -no time stamp on this instance but on the other one (take the content of the stamped one)
            // -both timestamps exist and the other one is newer (take the newer version)
            if(mode == SumElements || (!thisHasTimestamp && otherHasTimestamp) || ( (thisHasTimestamp && otherHasTimestamp) && (otherDate > thisDate) ) ){
              if(verbose) VERBOSEclass("Sample '%s' with no sub-samples: merge '%s' with timestamps: this is %s, other is %s",otherSample->getPath().Data(),otherSample->GetName(),
                                       thisSample->getTagStringDefault("."+traceID+".timestamp.human","<unknown>").Data(),
                                       otherSample->getTagStringDefault("."+traceID+".timestamp.human","<unknown>").Data());
              
              didMerge = thisSample->mergeAsFolder(otherSample,mode) || didMerge;
            } else {
              if(verbose) VERBOSEclass("Sample '%s' with no sub-samples: skip '%s' due to timestamps: this is %s, other is %s",otherSample->getPath().Data(),otherSample->GetName(),
                                       thisSample->getTagStringDefault("."+traceID+".timestamp.human","<unknown>").Data(),
                                       otherSample->getTagStringDefault("."+traceID+".timestamp.human","<unknown>").Data());
            }
          } else if(otherSample->hasSubSamples() && thisSample->hasSubSamples()){
            // both samples have sub samples
            // need to merge recursively to go deeper down the sample structure
            if(verbose) VERBOSEclass("Sample with sub-samples, going deeper for '%s'",otherSample->getPath().Data());
            didMerge = thisSample->mergeAsSampleFolder(otherSample,traceID,mode,verbose) || didMerge;
          } else {
            // one of the samples has sub samples, but the other one doesn't
            // we should never do this!
            ERRORclass("refusing to merge sample that has sub samples with one that has none at '%s'",thisSample->getPath().Data());
          }
        }
      } else if(otherSF){
        // the other one is a sample folder
        TQSampleFolder* thisSF = dynamic_cast<TQSampleFolder*>(thisObj);
        if(!thisSF){
          // there is a corresponding object in this sample folder, but it's not a sample folder
          ERRORclass("cannot merge sample folder '%s' with other object of type '%s'",otherSF->GetName(),thisObj->Class()->GetName());
        } else {
          // there is a corresponding object in this sample folder, and it's a sample folder
          if(otherSF->getTagBoolDefault("."+traceID+".visited~",false)){
            // the other one was visited
            // we need to look at its contents
            if(thisSF->getTagBoolDefault("."+traceID+".visited~",false)){
              // this one was visited as well
              // in this case, we need to merge recursively
              if(verbose) VERBOSEclass("merging from '%s'",otherSF->getPath().Data());
              didMerge = thisSF->mergeAsSampleFolder(otherSF,traceID,mode,verbose) || didMerge;
            } else {
              // this one was not visited
              // we just use the other one
              if(verbose) VERBOSEclass("grabbing '%s'",otherSF->getPath().Data());
              otherSF->detachFromBase();
              thisSF->detachFromBase();
              delete thisSF;
              this->addObject(otherSF);
              didMerge = true;
            }
          } else {
            // the other one was not visited
            // we can ignore it
            if(verbose) VERBOSEclass("ignoring '%s'",thisSF->getPath().Data());
          }
        }
      } else if(otherF){
        // the other one is a regular folder
        TQFolder* thisF = dynamic_cast<TQFolder*>(thisObj);
        if(!thisF){
          // there is a corresponding object in this sample folder, but it's not a folder
          ERRORclass("cannot merge folder '%s' with other object of type '%s'",otherF->GetName(),thisObj->Class()->GetName());
        } else {
          int thisDate, otherDate;
          bool otherHasTimestamp = false; bool thisHasTimestamp = false;
          if (mode != SumElements) {
            //only get the time stamps if it's relevant
            bool otherHasTimestamp = f->getTagInteger("."+traceID+".timestamp.machine",otherDate);
            bool thisHasTimestamp = this->getTagInteger("."+traceID+".timestamp.machine",thisDate);
          } 
          //we want to perform merging if one of the following conditions is true
          // -sumElement
          // -no time stamp on this instance but on the other one (take the content of the stamped one)
          // -both timestamps exist and the other one is newer (take the newer version)
          if(mode == SumElements || (!thisHasTimestamp && otherHasTimestamp) || ( (thisHasTimestamp && otherHasTimestamp) && (otherDate > thisDate) ) ){
            if(verbose) VERBOSEclass("Sample folder '%s': merge '%s' with timestamps: this is %s, other is %s",f->getPath().Data(),otherF->GetName(),
                                     this->getTagStringDefault("."+traceID+".timestamp.human","<unknown>").Data(),
                                     f->getTagStringDefault("."+traceID+".timestamp.human","<unknown>").Data());
            // there is a corresponding object in this sample folder, and its also a folder
            // there are some issues with protection, hence the following line looks a bit silly
            didMerge = (thisF->*&TQSampleFolder::mergeAsFolder)(otherF,mode) || didMerge;
            // what it actually does is just 
            // thisF->mergeAsFolder(otherF,mode)
            // for reference, see 
            // http://stackoverflow.com/questions/30111578/trying-to-call-protected-function-of-parent-class-in-child-class-in-c
          } else {
            if(verbose) VERBOSEclass("Sample folder '%s': skip '%s' due to timestamps: this is %s, other is %s",f->getPath().Data(),otherF->GetName(),
                                     this->getTagStringDefault("."+traceID+".timestamp.human","<unknown>").Data(),
                                     f->getTagStringDefault("."+traceID+".timestamp.human","<unknown>").Data());
          }
        }
      }
    }
  }
  this->mergeTags(f);
  this->mergeObjects(f,mode);
  return didMerge;
}

//__________________________________________________________________________________|___________



void TQSampleFolder::findFriends(const TString& pathpattern, bool forceUpdateSubfolder){
  // find the friends of this sample folder in a branch of the current folder tree
  TQSampleFolder* base = this->getBaseSampleFolder();
  TQSampleFolderIterator itr(base->getListOfSampleFolders(pathpattern),true);
  while(itr.hasNext()){
    TQSampleFolder* sf= itr.readNext();
    this->findFriends(sf, forceUpdateSubfolder);
  }
}

//__________________________________________________________________________________|___________

void TQSampleFolder::findFriends(bool forceUpdateSubfolder){
    // find the friends of this sample folder in the current folder tree
  TQSampleFolder* base = this->getRootSampleFolder();
  this->findFriends(base, forceUpdateSubfolder);
}

//__________________________________________________________________________________|___________

void TQSampleFolder::findFriends(TQSampleFolder* otherSF, bool forceUpdateSubfolder){
  if (this->fIsFindingFriends) return; //we're already running this method for this instance, prevent infinite recursions!
  this->fIsFindingFriends = true;
  if (forceUpdateSubfolder || (this->countFriends() == 0) ) this->findFriendsInternal(otherSF,forceUpdateSubfolder);
  this->fIsFindingFriends = false;
  return;
}

void TQSampleFolder::findFriendsInternal(TQSampleFolder* otherSF, bool forceUpdateSubfolder){
  // find the friends of this sample folder in some other sample folder
  
  TQSampleFolderIterator itr(otherSF->getListOfSampleFolders("*"));
  //TQSampleFolderIterator itr(otherSF->getListOfSampleFolders(this->getPathWildcarded()));
  //now check if all sub-samples are friends and if so, befriend with their sup-folder
  DEBUGclass("checking for friends of '%s'",this->getPath().Data());
  itr.reset();
  TQSampleFolderIterator subitr(this->getListOfSampleFolders("?"));
  while(itr.hasNext()){
    TQSampleFolder* other = itr.readNext();
    if (!other) continue;
    if (this->areRelated(other) < 0 || this->areRelated(other) > 1) continue; //skip combination of "this" and "other" if they are neither the same nor in different paths. This is neededto prevent infinite recursions if a folder name appear multiple times in a path (e.g., /foo/bar/buzz/foo)
    //if (this->isFriend(other)) continue; //if they are already friends we don't need to do anything
    if (!forceUpdateSubfolder && other->hasFriends() ) continue; //if the other sample folder already has friends its list should already be complete (using symmetry of the equivalence relation). NOTE: this cannot be applied to 'this' as we're just constructing its list!
    //we apply the requirements: existence (not a null pointer), has sub samples, same number of (sub) samples as this sample, 
    if (!other || (this->getNSamples() != other->getNSamples()) || (this->getNSampleFolders() != other->getNSampleFolders()) ) continue;
    bool befriend = subitr.hasNext(); //check if there is even an entry
    while(subitr.hasNext() && befriend) {
      TQSampleFolder* thisSub = subitr.readNext();
      if (!thisSub) continue;
      TQSampleFolder* otherSub = other->getSampleFolder(thisSub->GetName());
      if (!otherSub) {befriend = false; break;} //if the friend candidate doesn't have a SampleFolder this folder has, we don't consider it a friend.
      if (!thisSub->isFriend(otherSub)) {
      thisSub->findFriends(otherSF, forceUpdateSubfolder);//can this be a place for optimization? Maybe only have
      //otherSub->findFriends(otherSF, forceUpdateSubfolder); //do not enable this line it's just a reminder (not to do this)!
      }
      
      befriend = befriend && thisSub->isFriend(otherSub);
      //if (!befriend) INFOclass("these two are not friends: '%s' and '%s'",thisSub->getPath().Data(),otherSub->getPath().Data());
    }
    subitr.reset();
    if (befriend) {
      DEBUGclass("befriending '%s' with '%s'",this->getPath().Data(),other->getPath().Data());
      this->befriend(other);
    }
  }
}

//__________________________________________________________________________________|___________

void TQSampleFolder::befriend(TQSampleFolder* other){
  // add another sample as a friend
  
  if (!other) return;
  if (this->hasFriends() && other->hasFriends()) {
    //check that the friend sets are the same
    if ( !this->isFriend(other) ) {
      //something is really wrong!
      this->print("dt");
      other->print("dt");
      throw std::runtime_error(TString::Format("Cannot befriend sample(folder) '%s' and '%s': Both already have friends but the latter is not a friend of the former",this->getPath().Data(),other->getPath().Data()).Data());
    } else if ( !other->isFriend(this) ) {
      throw std::runtime_error(TString::Format("Cannot befriend sample(folder) '%s' and '%s': Both already have friends and the latter is a friend of the former but not the other way around. This should never happen! Unless you expect that your setup causes this issue, please report this incident (with a way to reproduce it) to qframework-users@SPAMNOTcern.ch !",this->getPath().Data(),other->getPath().Data()).Data());
    }
    //everything is fine, the two sample folders are already friends
  } else if (other->hasFriends() && !this->hasFriends()) {
    //other samplefolder already has friends. Let's get his list and add this samplefolder
    this->fFriends = other->getFriends();
    this->fFriends->insert(this);
    
  } else if( this->hasFriends() && !other->hasFriends() ){
    //we already have a friend set but the other one doesn't. This is the mirrored case of the one before, so let's just flip 'this' and 'other'
    other->befriend(this);

  } else if( !this->hasFriends() && !other->hasFriends() ) {
    //both, this and other, don't have any friends yet. We create a new set, add 'this' and notify the 'other'
    this->fFriends = std::shared_ptr<std::set<TQSampleFolder*>>(new std::set<TQSampleFolder*>()); //new TList();
    this->fFriends->insert(this);
    other->befriend(this); //if 'other' doesn't have a set of friends yet it will ask for the one just created and add itself
  } else {
    throw std::runtime_error("There seems to be a case the code monkey has not thought of, better get him another cup of coffee!");
  }
  
}

//__________________________________________________________________________________|___________

void TQSampleFolder::clearFriends(){
  // clear the list of friends
  if(! (this->fFriends == nullptr) ){
    this->fFriends.reset();
    this->fFriends = nullptr;
  }
}

//__________________________________________________________________________________|___________

std::shared_ptr<std::set<TQSampleFolder*>> TQSampleFolder::getFriends(){
  // retrieve the set of friends
  return this->fFriends;
}

//__________________________________________________________________________________|___________

int TQSampleFolder::countFriends(){
  // retrieve the list of friends
  if(this->fFriends == nullptr) return 0;
  return this->fFriends->size();
}

//__________________________________________________________________________________|___________

bool TQSampleFolder::hasFriends(){
  // retrieve the list of friends
  return (this->countFriends() > 0);
}

//__________________________________________________________________________________|___________

bool TQSampleFolder::isFriend(TQSampleFolder* other){
  // print the list of friends
  if (this->fFriends == nullptr) return false;
  if (!other) return false;
  return (this->fFriends->count(other)>0);
}

//__________________________________________________________________________________|___________

void TQSampleFolder::printFriends(){
  // print the list of friends 
  //TQSampleFolderIterator itr(this->fFriends);
  //while(itr.hasNext()){
  if (this->fFriends == nullptr) {
    std::cout<< TQStringUtils::makeBoldWhite("<no friends>")<<std::endl;
    return;
  }
  for (auto s : (*this->fFriends)) {
    //TQSampleFolder* s = itr.readNext();
    if(!s) continue;
    TString path(s->getPath());
    path.Prepend(":");
    path.Prepend(s->getBase()->GetName());
    std::cout << TQStringUtils::makeBoldWhite(TQStringUtils::fixedWidth(s->GetName(),20)) << " " << TQStringUtils::fixedWidth(path,50) << std::endl;
  }
  //}
}

//__________________________________________________________________________________|___________

void TQSampleFolder::purgeWithoutTag(const TString& tag){
  TQSampleFolderIterator itr(this->getListOfSampleFolders("*"),true);
  std::set<TQSampleFolder*> kill;
  std::set<TQSampleFolder*> protect;
  while(itr.hasNext()){
    TQSampleFolder* sf = itr.readNext();
    if(!sf) continue;
    if(!sf->hasTag(tag)){
      kill.insert(sf);
    } else {
      protect.insert(sf);
    }
  }
  for(auto sf:protect){
    TQSampleFolder* base = sf->getBaseSampleFolder();
    while(base){
      if(kill.find(base) != kill.end()){
        kill.erase(base);
      }
      base = base->getBaseSampleFolder();
    }
  }
  if(kill.find(this) != kill.end()){
    kill.erase(this);
  }
  std::set<TQSampleFolder*> dependent;
  for(auto sf:kill){
    if(kill.find(sf->getBaseSampleFolder())!=kill.end()){
      dependent.insert(sf);
    }
  }
  for(auto sf:dependent){
    kill.erase(sf);
  }
  for(auto sf:kill){
    sf->detachFromBase();
    delete sf;
  }
}
