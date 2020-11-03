// ROOT
#include "TString.h"
#include "TSystemDirectory.h"
#include "TSystemFile.h"
#include "TTreeFormula.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TError.h"
#include "TFile.h"
#include "TMatrixD.h"
#include "TSystem.h"
#include "TPad.h"
#include "TMD5.h"
#include "TDatime.h"

// STL
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>
#include <limits>

// STDC
#include "dirent.h"

#include <QFramework/ASG.h>

#ifdef HAS_XAOD
#warning using ASG_RELEASE compilation scheme
#define XAOD_STANDALONE 1
#define private public
#define protected public
#include "xAODRootAccess/TEvent.h"

#include "xAODCutFlow/CutBookkeeper.h"
#include "xAODCutFlow/CutBookkeeperContainer.h"
#include "xAODCutFlow/CutBookkeeperAuxContainer.h"
#include "xAODRootAccess/tools/TReturnCode.h"
#include "xAODTruth/TruthParticleContainer.h"
#include "xAODTruth/TruthParticle.h"
#undef private
#undef protected
#define ASG_RELEASE 1
#endif

// QFramework
#include "QFramework/TQUtils.h" 
#include "QFramework/TQListUtils.h"
#include "QFramework/TQStringUtils.h"
#include "QFramework/TQFolder.h"
#include "QFramework/TQLibrary.h"
#include "QFramework/TQIterator.h"
// local stuff
#include "definitions.h"

using std::string;
using std::vector;

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQUtils:
//
// The TQUtils namespace provides a variety of utility methods.
//
////////////////////////////////////////////////////////////////////////////////////////////////



//__________________________________________________________________________________|___________

int TQUtils::getMinIndex(int val0, int val1, int val2,
                         bool accept0, bool accept1, bool accept2) {
 
  vector<int> vals;
  vals.push_back(val0);
  vals.push_back(val1);
  vals.push_back(val2);
 
  vector<bool> accepts;
  accepts.push_back(accept0);
  accepts.push_back(accept1);
  accepts.push_back(accept2);

  return getMinIndex(vals, accepts);
}


//__________________________________________________________________________________|___________

int TQUtils::getMinIndex(const vector<int>& vals, const vector<bool>& accepts) {
 
  size_t minIndex = -1;
  for (size_t i = 0; i < 3; i++) {
    if ((minIndex == (size_t)(-1) || vals[i] < vals[minIndex]) && i < accepts.size() && accepts[i]) {
      minIndex = i;
    }
  }
 
  return minIndex;
}


//__________________________________________________________________________________|___________

bool TQUtils::areEquivalent(TObject * obj1, TObject * obj2) {
  // Returns true if objects <obj1> and <obj2> are considered equivalent. [not yet
  // implemented]
 
  if (!obj1 || !obj2) {
    return false;
  }
 
  if (obj1->IsA() != obj2->IsA()) {
    return false;
  }
 
  return false;
 
  // return true;
}


//__________________________________________________________________________________|___________

bool TQUtils::ensureDirectoryForFile(TString filename) {
  // ensure that the directory for the given file exists
  TString dir = TQFolder::getPathWithoutTail(filename);
  if(dir.IsNull()) return true;
  return ensureDirectory(dir);
}

//__________________________________________________________________________________|___________

char TQUtils::getLastCharacterInFile(const char* filename){
  // retrieve the last character in a text file
  FILE * f = fopen (filename, "r");
  if (!f) return '\0';
  if (fseek( f, -1, SEEK_END ) != 0 ) return '\0';
  char last = fgetc(f);
  return last;
}

//__________________________________________________________________________________|___________

bool TQUtils::ensureTrailingNewline(const char* filename){
  // check if the given text file has a terminating newline character
  // if this is not the case, append a newline at the end
  if(getLastCharacterInFile(filename)=='\n'){
    return true;
  }
  std::fstream filestr;
  filestr.open (filename, std::fstream::in | std::fstream::out | std::fstream::app);
  if(!filestr.is_open()) return false;
  filestr<< '\n';
  filestr.close();
  return true;
}

//__________________________________________________________________________________|___________

bool TQUtils::ensureDirectory(TString path) {
  // ensure that the directory with the given path exists
  // check if directory <path> exists
  Long_t flags = 0;
  gSystem->GetPathInfo(path.Data(), (Long_t*)0, (Long_t*)0, &flags, (Long_t*)0);
  if (flags & 2) {
    // directory exists
    return true;
  } 
  //create directory
  if (0 == gSystem->mkdir(path.Data(),true)) return true;
  else return false;
}

//__________________________________________________________________________________|___________

TFile * TQUtils::openFile(const TString& filename) {

  TFile* file = TFile::Open(filename.Data());

  if(file && file->IsOpen()){

    return file;
    
  } else {

    if(file) {
      delete file;
      file = 0;
    }
    return NULL;

  }

}

//__________________________________________________________________________________|___________

bool TQUtils::fileExists(const TString& filename) {
  // Check whether the file <filename> exists and return true if so,
  // and false otherwise
 
  if (filename.BeginsWith("root:") || filename.BeginsWith("dcap:")) return TQUtils::fileExistsEOS(filename);
  else return TQUtils::fileExistsLocal(filename);

  return false;
}

//__________________________________________________________________________________|___________

bool TQUtils::fileExistsEOS(const TString& filename) {
  // Check whether the file <filename> exists and return true if so,
  // and false otherwise. Works on eos paths.

  bool exists = false;
  /* temporarily increase error ignore level */
  int tmp = gErrorIgnoreLevel;
  gErrorIgnoreLevel = 5000;
  // TQLibrary::redirect_stderr("/dev/null");
  TFile * file = TFile::Open(filename.Data(), "READONLY");
  if(file){
    if(file->IsOpen()){
      file->Close();
      exists = true;
    }
    delete file;
  }
  // TQLibrary::restore_stderr();

  /* restore error ignore level */
  gErrorIgnoreLevel = tmp;
 
  /* return true if we successfully accessed the file */
  return exists;
}

//__________________________________________________________________________________|___________

bool TQUtils::fileExistsLocal(const TString& filename) {
  // check if the file exists and is readable;
 
  /* check if the file exists by trying to open it with an ifstream */
  std::ifstream ifile(filename.Data(), std::ifstream::binary);
  bool exists = false;

  /* if this succeeded, we need to close it again */
  if (ifile.is_open()) {
    exists = true;
    ifile.close();
  } 
 
  /* return true if we successfully accessed the file */
  return exists;
}


//__________________________________________________________________________________|___________

TList * TQUtils::getListOfFilesMatching(TString pattern) {
  // Scans the file system for files whose names match the string pattern <pattern>
  // and returns a list (instance of TList) of matching filenames (as instances of
  // TObjString).

  pattern = TQStringUtils::trim(pattern);
  if (!pattern.BeginsWith("./") && !pattern.BeginsWith("/") && !pattern.BeginsWith("root://")) {
    pattern.Prepend("./");
  }

  /* split input path pattern */
  TString head;
  TString core;
  TString tail = pattern;
  bool stop = false;
  int pre = TQStringUtils::removeLeading(tail, "/");
  while (!stop && !tail.IsNull()) {
    core = TQFolder::getPathHead(tail);
    if (!TQStringUtils::hasWildcards(core)) {
      head = TQFolder::concatPaths(head, core);
      core.Clear();
    } else {
      stop = true;
    }
  }
  if (head.IsNull() && pre > 0)
    head = ".";
  if (pre > 0)
    head.Prepend(TQStringUtils::repeat("/", pre));
  if (head.CompareTo("~") == 0)
    head.Append("/");

  /* done if no wildcards left to resolve */
  if (core.IsNull()) {
    TString name = gSystem->ExpandPathName(pattern.Data());
    if (fileExists(name)) {
      TList * files = new TList();
      files->SetOwner(true);
      files->Add(new TObjString(name));
      return files;
    } else {
      return NULL;
    } 
  }

  TList * files = new TList();
  files->SetOwner(true);

  /* resolve core wildcard */
  TSystemDirectory * dir = new TSystemDirectory("", gSystem->ExpandPathName(head.Data()));
  TList * subFiles = dir->GetListOfFiles();
  if (subFiles) {
    TSystemFile * file;
    TIterator * itr = subFiles->MakeIterator();
    while ((file = (TSystemFile*)itr->Next())) {
      TString name = file->GetName();
      /* don't let wildcards match relative references */
      if (name.CompareTo(".") == 0 || name.CompareTo("..") == 0)
        continue;
      if (TQStringUtils::matches(name, core)) {
        TString path = TQFolder::concatPaths(head, name);
        path = TQFolder::concatPaths(path, tail);

        TList * subList = getListOfFilesMatching(path);
        if (subList) {
          subList->SetOwner(false);
          files->AddAll(subList);
          delete subList;
        }
      }
    }
    delete itr;
    delete subFiles;
  }
  delete dir;

  // return a null pointer instead of an empty list
  if (files->GetEntries() == 0) {
    delete files;
    files = NULL;
  }

  return files;
}


//__________________________________________________________________________________|___________

bool TQUtils::parseAssignment(TString input, TString * dest, TString * source) {
 
  TList * tokens = TQStringUtils::tokenize(input, "=");

  /* we expect exactly two tokens */
  if (tokens->GetEntries() != 2) { return false; }
 
  /* make sure the string pointers a valid before assigning the results */
  if (dest) { *dest = TQStringUtils::trim(tokens->At(0)->GetName()); }
  if (source) { *source = TQStringUtils::trim(tokens->At(1)->GetName()); }
 
  tokens->Delete();
  delete tokens;
 
  return true;

}

//__________________________________________________________________________________|___________
TObjArray * TQUtils::getBranchNames(const TString& input) {
  /* Strip all valid branch names from a TString
   * and return them in a TObjArray */
  TString startChars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
  TString validChars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_";
  TString sepChars = " =<>!*/+-()";
  TObjArray * params = new TObjArray();

  std::string str(input.Data());
  int startPos = str.find_first_of(startChars.Data());
  int endPos = str.find_first_not_of(validChars.Data(), startPos);

  while (true) {
    if (endPos == -1)
      endPos = str.size();
 
    TString bname = TQStringUtils::trim(input(startPos,endPos-startPos));
    bool addfCoordinates = false;
    if(endPos < (int)(str.size())){
      char next = input[endPos];
      if(next == '[' || next == '.'){
        bname.Append("*");
        addfCoordinates = true;
      }
    }
    params -> Add(new TObjString(bname));
    if(addfCoordinates)
        params -> Add(new TObjString("fCoordinates*"));
 
    int nextPos = str.find_first_of(sepChars.Data(), endPos);
    if(nextPos == -1) break;
    startPos = str.find_first_of(startChars.Data(), nextPos);
    if(startPos == -1) break;
    endPos = str.find_first_not_of(validChars.Data(), startPos);
  }
 
  return params;
}


//__________________________________________________________________________________|___________

bool TQUtils::isNum(double a){
  /* Determine if a given number is a valid and finite numerical value */

  // detect NaN
  if (a!=a) return false;
  // determine if the value is finite
  if ((a<std::numeric_limits<double>::infinity()) && (a>-std::numeric_limits<double>::infinity())) return true;
  return false;
}

//__________________________________________________________________________________|___________

bool TQUtils::inRange(double x, double a, double b){
  /* Determine if a given number is in a given range 
   * if a<b, inRange(x,a,b) will return true if a<=x<=b, else false
   * if a>b, inRange(x,a,b) will return false if a<x<b, else true
   */
 
  if(a<b) return (bool)((a<=x) && (x<=b));
  else return (bool)((x<=b) || (a<=x));
}

//__________________________________________________________________________________|___________

unsigned long TQUtils::getCurrentTime(){
  // returns the current time in milliseconds since 01 Jan 1970, 0:00
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return tp.tv_sec * 1000 + tp.tv_usec / 1000;
}

//__________________________________________________________________________________|___________

TString TQUtils::getTimeStamp(const TString& format, size_t len){
  // return the current time formatted according to the given string
  time_t rawtime;
  struct tm * timeinfo;
  char* buffer = (char*)malloc(len*sizeof(char));
  time (&rawtime);
  timeinfo = localtime (&rawtime);
  strftime (buffer,80,format.Data(),timeinfo);
  TString retval(buffer);
  free(buffer);
  return retval;
}

//__________________________________________________________________________________|___________

double TQUtils::convertXtoNDC(double x) {
  // convert an x-coordinate (horizontal) 
  // from user to NDC coordinates
  // using the current pad
  gPad->Update();
  return (x - gPad->GetX1())/(gPad->GetX2()-gPad->GetX1());
}

//__________________________________________________________________________________|___________

double TQUtils::convertYtoNDC(double y) {
  // convert an y-coordinate (vertical) 
  // from user to NDC coordinates
  // using the current pad
  gPad->Update();
  return (y - gPad->GetY1())/(gPad->GetY2()-gPad->GetY1());
}

//__________________________________________________________________________________|___________

double TQUtils::convertXtoPixels(double x) {
  // convert an x-coordinate (horizontal) 
  // from user to absolute (pixel) coordinates
  // using the current pad
  return TQUtils::convertXtoNDC(x) * gPad->GetWw();
}

//__________________________________________________________________________________|___________

double TQUtils::convertYtoPixels(double y) {
  // convert an y-coordinate (vertical) 
  // from user to absolute (pixel) coordinates
  // using the current pad
  return TQUtils::convertXtoNDC(y) * gPad->GetWh();
}

//__________________________________________________________________________________|___________

double TQUtils::convertXfromNDC(double x) {
  // convert an x-coordinate (horizontal) 
  // from NDC to user coordinates
  // using the current pad
  gPad->Update();
  return gPad->GetX1() + x*(gPad->GetX2()-gPad->GetX1());
}

//__________________________________________________________________________________|___________

double TQUtils::convertYfromNDC(double y) {
  // convert an y-coordinate (vertical) 
  // from NDC to user coordinates
  // using the current pad
  gPad->Update();
  return gPad->GetY1() + y*(gPad->GetY2()-gPad->GetY1());
}

//__________________________________________________________________________________|___________

double TQUtils::convertdXtoNDC(double x) {
  // convert an x-distance (horizontal) 
  // from user to NDC coordinates
  // using the current pad
  gPad->Update();
  return (x)/(gPad->GetX2()-gPad->GetX1());
}

//__________________________________________________________________________________|___________

double TQUtils::convertdYtoNDC(double y) {
  // convert an y-distance (vertical) 
  // from user to NDC coordinates
  // using the current pad
  gPad->Update();
  return (y)/(gPad->GetY2()-gPad->GetY1());
}

//__________________________________________________________________________________|___________

double TQUtils::convertdXtoPixels(double x) {
  // convert an x-distance (horizontal) 
  // from user to absolute (pixel) coordinates
  // using the current pad
  return TQUtils::convertdXtoNDC(x) * gPad->GetWw();
}

//__________________________________________________________________________________|___________

double TQUtils::convertdYtoPixels(double y) {
  // convert an y-distance (vertical) 
  // from user to absolute (pixel) coordinates
  // using the current pad
  return TQUtils::convertdYtoNDC(y) * gPad->GetWh();
}

//__________________________________________________________________________________|___________

double TQUtils::convertdXfromNDC(double x) {
  // convert an x-distance (horizontal) 
  // from NDC to user coordinates
  // using the current pad
  gPad->Update();
  return x*(gPad->GetX2()-gPad->GetX1());
}

//__________________________________________________________________________________|___________

double TQUtils::convertdYfromNDC(double y) {
  // convert an y-distance (vertical) 
  // from NDC to user coordinates
  // using the current pad
  gPad->Update();
  return y*(gPad->GetY2()-gPad->GetY1());
}

//__________________________________________________________________________________|___________

double TQUtils::getAverage(std::vector<double>& vec) {
  if (vec.size() < 1 ) {
    WARNfunc("Too few entries in argument vector<double>* vec : %i found, 1 or more required, returning 0!",vec.size());
    return 0;
  }
  double sum = 0;
  for (uint i=0;i<vec.size();i++) {
    sum += vec.at(i);
  }
  sum /=vec.size();
  return sum;
}

//__________________________________________________________________________________|___________

double TQUtils::getSampleVariance(std::vector<double>& vec) {
  //returns unbiased sample variance 1/(n-1)*sum (x_i-x)^2
  if (vec.size() < 2 ) {
    ERRORfunc("Too few entries in argument vector<double>* vec : %i found, 2 or more required, returning 0!",vec.size());
    return 0;
  }
  double avg = getAverage(vec);
  double sqsum = 0;
  for (uint i=0;i<vec.size();i++) {
    sqsum += pow(vec.at(i)-avg,2);
  }
  sqsum /=(vec.size()-1);
  return sqsum;
}

//__________________________________________________________________________________|___________

double TQUtils::getSampleCovariance(std::vector<double>& vec1, std::vector<double>& vec2, bool /*verbose*/) {
  //returns unbiased sample covariance 1/(n-1)*sum (x_i-x)^2
  if (vec1.size() != vec2.size()) {
    ERRORfunc("Samples do not match in size, returning 0!");
    return 0;
  }
  if (vec1.size() < 2 ) {
    ERRORfunc("Too few entries in argument vector<double>& vec1 and vec2 : %i found, 2 or more required, returning 0!",vec1.size());
    return 0;
  }
 
  double avg1 = getAverage(vec1);
  double avg2 = getAverage(vec2);
  double sqsum = 0;
  for (uint i=0;i<vec1.size();i++) {//we checked that vec1.size == vec2.size before.
    sqsum += (vec1.at(i)-avg1)*(vec2.at(i)-avg2);
  }
  sqsum /=(vec1.size()-1);
  return sqsum;
}

//__________________________________________________________________________________|___________

double TQUtils::getSampleCorrelation(std::vector<double>& vec1, std::vector<double>& vec2, bool verbose) {
  if (vec1.size() != vec2.size()) {
    ERRORfunc("Samples do not match in size, returning 0!");
    return 0;
  }
  if (vec1.size() < 2 ) {
    ERRORfunc("Too few entries in argument vector<double>& vec1 and vec2 : %i found, 2 or more required, returning 0!",vec1.size());
    return 0;
  }
  double var1 = getSampleVariance(vec1);
  double var2 = getSampleVariance(vec2);
  if (var1 ==0 || var2 == 0) {
    if(verbose) WARNfunc("Variance of either sample is zero, returning NaN!");
    return std::numeric_limits<double>::quiet_NaN();
  }
  return getSampleCovariance(vec1,vec2,verbose)/sqrt(var1*var2);
}

//__________________________________________________________________________________|___________

double TQUtils::getAverageOrderOfMagnitude(TMatrixD* mat){
  // Calculates the average order of magnitude of the matrix entries through log10(value). 
  // If value is zero, the order of magnitude is assumed to be zero.
  // This version does not round at any step.
  if (!mat || mat->GetNrows() == 0 || mat->GetNcols() == 0) return 0.;
  double res = 0.;
  for (int i=0; i<mat->GetNrows(); ++i) {
    for (int j=0; j<mat->GetNcols(); ++j) {
      res += TQUtils::getOrderOfMagnitude((*mat)(i,j));
    }
  }
  return res/(mat->GetNrows()*mat->GetNcols());
}

//__________________________________________________________________________________|___________

int TQUtils::getAverageOrderOfMagnitudeInt(TMatrixD* mat){
  // Calculates the average order of magnitude of the matrix entries through log10(value). 
  // If value is zero, the order of magnitude is assumed to be zero.
  // This version rounds down to the next integer at each step (i.e. at each log10(value) as well as for the average)
  if (!mat || mat->GetNrows() == 0 || mat->GetNcols() == 0) return 0;
  int res = 0;
  for (int i=0; i<mat->GetNrows(); ++i) {
    for (int j=0; j<mat->GetNcols(); ++j) {
      res += TQUtils::getOrderOfMagnitudeInt((*mat)(i,j));
    }
  }
  return (int) res/(mat->GetNrows()*mat->GetNcols());
}





//__________________________________________________________________________________|___________

TList* TQUtils::execute(const TString& cmd, size_t maxOutputLength){
  // execute the commmand, returning a TList containing lines of output it produced
  FILE *out = popen(cmd.Data(), "r");
  if(!out) return NULL;
  char* line = (char*)malloc(maxOutputLength*sizeof(char));
  if(!line){ pclose(out); return NULL; }
  TList* l = new TList();
  l->SetOwner(true);
  bool esc = false;
  while (!feof(out)) {
    TString sLine = "";
    while(TQStringUtils::isASCII(sLine)) {
      if(!fgets(line, maxOutputLength, out)) {
        esc = true;
        break;
      }
      sLine += line;
      memset(line, 0, maxOutputLength*sizeof(char));
    }
    if (esc) break;
    TString s = TQStringUtils::trim(sLine,TQStringUtils::blanksAll);
    if(!s.IsNull())
      l->Add(new TObjString(s));
  }
  free(line); 
  pclose(out);
  return l;
}

//__________________________________________________________________________________|___________

TString TQUtils::findFileEOS(const TString& basepath, const TString& filename, const TString& eosprefix){
  // return the full valid path of a file residing on EOS, given the filename and a base folder
  // returns empty string if no match found
  TString path = eosprefix+TQFolder::concatPaths(basepath,filename);
  if(TQUtils::fileExists(path)){
    return path;
  }
  TQIterator itr(TQUtils::execute(TQLibrary::getEOScmd()+" ls "+basepath),true);
  while(itr.hasNext()){
    TObject* obj = itr.readNext();
    TString name = obj->GetName();
    path = TQFolder::concatPaths(basepath,name);
    TList* l = TQUtils::execute(TQLibrary::getEOScmd()+" stat "+path,1024);
    if(!l) return "";
    TString status = TQStringUtils::concat(l,"");
    delete l;
    if(status.IsNull())
      return "";
    TObjArray* a = status.Tokenize(" ");
    if(!a) return "";
    if(a->IsEmpty()){
      delete a;
      return "";
    }
    TString type = a->Last()->GetName();
    delete a;
    if(type == "directory"){
      TString retval = TQUtils::findFileEOS(path,filename,eosprefix);
      if(!retval.IsNull()){
        return retval;
      }
    } else if(type == "file"){
      if(filename.CompareTo(name) == 0){
        return eosprefix+TQFolder::concatPaths(basepath,name);
      }
    }
  }
  return "";
}

//__________________________________________________________________________________|___________

TString TQUtils::findFileLocal(const TString& basepath, const TString& filename){
  // return the full valid path of a local file, given the filename and a base folder
  // returns empty string if no match found
  TString path = TQFolder::concatPaths(basepath,filename);
  if(TQUtils::fileExists(path))
    return path;
  DIR* dirp = opendir(basepath.Data());
  dirent* dp;
  while ((dp = readdir(dirp)) != NULL){
    if(dp->d_type == DT_DIR){
      TString dname(dp->d_name);
      if(dname == "." || dname == "..") continue;
      TString subpath = TQFolder::concatPaths(basepath,dname);
      TString found = TQUtils::findFileLocal(subpath,filename);
      if(!found.IsNull()){
        closedir(dirp);
        return found;
      }
    } else {
      if(filename.CompareTo(dp->d_name) == 0)
        return TQFolder::concatPaths(basepath,dp->d_name);
    } 
  }
  closedir(dirp);
  return "";
}


//__________________________________________________________________________________|___________

TString TQUtils::findFile(const TString& basepath, const TString& filename){
  // return the full valid path of a file at any location, given the filename and a list of base folders
  // returns empty string if no match found
  if(basepath.BeginsWith("root://")){
    size_t pathpos = basepath.Index("/eos/");
    TString eosprefix = basepath(0,pathpos);
    TString path = basepath(pathpos,basepath.Length());
    return TQUtils::findFileEOS(path,filename,eosprefix);
  } else {
    return TQUtils::findFileLocal(basepath,filename);
  }
  return "";
}


//__________________________________________________________________________________|___________

TString TQUtils::findFile(TList* basepaths, const TString& filename, int& index){
  // return the full valid path of a file, given the filename and a list of base folders
  // returns empty string if no match found
  if(!basepaths || filename.IsNull()){
    return "";
  }
  TQIterator itr(basepaths);
  index = 0;
  while(itr.hasNext()){
    index++;
    TObject* obj=itr.readNext();
    TString result = TQUtils::findFile(obj->GetName(),filename);
    if(!result.IsNull()){
      return result;
    }
  }
  return "";
}

//__________________________________________________________________________________|___________

TString TQUtils::findFile(TList* basepaths, const TString& filename){
  // return the full valid path of a file, given the filename and a list of base folders
  // returns empty string if no match found
  int index;
  return TQUtils::findFile(basepaths,filename,index);
}

//__________________________________________________________________________________|___________

TList* TQUtils::ls(TString exp){
  // return a list of files matching the expression
  // if used on local files, the output is equivalent to bash "ls"
  // 
  // this function also supports remote file systems
  // - for EOS, please make sure that TQLibrary knows about
  // the location of the "eos.select" binary
  // if not properly detected by the Makefile, this can be 
  // edited by calling TQLibrary::getQLibary()->setEOScmd("...")
  // please prefix EOS paths with "root://"
  // - for Grid dCache, please make sure that TQLibrary knows about
  // * the name of the local group disk 
  // set with TQLibary::getQLibrary()->setLocalGroupDisk("...")
  // * a dq2 suite (default "dq2")
  // set with TQLibary::getQLibrary()->setDQ2cmd("...")
  // * if necessary, path head replacements. Set with
  // TQLibrary::getQLibrary()->setDQ2Head() for the old head to be removed
  // TQLibrary::getQLibrary()->setdCacheHead() for the new head to be prepended
  // all of these values can be set in a "locals.h" defintion file, for example, put
  // #define LOCALGROUPDISK "UNI-FREIBURG_LOCALGROUPDISK"
  // #define DQ2PATHHEAD "srm://se.bfg.uni-freiburg.de"
  // #define DCACHEPATHHEAD "dcap://se.bfg.uni-freiburg.de:22125"
  // please prefix dataset expressions with "dCache:"
  if(exp.BeginsWith("root://")){
    size_t pathpos = exp.Index("/eos/")+1;
    TString eosprefix = exp(0,pathpos);
    TQStringUtils::ensureTrailingText(eosprefix,"//");
    TString path = exp(pathpos,exp.Length());
    return TQUtils::lsEOS(path,eosprefix);
  } else if(TQStringUtils::removeLeadingText(exp,"dCache:")){
    return TQUtils::lsdCache(exp, TQLibrary::getLocalGroupDisk(), TQLibrary::getDQ2PathHead(), TQLibrary::getdCachePathHead(), TQLibrary::getDQ2cmd());
  } else {
    return TQUtils::lsLocal(exp);
  }
  return NULL; 
}

//__________________________________________________________________________________|___________

TList* TQUtils::lsEOS(TString exp, const TString& eosprefix, TString path){
  // return a full list of all files in the given location on EOS
  // please make sure that TQLibrary knows about the location of the
  // binary "eos.select". 
  // if not properly detected by the Makefile, this can be 
  // edited by calling TQLibrary::getQLibary()->setEOScmd("...")
  TList* retval = new TList();
  retval->SetOwner(true);
  while(!exp.IsNull()){
    TString head = TQFolder::getPathHead(exp);
    if(head.First("*?") == kNPOS){
      path = TQFolder::concatPaths(path,head);
    } else {
      TList* l = TQUtils::execute(TQLibrary::getEOScmd()+" ls "+path);
      TQIterator itr(l,true);
      while(itr.hasNext()){
        TObject* obj = itr.readNext();
        if(TQStringUtils::matches(obj->GetName(),head)){
          if(exp.IsNull()){
            retval->Add(new TObjString(eosprefix+TQFolder::concatPaths(path,obj->GetName())));
          } else {
            TList* matches = TQUtils::lsEOS(exp,eosprefix,path);
            TQListUtils::moveListContents(matches,retval);
            delete matches;
          }
        }
      }
      return retval;
    }
  }
  retval->Add(new TObjString(eosprefix+path));
  return retval;
}

//__________________________________________________________________________________|___________

TList* TQUtils::lsdCache(const TString& fname, const TString& localGroupDisk, const TString& oldHead, const TString& newHead, const TString& dq2cmd){
  // return a full list of all files locally available in the dCache
  // the dataset and localGroupDisk names are forwarded to dq2-ls
  // the dq2-command can be set as optional parameter (use, e.g. 'dq2' or 'rucio')
  // if oldHead and newHead are given, the resulting path head will be exchanged accordingly
  TList* retval = new TList();
  retval->SetOwner(true);
  if(fname.IsNull()) return retval;
  std::stringstream s;
  s << dq2cmd << "-ls -L " << localGroupDisk << " -fpD " << fname;
  TList* items = TQUtils::execute(s.str().c_str(),1024);
  items->SetOwner(true);
  TQStringIterator itr(items,true);
  while(itr.hasNext()){
    TObjString* obj = itr.readNext();
    if(!obj) continue;
    TString str(obj->GetName());
    if(!oldHead.IsNull()) if(!TQStringUtils::removeLeadingText(str,oldHead)) continue;
    if(!newHead.IsNull()) str.Prepend(newHead);
    retval->Add(new TObjString(str));
  }
  return retval;
}

//__________________________________________________________________________________|___________

TList* TQUtils::lsLocal(const TString& exp){
  // execute an "ls" command locally and return the result as a TList
  return TQUtils::execute(TString::Format("ls %s",exp.Data()),1024);
}

//__________________________________________________________________________________|___________

TList* TQUtils::getObjectsFromFile(TString identifier, TClass* objClass){
  // the identifier is formatted "filename:objectname"
  // retrieves a list of all objects in the given file
  // that have a key matching the given class
  // and object name pattern (may contain wildcards)
  TString filename;
  TQStringUtils::readUpTo(identifier,filename,":");
  TQStringUtils::removeLeading(identifier,":");
  TString objname = identifier;
  return TQUtils::getObjectsFromFile(filename,objname,objClass);
}

//__________________________________________________________________________________|___________

TList* TQUtils::getObjectsFromFile(const TString& filename, const TString& objName, TClass* objClass){
  // retrieves a list of all objects in the given file
  // that have a key matching the given class
  // and object name pattern (may contain wildcards)
  TFile* f = TFile::Open(filename,"READ");
  gDirectory->cd();
  if(!f || !f->IsOpen()){
    ERRORfunc("unable to open file '%s!",filename.Data());
    if(f) delete f;
    return NULL;
  }
  TList* retval = TQUtils::getObjects(objName,objClass,f);
  TQIterator itr(retval);
  while(itr.hasNext()){
    TObject* o = itr.readNext();
    f->Remove(o);//Detach the object from the file, otherwise it's gone once we close the file! (SetDirectory(0) is not implemented for TObject, but only for (some) derived classes, this basically emulates the functionality)
    TQFolder* folder = dynamic_cast<TQFolder*>(o);
    if(folder) folder->setDirectory(gDirectory);
  }
  f->Close();
  delete f;
  return retval;
}

//__________________________________________________________________________________|___________

TList* TQUtils::getObjects(const TString& objName, TClass* objClass, TDirectory* d){
  // retrieves a list of all objects in the given directory
  // that have a key matching the given class
  // and object name pattern (may contain wildcards)
  if(!d) return NULL;
  if(!objClass) return NULL;
  TQIterator itr(d->GetListOfKeys());
  std::map<TString,bool> map;
  TList* retval = new TList();
  while(itr.hasNext()){
    TKey* key = (TKey*)(itr.readNext());
    TClass* c = TClass::GetClass(key->GetClassName());
    if(!c) continue;
    if(!c->InheritsFrom(objClass))
      continue;
    TString name = key->GetName();
    if(!TQStringUtils::matches(name,objName))
      continue;
    if(map.find(name) != map.end())
      continue;
    TObject* obj = d->Get(name);
    if(!obj) continue;
    TQTaggable* taggable = dynamic_cast<TQTaggable*>(obj);
    if(taggable){
      taggable->setTagString(".origin",d->GetName());
      taggable->setTagString(".key",name);
    }
    map[name] = true;
    retval->Add(obj);
  }
  return retval;
}

//__________________________________________________________________________________|___________

void TQUtils::printBranches(TTree* t){
  // print the branches of a tree in a nice format
  TCollection* branches = t->GetListOfBranches();
  int width = TQListUtils::getMaxNameLength(branches);
  int typewidth = 40;
  TQIterator itr(branches);
  std::cout << TQStringUtils::makeBoldWhite(TQStringUtils::fixedWidth(t->GetName(),width,false)) << " " << TQStringUtils::makeBoldWhite("status") << " " << TQStringUtils::makeBoldWhite(TQStringUtils::fixedWidth("type",typewidth,false)) << std::endl;
  while(itr.hasNext()){
    TBranch* obj = (TBranch*)(itr.readNext());
    std::cout << TQStringUtils::fixedWidth(obj->GetName(),width,false);
    std::cout << " " << t->GetBranchStatus(obj->GetName());
    std::cout << " " << TQStringUtils::fixedWidth(obj->GetClassName(),typewidth,false) << std::endl;
  }
}

//__________________________________________________________________________________|___________

void TQUtils::printActiveBranches(TTree* t){
  // print the active branches of a tree in a nice format
  TCollection* branches = t->GetListOfBranches();
  int width = TQListUtils::getMaxNameLength(branches);
  int typewidth = 40;
  int count = 0;
  TQIterator itr(branches);
  std::cout << TQStringUtils::makeBoldWhite(TQStringUtils::fixedWidth(t->GetName(),width,false)) << " " << TQStringUtils::makeBoldWhite(TQStringUtils::fixedWidth("type",typewidth,false)) << std::endl;
  while(itr.hasNext()){
    TBranch* obj = (TBranch*)(itr.readNext());
    if(t->GetBranchStatus(obj->GetName())){
      std::cout << TQStringUtils::fixedWidth(obj->GetName(),width,false);
      std::cout << " " << TQStringUtils::fixedWidth(obj->GetClassName(),typewidth,false) << std::endl;
      count++;
    }
  }
  if(count < 1){
    std::cout << "< no active branches >" << std::endl;
  }
}

//__________________________________________________________________________________|___________

#ifdef HAS_XAOD
void printProductionChainLocal(const xAOD::TruthParticle* p, size_t indent){
  if(!p) return;
  for(size_t i=0; i<indent; ++i){
    std::cout << "  ";
  }
  std::cout << "idx=" << p->index() << ", id=" << p->pdgId() << ", status=" << p->status();
  if(p->nParents() == 0){
    std::cout << " (initial)" << std::endl;
  } else {
    std::cout << std::endl;
    for(size_t i=0; i<p->nParents(); ++i){
      printProductionChainLocal(p->parent(i),indent+1);
    }
  }
}
#endif


void TQUtils::printProductionChain(TTree* t, Long64_t evt, int ptcl, const TString& bName){
#ifdef HAS_XAOD
  if(!t->GetCurrentFile()){
    ERRORfunc("cannot read trees from memory - please use a file-based TTree instead. are you using a TransientTree?");
    return;
  }
  xAOD::TEvent event(t, xAOD::TEvent::kClassAccess);
  const xAOD::TruthParticleContainer* truthrecord = 0;
  event.getEntry( evt );
  if(event.retrieve(truthrecord, bName.Data()).isFailure()){
    ERRORfunc("unable to retrieve event!");
    return;
  }
  const xAOD::TruthParticle* p = truthrecord->at(ptcl);
  printProductionChainLocal(p,0);
#endif
}

#ifdef HAS_XAOD
void printDecayChainLocal(const xAOD::TruthParticle* p, size_t indent){
  if(!p) return;
  for(size_t i=0; i<indent; ++i){
    std::cout << "  ";
  }
  std::cout << "idx=" << p->index() << ", id=" << p->pdgId() << ", status=" << p->status();
  if(p->nChildren() == 0){
    std::cout << " (final)" << std::endl;
  } else {
    std::cout << std::endl;
    for(size_t i=0; i<p->nChildren(); ++i){
      printDecayChainLocal(p->child(i),indent+1);
    }
  }
}
#endif

void TQUtils::printDecayChain(TTree* t, Long64_t evt, int ptcl, const TString& bName){
#ifdef HAS_XAOD
  if(!t->GetCurrentFile()){
    ERRORfunc("cannot read trees from memory - please use a file-based TTree instead. are you using a TransientTree?");
    return;
  }
  xAOD::TEvent event(t, xAOD::TEvent::kClassAccess);
  const xAOD::TruthParticleContainer* truthrecord = 0;
  event.getEntry( evt );
  if(event.retrieve(truthrecord, bName.Data()).isFailure()){
    ERRORfunc("unable to retrieve event!");
    return;
  }
  const xAOD::TruthParticle* p = truthrecord->at(ptcl);
  printDecayChainLocal(p,0);
#endif
}


void TQUtils::printParticles(TTree* t, Long64_t evt, int id, const TString& bName){
#ifdef HAS_XAOD
  if(!t->GetCurrentFile()){
    ERRORfunc("cannot read trees from memory - please use a file-based TTree instead. are you using a TransientTree?");
    return;
  }
  xAOD::TEvent event(t, xAOD::TEvent::kClassAccess);
  const xAOD::TruthParticleContainer* truthrecord = 0;
  event.getEntry( evt );
  if(event.retrieve(truthrecord, bName.Data()).isFailure()){
    ERRORfunc("unable to retrieve event!");
    return;
  }
  for(size_t i=0; i<truthrecord->size(); ++i){
    const xAOD::TruthParticle* p = truthrecord->at(i);
    if(p && p->pdgId() == id)
      std::cout << "idx=" << i << ", " << "id=" << p->pdgId() << ", status=" << p->status() << std::endl;
  }
#endif
}


void TQUtils::printParticle(TTree* t, Long64_t evt, int index, const TString& bName){
#ifdef HAS_XAOD
  if(!t->GetCurrentFile()){
    ERRORfunc("cannot read trees from memory - please use a file-based TTree instead. are you using a TransientTree?");
    return;
  }
  xAOD::TEvent event(t, xAOD::TEvent::kClassAccess);
  const xAOD::TruthParticleContainer* truthrecord = 0;
  event.getEntry( evt );
  if(event.retrieve(truthrecord, bName.Data()).isFailure()){
    ERRORfunc("unable to retrieve event!");
    return;
  }
  if(index >= (int)(truthrecord->size())){
    ERRORfunc("cannot access particle %d, truth record only has %d entries",index,(int)(truthrecord->size()));
    return;
  }
  const xAOD::TruthParticle* p = truthrecord->at(index);
  std::cout << "idx=" << index << ", " << "id=" << p->pdgId() << ", status=" << p->status() << ", pt=" << p->pt() << ", eta=" << p->eta() << ", phi=" << p->phi() << ", E=" << p->e() << ", m=" << p->m() << ", px=" << p->px() << ", py=" << p->py() << ", pz=" << p->pz() << std::endl;
#endif
}

//__________________________________________________________________________________|___________

double TQUtils::getValue(TTree* t, const TString& bname, Long64_t iEvent){
  // evaluate an expression on a specific event of a TTree
  if(!t || bname.IsNull() || iEvent < 0) return std::numeric_limits<double>::quiet_NaN();
  t->SetBranchStatus("*",1);
  TTreeFormula* f = new TTreeFormula("tmp",bname,t);
  t->GetEvent(iEvent);
  double val = f->EvalInstance(0.);
  delete f;
  return val;
}

//__________________________________________________________________________________|___________

double TQUtils::getSum(TTree* t, const TString& bname){
  // evaluate an expression on all events of a TTree and return the sum
  if(!t || bname.IsNull()) return std::numeric_limits<double>::quiet_NaN();
  TCollection* branches = TQUtils::getBranchNames(bname);
  TQIterator it(branches);
  t->SetBranchStatus("*",0);
  while(it.hasNext()){
    t->SetBranchStatus(it.readNext()->GetName(),1);
  }
  TTreeFormula* f = new TTreeFormula("tmp",bname,t);
  Long64_t nEvents = t->GetEntries();
  double val = 0;
  for(Long64_t iEvent =0; iEvent<nEvents; ++iEvent){
    t->GetEvent(iEvent);
    val += f->EvalInstance(0.);
  }
  delete f;
  return val;
}

//__________________________________________________________________________________|___________

#ifdef HAS_XAOD
double TQUtils::xAODMetaDataGetSumWeightsInitialFromEventBookkeeper(TTree* metaData){
  // retrieve the initial sum of weights from the event bookkeepers in this tree
  // thanks to a dirty hack, this also works in standalone mode
  if (!metaData) return std::numeric_limits<double>::quiet_NaN();
  TTreeFormula tf("tf","EventBookkeepers.m_nWeightedAcceptedEvents",metaData);
  metaData->LoadTree(0);
  tf.UpdateFormulaLeaves();
  tf.GetNdata();
  double initialSumOfWeightsInThisFile = tf.EvalInstance(1);
  return initialSumOfWeightsInThisFile;
}

namespace TQUtils {
  xAOD::TEvent* xAODMakeEvent(TFile* file){
    if (!file){
      throw std::runtime_error("invalid file");
    }
    xAOD::TEvent* event = new xAOD::TEvent(xAOD::TEvent::kClassAccess);
    if(!event->readFrom(file).isSuccess()){
      throw(TString::Format("unable to read tree from file '%s' into xAOD::TEvent -- possible version incompatibilities?",file->GetName()).Data());
    }
    return event;
  }
  
  namespace {
    const xAOD::CutBookkeeper* findCutBookkeeper(xAOD::CutBookkeeperContainer* cont, const std::string& name, bool useMaxCycle, const std::string& inputStream){
      if (!cont->size()) {
        throw std::runtime_error("Size of CutBookkeeper container equal to 0");
      }
      // Recommendation is to use the highest cycle for initial events/sum-of-weigths
      int maxCycle = -1;
      const xAOD::CutBookkeeper* c = 0;
      for (const xAOD::CutBookkeeper* cbk: *cont) {
        if(!cbk) {
          WARNfunc("Found invalid CutBookkeeper in container.");
          continue;
        }
        try {
          // This approach prevents errors if for some reason the highest cycle has no entry for "AllExecutedEvents" (which doesn't necessarily points to a problem, maybe add a warning?)
          if ( (!useMaxCycle || (cbk->cycle() > maxCycle)) && (cbk->name() == name) && (cbk->inputStream() == inputStream) ) {
            c = cbk;
            maxCycle = cbk->cycle();
          }
        } catch (const std::exception& e){
          throw std::runtime_error(TString::Format("unable to access content of CutBookkeeper. %s",e.what()).Data());
        }
      }
      if(!c) {
        throw std::runtime_error("Reaching end of function without having found CutBookkeeper result.");
      }

      return c;
    }
  }
  
  xAOD::CutBookkeeper::Payload xAODMetaDataGetCutBookkeeper(xAOD::TEvent& event, const char* container, const char* bookkeeper, const char* kernelname){
    // First, check that we have no incomplete bookkeepers.
    // This would be an indication that somewhere in the processing
    // chain, an input file was not completely processed. You should 
    // thus disregard the current file as it is not trustworthy.
    const xAOD::CutBookkeeperContainer* constCont = NULL;
    if(event.retrieveMetaInput(constCont, "CutBookkeepers").isFailure()){
      throw std::runtime_error("unable to retrieve event!");
    }
    // due to problems connecting the containers to their aux data store, we need to do this manually
    xAOD::CutBookkeeperContainer* cont = const_cast<xAOD::CutBookkeeperContainer*>(constCont);
//     const xAOD::CutBookkeeperContainer* incompleteCBC = 0;
//     if(event.retrieveMetaInput(incompleteCBC, "IncompleteCutBookkeepers").isSuccess()){
//       xAOD::CutBookkeeperContainer* incomplete = const_cast<xAOD::CutBookkeeperContainer*>(incompleteCBC);
//       if ( incomplete->size() != 0 ) {
//         // for now, we just skip files that have incomplete bookkeepers
//         WARNfunc("encountered incomplete cut bookkeepers in not fully processed input file '%s'",event.m_inTree->GetCurrentFile()->GetName());
//         const xAOD::CutBookkeeper* allexecutedevents = findCutBookkeeper(incomplete,bookkeeper,true,container);
//         const xAOD::CutBookkeeper* kernel = findCutBookkeeper(incomplete,kernelname,false,container); // TODO: find input stream name automatically
//         const xAOD::CutBookkeeper* allseenevents = findCutBookkeeper(cont,bookkeeper,true,container); 
//         int seen = allseenevents->payload().nAcceptedEvents;
//         if(seen == 0){
//           throw std::runtime_error("no events listed in incomplete CutBookkkeeper!");
//         }
//         if(!kernel){
//           throw std::runtime_error("unable to find derivation kernel in incomplete CutBookkeepers!");
//         }
//         double inv_efficiency = (double)(kernel->payload().nAcceptedEvents) / seen;
//         xAOD::CutBookkeeper::Payload retval(allexecutedevents->payload());
//         retval.nAcceptedEvents *= inv_efficiency;
//         retval.sumOfEventWeights *= inv_efficiency;
//         retval.sumOfEventWeightsSquared *= inv_efficiency; // this is probably wrong, but won't be read out anyway
//         return retval;
//       }
//     }
    // in the xAOD EDM, the CutBookkeeper bookkeeper contains the initial processing
    const xAOD::CutBookkeeper* c = findCutBookkeeper(cont,bookkeeper,true,container);
    if(!c){
      throw ("unable to find CutBookkeeper 'AllExecutedEvents'");
    }
    return c->payload();
  }
}

double TQUtils::xAODMetaDataGetSumWeightsInitialFromCutBookkeeper(xAOD::TEvent& event, const char* container, const char* bookkeeper, const char* kernelname){
  // retrieve the initial sum of weights from the cut bookkeepers in this file
  try {
    // xAOD::TEvent event(TQUtils::xAODMakeEvent(file));
    xAOD::CutBookkeeper::Payload payload(TQUtils::xAODMetaDataGetCutBookkeeper(event,container,bookkeeper,kernelname));
    return payload.sumOfEventWeights;
  } catch (const std::exception& e){
    ERRORfunc(e.what());
  }
  return std::numeric_limits<double>::quiet_NaN();
}
double TQUtils::xAODMetaDataGetNEventsInitialFromCutBookkeeper(xAOD::TEvent& event, const char* container, const char* bookkeeper, const char* kernelname){
  // retrieve the initial number of events from the cut bookkeepers in this file
  try {
    // xAOD::TEvent event(TQUtils::xAODMakeEvent(file));
    xAOD::CutBookkeeper::Payload payload(TQUtils::xAODMetaDataGetCutBookkeeper(event,container,bookkeeper,kernelname));
    return payload.nAcceptedEvents;
  } catch (const std::exception& e){
    ERRORfunc(e.what());
  }
  return std::numeric_limits<double>::quiet_NaN();
}
#endif

//__________________________________________________________________________________|___________

TList* TQUtils::getListOfFoldersInFile(const TString& filename, TClass* type){
  // retrieve the list of TQFolders contained in a given file
  // if the class type is given, only TQFolder instances 
  // of the given class type are considered
  TList* list = new TList();
  TQUtils::appendToListOfFolders(filename, list, type);
  return list;
}

//__________________________________________________________________________________|___________

int TQUtils::appendToListOfFolders(const TString& filename, TList* list, TClass* type){
  // append all TQfolders contained in a given file to a list
  // if the class type is given, only TQFolder instances 
  // of the given class type are considered
  if(!type || !list || !type->InheritsFrom(TQFolder::Class())) return 0;
  TFile* f = TFile::Open(filename,"READ");
  if(!f) return 0;
  if(!f->IsOpen()) { delete f; return 0; };
  TQIterator* itr = new TQIterator(f->GetListOfKeys());
  int num = 0;
  while(itr->hasNext()){
    TKey* key = (TKey*)(itr->readNext());
    TClass * c = TClass::GetClass(key->GetClassName());
    if(!c) continue;
    if(!c->InheritsFrom(type))
      continue;
    TString name = key->GetName();
    if(name.First('-') != kNPOS)
      continue;
    TQFolder* folder = dynamic_cast<TQFolder*>(f->Get(name));
    if(!folder) continue;
    folder->setTagString(".key",name);
    folder->resolveImportLinks();
    list->Add(folder);
    num++;
  }
  f->Close();
  return num;
}
 
//__________________________________________________________________________________|___________

TList* TQUtils::getListOfFolderNamesInFile(const TString& filename, TClass* type){
  // retrieve the list of names of TQFolders contained in a given file
  // if the class type is given, only TQFolder instances 
  // of the given class type are considered
  TList* list = new TList();
  TQUtils::appendToListOfFolderNames(filename, list, type);
  return list;
}

//__________________________________________________________________________________|___________

int TQUtils::appendToListOfFolderNames(const TString& filename, TList* list, TClass* type, const TString& prefix){
  // append all names of TQFolders contained in a given file to a list
  // if the class type is given, only TQFolder instances 
  // of the given class type are considered
  // the prefix is prepended to every single name
  if(!type || !list || !type->InheritsFrom(TQFolder::Class())) return 0;
  TFile* f = TFile::Open(filename,"READ");
  if(!f) return 0;
  if(!f->IsOpen()) { delete f; return 0; };
  TQIterator* itr = new TQIterator(f->GetListOfKeys());
  int num = 0;
  while(itr->hasNext()){
    TKey* key = (TKey*)(itr->readNext());
    TClass * c = TClass::GetClass(key->GetClassName());
    if(!c) continue;
    if(!c->InheritsFrom(type))
      continue;
    TString name = key->GetName();
    if(name.First('-') != kNPOS)
      continue;
    list->Add(new TObjString(prefix+name));
    num++;
  }
  f->Close();
  return num;
}


//__________________________________________________________________________________|___________

double TQUtils::roundAuto(double d, int nSig){
  // round some double to a number of significant digits
  if(d == 0 || !TQUtils::isNum(d)) return d;
  int ndigits = floor(log10(fabs(d))) + 1 - nSig;
  if(fabs(d / pow(10,ndigits)) < 2) ndigits--;
  return pow(10.,ndigits)*floor(d / pow(10,ndigits)+0.5);
}

//__________________________________________________________________________________|___________

double TQUtils::roundAutoDown(double d, int nSig){
  // round down some double to a number of significant digits
  if(d == 0 || !TQUtils::isNum(d)) return d;
  int ndigits = floor(log10(fabs(d))) + 1 - nSig;
  if(fabs(d / pow(10,ndigits)) < 2) ndigits--;
  return pow(10.,ndigits)*floor(d / pow(10,ndigits));
}

//__________________________________________________________________________________|___________

double TQUtils::roundAutoUp(double d, int nSig){
  // round up some double to a number of significant digits
  if(d == 0 || !TQUtils::isNum(d)) return d;
  int ndigits = floor(log10(fabs(d))) + 1 - nSig;
  if(fabs(d / pow(10,ndigits)) < 2) ndigits--;
  return pow(10.,ndigits)*ceil(d / pow(10,ndigits));
}


//__________________________________________________________________________________|___________

double TQUtils::round(double d, int ndigits){
  // round some double to a number of digits
  return pow(10.,-ndigits)*floor(d * pow(10,ndigits)+0.5);
}

//__________________________________________________________________________________|___________

double TQUtils::roundDown(double d, int ndigits){
  // round down some double to a number of digits
  return pow(10.,-ndigits)*floor(d * pow(10,ndigits));
}

//__________________________________________________________________________________|___________

double TQUtils::roundUp(double d, int ndigits){
  // round up some double to a number of digits
  return pow(10.,-ndigits)*ceil(d * pow(10,ndigits));
}

//__________________________________________________________________________________|___________

TString TQUtils::getMD5(const TString& filepath){
  // retrieve the MD5 sum (hex value) of some file as a TString
  if(filepath.BeginsWith("root://eos")){
    return TQUtils::getMD5EOS(filepath);
  }
  return TQUtils::getMD5Local(filepath);
}

//__________________________________________________________________________________|___________

TString TQUtils::getMD5Local(const TString& filepath){
  // retrieve the MD5 sum (hex value) of some local file as a TString
  TMD5* sum = TMD5::FileChecksum(filepath);
  if(!sum) return "";
  TString retval(sum->AsString());
  delete sum;
  return retval;
}

//__________________________________________________________________________________|___________

TString TQUtils::getMD5EOS(TString filepath){
  // retrieve the MD5 sum (hex value) of some EOS file as a TString
  if(!(TQStringUtils::removeLeadingText(filepath,"root://eos") && 
       (TQStringUtils::removeLeadingText(filepath,"user") || TQStringUtils::removeLeadingText(filepath,"atlas")) && 
       TQStringUtils::removeLeadingText(filepath,"/"))) return "<unknown>";
  filepath.Prepend(" stat ");
  filepath.Prepend(TQLibrary::getEOScmd());
  filepath.Append(" | md5sum | cut -f 1 -d ' '");
  TQLibrary::redirect("/dev/null");
  TList* ret = TQUtils::execute(filepath);
  if(ret->GetEntries() != 1){
    delete ret;
    return "<error>";
  }
  TQLibrary::restore();
  TString retval(ret->First()->GetName());
  delete ret;
  return retval;
}

//__________________________________________________________________________________|___________

TString TQUtils::getModificationDate(TFile* f){
  // return the modification date of a TFile as a TString
  if(!f) return "";
  TDatime t(f->GetModificationDate());
  return t.AsString();
}


//__________________________________________________________________________________|___________

unsigned long TQUtils::getFileSize(const TString& path) {
  struct stat st;
  if(stat(path.Data(), &st) != 0) {
      return 0.;
  }
  return st.st_size;
}

//__________________________________________________________________________________|___________
 
void TQUtils::dumpTop(TString basepath, TString prefix, TString message ) {
  TQUtils::ensureDirectory(basepath);
  TString file = TString::Format("%s/%s_topDump.txt",basepath.Data(),prefix.Data());
  TQUtils::execute(TString::Format("echo \"%s\" > %s ",message.Data(),file.Data() ));
  TQUtils::execute(TString::Format("top -n 1 -u $(whoami) >> %s",file.Data() ));
  return;
}

// the following functions (getPeakRSS and getCurrentRSS) are based on work of David Robert Nadeau (http://NadeauSoftware.com/) provided under Creative Commons Attribution 3.0 Unported License. Origin: http://nadeausoftware.com/articles/2012/07/c_c_tip_how_get_process_resident_set_size_physical_memory_use 
#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
#include <unistd.h>
#include <sys/resource.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
#include <fcntl.h>
#include <procfs.h>

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
#include <stdio.h>

#endif

#else
#warning "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS, they will not return meaningfull values!"
#endif

/**
 * Returns the peak (maximum so far) resident set size (physical
 * memory use) measured in bytes, or zero if the value cannot be
 * determined on this OS.
 */
size_t TQUtils::getPeakRSS( ) {
#if defined(_WIN32)
	/* Windows -------------------------------------------------- */
	PROCESS_MEMORY_COUNTERS info;
	GetProcessMemoryInfo( GetCurrentProcess( ), &info, sizeof(info) );
	return (size_t)info.PeakWorkingSetSize;

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
	/* AIX and Solaris ------------------------------------------ */
	struct psinfo psinfo;
	int fd = -1;
	if ( (fd = open( "/proc/self/psinfo", O_RDONLY )) == -1 )
		return (size_t)0L;		/* Can't open? */
	if ( read( fd, &psinfo, sizeof(psinfo) ) != sizeof(psinfo) )
	{
		close( fd );
		return (size_t)0L;		/* Can't read? */
	}
	close( fd );
	return (size_t)(psinfo.pr_rssize * 1024L);

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
	/* BSD, Linux, and OSX -------------------------------------- */
	struct rusage rusage;
	getrusage( RUSAGE_SELF, &rusage );
#if defined(__APPLE__) && defined(__MACH__)
	return (size_t)rusage.ru_maxrss;
#else
	return (size_t)(rusage.ru_maxrss * 1024L);
#endif

#else
	/* Unknown OS ----------------------------------------------- */
	return (size_t)0L;			/* Unsupported. */
#endif
}

/**
 * Returns the current resident set size (physical memory use) measured
 * in bytes, or zero if the value cannot be determined on this OS.
 */
size_t TQUtils::getCurrentRSS( ) {
#if defined(_WIN32)
	/* Windows -------------------------------------------------- */
	PROCESS_MEMORY_COUNTERS info;
	GetProcessMemoryInfo( GetCurrentProcess( ), &info, sizeof(info) );
	return (size_t)info.WorkingSetSize;

#elif defined(__APPLE__) && defined(__MACH__)
	/* OSX ------------------------------------------------------ */
	struct mach_task_basic_info info;
	mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
	if ( task_info( mach_task_self( ), MACH_TASK_BASIC_INFO,
		(task_info_t)&info, &infoCount ) != KERN_SUCCESS )
		return (size_t)0L;		/* Can't access? */
	return (size_t)info.resident_size;

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
	/* Linux ---------------------------------------------------- */
	long rss = 0L;
	FILE* fp = NULL;
	if ( (fp = fopen( "/proc/self/statm", "r" )) == NULL )
		return (size_t)0L;		/* Can't open? */
	if ( fscanf( fp, "%*s%ld", &rss ) != 1 )
	{
		fclose( fp );
		return (size_t)0L;		/* Can't read? */
	}
	fclose( fp );
	return (size_t)rss * (size_t)sysconf( _SC_PAGESIZE);

#else
	/* AIX, BSD, Solaris, and Unknown OS ------------------------ */
	return (size_t)0L;			/* Unsupported. */
#endif
}


