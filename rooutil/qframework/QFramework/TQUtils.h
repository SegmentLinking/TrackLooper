//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQUtils__
#define __TQUtils__

#include "QFramework/TQFlags.h"

#include "TObject.h"
#include "QFramework/TQFolder.h"
#include "TString.h"
#include "TDirectory.h"
#include "TClass.h"
#include "TCollection.h"
#include "QFramework/TQTaggable.h"
#include "TTree.h"
#include "TMatrixD.h"
#include "math.h"
#include <sys/stat.h>

#include "QFramework/ASG.h"
#ifdef HAS_XAOD
namespace xAOD { //EXCLUDE
  class TEvent;
}
#endif

namespace TQUtils {
  int getMinIndex(const std::vector<int>& vals, const std::vector<bool>& accepts);
  int getMinIndex(int val0, int val1, int val2, bool accept0, bool accept1, bool accept2);

  TFile * openFile(const TString& filename);
  bool fileExists(const TString& filename);
  bool fileExistsLocal(const TString& filename);
  bool fileExistsEOS(const TString& filename);
  TList * getListOfFilesMatching(TString pattern);
  bool ensureDirectoryForFile(TString filename);
  bool ensureDirectory(TString path);

  bool areEquivalent(TObject * obj1, TObject * obj2);

  /* to be moved to TQStringUtils */
 
  inline int sgn(int val){ return (val<0 ? -1 : 1); }
  inline double sgn(double val){ return (val<0 ? -1. : 1.); }
 
  bool parseAssignment(TString input, TString * dest, TString * source);
 
  TObjArray * getBranchNames(const TString& input);
 
  bool isNum(double a);
  bool inRange(double x, double a, double b);
 
  unsigned long getCurrentTime();
  TString getTimeStamp(const TString& format = "%c", size_t len = 80);
  TString getMD5(const TString& filepath);
  TString getMD5Local(const TString& filepath);
  TString getMD5EOS(TString filepath);
  TString getModificationDate(TFile* f);
  
  unsigned long getFileSize(const TString& path);
 
  double convertXtoNDC(double x);
  double convertYtoNDC(double y);
  double convertXtoPixels(double x);
  double convertYtoPixels(double y);
  double convertXfromNDC(double x);
  double convertYfromNDC(double y);
  double convertdXtoNDC(double x);
  double convertdYtoNDC(double y);
  double convertdXtoPixels(double x);
  double convertdYtoPixels(double y);
  double convertdXfromNDC(double x);
  double convertdYfromNDC(double y);

  double getAverage(std::vector<double>& vec);
  double getSampleVariance(std::vector<double>& vec);
  double getSampleCovariance(std::vector<double>& vec1, std::vector<double>& vec2, bool verbose=true);
  double getSampleCorrelation(std::vector<double>& vec1, std::vector<double>& vec2, bool verbose=true);
  
  inline double getOrderOfMagnitude(double value) {return value==0 ? 0. : log10(fabs(value)) ;}
  inline int getOrderOfMagnitudeInt(double value) {return int (value==0 ? 0. : log10(fabs(value))) ;}
  double getAverageOrderOfMagnitude(TMatrixD* mat);
  int getAverageOrderOfMagnitudeInt(TMatrixD* mat);
  
 
  char getLastCharacterInFile(const char* filename);
  bool ensureTrailingNewline(const char* filename);
 
  TString findFile(const TString& basepath, const TString& filename);
  TString findFileLocal(const TString& basepath, const TString& filename);
  TString findFileEOS(const TString& basepath, const TString& filename, const TString& eosprefix);
  TString findFile(TList* basepaths, const TString& filename, int& index);
  TString findFile(TList* basepaths, const TString& filename);
  TList* execute(const TString& cmd, size_t maxOutputLength = 80);

  TList* lsEOS(TString exp, const TString& eosprefix, TString path="");
  TList* lsdCache(const TString& fname, const TString& localGroupDisk, const TString& oldHead, const TString& newHead, const TString& dq2cmd = "dq2");
  TList* lsLocal(const TString& exp);
  TList* ls(TString exp);

  TList* getObjectsFromFile(TString identifier, TClass* objClass = TObject::Class());
  TList* getObjectsFromFile(const TString& filename, const TString& objName, TClass* objClass = TObject::Class());
  TList* getObjects(const TString& objName, TClass* objClass = TObject::Class(), TDirectory* d = gDirectory);

  void printBranches(TTree* t);
  void printActiveBranches(TTree* t);
  double getValue(TTree* t, const TString& bname, Long64_t iEvent = 0);
  double getSum(TTree* t, const TString& bname);

  #ifdef ROOTCORE_PACKAGE_xAODRootAccess
  double xAODMetaDataGetSumWeightsInitialFromEventBookkeeper(TTree* MetaData);
  double xAODMetaDataGetSumWeightsInitialFromCutBookkeeper(xAOD::TEvent& event, const char* container = "StreamAOD", const char* bookkeeper = "AllExecutedEvents", const char* kernelname = "HIGG3D1Kernel");
  double xAODMetaDataGetNEventsInitialFromCutBookkeeper(xAOD::TEvent& event, const char* container = "StreamAOD", const char* bookkeeper = "AllExecutedEvents", const char* kernelname = "HIGG3D1Kernel");
  #endif

  TList* getListOfFoldersInFile(const TString& filename, TClass* type = TQFolder::Class());
  int appendToListOfFolders(const TString& filename, TList* list, TClass* type = TQFolder::Class());
  TList* getListOfFolderNamesInFile(const TString& filename, TClass* type = TQFolder::Class());
  int appendToListOfFolderNames(const TString& filename, TList* list, TClass* type = TQFolder::Class(), const TString& prefix="");

  double roundAuto(double d, int nSig=1);
  double roundAutoUp(double d, int nSig=1);
  double roundAutoDown(double d, int nSig=1);

  double round (double d, int nDigits=0);
  double roundUp (double d, int nDigits=0);
  double roundDown(double d, int nDigits=0);

  void printParticles(TTree* t, Long64_t evt, int id, const TString& bName = "TruthParticles");
  void printParticle(TTree* t, Long64_t evt, int index, const TString& bName = "TruthParticles");
  void printProductionChain(TTree* t, Long64_t evt = 0, int ptcl = 0, const TString& bName = "TruthParticles");
  void printDecayChain(TTree* t, Long64_t evt = 0, int ptcl = 0, const TString& bName = "TruthParticles");
  
  void dumpTop(TString folder, TString prefix = "qframework", TString message = "");
  size_t getPeakRSS();
  size_t getCurrentRSS();
}

#endif
