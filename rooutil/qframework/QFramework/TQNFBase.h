//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQNFBase__
#define __TQNFBase__

#include "QFramework/TQFolder.h"
#include "TString.h"
#include "QFramework/TQSampleDataReader.h"
#include "QFramework/TQMessageStream.h"

#include <math.h>
#include <limits>

class TQNFChainloader; 

class TQNFBase : public TQNamedTaggable {
  // Classes inheriting from this abstract base class must include "QFramework/TQNFChainloader.h" in their source file!
protected:
  TQNFBase();
  TQNFBase(const TString& name);

  TQSampleDataReader* fReader;
  bool fOwnerOfReader;

  TQMessageStream messages; //!
  int verbosity;

  std::vector<TString> vNFpaths; // the non-abstract class has to keep this vector updated with the complete paths where NFs have been set, e.g. "/bkg/ee/Zjets/.default:CutZCR_VBF_C". It is handy to fill this vector if execute(0) is called, i.e. during the first iteration run by TQNFChainloader.
  TQNFChainloader* chainLoader;
 
  TQFolder * infoFolder;
  TQFolder * cutInfoFolder;

  bool initialized;
  int iterationNumber;

  virtual bool initializeSelf() = 0; 
  virtual bool finalizeSelf() = 0; 

public:

  virtual ~TQNFBase();

  TString getPathTag(const TString& tagname);

  void setSampleFolder(TQSampleFolder* sf);
  void setReader(TQSampleDataReader* rd);
  TQSampleFolder* getSampleFolder();
  TQSampleDataReader* getReader();

  void setOutputStream(const TString& outfile = "", int verbosity = 2);
  void closeOutputStream();
  void setVerbosity(int verbosity);

  void setInfoFolder(TQFolder* f);
  void setCutInfoFolder(TQFolder* f);
 
  virtual bool readConfiguration(TQFolder* f) = 0;
  //virtual int writeResultsToStream(std::ostream* out) = 0;
  //virtual int writeResultsToFile(const TString& filename) = 0;

  virtual TString getStatusMessage();
  virtual int getStatus();

  bool initialize();
  bool finalize();

  virtual int execute(int itrNumber) = 0; // If itrNumber=-1, no variation of input values should be performed. 
  virtual bool success() = 0;
 
  void setChainLoader(TQNFChainloader * loader);
 
  int deployResult(const std::vector<TString>& startAtCutNames, const std::vector<TString>& stopAtCuts = std::vector<TString>() );
  virtual int deployResult(const std::vector<TString>& startAtCutnames, const std::vector<TString>& stopAtCuts, int overwrite) = 0;
  
  std::vector<TString> getTargetCuts(const std::vector<TString>& startCuts, const std::vector<TString>& stopCuts);
  void getTargetCutsWorker(TList* targets, TQFolder* startCut, TList* stopCuts = NULL);

  const std::vector<TString>& getNFpaths(); 
  void addNFPath(const TString& path, const TString& cutname, const TString& scalescheme = ".default");
  void printNFPaths();

  ClassDef(TQNFBase, 0); // abstract base class for normalization factor calculators

};

#endif
