//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQ_NFCALCULATOR__
#define __TQ_NFCALCULATOR__

#include "QFramework/TQNFBase.h"
#include <iostream>
#ifdef __CINT__ 
#define override
#endif
class TQNFCalculator : public TQNFBase {
public:
  enum Method {
    Single,
    FractionFitter,
    MatrixInversion,
    Unity,
    FAIL,
    UNDEFINED
  };
protected:
  int status;

  inline Method getMethod(TString name){
    name.ToLower();
    if(name == "single" || name == "simple"){
      return Single;
    }
    if(name == "fractionfitter" || name == "tfractionfitter"){
      return FractionFitter;
    }
    if(name == "matrixinversion" || name == "matrix"){
      return MatrixInversion;
    }
    if(name == "unity" || name == "1" || name == "allone" || name == "noop"){
      return Unity;
    }
    if(name == "fail"){
      return FAIL;
    }
    if(this->verbosity > 0){
      this->messages.sendMessage(TQMessageStream::ERROR,"method '%s' not implemented!",name.Data());
    }
    this->status = -50;
    return UNDEFINED;
  }

  inline TString getMethodName(TQNFCalculator::Method method){
    switch(method){
    case Single:
      return "Single";
    case FractionFitter:
      return "TFractionFitter";
    case MatrixInversion:
      return "MatrixInversion";
    case Unity:
      return "Unity";
    case FAIL:
      return "FAIL";
    default:
      return "<unknown>";
    }
    return "<unknown>";
  }

  TString defaultDataPath;
  std::vector<TString> dataPaths;
  std::vector<TString> cutNames;
  std::vector<TString> mcPaths;
  std::vector<TString> mcNames;
  std::vector<bool> mcFixed;
  std::vector<double> nfBoundUpper;
  std::vector<double> nfBoundLower;
  std::vector<double> NFs;
  std::vector<double> NFuncertainties;

  std::vector<Method> methods;

  TH1F* data;
  TObjArray* mc;

  bool initializeSelf() override;
  bool finalizeSelf() override;

  bool initializeData();
  bool initializeMC();
  size_t findIndex(TString name);

  bool ensurePositiveMC();

  size_t getFloatingSamples();

  int calculateNFs(TQNFCalculator::Method method);
  void calculateNFs_singleMode();
  void calculateNFs_TFractionFitterMode();
  void calculateNFs_MatrixMode();
  void fail();

  double epsilon;
  TQTaggable* getResultsAsTags(size_t i);

public:
  void setNFsUnity();

  void setDefaultDataPath(TString path);

  bool addRegion(TString cutName, TString myDataPath="");
  bool addSample(TString mcPath, TString name="", bool fixed=false);
  bool addSample(TString mcPath, TString name, double boundLower, double boundUpper);
  bool addFixSample(TString mcPath, TString name="");

  void printRegions(std::ostream* os = &(std::cout));
  void printSamples(std::ostream* os = &(std::cout));
  void printResults(std::ostream* os = &(std::cout));

  TString getDefaultDataPath();

  TH1* getHistogram(const TString& name);
  bool scaleSample(const TString& name, double val);

  int deployNF(const TString& name, const std::vector<TString>& startCutNames, const std::vector<TString>& stopAtCut = std::vector<TString>(), int overwrite=1);

  int deployResult(const std::vector<TString>& startCutNames, const std::vector<TString>& stopAtCut, int overwrite) override;

  bool addMethod(const TString& methodName);
  void clearMethods();
  void printMethods();

  bool success() override;
  int execute(int itrNumber = -1) override;
  void clear();
  int calculateNFs();
  int calculateNFs(const TString& methodName);

  TString getStatusMessage() override;
  int getStatus() override;

  void printStatus();

  double getNF(TString name);
  double getNFUncertainty(TString name);
  bool getNFandUncertainty(TString name, double& nf, double& sigma);

  const TString& getMCPath(const TString& name);
  TH1* getMCHistogram(const TString& name);

  TQSampleFolder* exportHistograms(bool postFit = true);
  bool exportHistogramsToSampleFolder(TQSampleFolder* folder, bool postFit = true);
  bool exportHistogramsToFile(const TString& destination,bool recreate = true, bool postFit = true);

  void setEpsilon(double e);
  double getEpsilon();

  int writeResultsToFile(const TString& filename);
  int writeResultsToStream(std::ostream* out);
  TString getResultsAsString(const TString& name);
  TQTaggable* getResultsAsTags(const TString& name);

  bool readConfiguration(TQFolder* f) override;

  TQNFCalculator(TQSampleFolder* f = NULL);
  TQNFCalculator(TQSampleDataReader* rd);
  virtual ~TQNFCalculator();

  ClassDef(TQNFCalculator,1) // calculator class for normalization factors

};

#endif
