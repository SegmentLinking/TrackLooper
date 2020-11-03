//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQ_GRIDSCANNER__
#define __TQ_GRIDSCANNER__

#include <vector>
#include "TString.h"

#include "QFramework/TQGridScanPoint.h"
#include "QFramework/TQFolder.h"
#include "QFramework/TQSignificanceEvaluator.h"
#include "TMVA/Timer.h"

class TPad;
class TLegend;
class TLine;
class TH1;
class TH2;
class TH1F;
class TH2F;


class TQGridScanner : public TQTaggable, public TNamed {
public:
  enum BoundType { LOWER, UPPER, SPLIT, LOWERSWITCH, UPPERSWITCH, LOWERFIXED, UPPERFIXED, WINDOWFIXED, UNDEFINED };

protected:
  unsigned long int nPointsProcessed; //!
  TString targetVarName;
  TString targetVarTitle;
  std::vector<TString> variables;
  std::vector<TString> histVars;
  std::vector<int> varIdx;
  std::vector<TString> varTitle;
  std::vector<double> currentVals; //!
  std::vector<size_t> currentBins; //!
  std::vector<BoundType> boundTypes;
  std::vector<TQGridScanPoint*> points;
  std::vector<size_t> splitVars; //!
  std::vector<bool> switchVars; //!
  std::vector<TString> currentSwitchStatus; //!
  std::vector<THnBase*> ndimHists; //!
  std::vector<int> axesToScan; //!
	
  std::vector<TString> requirements; //!
  std::vector<TString> requirements_types; //!
  std::vector<double> requirements_values; //!
  TQSignificanceEvaluator* evaluator; //!
  bool verbose;
  bool sorted;
  unsigned long int nPointsTotal; //!
  TString formatString; //!
  TString heartBeatCommand; //!
  unsigned long heartBeatInterval; //!

  void setStyle (TPad* pad, TH1* hist, TQTaggable& tags);
  void setStyle1D(TPad* pad, TH1* hist, TQTaggable& tags, TH1* histmax = NULL);
  // void setStyle2D(TPad* pad, TH2* hist, TQTaggable& tags);
  TLine* drawCutLine(TH1* hist);
  int drawCutLines(TH2* hist);
  void drawColAxisTitle(TQTaggable& tags, TH2* hist, const TString& title);
  void drawLabels(TQTaggable& tags);
  TLegend* drawLegend(TQTaggable& tags, TH1F* hist, const TString& histlabel, TH1F* histmax, TLine* cutline);
  void setYaxisTitleOffset(TQTaggable& tags, TH1* hist);
  void setXaxisTitleOffset(TQTaggable& tags, TH1* hist);
  void setAxisTitleOffset (TQTaggable& tags, TH1* hist);
 
  std::vector<double> axisMin;
  std::vector<double> axisMax;
  std::vector<size_t> nBins;
  std::vector<double> stepsize;
  std::vector<TString> boundedVariables;
  std::vector<double> lowerBounds;
  std::vector<double> upperBounds;
  std::vector<int> lowerBoundedBins;
  std::vector<int> upperBoundedBins;
  
  std::vector<double> scanLowerBounds;
  std::vector<double> scanUpperBounds;
  std::vector<int> scanLowerBins;
  std::vector<int> scanUpperBins;
  
  void setupRangesToScan();
  bool run(size_t varno);
  bool getSplitConfiguration(int n, size_t varno=0);
	
  size_t getMaxBin(size_t varno);
  size_t getMinBin(size_t varno);
	
  double getBinMax(size_t varno, size_t bin);
  double getBinMin(size_t varno, size_t bin);

  void setUpperRange(size_t varno);
  void setLowerRange(size_t varno);
  void setWindowRange(size_t varno);

  bool hasOtherVariable(size_t index);

  TString boundTypeToVarString(BoundType t);
  TString boundTypeToCutString(BoundType t);
 
  int getAxisIndex(THnBase* h, const TString& varname); 
		
  unsigned long heartbeat; //!
  bool updateTime();

  std::vector<int> variableOrdering; //!

  TMVA::Timer* runTimer = new TMVA::Timer("Scanning variables");

public:
  TString getVariableName (size_t index, bool includeBound = false);
  TString getVariableTitle (size_t index, bool includeBound = true);

  TString splitConfigInfoStr; //!

  TQGridScanner();
  TQGridScanner(const TString& name);
  ~TQGridScanner();
 
  bool run();
  void setHeartbeat(TString cmd, Long64_t time);

  bool isInVariables(const TString& varname);
  size_t getVariablePosition(const TString& varname);
  size_t getBoundedVariablePosition(const TString& varname);
  size_t findVariable(const TString& varname, BoundType type);
  size_t findVariable(const TString& varname);
 

  void setVerbose(bool v = true);
  void printConfiguration();
  void resetNdimHists();
  void addNdimHist(THnBase* hist);
  void reconfigureVariables();
  void addVariable(const TString& varname, BoundType type, double value = -999);
  void addVariableUpper(const TString& varname);
  void addVariableLower(const TString& varname);
  void addVariableSplit(const TString& varname);
  void addVariableUpperSwitch(const TString& varname);
  void addVariableLowerSwitch(const TString& varname);
  void addVariableUpperFixed(const TString& varname, double value);
  void addVariableLowerFixed(const TString& varname, double value);
  void addVariableWindowFixed(const TString& varname, double low, double up);
  bool setEvaluator(TQSignificanceEvaluator* e);
  bool updateEvaluatorHists(TString axisToEvaluate);

  void setFixedBounds(const TString& varname, double bound);
  void setFixedLowerBound(const TString& varname, double bound);
  void setFixedUpperBound(const TString& varname, double bound);
  void setUpperScanRange(const TString& varname, double bound);
  void setLowerScanRange(const TString& varname, double bound);
	
  void setCorrectBinBound(TAxis* axis, double& bound, TString boundtype);
  
  TString getBoundOperator(const TString& varname);
  TString getBoundOperator(size_t i);

  void clearRequirements();
  void addRequirement(const TString& varname, const TString& type, double val);
  bool isAcceptedPoint(size_t i);
 
  void setVariableBoundType (const TString& varname, BoundType type);
  TString getVariableTitle (const TString& varname,bool includeBound = true);

  void sortVariables(TString ordering = "");
  void consolidateVariables();
  void printVariableConfiguration();

  void printPoint(size_t n = 0);
  void printPoints(size_t first, size_t last);
  void printPoints(size_t last = -1);

  void sortPoints();
  size_t nPoints();
  void terminateAfter(unsigned long int n);
  void skipPoints(unsigned long int n);
  TQGridScanPoint* point(size_t i=0);
 
  TH1F* getSignificanceProfile (const TString& varname,  int topNumber = 1);
  TH1F* getSignificanceProfile (const TString& varname,  double topFraction);
  TH2F* getSignificanceProfile2D (const TString& varname1, const TString& varname2, int topNumber = 1);
  TH2F* getSignificanceProfile2D (const TString& varname1, const TString& varname2, double topFraction);
  void plotAndSaveSignificanceProfile (const TString& varname, const TString& path, int topNumber, const TString& options = "");
  void plotAndSaveSignificanceProfile (const TString& varname, const TString& path, double topFraction, const TString& options = "");
  void plotAndSaveSignificanceProfile2D (const TString& varname1, const TString& varname2, const TString& path, int topNumber = 1, const TString& options = "");
  void plotAndSaveSignificanceProfile2D (const TString& varname1, const TString& varname2, const TString& path, double topFraction, const TString& options = "");
  void plotAndSaveAllSignificanceProfiles (  const TString& path, int topNumber = 1, const TString& options = "");
  void plotAndSaveAllSignificanceProfiles (  const TString& path, double topFraction, const TString& options = "");
  // void plotAndSaveAllSignificanceProfiles2D(  const TString& path, int topNumber = 1 , const TString& options = ""); 
  // void plotAndSaveAllSignificanceProfiles2D(  const TString& path, double topFraction, const TString& options = "");
  void deploySignificanceProfile (const TString& varname, TQFolder* f, int topNumber = 1);
  void deploySignificanceProfile (const TString& varname, TQFolder* f, double topFraction);
  // void deploySignificanceProfile2D (const TString& varname1, const TString& varname2, TQFolder* f, int topNumber = 1);
  // void deploySignificanceProfile2D (const TString& varname1, const TString& varname2, TQFolder* f, double topFraction);
  void deployAllSignificanceProfiles (TQFolder* f,  int topNumber = 1);
  void deployAllSignificanceProfiles (TQFolder* f,  double topFraction);
  // void deployAllSignificanceProfiles2D (TQFolder* f,  int topNumber = 1);
  // void deployAllSignificanceProfiles2D (TQFolder* f,  double topFraction);
  TList* getAllSignificanceProfiles (   int topNumber = 1);
  TList* getAllSignificanceProfiles (   double topFraction);
  // TList* getAllSignificanceProfiles2D (   int topNumber = 1);
  // TList* getAllSignificanceProfiles2D (   double topFraction);


  TH1F* getHistogram (const TString& varname,  int topNumber = 1);
  TH1F* getHistogram (const TString& varname,  double topFraction);
  // TH2F* getHistogram2D (const TString& varname1, const TString& varname2, int topNumber = 1);
  // TH2F* getHistogram2D (const TString& varname1, const TString& varname2, double topFraction);
  void plotAndSaveHistogram (const TString& varname, const TString& path, int topNumber, const TString& options = "");
  void plotAndSaveHistogram (const TString& varname, const TString& path, double topFraction, const TString& options = "");
  // void plotAndSaveHistogram2D (const TString& varname1, const TString& varname2, const TString& path, int topNumber = 1, const TString& options = "");
  // void plotAndSaveHistogram2D (const TString& varname1, const TString& varname2, const TString& path, double topFraction, const TString& options = "");
  void plotAndSaveAllHistograms (  const TString& path, int topNumber = 1, const TString& options = "");
  void plotAndSaveAllHistograms (  const TString& path, double topFraction, const TString& options = "");
  // void plotAndSaveAllHistograms2D(  const TString& path, int topNumber = 1 , const TString& options = ""); 
  // void plotAndSaveAllHistograms2D(  const TString& path, double topFraction, const TString& options = "");
  void deployHistogram (const TString& varname, TQFolder* f, int topNumber = 1);
  void deployHistogram (const TString& varname, TQFolder* f, double topFraction);
  // void deployHistogram2D (const TString& varname1, const TString& varname2, TQFolder* f, int topNumber = 1);
  // void deployHistogram2D (const TString& varname1, const TString& varname2, TQFolder* f, double topFraction);
  void deployAllHistograms (TQFolder* f,  int topNumber = 1);
  void deployAllHistograms (TQFolder* f,  double topFraction);
  // void deployAllHistograms2D (TQFolder* f,  int topNumber = 1);
  // void deployAllHistograms2D (TQFolder* f,  double topFraction);
  TList* getAllHistograms (   int topNumber = 1);
  TList* getAllHistograms (   double topFraction);
  // TList* getAllHistograms2D (   int topNumber = 1);
  // TList* getAllHistograms2D (   double topFraction);

  ClassDefOverride(TQGridScanner,3) // helper class to facilitate cut optimization scans
};
 
#endif
