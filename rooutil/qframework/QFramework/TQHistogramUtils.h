//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQHistogramUtils__
#define __TQHistogramUtils__

class TH1;
class TH2;
class THStack;
class TGraph;
class TMultiGraph;
class TPrincipal;
class TGraphAsymmErrors;
class TGraph2D;
class TList;
class TAxis;
class TLine;
class TStyle;
class TCanvas;

class TQTaggable;
class TQCounter;

#include "TH1.h"
#include "TMatrixD.h"
#include "TString.h"

#include <vector>
#include <limits>

namespace TQHistogramUtils {

  enum FOM {
    kSoSqB,
    kSoSqBpdB,
    kSoSqSpB,
    kSoB,
    kPoisson,
    kUndefined
  };

  enum Axes {
    X=0,
    Y=1,
    Z=2
  };
 
  bool hasBinLabels(TH1* h);
  bool hasBinLabels(TAxis* a);
  TAxis* getAxis(TNamed* obj, int idx);

  std::vector<double> getBinLowEdges(TH1* histo, const std::vector<int>& binBorders);
  std::vector<int> getBinBorders(TH1* histo, const std::vector<double>& lowEdges);
  std::vector<double> getUniformBinEdges(int nBins, double min, double max);
  bool hasUniformBinning(TH1 * hist);
  bool hasUniformBinning(TAxis* axis);

  void scaleErrors(TH1* hist, double scale);

  bool extractBinning(TQTaggable * p, int &index, int &nBins, double &min, double &max, std::vector<double> &edges, TString &errMsg);
  bool extractRange(TQTaggable * p, int &index, double &min, double &max, TString &errMsg);
  std::vector<TString> histoBinsToCutStrings(TH1* hist, const TString& varexpr, TString cutname = "", const TString& basecutname = "");

  TString getHistogramContents(TH1 * histo);
  TString getHistogramDefinition(TH1 * histo);
  TString getGraphDefinition(TNamed * graph);
  TString getBinningDefinition(TAxis * axis);
  bool setHistogramContents(TH1 * histo, const TString& contents);
  TString convertToText(TH1 * histo, int detailLevel);
  TH1* convertFromText(TString input);
 

  inline int getNbinsGlobal(TH1* hist, bool ignoreLabels = false){
    if(!hist) return -1;
    if(TQHistogramUtils::hasBinLabels(hist) && !ignoreLabels) return hist->GetNbinsX()*hist->GetNbinsY()*hist->GetNbinsZ();
    return hist->FindBin(std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()) +1;
  }
  
  TH1 * createHistogram(TString definition, bool printErrMsg = false);
  TH1 * createHistogram(TString definition, TString &errMsg);
  TNamed * createGraph(TString definition, bool printErrMsg = false);
  TNamed * createGraph(TString definition, TString &errMsg);

  TString getDetailsAsString(TNamed * obj, int option = 1);
  TString getDetailsAsString(TH1 * histo, int option = 1);
  TString getDetailsAsString(TAxis * axis, int option = 1);
  TString getDetailsAsString(TGraph * g, int option = 1);
  TString getDetailsAsString(TGraph2D * g, int option = 1);
  TString getDetailsAsString(TPrincipal * p, int option = 1);
  int dumpData(TPrincipal * p, int cutoff = 100);

  /* significance calculation */

  double getPoisson(double b, double s);
  double getPoissonError(double b, double s, double db, double ds);

  double getPoissonWithError(double b, double berr, double s);

  double getSoverSqrtB(double b, double s);
  double getSoverSqrtSplusB(double b, double s);

  double getSoverB(double b, double s);

  double getSignificance(double b, double s, TString sgnfName, TString * sgnfTitle = 0);

  void getSgnfAndErr(double b, double bErr, double s, double sErr,
                     double & sgnf, double & sgnfErr, TString sgnfName, TString * sgnfTitle = 0);

  double getFOM(FOM fom, double b, double berr, double s, double serr);
  TString getFOMTitle(FOM fom);
  TH1* getFOMScan(TQHistogramUtils::FOM fom, TH1* sig, TH1* bkg, bool fromleft, double cutoff, bool verbose);
  TH1* getFOMHistogram(TQHistogramUtils::FOM fom, TH1* sig, TH1* bkg, double cutoff = 0);
  FOM readFOM(TString fom);

  double pValuePoisson(unsigned nObs, double nExp);
  double pValuePoissonError(unsigned nObs, double E=1, double V=1);
  double pValueToSignificance(double p, bool excess=true);
  TH1* pcmpObsVsExp(TH1* hObs, TH1* hExp, bool ignoreExpUnc=false);

  TH1* getUncertaintyHistogram(TH1* hist);
  TGraph* scatterPlot(const TString& name, double* vec1, double* vec2, int vLength, const TString& labelX = "x", const TString& labelY = "y");
  TGraph* scatterPlot(const TString& name, std::vector<double>& vec1, std::vector<double>& vec2, const TString& labelX = "x", const TString& labelY = "y");

  TMultiGraph* makeMultiColorGraph(const std::vector<double>& vecX, const std::vector<double>& vecY, const std::vector<short>& vecColors);
  
  /* ===== histogram utils===== */


  TH1 * invertShift(TH1 * var, TH1 * nom);

  TH1* symmetrizeFromTwo(TH1* var1, TH1* var2, TH1* nom);

  bool applyPoissonErrors(TH1 * histo);
 
  TH1 * copyHistogram(TH1 * histo, const TString& newName = "");
  TNamed * copyGraph(TNamed * histo, const TString& newName = "");
	void copyGraphAxisTitles(TNamed* copy, TNamed* graph);

  TH1 * convertTo1D(TH2 * histo, bool alongX = true, bool includeUnderflowsX = true,
                    bool includeOverflowsX = true, bool includeUnderflowsY = true, bool includeOverflowsY = true);

  int getNDips(TH1 * histo);

  void interpolateGraph(TGraph* g, size_t increasePoints, const char* option="S");

  /* ===== rebinning/remapping of histograms===== */

  std::vector<int> getBinBordersFlat(TH1 * histo, int nBins, bool includeOverflows);
  std::vector<int> getBinBordersFlat2D(TH2 * histo, int nBins, bool remapX, bool includeOverflows, bool remapSlices = true);


  void remap(TAxis* ax, double min=0, double max=1);
  bool rebin(TH1*& hist, const std::vector<double>& boundaries, bool doRemap = false);
  bool rebin(TH1*& hist, const std::vector<int>& boundaries, bool doRemap = false);
  bool rebin(TH1* hist, int rebinX, int rebinY, int rebinZ);

  TH1 * getRebinned(TH1 * histo, const std::vector<int>& binBorders, bool doRemap = false);
  TH1 * getRebinned(TH1 * histo, const std::vector<double>& lowEdges, bool doRemap = false);
  TH1 * getRebinned(TH1 * histo, const std::vector<int>& binBorders, const std::vector<double>& lowEdges, bool doRemap = false);
  
  TH2 * getRemapped2D(TH2 * histo, const std::vector<int>& binBorders, bool remapX = true);

  TH1 * getRebinnedFlat(TH1 * histo, int nBins);
  TH1 * getRemappedFlat(TH1 * histo, int nBins);
  TH2 * getRemappedFlat2D(TH2 * histo, int nBins, bool remapX = true);


  TH1 * cutHistogram(TH1 * histo, int xBinLow, int xBinHigh, int yBinLow = -1, int yBinHigh = -1,
                     bool keepInUVX = false, bool keepInOVX = false, bool keepInUVY = false, bool keepInOVY = false);
  TH1 * cutAndZoomHistogram(TH1 * histo, int cutBinLowX, int cutBinHighX, int cutBinLowY = -1, int cutBinHighY = -1,
                            int zoomBinLowX = -1, int zoomBinHighX = -1, int zoomBinLowY = -1, int zoomBinHighY = -1);

  TH2 * removeBins(TH2* in, const std::vector<TString>& blackList);
  TH2 * removeBins(TH2* in, TString blackList);

  bool includeSystematics(TH1 * histo, TH1 * systematics);

  TH1 * getSystematics(TH1 * h_nominal, TList * singleVars, TList * pairVars = 0);

  bool addHistogram(TH1 * histo1, TH1 * histo2, double scale = 1., double scaleUncertainty = 0., double corr12 = 0., bool includeScaleUncertainty=true);
  bool addHistogram(TH1 * histo1, TH1 * histo2, TQCounter* scale, double corr12 = 0., bool includeScaleUncertainty=false);
  bool scaleHistogram(TH1 * histo1, double scale = 1., double scaleUncertainty = 0., bool includeScaleUncertainty = true);
  bool scaleHistogram(TH1 * histo1, TQCounter* scale, bool includeScaleUncertainty = false);
  bool addGraph(TGraph * graph1, TGraph * graph2);
  bool addGraph(TGraph2D * graph1, TGraph2D * graph2);
  bool addGraph(TNamed * graph1, TNamed * graph2);

  bool drawHistograms(TList * histograms, TString drawOption = "", TString extOptions = "");

  bool resetBinErrors(TH1 * histo);
  int fixHoles1D(TH1* hist, double threshold);

  int getDimension(TH1 * histo);

  int getNBins(TH1 * histo, bool includeUnderflowOverflow = true);

  int getSizePerBin(TH1 * histo);

  int estimateSize(TH1 * histo);

  bool checkConsistency(TH1 * histo1, TH1 * histo2, bool verbose=false);
  bool areEqual(TH1* first, TH1* second, bool includeUnderflowOverflow = true, bool compareErrors = false, double tolerance=0.01);

  /* */

  TH1 * getCutEfficiencyHisto(TH1 * histo, TString options = "");

  TH1 * getSignificanceHisto(TH1 * histo_bkg, TH1 * histo_sig, TString options = "");

  TGraphAsymmErrors * getROCGraph(TH1 * h_bkg, TH1 * h_sig, bool lowerBound);

  TList * getProjectionHistograms(TH2 * histo, bool projectOnX, bool normalize = false);
  TList * getProjectionHistogramsX(TH2 * histo, bool normalize = false);
  TList * getProjectionHistogramsY(TH2 * histo, bool normalize = false);

  TH1 * getReweightedHistogram(TH2 * histo_input, TH1 * histo_weights, bool projectOnX);
  TH1 * getReweightedHistogramX(TH2 * histo_input, TH1 * histo_weights);
  TH1 * getReweightedHistogramY(TH2 * histo_input, TH1 * histo_weights);

  TQCounter * histogramToCounter(TH1 * histo);

  TH1 * counterToHistogram(TQCounter * counter);
  TH1 * countersToHistogram(TList * counters);
  
  TH1 * getEfficiencyHistogram(TH1* numerator, TH1* denominator);

  /* projections */

  TH1 * getProjection(TH1 * histo, bool onX, int binLow = -1, int binHigh = -1);
  TH1 * getProjectionX(TH1 * histo, int binLow = -1, int binHigh = -1);
  TH1 * getProjectionY(TH1 * histo, int binLow = -1, int binHigh = -1);


  /* */

  double getIntegral(TH1 * histo, bool userUnderOverflow = true);
  double getIntegralError(TH1 * histo);
  double getIntegralAndError(TH1 * histo, double &error, bool useUnderflowOverflow = true);

  TH1 * normalize(TH1 * histo, double normalizeTo = 1.);
  TH1 * normalize(TH1 * histo, TH1 * normalizeToHisto);

  TH1 * power(TH1 * histo, double exp);

  TH1 * getSlopeHistogram(TH1 * input, double slope);
  TH1 * applySlopeToHistogram(TH1 * input, double slope);

  double getChi2(TH1 * histo1, TH1 * histo2);

  TH1 * includeOverflowBins(TH1 * histo, bool underflow = true, bool overflow = true);

  void unifyMinMax(TCollection * histograms, double vetoFraction = .9);
  void unifyMinMax(TH1 * h1, TH1 * h2, TH1 * h3, double vetoFraction = .9);
  void unifyMinMax(TH1 * h1, TH1 * h2, double vetoFraction = .9);

  bool getMinMaxBin(TH1 * histo, int &minBin, int &maxBin, bool includeError = false, bool includeUnderflowOverflow = true, double minMin = -std::numeric_limits<double>::infinity(), double maxMax = std::numeric_limits<double>::infinity());
  bool getMinMax(TH1 * histo, double &min, double &max, bool includeError = false, bool includeUnderflowOverflow = true, double minMin = -std::numeric_limits<double>::infinity(), double maxMax = std::numeric_limits<double>::infinity());
  int getMinBin(TH1 * histo, bool includeError = false, bool includeUnderflowOverflow = true, double minMin = -std::numeric_limits<double>::infinity());
  int getMaxBin(TH1 * histo, bool includeError = false, bool includeUnderflowOverflow = true, double maxMax = std::numeric_limits<double>::infinity());
  double getMin(TH1 * histo, bool includeError = false, bool includeUnderflowOverflow = true, double minMin = -std::numeric_limits<double>::infinity());
  double getMax(TH1 * histo, bool includeError = false, bool includeUnderflowOverflow = true, double maxMax = std::numeric_limits<double>::infinity());
  double getMax(TCollection* c, bool includeUnderflowOverflow = true, double maxMax = std::numeric_limits<double>::infinity() );
  double getMin(TCollection* c, bool includeUnderflowOverflow = true, double minMin = -std::numeric_limits<double>::infinity() );

  bool getMaxArea2D(TH2 * histo, double frac, int &maxX, int &maxY,
                    int &maxX_low, int &maxX_high, int &maxY_low, int &maxY_high);

  bool extractStyle(TH1 * histo, TQTaggable * tags, const TString& styleScheme = "");
  int applyStyle(TH1 * histo, TQTaggable * tags, const TString& styleScheme = "", bool allowRecursion = true);
  bool copyStyle(TH1 * dest, TH1 * src);
  bool copyStyle(TGraph * dest, TGraph * src);
  bool copyStyle(TGraph2D * dest, TGraph2D * src);
  bool copyStyle(TNamed * dest, TNamed * src);  

  bool copyBinLabels(TH1* source, TH1* target);
  bool copyBinLabels(TAxis* source, TAxis* target);
  bool copyAxisStyle(TH1* source, TH1* target);
  bool copyAxisStyle(TAxis* source, TAxis* target);

  TGraphAsymmErrors * getGraph(TH1 * histo);
  TGraphAsymmErrors * getGraph(TH1* nom, TObjArray* sys);

  bool isCloseToOneRel(double val, double rel);
  bool areEqualRel(double val1, double val2, double rel);
  bool haveEqualShapeRel(TH1 * h1, TH1 * h2, double rel);

  int ensureMinimumBinContent(TH1 * histo, double min = 1E-12, bool forcePositive = false);

  template<class TMatrixTT> TH2* convertMatrixToHistogram(const TMatrixTT* matrix, const TString& name);
  TMatrixD* convertHistogramToMatrix(TH2* hist);
    
  bool envelopeUpper(TH1* hist, TH1* otherhist);
  bool envelopeLower(TH1* hist, TH1* otherhist);

  void setSliceX(TH2* hist2d, TH1* hist, double value);
  void setSliceY(TH2* hist2d, TH1* hist, double value);

  double getMinimumBinValue(TH1* hist, double xmin, double xmax, bool includeErrors=false);
  double getMaximumBinValue(TH1* hist, double xmin, double xmax, bool includeErrors=false);
  double getMinimumBinValue(TCollection* hist, double xmin, double xmax, bool includeErrors=false);
  double getMaximumBinValue(TCollection* hist, double xmin, double xmax, bool includeErrors=false);
  double getHistogramMaximum(size_t n, ...);
  double getHistogramMinimum(size_t n, ...);
  double getHistogramXmax(size_t n, ...);
  double getHistogramXmin(size_t n, ...);
  double getHistogramYmax(size_t n, ...);
  double getHistogramYmin(size_t n, ...);

  double getHistogramBinContentFromFile(const TString& fname, const TString& hname, const TString binlabel);
  double getHistogramBinContentFromFile(const TString& fname, const TString& hname, int);
  int edge(TH1* hist, double cutoff = std::numeric_limits<double>::epsilon());
  int edge(TH2* hist, double cutoff = std::numeric_limits<double>::epsilon());

  double getAxisYmin(TH1* hist);
  double getAxisYmax(TH1* hist);
  double getAxisXmin(TH1* hist);
  double getAxisXmax(TH1* hist);

  double getContourArea(TGraph* g);
  double getContourJump(TGraph* g);

  double getMinBinWidth(TAxis*a);
  double getMinBinArea(TH2* hist);

  
  TH1* getSoverSqrtBScan(TH1* signal, TH1* bkg, bool fromleft, double cutoff = 0.05, bool verbose=false);
  TH1* getSoverSqrtB(TH1* sig, TH1* bkg);
 
  void print(THStack* s, TString options="");
 
  int addPCA(TPrincipal* orig, TPrincipal* add);
 
  double sumLineDistSquares(TH2* hist, double a, double b, bool useUnderflowOverflow = false);
  TH1* getLineDistSquares(TH2* hist, double a, double b, bool useUnderflowOverflow = false);
  double calculateBinLineDistSquare(TH2* hist, double a, double b, int i, int j);
  TH1* rotationProfile(TH2* hist, double xUnit = -1, double yUnit = -1, int nStep = 36, double xOrig = 0, double yOrig = 0);
  TH2* rotationYtranslationProfile(TH2* hist, double xUnit = -1, double yUnit = -1, int nStepAngle = 36, int nStepOffset = -1, double xOrig = 0, double y0 = std::numeric_limits<double>::quiet_NaN());
  TH2* rotationXtranslationProfile(TH2* hist, double xUnit = -1, double yUnit = -1, int nStepAngle = 36, int nStepOffset = -1, double yOrig = 0, double x0 = std::numeric_limits<double>::quiet_NaN());
  TLine* makeBisectorLine(TH1* hist, double angle = 45, double xUnit = -1, double yUnit = -1, double xOrig = 0, double yOrig = 0);

  double clearBinsAboveX(TH1* hist, double xMax);
  double clearBinsBelowX(TH1* hist, double xMin);
  double clearBinsAboveX(TH2* hist, double xMax);
  double clearBinsBelowX(TH2* hist, double xMin);
  double clearBinsAboveY(TH2* hist, double yMax);
  double clearBinsBelowY(TH2* hist, double yMin);

  bool cropLine(TH1* hist, TLine* l);
  bool isGreaterThan(TH1* hist1, TH1* hist2);
  bool isUnderflowOverflowBin(TH1* hist, int bin);
  
  void printHistogramASCII(TH1* hist, const TString& tags = "");
  void printHistogramASCII(TH1* hist, TQTaggable& tags);
  void printHistogramASCII(std::ostream&, TH1* hist, const TString& tags);
  void printHistogramASCII(std::ostream&, TH1* hist, TQTaggable& tags);

  TH1* unrollHistogram(TH2* input, bool firstX = true, bool includeUnderflowOverflow = false);
  TObjArray* getSlices(TH2* input, bool alongX = true);

  void rerollGauss(TH1* hist, double zvalue = 1);
  void rerollPoisson(TH1* hist);
  
  TStyle* ATLASstyle();
  TCanvas* applyATLASstyle(TH1* histo, const TString& label = "Internal", double relPosX = 0.1, double relPosY = 0.1, double yResize = 1., const TString& xTitle = "none", const TString& yTitle = "none", bool square = false);
  TCanvas* applyATLASstyle(TGraph* graph, const TString& label = "Internal", double relPosX = 0.1, double relPosY = 0.1, double yResize = 1., const TString& xTitle = "none", const TString& yTitle = "none", bool square = false);
  TCanvas* applyATLASstyle(TMultiGraph* graph, const TString& label = "Internal", double relPosX = 0.1, double relPosY = 0.1, double yResize = 1., const TString& xTitle = "none", const TString& yTitle = "none", bool square = false);
}


#endif
