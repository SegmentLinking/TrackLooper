//#define private public
// this is because ROOT authors decided to make IsAlphanumeric a private method... :-(
#include "TAxis.h"
//#undef private
#include "TPDF.h"
#include "TGraphAsymmErrors.h"
#include "TLegend.h"
#include "TMath.h"
#include "TMatrixD.h"
#include "TGraph.h"
#include "TGraph2D.h"
#include "TMultiGraph.h"
#include "TH1.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TH1F.h"
#include "TH1D.h"
#include "TH2.h"
#include "TLine.h"
#include "THStack.h"
#include "TPrincipal.h"
#include "TProfile.h"
#include "TProfile2D.h"
#include "TH2F.h"
#include "TH2D.h"
#include "TH3.h"
#include "THashList.h"
#include "TH3F.h"
#include "TH3D.h"
#include "TROOT.h"
#include "TFile.h"
#include "TLatex.h"
#include "TRandom3.h"

#include "QFramework/TQHistogramUtils.h"
#include "QFramework/TQStringUtils.h"
#include "QFramework/TQTaggable.h"
#include "QFramework/TQCounter.h"
#include "QFramework/TQIterator.h"
#include "QFramework/TQUtils.h"

#include "QFramework/TQLibrary.h"

// #define _DEBUG_

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <limits>

#include <stdio.h> 
#include <stdarg.h> 

#include "Math/Math.h"
#include "Math/QuantFuncMathCore.h"
#include "Math/SpecFuncMathCore.h"


//somewhat complex but apparently standard conform hack to access TAxis::IsAlphanumeric. This is only needed until we regularly use a newer root version (6.10 at latest) where this has finally been changed to be a public method...
template <typename TAxisTag>
struct TAxisHackResult {
  typedef typename TAxisTag::type type;
  static type ptr;
};

template <typename TAxisTag>
typename TAxisHackResult<TAxisTag>::type TAxisHackResult<TAxisTag>::ptr;

template<typename TAxisTag, typename TAxisTag::type p>
struct TAxisRob : TAxisHackResult<TAxisTag> {
  struct TAxisFiller {
    TAxisFiller() {TAxisHackResult<TAxisTag>::ptr = p;}
  };
  static TAxisFiller taxisfiller_obj;
};

template<typename TAxisTag, typename TAxisTag::type p>
typename TAxisRob<TAxisTag, p>::TAxisFiller TAxisRob<TAxisTag, p>::taxisfiller_obj;

//now expose some members of TAxis that we need to access
struct TAxisIsAlphanumeric { typedef Bool_t(TAxis::*type)(); };
template class TAxisRob<TAxisIsAlphanumeric, &TAxis::IsAlphanumeric>;

//struct TAxisLabels {typedef void(TAxis::*type)(); };
//template class TAxisRob<TAxisLabels, &TAxis::fLabels>;

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQHistogramUtils:
//
// TQHistogramUtils is a namespace providing a set of static utility methods related to the
// creation, inspection, and manipulation of histograms.
//
////////////////////////////////////////////////////////////////////////////////////////////////


//__________________________________________________________________________________|___________

TString TQHistogramUtils::getGraphDefinition(TNamed * graph) {
  // Returns a string representing the definition of the histogram
  // following a syntax compatible with
  // TQHistogramUtils::createGraph(...). An empty string is returned
  // in case an invalid histogram is passed.

  if (!graph) {
    return "";
  }
 
  TString title(graph->GetTitle());
  
  for(size_t idx=0; idx<3; idx++){
    TAxis* axis = TQHistogramUtils::getAxis(graph,idx);
    if(!axis) break;
    // embed axis title in graph title
    TString axtitle = axis->GetTitle();
    if (!axtitle.IsNull()) {
      TQStringUtils::append(title, axtitle, ";");
    }
  }
  
  //  the full graph definition
  TString def = TString::Format("%s(\"%s\", \"%s\")", graph->IsA()->GetName(), graph->GetName(),graph->GetTitle());
  return def;
}


//__________________________________________________________________________________|___________

TString TQHistogramUtils::getHistogramDefinition(TH1 * histo) {
  // Returns a string representing the definition of the histogram following a syntax
  // compatible with TQHistogramUtils::createHistogram(...) and being similar to the
  // constructor of the corresponding histogram class. An empty string is returned
  // in case an invalid histogram is passed.

  // invalid histogram?
  if (!histo) {
    return "";
  }
 
  // will become the full histogram definition
  TString def = TString::Format("%s(\"%s\", ", histo->IsA()->GetName(), histo->GetName());

  // will become full title of histogram (including axis titles)
  TString titleDef = histo->GetTitle();

  // will become binning definition
  TString binDef;
 
  // iterating over dimensions of histogram
  int dim = getDimension(histo);

  for (int i = 1; i <= abs(dim); ++i) {
    TAxis * axis = NULL;
    if (i == 1) {
      axis = histo->GetXaxis();
    } else if (i == 2) {
      axis = histo->GetYaxis();
    } else if (i == 3) {
      axis = histo->GetZaxis();
    }
    if (!axis) {
      // should never happen
      break;
    }
 
    // embed axis title in histogram title
    TString title(axis->GetTitle());
    if (!title.IsNull()) {
      TQStringUtils::append(titleDef, title, ";");
    }
 
    // compile binning definition string
    TQStringUtils::append(binDef, getBinningDefinition(axis), ", ");

    TProfile* p=dynamic_cast<TProfile*>(histo);
    if(p){
      TQStringUtils::append(binDef, TString::Format("%g, %g",p->GetYmin(),p->GetYmax()));
    }
  }
 
  // now combine all parts to one definition string
  def.Append(TString::Format("\"%s\", ", titleDef.Data()));
  def.Append(binDef + ")");
 
  return def;
}

//__________________________________________________________________________________|___________

TString TQHistogramUtils::getHistogramContents(TH1 * histo) {
  // convert the histogram contents to a string
  std::stringstream ss;
  ss << histo->GetEntries() << "@{";
  for(size_t i=0; i<(size_t)(histo->GetNbinsX())+2; ++i){
    if(i!=0) ss << ",";
    if(histo->GetNbinsY()>1){
      ss << "{";
      for(size_t j=0; j<(size_t)(histo->GetNbinsY())+2; ++j){
        if(j!=0) ss << ",";
        if(histo->GetNbinsZ()>1){
          for(size_t k=0; k<(size_t)(histo->GetNbinsZ())+2; ++k){
            if(k!=0) ss << ",";
            ss << histo->GetBinContent(i,j,k) << "+-"<<histo->GetBinError(i,j,k);
          }
          ss << "}";
        } else {
          ss << histo->GetBinContent(i,j) << "+-"<<histo->GetBinError(i,j);
        }
      }
      ss << "}";
    } else {
      ss << histo->GetBinContent(i) << "+-"<<histo->GetBinError(i);
    }
  }
  ss << "}";
  return ss.str();
}

//__________________________________________________________________________________|___________

TH1* TQHistogramUtils::convertFromText(TString input){
  // convert a string to a histogram
  TString def;
  TQStringUtils::readUpToText(input,def,"<<");
  TString contents;
  TQStringUtils::removeLeading(input,"< ");
  TH1* hist = TQHistogramUtils::createHistogram(def);
  if(!TQStringUtils::readUpToText(input,contents,"<<")){
    if(!TQHistogramUtils::setHistogramContents(hist,input)){
      ERROR("unable to read histogram contents!");
    }
  } else {
    TQStringUtils::removeLeading(input,"< ");
    if(!TQHistogramUtils::setHistogramContents(hist,contents)){
      ERROR("unable to read histogram contents!");
    }
    TQTaggable tags(input);
    TQHistogramUtils::applyStyle(hist,&tags);
  }
  return hist;
}

//__________________________________________________________________________________|___________

TString TQHistogramUtils::convertToText(TH1 * histo, int detailLevel) {
  // convert histogram to a string
  TString histstr = TQHistogramUtils::getHistogramDefinition(histo);
  if(detailLevel > 0){
    histstr += " << ";
    histstr += TQHistogramUtils::getHistogramContents(histo);
  }
  if(detailLevel > 1){
    histstr += " << ";
    TQTaggable tags;
    TQHistogramUtils::extractStyle(histo,&tags);
    histstr += tags.exportTagsAsString();
  }
  return histstr;
}

//__________________________________________________________________________________|___________

namespace {
  inline char* next(std::stringstream& ss, int nchars){
    ss.unget();
    char* text = (char*)malloc((nchars+1)*sizeof(char));
    ss.read(text,nchars);
    text[nchars]='\0';
    return text;
  }
}

bool TQHistogramUtils::setHistogramContents(TH1 * histo, const TString& contents) {
  // set the histogram contents from a string
  if(contents.IsNull()) return false;
  std::stringstream ss(contents.Data());
  int nentries;
  ss >> nentries;
  while(ss.good() && ss.peek() != '{'){
    ss.get();
  }
  if(!ss.good() || ss.get() != '{'){
    ERRORfunc("ill-formatted string '%s'",TString(contents(0,10)).Data());
    return false;
  }
  double val, err;
  for(size_t i=0; i<(size_t)(histo->GetNbinsX())+2 && ss.good(); ++i){
    if(histo->GetNbinsY() > 1){
      if(ss.get() != '{'){
        ERRORfile("missing opening brace at '%s'!",next(ss,10));
        return false;
      }
      for(size_t j=0; j<(size_t)(histo->GetNbinsY())+2 && ss.good(); ++j){
        if(histo->GetNbinsZ() > 1){
          if(ss.get() != '{'){
            ERRORfile("missing opening brace at '%s'!",next(ss,10));
            return false;
          }
          for(size_t k=0; k<(size_t)(histo->GetNbinsZ())+2 && ss.good(); ++k){
            ss >> val;
            if(!ss.good()){
              ERRORfile("stream ended after reading '%g' at '%d'/'%d'/'%d'",val,i,j,k);
              return false;
            }
            if(ss.get() != '+'){
              ERRORfile("missing '+' at '%s'!",next(ss,10));
              return false;
            }
            if(ss.get() != '-'){
              ERRORfile("missing '-' at '%s'!",next(ss,10));
              return false;
            }
            ss >> err;
            if(!ss.good()){
              ERRORfile("stream ended after reading '%g' at '%d'/'%d'/'%d'",err,i,j,k);
              return false;
            }
            if(ss.peek() != ',' && ss.peek() != '}'){
              ERRORfile("missing terminating character at '%s'",next(ss,10));
              return false;
            } else ss.get();
            histo->SetBinContent(i,j,k,val);
            histo->SetBinError(i,j,k,err);
          }
          if(ss.peek() != ',' && ss.peek() != '}'){
            ERRORfile("missing terminating character at '%s'",next(ss,10));
            return false;
          } else ss.get();
        } else {
          ss >> val;
          if(!ss.good()){
            ERRORfile("stream ended after reading '%g' at '%d'/'%d'",val,i,j);
            return false;
          }
          if(ss.get() != '+'){
            ERRORfile("missing '+' at '%s'!",next(ss,10));
            return false;
          }
          if(ss.get() != '-'){
            ERRORfile("missing '-' at '%s'!",next(ss,10));
            return false;
          }
          ss >> err;
          if(!ss.good()){
            ERRORfile("stream ended after reading '%g' at '%d'/'%d'",err,i,j);
            return false;
          }
          if(ss.peek() != ',' && ss.peek() != '}'){
            ERRORfile("missing terminating character at '%s'",next(ss,10));
            return false;
          } else ss.get();
          histo->SetBinContent(i,j,val);
          histo->SetBinError(i,j,err);
        }
      }
      if(ss.peek() != ',' && ss.peek() != '}'){
        ERRORfile("missing terminating character at '%s'",next(ss,10));
        return false;
      } else ss.get();
    } else {
      ss >> val;
      if(!ss.good()){
        ERRORfile("stream ended after reading '%g' at '%d'",val,i);
        return false;
      }
      if(ss.get() != '+'){
        ERRORfile("missing '+' at '%s'!",next(ss,10));
        return false;
      }
      if(ss.get() != '-'){
        ERRORfile("missing '-' at '%s'!",next(ss,10));
        return false;
      }
      ss >> err;
      if(!ss.good()){
        ERRORfile("stream ended after reading '%g' at '%d'",err,i);
        return false;
      }
      if(ss.peek() != ',' && ss.peek() != '}'){
        ERRORfile("missing terminating character at '%s'",next(ss,10));
        return false;
      } else ss.get();
      histo->SetBinContent(i,val);
      histo->SetBinError(i,err);
    }
  }
  histo->SetEntries(nentries);
  return true;
}


//__________________________________________________________________________________|___________

bool TQHistogramUtils::hasUniformBinning(TH1 * hist) {
	// return true if all axes of a histogram have uniform binning, false otherwise
  const int n = TQHistogramUtils::getDimension(hist);
  if(n==1){
    return hasUniformBinning(hist->GetXaxis());
  } else if(n==2){
    return 
      hasUniformBinning(hist->GetXaxis()) && 
      hasUniformBinning(hist->GetYaxis());
  } else if(n==3){
    return 
      hasUniformBinning(hist->GetXaxis()) && 
      hasUniformBinning(hist->GetYaxis()) && 
      hasUniformBinning(hist->GetZaxis());
  }
  return false;
}

//__________________________________________________________________________________|___________

bool TQHistogramUtils::hasUniformBinning(TAxis * axis) {
	// return true if an axis has uniform binning, false otherwise
  bool uniform = true;
  double binWidth = -1.;
  int n = axis->GetNbins();
  for (int i = 1; i < n + 1; i++) {
    // uniform binning?
    double l = axis->GetBinLowEdge(i);
    double u = axis->GetBinUpEdge(i);
    double w = u - l;
    if (binWidth < 0.) {
      binWidth = w;
    } else if (!TMath::AreEqualRel(binWidth, w, 1E-06)) {
      // bin widths not equal => non-uniform binning
      uniform = false;
    }
  }
  return uniform;
}
 
//__________________________________________________________________________________|___________

TString TQHistogramUtils::getBinningDefinition(TAxis * axis) {
  // Returns a string representing a definition of the binning on axis <axis>. For
  // uniformly binned axes the format is "<n>, <left>, <right>" where <n> refers to
  // the number of bins, <left> to the lower edge of the first bin and <right> to
  // the upper edge of the last bin. For non-uniformly binned axis the format is
  // "{<e1>, ..., <en>, <eX>}" where <ei> refers to the lower bin edge of bin i and
  // <eX> to the upper bin edge of the last bin. The format is compatible with
  // TQHistogramUtils::createHistogram(...) An empty string is returned in case an
  // invalid histogram is passed.
 
  // invalid axis
  if (!axis) {
    return "";
  }

  const int n = axis->GetNbins();

  if (!TQHistogramUtils::hasUniformBinning(axis)) {
    // non-uniform binning => list bin edges
    TString edges;
    
    for (int i = 1; i <= n + 1; i++) {
      // lower edge of current bin
      double l = axis->GetBinLowEdge(i);
      // compile comma-separated list of bin edges
      TQStringUtils::append(edges, TString::Format("%g", l), ", ");
    }
    
    edges.Prepend("{");
    edges.Append("}");
    return edges;
  } else {
    // uniform binning
    return TString::Format("%d, %g, %g", n, axis->GetBinLowEdge(1), axis->GetBinUpEdge(n));
  }
}
 
//__________________________________________________________________________________|___________

// return a variation that is symmetric to the var histogram relative to nom                                                                                                                                       
TH1* TQHistogramUtils::invertShift(TH1* var, TH1* nom){
  TH1D* hshift  = (TH1D*)nom->Clone();
  //  hshift->Sumw2(); // AL: NEEDED!                                                                                                                                                                                  
  hshift->Add(var,-1);
  hshift->Add(nom);// add to it the nominal ttbar                                                                                                                                                                  

  for (int bin=1; bin<=nom->GetNbinsX(); bin++){
    hshift->SetBinError(bin,nom->GetBinError(bin));
  }

  return hshift;
}


//__________________________________________________________________________________|___________

TH1* TQHistogramUtils::symmetrizeFromTwo(TH1* var1, TH1* var2, TH1* nom){
  std::cout << "entering SymmetrizeFromTwo" << std::endl;

  TH1D* tmp1 = (TH1D* )var1->Clone();
  tmp1->Divide(nom);
  //tmp1->Sumw2();                                                                                                                                                                                          

  TH1D* tmp2 = (TH1D* )var2->Clone();
  tmp2->Divide(nom);
  //tmp2->Sumw2();                                                                                                                                                                                          

  TH1D* unit = (TH1D* )nom->Clone();
  //unit->Sumw2();                                                                                                                                                                                          
  for (int bin=1; bin<= unit->GetNbinsX(); bin++){
    unit->SetBinContent(bin,1);
    unit->SetBinError(bin,0.0);
  }
  tmp1->Add(unit,-1);
  tmp2->Add(unit,-1);
  tmp1->Add(tmp2,-1);
  tmp1->Scale(0.5);
  tmp1->Add(unit);

  tmp1->Multiply(nom);

  for (int bin=1; bin<= unit->GetNbinsX(); bin++){
    tmp1->SetBinError(bin,nom->GetBinError(bin));
  }
  return tmp1;

} 


//__________________________________________________________________________________|___________

TH1 * TQHistogramUtils::createHistogram(TString definition, bool printErrMsg) {
  // Creates a new instance of a histogram (TH1F, TH1D, ...) from a definition string
  TString errMsg;
  TH1 * histo = createHistogram(definition, errMsg);
  if (!histo && printErrMsg) {
    std::cout << TQStringUtils::makeBoldRed(errMsg.Prepend("TQHistogramUtils::createHistogram(...): ")).Data() << std::endl;
  }
  return histo;
}

//__________________________________________________________________________________|___________

TNamed * TQHistogramUtils::createGraph(TString definition, bool printErrMsg) {
  // Creates a new instance of a histogram (TH1F, TH1D, ...) from a definition string
  TString errMsg;
  TNamed * graph = createGraph(definition, errMsg);
  if (!graph && printErrMsg) {
    std::cout << TQStringUtils::makeBoldRed(errMsg.Prepend("TQHistogramUtils::createGraph(...): ")).Data() << std::endl;
  }
  return graph;
}

//__________________________________________________________________________________|___________

TNamed * TQHistogramUtils::createGraph(TString definition, TString &errMsg) {
  // Creates a new instance of a graph (TGraph, TGraph2D, ...) from a definition string
  // that uses a simple Class(name,title) syntax.

  // read graph type (e.g. "TGraph", ...)
  TString type;
  TQStringUtils::readBlanksAndNewlines(definition);
  if (TQStringUtils::readToken(definition,type,TQStringUtils::getLetters() + TQStringUtils::getNumerals()) == 0) {
    errMsg = TString::Format("Missing valid graph type, received '%s' from '%s'",type.Data(),definition.Data());
    return NULL;
  }

  // histogram type to create
  bool isTGraph = (type.CompareTo("TGraph") == 0);
  bool isTGraph2D = (type.CompareTo("TGraph2D") == 0);
  bool isTGraphAsymmErrors = (type.CompareTo("TGraphAsymmErrors") == 0);

  if (!isTGraph && !isTGraph2D && !isTGraphAsymmErrors){
    errMsg = TString::Format("Unknown graph type '%s'", type.Data());
    return NULL;
  }

  // read parameter block
  TString parameter;
  TQStringUtils::readBlanksAndNewlines(definition);
  if (TQStringUtils::readBlock(definition, parameter, "()", "''\"\"", false) == 0) {
    errMsg = TString::Format("Missing parameter block '(...)' after '%s'", type.Data());
    return NULL;
  }

  // make sure there is nothing left after the parameter block
  TQStringUtils::readBlanksAndNewlines(definition);
  if (!definition.IsNull()) {
    errMsg = TString::Format("Unrecognized token '%s'", definition.Data());
    return NULL;
  }

  // parse parameter block
  TQTaggable * pars = TQTaggable::parseParameterList(parameter, ",", true, "{}[]()", "''\"\"");
  if (!pars) {
    errMsg = TString::Format("Failed to parse parameters '%s'", parameter.Data());
    return NULL;
  }

  // keep track of parameters read (to find unexpected parameters)
  pars->resetReadFlags();

  // name of histogram
  if (!pars->tagIsOfTypeString("0")) {
    errMsg = "Missing valid graph name";
    delete pars;
    return NULL;
  }
  TString name = pars->getTagStringDefault("0");

  // title of histogram
  if (!pars->tagIsOfTypeString("1")) {
    errMsg = "Missing valid graph title";
    delete pars;
    return NULL;
  }
  TString title = pars->getTagStringDefault("1");

  // now create the graph calling the corresponding constructor
  TNamed * graph = NULL;
  if (isTGraph) {
    graph = new TGraph();
  } else if(isTGraph2D){
    graph = new TGraph2D();
  } else if(isTGraphAsymmErrors){
    graph = new TGraphAsymmErrors();
  }

  graph->SetName(name);
  graph->SetTitle(title);
 
  delete pars;

  // finally return the histogram
  if(!graph){
    errMsg = "unknown error: graph type is '"+type+"'";
  }
  return graph;
}

//__________________________________________________________________________________|___________

TH1 * TQHistogramUtils::createHistogram(TString definition, TString &errMsg) {
  // Creates a new instance of a histogram (TH1F, TH1D, ...) from a definition string
  // that uses a similar syntax as the constructor of the corresponding histogram
  // class. Currently, TH1F, TH1D, TH2F, TH2D, TH3F, and TH3D are supported.
  //
  // Examples:
  //
  // - a TH1F with 5 bins between 0. and 1.
  //
  // createHistogram("TH1F('histo', 'title', 5, 0, 1)")
  //
  // - similarly, but specifying the bin edges explicitly
  //
  // createHistogram("TH1F('histo', 'title', {0, 0.2, 0.4, 0.6, 0.8, 1})")
  //
  // - a TH2D with 10 times 10 bins between -5 ... 5 and -1 ... 1, respectively
  //
  // createHistogram("TH2F('histo', 'title', 10, -5., 5., 10, -1., 1.)")
  //
  // - a TH2D with 3 times 3 bins between -5 ... 5 and variable bins between
  // -1 ... 1, respectively
  //
  // createHistogram("TH2F('histo', 'title', 3, -5., 5., {-1., -0.8, 0.1, 1.})")
  //

  // read histogram type (e.g. "TH1F", ...)
  TString type;
  TQStringUtils::readBlanksAndNewlines(definition);
  if (TQStringUtils::readToken(definition,type,TQStringUtils::getLetters() + TQStringUtils::getNumerals()) == 0) {
    errMsg = TString::Format("Missing valid histogram type, received '%s' from '%s'",type.Data(),definition.Data());
    return NULL;
  }

  // histogram type to create
  bool isTH1F = (type.CompareTo("TH1F") == 0);
  bool isTH1D = (type.CompareTo("TH1D") == 0);
  bool isTH2F = (type.CompareTo("TH2F") == 0);
  bool isTH2D = (type.CompareTo("TH2D") == 0);
  bool isTH3F = (type.CompareTo("TH3F") == 0);
  bool isTH3D = (type.CompareTo("TH3D") == 0);
  bool isTProfile = (type.CompareTo("TProfile") == 0);
  bool isTProfile2D = (type.CompareTo("TProfile2D") == 0);

  if (!isTH1F && !isTH1D && !isTH2F && !isTH2D&& !isTH3F && !isTH3D && !isTProfile && !isTProfile2D) {
    errMsg = TString::Format("Unknown histogram type '%s'", type.Data());
    return NULL;
  }

  // read parameter block
  TString parameter;
  TQStringUtils::readBlanksAndNewlines(definition);
  if (TQStringUtils::readBlock(definition, parameter, "()", "''\"\"", false) == 0) {
    errMsg = TString::Format("Missing parameter block '(...)' after '%s'", type.Data());
    return NULL;
  }

  // make sure there is nothing left after the parameter block
  TQStringUtils::readBlanksAndNewlines(definition);
  if (!definition.IsNull()) {
    errMsg = TString::Format("Unrecognized token '%s'", definition.Data());
    return NULL;
  }

  // parse parameter block
  TQTaggable * pars = TQTaggable::parseParameterList(parameter, ",", true, "{}[]()", "''\"\"");
  if (!pars) {
    errMsg = TString::Format("Failed to parse parameters '%s'", parameter.Data());
    return NULL;
  }

  // keep track of parameters read (to find unexpected parameters)
  pars->resetReadFlags();

  // name of histogram
  if (!pars->tagIsOfTypeString("0")) {
    errMsg = "Missing valid histogram name";
    delete pars;
    return NULL;
  }
  TString name = pars->getTagStringDefault("0");

  // title of histogram
  if (!pars->tagIsOfTypeString("1")) {
    errMsg = "Missing valid histogram title";
    delete pars;
    return NULL;
  }
  TString title = pars->getTagStringDefault("1");

  // the number of bins on axes
  int nBinsX = 0;
  int nBinsY = 0;
  int nBinsZ = 0;

  // left and right bound on axes
  double minX = 0.;
  double maxX = 0.;
  double minY = 0.;
  double maxY = 0.;
  double minZ = 0.;
  double maxZ = 0.;

  // parameter index for binning definitions
  int pIndex = 2;

  // vector of bin edges
  std::vector<double> edgesX;
  std::vector<double> edgesY;
  std::vector<double> edgesZ;

  // parse binning of X axis (for 1D, 2D, and 3D histograms)
  if (!extractBinning(pars, pIndex, nBinsX, minX, maxX, edgesX, errMsg)) {
    errMsg.Append(" on X axis");
    delete pars;
    return NULL;
  }

  // parse binning of Y axis (for 2D and 3D histograms)
  if ((isTH2F || isTH2D || isTH3F || isTH3D ||isTProfile2D)  && !extractBinning(pars, pIndex, nBinsY, minY, maxY, edgesY, errMsg)) {

    errMsg.Append(" on Y axis");
    delete pars;
    return NULL;
  }
  if (isTProfile && !extractRange(pars, pIndex, minY, maxY, errMsg)) {
    errMsg.Append(" on Y axis");
    delete pars;
    return NULL;
  }

  // parse binning of Z axis (for 3D histograms)
  if ((isTH3F || isTH3D) && !extractBinning(pars, pIndex, nBinsZ, minZ, maxZ, edgesZ, errMsg)) {
    errMsg.Append(" on Z axis");
    delete pars;
    return NULL;
  }

  // unread parameters left?
  if (pars->hasUnreadKeys()) {
    errMsg.Append(TString::Format("Too many parameters for '%s'", type.Data()));
    delete pars;
    return NULL;
  }

  // handle heterogeneous definition of binning on axes of 3-dimensional histograms
  if (isTH3F || isTH3D) {
    if (edgesX.size()==0 && (edgesY.size()>0 || edgesZ.size()>0)) {
      edgesX = getUniformBinEdges(nBinsX, minX, maxX);
    }
    if (edgesY.size()==0 && (edgesX.size()>0 || edgesZ.size()>0)) {
      edgesY = getUniformBinEdges(nBinsY, minY, maxY);
    }
    if (edgesZ.size()==0 && (edgesX.size()>0 || edgesY.size()>0)) {
      edgesZ = getUniformBinEdges(nBinsZ, minZ, maxZ);
    }
  }

  // the current directory might contain a histogram with the same name
  int i = 2;
  TString finalName = name;
  while (gDirectory && gDirectory->FindObject(name.Data())) {
    name = TString::Format("%s_i%d", finalName.Data(), i++);
  }

  // now create the histogram calling the corresponding constructor
  TH1 * histo = NULL;
  if (isTH1F) {
    if (edgesX.size()>0) {
      histo = new TH1F(name.Data(), title.Data(), nBinsX, &(edgesX[0]));
    } else {
      histo = new TH1F(name.Data(), title.Data(), nBinsX, minX, maxX);
    }
  } else if (isTH1D) {
    if (edgesX.size()>0) {
      histo = new TH1D(name.Data(), title.Data(), nBinsX, &(edgesX[0]));
    } else {
      histo = new TH1D(name.Data(), title.Data(), nBinsX, minX, maxX);
    }
  } else if (isTH2F) {
    if (edgesX.size()>0 && edgesY.size()>0) {
      histo = new TH2F(name.Data(), title.Data(), nBinsX, &(edgesX[0]), nBinsY, &(edgesY[0]));
    } else if (edgesX.size()>0 && edgesY.size()==0) {
      histo = new TH2F(name.Data(), title.Data(), nBinsX, &(edgesX[0]), nBinsY, minY, maxY);
    } else if (edgesX.size()==0 && edgesY.size()>0) {
      histo = new TH2F(name.Data(), title.Data(), nBinsX, minX, maxX, nBinsY, &(edgesY[0]));
    } else {
      histo = new TH2F(name.Data(), title.Data(), nBinsX, minX, maxX, nBinsY, minY, maxY);
    }
  } else if (isTH2D) {
    if (edgesX.size()>0 && edgesY.size()>0) {
      histo = new TH2D(name.Data(), title.Data(), nBinsX, &(edgesX[0]), nBinsY, &(edgesY[0]));
    } else if (edgesX.size()>0 && edgesY.size()==0) {
      histo = new TH2D(name.Data(), title.Data(), nBinsX, &(edgesX[0]), nBinsY, minY, maxY);
    } else if (edgesX.size()==0 && edgesY.size()>0) {
      histo = new TH2D(name.Data(), title.Data(), nBinsX, minX, maxX, nBinsY, &(edgesY[0]));
    } else {
      histo = new TH2D(name.Data(), title.Data(), nBinsX, minX, maxX, nBinsY, minY, maxY);
    }
  } else if (isTH3F) {
    if (edgesX.size()==0 && edgesY.size()==0 && edgesZ.size()==0) {
      histo = new TH3F(name.Data(), title.Data(), nBinsX, minX, maxX,
                       nBinsY, minY, maxY, nBinsZ, minZ, maxZ);
    } else {
      histo = new TH3F(name.Data(), title.Data(), nBinsX, &(edgesX[0]),
                       nBinsY, &(edgesY[0]), nBinsZ, &(edgesZ[0]));
    }
  } else if (isTH3D) {
    if (edgesX.size()==0 && edgesY.size()==0 && edgesZ.size()==0) {
      histo = new TH3D(name.Data(), title.Data(), nBinsX, minX, maxX,
                       nBinsY, minY, maxY, nBinsZ, minZ, maxZ);
    } else {
      histo = new TH3D(name.Data(), title.Data(), nBinsX, &(edgesX[0]),
                       nBinsY, &(edgesY[0]), nBinsZ, &(edgesZ[0]));
    }
  } else if (isTProfile) {
    if (edgesX.size()>0) {
      histo = new TProfile(name.Data(), title.Data(), nBinsX, &(edgesX[0]), minY, maxY);
    } else {
      histo = new TProfile(name.Data(), title.Data(), nBinsX, minX, maxX, minY, maxY);
    }
    
  } else if (isTProfile2D) {
    if (edgesX.size()>0 && edgesY.size()>0) {
      histo = new TProfile2D(name.Data(), title.Data(), nBinsX, &(edgesX[0]), nBinsY, &(edgesY[0]));
    } else if (edgesX.size()>0 && edgesY.size()==0) {
      histo = new TProfile2D(name.Data(), title.Data(), nBinsX, &(edgesX[0]), nBinsY, minY, maxY);
    } else if (edgesX.size()==0 && edgesY.size()>0) {
      histo = new TProfile2D(name.Data(), title.Data(), nBinsX, minX, maxX, nBinsY, &(edgesY[0]));
    } else {
      histo = new TProfile2D(name.Data(), title.Data(), nBinsX, minX, maxX, nBinsY, minY, maxY);
    }
  }
 
  if (histo) {
    // don't put new histogram into "current directory"
    histo->SetDirectory(NULL);
    // set original (final) name
    histo->SetName(finalName.Data());
    // sum weights
    histo->Sumw2();
  }

  delete pars;

  // finally return the histogram
  if(!histo){
    errMsg = "unknown error: histogram type is '"+type+"'";
  }
  return histo;
}


//__________________________________________________________________________________|___________

std::vector<double> TQHistogramUtils::getUniformBinEdges(int nBins, double min, double max) {
  // Returns a pointer to a vector of doubles listing bin edges of <nBins> bins
  // between <min> and <max>. The resulting vector will have <nBins> + 1 entries
  // (the user is responsible for deleting the returned vector). A null pointer is
  // returned in case <nBins> is smaller than one or <max> is not larger than <min>.

  // create a new vector of doubles
  std::vector<double> edges;
  
  if (nBins < 1 || max <= min) {
    // invalid input
    return edges;
  }
 
  // width of one bin
  double width = (max - min) / (double)nBins;
 
  // set equidistant bin edges
  for (int i = 0; i <= nBins; i++) {
    edges.push_back(min + (double)i * width);
  }
 
  // return vector
  return edges;
}


//__________________________________________________________________________________|___________

bool TQHistogramUtils::extractBinning(TQTaggable * p, int &index, int &nBins,
                                      double &min, double &max, std::vector<double> &edges, TString &errMsg) {

  // parameter name of number of bins (or array of bin edges)
  TString p_bins = TString::Format("%d", index++);

  // check if parameter represents a list
  int i = 0;
  double edge = 0.;
  while (p->hasTag(p_bins + TString::Format(".%d", i))) {
    if (!p->getTagDouble(p_bins + TString::Format(".%d", i), edge)) {
      errMsg = "Invalid array of bin edges";
      return false;
    }
    if (edges.size() > 0 && edges.back() >= edge) {
      errMsg = "Bin edges need to be in increasing order";
      return false;
    }
    edges.push_back(edge);
    i++;
  }

  if (edges.size()==0) {

    // parameter names of min and max in axis
    TString p_min = TString::Format("%d", index++);
    TString p_max = TString::Format("%d", index++);
 
    // number of bins of histogram
    if (!p->tagIsOfTypeInteger(p_bins)) {
      errMsg = "Missing valid number of bins";
      return false;
    }
 
    // lower bound of histogram
    if (!(p->tagIsOfTypeInteger(p_min) || p->tagIsOfTypeDouble(p_min))) {
      errMsg = "Missing valid lower bound";
      return false;
    }
 
    // upper bound of histogram
    if (!(p->tagIsOfTypeInteger(p_max) || p->tagIsOfTypeDouble(p_max))) {
      errMsg = "Missing valid upper bound";
      return false;
    }
 
    // now really read parameter
    nBins = p->getTagIntegerDefault(p_bins);
    min = p->getTagDoubleDefault(p_min);
    max = p->getTagDoubleDefault(p_max);

    // check range on axis is not empty
    if (min >= max) {
      errMsg = "Empty range";
      return false;
    }

  } else {
    nBins = edges.size() - 1;
  }

  // check number of bins is correct
  if (nBins <= 0) {
    errMsg = "Number of bins needs to be larger than zero";
    return false;
  }

  return true;
}

//__________________________________________________________________________________|___________

bool TQHistogramUtils::extractRange(TQTaggable * p, int &index,
                                      double &min, double &max, TString &errMsg) {


  // parameter names of min and max in axis
  TString p_min = TString::Format("%d", index++);
  TString p_max = TString::Format("%d", index++);

    
  // lower bound of histogram
  if (!(p->tagIsOfTypeInteger(p_min) || p->tagIsOfTypeDouble(p_min))) {
    errMsg = "Missing valid lower bound";
    return false;
  }

  // upper bound of histogram
  if (!(p->tagIsOfTypeInteger(p_max) || p->tagIsOfTypeDouble(p_max))) {
    errMsg = "Missing valid upper bound";
    return false;
  }

  // now really read parameter
  min = p->getTagDoubleDefault(p_min);
  max = p->getTagDoubleDefault(p_max);

  // check range on axis is not empty
  if (min >= max) {
    errMsg = "Empty range";
    return false;
  }

  return true;
}

//__________________________________________________________________________________|___________

TString TQHistogramUtils::getDetailsAsString(TGraph * g, int/*option*/) {
	// retrieve details of a TGraph as a string
  return TString::Format("%d points",g->GetN());
}

//__________________________________________________________________________________|___________

TString TQHistogramUtils::getDetailsAsString(TGraph2D * g, int/*option*/) {
	// retrieve details of a TGraph2D as a string
  return TString::Format("%d points",g->GetN());
}

//__________________________________________________________________________________|___________

TString TQHistogramUtils::getDetailsAsString(TH1 * histo, int option) {
  // Returns a string summarizing properties of the histogram passed as argument
  // <histo>. The optional parameter <option> allows to control the degree of detail
  // of the resulting string (<option> is 1 by default):
  //
  // - <option> == 0 prints the number of bins on each axis
  // - <option> == 1 additionally prints the sum of weights an the corresponding
  // uncertainty (from root of sum of squares of weights)
  // - <option> == 2 additionally prints the ranges and binning of axes
  // - <option> == 3 additionally prints the units on axes
  // - <option> == 4 additionally prints the plotting style


  // check dimensionality of histogram
  int dim = TQHistogramUtils::getDimension(histo);

  // dim == 0 means invalid pointer
  if (dim == 0) {
    return TString("Invalid histogram");
  }

  // get details of X axis
  TString details = getDetailsAsString(histo->GetXaxis(), option);

  // in case histogram has more than one dimension ...
  if (dim > 1) {
    // ... add details of second dimension
    details.Append(TString(" X ") + getDetailsAsString(((TH2*)histo)->GetYaxis(), option));
  }
  // in case histogram has more than two dimension ...
  if (dim > 2) {
    // ... add details of third dimension
    details.Append(TString(" X ") + getDetailsAsString(((TH3*)histo)->GetZaxis(), option));
  }

  // "bins" label is added by getDetailsAsString(...) on TAxis for option >= 2
  if (option < 2) {
    details.Append(" bin(s)");
  }

  // append the sum of weights and the corresponding uncertainty if option > 0
  if (option > 0) {
    double err = 0.;
    double sum = TQHistogramUtils::getIntegralAndError(histo, err);
    details.Append(TString::Format(", S(w) = %g +/- %g, N=%d", sum, err,(int)(histo->GetEntries())));
  }
 
  if(option > 3){
    details.Append(TString::Format(" - linecolor=%d, fillcolor=%d, fillstyle=%d, markercolor=%d, markerstyle=%d, markersize=%.1f",histo->GetLineColor(),histo->GetFillColor(),histo->GetFillStyle(),histo->GetMarkerColor(),histo->GetMarkerStyle(),histo->GetMarkerSize()));
  }

  // return the string
  return details;
}


//__________________________________________________________________________________|___________

TString TQHistogramUtils::getDetailsAsString(TAxis * axis, int option) {
  // Returns a string summarizing properties of the axis passed as argument
  // <axis>. The optional parameter <option> allows to control the degree of detail
  // of the resulting string (<option> is 1 by default):
  //
  // - <option> == 1 prints the number of bins on axis
  // - <option> == 2 additionally prints the ranges and binning of axis
  // - <option> == 3 additionally prints the units on axis

  if (!axis) {
    // invalid axis
    return TString("");
  } else if (option < 2) {
    // just the number of bins on axis
    TString retval = TString::Format("%d", axis->GetNbins());
    if(axis->GetLabels()){
      retval.Append("l");
    }
    return retval;
  } else {
    // an optional additional "s" if more than one bin
    TString s;
    if (axis->GetNbins() > 1) {
      s = "s";
    }

    // add unit of quantity (extracted from label as "... [unit]") if option > 3
    TString unit;
    if (option > 2) {
      unit = TQStringUtils::getUnit(axis->GetTitle());
      if (!unit.IsNull()) {
        unit.Prepend(" ");
      }
    }
    if(axis->GetLabels()){
      unit.Append(", labeled");
    }

    // compile the final string
    return TString::Format("(%g ... %g%s, %d bin%s)", axis->GetBinLowEdge(1),
                           axis->GetBinUpEdge(axis->GetNbins()), unit.Data(), axis->GetNbins(), s.Data());
  }
}


//__________________________________________________________________________________|___________

int TQHistogramUtils::getNDips(TH1 * histo) {
  // Counts and returns the number of dips (bins without entries surrounded by bins
  // with non-zero entries) in input histogram <histo>. -1 is returned in case an
  // invalid input histogram is provided. Please note: currently, only one and two
  // dimensional histograms are supported.

  // will be the number of dips
  int nDips = 0;

  // histogram properties
  int dim = getDimension(histo);
 
  if (dim == 1) {
    // scan histogram for empty bin between non-empty bins
    int nX = histo->GetXaxis()->GetNbins();
    for (int iX = 0; iX <= nX - 1; iX++) {
      if ((histo->GetBinContent(iX) > 0. || histo->GetBinError(iX) > 0.) &&
          (histo->GetBinContent(iX + 1) == 0. && histo->GetBinError(iX + 1) == 0.) && 
          (histo->GetBinContent(iX + 2) > 0. || histo->GetBinError(iX + 2) > 0.)) {
        nDips++;
      }
    } 
  } else if (dim == 2) {
    // scan histogram for empty bin surrounded by non-empty bins
    int nX = histo->GetNbinsX();
    int nY = histo->GetNbinsY();
    for (int iX = 0; iX <= nX - 1; iX++) {
      for (int iY = 0; iY <= nY - 1; iY++) {
        if ((histo->GetBinContent(iX, iY) > 0. || histo->GetBinError(iX, iY) > 0.) &&
            (histo->GetBinContent(iX, iY + 1) > 0. || histo->GetBinError(iX, iY + 1) > 0.) &&
            (histo->GetBinContent(iX, iY + 2) > 0. || histo->GetBinError(iX, iY + 2) > 0.) &&
            (histo->GetBinContent(iX + 1, iY) > 0. || histo->GetBinError(iX + 1, iY) > 0.) && 
            (histo->GetBinContent(iX + 1, iY + 1) == 0. && histo->GetBinError(iX + 1, iY + 1) == 0.) && 
            (histo->GetBinContent(iX + 1, iY + 2) > 0. || histo->GetBinError(iX + 1, iY + 2) > 0.) && 
            (histo->GetBinContent(iX + 2, iY) > 0. || histo->GetBinError(iX + 2, iY) > 0.) &&
            (histo->GetBinContent(iX + 2, iY + 1) > 0. || histo->GetBinError(iX + 2, iY + 1) > 0.) &&
            (histo->GetBinContent(iX + 2, iY + 2) > 0. || histo->GetBinError(iX + 2, iY + 2) > 0.)) {
          nDips++;
        }
      } 
    } 
  } else {
    nDips = -1;
  }
 
  return nDips;
}


//__________________________________________________________________________________|___________

bool TQHistogramUtils::getMaxArea2D(TH2 * histo, double frac, int &maxX, int &maxY,
                                    int &maxX_low, int &maxX_high, int &maxY_low, int &maxY_high) {
  // Scans the 2D input histogram <histo> for the maximum bin as well as the area
  // around the maximum bin where the bin content exceeds <frac> times the maximum
  // and returns true in case of success or false otherwise. The coordinates of the
  // maximum bin are stored in <maxX> and <maxY> while the span of the maximum area
  // is stored in <maxX_low>, <maxX_high>, <maxY_low>, and <maxY_high>.

  // stop if input histogram is invalid or not a 2D histogram
  if (getDimension(histo) != 2) {
    return false;
  }

  // the bin with maximum bin content
  int iBinX, iBinY, iBinZ;
  int iMaxBin = histo->GetMaximumBin();
  histo->GetBinXYZ(iMaxBin, iBinX, iBinY, iBinZ);

  // the maximum bin content
  double max = histo->GetBinContent(iBinX, iBinY);

  // boundaries of the area exceeding a certain threshold
  // (here, the starting point is the maximum bin)
  int iBinX_low = iBinX;
  int iBinX_high = iBinX;
  int iBinY_low = iBinY;
  int iBinY_high = iBinY;

  // find the area exceeding a certain threshold by starting from
  // the maximum bin and going into each direction up to the bin
  // where the bin contents drops below the threshold
  for (int i = iBinX; i <= histo->GetNbinsX(); i++)
    if (histo->GetBinContent(i, iBinY) > frac * max)
      iBinX_high = i;
  for (int i = iBinX; i > 0; i--)
    if (histo->GetBinContent(i, iBinY) > frac * max)
      iBinX_low = i;
  for (int i = iBinY; i <= histo->GetNbinsY(); i++)
    if (histo->GetBinContent(iBinX, i) > frac * max)
      iBinY_high = i;
  for (int i = iBinY; i > 0; i--)
    if (histo->GetBinContent(iBinX, i) > frac * max)
      iBinY_low = i;

  maxX = iBinX;
  maxY = iBinY;
  maxX_high = iBinX_high;
  maxX_low = iBinX_low;
  maxY_high = iBinY_high;
  maxY_low = iBinY_low;

  return true;
}

//__________________________________________________________________________________|___________

std::vector<double> TQHistogramUtils::getBinLowEdges(TH1* histo, const std::vector<int>& binBorders){
  /* create the low-edges array */
  std::vector<double> lowEdges;
  const size_t nBins = binBorders.size()+1;
  
  /* set the low-edges of rebinned histogram */
  for (size_t i = 0; i < nBins + 1; ++i) {
    if (i == 0)
      /* the left most bin */
      lowEdges.push_back(histo->GetBinLowEdge(1));
    else if (i == nBins)
      /* the right most bin */
      lowEdges.push_back(histo->GetBinLowEdge(histo->GetNbinsX() + 1));
    else
      /* bins in-between */
      lowEdges.push_back(histo->GetBinLowEdge(binBorders[i - 1]));
  }
  return lowEdges;
}

//__________________________________________________________________________________|___________

std::vector<int> TQHistogramUtils::getBinBorders(TH1* histo, const std::vector<double>& lowEdges){
  /* create the low-edges array */
  std::vector<int> binBorders;
  
  /* set the low-edges of rebinned histogram */
  for (size_t i = 1; i < lowEdges.size()-1; ++i) {
    binBorders.push_back(histo->FindBin(lowEdges[i]));
  }
  return binBorders;
}

//__________________________________________________________________________________|___________
  
TH1 * TQHistogramUtils::getRebinned(TH1 * histo, const std::vector<int>& binBorders, bool doRemap) {
  std::vector<double> lowEdges = TQHistogramUtils::getBinLowEdges(histo,binBorders);
  return TQHistogramUtils::getRebinned(histo, binBorders, lowEdges, doRemap);
}

//__________________________________________________________________________________|___________

TH1 * TQHistogramUtils::getRebinned(TH1 * histo, const std::vector<double>& lowEdges, bool doRemap) {
  std::vector<int> binBorders = TQHistogramUtils::getBinBorders(histo,lowEdges);
  return TQHistogramUtils::getRebinned(histo, binBorders, lowEdges, doRemap);
}

//__________________________________________________________________________________|___________

TH1 * TQHistogramUtils::getRebinned(TH1 * histo, const std::vector<int>& binBorders, const std::vector<double>& lowEdges, bool doRemap) {

  /* stop if input histogram or bin border array is invalid */
  if (!histo || getDimension(histo) != 1)
    return 0;

  /* the number of bins after rebinning */
  const int nBins = binBorders.size() + 1;

  TH1 * newHisto = 0;
  if(doRemap){
    /* create the remapped histogram */
    if (histo->InheritsFrom(TH1F::Class()))
      newHisto = new TH1F(histo->GetName(), histo->GetTitle(),nBins, 0., 1.);
    else if (histo->InheritsFrom(TH1D::Class()))
      newHisto = new TH1D(histo->GetName(), histo->GetTitle(),nBins, 0., 1.);

    if(newHisto){
      /* set title of x axis */
      TString label = TQStringUtils::getWithoutUnit(histo->GetXaxis()->GetTitle());
      label.Prepend("Remapped ");
      newHisto->GetXaxis()->SetTitle(label.Data());
    }
  } else {
      
    if (histo->InheritsFrom(TH1F::Class()))
      newHisto = new TH1F(histo->GetName(), histo->GetTitle(), nBins, &(lowEdges[0]));
    else if (histo->InheritsFrom(TH1D::Class()))
      newHisto = new TH1D(histo->GetName(), histo->GetTitle(), nBins, &(lowEdges[0]));
    
    if(newHisto){
      newHisto->GetXaxis()->SetTitle(histo->GetXaxis()->GetTitle());
    } 
  }
    
  if (newHisto) {
    newHisto->GetYaxis()->SetTitle(histo->GetYaxis()->GetTitle());

    newHisto->Sumw2();

    /* make the histogram memory resident */
    newHisto->SetDirectory(histo->GetDirectory());

    /* style parameter */
    copyStyle(newHisto, histo);

    /* set the histograms bin content and error */
    for (int i = 0; i < nBins; i++) {
      
      int lowerBin = 0;
      int upperBin = histo->GetNbinsX() + 1;

      if (i > 0)
        lowerBin = binBorders[i - 1] + 1;
      if (i != (nBins - 1))
        upperBin = binBorders[i];
 
      /* set bin content and error */
      double binError = 0.;
      newHisto->SetBinContent(i + 1,histo->IntegralAndError(lowerBin, upperBin, binError));
      newHisto->SetBinError(i + 1, binError);
    }

    if(!doRemap){
      /* set bin content and error of under- and overflow bins */
      newHisto->SetBinContent(0, histo->GetBinContent(0));
      newHisto->SetBinError (0, histo->GetBinError(0));
      newHisto->SetBinContent(nBins + 1,histo->GetBinContent(histo->GetNbinsX() + 1));
      newHisto->SetBinError (nBins + 1,histo->GetBinError(histo->GetNbinsX() + 1));
    }
    
    newHisto->SetEntries(histo->GetEntries());
  }
 
  /* return remapped histogram */
  return newHisto;
}


//__________________________________________________________________________________|___________

TH2 * TQHistogramUtils::getRemapped2D(TH2 * histo, const std::vector<int>& binBorders, bool remapX) {
	// return a remapped version of a 2D histogram with the given bin borders
  if (!histo || getDimension(histo) != 2)
    return 0;

  /* the number of bins after rebinning and the number of "slices" */
  int nSlices = (remapX ? histo->GetNbinsY() : histo->GetNbinsX()) + 2;
  if (binBorders.size() % nSlices != 0) {
    return NULL;
  }
  int nBins = binBorders.size() / nSlices + 1;

  /* create the remapped histogram */
  TH2 * remappedHisto = 0;
  if (histo->InheritsFrom(TH2F::Class())) {
    if (remapX) {
      remappedHisto = new TH2F(histo->GetName(), histo->GetTitle(),
                               nBins, 0., 1., nSlices - 2, histo->GetYaxis()->GetBinLowEdge(1),
                               histo->GetYaxis()->GetBinUpEdge(histo->GetNbinsY()));
    } else {
      remappedHisto = new TH2F(histo->GetName(), histo->GetTitle(),
                               nSlices - 2, histo->GetXaxis()->GetBinLowEdge(1),
                               histo->GetXaxis()->GetBinUpEdge(histo->GetNbinsX()), nBins, 0., 1.);
    }
  } else if (histo->InheritsFrom(TH2D::Class())) {
    if (remapX) {
      remappedHisto = new TH2D(histo->GetName(), histo->GetTitle(),
                               nBins, 0., 1., nSlices - 2, histo->GetYaxis()->GetBinLowEdge(1),
                               histo->GetYaxis()->GetBinUpEdge(histo->GetNbinsY()));
    } else {
      remappedHisto = new TH2D(histo->GetName(), histo->GetTitle(),
                               nSlices - 2, histo->GetXaxis()->GetBinLowEdge(1),
                               histo->GetXaxis()->GetBinUpEdge(histo->GetNbinsX()), nBins, 0., 1.);
    }
  }

  if (remappedHisto) {

    remappedHisto->Sumw2();

    /* make the histogram memory resident */
    remappedHisto->SetDirectory(histo->GetDirectory());

    /* set title of remapped axis */
    if (remapX) {
      TString label = TQStringUtils::getWithoutUnit(histo->GetXaxis()->GetTitle());
      label.Prepend("Remapped ");
      remappedHisto->GetXaxis()->SetTitle(label.Data());
      remappedHisto->GetYaxis()->SetTitle(histo->GetYaxis()->GetTitle());
    } else {
      TString label = TQStringUtils::getWithoutUnit(histo->GetYaxis()->GetTitle());
      label.Prepend("Remapped ");
      remappedHisto->GetYaxis()->SetTitle(label.Data());
      remappedHisto->GetXaxis()->SetTitle(histo->GetXaxis()->GetTitle());
    }

    /* style parameter */
    copyStyle(remappedHisto, histo);

    /* set the histograms bin content and error */
    for (int j = 0; j < nSlices; j++) {
      for (int i = 0; i < nBins; i++) {
 
        int lowerBin = 0;
        int upperBin = remapX ? histo->GetNbinsX() + 1 : histo->GetNbinsY() + 1;
 
        if (i > 0)
          lowerBin = binBorders[j * (nBins - 1) + i - 1] + 1;
        if (i != (nBins - 1))
          upperBin = binBorders[j * (nBins - 1) + i];
 
        /* set bin content and error */
        double binError = 0.;
        if (remapX) {
          remappedHisto->SetBinContent(i + 1, j,
                                       histo->IntegralAndError(lowerBin, upperBin, j, j, binError));
          remappedHisto->SetBinError(i + 1, j, binError);
        } else {
          remappedHisto->SetBinContent(j, i + 1,
                                       histo->IntegralAndError(j, j, lowerBin, upperBin, binError));
          remappedHisto->SetBinError(j, i + 1, binError);
        }
      }
    }
    remappedHisto->SetEntries(histo->GetEntries());
  }

  /* return remapped histogram */
  return remappedHisto;
}


//__________________________________________________________________________________|___________

TH1 * TQHistogramUtils::getRebinnedFlat(TH1 * histo, int nBins) {
	// obtain a version of a 1D histogram remapped to a flat distribution
  if (getDimension(histo) != 1)
    return 0;

  // get the optimal bin borders 
  std::vector<int> borders = getBinBordersFlat(histo, nBins, false);

  // rebin the histogram 
  TH1 * rebinnedHisto = getRebinned(histo, borders, false);

  // return the histogram 
  return rebinnedHisto;
}


//__________________________________________________________________________________|___________

TH1 * TQHistogramUtils::getRemappedFlat(TH1 * histo, int nBins) {
	// obtain a version of a 1D histogram rebinned to a flat distribution
  if (getDimension(histo) != 1)
    return 0;

  // get the optimal bin borders 
  std::vector<int> borders = getBinBordersFlat(histo, nBins, true);

  // rebin the histogram 
  TH1 * rebinnedHisto = getRebinned(histo, borders, true);

  // return the histogram 
  return rebinnedHisto;
}


//__________________________________________________________________________________|___________

TH2 * TQHistogramUtils::getRemappedFlat2D(TH2 * histo, int nBins, bool remapX) {
	// obtain a version of a 2D histogram remapped to a flat distribution
  if (getDimension(histo) != 2)
    return 0;

  // get the optimal bin borders 
  std::vector<int> borders = getBinBordersFlat2D(histo, nBins, remapX, true);

  // rebin the histogram 
  TH2 * remappedHisto = getRemapped2D(histo, borders, remapX);

  // return the histogram 
  return remappedHisto;
}


//__________________________________________________________________________________|___________

TH1 * TQHistogramUtils::cutAndZoomHistogram(TH1 * histo,
                                            int cutBinLowX, int cutBinHighX, int cutBinLowY, int cutBinHighY,
                                            int zoomBinLowX, int zoomBinHighX, int zoomBinLowY, int zoomBinHighY) {
	// cut and zoom a histogram to the given bin boundaries in X, Y and Z
  int dim = TQHistogramUtils::getDimension(histo);
  if (dim == 0) {
    return NULL;
  }

  if (cutBinLowX > 1 && zoomBinLowX >= 0) {
    zoomBinLowX -= cutBinLowX - 1;
    if (zoomBinLowX <= 0) {
      return NULL;
    }
  }
  if (cutBinLowX > 1 && zoomBinHighX >= 0) {
    zoomBinHighX -= cutBinLowX - 1;
    if (zoomBinHighX <= 0) {
      return NULL;
    }
  }
  if (cutBinLowY > 1 && zoomBinLowY >= 0) {
    zoomBinLowY -= cutBinLowY - 1;
    if (zoomBinLowY <= 0) {
      return NULL;
    }
  }
  if (cutBinLowY > 1 && zoomBinHighY >= 0) {
    zoomBinHighY -= cutBinLowY - 1;
    if (zoomBinHighY <= 0) {
      return NULL;
    }
  }

  TH1 * histo_cut = cutHistogram(histo, cutBinLowX, cutBinHighX, cutBinLowY, cutBinHighY);
  if (!histo_cut) {
    return NULL;
  }

  TH1 * histo_zoomed = NULL;
  if (dim == 1) {
    if (zoomBinHighX <= histo_cut->GetNbinsX()) {
      histo_zoomed = cutHistogram(histo_cut, zoomBinLowX, zoomBinHighX,
                                  zoomBinLowY, zoomBinHighY, true, true);
    }
  } else if (dim == 2) {
    if (zoomBinHighX <= histo_cut->GetNbinsX() && zoomBinHighY <= histo_cut->GetNbinsY()) {
      histo_zoomed = cutHistogram(histo_cut, zoomBinLowX, zoomBinHighX,
                                  zoomBinLowY, zoomBinHighY, true, true, true, true);
    }
  }

  delete histo_cut;
  return histo_zoomed;
}


//__________________________________________________________________________________|___________

TH1 * TQHistogramUtils::cutHistogram(TH1 * histo,
                                     int xBinLow, int xBinHigh, int yBinLow, int yBinHigh,
                                     bool keepInUVX, bool keepInOVX, bool keepInUVY, bool keepInOVY) {
	// cut a histogram to the given bin boundaries in X, Y and Z
  // giving a bin boundary of -1 implies no cut.

  int dim = getDimension(histo);
  // only support 1- and 2-dimensional histograms 
  if (dim < 1 || dim > 2) {
    return NULL;
  }

  int nBinsX = histo->GetNbinsX();
  int nEntries = histo->GetEntries();
  // not cutting at all? 
  if (xBinLow < 0) {
    xBinLow = 0;
  }
  if (xBinHigh < 0) {
    xBinHigh = nBinsX + 1;
  }
  // cutting out of range? 
  if (xBinLow > nBinsX || xBinHigh == 0 || xBinHigh < xBinLow || xBinHigh > nBinsX + 1) {
    return NULL;
  }
  // cutting in second dimension of 1D histogram? 
  if (dim < 2 && (yBinLow != -1 || yBinHigh != -1 || keepInUVY || keepInOVY)) {
    return NULL;
  }
  if (xBinHigh > 0 && xBinHigh < nBinsX) {
    nBinsX -= (nBinsX - xBinHigh);
  }
  if (xBinLow > 1) {
    nBinsX -= (xBinLow - 1);
  }
  double xMin = histo->GetXaxis()->GetBinLowEdge(TMath::Max(xBinLow, 1));
  double xMax = histo->GetXaxis()->GetBinUpEdge(TMath::Min(xBinHigh, histo->GetNbinsX()));

  TH1 * newHisto = 0;

  if (dim == 1) {

    if (histo->InheritsFrom(TH1F::Class())) {
      newHisto = new TH1F(histo->GetName(), histo->GetTitle(), nBinsX, xMin, xMax);
    } else if (histo->InheritsFrom(TH1D::Class())) {
      newHisto = new TH1D(histo->GetName(), histo->GetTitle(), nBinsX, xMin, xMax);
    }
    if (newHisto) {
      newHisto->SetDirectory(histo->GetDirectory());
      newHisto->Sumw2();
 
      double val;
      double err;
      for (int xBinOld = xBinLow; xBinOld <= xBinHigh; xBinOld++) {
        int xBinNew = xBinOld;
        if (xBinLow > 1) {
          xBinNew -= (xBinLow - 1);
        }
        if (xBinLow > 0 && xBinNew <= 0) {
          continue;
        }
        if (xBinHigh <= histo->GetNbinsX() && xBinNew > nBinsX) {
          continue;
        }

        val = histo->GetBinContent(xBinOld);
        err = histo->GetBinError(xBinOld);
        if (val != 0.) {
          newHisto->SetBinContent(xBinNew, val);
        }
        if (err != 0.) {
          newHisto->SetBinError(xBinNew, err);
        }
      }

      if (keepInUVX && xBinLow > 0) {
        val = histo->IntegralAndError(0, xBinLow - 1, err);
        if (val != 0.) {
          newHisto->SetBinContent(0, val);
        }
        if (err != 0.) {
          newHisto->SetBinError(0, err);
        }
      }
      if (keepInOVX && xBinHigh < histo->GetNbinsX() + 1) {
        val = histo->IntegralAndError(xBinHigh + 1, histo->GetNbinsX() + 1, err);
        if (val != 0.) {
          newHisto->SetBinContent(nBinsX + 1, val);
        }
        if (err != 0.) {
          newHisto->SetBinError(nBinsX + 1, err);
        }
      }
    }

  } else if (dim == 2) {

    TH2 * histo2d = (TH2*)histo;

    int nBinsY = histo->GetNbinsY();
    if (yBinLow < 0) {
      yBinLow = 0;
    }
    if (yBinHigh < 0) {
      yBinHigh = nBinsY + 1;
    }
    if (yBinLow > nBinsY || yBinHigh == 0 || yBinHigh < yBinLow || yBinHigh > nBinsY + 1) {
      return NULL;
    }
    if (yBinHigh > 0 && yBinHigh < nBinsY) {
      nBinsY -= (nBinsY - yBinHigh);
    }
    if (yBinLow > 1) {
      nBinsY -= (yBinLow - 1);
    }
    double yMin = histo->GetYaxis()->GetBinLowEdge(TMath::Max(yBinLow, 1));
    double yMax = histo->GetYaxis()->GetBinUpEdge(TMath::Min(yBinHigh, histo->GetNbinsY()));

    if (histo->InheritsFrom(TH2F::Class())) {
      newHisto = new TH2F(histo->GetName(), histo->GetTitle(), nBinsX, xMin, xMax, nBinsY, yMin, yMax);
    } else if (histo->InheritsFrom(TH2D::Class())) {
      newHisto = new TH2D(histo->GetName(), histo->GetTitle(), nBinsX, xMin, xMax, nBinsY, yMin, yMax);
    }
    if (newHisto) {
      newHisto->SetDirectory(histo->GetDirectory());
      newHisto->Sumw2();
 
      double val;
      double err;
      for (int xBinOld = xBinLow; xBinOld <= xBinHigh; xBinOld++) {
        int xBinNew = xBinOld;
        if (xBinLow > 1) {
          xBinNew -= (xBinLow - 1);
        }
        if (xBinLow > 0 && xBinNew <= 0) {
          continue;
        }
        if (xBinHigh <= histo->GetNbinsX() && xBinNew > nBinsX) {
          continue;
        }
        for (int yBinOld = yBinLow; yBinOld <= yBinHigh; yBinOld++) {
          int yBinNew = yBinOld;
          if (yBinLow > 1) {
            yBinNew -= (yBinLow - 1);
          }
          if (yBinLow > 0 && yBinNew <= 0) {
            continue;
          }
          if (yBinHigh <= histo->GetNbinsY() && yBinNew > nBinsY) {
            continue;
          }

          val = histo->GetBinContent(xBinOld, yBinOld);
          err = histo->GetBinError(xBinOld, yBinOld);
          if (val != 0.) {
            newHisto->SetBinContent(xBinNew, yBinNew, val);
          }
          if (err != 0.) {
            newHisto->SetBinError(xBinNew, yBinNew, err);
          }

          if (xBinOld == xBinLow && keepInUVX && xBinLow > 0) {
            val = histo2d->IntegralAndError(0, xBinLow - 1, yBinOld, yBinOld, err);
            if (val != 0.) {
              newHisto->SetBinContent(0, yBinNew, val);
            }
            if (err != 0.) {
              newHisto->SetBinError(0, yBinNew, err);
            }
          }
          if (xBinOld == xBinLow && keepInOVX && xBinHigh < histo->GetNbinsX() + 1) {
            val = histo2d->IntegralAndError(xBinHigh + 1, histo->GetNbinsX() + 1, yBinOld, yBinOld, err);
            if (val != 0.) {
              newHisto->SetBinContent(nBinsX + 1, yBinNew, val);
            }
            if (err != 0.) {
              newHisto->SetBinError(nBinsX + 1, yBinNew, err);
            }
          }
          if (yBinOld == yBinLow && keepInUVY && yBinLow > 0) {
            val = histo2d->IntegralAndError(xBinOld, xBinOld, 0, yBinLow - 1, err);
            if (val != 0.) {
              newHisto->SetBinContent(xBinNew, 0, val);
            }
            if (err != 0.) {
              newHisto->SetBinError(xBinNew, 0, err);
            }
          }
          if (yBinOld == yBinLow && keepInOVY && yBinHigh < histo->GetNbinsY() + 1) {
            val = histo2d->IntegralAndError(xBinOld, xBinOld, yBinHigh + 1, histo->GetNbinsY() + 1, err);
            if (val != 0.) {
              newHisto->SetBinContent(xBinNew, nBinsY + 1, val);
            }
            if (err != 0.) {
              newHisto->SetBinError(xBinNew, nBinsY + 1, err);
            }
          }
        }
      }

      if (keepInUVX && keepInUVY && xBinLow > 0 && yBinLow > 0) {
        val = histo2d->IntegralAndError(0, xBinLow - 1, 0, yBinLow - 1, err);
        if (val != 0.) {
          newHisto->SetBinContent(0, 0, val);
        }
        if (err != 0.) {
          newHisto->SetBinError(0, 0, err);
        }
      }
      if (keepInOVX && keepInUVY && xBinHigh < histo->GetNbinsX() + 1 && yBinLow > 0) {
        val = histo2d->IntegralAndError(xBinHigh + 1, histo->GetNbinsX() + 1, 0, yBinLow - 1, err);
        if (val != 0.) {
          newHisto->SetBinContent(nBinsX + 1, 0, val);
        }
        if (err != 0.) {
          newHisto->SetBinError(nBinsX + 1, 0, err);
        }
      }
      if (keepInUVX && keepInOVY && xBinLow > 0 && yBinHigh < histo->GetNbinsY() + 1) {
        val = histo2d->IntegralAndError(0, xBinLow - 1, yBinHigh + 1, histo->GetNbinsY() + 1, err);
        if (val != 0.) {
          newHisto->SetBinContent(0, nBinsY + 1, val);
        }
        if (err != 0.) {
          newHisto->SetBinError(0, nBinsY + 1, err);
        }
      }
      if (keepInOVX && keepInOVY && xBinHigh < histo->GetNbinsX() + 1 && yBinHigh < histo->GetNbinsY() + 1) {
        val = histo2d->IntegralAndError(xBinHigh + 1, histo->GetNbinsX() + 1,
                                        yBinHigh + 1, histo->GetNbinsY() + 1, err);
        if (val != 0.) {
          newHisto->SetBinContent(nBinsX + 1, nBinsY + 1, val);
        }
        if (err != 0.) {
          newHisto->SetBinError(nBinsX + 1, nBinsY + 1, err);
        }
      }
    }
  }

  if (newHisto) {
    /* style parameter */
    copyStyle(newHisto, histo);
    newHisto->GetXaxis()->SetTitle(histo->GetXaxis()->GetTitle());
    newHisto->GetYaxis()->SetTitle(histo->GetYaxis()->GetTitle());
    newHisto->SetEntries(nEntries);
  }

  return newHisto;
}

//__________________________________________________________________________________|___________


TH2 * TQHistogramUtils::removeBins(TH2* in, TString blackList) {
  // copies a (labled) TH2 skipping all bins (X and Y) whose labels match blackList.
  std::vector<TString> bl;
  bl.push_back(blackList);
  return removeBins(in, bl);
}

//__________________________________________________________________________________|___________

TH2 * TQHistogramUtils::removeBins(TH2* in, const std::vector<TString>& blackList) {
  // copies a (labled) TH2 skipping all bins (X and Y) whose labels match any of the patterns
  // in blackList.
  
  if (!in) return NULL;

  //create new TH2 and copy contents
  TH2* retval = static_cast<TH2*>(TQHistogramUtils::copyHistogram(in));
  if (!retval) return NULL;
  //if nothing is blacklisted, we just return the copy
 // if (blackList.size() == 0) return retval;
    
  std::vector<int> binsX;
  std::vector<int> binsY;
  
  //determine bins to keep
  TAxis* xAxis = in->GetXaxis();
  bool keep;
  for (int i=1; i<xAxis->GetNbins()+1; i++) {
    keep = true;
    for(size_t j=0; j<blackList.size() && keep; j++) {
      keep = keep && !TQStringUtils::matches(xAxis->GetBinLabel(i), blackList[j]);
    }
    if (keep) binsX.push_back(i);   
  }
  TAxis* yAxis = in->GetYaxis();
  for (int i=1; i<yAxis->GetNbins()+1; i++) {
    keep = true;
    for(size_t j=0; j<blackList.size() && keep; j++) {
      keep = keep && !TQStringUtils::matches(yAxis->GetBinLabel(i), blackList[j]);
    }
    if (keep) binsY.push_back(i);
  }

  if (binsX.size()== 0 || binsY.size() == 0) return NULL; //resulting histogram would be empty
  
  int nX = binsX.size();
  int nY = binsY.size();

  retval->SetBins(nX, 0., (double)nX, nY, 0., (double)nY);//this resets all bin contents!
    //set bin labels
  for (size_t x = 0; x<binsX.size(); x++) {
    retval->GetXaxis()->SetBinLabel(x+1, in->GetXaxis()->GetBinLabel(binsX[x]));
  }
  for (size_t y = 0; y<binsY.size(); y++) {
    retval->GetYaxis()->SetBinLabel(y+1, in->GetYaxis()->GetBinLabel(binsY[y]));
  }
  
  for (size_t x=0; x<binsX.size(); x++) {
    for (size_t y = 0; y<binsY.size(); y++) {
      retval->SetBinContent(x+1,y+1, in->GetBinContent(binsX[x],binsY[y]));
      retval->SetBinError(x+1,y+1, in->GetBinError(binsX[x],binsY[y]));
    }
  }
  
  return retval;  
}

//__________________________________________________________________________________|___________

namespace {
std::vector<int> getBinBordersFlatPrivate(TH1 * histo, int nBins, int iFirstBin, double meanPerBin, double &quality, bool includeOverflows) {
	
  if (nBins <= 1) {

    /* in case the number of bins aimed for is equal to one,
     * there is no freedom left to set a bin border: we need
     * to set the last bin border to include the rightmost bin */
    quality = TMath::Power(histo->Integral(iFirstBin,
                                           histo->GetNbinsX() + (includeOverflows ? 1 : 0)) - meanPerBin, 2.);
    return std::vector<int>();

  } else {

    /* the optimal bin borders as a function of the first bin border */
    std::vector<double> nxtChi2;
    std::vector<std::vector<int> > nxtBorders;

    int left = iFirstBin;
    int right = histo->GetNbinsX() + 1 - (nBins - (includeOverflows ? 2 : 1));

    double thisBin = 0.;
    int iThisBin = left;
    while (iThisBin < right && thisBin < meanPerBin)
      thisBin += histo->GetBinContent(iThisBin++);

    left = TMath::Max(left, iThisBin - 2);
    right = TMath::Min(right, iThisBin + 1);

    for (int iBin = left; iBin < right; iBin++) {
 
      double thisQuality = 0.;
      nxtBorders.push_back(getBinBordersFlatPrivate(histo, nBins - 1, iBin + 1,
                                                     meanPerBin, thisQuality, includeOverflows));

      /* calculate and save the quality of current binning */
      thisQuality += TMath::Power(histo->Integral(iFirstBin, iBin) - meanPerBin, 2.);
      nxtChi2.push_back(thisQuality);
    }

    /* find the optimal bin borders by looking for the highest quality */
    int minIndex = -1;
    for (unsigned int i = 0; i < nxtChi2.size(); i++)
      if (minIndex < 0 || nxtChi2[i] < nxtChi2[minIndex])
        minIndex = i;

    /* the optimal bin borders to return */
    std::vector<int> binBorders;

    /* get the best bin borders and delete the
     * information about the non-optimal ones */
    for (unsigned int i = 0; i < nxtChi2.size(); i++) {

      if (i == (unsigned int)minIndex) {
        /* compile the list of optimal bin borders */
        binBorders.push_back(left + minIndex);
        for (unsigned int j = 0; j < nxtBorders[i].size(); j++)
          binBorders.push_back(nxtBorders[i][j]);
      }
    }

    /* return the optimal bin borders */
    return binBorders;
  }
}
}

//__________________________________________________________________________________|___________

std::vector<int> TQHistogramUtils::getBinBordersFlat2D(TH2 * histo, int nBins, bool remapX, bool includeOverflows, bool remapSlices) {
	// obtain a new set of boundaries that remap a given histogram to a flat distribution
  if (!histo) {
    std::vector<int>();
  }

  int n = (remapX ? histo->GetNbinsY() : histo->GetNbinsX()) + 2;

  /* the reference histogram and the corresponding bin borders */
  std::vector<int> bordersRef;

  std::vector<int> borders2D;
  for (int i = 0; i < n; i++) {

    TH1 * h_ref = NULL;
    if (bordersRef.size()==0) {
      h_ref = getProjection(histo, remapX);
    } else {
      h_ref = getProjection(histo, remapX, i, i);
    }
    if (h_ref && bordersRef.size() == 0) {
      bordersRef = getBinBordersFlat(h_ref, nBins, includeOverflows);
      delete h_ref;
    }

    for (size_t j = 0; j < bordersRef.size(); j++) {
      borders2D.push_back(bordersRef[j]);
    }

    if (remapSlices) {
      bordersRef.clear();
    }
  }
  return borders2D;
}


//__________________________________________________________________________________|___________

std::vector<int> TQHistogramUtils::getBinBordersFlat(TH1 * histo, int nBins, bool includeOverflows) {
	// obtain a new set of boundaries that remap a given histogram to a flat distribution
  if (getDimension(histo) != 1 || nBins <= 1) {
    return std::vector<int>();
  }

  // calculate the integral 
  double integral = getIntegral(histo);
  if (!includeOverflows) {
    integral -= histo->GetBinContent(0);
    integral -= histo->GetBinContent(histo->GetNbinsX() + 1);
  }

  // calculate the expected mean content per bin 
  double meanPerBin = integral / (double)nBins;

  // return the optimal bin borders 
  double dummy = 0.;
  return getBinBordersFlatPrivate(histo, nBins, includeOverflows ? 0 : 1, meanPerBin, dummy, includeOverflows);
}

//__________________________________________________________________________________|___________

TH1 * TQHistogramUtils::convertTo1D(TH2 * histo, bool alongX, bool includeUnderflowsX,
                                    bool includeOverflowsX, bool includeUnderflowsY, bool includeOverflowsY) {
	// convert a 2D histogram to a 1D histogram by unrolling it
  if (!histo) {
    return NULL;
  }

  int nX = histo->GetNbinsX() + (includeUnderflowsX ? 1 : 0) + (includeOverflowsX ? 1 : 0);
  int nY = histo->GetNbinsY() + (includeUnderflowsY ? 1 : 0) + (includeOverflowsY ? 1 : 0);
  int nBins;
  int nBlocks;
  if (alongX) {
    nBins = nX;
    nBlocks = nY;
  } else {
    nBins = nY;
    nBlocks = nX;
  }

  TH1 * h_result = NULL;
  if (histo->InheritsFrom(TH2F::Class())) {
    h_result = new TH1F(histo->GetName(), histo->GetTitle(), nBins * nBlocks, 0., 1.);
  } else if (histo->InheritsFrom(TH2D::Class())) {
    h_result = new TH1D(histo->GetName(), histo->GetTitle(), nBins * nBlocks, 0., 1.);
  }

  if (h_result) {
    h_result->SetDirectory(histo->GetDirectory());
  } else {
    return NULL;
  }

  h_result->Sumw2();

  TString titleX = TQStringUtils::getWithoutUnit(histo->GetXaxis()->GetTitle());
  TString titleY = TQStringUtils::getWithoutUnit(histo->GetYaxis()->GetTitle());

  TString ufX;
  TString ofX;
  TString ufY;
  TString ofY;
  if (includeUnderflowsX) {
    ufX = "<UF> ";
  }
  if (includeOverflowsX) {
    ofX = " <OF>";
  }
  if (includeUnderflowsY) {
    ufY = "<UF> ";
  }
  if (includeOverflowsY) {
    ofY = " <OF>";
  }

  TString title = "Reordered (%s%s%s) #otimes (%s%s%s)";

  if (alongX) {
    h_result->GetXaxis()->SetTitle(TString::Format(title.Data(), ufX.Data(),
                                                   titleX.Data(), ofX.Data(), ufY.Data(), titleY.Data(), ofY.Data()).Data());
  } else {
    h_result->GetXaxis()->SetTitle(TString::Format(title.Data(), ufY.Data(),
                                                   titleY.Data(), ofY.Data(), ufX.Data(), titleX.Data(), ofX.Data()).Data());
  }

  int i = 1;
  for (int iBlock = 0; iBlock < nBlocks; iBlock++) {
    for (int iBin = 0; iBin < nBins; iBin++) {
      int iX;
      int iY;
      if (alongX) {
        iX = iBin + (includeUnderflowsX ? 0 : 1);
        iY = iBlock + (includeUnderflowsY ? 0 : 1);
      } else {
        iX = iBlock + (includeUnderflowsX ? 0 : 1);
        iY = iBin + (includeUnderflowsY ? 0 : 1);
      }
      h_result->SetBinContent(i, histo->GetBinContent(iX, iY));
      h_result->SetBinError(i, histo->GetBinError(iX, iY));
      i++;
    }
  }

  /* style parameter */
  copyStyle(h_result, histo);

  return h_result;
}


//__________________________________________________________________________________|___________

TH1 * TQHistogramUtils::getProjection(TH1 * histo, bool onX, int binLow, int binHigh) {
  // Creates a one-dimensional projection from a two-dimensional histogram onto any
  // of its axes and returns a pointer to the projected histogram or a NULL pointer
  // in case of failure. The histogram <histo> is projected onto its X axis if
  // <onX> == true and onto its Y axis if <onX> == false. Optionally, the projection
  // range on the axis that is projected out can be chosen via <binLow> and <binHigh>
  // which refer to the first and the last bin to be considered, respectively. The
  // projection range reaches to the minimum (including underflow bin) and/or the
  // maximum (including the overflow bin) if <binLow> and/or <binHigh> are less than
  // zero (both are -1 by default). The projection histogram adopts both style options
  // as well as the association to a TDirectory from the source histogram (ensuring
  // unique names in case of an association to a TDirectory).

  // we expect a 2D histogram as input
  if (getDimension(histo) != 2) {
    // ... return NULL pointer if not a 2D histogram
    return NULL;
  }

  // projection range with lower bin less than zero is
  // treated as zero (= projection starts from underflow bin)
  if (binLow < 0) {
    binLow = 0;
  }

  // projection range with upper bin greater than index of overflow bin or less
  // than zero are treated as index of overflow bin (= projection including overflow bin)
  int nBins = (onX ? histo->GetNbinsY() : histo->GetNbinsX());
  if (binHigh < 0 || binHigh > nBins + 1) {
    binHigh = nBins + 1;
  }

  // expect a non-empty projection range
  if (binLow > binHigh) {
    return NULL;
  }

  // the TDirectory the histogram is associated to
  TDirectory * histoDir = histo->GetDirectory();

  // determine a unique name for the projection histogram
  // to avoid collisions with existing histograms
  TString prefix = TString::Format("__h_proj_%s", histo->GetName());
  TString name = prefix;
  int i = 2;
  while ((gDirectory && gDirectory->FindObject(name.Data()))
         || (histoDir && histoDir->FindObject(name.Data()))) {
    name = TString::Format("%s_i%d", prefix.Data(), i++);
  }

  // make projection using ROOT's projection features
  TH1D * h_proj;
  if (onX) {
    // project onto X axis
    h_proj = ((TH2*)histo)->ProjectionX(name.Data(), binLow, binHigh, "e");
  } else {
    // project onto Y axis
    h_proj = ((TH2*)histo)->ProjectionY(name.Data(), binLow, binHigh, "e");
  }

  // if the projection succeeded ...
  if (h_proj) {
    // put the projected histogram into the same TDirectory as the original one
    // (or remove association if original histogram was not associated to any directory)
    h_proj->SetDirectory(histoDir);
    // restore name of histogram (equal to input histogram) if not associated
    // to a TDirectory (otherwise we would end up in another name collision)
    if (!histoDir) {
      h_proj->SetName(histo->GetName());
    }
    // copy style from original histogram to projected histogram
    copyStyle(h_proj, histo);
  }

  // return the projected histogram
  return h_proj;
}


//__________________________________________________________________________________|___________

TH1 * TQHistogramUtils::getProjectionX(TH1 * histo, int binLow, int binHigh) {
  // Creates a one-dimensional projection from a two-dimensional histogram onto its
  // X axis and returns a pointer to the projected histogram or a NULL pointer
  // in case of failure. Please refer to the documentation of
  // TQHistogramUtils::getProjection(...) for additional information.
  //
  // [Please note: this is wrapper to TQHistogramUtils::getProjection(..., true, ...)]

  // make projection
  return getProjection(histo, true, binLow, binHigh);
}


//__________________________________________________________________________________|___________

TH1 * TQHistogramUtils::getProjectionY(TH1 * histo, int binLow, int binHigh) {
  // Creates a one-dimensional projection from a two-dimensional histogram onto its
  // Y axis and returns a pointer to the projected histogram or a NULL pointer
  // in case of failure. Please refer to the documentation of
  // TQHistogramUtils::getProjection(...) for additional information.
  //
  // [Please note: this is wrapper to TQHistogramUtils::getProjection(..., false, ...)]

  // make projection
  return getProjection(histo, false, binLow, binHigh);
}


//__________________________________________________________________________________|___________

bool TQHistogramUtils::applyPoissonErrors(TH1 * histo) {
  // Sets the bin errors for each bin of histogram histo to square-root of bin
  // contents and returns false in case of an invalid histogram or if there are
  // bins with negative entries or true otherwise. For bins with negative bin
  // content the bin error is set to zero.
 
  if (!histo) {
    return false;
  }
 
  bool neg = false;
 
  // iterate over bins
  int n = getNBins(histo);
  for (int i = 0; i < n; i++) {
    double bin = histo->GetBinContent(i);
    if (bin >= 0.) {
      histo->SetBinError(i, TMath::Sqrt(bin));
    } else {
      histo->SetBinError(i, 0.);
      neg = true;
    }
  }
 
  return !neg;
}

//__________________________________________________________________________________|___________

void TQHistogramUtils::copyGraphAxisTitles(TNamed* copy, TNamed* graph) {
	// copy axis titles from one graph to the other
  for(size_t idx=0; idx<3; idx++){
    TAxis* oaxis = TQHistogramUtils::getAxis(graph,idx);
    TAxis* caxis = TQHistogramUtils::getAxis(copy,idx);
    if(!oaxis || !caxis) break;
    caxis->SetTitle(oaxis->GetTitle());
  }
}

//__________________________________________________________________________________|___________

TNamed * TQHistogramUtils::copyGraph(TNamed* graph, const TString& newName) {
  // a variant of copyHistogram that works on TGraphs instead
  if (!graph) {
    return NULL;
  }

  // make a copy of the input Graph
  TNamed * copy = TQHistogramUtils::createGraph(TQHistogramUtils::getGraphDefinition(graph));
  if (!copy) {
    return NULL;
  }
  TQHistogramUtils::copyStyle(copy,graph);
  TQHistogramUtils::addGraph(copy,graph);

	TQHistogramUtils::copyGraphAxisTitles(copy,graph);
  
  // set the directory of the original histogram
  if (newName.IsNull()) {
    copy->SetName(graph->GetName());
  } else {
    copy->SetName(newName);
  }
  
  // return the copied graph
  return copy;
}

//__________________________________________________________________________________|___________

TH1 * TQHistogramUtils::copyHistogram(TH1 * histo, const TString& newName) {
  // Creates an independent copy of the input histogram <histo> and returns a pointer
  // to the copy or a NULL pointer in case of failure. Please note: currently, only
  // instances of TH1F, TH1D, TH2F, TH2D, TH3F, and TH3D are supported. Optionally,
  // a new name for the copy can be specified via <newName>. Otherwise, the copied
  // histogram will adopt the name of the input histogram. In any case it will adopt
  // its potential association to an instance of TDirectory (ensuring a unique name).

  // stop if the histogram to copy is invalid
  if (!histo) {
    return NULL;
  }

  // make a copy of the input histogram

  TH1 * copy = TQHistogramUtils::createHistogram(TQHistogramUtils::getHistogramDefinition(histo));
  if (!copy) {
    return NULL;
  }
  TQHistogramUtils::copyStyle(copy,histo);
  TQHistogramUtils::addHistogram(copy,histo);

  if(histo->GetXaxis()) copy->GetXaxis()->SetTitle(histo->GetXaxis()->GetTitle());
  if(histo->GetYaxis()) copy->GetYaxis()->SetTitle(histo->GetYaxis()->GetTitle());
  if(histo->GetZaxis()) copy->GetZaxis()->SetTitle(histo->GetZaxis()->GetTitle());
  
  // set the directory of the original histogram
  TDirectory * histoDir = histo->GetDirectory();
  if (!histoDir || newName == "NODIR") {
    copy->SetDirectory(NULL);
    copy->SetName(histo->GetName());
  } else {
    if (newName.IsNull()) {
      // find a unique name
      TString prefix = histo->GetName();
      TString name = prefix;
      int i = 2;
      while (histoDir->FindObject(name.Data())) {
        name = TString::Format("%s_i%d", prefix.Data(), i++);
      }
      copy->SetName(name.Data());
    } else {
      copy->SetName(newName);
    }
    copy->SetDirectory(histoDir);
  }
  
  // return the copied histogram
  return copy;
}

//__________________________________________________________________________________|___________

bool TQHistogramUtils::copyBinLabels(TH1* source, TH1* target) {
  // copy bin labels from axes of one histogram to the corresponding axes of the 
  // other returns false if no valid arguments are given or source and target 
  // histograms/axes are incompatible (different dimensionality / different number
  // of bins)
  if (!source || !target) return false;
  int dim = abs(TQHistogramUtils::getDimension(source));
  if (dim != abs(TQHistogramUtils::getDimension(target))) return false;
  
  switch (dim) {
    case 3:
      if (!TQHistogramUtils::copyBinLabels(source->GetZaxis(), target->GetZaxis())) return false;
    case 2:
      if (!TQHistogramUtils::copyBinLabels(source->GetYaxis(), target->GetYaxis())) return false;
    case 1:
      if (!TQHistogramUtils::copyBinLabels(source->GetXaxis(), target->GetXaxis())) return false;
      if (TQHistogramUtils::hasBinLabels(target->GetXaxis())) target->GetXaxis()->LabelsOption("v");
      break;
    default:
      WARN("Unsupported histogram dimension '%i' in TQHistogramUtils::copyBinLabels",dim);
      return false;
      break; 
  }
  return true;
}

//__________________________________________________________________________________|___________

bool TQHistogramUtils::hasBinLabels(TH1* h){
  // return true if this histogram has bin labels on any axis
  int dim = abs(TQHistogramUtils::getDimension(h));
  bool retval = TQHistogramUtils::hasBinLabels(h->GetXaxis());
  if(dim > 1) retval = retval && TQHistogramUtils::hasBinLabels(h->GetYaxis());
  if(dim > 2) retval = retval && TQHistogramUtils::hasBinLabels(h->GetZaxis());
  return retval;
}

//__________________________________________________________________________________|___________

bool TQHistogramUtils::hasBinLabels(TAxis* a){
  // return true if this axis has bin labels
  if (!a) return false;
  THashList* labels = a->GetLabels();
  return labels && labels->GetSize()>0;
}

//__________________________________________________________________________________|___________

bool TQHistogramUtils::copyBinLabels(TAxis* source, TAxis* target) {
  // copy bin labels from one axis to another. Returns false if arguments are invalid or axes are incompatible
  if (!source || !target) return false;
  if (!TQHistogramUtils::hasBinLabels(source)) return true; //nothing to do here, axis has no labels //note that this line requires a public-private hack due to the stupidity of ROOT...
  int nBins = source->GetNbins();
  if (nBins != target->GetNbins()) return false;
  for (int i=1; i<= nBins; i++) {
    target->SetBinLabel(i,source->GetBinLabel(i));
  }
  return true;
}

//__________________________________________________________________________________|___________

bool TQHistogramUtils::copyAxisStyle(TH1* source, TH1* target) {
  // copy axis style settings from axes of one histogram to the corresponding axes of the 
  // other returns false if no valid arguments are given or source and target 
  // histograms/axes are incompatible (different dimensionality / different number
  // of bins)
  if (!source || !target) return false;
  int dim = abs(TQHistogramUtils::getDimension(source));
  if (dim != abs(TQHistogramUtils::getDimension(target))) return false;
  
  switch (dim) {
    case 3:
      if (!TQHistogramUtils::copyAxisStyle(source->GetZaxis(), target->GetZaxis())) return false;
    case 2:
      if (!TQHistogramUtils::copyAxisStyle(source->GetYaxis(), target->GetYaxis())) return false;
    case 1:
      if (!TQHistogramUtils::copyAxisStyle(source->GetXaxis(), target->GetXaxis())) return false;
      break;
    default:
      WARN("Unsupported histogram dimension '%i' in TQHistogramUtils::copyAxisStyle",dim);
      return false;
      break; 
  }
  return true;

}

//__________________________________________________________________________________|___________

bool TQHistogramUtils::copyAxisStyle(TAxis* source, TAxis* target) {
  //add style settings to copy as needed
  if (!source || !target) return false;
  target->SetAxisColor(source->GetAxisColor());
  target->SetLabelColor(source->GetLabelColor());
  target->SetLabelFont(source->GetLabelFont());
  target->SetLabelOffset(source->GetLabelOffset());
  target->SetLabelSize(source->GetLabelSize());
  target->SetNdivisions(source->GetNdivisions());
  target->SetTickLength(source->GetTickLength());
  target->SetTitleColor(source->GetTitleColor());
  target->SetTitleFont(source->GetTitleFont());
  target->SetTitleOffset(source->GetTitleOffset());
  target->SetTitleSize(source->GetTitleSize());
  return true;
}

//__________________________________________________________________________________|___________

bool TQHistogramUtils::includeSystematics(TH1 * histo, TH1 * systematics) {

  /* stop if input histograms are invalid */
  if (!checkConsistency(histo, systematics))
    return false;

  /* only one dimensional histograms supported up to now */
  if (getDimension(histo) != 1)
    return false;

  /* loop on bins */
  int nBins = histo->GetNbinsX();
  for (int iBin = 0; iBin < nBins + 2; iBin++) {

    /* the new total bin error */
    double binErrorSquared = TMath::Power(histo->GetBinError(iBin), 2.);

    /* include systematics */
    binErrorSquared += TMath::Power(systematics->GetBinError(iBin), 2.);
    histo->SetBinError(iBin, TMath::Sqrt(binErrorSquared) + systematics->GetBinContent(iBin));
  }

  /* operation finished successfully */
  return true;
}


//__________________________________________________________________________________|___________

TH1 * TQHistogramUtils::getSystematics(TH1 * h_nominal, TList * singleVars, TList * pairVars) {

  /* get the input variations */
  int nSingleVars = (singleVars ? singleVars->GetEntries() : 0);
  int nPairVars = (pairVars ? pairVars->GetEntries() : 0);

  /* stop if number of Up/Down paired variations is not an even number */
  if (nPairVars % 2 == 1)
    return 0;

  /* use nominal input histogram as binning template */
  TH1 * systematics = copyHistogram(h_nominal);

  /* stop if copy is invalid */
  if (!h_nominal)
    return 0;

  /* reset bin content */
  systematics->Reset();

  /* loop over histogram bins */
  int nBins = getNBins(systematics);
  for (int iBin = 0; iBin < nBins; iBin++) {

    /* nominal bin content */
    double nom = h_nominal->GetBinContent(iBin);

    /* the total systematic uncertainty of this bin */
    double totalSysSquared = 0.;

    /* loop and include single variations */
    for (int iVar = 0; iVar < nSingleVars; iVar++) {

      /* get the list object */
      TObject * obj = singleVars->At(iVar);

      /* check the histograms validity */
      if (obj && obj->InheritsFrom(TH1::Class())
          && checkConsistency(h_nominal, (TH1*)obj))
        /* add the systematic uncertainty of this source */
        totalSysSquared += TMath::Power(((TH1*)obj)->GetBinContent(iBin) - nom, 2.);
      else
        /* invalid histogram...stop? */
        return 0;
    }

    /* loop and include Up/Down pair variations */
    for (int iVar = 0; iVar < (nPairVars / 2); iVar++) {

      /* get the list object */
      TObject * obj0 = pairVars->At(2*iVar);
      TObject * obj1 = pairVars->At(2*iVar + 1);

      /* check the histograms validity */
      if (obj0 && obj0->InheritsFrom(TH1::Class()) 
          && checkConsistency(h_nominal, (TH1*)obj0) &&
          obj1 && obj1->InheritsFrom(TH1::Class())
          && checkConsistency(h_nominal, (TH1*)obj1))
        /* add the systematic uncertainty of this source */
        totalSysSquared += .5 * (
                                 TMath::Power(((TH1*)obj0)->GetBinContent(iBin) - nom, 2.) +
                                 TMath::Power(((TH1*)obj1)->GetBinContent(iBin) - nom, 2.));
      else
        /* invalid histogram...stop? */
        return 0;
    }

    systematics->SetBinError(iBin, TMath::Sqrt(totalSysSquared));
  }

  /* return the systematics histogram */
  return systematics;
}

//__________________________________________________________________________________|___________


bool TQHistogramUtils::addHistogram(TH1 * histo1, TH1 * histo2, TQCounter* scale, double corr12, bool includeScaleUncertainty) {
  // add two histograms, just like TH1::Add does 
  // this function will handle scale uncertainties and possible correlations
  // between the two histograms properly on a bin-by-bin basis
  if(scale)
    return TQHistogramUtils::addHistogram(histo1,histo2,scale->getCounter(),scale->getError(),corr12,includeScaleUncertainty);
  else
    return TQHistogramUtils::addHistogram(histo1,histo2,1.,0.,corr12,false);
}


//__________________________________________________________________________________|___________


bool TQHistogramUtils::addHistogram(TH1 * histo1, TH1 * histo2, double scale, double scaleUncertainty, double corr12, bool includeScaleUncertainty) {
  // add two histograms, just like TH1::Add does 
  // this function will handle scale uncertainties and possible correlations
  // between the two histograms properly on a bin-by-bin basis

  if (!histo1 || !histo2 || !checkConsistency(histo1, histo2)) {
    // check validity of input histograms
    return false;
  }

  TProfile* p1 = dynamic_cast<TProfile*>(histo1);
  TProfile* p2 = dynamic_cast<TProfile*>(histo2);
  if(p1 && p2){
    p1->Add(p2,scale);
  } else {
    const int n1 = histo1->GetEntries();
    const int n2 = histo2->GetEntries();  

    int nbins = TQHistogramUtils::getNbinsGlobal(histo1);
    int binmin = 0;
    int binmax = nbins+1;
    if(((*histo1->GetXaxis()).*TAxisHackResult<TAxisIsAlphanumeric>::ptr)() ){
      binmin = 0;
      binmax = nbins-1;
    }

    for(int i=binmin; i<binmax; ++i){
      const double val1 = histo1->GetBinContent(i);
      const double err1 = histo1->GetBinError(i);
      const double val2 = histo2->GetBinContent(i);
      const double err2 = histo2->GetBinError(i);
      const double val = val1 + scale*val2;
      histo1->SetBinContent(i,val);
      if(includeScaleUncertainty){
        const double err_squared = err1*err1 + err2*err2*scale*scale + val2*val2*scaleUncertainty*scaleUncertainty + 2 * corr12 * scale * err1 * err2;
        histo1->SetBinError (i,sqrt(err_squared));
      } else {
        const double err_squared = err1*err1 + err2*err2*scale*scale + 2 * corr12 * scale * err1 * err2;
        histo1->SetBinError (i,sqrt(err_squared));
      }
    }
    histo1->SetEntries(n1+n2);
  }
  return true;
}


//__________________________________________________________________________________|___________


bool TQHistogramUtils::scaleHistogram(TH1 * histo, TQCounter* scale, bool includeScaleUncertainty) {
  // scales a histogram, just like TH1::Scale does
  // this function will handle scale uncertainties if requested
  if(scale)
    return TQHistogramUtils::scaleHistogram(histo,scale->getCounter(),scale->getError(),includeScaleUncertainty);
  else
    return TQHistogramUtils::scaleHistogram(histo,1.,0.,false);
}

//__________________________________________________________________________________|___________

bool TQHistogramUtils::scaleHistogram(TH1 * histo, double scale, double scaleUncertainty, bool includeScaleUncertainty) {
  // scales a histogram, just like TH1::Scale does
  // this function will handle scale uncertainties if requested
  if(histo->InheritsFrom(TProfile::Class())) return false;
  const int nentries(histo->GetEntries());
  for(int i=0; i<TQHistogramUtils::getNbinsGlobal(histo); ++i){
    const double val = histo->GetBinContent(i);
    const double err = histo->GetBinError(i);
    const double val_new = val * scale;
    histo->SetBinContent(i,val_new);
    if(includeScaleUncertainty){
      histo->SetBinError (i,sqrt(err * err * scale * scale + val * val * scaleUncertainty*scaleUncertainty));
    } else {
      histo->SetBinError (i,err * scale);
    }
  }
  histo->SetEntries(nentries);
  return true;
}

//__________________________________________________________________________________|___________

bool TQHistogramUtils::drawHistograms(TList * histograms,
                                      TString drawOption, TString extOptions) {
	// draw a list of histograms
  if (!histograms)
    return false;

  bool applyColors = false;
  bool drawLegend = false;
  TString legendOptions = "l";
  bool setTitle = false;
  TString title;

  if (!extOptions.IsNull()) {

    /* parse the extended options string */
    TQTaggable * tags = new TQTaggable(extOptions);

    tags->getTagBool ("applyColors", applyColors);
    tags->getTagBool ("legend", drawLegend);
    tags->getTagString ("legend.drawOptions", legendOptions);

    setTitle = tags->getTagString("title", title);

    /* delete the taggable object */
    delete tags;
  }

  /* true if every entry in the input list is valid */
  bool success = true;

  /* false for the first histogram to draw */
  bool drawSame = false;


  /* ===== create the legend to draw ===== */

  /* the legend to draw */
  TLegend * legend = 0;

  if (drawLegend) {
    /* create the legend */
    legend = new TLegend(0.65, 0.65, .95, .9);
    /* set the legend's options */
    legend->SetFillColor(kWhite);
  }


  /* ===== draw the histograms ===== */

  /* the first color */
  Color_t color = 1;

  /* loop over entries in input list */
  TIterator * itr = histograms->MakeIterator();
  TObject * obj;
  while ((obj = itr->Next())) {

    /* check the entry's validity */
    if (obj->InheritsFrom(TH1::Class())) {

      /* the histogram */
      TH1 * histo = (TH1*)obj;

      /* set the line color if requested */
      if (applyColors)
        histo->SetLineColor(color++);

      /* add an entry to the legend (if requested) */
      if (legend)
        legend->AddEntry(histo, histo->GetTitle(), legendOptions.Data());

      /* set the title */
      if (setTitle)
        histo->SetTitle(title.Data());

      /* draw the histogram */
      if (drawSame)
        histo->Draw(TString::Format("%s same", drawOption.Data()).Data());
      else
        histo->Draw(drawOption.Data());

      /* the next histograms will be drawn with "same" option */
      drawSame = true;

    } else {
      success = false;
    }
  }

  /* draw the legend if requested */
  if (legend)
    legend->Draw("same");

  /* delete iterator */
  delete itr;

  /* return true if drawing was successful */
  return success;
}


//__________________________________________________________________________________|___________

bool TQHistogramUtils::resetBinErrors(TH1 * histo) {
	// reset all the bin errors on a histogram

  if (!histo)
    return false;

  // loop on bins and reset bin errors 
  int nBins = getNBins(histo);
  for (int iBin = 0; iBin < nBins; iBin++)
    histo->SetBinError(iBin, 0.);

  // operation finished successfully
  return true;
}

//__________________________________________________________________________________|___________

int TQHistogramUtils::getDimension(TH1 * histo) {
  // Returns the dimensionality of the histogram <histo> (e.g. 1 for TH1F, 2 for
  // TH2D, ...). Returns 0 if the input histogram is invalid.

  if (!histo) {
    // is an invalid pointer
    return 0;
  } else if (histo->InheritsFrom(TProfile2D::Class())) {
    // is a TProfile2D
    return -2;
  } else if (histo->InheritsFrom(TProfile::Class())) {
    // is a TProfile
    return -1;
  } else if (histo->InheritsFrom(TH3::Class())) {
    // is a 3D histogram
    return 3;
  } else if (histo->InheritsFrom(TH2::Class())) {
    // is a 2D histogram
    return 2;
  } else {
    // must be a 1D histogram (no??)
    return 1;
  }
}

//__________________________________________________________________________________|___________

TAxis*  TQHistogramUtils::getAxis(TNamed* obj, int idx){
  // get the X (idx=0), Y (idx=1) or Z (idx=3) axis of a TH1 or TGraph or TGraph2D object
  if(obj->InheritsFrom(TH1::Class())){
    TH1* hist = (TH1*)obj;
    switch(idx){
    case 0: return hist->GetXaxis();
    case 1: return hist->GetYaxis(); 
    case 2: return hist->GetZaxis();
    default: return NULL;
    }
  }
  if(obj->InheritsFrom(TGraph::Class())){
    TGraph* graph = (TGraph*)obj;
    switch(idx){
    case 0: return graph->GetXaxis();
    case 1: return graph->GetYaxis(); 
    default: return NULL;
    }
  }
  if(obj->InheritsFrom(TGraph2D::Class())){
    TGraph2D* graph = (TGraph2D*)obj;
    switch(idx){
    case 0: return graph->GetXaxis();
    case 1: return graph->GetYaxis(); 
    case 2: return graph->GetZaxis(); 
    default: return NULL;
    }
  }
  return NULL;
}

//__________________________________________________________________________________|___________

int TQHistogramUtils::getNBins(TH1 * histo, bool includeUnderflowOverflow) {
  // Returns the total number of bins of the input histogram <histo> including over-
  // and underflow bins as well as all dimensions of the histogram or 0 in case of
  // failure. That is, for a 10x10 bins 2D histogram 144 = (10+2)*(10+2) is returned.

  /* stop if histogram is invalid */
  if (!histo)
    return 0;

  /* the number of bins */
  int nBins = histo->GetNbinsX() + (includeUnderflowOverflow ? 2 : 0);

  if (histo->InheritsFrom(TH3::Class())) {
    nBins *= histo->GetNbinsY() + (includeUnderflowOverflow ? 2 : 0);
    nBins *= histo->GetNbinsZ() + (includeUnderflowOverflow ? 2 : 0);
  } else if (histo->InheritsFrom(TH2::Class())) {
    nBins *= histo->GetNbinsY() + (includeUnderflowOverflow ? 2 : 0);
  }

  /* return the number of bins */
  return nBins;
}


//__________________________________________________________________________________|___________

int TQHistogramUtils::getSizePerBin(TH1 * histo) {
	// return the size of bits per bin
  if (!histo)
    return 0;
  else if ( histo->InheritsFrom(TH1F::Class()) ||
            histo->InheritsFrom(TH2F::Class()) ||
            histo->InheritsFrom(TH3F::Class()))
    return 2 * 4;
  else if ( histo->InheritsFrom(TH1D::Class()) ||
            histo->InheritsFrom(TH2D::Class()) ||
            histo->InheritsFrom(TH3D::Class()))
    return 2 * 8;
  else
    return 0;
}


//__________________________________________________________________________________|___________

int TQHistogramUtils::estimateSize(TH1 * histo) {
	// estimate the memory size of a histogram
  if (histo)
    return getNBins(histo) * getSizePerBin(histo);
  else
    return 0;
}


//__________________________________________________________________________________|___________

bool TQHistogramUtils::checkConsistency(TH1 * histo1, TH1 * histo2, bool verbose) {
  // Check if two histograms have consistent binning

  if (!histo1 || !histo2){
    if(verbose) ERRORfunc("received NULL pointer");
    return false;
  }

  DEBUGfunc("function called on histograms '%s' and '%s'",histo1->GetName(),histo2->GetName());

  /* get histogram dimensions */
  int dim = getDimension(histo1);

  /* stop if the histograms have different dimensions */
  if (dim != getDimension(histo2)){
    if(verbose) ERRORfunc("inconsistent histogram dimensions");
    return false;
  }

  /* check consistency of first dimension
   * =================================================== */

  /* stop if the number of bins on the X axis are different */
  int nBinsX1 = histo1->GetNbinsX();
  int nBinsX2 = histo2->GetNbinsX();
  if (nBinsX1 != nBinsX2){
    if(verbose) ERRORfunc("inconsistent number of bins in X-direction");
    return false;
  }

  /* stop if the ranges of the X axis are different */
  if (!TMath::AreEqualRel(
                          histo1->GetXaxis()->GetBinLowEdge(0),
                          histo2->GetXaxis()->GetBinLowEdge(0), 1E-12) ||
      !TMath::AreEqualRel(
                          histo1->GetXaxis()->GetBinUpEdge(nBinsX1),
                          histo2->GetXaxis()->GetBinUpEdge(nBinsX2), 1E-12)){
    if(verbose) ERRORfunc("inconsistent axis ranges in X-direction");
    return false;
  }

  /* TODO: check the bin width individually */

  /* for 1dim histograms we are done */
  if (dim < 2)
    return true;

  /* check consistency of second dimension
   * =================================================== */

  /* stop if the number of bins on the Y axis are different */
  int nBinsY1 = histo1->GetNbinsY();
  int nBinsY2 = histo2->GetNbinsY();
  if (nBinsY1 != nBinsY2){
    if(verbose) ERRORfunc("inconsistent number of bins in Y-direction");
    return false;
  }

  /* stop if the ranges of the Y axis are different */
  if (!TMath::AreEqualRel(
                          histo1->GetYaxis()->GetBinLowEdge(0),
                          histo2->GetYaxis()->GetBinLowEdge(0), 1E-12) ||
      !TMath::AreEqualRel(
                          histo1->GetYaxis()->GetBinUpEdge(nBinsY1),
                          histo2->GetYaxis()->GetBinUpEdge(nBinsY2), 1E-12)){
    if(verbose) ERRORfunc("inconsistent axis ranges in Y-direction");
    return false;
  }

  /* TODO: check the bin width individually */

  /* for 2dim histograms we are done */
  if (dim < 3)
    return true;

  /* check consistency of third dimension
   * =================================================== */

  /* stop if the number of bins on the Z axis are different */
  int nBinsZ1 = histo1->GetNbinsZ();
  int nBinsZ2 = histo2->GetNbinsZ();
  if (nBinsZ1 != nBinsZ2){
    if(verbose) ERRORfunc("inconsistent number of bins in Y-direction");
    return false;
  }

  /* stop if the ranges of the Z axis are different */
  if (!TMath::AreEqualRel(
                          histo1->GetZaxis()->GetBinLowEdge(0),
                          histo2->GetZaxis()->GetBinLowEdge(0), 1E-12) ||
      !TMath::AreEqualRel(
                          histo1->GetZaxis()->GetBinUpEdge(nBinsZ1),
                          histo2->GetZaxis()->GetBinUpEdge(nBinsZ2), 1E-12)){
    if(verbose) ERRORfunc("inconsistent axis ranges in Z-direction");
    return false;
  }

  if(dim < 4)
    return true;

  /* TODO: check the bin width individually */

  if(verbose) ERRORfunc("histograms have unknown dimensionality!");
  return false;
}


//__________________________________________________________________________________|___________

TGraphAsymmErrors * TQHistogramUtils::getGraph(TH1 * histo) {
	// convert a histogram into a TGraph
  if (!histo)
    return 0;

  // only 1 dimensional histograms supported 
  if (getDimension(histo) != 1)
    return 0;

  // the number of bins of the histogram 
  int nBins = getNBins(histo);

  // the error graph to return 
  TGraphAsymmErrors * graph = new TGraphAsymmErrors(nBins);
  graph->SetName(TString::Format("g_%s",histo->GetName()));
  graph->SetTitle(histo->GetTitle());

  // loop over bins and set graph point properties 
  for (int iBin = 1; iBin <= nBins; iBin++) {
    // set the point 
    graph->SetPoint(iBin,
                    histo->GetBinCenter(iBin),
                    histo->GetBinContent(iBin));
    /* set the point error */
    graph->SetPointError(iBin,
                         histo->GetBinWidth(iBin) / 2., histo->GetBinWidth(iBin) / 2.,
                         histo->GetBinError(iBin), histo->GetBinError(iBin));
  }

  /* return error graph */
  return graph;
}


//__________________________________________________________________________________|___________

TGraphAsymmErrors * TQHistogramUtils::getGraph(TH1* nom, TObjArray* sys) {
	// convert a histogram into a TGraph
  if (!nom || !sys->GetEntries())
    return 0;

  /* only 1 dimensional histograms supported */
  if (getDimension(nom) != 1 || getDimension((TH1*)sys->First()) != 1)
    return 0;

  /* the number of bins of the histogram */
  int nBins = getNBins(nom);

  /* copy from the nominal to create template */
  TH1* upper = (TH1*)nom->Clone();
  TH1* lower = (TH1*)nom->Clone();
  upper->Reset();
  lower->Reset();

  /* loop over systematics and generate upper and lower */
  for (int iSys = 0; iSys < sys->GetEntries(); ++iSys) {
    TH1* thisSys = (TH1*) sys->At(iSys);
    for (int ibin = 0; ibin <= thisSys->GetNbinsX()+1; ++ibin) {
      double add_sys = thisSys->GetBinContent(ibin);
      double add_sys_err = thisSys->GetBinError(ibin);
      double cur_sys = upper->GetBinContent(ibin);
      double new_sys = TMath::Sqrt(cur_sys*cur_sys + add_sys*add_sys); // assumes that everything is un-correlated
      //double new_sys = cur_sys + add_sys; //linear sum. extra conservative approach
      double cur_sys_err = upper->GetBinError(ibin);
      double new_sys_err = TMath::Sqrt(cur_sys_err*cur_sys_err + add_sys_err*add_sys_err);
      if (add_sys >= 0) {
        upper->SetBinContent(ibin, new_sys);
        upper->SetBinError(ibin, new_sys_err);
      } else {
        lower->SetBinContent(ibin, new_sys);
        lower->SetBinError(ibin, new_sys_err);
      }
    }
  }

  /* the error graph to return */
  TGraphAsymmErrors * graph = new TGraphAsymmErrors(nBins);

  /* loop over bins and set graph point properties */
  for (int iBin = 1; iBin <= nBins; iBin++) {
    /* set the point */
    graph->SetPoint(iBin,
                    nom->GetBinCenter(iBin),
                    nom->GetBinContent(iBin));

    /* add the statistical error */
    double statErr = nom->GetBinError(iBin);
    double lowerErr = TMath::Sqrt(statErr*statErr + lower->GetBinContent(iBin)*lower->GetBinContent(iBin));
    double upperErr = TMath::Sqrt(statErr*statErr + upper->GetBinContent(iBin)*upper->GetBinContent(iBin));

    /* set the point error */
    graph->SetPointError(iBin,
                         nom->GetBinWidth(iBin) / 2., nom->GetBinWidth(iBin) / 2.,
                         lowerErr, upperErr);
  }

  /* return error graph */
  return graph;
}

//__________________________________________________________________________________|___________

std::vector<TString> TQHistogramUtils::histoBinsToCutStrings(TH1* hist, const TString& varexpr, TString cutname, const TString& basecutname){
  // creates a std::vector of TString that represent cut definitions according to the histogram bins
  // the cutname should contain a %d placeholder which will be replaced by the cut (bin) index
  // if the cutname does not contain such a placeholder, "_%d" will be appended
  // if no cutname is given, the histogram name will be used
  // if a non-empty basecutname is given, it will be included in the cut definition strings
  // underflow and overflow bins will automatically be included as first and last vector element
  if(cutname.IsNull()) cutname = hist->GetName();
  if(!cutname.Contains("%d")) cutname += "_%d";
  cutname.Append(":");
  if(!basecutname.IsNull()){
    cutname.Append(" ");
    cutname.Append(basecutname);
    cutname.Append(" <<");
  }
  cutname.Append(" ");
  std::vector<TString> cutstrings;
  cutstrings.push_back(TString::Format(cutname.Data(),0) + varexpr + TString::Format(" < %f ;",hist->GetBinLowEdge(1)));
  TString cutstringtemplate = cutname + "(%s < %f) && (%s > %f);";
  for(int i=2; i<=hist->GetNbinsX(); i++){
    cutstrings.push_back(TString::Format(cutstringtemplate.Data(),i-1,varexpr.Data(),hist->GetBinLowEdge(i),varexpr.Data(),hist->GetBinLowEdge(i-1)));
  }
  cutstrings.push_back(TString::Format(cutname.Data(),hist->GetNbinsX()) + varexpr + TString::Format(" > %f ;",hist->GetBinLowEdge(hist->GetNbinsX()+1)));
  return cutstrings;
}
 

//__________________________________________________________________________________|___________

double TQHistogramUtils::getPoisson(double b, double s) {
  // Returns the Poisson significance according to <b> expected background events
  // and <s> expected signal events. Returns 0 if any term becomes invalid
  //
  // Poisson significance = TMath::Sqrt(2. * ((s + b) * TMath::Log(1. + s / b) - s))

  if (b <= 0. || s < 0.) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  double sqrtInput = 2. * ((s + b) * TMath::Log(1. + s / b) - s);
  if (sqrtInput < 0.) {
    return std::numeric_limits<double>::quiet_NaN();
  }
 
  return TMath::Sqrt(sqrtInput);
}

//__________________________________________________________________________________|___________

double TQHistogramUtils::getPoissonError(double b, double s, double db, double ds) {
  // Returns the error on the Poisson significance according to <b> expected background events
  // and <s> expected signal events. Returns 0 if any term becomes invalid
  //
  // Poisson significance = TMath::Sqrt(2. * ((s + b) * TMath::Log(1. + s / b) - s))
  // The error is propagated from errors in b and s according to the most general formula
  // for independent error propagation.

  if (b <= 0. || s < 0.) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  // negative radicand check
  double sqrtInput = 2. * ((s + b) * TMath::Log(1. + s / b) - s);
  if (sqrtInput < 0.) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  // Poisson significance
  double P = TMath::Sqrt(sqrtInput);

  double logTerm = TMath::Log(1 + s / b);

  // dP/ds, dP/db
  double dPds = ( logTerm - 1 +     (b + s) / (b *     (1 + s / b)) ) / P;
  double dPdb = ( logTerm     - s * (b + s) / (b * b * (1 + s / b)) ) / P;

  return TMath::Sqrt( TMath::Power((dPds * ds), 2) + TMath::Power((dPdb * db), 2) );
}

//__________________________________________________________________________________|___________

double TQHistogramUtils::getPoissonWithError(double b, double berr, double s) {
  // Returns the Poisson significance according to <b> expected background events
  // and <s> expected signal events. Returns 0 if any term becomes invalid
  //
  // Poisson significance = TMath::Sqrt(2. * ((s + b) * TMath::Log(1. + s / b) - s))

  if (b <= 0. || s < 0.) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  double sqrtInput = 2. * ((s + b) * TMath::Log( (s + b) * (b + berr*berr) / ( b*b + (s + b) * berr * berr) ) - b*b / berr*berr * TMath::Log(1. + berr*berr*s / (b * (b + berr*berr) ) ) );
  if (sqrtInput < 0.) {
    return std::numeric_limits<double>::quiet_NaN();
  }
 
  return TMath::Sqrt(sqrtInput);
}


//__________________________________________________________________________________|___________

double TQHistogramUtils::getSoverSqrtB(double b, double s) {
  // Returns <s> over square-root of <b> or zero in case <s> is negative or <b> is
  // not larger than zero.

  if (b > 0. && s >= 0.) {
    return s / TMath::Sqrt(b);
  } else {
    return 0.;
  }
}

//__________________________________________________________________________________|___________

double TQHistogramUtils::getSoverSqrtSplusB(double b, double s) {
  // Returns <s> over square-root of <b> or zero in case <s> is negative or <b> is
  // not larger than zero.

  if (b > 0. && s >= 0.) {
    return s / TMath::Sqrt(s+b);
  } else {
    return 0.;
  }
}

//__________________________________________________________________________________|___________

double TQHistogramUtils::getSoverB(double b, double s) {
  // Returns <s> over <b> or zero in case <s> is negative or <b> is not larger than
  // zero.

  if (b > 0. && s >= 0.) {
    return s / b;
  } else {
    return 0.;
  }
}


//__________________________________________________________________________________|___________

double TQHistogramUtils::getSignificance(double b, double s,
                                         TString sgnfName, TString * sgnfTitle) {
	// calculate the significance given a certain string identifier
	// supported versions are:
	// poission, soversqrtb, soversqrtsplusb, soverb
  double sgnf = -1.;

  if (sgnfName.CompareTo("poisson", TString::kIgnoreCase) == 0) {
    if (sgnfTitle)
      *sgnfTitle = "Poisson Significance";
    sgnf = getPoisson(b, s);
  } else if (sgnfName.CompareTo("soversqrtb", TString::kIgnoreCase) == 0) {
    if (sgnfTitle)
      *sgnfTitle = "signal / #sqrt{background}";
    sgnf = getSoverSqrtB(b, s);
  } else if (sgnfName.CompareTo("soversqrtsplusb", TString::kIgnoreCase) == 0) {
    if (sgnfTitle)
      *sgnfTitle = "signal / #sqrt{signal+background}";
    sgnf = getSoverSqrtSplusB(b, s);
  } else if (sgnfName.CompareTo("soverb", TString::kIgnoreCase) == 0) {
    if (sgnfTitle)
      *sgnfTitle = "signal / background";
    sgnf = getSoverB(b, s);
  }

  return sgnf;

}

//__________________________________________________________________________________|___________

bool TQHistogramUtils::areEqual(TH1* first, TH1* second, bool includeUnderflowOverflow, bool compareErrors, double tolerance){
  // return true if two histograms are completely equal
  // false otherwise
  if(!first || !second) return false;
  if(!checkConsistency(first,second)) return false;
  if(first->GetNbinsX() != second->GetNbinsX()) return false;
  double integral = first->Integral() + second->Integral();
  for(int i= !includeUnderflowOverflow; i<first->GetNbinsX()-!includeUnderflowOverflow; i++){
    if(fabs((first->GetBinContent(i) - second->GetBinContent(i))/integral) > tolerance) return false;
    if(compareErrors && fabs((first->GetBinError(i) - second->GetBinError(i))/integral) > tolerance) return false;
  }
  return true;
}

//__________________________________________________________________________________|___________

void TQHistogramUtils::getSgnfAndErr(double b, double bErr, double s,
                                     double sErr, double & sgnf, double & sgnfErr, TString sgnfName, TString * sgnfTitle) {
	// calculate the significance given a certain string identifier
	// supported versions are:
	// poission, soversqrtb, soversqrtsplusb, soverb
  sgnf = getSignificance(b, s, sgnfName, sgnfTitle);

  /* variations of significance */
  double sgnf_bkg_up = getSignificance(b + bErr, s, sgnfName);
  double sgnf_bkg_down = getSignificance(b - bErr, s, sgnfName);
  double sgnf_sig_up = getSignificance(b, s + sErr, sgnfName);
  double sgnf_sig_down = getSignificance(b, s - sErr, sgnfName);

  /* the mean uncertainty on the significance caused
   * by background and signal variation separatly */
  double sgnf_bkg_err = (TMath::Abs(sgnf - sgnf_bkg_up)
                         + TMath::Abs(sgnf - sgnf_bkg_down)) / 2.;
  double sgnf_sig_err = (TMath::Abs(sgnf - sgnf_sig_up)
                         + TMath::Abs(sgnf - sgnf_sig_down)) / 2.;

  /* the mean total uncertainty of the significance */
  sgnfErr = TMath::Sqrt(TMath::Power(sgnf_bkg_err, 2)
                        + TMath::Power(sgnf_sig_err, 2));

}


//__________________________________________________________________________________|___________

TH1 * TQHistogramUtils::getCutEfficiencyHisto(TH1 * histo, TString options) {
  // Calculate the efficiency of a cut on the distribution given by the histogram
  // histo.

  /* stop if the input histogram is invalid */
  if (!histo)
    return 0;

  /* clone and reset the input histogram to get the binning */
  TH1 * h_eff = copyHistogram(histo, TString(histo->GetName()) + "_CutEff");
  h_eff->Reset();

  bool normToUnity = false;
  if (options.Contains("norm:yes", TString::kIgnoreCase))
    normToUnity = true;

  /* distinguish between one, two and three dimensional
   * histograms: this is a 3-dimensional histogram */
  if (histo->InheritsFrom(TH3::Class())) {

    /* not yet supported */
    return 0;

    /* this is a 2-dimensional histogram */
  } else if (histo->InheritsFrom(TH2::Class())) {

    TH2 * histo2 = (TH2*)histo;

    bool upperBoundOnX = true;
    bool upperBoundOnY = true;
    if (options.Contains("x:lower", TString::kIgnoreCase))
      upperBoundOnX = false;
    if (options.Contains("y:lower", TString::kIgnoreCase))
      upperBoundOnY = false;

    /* replace title of X axis */
    TString axisLabel = histo2->GetXaxis()->GetTitle();
    if (upperBoundOnX)
      axisLabel.Prepend("upper bound on ");
    else 
      axisLabel.Prepend("lower bound on ");
    h_eff->GetXaxis()->SetTitle(axisLabel.Data());

    /* replace title of Y axis */
    axisLabel = histo2->GetYaxis()->GetTitle();
    if (upperBoundOnY)
      axisLabel.Prepend("upper bound on ");
    else 
      axisLabel.Prepend("lower bound on ");
    h_eff->GetYaxis()->SetTitle(axisLabel.Data());


    int nBinsX = histo2->GetNbinsX();
    int nBinsY = histo2->GetNbinsY();

    for (int iBinX = 0; iBinX < nBinsX + 2; iBinX++) {
      for (int iBinY = 0; iBinY < nBinsY + 2; iBinY++) {

        int iBinXFrom = (upperBoundOnX ? 0 : iBinX);
        int iBinXTo = (upperBoundOnX ? iBinX : nBinsX + 1);
        int iBinYFrom = (upperBoundOnY ? 0 : iBinY);
        int iBinYTo = (upperBoundOnY ? iBinY : nBinsY + 1);

        double integral = histo2->Integral(iBinXFrom, iBinXTo, iBinYFrom, iBinYTo);

        h_eff->SetBinContent(h_eff->GetBin(iBinX, iBinY), integral);

      }
    }

    if (normToUnity)
      h_eff->Scale(1. / histo2->Integral(0, nBinsX + 1, 0, nBinsY + 1));

    /* this is a 1-dimensional histogram */
  } else {

    bool upperBoundOnX = true;
    if (options.Contains("x:lower", TString::kIgnoreCase))
      upperBoundOnX = false;

    /* replace title of X axis */
    TString axisLabel = histo->GetXaxis()->GetTitle();
    if (upperBoundOnX)
      axisLabel.Prepend("upper bound on ");
    else 
      axisLabel.Prepend("lower bound on ");
    h_eff->GetXaxis()->SetTitle(axisLabel.Data());

    double integral = 0.;
    double integralErr = 0.;
 
    /* loop over bins and integrate */
    int nBinsX = histo->GetNbinsX();
    for (int iBinX = (upperBoundOnX ? 0 : nBinsX + 1);
         upperBoundOnX ? (iBinX <= nBinsX + 1) : iBinX >= 0;
         iBinX += (upperBoundOnX ? 1 : -1)) {

      integral += histo->GetBinContent(iBinX);
      integralErr += TMath::Power(histo->GetBinError(iBinX), 2);

      h_eff->SetBinContent(iBinX, integral);
      h_eff->SetBinError(iBinX, TMath::Sqrt(integralErr));

    }

    if (normToUnity)
      h_eff->Scale(1. / integral);

  }

  return h_eff;

}


//__________________________________________________________________________________|___________

TH1 * TQHistogramUtils::getSignificanceHisto(TH1 * histo_bkg,
                                             TH1 * histo_sig, TString options) {

  /* stop if the input histograms are invalid/inconsitent */
  if (!checkConsistency(histo_bkg, histo_sig))
    return 0;

  /* 3 dimensional histograms not yet supported */
  if (histo_bkg->InheritsFrom(TH3::Class()))
    return 0;

  /* select the significance to use */
  TString sgnfName;
  if (options.Contains("sgnf:poisson", TString::kIgnoreCase))
    /* poisson significance */
    sgnfName = "poisson";
  else if (options.Contains("sgnf:soversqrtb", TString::kIgnoreCase))
    /* signal over square root background */
    sgnfName = "soversqrtb";
  else if (options.Contains("sgnf:soversqrtsplusb", TString::kIgnoreCase))
    /* signal over square root signal plus background */
    sgnfName = "soversqrtsplusb";
  else if (options.Contains("sgnf:soverb", TString::kIgnoreCase))
    /* signal over background */
    sgnfName = "soverb";
  else
    /* unknown significance */
    return 0;

  /* create cut efficiency histograms */
  TH1 * h_bkg_CutEff = getCutEfficiencyHisto(histo_bkg, options);
  TH1 * h_sig_CutEff = getCutEfficiencyHisto(histo_sig, options);

  /* clone and reset the histogram to get the binning */
  TH1 * h_sgnf = copyHistogram(h_sig_CutEff, TString(histo_sig->GetName()) + "_" + sgnfName);
  h_sgnf->Reset();

  TString * sgnfTitle = new TString();

  if (histo_bkg->InheritsFrom(TH2::Class())) {

    TH2 * h_bkg_CutEff2 = (TH2*)h_bkg_CutEff;
    TH2 * h_sig_CutEff2 = (TH2*)h_sig_CutEff;

    int nBinsX = h_sgnf->GetNbinsX();
    int nBinsY = h_sgnf->GetNbinsY();

    for (int iBinX = 0; iBinX < nBinsX + 2; iBinX++) {
      for (int iBinY = 0; iBinY < nBinsY + 2; iBinY++) {

        double significance = getPoisson(
                                         h_bkg_CutEff2->GetBinContent(iBinX, iBinY),
                                         h_sig_CutEff2->GetBinContent(iBinX, iBinY));

        h_sgnf->SetBinContent(iBinX, iBinY, significance);

      }
    }

  } else {

    int nBinsX = h_sgnf->GetNbinsX();
    for (int iBinX = 0; iBinX < nBinsX + 2; iBinX++) {

      double sgnf = 0.;
      double sgnfErr = 0.;
 
      getSgnfAndErr( h_bkg_CutEff->GetBinContent(iBinX),
                     h_bkg_CutEff->GetBinError(iBinX),
                     h_sig_CutEff->GetBinContent(iBinX),
                     h_sig_CutEff->GetBinError(iBinX),
                     sgnf, sgnfErr, sgnfName, sgnfTitle);
 
      h_sgnf->SetBinContent (iBinX, sgnf);
      h_sgnf->SetBinError (iBinX, sgnfErr);

    }

    h_sgnf->GetYaxis()->SetTitle(sgnfTitle->Data());

  }

  /* delete the cut efficiency histograms */
  delete h_bkg_CutEff;
  delete h_sig_CutEff;
  /* delete the title string */
  delete sgnfTitle;

  /* return the significance histogram */
  return h_sgnf;

}




//__________________________________________________________________________________|___________

TGraphAsymmErrors * TQHistogramUtils::getROCGraph(TH1 * h_bkg, TH1 * h_sig, bool lowerBound) {
  if (!checkConsistency(h_bkg, h_sig))
    return 0;

  /* only 1 dimensional histograms supported */
  if (getDimension(h_bkg) != 1)
    return 0;

  /* the number of bins of the histogram */
  int nBins = getNBins(h_bkg);

  /* the error graph to return */
  TGraphAsymmErrors * graph = new TGraphAsymmErrors(nBins);

  /* signal and background */
  double B = getIntegral(h_bkg);
  double S = getIntegral(h_sig);
  double b = 0.;
  double s = 0.;
  double berr2 = 0.;
  double serr2 = 0.;

  /* loop over bins and set graph point properties */
  for (int iBin = 0; iBin < nBins; iBin++) {

    b += h_bkg->GetBinContent(lowerBound ? nBins - iBin - 1 : iBin);
    s += h_sig->GetBinContent(lowerBound ? nBins - iBin - 1 : iBin);
    berr2 += TMath::Power(h_bkg->GetBinError(lowerBound ? nBins - iBin - 1 : iBin), 2.);
    serr2 += TMath::Power(h_sig->GetBinError(lowerBound ? nBins - iBin - 1 : iBin), 2.);

    /* set the point */
    graph->SetPoint(iBin, s / S, 1. - b / B);
  }

  graph->GetXaxis()->SetTitle("Signal efficiency");
  graph->GetYaxis()->SetTitle("Background rejection");

  /* return error graph */
  return graph;
}



//__________________________________________________________________________________|___________

TList * TQHistogramUtils::getProjectionHistograms(
                                                  TH2 * histo, bool projectOnX, bool normalize) {
  if (!histo)
    return 0;

  /* the list of profile histogram sto return */
  TList * histograms = new TList();

  /* get the number of projections (the number of bins perpen-
   * dicular to the axis to project on) and the the axis' title */
  int nBins;
  TString axisTitle;
  if (projectOnX) {
    nBins = histo->GetNbinsY();
    axisTitle = histo->GetYaxis()->GetTitle();
  } else {
    nBins = histo->GetNbinsX();
    axisTitle = histo->GetXaxis()->GetTitle();
  }

  /* get the variable and its unit from axis title */
  TString variable = axisTitle;
  TString unit = TQStringUtils::cutUnit(variable);

  /* loop over bins */
  for (int iBin = 0; iBin < nBins + 2; iBin++) {

    /* ===== create the projection ===== */

    /* the projection histogram */
    TH1 * projection;

    /* the "width of the projection" */
    double lower;
    double upper;

    if (projectOnX) {
      /* create the projection histogram on X */
      projection = histo->ProjectionX(TString::Format(
                                                      "%s_ProjX_%d", histo->GetName(), iBin).Data(), iBin, iBin, "e");
      /* get the bounderies along Y axis */
      lower = histo->GetYaxis()->GetBinLowEdge(iBin);
      upper = histo->GetYaxis()->GetBinUpEdge(iBin);
    } else {
      /* create the projection histogram on Y */
      projection = histo->ProjectionY(TString::Format(
                                                      "%s_ProjY_%d", histo->GetName(), iBin).Data(), iBin, iBin, "e");
      /* get the bounderies along Y axis */
      lower = histo->GetXaxis()->GetBinLowEdge(iBin);
      upper = histo->GetXaxis()->GetBinUpEdge(iBin);
    }

    /* set the histograms directory to the one of the input histogram */
    projection->SetDirectory(histo->GetDirectory());

    /* set the histograms title */
    if (iBin == 0)
      projection->SetTitle(TString::Format("%s < %g %s",
                                           variable.Data(), upper, unit.Data()).Data());
    else if (iBin > nBins)
      projection->SetTitle(TString::Format("%g %s #leq %s",
                                           lower, unit.Data(), variable.Data()).Data());
    else
      projection->SetTitle(TString::Format("%g %s #leq %s < %g %s",
                                           lower, unit.Data(), variable.Data(), upper, unit.Data()).Data());

    /* normalize the projection histogram if requested */
    if (normalize)
      TQHistogramUtils::normalize(projection);

    /* add the projection to the list */
    histograms->Add(projection);
  }

  /* return the list of profile histograms */
  return histograms;
}


//__________________________________________________________________________________|___________

TList * TQHistogramUtils::getProjectionHistogramsX(TH2 * histo, bool normalize) {

  return getProjectionHistograms(histo, true, normalize);
}


//__________________________________________________________________________________|___________

TList * TQHistogramUtils::getProjectionHistogramsY(TH2 * histo, bool normalize) {

  return getProjectionHistograms(histo, false, normalize);
}


//__________________________________________________________________________________|___________

TH1 * TQHistogramUtils::getReweightedHistogram(
                                               TH2 * histo_input, TH1 * histo_weights, bool projectOnX) {
  if (!histo_input || !histo_weights)
    return 0;

  /* ===== check consistency of weight histogram ===== */

  /* get the projection histogram perpendicular to final projection */
  TH1D * h_check = 0;
  if (projectOnX)
    h_check = histo_input->ProjectionY();
  else
    h_check = histo_input->ProjectionX();

  /* remove from current directory */
  h_check->SetDirectory(0);

  /* check its consistency with weights histogram */
  bool consistent = checkConsistency(h_check, histo_weights);

  /* delete the projection histogram since
   * it was only used for consistency check */
  delete h_check;

  /* stop if histograms are inconsistent */
  if (!consistent)
    return 0;

  /* ===== now do the reweighting ===== */

  /* the reweighted histogram to return */
  TH1 * h_result = 0;

  /* the number of bins along reweighting axis */
  int nBinsRew = projectOnX ? histo_input->GetNbinsY() : histo_input->GetNbinsX();

  /* loop over bins on reweighting axis */
  for (int iBinRew = 0; iBinRew < nBinsRew + 2; iBinRew++) {

    /* the temporary name of the projection */
    TString name = TString::Format("__h_proj_%s_%d", histo_input->GetName(), iBinRew);

    /* project the reweighting slice on projection axis */
    TH1 * h_proj;
    if (projectOnX)
      h_proj = histo_input->ProjectionX(name.Data(), iBinRew, iBinRew, "e");
    else
      h_proj = histo_input->ProjectionY(name.Data(), iBinRew, iBinRew, "e");

    /* remove from current directory */
    h_proj->SetDirectory(0);

    /* apply the weight */
    h_proj->Scale(histo_weights->GetBinContent(iBinRew));

    /* add up slices */
    if (h_result) {
      h_result->Add(h_proj);
      delete h_proj;
    } else {
      h_result = h_proj;
    }
  }

  /* set the name of the output histogram */
  if (h_result) {
    if (projectOnX)
      h_result->SetName(TString::Format("%s_ProjXRewY", histo_input->GetName()).Data());
    else
      h_result->SetName(TString::Format("%s_ProjYRewX", histo_input->GetName()).Data());
  }

  /* return the reweighted histogram */
  return h_result;
}


//__________________________________________________________________________________|___________

TH1 * TQHistogramUtils::getReweightedHistogramX(TH2 * histo_input, TH1 * histo_weights) {

  return getReweightedHistogram(histo_input, histo_weights, true);
}


//__________________________________________________________________________________|___________

TH1 * TQHistogramUtils::getReweightedHistogramY(TH2 * histo_input, TH1 * histo_weights) {

  return getReweightedHistogram(histo_input, histo_weights, false);
}


//__________________________________________________________________________________|___________

TQCounter * TQHistogramUtils::histogramToCounter(TH1 * histo) {
	// convert a histogram to a TQCounter
  if (!histo) {
    return NULL;
  }

  double err = 0.;
  double sum = getIntegralAndError(histo, err);

  return new TQCounter(histo->GetName(), sum, err);
}


//__________________________________________________________________________________|___________

TH1 * TQHistogramUtils::counterToHistogram(TQCounter * counter) {
	// convert a TQCounter to a histogram
  if (!counter) {
    return NULL;
  }

  TH1 * histo = new TH1D(counter->GetName(), counter->GetName(), 1, 0., 1.);
  histo->SetDirectory(NULL);
  histo->Sumw2();
  histo->SetBinContent(1, counter->getCounter());
  histo->SetBinError(1, counter->getError());
  histo->SetEntries(counter->getRawCounter());
  return histo;
}


//__________________________________________________________________________________|___________

TH1 * TQHistogramUtils::countersToHistogram(TList * counters) {
	// convert a list of TQCounters to a histogram
  int n = 0;
  TQIterator itr(counters);
  while (itr.hasNext()) {
    TObject * obj = itr.readNext();
    if (obj->InheritsFrom(TQCounter::Class())) {
      n++;
    }
  }

  TH1 * histo = NULL;
  if (n > 0) {
    TString prefix = counters->GetName();
    if (prefix.IsNull()) {
      prefix = "histogramFromCounters";
    }
    TString name = prefix;
    int i = 2;
    while (gDirectory && gDirectory->FindObject(name.Data())) {
      name = TString::Format("%s_i%d", prefix.Data(), i++);
    }
    histo = new TH1D(name, name, n, 0., (double)n);
    histo->SetDirectory(NULL);
    histo->Sumw2();

    i = 1;
    itr.reset();
    while (itr.hasNext()) {
      TObject * obj = itr.readNext();
      if (obj->InheritsFrom(TQCounter::Class())) {
        TQCounter * cnt = (TQCounter*)obj;
        histo->SetBinContent(i, cnt->getCounter());
        histo->SetBinError(i, cnt->getError());
        histo->GetXaxis()->SetBinLabel(i++, cnt->GetName());
      }
    }
  }

  return histo;
}


//__________________________________________________________________________________|___________

double TQHistogramUtils::getIntegral(TH1 * histo, bool useUnderOverflow) {
  // Return the integral of a histogram

  if (histo) {
    int xLow, xHigh, yLow, yHigh, zLow, zHigh;
    xLow = yLow = zLow = 0;
    xHigh = histo->GetNbinsX() + 1;
    yHigh = histo->GetNbinsY() + 1;
    zHigh = histo->GetNbinsZ() + 1;
    if (!useUnderOverflow) { 
        xLow++; yLow++; zLow++;
        xHigh--; yHigh--; zHigh--;
      }
    double integral = 0.;

    if (histo->InheritsFrom(TH3::Class())) {
      /* 3 dimensional integral */
      integral = ((TH3*)histo)->Integral(
                                         xLow, xHigh,
                                         zLow, yHigh, 
                                         zLow, zHigh);
    } else if (histo->InheritsFrom(TH2::Class())) {
      /* 2 dimensional integral */
      integral = ((TH2*)histo)->Integral(
                                         xLow, xHigh,
                                         yLow, yHigh);
    } else {
      /* 1 dimensional integral */
      integral = histo->Integral(xLow, xHigh);
    }

    return integral;

  } else {
    /* invalid input histogram given */
    return 0.;
  }
}


//__________________________________________________________________________________|___________

TH1* TQHistogramUtils::getEfficiencyHistogram(TH1* numerator, TH1* denominator) {
  // computes the ratio of the two given histograms. Errors are recalculated assuming
  // that events in the numerator histogram are also present in the denominator,
  // i.e., that the selection of events/entries in the numerator represents a true 
  // subset of those in the denominator. If the calculation of the resulting uncertainty
  // fails at some point NULL is returned.
  
  if (!numerator || !denominator) return NULL;
  if (!TQHistogramUtils::checkConsistency(numerator,denominator)) {
    WARN("Cannot create efficiency histogram, input histograms ('%s' and '%s') have inconsistent binning!",numerator->GetName(),denominator->GetName());
    return NULL;
  }
  //TH1* copy = TQHistogramUtils::copyHistogram(numerator);
  TH1* copy = dynamic_cast<TH1*>( numerator->Clone("hcopy") );
  if (!copy) {
    WARN("Failed to create clone of numerator histogram '%s' !",numerator->GetName());
    return NULL;
  }
  //int nBins = TQHistogramUtils::getNbinsGlobal(copy,true);
  if (!copy->Divide(numerator,denominator,1.,1.,"B")) {
    WARN("Failed to divide histograms");
    return NULL;
  }
  /*
  double delta,sDeltaSq,drdn,drdd,num;
  for (int i=0; i<nBins; ++i) {
    if (denominator->GetBinContent(i)!= 0.) {
      delta = denominator->GetBinContent(i)-numerator->GetBinContent(i);
      sDeltaSq = pow(denominator->GetBinError(i),2.) - pow(numerator->GetBinError(i),2.);
      num = numerator->GetBinContent(i);
      drdn = 1/(delta+num)*(1-1/(delta+num));
      drdd = -num/pow(num+delta,2.);
      if (sDeltaSq<0) {
        WARN("Difference between numerator and denominator has complex uncertainty (squared uncertainty is negative)!");
        delete copy;
        return NULL;
      }
      copy->SetBinContent(i, num/(delta+num)); 
      copy->SetBinError(i, sqrt( pow(drdn*denominator->GetBinError(i),2.) + pow(drdd,2.)*sDeltaSq ) );
    } else {
      copy->SetBinContent(i,0.); copy->SetBinError(i,0.);
    }
    
  }  
  */   
  return copy;
}


//__________________________________________________________________________________|___________

double TQHistogramUtils::getIntegralError(TH1 * histo) {
  // Return the error of the integral of a histogram including under- and over-
  // flow bins

  if (histo) {

    double error = 0.;

    /* calculate error of integral */
    getIntegralAndError(histo, error);

    /* return error of integral */
    return error;

  } else {
    /* invalid input histogram given */
    return 0.;
  }
}


//__________________________________________________________________________________|___________

double TQHistogramUtils::getIntegralAndError(TH1 * histo, double &error, bool useUnderOverflow) {
  // Return the integral and its error of a histogram including under- and over-
  // flow bins

  if (histo) {
    int xLow, xHigh, yLow, yHigh, zLow, zHigh;
    xLow = yLow = zLow = 0;
    xHigh = histo->GetNbinsX() + 1;
    yHigh = histo->GetNbinsY() + 1;
    zHigh = histo->GetNbinsZ() + 1;
    if (!useUnderOverflow) { 
        xLow++; yLow++; zLow++;
        xHigh--; yHigh--; zHigh--;
      }
    double integral = 0.;

    if (histo->InheritsFrom(TH3::Class())) {
      /* 3 dimensional integral */
      integral = ((TH3*)histo)->IntegralAndError(
                                         xLow, xHigh,
                                         zLow, yHigh, 
                                         zLow, zHigh, error);
    } else if (histo->InheritsFrom(TH2::Class())) {
      /* 2 dimensional integral */
      integral = ((TH2*)histo)->IntegralAndError(
                                         xLow, xHigh,
                                         yLow, yHigh, error);
    } else {
      /* 1 dimensional integral */
      integral = histo->IntegralAndError(xLow, xHigh, error);
    }

    return integral;

  } else {
    /* invalid input histogram given */
    return 0.;
  }
}


//__________________________________________________________________________________|___________

TH1 * TQHistogramUtils::normalize(TH1 * histo, double normalizeTo) {
  // Normalize this histogram to a given integral value

  if (histo) {

    /* get the histogram's integral */
    double integral = getIntegral(histo);

    /* normalize the histogram to 'normalizeTo' */
    if (integral != 0.)
      histo->Scale(normalizeTo / integral);

    return histo;

  } else {
    /* invalid input histogram given */
    return 0;
  }
}


//__________________________________________________________________________________|___________

TH1 * TQHistogramUtils::normalize(TH1 * histo, TH1 * normalizeToHisto) {
  // Normalize this histogram to the integral value of another histogram

  if (histo && normalizeToHisto) {
    /* normalize histogram to integral of second histogram */
    normalize(histo, getIntegral(normalizeToHisto));
    return histo;
  } else {
    /* invalid input histogram given */
    return 0;
  }
}


//__________________________________________________________________________________|___________

TH1 * TQHistogramUtils::power(TH1 * histo, double exp) {
	// exponentiate every entry in a histogram by a given power
  if (!histo) {
    return 0;
  }

  int n = TQHistogramUtils::getNBins(histo);
  for (int i = 0; i < n; i++) {
    double bin = histo->GetBinContent(i);
    double err = histo->GetBinError(i);
    histo->SetBinContent(i, TMath::Power(bin, exp));
    histo->SetBinError(i, exp * TMath::Power(bin, exp - 1.) * err);
  }

  return histo;
}


//__________________________________________________________________________________|___________

TH1 * TQHistogramUtils::getSlopeHistogram(TH1 * input, double slope) {
	// set the contents of a histogram to reflect a certain slope
  if (TQHistogramUtils::getDimension(input) != 1) {
    return NULL;
  }

  double min = input->GetXaxis()->GetBinLowEdge(1);
  double max = input->GetXaxis()->GetBinUpEdge(input->GetNbinsX());
  double avg = (max + min) / 2.;

  TH1 * output = TQHistogramUtils::copyHistogram(input);
  output->Reset();

  for (int i = 1; i <= input->GetNbinsX(); i++) {
    output->SetBinContent(i, 1. + slope * (avg - input->GetXaxis()->GetBinCenter(i)));
  }

  return output;
}


//__________________________________________________________________________________|___________

TH1 * TQHistogramUtils::applySlopeToHistogram(TH1 * input, double slope) {
	// apply a slope to a given histogram
  TH1 * h_slope = getSlopeHistogram(input, slope);
  if (!h_slope || !input) {
    return NULL;
  }

  input->Multiply(h_slope);
  delete h_slope;
  return input;
}


//__________________________________________________________________________________|___________

double TQHistogramUtils::getChi2(TH1 * histo1, TH1 * histo2) {
	// calculate the Chi2 difference between two histograms
  if (!histo1 || !histo2)
    return -1.;

  /* stop if histograms are inconsistent */
  if (!checkConsistency(histo1, histo2))
    return -1.;

  if (getDimension(histo1) == 1) {

    double chi2 = 0.;

    int nBins = histo1->GetNbinsX();
    for (int iBin = 0; iBin <= nBins + 1; iBin++) {

      /* get bin content and errors */
      double cnt1 = histo1->GetBinContent(iBin);
      double cnt2 = histo2->GetBinContent(iBin);
      double err1 = histo1->GetBinError(iBin);
      double err2 = histo2->GetBinError(iBin);

      /* calculate difference and error of difference */
      double diff = cnt1 - cnt2;
      double error = TMath::Sqrt(TMath::Power(err1, 2.) + TMath::Power(err2, 2.));

      /* sum up chi^2 */
      if (error != 0.)
        chi2 += TMath::Power(diff, 2.) / TMath::Power(error, 2.);
    }

    /* return chi2 / ndf */
    return chi2 / (double)(nBins + 2);

  } else {
    return -1.;
  }
}


//__________________________________________________________________________________|___________

TH1 * TQHistogramUtils::includeOverflowBins(TH1 * histo, bool underflow, bool overflow) {
	// include overflow and/or underflow bins in a histogram
  if(!underflow && !overflow) return histo;

  if (histo) {
    size_t nentries = histo->GetEntries();
    if (histo->InheritsFrom(TH3::Class()) || histo->InheritsFrom(TH2::Class())) {
      /* not applicable */
      return 0;
    } else {

      if (underflow) {
        /* move the content of the underflow bin to the first nominal bin */
        histo->SetBinContent(1,
                             histo->GetBinContent(0) +
                             histo->GetBinContent(1));
        histo->SetBinContent(0, 0.);
        /* propagate the error of the underflow bin to the first nominal bin error */
        histo->SetBinError(1, TMath::Sqrt(
                                          TMath::Power(histo->GetBinError(0), 2) +
                                          TMath::Power(histo->GetBinError(1), 2)));
        histo->SetBinError(0, 0.);
      }

      if (overflow) {
        /* get the number of bins to access
         * the last and the overflow bin */
        int nBins = histo->GetNbinsX();
        /* move the content of the overflow bin to the last nominal bin */
        histo->SetBinContent(nBins,
                             histo->GetBinContent(nBins) +
                             histo->GetBinContent(nBins + 1));
        histo->SetBinContent(nBins + 1, 0.);
        /* propagate the error of the overflow bin to the last nominal bin error */
        histo->SetBinError(nBins, TMath::Sqrt(
                                              TMath::Power(histo->GetBinError(nBins), 2) +
                                              TMath::Power(histo->GetBinError(nBins + 1), 2)));
        histo->SetBinError(nBins + 1, 0.);
      }

    }
    histo->SetEntries(nentries);

    return histo;

  } else {
    /* invalid input histogram given */
    return 0;
  }

}


//__________________________________________________________________________________|___________

void TQHistogramUtils::unifyMinMax(TCollection * histograms, double vetoFraction) {
	// unify the minima and maxima of a list of histograms
  if (!histograms)
    return;

  /* the maximum/minimum */
  double max = 0.;
  double min = 0.;

  /* loop over list of histograms to find global maximum and minimum */
  bool first = true;
  TQTH1Iterator itr(histograms);
  while(itr.hasNext()){
    /* cast to histogram */
    TH1 * histo = itr.readNext();
    if(!histo) continue;
    /* maximum/minimum of this histogram */
    double thisMax = histo->GetBinContent(histo->GetMaximumBin());
    double thisMin = histo->GetBinContent(histo->GetMinimumBin());
    /* calculate the new maximum/minimum */
    max = (first || thisMax > max) ? thisMax : max;
    min = (first || thisMin < min) ? thisMin : min;
    first = false;
  }

  /* allow some space between maximum and top of plot */
  max = min + (max - min)/(1-vetoFraction);

  /* loop over list of histograms to set the new global maximum and minimum */
  itr.reset();
  while(itr.hasNext()){
    TH1 * histo = itr.readNext();
    if(!histo) continue;
    histo->SetMaximum(max);
    histo->SetMinimum(min);
  }
}


//__________________________________________________________________________________|___________

void TQHistogramUtils::unifyMinMax(TH1 * h1, TH1 * h2, TH1 * h3, double vetoFraction) {
	// unify the minima and maxima of three histograms
  TList * histograms = new TList();
  if (h1) histograms->Add(h1);
  if (h2) histograms->Add(h2);
  if (h3) histograms->Add(h3);

  /* unify maximum and minimum */
  unifyMinMax(histograms, vetoFraction);

  /* delete the list of histograms */
  delete histograms;
}


//__________________________________________________________________________________|___________

void TQHistogramUtils::unifyMinMax(TH1 * h1, TH1 * h2, double vetoFraction) {
	// unify the minima and maxima of two histograms
  unifyMinMax(h1, h2, (TH1*)0, vetoFraction);
}


//__________________________________________________________________________________|___________

bool TQHistogramUtils::getMinMaxBin(
                                    TH1 * histo,
                                    int &iBinMin,
                                    int &iBinMax,
                                    bool includeError,
                                    bool includeUnderflowOverflow,
                                    double minMin,
                                    double maxMax) {
	// find the minimum and maximum bins of a histogram
  if (!histo)
    return false;

  int nBins = getNBins(histo);
  double min = std::numeric_limits<double>::infinity();
  double max = -std::numeric_limits<double>::infinity();
  /* loop over bins and find the minimum/maximum */
  for (int i = 0; i < nBins; ++i) {
    if (!includeUnderflowOverflow && (histo->IsBinUnderflow(i) || histo->IsBinOverflow(i)) ) continue;
    double lower = (histo->GetBinContent(i) - (includeError ? histo->GetBinError(i) : 0.));
    double upper = (histo->GetBinContent(i) + (includeError ? histo->GetBinError(i) : 0.));
    if ( (iBinMin == -1 || lower < min) && lower>minMin ){ //only consider value if not below lower cutoff 'minMin'
      min = lower;
      iBinMin = i;
    }
    if ( (iBinMax == -1 || upper > max) && upper<maxMax ){ //only consider value if not exeeding upper cutoff 'maxMax'
      max = upper;
      iBinMax = i;
    }
  }
  return true;
}


//__________________________________________________________________________________|___________

bool TQHistogramUtils::getMinMax(
                                 TH1 * histo,
                                 double &min,
                                 double &max,
                                 bool includeError,
                                 bool includeUnderflowOverflow,
                                 double minMin,
                                 double maxMax) {
	// find the minimum and maximum coordinates of a histogram
  int minBin = -1;
  int maxBin = -1;
  if (getMinMaxBin(histo, minBin, maxBin, includeError, includeUnderflowOverflow, minMin, maxMax) && (minBin!=-1 || maxBin!=-1) ) { //at least one limit should have been successfully obtained, otherwise something went clearly wrong (->return false)
    if (minBin!=-1) min = histo->GetBinContent(minBin) - (includeError ? histo->GetBinError(minBin) : 0.);
    if (maxBin!=-1) max = histo->GetBinContent(maxBin) + (includeError ? histo->GetBinError(maxBin) : 0.);
    return true;
  } else {
    return false;
  }
}


//__________________________________________________________________________________|___________

int TQHistogramUtils::getMinBin(TH1 * histo, bool includeError, bool includeUnderflowOverflow, double minMin) {
	// find the minimum bin of a histogram
  int minBin = -1;
  int maxBin = -1;
  getMinMaxBin(histo, minBin, maxBin, includeError, includeUnderflowOverflow, minMin, std::numeric_limits<double>::infinity() );
  return minBin;
}


//__________________________________________________________________________________|___________

int TQHistogramUtils::getMaxBin(TH1 * histo, bool includeError, bool includeUnderflowOverflow, double maxMax) {
	// find the maximum bin of a histogram
  int minBin = -1;
  int maxBin = -1;
  getMinMaxBin(histo, minBin, maxBin, includeError, includeUnderflowOverflow, -std::numeric_limits<double>::infinity(), maxMax );
  return maxBin;
}


//__________________________________________________________________________________|___________

double TQHistogramUtils::getMin(TH1 * histo, bool includeError, bool includeUnderflowOverflow, double minMin) {
	// find the minimum coordinate of a histogram
  double min = 0.;
  double max = 0.;
  getMinMax(histo, min, max, includeError, includeUnderflowOverflow, minMin, std::numeric_limits<double>::infinity() );
  return min;
}


//__________________________________________________________________________________|___________

double TQHistogramUtils::getMax(TH1 * histo, bool includeError, bool includeUnderflowOverflow, double maxMax) {
	// find the maximum coordinate of a histogram
  double min = 0.;
  double max = 0.;
  getMinMax(histo, min, max, includeError, includeUnderflowOverflow, -std::numeric_limits<double>::infinity(), maxMax);
  return max;
}


//__________________________________________________________________________________|___________

bool TQHistogramUtils::extractStyle(TH1 * histo, TQTaggable * tags, const TString& styleScheme) {
	// extract the style from a histogram and save it to a TQTaggable object
  if (!histo || !tags) {
    return false;
  }

  // the style tag prefix
  TString prefix;
  if (!styleScheme.IsNull()) {
    if (TQTaggable::isValidKey(styleScheme)) {
      prefix = TString("style.") + styleScheme + ".";
    } else {
      prefix = TString("style.");
    }
  }

  tags->setTagString (prefix + "title", histo->GetTitle());
  tags->setTagInteger (prefix + "histFillColor", histo->GetFillColor());
  tags->setTagInteger (prefix + "histFillStyle", histo->GetFillStyle());
  tags->setTagInteger (prefix + "histLineColor", histo->GetLineColor());
  tags->setTagInteger (prefix + "histLineWidth", histo->GetLineWidth());
  tags->setTagInteger (prefix + "histLineStyle", histo->GetLineStyle());
  tags->setTagInteger (prefix + "histMarkerColor", histo->GetMarkerColor());
  tags->setTagDouble (prefix + "histMarkerSize", histo->GetMarkerSize());
  tags->setTagInteger (prefix + "histMarkerStyle", histo->GetMarkerStyle());

  if(TQHistogramUtils::hasBinLabels(histo->GetXaxis())){
    for(int i=0; i<histo->GetXaxis()->GetNbins(); ++i){
      tags->setTagString(TString::Format("%sxLabels.%d",prefix.Data(),i),histo->GetXaxis()->GetBinLabel(i+1));
    }
  }
  if(TQHistogramUtils::hasBinLabels(histo->GetYaxis())){
    for(int i=0; i<histo->GetYaxis()->GetNbins(); ++i){
      tags->setTagString(TString::Format("%syLabels.%d",prefix.Data(),i),histo->GetXaxis()->GetBinLabel(i+1));
    }
  }
  
  return true;
}


//__________________________________________________________________________________|___________

int TQHistogramUtils::applyStyle(TH1 * histo, TQTaggable * tags, const TString& styleScheme, bool allowRecursion) {
	// apply the style given by a TQTaggable object to a histogram
  if (!histo || !tags) {
    return 0;
  }

  // the number of style tags applied
  int nStyleTags = 0;

  // the style tag prefix
  TString prefix;
  if (!styleScheme.IsNull()) {
    if (TQTaggable::isValidKey(styleScheme)) {
      prefix = TString(allowRecursion ? "~style." : "style.") + styleScheme + ".";
    } else {
      prefix = TString(allowRecursion ? "~style." : "style.");
    }
  }

  /* apply style: the histogram title */
  TString title;
  if (tags->getTagString(prefix + "title", title)) {
    histo->SetTitle(title.Data());
    nStyleTags++;
  }

  /* apply style: the common color */
  int commonColor;
  if (tags->getTagInteger(prefix + "color", commonColor)) {
    histo->SetFillColor (commonColor);
    histo->SetLineColor (commonColor);
    histo->SetMarkerColor (commonColor);
    nStyleTags += 3;
  }

  /* apply style: the histogram fill color */
  int histFillColor;
  if (tags->getTagInteger(prefix + "histFillColor", histFillColor)) {
    histo->SetFillColor(histFillColor);
    nStyleTags++;
  }

  /* apply style: the histogram fill style */
  int histFillStyle;
  if (tags->getTagInteger(prefix + "histFillStyle", histFillStyle)) {
    histo->SetFillStyle(histFillStyle);
    nStyleTags++;
  }

  /* apply style: the histogram line color */
  int histLineColor;
  if (tags->getTagInteger(prefix + "histLineColor", histLineColor)) {
    histo->SetLineColor(histLineColor);
    if (histo->InheritsFrom(TProfile::Class()))
      histo->SetLineColor(histFillColor);
    nStyleTags++;
  }

  /* apply style: the histogram line width */
  int histLineWidth;
  if (tags->getTagInteger(prefix + "histLineWidth", histLineWidth)) {
    histo->SetLineWidth(histLineWidth);
    nStyleTags++;
  }

  /* apply style: the histogram line style */
  int histLineStyle;
  if (tags->getTagInteger(prefix + "histLineStyle", histLineStyle)) {
    histo->SetLineStyle(histLineStyle);
    nStyleTags++;
  }

  /* apply style: the histogram marker color */
  int histMarkerColor;
  if (tags->getTagInteger(prefix + "histMarkerColor", histMarkerColor)) {
    histo->SetMarkerColor(histMarkerColor);
    nStyleTags++;
  }

  /* apply style: the histogram marker size */
  double histMarkerSize;
  if (tags->getTagDouble(prefix + "histMarkerSize", histMarkerSize)) {
    histo->SetMarkerSize(histMarkerSize);
    nStyleTags++;
  }

  /* apply style: the histogram marker style */
  int histMarkerStyle;
  if (tags->getTagInteger(prefix + "histMarkerStyle", histMarkerStyle)) {
    histo->SetMarkerStyle(histMarkerStyle);
    nStyleTags++;
  }

  std::vector<TString> xlabels = tags->getTagVString(prefix+"xLabels");
  for(size_t i=0; i<xlabels.size(); ++i){
    histo->GetXaxis()->SetBinLabel(i+1,xlabels[i]);
  }
  std::vector<TString> ylabels = tags->getTagVString(prefix+"yLabels");
  for(size_t i=0; i<ylabels.size(); ++i){
    histo->GetYaxis()->SetBinLabel(i+1,ylabels[i]);
  }
  
  /* return the number of style tags applied */
  return nStyleTags;
}


//__________________________________________________________________________________|___________

bool TQHistogramUtils::copyStyle(TH1 * h_dest, TH1 * h_src) {
  // Copies the style settings (fill color, fill style, line color, line style, line
  // width, marker size, marker color, marker style) from the source histogram
  // <h_src> to the destionation histogram <h_dest> and returns true in case of
  // success or false in case of failure.

  if (!h_dest || !h_src) {
    return false;
  }
  
  h_dest->SetFillColor(h_src->GetFillColor());
  h_dest->SetFillStyle(h_src->GetFillStyle());
  h_dest->SetLineColor(h_src->GetLineColor());
  h_dest->SetLineStyle(h_src->GetLineStyle());
  h_dest->SetLineWidth(h_src->GetLineWidth());
  h_dest->SetMarkerSize(h_src->GetMarkerSize());
  h_dest->SetMarkerColor(h_src->GetMarkerColor());
  h_dest->SetMarkerStyle(h_src->GetMarkerStyle());
  
  TQHistogramUtils::copyBinLabels(h_src,h_dest);
  TQHistogramUtils::copyAxisStyle(h_src,h_dest);
  
  return true;
}

//__________________________________________________________________________________|___________

bool TQHistogramUtils::copyStyle(TNamed * dest, TNamed * src) {
  // copies the style of histograms or graphs
  if(dest->InheritsFrom(TH1::Class())&&src->InheritsFrom(TH1::Class())) return TQHistogramUtils::copyStyle(static_cast<TH1*>(dest),static_cast<TH1*>(src));
  if(dest->InheritsFrom(TGraph::Class())&&src->InheritsFrom(TGraph::Class())) return TQHistogramUtils::copyStyle(static_cast<TGraph*>(dest),static_cast<TGraph*>(src));
  if(dest->InheritsFrom(TGraph::Class())&&src->InheritsFrom(TGraph2D::Class())) return TQHistogramUtils::copyStyle(static_cast<TGraph2D*>(dest),static_cast<TGraph*>(src));
  return false;
}

//__________________________________________________________________________________|___________

bool TQHistogramUtils::copyStyle(TGraph * g_dest, TGraph * g_src) {
  // Copies the style settings (fill color, fill style, line color, line style, line
  // width, marker size, marker color, marker style) from the source graph
  // <g_src> to the destionation graph <g_dest> and returns true in case of
  // success or false in case of failure.

  if (!g_dest || !g_src) {
    return false;
  }
  
  g_dest->SetFillColor(g_src->GetFillColor());
  g_dest->SetFillStyle(g_src->GetFillStyle());
  g_dest->SetLineColor(g_src->GetLineColor());
  g_dest->SetLineStyle(g_src->GetLineStyle());
  g_dest->SetLineWidth(g_src->GetLineWidth());
  g_dest->SetMarkerSize(g_src->GetMarkerSize());
  g_dest->SetMarkerColor(g_src->GetMarkerColor());
  g_dest->SetMarkerStyle(g_src->GetMarkerStyle());
  
  TQHistogramUtils::copyAxisStyle(g_src->GetXaxis(),g_dest->GetXaxis());
  TQHistogramUtils::copyAxisStyle(g_src->GetYaxis(),g_dest->GetYaxis());
  
  return true;
}

//__________________________________________________________________________________|___________

bool TQHistogramUtils::copyStyle(TGraph2D * g_dest, TGraph2D * g_src) {
  // Copies the style settings (fill color, fill style, line color, line style, line
  // width, marker size, marker color, marker style) from the source graph
  // <g_src> to the destionation graph <g_dest> and returns true in case of
  // success or false in case of failure.

  if (!g_dest || !g_src) {
    return false;
  }
  
  g_dest->SetFillColor(g_src->GetFillColor());
  g_dest->SetFillStyle(g_src->GetFillStyle());
  g_dest->SetLineColor(g_src->GetLineColor());
  g_dest->SetLineStyle(g_src->GetLineStyle());
  g_dest->SetLineWidth(g_src->GetLineWidth());
  g_dest->SetMarkerSize(g_src->GetMarkerSize());
  g_dest->SetMarkerColor(g_src->GetMarkerColor());
  g_dest->SetMarkerStyle(g_src->GetMarkerStyle());
  
  TQHistogramUtils::copyAxisStyle(g_src->GetXaxis(),g_dest->GetXaxis());
  TQHistogramUtils::copyAxisStyle(g_src->GetYaxis(),g_dest->GetYaxis());
  TQHistogramUtils::copyAxisStyle(g_src->GetZaxis(),g_dest->GetZaxis());

  return true;
}

//__________________________________________________________________________________|___________

bool TQHistogramUtils::isCloseToOneRel(double val, double rel) {
  // Returns true if the input value <val> is compatible with one with a relative
  // deviation smaller than <rel> and false otherwise. The return value is invariant
  // under the transformation <val> --> 1/<val>.

  if (val < 0.) {
    // negative value are always incompatible with one
    return false;
  } else if (val > 1.) {
    // for values larger than 1.
    return (val - 1.) < rel;
  } else if (val < 1.) {
    // for values smaller than one (but larger than zero)
    return (1. / val - 1.) < rel;
  } else {
    // val exactly equal to one
    return true;
  }
}


//__________________________________________________________________________________|___________

bool TQHistogramUtils::areEqualRel(double val1, double val2, double rel) {
  // Returns true if the two input values <val1> and <val2> are compatible with a
  // relative deviation smaller than <rel> and false otherwise. If either of the
  // input values is zero or one is positive while the other is negative false is
  // returned. The return value is invariant under exchange of <val1> and <val2>.

  if (val1 * val2 > 0.) {
    return isCloseToOneRel(val1 / val2, rel);
  } else {
    // at least one value is zero or one is positive while the other is negative
    return false;
  }
}


//__________________________________________________________________________________|___________

bool TQHistogramUtils::haveEqualShapeRel(TH1 * h1, TH1 * h2, double rel) {
	// check if two histograms have a relatively equal shape by calling areEqualRel on every bin
  if (!h1 || !h2)
    return false;

  /* histograms have to have same binning */
  if (!checkConsistency(h1, h2))
    return false;

  /* true if shapes are equal */
  bool equal = true;

  /* loop over individual bins */
  int nBins = getNBins(h1);
  for (int i = 0; equal && i < nBins; i++) {
    double bin1 = h1->GetBinContent(i);
    double bin2 = h2->GetBinContent(i);
    if (bin1 > 0. && bin2 > 0.)
      equal = areEqualRel(bin1, bin2, rel);
    else if (bin1 < 0. && bin2 < 0.)
      equal = areEqualRel(-bin1, -bin2, rel);
    else if (bin1 != 0. || bin2 != 0.)
      equal = false;
  }

  return equal;
}


//__________________________________________________________________________________|___________

int TQHistogramUtils::ensureMinimumBinContent(
                                              TH1 * histo, double min, bool forcePositive) {
	// ensure that a histogram has a certain minimum content in every bin
  int dim = getDimension(histo);
  if (dim < 1 || dim > 2) {
    return -1;
  }

  int nEntries = histo->GetEntries();

  /* the number of bins where additional bin content has been set */
  int set = 0;

  /* loop over individual bins */
  int nBinsX = histo->GetNbinsX();
  if (dim == 1) {
    for (int iX = 1; iX <= nBinsX; iX++) {
      double content = histo->GetBinContent(iX);
      if (!forcePositive) {
        content = TMath::Abs(content);
      }
      if (content < TMath::Abs(min)) {
        histo->SetBinContent(iX, min);
        set++;
      }
    }
  } else if (dim == 2) {
    int nBinsY = histo->GetNbinsY();
    for (int iX = 1; iX <= nBinsX; iX++) {
      for (int iY = 1; iY <= nBinsY; iY++) {
        double content = histo->GetBinContent(iX, iY);
        if (!forcePositive) {
          content = TMath::Abs(content);
        }
        if (content < TMath::Abs(min)) {
          histo->SetBinContent(iX, iY, min);
          set++;
        }
      }
    }
  }

  histo->SetEntries(nEntries);

  return set;
}


//__________________________________________________________________________________|___________

double TQHistogramUtils::pValuePoisson(unsigned nObs, // observed counts
                                       double nExp){ // Poisson parameter
  /*
    Consider Poi(k|nExp) and compute the p-value which corresponds to
    the observation of nObs counts.
 
    When nObs > nExp there is an excess of observed events and
 
    p-value = P(n>=nObs|nExp) = \sum_{n=nObs}^{\infty} Poi(n|nExp)
    = 1 - \sum_{n=0}^{nObs-1} Poi(n|nExp)
    = 1 - e^{-nExp} \sum_{n=0}^{nObs-1} nExp^n / n!
 
    Otherwise (nObs <= nExp) there is a deficit and
 
    p-value = P(n<=nObs|nExp) = \sum_{n=0}^{nObs} Poi(n|nExp)
    = e^{-nExp} \sum_{n=0}^{nObs} nExp^n / n!
  */
 
  if (nObs>nExp) // excess
    return 1-ROOT::Math::inc_gamma_c(nObs,nExp);
  else // deficit
    return ROOT::Math::inc_gamma_c(nObs+1,nExp);

}


//__________________________________________________________________________________|___________

double TQHistogramUtils::pValuePoissonError(unsigned nObs, // observed counts
                                            double E, // expected counts
                                            double V){ // variance of expectation

  /*
    Consider Poi(k|nExp) and compute the p-value which corresponds to
    the observation of nObs counts, in the case of uncertain nExp whose
    variance is provided.
 
    The prior for nExp is a Gamma density which matches the expectation
    and variance provided as input. The marginal model is provided by
    the Poisson-Gamma mixture, which is used to compute the p-value.
 
    Gamma density: the parameters are
    * a = shape param [dimensionless]
    * b = rate param [dimension: inverse of x]
 
    nExp ~ Ga(x|a,b) = [b^a/Gamma(a)] x^{a-1} exp(-bx)
 
    One has E[x] = a/b and V[x] = a/b^2 hence
    * b = E/V
    * a = E*b
 
    The integral of Poi(n|x) Ga(x|a,b) over x gives the (marginal)
    probability of observing n counts as
 
    b^a [Gamma(n+a) / Gamma(a)]
    P(n|a,b) = -----------------------------
    n! (1+b)^{n+a}
 
    When nObs > nExp there is an excess of observed events and
 
    p-value = P(n>=nObs) = \sum_{n=nObs}^{\infty} P(n)
    = 1 - \sum_{n=0}^{nObs-1} P(n)
 
    Otherwise (nObs <= nExp) there is a deficit and
 
    p-value = P(n<=nObs) = \sum_{n=0}^{nObs} P(n)
 
    To compute the sum, we use the following recurrent relation:
 
    P(n=0) = [b/(1+b)]^a
    P(n=1) = [b/(1+b)]^a a/(1+b) = P(n=0) a/(1+b)
    P(n=2) = [b/(1+b)]^a a/(1+b) (a+1)/[2(1+b)] = P(n=1) (a+1)/[2(1+b)]
    ... ...
    P(n=k) = P(n=k-1) (a+k-1) / [k(1+b)]
 
    and to avoid rounding errors, we work with logarithms.
  */

//  if (nObs<0) {
//    std::cerr << "ERROR in pValuePoissonError(): the number of observed events cannot be negative" << std::endl;
//    return 0;
//  }
  if (E<=0 || V<=0) {
    std::cerr << "ERROR in pValuePoissonError(): expectation and variance must be positive" << std::endl;
    return 0;
  }
  double B = E/V;
  double A = E*B;
 
  // relative syst = sqrt(V)/E = 1/sqrt(A)
  // relative stat = 1/sqrt(nObs)
  // if syst < 0.1*stat there is no need for syst:
  // save a bit of CPU time (comment if not needed)
  if (A>100*nObs) return TQHistogramUtils::pValuePoisson(nObs,E);

  // explicit treatment for systematics:
  unsigned stop=nObs;
  if (nObs>E) --stop;

  //double prob=pow(B/(1+B), A);
  // NB: must work in log-scale otherwise troubles!
  double logProb = A*log(B/(1+B));
  double sum=exp(logProb); // P(n=0)
  for (unsigned u=1; u<stop; ++u) {
    logProb += log((A+u-1)/(u*(1+B)));
    sum += exp(logProb);
  }
  if (nObs>E) // excess
    return 1-sum;
  else // deficit
    return sum;
}

//__________________________________________________________________________________|___________

double TQHistogramUtils::pValueToSignificance(double p, // p-value
                                              bool excess){ // false if deficit
  /*
    Convert a p-value into a right-tail normal significance, i.e. into
    the number of Gaussian standard deviations which correspond to it.
  */

  if (p<0 || p>1) {
    std::cerr << "ERROR: p-value must belong to [0,1] but input value is " << p << std::endl;
    return 0;
  }

  if (excess)
    return ROOT::Math::normal_quantile(1-p,1);
  else
    return ROOT::Math::normal_quantile(p,1);
}

//__________________________________________________________________________________|___________



TH1* TQHistogramUtils::pcmpObsVsExp(TH1* hObs, TH1* hExp, bool ignoreExpUnc){

  /*
    Find the significance of the excess/deficit of counts with respect
    to the expectation.
 
    The input histograms are:
    * hObs = observed counts (integers)
    * hExp = expected yields (real values)
    and the returned histogram contains the z-value or significance
    of the bin-wise deviations between hObs and hExp.
 
    The uncertainty on the expected yields must be provided in the
    form of bin "errors" (i.e. standard deviations). If they are
    not null and the boolean flag ignoreExpUnc is not true, such
    uncertainties are accounted for with a Bayesian treatment.
 
    The uncertainties on the expectation have the effect of reducing
    the significance of any deviation. A Gamma density is found
    which has the same expectation and standard deviation in each
    bin, and the marginal model is used to compute the p-value.
 
    The marginal model is a Poisson-Gamma mixture (also known as
    negative binomial).
  */

  if (hObs==0 || hExp==0) 
    return 0;

  TString name=hObs->GetName();

  name+="_cmp_";
  name+=hExp->GetName();

  int Nbins = hObs->GetNbinsX();

  TH1* hOut = 0;

  if (hObs->InheritsFrom(TH1F::Class()))
    hOut = new TH1F(name.Data(),hObs->GetTitle(), hObs->GetNbinsX(), hObs->GetXaxis()->GetXmin(), hObs->GetXaxis()->GetXmax() );
  else if (hObs->InheritsFrom(TH1D::Class()))
    hOut = new TH1D(name.Data(),hObs->GetTitle(), hObs->GetNbinsX(), hObs->GetXaxis()->GetXmin(), hObs->GetXaxis()->GetXmax() );

  hOut->GetXaxis()->SetTitle( hObs->GetXaxis()->GetTitle() );
  hOut->GetYaxis()->SetTitle("significance");
 
  hOut->SetFillColor(2);

  for (int i=1; i<Nbins; ++i) { // SKIP UNDER-, OVER-FLOWS
    int nObs = (int) hObs->GetBinContent(i);
    if (nObs<=0) 
      continue;
    float nExp = hExp->GetBinContent(i);
    float vrnc = hExp->GetBinError(i);
    vrnc *= vrnc; // variance
    float sig = 0;
    if (vrnc>0 && !ignoreExpUnc) {
      // account for systematic uncertainty
      float pValue = pValuePoissonError(nObs, nExp, vrnc);
      if (pValue<0.5) sig = pValueToSignificance(pValue, (nObs>nExp));
    } else {
      // assume perfect knowledge of Poisson parameter
      float pValue = pValuePoisson(nObs,nExp);
      if (pValue<0.5) sig = pValueToSignificance(pValue, (nObs>nExp));
    }
    hOut->SetBinContent(i, sig);
  }

  return hOut;
}

//__________________________________________________________________________________|___________

TH1* TQHistogramUtils::getUncertaintyHistogram(TH1* hist) {
  // fills a new histogram with uncertainties of the argument.
  if (!hist) return 0;
  TH1* histUncert = copyHistogram(hist, TString(hist->GetName())+"_uncertanties");
  for (int i=0; i<=getNbinsGlobal(histUncert); i++) {
    histUncert->SetBinContent(i,hist->GetBinError(i));
    histUncert->SetBinError(i,0.);
  }
  return histUncert;
}

//__________________________________________________________________________________|___________

TGraph* TQHistogramUtils::scatterPlot(const TString& name, double* vec1, double* vec2, int vLength, const TString& labelX, const TString& labelY) {
	// obtain a TGraph scatter plot from two lists of numbers
  std::vector<double> v1(vec1, vec1 + vLength);
  std::vector<double> v2(vec2, vec2 + vLength);
  return TQHistogramUtils::scatterPlot(name , v1 , v2 , labelX , labelY);
}

//__________________________________________________________________________________|___________


TGraph* TQHistogramUtils::scatterPlot(const TString& name, std::vector<double>& vec1, std::vector<double>& vec2, const TString& labelX, const TString& labelY) {
	// obtain a TGraph scatter plot from two lists of numbers
  if (vec1.size() != vec2.size()) {
    ERRORfunc("Vectors have different length!");
    return NULL;
  }
  TGraph* graph = new TGraph(vec1.size(),&vec1[0],&vec2[0]);
  if (!graph) {
    ERRORfunc("Failed to create graph");
    return NULL;
  }
  graph->SetNameTitle(name.Data(),name.Data());
  graph->GetXaxis()->SetTitle(labelX.Data());
  graph->GetYaxis()->SetTitle(labelY.Data());
  return graph;
}

//__________________________________________________________________________________|___________

TMultiGraph* TQHistogramUtils::makeMultiColorGraph(const std::vector<double>& vecX, const std::vector<double>& vecY, const std::vector<short>& vecColors) {
  // creates multiple TGraphs combined into a TMultiGraph. vecColors idicates the
  // color of each point (according to the usual ROOT colors) and is hence required
  // to have the same number of elements as the coordinates of the points (vecX, vecY).
  // For each color present in vecColors one TGraph is created and added to the 
  // TMultiGraph.
  if (vecX.size() != vecY.size() || vecX.size() != vecColors.size()) {
    WARN("Cannot create multiColorGraph: input sizes do not match!");
    return nullptr;
  }
  std::map< short,std::pair < std::vector<double>,std::vector<double> > > pointMap;
  for (size_t i=0; i<vecX.size(); ++i) {
    short color = vecColors[i];
    //ensure there is a pair of vectors for this color
    if (pointMap.count(color)<1) {
      std::pair< std::vector<double>,std::vector<double> > points;
      pointMap[color] = points;
    }
    pointMap[color].first.push_back(vecX[i]);
    pointMap[color].second.push_back(vecY[i]);
  }
  TMultiGraph* multi = new TMultiGraph();
  
  for (auto const& graphDef : pointMap) {
    TGraph* gr = new TGraph(graphDef.second.first.size(), &(graphDef.second.first[0]), &(graphDef.second.second[0]) );
    gr->SetMarkerColor(graphDef.first);
    multi->Add(gr);  //multi graph takes ownership
  }
  return multi;
}

//__________________________________________________________________________________|___________


void TQHistogramUtils::setSliceX(TH2* hist2d, TH1* hist, double value){
  // set an X-slice of a TH2*-type histogram to the content of a TH1*-type histogram
  // the dimension of the TH1* will become the Y-direction of the TH2*
  for(int i=1; i<hist->GetNbinsX()+2; i++){
    int bin = hist2d->FindBin(value,hist->GetBinCenter(i));
    if(bin <= 0) continue;
    hist2d->SetBinContent(bin,hist->GetBinContent(i));
    hist2d->SetBinError(bin,hist->GetBinError(i));
  }
}

//__________________________________________________________________________________|___________

void TQHistogramUtils::setSliceY(TH2* hist2d, TH1* hist, double value){
  // set an Y-slice of a TH2*-type histogram to the content of a TH1*-type histogram
  // the dimension of the TH1* will become the X-direction of the TH2*
  for(int i=1; i<hist->GetNbinsX()+2; i++){
    int bin = hist2d->FindBin(hist->GetBinCenter(i),value);
    if(bin <= 0) continue;
    hist2d->SetBinContent(bin,hist->GetBinContent(i));
    hist2d->SetBinError(bin,hist->GetBinError(i));
  }
}


//__________________________________________________________________________________|___________

double TQHistogramUtils::getMinimumBinValue(TH1* hist, double xmin, double xmax, bool includeErrors){
  // find the minimum bin value of the given histogram 
  // in the given x-range
  double min = std::numeric_limits<double>::infinity();
  if(!hist) return min;
  int minbin = hist->FindBin(xmin);
  int maxbin = hist->FindBin(xmax);
  for(int i=minbin; i<maxbin; i++){
    double val = hist->GetBinContent(i);
    if(includeErrors){
      double err = hist->GetBinError(i);
      if(TQUtils::isNum(err)) val -= err;
    }
    if(TQUtils::isNum(val) && val < min)
      min = val;
  }
  return min;
}

//__________________________________________________________________________________|___________

double TQHistogramUtils::getMinimumBinValue(TCollection* histograms, double xmin, double xmax, bool includeErrors){
  // find the minimum bin value of a list of histograms
  double min = std::numeric_limits<double>::infinity();
  TQTH1Iterator itr(histograms);
  while(itr.hasNext()){
    TH1 * histo = itr.readNext();
    if(!histo) continue;
    min = std::min(min,TQHistogramUtils::getMinimumBinValue(histo,xmin,xmax,includeErrors));
  }
  return min;
}

//__________________________________________________________________________________|___________

double TQHistogramUtils::getMaximumBinValue(TH1* hist, double xmin, double xmax, bool includeErrors){
  // find the maximum bin value of the given histogram 
  // in the given x-range
  double max = -std::numeric_limits<double>::infinity();
  if(!hist) return max;
  int minbin = hist->FindBin(xmin);
  int maxbin = hist->FindBin(xmax);
  for(int i=minbin; i<maxbin; i++){
    double val = hist->GetBinContent(i);
    if(includeErrors){
      double err = hist->GetBinError(i);
      if(TQUtils::isNum(err)) val += err;
    }
    if(TQUtils::isNum(val) && val > max)
      max = val;
  }
  return max;
}

//__________________________________________________________________________________|___________

double TQHistogramUtils::getMaximumBinValue(TCollection* histograms, double xmin, double xmax, bool includeErrors){
  // find the maximum bin value of a list of histograms
  double max = -std::numeric_limits<double>::infinity();
  TQTH1Iterator itr(histograms);
  while(itr.hasNext()){
    TH1 * histo = itr.readNext();
    if(!histo) continue;
    max = std::max(max,TQHistogramUtils::getMaximumBinValue(histo,xmin,xmax,includeErrors));
  }
  return max;
}

//__________________________________________________________________________________|___________

double TQHistogramUtils::getHistogramMaximum(size_t n, ...){
  // find the total maximum of a set of histograms
  // the first argument is the number of histograms to consider
  // an arbitrary number of subsequent histogram pointers can be passed
  va_list vl;
  va_start(vl,n);
  double max = -std::numeric_limits<double>::infinity();
  for(size_t i=0; i<n; i++){
    TH1* h = va_arg(vl,TH1*);
    if(h) max = std::max(max,h->GetMaximum());
  }
  va_end(vl);
  return max;
}

//__________________________________________________________________________________|___________

double TQHistogramUtils::getHistogramMinimum(size_t n, ...){
  // find the total minimum of a set of histograms
  // the first argument is the number of histograms to consider
  // an arbitrary number of subsequent histogram pointers can be passed
  va_list vl;
  va_start(vl,n);
  double min = std::numeric_limits<double>::infinity();
  for(size_t i=0; i<n; i++){
    TH1* h = va_arg(vl,TH1*);
    if(h) min = std::min(min,h->GetMinimum());
  }
  va_end(vl);
  return min;
}

//__________________________________________________________________________________|___________

double TQHistogramUtils::getHistogramXmax(size_t n, ...){
  // find x axis maximum of a set of histograms
  // the first argument is the number of histograms to consider
  // an arbitrary number of subsequent histogram pointers can be passed
  va_list vl;
  va_start(vl,n);
  double max = -std::numeric_limits<double>::infinity();
  for(size_t i=0; i<n; i++){
    TH1* h = va_arg(vl,TH1*);
    if(h) max = std::max(max,h->GetXaxis()->GetXmax());
  }
  va_end(vl);
  return max;
}

//__________________________________________________________________________________|___________

double TQHistogramUtils::getHistogramXmin(size_t n, ...){
  // find the x axis minimum of a set of histograms
  // the first argument is the number of histograms to consider
  // an arbitrary number of subsequent histogram pointers can be passed
  va_list vl;
  va_start(vl,n);
  double min = std::numeric_limits<double>::infinity();
  for(size_t i=0; i<n; i++){
    TH1* h = va_arg(vl,TH1*);
    if(h) min = std::min(min,h->GetXaxis()->GetXmin());
  }
  va_end(vl);
  return min;
}

//__________________________________________________________________________________|___________

double TQHistogramUtils::getHistogramYmax(size_t n, ...){
  // find the y axis maximum of a set of histograms
  // the first argument is the number of histograms to consider
  // an arbitrary number of subsequent histogram pointers can be passed
  va_list vl;
  va_start(vl,n);
  double max = -std::numeric_limits<double>::infinity();
  for(size_t i=0; i<n; i++){
    TH1* h = va_arg(vl,TH1*);
    if(h) max = std::max(max,h->GetYaxis()->GetXmax());
  }
  va_end(vl);
  return max;
}

//__________________________________________________________________________________|___________

double TQHistogramUtils::getHistogramYmin(size_t n, ...){
  // find the y axis minimum of a set of histograms
  // the first argument is the number of histograms to consider
  // an arbitrary number of subsequent histogram pointers can be passed
  va_list vl;
  va_start(vl,n);
  double min = std::numeric_limits<double>::infinity();
  for(size_t i=0; i<n; i++){
    TH1* h = va_arg(vl,TH1*);
    if(h) min = std::min(min,h->GetYaxis()->GetXmin());
  }
  va_end(vl);
  return min;
}

//__________________________________________________________________________________|___________

double TQHistogramUtils::getAxisYmin(TH1* hist){
  // retrieve lower edge value of y-axis from histogram
  if(!hist) return std::numeric_limits<double>::quiet_NaN();
  return hist->GetYaxis()->GetBinLowEdge(hist->GetYaxis()->GetFirst()); 
}

//__________________________________________________________________________________|___________

double TQHistogramUtils::getAxisYmax(TH1* hist){
  // retrieve upper edge value of y-axis from histogram
  if(!hist) return std::numeric_limits<double>::quiet_NaN();
  return hist->GetYaxis()->GetBinUpEdge(hist->GetYaxis()->GetLast()); 
}

//__________________________________________________________________________________|___________

double TQHistogramUtils::getAxisXmin(TH1* hist){
  // retrieve lower edge value of x-axis from histogram
  if(!hist) return std::numeric_limits<double>::quiet_NaN();
  return hist->GetXaxis()->GetBinLowEdge(hist->GetXaxis()->GetFirst()); 
}

//__________________________________________________________________________________|___________

double TQHistogramUtils::getAxisXmax(TH1* hist){
  // retrieve upper edge value of x-axis from histogram
  if(!hist) return std::numeric_limits<double>::quiet_NaN();
  return hist->GetXaxis()->GetBinUpEdge(hist->GetXaxis()->GetLast()); 
}

//__________________________________________________________________________________|___________

double TQHistogramUtils::getHistogramBinContentFromFile(const TString& fname, const TString& hname, const TString binlabel){
  // retrieve the contents of the bin labeled binlabel
  // from a histogram named hname which resides in a TFile fname
  // will return NaN and print an error message if no such file, histogram or bin exists
  TFile* f = TFile::Open(fname,"READONLY");
  if(!f){
    ERRORfunc("unable to open file: '%s'",fname.Data());
    return std::numeric_limits<double>::quiet_NaN();
  }
  if(!f->IsOpen()){
    delete f;
    ERRORfunc("unable to open file '%s'",fname.Data());
    return std::numeric_limits<double>::quiet_NaN();
  }
  TH1* h = dynamic_cast<TH1*>(f->Get(hname));
  if(!h){
    f->Close();
    delete f;
    ERRORfunc("unable to obtain histogram '%s' from file '%s'",hname.Data(),fname.Data());
    return std::numeric_limits<double>::quiet_NaN();
  }
  TAxis* a = h->GetXaxis();
  if(!a){
    f->Close();
    delete f;
    ERRORfunc("unable to obtain x-axis from histogram '%s' in file '%s'",hname.Data(),fname.Data());
    return std::numeric_limits<double>::quiet_NaN();
  }
  int bin = a->FindBin(binlabel);
  if(bin < 0){
    f->Close();
    delete f;
    ERRORfunc("histogram '%s' from file '%s' does not have any bin labeled '%s'",hname.Data(),fname.Data(),binlabel.Data());
    return std::numeric_limits<double>::quiet_NaN();
  }
  double val = h->GetBinContent(bin);
  f->Close();
  delete f;
  return val;
}

//__________________________________________________________________________________|___________

double TQHistogramUtils::getHistogramBinContentFromFile(const TString& fname, const TString& hname, int bin){
  // retrieve the contents of the given bin
  // from a histogram named hname which resides in a TFile fname
  // will return NaN and print an error message if no such file, histogram or bin exists
  TFile* f = TFile::Open(fname,"READONLY");
  if(!f){
    ERRORfunc("unable to open file: '%s'",fname.Data());
    return std::numeric_limits<double>::quiet_NaN();
  }
  if(!f->IsOpen()){
    delete f;
    ERRORfunc("unable to open file '%s'",fname.Data());
    return std::numeric_limits<double>::quiet_NaN();
  }
  TH1* h = dynamic_cast<TH1*>(f->Get(hname));
  if(!h){
    f->Close();
    delete f;
    ERRORfunc("unable to obtain histogram '%s' from file '%s'",hname.Data(),fname.Data());
    return std::numeric_limits<double>::quiet_NaN();
  }
  double val = h->GetBinContent(bin);
  f->Close();
  delete f;
  return val;
}


//__________________________________________________________________________________|___________

int TQHistogramUtils::edge(TH1* hist, double cutoff){
  // edges a one-dimensional histogram
  // adjusts the axis to hide marginal bins with a bin content below the cutoff
  if(!hist) return -1;
  TAxis* a = hist->GetXaxis();
  if(!a) return -1;
  size_t min=1;
  size_t max = hist->GetNbinsX();
  for(int i=min; i<hist->GetNbinsX(); i++){
    if(hist->GetBinContent(i) <= cutoff) min++;
    else break;
  }
  for(int i=max; i>0; i--){
    if(hist->GetBinContent(i) <= cutoff) max--;
    else break;
  }
  a->SetRange(min,max);
  return (hist->GetNbinsX() - max + min);
}

//__________________________________________________________________________________|___________

int TQHistogramUtils::edge(TH2* hist, double cutoff){
  // edges a two-dimensional histogram
  // adjusts the axis to hide marginal bins with a bin content below the cutoff
  if(!hist) return -1;
  TAxis* ax = hist->GetXaxis();
  TAxis* ay = hist->GetYaxis();
  if(!ax || !ay) return -1;
  size_t xmin=1;
  size_t xmax = hist->GetNbinsX();
  size_t ymin=1;
  size_t ymax = hist->GetNbinsY();
  for(int i=xmin; i<hist->GetNbinsX(); i++){
    bool emptyline = true;
    for(size_t j=1; j<ymax; j++){
      int bin = hist->GetBin(i,j);
      if(hist->GetBinContent(bin) > cutoff){
        emptyline = false;
        break;
      }
    }
    if(emptyline) xmin++;
    else break;
  }
  for(size_t i=xmax; i>0; i--){
    bool emptyline = true;
    for(size_t j=1; j<ymax; j++){
      int bin = hist->GetBin(i,j);
      if(hist->GetBinContent(bin) > cutoff){
        emptyline = false;
        break;
      }
    }
    if(emptyline) xmax--;
    else break;
  }
  for(int i=ymin; i<hist->GetNbinsY(); i++){
    bool emptyline = true;
    for(size_t j=xmin; j<xmax; j++){
      int bin = hist->GetBin(j,i);
      if(hist->GetBinContent(bin) > cutoff){
        emptyline = false;
        break;
      }
    }
    if(emptyline) ymin++;
    else break;
  }
  for(size_t i=ymax; i>0; i--){
    bool emptyline = true;
    for(size_t j=xmin; j<xmax; j++){
      int bin = hist->GetBin(j,i);
      if(hist->GetBinContent(bin) > cutoff){
        emptyline = false;
        break;
      }
    }
    if(emptyline) ymax--;
    else break;
  }
  ax->SetRange(xmin,xmax);
  ay->SetRange(ymin,ymax);
  return hist->GetNbinsX()*hist->GetNbinsY() - (xmax + xmin -1 )*(ymax+ymin-1);
}

//__________________________________________________________________________________|___________

bool TQHistogramUtils::rebin(TH1*& hist, const std::vector<int>& boundaries, bool doRemap) {
	// rebin and/or remap a histogram to given set of bin boundaries
  std::vector<double> bounds = TQHistogramUtils::getBinLowEdges(hist,boundaries);
  return TQHistogramUtils::rebin(hist,bounds,doRemap);
}

//__________________________________________________________________________________|___________

bool TQHistogramUtils::rebin(TH1*& hist, const std::vector<double>& boundaries, bool doRemap) {
	// rebin and/or remap a histogram to given set of bin boundaries
  TH1* oldhist = hist;
  hist = NULL;
  TString name(oldhist->GetName());
  TDirectory* dir = oldhist->GetDirectory();
  if(dir) dir->Remove(oldhist);
  hist = oldhist->Rebin(boundaries.size()-1, name, &boundaries[0]);
 
  // If remap is true, the variable binning is changed to flat.
  if (doRemap) {
    remap(hist->GetXaxis(),0,1);
  } 

  hist->SetDirectory(dir);
  delete oldhist;
  return true;
}

//__________________________________________________________________________________|___________

bool TQHistogramUtils::rebin(TH1* histo, int rebinX, int rebinY, int rebinZ){
	// rebin and/or remap a histogram to given number of bins in X, Y and Z direction
  int tmp = gErrorIgnoreLevel;
  gErrorIgnoreLevel = 5000;
  // rebin the histogram
  bool ok = true;
  int dim = TQHistogramUtils::getDimension(histo);
  if (dim == 1 && rebinX > 0 && rebinY == 0 && rebinZ == 0) {
    histo->Rebin(rebinX);
  } else if (dim == 2 && (rebinX > 0 || rebinY > 0) && rebinZ == 0) {
    if (rebinX > 0)
      ((TH2*)histo)->RebinX(rebinX);
    if (rebinY > 0)
      ((TH2*)histo)->RebinY(rebinY);
  } else if (dim == 3 && (rebinX > 0 || rebinY > 0 || rebinZ > 0)) {
    if (rebinX > 0)
      ((TH3*)histo)->Rebin3D(rebinX, 1, 1, "");
    if (rebinY > 0)
      ((TH3*)histo)->Rebin3D(1, rebinY, 1, "");
    if (rebinZ > 0)
      ((TH3*)histo)->Rebin3D(1, 1, rebinZ, "");
  } else {
    ok = false;
  }
  // restore error ignore level
  gErrorIgnoreLevel = tmp;
  return ok;
}

//__________________________________________________________________________________|___________

void TQHistogramUtils::remap(TAxis* ax,double min, double max) {
  // remap the axis of a histogram to a given range (without actually changing the binning)
  TString label = TQStringUtils::getWithoutUnit(ax->GetTitle());
  label.Prepend("Remapped ");
  ax->SetTitle(label.Data());
  ax->Set(ax->GetNbins(),min,max);
}

//__________________________________________________________________________________|___________

void TQHistogramUtils::print(THStack* s, TString/*options*/){
  // print the contents of a histogram stack to the console
  TQIterator itr(s->GetHists());
  while(itr.hasNext()){
    TH1* hist = dynamic_cast<TH1*>(itr.readNext());
    if(!hist) continue;
    std::cout<<TQStringUtils::fixedWidth(hist->GetName(),20) << TQStringUtils::fixedWidth(hist->GetTitle(),20) << TQHistogramUtils::getDetailsAsString(hist,4) << std::endl;
  }
}

//__________________________________________________________________________________|___________

double TQHistogramUtils::getMax(TCollection* c, bool includeUnderflowOverflow, double maxMax){
  // get the absolute maximum of a collection of histograms
  TQIterator itr(c);
  double max = -std::numeric_limits<double>::infinity();
  while(itr.hasNext()){
    TH1* hist = dynamic_cast<TH1*>(itr.readNext());
    if(!hist) continue;
    max = std::max(TQHistogramUtils::getMax(hist,false,includeUnderflowOverflow, maxMax) , max);
  }
  return max;
}

//__________________________________________________________________________________|___________

double TQHistogramUtils::getMin(TCollection* c, bool includeUnderflowOverflow, double minMin){
  // get the absolute minimum of a collection of histograms
  TQIterator itr(c);
  double min = std::numeric_limits<double>::infinity();
  while(itr.hasNext()){
    TH1* hist = dynamic_cast<TH1*>(itr.readNext());
    if(!hist) continue;
    min = std::min(TQHistogramUtils::getMin(hist,false,includeUnderflowOverflow, minMin) , min);
  }
  return min;
}

//__________________________________________________________________________________|___________

double TQHistogramUtils::getContourArea(TGraph* g){
  // calculate area of the graph contour (polygon) in the x/y plane
  // will only yield an accurate result if the graph is a regular polygon
  // without any intersecting lines
  if(!g) return std::numeric_limits<double>::quiet_NaN();
  double x,y,xOld,yOld;
  g->GetPoint(0,xOld,yOld);
  int i=1;
  double_t area = 0;
  while(g->GetPoint(i,x,y) > 0){
    area += y*xOld -x*yOld;
    xOld = x;
    yOld = y;
    i++;
  }
  g->GetPoint(0,x,y);
  area += y*xOld -x*yOld;
  return 0.5*area;
}


//__________________________________________________________________________________|___________

double TQHistogramUtils::getContourJump(TGraph* g){
  // calculate distance between first and last point of graph
  // in the x/y plane
  if(!g) return std::numeric_limits<double>::quiet_NaN();
  double x,y,xOld,yOld;
  g->GetPoint(0,xOld,yOld);
  g->GetPoint(g->GetN(),x,y);
  return sqrt(pow(x-xOld,2)+pow(y-yOld,2));
}

//__________________________________________________________________________________|___________

void TQHistogramUtils::interpolateGraph(TGraph* g, size_t increasePoints, const char* option){
  // interpolate a graph, increasing the number of points by the given factor
  if(!g) return;
  g->Sort();
  const size_t n = g->GetN();
  std::map<double,double> morePoints;
  for(size_t i=0; i<n-1; ++i){
    double minx,maxx,miny,maxy;
    g->GetPoint(  i,minx,miny);
    g->GetPoint(i+1,maxx,maxy);
    if(minx >= maxx) continue;
    
    double stepsize = fabs(maxx-minx) / (increasePoints);
    minx+=stepsize;
    maxx-=stepsize;
    
    for (double thisX = minx; thisX <= maxx; thisX += stepsize) {
      double thisY = g->Eval(thisX, NULL, option);
      if(TQUtils::isNum(thisY)){
        morePoints[thisX]=thisY;
      }
    }
  }
  size_t i=n;
  for(auto it:morePoints){
    g->SetPoint(i,it.first,it.second);
    i++;
  }
  g->Sort();
}

//__________________________________________________________________________________|___________

double TQHistogramUtils::getMinBinWidth(TAxis*a){
  // calculate width of smallest bin in axis coordinates
  if(!a) return std::numeric_limits<double>::quiet_NaN();
  double min = std::numeric_limits<double>::infinity(); 
  for(int i=1; i<=a->GetNbins(); i++){
    min = std::min(a->GetBinWidth(i),min);
  }
  return min;
}


//__________________________________________________________________________________|___________

double TQHistogramUtils::getMinBinArea(TH2* hist){
  // calculate area of smallest bin in axis coordinates
  if(!hist) return std::numeric_limits<double>::quiet_NaN();
  return TQHistogramUtils::getMinBinWidth(hist->GetXaxis())*TQHistogramUtils::getMinBinWidth(hist->GetYaxis());
}

//__________________________________________________________________________________|___________


int TQHistogramUtils::addPCA(TPrincipal* orig, TPrincipal* add){
  // add two objects of type TPrincipal
  // will add the contents of the second argument to the first one
  // the second one will stay unchanged
  if(!orig || !add) return -1;
 
  if(orig->GetSigmas()->GetNrows() != orig->GetSigmas()->GetNrows()) return -1;
 
  int nRow = 0;
  while(true){
    const double * row = add->GetRow(nRow);
    if(!row) break;
    orig->AddRow(row);
    nRow++;
  }
  return nRow;
}

//__________________________________________________________________________________|___________

bool TQHistogramUtils::addGraph(TGraph * orig, TGraph * add){
  // add two objects of type TGraph
  if(!orig || !add) return false;

  const size_t n = orig->GetN();
  const size_t nnew = add->GetN();

	// the standard expansion of TGraph is pretty inefficient, so we manually put it to the right size
	orig->Expand(n+nnew);

  for(size_t i=0; i<nnew; ++i){
    double x,y;
    add->GetPoint(i,x,y);
    orig->SetPoint(n+i,x,y);
  }
  return true;
}

//__________________________________________________________________________________|___________

bool TQHistogramUtils::addGraph(TGraph2D * orig, TGraph2D * add){
  // add two objects of type TGraph2D
  if(!orig || !add) return false;

  const size_t n = orig->GetN();
  const size_t nnew = add->GetN();

	// TGraph2D automatically expands in a smart way, so nothing to do here except adding the points

  const double* x = add->GetX();
  const double* y = add->GetY();
  const double* z = add->GetZ();

  for(size_t i=0; i<nnew; ++i){
    orig->SetPoint(n+i,x[i],y[i],z[i]);
  }
  return true;
}

//__________________________________________________________________________________|___________

bool TQHistogramUtils::addGraph(TNamed * orig, TNamed * add){
  // add two graphs
  if(!orig || !add) return false;
  if(orig->InheritsFrom(TGraph::Class()) && add->InheritsFrom(TGraph::Class())){
    return addGraph(static_cast<TGraph*>(orig),static_cast<TGraph*>(add));
  }
  if(orig->InheritsFrom(TGraph2D::Class()) && add->InheritsFrom(TGraph2D::Class())){
    return addGraph(static_cast<TGraph2D*>(orig),static_cast<TGraph2D*>(add));
  }
  ERRORfunc("unable to add graphs of types '%s' and '%s'",orig->ClassName(),add->ClassName());
  return false;
}
  
//__________________________________________________________________________________|___________

TString TQHistogramUtils::getDetailsAsString(TNamed * obj, int option) {
  // retrieve an info-string for some TPrincipal object
  if(obj->InheritsFrom(TH1::Class())) return TQHistogramUtils::getDetailsAsString((TH1*)obj,option);
  if(obj->InheritsFrom(TGraph::Class())) return TQHistogramUtils::getDetailsAsString((TGraph*)obj,option);
  if(obj->InheritsFrom(TPrincipal::Class())) return TQHistogramUtils::getDetailsAsString((TPrincipal*)obj,option);
  if(obj->InheritsFrom(TAxis::Class())) return TQHistogramUtils::getDetailsAsString((TAxis*)obj,option);
  if(obj->InheritsFrom(TGraph2D::Class())) return TQHistogramUtils::getDetailsAsString((TGraph2D*)obj,option);
  return TString::Format("%s: %s (unknown)",obj->GetName(),obj->GetTitle());
}

//__________________________________________________________________________________|___________

TString TQHistogramUtils::getDetailsAsString(TPrincipal * p, int/*option*/) {
  // retrieve an info-string for some TPrincipal object
  if(!p) return "NULL";
  TQTaggable tags(p->GetTitle());
  std::vector<TString> varnames;
  TString retval = TString::Format("%d entries",p->GetUserData()->GetNrows());
  if(tags.getTag("varname",varnames) > 0){
    retval += " in " + TQStringUtils::concat(varnames,",");
  } else {
    retval += TString::Format("(%s)",p->GetTitle());
  }
  return retval;
}

//__________________________________________________________________________________|___________

int TQHistogramUtils::dumpData(TPrincipal * p, int cutoff){
  // print the entire data contents of a TPrincipal object to the console
  if(!p) return -1;

  TQTaggable tags(p->GetTitle());
  size_t nVars = p->GetSigmas()->GetNrows();
  int width = 20;
  int numWidth = std::min((int)ceil(log10(cutoff))+1,10);
  std::vector<TString> varnames;
  if(tags.getTag("varname",varnames) == (int)nVars){
    for(size_t i=0; i<varnames.size(); i++){
      std::cout << TQStringUtils::makeBoldWhite(TQStringUtils::fixedWidth(varnames[i],width,"r"));
    }
  } else {
    for(size_t i=0; i<nVars; i++){
      std::cout << TQStringUtils::makeBoldWhite(TQStringUtils::fixedWidth(TString::Format("var%u",(unsigned int)i),width,"r"));
    }
  }
  std::cout << std::endl;
  int nRows = 0;
  while(nRows < cutoff){
    const double * row = p->GetRow(nRows);
    if(!row) break;
    std::cout << TQStringUtils::fixedWidth(TString::Format("%d",nRows),numWidth,"r");
    for(size_t i=0; i<nVars; i++){
      std::cout << TQStringUtils::fixedWidth(TString::Format("%f",row[i]),width,"r");
    }
    nRows++;
    std::cout << std::endl;
  }
  if(nRows >= cutoff){
    std::cout << " ... truncated after " << nRows << " entries ..." << std::endl;
  }
  return nRows;
}

//__________________________________________________________________________________|___________

TH1* TQHistogramUtils::getSoverSqrtBScan(TH1* sig, TH1* bkg, bool fromleft, double cutoff, bool verbose) {
  // retrieve an integrated s/sqrt(b) histogram, scanning from left or right
  return TQHistogramUtils::getFOMScan(TQHistogramUtils::kSoSqB,sig,bkg,fromleft,cutoff,verbose);
}

//__________________________________________________________________________________|___________

double TQHistogramUtils::getFOM(TQHistogramUtils::FOM fom, double b, double berr, double s, double/*serr*/){
  // calculate a figure of merit from the source numbers
  switch(fom){
  case kSoSqB: 
    return s/sqrt(b);
  case kSoSqBpdB: 
    return s/sqrt(b + berr*berr);
  case kPoisson: 
    return TQHistogramUtils::getPoisson(b,s);
  case kSoB: 
    return s/b;
  case kSoSqSpB: 
    return s/sqrt(s+b);
  default :
    return std::numeric_limits<double>::quiet_NaN();
  }
  return std::numeric_limits<double>::quiet_NaN();
}

//__________________________________________________________________________________|___________

TQHistogramUtils::FOM TQHistogramUtils::readFOM(TString fom){
  // extract a figure of merit from a string
  fom.ToLower();
  fom.ReplaceAll(" ","");
  if(TQStringUtils::matches(fom,"s/sqrt(s+b)")){
    return TQHistogramUtils::kSoSqSpB;
  }
  if(TQStringUtils::matches(fom,"s/b")){
    return TQHistogramUtils::kSoB;
  }
  if(TQStringUtils::matches(fom,"poisson")){
    return TQHistogramUtils::kPoisson;
  }
  if(TQStringUtils::matches(fom,"s/sqrt(b)")){
    return TQHistogramUtils::kSoSqB;
  }
  if(TQStringUtils::matches(fom,"s/sqrt(b+db2)")){
    return TQHistogramUtils::kSoSqBpdB;
  }
  return TQHistogramUtils::kUndefined;
}

//__________________________________________________________________________________|___________

TString TQHistogramUtils::getFOMTitle(TQHistogramUtils::FOM fom){
  // return a title string identifying a figure of merit
  switch(fom){
  case TQHistogramUtils::kSoSqB: return "s/#sqrt{b}";
  case TQHistogramUtils::kSoSqBpdB: return "s/#sqrt{b+db^{2}}";
  case TQHistogramUtils::kPoisson: return "#sqrt{2(s+b)ln(1+s/b)-s}";
  case TQHistogramUtils::kSoB: return "s/b";
  case TQHistogramUtils::kSoSqSpB: return "s/#sqrt{s+b}";
  default: return "<unknown>";
  }
  return "<unknown>";
}

//__________________________________________________________________________________|___________

TH1* TQHistogramUtils::getSoverSqrtB(TH1* sig, TH1* bkg) {
  // retrieve an s/sqrt(b) histogram (bin-by-bin mode)
  return TQHistogramUtils::getFOMHistogram(TQHistogramUtils::kSoSqB,sig,bkg);
}

//__________________________________________________________________________________|___________

TH1* TQHistogramUtils::getFOMHistogram(TQHistogramUtils::FOM fom, TH1* sig, TH1* bkg, double cutoff) {
  // retrieve a significance histogram (bin-by-bin mode)
  if(!checkConsistency(sig,bkg,true)) return NULL;

  /* significance plot */
  TH1* retval = (TH1*)copyHistogram(sig,"NODIR");
  retval->Reset();
  retval->GetYaxis()->SetTitle(TQHistogramUtils::getFOMTitle(fom));
  for(int i=0; i<sig->GetNbinsX(); i++){
    if(bkg->GetBinContent(i) > cutoff){
      retval->SetBinContent(i,TQHistogramUtils::getFOM(fom,bkg->GetBinContent(i),bkg->GetBinError(i),sig->GetBinContent(i),sig->GetBinError(i)));
    } else {
      retval->SetBinContent(i,0);
    }
    retval->SetBinError(i,0);
  }
  return retval;
}

//__________________________________________________________________________________|___________

TH1* TQHistogramUtils::getFOMScan(TQHistogramUtils::FOM fom, TH1* sig, TH1* bkg, bool fromleft, double cutoff, bool verbose) {
  // retrieve an integrated FOM histogram, scanning from left or right
  if(!checkConsistency(sig,bkg,true)) return NULL;
  if(!(getDimension(sig) == 1)){
    ERRORfunc("cannot perform scan on multi-dimensional histograms!");
  }

  /* significance plot */
  TH1* retval = (TH1*)copyHistogram(sig,"NODIR");
  if (fromleft)
    retval->SetLineColor(2);
  else
    retval->SetLineColor(4);
 
  int lbound = 0;
  int rbound = sig->GetNbinsX()+1;
 
  for (int ib = 0; ib <= sig->GetNbinsX(); ++ib) {
 
    double cutval = 0;
    if (fromleft){
      lbound = ib;
      cutval = sig->GetBinLowEdge(ib);
    } else {
      rbound = ib;
      cutval = sig->GetBinLowEdge(ib) + sig->GetBinWidth(ib);
    }

    double serr = 0;
    double s = sig->IntegralAndError(lbound,rbound,serr);
    double berr = 0;
    double b = bkg->IntegralAndError(lbound,rbound,berr);
    double z = TQHistogramUtils::getFOM(fom,b,berr,s,serr);
    if (b<cutoff)
      z = 0;
    if(verbose) VERBOSEfunc("@%s=%g: s=%g, b=%g, z=%g",bkg->GetXaxis()->GetTitle(),cutval,s,b,z);
 
    retval->SetBinContent(ib, z);
  }
  retval->GetYaxis()->SetTitle(TQHistogramUtils::getFOMTitle(fom));
  return retval;
}

//__________________________________________________________________________________|___________

double TQHistogramUtils::sumLineDistSquares(TH2* hist, double a, double b, bool useUnderflowOverflow){
  // calculate the sum of squared distances between some line and all histogram entries in a 2D histogram
  if(!hist) return 0;
  double sum = 0;
  for(size_t i=1 - useUnderflowOverflow; i<=(size_t)hist->GetNbinsX() + useUnderflowOverflow; i++){
    for(size_t j=1 - useUnderflowOverflow; j<=(size_t)hist->GetNbinsY() + useUnderflowOverflow; j++){
      sum += (hist->GetBinContent(i,j))*calculateBinLineDistSquare(hist,a,b,i,j);
    }
  }
  return sum;
}

//__________________________________________________________________________________|___________

double TQHistogramUtils::calculateBinLineDistSquare(TH2* hist, double a, double b, int i, int j){
  // calculate the squared distance between some line and a given bin in a 2D histogram
  // line is defined by slope b and y-axis offset a
  double xp = hist->GetXaxis()->GetBinCenter(i);
  double yp = hist->GetYaxis()->GetBinCenter(j);
  if(b == 0) return pow(yp,2);
  if(!TQUtils::isNum(b)) return pow(xp,2);
  // these are the legs of the right triangle
  // between the line and the two perpendiculars of the point
  // with respect to the x- and y-axes
  double yf = a + b*xp;
  double xf = (yp - a)/b;
  if((yf == yp) && (xf == xp)) return 0;
  double firstLeg2 = pow(xp-xf,2);
  double secondLeg2 = pow(yp-yf,2);
  double dist2 = firstLeg2 * secondLeg2 / (firstLeg2 + secondLeg2);
  // std::cout << xp << " " << xf << " " << yp << " " << yf << " " << firstLeg2 << " " << secondLeg2 << " " << dist2 << " " << weight << std::endl ;
  return dist2;
}

//__________________________________________________________________________________|___________

TH1* TQHistogramUtils::getLineDistSquares(TH2* hist, double a, double b, bool useUnderflowOverflow){
  // generate distribution of squared distances between some line and all histogram entries in a 2D histogram
  // line is defined by slope b and y-axis offset a
  if(!hist) return 0;
  double maxDist2 = pow(TQHistogramUtils::getAxisXmax(hist) - TQHistogramUtils::getAxisXmin(hist),2) + pow(TQHistogramUtils::getAxisYmax(hist) - TQHistogramUtils::getAxisYmin(hist),2);
  TH1F* distances = new TH1F(hist->GetName(),hist->GetTitle(),sqrt(hist->GetNbinsX()*hist->GetNbinsY()),0,maxDist2);
  distances->SetDirectory(NULL);
  for(size_t i=1 - useUnderflowOverflow; i<=(size_t)hist->GetNbinsX() + useUnderflowOverflow; i++){
    for(size_t j=1 - useUnderflowOverflow; j<=(size_t)hist->GetNbinsY() + useUnderflowOverflow; j++){
      distances->Fill(calculateBinLineDistSquare(hist,a,b,i,j),hist->GetBinContent(i,j));
    }
  }
  return distances;
}
 
//__________________________________________________________________________________|___________

TH1* TQHistogramUtils::rotationProfile(TH2* hist, double xUnit, double yUnit, int nStep, double xOrig, double yOrig){
  // produces the 'rotation profile' of a histogram in the following sense
  // given a line with angle alpha, what is the sum of squared distances of all histogram entries
  // with respect to the given line?
  // the returned histogram holds the dependency of the inverse of this sum as a function of the angle alpha
  // since the definition of the angle depends on the relative scaling of the two axis,
  // units must be given for both dimensions. 
  if(!hist) return NULL;
  if(xUnit <= 0) xUnit = TQHistogramUtils::getAxisXmax(hist) - xOrig;
  if(yUnit <= 0) yUnit = TQHistogramUtils::getAxisYmax(hist) - yOrig;
 
  double steprad = 0.5*TMath::Pi()/nStep;
  double stepdeg = 90./nStep;
  TH1F* profile = new TH1F(hist->GetName(), hist->GetTitle(), nStep+1, -stepdeg*0.5 , 0.5*stepdeg+90.);
  profile->SetDirectory(NULL);
  profile->GetXaxis()->SetTitle("#alpha [deg]");
  profile->GetYaxis()->SetTitle("\\left(\\sum_{N} n\\cdot d^{2}\\right)^{-1}");
  for(size_t i=0; i<(size_t)nStep+1; i++){
    double angle = i * steprad;
    double slope = (sin(angle)*yUnit) / (cos(angle)*xUnit);
    double yOffset = yOrig - slope*xOrig;
    double value = 1./TQHistogramUtils::sumLineDistSquares(hist,yOffset,slope);
    // std::cout << "considering slope " << slope << " with angle " << angle << " : " << value << std::endl;
    profile->SetBinContent(i+1,value);
  }
  return profile;
}

//__________________________________________________________________________________|___________

TH2* TQHistogramUtils::rotationYtranslationProfile(TH2* hist, double xUnit, double yUnit, int nStepAngle, int nStepOffset, double xOrig, double y0){
  // produces the 'rotation-translation profile' of a histogram in the following sense
  // given a line with angle alpha and a vertical offset y, what is the sum of squared distances of all histogram entries
  // with respect to the given line?
  // the returned histogram holds the dependency of the inverse of this sum as a function of the angle alpha and the offset y
  // since the definition of the angle depends on the relative scaling of the two axis,
  // units must be given for both dimensions. 
  if(!hist) return NULL;
  if(xUnit <= 0) xUnit = TQHistogramUtils::getAxisXmax(hist) - xOrig;
  if(yUnit <= 0) yUnit = TQHistogramUtils::getAxisYmax(hist);
  if(nStepOffset < 1) nStepOffset = hist->GetYaxis()->GetLast();
 
  double steprad = 0.5*TMath::Pi()/nStepAngle;
  double stepdeg = 90./nStepAngle;
  if(!TQUtils::isNum(y0)) y0 = TQHistogramUtils::getAxisYmin(hist);
  double yStep = (TQHistogramUtils::getAxisYmax(hist) - y0)/nStepOffset;
  TH2F* profile = new TH2F(hist->GetName(), hist->GetTitle(), nStepAngle+1, -stepdeg*0.5 , 0.5*stepdeg+90., nStepOffset+1, y0-0.5*yStep, y0 + (0.5+nStepOffset)*yStep);
  profile->SetDirectory(NULL);
  profile->GetXaxis()->SetTitle("#alpha [deg]");
  profile->GetYaxis()->SetTitle(hist->GetYaxis()->GetTitle());
  profile->GetZaxis()->SetTitle("\\left(\\sum_{N} n\\cdot d^{2}\\right)^{-1}");
  for(size_t i=0; i<(size_t)nStepAngle+1; i++){
    double angle = i * steprad;
    double slope = (sin(angle)*yUnit) / (cos(angle)*xUnit);
    for(size_t j=0; j<(size_t)nStepOffset+1; j++){
      double yOffset = y0 + j*yStep;
      double value = 1./TQHistogramUtils::sumLineDistSquares(hist,yOffset,slope);
      profile->SetBinContent(i+1,j+1,value);
    }
  }
  return profile;
}

//__________________________________________________________________________________|___________

TH2* TQHistogramUtils::rotationXtranslationProfile(TH2* hist, double xUnit, double yUnit, int nStepAngle, int nStepOffset, double yOrig, double x0){
  // produces the 'rotation-translation profile' of a histogram in the following sense
  // given a line with angle alpha and a horizontal offset x, what is the sum of squared distances of all histogram entries
  // with respect to the given line?
  // the returned histogram holds the dependency of the inverse of this sum as a function of the angle alpha and the offset x
  // since the definition of the angle depends on the relative scaling of the two axis,
  // units must be given for both dimensions. 
  if(!hist) return NULL;
  if(xUnit <= 0) xUnit = TQHistogramUtils::getAxisXmax(hist);
  if(yUnit <= 0) yUnit = TQHistogramUtils::getAxisYmax(hist) - yOrig;;
  if(nStepOffset < 1) nStepOffset = hist->GetXaxis()->GetLast();
 
  double steprad = 0.5*TMath::Pi()/nStepAngle;
  double stepdeg = 90./nStepAngle;
  double y0 = TQHistogramUtils::getAxisXmin(hist);
  if(!TQUtils::isNum(x0)) x0 = TQHistogramUtils::getAxisXmin(hist);
  double xStep = (TQHistogramUtils::getAxisXmax(hist) - x0)/nStepOffset;
  TH2F* profile = new TH2F(hist->GetName(), hist->GetTitle(), nStepAngle+1, -stepdeg*0.5 , 0.5*stepdeg+90., nStepOffset+1, x0-0.5*xStep, x0 + (0.5+nStepOffset)*xStep);
  profile->SetDirectory(NULL);
  profile->GetXaxis()->SetTitle("#alpha [deg]");
  profile->GetYaxis()->SetTitle(hist->GetXaxis()->GetTitle());
  profile->GetZaxis()->SetTitle("\\left(\\sum_{N} n\\cdot d^{2}\\right)^{-1}");
  for(size_t i=0; i<(size_t)nStepAngle+1; i++){
    double angle = i * steprad;
    double slope = (angle == 0.5*TMath::Pi() ? std::numeric_limits<double>::infinity() : (sin(angle)*yUnit) / (cos(angle)*xUnit));
    for(size_t j=0; j<(size_t)nStepOffset+1; j++){
      double xOffset = x0 + j*xStep;
      double yOffset = (angle == 0.5*TMath::Pi() ? std::numeric_limits<double>::quiet_NaN() : y0 - xOffset*slope);
      double value = 1./TQHistogramUtils::sumLineDistSquares(hist,yOffset,slope);
      // std::cout << "angle= " << i*stepdeg << ", xOffset = " << xOffset << ", yOffset=" << yOffset << ", slope=" << slope << " : " << value << std::endl;
      profile->SetBinContent(i+1,j+1,value);
    }
  }
  return profile;
}

//__________________________________________________________________________________|___________

TLine* TQHistogramUtils::makeBisectorLine(TH1* hist, double angle, double xUnit, double yUnit, double xOrig, double yOrig){
  // create a TLine with a given angle (in xUnit and yUnit coordinates, axis coordinates if not given)
  // measured from origin at xOrig and yOrig
  // will be cropped to the size fo the histogram
  if(!hist) return NULL;
  if(xUnit <= 0) xUnit = TQHistogramUtils::getAxisXmax(hist);
  if(yUnit <= 0) yUnit = TQHistogramUtils::getAxisYmax(hist);
  double angleRad = angle / 180. * TMath::Pi();
  double slope = (sin(angleRad)*yUnit) / (cos(angleRad)*xUnit);
  double yOffset = yOrig - slope*xOrig;

  double xAxMin = TQHistogramUtils::getAxisXmin(hist);
  double yAxMin = TQHistogramUtils::getAxisYmin(hist);
  double xAxMax = TQHistogramUtils::getAxisXmax(hist);
  double yAxMax = TQHistogramUtils::getAxisYmax(hist);

  if(!TQUtils::isNum(slope)){
    return new TLine(xOrig, yAxMin, xOrig, yAxMax);
  } else if(slope > 0){
    double xMin = std::max(xAxMin,(yAxMin-yOffset)/slope);
    double yMin = std::max(yAxMin,slope*xAxMin+yOffset);
    double xMax = std::min(xAxMax,(yAxMax-yOffset)/slope);
    double yMax = std::min(yAxMax,slope*xAxMax+yOffset);
 
    // std::cout << slope << " " << xOrig << " " << yOrig << " " << yOffset << std::endl;
    // std::cout << xAxMin << " " << yAxMin << " " << xAxMax << " " << yAxMax << std::endl;
    // std::cout << xMin << " " << yMin << " " << xMax << " " << yMax << std::endl;
 
    TLine* l = new TLine(xMin, yMin, xMax, yMax);
    return l;
  } else if (slope < 0){
    double xMin = std::max(xAxMin,(yAxMax-yOffset)/slope);
    double yMin = std::max(yAxMin,slope*xAxMax+yOffset);
    double xMax = std::min(xAxMax,(yAxMin-yOffset)/slope);
    double yMax = std::min(yAxMax,slope*xAxMin+yOffset);
 
    // std::cout << slope << " " << xOrig << " " << yOrig << " " << yOffset << std::endl;
    // std::cout << xAxMin << " " << yAxMin << " " << xAxMax << " " << yAxMax << std::endl;
    // std::cout << xMin << " " << yMin << " " << xMax << " " << yMax << std::endl;
 
    TLine* l = new TLine(xMin, yMax, xMax, yMin);
    return l;
  } else {
    return new TLine(xAxMin, yOrig, xAxMax, yOrig);
  }
  return NULL;

}

//__________________________________________________________________________________|___________

bool TQHistogramUtils::cropLine(TH1* hist, TLine* l){
  // crop a line so that it does not exceed the drawing area of the histogram
  // this function will swap the line orientation such that x1<x2
  if(!hist) return false;
  if(!l) return false;
  double xAxMin = TQHistogramUtils::getAxisXmin(hist);
  double yAxMin = TQHistogramUtils::getAxisYmin(hist);
  double xAxMax = TQHistogramUtils::getAxisXmax(hist);
  double yAxMax = TQHistogramUtils::getAxisYmax(hist);

  double x1 = std::min(l->GetX1(),l->GetX2());
  double x2 = std::max(l->GetX1(),l->GetX2());
  double y1 = (l->GetX2() > l->GetX1()) ? l->GetY1() : l->GetY2();
  double y2 = (l->GetX2() > l->GetX1()) ? l->GetY2() : l->GetY1();

  double slope = (y2-y1)/(x2-x1);

  // std::cout << x1 << "/" << y1 << " -- " << x2 << "/" << y2 << " : slope=" << slope << std::endl; 

  if(x2 < xAxMin) return false;
  if(x1 > xAxMax) return false;
  if(std::min(y1,y2) > yAxMax) return false;
  if(std::max(y1,y2) < yAxMin) return false;
 
  if(x2 > xAxMax){
    // std::cout << "x2 is too large" << std::endl;
    if(slope > 0){
      l->SetX2(std::min(xAxMax,x1 + (yAxMax - y1)/slope));
    } else if (slope < 0){
      l->SetX2(std::min(xAxMax,x1 - (y1 - yAxMin)/slope));
    } else {
      l->SetX2(xAxMax);
    }
  } else {
    l->SetX2(x2);
  }
  if(l->GetX2() < xAxMin) return false;
  if(x1 < xAxMin){
    // std::cout << "x1 is too small" << std::endl;
    if(slope > 0){
      l->SetX1(std::max(xAxMin,x2 - (y1 - yAxMin)/slope));
    } else if (slope < 0){
      l->SetX1(std::max(xAxMin,x2 + (yAxMin - y1)/slope));
    } else {
      l->SetX1(xAxMin);
    }
  } else {
    l->SetX1(x1);
  } 
  if(l->GetX1() > xAxMax) return false;
  if(y2 > yAxMax){
    // std::cout << "y2 is too large" << std::endl;
    l->SetY2(std::min(yAxMax,y1 + slope*(l->GetX2() - x1)));
    l->SetX2(std::min(l->GetX2(),x1 + (l->GetY2() - y1)/slope));
  } else if(y2 < yAxMin){
    // std::cout << "y2 is too small" << std::endl;
    l->SetY2(std::max(yAxMin,y1 + slope*(l->GetX2() - x1)));
    l->SetX2(std::min(l->GetX2(),x1 + (l->GetY2() - y1)/slope));
  } else {
    l->SetY2(y2);
  }
  if(l->GetY2() > yAxMax || l->GetY2() < yAxMin) return false;
  if(y1 > yAxMax){
    // std::cout << "y1 is too large" << std::endl;
    l->SetY1(std::min(yAxMax,y2 - slope*(x2 - l->GetX1())));
    l->SetX1(std::max(l->GetX1(),x2 + fabs(y2 - l->GetY1())/slope));
  } else if(y1 < yAxMin){
    // std::cout << "y1 is too small" << std::endl;
    l->SetY1(std::max(yAxMin,y2 - slope*(x2 - l->GetX1())));
    l->SetX1(std::max(l->GetX1(),x2 + fabs(y2 - l->GetY1())/slope));
  } else {
    l->SetY1(y1);
  }
  if(l->GetY1() > yAxMax || l->GetY1() < yAxMin) return false;
  return true;
  // std::cout << l->GetX1() << "/" << l->GetY1() << " -- " << l->GetX2() << "/" << l->GetY2() << std::endl; 
}
 
//__________________________________________________________________________________|___________  
 
double TQHistogramUtils::clearBinsAboveX(TH1* hist, double xMax){
  // clear all histogram bins beyond xMax
  if(!hist) return 0;
  int border = hist->GetXaxis()->FindBin(xMax);
  double retval = 0;
  for(size_t i=border; i<(size_t)hist->GetNbinsX()+1; i++){
    retval += hist->GetBinContent(i);
    hist->SetBinError(i,0);
    hist->SetBinContent(i,0);
  }
  return retval;
}

//__________________________________________________________________________________|___________

double TQHistogramUtils::clearBinsBelowX(TH1* hist, double xMin){
  // clear all histogram bins below xMin
  if(!hist) return 0;
  size_t border = hist->GetXaxis()->FindBin(xMin);
  double retval = 0;
  for(size_t i=0; i<border-1; i++){
    retval += hist->GetBinContent(i);
    hist->SetBinError(i,0);
    hist->SetBinContent(i,0);
  }
  return retval;
}

//__________________________________________________________________________________|___________

double TQHistogramUtils::clearBinsAboveX(TH2* hist, double xMax){
  // clear all histogram bins beyond xMax (for 2D-histograms)
  if(!hist) return 0;
  size_t border = hist->GetXaxis()->FindBin(xMax);
  double retval = 0;
  for(size_t i=border; i<(size_t)hist->GetNbinsX()+1; i++){
    for(size_t j=0; j<(size_t)hist->GetNbinsY()+1; j++){
      retval += hist->GetBinContent(i,j);
      hist->SetBinError(i,j,0);
      hist->SetBinContent(i,j,0);
    }
  }
  return retval;
}

//__________________________________________________________________________________|___________

double TQHistogramUtils::clearBinsBelowX(TH2* hist, double xMin){
  // clear all histogram bins below xMin (for 2D-histograms)
  if(!hist) return 0;
  size_t border = hist->GetXaxis()->FindBin(xMin);
  double retval = 0;
  for(size_t i=0; i<border; i++){
    for(size_t j=0; j<(size_t)hist->GetNbinsY()+1; j++){
      retval += hist->GetBinContent(i,j);
      hist->SetBinError(i,j,0);
      hist->SetBinContent(i,j,0);
    }
  }
  return retval;
}

//__________________________________________________________________________________|___________

double TQHistogramUtils::clearBinsAboveY(TH2* hist, double yMax){
  // clear all histogram bins beyond yMax (for 2D-histograms)
  if(!hist) return 0;
  int border = hist->GetYaxis()->FindBin(yMax);
  double retval = 0;
  for(size_t i=0; i<(size_t)hist->GetNbinsX()+1; i++){
    for(size_t j=border; j<(size_t)hist->GetNbinsY()+1; j++){
      retval += hist->GetBinContent(i,j);
      hist->SetBinError(i,j,0);
      hist->SetBinContent(i,j,0);
    }
  }
  return retval;
}

//__________________________________________________________________________________|___________

double TQHistogramUtils::clearBinsBelowY(TH2* hist, double yMin){
  // clear all histogram bins below yMin (for 2D-histograms)
  if(!hist) return 0;
  size_t border = hist->GetYaxis()->FindBin(yMin);
  double retval = 0;
  for(size_t i=0; i<(size_t)hist->GetNbinsX()+1; i++){
    for(size_t j=0; j<border; j++){
      retval += hist->GetBinContent(i,j);
      hist->SetBinError(i,j,0);
      hist->SetBinContent(i,j,0);
    }
  }
  return retval;
}

//__________________________________________________________________________________|___________

void TQHistogramUtils::printHistogramASCII(TH1* hist, const TString& tags){
  // print a histogram to the console (ascii-art)
  TQHistogramUtils::printHistogramASCII(std::cout, hist, tags);
}

//__________________________________________________________________________________|___________

void TQHistogramUtils::printHistogramASCII(TH1* hist, TQTaggable& tags){
  // print a histogram to the console (ascii-art)
  TQHistogramUtils::printHistogramASCII(std::cout, hist, tags);
}

//__________________________________________________________________________________|___________

void TQHistogramUtils::printHistogramASCII(std::ostream& os, TH1* hist, const TString& tags){
  // print a histogram to a stream (ascii-art)
  TQTaggable taggable(tags);
  TQHistogramUtils::printHistogramASCII(os, hist,taggable);
}

//__________________________________________________________________________________|___________

void TQHistogramUtils::printHistogramASCII(std::ostream& os, TH1* hist, TQTaggable& tags){
  // print a histogram to a stream (ascii-art)
  bool showTitle = tags.getTagBoolDefault("showTitle",true);
  bool showMeanRMS = tags.getTagBoolDefault("showMeanRMS",showTitle);
  bool showEntries = tags.getTagBoolDefault("showEntries",showTitle);
  bool includeUnderflow = tags.getTagBoolDefault("includeUnderflow",false);
  bool includeOverflow = tags.getTagBoolDefault("includeOverflow",false);
  TString token = tags.getTagStringDefault("token","█");
  size_t yScale = tags.getTagIntegerDefault("yScale",10);
  int rebin = tags.getTagIntegerDefault("rebin",1);
  int nSigDigitsY = tags.getTagIntegerDefault("showDigitsY",2);
  int nSigDigitsX = tags.getTagIntegerDefault("showDigitsX",2);
  if(showTitle){
    os << hist->ClassName() << " " << hist->GetName() << ": " << hist->GetTitle();
    if(showMeanRMS){
      os << " - " << "Mean = " << hist->GetMean() << ", RMS = " << hist->GetRMS(); 
    }
    if(showEntries){
      os << " - " << hist->GetEntries() << " Entries";
    }
    os << std::endl;
  }
  double max = 0;
  double val = 0;
  int count = 0;
  int lastbin = 0;
  std::vector<double> nBlocks;
  std::vector<double> binCenters;
  if(includeUnderflow){
    nBlocks.push_back(hist->GetBinContent(0));
    binCenters.push_back(.5*(hist->GetBinLowEdge(0) + hist->GetBinLowEdge(1)));
  }
  for(int i=0; i<hist->GetNbinsX(); i++){
    val += hist->GetBinContent(i);
    count++;
    if(count >= rebin){
      binCenters.push_back(.5*(hist->GetBinLowEdge(lastbin) + hist->GetBinLowEdge(i+1)));
      max = std::max(val,max);
      nBlocks.push_back(val);
      val = 0;
      count = 0;
      lastbin = i;
    }
  }
  if(tags.getTagBoolDefault("vertical",false)){
    bool showBinContents = tags.getTagBoolDefault("printBinContents",true);
    bool showHlines = tags.getTagBoolDefault("printHorizontalLines",true);
    int rightMargin = showBinContents ? std::max(10,nSigDigitsX+2) : 0;
    int leftMargin = nSigDigitsY+2+std::max(ceil(log10(TQHistogramUtils::getAxisXmax(hist))),0.);
    int histWidth = tags.getTagIntegerDefault("width",TQLibrary::getConsoleWidth() - rightMargin - 2 - leftMargin); 
    TString fmtx = TString::Format("%%.%df",nSigDigitsX);
    TString fmty = TString::Format("%%.%df",nSigDigitsY);
    double scale = ((double)(histWidth))/max;
    if(showHlines) os << TQStringUtils::repeat(" ",leftMargin) << "+" << TQStringUtils::repeat("-",histWidth) << std::endl;
    for(size_t i=0; i<nBlocks.size(); i++){
      size_t n = scale*nBlocks[i];
      os << TQStringUtils::fixedWidth(TString::Format(fmtx,binCenters[i]),leftMargin,"r");
      os << "|";
      os << TQStringUtils::repeat(token,n);
      if(showBinContents){
        os << " " << TString::Format(fmty,nBlocks[i]);
      } 
      os << std::endl;
    }
    if(showHlines) os << TQStringUtils::repeat(" ",leftMargin) << "+" << TQStringUtils::repeat("-",histWidth) << std::endl;
  } else {
    int width = floor(log10(max));
    if(includeOverflow) nBlocks.push_back(hist->GetBinContent(hist->GetNbinsX()));
    double scale = ((double)yScale)/max;
    for(size_t i=yScale; i>=0 && i<=yScale; i--){
      os << " |";
      for(size_t j=0; j<nBlocks.size(); j++){
        if(nBlocks[j]*scale > i){
          os << token;
        } else {
          os << " ";
        }
      }
      os << "| ";
      if(i==0){
        os << TQStringUtils::repeat(" ",width) << 0<< std::endl;
      } else {
        os << TQStringUtils::repeat(" ",width-floor(log10(i/scale))) << TQUtils::round(i/scale,nSigDigitsY) << std::endl;
      }
    }
    os << " +" << TQStringUtils::repeat("-",nBlocks.size()) << "+" << std::endl;
    TString fmt = TString::Format("%%.%df",nSigDigitsX);
    os << " " << TQStringUtils::fixedWidth(TString::Format(fmt,TQHistogramUtils::getAxisXmin(hist)),ceil(0.5*nBlocks.size())+1,"l");
    os << TQStringUtils::fixedWidth(TString::Format(fmt,TQHistogramUtils::getAxisXmax(hist)),ceil(0.5*nBlocks.size())+1,"r");
    os << std::endl;
  }
}

//__________________________________________________________________________________|___________

TStyle* TQHistogramUtils::ATLASstyle() 
{
  // copied from official ATLAS style scripts
  // returns a TStyle* object according to the ATLAS style guidelines
  // can then be enabled via gROOT->SetStyle("ATLAS");
  // followed by either calling TH1::UseCurrentStyle() on each histogram or gROOT->ForceStyle();
  
  TStyle *atlasStyle = new TStyle("ATLAS","Atlas style");

  // use plain black on white colors
  Int_t icol=0; // WHITE
  atlasStyle->SetFrameBorderMode(icol);
  atlasStyle->SetFrameFillColor(icol);
  atlasStyle->SetCanvasBorderMode(icol);
  atlasStyle->SetCanvasColor(icol);
  atlasStyle->SetPadBorderMode(icol);
  atlasStyle->SetPadColor(icol);
  atlasStyle->SetStatColor(icol);
  //atlasStyle->SetFillColor(icol); // don't use: white fill color for *all* objects

  // set the paper & margin sizes
  atlasStyle->SetPaperSize(20,26);

  // set margin sizes
  atlasStyle->SetPadTopMargin(0.05);
  atlasStyle->SetPadRightMargin(0.05);
  atlasStyle->SetPadBottomMargin(0.16);
  atlasStyle->SetPadLeftMargin(0.16);

  // set title offsets (for axis label)
  atlasStyle->SetTitleXOffset(1.4);
  atlasStyle->SetTitleYOffset(1.4);

  // use large fonts
  //Int_t font=72; // Helvetica italics
  Int_t font=42; // Helvetica
  Double_t tsize=0.05;
  atlasStyle->SetTextFont(font);

  atlasStyle->SetTextSize(tsize);
  atlasStyle->SetLabelFont(font,"x");
  atlasStyle->SetTitleFont(font,"x");
  atlasStyle->SetLabelFont(font,"y");
  atlasStyle->SetTitleFont(font,"y");
  atlasStyle->SetLabelFont(font,"z");
  atlasStyle->SetTitleFont(font,"z");
  
  atlasStyle->SetLabelSize(tsize,"x");
  atlasStyle->SetTitleSize(tsize,"x");
  atlasStyle->SetLabelSize(tsize,"y");
  atlasStyle->SetTitleSize(tsize,"y");
  atlasStyle->SetLabelSize(tsize,"z");
  atlasStyle->SetTitleSize(tsize,"z");

  // use bold lines and markers
  atlasStyle->SetMarkerStyle(20);
  atlasStyle->SetMarkerSize(1.2);
  atlasStyle->SetHistLineWidth(2.);
  atlasStyle->SetLineStyleString(2,"[12 12]"); // postscript dashes

  // get rid of X error bars 
  //atlasStyle->SetErrorX(0.001);
  // get rid of error bar caps
  atlasStyle->SetEndErrorSize(0.);

  // do not display any of the standard histogram decorations
  atlasStyle->SetOptTitle(0);
  //atlasStyle->SetOptStat(1111);
  atlasStyle->SetOptStat(0);
  //atlasStyle->SetOptFit(1111);
  atlasStyle->SetOptFit(0);

  // put tick marks on top and RHS of plots
  atlasStyle->SetPadTickX(1);
  atlasStyle->SetPadTickY(1);

  return atlasStyle;

}

//__________________________________________________________________________________|___________

TCanvas* TQHistogramUtils::applyATLASstyle(TH1* histo, const TString& label, double x, double y, double yResize, const TString& xTitle, const TString& yTitle, bool square) {
  //Turns a simple histogram into a plot following ATLAS style guidelines. This is mostly intended for quick plotting and interactive use
  if (!histo) return NULL;
  TH1* h = dynamic_cast<TH1*>( histo->Clone("hcopy") );
  if (!gROOT->GetStyle("ATLAS")) {
  TQHistogramUtils::ATLASstyle(); //yes, this returns a pointer that is not cleared (it is automatically recognized by TROOT, see next line)
  gROOT->SetStyle("ATLAS");
  gROOT->ForceStyle();
  } 
  h->GetXaxis()->SetTitleSize(0.05);
  h->GetYaxis()->SetTitleSize(0.05);
  TCanvas* can = new TCanvas("can_" + TString(histo->GetName()), "can_"+ TString(histo->GetTitle()), square ? 600 : 800, 600); 
  if (xTitle != "none") h->GetXaxis()->SetTitle(xTitle);
  if (yTitle != "none") h->GetYaxis()->SetTitle(yTitle);
  if (yResize > 0) h->SetMaximum(histo->GetMaximum()/yResize);
  h->Draw();
  TLatex l; //l.SetTextAlign(12); l.SetTextSize(tsize); 
  l.SetNDC();
  l.SetTextFont(72);
  //l.SetTextColor(color);
  
  double delx = 0.115*696*gPad->GetWh()/(472*gPad->GetWw());

  l.DrawLatex(x,y,"ATLAS");
  if (label) {
    TLatex p; 
    p.SetNDC();
    p.SetTextFont(42);
    //p.SetTextColor(color);
    p.DrawLatex(x+delx,y,label);
  }
  
  //delete atlasStyle;
  return can;
}

//____________________________________________________________________________________________________

TH1* TQHistogramUtils::unrollHistogram(TH2* input, bool firstX, bool includeUnderflowOverflow){
  // unroll a two-dimensional histogram, concatenating the slices to a single one-dimensional histogram
  const bool binlabels = TQHistogramUtils::hasBinLabels(input);
  const size_t ntot = TQHistogramUtils::getNbinsGlobal(input);
  if(!includeUnderflowOverflow && binlabels){
    return NULL;
  }
  const size_t nx = input->GetXaxis()->GetNbins()+2*includeUnderflowOverflow;
  const size_t ny = input->GetYaxis()->GetNbins()+2*includeUnderflowOverflow;
  const bool extra = includeUnderflowOverflow;
  TH1* hist = new TH1F(input->GetName(),input->GetTitle(),nx*ny,0,1);
  std::cout << ntot << " vs " << TQHistogramUtils::getNbinsGlobal(hist) << std::endl;
  hist->SetDirectory(NULL);
  for(size_t j=0; j<ny; ++j){
    for(size_t i=0; i<nx; ++i){
      size_t newidx;
      if(firstX) newidx = 1 + j*nx + i;
      else       newidx = 1 + i*ny + j;
      size_t bin = input->GetBin(i+!extra,j+!extra);
      hist->SetBinContent(newidx,input->GetBinContent(bin));

      TString labelX,labelY;
      if(i==0 && extra){
        labelX = TString::Format("%s<%g",input->GetXaxis()->GetTitle(),input->GetXaxis()->GetBinLowEdge(1));
      } else if(i+1==nx && extra){
        labelX = TString::Format("%s>%g",input->GetXaxis()->GetTitle(),input->GetXaxis()->GetBinLowEdge(nx));
      } else if(binlabels){
        labelX = input->GetXaxis()->GetBinLabel(i+!extra);
      } else {
        labelX = TString::Format("%s=%g",input->GetXaxis()->GetTitle(),input->GetXaxis()->GetBinCenter(i+!extra));
      }

      if(j==0 && extra){
        labelY = TString::Format("%s<%g",input->GetYaxis()->GetTitle(),input->GetYaxis()->GetBinLowEdge(1));
      } else if(j+1==ny && extra){
        labelY = TString::Format("%s>%g",input->GetYaxis()->GetTitle(),input->GetYaxis()->GetBinLowEdge(ny));
      } else if(binlabels){
        labelY = input->GetYaxis()->GetBinLabel(j+!extra);
      } else {
        labelY = TString::Format("%s=%g",input->GetYaxis()->GetTitle(),input->GetYaxis()->GetBinCenter(j+!extra));
      }
      std::cout << "i=" << i << "/" << nx << ", j=" << j << "/" << ny << " = " << bin << "/" << ntot << " => " << newidx << "/" << hist->GetXaxis()->GetNbins() << std::endl;
      hist->GetXaxis()->SetBinLabel(newidx,labelX+", "+labelY);
    }
  }
  return hist;
}


//____________________________________________________________________________________________________

TObjArray* TQHistogramUtils::getSlices(TH2* input, bool alongX){
  // chop a 2d-histogram into slices along X (or Y)
  TObjArray* list = new TObjArray();
  TAxis* firstaxis = (alongX ? input->GetXaxis() : input->GetYaxis());
  TAxis* secondaxis = (alongX ? input->GetYaxis() : input->GetXaxis());
  for(Int_t i=1; i<=firstaxis->GetNbins(); ++i){
    TString label = firstaxis->GetBinLabel(i);
    if(label.IsNull()){
      label = TString::Format("%s=%g",firstaxis->GetTitle(),firstaxis->GetBinCenter(i));
    }
    TH1F* hist = new TH1F(input->GetName(), label, secondaxis->GetNbins(), secondaxis->GetXmin(), secondaxis->GetXmax());
    hist->GetXaxis()->SetTitle(secondaxis->GetTitle());
    hist->GetYaxis()->SetTitle(input->GetTitle());
    hist->SetDirectory(NULL);
    for(Int_t j=1; j<=secondaxis->GetNbins(); ++j){
      hist->SetBinContent(j, alongX ? input->GetBinContent(i,j) : input->GetBinContent(j,i));
    }
    list->Add(hist);
  }
  return list;
}

//____________________________________________________________________________________________________

bool TQHistogramUtils::isUnderflowOverflowBin(TH1* hist, int bin){
  // return true if the bin is an underflow or overflow bin
  int x,y,z;
  hist->GetBinXYZ(bin, x,y,z);
  if(x == 0 || x == hist->GetNbinsX()) return true;
  if(y == 0 || y == hist->GetNbinsX()) return true;
  if(z == 0 || z == hist->GetNbinsX()) return true;
  return false;
}
  

//____________________________________________________________________________________________________

bool TQHistogramUtils::isGreaterThan(TH1* hist1, TH1* hist2){
  // returns true if each bin of hist1 has a larger or equal value than the corresponding bin of hist2
  const size_t n = TQHistogramUtils::getNbinsGlobal(hist1);
  for(size_t i=0; i<n; ++i){
    if(TQHistogramUtils::isUnderflowOverflowBin(hist1,i)) continue;
    if(hist1->GetBinContent(i) < hist2->GetBinContent(i)){
      DEBUGfunc("bin %d: %g < %g",i,hist1->GetBinContent(i),hist2->GetBinContent(i));
      return false;
    }
  }
  return true;
}

//____________________________________________________________________________________________________

TCanvas* TQHistogramUtils::applyATLASstyle(TGraph* histo, const TString& label, double x, double y, double yResize, const TString& xTitle, const TString& yTitle, bool square) {
  //Turns a TGraph based scatter plot into a plot following ATLAS style guidelines. This is mostly intended for quick plotting and interactive use
  
  if (!histo) return NULL;
  TGraph* h = dynamic_cast<TGraph*>( histo->Clone("hcopy") );
  if (!gROOT->GetStyle("ATLAS")) {
  TQHistogramUtils::ATLASstyle(); //yes, this returns a pointer that is not cleared (it is automatically recognized by TROOT, see next line)
  gROOT->SetStyle("ATLAS");
  gROOT->ForceStyle();
  } 
  int nPoints = h->GetN();
  double xMin,xMax,yMin,yMax;
  h->GetPoint(0,xMin,yMin);
  h->GetPoint(0,xMax,yMax);
  double xi=0.;
  double yi=0.;
  for (int i=0; i<nPoints; i++)
  {
    h->GetPoint(i, xi, yi);
    if (yMax<yi) yMax=yi;
    if (yMin>yi) yMin=yi;
    if (xMax<xi) xMax=xi;
    if (xMin>xi) xMin=xi;
  }
  
  TCanvas* can = new TCanvas("can_" + TString(histo->GetName()), "can_"+ TString(histo->GetTitle()), square ? 600 : 800, 600); 
  if (xTitle != "none") h->GetXaxis()->SetTitle(xTitle);
  if (yTitle != "none") h->GetYaxis()->SetTitle(yTitle);
  yMin -= 0.05*(yMax-yMin);
  if (yResize > 0) h->SetMaximum((yMax-yMin)/yResize+yMin);
  if (yResize > 0) h->SetMinimum(yMin);
  h->GetXaxis()->SetRangeUser(xMin-0.02*(xMax-xMin),xMax+0.02*(xMax-xMin));
  //h->GetXaxis()->SetRangeUser(0.,1.);
  //apply style settings
  h->SetMarkerSize(1);
  h->SetMarkerStyle(20);
  
  h->Draw("AP");
  TLatex l; //l.SetTextAlign(12); l.SetTextSize(tsize); 
  l.SetNDC();
  l.SetTextFont(72);
  //l.SetTextColor(color);
  
  double delx = 0.115*696*gPad->GetWh()/(472*gPad->GetWw());

  l.DrawLatex(x,y,"ATLAS");
  if (label) {
    TLatex p; 
    p.SetNDC();
    p.SetTextFont(42);
    //p.SetTextColor(color);
    p.DrawLatex(x+delx,y,label);
  }
  
  //delete atlasStyle;
  return can;
}

//____________________________________________________________________________________________________

TCanvas* TQHistogramUtils::applyATLASstyle(TMultiGraph* histo, const TString& label, double x, double y, double yResize, const TString& xTitle, const TString& yTitle, bool square) {
  //Turns a TGraph based scatter plot into a plot following ATLAS style guidelines. This is mostly intended for quick plotting and interactive use
  
  if (!histo) {
    WARN("Cannot create canvas, TMultiGraph is a null pointer");    
    return NULL;
  }
  TMultiGraph* h = dynamic_cast<TMultiGraph*>( histo->Clone("hcopy") );
  if (!h) {
    WARN("Failed to clone TMultiGraph");
    return nullptr;
  }
  if (!gROOT->GetStyle("ATLAS")) {
  TQHistogramUtils::ATLASstyle(); //yes, this returns a pointer that is not cleared (it is automatically recognized by TROOT, see next line)
  gROOT->SetStyle("ATLAS");
  gROOT->ForceStyle();
  } 
  
  double xMin,xMax,yMin,yMax;
  bool first = true;
  TQGraphIterator itr(h->GetListOfGraphs());
  while(itr.hasNext()) {
    TGraph* gr = itr.readNext();
    if (!gr) continue;
    gr->SetMarkerSize(1);
    gr->SetMarkerStyle(20);
    int nPoints = gr->GetN();
    if (first) {
      gr->GetPoint(0,xMin,yMin);
      gr->GetPoint(0,xMax,yMax);
      first = false;
    }
    double xi=0.;
    double yi=0.;
    for (int i=0; i<nPoints; i++)
    {
      gr->GetPoint(i, xi, yi);
      if (yMax<yi) yMax=yi;
      if (yMin>yi) yMin=yi;
      if (xMax<xi) xMax=xi;
      if (xMin>xi) xMin=xi;
    }
  }
  
  TCanvas* can = new TCanvas("can_" + TString(histo->GetName()), "can_"+ TString(histo->GetTitle()), square ? 600 : 800, 600); 
  yMin -= 0.05*(yMax-yMin);
  if (yResize > 0) h->SetMaximum((yMax-yMin)/yResize+yMin);
  if (yResize > 0) h->SetMinimum(yMin);
  //h->GetXaxis()->SetRangeUser(0.,1.);
  //apply style settings
  //h->SetMarkerSize(1);
  //h->SetMarkerStyle(20);
  
  h->Draw("AP"); //dummy Draw call to create x/y axis member objects. "A" option must be present!
  if (h->GetXaxis()) {
  if (xTitle != "none") h->GetXaxis()->SetTitle(xTitle);
  if (yTitle != "none") h->GetYaxis()->SetTitle(yTitle);
  h->GetXaxis()->SetRangeUser(xMin-0.02*(xMax-xMin),xMax+0.02*(xMax-xMin));
  }
  h->Draw("AP");
  TLatex l; //l.SetTextAlign(12); l.SetTextSize(tsize); 
  l.SetNDC();
  l.SetTextFont(72);
  //l.SetTextColor(color);
  
  double delx = 0.115*696*gPad->GetWh()/(472*gPad->GetWw());

  l.DrawLatex(x,y,"ATLAS");
  if (label) {
    TLatex p; 
    p.SetNDC();
    p.SetTextFont(42);
    //p.SetTextColor(color);
    p.DrawLatex(x+delx,y,label);
  }
  
  //delete atlasStyle;
  return can;
}

//____________________________________________________________________________________________________

void TQHistogramUtils::rerollGauss(TH1* hist, double zvalue){
  // reroll a histogram, i.e. replace every bin content with a new
  // number randomly selected from a gaussian distribution where mean
  // and width are given by bin content and bin error of the histogram.
  // zvalue can be used to scale up or down the width of the gaussian.
  const size_t n = TQHistogramUtils::getNbinsGlobal(hist);
  TRandom3 rand;
  for(size_t i=0; i<n; ++i){
    double newvalue = rand.Gaus(hist->GetBinContent(i),zvalue*hist->GetBinError(i));
    hist->SetBinContent(i,newvalue);
  }
}

//____________________________________________________________________________________________________

void TQHistogramUtils::rerollPoisson(TH1* hist){
  // reroll a histogram, i.e. replace every bin content with a new
  // number randomly selected from a poisson distribution where the mean
  // is given by bin content of the histogram.
  const size_t n = TQHistogramUtils::getNbinsGlobal(hist);
  TRandom3 rand;
  for(size_t i=0; i<n; ++i){
    int newvalue = rand.Poisson(hist->GetBinContent(i));
    hist->SetBinContent(i,newvalue);
    hist->SetBinError(i,sqrt(newvalue));
  }
}

//____________________________________________________________________________________________________

bool TQHistogramUtils::envelopeUpper(TH1* hist, TH1* otherhist){
  // envelope a histogram by another one, taking the bin-by-bin maximum
  const size_t n = TQHistogramUtils::getNbinsGlobal(hist);
  const size_t entries = std::max(hist->GetEntries(),otherhist->GetEntries());
  for(size_t i=0; i<n; ++i){
    const double thisval = hist->GetBinContent(i);
    const double otherval = otherhist->GetBinContent(i);
    if(otherval > thisval){
      hist->SetBinContent(i,otherval);
      hist->SetBinError(i,otherhist->GetBinError(i));
    }
  }
  hist->SetEntries(entries);
  return true;
}

//____________________________________________________________________________________________________

bool TQHistogramUtils::envelopeLower(TH1* hist, TH1* otherhist){
  // envelope a histogram by another one, taking the bin-by-bin minimum
  const size_t n = TQHistogramUtils::getNbinsGlobal(hist);
  const size_t entries = std::min(hist->GetEntries(),otherhist->GetEntries());
  for(size_t i=0; i<n; ++i){
    const double thisval = hist->GetBinContent(i);
    const double otherval = otherhist->GetBinContent(i);
    if(otherval < thisval){
      hist->SetBinContent(i,otherval);
      hist->SetBinError(i,otherhist->GetBinError(i));
    }
  }
  hist->SetEntries(entries);
  return true;
}

//____________________________________________________________________________________________________

template<class TMatrixTT>
TH2* TQHistogramUtils::convertMatrixToHistogram(const TMatrixTT* matrix, const TString& name){
  // convert a matrix into a 2d histogram
  if(!matrix) return NULL;
  const size_t ncols = matrix->GetNcols();
  const size_t nrows = matrix->GetNrows();
  TH2* hist = new TH2D(name,name,ncols,0,ncols,nrows,0,nrows);
  hist->SetDirectory(NULL);
  for(size_t i=0; i<nrows; ++i){
    for(size_t j=0; j<nrows; ++j){
      size_t bin = hist->GetBin(i+1,j+1);
      hist->SetBinContent(bin,(*matrix)(i,j));
      hist->SetBinError(bin,0);
    }
  }
  hist->SetEntries(ncols*nrows);
  return hist;
}

namespace TQHistogramUtils {
  template TH2* convertMatrixToHistogram<TMatrixT<float > >(const TMatrixT<float >* matrix, const TString& name);
  template TH2* convertMatrixToHistogram<TMatrixT<double> >(const TMatrixT<double>* matrix, const TString& name);
  template TH2* convertMatrixToHistogram<TMatrixTSym<float > >(const TMatrixTSym<float >* matrix, const TString& name);
  template TH2* convertMatrixToHistogram<TMatrixTSym<double> >(const TMatrixTSym<double>* matrix, const TString& name);
}

//____________________________________________________________________________________________________

TMatrixD* TQHistogramUtils::convertHistogramToMatrix(TH2* hist){
  // convert a 2d histogram into a matrix
  if(!hist) return NULL;
  const size_t ncols = hist->GetNbinsX();
  const size_t nrows = hist->GetNbinsY();
  TMatrixD* matrix = new TMatrixD(nrows,ncols);
  for(size_t i=0; i<nrows; ++i){
    for(size_t j=0; j<nrows; ++j){
      size_t bin = hist->GetBin(i+1,j+1);
      (*matrix)(i,j) = hist->GetBinContent(bin);
    }
  }
  return matrix;
}

//____________________________________________________________________________________________________

int TQHistogramUtils::fixHoles1D(TH1* hist, double threshold) {
  // fix holes (empty bins below threshold) in a 1D histogram
  // this function uses simple linear interpolation/extrapolation from the 2 adjacent bins
  // underflow/overflow bins are ignored
  int nDim = TQHistogramUtils::getDimension(hist);
  if(nDim != 1){
    throw std::runtime_error("unable to patch shape systematics for multi-dimensional histograms, skipping!");
  }
  
  if(hist->GetNbinsX() < 3){
    throw std::runtime_error("unable to patch shape systematics for histograms with less than 3 bins, skipping!");
  }
  
  int fixed = 0;
  if(hist->GetBinContent(1) < threshold){
    if((fabs(hist->GetBinContent(2)) < threshold) || (fabs(hist->GetBinContent(3)) < threshold)){
      throw std::runtime_error("refusing to fix holes greater than single bins!");
    }
    hist->SetBinContent(1,     2*    hist->GetBinContent(2)    -     hist->GetBinContent(3));
    hist->SetBinError  (1,sqrt(2*pow(hist->GetBinError  (2),2) + pow(hist->GetBinError  (3),2)));
    fixed++;
  }
  
  int n = hist->GetNbinsX();
  if(hist->GetBinContent(n) < threshold){
    if((fabs(hist->GetBinContent(n-1)) < threshold) || (fabs(hist->GetBinContent(n-2)) < threshold)){
      throw std::runtime_error("refusing to fix holes greater than single bins!");
    }
    hist->SetBinContent(n,     2*    hist->GetBinContent(n-1)    -     hist->GetBinContent(n-2));
    hist->SetBinError  (n,sqrt(2*pow(hist->GetBinError  (n-1),2) + pow(hist->GetBinError  (n-2),2)));
    fixed++;
  }
  
  // loop over all "internal" bins
  for(int i=2; i<n; ++i){
    if(hist->GetBinContent(i) < threshold){
      if((fabs(hist->GetBinContent(i-1)) < threshold) || (fabs(hist->GetBinContent(i+1)) < threshold)){
        throw std::runtime_error("refusing to fix holes greater than single bins!");
      } 
      hist->SetBinContent(i,0.5 *     (    hist->GetBinContent(i-1)    +     hist->GetBinContent(i+1)));
      hist->SetBinError  (i,0.5 * sqrt(pow(hist->GetBinError  (i-1),2) + pow(hist->GetBinError  (i+1),2)));
      fixed++;
    }
  }
  return fixed;
}

//__________________________________________________________________________________|___________

void TQHistogramUtils::scaleErrors(TH1* hist, double scale){
  // scale the errors of a histogram by an arbitrary number
  size_t nbins = TQHistogramUtils::getNbinsGlobal(hist);
  for(size_t i=0; i<nbins; ++i){
    hist->SetBinError(i,scale*hist->GetBinError(i));
  }
}

//__________________________________________________________________________________|___________
/*
void TQHistogramUtils::saveAs(TPad* canvas, const TString& filename_) {
  // wraper around TPad::SaveAs which hacks into the ROOT pdf creation to set more 
  // usefull meta information
  if (!filename_.EndsWith(".pdf")) { //meta information is only added for pdfs. For other formats we leave things untouched
    canvas->SaveAs(filename_);
    return;
  }
  TString filename = filename_;
  //at this point we should be sure to output as a pdf
  //TPDF* pdf = new TPDF(filename);
  canvas->Print(filename+"(","pdf");
  TSeqCollection* specials = gROOT->GetListOfSpecials();
  if (specials) {
    TPDF* pdf = dynamic_cast<TPDF*>(specials->FindObject(filename.Data()));
    if (!pdf) return;
    //specials->AddLast(pdf);
    pdf->WriteCompressedBuffer(); //otherwise TPDF seems to just ignore the following calls....
    pdf->NewObject(2); //2 is the index root used for the meta information block. We simply create our own, replacing the refference to the original with a refference (toc enty) to our own.
    pdf->PrintStr("<<@"); //@ is a fancy way of denoting a newline character in TPDF. Likely doesn't make a difference, but oh well....
    pdf->PrintStr("/Creator (rgugel using QFramework/CAF)@");
    pdf->PrintStr("/Keywords (Dear ROOT, get hacked!)@");
    pdf->PrintStr(">>@");
    pdf->PrintStr("endobj@");
    canvas->Print(filename.Append("]"),"pdf"); // appended ']' = force close file (we already opened it when creating the TPDF object)
    //we should be done now, so let's clean up a bit
    //specials->Remove(pdf);
    //delete pdf;
    
  } else { //fallback if something goes wrong (should never happen)
    canvas->SaveAs(filename);
  }
  
  return;
}
*/

