#include "TObjArray.h"
#include "TList.h"
#include "TObjString.h"
#include "TFolder.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TProfile.h"
#include "TProfile2D.h"

#include "QFramework/TQHistoMakerAnalysisJob.h"
#include "QFramework/TQIterator.h"
#include "QFramework/TQHistogramUtils.h"
#include "QFramework/TQObservable.h"
#include "QFramework/TQStringUtils.h"
#include "QFramework/TQCompiledCut.h"
#include "QFramework/TQTaggable.h"
#include "QFramework/TQSample.h"
#include "QFramework/TQUtils.h"

// #define _DEBUG_
#include "QFramework/TQLibrary.h"

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <cstdlib>

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQHistoMakerAnalysisJob:
//
// The TQHistoMakerAnalysisJob is the de-facto standard analaysis job for booking histograms.
// The booking can be done with expressions of the following type:
// 
// A one-dimensional histogram is defined. The value "Mjj/1000" is
// filled into the histogram, and the label of the x-axis will be
// '\#it{m}_{jj} [GeV]'. Please note that the '#'-symbol is escaped by
// a backslash. If the backslash is omitted, the parser will interpret
// anything following the '#' as a comment and ignore it, resulting in
// a syntax error while parsing this histogram.
//
//    TH1F('Mjj', '', 40, 0., 800.) << (Mjj/1000. : '\#it{m}_{jj} [GeV]'); 
//
// A one-dimensional histogram with variable binning is defined.
//
//    TH1F('Mjj2', '', {0,200,600,800,1200,2000}) << (Mjj/1000. : '\#it{m}_{jj} [GeV]'); 
//
// A two-dimensional histogram is defined with labels for the x- and y-axis.
//
//    TH2F('Mjj_DYjj', '', 40, 0., 800., 30, 0., 5.) << (Mjj/1000. : '\#it{m}_{jj} [GeV]', DYjj : '\#it{\#Delta Y}_{jj}'); 
//
// Two of the above histograms will be attached to the Cut named "Cut_2jetincl" and all descendant cuts. 
//
//    @Cut_2jetincl/*: Mjj_DYjj,Mjj2; 
//
// Please note that the TQHistoMakerAnalysisJob will also understand
// definitions of TH[1/2/3][D/F/C/I] as well as TProfile and
// TProfile2D.
//
// Additional option for individual histograms can be specified as in the following example: 
//
//    TH1F('Mjj', '', 40, 0., 800.) << (Mjj/1000. : '\#it{m}_{jj} [GeV]') << (fillRaw=true, someOtherOption="someSetting"); 
//
// The content between the last pair of parentheses is read as a list
// of tags (see TQTaggable::importTags). Supported options are then
// applied to all booking instances (i.e. for all different cuts) of
// the particular histogram. Currently supported options include:
//
//    fillRaw=true 
//
// Ignore event weights when filling this histogram/profile:
//
//    fillSynchronized=true 
//
// Used for profiles and multidimensional histograms, i.e., everything
// but TH1x. When using vector type observables (observables with
// possibly more than one evaluation per event), by default, all
// combinations of values provided by the observables corresponding to
// the individual axes are filled. By specifying this option the
// behavior is changed such that only combinations of values are
// filled where their indices are equal. This then also enforces all
// non-scalar observables for the histogram/profile to have the same
// number of evaluations (same 'length' of the vector/list of values
// they represent). Scalar type observables are implicitly expanded to
// match the size of the vector observables as if it would return the
// same value for every index.
//
//    weightExpression="myFancyWeight" 
// 
// Similar to a weight expression used in a cut but only applied to
// the particular histogram. If "myFancyWeight" corresponds to a
// vector type observable, its number of evaluations must be equal to
// the number of evaluations of the regular observables used for this
// histogram/profile. For multidimensional histograms and profiles the
// use of a vector valued weight is only supported in combination with
// fillSynchronized=true and enforces the weight observable to be of
// equal length as the regular (non-scalar) observables
//
////////////////////////////////////////////////////////////////////////////////////////////////

TQMessageStream TQHistoMakerAnalysisJob::f_ErrMsg(new std::stringstream());
bool TQHistoMakerAnalysisJob::g_useHistogramObservableNames(false);

ClassImp(TQHistoMakerAnalysisJob)

//__________________________________________________________________________________|___________

TQHistoMakerAnalysisJob::TQHistoMakerAnalysisJob() : 
TQAnalysisJob("TQHistoMakerAnalysisJob"),
  f_Verbose(0),
  poolAt(NULL)
{
  // standard constructor
}

//__________________________________________________________________________________|___________

TQHistoMakerAnalysisJob::TQHistoMakerAnalysisJob(TQHistoMakerAnalysisJob* other) :
  TQAnalysisJob(other ? other->GetName() : "TQHistoMakerAnalysisJob"),
  f_Verbose(other ? other->f_Verbose : 0),
  poolAt(other ? other->poolAt : NULL)
{
  // copy constructor
  for(size_t i=0; i<other->fHistogramTemplates.size(); ++i){
    this->fHistogramTemplates.push_back(TQHistogramUtils::copyHistogram(other->fHistogramTemplates[i]));
    this->fFillSynchronized.push_back(other->fFillSynchronized[i]);
    this->fFillRaw.push_back(other->fFillRaw[i]);
    this->fHistoTypes.push_back(other->fHistoTypes[i]);
    this->fExpressions.push_back(std::vector<TString>());
    for(size_t j=0; j<other->fExpressions[i].size(); ++j){
      this->fExpressions[i].push_back(other->fExpressions[i][j]);
    }
    this->fWeightExpressions.push_back(other->fWeightExpressions[i]);
  }
}

//__________________________________________________________________________________|___________

void TQHistoMakerAnalysisJob::setVerbose(int verbose) {
  // set verbosity
  f_Verbose = verbose;
}


//__________________________________________________________________________________|___________

int TQHistoMakerAnalysisJob::getVerbose() {
  // retrieve verbosity
  return f_Verbose;
}


//__________________________________________________________________________________|___________

void TQHistoMakerAnalysisJob::setErrorMessage(TString message) {
 
  /* update the error message */
  f_ErrMsg.sendClassFunctionMessage(TQMessageStream::INFO,this->Class(),"<anonymous>",message);
 
  /* print the error message if in verbose mode */
  if (f_Verbose > 0) INFOclass(message);
}

//__________________________________________________________________________________|___________

void TQHistoMakerAnalysisJob::clearMessages(){
  // clear the error messages
  f_ErrMsg.clearMessages();
}

//__________________________________________________________________________________|___________

TString TQHistoMakerAnalysisJob::getErrorMessage() {
  // Return the latest error message
  return f_ErrMsg.getMessages();
}


//__________________________________________________________________________________|___________

const TString& TQHistoMakerAnalysisJob::getValidNameCharacters() {
  // retrieve a string with valid name characters
  return TQFolder::getValidNameCharacters();
}


//__________________________________________________________________________________|___________

void TQHistoMakerAnalysisJob::cancelHistogram(const TString& name) {
  // cancel the histgogram with the given name
  for(size_t i=fHistogramTemplates.size(); i >0 ; i--){
    if(fHistogramTemplates.at(i)->GetName() == name){
      fHistogramTemplates.erase(fHistogramTemplates.begin()+i-1);
      fFillSynchronized.erase(fFillSynchronized.begin()+i-1);
      fFillRaw.erase(fFillRaw.begin()+i-1);
      fExpressions.at(i-1).clear();
      fExpressions.erase(fExpressions.begin()+i-1);
      fWeightExpressions.erase(fWeightExpressions.begin()+i-1);
      fHistoTypes.erase(fHistoTypes.begin()+i-1);
      return;
    }
  }
}

//__________________________________________________________________________________|___________

namespace {
  void setupAxis(TAxis* axis, const TString& title, const TString& expr, const std::vector<TString>& labels){
    axis->SetTitle(title);
    axis->SetName(expr);
    for(size_t i=0; i<labels.size(); ++i){
      axis->SetBinLabel(i+1,labels[i]);
    }
  }
}
//__________________________________________________________________________________|___________

bool TQHistoMakerAnalysisJob::bookHistogram(TString definition, TQTaggable* aliases) {
  DEBUGclass("entering function - booking histogram '%s'",definition.Data());

  if(definition.IsNull()){
    this->f_ErrMsg.sendClassFunctionMessage(TQMessageStream::INFO,this->Class(),__FUNCTION__,"obtained empty histogram definition");
    return false;
  }
  
  // histogram definition
  TString histoDef;
  TQStringUtils::readUpTo(definition, histoDef, "<", "()[]{}", "''\"\"");

  // create histogram template from definition
  TString msg;
  DEBUGclass("creating histogram '%s'",histoDef.Data());


  if(histoDef.IsNull()){
    this->f_ErrMsg.sendClassFunctionMessage(TQMessageStream::INFO,this->Class(),__FUNCTION__,"histogram constructor is empty, remainder is '%s'",definition.Data());
    return false;
  }
  
  TH1 * histo = TQHistogramUtils::createHistogram(histoDef, msg);
  DEBUGclass(histo ? "success" : "failure");

  if (!histo) {
    this->f_ErrMsg.sendClassFunctionMessage(TQMessageStream::ERROR,this->Class(),__FUNCTION__,msg);
    return false;
  }

  // invalid histogram name?
  if (!TQFolder::isValidName(histo->GetName())) {
    this->f_ErrMsg.sendClassFunctionMessage(TQMessageStream::INFO,this->Class(),__FUNCTION__,"'%s' is an invalid histogram name", histo->GetName());
    delete histo;
    return false;
  }

  // read "<<" operator
  if (TQStringUtils::removeLeading(definition, "<") != 2) {
    this->f_ErrMsg.sendClassFunctionMessage(TQMessageStream::INFO,this->Class(),__FUNCTION__,"Operator '<<' expected after histogram definition");
    delete histo;
    return false;
  }

  //split off a possile option block
  std::vector<TString> settingTokens = TQStringUtils::split(definition, "<<", "([{'\"", ")]}'\"");
  if (settingTokens.size()<1) {
    this->f_ErrMsg.sendClassFunctionMessage(TQMessageStream::INFO,this->Class(),__FUNCTION__,"cannot parse definition block '%s'",definition.Data());
    delete histo; 
    return false;
  }
  definition = settingTokens[0];
  TQTaggable options;
  if (settingTokens.size()>1) {
    TString optionBlock;
    TQStringUtils::readBlanksAndNewlines(settingTokens[1]);
    if (!TQStringUtils::readBlock(settingTokens[1], optionBlock, "()[]{}", "''\"\"")) {
      this->f_ErrMsg.sendClassFunctionMessage(TQMessageStream::INFO,this->Class(),__FUNCTION__,"Failed to parse histogram option block '%s'", settingTokens[1].Data());
      delete histo;
      return false;
    }
    options.importTags(optionBlock);
  }
  
  // read expression block
  TString expressionBlock;
  TQStringUtils::readBlanksAndNewlines(definition);
  if (!TQStringUtils::readBlock(definition, expressionBlock, "()[]{}", "''\"\"")) {
    this->f_ErrMsg.sendClassFunctionMessage(TQMessageStream::INFO,this->Class(),__FUNCTION__,"Missing expression block after '<<' operator");
    delete histo;
    return false;
  }

  // tokenize expression block (one token per histogram dimension)
  TList * expressionTokens = TQStringUtils::tokenize(expressionBlock, ",", true, "()[]{}", "''\"\"");
  if(expressionTokens->GetEntries() < 1){
    this->f_ErrMsg.sendClassFunctionMessage(TQMessageStream::INFO,this->Class(),__FUNCTION__,"cannot parse expression block '%s'",expressionBlock.Data());
    delete histo;
    return false;
  }

  DEBUGclass("parsing expression block '%s', found %d entries",expressionBlock.Data(),expressionTokens->GetEntries());

  // read expression block tokens
  std::vector<TString> exprs;
  std::vector<TString> titles;
  std::vector<std::vector<TString> > labels ;
  TQIterator itr(expressionTokens);
  while (itr.hasNext()) {
    TString token(itr.readNext()->GetName());
    // read expression 
    TString expr;
    int nColon = 0;
    while(nColon != 1){
      TQStringUtils::readUpTo(token, expr, "\\:", "()[]{}", "''\"\"");
      nColon = TQStringUtils::countLeading(token, ":");
      if (nColon == 1) {
        break;
      } else if (nColon == 0) {
        if(TQStringUtils::removeLeading(token,"\\")){
          TQStringUtils::readToken(token,expr,":");
          continue;
        }
        this->f_ErrMsg.sendClassFunctionMessage(TQMessageStream::INFO,this->Class(),__FUNCTION__,"':' expected after expression");
        DEBUGclass("':' expected after expression");
        delete histo;
        return false;
      } else {
        TQStringUtils::readToken(token,expr,":");
        continue;
      }
    }
    TQStringUtils::readBlanksAndNewlines(token);
    TQStringUtils::removeLeading(token,":");
    TQStringUtils::readBlanksAndNewlines(token);
    // use TQTaggable to read title (handling of e.g. quotes)
    TString buffer;
    TString title;
    std::vector<TString> binLabels;
    if(TQStringUtils::readBlock(token,buffer,"''","",false,false) > 0 || TQStringUtils::readBlock(token,buffer,"\"\"","",false,false) > 0){
      title = buffer;
      buffer.Clear();
      TQStringUtils::readBlanksAndNewlines(token);
      if(TQStringUtils::removeLeading(token,":") == 1){
        TQStringUtils::readBlanksAndNewlines(token);
        TQStringUtils::readBlock(token,buffer,"(){}[]","",false,false);
        binLabels = TQStringUtils::split(buffer,",");
        for(size_t i=0; i<binLabels.size(); ++i){
          TQStringUtils::removeLeading(binLabels[i]," ");
          TQStringUtils::removeTrailing(binLabels[i]," ");
          TQStringUtils::unquoteInPlace(binLabels[i]);
        }
      }
    } else {
      title = TQStringUtils::unquote(token);
    }
    
    // store expression and title
    const TString expression(aliases ? aliases->replaceInTextRecursive(expr) : expr);
    exprs.push_back(TQStringUtils::trim(expression));
    titles.push_back(TQStringUtils::trim(title));
    labels.push_back(binLabels);
    DEBUGclass("found expression and title: '%s' and '%s'",expr.Data(),title.Data());
  }

  // histogram properties
  TString name = histo->GetName();
  int dim = TQHistogramUtils::getDimension(histo);

  // check dimension of histogram and expression block
  if ( ( dim > 0 && dim != (int)exprs.size() ) || ( dim < 0 && (int)exprs.size() != abs(dim)+1) ) { // last ist the TProfile case
    this->f_ErrMsg.sendClassFunctionMessage(TQMessageStream::INFO,this->Class(),__FUNCTION__,"Dimensionality of histogram (%d) and expression block (%d) don't match", dim, (int)(exprs.size()));
    delete histo;
    DEBUGclass("Dimensionality of histogram (%d) and expression block (%d) don't match", dim, (int)(exprs.size()));
    return false;
  }

  // check name of histogram
  if (!TQStringUtils::isValidIdentifier(name, getValidNameCharacters())) {
    this->f_ErrMsg.sendClassFunctionMessage(TQMessageStream::INFO,this->Class(),__FUNCTION__,"Invalid histogram name '%s'", name.Data());
    DEBUGclass("Invalid histogram name '%s'", name.Data());
    delete histo;
    return false;
  }

  // stop if histogram with 'name' already has been booked
  bool exists = false;
  int i = 0;
  while (!exists && i < (int)(fHistogramTemplates.size()))
    exists = (name.CompareTo(fHistogramTemplates.at(i++)->GetName()) == 0);
  if (exists) {
    this->f_ErrMsg.sendClassFunctionMessage(TQMessageStream::INFO,this->Class(),__FUNCTION__,"Histogram with name '%s' has already been booked", name.Data());
    DEBUGclass("Histogram with name '%s' has already been booked", name.Data());
    delete histo;
    return false;
  }
 
  // set up tree observables corresponding to expressions
  for (int i = 0; i < (int)exprs.size(); i++) {
    if (i == 0) {
      setupAxis(histo->GetXaxis(),titles[i],exprs[i],labels[i]);
    }
    if (i == 1) {
      setupAxis(histo->GetYaxis(),titles[i],exprs[i],labels[i]);
    }
    if (i == 2) {
      setupAxis(histo->GetZaxis(),titles[i],exprs[i],labels[i]);
    }
  }
  
  fExpressions.push_back(exprs);
  fHistoTypes.push_back(TQHistogramUtils::getDimension(histo));
  fHistogramTemplates.push_back(histo);
  //@tag: [fillSynchronized] This tag is read from an additional option block in the histogram definition, e.g. TH2(<histogram definition>) << (<variable definitions>) << (fillSynchronized = true) . This tag defaults to false and is only relevant if vector observables are used in multi dimensional histograms. By default all combinations of values from the different observables are filled. If this tag is set to 'true', however, all vector valued observables are required to have the same number of evaluations (same dimensionality). Only combinations of values with the same index in the respective vector observables are filled. If a vector observable is used in combination with a non-vector observable the latter one is evaluated the usual way for each entry of the vector observable.
  fFillSynchronized.push_back(options.getTagBoolDefault("fillSynchronized",false));
  //@tag: [weightExpression] This tag is read from an additional option block in the histogram definition (see tag 'fillSynchronized'). It allows to specify an expression which is multiplied on top of the weight provided by the respective cut. If a vector observable is used for the values to be filled into the histogram, the weight expression needs to either correspond to a scalar observable, or to a vector observable with the same number of evaluations as the value observable. If the latter case (vector valued weight) is used for multidimensional histograms (TH2, TH3), 'fillSynchronized=true' must be specified or an error will be thrown  (reason: if every combination of evaluations of vector observables is filled the matching to a particular weight for this combination is non-trivial. If there is a particular use case where this is needed, please address suggestions to the core developers)
  const TString wExpr(aliases ? aliases->replaceInTextRecursive(options.getTagStringDefault("weightExpression","")) : options.getTagStringDefault("weightExpression",""));
  fWeightExpressions.push_back(wExpr);
  //if (options.hasTagString("weightExpression")) {
  //  std::cout<<"found histogram with weight expression '"<<options.getTagStringDefault("weightExpression","")<<"'"<<std::endl;
  //}
  //@tag: [fillRaw] This tag is read from an additional option block in the histogam definition (see tag 'fillSynchronized'). It causes all event weights to be ignored for the corresponding histogram. Please note that if a dedicated weight expression (see tag 'weightExpression') is provided it will still be used. Default: false
  fFillRaw.push_back(options.getTagBoolDefault("fillRaw",false));
  
  return true;
}

//__________________________________________________________________________________|___________

void TQHistoMakerAnalysisJob::print(const TString& options) {
  // overloading the standard print-routine, internally calling printBooking[TeX], depending on options
  if(options.Contains("TeX")){
    this->printBookingTeX(this->GetName());
  } else {
    this->printBooking(this->GetName());
  }
}

//__________________________________________________________________________________|___________

void TQHistoMakerAnalysisJob::printBookingTeX(const TString& moretext) {
  // print booked histograms (LaTeX format)
  std::cout << "\\begin{tabular}{l l l l l}" << std::endl;;
  std::cout << "\\multicolumn{2}{l}{\\bfseries Booked Histograms}" << " & \\multicolumn{2}{l}{" << moretext  << "} & Options \\tabularnewline" << std::endl;
  for (size_t i = 0;i < fHistogramTemplates.size(); i++) {
    TString exp(TQStringUtils::concat(fExpressions.at(i),","));
    exp.ReplaceAll("$","\\$");
    //create a list of additional options
    std::vector<TString> options;
    if (fFillSynchronized.size()>i && fFillSynchronized.at(i)) options.push_back(TString("fillSynchronized=true"));
    if (fFillRaw.size()>i && fFillRaw.at(i)) options.push_back(TString("fillRaw=true"));
    if (fWeightExpressions.size()>i && fWeightExpressions.at(i).Length()>0) options.push_back(TString::Format("weightExpression='%s'",fWeightExpressions.at(i).Data()).ReplaceAll("$","\\$"));
    //merge the list into a single string
    TString optionString(TQStringUtils::concat(options,","));
    std::cout << fHistogramTemplates.at(i)->GetName() << " & "
              << fHistogramTemplates.at(i)->GetTitle()  << " & "
              << TQHistogramUtils::getDetailsAsString(fHistogramTemplates.at(i), 2)  << " & "
              << exp << " & "
              << (optionString.Length()>0 ? optionString.Data() : "");
    if(i != fHistogramTemplates.size() -1) std::cout << " \\tabularnewline ";
    std::cout<< std::endl;
  }
  std::cout << "\\end{tabular}" << std::endl;
}

//__________________________________________________________________________________|___________

void TQHistoMakerAnalysisJob::printBooking(const TString& moretext) {
  // print booked histograms
  std::cout << TQStringUtils::makeBoldBlue("Booked Histograms") << " " << TQStringUtils::makeBoldWhite(moretext) << std::endl;
  for (size_t i = 0;i < fHistogramTemplates.size(); i++) {
    //compile list of additional options for this histogram
    std::vector<TString> options;
    if (fFillSynchronized.size()>i && fFillSynchronized.at(i)) options.push_back(TString("fillSynchronized=true"));
    if (fFillRaw.size()>i && fFillRaw.at(i)) options.push_back(TString("fillRaw=true"));
    if (fWeightExpressions.size()>i && fWeightExpressions.at(i).Length()>0) options.push_back(TString::Format("weightExpression='%s'",fWeightExpressions.at(i).Data()).ReplaceAll("$","\\$"));
    
    std::cout << TQStringUtils::fixedWidth(fHistogramTemplates.at(i)->GetName(),20)
              << TQStringUtils::fixedWidth(fHistogramTemplates.at(i)->GetTitle(),20)
              << TQHistogramUtils::getDetailsAsString(fHistogramTemplates.at(i), 2)
              << " << " << TQStringUtils::concat(fExpressions.at(i)," : ");
    if (options.size()>0) std::cout << " << ( " << TQStringUtils::concat(options,",") << " ) "; //append option block if applicable
              std::cout << std::endl;
  }
}


//__________________________________________________________________________________|___________

bool TQHistoMakerAnalysisJob::execute(double weight) {
  DEBUGclass("filling histograms for event...");
  // execute this analysis job, filling all histograms
  int nEvals = 0;
  int nWeightEvals = 0;
  
  for (unsigned int i = 0; i < fHistograms.size(); ++i) {
    #ifdef _DEBUG_
    if(this->fObservables.size() < i){
      throw std::runtime_error("insufficient size of observable vector!");
    }
    #endif
    // DEBUGclass("switching for histo type");
    switch (fHistoTypes[i]) {
    case -2:
      if (this->fFillSynchronized[i]) {
        TRY(
        nEvals = std::max(std::max(fObservables[i][0]->getNevaluations(),fObservables[i][1]->getNevaluations()),fObservables[i][2]->getNevaluations());
        ,TString::Format("Failed to get number of evaluations from at least one of the observables '%s', '%s', '%s' for filling histogram '%s' at cut '%s'.", fObservables[i][0]->GetName(), fObservables[i][1]->GetName(), fObservables[i][2]->GetName(), fHistograms[i]->GetName(), this->getCut()->GetName())
        )
        //at this point we should be safe calling getNevaluations
        if (fObservables[i][0]->getNevaluations() != nEvals &&  fObservables[i][0]->getNevaluations() != 1) throw std::runtime_error("Cannot perform synchonized histogram filling: Number of evaluations do not match!");
        if (fObservables[i][1]->getNevaluations() != nEvals &&  fObservables[i][1]->getNevaluations() != 1) throw std::runtime_error("Cannot perform synchonized histogram filling: Number of evaluations do not match!");
        if (fObservables[i][2]->getNevaluations() != nEvals &&  fObservables[i][2]->getNevaluations() != 1) throw std::runtime_error("Cannot perform synchonized histogram filling: Number of evaluations do not match!");
        TRY(
        nWeightEvals = fWeightObservables[i] ? fWeightObservables[i]->getNevaluations() : 1;
        ,TString::Format("Failed to get number of evaluations from histogram-entry-weight observable '%s' for filling histogram '%s' at cut '%s'", fWeightObservables[i]?fWeightObservables[i]->GetName():"", fHistograms[i]->GetName(), this->getCut()->GetName())
        )
        if (nWeightEvals != 1 && nWeightEvals != nEvals) {
          throw std::runtime_error( TString::Format("Histogram specific weight is neither scalar nor does the number of evaluations of the responsible observable (%d) match the number of evaluations of the value observable(s) (%d). Please check the definition of histogram '%s' with weight observable '%s'! ",nWeightEvals,nEvals,fHistograms[i]->GetName(),fWeightObservables[i]?fWeightObservables[i]->GetName():"").Data());
        }
        
        TRY(
        for (int a = 0; a<nEvals; ++a) {
          ((TProfile2D*)(fHistograms[i]))->Fill(fObservables[i][0]->getValueAt(fObservables[i][0]->getNevaluations()==1?0:a), 
                                                fObservables[i][1]->getValueAt(fObservables[i][1]->getNevaluations()==1?0:a), 
                                                fObservables[i][2]->getValueAt(fObservables[i][2]->getNevaluations()==1?0:a), 
                                                (this->fFillRaw[i]? 1. : weight*fSample->getNormalisation())
                                                * ( fWeightObservables[i]?fWeightObservables[i]->getValueAt(std::min(a,std::max(0,nWeightEvals-1)) ) : 1.) );
        }
        ,TString::Format("Failed to fill histogram '%s' using the observables '%s', '%s', '%s' at cut '%s'.", fHistograms[i]->GetName(), fObservables[i][0]->GetName(), fObservables[i][1]->GetName(), fObservables[i][2]->GetName(), this->getCut()->GetName())
        )
        
      } else { //filling every combination of entries provided by the different (vector) observables
        TRY(
        nWeightEvals = fWeightObservables[i] ? fWeightObservables[i]->getNevaluations() : 1;
        ,TString::Format("Failed to get number of evaluations from histogram-entry-weight observable '%s' for filling histogram '%s' at cut '%s'", fWeightObservables[i]?fWeightObservables[i]->GetName():"", fHistograms[i]->GetName(), this->getCut()->GetName())
        )
        if (nWeightEvals != 1) {
          throw std::runtime_error( TString::Format("Vector type observables cannot be used as entry specific weights for multidimensional histograms / profiles unless synchronized filling mode is chosen. Please verify your additional options for histogram '%s', weight observable '%s'",fHistograms[i]->GetName(),fWeightObservables[i]?fWeightObservables[i]->GetName():"").Data());
        }
        TRY(
        for (int a = 0; a<fObservables[i][0]->getNevaluations(); a++) {
        for (int b = 0; b<fObservables[i][1]->getNevaluations(); b++) {
        for (int c = 0; c<fObservables[i][2]->getNevaluations(); c++) {
          ((TProfile2D*)(fHistograms[i]))->Fill(fObservables[i][0]->getValueAt(a), 
                                                fObservables[i][1]->getValueAt(b), 
                                                fObservables[i][2]->getValueAt(c), 
                                                (this->fFillRaw[i]? 1. : weight*fSample->getNormalisation())
                                                * ( fWeightObservables[i] ? fWeightObservables[i]->getValueAt(0)  : 1. )   );
        }}}
        ,TString::Format("Failed to fill histogram '%s' using the observables '%s', '%s', '%s' at cut '%s'.", fHistograms[i]->GetName(), fObservables[i][0]->GetName(), fObservables[i][1]->GetName(), fObservables[i][2]->GetName(), this->getCut()->GetName())
        )
        
      }
      break;
    case -1:
      if (this->fFillSynchronized[i]) {
        TRY(
        nEvals = std::max(fObservables[i][0]->getNevaluations(),fObservables[i][1]->getNevaluations());
        ,TString::Format("Failed to get number of evaluations from at least one of the observables '%s', '%s' for filling histogram '%s' at cut '%s'.", fObservables[i][0]->GetName(), fObservables[i][1]->GetName(), fHistograms[i]->GetName(), this->getCut()->GetName())
        )
        if (fObservables[i][0]->getNevaluations() != nEvals &&  fObservables[i][0]->getNevaluations() != 1) throw std::runtime_error("Cannot perform synchonized histogram filling: Number of evaluations do not match!");
        if (fObservables[i][1]->getNevaluations() != nEvals &&  fObservables[i][1]->getNevaluations() != 1) throw std::runtime_error("Cannot perform synchonized histogram filling: Number of evaluations do not match!");
        TRY(
        nWeightEvals = fWeightObservables[i] ? fWeightObservables[i]->getNevaluations() : 1;
        ,TString::Format("Failed to get number of evaluations from histogram-entry-weight observable '%s' for filling histogram '%s' at cut '%s'", fWeightObservables[i]?fWeightObservables[i]->GetName():"", fHistograms[i]->GetName(), this->getCut()->GetName())
        )
        if (nWeightEvals != 1 && nWeightEvals != nEvals) {
          throw std::runtime_error( TString::Format("Histogram specific weight is neither scalar nor does the number of evaluations of the responsible observable (%d) match the number of evaluations of the value observable(s) (%d). Please check the definition of histogram '%s' with weight observable '%s'! ",nWeightEvals,nEvals,fHistograms[i]->GetName(),fWeightObservables[i]?fWeightObservables[i]->GetName():"").Data());
        }
        
        TRY(
        for (int a = 0; a<nEvals; a++) {
          ((TProfile*)(fHistograms[i]))->Fill(fObservables[i][0]->getValueAt(fObservables[i][0]->getNevaluations()==1?0:a), 
                                              fObservables[i][1]->getValueAt(fObservables[i][1]->getNevaluations()==1?0:a), 
                                              ( this->fFillRaw[i]? 1. : weight*fSample->getNormalisation() )
                                              * ( fWeightObservables[i]?fWeightObservables[i]->getValueAt(std::min(a,std::max(0,nWeightEvals-1)) ) : 1.) );
        }
        ,TString::Format("Failed to fill histogram '%s' using the observables '%s', '%s' at cut '%s'.", fHistograms[i]->GetName(), fObservables[i][0]->GetName(), fObservables[i][1]->GetName(), this->getCut()->GetName())
        )
      } else { //filling every combination of entries provided by the different (vector) observables
        TRY(
        nWeightEvals = fWeightObservables[i] ? fWeightObservables[i]->getNevaluations() : 1;
        ,TString::Format("Failed to get number of evaluations from histogram-entry-weight observable '%s' for filling histogram '%s' at cut '%s'", fWeightObservables[i]?fWeightObservables[i]->GetName():"", fHistograms[i]->GetName(), this->getCut()->GetName())
        )
        if (nWeightEvals != 1) {
          throw std::runtime_error( TString::Format("Vector type observables cannot be used as entry specific weights for multidimensional histograms / profiles unless synchronized filling mode is chosen. Please verify your additional options for histogram '%s', weight observable '%s'",fHistograms[i]->GetName(),fWeightObservables[i]?fWeightObservables[i]->GetName():"").Data());
        }
        TRY(
        for (int a = 0; a<fObservables[i][0]->getNevaluations(); a++) {
        for (int b = 0; b<fObservables[i][1]->getNevaluations(); b++) {
          ((TProfile*)(fHistograms[i]))->Fill(fObservables[i][0]->getValueAt(a), 
                                                fObservables[i][1]->getValueAt(b), 
                                                ( this->fFillRaw[i]? 1. : weight*fSample->getNormalisation() )
                                                * ( fWeightObservables[i] ? fWeightObservables[i]->getValueAt(0)  : 1. )   );
        }}
        ,TString::Format("Failed to fill histogram '%s' using the observables '%s', '%s' at cut '%s'.", fHistograms[i]->GetName(), fObservables[i][0]->GetName(), fObservables[i][1]->GetName(), this->getCut()->GetName())
        )
      }
      break;
    case 1:
      {
        // DEBUGclass("found 1D-histogram");
#ifdef _DEBUG_
        // a set protections only active during debugging
        if(!this->fHistograms[i]){
          throw std::runtime_error("histogram slot is empty!");
        } 
        if(this->fObservables[i].size() < 1){
          throw std::runtime_error("no observable found for this histogram!");
        } else if(!this->fObservables[i][0]){
          throw std::runtime_error("observable slot for histogram is empty!");
        } else if(fObservables[i].size() != 1){
          throw std::runtime_error("wrong number of observables for one-dimensional histogram!");
        }
#endif
        DEBUGclass("evaluating observable at %p",fObservables[i][0]);
        DEBUGclass("observable is '%s' of type '%s'",fObservables[i][0]->GetName(),fObservables[i][0]->ClassName());
        //        try {
        TRY(
        nEvals = fObservables[i][0]->getNevaluations();
        ,TString::Format("Failed to get number of evaluations from observable '%s' for filling histogram '%s' at cut '%s'.", fObservables[i][0]->GetName(), fHistograms[i]->GetName(), this->getCut()->GetName())
        )
        TRY(
        nWeightEvals = fWeightObservables[i] ? fWeightObservables[i]->getNevaluations() : 1;
        ,TString::Format("Failed to get number of evaluations from histogram-entry-weight observable '%s' for filling histogram '%s' at cut '%s'", fWeightObservables[i]?fWeightObservables[i]->GetName():"", fHistograms[i]->GetName(), this->getCut()->GetName())
        )
        if (nWeightEvals != 1 && nWeightEvals != nEvals) {
          throw std::runtime_error( TString::Format("Histogram specific weight is neither scalar nor does the number of evaluations of the responsible observable (%d) match the number of evaluations of the value observable(s) (%d). Please check the definition of histogram '%s' with weight observable '%s'! ",nWeightEvals,nEvals,fHistograms[i]->GetName(),fWeightObservables[i]?fWeightObservables[i]->GetName():"").Data());
        }
        
          TRY(
          for (int a=0; a<fObservables[i][0]->getNevaluations(); a++) {
            const double val = fObservables[i][0]->getValueAt(a);
            DEBUGclass("done evaluating");
            DEBUGclass("filling histogram '%s' from '%s' with value %f %s",fHistograms[i]->GetName(),fObservables[i][0]->getActiveExpression().Data(),val, (fWeightObservables[i]? TString::Format("and individual weight %f",fWeightObservables[i]->getValueAt( std::min(a,std::max(0,nWeightEvals-1)) )).Data() : "") );
            fHistograms[i]->Fill(val, 
                                  ( this->fFillRaw[i]? 1. : weight*fSample->getNormalisation() ) * ( fWeightObservables[i]?fWeightObservables[i]->getValueAt(std::min(a,std::max(0,nWeightEvals-1)) ) : 1.)  ); 
          }
          ,TString::Format("Failed to fill histogram '%s' using the observable '%s' at cut '%s'.", fHistograms[i]->GetName(), fObservables[i][0]->GetName(), this->getCut()->GetName())
          )
//        } catch (const std::exception& e){
//          BREAK("ERROR in '%s': %s",fObservables[i][0]->GetName(),e.what());
//        }
      }
      break;
    case 2:
    if (this->fFillSynchronized[i]) {
        TRY(
        nEvals = std::max(fObservables[i][0]->getNevaluations(),fObservables[i][1]->getNevaluations());
        ,TString::Format("Failed to get number of evaluations from at least one of the observables '%s', '%s' for filling histogram '%s' at cut '%s'.", fObservables[i][0]->GetName(), fObservables[i][1]->GetName(), fHistograms[i]->GetName(), this->getCut()->GetName())
        )
        if (fObservables[i][0]->getNevaluations() != nEvals &&  fObservables[i][0]->getNevaluations() != 1) throw std::runtime_error("Cannot perform synchonized histogram filling: Number of evaluations do not match!");
        if (fObservables[i][1]->getNevaluations() != nEvals &&  fObservables[i][1]->getNevaluations() != 1) throw std::runtime_error("Cannot perform synchonized histogram filling: Number of evaluations do not match!");
        TRY(
        nWeightEvals = fWeightObservables[i] ? fWeightObservables[i]->getNevaluations() : 1;
        ,TString::Format("Failed to get number of evaluations from histogram-entry-weight observable '%s' for filling histogram '%s' at cut '%s'", fWeightObservables[i]?fWeightObservables[i]->GetName():"", fHistograms[i]->GetName(), this->getCut()->GetName())
        )
        if (nWeightEvals != 1 && nWeightEvals != nEvals) {
          throw std::runtime_error( TString::Format("Histogram specific weight is neither scalar nor does the number of evaluations of the responsible observable (%d) match the number of evaluations of the value observable(s) (%d). Please check the definition of histogram '%s' with weight observable '%s'! ",nWeightEvals,nEvals,fHistograms[i]->GetName(),fWeightObservables[i]?fWeightObservables[i]->GetName():"").Data());
        }
        
        TRY(
        for (int a = 0; a<nEvals; a++) {
          ((TH2*)(fHistograms[i]))->Fill(fObservables[i][0]->getValueAt(fObservables[i][0]->getNevaluations()==1?0:a), 
                                              fObservables[i][1]->getValueAt(fObservables[i][1]->getNevaluations()==1?0:a), 
                                              ( this->fFillRaw[i]? 1. : weight*fSample->getNormalisation() )
                                              * ( fWeightObservables[i]?fWeightObservables[i]->getValueAt(std::min(a,std::max(0,nWeightEvals-1)) ) : 1.) );
        }
        ,TString::Format("Failed to fill histogram '%s' using the observables '%s', '%s' at cut '%s'.", fHistograms[i]->GetName(), fObservables[i][0]->GetName(), fObservables[i][1]->GetName(), this->getCut()->GetName())
        )
      } else { //filling every combination of entries provided by the different (vector) observables
        TRY(
        nWeightEvals = fWeightObservables[i] ? fWeightObservables[i]->getNevaluations() : 1;
        ,TString::Format("Failed to get number of evaluations from histogram-entry-weight observable '%s' for filling histogram '%s' at cut '%s'", fWeightObservables[i]?fWeightObservables[i]->GetName():"", fHistograms[i]->GetName(), this->getCut()->GetName())
        )
        if (nWeightEvals != 1) {
          throw std::runtime_error( TString::Format("Vector type observables cannot be used as entry specific weights for multidimensional histograms / profiles unless synchronized filling mode is chosen. Please verify your additional options for histogram '%s', weight observable '%s'",fHistograms[i]->GetName(),fWeightObservables[i]?fWeightObservables[i]->GetName():"").Data());
        }
        TRY(
        for (int a = 0; a<fObservables[i][0]->getNevaluations(); a++) {
        for (int b = 0; b<fObservables[i][1]->getNevaluations(); b++) {
          ((TH2*)(fHistograms[i]))->Fill(fObservables[i][0]->getValueAt(a), 
                                                fObservables[i][1]->getValueAt(b), 
                                                ( this->fFillRaw[i]? 1. : weight*fSample->getNormalisation() )
                                                * ( fWeightObservables[i] ? fWeightObservables[i]->getValueAt(0)  : 1. )   );
        }}
        ,TString::Format("Failed to fill histogram '%s' using the observables '%s', '%s' at cut '%s'.", fHistograms[i]->GetName(), fObservables[i][0]->GetName(), fObservables[i][1]->GetName(), this->getCut()->GetName())
        )
      }
      break;
    case 3:
      if (this->fFillSynchronized[i]) { 
        TRY(
        nEvals = std::max(std::max(fObservables[i][0]->getNevaluations(),fObservables[i][1]->getNevaluations()),fObservables[i][2]->getNevaluations());
        ,TString::Format("Failed to get number of evaluations from at least one of the observables '%s', '%s', '%s' for filling histogram '%s' at cut '%s'.", fObservables[i][0]->GetName(), fObservables[i][1]->GetName(), fObservables[i][2]->GetName(), fHistograms[i]->GetName(), this->getCut()->GetName())
        )
        if (fObservables[i][0]->getNevaluations() != nEvals &&  fObservables[i][0]->getNevaluations() != 1) throw std::runtime_error("Cannot perform synchonized histogram filling: Number of evaluations do not match!");
        if (fObservables[i][1]->getNevaluations() != nEvals &&  fObservables[i][1]->getNevaluations() != 1) throw std::runtime_error("Cannot perform synchonized histogram filling: Number of evaluations do not match!");
        if (fObservables[i][2]->getNevaluations() != nEvals &&  fObservables[i][2]->getNevaluations() != 1) throw std::runtime_error("Cannot perform synchonized histogram filling: Number of evaluations do not match!");
        TRY(
        nWeightEvals = fWeightObservables[i] ? fWeightObservables[i]->getNevaluations() : 1;
        ,TString::Format("Failed to get number of evaluations from histogram-entry-weight observable '%s' for filling histogram '%s' at cut '%s'", fWeightObservables[i]?fWeightObservables[i]->GetName():"", fHistograms[i]->GetName(), this->getCut()->GetName())
        )
        if (nWeightEvals != 1 && nWeightEvals != nEvals) {
          throw std::runtime_error( TString::Format("Histogram specific weight is neither scalar nor does the number of evaluations of the responsible observable (%d) match the number of evaluations of the value observable(s) (%d). Please check the definition of histogram '%s' with weight observable '%s'! ",nWeightEvals,nEvals,fHistograms[i]->GetName(),fWeightObservables[i]?fWeightObservables[i]->GetName():"").Data());
        }
        
        TRY(
        for (int a = 0; a<nEvals; ++a) {
          ((TH3*)(fHistograms[i]))->Fill(fObservables[i][0]->getValueAt(fObservables[i][0]->getNevaluations()==1?0:a), 
                                                fObservables[i][1]->getValueAt(fObservables[i][1]->getNevaluations()==1?0:a), 
                                                fObservables[i][2]->getValueAt(fObservables[i][2]->getNevaluations()==1?0:a), 
                                                (this->fFillRaw[i]? 1. : weight*fSample->getNormalisation() )
                                                * ( fWeightObservables[i]?fWeightObservables[i]->getValueAt(std::min(a,std::max(0,nWeightEvals-1)) ) : 1.) );
        }
        ,TString::Format("Failed to fill histogram '%s' using the observables '%s', '%s', '%s' at cut '%s'.", fHistograms[i]->GetName(), fObservables[i][0]->GetName(), fObservables[i][1]->GetName(), fObservables[i][2]->GetName(), this->getCut()->GetName())
        )
      } else { //filling every combination of entries provided by the different (vector) observables
        TRY(
        nWeightEvals = fWeightObservables[i] ? fWeightObservables[i]->getNevaluations() : 1;
        ,TString::Format("Failed to get number of evaluations from histogram-entry-weight observable '%s' for filling histogram '%s' at cut '%s'", fWeightObservables[i]?fWeightObservables[i]->GetName():"", fHistograms[i]->GetName(), this->getCut()->GetName())
        )
        if (nWeightEvals != 1) {
          throw std::runtime_error( TString::Format("Vector type observables cannot be used as entry specific weights for multidimensional histograms / profiles unless synchronized filling mode is chosen. Please verify your additional options for histogram '%s', weight observable '%s'",fHistograms[i]->GetName(),fWeightObservables[i]?fWeightObservables[i]->GetName():"").Data());
        }
        TRY(
        for (int a = 0; a<fObservables[i][0]->getNevaluations(); ++a) {
        for (int b = 0; b<fObservables[i][1]->getNevaluations(); ++b) {
        for (int c = 0; c<fObservables[i][2]->getNevaluations(); ++c) {
          ((TH3*)(fHistograms[i]))->Fill(fObservables[i][0]->getValueAt(a), 
                                                fObservables[i][1]->getValueAt(b), 
                                                fObservables[i][2]->getValueAt(c), 
                                                ( this->fFillRaw[i]? 1. : weight*fSample->getNormalisation() )
                                                * ( fWeightObservables[i] ? fWeightObservables[i]->getValueAt(0)  : 1. )  );
        }}}
        ,TString::Format("Failed to fill histogram '%s' using the observables '%s', '%s', '%s' at cut '%s'.", fHistograms[i]->GetName(), fObservables[i][0]->GetName(), fObservables[i][1]->GetName(), fObservables[i][2]->GetName(), this->getCut()->GetName())
        )
      }
    } 
  }
  return true;
}

//__________________________________________________________________________________|___________

TObjArray * TQHistoMakerAnalysisJob::getBranchNames() {
  // retrieve the list of branch names used by this job
  if(!this->fSample){
    throw std::runtime_error("cannot retrieve branches on uninitialized object!");
  }
  TObjArray * bNames = new TObjArray();
 
  /* return all observable expressions (containing branch names) */
  for (size_t i = 0; i < fObservables.size(); ++i) {
    for (size_t j = 0; j < fObservables[i].size(); ++j) {
      TQObservable* obs = fObservables[i][j];
      if(obs){
        TCollection* c = obs->getBranchNames();
        if(c){
          if(c->GetEntries() > 0) bNames -> AddAll(c);
          c->SetOwner(false);
          delete c;
        }
      }
    }
  }
 
  return bNames;
}

//__________________________________________________________________________________|___________

bool TQHistoMakerAnalysisJob::initializeSelf() {
  // initialize this analysis job
  DEBUGclass("initializing analysis job '%s'",this->GetName());

  if(fHistograms.size() < 1){
    this->poolAt = this->fSample;
    DEBUGclass("initializing histograms");
    this->initializeHistograms();
  }

  bool success = true;
  /* initialize TQObservables */
  DEBUGclass("initializing observables");
  for (unsigned int i = 0; i < fExpressions.size(); ++i) {
    std::vector<TQObservable*> observables;
    for (unsigned int j = 0; j < fExpressions[i].size(); ++j) {
      TQObservable* obs = TQObservable::getObservable(fExpressions[i][j],this->fSample);
      if(obs && success){
        DEBUGclass("initializing...");
        if (!obs->initialize(this->fSample)) {
          ERRORclass("Failed to initialize observable created from expression '%s' for sample '%s' in TQHistomakerAnalysisJob '%s' for histogram named '%s'",this->fExpressions[i][j].Data(), this->fSample->getPath().Data(), this->GetName(), this->fHistograms[i]->GetName());
          success=false;
        }
        DEBUGclass("initialized observable '%s' of type '%s' with '%s'",
                   obs->getExpression().Data(),
                   obs->ClassName(),
                   obs->getActiveExpression().Data());
      }
      if(!obs){//can this really happen? TQObservable::getObservable should always fall back to an instance of TQTreeFormulaObservable created via a factory (this might not be correct, but at least obs should never be a null-pointer)
        DEBUGclass("creating const observable");
        obs = TQObservable::getObservable("Const:nan",this->fSample);
        obs->initialize(this->fSample);
      }
      observables.push_back(obs);
    }
    this->fObservables.push_back(observables);
  }
  //get observables for individual weights if needed
  for (unsigned  int i=0; i< fWeightExpressions.size(); ++i) {
    if (fWeightExpressions[i].Length()<1) {
      fWeightObservables.push_back(NULL);
      continue;
    }
    
    TQObservable* wObs = TQObservable::getObservable(fWeightExpressions[i],this->fSample);
      if(wObs){
        DEBUGclass("initializing...");
        if (!wObs->initialize(this->fSample)) {
          ERRORclass("Failed to initialize weight observable created from expression '%s' for sample '%s' in TQHistomakerAnalysisJob '%s' for histogram named '%s'",this->fWeightExpressions[i].Data(), this->fSample->getPath().Data(), this->GetName(), this->fHistograms[i]->GetName());
          success=false;
        }
        DEBUGclass("initialized observable '%s' of type '%s' with '%s'",
                   wObs->getExpression().Data(),
                   wObs->ClassName(),
                   wObs->getActiveExpression().Data());
      }
      if(!wObs){//can this really happen? TQObservable::getObservable should always fall back to an instance of TQTreeFormulaObservable created via a factory (this might not be correct, but at least obs should never be a null-pointer)
        DEBUGclass("creating const observable");
        wObs = TQObservable::getObservable("Const:nan",this->fSample);
        wObs->initialize(this->fSample);
      }
      fWeightObservables.push_back(wObs);
  
  }
  DEBUG("successfully initialized histogram job");
  return success;
}

//__________________________________________________________________________________|___________

void TQHistoMakerAnalysisJob::initializeHistograms(){
  // create histograms from templates */
  DEBUGclass("Size of histogram template vector : %i", fHistogramTemplates.size());
  for (unsigned int i = 0; i < fHistogramTemplates.size(); i++) {
    
    /* copy/clone the template histogram */
    //TH1 * histo = (TH1*)(*fHistogramTemplates)[i]->Clone();
    TH1 * histo = TQHistogramUtils::copyHistogram((fHistogramTemplates)[i]);
    // std::cout << "initialized " << histo->GetName() << std::endl;
    histo->SetDirectory(0);
    fHistograms.push_back(histo);

  }
}
 

//__________________________________________________________________________________|___________

bool TQHistoMakerAnalysisJob::finalizeSelf() {
  // finalize TQObservables
  for (unsigned int i = 0; i < fObservables.size(); ++i) {
    for (unsigned int j = 0; j < fObservables[i].size(); ++j) {
      fObservables[i][j]->finalize();
    }
  }
  this->fObservables.clear();
  for (unsigned int i = 0; i < fWeightObservables.size(); ++i) {
      if (fWeightObservables[i]) fWeightObservables[i]->finalize(); //weight observables are optional and might be just dummy entries (i.e. null pointers)!
  }
  this->fWeightObservables.clear();

  if(this->poolAt == this->fSample)
    if(!this->finalizeHistograms())
      return false;
  
  return true;
}

//__________________________________________________________________________________|___________

bool TQHistoMakerAnalysisJob::finalizeHistograms(){
  // store the histograms in the sample folder hierarchy
  if (!getCut()) { return false; }

  /* get the histogram folder */
  TQFolder * folder = this->poolAt->getFolder(TString::Format(
                                                              ".histograms/%s+", getCut()->GetName()));
  if (!folder) { return false; }
  DEBUGclass("successfully created folder for cut %s", getCut()->GetName());

  /* scale and store histograms */
  DEBUGclass("length of histogram list is %i", fHistograms.size());
  for (unsigned int i = 0; i < fHistograms.size(); i++) {
    TH1 * histo = (fHistograms)[i];
    if (!histo){ DEBUGclass("Histogram is 0!"); };
    /* delete existing histogram */
    TObject *h = folder->FindObject(histo->GetName());
    if (h)
      {
        DEBUGclass("removing previous object %s", h->GetName());
        folder->Remove(h);
      }
    /* save the new histogram */
    DEBUGclass("saving histogram %s", histo->GetName());
    folder->Add(histo);
  }

  /* delete the list of histograms */
  this->fHistograms.clear();
  this->poolAt = NULL;

  return true;
}

//__________________________________________________________________________________|___________

bool TQHistoMakerAnalysisJob::initializeSampleFolder(TQSampleFolder* sf){
  // initialize this job on a sample folder (taking care of pooling)
  bool pool = false;
  sf->getTagBool(".aj.pool.histograms",pool);
  // std::cout << std::endl << "initialize samplefolder called on " << sf->GetName() << " pool=" << pool << ", fHistograms=" << fHistograms << std::endl << std::endl;

  if(pool && (this->fHistograms.size() == 0)){
    this->initializeHistograms();
    this->poolAt = sf;
  }

  return true;
}

//__________________________________________________________________________________|___________

bool TQHistoMakerAnalysisJob::finalizeSampleFolder(TQSampleFolder* sf){
  // finalize this job on a sample folder (taking care of pooling)
  if(sf == this->poolAt)
    return this->finalizeHistograms();
  return true;
}

//__________________________________________________________________________________|___________

TQHistoMakerAnalysisJob::~TQHistoMakerAnalysisJob() {
  // destructor
  for (unsigned int i = 0; i < fHistogramTemplates.size(); i++) {
    delete (fHistogramTemplates)[i]; }
}

//__________________________________________________________________________________|___________

TQAnalysisJob* TQHistoMakerAnalysisJob::getClone(){
  // retrieve a clone of this job
  TQHistoMakerAnalysisJob* newJob = new TQHistoMakerAnalysisJob(this);
  return newJob;
}

//__________________________________________________________________________________|___________

void TQHistoMakerAnalysisJob::reset() {
  // Reset this analysis job. This method is called after an analysis job was
  // cloned.

  // call the reset function of the parent class
  TQAnalysisJob::reset();
  // do class-specific stuff
  fHistograms.clear();
}

//__________________________________________________________________________________|___________

int TQHistoMakerAnalysisJob::importJobsFromTextFiles(const TString& files, TQCompiledCut* basecut, const TString& channelFilter, bool verbose){
  // open a list of files (comma-separated), parse all histogram definitions inside
  // for each assigment encountered matching the channelfilter (or with no channel),
  // create a histogram job, fill it with all appropriate histograms and add it to the basecut
  std::vector <TString> filenames = TQStringUtils::split(files,",");
  return TQHistoMakerAnalysisJob::importJobsFromTextFiles(filenames,basecut,channelFilter,verbose);
}

//__________________________________________________________________________________|___________

int TQHistoMakerAnalysisJob::importJobsFromTextFiles(const TString& files, TQCompiledCut* basecut, TQTaggable* aliases, const TString& channelFilter, bool verbose){
  // open a list of files (comma-separated), parse all histogram definitions inside
  // for each assigment encountered matching the channelfilter (or with no channel),
  // create a histogram job, fill it with all appropriate histograms and add it to the basecut
  std::vector <TString> filenames = TQStringUtils::split(files,",");
  return TQHistoMakerAnalysisJob::importJobsFromTextFiles(filenames,basecut,aliases,channelFilter,verbose);
}

//__________________________________________________________________________________|___________

int TQHistoMakerAnalysisJob::importJobsFromTextFiles(const std::vector<TString>& filenames, TQCompiledCut* basecut, const TString& channelFilter, bool verbose){
  // open a list of files (std::vector), parse all histogram definitions inside
  // for each assigment encountered matching the channelfilter (or with no channel),
  // create a histogram job, fill it with all appropriate histograms and add it to the basecut
  return TQHistoMakerAnalysisJob::importJobsFromTextFiles(filenames, basecut, NULL, channelFilter, verbose);
}

//__________________________________________________________________________________|___________

int TQHistoMakerAnalysisJob::importJobsFromTextFiles(const std::vector<TString>& filenames, TQCompiledCut* basecut, TQTaggable* aliases, const TString& channelFilter, bool verbose){
  // open a list of files (std::vector), parse all histogram definitions inside
  // for each assigment encountered matching the channelfilter (or with no channel),
  // create a histogram job, fill it with all appropriate histograms and add it to the basecut
  if(filenames.size() < 1){
    ERRORfunc("importing no histograms from empty files list!");
    return -1;
  }
  std::map<TString,TString> histogramDefinitions;
  std::vector<TString> assignments;
  TString buffer;
  for(size_t i=0; i<filenames.size(); i++){
    std::vector<TString>* lines = TQStringUtils::readFileLines(filenames[i],2048);
    if(!lines){
      ERRORfunc("unable to open file '%s'",filenames[i].Data());
      continue;
    }
    for(size_t j=0; j<lines->size(); ++j){
      TString line(TQStringUtils::trim(lines->at(j)));
      DEBUGclass("looking at line '%s'",line.Data());
      if(line.IsNull()) continue;
      if(line.BeginsWith("T")){
        size_t namestart = TQStringUtils::findFirstOf(line,"'\"",0)+1;
        size_t nameend = TQStringUtils::findFirstOf(line,"'\"",namestart);
        if(namestart == 0 || namestart > (size_t)line.Length() || nameend > (size_t)line.Length() || nameend == namestart){
          ERRORfunc("unable to parse histogram definition '%s'",line.Data());
          continue;
        }
        TString name(TQStringUtils::trim(line(namestart,nameend-namestart),"\t ,"));
        DEBUGclass("found definition: '%s', assigning as '%s'",line.Data(),name.Data());
        histogramDefinitions[name] = line;
      } else if(TQStringUtils::removeLeading(line,"@") == 1){ 
        DEBUGclass("found assignment: '%s'",line.Data());
        assignments.push_back(line);
      } else {
        WARNfunc("encountered unknown token: '%s'",line.Data());
      }
    }
    delete lines;
  }
  if(verbose) VERBOSEfunc("going to create '%d' jobs",(int)(assignments.size()));
  int retval = 0;
  for(size_t i=0; i<assignments.size(); i++){
    TString assignment = assignments[i];
    DEBUGclass("looking at assignment '%s'",assignment.Data());
    TString channel;
    if(TQStringUtils::readBlock(assignment,channel) && !channel.IsNull() && !TQStringUtils::matches(channel,channelFilter)) continue;
    TString cuts,histograms;
    TQStringUtils::readUpTo(assignment,cuts,":");
    TQStringUtils::readToken(assignment,buffer," :");
    TQStringUtils::readUpTo(assignment,histograms,";");
    TQStringUtils::readToken(assignment,buffer,"; ");
    DEBUGclass("histograms: '%s'",histograms.Data());
    DEBUGclass("cuts: '%s'",cuts.Data());
    if(verbose) VERBOSEfunc("building job for cuts '%s'",cuts.Data());
    DEBUGclass("spare symbols: '%s'",buffer.Data());
    std::vector<TString> vHistos = TQStringUtils::split(histograms,",");
    if(vHistos.size() < 1){
      ERRORfunc("no histograms listed in assignment '%s'",assignments[i].Data());
      continue;
    }
    TQHistoMakerAnalysisJob* job = new TQHistoMakerAnalysisJob();
    for(size_t j=0; j<vHistos.size(); ++j){
      const TString def = histogramDefinitions[TQStringUtils::trim(vHistos[j],"\t ,")];
      if(def.IsNull()){
        ERRORfunc("unable to find histogram definition for name '%s', skipping",TQStringUtils::trim(vHistos[j],"\t ,").Data());
        continue;
      }
      bool ok = job->bookHistogram(def,aliases);
      if(ok){
        if(verbose) VERBOSEfunc("\tbooked histogram '%s'",def.Data());
      } else {
        retval += 1;
        if(verbose) std::cout << f_ErrMsg.getMessages() << std::endl;
        DEBUGclass("error booking histogram for '%s', function says '%s'",def.Data(),f_ErrMsg.getMessages().Data());
      }
    }
    if(verbose) job->printBooking(cuts);
    basecut->addAnalysisJob(job,cuts);
    delete job;
  } 

  DEBUGclass("end of function call, encountered %d error messages",retval);
  return retval;
}
