#include "QFramework/TQSampleDataReader.h"
#include "QFramework/TQHistogramUtils.h"
#include "QFramework/TQTHnBaseUtils.h"
#include "QFramework/TQStringUtils.h"
#include "QFramework/TQSampleFolder.h"
#include "QFramework/TQFolder.h"
#include "QFramework/TQCounter.h"
#include "QFramework/TQTaggable.h"
#include "QFramework/TQIterator.h"
#include "QFramework/TQUtils.h"
#include "QFramework/TQListUtils.h"
#include "QFramework/TQValue.h"
#include "QFramework/TQPCA.h"
#include "QFramework/TQCut.h"

#include "THashList.h"
#include "TMap.h"
#include "TTree.h"
#include "TList.h"
#include "TProfile.h"
#include "TProfile2D.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TGraph.h"
#include "TGraph2D.h"
#include "THnBase.h"
#include "THnSparse.h"
#include "TMath.h"
#include "TFile.h"

// #define _DEBUG_
#include "QFramework/TQLibrary.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>


////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQSampleDataReader:
//
// The TQSampleDataReader class provides methods to retrieve analysis results (histograms,
// cutflow counters, ...) from a structure of instances of the TQSampleFolder class. Generally,
// those elements are obtained by summing individual contributions in sample folders recursively.
//
// Retrieving histograms (in subfolders ".histograms"):
//
// - TQSampleDataReader::getHistogram("<path>", "<cutName/histogram>", ...) returning TH1*
//
// Retrieving cutflow counter (in subfolders ".cutflow"):
//
// - TQSampleDataReader::getCounter("<path>", "<cutName>", ...) returning TQCounter*
//
//
//
//
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQSampleDataReader)


//__________________________________________________________________________________|___________

TQSampleDataReader::TQSampleDataReader():
f_errMsg(new std::stringstream(),true,true)
{
  // default constructor
  reset();
}


//__________________________________________________________________________________|___________

TQSampleDataReader::TQSampleDataReader(TQSampleFolder * sampleFolder):
  f_errMsg(new std::stringstream(),true,true)
{
  // constructor using a TQSamplefolder
  reset();
  f_baseSampleFolder = sampleFolder;
}

//__________________________________________________________________________________|___________

TQSampleFolder* TQSampleDataReader::getSampleFolder(){
  // retrieve the sample folder
  return f_baseSampleFolder;
}

//__________________________________________________________________________________|___________

void TQSampleDataReader::reset() {
  // reset this TQSampleDataReader
  f_baseSampleFolder = 0;
  f_localMode = false;
  f_Verbose = 0;
  f_errMsg.clearMessages();

  // reset the schemes
  f_styleScheme = "default";
  f_filterScheme = "";
  f_normScheme = "";
  f_scaleScheme = ".default";

  // reset object paths
  f_pathHistograms = ".histograms";
  f_pathGraphs = ".graphs";
  f_pathGridScan = ".gridscan";
  f_pathCutflow = ".cutflow";
  f_pathEventlists = ".eventlists";
  f_pathScaleFactors = ".scalefactors";
  f_pathPCA = ".pca";
  f_pathTrees = ".trees";
}


//__________________________________________________________________________________|___________

void TQSampleDataReader::setLocalMode(bool localMode) {
  // Enables (<localMode> == true) or disables (<localMode> == false) the 'local
  // mode' of this instance of TQSampleDataReader (local mode is disabled by default).
  // In local mode, scale factors (better: normalization factors) or ignored for
  // sample folders starting from the base sample folder of this instance of
  // TQSampleDataReader up to the root sample folder. If the local mode is disabled
  // scale factors assigned to any sample folder up the sample folder tree are
  // accounted for. Examples:
  //
  // Let <samples> refer to a sample folder hierarchy with the following structure
  // samples/
  // bkg/
  // top/
  // WW/
  // sig/
  // and let the sample folder "bkg" have a scale factor of 2. assigned to it (at
  // some object folder). Defining two different instances of TQSampleDataReader
  //
  // TQSampleDataReader rd1(samples);
  // TQSampleDataReader rd2(samples->getSampleFolder("bkg"));
  //
  // both rd1.getHistogram(...) and rd2.getHistogram(...) will account for the scale
  // factor at sample folder "bkg" (provided the requested object is stored in
  // the corresponding object folder). However, in local mode, set by
  //
  // rd1.setLocalMode(true);
  // rd2.setLocalMode(true);
  //
  // only rd1.getHistogram(...) will account for the scale factor at sample folder
  // "bkg" (again, provided the requested object is stored in the corresponding object
  // folder), while rd2.getHistogram(...) will ignore this scale factor.
  //
  // Please note: the local mode does equivalently affect the retrieval of counter
  // and other scaleable objects.

  f_localMode = localMode;
}


//__________________________________________________________________________________|___________

bool TQSampleDataReader::getLocalMode() {
  // Returns true if this instance of TQSampleDataReader is in local mode and false
  // otherwise.

  return f_localMode;
}


//__________________________________________________________________________________|___________

void TQSampleDataReader::setVerbose(int verbose) {
  // Sets the verbosity of this instance of TQSampleDataReader to level <verbose>.
  // Verbosity levels are:
  // - 0: No output on standard out at all (default)
  // - 1: Print an error message on standard out if an operation fails
  // - 2: Print info for main internal function calls
  // - 3: Print info for nternal function calls
  // - 4: Trace detailed way through structure of sample folders

  f_Verbose = verbose;
}


//__________________________________________________________________________________|___________

int TQSampleDataReader::getVerbose() {
  // Returns the level of verbosity of this instance of TQSampleDataReader (see
  // documentation of TQSampleDataReader::setVerbose(...) for additional information
  // on verbosity levels).
 
  return f_Verbose;
}


//__________________________________________________________________________________|___________

void TQSampleDataReader::setErrorMessage(const TString& fname, const TString& message) {
 
  // update the error message
  f_errMsg.sendClassFunctionMessage(TQMessageStream::ERROR,this->Class(),fname,message);
 
  // print the error message if in verbose mode
  if (f_Verbose > 0) {
    TQLibrary::msgStream.sendClassFunctionMessage(TQMessageStream::ERROR,this->Class(),fname,message);
  }
}


//__________________________________________________________________________________|___________

TString TQSampleDataReader::getErrorMessage() {
  // Return the last error message
  return f_errMsg.getMessages();
}


//__________________________________________________________________________________|___________

TList * TQSampleDataReader::parsePaths(TString paths, TList * inputTokens, TString pathPrefix) {
  DEBUGclass("called with '%s'",paths.Data());
  paths.ReplaceAll(",",";");
  if(paths.Contains("$")){
    DEBUGclass("path '%s' contains unresolved tags, unable to parse!",paths.Data());
    return NULL;
  }

  // clear error message
  f_errMsg.clearMessages();

  // the list of alternative paths
  TList * tokens = inputTokens;
  if (!tokens) {
    tokens = new TList();
    tokens->SetOwner(true);
  }

  // add the first alternative
  TList * subList = new TList();
  subList->SetOwner(true);
  tokens->Add(subList);

  bool firstPath = true;
  bool done = false;
  bool error = false;

  // read paths tokens
  do {
    // the number of characters read (excluding blanks)
    int nCharsRead = 0;
    // read operator (get rid of leading blanks before)
    TString thisOperator;
    DEBUGclass("remaining text: '%s'",paths.Data());
    TQStringUtils::readBlanks(paths);
    nCharsRead += TQStringUtils::readToken(paths, thisOperator, "+-");
    TString operatorAppendix = paths;

    // read paths token (get rid of leading blanks before)
    TString thisPath;
    TString pathBuffer;
    TQStringUtils::readBlanks(paths);
    int nNewChars = 0;

    // evaluate operator token
    double factor = 1.;
    double statCor = 1.;
    
    // read any leading multipliers, e.g. "2*/path/to/my/whatever"
    int endidx = TQStringUtils::findFirstNotOf(paths,TQStringUtils::numerals+". ");
    if(endidx > 0 && endidx < paths.Length() && paths[endidx] == '*'){
      TQStringUtils::readToken(paths,pathBuffer,TQStringUtils::numerals+".");
      TQStringUtils::removeLeading(paths,"*");
      nNewChars = 0;
      factor = std::atof(pathBuffer.Data());
      pathBuffer.Clear();
      statCor = factor;
    }
    
    nNewChars += TQStringUtils::readTokenAndBlock(paths, pathBuffer,TQStringUtils::getDefaultIDCharacters() + "/*?","[]");
    TString innerBuffer = "";
    if(TQStringUtils::readUpTo(pathBuffer,thisPath,"[") < nNewChars){
      TQStringUtils::readBlock(pathBuffer,innerBuffer,"[]");
      TList* parsed = this->parsePaths(innerBuffer,NULL,"");
      if(!parsed || parsed->GetEntries() < 1){
        DEBUGclass("encountered error while parsing '%s'",innerBuffer.Data());
        return NULL;
      }
      TList* subtokens = (TList*)(parsed->At(0));
      for(int i=subtokens->GetEntries()-1; i>=0; i--){
        TString subtoken = ((TObjString*)(((TList*)subtokens->At(i))->At(1)))->GetString();
        double subop_factor = ((TQValue*)(((TList*)subtokens->At(i))->At(0)))->getDouble()*factor;
        double subop_statcor = ((TQValue*)(((TList*)subtokens->At(i))->At(2)))->getDouble()*factor; //tmp *factor
        TString subop = "+";
        if(subop_factor<0){
          if(thisOperator.BeginsWith("-"))
            subop="+";
          else
            subop="-";
        } else {
          if(thisOperator.BeginsWith("-"))
            subop="-";
          else
            subop="+";
        }
        if(subop_statcor<0 || (thisOperator.Length()>1))
          subop+=subop;
        if(fabs(subop_factor) != 1){
          subop+=TString::Format("%f*",fabs(subop_factor));
        }
        paths = subop+thisPath+subtoken+pathBuffer+paths;
      }
      delete parsed;
      continue;
    }
 
    nCharsRead+=nNewChars;
    TString pathAppendix = paths;
    // read ";" separating paths
    TString thisSep;
    TQStringUtils::readBlanks(paths);
    paths = TQStringUtils::trim(paths,"()/");
    nCharsRead += TQStringUtils::readToken(paths, thisSep, ";", 1);

    // prepend path prefix
    if (!thisPath.IsNull() && !pathPrefix.IsNull()){
      thisPath = TQFolder::concatPaths(pathPrefix, thisPath);
    }

    // we might have ran out of valid tokens in the input string
    if (nCharsRead == 0) {
      if (!paths.IsNull()) {
        // unexpected character in paths
        setErrorMessage(__FUNCTION__,TString::Format("syntax error near unexpected token '%s'", paths.Data()));
        error = true;
      } else if (firstPath) {
        // no (alternative) path at all
        if (tokens->GetEntries() > 1)
          setErrorMessage(__FUNCTION__,"alternative path expected");
        else
          setErrorMessage(__FUNCTION__,"path expected");
        error = true;
      } else {
        /* we are done if <paths> is an empty
         * string now (we read all of it) */
        done = true;
      }
      continue;
    }

    if ((thisOperator.IsNull() && firstPath) || thisOperator.CompareTo("+") == 0) {
      // add contributions
      factor *= 1.;
    } else if (thisOperator.CompareTo("-") == 0) {
      // subtract contributions
      factor *= -1.;
    } else if (thisOperator.CompareTo("--") == 0) {
      // subtract contributions with fully correlated statistical uncertainty
      factor *= -1.;
      statCor *= -1.;
    } else {
      // an unknown string instead of an operator: compile an error message
      if (thisOperator.IsNull())
        setErrorMessage(__FUNCTION__,TString::Format("operator (+, -, --) expected near '%s'", operatorAppendix.Data()));
      else
        setErrorMessage(__FUNCTION__,TString::Format("operator (+, -, --) expected near '%s'", thisOperator.Data()));
      error = true;
      continue;
    }

    // evaluate paths
    if (!thisPath.IsNull()) {
      // create a new path item
      TList * altItem = new TList();
      altItem->SetOwner(true);
      altItem->Add(TQValue::newDouble("factor", factor));
      DEBUGclass("adding '%s'",thisPath.Data());
      altItem->Add(new TObjString(thisPath.Data()));
      altItem->Add(TQValue::newDouble("statCor", statCor));
      // add the path item to the list of paths of current path alternative
      TList * pathAlt = (TList*)tokens->Last();
      pathAlt->Add(altItem);
      // expect operator preceeding next path
      firstPath = false;
    } else {
      // expected a paths but found nothing
      if (thisOperator.IsNull())
        setErrorMessage(__FUNCTION__,TString::Format("path expected near '%s'", pathAppendix.Data()));
      else
        setErrorMessage(__FUNCTION__,TString::Format("path expected after operator '%s'", thisOperator.Data()));
      error = true;
      continue;
    }

    if (thisSep.CompareTo(";") == 0) {
      // create a new alternative sub list
      TList * subList = new TList();
      subList->SetOwner(true);
      tokens->Add(subList);
      firstPath = true;
    }

  } while (!error && !done);

  // delete the whole list if an error occured
  if (error) {
    delete tokens;
    tokens = 0;
  }

  return tokens;
}

//__________________________________________________________________________________|___________

bool TQSampleDataReader::compareHistograms(const TString& histName1, const TString& histName2, const TString path, double maxdiff, bool print){
  // compare two histograms with each other via chi2 test (chi2 < maxdiff)
  // return true if they are completely equal, false otherwise
  if(!this->f_baseSampleFolder){
    ERRORclass("no sample folder assigned!");
    return false;
  }
  TString fullpath = TQFolder::concatPaths(path,this->f_pathHistograms);
  TList* histfolders = this->f_baseSampleFolder->getListOfFolders(fullpath);
  if(histfolders->GetEntries() < 1){
    WARNclass("could not find any histograms under folder matching '%s'",path.Data());
  }
  int found = 0;
  int ok = 0;
  TString fstring = TString::Format("discrepancy of X²=%%.%df detected",std::max((int)(ceil(-log10(maxdiff))),3));
  TQFolderIterator itr(histfolders,true);
  while(itr.hasNext()){
    TQFolder* f = itr.readNext();
    if(!f) continue;
    TH1* h1 = dynamic_cast<TH1*>(f->getObject(histName1));
    TH1* h2 = dynamic_cast<TH1*>(f->getObject(histName2));
    TString fpath = f->getBase()->getPath();
    if(!h1 && !h2) continue;
    if(!h1 && print){
      std::cout << TQStringUtils::fixedWidth(fpath,60, "l") << " " << TQStringUtils::makeBoldYellow(TString::Format("unable to find histogram '%s'",histName1.Data())) << std::endl;
    }
    if(!h2 && print){
      std::cout << TQStringUtils::fixedWidth(fpath,60, "l") << " " << TQStringUtils::makeBoldYellow(TString::Format("unable to find histogram '%s'",histName2.Data())) << std::endl;
    }
    if(!h1 || !h2) continue;
    found++;
    double diff = TQHistogramUtils::getChi2(h1,h2);
    if(diff < maxdiff){
      ok++;
    } else if(print){
      std::cout << TQStringUtils::fixedWidth(fpath,60, "l") << " " << TString::Format(fstring.Data(),diff) << std::endl;
    }

  }
  if(ok != found){
    if(found > 0){
      if(print) std::cout << TString::Format("found disagreement: only %d/%d histograms under '%s' were found equal!",ok,found,path.Data()) << std::endl;
    } else {
      if(print) std::cout << TQStringUtils::makeBoldRed("could not find any histograms!") << std::endl;
    }
    return false;
  }
  if(print) std::cout << TString::Format("found agreement for all %d histograms under '%s'!",ok,path.Data()) << std::endl;
  return true;
}

//__________________________________________________________________________________|___________

void TQSampleDataReader::printPaths(TString paths) {

  TList * list = parsePaths(paths);

  for (int i = 0; list && i < list->GetEntries(); i++) {
    TList * subList = (TList*)list->At(i);
    for (int j = 0; j < subList->GetEntries(); j++) {
      TList * subsubList = (TList*)subList->At(j);
      std::cout << ((TQValue*)subsubList->At(0))->getValueAsString().Data() << ": " <<
        subsubList->At(1)->GetName() << std::endl;
    }
    if (i < list->GetEntries() - 1)
      std::cout << "---------" << std::endl;
  }
}


//__________________________________________________________________________________|___________

bool TQSampleDataReader::passesFilter(TQSampleFolder * sampleFolder, TString filterName) {

  // let every sample folder pass if no filter name is specified
  if (filterName.Length() == 0)
    return true;

  // invalid sample folders don't pass the filter
  if (!sampleFolder)
    return false;

  // let every sample folder pass the folter for now
  return true;
}

//__________________________________________________________________________________|___________

TList * TQSampleDataReader::getListOfSampleFoldersTrivial(TString path, TClass* tclass) {
  // no valid base sample folder available?
  if (!f_baseSampleFolder) {
    setErrorMessage(__FUNCTION__,"no valid base sample folder specified");
    return 0;
  }
  TList* list = f_baseSampleFolder->getListOfSampleFolders(path, tclass);
  if(!list || list->GetEntries() < 1){
    setErrorMessage(__FUNCTION__,TString::Format("unknown path '%s'", path.Data()));
    return NULL;
  }
  return list;
}

//__________________________________________________________________________________|___________

TList * TQSampleDataReader::getListOfSampleFolders(const TString& path, TClass* tclass) {
  // retrieve the list of sample folders matching the given path and class

  // no valid base sample folder available?
  if (!f_baseSampleFolder) {
    setErrorMessage(__FUNCTION__,"no valid base sample folder specified");
    return 0;
  }

  TList* folders = new TList();
  TList* list = this->parsePaths(path);
  if(!list || list->GetEntries() < 1){
    DEBUGclass("unable to find paths for '%s'",path.Data());
    if(list) delete list;
    return NULL;
  }
  for (int i = 0; i < list->GetEntries(); i++) {
    TList * subList = dynamic_cast<TList*>(list->At(i));
    if(!subList) continue;
    for (int j = 0; j < subList->GetEntries(); j++) {
      TList * subsubList = dynamic_cast<TList*>(subList->At(j));
      if(!subsubList) continue;
      TString sfname = subsubList->At(1)->GetName();
      TList* sflist = f_baseSampleFolder->getListOfSampleFolders(sfname,tclass);
      if (!sflist) continue; 
      for (int k = 0; k < sflist->GetEntries(); k++) {
        TObject* obj = sflist->At(k);
        if(obj && obj->InheritsFrom(tclass))
          folders->Add(obj);
      }
      delete sflist;
    }
  }
  delete list;

  // unknown path (no matching elements)?
  if (!folders || folders->GetEntries() == 0) {
    setErrorMessage(__FUNCTION__,TString::Format("unknown path '%s'", path.Data()));
    return NULL;
  }
  return folders;
}


//__________________________________________________________________________________|___________

TH1 * TQSampleDataReader::getRatesHistogram(const TString& path, const TString& name, const TString& options, TList * sfList) {

  TQTaggable tagsOptions(options);
  return getRatesHistogram(path, name, &tagsOptions, sfList);
}


//__________________________________________________________________________________|___________

TH1 * TQSampleDataReader::getRatesHistogram(const TString& path, TString name, TQTaggable * options, TList * sfList) {
  // Returns a histogram which is created from of a list of hitograms and cutflow
  // counter with each bin of the final histogram corresponding to the integral or
  // rate of one histogram or cutflow counter . The list of names of the input
  // histograms or counters has to be provided as a comma-separated list with the
  // full list being enclosed in "{}". For each name in the list first a counter is
  // searched for and only in case of failure a histogram is searched for. Each
  // individual histogram or counter is read from path <path>. 
 
  // tokenize list of bins
  TString innerList;
  TQStringUtils::readBlock(name, innerList, "{}");
  TQStringUtils::readBlanks(name);
  TList * list = TQStringUtils::tokenize(innerList, ",", true, "()[]{}");
  if (!list || !name.IsNull()) {
    setErrorMessage(__FUNCTION__,"Invalid list of bins for rates histogram");
    return NULL;
  }

  // the style scheme to apply to the final histogram
  TString styleScheme = options ? options->getTagStringDefault("styleScheme", f_styleScheme) : f_styleScheme;

  int n = list->GetEntries();
  int i = 1;

  // create histogram
  TH1D * histo = new TH1D("rates_histogram", "rates_histogram", n, 0., (double)n);
  histo->SetDirectory(NULL);

  // temporarily stop output
  int tmpVerbosity = f_Verbose;
  f_Verbose = 0;

  TString lastName;
  TList * thisSfList = new TList();
  thisSfList->SetOwner(false);

  bool stop = false;
  TQIterator itr(list, true);
  while (itr.hasNext() && !stop) {
    lastName = itr.readNext()->GetName();

    // try to obtain counter or histogram
    TQCounter * cnt = this->getCounter(path, lastName, options, thisSfList);
    if (!cnt) {
      TH1 * h = this->getHistogram(path, lastName, options, thisSfList);
      cnt = TQHistogramUtils::histogramToCounter(h);
      if (h) {
        delete h;
      }
    }

    if (cnt) {
      // apply style settings corresponding to <path>
      if (thisSfList) {
        TQHistogramUtils::applyStyle(histo,
                                     (TQSampleFolder*)thisSfList->First(), styleScheme);
        delete thisSfList;
        thisSfList = NULL;
      }

      TString title = cnt->GetTitle();
      if(title.IsNull()){
        title=lastName;
      }
      histo->GetXaxis()->SetBinLabel(i, title);
      histo->SetBinContent(i, cnt->getCounter());
      histo->SetBinError(i, cnt->getError());
      i++;
      delete cnt;
    } else {
      stop = true;
    }
  }

  // restore verbosity level
  f_Verbose = tmpVerbosity;

  // clean up
  if (thisSfList) {
    delete thisSfList;
    thisSfList = NULL;
  }

  if (stop) {
    delete histo;
    histo = NULL;
    setErrorMessage(__FUNCTION__,TString::Format("Element '%s' not found in '%s'",
                                                 lastName.Data(), path.Data()));
  }

  return histo;
}

//__________________________________________________________________________________|___________

TProfile * TQSampleDataReader::getProfile(const TString& path, const TString& name, const TString& options, TList * sfList) {
  TQTaggable * tagsOptions = new TQTaggable(options);
  TProfile * histo = getProfile(path, name, tagsOptions, sfList);
  delete tagsOptions;
  return histo;
}


//__________________________________________________________________________________|___________

TProfile * TQSampleDataReader::getProfile(const TString& path, TString name, TQTaggable * options, TList * sfList) {

  // restore original verbosity
  int tmpVerbose = f_Verbose;

  TProfile * histo = NULL;
  TQStringUtils::removeLeading(name, " \t");
  // this rather complicted block is used to parse histogram names like
  // [CutA+CutB-CutC]/VarName
  // and obtain the corresponding histogram
  TList* histNames = this->parsePaths(name);
  TQIterator itr(histNames);
  while(itr.hasNext()){
    TList * subList = dynamic_cast<TList*>(itr.readNext());
    TQIterator subItr(subList);
    while(subItr.hasNext()){
      TList * subsubList = dynamic_cast<TList*>(subItr.readNext());
      double factor = ((TQValue*)subsubList->At(0))->getDouble();
      TString histName(subsubList->At(1)->GetName());
      TProfile* newHisto = this->getElement<TProfile>(path, histName, this->f_pathHistograms, options, sfList);
      if(!newHisto) continue;
      if(!histo){
        histo = newHisto;
        histo->Scale(factor);
      } else {
        histo->Add(newHisto,factor);
        delete newHisto;
      }
    }
    delete histNames;
  }

  if (f_Verbose > 1) {
    VERBOSEclass("Profiles");
    VERBOSEclass("- original: %s",TQHistogramUtils::getDetailsAsString(histo, 2).Data());
  }
 

  // normalize the histogram if 'norm = true'
  if (options->getTagBoolDefault("norm", false)) {
    TQHistogramUtils::normalize(histo);
  }

  // apply additional style options
  TQHistogramUtils::applyStyle(histo, options);

  // reset histograms if requested
  if (histo && options->getTagBoolDefault("reset", false)) {
    histo->Reset();
  }

  // scale histogram
  double scale = 1.;
  if (histo && options->getTagDouble("scale", scale)) {
    histo->Scale(scale);

    if (!TMath::AreEqualRel(scale, 1., 0.01)) {
      // append the scale factor
      TString title(histo->GetTitle());
      title.Append(TString::Format(" (#times%g)", scale));
      histo->SetTitle(title);
    }
  }

  // restore original verbosity
  f_Verbose = tmpVerbose;

  // return the histogram
  return histo;
}


//__________________________________________________________________________________|___________

TH1 * TQSampleDataReader::getHistogram(const TString& path, const TString& name, const TString& options, TList * sfList) {

  TQTaggable * tagsOptions = new TQTaggable(options);
  TH1 * histo = getHistogram(path, name, tagsOptions, sfList);
  delete tagsOptions;
  return histo;
}


//__________________________________________________________________________________|___________

TH1 * TQSampleDataReader::getHistogram(const TString& path, TString name, TQTaggable * options, TList * sfList) {
  // Retrieves the histogram <name> by recursively summing corresponding contributions
  // from sample folders matching <path> and returns a pointer to an independent copy
  // of an instance of TH1 or a NULL pointer in case an error occured. Examples:
  //
  // - getHistogram("bkg", "Cut_0jet/MT") returns a pointer to the sum of all
  // contributing histograms "Cut_0jet/MT" in sample folder "bkg"
  //
  // Please note: the user is responsible for deleting the returned histogram!
  //
  // Manipulations to the retrieved histogram do not affect the original source
  // histograms. Contributing histograms are checked for consistency (dimension and
  // binning) before being summed.
  //
  // Multiple sample folder paths as well as histogram names may be summed and
  // subtracted 'in-situ' by using the addition and subtraction operators "+" and
  // "-", respectively. Examples:
  //
  // - getHistogram("bkg + sig", "Cut_0jet/MT") returns a pointer to the sum of
  // all contributing histograms "Cut_0jet/MT" in sample folder "bkg" and "sig"
  // - getHistogram("data - bkg", "Cut_0jet/MT") returns a pointer to the sum of
  // all contributing histograms "Cut_0jet/MT" in sample folder "data" subtracted
  // by the corresponding sum in sample folder "bkg"
  // - getHistogram("bkg", "Cut_0jet/MT + Cut_1jet/MT") returns a pointer to the
  // sum of all contributing histograms "Cut_0jet/MT" and "Cut_1jet/MT" in sample
  // folder "bkg"
  //
  // Furthermore, "rate histograms" with each bin corresponding to the total yield of one
  // category may be obtained by providing a comma-separated list of histogram or counter
  // names (see also TQSampleDataReader::getRatesHistogram(...)):
  //
  // - getHistogram("bkg", "{Cut_0jet, Cut_1jet}") returns a pointer to a histogram with 2
  // bins with the first (second) bin corresponding to the event yield obtained from the
  // sum of counter "Cut_0jet" ("Cut_1jet") in sample folder "bkg".
  //
  //
  // Several options can be used to manipulate the histogram before it is returned:
  //
  // - Applying histogram style options:
  //
  // Using style options various histogram style parameter can be set:
  //
  // @tag:title: sets the title using h->SetTitle(...)
  // @tag:histFillColor: sets the fill color using h->SetFillColor(...)
  // @tag:histFillStyle: sets the fill style using h->SetFillStyle(...)
  // @tag:histLineColor: sets the line color using h->SetLineColor(...)
  // @tag:histLineWidth: sets the line width using h->SetLineWidth(...)
  // @tag:histLineStyle: sets the line style using h->SetLineStyle(...)
  // @tag:histMarkerColor: sets the marker color using h->SetMarkerColor(...)
  // @tag:histMarkerSize: sets the marker width using h->SetMarkerSize(...)
  // @tag:histMarkerStyle: sets the marker style using h->SetMarkerStyle(...)
  // @tag:color: default to histFillColor, histLineColor, and histMarkerColor
  //
  // For colors either the ROOT color code or the corresponding color code name
  // may be used, e.g. "histFillColor = kRed" is equivalent to "histFillColor = 632".
  //
  // Defaults:
  // color >> (histFillColor, histLineColor, histMarkerColor)
  //
  // [Please note: this notation means that the tag <color> will be propagated to
  // tags <histFillColor>, <histLineColor>, and <histMarkerColor> as default,
  // which can overwrite the default]
  //
  //
  // - Cutting on histograms (removing bins):
  /*@tag: [cutBinLowX, cutBinHighX, cutBinLowY, cutBinHighY, cutBinLow, cutBinHigh]
    
    Cutting on histograms refers to completely removing bins to the left or right
    of any axis resulting in a new histogram with modified range of the corresponding
    axes. The bin contents of bins that are removed by this operation is NOT kept.
    The parameter <cutBinLow> and <cutBinHigh> refer to the bin index of those bins
    that form the first and last bin, respectively, of the axes of the new histogram.
    No cut is placed if the corresponding parameter is -1.
    
    Defaults:
    -1 >> cutBinLow >> cutBinLowX
    -1 >> cutBinHigh >> cutBinHighX
    -1 >> cutBinLowY
    -1 >> cutBinHighY
  */
  //
  // - Zooming into histograms:
  /*@tag: [zoomBinLowX, zoomBinHighX, zoomBinLowY, zoomBinHighY, zoomBinLow, zoomBinHigh]
    
    Zooming into histograms refers to selecting a sub-range of a histogram by merging
    bins to the left or right of any axis and accounting both bin content as well as
    bin errors to the corresponding underflow and overflow bins. The resulting histogram
    has a modified range of the corresponding axes but the same total integral as the
    original histogram. The parameter <zoomBinLow> and <zoomBinHigh> refer to the bin
    index of those bins that form the first and last bin, respectively, of the axes of
    the new histogram. Bins are kept if the corresponding parameter is -1.
    
    Defaults:
    -1 >> zoomBinLow >> zoomBinLowX
    -1 >> zoomBinHigh >> zoomBinHighX
    -1 >> zoomBinLowY
    -1 >> zoomBinHighY
  */
  // - Projecting 2D -> 1D:
  /*@tag: [projXBinLowY, projXBinY, projXBinHighY, projXBinY, projYBinLowX,
          projYBinX, projYBinHighX, projYBinX, projX, projX, projY, projY]
  
     Allows to project a 2D histogram onto one of its axes.
  
     Defaults:
     false >> projX ___(== false): -2 ___ projXBinY >> (projXBinLowY, projXBinHighY)
     \_(== true) : -1 _/
     false >> projY ___(== false): -2 ___ projYBinX >> (projYBinLowX, projYBinHighX)
     \_(== true) : -1 _/
  */
  //
  // - Rebinning (merging bins):
  /*@tag: [rebinX, rebinY, rebinZ, rebin]
    
     Histograms can be rebinned using rebinning options:
    
     > rebinX: merged <rebinX> bins on X axis to one bin using TH1::RebinX(...)
     or TH1::Rebin3D(...). No rebinning on X axis is performed if
     <rebinX> = 0
     > rebinY: merged <rebinY> bins on Y axis to one bin using TH1::RebinY(...)
     or TH1::Rebin3D(...). No rebinning on Y axis is performed if
     <rebinY> = 0
     > rebinZ: merged <rebinZ> bins on Z axis to one bin using TH1::Rebin3D(...).
     No rebinning on Z axis is performed if <rebinZ> = 0
     > rebin: default to <rebinX>
    
     Defaults:
     0 >> rebin >> rebinX
     0 >> rebinY
     0 >> rebinZ
  */
  //
  // The normalization of histograms can be changed using <norm> and <scale>:
  //
  // @tag:norm: scales the histogram such that the total integral of it is 1 using TH1::Scale(...)
  // @tag:scale: scales the histogram by a factor <scale> using TH1::Scale(...). This operation is performed after <norm>.
  //
  // - Reseting histograms:
  // @tag:[reset]: reset the histogram (set all contents to zero)
  //
  // - Applying a slope (reweighting):
  // @tag:[slope]: reweight a histogam with a slope
  //
  // - Applying Poisson errors:
  // @tag:[applyPoissonErrors]: apply poisson errors to all bins
  //
  // @tag:rerollGauss: randomize the bin contents of the histogram according to the erros set, assuming gaussian errors. \
  //                   if tag value is numeric, errors are scaled with that number
  // @tag:rerollPoisson: randomize the bin contents of the histogram according to the bin contents, assuming poisson errors
  // @tag:includeUnderflow: include the underflow bin in the first bin of the histogram
  // @tag:includeOverflow: include the overflow bin in the last bin of the histogram

  int tmpVerbose = f_Verbose;

  TH1 * histo = NULL;
  TQStringUtils::removeLeading(name, " \t");
  if (name.BeginsWith("{")) {
    histo = this->getRatesHistogram(path, name, options, sfList);
  } else {
    histo = dynamic_cast<TH1*>(this->getElement<TH1>(path, name, this->f_pathHistograms, options, sfList));
  }
	
  if (f_Verbose > 1) {
    VERBOSEclass("Histograms");
    VERBOSEclass("- original: %s",TQHistogramUtils::getDetailsAsString(histo, 2).Data());
  }
 
  if (histo){
    // cut/zoom histogram
    int cutBinLowX  = options->getTagIntegerDefault("cutBinLowX",  options->getTagIntegerDefault("cutBinLow",  -1));
    int cutBinHighX = options->getTagIntegerDefault("cutBinHighX", options->getTagIntegerDefault("cutBinHigh", -1));
    int cutBinLowY  = options->getTagIntegerDefault("cutBinLowY",  -1);
    int cutBinHighY = options->getTagIntegerDefault("cutBinHighY", -1);
    
    int zoomBinLowX  = options->getTagIntegerDefault("zoomBinLowX",  options->getTagIntegerDefault("zoomBinLow",  -1));
    int zoomBinHighX = options->getTagIntegerDefault("zoomBinHighX", options->getTagIntegerDefault("zoomBinHigh", -1));
    int zoomBinLowY  = options->getTagIntegerDefault("zoomBinLowY", -1);
    int zoomBinHighY = options->getTagIntegerDefault("zoomBinHighY", -1);
    
    double cutLowX,cutHighX,cutLowY,cutHighY,zoomLowX,zoomHighX,zoomLowY,zoomHighY = 0;
    if(options->getTagDouble("cutLowX",  cutLowX ) || options->getTagDouble("cutLow",  cutLowX )) cutBinLowX  = histo->GetXaxis()->FindBin(cutLowX);
    if(options->getTagDouble("cutHighX", cutHighX) || options->getTagDouble("cutHigh", cutHighX)) cutBinHighX = histo->GetXaxis()->FindBin(cutHighX)-1;
    if(options->getTagDouble("cutLowY",  cutLowY )) cutBinLowY = histo->GetYaxis()->FindBin(cutLowY );
    if(options->getTagDouble("cutHighY", cutHighY )) cutBinHighY= histo->GetYaxis()->FindBin(cutHighY)-1;
    if(options->getTagDouble("zoomLowX",  zoomLowX ) || options->getTagDouble("zoomLow",  zoomLowX )) zoomBinLowX  = histo->GetXaxis()->FindBin(zoomLowX);
    if(options->getTagDouble("zoomHighX", zoomHighX) || options->getTagDouble("zoomHigh", zoomHighX)) zoomBinHighX = histo->GetXaxis()->FindBin(zoomHighX);
    if(options->getTagDouble("zoomLowY",  zoomLowY )) zoomBinLowY = histo->GetYaxis()->FindBin(zoomLowY );
    if(options->getTagDouble("zoomHighY", zoomHighY )) zoomBinHighY= histo->GetYaxis()->FindBin(zoomHighY);
    
    if(cutBinLowX != -1 || cutBinHighX != -1 || cutBinLowY != -1 || cutBinHighY != -1 || zoomBinLowX != -1 || zoomBinHighX != -1 || zoomBinLowY != -1 || zoomBinHighY != -1) {
      TH1 * newHisto = TQHistogramUtils::cutAndZoomHistogram(histo,
                                                             cutBinLowX, cutBinHighX, cutBinLowY, cutBinHighY,
                                                             zoomBinLowX, zoomBinHighX, zoomBinLowY, zoomBinHighY);
      if (!newHisto) {
        setErrorMessage(__FUNCTION__,TString::Format(
                                                     "Invalid cut/zoom parameter for histogram '%s'", name.Data()));
      }
      delete histo;
      histo = newHisto;
      if (f_Verbose > 1) {
        VERBOSEclass("- cut/zoomed: %s",TQHistogramUtils::getDetailsAsString(histo, 2).Data());
      }
    }
 
    // ===== make projection =====
    
    int projXBinLowY = -2;
    int projXBinHighY = -2;
    int projYBinLowX = -2;
    int projYBinHighX = -2;
    if (!options->getTagInteger("projXBinLowY", projXBinLowY)) {
      if (!options->getTagInteger("projXBinY", projXBinLowY)) {
        projXBinLowY = (options->getTagBoolDefault("projX", false) ? -1 : -2);
      }
    }
    if (!options->getTagInteger("projXBinHighY", projXBinHighY)) {
      if (!options->getTagInteger("projXBinY", projXBinHighY)) {
        projXBinHighY = (options->getTagBoolDefault("projX", false) ? -1 : -2);
      }
    }
    if (!options->getTagInteger("projYBinLowX", projYBinLowX)) {
      if (!options->getTagInteger("projYBinX", projYBinLowX)) {
        projYBinLowX = (options->getTagBoolDefault("projY", false) ? -1 : -2);
      }
    }
    if (!options->getTagInteger("projYBinHighX", projYBinHighX)) {
      if (!options->getTagInteger("projYBinX", projYBinHighX)) {
        projYBinHighX = (options->getTagBoolDefault("projY", false) ? -1 : -2);
      }
    }
    if ((projXBinLowY != -2 || projXBinHighY != -2 || projYBinLowX != -2 || projYBinHighX != -2)) {
      TH1 * newHisto = NULL;
      if ((projXBinLowY != -2 || projXBinHighY != -2) && projYBinLowX == -2 && projYBinHighX == -2) {
        // projection on X
        newHisto = TQHistogramUtils::getProjectionX(histo, projXBinLowY, projXBinHighY);
      } else if ((projYBinLowX != -2 || projYBinHighX != -2) && projXBinLowY == -2 && projXBinHighY == -2) {
        // projection on Y
        newHisto = TQHistogramUtils::getProjectionY(histo, projYBinLowX, projYBinHighX);
    }
      if (!newHisto) {
        setErrorMessage(__FUNCTION__,TString::Format(
                                                     "Invalid projection parameter for histogram '%s'", name.Data()));
      }
      delete histo;
      histo = newHisto;
      if (f_Verbose > 1) {
        VERBOSEclass("- projected: %s",TQHistogramUtils::getDetailsAsString(histo, 2).Data());
      }
    }
    
    
  // ===== rebin histogram =====
    
    int rebinX = options->getTagIntegerDefault("rebinX", options->getTagIntegerDefault("rebin", 0));
    int rebinY = options->getTagIntegerDefault("rebinY", 0);
    int rebinZ = options->getTagIntegerDefault("rebinZ", 0);
    if ((rebinX > 0 || rebinY > 0 || rebinZ > 0)) {
      if(!TQHistogramUtils::rebin(histo,rebinX,rebinY,rebinZ)){
        delete histo;
        histo = 0;
        setErrorMessage(__FUNCTION__,TString::Format(
                                                     "Invalid rebinning parameter for histogram '%s'", name.Data()));
      }
      if (f_Verbose > 1) {
        VERBOSEclass("- rebinned: %s",TQHistogramUtils::getDetailsAsString(histo, 2).Data());
      }
    }

    //@tag: rebinXList: give a list "{1,2,3}" of bin boundaries for rebinning
    std::vector<double> rebinXList = options->getTagVDouble("rebinXList");
    if(rebinXList.size() > 0){
      TQHistogramUtils::rebin(histo,rebinXList,options->getTagBoolDefault("remap",false));
    }
    
    //@tag: unroll a two-dimensional histogram
    if(options->getTagBoolDefault("unroll",false)){
      TH2* h2 = dynamic_cast<TH2*>(histo);
      if(h2){
        bool firstX = options->getTagBoolDefault("unroll.firstX",true);
        bool includeUO = options->getTagBoolDefault("unroll.includeUndervlowOverflow",false);
        TH1* newhisto = TQHistogramUtils::unrollHistogram(h2,firstX,includeUO);
        if(newhisto){
          histo = newhisto;
          TQHistogramUtils::copyStyle(histo,h2);
          delete h2;
        }
      }
    }

    // apply slope to histogram
    double slope = 1.;
    if (options->getTagDouble("slope", slope)) {
      if (!TQHistogramUtils::applySlopeToHistogram(histo, slope)) {
        delete histo;
        histo = 0;
        setErrorMessage(__FUNCTION__,TString::Format(
                                                     "Invalid slope parameter for histogram '%s'", name.Data()));
      }
    }
    
    // normalize the histogram if 'norm = true'
    if (options->getTagBoolDefault("norm", false)) {
      TQHistogramUtils::normalize(histo);
    }
    
    // apply additional style options
    TQHistogramUtils::applyStyle(histo, options);
    
    // reset histograms if requested
    if (options->getTagBoolDefault("reset", false)) {
      histo->Reset();
    }

    // print norm factor in legend
    if(options->getTagBoolDefault("printNFInLegend",false)){
      TQTaggable newoptions(*options);
      
      newoptions.setTagString("scaleScheme","none"); // disable NF application
      newoptions.setTagBool("printNFInLegend",false); // needed to avoid infinite recursion
      TH1* unscaled = this->getHistogram(path,name,&newoptions,NULL);
      
      double nf = TQHistogramUtils::getIntegral(histo) / TQHistogramUtils::getIntegral(unscaled);

      delete unscaled;
      
      TString title(histo->GetTitle());
      title.Append(TString::Format(" (#times%.3g)", nf));
      histo->SetTitle(title);
    }

    // scale histogram
    double scale = 1.;
    if (options->getTagDouble("scale", scale)) {
      histo->Scale(scale);
      
      if (!TMath::AreEqualRel(scale, 1., 0.01)) {
        // append the scale factor
        TString title(histo->GetTitle());
        title.Append(TString::Format(" (#times%g)", scale));
        histo->SetTitle(title);
      }
    }
    
    int densityBin = 0;
    //@tag: scaleDensityToBin: scale all bin contents to densities, normalizing to the bin with the given index
    if (options->getTagInteger("scaleDensityToBin",densityBin)){
      histo->Scale(histo->GetXaxis()->GetBinWidth(densityBin),"width");
    }
    
    // apply Poisson errors if 'applyPoissonErrors = true'
    if (options->getTagBoolDefault("applyPoissonErrors", false)) {
      TQHistogramUtils::applyPoissonErrors(histo);
    }

    // scale errors if requested
    double scaleErrors;
    if (options->getTagDouble("scaleErrors", scaleErrors)) {
      DEBUGclass("scaling errors of histogram '%s' to %g",histo->GetName(),scaleErrors);
      TQHistogramUtils::scaleErrors(histo,scaleErrors);
    }
    
    // reroll the histogram if required
    double zvalue;
    if(options->getTagDouble("rerollGauss",zvalue)){
      TQHistogramUtils::rerollGauss(histo,zvalue);
    } else if(options->getTagBoolDefault("rerollPoisson")){
      TQHistogramUtils::rerollPoisson(histo);
    }
    
    // include under- and overflow bins
    TQHistogramUtils::includeOverflowBins(histo, options->getTagBoolDefault("includeUnderflow",false), options->getTagBoolDefault("includeOverflow",false));
    
    // ensure a minimum bin content as given
    double minBinContent;
    if (options->getTagDouble("ensureMinimumBinContent", minBinContent)) {
      TQHistogramUtils::ensureMinimumBinContent(histo,minBinContent,true);
    }
    
    std::vector<TString> labelsX = options->getTagVString("relabelX");
    for(size_t i=0; i<labelsX.size(); ++i){
      histo->GetXaxis()->SetBinLabel(i+1,labelsX[i]);
      histo->GetXaxis()->LabelsOption("v");
    }
    std::vector<TString> labelsY = options->getTagVString("relabelY");
    for(size_t i=0; i<labelsY.size(); ++i){
      histo->GetYaxis()->SetBinLabel(i+1,labelsY[i]);
    }
    std::vector<TString> labelsZ = options->getTagVString("relabelZ");
    for(size_t i=0; i<labelsZ.size(); ++i){
      histo->GetZaxis()->SetBinLabel(i+1,labelsZ[i]);
    }
    
  }
  
  // restore original verbosity
  f_Verbose = tmpVerbose;
 
  // return the histogram
  return histo;
}

//__________________________________________________________________________________|___________

THnBase * TQSampleDataReader::getTHnBase(const TString& path, const TString& name, const TString& options, TList * sfList) {
  // Does the same as getHistogram with support for multidimentional histograms (THnBase)
  TQTaggable * tagsOptions = new TQTaggable(options);
  THnBase * histo = getTHnBase(path, name, tagsOptions, sfList);
  delete tagsOptions;
  return histo;
}

//__________________________________________________________________________________|___________

THnBase * TQSampleDataReader::getTHnBase(const TString& path, TString name, TQTaggable * options, TList * sfList) {
  // Does the same as getHistogram with support for multidimentional histograms (THnBase)
  int tmpVerbose = f_Verbose;

  THnBase * histo = NULL;
  TQStringUtils::removeLeading(name, " \t");
  histo = dynamic_cast<THnBase*>(this->getElement<THnBase>(path, name, this->f_pathHistograms, options, sfList));
  
  // restore original verbosity
  f_Verbose = tmpVerbose;
 
  // return the histogram
  return histo;
}

//__________________________________________________________________________________|___________

TGraph * TQSampleDataReader::getGraph(const TString& path, const TString& name, const TString& options, TList * sfList) {
  // Does the same as getHistogram for TGraph
  TQTaggable * tagsOptions = new TQTaggable(options);
  TGraph * graph = getGraph(path, name, tagsOptions, sfList);
  delete tagsOptions;
  return graph;
}

//__________________________________________________________________________________|___________

TGraph * TQSampleDataReader::getGraph(const TString& path, TString name, TQTaggable * options, TList * sfList) {
  // Does the same as getHistogram for TGraph
  int tmpVerbose = f_Verbose;

  TGraph * graph = NULL;
  TQStringUtils::removeLeading(name, " \t");
  graph = dynamic_cast<TGraph*>(this->getElement<TGraph>(path, name, this->f_pathGraphs, options, sfList));
  
  // restore original verbosity
  f_Verbose = tmpVerbose;
 
  // return the histogram
  return graph;
}

//__________________________________________________________________________________|___________

TGraph2D * TQSampleDataReader::getGraph2D(const TString& path, const TString& name, const TString& options, TList * sfList) {
  // Does the same as getHistogram for TGraph
  TQTaggable * tagsOptions = new TQTaggable(options);
  TGraph2D * graph = getGraph2D(path, name, tagsOptions, sfList);
  delete tagsOptions;
  return graph;
}

//__________________________________________________________________________________|___________

TGraph2D * TQSampleDataReader::getGraph2D(const TString& path, TString name, TQTaggable * options, TList * sfList) {
  // Does the same as getHistogram for TGraph
  int tmpVerbose = f_Verbose;

  TGraph2D * graph = NULL;
  TQStringUtils::removeLeading(name, " \t");
  graph = dynamic_cast<TGraph2D*>(this->getElement<TGraph2D>(path, name, this->f_pathGraphs, options, sfList));
  
  // restore original verbosity
  f_Verbose = tmpVerbose;
  
  // return the histogram
  return graph;
}

//__________________________________________________________________________________|___________

template<class T>
TList * TQSampleDataReader::collectElements(TList* paths, TList* elements, const TString& subPath, TList * sfList, TQTaggable * options) {
  TList * list = new TList();
  list->SetOwner(true);
  
  TString scaleScheme = options ? options->getTagStringDefault("scaleScheme", f_scaleScheme) : f_scaleScheme;

  // the number of elements added up
  int nElements = 0;

  // loop over sample folders
  TQListIterator pIter(paths);
  TQListIterator eIter(elements);
  if(!pIter.isValid()) return NULL;
  if(!eIter.isValid()) return NULL;
  
  while (pIter.hasNext() && nElements >= 0) {
    TList* path = pIter.readNext();
    TString pathName = path->At(1)->GetName();
    double pathfactor = ((TQValue*)path->At(0))->getDouble();
    if (this->f_Verbose > 3) {
      INFOclass("Collecting using path expression %s (factor:%f)",pathName.Data(),pathfactor);
    }
    TQSampleFolderIterator pIter2(this->f_baseSampleFolder->getListOfSampleFolders(pathName, TQSampleFolder::Class(), true), true);
    while(pIter2.hasNext() && nElements >= 0){
      TQSampleFolder* sf = pIter2.readNext();
      if(!sf) continue;
      if (this->f_Verbose > 3) {
        INFOclass("Collecting from path match %s",sf->getPath().Data());
      }
      // get the base scale factor
      eIter.reset();
      while(eIter.hasNext() && nElements >= 0) {
        TList* element = eIter.readNext();
        TString elementName = element->At(1)->GetName();
        double elementfactor =  ((TQValue*)element->At(0))->getDouble();
        TQCounter* scale = new TQCounter("prefactor",pathfactor*elementfactor,0.);
        
        TList * baseScaleList = this->getBaseScaleFactors(sf, elementName, scaleScheme);
        if (!baseScaleList) {
          baseScaleList = new TList();
        }
        baseScaleList->Add(scale); //pathFactor*elementFactor
        
        // dump some info text if in verbose mode
        if (f_Verbose > 3) {
          VERBOSEclass("now collecting elements for path '%s' (with %d base scale factors)",sf->getPath().Data(),baseScaleList->GetEntries());
          if (f_Verbose > 4) {
            TQCounterIterator itr(baseScaleList);
            while(itr.hasNext()){
              TQCounter* c = itr.readNext();
              VERBOSEclass("%s = %g",c->GetName(),c->getCounter());
            }
          }
        }
        
        // sum up the Elements of this path
        nElements += this->collectElementsWorker<T>(sf, elementName, subPath, list, baseScaleList, sfList, options, 1);
        
        baseScaleList->SetOwner(false);
        delete baseScaleList;
      }        
    }
  }
  // return the number of element added
  return list;
}

//__________________________________________________________________________________|___________

void TQSampleDataReader::setDefaultScaleScheme(const TString& defaultScheme){
  this->f_scaleScheme = defaultScheme;
}

//__________________________________________________________________________________|___________

template<class T>
int TQSampleDataReader::collectElementsWorker(TQSampleFolder * sampleFolder, const TString& elementName, const TString& subPath, TList* list, TList * baseScaleList, TList * sfList, TQTaggable * options, int indent) {
  TList * scaleList = new TList();
  list->SetOwner(false);
  scaleList->SetOwner(false);
  scaleList->AddAll(baseScaleList);
  // stop if the sample folder is invalid
  if (!sampleFolder) return 0;
  TString styleScheme = options ? options->getTagStringDefault("styleScheme", f_styleScheme) : f_styleScheme;
  TString filterScheme = options ? options->getTagStringDefault("filterScheme",f_filterScheme) : f_filterScheme;
  TString normScheme = options ? options->getTagStringDefault("normScheme", f_normScheme) : f_normScheme;
  TString scaleScheme = options ? options->getTagStringDefault("scaleScheme", f_scaleScheme) : f_scaleScheme;
 
  // get the local scale factor
  // skip local scale factor of base sample folder in local mode
  if (!(f_localMode && sampleFolder == f_baseSampleFolder)) {

    // dump some info text if in verbose mode
    TString text;
    TString sftext;
    bool printsftext = (f_Verbose > 4);
    if (f_Verbose > 3) {
      // indentation
      text.Append(TQStringUtils::repeat(" ", indent * 2));
      // sample folder name
      text.Append(TQStringUtils::makeBoldBlue(sampleFolder->GetName()));
      text.Append(" ");
      text.Append(subPath);
      // the scale factors
      sftext.Append(TQStringUtils::makeBoldRed(" [local SF = ("));
    }

    TString scaleFactorPath = elementName;
    while (!scaleFactorPath.IsNull()) {
      TQCounter * localScaleItemCounter = sampleFolder->getScaleFactorCounterInternal(scaleScheme + ":" + scaleFactorPath);
      if (localScaleItemCounter) scaleList->Add(localScaleItemCounter);
      //Important: handle the pointer with care (i.e. do not replace by a copy), correlations might be determined later via pointer comparison! 
      TQFolder::getPathTail(scaleFactorPath);
 
      if (f_Verbose > 3) {
        if (!localScaleItemCounter || localScaleItemCounter->getCounter() == 1.) {
          sftext.Append(TQStringUtils::makeBoldWhite("1."));
        } else {
          printsftext = true;
          sftext.Append(TQStringUtils::makeBoldRed(TString::Format("%f", localScaleItemCounter->getCounter() )));
        }
        if (!scaleFactorPath.IsNull()) {
          sftext.Append(TQStringUtils::makeBoldRed(", "));
        } else {
          sftext.Append(TQStringUtils::makeBoldRed(")]"));
        }
      }
    }
    if (f_Verbose > 3) {
      if(printsftext)
        VERBOSEclass(text + sftext);
      else
        VERBOSEclass(text);
    }
  }
 
  // the number of elements added
  int nElements = 0;
  sampleFolder->resolveImportLinks();
  T* obj = NULL;
  obj = dynamic_cast<T*>(sampleFolder->getObject(TQFolder::concatPaths(subPath, elementName)));
  if (obj){
    TList * localList = new TList();
    TString folderName = elementName;
    TQFolder::getPathTail(folderName); //remove name of actual object
    localList->Add(sampleFolder->getFolder(TQFolder::concatPaths(subPath,folderName)));
    localList->Add(obj);
    localList->AddAll(scaleList);
    list->Add(localList);
    nElements = 1;
    obj = 0;
    if (sfList) sfList->Add(sampleFolder);
  } else {
    // loop over sub sample folders
    TQIterator itr(sampleFolder->getListOfSampleFolders("?"),true);
    while (itr.hasNext() && nElements >= 0) {
      // collect the histograms for every sub sample folder
      TQSampleFolder* sf = dynamic_cast<TQSampleFolder*>(itr.readNext());
      if(!sf) continue;
      // step into the sub sample folder recursively
      nElements+=this->collectElementsWorker<T>(sf, elementName, subPath, list, scaleList, sfList, options, indent + 1);
    }
  }
 
  delete scaleList;
  // return the number of elements added
  return nElements;
}

//__________________________________________________________________________________|___________

template<class T>
int TQSampleDataReader::sumElements( TList * list, T * &histo, TQTaggable * options ) {
  // sum all elements in the list (histogram variant)
  int nElements = 0;
  if (!list) return -1;
  bool simpleUseScaleUncertainty = options ? options->getTagBoolDefault("simpleUseScaleUncertainty", true) : true;
  //@tag: [includeScaleUncertainty] This can is used to enable the inclusion of scale ("NF") uncertainties in histograms and counters. Default: false
  bool useScaleUncertainty = options ? options->getTagBoolDefault("includeScaleUncertainty", false) : false;
  //@tag: [.correlate.path, .correlate.value] See "useManualCorrelations"
  //@tag: [useManualCorrelations] If set to true this option enables the inclusion of correlations between (unscaled) histograms and (unscaled cutflow) counters. The correlation information in this regard is retrieved in the following way:
  /*@tag  Starting from the folder in which the original element (=histogram or counter) is stored the folder strucutre is searched towards the root folder. The search stops when the tag ".correlate.path.0" is found (in other words a vector tag with the key ".correlate.path") or the root folder is searched without success. Tags from folders located in the hierarchy above the folder with the first match are ignored. For each entry in the ".correlate.path" vector a corresponding correlation value is retrieved via the key ".correlate.value.x", where "x" is the index in the former vector (in case no value can be obtained from ".correlate.value.x", 1.0 is used as a default). The value of the ".correlate.path.x" tag should give the path to a single folder. Elements in (sub) folders of the one where this tag is set and elements from the referenced folder are considered as correlated during summation according to the aforementioned correlation value IF they represent the same element. Representing the same element is determined by comparing the the object names  (typically the name of the cut for cutflow counters / the name of the distribution for histograms) concatenated with their path in the folder structure starting (but not including) the last instance of TQSample(Folder) above the elements location.
  
  */
  bool useManualCorrelations = options ? options->getTagBoolDefault("useManualCorrelations", false) : false;
  bool advancedUncertaintyTreatment = useScaleUncertainty || useManualCorrelations;
  // if there is a valid histogram in this sample folder...
  TQListIterator itr(list);
  while(itr.hasNext()){
    TList* l = itr.readNext();
    if(!l || l->GetSize() < 1) continue;
    double scale = 1.;
    double scaleerr = 0.;
    for (int j=2; j<l->GetSize(); j++) {
      TQCounter* cnt = dynamic_cast<TQCounter*>(l->At(j));
      if(cnt) scale *= cnt->getCounter();
      if (cnt) scaleerr = sqrt(scaleerr*scaleerr + cnt->getError()*cnt->getError());
    }
    T* h = dynamic_cast<T*>(l->At(1));
    if (!histo) {
      if(!h){
        ERRORclass("object list [1] is not a histogram!");
        continue;
      }
      histo = (T*)TQHistogramUtils::copyHistogram(h,"NODIR");
      if (!histo) {
        ERRORclass("Failed to copy histogram!");
        continue;
      }
      if (simpleUseScaleUncertainty)
        TQHistogramUtils::scaleHistogram(histo,scale,scaleerr,true);
      else
        TQHistogramUtils::scaleHistogram(histo,scale,0.,false);
      DEBUGclass("starting summation with histogram: s(w)=%g, n=%d",histo->Integral(),(int)(histo->GetEntries()));
      nElements = 1;
    } else {
      //ToDo: check for histogram consistency, abort with error message if not!
      DEBUGclass("adding histogram: s(w)=%g, n=%d",h->Integral(),(int)(h->GetEntries()));
      if (simpleUseScaleUncertainty)
        TQHistogramUtils::addHistogram(histo,h,scale,scaleerr,0.,true);
      else
        TQHistogramUtils::addHistogram(histo,h,scale,0.,0.,false);
      nElements++;
    }
  }
  if (advancedUncertaintyTreatment && nElements>0) {
    //You do not want to do this yourself!
    //reset all bin errors (this is the resulting histogram, uncertainties are recomputed from all input histograms + counters in the following)
    double correlationAux = 0.; //auxillary variable to retrieve correlations between different paths
    const int nbins = TQHistogramUtils::getNbinsGlobal(histo);
    std::vector<double> binErrors2(nbins);
    TQListIterator cont1(list);
    while(cont1.hasNext()){
      TList* c1 = cont1.readNext();
      if(!c1 || c1->GetSize()<2) continue;
      TQFolder * oFolder1 = dynamic_cast<TQFolder*>(c1->At(0));
      
      
      T* h1 = dynamic_cast<T*>(c1->At(1)); //index 0 is the TQFolder for which this list was created.
      if(!h1) continue;
      if(!TQHistogramUtils::checkConsistency(histo, h1) ) continue;
      for (int j=1; j<c1->GetSize(); j++) {
        if (!c1->At(j)) continue;
        TQListIterator cont2(list);
        while(cont2.hasNext()){
          TList* c2 = cont2.readNext();
          if(!c2 || c2->GetSize()<2) continue;
          TQFolder * oFolder2 = dynamic_cast<TQFolder*>(c2->At(0));
          
          T* h2 = dynamic_cast<T*>(c2->At(1));
          if(!h2) continue;
          if(!TQHistogramUtils::checkConsistency(histo, h2) ) continue;
          correlationAux = 0.;//reset, just to be sure
          for (int l=1; l<c2->GetSize(); l++) {
            if (!c2->At(l)) continue;
            if ( j == 1 && l == 1) { // if we are at the right position for histograms and
              if ( h1 == h2 || ( useManualCorrelations && TQStringUtils::equal(h1->GetName(),h2->GetName()) && areFoldersCorrelated(oFolder1,oFolder2,correlationAux) ) ) { //the histograms are the same, do:
                if (h1 == h2) correlationAux = 1.;
                //uncertainty treatment for histogram part
                double tempErr = correlationAux;
                for (int j1=2; j1<c1->GetSize(); j1++) {
                  TQCounter* cnt = dynamic_cast<TQCounter*>(c1->At(j1));
                  if(!cnt) continue;
                  tempErr *= cnt->getCounter();
                }
                for (int l1=2; l1<c2->GetSize(); l1++) {
                  TQCounter* cnt = dynamic_cast<TQCounter*>(c2->At(l1));
                  if(!cnt) continue;
                  tempErr *= cnt->getCounter();
                }
                for (int bin = 0; bin<nbins; bin++) {
                  binErrors2[bin] += h1->GetBinError(bin) * h2->GetBinError(bin)*tempErr;
                }
              } 
              //uncertainty treatment for scale factor part
            } else if ( j > 1 && l > 1 && useScaleUncertainty ) {// if we are at the right position for NF counters and want to include scale uncertainties
              //the combination of histogram (l or j == 1) and scale factor (j or l > 1) is always uncorrelated, i.e. the contribution vanishes anyway.
              //uncertainty treatment for NF counter part
              //note that this relies heavily on how the TQNFChainloader stores its results, not using the chainloader can easily break something.
              //if the current entries in question do not have any uncertainties, we don't need to do anything. Also: protect against non-TQCounter objects.
              if ( !(c1->At(j) && c1->At(j)->IsA() == TQCounter::Class() ) || !((c2->At(l)) && c2->At(l)->IsA() == TQCounter::Class()) || (static_cast<TQCounter*>(c1->At(j)))->getError() == 0. || (static_cast<TQCounter*>(c2->At(l)))->getError() == 0.) continue;
              
              TString id1 = c1->At(j)->GetTitle();
              TString id2 = c2->At(l)->GetTitle();
              
              double tempErr = 1.;
              for (int j1=2; j1<c1->GetSize(); j1++) {
                if (!c1->At(j1) || c1->At(j1)->IsA() != TQCounter::Class()) continue;
                TQCounter* cnt = static_cast<TQCounter*>(c1->At(j1));
                tempErr *= ( j == j1 ? cnt->getError() : cnt->getCounter() );
              }
              for (int l1=2; l1<c2->GetSize(); l1++) {
                if (!c2->At(l1) || c2->At(l1)->IsA() != TQCounter::Class()) continue;
                TQCounter* cnt = static_cast<TQCounter*>(c2->At(l1));
                tempErr *= ( l == l1 ? cnt->getError() : cnt->getCounter() );
              }
              if ( !id1.IsNull() && !id2.IsNull() ) {
                //try to retrieve the folders containing the correlations between different NFs
                TQFolder* f1 = this->getCorrelationFolderByIdentifier(id1);
                TQFolder* f2 = this->getCorrelationFolderByIdentifier(id2);
                if ( f1 && f2 && f1->getObject(id2) ) {
                  double corr = (dynamic_cast<TQCounter*>( f1->getObject(id2) ))->getCounter();
                  //debug only code: some additional checks to verify that the correlation matrix is symetric (i.e. the correlation storage is working)
                  #ifdef _DEBUG_
                  if ( f2->getObject(id1) && !TMath::AreEqualAbs(corr, (dynamic_cast<TQCounter*>( f2->getObject(id1) ))->getCounter() , std::numeric_limits<double>::epsilon()) ) {
                    WARNclass("Correlations do not match: cor(x1,x2) = %d, cor(x2,x1) = %d \nID(x1) = '%s'\nID(x2) = '%s'", corr, (dynamic_cast<TQCounter*>( f2->getObject(id1) ))->getCounter(), id1.Data(), id2.Data() );
                  } else if (!f2->getObject(id1)) {
                    ERRORclass("Failed to retrieve secondary correlation counter (ID '%s') from folder %s", id1.Data(), f2->getPath().Data());
                  }
                  #endif
                  //apply correlation factor to the current term in the uncertainty calculation
                  tempErr*= corr;
                } else {
                  //fall back solution: at least try to correlate identical counters. This also offers support for some simple (manual) implementation of correlations by setting the title of different scale factors to the same value
                  WARNclass("Failed to retrieve correlation counter from folder '%s' for NF counters with identifiers (titles) %s and %s!",f1->getPath().Data(),id1.Data(),id2.Data());
                  if (TQStringUtils::equal(id1,id2)) {
                    WARNclass("identifiers are equal, assuming full correlation (100%)");
                  } else {
                    WARNclass("identifiers are equal, assuming no correlation (0%)");
                    tempErr *= 0;
                  }
                }
                for (int bin = 0; bin<nbins; bin++) {
                  binErrors2[bin] += h1->GetBinContent(bin) * h2->GetBinContent(bin) * tempErr; //apply to bins
                }
              } else if (!(TQStringUtils::equal(c1->At(j)->GetName(),"prefactor") || TQStringUtils::equal(c2->At(l)->GetName(),"prefactor") )){
                WARNclass("At least one counter has no title, skipping uncertainty contribution involving this counter");
                (dynamic_cast<TQCounter*>(c1->At(j)))->print();
                (dynamic_cast<TQCounter*>(c2->At(l)))->print();
              }
            }
          } 
        }
      }
    }
    for (int bin=0; bin<nbins; ++bin) {
      histo->SetBinError(bin, sqrt(binErrors2[bin]));
    }
  }
  # ifdef _DEBUG_
  if(histo){
    DEBUGclass("resulting histogram: s(w)=%g, n=%d",histo->Integral(),(int)(histo->GetEntries()));
  } else {
    DEBUGclass("no histogram generated!");
  }
  #endif
  return nElements;
}

//__________________________________________________________________________________|___________

int TQSampleDataReader::sumElements( TList * list, THnBase * &histo, TQTaggable * /*options*/) {
  // sum all elements in the list (multidimensional histogram (THnBase) variant)
	// Remark: No support for advanced uncertainty treatment as for TH1 histograms

  int nElements = 0;
  if (!list) return -1;

  // bool useScaleUncertainty = options ? options->getTagBoolDefault("includeScaleUncertainty", false) : false;
  // bool useManualCorrelations = options ? options->getTagBoolDefault("useManualCorrelations", false) : false;
  //  bool advancedUncertaintyTreatment = useScaleUncertainty || useManualCorrelations;
  // if there is a valid histogram in this sample folder...
  TQListIterator itr(list);
  while(itr.hasNext()){
    TList* l = itr.readNext();
    if(!l || l->GetSize() < 1) continue;
    double scale = 1.;
    for (int j=2; j<l->GetSize(); j++) {
      TQCounter* cnt = dynamic_cast<TQCounter*>(l->At(j));
      if(cnt) scale *= cnt->getCounter();
    }
    THnBase* h = dynamic_cast<THnBase*>(l->At(1));
    if (!histo) {
      if(!h){
        ERRORclass("object list [1] is not a histogram!");
        continue;
      }
      
      histo = (THnBase*)TQTHnBaseUtils::copyHistogram(h,"NODIR");
      
      if (!histo) {
        ERRORclass("Failed to copy histogram!");
        continue;
      }
      TQTHnBaseUtils::scaleHistogram(histo,scale,0.,false);
      nElements = 1;
    } else {
      TQTHnBaseUtils::addHistogram(histo,h,scale,0.,0.,false);
      nElements++;
    }
  }

  # ifdef _DEBUG_
  if(histo){
    DEBUGclass("resulting histogram: n=%d",(int)(histo->GetEntries()));
  } else {
    DEBUGclass("no histogram generated!");
  }
  #endif
  return nElements;
}

//__________________________________________________________________________________|___________

int TQSampleDataReader::sumElements( TList * list, TQTable * &table, TQTaggable * /*options*/ ) {
  // sum all elements in the list (eventlist variant)
  int nElements = 0;
  if (!list) return -1;
  TQListIterator itr(list);
  while(itr.hasNext()){
    TList* l = itr.readNext();
    if(!l || l->GetSize() < 2) continue;
    TQTable* t = dynamic_cast<TQTable*>(l->At(1));
    if(!t) continue;
    if (table) {
      table->appendLines(t,1,true);
    } else {
      table = new TQTable(t);
    }
    nElements++;
  }
  return nElements;
}

//__________________________________________________________________________________|___________

int TQSampleDataReader::sumElements( TList * list, TGraph * &graph, TQTaggable * /*options*/ ) {
  // sum all elements in the list (TGraph variant)
  int nElements = 0;
  if (!list) return -1;
  TQListIterator itr(list);
  while(itr.hasNext()){
    TList* l = itr.readNext();
    if(!l || l->GetSize() < 2) continue;
    TGraph* g = dynamic_cast<TGraph*>(l->At(1));
    if(!g) continue;
    if (graph) {
      TQHistogramUtils::addGraph(graph,g);
    } else {
      graph = (TGraph*)TQHistogramUtils::copyGraph(g);
    }
    nElements++;
  }
  return nElements;
}

//__________________________________________________________________________________|___________

int TQSampleDataReader::sumElements( TList * list, TGraph2D * &graph, TQTaggable * /*options*/ ) {
  // sum all elements in the list (TGraph variant)
  int nElements = 0;
  if (!list) return -1;
  TQListIterator itr(list);
  while(itr.hasNext()){
    TList* l = itr.readNext();
    if(!l || l->GetSize() < 2) continue;
    TGraph2D* g = dynamic_cast<TGraph2D*>(l->At(1));
    if(!g) continue;
    if (graph) {
      TQHistogramUtils::addGraph(graph,g);
    } else {
      graph = (TGraph2D*)TQHistogramUtils::copyGraph(g);
    }
    nElements++;
  }
  return nElements;
}

//__________________________________________________________________________________|___________

int TQSampleDataReader::sumElements( TList * list, TQPCA * &pca, TQTaggable * /*options*/ ) {
  // sum all elements in the list (TQPCA variant)
  int nElements = 0;
  if (!list) return -1;
  TQListIterator itr(list);
  while(itr.hasNext()){
    TList* l = itr.readNext();
    if(!l || l->GetSize() < 2) continue;
    TQPCA* p = dynamic_cast<TQPCA*>(l->At(1));
    if(!p) continue;
    if (pca) {
      pca->add(p);
    } else {
      pca = (TQPCA*)(p->Clone());
    }
    nElements++;
  }
  return nElements;
}

//__________________________________________________________________________________|___________

int TQSampleDataReader::sumElements( TList * list, TTree * &tree, TQTaggable * /*options*/ ) {
  // sum all elements in the list (TTree variant)
  if (!gDirectory->IsWritable()) { 
    // else create the tree out of the trees in the TList,
    // in case there's no writabel file return an error message */
    ERRORclass("current directory is not writable and you can't create a tree on memory ");
    ERRORclass("please create a new TFile before getting the tree ");
    tree = NULL;
    return -1;
  }
  int nElements = 0;
  if (!list) return -1;
  TList* trees = new TList();
  trees->SetOwner(false);
  TQListIterator itr(list);
  while(itr.hasNext()){
    TList* l = itr.readNext();
    if(!l || l->GetSize() < 2) continue;
    TTree* t = dynamic_cast<TTree*>(l->At(1));
    if(!t) continue;
    trees->Add(t);
    nElements++;
  }
  if (trees->GetSize()==0){
    // without any tree, there's nothing to return
    tree = NULL;
    delete trees;
    return 0;
  } else if (trees->GetSize()==1){
    // if there's only one tree in the path it's easy, just return this one
    tree = (TTree*)trees->At(0)->Clone(); 
    delete trees;
    return 1;
  } else {
    // in this case, we need to merge
    tree = TTree::MergeTrees(trees);
    delete trees;
    return nElements;
  }
}

//__________________________________________________________________________________|___________

TString TQSampleDataReader::getStoragePath(TQFolder* f) {
  // get the tail of the folders path which corresponds to folders not inheriting 
  // from TQSampleFolder
  TString path = "";
  TQFolder* thisFolder = f;
  while (!thisFolder->InheritsFrom(TQSampleFolder::Class())) {
    path = TQFolder::concatPaths(thisFolder->GetName(),path);
    thisFolder = thisFolder->getBase();
  }
  return path;
}

bool TQSampleDataReader::areFoldersCorrelated(TQFolder* f1, TQFolder* f2) {
  double tmp = 0.;
  return this->areFoldersCorrelated(f1,f2,tmp);
}

//__________________________________________________________________________________|___________

bool TQSampleDataReader::areFoldersCorrelated(TQFolder* f1, TQFolder* f2, double& correlation, bool reversed) {
  // checks if tags are set indicating that elements in <f1> should be correlated 
  // with elements in folder <f2>. the corresponding correlation value if then 
  // stored in <correlation>. Please note that one should check in addition if the
  // actual objects in question are correlated (e.g. same distribution / same cut 
  // stage).
  
  if (!f1 || !f2) {
    ERRORclass("Recieved null-pointer, returning false!");
    return false;
  }
  correlation = 0.; //just to be sure

  //perform pre-check (we only want to correlate elements from the same "storage" folders, i.e. the parts of the path which do not correspond to instances of TQSample(Folder) but are basic TQFolders. This is used to only correlate objects representing the same distribution (for histograms) at the same cut
  if (!TQStringUtils::equal( getStoragePath(f1), getStoragePath(f2) )) return false;  
    
  TQFolder* rootFolder = f1->getRoot(); //we could also use f_baseSampleFolder instead, but this way we could make this a static method if needed and/or shift it to another class
  bool found = false;
  TQFolder* currentFolder = f1;
  std::vector<TString> paths;
  while ( !found && currentFolder ) { //if current
    currentFolder->getTag(".correlate.path",paths);
    if (paths.size() > 0) { //there is some information here, now evaluate it
      found = true;
      for (size_t i=0; i<paths.size(); i++) {
        TQFolder* f = rootFolder->getFolder(paths[i]);
        if (f && (f == f2 || f->isBaseOf(f2)) ) {  //we found a match if the path points to the second folder (f2) or a base folder of it
          //get correlation value from the folder we found the matching reference in. If no corresponding tag is found, issue a warning and assume full correlation (correlation = 1.)
          if (!currentFolder->getTagDouble(TString::Format(".correlate.value.%d",(int)i), correlation)) {
            WARNclass("Found correlation reference from folder '%s' to folder '%s' (from tag '.correlate.path.%d') but no corresponding correlation value. Full correlation (1.0) will be assumed!",currentFolder->getPath().Data(),f->getPath().Data(),i);
            correlation = 1.;
          }
          return true;
        }
      }
      //if we get to this point, we found some information but the second folder (f2) does not match any reference. We do not consider any other information from folders higher in the hierarchy (the user is expected to always re-write the full correlation set when overwriting further down the hierarchy! This allows to drop some correlations for a sub folder if needed). Since we found some information but no match we are done and return false since the two folders given seem to be uncorrelated
      return false;
    } else { //no information found, continue search on the base folder
      currentFolder = currentFolder->getBase();
    }
  }
  if (!reversed) return this->areFoldersCorrelated(f2,f1,correlation,true); //check with reversed order of f1,f2 if no information was found on f1 (we will now search in the tree leading to f2)
  //if we come to this point and already searched the opposite order, then there is no correlation.
  return false;
  
}

int TQSampleDataReader::sumElements( TList * list, TQCounter * &counter, TQTaggable * options ) {
  // sum all elements in the list (counter variant)
  
  // The first TList argument is expected to have a 2D structure: It is a list of sublists.
  // Each sublist should be composed as follows: A TQFolder object at the very first place (index 0)
  // which is the origin in the sample folder structure of the following entry (index 1). This is 
  // one of the actual elements to be summed up followed by an arbitrary number of scale factors
  // (usually "NFs") with which the index 1 element is multiplied (scaled) before the summation.
  // The summation essentially corresponds to an iteration over the initial list, each sublist
  // contribution one term.
  
  int nElements = 0;
  if (!list) return -1;
  bool simpleUseScaleUncertainty = options ? options->getTagBoolDefault("simpleUseScaleUncertainty", true) : true;
  bool useScaleUncertainty = options ? options->getTagBoolDefault("includeScaleUncertainty", false) : false;
  bool useManualCorrelations = options ? options->getTagBoolDefault("useManualCorrelations", false) : false;
  bool advancedUncertaintyTreatment = useScaleUncertainty || useManualCorrelations;
  //calculate the weighted/scaled sum of the counters:
  TQListIterator itr(list);
  while(itr.hasNext()){
    TList* l = itr.readNext();
    if(!l || l->GetSize() < 2) continue;
    TQCounter* cnt = dynamic_cast<TQCounter*>(l->At(1));
    if(!cnt) continue;
    double scale = 1.;
    double scaleerr = 0.;
    for (int j=2; j<l->GetSize(); j++) {
      TQCounter * c = dynamic_cast<TQCounter*>(l->At(j));
      if (c) scale *=c->getCounter();
      if (c) scaleerr = sqrt(scaleerr*scaleerr + c->getError()*c->getError());
    }
    if (counter) {
      //do not take scale uncertainties into account (yet). Will be done later if useScaleUncert == true.
      if (simpleUseScaleUncertainty)
        counter->add(cnt,scale,scaleerr,0.,true); 
      else
        counter->add(cnt,scale,0.,0.,false); 
    } else {
      counter = new TQCounter(cnt);
      if (simpleUseScaleUncertainty)
        counter->scale(scale,scaleerr,true);
      else
        counter->scale(scale,0.,false);
    }
    nElements++;
  }
  if (advancedUncertaintyTreatment && nElements>0) {
    double correlationAux = 0.; //auxillary variable to retrieve correlations between different paths
    double uncertSq = 0;
    TQListIterator itr1(list);
    while(itr1.hasNext()){
      TList* l1 = itr1.readNext();
      if(!l1 || l1->GetSize() < 2) continue;
      TQFolder * oFolder1 = dynamic_cast<TQFolder*>(l1->At(0));
      for (int j=1; j<l1->GetSize();j++) { //loop over entries in current sublist (index=0 is a TQFolder! It is the folder from which the (base) counter was retrieved.)
        TQCounter* cnt1 = dynamic_cast<TQCounter*>(l1->At(j));
        if(!cnt1) continue;
        TQListIterator itr2(list);
        while(itr2.hasNext()){
          TList* l2 = itr2.readNext();
          if(!l2 || l2->GetSize() < 2) continue;
          TQFolder * oFolder2 = dynamic_cast<TQFolder*>(l1->At(0));
          
          for (int l=1;l<l2->GetSize();l++) { //second loop over sublist (double loop since we need to examine each combination. Each element could be correlated to any other element)
            TQCounter* cnt2 = dynamic_cast<TQCounter*>(l2->At(l));
            if(!cnt2) continue;
            
            double tempErr = 1.;
            //now do stuff
            if (l == 1 && j == 1 ) { //this part deals with correlations between base counters (i.e. non-scale factors)
              if ( cnt1 != cnt2 ) { //if the counters are the same, we have full correlation anyways.
                //if we have a self-correlation term, the correlation is 1, so we don't need to apply an additional factor
                if (useManualCorrelations && TQStringUtils::equal(cnt1->GetName(),cnt2->GetName()) && areFoldersCorrelated(oFolder1,oFolder2,correlationAux) ) {
                  tempErr*=correlationAux;//insert correlation coefficient
                }
              }
            } else if (!useScaleUncertainty) continue; //if no inclusion of scale factor uncertainties is desired and we don't deal with base counter uncertainties/correlations, let's just skipp this part.
           
            
            
            //the structure of terms processed here is a sum of products. Differentiating  w.r.t. any of the contributions and multiplying with the respective element of the covariance matrix (here realized as: element(covariance matrix) * sigma) therefore has the form
            // x1 * x2 * ... * sigma_xi * ... * xN
            // hence, the contribution to the resulting uncertainty from the current pair of two quantities is
            // (x1 * x2 * ... * sigma_xi * ... * xN) * (y1 * y2 * ... * sigma_yi * ... * yN)
            
            for (int i = 1; i<l1->GetSize(); i++) {
              TQCounter* cnt = dynamic_cast<TQCounter*>(l1->At(i));
              if(!cnt) continue;
              tempErr *= ( j == i ? cnt->getError() : cnt->getCounter() );
            }
            for (int k = 1; k<l2->GetSize(); k++) {
              TQCounter* cnt = dynamic_cast<TQCounter*>(l2->At(k));
              if(!cnt) continue;
              tempErr *= ( l == k ? cnt->getError() : cnt->getCounter() ) ;
            }
            //The title of scale factors serves as a global identifier and is used to retrieve information about the correlation of the two counters in question here
            TString id1 = cnt1->GetTitle();
            TString id2 = cnt2->GetTitle();
            //the following treatment is only applicable for scale factors, not for the original counters (it is assumed that the original counters are uncorrelated with their scale factors. This is not always true, however, we currently have no way to determine these correlations and/or reasonably store/retrieve it)
            if (l>1 && j>1 && !(id1.IsNull() || id2.IsNull()) ) {
              //for each counter we have a folder (name given by id1, i.e., the counters title) filled with counters holding the correlation values w.r.t. other scale factors. Note that this is also used to correlate identical scale factors which are stored at multiple places in the sample folder structure (plus: this automatically also includes the contributions from the diagonal entries of the correlation/covariance matrix)!
              TQFolder* f1 = this->getCorrelationFolderByIdentifier(id1);
              if (!f1) {
                //print an error message if the correlation folder could not be found 
               ERRORclass("Failed to retrieve primary correlation folder!");
               return -1;
              } else if (f1->getObject(id2)) {
                double corr = (dynamic_cast<TQCounter*>( f1->getObject(id2) ))->getCounter();
                #ifdef _DEBUG_
                //debug only code: check that the transposed element in the correlation matrix is equal to the one already retrieved
                TQFolder* f2 = this->getCorrelationFolderByIdentifier(id2);
                if (!f2) WARNclass("Unable to retrieve secondary correlation folder (id '%s')", id2.Data());
                else if (f2->getObject(id1) && !TMath::AreEqualAbs(corr, (dynamic_cast<TQCounter*>( f2->getObject(id1) ))->getCounter() , std::numeric_limits<double>::epsilon()) ) {
                  WARNclass("Correlations do not match: cor(x1,x2) = %d, cor(x2,x1) = %d", corr, (dynamic_cast<TQCounter*>( f2->getObject(id1) ))->getCounter() );
                } else if (!f2->getObject(id1)) {
                  WARNclass("Unable to retrieve second correlation counter (id '%s')",id1.Data());
                }
                #endif
                tempErr *= corr;
              } else {
                //fall back solution: at least try to correlate identical counters. This also offers support for some simple (manual) implementation of correlations by setting the title of different scale factors to the same value
                WARNclass("Failed to retrieve correlation counter from folder '%s' for NF counters with identifiers (titles) %s and %s!",f1->getPath().Data(),id1.Data(),id2.Data());
                if (TQStringUtils::equal(id1,id2)) {
                  WARNclass("identifiers are equal, assuming full correlation (100%)");
                } else {
                  WARNclass("identifiers are equal, assuming no correlation (0%)");
                  tempErr *= 0;
                }
              } 
              
            } else if (l>1 && j>1 ) { //we are dealing with scale factors and at least one of the two elements in question has an empty title. This makes matching a bit hard, so we ulimatelly fall back to only consider this contribution if it is a diagonal term in the sense that the two pointers match, otherwise we consider the correlation to be zero.
               if (cnt1 != cnt2) tempErr *= 0;
              //the correlation or not correlation of the base counters is already taken care of before
            }
            
            //add this term to (squared) uncertainty
            uncertSq += tempErr;
          }
        }
      }
    }
    counter->setErrorSquared(uncertSq);
  }
  return nElements;
}

//__________________________________________________________________________________|___________

TQCounter * TQSampleDataReader::getCounter(const TString& path,const TString& name, const TString& options, TList * sfList) {
  // retrieve a counter path/name with the given options
  TQTaggable * tagsOptions = new TQTaggable(options);
  TQCounter * counter = getCounter(path, name, tagsOptions, sfList);
  delete tagsOptions;
  return counter;
}

//__________________________________________________________________________________|___________

TQTable * TQSampleDataReader::getEventlist(const TString& path, const TString& name, const TString& options, TList * sfList) {
  // retrieve an eventlist path/name with the given options
  TQTaggable * tagsOptions = new TQTaggable(options);
  TQTable * evtlist = getEventlist(path, name, tagsOptions, sfList);
  delete tagsOptions;
  return evtlist;
}

//__________________________________________________________________________________|___________

TQTable * TQSampleDataReader::getEventlist(const TString& path, const TString& name, TQTaggable * options, TList * sfList) {
  // retrieve an eventlist path/name with the given options
  TQTable* evtlist = dynamic_cast<TQTable*>(this->getElement<TQTable>(path, name, this->f_pathEventlists, options, sfList));
  if(!evtlist){
    TList* alt = dynamic_cast<TList*>(this->getElement<TQTable>(path, name, this->f_pathEventlists, options, sfList));
    if(alt){
      evtlist = new TQTable(alt);
      delete alt;
    }
  }
  if(!evtlist) return NULL;
  //@tag:recognizeIntegers: automatically recognize integer-valued columns and format them accordingly (default:true)
  if(options->getTagBoolDefault("recognizeIntegers",true)){
    for(int i=0; i<evtlist->getNcols(); ++i){
      bool isInt = true;
      for(int j=1; j<evtlist->getNrows(); ++j){
        const double val = evtlist->getEntryValue(j,i,0);
        const double diff = fabs((double(int(val))/val) - 1);
        if( diff > 10*std::numeric_limits<double>::epsilon() ) isInt = false;
      }
      if(isInt){
        for(int j=1; j<evtlist->getNrows(); ++j){
          const int val = evtlist->getEntryValue(j,i,0);
          evtlist->setEntryValue(j,i,val);
        }
      }
    }
  }
  return evtlist;
}

//__________________________________________________________________________________|___________

TQCounter * TQSampleDataReader::getCounter(const TString& path, const TString& name, TQTaggable * options, TList * sfList) {
  // retrieve a counter path/name with the given options
  TQCounter* retval = NULL;
  retval = dynamic_cast<TQCounter*>(this->getElement<TQCounter>(path, name, this->f_pathCutflow, options, sfList));
  if(!retval){
    TH1* hist = getHistogram(path,name,options,sfList);
    if(hist){
      double val,err;
      val = TQHistogramUtils::getIntegralAndError(hist,err);
      retval = new TQCounter(name,val,err,hist->GetEntries());
      retval->SetTitle(hist->GetTitle());
      delete hist;
    }
  }
  return retval;
}

//__________________________________________________________________________________|___________

TQPCA * TQSampleDataReader::getPCA(const TString& path, const TString& name, const TString& options, TList * sfList) {
  // retrieve a PCA object path/name with the given options
  TQTaggable * tagsOptions = new TQTaggable(options);
  TQPCA * grid = getPCA(path, name, tagsOptions, sfList);
  delete tagsOptions;
  return grid;
}


//__________________________________________________________________________________|___________

TQPCA * TQSampleDataReader::getPCA(const TString& path, const TString& name, TQTaggable * options, TList * sfList) {
  // retrieve a PCA object path/name with the given options
  return dynamic_cast<TQPCA*>(this->getElement<TQPCA>(path, name, this->f_pathPCA, options, sfList));
}


//__________________________________________________________________________________|___________

TTree * TQSampleDataReader::getTree(const TString& path, const TString& name, const TString& options, TList * sfList) {
  // retrieve a tree path/name with the given options
  // please note that you will need a writeable TDirectory open to allow
  // merging, because ROOT will not allow TTrees to be cloned/merged in memory!
  TQTaggable * tagsOptions = new TQTaggable(options);
  TTree * tree = getTree(path, name, tagsOptions, sfList);
  delete tagsOptions;
  return tree;
}


//__________________________________________________________________________________|___________

TTree * TQSampleDataReader::getTree(const TString& path, const TString& name, TQTaggable * options, TList * sfList) {
  // retrieve a tree path/name with the given options
  // please note that you will need a writeable TDirectory open to allow
  // merging, because ROOT will not allow TTrees to be cloned/merged in memory!
  return this->getElement<TTree>(path, name, this->f_pathTrees, options, sfList);
}

//__________________________________________________________________________________|___________

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

template <class CLASS>
int TQSampleDataReader::getElementWorker(TList* paths, TList* elements, CLASS*&element,
																				 const TString& subPath, TList* sfList,
																				 TQTaggable* options) {

  TList * elementList = this->collectElements<CLASS>(paths, elements, subPath, sfList, options);
  
  if(elementList) {
    int nSubElements = 0;
		nSubElements = this->sumElements(elementList,element,options);
    elementList->SetOwner(true);
    delete elementList;
    return nSubElements;
  }
  
  return -1;                                                            
}

#pragma GCC diagnostic pop

//__________________________________________________________________________________|___________

TObject * TQSampleDataReader::getElement(const TString& path, const TString& name, TClass* objClass, const TString& subPath, TQTaggable * options, TList * sfList){
  // this is a legacy wrapper for getElement that still uses the root TClass pointer functionality
#define CALL_GET_ELEMENT(CLASS) if(!objClass || objClass->InheritsFrom(CLASS::Class())) { TObject* obj = this->getElement<CLASS>(path,name,subPath,options,sfList); if(obj) return obj; }
  CALL_GET_ELEMENT(TQCounter);
  CALL_GET_ELEMENT(TProfile2D);
  CALL_GET_ELEMENT(TProfile);
  CALL_GET_ELEMENT(TH3);
  CALL_GET_ELEMENT(TH2);
  CALL_GET_ELEMENT(TH1);
  CALL_GET_ELEMENT(TQTable);
  CALL_GET_ELEMENT(TQPCA);
  CALL_GET_ELEMENT(THnBase);
  CALL_GET_ELEMENT(TGraph);
  CALL_GET_ELEMENT(TGraph2D);
  ERRORclass("cannot get element of unsupproted class '%s'",objClass->GetName());
  return NULL;
}

//__________________________________________________________________________________|___________

template <class T>
T * TQSampleDataReader::getElement(const TString& path, const TString& name, const TString& subPath, TQTaggable * options, TList * sfList) {
  if (f_Verbose > 1) {
    VERBOSEclass("path='%s', name='%s'",path.Data(),name.Data());
  }
  
  // clear the error message
  f_errMsg.clearMessages();

  // parse the paths
  TList * altPaths = parsePaths(path, 0, options->getTagStringDefault("prefix.path", ""));
  TList * altElements = parsePaths(name, 0, options->getTagStringDefault("prefix.name", ""));
  bool tmpSFList = false;
  if(!sfList){
    tmpSFList = true;
    sfList = new TList();
  }

  // stop if an error occured when parsing the paths
  if (!altPaths){
    ERRORclass("Failed to parse paths '%s'",path.Data());
    return 0;
  }
  if(!altElements) {
    ERRORclass("Failed to parse elements from '%s'",name.Data());
    return 0;
  }
  
 
  // temporary verbosity
  int tmpVerbose = f_Verbose;
  f_Verbose = options->getTagIntegerDefault("verbosity", f_Verbose);

  // the histograms to return
  T * element = 0;

  /* the number of elements added up
   * to obtain the final element */
  int nElements = 0;

  bool globalError = false;
  bool invalidPath = true;
  TString errorMsg;

  // loop over alternative paths
  int iAltPaths = -1;
  while (!globalError && invalidPath && ++iAltPaths < altPaths->GetEntries()) {
    invalidPath = false;
    nElements = 0;

    // clear histograms
    delete element;
    element = 0;

    // loop over the list of paths of this alternative
    TList * paths = (TList*)altPaths->At(iAltPaths);
    
    // loop over the list of element
    int iAltElement = -1;
    while (!globalError && !invalidPath && ++iAltElement < altElements->GetEntries()) {
      TList * elements = (TList*)altElements->At(iAltElement);
      
      // read and add element
      int nSubElements = 0;
      
      nSubElements = this->getElementWorker<T>(paths, elements, element, subPath, sfList, options);
      
      if (nSubElements < 0) {
        // an error occured: stop without returning an element (an error message was already compiled)
        nElements = 0;
        globalError = true;
      } else if (nSubElements == 0) {
        /* histogram not found: compile an error
         * message and go to next path alternative */
        errorMsg = TString::Format("no elements found in path variant %d, element variant %d.", iAltPaths, iAltElement);
        nElements = 0;
        invalidPath = true;
      } else {
        nElements+=nSubElements;
      }
    }
  }
  if(f_Verbose > 3){
    VERBOSEclass("done collecting elements, calculating style");
  }
  if(nElements > 0){
    this->applyStyleToElement(element,sfList,options);
  }
 
  // delete the parsed paths
  delete altPaths;
  delete altElements;

  if(tmpSFList) delete sfList;

  if (nElements == 0) {
    /* an error occured: delete the histograms
     * that may have been produced up to now */
    delete element;
    element = 0;
  }

  if (invalidPath)
    setErrorMessage(__FUNCTION__,errorMsg);

  // restore original verbosity
  if(f_Verbose > 4) {
    if(element) VERBOSEclass("returning element '%s'",element->GetName()); else VERBOSEclass("returning NULL");
  }
  f_Verbose = tmpVerbose;
  return element;
 
}

//__________________________________________________________________________________|___________

void TQSampleDataReader::applyStyleToElement(TObject* element, TCollection* sfList, TQTaggable* options){
  TQFolder* baseSF = this->f_baseSampleFolder->findCommonBaseFolder(sfList,true);
  if(baseSF){
    if(element->InheritsFrom(TH1::Class())){
      if(f_Verbose > 3){
        VERBOSEclass("applying style to %s: contributing folders are",element->GetName());
        TQFolderIterator fitr(sfList);
        while(fitr.hasNext()){
          TQFolder* f = fitr.readNext();
          if(!f) continue;
          VERBOSEclass("\t%s",f->getPathWildcarded().Data());
        }
        VERBOSEclass(" ==> applying tags from %s",baseSF->getPath().Data());
        VERBOSEclass("\t%s",baseSF->exportTagsAsString().Data());
      }
      TString styleScheme = options ? options->getTagStringDefault("styleScheme", f_styleScheme) : f_styleScheme;
      TQHistogramUtils::applyStyle((TH1*)(element), baseSF, styleScheme,true);
    } else if (element->InheritsFrom(TNamed::Class())){
      // apply style: the histogram title
      TString title;
      if (baseSF->getTagString("style.title", title)) {
        ((TNamed*)(element))->SetTitle(title.Data());
      }
    }
  }
}

//__________________________________________________________________________________|___________

void TQSampleDataReader::addObjectNames(TQSampleFolder * sampleFolder, const TString& objectPath,
                                        TClass * objectClass, TList * &objectNames, TList * sfList, const TString& filterScheme) {

  // stop if sample folder is invalid
  if (!sampleFolder) {
    return;
  }

  // get the list of objects in this sample folder...
  TList * objects = sampleFolder->getListOfObjectNames(objectClass, true, objectPath);

  // ...and add them to the list
  if (objects) {

    // make sure the resulting list exists
    if (!objectNames) {
      objectNames = new THashList();
      objectNames->SetOwner(true);
    }

    // iterate over object names and add them to the
    // resulting list unless it is already in the list
    TQIterator itr(objects, true);
    while (itr.hasNext()) {
      TObject * obj = itr.readNext();
      if (objectNames->FindObject(obj->GetName())) {
        delete obj;
      } else {
        objectNames->Add(obj);
      }
    }
  }
 
  // loop over sub sample folders and ...
  TQSampleFolderIterator itr(sampleFolder->getListOfSampleFolders("?"), true);
  while (itr.hasNext()) {
    // ... recursively add object names in sub sample folders
    this->addObjectNames(itr.readNext(), objectPath, objectClass,objectNames, sfList, filterScheme);
  }
}


//__________________________________________________________________________________|___________

void TQSampleDataReader::addObjectNames(const TString& path, const TString& objectPath,
                                        TClass * objectClass, TList * &objectNames, TList * sfList, const TString& filterScheme) {

  // get the list of sample folders matching path
  TQSampleFolderIterator itr(this->getListOfSampleFolders(path));
  while(itr.hasNext()){
    TQSampleFolder* sf = itr.readNext();
    if(!sf) continue;
    DEBUGclass("adding '%s' object names for '%s'",objectClass->GetName(),sf->getPath().Data());
    this->addObjectNames(sf, objectPath, objectClass, objectNames, sfList, filterScheme);
  }
}


//__________________________________________________________________________________|___________

TList * TQSampleDataReader::getBaseScaleFactors(TQSampleFolder * sampleFolder, const TString& path, const TString& scaleScheme) {
  // dump some info text if in verbose mode
  if (f_Verbose > 3 && sampleFolder) {
    // the method headline
    VERBOSEclass("path='%s'",sampleFolder->getPath().Data());
  }
 
  // the global scale factor
  TList * scaleList = new TList();
  scaleList->SetOwner(false);
 
  // in local mode the base sample folder must be the base of <sampleFolder>
  if (f_localMode && (!f_baseSampleFolder || !f_baseSampleFolder->isBaseOf(sampleFolder)))
    return scaleList;
 
  // the pointer to the current sample folder
  TQSampleFolder * sf = sampleFolder;
 
  while (sf) {
 
    /* we are only intersted in scale factors of the
       base folders, thus skip the starting sample folder */
    if (sf != sampleFolder) {
 
      // get the local scale factor
      //TQCounter * localScale = new TQCounter("localScale",1.,0.);
 
      // dump some info text if in verbose mode
      if (f_Verbose > 3) {
        VERBOSEclass("looking at folder '%s'",sf->getPath().Data());
      }
 
      TString scaleFactorPath = path;
 
      while (!scaleFactorPath.IsNull()) {
        TQCounter * localScaleItem = sf->getScaleFactorCounterInternal(scaleScheme + ":" + scaleFactorPath);
 
        scaleFactorPath = TQFolder::getPathWithoutTail(scaleFactorPath);
        if (!localScaleItem) {
          continue;
        }
        if (f_Verbose > 3) {
          VERBOSEclass("at path '%s', found '%g'",scaleFactorPath.Data(),localScaleItem->getCounter());
        }
        scaleList->Add(localScaleItem);
      }
    }
 
    // get base sample folder to step up to
    sf = sf->getBaseSampleFolder();
 
    /* stop at the root sample folder or before
     * the base sample folder in local mode */
    if (f_localMode && sf == f_baseSampleFolder) {
      sf = NULL;
    }
  }
 
  // return the total scale factor list
  return scaleList;
}


//__________________________________________________________________________________|___________

bool TQSampleDataReader::hasHistogram(const TString& path, const TString& name, const TString& options) {
  // Tries to retrieve the histogram <name> from <path> applying options <options>
  // and returns true in case of success or false otherwise.

  // try to retrieve the histogram
  TH1 * histo = getHistogram(path, name, options);

  if (histo) {
    // we are not interested in the histogram itself => delete it
    delete histo;
    return true;
  } else {
    // failed to retrieve the histogram
    return false;
  }
}


//__________________________________________________________________________________|___________

bool TQSampleDataReader::hasCounter(const TString& path, const TString& name, const TString& options) {
  // Tries to retrieve the counter <name> from <path> applying options <options>
  // and returns true in case of success or false otherwise.

  // try to retrieve the counter
  TQCounter * counter = getCounter(path, name, options);

  if (counter) {
    // we are not interested in the counter itself => delete it
    delete counter;
    return true;
  } else {
    // failed to retrieve the counter
    return false;
  }
}


//__________________________________________________________________________________|___________

TList * TQSampleDataReader::getListOfHistogramNames(const TString& path, TList * sfList) {
  // Returns a list (instance of TList* with instances of TObjString*) of the names
  // of all histograms available in the sample folder structure referred to by <path>
  // or a NULL pointer in case of failure or if no histogram is available. Please
  // note: the user is responsible to delete the list.

  // the list to return
  TList * objectNames = NULL;

  // add histogram names to the list
  this->addObjectNames(path, f_pathHistograms, TH1::Class(), objectNames, sfList, "");

  // return the list
  return objectNames;
}


//__________________________________________________________________________________|___________

TList * TQSampleDataReader::getListOfObjectNames(TClass* objClass, const TString& subpath, const TString& path, TList * sfList){

  // the list to return
  TList * objectNames = 0;

  // add the object names to the list
  this->addObjectNames(path, subpath, objClass, objectNames, sfList, "");

  // return the list
  return objectNames;
}



//__________________________________________________________________________________|___________

TList * TQSampleDataReader::getListOfCounterNames(const TString& path, TList * sfList) {
  // Returns a list (instance of TList* with instances of TObjString*) of the names
  // of all counter available in the sample folder structure referred to by <path>
  // or a NULL pointer in case of failure or if no counter is available. Please
  // note: the user is responsible to delete the list.

  // the list to return
  TList * objectNames = 0;

  // add counter names to the list
  this->addObjectNames(path, f_pathCutflow, TQCounter::Class(), objectNames, sfList, "");

  // return the list
  return objectNames;
}

//__________________________________________________________________________________|___________

TList * TQSampleDataReader::getListOfEventlistNames(const TString& path, TList * sfList) {

  // the list to return
  TList * objectNames = 0;

  // add the object names to the list
  this->addObjectNames(path, f_pathEventlists, TList::Class(), objectNames, sfList, "");
  this->addObjectNames(path, f_pathEventlists, TQTable::Class(), objectNames, sfList, "");

  // return the list
  return objectNames;
}

//__________________________________________________________________________________|___________

void TQSampleDataReader::printListOfHistogramLocations(const TString& name) {
  TList* sfList = new TList();
  TH1 * histo = getHistogram(".", name, "", sfList);
  delete histo;
  TQIterator itr(sfList,true);
  while(itr.hasNext()){
    TQSampleFolder* sf = dynamic_cast<TQSampleFolder*>(itr.readNext());
    if(!sf) continue;
    TH1* hist = dynamic_cast<TH1*>(sf->getObject(name,f_pathHistograms));
    std::cout << TQStringUtils::fixedWidth(sf->getPath(),40,"l");
    if(hist) std::cout << TQHistogramUtils::getDetailsAsString(hist);
    std::cout << std::endl;
  }
}

//__________________________________________________________________________________|___________

void TQSampleDataReader::printListOfHistograms(TString options) {

  // temporarily disable verbose mode
  int tmpVerbose = f_Verbose;
  f_Verbose = 0;

  // read the path to print
  TString path = TQStringUtils::readPrefix(options, ":", ".");

  // get the list of histogram names matching <path>
  TList * names = this->getListOfHistogramNames(path);

  // ===== print header =====
  TString line;
  line.Append(TQStringUtils::fixedWidth("Histogram name", 50, "l"));
  line.Append(TQStringUtils::fixedWidth("Type", 8, "l"));
  line.Append(TQStringUtils::fixedWidth("Binning", 70, "l"));
  line.Append(TQStringUtils::fixedWidth("# Folders", 12));
  std::cout << TQStringUtils::makeBoldWhite(line) << std::endl;
  std::cout << TQStringUtils::makeBoldWhite(TQStringUtils::repeat("=", TQStringUtils::getWidth(line))) << std::endl;

  // stop if no histograms were found
  if (!names) {
    return;
  }

  // the histogram name prefix
  TString prefix;

  // sort the list
  names->Sort();

  // loop over list of histogram names
  TQIterator itr(names,true);
  while (itr.hasNext()){
    TObject* obj = itr.readNext();
    // get the histogram name
    TString name = obj->GetName();

    // group histogram prefixes
    TString newPrefix = TQFolder::getPathWithoutTail(name);
    if (newPrefix.IsNull()) {
      newPrefix = ".";
    }
    if (newPrefix.CompareTo(prefix) != 0) {
      prefix = newPrefix;
      std::cout << TString::Format("%*s\033[1;34m%s\033[0m/", 0, "", prefix.Data()) << std::endl;
    }

    // get the histogram and the list of contributing sample folders
    TList * sfList = new TList();
    TH1 * histo = getHistogram(path, name, "", sfList);

    // remember the error message for this histogram
    TString errorMsg = this->getErrorMessage();

    // get the name of the histogram class
    TString histoType;
    if (histo)
      histoType = histo->IsA()->GetName();

    // prepare line to print
    line.Clear();

    if (histo) {

      // histogram details
      TString histoDetails = TQHistogramUtils::getDetailsAsString(histo, 2);

      line.Append(TQStringUtils::fixedWidth(
                                            TQStringUtils::repeat(" ", 2 * TQFolder::countPathLevels(prefix, false)) +
                                            TQFolder::getPathTail(name), 50,"l"));
      line.Append(TQStringUtils::fixedWidth(histoType, 8, "l"));
      line.Append(TQStringUtils::fixedWidth(histoDetails, 70, "l"));
      line.Append(TQStringUtils::fixedWidth(TString::Format("%d", sfList->GetEntries()), 12,"l"));
    } else {
      line.Append(TQStringUtils::fixedWidth(
                                            TQStringUtils::repeat(" ", 2 * TQFolder::countPathLevels(prefix)) +
                                            "\033[0;31m" + TQFolder::getPathTail(name) + "\033[0m", 50, "l"));
      line.Append(TQStringUtils::fixedWidth("--", 8, "l"));
      line.Append(TQStringUtils::fixedWidth(errorMsg, 70, "l"));
      line.Append(TQStringUtils::fixedWidth("--", 12,"l"));
    }

    // print line
    std::cout << line << std::endl;

    // delete histogram
    delete histo;
    // delete list of sample folders
    delete sfList;
  }

  // reactivate verbose mode (of set before)
  f_Verbose = tmpVerbose;

}

//__________________________________________________________________________________|___________

void TQSampleDataReader::printListOfCounterLocations(const TString& name) {
  TList* sfList = new TList();
  TQCounter* cnt = this->getCounter(".", name, "", sfList);
  delete cnt;
  TQIterator itr(sfList,true);
  while(itr.hasNext()){
    TQSampleFolder* sf = dynamic_cast<TQSampleFolder*>(itr.readNext());
    if(!sf) continue;
    TQCounter* cnt = dynamic_cast<TQCounter*>(sf->getObject(name,f_pathCutflow));
    std::cout << TQStringUtils::fixedWidth(sf->getPath(),40,"l");
    if(cnt) cnt->getAsString();
    std::cout << std::endl;
  }
}

//__________________________________________________________________________________|___________

void TQSampleDataReader::printListOfCounters(TString options) {

  // temporarily disable verbose mode
  int tmpVerbose = f_Verbose;
  f_Verbose = 0;

  // read the path to print
  TString path = TQStringUtils::readPrefix(options, ":", ".");

  // get the list of counter names matching <path>
  TList * names = this->getListOfCounterNames(path);

  // ===== print header =====
  TString line;
  line.Append(TQStringUtils::fixedWidth("Counter name", 30, "l"));
  line.Append(TQStringUtils::fixedWidth("Counter title", 50, "l"));
  line.Append(TQStringUtils::fixedWidth("Details", 50));
  line.Append(TQStringUtils::fixedWidth("# Folders", 10));
  std::cout << TQStringUtils::makeBoldWhite(line) << std::endl;
  std::cout << TQStringUtils::makeBoldWhite(TQStringUtils::repeat("=", TQStringUtils::getWidth(line))) << std::endl;

  // stop if no counters were found
  if (!names) {
    return;
  }

  // loop over list of counter names
  TQIterator itr(names->MakeIterator(kIterBackward),true);
  while (itr.hasNext()){
    TObject* obj = itr.readNext();
    // get the counter name
    TString name(obj->GetName());

    // get the counter and the list of contributing sample folders
    TList * sfList = new TList();
    TQCounter * cnt = getCounter(path, name, "", sfList);
    TString title(cnt->GetTitle());

    // remember the error message for this counter
    TString errorMsg = this->getErrorMessage();

    // prepare line to print
    line.Clear();

    if (cnt) {
      // counter details
      TString details = cnt->getAsString();

      line.Append(TQStringUtils::fixedWidth(TQStringUtils::makeBoldWhite(name),30,"l"));
      line.Append(TQStringUtils::fixedWidth(title,50,"l"));
      line.Append(TQStringUtils::fixedWidth(details, 50, "l"));
      line.Append(TQStringUtils::fixedWidth(TString::Format("%d", sfList->GetEntries()), 10,"l"));
    } else {
      line.Append(TQStringUtils::fixedWidth(TQStringUtils::makeBoldWhite(name), 60, "l"));
      line.Append(TQStringUtils::fixedWidth(errorMsg, 50, "l"));
      line.Append(TQStringUtils::fixedWidth("--", 10,"c"));
    }

    // print line
    std::cout << line << std::endl;

    // delete counter
    delete cnt;
    // delete list of sample folders
    delete sfList;
  }
  delete names;

  // reactivate verbose mode (of set before)
  f_Verbose = tmpVerbose;

}


//__________________________________________________________________________________|___________

TQSampleDataReader::~TQSampleDataReader() {
}

//__________________________________________________________________________________|___________

TFolder* TQSampleDataReader::exportHistograms(const TString& sfpath, const TString& tags){
  // export all histograms from a certain path to a TDirectory object
  TQTaggable taggable(tags);
  return this->exportHistograms(sfpath,taggable);
}

//__________________________________________________________________________________|___________

TFolder* TQSampleDataReader::exportHistograms(const TString& sfpath, TQTaggable& tags){
  // export all histograms from a certain path to a TFolder object
  // tags will be passed along to the histogram retrieval
  // the tag "filter" allows to filter the histogram names
  // if the tag "cutdiagram" is set, it will be used to retrieve a cut hierarchy from a folder
  // under the path given as value to this tag and export a cut diagram as TNamed
  // tags prefixed with "cutdiagram." will be passed through to the cut diagram making
  TList* sflist = NULL;
  TQSampleFolder* sf = this->f_baseSampleFolder->getSampleFolder(sfpath);
  if(!sf){
    ERRORclass("invalid path '%s'",sfpath.Data());
    return NULL;
  }
  TList* names = this->getListOfHistogramNames();
  if(!names || names->GetEntries() < 1){
    if(names) delete names;
    ERRORclass("no histograms found in '%s'",sfpath.Data());
    return NULL;
  }
  TQFolder* basesf = this->f_baseSampleFolder->findCommonBaseFolder(sflist);
  TFolder* base = basesf ? new TFolder(basesf->GetName(),basesf->getTagStringDefault("style.default.title",basesf->GetTitle())) : new TFolder("histograms",sfpath);
  TQIterator itr(names);
  TString filter = tags.getTagStringDefault("filter","*");
  while(itr.hasNext()){
    TFolder* current = base;
    TObject* obj = itr.readNext();
    if(!obj) continue;
    TString path(obj->GetName());
    std::cout<< path << std::endl;
    if(!TQStringUtils::matches(path,filter)) continue;
    TH1* hist = this->getHistogram(sfpath,path,&tags);
    if(!hist){
      WARNclass("unable to retrieve histogram '%s'",path.Data());
      continue;
    }
    TString name = TQFolder::getPathTail(path);
    while(!path.IsNull()){
      TString fname = TQFolder::getPathHead(path);
      TFolder* newFolder = dynamic_cast<TFolder*>(current->FindObject(fname));
      if(!newFolder){
        newFolder = current->AddFolder(fname,fname);
      }
      current = newFolder;
    }
    current->Add(hist);
  }
  delete names;
  delete sflist;
  // create the cut diagram
  TString cutinfo;
  if(tags.getTagString("cutdiagram",cutinfo)){
    TQFolder* f = this->f_baseSampleFolder->getFolder(cutinfo);
    if(!f){
      WARNclass("unable to load cut hierarchy from folder '%s' - no such folder!",cutinfo.Data());
    } else {
      TQTaggable cutdiagramtags;
      cutdiagramtags.importTagsWithoutPrefix(tags,"cutdiagram.");
      TQCut* cut = TQCut::importFromFolder(f);
      if(!cut){
        WARNclass("unable to load cut hierarchy from folder '%s' - invalid folder structure!",cutinfo.Data());
      } else {
        TString diagram = cut->writeDiagramToString(cutdiagramtags);
        TNamed* cutDiagram = new TNamed("cutdiagram.tex",diagram.Data());
        base->Add(cutDiagram);
        delete cut;
      }
    }
  }
  return base;
}

//__________________________________________________________________________________|___________

bool TQSampleDataReader::exportHistograms(TDirectory* d, const TString& sfpath, const TString& tags){
  // export all histograms from a certain path to a TDirectory object
  // tags will be passed along to the histogram retrieval
  // the tag "filter" allows to filter the histogram names
  TQTaggable t(tags);
  return exportHistograms(d,sfpath,t);
}

//__________________________________________________________________________________|___________

bool TQSampleDataReader::exportHistograms(TDirectory* d, const TString& sfpath, TQTaggable& tags){
  // export all histograms from a certain path to a TDirectory object
  // tags will be passed along to the histogram retrieval
  // the tag "filter" allows to filter the histogram names
  TList* names = this->getListOfHistogramNames();
  if(!names || names->GetEntries() < 1){
    if(names) delete names;
    ERRORclass("no histograms found in '%s'",sfpath.Data());
    return false;
  }
  TQIterator itr(names,true);
  TString filter = tags.getTagStringDefault("filter","*");
  while(itr.hasNext()){
    TObject* obj = itr.readNext();
    if(!obj) continue;
    TString path(obj->GetName());
    if(!TQStringUtils::matches(path,filter)) continue;
    TH1* hist = this->getHistogram(sfpath,path,&tags);
    if(!hist){
      WARNclass("unable to retrieve histogram '%s'",path.Data());
      continue;
    }
    if (tags.getTagBoolDefault("useShortNames",false)) TQStringUtils::readPrefix(path,"/");
    TString name = tags.getTagBoolDefault("useShortNames",false) ? path : TQFolder::makeValidIdentifier(path,"_");
    hist->SetName(name);
    d->Add(hist);
  }
  return true;
}


//__________________________________________________________________________________|___________

bool TQSampleDataReader::exportHistogramsToFile(const TString& fname, const TString& sfpath, const TString& tags){
  // export all histograms from a certain path to a file with the given name
  // the file will be opened with the RECREATE option, and if ensureDirectory=true is passed,
  // missing parent directories will be created
  TQTaggable taggable(tags);
  return this->exportHistogramsToFile(fname,sfpath,taggable);
}

//__________________________________________________________________________________|___________

bool TQSampleDataReader::exportHistogramsToFile(const TString& fname, const TString& sfpath, TQTaggable& tags){
  // export all histograms from a certain path to a file with the given name
  // the file will be opened with the RECREATE option, and if ensureDirectory=true is passed,
  // missing parent directories will be created
  bool flat = tags.getTagBoolDefault("flat",false);
  TFolder* result = flat ? NULL : this->exportHistograms(sfpath,tags);
  if(result || flat){
    if(tags.getTagBoolDefault("ensureDirectory",false)){
      if(!TQUtils::ensureDirectoryForFile(fname)){
        ERRORclass("unable to ensure directory for file '%s'!",fname.Data());
        return false;
      }
    }
  } else {
    return false;
  }
  TFile* f = TFile::Open(fname,"RECREATE");
  if(!f || !f->IsOpen()){
    ERRORclass("unable to open file '%s' for writing",fname.Data());
    if(f) delete f;
    return false;
  }
  bool retval = true;
  if(flat){
    retval = this->exportHistograms(f,sfpath,tags);
  } else {
    f->Add(result);
  }
  f->Write();
  f->Close();
  return retval;
}

//__________________________________________________________________________________|___________

TQFolder* TQSampleDataReader::getCorrelationFolderByIdentifier(const TString& id, bool forceUpdate) {
//resolves a sample identifier for its path in the folder hierarchy
  TString path;
  if ( !(f_identifierToFolderMap.count(id) > 0) || !f_identifierToFolderMap.count(id) || forceUpdate ) {
    f_identifierToFolderMap[id] = this->f_baseSampleFolder->getFolder(TQFolder::concatPaths("*",".correlations",id));
    DEBUGclass("Added folder with path '%s' to folder map (id: '%s')",f_identifierToFolderMap[id]->getPath().Data(), id.Data());
  }
  return f_identifierToFolderMap[id];
}

//__________________________________________________________________________________|___________

void TQSampleDataReader::copyData(const TString& source, const TString&target, const TString&options){
  // copy the data from one part of the sample folder to another
  TQSampleFolder* targetFolder = this->f_baseSampleFolder->getSampleFolder(target+"+");
  if(!targetFolder) return;
  TQIterator itr(this->getListOfHistogramNames(),true);
  while(itr.hasNext()){
    TObject* histname = itr.readNext();
    if(!histname) continue;
    TString path(histname->GetName());
    TH1* hist = this->getHistogram(source,path,options);
    TString name(TQFolder::getPathTail(path));
    TQFolder* histograms = targetFolder->getFolder(TQFolder::concatPaths(this->f_pathHistograms,path)+"+");
    hist->SetName(name);
    histograms->addObject(hist);
  }
}
