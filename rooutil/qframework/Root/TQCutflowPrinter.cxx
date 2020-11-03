#include "QFramework/TQNamedTaggable.h"
#include "QFramework/TQSampleFolder.h"
#include "QFramework/TQCutflowPrinter.h"
#include "QFramework/TQStringUtils.h"
#include "QFramework/TQHistogramUtils.h"
#include "QFramework/TQCounter.h"
#include "TNamed.h"
#include "QFramework/TQUtils.h"
#include "QFramework/TQIterator.h"
#include "TList.h"
#include "TObjString.h"
#include "TObjArray.h"
#include "TMath.h"
#include <time.h> 

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

#include "QFramework/TQTable.h"
// #define _DEBUG_

#include "QFramework/TQLibrary.h"

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQCutflowPrinter:
//
// The TQCutflowPrinter class uses an TQSampleDataReader to obtain
// cutflow numbers to print cutflow tables in ordinary text style, in
// latex style or in HTML style.
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQCutflowPrinter)


//______________________________________________________________________________________________

TQCutflowPrinter::TQCutflowPrinter() : 
TQPresenter()
{
  // Default constructor of TQCutflowPrinter class:
}

//______________________________________________________________________________________________

TQCutflowPrinter::TQCutflowPrinter(TQSampleFolder * samples) : 
  TQPresenter(samples)
{
  // Constructor of TQCutflowPrinter class:
}

//__________________________________________________________________________________|___________

TQCutflowPrinter::TQCutflowPrinter(TQSampleDataReader * reader) : 
  TQPresenter(reader)
{
  // Constructor of TQCutflowPrinter class:
}

//__________________________________________________________________________________|___________

void TQCutflowPrinter::addCutflowCut(TString cutName, TString cutTitle, int sfPolicy) {
  // add a cut to be listed in the cutflow.

  /* the default title is the cut name itself */
  if (cutTitle.IsNull())
    cutTitle = cutName;
  TQNamedTaggable* cut = new TQNamedTaggable(cutName);
  cut->setTag(".name",TQStringUtils::trim(cutName));
  cut->setTag(".title",TQStringUtils::trim(cutTitle));
  cut->setTag(".sfPolicy",sfPolicy);
  fCuts->Add(cut);
}


//__________________________________________________________________________________|___________

void TQCutflowPrinter::addCutflowProcess(TString processName, TString processTitle) {
  // Add a process to be listed in the cutflow. The first argument (processName)
  // specifies the process path (e.g. "bkg/ee/top"), the second (optional) argument
  // may be used to specify a title shown in the table for this process. You can
  // specify additional format specifications (e.g. "bkg/ee/top,raw" to obtain raw
  // numbers).
  //
  // There are several additional numbers which can be plotted:
  //
  // - the ratio of two processes at a given cut stage
  //
  // addCutflowProcess("$ratio(data/ee,bkg/ee)");
  //
  // - the signal significance at a given cut stage
  //
  // addCutflowProcess("$signif(sig/ee,bkg/ee)");
  //
  // - the pull of two processes
  //
  // addCutflowProcess("$pull(data/ee,bkg/ee)");
  //

  TQNamedTaggable* process = new TQNamedTaggable(processName);
  process->SetName(processName);
  process->SetTitle(processTitle);

  TString processOptions = "";

  int commaPos = kNPOS;
  if ( processName.BeginsWith("$") ){
    commaPos = processName.Index(")");
    commaPos = processName.Index(",", commaPos);
  } else {
    commaPos = processName.Index(",");
  }
  if (commaPos != kNPOS) {
    processOptions = processName(commaPos + 1, processName.Length());
    processName.Remove(commaPos, processName.Length());
  }
 
  process->setTagString(".name",TQFolder::makeValidIdentifier(processName));

  bool removeSpecial = false;
  if (processName.BeginsWith("$ratio(")){
    process->setTagString(".special","ratio");
    removeSpecial=true;
  }
  if (processName.BeginsWith("$pull(")){
    process->setTagString(".special","pull");
    removeSpecial=true;
  }
  if (processName.BeginsWith("$signif(")){
    process->setTagString(".special","significance");
    removeSpecial=true;
  }

  TString special;
  if(removeSpecial){
    /* remove "$ratio(" or "$pull(" */
    TString function = processName(0, processName.Index("("));
    DEBUGclass("found special process '%s' with old notation",special.Data());
    processName.Remove(0, processName.Index("(") + 1);
    /* extract the two processes */
    Ssiz_t commaPos = processName.Index(",");
    Ssiz_t bracePos = processName.Index(")");
    TString process1 = processName(0, commaPos);
    TString process2 = processName(commaPos + 1, bracePos - commaPos - 1);
    
    process->setTagString(".path1",process1);
    process->setTagString(".path2",process2);
    process->setTagString(".path",process1+","+process2);
    //@tag: [.special] If this process tag is set, the process is marked as a special process. Possible values are "ratio", "pull" and "significance"
  } 
  //it seems, this code is never executed. TODO: remove if no problems appear.
  /*else if(process->getTagString(".special",special)){
    if(process->hasTagString(".path1") && process->hasTagString(".path2")){
      DEBUGclass("found fully specified special process '%s'",special.Data());
      // nothing to do here
    } else {
      DEBUGclass("found incomplete special process '%s', auto-fixing",special.Data());
      int commaPos = processName.First(",");
      TString process1 = processName(0, commaPos);
      TString process2 = processName(commaPos);
    }
  }*/
   else {
    process->setTagString(".path",TQStringUtils::trim(processName));
  }

  if(processOptions.Contains("raw",TString::kIgnoreCase)) {
    process->setTagBool(".showRaw",true);
  }
  if(processOptions.Contains("scaled",TString::kIgnoreCase)) {
    process->setTagBool(".showScaled",true);
  }
  if(processOptions.Contains("errors",TString::kIgnoreCase)) {
    process->setTagBool(".showErrors",true);
  }
  if(processOptions.Contains("sys",TString::kIgnoreCase)) {
    process->setTagString(".includeSysError","*");
  }

  process->setTagString(".title",TQStringUtils::trim(processTitle));
  fProcesses->Add(process);
}

//__________________________________________________________________________________|___________

TQTable * TQCutflowPrinter::createTable(const TString& tags) {
  // create a TQTable object

  /* create the taggable object and import the tags */
  TQTaggable taggable(tags);
 
  return this->createTable(taggable);
}

//__________________________________________________________________________________|___________

TQTable * TQCutflowPrinter::createTable(TQTaggable* tags) {
  // create a TQTable object
  if(tags) return this->createTable(*tags);
 
  TQTaggable newTags;
  return this->createTable(newTags);
}

//__________________________________________________________________________________|___________

bool TQCutflowPrinter::getScaleFactors(TQTaggable& tags, TQNamedTaggable* cut, TQNamedTaggable* process, double& number, double& error, bool& applied, bool& equal, TString& info){
  // retrieve scale factors for some cut/process combination

  // by using the TQSampleDataReader::getListOfSampleFolders
  // we profit from it's capabilities of arithmetic string expansion

  TString processName = tags.replaceInText(process->getTagStringDefault(".path",process->GetName()));
  TString cutName = tags.replaceInText(cut->getTagStringDefault(".name",cut->GetName()));
  
  bool foundNF = false;
	while(!foundNF && !processName.IsNull()) { //quick and dirty fix for missing NFs in cutflow (e.g. top NFs)
    TList* paths = fReader->getListOfSampleFolders(processName);
    if (!paths){
      ERRORclass("no such sample folder: '%s'",processName.Data());
      return false;
    }
    number = 1;
    error = 0;
    applied = false;
    equal = true;
    TQSampleFolderIterator itr(paths);
    while(itr.hasNext()){
      TQSampleFolder* sf = itr.readNext();
      if(!sf) continue;
      double thisSF, thisSFerr;
      DEBUGclass("getting scale factor for '%s' from '%s'",cutName.Data(),sf->getPath().Data());
      bool sfSet = sf->getScaleFactor(cutName,thisSF,thisSFerr,true);
      foundNF = sfSet || foundNF;
      DEBUGclass("retrieved scale factor %.3f +/- %.3f",thisSF,thisSFerr);
      if(!sfSet) continue;
      if(!info.IsNull()) info += ", ";
      info += sf->getPath() + TString::Format(":%g",thisSF);
      if (!applied){
        number = thisSF;
        error = thisSFerr;
        applied = true;
      } else {
        if (!TMath::AreEqualRel(thisSF, number, 10E-5)) equal = false;
      }
    } 
    delete paths;
  		if (!foundNF) processName = TQFolder::getPathWithoutTail(processName); //quick and dirty fix for missing NFs in cutflow
	}
  return true;
}


//__________________________________________________________________________________|___________

bool TQCutflowPrinter::getValues(TQTaggable& tags, TQNamedTaggable* cut, TQNamedTaggable* process, double& number, double& error, int& raw, TString& info, TString& defaultTitle){
  // retrieve counter values for some cut/process combination
  double scale = tags.getTagDoubleDefault ("scale",1.);

  TString processName = tags.replaceInText(process->getTagStringDefault(".path",process->GetName()));
  TString cutName = tags.replaceInText(cut->getTagStringDefault(".name",cut->GetName()));

  bool showNumbers = false;
  number = 0.;
  error = 0.;
  raw = 0 ;
  info = "";
  DEBUGclass("processName '%s' ",processName.Data());
  TString special;
  if (process->getTagString(".special",special)){
    if (special == "ratio"){
      TString path1(tags.replaceInText(process->getTagStringDefault(".numerator",process->getTagStringDefault(".path1",""))));
      TString path2(tags.replaceInText(process->getTagStringDefault(".denominator",process->getTagStringDefault(".path2",""))));
      TQCounter * cnt1 = fReader->getCounter(path1, cutName, &tags);
      TQCounter * cnt2 = fReader->getCounter(path2, cutName, &tags);
      if (cnt1 && cnt2) {
        DEBUG("making ratio");
        showNumbers = true;
        number = cnt1->getCounter() / cnt2->getCounter();
        error = number * TMath::Sqrt(TMath::Power(cnt1->getError() / cnt1->getCounter(), 2) + TMath::Power(cnt2->getError() / cnt2->getCounter(), 2));
        if(error != error) error = 0;
        raw = -1;
        info = cnt1->getAsString() + ", "+cnt2->getAsString() + ", " + info.Append(TString::Format("ratio=%g+/-%g",number,error));
      }
      if(cnt1) delete cnt1;
      if(cnt2) delete cnt2;
    } else if (special == "significance"){
      TString path1(tags.replaceInText(process->getTagStringDefault(".signal",process->getTagStringDefault(".path1",""))));
      TString path2(tags.replaceInText(process->getTagStringDefault(".background",process->getTagStringDefault(".path2",""))));
      TQCounter * cnt1 = fReader->getCounter(path1, cutName, &tags);
      TQCounter * cnt2 = fReader->getCounter(path2, cutName, &tags);
      if (cnt1 && cnt2) {
        DEBUG("making significance");
        showNumbers = true;
        number = TQHistogramUtils::getPoisson(cnt2->getCounter(),cnt1->getCounter());
        error = TQHistogramUtils::getPoissonError(cnt2->getCounter(), cnt1->getCounter(), cnt2->getError(), cnt1->getError());
        raw = -1;
        info = cnt1->getAsString() + ", "+cnt2->getAsString() +  ", " + TString::Format("signif.=%g+/-%g",number,error);
      }
      if(cnt1) delete cnt1;
      if(cnt2) delete cnt2;      
    } else if (special == "pull"){
      TString path1(tags.replaceInText(process->getTagStringDefault(".path1","")));
      TString path2(tags.replaceInText(process->getTagStringDefault(".path2","")));
      TQCounter * cnt1 = fReader->getCounter(path1, cutName, &tags);
      TQCounter * cnt2 = fReader->getCounter(path2, cutName, &tags);
      if (cnt1 && cnt2) {
        DEBUG("making pull");
        showNumbers = true;
        number = (cnt1->getCounter() - cnt2->getCounter()) /
          TMath::Sqrt(
                      TMath::Power(cnt1->getError(), 2) +
                      TMath::Power(cnt2->getError(), 2));
        raw = -1;
        error = std::numeric_limits<double>::quiet_NaN();
        info = cnt1->getAsString() + ", "+cnt2->getAsString() +  ", " + TString::Format("pull=%g",number);
      }
      if(cnt1) delete cnt1;
      if(cnt2) delete cnt2;
    } else if (special == "acceptance"){
      TString otherCut(tags.replaceInText(process->getTagStringDefault(".denominator","")));
      TQCounter * cnt1 = fReader->getCounter(processName, cutName, &tags);
      TQCounter * cnt2 = fReader->getCounter(processName, otherCut, &tags);
      if (cnt1 && cnt2) {
        DEBUG("making acceptance");
        showNumbers = true;
        number = cnt1->getCounter() / cnt2->getCounter();
        error = cnt1->getError() / cnt2->getCounter() * ( 1. - number);
        if(error != error) error = 0;
        raw = -1;
        info = cnt1->getAsString() + ", "+cnt2->getAsString() + ", " + info.Append(TString::Format("acceptance=%g+/-%g",number,error));
      }
      if(cnt1) delete cnt1;
      if(cnt2) delete cnt2;
    }
  } else {
    TQTaggable tmpTags(tags);
    tmpTags.importTags(process);
    tmpTags.importTags(cut);
    TQCounter * counter = fReader->getCounter(processName, cutName, &tmpTags);
    if (counter) {
      showNumbers = true;
      number = counter->getCounter() * scale;
      error = counter->getError() * scale;
      TString systID;
      if(process->getTagString(".includeSysError",systID)){
        TQFolder* sysFolder = this->fSystematics->getFolder(TQFolder::concatPaths(systID,cutName));
        if(sysFolder)
          error *= (1.+sysFolder->getTagDoubleDefault("yield",0));
      }
      raw = counter->getRawCounter();
      info = counter->getAsString();
      if(defaultTitle.IsNull()) defaultTitle = counter->GetTitle();
      /* delete the counter */
      delete counter;
    }
  }
  return showNumbers;
}

//__________________________________________________________________________________|___________

TString TQCutflowPrinter::makeSFCellContents(TQTaggable& tags, TQNamedTaggable* cut, TQNamedTaggable* process, TString& info){
  // create textual representation of a given scale factor cell, given by cutName and processName
  bool forceAllSf = tags.getTagBoolDefault ("style.forceSF",false);
  bool usePercNotn = tags.getTagBoolDefault ("style.usePercNotn",false);
  int sfPolicy = cut->getTagIntegerDefault(".sfPolicy",1);
  TString sfcell = "";

  TString processName = tags.replaceInText(process->getTagStringDefault(".path",process->GetName()));
  TString cutName = tags.replaceInText(cut->getTagStringDefault(".name",cut->GetName()));

  // by using the TQSampleDataReader::getListOfSampleFolders
  // we profit from it's capabilities of arithmetic string expansion
  TList* paths = fReader->getListOfSampleFolders(processName);
  if (paths){
    double * scaleFactor = 0;
    bool equal = true;
    for (int i = 0; i < paths->GetEntries(); i++) {
      TQSampleFolder* sf = dynamic_cast<TQSampleFolder*>(paths->At(i));
      if(!sf) continue;
      double thisSF = sf->getScaleFactor(cutName,true);
      if(!info.IsNull()) info += ", ";
      info += sf->getPath() + TString::Format(":%g",thisSF);
      if (!scaleFactor)
        scaleFactor = new double(thisSF);
      else if (!TMath::AreEqualRel(thisSF, *scaleFactor, 10E-5))
        equal = false;
    }
    if(scaleFactor){
      if(!TMath::AreEqualRel(*scaleFactor,1.0,0.005) || forceAllSf || (sfPolicy==2)){
        if (equal) {
          if (usePercNotn) {
            sfcell = TString::Format("%.2f", *scaleFactor);
          } else {
            sfcell = TString::Format("NF = %.2f", *scaleFactor);
          }
        } else {
          sfcell = "NFs applied";
        }
      } 
      delete scaleFactor;
    }
    delete paths;
  } else {
    ERRORclass("no such sample folder: '%s'",processName.Data());
  }
  return sfcell;
}

//__________________________________________________________________________________|___________

int TQCutflowPrinter::sanitizeProcesses() {
  // sanitize all processes 
  TQTaggableIterator itr(fProcesses);
  std::vector<TQNamedTaggable*> removals;
  int retval = 0;
  while(itr.hasNext()){
    TQNamedTaggable* process = itr.readNext();
    if(!process) continue;
    TString path,special;
    if(!process->getTagString(".path",path) && !process->getTagString(".special",special)){
      removals.push_back(process);
    } else if(!process->getTagString(".name",path)){
      if(path.IsNull()){
        process->setTagString(".name",TQFolder::makeValidIdentifier(special));
        process->SetName(special);
      } else {
        process->setTagString(".name",TQFolder::makeValidIdentifier(path));
        process->SetName(path);
      }
    }
  }
  for(size_t i=0; i<removals.size(); i++){
    fProcesses->Remove(removals[i]);
    delete removals[i];
    retval++;
  }
  return retval;
}

//__________________________________________________________________________________|___________

int TQCutflowPrinter::sanitizeCuts() {
  // sanitize all cuts 
  TQTaggableIterator itr(fCuts);
  std::vector<TQNamedTaggable*> removals;
  int retval = 0;
  while(itr.hasNext()){
    TQNamedTaggable* cut = itr.readNext();
    if(!cut) continue;
    TString name;
    if(!cut->getTagString(".name",name)){
      removals.push_back(cut);
    }
  }
  for(size_t i=0; i<removals.size(); i++){
    fCuts->Remove(removals[i]);
    delete removals[i];
    retval++;
  }
  return retval;
}


//__________________________________________________________________________________|___________

TQTable * TQCutflowPrinter::createTable(TQTaggable tags) {
  // create a TQTable object
  tags.importTags(this);

  if(!fCuts){
    ERRORclass("cannot create cutflow: no cuts scheduled");
    return NULL;
  }
  if(!fProcesses){
    ERRORclass("cannot create cutflow: no processes scheduled");
    return NULL;
  }

  DEBUGclass("retrieving tags");

  int cellWidth = -1;
  int firstColumnWidth = -1;
  int precision = 2;
  double scale = 1.;
  bool hideAllSf = false;
  bool showAllSf = false;
  bool forceAllSf = false;
  TString nfPrefixText = this->getTagStringDefault("nfPrefixText","NF = ");
  this->getTagString("sfPrefixText",nfPrefixText);

  //@tag: [style.cutline,style.cellWidth,style.firstColumnWidth,style.cellAlign,style.firstColumnAlign,style.precision,style.usePercNotn,style.useExpNotn] These tags control various style settings for the cell contents of the cutflow table. The tags "cutline" (default: true) and "usePercNotn" (default:false) are boolean valued tags, all others are integer valued tags. Argument tag overrides object tag.
  //@tag: [style.latex] Boolean argument tag determining if cutflow is produced in LaTeX format. Default: true. Argument tag overrides object tag.
  bool useCutline = tags.getTagBoolDefault ("style.cutline", true);
  tags.getTagInteger ("style.cellWidth", cellWidth);
  tags.getTagInteger ("style.firstColumnWidth", firstColumnWidth);
  tags.getTagInteger ("style.precision", precision);
  tags.getTagDouble ("scale", scale);
  tags.getTagBool ("style.hideSF", hideAllSf);
  tags.getTagBool ("style.showSF", showAllSf);
  bool sfAfter = tags.getTagBoolDefault("style.showSFsAfter", false);
  tags.getTagBool ("style.forceSF", forceAllSf);
  bool usePercNotn = tags.getTagBoolDefault ("style.usePercNotn",false);
  bool useExpNotn = tags.getTagBoolDefault ("style.useExpNotn",false);

  //@tag: [.showErrors] If this tag is set to true, uncertainties are shown in addition to nominal values. Default: true. argument tag overrides object tag.
  bool showErrors = tags.getTagBoolDefault("style.showErrors",true);

  int nCuts = fCuts->GetEntries();
  int nProcesses = fProcesses->GetEntries();

  int row = 0;
  int col = 0;
  /* ----- prepare the table --- */
 
  TQTable* table = new TQTable("Cutflow");
  table->setTagString("colAlign",tags.getTagStringDefault("style.cellAlign","r"));
  table->setTagInteger("cellWidth",tags.getTagIntegerDefault("style.cellWidth",20));
  table->setColAlign(0,tags.getTagStringDefault("style.firstColumnAlign",tags.getTagStringDefault("style.cellAlign","r")));
  table->setTagBool("standalone",false);
  table->setTagInteger("format.nSignificantDigits",precision);
  if(usePercNotn) table->setTagBool("format.useRelativeUncertainties",true);
  if(useExpNotn) table->setTagBool("format.useExponentialNotation",true);

  DEBUGclass("creating preamble and comments");

  TString comment = "created on ";
  comment.Append(TQUtils::getTimeStamp());
  comment.Append(" from TQSampleFolder '");
  comment.Append(fReader->getSampleFolder()->getName());
  comment.Append("' with TQLibrary ");
  comment.Append(TQLibrary::getQLibrary()->getVersion());
  comment.Append(" compiled with GCC ");
  comment.Append(TQLibrary::getQLibrary()->getGCCVersion());
  comment.Append(" against ROOT ");
  comment.Append(TQLibrary::getQLibrary()->getROOTVersion());
  table->setTagString("comment",comment);

  TString preamble;
  if(tags.getTagString("style.preamble",preamble)) table->setTagString("preamble.latex",preamble);
 
  if(useCutline) table->expand(2*nCuts+1,nProcesses+1);
  else table->expand(nProcesses+1,2*nCuts+1);
 
  TString titlecell;
  if(tags.getTagString("style.title",titlecell)) table->setEntry(0,0,titlecell,TQStringUtils::findFormat(titlecell));
  if(tags.getTagString("style.title.latex",titlecell)) table->setProperty(0,0,"content.latex",titlecell);
  if(tags.getTagString("style.title.html",titlecell)) table->setProperty(0,0,"content.html",titlecell);
 
  TQTaggableIterator pitr(fProcesses);
  while(pitr.hasNext()){
    // get the name of the process
    TQNamedTaggable* process = pitr.readNext();
    if(!process) continue;
    TString processName = tags.replaceInText(process->getTagStringDefault(".path",process->GetName()));

    DEBUGclass("looping over process '%s' with path '%s'",process->GetName(),processName.Data());

    if(processName.Length() < 1) continue;
    if (processName.Contains("|")){
      if(useCutline) table->setVline(col+1,TQStringUtils::countText(processName,"|"));
      else table->setHline(row+1,TQStringUtils::countText(processName,"|"));
      continue;
    }
    // get the title of the process
    TString processTitle;
    TList* l = fReader->getListOfSampleFolders(processName);
    if(l){
      TQFolder* f = fReader->getSampleFolder()->findCommonBaseFolder(l);
      if(f){
        processTitle = f->getTagStringDefault("style.default.title","");
        bool showRaw = f->getTagBoolDefault("style.default.showRaw",process->getTagBoolDefault(".isData",this->getTagBoolDefault("style.default.showRaw",false)));
        if(!process->hasTagBool(".showRaw")) process->setTagBool(".showRaw",showRaw);
      }
      delete l;
    }
    if(processTitle.IsNull()){
      processTitle = process->GetTitle();
    }
    if(processTitle.IsNull()){
      processTitle = process->GetName();
    }
    process->getTagString(".title",processTitle);
    processTitle = tags.replaceInText(processTitle);
    // set name and title
    if(useCutline){
      col++;
      table->setEntry(0,col,processTitle,TQStringUtils::findFormat(processTitle));
      table->setProperty(0,col,"tooltip",processName);
    } else {
      row++;
      table->setEntry(row,0,processTitle,TQStringUtils::findFormat(processTitle));
      table->setProperty(row,0,"tooltip",processName);
    }
  }
  TQTaggableIterator citr(fCuts);
  while(citr.hasNext()){
    /* get the name and description of the cut */
    TQNamedTaggable* cut = citr.readNext();
    if(!cut) continue;
    TString cutName = tags.replaceInText(cut->getTagStringDefault(".name",cut->GetName()));
    if(!tags.hasTag(".previousCut")){
      tags.setTagString(".previousCut",cutName);
    }
    DEBUGclass("looping over cut '%s'",cutName.Data());
    int sfPolicy = cut->getTagIntegerDefault(".sfPolicy",1);
    if(hideAllSf) sfPolicy = -1;
    if(cutName.Length() < 1){
      WARNclass("skipping empty cut!");
      continue;
    }

    if (cutName.Contains("|")){
      if(useCutline) table->setHline(row+1,TQStringUtils::countText(cutName,"|"));
      else table->setVline(col+1,TQStringUtils::countText(cutName,"|"));
      tags.removeTag(".previousCut");
      continue;
    }
    
    if (cutName == "!" && useCutline){
      TString cutTitle = tags.replaceInText(cut->getTagStringDefault(".title","Section"));
      table->setEntry (row,0,cutTitle);
      table->setProperty(row,0,"multicol",-1);
      row++;
      continue;
    }
    
    if(useCutline){
      col = 0;
      row++;
    } else {
      row = 0;
      col++;
    }

    TString defaultTitle; 
    int valrow = useCutline ? (row + !sfAfter) : 0;
    int valcol = useCutline ? 0 : (col + !sfAfter);
    int sfrow = useCutline ? (row + sfAfter) : 0;
    int sfcol = useCutline ? 0 : (col + sfAfter);
    bool cutHasSF = false;

    pitr.reset();
    while(pitr.hasNext()){
      /* get the name and description of the process */
      TQNamedTaggable* process = pitr.readNext();

      if(!process) continue;
      TString processName = tags.replaceInText(process->getTagStringDefault(".path",process->GetName()));
      TString processTitle = tags.replaceInText(process->getTagStringDefault(".title",process->GetTitle()));
      bool special = process->hasTagString(".special");
      if(processName.Length() < 1) continue;
      if (processName.Contains("|")) continue;

      if(useCutline){
        col++;
        valcol++;
        sfcol++;
      } else {
        row++;
        valrow++;
        sfrow++;
      }

      double value, error;
      double sfValue,sfError;
      int raw;
      TString info;
      TString sfInfo;
      bool sfApplied, sfEqual;
      bool showNumbers = this->getValues(tags,cut,process,value,error,raw,info,defaultTitle);
      if(!showNumbers) continue;
      bool sfset = special ? false : this->getScaleFactors(tags,cut,process,sfValue,sfError,sfApplied,sfEqual,sfInfo);
      bool sfUnity = TMath::AreEqualRel(sfValue,1., 10E-5);
      bool sfZero = TMath::AreEqualRel(sfValue,0., 10E-5);
      bool showSF = false;
      if(sfPolicy > 0){
        if(forceAllSf || (sfPolicy == 2)) showSF = true;
        else if((sfset && sfApplied) 
                && (sfPolicy == 1 || showAllSf) 
                && !sfUnity && !sfZero) showSF = true;
      }
      DEBUGclass("@%s:%s: showSF=%d with NF=%.3f +/- %.3f (unity=%d,equal=%d,applied=%d,policy=%d)",cutName.Data(),processName.Data(),showSF,sfValue,sfError,sfUnity,sfEqual,sfApplied,sfPolicy);
 
      //@tag: [.showRaw] If this process tag is set to true, raw numbers are shown instead of weighted ones. Default: false.
      //the following documentation might seem a bit odd, but due to the structure of this code, simply setting showRaw=true won't do anything: We set the tag explicitly on each process earlier on taking a defualt value from style.default.showRaw
      //@tag: [style.default.showRaw] If this tag is set to true, raw numbers are shown everywhere instead of weighted ones. Default: false.
      if(process->getTagBoolDefault(".showRaw",tags.getTagBoolDefault("showRaw",false))){
        table->setEntryValue(valrow,valcol,raw);
        DEBUGclass("@%s:%s: %d (raw)",cutName.Data(),processName.Data(),raw);
      } else if(process->getTagBoolDefault(".showErrors",showErrors) && TQUtils::isNum(error)){
        DEBUGclass("@%s:%s: %g +/- %g (error)",cutName.Data(),processName.Data(),value,error);
        table->setEntryValueAndUncertainty(valrow,valcol,value,error);
      } else {
        DEBUGclass("@%s:%s: %g (noError)",cutName.Data(),processName.Data(),value,error);
        table->setEntryValue(valrow,valcol,value);
      }
      int nDigits;
      if(process->getTagInteger("precision",nDigits)){
        table->setProperty(valrow,valcol,"format.nSignificantDigits",nDigits);
      }
      if(!info.IsNull()){
        table->setProperty(valrow,valcol,"tooltip",info);
      }
      
      if(showSF){
        cutHasSF = true;
        if(sfEqual){
          table->setProperty (sfrow,sfcol,"prefixText.latex",nfPrefixText);
          table->setProperty (sfrow,sfcol,"prefixText.html", nfPrefixText);
          table->setProperty (sfrow,sfcol,"prefixText.plain",nfPrefixText);
          if(sfError != 0){
            table->setEntryValueAndUncertainty (sfrow,sfcol,sfValue,sfError);
          } else if(sfValue){
            table->setEntryValue (sfrow,sfcol,sfValue);
          }
        } else {
          table->setEntry (sfrow,sfcol,"NFs Applied");
        }
        table->setProperty(sfrow,sfcol,"textcolor","blue");
        if(!sfInfo.IsNull()){
          table->setProperty(sfrow,sfcol,"tooltip",sfInfo);
        } else if(!sfApplied && sfUnity){
          table->setProperty(sfrow,sfcol,"tooltip","no explicit NF set");
        } else if(sfZero){
          table->setProperty(sfrow,sfcol,"tooltip","NF explicity set to zero");
        } else {
          table->setProperty(sfrow,sfcol,"tooltip","undefined");
        }
      }
    }

    int labelRow = (useCutline ? valrow : 0);
    int labelCol = (useCutline ? 0 : valcol);
    TString cutTitle;
    if(cut->getTagString(".title",cutTitle) || cut->getTagString("title",cutTitle)){
      table->setEntry (labelRow,labelCol,tags.replaceInText(cutTitle));
    } else if(!defaultTitle.IsNull()){
      table->setEntry (labelRow,labelCol,defaultTitle);
    } else {
      table->setEntry (labelRow,labelCol,cutName, "plain");
    }
    table->setProperty(labelRow,labelCol,"tooltip",cutName);
    if(cutHasSF){
      int sfLabelRow = (useCutline ? sfrow : 0);
      int sfLabelCol = (useCutline ? 0 : sfcol);
      table->setEntry (sfLabelRow,sfLabelCol,"Scale factors");
      table->setProperty(sfLabelRow,sfLabelCol,"tooltip",TString::Format("Scale factors for %s",cutName.Data()));

      table->setProperty(sfLabelRow,sfLabelCol,"textcolor","blue");
    }

    if(useCutline){
      row+= 1;
    } else {
      col+= 1;
    }
    tags.setTagString(".previousCut",cutName);
  }
  table->shrink();
  return table;
}

//______________________________________________________________________________________________

TQCutflowPrinter::~TQCutflowPrinter() {
  // Destructor of TQCutflowPrinter class:
}

//______________________________________________________________________________________________

bool TQCutflowPrinter::readProcessesFromFile(TString fileName,TString channel){
  // read processes from a config file (deprecated, please use importProcessesFromFile)
  std::vector<TString>* lines = TQStringUtils::readFileLines(TQStringUtils::trim(fileName));
  if(!lines){
    ERRORclass( "unable to open file '%s'",fileName.Data());
    return false;
  }
  // parse the config file for processes and labels
  for(size_t i=0; i<lines->size(); i++){
    TString process("");
    TString label("");
    TQStringUtils::readUpTo(lines->at(i),process,";");
    TQStringUtils::removeLeading(lines->at(i),"; \t");
    TQStringUtils::readUpTo(lines->at(i),label,";");
    TQStringUtils::removeLeading(lines->at(i),"; \t");
    this->addCutflowProcess(TQStringUtils::trim(process).ReplaceAll("%ch%",channel),label);
  }
  return true;
}

//______________________________________________________________________________________________

bool TQCutflowPrinter::readCutsFromFile(TString fileName, bool addscline){
  // read cuts from a config file (deprecated, please use importCutsFromFile)
  std::vector<TString>* lines = TQStringUtils::readFileLines(TQStringUtils::trim(fileName));
  if(!lines){
    ERRORclass( "unable to open file '%s'",fileName.Data());
    return false;
  }
  for(size_t i=0; i<lines->size(); i++){
    TString cut("");
    TString label("");
    TQStringUtils::readUpTo(lines->at(i),cut,";");
    TQStringUtils::removeLeading(lines->at(i),"; \t");
    TQStringUtils::readUpTo(lines->at(i),label,";");
    TQStringUtils::removeLeading(lines->at(i),"; \t");
    int sfPolicy = 1;
    TString str = TQStringUtils::trim(lines->at(i));
    if(str.Length() > 0)
      sfPolicy = atoi(str.Data());
    if(addscline)
      sfPolicy = 2;
    this->addCutflowCut(cut, label, sfPolicy);
  }
  return true;
}

//______________________________________________________________________________________________

void TQCutflowPrinter::dumpProcesses(std::ostream& out){
  // dump processes to a config file (deprecated, please use exportProcessesToFile)
  for (int iProcess = 0; iProcess < fProcesses->GetEntries(); iProcess++) {
    out << dynamic_cast<TNamed*>(fProcesses->At(iProcess))->GetName() << ";\t";
    TString title = dynamic_cast<TNamed*>(fProcesses->At(iProcess))->GetTitle();
    if(title.Length() > 0)
      out << title << ";";
    out << std::endl;
  }
}

//______________________________________________________________________________________________

void TQCutflowPrinter::dumpCuts(std::ostream& out){
  // dump cuts to a config file (deprecated, please use exportCutsToFile)
  TQIterator itr(fCuts);
  while(itr.hasNext()){
    TQNamedTaggable* cut = dynamic_cast<TQNamedTaggable*>(itr.readNext());
    TString cutName = cut->getTagStringDefault(".name","");
    TString cutTitle = cut->getTagStringDefault(".title","");
    int sfPolicy = cut->getTagIntegerDefault(".sfPolicy",1);
    out << TQStringUtils::fixedWidth(cutName,30);
    if(cutTitle.Length() > 0){
      out << TQStringUtils::fixedWidth(cutTitle,70) << " ; ";
      out << sfPolicy << std::endl;
    }
    out << std::endl;
  }
}
