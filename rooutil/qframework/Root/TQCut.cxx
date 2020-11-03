#include "QFramework/TQCut.h"
#include "TLeaf.h"  
#include "TObjArray.h"
#include "QFramework/TQAnalysisJob.h"
#include "QFramework/TQStringUtils.h"
#include "QFramework/TQValue.h"
#include "QFramework/TQSample.h"
#include "QFramework/TQIterator.h"
#include "QFramework/TQUtils.h"
#include "QFramework/TQUniqueCut.h"

// #define _DEBUG_
#include "QFramework/TQLibrary.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

#include <stdexcept>

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQCut 
//
// The TQCut class is a representation of an event selection cut. Instances of this
// class ("cut") may build up a tree-like hierarchy. Every instance has zero or exactly one 
// base instance and any number of descendant instances. Navigation in upward direction is 
// possible by using getBase() and getRoot(), which return the base instance and the root 
// instance of this instance respectively. 
//
// Internally, a TQObservable is used to evaluate the cut on each individual event. 
// 
// The TQCut can be constructed from a string expression of the following syntax:
// 
// cutName : parentCutName << cutExpression ;
// 
// which passes events where cutExpression evaluates to true (1.) 
// and fails events where cutExpression evaluates to false (0.)
// 
// It is also possible to apply weights by extending the syntax as follows
// 
// cutName : parentCutName << cutExpression : weightExpression ;
//
// where weightExpression may evaluate to any floating number that is multiplied 
// to the event weight.
//
// Both, cutExpression and weightExpression, may use any type of expression
// that TQObservable::getTreeObservable function can make sense of. This
// especially includes arithmetic expressions using branch names present in the
// TTree.
//
// Please consider the following example cut definitions for guidance
//
// CutPreSelect << MET > 20000. ;
// CutLeptonPt : CutPreSelect << (lepPt0 > 22000. || lepPt1 > 22000.) ;
// CutBVeto : CutLeptonPt << nJets_Pt25_MV1_85 == 0 : bTagEventWeight ; 
// CutZVeto : CutBVeto << abs(Mll - 91187.6) <= 15000 ; 
// CutZttVeto : CutZVeto << !(x1 > 0 && x2 >0 && abs(Mtt - 91187.6) <= 25000.) ;
// 
// It is also possible and even advisable to construct a TQCut based on an
// instance of TQFolder. Here, the tree-like folder structure can be mapped 1:1
// to the cut structure created, where each folder in the structure may contain
// tags that control the effect of the cut that is created:
// .name: name of this cut, leave empty or omit to use folder name
// .title: title of this cut, leave empty or omit to use name
// .cutExpression: cut expression, leave empty or omit to pass all events
// .weightExpression: weight expression, leave empty or omit to apply unity weight
// .cutObservableName: name of the cut observable to be used, leave empty or
// omit to use "CUTNAME_cut"
// .weightObservableName: name of the weight observable to be used, leave empty or
// omit to use "CUTNAME_weight"
//
// The expressions may contain placeholders that will be filled from tags in
// the sample folder structure. You may choose to use $(bTagWeightName) as a
// placeholder and set the tag "bTagWeightName=bTagEventWeight" on your base
// sample folder. The replacement will be performed each time a new sample is
// opened, which allows you to use different cut or weight expressions for
// different branches of your sample folder structure. For further details,
// please read the documentation of the function
// TQObservable::compileExpression.
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQCut)


//______________________________________________________________________________________________

bool TQCut::isValidName(const TString& name_) {
  // Check whether the string passed is a valid cut name. Return
  // true if name is valid and false otherwise. Valid cut names
  // may contain the letters a..z, A..Z, 0..9 and an underscore "_".
  return TQStringUtils::isValidIdentifier(name_,"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_",1);
}


//______________________________________________________________________________________________

bool TQCut::parseCutDefinition(const TString& definition_, TString * name_, 
                               TString * baseCutName_, TString * cutExpr_, TString * weightExpr_) {
  // parse the definition of the cut and deploy the parsed fields in the argument pointers
  if(name_ && baseCutName_ && cutExpr_ && weightExpr_)
    return TQCut::parseCutDefinition(definition_, *name_, *baseCutName_, *cutExpr_, *weightExpr_);
  return false;
}


//______________________________________________________________________________________________

bool TQCut::parseCutDefinition(TString definition, TString& name, TString& baseCutName, TString& cutExpr, TString& weightExpr) {
  // parse the definition of the cut and deploy the parsed fields in the argument references

  TString buf;

  TQStringUtils::readUpTo(definition,buf, ":","{}","\"\"");
  if(buf.IsNull()) return false;
  name = TQStringUtils::trim(buf);
  if (!isValidName(name)) { return false; }
 
  TQStringUtils::readToken(definition,buf,":");
  buf.Clear();

  if(definition.Contains("<<")){
    TQStringUtils::readUpTo(definition,buf,"<","{}","\"\"");
    baseCutName = TQStringUtils::trim(buf);
    TQStringUtils::readToken(definition,buf,"<");
    if(!isValidName(baseCutName)) return false;
  } else {
    baseCutName="";
  }
  buf.Clear(); 

  TQStringUtils::readUpTo(definition,buf,":;","{}()[]","\"\"");
  if(definition.IsNull()){
    cutExpr = TQStringUtils::compactify(buf);
  } else {
    cutExpr = TQStringUtils::compactify(buf);
    buf.Clear();
    while(TQStringUtils::readToken(definition,buf,":")>=2) {
      if (buf.Length()>2) {
        TQLibrary::ERRORclass("Unexpected number of consecutive ':'s in basecut definition found (max. 2)!");
        return false;
      }
      TQStringUtils::readUpTo(definition,buf,":","{}()[]","\"\"");
      cutExpr+=buf;
      buf.Clear();
    }
    TQStringUtils::readToken(definition,buf," ");
    buf.Clear();
    TQStringUtils::readUpTo(definition,buf,";","{}()[]","\"\"");
    weightExpr = TQStringUtils::compactify(buf); 
  }
  if(cutExpr.IsNull()) return false;

  return true;

}


//______________________________________________________________________________________________

TQCut * TQCut::createCut(const TString& definition_) {
  // create a new TQCut from the given definition

  TString name, baseCutName, cutExpr, weightExpr;

  TQCut * cut = NULL;

  if (parseCutDefinition(definition_, name, baseCutName, cutExpr, weightExpr)) {

    /* we don't allow a base cut in the static method */
    if (baseCutName.Length() > 0) { return 0; }

    /* create the cut */

    cut = new TQCut(name);

    /* set its parameters */
    cut->setCutExpression(cutExpr);
    cut->setWeightExpression(weightExpr);
  } 

  return cut;
}


//______________________________________________________________________________________________

TQCut * TQCut::createCut(const TString& name, const TString& cutExpr, const TString& weightExpr) {
  // create a new TQCut from the given parameters
 
  TQCut* cut = new TQCut(name);
  cut->setCutExpression(cutExpr);
  cut->setWeightExpression(weightExpr);
 
  return cut;
}



//______________________________________________________________________________________________

TQCut* TQCut::createFromFolderInternal(TQFolder* folder, TQTaggable* tags){
  // import the cut from a folder created with TQCut::dumpToFolder(...)

  /* ===== check that the folder is a valid cut description ===== */
  if (!folder) return 0;

  TString name = folder->GetName();
  TString title = folder->GetTitle();
  //@tag: [.name,.title] These folder tags determine the name and title of the cut when importing the cut definition from a TQFolder.
  folder->getTagString(".name",name);
  folder->getTagString(".title",title);

  TQValue * objCutExpr = dynamic_cast<TQValue*>(folder->getObject(".cutExpression"));
  TQValue * objWeightExpr = dynamic_cast<TQValue*>(folder->getObject(".weightExpression"));
  //@tag: [.cutExpression,.weightExpression] These folder tags determine the cut and weight expressions of the cut when importing the cut definition from a TQFolder. Defaults are the values of TQValue objects with the same name as the corresponding tag, if these TQValue objects are found in the TQFolder. Otherwise default is an empty string "".
  TString cutExpression = folder->getTagStringDefault(".cutExpression", objCutExpr ? objCutExpr ->getString() : "");
  TString weightExpression = folder->getTagStringDefault(".weightExpression",objWeightExpr ? objWeightExpr->getString() : "");
  if (cutExpression.Length()==0) {
    if (weightExpression.Length()==0) {
      throw std::runtime_error(TString::Format("Cut with name '%s' and title '%s' has neither tags '.cutExpression' nor '.weightExpression', please check your cut definitions!",name.Data(),title.Data()).Data());
    }else{
      WARNclass("Found weight expression but no cut expression found when creating cut with name '%s' and title '%s'. If you do not intend to veto events at this cut please consider adding '.cutExpression=\"1.\"' to this cut.",name.Data(),title.Data());
    }
  }
  //@tag: [.skipJobs] Folder tag when creating cut hierarchy from a folder structure. Allows to ignore all analysis jobs attached to the particular cut. Default: false.
  bool skipAnalysisJobsGlobal = folder->getTagBoolDefault(".skipJobs",false); //option to blacklist all analysis jobs for the cut we create (applies accros all samples/sample folders)
  /* create the new cut */
  if(cutExpression.BeginsWith("#UNIQUE")){
    TString buffer;
    TString runno,evtno;
    TQStringUtils::readUpTo(cutExpression,buffer,"(");
    buffer.Clear();
    TQStringUtils::readBlock(cutExpression,buffer ,"()");
    TQStringUtils::readUpTo(buffer,runno ,",");
    TQStringUtils::removeLeading(buffer,",");
    evtno = buffer;
    TQUniqueCut* newcut = new TQUniqueCut(name,runno,evtno);
    newcut->SetTitle(title);
    return newcut;
  } else {
    TQCut * newcut = new TQCut(name,title);
    newcut->setCutExpression (tags ? tags->replaceInTextRecursive(cutExpression) : cutExpression );
    newcut->setWeightExpression(tags ? tags->replaceInTextRecursive(weightExpression) : weightExpression);
    newcut->setSkipAnalysisJobsGlobal(skipAnalysisJobsGlobal);  
    return newcut;
  }
  return NULL;
}

//______________________________________________________________________________________________

void TQCut::importFromFolderInternal(TQFolder * folder, TQTaggable* tags){
  /* import descendent cuts from */
  TQFolderIterator itr(folder->getListOfFolders("?"),true);
  while(itr.hasNext()){
    TQFolder* f = itr.readNext();
    if(!f) continue;
    DEBUGclass("reading folder '%s' for import",f->GetName());
    TQCut * subCut = TQCut::createFromFolderInternal(f,tags);
    if (subCut){
      if(!this->addCut(subCut)){
        WARNclass("cannot add cut '%s' to '%s' - a cut with the given name already exists at '%s'!",subCut->GetName(),this->getPath().Data(),this->getRoot()->getCut(subCut->GetName())->getBase()->getPath().Data());
        delete subCut;
      } else {
        subCut->importFromFolderInternal(f,tags);
      }
    } else {
      WARNclass("unable to import cut from folder '%s'",f->GetName());
    }
  }

  this->sort();
}

//______________________________________________________________________________________________

TQCut * TQCut::importFromFolder(TQFolder * folder, TQTaggable* tags) {
  TQCut* cut = createFromFolderInternal(folder,tags);
  cut->importFromFolderInternal(folder,tags);
  return cut;
}


//_____________________________________________________________________________________________

TQCut::TQCut() : 
  TNamed("<template>", "template"),
  fCutItr(this->fCuts),
  fJobItr(this->fAnalysisJobs)
{
  // default constructor, needed for streaming
}


//______________________________________________________________________________________________

TQCut::TQCut(const TString& name_, const TString& title_, const TString& cutExpression, const TString& weightExpression) : 
  TNamed(name_, title_),
  fCutItr(this->fCuts),
  fJobItr(this->fAnalysisJobs)
{
  // constructor taking additioinal arguments
  this->setCutExpression(cutExpression);
  this->setWeightExpression(weightExpression);
}

//______________________________________________________________________________________________

TList * TQCut::exportDefinitions(bool terminatingColon) {
  // export all cut definitions as a TList

  // create new list
  TList * defs = new TList();
  defs->SetOwner(true);
 
  // add this cut
  TString cutDef = TString(this->GetName()) + ": ";
  if (fBase) {
    cutDef.Append(TString::Format("%s << ", fBase->GetName()));
  }
  cutDef.Append(this->getCutExpression());
  TString weights = this->getWeightExpression();
  if (!weights.IsNull()) {
    cutDef.Append(weights.Prepend(" : "));
  }
  if (terminatingColon) {
    cutDef.Append(";");
  }
  defs->Add(new TObjString(cutDef.Data()));
 
  // iterate over descendants
  this->fCutItr.reset();
  while (this->fCutItr.hasNext()){
    TQCut * c = this->fCutItr.readNext();
    TList * subDefs = c->exportDefinitions(terminatingColon);
    defs->AddAll(subDefs);
    subDefs->SetOwner(false);
    delete subDefs;
  }
 
  // return list of definition strings
  return defs;
}


//______________________________________________________________________________________________

void TQCut::setBase(TQCut * base_) {
  // set the base cut
  fBase = base_;
}


//______________________________________________________________________________________________

TQCut * TQCut::getBase() {
  // Return the base cut of this cut

  return fBase;
}


//______________________________________________________________________________________________

TString TQCut::getPath() {
  // Return the path to this cut
  if(this->fBase) return TQFolder::concatPaths(this->fBase->getPath(),this->GetName());
  else return this->GetName();
}

//______________________________________________________________________________________________

const TQCut * TQCut::getBaseConst() const {
  // Return the base cut of this cut
  return fBase;
}


//______________________________________________________________________________________________

TQCut * TQCut::getRoot() {
  // Return the root cut of this cut hierarchy

  if (fBase)
    /* this cut has a base cut: return the root of it */
    return fBase->getRoot();
  else
    /* this cut is the root */
    return this;
}


//______________________________________________________________________________________________

void TQCut::printEvaluation() const {
  // print the evaluation steps for this cut expression 
  // on the currently active event
  // WARNING: THIS FUNCTION ONLY MAKES SENSE FOR PROPERLY INITIALIZED CUTS
  // IT WILL PRODUCE SEVERAL OUTPUT LINES FOR EACH EVENT - EXTREMELY VERBOSE!
  // USE FOR DEBUGGING ONLY!
  Long64_t iEvent = this->fTree->GetReadEntry();
  this->printEvaluation(iEvent);
}

//______________________________________________________________________________________________

void TQCut::printWeightComponents() const{
  // print the evaluation steps for this weight expression 
  // on the currently active event
  // WARNING: THIS FUNCTION ONLY MAKES SENSE FOR PROPERLY INITIALIZED CUTS
  // IT WILL PRODUCE SEVERAL OUTPUT LINES FOR EACH EVENT - EXTREMELY VERBOSE!
  // USE FOR DEBUGGING ONLY!
  Long64_t iEvent = this->fTree->GetReadEntry();
  this->printWeightComponents(iEvent);
}

//______________________________________________________________________________________________

void TQCut::printWeightComponents(Long64_t/*iEvent*/) const{
  // print the evaluation steps for this weight expression 
  // on event number iEvent 
  // WARNING: THIS FUNCTION ONLY MAKES SENSE FOR PROPERLY INITIALIZED CUTS
  // IT WILL PRODUCE SEVERAL OUTPUT LINES FOR EACH EVENT - EXTREMELY VERBOSE!
  // USE FOR DEBUGGING ONLY!
  TQIterator itr(this->fWeightObservable->getBranchNames(),true);
  int index = 0;
  while(itr.hasNext()){
    if(index != 0){
      std::cout << ", ";
    }
    TObject* next = itr.readNext();
    if(!next) continue;
    TLeaf *leaf = fTree->GetLeaf(next->GetName());
    if(!leaf) continue;
    if(fTree->GetBranchStatus(next->GetName()) == 1){
      std::cout << next->GetName() << "=" << leaf->GetValue();
    } else {
      std::cout << TQStringUtils::makeBoldRed(next->GetName()) << " " << TQStringUtils::makeBoldRed("undefined");
    }
    index++;
  }
  std::cout << std::endl;
}

//______________________________________________________________________________________________

void TQCut::printEvaluation(Long64_t iEvent) const {
  // print the evaluation steps for this cut expression 
  // on event number iEvent 
  // WARNING: THIS FUNCTION ONLY MAKES SENSE FOR PROPERLY INITIALIZED CUTS
  // IT WILL PRODUCE SEVERAL OUTPUT LINES FOR EACH EVENT - EXTREMELY VERBOSE!
  // USE FOR DEBUGGING ONLY!
  std::cout << TQStringUtils::makeBoldWhite(TString::Format("considering entry %lld...",iEvent)) << std::endl;
  this->printEvaluationStep(0);
  int index = 0;
  const TQCut* c = this;
  do {
    TQIterator itr(c->fCutObservable->getBranchNames(),true);
    while(itr.hasNext()){
      if(index != 0){
        std::cout << ", ";
      }
      TObject* next = itr.readNext();
      if(!next) continue;
      TLeaf *leaf = fTree->GetLeaf(next->GetName());
      if(!leaf) continue;
      if(fTree->GetBranchStatus(next->GetName()) == 1){
        std::cout << next->GetName() << "=" << leaf->GetValue();
      } else {
        std::cout << TQStringUtils::makeBoldRed(next->GetName()) << " " << TQStringUtils::makeBoldRed("undefined");
      }
      index++;
    }
    c = c->passed() ? c->getBaseConst() : NULL;
  } while (c);
  std::cout << std::endl;
}

//______________________________________________________________________________________________

bool TQCut::printEvaluationStep(size_t indent) const {
  // print a single evaluation step for this cut
  bool passed = this->passed();
  std::cout << TQStringUtils::makeBoldWhite("[");
  if(passed) std::cout << TQStringUtils::makeBoldGreen(" OK ");
  else std::cout << TQStringUtils::makeBoldRed("FAIL");
  std::cout << TQStringUtils::makeBoldWhite("]");
  std::cout << TQStringUtils::repeat("\t",indent);
  if(indent > 0) std::cout << "&& ";
  std::cout << this->getActiveCutExpression();
  std::cout << std::endl;
  if(passed && this->fBase){
    passed = this->fBase->printEvaluationStep(indent+1);
  }
  return passed;
}

//______________________________________________________________________________________________

void TQCut::setCutExpression(const TString& cutExpression){
  // set the cut expression to some string
  this->fCutExpression = TQStringUtils::minimize(cutExpression);
}

//______________________________________________________________________________________________

void TQCut::setWeightExpression(const TString& weightExpression) {
  // set the weight expression to some string
  this->fWeightExpression = TQStringUtils::minimize(weightExpression);
}

//______________________________________________________________________________________________

TQObservable* TQCut::getCutObservable() {
  // get the cut observable
  return this->fCutObservable;
}

//______________________________________________________________________________________________

TQObservable* TQCut::getWeightObservable(){
  // get the weight observable
  return this->fWeightObservable;
}


//______________________________________________________________________________________________

const TString& TQCut::getCutExpression() const {
  // retrieve the cut expression
  return this->fCutExpression;
}


//______________________________________________________________________________________________

TString TQCut::getActiveCutExpression() const {
  // retrieve the currently active cut expression
  return (this->fCutObservable ? this->fCutObservable->getActiveExpression() : TQStringUtils::emptyString);
}

//______________________________________________________________________________________________

TString TQCut::getCompiledCutExpression(TQTaggable* tags) {
  // retrieve the cut expression
  return TQObservable::compileExpression(this->fCutExpression,tags);
}

//______________________________________________________________________________________________

TString TQCut::getGlobalCutExpression(TQTaggable* tags) {
  // retrieve the cut expression
  if(this->fBase){
    TString expr = this->fBase->getGlobalCutExpression(tags);
    expr.Append(" && ( ");
    expr.Append(this->getCompiledCutExpression(tags));
    expr.Append(")");
    return expr;
  }
  return this->getCompiledCutExpression(tags);
}


//______________________________________________________________________________________________

const TString& TQCut::getWeightExpression() const {
  // retrieve the weight expression
  return this->fWeightExpression;
}

//______________________________________________________________________________________________

TString TQCut::getActiveWeightExpression() const {
  // retrieve the currently active cut expression
  return (this->fWeightObservable ? this->fWeightObservable->getActiveExpression() : TQStringUtils::emptyString);
}

//______________________________________________________________________________________________

TString TQCut::getCompiledWeightExpression(TQTaggable* tags) {
  // retrieve the cut expression
  return TQObservable::compileExpression(this->fWeightExpression,tags);
}

//______________________________________________________________________________________________

TString TQCut::getGlobalWeightExpression(TQTaggable* tags) {
  // retrieve the weight expression
  if(this->fBase){
    TString expr = this->fBase->getGlobalWeightExpression(tags);
    expr.Append(" * ");
    expr.Append(this->getCompiledWeightExpression(tags));
    return expr;
  }
  return this->getCompiledWeightExpression(tags);
}


//______________________________________________________________________________________________

TQCut * TQCut::addAndReturnCut(const TString& definition_) {
  // add a cut defined by a string and return it

  TString name, baseCutName, cutExpr, weightExpr;

  if (!parseCutDefinition(definition_, name, baseCutName, cutExpr, weightExpr))
    return NULL;

  /* create the cut */
  TQCut * cut = new TQCut(name);
 
  /* set its parameters */
  cut->setCutExpression(cutExpr);
  cut->setWeightExpression(weightExpr);
 
  /* add the cut */
  if (addCut(cut, baseCutName)) {
    return cut;
  } else {
    delete cut;
  }

  return NULL;
}


//______________________________________________________________________________________________

bool TQCut::addCut(const TString& definition_) {
  // add a cut defined by a string
  // internally calles TQCut::addAndReturnCut

  if (addAndReturnCut(definition_)) {
    return true;
  } else {
    return false;
  }
}


//______________________________________________________________________________________________

bool TQCut::addCut(TQCut * cut_, const TString& baseCutName) {
  // add a given instance of TQCut to a basecut identified by name

  /* try to find the base cut */
  TQCut * cut = getCut(baseCutName);
  
  /* stop if we couldn't find the base cut */
  if (!cut) { return false; }
  
  /* add the cut to the base cut */
  return cut->addCut(cut_);
}

//______________________________________________________________________________________________

bool TQCut::addCut(TQCut * cut_) {
  // add a given instance of TQCut to a basecut identified by name

  /* stop if no cut is given */
  if (!cut_) { return false; }

  /* stop if there is already a cut with the same name */
  if (this->getRoot()->getCut(cut_->GetName())) { return false; }
 
  /* add the new cut to this cut */
  cut_->setBase(this);
  fCuts->Add(cut_);
  return true;
}


//______________________________________________________________________________________________

bool TQCut::isMergeable() const {
  // returns true if this cut is mergeable, false otherwise
  return true;
}


//______________________________________________________________________________________________

void TQCut::consolidate() {
  // try to consolidate the cut hierarchy
  // this function attempts to merge as many cuts as possibel 
  // without obstructing any existing analysis jobs
  // STILL IN EXPERIMENTAL STATE

  /* try to consolidate every child */
  for (int iChild = 0; iChild < fCuts->GetEntries(); iChild++) {
    ((TQCut*)fCuts->At(iChild))->consolidate();
  }

  /* try to remove every child */
  int i = 0;
  while (i < fCuts->GetEntries()) {

    if (removeCut(fCuts->At(i)->GetName()))
      i = 0;
    else
      i++;

  }

}

//______________________________________________________________________________________________

int TQCut::getDepth() const{
  // return the maximum depth of this cut (branch)
  int retval = 0;
  this->fCutItr.reset();
  while(this->fCutItr.hasNext()){
    TQCut* cut = this->fCutItr.readNext();
    if(!cut) continue;
    retval = std::max(retval,cut->getDepth()+1);
  }
  return retval;
}

//______________________________________________________________________________________________

int TQCut::getWidth() const{
  // return the width of this cut (number of immediate child cuts)
  return this->fCuts->GetEntries();
}

//______________________________________________________________________________________________

int TQCut::Compare(const TObject* obj) const {
  const TQCut* c = dynamic_cast<const TQCut*>(obj);
  if(!c) return 0;
  int otherDepth = c->getDepth();
  int myDepth = this->getDepth();
  if(otherDepth > myDepth) return 1;
  if(otherDepth < myDepth) return -1;
  int otherWidth = c->getWidth();
  int myWidth = this->getWidth();
  if(otherWidth > myWidth) return -1;
  if(otherWidth < myWidth) return 1;
  return 0;
}

//______________________________________________________________________________________________

void TQCut::sort() {
  // sort the cut hierarchy
  // this will have no effect on the results of the analysis
  // however, it will reorganize the layout of the cut diagrams
  // in such a way that a vertical structure is most eminent
 
  if(this->fCuts->GetEntries() < 2) return;

  this->fCutItr.reset();
  while(this->fCutItr.hasNext()){
    TQCut* cut = this->fCutItr.readNext();
    if(!cut) continue;
    cut->sort();
  }

  this->fCuts->Sort();
}


//______________________________________________________________________________________________


bool TQCut::isTrivialTrue(const TString& expr){
  // return true if this cut is trivally passed by every event
  TString e(TQStringUtils::trim(expr));
  if(e == "1" || e == "1.") return true;
  return false;
}

//______________________________________________________________________________________________


bool TQCut::isTrivialFalse(const TString& expr){
  // return true if this cut is trivally failed by every event
  TString e(TQStringUtils::trim(expr));
  if(e == "0" || e == "0.") return true;
  return false;
}

//______________________________________________________________________________________________

bool TQCut::includeBase() {
  // merge this cut with its base cut

  /* stop if this is the root */
  if (!fBase) { return false; }

  /* stop if base cut or this cut isn't mergeable */
  if (!isMergeable() || !fBase->isMergeable()) { return false; }

  /* merge cut expressions */
  if(TQCut::isTrivialTrue(this->getCutExpression())){
    this->setCutExpression(this->fBase->getCutExpression());
  } else if(TQCut::isTrivialTrue(this->fBase->getCutExpression()) || TQCut::isTrivialFalse(this->getCutExpression())){
    // we don't need to do anything
  } else if(TQCut::isTrivialFalse(this->fBase->getCutExpression())){
    this->setCutExpression("0.");
  } else {
    this->setCutExpression(TString::Format("( %s ) && ( %s )",this->fBase->getCutExpression().Data(),this->getCutExpression().Data()));
  } 

  /* merge weight expressions */
  TString newWeightExpression = fBase->getWeightExpression();
  if (newWeightExpression.Length() > 0 && this->fWeightObservable){
    newWeightExpression.Append(" * "); }
  newWeightExpression.Append(this->getWeightExpression());
 
  this->setWeightExpression(newWeightExpression);
 
  return true;
 
}


//______________________________________________________________________________________________

bool TQCut::removeCut(const TString& name) {
  // remove a cut from the cut hierarchy (by name)

  /* the index of the cut to be removed */
  int iRemove = -1;

  /* try to find the cut to be removed in the list of children */
  for (int iChild = 0; iChild < fCuts->GetEntries(); iChild++) {
    if (name.CompareTo(fCuts->At(iChild)->GetName(), TString::kIgnoreCase) == 0) {
      iRemove = iChild;
    }
  }

  if (iRemove >= 0) {

    /* get the cut to be removed */
    TQCut * cutToRemove = (TQCut*)fCuts->At(iRemove);

    /* we cannot remove the cut if it has analysis jobs or it cannot be merged */
    if (cutToRemove->getNAnalysisJobs() > 0 || !cutToRemove->isMergeable()) 
      return false;

    /* prepare a new list of sub cuts */
    TObjArray * newSubCuts = new TObjArray();

    /* try to merge the cut to be removed and its subcuts */
    bool success = true;
    TObjArray * subCuts = cutToRemove->getCuts();
    for (int iSubChild = 0; iSubChild < subCuts->GetEntries(); iSubChild++) { 
      /* get a clone of the subcut */
      TQCut * subChild = (TQCut*)(subCuts->At(iSubChild)->Clone());
      /* try to merge the cuts */
      success = subChild->includeBase() && success;
      /* add it to the new list */
      newSubCuts->Add(subChild);
      subChild->setBase(this); 
    }

    if (success) {
 
      /* delete the cut to be removed */
      delete cutToRemove;

      /* replace list of children */
      TObjArray * newCuts = new TObjArray(); 
      for (int iChild = 0; iChild < fCuts->GetEntries(); iChild++) {
        if (iChild < iRemove || iChild > iRemove)
          newCuts->Add(fCuts->At(iChild));
        else 
          for (int iSubChild = 0; iSubChild < newSubCuts->GetEntries(); iSubChild++)
            newCuts->Add(newSubCuts->At(iSubChild));
      }
 
      delete fCuts;
      fCuts = newCuts;

    } else {
 
      /* delete list of clones */
      for (int j = 0; j < newSubCuts->GetEntries(); j++)
        delete newSubCuts->At(j);

    }

    return success; 

  } else {

    /* loop over children and try to remove cut recursively */
    for (int iChild = 0; iChild < fCuts->GetEntries(); iChild++) {
      if (((TQCut*)fCuts->At(iChild))->removeCut(name)) {
        return true;
      }
    }

    return false;

  }

}

//__________________________________________________________________________________|___________

void TQCut::printCut(const TString& options) {
  // wrapper for the "print" function
  this->printInternal(options,0);
}

//______________________________________________________________________________________________

void TQCut::printActiveCutExpression(size_t indent) const {
  // print the currently active cut expression for this cut and all parent cuts
  std::cout << TQStringUtils::repeat("\t",indent);
  if(indent > 0) std::cout << "&& ";
  std::cout << this->getActiveCutExpression();
  std::cout << std::endl;
  if(this->fBase){
    this->fBase->printActiveCutExpression(indent+1);
  }
}


//__________________________________________________________________________________|___________

void TQCut::print(const TString& options) {
  // Print a summary of this cut instance. The following options my be specified:
  // - "r" (default) also print a summary of descendant cuts
  this->printInternal(options,0);
}

//__________________________________________________________________________________|___________

void TQCut::printCuts(const TString& options) {
  // Print a summary of this cut instance. The following options my be specified:
  // - "r" (default) also print a summary of descendant cuts
  this->printInternal(options,0);
}
  
//__________________________________________________________________________________|___________

void TQCut::printInternal(const TString& options, int indent) {
  // Print a summary of this cut instance. The following options my be specified:
  // - "r" (default) also print a summary of descendant cuts
  const int cColWidth_Total = TQLibrary::getConsoleWidth() - 4;
  const int cColWidth_Name = 0.5*cColWidth_Total;
  const int cColWidth_nJobs = 10;
  const int cColWidth_CutExpr = 0.3*cColWidth_Total;
  const int cColWidth_WeightExpr = cColWidth_Total - cColWidth_Name - cColWidth_nJobs - cColWidth_CutExpr;
 
  bool printRecursive = options.Contains("r", TString::kIgnoreCase);
 
  /* print headline if this is first indent */
  if (indent == 0) { 
    TString headline = TString::Format("%-*s %*s %-*s %-*s",
                                       cColWidth_Name, "Name", cColWidth_nJobs, "# Jobs", 
                                       cColWidth_CutExpr, "Cut Expression", cColWidth_WeightExpr, "Weight Expression");
    std::cout << TQStringUtils::makeBoldWhite(headline) << std::endl;
    std::cout << TQStringUtils::makeBoldWhite(TQStringUtils::repeat("=", TQStringUtils::getWidth(headline))) << std::endl;
  }
 
  TString line;
  line.Append(TQStringUtils::fixedWidth(TString::Format("%*s%s", indent, "", GetName()), cColWidth_Name, "l."));
  line.Append(" ");
  line.Append(TQStringUtils::fixedWidth(TString::Format("%d", getNAnalysisJobs()), cColWidth_nJobs,"r"));
  line.Append(" ");
  line.Append(TQStringUtils::fixedWidth(this->getCutExpression(), cColWidth_CutExpr, "l."));
  line.Append(" ");
  line.Append(TQStringUtils::fixedWidth(this->getWeightExpression(), cColWidth_WeightExpr, "l."));
  std::cout << line.Data() << std::endl;

  if (printRecursive) {
    TQIterator itr(fCuts);
    while(itr.hasNext()){
      TQCut* c = dynamic_cast<TQCut*>(itr.readNext());
      if(!c) continue;
      c->printInternal(options, indent + 1);
    }
  }
 
}

//__________________________________________________________________________________|___________

void TQCut::writeDiagramHeader(std::ostream & os, TQTaggable& tags){
  // write the header of the cut diagram file to a stream

  //@tag:[standalone] If this argument tag is set to true, standalone LaTeX document with the cut diagrams is created. Default: false
  bool standalone = tags.getTagBoolDefault("standalone",false);
  if(standalone) os << "\\documentclass{standalone}" << std::endl << "\\usepackage{tikz}" << std::endl << "\\usepackage{underscore}" << std::endl << "\\usetikzlibrary{positioning,arrows}" << std::endl << "\\begin{document}" << std::endl;
  //@tag:[format] This argument tag sets the format in which the cut diagram is produced. Default: "tikz" (other formats are currently not cupported)
  TString format = tags.getTagStringDefault("format","tikz");
  if(format == "tikz"){
    os << "\\begingroup" << std::endl;
    //@tag:[nodes.color,nodes.opacity,nodes.width,nodes.padding,nodes.height,nodes.distance] These argument tags define how nodes are drawn in the cut diagram produced in tikz/LaTeX format. Defaults (in order): "blue", 20, "10em", "2.5pt", "2em", "0.1cm"
    os << "\\tikzstyle{block} = [rectangle, draw, fill=" << tags.getTagStringDefault("nodes.color","blue") << "!" << tags.getTagIntegerDefault("nodes.opacity",20) << ", text width=" << tags.getTagStringDefault("nodes.width","10em") <<", inner sep=" << tags.getTagStringDefault("nodes.padding","2.5pt") << ", text centered, rounded corners, minimum height=" << tags.getTagStringDefault("nodes.height","2em") << "]" << std::endl;
    //@tag:[jobs.color,jobs.opacity,jobs.padding] These argument tags define how jobs (cuts) are drawn in the cut diagram produced in tikz/LaTeX format. Defaults (in order): "red", 20, "2.5pt"
    os << "\\tikzstyle{job} = [rectangle, draw, fill=" << tags.getTagStringDefault("jobs.color","red") << "!" << tags.getTagIntegerDefault("jobs.opacity",20) << ", inner sep=" << tags.getTagStringDefault("jobs.padding","2.5pt") << ", rounded corners, font={\\tiny}]" << std::endl;
    os << "\\tikzstyle{line} = [draw, -latex']" << std::endl;
    os << std::endl;
    os << "\\begin{tikzpicture}[" << "]" << std::endl;
  }
}

//__________________________________________________________________________________|___________

void TQCut::writeDiagramFooter(std::ostream & os, TQTaggable& tags){
  // write the footer of the cut diagram file to a stream
  
  //tag documentation see writeDiagramHeader
  bool standalone = tags.getTagBoolDefault("standalone",false);
  TString format = tags.getTagStringDefault("format","tikz");
  if(format == "tikz"){
    os << "\\end{tikzpicture}" << std::endl;
    os << "\\endgroup" << std::endl;
  }
  if(standalone) os << "\\end{document}" << std::endl;
 
}

//__________________________________________________________________________________|___________

TString TQCut::getNodeName(){
  // retrieve the name of this node for display 

  TString name(this->GetName());
  name.ToLower();
  return TQStringUtils::makeValidIdentifier(name,TQStringUtils::lowerLetters+TQStringUtils::numerals);
}

//__________________________________________________________________________________|___________

int TQCut::writeDiagramText(std::ostream& os, TQTaggable& tags, TString pos){
  // write the code for the cut diagram corresponding to this node

  TString format = tags.getTagStringDefault("format","tikz");
  if(format == "tikz"){
 
    os << "\\node [block";
    if(!pos.IsNull()) os << ", " << pos;
    os << "] (" << this->getNodeName() << ") {" << (TQStringUtils::isEmpty(this->GetTitle()) ? this->GetName() : this->GetTitle())<< "};" << std::endl;
    
  } else {
    WARNclass("unknown format '%s'!",format.Data());
    return 0;
  }
  TQCutIterator itr(fCuts);
  int width = 0;
  pos = "below=of " + this->getNodeName();
  TString anchor = "south";
  while(itr.hasNext()){
    TQCut* c = itr.readNext();
    if(!c) continue;
    int newwidth = c->writeDiagramText(os,tags,pos);
    os << "\\path [line] (node cs:name=" << this->getNodeName() << ", anchor=" << anchor << ") -| (node cs:name=" << c->getNodeName() << ", anchor=north);" << std::endl;
    pos = TString::Format("right=%d*%s+%d*(%s+2*%s) of %s",newwidth,tags.getTagStringDefault("nodes.distance","0.1cm").Data(),newwidth-1,tags.getTagStringDefault("nodes.width","10em").Data(),tags.getTagStringDefault("nodes.padding","2.5pt").Data(),c->getNodeName().Data());
    anchor = "east";
    width += newwidth;
  }
  //@tag: [showJobs] If this argument tag is set to true, the analysis jobs book at each cut are shown in the cut diagram.
  if(tags.getTagBoolDefault("showJobs",false)){
    TQAnalysisJobIterator itr(fAnalysisJobs);
    TString tmppos("below = 1mm of ");
    tmppos.Append(this->getNodeName());
    tmppos.Append(".south east");
    while(itr.hasNext()){
      TQAnalysisJob* aj = itr.readNext();
      if(!aj) continue;
      TString name = this->getNodeName() + ".job" + TString::Format("%d",itr.getLastIndex());
      os << "\\node[job, " << tmppos << "] (" << name << ") {" << aj->GetName() << "};" << std::endl;
      tmppos = "below = 1mm of " + name + ".south";
    }
  }
  return std::max(width,1);
}

//__________________________________________________________________________________|___________

bool TQCut::writeDiagramToFile(const TString& filename, const TString& options){
  // create a cut hierarchy diagram 
  TQTaggable tags(options);
  return this->writeDiagramToFile(filename,tags);
}

//__________________________________________________________________________________|___________

bool TQCut::writeDiagramToFile(const TString& filename, TQTaggable& tags){
  // create a cut hierarchy diagram 
  if(filename.IsNull()) return false;
  TQUtils::ensureDirectoryForFile(filename);
  std::ofstream of(filename.Data());
  if(!of.is_open()){
    ERRORclass("unable to open file '%s'",filename.Data());
    return false;
  }
  TQCut::writeDiagramHeader(of,tags);
  this->writeDiagramText(of,tags);
  TQCut::writeDiagramFooter(of,tags);
  return true;
}


//__________________________________________________________________________________|___________

bool TQCut::printDiagram(const TString& options){
  // create a cut hierarchy diagram 
  TQTaggable tags(options);
  return this->printDiagram(tags);
}

//__________________________________________________________________________________|___________

bool TQCut::printDiagram(TQTaggable& tags){
  // create a cut hierarchy diagram 
  TQCut::writeDiagramHeader(std::cout,tags);
  this->writeDiagramText(std::cout,tags);
  TQCut::writeDiagramFooter(std::cout,tags);
  return true;
}

//__________________________________________________________________________________|___________

TString TQCut::writeDiagramToString(TQTaggable& tags){
  // create a cut hierarchy diagram (as TString)
  std::stringstream of;
  TQCut::writeDiagramHeader(of,tags);
  this->writeDiagramText(of,tags);
  TQCut::writeDiagramFooter(of,tags);
  TString retval(of.str().c_str());
  return retval;
}

//__________________________________________________________________________________|___________

int TQCut::dumpToFolder(TQFolder * folder) {
  // dump the entire cut hierarchy to a TQFolder

  /* stop if folder to write to is invalid */
  if (!folder)
    return 0;

  /* the number of cuts dumped to the folder */
  int nCuts = 1;

  /* create a sub-folder corresponding to this cut */
  TQFolder * cutFolder = folder->getFolder(TString::Format("%s+", GetName()));

  /* stop if we failed to create the sub folder */
  if (!cutFolder)
    return 0;

  cutFolder->setTagString(".name",this->GetName());
  cutFolder->setTagString(".title",this->GetTitle());

  /* set the properties of this cut */
  cutFolder->setTagString(".cutExpression",this->getCutExpression());
  cutFolder->setTagString(".weightExpression",this->getWeightExpression());
  //@tag:[.nAnalysisJobs] This folder tag is set when exporting a TQCut to a TQFolder containing the number of analysis jobs booked at the cut.
  cutFolder->setTagInteger(".nAnalysisJobs",this->getNAnalysisJobs());

  /* add all the descendant cuts */
  TQIterator itr(fCuts);
  while (itr.hasNext()){
    TQCut* c = dynamic_cast<TQCut*>(itr.readNext());
    if(!c) continue;
    nCuts += c->dumpToFolder(cutFolder);
  }

  /* return the number of cuts dumped to the folder */
  return nCuts;
}


//__________________________________________________________________________________|___________
TQCut * TQCut::getCut(const TString& name) {
  // Return the cut matching "name". The hierarchy of cuts will be searched 
  // starting from this cut downwards. Please note: the result might be a null 
  // pointer, if no matching element can be found

  /* Is this instance the cut that is requested? */
  if (name.CompareTo(this->GetName()) == 0) 
    /* return this instance if it is */
    return this;

  /* if not, look for the cut that is requested
   * recursively in the list of descendant cuts */
  TQCut * found = 0;
  this->fCutItr.reset();
  while(this->fCutItr.hasNext() && !found){
    TQCut* c = this->fCutItr.readNext();
    if(c) found = c->getCut(name);
  }
  
  /* return the cut that might have been found */ 
  return found;
}

//__________________________________________________________________________________|___________
void TQCut::getMatchingCuts(TObjArray& matchingCuts, const TString& name) {
    // Add all cuts matching 'name' to the given TObjArray 'matchingCuts'. The
    // syntax of matching cuts is borrowed from TQFolder: * matches any
    // cuts/part of a cut name, ? matches exactly one cut level in the cut
    // hierarchy.

    TObjArray name_segments;
    TQIterator itr(TQStringUtils::tokenize(name, "/"),true);
    while (itr.hasNext()) {
        name_segments.Add(itr.readNext());
    }
    
    getMatchingCuts(matchingCuts, name_segments);
}

//__________________________________________________________________________________|___________
void TQCut::propagateMatchingCuts(TObjArray& matchingCuts, const TObjArray& name_segments, int offset) {
  // Propagate getMatchingCuts to all attached cuts.
  this->fCutItr.reset();
  while(this->fCutItr.hasNext()) {
    TQCut* c = this->fCutItr.readNext();
    if (c) {
        c->getMatchingCuts(matchingCuts, name_segments, offset);
    }
  }
}

//__________________________________________________________________________________|___________
void TQCut::getMatchingCuts(TObjArray& matchingCuts, const TObjArray& name_segments, int offset) {
    // Same as getMatchingCuts(TObjArray& matchingCuts, const TString& name),
    // but name is a tokenized by the directory delimiter.
    // If offset != 0, it means that the first segments have already been
    // parsed.
    
    if (offset >= name_segments.GetEntries()) {
        // offset of beyond the last name segment
        return;
    }
    
    TObjString* first_obj = dynamic_cast<TObjString*>(name_segments.At(0 + offset));
    if (!first_obj) {
        // list of path segments is empty
        return;
    }
    TString first = first_obj->GetString();
    DEBUGclass("%s: current token: %s; offset: %d", this->GetName(), first.Data(), offset);

    if (first.CompareTo("*") == 0) {
        // current path segment is an asterisk
        if (offset + 1 < name_segments.GetEntries()) {
            // part after asterisk exists
            DEBUGclass("%s: parsed asterisk; propagate next segment", this->GetName());

            // * matches none
            getMatchingCuts(matchingCuts, name_segments, offset + 1);

            if (offset != 0) {
                DEBUGclass("%s: propagate asterisk", this->GetName());
                // * matches many
                propagateMatchingCuts(matchingCuts, name_segments, offset);
            }
        } else {
            // asterisk is final part
            DEBUGclass("%s: parsed asterisk; asterisk is final part", this->GetName());
            if (matchingCuts.IndexOf(this) == -1) {
                // add only if not in list
                matchingCuts.Add(this);
            }
            DEBUGclass("%s: added <----", this->GetName());

            if (offset != 0) {
                DEBUGclass("%s: propagate asterisk", this->GetName());
                propagateMatchingCuts(matchingCuts, name_segments, offset);
            }
        }

    } else if (first.CompareTo("?") == 0) {
        // current path segment is an question mark
        if (offset + 1 < name_segments.GetEntries()) {
            // part after question mark exists
            DEBUGclass("%s: parsed question mark; propagate next segment", this->GetName());

            if (isResidualMatchingSegmentOptional(name_segments, offset)) {
                // path ends in a(n) (series of) asterisk(s)
                DEBUGclass("%s: residual is optional; added <----", this->GetName());
                if (matchingCuts.IndexOf(this) == -1) {
                    // add only if not in list
                    matchingCuts.Add(this);
                }
            }
            // consuming the wildcard, and propagate
            propagateMatchingCuts(matchingCuts, name_segments, offset + 1);
        } else {
            // question mark is final part
            DEBUGclass("%s: parsed question mark; wildcard is final part", this->GetName());
            if (matchingCuts.IndexOf(this) == -1) {
                // add only if not in list
                matchingCuts.Add(this);
            }
            DEBUGclass("%s: added <----", this->GetName());
        }

    } else {
        // regular path segment
        DEBUGclass("%s: regular path segment (%s)", this->GetName(), first.Data());
        if (TQStringUtils::matches(this->GetName(), first)) {
            // the cut name matches the current name segment
            if (offset + 1 < name_segments.GetEntries()) {
                if (isResidualMatchingSegmentOptional(name_segments, offset)) {
                    // path ends in a(n) (series of) asterisk(s)
                    DEBUGclass("%s: residual is optional; added <----", this->GetName());
                    if (matchingCuts.IndexOf(this) == -1) {
                        // add only if not in list
                        matchingCuts.Add(this);
                    }
                }

                // there are other segments
                DEBUGclass("%s: matched; propagate next segment", this->GetName());
                propagateMatchingCuts(matchingCuts, name_segments, offset + 1);

            } else {
                // this was final segment
                DEBUGclass("%s: this was final; added <----", this->GetName());
                if (matchingCuts.IndexOf(this) == -1) {
                    // add only if not in list
                    matchingCuts.Add(this);
                }
            }
        }
    }

    if (offset == 0) {
        // no name segments have been matched so far
        // also search recursively for other matching cuts, 
        DEBUGclass("%s: offset==0, propagate recursively", this->GetName());
        propagateMatchingCuts(matchingCuts, name_segments, 0);
    }
}

//______________________________________________________________________________________________
bool TQCut::isResidualMatchingSegmentOptional(const TObjArray& name_segments, int offset) {
  // check if residual parts are optional
  for (int i = offset + 1; i < name_segments.GetEntries(); i++) {
      TObjString* obj = dynamic_cast<TObjString*>(name_segments.At(i));
      if (!obj) { continue; }
      TString seg = obj->GetString();
      if (seg.CompareTo("*") != 0) {
          return false;
      }
  }
  return true;
}

//______________________________________________________________________________________________

TObjArray * TQCut::getCuts() {
  // Return the list of descendant cuts of this instance 
  return fCuts;
}


//______________________________________________________________________________________________

TObjArray * TQCut::getJobs() {
  // Return the list of analysis jobs at this cut
  return fAnalysisJobs;
}


//______________________________________________________________________________________________

void TQCut::setCuts(TObjArray* cuts){
  // set the internal list of descendant cuts to something different
  fCuts = cuts;
}

//______________________________________________________________________________________________

void TQCut::printAnalysisJobs(const TString& options) {
  // print the list of analysis jobs directly appended to this cut
  TQAnalysisJobIterator itr(this->fAnalysisJobs);
  while(itr.hasNext()){
    TQAnalysisJob* job = itr.readNext();
    job->print(options);
  }
}
  

//______________________________________________________________________________________________

TObjArray * TQCut::getListOfCuts() {
  // Return a list of all cuts in the hierarchy of cuts starting 
  // from this cut downwards

  /* create a new TObjArray and add 
   * this instance of TQCut */
  TObjArray * result = new TObjArray();
  result->Add(this);

  if(!this->fCuts){
    ERRORclass("unable to retrieve child list from cut '%s'!",this->GetName());
    return result;
  }

  /* add the descendant cuts recursively */
  TQCutIterator itr(fCuts);
  while(itr.hasNext()){
    TQCut* c = itr.readNext();
    if(!c) continue;
    TObjArray* list = c->getListOfCuts();
    if(!list){
      ERRORclass("unable to retrieve child list from cut '%s'!",c->GetName());
      continue;
    }
    result->AddAll(list);
    delete list;
  }
 
  /* return the list */
  return result;

}

//______________________________________________________________________________________________


TObjArray* TQCut::getOwnBranches() {
  // add the branches needed by this cut to the internal branch list
  if(!this->fSample){
    throw std::runtime_error("cannot retrieve branches on uninitialized object!");
  }
  TObjArray* branchNames = new TObjArray();

  if(this->fCutObservable){
    DEBUGclass("retrieving branches from cutObservable %s %s", this->getCutExpression().Data(), this->fCutObservable->ClassName());
    TCollection* cutBranches = this->fCutObservable->getBranchNames();
    if(cutBranches){
      cutBranches->SetOwner(false);
      branchNames->AddAll(cutBranches);
      delete cutBranches;
    }
  }
  if(this->fWeightObservable){
    DEBUGclass("retrieving branches from weightObservable %s %s", this->getWeightExpression().Data(), this->fWeightObservable->ClassName());
    TCollection* weightBranches = this->fWeightObservable->getBranchNames();
    if(weightBranches){
      weightBranches->SetOwner(false);
      branchNames->AddAll(weightBranches);
      delete weightBranches;
    }
  }
  return branchNames;
}

//______________________________________________________________________________________________

TObjArray* TQCut::getListOfBranches() {
  // collect all used branch names from cuts, weights and analysis jobs
  // and return then in a TObjArray* of TObjString

  TString expr;
  DEBUGclass("getting own branches");
  TObjArray* branchNames = this->getOwnBranches();

  DEBUGclass("getting analysis job branches");
  {
    /* get all branch names from analysis jobs */
    TQAnalysisJobIterator itr(fAnalysisJobs);
    while(itr.hasNext()){
      TQAnalysisJob* job = itr.readNext();
      if(!job) continue;
      TObjArray* bNames = job -> getBranchNames();
      
#ifdef _DEBUG_
      if(bNames && bNames->GetEntries()>0){
        DEBUGclass("recieved list of branches from analysis job '%s':",job->GetName());
        bNames->Print();
      } else {
        DEBUGclass("recieved empty list of branches from analysis job '%s'",job->GetName());
      }
#endif

      /* strip all valid branch names from each string */
      if(bNames){
        branchNames -> AddAll(bNames);
        bNames->SetOwner(false);
        delete bNames;
      }
    }
  }

  DEBUGclass("getting sub-cut branches");
  {
    TQCutIterator itr(fCuts);
    while(itr.hasNext()){
      TQCut* cut = itr.readNext();
      if(!cut) continue;
      TObjArray* bNames = cut -> getListOfBranches();
      if(bNames){
        branchNames ->AddAll(bNames);
        bNames->SetOwner(false);
        delete bNames;
      }
    }
  }

  DEBUGclass("merging branches");
  TQToken* tok = this->fSample ? this->fSample->getTreeToken() : NULL;
  if(tok){
    tok->setOwner(this);
    TTree* tree = (TTree*)(tok->getContent());
    if(tree){
      TQIterator itr(branchNames);
      while(itr.hasNext()){
        TObject* obj = itr.readNext();
        TString name(obj->GetName());
        if(name.First('*') == kNPOS && !tree->FindBranch(obj->GetName())){
          branchNames->Remove(obj);
          delete obj;
        }
      }
    }
    this->fSample->returnTreeToken(tok);
  }
 
  branchNames -> Compress();

  /* remove duplicated names */
  for (int iName = 0; iName < branchNames -> GetEntries(); iName++) {
    TString name1 = branchNames -> At(iName)->GetName();
    for (int jName = branchNames->GetEntries()-1; jName > iName; jName--) {
      TString name2 = branchNames -> At(jName)->GetName();
      if (name1 == name2) {
        branchNames -> RemoveAt(jName);
      } 
    }
    branchNames -> Compress();
  }
 
  return branchNames;
}


//______________________________________________________________________________________________

bool TQCut::addAnalysisJob(TQAnalysisJob * newJob_, const TString& cuts_) {
  // add a new analysis job to this cut
  // and any descendent cuts matching the string

  /* stop if no analysis job is given */
  if (!newJob_) { return false; }

  if (cuts_.IsNull()) {

    /* make a copy of the job */
    TQAnalysisJob * newJob = newJob_->getClone();

    /* set the parent cut of the analysis job */
    newJob->setCut(this);

    // meaningful title for analysis job
    newJob->SetTitle(TString::Format("%s @ %s", newJob->IsA()->GetName(), this->GetName()));

    /* add this job to the list */
    fAnalysisJobs->Add(newJob);
 
    return true;

  } 

  TString cuts = TQStringUtils::trim(cuts_);
 
  /* prepare the list of cuts */

  bool success = true;
 
    /* loop over all cuts given in the list */
    TQIterator itr(TQStringUtils::tokenize(cuts_), true);
    while(itr.hasNext()){
      TObject* obj = itr.readNext();
      if(!obj) continue;

      /* extract the name */
      TString cutName = TQStringUtils::trim(obj->GetName());

      TObjArray matchingCuts;
      getMatchingCuts(matchingCuts, cutName);
      if (matchingCuts.GetEntries() == 0) {
        ERRORclass("unable to find cut '%s'",cutName.Data());
        success = false;
      }
      for (int i = 0; i < matchingCuts.GetEntries(); i++) {
        TQCut* cut = dynamic_cast<TQCut*>(matchingCuts.At(i));
        if (!cut) {
            continue;
        }
        if (!cut->addAnalysisJob(newJob_)) {
             success = false;
        }
      }
    } 
 
  return success;
}


//______________________________________________________________________________________________

bool TQCut::executeAnalysisJobs(double weight) {
  // Execute analysis jobs attached to this cut instance. The parameter
  // "weight" will be propagated to the job's execute(...) methods.
  // Returns true if every job's execute(...) method returned true and
  // false otherwise

  if (this->fSkipAnalysisJobs || this->fSkipAnalysisJobsGlobal) {
    return true;
  }

  DEBUGclass("executing %d analysis jobs at cut '%s'",fAnalysisJobs->GetEntriesFast(),this->GetName());
  bool success = true;
  this->fJobItr.reset();
  while(this->fJobItr.hasNext()){
    TQAnalysisJob* j = this->fJobItr.readNext();
    if(!j){
      throw std::runtime_error("encountered NULL job in execute!");
    }
    DEBUGclass("executing job '%s'",j->GetName());
    success =  j->execute(weight) && success;
    DEBUGclass("done executing job!");
  }

  return success;
}


//______________________________________________________________________________________________

int TQCut::getNAnalysisJobs() {
  // Return the number of analysis jobs attached to 
  // this cut

  return fAnalysisJobs->GetEntriesFast();
}


//______________________________________________________________________________________________

bool TQCut::passed() const {
  // checks if the currently investigated event passes this cut
  if(!fCutObservable){
    // DEBUGclass("%s: passed, observable NULL",this->GetName());
    return true;
  }
  DEBUGclass("checking '%s'...",this->GetName());
  //  try {
  const double val = fCutObservable->getValue();
  return (val != 0);
  //  } catch (const std::exception& e){
  //    BREAK("ERROR in '%s': %s",fCutObservable->GetName(),e.what());
  //  }
  // DEBUGclass("%s: %s == %f",this->GetName(),fCutObservable->getActiveExpression().Data(),val);
  return false;
}


//______________________________________________________________________________________________

bool TQCut::passedGlobally() const {
  // checks if the currently investigated event passes this cut and all parent cuts
  return (!fBase || fBase->passedGlobally()) && this->passed();
}


//______________________________________________________________________________________________

double TQCut::getWeight() const {
  // retrieve the weight assigned to the current event at this cut
  if (fWeightObservable){
    DEBUGclass("retrieving weight for '%s'...",this->GetName());
    // std::cout << this->GetName() << ":" << fWeightObservable->getExpression() << " = " << fWeightObservable->getValue() << std::endl;
    // std::cout << this->fWeightObservable->getValue() << ": "; this->printWeightComponents();
    return fWeightObservable->getValue();
  } 
  return 1.;
}


//______________________________________________________________________________________________

double TQCut::getGlobalWeight() const {
  // retrieve the weight assigned to the current event by this cut and all parent cuts
  return getWeight() * (fBase ? fBase->getGlobalWeight() : 1.);
}


//______________________________________________________________________________________________

bool TQCut::skipAnalysisJobs(TQSampleFolder * sf) {
  // skips the analysis jobs for some folder

  if (!sf) {
    return false;
  }

  TString skipList;
  //@tag:[~.cc.skipAnalysisJobs] SampleFolder tag specifying a csv list of cuts where the execution of analysis jobs should be skipped. 
  if (!sf->getTagString("~.cc.skipAnalysisJobs", skipList)) {
    return false;
  }

  TQIterator itr(TQStringUtils::tokenize(skipList, ",", true), true);
  while (itr.hasNext()) {
    TString item = itr.readNext()->GetName();
    if ((item.CompareTo("*") == 0) || (TQStringUtils::removeTrailing(item, "*", 1) && this->isDescendantOf(item))
        || (item.CompareTo(this->GetName()) == 0)) {
      //@tag: [.cc.nSkippedAnalysisJobs.<cutName>] This sampleFolder tag contains the number of analysis jobs skipped.
      sf->setTagInteger(TString::Format(".cc.nSkippedAnalysisJobs.%s", this->GetName()),
                        fAnalysisJobs->GetEntriesFast());
      return true;
    }
  }

  return false;
}


//______________________________________________________________________________________________

bool TQCut::isDescendantOf(TString cutName) {
  // returns true if this cut is a descendant of the cut identified by the name given as argument 
  // returns false otherwise

  if (fBase) {
    if (cutName.CompareTo(fBase->GetName()) == 0) {
      // parent cut is the one looked for
      return true;
    } else {
      // check parent cut
      return fBase->isDescendantOf(cutName);
    }
  } else {
    // no parent cut present
    return false;
  }
}

//______________________________________________________________________________________________

bool TQCut::initializeObservables() {
  // initialize the observables directly related to this cut on the given sample

  DEBUGclass("\tcut: %s",this->fCutExpression.Data());
  DEBUGclass("\tweight:%s",this->fWeightExpression.Data());

  if(!this->fCutExpression.IsNull() ){
    this->fCutObservable = TQObservable::getObservable(this->fCutExpression,this->fSample);
    if(!this->fCutObservable) return false;
    if(!fCutObservable->initialize(this->fSample)){
      ERRORclass("Failed to initialize cut observable with expression '%s' for cut '%s'.",this->fCutExpression.Data(),this->GetName());
      this->fCutObservable = NULL;
      return false;
    }
    //#ifdef _DEBUG_
//    if(!TQStringUtils::equal(fCutObservable->getExpression(),this->fCutExpression.Data())){
//      throw std::runtime_error(TString::Format("cut '%s' has retrieved dissonant cut observable '%s' for '%s'",this->GetName(),fCutObservable->getExpression().Data(),this->fCutExpression.Data()).Data());
//    }
    //#endif
    DEBUGclass("cut '%s' with expression '%s' has retrieved cut observable '%s' with expression '%s'",this->GetName(),this->fCutExpression.Data(),this->fCutObservable->GetName(),this->fCutObservable->getExpression().Data());
  }

  if(!this->fWeightExpression.IsNull() ){
    this->fWeightObservable = TQObservable::getObservable(this->fWeightExpression,this->fSample);
    if(!this->fWeightObservable) return false;
    if(!fWeightObservable->initialize(this->fSample)){
      ERRORclass("Failed to initialize weight observable with expression '%s' for cut '%s'.",this->fWeightExpression.Data(),this->GetName());
      this->fWeightObservable = NULL;
      this->fCutObservable = NULL;
      return false;
    }
    //#ifdef _DEBUG_
//    if(!TQStringUtils::equal(fWeightObservable->getExpression(),this->fWeightExpression.Data())){
//      throw std::runtime_error(TString::Format("cut '%s' has retrieved dissonant weight observable '%s' for '%s'",this->GetName(),fWeightObservable->getExpression().Data(),this->fWeightExpression.Data()).Data());
//    }
    //#endif
    DEBUGclass("cut '%s' with expression '%s' has retrieved weight observable '%s' with expression '%s'",this->GetName(),this->fWeightExpression.Data(),this->fWeightObservable->GetName(),this->fWeightObservable->getExpression().Data());
  }
  return true;
}

//______________________________________________________________________________________________

bool TQCut::finalizeObservables() {
  // finalize the observables directly related to this cut on the given sample

  bool retval = true;
  if (fCutObservable){
    DEBUGclass("finalizing observable '%s'",this->fCutObservable->GetName());
    if(!fCutObservable->finalize()) retval = false;
    this->fCutObservable = NULL;
  }
  
  if (fWeightObservable){
    DEBUGclass("finalizing observable '%s'",this->fWeightObservable->GetName());
    if(!fWeightObservable->finalize()) retval = false;
    this->fWeightObservable = NULL;
  }
  return retval;
}

//______________________________________________________________________________________________

bool TQCut::initialize(TQSample * sample) {
  // initialize this cut and all observables on the given sample
  if (!sample) {
    throw std::runtime_error(TString::Format("cannot initialize cut '%s' with NULL sample",this->GetName()).Data());
    return false;
  }
  
  TQSampleFolder* previousInit = fInitializationHistory.size()>0 ? fInitializationHistory[fInitializationHistory.size()-1] : nullptr;
  if ( previousInit && !(previousInit->areRelated(sample)>0) ) { //this cut was initialized before on a sample folder but the sample is not a descendant folder of the sample folder -> things are prone to get inconsistent here, abort!
    throw std::runtime_error(TString::Format("Caught attempt to initialize cut '%s' using sample with path '%s' while it was previously initialized on sample folder '%s'. Either TQCut::finalizeSampleFolder was not called correctly or something went terribly wrong.", this->GetName(), sample->getPath().Data(), previousInit->getPath().Data() ).Data());
    return false;
  }
  
  if (fTreeToken){ 
    return false;
  }
 
  DEBUGclass("retrieving tree");

  /* try to get tree token */
  this->fTreeToken = sample->getTreeToken();
 
  if (!this->fTreeToken){
    throw std::runtime_error(TString::Format("unable to initialize cut '%s': unable to obtain tree token",this->GetName()).Data());
    DEBUGclass("unable to retrieve tree token");
    return false;
  }

  this->fTreeToken->setOwner(this);
  this->fSample = sample;
  this->fTree = (TTree*)(fTreeToken->getContent());

  if(!fTree){
    DEBUGclass("received invalid tree pointer");
    sample->returnTreeToken(fTreeToken);
    throw std::runtime_error(TString::Format("unable to initialize cut '%s': received invalid tree pointer",this->GetName()).Data());
    return false;
  }

  // check whether to skip assiociated analysis jobs for this sample
  this->fSkipAnalysisJobs = skipAnalysisJobs(sample);

  if(fCuts){
    /* initialize descendant cuts */
    DEBUGclass("iterating over child cuts");
    this->fCutItr.reset();
    while(this->fCutItr.hasNext()){
      TQCut* c = this->fCutItr.readNext();
      if(!c) throw std::runtime_error("encountered NULL cut");
      if(!c->initialize(sample)) throw std::runtime_error(TString::Format("unable to initialize cut '%s'",c->GetName()).Data());
    }
  }
  /* initialize analysis jobs */
  if ( (!this->fSkipAnalysisJobs) && (!this->fSkipAnalysisJobsGlobal) ) {
    DEBUGclass("iterating over analysis jobs for cut '%s'",this->GetName());
    TQAnalysisJobIterator itr(fAnalysisJobs);
    while(itr.hasNext()){
      TQAnalysisJob* job = itr.readNext();
      if(!job) throw std::runtime_error("encountered NULL job");
      if(!job->initialize(sample)) throw std::runtime_error(TString::Format("unable to initialize analysis job %s at %s",job->GetName(),this->GetName()).Data());
    }
  }
  
  if(!this->initializeObservables()) throw std::runtime_error(TString::Format("unable to initialize observables for cut '%s'",this->GetName()).Data());
  
  return true;
}

//______________________________________________________________________________________________

bool TQCut::finalize() {
  // finalize this cut and all observables on the given sample
  this->finalizeObservables();

  bool success = true;
  
  /* finalize descendant cuts */
  this->fCutItr.reset();
  while(this->fCutItr.hasNext()){
    TQCut* cut = this->fCutItr.readNext();
    if(!cut) continue;
    if(!cut->finalize()) success = false;
  }

  /* finalize analysis jobs */
  if ( (!fSkipAnalysisJobsGlobal) && (!fSkipAnalysisJobs) ) { //only finalize if they were also initialized
    this->fJobItr.reset();
    while(this->fJobItr.hasNext()){
      TQAnalysisJob* job = this->fJobItr.readNext();
      if(!job) continue;
      if(!job->finalize()) success = false;
    }
  }
  
  if(this->fSample){
    success = this->fSample->returnToken(this->fTreeToken) && success;
  }
  
  this->fTreeToken = 0;
  this->fSample = 0;
  this->fTree = 0;
  
  return success;
}

//______________________________________________________________________________________________

bool TQCut::canInitialize(TQSampleFolder* sf) const {
  // returns true if the sample folder is eligible for initializing this cut on
  if (!sf) return false;
  TQSampleFolder* previousInit = fInitializationHistory.size()>0 ? fInitializationHistory[fInitializationHistory.size()-1] : nullptr;
  if ( previousInit && !(previousInit->areRelated(sf)>0) ) { //this cut was initialized before on a sample folder but the new folder (sf) is not a descendant folder of the previous one -> This is not a valid sample folder to initialize this cut on!
    DEBUGclass("Sample (folder) with path '%s' is not eligible for initializing cut '%s'. The cut was last initialized on '%s'.",sf->getPath().Data(),this->GetName(),previousInit->getPath().Data());
    return false;
  }
  return true; //no reason not to initialize, go ahead!
}



bool TQCut::initializeSampleFolder(TQSampleFolder* sf){
  // initialize this cut and all observables on the given sample folder
  this->initializeSelfSampleFolder(sf);
  
  //check if we are initializing on a valid sample folder, i.e., if we already initialized on some sample folder before the new one has to be a subfolder
  if ( !this->canInitialize(sf) ) { //this cut was initialized before on a sample folder but the new folder (sf) is not a descendant folder of the previous one -> things are prone to get inconsistent here, abort!
    if (!sf) {//check if it is "only" a nullptr
      WARNclass("Cannot initialize cut '%s' with a nullptr to a TQSampleFolder",this->GetName());
      return false;
    }
    TQSampleFolder* previousInit = this->fInitializationHistory.size()>0 ? fInitializationHistory[fInitializationHistory.size()-1] : nullptr;
    throw std::runtime_error(TString::Format("Caught attempt to initialize cut '%s' using sample folder with path '%s' while it was previously initialized on sample folder '%s'. Either TQCut::finalizeSampleFolder was not called or something went terribly wrong.", this->GetName(), sf->getPath().Data(), previousInit==nullptr? "<none>" : previousInit->getPath().Data() ).Data());
    return false;
  } 
  
  fInitializationHistory.push_back(sf); 
  if (!fBase) {
    DEBUGclass("Initializing cut '%p' on sample folder '%s'",this,sf->getPath().Data());
  }

  bool success = true;

  /* initialize descendant cuts */
  this->fCutItr.reset();
  while(this->fCutItr.hasNext()){
    TQCut* cut = this->fCutItr.readNext();
    if(!cut) continue;
    if(!cut->initializeSampleFolder(sf)) success = false;
  }

  /* initialize analysis jobs */
  if ( (!fSkipAnalysisJobsGlobal) && (!fSkipAnalysisJobs) ) {
    this->fJobItr.reset();
    while(this->fJobItr.hasNext()){
      TQAnalysisJob* job = this->fJobItr.readNext();
      if(!job) continue;
      if(!job->initializeSampleFolder(sf)) success = false;
    }
  }
  
  return success;
}

//______________________________________________________________________________________________

bool TQCut::canFinalize(TQSampleFolder* sf) const {
  // returns true if this cut was previously initialized on this sample folder
  if (!sf) return false;
  if (sf == this->fSample) return true; //the acutal sample is not stored in the fInitializationHistoy
  for (auto hist : fInitializationHistory) {
    if (sf == hist) return true;
  }
  //no folder in the history matched, i.e., this cut was not previously initialized on the sample folder
  return false;
}


bool TQCut::finalizeSampleFolder(TQSampleFolder* sf){
  // finalize this cut and all observables on the given sample folder
  
  //if we still have a sample we should finalize it first
  if (this->fSample && !this->finalize()) return false;
  if (!sf) return false;
  
  this->finalizeSelfSampleFolder(sf);
  
  if (!fBase) {
    DEBUGclass("Finalizing cut '%p' on sample folder '%s'. History has %d entries",this,sf->getPath().Data(),(int)fInitializationHistory.size());
  }
  
  TQSampleFolder* lastInit = fInitializationHistory.size()>0 ? fInitializationHistory[fInitializationHistory.size()-1] : nullptr;
  while (lastInit != nullptr && (lastInit->areRelated(sf)<0) ) { //read: while the last initialization was performed on a descendant folder of sf (excluding lastInit==sf)
    if (!this->finalizeSampleFolder(lastInit)) return false; //recursively go back in history to finalize at all places this cut was initialized
    lastInit = fInitializationHistory.size()>0 ? fInitializationHistory[fInitializationHistory.size()-1] : nullptr;
  }
  //At this point lastInit should either be equal to sf (in which case areRelated returns 1) or the folders have no relation to each other (=error!). We also throw an error if it was requested to finalize this cut on any folder if it has never been initialized on a sample folder (case lastInit==nullptr).
  
  if ( !lastInit || lastInit != sf ) { //this cut was not initialized on 'sf' -> things are prone to get inconsistent here, abort!
    throw std::runtime_error(TString::Format("Caught attempt to finalize cut '%s' using sample folder with path '%s' while it was not initialized on it.", this->GetName(), sf->getPath().Data() ).Data());
    return false;
  }
  if (fInitializationHistory.size()>0) fInitializationHistory.pop_back(); //remove last element (which at this point should be ensured to be 'sf')
    
  bool success = true;

  /* finalize descendant cuts */
  this->fCutItr.reset();
  while(this->fCutItr.hasNext()){
    TQCut* cut = this->fCutItr.readNext();
    if(!cut) continue;
    if(!cut->finalizeSampleFolder(sf)) success = false;
  }

  /* finalize analysis jobs */
  if ( (!fSkipAnalysisJobsGlobal) && (!fSkipAnalysisJobs) ) { //only finalize if they were also initialized
    this->fJobItr.reset();
    while(this->fJobItr.hasNext()){
      TQAnalysisJob* job = this->fJobItr.readNext();
      if(!job) continue;
      if(!job->finalizeSampleFolder(sf)) success = false;
    }
  }
  
  return success;
}


//______________________________________________________________________________________________

TQCut::~TQCut() {
  // destructor
 
  /* finalize cuts */
  this->finalize();
 
  /* delete descendant cuts */
  this->fCuts->Delete();
  /* delete the list of cuts itself */
  delete fCuts;
 
  /* delete analysis jobs */
  this->fAnalysisJobs->Delete();
  /* delete the list of analysis jobs itself */
  delete fAnalysisJobs;
}

//______________________________________________________________________________________________

void TQCut::clearAnalysisJobs(){
  // clear all analysis jobs from this cut
  if(fAnalysisJobs)
    fAnalysisJobs->Delete();
}

//__________________________________________________________________________________|___________

TQCut * TQCut::getSingleCut(TString name, TString excl_pattern) {
  // Return the cut matching "name". The hierarchy of cuts will be searched 
  // starting from this cut downwards. Please note: the result might be a null 
  // pointer, if no matching element can be found
  // The difference between this function and getCut() is that this function
  // will return TQCut with standalone single cut (all the hierarchy collapsed
  // down to the cut requested.)

  TString tmpName = name;
  bool addDescendants = TQStringUtils::removeTrailing(tmpName, "*") > 0;

  /* Is this instance the cut that is requested? */
  if (tmpName.CompareTo(this->GetName()) == 0) {

    /* get base cut */
    TQCut * tmpcut = this->getBase();

    /* create return object */
    TQCut * aggregatedCut = new TQCut(this->GetName());
    TString thiscutname = this->GetName();

    if (thiscutname.Contains(excl_pattern)) {
      /* if this cut is the one being excluded the aggregated cut will start with null cut */
      aggregatedCut->setCutExpression("1");
      aggregatedCut->setWeightExpression("1");
    } else {
      /* if not we start with this cut */
      aggregatedCut->setCutExpression(this->getCutExpression());
      if (!(this->getWeightExpression().IsNull()))
        aggregatedCut->setWeightExpression(this->getWeightExpression());
      else
        aggregatedCut->setWeightExpression("1");
    }

    /* if requested also add all the necessary descendant cuts */
    if (addDescendants)
      aggregatedCut->setCuts(this->getCuts());

    while (tmpcut) {
      TString cutexprorg = aggregatedCut->getCutExpression();
      TString cutexprnew = tmpcut->getCutExpression();
      TString cutnamenew = tmpcut->GetName();
      /* if exclusion pattern exists for the cuts above we skip that one. */
      if (cutnamenew.Contains(excl_pattern)) {
        tmpcut = tmpcut->getBase();
        continue;
      }
      if (!cutexprnew.IsNull())
        aggregatedCut->setCutExpression("(" + cutexprorg + ")*(" + cutexprnew + ")");
      TString wgtexprorg = aggregatedCut->getWeightExpression();
      TString wgtexprnew = tmpcut->getWeightExpression();
      if (!wgtexprnew.IsNull())
        aggregatedCut->setWeightExpression("(" + wgtexprorg + ")*(" + wgtexprnew + ")");
      tmpcut = tmpcut->getBase();
    }

    /* before returning print the full cut expr */
    TString cutexpr = aggregatedCut->getCutExpression();
    cutexpr.ReplaceAll(" ","");
    //std::cout << "" << std::endl;
    //std::cout << name << " " << excl_pattern<< std::endl;
    //std::cout << "-----------------------" << std::endl;
    //std::cout << cutexpr << std::endl;
    //std::cout << "-----------------------" << std::endl;
    //std::cout << "" << std::endl;

    /* return the aggregatedCut if it is this. */
    return aggregatedCut;

  } else {
 
    /* if not, look for the cut that is requested
     * recursively in the list of descendant cuts */
    this->fCutItr.reset();
    while (this->fCutItr.hasNext()){
      TQCut* c = this->fCutItr.readNext();
      if(!c) continue;
      TQCut* found = c->getSingleCut(name, excl_pattern);
      if(found) return found;
    }

    /* return the cut that might have been found */ 
    return NULL;

  }
}

//__________________________________________________________________________________|___________

TQSample* TQCut::getSample(){
  // retrieve the sample currently assigned to this cut
  return this->fSample;
}


//__________________________________________________________________________________|___________

TQCut* TQCut::getClone(){
  // return a clone of this cut (and all descendants)
  TQCut* clone = new TQCut(this->GetName(),this->GetTitle(),this->getCutExpression(),this->getWeightExpression());
  clone->setSkipAnalysisJobsGlobal(this->getSkipAnalysisJobsGlobal());
  
  this->fJobItr.reset();
  while(this->fJobItr.hasNext()){
    TQAnalysisJob* j = this->fJobItr.readNext();
    clone->addAnalysisJob(j->getClone());
  }    
  this->fCutItr.reset();
  while(this->fCutItr.hasNext()){
    TQCut* c = this->fCutItr.readNext();
    TQCut* cl = c->getClone();
    clone->addCut(cl);
  }
  return clone;
}

//__________________________________________________________________________________|___________

TQCut* TQCut::getCompiledClone(TQTaggable* tags){
  // return a clone of this cut (and all descendants)
  if(!tags) return this->getClone();
  
  TQCut* clone = new TQCut(this->GetName(),this->GetTitle(),
                           tags->replaceInText(this->getCutExpression()),
                           tags->replaceInText(this->getWeightExpression()));
  
  this->fJobItr.reset();
  while(this->fJobItr.hasNext()){
    TQAnalysisJob* j = this->fJobItr.readNext();
    clone->addAnalysisJob(j->getClone());
  }    
  this->fCutItr.reset();
  while(this->fCutItr.hasNext()){
    TQCut* c = this->fCutItr.readNext();
    TQCut* cl = c->getCompiledClone(tags);
    clone->addCut(cl);
  }
  return clone;
}


//__________________________________________________________________________________|___________

void TQCut::analyse(double weight, bool useWeights) {
  // apply this cut and execute the analyis jobs appended to this cut
  const bool passed = this->passed();
  if (!passed) {
    return;
  }
  DEBUGclass("passed cut '%s' with expression '%s'",this->GetName(),this->fCutObservable ? this->fCutObservable->getActiveExpression().Data() : "");
  if (useWeights) {
    weight *= this->getWeight();
  }
  this->executeAnalysisJobs(weight);
  this->fCutItr.reset();
  while(this->fCutItr.hasNext()){
    this->fCutItr.readNext()->analyse(weight, useWeights);
  } 
}

//__________________________________________________________________________________|___________

bool TQCut::initializeSelfSampleFolder(TQSampleFolder*/*sf*/){  return true;}

//__________________________________________________________________________________|___________

bool TQCut::finalizeSelfSampleFolder  (TQSampleFolder*/*sf*/){  return true;}
