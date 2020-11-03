#include "QFramework/TQGraphMakerAnalysisJob.h"
#include "QFramework/TQIterator.h"
#include "QFramework/TQHistogramUtils.h"
#include "QFramework/TQObservable.h"
#include "QFramework/TQStringUtils.h"
#include "QFramework/TQCompiledCut.h"
#include "QFramework/TQTaggable.h"
#include "TObjArray.h"
#include "TList.h"
#include "TObjString.h"
#include "QFramework/TQSample.h"
#include "TFolder.h"
#include "TGraph.h"
#include "TGraph2D.h"
#include "QFramework/TQUtils.h"
#include "TH1.h"
#include "TH1F.h"
#include "TH2F.h"

// #define _DEBUG_
#include "QFramework/TQLibrary.h"

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <cstdlib>

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQGraphMakerAnalysisJob:
//
// The TQGraphMakerAnalysisJob allows to book instances of TGraph
// showing multi-dimensional distributions of variables in a way
// similar to the TQHistoMakerAnalysisJob.
//
//   TGraph('subleadLepPt', '') << ( $(lep1).pt() : 'Number of Leptons', [Weight_$(weightname):$(cand)]*[SampleNorm] : 'Event Weight' );
// 
//   @CutMET: subleadLepPt;
//
// Caution: The TGraphs can become extremely large for large numbers
// of events!
//
////////////////////////////////////////////////////////////////////////////////////////////////

TQMessageStream TQGraphMakerAnalysisJob::f_ErrMsg(new std::stringstream());

ClassImp(TQGraphMakerAnalysisJob)

//__________________________________________________________________________________|___________

TQGraphMakerAnalysisJob::TQGraphMakerAnalysisJob() : 
TQAnalysisJob("TQGraphMakerAnalysisJob"),
  f_Verbose(0),
  poolAt(NULL)
{
  // standard constructor
}

//__________________________________________________________________________________|___________

TQGraphMakerAnalysisJob::TQGraphMakerAnalysisJob(TQGraphMakerAnalysisJob* other) :
  TQAnalysisJob(other ? other->GetName() : "TQGraphMakerAnalysisJob"),
  f_Verbose(other ? other->f_Verbose : 0),
  poolAt(other ? other->poolAt : NULL)
{
  // copy constructor
  for(size_t i=0; i<other->fGraphTemplates.size(); ++i){
    this->fGraphTemplates.push_back(TQHistogramUtils::copyGraph(other->fGraphTemplates[i]));
    this->fGraphTypes.push_back(other->fGraphTypes[i]);
    this->fExpressions.push_back(std::vector<TString>());
    for(size_t j=0; j<other->fExpressions[i].size(); ++j){
      this->fExpressions[i].push_back(other->fExpressions[i][j]);
    }
  }
}

//__________________________________________________________________________________|___________

void TQGraphMakerAnalysisJob::setVerbose(int verbose) {
  // set verbosity
  f_Verbose = verbose;
}


//__________________________________________________________________________________|___________

int TQGraphMakerAnalysisJob::getVerbose() {
  // retrieve verbosity
  return f_Verbose;
}


//__________________________________________________________________________________|___________

void TQGraphMakerAnalysisJob::setErrorMessage(TString message) {
 
  // update the error message 
  f_ErrMsg.sendClassFunctionMessage(TQMessageStream::INFO,this->Class(),"<anonymous>",message);
 
  // print the error message if in verbose mode 
  if (f_Verbose > 0) INFOclass(message);
}

//__________________________________________________________________________________|___________

void TQGraphMakerAnalysisJob::clearMessages(){
  // clear the error messages
  f_ErrMsg.clearMessages();
}

//__________________________________________________________________________________|___________

TString TQGraphMakerAnalysisJob::getErrorMessage() {
  // Return the latest error message
  return f_ErrMsg.getMessages();
}


//__________________________________________________________________________________|___________

const TString& TQGraphMakerAnalysisJob::getValidNameCharacters() {
  // retrieve a string with valid name characters
  return TQFolder::getValidNameCharacters();
}


//__________________________________________________________________________________|___________

void TQGraphMakerAnalysisJob::cancelGraph(const TString& name) {
  // cancel the histgogram with the given name
  for(size_t i=fGraphTemplates.size(); i >0 ; i--){
    if(fGraphTemplates.at(i)->GetName() == name){
      fGraphTemplates.erase(fGraphTemplates.begin()+i-1);
      fExpressions.at(i-1).clear();
      fExpressions.erase(fExpressions.begin()+i-1);
      fGraphTypes.erase(fGraphTypes.begin()+i-1);
      return;
    }
  }
}

//__________________________________________________________________________________|___________

bool TQGraphMakerAnalysisJob::bookGraph(TString definition, TQTaggable* aliases) {
	// book a new graph, given a definition
  DEBUGclass("entering function - booking graph '%s'",definition.Data());

  if(definition.IsNull()){
    this->f_ErrMsg.sendClassFunctionMessage(TQMessageStream::INFO,this->Class(),__FUNCTION__,"obtained empty graph definition");
    return false;
  }
  
  // graph definition
  TString graphDef;
  TQStringUtils::readUpTo(definition, graphDef, "<", "()[]{}", "''\"\"");

  if(graphDef.IsNull()){
    this->f_ErrMsg.sendClassFunctionMessage(TQMessageStream::INFO,this->Class(),__FUNCTION__,"graph constructor is empty, remainder is '%s'",definition.Data());
    return false;
  }

  // create graph template from definition
  DEBUGclass("creating graph '%s'",graphDef.Data());
  TString msg; 
  TNamed * graph = TQHistogramUtils::createGraph(graphDef, msg);
  DEBUGclass(graph ? "success" : "failure");

  if (!graph) {
    this->f_ErrMsg.sendClassFunctionMessage(TQMessageStream::ERROR,this->Class(),__FUNCTION__,msg);
    return false;
  }

  // invalid name?
  if (!TQFolder::isValidName(graph->GetName())) {
    this->f_ErrMsg.sendClassFunctionMessage(TQMessageStream::INFO,this->Class(),__FUNCTION__,"'%s' is an invalid graph name", graph->GetName());
    delete graph;
    return false;
  }

  // read "<<" operator
  if (TQStringUtils::removeLeading(definition, "<") != 2) {
    this->f_ErrMsg.sendClassFunctionMessage(TQMessageStream::INFO,this->Class(),__FUNCTION__,"Operator '<<' expected after graph definition");
    delete graph;
    return false;
  }

  //split off a possile option block
  std::vector<TString> settingTokens = TQStringUtils::split(definition, "<<", "([{'\"", ")]}'\"");
  if (settingTokens.size()<1) {
    this->f_ErrMsg.sendClassFunctionMessage(TQMessageStream::INFO,this->Class(),__FUNCTION__,"cannot parse definition block '%s'",definition.Data());
    delete graph; 
    return false;
  }
  definition = settingTokens[0];
  TQTaggable options;
  if (settingTokens.size()>1) {
    TString optionBlock;
    TQStringUtils::readBlanksAndNewlines(settingTokens[1]);
    if (!TQStringUtils::readBlock(settingTokens[1], optionBlock, "()[]{}", "''\"\"")) {
      this->f_ErrMsg.sendClassFunctionMessage(TQMessageStream::INFO,this->Class(),__FUNCTION__,"Failed to parse graph option block '%s'", settingTokens[1].Data());
      delete graph;
      return false;
    }
    options.importTags(optionBlock);
  }
  
  // read expression block
  TString expressionBlock;
  TQStringUtils::readBlanksAndNewlines(definition);
  if (!TQStringUtils::readBlock(definition, expressionBlock, "()[]{}", "''\"\"")) {
    this->f_ErrMsg.sendClassFunctionMessage(TQMessageStream::INFO,this->Class(),__FUNCTION__,"Missing expression block after '<<' operator");
    delete graph;
    return false;
  }

  // tokenize expression block (one token per dimension)
  TList * expressionTokens = TQStringUtils::tokenize(expressionBlock, ",", true, "()[]{}", "''\"\"");
  if(expressionTokens->GetEntries() < 1){
    this->f_ErrMsg.sendClassFunctionMessage(TQMessageStream::INFO,this->Class(),__FUNCTION__,"cannot parse expression block '%s'",expressionBlock.Data());
    delete graph;
    return false;
  }

  DEBUGclass("parsing expression block '%s', found %d entries",expressionBlock.Data(),expressionTokens->GetEntries());

  // read expression block tokens
  std::vector<TString> exprs;
  std::vector<TString> titles;
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
        delete graph;
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
    TString title = TQStringUtils::unquote(token);
    
    // store expression and title
    const TString expression(aliases ? aliases->replaceInTextRecursive(expr) : expr);
    exprs.push_back(TQStringUtils::trim(expression));
    titles.push_back(TQStringUtils::trim(title));
    DEBUGclass("found expression and title: '%s' and '%s'",expr.Data(),title.Data());
  }

  // graph properties
  TString name = graph->GetName();
  int dim = (graph->InheritsFrom(TGraph2D::Class()) ? 3 : 2);

  // check dimension of graph and expression block
  if ( ( dim > 0 && dim != (int)exprs.size() ) || ( dim < 0 && (int)exprs.size() != abs(dim)+1) ) { // last ist the TProfile case
    this->f_ErrMsg.sendClassFunctionMessage(TQMessageStream::INFO,this->Class(),__FUNCTION__,"Dimensionality of graph (%d) and expression block (%d) don't match", dim, (int)(exprs.size()));
    delete graph;
    DEBUGclass("Dimensionality of graph (%d) and expression block (%d) don't match", dim, (int)(exprs.size()));
    return false;
  }

  // check name of graph
  if (!TQStringUtils::isValidIdentifier(name, getValidNameCharacters())) {
    this->f_ErrMsg.sendClassFunctionMessage(TQMessageStream::INFO,this->Class(),__FUNCTION__,"Invalid graph name '%s'", name.Data());
    DEBUGclass("Invalid graph name '%s'", name.Data());
    delete graph;
    return false;
  }

  // stop if graph with 'name' already has been booked
  bool exists = false;
  int i = 0;
  while (!exists && i < (int)(fGraphTemplates.size()))
    exists = (name.CompareTo(fGraphTemplates.at(i++)->GetName()) == 0);
  if (exists) {
    this->f_ErrMsg.sendClassFunctionMessage(TQMessageStream::INFO,this->Class(),__FUNCTION__,"Graph with name '%s' has already been booked", name.Data());
    DEBUGclass("Graph with name '%s' has already been booked", name.Data());
    delete graph;
    return false;
  }
 
  // set up tree observables corresponding to expressions
  for (int i = 0; i < (int)exprs.size(); i++) {
		TAxis* axis = TQHistogramUtils::getAxis(graph,i);
		axis->SetTitle(titles[i]);
    axis->SetName(exprs[i]);
  }
  
  fExpressions.push_back(exprs);
  fGraphTypes.push_back(graph->IsA());
  fGraphTemplates.push_back(graph);
  return true;
}

//__________________________________________________________________________________|___________

void TQGraphMakerAnalysisJob::print(const TString& options) {
  // overloading the standard print-routine, internally calling printBooking[TeX], depending on options
  if(options.Contains("TeX")){
    this->printBookingTeX(this->GetName());
  } else {
    this->printBooking(this->GetName());
  }
}

//__________________________________________________________________________________|___________

void TQGraphMakerAnalysisJob::printBookingTeX(const TString& moretext) {
  // print booked graphs (LaTeX format)
  std::cout << "\\begin{tabular}{l l l l }" << std::endl;;
  std::cout << "\\multicolumn{2}{l}{\\bfseries Booked Graphs}" << " & \\multicolumn{2}{l}{" << moretext  << "}\\tabularnewline" << std::endl;
  for (size_t i = 0;i < fGraphTemplates.size(); i++) {
    TString exp(TQStringUtils::concat(fExpressions.at(i),","));
    exp.ReplaceAll("$","\\$");
    std::cout << fGraphTemplates.at(i)->GetName() << " & "
              << fGraphTemplates.at(i)->GetTitle()  << " & "
              << TQHistogramUtils::getDetailsAsString(fGraphTemplates.at(i), 2)  << " & "
              << exp;
    if(i != fGraphTemplates.size() -1) std::cout << " \\tabularnewline ";
    std::cout<< std::endl;
  }
  std::cout << "\\end{tabular}" << std::endl;
}

//__________________________________________________________________________________|___________

void TQGraphMakerAnalysisJob::printBooking(const TString& moretext) {
  // print booked graphs
  std::cout << TQStringUtils::makeBoldBlue("Booked Graphs") << " " << TQStringUtils::makeBoldWhite(moretext) << std::endl;
  for (size_t i = 0;i < fGraphTemplates.size(); i++) {
    std::cout << TQStringUtils::fixedWidth(fGraphTemplates.at(i)->GetName(),20)
              << TQStringUtils::fixedWidth(fGraphTemplates.at(i)->GetTitle(),20)
              << TQHistogramUtils::getDetailsAsString(fGraphTemplates.at(i), 2)
              << " << " << TQStringUtils::concat(fExpressions.at(i)," : ")
              << std::endl;
  }
}


//__________________________________________________________________________________|___________

bool TQGraphMakerAnalysisJob::execute(double/*weight*/) {
  DEBUGclass("filling graphs for event...");
  // execute this analysis job, filling all graphs
  for (unsigned int i = 0; i < fGraphs.size(); ++i) {
    #ifdef _DEBUG_
    if(this->fObservables.size() < i){
      throw std::runtime_error("insufficient size of observable vector!");
    }
    #endif
    if(fGraphTypes[i] == TGraph::Class()){
      TGraph* g = (TGraph*)(fGraphs[i]);
			// aggressively expand the graph memoy to avoid frequent reallocation
			g->Expand(g->GetN()+10,1000);
      TRY(
					g->SetPoint(g->GetN(),fObservables[i][0]->getValue(),fObservables[i][1]->getValue())
					,TString::Format("Failed to add event to graph '%s' using the observable '%s' at cut '%s'.", g->GetName(), fObservables[i][0]->GetName(), this->getCut()->GetName())
					);
    }
  }
  return true;
}

//__________________________________________________________________________________|___________

TObjArray * TQGraphMakerAnalysisJob::getBranchNames() {
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

bool TQGraphMakerAnalysisJob::initializeSelf() {
  // initialize this analysis job
  DEBUGclass("initializing analysis job '%s'",this->GetName());

  if(fGraphs.size() < 1){
    this->poolAt = this->fSample;
    DEBUGclass("initializing graphs");
    this->initializeGraphs();
  }

  bool success = true;
  /* initialize TQObservables */
  DEBUGclass("initializing observables");
  for (unsigned int i = 0; i < fExpressions.size(); ++i) {
    std::vector<TQObservable*> observables;
    for (unsigned int j = 0; j < fExpressions[i].size(); ++j) {
      TString expr(fExpressions[i][j]);
      TQObservable* obs = TQObservable::getObservable(expr,this->fSample);
      if(obs && success){
        DEBUGclass("initializing...");
        if (!obs->initialize(this->fSample)) {
          ERRORclass("Failed to initialize observable created from expression '%s' for sample '%s' in TQGraphMAnalysisJob '%s' for graph named '%s'",this->fExpressions[i][j].Data(), this->fSample->getPath().Data(), this->GetName(), this->fGraphs[i]->GetName());
          success=false;
        }
        DEBUGclass("initialized observable '%s' of type '%s' with '%s'",
                   obs->getExpression().Data(),
                   obs->ClassName(),
                   obs->getActiveExpression().Data());
      }
      if(!obs){
        DEBUGclass("creating const observable");
        obs = TQObservable::getObservable("Const:nan",this->fSample);
        obs->initialize(this->fSample);
      }
      observables.push_back(obs);
    }
    this->fObservables.push_back(observables);
  }
  DEBUG("successfully initialized graph job");
  return success;
}

//__________________________________________________________________________________|___________

void TQGraphMakerAnalysisJob::initializeGraphs(){
  // create graphs from templates 
  DEBUGclass("Size of graph template vector : %i", fGraphTemplates.size());
  for (unsigned int i = 0; i < fGraphTemplates.size(); i++) {
    
    // copy/clone the template graph
    TNamed * graph = TQHistogramUtils::copyGraph((fGraphTemplates)[i]);
    fGraphs.push_back(graph);
  }
}
 

//__________________________________________________________________________________|___________

bool TQGraphMakerAnalysisJob::finalizeSelf() {
  // finalize TQObservables
  for (unsigned int i = 0; i < fObservables.size(); ++i) {
    for (unsigned int j = 0; j < fObservables[i].size(); ++j) {
      fObservables[i][j]->finalize();
    }
  }
  this->fObservables.clear();

  if(this->poolAt == this->fSample)
    if(!this->finalizeGraphs())
      return false;
  
  return true;
}

//__________________________________________________________________________________|___________

bool TQGraphMakerAnalysisJob::finalizeGraphs(){
  // store the graphs in the sample folder hierarchy
  if (!getCut()) { return false; }

  /* get the graph folder */
  TQFolder * folder = this->poolAt->getFolder(TString::Format(
                                                              ".graphs/%s+", getCut()->GetName()));
  if (!folder) { return false; }
  DEBUGclass("successfully created folder for cut %s", getCut()->GetName());

  /* scale and store graphs */
  DEBUGclass("length of graph list is %i", fGraphs.size());
  for (unsigned int i = 0; i < fGraphs.size(); i++) {
    TNamed * graph = (fGraphs)[i];
    if (!graph){ DEBUGclass("Graph is 0!"); };
    /* delete existing graph */
    TObject *g = folder->FindObject(graph->GetName());
    if (g){
      DEBUGclass("removing previous object %s", g->GetName());
      folder->Remove(g);
    }
		// stupidly enough, TGraph resets the axis titles whenever a point is added, so now, we need to copy the axis titles again
		TQHistogramUtils::copyGraphAxisTitles(graph,fGraphTemplates[i]);
    // save the new graph 
    DEBUGclass("saving graph %s", graph->GetName());
    folder->Add(graph);
  }

  /* delete the list of graphs */
  this->fGraphs.clear();
  this->poolAt = NULL;

  return true;
}

//__________________________________________________________________________________|___________

bool TQGraphMakerAnalysisJob::initializeSampleFolder(TQSampleFolder* sf){
  // initialize this job on a sample folder (taking care of pooling)
  bool pool = false;
  sf->getTagBool(".aj.pool.graphs",pool);
  if(pool && (this->fGraphs.size() == 0)){
    this->initializeGraphs();
    this->poolAt = sf;
  }

  return true;
}

//__________________________________________________________________________________|___________

bool TQGraphMakerAnalysisJob::finalizeSampleFolder(TQSampleFolder* sf){
  // finalize this job on a sample folder (taking care of pooling)
  if(sf == this->poolAt)
    return this->finalizeGraphs();
  return true;
}

//__________________________________________________________________________________|___________

TQGraphMakerAnalysisJob::~TQGraphMakerAnalysisJob() {
  // destructor
  for (unsigned int i = 0; i < fGraphTemplates.size(); i++) {
    delete (fGraphTemplates)[i]; }
}

//__________________________________________________________________________________|___________

TQAnalysisJob* TQGraphMakerAnalysisJob::getClone(){
  // retrieve a clone of this job
  TQGraphMakerAnalysisJob* newJob = new TQGraphMakerAnalysisJob(this);
  return newJob;
}

//__________________________________________________________________________________|___________

void TQGraphMakerAnalysisJob::reset() {
  // Reset this analysis job. This method is called after an analysis job was
  // cloned.

  // call the reset function of the parent class
  TQAnalysisJob::reset();
  // do class-specific stuff
  fGraphs.clear();
}

//__________________________________________________________________________________|___________

int TQGraphMakerAnalysisJob::importJobsFromTextFiles(const TString& files, TQCompiledCut* basecut, const TString& channelFilter, bool verbose){
  // open a list of files (comma-separated), parse all graph definitions inside
  // for each assigment encountered matching the channelfilter (or with no channel),
  // create a graph job, fill it with all appropriate graphs and add it to the basecut
  std::vector <TString> filenames = TQStringUtils::split(files,",");
  return TQGraphMakerAnalysisJob::importJobsFromTextFiles(filenames,basecut,channelFilter,verbose);
}

//__________________________________________________________________________________|___________

int TQGraphMakerAnalysisJob::importJobsFromTextFiles(const TString& files, TQCompiledCut* basecut, TQTaggable* aliases, const TString& channelFilter, bool verbose){
  // open a list of files (comma-separated), parse all graph definitions inside
  // for each assigment encountered matching the channelfilter (or with no channel),
  // create a graph job, fill it with all appropriate graphs and add it to the basecut
  std::vector <TString> filenames = TQStringUtils::split(files,",");
  return TQGraphMakerAnalysisJob::importJobsFromTextFiles(filenames,basecut,aliases,channelFilter,verbose);
}

//__________________________________________________________________________________|___________

int TQGraphMakerAnalysisJob::importJobsFromTextFiles(const std::vector<TString>& filenames, TQCompiledCut* basecut, const TString& channelFilter, bool verbose){
  // open a list of files (std::vector), parse all graph definitions inside
  // for each assigment encountered matching the channelfilter (or with no channel),
  // create a graph job, fill it with all appropriate graphs and add it to the basecut
  return TQGraphMakerAnalysisJob::importJobsFromTextFiles(filenames, basecut, NULL, channelFilter, verbose);
}

//__________________________________________________________________________________|___________

int TQGraphMakerAnalysisJob::importJobsFromTextFiles(const std::vector<TString>& filenames, TQCompiledCut* basecut, TQTaggable* aliases, const TString& channelFilter, bool verbose){
  // open a list of files (std::vector), parse all graph definitions inside
  // for each assigment encountered matching the channelfilter (or with no channel),
  // create a graph job, fill it with all appropriate graphs and add it to the basecut
  if(filenames.size() < 1){
    ERRORfunc("importing no graphs from empty files list!");
    return -1;
  }
  std::map<TString,TString> graphDefinitions;
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
          ERRORfunc("unable to parse graph definition '%s'",line.Data());
          continue;
        }
        TString name(TQStringUtils::trim(line(namestart,nameend-namestart),"\t ,"));
        DEBUGclass("found definition: '%s', assigning as '%s'",line.Data(),name.Data());
        graphDefinitions[name] = line;
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
    TString cuts,graphs;
    TQStringUtils::readUpTo(assignment,cuts,":");
    TQStringUtils::readToken(assignment,buffer," :");
    TQStringUtils::readUpTo(assignment,graphs,";");
    TQStringUtils::readToken(assignment,buffer,"; ");
    DEBUGclass("graphs: '%s'",graphs.Data());
    DEBUGclass("cuts: '%s'",cuts.Data());
    if(verbose) VERBOSEfunc("building job for cuts '%s'",cuts.Data());
    DEBUGclass("spare symbols: '%s'",buffer.Data());
    std::vector<TString> vGraphs = TQStringUtils::split(graphs,",");
    if(vGraphs.size() < 1){
      ERRORfunc("no graphs listed in assignment '%s'",assignments[i].Data());
      continue;
    }
    TQGraphMakerAnalysisJob* job = new TQGraphMakerAnalysisJob();
    for(size_t j=0; j<vGraphs.size(); ++j){
      const TString def = graphDefinitions[TQStringUtils::trim(vGraphs[j],"\t ,")];
      if(def.IsNull()){
        ERRORfunc("unable to find graph definition for name '%s', skipping",TQStringUtils::trim(vGraphs[j],"\t ,").Data());
        continue;
      }
      bool ok = job->bookGraph(def,aliases);
      if(ok){
        if(verbose) VERBOSEfunc("\tbooked graph '%s'",def.Data());
      } else {
        retval += 1;
        if(verbose) std::cout << f_ErrMsg.getMessages() << std::endl;
        DEBUGclass("error booking graph for '%s', function says '%s'",def.Data(),f_ErrMsg.getMessages().Data());
      }
    }
    if(verbose) job->printBooking(cuts);
    basecut->addAnalysisJob(job,cuts);
    delete job;
  } 

  DEBUGclass("end of function call, encountered %d error messages",retval);
  return retval;
}
