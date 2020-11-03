#include <QFramework/TQNTupleDumperAnalysisJob.h>
#include <QFramework/TQStringUtils.h>
#include <QFramework/TQUtils.h>
#include <stdexcept>

//#define _DEBUG_
#include <QFramework/TQLibrary.h>

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQNTupleDumperAnalysisJob:
//
// The TQNTupleDumperAnalysisJob is an analysis job that provides an
// easy interface allowing to write out flat, sk(l)immed mini-nTuples
// with a carefully selected set of variables that can further be used
// for MVA or statistics processing.
//
// Once the job has been created, branches (variables) can be booked like this:
// TQNTupleDumperAnalysisJob::bookVariable("int","runNumber","EventInfo.runNumber()")
//
// It is also possible to mass-book nTuple jobs from config files, using the static
// TQNTupleDumperAnalysisJob::importJobsFromTextFiles
// function. The syntax of these files may look like this:
//
//  ntup: int runNumber << float, mjj << Mjj/1000. , float dyjj << DYjj;
//  @Cut_2jet: ntup >> data/ntup/$(SampleID).root:HWWTree_$(channel);
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQNTupleDumperAnalysisJob)

TQMessageStream TQNTupleDumperAnalysisJob::fErrMsg(new std::stringstream());

const std::map<TQNTupleDumperAnalysisJob::VarType,bool> TQNTupleDumperAnalysisJob::isVectorBranchMap { //used to determine if the branch associated with a particular VarType is vector valued
          {TQNTupleDumperAnalysisJob::VarType::UNKNOWN, false},
          {TQNTupleDumperAnalysisJob::VarType::INT, false},
          {TQNTupleDumperAnalysisJob::VarType::FLOAT, false},
          {TQNTupleDumperAnalysisJob::VarType::DOUBLE, false},
          {TQNTupleDumperAnalysisJob::VarType::ULL, false},
          {TQNTupleDumperAnalysisJob::VarType::VECTORINT, true},
          {TQNTupleDumperAnalysisJob::VarType::VECTORFLOAT, true},
          {TQNTupleDumperAnalysisJob::VarType::VECTORDOUBLE, true},
          {TQNTupleDumperAnalysisJob::VarType::VECTORULL, true}
      };


namespace TQNTupleDumperAnalysisJobHelpers {
  class BranchHandle {
  public:
    const TString name;
    TBranch* branch = 0;
    virtual ~BranchHandle(){};
    BranchHandle(const TString& bname) :
      name(bname) {}
    virtual void setValue(double x, int index = 0) = 0;
    virtual double getValue(int index = 0) const = 0;
    virtual int getNValues() const = 0;
    virtual void resetValue() = 0; //no op for scalar case, calls std::vector<T>::clear() for vector case
    virtual TString getValuesAsString() const { return TString(""); }
    const char* getName() const {
      return name.Data();
    }
  };
  
  template<class T>
  class BranchHandleT : public BranchHandle {
  public:
    mutable T element = 0;
    virtual void setValue(double x, int index = 0) final override {
      DEBUGfunc("setting value of branch %s@%p to %g",this->getName(),this->branch,x);
      if (index != 0) {throw std::runtime_error(TString::Format("Caught attempt to set value of scalar branch handle with index != 0 (%d)",index).Data()); return;}
      this->element = (T)x;
    }
    virtual double getValue(int index = 0) const final override {
      return (double)(this->element);
    }
    virtual int getNValues() const final override {return 1;}
    virtual void resetValue() final override {
      this->element = (T)0.;
    }
    virtual TString getValuesAsString() const final override {
      return TString::Format("%.4f",this->getValue());
    }
    BranchHandleT(TTree* tree, const TString& bname) :
      BranchHandle(bname) {
      this->branch = tree->GetBranch(name);
      if(this->branch){
        tree->SetBranchAddress(name,&element);
      } else {
        this->branch = tree->Branch(name, &element);
      }
      DEBUGfunc("setting branch %s@%p to address %p",this->getName(),this->branch,&element);
    }
  };
  
  template<class T>
  class BranchHandleVectorT : public BranchHandle {
  public:
    mutable std::vector<T> element;
    virtual void setValue(double x, int index = 0) final override {
      DEBUGfunc("setting value of branch %s[%d]@%p to %g",this->getName(),index,this->branch,x);
      if (! (index<(int)element.size()) ) element.resize(index+1); //adjust vector size if insufficient
      this->element[index] = (T)x;
    }
    virtual double getValue(int i) const final override {
      if (i<(int)element.size()) return (double)(this->element[i]);
      throw std::runtime_error(TString::Format("Caught attempt to access TQNTupleDumperAnalysisJobHelpers::BranchHandleVectorT::element out of bounds in handle '%s'!",this->getName()).Data());
    }
    virtual int getNValues() const final override {return (int)element.size();}
    virtual void resetValue() final override {
      element.clear();
    }
    virtual TString getValuesAsString() const final override {
      TString retval = "{";
      for (int i=0; i<this->getNValues(); ++i) {
        if (i!=0) retval.Append(", "); //not required for first element
        retval.Append(TString::Format("%.4f",this->getValue(i)));
      }
      retval.Append("}");
      return retval;
    }
    
    BranchHandleVectorT(TTree* tree, const TString& bname) :
      BranchHandle(bname) {
      this->branch = tree->GetBranch(name);
      if(this->branch){
        tree->SetBranchAddress(name,&element);
      } else {
        this->branch = tree->Branch(name, &element);
      }
      DEBUGfunc("setting branch %s@%p to address %p",this->getName(),this->branch,&element);
    }
  };
  
  class TreeHandle {
  public:
    TTree* tree = 0;
    std::vector<TQToken*> tokens;
    std::vector<BranchHandle*> branches;

    ~TreeHandle() noexcept(false){
      for(auto b:branches){
        delete b;
      }
      if(tokens.size() > 0){
        throw std::runtime_error("trying to delete TreeHandle with remaining tree tokens!");
      }
    }
    
    void empty(){
      for(auto branch:branches){
        branch->resetValue();
      }
    }
    void fill(){
      this->tree->Fill();
    }
    void print(){
      for(auto b:branches){
        std::cout << b->getName() << " = " << b->getValuesAsString().Data() << std::endl;
      }
    }
    
    BranchHandle* getBranch(const TString& name,TQNTupleDumperAnalysisJob::VarType type){
      DEBUGfunc("requested branch '%s'",name.Data());
      for(size_t i=0; i<branches.size(); ++i){
        if(TQStringUtils::equal(branches[i]->getName(),name)){
          DEBUGfunc("returning existing branch '%s@%p'",name.Data(),branches[i]);
          return branches[i];
        }
      }
      BranchHandle* b = NULL;
      switch(type){
      case TQNTupleDumperAnalysisJob::INT:
        b = new BranchHandleT<int>(tree,name); break;
      case TQNTupleDumperAnalysisJob::DOUBLE:
        b = new BranchHandleT<double>(tree,name); break;
      case TQNTupleDumperAnalysisJob::FLOAT:
        b = new BranchHandleT<float>(tree,name); break;
      case TQNTupleDumperAnalysisJob::ULL:
        b = new BranchHandleT<unsigned long long>(tree,name); break;
      case TQNTupleDumperAnalysisJob::VECTORINT:
        b = new BranchHandleVectorT<int>(tree,name); break;
      case TQNTupleDumperAnalysisJob::VECTORDOUBLE:
        b = new BranchHandleVectorT<double>(tree,name); break;
      case TQNTupleDumperAnalysisJob::VECTORFLOAT:
        b = new BranchHandleVectorT<float>(tree,name); break;
      case TQNTupleDumperAnalysisJob::VECTORULL:
        b = new BranchHandleVectorT<unsigned long long>(tree,name); break;
      default:
        throw std::runtime_error(TString::Format("in TQNTupleDumperAnalysisJob, encountered unknown variable type for '%s'",name.Data()).Data());
      }
      this->branches.push_back(b);
      DEBUGfunc("returning new branch '%s@%p'",name.Data(),b);
      return b;
    }
    TQToken* getToken(){
      DEBUGfunc("attempting to get token for '%s'",tree->GetName());
      TQToken* tok = new TQToken();
      tok->setContent(this);
      this->tokens.push_back(tok);
      return tok;
    }
    bool returnToken(TQToken* tok){
      DEBUGfunc("attempting to return token for '%s'",tree->GetName());
      bool found = false;
      for(size_t i=0; i<tokens.size(); ++i){
        if(tok == tokens[i]){
          DEBUGfunc("returning token %p",tokens[i]);
          tokens.erase(tokens.begin()+i);
          found = true;
        }
      }
      if(found){
        delete tok;
      } else {
        DEBUGfunc("unable to return token %p",tok);
      }
      return found;
    }
    bool hasTokens(){
      return (tokens.size() > 0);
    }
    void close(){
      TDirectory* d = gDirectory;
      gDirectory = tree->GetDirectory();
      DEBUGfunc("writing tree '%s' with %d entries to directory '%s'",tree->GetName(),tree->GetEntries(),gDirectory->GetName());
      tree->Write("", TObject::kOverwrite); 
      gDirectory = d;
      for(auto b:this->branches){
        delete b;
      }
      this->branches.clear();
    }
    TreeHandle(TTree* t) : tree(t) {};
  };
  
  class FileHandle {
    bool hasBeenOpened = false;
    TString filename = "";
    TFile* file = 0;
    std::map<const TString,TreeHandle*> trees;
  public:
    FileHandle(const TString& name) : filename(name) {};
    TQToken* getTreeToken(const TString& treename){
      if(!file){
        DEBUGfunc("attempting to open file '%s'",filename.Data());
        if(hasBeenOpened){
          file = TFile::Open(filename,"UPDATE");
        } else {
          TQUtils::ensureDirectoryForFile(filename);
          file = TFile::Open(filename,"RECREATE");
        }
      }
      if(!file || !file->IsOpen()){
        DEBUGfunc("invalid file '%s'",filename.Data());
        if(file) delete file;
        return NULL;
      }
      hasBeenOpened = true;
      DEBUGfunc("attempting to find tree '%s' in file '%s'",treename.Data(),filename.Data());
      if(trees.find(treename) == trees.end()){
        TTree* t = dynamic_cast<TTree*>(file->Get(treename));
        if(!t){
          t = new TTree(treename,treename);
          t->SetMaxVirtualSize(1e15);
          t->SetDirectory(file);
        }
        trees.emplace(treename,new TreeHandle(t));
      }
      DEBUGfunc("attempting to get tree token for '%s':'%s'",filename.Data(),treename.Data());
      return trees[treename]->getToken();
    }
    bool returnTreeToken(const TString& treename, TQToken* tok){
      DEBUGfunc("attempting to return tree token for '%s':'%s'",filename.Data(),treename.Data());
      if(trees.find(treename) == trees.end()){
        return false;
      }
      bool found = trees[treename]->returnToken(tok);
      for(auto it:trees){
        if(!it.second->hasTokens()){
          it.second->close();
          delete it.second;
          trees.erase(it.first);
        }
      }
      if(trees.size() == 0){
        file->Close();
        delete file;
        file = NULL;
      }
      return found;
    }
  };
  
  std::map<const TString,FileHandle*> gFiles;
  
}

using namespace TQNTupleDumperAnalysisJobHelpers;


//__________________________________________________________________________________|___________

TQToken* TQNTupleDumperAnalysisJob::getTreeToken(const TString& filename, const TString& treename){
  if(gFiles.find(filename) == gFiles.end()){
    gFiles.emplace(filename,new FileHandle(filename));
  }
  TQToken* tok = gFiles[filename]->getTreeToken(treename);
  return tok;
}

//__________________________________________________________________________________|___________
    
bool TQNTupleDumperAnalysisJob::returnTreeToken(const TString& filename, const TString& treename, TQToken* tok){
  if(gFiles.find(filename) == gFiles.end()){
    return false;
  }
  return gFiles[filename]->returnTreeToken(treename,tok);
}  

//__________________________________________________________________________________|___________

TQNTupleDumperAnalysisJob::TQNTupleDumperAnalysisJob(const TString& name) : 
  TQAnalysisJob(name)
{
  // default constructor
}

//__________________________________________________________________________________|___________

TQNTupleDumperAnalysisJob::VarType TQNTupleDumperAnalysisJob::getVarType (TString typestr){
  // convert a VarType string to the corresponding enum
  typestr.ToLower();
  DEBUGclass("trying to match type '%s'",typestr.Data());
  if(TQStringUtils::equal(typestr,"int")){
    return VarType::INT;
  } 
  if(TQStringUtils::equal(typestr,"float")){
    return VarType::FLOAT;
  }
  if(TQStringUtils::equal(typestr,"double")){
    return VarType::DOUBLE;
  }
  if(TQStringUtils::equal(typestr,"unsigned long long")){
    return VarType::ULL;
  }
  if(TQStringUtils::equal(TQStringUtils::minimize(typestr),"vector<int>")) {
    return VarType::VECTORINT;
  }
  if(TQStringUtils::equal(TQStringUtils::minimize(typestr),"vector<float>")) {
    return VarType::VECTORFLOAT;
  }
  if(TQStringUtils::equal(TQStringUtils::minimize(typestr),"vector<double>")) {
    return VarType::VECTORDOUBLE;
  }
  if(TQStringUtils::equal(TQStringUtils::minimize(typestr),"vector<unsignedlonglong>")) { //minimize removes all whitespaces
    return VarType::VECTORULL;
  }
    
  return VarType::UNKNOWN;
}

//__________________________________________________________________________________|___________

TString TQNTupleDumperAnalysisJob::getTypeString (VarType type){
  // convert a VarType enum to the corresponding string
    switch(type){
    case TQNTupleDumperAnalysisJob::VarType::INT:
      return "int";
    case TQNTupleDumperAnalysisJob::VarType::FLOAT:
      return "float";
    case TQNTupleDumperAnalysisJob::VarType::DOUBLE:
      return "double";
    case TQNTupleDumperAnalysisJob::VarType::ULL:
      return "unsigned long long";
    default:
      return TQStringUtils::makeBoldRed("invalid");
    }
}


//__________________________________________________________________________________|___________

int TQNTupleDumperAnalysisJob::nVariables(){
  // retrieve the number of booked variables
  return this->fTypes.size();
}

//__________________________________________________________________________________|___________

void TQNTupleDumperAnalysisJob::printBranches(){
  // print the branches booked in this job
  std::cout<< TQStringUtils::makeBoldBlue(this->Class()->GetName()) << " " << TQStringUtils::makeBoldWhite(this->GetName()) << std::endl;
  for(size_t i=0; i<this->fTypes.size(); ++i){
    std::cout << TQStringUtils::fixedWidth(this->getTypeString(this->fTypes[i]),10,"l") << " " << 
      TQStringUtils::makeBoldWhite(TQStringUtils::fixedWidth(this->fVarNames[i],20,"l")) << " " << 
      TQStringUtils::fixedWidth(this->fExpressions[i],50,"l") << std::endl;
  }
}


//__________________________________________________________________________________|___________

bool TQNTupleDumperAnalysisJob::bookVariable(const TString& type, const TString& name, const TString& definition){
  // book a variable 
  DEBUGclass("called with arguments '%s','%s','%s'",type.Data(),name.Data(),definition.Data());
  return this->bookVariable(this->getVarType(type),name,definition);
}

//__________________________________________________________________________________|___________

bool TQNTupleDumperAnalysisJob::bookVariable(VarType type, const TString& name, const TString& definition){
  // book a variable 
  DEBUGclass("called with arguments '%d','%s','%s'",type,name.Data(),definition.Data());
  if(type == VarType::UNKNOWN){
    ERRORclass("cannot book variable '%s' of unknown type!",name.Data());
    return false;
  }
  this->fTypes.push_back(type);
  this->fExpressions.push_back(definition);
  this->fVarNames.push_back(name);
  
  return true;
}

//__________________________________________________________________________________|___________

bool TQNTupleDumperAnalysisJob::initializeTree(TQTaggable* tags) {
  // initialize the output tree for this job
  DEBUGclass("entering function");
  this->fActiveFileName = tags->replaceInText(this->fFileName,"~",false);
  this->fActiveTreeName = tags->replaceInText(this->fTreeName.Data(),"~",false);
  if(fActiveFileName.Contains("$") || fActiveTreeName.Contains("$")){
    ERRORclass("unexpanded placeholders encountered in TQNTupleDumperAnalysisJob::initializeTree: file='%s', tree='%s'. available tags:",fActiveFileName.Data(),fActiveTreeName.Data());
    tags->printTags();
    throw std::runtime_error("unexpanded placeholders");
    return false;
  }

  DEBUGclass("initializing output file '%s' for '%s'",fActiveFileName.Data(),fActiveTreeName.Data());

  this->fTreeToken = TQNTupleDumperAnalysisJob::getTreeToken(fActiveFileName,fActiveTreeName);
  if(!this->fTreeToken){
    throw std::runtime_error(TString::Format("unable to open ouptut for '%s':'%s'",fActiveFileName.Data(),fActiveTreeName.Data()).Data());
    return false;
  }
  DEBUGclass("retrieving tree handler");
  this->fTreeHandler = (TQNTupleDumperAnalysisJobHelpers::TreeHandle*)(this->fTreeToken->getContent());

  DEBUGclass("getting branches");
  const size_t size = this->fVarNames.size();
  for(size_t i=0; i<size; ++i){
    this->fBranches.push_back(this->fTreeHandler->getBranch(this->fVarNames[i],this->fTypes[i]));
    DEBUGclass("registered branch '%s' at position %p",this->fVarNames[i].Data(),this->fBranches[i]);
  }
  
  if(this->fWriteWeight){
    this->fWeightBranch = this->fTreeHandler->getBranch(this->fWeightName,VarType::DOUBLE);
    DEBUGclass("registered weight branch '%s' at position %p",this->fWeightName.Data(),this->fWeightBranch);
  }
  DEBUGclass("done");
  return true;
}

//__________________________________________________________________________________|___________

void TQNTupleDumperAnalysisJob::writeWeights(bool write, const TString& name){
  // determine whether to write weights, and what name the weight branch should have
  this->fWriteWeight = write;
  this->fWeightName = name;
}

//__________________________________________________________________________________|___________

bool TQNTupleDumperAnalysisJob::finalizeTree() {
  // finalize the output tree and close the file
  DEBUGclass("finalizing...");
  bool ok = TQNTupleDumperAnalysisJob::returnTreeToken(this->fActiveFileName,this->fActiveTreeName,this->fTreeToken);
  this->fActiveFileName.Clear();
  this->fActiveTreeName.Clear();
  this->fBranches.clear();
  this->fTreeToken = NULL;
  this->fTreeHandler = NULL;
  return ok;
}

//__________________________________________________________________________________|___________

bool TQNTupleDumperAnalysisJob::initializeSelf() {
  // initialize this analysis job
  DEBUGclass("initializing observables");
  for(size_t i=0; i<this->fVarNames.size(); ++i){
    TQObservable* obs = TQObservable::getObservable(this->fExpressions[i],this->fSample);
    if (!obs->initialize(this->fSample)) {
      ERRORclass("Failed to initialize observable obtained from expression '%s' in TQNTupleDumperAnalysisJob with filename:treename='%s:%s' for sample '%s'",this->fExpressions[i].Data(), this->fFileName.Data(), this->fTreeName.Data() ,this->fSample->getPath().Data());
      return false;
    }
    this->fObservables.push_back(obs);
  }

  if(!this->fTreeHandler){
    DEBUGclass("initializing tree");
    this->poolAt = this->fSample;
    if(!this->initializeTree(this->fSample)) return false;
  }

  DEBUGclass("done");
  return true;
}

//__________________________________________________________________________________|___________

bool TQNTupleDumperAnalysisJob::finalizeSelf() {
  // finalize this analysis job
  DEBUGclass("finalizing observables");
  for(size_t i=0; i<this->fVarNames.size(); ++i){
    this->fObservables[i]->finalize();
  }
  this->fObservables.clear();

  if(this->poolAt == this->fSample){
    DEBUGclass("finalizing tree");
    if(!this->finalizeTree()) return false;
  }
  this->fExpectSingleEntryPerEvent = false; //reset expectation of single entry in output tree per input event (i.e. per 'execute' call)

  return true;
}

//__________________________________________________________________________________|___________

bool TQNTupleDumperAnalysisJob::initializeSampleFolder(TQSampleFolder* sf){
  // initialize this job on a sample folder (taking care of pooling)
  if (!sf) return false;
  if(!this->poolAt){
    // by default, we pool as soon as the filename is uniquely determined
    TString filename = sf->replaceInText(this->fFileName,"~");
    bool pool = !(filename.Contains("$"));
    sf->getTagBool(".aj.pool.trees",pool);
    if(pool && !this->fTreeHandler){
      DEBUG("starting to pool at sample folder '%s'",sf->GetName());
      this->poolAt = sf;
      this->initializeTree(sf);
    }
  }
  
  return true;
}

//__________________________________________________________________________________|___________

bool TQNTupleDumperAnalysisJob::finalizeSampleFolder(TQSampleFolder* sf){
  // finalize this job on a sample folder (taking care of pooling)
  bool ok = true;
  if(sf == this->poolAt){
    ok = this->finalizeTree();
    this->poolAt = NULL;
  }
  return ok;
}

//__________________________________________________________________________________|___________

int TQNTupleDumperAnalysisJob::getNentriesToCreate() const {
  // returns the number of entries to be created in the output tree for the current 
  // input event. A std::runtime_error is thrown if an incompatible combination of 
  // observables is found. Checks are skipped and 1 is returned if for the current 
  // TQSample the booked ntuple structure can be expected to always yield one output
  // entry (event) per input event 
  if (fExpectSingleEntryPerEvent) return 1; //shortcut if we have already assesed that there are no vector observables used for non-vector branches (in which case we should create one entry in the output tree per entry of the vector observable
  int nEntries = -1;
  #ifdef _DEBUG_
  TQObservable* refObs = nullptr;
  #endif
  for (size_t i=0; i<fTypes.size(); ++i) {
    if (fObservables[i]->getObservableType() != TQObservable::ObservableType::vector) continue; //scalar observables are evaluated as needed
    if (isVectorBranchMap.at(fTypes[i])) continue; //for vector branches we use the full range of the corresponding observable in any case (but still have only one tree entry in the output)
    DEBUGclass("found non-trivial observable / branch type combination in TQNTupleDumperAnalysisJob '%s'",this->GetName());
    if (nEntries<0) {
      nEntries = fObservables[i]->getNevaluations(); //store the number of evaluations if this is the first vector observable (which is not associated to a vector branch!)
      #ifdef _DEBUG_
      refObs = fObservables[i];
      #endif
    }
    
    else if (fObservables[i]->getNevaluations() != nEntries) { //check for inconsistencies for which there is no obvious way out -> throw! 
      #ifdef _DEBUG_
      throw std::runtime_error(TString::Format("Found inconsistent number of evaluations between different vector observables in TQNTupleDumperAnalysisJob '%s': %d vs %d evaluations (from observables with expressions '%s' vs '%s').",
              this->GetName(),
              refObs ? refObs->getNevaluations() : std::numeric_limits<int>::quiet_NaN(),
              fObservables[i] ? fObservables[i]->getNevaluations() : std::numeric_limits<int>::quiet_NaN(),
              refObs ? refObs->getExpression().Data() : "<observable n/a>",
              fObservables[i] ? fObservables[i]->getExpression().Data() : "<observable n/a>"
              ).Data());
      #else//simplified error message
      throw std::runtime_error(TString::Format("Found inconsistent number of evaluations between different vector observables in TQNTupleDumperAnalysisJob '%s'",this->GetName()).Data());
      #endif
      return -1;
    }
  }
  
  if (nEntries<0) { //if no Observable+branchType combination might imply multiple tree entries per input event we can skipp these tests for the rest of the sample
    this->fExpectSingleEntryPerEvent = true;
    return 1;
  }
  //we have a non-trivial setup so we return the number of output entries to create for the current input event
  return nEntries;
}


//__________________________________________________________________________________|___________

bool TQNTupleDumperAnalysisJob::execute(double weight) {
  // execute this analysis job on one event
  // fill the output tree with the event data
  if(!this->fTreeHandler){
    DEBUGclass("invalid treehandler!");
    return false;
  }
  
  int nToCreate = this->getNentriesToCreate();
  if (nToCreate<0) {
    ERRORclass("Failed to obtain valid number of entries to create in TQNTupleDumperAnalysisJob '%s'",this->GetName());
    return false;
  }
  
  for (int entry = 0; entry<nToCreate; ++entry) {
    this->fTreeHandler->empty(); //initialize all branch values to 0 (for vector branches length 1, vec[0]=0.) 

    for(size_t i=0; i<this->fBranches.size(); ++i){
      //differetiate two cases: 'vector observable with scalar branch' or 'anything else'
      if (this->fObservables[i]->getObservableType() == TQObservable::ObservableType::vector && !isVectorBranchMap.at(fTypes[i])) {
        TRY(
          fBranches[i]->setValue(this->fObservables[i]->getValueAt(entry),0);
          ,TString::Format("Failed to set value for branch '%s' using observable '%s'",fBranches[i]->getName(),this->fObservables[i]->GetName())
          )
      } else {
      //'anything else'
        for (int vecIndex = this->fObservables[i]->getNevaluations()-1; vecIndex>=0; --vecIndex) { //iterate backwards to avoid resizing the internal std::vector<T> each time
          TRY(
          fBranches[i]->setValue(this->fObservables[i]->getValueAt(vecIndex),vecIndex);
          ,TString::Format("Failed to set value for branch '%s' using observable '%s'",fBranches[i]->getName(),this->fObservables[i]->GetName())
          )
        }
      }
    }
    if(this->fWriteWeight){
      fWeightBranch->setValue(weight*fSample->getNormalisation());
      //    fWeightBranch->setValue(fObservables[0]->getCurrentEntry());
    }
    DEBUGclass("filling event %d from instance %s at cut %s",fObservables[0]->getCurrentEntry() ,this->GetName(),this->fCut->GetName());
    #ifdef _DEBUG_
    this->fTreeHandler->print();
    #endif
    this->fTreeHandler->fill();
  }
    
  
  return true;
}

//__________________________________________________________________________________|___________

TObjArray * TQNTupleDumperAnalysisJob::getBranchNames() {
  // retrieve the list of branch names used by this job
  TObjArray * bNames = new TObjArray();

  for (size_t i = 0; i < this->fObservables.size(); ++i) {
    TQObservable* obs = this->fObservables[i];
    if(obs){
      TCollection* c = obs->getBranchNames();
      if(c){
        if(c->GetEntries() > 0) bNames -> AddAll(c);
        delete c;
      }
    }
  }
 
  return bNames;
}

//__________________________________________________________________________________|___________

TQNTupleDumperAnalysisJob::TQNTupleDumperAnalysisJob(TQNTupleDumperAnalysisJob* other) :
  TQAnalysisJob(other ? other->GetName() : "TQNTupleDumperAnalysisJob"),
  fTreeName(other ? other->fTreeName : "invalid"),
  fFileName(other ? other->fFileName : "invalid.root"),
  fWriteWeight(other ? other->fWriteWeight : true),
  fWeightName(other ? other->fWeightName : "weight")
{
  // copy constructor
  if(other){
    for(int i=0; i<other->nVariables(); ++i){
      this->bookVariable(other->fTypes[i], other->fVarNames[i], other->fExpressions[i]);
    }
  }
}

//__________________________________________________________________________________|___________

TQAnalysisJob * TQNTupleDumperAnalysisJob::getClone() {
  // cloning function, internally calls copy constructor
  return new TQNTupleDumperAnalysisJob(this);
}

//__________________________________________________________________________________|___________

int TQNTupleDumperAnalysisJob::importJobsFromTextFiles(const std::vector<TString>& filenames, TQCut* basecut, TQTaggable* aliases, const TString& channelFilter, bool verbose){
  // open a list of files (std::vector), parse all nTuple definitions inside
  // for each assigment encountered matching the channelfilter (or with no channel),
  // create an nTuple job for each nTuple and add it to the basecut
  if(filenames.size() < 1){
    ERRORfunc("importing no nTuples from empty files list!");
    return -1;
  }
  std::map<TString,std::vector<TString> > nTupleDefinitions;
  std::vector<TString> assignments;
  TString buffer;
  for(size_t i=0; i<filenames.size(); i++){
    std::vector<TString>* lines = TQStringUtils::readFileLines(filenames[i]);
    if(!lines){
      if(verbose) ERRORfunc("unable to open file '%s'",filenames[i].Data());
      continue;
    }
    for(size_t j=0; j<lines->size(); j++){
      TString line(lines->at(j));
      TQStringUtils::readBlanks(line);
      if(line.IsNull()) continue;
      if(!line.BeginsWith("@")){
        TString name, def;
        if(!TQStringUtils::readUpTo(line,name,":")){
          if(verbose) ERRORfunc("unable to parse nTuple definition '%s'",line.Data());
          continue;
        }
        TQStringUtils::removeLeading(line,": ");
        TQStringUtils::readUpTo(line,def,";");
        DEBUGclass("found definition: '%s', assigning as '%s'",def.Data(),name.Data());
        nTupleDefinitions[TQStringUtils::trim(name)] = TQStringUtils::tokenizeVector(def,",",true,"{}()[]","\"\"''"); //this should be a bit safer as it respects more parentheses
        //nTupleDefinitions[TQStringUtils::trim(name)] = TQStringUtils::split(def,",","{","}");
      } else if(TQStringUtils::removeLeading(line,"@") == 1){ 
        DEBUGclass("found assignment: '%s'",line.Data());
        assignments.push_back(line);
      } else {
        if(verbose) WARNfunc("encountered unknown token: '%s'",line.Data());
      }
    }
    delete lines;
  }

  int retval = 0;
  for(size_t i=0; i<assignments.size(); i++){
    TString assignment = assignments[i];
    DEBUGclass("looking at assignment '%s'",assignment.Data());
    TString channel;
    if(TQStringUtils::readBlock(assignment,channel) && !channel.IsNull() && !TQStringUtils::matches(channel,channelFilter)) continue;
    TString cuts,nTuples;
    TQStringUtils::readUpTo(assignment,cuts,":");
    TQStringUtils::readToken(assignment,buffer," :");
    TQStringUtils::readUpTo(assignment,nTuples,";");
    TQStringUtils::readToken(assignment,buffer,"; ");
    DEBUGclass("nTuples: '%s'",nTuples.Data());
    DEBUGclass("cuts: '%s'",cuts.Data());
    DEBUGclass("spare symbols: '%s'",buffer.Data());
    std::vector<TString> vNtups = TQStringUtils::split(nTuples,",");
    if(vNtups.size() < 1){
      if(verbose) ERRORfunc("no nTuples listed in assignment '%s'",assignments[i].Data());
      continue;
    }
    for(size_t j=0; j<vNtups.size(); j++){
      TString ntupbooking(aliases ? aliases->replaceInText(vNtups[j]) : vNtups[j]);
      TString ntupname,filename,treename;
      TQStringUtils::removeLeading(ntupbooking," ");
      TQStringUtils::readUpTo(ntupbooking,ntupname," >");
      TQStringUtils::readToken(ntupbooking,buffer," >");
      TQStringUtils::readUpTo(ntupbooking,filename," :");
      TQStringUtils::readToken(ntupbooking,buffer," :");
      TQStringUtils::readUpTo(ntupbooking,treename," ;");
      DEBUGclass("ntup: '%s'",ntupname.Data());
      DEBUGclass("filname: '%s'",filename.Data());
      DEBUGclass("treename: '%s'",treename.Data());
      std::vector<TString> def = nTupleDefinitions[ntupname];
      if(def.empty()){
        if(verbose) ERRORfunc("unable to find nTuple definition for name '%s', skipping",vNtups[j].Data());
        continue;
      }
      TQNTupleDumperAnalysisJob* job = new TQNTupleDumperAnalysisJob(ntupname);
      job->setTreeName(treename);
      job->setFileName(filename);
      for(size_t i=0; i<def.size(); i++){
        TString branch(aliases ? aliases->replaceInText(def[i]) : def[i]);
        const TString bakbranch(branch);
        DEBUGclass("parsing definition '%s'",branch.Data());
        TQStringUtils::removeLeading(branch," \t");
        TString type_name;
        TQStringUtils::readUpToText(branch,type_name,"<<");
        TQStringUtils::removeLeading(type_name," \t");
        TQStringUtils::removeTrailing(type_name," \t");
        //alternative code ahead in case we need to make the parsing more robust/flexible (untested)
        /*
        //ok, here begins the ugly part: there can be whitespaces in the variable type (e.g. 'unsigned long long') but also in the observable expression, e.g. 'TH2Map:someFile.root:someHist(subObs1, subObs2)'. Hence, we can't easily tell where the type definition ends and where the observable expression starts. => check at each whitespace if substring of type_name up to that point is a valid type.
        TString typeStr = "";
        TQNTupleDumperAnalysisJob::VarType type = TQNTupleDumperAnalysisJob::VarType::UNKNOWN ;
        while( (type == TQNTupleDumperAnalysisJob::VarType::UNKNOWN) && (TQStringUtils::readUpTo(type_name,typeStr," \t") > 0 || TQStringUtils::readToken(type_name,typeStr," \t") > 0) ) { //readUpTo and readToken > 0 implicitly checks for length of type_name being > 0, i.e., protects against infinite loops
          type = getVarType(typeStr); //check if a valid type was read thus far
        }
        TString name = TQStringUtils::trim(type_name);
        */
        int sep = TQStringUtils::findLastOf(type_name,"\t ");
        TString typeStr = TQStringUtils::trim(type_name(0,sep));
	TString name = type_name(sep+1,type_name.Length()-sep-1); 
	DEBUGclass("type of the branch '%s'",typeStr.Data());
	DEBUGclass("name of the branch '%s'",name.Data());

        TQStringUtils::removeLeading(branch," <\t");
        if( !typeStr.IsNull() && !name.IsNull() && !branch.IsNull()){
          DEBUGclass("booking variable '%s' (%s) with expression '%s'",name.Data(),typeStr.Data(),branch.Data());
          job->bookVariable(typeStr,name,branch);
        } else {
          TQNTupleDumperAnalysisJob::setErrorMessage(TString::Format("unable to parse variable definition: '%s', expected syntax is 'type name << expression'. The following elements have been extracted from your definition: type = '%s', name = '%s', expression = '%s'",bakbranch.Data(), typeStr.Data(), name.Data(), branch.Data()  ));
        }
      }
      if(job->nVariables() < 1){
        DEBUGclass("error booking nTuple for '%s', function says '%s'",fErrMsg.getMessages().Data());
      } else {
        if(verbose) job->printBranches();
        basecut->addAnalysisJob(job,cuts);
        retval += 1;
      }
      delete job;
    }
  } 

  DEBUGclass("end of function call, found %d ntuple definitions",retval);
  return retval;
}

//__________________________________________________________________________________|___________

TString TQNTupleDumperAnalysisJob::getErrorMessage() {
  // Return the latest error message
  return fErrMsg.getMessages();
}

//__________________________________________________________________________________|___________

void TQNTupleDumperAnalysisJob::setErrorMessage(const TString& message) {
  // update the error message 
  fErrMsg.sendClassFunctionMessage(TQMessageStream::INFO,TQNTupleDumperAnalysisJob::Class(),"<anonymous>",message);
}

//__________________________________________________________________________________|___________

TString TQNTupleDumperAnalysisJob::getTreeName() const {
  // get the tree name
  return this->fTreeName;
}
//__________________________________________________________________________________|___________

void TQNTupleDumperAnalysisJob::setTreeName (const TString& treename){
  // set the tree name
  this->fTreeName=treename;
}

//__________________________________________________________________________________|___________

TString TQNTupleDumperAnalysisJob::getFileName() const {
  // get the file name
  return this->fFileName;
}

//__________________________________________________________________________________|___________

void TQNTupleDumperAnalysisJob::setFileName (const TString& filename){
  // set the file name
  this->fFileName = filename;
}


//__________________________________________________________________________________|___________

