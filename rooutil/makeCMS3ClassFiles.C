// -*- C++ -*-
// Original author: Puneeth Kalavase (UCSB)
// Updates: Frank Golf (UCSD)
// Updates: Ryan Kelley (UCSD)
// Updates: Alex George (UCSB)
/**********************
ROOT macro to make CMS3.h and ScanChain.C files for basic analysis
of CMS3 ntuples. Usage:

root [0] .L makeCMS3ClassFiles.C++
root [1] makeCMS3ClassFiles(filename, treename, classname, namespace, objname, paranoia = 0)

filename = location+name of ntuple file
paranoia = boolean. If true, will add checks for branches that have nans
branchfilename = See http://www.t2.ucsd.edu/tastwiki/bin/view/CMS/SkimNtuples
classname = you can change the default name of the class "CMS3" to whatever you want
namespace = you can change the default namepace of "tas" to whatever you want
ojbname = you can change the default classname object of "cms2" to whatever you want
BranchNamesFile are hardcoded below!
 --BranchNamesFile is a const string&: see See http://www.t2.ucsd.edu/tastwiki/bin/view/CMS/SkimNtuples
***********************/

const string& branchNamesFile_ = "";

#include "TBranch.h"
#include "TString.h"
#include "TFile.h"
#include "TTree.h"
#include "TSeqCollection.h"
#include <iostream>
#include <fstream>
#include <set>
#include <algorithm>
#include <vector>
#include "Math/LorentzVector.h"
#include <sys/stat.h>

using namespace std;

ofstream headerf;
ofstream codef;
ofstream implf;
ofstream branchfile;

void makeHeaderFile(TFile *f, const string& treeName, bool paranoid, const string& Classname, const string& nameSpace, const string& objName);
void makeCCFile(TFile* f, const string& Classname, const string& nameSpace, const string& objName, const string& branchNamesFile, const string& treeName, bool paranoid);
void makeSrcFile(const string& Classname, const string& nameSpace, const string& objName, const string& branchNamesFile, const string& treeName);
void makeBranchFile(string branchNamesFile, string treeName);
void makeDriverFile(string fname, string treeName);


//-------------------------------------------------------------------------------------------------
void makeCMS3ClassFiles(const std::string& fname, const std::string& treeName="Events", const std::string& className="CMS3",
                        const std::string& nameSpace="tas", const std::string& objName="cms3", const bool paranoid = false) {

  const string& branchNamesFile = branchNamesFile_;

  TFile *f = TFile::Open( fname.c_str() );

  if (f==NULL) {
    cout << "File does not exist. Exiting program" << endl;
    return;
  }

  if (f->IsZombie()) {
    cout << "File is not a valid root file, or root file is corruped" << endl;
    cout << "Exiting..." << endl;
    return;
  }

  bool tree_exists = f->GetListOfKeys()->Contains(Form("%s", treeName.c_str()));
  if (!tree_exists){
    cout << "No tree with name " << treeName.c_str() << " in file " << f->GetName() << endl;
    cout << "Exiting..." << endl;
    return;
  }

  //check if the branchNamesFile exists
  if (branchNamesFile != "") {
    struct stat results;
    int intStat = stat(branchNamesFile.c_str(), &results);
    if (intStat != 0) {
      cout << "Cannot open " << branchNamesFile << endl;
      cout << "Please make sure that the file exists" << endl;
      return;
    }
  }

  headerf.open((className+".h").c_str());
  implf.open((className+".cc").c_str());
//	  codef.open("ScanChain.C");

  makeHeaderFile(f, treeName, paranoid, className, nameSpace, objName);
  makeCCFile(f, className, nameSpace, branchNamesFile, treeName, objName, paranoid);
//	  makeSrcFile(className, nameSpace, branchNamesFile, treeName, objName);
  if (branchNamesFile!="")
    makeBranchFile(branchNamesFile, treeName);

  implf.close();
  headerf.close();
  codef.close();

//	  makeDriverFile(fname, treeName);

  f->Close();
}


//-------------------------------------------------------------------------------------------------
void makeHeaderFile(TFile *f, const string& treeName, bool paranoid, const string& Classname, const string& nameSpace, const string& objName)
{
  headerf << "// -*- C++ -*-" << endl;
  headerf << "// This is a header file generated with the command:" << endl;
  headerf << "// makeCMS3ClassFiles(\"" << f->GetName() << "\", \"" << treeName << "\", \"" << Classname << "\", \"" << nameSpace << "\", \"" << objName << "\")" << endl << endl;
  headerf << "#ifndef " << Classname << "_H" << endl;
  headerf << "#define " << Classname << "_H" << endl << endl;
  headerf << "#include \"Math/LorentzVector.h\"" << endl;
  headerf << "#include \"Math/Point3D.h\"" << endl;
  headerf << "#include \"TMath.h\"" << endl;
  headerf << "#include \"TBranch.h\"" << endl;
  headerf << "#include \"TTree.h\"" << endl;
  headerf << "#include \"TH1F.h\""  << endl;
  headerf << "#include \"TFile.h\"" << endl;
  headerf << "#include \"TBits.h\"" << endl;
  headerf << "#include <vector> " << endl;
  headerf << "#include <unistd.h> " << endl;
  headerf << "typedef ROOT::Math::LorentzVector< ROOT::Math::PxPyPzE4D<float> > LorentzVector;" << endl << endl;
  if (paranoid)
    headerf << "#define PARANOIA" << endl << endl;
  headerf << "using namespace std; " << endl;
  headerf << "class " << Classname << " {" << endl;
  headerf << " private: " << endl;
  headerf << " protected: " << endl;
  headerf << "  unsigned int index;" << endl;
  // TTree *ev = (TTree*)f->Get("Events");
  TList* list_of_keys = f->GetListOfKeys();
  std::string tree_name = "";
  if (treeName.empty()) {
    unsigned int ntrees = 0;
    for (unsigned int idx = 0; idx < (unsigned int)list_of_keys->GetSize(); idx++) {
      const char* obj_name = list_of_keys->At(idx)->GetName();
      TObject* obj = f->Get(obj_name);
      if (obj->InheritsFrom("TTree")) {
        ++ntrees;
        tree_name = obj_name;
      }
    }
    if (ntrees == 0) {
      std::cout << "Did not find a tree. Exiting." << std::endl;
      return;
    }
    if (ntrees > 1) {
      std::cout << "Found more than one tree.  Please specify a tree to use." << std::endl;
      return;
    }
  }
  else
    tree_name = treeName;

  TTree *ev = (TTree*)f->Get(tree_name.c_str());

  TSeqCollection *fullarray = ev->GetListOfAliases();
  bool have_aliases = true;
  if (!fullarray) {
    have_aliases = false;
    fullarray = ev->GetListOfBranches();
  }

  // if (have_aliases && fullarray->GetSize() != ev->GetListOfBranches()->GetSize()) {
  //     std::cout << "Tree has " << fullarray->GetSize() << " aliases and " << ev->GetListOfBranches()->GetSize() << " branches. Exiting." << std::endl;
  //     return;
  // }

  TList *aliasarray = new TList();
  for (Int_t i = 0; i < fullarray->GetSize(); ++i) {

    TBranch *branch = 0;
    if (fullarray->At(i) == 0) continue;
    TString aliasname(fullarray->At(i)->GetName());
    if (have_aliases){
      branch = ev->GetBranch(ev->GetAlias(aliasname.Data()));
      aliasname = fullarray->At(i)->GetName();
    }
    else
      branch = (TBranch*)fullarray->At(i);

    if (branch == 0) continue;

    TString branchname(branch->GetName());
    TString branchtitle(branch->GetTitle());
    TString branchclass(branch->GetClassName());
    std::cout << branchname << std::endl;
    std::cout << branchtitle << std::endl;
    std::cout << branchclass << std::endl;
    if (!branchname.BeginsWith("int") &&
        !branchname.BeginsWith("uint") &&
        !branchname.BeginsWith("ull") &&
        !branchname.BeginsWith("ullong") &&
        !branchname.BeginsWith("bool") &&
        !branchname.BeginsWith("float") &&
        !branchname.BeginsWith("double") &&
        !branchname.BeginsWith("uchars") &&
        !branchtitle.EndsWith("/F") &&
        !branchtitle.EndsWith("/I") &&
        !branchtitle.EndsWith("/i") &&
        !branchtitle.EndsWith("/l") &&
        !branchtitle.EndsWith("/O") &&
        !branchtitle.EndsWith("/D") &&
        !branchtitle.BeginsWith("TString") &&
        !branchtitle.BeginsWith("TBits") &&
        !branchclass.Contains("LorentzVector") &&
        !branchclass.Contains("int") &&
        !branchclass.Contains("uint") &&
        !branchclass.Contains("bool") &&
        !branchclass.Contains("float") &&
        !branchclass.Contains("double") &&
        !branchclass.Contains("string") &&
        !branchclass.Contains("TString") &&
        !branchclass.Contains("long long"))
        {
      continue;
        }

    // if (branchclass.Contains("TString"))
    // {
    //     std::cout << "Adding branch " << branchtitle.Data() << " to list." << std::endl;
    //     std::cout.flush();
    // }

    aliasarray->Add(fullarray->At(i));
  }

  for (Int_t i = 0; i < aliasarray->GetSize(); ++i) {

    //Class name is blank for a int of float
    TString aliasname(aliasarray->At(i)->GetName());
    // TBranch *branch = ev->GetBranch(ev->GetAlias(aliasname.Data()));
    TBranch *branch = 0;
    if (have_aliases)
      branch = ev->GetBranch(ev->GetAlias(aliasname.Data()));
    else
      branch = (TBranch*)aliasarray->At(i);

    TString classname = branch->GetClassName();
    TString title     = branch->GetTitle();
    if (classname.Contains("vector")) {
      if(classname.Contains("edm::Wrapper<") ) {
        classname = classname(0,classname.Length()-2);
        classname.ReplaceAll("edm::Wrapper<","");
        headerf << "  " << classname << " " << aliasname << "_;" << endl;
      }
      //else if (classname.Contains("TString")) {
      //    headerf << "  " << classname << " " << aliasname << "_;" << endl;
      //}
      else {
        headerf << "  " << classname << " *" << aliasname << "_;" << endl;
      }
    } else {
      if(classname != "" ) { //LorentzVector
        if(classname.Contains("edm::Wrapper<") ) {
          classname = classname(0,classname.Length()-1);
          classname.ReplaceAll("edm::Wrapper<","");
          headerf << "  " << setw(8) << left << classname << " " << aliasname << "_;" << endl;
        }
        //else if (classname.Contains("TString")) {
        //    headerf << "  " << classname << " " << aliasname << "_;" << endl;
        //}
        else {
          headerf << "  " << classname << " *" << aliasname << "_;" << endl;
        }
      } else {
        if(title.EndsWith("/i"))
          headerf << "  unsigned int " << aliasname << "_;" << endl;
        if(title.EndsWith("/l"))
          headerf << "  unsigned long long " << aliasname << "_;" << endl;
        if(title.EndsWith("/F"))
          headerf << "  float    " << aliasname << "_;" << endl;
        if(title.EndsWith("/I"))
          headerf << "  int      " << aliasname << "_;" << endl;
        if(title.EndsWith("/O"))
          headerf << "  bool     " << aliasname << "_;" << endl;
        if(title.EndsWith("/D"))
          headerf << "  double   " << aliasname << "_;" << endl;
      }
    }
    headerf << "  TBranch *" << Form("%s_branch",aliasname.Data()) << ";" << endl;
    headerf << "  bool     " << Form("%s_isLoaded",aliasname.Data()) << ";" << endl;
  }
  headerf << "public: " << endl;
  headerf << "void Init(TTree *tree);" << endl;

  // GetEntry
  headerf << "void GetEntry(unsigned int idx); " << endl;

  // LoadAllBranches
  headerf << "void LoadAllBranches(); " << endl;

  // accessor functions
  for (Int_t i = 0; i< aliasarray->GetSize(); i++) {
    TString aliasname(aliasarray->At(i)->GetName());
    // TBranch *branch = ev->GetBranch(ev->GetAlias(aliasname.Data()));
    TBranch *branch = 0;
    if (have_aliases)
      branch = ev->GetBranch(ev->GetAlias(aliasname.Data()));
    else
      branch = (TBranch*)aliasarray->At(i);

    TString classname = branch->GetClassName();
    TString title = branch->GetTitle();
    bool isSkimmedNtuple = false;
    if (!classname.Contains("edm::Wrapper<") &&
        (classname.Contains("vector") || classname.Contains("LorentzVector") ) )
      isSkimmedNtuple = true;
    if (classname.Contains("vector")) {
      if(classname.Contains("edm::Wrapper<") ) {
        classname = classname(0,classname.Length()-2);
        classname.ReplaceAll("edm::Wrapper<","");
      }
      headerf << "  const " << classname << " &" << aliasname << "();" << endl;
    } else {
      if(classname.Contains("edm::Wrapper<")) {
        classname = classname(0,classname.Length()-1);
        classname.ReplaceAll("edm::Wrapper<","");
      }
      if(classname != "" ) {
        headerf << "  const " << classname << " &" << aliasname << "();" << endl;
      } else {
        if(title.EndsWith("/i"))
          headerf << "  const unsigned int &" << aliasname << "();" << endl;
        if(title.EndsWith("/l"))
          headerf << "  const unsigned long long &" << aliasname << "();" << endl;
        if(title.EndsWith("/F"))
          headerf << "  const float &" << aliasname << "();" << endl;
        if(title.EndsWith("/I"))
          headerf << "  const int &" << aliasname << "();" << endl;
        if(title.EndsWith("/O"))
          headerf << "  const bool &" << aliasname << "();" << endl;
        if(title.EndsWith("/D"))
          headerf << "  const double &" << aliasname << "();" << endl;
      }
    }
  } // end of accessor header

  bool haveHLTInfo = false;
  bool haveL1Info  = false;
  bool haveHLT8E29Info = false;
  bool haveTauIDInfo = false;
  bool havebtagInfo = false;
  for (int i = 0; i < aliasarray->GetSize(); i++) {
    TString aliasname(aliasarray->At(i)->GetName());
    if(aliasname=="hlt_trigNames")
      haveHLTInfo = true;
    if(aliasname=="l1_trigNames")
      haveL1Info = true;
    if(aliasname=="hlt8e29_trigNames")
      haveHLT8E29Info = true;
    if(aliasname=="taus_pf_IDnames")
      haveTauIDInfo = true;
    if(aliasname=="pfjets_bDiscriminatorNames")
      havebtagInfo = true;
  }

  if(haveHLTInfo) {
    //functions to return whether or not trigger fired - HLT
    headerf << "  " << "bool passHLTTrigger(TString trigName);" << endl;
  }//if(haveHLTInfo)

  if(haveHLT8E29Info) {
    //functions to return whether or not trigger fired - HLT
    headerf << "  " << "bool passHLT8E29Trigger(TString trigName);" << endl;
  }//if(haveHLT8E29Info)


  if(haveL1Info) {
    //functions to return whether or not trigger fired - L1
    headerf << "  " << "bool passL1Trigger(TString trigName);" << endl;
  }//if(haveL1Info)

  if(haveTauIDInfo) {
    //functions to return whether or not trigger fired - HLT
    headerf << "  " << "float passTauID(TString idName, unsigned int tauIndx);" << endl;
  }//if(haveTauIDInfo)

  if(havebtagInfo) {
    //functions to return whether or not trigger fired - HLT
    headerf << "  " << "float getbtagvalue(TString bDiscriminatorName, unsigned int jetIndx);" << endl;
  }//if(haveTauIDInfo)

  headerf << endl;
  headerf << "  static void progress(int nEventsTotal, int nEventsChain);" << endl;

  headerf << "};" << endl << endl;

  headerf << "#ifndef __CINT__" << endl;
  headerf << "extern " << Classname << " " << objName << ";" << endl;
  headerf << "#endif" << endl << endl;

  // Create namespace that can be used to access the extern'd cms2
  // object methods without having to type cms2. everywhere.
  // Does not include cms2.Init and cms2.GetEntry because I think
  // it is healthy to leave those methods as they are
  headerf << "namespace " << nameSpace << " {" << endl;
  for (Int_t i = 0; i< aliasarray->GetSize(); i++) {
    TString aliasname(aliasarray->At(i)->GetName());
    // TBranch *branch = ev->GetBranch(ev->GetAlias(aliasname.Data()));
    TBranch *branch = 0;
    if (have_aliases)
      branch = ev->GetBranch(ev->GetAlias(aliasname.Data()));
    else
      branch = (TBranch*)aliasarray->At(i);

    TString classname = branch->GetClassName();
    TString title = branch->GetTitle();
    if (classname.Contains("vector")) {
      if(classname.Contains("edm::Wrapper") ) {
        classname = classname(0,classname.Length()-2);
        classname.ReplaceAll("edm::Wrapper<","");
      }
      headerf << "  const " << classname << " &" << aliasname << "()";
    } else {
      if(classname.Contains("edm::Wrapper<") ) {
        classname = classname(0,classname.Length()-1);
        classname.ReplaceAll("edm::Wrapper<","");
      }
      if(classname != "" ) {
        headerf << "  const " << classname << " &" << aliasname << "()";
      } else {
        if(title.EndsWith("/i")){
          headerf << "  const unsigned int &" << aliasname << "()";
        }
        if(title.EndsWith("/l")){
          headerf << "  const unsigned long long &" << aliasname << "()";
        }
        if(title.EndsWith("/F")){
          headerf << "  const float &" << aliasname << "()";
        }
        if(title.EndsWith("/I")){
          headerf << "  const int &" << aliasname << "()";
        }
        if(title.EndsWith("/O")){
          headerf << "  const bool &" << aliasname << "()";
        }
        if(title.EndsWith("/D")){
          headerf << "  const double &" << aliasname << "()";
        }
      }
    }
    headerf << ";" << endl;
  }
  if(haveHLTInfo) {
    //functions to return whether or not trigger fired - HLT
    headerf << "  " << "bool passHLTTrigger(TString trigName);" << endl;
  }//if(haveHLTInfo)
  if(haveHLT8E29Info) {
    //functions to return whether or not trigger fired - HLT
    headerf << "  " << "bool passHLT8E29Trigger(TString trigName);" << endl;
  }//if(haveHLT8E29Info)
  if(haveL1Info) {
    //functions to return whether or not trigger fired - L1
    headerf << "  " << "bool passL1Trigger(TString trigName);" << endl;
  }//if(haveL1Info)
  if(haveTauIDInfo) {
    //functions to return whether or not trigger fired - HLT
    headerf << "  " << "float passTauID(TString idName, unsigned int tauIndx);" << endl;
  }//if(haveTauIDInfo)
  if(havebtagInfo) {
    //functions to return whether or not trigger fired - HLT
    headerf << "  " << "float getbtagvalue(TString bDiscriminatorName, unsigned int jetIndx);" << endl;
  }//if(haveTauIDInfo)

  headerf << "}" << endl;
  headerf << "#include \"rooutil.h\"" << endl;
  headerf << "#endif" << endl;

}

//-------------------------------------------------------------------------------------------------
void makeCCFile(TFile *f, const string& Classname, const string& nameSpace, const string& branchNamesFile, const string& treeName, const string& objName, bool paranoid) {

  // TTree *ev = (TTree*)f->Get("Events");

  implf << "#include \"" << Classname+".h" << "\"\n" << Classname << " " << objName << ";" << endl << endl;
  implf << "void " << Classname << "::Init(TTree *tree) {" << endl;

  TList* list_of_keys = f->GetListOfKeys();
  std::string tree_name = "";
  if (treeName.empty()) {
    unsigned int ntrees = 0;
    for (unsigned int idx = 0; idx < (unsigned int)list_of_keys->GetSize(); idx++) {
      const char* obj_name = list_of_keys->At(idx)->GetName();
      TObject* obj = f->Get(obj_name);
      if (obj->InheritsFrom("TTree")) {
        ++ntrees;
        tree_name = obj_name;
      }
    }
    if (ntrees == 0) {
      std::cout << "Did not find a tree. Exiting." << std::endl;
      return;
    }
    if (ntrees > 1) {
      std::cout << "Found more than one tree.  Please specify a tree to use." << std::endl;
      return;
    }
  }
  else
    tree_name = treeName;

  TTree *ev = (TTree*)f->Get(tree_name.c_str());

  TSeqCollection *fullarray = ev->GetListOfAliases();
  bool have_aliases = true;
  if (!fullarray) {
    have_aliases = false;
    fullarray = ev->GetListOfBranches();
  }


  TList *aliasarray = new TList();
  for (Int_t i = 0; i < fullarray->GetSize(); ++i) {
    TBranch *branch = 0;
    if (fullarray->At(i) == 0) continue;
    TString aliasname(fullarray->At(i)->GetName());
    if (have_aliases){
      branch = ev->GetBranch(ev->GetAlias(aliasname.Data()));
      aliasname = fullarray->At(i)->GetName();
    }
    else
      branch = (TBranch*)fullarray->At(i);

    if (branch == 0) continue;

    TString branchname(branch->GetName());
    TString branchtitle(branch->GetTitle());
    TString branchclass(branch->GetClassName());
    if (!branchname.BeginsWith("int") &&
        !branchname.BeginsWith("uint") &&
        !branchname.BeginsWith("ull") &&
        !branchname.BeginsWith("ullong") &&
        !branchname.BeginsWith("bool") &&
        !branchname.BeginsWith("float") &&
        !branchname.BeginsWith("double") &&
        !branchname.BeginsWith("uchars") &&
        !branchtitle.EndsWith("/F") &&
        !branchtitle.EndsWith("/I") &&
        !branchtitle.EndsWith("/i") &&
        !branchtitle.EndsWith("/l") &&
        !branchtitle.EndsWith("/O") &&
        !branchtitle.EndsWith("/D") &&
        !branchtitle.BeginsWith("TString") &&
        !branchtitle.BeginsWith("TBits") &&
        !branchclass.Contains("LorentzVector") &&
        !branchclass.Contains("int") &&
        !branchclass.Contains("uint") &&
        !branchclass.Contains("bool") &&
        !branchclass.Contains("float") &&
        !branchclass.Contains("double") &&
        !branchclass.Contains("string") &&
        !branchclass.Contains("TString") &&
        !branchclass.Contains("long long"))
      continue;

    // if (branchclass.Contains("TString"))
    // {
    //     std::cout << "Adding branch " << branchtitle.Data() << " to list." << std::endl;
    //     std::cout.flush();
    // }

    aliasarray->Add(fullarray->At(i));
  }

  // SetBranchAddresses for LorentzVectors
  // TBits also needs SetMakeClass(0)...
  for (Int_t i = 0; i< aliasarray->GetSize(); i++) {
    TString aliasname(aliasarray->At(i)->GetName());
    // TBranch *branch = ev->GetBranch(ev->GetAlias(aliasname.Data()));
    TBranch *branch = 0;
    if (have_aliases)
      branch = ev->GetBranch(ev->GetAlias(aliasname.Data()));
    else
      branch = (TBranch*)aliasarray->At(i);

    TString classname = branch->GetClassName();
    TString branch_ptr = Form("%s_branch",aliasname.Data());
    if (!classname.Contains("vector<vector")) {
      if (classname.Contains("Lorentz") || classname.Contains("PositionVector") || classname.Contains("TBits")) {
        // implf << "  " << Form("%s_branch",aliasname.Data()) << " = 0;" << endl;
        if (have_aliases) {
          // implf << "  " << "if (tree->GetAlias(\"" << aliasname << "\") != 0) {" << endl;
          implf << "  " << Form("%s_branch",aliasname.Data()) << " = tree->GetBranch(tree->GetAlias(\"" << aliasname << "\"));" << endl;
          //implf << "    " << Form("%s_branch",aliasname.Data()) << "->SetAddress(&" << aliasname << "_);" << endl << "  }" << endl;
          implf << Form("  if (%s) %s->SetAddress(&%s_);", branch_ptr.Data(), branch_ptr.Data(), aliasname.Data()) << endl;
        }
        else {
          // implf << "  " << "if (tree->GetBranch(\"" << aliasname << "\") != 0) {" << endl;
          implf << "  " << Form("%s_branch",aliasname.Data()) << " = tree->GetBranch(\"" << aliasname << "\");" << endl;
          //implf << "    " << Form("%s_branch",aliasname.Data()) << "->SetAddress(&" << aliasname << "_);" << endl << "  }" << endl;
          implf << Form("  if (%s) %s->SetAddress(&%s_);", branch_ptr.Data(), branch_ptr.Data(), aliasname.Data()) << endl;
        }
      }
    }
  }

  // SetBranchAddresses for everything else
  implf << "" << endl;
  implf << "  tree->SetMakeClass(1);" << endl << endl;
  for (Int_t i = 0; i< aliasarray->GetSize(); i++) {
    TString aliasname(aliasarray->At(i)->GetName());
    // TBranch *branch = ev->GetBranch(ev->GetAlias(aliasname.Data()));
    TBranch *branch = 0;
    if (have_aliases)
      branch = ev->GetBranch(ev->GetAlias(aliasname.Data()));
    else
      branch = (TBranch*)aliasarray->At(i);

    TString classname = branch->GetClassName();
    TString branch_ptr = Form("%s_branch",aliasname.Data());
    if (! (classname.Contains("Lorentz") || classname.Contains("PositionVector") || classname.Contains("TBits")) || classname.Contains("vector<vector") ) {
      // implf << "  " << Form("%s_branch",aliasname.Data()) << " = 0;" << endl;
      if (have_aliases) {
        // implf << "  " << "if (tree->GetAlias(\"" << aliasname << "\") != 0) {" << endl;
        implf << "  " << Form("%s_branch",aliasname.Data()) << " = tree->GetBranch(tree->GetAlias(\"" << aliasname << "\"));" << endl;
        //implf << "    " << Form("%s_branch",aliasname.Data()) << "->SetAddress(&" << aliasname << "_);" << endl << "  }" << endl;
        implf << Form("  if (%s) %s->SetAddress(&%s_);", branch_ptr.Data(), branch_ptr.Data(), aliasname.Data()) << endl;
      }
      else {
        // implf << "  " << "if (tree->GetBranch(\"" << aliasname << "\") != 0) {" << endl;
        implf << "  " << Form("%s_branch",aliasname.Data()) << " = tree->GetBranch(\"" << aliasname << "\");" << endl;
        //implf << "    " << Form("%s_branch",aliasname.Data()) << "->SetAddress(&" << aliasname << "_);" << endl << "  }" << endl;
        implf << Form("  if (%s) %s->SetAddress(&%s_);", branch_ptr.Data(), branch_ptr.Data(), aliasname.Data()) << endl;
      }
    }
  }

  implf << "" << endl;
  implf << "  tree->SetMakeClass(0);" << endl;
  implf << "}" << endl << endl;

  // GetEntry
  implf << "void " << Classname << "::GetEntry(unsigned int idx) {" << endl;
  implf << "  // this only marks branches as not loaded, saving a lot of time" << endl;
  implf << "  index = idx;" << endl;
  for (Int_t i = 0; i< aliasarray->GetSize(); i++) {
    TString aliasname(aliasarray->At(i)->GetName());
    implf << "  " << Form("%s_isLoaded",aliasname.Data()) << " = false;" << endl;
  }
  implf << "}" << endl << endl;

  // LoadAllBranches
  implf << "void " << Classname << "::LoadAllBranches() {" << endl;
  implf << "  // load all branches" << endl;
  for (Int_t i = 0; i< aliasarray->GetSize(); i++) {
    TString aliasname(aliasarray->At(i)->GetName());
    implf << "  " << "if (" << aliasname.Data() <<  "_branch != 0) " << Form("%s();",aliasname.Data()) << endl;
  }
  implf << "}" << endl << endl;

  // accessor functions
  for (Int_t i = 0; i< aliasarray->GetSize(); i++) {
    TString aliasname(aliasarray->At(i)->GetName());
    TString funcname = Form("%s::%s",Classname.c_str(),aliasname.Data());
    // TBranch *branch = ev->GetBranch(ev->GetAlias(aliasname.Data()));
    TBranch *branch = 0;
    if (have_aliases)
      branch = ev->GetBranch(ev->GetAlias(aliasname.Data()));
    else
      branch = (TBranch*)aliasarray->At(i);

    TString classname = branch->GetClassName();
    TString title = branch->GetTitle();
    bool isSkimmedNtuple = false;
    if (!classname.Contains("edm::Wrapper<") &&
        (classname.Contains("vector") || classname.Contains("LorentzVector")))
      isSkimmedNtuple = true;
    if (classname.Contains("vector")) {
      if (classname.Contains("edm::Wrapper<") ) {
        classname = classname(0,classname.Length()-2);
        classname.ReplaceAll("edm::Wrapper<","");
      }
      implf << "const " << classname << " &" << funcname << "() {" << endl;
    } else {
      if (classname.Contains("edm::Wrapper<") ) {
        classname = classname(0,classname.Length()-1);
        classname.ReplaceAll("edm::Wrapper<","");
      }
      if (classname != "" ) {
        implf << "const " << classname << " &" << funcname << "() {" << endl;
      } else {
        if (title.EndsWith("/i"))
          implf << "const unsigned int &" << funcname << "() {" << endl;
        if (title.EndsWith("/l"))
          implf << "const unsigned long long &" << funcname << "() {" << endl;
        if (title.EndsWith("/F"))
          implf << "const float &" << funcname << "() {" << endl;
        if (title.EndsWith("/I"))
          implf << "const int &" << funcname << "() {" << endl;
        if (title.EndsWith("/O"))
          implf << "const bool &" << funcname << "() {" << endl;
        if (title.EndsWith("/D"))
          implf << "const double &" << funcname << "() {" << endl;
      }
    }
    aliasname = aliasarray->At(i)->GetName();
    implf << "  " << "if (not " << Form("%s_isLoaded) {",aliasname.Data()) << endl;
    implf << "    " << "if (" << Form("%s_branch",aliasname.Data()) << " != 0) {" << endl;
    implf << "      " << Form("%s_branch",aliasname.Data()) << "->GetEntry(index);" << endl;
    if (paranoid) {
      implf << "      #ifdef PARANOIA" << endl;
      if (classname == "vector<vector<float> >") {
        if (isSkimmedNtuple) {
          implf << "      " << "for (vector<vector<float> >::const_iterator i = "
                << aliasname << "_->begin(); i != "<< aliasname << "_->end(); ++i) {" << endl;
        } else {
          implf << "      " << "for (vector<vector<float> >::const_iterator i = "
                << aliasname << "_.begin(); i != "<< aliasname << "_.end(); ++i) {" << endl;
        }
        implf << "        " << "for (vector<float>::const_iterator j = i->begin(); "
          "j != i->end(); ++j) {" << endl;
        implf << "          " << "if (not isfinite(*j)) {" << endl;
        implf << "            " << "printf(\"branch " << Form("%s_branch",aliasname.Data())
              << " contains a bad float: %f\\n\", *j);" << endl << "            " << "exit(1);"
              << endl;
        implf << "          }\n        }\n      }" << endl;
      } else if (classname == "vector<float>") {
        if (isSkimmedNtuple) {
          implf << "      " << "for (vector<float>::const_iterator i = "
                << aliasname << "_->begin(); i != "<< aliasname << "_->end(); ++i) {" << endl;
        } else {
          implf << "      " << "for (vector<float>::const_iterator i = "
                << aliasname << "_.begin(); i != "<< aliasname << "_.end(); ++i) {" << endl;
        }
        implf << "        " << "if (not isfinite(*i)) {" << endl;
        implf << "          " << "printf(\"branch " << Form("%s_branch",aliasname.Data())
              << " contains a bad float: %f\\n\", *i);" << endl << "          " << "exit(1);"
              << endl;
        implf << "        }\n      }" << endl;
      } else if (classname == "float") {
        implf << "      " << "if (not isfinite(" << aliasname << "_)) {" << endl;
        implf << "        " << "printf(\"branch " << Form("%s_branch",aliasname.Data())
              << " contains a bad float: %f\\n\", " << aliasname << "_);" << endl
              << "        " << "exit(1);"
              << endl;
        implf << "      }" << endl;
      } else if (classname.BeginsWith("vector<vector<ROOT::Math::LorentzVector")) {
        if (isSkimmedNtuple) {
          implf << "      " << "for (" << classname.Data() <<"::const_iterator i = "
                << aliasname << "_->begin(); i != "<< aliasname << "_->end(); ++i) {" << endl;
        } else {
          implf << "      " << "for (" << classname.Data() <<"::const_iterator i = "
                << aliasname << "_.begin(); i != "<< aliasname << "_.end(); ++i) {" << endl;
        }
        // this is a slightly hacky way to get rid of the outer vector< > ...
        std::string str = classname.Data() + 7;
        str[str.length() - 2] = 0;
        implf << "        " << "for (" << str.c_str() << "::const_iterator j = i->begin(); "
          "j != i->end(); ++j) {" << endl;
        implf << "          " << "int e;" << endl;
        implf << "          " << "frexp(j->pt(), &e);" << endl;
        implf << "          " << "if (not isfinite(j->pt()) || e > 30) {" << endl;
        implf << "            " << "printf(\"branch " << Form("%s_branch",aliasname.Data())
              << " contains a bad float: %f\\n\", j->pt());" << endl << "            " << "exit(1);"
              << endl;
        implf << "          }\n        }\n      }" << endl;
      } else if (classname.BeginsWith("vector<ROOT::Math::LorentzVector")) {
        if (isSkimmedNtuple) {
          implf << "      " << "for (" << classname.Data() << "::const_iterator i = "
                << aliasname << "_->begin(); i != "<< aliasname << "_->end(); ++i) {" << endl;
        } else {
          implf << "      " << "for (" << classname.Data() << "::const_iterator i = "
                << aliasname << "_.begin(); i != "<< aliasname << "_.end(); ++i) {" << endl;
        }
        implf << "        " << "int e;" << endl;
        implf << "        " << "frexp(i->pt(), &e);" << endl;
        implf << "        " << "if (not isfinite(i->pt()) || e > 30) {" << endl;
        implf << "          " << "printf(\"branch " << Form("%s_branch",aliasname.Data())
              << " contains a bad float: %f\\n\", i->pt());" << endl << "          " << "exit(1);"
              << endl;
        implf << "        }\n      }" << endl;
      } else if (classname.BeginsWith("ROOT::Math::LorentzVector")) {
        implf << "      " << "int e;" << endl;
        if (isSkimmedNtuple) {
          implf << "      " << "frexp(" << aliasname << "_->pt(), &e);" << endl;
          implf << "      " << "if (not isfinite(" << aliasname << "_->pt()) || e > 30) {" << endl;
          implf << "        " << "printf(\"branch " << Form("%s_branch",aliasname.Data())
                << " contains a bad float: %f\\n\", " << aliasname << "_->pt());" << endl
                << "        " << "exit(1);"
                << endl;
        } else {
          implf << "      " << "frexp(" << aliasname << "_.pt(), &e);" << endl;
          implf << "      " << "if (not isfinite(" << aliasname << "_.pt()) || e > 30) {" << endl;
          implf << "        " << "printf(\"branch " << Form("%s_branch",aliasname.Data())
                << " contains a bad float: %f\\n\", " << aliasname << "_.pt());" << endl
                << "        " << "exit(1);"
                << endl;
        }
        implf << "      }" << endl;
      }
      implf << "      #endif // #ifdef PARANOIA" << endl;
    }
    implf << "    " << "} else {" << endl;
    implf << "      " << "printf(\"branch " << Form("%s_branch",aliasname.Data())
          << " does not exist!\\n\");" << endl;
    implf << "      " << "exit(1);" << endl << "    }" << endl;
    implf << "    " << Form("%s_isLoaded",aliasname.Data()) << " = true;" << endl;
    implf << "  " << "}" << endl;
    if (isSkimmedNtuple) {
      implf << "  " << "return *" << aliasname << "_;" << endl << "}" << endl << endl;
    }
    else if (classname == "TString" || classname == "string") {
      implf << "  " << "return *" << aliasname << "_;" << endl << "}" << endl << endl;
    }
    else {
      implf << "  " << "return " << aliasname << "_;" << endl << "}" << endl << endl;
    }
  }

  bool haveHLTInfo = false;
  bool haveL1Info  = false;
  bool haveHLT8E29Info = false;
  bool haveTauIDInfo = false;
  bool havebtagInfo = false;
  for (int i = 0; i < aliasarray->GetSize(); i++) {
    TString aliasname(aliasarray->At(i)->GetName());
    if (aliasname=="hlt_trigNames")
      haveHLTInfo = true;
    if (aliasname=="l1_trigNames")
      haveL1Info = true;
    if (aliasname=="hlt8e29_trigNames")
      haveHLT8E29Info = true;
    if (aliasname=="taus_pf_IDnames")
      haveTauIDInfo = true;
    if (aliasname=="pfjets_bDiscriminatorNames")
      havebtagInfo = true;
  }

  if (haveHLTInfo) {
    //functions to return whether or not trigger fired - HLT
    implf << "  " << "bool " << Classname << "::passHLTTrigger(TString trigName) {" << endl;
    implf << "    " << "int trigIndx;" << endl;
    implf << "    " << "vector<TString>::const_iterator begin_it = hlt_trigNames().begin();" << endl;
    implf << "    " << "vector<TString>::const_iterator end_it = hlt_trigNames().end();" << endl;
    implf << "    " << "vector<TString>::const_iterator found_it = find(begin_it, end_it, trigName);" << endl;
    implf << "    " << "if (found_it != end_it)" << endl;
    implf << "      " << "trigIndx = found_it - begin_it;" << endl;
    implf << "    " << "else {" << endl;
    implf << "      " << "cout << \"Cannot find Trigger \" << trigName << endl; " << endl;
    implf << "      " << "return 0;" << endl;
    implf << "    "   << "}" << endl << endl;
    implf << "  " << "return hlt_bits().TestBitNumber(trigIndx);" << endl;
    implf << "  " << "}" << endl;
  }//if (haveHLTInfo)

  if (haveHLT8E29Info) {
    //functions to return whether or not trigger fired - HLT
    implf << "  " << "bool " << Classname << "::passHLT8E29Trigger(TString trigName) {" << endl;
    implf << "    " << "int trigIndx;" << endl;
    implf << "    " << "vector<TString>::const_iterator begin_it = hlt8e29_trigNames().begin();" << endl;
    implf << "    " << "vector<TString>::const_iterator end_it = hlt8e29_trigNames().end();" << endl;
    implf << "    " << "vector<TString>::const_iterator found_it = find(begin_it, end_it, trigName);" << endl;
    implf << "    " << "if (found_it != end_it)" << endl;
    implf << "      " << "trigIndx = found_it - begin_it;" << endl;
    implf << "    " << "else {" << endl;
    implf << "      " << "cout << \"Cannot find Trigger \" << trigName << endl; " << endl;
    implf << "      " << "return 0;" << endl;
    implf << "    "   << "}" << endl << endl;
    implf << "  " << "return hlt8e29_bits().TestBitNumber(trigIndx);" << endl;
    implf << "  " << "}" << endl;
  }//if (haveHLT8E29Info)


  if (haveL1Info) {
    //functions to return whether or not trigger fired - L1
    implf << "  " << "bool " << Classname << "::passL1Trigger(TString trigName) {" << endl;
    implf << "    " << "int trigIndx;" << endl;
    implf << "    " << "vector<TString>::const_iterator begin_it = l1_trigNames().begin();" << endl;
    implf << "    " << "vector<TString>::const_iterator end_it = l1_trigNames().end();" << endl;
    implf << "    " << "vector<TString>::const_iterator found_it = find(begin_it, end_it, trigName);" << endl;
    implf << "    " << "if (found_it != end_it)" << endl;
    implf << "      " << "trigIndx = found_it - begin_it;" << endl;
    implf << "    " << "else {" << endl;
    implf << "      " << "cout << \"Cannot find Trigger \" << trigName << endl; " << endl;
    implf << "      " << "return 0;" << endl;
    implf << "    "   << "}" << endl << endl;
    //get the list of branches that hold the L1 bitmasks
    //store in a set 'cause its automatically sorted
    set<TString> s_L1bitmasks;
    for (int j = 0; j < aliasarray->GetSize(); j++) {
      TString aliasname(aliasarray->At(j)->GetName());
      // TBranch *branch = ev->GetBranch(ev->GetAlias(aliasname.Data()));
      TBranch *branch = 0;
      if (have_aliases)
        branch = ev->GetBranch(ev->GetAlias(aliasname.Data()));
      else
        branch = (TBranch*)aliasarray->At(j);

      TString classname = branch->GetClassName();
      if (aliasname.Contains("l1_bits") && classname.Contains("int")) {
        s_L1bitmasks.insert(aliasname);
      }

    }
    int i = 0;
    for (set<TString>::const_iterator s_it = s_L1bitmasks.begin();
         s_it != s_L1bitmasks.end(); s_it++, i++) {

      if (i==0) {
        implf << "    " << "if (trigIndx <= 31) {" << endl;
        implf << "      " << "unsigned int bitmask = 1;" << endl;
        implf << "      " << "bitmask <<= trigIndx;" << endl;
        implf << "      " << "return " << *s_it << "() & bitmask;" << endl;
        implf << "    " << "}" << endl;
        continue;
      }
      implf << "    " << "if (trigIndx >= " << Form("%d && trigIndx <= %d", 32*i, 32*i+31) << ") {" << endl;
      implf << "      " << "unsigned int bitmask = 1;" << endl;
      implf << "      " << "bitmask <<= (trigIndx - " << Form("%d",32*i) << "); " << endl;
      implf << "      " << "return " << *s_it << "() & bitmask;" << endl;
      implf << "    " << "}" << endl;
    }
    implf << "  " << "return 0;" << endl;
    implf << "  " << "}" << endl;
    implf << "" << endl;
  }//if (haveL1Info)

  if (haveTauIDInfo) {
    //functions to return whether or not trigger fired - HLT
    implf << "  " << "float " << Classname << "::passTauID(TString idName, unsigned int tauIndx) {" << endl;
    implf << "    " << "int idIndx;" << endl;
    implf << "    " << "vector<TString>::const_iterator begin_it = taus_pf_IDnames().begin();" << endl;
    implf << "    " << "vector<TString>::const_iterator end_it = taus_pf_IDnames().end();" << endl;
    implf << "    " << "vector<TString>::const_iterator found_it = find(begin_it, end_it, idName);" << endl;
    implf << "    " << "if (found_it != end_it)" << endl;
    implf << "      " << "idIndx = found_it - begin_it;" << endl;
    implf << "    " << "else {" << endl;
    implf << "      " << "cout << \"Cannot find Tau ID \" << idName << endl; " << endl;
    implf << "      " << "return 0;" << endl;
    implf << "    "   << "}" << endl << endl;
    implf << "    "   << "if (tauIndx < taus_pf_IDs().size()) " << endl;
    implf << "      " << "return taus_pf_IDs().at(tauIndx).at(idIndx);" << endl;
    implf << "    "   << "else {" << endl;
    implf << "      " << "cout << \"Cannot find tau # \"<< tauIndx << endl;" << endl;
    implf << "      " << "return 0;" << endl;
    implf << "    "   << "}" << endl;
    implf << "  " << "}" << endl;
  }//if (haveTauIDInfo)

  if (havebtagInfo) {
    //functions to return whether or not trigger fired - HLT
    implf << "  " << "float " << Classname << "::getbtagvalue(TString bDiscriminatorName, unsigned int jetIndx) {" << endl;
    implf << "    " << "size_t bDiscriminatorIndx;" << endl;
    implf << "    " << "vector<TString>::const_iterator begin_it = pfjets_bDiscriminatorNames().begin();" << endl;
    implf << "    " << "vector<TString>::const_iterator end_it = pfjets_bDiscriminatorNames().end();" << endl;
    implf << "    " << "vector<TString>::const_iterator found_it = find(begin_it, end_it, bDiscriminatorName);" << endl;
    implf << "    " << "if (found_it != end_it)" << endl;
    implf << "      " << "bDiscriminatorIndx = found_it - begin_it;" << endl;
    implf << "    " << "else {" << endl;
    implf << "      " << "cout << \"Cannot find b discriminator \" << bDiscriminatorName << endl; " << endl;
    implf << "      " << "return 0;" << endl;
    implf << "    " << "}" << endl << endl;
    implf << "    " << "if (jetIndx < pfjets_bDiscriminators().size()) " << endl;
    implf << "      " << "return pfjets_bDiscriminators().at(jetIndx).at(bDiscriminatorIndx);" << endl;
    implf << "    " << "else {" << endl;
    implf << "      " << "cout << \"Cannot find jet # \"<< jetIndx << endl;" << endl;
    implf << "      " << "return 0;" << endl;
    implf << "    " << "}" << endl;
    implf << "  " << "}" << endl;
  }//if (haveTauIDInfo)

  implf << endl;
  implf << "void " << Classname << "::progress( int nEventsTotal, int nEventsChain ){" << endl;
  implf << "  int period = 1000;" << endl;
  implf << "  if (nEventsTotal%1000 == 0) {" << endl;
  implf << "    // xterm magic from L. Vacavant and A. Cerri" << endl;
  implf << "    if (isatty(1)) {" << endl;
  implf << "      if ((nEventsChain - nEventsTotal) > period) {" << endl;
  implf << "        float frac = (float)nEventsTotal/(nEventsChain*0.01);" << endl;
  implf << "        printf(\"\\015\\033[32m ---> \\033[1m\\033[31m%4.1f%%\"" << endl;
  implf << "             \"\\033[0m\\033[32m <---\\033[0m\\015\", frac);" << endl;
  implf << "        fflush(stdout);" << endl;
  implf << "      }" << endl;
  implf << "      else {" << endl;
  implf << "        printf(\"\\015\\033[32m ---> \\033[1m\\033[31m%4.1f%%\"" << endl;
  implf << "               \"\\033[0m\\033[32m <---\\033[0m\\015\", 100.);" << endl;
  implf << "        cout << endl;" << endl;
  implf << "      }" << endl;
  implf << "    }" << endl;
  implf << "  }" << endl;
  implf << "}" << endl;
  implf << "" << endl;

  // Create namespace that can be used to access the extern'd cms2
  // object methods without having to type cms2. everywhere.
  // Does not include cms2.Init and cms2.GetEntry because I think
  // it is healthy to leave those methods as they are
  implf << "namespace " << nameSpace << " {" << endl << endl;
  for (Int_t i = 0; i< aliasarray->GetSize(); i++) {
    TString aliasname(aliasarray->At(i)->GetName());
    // TBranch *branch = ev->GetBranch(ev->GetAlias(aliasname.Data()));
    TBranch *branch = 0;
    if (have_aliases)
      branch = ev->GetBranch(ev->GetAlias(aliasname.Data()));
    else
      branch = (TBranch*)aliasarray->At(i);

    TString classname = branch->GetClassName();
    TString title = branch->GetTitle();
    if (classname.Contains("vector")) {
      if (classname.Contains("edm::Wrapper") ) {
        classname = classname(0,classname.Length()-2);
        classname.ReplaceAll("edm::Wrapper<","");
      }
      implf << "const " << classname << " &" << aliasname << "()";
    } else {
      if (classname.Contains("edm::Wrapper<") ) {
        classname = classname(0,classname.Length()-1);
        classname.ReplaceAll("edm::Wrapper<","");
      }
      if (classname != "" ) {
        implf << "const " << classname << " &" << aliasname << "()";
      } else {
        if (title.EndsWith("/i")){
          implf << "const unsigned int &" << aliasname << "()";
        }
        if (title.EndsWith("/l")){
          implf << "const unsigned long long &" << aliasname << "()";
        }
        if (title.EndsWith("/F")){
          implf << "const float &" << aliasname << "()";
        }
        if (title.EndsWith("/I")){
          implf << "const int &" << aliasname << "()";
        }
        if (title.EndsWith("/O")){
          implf << "const bool &" << aliasname << "()";
        }
        if (title.EndsWith("/D")){
          implf << "const double &" << aliasname << "()";
        }
      }
    }
    implf << " { return " << objName << "." << aliasname << "(); }" << endl;
  }
  if (haveHLTInfo) {
    //functions to return whether or not trigger fired - HLT
    implf << "bool passHLTTrigger(TString trigName) { return " << objName << ".passHLTTrigger(trigName); }" << endl;
  }//if (haveHLTInfo)
  if (haveHLT8E29Info) {
    //functions to return whether or not trigger fired - HLT
    implf << "bool passHLT8E29Trigger(TString trigName) { return " << objName << ".passHLT8E29Trigger(trigName); }" << endl;
  }//if (haveHLT8E29Info)
  if (haveL1Info) {
    //functions to return whether or not trigger fired - L1
    implf << "bool passL1Trigger(TString trigName) { return " << objName << ".passL1Trigger(trigName); }" << endl;
  }//if (haveL1Info)
  if (haveTauIDInfo) {
    //functions to return whether or not trigger fired - HLT
    implf << "float passTauID(TString idName, unsigned int tauIndx) { return " << objName << ".passTauID(idName, tauIndx); }" << endl;
  }//if (haveTauIDInfo)
  if (havebtagInfo) {
    //functions to return whether or not trigger fired - HLT
    implf << "float getbtagvalue(TString bDiscriminatorName, unsigned int jetIndx) { return " << objName << ".getbtagvalue( bDiscriminatorName, jetIndx); }" << endl;
  }//if (haveTauIDInfo)

  implf << endl << "}" << endl;
  implf << endl << "#include \"rooutil.cc\"" << endl;

}


//-------------------------------------------------------------------------------------------------
void makeSrcFile(const string& Classname, const string& nameSpace, const string& branchNamesFile, const string& treeName, const string& objName) {

  // TTree *ev = (TTree*)f->Get("Events");

  codef << "// -*- C++ -*-" << endl;
  codef << "// Usage:" << endl;
  codef << "// > root -b -q doAll.C" << endl;
  codef << "" << endl;
  codef << "#include <iostream>" << endl;
  codef << "#include <vector>" << endl;
  codef << "" << endl;
  codef << "// ROOT" << endl;
  codef << "#include \"TBenchmark.h\"" << endl;
  codef << "#include \"TChain.h\"" << endl;
  codef << "#include \"TDirectory.h\"" << endl;
  codef << "#include \"TFile.h\"" << endl;
  codef << "#include \"TROOT.h\"" << endl;
  codef << "#include \"TTreeCache.h\"" << endl;
  codef << "" << endl;
  codef << "// " << Classname << endl;
  codef << "#include \"" + Classname+".cc\"" << endl;
  if (branchNamesFile!="")
    codef << "#include \"branches.h\"" << endl;
  codef << "" << endl;
  codef << "using namespace std;" << endl;
  codef << "using namespace " << nameSpace << ";" << endl;
  codef << endl;

  codef << "int ScanChain(TChain* chain, bool fast = true, int nEvents = -1, string skimFilePrefix = \"test\") {" << endl;
  codef << "" << endl;
  codef << "  // Benchmark" << endl;
  codef << "  TBenchmark *bmark = new TBenchmark();" << endl;
  codef << "  bmark->Start(\"benchmark\");" << endl;
  codef << "" << endl;
  codef << "  // Example Histograms" << endl;
  codef << "  TDirectory *rootdir = gDirectory->GetDirectory(\"Rint:\");" << endl;
  codef << "  TH1F *samplehisto = new TH1F(\"samplehisto\", \"Example histogram\", 200,0,200);" << endl;
  codef << "  samplehisto->SetDirectory(rootdir);" << endl;
  codef << "" << endl;
  codef << "  // Loop over events to Analyze" << endl;
  codef << "  unsigned int nEventsTotal = 0;" << endl;
  codef << "  unsigned int nEventsChain = chain->GetEntries();" << endl;
  codef << "  if (nEvents >= 0) nEventsChain = nEvents;" << endl;
  if (branchNamesFile!="")
    codef << "  InitSkimmedTree(skimFilePrefix);" << endl;
  codef << "  TObjArray *listOfFiles = chain->GetListOfFiles();" << endl;
  codef << "  TIter fileIter(listOfFiles);" << endl;
  codef << "  TFile *currentFile = 0;" << endl;
  codef << "" << endl;
  codef << "  // File Loop" << endl;
  codef << "  while ( (currentFile = (TFile*)fileIter.Next()) ) {" << endl;
  codef << "" << endl;
  codef << "    // Get File Content" << endl;
  codef << "    TFile file(currentFile->GetTitle());" << endl;
  codef << "    TTree *tree = (TTree*)file.Get(\"" << treeName << "\");" << endl;
  codef << "    if (fast) TTreeCache::SetLearnEntries(10);" << endl;
  codef << "    if (fast) tree->SetCacheSize(128*1024*1024);" << endl;
  codef << "    " << objName << ".Init(tree);" << endl;
  codef << "" << endl;
  codef << "    // Loop over Events in current file" << endl;
  codef << "    if (nEventsTotal >= nEventsChain) continue;" << endl;
  codef << "    unsigned int nEventsTree = tree->GetEntriesFast();" << endl;
  codef << "    for (unsigned int event = 0; event < nEventsTree; ++event) {" << endl;
  codef << "" << endl;
  codef << "      // Get Event Content" << endl;
  codef << "      if (nEventsTotal >= nEventsChain) continue;" << endl;
  codef << "      if (fast) tree->LoadTree(event);" << endl;
  codef << "      " << objName << ".GetEntry(event);" << endl;
  codef << "      ++nEventsTotal;" << endl;
  codef << "" << endl;
  codef << "      // Progress" << endl;
  codef << "      " << Classname << "::progress( nEventsTotal, nEventsChain );" << endl << endl;
  codef << "      // Analysis Code" << endl << endl;
  codef << "    }" << endl;
  codef << "  " << endl;
  codef << "    // Clean Up" << endl;
  codef << "    delete tree;" << endl;
  codef << "    file.Close();" << endl;
  codef << "  }" << endl;
  codef << "  if (nEventsChain != nEventsTotal) {" << endl;
  codef << "    cout << Form( \"ERROR: number of events from files (\%d) is not equal to total number of events (\%d)\", nEventsChain, nEventsTotal ) << endl;" << endl;
  codef << "  }" << endl;
  if (branchNamesFile!="") {
    codef << "  outFile_->cd();" << endl;
    codef << "  outTree_->Write();" << endl;
    codef << "  outFile_->Close();" << endl;
  }
  codef << "  " << endl;
  codef << "  // Example Histograms" << endl;
  codef << "  samplehisto->Draw();" << endl;
  codef << "  " << endl;
  codef << "  // return" << endl;
  codef << "  bmark->Stop(\"benchmark\");" << endl;
  codef << "  cout << endl;" << endl;
  codef << "  cout << nEventsTotal << \" Events Processed\" << endl;" << endl;
  codef << "  cout << \"------------------------------\" << endl;" << endl;
  codef << "  cout << \"CPU  Time: \" << Form( \"\%.01f\", bmark->GetCpuTime(\"benchmark\")  ) << endl;" << endl;
  codef << "  cout << \"Real Time: \" << Form( \"\%.01f\", bmark->GetRealTime(\"benchmark\") ) << endl;" << endl;
  codef << "  cout << endl;" << endl;
  codef << "  delete bmark;" << endl;
  codef << "  return 0;" << endl;
  codef << "}" << endl;

}


//-------------------------------------------------------------------------------------------------
void makeBranchFile(std::string branchNamesFile, std::string treeName) {

  ifstream branchesF(branchNamesFile.c_str());
  vector<TString> v_datatypes;
  vector<TString> v_varNames;
  while (!branchesF.eof()) {
    string temp;
    getline(branchesF, temp);
    TString line(temp);
    // do not use commented lines
    if (line.BeginsWith("//"))
      continue;
    vector<TString> v_line;
    TIter objIt((TObjArray*)line.Tokenize(" "));
    while (TObject *obj = (TObject*)objIt.Next()) {
      if (obj==NULL || obj==0)
        continue;
      v_line.push_back(obj->GetName());
    }

    if (v_line.size() == 0)
      continue;
    TString varName;
    // loop over v_line until you get to the first element thats not a space
    for (vector<TString>::iterator it = v_line.begin();
         it != v_line.end(); it++) {
      if ( *it != " " ) {
        varName = *it;
      }
    }

    v_varNames.push_back(varName.Strip());  //paranoid....strips trailing spaces
    TString datatype("");
    for (unsigned int i = 0; i < v_line.size()-1; i++) {
      TString temp2 = v_line[i];
      if (temp2.Contains("vector") && !temp2.Contains("std::")) temp2.ReplaceAll("vector", "std::vector");
      if (temp2.Contains("LorentzVector") && !temp2.Contains("ROOT::Math::LorentzVector")) temp2.ReplaceAll("LorentzVector", "ROOT::Math::LorentzVector< ROOT::Math::PxPyPzE4D<float> >");
      temp2.ReplaceAll(">>", "> >");
      temp2.ReplaceAll(">>>", "> > >");
      if (i!=0) datatype = datatype+" " + temp2;
      else datatype = datatype+temp2;
    }
    v_datatypes.push_back(datatype);

  }
  branchfile.open("branches.h");
  branchfile << "#ifndef BRANCHES_H" << endl << "#define BRANCHES_H" << endl;
  branchfile << "#include <vector>" << endl;
  branchfile << "#include \"TFile.h\"" << endl;
  branchfile << "#include \"TTree.h\"" << endl << endl << endl << endl;

  for (unsigned int i = 0; i < v_datatypes.size(); i++) {
    TString temp2(v_varNames.at(i));
    if (v_datatypes.at(i).Contains("vector") || v_datatypes.at(i).Contains("LorentzVector")) {
      branchfile << v_datatypes.at(i) << " *" << temp2.ReplaceAll("_","")+"_;" << endl;
      continue;
    }
    branchfile << v_datatypes.at(i) << " " << temp2.ReplaceAll("_","")+"_;" << endl;
  }


  branchfile << "TFile *outFile_;" << endl;
  branchfile << "TTree *outTree_;" << endl;

  //now declare the branches and set aliases in the InitSkimmedTree function
  branchfile << "void InitSkimmedTree(std::string skimFilePrefix=\"\") {\n\n";
  branchfile << "   if (skimFilePrefix != \"\")" << endl;
  branchfile << "      outFile_ = TFile::Open(string(skimFilePrefix + \"_skimmednTuple.root\").c_str(),\"RECREATE\");\n";
  branchfile << "   else outFile_ = TFile::Open(\"skimmednTuple.root\", \"RECREATE\");\n";
  branchfile << "   outFile_->cd();" << endl;
  branchfile << "   outTree_ = new TTree(\"" << treeName << "\", \"\");\n\n";
  branchfile << "   //book the branches\n";
  for (unsigned int i = 0; i < v_datatypes.size(); i++) {
    TString varName = v_varNames[i];
    varName = varName.ReplaceAll("_", "") + "_";
    TString varType = v_datatypes[i];
    if (varType.BeginsWith("std::vector")
        || varType.BeginsWith("ROOT::Math") ) {
      TString prefix = "";
      if (varType.BeginsWith("std::vector<float>") )
        prefix = "floats";
      if (varType.BeginsWith("std::vector<int>") )
        prefix = "ints";
      if (varType.BeginsWith("std::vector<unsigned int>") )
        prefix = "uints";
      if (varType.BeginsWith("std::vector<unsigned char>") )
        prefix = "uchars";
      if (varType.BeginsWith("ROOT::Math::LorentzVector<") )
        prefix = "floatROOTMathPxPyPzE4DROOTMathLorentzVector";
      if (varType.BeginsWith("std::vector<ROOT::Math::LorentzVector<") )
        prefix = "floatROOTMathPxPyPzE4DROOTMathLorentzVectors";
      if (varType.Contains("std::vector<std::vector<ROOT::Math::LorentzVector<") )
        prefix = "floatROOTMathPxPyPzE4DROOTMathLorentzVectorss";
      branchfile << "   outTree_->Branch(\"" << prefix + "_" +varName << "\",   \""
                 << varType << "\",   &" << varName << ");" << endl;
      branchfile << "   outTree_->SetAlias(\"" << v_varNames[i] << "\",   "
                 << "\"" << prefix+"_"+varName << "\");" << endl;
      continue;
    }
    if (varType=="float" || varType == "Float_t") {
      branchfile << "   outTree_->Branch(\"" << varName << "\",   &" << varName;
      branchfile << ",   \"" << varName + "/F\");" << endl;
      branchfile << "   outTree_->SetAlias(\"" << v_varNames[i] << "\",   "
                 << "\"" << varName << "\");" << endl;
      continue;
    }
    if (varType=="unsigned int" || varType == "UInt_t") {
      branchfile << "   outTree_->Branch(\"" << varName << "\",   &" << varName;
      branchfile << ",   \"" << varName + "/i\");" << endl;
      branchfile << "   outTree_->SetAlias(\"" << v_varNames[i] << "\",   "
                 << "\"" << varName << "\");" << endl;
      continue;
    }
    if (varType=="unsigned long long" || varType == "ULong64_t" || varType == "const unsigned long long") {
      branchfile << "   outTree_->Branch(\"" << varName << "\",   &" << varName;
      branchfile << ",   \"" << varName + "/l\");" << endl;
      branchfile << "   outTree_->SetAlias(\"" << v_varNames[i] << "\",   "
                 << "\"" << varName << "\");" << endl;
      continue;
    }
    if (varType=="int" || varType == "Int_t") {
      branchfile << "   outTree_->Branch(\"" << varName << "\",   &" << varName;
      branchfile << ",   \"" << varName + "/I\");" << endl;
      branchfile << "   outTree_->SetAlias(\"" << v_varNames[i] << "\",   "
                 << "\"" << varName << "\");" << endl;
      continue;
    }
    if (varType=="double" || varType == "Double_t") {
      branchfile << "   outTree_->Branch(\"" << varName << "\",   &" << varName;
      branchfile << ",   \"" << varName + "/D\");" << endl;
      branchfile << "   outTree_->SetAlias(\"" << v_varNames[i] << "\",   "
                 << "\"" << varName << "\");" << endl;
      continue;
    }
  }
  branchfile << "} " <<  endl;
  branchfile << "#endif" << endl;

  branchfile.close();
}


void makeDriverFile(std::string fname, std::string treeName) {

  ofstream driverF;
  driverF.open("doAll.C");

  driverF << "{" << endl << endl;
  driverF << "  gROOT->ProcessLine(\".L ScanChain.C+\");" << endl << endl;
  driverF << "  TChain *ch = new TChain(\"" << treeName << "\"); " << endl;
  driverF << "  ch->Add(\"" + fname + "\");" << endl;
  driverF << "  ScanChain(ch); " << endl;
  driverF << "}";
  driverF.close();
}
