#include "QFramework/TQWWWMTOneLep.h"
#include <limits>

// uncomment the following line to enable debug printouts
// #define _DEBUG_
// you can perform debug printouts with statements like this
// DEBUG("error number %d occurred",someInteger);

// be careful to not move the _DEBUG_ flag behind the following line
// otherwise, it will show no effect
#include "QFramework/TQLibrary.h"

#include <iostream>

ClassImp(TQWWWMTOneLep)

//______________________________________________________________________________________________
TQWWWMTOneLep::TQWWWMTOneLep()
{
    // default constructor
    DEBUGclass("default constructor called");
}

//______________________________________________________________________________________________
TQWWWMTOneLep::~TQWWWMTOneLep()
{
    // default destructor
    DEBUGclass("destructor called");
} 


//______________________________________________________________________________________________
TObjArray* TQWWWMTOneLep::getBranchNames() const
{
    // retrieve the list of branch names 
    // ownership of the list belongs to the caller of the function
    DEBUGclass("retrieving branch names");
    TObjArray* bnames = new TObjArray();
    bnames->SetOwner(false);

    // add the branch names needed by your observable here, e.g.
    // bnames->Add(new TObjString("someBranch"));

    return bnames;
}

//______________________________________________________________________________________________
double TQWWWMTOneLep::getValue() const 
{
    // in the rest of this function, you should retrieve the data and calculate your return value
    // here is the place where most of your custom code should go
    // a couple of comments should guide you through the process
    // when writing your code, please keep in mind that this code can be executed several times on every event
    // make your code efficient. catch all possible problems. when in doubt, contact experts!

    // here, you should calculate your return value
    // of course, you can use other data members of your observable at any time
    /* example block for TTreeFormula method:
       const double retval = this->fFormula->Eval(0.);
       */
    /* exmple block for TTree::SetBranchAddress method:
       const double retval = this->fBranch1 + this->fBranch2;
       */

    this->fFormula->GetNdata();
    const double retval = this->fFormula->EvalInstance();

    DEBUGclass("returning");
    return retval;
}

//______________________________________________________________________________________________
bool TQWWWMTOneLep::initializeSelf()
{
    // initialize this observable on a sample/tree
    DEBUGclass("initializing");

    // since this function is only called once per sample, we can
    // perform any checks that seem necessary
    if(!this->fTree){
        DEBUGclass("no tree, terminating");
        return false;
    }
    this->fFormula = new TTreeFormula("MTOneLep", "(TMath::Sqrt(2*met_pt*lep_pt[0]*(1.0-TMath::Cos(lep_phi[0]-met_phi))))",this->fTree);

    // if you want to use a TTreeFormula, can may construct it here
    /* example block for TTreeFormula method:
       this->fFormula = new TTreeFormula("branch1 + branch2",this->fTree);
       */

    // if you want to use the TTree::SetBranchAddress method, you can
    // call TTree::SetBranchAddress here
    // please note that this method is highly discouraged.
    // if a branch you access via this method is used by any other
    // observable, you will 'steal' the branch address from that
    // observable, leading to the other observable returning wrong
    // results
    /* example block for TTree::SetBranchAddress method:
       this->fTree->SetBranchAddress("branch1",&(this->fBranch1));
       this->fTree->SetBranchAddress("branch2",&(this->fBranch2));
       */

    return true;
}

//______________________________________________________________________________________________
bool TQWWWMTOneLep::finalizeSelf()
{
    // finalize this observable on a sample/tree
    DEBUGclass("finalizing");
    // here, you should undo anything that you did while initializing
    /* example block for TTreeFormula method:
       delete this->fFormula;
       this->fFormula = NULL;
       */

    return true;
}

//______________________________________________________________________________________________
TQWWWMTOneLep::TQWWWMTOneLep(const TString& name):
    TQTreeObservable(name)
{
    // constructor with name argument
    DEBUGclass("constructor called with '%s'",name.Data());
}
