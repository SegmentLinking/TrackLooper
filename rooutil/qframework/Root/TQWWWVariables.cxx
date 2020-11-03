#include "QFramework/TQWWWVariables.h"
#include <limits>

// uncomment the following line to enable debug printouts
// #define _DEBUG_
// you can perform debug printouts with statements like this
// DEBUG("error number %d occurred",someInteger);

// be careful to not move the _DEBUG_ flag behind the following line
// otherwise, it will show no effect
#include "QFramework/TQLibrary.h"

ClassImp(TQWWWVariables)

//______________________________________________________________________________________________

TQWWWVariables::TQWWWVariables()
{
    // default constructor
    DEBUGclass("default constructor called");
    vartype = kNotSet;
}

//______________________________________________________________________________________________

TQWWWVariables::~TQWWWVariables()
{
    // default destructor
    DEBUGclass("destructor called");
}


//______________________________________________________________________________________________

TObjArray* TQWWWVariables::getBranchNames() const
{
    // retrieve the list of branch names
    // ownership of the list belongs to the caller of the function
    DEBUGclass("retrieving branch names");
    TObjArray* bnames = new TObjArray();
    bnames->SetOwner(false);
    // add the branch names needed by your observable here, e.g.
    // bnames->Add(new TObjString("someBranch"));
    bnames->Add(new TObjString("nVlep"));
    bnames->Add(new TObjString("nLlep"));
    bnames->Add(new TObjString("lep_pdgId"));
    bnames->Add(new TObjString("mc_HLT_DoubleEl"));
    bnames->Add(new TObjString("mc_HLT_DoubleEl_DZ"));
    bnames->Add(new TObjString("mc_HLT_MuEG"));
    bnames->Add(new TObjString("mc_HLT_DoubleMu"));
    bnames->Add(new TObjString("lep_p4.fCoordinates.fX"));
    bnames->Add(new TObjString("lep_p4.fCoordinates.fY"));
    bnames->Add(new TObjString("lep_p4.fCoordinates.fZ"));
    bnames->Add(new TObjString("lep_p4.fCoordinates.fT"));
    bnames->Add(new TObjString("jets_p4"));
    bnames->Add(new TObjString("jets_p4.fCoordinates.fX"));
    bnames->Add(new TObjString("jets_p4.fCoordinates.fY"));
    bnames->Add(new TObjString("jets_p4.fCoordinates.fZ"));
    bnames->Add(new TObjString("jets_p4.fCoordinates.fT"));
    bnames->Add(new TObjString("jets_up_p4"));
    bnames->Add(new TObjString("jets_up_p4.fCoordinates.fX"));
    bnames->Add(new TObjString("jets_up_p4.fCoordinates.fY"));
    bnames->Add(new TObjString("jets_up_p4.fCoordinates.fZ"));
    bnames->Add(new TObjString("jets_up_p4.fCoordinates.fT"));
    bnames->Add(new TObjString("jets_dn_p4"));
    bnames->Add(new TObjString("jets_dn_p4.fCoordinates.fX"));
    bnames->Add(new TObjString("jets_dn_p4.fCoordinates.fY"));
    bnames->Add(new TObjString("jets_dn_p4.fCoordinates.fZ"));
    bnames->Add(new TObjString("jets_dn_p4.fCoordinates.fT"));
    bnames->Add(new TObjString("met_pt"));
    bnames->Add(new TObjString("met_phi"));
    bnames->Add(new TObjString("met_up_pt"));
    bnames->Add(new TObjString("met_up_phi"));
    bnames->Add(new TObjString("met_dn_pt"));
    bnames->Add(new TObjString("met_dn_phi"));
    return bnames;
}

//______________________________________________________________________________________________

float TQWWWVariables::mT(LorentzVector p4, float met_pt, float met_phi) const
{
    float phi1 = p4.Phi();
    float phi2 = met_phi;
    float Et1  = p4.Et();
    float Et2  = met_pt;
    return sqrt(2 * Et1 * Et2 * (1.0 - cos(phi1 - phi2)));
}

//______________________________________________________________________________________________

float TQWWWVariables::MTlvlv(int syst) const
{
    TString met_pt_bn  = syst == 0 ? "met_pt"  : syst == 1 ? "met_up_pt"  : "met_dn_pt";
    TString met_phi_bn = syst == 0 ? "met_phi" : syst == 1 ? "met_up_phi" : "met_dn_phi";

    std::vector<LorentzVector>* lep_p4     = ((std::vector<LorentzVector>*) ( (TLeaf*) (this -> fTree -> GetBranch("lep_p4")     -> GetListOfLeaves() -> At(0))) -> GetValuePointer());
    float met_pt                           = *((float*)                     (((TLeaf*) (this -> fTree -> GetBranch(met_pt_bn)    -> GetListOfLeaves() -> At(0))) -> GetValuePointer()));
    float met_phi                          = *((float*)                     (((TLeaf*) (this -> fTree -> GetBranch(met_phi_bn)   -> GetListOfLeaves() -> At(0))) -> GetValuePointer()));

    if (lep_p4->size() < 2)
        return -999;

    LorentzVector dilep = lep_p4->at(0) + lep_p4->at(1);

    LorentzVector met;
    met.SetPxPyPzE(met_pt * TMath::Cos(met_phi), met_pt * TMath::Sin(met_phi), 0, met_pt);

    return mT(dilep, met_pt, met_phi);
}

//______________________________________________________________________________________________

float TQWWWVariables::Trigger() const
{
    std::vector<int>* lep_pdgId = ((std::vector<int>*) ( (TLeaf*) (this -> fTree -> GetBranch("lep_pdgId") -> GetListOfLeaves() -> At(0))) -> GetValuePointer());
    int mc_HLT_DoubleEl = *((int*) (((TLeaf*) (this -> fTree -> GetBranch("mc_HLT_DoubleEl") -> GetListOfLeaves() -> At(0))) -> GetValuePointer()));
    int mc_HLT_DoubleEl_DZ = *((int*) (((TLeaf*) (this -> fTree -> GetBranch("mc_HLT_DoubleEl_DZ") -> GetListOfLeaves() -> At(0))) -> GetValuePointer()));
    int mc_HLT_MuEG = *((int*) (((TLeaf*) (this -> fTree -> GetBranch("mc_HLT_MuEG") -> GetListOfLeaves() -> At(0))) -> GetValuePointer()));
    int mc_HLT_DoubleMu = *((int*) (((TLeaf*) (this -> fTree -> GetBranch("mc_HLT_DoubleMu") -> GetListOfLeaves() -> At(0))) -> GetValuePointer()));
    int nVlep = *((int*) (((TLeaf*) (this -> fTree -> GetBranch("nVlep") -> GetListOfLeaves() -> At(0))) -> GetValuePointer()));
    int nLlep = *((int*) (((TLeaf*) (this -> fTree -> GetBranch("nLlep") -> GetListOfLeaves() -> At(0))) -> GetValuePointer()));

    if (nVlep != 2 && nVlep != 3)
        return 0;

    if (nLlep != 2 && nLlep != 3)
        return 0;

    if (lep_pdgId->size() < 2)
        return 0;

    if (nVlep == 2 && nLlep == 2)
    {
        int lepprod = lep_pdgId->at(0)*lep_pdgId->at(1);
        if (abs(lepprod) == 121)
            return (mc_HLT_DoubleEl || mc_HLT_DoubleEl_DZ);
        else if (abs(lepprod) == 143)
            return mc_HLT_MuEG;
        else if (abs(lepprod) == 169)
            return mc_HLT_DoubleMu;
        else
            return 0;
    }
    else if (nVlep == 3 && nLlep == 3)
    {
        int lepprod01 = lep_pdgId->at(0)*lep_pdgId->at(1);
        if (abs(lepprod01) == 121 && (mc_HLT_DoubleEl || mc_HLT_DoubleEl_DZ))
            return true;
        else if (abs(lepprod01) == 143 && mc_HLT_MuEG)
            return true;
        else if (abs(lepprod01) == 169 && mc_HLT_DoubleMu)
            return true;

        int lepprod02 = lep_pdgId->at(0)*lep_pdgId->at(2);
        if (abs(lepprod02) == 121 && (mc_HLT_DoubleEl || mc_HLT_DoubleEl_DZ))
            return true;
        else if (abs(lepprod02) == 143 && mc_HLT_MuEG)
            return true;
        else if (abs(lepprod02) == 169 && mc_HLT_DoubleMu)
            return true;

        int lepprod12 = lep_pdgId->at(1)*lep_pdgId->at(2);
        if (abs(lepprod12) == 121 && (mc_HLT_DoubleEl || mc_HLT_DoubleEl_DZ))
            return true;
        else if (abs(lepprod12) == 143 && mc_HLT_MuEG)
            return true;
        else if (abs(lepprod12) == 169 && mc_HLT_DoubleMu)
            return true;

        return false;
    }
    else
    {
        return 0;
    }
}

//______________________________________________________________________________________________

float TQWWWVariables::MTlvlvjj(int syst) const
{
    TString jets_p4_bn = syst == 0 ? "jets_p4" : syst == 1 ? "jets_up_p4" : "jets_dn_p4";
    TString met_pt_bn  = syst == 0 ? "met_pt"  : syst == 1 ? "met_up_pt"  : "met_dn_pt";
    TString met_phi_bn = syst == 0 ? "met_phi" : syst == 1 ? "met_up_phi" : "met_dn_phi";

    std::vector<LorentzVector>* lep_p4     = ((std::vector<LorentzVector>*) ( (TLeaf*) (this -> fTree -> GetBranch("lep_p4")     -> GetListOfLeaves() -> At(0))) -> GetValuePointer());
    std::vector<LorentzVector>* jets_p4    = ((std::vector<LorentzVector>*) ( (TLeaf*) (this -> fTree -> GetBranch(jets_p4_bn)   -> GetListOfLeaves() -> At(0))) -> GetValuePointer());
    float met_pt                           = *((float*)                     (((TLeaf*) (this -> fTree -> GetBranch(met_pt_bn)    -> GetListOfLeaves() -> At(0))) -> GetValuePointer()));
    float met_phi                          = *((float*)                     (((TLeaf*) (this -> fTree -> GetBranch(met_phi_bn)   -> GetListOfLeaves() -> At(0))) -> GetValuePointer()));

    if (lep_p4->size() < 2 || jets_p4->size() < 2)
        return -999;

    LorentzVector dijet;
    float tmpDR = 9999;
    for (unsigned int i = 0; i < jets_p4->size(); ++i)
    {
        const LorentzVector& p4 = jets_p4->at(i);

        if (!( p4.pt() > 30. ))
            continue;

        // Compute Mjj using the closest two jets
        for (unsigned int j = i + 1; j < jets_p4->size(); ++j)
        {
            const LorentzVector& p4_2 = jets_p4->at(j);

            if (!( p4_2.pt() > 30. ))
                continue;

            // central eta
            if (fabs(p4.eta()) < 2.5 && fabs(p4_2.eta()) < 2.5)
            {
                // Choose the closest two jets
                float this_dR = ROOT::Math::VectorUtil::DeltaR(p4, p4_2);
                if (this_dR < tmpDR)
                {
                    tmpDR = this_dR;
                    dijet = (p4 + p4_2);
                }
            }
        }
    }

    LorentzVector dilep = lep_p4->at(0) + lep_p4->at(1);
    dijet = jets_p4->at(0) + jets_p4->at(1);

    LorentzVector met;
    met.SetPxPyPzE(met_pt * TMath::Cos(met_phi), met_pt * TMath::Sin(met_phi), 0, met_pt);

    return (dijet + dilep + met).M();
    return mT(dilep + dijet, met_pt, met_phi);
}

//______________________________________________________________________________________________

double TQWWWVariables::getValue() const
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
    std::vector<LorentzVector>* lep_p4     = ((std::vector<LorentzVector>*) ( (TLeaf*) (this -> fTree -> GetBranch("lep_p4")     -> GetListOfLeaves() -> At(0))) -> GetValuePointer());
    std::vector<LorentzVector>* jets_p4    = ((std::vector<LorentzVector>*) ( (TLeaf*) (this -> fTree -> GetBranch("jets_p4")    -> GetListOfLeaves() -> At(0))) -> GetValuePointer());
    std::vector<LorentzVector>* jets_up_p4 = ((std::vector<LorentzVector>*) ( (TLeaf*) (this -> fTree -> GetBranch("jets_up_p4") -> GetListOfLeaves() -> At(0))) -> GetValuePointer());
    std::vector<LorentzVector>* jets_dn_p4 = ((std::vector<LorentzVector>*) ( (TLeaf*) (this -> fTree -> GetBranch("jets_dn_p4") -> GetListOfLeaves() -> At(0))) -> GetValuePointer());
    float met_pt                           = *((float*)                     (((TLeaf*) (this -> fTree -> GetBranch("met_pt")     -> GetListOfLeaves() -> At(0))) -> GetValuePointer()));
    float met_phi                          = *((float*)                     (((TLeaf*) (this -> fTree -> GetBranch("met_phi")    -> GetListOfLeaves() -> At(0))) -> GetValuePointer()));
    float met_up_pt                        = *((float*)                     (((TLeaf*) (this -> fTree -> GetBranch("met_up_pt")  -> GetListOfLeaves() -> At(0))) -> GetValuePointer()));
    float met_up_phi                       = *((float*)                     (((TLeaf*) (this -> fTree -> GetBranch("met_up_phi") -> GetListOfLeaves() -> At(0))) -> GetValuePointer()));
    float met_dn_pt                        = *((float*)                     (((TLeaf*) (this -> fTree -> GetBranch("met_dn_pt")  -> GetListOfLeaves() -> At(0))) -> GetValuePointer()));
    float met_dn_phi                       = *((float*)                     (((TLeaf*) (this -> fTree -> GetBranch("met_dn_phi") -> GetListOfLeaves() -> At(0))) -> GetValuePointer()));

//    std::cout <<  " met_pt: " << met_pt <<  " met_phi: " << met_phi <<  std::endl;
//    std::cout << lep_p4->size() << std::endl;


    double retval = 0;
    switch (vartype)
    {
        case kVarMTlvlvjj:
            retval = MTlvlvjj(0);
//            std::cout <<  " lep_p4->size(): " << lep_p4->size() <<  " dilep.Pt(): " << dilep.Pt() <<  " dilep.Phi(): " << dilep.Phi() <<  " met_pt: " << met_pt <<  " met_phi: " << met_phi <<  std::endl;
//            std::cout <<  " retval: " << retval <<  std::endl;
            break;
        case kVarMTlvlv:
            retval = MTlvlv(0);
//            std::cout <<  " lep_p4->size(): " << lep_p4->size() <<  " dilep.Pt(): " << dilep.Pt() <<  " dilep.Phi(): " << dilep.Phi() <<  " met_pt: " << met_pt <<  " met_phi: " << met_phi <<  std::endl;
//            std::cout <<  " retval: " << retval <<  std::endl;
            break;
        case kVarTrigger:
            retval = Trigger();
//            std::cout <<  " lep_p4->size(): " << lep_p4->size() <<  " dilep.Pt(): " << dilep.Pt() <<  " dilep.Phi(): " << dilep.Phi() <<  " met_pt: " << met_pt <<  " met_phi: " << met_phi <<  std::endl;
//            std::cout <<  " retval: " << retval <<  std::endl;
            break;
    }

    DEBUGclass("returning");
    return retval;
}
//______________________________________________________________________________________________

bool TQWWWVariables::initializeSelf()
{
    // initialize this observable on a sample/tree
    DEBUGclass("initializing");
    // since this function is only called once per sample, we can
    // perform any checks that seem necessary
    if (!this->fTree)
    {
        DEBUGclass("no tree, terminating");
        return false;
    }
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

    if (this->fExpression.EqualTo("MTlvlvjj"))
    {
        vartype = kVarMTlvlvjj;
    }
    else if (this->fExpression.EqualTo("MTlvlv"))
    {
        vartype = kVarMTlvlv;
    }
    else if (this->fExpression.EqualTo("Trigger"))
    {
        vartype = kVarTrigger;
    }
    else
    {
        vartype = kNotSet;
        DEBUGclass("Does not know what to set it to! '%s'", this->fExpression.Data());
    }


    return true;
}

//______________________________________________________________________________________________

bool TQWWWVariables::finalizeSelf()
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

TQWWWVariables::TQWWWVariables(const TString& expression):
    TQTreeObservable(expression)
{
    // constructor with expression argument
    DEBUGclass("constructor called with '%s'", expression.Data());
    // the predefined string member "expression" allows your observable to store an expression of your choice
    // this string will be the verbatim argument you passed to the constructor of your observable
    // you can use it to choose between different modes or pass configuration options to your observable
    this->SetName(TQObservable::makeObservableName(expression));
    this->setExpression(expression);
}

//______________________________________________________________________________________________

const TString& TQWWWVariables::getExpression() const
{
    // retrieve the expression associated with this observable
    return this->fExpression;
}

//______________________________________________________________________________________________

bool TQWWWVariables::hasExpression() const
{
    // check if this observable type knows expressions
    return true;
}

//______________________________________________________________________________________________

void TQWWWVariables::setExpression(const TString& expr)
{
    // set the expression to a given string
    this->fExpression = expr;
}
