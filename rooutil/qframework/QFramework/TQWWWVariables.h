//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQWWWVARIABLES__
#define __TQWWWVARIABLES__
#include "QFramework/TQTreeObservable.h"

#include "TLeaf.h"
#include "TMath.h"

#include <vector>
#include <iostream>

#include "Math/LorentzVector.h"
#include "Math/GenVector/LorentzVector.h"
#include "Math/VectorUtil.h"
typedef ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<float> > LorentzVector;

class TQWWWVariables : public TQTreeObservable
{
protected:
    // put here any data members your class might need
    enum Vars
    {
        kNotSet,
        kVarMTlvlvjj,
        kVarMTlvlv,
        kVarTrigger,
    };

    Vars vartype;

public:
    virtual double getValue() const override;
    virtual TObjArray* getBranchNames() const override;
protected:
    virtual bool initializeSelf() override;
    virtual bool finalizeSelf() override;
protected:
    TString fExpression = "";

public:
    virtual bool hasExpression() const override;
    virtual const TString& getExpression() const override;
    virtual void setExpression(const TString& expr) override;

    float mT(LorentzVector p4, float met_pt, float met_phi) const;
    float MTlvlvjj(int syst) const;
    float MTlvlv(int syst) const;
    float Trigger() const;

    TQWWWVariables();
    TQWWWVariables(const TString& expression);
    virtual ~TQWWWVariables();
    ClassDefOverride(TQWWWVariables, 1);


};
#endif
