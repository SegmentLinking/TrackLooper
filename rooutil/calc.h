//  .
// ..: P. Chang, philip@physics.ucsd.edu

#ifndef calc_h
#define calc_h

// C/C++

// ROOT
#include "TLorentzVector.h"
#include "TVector3.h"
#include "TVector2.h"
#include "TMath.h"
#include "Math/LorentzVector.h"
#include "Math/GenVector/PtEtaPhiM4D.h"
#include "Math/VectorUtil.h"

///////////////////////////////////////////////////////////////////////////////////////////////
// LorentzVector typedef that we use very often
///////////////////////////////////////////////////////////////////////////////////////////////
#ifdef LorentzVectorPtEtaPhiM4D
typedef ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<float> > LV;
#else
typedef ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<float> > LV;
#endif

namespace RooUtil
{
    namespace Calc {
        TLorentzVector getTLV(const LV& a);
        LV getLV(const TLorentzVector& a);
        LV getLV(float pt, float eta, float phi, float m);
        TVector3 boostVector(const LV& a);
        LV getBoosted(const LV& a, const TVector3& b);
        void boost(LV& a, const TVector3& b);
        float DeltaR(const LV& a, const LV& b);
        float alpha(const LV& a, const LV& b);
        float pPRel(const LV& a, const LV& b);
        float DeltaEta(const LV& a, const LV& b);
        float DeltaPhi(const LV& a, const LV& b);
        float mT(const LV& lep, const LV& met);
        LV getNeutrinoP4(const LV& lep, const float& met_pt, const float& met_phi, float mw=80.385, bool getsol2=false, bool invertpz=false, bool debug=false);
        float getNeutrinoPz(const LV& lep, const float& met_pt, const float& met_phi, float mw=80.385, bool getsol2=false, bool debug=false);
        float getNeutrinoPzDet(const LV& lep, const float& met_pt, const float& met_phi, float mw=80.385, bool debug=false);
        TVector2 getEtaPhiVecRotated(const LV& target, const LV& ref, const LV& axis_ref);
        float getRho(const LV& ref, const LV& target);
        void printTLV(const TLorentzVector& a);
        void printLV(const LV& a);
        int calcBin2D(const std::vector<float>& xbounds, const std::vector<float>& ybounds, float xval, float yval);
        std::tuple<bool, int, int, float> pickZcandidateIdxs(
                const std::vector<int>& lepton_pdgids,
                const std::vector<LV>& lepton_p4s,
                std::vector<int> to_skip=std::vector<int>());
    }
}

#endif
