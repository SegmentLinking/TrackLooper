//  .
// ..: P. Chang, philip@physics.ucsd.edu

#include "calc.h"

using namespace RooUtil;

//_________________________________________________________________________________________________
TLorentzVector RooUtil::Calc::getTLV(const LV& a)
{
    TLorentzVector r;
    r.SetPtEtaPhiM(a.pt(), a.eta(), a.phi(), a.mass());
    return r;
}

//_________________________________________________________________________________________________
LV RooUtil::Calc::getLV(const TLorentzVector& a)
{
    LV r;
    r.SetPxPyPzE(a.Px(), a.Py(), a.Pz(), a.E());
    return r;
}

//_________________________________________________________________________________________________
LV RooUtil::Calc::getLV(float pt, float eta, float phi, float m)
{
    TLorentzVector tmp;
    tmp.SetPtEtaPhiM(pt, eta, phi, m);
    return getLV(tmp);
}

//_________________________________________________________________________________________________
TVector3 RooUtil::Calc::boostVector(const LV& a)
{
    return getTLV(a).BoostVector();
}

//_________________________________________________________________________________________________
LV RooUtil::Calc::getBoosted(const LV& a, const TVector3& b)
{
    TLorentzVector tlv_r = getTLV(a);
    tlv_r.Boost(b);
    return getLV(tlv_r);
}

//_________________________________________________________________________________________________
void RooUtil::Calc::boost(LV& a, const TVector3& b)
{
    TLorentzVector tlv_r = getTLV(a);
    tlv_r.Boost(b);
    a = getLV(tlv_r);
}

//_________________________________________________________________________________________________
float RooUtil::Calc::DeltaR(const LV& a, const LV& b)
{
    return ROOT::Math::VectorUtil::DeltaR(a, b);
}

//_________________________________________________________________________________________________
float RooUtil::Calc::alpha(const LV& p1, const LV& p2)
{
    double phi = p2.phi() - p1.phi();
    if (abs(phi) > TMath::Pi())
        phi = p2.phi() + p1.phi();
    double eta = p2.eta() - p1.eta();
    return TMath::ATan2(eta, phi);
}

//_________________________________________________________________________________________________
float RooUtil::Calc::pPRel(const LV& pCand, const LV& pLep)
{
    if (pLep.pt()<=0.) return 0.;
    double dot = pCand.Vect().Dot( pLep.Vect() );
    return sqrt((dot*dot)/pLep.P2());
}

//_________________________________________________________________________________________________
float RooUtil::Calc::DeltaEta(const LV& a, const LV& b)
{
    return a.eta() - b.eta();
}

//_________________________________________________________________________________________________
float RooUtil::Calc::DeltaPhi(const LV& a, const LV& b)
{
    return ROOT::Math::VectorUtil::DeltaPhi(a, b);
}

//_________________________________________________________________________________________________
float RooUtil::Calc::mT(const LV& lep, const LV& met)
{
    return sqrt(2 * met.pt() * lep.Et() * (1.0 - cos(lep.phi() - met.phi())));
}

//*************************************************************************************************
//
// Math for solving neutrino momentum based on parent particle mass (e.g. W -> lv)
//
// def pv = (sqrt(vpt^2 + vz^2), vpt, vz), pl = (|pl|, lpt, lz)
// def vpt = (vx, vy), lpt = (lx, ly)
//
// mw^2 = 2 pv pl (assuming lepton is massless)
// mw^2 = 2 |pl| sqrt(vpt^2 + vz^2) - vpt . lpt - vz lz
//
// def k = mw^2 / 2 + vpt . lpt
//
// re-arrange and square both sides...
//
// (k + vz lz)^2 = |pl|^2 (vpt^2 + vz^2)
//
// Expand both sides and cancel the term lz^2 vz^2 that shows up in both sides
// after expanding |pl|^2 = lpt^2 + lz^2
// 
// k^2 + 2k vz lz = |pl|^2 vpt^2 + lpt^2 vz^2
//
// Re-arrange into quadratic form
//
// lpt^2 vz^2 - 2k lz vz + |pl|^2 vpt^2 -k^2 = 0
// ^^^^^      ^^^^^^^      ^^^^^^^^^^^^^^^^^
//   a           b                 c
//
// define "det" = b^2 - 4ac ("determinant" to check whether the solution is imaginary or real)
//
// --------------------------
//
// More general version where the mass of the "lepton" system is not massless
//
// (H->WW->lvjj)
//
// def k = (mw^2 - ml^2)  / 2 + vpt . lpt
//
// Re-arrange into quadratic form
//
// (lpt^2 + ml^2) vz^2 - 2k lz vz + El^2 vpt^2 -k^2 = 0
// ^^^^^^^^^^^^^^      ^^^^^^^      ^^^^^^^^^^^^^^^
//       a                b                c
//
// If ml = 0 it's the same as above
//
//*************************************************************************************************

//_________________________________________________________________________________________________
LV RooUtil::Calc::getNeutrinoP4(const LV& lep, const float& met_pt, const float& met_phi, float mw, bool getsol2, bool invertpz, bool debug)
{
    float pz = getNeutrinoPz(lep, met_pt, met_phi, mw, getsol2, debug);
    if (invertpz)
        pz = -pz;
    LV neutrino;
    using namespace TMath;
    float E = Sqrt(Power(met_pt*Cos(met_phi),2) + Power(met_pt*Sin(met_phi),2) + pz*pz);
    neutrino.SetPxPyPzE(met_pt*TMath::Cos(met_phi), met_pt*TMath::Sin(met_phi), pz, E);
    LV W = lep + neutrino;
    if (debug)
        std::cout <<  " W.mass(): " << W.mass() <<  " mw: " << mw <<  std::endl;
    return neutrino;
}

//_________________________________________________________________________________________________
float RooUtil::Calc::getNeutrinoPzDet(const LV& lep, const float& met_pt, const float& met_phi, float mw, bool debug)
{
    // Not ideal to call twice but... OK
    float lx = lep.px();
    float ly = lep.py();
    float lz = lep.pz();
    float ml = lep.mass();
    float lpt = lep.pt(); // |lpt|
    float pl2 = (lx * lx + ly * ly + lz * lz) + ml * ml; // El^2
    TLorentzVector met;
    met.SetPtEtaPhiM(met_pt, 0, met_phi, 0);
    float vx = met.Px();
    float vy = met.Py();
    float vpt = met_pt;

    if (debug)
        std::cout <<  " mw: " << mw <<  " ml: " << ml <<  std::endl;

    if (debug)
        std::cout <<  " lx: " << lx <<  " ly: " << ly <<  " lz: " << lz <<  " lpt: " << lpt <<  " pl2: " << pl2 <<  " vx: " << vx <<  " vy: " << vy <<  " vpt: " << vpt <<  std::endl;

    float k = (mw * mw - ml * ml) / 2. + vx * lx + vy * ly;
    float b = -2. * k * lz;
    float b2 = b * b;
    float ac = (lpt * lpt + ml * ml) * (pl2 * vpt * vpt - k * k);
    //         ^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^
    //                   a                          c
    float det = b2 - 4 * ac;

    if (debug)
        std::cout <<  " k: " << k <<  " b: " << b <<  " b2: " << b2 <<  " ac: " << ac <<  " det: " << det <<  std::endl;

    return det;
}

//_________________________________________________________________________________________________
float RooUtil::Calc::getNeutrinoPz(const LV& lep, const float& met_pt, const float& met_phi, float mw, bool getsol2, bool debug)
{
    // Not ideal to call twice but... OK
    float lx = lep.px();
    float ly = lep.py();
    float lz = lep.pz();
    float ml = lep.mass();
    float lpt = lep.pt(); // |lpt|
    // float pl2 = (lx * lx + ly * ly + lz * lz); // |pl|^2
    TLorentzVector met;
    met.SetPtEtaPhiM(met_pt, 0, met_phi, 0);
    float vx = met.Px();
    float vy = met.Py();
    // float vpt = met_pt;

    using namespace TMath;
    float det = getNeutrinoPzDet(lep, met_pt, met_phi, mw, debug);

    // If imaginary then ignore imaginary component and take the real value only
    if (det < 0)
        det = 0;

    float k = (mw * mw - ml * ml) / 2. + vx * lx + vy * ly;
    float a = (lpt * lpt + ml * ml);
    float b = -2. * k * lz;

    float sol1 = (-b -Sqrt(det))/(2. * a);
    float sol2 = (-b +Sqrt(det))/(2. * a);

    if (debug)
        std::cout <<  " sol1: " << sol1 <<  " sol2: " << sol2 <<  std::endl;

    float ans  = fabs(sol1) < fabs(sol2) ? sol1 : sol2;
    float ans2 = fabs(sol1) < fabs(sol2) ? sol2 : sol1;

    if (getsol2)
        return ans2;
    else
        return ans;
}

/*
//_________________________________________________________________________________________________
//
//            axis_ref
//                /
//               /  ref_vec.Phi()
//              /
//            ref=============>  +x
//              \
//               \
//                \
//                 \
//               target
//
//  ================================
//
//         axis_ref
//             |
//             |
//             |
//            ref
//              \__
//                 \__
//                    target
//
//
// The function rotates the axis_ref to be directly above ref and return target's TVector2
//
//
*/
TVector2 RooUtil::Calc::getEtaPhiVecRotated(const LV& target, const LV& ref, const LV& axis_ref)
{
    float deta = RooUtil::Calc::DeltaEta(axis_ref, ref);
    float dphi = RooUtil::Calc::DeltaPhi(axis_ref, ref);
    TVector2 ref_vec(deta, dphi);

    float target_deta = RooUtil::Calc::DeltaEta(target, ref);
    float target_dphi = RooUtil::Calc::DeltaEta(target, ref);
    TVector2 target_vec(target_deta, target_dphi);

    // The rotation can be thought of as "rotate it to align with +x axis, then add 90 degrees"
    return target_vec.Rotate(-ref_vec.Phi() + TMath::Pi() / 2.);
}

float RooUtil::Calc::getRho(const LV& ref, const LV& target)
{
/*
            phi
             ^
             | target
             |   /
             |  /
             | / "rho"
            ref-------------> eta

*/

    float dy = DeltaPhi(ref, target);
    float dx = DeltaEta(ref, target);
    return TMath::ATan(dy / dx);
}

void RooUtil::Calc::printTLV(const TLorentzVector& a)
{
    std::cout <<  " a.Pt(): " << a.Pt() <<  " a.Eta(): " << a.Eta() <<  " a.Phi(): " << a.Phi() <<  " a.M(): " << a.M() <<  " a.E(): " << a.E() <<  std::endl;
}

//_________________________________________________________________________________________________
void RooUtil::Calc::printLV(const LV& a)
{
    std::cout <<  " a.pt(): " << a.pt() <<  " a.eta(): " << a.eta() <<  " a.phi(): " << a.phi() <<  " a.mass(): " << a.mass() <<  " a.energy(): " << a.energy() <<  std::endl;
}

//_________________________________________________________________________________________________
// Two bounds are provided, and a point. computes the bin number
int RooUtil::Calc::calcBin2D(const std::vector<float>& xbounds, const std::vector<float>& ybounds, float xval, float yval)
{

//  -1
//
//   2  2.4
//             7    8    9   10   11   12
//   1  1.6
//             1    2    3    4    5    6
//   0  0.0
//           0   20   25   30   35   50   150
//           0    1    2    3    4    5    6    -1

//    1   2   3   4
//    5   6   7   8

    int cx = -1;
    int cy = -1;

    for (unsigned ix = 0; ix < xbounds.size(); ++ix)
    {
        // If the bound value is smaller than the xval I need to continue until the xbound is bigger than the given xval
        if (xbounds[ix] < xval)
            continue;

        // Set the cx (chosen x index) to ix
        cx = ix;
        break;
    }

    // If it reached the highest bound set it to the last one
    if (cx == -1) cx = xbounds.size() - 1;

    for (unsigned iy = 0; iy < ybounds.size(); ++iy)
    {
        // If the bound value is smaller than the xval I need to continue until the xbound is bigger than the given xval
        if (ybounds[iy] < yval)
            continue;

        // Set the cy (chosen y index) to iy
        cy = iy;
        break;
    }

    // If it reached the highest bound set it to the last one
    if (cy == -1) cy = ybounds.size() - 1;

    // If neither hit the value i want it failed to find one
    if (cx == 0 or cy == 0)
        return -1;

    return (cx - 1) + (cy - 1) * (xbounds.size()-1);
}

//_________________________________________________________________________________________________
std::tuple<bool, int, int, float> RooUtil::Calc::pickZcandidateIdxs(
        const std::vector<int>& lepton_pdgids,
        const std::vector<LV>& lepton_p4s,
        const std::vector<int> to_skip)
{
    // Loop over all pairs and if it is SFOS check whether it forms a Z if it does not then skip
    // Ultimately select the two idxs with invariant mass closest to the Z pole
    bool has_sfos = false;
    int z_idx_1 = -999;
    int z_idx_2 = -999;
    float closest_mll = 999;
    for (unsigned int idx = 0; idx < lepton_p4s.size(); ++idx)
    {

        // If an index is part of to_skip then skip
        if (std::find(to_skip.begin(), to_skip.end(), idx) != to_skip.end())
            continue;

        // First lepton 4 vector
        const LV& i_p4 = lepton_p4s[idx];

        // First lepton pdgid
        int i_pdgid = lepton_pdgids[idx];

        // Nested loop
        for (unsigned int jdx = idx + 1; jdx < lepton_p4s.size(); ++jdx)
        {

            // If an index is part of to_skip then skip
            if (std::find(to_skip.begin(), to_skip.end(), jdx) != to_skip.end())
                continue;

            // Second lepton 4 vector
            const LV& j_p4 = lepton_p4s[jdx];

            // Second lepton pdgid
            int j_pdgid = lepton_pdgids[jdx];

            // If the pair is not SFOS then skip
            if (i_pdgid != -j_pdgid)
                continue;

            // invariant mass
            float mll = (i_p4 + j_p4).mass();

            // invariant mass "distance" from MZ
            if (not has_sfos || // if it is the first pair then fill it regardless
                abs(closest_mll - 91.1876) > abs(mll - 91.1876))
            {
                closest_mll = mll;
                z_idx_1 = idx;
                z_idx_2 = jdx;
            }

            // Now set the has_sfos as we found at least one
            has_sfos = true;
        }
    }
    return std::make_tuple(has_sfos, z_idx_1, z_idx_2, closest_mll);
}

//eof
